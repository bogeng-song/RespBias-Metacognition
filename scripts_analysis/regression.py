"""
regression.py
Fits Mixed-Effects Linear Models progressively, adding one regressor at a time.
Implements a robust fallback mechanism for convergence issues (Singular Matrices).
"""
import pandas as pd
import numpy as np
import warnings
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

def get_mixed_model_coefficients_random(df, dependent_var, regressors, target_regressor=None, group_col='subject_idx', max_iter=2000, method='powell', print_summary=True):
    if target_regressor is None:
        target_regressor = regressors[0]
        
    coeffs_list = []
    model_results = {}
    
    def is_valid_fit(res):
        if res is None: return False
        if hasattr(res, 'bse') and pd.isna(res.bse).any(): return False
        return True

    for i in range(len(regressors)):
        current_regressors = regressors[:(i+1)]
        formula = f"{dependent_var} ~ " + " + ".join(current_regressors)
        print(f"\n{'='*60}\nFitting Model {i+1}: {formula}")
        
        clean_cols = [dependent_var, group_col] + current_regressors
        df_clean = df[clean_cols].dropna()
        subjects = df_clean[group_col].unique()
        
        result = None
        is_ols = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # ATTEMPT 1: Random Slope for the target regressor
                if target_regressor in current_regressors:
                    print(f"  Attempt 1: Random Slope for '{target_regressor}'...")
                    re_formula = "~ " + target_regressor
                else:
                    re_formula = None 
                
                model = smf.mixedlm(formula, data=df_clean, groups=df_clean[group_col], re_formula=re_formula)
                res_temp = model.fit(method=method, maxiter=max_iter)
                
                if not is_valid_fit(res_temp):
                    res_temp = model.fit(method="lbfgs", maxiter=max_iter)
                
                if is_valid_fit(res_temp):
                    print("  [✓] Attempt 1 Converged.")
                    result = res_temp
                else:
                    # ATTEMPT 2: Random Intercepts Only
                    print("  [!] Attempt 1 Singular. Trying Attempt 2: Random Intercepts...")
                    model_int = smf.mixedlm(formula, data=df_clean, groups=df_clean[group_col])
                    res_temp = model_int.fit(method="lbfgs", maxiter=max_iter)
                    
                    if is_valid_fit(res_temp):
                        print("  [✓] Attempt 2 Converged.")
                        result = res_temp
                    else:
                        # ATTEMPT 3: OLS with Cluster-Robust SEs
                        print("  [!] Attempt 2 Singular. Trying Attempt 3: OLS with Clustered SEs...")
                        model_ols = smf.ols(formula, data=df_clean)
                        result = model_ols.fit(cov_type='cluster', cov_kwds={'groups': df_clean[group_col]})
                        is_ols = True
                        print("  [✓] Attempt 3 Converged.")
                        
            except Exception as e:
                print(f"  [x] MixedLM crashed. Falling back to Attempt 3 (Cluster OLS)...")
                model_ols = smf.ols(formula, data=df_clean)
                result = model_ols.fit(cov_type='cluster', cov_kwds={'groups': df_clean[group_col]})
                is_ols = True
                print("  [✓] Attempt 3 Converged.")

        model_results[i] = result
        if print_summary and result is not None:
            print(result.summary())
        
        # Extract individual slopes for plotting
        fixed = result.params if is_ols else result.fe_params
        individual_slopes = {subj: {} for subj in subjects}
        
        if is_ols:
            for subj in subjects:
                for reg in current_regressors:
                    individual_slopes[subj][reg] = fixed.get(reg, 0)
        else:
            try:
                random = result.random_effects
                for subj in subjects:
                    re = random.get(subj, {})
                    for reg in current_regressors:
                        re_val = re[reg] if hasattr(re, 'index') and reg in re.index else re.get(reg, 0)
                        individual_slopes[subj][reg] = fixed.get(reg, 0) + re_val
            except Exception as e:
                for subj in subjects:
                    for reg in current_regressors:
                        individual_slopes[subj][reg] = fixed.get(reg, 0)
        
        if target_regressor in current_regressors:
            coeffs = np.array([individual_slopes[subj].get(target_regressor, np.nan) for subj in subjects])
        else:
            coeffs = np.array([])

        coeffs_list.append(coeffs)

    return coeffs_list, model_results


# ══════════════════════════════════════════════════════════════════════════════
# JOINT (MODERATED) MIXED-EFFECTS REGRESSION — Experiment 2 (speed vs. accuracy)
#
# Experiment 2 is now analysed with the two speed-accuracy-tradeoff conditions
# fit together in a single model, with condition as an effect-coded moderator.
# This replaces the earlier approach of fitting each condition separately and
# lets us test the Bias x Condition interaction directly.
#
# Condition is effect-coded (Accuracy focus = +0.5, Speed focus = -0.5). The
# focal quantities are:
#   * the Bias x Condition interaction on confidence (does the residual bias
#     effect differ between conditions?), and
#   * the per-condition simple slopes of bias on confidence.
# ══════════════════════════════════════════════════════════════════════════════

# The three nested joint models (bias term is a placeholder filled per call):
#   1. confidence_z ~ {bias} * Cond_eff
#   2. confidence_z ~ {bias} * Cond_eff + accuracy_z * Cond_eff
#   3. confidence_z ~ {bias} * Cond_eff + accuracy_z * Cond_eff + rt_z * Cond_eff
MODERATED_FORMULAS = [
    "{y} ~ {b} * Cond_eff",
    "{y} ~ {b} * Cond_eff + accuracy_z * Cond_eff",
    "{y} ~ {b} * Cond_eff + accuracy_z * Cond_eff + rt_z * Cond_eff",
]


def add_effect_coded_condition(df, condition_col='Condition',
                               accuracy_label='Accuracy', out_col='Cond_eff'):
    """Add an effect-coded moderator column: Accuracy focus = +0.5, Speed = -0.5."""
    df = df.copy()
    df[out_col] = np.where(df[condition_col] == accuracy_label, 0.5, -0.5)
    return df


def fit_interaction_mixedlm(df, formula, bias_var='FAR_z', group_col='subject_idx',
                            max_iter=2000):
    """Fit one joint interaction model with a random intercept and a random slope
    for the bias term, falling back to a random-intercept-only structure if the
    random-slope model is singular.

    Returns (result, re_structure_label).
    """
    df_clean = df.dropna(subset=[c for c in df.columns if c in formula or c == group_col])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(formula, df_clean, groups=df_clean[group_col],
                                re_formula=f"~ {bias_var}")
            res = model.fit(method="lbfgs", maxiter=max_iter)
            if res is not None and not pd.isna(res.bse).any():
                return res, f"random intercept + random slope ({bias_var})"
        except Exception:
            pass
        # Fallback: random intercept only.
        model = smf.mixedlm(formula, df_clean, groups=df_clean[group_col])
        res = model.fit(method="lbfgs", maxiter=max_iter)
        return res, "random intercept only"


def fixed_effects_table(result):
    """Tidy fixed-effects table (Coef, SE, z, p, 95% CI) from a mixedlm result."""
    ci = result.conf_int()
    rows = []
    for name in result.fe_params.index:
        rows.append({
            'term': name,
            'coef': result.fe_params[name],
            'se': result.bse[name],
            'z': result.tvalues[name],
            'p': result.pvalues[name],
            'ci_low': ci.loc[name, 0],
            'ci_high': ci.loc[name, 1],
        })
    return pd.DataFrame(rows)


def simple_slope(result, main_term, interaction_term, w):
    """Condition-specific simple slope of the bias term.

    slope(w) = beta(main_term) + w * beta(interaction_term), with SE from the
    fixed-effects covariance matrix. Use w = +0.5 for Accuracy focus and
    w = -0.5 for Speed focus.

    Returns (estimate, se, z, p).
    """
    from scipy.stats import norm

    beta = result.fe_params
    cov = result.cov_params()
    est = beta[main_term] + w * beta[interaction_term]
    var = (cov.loc[main_term, main_term]
           + (w ** 2) * cov.loc[interaction_term, interaction_term]
           + 2 * w * cov.loc[main_term, interaction_term])
    se = np.sqrt(var)
    z = est / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(est), float(se), float(z), float(p)


def get_moderated_regression(df, bias_var='FAR_z', outcome_var='confidence_z',
                             group_col='subject_idx', condition_col='Condition',
                             accuracy_label='Accuracy'):
    """Fit the three nested joint (moderated) models for Experiment 2 and extract
    the Bias x Condition interaction plus per-condition simple slopes.

    ``df`` must contain the outcome, the bias term (e.g. ``FAR_z`` or ``bias_z``),
    ``accuracy_z``, ``rt_z``, a subject grouping column, and a ``Condition`` column
    with two levels. Returns a dict:
        {
          'models':       [result_1, result_2, result_3],
          're_structures':[...],
          'fixed_tables': [DataFrame, ...],
          'interaction':  DataFrame of the Bias:Cond_eff term across models,
          'simple_slopes':DataFrame of accuracy-focus / speed-focus slopes per model,
        }
    """
    df = add_effect_coded_condition(df, condition_col=condition_col,
                                    accuracy_label=accuracy_label)
    interaction_term = f"{bias_var}:Cond_eff"

    models, re_structs, fixed_tables = [], [], []
    inter_rows, slope_rows = [], []

    for i, template in enumerate(MODERATED_FORMULAS, start=1):
        formula = template.format(y=outcome_var, b=bias_var)
        print(f"\n{'='*60}\nJoint model {i}: {formula}")
        res, re_label = fit_interaction_mixedlm(df, formula, bias_var=bias_var,
                                                group_col=group_col)
        models.append(res)
        re_structs.append(re_label)
        ft = fixed_effects_table(res)
        fixed_tables.append(ft)
        print(f"  [random effects: {re_label}]")
        print(ft.to_string(index=False))

        if interaction_term in res.fe_params.index:
            inter_rows.append({
                'model': i,
                'coef': res.fe_params[interaction_term],
                'se': res.bse[interaction_term],
                'z': res.tvalues[interaction_term],
                'p': res.pvalues[interaction_term],
            })
            s_acc = simple_slope(res, bias_var, interaction_term, w=0.5)
            s_spd = simple_slope(res, bias_var, interaction_term, w=-0.5)
            slope_rows.append({
                'model': i,
                'accuracy_focus_beta': s_acc[0], 'accuracy_focus_se': s_acc[1],
                'accuracy_focus_z': s_acc[2], 'accuracy_focus_p': s_acc[3],
                'speed_focus_beta': s_spd[0], 'speed_focus_se': s_spd[1],
                'speed_focus_z': s_spd[2], 'speed_focus_p': s_spd[3],
            })

    return {
        'models': models,
        're_structures': re_structs,
        'fixed_tables': fixed_tables,
        'interaction': pd.DataFrame(inter_rows),
        'simple_slopes': pd.DataFrame(slope_rows),
    }