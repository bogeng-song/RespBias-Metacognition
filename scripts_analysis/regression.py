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