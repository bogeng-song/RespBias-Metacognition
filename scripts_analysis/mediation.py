"""
mediation.py
Bayesian multilevel mediation of the bias -> accuracy -> confidence path, with
RANDOM a, b AND c' SLOPES.

    bias (FAR or M-SDT bias)  --a-->  accuracy  --b-->  confidence
                              \\------- c' --------/       (direct effect)

The direct effect c' is the quantity of interest: a NEGATIVE c' means confidence
is discounted for alternatives the observer over-selects, once the accuracy
route has been accounted for. That down-weighting IS the signature of bias-aware
confidence; a positive c' indicates bias-blind confidence, where a tendency to
choose an alternative inflates rather than discounts confidence in it.

WHY RANDOM SLOPES ON ALL THREE PATHS
------------------------------------
Observers differ not only in baseline confidence (a random intercept) but in how
strongly bias drives their accuracy (a), how strongly accuracy drives their
confidence (b), and how strongly bias directly discounts their confidence (c').
Holding a, b and c' fixed while letting only intercepts vary understates the
uncertainty on all three, because between-observer slope variability is forced
into the residual. All five subject effects (two intercepts plus three slopes)
share a joint LKJ-prior covariance, so they may correlate rather than being
assumed independent.

That correlation matters for the indirect effect. With random a and b slopes the
population indirect effect is NOT simply a*b:

    E[(a + u_a)(b + u_b)] = a*b + Cov(u_a, u_b)

so ``indirect_effect`` carries the covariance term explicitly. Reporting a*b
alone would be biased whenever observers with a stronger a path also have a
stronger b path.

MODELS
------
``run_mediation``            one condition (each Experiment 1 condition, each ANN row).
``run_moderated_mediation``  both Experiment 2 SAT conditions fitted TOGETHER with
                             an effect-coded moderator (accuracy = +0.5, speed =
                             -0.5), so the accuracy-minus-speed difference in the
                             direct effect (``mod_direct``) and in the indirect
                             effect (``index_modmed``) each get their own
                             posterior instead of being compared across two
                             separate fits.

Usage
-----
    from scripts_analysis.mediation import run_mediation, summarize_mediation

    trace = run_mediation(df, predictor='FAR_z')
    print(summarize_mediation(trace))
"""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

MEDIATION_MODEL_VERSION = 'joint_correlated_random_abc_v1'

# Four chains are required for useful R-hat. Raise draws/tune if a fit is flagged.
DRAWS = 1000
TUNE = 1500
CHAINS = 4
TARGET_ACCEPT = 0.99
SEED = 20250720

# Random-effect order inside the joint 5-dimensional covariance.
RE_M_INTERCEPT, RE_Y_INTERCEPT, RE_A, RE_C, RE_B = range(5)
RE_LABELS = ('mediator_intercept', 'outcome_intercept', 'a', 'c_prime', 'b')

STANDARD_SUMMARY_VARS = [
    'a_path', 'b_mediator', 'c_prime', 'indirect_effect', 'total_effect',
    'mean_subject_indirect', 'sd_a', 'sd_b', 'sd_c_prime',
    'corr_a_b', 'corr_a_c_prime', 'corr_b_c_prime', 'cov_a_b',
]

MODERATED_SUMMARY_VARS = [
    'a_path_acc', 'a_path_speed', 'b_path_acc', 'b_path_speed',
    'direct_acc', 'direct_speed', 'indirect_acc', 'indirect_speed',
    'total_acc', 'total_speed', 'mod_direct', 'index_modmed',
    'a_by_condition', 'b_by_condition', 'c_by_condition',
    'sd_a', 'sd_b', 'sd_c_prime', 'corr_a_b', 'corr_a_c_prime',
    'corr_b_c_prime', 'cov_a_b',
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def hdi_star_rating(samples):
    """Star label from the widest HDI that still excludes zero ('n.s.' if none)."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    for probability, stars in ((0.999, '***'), (0.99, '**'), (0.95, '*')):
        lo, hi = az.hdi(samples, hdi_prob=probability)
        if lo > 0 or hi < 0:
            return stars
    return 'n.s.'


def prepare_mediation_data(df, predictor, mediator='accuracy_z', outcome='confidence_z',
                           moderator=None, subject_id_col='subject_idx'):
    """Validate and subset the frame; add a contiguous 0-based subject index."""
    columns = [predictor, mediator, outcome, subject_id_col]
    if moderator is not None:
        columns.append(moderator)
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f'Missing mediation columns: {missing}')
    clean = df[columns].dropna().copy()
    if clean.empty:
        raise ValueError('No complete rows remain for mediation.')
    for column in [predictor, mediator, outcome] + ([moderator] if moderator else []):
        clean[column] = pd.to_numeric(clean[column], errors='raise')
        if not np.isfinite(clean[column]).all():
            raise ValueError(f'Non-finite values in {column}.')
    clean['_subject'] = pd.Categorical(clean[subject_id_col]).codes.astype('int64')
    smallest = clean.groupby('_subject').size().min()
    if smallest < 5:
        warnings.warn(
            f'Only {int(smallest)} rows in the smallest cluster. Random a/b/c slopes, '
            'especially their covariance, may be weakly identified; inspect the '
            'diagnostics and HDIs.', RuntimeWarning)
    return clean


def _joint_random_effects(n_subjects):
    """Correlated subject effects: mediator intercept, outcome intercept, a, c', b."""
    chol, corr, stds = pm.LKJCholeskyCov(
        're_cov', n=5, eta=2.0,
        sd_dist=pm.HalfNormal.dist(1.0, shape=5),
        compute_corr=True,
    )
    z = pm.Normal('re_z', 0.0, 1.0, shape=(n_subjects, 5))
    effects = pm.Deterministic('subject_random_effects', pm.math.dot(z, chol.T))
    # Scalar aliases so summaries stay readable.
    pm.Deterministic('sd_a', stds[RE_A])
    pm.Deterministic('sd_c_prime', stds[RE_C])
    pm.Deterministic('sd_b', stds[RE_B])
    pm.Deterministic('corr_a_b', corr[RE_A, RE_B])
    pm.Deterministic('corr_a_c_prime', corr[RE_A, RE_C])
    pm.Deterministic('corr_b_c_prime', corr[RE_B, RE_C])
    pm.Deterministic('cov_a_b', corr[RE_A, RE_B] * stds[RE_A] * stds[RE_B])
    return effects


def _sample(model, seed, draws, tune, chains, cores, target_accept):
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                         target_accept=target_accept, random_seed=seed,
                         init='jitter+adapt_diag', return_inferencedata=True,
                         idata_kwargs={'log_likelihood': False})


# ──────────────────────────────────────────────────────────────────────────────
# Model 1: one condition
# ──────────────────────────────────────────────────────────────────────────────
def run_mediation(df, predictor='FAR_z', mediator='accuracy_z', outcome='confidence_z',
                  subject_id_col='subject_idx', draws=DRAWS, tune=TUNE, chains=CHAINS,
                  cores=2, target_accept=TARGET_ACCEPT, seed=SEED):
    """Multilevel mediation with correlated random a, b and c' slopes.

    Returns an ArviZ InferenceData; summarise with ``summarize_mediation``.
    """
    clean = prepare_mediation_data(df, predictor, mediator, outcome,
                                   subject_id_col=subject_id_col)
    x = clean[predictor].to_numpy(dtype=float)
    m = clean[mediator].to_numpy(dtype=float)
    y = clean[outcome].to_numpy(dtype=float)
    subject = clean['_subject'].to_numpy(dtype='int64')
    n_subjects = int(subject.max()) + 1

    with pm.Model() as model:
        random_effects = _joint_random_effects(n_subjects)

        mediator_intercept = pm.Normal('mediator_intercept', 0.0, 1.5)
        outcome_intercept = pm.Normal('outcome_intercept', 0.0, 1.5)
        a_path = pm.Normal('a_path', 0.0, 1.0)
        c_prime = pm.Normal('c_prime', 0.0, 1.0)
        b_mediator = pm.Normal('b_mediator', 0.0, 1.0)

        a_subject = pm.Deterministic('a_subject', a_path + random_effects[:, RE_A])
        c_subject = pm.Deterministic('c_prime_subject', c_prime + random_effects[:, RE_C])
        b_subject = pm.Deterministic('b_subject', b_mediator + random_effects[:, RE_B])

        sigma_m = pm.HalfNormal('sigma_m', 1.0)
        mu_m = (mediator_intercept + random_effects[subject, RE_M_INTERCEPT]
                + a_subject[subject] * x)
        pm.Normal('mediator_observed', mu=mu_m, sigma=sigma_m, observed=m)

        sigma_y = pm.HalfNormal('sigma_y', 1.0)
        mu_y = (outcome_intercept + random_effects[subject, RE_Y_INTERCEPT]
                + c_subject[subject] * x + b_subject[subject] * m)
        pm.Normal('outcome_observed', mu=mu_y, sigma=sigma_y, observed=y)

        # E[(a + u_a)(b + u_b)] = a*b + Cov(u_a, u_b)
        indirect = pm.Deterministic('indirect_effect',
                                    a_path * b_mediator + model['cov_a_b'])
        pm.Deterministic('total_effect', c_prime + indirect)
        pm.Deterministic('mean_subject_indirect',
                         pm.math.sum(a_subject * b_subject) / n_subjects)

    return _sample(model, seed, draws, tune, chains, cores, target_accept)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2: both Experiment 2 conditions fitted together
# ──────────────────────────────────────────────────────────────────────────────
def run_moderated_mediation(df, predictor='FAR_z', mediator='accuracy_z',
                            outcome='confidence_z', moderator='Cond_eff',
                            subject_id_col='subject_idx', draws=DRAWS, tune=TUNE,
                            chains=CHAINS, cores=2, target_accept=TARGET_ACCEPT, seed=SEED):
    """Moderated mediation: both SAT conditions in ONE model.

    ``moderator`` must be effect-coded (accuracy = +0.5, speed = -0.5), which
    makes ``mod_direct`` the accuracy-minus-speed difference in the direct effect
    and ``index_modmed`` the corresponding difference in the indirect effect.
    """
    clean = prepare_mediation_data(df, predictor, mediator, outcome, moderator,
                                   subject_id_col=subject_id_col)
    x = clean[predictor].to_numpy(dtype=float)
    m = clean[mediator].to_numpy(dtype=float)
    y = clean[outcome].to_numpy(dtype=float)
    w = clean[moderator].to_numpy(dtype=float)
    subject = clean['_subject'].to_numpy(dtype='int64')
    n_subjects = int(subject.max()) + 1

    with pm.Model() as model:
        random_effects = _joint_random_effects(n_subjects)

        mediator_intercept = pm.Normal('mediator_intercept', 0.0, 1.5)
        outcome_intercept = pm.Normal('outcome_intercept', 0.0, 1.5)

        a_path = pm.Normal('a_path', 0.0, 1.0)
        mediator_condition = pm.Normal('mediator_condition', 0.0, 1.0)
        a_by_condition = pm.Normal('a_by_condition', 0.0, 1.0)

        c_prime = pm.Normal('c_prime', 0.0, 1.0)
        outcome_condition = pm.Normal('outcome_condition', 0.0, 1.0)
        c_by_condition = pm.Normal('c_by_condition', 0.0, 1.0)
        b_mediator = pm.Normal('b_mediator', 0.0, 1.0)
        b_by_condition = pm.Normal('b_by_condition', 0.0, 1.0)

        a_subject = pm.Deterministic('a_subject', a_path + random_effects[:, RE_A])
        c_subject = pm.Deterministic('c_prime_subject', c_prime + random_effects[:, RE_C])
        b_subject = pm.Deterministic('b_subject', b_mediator + random_effects[:, RE_B])

        sigma_m = pm.HalfNormal('sigma_m', 1.0)
        mu_m = (mediator_intercept + random_effects[subject, RE_M_INTERCEPT]
                + a_subject[subject] * x
                + mediator_condition * w + a_by_condition * x * w)
        pm.Normal('mediator_observed', mu=mu_m, sigma=sigma_m, observed=m)

        sigma_y = pm.HalfNormal('sigma_y', 1.0)
        mu_y = (outcome_intercept + random_effects[subject, RE_Y_INTERCEPT]
                + c_subject[subject] * x
                + outcome_condition * w + c_by_condition * x * w
                + b_subject[subject] * m + b_by_condition * m * w)
        pm.Normal('outcome_observed', mu=mu_y, sigma=sigma_y, observed=y)

        a_acc = pm.Deterministic('a_path_acc', a_path + 0.5 * a_by_condition)
        a_speed = pm.Deterministic('a_path_speed', a_path - 0.5 * a_by_condition)
        b_acc = pm.Deterministic('b_path_acc', b_mediator + 0.5 * b_by_condition)
        b_speed = pm.Deterministic('b_path_speed', b_mediator - 0.5 * b_by_condition)
        direct_acc = pm.Deterministic('direct_acc', c_prime + 0.5 * c_by_condition)
        direct_speed = pm.Deterministic('direct_speed', c_prime - 0.5 * c_by_condition)

        indirect_acc = pm.Deterministic('indirect_acc', a_acc * b_acc + model['cov_a_b'])
        indirect_speed = pm.Deterministic('indirect_speed',
                                          a_speed * b_speed + model['cov_a_b'])
        pm.Deterministic('total_acc', direct_acc + indirect_acc)
        pm.Deterministic('total_speed', direct_speed + indirect_speed)
        pm.Deterministic('mod_direct', direct_acc - direct_speed)
        pm.Deterministic('index_modmed', indirect_acc - indirect_speed)

    return _sample(model, seed, draws, tune, chains, cores, target_accept)


# ──────────────────────────────────────────────────────────────────────────────
# Summaries and diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def _posterior_scalar(trace, variable):
    values = np.asarray(trace.posterior[variable]).reshape(-1)
    lo, hi = az.hdi(values, hdi_prob=0.95)
    return {
        'parameter': variable,
        'mean': float(np.mean(values)),
        'sd': float(np.std(values, ddof=1)),
        'hdi_2.5%': float(lo),
        'hdi_97.5%': float(hi),
        'r_hat': float(np.nanmax(np.asarray(az.rhat(trace, var_names=[variable]).to_array()))),
        'ess_bulk': float(np.nanmin(np.asarray(
            az.ess(trace, var_names=[variable], method='bulk').to_array()))),
        'ess_tail': float(np.nanmin(np.asarray(
            az.ess(trace, var_names=[variable], method='tail').to_array()))),
        'stars': hdi_star_rating(values),
        'p_gt_zero': float(np.mean(values > 0)),
    }


def summarize_mediation(trace, variables=None):
    """Posterior mean, SD, 95% HDI, R-hat, ESS and star label per parameter."""
    return pd.DataFrame([_posterior_scalar(trace, v)
                         for v in (variables or STANDARD_SUMMARY_VARS)])


def summarize_moderated_mediation(trace, variables=None):
    """As ``summarize_mediation``, for the joint Experiment 2 model."""
    return pd.DataFrame([_posterior_scalar(trace, v)
                         for v in (variables or MODERATED_SUMMARY_VARS)])


def diagnose(trace, summary, label=''):
    """Convergence audit. Returns (status, issues); status is 'OK' or 'CHECK'.

    A 'CHECK' does not automatically invalidate a result, but it must be looked
    at before the number is reported.
    """
    stats = trace.sample_stats
    divergences = int(np.asarray(stats['diverging']).sum())
    bfmi = np.asarray(az.bfmi(trace), dtype=float)
    if 'reached_max_treedepth' in stats:
        depth_hits = int(np.asarray(stats['reached_max_treedepth']).sum())
    elif 'tree_depth' in stats:
        depths = np.asarray(stats['tree_depth'])
        depth_hits = int(np.sum(depths == depths.max())) if depths.max() >= 10 else 0
    else:
        depth_hits = -1

    issues = []
    if divergences:
        issues.append(f'{divergences} divergences')
    if summary['r_hat'].max() > 1.01:
        issues.append(f"max R-hat {summary['r_hat'].max():.3f}")
    if summary['ess_bulk'].min() < 400:
        issues.append(f"min bulk ESS {summary['ess_bulk'].min():.0f}")
    if summary['ess_tail'].min() < 400:
        issues.append(f"min tail ESS {summary['ess_tail'].min():.0f}")
    if np.nanmin(bfmi) < 0.30:
        issues.append(f'min BFMI {np.nanmin(bfmi):.3f}')
    if depth_hits > 0:
        issues.append(f'{depth_hits} maximum-tree-depth hits')
    corr_rows = summary.loc[summary['parameter'].str.startswith('corr_'), 'mean']
    if len(corr_rows) and corr_rows.abs().max() > 0.95:
        issues.append(f'random-effect correlation |r| = {corr_rows.abs().max():.3f}')

    status = 'OK' if not issues else 'CHECK'
    print(f'[{status}] {label or "mediation"}'
          + (f' -- {"; ".join(issues)}' if issues else ''))
    return status, issues
