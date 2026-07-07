"""
mediation.py
Hierarchical Bayesian Mediation Analysis using PyMC.
Decomposes the effect of Response Bias on Confidence into indirect (via Accuracy) and direct paths.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import warnings

warnings.filterwarnings("ignore")

def hdi_star_rating(samples):
    """Return significance stars based on HDI exclusion of zero."""
    hdi_999 = az.hdi(samples, hdi_prob=0.999)
    if hdi_999[0] > 0 or hdi_999[1] < 0: return '***'
    
    hdi_99 = az.hdi(samples, hdi_prob=0.99)
    if hdi_99[0] > 0 or hdi_99[1] < 0: return '**'
    
    hdi_95 = az.hdi(samples, hdi_prob=0.95)
    if hdi_95[0] > 0 or hdi_95[1] < 0: return '*'
    
    return 'ns'

def run_mixed_effect_mediation(df, predictor_var, mediator_var, outcome_var, subject_id_col='subject_idx', n_samples=2000, n_tune=1000):
    n_subjects = df[subject_id_col].nunique()
    subject_idxs = df[subject_id_col].values
    
    with pm.Model() as model:
        # Priors for group-level intercepts
        mu_a_m = pm.Normal('mu_a_m', mu=0, sigma=1)
        mu_a_y = pm.Normal('mu_a_y', mu=0, sigma=1)

        # Random intercepts per subject
        sigma_a_m = pm.HalfNormal('sigma_a_m', sigma=1)
        sigma_a_y = pm.HalfNormal('sigma_a_y', sigma=1)

        a_m = pm.Normal('a_m', mu=mu_a_m, sigma=sigma_a_m, shape=n_subjects)
        a_y = pm.Normal('a_y', mu=mu_a_y, sigma=sigma_a_y, shape=n_subjects)

        # Slopes (fixed effects)
        b_path = pm.Normal('b_path', mu=0, sigma=1)         # Predictor -> Mediator
        c_prime = pm.Normal('c_prime', mu=0, sigma=1)       # Predictor -> Outcome (Direct Effect)
        b_mediator = pm.Normal('b_mediator', mu=0, sigma=1) # Mediator -> Outcome

        # Residuals
        sigma_m = pm.HalfNormal('sigma_m', 1)
        sigma_y = pm.HalfNormal('sigma_y', 1)

        # Expected values
        mediator_hat = a_m[subject_idxs] + b_path * df[predictor_var].values
        outcome_hat = (
            a_y[subject_idxs] 
            + c_prime * df[predictor_var].values
            + b_mediator * df[mediator_var].values
        )

        # Likelihoods
        pm.Normal('mediator_obs', mu=mediator_hat, sigma=sigma_m, observed=df[mediator_var].values)
        pm.Normal('outcome_obs', mu=outcome_hat, sigma=sigma_y, observed=df[outcome_var].values)

        # Indirect and total effect
        pm.Deterministic('indirect_effect', b_path * b_mediator)
        pm.Deterministic('total_effect', (b_path * b_mediator) + c_prime)

        # Sample (NUTS)
        trace = pm.sample(n_samples, tune=n_tune, target_accept=0.95, return_inferencedata=True, cores=4)
        
    return trace

def summarize_mediation(trace):
    """Extracts HDI intervals and significance from PyMC trace into a DataFrame."""
    var_names = ['b_path', 'b_mediator', 'c_prime', 'indirect_effect', 'total_effect']
    posterior = trace.posterior

    results = []
    for var in var_names:
        samples = posterior[var].values.flatten()
        hdi95 = az.hdi(samples, hdi_prob=0.95)
        hdi99 = az.hdi(samples, hdi_prob=0.99)
        hdi999 = az.hdi(samples, hdi_prob=0.999)
        stars = hdi_star_rating(samples)
        results.append({
            "parameter": var,
            "mean": np.mean(samples),
            "hdi_95": hdi95,
            "hdi_99": hdi99,
            "hdi_999": hdi999,
            "significance": stars
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# JOINT (MODERATED) BAYESIAN MEDIATION — Experiment 2 (speed vs. accuracy)
#
# Fits both speed-accuracy-tradeoff conditions together with condition as an
# effect-coded moderator (W: Accuracy focus = +0.5, Speed focus = -0.5), and
# decomposes bias -> accuracy -> confidence into condition-specific direct and
# indirect paths.
#
#   Mediator model:  M = a_m[subj] + a1*X + a2*W + a3*(X*W)
#   Outcome model:   Y = a_y[subj] + c1*X + c2*W + c3*(X*W) + b1*M + b2*(M*W)
#
# with X = response bias, M = accuracy, Y = confidence. The key inferential
# quantities are the per-condition direct effects (direct_acc / direct_speed),
# the moderation of the direct path (mod_direct == c3), and the index of
# moderated mediation (index_modmed = indirect_acc - indirect_speed).
# ══════════════════════════════════════════════════════════════════════════════
def run_moderated_mediation(df, predictor='FAR_z', mediator='accuracy_z',
                            outcome='confidence_z', moderator='Cond_eff',
                            subject_id_col='subject_idx',
                            w_acc=0.5, w_speed=-0.5,
                            n_samples=2000, n_tune=1000, target_accept=0.95,
                            random_seed=42):
    """Joint Bayesian moderated mediation for the two Experiment-2 conditions.

    ``df`` must contain the predictor (bias), mediator (accuracy), outcome
    (confidence), an effect-coded ``moderator`` column, and a 0-indexed
    ``subject_id_col``. Returns the PyMC ``trace``.
    """
    codes = pd.Categorical(df[subject_id_col]).codes
    n_subj = int(codes.max()) + 1
    X = df[predictor].values
    M = df[mediator].values
    Y = df[outcome].values
    W = df[moderator].values

    with pm.Model() as model:
        # --- Mediator model (bias -> accuracy) ---
        mu_a_m = pm.Normal('mu_a_m', 0, 1)
        sigma_a_m = pm.HalfNormal('sigma_a_m', 1)
        a_m = pm.Normal('a_m', mu_a_m, sigma_a_m, shape=n_subj)
        a1 = pm.Normal('a1', 0, 1)   # bias -> accuracy
        a2 = pm.Normal('a2', 0, 1)   # condition -> accuracy
        a3 = pm.Normal('a3', 0, 1)   # bias x condition -> accuracy
        sigma_m = pm.HalfNormal('sigma_m', 1)
        mediator_hat = a_m[codes] + a1 * X + a2 * W + a3 * (X * W)
        pm.Normal('M_obs', mu=mediator_hat, sigma=sigma_m, observed=M)

        # --- Outcome model (bias + accuracy -> confidence) ---
        mu_a_y = pm.Normal('mu_a_y', 0, 1)
        sigma_a_y = pm.HalfNormal('sigma_a_y', 1)
        a_y = pm.Normal('a_y', mu_a_y, sigma_a_y, shape=n_subj)
        c1 = pm.Normal('c1', 0, 1)   # bias -> confidence (direct)
        c2 = pm.Normal('c2', 0, 1)   # condition -> confidence
        c3 = pm.Normal('c3', 0, 1)   # bias x condition -> confidence (moderation of direct path)
        b1 = pm.Normal('b1', 0, 1)   # accuracy -> confidence
        b2 = pm.Normal('b2', 0, 1)   # accuracy x condition -> confidence
        sigma_y = pm.HalfNormal('sigma_y', 1)
        outcome_hat = a_y[codes] + c1 * X + c2 * W + c3 * (X * W) + b1 * M + b2 * (M * W)
        pm.Normal('Y_obs', mu=outcome_hat, sigma=sigma_y, observed=Y)

        # --- Condition-specific deterministics ---
        a_acc = pm.Deterministic('a_path_acc', a1 + a3 * w_acc)
        a_spd = pm.Deterministic('a_path_speed', a1 + a3 * w_speed)
        b_acc = pm.Deterministic('b_path_acc', b1 + b2 * w_acc)
        b_spd = pm.Deterministic('b_path_speed', b1 + b2 * w_speed)
        d_acc = pm.Deterministic('direct_acc', c1 + c3 * w_acc)
        d_spd = pm.Deterministic('direct_speed', c1 + c3 * w_speed)
        i_acc = pm.Deterministic('indirect_acc', a_acc * b_acc)
        i_spd = pm.Deterministic('indirect_speed', a_spd * b_spd)
        pm.Deterministic('total_acc', d_acc + i_acc)
        pm.Deterministic('total_speed', d_spd + i_spd)
        pm.Deterministic('mod_direct', d_acc - d_spd)        # == c3
        pm.Deterministic('index_modmed', i_acc - i_spd)      # index of moderated mediation

        trace = pm.sample(n_samples, tune=n_tune, target_accept=target_accept,
                          random_seed=random_seed, return_inferencedata=True)

    return trace


def summarize_moderated_mediation(trace):
    """Grouped path table (per-condition a/b/direct/indirect/total plus the
    moderation index) with HDI-based significance stars."""
    var_names = [
        'a_path_acc', 'b_path_acc', 'direct_acc', 'indirect_acc', 'total_acc',
        'a_path_speed', 'b_path_speed', 'direct_speed', 'indirect_speed', 'total_speed',
        'mod_direct', 'index_modmed',
    ]
    posterior = trace.posterior
    rows = []
    for var in var_names:
        samples = posterior[var].values.flatten()
        rows.append({
            "parameter": var,
            "mean": float(np.mean(samples)),
            "hdi_95": az.hdi(samples, hdi_prob=0.95),
            "significance": hdi_star_rating(samples),
        })
    return pd.DataFrame(rows)