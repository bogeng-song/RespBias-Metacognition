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