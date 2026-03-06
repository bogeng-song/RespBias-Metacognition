"""
metrics.py
Utility functions for calculating Signal Detection Theory (SDT) parameters.
"""
import numpy as np
from scipy.stats import norm

def compute_sdt_params(stim, resp):
    """
    Computes Signal Detection Theory parameters with Log-Linear Correction (Hautus, 1995).
    stim: Binary array (1 = Target, 0 = Non-Target)
    resp: Binary array (1 = Target Selected, 0 = Target Not Selected)
    """
    stim = np.asarray(stim) > 0
    resp = np.asarray(resp) > 0

    n_signal = np.sum(stim)
    n_noise = np.sum(~stim)
    
    hits = np.sum(stim & resp)
    false_alarms = np.sum((~stim) & resp)

    # Apply Log-Linear Correction (Hautus, 1995) to prevent infinite d'
    if hits == n_signal: hits = n_signal - 0.5
    elif hits == 0: hits = 0.5
        
    if false_alarms == n_noise: false_alarms = n_noise - 0.5
    elif false_alarms == 0: false_alarms = 0.5

    # Calculate rates
    HR = np.clip(hits / n_signal, 1e-9, 1 - 1e-9) if n_signal > 0 else 0
    FAR = np.clip(false_alarms / n_noise, 1e-9, 1 - 1e-9) if n_noise > 0 else 0

    # Calculate SDT parameters
    dprime = norm.ppf(HR) - norm.ppf(FAR)
    c = -0.5 * (norm.ppf(HR) + norm.ppf(FAR))
    beta = np.exp(dprime * c)

    return dprime, c, beta, HR, FAR