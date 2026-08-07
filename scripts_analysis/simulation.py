"""
simulation.py
Multi-alternative SDT simulation of bias-blind versus bias-aware confidence.

THE MODEL
---------
On each trial an observer sees one of K alternatives. Evidence for every
alternative is drawn from a unit normal, with the true alternative's mean raised
by ``mu``. Each observer also carries a fixed, mean-centred response-bias vector
``beta ~ N(0, sigma_b^2)`` that is added to the evidence before the choice:

    decision variable   = x + beta          -> the response (argmax)
    confidence evidence = x + beta - alpha*beta

``alpha`` is the BIAS-CORRECTION STRENGTH and is the only manipulated quantity:

    alpha = 0   confidence uses the same biased evidence that drove the choice
                -> BIAS-BLIND confidence
    alpha = 1   confidence removes the bias vector entirely
                -> fully BIAS-AWARE confidence

Because ``beta`` enters the decision variable regardless of ``alpha``, the
observer's choices and accuracy are identical across the whole sweep. Only the
confidence readout changes, which is what isolates the effect of correction.

Confidence itself is the top-2 evidence gap: the chosen alternative's evidence
minus the best of the remaining alternatives. The runner-up is RE-SELECTED
inside whichever evidence space is used, so a bias-corrected readout compares
the chosen alternative against the best non-chosen alternative *in corrected
space* -- generally a different alternative from the biased-space runner-up.

WHAT IT PREDICTS (manuscript Figure 2)
--------------------------------------
Regressing digit-level confidence on digit-level FAR while controlling for
digit-level accuracy gives a FAR coefficient whose SIGN is a signature of the
readout: positive under bias-blind confidence, becoming negative as correction
strengthens. Metacognitive sensitivity (Phi, the within-observer correlation
between confidence and accuracy) increases with alpha.

CALIBRATION
-----------
``sigma_b`` is solved so the simulated within-observer dispersion of FAR across
alternatives matches the empirical dispersion (.0704 for 4-choice, .0433 for
8-choice), by bounded Brent root search over [0.001, 1.5] to tolerance 1e-4. At
every candidate, ``mu`` is recalibrated and the objective is averaged over 10
independently generated full designs. The solved values are the shipped defaults:

    4-choice  sigma_b = 0.368
    8-choice  sigma_b = 0.446

``calibrate_sigma_b`` re-runs that search; it is not needed to reproduce the
figure, only to audit the calibration.

``mu`` is calibrated per condition so OVERALL accuracy matches ``target_acc``
(0.64, approximating the empirical .63), so the two conditions are matched on
performance rather than on evidence strength.

``no_bias_benchmark`` is the separate null used with the behavioural checks: the
response-bias vector is fixed at zero, giving the FAR dispersion expected from
finite sampling alone in an observer with no stable response preference.

FAR BOUNDARY CORRECTION
-----------------------
The digit-level analyses and the sigma_b calibration apply the half-count
boundary correction (0 -> 0.5, n -> n - 0.5), matching the behavioural
preprocessing they are calibrated against. The trial-level analysis uses RAW
counts, because there FAR is only a per-trial predictor and is never aggregated
to the alternative level. ``observer_far(..., half_count=...)`` makes the choice
explicit at each call site.

KNOWN LIMITATION (stated, not modelled)
---------------------------------------
Every alternative has identical discriminability here, so between-alternative
accuracy variation comes almost entirely from response bias. Simulated
within-observer corr(FAR, accuracy) is ~0.95 (K=4) and ~0.92 (K=8), against
~0.64 and ~0.60 in the human data, where digits also differ in intrinsic
difficulty (38%-79% accuracy across digits in the 8-choice condition).

Reported seeds: generation 1, calibration 2026, trial-level analysis 20250722.

Usage
-----
python scripts_analysis/simulation.py --output figure2_simulation.pdf
python scripts_analysis/simulation.py --trial-level --csv supp_trial_level.csv
python scripts_analysis/simulation.py --no-bias-benchmark --csv null_far.csv
python scripts_analysis/simulation.py --calibrate          # re-solve sigma_b (slow)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import brentq
from scipy.stats import pointbiserialr

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (the reported design)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SIGMA_B = {4: 0.368, 8: 0.446}   # calibrated to the empirical FAR spread
TARGET_ACC = 0.64                        # overall p(correct); empirical ~0.628
N_SUBJ = 200                             # matches the empirical sample after exclusions
# The reported digit-level sweep (Figure 2) uses 1000 observers. The point
# estimates are the same at 200; the extra observers only narrow the intervals,
# and the sweep is cheap enough that there is no reason to publish the wider ones.
# The trial-level sweep and the calibration stay at N_SUBJ, where matching the
# empirical sample size is the point.
N_SUBJ_SWEEP = 1000
N_TRIAL = 400                            # matches the empirical trials per observer
CONF_NOISE = 0.0                         # late/readout noise SD on the finished confidence
ALPHA_VALUES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)

# Empirical within-observer FAR dispersion the calibration targets (mean across
# observers of the SD of FAR across alternatives).
EMPIRICAL_FAR_SD = {4: 0.0704, 8: 0.0433}

# Fixed seeds for the reported runs: generation, calibration, trial-level analysis.
SEED_GENERATION = 1
SEED_CALIBRATION = 2026
SEED_TRIAL_LEVEL = 20250722

N_BOOT = 2000          # observer-level bootstrap resamples for the Phi CI
N_NULL_OBSERVERS = 2000   # null observers per choice-set size for the no-bias benchmark

# Two nested models per alpha, mirroring the behavioural analysis (no RT in the model).
MODEL_SPECS = {
    'far':     ('FARz',            'FAR only'),
    'far_acc': ('FARz + acc_hitz', 'FAR + accuracy'),
}


# ──────────────────────────────────────────────────────────────────────────────
# Generative model
# ──────────────────────────────────────────────────────────────────────────────
def calibrate_mu(K, sigma_b, rng, target=TARGET_ACC, n_subj=400, n_per=500, n_iter=40):
    """Bisect ``mu`` so overall p(correct) equals ``target`` under the same bias structure."""
    lo, hi = 0.1, 7.0
    ii = np.arange(n_subj)[:, None]
    jj = np.arange(n_per)[None, :]
    for _ in range(n_iter):
        mu = 0.5 * (lo + hi)
        beta = rng.standard_normal((n_subj, 1, K)) * sigma_b
        beta -= beta.mean(2, keepdims=True)
        stim = rng.integers(0, K, (n_subj, n_per))
        x = rng.standard_normal((n_subj, n_per, K))
        x[ii, jj, stim] += mu
        acc = ((x + beta).argmax(2) == stim).mean()
        lo, hi = (mu, hi) if acc < target else (lo, mu)
    return 0.5 * (lo + hi)


def alpha_corrected_evidence(x, beta, alpha):
    """Evidence entering the confidence readout: ``x + beta - alpha * beta``."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'alpha must lie in [0, 1], got {alpha}')
    return x + beta - alpha * beta


def top2diff(v, choice):
    """Chosen alternative's evidence minus the best of the REMAINING alternatives.

    The runner-up is re-selected inside the evidence space of ``v``, so a
    bias-corrected readout may compare against a different alternative than the
    biased readout would. No per-trial normalisation is applied.
    """
    n = len(choice)
    chosen = v[np.arange(n), choice]
    rest = v.copy()
    rest[np.arange(n), choice] = -np.inf
    return chosen - rest.max(1)


def generate_observer(K, mu, sigma_b, n_trials, conf_noise, rng, alphas=ALPHA_VALUES):
    """Simulate one observer. Returns (stim, choice, correct, {alpha: confidence}).

    The metacognitive noise is LATE (readout) noise: one scalar per trial, added
    to the finished confidence and SHARED across alpha values (common random
    numbers), so the alpha trajectory isolates the correction manipulation and
    cannot change the choice, the runner-up, or any FAR quantity.
    """
    beta = rng.standard_normal(K) * sigma_b
    beta -= beta.mean()
    stim = rng.integers(0, K, n_trials)
    x = rng.standard_normal((n_trials, K))
    x[np.arange(n_trials), stim] += mu

    choice = (x + beta).argmax(1)          # alpha changes confidence, never the choice
    correct = (choice == stim).astype(int)

    late_noise = rng.standard_normal(n_trials) * conf_noise
    confidence = {}
    for alpha in np.asarray(alphas, dtype=float):
        evidence = alpha_corrected_evidence(x, beta, alpha)
        confidence[float(alpha)] = top2diff(evidence, choice) + late_noise
    return stim, choice, correct, confidence


def validate_alpha_endpoints():
    """Algebraic check: alpha=0 leaves the bias in, alpha=1 removes it exactly."""
    rng = np.random.default_rng(20250721)
    x = rng.normal(size=(20, 4))
    beta = rng.normal(size=4)
    beta -= beta.mean()
    assert np.allclose(alpha_corrected_evidence(x, beta, 0.0), x + beta)
    assert np.allclose(alpha_corrected_evidence(x, beta, 1.0), x)
    return True


def observer_far(stim, choice, K, half_count=True):
    """FAR per alternative: P(resp = k | stim != k).

    ``half_count`` applies the boundary correction used by the behavioural
    preprocessing: an alternative with zero false alarms is counted as 0.5, and
    one with false alarms on every eligible trial as n - 0.5. This keeps FAR away
    from 0 and 1, which matters when FAR is aggregated to the alternative level.

    The DIGIT-LEVEL analyses and the sigma_b calibration use it (half_count=True),
    matching the behavioural pipeline they are calibrated against. The TRIAL-LEVEL
    analysis does NOT: there FAR is only a per-trial predictor, never aggregated,
    so the boundary correction would shift the predictor without purpose. Passing
    the choice explicitly keeps the two paths auditable rather than implicit.
    """
    far = np.empty(K)
    for k in range(K):
        notk = stim != k
        n_noise = float(notk.sum())
        count = float(((choice == k) & notk).sum())
        if n_noise == 0:
            far[k] = np.nan
            continue
        if half_count:
            if count == 0.0:
                count = 0.5
            elif count == n_noise:
                count = n_noise - 0.5
        far[k] = count / n_noise
    return far


# ──────────────────────────────────────────────────────────────────────────────
# Digit-level aggregation -- matches the human pipeline exactly
# ──────────────────────────────────────────────────────────────────────────────
def digit_level_table(stim, choice, correct, confidence, K, sid):
    """One row per alternative, with the same conditioning as the human arrays.

        FAR     = P(resp = k | stim != k), half-count corrected
        acc_hit = P(correct | stim = k)
        conf    = E(confidence | stim = k)

    Confidence is keyed by STIMULUS, not by response, because the empirical
    pipeline pairs a response-based FAR with a stimulus-based confidence. Every
    alternative is kept, matching the human arrays which contain all K digits for
    every participant.
    """
    far = observer_far(stim, choice, K)
    rows = []
    for k in range(K):
        stimk = stim == k
        if stimk.sum() == 0 or np.isnan(far[k]):
            continue
        rows.append((sid, far[k], correct[stimk].mean(), confidence[stimk].mean()))
    return pd.DataFrame(rows, columns=['sid', 'FAR', 'acc_hit', 'conf'])


def add_global_z(df, columns=('FAR', 'acc_hit', 'conf')):
    """Z-score each column globally, across all observers x alternatives.

    Matches the human pipeline, which applies sklearn's StandardScaler to the
    whole column rather than within observer; StandardScaler divides by the
    population SD, hence ddof=0.
    """
    out = df.copy()
    for column in columns:
        sd = out[column].std(ddof=0)
        out[column + 'z'] = (out[column] - out[column].mean()) / sd if sd > 0 else 0.0
    return out


def bootstrap_ci(values, n_boot=N_BOOT, seed=0, alpha=0.05):
    """Percentile CI for the mean, resampling OBSERVERS (not trials)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def add_within_subject_z(df, columns=('FAR', 'conf'), group='sid'):
    """Z-score each column WITHIN observer.

    The trial-level analysis standardises within observer, unlike the
    alternative-level analysis which standardises across all observer-by-
    alternative rows. Within-observer scaling removes between-observer
    differences in confidence scale, which matter far more at the trial level
    where every observer contributes hundreds of rows.
    """
    out = df.copy()
    for column in columns:
        grouped = out.groupby(group)[column]
        sd = grouped.transform('std')
        out[column + 'z'] = np.where(sd > 0, (out[column] - grouped.transform('mean')) / sd, 0.0)
    return out


def phi(confidence, correct):
    """Metacognitive sensitivity: within-observer confidence-accuracy correlation.

    Point-biserial r, which for a binary variable is identical to Pearson's r.
    """
    if len(np.unique(correct)) < 2 or np.std(confidence) == 0:
        return np.nan
    return float(pointbiserialr(correct, confidence)[0])


def far_accuracy_vif(df, columns=('FARz', 'acc_hitz')):
    """VIF between FAR and accuracy in the adjusted model.

    With two predictors both VIFs are equal and reduce to 1 / (1 - r^2). This is
    reported per alpha because the whole point of the adjusted model is that FAR
    and accuracy are correlated by construction -- they are computed from the
    same finite trials and driven by the same per-alternative bias.
    """
    x, y = df[columns[0]].to_numpy(dtype=float), df[columns[1]].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    return 1.0 / (1.0 - r ** 2), r


# ──────────────────────────────────────────────────────────────────────────────
# Estimator -- the same specification as the empirical regressions
# ──────────────────────────────────────────────────────────────────────────────
def mixed_far_coef(df, rhs, group='sid', target='FARz'):
    """Mixed model ``confz ~ rhs`` returning the target term's (coef, lo, hi, se, z, p).

    Random intercept + random slope on the target, matching the human
    specification, with the same fallback ladder: if the random slope is not
    estimable the fit degrades to a random intercept and says so, because the
    intercept-only SE is understated relative to a random-slope fit.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ('powell', 'lbfgs', 'cg', 'bfgs', 'nm'):
            try:
                m = smf.mixedlm(f'confz ~ {rhs}', df, groups=df[group],
                                re_formula=f'~ {target}').fit(method=method)
                se = float(m.bse_fe[target])
                if np.isfinite(se) and se > 0 and np.isfinite(m.params[target]):
                    ci = m.conf_int().loc[target]
                    return (float(m.params[target]), float(ci.iloc[0]), float(ci.iloc[1]),
                            se, float(m.tvalues[target]), float(m.pvalues[target]))
            except Exception:
                continue
        for method in ('lbfgs', 'powell', 'cg'):
            try:
                m = smf.mixedlm(f'confz ~ {rhs}', df, groups=df[group]).fit(method=method)
                se = float(m.bse_fe[target])
                if np.isfinite(se) and se > 0:
                    ci = m.conf_int().loc[target]
                    print('    [!] random slope not estimable -> random intercept only; '
                          'SE is understated.')
                    return (float(m.params[target]), float(ci.iloc[0]), float(ci.iloc[1]),
                            se, float(m.tvalues[target]), float(m.pvalues[target]))
            except Exception:
                continue
    return (np.nan,) * 6


# ──────────────────────────────────────────────────────────────────────────────
# The alpha sweep (Figure 2)
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_observers(K, sigma_b, n_subj, n_trial, conf_noise, target_acc, alphas, rng,
                        verbose=True, label=''):
    mu = calibrate_mu(K, sigma_b, rng, target=target_acc)
    if verbose:
        print(f'{label}: sigma_b={sigma_b:.3f}  mu={mu:.3f}  '
              f'({n_subj} observers x {n_trial} trials)')
    observers = [generate_observer(K, mu, sigma_b, n_trial, conf_noise, rng, alphas)
                 for _ in range(n_subj)]
    return mu, observers


def run_alpha_sweep(k_list=(4, 8), n_subj=N_SUBJ_SWEEP, n_trial=N_TRIAL, sigma_b=None,
                    conf_noise=CONF_NOISE, target_acc=TARGET_ACC,
                    alphas=ALPHA_VALUES, seed=SEED_GENERATION, n_boot=N_BOOT,
                    verbose=True):
    """Digit-level FAR coefficient and Phi across bias-correction strength.

    Returns a tidy DataFrame with one row per (condition, alpha, nested model).
    """
    validate_alpha_endpoints()
    sigma_b = dict(DEFAULT_SIGMA_B) if sigma_b is None else dict(sigma_b)
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=float)
    rows = []

    for K in k_list:
        label = f'{K}-choice'
        sb = sigma_b[K]
        mu, observers = _simulate_observers(K, sb, n_subj, n_trial, conf_noise, target_acc,
                                            alphas, rng, verbose, label)
        overall_acc = float(np.mean([o[2].mean() for o in observers]))

        for alpha in alphas:
            frames, phis = [], []
            for sid, (stim, choice, correct, conf) in enumerate(observers):
                frames.append(digit_level_table(stim, choice, correct,
                                                conf[float(alpha)], K, sid))
                phis.append(phi(conf[float(alpha)], correct))
            df = add_global_z(pd.concat(frames, ignore_index=True))

            # Phi CI from an OBSERVER-level bootstrap; VIF reported per alpha
            # because FAR and accuracy are correlated by construction.
            phi_lo, phi_hi = bootstrap_ci(phis, n_boot=n_boot, seed=seed)
            vif, r_far_acc = far_accuracy_vif(df)
            for key, (rhs, model_label) in MODEL_SPECS.items():
                coef, lo, hi, se, z, p = mixed_far_coef(df, rhs)
                rows.append({'condition': label, 'K': K, 'alpha': float(alpha),
                             'model': model_label, 'model_key': key,
                             'coef': coef, 'ci_low': lo, 'ci_high': hi,
                             'se': se, 'z': z, 'p': p,
                             'phi': float(np.nanmean(phis)),
                             'phi_ci_low': phi_lo, 'phi_ci_high': phi_hi,
                             'phi_sem': float(np.nanstd(phis, ddof=1) / np.sqrt(len(phis))),
                             'vif_far_accuracy': vif, 'r_far_accuracy': r_far_acc,
                             'overall_accuracy': overall_acc,
                             'n_subj': n_subj, 'n_trial': n_trial, 'sigma_b': sb, 'mu': mu})
            if verbose:
                last = rows[-1]
                print(f'  alpha={alpha:.1f}  FAR|accuracy beta={last["coef"]:+.4f} '
                      f'[{last["ci_low"]:+.4f}, {last["ci_high"]:+.4f}]  '
                      f'Phi={last["phi"]:.3f} '
                      f'[{last["phi_ci_low"]:.3f}, {last["phi_ci_high"]:.3f}]  '
                      f'VIF={last["vif_far_accuracy"]:.2f}')
    return pd.DataFrame(rows)


def alpha_digit_frames(k_list=(4, 8), n_subj=N_SUBJ, n_trial=N_TRIAL, sigma_b=None,
                       conf_noise=CONF_NOISE, target_acc=TARGET_ACC,
                       alphas=ALPHA_VALUES, seed=SEED_GENERATION, verbose=True):
    """Digit-level frames per (condition, alpha), named for the mediation model.

    The same simulated observers ``run_alpha_sweep`` regresses, handed over with
    the column names ``mediation.run_mediation`` expects (``FAR_z``,
    ``accuracy_z``, ``confidence_z``, ``subject_idx``). Feeding one generative run
    to both the regression and the mediation is the point: any difference between
    them is the analysis, not a different simulation.

    Returns ``{(condition, alpha): DataFrame}``.
    """
    validate_alpha_endpoints()
    sigma_b = dict(DEFAULT_SIGMA_B) if sigma_b is None else dict(sigma_b)
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=float)
    out = {}

    for K in k_list:
        label = f'{K}-choice'
        _, observers = _simulate_observers(K, sigma_b[K], n_subj, n_trial, conf_noise,
                                           target_acc, alphas, rng, verbose,
                                           f'{label} (mediation)')
        for alpha in alphas:
            frames = [digit_level_table(stim, choice, correct, conf[float(alpha)], K, sid)
                      for sid, (stim, choice, correct, conf) in enumerate(observers)]
            df = add_global_z(pd.concat(frames, ignore_index=True))
            out[(label, float(alpha))] = df.rename(columns={
                'FARz': 'FAR_z', 'acc_hitz': 'accuracy_z', 'confz': 'confidence_z',
                'sid': 'subject_idx'})
    return out


def run_trial_level_alpha_sweep(k_list=(4, 8), n_subj=N_SUBJ, n_trial=N_TRIAL, sigma_b=None,
                                conf_noise=CONF_NOISE, target_acc=TARGET_ACC,
                                alphas=ALPHA_VALUES, seed=SEED_TRIAL_LEVEL, verbose=True):
    """Trial-level counterpart: FAR of the CHOSEN alternative predicting trial confidence.

    The digit-level analysis collapses each observer to K points; here every
    trial is a row. FAR is computed per alternative from that observer's own
    response frequencies (identical across alpha, since alpha never changes
    choices) and attached to each trial by the chosen alternative.

    FAR and confidence are standardised WITHIN observer here, unlike the
    alternative-level analysis which standardises globally: at the trial level
    every observer contributes hundreds of rows, so between-observer differences
    in confidence scale would otherwise dominate.
    """
    validate_alpha_endpoints()
    sigma_b = dict(DEFAULT_SIGMA_B) if sigma_b is None else dict(sigma_b)
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=float)
    rows = []

    for K in k_list:
        label = f'{K}-choice'
        sb = sigma_b[K]
        _, observers = _simulate_observers(K, sb, n_subj, n_trial, conf_noise, target_acc,
                                           alphas, rng, verbose, f'{label} (trial level)')
        # raw counts here: FAR is a per-trial predictor, never aggregated (see observer_far)
        far_by_observer = [observer_far(stim, choice, K, half_count=False)
                           for stim, choice, _, _ in observers]

        for alpha in alphas:
            frames = []
            for sid, (stim, choice, correct, conf) in enumerate(observers):
                frames.append(pd.DataFrame({'sid': sid,
                                            'FAR': far_by_observer[sid][choice],
                                            'conf': conf[float(alpha)]}))
            df = add_within_subject_z(pd.concat(frames, ignore_index=True),
                                      columns=('FAR', 'conf'))
            coef, lo, hi, se, z, p = mixed_far_coef(df, 'FARz')
            rows.append({'condition': label, 'K': K, 'alpha': float(alpha),
                         'coef': coef, 'ci_low': lo, 'ci_high': hi,
                         'se': se, 'z': z, 'p': p,
                         'n_subj': n_subj, 'n_trial': n_trial, 'sigma_b': sb})
            if verbose:
                print(f'  alpha={alpha:.1f}  trial-level FAR beta={coef:+.4f} '
                      f'[{lo:+.4f}, {hi:+.4f}]')
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Optional: recalibrate sigma_b against an empirical FAR array
# ──────────────────────────────────────────────────────────────────────────────
def simulated_far_dispersion(K, sigma_b, rng, n_subj=N_SUBJ, n_trial=N_TRIAL,
                             target_acc=TARGET_ACC, n_rep=10):
    """Mean within-observer SD of FAR across alternatives, at this sigma_b.

    ``mu`` is recalibrated at every candidate sigma_b (bias strength shifts
    accuracy), and the objective is averaged over ``n_rep`` independently
    generated full designs so the calibration is not matched to a single
    stochastic draw.
    """
    means = []
    for _ in range(n_rep):
        mu = calibrate_mu(K, sigma_b, rng, target=target_acc)
        spreads = []
        for _ in range(n_subj):
            stim, choice, _, _ = generate_observer(K, mu, sigma_b, n_trial, 0.0, rng,
                                                   alphas=[0.0])
            spreads.append(np.nanstd(observer_far(stim, choice, K), ddof=1))
        means.append(np.mean(spreads))
    return float(np.mean(means))


def calibrate_sigma_b(K, target_far_sd=None, seed=SEED_CALIBRATION, bracket=(0.001, 1.5),
                      tol=1e-4, n_rep=10, n_subj=N_SUBJ, n_trial=N_TRIAL,
                      target_acc=TARGET_ACC, verbose=True):
    """Solve for the sigma_b that reproduces the empirical FAR dispersion.

    Bounded Brent root search over ``bracket``, to absolute and relative
    tolerance ``tol``. ``target_far_sd`` defaults to the empirical value for this
    choice-set size (EMPIRICAL_FAR_SD).

    Not needed to reproduce the figure -- the solved values are the shipped
    DEFAULT_SIGMA_B (4-choice 0.368, 8-choice 0.446). Provided so the calibration
    is auditable and re-runnable. It is expensive: each evaluation regenerates
    ``n_rep`` full designs.
    """
    target = EMPIRICAL_FAR_SD[K] if target_far_sd is None else float(target_far_sd)
    rng = np.random.default_rng(seed)

    def objective(sigma_b):
        simulated = simulated_far_dispersion(K, sigma_b, rng, n_subj, n_trial,
                                             target_acc, n_rep)
        if verbose:
            print(f'    sigma_b={sigma_b:.4f} -> simulated SD_FAR={simulated:.5f} '
                  f'(target {target:.5f})')
        return simulated - target

    solved = float(brentq(objective, bracket[0], bracket[1], xtol=tol, rtol=tol))
    if verbose:
        print(f'K={K}: sigma_b* = {solved:.3f}  (empirical SD_FAR target {target:.4f})')
    return solved


def no_bias_benchmark(K, n_observers=N_NULL_OBSERVERS, n_trial=N_TRIAL,
                      target_acc=TARGET_ACC, seed=SEED_CALIBRATION, verbose=True,
                      half_count=False):
    """FAR dispersion expected from finite sampling alone, with NO response bias.

    The response-bias vector is fixed at zero, so the chosen alternative is simply
    the one with the largest evidence. This gives the null distribution of
    within-observer FAR dispersion against which the empirical spread is compared:
    it answers "how uneven would FAR look even in an observer with no stable
    response preference, purely because each alternative is sampled a finite
    number of times?"

    This is the reference ``controls.far_variability`` tests against, and the
    dashed line in Figure 1 panel D. Match ``target_acc`` and ``n_trial`` to the
    condition being compared -- the null is only meaningful when the simulated
    observers are as accurate, and run as many trials, as the real ones.

    ``half_count`` is False here to match the empirical FAR the comparison uses
    (raw counts, ``controls.far_matrix_from_trials``). At a few hundred trials per
    observer the correction is inert either way: a zero cell does not occur.

    Returns (dispersions, summary) where ``dispersions`` has one value per null
    observer.
    """
    rng = np.random.default_rng(seed)
    mu = calibrate_mu(K, 0.0, rng, target=target_acc)
    dispersions = np.empty(n_observers)
    for i in range(n_observers):
        stim, choice, _, _ = generate_observer(K, mu, 0.0, n_trial, 0.0, rng, alphas=[0.0])
        dispersions[i] = np.nanstd(observer_far(stim, choice, K, half_count=half_count),
                                   ddof=1)
    summary = {
        'K': K, 'n_observers': int(n_observers), 'n_trial': int(n_trial), 'mu': mu,
        'mean_far_sd': float(dispersions.mean()),
        'sd_far_sd': float(dispersions.std(ddof=1)),
        'pct_2.5': float(np.percentile(dispersions, 2.5)),
        'pct_97.5': float(np.percentile(dispersions, 97.5)),
        'empirical_far_sd': EMPIRICAL_FAR_SD.get(K, np.nan),
    }
    if verbose:
        print(f'K={K} no-bias null ({n_observers} observers): FAR SD = '
              f'{summary["mean_far_sd"]:.4f} '
              f'[{summary["pct_2.5"]:.4f}, {summary["pct_97.5"]:.4f}]   '
              f'empirical = {summary["empirical_far_sd"]:.4f}')
    return dispersions, summary


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
# One panel per MODEL, both conditions in each. Splitting by model rather than by
# condition is what makes the paper's point visible: panel A is the bias-blind
# result and panel B the same sweep once accuracy is controlled, so the sign flip
# is a comparison between two panels rather than two lines inside one.
FIGURE2_PANELS = [('far',     'Bias-only regression',
                   'FAR coefficient\n(no accuracy control)'),
                  ('far_acc', 'Bias + Acc regression',
                   'FAR coefficient\n(accuracy controlled)')]
FIGURE2_PHI_PANEL = ('Metacognitive sensitivity', 'Phi')
CONDITION_COLORS = ('#4E79A7', '#E15759')          # 4-choice, 8-choice
FIGURE2_XLABEL = r'Bias-correction strength $\alpha$'


def plot_alpha_sweep(results, output_path=None, phi_panel=True, fontsizes=None,
                     figsize=None, colors=CONDITION_COLORS):
    """Figure 2: the alpha sweep, one panel per nested model plus metacognitive sensitivity.

    A  bias-only regression, both conditions
    B  bias + accuracy regression, both conditions
    C  metacognitive sensitivity across alpha

    ``fontsizes`` overrides any of ``title``, ``axis_label``, ``tick``, ``legend``,
    ``panel_letter``.
    """
    import matplotlib.pyplot as plt

    ft = {'title': 15.0, 'axis_label': 13.0, 'tick': 11.5, 'legend': 12.0,
          'panel_letter': 23.0}
    ft.update(fontsizes or {})
    conditions = list(dict.fromkeys(results['condition']))
    panels = list(FIGURE2_PANELS)
    n_panels = len(panels) + (1 if phi_panel else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize or (5.2 * n_panels, 4.6))
    axes = np.atleast_1d(axes)

    for ax, (key, title, ylabel) in zip(axes, panels):
        for i, condition in enumerate(conditions):
            block = results[(results['condition'] == condition)
                            & (results['model_key'] == key)].sort_values('alpha')
            colour = colors[i % len(colors)]
            ax.plot(block['alpha'], block['coef'], '-o', ms=5.5, lw=2.2, color=colour,
                    label=condition)
            ax.fill_between(block['alpha'], block['ci_low'], block['ci_high'],
                            color=colour, alpha=0.18, lw=0)
        ax.axhline(0, color='black', lw=1.0, zorder=1)     # the sign flip matters here
        ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
        ax.set_title(title, fontsize=ft['title'], fontweight='bold', pad=8)

    if phi_panel:
        ax = axes[-1]
        title, ylabel = FIGURE2_PHI_PANEL
        for i, condition in enumerate(conditions):
            block = (results[(results['condition'] == condition) &
                             (results['model_key'] == 'far_acc')].sort_values('alpha'))
            colour = colors[i % len(colors)]
            ax.plot(block['alpha'], block['phi'], '-o', ms=5.5, lw=2.2, color=colour,
                    label=condition)
            lo = block['phi_ci_low'] if 'phi_ci_low' in block else block['phi'] - block['phi_sem']
            hi = block['phi_ci_high'] if 'phi_ci_high' in block else block['phi'] + block['phi_sem']
            ax.fill_between(block['alpha'], lo, hi, color=colour, alpha=0.18, lw=0)
        ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
        ax.set_title(title, fontsize=ft['title'], fontweight='bold', pad=8)

    for ax in axes:
        ax.set_xlabel(FIGURE2_XLABEL, fontsize=ft['axis_label'], fontweight='bold')
        ax.tick_params(labelsize=ft['tick'])
        ax.grid(axis='y', linestyle='--', alpha=0.30)
        ax.legend(frameon=False, fontsize=ft['legend'])

    fig.tight_layout()
    for ax, letter in zip(axes, 'ABC'):
        pos = ax.get_position()
        fig.text(max(0.002, pos.x0 - 0.048), min(0.999, pos.y1 + 0.075), letter,
                 fontsize=ft['panel_letter'], fontweight='bold', ha='left', va='center')
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f'Saved figure -> {output_path}')
    return fig


def print_summary(results):
    """Endpoint summary: the sign flip and the Phi gain that Figure 2 reports."""
    print('\n' + '=' * 78)
    print('Bias-correction sweep: endpoints of the alpha range')
    print('=' * 78)
    for condition in dict.fromkeys(results['condition']):
        sub = results[(results['condition'] == condition) & (results['model_key'] == 'far_acc')]
        lo = sub[sub['alpha'] == sub['alpha'].min()].iloc[0]
        hi = sub[sub['alpha'] == sub['alpha'].max()].iloc[0]
        print(f'\n{condition}  (overall accuracy {lo["overall_accuracy"]:.3f})')
        print(f'  alpha={lo["alpha"]:.1f} (bias-blind): FAR|accuracy beta = {lo["coef"]:+.4f} '
              f'[{lo["ci_low"]:+.4f}, {lo["ci_high"]:+.4f}]   Phi = {lo["phi"]:.3f}')
        print(f'  alpha={hi["alpha"]:.1f} (bias-aware): FAR|accuracy beta = {hi["coef"]:+.4f} '
              f'[{hi["ci_low"]:+.4f}, {hi["ci_high"]:+.4f}]   Phi = {hi["phi"]:.3f}')
        print(f'  sign flip positive -> negative: '
              f'{"yes" if lo["coef"] > 0 > hi["coef"] else "no"}')


def main():
    p = argparse.ArgumentParser(
        description='Multi-alternative SDT simulation of bias-blind vs bias-aware confidence.')
    p.add_argument('--n_subj', type=int, default=None,
                   help=f'Observers per condition. Defaults to {N_SUBJ_SWEEP} for the '
                        f'digit-level sweep (the reported run) and {N_SUBJ} elsewhere.')
    p.add_argument('--n_trial', type=int, default=N_TRIAL)
    p.add_argument('--conf_noise', type=float, default=CONF_NOISE,
                   help='Late (readout) metacognitive noise SD.')
    p.add_argument('--target_acc', type=float, default=TARGET_ACC)
    p.add_argument('--k_list', type=int, nargs='+', default=[4, 8])
    p.add_argument('--alpha_step', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None,
                   help='Defaults to the reported seed for the chosen analysis.')
    p.add_argument('--trial-level', dest='trial_level', action='store_true',
                   help='Run the trial-level sweep instead of the digit-level one.')
    p.add_argument('--csv', type=str, default=None, help='Write the tidy results to CSV.')
    p.add_argument('--output', type=str, default=None, help='Save the figure (PDF/PNG).')
    p.add_argument('--no-bias-benchmark', dest='no_bias', action='store_true',
                   help='Run the no-bias null benchmark instead of the alpha sweep.')
    p.add_argument('--calibrate', action='store_true',
                   help='Re-solve sigma_b against the empirical FAR dispersion (slow).')
    args = p.parse_args()
    default_n_subj = N_SUBJ if (args.trial_level or args.calibrate) else N_SUBJ_SWEEP
    n_subj = default_n_subj if args.n_subj is None else args.n_subj

    if args.calibrate:
        return {K: calibrate_sigma_b(K, n_subj=n_subj, n_trial=args.n_trial,
                                     target_acc=args.target_acc)
                for K in args.k_list}
    if args.no_bias:
        summaries = [no_bias_benchmark(K, n_trial=args.n_trial,
                                       target_acc=args.target_acc)[1]
                     for K in args.k_list]
        out = pd.DataFrame(summaries)
        if args.csv:
            out.to_csv(args.csv, index=False)
            print(f'Saved results -> {args.csv}')
        return out

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 3)
    runner = run_trial_level_alpha_sweep if args.trial_level else run_alpha_sweep
    seed = args.seed if args.seed is not None else (
        SEED_TRIAL_LEVEL if args.trial_level else SEED_GENERATION)
    results = runner(k_list=tuple(args.k_list), n_subj=n_subj, n_trial=args.n_trial,
                     conf_noise=args.conf_noise, target_acc=args.target_acc,
                     alphas=alphas, seed=seed)

    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f'Saved results -> {args.csv}')
    if args.trial_level:
        if args.output:
            print('(--output is only produced for the digit-level sweep; '
                  'use --csv for the trial-level results)')
    else:
        print_summary(results)
        if args.output:
            plot_alpha_sweep(results, args.output)
    return results


if __name__ == '__main__':
    main()
