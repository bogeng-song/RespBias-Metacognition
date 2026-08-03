"""
controls.py
Guessing controls and individual-difference summaries for the behavioural data.

GUESSING CONTROLS
-----------------
The negative accuracy-controlled FAR effect could in principle be an artefact of
trials where the participant was not really deciding -- very fast guesses, very
slow lapses, or simply error trials, on which a biased default response is most
likely. Two controls address that:

``rt_window_sweep``   refits the main regression after trimming trials outside a
                      range of RT cutoffs. If the effect is carried by guesses it
                      should weaken as fast trials are removed.
``correct_only``      refits using correct trials only, where the response cannot
                      be a bias-driven error.

Both report the FAR coefficient under each restriction next to the unrestricted
fit, so the comparison is direct.

INDIVIDUAL DIFFERENCES
----------------------
``far_variability`` quantifies how unevenly each participant distributes their
false alarms across alternatives (the SD of FAR across alternatives) and tests it
against the spread expected under uniform responding. A participant with no
response bias would show only sampling noise.

Usage
-----
    from scripts_analysis.controls import rt_window_sweep, correct_only, far_variability
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def _fit_far(df, formula='confidence_z ~ FAR_z + accuracy_z + rt_z',
             target='FAR_z', group='subject_idx'):
    """Fit the digit-level model and return the target term's estimate."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = smf.mixedlm(formula, df, groups=df[group], re_formula=f'~ {target}')
        for method in ('powell', 'lbfgs', 'cg', 'bfgs', 'nm'):
            try:
                res = model.fit(method=method, maxiter=2000)
            except Exception:
                continue
            se = float(res.bse_fe[target])
            if np.isfinite(se) and se > 0:
                ci = res.conf_int().loc[target]
                return {'coefficient': float(res.params[target]), 'se': se,
                        'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
                        'p': float(res.pvalues[target]),
                        're_structure': f'random slope ({method})'}
        res = smf.mixedlm(formula, df, groups=df[group]).fit(method='lbfgs', maxiter=2000)
        ci = res.conf_int().loc[target]
        return {'coefficient': float(res.params[target]), 'se': float(res.bse_fe[target]),
                'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
                'p': float(res.pvalues[target]), 're_structure': 'random intercept only'}


def rt_window_sweep(trials, aggregate_fn, lower_grid=(0.0, 0.2, 0.3, 0.4, 0.5),
                    upper_grid=(np.inf, 5.0, 4.0, 3.0, 2.0), rt_col='RT decision',
                    formula='confidence_z ~ FAR_z + accuracy_z + rt_z'):
    """Refit the main model across RT cutoffs.

    Parameters
    ----------
    trials : raw trial-level DataFrame.
    aggregate_fn : callable mapping a trial-level frame to the digit-level,
        z-scored regression frame (normally a partial of ``preprocess`` +
        ``aggregate.build_human_frame``), so this function stays agnostic to the
        dataset's column naming.
    lower_grid, upper_grid : RT cutoffs in seconds; trials outside [lo, hi] are
        dropped before aggregation. Aggregation happens AFTER trimming so FAR and
        accuracy are recomputed from the retained trials, not merely re-filtered.
    """
    rows = []
    for lo in lower_grid:
        for hi in upper_grid:
            keep = trials[(trials[rt_col] >= lo) & (trials[rt_col] <= hi)]
            if keep.empty:
                continue
            frame = aggregate_fn(keep)
            result = _fit_far(frame, formula=formula)
            rows.append({'rt_low': lo, 'rt_high': hi,
                         'n_trials': int(len(keep)),
                         'trials_retained': float(len(keep) / len(trials)),
                         **result})
            print(f'  RT in [{lo:.1f}, {hi if np.isfinite(hi) else "inf"}]: '
                  f'{len(keep):>7d} trials ({100 * len(keep) / len(trials):.1f}%)  '
                  f'FAR beta = {result["coefficient"]:+.4f} '
                  f'[{result["ci_low"]:+.4f}, {result["ci_high"]:+.4f}]')
    return pd.DataFrame(rows)


def correct_only(trials, aggregate_fn, correct_col='Correct',
                 formula='confidence_z ~ FAR_z + accuracy_z + rt_z'):
    """Refit using correct trials only, alongside the all-trials fit.

    On a correct trial the chosen alternative was the true one, so the response
    cannot be a bias-driven error; if the effect survives here it is not carried
    by errors.
    """
    rows = []
    for label, subset in (('all trials', trials),
                          ('correct only', trials[trials[correct_col] == 1])):
        frame = aggregate_fn(subset)
        result = _fit_far(frame, formula=formula)
        rows.append({'subset': label, 'n_trials': int(len(subset)), **result})
        print(f'  {label:<13s} {len(subset):>7d} trials  FAR beta = '
              f'{result["coefficient"]:+.4f} [{result["ci_low"]:+.4f}, {result["ci_high"]:+.4f}]'
              f'  p = {result["p"]:.3g}')
    return pd.DataFrame(rows)


def far_variability(far_matrix, n_alternatives=None, label=''):
    """Per-participant SD of FAR across alternatives, tested against uniform responding.

    ``far_matrix`` is (n_participants x n_alternatives). Under uniform responding
    a participant's FAR would be identical across alternatives up to sampling
    noise, so the observed SD is compared with zero by a one-sample t-test; the
    expected-under-uniform reference is reported alongside for context.
    """
    far = np.asarray(far_matrix, dtype=float)
    if n_alternatives is None:
        n_alternatives = far.shape[1]
    spread = np.nanstd(far, axis=1, ddof=1)
    t, p = stats.ttest_1samp(spread, 0.0)
    uniform_far = 1.0 / n_alternatives
    summary = {
        'dataset': label,
        'n_participants': int(far.shape[0]),
        'n_alternatives': int(n_alternatives),
        'mean_sd_far': float(np.nanmean(spread)),
        'sem_sd_far': float(np.nanstd(spread, ddof=1) / np.sqrt(len(spread))),
        'uniform_far': uniform_far,
        't': float(t), 'df': int(len(spread) - 1), 'p': float(p),
        'cohen_d': float(np.nanmean(spread) / np.nanstd(spread, ddof=1)),
    }
    print(f'{label}: mean SD of FAR across alternatives = {summary["mean_sd_far"]:.4f} '
          f'+/- {summary["sem_sd_far"]:.4f} (SEM), t({summary["df"]}) = {t:.2f}, p = {p:.3g}')
    return summary, spread
