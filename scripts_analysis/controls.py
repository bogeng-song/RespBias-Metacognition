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

MANIPULATION CHECK
------------------
``rt_manipulation_check`` confirms that the Experiment 2 speed/accuracy
instruction did what it was meant to: it compares each participant's mean
decision RT and mean confidence RT between the two blocks. This is Figure 4
panel A.

INDIVIDUAL DIFFERENCES AND THE BEHAVIOURAL CHECKS (FIGURE 1)
------------------------------------------------------------
``confidence_by_accuracy``   mean confidence on correct versus error trials, per
                             participant, with the paired test. Figure 1 panel B.
``select_example_profiles``  picks strong, shape-diverse FAR profiles to show as
                             examples. Figure 1 panel C.
``far_variability``          the SD of FAR across alternatives per participant,
                             tested against the spread an UNBIASED observer would
                             still show from finite sampling alone. Figure 1
                             panel D.
``individual_differences``   which alternative each participant over- and
                             under-selects, the within-participant FAR range, and
                             the per-alternative FAR distribution. Supplementary
                             Figure 5.

The reference for ``far_variability`` is not zero. Any participant, biased or
not, shows some FAR spread purely because each alternative is sampled a finite
number of times, so zero is not a meaningful null and testing against it would
be trivially significant. The null comes from
``simulation.no_bias_benchmark``, which simulates accuracy- and trial-matched
M-SDT observers with the bias vector fixed at zero.

Usage
-----
    from scripts_analysis.controls import (rt_window_sweep, correct_only,
                                           far_variability, behavioural_checks)
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


def far_matrix_from_trials(trials, alternatives=None, columns=None):
    """Participant x alternative FAR matrix from a raw trial file.

    Raw counts, no boundary correction: with a few hundred trials per participant
    a zero cell essentially never occurs, and everything downstream is a spread
    across alternatives rather than a rate near a boundary.

    Returns ``(far, participant_ids, alternatives)``.
    """
    columns = columns or {'subject': 'Sub_id', 'stimulus': 'Stimulus',
                          'response': 'Response'}
    if alternatives is None:
        alternatives = np.sort(trials[columns['stimulus']].dropna().unique())
    ids = np.asarray(sorted(trials[columns['subject']].unique()))
    far = np.full((len(ids), len(alternatives)), np.nan)
    for row, subject in enumerate(ids):
        block = trials[trials[columns['subject']] == subject]
        stim = block[columns['stimulus']].to_numpy()
        resp = block[columns['response']].to_numpy()
        for col, alternative in enumerate(alternatives):
            notk = stim != alternative
            if notk.any():
                far[row, col] = ((resp == alternative) & notk).sum() / notk.sum()
    return far, ids, np.asarray(alternatives)


def far_variability(far_matrix, null_far_sd, n_alternatives=None, label=''):
    """Per-participant SD of FAR across alternatives, tested against the no-bias null.

    ``far_matrix`` is (n_participants x n_alternatives). ``null_far_sd`` is the
    FAR spread expected from an observer with NO response preference, matched to
    this condition's accuracy and trial count -- take it from
    ``simulation.no_bias_benchmark(K, target_acc=..., n_trial=...)[1]['mean_far_sd']``.

    Because the null is a single value the paired comparison reduces exactly to a
    one-sample t-test of the observed spreads against it.
    """
    far = np.asarray(far_matrix, dtype=float)
    if n_alternatives is None:
        n_alternatives = far.shape[1]
    spread = np.nanstd(far, axis=1, ddof=1)
    null_far_sd = float(null_far_sd)
    t, p = stats.ttest_1samp(spread, null_far_sd)
    difference = spread - null_far_sd
    summary = {
        'dataset': label,
        'n_participants': int(far.shape[0]),
        'n_alternatives': int(n_alternatives),
        'mean_sd_far': float(np.nanmean(spread)),
        'sem_sd_far': float(np.nanstd(spread, ddof=1) / np.sqrt(len(spread))),
        'no_bias_null_sd_far': null_far_sd,
        'mean_excess_over_null': float(np.nanmean(difference)),
        'uniform_far': 1.0 / n_alternatives,
        't': float(t), 'df': int(len(spread) - 1), 'p': float(p),
        'cohen_dz': float(np.nanmean(difference) / np.nanstd(difference, ddof=1)),
    }
    print(f'{label}: mean SD of FAR = {summary["mean_sd_far"]:.4f} '
          f'+/- {summary["sem_sd_far"]:.4f} (SEM) vs no-bias null {null_far_sd:.4f}, '
          f't({summary["df"]}) = {t:.2f}, p = {p:.3g}')
    return summary, spread


def confidence_by_accuracy(trials, columns=None, label=''):
    """Mean confidence on correct versus error trials, per participant (Figure 1 B).

    The most basic check that confidence tracks performance at all. Returns the
    per-participant pair and the paired t-test; participants without both kinds of
    trial cannot contribute a pair and are dropped.
    """
    columns = columns or {'subject': 'Sub_id', 'correct': 'Correct',
                          'confidence': 'Confidence'}
    paired = (trials.groupby([columns['subject'], columns['correct']], observed=True)
                    [columns['confidence']].mean().unstack(columns['correct'])
                    .rename(columns={0: 'error', 1: 'correct'}))
    n_all = len(paired)
    paired = paired[['correct', 'error']].dropna()
    if len(paired) != n_all:
        warnings.warn(f'{n_all - len(paired)} participant(s) lacked correct or error '
                      f'trials and were dropped.', RuntimeWarning)
    t, p = stats.ttest_rel(paired['correct'], paired['error'])
    difference = paired['correct'] - paired['error']
    summary = {'dataset': label, 'n_participants': int(len(paired)),
               'mean_correct': float(paired['correct'].mean()),
               'mean_error': float(paired['error'].mean()),
               'mean_difference': float(difference.mean()),
               't': float(t), 'df': int(len(paired) - 1), 'p': float(p),
               'cohen_dz': float(difference.mean() / difference.std(ddof=1))}
    print(f'{label}: confidence correct {summary["mean_correct"]:.3f} vs error '
          f'{summary["mean_error"]:.3f}, t({summary["df"]}) = {t:.2f}, p = {p:.3g}')
    return summary, paired


def select_example_profiles(far_matrix, participant_ids, requested=None, n_examples=3,
                            quantile=0.75):
    """Pick strong, shape-diverse FAR profiles to show as examples (Figure 1 C).

    Chooses among participants whose FAR spread is in the top quartile, starting
    from the strongest and then repeatedly adding whichever candidate profile is
    farthest in SHAPE from those already chosen (cosine distance on the
    mean-centred profile). Selecting on spread alone would return three
    participants with the same-looking preference; the point of the panel is that
    the shape differs between people.

    ``requested`` pins the choice to specific participant IDs instead, which is
    what to use once the examples in the manuscript are fixed.
    """
    far = np.asarray(far_matrix, dtype=float)
    participant_ids = np.asarray(participant_ids)
    spread = np.nanstd(far, axis=1, ddof=1)
    if requested:
        requested = [int(v) for v in requested]
        unknown = sorted(set(requested) - set(participant_ids.tolist()))
        if unknown:
            raise ValueError(f'Unknown participant IDs in the override: {unknown}')
        return np.array([int(np.flatnonzero(participant_ids == pid)[0])
                         for pid in requested])

    candidates = np.flatnonzero(spread >= float(np.nanquantile(spread, quantile)))
    centred = far[candidates] - far[candidates].mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    unit = centred / np.where(norms > 0, norms, 1.0)
    chosen = [int(np.argmax(spread[candidates]))]
    while len(chosen) < min(n_examples, len(candidates)):
        distance = np.linalg.norm(unit[:, None, :] - unit[np.asarray(chosen)][None, :, :],
                                  axis=2).min(axis=1)
        distance[chosen] = -np.inf
        chosen.append(int(np.argmax(distance)))
    return candidates[np.asarray(chosen)]


def individual_differences(far_matrix, alternatives, label=''):
    """How response bias varies from participant to participant (Supp. Figure 5).

    Returns the four things the supplementary figure shows:

        most_chosen / least_chosen  % of participants whose highest / lowest FAR
                                    falls on each alternative, against the 100/K
                                    that uniform preference would give
        far_range                   per-participant max FAR minus min FAR
        per_alternative             the FAR column for each alternative, for the
                                    box plot, against the 1/K uniform reference
    """
    far = np.asarray(far_matrix, dtype=float)
    K = far.shape[1]
    most, least = np.nanargmax(far, axis=1), np.nanargmin(far, axis=1)
    counts = lambda idx: 100.0 * np.bincount(idx, minlength=K) / len(idx)
    far_range = np.nanmax(far, axis=1) - np.nanmin(far, axis=1)
    out = {
        'dataset': label,
        'alternatives': list(alternatives),
        'n_participants': int(far.shape[0]),
        'most_chosen_pct': counts(most),
        'least_chosen_pct': counts(least),
        'chance_pct': 100.0 / K,
        'far_range': far_range,
        'mean_far_range': float(np.nanmean(far_range)),
        'median_far_range': float(np.nanmedian(far_range)),
        'per_alternative': [far[:, j] for j in range(K)],
        'uniform_far': 1.0 / K,
    }
    print(f'{label}: FAR range across alternatives, mean {out["mean_far_range"]:.3f} / '
          f'median {out["median_far_range"]:.3f} (uniform FAR would be {1.0 / K:.3f})')
    return out


def rt_manipulation_check(accuracy_csv, speed_csv, columns=(
        ('resp_rt', 'Decision RT', 'RT (s)'),
        ('conf_rt', 'Confidence RT', 'cRT (s)')), subject_col='subject'):
    """Did the Experiment 2 instruction change RT? (Figure 4 panel A.)

    Returns ``{title: {'a': accuracy means, 'b': speed means, 'p': ..., 'ylabel': ...}}``,
    the shape ``figures.main_figures.figure4_speed_accuracy`` expects for its
    ``'rt'`` entry. One row per participant per condition, paired across the two
    blocks the same people completed.
    """
    files = {'Accuracy': accuracy_csv, 'Speed': speed_csv}
    out = {}
    for column, title, ylabel in columns:
        means = {label: pd.read_csv(path).groupby(subject_col)[column].mean()
                 for label, path in files.items()}
        subjects = sorted(set(means['Accuracy'].index) & set(means['Speed'].index))
        a = means['Accuracy'].loc[subjects].to_numpy(dtype=float)
        b = means['Speed'].loc[subjects].to_numpy(dtype=float)
        t, p = stats.ttest_rel(a, b)
        difference = a - b
        out[title] = {'a': a, 'b': b, 'ylabel': ylabel, 'p': float(p), 't': float(t),
                      'df': len(a) - 1,
                      'cohen_dz': float(difference.mean() / difference.std(ddof=1))}
        print(f'  {title}: accuracy {a.mean():.3f} s vs speed {b.mean():.3f} s, '
              f't({len(a) - 1}) = {t:.2f}, p = {p:.3g}')
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 driver
# ──────────────────────────────────────────────────────────────────────────────
FIGURE1_CONDITIONS = {
    '4-choice': {'csv': 'data/human_data/Experiment1_4_choice.csv',
                 'alternatives': [5, 6, 7, 8]},
    '8-choice': {'csv': 'data/human_data/Experiment1_8_choice.csv',
                 'alternatives': [1, 2, 3, 4, 5, 6, 7, 8]},
}


def behavioural_checks(conditions=None, n_null_observers=2000, seed=2026,
                       example_ids=None, columns=None):
    """Everything panels B, C and D of Figure 1 need, per condition.

    Runs the three checks and Holm-corrects across all of them together, since
    they are reported as one family. The null costs ``n_null_observers`` simulated
    observers per condition, so the first call takes a minute or so -- cache the
    return value rather than recomputing it while tuning the figure.

    Returns ``{condition: {...}}``; feed it straight to
    ``figures.main_figures.figure1_task_and_checks``.
    """
    from statsmodels.stats.multitest import multipletests
    from scripts_analysis.simulation import no_bias_benchmark

    conditions = conditions or FIGURE1_CONDITIONS
    example_ids = example_ids or {}
    columns = columns or {'subject': 'Sub_id', 'stimulus': 'Stimulus',
                          'response': 'Response', 'correct': 'Correct',
                          'confidence': 'Confidence'}
    out = {}
    for index, (label, config) in enumerate(conditions.items()):
        trials = pd.read_csv(config['csv'])
        alternatives = config.get('alternatives')
        far, ids, alternatives = far_matrix_from_trials(trials, alternatives, columns)

        confidence_summary, confidence = confidence_by_accuracy(trials, columns, label)

        # The null is matched to THIS condition: same accuracy, same trials per
        # participant. An unmatched null would be comparing different designs.
        accuracy = float(trials[columns['correct']].mean())
        n_trial = int(round(trials.groupby(columns['subject']).size().median()))
        _, null = no_bias_benchmark(len(alternatives), n_observers=n_null_observers,
                                    n_trial=n_trial, target_acc=accuracy,
                                    seed=seed + index, verbose=False)
        spread_summary, spread = far_variability(far, null['mean_far_sd'],
                                                 label=label)

        out[label] = {
            'alternatives': list(alternatives),
            'participant_ids': ids,
            'far': far,
            'confidence': confidence,
            'confidence_test': confidence_summary,
            'far_sd': spread,
            'far_sd_test': spread_summary,
            'null_far_sd': float(null['mean_far_sd']),
            'null': null,
            'n_trial': n_trial,
            'accuracy': accuracy,
            'examples': select_example_profiles(far, ids, example_ids.get(label)),
        }

    tests = [(label, key) for label in out
             for key in ('confidence_test', 'far_sd_test')]
    _, holm, _, _ = multipletests([out[l][k]['p'] for l, k in tests], alpha=0.05,
                                  method='holm')
    for (label, key), value in zip(tests, holm):
        out[label][key]['p_holm'] = float(value)
    return out
