"""
trial_level.py
Trial-level counterpart to the digit-level analyses.

The main analyses summarise each participant (or ANN instance) as one row per
alternative: FAR, accuracy and mean confidence per digit. That aggregation is
what makes "confidence controlling for accuracy" estimable, but it also throws
away the trial structure. These functions keep every trial:

    for each trial, take the FAR of the CHOSEN alternative -- computed from that
    participant's own overall response frequencies -- and use it to predict that
    trial's confidence.

A negative trial-level FAR coefficient means a participant gave lower confidence
on trials where they picked an alternative they generally over-select. Because
there is only one predictor, no accuracy control is possible here, so the
trial-level result is a complement to the digit-level analysis rather than a
replacement: it shows the effect survives without aggregation.

Covers the trial-level supplementary figures for humans and for ANNs. The
simulation counterpart lives in ``simulation.run_trial_level_alpha_sweep``.

Usage
-----
    from scripts_analysis.trial_level import human_trial_level, ann_trial_level

    result = human_trial_level('data/human_data/Experiment1_8_choice.csv',
                               n_alternatives=8)
    print(result['coefficient'], result['p'])
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Column naming per dataset family (the OSF schema).
EXP1_COLUMNS = {'subject': 'Sub_id', 'stimulus': 'Stimulus', 'response': 'Response',
                'correct': 'Correct', 'confidence': 'Confidence'}
EXP2_COLUMNS = {'subject': 'subject', 'stimulus': 'stim', 'response': 'response',
                'correct': 'correct', 'confidence': 'confidence'}


def far_by_alternative(stim, resp, n_alternatives):
    """FAR per alternative: P(resp = k | stim != k), with the half-count correction.

    The half-count correction (0 -> 0.5, n -> n - 0.5) keeps FAR away from the
    0 and 1 boundaries, matching the digit-level pipeline.
    """
    stim = np.asarray(stim)
    resp = np.asarray(resp)
    far = np.full(n_alternatives, np.nan)
    for k in range(n_alternatives):
        notk = stim != k
        n_noise = float(notk.sum())
        if n_noise == 0:
            continue
        count = float(((resp == k) & notk).sum())
        if count == 0.0:
            count = 0.5
        elif count == n_noise:
            count = n_noise - 0.5
        far[k] = count / n_noise
    return far


def build_trial_frame(df, n_alternatives, columns=None, alternatives=None):
    """Attach each trial the FAR of the alternative that was chosen on that trial.

    ``alternatives`` maps the raw stimulus/response labels onto 0..K-1. If it is
    None the sorted unique stimulus values are used, so the function works
    whether digits are labelled 0-7, 1-8, or any other consistent scheme.
    """
    columns = columns or EXP1_COLUMNS
    missing = [c for c in columns.values() if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns {missing}; got {list(df.columns)}')

    if alternatives is None:
        alternatives = np.sort(df[columns['stimulus']].dropna().unique())
    lookup = {value: index for index, value in enumerate(alternatives)}
    if len(lookup) != n_alternatives:
        raise ValueError(f'Found {len(lookup)} alternatives, expected {n_alternatives}')

    frames = []
    for subject, block in df.groupby(columns['subject']):
        stim = block[columns['stimulus']].map(lookup).to_numpy()
        resp = block[columns['response']].map(lookup).to_numpy()
        keep = ~(pd.isna(stim) | pd.isna(resp))
        stim, resp = stim[keep].astype(int), resp[keep].astype(int)
        if len(stim) == 0:
            continue
        far = far_by_alternative(stim, resp, n_alternatives)
        frames.append(pd.DataFrame({
            'subject': subject,
            'chosen_FAR': far[resp],
            'confidence': block[columns['confidence']].to_numpy()[keep],
            'correct': block[columns['correct']].to_numpy()[keep],
        }))
    out = pd.concat(frames, ignore_index=True).dropna(subset=['chosen_FAR', 'confidence'])
    for column in ('chosen_FAR', 'confidence'):
        sd = out[column].std(ddof=0)
        out[column + '_z'] = (out[column] - out[column].mean()) / sd if sd > 0 else 0.0
    return out


def fit_trial_level(frame, predictor='chosen_FAR_z', outcome='confidence_z',
                    group='subject'):
    """Mixed model ``outcome ~ predictor`` with a random intercept and random slope.

    Same specification and the same optimiser ladder as the digit-level
    regressions, so the two analyses differ only in the unit of analysis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = smf.mixedlm(f'{outcome} ~ {predictor}', frame, groups=frame[group],
                            re_formula=f'~ {predictor}')
        for method in ('powell', 'lbfgs', 'cg', 'bfgs', 'nm'):
            try:
                res = model.fit(method=method, maxiter=2000)
            except Exception:
                continue
            se = float(res.bse_fe[predictor])
            if np.isfinite(se) and se > 0:
                ci = res.conf_int().loc[predictor]
                return {'coefficient': float(res.params[predictor]), 'se': se,
                        'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
                        'z': float(res.tvalues[predictor]),
                        'p': float(res.pvalues[predictor]),
                        'n_trials': int(len(frame)),
                        'n_subjects': int(frame[group].nunique()),
                        're_structure': f'random slope ({method})'}
        res = smf.mixedlm(f'{outcome} ~ {predictor}', frame,
                          groups=frame[group]).fit(method='lbfgs', maxiter=2000)
        ci = res.conf_int().loc[predictor]
        print('    [!] random slope not estimable -> random intercept only; SE understated.')
        return {'coefficient': float(res.params[predictor]), 'se': float(res.bse_fe[predictor]),
                'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
                'z': float(res.tvalues[predictor]), 'p': float(res.pvalues[predictor]),
                'n_trials': int(len(frame)), 'n_subjects': int(frame[group].nunique()),
                're_structure': 'random intercept only'}


def human_trial_level(csv_path, n_alternatives, columns=None, label=None):
    """Trial-level FAR -> confidence for one human dataset."""
    columns = columns or EXP1_COLUMNS
    frame = build_trial_frame(pd.read_csv(csv_path), n_alternatives, columns)
    result = fit_trial_level(frame)
    result['dataset'] = label or str(csv_path)
    print(f"{result['dataset']}: trial-level FAR beta = {result['coefficient']:+.4f} "
          f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], p = {result['p']:.3g} "
          f"({result['n_trials']} trials, {result['n_subjects']} subjects)")
    return result


def ann_trial_level(csv_path, conf_col='conf_meta', n_alternatives=10, label=None):
    """Trial-level FAR -> confidence for one ANN result CSV.

    ``csv_path`` is the per-image CSV written by ``scripts_ann/test_metacognitive.py``
    (or ``test_baseline.py``). ``conf_col`` selects the readout:

        conf_top2diff / max_softmax  standard, untrained ANN confidence
        conf_meta                    the learned metacognitive head

    Network instances play the role of participants.
    """
    df = pd.read_csv(csv_path, usecols=['instance', 'true_label', 'pred_label', conf_col])
    frame = build_trial_frame(
        df.rename(columns={'instance': 'Sub_id', 'true_label': 'Stimulus',
                           'pred_label': 'Response', conf_col: 'Confidence'})
          .assign(Correct=lambda d: (d['Stimulus'] == d['Response']).astype(int)),
        n_alternatives)
    result = fit_trial_level(frame)
    result['dataset'] = label or f'{csv_path} [{conf_col}]'
    result['conf_column'] = conf_col
    print(f"{result['dataset']}: trial-level FAR beta = {result['coefficient']:+.4f} "
          f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], p = {result['p']:.3g}")
    return result
