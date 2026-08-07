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

TWO CONVENTIONS THAT DIFFER FROM THE DIGIT-LEVEL PIPELINE
---------------------------------------------------------
FAR uses RAW counts, with no half-count boundary correction. The correction
exists to keep an aggregated per-alternative rate off the 0 and 1 boundaries;
here FAR is only a per-trial predictor and is never itself an outcome, so
correcting it would bias the predictor for no gain. This matches
``simulation.observer_far(..., half_count=False)``.

FAR and confidence are standardised WITHIN participant, not globally. At the
trial level every participant contributes hundreds of rows, so between-participant
differences in how the confidence scale is used would otherwise dominate the
variance and the coefficient would partly describe between-participant
differences rather than the within-participant effect being tested.

Covers the trial-level supplementary figures for humans and for ANNs. The
simulation counterpart lives in ``simulation.run_trial_level_alpha_sweep``.

Usage
-----
    from scripts_analysis.trial_level import human_trial_level, exp2_trial_level

    result = human_trial_level('data/human_data/Experiment1_8_choice.csv',
                               n_alternatives=8)
    print(result['coefficient'], result['p'])

    joint, per_condition = exp2_trial_level(
        'data/human_data/Experiment2_accuracy.csv',
        'data/human_data/Experiment2_speed.csv')
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


def far_by_alternative(stim, resp, n_alternatives, half_count=False):
    """FAR per alternative: P(resp = k | stim != k), from RAW counts by default.

    ``half_count=True`` applies the boundary correction (0 -> 0.5, n -> n - 0.5)
    used by the digit-level pipeline. It is deliberately OFF here: see the module
    docstring. The flag exists so the choice is explicit rather than implied.
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
        if half_count:
            if count == 0.0:
                count = 0.5
            elif count == n_noise:
                count = n_noise - 0.5
        far[k] = count / n_noise
    return far


def build_trial_frame(df, n_alternatives, columns=None, alternatives=None,
                      half_count=False, extra=None):
    """Attach each trial the FAR of the alternative that was chosen on that trial.

    ``alternatives`` maps the raw stimulus/response labels onto 0..K-1. If it is
    None the sorted unique stimulus values are used, so the function works
    whether digits are labelled 0-7, 1-8, or any other consistent scheme.

    ``chosen_FAR`` and ``confidence`` are standardised WITHIN participant (see the
    module docstring). A participant with no variation in either -- one confidence
    rating throughout, say -- contributes no slope and is dropped.

    ``extra`` names further columns to carry through unchanged, which is how the
    Experiment 2 condition code reaches the joint model.
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
    dropped = []
    for subject, block in df.groupby(columns['subject']):
        stim = block[columns['stimulus']].map(lookup).to_numpy()
        resp = block[columns['response']].map(lookup).to_numpy()
        keep = ~(pd.isna(stim) | pd.isna(resp))
        stim, resp = stim[keep].astype(int), resp[keep].astype(int)
        if len(stim) == 0:
            continue
        far = far_by_alternative(stim, resp, n_alternatives, half_count=half_count)
        chosen = far[resp]
        confidence = block[columns['confidence']].to_numpy(dtype=float)[keep]
        far_sd, conf_sd = np.nanstd(chosen, ddof=1), np.nanstd(confidence, ddof=1)
        if not np.isfinite(far_sd) or far_sd == 0 or not np.isfinite(conf_sd) or conf_sd == 0:
            dropped.append(subject)          # no within-participant variation to fit
            continue
        piece = pd.DataFrame({
            'subject': subject,
            'chosen_FAR': chosen,
            'confidence': confidence,
            'correct': block[columns['correct']].to_numpy()[keep],
            'chosen_FAR_z': (chosen - np.nanmean(chosen)) / far_sd,
            'confidence_z': (confidence - np.nanmean(confidence)) / conf_sd,
        })
        for name in (extra or ()):
            piece[name] = block[name].to_numpy()[keep]
        frames.append(piece)

    if dropped:
        warnings.warn(f'{len(dropped)} participant(s) had no within-participant variation '
                      f'in FAR or confidence and were dropped: {dropped}', RuntimeWarning)
    return pd.concat(frames, ignore_index=True).dropna(
        subset=['chosen_FAR_z', 'confidence_z'])


def fit_trial_level(frame, predictor='chosen_FAR_z', outcome='confidence_z',
                    group='subject', formula=None, report=None):
    """Mixed model with a random intercept and a random slope on ``predictor``.

    Same specification and the same optimiser ladder as the digit-level
    regressions, so the two analyses differ only in the unit of analysis.

    ``formula`` overrides the default ``outcome ~ predictor`` -- pass
    ``'confidence_z ~ chosen_FAR_z * Cond_eff'`` for the Experiment 2 joint model.
    ``report`` names the term to return; it defaults to ``predictor``, so ask for
    the interaction term explicitly when that is the test of interest.

    An optimiser is accepted only when EVERY fixed-effect standard error comes
    back finite and positive, not merely the reported term's. A run that has
    converged for one term while another is degenerate can still return a wildly
    inflated interval for the term you asked about.
    """
    formula = formula or f'{outcome} ~ {predictor}'
    report = report or predictor

    def extract(res, structure):
        ci = res.conf_int().loc[report]
        params = res.params
        return {'term': report, 'formula': formula,
                'coefficient': float(params[report]),
                'se': float(getattr(res, 'bse_fe', res.bse)[report]),
                'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
                'z': float(res.tvalues[report]), 'p': float(res.pvalues[report]),
                'n_trials': int(len(frame)),
                'n_subjects': int(frame[group].nunique()),
                're_structure': structure}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = smf.mixedlm(formula, frame, groups=frame[group],
                            re_formula=f'~ {predictor}')
        for method in ('lbfgs', 'powell', 'cg', 'bfgs', 'nm'):
            try:
                res = model.fit(method=method, maxiter=2000)
            except Exception:
                continue
            se = np.asarray(res.bse_fe, dtype=float)
            if np.all(np.isfinite(se)) and np.all(se > 0):
                return extract(res, f'random slope ({method})')
        res = smf.mixedlm(formula, frame, groups=frame[group]).fit(method='lbfgs',
                                                                  maxiter=2000)
        print('    [!] random slope not estimable -> random intercept only; SE understated.')
        return extract(res, 'random intercept only')


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


def exp2_trial_level(accuracy_csv, speed_csv, n_alternatives=8, columns=None,
                     exclude=()):
    """Experiment 2 trial level: the joint condition model AND the two simple fits.

    Two things are reported, and they answer different questions:

    ``joint``  ONE model over both conditions,
               ``confidence_z ~ chosen_FAR_z * Cond_eff`` with effect-coded
               condition (accuracy +0.5, speed -0.5). The interaction term is the
               test of whether the trial-level bias effect differs between speed
               and accuracy focus. This is the reported test.
    ``per_condition``  the same single-predictor model fitted separately within
               each condition, which is what the supplementary figure's two bars
               show. Separate fits cannot themselves test the difference.

    Note that the ``Cond_eff`` MAIN effect in the joint model is structurally
    zero: confidence is standardised within participant and therefore within
    condition, so each condition's mean is 0 by construction. Only the
    interaction is interpretable, and the main effect must not be read as
    evidence that the instruction did not change confidence.

    ``exclude`` is empty by default because the shipped Experiment 2 CSVs are
    ALREADY the post-exclusion sample (60 participants x 480 trials x 2
    conditions); the four participants dropped upstream are not in the files.
    Passing IDs here would exclude them a second time.
    """
    columns = columns or EXP2_COLUMNS
    frames = []
    for label, path, code in (('Accuracy', accuracy_csv, 0.5), ('Speed', speed_csv, -0.5)):
        df = pd.read_csv(path)
        df = df[~df[columns['subject']].isin(list(exclude))].copy()
        frame = build_trial_frame(df, n_alternatives, columns)
        frame['Condition'] = label
        frame['Cond_eff'] = code
        frames.append(frame)
    both = pd.concat(frames, ignore_index=True)

    joint = fit_trial_level(both, formula='confidence_z ~ chosen_FAR_z * Cond_eff',
                            report='chosen_FAR_z:Cond_eff')
    joint['dataset'] = 'Experiment 2 (joint)'
    print(f"Experiment 2 joint model: FAR x condition beta = {joint['coefficient']:+.4f} "
          f"[{joint['ci_low']:+.4f}, {joint['ci_high']:+.4f}], p = {joint['p']:.3g}")

    per_condition = []
    for label, frame in zip(('Accuracy', 'Speed'), frames):
        result = fit_trial_level(frame)
        result['dataset'] = f'Experiment 2 ({label.lower()} focus)'
        result['condition'] = label
        per_condition.append(result)
        print(f"  {label:<8} trial-level FAR beta = {result['coefficient']:+.4f} "
              f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], p = {result['p']:.3g} "
              f"({result['n_trials']} trials, {result['n_subjects']} subjects)")
    return joint, pd.DataFrame(per_condition)


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
