"""
aggregate.py
Bridge helpers that turn preprocessed behavioral CSVs and per-image ANN CSVs into
the z-scored, digit-level DataFrames consumed by regression.py / mediation.py.

Every builder returns a long DataFrame with (at least):
    subject_idx        integer grouping index (participant, or ANN instance)
    FAR_z              z-scored response bias (false-alarm rate)
    accuracy_z         z-scored accuracy
    confidence_z       z-scored confidence
    rt_z               z-scored RT (human data only)
Variables are z-scored globally (across all subject x digit rows), matching the
analysis notebooks.
"""

import numpy as np
import pandas as pd

from scripts_analysis.metrics import compute_sdt_params


def _zscore(series):
    s = series.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def build_human_frame(agg_df):
    """Build the regression frame from a ``preprocess.py`` output (one row per
    subject x digit, columns Sub_id, Digit, FAR, Accuracy, Confidence, RT).

    Returns a frame with subject_idx and FAR_z / accuracy_z / confidence_z /
    (rt_z if available). Use regressors ['FAR_z', 'accuracy_z', 'rt_z'].
    """
    df = agg_df.dropna(subset=['FAR', 'Accuracy', 'Confidence']).copy()
    df['subject_idx'] = pd.Categorical(df['Sub_id']).codes
    df['FAR_z'] = _zscore(df['FAR'])
    df['accuracy_z'] = _zscore(df['Accuracy'])
    df['confidence_z'] = _zscore(df['Confidence'])
    if 'RT' in df.columns and df['RT'].notna().any():
        df['rt_z'] = _zscore(df['RT'])
    return df


def build_combined_exp2_frame(accuracy_agg, speed_agg,
                              accuracy_label='Accuracy', speed_label='Speed'):
    """Combine the two Experiment-2 conditions into one frame for the joint
    (moderated) analysis.

    ``accuracy_agg`` / ``speed_agg`` are ``preprocess.py`` outputs for the two
    conditions. The returned frame carries a ``Condition`` column and a shared
    ``subject_idx`` (the same participant keeps one index across conditions, so
    random intercepts pool within participant). Variables are z-scored globally
    across the combined data. Pass the frame to
    ``regression.get_moderated_regression`` and ``mediation.run_moderated_mediation``
    (the latter after adding an effect-coded ``Cond_eff`` column via
    ``regression.add_effect_coded_condition``).
    """
    a = accuracy_agg.copy(); a['Condition'] = accuracy_label
    s = speed_agg.copy(); s['Condition'] = speed_label
    df = pd.concat([a, s], ignore_index=True)
    df = df.dropna(subset=['FAR', 'Accuracy', 'Confidence']).copy()
    df['subject_idx'] = pd.Categorical(df['Sub_id']).codes
    df['FAR_z'] = _zscore(df['FAR'])
    df['accuracy_z'] = _zscore(df['Accuracy'])
    df['confidence_z'] = _zscore(df['Confidence'])
    if 'RT' in df.columns and df['RT'].notna().any():
        df['rt_z'] = _zscore(df['RT'])
    return df


def aggregate_ann_csv(csv_path, conf_col='conf_meta', n_digits=10,
                      true_col='true_label', pred_col='pred_label',
                      correct_col='correct', instance_col='instance'):
    """Aggregate a per-image ANN CSV (from test_metacognitive.py or
    test_baseline.py) to the digit level and z-score.

    For each (instance, digit): accuracy and confidence are averaged over trials
    whose TRUE label is that digit; FAR is the one-vs-rest false-alarm rate
    (tendency to respond that digit when it is not the target), computed with the
    Hautus-corrected SDT routine. ``conf_col`` selects the confidence readout:
    'conf_meta' is the learned metacognitive head (Figure 6), 'conf_top2diff' and
    'max_softmax' are the standard, untrained ANN readouts (Figure 5). Because the
    backbone is frozen, FAR and accuracy are identical across readouts.

    Returns a long frame with subject_idx = ANN instance index; use regressors
    ['FAR_z', 'accuracy_z'] (ANNs have no RT).
    """
    raw = pd.read_csv(csv_path)
    # Tolerate the older column names from test_baseline.py.
    if pred_col not in raw.columns and 'prediction' in raw.columns:
        pred_col = 'prediction'
    if conf_col not in raw.columns:
        raise KeyError(f"Confidence column '{conf_col}' not in {csv_path}. "
                       f"Available: {list(raw.columns)}")

    rows = []
    for sub_idx, (inst, d) in enumerate(raw.groupby(instance_col)):
        true = d[true_col].values
        pred = d[pred_col].values
        for digit in range(n_digits):
            target = d[d[true_col] == digit]
            if len(target) == 0:
                continue
            acc = target[correct_col].mean()
            conf = target[conf_col].mean()
            _, _, _, _, far = compute_sdt_params((true == digit).astype(int),
                                                 (pred == digit).astype(int))
            rows.append({'subject_idx': sub_idx, 'instance': inst, 'digit': digit,
                         'accuracy': acc, 'confidence': conf, 'FAR': far})

    df = pd.DataFrame(rows)
    df['FAR_z'] = _zscore(df['FAR'])
    df['accuracy_z'] = _zscore(df['accuracy'])
    df['confidence_z'] = _zscore(df['confidence'])
    return df
