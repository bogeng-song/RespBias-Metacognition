"""
preprocess.py
Converts raw trial-level behavioral CSVs into the per-subject, per-digit aggregated
format required by the regression and mediation analysis pipeline.

For each (subject, digit) combination, computes:
  - FAR       : False Alarm Rate — proportion of non-stimulus trials where digit was chosen
  - Accuracy  : Hit Rate — proportion of stimulus trials answered correctly
  - Confidence: Mean confidence rating on stimulus trials
  - RT        : Mean decision RT on stimulus trials

Outputs a long-format CSV (one row per subject × digit) directly readable by
regression.py and mediation.py. Eliminates the need for external .npy files.

Usage
-----
# Experiment 1 — 8-choice:
python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_8_choice.csv \
    --output data/human_data/Experiment1_8_choice_aggregated.csv \
    --condition 8choice

# Experiment 1 — 4-choice:
python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_4_choice.csv \
    --output data/human_data/Experiment1_4_choice_aggregated.csv \
    --condition 4choice

# Experiment 2 — Accuracy focus:
python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment2_accuracy.csv \
    --output data/human_data/Experiment2_accuracy_aggregated.csv \
    --condition exp2

# Experiment 2 — Speed focus:
python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment2_speed.csv \
    --output data/human_data/Experiment2_speed_aggregated.csv \
    --condition exp2
"""

import argparse
import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Column name normaliser
# Handles the different column naming conventions across the two experiments.
# ──────────────────────────────────────────────────────────────────────────────
EXP1_COL_MAP = {
    'Sub_id':       'Sub_id',
    'Stimulus':     'Stimulus',
    'Response':     'Response',
    'Correct':      'Correct',
    'Confidence':   'Confidence',
    'RT decision':  'RT',
}

EXP2_COL_MAP = {
    'subject':      'Sub_id',
    'stim':         'Stimulus',
    'response':     'Response',
    'correct':      'Correct',
    'confidence':   'Confidence',
    'resp_rt':      'RT',
}


def normalise_columns(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Rename raw columns to a unified schema."""
    col_map = EXP2_COL_MAP if condition == 'exp2' else EXP1_COL_MAP
    rename = {orig: norm for orig, norm in col_map.items() if orig in df.columns}
    return df.rename(columns=rename)


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_digit_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FAR, Accuracy, Confidence, and RT for each (subject, digit) pair.

    FAR for digit d, subject s:
        Proportion of trials where subject s chose digit d on trials where
        the actual stimulus was NOT digit d.

    Accuracy, Confidence, RT for digit d, subject s:
        Means over trials where the actual stimulus WAS digit d.
    """
    records = []
    digits = sorted(df['Stimulus'].unique())

    for sub_id, sub_df in df.groupby('Sub_id'):
        for digit in digits:
            # FAR: response==digit on non-stimulus trials
            non_stim = sub_df[sub_df['Stimulus'] != digit]
            far = (non_stim['Response'] == digit).mean() if len(non_stim) > 0 else np.nan

            # Accuracy, Confidence, RT: stimulus==digit trials
            stim = sub_df[sub_df['Stimulus'] == digit]
            n_stim = len(stim)
            acc  = stim['Correct'].mean()    if n_stim > 0 else np.nan
            conf = stim['Confidence'].mean() if n_stim > 0 else np.nan
            rt   = stim['RT'].mean()         if n_stim > 0 else np.nan

            records.append({
                'Sub_id':     sub_id,
                'Digit':      digit,
                'FAR':        far,
                'Accuracy':   acc,
                'Confidence': conf,
                'RT':         rt,
                'N_stim':     n_stim,
                'N_non_stim': len(non_stim),
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
def validate_output(df_agg: pd.DataFrame) -> None:
    n_subj  = df_agg['Sub_id'].nunique()
    n_digit = df_agg['Digit'].nunique()
    expected_rows = n_subj * n_digit
    actual_rows   = len(df_agg)

    print(f"\n{'─'*55}")
    print(f"  Subjects : {n_subj}")
    print(f"  Digits   : {n_digit}")
    print(f"  Rows     : {actual_rows}  (expected {expected_rows})")

    nan_counts = df_agg[['FAR', 'Accuracy', 'Confidence', 'RT']].isna().sum()
    if nan_counts.any():
        print(f"\n  ⚠️  NaN counts:\n{nan_counts[nan_counts > 0]}")
    else:
        print("  NaN counts : 0  ✅")

    print(f"\n  FAR range        : [{df_agg['FAR'].min():.4f}, {df_agg['FAR'].max():.4f}]")
    print(f"  Accuracy range   : [{df_agg['Accuracy'].min():.4f}, {df_agg['Accuracy'].max():.4f}]")
    print(f"  Confidence range : [{df_agg['Confidence'].min():.4f}, {df_agg['Confidence'].max():.4f}]")
    print(f"{'─'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw trial CSVs to per-subject per-digit aggregated format."
    )
    parser.add_argument('--input',     type=str, required=True,
                        help='Path to the raw trial-level CSV file.')
    parser.add_argument('--output',    type=str, required=True,
                        help='Path for the output aggregated CSV.')
    parser.add_argument('--condition', type=str,
                        choices=['4choice', '8choice', 'exp2'],
                        default='8choice',
                        help='Experiment condition — determines column-name mapping.')
    args = parser.parse_args()

    print(f"\n📂  Loading  : {args.input}")
    df_raw = pd.read_csv(args.input)
    print(f"    Raw shape: {df_raw.shape}")

    df = normalise_columns(df_raw, args.condition)

    required = {'Sub_id', 'Stimulus', 'Response', 'Correct', 'Confidence', 'RT'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"After column normalisation, these columns are still missing: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print("🔄  Computing digit-level metrics …")
    df_agg = compute_digit_level_metrics(df)

    validate_output(df_agg)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_agg.to_csv(args.output, index=False)
    print(f"💾  Saved to : {args.output}\n")


if __name__ == '__main__':
    main()
