"""
collinearity.py
Variance inflation factors for the behavioural and ANN regression models.

FAR and accuracy are computed from the same finite set of trials and are both
driven by the same per-digit response tendency, so they are correlated by
construction. That is exactly why the accuracy-controlled FAR coefficient is the
quantity of interest -- and exactly why the collinearity has to be reported.

Two model families, because they carry different predictor sets:

    behavioural   FAR_z, accuracy_z, rt_z          (three predictors)
    ANN           FAR_z, accuracy_z                (two -- networks have no RT)

With only two predictors both VIFs are necessarily equal and reduce to
1 / (1 - r^2), so ``ann_vif_table`` also reports r; ``vif_table`` asserts the
identity, which catches a mis-specified design matrix.

A further consequence for the ANN table: the metacognitive module trains on a
FROZEN backbone, so it cannot change accuracy or FAR -- only the confidence
readout. VIF depends on the predictors alone, so a given architecture returns
identical diagnostics for every confidence readout. That is a property of the
design, not duplicated output.

Usage
-----
    from scripts_analysis.collinearity import vif_table, ann_vif_table

    print(vif_table(df))                       # behavioural, 3 predictors
    print(ann_vif_table(ann_df))               # ANN, 2 predictors + r
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

BEHAVIOUR_PREDICTORS = ('FAR_z', 'accuracy_z', 'rt_z')
ANN_PREDICTORS = ('FAR_z', 'accuracy_z')


def _condition_number(X):
    """Condition number of the standardised design (excluding the intercept)."""
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    singular_values = np.linalg.svd(Xs, compute_uv=False)
    return float(singular_values.max() / singular_values.min())


def vif_table(df, predictors=BEHAVIOUR_PREDICTORS, label=None):
    """VIF and tolerance per predictor, plus the design's condition number."""
    predictors = list(predictors)
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise ValueError(f'Missing predictor columns: {missing}')
    X = df[predictors].dropna().to_numpy(dtype=float)
    if len(X) < len(predictors) + 2:
        raise ValueError('Too few complete rows to compute VIF.')
    X_const = np.column_stack([np.ones(len(X)), X])

    rows = []
    for j, predictor in enumerate(predictors, start=1):
        vif = float(variance_inflation_factor(X_const, j))
        rows.append({'predictor': predictor, 'VIF': vif, 'tolerance': 1.0 / vif})
    out = pd.DataFrame(rows)
    out['condition_number'] = _condition_number(X)
    out['n_rows'] = len(X)
    if label is not None:
        out.insert(0, 'model', label)
    return out


def ann_vif_table(df, predictors=ANN_PREDICTORS, label=None):
    """Two-predictor VIF table that also reports r and checks VIF = 1 / (1 - r^2)."""
    out = vif_table(df, predictors, label=label)
    X = df[list(predictors)].dropna().to_numpy(dtype=float)
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    out['r_predictors'] = r
    implied = 1.0 / (1.0 - r ** 2)
    if not np.allclose(out['VIF'], implied):
        raise AssertionError(
            f'two-predictor VIF {out["VIF"].tolist()} does not match 1/(1-r^2) = {implied:.6f}; '
            'check the design matrix')
    return out


def summarize_vif(tables, threshold=5.0):
    """Stack per-model VIF tables into one wide summary with a pass/flag column.

    ``threshold`` is the conventional level above which collinearity is usually
    considered problematic; values near 1 indicate essentially none.
    """
    rows = []
    for label, table in tables.items():
        wide = {'model': label,
                'n_rows': int(table['n_rows'].iloc[0]),
                'max_VIF': float(table['VIF'].max()),
                'condition_number': float(table['condition_number'].iloc[0])}
        for _, row in table.iterrows():
            wide[f"VIF_{row['predictor']}"] = float(row['VIF'])
        if 'r_predictors' in table.columns:
            wide['r_predictors'] = float(table['r_predictors'].iloc[0])
        wide['flag'] = 'CHECK' if wide['max_VIF'] >= threshold else 'ok'
        rows.append(wide)
    return pd.DataFrame(rows)
