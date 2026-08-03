"""
metacognitive_sensitivity.py
Within-instance confidence-accuracy correlation (Phi) for the ANNs.

Phi is the point-biserial correlation between trial-level confidence and
trial-level correctness, computed separately for each network instance and then
averaged, with the SEM across instances. Point-biserial r IS Pearson's r when
one variable is binary, so this is a Pearson correlation between confidence and
accuracy; scipy returns the same number either way.

HOW TO READ THE RESULT
----------------------
The readouts being compared are not on equal footing, and the figure should not
be read as if they were:

    conf_top2diff, max_softmax   FREE readouts of the network's own output
                                 evidence. Nothing was trained to predict
                                 correctness; Phi measures how much correctness
                                 information the classifier's outputs happen to
                                 carry.

    conf_meta                    a SUPERVISED correctness classifier, trained
                                 with binary cross-entropy against
                                 (argmax(logits) == label) at the same noise
                                 level used at test. Reading it back out against
                                 correctness is close to reading out its own
                                 training target.

Evaluation is on held-out data (MNIST train=False; the head is trained on
train=True), so a high Phi for conf_meta is not leakage. But it is near-ceiling
by construction, and it stays above 0.9 even for instances whose own task
accuracy is under 25% -- Phi for the trained head correlates NEGATIVELY with
instance accuracy. The blue-vs-red gap in the figure is therefore a
supervised-versus-unsupervised contrast, not evidence that these networks are
metacognitive. ``phi_accuracy_relationship`` computes that diagnostic.

Usage
-----
    from scripts_analysis.metacognitive_sensitivity import phi_table

    table = phi_table({'AlexNet': 'data/model_data/meta/alexnet_logit_only.csv'},
                      readouts=['conf_top2diff', 'max_softmax', 'conf_meta'])
"""

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def phi_by_instance(df, conf_col, instance_col='instance', correct_col='correct'):
    """Point-biserial (== Pearson) r between confidence and correctness, per instance.

    Returns NaN for an instance whose correctness or confidence has no variance,
    where the correlation is undefined rather than zero.
    """
    values = []
    for _, block in df.groupby(instance_col):
        correct = block[correct_col].to_numpy(dtype=float)
        conf = block[conf_col].to_numpy(dtype=float)
        if len(np.unique(correct)) < 2 or conf.std(ddof=1) == 0:
            values.append(np.nan)
            continue
        values.append(float(pointbiserialr(correct, conf)[0]))
    return np.asarray(values, dtype=float)


def phi_table(csv_by_architecture, readouts=('conf_top2diff', 'max_softmax', 'conf_meta'),
              instance_col='instance', correct_col='correct'):
    """Mean Phi with the SEM across instances, for each architecture x readout.

    ``csv_by_architecture`` maps a label to a per-image CSV from
    ``scripts_ann/test_metacognitive.py``. Each CSV is read once, with every
    requested readout column pulled in the same pass -- these files are large
    (~500k rows).
    """
    rows, per_instance = [], {}
    readouts = list(readouts)
    for architecture, csv_path in csv_by_architecture.items():
        df = pd.read_csv(csv_path, usecols=[instance_col, correct_col] + readouts)
        for readout in readouts:
            values = phi_by_instance(df, readout, instance_col, correct_col)
            per_instance[(architecture, readout)] = values
            finite = values[np.isfinite(values)]
            n = len(finite)
            rows.append({
                'architecture': architecture,
                'readout': readout,
                'supervised_on_correctness': readout == 'conf_meta',
                'phi': float(finite.mean()) if n else np.nan,
                'sem': float(finite.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                'sd_across_instances': float(finite.std(ddof=1)) if n > 1 else np.nan,
                'n_instances': int(n),
            })
            print(f"  {architecture:<10} {readout:<15} Phi = {rows[-1]['phi']:+.3f} "
                  f"+/- {rows[-1]['sem']:.3f} (SEM, n = {n})")
        del df
    return pd.DataFrame(rows), per_instance


def phi_accuracy_relationship(df, conf_col='conf_meta', instance_col='instance',
                              correct_col='correct'):
    """Diagnostic: does Phi track how well the network actually performs?

    A readout that reflects genuine uncertainty should score lower on instances
    that perform worse. A trained correctness classifier need not: it can learn
    the instance's systematic error structure, so it can score HIGHER on a badly
    performing network. Returns per-instance accuracy and Phi plus their
    correlation; a strong negative correlation is the tell.
    """
    accuracy = df.groupby(instance_col)[correct_col].mean().to_numpy(dtype=float)
    phi = phi_by_instance(df, conf_col, instance_col, correct_col)
    keep = np.isfinite(accuracy) & np.isfinite(phi)
    r = float(np.corrcoef(accuracy[keep], phi[keep])[0, 1])
    out = pd.DataFrame({'instance_accuracy': accuracy, 'phi': phi})
    print(f'  corr(instance accuracy, Phi[{conf_col}]) = {r:+.3f}   '
          f'accuracy range [{np.nanmin(accuracy):.3f}, {np.nanmax(accuracy):.3f}]   '
          f'Phi range [{np.nanmin(phi):.3f}, {np.nanmax(phi):.3f}]')
    return out, r
