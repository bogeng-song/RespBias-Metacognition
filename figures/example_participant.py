"""
example_participant.py
The four relationships that Figure 3 panel A shows for a single participant.

Every group-level result in the paper rests on the same four quantities computed
per response alternative. Showing them for one person makes the regression and
the mediation that follow readable: the reader can see the points the model is
fitted to before seeing the coefficients.

    confidence -> accuracy     confidence tracks performance at all
    bias -> accuracy           over-selected alternatives are answered worse
    bias -> confidence         over-selected alternatives are rated MORE confident
    bias -> confidence residual   ... and still are, with accuracy partialled out

The last column is the paper's claim in one panel: the raw bias-confidence slope
could be explained away by accuracy, and it is not.

CONVENTIONS
-----------
Everything is stimulus-keyed, matching the group arrays. FAR carries the
one-vs-rest Hautus (half-count) correction, because these ARE alternative-level
rates -- unlike the trial-level analyses, which use raw counts. The residual
fits raw ``confidence ~ accuracy + FAR`` and subtracts only the accuracy
component, leaving confidence with accuracy partialled out rather than with the
bias effect removed along with it.

Usage
-----
    from figures.example_participant import example_participant_arrays

    arrays = example_participant_arrays(
        pd.read_csv('data/human_data/Experiment1_8_choice.csv'),
        subject_id=12, alternatives=[1, 2, 3, 4, 5, 6, 7, 8])

Rendering is done by ``figures.main_figures.figure3_humans(..., example=...)``;
this module only computes.
"""

import numpy as np
from scipy.stats import norm

EXP1_COLUMNS = {'subject': 'Sub_id', 'stimulus': 'Stimulus', 'response': 'Response',
                'correct': 'Correct', 'confidence': 'Confidence'}

# The participant shown in the manuscript.
DEFAULT_SUBJECT = 12

# Column headers and axis labels, in the order the panels appear.
COLUMN_TITLES = ['Confidence-accuracy\nrelationship',
                 'Bias-accuracy\nrelationship',
                 'Bias-confidence\nrelationship',
                 'Bias-confidence\nrelationship\n(accuracy accounted for)']
X_LABELS = ['Confidence', 'Bias', 'Bias', 'Bias']
Y_LABELS = ['Accuracy', 'Accuracy', 'Confidence',
            'Confidence residual\n(accuracy controlled)']


def one_vs_rest_far(is_stimulus, is_response):
    """FAR for one alternative against all others, with the Hautus correction.

    A count of 0 becomes 0.5 and a count of n becomes n - 0.5, which keeps the
    rate off the boundaries where the normal-deviate transform is undefined.
    """
    is_stimulus = np.asarray(is_stimulus) > 0
    is_response = np.asarray(is_response) > 0
    n_noise = np.sum(~is_stimulus)
    false_alarms = np.sum((~is_stimulus) & is_response)
    if false_alarms == n_noise:
        false_alarms = n_noise - 0.5
    elif false_alarms == 0:
        false_alarms = 0.5
    return float(np.clip(false_alarms / n_noise, 1e-9, 1 - 1e-9))


def sdt_parameters(is_stimulus, is_response):
    """(d', criterion, beta, HR, FAR) for one alternative against all others."""
    is_stimulus = np.asarray(is_stimulus) > 0
    is_response = np.asarray(is_response) > 0
    n_signal = np.sum(is_stimulus)
    hits = np.sum(is_stimulus & is_response)
    if hits == n_signal:
        hits = n_signal - 0.5
    elif hits == 0:
        hits = 0.5
    hit_rate = float(np.clip(hits / n_signal, 1e-9, 1 - 1e-9))
    far = one_vs_rest_far(is_stimulus, is_response)
    dprime = norm.ppf(hit_rate) - norm.ppf(far)
    criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(far))
    return dprime, criterion, float(np.exp(dprime * criterion)), hit_rate, far


def example_participant_arrays(trials, subject_id=DEFAULT_SUBJECT, alternatives=None,
                               columns=None):
    """Per-alternative accuracy, confidence, FAR and confidence residual, one participant.

    Returns a dict with ``accuracy``, ``confidence``, ``far``, ``residual`` and
    ``alternatives``; ``figure3_humans`` reads exactly those keys.
    """
    columns = columns or EXP1_COLUMNS
    block = trials[trials[columns['subject']] == subject_id]
    if block.empty:
        raise ValueError(f'Participant {subject_id} not found in this dataset.')
    if alternatives is None:
        alternatives = np.sort(trials[columns['stimulus']].dropna().unique())

    stim = block[columns['stimulus']].to_numpy()
    resp = block[columns['response']].to_numpy()
    accuracy, confidence, far = [], [], []
    for alternative in alternatives:
        rows = block[block[columns['stimulus']] == alternative]
        accuracy.append(rows[columns['correct']].mean())
        confidence.append(rows[columns['confidence']].mean())
        far.append(sdt_parameters(stim == alternative, resp == alternative)[4])

    accuracy = np.asarray(accuracy, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    far = np.asarray(far, dtype=float)
    design = np.column_stack([np.ones(len(accuracy)), accuracy, far])
    beta_accuracy = np.linalg.lstsq(design, confidence, rcond=None)[0][1]
    return {'accuracy': accuracy, 'confidence': confidence, 'far': far,
            'residual': confidence - beta_accuracy * accuracy,
            'alternatives': list(alternatives), 'subject_id': subject_id}


def example_panel_series(arrays):
    """The four (x, y) pairs of panel A, in column order."""
    return [(arrays['confidence'], arrays['accuracy']),
            (arrays['far'], arrays['accuracy']),
            (arrays['far'], arrays['confidence']),
            (arrays['far'], arrays['residual'])]
