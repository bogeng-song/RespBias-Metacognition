# Bias-aware versus bias-blind confidence in humans and machines

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OSF Data](https://img.shields.io/badge/Data_%26_Weights-OSF-green.svg)](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)

> **Official repository for the manuscript:**
> Song, B., & Rahnev, D. (2026). *Bias-aware versus bias-blind confidence in humans and machines*.

---

## Overview

Confidence evaluates the likely accuracy of a current decision. To be maximally informative about accuracy, it should also take account of a decision-maker's broader tendencies — such as a propensity to favour particular alternatives. This work distinguishes:

- **Bias-aware confidence**, which incorporates those response tendencies, from
- **Bias-blind confidence**, which uses only the evidence available on the current trial.

To adjudicate between them, the analyses identify a **signature of bias-aware confidence**: the down-weighting of confidence for alternatives a decision-maker is biased toward. Across 4- and 8-alternative digit classification (N = 200), a generative SDT simulation, mixed-effects regression, and hierarchical Bayesian mediation, the work shows:

1. A **simulation with graded bias correction** establishes the signature to look for. As the confidence readout corrects more strongly for the observer's own response bias, the accuracy-controlled bias→confidence coefficient flips from **positive to negative**, and metacognitive sensitivity rises.
2. **Humans** show the negative signature — they discount confidence for alternatives they over-select — and it is **attenuated under speed pressure**.
3. **Standard ANNs** show the opposite sign: response bias *inflates* their confidence. Their confidence is bias-blind.
4. **ANNs with a metacognitive module** produce bias-aware confidence, reproducing the human-like negative relationship without any change to the perceptual decision itself.

---

## 1. System requirements

### Software

| Package | Minimum | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| PyTorch | 2.0 | ANN training and evaluation |
| torchvision | 0.15 | Model architectures |
| pytorch-lightning | 2.0 | Training loop |
| torchmetrics | 1.0 | Accuracy logging |
| PyMC | 5.0 | Bayesian mediation, M-SDT |
| ArviZ | 0.17 | MCMC diagnostics |
| statsmodels | 0.14 | Mixed-effects regression, VIF |
| pandas / numpy / scipy | 2.0 / 1.24 / 1.10 | Data and statistics |
| scikit-learn | 1.2 | Standardisation |
| matplotlib | 3.7 | Figures |

Full list in `requirements.txt`.

### Tested on

macOS 13–14 (Intel and Apple Silicon), Ubuntu 22.04 LTS, Windows 10/11 (native and WSL2).

### Hardware

- **Simulation** — any modern CPU, no data download, no GPU. Minutes.
- **Statistical pipeline** — any modern CPU, ≥ 8 GB RAM. Bayesian mediation is the slow step (a few minutes per model).
- **ANN training** — a CUDA GPU is strongly recommended (tested on RTX 3090 and A100). Pre-trained weights are provided, so a GPU is only needed to retrain.
- **ANN evaluation** — runs on CPU, faster on GPU.

---

## 2. Installation

```bash
git clone https://github.com/bogeng-song/RespBias-Metacognition.git
cd RespBias-Metacognition
```

PyMC needs specific C libraries, so conda-forge is more reliable than plain pip:

```bash
conda create -n metacog -c conda-forge python=3.11 pymc pytensor arviz -y
conda activate metacog
pip install -r requirements.txt
```

**Data and weights** are hosted externally:

- **[OSF](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)** — human behavioural CSVs → `data/human_data/`
- **[Google Drive](https://drive.google.com/drive/folders/1DlBBSSz3avxgukT13VW2zkMaTNDQ7qdV?usp=sharing)** — pre-trained ANN base weights → `weights/`

```
data/human_data/
    Experiment1_4_choice.csv       # Exp 1, 4-choice
    Experiment1_8_choice.csv       # Exp 1, 8-choice
    Experiment2_accuracy.csv       # Exp 2, accuracy focus
    Experiment2_speed.csv          # Exp 2, speed focus
```

| Step | Time |
|---|---|
| clone | < 1 min |
| conda + pip | 5–15 min |
| behavioural data (~8 MB) | 2–5 min |
| ANN weights (~2 GB) | 10–30 min |

---

## 3. Quick start

The simulation needs no data, no weights and no GPU, and it produces the paper's central theoretical claim:

```bash
python scripts_analysis/simulation.py --output figure2_simulation.pdf
```

**Expected output.** For both the 4- and 8-choice conditions, the accuracy-controlled FAR coefficient is **positive at α = 0** (bias-blind) and **negative at α = 1** (bias-aware), while metacognitive sensitivity Φ increases with α. A reduced run (`--n_subj 60 --alpha_step 0.5`, ~2 minutes) gives:

```
4-choice   α=0.0  β = +0.473   Φ = 0.338
           α=1.0  β = -0.900   Φ = 0.411
8-choice   α=0.0  β = +0.608   Φ = 0.387
           α=1.0  β = -0.528   Φ = 0.522
```

Then the human pipeline (~5 minutes, no GPU):

```bash
python scripts_analysis/preprocess.py --input data/human_data/Experiment1_8_choice.csv --output data/human_data/Experiment1_8_choice_aggregated.csv --condition 8choice
```

`notebooks/analysis_walkthrough.ipynb` runs every analysis end to end.

---

## 4. Repository structure

```
RespBias-Metacognition/
    core/                          Shared PyTorch modules
        models.py                  Base classifiers: AlexNet, ResNet18, VGG19-BN (1-channel, 10-class)
        meta_modules.py            The two metacognitive readouts on a frozen backbone
        datasets.py                Balanced MNIST loaders + GPU resize/noise

    scripts_ann/                   ANN training and evaluation
        train.py                   60 random-seed base instances per architecture
        test_baseline.py           Noise calibration + standard confidence readouts
        train_metacognitive.py     Train the metacognitive head (logit_only / pen_only)
        test_metacognitive.py      Evaluate the head -> per-image CSV

    scripts_analysis/              Statistics
        simulation.py              Graded bias-correction simulation (Figure 2)
        preprocess.py              Raw CSVs -> per-subject x per-digit aggregates
        aggregate.py               Aggregated/ANN CSVs -> z-scored regression frames
        metrics.py                 SDT math (d', FAR, Hautus correction)
        msdt_model.py              Multi-alternative SDT estimation (PyMC)
        regression.py              Mixed-effects regression + joint moderated model
        mediation.py               Bayesian mediation, random a/b/c' slopes
        trial_level.py             Trial-level FAR -> confidence (humans, Exp 2 joint, ANNs)
        controls.py                Behavioural checks, guessing controls, individual differences
        collinearity.py            VIF diagnostics (behavioural and ANN)
        metacognitive_sensitivity.py   Within-instance confidence-accuracy correlation

    figures/                       Figure generation
        style.py                   Fonts / layout / colours, shared point convention
        panels.py                  Reusable panel drawers (no panel letters)
        main_figures.py            Figures 1, 3-6, or any single panel standalone
        example_participant.py     The four per-alternative relationships (Figure 3A)

    notebooks/
        analysis_walkthrough.ipynb End-to-end walkthrough
        visualization.py           Plotting helpers
```

### Using your own data

`preprocess.py` accepts any CSV following the OSF schema:

| Column | Description |
|---|---|
| `Sub_id` | Participant identifier |
| `Stimulus` | Digit shown |
| `Response` | Digit selected |
| `Correct` | 1 / 0 |
| `Confidence` | Rating (1–4) |
| `RT decision` | Decision RT (s) |

Use `--condition 4choice` / `8choice` for Experiment 1, or `--condition exp2` for the Experiment 2 naming (`subject`, `stim`, `response`, `correct`, `confidence`, `resp_rt`).

---

## 5. Reproducing the figures

### Figure 1 — Task structure and behavioural checks

Panel A is the task schematic, supplied as artwork. Panels B–D are computed:

```python
from scripts_analysis.controls import behavioural_checks
from figures.main_figures import figure1_task_and_checks

checks = behavioural_checks()                  # ~1 min: the null is 2,000 simulated observers
figure1_task_and_checks(checks, artwork="figures_out/task_schematic.png",
                        out_stem="figures_out/figure1")
```

**B** confidence on correct versus error trials. **C** the FAR profile of three example participants, chosen automatically for a strong and shape-distinct bias. **D** the per-participant spread of FAR across alternatives.

The reference in panel D is **not zero**. Any participant shows some FAR spread purely because each alternative is sampled a finite number of times, so the comparison is against `simulation.no_bias_benchmark` — M-SDT observers with the bias vector fixed at zero, matched to each condition's own accuracy and trial count. p-values across the four tests are Holm-corrected together.

`figure1_task_and_checks` reports the artwork's effective resolution at the rendered panel width and says so if it falls below 300 DPI, because a schematic that reads fine on screen is routinely too coarse for print.

### Figure 2 — Simulation

Corrected evidence is `y = x + (1 - alpha) * b`, where `alpha` is the proportion of the observer's response-bias vector removed before confidence is read out. `alpha = 0` is bias-blind (confidence uses the same biased decision variables that produced the choice); `alpha = 1` is fully bias-aware. Because `b` always enters the decision variable, **the response is held fixed while the confidence computation varies** — choices and accuracy are identical across the whole sweep.

Confidence is the raw evidence margin between the chosen alternative and the strongest non-chosen alternative *in corrected space*. The competitor is reselected at each `alpha`, so its identity can change even though the choice does not. Negative margins are retained: they mark trials where bias correction made a non-chosen alternative stronger than the selected one.

```bash
python scripts_analysis/simulation.py --output figure2_simulation.pdf --csv figure2_simulation.csv
```

**Calibration.** `sigma_b` is solved so the simulated within-observer FAR dispersion matches the empirical value (**.0704** for 4-choice, **.0433** for 8-choice), by bounded Brent root search over [0.001, 1.5] to tolerance 1e-4. At each candidate, `mu` is recalibrated by 40 bisection iterations and the objective is averaged over 10 independently generated full designs. The solved values — **.368** and **.446** — are the shipped defaults, fixed for all subsequent simulations. `mu` is calibrated per condition to .64 overall accuracy, approximating the empirical .63.

```bash
python scripts_analysis/simulation.py --calibrate            # re-solve sigma_b (slow)
python scripts_analysis/simulation.py --no-bias-benchmark    # null FAR dispersion
```

**Reported statistics.** Both nested models carry an observer-specific random intercept and a random slope for FAR. VIF between FAR and accuracy is reported per `alpha`. Metacognitive sensitivity Φ is the per-observer point-biserial correlation between trial confidence and correctness, with 95% CIs from 2,000 **observer-level** bootstrap samples.

**Seeds.** Generation 1, calibration 2026, trial-level analysis 20250722.

### Figures 3 and 4 — Humans

Preprocess all four datasets first:

```bash
python scripts_analysis/preprocess.py --input data/human_data/Experiment1_4_choice.csv --output data/human_data/Experiment1_4_choice_aggregated.csv --condition 4choice
```

**Figure 3 (Experiment 1)** — each condition is fit separately, since they differ in the number of alternatives:

```python
import pandas as pd
from scripts_analysis.aggregate import build_human_frame
from scripts_analysis.regression import get_mixed_model_coefficients_random
from scripts_analysis.mediation import run_mediation, summarize_mediation

df = build_human_frame(pd.read_csv("data/human_data/Experiment1_8_choice_aggregated.csv"))
coeffs, models = get_mixed_model_coefficients_random(
    df, dependent_var="confidence_z",
    regressors=["FAR_z", "accuracy_z", "rt_z"], target_regressor="FAR_z")

trace = run_mediation(df, predictor="FAR_z")
print(summarize_mediation(trace))
```

Panel A shows the same four quantities for a single participant, before any group model:

```python
from figures.example_participant import example_participant_arrays
from figures.main_figures import figure3_humans

example = {"8-choice condition": example_participant_arrays(
    pd.read_csv("data/human_data/Experiment1_8_choice.csv"), subject_id=12)}
figure3_humans(results, example=example, out_stem="figures_out/figure3")
```

Supplying `example` makes it the three-panel manuscript figure (A example participant, B regression, C mediation); omitting it gives the two-panel form.

**Figure 4 (Experiment 2)** — panel A is the manipulation check, `controls.rt_manipulation_check`, which confirms the instruction moved decision RT and confidence RT. Panels B and C fit both speed–accuracy conditions **together**, with condition effect-coded, so the Bias × Condition interaction and the accuracy-versus-speed difference in the direct effect each get their own estimate:

```python
from scripts_analysis.aggregate import build_combined_exp2_frame
from scripts_analysis.regression import get_moderated_regression, add_effect_coded_condition
from scripts_analysis.mediation import run_moderated_mediation, summarize_moderated_mediation

df = add_effect_coded_condition(build_combined_exp2_frame(acc, spd))
print(get_moderated_regression(df, bias_var="FAR_z")["interaction"])

trace = run_moderated_mediation(df, predictor="FAR_z")
print(summarize_moderated_mediation(trace))   # mod_direct = accuracy - speed
```

### Figures 5 and 6 — ANNs

**Step 1 — base classifiers** (skip if using pre-trained weights):

```bash
for i in $(seq 0 59); do python scripts_ann/train.py --model alexnet --instance $i --save_dir ./weights/alexnet/; done
```

**Step 2 — metacognitive head.** Architecture-specific noise levels, calibrated to ~64% base accuracy: **AlexNet 1.05, ResNet18 0.24, VGG19 0.19**.

```bash
for i in $(seq 0 59); do python scripts_ann/train_metacognitive.py --model alexnet --instance $i --mode logit_only --base_dir ./weights/alexnet/ --save_dir ./weights_meta/alexnet/; done
```

**Step 3 — evaluate → per-image CSV:**

```bash
python scripts_ann/test_metacognitive.py --model alexnet --mode logit_only --base_dir ./weights/alexnet/ --meta_dir ./weights_meta/alexnet/ --output_dir ./data/model_data/meta/
```

**Step 4 — analyse.** The confidence column selects the readout, and one CSV covers both figures:

```python
from scripts_analysis.aggregate import aggregate_ann_csv
from scripts_analysis.regression import get_mixed_model_coefficients_random

# Figure 5 (standard ANN): conf_top2diff.   Figure 6 (metacognitive): conf_meta.
df = aggregate_ann_csv("data/model_data/meta/alexnet_logit_only.csv", conf_col="conf_meta")
coeffs, models = get_mixed_model_coefficients_random(
    df, dependent_var="confidence_z",
    regressors=["FAR_z", "accuracy_z"], target_regressor="FAR_z")   # ANNs have no RT
```

#### The metacognitive modules

| Module | How to run | Head input | Reported in |
|---|---|---|---|
| **Logit-only** | `--mode logit_only` | The C raw final-layer logits | Main text (Figure 6) |
| **Penultimate-only** | `--mode pen_only` | The penultimate representation | Supplement |

Both freeze the base classifier and train only the readout, with the same data, loss, optimiser and schedule. They differ **only** in what feeds the head, which is what makes them a clean comparison. Because the backbone is frozen, the perceptual responses, digit-level FAR and digit-level accuracy are identical across modules — only the confidence changes.

---

## 6. Supplementary analyses

Numbering follows the submitted supplement.

| # | Analysis | Code |
|---|---|---|
| S1 | Trial-level bias signature across correction strength (simulation) | `simulation.run_trial_level_alpha_sweep` |
| S2 | Trial-level bias effect in humans, both experiments | `trial_level.human_trial_level`, `trial_level.exp2_trial_level` |
| S3 | Trial-level bias effect in ANNs, standard and learned readouts | `trial_level.ann_trial_level` |
| S4 | RT-window sweep and correct-trials-only controls | `controls.rt_window_sweep`, `controls.correct_only` |
| S5 | Direct effect c′ across correction strength (simulation mediation) | `simulation.alpha_digit_frames` → `mediation.run_mediation_alpha_sweep` |
| S6 | SoftMax confidence readout | `aggregate.aggregate_ann_csv(conf_col="max_softmax")` |
| S7 | Penultimate-feature metacognitive readout | `train_metacognitive.py --mode pen_only` |
| S8 | Response bias is stable across testing days | `controls.day_split_reliability` |
| Table 1 | Collinearity diagnostics (behavioural and ANN) | `collinearity.vif_table`, `collinearity.ann_vif_table` |
| — | No-bias benchmark: FAR dispersion from finite sampling alone (Figure 1D) | `simulation.no_bias_benchmark` |
| — | ANN metacognitive sensitivity (confidence–accuracy Φ, Figure 6C) | `metacognitive_sensitivity.phi_table` |
| — | M-SDT bias as the predictor instead of FAR | `msdt_model.py`, then rerun the Figure 3/4 analyses |

### Supplementary Figure 8 needs to know which day each trial came from

`day_split_reliability` measures response bias separately in each of the two testing sessions. The shipped Experiment 1 CSVs carry no session column, so `split_by_session` falls back to splitting each participant's rows by order — the first 200 are day 1 — and warns when it does.

That fallback is only correct because the rows are chronological within participant. It was verified against the source file that does carry the session label, and the two agree for all 200 participants in both conditions. **Adding an explicit `Session` column to the OSF CSVs would remove the assumption**; `split_by_session` uses it automatically when present.

### Two conventions worth knowing before reading the code

**FAR boundary correction.** The digit-level analyses and the `sigma_b` calibration apply the half-count correction (0 → 0.5, *n* → *n* − 0.5), matching the behavioural preprocessing. The **trial-level** analyses use raw counts: there FAR is only a per-trial predictor and is never aggregated to the alternative level, so the correction would shift the predictor for no gain. Both `simulation.observer_far` and `trial_level.far_by_alternative` take an explicit `half_count` flag rather than deciding silently.

**Standardisation.** Digit-level frames are z-scored globally; **trial-level** frames are z-scored *within participant*. At the trial level each participant contributes hundreds of rows, so between-participant differences in how the confidence scale is used would otherwise dominate the variance.

One consequence: in `trial_level.exp2_trial_level` the `Cond_eff` **main effect is structurally zero**, because standardising within participant also standardises within condition. Only the interaction is interpretable, and the main effect is not a test of whether the instruction changed confidence — `controls.rt_manipulation_check` is.

### Reading the ANN metacognitive-sensitivity panel

The readouts compared there are not on equal footing, and the code says so in `metacognitive_sensitivity.py`. `conf_top2diff` and `max_softmax` are **free** readouts of the classifier's own outputs; `conf_meta` is a **supervised correctness classifier**, trained with binary cross-entropy against `argmax(logits) == label`. Evaluation is on held-out data (the head trains on MNIST train, evaluation uses MNIST test), so its high Φ is not leakage — but it is near-ceiling by construction, and it stays above 0.9 even for instances whose own task accuracy is under 25%. `phi_accuracy_relationship` reports the diagnostic: Φ for the trained head correlates **negatively** with instance accuracy. The gap between the two families is supervised-versus-unsupervised, not evidence that these networks are metacognitive.

---

## 7. Making the figures

```python
from figures.main_figures import (figure1_task_and_checks, figure3_humans,
                                  figure4_speed_accuracy, figure_ann)

figure3_humans(results, example=example, out_stem="figures_out/figure3")  # whole figure
figure3_humans(results, panel="B", out_stem="figures_out/fig3B")          # one panel, no letter
```

Every panel renders standalone, without its panel letter, so figures can be assembled by hand. Appearance is controlled by three dicts in `figures/style.py` — `FONTS`, `LAYOUT`, `STYLE` — each with a resolver that takes per-key overrides and raises `KeyError` on a typo:

```python
figure_ann(results, phi=phi, font_scale=1.4,
           style={"dot_size": 10, "dot_color": "#666666"},
           layout={"figsize": (15, 17)})
```

Panel A of Figure 1 is the only element not generated from data; everything else, including Figure 1 B–D, comes from the analysis functions above.

---

## Reproducibility notes

- **Seeded training** — all 60 base instances per architecture use seeds derived from the instance index, so re-running `train.py` / `train_metacognitive.py` with the same `--instance` reproduces the model.
- **Balanced, disjoint splits** — the metacognitive head trains on a class-balanced MNIST **train** subset (seed 12345) and is evaluated on a class-balanced MNIST **test** subset (seed 12346). Per-instance noise is seeded by the instance index.
- **Mixed-effects fallback ladder** — random slope (five optimisers) → random intercept → cluster-robust OLS. The tier actually used is printed, because dropping the random slope understates the standard error.
- **Mediation** — random `a`, `b` **and** `c'` slopes with a joint LKJ covariance. With random slopes the population indirect effect is `a*b + Cov(u_a, u_b)`, not `a*b`, and the code carries that term explicitly.
- **MCMC** — NUTS, `target_accept = 0.99`, 4 chains. `mediation.diagnose` flags divergences, R-hat > 1.01, low ESS, low BFMI and tree-depth saturation.
- **Experiment 2 FAR** — the public Experiment 2 CSVs are pre-split by condition, so FAR is computed within each file; the practical difference from computing it across both is < 0.006 per digit per subject.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## Contact

Questions about the code, data or manuscript: **Bogeng Song**, bsong91@gatech.edu, or open an issue.
