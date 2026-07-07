# Self-monitoring in perceptual decisions by humans and machines

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OSF Data](https://img.shields.io/badge/Data_%26_Weights-OSF-green.svg)](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)

> **Official repository for the manuscript:**
> Song, B., & Rahnev, D. (2026). *Self-monitoring in perceptual decisions by humans and machines*.

---

## Overview

Perceptual confidence is typically conceptualized as monitoring stimulus uncertainty. This project investigates whether humans and Artificial Neural Networks (ANNs) go beyond stimulus uncertainty by engaging in **self-monitoring** of their own decisional tendencies (response biases).

Using multi-alternative (4- and 8-choice) perceptual decision-making tasks, a generative simulation, mixed-effects regressions, and hierarchical Bayesian mediation, we demonstrate:

1. A **generative SDT simulation** shows that a confidence readout that ignores response tendencies produces a *positive* accuracy-controlled bias–confidence relationship, whereas a bias-aware readout produces a *negative* one — and higher metacognitive sensitivity.
2. **Humans** naturally down-weight their confidence when biased toward a specific response (a negative direct effect of bias on confidence), and this effect is attenuated under speed pressure.
3. **Standard ANNs** (AlexNet, ResNet18, VGG19-BN) do *not* self-monitor; response bias inflates their confidence.
4. **Metacognitive ANNs**, augmented with a dedicated confidence readout, reproduce the human-like negative bias–confidence relationship. Four metacognitive modules are provided (see [ANN metacognitive modules](#ann-metacognitive-modules)).

---

## 1. System Requirements

### Software dependencies

| Package | Minimum version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| PyTorch | 2.0 | ANN training and evaluation |
| torchvision | 0.15 | Model architectures |
| pytorch-lightning | 2.0 | Training loop |
| torchmetrics | 1.0 | Accuracy logging |
| PyMC | 5.0 | Bayesian mediation and M-SDT |
| PyTensor | 2.11 | PyMC backend |
| ArviZ | 0.17 | MCMC diagnostics |
| statsmodels | 0.14 | Mixed-effects regression |
| pingouin | 0.5.3 | Repeated-measures correlation |
| pandas | 2.0 | Data manipulation |
| numpy | 1.24 | Numerical computing |
| scipy | 1.10 | Statistics |
| scikit-learn | 1.2 | Preprocessing / AUROC |
| matplotlib | 3.7 | Plotting |
| seaborn | 0.12 | Plotting |
| joblib | 1.3 | Parallel M-SDT fitting |
| tqdm | 4.65 | Progress bars |

All dependencies are listed in `requirements.txt`.

### Operating systems tested

- macOS 13 (Ventura) and 14 (Sonoma) on Intel and Apple Silicon
- Linux (Ubuntu 22.04 LTS)
- Windows 10/11 (native and via WSL2)

### Hardware requirements

- **Simulation** (`scripts_analysis/simulation.py`): Any modern CPU. No GPU, no external data required. Runs in seconds to a couple of minutes.
- **Statistical analysis pipeline** (preprocessing, regression, mediation): Any modern CPU with at least 8 GB RAM. No GPU required.
- **ANN training** (`scripts_ann/train.py`, `scripts_ann/train_metacognitive.py`): A CUDA-compatible GPU is strongly recommended. Tested on NVIDIA RTX 3090 and A100.
- **ANN evaluation** (`test_baseline.py`, `test_metacognitive.py`, `distribution_shift.py`): Benefits from a GPU but runs in reasonable time on CPU.
- Pre-trained weights are provided (see Installation), so GPU hardware is only needed to retrain from scratch.

---

## 2. Installation Guide

**Step 1 — Clone the repository**
```bash
git clone https://github.com/bogeng-song/RespBias-Metacognition.git
cd RespBias-Metacognition
```

**Step 2 — Create the conda environment**

Because PyMC requires specific C-library versions, we recommend conda-forge over a plain pip install:
```bash
conda create -n metacog -c conda-forge python=3.11 pymc pytensor arviz -y
conda activate metacog
pip install -r requirements.txt
```

**Step 3 — Download data and model weights**

The behavioral data and pre-trained ANN weights are hosted externally:

- **[OSF Repository](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)** — human behavioral CSVs
- **[Google Drive](https://drive.google.com/drive/folders/1DlBBSSz3avxgukT13VW2zkMaTNDQ7qdV?usp=sharing)** — pre-trained ANN weights

Place the behavioral CSVs in `data/human_data/` and the ANN base weights in a `weights/` directory at the project root:
```
data/human_data/
    Experiment1_4_choice.csv       # Exp 1, 4-choice condition
    Experiment1_8_choice.csv       # Exp 1, 8-choice condition
    Experiment2_accuracy.csv       # Exp 2, accuracy focus
    Experiment2_speed.csv          # Exp 2, speed focus
```

### Typical install time

| Step | Estimated time |
|---|---|
| git clone | < 1 minute |
| conda create + pip install | 5–15 minutes |
| Behavioral data download (OSF) | 2–5 minutes (~8 MB) |
| ANN weights download | 10–30 minutes (~2 GB) |

---

## 3. Demo

### Demo A — Generative simulation (no GPU, no data, < 2 minutes)

The simulation is fully self-contained and is the quickest way to see the core theoretical result: a bias-blind confidence readout yields a *positive* accuracy-controlled bias coefficient, whereas a bias-aware readout yields a *negative* one.

```bash
python scripts_analysis/simulation.py --output figure1_simulation.pdf
```

**Expected output:** a printed table showing, for both the 4- and 8-choice simulations, a positive accuracy-controlled FAR coefficient for the "raw/blind" readout and a negative one for the "bias-aware" readout, together with higher metacognitive sensitivity (Φ) for the bias-aware readout. A two-panel figure is saved to `figure1_simulation.pdf`.

### Demo B — Human behavioral pipeline (no GPU, < 5 minutes)

```bash
conda activate metacog

python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_8_choice.csv \
    --output data/human_data/Experiment1_8_choice_aggregated.csv \
    --condition 8choice

python notebooks/figure2.py \
    --data4 data/human_data/Experiment1_4_choice.csv \
    --data8 data/human_data/Experiment1_8_choice.csv \
    --output figure3_example_participant.pdf
```

`preprocess.py` prints a validation summary (subject/digit counts, NaN counts, FAR/accuracy/confidence ranges). `figure2.py` produces the example-participant figure (manuscript Figure 3): without accounting for accuracy, response bias positively predicts confidence; after accuracy is controlled, the relationship reverses.

---

## 4. Repository structure

```
RespBias-Metacognition/
    data/
        human_data/            Behavioral CSVs (download from OSF)
        model_data/            ANN output CSVs (generated by evaluation scripts)

    core/                      Shared PyTorch modules
        models.py              Base classifiers: AlexNet, ResNet18, VGG19-BN (1-channel, 10-class)
        meta_modules.py        Architecture-general learned metacognitive heads
                               (fixed_base / chen / dual) on a frozen backbone
        datasets.py            Clean / noisy / balanced MNIST loaders + GPU resize+noise

    scripts_ann/               ANN training and evaluation
        train.py               Train 60 random-seed base instances per architecture
        test_baseline.py       Noise calibration + standard confidence extraction (Fig 6)
        train_metacognitive.py Train the learned metacognitive heads (fixed_base/chen/dual)
        test_metacognitive.py  Evaluate the learned heads -> per-image CSV (Fig 7)
        distribution_shift.py  Feedback-free response-frequency module (Supp 6.3)

    scripts_analysis/          Statistical analysis modules
        simulation.py          Generative SDT simulation (Fig 1 + Supp Fig 1)
        preprocess.py          Raw CSVs -> per-subject x per-digit aggregates
        aggregate.py           Aggregated/ANN CSVs -> z-scored regression frames
        metrics.py             SDT math (d-prime, FAR, Hautus correction)
        msdt_model.py          Multi-Alternative SDT estimation (PyMC)
        regression.py          Mixed-effects regression + joint moderated model (Exp 2)
        mediation.py           Bayesian mediation + joint moderated mediation (Exp 2)

    notebooks/                 Figures and interactive analysis
        analysis_walkthrough.ipynb   End-to-end walkthrough of every analysis
        figure2.py             Example-participant figure (manuscript Fig 3)
        visualization.py       Shared plotting utilities

    requirements.txt
    README.md
```

### Running on your own data

`preprocess.py` accepts any CSV following the OSF column schema. Required columns:

| Column | Description |
|---|---|
| `Sub_id` | Participant identifier (integer) |
| `Stimulus` | Digit shown on that trial |
| `Response` | Digit selected by the participant |
| `Correct` | 1 = correct, 0 = incorrect |
| `Confidence` | Confidence rating (1–4) |
| `RT decision` | Decision reaction time (seconds) |

Use `--condition 4choice` / `8choice` for Experiment 1, or `--condition exp2` for the Experiment 2 column naming (`subject`, `stim`, `response`, `correct`, `confidence`, `resp_rt`).

---

## 5. Reproduction Instructions

### 5.1 Simulation (Figure 1, Supplementary Figure 1)

```bash
# Figure 1 — main simulation (matched at ~63% accuracy):
python scripts_analysis/simulation.py --output figure1_simulation.pdf

# Use the mixed-effects estimator for the headline coefficient (matches the
# empirical regressions) instead of pooled OLS:
python scripts_analysis/simulation.py --mixed --output figure1_simulation.pdf

# Supplementary Figure 1 — robustness grid over response-bias strength x
# metacognitive noise:
python scripts_analysis/simulation.py --sweep --output supp_fig1_sweep.pdf
```

### 5.2 Human behavioral pipeline

**Step 1 — Preprocess all four datasets**
```bash
python scripts_analysis/preprocess.py --input data/human_data/Experiment1_8_choice.csv \
    --output data/human_data/Experiment1_8_choice_aggregated.csv --condition 8choice
python scripts_analysis/preprocess.py --input data/human_data/Experiment1_4_choice.csv \
    --output data/human_data/Experiment1_4_choice_aggregated.csv --condition 4choice
python scripts_analysis/preprocess.py --input data/human_data/Experiment2_accuracy.csv \
    --output data/human_data/Experiment2_accuracy_aggregated.csv --condition exp2
python scripts_analysis/preprocess.py --input data/human_data/Experiment2_speed.csv \
    --output data/human_data/Experiment2_speed_aggregated.csv --condition exp2
```

**Step 2 — Experiment 1 regression and mediation (Figure 4)**

Each Experiment-1 condition is fit separately (the two conditions differ in the number of alternatives). In Python (or via the walkthrough notebook):
```python
import pandas as pd
from scripts_analysis.aggregate import build_human_frame
from scripts_analysis.regression import get_mixed_model_coefficients_random
from scripts_analysis.mediation import run_mixed_effect_mediation, summarize_mediation

df = build_human_frame(pd.read_csv("data/human_data/Experiment1_8_choice_aggregated.csv"))

# Nested mixed-effects regression: confidence ~ FAR (+ accuracy) (+ RT)
coeffs, models = get_mixed_model_coefficients_random(
    df, dependent_var="confidence_z",
    regressors=["FAR_z", "accuracy_z", "rt_z"], target_regressor="FAR_z")

# Bayesian mediation: bias -> accuracy -> confidence
trace = run_mixed_effect_mediation(df, "FAR_z", "accuracy_z", "confidence_z")
print(summarize_mediation(trace))
```

**Step 3 — Experiment 2 joint (moderated) analysis (Figure 5)**

Experiment 2 is analysed with **both** speed-accuracy-tradeoff conditions fit **together**, with condition as an effect-coded moderator, so the Bias × Condition interaction is tested directly:
```python
import pandas as pd
from scripts_analysis.aggregate import build_combined_exp2_frame
from scripts_analysis.regression import get_moderated_regression, add_effect_coded_condition
from scripts_analysis.mediation import run_moderated_mediation, summarize_moderated_mediation

acc = pd.read_csv("data/human_data/Experiment2_accuracy_aggregated.csv")
spd = pd.read_csv("data/human_data/Experiment2_speed_aggregated.csv")
df = build_combined_exp2_frame(acc, spd)                # Condition + shared subject_idx

# Joint moderated mixed-effects regression (Bias x Condition, then + Accuracy, + RT):
mod = get_moderated_regression(df, bias_var="FAR_z", outcome_var="confidence_z")
print(mod["interaction"])       # Bias x Condition interaction across nested models
print(mod["simple_slopes"])     # per-condition simple slopes (accuracy vs. speed focus)

# Joint Bayesian moderated mediation (per-condition direct/indirect paths):
df = add_effect_coded_condition(df)
trace = run_moderated_mediation(df, predictor="FAR_z")
print(summarize_moderated_mediation(trace))
```

**Step 4 — (Optional) M-SDT parameter estimation (Supplementary)**
```bash
python scripts_analysis/msdt_model.py --data_path data/human_data/Experiment1_8_choice.csv \
    --output_path data/human_data/Experiment1_8_choice_MSDT.csv \
    --n_alts 8 --draws 2000 --tune 1000 --chains 4
```
Re-run the Step 2/3 analyses with the M-SDT bias parameter as the predictor to reproduce Supplementary Figures 2–3.

An end-to-end demonstration of every analysis is in `notebooks/analysis_walkthrough.ipynb`.

---

### 5.3 ANN pipeline

**Step 1 — Train base classifiers** (skip if using pre-trained weights)
```bash
for i in $(seq 0 59); do
    python scripts_ann/train.py --model alexnet --instance $i --save_dir ./weights/alexnet/
done
# Repeat with --model resnet18 and --model vgg19
```
Base weights are saved as `<model>-224-<instance>-final.pt`.

**Step 2 — Standard (baseline) ANN behavior → Figure 6**
```bash
python scripts_ann/test_baseline.py --model alexnet \
    --model_dir ./weights/alexnet/ --output_dir ./data/model_data/base/ --target_acc 0.64
```
This calibrates Gaussian noise to human-level accuracy (~64%) and extracts the standard logit-margin confidence. The metacognitive evaluation CSVs (Step 3) also carry a `conf_top2diff` column, which reproduces the same standard-ANN result from a single run.

**Step 3 — Metacognitive modules → Figure 7 and Supplementary**

Architecture-specific noise levels (calibrated to ~64% base accuracy): **AlexNet = 1.05, ResNet18 = 0.24, VGG19 = 0.19**.

*3a. Train the learned heads* (`fixed_base` = main; `chen`, `dual` = supplementary):
```bash
for i in $(seq 0 59); do
    python scripts_ann/train_metacognitive.py --model alexnet --instance $i \
        --mode fixed_base --base_dir ./weights/alexnet/ --save_dir ./weights_meta/alexnet/
done
# Repeat with --mode chen and --mode dual, and for resnet18 / vgg19
```

*3b. Evaluate the learned heads → per-image CSV:*
```bash
python scripts_ann/test_metacognitive.py --model alexnet --mode fixed_base \
    --base_dir ./weights/alexnet/ --meta_dir ./weights_meta/alexnet/ \
    --output_dir ./data/model_data/meta/
```

*3c. Feedback-free distribution-shift module* (no training, no labels):
```bash
python scripts_ann/distribution_shift.py --model alexnet \
    --base_dir ./weights/alexnet/ --output_dir ./data/model_data/distribution_shift/
```

**Step 4 — Analyse ANN behavior**

The ANN CSVs are aggregated to the digit level and analysed with the same regression / mediation functions used for the human data. The confidence column selects the readout:
```python
from scripts_analysis.aggregate import aggregate_ann_csv
from scripts_analysis.regression import get_mixed_model_coefficients_random
from scripts_analysis.mediation import run_mixed_effect_mediation, summarize_mediation

# Standard ANN (Fig 6): use conf_top2diff.  Metacognitive (Fig 7): use conf_meta.
df = aggregate_ann_csv("data/model_data/meta/alexnet_fixed_base.csv", conf_col="conf_meta")
coeffs, models = get_mixed_model_coefficients_random(
    df, dependent_var="confidence_z",
    regressors=["FAR_z", "accuracy_z"], target_regressor="FAR_z")   # ANNs have no RT
trace = run_mixed_effect_mediation(df, "FAR_z", "accuracy_z", "confidence_z")
print(summarize_mediation(trace))
```

#### ANN metacognitive modules

| Module | How to run | Manuscript | Trains a readout? | Uses correctness labels? |
|---|---|---|---|---|
| **Learned correctness head** | `train_metacognitive.py --mode fixed_base` → `test_metacognitive.py --mode fixed_base` | Main (Fig 7) | Yes | Yes |
| **Chen-style white-box** | `--mode chen` | Supp 6.1 | Yes (+ linear probes) | Yes |
| **Dual-output + auxiliary false-response** | `--mode dual` | Supp 6.2 | Yes (two heads) | Yes |
| **Feedback-free distribution-shift** | `distribution_shift.py` | Supp 6.3 | No (calibration only) | No |

All modules keep the base classifier **frozen**, so the perceptual responses, digit-level FAR, and digit-level accuracy are identical across modules; only the confidence readout differs.

---

## Reproducibility Notes

- **Seeded training**: All 60 base ANN instances per architecture use seeds derived deterministically from the instance index, so rerunning `train.py`/`train_metacognitive.py` with the same `--instance` produces the same model.
- **Balanced, reproducible evaluation**: Metacognitive calibration uses a class-balanced MNIST-TRAIN subset (seed 12345) and evaluation uses a class-balanced MNIST-TEST subset (seed 12346). Per-instance Gaussian noise is seeded by the instance index, so evaluation is fully reproducible.
- **Convergence fallback**: Mixed-effects models use a random-slope → random-intercept → cluster-robust OLS fallback; the tier used is printed for each fit.
- **MCMC convergence**: Bayesian models use NUTS with `target_accept=0.95`; convergence is assessed via R-hat.
- **Experiment 2 FAR**: The public Experiment 2 CSVs are pre-split by SAT condition, so `preprocess.py` computes FAR within each file (the practical difference from computing FAR across both conditions is < 0.006 per digit per subject).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. This license is approved by the [Open Source Initiative](https://opensource.org/licenses/MIT).

## Contact

For questions about the code, data, or manuscript, contact **Bogeng Song** at bsong91@gatech.edu or open an issue in this repository.
