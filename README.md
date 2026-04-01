# Self-monitoring in perceptual decisions by humans and machines

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OSF Data](https://img.shields.io/badge/Data_%26_Weights-OSF-green.svg)](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)

> **Official repository for the manuscript:**
> Song, B., & Rahnev, D. (2026). *Self-monitoring in perceptual decisions by humans and machines*.

---

## Overview

Perceptual confidence is typically conceptualized as monitoring stimulus uncertainty. This project investigates whether humans and Artificial Neural Networks (ANNs) go beyond stimulus uncertainty by engaging in **self-monitoring** of their own decisional tendencies (response biases).

Using multi-alternative (4- and 8-choice) perceptual decision-making tasks, mixed-effects regressions, and hierarchical Bayesian mediation, we demonstrate:

1. **Humans** naturally down-weight their confidence when biased toward a specific response (a negative direct effect of bias on confidence).
2. **Standard ANNs** (AlexNet, ResNet18, VGG19-BN) do *not* self-monitor; response bias inflates their confidence.
3. **Metacognitive ANNs**, augmented with a novel unsupervised self-monitoring module, successfully reproduce the human-like negative bias-confidence relationship.

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
| scikit-learn | 1.2 | Preprocessing |
| matplotlib | 3.7 | Plotting |
| seaborn | 0.12 | Plotting |
| wandb | 0.15 | ANN training logging |
| joblib | 1.3 | Parallel M-SDT fitting |
| tqdm | 4.65 | Progress bars |

All dependencies are listed in `requirements.txt`.

### Operating systems tested

- macOS 13 (Ventura) and 14 (Sonoma) on Intel and Apple Silicon
- Linux (Ubuntu 22.04 LTS)
- Windows 10/11 via WSL2

### Hardware requirements

- **Statistical analysis pipeline** (preprocessing, regression, mediation, Figure 2): Any modern CPU with at least 8 GB RAM. No GPU required.
- **ANN training** (`scripts_ann/train.py`): A CUDA-compatible GPU is strongly recommended. Tested on NVIDIA RTX 3090 and A100. Training on CPU is possible but will take many hours per instance rather than minutes.
- **ANN evaluation** (`test_baseline.py`, `test_metacognitive.py`): Benefits from a GPU but runs in reasonable time on CPU.
- Pre-trained weights for all 180 model instances are provided (see Installation below), so GPU hardware is only needed if you wish to retrain from scratch.

---

## 2. Installation Guide

### Instructions

**Step 1 - Clone the repository**
```bash
git clone https://github.com/bogeng-song/RespBias-Metacognition.git
cd RespBias-Metacognition
```

**Step 2 - Create the conda environment**

Because PyMC requires specific C-library versions, we strongly recommend conda-forge over a plain pip install:
```bash
conda create -n metacog -c conda-forge python=3.11 pymc pytensor arviz -y
conda activate metacog
pip install -r requirements.txt
```

**Step 3 - Download data and model weights**

Due to GitHub file-size limits, the behavioral data and pre-trained ANN weights are hosted externally:

- **[OSF Repository](https://osf.io/nz25w/overview?view_only=36e5bcc2225f4b55a54b77b5f690d786)** - human behavioral CSVs
- **[Google Drive](https://drive.google.com/drive/folders/1DlBBSSz3avxgukT13VW2zkMaTNDQ7qdV?usp=sharing)** - pre-trained ANN weights (180 instances)

Place the behavioral CSVs in `data/human_data/` and the ANN weights in a `weights/` directory at the project root.

After downloading, the data folder should contain:
```
data/human_data/
    Experiment1_4_choice.csv       # Exp 1, 4-choice condition  (N=200, 80,000 trials)
    Experiment1_8_choice.csv       # Exp 1, 8-choice condition  (N=200, 80,000 trials)
    Experiment2_accuracy.csv       # Exp 2, accuracy focus      (N=60,  28,800 trials)
    Experiment2_speed.csv          # Exp 2, speed focus         (N=60,  28,800 trials)
```

### Typical install time

| Step | Estimated time |
|---|---|
| git clone | Less than 1 minute |
| conda create + pip install | 5-15 minutes (varies by internet speed and cache) |
| Behavioral data download from OSF | 2-5 minutes (CSV files, ~8 MB total) |
| ANN weights download | 10-30 minutes (~2 GB total) |

---

## 3. Demo

This demo runs the complete human behavioral pipeline on one dataset and generates Figure 2. It requires **no GPU** and completes in under **5 minutes** on a standard desktop CPU.

### Demo Step 1 - Preprocess one dataset

```bash
conda activate metacog

python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_8_choice.csv \
    --output data/human_data/Experiment1_8_choice_aggregated.csv \
    --condition 8choice
```

**Expected console output:**
```
Loading  : data/human_data/Experiment1_8_choice.csv
    Raw shape: (80000, 9)
Computing digit-level metrics ...

  Subjects : 200
  Digits   : 8
  Rows     : 1600  (expected 1600)
  NaN counts : 0

  FAR range        : [0.0000, 0.2955]
  Accuracy range   : [0.0784, 0.9815]
  Confidence range : [1.0000, 4.0000]

Saved to : data/human_data/Experiment1_8_choice_aggregated.csv
```

**Expected run time:** Less than 30 seconds.

### Demo Step 2 - Generate Figure 2

```bash
python notebooks/figure2.py \
    --data4 data/human_data/Experiment1_4_choice.csv \
    --data8 data/human_data/Experiment1_8_choice.csv \
    --output figure2.pdf
```

**Expected output:** A two-panel PDF figure saved to `figure2.pdf`.

- **Panel A** shows two scatter plots (one per condition) where each dot is a digit category. Higher response bias (FAR on x-axis) is associated with lower optimal confidence (y-axis). Correlations are approximately r = -0.96 for the 4-choice condition and r = -0.93 for the 8-choice condition.
- **Panel B** shows six scatter plots for an example participant (default: Sub_id = 12). The three columns show: (1) a positive bias-confidence relationship before accuracy control, (2) a positive bias-accuracy relationship, and (3) a reversed negative bias-confidence relationship after accuracy is controlled. This pattern is shown for both the 4-choice (top row) and 8-choice (bottom row) conditions. The beta values annotated on each subplot match those reported for the example participant in the manuscript.

**Expected run time:** Less than 2 minutes.

---

## 4. Instructions for Use

### Repository structure

```
RespBias-Metacognition/
    data/
        human_data/            Behavioral CSVs (download from OSF)
        model_data/            ANN output CSVs (generated by evaluation scripts)

    core/                      Shared PyTorch modules
        models.py              AlexNet, ResNet18, VGG19-BN (1-channel, 10-class)
        datasets.py            Clean and noisy MNIST dataloaders

    scripts_ann/               ANN training and evaluation
        train.py               Train 60 random-seed instances per architecture
        test_baseline.py       Noise calibration and confidence extraction
        test_metacognitive.py  Apply unsupervised self-monitoring module

    scripts_analysis/          Statistical analysis modules
        preprocess.py          Convert raw CSVs to per-subject x per-digit aggregates
        metrics.py             SDT math (d-prime, FAR, Hautus correction)
        msdt_model.py          Multi-Alternative SDT estimation (PyMC)
        regression.py          Mixed-effects regression (Statsmodels)
        mediation.py           Hierarchical Bayesian mediation (PyMC)

    notebooks/                 Figures and interactive analysis
        figure2.py             Generates Figure 2
        Regression_Mediation.ipynb   All regression and mediation analyses, Figures 3-6
        visualization.py       Shared plotting utilities

    requirements.txt
    README.md
```

### Running on your own data

`preprocess.py` and `figure2.py` accept any CSV that follows the column schema of the OSF data files. Required columns:

| Column | Description |
|---|---|
| `Sub_id` | Participant identifier (integer) |
| `Stimulus` | Digit shown on that trial |
| `Response` | Digit selected by the participant |
| `Correct` | 1 = correct, 0 = incorrect |
| `Confidence` | Confidence rating (1-4) |
| `RT decision` | Decision reaction time (seconds) |

Use `--condition 4choice` for 4-alternative tasks, `--condition 8choice` for 8-alternative tasks, or `--condition exp2` for the Experiment 2 column naming convention (`subject`, `stim`, `response`, `correct`, `confidence`, `resp_rt`).

---

## 5. Reproduction Instructions

### Human Behavioral Pipeline

**Step 1 - Preprocess all four datasets** (run time: under 2 minutes total)

```bash
python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_8_choice.csv \
    --output data/human_data/Experiment1_8_choice_aggregated.csv \
    --condition 8choice

python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment1_4_choice.csv \
    --output data/human_data/Experiment1_4_choice_aggregated.csv \
    --condition 4choice

python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment2_accuracy.csv \
    --output data/human_data/Experiment2_accuracy_aggregated.csv \
    --condition exp2

python scripts_analysis/preprocess.py \
    --input  data/human_data/Experiment2_speed.csv \
    --output data/human_data/Experiment2_speed_aggregated.csv \
    --condition exp2
```

**Step 2 - Generate Figure 2** (run time: under 2 minutes)

```bash
python notebooks/figure2.py \
    --data4 data/human_data/Experiment1_4_choice.csv \
    --data8 data/human_data/Experiment1_8_choice.csv \
    --output figure2.pdf
```

Optional arguments: `--example_sub INT` (default 12), `--fontsize INT` (default 11).

**Step 3 - (Optional) M-SDT parameter estimation for Supplementary analyses**

```bash
python scripts_analysis/msdt_model.py \
    --data_path  data/human_data/Experiment1_8_choice.csv \
    --output_path data/human_data/Experiment1_8_choice_MSDT.csv \
    --n_alts 8 --draws 2000 --tune 1000 --chains 4

python scripts_analysis/msdt_model.py \
    --data_path  data/human_data/Experiment1_4_choice.csv \
    --output_path data/human_data/Experiment1_4_choice_MSDT.csv \
    --n_alts 4 --draws 2000 --tune 1000 --chains 4
```

Run time: 30-60 minutes per dataset (subjects fitted in parallel; scales with CPU core count).

**Step 4 - Regression and mediation analyses (Figures 3-6)**

```bash
conda activate metacog
jupyter notebook notebooks/Regression_Mediation.ipynb
```

Run all cells top to bottom. Expected run times per section:

| Section | Run time |
|---|---|
| Mixed-effects regression, one condition | 1-5 minutes |
| Bayesian mediation, one condition (2000 samples) | 5-15 minutes |
| All four human conditions | ~60 minutes |
| ANN regression, one architecture | 2-5 minutes |
| ANN mediation, one architecture | 5-15 minutes |

Note on Figures 3-6: The regression bar-chart panels are produced directly by the notebook. The mediation path-diagram panels (arrows showing path coefficients) are drawn manually from the posterior means and HDI intervals printed by the notebook; these numbers are reported verbatim in the manuscript.

---

### ANN Pipeline

**Step 1 - Train base models** (skip if using pre-trained weights)

```bash
for i in $(seq 0 59); do
    python scripts_ann/train.py \
        --model alexnet --instance $i --save_dir ./weights/alexnet/
done
# Repeat with --model resnet18 and --model vgg19
```

Run time: 5-10 minutes per instance on a modern GPU; 2-4 hours per instance on CPU. Pre-trained weights are available on Google Drive.

**Step 2 - Evaluate standard (baseline) behavior**

```bash
python scripts_ann/test_baseline.py \
    --model alexnet \
    --model_dir ./weights/alexnet/ \
    --output_dir ./data/model_data/base/ \
    --target_acc 0.64
# Repeat for resnet18 and vgg19
```

The script calibrates Gaussian noise to match human accuracy (~64%) using 5 instances, then evaluates all 60. Run time: 10-20 minutes per architecture on GPU; ~60 minutes on CPU.

**Step 3 - Apply the metacognitive module**

```bash
python scripts_ann/test_metacognitive.py \
    --model      alexnet \
    --model_dir  ./weights/alexnet/ \
    --output_dir ./data/model_data/meta/ \
    --noise_level 1.19
# Repeat for resnet18 and vgg19 using their respective noise levels
```

Use the noise level printed by Step 2. Run time: 10-20 minutes per architecture on GPU; ~60 minutes on CPU.

The ANN output CSVs use the same column schema as the human data and are processed by the same `Regression_Mediation.ipynb` notebook.

---

## Reproducibility Notes

- **Seeded training**: All 60 ANN instances per architecture use seeds derived deterministically from the instance index, so rerunning `train.py` with the same `--instance` number produces the same model.
- **Convergence fallback**: Mixed-effects models use a 3-tier fallback (random slopes -> random intercepts -> cluster-robust OLS). The tier used is printed to console for each model fit.
- **MCMC convergence**: Bayesian models use NUTS sampling with `target_accept=0.95`. Convergence is assessed via R-hat (threshold R-hat < 1.01) and reported in the output.
- **Experiment 2 FAR**: The public Experiment 2 CSVs are pre-split by SAT condition. `preprocess.py` therefore computes FAR within each file. In the original analysis, FAR was computed across both conditions combined; the practical difference is less than 0.006 per digit per subject.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
This license is approved by the [Open Source Initiative](https://opensource.org/licenses/MIT).

## Contact

For questions about the code, data, or manuscript, contact **Bogeng Song** at bsong91@gatech.edu or open an issue in this repository.