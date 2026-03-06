# Self-monitoring in perceptual decisions by humans and machines

[![DOI](https://img.shields.io/badge/DOI-Pending-blue.svg)](#)
[![OSF Data](https://img.shields.io/badge/Data_%26_Models-OSF-success.svg)](https://osf.io/[YOUR_OSF_LINK])
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for the manuscript: **"Self-monitoring in perceptual decisions by humans and machines"** by Bogeng Song and Dobromir Rahnev (School of Psychology, Georgia Institute of Technology).

## 🧠 Overview
Do perceptual confidence judgments reflect the active self-monitoring of one’s own decisional tendencies, or merely the monitoring of external stimulus uncertainty? 

This repository contains the code to reproduce our findings that:
1. **Humans** intuitively down-weight their confidence when biased toward a specific category in multi-alternative tasks, a signature of self-monitoring that disappears under speed pressure.
2. **Standard ANNs** (AlexNet, ResNet18, VGG19) lack this capacity, instead exhibiting artificially inflated confidence for biased categories.
3. An **unsupervised metacognitive module** can track historical decisional tendencies without trial-by-trial accuracy feedback, applying a bias shift that successfully endows standard ANNs with human-like self-monitoring.

## 📂 Repository Structure
* `src/behavior_stats/`: Scripts for mixed-effects regressions (`statsmodels`) and Bayesian hierarchical mediation (`PyMC`) on human psychophysics data.
* `src/msdt/`: Generative models for Multi-Alternative Signal Detection Theory (M-SDT) parameter estimation.
* `src/ann_pipeline/`: `PyTorch` scripts to train base models on MNIST, test on noisy inputs, and evaluate raw confidence behavior.
* `src/metacognition/`: Implementation of the unsupervised metacognitive module (confidence penalty shift $S_k$ based on category selection priors).
* `notebooks/`: Jupyter notebooks containing the code to generate all main text and supplementary figures.

## ⚙️ Installation
We recommend creating a virtual environment (e.g., using `conda`) to run this code to prevent dependency conflicts with PyMC's C-compiler backend.

```bash
git clone https://github.com/bogeng-song/self-monitoring-perceptual-decisions.git
cd self-monitoring-perceptual-decisions

conda create -n metacognition python=3.10
conda activate metacognition

pip install -r requirements.txt
