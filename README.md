# Self-monitoring in perceptual decisions by humans and machines

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OSF Data](https://img.shields.io/badge/Data_%26_Weights-OSF-green.svg)](https://osf.io/[YOUR_OSF_LINK])
[![Paper](https://img.shields.io/badge/Paper-bioRxiv-red.svg)](https://doi.org/[YOUR_DOI_HERE])

> **Official repository for the manuscript:**  
> Song, B., & Rahnev, D. (2026). *Self-monitoring in perceptual decisions by humans and machines*. 

## 📖 Overview
Perceptual confidence is typically conceptualized as monitoring stimulus uncertainty. This project investigates whether humans and Artificial Neural Networks (ANNs) go beyond stimulus uncertainty by engaging in **self-monitoring** of their own decisional tendencies (response biases). 

Using multi-alternative (4- and 8-choice) perceptual decision-making tasks, mixed-effects regressions, and Bayesian multilevel mediation, we demonstrate:
1. **Humans** naturally down-weight their confidence when biased toward a specific response (a negative direct effect of bias on confidence).
2. **Standard ANNs** (AlexNet, ResNet18, VGG19) do *not* self-monitor; response bias inflates their confidence.
3. **Metacognitive ANNs**, augmented with a novel unsupervised self-monitoring module, successfully reproduce the human-like negative bias-confidence relationship.

## 🗂️ Repository Structure

*   `data/`: Directory for human behavioral datasets and model-generated outputs. *(Note: Large neural network weight files are hosted on OSF).*
*   `core/`: Shared PyTorch implementations of ANN architectures and datasets.
*   `scripts_ann/`: PyTorch pipelines to train ANNs, calibrate visual noise, and test baseline vs. metacognitive behaviors.
*   `scripts_analysis/`: Python scripts for statistical evaluation (`statsmodels` for regressions, `PyMC` for Bayesian mediation and M-SDT).
*   `notebooks/`: Jupyter notebooks used to generate the figures presented in the manuscript.

## ⚙️ Installation


```bash
git clone https://github.com/bogeng-song/self-monitoring-perceptual-decisions.git
cd self-monitoring-perceptual-decisions

conda create -n metacognition python=3.10
conda activate metacognition

pip install -r requirements.txt
