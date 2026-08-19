# Gaze-Regularized LoRA Adaptation of Swin Transformers for Dental Panoramic Lesion Detection

Code, configuration, fold splits, and result artifacts for:

> S M Zain & Eram Mahamud. *Gaze-Regularized Low-Rank Adaptation of Swin Transformers for Dental Panoramic Lesion Detection: A Rigorously Controlled Negative Result.*

This repository reproduces the two-phase pipeline in the paper: OralXrays-9 domain pre-training, Tufts gaze-regularized LoRA fine-tuning, a matched λ = 0 ablation, a ResNet-50 baseline, DENTEX out-of-distribution evaluation, and a faithfulness / error-taxonomy audit.

## What is in this repository

| Path | Contents |
|---|---|
| `config.json` | Frozen hyperparameters used for the reported run (`224×224`, LoRA r=8, `gaze_lambda=0.05`) |
| `*.py` | Standalone modules (dataset, model, Stage-2 CV/refit, baseline, faithfulness, statistics, figures) |
| `Notebooks/` | Orchestration notebooks `novelty_checks` → `NB00` … `NB08` (+ `NB04b`) |
| `splits/` | Train / val / test / CV-pool / DENTEX / OralXrays split files |
| `results/` | Detection metrics, per-image predictions, negative-result report, post-run checklist |
| `statistics/` | Bootstrap / McNemar / power artifacts |
| `faithfulness/` | AOPC, NSS/CC, error-taxonomy tables |
| `figures/` | Publication-quality figure PDFs |
| `hyperparameter_sweep/` | Optuna best-parameter snapshot and trial table |


