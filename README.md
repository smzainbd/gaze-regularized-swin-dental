# Gaze-Regularized LoRA Adaptation of Swin Transformers for Dental Panoramic Lesion Detection

Code, configuration, fold splits, and result artifacts for:

> S. M. Zain & Eram Mahamud. *Gaze-Regularized Low-Rank Adaptation of Swin Transformers for Dental Panoramic Lesion Detection: A Rigorously Controlled Negative Result.*

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

## What is not included (by design)

Radiographs, gaze maps, and trained weights are **not** in this repository:

- **Images** belong to TDD, OralXrays-9, and DENTEX and must be downloaded from their official sources (see the paper references).
- **Gaze maps** (`gaze_maps_48x48/*.npy`) are derived from TDD eye-tracking files by `Notebooks/NB00_prep.ipynb`.
- **Checkpoints** (`checkpoints/*.pth`) are large binary weights; they are produced by `NB03`, `NB04`, `NB04b`, and `NB05`.

Do not upload those folders to GitHub.

## Datasets

Place extracted data so that `data/` matches the layout expected by `dental_dataset.py` / `NB00_prep.ipynb` (Tufts under `data/tdd`, OralXrays-9 and DENTEX in the same `data/` tree used in the paper). Then run notebooks in order:

```
novelty_checks.ipynb
NB00_prep.ipynb
NB01_model.ipynb
NB02_hyperparameter_sweep.ipynb
NB03_stage1_train.ipynb
NB04_stage2_train.ipynb
NB04b_resnet_baseline.ipynb
NB05_cross_domain_eval.ipynb
NB06_faithfulness.ipynb
NB07_statistics.ipynb
NB08_figures.ipynb
```

On Google Colab the notebooks mount Drive and set `PROJECT` to the repository root. Locally they treat the parent of `Notebooks/` as `PROJECT`.

## Environment

Python 3.10+ with a CUDA GPU is recommended for training. CPU is sufficient to inspect artifacts and regenerate some figures.

```bash
pip install -r requirements.txt
```

Core libraries: PyTorch, torchvision, Hugging Face `transformers` + `peft`, scikit-learn, SciPy, pandas, matplotlib.

## Reproducibility notes

- All reported training hyperparameters are in `config.json` (not hardcoded in notebooks).
- Fold membership is in `splits/` (5-fold stratified CV on the 900-image development pool; 100-image held-out test).
- `results/stage2_post_run_checklist.json` records the frozen Part-1 GREEN checks.
- `results/stage2_part1_negative_report.json` and `statistics/p_values.json` are the numerical source for the paper's detection and inference claims.
- Training histories and model weights are omitted to stay within GitHub's web-upload file limit; metrics and predictions needed to recompute tables are included.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{zain2026gaze,
  title={Gaze-Regularized Low-Rank Adaptation of Swin Transformers for Dental Panoramic Lesion Detection: A Rigorously Controlled Negative Result},
  author={Zain, S. M. and Mahamud, Eram},
  year={2026}
}
```

