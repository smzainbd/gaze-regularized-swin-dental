# -*- coding: utf-8 -*-
"""NB08 paper figures + master summary table. No training — reads saved artifacts only."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import auc, roc_curve

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Modern unified palette (colorblind-friendly) ─────────────────────────────
PALETTE = {
    "stage1": "#64748B",
    "gaze": "#2563EB",
    "ablation": "#9333EA",
    "refit": "#0D9488",
    "resnet": "#EA580C",
    "ensemble": "#0891B2",
    "dentex_s1": "#64748B",
    "dentex_s2": "#DC2626",
    "positive": "#16A34A",
    "negative": "#DC2626",
    "ci_fill": "#94A3B8",
    "grid": "#E2E8F0",
    "text": "#1E293B",
    "muted": "#64748B",
}

K_COLS = ["k_5pct", "k_10pct", "k_15pct", "k_20pct", "k_50pct"]
K_PCT = [5, 10, 15, 20, 50]


def _json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rel(project: Path, p: Path) -> str:
    try:
        return str(p.relative_to(project))
    except ValueError:
        return str(p)


def apply_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": PALETTE["text"],
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.labelsize": 11,
            "axes.labelweight": "500",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Segoe UI",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.alpha": 0.45})


def save_figure(fig: plt.Figure, out_base: Path) -> List[str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for ext in (".png", ".pdf"):
        path = out_base.with_suffix(ext)
        fig.savefig(path, facecolor=fig.get_facecolor())
        written.append(str(path))
    plt.close(fig)
    return written


def bootstrap_curve_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise bootstrap mean curve with 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    boots = np.empty((n_boot, values.shape[1]), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = values[idx].mean(axis=0)
    mean = boots.mean(axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return mean, lo, hi


def _paired_diff_block(block: Optional[dict]) -> dict:
    if not block:
        return {}
    md = block.get("mean_diff", block.get("mean"))
    return {
        "mean_diff": md,
        "lower": block.get("lower"),
        "upper": block.get("upper"),
        "n": block.get("n"),
    }


def _metric_ci_block(block: Optional[dict]) -> dict:
    if not block:
        return {}
    return {
        "point": block.get("point_estimate", block.get("mean")),
        "mean": block.get("mean"),
        "lower": block.get("lower"),
        "upper": block.get("upper"),
        "n": block.get("n"),
    }


def build_paper_summary_table(project: Path) -> dict:
    part1 = _json_load(project / "results" / "stage2_part1_negative_report.json")
    refit = _json_load(project / "results" / "stage2_refit_metrics.json")
    resnet_cv = _json_load(project / "results" / "resnet50_baseline_cv_summary.json")
    resnet_refit = _json_load(project / "results" / "resnet50_baseline_refit_metrics.json")
    faith = _json_load(project / "faithfulness" / "faithfulness_metrics.json")
    dentex = _json_load(project / "statistics" / "dentex_ood_hardened.json")
    pvals_path = project / "statistics" / "p_values.json"
    pvals = _json_load(pvals_path) if pvals_path.is_file() else {}

    ablation_folds = part1.get("cv_ablation_fold_metrics") or []
    ablation_val = [float(r["val_AUROC"]) for r in ablation_folds]
    ablation_test = [float(r["test_AUROC"]) for r in ablation_folds]

    gaze_val = part1["aggregate_gaze_val"]["AUROC"]
    gaze_test = part1["aggregate_gaze_test"]["AUROC"]
    stage1_val = part1["aggregate_stage1_val"]["AUROC"]
    stage1_test = part1["aggregate_stage1_test"]["AUROC"]

    pooled_cv = part1.get("pooled_cv_val_negative_summary") or {}
    held_out = part1.get("held_out_test_negative_summary") or {}
    refit_neg = refit.get("negative_summary") or {}

    table = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": "NB08_figures",
        "primary_claim": "NB04 5-fold CV + held-out bootstrap (null gaze benefit on test)",
        "tufts_detection": {
            "cv_val_auroc": {
                "stage1": {
                    "mean": stage1_val["mean"],
                    "std": stage1_val["std"],
                    "per_fold": stage1_val["per_fold"],
                    "bootstrap_pooled": _metric_ci_block(
                        pooled_cv.get("stage1_bootstrap_auroc")
                    ),
                },
                "gaze_lambda_gt0": {
                    "mean": gaze_val["mean"],
                    "std": gaze_val["std"],
                    "per_fold": gaze_val["per_fold"],
                    "bootstrap_pooled": _metric_ci_block(
                        pooled_cv.get("stage2_bootstrap_auroc")
                    ),
                },
                "ablation_lambda0": {
                    "mean": float(np.mean(ablation_val)) if ablation_val else None,
                    "std": float(np.std(ablation_val, ddof=0)) if ablation_val else None,
                    "per_fold": ablation_val,
                    "note": "Fold means only; no pooled-image bootstrap for ablation.",
                },
                "resnet50_external": {
                    "mean": resnet_cv["aggregate_val"]["val_AUROC"]["mean"],
                    "std": resnet_cv["aggregate_val"]["val_AUROC"]["std"],
                    "per_fold": resnet_cv["aggregate_val"]["val_AUROC"]["per_fold"],
                    "note": "External ImageNet ResNet-50; fold means only on CV val.",
                },
                "paired_gaze_minus_stage1_bootstrap": _paired_diff_block(
                    pooled_cv.get("paired_auroc_diff_stage2_minus_stage1")
                ),
            },
            "held_out_test_auroc": {
                "stage1": _metric_ci_block(held_out.get("stage1_bootstrap_auroc")),
                "gaze_cv_ensemble": _metric_ci_block(
                    held_out.get("stage2_bootstrap_auroc")
                ),
                "ablation_fold_mean": {
                    "mean": float(np.mean(ablation_test)) if ablation_test else None,
                    "std": float(np.std(ablation_test, ddof=0)) if ablation_test else None,
                    "per_fold": ablation_test,
                },
                "stage2_refit_secondary": _metric_ci_block(
                    refit_neg.get("stage2_bootstrap_auroc")
                ),
                "resnet50_refit_style": _metric_ci_block(
                    resnet_refit.get("test_bootstrap_auroc")
                ),
                "paired_gaze_minus_stage1": _paired_diff_block(
                    held_out.get("paired_auroc_diff_stage2_minus_stage1")
                ),
                "paired_refit_minus_stage1": _paired_diff_block(
                    refit_neg.get("paired_auroc_diff_stage2_minus_stage1")
                ),
                "paired_refit_minus_resnet": _paired_diff_block(
                    resnet_refit.get("paired_diff_stage2_minus_resnet")
                ),
                "paired_resnet_minus_stage1": _paired_diff_block(
                    resnet_refit.get("paired_diff_resnet_minus_stage1")
                ),
            },
            "refit_secondary_test": {
                "stage2_auroc": refit["refit_test_metrics"]["AUROC"],
                "stage1_auroc": refit["stage1_test_metrics"]["AUROC"],
                "bootstrap": {
                    "stage2": _metric_ci_block(
                        refit_neg.get("stage2_bootstrap_auroc")
                    ),
                    "stage1": _metric_ci_block(
                        refit_neg.get("stage1_bootstrap_auroc")
                    ),
                },
                "note": refit.get("note"),
            },
            "power_analysis_held_out": held_out.get("power_analysis"),
        },
        "resnet50_baseline": {
            "cv_val_mean": resnet_cv["aggregate_val"]["val_AUROC"]["mean"],
            "cv_test_mean": resnet_cv["aggregate_test"]["AUROC"]["mean"],
            "cv_test_std": resnet_cv["aggregate_test"]["AUROC"]["std"],
            "refit_test_auroc": resnet_refit["test_metrics"]["AUROC"],
            "refit_test_bootstrap": _metric_ci_block(
                resnet_refit.get("test_bootstrap_auroc")
            ),
            "in_statistics_p_values": False,
            "note": "Merged here for paper table; not duplicated in statistics/p_values.json.",
        },
        "dentex_ood": {
            "n_labeled": dentex.get("n_images"),
            "prevalence": dentex.get("prevalence"),
            "auroc": {
                "stage1": _metric_ci_block(dentex["auroc"]["stage1"]),
                "stage2": _metric_ci_block(dentex["auroc"]["stage2"]),
                "paired_diff_stage2_minus_stage1": dentex["auroc"].get(
                    "paired_diff_stage2_minus_stage1"
                ),
            },
            "average_precision": dentex.get("average_precision"),
            "balanced_subsample_auroc": dentex.get("balanced_subsample_auroc"),
            "interpretation": dentex.get("interpretation"),
            "cohort_note": (
                "Labeled DENTEX subset is ~93% positive (n=1005). "
                "Report AUROC + balanced subsample; avoid threshold accuracy claims."
            ),
        },
        "faithfulness": {
            "n_images": faith.get("n_test_images"),
            "aopc_auc": _metric_ci_block(faith.get("AOPC_AUC_per_image_bootstrap")),
            "nss": _metric_ci_block(faith.get("NSS_bootstrap")),
            "cc": _metric_ci_block(faith.get("CC_bootstrap")),
            "ablation": {
                "aopc_auc": _metric_ci_block(
                    faith.get("ablation", {}).get("AOPC_AUC_per_image_bootstrap")
                ),
                "nss": _metric_ci_block(
                    faith.get("ablation", {}).get("NSS_bootstrap")
                ),
                "cc": _metric_ci_block(
                    faith.get("ablation", {}).get("CC_bootstrap")
                ),
                "paired_diff_gaze_minus_ablation": faith.get("ablation", {}).get(
                    "paired_diff_gaze_minus_ablation"
                ),
            },
            "error_taxonomy_counts": faith.get("taxonomy_counts"),
        },
        "secondary_statistics": {
            "mcnemar_tufts_healthy": pvals.get("mcnemar_tufts_healthy_subset"),
            "forgetting_ratio": pvals.get("forgetting_ratio"),
            "dentex_discrimination": pvals.get("dentex_ood_discrimination"),
        },
        "sources": {
            "part1_report": "results/stage2_part1_negative_report.json",
            "refit_metrics": "results/stage2_refit_metrics.json",
            "resnet_cv": "results/resnet50_baseline_cv_summary.json",
            "resnet_refit": "results/resnet50_baseline_refit_metrics.json",
            "faithfulness": "faithfulness/faithfulness_metrics.json",
            "dentex_ood": "statistics/dentex_ood_hardened.json",
            "p_values": "statistics/p_values.json",
        },
    }
    return table


def fig_roc_tufts_test(project: Path, figures: Path) -> List[str]:
    refit = pd.read_csv(project / "results" / "stage2_refit_test_predictions.csv")
    part1 = pd.read_csv(project / "results" / "stage2_part1_test_predictions.csv")
    resnet = pd.read_csv(project / "results" / "resnet50_baseline_test_predictions.csv")

    y = refit["has_lesion_gt"].astype(int).values
    series = [
        (refit["prob_stage1"].values, "Stage-1 (zero-shot)", PALETTE["stage1"]),
        (part1["prob_stage2_gaze"].values, "Stage-2 gaze (CV ensemble)", PALETTE["gaze"]),
        (refit["prob_stage2_refit"].values, "Stage-2 refit", PALETTE["refit"]),
        (resnet["prob_resnet50"].values, "ResNet-50 baseline", PALETTE["resnet"]),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for probs, label, color in series:
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(
            fpr,
            tpr,
            color=color,
            label=f"{label} (AUC={auc(fpr, tpr):.3f})",
            lw=2.4,
        )
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1.2, alpha=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Tufts held-out test — lesion detection ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    return save_figure(fig, figures / "fig01_roc_tufts_test")


def fig_dentex_ood_summary(project: Path, figures: Path) -> List[str]:
    dentex = _json_load(project / "statistics" / "dentex_ood_hardened.json")
    metrics = [
        ("AUROC (raw)", "auroc", "stage1", "stage2"),
        ("AUROC (balanced)", "balanced_subsample_auroc", "stage1_auroc_balanced", "stage2_auroc_balanced"),
        ("Average precision", "average_precision", "stage1", "stage2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    for ax, (title, block_key, s1_key, s2_key) in zip(axes, metrics):
        block = dentex[block_key]
        s1 = block[s1_key]
        s2 = block[s2_key]
        x = np.array([0, 1])
        means = [s1["mean"], s2["mean"]]
        errs = [
            [means[0] - s1["lower"], s2["mean"] - s2["lower"]],
            [s1["upper"] - means[0], s2["upper"] - means[1]],
        ]
        colors = [PALETTE["dentex_s1"], PALETTE["dentex_s2"]]
        ax.bar(x, means, color=colors, width=0.55, alpha=0.92, edgecolor="white", linewidth=0.8)
        ax.errorbar(
            x,
            means,
            yerr=errs,
            fmt="none",
            ecolor=PALETTE["text"],
            capsize=5,
            capthick=1.2,
            lw=1.2,
        )
        ax.set_xticks(x, ["Stage-1", "Stage-2"])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if block_key == "average_precision":
            ax.axhline(
                dentex.get("random_ap_baseline", 0.93),
                color=PALETTE["muted"],
                ls=":",
                lw=1.2,
                label="Prevalence baseline",
            )
            ax.legend(loc="lower right")
    fig.suptitle(
        f"DENTEX OOD (n={dentex['n_images']}, prevalence={dentex['prevalence']:.1%})",
        y=1.02,
        fontsize=13,
        fontweight="600",
    )
    plt.tight_layout()
    return save_figure(fig, figures / "fig02_dentex_ood_summary")


def fig_aopc_curve(project: Path, figures: Path) -> List[str]:
    df = pd.read_csv(project / "faithfulness" / "aopc_per_image.csv")
    vals = df[K_COLS].to_numpy(dtype=float)
    mean, lo, hi = bootstrap_curve_ci(vals)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(K_PCT, lo, hi, color=PALETTE["gaze"], alpha=0.18, label="95% bootstrap CI")
    ax.plot(K_PCT, mean, "o-", color=PALETTE["gaze"], markersize=7, label="Mean confidence drop")
    ax.axhline(0, color=PALETTE["muted"], ls="--", lw=1.1)
    ax.set_xlabel("Occluded fraction k (%)")
    ax.set_ylabel("Mean confidence drop")
    ax.set_title("AOPC faithfulness curve (Stage-2 refit, n=100)")
    ax.set_xticks(K_PCT)
    ax.legend(loc="upper left")
    return save_figure(fig, figures / "fig03_aopc_faithfulness_curve")


def fig_forest_detection(project: Path, figures: Path, summary: dict) -> List[str]:
    rows: List[Tuple[str, float, float, float, str]] = []
    cv = summary["tufts_detection"]["cv_val_auroc"]
    test = summary["tufts_detection"]["held_out_test_auroc"]

    def add_row(label: str, block: dict, cohort: str) -> None:
        if not block or block.get("lower") is None:
            if block.get("mean") is not None:
                rows.append((label, block["mean"], block["mean"], block["mean"], cohort))
            return
        pt = block.get("point") or block.get("mean")
        rows.append((label, pt, block["lower"], block["upper"], cohort))

    add_row("Stage-1 CV val", cv["stage1"]["bootstrap_pooled"], "CV val (pooled)")
    add_row("Gaze CV val", cv["gaze_lambda_gt0"]["bootstrap_pooled"], "CV val (pooled)")
    add_row("Stage-1 test", test["stage1"], "Held-out test")
    add_row("Gaze CV ens. test", test["gaze_cv_ensemble"], "Held-out test")
    add_row("Stage-2 refit test", test["stage2_refit_secondary"], "Held-out test")
    add_row("ResNet-50 test", test["resnet50_refit_style"], "Held-out test")
    add_row("Stage-1 DENTEX", summary["dentex_ood"]["auroc"]["stage1"], "DENTEX OOD")
    add_row("Stage-2 DENTEX", summary["dentex_ood"]["auroc"]["stage2"], "DENTEX OOD")

    color_map = {
        "CV val (pooled)": PALETTE["gaze"],
        "Held-out test": PALETTE["refit"],
        "DENTEX OOD": PALETTE["dentex_s2"],
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(rows))
    for i, (label, pt, lo, hi, cohort) in enumerate(rows):
        c = color_map.get(cohort, PALETTE["stage1"])
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter([pt], [i], color=c, s=48, zorder=3, edgecolors="white", linewidths=0.6)
        ax.text(
            hi + 0.012,
            i,
            f"{pt:.3f} [{lo:.3f}, {hi:.3f}]",
            va="center",
            fontsize=8.5,
            color=PALETTE["text"],
        )
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("AUROC")
    ax.set_title("Detection performance — point estimates with 95% bootstrap CI")
    ax.set_xlim(0.35, 0.85)
    ax.invert_yaxis()
    handles = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
    ax.legend(handles=handles, loc="lower right")
    return save_figure(fig, figures / "fig04_forest_detection_auroc")


def fig_table1(project: Path, figures: Path, summary: dict) -> Tuple[List[str], pd.DataFrame]:
    cv = summary["tufts_detection"]["cv_val_auroc"]
    test = summary["tufts_detection"]["held_out_test_auroc"]

    def fmt_ci(block: Optional[dict], fold_mean: Optional[float] = None, fold_std: Optional[float] = None) -> str:
        if block and block.get("lower") is not None:
            pt = block.get("point") or block.get("mean")
            return f"{pt:.3f} [{block['lower']:.3f}, {block['upper']:.3f}]"
        if fold_mean is not None:
            if fold_std is not None:
                return f"{fold_mean:.3f} ± {fold_std:.3f}"
            return f"{fold_mean:.3f}"
        return "—"

    rows = [
        ["Stage-1 (OralXrays → Tufts)", fmt_ci(cv["stage1"]["bootstrap_pooled"]), fmt_ci(test["stage1"])],
        ["Stage-2 gaze (λ>0)", fmt_ci(cv["gaze_lambda_gt0"]["bootstrap_pooled"]), fmt_ci(test["gaze_cv_ensemble"])],
        [
            "Stage-2 ablation (λ=0)",
            fmt_ci(None, cv["ablation_lambda0"]["mean"], cv["ablation_lambda0"]["std"]),
            fmt_ci(None, test["ablation_fold_mean"]["mean"], test["ablation_fold_mean"]["std"]),
        ],
        [
            "ResNet-50 external",
            fmt_ci(None, cv["resnet50_external"]["mean"], cv["resnet50_external"]["std"]),
            fmt_ci(test["resnet50_refit_style"]),
        ],
        ["Stage-2 refit (secondary)", "—", fmt_ci(test["stage2_refit_secondary"])],
    ]
    cols = ["Model", "CV validation AUROC", "Held-out test AUROC"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(figures / "paper_table1.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colColours=["#E2E8F0"] * 3,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="600", color=PALETTE["text"])
        cell.set_edgecolor("#CBD5E1")
    ax.set_title("Table 1 — Tufts lesion detection (AUROC with 95% CI where available)", pad=16, fontweight="600")
    written = save_figure(fig, figures / "fig05_table1_detection")
    return written, df


def fig_cv_fold_bars(project: Path, figures: Path) -> List[str]:
    part1 = _json_load(project / "results" / "stage2_part1_negative_report.json")
    resnet = _json_load(project / "results" / "resnet50_baseline_cv_summary.json")
    folds = np.arange(1, 6)

    gaze = [r["val_AUROC"] for r in part1["cv_gaze_fold_metrics"]]
    ablation = [r["val_AUROC"] for r in part1["cv_ablation_fold_metrics"]]
    stage1 = [r["val_AUROC"] for r in part1["cv_stage1_fold_metrics"]]
    resnet_v = resnet["aggregate_val"]["val_AUROC"]["per_fold"]

    w = 0.2
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * w, stage1, w, label="Stage-1", color=PALETTE["stage1"])
    ax.bar(x - 0.5 * w, gaze, w, label="Gaze", color=PALETTE["gaze"])
    ax.bar(x + 0.5 * w, ablation, w, label="Ablation (λ=0)", color=PALETTE["ablation"])
    ax.bar(x + 1.5 * w, resnet_v, w, label="ResNet-50", color=PALETTE["resnet"])
    ax.set_xticks(x, [f"Fold {i}" for i in folds])
    ax.set_ylabel("Validation AUROC")
    ax.set_title("5-fold CV validation AUROC by fold")
    ax.set_ylim(0.45, 0.82)
    ax.legend(ncol=2, loc="lower right")
    return save_figure(fig, figures / "fig06_cv_fold_val_auroc")


def fig_gaze_vs_ablation(project: Path, figures: Path) -> List[str]:
    part1 = _json_load(project / "results" / "stage2_part1_negative_report.json")
    gaze = [r["val_AUROC"] for r in part1["cv_gaze_fold_metrics"]]
    ablation = [r["val_AUROC"] for r in part1["cv_ablation_fold_metrics"]]
    folds = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(folds, gaze, "o-", color=PALETTE["gaze"], label="Gaze (λ>0)", markersize=8)
    ax.plot(folds, ablation, "s--", color=PALETTE["ablation"], label="Ablation (λ=0)", markersize=7)
    for i in range(5):
        ax.plot([folds[i], folds[i]], [gaze[i], ablation[i]], color=PALETTE["ci_fill"], lw=1.5, zorder=0)
    ax.set_xticks(folds)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Gaze vs ablation — per-fold validation")
    ax.legend()
    return save_figure(fig, figures / "fig07_gaze_vs_ablation_cv")


def fig_faithfulness_metrics(project: Path, figures: Path, summary: dict) -> List[str]:
    faith = summary["faithfulness"]
    metrics = ["AOPC AUC", "NSS", "CC"]
    gaze_vals = [
        faith["aopc_auc"],
        faith["nss"],
        faith["cc"],
    ]
    ab_vals = [
        faith["ablation"]["aopc_auc"],
        faith["ablation"]["nss"],
        faith["ablation"]["cc"],
    ]

    x = np.arange(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for offset, blocks, color, label in [
        (-w / 2, gaze_vals, PALETTE["gaze"], "Gaze"),
        (w / 2, ab_vals, PALETTE["ablation"], "Ablation"),
    ]:
        means = [b["mean"] for b in blocks]
        errs = [
            [m - b["lower"] for m, b in zip(means, blocks)],
            [b["upper"] - m for m, b in zip(means, blocks)],
        ]
        ax.bar(x + offset, means, w, color=color, label=label, alpha=0.9)
        ax.errorbar(x + offset, means, yerr=errs, fmt="none", ecolor=PALETTE["text"], capsize=4)
    ax.axhline(0, color=PALETTE["muted"], ls="--", lw=1)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Metric value")
    ax.set_title("Faithfulness & gaze alignment (bootstrap 95% CI)")
    ax.legend()
    return save_figure(fig, figures / "fig08_faithfulness_alignment")


def fig_error_taxonomy(project: Path, figures: Path) -> List[str]:
    tax = _json_load(project / "faithfulness" / "error_taxonomy_summary.json")
    counts = tax["counts"]
    labels = {
        "tn": "True negative",
        "fp_type_a_hallucination": "FP Type A (hallucination)",
        "fp_type_b_misattribution": "FP Type B (misattribution)",
        "fn_blind": "FN blind",
        "fn_attentive": "FN attentive",
    }
    items = [(labels.get(k, k), v) for k, v in counts.items()]
    items.sort(key=lambda t: t[1], reverse=True)
    names, vals = zip(*items)
    colors = [
        PALETTE["positive"] if "True" in n else PALETTE["negative"] if "FN" in n else PALETTE["resnet"]
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Count (n=100 faithfulness subset)")
    ax.set_title("Attribution error taxonomy")
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=9)
    return save_figure(fig, figures / "fig09_error_taxonomy")


def fig_power_analysis(project: Path, figures: Path, summary: dict) -> List[str]:
    power = summary["tufts_detection"].get("power_analysis_held_out") or {}
    n_needed = power.get("approx_total_same_set", 2230)
    n_have = 100

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([0], [n_needed], color=PALETTE["ci_fill"], height=0.35, label="Required n (80% power, ΔAUROC=0.05)")
    ax.barh([0], [n_have], color=PALETTE["gaze"], height=0.35, label=f"Held-out test n={n_have}")
    ax.set_xlim(0, max(n_needed * 1.1, 100))
    ax.set_yticks([])
    ax.set_xlabel("Sample size")
    ax.set_title("Held-out test is underpowered for ΔAUROC = 0.05")
    ax.legend(loc="lower right")
    ax.text(n_have + 40, 0, f"{n_have}", va="center", fontsize=10, color=PALETTE["gaze"])
    ax.text(n_needed + 40, 0, f"{n_needed}", va="center", fontsize=10, color=PALETTE["muted"])
    return save_figure(fig, figures / "fig10_power_sample_size")


def fig_paired_diffs(project: Path, figures: Path, summary: dict) -> List[str]:
    test = summary["tufts_detection"]["held_out_test_auroc"]
    diffs = [
        ("Gaze − Stage-1", test.get("paired_gaze_minus_stage1"), PALETTE["gaze"]),
        ("Refit − Stage-1", test.get("paired_refit_minus_stage1"), PALETTE["refit"]),
        ("ResNet − Stage-1", test.get("paired_resnet_minus_stage1"), PALETTE["resnet"]),
        ("Refit − ResNet", test.get("paired_refit_minus_resnet"), PALETTE["ensemble"]),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(diffs))
    for i, (label, block, color) in enumerate(diffs):
        if not block or block.get("lower") is None:
            continue
        md = block.get("mean_diff")
        if md is None:
            continue
        lo, hi = block["lower"], block["upper"]
        ax.plot([lo, hi], [i, i], color=color, lw=2.4)
        ax.scatter([md], [i], color=color, s=50, zorder=3)
        ax.text(hi + 0.005, i, f"{md:+.3f}", va="center", fontsize=9)
    ax.axvline(0, color=PALETTE["muted"], ls="--", lw=1.1)
    ax.set_yticks(y, [d[0] for d in diffs])
    ax.set_xlabel("ΔAUROC (95% bootstrap CI)")
    ax.set_title("Paired AUROC differences — Tufts held-out test")
    ax.invert_yaxis()
    return save_figure(fig, figures / "fig11_paired_auroc_diffs")


def fig_nss_scatter(project: Path, figures: Path) -> List[str]:
    path = project / "faithfulness" / "alignment_per_image.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(df["nss"], df["cc"], alpha=0.55, color=PALETTE["gaze"], edgecolors="white", linewidths=0.4, s=42)
    ax.axhline(0, color=PALETTE["muted"], ls="--", lw=1)
    ax.axvline(0, color=PALETTE["muted"], ls="--", lw=1)
    ax.set_xlabel("NSS (gaze alignment)")
    ax.set_ylabel("Pearson CC (attention vs gaze)")
    ax.set_title("Per-image gaze–attention alignment")
    return save_figure(fig, figures / "fig12_nss_cc_scatter")


def fig_attention_overlays(project: Path, figures: Path) -> Tuple[List[str], str]:
    """Optional qualitative figure — requires torch, images, and checkpoint."""
    ckpt = project / "checkpoints" / "stage2_refit_best.pth"
    if not ckpt.is_file():
        return [], "skipped: checkpoint missing"
    try:
        import random
        import torch
        from torch.utils.data import DataLoader, Subset

        from core_modules import SwinGazeModel, align_attention_to_gaze
        from dental_dataset import DentalDataset
    except ImportError as exc:
        return [], f"skipped: {exc}"

    config = _json_load(project / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinGazeModel(config).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=False)
    model.eval()

    split_path = project / "splits" / "test_split.csv"
    if not split_path.is_file():
        return [], "skipped: splits/test_split.csv missing"

    test_df = pd.read_csv(split_path)
    lesion_idx = test_df.index[test_df["has_lesion_gt"] == 1].tolist()
    healthy_idx = test_df.index[test_df["has_lesion_gt"] == 0].tolist()
    random.seed(42)
    if len(lesion_idx) < 2 or len(healthy_idx) < 1:
        return [], "skipped: insufficient test examples"
    selected = random.sample(lesion_idx, min(2, len(lesion_idx))) + random.sample(
        healthy_idx, min(1, len(healthy_idx))
    )

    ds = DentalDataset(
        str(project / "splits" / "test_split.csv"),
        str(project),
        image_root=str(project),
        augment=False,
    )
    loader = DataLoader(Subset(ds, selected), batch_size=len(selected), shuffle=False)
    batch = next(iter(loader))
    if batch["image"] is None:
        return [], "skipped: images not available on this machine"

    with torch.no_grad():
        _, _, out = model(batch["image"].to(device))
    attn = out.attentions[-1].mean(dim=1).mean(dim=1).reshape(-1, 7, 7)
    attn_up = align_attention_to_gaze(attn, 224).cpu().numpy()
    imgs = batch["image"].permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-9)

    n = len(selected)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(n):
        axes[0, i].imshow(imgs[i], cmap="gray")
        axes[0, i].axis("off")
        gt = "Lesion" if float(batch["has_lesion_gt"][i]) > 0.5 else "Healthy"
        axes[0, i].set_title(gt, fontsize=9)
        heat = attn_up[i]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        overlay = 0.62 * imgs[i] + 0.38 * cm.magma(heat)[..., :3]
        axes[1, i].imshow(np.clip(overlay, 0, 1))
        axes[1, i].axis("off")
    fig.suptitle("Attention overlays (qualitative)", y=1.02, fontweight="600")
    return save_figure(fig, figures / "fig13_attention_overlays"), "ok"


def generate_all_figures(
    project: Path,
    run_attention: bool = False,
) -> dict:
    apply_theme()
    figures = project / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    theme_path = figures / "nb08_theme.json"
    with open(theme_path, "w", encoding="utf-8") as f:
        json.dump({"palette": PALETTE, "k_levels_pct": K_PCT}, f, indent=2)

    summary = build_paper_summary_table(project)
    summary_path = project / "results" / "paper_summary_table.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    outputs: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_table": _rel(project, summary_path),
        "theme": _rel(project, theme_path),
        "figures": [],
        "attention_overlays": "not_requested",
    }

    steps = [
        ("fig01_roc_tufts_test", lambda: fig_roc_tufts_test(project, figures)),
        ("fig02_dentex_ood_summary", lambda: fig_dentex_ood_summary(project, figures)),
        ("fig03_aopc_faithfulness_curve", lambda: fig_aopc_curve(project, figures)),
        ("fig04_forest_detection_auroc", lambda: fig_forest_detection(project, figures, summary)),
        ("fig06_cv_fold_val_auroc", lambda: fig_cv_fold_bars(project, figures)),
        ("fig07_gaze_vs_ablation_cv", lambda: fig_gaze_vs_ablation(project, figures)),
        ("fig08_faithfulness_alignment", lambda: fig_faithfulness_metrics(project, figures, summary)),
        ("fig09_error_taxonomy", lambda: fig_error_taxonomy(project, figures)),
        ("fig10_power_sample_size", lambda: fig_power_analysis(project, figures, summary)),
        ("fig11_paired_auroc_diffs", lambda: fig_paired_diffs(project, figures, summary)),
        ("fig12_nss_cc_scatter", lambda: fig_nss_scatter(project, figures)),
    ]

    table_written, _ = fig_table1(project, figures, summary)
    outputs["figures"].append({"name": "fig05_table1_detection", "paths": table_written})

    for name, fn in steps:
        try:
            paths = fn()
            outputs["figures"].append({"name": name, "paths": paths, "status": "ok"})
        except Exception as exc:
            outputs["figures"].append({"name": name, "status": "error", "error": str(exc)})

    if run_attention:
        paths, note = fig_attention_overlays(project, figures)
        outputs["attention_overlays"] = note
        if paths:
            outputs["figures"].append({"name": "fig13_attention_overlays", "paths": paths, "status": "ok"})

    status_path = figures / "figures_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    outputs["status_file"] = _rel(project, status_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NB08 paper figures and summary table.")
    parser.add_argument("--project", default=".", help="Project root (default: cwd)")
    parser.add_argument(
        "--attention-overlays",
        action="store_true",
        help="Attempt qualitative attention overlays (needs GPU/images)",
    )
    args = parser.parse_args()
    project = Path(args.project).resolve()
    os.chdir(project)
    result = generate_all_figures(project, run_attention=args.attention_overlays)
    n_ok = sum(1 for f in result["figures"] if f.get("status", "ok") == "ok")
    print(json.dumps({"figures_ok": n_ok, "summary": result["summary_table"]}, indent=2))


if __name__ == "__main__":
    main()
