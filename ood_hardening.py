# -*- coding: utf-8 -*-
"""OOD hardening for DENTEX: AP + balanced-subsample AUROC (no re-inference)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return float("nan")


def bootstrap_scalar_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    point = metric_fn(y, p)
    if not np.isfinite(point):
        point = float("nan")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        v = metric_fn(y[idx], p[idx])
        if np.isfinite(v):
            samples.append(float(v))
    if not samples:
        return {
            "point_estimate": float(point) if np.isfinite(point) else None,
            "mean": None,
            "lower": None,
            "upper": None,
            "n": n,
            "n_boot_effective": 0,
        }
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "point_estimate": float(point),
        "mean": float(np.mean(samples)),
        "lower": float(np.percentile(samples, lo)),
        "upper": float(np.percentile(samples, hi)),
        "n": n,
        "n_boot_effective": len(samples),
    }


def bootstrap_paired_diff_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap CI for metric(B) - metric(A)."""
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    n = len(y)
    a0, b0 = metric_fn(y, p_a), metric_fn(y, p_b)
    point = (b0 if np.isfinite(b0) else np.nan) - (a0 if np.isfinite(a0) else np.nan)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a = metric_fn(y[idx], p_a[idx])
        b = metric_fn(y[idx], p_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(b - a))
    if not diffs:
        return {
            "mean_diff": float(point) if np.isfinite(point) else None,
            "lower": None,
            "upper": None,
            "n": n,
            "n_boot_effective": 0,
        }
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "mean_diff": float(np.mean(diffs)),
        "lower": float(np.percentile(diffs, lo)),
        "upper": float(np.percentile(diffs, hi)),
        "n": n,
        "n_boot_effective": len(diffs),
        "point_diff": float(point) if np.isfinite(point) else None,
    }


def balanced_subsample_auroc(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_draws: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Stress-test discrimination under class balance.

    Each draw: all negatives + equal number of random positives.
    Reports AUROC for A/B and paired Δ on those balanced subsets.
    """
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]
    n_neg = len(neg_idx)
    if n_neg < 2 or len(pos_idx) < n_neg:
        return {
            "ok": False,
            "reason": f"need >=2 negatives and >=n_neg positives; got n_neg={n_neg}, n_pos={len(pos_idx)}",
        }

    rng = np.random.default_rng(seed)
    a_scores, b_scores, diffs = [], [], []
    for _ in range(int(n_draws)):
        take_pos = rng.choice(pos_idx, size=n_neg, replace=False)
        idx = np.concatenate([neg_idx, take_pos])
        a = _safe_auroc(y[idx], p_a[idx])
        b = _safe_auroc(y[idx], p_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            a_scores.append(a)
            b_scores.append(b)
            diffs.append(b - a)

    def _summ(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return None
        lo = (alpha / 2) * 100
        hi = (1 - alpha / 2) * 100
        return {
            "mean": float(np.mean(arr)),
            "lower": float(np.percentile(arr, lo)),
            "upper": float(np.percentile(arr, hi)),
            "n_draws_effective": int(len(arr)),
        }

    dsum = _summ(diffs)
    return {
        "ok": True,
        "n_negatives": int(n_neg),
        "n_positives_per_draw": int(n_neg),
        "balanced_n_per_draw": int(2 * n_neg),
        "n_draws_requested": int(n_draws),
        "stage1_auroc_balanced": _summ(a_scores),
        "stage2_auroc_balanced": _summ(b_scores),
        "paired_auroc_diff_stage2_minus_stage1_balanced": dsum,
        "discrimination_significant_balanced": bool(
            dsum and (dsum["lower"] > 0 or dsum["upper"] < 0)
        ),
        "interpretation": (
            "Stage-2 better on balanced DENTEX draws"
            if dsum and dsum["lower"] > 0
            else "Stage-1 better on balanced DENTEX draws"
            if dsum and dsum["upper"] < 0
            else "No discrimination difference on balanced DENTEX draws (null OOD)"
        ),
    }


def _prob_col(df: pd.DataFrame) -> str:
    for c in ("probs", "prob", "prob_stage2", "score"):
        if c in df.columns:
            return c
    raise KeyError(f"No probability column in {list(df.columns)}")


def harden_dentex_ood(
    stage2_csv: str,
    stage1_csv: str,
    n_boot: int = 2000,
    n_balanced_draws: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    s2 = pd.read_csv(stage2_csv)
    s1 = pd.read_csv(stage1_csv)
    y2 = s2["has_lesion_gt"].astype(int).to_numpy()
    y1 = s1["has_lesion_gt"].astype(int).to_numpy()
    n = min(len(y2), len(y1))
    y2, y1 = y2[:n], y1[:n]
    if not np.array_equal(y2, y1):
        raise ValueError("DENTEX GT mismatch between Stage-2 and Stage-1 prediction files")
    y = y2
    p2 = s2[_prob_col(s2)].to_numpy()[:n]
    p1 = s1[_prob_col(s1)].to_numpy()[:n]

    auroc2 = bootstrap_scalar_ci(y, p2, _safe_auroc, n_boot=n_boot, seed=seed)
    auroc1 = bootstrap_scalar_ci(y, p1, _safe_auroc, n_boot=n_boot, seed=seed)
    d_auroc = bootstrap_paired_diff_ci(y, p1, p2, _safe_auroc, n_boot=n_boot, seed=seed)

    ap2 = bootstrap_scalar_ci(y, p2, _safe_ap, n_boot=n_boot, seed=seed + 1)
    ap1 = bootstrap_scalar_ci(y, p1, _safe_ap, n_boot=n_boot, seed=seed + 1)
    d_ap = bootstrap_paired_diff_ci(y, p1, p2, _safe_ap, n_boot=n_boot, seed=seed + 1)

    balanced = balanced_subsample_auroc(
        y, p1, p2, n_draws=n_balanced_draws, seed=seed + 2
    )

    prevalence = float(y.mean())
    base_ap = float(prevalence)  # random ranking AP ≈ prevalence for binary

    return {
        "n_images": int(n),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "prevalence": prevalence,
        "random_ap_baseline": base_ap,
        "stage2_file": stage2_csv,
        "stage1_file": stage1_csv,
        "auroc": {
            "stage2": auroc2,
            "stage1": auroc1,
            "paired_diff_stage2_minus_stage1": d_auroc,
            "significant_at_0.05": bool(
                d_auroc.get("lower") is not None
                and (d_auroc["lower"] > 0 or d_auroc["upper"] < 0)
            ),
        },
        "average_precision": {
            "stage2": ap2,
            "stage1": ap1,
            "paired_diff_stage2_minus_stage1": d_ap,
            "significant_at_0.05": bool(
                d_ap.get("lower") is not None and (d_ap["lower"] > 0 or d_ap["upper"] < 0)
            ),
            "note": "AP is prevalence-sensitive; compare also to random_ap_baseline ≈ prevalence.",
        },
        "balanced_subsample_auroc": balanced,
        "interpretation": balanced.get("interpretation")
        if balanced.get("ok")
        else "Balanced stress test unavailable",
        "note": (
            "DENTEX binary labels are heavily positive. Report AUROC + AP + balanced-subsample "
            "AUROC; do not report shipped-threshold accuracy/Dice as OOD claims."
        ),
    }


def run_and_save(
    project: str = ".",
    n_boot: int = 2000,
    n_balanced_draws: int = 1000,
) -> Dict[str, Any]:
    s2 = os.path.join(project, "results", "dentex_per_image_predictions.csv")
    s1 = os.path.join(project, "results", "dentex_baseline_predictions.csv")
    if not os.path.isfile(s2):
        s2 = os.path.join(project, "results", "stage2_dentex_per_image_predictions.csv")
    out = harden_dentex_ood(s2, s1, n_boot=n_boot, n_balanced_draws=n_balanced_draws)
    os.makedirs(os.path.join(project, "statistics"), exist_ok=True)
    path = os.path.join(project, "statistics", "dentex_ood_hardened.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Merge into p_values.json if present
    pv_path = os.path.join(project, "statistics", "p_values.json")
    if os.path.isfile(pv_path):
        with open(pv_path, "r", encoding="utf-8") as f:
            pv = json.load(f)
        pv["dentex_ood_hardened"] = out
        with open(pv_path, "w", encoding="utf-8") as f:
            json.dump(pv, f, indent=2)
    return out


if __name__ == "__main__":
    report = run_and_save()
    bal = report["balanced_subsample_auroc"]
    print("DENTEX hardened OOD")
    print(
        f"  prevalence={report['prevalence']:.3f} | n={report['n_images']} "
        f"(pos={report['n_positive']}, neg={report['n_negative']})"
    )
    print(
        f"  AUROC S2={report['auroc']['stage2']['point_estimate']:.4f} "
        f"S1={report['auroc']['stage1']['point_estimate']:.4f} "
        f"d={report['auroc']['paired_diff_stage2_minus_stage1']['mean_diff']:+.4f}"
    )
    print(
        f"  AP    S2={report['average_precision']['stage2']['point_estimate']:.4f} "
        f"S1={report['average_precision']['stage1']['point_estimate']:.4f} "
        f"(random≈{report['random_ap_baseline']:.3f})"
    )
    if bal.get("ok"):
        d = bal["paired_auroc_diff_stage2_minus_stage1_balanced"]
        print(
            f"  Balanced AUROC draws n={bal['balanced_n_per_draw']}: "
            f"d={d['mean']:+.4f} [{d['lower']:+.4f}, {d['upper']:+.4f}]"
        )
        print(f"  -> {bal['interpretation']}")
    print("Saved statistics/dentex_ood_hardened.json")
