"""Statistical helpers for Part 1 negative-result reporting (CV, bootstrap, power)."""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        v = float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = _safe_auroc,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap mean and percentile CI for a scalar metric."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    if not np.isfinite(point):
        point = 0.5
    if n < 2:
        return {"mean": float(point), "lower": float(point), "upper": float(point), "n": n}

    rng = np.random.default_rng(seed)
    samples: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            v = float(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
        if np.isfinite(v):
            samples.append(v)
    if not samples:
        return {"mean": float(point), "lower": float(point), "upper": float(point), "n": n}
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "mean": float(np.mean(samples)),
        "lower": float(np.percentile(samples, lo)),
        "upper": float(np.percentile(samples, hi)),
        "n": n,
        "n_boot_effective": len(samples),
        "point_estimate": float(point),
    }


def bootstrap_paired_diff_ci(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap CI for AUROC(B) - AUROC(A) on the same images."""
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    n = len(y_true)
    if n < 2:
        d = _safe_auroc(y_true, probs_b) - _safe_auroc(y_true, probs_a)
        return {"mean_diff": d, "lower": d, "upper": d, "n": n}

    point_a = _safe_auroc(y_true, probs_a)
    point_b = _safe_auroc(y_true, probs_b)
    point_d = (0.5 if not np.isfinite(point_b) else point_b) - (0.5 if not np.isfinite(point_a) else point_a)

    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = _safe_auroc(y_true[idx], probs_a[idx])
        b = _safe_auroc(y_true[idx], probs_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    if not diffs:
        return {"mean_diff": float(point_d), "lower": float(point_d), "upper": float(point_d), "n": n}
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "mean_diff": float(np.mean(diffs)),
        "lower": float(np.percentile(diffs, lo)),
        "upper": float(np.percentile(diffs, hi)),
        "n": n,
        "n_boot_effective": len(diffs),
    }


def aggregate_fold_metrics(
    fold_rows: Sequence[Dict],
    metric_keys: Sequence[str] = ("AUROC", "Dice", "F1", "Recall"),
) -> Dict[str, Dict[str, float]]:
    """Mean ± std across folds for each metric key."""
    out: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in fold_rows if key in r and r[key] is not None]
        if not vals:
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": vals,
            "n_folds": len(vals),
        }
    return out


def power_analysis_auroc_sample_size(
    delta: float = 0.05,
    null_auroc: float = 0.58,
    alpha: float = 0.05,
    power: float = 0.8,
    prevalence: float = 0.34,
) -> Dict[str, float]:
    """
    Approximate sample size to detect AUROC improvement of `delta` (Hanley–McNeil style).
    Returns per-group n (total ≈ 2n for balanced comparison on same set).
    """
    delta = float(delta)
    null_auroc = float(null_auroc)
    alt_auroc = min(max(null_auroc + delta, 0.0), 1.0)
    p = float(prevalence)
    # Variance of AUC under null (Hanley & McNeil 1982 approximation)
    q1 = null_auroc / (2 - null_auroc)
    q2 = (2 * null_auroc ** 2) / (1 + null_auroc)
    var_null = (
        null_auroc * (1 - null_auroc)
        + (p - 1) * (q1 - null_auroc ** 2)
        + (1 - p) * (q2 - null_auroc ** 2)
    )
    q1a = alt_auroc / (2 - alt_auroc)
    q2a = (2 * alt_auroc ** 2) / (1 + alt_auroc)
    var_alt = (
        alt_auroc * (1 - alt_auroc)
        + (p - 1) * (q1a - alt_auroc ** 2)
        + (1 - p) * (q2a - alt_auroc ** 2)
    )
    z_alpha = 1.96 if alpha == 0.05 else 1.645
    z_beta = 0.84 if power == 0.8 else 1.28
    se = math.sqrt(max(var_null + var_alt, 1e-9))
    n_raw = ((z_alpha + z_beta) ** 2 * se) / max(delta ** 2, 1e-9)
    if not math.isfinite(n_raw) or n_raw <= 0:
        n_required = 0
    else:
        n_required = int(math.ceil(n_raw))
    return {
        "delta_auroc": delta,
        "null_auroc": null_auroc,
        "alt_auroc": alt_auroc,
        "prevalence": p,
        "alpha": alpha,
        "power": power,
        "approx_n_per_group": n_required,
        "approx_total_same_set": n_required,
        "note": "Approximate; use bootstrap CI on actual CV predictions for reporting.",
    }


def summarize_negative_detection_result(
    stage1_metrics: Dict,
    stage2_metrics: Dict,
    y_true: np.ndarray,
    probs_stage1: np.ndarray,
    probs_stage2: np.ndarray,
    n_boot: int = 2000,
    power_delta: float = 0.05,
) -> Dict:
    """Full Part 1 negative-result summary for one eval split."""
    s1_ci = bootstrap_ci(y_true, probs_stage1, n_boot=n_boot)
    s2_ci = bootstrap_ci(y_true, probs_stage2, n_boot=n_boot)
    diff_ci = bootstrap_paired_diff_ci(y_true, probs_stage1, probs_stage2, n_boot=n_boot)
    null_auroc = float(s1_ci.get("point_estimate", s1_ci["mean"]))
    if not np.isfinite(null_auroc):
        null_auroc = float(stage1_metrics.get("AUROC", 0.58) or 0.58)
    power = power_analysis_auroc_sample_size(delta=power_delta, null_auroc=null_auroc)
    significant_improvement = bool(diff_ci["lower"] > 0)
    return {
        "stage1_auroc": stage1_metrics.get("AUROC"),
        "stage2_auroc": stage2_metrics.get("AUROC"),
        "stage1_bootstrap_auroc": s1_ci,
        "stage2_bootstrap_auroc": s2_ci,
        "paired_auroc_diff_stage2_minus_stage1": diff_ci,
        "significant_improvement": significant_improvement,
        "power_analysis": power,
        "interpretation": (
            "Stage 2 significantly better than Stage 1 transfer"
            if significant_improvement
            else "No significant Stage 2 improvement over Stage 1 (supports negative/null result)"
        ),
    }
