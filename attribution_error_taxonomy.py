"""
Error-attribution taxonomy adapted from the CXR pre-deployment audit
(Type A / Type B false positives; attentive vs blind false negatives).

Dental / binary lesion mapping
------------------------------
CXR paper (multi-label):
  - Type A FP: predict disease on a radiologically healthy image
  - Type B FP: predict class c on a pathological image whose GT does not include c
  - FN attentive vs blind: IG peak/mass vs consensus box

Tufts binary (has_lesion_gt + anomaly mask):
  - Type A FP: y=0, pred=1  (hallucination) — same as CXR
  - Type B (adapted): y=1, pred=1, but attribution mass inside lesion mask
    is NOT above the analytic spatial null (right label, wrong place).
    This is NOT a label-false-positive; it is the binary stand-in for CXR
    "looks at pathology / anatomy but wrong decision evidence".
    Report separately from FP rate. On multi-label DENTEX, prefer true
    wrong-class Type B when disease IDs exist.
  - FN attentive: y=1, pred=0, and (peak in lesion OR mass_in_lesion >= tau)
  - FN blind: y=1, pred=0, otherwise

Primary maps for dental NB06: Swin last-attn upsampled (same as AOPC).
Optional: pass Integrated Gradients maps later without changing bucket logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _to_2d_float(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 3:
        # [1,H,W] or [C,H,W] -> max over channel
        a = a.max(axis=0) if a.shape[0] <= 4 else a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {a.shape}")
    return a


def _normalize_map(attr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.clip(_to_2d_float(attr), 0.0, None)
    s = float(a.sum())
    if s < eps:
        return np.zeros_like(a)
    return a / s


def _binarize_mask(mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    m = _to_2d_float(mask)
    return (m > thr).astype(np.float64)


def top_mass_mask(attr: np.ndarray, mass_frac: float = 0.3) -> np.ndarray:
    """Binary mask of pixels covering the top `mass_frac` of attribution mass."""
    a = _normalize_map(attr)
    flat = a.ravel()
    if flat.sum() <= 0:
        return np.zeros_like(a, dtype=np.float64)
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    keep = order[csum <= max(mass_frac, 1e-8)]
    if keep.size == 0:
        keep = order[:1]
    out = np.zeros_like(flat)
    out[keep] = 1.0
    return out.reshape(a.shape)


def analytic_null_mass(lesion_mask: np.ndarray) -> float:
    """Expected mass inside lesion if attribution were spatially uniform."""
    m = _binarize_mask(lesion_mask)
    h, w = m.shape
    return float(m.sum() / max(h * w, 1))


def mass_in_mask(attr: np.ndarray, lesion_mask: np.ndarray, top_frac: Optional[float] = None) -> float:
    """
    Fraction of attribution mass inside lesion.
    If top_frac is set, first restrict to top-mass mask (CXR Type B used top 30%).
    """
    a = _normalize_map(attr)
    m = _binarize_mask(lesion_mask)
    if m.shape != a.shape:
        # nearest resize via simple zoom indices
        ys = (np.linspace(0, m.shape[0] - 1, a.shape[0])).astype(int)
        xs = (np.linspace(0, m.shape[1] - 1, a.shape[1])).astype(int)
        m = m[ys][:, xs]
    if top_frac is not None:
        tm = top_mass_mask(a, top_frac)
        denom = float(tm.sum()) + 1e-12
        return float((tm * m).sum() / denom)
    return float((a * m).sum())


def peak_in_mask(attr: np.ndarray, lesion_mask: np.ndarray) -> bool:
    a = _normalize_map(attr)
    m = _binarize_mask(lesion_mask)
    if m.shape != a.shape:
        ys = (np.linspace(0, m.shape[0] - 1, a.shape[0])).astype(int)
        xs = (np.linspace(0, m.shape[1] - 1, a.shape[1])).astype(int)
        m = m[ys][:, xs]
    yy, xx = np.unravel_index(int(np.argmax(a)), a.shape)
    return bool(m[yy, xx] > 0.5)


def classify_error(
    y_true: int,
    y_pred: int,
    attr_map: np.ndarray,
    lesion_mask: np.ndarray,
    tau: float = 0.10,
    type_b_top_frac: float = 0.30,
) -> Dict[str, Any]:
    """
    Classify one image into FP-A / FP-B(adapted) / FN-attentive / FN-blind / other.

    Returns a dict with bucket name and spatial diagnostics.
    """
    y_true = int(y_true)
    y_pred = int(y_pred)
    real_mass = mass_in_mask(attr_map, lesion_mask, top_frac=type_b_top_frac)
    null_mass = analytic_null_mass(lesion_mask)
    delta = real_mass - null_mass
    peak = peak_in_mask(attr_map, lesion_mask)
    mass_full = mass_in_mask(attr_map, lesion_mask, top_frac=None)

    out: Dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "peak_in_lesion": peak,
        "mass_in_lesion": float(mass_full),
        "top_mass_in_lesion": float(real_mass),
        "null_mass": float(null_mass),
        "delta_mass_minus_null": float(delta),
        "bucket": "other",
    }

    # --- False positives ---
    if y_true == 0 and y_pred == 1:
        out["bucket"] = "fp_type_a_hallucination"
        return out

    # --- Type B adapted (right label, attribution not above null inside lesion) ---
    if y_true == 1 and y_pred == 1:
        if delta <= 0.0:
            out["bucket"] = "fp_type_b_misattribution"  # adapted; not a label-FP
        else:
            out["bucket"] = "tp_localized"
        return out

    # --- False negatives ---
    if y_true == 1 and y_pred == 0:
        attentive = peak or (mass_full >= tau)
        out["bucket"] = "fn_attentive" if attentive else "fn_blind"
        return out

    # TN
    if y_true == 0 and y_pred == 0:
        out["bucket"] = "tn"
        return out

    out["bucket"] = "other"
    return out


def summarize_taxonomy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-image classify_error rows into rates + counts."""
    n = len(rows)
    counts: Dict[str, int] = {}
    for r in rows:
        b = str(r.get("bucket", "other"))
        counts[b] = counts.get(b, 0) + 1

    n_healthy = sum(1 for r in rows if int(r["y_true"]) == 0)
    n_lesion = sum(1 for r in rows if int(r["y_true"]) == 1)
    n_fn = counts.get("fn_attentive", 0) + counts.get("fn_blind", 0)
    n_tp = counts.get("tp_localized", 0) + counts.get("fp_type_b_misattribution", 0)

    type_a = counts.get("fp_type_a_hallucination", 0)
    type_b = counts.get("fp_type_b_misattribution", 0)
    fn_att = counts.get("fn_attentive", 0)
    fn_blind = counts.get("fn_blind", 0)

    # Spatial deltas for Type B rows (CXR-style)
    type_b_deltas = [
        float(r["delta_mass_minus_null"])
        for r in rows
        if r.get("bucket") == "fp_type_b_misattribution"
    ]

    return {
        "n_images": n,
        "n_healthy": n_healthy,
        "n_lesion": n_lesion,
        "counts": counts,
        "fp_type_a_rate_among_healthy": (type_a / n_healthy) if n_healthy else None,
        "fp_type_b_rate_among_tp_label": (type_b / n_tp) if n_tp else None,
        "fn_attentive_rate_among_fn": (fn_att / n_fn) if n_fn else None,
        "fn_blind_rate_among_fn": (fn_blind / n_fn) if n_fn else None,
        "fp_type_a_count": type_a,
        "fp_type_b_misattribution_count": type_b,
        "fn_attentive_count": fn_att,
        "fn_blind_count": fn_blind,
        "type_b_delta_mean": float(np.mean(type_b_deltas)) if type_b_deltas else None,
        "definitions": {
            "fp_type_a": "y=0, pred=1 (hallucination) — same as CXR Type A",
            "fp_type_b_adapted": (
                "y=1, pred=1 but top-mass in lesion <= analytic null "
                "(right label, wrong place). Binary stand-in for CXR Type B; "
                "not counted in FP rate."
            ),
            "fn_attentive": "y=1, pred=0 and (peak in lesion OR mass>=tau)",
            "fn_blind": "y=1, pred=0 and not attentive",
        },
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Optional[Dict[str, float]]:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "lower": lo, "upper": hi, "n": int(len(arr))}
