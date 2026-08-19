# -*- coding: utf-8 -*-
"""Faithfulness helpers: per-image AOPC, bootstrap CIs, NSS/CC gaze alignment."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def aopc_auc_from_drops(k_levels: Sequence[float], drops: Sequence[float]) -> float:
    """Trapezoidal AOPC normalized by k-span (Samek-style area under drop curve)."""
    k = np.asarray(k_levels, dtype=float)
    d = np.asarray(drops, dtype=float)
    if len(k) < 2:
        return float(d.mean()) if len(d) else float("nan")
    return float(np.trapezoid(d, x=k) / (k[-1] - k[0]))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        samples.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "mean": float(arr.mean()),
        "lower": float(np.percentile(samples, lo)),
        "upper": float(np.percentile(samples, hi)),
        "n": int(len(arr)),
        "n_boot": int(n_boot),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def bootstrap_paired_diff_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap CI for mean(B - A)."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"mean_diff": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    diff0 = b - a
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(diff0), len(diff0))
        samples.append(float(diff0[idx].mean()))
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return {
        "mean_diff": float(diff0.mean()),
        "lower": float(np.percentile(samples, lo)),
        "upper": float(np.percentile(samples, hi)),
        "n": int(len(diff0)),
        "n_boot": int(n_boot),
    }


def _to_numpy_map(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    return x


def upsample_map(attn: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(_to_numpy_map(attn))[None, None].float()
    u = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return u.squeeze().numpy()


def normalize_map(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = _to_numpy_map(m)
    m = np.maximum(m, 0.0)
    s = m.sum()
    if s < eps:
        return np.zeros_like(m)
    return m / s


def nss_score(attention: np.ndarray, gaze: np.ndarray, eps: float = 1e-8) -> float:
    """
    Normalized Scanpath Saliency: mean z-scored attention at gaze-mass locations.
    Gaze treated as a continuous fixation density (thresholded soft mass).
    """
    att = _to_numpy_map(attention)
    gz = _to_numpy_map(gaze)
    if att.shape != gz.shape:
        att = upsample_map(att, gz.shape[-2:])
    mu, sd = float(att.mean()), float(att.std())
    if sd < eps:
        return 0.0
    z = (att - mu) / (sd + eps)
    w = np.maximum(gz, 0.0)
    if w.sum() < eps:
        return float("nan")
    w = w / w.sum()
    return float((z * w).sum())


def cc_score(attention: np.ndarray, gaze: np.ndarray, eps: float = 1e-8) -> float:
    """Pearson correlation between attention and gaze maps."""
    att = _to_numpy_map(attention).ravel()
    gz = _to_numpy_map(gaze).ravel()
    if att.size != gz.size:
        # caller should upsample; defensive
        side = int(np.sqrt(gz.size))
        att = upsample_map(att.reshape(int(np.sqrt(att.size)), -1), (side, side)).ravel()
    att = att - att.mean()
    gz = gz - gz.mean()
    denom = float(np.sqrt((att * att).sum() * (gz * gz).sum()))
    if denom < eps:
        return 0.0
    return float((att * gz).sum() / denom)


def mask_top_k_attention(images: torch.Tensor, attention_maps: torch.Tensor, k: float) -> torch.Tensor:
    B, C, H, W = images.shape
    attn_resized = F.interpolate(
        attention_maps.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)
    attn_flat = attn_resized.reshape(B, -1)
    k_count = max(1, int(k * attn_flat.shape[1]))
    topk_vals, _ = torch.topk(attn_flat, k_count, dim=1)
    threshold = topk_vals[:, -1].reshape(B, 1, 1)
    keep_mask = (attn_resized < threshold).float().unsqueeze(1)
    return images * keep_mask


@torch.no_grad()
def per_image_aopc_and_alignment(
    model,
    batch: dict,
    device,
    k_levels=(0.05, 0.10, 0.15, 0.20, 0.50),
    extract_last_attn=None,
) -> Dict[str, np.ndarray]:
    """
    Returns per-image arrays:
      aopc_auc, drops[k], nss, cc, has_gaze
    """
    images = batch["image"].to(device)
    logits, _, last_attn, out = model(images)
    probs = torch.sigmoid(logits.squeeze(-1))
    if last_attn is None:
        if extract_last_attn is None:
            raise RuntimeError("last_attn missing and no extract_last_attn")
        attn = extract_last_attn(out)
    else:
        attn = last_attn

    B = images.shape[0]
    drops = {f"k_{int(k*100)}pct": np.zeros(B, dtype=np.float64) for k in k_levels}
    for k in k_levels:
        masked = mask_top_k_attention(images, attn, k=k)
        logits_m, _, _, _ = model(masked)
        drop = (probs - torch.sigmoid(logits_m.squeeze(-1))).detach().cpu().numpy()
        drops[f"k_{int(k*100)}pct"] = drop.astype(np.float64)

    aopc = np.zeros(B, dtype=np.float64)
    for i in range(B):
        dvec = [drops[f"k_{int(k*100)}pct"][i] for k in k_levels]
        aopc[i] = aopc_auc_from_drops(k_levels, dvec)

    nss = np.full(B, np.nan)
    cc = np.full(B, np.nan)
    has_gaze = np.zeros(B, dtype=bool)
    if "gaze_map" in batch:
        gaze = batch["gaze_map"]
        if "has_gaze_map" in batch:
            has_gaze = batch["has_gaze_map"].cpu().numpy().astype(bool)
        else:
            has_gaze = np.ones(B, dtype=bool)
        attn_np = attn.detach().cpu().numpy()
        for i in range(B):
            if not has_gaze[i]:
                continue
            g = gaze[i].detach().cpu().numpy()
            if g.ndim == 3:
                g = g[0]
            a = attn_np[i]
            if a.shape != g.shape:
                a = upsample_map(a, g.shape[-2:])
            nss[i] = nss_score(a, g)
            cc[i] = cc_score(a, g)

    ids = None
    for key in ("id", "image_id"):
        if key in batch:
            ids = batch[key]
            break
    if ids is not None and torch.is_tensor(ids):
        ids = ids.cpu().tolist()

    return {
        "aopc_auc": aopc,
        "drops": drops,
        "nss": nss,
        "cc": cc,
        "has_gaze": has_gaze,
        "ids": ids,
        "probs": probs.detach().cpu().numpy(),
    }
