# -*- coding: utf-8 -*-
"""Standalone faithfulness hardening (refit + optional ablation). Works CPU/GPU."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from dental_dataset import DentalDataset
from core_modules import SwinGazeModel, calibrate_decision_threshold
from faithfulness_metrics import (
    per_image_aopc_and_alignment,
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    aopc_auc_from_drops,
)
from attribution_error_taxonomy import classify_error, summarize_taxonomy
import torch.nn.functional as F


def resolve_stage2_ckpt(project="."):
    candidates = [
        os.path.join(project, "checkpoints", "stage2_refit_best.pth"),
        os.path.join(project, "checkpoints", "stage2_best.pth"),
    ]
    metrics_path = os.path.join(project, "results", "stage2_cv_fold_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            best = max(rows, key=lambda r: float(r.get("val_AUROC", -1)))
            ck = best.get("checkpoint") or f"checkpoints/stage2_gaze_fold{best['fold']}_best.pth"
            candidates.insert(1, ck if os.path.isabs(ck) else os.path.join(project, ck))
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("No Stage-2 checkpoint")


def resolve_ablation_ckpt(project="."):
    metrics_path = os.path.join(project, "results", "stage2_part1_negative_report.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        rows = rep.get("cv_ablation_fold_metrics") or []
        if rows:
            best = max(rows, key=lambda r: float(r.get("val_AUROC", -1)))
            ck = best.get("checkpoint") or f"checkpoints/stage2_ablation_fold{best['fold']}_best.pth"
            p = ck if os.path.isabs(ck) else os.path.join(project, ck)
            if os.path.isfile(p):
                return os.path.normpath(p), best
    for fold in range(5):
        p = os.path.join(project, "checkpoints", f"stage2_ablation_fold{fold}_best.pth")
        if os.path.isfile(p):
            return os.path.normpath(p), {"fold": fold}
    return None, None


def load_model(ckpt, CONFIG, device):
    m = SwinGazeModel(CONFIG).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def extract_last_attn(out):
    last = out.attentions[-1]
    return last.mean(dim=1).mean(dim=1).reshape(-1, 7, 7)


def eval_loader(net, loader, device, k_levels):
    rows_aopc, rows_align = [], []
    net.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", unit="batch"):
            out = per_image_aopc_and_alignment(
                net, batch, device, k_levels=k_levels, extract_last_attn=extract_last_attn
            )
            B = len(out["aopc_auc"])
            ids = out["ids"] if out["ids"] is not None else list(range(len(rows_aopc), len(rows_aopc) + B))
            for i in range(B):
                r = {"id": ids[i] if i < len(ids) else i, "aopc_auc": float(out["aopc_auc"][i])}
                for k in k_levels:
                    r[f"k_{int(k*100)}pct"] = float(out["drops"][f"k_{int(k*100)}pct"][i])
                rows_aopc.append(r)
                rows_align.append(
                    {
                        "id": r["id"],
                        "has_gaze": bool(out["has_gaze"][i]),
                        "nss": None if not np.isfinite(out["nss"][i]) else float(out["nss"][i]),
                        "cc": None if not np.isfinite(out["cc"][i]) else float(out["cc"][i]),
                    }
                )
    return pd.DataFrame(rows_aopc), pd.DataFrame(rows_align)


def paired_metric_diff(df_g, df_a, col, n_boot):
    m = df_g.merge(df_a, on="id", suffixes=("_g", "_a"))
    if col in ("nss", "cc"):
        m = m[m["has_gaze_g"] & m["has_gaze_a"]]
    a_vals, b_vals = [], []
    for _, row in m.iterrows():
        va, vb = row[f"{col}_a"], row[f"{col}_g"]
        if va is None or vb is None:
            continue
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        a_vals.append(float(va))
        b_vals.append(float(vb))
    return bootstrap_paired_diff_ci(a_vals, b_vals, n_boot=n_boot)


def run_taxonomy(model, loader, device):
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _, _, _ = model(batch["image"].to(device))
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_y.extend(batch["has_lesion_gt"].cpu().numpy().tolist())
            all_p.extend(probs.tolist())
    thr = float(calibrate_decision_threshold(np.asarray(all_y), np.asarray(all_p)))
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="taxonomy", unit="batch"):
            images = batch["image"].to(device)
            logits, _, last_attn, out = model(images)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            preds = (probs > thr).astype(int)
            ys = batch["has_lesion_gt"].cpu().numpy().astype(int)
            masks = batch["bbox_mask"].cpu().numpy()
            attn = (last_attn if last_attn is not None else extract_last_attn(out)).cpu().numpy()
            ids = batch.get("image_id", batch.get("id", list(range(len(ys)))))
            if torch.is_tensor(ids):
                ids = ids.tolist()
            for i in range(len(ys)):
                mask_i = masks[i][0] if masks[i].ndim == 3 else masks[i]
                a = attn[i]
                if a.shape != mask_i.shape[-2:]:
                    a = F.interpolate(
                        torch.from_numpy(a)[None, None].float(),
                        size=mask_i.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze().numpy()
                rec = classify_error(int(ys[i]), int(preds[i]), a, mask_i, tau=0.10, type_b_top_frac=0.30)
                rec["id"] = ids[i] if i < len(ids) else i
                rec["prob"] = float(probs[i])
                rec["threshold"] = thr
                rows.append(rec)
    summary = summarize_taxonomy(rows)
    summary["threshold"] = thr
    return pd.DataFrame(rows), summary


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    K_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.50]
    K_COLS = [f"k_{int(k*100)}pct" for k in K_LEVELS]
    N_BOOT = int(CONFIG.get("negative_result_bootstrap_n", 2000))

    ds = DentalDataset("splits/test_split.csv", PROJECT, image_root=PROJECT, augment=False)
    max_n = os.environ.get("FAITHFULNESS_MAX_N")
    if max_n:
        ds.df = ds.df.head(int(max_n)).reset_index(drop=True)
        print("capped to", len(ds))
    loader = DataLoader(ds, batch_size=4 if device.type == "cpu" else 16, shuffle=False, num_workers=0)

    ckpt = resolve_stage2_ckpt(PROJECT)
    print("gaze/refit:", ckpt)
    model = load_model(ckpt, CONFIG, device)
    ab_ckpt, ab_meta = resolve_ablation_ckpt(PROJECT)
    model_ab = load_model(ab_ckpt, CONFIG, device) if ab_ckpt else None
    if ab_ckpt:
        print("ablation:", ab_ckpt, ab_meta)

    os.makedirs("faithfulness", exist_ok=True)
    aopc_df, align_df = eval_loader(model, loader, device, K_LEVELS)
    aopc_df.to_csv("faithfulness/aopc_per_image.csv", index=False)
    align_df.to_csv("faithfulness/alignment_per_image.csv", index=False)

    mean_drops = aopc_df[K_COLS].mean(axis=0).values
    aopc_curve = float(aopc_auc_from_drops(K_LEVELS, mean_drops))
    aopc_ci = bootstrap_mean_ci(aopc_df["aopc_auc"].tolist(), n_boot=N_BOOT)
    nss_vals = [v for v in align_df.loc[align_df["has_gaze"], "nss"].tolist() if v is not None]
    cc_vals = [v for v in align_df.loc[align_df["has_gaze"], "cc"].tolist() if v is not None]
    nss_ci = bootstrap_mean_ci(nss_vals, n_boot=N_BOOT, seed=43)
    cc_ci = bootstrap_mean_ci(cc_vals, n_boot=N_BOOT, seed=44)

    report = {
        "checkpoint": ckpt,
        "n_test_images": int(len(aopc_df)),
        "n_with_gaze": int(align_df["has_gaze"].sum()),
        "k_levels": K_LEVELS,
        "AOPC_AUC_curve_mean": aopc_curve,
        "AOPC_naive_mean_drop": float(mean_drops.mean()),
        "AOPC_AUC_per_image_bootstrap": aopc_ci,
        "NSS_bootstrap": nss_ci,
        "CC_bootstrap": cc_ci,
        "mean_confidence_drops": mean_drops.tolist(),
        "ablation": None,
        "device": str(device),
    }
    pd.DataFrame([dict(zip(K_COLS, mean_drops))]).to_csv("faithfulness/aopc_curves.csv", index=False)

    if model_ab is not None:
        aopc_ab, align_ab = eval_loader(model_ab, loader, device, K_LEVELS)
        aopc_ab.to_csv("faithfulness/aopc_per_image_ablation.csv", index=False)
        align_ab.to_csv("faithfulness/alignment_per_image_ablation.csv", index=False)
        report["ablation"] = {
            "checkpoint": ab_ckpt,
            "meta": ab_meta,
            "AOPC_AUC_per_image_bootstrap": bootstrap_mean_ci(aopc_ab["aopc_auc"].tolist(), n_boot=N_BOOT, seed=48),
            "NSS_bootstrap": bootstrap_mean_ci(
                [v for v in align_ab.loc[align_ab["has_gaze"], "nss"].tolist() if v is not None],
                n_boot=N_BOOT,
                seed=49,
            ),
            "CC_bootstrap": bootstrap_mean_ci(
                [v for v in align_ab.loc[align_ab["has_gaze"], "cc"].tolist() if v is not None],
                n_boot=N_BOOT,
                seed=50,
            ),
            "paired_diff_gaze_minus_ablation": {
                "AOPC_AUC": paired_metric_diff(aopc_df, aopc_ab, "aopc_auc", N_BOOT),
                "NSS": paired_metric_diff(align_df, align_ab, "nss", N_BOOT),
                "CC": paired_metric_diff(align_df, align_ab, "cc", N_BOOT),
            },
        }

    tax_df, tax_sum = run_taxonomy(model, loader, device)
    tax_sum["checkpoint"] = ckpt
    tax_df.to_csv("faithfulness/error_taxonomy_per_image.csv", index=False)
    with open("faithfulness/error_taxonomy_summary.json", "w", encoding="utf-8") as f:
        json.dump(tax_sum, f, indent=2)
    report["taxonomy_counts"] = tax_sum.get("counts")

    with open("faithfulness/faithfulness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    status = {
        "real_aopc_done": True,
        "real_taxonomy_done": True,
        "checkpoint": ckpt,
        "ablation_checkpoint": ab_ckpt,
        "aopc_per_image_ci": aopc_ci,
        "nss_ci": nss_ci,
        "cc_ci": cc_ci,
        "n_test_images": int(len(aopc_df)),
        "taxonomy_counts": tax_sum.get("counts"),
        "device": str(device),
    }
    with open("faithfulness/faithfulness_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(json.dumps(status, indent=2))
    if report.get("ablation"):
        d = report["ablation"]["paired_diff_gaze_minus_ablation"]
        print("gaze - ablation:", {k: v.get("mean_diff") for k, v in d.items()})


if __name__ == "__main__":
    main()
