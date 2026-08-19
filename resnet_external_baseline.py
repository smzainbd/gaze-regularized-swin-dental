# -*- coding: utf-8 -*-
"""
External baseline: ImageNet ResNet-50 on the SAME Tufts splits as NB04/NB05.

Purpose (paper): show the gaze null is not an artifact of a broken stack.
This is NOT a gaze model and does NOT use gaze maps.

Modes:
  --mode cv      5-fold on train_split (primary table row; matches NB04 protocol)
  --mode refit   train on all 900, eval once on test_split (secondary; matches NB05 refit)
  --mode both    run cv then refit (default)

Outputs:
  results/resnet50_baseline_fold_metrics.json
  results/resnet50_baseline_cv_summary.json
  results/resnet50_baseline_refit_metrics.json
  results/resnet50_baseline_test_predictions.csv
  checkpoints/resnet50_fold{k}_best.pth
  checkpoints/resnet50_refit_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import models

from dental_dataset import DentalDataset
from negative_result_stats import bootstrap_ci, bootstrap_paired_diff_ci, aggregate_fold_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResNet50Binary(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.backbone = models.resnet50(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def _make_loader(df, csv_path, project, image_root, batch_size, augment, shuffle):
    ds = DentalDataset(csv_path, project, augment=augment, image_root=image_root, df=df)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    ), ds


@torch.no_grad()
def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray, List]:
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        y = batch["has_lesion_gt"].cpu().numpy().astype(int)
        ys.append(y)
        ps.append(prob)
        bid = batch.get("image_id", batch.get("id"))
        if torch.is_tensor(bid):
            bid = bid.tolist()
        if bid is None:
            bid = list(range(len(y)))
        ids.extend(list(bid))
    return np.concatenate(ys), np.concatenate(ps), ids


def eval_auroc(model, loader, device) -> Dict[str, float]:
    y, p, _ = collect_probs(model, loader, device)
    if len(np.unique(y)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y, p))
    # Youden threshold for Dice-ish reporting optional
    from core_modules import calibrate_decision_threshold, compute_dice

    thr = float(calibrate_decision_threshold(y, p))
    pred = (p > thr).astype(int)
    return {
        "AUROC": auroc,
        "Dice": float(compute_dice(pred, y)),
        "opt_threshold": thr,
        "n": int(len(y)),
        "prevalence": float(y.mean()),
    }


def train_one(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_pos_weight: float,
    ckpt_path: str,
    patience: int = 8,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = torch.tensor([class_pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_auroc = -1.0
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["has_lesion_gt"].float().to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_auroc(model, val_loader, device) if val_loader is not None else {"AUROC": float("nan")}
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(f"  epoch {epoch:02d} loss={row['train_loss']:.4f} val_AUROC={val_m.get('AUROC', float('nan')):.4f}")
        auroc = val_m.get("AUROC", float("nan"))
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = float(auroc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save({"model": best_state, "val_AUROC": best_auroc, "epoch": epoch}, ckpt_path)
        else:
            bad += 1
            if patience and bad >= patience:
                print(f"  early stop at epoch {epoch} (best val AUROC={best_auroc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_auroc


def run_cv(project, image_root, device, args, CONFIG) -> Dict:
    train_csv = os.path.join(project, "splits", "train_split.csv")
    test_csv = os.path.join(project, "splits", "test_split.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    if args.smoke:
        # tiny stratified heads
        train_df = (
            train_df.groupby("fold", group_keys=False)
            .apply(lambda g: g.head(max(8, len(g) // 10)))
            .reset_index(drop=True)
        )
        test_df = test_df.head(16).reset_index(drop=True)
        print("SMOKE: shrunk train/test")

    os.makedirs(os.path.join(project, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project, "results"), exist_ok=True)

    bs = int(args.batch_size or CONFIG.get("chosen_batch_size", 32))
    epochs = int(args.epochs or (3 if args.smoke else 25))
    lr = float(args.lr or 1e-4)
    wd = float(CONFIG.get("weight_decay", 1e-3))
    pos_w = float((CONFIG.get("class_weights") or [2.5])[0])
    seed = int(CONFIG.get("random_seed", 42))

    fold_rows = []
    folds = sorted(train_df["fold"].dropna().unique().astype(int).tolist())
    for fold in folds:
        print(f"\n=== ResNet-50 fold {fold} ===")
        set_seed(seed + fold)
        tr = train_df[train_df["fold"] != fold].copy()
        va = train_df[train_df["fold"] == fold].copy()
        train_loader, _ = _make_loader(tr, train_csv, project, image_root, bs, True, True)
        val_loader, _ = _make_loader(va, train_csv, project, image_root, bs, False, False)
        test_loader, _ = _make_loader(test_df, test_csv, project, image_root, bs, False, False)

        model = ResNet50Binary().to(device)
        ckpt = os.path.join(project, "checkpoints", f"resnet50_fold{fold}_best.pth")
        hist, best_val = train_one(
            model, train_loader, val_loader, device, epochs, lr, wd, pos_w, ckpt,
            patience=5 if args.smoke else 8,
        )
        test_m = eval_auroc(model, test_loader, device)
        row = {
            "fold": int(fold),
            "val_AUROC": float(best_val),
            "test_AUROC": float(test_m["AUROC"]),
            "test_Dice": float(test_m["Dice"]),
            "checkpoint": ckpt,
            "history_len": len(hist),
        }
        fold_rows.append(row)
        with open(os.path.join(project, "results", f"resnet50_fold{fold}_history.json"), "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
        print(f"  fold {fold}: val={row['val_AUROC']:.4f} test={row['test_AUROC']:.4f}")

    agg_val = aggregate_fold_metrics(fold_rows, metric_keys=("val_AUROC",))
    agg_test = aggregate_fold_metrics(
        [{"AUROC": r["test_AUROC"], "Dice": r["test_Dice"]} for r in fold_rows],
        metric_keys=("AUROC", "Dice"),
    )
    summary = {
        "model": "resnet50_imagenet",
        "role": "external_baseline",
        "protocol": "5-fold on train_split; held-out test each fold",
        "fold_metrics": fold_rows,
        "aggregate_val": agg_val,
        "aggregate_test": agg_test,
        "smoke": bool(args.smoke),
    }
    path = os.path.join(project, "results", "resnet50_baseline_cv_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(project, "results", "resnet50_baseline_fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_rows, f, indent=2)
    print("Saved", path)
    return summary


def run_refit(project, image_root, device, args, CONFIG) -> Dict:
    train_csv = os.path.join(project, "splits", "train_split.csv")
    test_csv = os.path.join(project, "splits", "test_split.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    if args.smoke:
        train_df = train_df.head(64).reset_index(drop=True)
        test_df = test_df.head(16).reset_index(drop=True)

    # Use fold 0 as internal val for early stopping (not the held-out test).
    va = train_df[train_df["fold"] == 0].copy()
    tr = train_df[train_df["fold"] != 0].copy()
    if args.smoke and len(va) < 4:
        va = train_df.tail(8).copy()
        tr = train_df.head(max(16, len(train_df) - 8)).copy()

    bs = int(args.batch_size or CONFIG.get("chosen_batch_size", 32))
    epochs = int(args.epochs or (3 if args.smoke else 25))
    lr = float(args.lr or 1e-4)
    wd = float(CONFIG.get("weight_decay", 1e-3))
    pos_w = float((CONFIG.get("class_weights") or [2.5])[0])
    set_seed(int(CONFIG.get("random_seed", 42)))

    train_loader, _ = _make_loader(tr, train_csv, project, image_root, bs, True, True)
    val_loader, _ = _make_loader(va, train_csv, project, image_root, bs, False, False)
    test_loader, _ = _make_loader(test_df, test_csv, project, image_root, bs, False, False)

    print("\n=== ResNet-50 refit-style (train≈720, val=fold0, test=100) ===")
    model = ResNet50Binary().to(device)
    ckpt = os.path.join(project, "checkpoints", "resnet50_refit_best.pth")
    hist, best_val = train_one(
        model, train_loader, val_loader, device, epochs, lr, wd, pos_w, ckpt,
        patience=5 if args.smoke else 8,
    )
    test_m = eval_auroc(model, test_loader, device)
    y, p, ids = collect_probs(model, test_loader, device)
    pred_df = pd.DataFrame({"id": ids, "has_lesion_gt": y.astype(int), "prob_resnet50": p})
    # Attach Stage-1 / Stage-2 probs if available for paired bootstrap
    for cand, col in (
        ("results/stage2_refit_test_predictions.csv", None),
        ("results/stage2_part1_test_predictions.csv", None),
    ):
        path = os.path.join(project, cand)
        if os.path.isfile(path):
            other = pd.read_csv(path)
            if "prob_stage1" in other.columns:
                pred_df["prob_stage1"] = other["prob_stage1"].to_numpy()[: len(pred_df)]
            s2col = next(
                (c for c in ("prob_stage2_refit", "prob_stage2_gaze", "prob_stage2") if c in other.columns),
                None,
            )
            if s2col:
                pred_df["prob_stage2"] = other[s2col].to_numpy()[: len(pred_df)]
            break

    pred_path = os.path.join(project, "results", "resnet50_baseline_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    n_boot = int(CONFIG.get("negative_result_bootstrap_n", 2000))
    report = {
        "model": "resnet50_imagenet",
        "role": "external_baseline_refit_style",
        "checkpoint": ckpt,
        "val_AUROC_fold0": float(best_val),
        "test_metrics": test_m,
        "test_bootstrap_auroc": bootstrap_ci(y, p, n_boot=n_boot),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(test_df)),
        "smoke": bool(args.smoke),
        "predictions_csv": pred_path,
    }
    if "prob_stage1" in pred_df.columns:
        # AUROC(resnet) - AUROC(stage1)
        report["paired_diff_resnet_minus_stage1"] = bootstrap_paired_diff_ci(
            y, pred_df["prob_stage1"].to_numpy(), p, n_boot=n_boot
        )
    if "prob_stage2" in pred_df.columns:
        # AUROC(stage2) - AUROC(resnet)
        report["paired_diff_stage2_minus_resnet"] = bootstrap_paired_diff_ci(
            y, p, pred_df["prob_stage2"].to_numpy(), n_boot=n_boot
        )

    out = os.path.join(project, "results", "resnet50_baseline_refit_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(project, "results", "resnet50_refit_history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print("Saved", out)
    print(f"ResNet test AUROC={test_m['AUROC']:.4f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cv", "refit", "both"), default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--project", default=None)
    args = ap.parse_args()

    project = os.path.abspath(args.project or os.path.dirname(__file__))
    os.chdir(project)
    with open(os.path.join(project, "config.json"), "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = project
    print("PROJECT", project, "| device", device, "| mode", args.mode, "| smoke", args.smoke)

    if args.mode in ("cv", "both"):
        run_cv(project, image_root, device, args, CONFIG)
    if args.mode in ("refit", "both"):
        run_refit(project, image_root, device, args, CONFIG)
    print("\nResNet external baseline DONE.")


if __name__ == "__main__":
    main()
