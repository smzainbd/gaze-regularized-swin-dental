"""
Stage-2 full-pool refit (NB05): train on all train_split images (e.g. 900),
then evaluate once on held-out test (secondary metric — CV remains primary).
"""
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core_modules import (
    classification_loss,
    compute_gaze_kl_loss,
    segmentation_aux_loss,
    pseudo_label_loss,
    calibrate_decision_threshold,
    compute_metrics,
)
from dental_dataset import build_refit_dataloaders, verify_tufts_anomaly_masks, DentalDataset
from negative_result_stats import bootstrap_ci, summarize_negative_detection_result
from stage2_train_utils import (
    apply_stage2_config,
    build_model_and_optimizer,
    build_pseudo_label_map,
    evaluate_model_on_loader,
    optimizer_steps_per_epoch,
    stage2_learning_rate,
    try_resume_checkpoint,
)
from torch.utils.data import DataLoader


def median_best_epoch_from_cv(
    project_root: str,
    num_folds: int = 5,
    run_tag: str = "gaze",
) -> Optional[int]:
    """Median of per-fold best val-AUROC epochs from NB04 histories."""
    epochs: List[int] = []
    for fold in range(int(num_folds)):
        path = os.path.join(project_root, "results", f"stage2_{run_tag}_fold{fold}_history.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        best_ep, best_au = None, -1.0
        for rec in hist:
            if "val_AUROC" not in rec:
                continue
            au = float(rec["val_AUROC"])
            if au > best_au:
                best_au = au
                best_ep = int(rec["epoch"])
        if best_ep is not None:
            epochs.append(best_ep)
    if not epochs:
        return None
    return int(np.median(epochs))


def run_stage2_refit(
    config: Dict,
    project_root: str,
    image_root: str,
    device: torch.device,
    smoke_test: bool = False,
    resume: bool = True,
    on_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Train Stage-2 gaze model on the full CV pool (train_split), fixed epoch budget
    from CV median best epoch (or config override). One held-out test eval at end.
    """
    if not on_gpu:
        print("SKIPPING refit — CPU dry run.")
        return {"train_done": False, "checkpoint": None}

    os.chdir(project_root)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    cfg = apply_stage2_config(dict(config))
    ok, msg = verify_tufts_anomaly_masks(
        project_root, image_root=image_root, csv_relpath="splits/train_split.csv",
    )
    print(f"Anomaly mask check: {msg}")
    if not ok:
        raise RuntimeError(msg)

    loaders = build_refit_dataloaders(
        cfg, project_root, image_root=image_root, smoke_test=smoke_test,
    )
    train_loader = loaders["tdd_train"]
    test_loader = loaders["tdd_test"]
    n_train = int(loaders["n_train"])
    print(f"Refit train pool: {n_train} | held-out test: {loaders['n_test']}")

    # Epoch budget: config override > CV median > half of stage2 epochs
    num_folds = int(cfg.get("num_folds", 5))
    cv_med = median_best_epoch_from_cv(project_root, num_folds=num_folds, run_tag="gaze")
    if cfg.get("refit_epochs") is not None:
        num_epochs = int(cfg["refit_epochs"])
        epoch_source = "config.refit_epochs"
    elif cv_med is not None:
        # histories store 0-based epoch index; train that many + 1
        num_epochs = int(cv_med) + 1
        epoch_source = f"cv_median_best_epoch={cv_med}"
    else:
        num_epochs = max(5, int(cfg.get("num_epochs_stage2", 35)) // 2)
        epoch_source = "fallback_half_stage2_epochs"
    if smoke_test:
        num_epochs = 1
    print(f"Refit epochs={num_epochs} ({epoch_source})")

    accum = int(cfg["chosen_grad_accum"])
    warmup_epochs = int(cfg.get("gaze_warmup_epochs", 8))
    gaze_lambda_max = float(cfg.get("gaze_lambda", 0.05))
    aux_weight = float(cfg.get("aux_weight", 0.35))
    aux_dice_w = float(cfg.get("aux_dice_weight", 0.5))
    pseudo_w = float(cfg.get("pseudo_label_weight", 0.25))
    label_smooth = float(cfg.get("label_smooth", 0.05))
    lr = stage2_learning_rate(cfg)
    save_interval = int(cfg.get("save_interval", 5))

    train_df = train_loader.dataset.df
    n_pos = int(train_df["has_lesion_gt"].sum())
    n_neg = int((train_df["has_lesion_gt"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    nw = 0 if os.name == "nt" else 2
    pseudo_loader = DataLoader(
        DentalDataset(
            os.path.join(project_root, "splits/train_split.csv"),
            project_root, augment=False, image_root=image_root, df=train_df,
        ),
        batch_size=cfg["chosen_batch_size"], shuffle=False,
        num_workers=nw, pin_memory=device.type == "cuda",
    )

    model, optimizer = build_model_and_optimizer(cfg, device)
    pseudo_map = build_pseudo_label_map(cfg, pseudo_loader, device)

    steps_per_epoch = optimizer_steps_per_epoch(train_loader, accum)
    total_steps = max(steps_per_epoch * num_epochs, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr / 10, lr, lr], total_steps=total_steps, pct_start=0.1,
    )

    last_ckpt = "checkpoints/stage2_refit_last.pth"
    best_ckpt = "checkpoints/stage2_refit_best.pth"
    history_path = "results/stage2_refit_history.json"
    start_epoch, _, _ = try_resume_checkpoint(
        last_ckpt, model, optimizer, scheduler, device, resume,
    )

    history: List[Dict] = []
    if start_epoch > 0 and os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    def _opt_step():
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], 1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    model.train()
    for epoch in range(start_epoch, num_epochs):
        gaze_lam = gaze_lambda_max * min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        optimizer.zero_grad()
        run_loss = run_main = run_kl = run_aux = run_pseudo = 0.0
        n_seen = 0
        pbar = tqdm(
            train_loader, desc=f"REFIT E{epoch:02d}/{num_epochs - 1}",
            unit="batch", leave=True, dynamic_ncols=True,
        )
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["has_lesion_gt"].to(device).float()
            targets_smooth = targets * (1 - label_smooth) + label_smooth * 0.5
            has_gaze = batch["has_gaze_map"].to(device).bool()
            gaze_maps = batch["gaze_map"].to(device, non_blocking=True)
            anomaly_masks = batch["bbox_mask"].to(device, non_blocking=True)
            image_ids = batch.get("image_id", [])

            logits, seg_map, last_attn, _ = model(images)
            main_loss = classification_loss(logits, targets_smooth, cfg, pos_weight=pos_weight)
            aux_loss = aux_weight * segmentation_aux_loss(
                seg_map, anomaly_masks, pos_weight=pos_weight, dice_weight=aux_dice_w,
            )
            gaze_kl = compute_gaze_kl_loss("DUAL_ARM", last_attn, gaze_maps, has_gaze)
            pl_loss = pseudo_label_loss(logits, image_ids, pseudo_map, device, weight=pseudo_w)
            loss = main_loss + gaze_lam * gaze_kl + aux_loss + pl_loss

            (loss / accum).backward()
            run_loss += loss.item()
            run_main += main_loss.item()
            run_kl += gaze_kl.item()
            run_aux += aux_loss.item()
            run_pseudo += pl_loss.item()
            n_seen += 1
            if (step + 1) % accum == 0:
                _opt_step()
            pbar.set_postfix(loss=f"{run_loss / max(n_seen, 1):.4f}")

        pbar.close()
        if n_seen % accum != 0:
            _opt_step()

        rec = {
            "epoch": epoch,
            "loss": run_loss / max(n_seen, 1),
            "main_loss": run_main / max(n_seen, 1),
            "gaze_kl": run_kl / max(n_seen, 1),
            "aux_loss": run_aux / max(n_seen, 1),
            "pseudo_loss": run_pseudo / max(n_seen, 1),
            "gaze_lambda": gaze_lam,
            "n_train": n_train,
        }
        history.append(rec)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if ((epoch + 1) % save_interval == 0) or (epoch == num_epochs - 1):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "n_train": n_train,
                    "refit": True,
                    "epoch_source": epoch_source,
                },
                last_ckpt,
            )
        print(
            f"REFIT E{epoch:02d}: loss={rec['loss']:.4f} "
            f"(main={rec['main_loss']:.4f}, kl={rec['gaze_kl']:.4f}, aux={rec['aux_loss']:.4f})"
        )

    # No val early-stop: final weights are the refit model
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": num_epochs - 1,
            "n_train": n_train,
            "refit": True,
            "epoch_source": epoch_source,
            "role": "full_pool_refit_for_faithfulness_secondary_test",
        },
        best_ckpt,
    )
    print(f"Saved {best_ckpt}")

    # Secondary held-out test (not for model selection)
    test_m, test_t, test_p, test_probs = evaluate_model_on_loader(model, test_loader, device)
    # Stage-1 baseline on same test for paired summary
    s1_cfg = apply_stage2_config(dict(config))
    s1_cfg["cls_seg_blend"] = 0.0
    s1_model, _ = build_model_and_optimizer(s1_cfg, device, load_stage1=True)
    s1_m, s1_t, _, s1_probs = evaluate_model_on_loader(s1_model, test_loader, device)
    del s1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_boot = int(cfg.get("negative_result_bootstrap_n", 2000))
    neg = summarize_negative_detection_result(
        {"AUROC": s1_m["AUROC"]}, {"AUROC": test_m["AUROC"]},
        test_t, s1_probs, test_probs, n_boot=n_boot,
    )

    report = {
        "role": "secondary_refit_eval",
        "note": (
            "Trained on full train_split (CV pool). Held-out test evaluated once at the end. "
            "Primary detection claims must use NB04 5-fold CV, not this table alone."
        ),
        "n_train": n_train,
        "num_epochs": num_epochs,
        "epoch_source": epoch_source,
        "refit_test_metrics": test_m,
        "stage1_test_metrics": s1_m,
        "negative_summary": neg,
        "checkpoint": best_ckpt,
    }
    report_path = os.path.join(project_root, "results", "stage2_refit_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    pd.DataFrame({
        "has_lesion_gt": test_t,
        "prob_stage1": s1_probs,
        "prob_stage2_refit": test_probs,
    }).to_csv(
        os.path.join(project_root, "results", "stage2_refit_test_predictions.csv"),
        index=False,
    )

    print("\n=== Refit secondary TEST (not primary CV claim) ===")
    print(f"  Stage1 AUROC: {s1_m['AUROC']:.4f}")
    print(f"  Refit  AUROC: {test_m['AUROC']:.4f}")
    print(f"  {neg.get('interpretation', '')}")
    print(f"Saved {report_path}")

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "train_done": True,
        "checkpoint": best_ckpt,
        "report": report,
        "report_path": report_path,
        "test_metrics": test_m,
    }
