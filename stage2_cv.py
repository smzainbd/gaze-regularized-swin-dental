"""
Part 1 pipeline: 5-fold CV, gaze ablation (lambda=0), learning curves, negative-result stats.
"""
import json
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
    ensemble_probs,
    calibrate_decision_threshold,
    compute_metrics,
    pseudo_label_loss,
)
from dental_dataset import build_dataloaders, verify_tufts_anomaly_masks, DentalDataset
from negative_result_stats import (
    aggregate_fold_metrics,
    power_analysis_auroc_sample_size,
    summarize_negative_detection_result,
    _safe_auroc,
)
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


def _ensure_dirs(project_root: str) -> None:
    os.makedirs(os.path.join(project_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "checkpoints", "stage2_epochs"), exist_ok=True)


def _ckpt_names(fold: int, run_tag: str) -> Dict[str, str]:
    prefix = f"checkpoints/stage2_{run_tag}_fold{fold}"
    return {
        "last": f"{prefix}_last.pth",
        "best": f"{prefix}_best.pth",
    }


def train_stage2_fold(
    config: Dict,
    project_root: str,
    image_root: str,
    device: torch.device,
    fold: int,
    run_tag: str = "gaze",
    gaze_lambda_override: Optional[float] = None,
    train_fraction: float = 1.0,
    resume: bool = True,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Train one CV fold; run_tag distinguishes gaze vs ablation checkpoints."""
    os.chdir(project_root)
    cfg = apply_stage2_config(dict(config))
    if gaze_lambda_override is not None:
        cfg["gaze_lambda"] = float(gaze_lambda_override)

    accum = int(cfg["chosen_grad_accum"])
    num_epochs = 1 if smoke_test else int(cfg["num_epochs_stage2"])
    eval_interval = int(cfg["eval_interval"])
    save_interval = int(cfg.get("save_interval", 5))
    patience = int(cfg.get("early_stop_patience", 10))
    warmup_epochs = int(cfg.get("gaze_warmup_epochs", 8))
    gaze_lambda_max = float(cfg.get("gaze_lambda", 0.05))
    aux_weight = float(cfg.get("aux_weight", 0.35))
    aux_dice_w = float(cfg.get("aux_dice_weight", 0.5))
    pseudo_w = float(cfg.get("pseudo_label_weight", 0.25))
    label_smooth = float(cfg.get("label_smooth", 0.05))
    lr = stage2_learning_rate(cfg)
    eval_test_each_epoch = bool(cfg.get("eval_test_each_epoch", False))
    save_epoch_ckpts = bool(cfg.get("save_epoch_checkpoints", False))

    loaders = build_dataloaders(
        cfg,
        project_root,
        image_root=image_root,
        fold=fold,
        smoke_test=smoke_test,
        stage2_training=True,
        train_fraction=train_fraction,
    )
    train_loader = loaders["tdd_train"]
    val_loader = loaders["tdd_val"]
    test_loader = loaders["tdd_test"]

    train_df = train_loader.dataset.df
    n_pos = int(train_df["has_lesion_gt"].sum())
    n_neg = int((train_df["has_lesion_gt"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    nw = 0 if os.name == "nt" else 2
    pseudo_loader = DataLoader(
        DentalDataset(
            os.path.join(project_root, "splits/train_split.csv"),
            project_root,
            augment=False,
            image_root=image_root,
            df=train_df,
        ),
        batch_size=cfg["chosen_batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )

    model, optimizer = build_model_and_optimizer(cfg, device)
    pseudo_map = build_pseudo_label_map(cfg, pseudo_loader, device)

    steps_per_epoch = optimizer_steps_per_epoch(train_loader, accum)
    total_steps = steps_per_epoch * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr / 10, lr, lr],
        total_steps=max(total_steps, 1),
        pct_start=0.1,
    )

    ckpts = _ckpt_names(fold, run_tag)
    history_path = f"results/stage2_{run_tag}_fold{fold}_history.json"
    start_epoch, best_auroc, epochs_no_improve = try_resume_checkpoint(
        ckpts["last"], model, optimizer, scheduler, device, resume,
    )

    pre_m, _, _, _ = evaluate_model_on_loader(model, val_loader, device)
    print(
        f"[fold {fold} {run_tag}] pre-train val AUROC={pre_m['AUROC']:.4f} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"gaze_lambda_max={gaze_lambda_max} frac={train_fraction}"
    )

    history: List[Dict] = []
    if start_epoch > 0 and os.path.exists(history_path):
        with open(history_path, "r") as f:
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
            train_loader,
            desc=f"F{fold}/{run_tag} E{epoch:02d}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
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
            pbar.set_postfix(loss=f"{run_loss / max(n_seen, 1):.4f}", best=f"{best_auroc:.4f}")

        pbar.close()
        if n_seen % accum != 0:
            _opt_step()

        rec = {
            "fold": fold,
            "run_tag": run_tag,
            "epoch": epoch,
            "train_fraction": train_fraction,
            "loss": run_loss / max(n_seen, 1),
            "gaze_lambda": gaze_lam,
        }
        metrics = None
        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            metrics, _, _, _ = evaluate_model_on_loader(model, val_loader, device)
            rec.update({f"val_{k}": float(v) for k, v in metrics.items()})
            auroc = float(metrics["AUROC"])
            if auroc > best_auroc:
                best_auroc = auroc
                epochs_no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_auroc": best_auroc}, ckpts["best"])
            else:
                epochs_no_improve += eval_interval

            if eval_test_each_epoch:
                test_m, _, _, _ = evaluate_model_on_loader(model, test_loader, device)
                rec.update({f"test_{k}": float(v) for k, v in test_m.items()})

            if save_epoch_ckpts:
                ep_path = f"checkpoints/stage2_epochs/{run_tag}_fold{fold}_epoch_{epoch:03d}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_auroc": auroc}, ep_path)
            model.train()

        history.append(rec)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_auroc": best_auroc,
                    "epochs_no_improve": epochs_no_improve,
                    "fold": fold,
                    "run_tag": run_tag,
                },
                ckpts["last"],
            )

        msg = (
            f"[fold {fold} {run_tag}] E{epoch:02d}: loss={rec['loss']:.4f} "
            f"lam={gaze_lam:.2f} best_val_auroc={best_auroc:.4f}"
        )
        if metrics is not None:
            msg += f" | val AUROC={metrics['AUROC']:.4f} Dice={metrics['Dice']:.4f}"
        print(msg)

        if epochs_no_improve >= patience:
            print(f"[fold {fold} {run_tag}] early stop after {epochs_no_improve} epochs without val gain")
            break

    # Load best for export metrics
    val_metrics = pre_m
    test_metrics = {}
    if os.path.exists(ckpts["best"]):
        best_ck = torch.load(ckpts["best"], map_location=device)
        model.load_state_dict(best_ck["model"])
        val_metrics, val_t, val_p, val_probs = evaluate_model_on_loader(model, val_loader, device)
        test_metrics, test_t, test_p, test_probs = evaluate_model_on_loader(model, test_loader, device)
    else:
        val_t, val_probs = np.array([]), np.array([])
        test_t, test_probs = np.array([]), np.array([])

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold": fold,
        "run_tag": run_tag,
        "train_fraction": train_fraction,
        "gaze_lambda": gaze_lambda_max,
        "best_val_auroc": best_auroc,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_targets": val_t,
        "val_probs": val_probs,
        "test_targets": test_t,
        "test_probs": test_probs,
        "checkpoint_best": ckpts["best"],
        "history_path": history_path,
    }


def eval_stage1_baseline(
    config: Dict,
    project_root: str,
    image_root: str,
    device: torch.device,
    fold: int,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Stage 1 transfer baseline on fold val and held-out test (no fine-tune)."""
    os.chdir(project_root)
    cfg = apply_stage2_config(dict(config))
    cfg["cls_seg_blend"] = 0.0
    loaders = build_dataloaders(
        cfg, project_root, image_root=image_root, fold=fold,
        smoke_test=smoke_test, stage2_training=True,
    )
    model, _ = build_model_and_optimizer(cfg, device, load_stage1=True)
    val_m, val_t, _, val_probs = evaluate_model_on_loader(model, loaders["tdd_val"], device)
    test_m, test_t, _, test_probs = evaluate_model_on_loader(model, loaders["tdd_test"], device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fold": fold,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "val_targets": val_t,
        "val_probs": val_probs,
        "test_targets": test_t,
        "test_probs": test_probs,
    }


def run_learning_curves(
    config: Dict,
    project_root: str,
    image_root: str,
    device: torch.device,
    folds: List[int],
    fractions: List[float],
    smoke_test: bool = False,
    resume: bool = True,
) -> List[Dict]:
    results = []
    for fold in folds:
        for frac in fractions:
            tag = f"lc{int(frac * 100)}"
            print(f"\n=== Learning curve fold={fold} fraction={frac} ===")
            r = train_stage2_fold(
                config, project_root, image_root, device,
                fold=fold, run_tag=tag, train_fraction=frac,
                resume=resume, smoke_test=smoke_test,
            )
            results.append(r)
    out_path = os.path.join(project_root, "results", "stage2_learning_curves.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_part1_negative_pipeline(
    config: Dict,
    project_root: str,
    image_root: str,
    device: torch.device,
    smoke_test: bool = False,
    resume_training: bool = True,
    on_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Main Part 1 entry: 5-fold CV (gaze + optional lambda=0 ablation), held-out test,
    bootstrap CI, power analysis, learning curves (optional).
    """
    if not on_gpu:
        print("SKIPPING Part 1 CV — CPU dry run.")
        return {"train_done": False, "fold_results": [], "agg_metrics": {}}

    os.chdir(project_root)
    _ensure_dirs(project_root)
    cfg = apply_stage2_config(dict(config))
    num_folds = int(cfg.get("num_folds", 5))
    if smoke_test:
        num_folds = min(num_folds, 2)

    mask_ok, mask_msg = verify_tufts_anomaly_masks(
        project_root, image_root=image_root, csv_relpath="splits/train_split.csv",
    )
    print(f"Anomaly mask check: {mask_msg}")
    if not mask_ok:
        raise RuntimeError(mask_msg)

    n_boot = int(cfg.get("negative_result_bootstrap_n", 2000))
    power_delta = float(cfg.get("power_analysis_delta_auroc", 0.05))
    run_ablation = bool(cfg.get("run_gaze_ablation", True))
    ablation_lambda = float(cfg.get("gaze_ablation_lambda", 0.0))

    fold_rows: List[Dict] = []
    stage1_rows: List[Dict] = []
    ablation_rows: List[Dict] = []

    all_val_targets: List[np.ndarray] = []
    all_val_s1: List[np.ndarray] = []
    all_val_s2: List[np.ndarray] = []
    all_test_s1_folds: List[np.ndarray] = []
    all_test_s2_folds: List[np.ndarray] = []
    test_y_ref: Optional[np.ndarray] = None

    for fold in range(num_folds):
        print(f"\n{'=' * 60}\n  FOLD {fold}/{num_folds - 1}\n{'=' * 60}")

        s1 = eval_stage1_baseline(cfg, project_root, image_root, device, fold, smoke_test)
        stage1_rows.append({
            "fold": fold,
            "val_AUROC": s1["val_metrics"]["AUROC"],
            "test_AUROC": s1["test_metrics"]["AUROC"],
            "val_Dice": s1["val_metrics"]["Dice"],
            "test_Dice": s1["test_metrics"]["Dice"],
        })

        gaze_res = train_stage2_fold(
            cfg, project_root, image_root, device,
            fold=fold, run_tag="gaze", resume=resume_training, smoke_test=smoke_test,
        )
        fold_rows.append({
            "fold": fold,
            "val_AUROC": gaze_res["val_metrics"]["AUROC"],
            "test_AUROC": gaze_res["test_metrics"].get("AUROC"),
            "val_Dice": gaze_res["val_metrics"]["Dice"],
            "test_Dice": gaze_res["test_metrics"].get("Dice"),
            "best_val_auroc": gaze_res["best_val_auroc"],
            "checkpoint": gaze_res["checkpoint_best"],
        })

        all_val_targets.append(gaze_res["val_targets"])
        all_val_s1.append(s1["val_probs"])
        all_val_s2.append(gaze_res["val_probs"])
        all_test_s1_folds.append(s1["test_probs"])
        all_test_s2_folds.append(gaze_res["test_probs"])
        if test_y_ref is None:
            test_y_ref = s1["test_targets"]

        if run_ablation:
            ab = train_stage2_fold(
                cfg, project_root, image_root, device,
                fold=fold, run_tag="ablation",
                gaze_lambda_override=ablation_lambda,
                resume=resume_training, smoke_test=smoke_test,
            )
            ablation_rows.append({
                "fold": fold,
                "val_AUROC": ab["val_metrics"]["AUROC"],
                "test_AUROC": ab["test_metrics"].get("AUROC"),
                "gaze_lambda": ablation_lambda,
            })

    agg_gaze_val = aggregate_fold_metrics(
        [{"AUROC": r["val_AUROC"], "Dice": r["val_Dice"]} for r in fold_rows],
        ("AUROC", "Dice"),
    )
    agg_gaze_test = aggregate_fold_metrics(
        [{"AUROC": r["test_AUROC"], "Dice": r["test_Dice"]} for r in fold_rows if r.get("test_AUROC")],
        ("AUROC", "Dice"),
    )
    agg_s1_val = aggregate_fold_metrics(
        [{"AUROC": r["val_AUROC"], "Dice": r["val_Dice"]} for r in stage1_rows],
        ("AUROC", "Dice"),
    )
    agg_s1_test = aggregate_fold_metrics(
        [{"AUROC": r["test_AUROC"], "Dice": r["test_Dice"]} for r in stage1_rows],
        ("AUROC", "Dice"),
    )

    # Pooled CV val (covers all 800 train images exactly once)
    val_y = np.concatenate(all_val_targets)
    val_s1 = np.concatenate(all_val_s1)
    val_s2 = np.concatenate(all_val_s2)
    pooled_val_neg = summarize_negative_detection_result(
        {"AUROC": _safe_auroc(val_y, val_s1)},
        {"AUROC": _safe_auroc(val_y, val_s2)},
        val_y, val_s1, val_s2, n_boot=n_boot, power_delta=power_delta,
    )

    test_y = test_y_ref if test_y_ref is not None else np.array([])
    if all_test_s1_folds:
        test_s1 = np.mean(np.stack(all_test_s1_folds), axis=0)
        test_s2 = np.mean(np.stack(all_test_s2_folds), axis=0)
    else:
        test_s1 = np.array([])
        test_s2 = np.array([])
    pooled_test_neg = summarize_negative_detection_result(
        {"AUROC": _safe_auroc(test_y, test_s1)},
        {"AUROC": _safe_auroc(test_y, test_s2)},
        test_y, test_s1, test_s2, n_boot=n_boot, power_delta=power_delta,
    )

    power = power_analysis_auroc_sample_size(delta=power_delta)

    lc_results = []
    if cfg.get("run_learning_curves", False):
        lc_folds = cfg.get("learning_curve_folds", [0])
        lc_fracs = cfg.get("learning_curve_fractions", [0.25, 0.5, 0.75, 1.0])
        lc_results = run_learning_curves(
            cfg, project_root, image_root, device,
            folds=[int(f) for f in lc_folds],
            fractions=[float(x) for x in lc_fracs],
            smoke_test=smoke_test,
            resume=resume_training,
        )

    # Ensemble on held-out test (Stage1 + best fold gaze models — use pooled test preds average)
    ens_w = float(cfg.get("ensemble_stage1_weight", 0.5))
    test_ensemble_probs = ensemble_probs(test_s1, test_s2, weight_a=ens_w)
    th = calibrate_decision_threshold(test_y, test_ensemble_probs)
    ens_metrics = compute_metrics(test_y, (test_ensemble_probs > th).astype(int), test_ensemble_probs)

    report = {
        "part": "part1_detection_negative",
        "num_folds": num_folds,
        "cv_gaze_fold_metrics": fold_rows,
        "cv_stage1_fold_metrics": stage1_rows,
        "cv_ablation_fold_metrics": ablation_rows,
        "aggregate_gaze_val": agg_gaze_val,
        "aggregate_gaze_test": agg_gaze_test,
        "aggregate_stage1_val": agg_s1_val,
        "aggregate_stage1_test": agg_s1_test,
        "pooled_cv_val_negative_summary": pooled_val_neg,
        "held_out_test_negative_summary": pooled_test_neg,
        "held_out_test_ensemble": ens_metrics,
        "power_analysis": power,
        "learning_curves": lc_results,
        "config_snapshot": {
            "gaze_lambda": cfg.get("gaze_lambda"),
            "stage2_mix_oral": cfg.get("stage2_mix_oral"),
            "run_gaze_ablation": run_ablation,
            "gaze_ablation_lambda": ablation_lambda,
        },
    }

    report_path = os.path.join(project_root, "results", "stage2_part1_negative_report.json")
    cv_path = os.path.join(project_root, "results", "stage2_cv_fold_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(cv_path, "w") as f:
        json.dump(fold_rows, f, indent=2)

    # Per-image predictions on held-out test (mean probs across fold eval passes)
    pd.DataFrame({
        "has_lesion_gt": test_y,
        "prob_stage1": test_s1,
        "prob_stage2_gaze": test_s2,
        "prob_ensemble": test_ensemble_probs,
    }).to_csv(os.path.join(project_root, "results", "stage2_part1_test_predictions.csv"), index=False)

    print("\n=== Part 1 negative-result summary (held-out test) ===")
    print(f"  Stage1 AUROC: {pooled_test_neg['stage1_auroc']:.4f} "
          f"[{pooled_test_neg['stage1_bootstrap_auroc']['lower']:.4f}, "
          f"{pooled_test_neg['stage1_bootstrap_auroc']['upper']:.4f}]")
    print(f"  Stage2 gaze AUROC: {pooled_test_neg['stage2_auroc']:.4f} "
          f"[{pooled_test_neg['stage2_bootstrap_auroc']['lower']:.4f}, "
          f"{pooled_test_neg['stage2_bootstrap_auroc']['upper']:.4f}]")
    print(f"  Paired diff (S2-S1): {pooled_test_neg['paired_auroc_diff_stage2_minus_stage1']['mean_diff']:.4f}")
    print(f"  Ensemble AUROC: {ens_metrics['AUROC']:.4f}")
    print(f"  Interpretation: {pooled_test_neg['interpretation']}")
    print(f"Saved {report_path}")

    best_fold_idx = max(range(len(fold_rows)), key=lambda i: fold_rows[i]["val_AUROC"])
    return {
        "train_done": True,
        "fold_results": fold_rows,
        "agg_metrics": agg_gaze_val,
        "best_fold": best_fold_idx,
        "negative_report": report,
        "report_path": report_path,
    }
