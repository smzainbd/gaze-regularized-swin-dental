"""Shared Stage 2 training utilities (used by NB04 and stage2_cv)."""
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch

from core_modules import (
    SwinGazeModel,
    build_stage1_pseudo_labels,
    evaluate_loader,
    pseudo_label_loss,
)


def apply_stage2_config(config: Dict) -> Dict:
    cfg = dict(config)
    blend = float(cfg.get("stage2_cls_seg_blend", cfg.get("cls_seg_blend", 0.0)))
    cfg["cls_seg_blend"] = blend
    return cfg


def stage2_learning_rate(config: Dict) -> float:
    return float(config.get("stage2_learning_rate", config["learning_rate"]))


def build_model_and_optimizer(
    config: Dict,
    device: torch.device,
    load_stage1: bool = True,
    stage1_ckpt: str = "checkpoints/stage1_best.pth",
) -> Tuple[SwinGazeModel, torch.optim.Optimizer]:
    cfg = apply_stage2_config(dict(config))
    model = SwinGazeModel(cfg).to(device)
    model.verify_lora_applied()
    if load_stage1 and os.path.exists(stage1_ckpt):
        ck = torch.load(stage1_ckpt, map_location=device)
        state = ck.get("model", ck)
        res = model.load_state_dict(state, strict=False)
        print(
            f" [INFO] Loaded Stage 1 weights from {stage1_ckpt} "
            f"(epoch {ck.get('epoch', 'unknown')}): {res}"
        )
    elif load_stage1:
        print(f" [WARNING] Stage 1 checkpoint {stage1_ckpt} not found — training from scratch.")

    lr = stage2_learning_rate(config)
    encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    cls_params = list(model.cls_head.parameters())
    param_groups = [
        {"params": encoder_params, "lr": lr / 10},
        {"params": decoder_params, "lr": lr},
        {"params": cls_params, "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    return model, optimizer


@torch.no_grad()
def evaluate_model_on_loader(
    model: SwinGazeModel,
    loader,
    device: torch.device,
    collect_ids: bool = False,
    use_tta: bool = False,
    tta_views: int = 1,
    threshold: Optional[float] = None,
):
    model.eval()
    return evaluate_loader(
        model,
        loader,
        device,
        use_tta=use_tta,
        tta_views=tta_views,
        threshold=threshold,
        collect_ids=collect_ids,
    )


def build_pseudo_label_map(
    config: Dict,
    train_loader,
    device: torch.device,
    stage1_ckpt: str = "checkpoints/stage1_best.pth",
) -> Dict[str, float]:
    if not config.get("use_pseudo_labels", False):
        return {}
    if not os.path.exists(stage1_ckpt):
        print("[pseudo] stage1_best.pth missing — skipping pseudo labels")
        return {}
    s1_cfg = dict(config)
    s1_cfg["cls_seg_blend"] = 0.0
    s1_model = SwinGazeModel(s1_cfg).to(device)
    ck = torch.load(stage1_ckpt, map_location=device)
    s1_model.load_state_dict(ck.get("model", ck), strict=False)
    neg_t = float(config.get("pseudo_neg_thresh", 0.15))
    pseudo = build_stage1_pseudo_labels(s1_model, train_loader, device, neg_thresh=neg_t)
    del s1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[pseudo] confident healthy negatives: {len(pseudo)} (thresh<{neg_t})")
    return pseudo


def try_resume_checkpoint(
    last_ckpt: str,
    model: SwinGazeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    resume: bool,
) -> Tuple[int, float, int]:
    """Returns (start_epoch, best_auroc, epochs_no_improve)."""
    if not resume or not os.path.exists(last_ckpt):
        return 0, -1.0, 0
    ck = torch.load(last_ckpt, map_location=device)
    try:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_auroc = float(ck.get("best_auroc", -1.0))
        epochs_no_improve = int(ck.get("epochs_no_improve", 0))
        print(f"Resumed from {last_ckpt} epoch {start_epoch} (best AUROC {best_auroc:.4f})")
        return start_epoch, best_auroc, epochs_no_improve
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f" [WARNING] Partial resume from {last_ckpt}: {exc}")
        model.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        return start_epoch, float(ck.get("best_auroc", -1.0)), int(ck.get("epochs_no_improve", 0))


def optimizer_steps_per_epoch(train_loader, grad_accum: int) -> int:
    return max(1, math.ceil(len(train_loader) / max(int(grad_accum), 1)))
