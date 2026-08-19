import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve

# ==========================================
# 1. Architecture: SwinGazeModel + Decoder
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SpatialDecoder(nn.Module):
    def __init__(self, hidden_dims=[96, 192, 384, 768], out_classes=1):
        super().__init__()
        self.up4   = nn.ConvTranspose2d(hidden_dims[3], hidden_dims[2], 2, stride=2)
        self.conv3 = ConvBlock(hidden_dims[2] * 2, hidden_dims[2])
        self.up3   = nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], 2, stride=2)
        self.conv2 = ConvBlock(hidden_dims[1] * 2, hidden_dims[1])
        self.up2   = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 2, stride=2)
        self.conv1 = ConvBlock(hidden_dims[0] * 2, hidden_dims[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], 32, 4, stride=4),
            nn.Conv2d(32, out_classes, 1)
        )

    def forward(self, features):
        f1, f2, f3, f4 = features
        def reshape_state(f):
            B, L, C = f.shape
            H = int(math.sqrt(L))
            assert H * H == L, f"Feature seq len {L} is not a perfect square; H={H}"
            return f.reshape(B, H, H, C).permute(0, 3, 1, 2).contiguous()
        f1 = reshape_state(f1)   # [B,  96, 56, 56]
        f2 = reshape_state(f2)   # [B, 192, 28, 28]
        f3 = reshape_state(f3)   # [B, 384, 14, 14]
        f4 = reshape_state(f4)   # [B, 768,  7,  7]
        x = self.up4(f4)
        x = self.conv3(torch.cat([x, f3], dim=1))
        x = self.up3(x)
        x = self.conv2(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        x = self.conv1(torch.cat([x, f1], dim=1))
        return self.final_up(x)

class SwinGazeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = SwinModel.from_pretrained(config['model_name'], attn_implementation="eager")
        self._use_lora = config.get('use_lora', True)  # [v21 R1] needed by forward() to route correctly

        if self._use_lora:
            lora_cfg = LoraConfig(
                r=config['lora_rank'],
                lora_alpha=config.get('lora_alpha', 16),
                target_modules=config['lora_modules']
            )
            self.encoder = get_peft_model(base, lora_cfg)
        else:
            self.encoder = base
            print(" [INFO] use_lora=False: running no-LoRA baseline")

        for name, param in self.encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.decoder = SpatialDecoder()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        # Stage 2: blend global cls with MIL max-pool on seg_map (0 = Stage-1 behaviour).
        self._cls_seg_blend = float(config.get('cls_seg_blend', 0.0))

    def verify_lora_applied(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc_trainable = sum(p.numel() for n, p in self.encoder.named_parameters() if p.requires_grad)
        dec_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        cls_trainable = sum(p.numel() for p in self.cls_head.parameters() if p.requires_grad)
        ratio = trainable / total * 100
        assert ratio > 0.0, "FATAL: 0% trainable parameters. Check LoRA target_modules."
        print(f" LoRA applied: {trainable:,} / {total:,} trainable ({ratio:.3f}%)")
        print(f"  encoder: {enc_trainable:,}  decoder: {dec_trainable:,}  cls_head: {cls_trainable:,}")
        return ratio

    def forward(self, x):
        # [v21 R1] Branch on use_lora — plain SwinModel has no .base_model attribute.
        if self._use_lora:
            out = self.encoder.base_model(
                pixel_values=x,
                output_attentions=True,
                output_hidden_states=True
            )
        else:
            out = self.encoder(
                pixel_values=x,
                output_attentions=True,
                output_hidden_states=True
            )
        features = [out.hidden_states[i] for i in (0, 1, 2, 4)]
        seg_map  = self.decoder(features)                             # [B, 1, 224, 224]
        cls_features = out.hidden_states[4]                           # [B, 49, 768]
        global_logits = self.cls_head(cls_features.mean(dim=1))      # [B, 1]
        if self._cls_seg_blend > 0.0:
            mil_logits = seg_map.flatten(1).max(dim=1).values.unsqueeze(1)
            cls_logits = global_logits + self._cls_seg_blend * mil_logits
        else:
            cls_logits = global_logits
        last_attn = out.attentions[-1].mean(dim=1).mean(dim=1).reshape(-1, 7, 7)
        return cls_logits, seg_map, last_attn, out

# ==========================================
# 2. Alignment, Loss, and Metric Utilities
# ==========================================

def align_attention_to_gaze(attn_map, target_size=48):
    # Interpolates [B, 7, 7] or spatial attention map to [B, target_size, target_size].
    if attn_map.dim() == 3:
        attn_map = attn_map.unsqueeze(1)
    aligned = F.interpolate(attn_map, size=(target_size, target_size),
                            mode="bilinear", align_corners=False)
    return aligned.squeeze(1)

def compute_gaze_kl_loss(mode, attn_map, gaze_map, has_gaze_mask, target_size=48):
    if mode != "DUAL_ARM" or not has_gaze_mask.bool().any():
        return torch.tensor(0.0, device=attn_map.device)
    has_gaze = has_gaze_mask.bool()
    spatial_attn = align_attention_to_gaze(attn_map, target_size)
    va = spatial_attn[has_gaze]
    vg = gaze_map[has_gaze]
    
    # 1. Squeeze extra channel dimension if present (e.g. [N, 1, 48, 48] -> [N, 48, 48])
    if vg.dim() == 4 and vg.size(1) == 1:
        vg = vg.squeeze(1)
        
    # 2. Normalize both predicted attention and target gaze to valid spatial probability distributions
    # A valid KL divergence D_KL(P || Q) requires sum(P) == 1 and sum(Q) == 1 over the spatial grid
    va_prob = va / (va.sum(dim=(-2, -1), keepdim=True) + 1e-9)
    vg_prob = vg / (vg.sum(dim=(-2, -1), keepdim=True) + 1e-9)
    
    return F.kl_div((va_prob + 1e-9).log(), vg_prob, reduction="batchmean")

def calibrate_decision_threshold(targets, probs):
    try:
        fpr, tpr, thresholds = roc_curve(targets, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    except Exception:
        return 0.5

def compute_dice(pred, target):
    # Dice score for binary numpy arrays.
    pred   = pred.astype(bool)
    target = target.astype(bool)
    intersection = (pred & target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-5)

def compute_metrics(targets, predictions, probs):
    metrics = {}
    metrics['Dice']   = compute_dice(predictions, targets)
    metrics['F1']     = f1_score(targets, predictions, zero_division=0)
    metrics['Recall'] = recall_score(targets, predictions, zero_division=0)
    try:
        metrics['AUROC'] = roc_auc_score(targets, probs)
    except ValueError:
        metrics['AUROC'] = 0.5
    return metrics

def dice_loss_with_logits(logits, targets, eps=1e-5):
    """Soft Dice loss for dense binary maps [B, 1, H, W] or flat logits."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    if probs.dim() == 4:
        dims = (1, 2, 3)
    else:
        dims = (1,)
        
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection) / (union + eps)
    
    # NEW: zero out Dice loss for empty masks (healthy images)
    has_lesion = (targets.sum(dim=dims) > 0).float()
    if has_lesion.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
        
    loss_per_image = 1.0 - dice
    return (loss_per_image * has_lesion).sum() / has_lesion.sum()


def segmentation_aux_loss(seg_map, masks, pos_weight=None, dice_weight=0.5):
    """BCE + Dice on decoder output vs expert/COCO masks."""
    dice_w = float(dice_weight)
    dice_w = min(max(dice_w, 0.0), 1.0)
    bce = F.binary_cross_entropy_with_logits(seg_map, masks, pos_weight=pos_weight)
    if dice_w <= 0.0:
        return bce
    dice = dice_loss_with_logits(seg_map, masks)
    return (1.0 - dice_w) * bce + dice_w * dice


def classification_loss(logits, targets, config, pos_weight=None):
    """Image-level BCE or focal loss on cls logits."""
    cl = logits.squeeze(-1)
    targets = targets.view_as(cl)
    if config.get('use_focal_loss', False):
        alpha = config.get('focal_alpha')
        if alpha is not None:
            alpha = float(alpha)
        return binary_focal_loss_with_logits(
            cl, targets,
            gamma=float(config.get('focal_gamma', 2.0)),
            alpha=alpha,
            pos_weight=pos_weight,
        )
    return F.binary_cross_entropy_with_logits(cl, targets, pos_weight=pos_weight)


def binary_focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None, pos_weight=None):
    """Binary focal loss (Lin et al.). Use for image-level cls; targets in [0, 1]."""
    logits = logits.view(-1)
    targets = targets.view(-1)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight,
    )
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_w = (1 - p_t).clamp(min=0) ** gamma
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_w = alpha_t * focal_w
    return (focal_w * bce).mean()

def evaluate_model(model, loader, device, threshold=0.5, use_tta=False, tta_views=1):
    # Run inference on loader; return metrics dict with Dice, F1, Recall, AUROC.
    targets, probs, _ = collect_loader_probs(model, loader, device, use_tta=use_tta, tta_views=tta_views)
    if threshold is None:
        opt_thresh = calibrate_decision_threshold(targets, probs)
    else:
        opt_thresh = float(threshold)
    preds = (probs > opt_thresh).astype(int)
    metrics = compute_metrics(targets, preds, probs)
    metrics['opt_threshold'] = opt_thresh
    metrics['threshold_used'] = opt_thresh
    return metrics


def _tta_transform_batch(images, view_idx):
    """Deterministic TTA on normalized [B,3,H,W] tensors."""
    import torchvision.transforms.functional as TF
    if view_idx == 0:
        return images
    if view_idx == 1:
        return TF.hflip(images)
    if view_idx == 2:
        return TF.rotate(images, 7.0)
    if view_idx == 3:
        return TF.rotate(images, -7.0)
    if view_idx == 4:
        return TF.rotate(TF.hflip(images), 7.0)
    if view_idx == 5:
        return TF.rotate(TF.hflip(images), -7.0)
    if view_idx == 6:
        return TF.rotate(images, 5.0)
    if view_idx == 7:
        return TF.rotate(images, -5.0)
    return images


def predict_probs_batch(model, images, device, use_tta=False, tta_views=1):
    """Per-image lesion probabilities [B]."""
    if not use_tta or int(tta_views) <= 1:
        logits, _, _, _ = model(images)
        return torch.sigmoid(logits.squeeze(-1))
    n_views = min(int(tta_views), 8)
    acc = []
    with torch.no_grad():
        for v in range(n_views):
            x = _tta_transform_batch(images, v)
            logits, _, _, _ = model(x)
            acc.append(torch.sigmoid(logits.squeeze(-1)))
    return torch.stack(acc, dim=0).mean(dim=0)


def collect_loader_probs(model, loader, device, use_tta=False, tta_views=1, collect_ids=False):
    model.eval()
    targets, probs, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            p = predict_probs_batch(model, images, device, use_tta=use_tta, tta_views=tta_views)
            probs.extend(p.detach().cpu().numpy().tolist())
            targets.extend(batch['has_lesion_gt'].cpu().numpy().tolist())
            if collect_ids:
                raw_ids = batch.get('image_id', batch.get('id', []))
                ids.extend([str(x) for x in raw_ids])
    return np.array(targets), np.array(probs), ids


def evaluate_loader(model, loader, device, use_tta=False, tta_views=1, threshold=None, collect_ids=False):
    targets, probs, ids = collect_loader_probs(
        model, loader, device, use_tta=use_tta, tta_views=tta_views, collect_ids=collect_ids,
    )
    opt_thresh = calibrate_decision_threshold(targets, probs) if threshold is None else float(threshold)
    preds = (probs > opt_thresh).astype(int)
    metrics = compute_metrics(targets, preds, probs)
    metrics['opt_threshold'] = opt_thresh
    if collect_ids:
        return metrics, targets, preds, probs, np.array(ids)
    return metrics, targets, preds, probs


def ensemble_probs(probs_a, probs_b, weight_a=0.5):
    wa = float(weight_a)
    wa = min(max(wa, 0.0), 1.0)
    return wa * probs_a + (1.0 - wa) * probs_b


def load_checkpoint_probs(config, ckpt_path, loader, device, cls_seg_blend=None,
                          use_tta=False, tta_views=1):
    """Load weights from checkpoint and return per-image probabilities."""
    cfg = dict(config)
    if cls_seg_blend is not None:
        cfg['cls_seg_blend'] = float(cls_seg_blend)
    model = SwinGazeModel(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get('model', ck), strict=False)
    _, probs, _ = collect_loader_probs(model, loader, device, use_tta=use_tta, tta_views=tta_views)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return probs


def build_stage1_pseudo_labels(model, loader, device, neg_thresh=0.15, dataset_filter='tufts'):
    """
    Confident Stage-1 negatives on healthy Tufts train images.
    Returns dict image_id -> 0.0 (reinforce healthy label during Stage-2).
    """
    model.eval()
    neg_thresh = float(neg_thresh)
    pseudo = {}
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            probs = predict_probs_batch(model, images, device, use_tta=False, tta_views=1)
            datasets = batch.get('dataset', ['tufts'] * images.shape[0])
            for i in range(images.shape[0]):
                ds = str(datasets[i]).lower()
                if dataset_filter and ds != str(dataset_filter).lower():
                    continue
                if int(batch['has_lesion_gt'][i].item()) != 0:
                    continue
                img_id = str(batch['image_id'][i])
                p = float(probs[i].item())
                if p < neg_thresh:
                    pseudo[img_id] = 0.0
    return pseudo


def pseudo_label_loss(logits, image_ids, pseudo_map, device, weight=0.25):
    """Extra BCE toward Stage-1 confident healthy pseudo labels."""
    if not pseudo_map or weight <= 0:
        return torch.tensor(0.0, device=device)
    cl = logits.squeeze(-1)
    losses = []
    for i, img_id in enumerate(image_ids):
        key = str(img_id)
        if key not in pseudo_map:
            continue
        target = torch.tensor(pseudo_map[key], device=device, dtype=cl.dtype)
        losses.append(F.binary_cross_entropy_with_logits(cl[i], target))
    if not losses:
        return torch.tensor(0.0, device=device)
    return float(weight) * torch.stack(losses).mean()
