import json

NOTEBOOK_PATH = "Notebooks/NB04_stage2_train.ipynb"

cell_source = """# === Stage 2 OralXray cross-domain evaluation (self-contained) ===
import os, json, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# --- Defensive setup ---
if 'ON_GPU' not in globals() or 'device' not in globals():
    ON_GPU = torch.cuda.is_available()
    device = torch.device('cuda' if ON_GPU else 'cpu')
if 'PROJECT' not in globals():
    PROJECT = os.getcwd()
if 'CONFIG' not in globals():
    with open(os.path.join(PROJECT, 'config.json'), 'r') as f:
        CONFIG = json.load(f)

STAGE2_CKPT = os.path.join(PROJECT, 'checkpoints', 'stage2_best.pth')

if not ON_GPU:
    print('CPU: skipping OralXray eval on CPU dry run.')
    sys.exit(0)

if 'build_model_and_optimizer' not in globals():
    raise RuntimeError('Run NB04 model cell first.')
if 'compute_metrics' not in globals():
    from core_modules import compute_metrics, calibrate_decision_threshold
if not os.path.isfile(STAGE2_CKPT):
    raise FileNotFoundError(f'Missing {STAGE2_CKPT}')

from dental_dataset import build_dataloaders
loaders = build_dataloaders(CONFIG, PROJECT, image_root=None, fold=0)
if 'oralxray_test' not in loaders:
    raise RuntimeError('oralxray_test loader not built. Check dataset paths.')
oral_loader = loaders['oralxray_test']

print(f'OralXray total: {len(oral_loader.dataset)}')

ckpt = torch.load(STAGE2_CKPT, map_location=device)
model, _ = build_model_and_optimizer(CONFIG, device, load_stage1=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Loaded Stage 2 best (epoch {ckpt.get("epoch", "?")})')

ids, targets, probs = [], [], []
with torch.no_grad():
    for batch in tqdm(oral_loader, desc='OralXray Stage2'):
        logits, _, _, _ = model(batch['image'].to(device))
        p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        probs.extend(p.tolist())
        targets.extend(batch['has_lesion_gt'].tolist())
        ids.extend(batch.get('image_id', []))

targets, probs = np.array(targets), np.array(probs)
mask_l = np.isin(targets, [0, 1])
t_lab, p_lab = targets[mask_l], probs[mask_l]
opt_thresh = calibrate_decision_threshold(t_lab, p_lab)
preds_lab = (p_lab > opt_thresh).astype(int)
metrics = compute_metrics(t_lab, preds_lab, p_lab)
metrics.update({
    'opt_threshold': float(opt_thresh),
    'n_total': len(targets),
    'n_labeled': int(mask_l.sum()),
    'n_unlabeled': int((~mask_l).sum()),
})

cm = confusion_matrix(t_lab, preds_lab)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
print(f"\\nSTAGE 2 ORALXRAY | thresh={opt_thresh:.3f}")
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Dice={metrics['Dice']:.4f} AUROC={metrics['AUROC']:.4f}")

os.makedirs('results', exist_ok=True)
with open('results/stage2_oralxray_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
pd.DataFrame({
    'id': ids, 'has_lesion_gt': targets, 'prob': probs,
    'pred': (probs > opt_thresh).astype(int),
}).to_csv('results/stage2_oralxray_per_image_predictions.csv', index=False)
print('Saved results/stage2_oralxray_metrics.json + stage2_oralxray_per_image_predictions.csv')
"""

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Use splitlines(keepends=True) to preserve internal newlines like \n in strings!
source_lines = cell_source.splitlines(keepends=True)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "id": "stage2_oralxray_eval"
    },
    "outputs": [],
    "source": source_lines
}

nb['cells'].append(new_cell)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Added clean OralXray cell successfully.")
