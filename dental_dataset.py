import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift

class DentalDataset(Dataset):
    # Loads dental radiographs + optional 48x48 expert gaze density map.
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, csv_path, project_root, augment=False, image_root=None, df=None):
        self.df           = df if df is not None else pd.read_csv(csv_path)
        self.root         = project_root
        self.image_root   = image_root or project_root
        self.augment      = augment
        self.to_tensor    = T.ToTensor()
        self.normalise    = T.Normalize(mean=self.MEAN, std=self.STD)
        self._init_bboxes(project_root)

    def _resolve_tufts_anomaly_mask_path(self, image_id):
        """Expert anomaly mask JPEGs: data/tdd/Expert/mask/{id}.JPG"""
        for base in (
            os.path.join(self.image_root, 'data/tdd/Expert/mask'),
            os.path.join(self.root, 'data/tdd/Expert/mask'),
        ):
            for ext in ('.JPG', '.jpg', '.png'):
                path = os.path.join(base, f'{image_id}{ext}')
                if os.path.exists(path):
                    return path
        return None

    def _load_anomaly_mask(self, image_id):
        # Tufts expert polygon raster masks (840x1615); resize to 224 for decoder supervision.
        path = self._resolve_tufts_anomaly_mask_path(image_id)
        if path is None:
            return np.zeros((1, 224, 224), dtype=np.float32)
        try:
            mask = Image.open(path).convert('L')
            mask = mask.resize((224, 224), Image.Resampling.NEAREST)
            arr = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
            return arr[np.newaxis, :, :]
        except (OSError, ValueError, Image.UnidentifiedImageError):
            return np.zeros((1, 224, 224), dtype=np.float32)

    def _init_bboxes(self, project_root):
        self.bbox_map = {}
        if 'dataset' not in self.df.columns or not (self.df['dataset'] == 'oralxrays').any():
            return
        for name in ['instances_train2017.json', 'instances_val2017.json']:
            p = os.path.join(project_root, 'data/oralxrays-91/annotations', name)
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        coco = json.load(f)
                    img_id_to_size = {
                        str(img['id']): (max(img.get('width', 1000), 1), max(img.get('height', 1000), 1))
                        for img in coco.get('images', [])
                    }
                    for ann in coco.get('annotations', []):
                        img_id = str(ann['image_id'])
                        if img_id not in self.bbox_map:
                            self.bbox_map[img_id] = []
                        orig_w, orig_h = img_id_to_size.get(img_id, (1000, 1000))
                        x, y, w, h = ann['bbox']
                        x1 = max(0, min(223, int((x / orig_w) * 224)))
                        y1 = max(0, min(223, int((y / orig_h) * 224)))
                        x2 = max(0, min(223, int(((x + w) / orig_w) * 224)))
                        y2 = max(0, min(223, int(((y + h) / orig_h) * 224)))
                        if x2 >= x1 and y2 >= y1:
                            self.bbox_map[img_id].append((x1, y1, x2, y2))
                except Exception:
                    pass

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        img = Image.open(full).convert('RGB')
        return img.resize((224, 224), Image.Resampling.BILINEAR)

    def _load_gaze(self, path):
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        if not os.path.exists(full):
            return np.zeros((48, 48), dtype=np.float32)
        arr = np.load(full).astype(np.float32)
        return arr

    def _sync_augment(self, img_pil, gaze_np, mask_arr):
        seed = random.randint(0, 2**31)
        BILINEAR = T.InterpolationMode.BILINEAR

        # Shared random decisions
        random.seed(seed);   do_flip = random.random() < 0.5
        random.seed(seed + 1); angle = random.uniform(-10, 10)
        random.seed(seed + 2); tx = random.uniform(-0.05, 0.05)
        random.seed(seed + 3); ty = random.uniform(-0.05, 0.05)

        mask_t = torch.from_numpy(mask_arr)  # [1, 224, 224]
        NEAREST = T.InterpolationMode.NEAREST

        # Horizontal flip
        if do_flip:
            img_pil = TF.hflip(img_pil)
            gaze_np = gaze_np[:, ::-1].copy()
            mask_t = TF.hflip(mask_t)

        # Rotation
        img_pil = TF.rotate(img_pil, angle, interpolation=BILINEAR)
        gaze_np = nd_rotate(gaze_np, angle, reshape=False, order=1)
        mask_t = TF.rotate(mask_t, angle, interpolation=NEAREST)

        # Translation
        shift_px = (ty * gaze_np.shape[0], tx * gaze_np.shape[1])
        img_pil  = TF.affine(img_pil, angle=0, translate=(int(tx * 224), int(ty * 224)), scale=1.0, shear=0, interpolation=BILINEAR)
        gaze_np  = nd_shift(gaze_np, shift_px, order=1)
        mask_t = TF.affine(mask_t, angle=0, translate=(int(tx * 224), int(ty * 224)),
                           scale=1.0, shear=0, interpolation=NEAREST)

        # Colour jitter (image only)
        cj = T.ColorJitter(brightness=0.15, contrast=0.15)
        img_pil = cj(img_pil)
        
        mask_arr = (mask_t.numpy() > 0.5).astype(np.float32)
        return img_pil, np.clip(gaze_np, 0, None), mask_arr

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = self._load_image(row['image_path'])
        has_gaze = int(row.get('has_gaze_map', 0))
        gaze = self._load_gaze(row['gaze_map_path']) if has_gaze else np.zeros((48, 48), dtype=np.float32)

        mask_arr = np.zeros((1, 224, 224), dtype=np.float32)
        img_id = str(row['id'])
        dataset = str(row.get('dataset', 'unknown'))
        if dataset == 'tufts':
            mask_arr = self._load_anomaly_mask(img_id)
        elif hasattr(self, 'bbox_map') and img_id in self.bbox_map:
            for (x1, y1, x2, y2) in self.bbox_map[img_id]:
                mask_arr[0, y1:y2+1, x1:x2+1] = 1.0

        if self.augment:
            img, gaze, mask_arr = self._sync_augment(img, gaze, mask_arr)
            
        img_t = self.to_tensor(img)
        img_t = self.normalise(img_t)
        
        has_lesion = int(row['has_lesion_gt'])
        label = torch.tensor(has_lesion, dtype=torch.float32)
        
        gaze_tensor = torch.from_numpy(gaze).unsqueeze(0)
        s = gaze_tensor.sum()
        if has_gaze and s > 0:
            gaze_tensor = gaze_tensor / s
        
        has_gaze_tensor = torch.tensor(has_gaze, dtype=torch.float32)
        bbox_mask_tensor = torch.from_numpy(mask_arr)

        return {
            'image':         img_t,
            'has_lesion_gt': label,
            'gaze_map':      gaze_tensor,
            'has_gaze_map':  has_gaze_tensor,
            'bbox_mask':     bbox_mask_tensor,
            'image_id':      str(row['id']),
            'dataset':       str(row.get('dataset', 'unknown'))
        }

def make_weighted_sampler(df, multiplier=1, tufts_weight=16, lesion_boost=1.0):
    # Tufts-only splits: balance lesion vs healthy and oversample epochs via multiplier.
    if 'dataset' in df.columns and (df['dataset'] == 'tufts').all():
        labels = df['has_lesion_gt'].values
        n_total = len(labels)
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        w_pos = n_total / (2.0 * max(n_pos, 1))
        w_neg = n_total / (2.0 * max(n_neg, 1))
        if lesion_boost > 1.0:
            w_pos *= lesion_boost
        weights = [w_pos if l == 1 else w_neg for l in labels]
        return WeightedRandomSampler(
            weights, num_samples=len(weights) * int(multiplier), replacement=True)

    # Mixed Oral + Tufts pools: upsample Tufts rows and optionally lesion positives.
    lb = float(lesion_boost)
    weights = []
    for _, row in df.iterrows():
        w = tufts_weight if str(row.get('dataset', '')).lower() == 'tufts' else 1.0
        if int(row.get('has_lesion_gt', 0)) == 1 and lb > 1.0:
            w *= lb
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights) * int(multiplier), replacement=True)

def _subsample_stratified(df, fraction, seed=42):
    """Stratified subsample of Tufts train rows for learning curves."""
    fraction = float(fraction)
    if fraction >= 1.0 or len(df) == 0:
        return df
    if fraction <= 0.0:
        raise ValueError("train_fraction must be > 0")
    parts = []
    for label in sorted(df["has_lesion_gt"].unique()):
        grp = df[df["has_lesion_gt"] == label]
        n = max(1, int(round(len(grp) * fraction)))
        parts.append(grp.sample(n=min(n, len(grp)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def build_dataloaders(CONFIG, project_root, batch_size=None, fold=None, smoke_test=False, image_root=None, stage2_training=False, train_fraction=1.0):
    bs = batch_size or CONFIG['chosen_batch_size']
    img_root = image_root or project_root
    train_csv = os.path.join(project_root, 'splits/train_split.csv')
    train_df_full = pd.read_csv(train_csv)

    use_cv_fold = (
        stage2_training
        and fold is not None
        and 'fold' in train_df_full.columns
        and int(CONFIG.get('num_folds', 1)) > 1
    )

    if use_cv_fold:
        fold_id = int(fold)
        fold_val_df = train_df_full[train_df_full['fold'] == fold_id].copy()
        fold_train_df = train_df_full[train_df_full['fold'] != fold_id].copy()
        if len(fold_val_df) == 0 or len(fold_train_df) == 0:
            raise ValueError(
                f"Invalid CV fold {fold_id}: train={len(fold_train_df)} val={len(fold_val_df)}. "
                "Check splits/train_split.csv fold column."
            )
        if float(train_fraction) < 1.0:
            fold_train_df = _subsample_stratified(
                fold_train_df, train_fraction, seed=int(CONFIG.get('random_seed', 42)) + fold_id,
            )
        train_ds = DentalDataset(train_csv, project_root, augment=True, image_root=img_root, df=fold_train_df)
        val_ds = DentalDataset(train_csv, project_root, augment=False, image_root=img_root, df=fold_val_df)
        train_df = fold_train_df
    else:
        train_df = train_df_full
        if float(train_fraction) < 1.0 and stage2_training:
            train_df = _subsample_stratified(
                train_df, train_fraction, seed=int(CONFIG.get('random_seed', 42)),
            )
        train_ds = DentalDataset(train_csv, project_root, augment=True, image_root=img_root, df=train_df)
        val_ds = DentalDataset(os.path.join(project_root, 'splits/val_split.csv'),
                               project_root, augment=False, image_root=img_root)

    test_ds    = DentalDataset(os.path.join(project_root, 'splits/test_split.csv'),
                               project_root, augment=False, image_root=img_root)
    dentex_ds  = DentalDataset(os.path.join(project_root, 'splits/dentex_test_split.csv'),
                               project_root, augment=False, image_root=img_root)

    oral_test_csv = os.path.join(project_root, 'splits/oralxray_test_split.csv')
    oral_test_ds  = DentalDataset(oral_test_csv, project_root, augment=False, image_root=img_root) if os.path.exists(oral_test_csv) else None

    oral_val_csv = os.path.join(project_root, 'splits/oralxray_val_split.csv')
    oral_val_ds  = DentalDataset(oral_val_csv, project_root, augment=False, image_root=img_root) if os.path.exists(oral_val_csv) else None
                             
    mult = CONFIG.get('dataset_multiplier', 3)
    lesion_boost = float(CONFIG.get('lesion_sampler_boost', 1.0))
    tufts_boost = float(CONFIG.get('tufts_sampler_boost', 16))
    stage2_mix_oral = bool(stage2_training and CONFIG.get('stage2_mix_oral', False))

    oral_train_csv = os.path.join(project_root, 'splits/oralxray_train_split.csv')
    oral_train_ds  = DentalDataset(oral_train_csv, project_root, augment=True, image_root=img_root) if os.path.exists(oral_train_csv) else None

    if stage2_mix_oral and oral_train_ds is not None and not use_cv_fold:
        oral_df = pd.read_csv(oral_train_csv)
        combined_df = pd.concat([train_df_full, oral_df], ignore_index=True)
        train_ds = DentalDataset(
            train_csv,
            project_root, augment=True, image_root=img_root, df=combined_df,
        )
        sampler = make_weighted_sampler(
            combined_df, multiplier=mult, tufts_weight=tufts_boost,
            lesion_boost=lesion_boost,
        )
    else:
        sampler = make_weighted_sampler(train_df, multiplier=mult, lesion_boost=lesion_boost)
    # For oralxrays-only training (Stage 1), we can just weight all equally (or by lesion, but let's just use equal)
    oral_sampler = WeightedRandomSampler([1.0]*len(oral_train_ds), num_samples=len(oral_train_ds), replacement=True) if oral_train_ds else None

    sweep_csv = os.path.join(project_root, 'splits/sweep_split.csv')
    sweep_ds  = DentalDataset(sweep_csv, project_root, augment=False, image_root=img_root) if os.path.exists(sweep_csv) else train_ds
    sweep_df_ref  = pd.read_csv(sweep_csv) if os.path.exists(sweep_csv) else train_df
    sweep_sampler = make_weighted_sampler(sweep_df_ref, multiplier=mult)

    # Use num_workers=0 on local Windows scripts to prevent multiprocessing locks, or 2 if protected
    nw = 0 if os.name == 'nt' else 2
    pin = bool(nw > 0)

    loaders = {
        'train':          DataLoader(train_ds,  batch_size=bs, sampler=sampler,
                                     num_workers=nw, pin_memory=pin, drop_last=True),
        'val':            DataLoader(val_ds,    batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=pin),
        'test_tufts':     DataLoader(test_ds,   batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=pin),
        'dentex':         DataLoader(dentex_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=pin),
        'sweep':          DataLoader(sweep_ds,  batch_size=bs, sampler=sweep_sampler,
                                     num_workers=nw, pin_memory=pin, drop_last=True),
        'oralxray_train': DataLoader(oral_train_ds, batch_size=bs, sampler=oral_sampler,
                                     num_workers=nw, pin_memory=pin, drop_last=True) if oral_train_ds else None,
        'oralxray_val':   DataLoader(oral_val_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=pin) if oral_val_ds else None,
        'test_oralxray':  DataLoader(oral_test_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=pin) if oral_test_ds else None,
    }
    loaders['tdd_train']     = loaders['train']
    loaders['tdd_val']       = loaders['val']
    loaders['tdd_test']      = loaders['test_tufts']
    loaders['oralxray_test'] = loaders['test_oralxray']
    return loaders


def build_refit_dataloaders(CONFIG, project_root, batch_size=None, image_root=None, smoke_test=False):
    """
    Full-pool Stage-2 refit loaders: train on ALL train_split rows (e.g. 900),
    no CV val (val=None). Held-out test_split for secondary eval only.
    """
    bs = batch_size or CONFIG['chosen_batch_size']
    img_root = image_root or project_root
    train_csv = os.path.join(project_root, 'splits/train_split.csv')
    train_df = pd.read_csv(train_csv)
    if smoke_test:
        train_df = train_df.head(min(64, len(train_df))).copy()

    train_ds = DentalDataset(train_csv, project_root, augment=True, image_root=img_root, df=train_df)
    test_ds = DentalDataset(
        os.path.join(project_root, 'splits/test_split.csv'),
        project_root, augment=False, image_root=img_root,
    )

    mult = CONFIG.get('dataset_multiplier', 3)
    lesion_boost = float(CONFIG.get('lesion_sampler_boost', 1.0))
    sampler = make_weighted_sampler(train_df, multiplier=mult, lesion_boost=lesion_boost)
    nw = 0 if os.name == 'nt' else 2
    pin = bool(nw > 0)

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )
    return {
        'tdd_train': train_loader,
        'tdd_val': None,
        'tdd_test': test_loader,
        'train': train_loader,
        'test_tufts': test_loader,
        'n_train': len(train_df),
        'n_test': len(test_ds),
    }


def verify_tufts_anomaly_masks(project_root, image_root=None, csv_relpath='splits/train_split.csv', sample_n=32):
    """Smoke-test that Tufts expert masks load into bbox_mask (nonzero for lesion images)."""
    csv_path = os.path.join(project_root, csv_relpath)
    if not os.path.exists(csv_path):
        return False, f'missing split csv: {csv_relpath}'
    df = pd.read_csv(csv_path)
    if 'dataset' in df.columns and not (df['dataset'] == 'tufts').all():
        return True, 'split is not Tufts-only — skip anomaly mask check'

    img_root = image_root or project_root
    mask_root = os.path.join(img_root, 'data/tdd/Expert/mask')
    if not os.path.isdir(mask_root):
        mask_root = os.path.join(project_root, 'data/tdd/Expert/mask')
    if not os.path.isdir(mask_root):
        return False, (
            'Tufts expert mask folder missing. Expected '
            f'{os.path.join(img_root, "data/tdd/Expert/mask")}'
        )

    ds = DentalDataset(csv_path, project_root, augment=False, image_root=img_root)
    lesion_pos = [i for i in range(len(df)) if int(df.iloc[i]['has_lesion_gt']) == 1]
    healthy_pos = [i for i in range(len(df)) if int(df.iloc[i]['has_lesion_gt']) == 0]
    if not lesion_pos:
        return False, 'no lesion-positive rows in split'

    sample_lesion = lesion_pos[:min(sample_n, len(lesion_pos))]
    sample_healthy = healthy_pos[:min(max(8, sample_n // 4), len(healthy_pos))]

    lesion_nz = sum(1 for i in sample_lesion if float(ds[i]['bbox_mask'].sum()) > 0)
    healthy_nz = sum(1 for i in sample_healthy if float(ds[i]['bbox_mask'].sum()) > 0)

    if lesion_nz == 0:
        return False, (
            f'0/{len(sample_lesion)} sampled lesion images have nonzero anomaly masks. '
            f'Check LOCAL_DATA unzipped data/tdd/Expert/mask/ (not Drive-only metadata).'
        )
        
    if healthy_nz > 0:
        return False, (
            f'FATAL: {healthy_nz}/{len(sample_healthy)} healthy samples have non-zero anomaly masks. '
            f'Healthy images must have strictly empty masks.'
        )

    msg = (
        f'OK: {lesion_nz}/{len(sample_lesion)} lesion samples have mask pixels; '
        f'{healthy_nz}/{len(sample_healthy)} healthy samples nonzero (expect 0)'
    )
    return True, msg
