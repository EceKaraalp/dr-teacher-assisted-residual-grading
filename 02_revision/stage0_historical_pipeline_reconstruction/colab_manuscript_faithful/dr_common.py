"""Shared helpers: seeds, data (with sample ids), splits, metrics, Drive sync.

The split protocol and the metric definitions are identical to the original
`dataset_loader.py` / training notebooks (stratified 80/10/10, random_state=42,
raw 224x224 bilinear resize, pixel range 0-255 kept as float).
"""
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

SMOKE = os.environ.get("DR_SMOKE", "0") == "1"
# Which validation sample moves to train: 'c0p0' = manuscript-count default (first class-0 sample),
# 'c<k>p<i>' = i-th validation sample of class k, 'none' = original 2928/367/367 split. Test IDs never change.
SPLIT_MOVE = os.environ.get("SPLIT_MOVE", "c0p0")
SMOKE_N = int(os.environ.get("DR_SMOKE_N", "300"))
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------- data
def load_aptos(dataset_path="APTOS2019", image_size=224):
    """Returns (ids, images uint8 [N,H,W,3], labels int32 [N]) in train.csv order.

    Uses `cache_<size>.npz` when present (verified against train.csv), otherwise
    builds it from `train_images/` with the same PIL bilinear resize as the
    original loader.
    """
    from PIL import Image

    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path / "train.csv")
    cache = dataset_path / f"cache_{image_size}.npz"
    if cache.exists():
        z = np.load(cache)
        images, labels = z["images"], z["labels"].astype(np.int32)
        if len(images) != len(df) or not np.array_equal(labels, df["diagnosis"].values.astype(np.int32)):
            raise RuntimeError(
                f"{cache} is not aligned with train.csv (len {len(images)} vs {len(df)} or labels differ). "
                "Delete the cache so it can be rebuilt from train_images/.")
        return df["id_code"].tolist(), images, labels

    img_dir = dataset_path / "train_images"
    images, labels, ids = [], [], []
    for _, row in df.iterrows():
        files = list(img_dir.glob(f"{row['id_code']}*"))
        if not files:
            raise FileNotFoundError(f"Image not found for {row['id_code']} in {img_dir}")
        im = Image.open(files[0]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        images.append(np.array(im, dtype=np.uint8))
        labels.append(int(row["diagnosis"]))
        ids.append(row["id_code"])
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    np.savez(cache, images=images, labels=labels)
    return ids, images, labels


def make_split_indices(labels, test_ratio=0.1, val_ratio=0.1, train_ratio=0.8, random_state=42):
    """Create the manuscript-count split while preserving the frozen test IDs.

    The original two-step split produced 2,928/367/367. Move one deterministic
    validation sample into training to match the manuscript's 2,929/366/367
    counts without changing the held-out test set.
    """
    idx = np.arange(len(labels))
    tmp, test = train_test_split(idx, test_size=test_ratio, random_state=random_state, stratify=labels)
    val_frac = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(tmp, test_size=val_frac, random_state=random_state, stratify=labels[tmp])
    if len(labels) != 3662 or (len(train), len(val), len(test)) != (2928, 367, 367):
        raise RuntimeError(
            "Manuscript-count adjustment expects the APTOS split to be 2928/367/367 "
            f"before adjustment; got {len(train)}/{len(val)}/{len(test)}"
        )
    move = SPLIT_MOVE
    if move == "none":
        return train, val, test
    if move == "c0p0":
        val_labels, val_counts = np.unique(labels[val], return_counts=True)
        selected_class = val_labels[np.argmax(val_counts)]
        moved_position = int(np.flatnonzero(labels[val] == selected_class)[0])
    else:
        m = re.fullmatch(r"c([0-4])p(\d+)", move)
        if not m:
            raise ValueError(f"SPLIT_MOVE must be 'none' or 'c<class>p<position>', got {move!r}")
        class_positions = np.flatnonzero(labels[val] == int(m.group(1)))
        if int(m.group(2)) >= len(class_positions):
            raise ValueError(f"SPLIT_MOVE {move!r}: class has only {len(class_positions)} validation samples")
        moved_position = int(class_positions[int(m.group(2))])
    moved_to_train = val[moved_position:moved_position + 1]
    val = np.delete(val, moved_position)
    train = np.concatenate((train, moved_to_train))
    return train, val, test


class DRDataset(Dataset):
    """Returns (float image CHW in 0-255, one-hot label) exactly like the original dataset.

    augment=True adds label-preserving geometric/photometric augmentation (training only).
    """

    def __init__(self, images, labels, augment=False, num_classes=NUM_CLASSES):
        self.images, self.labels, self.augment, self.nc = images, labels, augment, num_classes

    def __len__(self):
        return len(self.images)

    def _augment(self, x):
        import torchvision.transforms.functional as TF
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        angle = random.uniform(-20.0, 20.0)
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        m = x.mean()
        return ((x - m) * c + m * b).clamp(0.0, 255.0)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        if self.augment:
            x = self._augment(x)
        y = np.zeros(self.nc, dtype=np.float32)
        y[int(self.labels[i])] = 1.0
        return x, torch.from_numpy(y)


def get_data(dataset_path="APTOS2019", batch_size=16, augment_train=False, num_workers=0):
    """Returns dict with loaders, class_weights, ids/labels/indices per split."""
    ids, images, labels = load_aptos(dataset_path)
    ids = np.array(ids)
    if SMOKE:
        if SMOKE_N < 1 or SMOKE_N > len(labels):
            raise ValueError(f"DR_SMOKE_N must be in [1, {len(labels)}], got {SMOKE_N}")
        rng = np.random.RandomState(0)
        selected = []
        classes, counts = np.unique(labels, return_counts=True)
        if SMOKE_N < len(classes):
            raise ValueError(f"DR_SMOKE_N must be at least {len(classes)} to retain all classes")
        per_class = max(1, SMOKE_N // len(classes))
        for cls, count in zip(classes, counts):
            candidates = np.flatnonzero(labels == cls)
            selected.extend(rng.choice(candidates, min(per_class, count), replace=False).tolist())
        remaining = SMOKE_N - len(selected)
        if remaining > 0:
            available = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
            selected.extend(rng.choice(available, remaining, replace=False).tolist())
        keep = np.sort(np.asarray(selected, dtype=np.int64))
        ids, images, labels = ids[keep], images[keep], labels[keep]
        print(f"[SMOKE] using {len(labels)} images only - results are meaningless")
    tr, va, te = make_split_indices(labels)
    pin = torch.cuda.is_available()

    def mk(idx, shuffle, aug):
        return DataLoader(DRDataset(images[idx], labels[idx], augment=aug), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, pin_memory=pin, drop_last=False)

    uniq, cnt = np.unique(labels[tr], return_counts=True)
    class_weights = dict(zip(uniq.tolist(), (len(tr) / (len(uniq) * cnt)).astype(float).tolist()))
    return {
        "train_loader": mk(tr, True, augment_train),
        "val_loader": mk(va, False, False),
        "test_loader": mk(te, False, False),
        "class_weights": class_weights,
        "ids": {"train": ids[tr].tolist(), "val": ids[va].tolist(), "test": ids[te].tolist()},
        "labels": {"train": labels[tr], "val": labels[va], "test": labels[te]},
        "indices": {"train": tr, "val": va, "test": te},
        "images": images,
        "all_ids": ids,
        "all_labels": labels,
    }


def class_weight_tensor(cw, device, n_classes=NUM_CLASSES):
    return torch.tensor([float(cw.get(i, 1.0)) for i in range(n_classes)], dtype=torch.float32, device=device)


def duplicate_groups(images):
    """Groups of indices whose resized pixels are byte-identical."""
    groups = {}
    for i, im in enumerate(images):
        groups.setdefault(hashlib.md5(im.tobytes()).hexdigest(), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


# ------------------------------------------------------------------ metrics
def _mean_specificity(y_true, y_pred, n=NUM_CLASSES):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    s = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        s.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(s))


def compute_full_metrics(y_true, y_pred, y_prob, n=NUM_CLASSES):
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity": _mean_specificity(y_true, y_pred, n),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_absolute_grade_error": float(np.mean(np.abs(y_true - y_pred))),
        "within_one_grade_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "multiclass_brier_score": float(np.mean(np.sum(
            (np.eye(n, dtype=np.float64)[y_true] - y_prob) ** 2, axis=1
        ))),
        "negative_log_likelihood": float(-np.log(y_prob[np.arange(len(y_true)), y_true]).mean()),
        "expected_calibration_error_15_bins": float(ece),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except ValueError:
        m["roc_auc"] = None
    return m


def detailed_metrics(y_true, y_pred, y_prob, n=NUM_CLASSES):
    from sklearn.metrics import classification_report

    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    per_class = classification_report(
        y_true, y_pred, labels=list(range(n)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        per_class[name]["specificity"] = float(tn / (tn + fp)) if tn + fp else 0.0
        binary_target = (y_true == c).astype(np.int32)
        try:
            per_class[name]["roc_auc_ovr"] = float(roc_auc_score(binary_target, y_prob[:, c]))
        except ValueError:
            per_class[name]["roc_auc_ovr"] = None
    return {
        **compute_full_metrics(y_true, y_pred, y_prob, n),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def metrics_from_logits(y_true, logits):
    z = torch.as_tensor(logits, dtype=torch.float64)
    p = torch.softmax(z, dim=1).numpy()
    return compute_full_metrics(y_true, p.argmax(1), p)


# -------------------------------------------------------------- Drive sync
def sync_dirs(src, dst):
    """One-way mirror of src into dst (rsync when available, else copytree)."""
    src, dst = str(src), str(dst)
    os.makedirs(dst, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", src.rstrip("/") + "/", dst.rstrip("/") + "/"], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def start_background_sync(local_dir, drive_dir, interval_s=600):
    """Copies local_dir -> drive_dir every interval_s so a Colab disconnect loses little work."""
    stop = threading.Event()

    def loop():
        while not stop.wait(interval_s):
            try:
                sync_dirs(local_dir, drive_dir)
            except Exception as e:  # keep training alive
                print("[sync] failed:", e)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def env_report():
    import sklearn
    import torchvision
    info = {"torch": torch.__version__, "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        info["timm"] = None
    return info
