import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nested_subset(indices, labels, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected = []
    for grade in range(5):
        members = np.asarray(indices)[labels[indices] == grade].copy()
        rng.shuffle(members)
        selected.extend(members[:max(1, int(np.floor(len(members) * fraction)))])
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_data(folder, manifest):
    folder = Path(folder)
    expected = {
        "train.csv": "b2a3479695922e62c83c94a1a9ef669b4b6d68cc26a1d7090ebedecdd7dd79a5",
        "cache_224.npz": "03ae18b25fd50af259dd43a709415dde8bf5ab612c5f8a235f00ccb1a4d4f84e",
    }
    for name, digest in expected.items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"Source hash mismatch: {name}. Do not silently replace the verified cache/CSV pair.")
    frame = pd.read_csv(folder / "train.csv")
    if len(frame) != 3662 or not frame.id_code.is_unique:
        raise ValueError("Expected 3662 unique APTOS image IDs.")
    with np.load(folder / "cache_224.npz", allow_pickle=False) as cache:
        images = cache["images"].copy()
        labels = cache["labels"].astype(np.int64)
    if images.shape != (3662, 224, 224, 3) or images.dtype != np.uint8:
        raise ValueError("Invalid cache shape or dtype.")
    if not np.array_equal(labels, frame.diagnosis.to_numpy()):
        raise ValueError("Cache labels do not match train.csv.")
    mapping = json.loads(Path(manifest).read_text(encoding="utf-8"))["sample_ids"]
    positions = {str(value): i for i, value in enumerate(frame.id_code)}
    indices = {split: np.asarray([positions[str(v)] for v in mapping[split]], dtype=np.int64)
               for split in ("train", "val", "test")}
    if tuple(len(indices[k]) for k in ("train", "val", "test")) != (2928, 367, 367):
        raise ValueError("Unexpected frozen split sizes.")
    if len(set(np.concatenate(list(indices.values())).tolist())) != 3662:
        raise ValueError("Split overlap or missing IDs.")
    return frame.id_code.to_numpy(dtype=str), images, labels, indices


def preprocess(image, mode):
    if mode not in {"raw", "imagenet", "retinal"}:
        raise ValueError(mode)
    if mode == "retinal":
        import cv2
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    if mode != "raw":
        x = TF.normalize(x / 255.0, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x


class Images(Dataset):
    def __init__(self, images, labels, indices, mode, augment=False):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)
        self.mode, self.augment = mode, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        position = self.indices[index]
        image = self.images[position]
        if self.augment:
            image = Image.fromarray(image)
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.uniform(-20, 20), interpolation=TF.InterpolationMode.BILINEAR)
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
            image = np.asarray(image)
        return preprocess(image, self.mode), torch.tensor(self.labels[position], dtype=torch.long)


def loaders(bundle, run):
    ids, images, labels, splits = bundle
    train = nested_subset(splits["train"], labels, run.fraction, run.seed)
    counts = np.bincount(labels[train], minlength=5)
    if (counts == 0).any():
        raise ValueError("Training subset lost a class.")
    weights = torch.tensor(len(train) / (5 * counts), dtype=torch.float32)
    result = {}
    for split in ("train", "val", "test"):
        idx = train if split == "train" else splits[split]
        result[split] = DataLoader(
            Images(images, labels, idx, run.preprocessing, run.arm == "augmentation" and split == "train"),
            batch_size=16, shuffle=split == "train", num_workers=0, pin_memory=torch.cuda.is_available(),
        )
    return result, weights, ids[train].tolist()
