"""Table-1 Fusion CNN-ViT baseline (aptos_models_benchmark_pipeline.ipynb, cells 3/4/6) on the frozen split."""
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision import models

from data import nested_subset, preprocess


class FeatureFusionModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        dim_a = resnet.fc.in_features
        resnet.fc = nn.Identity()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        dim_b = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.backbone_a, self.backbone_b = resnet, vit
        self.classifier = nn.Sequential(
            nn.Linear(dim_a + dim_b, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        fa, fb = self.backbone_a(x), self.backbone_b(x)
        return self.classifier(torch.cat([torch.flatten(fa, 1), torch.flatten(fb, 1)], dim=1))


def clahe(image):
    import cv2
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class HistoricalImages(Dataset):
    """CLAHE -> RandomRotation(15, nearest, fill 0) -> ToTensor -> ImageNet normalization (training only rotates)."""

    def __init__(self, images, labels, indices, train):
        self.images, self.labels, self.indices, self.train = images, labels, np.asarray(indices), train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        position = self.indices[index]
        if not self.train:
            return preprocess(self.images[position], "retinal"), torch.tensor(self.labels[position], dtype=torch.long)
        image = Image.fromarray(clahe(self.images[position]))
        angle = float(torch.empty(1).uniform_(-15.0, 15.0))
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST, fill=[0, 0, 0])
        x = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x, torch.tensor(self.labels[position], dtype=torch.long)


def historical_loaders(bundle, seed, batch=8):
    ids, images, labels, splits = bundle
    train = nested_subset(splits["train"], labels, 1.0, seed)
    counts = np.bincount(labels[train], minlength=5).astype(np.float64)
    class_weights = counts.sum() / (5 * counts)
    # Global torch RNG (seeded by the runner and restored on resume) keeps interrupted runs identical.
    sampler = WeightedRandomSampler(torch.tensor(class_weights[labels[train]], dtype=torch.double),
                                    num_samples=len(train), replacement=True)
    pin = torch.cuda.is_available()
    result = {"train": DataLoader(HistoricalImages(images, labels, train, True), batch_size=batch, sampler=sampler,
                                  num_workers=0, pin_memory=pin)}
    for split in ("val", "test"):
        result[split] = DataLoader(HistoricalImages(images, labels, splits[split], False), batch_size=batch,
                                   shuffle=False, num_workers=0, pin_memory=pin)
    return result, torch.tensor(class_weights, dtype=torch.float32), ids[train].tolist()
