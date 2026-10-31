"""
Dual-Expert Attention Model for Diabetic Retinopathy Classification (PyTorch).

Architecture:
    Backbone CNN -> SE Expert + CBAM Expert -> Learnable Weighted Fusion -> GAP -> FC

Identical to the model defined in dual_expert_experiment.ipynb,
extracted here for reusable imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention sub-module for CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module for CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU with residual connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class BackboneCNN(nn.Module):
    """CNN backbone for feature extraction: 224x224x3 -> 7x7x512."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.layer2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.layer3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256))
        self.layer4 = nn.Sequential(ConvBlock(256, 512, stride=2), ConvBlock(512, 512))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DualExpertAttentionModel(nn.Module):
    """
    Dual-Expert Attention Model.

    Architecture:
        1. Backbone CNN extracts feature maps (B, 512, 7, 7)
        2. Expert 1: SE Attention
        3. Expert 2: CBAM Attention
        4. Learnable weighted fusion: fused = alpha * SE + beta * CBAM
        5. GAP -> Dropout(0.5) -> FC -> 5 classes
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = BackboneCNN()
        self.se_expert = SEBlock(512, reduction=16)
        self.cbam_expert = CBAMBlock(512, reduction=16)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        features = self.backbone(x)
        se_features = self.se_expert(features)
        cbam_features = self.cbam_expert(features)
        fused = self.alpha * se_features + self.beta * cbam_features
        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_dual_expert(num_classes=5):
    """Build and return a Dual-Expert Attention model."""
    return DualExpertAttentionModel(num_classes=num_classes)


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
