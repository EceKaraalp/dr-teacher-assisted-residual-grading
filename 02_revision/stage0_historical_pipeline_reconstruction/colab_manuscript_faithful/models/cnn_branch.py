"""
CNN Branch for LAMCA-Net
Extracts multi-scale local features using EfficientNet-B4 or DenseNet121.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class CNNBranch(nn.Module):
    """
    CNN Branch for extracting multi-scale features from retinal images.
    Supports EfficientNet-B4 and DenseNet121 backbones.
    """
    def __init__(self, backbone="efficientnet_b4", pretrained=True, out_indices=(2, 4, 6)):
        super().__init__()
        self.backbone_name = backbone
        self.out_indices = out_indices
        if backbone == "efficientnet_b4":
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
            self.out_channels = [self.backbone.features[idx][0].out_channels if isinstance(self.backbone.features[idx], nn.Sequential) else self.backbone.features[idx].out_channels for idx in out_indices]
        elif backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            self.out_channels = [self.backbone.features[idx].num_features if hasattr(self.backbone.features[idx], 'num_features') else self.backbone.features[idx].out_channels for idx in out_indices]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        features = []
        if self.backbone_name == "efficientnet_b4":
            feat = x
            for i, block in enumerate(self.backbone.features):
                feat = block(feat)
                if i in self.out_indices:
                    features.append(feat)
            return features
        elif self.backbone_name == "densenet121":
            feat = x
            for i, block in enumerate(self.backbone.features):
                feat = block(feat)
                if i in self.out_indices:
                    features.append(feat)
            return features
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

# Example usage:
# cnn = CNNBranch(backbone="efficientnet_b4", pretrained=True)
# feats = cnn(torch.randn(1, 3, 224, 224))
