"""
Transformer Branch for LAMCA-Net
Extracts global representations using Vision Transformer (ViT) or Swin Transformer.
"""
import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

class TransformerBranch(nn.Module):
    """
    Transformer Branch for extracting global features from retinal images.
    Supports ViT (Vision Transformer) and Swin Transformer.
    """
    def __init__(self, backbone="vit_b_16", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "vit_b_16":
            if not VIT_AVAILABLE:
                raise ImportError("torchvision >= 0.13 required for ViT")
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = vit_b_16(weights=weights)
            self.out_dim = self.backbone.hidden_dim
        elif backbone == "swin_t":
            try:
                from torchvision.models import swin_t, Swin_T_Weights
                weights = Swin_T_Weights.DEFAULT if pretrained else None
                self.backbone = swin_t(weights=weights)
                self.out_dim = self.backbone.head.in_features
            except ImportError:
                raise ImportError("torchvision >= 0.13 required for Swin Transformer")
        else:
            raise ValueError(f"Unsupported transformer backbone: {backbone}")

    def forward(self, x):
        # Remove classification head, return features
        if self.backbone_name.startswith("vit"):
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)
            # Return CLS token and optionally all tokens
            return x[:, 0], x  # (CLS, all tokens)
        elif self.backbone_name.startswith("swin"):
            # Swin returns features before head
            x = self.backbone.features(x)
            x = x.mean([-2, -1])  # Global average pooling
            return x, None
        else:
            raise ValueError(f"Unsupported transformer backbone: {self.backbone_name}")

# Example usage:
# trans = TransformerBranch(backbone="vit_b_16", pretrained=True)
# cls_token, all_tokens = trans(torch.randn(1, 3, 224, 224))
