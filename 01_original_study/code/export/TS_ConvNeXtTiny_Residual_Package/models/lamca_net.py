"""
LAMCA-Net: Lesion-Aware Multi-Scale Cross-Attention Hybrid Network
Ana model dosyası. Tüm modülleri birleştirir.
"""
import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .transformer_branch import TransformerBranch
from .cross_attention import CrossAttentionFusion
# LesionAttention ve MultiScaleFusion modülleri eklenmeli

class LAMCANet(nn.Module):
    """
    Lesion-Aware Multi-Scale Cross-Attention Hybrid Network
    """
    def __init__(self, num_classes=5, cnn_backbone="efficientnet_b4", trans_backbone="vit_b_16"):
        super().__init__()
        self.cnn_branch = CNNBranch(backbone=cnn_backbone)
        self.transformer_branch = TransformerBranch(backbone=trans_backbone)
        # Örnek: CNN son seviyesini ve transformer CLS tokenını çapraz dikkat ile birleştir
        cnn_dim = self.cnn_branch.out_channels[-1]
        trans_dim = self.transformer_branch.out_dim
        embed_dim = 512
        self.cross_attention = CrossAttentionFusion(cnn_dim, trans_dim, embed_dim)
        # Basit bir classifier (tam entegre için MultiScaleFusion ve LesionAttention eklenmeli)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feats = self.cnn_branch(x)
        # Sadece en son seviyeyi kullanıyoruz (örnek)
        cnn_feat = cnn_feats[-1]  # (B, C, H, W)
        b, c, h, w = cnn_feat.shape
        # EfficientNet: (B, C, H, W) -> (B, H*W, C)
        cnn_feat_flat = cnn_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, N, C)
        cls_token, trans_tokens = self.transformer_branch(x)
        # trans_tokens: (B, M, D)
        # Eğer trans_tokens None ise, sadece cls_token kullan
        if trans_tokens is None:
            # (B, 1, D) olarak genişlet
            trans_tokens = cls_token.unsqueeze(1)
        fused, _ = self.cross_attention(cnn_feat_flat, trans_tokens)
        # Sadece ortalamasını alıp sınıflandırıcıya veriyoruz (örnek)
        fused_pool = fused.mean(dim=1)
        out = self.classifier(fused_pool)
        return out
