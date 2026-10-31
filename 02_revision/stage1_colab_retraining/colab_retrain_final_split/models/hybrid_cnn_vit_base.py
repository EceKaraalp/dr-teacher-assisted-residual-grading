"""
Base Hybrid CNN+ViT Architecture for APTOS 2019 Diabetic Retinopathy Classification.

This modular architecture supports all 10 research ideas through configuration-driven
modifications at key components: fusion strategy, attention mechanisms, token refinement,
and calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class ChannelSpatialAttention(nn.Module):
    """Channel and spatial attention module for lesion localization."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa
        return x


class PatchEmbedding(nn.Module):
    """Convert image features to ViT-compatible patches."""
    
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class SimpleTransformerBlock(nn.Module):
    """Minimal transformer block for token processing."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_dim: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
    
    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x


class ConfidenceGatedFusion(nn.Module):
    """Idea 1: Confidence-gated local-global fusion."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.cnn_conf = nn.Linear(feat_dim, 1)
        self.vit_conf = nn.Linear(feat_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        # Compute confidences
        s_c = torch.sigmoid(self.cnn_conf(cnn_feat))  # (B, 1)
        s_t = torch.sigmoid(self.vit_conf(vit_feat))  # (B, 1)
        
        # Gate
        conf_cat = torch.cat([s_c, s_t], dim=1)  # (B, 2)
        alpha = self.gate(conf_cat)  # (B, 1)
        
        # Fuse
        fused = alpha * cnn_feat + (1 - alpha) * vit_feat
        return fused


class MultiScaleSpatialChannelAttention(nn.Module):
    """Idea 2: Multi-scale spatial-channel co-attention pyramid."""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        self.scales = nn.ModuleList([
            ChannelSpatialAttention(c) for c in channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [scale(feat) for scale, feat in zip(self.scales, features)]


class UncertaintyTokenRefiner(nn.Module):
    """Idea 3: Uncertainty-driven token refinement."""
    
    def __init__(self, embed_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, tokens: torch.Tensor, keep_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, num_patches, embed_dim)
        uncertainty = self.uncertainty_head(tokens)  # (B, num_patches, 1)
        
        # Retention score
        retention = torch.exp(-self.beta * uncertainty)  # (B, num_patches, 1)
        
        # Soft pruning
        refined_tokens = tokens * retention
        
        return refined_tokens, uncertainty.squeeze(-1)


class ProtoypeMemoryBank(nn.Module):
    """Idea 5: Prototype memory for class-aware learning."""
    
    def __init__(self, num_classes: int, feat_dim: int, num_prototypes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        # Prototypes per class
        self.register_buffer('prototypes', torch.randn(num_classes * num_prototypes, feat_dim))
        self.register_buffer('prototype_labels', torch.repeat_interleave(
            torch.arange(num_classes), num_prototypes))
    
    def update(self, features: torch.Tensor, labels: torch.Tensor, momentum: float = 0.99):
        """Update prototypes with momentum."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_features = features[mask].mean(0)
                idx_start = c * self.num_prototypes
                idx_end = idx_start + self.num_prototypes
                self.prototypes[idx_start:idx_end] = momentum * self.prototypes[idx_start:idx_end] + \
                                                     (1 - momentum) * class_features.unsqueeze(0)
    
    def get_prototypes(self, label: int) -> torch.Tensor:
        idx_start = label * self.num_prototypes
        idx_end = idx_start + self.num_prototypes
        return self.prototypes[idx_start:idx_end]


class OrdinalDistributionHead(nn.Module):
    """Idea 4: Ordinal-distribution aware head for QWK optimization."""
    
    def __init__(self, feat_dim: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.dist_model = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.dist_model(x)
        # Output class probabilities and ordinal logits
        class_probs = F.softmax(logits, dim=1)
        return logits, class_probs


class HybridCNNViTBase(nn.Module):
    """
    Base Hybrid Architecture supporting all 10 research ideas.
    
    Args:
        num_classes: Number of output classes (5 for APTOS)
        config: Dict with variant configurations
    """
    
    def __init__(self, num_classes: int = 5, config: Optional[Dict] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.config = config or {}
        
        # CNN Branch (ResNet-18 feature extractor)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self._make_cnn_layer(64, 64, 2, stride=1),
            *self._make_cnn_layer(64, 128, 2, stride=2),
            *self._make_cnn_layer(128, 256, 2, stride=2),
        )
        
        # Idea 2: Multi-scale spatial-channel attention
        self.multi_scale_attn = MultiScaleSpatialChannelAttention([64, 128, 256])
        
        # CNN to features
        self.cnn_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Patch embedding for ViT
        self.patch_embed = PatchEmbedding(3, 256, patch_size=16)
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[SimpleTransformerBlock(256, num_heads=8) for _ in range(6)]
        )
        
        # Idea 3: Uncertainty token refinement
        if self.config.get('use_uncertainty_refinement', False):
            self.token_refiner = UncertaintyTokenRefiner(256)
        
        # Idea 1: Confidence-gated fusion
        self.fusion_method = self.config.get('fusion_method', 'gate')
        if self.fusion_method == 'gate':
            self.fusion = ConfidenceGatedFusion(512)
        else:  # concat fusion
            self.fusion_concat = nn.Linear(1024, 512)
        
        # ViT to features
        self.vit_fc = nn.Linear(256, 512)
        
        # Idea 4: Ordinal distribution head
        if self.config.get('use_ordinal_head', False):
            self.output_head = OrdinalDistributionHead(512, num_classes)
        else:
            self.output_head = nn.Linear(512, num_classes)
        
        # Idea 5: Prototype memory
        if self.config.get('use_prototype_memory', False):
            self.prototype_bank = ProtoypeMemoryBank(num_classes, 512)
    
    def _make_cnn_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: CNN
        cnn_feat = self.cnn_backbone(x)  # (B, 256, H', W')
        cnn_pool = self.cnn_fc(cnn_feat)  # (B, 512)
        
        # Branch 2: ViT
        vit_tokens = self.patch_embed(x)  # (B, num_patches, 256)
        
        # Idea 3: Token refinement
        if self.config.get('use_uncertainty_refinement', False):
            vit_tokens, uncertainties = self.token_refiner(vit_tokens)
        
        vit_tokens = self.transformer(vit_tokens)  # (B, num_patches, 256)
        vit_feat = vit_tokens.mean(1)  # (B, 256)
        vit_pool = self.vit_fc(vit_feat)  # (B, 512)
        
        # Fusion
        if self.fusion_method == 'gate':
            fused = self.fusion(cnn_pool, vit_pool)
        else:  # concat
            fused = torch.cat([cnn_pool, vit_pool], dim=1)
            fused = self.fusion_concat(fused)
        
        # Output
        if self.config.get('use_ordinal_head', False):
            logits, probs = self.output_head(fused)
            return logits
        else:
            logits = self.output_head(fused)
            return logits
