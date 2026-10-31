"""
CViTS-Net Architecture Implementation (PyTorch)
Exact implementation as specified in the IEEE Access 2024 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """
    Converts image to patch embeddings with positional encoding.
    Input: (B, 3, 224, 224) -> Patches of 16x16 -> 196 patches -> 768 dimensions
    """
    
    def __init__(self, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (224 // patch_size) ** 2  # 196
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * 3, embedding_dim)
        
        # Sinusoidal positional encoding (fixed, not learned)
        pe = self._get_sinusoidal_positional_encoding(self.num_patches, embedding_dim)
        self.register_buffer('positional_embedding', torch.from_numpy(pe).float())
    
    def _get_sinusoidal_positional_encoding(self, seq_len: int, d: int) -> np.ndarray:
        """Generate sinusoidal positional encodings."""
        angle_rads = self._get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(d)[np.newaxis, :], d
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return angle_rads  # (seq_len, d)
    
    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        B = x.shape[0]
        
        # Extract patches: unfold spatial dims
        # (B, 3, 224, 224) -> (B, 3, 14, 16, 14, 16)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, 3, 14, 14, 16, 16)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)  # (B, 14, 14, 3, 16, 16)
        x = x.reshape(B, self.num_patches, self.patch_size * self.patch_size * 3)  # (B, 196, 768)
        
        # Project to embedding dimension
        embeddings = self.projection(x)  # (B, 196, 768)
        
        # Add positional embeddings
        embeddings = embeddings + self.positional_embedding.unsqueeze(0)
        
        return embeddings


class MultiScaleFeatureEnhancement(nn.Module):
    """
    Multi-Scale Feature Enhancement using Atrous (Dilated) Convolution.
    Three parallel branches with different dilation rates.
    """
    
    def __init__(self, channels: int = 768):
        super().__init__()
        self.channels = channels
        
        # Three parallel convolutions with different dilation rates (channels-first)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        
        # 1x1 conv for fusion
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, 196, 768) -> reshape to (B, 768, 14, 14) for convolutions
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, self.channels, 14, 14)
        
        # Three parallel branches
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        
        # Element-wise addition
        fused = out1 + out2 + out3
        
        # 1x1 convolution and ReLU
        output = F.relu(self.fusion_conv(fused))
        
        # Flatten back to (B, 196, 768)
        output = output.reshape(B, self.channels, 196).permute(0, 2, 1)
        
        return output


class DepthwiseSeparableConvolution(nn.Module):
    """Depthwise Separable Convolution for local feature extraction (DWC + PWC)."""
    
    def __init__(self, channels: int = 768, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        
        # Depthwise convolution (groups=channels means each channel is convolved independently)
        self.dwc = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                             padding=kernel_size // 2, groups=channels)
        # Pointwise convolution (1x1, channel mixing)
        self.pwc = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, 196, 768) -> (B, 768, 14, 14)
        B = x.shape[0]
        x_reshaped = x.permute(0, 2, 1).reshape(B, self.channels, 14, 14)
        
        # Depthwise convolution
        output = self.dwc(x_reshaped)
        # Pointwise convolution
        output = self.pwc(output)
        
        # Reshape back to (B, 196, 768)
        output = output.reshape(B, self.channels, 196).permute(0, 2, 1)
        return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention for global feature extraction (4 heads per paper)."""
    
    def __init__(self, num_heads: int = 4, embedding_dim: int = 768):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        # x: (B, 196, 768)
        B, N, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (B, 196, 2304)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, 196, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.embedding_dim)
        
        output = self.fc_out(context)
        return output


class DualGlobalLocalFeatureBlock(nn.Module):
    """
    Dual Global-Local Feature Block (DGL)
    Combines depthwise separable convolution (local) and multi-head attention (global).
    """
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dsc = DepthwiseSeparableConvolution(channels=embedding_dim)
        self.mha = MultiHeadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim)
    
    def forward(self, x):
        # x: (B, 196, 768)
        x_norm = self.layer_norm(x)
        
        # Branch 1: local features
        local_features = self.dsc(x_norm)
        
        # Branch 2: global features
        global_features = self.mha(x_norm)
        
        # Fuse: element-wise addition + residual
        output = local_features + global_features + x
        return output


class MLPBlock(nn.Module):
    """
    Feed-forward MLP block with exact architecture from paper.
    Dense(128)->GELU->Dense(64)->GELU with dropout and residual.
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128,
                 hidden_dim2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim2)
        self.dense3 = nn.Linear(hidden_dim2, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, 196, 768)
        hidden = F.gelu(self.dense1(x))
        hidden = self.dropout(hidden)
        hidden = F.gelu(self.dense2(hidden))
        hidden = self.dropout(hidden)
        output = self.dense3(hidden)
        output = self.dropout(output)
        
        # Residual connection
        return x + output


class ModifiedEncoderBlock(nn.Module):
    """Modified Encoder Block combining DGL and MLP."""
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 4,
                 hidden_dim: int = 128, hidden_dim2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dgl = DualGlobalLocalFeatureBlock(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, hidden_dim, hidden_dim2, dropout)
    
    def forward(self, x):
        x = self.dgl(x)
        x = self.mlp(x)
        return x


class ClassificationHead(nn.Module):
    """Classification head: LayerNorm -> Flatten -> Dropout -> Dense -> Softmax"""
    
    def __init__(self, embedding_dim: int = 768, num_patches: int = 196,
                 num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(num_patches * embedding_dim, num_classes)
    
    def forward(self, x):
        # x: (B, 196, 768)
        x = self.layer_norm(x)
        x = x.reshape(x.shape[0], -1)  # Flatten: (B, 196*768)
        x = self.dropout(x)
        output = self.dense(x)  # (B, num_classes) — raw logits
        return output


class CViTSNet(nn.Module):
    """
    Complete CViTS-Net model.
    
    Architecture:
    1. Patch Embedding (sinusoidal positional encoding)
    2. Multi-Scale Feature Enhancement (MSF)
    3. 4 Modified Encoder Blocks with skip connections (ME1->ME3, ME2->ME4)
    4. Classification Head
    """
    
    def __init__(self, num_classes: int = 5, image_size: int = 224):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=16, embedding_dim=768)
        self.msfe = MultiScaleFeatureEnhancement(channels=768)
        
        self.me1 = ModifiedEncoderBlock(768, num_heads=4, hidden_dim=128, hidden_dim2=64)
        self.me2 = ModifiedEncoderBlock(768, num_heads=4, hidden_dim=128, hidden_dim2=64)
        self.me3 = ModifiedEncoderBlock(768, num_heads=4, hidden_dim=128, hidden_dim2=64)
        self.me4 = ModifiedEncoderBlock(768, num_heads=4, hidden_dim=128, hidden_dim2=64)
        
        self.head = ClassificationHead(embedding_dim=768, num_patches=196,
                                       num_classes=num_classes, dropout=0.5)
    
    def forward(self, x):
        # x: (B, 3, 224, 224) uint8 or float
        # Normalize to [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Patch Embedding
        x = self.patch_embed(x)        # (B, 196, 768)
        
        # Multi-Scale Feature Enhancement
        x = self.msfe(x)               # (B, 196, 768)
        
        # Modified Encoder Blocks with skip connections
        me1 = self.me1(x)              # ME1
        me2 = self.me2(me1)            # ME2
        me3 = self.me3(me1 + me2)      # ME3 (skip from ME1)
        me4 = self.me4(me2 + me3)      # ME4 (skip from ME2)
        
        # Classification Head
        output = self.head(me4)         # (B, num_classes)
        return output


def build_cvitsnet(num_classes: int = 5, image_size: int = 224) -> CViTSNet:
    """
    Build complete CViTS-Net model.
    
    Args:
        num_classes: Number of output classes (5 for APTOS2019)
        image_size: Input image size (224x224)
        
    Returns:
        CViTSNet model (PyTorch nn.Module)
    """
    model = CViTSNet(num_classes=num_classes, image_size=image_size)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
