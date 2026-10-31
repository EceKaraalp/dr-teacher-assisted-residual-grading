"""
Cross-Attention Fusion Module for LAMCA-Net
Implements multi-head cross-attention between CNN and Transformer features.
"""
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion Module.
    Queries: CNN features
    Keys/Values: Transformer features
    """
    def __init__(self, cnn_dim, trans_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(cnn_dim, embed_dim)
        self.key_proj = nn.Linear(trans_dim, embed_dim)
        self.value_proj = nn.Linear(trans_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, cnn_feat, trans_feat):
        """
        cnn_feat: (B, N, C) - CNN features (flattened spatial dims)
        trans_feat: (B, M, D) - Transformer features (tokens)
        """
        Q = self.query_proj(cnn_feat)
        K = self.key_proj(trans_feat)
        V = self.value_proj(trans_feat)
        attn_out, attn_weights = self.attn(Q, K, V)
        out = self.out_proj(attn_out)
        return out, attn_weights

# Example usage:
# fusion = CrossAttentionFusion(cnn_dim=256, trans_dim=768, embed_dim=512, num_heads=4)
# fused, attn = fusion(cnn_feat, trans_feat)
