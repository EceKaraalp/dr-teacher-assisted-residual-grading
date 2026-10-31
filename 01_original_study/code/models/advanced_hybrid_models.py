import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import importlib

try:
    timm = importlib.import_module("timm")
except Exception:
    timm = None

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:
    vit_b_16 = None
    ViT_B_16_Weights = None

# Try to import Deformable Convolution, if not available, create a placeholder
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    print("Warning: DeformConv2d not found. Using standard Conv2d as a fallback for Idea 1.")
    DeformConv2d = nn.Conv2d # Fallback to standard convolution

# --- Helper Modules ---

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, channels, feature_dim):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        self.to_gamma_beta = nn.Linear(feature_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, features):
        gamma, beta = self.to_gamma_beta(features).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # Reshape for broadcasting
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (gamma + 1) + beta

class PrototypeLayer(nn.Module):
    """Learnable prototypes for each class."""
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, x):
        # Calculate squared distances to each prototype
        # (a-b)^2 = a^2 - 2ab + b^2
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_sq = (self.prototypes ** 2).sum(dim=1)
        dot_prod = x @ self.prototypes.t()
        distances = x_sq - 2 * dot_prod + p_sq
        return -distances # Return negative distances as logits

# --- Main Advanced Hybrid Model ---

class AdvancedHybridModel(nn.Module):
    def __init__(self, num_classes=5, config=None):
        super().__init__()
        self.config = config if config is not None else {}
        self.num_classes = num_classes
        
        # --- CNN Branch ---
        try:
            cnn_backbone = resnet50(pretrained=True)
        except Exception as e:
            print(f"Warning: Could not load pretrained ResNet50 weights ({e}). Using randomly initialized ResNet50.")
            cnn_backbone = resnet50(pretrained=False)
        self.cnn_features = nn.Sequential(*list(cnn_backbone.children())[:-2])
        cnn_feature_dim = 2048

        # --- Idea 1: Deformable Convolution for Lesion-Aware Tokenization ---
        if self.config.get('use_deformable_conv', False):
            # Replace the last block of ResNet with a deformable version
            # This is a simplified example. A real implementation might need more careful surgery.
            if DeformConv2d is not nn.Conv2d:
                # Create an offset convolution
                self.offset_conv = nn.Conv2d(cnn_feature_dim, 18, kernel_size=3, padding=1)
                self.deform_conv = DeformConv2d(cnn_feature_dim, cnn_feature_dim, kernel_size=3, padding=1)
            else: # Fallback if DeformConv2d is not available
                self.offset_conv = None
                self.deform_conv = None
        
        # --- ViT Branch ---
        # ViT backend selection:
        # 1) timm (preferred), 2) torchvision fallback.
        if timm is not None:
            try:
                self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            except Exception as e:
                print(f"Warning: Could not load pretrained ViT weights with timm ({e}). Using random-init timm ViT.")
                self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        else:
            if vit_b_16 is None:
                raise ImportError("Neither 'timm' nor torchvision ViT is available. Please install timm.")
            try:
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if ViT_B_16_Weights is not None else None
                self.vit = vit_b_16(weights=weights)
            except Exception as e:
                print(f"Warning: Could not load pretrained torchvision ViT weights ({e}). Using random-init torchvision ViT.")
                self.vit = vit_b_16(weights=None)
            # Make torchvision ViT output features [B, 768]
            self.vit.heads = nn.Identity()
        vit_feature_dim = 768

        # --- Fusion and Head ---
        fusion_method = self.config.get('fusion_method', 'concat')
        
        # --- Idea 4: FiLM Fusion ---
        if fusion_method == 'film':
            self.film_layer = FiLMLayer(channels=cnn_feature_dim, feature_dim=vit_feature_dim)
            fusion_dim = cnn_feature_dim
        else: # Default to concatenation
            fusion_dim = cnn_feature_dim + vit_feature_dim

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        
        # --- Idea 3: Prototype Learning ---
        if self.config.get('use_prototype_head', False):
            self.classifier = PrototypeLayer(num_classes, fusion_dim)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)

        # --- Idea 5: Spectral Normalization ---
        if self.config.get('use_spectral_norm', False):
            self.classifier = nn.utils.spectral_norm(self.classifier)

        # --- Idea 2: Contrastive Alignment ---
        if self.config.get('use_contrastive_loss', False):
            self.cnn_projection = nn.Linear(cnn_feature_dim, 128)
            self.vit_projection = nn.Linear(vit_feature_dim, 128)

    def forward(self, x):
        # --- CNN Branch ---
        cnn_map = self.cnn_features(x)

        # --- Idea 1: Deformable Convolution ---
        if self.config.get('use_deformable_conv', False) and self.deform_conv:
            offsets = self.offset_conv(cnn_map)
            cnn_map = self.deform_conv(cnn_map, offsets)

        cnn_feat_pooled = self.fusion_pool(cnn_map).flatten(1)

        # --- ViT Branch ---
        if hasattr(self.vit, 'forward_features'):
            vit_out = self.vit.forward_features(x)
            # timm version differences: some return token map [B, N, C], others return pooled [B, C]
            if vit_out.ndim == 3:
                vit_cls_token = vit_out[:, 0]
            else:
                vit_cls_token = vit_out
        else:
            # torchvision fallback path: model forward already returns class token features
            vit_cls_token = self.vit(x)

        # --- Fusion ---
        fusion_method = self.config.get('fusion_method', 'concat')
        
        # --- Idea 4: FiLM Fusion ---
        if fusion_method == 'film':
            fused_map = self.film_layer(cnn_map, vit_cls_token)
            final_feat = self.fusion_pool(fused_map).flatten(1)
        else: # Default: Concatenation
            final_feat = torch.cat([cnn_feat_pooled, vit_cls_token], dim=1)

        # --- Classification Head ---
        logits = self.classifier(final_feat)

        # --- Return values for auxiliary losses ---
        extra_outputs = {}
        if self.config.get('use_contrastive_loss', False):
            extra_outputs['cnn_proj'] = F.normalize(self.cnn_projection(cnn_feat_pooled), dim=1)
            extra_outputs['vit_proj'] = F.normalize(self.vit_projection(vit_cls_token), dim=1)
        
        if self.config.get('use_prototype_head', False):
            extra_outputs['final_feat'] = final_feat
            extra_outputs['prototypes'] = self.classifier.prototypes

        return logits, extra_outputs

# --- Loss Functions ---

class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2, labels):
        # Assumes features1 and features2 are from the same samples
        batch_size = features1.shape[0]
        
        # Create a mask to identify positive pairs (same sample, different view)
        # and negative pairs (different samples)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features1.device)
        
        # Concatenate features for easier calculation
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(features.unsqueeze(2), features.unsqueeze(1), dim=3)
        
        # Discard self-similarity
        mask_no_diag = mask.clone()
        mask_no_diag.fill_diagonal_(0)
        
        # Numerator: sum of similarities of positive pairs
        # Denominator: sum of similarities of all pairs
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        
        # For each sample, we have one positive pair (the other view)
        # and N-1 negative pairs from other samples
        log_prob = -torch.log(exp_sim.sum(1) / (exp_sim * mask_no_diag).sum(1))
        
        loss = log_prob.mean()
        return loss

class PrototypicalLoss(nn.Module):
    """Prototypical Loss for metric learning."""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, prototypes, labels):
        # Calculate distances and treat as logits
        # (a-b)^2 = a^2 - 2ab + b^2
        feat_sq = (features ** 2).sum(dim=1, keepdim=True)
        proto_sq = (prototypes ** 2).sum(dim=1)
        dot_prod = features @ prototypes.t()
        distances = feat_sq - 2 * dot_prod + proto_sq
        
        # The loss is cross-entropy on the negative distances
        return self.ce_loss(-distances, labels)
