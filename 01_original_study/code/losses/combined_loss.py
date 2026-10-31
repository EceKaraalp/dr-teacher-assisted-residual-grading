"""
Combined Loss for LAMCA-Net
CrossEntropy + Focal Loss + QWK Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .focal_loss import FocalLoss
from .qwk_loss import QWKLoss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss()
        self.qwk = QWKLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        focal_loss = self.focal(logits, targets)
        qwk_loss = self.qwk(logits, targets)
        return self.alpha * ce_loss + self.beta * focal_loss + self.gamma * qwk_loss
