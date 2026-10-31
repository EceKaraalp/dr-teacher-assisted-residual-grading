"""
Quadratic Weighted Kappa (QWK) Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QWKLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        preds = torch.argmax(logits, dim=1)
        hist2d = torch.zeros((self.num_classes, self.num_classes), device=logits.device)
        for t, p in zip(targets, preds):
            hist2d[int(t), int(p)] += 1
        weights = torch.zeros((self.num_classes, self.num_classes), device=logits.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weights[i, j] = ((i - j) ** 2) / ((self.num_classes - 1) ** 2)
        actual_hist = hist2d.sum(1)
        pred_hist = hist2d.sum(0)
        E = torch.outer(actual_hist, pred_hist) / hist2d.sum()
        O = hist2d
        num = (weights * O).sum()
        den = (weights * E).sum()
        qwk = 1.0 - num / (den + 1e-6)
        return 1.0 - qwk  # minimize loss