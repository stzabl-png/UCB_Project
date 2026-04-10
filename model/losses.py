#!/usr/bin/env python3
"""
Loss functions for affordance training.

- FocalLoss: Handles class imbalance by down-weighting easy examples.
- TverskyLoss: Allows independent tuning of FP/FN penalties.
- CombinedLoss: Weighted sum of Focal + Tversky.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: (B*N, 2), target: (B*N,)
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — independently weights FP and FN penalties."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight (low = tolerate FP)
        self.beta = beta    # FN weight (high = penalize FN)
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B*N, 2), target: (B*N,)
        prob = F.softmax(pred, dim=-1)[:, 1]
        target_f = target.float()

        tp = (prob * target_f).sum()
        fp = (prob * (1 - target_f)).sum()
        fn = ((1 - prob) * target_f).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Focal + Tversky combined loss."""

    def __init__(self, focal_weight=0.6, tversky_weight=0.4):
        super().__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.fw = focal_weight
        self.tw = tversky_weight

    def forward(self, pred, target):
        return self.fw * self.focal(pred, target) + self.tw * self.tversky(pred, target)
