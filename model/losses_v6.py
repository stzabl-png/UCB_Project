#!/usr/bin/env python3
"""
Loss functions for PointNet++ v6 — Continuous Affordance Heatmap.

Primary: Weighted L1 loss on soft Gaussian heatmap labels.
"""

import torch
import torch.nn as nn


class WeightedL1HeatmapLoss(nn.Module):
    """Weighted L1 loss for continuous affordance heatmaps.

    Applies higher weight to regions with non-zero soft labels to prevent
    the model from collapsing to all-zeros (since ~74% of points have
    soft_label ≈ 0).

    Args:
        bg_weight: weight for background points (soft_label < threshold)
        fg_weight: weight for foreground points (soft_label >= threshold)
        threshold: boundary between bg and fg (default 0.05)
    """

    def __init__(self, bg_weight=1.0, fg_weight=5.0, threshold=0.05):
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.threshold = threshold

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N) predicted heatmap in [0, 1]
            target: (B, N) soft label heatmap in [0, 1]
        Returns:
            scalar loss
        """
        l1 = torch.abs(pred - target)  # (B, N)

        # Weighted: foreground regions get higher weight
        weight = torch.where(
            target >= self.threshold,
            torch.tensor(self.fg_weight, device=pred.device),
            torch.tensor(self.bg_weight, device=pred.device),
        )

        weighted_l1 = (l1 * weight).mean()
        return weighted_l1


class SmoothL1HeatmapLoss(nn.Module):
    """Smooth L1 (Huber) variant with foreground weighting.

    Less sensitive to outliers than pure L1 near zero, more stable gradients.
    """

    def __init__(self, bg_weight=1.0, fg_weight=5.0, threshold=0.05, beta=0.1):
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.threshold = threshold
        self.beta = beta

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        smooth_l1 = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )

        weight = torch.where(
            target >= self.threshold,
            torch.tensor(self.fg_weight, device=pred.device),
            torch.tensor(self.bg_weight, device=pred.device),
        )

        return (smooth_l1 * weight).mean()
