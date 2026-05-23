#!/usr/bin/env python3
"""PointNet++ affordance training helpers (re-exports backbone + prob/center ops)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.pointnet2 import PointNet2Seg


def forward_seg_fc(
    model: torch.nn.Module,
    xyz: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segmentation logits + force center (zeros if head disabled)."""
    out = model(xyz, features)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, torch.zeros(xyz.shape[0], 3, device=xyz.device, dtype=xyz.dtype)


def affordance_probability(seg_logits: torch.Tensor) -> torch.Tensor:
    """Per-point affordance score in [0, 1]. seg_logits: (B, N, 2)."""
    if seg_logits.shape[-1] == 1:
        return torch.sigmoid(seg_logits.squeeze(-1))
    return F.softmax(seg_logits, dim=-1)[..., 1]


def center_from_affordance(
    prob: torch.Tensor,
    xyz: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Heatmap-induced center: Σ(p_i x_i) / Σ(p_i).
    prob: (B, N), xyz: (B, N, 3) → (B, 3)
    """
    w = prob.unsqueeze(-1)
    num = (w * xyz).sum(dim=1)
    den = prob.sum(dim=1, keepdim=True).clamp_min(eps)
    return num / den


def fc_valid_mask(fc_gt: torch.Tensor, threshold: float = 0.001) -> torch.Tensor:
    return fc_gt.norm(dim=1) > threshold


def contact_valid_mask(labels: torch.Tensor) -> torch.Tensor:
    """At least one positive contact point in the sample."""
    if labels.dim() == 3:
        labels = labels.reshape(labels.shape[0], -1)
    return labels.reshape(labels.shape[0], -1).sum(dim=1) > 0
