#!/usr/bin/env python3
"""PointNet++ segmentation-only backbone (no force-center head)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet2 import PointNetFeaturePropagation, PointNetSetAbstraction


class PointNet2SegOnly(nn.Module):
    """
    PointNet++ encoder-decoder + per-point MLP head → (B, N) in [0, 1].
    No FC / force-center branch.
    """

    def __init__(
        self,
        in_channel: int = 3,
        head_norm: str = "none",
    ):
        super().__init__()
        self.head_norm_kind = head_norm
        if head_norm == "groupnorm":
            self.head_norm = nn.GroupNorm(1, 128)
        elif head_norm == "layernorm":
            self.head_norm = nn.LayerNorm(128)
        else:
            self.head_norm = None

        self.sa1 = PointNetSetAbstraction(256, 0.05, 32, in_channel + 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.10, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 0.20, 128, 256 + 3, [256, 256, 512])

        self.fp3 = PointNetFeaturePropagation(256 + 512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel + 128, [128, 128, 128])

        self.seg_in_norm = nn.LayerNorm(128)
        self.seg_fc1 = nn.Linear(128, 128)
        self.seg_fc2 = nn.Linear(128, 1)

    def _apply_head_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_norm is None:
            return x
        if isinstance(self.head_norm, nn.LayerNorm):
            x = x.permute(0, 2, 1)
            x = self.head_norm(x)
            return x.permute(0, 2, 1)
        return self.head_norm(x)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        x = self.seg_in_norm(l0_points)
        x = self._apply_head_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = F.relu(self.seg_fc1(x))
        return torch.sigmoid(self.seg_fc2(h).squeeze(-1))
