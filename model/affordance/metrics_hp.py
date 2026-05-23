#!/usr/bin/env python3
"""Validation visualization for human-prior supervision training."""

from __future__ import annotations

import torch.nn as nn

from model.affordance.metrics import save_val_objects_grid


class _SegOnlyForwardWrapper(nn.Module):
    """Makes PointNet2SegOnly compatible with save_val_objects_grid / forward_seg_fc."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, xyz, features):
        return self.inner(xyz, features)


class _HpVisDataset:
    """GT rows = HP supervision; row-1 panel = robot_gt for comparison."""

    def __init__(self, base):
        self._base = base
        self.sample_obj_ids = base.sample_obj_ids
        if hasattr(base, "human_priors"):
            self.human_priors = base.human_priors

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        pts, feat, lbl, soft, robot, _hp = self._base[idx]
        return pts, feat, lbl, soft, robot.new_zeros(3), robot


def save_hp_objects_grid(
    model,
    dataset,
    sample_obj_ids,
    device,
    save_path: str,
    epoch: int,
    *,
    vis_object_ids: list[str] | None = None,
    title_prefix: str = "Val",
):
    wrapped = _SegOnlyForwardWrapper(model)
    vis_ds = _HpVisDataset(dataset)
    save_val_objects_grid(
        wrapped,
        vis_ds,
        sample_obj_ids,
        device,
        save_path,
        epoch,
        vis_object_ids=vis_object_ids,
        title_prefix=title_prefix,
    )
