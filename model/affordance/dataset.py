#!/usr/bin/env python3
"""Dataset with Gaussian soft affordance heatmaps (train_affordance pipeline)."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from model.affordance.augment import AugmentConfig
from model.affordance.heatmap import (
    gaussian_soft_heatmap,
    object_bbox_diagonal,
    sigma_from_scale,
)
from model.train import MultiTaskDataset

# PointNet++ per-point features (xyz only; normals are not fed to the network).
AFFORDANCE_IN_CHANNELS = 3


class SoftAffordanceDataset(MultiTaskDataset):
    """
    MultiTaskDataset + soft heatmap GT + obj_id for visualization.
    Heatmap is built after augmentation from binary labels on augmented xyz.
    """

    def __init__(
        self,
        h5_path: str,
        obj_ids_to_use=None,
        *,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        heatmap_sigma_ratio: float = 0.05,
        synthetic_label: str | None = None,
    ):
        self.sample_obj_ids: list[str] = []
        self.heatmap_sigma_ratio = float(heatmap_sigma_ratio)
        self.synthetic_label = synthetic_label
        if augment_config is not None:
            self.augment_cfg = augment_config
        elif augment:
            self.augment_cfg = AugmentConfig()
        else:
            self.augment_cfg = AugmentConfig(False, False, False, False, False)
        self.augment = any(
            (
                self.augment_cfg.rotation,
                self.augment_cfg.scale,
                self.augment_cfg.shift,
                self.augment_cfg.jitter,
                self.augment_cfg.dropout,
            ),
        )
        with h5py.File(h5_path, "r") as f:
            all_obj_ids = f["data/obj_ids"][:]
            n_total = len(f["data/points"])
            n_pts = int(f["data/points"].shape[1])
            if "data/human_priors" in f:
                all_hp = f["data/human_priors"][:]
            else:
                all_hp = np.zeros((n_total, n_pts), dtype=np.float32)
        decoded = [s.decode() if isinstance(s, bytes) else s for s in all_obj_ids]
        if obj_ids_to_use is not None:
            use = set(obj_ids_to_use)
            mask = np.array([d in use for d in decoded])
            self.sample_obj_ids = [d for d in decoded if d in use]
            self.human_priors = all_hp[mask]
        else:
            self.sample_obj_ids = decoded
            self.human_priors = all_hp
        super().__init__(h5_path, obj_ids_to_use, augment=self.augment)

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        lbl = self.labels[idx].copy()
        fc = self.force_centers[idx].copy()
        hp = self.human_priors[idx].copy()
        cfg = self.augment_cfg

        if self.augment:
            if cfg.rotation:
                z = np.random.randn(3, 3).astype(np.float32)
                q, r = np.linalg.qr(z)
                d = np.diagonal(r)
                ph = d / np.abs(d)
                R = (q @ np.diag(ph)).astype(np.float32)
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1
                pts = pts @ R.T
                fc = fc @ R.T

            if cfg.scale:
                scale = np.random.uniform(0.8, 1.2)
                pts *= scale
                fc *= scale

            if cfg.shift:
                shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
                pts += shift
                fc += shift.flatten()

            if cfg.jitter:
                pts += np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)

            if cfg.dropout and np.random.rand() < 0.3:
                n = len(pts)
                keep = np.random.choice(n, int(n * 0.9), replace=False)
                drop = np.setdiff1d(np.arange(n), keep)
                fill = np.random.choice(keep, len(drop), replace=True)
                pts[drop] = pts[fill]
                lbl[drop] = lbl[fill]
                hp[drop] = hp[fill]

        if self.synthetic_label == "x_positive":
            lbl_binary = (pts[:, 0] > np.median(pts[:, 0])).astype(np.float32)
        elif self.synthetic_label == "z_positive":
            lbl_binary = (pts[:, 2] > np.median(pts[:, 2])).astype(np.float32)
        else:
            lbl_binary = (lbl > 0.5).astype(np.float32)
        obj_scale = object_bbox_diagonal(pts)
        sigma = sigma_from_scale(obj_scale, self.heatmap_sigma_ratio)
        soft_aff = gaussian_soft_heatmap(pts, lbl_binary > 0.5, sigma)

        features = pts
        return (
            torch.from_numpy(pts),
            torch.from_numpy(features),
            torch.from_numpy(lbl_binary.astype(np.int64)),
            torch.from_numpy(soft_aff),
            torch.from_numpy(fc).float(),
            torch.from_numpy(hp.astype(np.float32)),
        )
