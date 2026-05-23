#!/usr/bin/env python3
"""Affordance dataset with human_prior as supervision (not robot_gt)."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from model.affordance.augment import AugmentConfig
from model.affordance.dataset import AFFORDANCE_IN_CHANNELS
from model.train import MultiTaskDataset


class HumanPriorAffordanceDataset(MultiTaskDataset):
    """
    Same HDF5 layout as SoftAffordanceDataset, but:
      - soft GT  = human_prior (clipped to [0, 1])
      - binary GT = human_prior > hp_threshold
    Robot labels kept in self.robot_labels for visualization only.
    """

    def __init__(
        self,
        h5_path: str,
        obj_ids_to_use=None,
        *,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        hp_threshold: float = 0.5,
    ):
        self.sample_obj_ids: list[str] = []
        self.hp_threshold = float(hp_threshold)
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
            all_robot = f["data/labels"][:]
        decoded = [s.decode() if isinstance(s, bytes) else s for s in all_obj_ids]
        if obj_ids_to_use is not None:
            use = set(obj_ids_to_use)
            mask = np.array([d in use for d in decoded])
            self.sample_obj_ids = [d for d in decoded if d in use]
            self.human_priors = all_hp[mask]
            self.robot_labels = all_robot[mask]
        else:
            self.sample_obj_ids = decoded
            self.human_priors = all_hp
            self.robot_labels = all_robot
        super().__init__(h5_path, obj_ids_to_use, augment=self.augment)
        # MultiTaskDataset.labels is robot_gt; supervision uses human_priors only.

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        hp = self.human_priors[idx].copy()
        robot = self.robot_labels[idx].copy()
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

            if cfg.scale:
                scale = np.random.uniform(0.8, 1.2)
                pts *= scale

            if cfg.shift:
                shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
                pts += shift

            if cfg.jitter:
                pts += np.random.normal(0, 0.002, size=pts.shape).astype(np.float32)

            if cfg.dropout and np.random.rand() < 0.3:
                n = len(pts)
                keep = np.random.choice(n, int(n * 0.9), replace=False)
                drop = np.setdiff1d(np.arange(n), keep)
                fill = np.random.choice(keep, len(drop), replace=True)
                pts[drop] = pts[fill]
                hp[drop] = hp[fill]
                robot[drop] = robot[fill]

        soft_gt = np.clip(hp, 0.0, 1.0).astype(np.float32)
        lbl_binary = (hp > self.hp_threshold).astype(np.int64)
        features = pts
        return (
            torch.from_numpy(pts),
            torch.from_numpy(features),
            torch.from_numpy(lbl_binary),
            torch.from_numpy(soft_gt),
            torch.from_numpy(robot.astype(np.float32)),
            torch.from_numpy(soft_gt.copy()),
        )


def build_train_val_datasets_hp(
    dataset_dir: str,
    train_obj_ids: set[str],
    val_obj_ids: set[str],
    *,
    hp_threshold: float = 0.5,
    train_augment_config: AugmentConfig | None = None,
) -> tuple[HumanPriorAffordanceDataset, HumanPriorAffordanceDataset]:
    """Same object split merge as train.build_train_val_datasets, HP supervision."""
    train_h5 = f"{dataset_dir.rstrip('/')}/affordance_train.h5"
    val_h5 = f"{dataset_dir.rstrip('/')}/affordance_val.h5"
    no_aug = AugmentConfig(False, False, False, False, False)
    train_aug = train_augment_config or AugmentConfig()

    train_dataset = HumanPriorAffordanceDataset(
        train_h5, train_obj_ids, augment_config=train_aug, hp_threshold=hp_threshold,
    )
    val_from_val = HumanPriorAffordanceDataset(
        val_h5, val_obj_ids, augment_config=no_aug, hp_threshold=hp_threshold,
    )
    val_from_train = HumanPriorAffordanceDataset(
        train_h5, val_obj_ids, augment_config=no_aug, hp_threshold=hp_threshold,
    )

    val_dataset = val_from_val
    val_dataset.points = np.concatenate([val_from_val.points, val_from_train.points])
    val_dataset.normals = np.concatenate([val_from_val.normals, val_from_train.normals])
    val_dataset.labels = np.concatenate([val_from_val.labels, val_from_train.labels])
    val_dataset.human_priors = np.concatenate(
        [val_from_val.human_priors, val_from_train.human_priors],
    )
    val_dataset.robot_labels = np.concatenate(
        [val_from_val.robot_labels, val_from_train.robot_labels],
    )
    val_dataset.sample_obj_ids = val_from_val.sample_obj_ids + val_from_train.sample_obj_ids
    val_dataset.num_samples = len(val_dataset.points)

    train_from_val = HumanPriorAffordanceDataset(
        val_h5, train_obj_ids, augment_config=train_aug, hp_threshold=hp_threshold,
    )
    train_dataset.points = np.concatenate([train_dataset.points, train_from_val.points])
    train_dataset.normals = np.concatenate([train_dataset.normals, train_from_val.normals])
    train_dataset.labels = np.concatenate([train_dataset.labels, train_from_val.labels])
    train_dataset.human_priors = np.concatenate(
        [train_dataset.human_priors, train_from_val.human_priors],
    )
    train_dataset.robot_labels = np.concatenate(
        [train_dataset.robot_labels, train_from_val.robot_labels],
    )
    train_dataset.sample_obj_ids = (
        train_dataset.sample_obj_ids + train_from_val.sample_obj_ids
    )
    train_dataset.num_samples = len(train_dataset.points)

    return train_dataset, val_dataset
