#!/usr/bin/env python3
"""Gaussian soft affordance heatmap from binary contact labels."""

from __future__ import annotations

import numpy as np


def object_bbox_diagonal(points: np.ndarray) -> float:
    """Object scale = bounding-box diagonal of the point cloud (meters)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return 1.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(extent) + 1e-8)


def gaussian_soft_heatmap(
    points: np.ndarray,
    binary_contact: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    soft_aff[i] = exp(-d_i^2 / (2σ²)), d_i = dist to nearest contact point.
    Contact points are set to 1.0. No contact → all zeros.
    """
    pts = np.asarray(points, dtype=np.float32)
    contact = np.asarray(binary_contact, dtype=bool).reshape(-1)
    n = pts.shape[0]
    heatmap = np.zeros(n, dtype=np.float32)
    if not contact.any():
        return heatmap

    pos_pts = pts[contact]
    # (N, 3) vs (P, 3) → min squared dist per point
    diff = pts[:, None, :] - pos_pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2).min(axis=1)
    heatmap = np.exp(-d2 / (2.0 * float(sigma) ** 2 + 1e-12)).astype(np.float32)
    heatmap[contact] = 1.0
    return heatmap


def sigma_from_scale(object_scale: float, sigma_ratio: float) -> float:
    return float(sigma_ratio * object_scale)
