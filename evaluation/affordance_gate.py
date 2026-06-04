"""Affordance contact gates for eval random-raycast ablations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

AFFORDANCE_THRESH_NORM = 0.3
AFFORDANCE_KNN_MAX_M = 0.015

ContactGate = Callable[[np.ndarray, np.ndarray], bool]


def contact_in_v6_region(
    contact: np.ndarray,
    pc: np.ndarray,
    labels_norm: np.ndarray,
    *,
    threshold: float = AFFORDANCE_THRESH_NORM,
    max_dist: float = AFFORDANCE_KNN_MAX_M,
) -> bool:
    """True if nearest point on the v6 cloud has normalized affordance >= threshold."""
    pts = np.asarray(pc, dtype=np.float64)
    labels = np.asarray(labels_norm, dtype=np.float64).reshape(-1)
    if pts.size == 0 or labels.size != len(pts):
        return False
    pt = np.asarray(contact, dtype=np.float64).reshape(1, 3)
    dist, idx = cKDTree(pts).query(pt, k=1)
    if float(dist[0]) > float(max_dist):
        return False
    return float(labels[int(idx[0])]) >= float(threshold)


def passes_both_contacts_v6(
    contact_l: np.ndarray,
    contact_r: np.ndarray,
    pc: np.ndarray,
    labels_norm: np.ndarray,
    *,
    threshold: float = AFFORDANCE_THRESH_NORM,
    max_dist: float = AFFORDANCE_KNN_MAX_M,
) -> bool:
    return contact_in_v6_region(
        contact_l, pc, labels_norm, threshold=threshold, max_dist=max_dist
    ) and contact_in_v6_region(
        contact_r, pc, labels_norm, threshold=threshold, max_dist=max_dist
    )


def make_both_contacts_v6_gate(
    pc: np.ndarray,
    labels_norm: np.ndarray,
    *,
    threshold: float = AFFORDANCE_THRESH_NORM,
    max_dist: float = AFFORDANCE_KNN_MAX_M,
) -> ContactGate:
    """Return a gate(contact_L, contact_R) for generate_one_batch_eval."""

    def _gate(contact_l: np.ndarray, contact_r: np.ndarray) -> bool:
        return passes_both_contacts_v6(
            contact_l,
            contact_r,
            pc,
            labels_norm,
            threshold=threshold,
            max_dist=max_dist,
        )

    return _gate
