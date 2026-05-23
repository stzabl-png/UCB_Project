"""Affordance training package (soft heatmap PointNet++ pipeline)."""

from model.affordance.dataset import SoftAffordanceDataset
from model.affordance.losses import (
    AffordanceLossWeights,
    AffordanceTrainingLoss,
    SimpleSoftScoreLoss,
)
from model.affordance.pointnet2_ops import PointNet2Seg, affordance_probability

__all__ = [
    "SoftAffordanceDataset",
    "AffordanceLossWeights",
    "AffordanceTrainingLoss",
    "SimpleSoftScoreLoss",
    "PointNet2Seg",
    "affordance_probability",
]
