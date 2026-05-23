"""Affordance training package (soft heatmap PointNet++ pipeline)."""

from model.affordance.dataset import SoftAffordanceDataset
from model.affordance.losses import AffordanceLossWeights, AffordanceTrainingLoss
from model.affordance.pointnet2_ops import PointNet2Seg, affordance_probability

__all__ = [
    "SoftAffordanceDataset",
    "AffordanceLossWeights",
    "AffordanceTrainingLoss",
    "PointNet2Seg",
    "affordance_probability",
]
