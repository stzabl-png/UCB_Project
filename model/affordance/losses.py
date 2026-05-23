#!/usr/bin/env python3
"""
Losses for soft affordance heatmap training (train_affordance pipeline).

L_total = L_aff + λ_bin·L_binary + λ_ch·L_center_heatmap (head/consistency off by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.affordance.pointnet2_ops import center_from_affordance, contact_valid_mask, fc_valid_mask


def object_scale_batch(xyz: torch.Tensor) -> torch.Tensor:
    """Per-sample bbox diagonal (meters), shape (B,)."""
    mn = xyz.min(dim=1).values
    mx = xyz.max(dim=1).values
    return (mx - mn).norm(dim=-1).clamp_min(1e-6)


class FocalLoss(nn.Module):
    """Hard binary focal (auxiliary)."""

    def __init__(self, alpha=0.5, gamma=2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, pred_logits, target):
        weight = torch.tensor([1.0, self.pos_weight], device=pred_logits.device)
        ce = F.cross_entropy(pred_logits, target, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred_logits, target):
        prob = F.softmax(pred_logits, dim=-1)[:, 1]
        target_f = target.float()
        tp = (prob * target_f).sum()
        fp = (prob * (1 - target_f)).sum()
        fn = ((1 - prob) * target_f).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky


class CombinedBinaryLoss(nn.Module):
    """Auxiliary L_binary = 0.6 Focal + 0.4 Tversky."""

    def __init__(
        self,
        *,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=0.5, gamma=2.0, pos_weight=1.0)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, pred_logits, target):
        return 0.6 * self.focal(pred_logits, target) + 0.4 * self.tversky(pred_logits, target)


def soft_focal_bce_elementwise(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Soft-label focal BCE per element.
    pt = y·p + (1-y)·(1-p)  (continuous target, not target > 0.5)
    """
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    y = target.float()
    bce = -(y * torch.log(pred) + (1.0 - y) * torch.log(1.0 - pred))
    pt = y * pred + (1.0 - y) * (1.0 - pred)
    alpha_t = (1.0 - alpha) * y + alpha * (1.0 - y)
    return alpha_t * (1.0 - pt) ** gamma * bce


def balanced_soft_focal_bce(
    pred: torch.Tensor,
    soft_gt: torch.Tensor,
    *,
    alpha: float = 0.5,
    gamma: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    """
    Foreground-balanced soft focal:
      L_pos = sum(loss * soft_gt) / sum(soft_gt)
      L_neg = sum(loss * (1-soft_gt)) / sum(1-soft_gt)
      return L_pos + background_weight * L_neg
    """
    loss_elem = soft_focal_bce_elementwise(pred, soft_gt, alpha=alpha, gamma=gamma)
    w_pos = soft_gt.float()
    w_neg = 1.0 - w_pos
    l_pos = (loss_elem * w_pos).sum() / w_pos.sum().clamp_min(1e-6)
    l_neg = (loss_elem * w_neg).sum() / w_neg.sum().clamp_min(1e-6)
    return l_pos + background_weight * l_neg


def soft_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum()
    den = pred.sum() + target.sum()
    dice = (2.0 * inter + smooth) / (den + smooth)
    return 1.0 - dice


def affordance_main_loss(
    pred: torch.Tensor,
    soft_gt: torch.Tensor,
    *,
    focal_weight: float = 0.7,
    dice_weight: float = 0.3,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    bf = balanced_soft_focal_bce(
        pred,
        soft_gt,
        alpha=focal_alpha,
        gamma=focal_gamma,
        background_weight=background_weight,
    )
    return focal_weight * bf + dice_weight * soft_dice(pred, soft_gt)


def center_heatmap_loss_l1_scaled(
    center_pred: torch.Tensor,
    center_gt: torch.Tensor,
    object_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-axis L1 in meters, divided by object scale (dimensionless).
    Equivalent to caring about mm error relative to object size.
    """
    err = (center_pred - center_gt).abs().mean(dim=-1)
    return (err / object_scale.clamp_min(1e-6)).mean()


@dataclass
class AffordanceLossWeights:
    lambda_aff: float = 0.3
    lambda_binary: float = 1.0
    lambda_peak: float = 0.0
    lambda_center_heatmap: float = 0.0
    lambda_center_head: float = 0.0
    lambda_consistency: float = 0.0
    lambda_smooth: float = 0.0


@dataclass
class LossHyperParams:
    binary_tversky_alpha: float = 0.5
    binary_tversky_beta: float = 0.5
    soft_background_weight: float = 0.25


def make_combined_binary_loss(hp: LossHyperParams | None = None) -> CombinedBinaryLoss:
    hp = hp or LossHyperParams()
    return CombinedBinaryLoss(
        tversky_alpha=hp.binary_tversky_alpha,
        tversky_beta=hp.binary_tversky_beta,
    )


class AffordanceTrainingLoss(nn.Module):
    """
    L_total = Σ λ_i · L_i for aff, binary, peak, center_heatmap, center_head,
    consistency, smooth. Weights come only from AffordanceLossWeights (CLI).
    """

    def __init__(
        self,
        weights: AffordanceLossWeights | None = None,
        *,
        loss_hyper: LossHyperParams | None = None,
    ):
        super().__init__()
        self.w = weights or AffordanceLossWeights()
        self.loss_hyper = loss_hyper or LossHyperParams()
        self.binary_loss = make_combined_binary_loss(self.loss_hyper)
        self.binary_neg_ratio = 1.0

    def forward(
        self,
        prob: torch.Tensor,
        seg_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        soft_gt: torch.Tensor,
        xyz: torch.Tensor,
        fc_head: torch.Tensor,
        fc_gt: torch.Tensor,
        *,
        binary_loss_fn=None,
    ) -> dict[str, torch.Tensor]:
        pred_flat = prob.reshape(-1)
        soft_flat = soft_gt.reshape(-1).float()

        l_aff = affordance_main_loss(
            pred_flat,
            soft_flat,
            background_weight=self.loss_hyper.soft_background_weight,
        )

        logits_flat = seg_logits.reshape(-1, 2)
        labels_flat = binary_labels.reshape(-1)
        if binary_loss_fn is not None:
            l_binary = binary_loss_fn(seg_logits, binary_labels)
        else:
            l_binary = self.binary_loss(logits_flat, labels_flat)

        center_hm = center_from_affordance(prob, xyz)
        scale = object_scale_batch(xyz)
        c_valid = contact_valid_mask(binary_labels) & fc_valid_mask(fc_gt)
        zero = prob.sum() * 0.0

        if c_valid.any():
            l_center_hm = center_heatmap_loss_l1_scaled(
                center_hm[c_valid], fc_gt[c_valid], scale[c_valid],
            )
        else:
            l_center_hm = zero

        if c_valid.any():
            l_center_head = F.l1_loss(fc_head[c_valid], fc_gt[c_valid])
            center_hm_det = center_hm.detach()
            l_consistency = F.l1_loss(fc_head[c_valid], center_hm_det[c_valid])
        else:
            l_center_head = zero
            l_consistency = zero

        l_smooth = zero
        l_peak = peak_contact_loss(prob, binary_labels)

        total = (
            self.w.lambda_aff * l_aff
            + self.w.lambda_binary * l_binary
            + self.w.lambda_peak * l_peak
            + self.w.lambda_center_heatmap * l_center_hm
            + self.w.lambda_center_head * l_center_head
            + self.w.lambda_consistency * l_consistency
            + self.w.lambda_smooth * l_smooth
        )

        return {
            "total": total,
            "aff": l_aff,
            "binary": l_binary,
            "peak": l_peak,
            "center_heatmap": l_center_hm,
            "center_head": l_center_head,
            "consistency": l_consistency,
            "smooth": l_smooth,
        }


AFFORDANCE_SEG_LOSS = CombinedBinaryLoss()  # legacy default


def peak_contact_loss(prob: torch.Tensor, binary_labels: torch.Tensor) -> torch.Tensor:
    """BCE on positive contact points only (target=1)."""
    pos = binary_labels.reshape(-1) > 0
    if not pos.any():
        return prob.sum() * 0.0
    p = prob.reshape(-1)[pos].clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(p, torch.ones_like(p))


def loss_weights_from_args(args) -> AffordanceLossWeights:
    return AffordanceLossWeights(
        lambda_aff=float(args.lambda_aff),
        lambda_binary=float(args.lambda_binary),
        lambda_peak=float(args.lambda_peak),
        lambda_center_heatmap=float(args.lambda_center_heatmap),
        lambda_center_head=float(args.lambda_center_head),
        lambda_consistency=float(args.lambda_consistency),
        lambda_smooth=float(args.lambda_smooth),
    )


def build_affordance_criterion(args) -> AffordanceTrainingLoss:
    criterion = AffordanceTrainingLoss(
        loss_weights_from_args(args),
        loss_hyper=loss_hyperparams_from_args(args),
    )
    criterion.binary_neg_ratio = float(args.binary_neg_ratio)
    return criterion


def loss_hyperparams_from_args(args) -> LossHyperParams:
    return LossHyperParams(
        binary_tversky_alpha=float(getattr(args, "binary_tversky_alpha", 0.5)),
        binary_tversky_beta=float(getattr(args, "binary_tversky_beta", 0.5)),
        soft_background_weight=float(getattr(args, "soft_background_weight", 0.25)),
    )
