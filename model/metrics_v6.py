#!/usr/bin/env python3
"""
Evaluation metrics for PointNet++ v6 — Continuous Affordance Heatmap.

Primary: Pearson Correlation (per-object average).
Secondary: MAE, Peak IoU (with automatic threshold search).
"""

import numpy as np
import torch


@torch.no_grad()
def pearson_correlation(pred, target):
    """Per-sample Pearson correlation, averaged over batch.

    Args:
        pred: (B, N) predicted heatmap
        target: (B, N) ground truth heatmap
    Returns:
        float: mean Pearson r across batch (higher is better, max=1.0)
    """
    B = pred.shape[0]
    total_r = 0.0
    valid = 0

    for i in range(B):
        p = pred[i]
        t = target[i]
        p_mean = p.mean()
        t_mean = t.mean()
        p_centered = p - p_mean
        t_centered = t - t_mean

        num = (p_centered * t_centered).sum()
        denom = torch.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum() + 1e-8)
        r = (num / denom).item()

        if not np.isnan(r):
            total_r += r
            valid += 1

    return total_r / max(valid, 1)


@torch.no_grad()
def compute_mae(pred, target):
    """Mean Absolute Error.

    Args:
        pred: (B, N) predicted heatmap
        target: (B, N) ground truth heatmap
    Returns:
        float: MAE (lower is better)
    """
    return torch.abs(pred - target).mean().item()


@torch.no_grad()
def peak_iou(pred, target, threshold=0.3):
    """IoU of high-activation regions.

    Args:
        pred: (B, N) predicted heatmap
        target: (B, N) ground truth heatmap
        threshold: activation threshold to define "peak" region
    Returns:
        float: IoU of peak regions (higher is better, max=1.0)
    """
    pred_peak = pred > threshold
    gt_peak = target > threshold

    intersection = (pred_peak & gt_peak).float().sum()
    union = (pred_peak | gt_peak).float().sum()

    return (intersection / (union + 1e-8)).item()


@torch.no_grad()
def compute_all_metrics(pred, target):
    """Compute all v6 metrics at once.

    Args:
        pred: (B, N) predicted heatmap in [0, 1]
        target: (B, N) ground truth soft labels in [0, 1]
    Returns:
        dict with keys: pearson, mae, peak_iou_03, peak_iou_05
    """
    return {
        "pearson": pearson_correlation(pred, target),
        "mae": compute_mae(pred, target),
        "peak_iou_03": peak_iou(pred, target, threshold=0.3),
        "peak_iou_05": peak_iou(pred, target, threshold=0.5),
    }


@torch.no_grad()
def threshold_search_v6(model, loader, device):
    """Post-training threshold sweep on predicted heatmaps.

    Finds the threshold τ that maximizes Peak IoU between
    pred > τ and gt > τ.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        xyz = batch[0].to(device)
        features = batch[1].to(device)
        target = batch[2].to(device)

        pred = model(xyz, features)
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)    # (N_total, 4096)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"\n{'Threshold':>10} | {'Peak IoU':>10} | {'Pred Pos%':>10} | {'GT Pos%':>10}")
    print(f"{'-' * 48}")

    best_iou = 0
    best_thresh = 0.3
    results = []

    for thresh in np.arange(0.05, 0.80, 0.05):
        pred_mask = all_preds > thresh
        gt_mask = all_targets > thresh

        inter = (pred_mask & gt_mask).float().sum().item()
        union = (pred_mask | gt_mask).float().sum().item()
        iou = inter / (union + 1e-8)

        pred_pos = pred_mask.float().mean().item() * 100
        gt_pos = gt_mask.float().mean().item() * 100

        results.append((thresh, iou, pred_pos, gt_pos))
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    for thresh, iou, pred_pos, gt_pos in results:
        marker = " ★" if abs(thresh - best_thresh) < 0.01 else ""
        print(f"{thresh:>10.2f} | {iou:>9.1%} | {pred_pos:>9.1f}% | {gt_pos:>9.1f}%{marker}")

    # Also compute Pearson at best threshold
    pearson_r = pearson_correlation(all_preds, all_targets)
    print(f"\n  Pearson Correlation: {pearson_r:.4f}")
    print(f"  Best threshold: {best_thresh:.2f} (IoU={best_iou:.1%})")

    return best_thresh, best_iou, pearson_r
