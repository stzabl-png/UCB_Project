#!/usr/bin/env python3
"""
Evaluation metrics and visualization utilities for affordance training.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_metrics(pred, target, threshold=0.5):
    """Compute classification metrics (accuracy, precision, recall, F1, IoU)."""
    prob = F.softmax(pred, dim=-1)[:, :, 1] if pred.dim() == 3 else F.softmax(pred, dim=-1)[:, 1]
    pred_cls = (prob > threshold).long()
    target_flat = target.reshape(-1)
    pred_flat = pred_cls.reshape(-1)

    correct = (pred_flat == target_flat).float().mean().item()
    tp = ((pred_flat == 1) & (target_flat == 1)).float().sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).float().sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "accuracy": correct, "precision": precision,
        "recall": recall, "f1": f1, "iou": iou
    }


@torch.no_grad()
def threshold_search(model, loader, device):
    """Post-training threshold sweep to find optimal F1."""
    model.eval()
    all_probs = []
    all_labels = []

    for xyz, features, labels, *_ in loader:
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)
        output = model(xyz, features)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        probs = F.softmax(pred, dim=-1)[:, :, 1]
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).reshape(-1)
    all_labels = torch.cat(all_labels, dim=0).reshape(-1)

    best_f1 = 0
    best_thresh = 0.5
    results = []

    for thresh in np.arange(0.3, 0.85, 0.05):
        pred_cls = (all_probs > thresh).long()
        tp = ((pred_cls == 1) & (all_labels == 1)).float().sum().item()
        fp = ((pred_cls == 1) & (all_labels == 0)).float().sum().item()
        fn = ((pred_cls == 0) & (all_labels == 1)).float().sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        results.append((thresh, prec, rec, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print(f"{'-'*48}")
    for thresh, prec, rec, f1 in results:
        marker = " ★" if abs(thresh - best_thresh) < 0.01 else ""
        print(f"{thresh:>10.2f} | {prec:>9.1%} | {rec:>9.1%} | {f1:>9.1%}{marker}")

    return best_thresh, best_f1


@torch.no_grad()
def save_visualization(model, val_dataset, device, save_path, epoch, history=None):
    """Save visualization PNG: sample predictions + training curves."""
    model.eval()
    n_samples = 4
    indices = np.random.choice(len(val_dataset), n_samples, replace=False)

    fig = plt.figure(figsize=(18, 5 * n_samples + (4 if history else 0)))
    n_rows = n_samples + (1 if history else 0)

    for row, idx in enumerate(indices):
        sample = val_dataset[idx]
        pts_t, feat_t, lbl_t = sample[0], sample[1], sample[2]
        pts = pts_t.numpy()
        lbl = lbl_t.numpy()

        output = model(pts_t.unsqueeze(0).to(device), feat_t.unsqueeze(0).to(device))
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        prob = F.softmax(pred, dim=-1)[0, :, 1].cpu().numpy()
        pred_mask = prob > 0.5
        gt_mask = lbl > 0

        tp = (gt_mask & pred_mask).sum()
        fp = (~gt_mask & pred_mask).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(gt_mask.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        for col, (title, colors) in enumerate([
            (f'GT (contact={gt_mask.sum()})', _gt_colors(gt_mask)),
            (f'Pred (TP={tp} FP={fp} F1={f1:.0%})', _pred_colors(gt_mask, pred_mask)),
            (f'Heatmap (max={prob.max():.2f})', _heat_colors(prob)),
        ]):
            ax = fig.add_subplot(n_rows, 3, row * 3 + col + 1, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=3, alpha=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    if history and len(history) > 1:
        epochs = [h['epoch'] for h in history]
        ax_loss = fig.add_subplot(n_rows, 3, n_samples * 3 + 1)
        ax_loss.plot(epochs, [h['train_loss'] for h in history], label='Train', color='blue')
        ax_loss.plot(epochs, [h['val_loss'] for h in history], label='Val', color='red')
        ax_loss.set_title('Loss'); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

        ax_f1 = fig.add_subplot(n_rows, 3, n_samples * 3 + 2)
        ax_f1.plot(epochs, [h['val_f1'] for h in history], color='green', linewidth=2)
        ax_f1.set_title(f'Val F1 (best={max(h["val_f1"] for h in history):.1%})')
        ax_f1.grid(True, alpha=0.3)

        ax_pr = fig.add_subplot(n_rows, 3, n_samples * 3 + 3)
        ax_pr.plot(epochs, [h.get('val_precision', 0) for h in history], label='Precision', color='orange')
        ax_pr.plot(epochs, [h.get('val_recall', 0) for h in history], label='Recall', color='purple')
        ax_pr.set_title('Precision / Recall'); ax_pr.legend(); ax_pr.grid(True, alpha=0.3)

    fig.suptitle(f'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"        📊 Visualization saved: {os.path.basename(save_path)}")


def _gt_colors(gt_mask):
    colors = np.full((len(gt_mask), 3), 0.75)
    colors[gt_mask] = [1, 0.2, 0.2]
    return colors


def _pred_colors(gt_mask, pred_mask):
    colors = np.full((len(gt_mask), 3), 0.75)
    colors[gt_mask & pred_mask] = [0.2, 0.9, 0.2]
    colors[~gt_mask & pred_mask] = [1.0, 0.5, 0.0]
    colors[gt_mask & ~pred_mask] = [0.5, 0.0, 0.5]
    return colors


def _heat_colors(prob):
    cmap = plt.cm.jet
    return cmap(prob)[:, :3]
