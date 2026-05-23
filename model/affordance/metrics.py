#!/usr/bin/env python3
"""
Metrics and validation visualization for affordance training (train_affordance.py).

Extends model/metrics.py with collapse detection and per-object val grids.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.metrics import _gt_colors, _heat_colors, _pred_colors
from model.affordance.logging_utils import training_log
from model.affordance.pointnet2_ops import center_from_affordance, fc_valid_mask, forward_seg_fc


def _numpy_binary_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    n_pos = float(y_true.sum())
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[prec[0]], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def _precision_at_top_frac(prob: np.ndarray, binary: np.ndarray, frac: float) -> float:
    k = max(1, int(len(prob) * frac))
    top = np.argpartition(prob, -k)[-k:]
    return float(binary[top].mean())


def _recall_at_top_frac(prob: np.ndarray, binary: np.ndarray, frac: float) -> float:
    k = max(1, int(len(prob) * frac))
    top = np.argpartition(prob, -k)[-k:]
    return float(binary[top].sum() / max(int(binary.sum()), 1))


@torch.no_grad()
def compute_affordance_metrics(
    prob: torch.Tensor,
    binary_labels: torch.Tensor,
    soft_gt: torch.Tensor,
    xyz: torch.Tensor,
    fc_head: torch.Tensor,
    fc_gt: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Per-batch metrics averaged over samples (F1, AP, top-K, center errors)."""
    b = prob.shape[0]
    keys = (
        "f1", "precision", "recall", "iou", "ap",
        "precision_top1pct", "precision_top5pct", "recall_top5pct",
        "center_heatmap_mm", "center_head_mm",
        "pred_pos_frac", "prob_mean", "prob_min", "prob_max", "gt_pos_frac", "collapsed",
    )
    acc = {k: 0.0 for k in keys}
    center_hm = center_from_affordance(prob, xyz)
    fc_ok = fc_valid_mask(fc_gt)

    for i in range(b):
        p = prob[i].cpu().numpy()
        y = binary_labels[i].reshape(-1).cpu().numpy().astype(bool)
        pred_cls = p > threshold
        tp = (pred_cls & y).sum()
        fp = (pred_cls & ~y).sum()
        fn = ((~pred_cls) & y).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        acc["f1"] += f1
        acc["precision"] += prec
        acc["recall"] += rec
        acc["iou"] += tp / max(tp + fp + fn, 1)
        acc["ap"] += _numpy_binary_ap(y, p)
        acc["precision_top1pct"] += _precision_at_top_frac(p, y, 0.01)
        acc["precision_top5pct"] += _precision_at_top_frac(p, y, 0.05)
        acc["recall_top5pct"] += _recall_at_top_frac(p, y, 0.05)
        acc["pred_pos_frac"] += pred_cls.mean()
        acc["prob_mean"] += p.mean()
        acc["prob_min"] += p.min()
        acc["prob_max"] += p.max()
        acc["gt_pos_frac"] += y.mean()

        if fc_ok[i] and y.any():
            hm = center_hm[i]
            acc["center_heatmap_mm"] += (hm - fc_gt[i]).norm().item() * 1000
            acc["center_head_mm"] += (fc_head[i] - fc_gt[i]).norm().item() * 1000

        acc["collapsed"] += float(
            _is_collapsed(
                pred_cls.mean(), rec, prec, y.mean(),
                prob_min=p.min(), prob_max=p.max(),
            )
        )

    n = max(b, 1)
    out = {k: acc[k] / b for k in acc if k != "collapsed"}
    out["collapsed"] = acc["collapsed"] >= (b * 0.5)
    return out


def compute_seg_logit_stats(
    seg_logits: torch.Tensor,
    binary_labels: torch.Tensor,
) -> dict[str, float]:
    """Per-point class-1 vs class-0 logits on GT contact / non-contact."""
    logits = seg_logits.reshape(-1, 2)
    y = binary_labels.reshape(-1) > 0
    contact_logit = logits[:, 1]
    noncontact_logit = logits[:, 0]
    gap = contact_logit - noncontact_logit

    def _stats(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
        if not mask.any():
            return 0.0, 0.0, 0.0
        v = values[mask]
        return float(v.min()), float(v.mean()), float(v.max())

    cmn, cmu, cmx = _stats(contact_logit, y)
    nmn, nmu, nmx = _stats(noncontact_logit, ~y)
    gmn, gmu, gmx = float(gap.min()), float(gap.mean()), float(gap.max())
    return {
        "contact_logit_min": cmn,
        "contact_logit_mean": cmu,
        "contact_logit_max": cmx,
        "noncontact_logit_min": nmn,
        "noncontact_logit_mean": nmu,
        "noncontact_logit_max": nmx,
        "logit_gap_min": gmn,
        "logit_gap_mean": gmu,
        "logit_gap_max": gmx,
    }


@torch.no_grad()
def compute_debug_overfit_metrics(
    prob: torch.Tensor,
    binary_labels: torch.Tensor,
    soft_gt: torch.Tensor,
    xyz: torch.Tensor,
    fc_head: torch.Tensor,
    fc_gt: torch.Tensor,
    *,
    seg_logits: torch.Tensor | None = None,
    seg_head_grad_norm: float = 0.0,
) -> dict:
    """Extended metrics for debug overfit logging."""
    base = compute_affordance_metrics(
        prob, binary_labels, soft_gt, xyz, fc_head, fc_gt,
    )
    p = prob.reshape(-1)
    y = binary_labels.reshape(-1) > 0
    if y.any():
        base["pred_mean_on_GT_contact"] = p[y].mean().item()
    else:
        base["pred_mean_on_GT_contact"] = 0.0
    if (~y).any():
        base["pred_mean_on_GT_noncontact"] = p[~y].mean().item()
    else:
        base["pred_mean_on_GT_noncontact"] = 0.0
    base["prob_span"] = base["prob_max"] - base["prob_min"]
    base["seg_head_grad_norm"] = float(seg_head_grad_norm)
    if seg_logits is not None:
        base.update(compute_seg_logit_stats(seg_logits, binary_labels))
    return base


def seg_head_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of gradients on segmentation head (conv2) parameters."""
    total_sq = 0.0
    n = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "conv2" not in name:
            continue
        g = param.grad.detach()
        total_sq += float(g.pow(2).sum())
        n += 1
    if n == 0:
        return 0.0
    return total_sq ** 0.5


def checkpoint_score(metrics: dict, *, f1_weight: float = 0.5) -> float:
    """best_score = w·F1 + (1-w)·AP for model selection."""
    return f1_weight * metrics.get("f1", 0.0) + (1.0 - f1_weight) * metrics.get("ap", 0.0)


def compute_metrics(pred, target, threshold=0.5):
    """Classification metrics + predict-all-positive collapse flags."""
    prob = (
        F.softmax(pred, dim=-1)[:, :, 1]
        if pred.dim() == 3
        else F.softmax(pred, dim=-1)[:, 1]
    )
    pred_cls = (prob > threshold).long()
    target_flat = target.reshape(-1)
    pred_flat = pred_cls.reshape(-1)
    prob_flat = prob.reshape(-1)

    correct = (pred_flat == target_flat).float().mean().item()
    tp = ((pred_flat == 1) & (target_flat == 1)).float().sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).float().sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    gt_pos_frac = target_flat.float().mean().item()
    pred_pos_frac = pred_flat.float().mean().item()

    return {
        "accuracy": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pred_pos_frac": pred_pos_frac,
        "prob_mean": prob_flat.mean().item(),
        "prob_min": prob_flat.min().item(),
        "prob_max": prob_flat.max().item(),
        "gt_pos_frac": gt_pos_frac,
        "collapsed": _is_collapsed(
            pred_pos_frac, recall, precision, gt_pos_frac,
            prob_min=prob_flat.min().item(),
            prob_max=prob_flat.max().item(),
        ),
    }


def _is_collapsed(
    pred_pos_frac: float,
    recall: float,
    precision: float,
    gt_pos_frac: float,
    *,
    prob_min: float,
    prob_max: float,
    pred_frac_thresh: float = 0.95,
    prob_span_thresh: float = 0.08,
) -> bool:
    """Degenerate segmentation: all-positive, all-negative, or near-constant prob."""
    if pred_pos_frac >= pred_frac_thresh:
        return True
    if recall >= 0.99 and precision <= max(gt_pos_frac * 1.5, 0.15):
        return True
    if pred_pos_frac <= 0.01 and recall <= 0.01:
        return True
    if (prob_max - prob_min) < prob_span_thresh:
        return True
    return False


def _add_val_grid_legend(fig, prob_threshold: float) -> None:
    from matplotlib.lines import Line2D

    def _m(rgb, label):
        return Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=rgb,
            markeredgecolor="#444",
            markeredgewidth=0.3,
            markersize=7,
            linestyle="None",
            label=label,
        )

    gt_handles = [
        _m([0.75, 0.75, 0.75], "non-contact"),
        _m([1.0, 0.2, 0.2], "GT contact"),
    ]
    pred_handles = [
        _m([0.75, 0.75, 0.75], "non-contact"),
        _m([0.2, 0.9, 0.2], "TP"),
        _m([1.0, 0.5, 0.0], "FP"),
        _m([0.5, 0.0, 0.5], "FN"),
    ]

    leg_gt = fig.legend(
        handles=gt_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01),
        ncol=2,
        fontsize=8,
        framealpha=0.85,
        facecolor="#2a2a3e",
        edgecolor="#555",
        labelcolor="#ddd",
        title="GT row",
        title_fontsize=9,
    )
    leg_gt.get_title().set_color("#ccc")

    leg_pr = fig.legend(
        handles=pred_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        ncol=4,
        fontsize=8,
        framealpha=0.85,
        facecolor="#2a2a3e",
        edgecolor="#555",
        labelcolor="#ddd",
        title=f"Pred row (prob ≥ {prob_threshold})",
        title_fontsize=9,
    )
    leg_pr.get_title().set_color("#ccc")
    fig.add_artist(leg_gt)
    fig.add_artist(leg_pr)


def _set_equal_3d(ax, pts: np.ndarray, pad: float = 1.12):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


@torch.no_grad()
def save_val_objects_grid(
    model,
    dataset,
    sample_obj_ids,
    device,
    save_path: str,
    epoch: int,
    *,
    prob_threshold: float = 0.5,
    point_size: float = 2.0,
    vis_object_ids: list[str] | None = None,
):
    """
    验证集每个物体一列：Human prior / GT / Pred（全物体一张图）。
    vis_object_ids: if set, only plot these objects (one column each, first sample).
    """
    model.eval()
    id_to_idx: dict[str, int] = {}
    for i, oid in enumerate(sample_obj_ids):
        if oid not in id_to_idx:
            id_to_idx[oid] = i
    obj_ids = sorted(set(sample_obj_ids))
    if vis_object_ids is not None:
        obj_ids = [o for o in vis_object_ids if o in id_to_idx]
    ncol = len(obj_ids)
    if ncol == 0:
        return

    n_rows = 5
    fig_w = max(2.8 * ncol, 10.0)
    fig = plt.figure(figsize=(fig_w, 15.5), facecolor="#1a1a2e")
    n_collapsed = 0
    row_labels = ("Human prior", "GT bin", "GT soft", "Pred bin", "Pred score")

    for col, oid in enumerate(obj_ids):
        idx = id_to_idx[oid]
        sample = dataset[idx]
        pts_t, feat_t, lbl_t = sample[0], sample[1], sample[2]
        soft_t = sample[3] if len(sample) > 3 else None
        hp_t = sample[5] if len(sample) > 5 else None
        pts = pts_t.numpy()
        lbl = lbl_t.numpy().astype(bool)
        soft_gt = soft_t.numpy() if soft_t is not None else lbl.astype(np.float32)
        if hp_t is not None:
            hp = hp_t.numpy().astype(np.float32)
        elif hasattr(dataset, "human_priors"):
            hp = dataset.human_priors[idx].astype(np.float32)
        else:
            hp = np.zeros(len(pts), dtype=np.float32)
        hp_missing = float(hp.max()) < 1e-6

        seg_pred, _ = forward_seg_fc(
            model,
            pts_t.unsqueeze(0).to(device),
            feat_t.unsqueeze(0).to(device),
        )
        prob = F.softmax(seg_pred, dim=-1)[0, :, 1].cpu().numpy()
        pred_mask = prob > prob_threshold
        gt_pos = int(lbl.sum())
        n_pos = int(pred_mask.sum())
        tp = int((lbl & pred_mask).sum())
        fp = int((~lbl & pred_mask).sum())
        fn = int((lbl & ~pred_mask).sum())
        collapsed = (
            n_pos >= int(0.95 * len(lbl))
            or (fn == 0 and fp > 0 and gt_pos < len(lbl))
        )
        if collapsed:
            n_collapsed += 1

        def _ax(row: int):
            return fig.add_subplot(
                n_rows, ncol, row * ncol + col + 1,
                projection="3d",
                facecolor="#1a1a2e",
            )

        ax_hp = _ax(0)
        ax_gt = _ax(1)
        ax_soft = _ax(2)
        ax_pr = _ax(3)
        ax_sc = _ax(4)

        hp_title = "HP missing" if hp_missing else f"max={hp.max():.2f}"
        warn = " ⚠" if collapsed else ""
        panels = (
            (ax_hp, _heat_colors(np.clip(hp, 0.0, 1.0)), f"Human prior\n{hp_title}"),
            (ax_gt, _gt_colors(lbl), f"{oid}\nGT bin={gt_pos}"),
            (ax_soft, _heat_colors(soft_gt), f"GT soft\nmax={soft_gt.max():.2f}"),
            (
                ax_pr,
                _pred_colors(lbl, pred_mask),
                f"Pred≥{prob_threshold}{warn}\n"
                f"+{n_pos}/{len(lbl)}  TP={tp} FP={fp} FN={fn}",
            ),
            (
                ax_sc,
                _heat_colors(prob),
                f"Pred score\n"
                f"min={prob.min():.2f} mean={prob.mean():.2f} max={prob.max():.2f}",
            ),
        )
        for ax, colors, title in panels:
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=colors, s=point_size, alpha=0.85, linewidths=0,
            )
            ax.set_title(
                title,
                fontsize=6,
                color="#f88" if collapsed and ax is ax_pr else "#ccc",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            _set_equal_3d(ax, pts)

    collapse_note = ""
    if n_collapsed > 0:
        collapse_note = (
            f"  —  ⚠ {n_collapsed}/{ncol} objects: predict-all-positive "
            f"(only gray/green/orange, no purple FN)"
        )
    fig.suptitle(
        f"Val — epoch {epoch}{collapse_note}",
        color="#f88" if n_collapsed == ncol else "#ddd",
        fontsize=10,
        y=0.99,
    )
    row_y_step = 0.90 / max(n_rows, 1)
    for row_i, label in enumerate(row_labels):
        fig.text(
            0.01,
            0.92 - row_i * row_y_step,
            label,
            color="#ccc",
            fontsize=9,
            fontweight="bold",
            va="center",
        )

    _add_val_grid_legend(fig, prob_threshold)
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.94])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    training_log(f"        📊 Val grid vis: {os.path.basename(save_path)} ({ncol} objects)")
