#!/usr/bin/env python3
"""Train/eval loops — soft affordance heatmap + auxiliary binary loss."""

from __future__ import annotations

import json
import os
from typing import Callable

import torch
from torch.utils.data import DataLoader

from model.affordance.debug import (
    compute_per_object_debug_metrics,
    format_debug_log_line,
    select_debug_vis_object_ids,
    summarize_object_metrics,
)
from model.affordance.losses import AffordanceTrainingLoss
from model.affordance.metrics import (
    compute_affordance_metrics,
    compute_debug_overfit_metrics,
    save_val_objects_grid,
    seg_head_grad_norm,
)
from model.affordance.pointnet2_ops import affordance_probability, forward_seg_fc


def _balanced_binary_loss(seg_pred, labels, criterion, *, neg_ratio: float = 1.0):
    pred_flat = seg_pred.reshape(-1, 2)
    labels_flat = labels.reshape(-1)
    pos_idx = (labels_flat == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels_flat == 0).nonzero(as_tuple=True)[0]
    n_pos, n_neg = pos_idx.numel(), neg_idx.numel()
    if n_pos == 0 or n_neg == 0:
        return criterion.binary_loss(pred_flat, labels_flat)

    perm = torch.randperm
    dev = seg_pred.device
    n_pos_pick = n_pos
    n_neg_pick = min(n_neg, max(1, int(neg_ratio * n_pos)))
    pos_pick = pos_idx[perm(n_pos, device=dev)[:n_pos_pick]]
    neg_pick = neg_idx[perm(n_neg, device=dev)[:n_neg_pick]]
    idx = torch.cat([pos_pick, neg_pick])
    return criterion.binary_loss(pred_flat[idx], labels_flat[idx])


def _accum_metrics(acc: dict, metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0.0) + float(v)
        elif isinstance(v, bool):
            acc[k] = acc.get(k, 0.0) + float(v)


def _avg_metrics(acc: dict, n_batches: int) -> dict:
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in acc.items()}


def _forward_losses(
    model,
    batch,
    criterion,
    device,
    *,
    balanced_binary: bool,
    debug_metrics: bool = False,
    raw_model=None,
):
    xyz, features, labels, soft_gt, fc_gt = [x.to(device) for x in batch[:5]]
    seg_pred, fc_pred = forward_seg_fc(model, xyz, features)
    prob = affordance_probability(seg_pred)

    use_balanced_binary = (
        balanced_binary
        and getattr(criterion, "loss_mode", "full") != "simple"
    )
    neg_ratio = float(getattr(criterion, "binary_neg_ratio", 1.0))

    if use_balanced_binary:

        def binary_fn(sp, lb):
            return _balanced_binary_loss(sp, lb, criterion, neg_ratio=neg_ratio)

        parts = criterion(
            prob, seg_pred, labels, soft_gt, xyz, fc_pred, fc_gt,
            binary_loss_fn=binary_fn,
        )
    else:
        parts = criterion(prob, seg_pred, labels, soft_gt, xyz, fc_pred, fc_gt)

    grad_norm = 0.0
    if debug_metrics and raw_model is not None:
        grad_norm = seg_head_grad_norm(raw_model)

    if debug_metrics:
        metrics = compute_debug_overfit_metrics(
            prob.detach(),
            labels,
            soft_gt,
            xyz,
            fc_pred.detach(),
            fc_gt,
            seg_logits=seg_pred.detach(),
            seg_head_grad_norm=grad_norm,
        )
    else:
        metrics = compute_affordance_metrics(
            prob.detach(), labels, soft_gt, xyz, fc_pred.detach(), fc_gt,
        )
    return parts, metrics


def train_epoch_mt(model, loader, optimizer, criterion, device, **_kwargs):
    model.train()
    totals = {
        k: 0.0
        for k in ("total", "aff", "binary", "peak", "center_heatmap", "center_head", "consistency")
    }
    all_metrics: dict = {}
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        parts, metrics = _forward_losses(
            model, batch, criterion, device, balanced_binary=True,
        )
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            if k in parts:
                totals[k] += parts[k].item()
        _accum_metrics(all_metrics, metrics)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, {}
    avg = lambda x: x / n_batches
    out_metrics = _avg_metrics(all_metrics, n_batches)
    out_metrics.update({f"train_{k}": avg(v) for k, v in totals.items() if k in totals})
    return (
        avg(totals["total"]),
        avg(totals["aff"]),
        avg(totals["binary"]),
        out_metrics,
    )


@torch.no_grad()
def eval_epoch_mt(model, loader, criterion, device, **_kwargs):
    model.eval()
    totals = {
        k: 0.0
        for k in ("total", "aff", "binary", "peak", "center_heatmap", "center_head", "consistency")
    }
    all_metrics: dict = {}
    n_batches = 0

    for batch in loader:
        parts, metrics = _forward_losses(
            model, batch, criterion, device, balanced_binary=False,
        )
        for k in totals:
            if k in parts:
                totals[k] += parts[k].item()
        _accum_metrics(all_metrics, metrics)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, {}
    avg = lambda x: x / n_batches
    m = _avg_metrics(all_metrics, n_batches)
    fc_hm_mm = m.get("center_heatmap_mm", 0.0)
    return (
        avg(totals["total"]),
        avg(totals["aff"]),
        avg(totals["binary"]),
        fc_hm_mm,
        m,
    )


def run_debug_overfit_loop(
    model,
    train_loader: DataLoader,
    optimizer,
    criterion,
    device,
    val_dataset,
    vis_dir: str,
    ckpt_dir: str,
    *,
    max_steps: int,
    log_interval: int,
    vis_interval: int,
    log_fn: Callable[[str], None],
    unwrap_model_fn: Callable,
    vis_max_objects: int = 10,
    object_metrics_interval: int = 0,
) -> list[dict]:
    """
    Step-based overfit loop on a tiny dataset; saves debug_step_*.png and history.
    """
    model.train()
    history: list[dict] = []
    data_iter = iter(train_loader)
    raw = unwrap_model_fn(model)

    log_fn(
        f"{'step':>5} | {'loss':>7} {'bin':>7} {'soft':>7} {'peak':>7} | "
        f"prob stats | μ+/μ- | F1 AP top1% | grad",
    )
    log_fn("-" * 110)

    for step in range(max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if step > 0:
            optimizer.zero_grad()
            parts, _ = _forward_losses(
                model,
                batch,
                criterion,
                device,
                balanced_binary=True,
                debug_metrics=False,
            )
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gnorm = seg_head_grad_norm(raw)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                parts_log, metrics = _forward_losses(
                    model,
                    batch,
                    criterion,
                    device,
                    balanced_binary=False,
                    debug_metrics=True,
                    raw_model=raw,
                )
            model.train()
            metrics["seg_head_grad_norm"] = gnorm

            if object_metrics_interval > 0 and step % object_metrics_interval == 0:
                per_obj = compute_per_object_debug_metrics(model, val_dataset, device)
                metrics.update(summarize_object_metrics(per_obj))

            parts_dict = {k: float(parts_log[k].item()) for k in parts_log}
            if step % log_interval == 0:
                log_fn(format_debug_log_line(step, parts_dict, metrics))

            row = {"step": step}
            row.update({f"loss_{k}": round(v, 5) for k, v in parts_dict.items()})
            row.update({
                k: round(v, 5) for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            history.append(row)
        elif step % log_interval == 0:
            model.eval()
            with torch.no_grad():
                parts_log, metrics = _forward_losses(
                    model, batch, criterion, device, balanced_binary=False, debug_metrics=True,
                )
            model.train()
            parts_dict = {k: float(parts_log[k].item()) for k in parts_log}
            log_fn(format_debug_log_line(step, parts_dict, metrics))

        if step % vis_interval == 0:
            per_obj_vis = None
            if step > 0 and object_metrics_interval > 0:
                per_obj_vis = compute_per_object_debug_metrics(model, val_dataset, device)
            vis_oids = select_debug_vis_object_ids(
                val_dataset.sample_obj_ids,
                per_obj_vis,
                max_objects=vis_max_objects,
            )
            vis_path = os.path.join(vis_dir, f"debug_step_{step:04d}.png")
            save_val_objects_grid(
                model,
                val_dataset,
                val_dataset.sample_obj_ids,
                device,
                vis_path,
                step,
                vis_object_ids=vis_oids,
            )

    hist_path = os.path.join(ckpt_dir, "debug_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log_fn(f"  Debug history: {hist_path}")
    return history
