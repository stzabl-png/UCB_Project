#!/usr/bin/env python3
"""Train/eval loops for human-prior L1-only affordance training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from model.affordance.losses_hp import L1HumanPriorLoss
from model.affordance.metrics import compute_affordance_metrics


def _accum_metrics(acc: dict, metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0.0) + float(v)
        elif isinstance(v, bool):
            acc[k] = acc.get(k, 0.0) + float(v)


def _avg_metrics(acc: dict, n_batches: int) -> dict:
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in acc.items()}


def train_epoch_hp(model, loader, optimizer, criterion, device):
    model.train()
    total_l1 = 0.0
    all_metrics: dict = {}
    n_batches = 0
    for batch in loader:
        xyz, features, labels, soft_gt = [x.to(device) for x in batch[:4]]
        optimizer.zero_grad()
        prob = model(xyz, features)
        parts = criterion(prob, soft_gt)
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_l1 += parts["l1"].item()
        with torch.no_grad():
            b = prob.shape[0]
            zero_fc = torch.zeros(b, 3, device=device)
            metrics = compute_affordance_metrics(
                prob, labels, soft_gt, xyz, zero_fc, zero_fc,
            )
        _accum_metrics(all_metrics, metrics)
        n_batches += 1

    if n_batches == 0:
        return 0.0, {}
    out = _avg_metrics(all_metrics, n_batches)
    out["train_l1"] = total_l1 / n_batches
    return total_l1 / n_batches, out


@torch.no_grad()
def eval_epoch_hp(model, loader, criterion, device):
    model.eval()
    total_l1 = 0.0
    all_metrics: dict = {}
    n_batches = 0
    for batch in loader:
        xyz, features, labels, soft_gt = [x.to(device) for x in batch[:4]]
        prob = model(xyz, features)
        parts = criterion(prob, soft_gt)
        total_l1 += parts["l1"].item()
        b = prob.shape[0]
        zero_fc = torch.zeros(b, 3, device=device)
        metrics = compute_affordance_metrics(
            prob, labels, soft_gt, xyz, zero_fc, zero_fc,
        )
        _accum_metrics(all_metrics, metrics)
        n_batches += 1

    if n_batches == 0:
        return 0.0, {}
    out = _avg_metrics(all_metrics, n_batches)
    out["val_l1"] = total_l1 / n_batches
    return total_l1 / n_batches, out
