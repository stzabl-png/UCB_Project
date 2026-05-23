"""Few-object / curriculum overfit debug mode for affordance training."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from model.affordance.augment import augment_config_from_args
from model.affordance.dataset import SoftAffordanceDataset
from model.affordance.metrics import compute_affordance_metrics
from model.affordance.pointnet2_ops import affordance_probability, forward_seg_fc
from model.train import get_object_split

if TYPE_CHECKING:
    import argparse


def apply_debug_config(args: argparse.Namespace) -> None:
    """Force debug-friendly defaults when --debug-overfit-one-object is set."""
    if not args.debug_overfit_one_object:
        return

    if not getattr(args, "no_augment", False):
        args.no_augment = True
        args.augment_mode = "none"

    args.disable_early_stop = True

    if args.disable_center_loss:
        args.lambda_center_heatmap = 0.0
        args.lambda_center_head = 0.0
        args.lambda_consistency = 0.0

    args.weight_decay = 0.0
    args.warmup_epochs = 0
    args.patience = 10**9
    args.ckpt_reject_collapse = False

    if not hasattr(args, "debug_num_objects") or args.debug_num_objects is None:
        args.debug_num_objects = 1


def _subset_dataset(ds: SoftAffordanceDataset, indices: np.ndarray) -> SoftAffordanceDataset:
    indices = np.asarray(indices, dtype=np.int64)
    ds.points = ds.points[indices]
    ds.normals = ds.normals[indices]
    ds.labels = ds.labels[indices]
    ds.force_centers = ds.force_centers[indices]
    if hasattr(ds, "human_priors"):
        ds.human_priors = ds.human_priors[indices]
    ds.sample_obj_ids = [ds.sample_obj_ids[int(i)] for i in indices]
    ds.num_samples = len(indices)
    return ds


def _object_to_positive_indices(ds: SoftAffordanceDataset) -> dict[str, list[int]]:
    """object_id → sample indices with at least one contact point."""
    obj_indices: dict[str, list[int]] = defaultdict(list)
    for i in range(ds.num_samples):
        if (ds.labels[i] > 0.5).any():
            obj_indices[ds.sample_obj_ids[i]].append(i)
    return dict(obj_indices)


def _pick_object_ids(
    obj_ids: list[str],
    k: int,
    *,
    mode: str,
    seed: int,
) -> list[str]:
    if k >= len(obj_ids):
        return list(obj_ids)
    if mode == "random":
        rng = random.Random(seed)
        return rng.sample(obj_ids, k)
    return sorted(obj_ids)[:k]


def _limit_samples_per_object(
    indices: list[int],
    ds: SoftAffordanceDataset,
    m: int,
    *,
    seed: int,
) -> list[int]:
    if m <= 0 or len(indices) <= m:
        return indices
    rng = random.Random(seed)
    return rng.sample(indices, m)


def select_debug_indices(ds: SoftAffordanceDataset, args: argparse.Namespace) -> np.ndarray:
    """
    Object-level (default) or sample-level subset.

    Priority:
      1. --debug-object-id
      2. --debug-use-sample-mode / legacy --debug-num-samples only path
      3. --debug-num-objects K (+ optional --debug-samples-per-object M)
    """
    if args.debug_object_id:
        oid = args.debug_object_id
        idxs = [
            i for i, o in enumerate(ds.sample_obj_ids)
            if o == oid and (ds.labels[i] > 0.5).any()
        ]
        if not idxs:
            idxs = [i for i, o in enumerate(ds.sample_obj_ids) if o == oid]
        if not idxs:
            raise ValueError(f"--debug-object-id {oid!r} not found in dataset")
        m = int(getattr(args, "debug_samples_per_object", 0) or 0)
        if m > 0:
            idxs = _limit_samples_per_object(idxs, ds, m, seed=args.debug_seed)
        return np.asarray(idxs, dtype=np.int64)

    use_sample_mode = bool(getattr(args, "debug_use_sample_mode", False))
    if use_sample_mode:
        valid = [
            i for i in range(ds.num_samples)
            if (ds.labels[i] > 0.5).any()
        ]
        if not valid:
            raise RuntimeError("No contact-positive samples for debug sample mode")
        n = max(1, int(args.debug_num_samples))
        return np.asarray(valid[:n], dtype=np.int64)

    obj_pos = _object_to_positive_indices(ds)
    if not obj_pos:
        raise RuntimeError(
            "No objects with positive contact labels; cannot run debug overfit",
        )

    k = max(1, int(args.debug_num_objects))
    selected_oids = _pick_object_ids(
        sorted(obj_pos.keys()),
        k,
        mode=args.debug_object_mode,
        seed=args.debug_seed,
    )
    m = int(getattr(args, "debug_samples_per_object", 0) or 0)
    out: list[int] = []
    for oid in selected_oids:
        idxs = obj_pos[oid]
        if m > 0:
            idxs = _limit_samples_per_object(idxs, ds, m, seed=args.debug_seed + hash(oid) % 997)
        out.extend(idxs)
    return np.asarray(out, dtype=np.int64)


def select_debug_vis_object_ids(
    sample_obj_ids: list[str],
    per_object_metrics: dict | None,
    *,
    max_objects: int = 10,
) -> list[str]:
    """Subset of object ids for debug visualization columns."""
    unique = sorted(set(sample_obj_ids))
    if len(unique) <= max_objects:
        return unique

    cap = max(1, max_objects)
    vis = unique[: cap - 1]
    worst = None
    if per_object_metrics:
        by_f1 = [
            (oid, per_object_metrics[oid].get("f1", 0.0))
            for oid in unique
            if oid in per_object_metrics
        ]
        if by_f1:
            worst = min(by_f1, key=lambda x: x[1])[0]
    if worst and worst not in vis:
        vis.append(worst)
    elif len(vis) < cap:
        vis.append(unique[cap - 1])
    return vis[:cap]


def build_debug_datasets(
    dataset_dir: str,
    args: argparse.Namespace,
) -> tuple[SoftAffordanceDataset, SoftAffordanceDataset, dict]:
    """Load pool, subset by debug object/sample rules; train == val."""
    train_h5 = os.path.join(dataset_dir, "affordance_train.h5")
    val_h5 = os.path.join(dataset_dir, "affordance_val.h5")
    train_obj_ids, val_obj_ids = get_object_split(
        train_h5, val_h5, val_ratio=args.val_ratio, seed=args.split_seed,
    )
    all_obj_ids = train_obj_ids | val_obj_ids

    synthetic = args.debug_synthetic_label if args.debug_synthetic_label else None
    aug_cfg = augment_config_from_args(args)

    pool = SoftAffordanceDataset(
        train_h5,
        all_obj_ids,
        augment=not args.no_augment,
        augment_config=aug_cfg,
        heatmap_sigma_ratio=args.heatmap_sigma_ratio,
        synthetic_label=synthetic,
    )
    from_val = SoftAffordanceDataset(
        val_h5,
        all_obj_ids,
        augment=not args.no_augment,
        augment_config=aug_cfg,
        heatmap_sigma_ratio=args.heatmap_sigma_ratio,
        synthetic_label=synthetic,
    )
    if from_val.num_samples > 0:
        pool.points = np.concatenate([pool.points, from_val.points])
        pool.normals = np.concatenate([pool.normals, from_val.normals])
        pool.labels = np.concatenate([pool.labels, from_val.labels])
        pool.force_centers = np.concatenate([pool.force_centers, from_val.force_centers])
        pool.sample_obj_ids = pool.sample_obj_ids + from_val.sample_obj_ids
        pool.num_samples = len(pool.points)

    indices = select_debug_indices(pool, args)
    train_ds = _subset_dataset(pool, indices)
    val_ds = train_ds

    info = {
        "debug_indices": indices.tolist(),
        "debug_object_ids": sorted(set(train_ds.sample_obj_ids)),
        "num_samples": len(indices),
        "num_objects": len(set(train_ds.sample_obj_ids)),
        "debug_num_objects": int(args.debug_num_objects),
        "debug_samples_per_object": int(getattr(args, "debug_samples_per_object", 0) or 0),
        "debug_object_mode": args.debug_object_mode,
        "debug_use_sample_mode": bool(getattr(args, "debug_use_sample_mode", False)),
        "synthetic_label": synthetic,
        "loss_weights": {
            "lambda_aff": args.lambda_aff,
            "lambda_binary": args.lambda_binary,
            "lambda_peak": args.lambda_peak,
            "lambda_center_heatmap": args.lambda_center_heatmap,
            "lambda_center_head": args.lambda_center_head,
            "lambda_consistency": args.lambda_consistency,
            "lambda_smooth": args.lambda_smooth,
        },
    }
    return train_ds, val_ds, info


@torch.no_grad()
def compute_per_object_debug_metrics(
    model: torch.nn.Module,
    dataset: SoftAffordanceDataset,
    device: torch.device,
) -> dict[str, dict]:
    """Per-object F1/AP averaged over all debug samples for that object."""
    model.eval()
    obj_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, oid in enumerate(dataset.sample_obj_ids):
        obj_to_idxs[oid].append(i)

    out: dict[str, dict] = {}
    for oid, idxs in obj_to_idxs.items():
        rows = []
        for idx in idxs:
            sample = dataset[idx]
            xyz, feat, lbl, soft, fc = [x.unsqueeze(0).to(device) for x in sample[:5]]
            seg, fc_p = forward_seg_fc(model, xyz, feat)
            prob = affordance_probability(seg)
            rows.append(
                compute_affordance_metrics(prob, lbl, soft, xyz, fc_p, fc),
            )
        out[oid] = {
            "f1": float(np.mean([r["f1"] for r in rows])),
            "ap": float(np.mean([r.get("ap", 0.0) for r in rows])),
            "collapsed": any(r.get("collapsed", False) for r in rows),
        }
    return out


def summarize_object_metrics(per_obj: dict[str, dict]) -> dict:
    if not per_obj:
        return {}
    f1s = [m["f1"] for m in per_obj.values()]
    aps = [m.get("ap", 0.0) for m in per_obj.values()]
    collapsed = sum(1 for m in per_obj.values() if m.get("collapsed", False))
    worst_f1 = min(per_obj.items(), key=lambda x: x[1]["f1"])
    worst_ap = min(per_obj.items(), key=lambda x: x[1].get("ap", 0.0))
    return {
        "per_object_f1_mean": float(np.mean(f1s)),
        "per_object_f1_min": float(np.min(f1s)),
        "per_object_f1_max": float(np.max(f1s)),
        "per_object_ap_mean": float(np.mean(aps)),
        "per_object_ap_min": float(np.min(aps)),
        "per_object_ap_max": float(np.max(aps)),
        "num_collapsed_objects": collapsed,
        "worst_object_f1": worst_f1[0],
        "worst_object_f1_value": float(worst_f1[1]["f1"]),
        "worst_object_ap": worst_ap[0],
        "worst_object_ap_value": float(worst_ap[1].get("ap", 0.0)),
    }


def format_debug_log_line(step: int, parts: dict, metrics: dict) -> str:
    span = metrics.get("prob_span", metrics.get("prob_max", 0) - metrics.get("prob_min", 0))
    base = (
        f"{step:>5} | loss={parts.get('total', 0):.4f} "
        f"bin={parts.get('binary', 0):.4f} soft={parts.get('aff', 0):.4f} "
        f"peak={parts.get('peak', 0):.4f} | "
        f"pμ={metrics.get('prob_mean', 0):.3f} "
        f"p[{metrics.get('prob_min', 0):.2f},{metrics.get('prob_max', 0):.2f}] "
        f"span={span:.3f} | "
        f"μ+={metrics.get('pred_mean_on_GT_contact', 0):.3f} "
        f"μ-={metrics.get('pred_mean_on_GT_noncontact', 0):.3f} | "
        f"F1={metrics.get('f1', 0):.2%} AP={metrics.get('ap', 0):.2%} "
        f"top1={metrics.get('precision_top1pct', 0):.2%} "
        f"grad={metrics.get('seg_head_grad_norm', 0):.2e}"
    )
    if "logit_gap_mean" in metrics:
        base += (
            f" | gapμ={metrics['logit_gap_mean']:.2f}"
            f"[{metrics.get('logit_gap_min', 0):.1f},{metrics.get('logit_gap_max', 0):.1f}]"
            f" cμ={metrics.get('contact_logit_mean', 0):.2f}"
            f" ncμ={metrics.get('noncontact_logit_mean', 0):.2f}"
        )
    if "per_object_f1_mean" in metrics:
        base += (
            f" | objF1 μ/min={metrics['per_object_f1_mean']:.0%}/"
            f"{metrics['per_object_f1_min']:.0%} "
            f"coll={int(metrics.get('num_collapsed_objects', 0))} "
            f"worst={metrics.get('worst_object_f1', '?')}"
        )
    return base


def write_debug_manifest(run_dir: str, info: dict, args: argparse.Namespace) -> None:
    path = os.path.join(run_dir, "debug_manifest.json")
    payload = {
        **info,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
