#!/usr/bin/env python3
"""
train_affordance.py — PointNet++ affordance training (no_rot executed dataset)

基于 model/train.py；主监督为 soft affordance heatmap；early stop / best_ckpt 按 0.5·F1+0.5·AP。
可选 DDP：--gpus 0 单卡；--gpus 0,1,2 多卡 DDP（batch_size 为每卡）。

用法:
    python model/train_affordance.py
    python -m model.affordance.train --gpus 0
    python model/train_affordance.py --gpus 0,1 --batch_size 8
    python model/train_affordance.py --resume output/affordance_no_rot_executed/training_runs/20260522_184530/ckpt/last_ckpt.pth

每次新 run（无 --resume）:
    output/affordance_no_rot_executed/training_runs/<YYYYMMDD_HHMMSS>/
        ckpt/   — best_ckpt.pth, last_ckpt.pth, training_history.json, ...
        vis/    — val_epoch_*.png（不在 ckpt 下）
        log/    — train.log（与终端相同的配置 + epoch 表格）

Debug few-object overfit (curriculum K=1,5,20):
    --debug-overfit-one-object --debug-num-objects 5
    --lambda-binary 1.0 --lambda-aff 0.3 --lambda-peak 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.affordance.dataset import AFFORDANCE_IN_CHANNELS, SoftAffordanceDataset
from model.affordance.augment import AugmentConfig, augment_config_from_args
from model.affordance.debug import (
    apply_debug_config,
    build_debug_datasets,
    select_debug_vis_object_ids,
    write_debug_manifest,
)
from model.affordance.logging_utils import close_training_log, open_training_log, training_log
from model.affordance.losses import (
    AffordanceLossWeights,
    build_affordance_criterion,
    loss_weights_from_args,
)
from model.affordance.metrics import checkpoint_score, save_val_objects_grid
from model.affordance.pointnet2_ops import PointNet2Seg
from model.affordance.train_loop import eval_epoch_mt, run_debug_overfit_loop, train_epoch_mt
from model.train import get_object_split

DEFAULT_DATASET_DIR = os.path.join(PROJ, "output", "affordance_no_rot_executed")
TRAINING_RUNS_DIRNAME = "training_runs"
LAST_CKPT_NAME = "last_ckpt.pth"
BEST_CKPT_NAME = "best_ckpt.pth"


def _ensure_unique_run_dir(run_dir: str) -> str:
    """If run_dir exists, append __YYYYMMDD_HHMMSS to avoid overwriting."""
    if not os.path.exists(run_dir):
        return run_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(run_dir.rstrip(os.sep))
    parent = os.path.dirname(run_dir.rstrip(os.sep))
    return os.path.join(parent, f"{base}__{stamp}")


def resolve_run_paths(
    dataset_dir: str,
    save_dir: str | None,
    resume: str | None,
    *,
    run_name: str | None = None,
    run_group: str | None = None,
) -> tuple[str, str, str, str]:
    """
    Returns (run_dir, ckpt_dir, vis_dir, log_dir).
    Priority: --resume paths > --save_dir > --run-name[/ --run-group] > timestamp.
    """
    runs_root = os.path.join(dataset_dir, TRAINING_RUNS_DIRNAME)

    if resume:
        resume_path = os.path.abspath(os.path.expanduser(resume))
        ckpt_dir = os.path.dirname(resume_path)
        if os.path.basename(ckpt_dir) == "ckpt":
            run_dir = os.path.dirname(ckpt_dir)
            vis_dir = os.path.join(run_dir, "vis")
            log_dir = os.path.join(run_dir, "log")
        else:
            # legacy layout: checkpoints directly under .../ckpt without training_runs parent
            run_dir = ckpt_dir
            vis_dir = os.path.join(run_dir, "vis")
            log_dir = os.path.join(run_dir, "log")
        return run_dir, ckpt_dir, vis_dir, log_dir

    if save_dir:
        path = os.path.abspath(os.path.expanduser(save_dir))
        if os.path.basename(path) == "ckpt":
            ckpt_dir = path
            run_dir = os.path.dirname(path)
            vis_dir = os.path.join(run_dir, "vis")
            log_dir = os.path.join(run_dir, "log")
        else:
            run_dir = path
            ckpt_dir = os.path.join(run_dir, "ckpt")
            vis_dir = os.path.join(run_dir, "vis")
            log_dir = os.path.join(run_dir, "log")
        return run_dir, ckpt_dir, vis_dir, log_dir

    if run_name:
        name = run_name.strip()
        if run_group:
            run_dir = os.path.join(runs_root, run_group.strip(), name)
        else:
            run_dir = os.path.join(runs_root, name)
        run_dir = _ensure_unique_run_dir(run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_root, stamp)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    vis_dir = os.path.join(run_dir, "vis")
    log_dir = os.path.join(run_dir, "log")
    return run_dir, ckpt_dir, vis_dir, log_dir


def write_run_manifest(run_dir: str, args: argparse.Namespace) -> None:
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "run_name": getattr(args, "run_name", None),
        "run_group": getattr(args, "run_group", None),
        "ckpt_dir": args.ckpt_dir,
        "vis_dir": args.vis_dir,
        "log_dir": args.log_dir,
        "dataset_dir": args.dataset_dir,
        "predict_force_center": bool(getattr(args, "predict_force_center", False)),
        "in_channel": AFFORDANCE_IN_CHANNELS,
        "early_stop_metric": "val_score_0.5_f1_0.5_ap",
        "loss": "soft_heatmap_v1",
        "args": vars(args),
    }
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def parse_gpu_list(gpus: str) -> list[int]:
    ids = [int(x.strip()) for x in gpus.split(",") if x.strip() != ""]
    if not ids:
        raise ValueError("--gpus must list at least one GPU id, e.g. 0 or 0,1")
    return ids


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_seg_head_bias(model: torch.nn.Module, pos_fraction: float) -> None:
    """Initialize class-1 logit bias from contact prior (avoids constant ~0.5 prob)."""
    raw = unwrap_model(model)
    if getattr(raw, "conv2", None) is None or raw.conv2.bias is None:
        return
    p = float(np.clip(pos_fraction, 1e-4, 1.0 - 1e-4))
    logit = float(np.log(p / (1.0 - p)))
    with torch.no_grad():
        raw.conv2.bias.zero_()
        raw.conv2.bias[1] = logit


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_main(rank: int, msg: str, *, flush: bool = False) -> None:
    if is_main_process(rank):
        training_log(msg, flush=flush)


def resolve_per_gpu_batch_size(
    num_train_samples: int,
    batch_size: int,
    world_size: int,
    *,
    drop_last: bool = True,
) -> int:
    """
    每 rank 样本数过少时缩小 batch，避免 DataLoader 0 batch（DDP 常见）。
    """
    if world_size > 1:
        per_rank = num_train_samples // world_size
    else:
        per_rank = num_train_samples
    per_rank = max(1, per_rank)
    if drop_last:
        max_bs = max(1, per_rank)
    else:
        max_bs = per_rank
    return max(1, min(batch_size, max_bs))


def build_train_val_datasets(
    dataset_dir: str,
    train_obj_ids: set[str],
    val_obj_ids: set[str],
    *,
    heatmap_sigma_ratio: float = 0.05,
    train_augment_config: AugmentConfig | None = None,
) -> tuple[SoftAffordanceDataset, SoftAffordanceDataset]:
    train_h5 = os.path.join(dataset_dir, "affordance_train.h5")
    val_h5 = os.path.join(dataset_dir, "affordance_val.h5")
    no_aug = AugmentConfig(False, False, False, False, False)
    train_aug = train_augment_config or AugmentConfig()

    train_dataset = SoftAffordanceDataset(
        train_h5,
        train_obj_ids,
        augment_config=train_aug,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
    )
    val_from_val = SoftAffordanceDataset(
        val_h5, val_obj_ids, augment_config=no_aug, heatmap_sigma_ratio=heatmap_sigma_ratio,
    )
    val_from_train = SoftAffordanceDataset(
        train_h5, val_obj_ids, augment_config=no_aug, heatmap_sigma_ratio=heatmap_sigma_ratio,
    )

    val_dataset = val_from_val
    val_dataset.points = np.concatenate([val_from_val.points, val_from_train.points])
    val_dataset.normals = np.concatenate([val_from_val.normals, val_from_train.normals])
    val_dataset.labels = np.concatenate([val_from_val.labels, val_from_train.labels])
    val_dataset.human_priors = np.concatenate(
        [val_from_val.human_priors, val_from_train.human_priors],
    )
    val_dataset.force_centers = np.concatenate(
        [val_from_val.force_centers, val_from_train.force_centers],
    )
    val_dataset.sample_obj_ids = val_from_val.sample_obj_ids + val_from_train.sample_obj_ids
    val_dataset.num_samples = len(val_dataset.points)

    train_from_val = SoftAffordanceDataset(
        val_h5,
        train_obj_ids,
        augment_config=train_aug,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
    )
    train_dataset.points = np.concatenate([train_dataset.points, train_from_val.points])
    train_dataset.normals = np.concatenate([train_dataset.normals, train_from_val.normals])
    train_dataset.labels = np.concatenate([train_dataset.labels, train_from_val.labels])
    train_dataset.human_priors = np.concatenate(
        [train_dataset.human_priors, train_from_val.human_priors],
    )
    train_dataset.force_centers = np.concatenate(
        [train_dataset.force_centers, train_from_val.force_centers],
    )
    train_dataset.sample_obj_ids = (
        train_dataset.sample_obj_ids + train_from_val.sample_obj_ids
    )
    train_dataset.num_samples = len(train_dataset.points)

    return train_dataset, val_dataset


def build_lr_scheduler(optimizer, args):
    """Warmup (linear) + cosine decay; resume 时从 last_ckpt 恢复 scheduler state。"""
    warmup = max(0, int(args.warmup_epochs))
    total = int(args.epochs)
    cosine_T = max(1, total - warmup)

    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_T,
            eta_min=args.lr_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total,
        eta_min=args.lr_min,
    )


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def try_resume(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    rank: int,
) -> dict | None:
    if not os.path.isfile(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    log_main(rank, f"  ↻ Resumed from {path} @ epoch {ckpt.get('epoch', '?')}")
    return ckpt


def broadcast_stop_flag(should_stop: bool, rank: int, world_size: int, device: torch.device) -> bool:
    if world_size <= 1:
        return should_stop
    flag = torch.zeros(1, device=device, dtype=torch.int32)
    if rank == 0:
        flag[0] = 1 if should_stop else 0
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train affordance (no_rot executed)")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids, e.g. 0 or 0,1,2 (1 GPU=single, 2+=DDP)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Per-GPU batch size (DDP: effective = batch_size × num_gpus)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Cosine floor LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs before cosine (0 = off)")
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--heatmap-sigma-ratio", type=float, default=0.05,
                        help="σ = ratio × object bbox diagonal for Gaussian heatmap")
    parser.add_argument("--lambda-aff", type=float, default=0.3,
                        help="Weight on soft affordance heatmap loss L_aff")
    parser.add_argument("--lambda-binary", type=float, default=1.0,
                        help="Weight on binary segmentation loss L_binary")
    parser.add_argument("--lambda-peak", type=float, default=0.0,
                        help="Weight on peak contact loss L_peak (positive-only BCE)")
    parser.add_argument("--lambda-center-heatmap", type=float, default=0.0,
                        help="Weight on heatmap-derived center L1 loss")
    parser.add_argument("--lambda-center-head", type=float, default=0.0,
                        help="Weight on FC head regression loss")
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--ckpt-f1-weight", type=float, default=0.5,
                        help="best_score = w·F1 + (1-w)·AP")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stop after N epochs without score gain (warmup excluded)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Run root or ckpt/ path (highest priority if set)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name under training_runs/[run-group]/ (auto-suffix if exists)",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional group folder under training_runs/ (e.g. sweep name)",
    )
    parser.add_argument(
        "--predict-force-center",
        action="store_true",
        help="Build FC head in PointNet++ (loss still controlled by --lambda-center-head)",
    )
    parser.add_argument(
        "--val-vis-max-objects",
        type=int,
        default=10,
        help="Max object columns in val_epoch_*.png (0 = all val objects)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="Resume from this checkpoint path (default: train from scratch)",
    )
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers per process")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="DDP master port (multi-GPU only)")
    parser.add_argument(
        "--augment-mode",
        type=str,
        default="full",
        choices=("none", "weak", "full"),
        help="Augmentation preset (overridden by --no-augment)",
    )
    parser.add_argument(
        "--debug-overfit-one-object",
        action="store_true",
        help="Few-object overfit debug (step-based; use --debug-num-objects)",
    )
    parser.add_argument("--debug-object-id", type=str, default=None,
                        help="Use all samples for this object id")
    parser.add_argument("--debug-num-objects", type=int, default=1,
                        help="Select K objects with contact labels (default 1)")
    parser.add_argument("--debug-samples-per-object", type=int, default=0,
                        help="Max samples per object (0 = all)")
    parser.add_argument(
        "--debug-object-mode",
        type=str,
        default="first",
        choices=("first", "random"),
        help="How to pick K objects from the pool",
    )
    parser.add_argument("--debug-seed", type=int, default=42,
                        help="RNG seed for random object/sample selection")
    parser.add_argument("--debug-num-samples", type=int, default=1,
                        help="With --debug-use-sample-mode: first N contact-positive samples")
    parser.add_argument(
        "--debug-use-sample-mode",
        action="store_true",
        help="Sample-level subset (legacy 1-sample) instead of object-level",
    )
    parser.add_argument("--debug-vis-max-objects", type=int, default=10,
                        help="Max object columns in debug_step_*.png")
    parser.add_argument("--debug-max-steps", type=int, default=1000,
                        help="Optimization steps in debug overfit mode")
    parser.add_argument("--binary-tversky-alpha", type=float, default=0.5,
                        help="Tversky FP weight (higher → stronger FP penalty)")
    parser.add_argument("--binary-tversky-beta", type=float, default=0.5,
                        help="Tversky FN weight")
    parser.add_argument("--binary-neg-ratio", type=float, default=1.0,
                        help="Balanced binary sampling: neg = ratio × pos count")
    parser.add_argument("--soft-background-weight", type=float, default=0.25,
                        help="Weight on background term in soft focal loss")
    parser.add_argument(
        "--head-norm",
        type=str,
        default="none",
        choices=("none", "layernorm", "groupnorm"),
        help="Norm on final per-point features before seg head",
    )
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable dataset augmentation")
    parser.add_argument("--disable-center-loss", action="store_true",
                        help="Zero center heatmap / head losses")
    parser.add_argument("--disable-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument(
        "--debug-synthetic-label",
        type=str,
        default=None,
        choices=("x_positive", "z_positive"),
        help="Replace contact labels with coordinate half-space toy labels",
    )
    parser.add_argument("--debug-log-interval", type=int, default=20,
                        help="Log debug metrics every N steps")
    parser.add_argument("--debug-vis-interval", type=int, default=50,
                        help="Save debug_step_*.png every N steps")
    return parser.parse_args()


def _apply_disable_center_loss(args) -> None:
    if args.disable_center_loss:
        args.lambda_center_heatmap = 0.0
        args.lambda_center_head = 0.0
        args.lambda_consistency = 0.0


def log_loss_weights(rank: int, args) -> None:
    log_main(rank, "  Loss weights:")
    log_main(rank, f"    lambda_aff              {args.lambda_aff}")
    log_main(rank, f"    lambda_binary           {args.lambda_binary}")
    log_main(rank, f"    lambda_peak             {args.lambda_peak}")
    log_main(rank, f"    lambda_center_heatmap   {args.lambda_center_heatmap}")
    log_main(rank, f"    lambda_center_head      {args.lambda_center_head}")
    log_main(rank, f"    lambda_consistency      {args.lambda_consistency}")
    log_main(rank, f"    lambda_smooth           {args.lambda_smooth}")
    log_main(
        rank,
        f"  Binary: Tversky α/β={args.binary_tversky_alpha}/{args.binary_tversky_beta}  "
        f"neg_ratio={args.binary_neg_ratio}",
    )
    log_main(rank, f"  Soft heatmap: background_weight={args.soft_background_weight}")
    log_main(rank, f"  Head norm:     {args.head_norm}")
    log_main(rank, f"  Heatmap σ:     {args.heatmap_sigma_ratio}")


def _train_debug_overfit(
    rank: int,
    args,
    device: torch.device,
    gpu_ids: list[int],
) -> None:
    """Step-based few-object overfit (rank 0 only)."""
    vis_dir = args.vis_dir
    log_main(rank, "=" * 70)
    log_main(rank, "DEBUG OVERFIT — few objects / small subset")
    log_main(rank, "=" * 70)
    log_main(rank, f"  GPUs:          {gpu_ids}")
    log_main(rank, f"  Max steps:     {args.debug_max_steps}")
    log_main(rank, f"  Object id:     {args.debug_object_id or '(auto)'}")
    log_main(rank, f"  Num objects:   {args.debug_num_objects}")
    log_main(rank, f"  Samples/obj:   {args.debug_samples_per_object or 'all'}")
    log_main(rank, f"  Object pick:   {args.debug_object_mode} (seed={args.debug_seed})")
    log_main(rank, f"  Sample mode:   {args.debug_use_sample_mode}")
    log_main(rank, f"  Synthetic lbl: {args.debug_synthetic_label or '(robot_gt)'}")
    log_main(rank, f"  Augment:       {not args.no_augment}")
    log_main(rank, f"  LR / WD:       {args.lr} / {args.weight_decay}")
    log_loss_weights(rank, args)

    train_dataset, val_dataset, debug_info = build_debug_datasets(args.dataset_dir, args)
    write_debug_manifest(args.run_dir, debug_info, args)
    log_main(
        rank,
        f"  Subset:        {debug_info['num_samples']} sample(s) / "
        f"{debug_info['num_objects']} object(s)  ids={debug_info['debug_object_ids']}",
    )

    bs = min(args.batch_size, max(1, len(train_dataset)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # fc_head BatchNorm needs batch>1; default --predict-force-center is off for debug.
    model = PointNet2Seg(
        num_classes=2,
        in_channel=AFFORDANCE_IN_CHANNELS,
        predict_force_center=args.predict_force_center,
        head_norm=args.head_norm,
    ).to(device)
    init_seg_head_bias(model, (train_dataset.labels > 0.5).mean())
    log_main(rank, f"  Input:        xyz ({AFFORDANCE_IN_CHANNELS}ch, no normals)")
    log_main(rank, f"  FC head:      {args.predict_force_center}")
    log_main(rank, f"  Head norm:     {args.head_norm}")

    criterion = build_affordance_criterion(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def _log(msg: str) -> None:
        log_main(rank, msg, flush=True)

    try:
        run_debug_overfit_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            val_dataset,
            vis_dir,
            args.ckpt_dir,
            max_steps=args.debug_max_steps,
            log_interval=args.debug_log_interval,
            vis_interval=args.debug_vis_interval,
            log_fn=_log,
            unwrap_model_fn=unwrap_model,
            vis_max_objects=args.debug_vis_max_objects,
            object_metrics_interval=args.debug_log_interval,
        )
        save_checkpoint(
            os.path.join(args.ckpt_dir, "debug_last_ckpt.pth"),
            {"model_state_dict": model.state_dict(), "debug_info": debug_info, "args": vars(args)},
        )
        log_main(rank, "\n  DEBUG OVERFIT COMPLETE")
        log_main(rank, f"  Vis: {vis_dir}/debug_step_*.png")
    finally:
        pass


def train_worker(rank: int, world_size: int, args) -> None:
    distributed = world_size > 1
    if args.debug_overfit_one_object and distributed:
        raise RuntimeError("--debug-overfit-one-object requires a single GPU")
    if distributed:
        setup_ddp(rank, world_size)
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for train_affordance.py")

    device = torch.device(f"cuda:{rank}")
    gpu_ids = parse_gpu_list(args.gpus)

    if is_main_process(rank):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        open_training_log(args.log_dir, resume=bool(args.resume))

    if distributed:
        dist.barrier()

    if args.debug_overfit_one_object:
        try:
            _train_debug_overfit(rank, args, device, gpu_ids)
        finally:
            if is_main_process(rank):
                close_training_log()
            cleanup_ddp()
        return

    vis_dir = args.vis_dir
    last_ckpt_path = os.path.join(args.ckpt_dir, LAST_CKPT_NAME)
    best_ckpt_path = os.path.join(args.ckpt_dir, BEST_CKPT_NAME)

    log_main(rank, "=" * 70)
    log_main(rank, "train_affordance — PointNet++ Multi-Task")
    log_main(rank, "=" * 70)
    log_main(rank, f"  Mode:         {'DDP × ' + str(world_size) if distributed else 'single-GPU'}")
    log_main(rank, f"  Physical GPUs: {gpu_ids}  (CUDA_VISIBLE_DEVICES)")
    log_main(rank, f"  Local device:  cuda:{rank}")
    log_main(rank, f"  Dataset:      {args.dataset_dir}")
    log_main(rank, f"  Run dir:      {args.run_dir}")
    if getattr(args, "run_name", None):
        grp = f"  group={args.run_group}" if args.run_group else ""
        log_main(rank, f"  Run name:     {args.run_name}{grp}")
    log_main(rank, f"  Input:        xyz ({AFFORDANCE_IN_CHANNELS}ch, no normals)")
    log_main(rank, f"  FC head:      {args.predict_force_center}")
    log_main(rank, f"  Checkpoints:  {args.ckpt_dir}")
    log_main(rank, f"  Val vis:      {vis_dir}")
    log_main(rank, f"  Train log:    {os.path.join(args.log_dir, 'train.log')}")
    log_main(rank, f"  Epochs:       {args.epochs}  val_ratio={args.val_ratio}")
    log_main(rank, f"  LR:           {args.lr} → {args.lr_min}  warmup={args.warmup_epochs}ep")
    log_main(
        rank,
        f"  Best ckpt:    {args.ckpt_f1_weight:.0%}·F1 + {1-args.ckpt_f1_weight:.0%}·AP",
    )
    log_main(
        rank,
        f"  Early stop:   patience={args.patience} after warmup "
        f"({args.warmup_epochs} ep excluded)",
    )
    log_main(rank, f"  Resume:       {args.resume or '(from scratch)'}")
    sys.stdout.flush()

    train_h5 = os.path.join(args.dataset_dir, "affordance_train.h5")
    val_h5 = os.path.join(args.dataset_dir, "affordance_val.h5")
    train_obj_ids, val_obj_ids = get_object_split(
        train_h5, val_h5, val_ratio=args.val_ratio, seed=args.split_seed,
    )
    log_main(rank, f"\n  Train objects: {len(train_obj_ids)}  Val objects: {len(val_obj_ids)}")
    assert not (train_obj_ids & val_obj_ids)

    train_aug = augment_config_from_args(args)
    train_dataset, val_dataset = build_train_val_datasets(
        args.dataset_dir,
        train_obj_ids,
        val_obj_ids,
        heatmap_sigma_ratio=args.heatmap_sigma_ratio,
        train_augment_config=train_aug,
    )
    log_main(
        rank,
        f"  Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}",
    )
    tr_pos = (train_dataset.labels > 0.5).mean() * 100
    va_pos = (val_dataset.labels > 0.5).mean() * 100
    log_main(rank, f"  Contact ratio: train={tr_pos:.2f}%  val={va_pos:.2f}%")
    log_main(rank, f"  Augment mode:   {args.augment_mode}  (no_augment={args.no_augment})")
    log_loss_weights(rank, args)

    train_batch_size = resolve_per_gpu_batch_size(
        len(train_dataset), args.batch_size, world_size, drop_last=True,
    )
    log_main(
        rank,
        f"  Train batch/GPU: {train_batch_size}  "
        f"(requested {args.batch_size}, effective≈{train_batch_size * world_size})",
    )
    if train_batch_size < args.batch_size:
        log_main(
            rank,
            f"  ⚠ capped batch_size: {len(train_dataset)} samples / DDP×{world_size} "
            f"→ max {train_batch_size}/GPU",
        )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PointNet2Seg(
        num_classes=2,
        in_channel=AFFORDANCE_IN_CHANNELS,
        predict_force_center=args.predict_force_center,
        head_norm=args.head_norm,
    ).to(device)
    if not args.resume:
        init_seg_head_bias(model, (train_dataset.labels > 0.5).mean())
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)

    loss_weights = loss_weights_from_args(args)
    criterion = build_affordance_criterion(args).to(device)
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_score = 0.0
    best_val_f1 = 0.0
    best_val_ap = 0.0
    epochs_without_improvement = 0
    history: list[dict] = []

    scheduler = build_lr_scheduler(optimizer, args)

    if args.resume:
        resume_path = os.path.abspath(os.path.expanduser(args.resume))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = try_resume(resume_path, model, optimizer, scheduler, device, rank)
        if ckpt is not None:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_val_score = float(ckpt.get("best_val_score", 0.0))
            best_val_f1 = float(ckpt.get("best_val_f1", 0.0))
            best_val_ap = float(ckpt.get("best_val_ap", 0.0))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
            if is_main_process(rank):
                history = list(ckpt.get("history", []))
            if ckpt.get("train_objects"):
                train_obj_ids = set(ckpt["train_objects"])
            if ckpt.get("val_objects"):
                val_obj_ids = set(ckpt["val_objects"])
        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()

    log_main(rank, f"\n{'='*98}")
    log_main(
        rank,
        f"{'Ep':>4} | {'trLoss':>7} | {'vaLoss':>7} | {'vaF1':>5} | {'vaAP':>5} | "
        f"{'score':>5} | {'FCmm':>5} | {'LR':>8} | {'stop':>4}",
    )
    log_main(rank, "-" * 98)

    stopped_early = False
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t0 = time.time()
            t_loss, t_aff, t_bin, t_metrics = train_epoch_mt(
                model, train_loader, optimizer, criterion, device,
            )

            v_loss, v_aff, v_bin, v_fc_mm, v_metrics = (0.0, 0.0, 0.0, 0.0, {})
            if is_main_process(rank):
                v_loss, v_aff, v_bin, v_fc_mm, v_metrics = eval_epoch_mt(
                    model, val_loader, criterion, device,
                )

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            improved = False
            if is_main_process(rank):
                collapsed = v_metrics.get("collapsed", False)
                val_score = checkpoint_score(v_metrics, f1_weight=args.ckpt_f1_weight)
                score_improved = val_score > best_val_score + 1e-8
                reject_collapse = getattr(args, "ckpt_reject_collapse", True)
                improved = score_improved and (not collapsed or not reject_collapse)
                in_warmup = epoch <= int(args.warmup_epochs)
                if improved:
                    best_val_score = val_score
                    best_val_f1 = v_metrics["f1"]
                    best_val_ap = v_metrics.get("ap", 0.0)
                    epochs_without_improvement = 0
                elif not in_warmup:
                    epochs_without_improvement += 1
                if v_loss < best_val_loss - 1e-6:
                    best_val_loss = v_loss

                collapse_tag = " COLLAPSE" if collapsed else ""
                stop_disp = "  wu" if in_warmup else f"{epochs_without_improvement:>4}"
                log_main(
                    rank,
                    f"{epoch:>4} | {t_loss:>7.3f} | {v_loss:>7.3f} | "
                    f"{v_metrics['f1']:>4.0%} | {v_metrics.get('ap', 0):>4.0%} | "
                    f"{val_score:>4.0%} | {v_fc_mm:>4.0f} | {lr:>8.6f} | "
                    f"{stop_disp}  ({elapsed:.0f}s){collapse_tag}",
                    flush=True,
                )
                if collapsed:
                    log_main(
                        rank,
                        f"        ⚠ seg collapse: pred+={v_metrics['pred_pos_frac']:.1%} "
                        f"prob μ={v_metrics['prob_mean']:.3f} "
                        f"span=[{v_metrics['prob_min']:.3f},{v_metrics['prob_max']:.3f}]",
                    )

                row = {
                    "epoch": epoch,
                    "train_loss": round(t_loss, 5),
                    "train_aff": round(t_aff, 5),
                    "train_binary": round(t_bin, 5),
                    **{
                        f"train_{k}": round(v, 4)
                        for k, v in t_metrics.items()
                        if isinstance(v, (int, float)) and not k.startswith("train_")
                    },
                    "val_loss": round(v_loss, 5),
                    "val_aff": round(v_aff, 5),
                    "val_binary": round(v_bin, 5),
                    "val_score": round(val_score, 4),
                    "val_fc_mm": round(v_fc_mm, 2),
                    **{
                        f"val_{k}": round(v, 4)
                        for k, v in v_metrics.items()
                        if isinstance(v, (int, float))
                    },
                    "lr": round(lr, 7),
                    "time_s": round(elapsed, 1),
                    "epochs_without_improvement": epochs_without_improvement,
                }
                history.append(row)

                vis_path = os.path.join(vis_dir, f"val_epoch_{epoch:04d}.png")
                max_vis = int(args.val_vis_max_objects)
                if max_vis > 0:
                    vis_obj_ids = select_debug_vis_object_ids(
                        val_dataset.sample_obj_ids,
                        None,
                        max_objects=max_vis,
                    )
                else:
                    vis_obj_ids = None
                save_val_objects_grid(
                    model,
                    val_dataset,
                    val_dataset.sample_obj_ids,
                    device,
                    vis_path,
                    epoch,
                    vis_object_ids=vis_obj_ids,
                )

                raw = unwrap_model(model)
                ckpt_common = {
                    "epoch": epoch,
                    "model_state_dict": raw.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_score": best_val_score,
                    "best_val_f1": best_val_f1,
                    "best_val_ap": best_val_ap,
                    "early_stop_metric": "val_score_0.5_f1_0.5_ap",
                    "heatmap_sigma_ratio": args.heatmap_sigma_ratio,
                    "loss_weights": vars(loss_weights),
                    "run_dir": args.run_dir,
                    "vis_dir": vis_dir,
                    "log_dir": args.log_dir,
                    "epochs_without_improvement": epochs_without_improvement,
                    "in_warmup": in_warmup,
                    "val_loss": v_loss,
                    "val_f1": v_metrics["f1"],
                    "val_ap": v_metrics.get("ap", 0.0),
                    "val_score": val_score,
                    "val_fc_mm": v_fc_mm,
                    "val_iou": v_metrics["iou"],
                    "predict_force_center": args.predict_force_center,
                    "in_channel": AFFORDANCE_IN_CHANNELS,
                    "train_objects": sorted(train_obj_ids),
                    "val_objects": sorted(val_obj_ids),
                    "history": history,
                    "args": vars(args),
                    "world_size": world_size,
                    "gpus": args.gpus,
                }

                if improved:
                    save_checkpoint(
                        best_ckpt_path,
                        {
                            **ckpt_common,
                            "val_score_at_best": best_val_score,
                            "val_f1_at_best": best_val_f1,
                            "val_ap_at_best": best_val_ap,
                            "val_loss_at_best": v_loss,
                        },
                    )
                    log_main(
                        rank,
                        f"        ★ best_ckpt  score={best_val_score:.1%}  "
                        f"F1={best_val_f1:.1%} AP={best_val_ap:.1%}  "
                        f"FC_hm={v_fc_mm:.0f}mm",
                    )

                save_checkpoint(last_ckpt_path, ckpt_common)

                if epoch % 10 == 0:
                    save_checkpoint(
                        os.path.join(args.ckpt_dir, f"checkpoint_epoch{epoch}.pth"),
                        {"epoch": epoch, "model_state_dict": raw.state_dict()},
                    )

            if distributed:
                dist.barrier()

            stop_local = (
                not args.disable_early_stop
                and epoch > int(args.warmup_epochs)
                and epochs_without_improvement >= args.patience
                if is_main_process(rank)
                else False
            )
            should_stop = broadcast_stop_flag(stop_local, rank, world_size, device)
            if should_stop:
                if is_main_process(rank):
                    log_main(
                        rank,
                        f"\n  Early stop: no val score improvement for {args.patience} epochs "
                        f"(best score={best_val_score:.1%}, F1={best_val_f1:.1%}, AP={best_val_ap:.1%}).",
                    )
                stopped_early = True
                break

        if is_main_process(rank):
            with open(os.path.join(args.ckpt_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)
            split_info = {
                "run_dir": args.run_dir,
                "ckpt_dir": args.ckpt_dir,
                "vis_dir": vis_dir,
                "log_dir": args.log_dir,
                "train_objects": sorted(train_obj_ids),
                "val_objects": sorted(val_obj_ids),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "heatmap_sigma_ratio": args.heatmap_sigma_ratio,
                "val_ratio": args.val_ratio,
                "split_seed": args.split_seed,
                "stopped_early": stopped_early,
                "early_stop_metric": "val_score_0.5_f1_0.5_ap",
                "best_val_score": best_val_score,
                "best_val_f1": best_val_f1,
                "best_val_ap": best_val_ap,
                "best_val_loss": best_val_loss,
                "world_size": world_size,
                "gpus": args.gpus,
                "version": "train_affordance_soft_v2",
            }
            with open(os.path.join(args.ckpt_dir, "split_info.json"), "w") as f:
                json.dump(split_info, f, indent=2)

            log_main(rank, f"\n{'='*70}")
            log_main(rank, "TRAINING COMPLETE")
            log_main(rank, f"  Run dir:         {args.run_dir}")
            log_main(rank, f"  Best val score:  {best_val_score:.1%}  (F1={best_val_f1:.1%} AP={best_val_ap:.1%})")
            log_main(rank, f"  Best val loss:   {best_val_loss:.4f}  (logged only)")
            log_main(rank, f"  best_ckpt:       {best_ckpt_path}")
            log_main(rank, f"  last_ckpt:       {last_ckpt_path}")
            log_main(rank, f"  Val vis:         {vis_dir}/val_epoch_*.png")
            log_main(rank, f"  Train log:       {os.path.join(args.log_dir, 'train.log')}")
            log_main(rank, f"  Early stopped:   {stopped_early}")
            log_main(rank, "=" * 70)
    finally:
        if is_main_process(rank):
            close_training_log()
        cleanup_ddp()


def launch_training(args) -> None:
    gpu_ids = parse_gpu_list(args.gpus)
    if args.debug_overfit_one_object:
        if len(gpu_ids) > 1:
            print("  [debug] Using single GPU only for overfit mode")
        gpu_ids = [gpu_ids[0]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    os.environ["MASTER_PORT"] = str(args.master_port)

    if len(gpu_ids) == 1:
        train_worker(0, 1, args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(
            train_worker,
            args=(len(gpu_ids), args),
            nprocs=len(gpu_ids),
            join=True,
        )


def main():
    args = parse_args()
    _apply_disable_center_loss(args)
    apply_debug_config(args)
    run_dir, ckpt_dir, vis_dir, log_dir = resolve_run_paths(
        args.dataset_dir,
        args.save_dir,
        args.resume,
        run_name=args.run_name,
        run_group=args.run_group,
    )
    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    args.vis_dir = vis_dir
    args.log_dir = log_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not args.resume:
        write_run_manifest(run_dir, args)
    launch_training(args)


if __name__ == "__main__":
    main()
