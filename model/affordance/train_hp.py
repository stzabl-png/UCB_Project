#!/usr/bin/env python3
"""
Human-prior supervision affordance training (separate from train.py).

- Model: PointNet2SegOnly (no FC head)
- Loss: L1(pred, human_prior) only
- GT: human_prior (not robot_gt)

Entry: python model/train_affordance_hp.py
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

from model.affordance.augment import AugmentConfig, augment_config_from_args
from model.affordance.dataset import AFFORDANCE_IN_CHANNELS
from model.affordance.dataset_hp import build_train_val_datasets_hp
from model.affordance.debug import pick_random_vis_object_ids
from model.affordance.logging_utils import close_training_log, open_training_log, training_log
from model.affordance.losses_hp import L1HumanPriorLoss
from model.affordance.metrics import checkpoint_score
from model.affordance.metrics_hp import save_hp_objects_grid
from model.affordance.pointnet2_seg_only import PointNet2SegOnly
from model.affordance.train import (
    LAST_CKPT_NAME,
    BEST_CKPT_NAME,
    DEFAULT_DATASET_DIR,
    broadcast_stop_flag,
    build_lr_scheduler,
    is_main_process,
    log_main,
    parse_gpu_list,
    resolve_per_gpu_batch_size,
    resolve_run_paths,
    save_checkpoint,
    setup_ddp,
    cleanup_ddp,
    try_resume,
    unwrap_model,
)
from model.affordance.train_loop_hp import eval_epoch_hp, train_epoch_hp
from model.train import get_object_split

HP_RUN_GROUP_DEFAULT = "hp_supervision"


def init_seg_head_bias_hp(model: torch.nn.Module, pos_fraction: float) -> None:
    raw = unwrap_model(model)
    if getattr(raw, "seg_fc2", None) is None or raw.seg_fc2.bias is None:
        return
    p = float(np.clip(pos_fraction, 1e-4, 1.0 - 1e-4))
    logit = float(np.log(p / (1.0 - p)))
    with torch.no_grad():
        raw.seg_fc2.bias.fill_(logit)


def write_hp_run_manifest(run_dir: str, args: argparse.Namespace) -> None:
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "supervision": "human_prior",
        "loss": "l1_only",
        "model": "PointNet2SegOnly",
        "run_name": getattr(args, "run_name", None),
        "run_group": getattr(args, "run_group", None),
        "hp_threshold": args.hp_threshold,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train affordance with human_prior GT (MSE only, no FC head)",
    )
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--run-group", type=str, default=HP_RUN_GROUP_DEFAULT)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--master_port", type=str, default="29501")
    p.add_argument(
        "--augment-mode",
        type=str,
        default="full",
        choices=("none", "weak", "full"),
    )
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--head-norm", type=str, default="none", choices=("none", "layernorm", "groupnorm"))
    p.add_argument("--hp-threshold", type=float, default=0.5,
                   help="Binary metrics: human_prior > threshold")
    p.add_argument("--val-vis-max-objects", type=int, default=10)
    p.add_argument("--train-vis-max-objects", type=int, default=10)
    p.add_argument("--train-vis-seed", type=int, default=None)
    p.add_argument("--ckpt-f1-weight", type=float, default=0.5)
    p.add_argument("--disable-early-stop", action="store_true")
    return p.parse_args()


def train_worker_hp(rank: int, world_size: int, args) -> None:
    distributed = world_size > 1
    if distributed:
        setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    gpu_ids = parse_gpu_list(args.gpus)

    if is_main_process(rank):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        open_training_log(args.log_dir, resume=bool(args.resume))

    if distributed:
        dist.barrier()

    vis_dir = args.vis_dir
    last_ckpt = os.path.join(args.ckpt_dir, LAST_CKPT_NAME)
    best_ckpt = os.path.join(args.ckpt_dir, BEST_CKPT_NAME)

    log_main(rank, "=" * 70)
    log_main(rank, "train_affordance_hp — human_prior supervision, MSE only")
    log_main(rank, "=" * 70)
    log_main(rank, f"  GPUs:         {gpu_ids}")
    log_main(rank, f"  Run dir:      {args.run_dir}")
    log_main(rank, f"  Supervision:  human_prior (soft), threshold={args.hp_threshold}")
    log_main(rank, f"  Loss:         L1(pred, human_prior)")
    log_main(rank, f"  Model:        PointNet2SegOnly (no FC head)")
    log_main(rank, f"  Input:        xyz ({AFFORDANCE_IN_CHANNELS}ch)")

    train_h5 = os.path.join(args.dataset_dir, "affordance_train.h5")
    val_h5 = os.path.join(args.dataset_dir, "affordance_val.h5")
    train_obj_ids, val_obj_ids = get_object_split(
        train_h5, val_h5, val_ratio=args.val_ratio, seed=args.split_seed,
    )

    train_aug = augment_config_from_args(args) if not args.no_augment else AugmentConfig(
        False, False, False, False, False,
    )
    no_aug = AugmentConfig(False, False, False, False, False)
    train_dataset, val_dataset = build_train_val_datasets_hp(
        args.dataset_dir,
        train_obj_ids,
        val_obj_ids,
        hp_threshold=args.hp_threshold,
        train_augment_config=train_aug,
    )
    train_vis_dataset, _ = build_train_val_datasets_hp(
        args.dataset_dir,
        train_obj_ids,
        val_obj_ids,
        hp_threshold=args.hp_threshold,
        train_augment_config=no_aug,
    )
    train_vis_seed = int(
        args.train_vis_seed if args.train_vis_seed is not None else args.split_seed + 7919,
    )
    train_vis_obj_ids = None
    if int(args.train_vis_max_objects) > 0:
        train_vis_obj_ids = pick_random_vis_object_ids(
            train_vis_dataset.sample_obj_ids,
            max_objects=int(args.train_vis_max_objects),
            seed=train_vis_seed,
        )

    hp_pos = (train_dataset.human_priors > args.hp_threshold).mean()
    log_main(rank, f"  Train/val:    {len(train_dataset)} / {len(val_dataset)} samples")
    log_main(rank, f"  HP contact:   {hp_pos * 100:.2f}% (train, >{args.hp_threshold})")

    train_bs = resolve_per_gpu_batch_size(
        len(train_dataset), args.batch_size, world_size, drop_last=True,
    )
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
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

    model = PointNet2SegOnly(
        in_channel=AFFORDANCE_IN_CHANNELS,
        head_norm=args.head_norm,
    ).to(device)
    if not args.resume:
        init_seg_head_bias_hp(model, hp_pos)
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = L1HumanPriorLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args)

    start_epoch = 1
    best_val_l1 = float("inf")
    best_val_score = 0.0
    epochs_without_improvement = 0
    history: list[dict] = []

    if args.resume:
        ckpt = try_resume(last_ckpt, model, optimizer, scheduler, device, rank)
        if ckpt:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_l1 = float(ckpt.get("best_val_l1", ckpt.get("best_val_mse", best_val_l1)))
            best_val_score = float(ckpt.get("best_val_score", 0.0))
            history = ckpt.get("history", [])

    if is_main_process(rank) and train_vis_obj_ids:
        with open(os.path.join(vis_dir, "train_vis_objects.json"), "w") as f:
            json.dump({"seed": train_vis_seed, "object_ids": train_vis_obj_ids}, f, indent=2)

    if distributed:
        dist.barrier()

    log_main(rank, f"\n{'Ep':>4} | {'trL1':>7} | {'vaL1':>7} | {'vaF1':>5} | {'score':>5} | {'LR':>8}")
    stopped_early = False
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t0 = time.time()
            t_l1, t_metrics = train_epoch_hp(
                model, train_loader, optimizer, criterion, device,
            )
            v_l1, v_metrics = (0.0, {})
            if is_main_process(rank):
                v_l1, v_metrics = eval_epoch_hp(model, val_loader, criterion, device)

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            improved = False
            if is_main_process(rank):
                val_score = checkpoint_score(v_metrics, f1_weight=args.ckpt_f1_weight)
                if val_score > best_val_score + 1e-8:
                    best_val_score = val_score
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
                if v_l1 < best_val_l1 - 1e-8:
                    best_val_l1 = v_l1

                log_main(
                    rank,
                    f"{epoch:>4} | {t_l1:>7.4f} | {v_l1:>7.4f} | "
                    f"{v_metrics.get('f1', 0):>4.0%} | {val_score:>4.0%} | {lr:>8.6f}  ({elapsed:.0f}s)",
                    flush=True,
                )

                history.append({
                    "epoch": epoch,
                    "train_l1": round(t_l1, 6),
                    "val_l1": round(v_l1, 6),
                    "val_f1": round(v_metrics.get("f1", 0), 4),
                    "val_ap": round(v_metrics.get("ap", 0), 4),
                    "val_score": round(val_score, 4),
                    "lr": round(lr, 7),
                })

                max_vis = int(args.val_vis_max_objects)
                vis_oids = None
                if max_vis > 0:
                    vis_oids = pick_random_vis_object_ids(
                        val_dataset.sample_obj_ids, max_objects=max_vis, seed=args.split_seed,
                    )
                save_hp_objects_grid(
                    model, val_dataset, val_dataset.sample_obj_ids, device,
                    os.path.join(vis_dir, f"val_epoch_{epoch:04d}.png"),
                    epoch, vis_object_ids=vis_oids, title_prefix="Val(HP-GT)",
                )
                if train_vis_obj_ids:
                    save_hp_objects_grid(
                        model, train_vis_dataset, train_vis_dataset.sample_obj_ids, device,
                        os.path.join(vis_dir, f"train_epoch_{epoch:04d}.png"),
                        epoch, vis_object_ids=train_vis_obj_ids, title_prefix="Train(HP-GT)",
                    )

                raw = unwrap_model(model)
                ckpt_common = {
                    "epoch": epoch,
                    "model_state_dict": raw.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_l1": best_val_l1,
                    "best_val_score": best_val_score,
                    "supervision": "human_prior",
                    "loss": "l1_only",
                    "history": history,
                    "args": vars(args),
                }
                if improved:
                    save_checkpoint(best_ckpt, {**ckpt_common, "val_score_at_best": val_score})
                save_checkpoint(last_ckpt, ckpt_common)

            should_stop = (
                not args.disable_early_stop
                and epochs_without_improvement >= args.patience
            )
            if broadcast_stop_flag(should_stop, rank, world_size, device):
                stopped_early = True
                log_main(rank, f"  Early stop @ epoch {epoch}")
                break

        if is_main_process(rank):
            with open(os.path.join(args.ckpt_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)
            log_main(rank, f"\nDone. best_val_l1={best_val_l1:.6f} score={best_val_score:.1%}")
    finally:
        if is_main_process(rank):
            close_training_log()
        cleanup_ddp()


def launch_training_hp(args) -> None:
    gpu_ids = parse_gpu_list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    os.environ["MASTER_PORT"] = str(args.master_port)
    if len(gpu_ids) == 1:
        train_worker_hp(0, 1, args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(train_worker_hp, args=(len(gpu_ids), args), nprocs=len(gpu_ids), join=True)


def main():
    args = parse_args()
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
        write_hp_run_manifest(run_dir, args)
    launch_training_hp(args)


if __name__ == "__main__":
    main()
