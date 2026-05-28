#!/usr/bin/env python3
"""Train PDM on merged successful grasp data.

Example:
  python3 -m model.pdm.train \
      --merged-dir output/grasp_collect_no_rot/merged \
      --affordance-h5 output/affordance_no_rot_executed/min20/affordance_all_soft.h5 \
      --save-dir output/checkpoints/pdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.pdm.build_condition_cache import DEFAULT_OUTPUT as DEFAULT_CONDITION_H5
from model.pdm.dataset import DEFAULT_MERGED_DIR, PDMMergedDataset, compute_pose_stats
from model.pdm.model import PDM, PDMConfig


def _to_device(
    batch: dict,
    device: torch.device,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    *,
    use_yaw_condition: bool,
):
    points = batch["points"].to(device=device, dtype=torch.float32)
    pose = batch["pose"].to(device=device, dtype=torch.float32)
    pose_norm = (pose - pose_mean) / pose_std
    yaw = None
    if use_yaw_condition:
        yaw = batch["yaw"].to(device=device, dtype=torch.float32)
    return points, pose_norm, yaw


def _verify_condition_h5(path: str) -> None:
    import warnings

    import h5py

    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "PDM training requires a v6-prediction condition cache. Build it with:\n"
            "  python3 -m model.pdm.build_condition_cache \\\n"
            "    --merged-dir output/grasp_collect_no_rot/merged \\\n"
            "    --output output/pdm/cache/conditions_4096_v6pred.h5"
        )
    with h5py.File(path, "r") as f:
        src = str(f["metadata"].attrs.get("affordance_source", "") or "")
    if src and src != "v6_prediction":
        warnings.warn(
            f"condition-h5 affordance_source={src!r} (expected 'v6_prediction'). "
            "Rebuild cache with build_condition_cache before training.",
            stacklevel=2,
        )
    elif not src:
        warnings.warn(
            "condition-h5 has no affordance_source metadata (old GT cache?). "
            "Rebuild with `python -m model.pdm.build_condition_cache`.",
            stacklevel=2,
        )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    if not args.condition_h5:
        raise RuntimeError("Pass --condition-h5 (v6-prediction cache from build_condition_cache).")
    _verify_condition_h5(os.path.abspath(args.condition_h5))

    dataset = PDMMergedDataset(
        merged_dir=args.merged_dir,
        condition_h5=args.condition_h5,
        affordance_h5=args.affordance_h5,
        n_points=args.n_points,
        require_trusted_tips=not args.allow_untrusted_tips,
        max_cmd_candidate_dist=args.max_cmd_candidate_dist,
        cache_conditions=not args.no_cache_conditions,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No PDM samples found under {args.merged_dir}")

    stats = compute_pose_stats(dataset)
    pose_mean = stats["pose_mean"].to(device)
    pose_std = stats["pose_std"].to(device)

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.num_workers // 2, 0),
        pin_memory=device.type == "cuda",
    )

    config = PDMConfig(
        point_channels=7,
        point_feat_dim=args.point_feat_dim,
        hidden_dim=args.hidden_dim,
        T=args.T,
        use_yaw_condition=args.use_yaw_condition,
    )
    model = PDM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    print("=" * 72)
    print("PDM training")
    print(f"  samples: {len(dataset)}  train: {train_size}  val: {val_size}")
    print(f"  skipped: {dataset.skipped}")
    print(f"  device:  {device}")
    print(f"  save:    {args.save_dir}")
    print("=" * 72)

    best_val = float("inf")
    history = []
    stats_cpu = {k: v.cpu() for k, v in stats.items()}
    torch.save(stats_cpu, os.path.join(args.save_dir, "pose_stats.pt"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            points, pose_norm, yaw = _to_device(
                batch,
                device,
                pose_mean,
                pose_std,
                use_yaw_condition=args.use_yaw_condition,
            )
            loss, _ = model.training_loss(pose_norm, points, yaw=yaw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                points, pose_norm, yaw = _to_device(
                    batch,
                    device,
                    pose_mean,
                    pose_std,
                    use_yaw_condition=args.use_yaw_condition,
                )
                loss, _ = model.training_loss(pose_norm, points, yaw=yaw)
                val_losses.append(float(loss.item()))

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch {epoch:04d}  train={train_loss:.6f}  "
                f"val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            model.save(
                os.path.join(args.save_dir, "best_model.pth"),
                epoch=epoch,
                best_loss=best_val,
                pose_stats=stats_cpu,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            model.save(
                os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"),
                epoch=epoch,
                best_loss=best_val,
                pose_stats=stats_cpu,
            )

    model.save(
        os.path.join(args.save_dir, "final_model.pth"),
        epoch=args.epochs,
        best_loss=best_val,
        pose_stats=stats_cpu,
    )
    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. best val={best_val:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Pose Diffusion Model (PDM)")
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument(
        "--condition-h5",
        default=DEFAULT_CONDITION_H5,
        help="PDM condition cache with v6-predicted affordance (build_condition_cache)",
    )
    parser.add_argument(
        "--affordance-h5",
        default=None,
        help="Deprecated for training; only used if an object is missing from --condition-h5",
    )
    parser.add_argument("--save-dir", default=os.path.join(PROJ, "output", "pdm", "checkpoints"))
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--point-feat-dim", type=int, default=512)
    parser.add_argument("--use-yaw-condition", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-cmd-candidate-dist", type=float, default=0.5)
    parser.add_argument("--allow-untrusted-tips", action="store_true")
    parser.add_argument("--no-cache-conditions", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
