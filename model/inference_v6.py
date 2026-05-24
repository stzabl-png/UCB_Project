#!/usr/bin/env python3
"""
PointNet++ v6 affordance inference on prepared HDF5 point clouds.

Usage:
    python -m model.inference_v6 --checkpoint output/.../best_v6_model.pth --obj ycb_dex_04
    python -m model.inference_v6 --checkpoint ... --random 4 --save-dir output/inf_v6
    python -m model.inference_v6 --checkpoint ... --all --dataset-dir output/affordance_no_rot_executed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.pointnet2_v6 import PointNet2AffordanceV6


def _decode_obj_ids(raw) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def resolve_h5(dataset_dir: str, split: str) -> str:
    dataset_dir = os.path.abspath(dataset_dir)
    if split == "all":
        for name in ("affordance_all_soft.h5", "affordance_all.h5"):
            p = os.path.join(dataset_dir, name)
            if os.path.isfile(p):
                return p
    elif split == "train":
        for name in ("affordance_train_soft.h5", "affordance_train.h5"):
            p = os.path.join(dataset_dir, name)
            if os.path.isfile(p):
                return p
    elif split == "val":
        for name in ("affordance_val_soft.h5", "affordance_val.h5"):
            p = os.path.join(dataset_dir, name)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"No HDF5 for split={split!r} under {dataset_dir}")


def resolve_h5_containing(dataset_dir: str, obj_ids: list[str]) -> str:
    """Pick first HDF5 (all → train → val) that contains every requested object."""
    for split in ("all", "train", "val"):
        try:
            path = resolve_h5(dataset_dir, split)
        except FileNotFoundError:
            continue
        ids = set(load_h5_index(path)["obj_ids"])
        if all(oid in ids for oid in obj_ids):
            return path
    raise KeyError(f"objects not found in any split under {dataset_dir}: {obj_ids}")


def load_h5_index(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        obj_ids = _decode_obj_ids(f["data/obj_ids"][:])
        has_soft = "data/soft_labels" in f
        return {
            "path": h5_path,
            "obj_ids": obj_ids,
            "has_soft": has_soft,
            "n": len(obj_ids),
        }


def load_sample(h5_path: str, index: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        pts = f["data/points"][index].astype(np.float32)
        nrm = f["data/normals"][index].astype(np.float32)
        lbl = None
        if "data/soft_labels" in f:
            lbl = f["data/soft_labels"][index].astype(np.float32)
        elif "data/labels" in f:
            lbl = f["data/labels"][index].astype(np.float32)
        oid = f["data/obj_ids"][index]
        oid = oid.decode() if isinstance(oid, bytes) else str(oid)
    return {"obj_id": oid, "points": pts, "normals": nrm, "gt": lbl}


def load_model(checkpoint: str, device: torch.device) -> tuple[PointNet2AffordanceV6, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = PointNet2AffordanceV6(in_channel=6).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_heatmap(model, pts: np.ndarray, nrm: np.ndarray, device: torch.device) -> np.ndarray:
    xyz = torch.from_numpy(pts).unsqueeze(0).to(device)
    feat = torch.from_numpy(np.concatenate([pts, nrm], axis=-1)).unsqueeze(0).to(device)
    return model(xyz, feat).squeeze(0).cpu().numpy()


def default_threshold(checkpoint: str, dataset_dir: str) -> float:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    for meta_path in (
        os.path.join(ckpt_dir, "run_info_v6.json"),
        os.path.join(dataset_dir, "soft_gt_export_meta.json"),
    ):
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "best_threshold" in meta:
                return float(meta["best_threshold"])
    return 0.3


def _set_equal_3d_limits(ax, pts: np.ndarray) -> None:
    """Equal-aspect limits around the point cloud (meters)."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float((hi - lo).max()) * 0.55 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _style_3d_background_axes(ax) -> None:
    """Matplotlib 3D box + ticks only (no triad on the object)."""
    ax.set_xlabel("X (m)", color="#aaa", fontsize=7, labelpad=2)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7, labelpad=2)
    ax.set_zlabel("Z (m)", color="#aaa", fontsize=7, labelpad=2)
    ax.tick_params(colors="#888", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color="#444", alpha=0.35, linewidth=0.5)


def save_vis_png(path: str, pts: np.ndarray, gt, pred: np.ndarray, obj_id: str, threshold: float):
    fig = plt.figure(figsize=(14, 4), facecolor="#1a1a2e")
    cmap = plt.cm.jet
    panels = [(f"Pred (max={pred.max():.2f})", pred)]
    if gt is not None:
        panels.insert(0, (f"GT (max={gt.max():.2f})", gt))
        panels.append(
            (f"|Error| MAE={np.abs(pred - gt).mean():.4f}", np.abs(pred - gt)),
        )
    panels.append(
        (f"Binary τ={threshold}", (pred > threshold).astype(np.float32)),
    )
    for col, (title, vals) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), col + 1, projection="3d", facecolor="#1a1a2e")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=cmap(vals)[:, :3], s=2, alpha=0.85, linewidths=0,
        )
        _set_equal_3d_limits(ax, pts)
        _style_3d_background_axes(ax)
        ax.set_title(f"{obj_id}\n{title}", fontsize=8, color="#ddd")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def select_indices(
    obj_ids: list[str],
    *,
    objects: list[str] | None,
    random_n: int | None,
    use_all: bool,
    seed: int,
) -> list[int]:
    id_to_idx = {oid: i for i, oid in enumerate(obj_ids)}
    if objects:
        out = []
        for oid in objects:
            if oid not in id_to_idx:
                raise KeyError(f"object {oid!r} not in HDF5 (have {len(obj_ids)} objects)")
            out.append(id_to_idx[oid])
        return out
    if use_all:
        return list(range(len(obj_ids)))
    if random_n is not None:
        rng = np.random.default_rng(seed)
        n = min(random_n, len(obj_ids))
        return sorted(rng.choice(len(obj_ids), size=n, replace=False).tolist())
    raise ValueError("specify --obj, --random N, or --all")


def parse_args():
    p = argparse.ArgumentParser(description="PointNet++ v6 affordance inference")
    p.add_argument("--checkpoint", required=True, help="best_v6_model.pth or checkpoint_ep*.pth")
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJ, "output", "affordance_no_rot_executed"),
    )
    p.add_argument(
        "--h5",
        default=None,
        help="Override HDF5 path (default: affordance_all_soft.h5 or train+val soft)",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "val", "auto"),
        default="auto",
        help="Which split file to use when --h5 not set (auto: all, else val soft)",
    )
    p.add_argument("--obj", action="append", default=None, help="Object id (repeatable)")
    p.add_argument("--random", type=int, default=None, metavar="N", help="Random N objects")
    p.add_argument("--all", action="store_true", help="All objects in HDF5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=None, help="Binarize pred (default: run_info)")
    p.add_argument(
        "--save-dir",
        default=None,
        help="Root dir; writes npz/{obj}.npz and png/{obj}.png",
    )
    p.add_argument("--no-vis", action="store_true", help="Skip PNG (only .npz if --save-dir set)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.h5:
        h5_path = os.path.abspath(args.h5)
    elif args.obj:
        h5_path = resolve_h5_containing(args.dataset_dir, args.obj)
    elif args.all:
        h5_path = resolve_h5(args.dataset_dir, "all")
    elif args.random is not None:
        h5_path = resolve_h5(
            args.dataset_dir,
            "all" if args.split == "auto" else args.split,
        )
    else:
        split = args.split if args.split != "auto" else "val"
        h5_path = resolve_h5(args.dataset_dir, split)

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    meta = load_h5_index(h5_path)
    indices = select_indices(
        meta["obj_ids"],
        objects=args.obj,
        random_n=args.random,
        use_all=args.all,
        seed=args.seed,
    )

    model, ckpt = load_model(args.checkpoint, device)
    thresh = args.threshold if args.threshold is not None else default_threshold(
        args.checkpoint, args.dataset_dir,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  epoch={ckpt.get('epoch', '?')}  val_pearson={ckpt.get('val_pearson', '?')}")
    print(f"HDF5: {h5_path}  ({meta['n']} objects, soft={meta['has_soft']})")
    print(f"Device: {device}  threshold={thresh:.3f}")
    print(f"Inferring {len(indices)} object(s)...")

    save_dir = args.save_dir
    npz_dir = png_dir = None
    if save_dir:
        save_dir = os.path.abspath(save_dir)
        npz_dir = os.path.join(save_dir, "npz")
        png_dir = os.path.join(save_dir, "png")
        os.makedirs(npz_dir, exist_ok=True)
        if not args.no_vis:
            os.makedirs(png_dir, exist_ok=True)

    for idx in indices:
        sample = load_sample(h5_path, idx)
        oid = sample["obj_id"]
        pred = predict_heatmap(model, sample["points"], sample["normals"], device)
        gt = sample["gt"]
        mae = float(np.abs(pred - gt).mean()) if gt is not None else float("nan")
        pearson = float("nan")
        if gt is not None:
            p = pred - pred.mean()
            t = gt - gt.mean()
            pearson = float((p * t).sum() / (np.sqrt((p * p).sum() * (t * t).sum()) + 1e-8))

        print(
            f"  {oid}: pred max={pred.max():.3f} mean={pred.mean():.4f}  "
            f"MAE={mae:.4f}  r={pearson:.4f}",
        )

        if save_dir:
            npz_path = os.path.join(npz_dir, f"{oid}.npz")
            np.savez(
                npz_path,
                points=sample["points"],
                normals=sample["normals"],
                pred=pred,
                gt=gt,
                obj_id=oid,
                threshold=thresh,
            )
            print(f"    → npz/{oid}.npz")
            if not args.no_vis:
                png_path = os.path.join(png_dir, f"{oid}.png")
                save_vis_png(png_path, sample["points"], gt, pred, oid, thresh)
                print(f"    → png/{oid}.png")


if __name__ == "__main__":
    main()
