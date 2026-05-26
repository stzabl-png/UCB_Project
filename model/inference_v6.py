#!/usr/bin/env python3
"""
PointNet++ v6 affordance inference on prepared HDF5 point clouds.

Usage:
    python -m model.inference_v6 --checkpoint output/.../best_v6_model.pth --obj ycb_dex_04
    python -m model.inference_v6 --checkpoint ... --random 4 --save-dir output/inf_v6
    python -m model.inference_v6 --checkpoint ... --split val
    python -m model.inference_v6 --checkpoint ... --h5 .../affordance_all_soft.h5 --all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from math import ceil

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
sys.path.insert(0, PROJ)

try:
    import torch  # type: ignore
    from model.pointnet2_v6 import PointNet2AffordanceV6  # type: ignore

    _no_grad = torch.no_grad
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    PointNet2AffordanceV6 = object  # type: ignore

    def _no_grad():  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap


def _is_trusted_grasp(g: h5py.Group) -> bool:
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def _count_trusted_in_merged(merged_path: str) -> int:
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        grp = f["successful_grasps"]
        return sum(1 for key in grp.keys() if _is_trusted_grasp(grp[key]))


def _find_pose_count_table(dataset_dir: str) -> str | None:
    dataset_dir = os.path.abspath(dataset_dir)
    parent = os.path.dirname(dataset_dir)
    for path in (
        os.path.join(dataset_dir, "qc", "summary.csv"),
        os.path.join(parent, "qc", "summary.csv"),
        os.path.join(dataset_dir, "filter_split_pose_stats.csv"),
    ):
        if os.path.isfile(path):
            return path
    return None


def load_trusted_pose_counts(
    obj_ids: list[str],
    *,
    dataset_dir: str,
    merged_dir: str,
    stats_csv: str | None,
) -> dict[str, int]:
    """trusted successful grasps per object (prepare / merged GT口径)."""
    counts: dict[str, int] = {}
    table = stats_csv or _find_pose_count_table(dataset_dir)
    if table and os.path.isfile(table):
        with open(table, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = (row.get("obj_id") or "").strip()
                if not oid:
                    continue
                key = (
                    "n_grasps_trusted"
                    if "n_grasps_trusted" in row
                    else "n_trusted"
                )
                if key in row and row[key].strip() != "":
                    try:
                        counts[oid] = int(row[key])
                    except ValueError:
                        pass
    merged_dir = os.path.abspath(merged_dir)
    for oid in obj_ids:
        if oid in counts:
            continue
        mp = os.path.join(merged_dir, f"{oid}_robot_gt_merged.hdf5")
        counts[oid] = _count_trusted_in_merged(mp) if os.path.isfile(mp) else -1
    return counts


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


def _read_sample_from_h5(f: h5py.File, index: int) -> dict:
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


def load_sample(h5_path: str, index: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        return _read_sample_from_h5(f, index)


def load_samples_batch(h5_path: str, indices: list[int]) -> list[dict]:
    if not indices:
        return []
    with h5py.File(h5_path, "r") as f:
        return [_read_sample_from_h5(f, i) for i in indices]


def load_model(checkpoint: str, device: torch.device) -> tuple[PointNet2AffordanceV6, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = PointNet2AffordanceV6(in_channel=6).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@_no_grad()
def predict_heatmap_batch(
    model: PointNet2AffordanceV6,
    pts: np.ndarray,
    nrm: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """pts/nrm: (B, N, 3) → pred (B, N)."""
    xyz = torch.from_numpy(np.asarray(pts, dtype=np.float32)).to(device)
    feat = torch.from_numpy(
        np.concatenate([pts, nrm], axis=-1).astype(np.float32),
    ).to(device)
    return model(xyz, feat).cpu().numpy()


def _pearson_1d(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred - pred.mean()
    t = gt - gt.mean()
    return float((p * t).sum() / (np.sqrt((p * p).sum() * (t * t).sum()) + 1e-8))


def _emit_sample_result(
    sample: dict,
    pred: np.ndarray,
    thresh: float,
    *,
    pose_counts: dict[str, int],
    save_dir: str | None,
    npz_dir: str | None,
    png_dir: str | None,
    no_vis: bool,
) -> None:
    oid = sample["obj_id"]
    gt = sample["gt"]
    mae = float(np.abs(pred - gt).mean()) if gt is not None else float("nan")
    pearson = _pearson_1d(pred, gt) if gt is not None else float("nan")
    n_trusted = pose_counts.get(oid) if pose_counts else None
    pose_str = (
        f"  trusted={n_trusted}"
        if n_trusted is not None and n_trusted >= 0
        else ""
    )
    print(
        f"  {oid}: pred max={pred.max():.3f} mean={pred.mean():.4f}  "
        f"MAE={mae:.4f}  r={pearson:.4f}{pose_str}",
    )
    if not save_dir or npz_dir is None:
        return
    npz_kw = dict(
        points=sample["points"],
        normals=sample["normals"],
        pred=pred,
        gt=gt,
        obj_id=oid,
        threshold=thresh,
    )
    if n_trusted is not None and n_trusted >= 0:
        npz_kw["n_trusted_grasps"] = np.int32(n_trusted)
    npz_path = os.path.join(npz_dir, f"{oid}.npz")
    np.savez(npz_path, **npz_kw)
    print(f"    → npz/{oid}.npz")
    if not no_vis and png_dir is not None:
        png_path = os.path.join(png_dir, f"{oid}.png")
        save_vis_png(
            png_path, sample["points"], gt, pred, oid, thresh,
            n_trusted=n_trusted,
        )
        print(f"    → png/{oid}.png")


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


def save_vis_png(
    path: str,
    pts: np.ndarray,
    gt,
    pred: np.ndarray,
    obj_id: str,
    threshold: float,
    *,
    n_trusted: int | None = None,
):
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
    if n_trusted is not None and n_trusted >= 0:
        suptitle = f"{obj_id}  |  trusted successful grasps = {n_trusted}"
    elif n_trusted == -1:
        suptitle = f"{obj_id}  |  trusted successful grasps = ?"
    else:
        suptitle = obj_id
    fig.suptitle(suptitle, fontsize=10, color="#eee", y=1.04)
    for col, (title, vals) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), col + 1, projection="3d", facecolor="#1a1a2e")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=cmap(vals)[:, :3], s=2, alpha=0.85, linewidths=0,
        )
        _set_equal_3d_limits(ax, pts)
        _style_3d_background_axes(ax)
        ax.set_title(title, fontsize=8, color="#ddd")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def compose_png_grid(
    image_paths: list[str],
    output_path: str,
    *,
    cols: int,
    max_cell_width: int = 512,
    bg: tuple[int, int, int] = (26, 26, 46),
) -> None:
    """Tile per-object inference PNGs into one montage."""
    from PIL import Image

    if not image_paths:
        raise ValueError("no PNG paths for grid")

    cells: list[Image.Image] = []
    cell_h = 0
    cell_w = max_cell_width
    for path in image_paths:
        im = Image.open(path).convert("RGB")
        scale = max_cell_width / max(im.width, 1)
        new_w = max(1, int(im.width * scale))
        new_h = max(1, int(im.height * scale))
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cell_w = max(cell_w, new_w)
        cell_h = max(cell_h, new_h)
        cells.append(im)

    n = len(cells)
    rows_n = ceil(n / cols)
    canvas_w = cols * cell_w
    canvas_h = rows_n * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    for i, im in enumerate(cells):
        row, col = divmod(i, cols)
        x = col * cell_w + (cell_w - im.width) // 2
        y = row * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, optimize=True)


def save_inference_montage(
    save_dir: str,
    obj_ids: list[str],
    *,
    cols: int,
    max_cell_width: int,
    max_per_page: int,
) -> list[str]:
    """Build all_objects_grid.png (and pages if max_per_page > 0)."""
    png_dir = os.path.join(save_dir, "png")
    paths: list[str] = []
    for oid in obj_ids:
        p = os.path.join(png_dir, f"{oid}.png")
        if os.path.isfile(p):
            paths.append(p)
    if not paths:
        # fallback: every png in folder, sorted by name
        paths = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    if not paths:
        print(f"  ⚠️ No PNGs under {png_dir}; skip grid")
        return []

    per_page = max_per_page if max_per_page > 0 else len(paths)
    n_pages = ceil(len(paths) / per_page)
    out_paths: list[str] = []
    for pi in range(n_pages):
        chunk = paths[pi * per_page : (pi + 1) * per_page]
        if n_pages == 1:
            name = "all_objects_grid.png"
        else:
            name = f"all_objects_grid_p{pi + 1:02d}.png"
        out = os.path.join(save_dir, name)
        compose_png_grid(
            chunk, out, cols=cols, max_cell_width=max_cell_width,
        )
        out_paths.append(out)
        print(f"  Grid montage: {out}  ({len(chunk)} panels)")
    return out_paths


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
    # Default: every object in the loaded HDF5 (e.g. all val when --split val).
    return list(range(len(obj_ids)))


def parse_args():
    p = argparse.ArgumentParser(description="PointNet++ v6 affordance inference")
    p.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help="best_v6_model.pth or checkpoint_ep*.pth (required unless --compose-grid-only)",
    )
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
    p.add_argument(
        "--all",
        action="store_true",
        help="Use affordance_all_soft.h5 (parent dataset-dir); default is all objects in chosen split HDF5",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=None, help="Binarize pred (default: run_info)")
    p.add_argument(
        "--save-dir",
        default=None,
        help="Root dir; writes npz/{obj}.npz and png/{obj}.png",
    )
    p.add_argument("--no-vis", action="store_true", help="Skip PNG (only .npz if --save-dir set)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument(
        "--merged-dir",
        default=DEFAULT_MERGED_DIR,
        help="Count trusted grasps from merged GT if not in qc/summary.csv",
    )
    p.add_argument(
        "--pose-stats-csv",
        default=None,
        help="Override pose count table (qc/summary.csv or filter_split_pose_stats.csv)",
    )
    p.add_argument(
        "--no-pose-count",
        action="store_true",
        help="Do not show trusted grasp count on PNG / npz",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Objects per GPU forward pass (default: 64)",
    )
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="Do not stitch png/ into all_objects_grid.png after inference",
    )
    p.add_argument("--grid-cols", type=int, default=4, help="Montage columns")
    p.add_argument(
        "--grid-max-cell-width",
        type=int,
        default=512,
        help="Resize each object PNG to this width in the montage",
    )
    p.add_argument(
        "--grid-max-per-page",
        type=int,
        default=0,
        help="Split montage into pages (0 = single image)",
    )
    p.add_argument(
        "--compose-grid-only",
        action="store_true",
        help="Only build montage from existing --save-dir/png (no model run)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.compose_grid_only:
        if not args.save_dir:
            raise SystemExit("--compose-grid-only requires --save-dir")
        save_dir = os.path.abspath(args.save_dir)
        obj_ids: list[str] = []
        if args.obj:
            obj_ids = list(args.obj)
        save_inference_montage(
            save_dir,
            obj_ids,
            cols=args.grid_cols,
            max_cell_width=args.grid_max_cell_width,
            max_per_page=args.grid_max_per_page,
        )
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required (unless --compose-grid-only)")

    if torch is None:
        raise SystemExit(
            "torch not available in this environment; run inference in the same env as training "
            "(or use --compose-grid-only which does not require torch)"
        )

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
    inferred_obj_ids = [meta["obj_ids"][i] for i in indices]

    model, ckpt = load_model(args.checkpoint, device)
    thresh = args.threshold if args.threshold is not None else default_threshold(
        args.checkpoint, args.dataset_dir,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  epoch={ckpt.get('epoch', '?')}  val_pearson={ckpt.get('val_pearson', '?')}")
    print(f"HDF5: {h5_path}  ({meta['n']} objects, soft={meta['has_soft']})")
    batch_size = max(1, int(args.batch_size))
    print(f"Device: {device}  threshold={thresh:.3f}  batch_size={batch_size}")
    print(f"Inferring {len(indices)} object(s)...")

    pose_counts: dict[str, int] = {}
    if not args.no_pose_count:
        pose_counts = load_trusted_pose_counts(
            meta["obj_ids"],
            dataset_dir=args.dataset_dir,
            merged_dir=args.merged_dir,
            stats_csv=args.pose_stats_csv,
        )
        table = args.pose_stats_csv or _find_pose_count_table(args.dataset_dir)
        print(
            f"Pose counts: {table or args.merged_dir}",
        )

    save_dir = args.save_dir
    npz_dir = png_dir = None
    if save_dir:
        save_dir = os.path.abspath(save_dir)
        npz_dir = os.path.join(save_dir, "npz")
        png_dir = os.path.join(save_dir, "png")
        os.makedirs(npz_dir, exist_ok=True)
        if not args.no_vis:
            os.makedirs(png_dir, exist_ok=True)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = load_samples_batch(h5_path, batch_indices)
        pts = np.stack([s["points"] for s in samples], axis=0)
        nrm = np.stack([s["normals"] for s in samples], axis=0)
        preds = predict_heatmap_batch(model, pts, nrm, device)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]
        bi = start // batch_size + 1
        nb = (len(indices) + batch_size - 1) // batch_size
        print(f"  [batch {bi}/{nb}] forward {len(samples)} object(s)")
        for i, sample in enumerate(samples):
            _emit_sample_result(
                sample,
                preds[i],
                thresh,
                pose_counts=pose_counts,
                save_dir=save_dir,
                npz_dir=npz_dir,
                png_dir=png_dir,
                no_vis=args.no_vis,
            )

    if (
        save_dir
        and not args.no_vis
        and not args.no_grid
    ):
        save_inference_montage(
            save_dir,
            inferred_obj_ids,
            cols=args.grid_cols,
            max_cell_width=args.grid_max_cell_width,
            max_per_page=args.grid_max_per_page,
        )


if __name__ == "__main__":
    main()
