#!/usr/bin/env python3
"""Batch visualization for PDM candidate HDF5 files.

Each object gets one PNG with multiple candidate grippers overlaid on the same
object point cloud. This is intentionally lightweight and uses the cached PDM
condition points when available.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.pdm.dataset import (
    DEFAULT_ROTATED_MESH_DIR,
    PDMConditionStore,
    sample_mesh_condition,
    _resample_rows,
)
from model.pdm.pose_codec import TCP_OFFSET


def load_candidates(path: str, top: int | None = None) -> list[dict]:
    cands = []
    with h5py.File(path, "r") as f:
        n = int(f["candidates"].attrs["n_candidates"])
        if top is not None:
            n = min(n, top)
        for i in range(n):
            g = f[f"candidates/candidate_{i}"]
            rot = g["rotation"][:].astype(np.float64)
            pos = g["position"][:].astype(np.float64)
            cands.append(
                {
                    "idx": i,
                    "name": str(g.attrs.get("name", f"pdm_{i:03d}")),
                    "score": float(g.attrs.get("score", 0.0)),
                    "position": pos,
                    "rotation": rot,
                    "finger_dir": rot[:, 0],
                    "approach_dir": rot[:, 2],
                    "width": float(g.attrs.get("gripper_width", 0.06)),
                }
            )
    return cands


def obj_id_from_hdf5(path: str) -> str:
    with h5py.File(path, "r") as f:
        if "metadata" in f and "obj_id" in f["metadata"].attrs:
            return str(f["metadata"].attrs["obj_id"])
    return os.path.basename(path).replace("_grasp.hdf5", "")


def list_candidate_files(candidates_dir: str) -> list[str]:
    if not os.path.isdir(candidates_dir):
        return []
    return sorted(
        os.path.join(candidates_dir, name)
        for name in os.listdir(candidates_dir)
        if name.endswith("_grasp.hdf5")
    )


def resolve_files(args: argparse.Namespace) -> list[str]:
    if args.hdf5:
        return args.hdf5
    files = list_candidate_files(args.candidates_dir)
    if args.obj:
        want = set(args.obj)
        files = [p for p in files if obj_id_from_hdf5(p) in want]
    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(files)
        files = files[: args.random]
    elif not args.all and not args.obj:
        raise ValueError("provide --hdf5, --obj, --all, or --random N")
    return files


def load_object_points(
    obj_id: str,
    *,
    condition_h5: str | None,
    n_points: int,
    mesh_root: str,
    use_condition_cache: bool = True,
) -> np.ndarray:
    """Background points for overlays (metric rotated_mesh frame)."""
    if use_condition_cache and condition_h5:
        store = PDMConditionStore(condition_h5)
        if store.has(obj_id):
            cond = store.load(obj_id)
            return _resample_rows(cond.points, n_points)[:, :3]
    return sample_mesh_condition(obj_id, n_points, mesh_root=mesh_root).points[:, :3]


def load_overlay_background(
    obj_id: str,
    hdf5_path: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Points (+ optional affordance) aligned with glb_to_pdm_grasp / batch_pdm inference."""
    use_cache = bool(getattr(args, "use_condition_cache", True))
    if "_pool_grasp" in os.path.basename(hdf5_path):
        use_cache = False
    if use_cache and args.condition_h5:
        store = PDMConditionStore(args.condition_h5)
        if store.has(obj_id):
            cond = store.load(obj_id)
            pts = _resample_rows(cond.points, args.bg_points)
            aff = pts[:, 6] if pts.shape[1] > 6 else None
            return pts[:, :3], aff
    cond = sample_mesh_condition(obj_id, args.bg_points, mesh_root=args.mesh_root)
    pts = cond.points
    aff = pts[:, 6] if pts.shape[1] > 6 else None
    return pts[:, :3], aff


def _axis_equal(ax, pts: np.ndarray) -> None:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) * 0.75, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(lo[2] - radius * 0.15, lo[2] + radius * 1.6)
    ax.set_xlabel("X", color="#aaa", fontsize=8)
    ax.set_ylabel("Y", color="#aaa", fontsize=8)
    ax.set_zlabel("Z", color="#aaa", fontsize=8)
    ax.tick_params(colors="#666", labelsize=6)


def draw_gripper(ax, cand: dict, color, width_scale: float = 1.0) -> None:
    pos = cand["position"]
    app = cand["approach_dir"]
    app = app / (np.linalg.norm(app) + 1e-8)
    fing = cand["finger_dir"]
    fing = fing / (np.linalg.norm(fing) + 1e-8)
    width = cand["width"] * width_scale
    hw = width / 2.0
    finger_depth = max(width * 0.55, 0.025)

    tip_l = pos - hw * fing
    tip_r = pos + hw * fing
    tip_l_back = tip_l - app * finger_depth
    tip_r_back = tip_r - app * finger_depth
    wrist = pos - app * TCP_OFFSET

    ax.scatter(*pos, c=[color], s=22, marker="o", depthshade=False)
    ax.plot(
        [tip_l[0], tip_r[0]],
        [tip_l[1], tip_r[1]],
        [tip_l[2], tip_r[2]],
        color=color,
        linewidth=1.8,
        alpha=0.95,
    )
    for a, b in ((tip_l, tip_l_back), (tip_r, tip_r_back), (tip_l_back, tip_r_back)):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.1, alpha=0.8)
    ax.plot(
        [wrist[0], pos[0]],
        [wrist[1], pos[1]],
        [wrist[2], pos[2]],
        color=color,
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
    )
    ax.quiver(*pos, *app, length=0.025, color=color, linewidth=1.2, arrow_length_ratio=0.35)


def _scatter_object_points(
    ax,
    pts: np.ndarray,
    affordance: np.ndarray | None = None,
) -> None:
    """Background point cloud for overlays (same frame as PDM candidates)."""
    if affordance is not None and affordance.shape[0] == pts.shape[0]:
        aff = np.asarray(affordance, dtype=np.float64).reshape(-1)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=aff,
            cmap="hot",
            vmin=0.0,
            vmax=max(float(aff.max()), 0.05),
            s=3.0,
            alpha=0.55,
            linewidths=0,
            depthshade=False,
        )
    else:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c="#5ab4d4",
            s=2.0,
            alpha=0.35,
            linewidths=0,
            depthshade=False,
        )


def save_candidate_overlay(
    hdf5_path: str,
    points: np.ndarray,
    out_path: str,
    *,
    top: int = 20,
    affordance: np.ndarray | None = None,
    bg_points: int = 8000,
    width_scale: float = 1.0,
    elev: float = 22.0,
    azim: float = 132.0,
    dpi: int = 140,
    title_suffix: str = "",
) -> str:
    """Overlay PDM grippers on a precomputed object point cloud (e.g. from glb_to_pdm_grasp)."""

    obj_id = obj_id_from_hdf5(hdf5_path)
    cands = load_candidates(hdf5_path, top=top)
    if not cands:
        raise RuntimeError(f"no candidates in {hdf5_path}")
    pts_full = np.asarray(points, dtype=np.float64)
    aff_full = None if affordance is None else np.asarray(affordance, dtype=np.float64).reshape(-1)
    if aff_full is not None and aff_full.shape[0] != pts_full.shape[0]:
        aff_full = None
    if pts_full.shape[0] > bg_points:
        replace = pts_full.shape[0] < bg_points
        idx = np.random.choice(pts_full.shape[0], bg_points, replace=replace)
        pts = pts_full[idx]
        if aff_full is not None:
            aff_full = aff_full[idx]
    else:
        pts = pts_full
    affordance = aff_full

    fig = plt.figure(figsize=(8, 8), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    _scatter_object_points(ax, pts, affordance=affordance)
    cmap = plt.get_cmap("hsv")
    for i, cand in enumerate(cands):
        draw_gripper(ax, cand, cmap(i / max(len(cands), 1))[:3], width_scale=width_scale)
    _axis_equal(ax, pts)
    ax.view_init(elev=elev, azim=azim)
    yaw_note = f"  {title_suffix}" if title_suffix else ""
    ax.set_title(
        f"{obj_id}  PDM candidates: {len(cands)}{yaw_note}\n"
        f"source={os.path.relpath(hdf5_path, PROJ)}",
        color="#ddd",
        fontsize=10,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out_path


def save_one(path: str, args: argparse.Namespace) -> str:
    obj_id = obj_id_from_hdf5(path)
    cands = load_candidates(path, top=args.top)
    if not cands:
        raise RuntimeError(f"no candidates in {path}")
    pts, affordance = load_overlay_background(obj_id, path, args)
    if len(pts) > args.bg_points:
        pts = _resample_rows(pts, args.bg_points)
        if affordance is not None and affordance.shape[0] > args.bg_points:
            idx = np.linspace(0, affordance.shape[0] - 1, args.bg_points).astype(int)
            affordance = affordance[idx]

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{obj_id}_pdm_overlay_top{len(cands)}.png")
    return save_candidate_overlay(
        path,
        pts,
        out_path,
        top=args.top,
        affordance=affordance,
        bg_points=args.bg_points,
        width_scale=args.width_scale,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )


def render_one_image(path: str, args: argparse.Namespace):
    """Render one object overlay and return (obj_id, figure)."""

    obj_id = obj_id_from_hdf5(path)
    cands = load_candidates(path, top=args.top)
    if not cands:
        raise RuntimeError(f"no candidates in {path}")
    pts, affordance = load_overlay_background(obj_id, path, args)
    if len(pts) > args.bg_points:
        pts = _resample_rows(pts, args.bg_points)
        if affordance is not None and affordance.shape[0] > args.bg_points:
            idx = np.linspace(0, affordance.shape[0] - 1, args.bg_points).astype(int)
            affordance = affordance[idx]

    fig = plt.figure(figsize=(8, 8), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    _scatter_object_points(ax, pts, affordance=affordance)
    cmap = plt.get_cmap("hsv")
    for i, cand in enumerate(cands):
        draw_gripper(ax, cand, cmap(i / max(len(cands), 1))[:3], width_scale=args.width_scale)
    _axis_equal(ax, pts)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(
        f"{obj_id}  PDM candidates: {len(cands)}\n"
        f"source={os.path.relpath(path, PROJ)}",
        color="#ddd",
        fontsize=10,
    )
    return obj_id, fig


def save_fig_to_path(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out_path


def make_overview(image_paths: list[str], out_path: str, cols: int = 5) -> None:
    if not image_paths:
        return
    from PIL import Image, ImageDraw

    tiles = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        tiles.append((path, img))
    if not tiles:
        return

    # Normalize tile sizes for a compact grid.
    target_w = min(img.size[0] for _, img in tiles)
    target_h = min(img.size[1] for _, img in tiles)
    resized = []
    for path, img in tiles:
        tile = img.resize((target_w, target_h))
        label = os.path.basename(path).split("_pdm_overlay_")[0]
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, min(target_w, 260), 24), fill=(26, 26, 46))
        draw.text((6, 5), label, fill=(235, 235, 235))
        resized.append(tile)

    cols = max(1, int(cols))
    rows = (len(resized) + cols - 1) // cols
    canvas = Image.new("RGB", (target_w * cols, target_h * rows), (26, 26, 46))
    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * target_w, r * target_h))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def make_overview_direct(files: list[str], args: argparse.Namespace, out_path: str) -> None:
    """Render overview directly without writing per-object PNGs."""

    if not files:
        return
    from PIL import Image, ImageDraw

    tiles = []
    tmp_dir = os.path.join(args.outdir, ".overview_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        for path in files:
            try:
                obj_id, fig = render_one_image(path, args)
                tmp_path = os.path.join(tmp_dir, f"{obj_id}.png")
                fig.savefig(tmp_path, dpi=args.dpi, bbox_inches="tight", facecolor="#1a1a2e")
                plt.close(fig)
                img = Image.open(tmp_path).convert("RGB")
                tiles.append((obj_id, img))
            except Exception as exc:
                print(f"  overview skip {path}: {type(exc).__name__}: {exc}")
        if not tiles:
            return
        target_w = min(img.size[0] for _, img in tiles)
        target_h = min(img.size[1] for _, img in tiles)
        resized = []
        for obj_id, img in tiles:
            tile = img.resize((target_w, target_h))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, min(target_w, 260), 24), fill=(26, 26, 46))
            draw.text((6, 5), obj_id, fill=(235, 235, 235))
            resized.append(tile)
        cols = max(1, int(args.overview_cols))
        rows = (len(resized) + cols - 1) // cols
        canvas = Image.new("RGB", (target_w * cols, target_h * rows), (26, 26, 46))
        for i, img in enumerate(resized):
            r, c = divmod(i, cols)
            canvas.paste(img, (c * target_w, r * target_h))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        canvas.save(out_path)
    finally:
        try:
            for name in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, name))
            os.rmdir(tmp_dir)
        except OSError:
            pass


def visualize(args: argparse.Namespace) -> None:
    files = resolve_files(args)
    if not files:
        raise RuntimeError("no candidate files selected")
    print(f"Visualizing {len(files)} object(s)")
    overview_path = os.path.join(args.outdir, args.overview_name)
    if args.overview_only:
        make_overview_direct(files, args, overview_path)
        print(f"overview -> {overview_path}")
        return

    outputs = []
    for i, path in enumerate(files, 1):
        try:
            out = save_one(path, args)
            outputs.append(out)
            print(f"[{i}/{len(files)}] {obj_id_from_hdf5(path)} -> {out}")
        except Exception as exc:
            print(f"[{i}/{len(files)}] skip {path}: {type(exc).__name__}: {exc}")
    if args.overview and len(outputs) > 1:
        overview_path = os.path.join(args.outdir, args.overview_name)
        make_overview(outputs, overview_path, cols=args.overview_cols)
        print(f"overview -> {overview_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize PDM candidates, one overlay image per object")
    parser.add_argument("--hdf5", nargs="*", default=None, help="Specific candidate HDF5 file(s)")
    parser.add_argument("--candidates-dir", default=os.path.join(PROJ, "output", "pdm", "candidates"))
    parser.add_argument("--condition-h5", default=os.path.join(PROJ, "output", "pdm", "cache", "conditions_4096.h5"))
    parser.add_argument(
        "--use-condition-cache",
        action="store_true",
        default=False,
        help="Use cached PDM condition points (training cache). Default: resample metric rotated_mesh.",
    )
    parser.add_argument("--outdir", default=os.path.join(PROJ, "output", "pdm", "vis_overlay"))
    parser.add_argument("--obj", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--random", type=int, default=0, metavar="N")
    parser.add_argument("--top", type=int, default=20, help="Number of poses to overlay per object")
    parser.add_argument("--bg-points", type=int, default=8000)
    parser.add_argument("--mesh-root", default=DEFAULT_ROTATED_MESH_DIR)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=132.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overview", action="store_true", default=True)
    parser.add_argument("--no-overview", dest="overview", action="store_false")
    parser.add_argument(
        "--overview-only",
        action="store_true",
        default=None,
        help="Only write overview.png, no per-object images. Defaults to true with --all.",
    )
    parser.add_argument(
        "--write-individual",
        dest="overview_only",
        action="store_false",
        help="Write per-object images as well as overview.",
    )
    parser.add_argument("--overview-name", default="overview.png")
    parser.add_argument("--overview-cols", type=int, default=5)
    args = parser.parse_args()
    if args.overview_only is None:
        args.overview_only = bool(args.all and not args.obj and not args.hdf5 and not args.random)
    return args


if __name__ == "__main__":
    visualize(build_parser())
