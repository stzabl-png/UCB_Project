#!/usr/bin/env python3
"""
**WORK IN PROGRESS** — debug viz for stable_orientations; output paths/layout may change.

vis_stable_orientations.py — 可视化 stable_orientations.json 中每个摆放朝向
====================================================================
对每个 orientation 将 raw+scale mesh 左乘 R 后渲染点云 + 坐标轴。

用法:
    python3 tools/vis_stable_orientations.py --obj A01026 --dataset oakink
    python3 tools/vis_stable_orientations.py --obj A01026 --dataset oakink --open

输出:
    output/stable_pose_vis/{dataset}/{obj_id}/pose_{id:02d}_{method}.png
    output/stable_pose_vis/{dataset}/{obj_id}/overview.png   (多宫格总览)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))
from mesh_utils import PROC_MESH_DIR, load_mesh_raw  # noqa: E402

OUT_ROOT = os.path.join(PROJ, "output", "stable_pose_vis")
N_SURFACE = 5000
VIEW_ELEV, VIEW_AZIM = 22, 135


def load_stable_doc(obj_id: str, dataset: str) -> dict:
    path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "stable_orientations.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}\n"
            f"请先: python3 data/estimate_stable_orientations.py --obj {obj_id} --dataset {dataset}"
        )
    with open(path) as f:
        return json.load(f)


def mesh_for_orientation(obj_id: str, dataset: str, R: np.ndarray) -> trimesh.Trimesh:
    mesh = load_mesh_raw(obj_id, dataset, apply_scale=True)
    mesh = mesh.copy()
    mesh.vertices = (R @ mesh.vertices.T).T
    return mesh


def _set_equal_aspect(ax, pts: np.ndarray) -> None:
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def draw_pose_ax(
    ax,
    mesh: trimesh.Trimesh,
    *,
    title: str,
    n_surface: int = N_SURFACE,
    show_axes: bool = True,
) -> None:
    pts, _ = trimesh.sample.sample_surface(mesh, min(n_surface, len(mesh.faces) * 3))
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c="#5ab4d4", s=2.0, alpha=0.55, linewidths=0, depthshade=False,
    )
    if show_axes:
        origin = mesh.centroid
        axis_len = float(mesh.bounding_box.extents.max()) * 0.4
        for direction, color in [([1, 0, 0], "r"), ([0, 1, 0], "g"), ([0, 0, 1], "b")]:
            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                length=axis_len, color=color, arrow_length_ratio=0.2, linewidth=1.5,
            )
    ext = mesh.bounding_box.extents * 100
    zmin = float(mesh.vertices[:, 2].min())
    ax.set_title(
        f"{title}\nbbox {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm  z_min={zmin*100:.1f} cm",
        color="#ddd",
        fontsize=8,
    )
    ax.set_xlabel("X", color="#aaa", fontsize=7)
    ax.set_ylabel("Y", color="#aaa", fontsize=7)
    ax.set_zlabel("Z", color="#aaa", fontsize=7)
    ax.tick_params(colors="#666", labelsize=6)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    _set_equal_aspect(ax, pts)
    ax.set_facecolor("#1a1a2e")


def render_pose_png(mesh: trimesh.Trimesh, title: str, out_path: str) -> None:
    fig = plt.figure(figsize=(6, 6), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    draw_pose_ax(ax, mesh, title=title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def render_overview(orientations: list[dict], obj_id: str, dataset: str, out_path: str) -> None:
    n = len(orientations)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows), facecolor="#1a1a2e")
    for i, ori in enumerate(orientations):
        R = np.array(ori["matrix"], dtype=np.float64)
        mesh = mesh_for_orientation(obj_id, dataset, R)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        euler = ori.get("euler_xyz_deg", [0, 0, 0])
        prob = ori.get("probability")
        prob_s = f" p={prob:.3f}" if prob is not None else ""
        title = f"id={ori['id']} {ori.get('method','?')}{prob_s}\n{[round(e, 1) for e in euler]}°"
        draw_pose_ax(ax, mesh, title=title, n_surface=2500, show_axes=i == 0)
    fig.suptitle(f"{obj_id} ({dataset}) stable_orientations", color="#eee", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="可视化 stable_orientations.json")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--dataset", default="oakink")
    parser.add_argument("--open", action="store_true", help="生成后用系统默认程序打开 overview.png")
    parser.add_argument("--outdir", default=None, help="默认 output/stable_pose_vis/{dataset}/{obj}")
    args = parser.parse_args()

    doc = load_stable_doc(args.obj, args.dataset)
    orientations = doc["orientations"]
    out_dir = args.outdir or os.path.join(OUT_ROOT, args.dataset, args.obj)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 {len(orientations)} orientations → {out_dir}")
    for ori in orientations:
        oid = int(ori["id"])
        method = str(ori.get("method", "unknown")).replace("/", "_")
        R = np.array(ori["matrix"], dtype=np.float64)
        mesh = mesh_for_orientation(args.obj, args.dataset, R)
        euler = ori.get("euler_xyz_deg", [0, 0, 0])
        prob = ori.get("probability")
        prob_s = f" prob={prob:.3f}" if prob is not None else ""
        title = f"{args.obj} id={oid} {method}{prob_s}\n{[round(e, 1) for e in euler]}°"
        png = os.path.join(out_dir, f"pose_{oid:02d}_{method}.png")
        render_pose_png(mesh, title, png)
        print(f"  ✅ {os.path.basename(png)}")

    overview = os.path.join(out_dir, "overview.png")
    render_overview(orientations, args.obj, args.dataset, overview)
    print(f"  ✅ overview → {overview}")

    if args.open and os.path.isfile(overview):
        import subprocess
        subprocess.run(["xdg-open", overview], check=False)


if __name__ == "__main__":
    main()
