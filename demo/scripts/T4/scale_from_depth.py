#!/usr/bin/env python3
"""
T4 — Metric scale from Razor depth + mask + K.

Input:  input/depth/depth.npy, input/calib/K.npy, output/segment/mask.png,
        output/mesh/object_raw.glb
Output: output/mesh/object_scaled.glb, scale.json,
        output/vis/T4_scale_scene_preview.png

Usage:
  python demo/scripts/T4/scale_from_depth.py \\
    --session-dir demo/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import repo_root, resolve_session_dirs  # noqa: E402

_T4_DIR = Path(__file__).resolve().parent
if str(_T4_DIR) not in sys.path:
    sys.path.insert(0, str(_T4_DIR))

from scale_common import (  # noqa: E402
    apply_uniform_scale,
    build_scale_payload,
    coarse_align_mesh_to_depth,
    compute_scale_factor,
    depth_mask_to_pointcloud,
    estimate_real_size,
    load_depth_m,
    load_K,
    load_mask_bool,
    mesh_characteristic_size,
    preprocess_mask,
    T_cam_mesh_from_Rt,
    write_scale_json,
)
from visualize import save_scale_scene_preview  # noqa: E402

_T3_DIR = _SCRIPTS_ROOT / "T3"
if str(_T3_DIR) not in sys.path:
    sys.path.insert(0, str(_T3_DIR))
from sam3d_common import load_rgb_pil, session_id_from_input  # noqa: E402


def run_t1_validate(session_dir: Path) -> bool:
    t1 = _SCRIPTS_ROOT / "T1" / "validate_input.py"
    r = subprocess.run(
        [sys.executable, str(t1), "--session-dir", str(session_dir)],
        cwd=str(repo_root()),
    )
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 T4: metric mesh scale from depth")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument(
        "--vis-only",
        action="store_true",
        help="Rebuild preview from existing object_scaled.glb + scale.json",
    )
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    if args.validate and not run_t1_validate(dirs.session_root):
        print("T1 validation failed", file=sys.stderr)
        return 1

    depth_path = dirs.input_rel("depth", "depth.npy")
    mask_path = dirs.output_rel("segment", "mask.png")
    raw_glb = dirs.output_rel("mesh", "object_raw.glb")
    scaled_glb = dirs.output_rel("mesh", "object_scaled.glb")
    scale_json = dirs.output_rel("mesh", "scale.json")
    vis_out = dirs.output_rel("vis", "T4_scale_scene_preview.png")
    rgb_path = dirs.input_rel("rgb", "left_rgb.png")

    session_id = session_id_from_input(dirs.input_dir, dirs.session_id)

    if args.vis_only:
        if not scaled_glb.is_file() or not scale_json.is_file():
            print("Need object_scaled.glb and scale.json for --vis-only", file=sys.stderr)
            return 1
        return _run_vis_only(
            rgb_path, depth_path, mask_path, dirs.input_dir, scaled_glb, scale_json, vis_out, session_id
        )

    if scaled_glb.is_file() and scale_json.is_file() and not args.redo:
        print(f"Already exists: {scaled_glb}  (use --redo)")
        if not args.no_vis and not vis_out.is_file():
            _run_vis_only(
                rgb_path, depth_path, mask_path, dirs.input_dir, scaled_glb, scale_json, vis_out, session_id
            )
        return 0

    for p, name in [
        (depth_path, "depth"),
        (mask_path, "mask"),
        (raw_glb, "object_raw.glb"),
        (rgb_path, "rgb"),
    ]:
        if not p.is_file():
            print(f"Missing {name}: {p}", file=sys.stderr)
            return 1

    depth = load_depth_m(depth_path)
    mask_raw = load_mask_bool(mask_path)
    mask, n_cc, n_raw = preprocess_mask(mask_raw)
    K = load_K(dirs.input_dir)
    rgb = load_rgb_pil(rgb_path)

    if mask.shape != depth.shape:
        from PIL import Image

        mask = (
            np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (depth.shape[1], depth.shape[0]), Image.NEAREST
                )
            )
            > 0
        )
    if rgb.shape[:2] != depth.shape:
        print(f"RGB {rgb.shape[:2]} != depth {depth.shape}", file=sys.stderr)
        return 1

    pts = depth_mask_to_pointcloud(depth, mask, K)
    if pts is None:
        print("Too few valid depth points in mask", file=sys.stderr)
        return 1

    est = estimate_real_size(
        pts, depth, mask, K, mask_cc_pixels=n_cc, mask_raw_pixels=n_raw
    )
    if est is None or est.d_real_m < 0.005:
        print("Invalid d_real estimate", file=sys.stderr)
        return 1

    mesh_raw = trimesh.load(str(raw_glb), force="mesh")
    if isinstance(mesh_raw, trimesh.Scene):
        mesh_raw = mesh_raw.dump(concatenate=True)
    d_mesh, d_mesh_pca, d_mesh_aabb = mesh_characteristic_size(mesh_raw)
    if d_mesh < 1e-9:
        print("Degenerate mesh diameter", file=sys.stderr)
        return 1

    scale_factor, clamped, clamp_note = compute_scale_factor(est.d_real_m, d_mesh)

    mesh_scaled = apply_uniform_scale(mesh_raw, scale_factor)
    mesh_scaled.export(str(scaled_glb), file_type="glb")

    # Coarse align scaled mesh (still in mesh frame) -> camera frame for visualization
    verts_scaled = np.asarray(mesh_scaled.vertices, dtype=np.float64)
    verts_cam, R, t = coarse_align_mesh_to_depth(verts_scaled, pts)
    T_coarse = T_cam_mesh_from_Rt(R, t)

    payload = build_scale_payload(
        session_id=session_id,
        scale_factor=scale_factor,
        est=est,
        d_mesh_raw=d_mesh,
        d_mesh_method="pca_max_span_primary",
        depth_pts=pts,
        n_mask_px=int(mask.sum()),
        clamped=clamped,
        clamp_note=clamp_note,
        coarse_T_cam_mesh=T_coarse.tolist(),
    )
    payload["object_raw_glb"] = raw_glb.name
    payload["object_scaled_glb"] = scaled_glb.name
    payload["d_mesh_pca_max"] = round(d_mesh_pca, 4)
    payload["d_mesh_aabb_max"] = round(d_mesh_aabb, 4)
    if not args.no_vis:
        payload["preview_png"] = str(vis_out)

    write_scale_json(scale_json, payload)

    if not args.no_vis:
        save_scale_scene_preview(
            rgb,
            mask,
            depth,
            K,
            pts,
            verts_cam,
            vis_out,
            session_id=session_id,
            scale_factor=scale_factor,
            d_real_m=est.d_real_m,
        )

    ext = np.asarray(mesh_scaled.vertices).max(0) - np.asarray(mesh_scaled.vertices).min(0)
    print(f"Session: {session_id}")
    print(f"Mask CC: {n_cc}/{n_raw} px  fusion={est.fusion_method} cues={est.fusion_cues_used}")
    print(
        f"d_real={est.d_real_m*100:.2f}cm (lateral={est.d_real_mask_lateral_m*100:.1f} "
        f"pca_max={est.d_real_pca_max_m*100:.1f} core={est.d_real_core_m*100:.1f})"
    )
    print(
        f"d_mesh={d_mesh:.4f} (pca={d_mesh_pca:.4f} aabb={d_mesh_aabb:.4f})  "
        f"scale={scale_factor:.6f}  clamped={clamped}"
    )
    print(f"Scaled bbox extent (m): [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}]")
    print(f"Saved: {scaled_glb}")
    print(f"Scale: {scale_json}")
    if not args.no_vis:
        print(f"Preview: {vis_out}")
    return 0


def _run_vis_only(
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    input_dir: Path,
    scaled_glb: Path,
    scale_json: Path,
    vis_out: Path,
    session_id: str,
) -> int:
    meta = json.loads(scale_json.read_text(encoding="utf-8"))
    sf = float(meta["scale_factor"])
    d_real = float(meta["d_real_m"])

    rgb = load_rgb_pil(rgb_path)
    depth = load_depth_m(depth_path)
    mask, _, _ = preprocess_mask(load_mask_bool(mask_path))
    K = load_K(input_dir)
    pts = depth_mask_to_pointcloud(depth, mask, K)
    if pts is None:
        print("Cannot rebuild vis: no depth points", file=sys.stderr)
        return 1

    mesh = trimesh.load(str(scaled_glb), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    T = meta.get("coarse_T_cam_mesh_vis")
    if T is not None:
        T = np.array(T, dtype=np.float64)
        ones = np.ones((len(verts), 1))
        verts_cam = (T @ np.hstack([verts, ones]).T).T[:, :3]
    else:
        verts_cam, _, _ = coarse_align_mesh_to_depth(verts, pts)

    save_scale_scene_preview(
        rgb, mask, depth, K, pts, verts_cam, vis_out,
        session_id=session_id, scale_factor=sf, d_real_m=d_real,
    )
    print(f"Preview: {vis_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
