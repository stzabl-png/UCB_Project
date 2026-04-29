#!/usr/bin/env python3
"""
estimate_obj_scale_ego.py — 第一人称视频物体 Scale 估计
(对应第三人称的 data/estimate_obj_scale.py)

原理与第三人称完全一致:
  1. 用 MegaSAM metric depth (depth.npz, 单位 m) + K.npy
  2. SAM2 mask (obj_recon_input/egocentric/{obj}/0.png) 筛选物体像素
  3. Back-project → 3D 点云 → 计算真实直径 d_real
  4. SAM3D mesh 直径 d_mesh
  5. scale_factor = d_real / d_mesh
  6. MANO 交叉验证 (可选)
  7. 保存 scale.json 至 obj_meshes/egocentric/{obj}/

区别于第三人称:
  - 深度来源: MegaSAM depth.npz（不需要 Depth Pro）
  - 选用 depth frame 最靠近 mask 注释帧的那一帧

用法:
    conda run -n hawor python data/estimate_obj_scale_ego.py
    conda run -n hawor python data/estimate_obj_scale_ego.py --obj assemble_tile
    conda run -n hawor python data/estimate_obj_scale_ego.py --redo
"""

import os, sys, json, json, argparse
import numpy as np
import cv2
import trimesh
from scipy.spatial import cKDTree

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPTH_BASE = os.path.join(PROJ, "data_hub", "ProcessedData", "egocentric_depth")
MASK_BASE  = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_recon_input", "egocentric")
MESH_BASE  = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes", "egocentric")

# ── 序列配置：从 JSON registry 动态加载 ──────────────────────────────────────
import sys as _sys
_sys.path.insert(0, os.path.join(PROJ, "tools"))
REGISTRY_JSON = os.path.join(PROJ, "tools", "egodex_sequence_registry.json")

_BUILTIN_OBJ_DEPTH_MAP = {
    "assemble_tile": ("egodex",
                      "egodex/assemble_disassemble_tiles/1"),
    "orange":        ("ph2d_avp",
                      "ph2d_avp/1407-picking_orange_tj_2025-03-12_16-42-19"
                      "/processed_episode_3"),
}


def load_obj_depth_map():
    """EGO-BUG-01 修复: 从 JSON registry 构建 obj → depth_dir 映射。
    跳过 skipped=True 的条目。
    """
    odm = dict(_BUILTIN_OBJ_DEPTH_MAP)
    if os.path.exists(REGISTRY_JSON):
        with open(REGISTRY_JSON) as f:
            jreg = json.load(f)
        for key, cfg in jreg.items():
            if cfg.get("skipped"):
                continue
            obj_name  = cfg["obj_name"]
            depth_dir = cfg["depth_dir"]   # absolute path
            # depth_dir is absolute; store as (dataset, abs_path) sentinel
            odm[obj_name] = (cfg["dataset"], depth_dir)
    return odm


OBJ_DEPTH_MAP = None  # lazy-loaded in main()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def depth_mask_to_pointcloud(depth_m, mask_resized, K, max_depth=3.0):
    """
    Back-project mask pixels into 3D using metric depth and K.
    max_depth: ignore pixels farther than this (arm-length based).
    Returns (N, 3) array or None.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_idx, u_idx = np.where((mask_resized > 0) &
                             (depth_m > 0.05) &
                             (depth_m < max_depth))
    if len(v_idx) < 10:
        return None

    z = depth_m[v_idx, u_idx].astype(np.float64)

    # Robust filter: keep z within ±40% of the median
    # (eliminates background leakage that slips through the mask at far depths)
    z_med = np.median(z)
    keep  = (z > z_med * 0.60) & (z < z_med * 1.40)
    if keep.sum() < 5:
        keep = np.ones(len(z), dtype=bool)   # fallback: keep all < max_depth

    z    = z[keep]
    x    = (u_idx[keep] - cx) * z / fx
    y    = (v_idx[keep] - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def pointcloud_diameter(pts):
    """
    Estimate object diameter from 3D point cloud.
    Uses the lateral (XY) span only to avoid depth-direction bias
    (when camera sees object face-on, Z-spread is artifactually large).
    Returns float diameter in metres.
    """
    if pts is None or len(pts) < 5:
        return None
    # Lateral (XY) extents at median depth
    x_span = np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5)
    y_span = np.percentile(pts[:, 1], 95) - np.percentile(pts[:, 1], 5)
    # 3D bounding-box diagonal in XY plane (not Z)
    diag_xy = float(np.sqrt(x_span**2 + y_span**2))
    # Also compute full 3D range as a secondary estimate
    ranges  = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
    full_3d = float(np.linalg.norm(ranges))
    # Use the smaller of XY-diagonal and full-3D (avoids inflation from depth spread)
    return min(diag_xy, full_3d)


def mesh_diameter(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    return float(mesh.bounding_sphere.primitive.radius * 2), mesh


def estimate_scale(obj_name, depth_dir, mask_path, mesh_path, depth_frame=0):
    """
    Core estimation function.
    Returns result dict or None.
    """
    # 1. Load metric depth + K
    npz_path = os.path.join(depth_dir, "depth.npz")
    K_path   = os.path.join(depth_dir, "K.npy")
    if not os.path.exists(npz_path) or not os.path.exists(K_path):
        print(f"  ❌ Missing depth.npz or K.npy in {depth_dir}")
        return None

    depths = np.load(npz_path)["depths"].astype(np.float32)  # (N, H_d, W_d)
    K      = np.load(K_path)                                   # 3×3 @ depth res
    N_d, H_d, W_d = depths.shape

    fi = min(depth_frame, N_d - 1)
    depth_m = depths[fi]   # (H_d, W_d), metric metres
    print(f"  depth frame {fi}: {H_d}×{W_d}  range [{depth_m.min():.2f}, {depth_m.max():.2f}]m")

    # 2. Load & resize mask to depth resolution
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"  ❌ Mask not found: {mask_path}")
        return None
    mask_d = cv2.resize(mask, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
    n_mask = int((mask_d > 0).sum())
    print(f"  mask: {mask.shape} → {mask_d.shape}  nonzero={n_mask}")
    if n_mask < 20:
        print("  ❌ Too few mask pixels after resize")
        return None

    # 3. Back-project → 3D PointCloud
    pts = depth_mask_to_pointcloud(depth_m, mask_d, K)
    if pts is None or len(pts) < 10:
        print(f"  ❌ Too few 3D points ({0 if pts is None else len(pts)})")
        return None
    print(f"  3D points: {len(pts)}  depth range [{pts[:,2].min():.3f},{pts[:,2].max():.3f}]m")

    # 4. Real diameter
    d_real = pointcloud_diameter(pts)
    if d_real is None or d_real < 0.005:
        print(f"  ❌ Degenerate diameter: {d_real}")
        return None
    print(f"  d_real = {d_real:.4f} m")

    # 5. Mesh diameter
    d_mesh, mesh = mesh_diameter(mesh_path)
    print(f"  d_mesh = {d_mesh:.4f} m")

    if d_mesh < 1e-6:
        return None

    scale_factor = d_real / d_mesh
    print(f"  scale_factor = {scale_factor:.6f}")

    return {
        "scale_factor": round(float(scale_factor), 6),
        "d_real_m":     round(float(d_real), 4),
        "d_mesh_raw":   round(float(d_mesh), 4),
        "depth_frame":  fi,
        "n_pts":        int(len(pts)),
        "method":       "megasam_depth_mask",
        "obj":          obj_name,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ego 物体 Scale 估计 (MegaSAM depth + SAM2 mask)")
    parser.add_argument("--obj",  default=None,
                        choices=None,  # dynamic — all registered obj_names
                        help="只处理此物体（默认全部）")
    parser.add_argument("--redo", action="store_true", help="重新计算已有 scale.json")
    parser.add_argument("--depth-frame", type=int, default=0,
                        help="使用第几帧 depth（默认 0，即标注帧）")
    args = parser.parse_args()

    global OBJ_DEPTH_MAP
    OBJ_DEPTH_MAP = load_obj_depth_map()   # EGO-BUG-01 修复
    objs = [args.obj] if args.obj else list(OBJ_DEPTH_MAP.keys())

    print("=" * 60)
    print(" Ego Object Scale Estimation")
    print(" Method: MegaSAM metric depth + SAM2 mask → d_real / d_mesh")
    print("=" * 60)

    results = []
    for obj_name in objs:
        print(f"\n── {obj_name} ──")
        ds, depth_sub = OBJ_DEPTH_MAP[obj_name]
        # depth_sub may be an absolute path (from JSON) or a relative subdir (builtins)
        depth_dir = depth_sub if os.path.isabs(depth_sub) else os.path.join(DEPTH_BASE, depth_sub)
        mask_path  = os.path.join(MASK_BASE, obj_name, "0.png")
        mesh_dir   = os.path.join(MESH_BASE, obj_name)
        mesh_path  = os.path.join(mesh_dir, "mesh.ply")
        scale_json = os.path.join(mesh_dir, "scale.json")

        if not args.redo and os.path.exists(scale_json):
            with open(scale_json) as f:
                cached = json.load(f)
            sf = cached.get("scale_factor", "?")
            print(f"  ⏭  Cached: scale_factor={sf}  (use --redo to recompute)")
            results.append(cached)
            continue

        if not os.path.exists(mesh_path):
            print(f"  ❌ mesh.ply not found: {mesh_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  ❌ mask not found: {mask_path}")
            continue

        result = estimate_scale(
            obj_name, depth_dir, mask_path, mesh_path,
            depth_frame=args.depth_frame)

        if result is None:
            print(f"  ❌ Scale estimation failed for {obj_name}")
            continue

        # Save scale.json
        os.makedirs(mesh_dir, exist_ok=True)
        with open(scale_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✅ Saved → {scale_json}")
        results.append(result)

    print(f"\n{'='*60}")
    print(f"{'Object':<20} {'scale_factor':>13} {'d_real_m':>10} {'d_mesh_raw':>12} {'method':>22}")
    print("-" * 80)
    for r in results:
        print(f"  {r['obj']:<18} {r['scale_factor']:>13.6f} "
              f"{r['d_real_m']:>8.4f}m {r['d_mesh_raw']:>10.4f}m "
              f"  {r.get('method','?'):>22}")

    print()
    print("下一步: 重跑接触图")
    print("  conda run -n hawor python tools/gen_ego_contact_map.py")


if __name__ == "__main__":
    main()
