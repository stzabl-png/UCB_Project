#!/usr/bin/env python3
"""
mesh_utils.py — 统一 mesh 加载工具
=====================================
所有脚本调用同一入口，确保 PLY mesh 的朝向与 USD/Sim 完全一致。

坐标系基准 (Isaac Sim 世界坐标系):
  +Z : 垂直桌面向上
  +Y : 机械臂前进方向
  +X : 横向

canonical rotation 来源: data_hub/ProcessedData/obj_meshes/{dataset}/{obj_id}/rotation.json
  → 由 estimate_obj_rotation.py 生成
  → convert_obj_usd.py 已正确将其 bake 进 USD 顶点
  → 本文件让 PLY 也应用同样旋转，使两者完全对齐
"""
import os, json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# ── 路径 ─────────────────────────────────────────────────────────────────────
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_MESH_DIR = os.path.join(_PROJ, "data_hub", "ProcessedData", "obj_meshes")
DATASETS = ["oakink", "dexycb", "arctic"]


def get_scale_factor(obj_id: str, dataset: str) -> float:
    """读取 scale.json，无则返回 1.0。"""
    path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "scale.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return float(d.get("scale_factor", 1.0))
    return 1.0


def get_canonical_euler(obj_id: str, dataset: str = "oakink") -> list:
    """
    返回 canonical rotation (euler_xyz_deg)。
    来源：obj_meshes/{dataset}/{obj_id}/rotation.json
    无文件或旋转≈0 时返回 [0.0, 0.0, 0.0]。
    """
    rot_path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "rotation.json")
    if os.path.exists(rot_path):
        with open(rot_path) as f:
            data = json.load(f)
        euler = data.get("euler_xyz_deg", [0.0, 0.0, 0.0])
        if any(abs(e) > 0.5 for e in euler):
            return [float(e) for e in euler]
    return [0.0, 0.0, 0.0]


def get_canonical_matrix(obj_id: str, dataset: str = "oakink") -> np.ndarray:
    """返回 3×3 旋转矩阵，无旋转时为 I。"""
    euler = get_canonical_euler(obj_id, dataset)
    if any(abs(e) > 0.5 for e in euler):
        return Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    return np.eye(3, dtype=np.float64)


def find_ply(obj_id: str, dataset: str = None) -> tuple:
    """
    查找 mesh.ply 文件路径。
    返回 (ply_path, dataset_found) 或 (None, None)。
    """
    search = [dataset] if dataset else DATASETS
    for ds in search:
        p = os.path.join(PROC_MESH_DIR, ds, obj_id, "mesh.ply")
        if os.path.exists(p):
            return p, ds
    return None, None


def load_mesh_canonical(
    obj_id: str,
    dataset: str = None,
    apply_scale: bool = True,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    加载 PLY mesh 并应用 scale + canonical rotation，
    使其朝向与 Isaac Sim 中的 USD 完全一致。

    Args:
        obj_id:       物体 ID
        dataset:      'oakink' / 'dexycb' / 'arctic'，None 则自动搜索
        apply_scale:  是否应用 scale.json 转换为米制
        verbose:      是否打印旋转信息

    Returns:
        trimesh.Trimesh，已旋转到 canonical 朝向（与 Sim/USD 一致）
    """
    ply_path, ds = find_ply(obj_id, dataset)
    if ply_path is None:
        raise FileNotFoundError(f"mesh.ply not found for {obj_id} (searched: {dataset or DATASETS})")

    mesh = trimesh.load(ply_path, force="mesh")

    # ── 1. 应用 scale ──────────────────────────────────────────────────────
    if apply_scale:
        scale = get_scale_factor(obj_id, ds)
        if abs(scale - 1.0) > 1e-6:
            mesh.vertices = mesh.vertices * scale

    # ── 2. 应用 canonical rotation (使朝向与 USD/Sim 对齐) ─────────────────
    euler = get_canonical_euler(obj_id, ds)
    if any(abs(e) > 0.5 for e in euler):
        R_mat = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        mesh.vertices = (R_mat @ mesh.vertices.T).T
        if not mesh.is_watertight:
            trimesh.repair.fix_normals(mesh)
        if verbose:
            print(f"  [mesh_utils] {obj_id}: canonical rot {[round(e,1) for e in euler]}°")
    else:
        if verbose:
            print(f"  [mesh_utils] {obj_id}: no rotation (identity)")

    return mesh


def load_mesh_raw(
    obj_id: str,
    dataset: str = None,
    apply_scale: bool = True,
) -> trimesh.Trimesh:
    """
    加载原始 PLY（不应用 canonical rotation）。
    仅在需要与旧数据对比时使用。
    """
    ply_path, ds = find_ply(obj_id, dataset)
    if ply_path is None:
        raise FileNotFoundError(f"mesh.ply not found for {obj_id}")
    mesh = trimesh.load(ply_path, force="mesh")
    if apply_scale:
        scale = get_scale_factor(obj_id, ds)
        if abs(scale - 1.0) > 1e-6:
            mesh.vertices = mesh.vertices * scale
    return mesh
