"""Rotate mesh frame so local XYZ align with robot base; preserve physical pose."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshAlignResult:
    """Outputs of base-axis alignment (for affordance / PDM / T6)."""

    T_fix_mesh: np.ndarray  # 4x4: p_mesh = T_fix @ p_mesh'
    T_cam_mesh_aligned: np.ndarray
    T_base_mesh_aligned: np.ndarray
    centroid_mesh: np.ndarray
    aligned_glb: Path
    max_point_err_m: float
    R_aligned_residual: float
    chain_err: float


def _as_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ri = R.T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Ri
    out[:3, 3] = -Ri @ t
    return out


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=np.float64)
    out = (T @ np.hstack([pts, ones]).T).T
    return out[:, :3]


def _rotation_error_from_identity(R: np.ndarray) -> float:
    return float(np.linalg.norm(R - np.eye(3), ord="fro"))


def align_mesh_to_base(
    mesh_path: Path,
    T_cam_mesh_fp: np.ndarray,
    T_base_cam: np.ndarray,
    aligned_glb_out: Path,
    *,
    rtol: float = 1e-5,
    atol_trans: float = 1e-4,
    atol_rot: float = 0.02,
    n_sample: int = 500,
) -> MeshAlignResult:
    """
    Re-express mesh in frame mesh' where axes match robot base (+X,+Y,+Z).

    p_mesh = T_fix @ p_mesh'  with T_fix rotation R_b about centroid c.
    T_base_mesh' = T_base_mesh_fp @ inv(T_fix),  R(T_base_mesh') ≈ I.
    """
    import trimesh

    T_cam_mesh_fp = np.asarray(T_cam_mesh_fp, dtype=np.float64).reshape(4, 4)
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    T_base_mesh_fp = T_base_cam @ T_cam_mesh_fp

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    c = verts.mean(axis=0)

    R_b = T_base_mesh_fp[:3, :3].copy()
    det = float(np.linalg.det(R_b))
    if abs(det - 1.0) > 0.01:
        raise ValueError(f"T_base_mesh rotation det={det:.4f} (expected +1)")

    # Frame fix: T_base' = T_base_fp @ T_fix,  R(T_base') ≈ I.
    # Vertex relabel: v' = R_b @ (v - c) + c  (so p_base unchanged)
    R_inv = R_b.T
    t_fix = c - R_inv @ c
    T_fix = _as_homogeneous(R_inv, t_fix)

    verts_aligned = (R_b @ (verts - c).T).T + c
    mesh_aligned = trimesh.Trimesh(
        vertices=verts_aligned, faces=mesh.faces.copy(), process=False
    )
    aligned_glb_out.parent.mkdir(parents=True, exist_ok=True)
    mesh_aligned.export(str(aligned_glb_out), file_type="glb")

    T_base_mesh_aligned = T_base_mesh_fp @ T_fix
    T_cam_mesh_aligned = _invert_rigid(T_base_cam) @ T_base_mesh_aligned

    # --- verification ---
    chain_err = float(
        np.max(np.abs(T_base_cam @ T_cam_mesh_fp - T_base_mesh_fp))
    )
    R_res = _rotation_error_from_identity(T_base_mesh_aligned[:3, :3])

    rng = np.random.default_rng(0)
    n = min(len(verts), n_sample)
    idx = rng.choice(len(verts), n, replace=False) if len(verts) > n else np.arange(len(verts))
    v_old = verts[idx]
    v_new = verts_aligned[idx]
    ones = np.ones((len(idx), 1))
    base_old = _transform_points(T_base_mesh_fp, v_old)
    # v' stores R^{-1}(v-c);  p_base = T_fp @ T_fix @ [v';1]
    p_new_h = (T_base_mesh_fp @ T_fix @ np.hstack([v_new, ones]).T).T[:, :3]
    max_point_err = float(np.max(np.linalg.norm(base_old - p_new_h, axis=1)))

    if chain_err > 1e-3:
        raise RuntimeError(f"FK chain error {chain_err:.4e} before align")
    if max_point_err > atol_trans:
        raise RuntimeError(
            f"Align point preservation failed: max_err={max_point_err:.4e} m "
            f"(tol {atol_trans})"
        )
    if R_res > atol_rot:
        raise RuntimeError(
            f"T_base_mesh_aligned rotation not identity: fro_err={R_res:.4f} "
            f"(tol {atol_rot})"
        )

    cam_err = float(
        np.max(np.abs(T_cam_mesh_fp @ T_fix - T_cam_mesh_aligned))
    )
    if cam_err > 1e-3:
        raise RuntimeError(f"T_cam_mesh align consistency err={cam_err:.4e}")

    return MeshAlignResult(
        T_fix_mesh=T_fix,
        T_cam_mesh_aligned=T_cam_mesh_aligned,
        T_base_mesh_aligned=T_base_mesh_aligned,
        centroid_mesh=c,
        aligned_glb=aligned_glb_out,
        max_point_err_m=max_point_err,
        R_aligned_residual=R_res,
        chain_err=chain_err,
    )


def write_mesh_frame_align_json(path: Path, result: MeshAlignResult, *, notes: str = "") -> None:
    payload: dict[str, Any] = {
        "mesh_frame_src": "sam3d_scaled",
        "mesh_frame_dst": "base_aligned",
        "method": "v' = R_base @ (v-c) + c; T_base' = T_base_fp @ T_fix",
        "composition": "p_mesh = T_fix @ p_mesh'; T_base_mesh' = T_base_mesh_fp @ T_fix",
        "T_fix_mesh": result.T_fix_mesh.tolist(),
        "centroid_mesh_m": result.centroid_mesh.tolist(),
        "checks": {
            "max_base_point_err_m": result.max_point_err_m,
            "R_base_mesh_aligned_fro_err": result.R_aligned_residual,
            "T_base_cam_T_cam_chain_err": result.chain_err,
        },
        "outputs": {
            "aligned_glb": str(result.aligned_glb.name),
            "use_for_inference": "object_base_aligned.glb + T_cam_mesh.json + T_base_mesh.json",
        },
        "notes": notes or "Local mesh +X/+Y/+Z parallel to robot base; physical pose unchanged.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
