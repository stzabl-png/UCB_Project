"""Metric rotated-mesh point clouds and affordance v6 inference (PDM / eval)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.infer_mesh_v6 import load_triangle_mesh, rescale_mesh_for_v6, sample_mesh_points


def resolve_metric_dataset(obj_id: str, dataset: str | None = None) -> str:
    if dataset and str(dataset) not in ("evaluation", ""):
        return str(dataset)
    from tools.random_grasp_sampler import infer_obj_dataset

    return infer_obj_dataset(obj_id, None)


def resolve_mesh_path(obj_id: str, mesh_root: str | Path, dataset: str | None = None) -> Path:
    root = Path(mesh_root).expanduser().resolve()
    ds_guess = resolve_metric_dataset(obj_id, dataset)
    candidates = [
        root / ds_guess / obj_id / "mesh.ply",
        root / obj_id / "mesh.ply",
    ]
    for ds in ("oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3"):
        candidates.append(root / ds / obj_id / "mesh.ply")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"rotated SAM3D mesh not found for {obj_id} under {root}")


def prepare_metric_point_cloud(
    obj_id: str,
    *,
    mesh_root: str | Path,
    dataset: str | None = None,
    num_points: int = 4096,
    seed: int = 0,
    target_max_extent: float = 0.28,
    auto_extent_lo: float = 0.02,
    auto_extent_hi: float = 0.80,
    min_scale_factor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample (points, normals) on metric-scaled rotated_mesh (same as glb_to_pdm / eval batch)."""
    metric_ds = resolve_metric_dataset(obj_id, dataset)
    mesh_path = resolve_mesh_path(obj_id, mesh_root, metric_ds)
    mesh = load_triangle_mesh(mesh_path)
    try:
        from tools import random_grasp_sampler as rgs

        sf = rgs.read_scale_factor(obj_id, metric_ds)
        if rgs.apply_metric_scale_to_mesh(obj_id, metric_ds) and abs(sf - 1.0) > 1e-8:
            mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) * float(sf)).astype(np.float64)
    except Exception as exc:
        print(f"  WARNING: metric scale lookup failed for {obj_id}: {exc}")
    mesh, _ = rescale_mesh_for_v6(
        mesh,
        target_max_extent=target_max_extent,
        scale_mode="never",
        extent_lo=auto_extent_lo,
        extent_hi=auto_extent_hi,
        min_scale_factor=min_scale_factor,
        center_mesh=False,
    )
    points, normals = sample_mesh_points(mesh, num_points, seed)
    return points, normals, str(mesh_path.resolve())


def predict_affordance_v6(
    model,
    points: np.ndarray,
    normals: np.ndarray,
    device,
) -> np.ndarray:
    """Run affordance v6 on a single object point cloud; returns (N,) heatmap."""
    import torch

    from model.inference_v6 import predict_heatmap_batch

    pts = np.asarray(points, dtype=np.float32)[None, ...]
    nrm = np.asarray(normals, dtype=np.float32)[None, ...]
    pred = predict_heatmap_batch(model, pts, nrm, device)
    if pred.ndim == 2:
        pred = pred[0]
    return np.asarray(pred, dtype=np.float32).reshape(-1)
