"""Metric scale from Dexmate depth + mask + K (Phase 2 T4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCALE_CLAMP = (0.05, 3.0)
MIN_DEPTH_PTS = 30
# After depth-based scale (and clamp), shrink mesh slightly for sim2real margin.
POST_METRIC_SCALE_MULTIPLIER = 0.95


@dataclass
class RealSizeEstimate:
    d_real_m: float
    d_real_pca_max_m: float
    d_real_pca_median_m: float
    d_real_core_m: float
    d_real_mask_lateral_m: float
    mask_width_m: float
    mask_height_m: float
    depth_median_m: float
    pca_spans_m: list[float]
    fusion_method: str
    fusion_cues_used: list[str]
    mask_cc_pixels: int
    mask_total_pixels: int


def load_K(session_input: Path) -> np.ndarray:
    k_path = session_input / "calib" / "K.npy"
    if k_path.is_file():
        return np.load(k_path).astype(np.float64)
    intr = session_input / "calib" / "intrinsics.json"
    if intr.is_file():
        return np.array(json.loads(intr.read_text())["K"], dtype=np.float64)
    raise FileNotFoundError(f"No K at {k_path} or {intr}")


def load_depth_m(depth_path: Path) -> np.ndarray:
    d = np.load(depth_path)
    if d.ndim != 2:
        raise ValueError(f"depth must be H×W, got {d.shape}")
    return d.astype(np.float32)


def load_mask_bool(mask_path: Path) -> np.ndarray:
    from PIL import Image

    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def mask_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest CC — reduces table speckle / stray SAM pixels (all objects)."""
    m = mask.astype(bool)
    if not m.any():
        return m
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(m)
        if n <= 1:
            return m
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep_id = int(counts.argmax())
        return labeled == keep_id
    except ImportError:
        return m


def preprocess_mask(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Largest CC; returns (mask, cc_pixels, raw_pixels)."""
    raw_n = int(mask.sum())
    cc = mask_largest_connected_component(mask)
    return cc, int(cc.sum()), raw_n


def depth_mask_to_pointcloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    max_depth: float = 3.0,
    z_band: float = 0.15,
) -> np.ndarray | None:
    """Back-project masked valid depth to camera frame (metres). Tighter Z band (v2)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = (mask > 0) & (depth_m > 0.05) & (depth_m < max_depth) & np.isfinite(depth_m)
    v_idx, u_idx = np.where(valid)
    if len(v_idx) < MIN_DEPTH_PTS:
        return None

    z = depth_m[v_idx, u_idx].astype(np.float64)
    z_med = float(np.median(z))
    lo, hi = 1.0 - z_band, 1.0 + z_band
    keep = (z > z_med * lo) & (z < z_med * hi)
    if keep.sum() < 20:
        keep = (z > z_med * 0.75) & (z < z_med * 1.25)
    if keep.sum() < 10:
        keep = np.ones(len(z), dtype=bool)

    z = z[keep]
    u = u_idx[keep].astype(np.float64)
    v = v_idx[keep].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def pca_axis_spans(pts: np.ndarray, percentile: float = 90) -> list[float]:
    """Per-principal-axis spans (not max — used for robust diameter)."""
    if pts is None or len(pts) < 10:
        return []
    c = pts.mean(axis=0)
    p = pts - c
    try:
        _, _, Vt = np.linalg.svd(p, full_matrices=False)
    except Exception:
        spans = []
        for axis in range(3):
            lo = np.percentile(pts[:, axis], 100 - percentile)
            hi = np.percentile(pts[:, axis], percentile)
            spans.append(float(hi - lo))
        return spans
    spans = []
    for v in Vt:
        proj = p @ v
        lo = np.percentile(proj, 100 - percentile)
        hi = np.percentile(proj, percentile)
        spans.append(float(hi - lo))
    return spans


def pointcloud_diameter_max(pts: np.ndarray, percentile: float = 95) -> float | None:
    spans = pca_axis_spans(pts, percentile)
    return float(max(spans)) if spans else None


def pointcloud_core_diameter(pts: np.ndarray, percentile: float = 90) -> float | None:
    """2 × p90 distance to centroid — down-weights far outliers."""
    if pts is None or len(pts) < 10:
        return None
    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c, axis=1)
    return float(2.0 * np.percentile(r, percentile))


def mask_lateral_extent_m(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    z_band: float = 0.12,
) -> tuple[float, float, float, float]:
    """
    Mask bounding box at median depth → physical width/height in metres.
    Strong for objects seen face-on (cans, boxes, tools); validated vs 3D cues.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    valid = (mask > 0) & (depth_m > 0.05) & (depth_m < 3.0) & np.isfinite(depth_m)
    v_idx, u_idx = np.where(valid)
    if len(u_idx) < 5:
        return 0.0, 0.0, 0.0, 0.0

    z = depth_m[v_idx, u_idx].astype(np.float64)
    z_med = float(np.median(z))
    keep = np.abs(z - z_med) <= z_band * z_med
    if keep.sum() < 5:
        keep = np.ones(len(z), dtype=bool)

    u = u_idx[keep].astype(np.float64)
    v = v_idx[keep].astype(np.float64)
    du = float(u.max() - u.min())
    dv = float(v.max() - v.min())
    width_m = du / fx * z_med
    height_m = dv / fy * z_med
    lateral = max(width_m, height_m)
    return lateral, width_m, height_m, z_med


def fuse_characteristic_size_m(cues: dict[str, float]) -> tuple[float, str, list[str]]:
    """
    Object-agnostic fusion: trimmed median of depth cues.

    - Always uses 3D depth cues (pca_max, core).
    - Adds mask lateral only if consistent with 3D (not table bleed).
    - Drops outliers >1.3× or <0.7× median before taking median.
    Works for cans, boxes, tools, elongated packages — not tuned to one SKU.
    """
    if "pca_max" not in cues or "core" not in cues:
        raise ValueError("cues need pca_max and core")

    used: list[str] = ["pca_max", "core"]
    values = [cues["pca_max"], cues["core"]]

    lateral = cues.get("lateral", 0.0)
    if lateral > 0.005:
        ref = float(np.median(values))
        if lateral <= 1.35 * ref and lateral >= 0.65 * ref:
            values.append(lateral)
            used.append("lateral")

    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    kept = arr[(arr >= 0.7 * med) & (arr <= 1.3 * med)]
    if len(kept) == 0:
        kept = arr
    d_real = float(np.median(kept))
    method = "trimmed_median" if len(kept) < len(arr) else "median"
    return d_real, method, used


def estimate_real_size(
    pts: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    mask_cc_pixels: int = 0,
    mask_raw_pixels: int = 0,
) -> RealSizeEstimate | None:
    """Adaptive d_real (v3) for general rigid objects with SAM2 mask + metric depth."""
    lateral, w_m, h_m, z_med = mask_lateral_extent_m(depth_m, mask, K)
    spans = pca_axis_spans(pts, percentile=90)
    if not spans:
        return None

    d_pca_max = float(max(spans))
    d_pca_med = float(np.median(spans))
    d_core = pointcloud_core_diameter(pts, percentile=90) or d_pca_max

    d_real, fusion_method, fusion_used = fuse_characteristic_size_m(
        {"pca_max": d_pca_max, "core": d_core, "lateral": lateral}
    )

    return RealSizeEstimate(
        d_real_m=d_real,
        d_real_pca_max_m=d_pca_max,
        d_real_pca_median_m=d_pca_med,
        d_real_core_m=d_core,
        d_real_mask_lateral_m=lateral,
        mask_width_m=w_m,
        mask_height_m=h_m,
        depth_median_m=z_med,
        pca_spans_m=[round(s, 4) for s in spans],
        fusion_method=fusion_method,
        fusion_cues_used=fusion_used,
        mask_cc_pixels=mask_cc_pixels,
        mask_total_pixels=mask_raw_pixels,
    )


# Back-compat alias
estimate_real_size_v2 = estimate_real_size


def mesh_characteristic_size(mesh) -> tuple[float, float, float]:
    """
    Mesh size for scale ratio — same statistic family as depth (PCA max span).
    Also returns AABB max for diagnostics.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    d_pca = pointcloud_diameter_max(verts, percentile=90)
    ext = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    d_aabb = float(np.max(ext))
    if d_pca is None or d_pca < 1e-9:
        return d_aabb, d_aabb, d_aabb
    # Prefer PCA max; if AABB wildly larger (diagonal frame), blend toward AABB min edge
    if d_aabb > 1.5 * d_pca:
        d_used = float(0.7 * d_pca + 0.3 * d_aabb)
    else:
        d_used = float(d_pca)
    return d_used, float(d_pca), d_aabb


def mesh_diameter_aabb(mesh) -> float:
    d_used, _, _ = mesh_characteristic_size(mesh)
    return d_used


def compute_scale_factor(
    d_real_m: float,
    d_mesh_raw: float,
    *,
    clamp: tuple[float, float] = SCALE_CLAMP,
) -> tuple[float, bool, str]:
    raw = d_real_m / d_mesh_raw
    lo, hi = clamp
    if raw < lo:
        return lo, True, f"clamped low ({raw:.4f} -> {lo})"
    if raw > hi:
        return hi, True, f"clamped high ({raw:.4f} -> {hi})"
    return raw, False, ""


def apply_post_metric_scale(scale_factor: float) -> float:
    """Apply fixed post-scale shrink (used by T4 and daemon pipeline)."""
    return float(scale_factor) * POST_METRIC_SCALE_MULTIPLIER


def apply_uniform_scale(mesh, scale_factor: float):
    import trimesh

    m = mesh.copy()
    m.vertices = np.asarray(m.vertices, dtype=np.float64) * scale_factor
    return m


def coarse_align_mesh_to_depth(
    mesh_verts_cam_scaled: np.ndarray,
    depth_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vis-only: translation to depth centroid (more stable overlay than PCA)."""
    vm = np.asarray(mesh_verts_cam_scaled, dtype=np.float64)
    vp = np.asarray(depth_pts, dtype=np.float64)
    t = vp.mean(0) - vm.mean(0)
    aligned = vm + t
    return aligned, np.eye(3, dtype=np.float64), t


def T_cam_mesh_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_cam_to_image(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = pts_cam[:, 2]
    valid = Z > 0.05
    u = K[0, 0] * pts_cam[:, 0] / np.maximum(Z, 1e-6) + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / np.maximum(Z, 1e-6) + K[1, 2]
    return u, v, valid & (v >= 0) & (v < 1e6)


def write_scale_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_scale_payload(
    *,
    session_id: str,
    scale_factor: float,
    scale_factor_depth: float | None = None,
    est: RealSizeEstimate,
    d_mesh_raw: float,
    d_mesh_method: str,
    depth_pts: np.ndarray,
    n_mask_px: int,
    clamped: bool,
    clamp_note: str,
    coarse_T_cam_mesh: list[list[float]] | None = None,
) -> dict[str, Any]:
    d_real = est.d_real_m
    plausible = 0.02 <= d_real <= 0.60
    warnings: list[str] = []
    if clamp_note:
        warnings.append(clamp_note)
    if d_real < 0.02:
        warnings.append("d_real < 2cm — check mask")
    elif d_real > 0.60:
        warnings.append("d_real > 60cm — check mask/depth")
    if est.d_real_pca_max_m > 1.35 * d_real:
        warnings.append(
            f"pca_max ({est.d_real_pca_max_m*100:.1f}cm) >> used ({d_real*100:.1f}cm); "
            "conservative min() applied"
        )

    if est.mask_total_pixels > 0 and est.mask_cc_pixels < 0.85 * est.mask_total_pixels:
        warnings.append(
            f"mask CC kept {est.mask_cc_pixels}/{est.mask_total_pixels} px "
            "(stray regions removed)"
        )

    sf_depth = float(scale_factor_depth if scale_factor_depth is not None else scale_factor)
    payload: dict[str, Any] = {
        "scale_factor": round(float(scale_factor), 6),
        "scale_factor_depth": round(sf_depth, 6),
        "post_metric_scale_multiplier": POST_METRIC_SCALE_MULTIPLIER,
        "method": "depth_mask_adaptive_v3",
        "fusion_method": est.fusion_method,
        "fusion_cues_used": est.fusion_cues_used,
        "d_real_m": round(d_real, 4),
        "d_real_pca_max_m": round(est.d_real_pca_max_m, 4),
        "d_real_pca_median_m": round(est.d_real_pca_median_m, 4),
        "d_real_core_m": round(est.d_real_core_m, 4),
        "d_real_mask_lateral_m": round(est.d_real_mask_lateral_m, 4),
        "mask_extent_m": {
            "width": round(est.mask_width_m, 4),
            "height": round(est.mask_height_m, 4),
        },
        "pca_spans_m": est.pca_spans_m,
        "d_mesh_raw": round(float(d_mesh_raw), 4),
        "d_mesh_method": d_mesh_method,
        "mask_cc_pixels": est.mask_cc_pixels,
        "mask_total_pixels": est.mask_total_pixels,
        "n_depth_pts": int(len(depth_pts)),
        "n_mask_pixels": int(n_mask_px),
        "depth_median_m": round(est.depth_median_m, 4),
        "scale_clamped": clamped,
        "clamp_range": list(SCALE_CLAMP),
        "plausible": plausible,
        "uniform_scale": True,
        "notes": (
            "General-object scale: largest mask CC, fused depth cues (trimmed median), "
            "mesh PCA-max (paired with depth). Uniform scale only. "
            "T5 FP refines pose; not for deformable/transparent objects."
        ),
        "warnings": warnings,
        "created_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "session_id": session_id,
    }
    if coarse_T_cam_mesh is not None:
        payload["coarse_T_cam_mesh_vis"] = coarse_T_cam_mesh
    return payload
