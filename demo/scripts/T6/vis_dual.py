"""Side-by-side T6 visualization: mesh-frame PDM + scene RGB."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from model.pdm.visualize import save_candidate_overlay

from scene_vis import render_grasp_scene_bgr  # noqa: E402


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _label_bar(width: int, text: str, *, bar_h: int = 28) -> np.ndarray:
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 44)
    cv2.putText(
        bar,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return bar


def stitch_panels(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    gap: int = 8,
    left_label: str = "mesh frame (PDM)",
    right_label: str = "camera RGB (T_cam_mesh)",
) -> np.ndarray:
    target_h = max(left_bgr.shape[0], right_bgr.shape[0])
    left = _resize_to_height(left_bgr, target_h)
    right = _resize_to_height(right_bgr, target_h)
    sep = np.full((target_h, gap, 3), 32, dtype=np.uint8)
    row = np.hstack([left, sep, right])
    bar = _label_bar(
        row.shape[1],
        f"  {left_label}  |  {right_label}",
    )
    return np.vstack([bar, row])


def save_t6_dual_vis(
    dirs,
    h5_path: Path,
    mesh_points: np.ndarray,
    affordance_norm: np.ndarray | None,
    out_path: Path,
    *,
    mesh_top: int = 20,
    scene_top: int = 10,
) -> Path:
    """Write single PNG: left = mesh PDM overlay, right = grasp on session RGB."""
    h5_path = Path(h5_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="t6_vis_") as tmp:
        mesh_png = Path(tmp) / "mesh_panel.png"
        save_candidate_overlay(
            str(h5_path),
            mesh_points,
            str(mesh_png),
            top=mesh_top,
            affordance=affordance_norm,
            affordance_vmax_fixed=1.0,
            title_suffix="",
        )
        mesh_bgr = cv2.imread(str(mesh_png), cv2.IMREAD_COLOR)
        if mesh_bgr is None:
            raise RuntimeError(f"Failed to read mesh panel: {mesh_png}")

    scene_bgr = render_grasp_scene_bgr(
        dirs,
        h5_path,
        top=scene_top,
    )
    combined = stitch_panels(mesh_bgr, scene_bgr)
    cv2.imwrite(str(out_path), combined)
    return out_path
