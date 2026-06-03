"""Project T6 grasp candidates onto session RGB (real capture)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from model.pdm.pose_codec import TCP_OFFSET
from model.pdm.visualize import load_candidates

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
_T4_DIR = _SCRIPTS_ROOT / "T4"
if str(_T4_DIR) not in sys.path:
    sys.path.insert(0, str(_T4_DIR))
from scale_common import load_K, load_mask_bool, project_cam_to_image  # noqa: E402

from _session_io import SessionDirs  # noqa: E402


def _bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color_rgb[2], color_rgb[1], color_rgb[0])


def _mesh_to_cam(pts_mesh: np.ndarray, T_cam_mesh: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cam_mesh, dtype=np.float64).reshape(4, 4)
    ones = np.ones((len(pts_mesh), 1), dtype=np.float64)
    return (T @ np.hstack([pts_mesh, ones]).T).T[:, :3]


def _gripper_keypoints_mesh(cand: dict) -> dict[str, np.ndarray]:
    """Same geometry as ``model.pdm.visualize.draw_gripper`` (mesh frame)."""
    if "grasp_point" in cand:
        pos = np.asarray(cand["grasp_point"], dtype=np.float64).reshape(3)
    else:
        pos = np.asarray(cand["position"], dtype=np.float64).reshape(3)
    R = np.asarray(cand["rotation"], dtype=np.float64).reshape(3, 3)
    app = R[:, 2]
    app = app / (np.linalg.norm(app) + 1e-8)
    fing = R[:, 0]
    fing = fing / (np.linalg.norm(fing) + 1e-8)
    width = float(cand.get("gripper_width", cand.get("width", 0.06)))
    hw = width / 2.0
    finger_depth = max(width * 0.55, 0.025)
    tip_l = pos - hw * fing
    tip_r = pos + hw * fing
    tip_l_back = tip_l - app * finger_depth
    tip_r_back = tip_r - app * finger_depth
    wrist = pos - app * float(TCP_OFFSET)
    app_tip = pos + app * 0.04
    return {
        "center": pos,
        "tip_l": tip_l,
        "tip_r": tip_r,
        "tip_l_back": tip_l_back,
        "tip_r_back": tip_r_back,
        "wrist": wrist,
        "app_tip": app_tip,
    }


def _project_polyline(
    pts_cam: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    u, v, ok = project_cam_to_image(pts_cam, K)
    out: list[tuple[int, int]] = []
    for i in range(len(pts_cam)):
        if not ok[i]:
            continue
        ui, vi = int(round(u[i])), int(round(v[i]))
        if 0 <= ui < W and 0 <= vi < H:
            out.append((ui, vi))
    return out


def _draw_polyline(img: np.ndarray, uv: list[tuple[int, int]], color, thickness: int) -> None:
    if len(uv) < 2:
        return
    pts = np.array(uv, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_gripper_on_image(
    img_bgr: np.ndarray,
    cand: dict,
    T_cam_mesh: np.ndarray,
    K: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    H, W = img_bgr.shape[:2]
    kp_mesh = _gripper_keypoints_mesh(cand)
    kp_cam = {k: _mesh_to_cam(v.reshape(1, 3), T_cam_mesh)[0] for k, v in kp_mesh.items()}

    def seg(a: str, b: str) -> list[tuple[int, int]]:
        return _project_polyline(
            np.stack([kp_cam[a], kp_cam[b]], axis=0), K, H, W
        )

    for a, b in (
        ("tip_l", "tip_r"),
        ("tip_l", "tip_l_back"),
        ("tip_r", "tip_r_back"),
        ("tip_l_back", "tip_r_back"),
    ):
        _draw_polyline(img_bgr, seg(a, b), color_bgr, thickness)

    wrist_uv = _project_polyline(np.stack([kp_cam["wrist"], kp_cam["center"]], 0), K, H, W)
    _draw_polyline(img_bgr, wrist_uv, color_bgr, 1)

    app_uv = _project_polyline(np.stack([kp_cam["center"], kp_cam["app_tip"]], 0), K, H, W)
    _draw_polyline(img_bgr, app_uv, (0, 220, 255), max(1, thickness - 1))


def _rank_color(rank: int, n: int) -> tuple[int, int, int]:
    if rank == 0:
        return (60, 220, 80)
    hue = int(180 * (rank + 1) / max(n, 1)) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_grasp_scene_bgr(
    dirs: SessionDirs,
    h5_path: Path,
    *,
    top: int = 10,
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """
    RGB + mask + gripper wireframes only (no center dots, no per-candidate labels).
    """
    h5_path = Path(h5_path).resolve()
    rgb_path = dirs.input_rel("rgb", "left_rgb.png")
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)

    reg_cam = dirs.output_rel("register", "T_cam_mesh.json")
    if not reg_cam.is_file():
        raise FileNotFoundError(reg_cam)
    T_cam_mesh = np.asarray(
        json.loads(reg_cam.read_text())["T_cam_mesh"], dtype=np.float64
    ).reshape(4, 4)

    K = load_K(dirs.input_dir)
    base = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(rgb_path)
    H, W = base.shape[:2]

    mask_path = dirs.output_rel("segment", "mask.png")
    if mask_path.is_file():
        m = load_mask_bool(mask_path)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        tint = np.zeros_like(base)
        tint[:] = (80, 200, 120)
        blend = base.astype(np.float32)
        blend[m] = blend[m] * (1.0 - mask_alpha) + tint[m].astype(np.float32) * mask_alpha
        base = np.clip(blend, 0, 255).astype(np.uint8)

    cands = load_candidates(str(h5_path), top=top)
    if not cands:
        raise RuntimeError(f"No candidates in {h5_path}")

    panel = base.copy()
    for i, cand in enumerate(cands):
        draw_gripper_on_image(
            panel,
            {
                "grasp_point": cand["position"],
                "rotation": cand["rotation"],
                "gripper_width": cand["width"],
            },
            T_cam_mesh,
            K,
            _rank_color(i, len(cands)),
            thickness=2 if i == 0 else 1,
        )
    return panel


def save_grasp_candidates_scene_vis(
    dirs: SessionDirs,
    h5_path: Path,
    out_path: Path,
    *,
    top: int = 10,
    mask_alpha: float = 0.35,
) -> Path:
    """Write scene panel BGR to ``out_path`` (legacy single-panel export)."""
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), render_grasp_scene_bgr(dirs, h5_path, top=top, mask_alpha=mask_alpha))
    return out_path
