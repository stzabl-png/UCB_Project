"""DP3 real-robot deployment — coordinate-frame transforms (pure numerical).

Single source of truth for the frame math in deploy/DP3_DEPLOY_LOGIC.md (§5 proprio,
§6 output retarget). Operates ONLY on 4x4 homogeneous matrices and (pos, quat) arrays —
NO pinocchio / robot / partner-env dependency, so it is fully offline-unit-testable.

Frames (all transforms are ``p_dst = T_dst_src @ p_src``, column-vector convention):
  base   : Dexmate Vega base. Z = up = gravity.
  mesh   : SAM3D object mesh, T5-aligned so its axes are parallel to base
           (object_base_aligned.glb). ``T_base_mesh`` has rotation ~= I.
  G      : DP3 object-centric frame = the mesh frame (origin at object, axes || gravity).
           base <-> G is just inv(T_base_mesh) / T_base_mesh.
  pinch  : UCB virtual two-finger frame. Columns [finger_open, y_body, approach],
           origin at thumb/index-tip midpoint. (= partner virtual_pinch_frame_in_base.)
  ee     : DP3's EE frame. Orientation == pinch orientation; origin == pinch - 0.10*approach.
  R_ee   : razer URDF Pinocchio end frame; the Pink-IK target.

Conventions (must match training, build_gt_replay.py):
  EE_OFFSET_M = 0.10   (build_gt_replay.py:54; DP3 EE sits this far behind pinch midpoint)
  approach    = column 2 of the rotation (0-based).
  quaternion  = wxyz order in the DP3 proprio/action 8-vector (scipy uses xyzw internally).

Proprio / action 8-vector layout (both identical):
  [ pos(3), quat_wxyz(4), gripper(1) ]   in the G/object frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# --- constants (training-locked; see DP3_DEPLOY_LOGIC.md §9.4) -----------------
EE_OFFSET_M: float = 0.10
APPROACH_COL: int = 2


# --- quaternion <-> matrix (wxyz public, xyzw internal via scipy) --------------
def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion -> 3x3 rotation."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return Rotation.from_quat(q_xyzw).as_matrix()


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> wxyz unit quaternion (w >= 0 for a canonical sign)."""
    q_xyzw = Rotation.from_matrix(np.asarray(R, dtype=np.float64)[:3, :3]).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if q_wxyz[0] < 0.0:  # quaternion double-cover: pick w>=0 representative
        q_wxyz = -q_wxyz
    return q_wxyz


# --- pose <-> (pos, quat_wxyz) -------------------------------------------------
def pose_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 -> (pos(3), quat_wxyz(4))."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), matrix_to_quat_wxyz(T[:3, :3])


def pos_quat_to_pose(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(pos(3), quat_wxyz(4)) -> 4x4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


# --- base <-> G (= mesh) -------------------------------------------------------
def base_to_G(T_base_X: np.ndarray, T_base_mesh: np.ndarray) -> np.ndarray:
    """Express a base-frame pose in the G/object frame: T_G_X = inv(T_base_mesh) @ T_base_X."""
    return np.linalg.inv(np.asarray(T_base_mesh, dtype=np.float64)) @ np.asarray(
        T_base_X, dtype=np.float64
    )


def G_to_base(T_G_X: np.ndarray, T_base_mesh: np.ndarray) -> np.ndarray:
    """Lift a G/object-frame pose into base: T_base_X = T_base_mesh @ T_G_X."""
    return np.asarray(T_base_mesh, dtype=np.float64) @ np.asarray(T_G_X, dtype=np.float64)


# --- pinch <-> DP3 EE (the 0.10 m approach back-off) ---------------------------
def pinch_to_dp3ee(T_base_pinch: np.ndarray, ee_offset_m: float = EE_OFFSET_M) -> np.ndarray:
    """pinch frame -> DP3 EE frame: same rotation, origin moved -offset along approach (col 2)."""
    T = np.asarray(T_base_pinch, dtype=np.float64).copy()
    approach = T[:3, APPROACH_COL]
    T[:3, 3] = T[:3, 3] - float(ee_offset_m) * approach
    return T


def dp3ee_to_pinch(T_base_ee: np.ndarray, ee_offset_m: float = EE_OFFSET_M) -> np.ndarray:
    """DP3 EE frame -> pinch frame: same rotation, origin moved +offset along approach (col 2)."""
    T = np.asarray(T_base_ee, dtype=np.float64).copy()
    approach = T[:3, APPROACH_COL]
    T[:3, 3] = T[:3, 3] + float(ee_offset_m) * approach
    return T


# --- pinch <-> R_ee (partner T_ee_pinch_closed bridge; ee_retarget.yaml) -------
def pinch_to_base_ee(T_base_pinch: np.ndarray, T_ee_pinch_closed: np.ndarray) -> np.ndarray:
    """T_base_Ree = T_base_pinch @ inv(T_ee_pinch_closed).

    Mirrors partner hand_retarget_geometry.base_ee_from_virtual_pinch_closed.
    """
    return np.asarray(T_base_pinch, dtype=np.float64) @ np.linalg.inv(
        np.asarray(T_ee_pinch_closed, dtype=np.float64)
    )


def base_ee_to_pinch(T_base_ee: np.ndarray, T_ee_pinch_closed: np.ndarray) -> np.ndarray:
    """Inverse of pinch_to_base_ee: T_base_pinch = T_base_Ree @ T_ee_pinch_closed."""
    return np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
        T_ee_pinch_closed, dtype=np.float64
    )


# === high-level: proprio (in) and output retarget (out) =======================
def pinch_to_proprio(
    T_base_pinch: np.ndarray,
    T_base_mesh: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """§5 — current robot pinch frame (from FK) -> DP3 8-D proprio in G frame.

    Args:
        T_base_pinch: 4x4, from partner virtual_pinch_frame_in_base(FK).
        T_base_mesh:  4x4, register/T_base_mesh.json.
        gripper:      0.0 open / 1.0 closed.
    Returns:
        agent_pos (8,) = [pos_G(3), quat_G_wxyz(4), gripper(1)].
    """
    T_base_ee = pinch_to_dp3ee(T_base_pinch)
    T_G_ee = base_to_G(T_base_ee, T_base_mesh)
    pos, quat = pose_to_pos_quat(T_G_ee)
    out = np.empty(8, dtype=np.float64)
    out[:3] = pos
    out[3:7] = quat
    out[7] = float(gripper)
    return out


def dp3ee_pose_in_G_to_base_ee(
    pos_G: np.ndarray,
    quat_G_wxyz: np.ndarray,
    T_base_mesh: np.ndarray,
    T_ee_pinch_closed: np.ndarray,
) -> np.ndarray:
    """§6 — one DP3 action EE pose (G frame) -> R_ee target in base (for Pink IK)."""
    T_G_ee = pos_quat_to_pose(pos_G, quat_G_wxyz)
    T_base_ee = G_to_base(T_G_ee, T_base_mesh)
    T_base_pinch = dp3ee_to_pinch(T_base_ee)
    return pinch_to_base_ee(T_base_pinch, T_ee_pinch_closed)


def action_to_base_ee(
    action8: np.ndarray,
    T_base_mesh: np.ndarray,
    T_ee_pinch_closed: np.ndarray,
) -> tuple[np.ndarray, float]:
    """§6 — full DP3 8-vector action -> (T_base_Ree 4x4, gripper float)."""
    a = np.asarray(action8, dtype=np.float64).reshape(8)
    T_base_ee = dp3ee_pose_in_G_to_base_ee(a[:3], a[3:7], T_base_mesh, T_ee_pinch_closed)
    return T_base_ee, float(a[7])
