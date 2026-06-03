"""Hard filters on PDM grasp poses (mesh / base-aligned frame)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from model.pdm.pose_codec import pose9_to_command

# Dexmate right-arm side approach sector in base XY (+Z up).
# Boundary rays: 45° from -X toward -Y, and 45° from +X toward -Y
# (bisectors of quadrants III and IV). Keeps the 90° wedge centered on -Y.
_APPROACH_SECTOR_LO_DEG = 225.0  # (-X,-Y) bisector
_APPROACH_SECTOR_HI_DEG = 315.0  # (+X,-Y) bisector


@dataclass(frozen=True)
class ApproachSectorFilterResult:
    kept: np.ndarray
    n_in: int
    n_kept: int
    n_rejected: int
    reject_reasons: dict[str, int]


def _normalize_angle_deg(theta: float) -> float:
    t = float(theta) % 360.0
    if t < 0.0:
        t += 360.0
    return t


def approach_xy_angle_deg(approach: np.ndarray) -> float:
    """CCW angle of approach projected onto XY, 0° = +X, 90° = +Y."""
    a = np.asarray(approach, dtype=np.float64).reshape(3)
    xy = a[:2]
    n = float(np.linalg.norm(xy))
    if n < 1e-12:
        return float("nan")
    return math.degrees(math.atan2(float(xy[1]), float(xy[0])))


def in_dexmate_right_approach_sector(
    direction: np.ndarray,
    *,
    min_horiz_norm: float = 0.25,
    lo_deg: float = _APPROACH_SECTOR_LO_DEG,
    hi_deg: float = _APPROACH_SECTOR_HI_DEG,
    vertical_z_min: float = 0.85,
    allow_vertical_top_down: bool = True,
) -> tuple[bool, str]:
    """
    Check gripper **arrival** direction in mesh/base (+Z up).

    - **Side reach:** XY projection in [225°, 315°] (-Y wedge for right arm).
    - **Top-down:** if ``||d_xy||`` is small but ``d_z >= vertical_z_min``, accept
      (gripper above object; approach ≈ −Z). The XY sector does not apply when
      the motion is dominated by vertical component.
    """
    a = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return False, "degenerate"
    a = a / n
    horiz = float(np.linalg.norm(a[:2]))
    if horiz < min_horiz_norm:
        if allow_vertical_top_down and float(a[2]) >= vertical_z_min:
            return True, "vertical_top_down"
        if horiz < 1e-3 and abs(float(a[2])) < vertical_z_min:
            return False, "degenerate"
        return False, "too_oblique_for_sector"
    ang = _normalize_angle_deg(approach_xy_angle_deg(a))
    lo = _normalize_angle_deg(lo_deg)
    hi = _normalize_angle_deg(hi_deg)
    if lo <= hi:
        ok = lo <= ang <= hi
    else:
        ok = ang >= lo or ang <= hi
    if not ok:
        return False, "outside_sector"
    return True, "ok"


def filter_poses9_dexmate_approach_sector(
    poses: np.ndarray,
    *,
    min_horiz_norm: float = 0.25,
    lo_deg: float = _APPROACH_SECTOR_LO_DEG,
    hi_deg: float = _APPROACH_SECTOR_HI_DEG,
    vertical_z_min: float = 0.85,
    allow_vertical_top_down: bool = True,
    use_arrival_direction: bool = True,
) -> ApproachSectorFilterResult:
    """
    Filter PDM pose9 rows; returns kept poses (N, 9) float32.

    ``use_arrival_direction=True`` (default): test **-approach** = direction the
    gripper comes from (into-object approach is roughly opposite, +Y wedge).
    """
    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 9)
    if len(poses) == 0:
        return ApproachSectorFilterResult(
            kept=poses,
            n_in=0,
            n_kept=0,
            n_rejected=0,
            reject_reasons={},
        )
    kept: list[np.ndarray] = []
    reasons: dict[str, int] = {}
    for p in poses:
        cmd = pose9_to_command(p)
        approach = cmd.rotation[:, 2]
        if use_arrival_direction:
            approach = -approach
        ok, why = in_dexmate_right_approach_sector(
            approach,
            min_horiz_norm=min_horiz_norm,
            lo_deg=lo_deg,
            hi_deg=hi_deg,
            vertical_z_min=vertical_z_min,
            allow_vertical_top_down=allow_vertical_top_down,
        )
        if ok:
            kept.append(p)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    if kept:
        out = np.stack(kept, axis=0).astype(np.float32)
    else:
        out = np.zeros((0, 9), dtype=np.float32)
    n_in = len(poses)
    n_kept = len(out)
    return ApproachSectorFilterResult(
        kept=out,
        n_in=n_in,
        n_kept=n_kept,
        n_rejected=n_in - n_kept,
        reject_reasons=reasons,
    )


def sector_filter_meta(
    result: ApproachSectorFilterResult,
    *,
    enabled: bool,
    min_horiz_norm: float,
    lo_deg: float,
    hi_deg: float,
    use_arrival_direction: bool = True,
    vertical_z_min: float = 0.85,
    allow_vertical_top_down: bool = True,
) -> dict:
    return {
        "enabled": enabled,
        "name": "dexmate_right_approach_xy_sector",
        "frame": "base_aligned_mesh",
        "vector": "neg_approach_column",
        "use_arrival_direction": True,
        "sector_lo_deg": lo_deg,
        "sector_hi_deg": hi_deg,
        "min_horiz_norm": min_horiz_norm,
        "vertical_z_min": vertical_z_min,
        "allow_vertical_top_down": allow_vertical_top_down,
        "notes": (
            "Side: 225°–315° on arrival (-approach) XY. Top-down: arrival +Z "
            f"(≥{vertical_z_min}) when ||arrival_xy|| small — approach ≈ −Z."
        ),
        "n_in": result.n_in,
        "n_kept": result.n_kept,
        "n_rejected": result.n_rejected,
        "reject_reasons": result.reject_reasons,
    }
