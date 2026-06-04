"""Offline round-trip unit tests for deploy/dp3_frames.py.

Pure numerical — no robot / pinocchio / partner env. Run:
    python deploy/test_dp3_frames.py        # self-contained runner
    pytest deploy/test_dp3_frames.py        # also pytest-discoverable

Verifies the §5 proprio (in) and §6 output-retarget (out) transforms compose to
identity, so the deployment frame math is internally consistent before any
real-robot / partner integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import dp3_frames as F

ATOL = 1e-9
RNG = np.random.default_rng(20260603)


# --- helpers ------------------------------------------------------------------
def _rand_se3(rng, *, max_rot_rad: float = np.pi, t_scale: float = 0.5) -> np.ndarray:
    """Random valid 4x4 SE3. max_rot_rad bounds the rotation magnitude."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(-max_rot_rad, max_rot_rad)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    T[:3, 3] = rng.uniform(-t_scale, t_scale, size=3)
    return T


def _rand_T_base_mesh(rng) -> np.ndarray:
    """T_base_mesh: rotation ~= I (<=0.02 rad residual, as partner T5 guarantees),
    translation = object origin in base (~ in front of the robot)."""
    T = _rand_se3(rng, max_rot_rad=0.02, t_scale=0.0)
    T[:3, 3] = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.7), rng.uniform(0.8, 1.0)])
    return T


def _assert_pose_close(A, B, msg, atol=ATOL):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    err = np.abs(A - B).max()
    assert err < atol, f"{msg}: max|Δ|={err:.2e} >= {atol:.1e}\nA=\n{A}\nB=\n{B}"


# --- tests --------------------------------------------------------------------
def test_quat_matrix_roundtrip():
    for _ in range(200):
        R = _rand_se3(RNG)[:3, :3]
        q = F.matrix_to_quat_wxyz(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9, "quat not unit"
        assert q[0] >= 0.0, "w should be canonicalized >= 0"
        _assert_pose_close(F.quat_wxyz_to_matrix(q), R, "quat<->matrix")


def test_pose_pos_quat_roundtrip():
    for _ in range(200):
        T = _rand_se3(RNG)
        pos, quat = F.pose_to_pos_quat(T)
        _assert_pose_close(F.pos_quat_to_pose(pos, quat), T, "pose<->pos/quat")


def test_base_G_roundtrip():
    for _ in range(200):
        T_base_mesh = _rand_T_base_mesh(RNG)
        T_base_X = _rand_se3(RNG)
        T_G = F.base_to_G(T_base_X, T_base_mesh)
        _assert_pose_close(F.G_to_base(T_G, T_base_mesh), T_base_X, "base<->G")


def test_pinch_dp3ee_roundtrip():
    for _ in range(200):
        T_pinch = _rand_se3(RNG)
        T_ee = F.pinch_to_dp3ee(T_pinch)
        _assert_pose_close(F.dp3ee_to_pinch(T_ee), T_pinch, "pinch<->dp3ee")
        # rotation unchanged; origin exactly EE_OFFSET behind along approach (col 2)
        _assert_pose_close(T_ee[:3, :3], T_pinch[:3, :3], "dp3ee rot == pinch rot")
        d = T_pinch[:3, 3] - T_ee[:3, 3]
        assert abs(np.linalg.norm(d) - F.EE_OFFSET_M) < 1e-9, "offset magnitude"
        approach = T_pinch[:3, F.APPROACH_COL]
        _assert_pose_close(d, F.EE_OFFSET_M * approach, "offset is +approach pinch-ee")


def test_pinch_baseee_roundtrip():
    for _ in range(200):
        T_pinch = _rand_se3(RNG)
        T_ee_pinch = _rand_se3(RNG)  # any valid SE3 cancels in the round trip
        T_Ree = F.pinch_to_base_ee(T_pinch, T_ee_pinch)
        _assert_pose_close(F.base_ee_to_pinch(T_Ree, T_ee_pinch), T_pinch, "pinch<->R_ee")


def test_full_pipeline_roundtrip():
    """The crux: proprio-extraction (§5) and output-retarget (§6) are inverses.

    A robot at pinch pose P, gripper g. Build proprio. Feed proprio[:7] back as a DP3
    action (same G-frame DP3-EE pose) through the output retarget to R_ee, then undo the
    R_ee bridge -> must recover the ORIGINAL pinch pose P. Gripper must pass through.
    """
    T_ee_pinch = _rand_se3(RNG)
    for _ in range(500):
        T_base_pinch = _rand_se3(RNG)
        T_base_mesh = _rand_T_base_mesh(RNG)
        g = float(RNG.integers(0, 2))

        proprio = F.pinch_to_proprio(T_base_pinch, T_base_mesh, g)
        assert proprio.shape == (8,)
        assert abs(np.linalg.norm(proprio[3:7]) - 1.0) < 1e-9, "proprio quat not unit"
        assert proprio[7] == g, "gripper passthrough (in)"

        # treat proprio pose as a DP3 action; retarget to R_ee, then undo bridge -> pinch
        action = np.concatenate([proprio[:7], [g]])
        T_base_Ree, g_out = F.action_to_base_ee(action, T_base_mesh, T_ee_pinch)
        T_base_pinch_rec = F.base_ee_to_pinch(T_base_Ree, T_ee_pinch)

        _assert_pose_close(T_base_pinch_rec, T_base_pinch, "FULL proprio->action->pinch")
        assert g_out == g, "gripper passthrough (out)"


def test_G_dp3ee_level_roundtrip():
    """Same as above but stop at the DP3-EE/base level (no R_ee bridge), tighter check."""
    for _ in range(500):
        T_base_pinch = _rand_se3(RNG)
        T_base_mesh = _rand_T_base_mesh(RNG)
        proprio = F.pinch_to_proprio(T_base_pinch, T_base_mesh, 0.0)
        T_G_ee = F.pos_quat_to_pose(proprio[:3], proprio[3:7])
        T_base_ee = F.G_to_base(T_G_ee, T_base_mesh)
        _assert_pose_close(F.dp3ee_to_pinch(T_base_ee), T_base_pinch, "G/dp3ee-level RT")


def test_with_real_ee_retarget_yaml():
    """If the partner ee_retarget.yaml is present, run the full RT with the REAL
    T_ee_pinch to confirm it is a valid SE3 and the math holds with real values."""
    yaml_path = Path("/home/accelerator/V2AP-demo/demo/phase2/calib/ee_retarget.yaml")
    if not yaml_path.exists():
        print("  [skip] real ee_retarget.yaml not found")
        return
    import yaml  # local import; optional dependency

    data = yaml.safe_load(yaml_path.read_text())
    T_ee_pinch = np.array(data["T_ee_pinch"], dtype=np.float64)
    assert T_ee_pinch.shape == (4, 4)
    _assert_pose_close(T_ee_pinch[3], np.array([0, 0, 0, 1.0]), "T_ee_pinch homog row")
    R = T_ee_pinch[:3, :3]
    _assert_pose_close(R @ R.T, np.eye(3), "T_ee_pinch R orthonormal", atol=1e-6)

    T_base_pinch = _rand_se3(RNG)
    T_base_mesh = _rand_T_base_mesh(RNG)
    proprio = F.pinch_to_proprio(T_base_pinch, T_base_mesh, 1.0)
    action = np.concatenate([proprio[:7], [1.0]])
    T_base_Ree, _ = F.action_to_base_ee(action, T_base_mesh, T_ee_pinch)
    T_rec = F.base_ee_to_pinch(T_base_Ree, T_ee_pinch)
    _assert_pose_close(T_rec, T_base_pinch, "real-yaml full RT")
    print(f"  [ok] real ee_retarget.yaml: T_ee_pinch t={T_ee_pinch[:3,3].round(4).tolist()}")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} test groups passed.")


if __name__ == "__main__":
    _main()
