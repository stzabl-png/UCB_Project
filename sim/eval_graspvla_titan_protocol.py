#!/usr/bin/env python3
"""
sim/eval_graspvla_titan_protocol.py
====================================
Closed-loop GraspVLA evaluation on the SAME titan-protocol scene as DP3.

Scene
-----
IDENTICAL to sim/eval_dp3_titan_protocol.py:
  - Franka at ROBOT_POSITION / ROBOT_ORIENTATION.
  - Table + ground.
  - Object placed at OBJECT_XY with identity quat + z_offset_m from titan
    SAM3D _meta.json (partner "common-sense placement").
  - 100-step free settle before policy starts.
  - 2 LIBERO-style RGB cameras (front + side) spawned at robot-base-relative
    positions and rotated through robot's yaw=90° into world.

Rollout (closed loop, base-frame delta actions)
-----
Per VLA chunk:
  1. Read EE pose in world → transform to panda_link0 (Franka base) frame.
  2. Convert quat → extrinsic XYZ euler (transforms3d 'sxyz').
  3. Build 7D proprio [x,y,z,roll,pitch,yaw,gripper]. Gripper: +1=open, -1=close.
  4. Append to proprio history buffer (length 4; server uses [-4] and [-1]).
  5. Render front + side cameras → (256, 256, 3) uint8.
  6. ZMQ ROUTER send {text, front_view_image, side_view_image, proprio_array}.
  7. Receive (16, 7) delta sequence in panda_link0 frame.
  8. For each delta (Δx,Δy,Δz,Δr,Δp,Δy,grip):
       - new_pos_R    = cur_pos_R + delta[:3]
       - new_rotmat_R = euler2mat(delta[3:6]) @ cur_rotmat_R    # LEFT-multiply
       - Transform new_pos_R / new_rotmat_R back to world frame.
  9. cuRobo IK chain on the world-frame waypoints (same path as DP3).
 10. Drive Franka through joint waypoints with set_joint_positions / PD.
 11. First grip < 0 (CLOSE signal) crossing → close_gripper + 80 settle.
 12. Stop on early-success (dz > 3 cm) or after --max-chunks.

After rollout: close + lift via cuRobo plan_grasp (re-used from DP3 path) if the
policy emitted a grip-close signal. Success: dz > 0.03 m.

Independence from DP3 path
-----
- No HTTP / no PC sampling / no titan PC sampler import.
- Re-uses: scene setup, cuRobo IK helpers, close+lift, frame helpers — all are
  policy-agnostic.
- Same titan-protocol placement so paper-comparable to DP3 + partner main method.

Usage
-----
First start GraspVLA inference server (separate `graspvla_env` env):
    cd third_party/GraspVLA
    conda activate graspvla_env
    python -m vla_network.scripts.serve --port 6666 --path <ckpt.safetensors>

Then in env_isaaclab:
    /home/accelerator/miniforge3/envs/env_isaaclab/bin/python \\
        sim/eval_graspvla_titan_protocol.py \\
        --episodes-glob /tmp/synth_zero_shot/A02029.hdf5 \\
        --n-rollouts 1 --headless \\
        --titan-protocol \\
        --titan-usd-dir /home/accelerator/UCB_Project_titan/output/obj_usd/oakink \\
        --server-addr tcp://127.0.0.1:6666 \\
        --result-dir output/eval_graspvla_titan_protocol/smoke
"""
from isaacsim import SimulationApp                     # MUST be first
import argparse, os, sys, glob, json, time, subprocess, shutil, pickle
import numpy as np
import h5py
from termcolor import cprint
from scipy.spatial.transform import Rotation

SIM_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(SIM_DIR)
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, PROJ_ROOT)

parser = argparse.ArgumentParser(description="GraspVLA titan-protocol eval (gate3 sim)")
parser.add_argument("--episodes-glob", type=str,
                    default=os.path.join(PROJ_ROOT, "Baseline1/data/episodes_b3_curobo/*.hdf5"),
                    help="glob of synth/training hdf5 — only obj_id metadata is used; placement "
                         "is overridden by --titan-protocol")
parser.add_argument("--n-rollouts", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-chunks", type=int, default=5,
                    help="receding-horizon chunks. Each chunk = N action_steps (=16) deltas.")
# ★ GraspVLA server ★
parser.add_argument("--server-addr", type=str, default="tcp://127.0.0.1:6666",
                    help="GraspVLA ZMQ ROUTER server")
parser.add_argument("--instruction", type=str, default="pick up the object",
                    help="natural-language instruction passed to VLA. Generic by default; "
                         "can be overridden per-obj for ablation.")
parser.add_argument("--request-timeout-ms", type=int, default=60000)
parser.add_argument("--proprio-history", type=int, default=4,
                    help="length of proprio history buffer (server uses [-4] and [-1])")
# ★ camera setup ★
parser.add_argument("--img-h", type=int, default=256)
parser.add_argument("--img-w", type=int, default=256)
parser.add_argument("--cam-warmup", type=int, default=10,
                    help="N world.step()s after camera spawn before first render (replicator warmup)")
parser.add_argument("--extended-finger", action="store_true",
                    help="if set, apply REAL_EEF_TO_SIM_EEF = +3cm Z offset to EE for proprio "
                         "(only true for the extended-finger Franka build).")
parser.add_argument("--ee-offset-m", type=float, default=None,
                    help="Override total panda_hand → EE z offset (meters). "
                         "Default: 0.1034 (standard Franka panda_EE TCP). "
                         "Use 0.08 to mimic GraspVLA-playground URDF eef_link. "
                         "When set, --extended-finger is ignored.")
# ★ general ★
parser.add_argument("--headless", action="store_true")
parser.add_argument("--video", type=str, default=None,
                    help="if set, capture per-episode mp4s into this dir (only successful kept)")
parser.add_argument("--video-every", type=int, default=3)
parser.add_argument("--video-all", action="store_true")
parser.add_argument("--result-dir", type=str,
                    default=os.path.join(PROJ_ROOT, "output/eval_graspvla_titan_protocol/smoke"))
parser.add_argument("--retry-physx", type=int, default=1,
                    help="N retries when an ep returns 'physx_corrupt'. Default 1.")
# ★ titan protocol overrides ★
parser.add_argument("--titan-protocol", action="store_true", default=False,
                    help="ENABLE titan placement: identity quat + titan z_offset (canonical).")
parser.add_argument("--titan-usd-dir",
                    default="/home/accelerator/UCB_Project_titan/output/obj_usd/oakink",
                    help="dir with titan SAM3D USDs + {obj_id}_meta.json")
parser.add_argument("--titan-mesh-root",
                    default="/home/accelerator/UCB_Project_titan/data_hub/meshes/SAM3DMesh/rotated_mesh",
                    help="(unused for VLA — kept for CLI compatibility with DP3 path)")
parser.add_argument("--mass-override", type=float, default=None,
                    help="If set, overrides the default hardcoded mass (0.05 kg) for the object.")
parser.add_argument("--object-xy-override", type=str, default=None,
                    help="Override OBJECT_XY (world frame). Format: 'x,y'. Default is gate3's "
                         "(0.0, 0.55) which puts obj at robot-frame Y=+0.2 (off-center left). "
                         "GraspVLA workspace is centered at robot-frame Y=0; use --object-xy-override "
                         "'0.2,0.55' to put obj at robot-frame Y=0 (workspace center).")
parser.add_argument("--render-method", choices=["replicator", "viewport"], default="viewport",
                    help="RGB capture: 'replicator' = rep.render_product+annotator (faster but "
                         "lower-quality default rendering); 'viewport' = set_camera_view + "
                         "capture_viewport_to_file (high-quality RTX default, ~2x slower).")
parser.add_argument("--use-red-cube", action="store_true",
                    help="Replace USD-loaded obj with a high-contrast bright RED DynamicCuboid "
                         "(8 cm by default). Useful for ideal-visibility ablation: removes "
                         "domain-shift from low-contrast SAM3D textures.")
parser.add_argument("--red-cube-size", type=float, default=0.08,
                    help="Edge length of red cube in meters (only if --use-red-cube).")
parser.add_argument("--home-joints", type=str, default=None,
                    help="Override HOME arm joints (7 comma-separated). Move the start EE "
                         "pose into the GraspVLA training proprio box (robot x>=0.323).")
parser.add_argument("--obj-color-override", type=str, default=None,
                    help="Repaint the USD-loaded obj with this RGB color (0..1, format 'r,g,b'). "
                         "Use e.g. '1,0,0' to make any SAM3D obj bright red — for high-contrast "
                         "ablation when default obj rendering is too dark/low-contrast.")
parser.add_argument("--lift-mode", choices=["vla", "curobo"], default="curobo",
                    help="What drives the LIFT after VLA emits close: "
                         "'vla' = let model deltas continue → IK → PD (can spike PhysX), "
                         "'curobo' = call solve_plan_grasp at close-fire EE pose, execute "
                         "the planned grasp+lift segments (smooth, collision-aware, "
                         "matches DP3 path).")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

# now IsaacSim is up, we can import everything else
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep
# NOTE: isaacsim.sensors.camera.Camera has a rendering bug (returns near-uniform
# image even when scene is fully visible — std≈4 vs replicator's std≈41 at same
# pose). Use rep.create.camera + rep.create.render_product + AnnotatorRegistry
# instead (this was isolated via a standalone camera-method bug test).

from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject
import grasp_physics

# ZMQ client for GraspVLA server
import zmq

# ============================================================
# Constants — IDENTICAL to sim/eval_dp3_titan_protocol.py for placement parity
# ============================================================
ROBOT_POSITION    = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION    = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE       = [2.0, 2.0, 0.1]
TABLE_TOP_Z       = 0.80
_DEFAULT_OBJECT_XY = np.array([0.0, 0.55])  # default gate3 (puts obj off-center for VLA)
if args.object_xy_override:
    _xy = [float(v) for v in args.object_xy_override.split(",")]
    OBJECT_XY = np.array(_xy[:2])
else:
    OBJECT_XY = _DEFAULT_OBJECT_XY
LIFT_HEIGHT       = 0.15
PREGRASP_BACKOFF  = 0.12

HOME_JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
# --home-joints CLI override (7 arm joints, comma-separated). Used to move the
# starting EE pose INSIDE the GraspVLA training proprio box (robot-frame
# x∈[0.323,0.773], z∈[-0.071,0.493]); the default HOME starts at x=0.307 (1.6cm
# below the box → chunk-0 proprio is OOD).
if getattr(args, "home_joints", None):
    _hj = [float(v) for v in args.home_joints.split(",")]
    assert len(_hj) == 7, "--home-joints needs 7 comma-separated arm joint values"
    HOME_JOINTS = np.array(_hj + [0.04, 0.04])
IK_POS_TOL, IK_ORI_TOL, IK_SEEDS = 0.005, 0.05, 1024
CIK_SCRIPT  = os.path.join(SIM_DIR, "curobo_ik.py")
CPLAN_SCRIPT = os.path.join(SIM_DIR, "curobo_plan.py")
PLAN_POS_TOL, PLAN_ORI_TOL = 0.01, 0.10

# Eval-only
SUCCESS_DZ_M    = 0.03

# ── GraspVLA-specific (OFFICIAL real-world calibration spec) ──
# Source: GraspVLA-real-world-controller README "Validate Alignment" section, with
# coordinate frame = ROBOT BASE (panda_link0). +X = robot forward, +Z = up, +Y by RHR.
# These specs are EQUIVALENT to LIBERO playground's cameras when the robot is
# table-top-mounted (robot base z = table top z), which matches our gate3 setup.
LIBERO_FOVY_DEG          = 43.0
GRASPVLA_FRONT_POS_R     = np.array([1.35, 0.00, 0.53], dtype=np.float64)
GRASPVLA_FRONT_LOOKAT_R  = np.array([0.20, 0.00, 0.00], dtype=np.float64)
GRASPVLA_SIDE_POS_R      = np.array([0.50, 0.69, 0.50], dtype=np.float64)
GRASPVLA_SIDE_LOOKAT_R   = np.array([0.50, 0.00, 0.10], dtype=np.float64)
# Render aspect: SQUARE 1:1 (matches LIBERO MuJoCo playground 256×256 + real-world
# RealSense's center-crop to 480×480 → resize to 256×256). Direct render at
# 640×480 then resize to 256×256 squashes aspect 4:3 → 1:1 (horizontal -25%),
# which mismatches model training. Use square render to avoid this.
# Camera render resolution: 640×480 so internal DLSS render dim ≥ 300 (the min).
# After render, we downsample to (img_h, img_w) for VLA server input.
CAM_RENDER_W, CAM_RENDER_H = 480, 480   # SQUARE 1:1 → matches model training aspect
# Whether to vertically flip captured images before sending to VLA server.
# agent.py uses obs[...][::-1] because MuJoCo image origin is bottom-left and model
# expects top-left. IsaacSim already gives top-left origin → DEFAULT FALSE.
FLIP_IMAGE_VERTICAL = False
# GraspVLA EE offset — depends on finger setup. Model is robust across these
# conventions (per real-world README: "works robustly with both original and
# extended fingers"). For standard (non-extended) Franka, use the real-world
# ORIGINAL finger convention: proprio EE = panda_EE = panda_hand + 0.1034 m
# (standard Franka TCP). See third_party/GraspVLA-real-world-controller/
# vla_client/controllers/franka_ros_controller.py lines 30-47 + 282:
#     EEF_FRAME_ID = 'panda_EE'  # real ROS frame is panda_EE
#     REAL_EEF_TO_SIM_EEF = identity (no shift) for original finger
#     sim_proprio_EE = panda_EE_pose @ REAL_EEF_TO_SIM_EEF
# Playground LIBERO URDF uses a custom eef_link at +0.08 m (different convention,
# only valid when using franka_with_extended_finger URDF). We use the real-world
# ORIGINAL-finger convention since our gate3 Franka has standard fingers.
PANDA_HAND_TO_EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
# REAL_EEF_TO_SIM_EEF: identity for standard Franka, +3 cm if --extended-finger
REAL_EEF_TO_SIM_EEF = np.array([0.0, 0.0, 0.03], dtype=np.float64)
# If --ee-offset-m is set, override the entire hand→EE shift to that value.
# Used to mimic GraspVLA-playground URDF eef_link (0.08 m) for ablation.
if args.ee_offset_m is not None:
    PANDA_HAND_TO_EE_OFFSET = np.array([0.0, 0.0, float(args.ee_offset_m)], dtype=np.float64)
    REAL_EEF_TO_SIM_EEF = np.array([0.0, 0.0, 0.0], dtype=np.float64)

_VID = {"on": False, "i": 0, "n": 0, "vp": None}
_VID_FRAMES = f"/tmp/vlaeval_video_frames_{os.getpid()}"


# ============================================================
# Frame helpers (same Franka convention as DP3 path)
# ============================================================
def _xyzw(q_wxyz): return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
def _wxyz(q_xyzw): return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

_FRANKA_FIX = Rotation.from_euler("z", -45, degrees=True)  # same as DP3 path

def retarget_to_franka_quat(q_wxyz):
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _FRANKA_FIX).as_quat())

def franka_to_retarget_quat(q_wxyz):
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _FRANKA_FIX.inv()).as_quat())

def read_panda_hand_pose(stage):
    """panda_hand world pose → (pos(3), quat_wxyz(4)) Franka convention."""
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_hand"))
    M = xf.ComputeLocalToWorldTransform(0)
    t = np.array(M.ExtractTranslation(), dtype=np.float64)
    q = M.ExtractRotationQuat()
    im = q.GetImaginary()
    return t, np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64)


# ============================================================
# panda_link0 (robot base) frame transforms
# ============================================================
def T_world_from_robot():
    """4x4 transform: a point in panda_link0 frame → world frame."""
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_wr
    T[:3,  3] = np.array(ROBOT_POSITION, dtype=np.float64)
    return T

def T_robot_from_world():
    """4x4 transform: a point in world → panda_link0 frame."""
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_wr.T
    T[:3,  3] = -R_wr.T @ np.array(ROBOT_POSITION, dtype=np.float64)
    return T

def pose_world_to_robot(pos_w, quat_w_wxyz):
    """(pos, quat) in world → (pos, quat) in panda_link0. Quat: Franka convention wxyz."""
    Trw = T_robot_from_world()
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    pos_r = (Trw @ np.append(np.asarray(pos_w, dtype=np.float64), 1.0))[:3]
    Rw = Rotation.from_quat(_xyzw(quat_w_wxyz)).as_matrix()
    Rr = R_wr.T @ Rw
    quat_r_wxyz = _wxyz(Rotation.from_matrix(Rr).as_quat())
    return pos_r, quat_r_wxyz

def pose_robot_to_world(pos_r, quat_r_wxyz):
    """(pos, quat) in panda_link0 → (pos, quat) in world."""
    Twr = T_world_from_robot()
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    pos_w = (Twr @ np.append(np.asarray(pos_r, dtype=np.float64), 1.0))[:3]
    Rr = Rotation.from_quat(_xyzw(quat_r_wxyz)).as_matrix()
    Rw = R_wr @ Rr
    quat_w_wxyz = _wxyz(Rotation.from_matrix(Rw).as_quat())
    return pos_w, quat_w_wxyz


# ============================================================
# Euler conventions — GraspVLA uses transforms3d 'sxyz' (extrinsic XYZ)
# Mapping: scipy "XYZ" (uppercase) = intrinsic XYZ = extrinsic ZYX
#          scipy "xyz" (lowercase) = extrinsic XYZ = transforms3d 'sxyz' ✓
# ============================================================
def quat_wxyz_to_euler_sxyz(quat_wxyz):
    """quat (Franka wxyz) → (roll, pitch, yaw) extrinsic XYZ, radians."""
    return Rotation.from_quat(_xyzw(quat_wxyz)).as_euler("xyz", degrees=False)

def euler_sxyz_to_mat(roll, pitch, yaw):
    """extrinsic XYZ euler (rad) → 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

def mat_to_quat_wxyz(R_mat):
    return _wxyz(Rotation.from_matrix(R_mat).as_quat())


# ============================================================
# cuRobo IK (out-of-process) — IDENTICAL to DP3 path
# ============================================================
def solve_ik_chain(waypoints, start_qpos=None):
    """waypoints: list of (pos_w(3), quat_franka_wxyz(4)). Returns (qpos, ok_mask).
    Uses curobo_ik.py --solve NPZ interface (matches DP3 path exactly)."""
    pos = np.array([w[0] for w in waypoints], dtype=np.float64)
    quat = np.array([w[1] for w in waypoints], dtype=np.float64)
    tag = f"/tmp/vlaeval_cik_{os.getpid()}"
    fin, fout = tag + "_in.npz", tag + "_out.npz"
    save_kw = dict(pos=pos, quat=quat,
                   robot_pos=np.array(ROBOT_POSITION, dtype=np.float64),
                   robot_ori=np.array(ROBOT_ORIENTATION, dtype=np.float64),
                   num_seeds=IK_SEEDS, pos_tol=IK_POS_TOL, ori_tol=IK_ORI_TOL)
    if start_qpos is not None:
        save_kw["start_qpos"] = np.asarray(start_qpos, dtype=np.float64).reshape(7)
    np.savez(fin, **save_kw)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, CIK_SCRIPT, "--solve", fin, fout],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(fout):
        cprint(f"  ❌ curobo_ik failed (rc={r.returncode}): {r.stderr[-300:]}", "red")
        return None, None
    d = np.load(fout)
    return d["qpos"], d["ok"].astype(bool)

def solve_single_ik(pos_world, quat_franka_wxyz):
    qpos, ok = solve_ik_chain([(pos_world, quat_franka_wxyz)])
    if qpos is None or not bool(ok[0]):
        return None
    return qpos[0]


# ============================================================
# cuRobo plan_grasp (for close+lift) — IDENTICAL to DP3 path
# ============================================================
def _T_robot_world():
    R_rw = Rotation.from_euler("z", -ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_rw
    T[:3,  3] = -R_rw @ np.array(ROBOT_POSITION, dtype=np.float64)
    return T

def _build_plan_world_dict(scene, obj_pos_w, obj_quat):
    from curobo_world import build_world_config_dict, object_pose_robot_frame
    T = _T_robot_world()
    table_center_w  = np.array([TABLE_POSITION[0], TABLE_POSITION[1], TABLE_TOP_Z - TABLE_SCALE[2] / 2])
    ground_center_w = np.array([0.0, 0.0, 0.0])
    table_dims      = (float(TABLE_SCALE[0]), float(TABLE_SCALE[1]), float(TABLE_SCALE[2]))
    table_pos_r     = (T @ np.append(table_center_w, 1.0))[:3]
    ground_pos_r    = (T @ np.append(ground_center_w, 1.0))[:3]
    mesh = scene.get("obj_mesh")
    if mesh is not None:
        obj_pose_r = object_pose_robot_frame(np.asarray(obj_pos_w, dtype=np.float64),
                                             np.asarray(obj_quat, dtype=np.float64), T)
        return build_world_config_dict(table_pos_r, ground_pos_r, table_dims,
                                       mesh_vertices=mesh["vertices"],
                                       mesh_faces=mesh["faces"],
                                       mesh_pose_robot=obj_pose_r)
    return build_world_config_dict(table_pos_r, ground_pos_r, table_dims)

def solve_plan_grasp(scene, start_qpos7, grasp_pos_w, grasp_quat_w_franka,
                     obj_pos_w, obj_quat,
                     approach_offset=-PREGRASP_BACKOFF, lift_offset=-LIFT_HEIGHT):
    from curobo_world import object_pose_robot_frame
    T = _T_robot_world()
    grasp_in_r = object_pose_robot_frame(np.asarray(grasp_pos_w, dtype=np.float64),
                                         np.asarray(grasp_quat_w_franka, dtype=np.float64), T)
    world_dict = _build_plan_world_dict(scene, obj_pos_w, obj_quat)
    inp = dict(
        start_qpos=np.asarray(start_qpos7, dtype=np.float64),
        grasp_pos_r=np.asarray(grasp_in_r[:3], dtype=np.float32),
        grasp_quat_r_wxyz=np.asarray(grasp_in_r[3:7], dtype=np.float32),
        world_dict=world_dict,
        approach_offset=float(approach_offset),
        lift_offset=float(lift_offset),
        pos_tol=PLAN_POS_TOL, ori_tol=PLAN_ORI_TOL, warmup_iters=2,
    )
    tag = f"/tmp/vlaeval_plan_{os.getpid()}"
    fin, fout = tag + "_in.pkl", tag + "_out.pkl"
    with open(fin, "wb") as f:
        pickle.dump(inp, f)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, CPLAN_SCRIPT, "--grasp", fin, fout],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(fout):
        cprint(f"  ❌ curobo_plan failed (rc={r.returncode}): {r.stderr[-300:]}", "red")
        return None
    with open(fout, "rb") as f:
        return pickle.load(f)


def curobo_close_and_lift(scene, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat):
    """After VLA emits close signal, plan a smooth grasp + lift trajectory via
    cuRobo MotionGen and execute it. Adapted from eval_dp3_titan_protocol.py L668.

    Returns (success, dz_m, status_str).
    """
    from omni.isaac.core.utils.types import ArticulationAction
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]

    def _pin():
        obj.rigid.set_world_pose(obj_pos_w, obj_quat)
        try:
            obj.rigid.set_linear_velocity(np.zeros(3))
            obj.rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    start_qpos7 = np.asarray(franka.get_joint_positions()[:7], dtype=np.float64)
    # approach_offset=0: EE is ALREADY at the model-chosen grasp pose. We do NOT
    # want to back off 12 cm (which would let the obj fall out of the open
    # gripper). Skip g_q approach segment by passing 0 offset.
    plan = solve_plan_grasp(scene, start_qpos7, grasp_pos_w, grasp_quat_w,
                            obj_pos_w, obj_quat,
                            approach_offset=0.0)
    if plan is None or not plan.get("success"):
        st = (plan or {}).get("status", "no plan")[:160]
        cprint(f"  ❌ close+lift plan_grasp failed: {st}", "red")
        return False, 0.0, f"plan_grasp_failed: {st}"

    g_q, l_q = plan["grasp_qpos"], plan["lift_qpos"]
    cprint(f"  ✅ cuRobo close+lift plan: grasp={len(g_q)} lift={len(l_q)} wp "
           f"(plan {plan.get('plan_seconds', float('nan')):.2f}s)", "cyan")

    # Grasp segment: kinematic set_joint_positions + obj pin to prevent collision
    # disturbing the obj as gripper aligns for final straight-in.
    for qpos7 in g_q:
        grip = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([qpos7, grip]))
        for _ in range(2):
            world.step(render=True)
        _pin()

    # Settle, release pin, read baseline z
    for _ in range(15):
        world.step(render=True)
    obj_init, _ = obj.get_obj_pos()
    initial_z = float(obj_init[2])

    # Close gripper + 80 step settle
    franka.close_gripper()
    for _ in range(80):
        world.step(render=True)

    # Lift segment via PD (collision-aware trajectory from cuRobo)
    franka.close_gripper()
    for qpos7 in l_q:
        franka.apply_action(ArticulationAction(
            joint_positions=np.concatenate([qpos7, np.array([None, None])])))
        for _ in range(3):
            world.step(render=True)
    for _ in range(80):
        world.step(render=True)

    obj_after, _ = obj.get_obj_pos()
    obj_z_after = float(obj_after[2])
    dz = obj_z_after - initial_z
    if not np.isfinite(dz) or not np.isfinite(obj_z_after) or abs(dz) > 1.0 or abs(obj_z_after) > 3.0:
        cprint(f"  ⚠️ PhysX overflow in cuRobo lift: dz={dz*100:+.1e}cm obj_z={obj_z_after:+.1e}", "red")
        return False, 0.0, "physx_corrupt"
    success = dz > SUCCESS_DZ_M
    cprint(f"  object Z Δ = {dz*100:+.1f}cm → "
           f"{'GRASPED + LIFTED (cuRobo lift)' if success else 'not lifted'}",
           "green" if success else "red")
    return success, dz, "ok"


# ============================================================
# Scene setup (same as DP3 path) + LIBERO cameras (new)
# ============================================================
def setup_world_b3_eval():
    world = World(backend="numpy")
    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    if args.video:
        set_camera_view(eye=[1.05, -0.25, 1.25], target=[0.02, 0.52, 0.90],
                        camera_prim_path="/OmniverseKit_Persp")
    else:
        set_camera_view(eye=[0.0, 4.5, 3.5], target=[0.0, 0.0, 0.0],
                        camera_prim_path="/OmniverseKit_Persp")
    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")
    GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0,
                color=np.array([0.08, 0.08, 0.10]))
    delete_prim("/World/Table")
    FixedCuboid(prim_path="/World/Table", name="table",
                position=TABLE_POSITION,
                orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True,
                color=np.array([0.80, 0.66, 0.43]))
    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))
    world.reset()
    for _ in range(40):
        world.step(render=True)
    franka.open_gripper()
    for _ in range(10):
        world.step(render=True)

    # Spawn the 2 LIBERO RGB cameras (front + side) via replicator render_product.
    cam_info = spawn_libero_cameras(world)
    for _ in range(args.cam_warmup):
        world.step(render=True)

    if args.video:
        os.makedirs(args.video, exist_ok=True)
        import omni.kit.viewport.utility as _vu
        _VID["vp"] = _vu.get_active_viewport()
        _orig_step = world.step
        def _step_capture(render=True):
            _orig_step(render=render)
            if not _VID["on"]:
                return
            _VID["n"] += 1
            if _VID["n"] % args.video_every == 0:
                _vu.capture_viewport_to_file(_VID["vp"],
                    f"{_VID_FRAMES}/f_{_VID['i']:05d}.png")
                _VID["i"] += 1
        world.step = _step_capture

    cprint("✅ World + Franka + 2 LIBERO cameras ready (replicator render_product)", "green")
    return {"world": world, "franka": franka, "obj": None, **cam_info}


def reset_scene_between_eps(scene, usd_path, mass):
    """Per-ep scene reset (clears PhysX solver state). Same as DP3 path."""
    world = scene["world"]
    world.reset()
    for _ in range(10):
        world.step(render=False)
    scene["obj"] = load_object(world, usd_path, mass)
    scene["obj_mesh"] = getattr(scene["obj"], "_curobo_mesh", None)
    scene["franka"].set_joint_positions(HOME_JOINTS)
    scene["franka"].open_gripper()
    for _ in range(5):
        world.step(render=False)


class _RedCubeWrapper:
    """Compatibility wrapper so DynamicCuboid behaves like RigidObject in eval code.
    Exposes .rigid (the DynamicCuboid prim), .rigid_prim_path, ._curobo_mesh,
    .get_obj_pos(), .set_obj_pose()."""
    def __init__(self, cube, prim_path):
        self.rigid = cube
        self.rigid_prim_path = prim_path
        self._curobo_mesh = None

    def get_obj_pos(self):
        return self.rigid.get_world_pose()

    def set_obj_pose(self, pos, ori=None):
        if ori is not None and len(ori) == 3:
            ori = euler_angles_to_quat(np.asarray(ori, dtype=np.float64), degrees=True)
        self.rigid.set_world_pose(pos, ori)


def _load_red_cube(world, mass, size):
    delete_prim("/World/RedCube")
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    spawn = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + size * 0.5 + 0.03])
    cube = DynamicCuboid(prim_path="/World/RedCube", name="red_cube",
                         position=spawn, scale=np.array([size, size, size]),
                         size=1.0, color=np.array([1.0, 0.0, 0.0]),
                         mass=mass)
    grasp_physics.setup_object_grasp_physics(world.stage, "/World/RedCube",
                                             log=lambda m: cprint(m, "green"))
    grasp_physics.setup_finger_friction(world.stage, log=lambda m: cprint(m, "green"))
    try:
        from curobo_world import prepare_curobo_mesh
        mesh = prepare_curobo_mesh(world.stage, "/World/RedCube")
        if mesh is not None:
            cprint(f"  🧊 cuRobo collision mesh: {len(mesh['vertices'])} v, {mesh['n_faces']} f", "green")
    except Exception as e:
        cprint(f"  ⚠️  prepare_curobo_mesh failed: {e}", "yellow")
        mesh = None
    for _ in range(10):
        cube.set_world_pose(spawn, np.array([1., 0., 0., 0.]))
        try:
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        world.step(render=True)
    wrap = _RedCubeWrapper(cube, "/World/RedCube")
    wrap._curobo_mesh = mesh
    return wrap


def load_object(world, usd_path, mass):
    if args.use_red_cube:
        return _load_red_cube(world, mass, args.red_cube_size)
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    spawn = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + 0.08])
    color_rgb = None
    if args.obj_color_override:
        color_rgb = tuple(float(v) for v in args.obj_color_override.split(","))
        cprint(f"  🎨 obj color override → RGB {color_rgb}", "cyan")
    obj = RigidObject(world, usd_path=usd_path, pos=spawn,
                      ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=mass,
                      color_material_rgb=color_rgb)
    grasp_physics.setup_object_grasp_physics(world.stage, obj.rigid_prim_path,
                                             log=lambda m: cprint(m, "green"))
    grasp_physics.setup_finger_friction(world.stage, log=lambda m: cprint(m, "green"))
    try:
        from curobo_world import prepare_curobo_mesh
        mesh = prepare_curobo_mesh(world.stage, obj.rigid_prim_path)
        if mesh is not None:
            cprint(f"  🧊 cuRobo collision mesh: {len(mesh['vertices'])} v, "
                   f"{mesh['n_faces']} f", "green")
        obj._curobo_mesh = mesh
    except Exception as e:
        cprint(f"  ⚠️  prepare_curobo_mesh failed: {e}", "yellow")
        obj._curobo_mesh = None
    for _ in range(10):
        obj.rigid.set_world_pose(spawn, np.array([1., 0., 0., 0.]))
        try:
            obj.rigid.set_linear_velocity(np.zeros(3))
            obj.rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        world.step(render=True)
    return obj


# ============================================================
# LIBERO 2-camera setup + RGB rendering
# ============================================================
def look_at_quat_x_forward(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
    """IsaacSim Camera convention: camera looks along local +X, +Z up, +Y left.
    Returns quat_wxyz such that camera at cam_pos points at target.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    up_proj = up - np.dot(up, forward) * forward
    if np.linalg.norm(up_proj) < 1e-6:
        up_proj = np.array([0.0, 1.0, 0.0])  # fallback when forward ∥ up
    up_proj = up_proj / np.linalg.norm(up_proj)
    # Right-handed: local_y = local_z × local_x
    local_y = np.cross(up_proj, forward); local_y = local_y / np.linalg.norm(local_y)
    R = np.stack([forward, local_y, up_proj], axis=1)
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def robot_frame_to_world(pos_R):
    """Transform a point from robot-base frame to our IsaacSim world frame.
    Accounts for ROBOT_POSITION translation + ROBOT_ORIENTATION yaw."""
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    return R_wr @ np.asarray(pos_R, dtype=np.float64) + np.asarray(ROBOT_POSITION, dtype=np.float64)


def _focal_length_for_fovy_deg(fovy_deg, vert_aperture_mm=15.2908):
    """Inverse of fovy = 2*atan(aperture/2/focal). Returns focal length in mm.
    FIX (C3): vertical aperture default in Omniverse Replicator is 15.2908 mm
    (the 20.955 we previously used is the HORIZONTAL aperture). With the wrong
    value the rendered fovy ≈ 54.8° instead of 43°.
    """
    fovy_rad = np.deg2rad(fovy_deg)
    return float(vert_aperture_mm / 2.0 / np.tan(fovy_rad / 2.0))


def spawn_libero_cameras(world):
    """Set up cameras per `--render-method`.
    - 'replicator': rep.create.camera + render_product + AnnotatorRegistry (fast)
    - 'viewport'  : just stores cam poses; rendering at use-time via
                    set_camera_view + capture_viewport_to_file (high-quality RTX,
                    same path that gate3 video recorder uses).
    """
    front_pos_w = robot_frame_to_world(GRASPVLA_FRONT_POS_R)
    front_tgt_w = robot_frame_to_world(GRASPVLA_FRONT_LOOKAT_R)
    side_pos_w  = robot_frame_to_world(GRASPVLA_SIDE_POS_R)
    side_tgt_w  = robot_frame_to_world(GRASPVLA_SIDE_LOOKAT_R)
    focal_mm = _focal_length_for_fovy_deg(LIBERO_FOVY_DEG)

    cprint(f"  📷 Front cam @ W {np.round(front_pos_w,3).tolist()} → target {np.round(front_tgt_w,3).tolist()}", "cyan")
    cprint(f"  📷 Side  cam @ W {np.round(side_pos_w,3).tolist()}  → target {np.round(side_tgt_w,3).tolist()}", "cyan")
    cprint(f"  📷 focal_length={focal_mm:.2f}mm (fovy={LIBERO_FOVY_DEG}°), render {CAM_RENDER_W}×{CAM_RENDER_H}", "cyan")
    cprint(f"  📷 render method: {args.render_method}", "cyan")

    if args.render_method == "replicator":
        front_cam = rep.create.camera(position=tuple(front_pos_w.tolist()),
                                      look_at=tuple(front_tgt_w.tolist()),
                                      focal_length=focal_mm)
        side_cam  = rep.create.camera(position=tuple(side_pos_w.tolist()),
                                      look_at=tuple(side_tgt_w.tolist()),
                                      focal_length=focal_mm)
        front_rp = rep.create.render_product(front_cam, (CAM_RENDER_W, CAM_RENDER_H))
        side_rp  = rep.create.render_product(side_cam,  (CAM_RENDER_W, CAM_RENDER_H))
        front_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        side_rgb_annot  = rep.AnnotatorRegistry.get_annotator("rgb")
        front_rgb_annot.attach([front_rp])
        side_rgb_annot.attach([side_rp])
        return {"front_annot": front_rgb_annot, "side_annot": side_rgb_annot,
                "front_rp": front_rp, "side_rp": side_rp,
                "render_method": "replicator"}
    # --- viewport ---
    # set persp viewport resolution to a square that matches CAM_RENDER_*
    try:
        import omni.kit.viewport.utility as vu
        vp = vu.get_active_viewport()
        vp.resolution = (CAM_RENDER_W, CAM_RENDER_H)
        cprint(f"  📷 persp viewport resolution → ({CAM_RENDER_W}, {CAM_RENDER_H})", "cyan")
    except Exception as e:
        cprint(f"  ⚠️  set viewport resolution failed: {e}", "yellow")
    # Per-PID tmp dir → no cross-process RGB collision when multiple eval
    # subprocesses run concurrently (e.g. batch wrapper or manual fan-out).
    _vp_tmp = f"/dev/shm/graspvla_viewport_{os.getpid()}"
    os.makedirs(_vp_tmp, exist_ok=True)
    cprint(f"  📷 viewport rgb tmp → {_vp_tmp}", "cyan")
    return {"front_pos_w": front_pos_w, "front_tgt_w": front_tgt_w,
            "side_pos_w": side_pos_w, "side_tgt_w": side_tgt_w,
            "front_path": os.path.join(_vp_tmp, "front.png"),
            "side_path":  os.path.join(_vp_tmp, "side.png"),
            "render_method": "viewport"}


def _resize_rgb_uint8(rgb_arr, out_w, out_h):
    """Downsample (H, W, 3) uint8 → (out_h, out_w, 3) uint8 via PIL."""
    from PIL import Image
    img = Image.fromarray(np.asarray(rgb_arr, dtype=np.uint8))
    img = img.resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _crop_center_square_then_resize(img_arr, out_w, out_h):
    """img_arr (H, W, 3) uint8 → center-crop to min(H,W) square → resize to (out_h, out_w)."""
    from PIL import Image
    h, w = img_arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    sq = img_arr[y0:y0+side, x0:x0+side]
    return np.asarray(Image.fromarray(sq).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


def _capture_viewport_via_persp(scene, pos_w, tgt_w, out_path, n_pose_settle=3, n_flush=12):
    """set_camera_view(/OmniverseKit_Persp) → step → capture_viewport_to_file → step until file exists.

    FIX (audit P0-1): pause video recording during persp-cam mutation so the cinematic
    video stays on the fixed third-person view, NOT mixed front/side client captures.

    FIX (C4): isaacsim.core.utils.viewports.set_camera_view ONLY sets position +
    orientation. It does NOT touch focalLength / aperture. The persp cam default
    is 50mm focal + 36×24mm sensor (fovy ≈ 28°), not the 43° GraspVLA was trained
    on. We must explicitly force focal & aperture before each capture.
    """
    import omni.kit.viewport.utility as vu
    from PIL import Image
    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass
    # Pause video frame capture while we hijack /OmniverseKit_Persp for client RGB.
    _was_recording = _VID.get("on", False)
    _VID["on"] = False
    set_camera_view(eye=pos_w.tolist(), target=tgt_w.tolist(),
                    camera_prim_path="/OmniverseKit_Persp")
    # FIX C4: force the persp cam's intrinsics to match the LIBERO training
    # convention (43° fovy on a square sensor). Without this, viewport mode
    # silently ships RGB at whatever the default persp focal/aperture is.
    try:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if persp_prim and persp_prim.IsValid():
            cam = UsdGeom.Camera(persp_prim)
            focal_mm = _focal_length_for_fovy_deg(LIBERO_FOVY_DEG)
            # Square sensor → fovx == fovy. Use 15.2908 for BOTH axes
            # (matches the corrected _focal_length_for_fovy_deg default).
            cam.GetFocalLengthAttr().Set(float(focal_mm))
            cam.GetHorizontalApertureAttr().Set(15.2908)
            cam.GetVerticalApertureAttr().Set(15.2908)
    except Exception as e:
        cprint(f"  ⚠️  persp cam intrinsics set failed: {e}", "yellow")
    world = scene["world"]
    for _ in range(n_pose_settle):
        world.step(render=True)
    vp = vu.get_active_viewport()
    vu.capture_viewport_to_file(vp, out_path)
    for _ in range(n_flush):
        world.step(render=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            break
    # Restore persp cam to the fixed cinematic third-person view, then resume video
    if args.video:
        set_camera_view(eye=[1.05, -0.25, 1.25], target=[0.02, 0.52, 0.90],
                        camera_prim_path="/OmniverseKit_Persp")
    _VID["on"] = _was_recording
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError(f"viewport capture file not produced: {out_path}")
    arr = np.asarray(Image.open(out_path).convert("RGB"), dtype=np.uint8)
    return arr


def render_rgb_pair(scene):
    """Capture (front_rgb, side_rgb), both (img_h, img_w, 3) uint8 for VLA server.
    Dispatches on scene['render_method']."""
    if scene.get("render_method") == "viewport":
        front_native = _capture_viewport_via_persp(scene, scene["front_pos_w"], scene["front_tgt_w"], scene["front_path"])
        side_native  = _capture_viewport_via_persp(scene, scene["side_pos_w"],  scene["side_tgt_w"],  scene["side_path"])
        front_rgb = _crop_center_square_then_resize(front_native, args.img_w, args.img_h)
        side_rgb  = _crop_center_square_then_resize(side_native,  args.img_w, args.img_h)
        return front_rgb, side_rgb
    # --- replicator ---
    front_data = scene["front_annot"].get_data()
    side_data  = scene["side_annot"].get_data()
    if front_data is None or side_data is None or front_data.size == 0 or side_data.size == 0:
        for _ in range(5):
            scene["world"].step(render=True)
        front_data = scene["front_annot"].get_data()
        side_data  = scene["side_annot"].get_data()
    if front_data is None or side_data is None or front_data.size == 0 or side_data.size == 0:
        raise RuntimeError("replicator annotator returned empty data after warmup")
    front_rgb_native = np.asarray(front_data[..., :3], dtype=np.uint8)
    side_rgb_native  = np.asarray(side_data [..., :3], dtype=np.uint8)
    front_rgb = _resize_rgb_uint8(front_rgb_native, args.img_w, args.img_h)
    side_rgb  = _resize_rgb_uint8(side_rgb_native,  args.img_w, args.img_h)
    return front_rgb, side_rgb


# ============================================================
# Per-episode video helpers (same as DP3 path)
# ============================================================
def video_begin():
    if not args.video:
        return
    shutil.rmtree(_VID_FRAMES, ignore_errors=True)
    os.makedirs(_VID_FRAMES, exist_ok=True)
    _VID["on"] = True
    _VID["i"] = 0
    _VID["n"] = 0

def video_end(world, out_name, keep):
    if not args.video:
        return
    _VID["on"] = False
    n = _VID["i"]
    if not keep or n == 0:
        shutil.rmtree(_VID_FRAMES, ignore_errors=True)
        return
    out = os.path.join(args.video, out_name)
    r = subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                        "-i", os.path.join(_VID_FRAMES, "f_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                       capture_output=True, text=True)
    if r.returncode == 0:
        cprint(f"  📹 {out}", "magenta")
    else:
        cprint(f"  📹 ffmpeg failed: {r.stderr[-200:]}", "yellow")
    shutil.rmtree(_VID_FRAMES, ignore_errors=True)


# ============================================================
# GraspVLA ZMQ client
# ============================================================
def graspvla_connect(addr):
    """Open ZMQ REQ socket → ROUTER server. Returns (context, socket)."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.RCVTIMEO = int(args.request_timeout_ms)
    sock.SNDTIMEO = int(args.request_timeout_ms)
    sock.LINGER = 0
    sock.connect(addr)
    return ctx, sock

def query_graspvla(sock, instruction, front_rgb, side_rgb, proprio_history):
    """Send obs to GraspVLA server, return (N_actions, 7) deltas.

    proprio_history: list of length >= proprio_history of 7D arrays.
    """
    request = {
        "text":              str(instruction),
        "front_view_image":  [np.asarray(front_rgb, dtype=np.uint8)],
        "side_view_image":   [np.asarray(side_rgb,  dtype=np.uint8)],
        "proprio_array":     [np.asarray(p, dtype=np.float32) for p in proprio_history],
    }
    sock.send_pyobj(request)
    resp = sock.recv_pyobj()
    if isinstance(resp, dict):
        # FIX (M3): match real-world grasp_mode.py:91 assertion on server status
        info = resp.get("info")
        if info is not None and info != "success":
            raise RuntimeError(f"GraspVLA server returned info={info!r}")
        result = resp["result"] if "result" in resp else resp
    else:
        result = resp
    deltas = np.asarray(result, dtype=np.float32)
    assert deltas.ndim == 2 and deltas.shape[1] == 7, f"unexpected action shape {deltas.shape}"
    return deltas


# ============================================================
# Proprio + delta integration
# ============================================================
def read_proprio_panda_link0(scene, gripper_state_signed):
    """Read current EE pose, convert to panda_link0 frame, return 7D proprio.

    Reads panda_hand pose then shifts by +0.1034 m along hand local +Z to get
    panda_EE pose (Franka standard TCP offset; verified our USD has no panda_EE
    prim). If --extended-finger, applies an additional REAL_EEF_TO_SIM_EEF offset.

    gripper_state_signed: +1.0 = open, -1.0 = closed (GraspVLA convention).
    Returns (proprio_7D, ee_pos_R, ee_rotmat_R) for use by delta integrator.
    """
    hand_pos_w, hand_q_w = read_panda_hand_pose(scene["world"].stage)
    R_hand_w = Rotation.from_quat(_xyzw(hand_q_w)).as_matrix()
    # Shift panda_hand → panda_EE (standard Franka TCP, ~10.3 cm along hand +Z)
    ee_pos_w = hand_pos_w + R_hand_w @ PANDA_HAND_TO_EE_OFFSET
    if args.extended_finger:
        ee_pos_w = ee_pos_w + R_hand_w @ REAL_EEF_TO_SIM_EEF
    ee_q_w = hand_q_w  # same orientation as hand (panda_EE inherits hand rotation)
    ee_pos_R, ee_q_R = pose_world_to_robot(ee_pos_w, ee_q_w)
    roll, pitch, yaw = quat_wxyz_to_euler_sxyz(ee_q_R)
    proprio = np.array([ee_pos_R[0], ee_pos_R[1], ee_pos_R[2],
                        roll, pitch, yaw, float(gripper_state_signed)],
                       dtype=np.float32)
    R_ee_R = Rotation.from_quat(_xyzw(ee_q_R)).as_matrix()
    return proprio, ee_pos_R, R_ee_R


def integrate_deltas_to_world_waypoints(start_pos_R, start_rotmat_R, deltas):
    """Apply VLA deltas in panda_link0 frame, accumulate, return WORLD-frame
    waypoints for cuRobo IK (panda_HAND pose, not panda_EE).

    Per-step rule (verified in graspvla-baseline memory):
        new_pos_R    = cur_pos_R + delta[:3]                       # in panda_EE space
        new_rotmat_R = euler2mat(delta[3:6]) @ cur_rotmat_R       # LEFT-multiply

    The integrated pose is in panda_EE frame (model's convention). For cuRobo IK
    which targets panda_HAND, shift back by -PANDA_HAND_TO_EE_OFFSET along EE
    local +Z.
    """
    cur_pos = np.asarray(start_pos_R, dtype=np.float64).copy()
    cur_rot = np.asarray(start_rotmat_R, dtype=np.float64).copy()
    waypoints_W, grips = [], []
    Twr = T_world_from_robot()
    R_wr = Rotation.from_euler("z", ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    for d in deltas:
        cur_pos = cur_pos + d[:3].astype(np.float64)
        dR = euler_sxyz_to_mat(float(d[3]), float(d[4]), float(d[5]))
        cur_rot = dR @ cur_rot
        # World-frame panda_EE pose
        ee_pos_w = (Twr @ np.append(cur_pos, 1.0))[:3]
        ee_rot_w = R_wr @ cur_rot
        # Reverse the same shifts read_proprio applied: panda_EE → panda_hand.
        # First undo --extended-finger's REAL_EEF_TO_SIM_EEF (P0-3 fix), then
        # undo standard panda_EE = panda_hand + 0.1034 m along hand +Z.
        if args.extended_finger:
            ee_pos_w = ee_pos_w - ee_rot_w @ REAL_EEF_TO_SIM_EEF
        hand_pos_w = ee_pos_w - ee_rot_w @ PANDA_HAND_TO_EE_OFFSET
        hand_quat_w = mat_to_quat_wxyz(ee_rot_w)  # same rotation as EE (rigid attachment)
        waypoints_W.append((hand_pos_w, hand_quat_w))
        grips.append(float(d[6]))
    return waypoints_W, grips, cur_pos, cur_rot


# ============================================================
# Eval rollout (chunked receding horizon, base-frame delta integration)
# ============================================================
def rollout_chunked(scene, sock, obj_pos_w, obj_quat, max_chunks):
    """Closed-loop GraspVLA rollout.

    Returns dict (success, dz, stage, ...) matching DP3 schema for batch wrapper.
    """
    from omni.isaac.core.utils.types import ArticulationAction
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]

    def _qpos_corrupt():
        try:
            qa = np.asarray(franka.get_joint_positions(), dtype=np.float64)
            return (not np.isfinite(qa).all()) or (np.max(np.abs(qa)) > 10.0)
        except Exception:
            return True

    obj_centroid_w = np.asarray(obj_pos_w, dtype=np.float64)
    executed = []
    grip_signal_idx = None
    gripper_closed = False
    initial_z = None

    # Open gripper + free step
    franka.open_gripper()
    for _ in range(5):
        world.step(render=True)

    # Prime proprio history (FIX C9 — match real-world grasp_mode.py:72-79 pattern).
    # Official format: proprio_array = [prev_pose, prev_pose, prev_pose, eef_pose]
    # where `prev_pose` is the pose ~0.3 s ago. Sending [cur,cur,cur,cur] feeds the
    # model zero motion signal in its [-4] vs [-1] feature.
    # We track the PREVIOUS chunk's proprio as `prev_pose` (chunk durations are
    # ~1–3 s, longer than the 0.3 s in real-world but at least nonzero motion).
    cur_proprio, _, _ = read_proprio_panda_link0(scene, gripper_state_signed=+1.0)
    prev_chunk_proprio = cur_proprio.copy()  # init = current (first chunk has no motion)
    last_qpos = np.asarray(franka.get_joint_positions()[:7], dtype=np.float64)

    for chunk in range(max_chunks):
        # ── Build obs: 7D proprio + 2 RGB ──────────────────────────────────────
        cur_proprio, cur_pos_R, cur_rot_R = read_proprio_panda_link0(
            scene, gripper_state_signed=(-1.0 if gripper_closed else +1.0))
        # FIX C9: build proprio buffer like real-world: [prev, prev, prev, current]
        proprio_buf = [prev_chunk_proprio.copy()] * (args.proprio_history - 1) + [cur_proprio]
        try:
            front_rgb, side_rgb = render_rgb_pair(scene)
        except Exception as e:
            cprint(f"  ❌ camera render failed chunk {chunk}: {e}", "red")
            return None
        # ★ DEBUG ★ Save RGB pair to disk if VLA_DEBUG_RGB_DIR env var set
        _dbg_dir = os.environ.get("VLA_DEBUG_RGB_DIR")
        if _dbg_dir:
            from PIL import Image
            os.makedirs(_dbg_dir, exist_ok=True)
            Image.fromarray(front_rgb).save(os.path.join(_dbg_dir, f"chunk_{chunk:02d}_front.png"))
            Image.fromarray(side_rgb ).save(os.path.join(_dbg_dir, f"chunk_{chunk:02d}_side.png"))

        # ── Query GraspVLA server ─────────────────────────────────────────────
        try:
            deltas = query_graspvla(sock, args.instruction,
                                    front_rgb, side_rgb, proprio_buf)
        except Exception as e:
            cprint(f"  ❌ GraspVLA server error chunk {chunk}: {e}", "red")
            return None

        # ── Integrate deltas → world-frame waypoints ──────────────────────────
        waypoints_W, chunk_grips, _, _ = integrate_deltas_to_world_waypoints(
            cur_pos_R, cur_rot_R, deltas)
        # ★ DEBUG ★ Inspect deltas + first/last waypoints to diagnose IK failures
        d0, d_last = deltas[0], deltas[-1]
        cprint(f"  [debug] chunk {chunk} N_deltas={len(deltas)}", "yellow")
        cprint(f"  [debug] cur_proprio (panda_link0): "
               f"pos={np.round(cur_proprio[:3],3).tolist()} rpy_deg={np.round(np.rad2deg(cur_proprio[3:6]),1).tolist()} "
               f"grip={cur_proprio[6]:+.1f}", "yellow")
        cprint(f"  [debug] delta[0]: dxyz={np.round(d0[:3]*1000,1).tolist()}mm "
               f"drpy_deg={np.round(np.rad2deg(d0[3:6]),2).tolist()} grip={d0[6]:+.2f}", "yellow")
        cprint(f"  [debug] delta[-1]: dxyz={np.round(d_last[:3]*1000,1).tolist()}mm "
               f"drpy_deg={np.round(np.rad2deg(d_last[3:6]),2).tolist()} grip={d_last[6]:+.2f}", "yellow")
        cprint(f"  [debug] |Δxyz| mean={np.mean(np.linalg.norm(deltas[:,:3], axis=1))*1000:.1f}mm "
               f"max={np.max(np.linalg.norm(deltas[:,:3], axis=1))*1000:.1f}mm", "yellow")
        wp0_W = waypoints_W[0][0]; wp_last_W = waypoints_W[-1][0]
        ee_pos_w_now, _ = read_panda_hand_pose(scene["world"].stage)
        cprint(f"  [debug] EE NOW (panda_hand_W): {np.round(ee_pos_w_now,3).tolist()}", "yellow")
        wp0_q = waypoints_W[0][1]; wp_last_q = waypoints_W[-1][1]
        ee_pos_w_dbg, ee_q_w_dbg = read_panda_hand_pose(scene["world"].stage)
        cprint(f"  [debug] EE quat now (wxyz): {np.round(ee_q_w_dbg,3).tolist()}", "yellow")
        cprint(f"  [debug] waypoint[0]  W: pos={np.round(wp0_W,3).tolist()} quat={np.round(wp0_q,3).tolist()}", "yellow")
        cprint(f"  [debug] waypoint[-1] W: pos={np.round(wp_last_W,3).tolist()} quat={np.round(wp_last_q,3).tolist()}", "yellow")
        # Try single IK on waypoint[0] to test reachability
        q_test = solve_single_ik(wp0_W, wp0_q)
        cprint(f"  [debug] single IK on wp[0]: {'OK qpos=%s' % np.round(q_test, 2).tolist() if q_test is not None else 'FAIL — unreachable'}", "yellow")
        # Convert each world-frame quat from Franka convention NOTE: deltas
        # produce Franka-convention world quats directly (we never applied the
        # _FRANKA_FIX retargeting). cuRobo IK expects panda_hand-frame target.
        # The DP3 path uses retarget_to_franka_quat() because DP3 outputs are in
        # *retarget* convention. Here the proprio we read came from
        # read_panda_hand_pose → already Franka convention → no retarget needed.

        # ── cuRobo IK chain on the full chunk ─────────────────────────────────
        qpos, ok = solve_ik_chain(waypoints_W, start_qpos=last_qpos)
        if qpos is None:
            cprint(f"  ❌ cuRobo IK chain failed for chunk {chunk}", "red")
            return None
        n_ok = int(ok.sum())
        first_ok = int(np.argmax(ok)) if n_ok > 0 else -1
        seed_jump = (np.abs(qpos[first_ok] - last_qpos).max()
                     if first_ok >= 0 else float("nan"))
        n_action = len(waypoints_W)
        cprint(f"  [chunk {chunk}] IK {n_ok}/{n_action} reachable, grip "
               f"[{min(chunk_grips):.2f}, {max(chunk_grips):.2f}], "
               f"seed→f0 Δ={np.rad2deg(seed_jump):.0f}°, "
               f"closed={gripper_closed}", "cyan")

        # ── Execute joint waypoints ───────────────────────────────────────────
        for k in range(n_action):
            if not ok[k]:
                continue

            # First CLOSE signal (grip < 0) → handle close + lift.
            # Two modes (--lift-mode):
            #   curobo (default): plan smooth grasp+lift via cuRobo MotionGen at
            #     the current EE pose, execute, return result. Matches DP3 path
            #     close_and_lift (eval_dp3_titan_protocol.py:668-728).
            #   vla: close_gripper + 80 settle + break → next chunk's VLA deltas
            #     drive lift via PD (can spike PhysX when fingers in contact).
            if chunk_grips[k] < 0.0 and not gripper_closed:
                obj_init_pos, obj_init_quat = obj.get_obj_pos()
                initial_z = float(obj_init_pos[2])
                grip_signal_idx = len(executed)
                ee_now, ee_quat_now = read_panda_hand_pose(world.stage)
                cprint(f"  ◉ VLA grip<0 (CLOSE) @ chunk {chunk} step {k}. "
                       f"EE={np.round(ee_now,3).tolist()}, "
                       f"obj_init_z={initial_z:.3f}  lift_mode={args.lift_mode}", "magenta")
                if args.lift_mode == "curobo":
                    success, dz, status = curobo_close_and_lift(
                        scene,
                        grasp_pos_w=ee_now, grasp_quat_w=ee_quat_now,
                        obj_pos_w=obj_init_pos, obj_quat=obj_init_quat,
                    )
                    if "physx" in status:
                        return {"success": False, "dz": 0.0, "n_chunks": chunk + 1,
                                "grip_signal_idx": grip_signal_idx,
                                "n_executed": len(executed), "stage": "physx_corrupt"}
                    dists = np.array([np.linalg.norm(p - obj_centroid_w) for p, _ in executed]) \
                            if executed else np.array([np.nan])
                    return {"success": bool(success), "dz": float(dz),
                            "n_chunks": chunk + 1, "grip_signal_idx": grip_signal_idx,
                            "n_executed": len(executed),
                            "min_dist_to_obj_cm": float(dists.min()) * 100,
                            "stage": "vla_approach_curobo_lift"}
                # VLA mode: fall through (original behavior)
                franka.close_gripper()
                for _ in range(80):
                    world.step(render=True)
                gripper_closed = True
                break  # FIX #3: stop executing this chunk; lift comes in next chunk

            # FIX #4: world.step per waypoint = 12 steps × 16.7ms = 200 ms = 5 Hz,
            # matching GraspVLA training control_freq=5 Hz
            # (benchmark_runner.py:49). Previously 2-3 steps = 20-30 Hz = 4-6×
            # faster than training → overshoot + object knocked off table.
            if gripper_closed:
                franka.close_gripper()
                franka.apply_action(ArticulationAction(
                    joint_positions=np.concatenate([qpos[k], np.array([None, None])])))
                for _ in range(6):    # FIX serve.py:71 — server interpolates ×2,
                                      # so each delta = 100 ms = 10 Hz.
                                      # 6 × 16.7 ms ≈ 100 ms matches.
                    world.step(render=True)
            else:
                grip_finger = franka.get_joint_positions()[7:9]
                full_q = np.concatenate([qpos[k], grip_finger])
                franka.set_joint_positions(full_q)
                franka.apply_action(ArticulationAction(joint_positions=full_q))
                for _ in range(6):    # FIX serve.py:71 — server interpolates ×2,
                                      # so each delta = 100 ms = 10 Hz.
                                      # 6 × 16.7 ms ≈ 100 ms matches.
                    world.step(render=True)

            if _qpos_corrupt():
                cprint(f"  ⚠️ PhysX corrupted Franka qpos at chunk {chunk} step {k}", "red")
                return {"success": False, "dz": 0.0, "n_chunks": chunk + 1,
                        "grip_signal_idx": grip_signal_idx,
                        "n_executed": len(executed),
                        "stage": "physx_corrupt"}

            executed.append((waypoints_W[k][0].copy(), waypoints_W[k][1].copy()))
            last_qpos = qpos[k].copy()

        # FIX C9: snapshot this chunk's proprio for next chunk's "prev_eef_pose".
        prev_chunk_proprio = cur_proprio.copy()

        # ── Early stop on lift success ────────────────────────────────────────
        # Gate the dz check with both a lower bound (real lift threshold) AND
        # an upper bound (catches PhysX numerical overflow where obj teleports
        # to ~10^7 m — observed in real run as dz=+642 m → false SUCCESS).
        if gripper_closed and initial_z is not None:
            try:
                obj_pos_now, _ = obj.get_obj_pos()
                dz_now = float(obj_pos_now[2]) - initial_z
                obj_z_now = float(obj_pos_now[2])
                if not np.isfinite(dz_now) or not np.isfinite(obj_z_now) or abs(dz_now) > 1.0 or abs(obj_z_now) > 3.0:
                    cprint(f"  ⚠️ PhysX overflow during lift: dz={dz_now*100:+.1e}cm  "
                           f"obj_z={obj_z_now:+.1e}m — abort as physx_corrupt", "red")
                    return {"success": False, "dz": 0.0,
                            "n_chunks": chunk + 1, "grip_signal_idx": grip_signal_idx,
                            "n_executed": len(executed), "stage": "physx_corrupt"}
                if dz_now > SUCCESS_DZ_M:
                    cprint(f"  🎯 EARLY STOP @ chunk {chunk}: dz={dz_now*100:+.1f}cm > "
                           f"{SUCCESS_DZ_M*100:.0f}cm", "green")
                    dists = np.array([np.linalg.norm(p - obj_centroid_w) for p, _ in executed]) \
                            if executed else np.array([np.nan])
                    return {"success": True, "dz": dz_now,
                            "n_chunks": chunk + 1, "grip_signal_idx": grip_signal_idx,
                            "n_executed": len(executed),
                            "min_dist_to_obj_cm": float(dists.min()) * 100,
                            "stage": "vla_close_lift_early_stop"}
            except Exception:
                pass

    # End-of-rollout settle + dz measurement
    for _ in range(80):
        world.step(render=True)

    if initial_z is None:
        cprint(f"  ⚠️ policy never emitted grip<0 across {max_chunks} chunks "
               f"({len(executed)} wp) — fail by default", "yellow")
        return {"success": False, "dz": 0.0, "n_chunks": max_chunks,
                "grip_signal_idx": None,
                "n_executed": len(executed), "stage": "no_grip_signal"}

    obj_after, _ = obj.get_obj_pos()
    obj_z_after = float(obj_after[2])
    dz = obj_z_after - initial_z
    # Same numerical-overflow gate as in EARLY STOP
    if not np.isfinite(dz) or not np.isfinite(obj_z_after) or abs(dz) > 1.0 or abs(obj_z_after) > 3.0:
        cprint(f"  ⚠️ PhysX overflow at end-of-rollout: dz={dz*100:+.1e}cm "
               f"obj_z={obj_z_after:+.1e}m — abort as physx_corrupt", "red")
        return {"success": False, "dz": 0.0, "n_chunks": max_chunks,
                "grip_signal_idx": grip_signal_idx,
                "n_executed": len(executed), "stage": "physx_corrupt"}
    success = dz > SUCCESS_DZ_M
    cprint(f"  object Z Δ = {dz*100:+.1f}cm → "
           f"{'GRASPED + LIFTED' if success else 'not lifted'}",
           "green" if success else "red")

    dists = np.array([np.linalg.norm(p - obj_centroid_w) for p, _ in executed]) \
            if executed else np.array([np.nan])
    return {"success": bool(success), "dz": dz,
            "n_chunks": chunk + 1,
            "grip_signal_idx": grip_signal_idx,
            "n_executed": len(executed),
            "min_dist_to_obj_cm": float(dists.min()) * 100,
            "stage": "vla_close_lift"}


# ============================================================
# Per-episode eval
# ============================================================
def eval_one_episode(scene, ep_path, sock):
    name = os.path.basename(ep_path)

    # Pre-ep poison check
    try:
        qcheck = np.asarray(scene["franka"].get_joint_positions(), dtype=np.float64)
        if (not np.isfinite(qcheck).all()) or (np.max(np.abs(qcheck)) > 10.0):
            cprint(f"  ⚠️ pre-ep sanity FAIL — parent poisoned", "red")
            return {"name": name, "success": False, "dz": 0.0,
                    "stage": "parent_poisoned", "parent_poisoned": True}
    except Exception as e:
        return {"name": name, "success": False, "dz": 0.0,
                "stage": "parent_poisoned", "parent_poisoned": True}

    with h5py.File(ep_path, "r") as h:
        obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64)
        obj_quat_G   = np.array(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)
        cid          = int(h.attrs["ycb_class_id"])
        obj_id_attr  = str(h.attrs.get("obj_id", ""))

    # ★ TITAN PROTOCOL OVERRIDE ★ — identity quat + z_off from titan _meta.json
    if args.titan_protocol and obj_id_attr:
        _is_dexycb = obj_id_attr.startswith("ycb_dex_") or "dexycb" in str(args.titan_usd_dir).lower()
        _is_unseen = obj_id_attr.startswith("unseen_")
        if _is_unseen and "/oakink" in args.titan_usd_dir:
            _usd_dir = args.titan_usd_dir.replace("/oakink", "/unseen")
        elif _is_dexycb and "/oakink" in args.titan_usd_dir:
            _usd_dir = args.titan_usd_dir.replace("/oakink", "/ycb")
        else:
            _usd_dir = args.titan_usd_dir
        titan_meta_path = os.path.join(_usd_dir, f"{obj_id_attr}_meta.json")
        if os.path.isfile(titan_meta_path):
            with open(titan_meta_path) as f:
                titan_meta = json.load(f)
            z_off_titan = float(titan_meta["z_offset_m"])
            obj_origin_G = np.array([0.0, 0.0, z_off_titan], dtype=np.float64)
            obj_quat_G = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            cprint(f"  ★ TITAN protocol: identity quat, z_off={z_off_titan:.4f}", "magenta")
        else:
            cprint(f"  ⚠️ titan-protocol requested but {titan_meta_path} missing", "yellow")

    # Place object
    sim_origin_W = np.array([OBJECT_XY[0] - obj_origin_G[0],
                             OBJECT_XY[1] - obj_origin_G[1], TABLE_TOP_Z])
    obj_pos_w    = obj_origin_G + sim_origin_W

    from omni.isaac.core.utils.types import ArticulationAction
    obj = scene["obj"]
    scene["franka"].set_joint_positions(HOME_JOINTS)
    scene["franka"].apply_action(ArticulationAction(
        joint_positions=HOME_JOINTS, joint_velocities=np.zeros(9)))
    scene["franka"].open_gripper()
    obj.rigid.set_world_pose(obj_pos_w, obj_quat_G)
    try:
        obj.rigid.set_linear_velocity(np.zeros(3))
        obj.rigid.set_angular_velocity(np.zeros(3))
    except Exception:
        pass
    for _ in range(100):
        scene["world"].step(render=True)

    # ── VLA rollout from HOME (no state[0] init — VLA decides full trajectory) ──
    rollout = rollout_chunked(scene, sock, obj_pos_w, obj_quat_G, args.max_chunks)
    if rollout is None:
        return {"name": name, "success": False, "dz": 0.0, "stage": "rollout_fail"}
    rollout["name"] = name
    rollout["ycb_class_id"] = cid
    return rollout


# ============================================================
# Main
# ============================================================
def main():
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)

    eps = sorted(glob.glob(args.episodes_glob))
    if not eps:
        cprint(f"❌ no episodes match {args.episodes_glob}", "red")
        simulation_app.close(); return

    n = min(args.n_rollouts, len(eps))
    if args.n_rollouts > len(eps):
        idx = list(range(len(eps))) + list(rng.integers(0, len(eps),
                                                        size=args.n_rollouts - len(eps)))
    else:
        idx = rng.choice(len(eps), size=n, replace=False)
    chosen = [eps[int(i)] for i in idx]

    cprint("=" * 64, "cyan")
    cprint(f"GraspVLA titan-protocol eval — {len(chosen)} rollouts", "cyan")
    cprint(f"  server: {args.server_addr}", "cyan")
    cprint(f"  instruction: '{args.instruction}'", "cyan")
    cprint(f"  scene : OBJECT_XY={OBJECT_XY.tolist()} TABLE_TOP_Z={TABLE_TOP_Z}", "cyan")
    cprint(f"  cams  : front_pos_R={GRASPVLA_FRONT_POS_R.tolist()} → lookat {GRASPVLA_FRONT_LOOKAT_R.tolist()}  "
           f"side_pos_R={GRASPVLA_SIDE_POS_R.tolist()} → lookat {GRASPVLA_SIDE_LOOKAT_R.tolist()}  "
           f"fovy={LIBERO_FOVY_DEG}°", "cyan")
    cprint("=" * 64, "cyan")

    scene = setup_world_b3_eval()

    with h5py.File(chosen[0], "r") as h:
        usd, obj_label = grasp_physics.usd_path_for_ep(h.attrs, proj_root=PROJ_ROOT)
        cid = int(h.attrs["ycb_class_id"])
        ds  = str(h.attrs.get("dataset", "dexycb"))
        oid = str(h.attrs.get("obj_id", ""))

    # ★ TITAN PROTOCOL: swap USD path ★
    if args.titan_protocol and oid:
        _is_dexycb = oid.startswith("ycb_dex_") or ds == "dexycb"
        _is_unseen = oid.startswith("unseen_")
        if _is_unseen and "/oakink" in args.titan_usd_dir:
            _usd_dir = args.titan_usd_dir.replace("/oakink", "/unseen")
        elif _is_dexycb and "/oakink" in args.titan_usd_dir:
            _usd_dir = args.titan_usd_dir.replace("/oakink", "/ycb")
        else:
            _usd_dir = args.titan_usd_dir
        titan_usd = os.path.join(_usd_dir, f"{oid}.usd")
        if os.path.isfile(titan_usd):
            cprint(f"  ★ TITAN protocol: USD switched {usd} → {titan_usd}", "magenta")
            usd = titan_usd
        else:
            cprint(f"  ⚠️ titan USD missing for {oid}: {titan_usd}", "yellow")

    for ep in chosen[1:]:
        with h5py.File(ep, "r") as h:
            ep_ds  = str(h.attrs.get("dataset", "dexycb"))
            ep_oid = str(h.attrs.get("obj_id", ""))
            ep_cid = int(h.attrs["ycb_class_id"])
        if (ep_ds, ep_oid, ep_cid) != (ds, oid, cid):
            cprint(f"❌ multi-class glob detected — single-obj glob required", "red")
            simulation_app.close(); return

    mass = float(args.mass_override) if args.mass_override is not None else 0.05
    cprint(f"\n=== object {obj_label}  mass={mass}kg "
           f"({'OVERRIDE' if args.mass_override is not None else 'default 0.05'}) ===", "cyan")
    if not os.path.exists(usd):
        cprint(f"  ❌ USD missing: {usd}", "red")
        simulation_app.close(); return
    scene["obj"] = load_object(scene["world"], usd, mass)
    scene["obj_mesh"] = getattr(scene["obj"], "_curobo_mesh", None)

    # Connect to GraspVLA server (singleton for the whole run)
    cprint(f"  🔌 connecting to GraspVLA @ {args.server_addr} ...", "cyan")
    ctx, sock = graspvla_connect(args.server_addr)
    cprint(f"  🔌 connected", "green")

    results = []
    poisoned = False
    for i, ep in enumerate(chosen):
        cprint(f"\n[{i+1}/{len(chosen)}] {os.path.basename(ep)}", "yellow")
        if i > 0:
            reset_scene_between_eps(scene, usd, mass)
        video_begin()
        try:
            r = eval_one_episode(scene, ep, sock)
        except Exception as e:
            import traceback
            cprint(f"  ❌ episode crashed: {type(e).__name__}: {e}", "red")
            cprint(traceback.format_exc()[-600:], "red")
            r = {"name": os.path.basename(ep), "success": False, "dz": 0.0,
                 "stage": f"crash_{type(e).__name__}"}
        retry_n = 0
        while (r.get("stage") == "physx_corrupt" and retry_n < args.retry_physx):
            retry_n += 1
            cprint(f"  🔄 physx_corrupt retry {retry_n}/{args.retry_physx}", "yellow")
            try:
                reset_scene_between_eps(scene, usd, mass)
                r2 = eval_one_episode(scene, ep, sock)
                r2["physx_retry_n"] = retry_n
                r2["physx_retry_prior_stage"] = r["stage"]
                r = r2
            except Exception as e:
                cprint(f"  ❌ retry crashed: {type(e).__name__}: {e}", "red")
                r["physx_retry_n"] = retry_n
                break
        results.append(r)
        video_end(scene["world"], r["name"].replace(".hdf5", ".mp4"),
                  keep=(r["success"] or args.video_all))
        if r.get("parent_poisoned"):
            poisoned = True
            for ep_rest in chosen[i+1:]:
                results.append({"name": os.path.basename(ep_rest),
                                "success": False, "dz": 0.0,
                                "stage": "skipped_parent_poisoned"})
            cprint(f"  ⛔ aborting run", "red")
            break

    # Summary + per-run JSON (same internal format as DP3 path; batch wrapper
    # transforms it to titan canonical schema)
    n_succ = sum(int(r["success"]) for r in results)
    cprint("\n" + "=" * 64, "cyan")
    cprint(f"  EVAL SUMMARY  {obj_label}", "cyan")
    cprint("=" * 64, "cyan")
    for r in results:
        tag = "✓" if r["success"] else "✗"
        cprint(f"  {tag} dz={r['dz']*100:+.1f}cm  stage={r.get('stage','')}  {r['name']}",
               "green" if r["success"] else "red")
    cprint("-" * 64, "cyan")
    cprint(f"  TOTAL: {n_succ}/{len(results)} grasped "
           f"({100*n_succ/max(len(results),1):.0f}%)",
           "green" if n_succ > 0 else "red")
    cprint("=" * 64, "cyan")

    out_json = os.path.join(args.result_dir, f"eval_{int(time.time())}.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args),
                   "policy": "graspvla_titan_protocol",
                   "n_success": n_succ, "n_total": len(results),
                   "results": results}, f, indent=2)
    cprint(f"wrote → {out_json}", "cyan")

    try:
        sock.close()
        ctx.term()
    except Exception:
        pass
    simulation_app.close()


main()
