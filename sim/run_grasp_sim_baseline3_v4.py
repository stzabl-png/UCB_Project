#!/usr/bin/env python3
"""baseline_3 grasp-trajectory collector.

For each retargeted DexYCB episode in Baseline1/data/episodes_g/ ("source episode"):
  1. extract the grasp-onset EE pose (the human's contact grasp pose = action[-1]),
  2. place the object at its retarget orientation obj_quat_G  (CAD USD, real YCB mass),
  3. SYNTHESIZE a clean straight-line approach  home -> pre-grasp -> grasp  (NO human
     trajectory), IK it with cuRobo, drive it in sim, close gripper, lift,
  4. if the object is lifted (>3cm), save the synthesized approach trajectory in
     episodes_g DP3-training format -> Baseline1/data/episodes_b3/.

ENVIRONMENT NOTE: run_grasp_sim.py targets IsaacSim 4.5 + old cuRobo (`curobo.wrap`,
in-process MotionGen). This machine (RTX 5090 / Blackwell sm_120) runs IsaacSim 5.1 +
cuRobo 0.8 (new `curobo.inverse_kinematics` API), and cuRobo cannot be imported inside
the IsaacSim process (Warp conflict). So baseline_3 — like gt_replay — calls cuRobo IK
OUT-OF-PROCESS via `sim/curobo_ik.py --solve`. Scene/Franka are IsaacSim-5.1 API.

Run:
    PY sim/run_grasp_sim_baseline3.py --object ycb_dex_05 --headless
    PY sim/run_grasp_sim_baseline3.py --headless --limit 5      # quick smoke test
"""
from isaacsim import SimulationApp
import argparse
import os
import sys
import glob
import re
import random
import subprocess
import shutil

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(SIM_DIR)

parser = argparse.ArgumentParser(description="baseline_3 grasp-trajectory collector")
parser.add_argument("--episodes", type=str,
                    default=os.path.join(PROJ_ROOT, "Baseline1/data/episodes_g/*.hdf5"),
                    help="glob of source retargeted episodes")
parser.add_argument("--object", type=str, default=None,
                    help="filter to one object, e.g. ycb_dex_05")
parser.add_argument("--limit", type=int, default=None, help="process at most N episodes")
parser.add_argument("--start", type=int, default=0, help="skip the first N episodes")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--out-dir", type=str,
                    default=os.path.join(PROJ_ROOT, "Baseline1/data/episodes_b3"),
                    help="where to write successful baseline_3 episodes")
parser.add_argument("--video", type=str, default=None,
                    help="capture PNG frames into this dir (a close grasp-view camera)")
parser.add_argument("--video-every", type=int, default=3,
                    help="capture 1 frame per N sim steps")
parser.add_argument("--video-all", action="store_true",
                    help="keep video for ALL episodes (default: only successful grasps)")
parser.add_argument("--inspect", action="store_true",
                    help="build ONE episode's scene and HOLD the IsaacSim GUI open for "
                         "manual inspection (no grasp run). Use WITHOUT --headless.")
parser.add_argument("--no-curobo-plan", action="store_true",
                    help="disable cuRobo collision-aware plan_grasp (object-mesh-aware) and "
                         "use the legacy synthesize+IK chain instead. Default is plan_grasp ON.")
parser.add_argument("--yaw-aug", action="store_true", default=True,
                    help="for each source ep, ALSO collect an extra trajectory after rotating "
                         "the object by a random yaw in {90,180,270}° around WORLD z-axis. "
                         "T_obj_grasp is invariant so the grasp pose follows the object rotation. "
                         "Saved with filename suffix _yawNNN.hdf5. Default ON.")
parser.add_argument("--no-yaw-aug", dest="yaw_aug", action="store_false",
                    help="disable yaw augmentation (each source ep → only 1 trajectory)")
parser.add_argument("--yaw-aug-seed", type=int, default=0,
                    help="seed for selecting random yaw per ep (reproducible).")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import h5py
from termcolor import cprint
from scipy.spatial.transform import Rotation, Slerp

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject
import grasp_physics

# ============================================================
# Scene config — IDENTICAL to run_grasp_sim.py (IK-reachability verified 95%)
# ============================================================
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80
OBJECT_XY = np.array([0.0, 0.55])          # run_grasp_sim OBJECT_POSITION xy
LIFT_HEIGHT = 0.15

# baseline_3 params
PREGRASP_BACKOFF = 0.12                    # m, back off along the gripper approach axis
N_APPROACH = 28                            # synthesized home->grasp waypoints
N_LIFT = 8                                 # synthesized grasp->lift waypoints
T_TARGET = 32                              # cap saved-episode length
HOME_JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
IK_POS_TOL, IK_ORI_TOL, IK_SEEDS = 0.005, 0.05, 1024
CIK_SCRIPT  = os.path.join(SIM_DIR, "curobo_ik.py")
CPLAN_SCRIPT = os.path.join(SIM_DIR, "curobo_plan.py")     # collision-aware MotionPlanner (sync from main)
# cuRobo plan_grasp tolerances — looser than IK because we need a feasible PATH,
# not just an exact end pose; the tail of the grasp segment lands the EE at the
# target (the IK-tight checks happen inside cuRobo's IK).
PLAN_POS_TOL, PLAN_ORI_TOL = 0.01, 0.10
# When True, route the approach via cuRobo's collision-aware planner (object mesh
# is an obstacle in approach; finger collisions auto-disabled in grasp segment).
# Falls back to the synthesized straight-line + per-pose IK chain when False or
# when planning fails — preserving the previous baseline_3 behaviour.
USE_CUROBO_PLAN_DEFAULT = True

_VID = {"on": False, "i": 0, "n": 0, "vp": None}    # per-episode video capture state
_VID_FRAMES = "/tmp/b3_video_frames"                # transient PNG dir (cleaned every episode)


# ============================================================
# Quaternion conventions  (retarget <-> Franka panda_hand)
# ============================================================
def _xyzw(q_wxyz): return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
def _wxyz(q_xyzw): return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
_RZ_NEG90 = Rotation.from_euler("z", -90, degrees=True)
_RZ_POS90 = Rotation.from_euler("z", +90, degrees=True)

def retarget_to_franka_quat(q_wxyz):
    """episode-stored retarget quat -> Franka panda_hand convention (post-multiply Rz(-90))."""
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _RZ_NEG90).as_quat())

def franka_to_retarget_quat(q_wxyz):
    """Franka panda_hand quat -> retarget convention (inverse; post-multiply Rz(+90))."""
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _RZ_POS90).as_quat())


# ============================================================
# v4: object-relative grasp pose helpers (copied from partner run_grasp_sim.py)
# ============================================================
TCP_OFFSET = 0.105      # panda_hand origin → fingertip midpoint (m)
MIN_GRASP_Z_OVER_TABLE = 0.02   # min Z clearance over table top (m)

def make_transform(pos, quat_wxyz):
    """位置 (3,) + 四元数 wxyz → 4x4 同构变换矩阵."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(_xyzw(quat_wxyz)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def transform_grasp_to_world(grasp_pos_obj, grasp_rot_obj, T_world_obj):
    """OBJ 坐标系下的 grasp pose → world 坐标系.
    grasp_pos_obj: (3,)  grasp_rot_obj: (3,3) rot matrix  T_world_obj: (4,4)."""
    pos_w = (T_world_obj @ np.append(grasp_pos_obj, 1.0))[:3]
    rot_w = T_world_obj[:3, :3] @ grasp_rot_obj
    return pos_w, rot_w


def read_panda_hand_pose(stage):
    """panda_hand world pose → (pos(3), quat_wxyz(4)) in Franka convention."""
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/panda_hand"))
    M = xf.ComputeLocalToWorldTransform(0)
    t = np.array(M.ExtractTranslation(), dtype=np.float64)
    q = M.ExtractRotationQuat()                       # Gf.Quatd
    im = q.GetImaginary()
    return t, np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64)


# ============================================================
# cuRobo IK — OUT-OF-PROCESS via curobo_ik.py (same path gt_replay uses)
# ============================================================
def solve_ik_chain(waypoints):
    """waypoints: [(pos_world(3), quat_wxyz_franka(4)), ...].
    Returns (qpos (N,7) — NaN rows for unreachable, ok (N,) bool), or (None,None)."""
    pos = np.array([w[0] for w in waypoints], dtype=np.float64)
    quat = np.array([w[1] for w in waypoints], dtype=np.float64)
    tag = f"/tmp/b3cik_{os.getpid()}"
    fin, fout = tag + "_in.npz", tag + "_out.npz"
    np.savez(fin, pos=pos, quat=quat,
             robot_pos=np.array(ROBOT_POSITION, dtype=np.float64),
             robot_ori=np.array(ROBOT_ORIENTATION, dtype=np.float64),
             num_seeds=IK_SEEDS, pos_tol=IK_POS_TOL, ori_tol=IK_ORI_TOL)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # avoid IsaacSim's Warp
    r = subprocess.run([sys.executable, CIK_SCRIPT, "--solve", fin, fout],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(fout):
        cprint(f"  ❌ curobo_ik failed (rc={r.returncode}): {r.stderr[-400:]}", "red")
        return None, None
    d = np.load(fout)
    return d["qpos"], d["ok"].astype(bool)


# ============================================================
# cuRobo MotionPlanner — out-of-process plan_grasp (3-phase, collision-aware)
# ============================================================
def _T_robot_world():
    """4x4 world→robot-base homogeneous transform (matches curobo_world conventions)."""
    R_rw = Rotation.from_euler("z", -ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_rw
    T[:3, 3] = -R_rw @ np.array(ROBOT_POSITION, dtype=np.float64)
    return T


def _build_plan_world_dict(scene, obj_pos_w, obj_quat, include_mesh=True):
    """Construct cuRobo world: table + ground (+ optional object mesh).

    v4 + Step B (titan-aligned): callers pass include_mesh=True for the pre-grasp
    phase (HOME → pre-grasp, want to avoid object during approach) and
    include_mesh=False for final-approach + lift (cuRobo would otherwise refuse to
    plan into the contact zone because sphere/mesh discrepancy says spheres-overlap).
    """
    from curobo_world import build_world_config_dict, object_pose_robot_frame  # local import

    T = _T_robot_world()
    table_center_w  = np.array([TABLE_POSITION[0], TABLE_POSITION[1], TABLE_TOP_Z - TABLE_SCALE[2] / 2])
    ground_center_w = np.array([0.0, 0.0, 0.0])
    table_dims      = (float(TABLE_SCALE[0]), float(TABLE_SCALE[1]), float(TABLE_SCALE[2]))
    table_pos_r     = (T @ np.append(table_center_w, 1.0))[:3]
    ground_pos_r    = (T @ np.append(ground_center_w, 1.0))[:3]

    mesh = scene.get("obj_mesh") if include_mesh else None
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
    """Out-of-process cuRobo plan_grasp.

    Returns dict from curobo_plan.py (success, approach/grasp/lift qpos (Nx7), …)
    or None on subprocess failure.
    """
    import pickle as _pkl
    from curobo_world import object_pose_robot_frame  # local import

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
        pos_tol=PLAN_POS_TOL,
        ori_tol=PLAN_ORI_TOL,
        warmup_iters=2,
    )
    tag = f"/tmp/b3plan_{os.getpid()}"
    fin, fout = tag + "_in.pkl", tag + "_out.pkl"
    with open(fin, "wb") as f:
        _pkl.dump(inp, f)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # avoid IsaacSim's Warp
    r = subprocess.run([sys.executable, CPLAN_SCRIPT, "--grasp", fin, fout],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(fout):
        cprint(f"  ❌ curobo_plan failed (rc={r.returncode}): {r.stderr[-400:]}", "red")
        return None
    with open(fout, "rb") as f:
        return _pkl.load(f)


def solve_plan_sequence(scene, start_qpos7, grasp_pos_w, grasp_quat_w_franka,
                        obj_pos_w, obj_quat):
    """Titan-aligned 3-phase plan_pose sequence (subprocess, single warmup).

    Returns dict { success: bool, phases: [{name, success, qpos_traj, ...}, ...], dt: float }
    or None on subprocess failure.

    Phase plan:
      pre-grasp: HOME → (grasp - PREGRASP_BACKOFF * approach_z), WITH object mesh
      final:    pre-grasp → grasp,                                  WITHOUT mesh
      lift:     grasp → (grasp + LIFT_HEIGHT * world_z),            WITHOUT mesh
    """
    import pickle as _pkl
    from curobo_world import object_pose_robot_frame  # local import

    T = _T_robot_world()
    grasp_pos_w   = np.asarray(grasp_pos_w, dtype=np.float64)
    grasp_quat_w  = np.asarray(grasp_quat_w_franka, dtype=np.float64)

    # gripper approach direction in WORLD (panda_hand z axis = points from wrist → fingertips)
    R_grasp_w = Rotation.from_quat(_xyzw(grasp_quat_w)).as_matrix()
    approach_z_w = R_grasp_w[:, 2]
    # pre-grasp world pose = grasp pulled back along approach axis
    pre_grasp_pos_w = grasp_pos_w - PREGRASP_BACKOFF * approach_z_w
    # lift world pose = grasp lifted along world Z (matches partner's lift_pos[2] += LIFT_HEIGHT)
    lift_pos_w = grasp_pos_w.copy()
    lift_pos_w[2] += LIFT_HEIGHT

    # transform all three targets to robot frame
    pre_in_r   = object_pose_robot_frame(pre_grasp_pos_w, grasp_quat_w, T)
    grasp_in_r = object_pose_robot_frame(grasp_pos_w,     grasp_quat_w, T)
    lift_in_r  = object_pose_robot_frame(lift_pos_w,      grasp_quat_w, T)

    world_with_mesh = _build_plan_world_dict(scene, obj_pos_w, obj_quat, include_mesh=True)
    world_no_mesh   = _build_plan_world_dict(scene, obj_pos_w, obj_quat, include_mesh=False)

    inp = dict(
        start_qpos=np.asarray(start_qpos7, dtype=np.float64),
        pos_tol=PLAN_POS_TOL, ori_tol=PLAN_ORI_TOL, warmup_iters=2,
        phases=[
            dict(name="pre-grasp",
                 target_pos_r=np.asarray(pre_in_r[:3], dtype=np.float32),
                 target_quat_r=np.asarray(pre_in_r[3:7], dtype=np.float32),
                 world_dict=world_with_mesh,
                 max_attempts=10),
            dict(name="final",
                 target_pos_r=np.asarray(grasp_in_r[:3], dtype=np.float32),
                 target_quat_r=np.asarray(grasp_in_r[3:7], dtype=np.float32),
                 world_dict=world_no_mesh,
                 max_attempts=10),
            dict(name="lift",
                 target_pos_r=np.asarray(lift_in_r[:3], dtype=np.float32),
                 target_quat_r=np.asarray(lift_in_r[3:7], dtype=np.float32),
                 world_dict=world_no_mesh,
                 max_attempts=10),
        ],
    )

    tag = f"/tmp/b3seq_{os.getpid()}"
    fin, fout = tag + "_in.pkl", tag + "_out.pkl"
    with open(fin, "wb") as f:
        _pkl.dump(inp, f)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, CPLAN_SCRIPT, "--sequence", fin, fout],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(fout):
        cprint(f"  ❌ curobo_plan sequence failed (rc={r.returncode}): {r.stderr[-400:]}", "red")
        return None
    with open(fout, "rb") as f:
        return _pkl.load(f)


# ============================================================
# Synthesize the approach / lift  (clean straight-line EE waypoints)
# ============================================================
def synthesize_approach(home_pos, home_quat, grasp_pos, grasp_quat, n=N_APPROACH):
    """home -> pre-grasp -> grasp.  Position: linear; orientation: SLERP home->grasp on
    the first segment, constant on the final straight-in segment.  Returns [(pos,quat)]."""
    Rg = Rotation.from_quat(_xyzw(grasp_quat))
    pre_grasp = grasp_pos - PREGRASP_BACKOFF * Rg.apply([0.0, 0.0, 1.0])
    n1 = max(2, int(round(n * 0.7)))               # home -> pre-grasp
    n2 = max(1, n - n1)                            # pre-grasp -> grasp
    slerp = Slerp([0.0, 1.0], Rotation.from_quat(
        np.array([_xyzw(home_quat), _xyzw(grasp_quat)])))
    wps = []
    for k in range(n1):
        t = k / (n1 - 1)
        wps.append((home_pos * (1 - t) + pre_grasp * t, _wxyz(slerp(t).as_quat())))
    for k in range(1, n2 + 1):
        t = k / n2
        wps.append((pre_grasp * (1 - t) + grasp_pos * t, np.asarray(grasp_quat, float).copy()))
    return wps


def synthesize_lift(grasp_pos, grasp_quat, n=N_LIFT):
    wps = []
    for k in range(1, n + 1):
        p = np.asarray(grasp_pos, float).copy()
        p[2] += LIFT_HEIGHT * (k / n)
        wps.append((p, np.asarray(grasp_quat, float).copy()))
    return wps


# ============================================================
# Scene
# ============================================================
def setup_world_b3():
    """World + table + Franka (no object — loaded per ycb class)."""
    world = World(backend="numpy")
    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    if args.video:                                          # close grasp-view camera
        set_camera_view(eye=[1.05, -0.25, 1.25], target=[0.02, 0.52, 0.90],
                        camera_prim_path="/OmniverseKit_Persp")
    else:
        set_camera_view(eye=[0.0, 4.5, 3.5], target=[0.0, 0.0, 0.0],
                        camera_prim_path="/OmniverseKit_Persp")
    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")
    # IsaacSim 5.1: GroundPlane needs an explicit prim_path (run_grasp_sim.py's
    # Real_Ground wrapper targets IsaacSim 4.5 and passes None → crash on 5.1).
    # Distinct floor/table colours so the table boundary is visible in the videos.
    GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0,
                color=np.array([0.08, 0.08, 0.10]))            # near-black floor (high contrast vs white Franka + tan table)

    delete_prim("/World/Table")
    FixedCuboid(prim_path="/World/Table", name="table",
                position=TABLE_POSITION,
                orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True,
                color=np.array([0.80, 0.66, 0.43]))            # warm tan table

    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))
    world.reset()
    for _ in range(40):
        world.step(render=True)
    franka.open_gripper()
    for _ in range(10):
        world.step(render=True)

    if args.video:                                          # wrap world.step → per-episode frame capture
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
                                             os.path.join(_VID_FRAMES, f"f_{_VID['i']:05d}.png"))
                _VID["i"] += 1
        world.step = _step_capture
        cprint(f"📹 per-episode video → {args.video}/  (only successful grasps kept)", "magenta")

    cprint("✅ World + Franka ready", "green")
    return {"world": world, "franka": franka, "obj": None}


def load_object(world, usd_path, mass):
    """(Re)load the object USD as a dynamic rigid body with real-mass grasp physics."""
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    spawn = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + 0.08])
    obj = RigidObject(world, usd_path=usd_path, pos=spawn,
                      ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=mass)
    grasp_physics.setup_object_grasp_physics(world.stage, obj.rigid_prim_path,
                                             log=lambda m: cprint(m, "green"))
    grasp_physics.setup_finger_friction(world.stage, log=lambda m: cprint(m, "green"))
    # cuRobo motion-planning collision mesh — extract once per USD load and cache
    # on the world's stage (sync with origin/main's run_grasp_sim.py).
    try:
        from curobo_world import prepare_curobo_mesh
        mesh = prepare_curobo_mesh(world.stage, obj.rigid_prim_path)
        if mesh is not None:
            cprint(f"  🧊 cuRobo collision mesh: {len(mesh['vertices'])} v, "
                   f"{mesh['n_faces']} f (raw {mesh['n_faces_raw']})", "green")
        else:
            cprint("  ⚠️  cuRobo mesh extraction returned None — planner will use table+ground only", "yellow")
        # stash on the obj (read later via scene['obj_mesh'])
        obj._curobo_mesh = mesh
    except Exception as e:
        cprint(f"  ⚠️  prepare_curobo_mesh failed: {e}", "yellow")
        obj._curobo_mesh = None
    for _ in range(10):                              # registration steps — pin so it can't free-fall
        obj.rigid.set_world_pose(spawn, np.array([1., 0., 0., 0.]))
        try:
            obj.rigid.set_linear_velocity(np.zeros(3))
            obj.rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        world.step(render=True)
    return obj


# ============================================================
# Per-episode video — capture frames, keep mp4 only for successful grasps
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
    """Stop capture; if keep, ffmpeg the frames → args.video/out_name. Always clean the PNGs."""
    if not args.video:
        return
    _VID["on"] = False
    for _ in range(8):                                      # flush pending async captures
        world.step(render=True)
    if keep and _VID["i"] > 4:
        out = os.path.join(args.video, out_name)
        r = subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-i",
                            os.path.join(_VID_FRAMES, "f_%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                           capture_output=True, text=True)
        if r.returncode == 0:
            cprint(f"  📹 video → {out}", "magenta")
        else:
            cprint(f"  📹 ffmpeg failed: {r.stderr[-200:]}", "yellow")
    shutil.rmtree(_VID_FRAMES, ignore_errors=True)


# ============================================================
# Grasp execution — drive the synthesized approach, close, lift
# ============================================================
def _execute_grasp_curobo(scene, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat):
    """cuRobo plan_grasp path (sync with origin/main).

    Returns (success, recorded_waypoints or None). recorded_waypoints is the
    actual sim-driven EE world trajectory [(pos(3), quat_wxyz_franka(4)), …],
    sampled after each waypoint — that goes into the saved DP3 episode.

    v4: NO pinning anywhere. Object is free to react to Franka contact. This
    avoids the partner-discovered failure mode where set_velocity(0) on an
    already-NaN PhysX body raises and corrupts the whole articulation.
    """
    from omni.isaac.core.utils.types import ArticulationAction
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]
    stage = world.stage
    obj_pos_w = np.asarray(obj_pos_w, dtype=np.float64)

    def _pin():
        """v4: no-op (kept so existing call sites compile). NO pinning."""
        pass

    # use the actual current Franka joints (not HOME_JOINTS) as plan start —
    # the main loop resets to HOME but reading is safer
    start_qpos7 = np.asarray(franka.get_joint_positions()[:7], dtype=np.float64)

    # === DIAG: compare what we pass to cuRobo vs IsaacSim's actual state ===
    actual_obj_pos, actual_obj_quat = obj.get_obj_pos()
    actual_obj_pos = np.asarray(actual_obj_pos, dtype=np.float64)
    actual_obj_quat = np.asarray(actual_obj_quat, dtype=np.float64)
    actual_full_q = np.asarray(franka.get_joint_positions(), dtype=np.float64)
    ee_actual_p, ee_actual_q = read_panda_hand_pose(stage)
    # object pose mismatch (passed-to-curobo vs actual)
    obj_pos_diff = np.linalg.norm(actual_obj_pos - np.asarray(obj_pos_w))
    qd_o = abs(float(np.dot(actual_obj_quat, np.asarray(obj_quat))))
    obj_quat_diff_deg = float(np.degrees(2 * np.arccos(min(1.0, qd_o))))
    # qpos sanity at plan-call time
    qpos_max = float(np.max(np.abs(actual_full_q))) if np.isfinite(actual_full_q).all() else float('nan')
    # distance from object to start EE (for sanity)
    ee_to_obj_d = float(np.linalg.norm(ee_actual_p - actual_obj_pos))
    cprint(f"  [pre-plan diag] start_qpos[:7]={np.round(start_qpos7,2).tolist()}  qpos_max={qpos_max:.2f} rad  "
           f"obj_pos passed={np.round(obj_pos_w,3).tolist()} actual={np.round(actual_obj_pos,3).tolist()}  "
           f"Δpos={obj_pos_diff*100:.2f}cm  Δquat={obj_quat_diff_deg:.1f}°  "
           f"EE@HOME pos={np.round(ee_actual_p,3).tolist()}  EE↔obj={ee_to_obj_d*100:.1f}cm  "
           f"grasp_target={np.round(grasp_pos_w,3).tolist()}", "blue")

    # v4 + titan-aligned: 3 separate plan_pose calls (pre-grasp WITH mesh,
    # final + lift WITHOUT mesh) in a single subprocess (one warmup).
    plan = solve_plan_sequence(scene, start_qpos7, grasp_pos_w, grasp_quat_w,
                               obj_pos_w, obj_quat)
    if plan is None:
        return None  # subprocess failure — caller falls back
    phases = plan.get("phases", [])
    p_by_name = {p["name"]: p for p in phases}
    pre_p   = p_by_name.get("pre-grasp", {})
    fin_p   = p_by_name.get("final", {})
    lift_p  = p_by_name.get("lift", {})
    if not plan.get("success", False):
        failed = [p["name"] for p in phases if not p.get("success")]
        if not failed:
            failed = ["?"]
        st = next((p.get("status","") for p in phases if not p.get("success")), "")
        cprint(f"  ❌ cuRobo plan sequence failed at phase={failed[0]}", "red")
        cprint(f"    status: {str(st)[:600]}", "red")
        return False, None

    a_q = pre_p["qpos_traj"]
    g_q = fin_p["qpos_traj"]
    l_q = lift_p["qpos_traj"]
    pre_t  = pre_p.get("plan_seconds", 0)
    fin_t  = fin_p.get("plan_seconds", 0)
    lift_t = lift_p.get("plan_seconds", 0)
    cprint(f"  ✅ cuRobo sequence: pre-grasp={len(a_q)}wp({pre_t:.2f}s)  "
           f"final={len(g_q)}wp({fin_t:.2f}s)  lift={len(l_q)}wp({lift_t:.2f}s)", "cyan")

    franka.open_gripper()
    for _ in range(20):
        world.step(render=True)
        # v4: no _pin() — let object respond naturally to any gripper-open motion.

    recorded = []
    def _record():
        p, q = read_panda_hand_pose(stage)
        recorded.append((p.copy(), q.copy()))

    def _qpos_corrupt():
        """True if Franka articulation state has gone NaN/extreme this step.
        v4 has no pin to mask the corruption, so we detect + abort cleanly so
        only this one ep is wasted (next ep's pre-ep sanity catches the rest)."""
        q = franka.get_joint_positions()
        try:
            qa = np.asarray(q, dtype=np.float64)
            return (not np.isfinite(qa).all()) or (np.max(np.abs(qa)) > 10.0)
        except Exception:
            return True

    # ---- pre-grasp phase: HOME → pre-grasp (planned WITH object mesh) ----
    for k, qpos7 in enumerate(a_q):
        grip = franka.get_joint_positions()[7:9]
        full_q = np.concatenate([qpos7, grip])
        franka.set_joint_positions(full_q)
        franka.apply_action(ArticulationAction(joint_positions=full_q))
        for _ in range(2):
            world.step(render=True)
        _record()
        if _qpos_corrupt():
            cprint(f"  ⚠ pre-grasp[{k}/{len(a_q)}] PhysX corrupted Franka qpos — abort ep", "red")
            return False, None

    # ---- final approach: pre-grasp → grasp (planned WITHOUT mesh, allows contact) ----
    for k, qpos7 in enumerate(g_q):
        grip = franka.get_joint_positions()[7:9]
        full_q = np.concatenate([qpos7, grip])
        franka.set_joint_positions(full_q)
        franka.apply_action(ArticulationAction(joint_positions=full_q))
        for _ in range(2):
            world.step(render=True)
        _record()
        if _qpos_corrupt():
            cprint(f"  ⚠ final[{k}/{len(g_q)}] PhysX corrupted Franka qpos — abort ep", "red")
            return False, None

    # ---- release the object; let it settle into the gripper.
    # Hold Franka in place via apply_action so PD doesn't drift the wrist
    # back toward HOME during these 15 free steps (was producing ~10° orient err).
    hold_q = np.asarray(franka.get_joint_positions(), dtype=np.float64)
    for _ in range(15):
        franka.apply_action(ArticulationAction(joint_positions=hold_q))
        world.step(render=True)
    obj_init, obj_q_now = obj.get_obj_pos()
    initial_z = float(obj_init[2])

    ee_p, ee_q_now = read_panda_hand_pose(stage)
    ee_err = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(grasp_pos_w)))
    pos_drift = float(np.linalg.norm(np.asarray(obj_init) - obj_pos_w))
    qd = abs(float(np.dot(np.asarray(obj_q_now), np.asarray(obj_quat))))
    quat_drift = float(np.degrees(2 * np.arccos(min(1.0, qd))))
    ee_to_obj = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_init)))
    # gripper orient err: actual EE quat vs plan target quat
    qd_ee = abs(float(np.dot(np.asarray(ee_q_now), np.asarray(grasp_quat_w))))
    ee_quat_err = float(np.degrees(2 * np.arccos(min(1.0, qd_ee))))
    # TCP (fingertip midpoint) world position = panda_hand_origin + 0.1034m * gripper_z_axis
    R_ee = Rotation.from_quat(_xyzw(np.asarray(ee_q_now))).as_matrix()
    tcp_p = np.asarray(ee_p) + 0.1034 * R_ee[:, 2]
    tcp_to_obj = float(np.linalg.norm(tcp_p - np.asarray(obj_init)))
    cprint(f"  [diag] gripper pos err {ee_err*1000:.0f}mm  orient err {ee_quat_err:.1f}° | "
           f"object release drift: pos {pos_drift*100:.1f}cm orient {quat_drift:.0f}° | "
           f"hand↔obj {ee_to_obj*100:.1f}cm  TCP↔obj {tcp_to_obj*100:.1f}cm", "magenta")

    franka.close_gripper()
    for _ in range(80):
        world.step(render=True)

    franka.close_gripper()
    for qpos7 in l_q:
        franka.apply_action(ArticulationAction(
            joint_positions=np.concatenate([qpos7, np.array([None, None])])))
        for _ in range(3):
            world.step(render=True)
        _record()
    for _ in range(80):
        world.step(render=True)

    obj_after, _ = obj.get_obj_pos()
    z_delta = float(obj_after[2]) - initial_z
    success = z_delta > 0.03
    cprint(f"  object Z Δ = {z_delta*100:+.1f}cm  →  "
           f"{'GRASPED + LIFTED' if success else 'not lifted'}",
           "green" if success else "red")
    # grasp_onset_idx = index (in `recorded`) of the FIRST lift waypoint — the policy
    # should learn to output grip=1 starting here and stay 1 through the lift segment.
    # (Previously save_episode_b3 hardcoded grip=1 only on the last frame → at lift
    # apex, which is too late for the policy to use as a "close now" signal.)
    grasp_onset_idx = len(a_q) + len(g_q)
    if success:
        return True, {"waypoints": recorded, "grasp_onset_idx": grasp_onset_idx}
    return False, None


def _execute_grasp_legacy(scene, home_ee, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat):
    """Legacy baseline_3 path: synthesized straight-line + per-pose IK chain.

    Fallback when cuRobo plan_grasp is unavailable or returns failure (e.g.
    object USD has no mesh, or planner ran out of IK seeds for an unusual pose)."""
    from omni.isaac.core.utils.types import ArticulationAction
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]
    obj_pos_w = np.asarray(obj_pos_w, dtype=np.float64)

    def _pin():
        """v4: no-op. Pin was the original poisoning source — see _execute_grasp_curobo."""
        pass

    approach = synthesize_approach(home_ee[0], home_ee[1], grasp_pos_w, grasp_quat_w)
    lift = synthesize_lift(grasp_pos_w, grasp_quat_w)
    n_app = len(approach)
    qpos, ok = solve_ik_chain(approach + lift)
    if qpos is None:
        return False, None
    ok_app = ok[:n_app]
    if not ok_app[-1] or ok_app.sum() < 0.7 * n_app:
        cprint(f"  ❌ approach not IK-reachable ({int(ok_app.sum())}/{n_app}, "
               f"grasp pose {'ok' if ok_app[-1] else 'UNREACHABLE'})", "red")
        return False, None

    franka.open_gripper()
    for _ in range(20):
        world.step(render=True)
    for k in range(n_app):
        if not ok[k]:
            continue
        grip = franka.get_joint_positions()[7:9]
        full_q = np.concatenate([qpos[k], grip])
        franka.set_joint_positions(full_q)
        franka.apply_action(ArticulationAction(joint_positions=full_q))
        for _ in range(2):
            world.step(render=True)
    # v4: hold Franka via apply_action during settle (no pin on object)
    hold_q = np.asarray(franka.get_joint_positions(), dtype=np.float64)
    for _ in range(15):
        franka.apply_action(ArticulationAction(joint_positions=hold_q))
        world.step(render=True)
    obj_init, obj_q_now = obj.get_obj_pos()
    initial_z = float(obj_init[2])
    ee_p, _ = read_panda_hand_pose(world.stage)
    ee_err = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(grasp_pos_w)))
    pos_drift = float(np.linalg.norm(np.asarray(obj_init) - obj_pos_w))
    qd = abs(float(np.dot(np.asarray(obj_q_now), np.asarray(obj_quat))))
    quat_drift = float(np.degrees(2 * np.arccos(min(1.0, qd))))
    ee_to_obj = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_init)))
    cprint(f"  [diag-legacy] gripper→grasp err {ee_err*1000:.0f}mm | "
           f"obj drift pos {pos_drift*100:.1f}cm ori {quat_drift:.0f}° | "
           f"gripper↔obj {ee_to_obj*100:.1f}cm", "magenta")
    franka.close_gripper()
    for _ in range(80):
        world.step(render=True)
    franka.close_gripper()
    for k in range(n_app, len(qpos)):
        if not ok[k]:
            continue
        franka.apply_action(ArticulationAction(
            joint_positions=np.concatenate([qpos[k], np.array([None, None])])))
        for _ in range(3):
            world.step(render=True)
    for _ in range(80):
        world.step(render=True)
    obj_after, _ = obj.get_obj_pos()
    z_delta = float(obj_after[2]) - initial_z
    success = z_delta > 0.03
    cprint(f"  object Z Δ = {z_delta*100:+.1f}cm  →  "
           f"{'GRASPED + LIFTED [legacy]' if success else 'not lifted [legacy]'}",
           "green" if success else "red")
    # Legacy path saves the synthesized approach only (lift not recorded). grasp_onset_idx=None
    # tells save_episode_b3 to use the original schedule (grip=1 on the last frame).
    if success:
        return True, {"waypoints": approach, "grasp_onset_idx": None}
    return False, None


def execute_grasp_b3(scene, home_ee, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat):
    """Returns (success, waypoints_for_dp3_save or None).

    Path A (preferred, sync with origin/main): cuRobo plan_grasp — collision-aware
    motion planning with the object mesh as an obstacle for the approach; finger
    collisions auto-disabled for the grasp segment. Saved waypoints = actual
    panda_hand world poses recorded during sim execution.

    Path B (fallback): the original baseline_3 synthesize+IK chain. Used when
    cuRobo's subprocess errors or no plan is returned. Saved waypoints = the
    synthesized straight-line EE waypoints.
    """
    if scene.get("use_curobo_plan", USE_CUROBO_PLAN_DEFAULT):
        r = _execute_grasp_curobo(scene, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat)
        if r is not None:                               # planner ran (success or clean fail)
            return r
        cprint("  ↩ falling back to legacy synthesize+IK path", "yellow")
    return _execute_grasp_legacy(scene, home_ee, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat)


# ============================================================
# Save a successful baseline_3 episode in episodes_g DP3-training format
# ============================================================
def save_episode_b3(out_path, rec, sim_origin_W, src_attrs, pc0, grasp_onset_idx=None):
    """rec: list of (pos_world, quat_world_franka) — approach (+grasp +lift if available).
    Writes state/action (G-frame, retarget convention) + point_cloud + attrs.

    grasp_onset_idx: index into the raw `rec` of the FIRST lift waypoint. If given,
      the gripper schedule is grip=1 from that point onward (so the policy learns to
      output grip=1 starting at the grasp pose and stay 1 through the lift). If None
      (legacy path), grip=1 only on the very last saved frame.
    """
    n = len(rec)
    if n > T_TARGET:
        sel = np.unique(np.linspace(0, n - 1, T_TARGET).round().astype(int))
    else:
        sel = np.arange(n)
    rec_sub = [rec[k] for k in sel]
    if grasp_onset_idx is None:
        sub_onset = len(rec_sub) - 1                                 # legacy: last frame only
    else:
        sub_onset = int(np.searchsorted(sel, grasp_onset_idx))       # first subsampled k whose raw idx ≥ onset
        sub_onset = min(sub_onset, len(rec_sub) - 1)

    states, prev_q = [], None
    for k, (pw, qw_f) in enumerate(rec_sub):
        pos_G = (np.asarray(pw) - sim_origin_W).astype(np.float32)
        q = franka_to_retarget_quat(qw_f).astype(np.float32)
        if prev_q is not None and float(np.dot(q, prev_q)) < 0.0:    # quaternion sign-continuity
            q = -q
        prev_q = q
        grip = np.float32(1.0 if k >= sub_onset else 0.0)            # close on/after grasp pose
        states.append(np.concatenate([pos_G, q, [grip]]).astype(np.float32))
    states = np.stack(states)
    state, action = states[:-1], states[1:]
    T = int(state.shape[0])
    pc = np.repeat(pc0[None].astype(np.float32), T, axis=0)          # object static → PC constant

    with h5py.File(out_path, "w") as h:
        h.create_dataset("state", data=state)
        h.create_dataset("action", data=action)
        h.create_dataset("point_cloud", data=pc)
        for k in ("dataset", "obj_id", "ycb_class_id", "session", "camera", "subject",
                  "mano_side", "origin_G_W", "table_z_G", "obj_quat_G_wxyz",
                  "obj_origin_G", "ee_offset_m", "gripper_span_m"):
            if k in src_attrs:
                h.attrs[k] = src_attrs[k]
        h.attrs["n_steps"] = T
        h.attrs["grasp_onset"] = sub_onset                            # index in saved (subsampled) frames
        h.attrs["grasp_onset_idx"] = sub_onset
        h.attrs["baseline"] = "baseline_3"
    return T


# ============================================================
# Main — loop over episodes
# ============================================================
def _ep_class(path):
    m = re.search(r"ycb_dex_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 999


def inspect_scene(ep_path):
    """Build ONE episode's baseline_3 scene and hold the IsaacSim GUI open for
    manual inspection — verify Franka/table/object poses vs run_grasp_sim.py."""
    name = os.path.basename(ep_path)
    with h5py.File(ep_path, "r") as h:
        cid = int(h.attrs["ycb_class_id"])
        obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64)
        obj_quat_G = np.array(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)
        action = h["action"][:]
    scene = setup_world_b3()
    usd = os.path.join(PROJ_ROOT, "output/obj_usd_cad/ycb", f"ycb_dex_{cid:02d}.usd")
    obj = load_object(scene["world"], usd, grasp_physics.object_mass_kg(cid))
    sim_origin_W = np.array([OBJECT_XY[0] - obj_origin_G[0],
                             OBJECT_XY[1] - obj_origin_G[1], TABLE_TOP_Z])
    obj_pos_w = obj_origin_G + sim_origin_W
    obj.rigid.set_world_pose(obj_pos_w, obj_quat_G)
    scene["franka"].set_joint_positions(HOME_JOINTS)
    for _ in range(30):
        scene["world"].step(render=True)
    grasp_pos_w = action[-1, :3] + sim_origin_W

    cprint("\n" + "=" * 74, "cyan")
    cprint(f"  INSPECT — episode {name}", "cyan")
    cprint("=" * 74, "cyan")
    cprint(f"  /World/Franka : pos={list(ROBOT_POSITION)}  euler_deg={list(ROBOT_ORIENTATION)}", "yellow")
    cprint("      run_grasp_sim.py ROBOT_POSITION [0.2,-0.05,0.8] / ROBOT_ORIENTATION [0,0,90]"
           "  →  IDENTICAL (constants copied verbatim)", "green")
    cprint(f"  /World/Table  : pos={list(TABLE_POSITION)}  scale={list(TABLE_SCALE)}  top_z={TABLE_TOP_Z}", "yellow")
    cprint("      run_grasp_sim.py TABLE_POSITION [0,1.0,0.75] / TABLE_SCALE [2,2,0.1]"
           "  →  IDENTICAL (constants copied verbatim)", "green")
    cprint(f"  /World/Rigid/rigid : pos={obj_pos_w.round(4).tolist()}  "
           f"quat_wxyz={obj_quat_G.round(4).tolist()}", "yellow")
    cprint("      baseline_3 places the object at the RETARGET pose (obj_origin_G + obj_quat_G).", "cyan")
    cprint("      run_grasp_sim.py uses FIXED OBJECT_POSITION [0,0.55,0.80]+z_offset, ORIENTATION [0,0,0].", "cyan")
    cprint(f"      → object xy same ({OBJECT_XY.tolist()}); z & orientation DIFFER BY DESIGN", "cyan")
    cprint("        (baseline_3's whole point: object placed at the retarget orientation).", "cyan")
    cprint(f"  grasp target (panda_hand, world) : pos={grasp_pos_w.round(4).tolist()}", "yellow")
    cprint("-" * 74, "cyan")
    cprint("  GUI is live — select the prims above in the Stage panel to read their", "magenta")
    cprint("  transforms; orbit/drag the camera freely. Close the window (or Ctrl-C) to exit.", "magenta")
    cprint("=" * 74, "cyan")
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()


def main():
    eps = sorted(glob.glob(args.episodes))
    if args.object:
        eps = [e for e in eps if args.object in os.path.basename(e)]
    eps.sort(key=lambda p: (_ep_class(p), p))               # group by object → fewer USD reloads
    if args.start:
        eps = eps[args.start:]
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        cprint("❌ no episodes matched", "red")
        simulation_app.close()
        return

    if args.inspect:
        inspect_scene(eps[0])
        return

    os.makedirs(args.out_dir, exist_ok=True)
    use_plan = not args.no_curobo_plan
    cprint("=" * 64, "cyan")
    cprint(f"baseline_3 collector — {len(eps)} episodes → {args.out_dir}", "cyan")
    cprint(f"  motion planning: {'cuRobo plan_grasp (mesh-aware)' if use_plan else 'legacy synthesize+IK'}",
           "cyan")
    cprint("=" * 64, "cyan")

    scene = setup_world_b3()
    scene["use_curobo_plan"] = use_plan
    stage = scene["world"].stage
    # the Franka home is fixed → its EE pose is constant; read it once
    scene["franka"].set_joint_positions(HOME_JOINTS)
    for _ in range(40):
        scene["world"].step(render=True)
    home_ee = read_panda_hand_pose(stage)
    cprint(f"Franka home EE: pos={home_ee[0].round(3)}", "cyan")

    cur_class = None
    stats = {}                                              # ycb_class_id -> [n_success, n_attempt]

    for i, ep in enumerate(eps):
        name = os.path.basename(ep)
        with h5py.File(ep, "r") as h:
            action = h["action"][:]
            # point_cloud[0] = the DexYCB CAD mesh (textured.obj) surface-sampled by
            # build_gt_replay.get_object_points (fixed seed → byte-identical), placed at
            # the object's frame-0 G-pose. It IS the CAD cloud — reused here instead of
            # re-sampling the same mesh. baseline_3's object is static → one cloud, all frames.
            pc0 = h["point_cloud"][0]
            cid = int(h.attrs["ycb_class_id"])
            obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64)
            obj_quat_G = np.array(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)
            src_attrs = dict(h.attrs)

        if cid != cur_class:
            usd = os.path.join(PROJ_ROOT, "output/obj_usd_cad/ycb", f"ycb_dex_{cid:02d}.usd")
            # v4 experiment: match partner main's hardcoded mass=0.05kg.
            # Hypothesis: real per-class mass (mustard 0.6kg, sugar 0.5kg) causes
            # PhysX collision force ~10x larger → solver overflow → NaN poison.
            mass = 0.05
            real_mass = grasp_physics.object_mass_kg(cid)
            cprint(f"\n=== object ycb_dex_{cid:02d}  mass={mass}kg (real {real_mass}kg, using main's 0.05) ===", "cyan")
            if not os.path.exists(usd):
                cprint(f"  ⚠️  USD missing: {usd} — skipping this object", "red")
                cur_class = cid
                scene["obj"] = None
                continue
            scene["obj"] = load_object(scene["world"], usd, mass)
            scene["obj_mesh"] = getattr(scene["obj"], "_curobo_mesh", None)
            cur_class = cid
        if scene["obj"] is None:
            continue

        # v4 + yaw aug: per source ep we collect (potentially) MULTIPLE trajectories
        # at different object yaws. The original (yaw=0) is always attempted; if
        # --yaw-aug is on we ALSO pick one random yaw ∈ {90,180,270} and attempt.
        sim_origin_W = np.array([OBJECT_XY[0] - obj_origin_G[0],
                                 OBJECT_XY[1] - obj_origin_G[1], TABLE_TOP_Z])
        obj_pos_w_initial = np.asarray(obj_origin_G + sim_origin_W, dtype=np.float64)
        obj = scene["obj"]
        from omni.isaac.core.utils.types import ArticulationAction

        # Build list of (yaw_deg, suffix) tuples to attempt for this source ep.
        # Apply ALL three yaws (90/180/270) for every source ep — no sampling.
        # Per-yaw output is saved separately as <name>_yaw{deg}.hdf5 so downstream
        # training can include/exclude individual yaw variants without re-collecting.
        yaw_attempts = [(0, "")]
        if args.yaw_aug:
            for yaw_deg in (90, 180, 270):
                yaw_attempts.append((yaw_deg, f"_yaw{yaw_deg}"))

        # T_obj_grasp is INVARIANT to where we put the object — compute once
        T_G_obj = make_transform(obj_origin_G, obj_quat_G)
        T_G_grasp = make_transform(action[-1, :3], action[-1, 3:7])
        T_obj_grasp = np.linalg.inv(T_G_obj) @ T_G_grasp

        sanity_failed_this_ep = False
        for yaw_deg, suffix in yaw_attempts:
            # ── compute target obj quat: rotate AROUND WORLD Z axis by yaw_deg ──
            # R_world_yaw composed on the LEFT of original obj_quat = rotation
            # in world frame about world-z, through obj origin.
            R_world_yaw = Rotation.from_euler("z", yaw_deg, degrees=True)
            obj_quat_target = _wxyz(
                (R_world_yaw * Rotation.from_quat(_xyzw(obj_quat_G))).as_quat()
            )

            # ── reset Franka to HOME + clear PD target ──
            scene["franka"].set_joint_positions(HOME_JOINTS)
            scene["franka"].apply_action(ArticulationAction(
                joint_positions=HOME_JOINTS, joint_velocities=np.zeros(9)))
            scene["franka"].open_gripper()

            # ── place object once + free settle (no pin) ──
            obj.rigid.set_world_pose(obj_pos_w_initial, obj_quat_target)
            try:
                obj.rigid.set_linear_velocity(np.zeros(3))
                obj.rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
            for _ in range(100):
                scene["world"].step(render=True)

            # ── sanity (previous attempt poisoned the parent process?) ──
            qpos_now = np.asarray(scene["franka"].get_joint_positions(), dtype=np.float64)
            franka_sane = np.isfinite(qpos_now).all() and (np.abs(qpos_now) < 10).all()
            if not franka_sane:
                cprint(f"  ⚠️ pre-ep sanity check FAIL — franka_sane=False "
                       f"qpos_max={np.nanmax(np.abs(qpos_now)):.2e}", "red")
                cprint(f"     parent process poisoned → EARLY-EXIT remaining {len(eps)-i} eps so wrapper can restart", "red")
                sanity_failed_this_ep = True
                break

            # ── read settled object pose (may have rolled/tipped, esp. after yaw) ──
            obj_pos_actual, obj_quat_actual = obj.get_obj_pos()
            obj_pos_actual = np.asarray(obj_pos_actual, dtype=np.float64)
            obj_quat_actual = np.asarray(obj_quat_actual, dtype=np.float64)
            settle_drift_pos = float(np.linalg.norm(obj_pos_actual - obj_pos_w_initial))
            qd_s = abs(float(np.dot(obj_quat_actual, obj_quat_target)))
            settle_drift_quat = float(np.degrees(2 * np.arccos(min(1.0, qd_s))))
            if (not np.isfinite(obj_pos_actual).all()) or settle_drift_pos > 0.30:
                cprint(f"  ⚠️ object settle FAIL (drift {settle_drift_pos*100:.0f}cm) — skip yaw={yaw_deg}", "yellow")
                continue
            if settle_drift_pos > 0.05 or settle_drift_quat > 30:
                cprint(f"  ⓘ yaw={yaw_deg} settled with drift: pos {settle_drift_pos*100:.1f}cm orient {settle_drift_quat:.0f}°", "yellow")

            # ── compute world grasp pose: T_world_obj_actual @ T_obj_grasp ──
            T_world_obj = make_transform(obj_pos_actual, obj_quat_actual)
            T_world_grasp = T_world_obj @ T_obj_grasp
            grasp_pos_w = T_world_grasp[:3, 3]
            grasp_rot_w = T_world_grasp[:3, :3]
            grasp_quat_w = retarget_to_franka_quat(
                _wxyz(Rotation.from_matrix(grasp_rot_w).as_quat())
            )
            # Z safety clamp
            min_grasp_z = TABLE_TOP_Z + MIN_GRASP_Z_OVER_TABLE
            if grasp_pos_w[2] < min_grasp_z:
                cprint(f"   ⓘ grasp Z={grasp_pos_w[2]:.3f} clamped to {min_grasp_z:.3f}", "yellow")
                grasp_pos_w[2] = min_grasp_z

            obj_pos_w = obj_pos_actual
            obj_quat_for_plan = obj_quat_actual

            label = f"yaw={yaw_deg}" if yaw_deg != 0 else "orig"
            cprint(f"\n[{i+1}/{len(eps)}] ({label}) {name}", "yellow")
            video_begin()
            try:
                ok, rec_info = execute_grasp_b3(scene, home_ee, grasp_pos_w, grasp_quat_w,
                                                obj_pos_w, obj_quat_for_plan)
            except Exception as e:
                cprint(f"  ERROR: {e}", "red")
                ok, rec_info = False, None

            st = stats.setdefault(cid, [0, 0])
            st[1] += 1
            success = bool(ok and rec_info is not None and len(rec_info["waypoints"]) >= 3)
            if success:
                out_name = name.replace(".hdf5", f"{suffix}.hdf5")
                out = os.path.join(args.out_dir, out_name)
                T = save_episode_b3(out, rec_info["waypoints"], sim_origin_W, src_attrs, pc0,
                                    grasp_onset_idx=rec_info["grasp_onset_idx"])
                st[0] += 1
                cprint(f"  ✅ saved baseline_3 episode ({T} steps, grasp_onset="
                       f"{rec_info['grasp_onset_idx']}) → {out}", "green")
            else:
                cprint(f"  ❌ grasp failed ({label}) — not saved", "red")
            video_end(scene["world"], name.replace(".hdf5", f"{suffix}.mp4"),
                      keep=(success or args.video_all))

        if sanity_failed_this_ep:
            break

    # ---- summary ----
    cprint("\n" + "=" * 64, "cyan")
    cprint("  baseline_3 collection summary", "cyan")
    cprint("=" * 64, "cyan")
    tot_s = tot_a = 0
    for cid in sorted(stats):
        s, a = stats[cid]
        tot_s += s; tot_a += a
        cprint(f"  ycb_dex_{cid:02d}:  {s}/{a} grasped  ({100*s/max(a,1):.0f}%)", "cyan")
    cprint("-" * 64, "cyan")
    cprint(f"  TOTAL: {tot_s}/{tot_a} grasped  ({100*tot_s/max(tot_a,1):.1f}%)  "
           f"→ {tot_s} episodes in {args.out_dir}", "green")
    cprint("=" * 64, "cyan")
    simulation_app.close()


main()
