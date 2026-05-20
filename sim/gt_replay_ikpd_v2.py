#!/usr/bin/env python3
"""
sim/gt_replay_ikpd_v2.py — Gate 3 with per-session sim setup + parallel-to-table pre-position.

What changed vs gt_replay_ikpd.py:
  1. Per-session config (sim_origin_W, Franka base) — frame 0 EE is reachable
  2. 3-stage pre-position to bridge Franka home → trajectory state[0]:
       (a) cartesian interp:   home_EE → (state[0].pos, parallel-to-table orientation)
       (b) orientation slerp:  hold pos, rotate to state[0].quat
       (c) ready to replay
  3. Object spawn uses identity orientation (master_chef_can is cylindrically symmetric;
     true GT orientation matters for asymmetric objects in later phases)

Stack stays IK (Lula) + PhysX implicit PD.

Usage:
  /home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/gt_replay_ikpd_v2.py \\
      --session 142601 --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True,
                    help="DexYCB session ID; short (142601) or full (20201015_142601)")
parser.add_argument("--object", default=None,
                    help="override ycb_dex_NN; default = SESSION_CONFIG[session]['object']")
parser.add_argument("--traj", default=None, help="override HDF5 path (else use template)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--phys-per-action", type=int, default=5)
parser.add_argument("--ik-pos-tol", type=float, default=0.005)
parser.add_argument("--ik-ori-tol", type=float, default=0.05)
parser.add_argument("--position-only", action="store_true",
                    help="ignore orientation in IK — pure positional control (sanity test)")
parser.add_argument("--grip-delay", type=int, default=0,
                    help="delay gripper close by N frames after grasp_onset_idx")
parser.add_argument("--video", default=None, help="save PNG frames here (omitted = no recording)")
parser.add_argument("--video-every", type=int, default=3, help="capture 1 frame every N sim steps")
parser.add_argument("--ik-solver", choices=["lula", "curobo"], default="lula",
                    help="offline IK backend: lula (18-seed local) or curobo (1024-seed GPU)")
parser.add_argument("--curobo-seeds", type=int, default=1024, help="cuRobo IK seed count")
args, _ = parser.parse_known_args()

PROJ_ROOT = "/home/accelerator/UCB_Project"
HDF5_TEMPLATE = "/tmp/gt_replay_{obj_short}_session{sess_short}_cam_master.hdf5"

# YCB class id → object name (used to auto-derive OBJECT_META for any ycb_dex_NN)
YCB_CLASS_NAME = {
    1: "master_chef_can", 2: "cracker_box", 3: "sugar_box", 4: "tomato_soup_can",
    5: "mustard_bottle", 6: "tuna_fish_can", 7: "pudding_box", 8: "gelatin_box",
    9: "potted_meat_can", 10: "banana", 11: "pitcher_base", 12: "bleach_cleanser",
    13: "bowl", 14: "mug", 15: "power_drill", 16: "wood_block", 17: "scissors",
    18: "large_marker", 19: "large_clamp", 20: "extra_large_clamp", 21: "foam_brick",
}

# ── Per-object metadata (CAD-frame dimensions, USD path, YCB class id) ───────
# height = longest CAD bbox axis (for spawn z = TABLE_TOP + height/2; assumes upright)
OBJECT_META = {
    "ycb_dex_01": {"name": "master_chef_can",   "ycb_class_id":  1, "height": 0.140,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_01.usd"},
    "ycb_dex_04": {"name": "tomato_soup_can",   "ycb_class_id":  4, "height": 0.102,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_04.usd"},
    "ycb_dex_05": {"name": "mustard_bottle",    "ycb_class_id":  5, "height": 0.191,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_05.usd"},
    "ycb_dex_09": {"name": "potted_meat_can",   "ycb_class_id":  9, "height": 0.102,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_09.usd"},
    "ycb_dex_12": {"name": "bleach_cleanser",   "ycb_class_id": 12, "height": 0.251,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_12.usd"},
    "ycb_dex_18": {"name": "large_marker",      "ycb_class_id": 18, "height": 0.121,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_18.usd"},
    "ycb_dex_21": {"name": "foam_brick",        "ycb_class_id": 21, "height": 0.078,
                   "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/ycb_dex_21.usd"},
}

# ── Per-session config ───────────────────────────────────────────────────────
# state[0]_G summary for the master_chef_can pilot:
#   142601: pos=(+0.34,-0.10,+0.08)  ang=149°
#   142646: pos=(+0.31,+0.01,+0.24)  ang=142°
#   142724: pos=(+0.19,-0.63,+0.05)  ang=140°    (Y=-63cm is huge!)
#   142815: pos=(+0.18,-0.30,+0.22)  ang=173°
#   142844: pos=(+0.26,-0.13,+0.29)  ang=126°
# Manual robot_pos/robot_ori kept for these 5 so regression tests reproduce v6 exactly.
# For other sessions, omit robot_pos/robot_ori → script auto-computes from state[0].
SESSION_CONFIG = {
    # master_chef_can (Phase 1 pilot — manually tuned values, baseline for regression)
    "20201015_142601": {"object": "ycb_dex_01", "sim_origin_xy": (0.0,  0.30),
                        "robot_pos": (0.69,  0.20, 0.80), "robot_ori": (0., 0., 180.)},
    "20201015_142646": {"object": "ycb_dex_01", "sim_origin_xy": (0.0,  0.30),
                        "robot_pos": (0.66,  0.31, 0.80), "robot_ori": (0., 0., 180.)},
    "20201015_142724": {"object": "ycb_dex_01", "sim_origin_xy": (0.0,  0.30),
                        "robot_pos": (0.54, -0.33, 0.80), "robot_ori": (0., 0., 180.)},
    "20201015_142815": {"object": "ycb_dex_01", "sim_origin_xy": (0.0,  0.30),
                        "robot_pos": (0.53,  0.00, 0.80), "robot_ori": (0., 0., 180.)},
    "20201015_142844": {"object": "ycb_dex_01", "sim_origin_xy": (0.0,  0.30),
                        "robot_pos": (0.61,  0.17, 0.80), "robot_ori": (0., 0., 180.)},
    # tomato_soup_can (auto robot pose from state[0])
    "20201015_143403": {"object": "ycb_dex_04", "sim_origin_xy": (0.0,  0.30)},
    "20201015_143429": {"object": "ycb_dex_04", "sim_origin_xy": (0.0,  0.30)},
    "20201015_143455": {"object": "ycb_dex_04", "sim_origin_xy": (0.0,  0.30)},
    "20201015_143524": {"object": "ycb_dex_04", "sim_origin_xy": (0.0,  0.30)},
    "20201015_143556": {"object": "ycb_dex_04", "sim_origin_xy": (0.0,  0.30)},
    # mustard_bottle (auto Franka)
    "20201015_143636": {"object": "ycb_dex_05", "sim_origin_xy": (0.0,  0.30)},
    # potted_meat_can (auto Franka)
    "20201015_144721": {"object": "ycb_dex_09", "sim_origin_xy": (0.0,  0.30)},
    # bleach_cleanser (auto Franka)
    "20201015_145515": {"object": "ycb_dex_12", "sim_origin_xy": (0.0,  0.30)},
    # foam_brick (auto Franka)
    "20201015_151450": {"object": "ycb_dex_21", "sim_origin_xy": (0.0,  0.30)},
}

# Normalize session id (accept short form like "142601")
_sess_id = args.session if args.session.startswith("2020") else f"20201015_{args.session}"
if _sess_id in SESSION_CONFIG:
    cfg = SESSION_CONFIG[_sess_id]
else:
    # Ad-hoc session not in the curated config (e.g. batch Gate-3 sweep): require --object,
    # default to the standard sim setup (auto Franka pose, origin 0.30m in front).
    if not args.object:
        raise SystemExit(f"Session '{args.session}' not in SESSION_CONFIG — pass --object ycb_dex_NN.")
    cfg = {"object": args.object, "sim_origin_xy": (0.0, 0.30)}
OBJECT = args.object or cfg["object"]
if OBJECT in OBJECT_META:
    obj_meta = OBJECT_META[OBJECT]
else:
    # Auto-derive for any ycb_dex_NN: USD from the standard CAD-USD dir, NN = ycb class id.
    # height is only a fallback spawn dim (unused when the HDF5 carries obj_origin_G).
    _nn = int(OBJECT.replace("ycb_dex_", ""))
    obj_meta = {"name": YCB_CLASS_NAME.get(_nn, OBJECT), "ycb_class_id": _nn, "height": 0.10,
                "usd": f"{PROJ_ROOT}/output/obj_usd_cad/ycb/{OBJECT}.usd"}

# Resolve HDF5 path (CLI override → config → template)
if args.traj:
    TRAJ = args.traj
elif "traj" in cfg:
    TRAJ = cfg["traj"]
else:
    _sess_short = _sess_id.split("_")[-1]
    _obj_short = OBJECT.replace("ycb_dex_", "ycb")     # ycb_dex_04 → ycb04
    TRAJ = HDF5_TEMPLATE.format(obj_short=_obj_short, sess_short=_sess_short)

SIM_ORIGIN_XY = cfg["sim_origin_xy"]
TABLE_POS = [0, 1.0, 0.75]; TABLE_SCALE = [2, 2, 0.1]; TABLE_TOP_Z = 0.80
SETTLE_INIT = 50
SETTLE_AT_STATE0 = 50     # PhysX steps after re-spawn at qpos[state[0]] (PD locks fast since qpos is exact)
HOLD_BEFORE_GRIP = 50     # settle steps before closing gripper
HOLD_AFTER_GRIP = 100     # let gripper close fully before lift starts

sim_app = SimulationApp({"headless": args.headless})

import numpy as np, h5py
from scipy.spatial.transform import Rotation
from termcolor import cprint
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
import omni.replicator.core as rep

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject


# ── helpers ──────────────────────────────────────────────────────────────────
def quat_wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def quat_xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

def auto_robot_pose(state0_pos_G, sim_origin_xy, reach_dist=0.35, base_z=0.80):
    """Place Franka base reach_dist further outward from state[0] along the
    object→state[0] direction (in xy), facing state[0]. Used when the session
    config doesn't pin robot_pos/robot_ori manually."""
    out = np.array([state0_pos_G[0], state0_pos_G[1]], dtype=np.float64)  # state[0]_xy in G = state[0]_xy - origin_xy
    n = np.linalg.norm(out)
    out = out / n if n > 1e-3 else np.array([1.0, 0.0])
    state0_sx = state0_pos_G[0] + sim_origin_xy[0]
    state0_sy = state0_pos_G[1] + sim_origin_xy[1]
    robot_xy = (state0_sx + reach_dist * out[0], state0_sy + reach_dist * out[1])
    yaw_deg = float(np.degrees(np.arctan2(-out[1], -out[0])))               # face back toward state[0]
    return [robot_xy[0], robot_xy[1], base_z], [0.0, 0.0, yaw_deg]

# CONVENTION FIX: retarget builds R with local +X = opening axis (thumb→index),
# local +Z = approach. Franka panda_hand has local +Y = opening, local +Z = approach.
# So data and robot differ by a 90° rotation around the local +Z (approach) axis.
# Empirically verified: panda_hand FK shows fingers separated along +Y by 0.05m, not +X.
# We post-multiply every retarget quat by a -90° rotation around local +Z so that
# what was retarget's +X (= opening direction in world) becomes Franka's +Y (= opening dir).
_RETARGET_TO_FRANKA_R = Rotation.from_euler("z", -90, degrees=True)
def retarget_to_franka_quat(q_wxyz_retarget):
    r_retarget = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q_wxyz_retarget)))
    r_franka = r_retarget * _RETARGET_TO_FRANKA_R     # right-multiply = local-frame post-rotation
    return quat_xyzw_to_wxyz(r_franka.as_quat())

def parallel_to_table_quat(ee_pos_W, obj_pos_W):
    """Quat (wxyz) — approach horizontal toward object, opening axis horizontal ⊥ approach.
    Builds R with retarget convention (local +X = opening, +Z = approach), THEN applies
    retarget→Franka swap so what we send to IK matches Franka's panda_hand axis convention."""
    delta = np.asarray(obj_pos_W) - np.asarray(ee_pos_W)
    delta_h = delta.copy(); delta_h[2] = 0.0
    if np.linalg.norm(delta_h) < 1e-6:
        delta_h = np.array([1., 0., 0.])
    ez = delta_h / np.linalg.norm(delta_h)
    ex = np.cross(np.array([0., 0., 1.]), ez)
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0., 1., 0.])
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    R = np.column_stack([ex, ey, ez])
    q_xyzw_retarget = Rotation.from_matrix(R).as_quat()
    return retarget_to_franka_quat(quat_xyzw_to_wxyz(q_xyzw_retarget))


# ── load trajectory ──────────────────────────────────────────────────────────
with h5py.File(TRAJ, "r") as h:
    states  = h["state"][:].copy()
    actions = h["action"][:].copy()
    grasp_onset_idx = int(h.attrs["grasp_onset_idx"])
    n_steps = int(h.attrs["n_steps"])
    # GT object orientation at frame 0 in G frame. If missing (old HDF5), fall back to identity.
    obj_quat_G_wxyz = np.array(h.attrs.get("obj_quat_G_wxyz", [1., 0., 0., 0.]), dtype=np.float64)
    # Object origin position in G frame — placed in sim at obj_origin_G + sim_origin_W so the
    # object exactly matches what the (object-centric) trajectory expects. None → fall back to
    # the old physics-settle behavior (for HDF5s built before this attr existed).
    obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64) if "obj_origin_G" in h.attrs else None
# Apply retarget→Franka 90° axis swap to all quats in trajectory
for arr in (states, actions):
    for k in range(arr.shape[0]):
        arr[k, 3:7] = retarget_to_franka_quat(arr[k, 3:7])
cprint(f"[{_sess_id}] object={OBJECT} ({obj_meta['name']})  loaded {n_steps} steps, gripper onset @ {grasp_onset_idx}", "cyan")
cprint(f"  state[0]:  pos={states[0,:3].round(3)} quat={states[0,3:7].round(3)} grip={states[0,7]:.0f}  (axis-swap applied)", "cyan")

# Resolve Franka base placement (manual from config OR auto from state[0]_G direction)
if "robot_pos" in cfg and "robot_ori" in cfg:
    ROBOT_POS = list(cfg["robot_pos"]); ROBOT_ORI = list(cfg["robot_ori"])
    _src = "config (manual)"
else:
    ROBOT_POS, ROBOT_ORI = auto_robot_pose(states[0, :3], SIM_ORIGIN_XY)
    _src = "auto (from state[0])"
cprint(f"  config:    sim_origin_xy={SIM_ORIGIN_XY}  robot_pos={[round(v,3) for v in ROBOT_POS]} robot_ori={[round(v,1) for v in ROBOT_ORI]}  ← {_src}", "cyan")


# ── scene ────────────────────────────────────────────────────────────────────
world = World(backend="numpy")
phys = world.get_physics_context()
phys.enable_ccd(True); phys.enable_gpu_dynamics(True); phys.set_broadphase_type("gpu")
phys.enable_stablization(True); phys.set_solver_type("TGS")
set_camera_view(eye=[1.5, 1.5, 1.5], target=[0, 0.4, 0.85], camera_prim_path="/OmniverseKit_Persp")

# ── video recording (PNG frames; ffmpeg to mp4 after) ────────────────────────
VIDEO_DIR = args.video
_video_idx = 0
_video_step = 0
if VIDEO_DIR:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    # clean old frames
    for p in os.listdir(VIDEO_DIR):
        if p.endswith(".png"): os.remove(os.path.join(VIDEO_DIR, p))
    import omni.kit.viewport.utility as _vu
    _viewport = _vu.get_active_viewport()
    cprint(f"📹 video recording on → {VIDEO_DIR}/  every {args.video_every} steps", "magenta")

# Recording stays off until we're done with the (invisible) IK setup and Franka has
# been respawned at qpos[state[0]]. This way the video opens with Franka already in
# the correct grasp-start pose — no "default USD home → teleport" detour to confuse partners.
_record_enabled = False
def _capture_step():
    global _video_idx, _video_step
    if not VIDEO_DIR or not _record_enabled: return
    _video_step += 1
    if _video_step % args.video_every != 0: return
    _vu.capture_viewport_to_file(_viewport, os.path.join(VIDEO_DIR, f"f_{_video_idx:05d}.png"))
    _video_idx += 1

# wrap world.step so every call may capture a frame
_orig_world_step = world.step
def world_step_with_capture(render=True):
    _orig_world_step(render=render)
    _capture_step()
world.step = world_step_with_capture

delete_prim("/Replicator/DomeLight_Xform")
rep.create.light(position=[0, 0, 0], light_type="dome")
GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0,
            physics_material=PhysicsMaterial(prim_path="/World/PM/g",
                                             static_friction=0.5, dynamic_friction=0.5, restitution=0.8),
            visual_material=None)
delete_prim("/World/Table")
FixedCuboid(prim_path="/World/Table", name="table", position=TABLE_POS,
            orientation=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
            scale=TABLE_SCALE, size=1.0, visible=True)
delete_prim("/World/Franka")
franka = Franka(world, np.array(ROBOT_POS), np.array(ROBOT_ORI))
world.reset()
for _ in range(SETTLE_INIT): world.step(render=True)
franka.open_gripper()

ik = KinematicsSolver(franka, end_effector_frame_name="panda_hand")
# CRITICAL: Lula IK uses world-frame targets, but it doesn't auto-discover the robot's
# world-frame base pose. Default = (0,0,0) identity — wrong when ROBOT_POS != origin.
# Must explicitly inform it of where the robot is in the world.
_franka_base_quat_wxyz = euler_angles_to_quat(np.array(ROBOT_ORI), degrees=True)
ik._kinematics.set_robot_base_pose(np.array(ROBOT_POS, dtype=np.float64),
                                    np.asarray(_franka_base_quat_wxyz, dtype=np.float64))
cprint(f"✓ Lula KinematicsSolver ready  (base_W={ROBOT_POS}, ori={ROBOT_ORI}°, ee_frame=panda_hand)", "green")

def measure_ee_W():
    """EE pose at panda_hand frame in WORLD coords (consistent with IK frame).
    Franka.get_cur_ee_pos returns panda_rightfinger which is offset 5-10cm — don't mix."""
    p, R = ik._kinematics_solver.compute_forward_kinematics("panda_hand", franka.get_joint_positions()[:7])
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.asarray(p), quat_xyzw_to_wxyz(q_xyzw)


# ── object spawn ─────────────────────────────────────────────────────────────
# The trajectory is object-centric (G-frame): EE state and object point cloud share one
# frame. To keep sim consistent, the object must sit at EXACTLY its G-frame pose, offset
# by sim_origin_W. We do NOT let physics settle it (settling lands it at an uncontrolled
# spot — verified to break the EE-vs-object geometry by ~10cm for tall objects). Instead
# we place it at obj_origin_G + sim_origin_W with obj_quat_G, then freeze it (kinematic,
# collision off) — a pure visual reference the replay drives past.
sim_origin_W = np.array([SIM_ORIGIN_XY[0], SIM_ORIGIN_XY[1], TABLE_TOP_Z])
if obj_origin_G is not None:
    obj_place_pos = obj_origin_G + sim_origin_W
else:
    obj_place_pos = np.array([sim_origin_W[0], sim_origin_W[1], TABLE_TOP_Z + obj_meta["height"] / 2])
for i in range(10): delete_prim(f"/World/Rigid/rigid_{i}")
delete_prim("/World/Rigid/rigid")
obj = RigidObject(world, usd_path=obj_meta["usd"], pos=np.array(obj_place_pos),
                  ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=0.1)
obj.rigid.set_world_pose(np.asarray(obj_place_pos, dtype=np.float64), obj_quat_G_wxyz)
cprint(f"  obj placed at G-frame pose: pos={obj_place_pos.round(3)}  quat_G(wxyz)={obj_quat_G_wxyz.round(3)}", "cyan")

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
stage = world.stage
obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
# Settle a few steps just to register the prim (object stays put — it's about to be frozen)
for _ in range(5): world.step(render=True)

ee_home_W, q_home = measure_ee_W()
cprint(f"Franka home: pos={ee_home_W.round(3)} quat={q_home.round(3)}", "cyan")


# ── OFFLINE IK: precompute the full qpos trajectory before sim runs ──────────
# Why: per-step online IK has 3 problems we want to fix:
#   (a) wrist-flip / branch-switching when IK has multiple solutions (no warm-start chain)
#   (b) failures only surface after sim already stepped → wasted work
#   (c) IK calls inside the physics loop block CPU; offline batches once
# We build ONE qpos sequence covering pre-position A + B + 73 trajectory frames,
# using Lula's warm_start (current joint positions at call time) for continuity.
# At sim run time, we just apply_action(qpos[k]) per step. Zero IK in the hot loop.

ARM_DOF = 7  # panda_joint1..7

def precompute_ik_sequence(targets_pose, seed_qpos):
    """targets_pose: list of (pos_W, quat_wxyz, label_str)
    seed_qpos: initial joint positions (used as warm-start for the FIRST IK).
    Subsequent IKs warm-start from the PREVIOUS IK's qpos → forces a continuous IK branch.
    Returns: (qpos_list, ok_list, where qpos_list[i] is np.array(ARM_DOF,) or None if failed)."""
    qpos_list, ok_list = [], []
    cur_seed = np.array(seed_qpos[:ARM_DOF], dtype=np.float64)
    for i, (pos, quat, _label) in enumerate(targets_pose):
        kw = dict(target_position=np.asarray(pos, dtype=np.float64),
                  position_tolerance=args.ik_pos_tol)
        if not args.position_only and quat is not None:
            kw["target_orientation"] = np.asarray(quat, dtype=np.float64)
            kw["orientation_tolerance"] = args.ik_ori_tol
        # Lula's IK warm-starts from the current articulation joint positions; to bias
        # toward the previous solution, we temporarily set the articulation to cur_seed.
        franka.set_joint_positions(np.concatenate([cur_seed, franka.get_joint_positions()[ARM_DOF:]]))
        action, success = ik.compute_inverse_kinematics(**kw)
        if success:
            qpos = np.asarray(action.joint_positions[:ARM_DOF], dtype=np.float64)
            qpos_list.append(qpos); ok_list.append(True)
            cur_seed = qpos          # next IK warm-starts from this qpos
        else:
            qpos_list.append(None); ok_list.append(False)
    return qpos_list, ok_list

def analyze_qpos_continuity(qpos_list, label):
    """Report max joint-step between consecutive frames (wrist flips show up as huge jumps)."""
    valid = [(i, q) for i, q in enumerate(qpos_list) if q is not None]
    if len(valid) < 2: return
    max_step_per_joint = np.zeros(ARM_DOF); max_step_frame = -1
    for k in range(1, len(valid)):
        _, q_prev = valid[k-1]; _, q_curr = valid[k]
        step = np.abs(q_curr - q_prev)
        if step.max() > max_step_per_joint.max():
            max_step_per_joint = step; max_step_frame = valid[k][0]
    cprint(f"   [{label}] joint-step max: |Δ|_∞={max_step_per_joint.max():.2f}rad ({np.rad2deg(max_step_per_joint.max()):.0f}°) "
           f"on joint{int(np.argmax(max_step_per_joint))+1} at frame {max_step_frame}; "
           f"per-joint max={np.round(np.rad2deg(max_step_per_joint),0).astype(int).tolist()}°",
           "green" if max_step_per_joint.max() < 0.5 else ("yellow" if max_step_per_joint.max() < 1.0 else "red"))


def _chain_max_step(qpos_list):
    """Largest single-frame joint angle jump across the whole chain (∞-norm). inf if any None."""
    if any(q is None for q in qpos_list): return float("inf")
    return max((np.abs(qpos_list[k] - qpos_list[k - 1]).max() for k in range(1, len(qpos_list))), default=0.0)


def find_best_ik_chain(targets, seed_candidates, label=""):
    """Try each seed candidate, run the full IK chain from it, return the chain with the
       smallest max-joint-step. This is how we dodge IK branch flips: different seeds bias
       IK toward different redundant-DoF branches; the one whose 'whole-trajectory ribbon'
       has the smallest discontinuity is the one PD can physically follow."""
    best = dict(qpos=None, max_step=float("inf"), seed_idx=-1, ok=None)
    for idx, seed in enumerate(seed_candidates):
        qpos_list, ok_list = precompute_ik_sequence(targets, seed)
        ms = _chain_max_step(qpos_list)
        flag = "✓" if ms < 0.5 else ("·" if ms < 1.5 else "✗")
        cprint(f"   [{label}] seed {idx}: IK={sum(ok_list)}/{len(ok_list)}  max_step={np.rad2deg(ms):6.0f}°  {flag}",
               "cyan" if ms < best["max_step"] else None)
        if ms < best["max_step"]:
            best = dict(qpos=qpos_list, max_step=ms, seed_idx=idx, ok=ok_list)
    return best


# ── build target sequence ────────────────────────────────────────────────────
# Rationale: training data IS the human trajectory. We DON'T drive Franka through
# a pre-position phase in sim — we set Franka directly to IK(state[0]) at scene
# init. But the IK *seed* for state[0] matters: seeding directly from default
# Franka home picks a poor branch that causes a wrist flip ~24 frames into the
# replay. So we keep a SHORT offline IK seeding chain (home → parallel-to-table
# → state[0].quat, 30 hops) just to bias the seed onto a smooth branch. No sim
# steps executed during this — it's pure IK arithmetic.
state0_pos_W = states[0, :3] + sim_origin_W
state0_quat = states[0, 3:7]
pre_quat = parallel_to_table_quat(state0_pos_W, sim_origin_W)

# Seeding chain (offline IK, no sim drive) — 30 frames is enough to bias the branch
SEED_HOPS_A = 30   # cartesian-and-quat interp: home_EE → (state[0].pos, parallel-to-table)
SEED_HOPS_B = 20   # orientation slerp at state[0].pos: parallel-to-table → state[0].quat
key_times = [0.0, 1.0]
from scipy.spatial.transform import Slerp as _Slerp
slerp_A_seed = _Slerp(key_times, Rotation.from_quat(np.vstack([quat_wxyz_to_xyzw(q_home), quat_wxyz_to_xyzw(pre_quat)])))
slerp_B_seed = _Slerp(key_times, Rotation.from_quat(np.vstack([quat_wxyz_to_xyzw(pre_quat), quat_wxyz_to_xyzw(state0_quat)])))
targets_seed_A = [((1-a/SEED_HOPS_A)*ee_home_W + (a/SEED_HOPS_A)*state0_pos_W,
                   quat_xyzw_to_wxyz(slerp_A_seed([a/SEED_HOPS_A]).as_quat()[0]),
                   f"sA_{a}") for a in range(1, SEED_HOPS_A + 1)]
targets_seed_B = [(state0_pos_W,
                   quat_xyzw_to_wxyz(slerp_B_seed([a/SEED_HOPS_B]).as_quat()[0]),
                   f"sB_{a}") for a in range(1, SEED_HOPS_B + 1)]
targets_traj = [(actions[t, :3] + sim_origin_W, actions[t, 3:7], f"T_{t}") for t in range(n_steps)]

# ── precompute IK chain: state[0] + trajectory (offline, no sim drive) ───────
# HDF5 stores action[t]=state[t+1] → targets_traj[0] is state[1]. The combined chain
# [state0]+traj: qpos[0] becomes Franka's spawn pose, qpos[1:] drives the replay.
home_qpos = franka.get_joint_positions()[:ARM_DOF].copy()
gripper_q = franka.get_joint_positions()[ARM_DOF:].copy()
state0_target = (state0_pos_W, state0_quat, "state0")
targets_combined = [state0_target] + targets_traj

def _ok_str(oks): return f"{sum(oks)}/{len(oks)} ({100*sum(oks)/max(len(oks),1):.0f}%)"

if args.ik_solver == "curobo":
    # cuRobo GPU IK: 1024-seed parallel search + minimax-DP continuity chain selection.
    # Run OUT-OF-PROCESS: cuRobo 0.8's collision module needs a newer Warp than IsaacSim
    # bundles, so importing it in this process crashes. The offline IK is a standalone
    # precompute — a fresh subprocess picks up the correct (pip-installed) Warp.
    cprint(f"\n🧮 Offline IK — cuRobo GPU solver, out-of-process  ({len(targets_combined)} frames)", "yellow")
    import subprocess
    _tag = f"/tmp/cik_{os.getpid()}"
    _cik_in, _cik_out = _tag + "_in.npz", _tag + "_out.npz"
    np.savez(_cik_in,
             pos=np.array([p for (p, q, _) in targets_combined], dtype=np.float64),
             quat=np.array([q for (p, q, _) in targets_combined], dtype=np.float64),
             robot_pos=np.array(ROBOT_POS, dtype=np.float64),
             robot_ori=np.array(ROBOT_ORI, dtype=np.float64),
             num_seeds=args.curobo_seeds, pos_tol=args.ik_pos_tol, ori_tol=args.ik_ori_tol)
    _cik_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_ik.py")
    _env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # avoid IsaacSim's Warp
    _r = subprocess.run([sys.executable, _cik_script, "--solve", _cik_in, _cik_out],
                        capture_output=True, text=True, env=_env)
    if _r.returncode != 0 or not os.path.exists(_cik_out):
        cprint(f"   ❌ FATAL: cuRobo IK subprocess failed (rc={_r.returncode})", "red")
        cprint((_r.stdout or "")[-1500:] + "\n" + (_r.stderr or "")[-1500:], "red")
        sim_app.close(); sys.exit(1)
    for _ln in _r.stdout.strip().splitlines()[-3:]:
        cprint(f"   {_ln}", "cyan")
    _d = np.load(_cik_out)
    _qp, _ok = _d["qpos"], _d["ok"]
    qpos_combined = [(_qp[i] if _ok[i] else None) for i in range(len(_ok))]
    ok_combined   = [bool(x) for x in _ok]
    qpos_state0 = qpos_combined[0]
    qpos_traj   = qpos_combined[1:]
    ok_traj     = ok_combined[1:]
    if qpos_state0 is None:
        cprint(f"   ❌ FATAL: cuRobo IK failed at state[0]", "red")
        sim_app.close(); sys.exit(1)
    cprint(f"   IK success — Traj: {_ok_str(ok_traj)}", "cyan")
    analyze_qpos_continuity(qpos_traj, "trajectory")
else:
    # ── Lula multi-seed IK (the wrist-flip fix) ──────────────────────────────
    # Same EE pose has multiple valid 7-DoF configs; consecutive frames can flip
    # branch abruptly (260° joint jumps). Run the whole chain from N seeds, keep the
    # one with the smallest joint step. Seeds: perturbations of a home→state[0]
    # seeding chain, biased on joints 5 & 7 (the flip-prone wrist DoFs).
    cprint(f"\n🧮 Offline IK chain (Lula) — seed: {len(targets_seed_A)} A + {len(targets_seed_B)} B  +  trajectory: {len(targets_traj)}", "yellow")
    qpos_sA, ok_sA = precompute_ik_sequence(targets_seed_A, home_qpos)
    qpos_sB, ok_sB = precompute_ik_sequence(targets_seed_B,
                                            next((q for q in reversed(qpos_sA) if q is not None), home_qpos))
    seed_for_state0 = next((q for q in reversed(qpos_sB) if q is not None),
                           next((q for q in reversed(qpos_sA) if q is not None), home_qpos))
    def _perturb(base, joint_idx, delta):
        p = base.copy(); p[joint_idx] += delta; return p
    seed_candidates = [seed_for_state0, home_qpos]
    for j in (4, 6):
        for d in (-2.0, -1.0, 1.0, 2.0):
            seed_candidates.append(_perturb(seed_for_state0, j, d))
            seed_candidates.append(_perturb(home_qpos,        j, d))
    best = find_best_ik_chain(targets_combined, seed_candidates, label="combined (state0+traj)")
    if best["qpos"] is None:
        cprint(f"   ❌ FATAL: no seed produced a complete IK chain", "red")
        sim_app.close(); sys.exit(1)
    cprint(f"   → picked seed #{best['seed_idx']}  max_step={np.rad2deg(best['max_step']):.1f}°  ({len(seed_candidates)} candidates tried)", "green")
    qpos_state0 = best["qpos"][0]
    qpos_traj   = best["qpos"][1:]
    ok_traj     = best["ok"][1:]
    cprint(f"   IK success — seed_A: {_ok_str(ok_sA)}  seed_B: {_ok_str(ok_sB)}  Traj: {_ok_str(ok_traj)}", "cyan")
    analyze_qpos_continuity(qpos_sA + qpos_sB, "seeding chain (offline)")
    analyze_qpos_continuity(qpos_traj, "trajectory")
if not all(q is not None for q in qpos_traj):
    bad = [i for i, q in enumerate(qpos_traj) if q is None]
    cprint(f"   ⚠️  trajectory has {len(bad)} unreachable frames: {bad[:10]}{'...' if len(bad)>10 else ''}", "red")

if qpos_state0 is None:
    cprint(f"   ❌ FATAL: IK at state[0] failed — cannot init Franka at trajectory start", "red")
    cprint(f"      (Phase 1's 5 sessions all succeed; check session config if you see this)", "red")
    sim_app.close(); sys.exit(1)


# ── re-spawn Franka with qpos[state[0]] as its DEFAULT joint state ───────────
# Cleaner than teleporting after spawn: by overriding the USD's default joint values
# *before* the next world.reset(), Franka physically materializes already at state[0].
# Video viewer never sees the "USD home pose → teleport" jump cut.
from isaacsim.core.utils.types import ArticulationAction
cprint(f"\n⚡ Re-spawn Franka at IK(state[0]) via set_joints_default_state + world.reset()", "yellow")
new_default = np.concatenate([qpos_state0, gripper_q])
franka.set_joints_default_state(positions=new_default)
world.reset()                                                                # Franka now spawns at qpos[state[0]]
franka.open_gripper()                                                        # safety: ensure gripper open after reset
franka._articulation_controller.apply_action(
    ArticulationAction(joint_positions=np.concatenate([qpos_state0, np.array([np.nan, np.nan])])))

# world.reset() snapped the object back to its USD-default pose. Re-place it at the EXACT
# G-frame pose (obj_origin_G + sim_origin_W, orientation obj_quat_G) so the object exactly
# matches what the object-centric trajectory expects — NO physics settling.
obj.rigid.set_world_pose(np.asarray(obj_place_pos, dtype=np.float64), obj_quat_G_wxyz)

ee_init, q_init = measure_ee_W()
dist_init = float(np.linalg.norm(ee_init - state0_pos_W))
quat_err_init = np.rad2deg(2 * np.arccos(min(1.0, abs(np.dot(q_init, state0_quat)))))
cprint(f"   after re-spawn: ee={ee_init.round(3)} dist={dist_init*100:.2f}cm  quat_err={quat_err_init:.1f}°",
       "green" if (dist_init < 0.02 and quat_err_init < 5) else "yellow")

# ── freeze object as a non-colliding visual reference ───────────────────────
# Baseline1 Gate 3 only verifies the EE trajectory — it does NOT grasp. The object is a
# pure visual/geometric reference placed at its exact G-frame pose. Make it kinematic
# (frozen, gravity-immune) AND collision-off so the replay drives past without the open
# gripper ramming it (a collision impulse blows up the Franka articulation → joint NaN).
_rb = UsdPhysics.RigidBodyAPI.Get(stage, obj_prim.GetPath())
if _rb:
    _ke = _rb.GetKinematicEnabledAttr()
    (_ke if _ke else _rb.CreateKinematicEnabledAttr()).Set(True)
_n_col_off = 0
for prim in Usd.PrimRange(obj_prim):
    if prim.IsA(UsdGeom.Mesh):
        _ca = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if _ca:
            _ce = _ca.GetCollisionEnabledAttr()
            (_ce if _ce else _ca.CreateCollisionEnabledAttr()).Set(False)
            _n_col_off += 1
# Re-assert the pose after freezing (kinematic flag can reset it to the prim default)
obj.rigid.set_world_pose(np.asarray(obj_place_pos, dtype=np.float64), obj_quat_G_wxyz)
for _ in range(SETTLE_AT_STATE0): world.step(render=True)            # let Franka PD lock; object stays put
obj_pos_chk, obj_quat_chk = obj.get_obj_pos()
qe = np.rad2deg(2 * np.arccos(min(1.0, abs(np.dot(obj_quat_chk, obj_quat_G_wxyz)))))
cprint(f"   object frozen at G-frame pose: pos={obj_pos_chk.round(3)} (target {np.asarray(obj_place_pos).round(3)}) "
       f"quat_err={qe:.1f}°  kinematic + collision-OFF on {_n_col_off} mesh(es)", "cyan")

# ── flip on video recording: from here the viewer sees Franka in state[0] pose ──
if VIDEO_DIR:
    _record_enabled = True
    cprint(f"📹 video recording enabled (frame 0 = Franka in state[0] pose)", "magenta")
# Brief hold so viewers can see Franka in starting pose before motion kicks in
for _ in range(30): world.step(render=True)
obj_pos_post, _ = obj.get_obj_pos()
cprint(f"   object reference at {obj_pos_post.round(3)}", "cyan")


# ── prep online driver (zero IK in loop) ─────────────────────────────────────
def drive_qpos(qpos, n_phys_steps):
    """Set joint position target → step physics. Skip if qpos is None (IK failed for this frame)."""
    if qpos is not None:
        franka._articulation_controller.apply_action(
            ArticulationAction(joint_positions=np.concatenate([qpos, np.array([np.nan, np.nan])])))
    for _ in range(n_phys_steps): world.step(render=True)


# ── Trajectory replay ────────────────────────────────────────────────────────
# Baseline1 scope: the HDF5 trajectory is the APPROACH only (already truncated at the
# grasp moment by build_gt_replay.py). We drive Franka through it and verify two things:
#   (1) Franka EE faithfully tracks the retargeted (human-faithful) trajectory
#   (2) the final EE pose lands beside the object = where the human hand grasped
# NO gripper close, NO lift, NO dz — those are out of scope for Baseline1.
cprint(f"\n🎬 Replay {n_steps} cached qpos targets (phys_per_action={args.phys_per_action})", "green")
ik_fail_count = sum(1 for q in qpos_traj if q is None)
ee_track_errs_mm = []
last_valid_qpos = qpos_state0

for t in range(n_steps):
    qp = qpos_traj[t]
    if qp is not None: last_valid_qpos = qp
    drive_qpos(qp if qp is not None else last_valid_qpos, args.phys_per_action)

    ee_now_W, _ = measure_ee_W()
    target_pos_W = actions[t, :3] + sim_origin_W
    track_err_mm = float(np.linalg.norm(ee_now_W - target_pos_W) * 1000.0)
    ee_track_errs_mm.append(track_err_mm)
    if t % 10 == 0:
        obj_now_W, _ = obj.get_obj_pos()
        cprint(f"  t={t:3d} target={target_pos_W.round(3)} ee={ee_now_W.round(3)} "
               f"track={track_err_mm:4.0f}mm", "cyan")


# ── result: did the EE faithfully reach the human grasp pose? ────────────────
for _ in range(30): world.step(render=True)
ee_final_W, q_final = measure_ee_W()
obj_final_W, _ = obj.get_obj_pos()
# Final EE target = last action (= the human grasp pose, retargeted)
final_target_W = actions[n_steps - 1, :3] + sim_origin_W
final_track_mm = float(np.linalg.norm(ee_final_W - final_target_W) * 1000.0)
# Fingertip-center → object distance: "is the gripper beside the object?"
lf_W, _ = ik._kinematics_solver.compute_forward_kinematics("panda_leftfingertip", franka.get_joint_positions()[:7])
rf_W, _ = ik._kinematics_solver.compute_forward_kinematics("panda_rightfingertip", franka.get_joint_positions()[:7])
fingertip_mid = (np.asarray(lf_W) + np.asarray(rf_W)) / 2
ft_to_obj_cm = float(np.linalg.norm(fingertip_mid - obj_final_W) * 100.0)

cprint(f"\n{'=' * 60}", "yellow")
cprint(f"=== Gate 3 RESULT (session {args.session}, object {obj_meta['name']}) ===", "yellow")
cprint(f"  Offline IK — Traj: {_ok_str(ok_traj)}", "yellow")
cprint(f"  Initial qpos jump — pos_err={dist_init*100:.2f}cm  quat_err={quat_err_init:.1f}°", "yellow")
cprint(f"  Replay unreachable frames (skipped): {ik_fail_count}/{n_steps}", "yellow")
if ee_track_errs_mm:
    cprint(f"  Replay EE tracking: avg {np.mean(ee_track_errs_mm):.0f}mm  "
           f"max {np.max(ee_track_errs_mm):.0f}mm  p50 {np.median(ee_track_errs_mm):.0f}mm", "cyan")
cprint(f"  Final EE vs human grasp pose: track_err={final_track_mm:.0f}mm  ← the real fidelity metric", "cyan")
cprint(f"  Final fingertip-center → object centroid: {ft_to_obj_cm:.1f}cm  (info only; large for tall "
       f"objects since the human grasps the body, not the centroid)", "cyan")

# Baseline1 Gate 3 PASS = the sim Franka FAITHFULLY reproduces the retargeted human
# trajectory. Three conditions:
#   (1) every frame IK-solvable        → ik_fail_count == 0
#   (2) tracking is tight              → mean replay error < 20mm
#   (3) the EE actually reaches the human grasp pose at the end → final_track < 30mm
# We do NOT gate on grasp success or fingertip-to-centroid distance — Baseline1's job is
# only to put the EE where the human hand was; grasping is out of scope.
c1 = (ik_fail_count == 0)
c2 = (len(ee_track_errs_mm) > 0 and np.mean(ee_track_errs_mm) < 20.0)
c3 = (final_track_mm < 30.0)
if c1 and c2 and c3:
    cprint(f"  ✅ GATE 3 PASS: sim Franka faithfully replays the retargeted human EE trajectory", "green")
else:
    fails = []
    if not c1: fails.append(f"IK unreachable {ik_fail_count}/{n_steps}")
    if not c2: fails.append(f"mean tracking {np.mean(ee_track_errs_mm):.0f}mm > 20mm")
    if not c3: fails.append(f"final EE off human grasp pose by {final_track_mm:.0f}mm > 30mm")
    cprint(f"  ❌ GATE 3 FAIL: {'; '.join(fails)}", "red")

sim_app.close()
