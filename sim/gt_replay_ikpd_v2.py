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
parser.add_argument("--curobo-seeds", type=int, default=1024, help="cuRobo IK seed count")
parser.add_argument("--drive", choices=["pd", "kinematic"], default="pd",
                    help="pd = PhysX implicit-PD position control (ArticulationController); "
                         "kinematic = set_joint_positions teleport (no PD dynamics — diagnostic)")
# ── opt-in DP3 closed-loop eval mode (additive; --dp3 off ⇒ identical replay behaviour) ──
parser.add_argument("--dp3", action="store_true",
                    help="run the trained DP3 policy CLOSED-LOOP instead of replaying the "
                         "recorded trajectory (reuses all of gt_replay's sim setup)")
parser.add_argument("--dp3-server", default="http://127.0.0.1:8765",
                    help="DP3 inference server base URL (--dp3 mode)")
parser.add_argument("--dp3-max-steps", type=int, default=60,
                    help="max DP3 query steps per closed-loop rollout (--dp3 mode)")
parser.add_argument("--grasp-lift", action="store_true",
                    help="after the replay, close the gripper and lift the EE +15cm")
parser.add_argument("--grasp-collision", action="store_true",
                    help="real grasp eval: object is a DYNAMIC rigid body with a convex-hull "
                         "collider, collision ON — the closing gripper actually contacts and "
                         "lifts it. Off (default) = frozen non-colliding visual reference. "
                         "Ignored in --dp3 mode (Phase 1 keeps the object frozen).")
parser.add_argument("--object-mass", type=float, default=None,
                    help="override object mass in kg for the grasp eval "
                         "(default = grasp_physics.GRASP_OBJECT_MASS_KG)")
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
import grasp_physics                       # shared grasp-physics setup (= run_grasp_sim.py)
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


# ── DP3 closed-loop helpers (only used when --dp3 is set; additive) ──────────
# These mirror sim/eval_dp3_sim.py so the closed-loop observation + EE-convention
# conversions exactly match what the DP3 policy was trained on (Baseline1).
_FRANKA_TO_RETARGET_R = Rotation.from_euler("z", +90, degrees=True)
def franka_to_retarget_quat(q_wxyz_franka):
    """Franka panda_hand convention quat → RETARGET-convention quat (both wxyz).
    Exact INVERSE of retarget_to_franka_quat (which post-multiplies by Rz(-90°)).
    Used when BUILDING observations: panda_hand FK gives the Franka-convention quat,
    but the policy was trained on retarget-convention orientations (local +X = gripper
    opening axis, +Z = approach)."""
    r_franka = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q_wxyz_franka)))
    r_retarget = r_franka * _FRANKA_TO_RETARGET_R     # post-multiply by Rz(+90°)
    return quat_xyzw_to_wxyz(r_retarget.as_quat())

# YCB class id → DexYCB CAD model folder (textured.obj). Identical mapping to
# build_gt_replay.YCB_CLASS_TO_CAD — the eval point cloud MUST come from the same mesh.
YCB_CLASS_TO_CAD = {
    1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle", 6: "007_tuna_fish_can",
    7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can",
    10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
    13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block",
    17: "037_scissors", 18: "040_large_marker", 19: "051_large_clamp",
    20: "052_extra_large_clamp", 21: "061_foam_brick",
}
_DEXYCB_RAW = f"{PROJ_ROOT}/data_hub/RawData/ThirdPersonRawData/dexycb"

def load_cad_points(ycb_class_id, n_points=4096):
    """Surface-sample the DexYCB CAD model textured.obj. Returns local CAD-frame points.
    The CAD mesh is already in metres + the CAD/object frame — NO scaling, NO extra
    rotation (Phase 1 CAD-first path; identical source to build_gt_replay)."""
    import trimesh
    cad_name = YCB_CLASS_TO_CAD.get(ycb_class_id)
    if cad_name is None:
        raise ValueError(f"no CAD mapping for ycb_class_id {ycb_class_id}")
    mesh_path = f"{_DEXYCB_RAW}/models/{cad_name}/textured.obj"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"CAD mesh missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float64)              # (N, 3) CAD local frame

def cad_points_to_world(cad_pts_local, obj_pos, obj_quat_wxyz):
    """Transform local CAD points to world frame using the object's current sim pose."""
    R = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(obj_quat_wxyz))).as_matrix()
    return (R @ cad_pts_local.T).T + np.asarray(obj_pos, dtype=np.float64)

def compute_origin_world(cad_pts_local, obj_pos, obj_quat_wxyz):
    """G-frame origin in the sim world frame (replicates build_gt_replay's
    compute_session_origin_G): object xy-centroid + object-bottom z (1st-pct z),
    measured from the CAD surface points placed at the object's current sim pose.
    Because sim gravity is -Z_world and the policy is yaw-invariant, the G-frame
    rotation relative to the world is IDENTITY → this origin IS the pure translation
    between world and G frame:  v_G = v_world - origin_world."""
    pts_world = cad_points_to_world(cad_pts_local, obj_pos, obj_quat_wxyz)
    return np.array([pts_world[:, 0].mean(),                  # object x-centroid
                     pts_world[:, 1].mean(),                  # object y-centroid
                     np.percentile(pts_world[:, 2], 1)],      # object bottom (1st pct z)
                    dtype=np.float64)

def dp3_get_policy_info(server_url):
    """GET /info → {horizon, n_obs_steps, n_action_steps, action_dim, ...}."""
    import requests
    return requests.get(f"{server_url}/info", timeout=10).json()

def dp3_query_policy(server_url, pc_obs, ap_obs, timeout=10.0):
    """POST the observation window to the DP3 server.
    pc_obs: (n_obs, N, 3)  ap_obs: (n_obs, 8) → returns action (n_action, 8)."""
    import requests
    r = requests.post(f"{server_url}/predict",
                      json={"point_cloud": pc_obs.tolist(),
                            "agent_pos":   ap_obs.tolist()},
                      timeout=timeout)
    r.raise_for_status()
    return np.asarray(r.json()["action"], dtype=np.float32)   # (n_action, 8)


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
obj_mass_kg = args.object_mass if args.object_mass is not None else grasp_physics.GRASP_OBJECT_MASS_KG
obj = RigidObject(world, usd_path=obj_meta["usd"], pos=np.array(obj_place_pos),
                  ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]),
                  mass=obj_mass_kg)
if args.object_mass is not None:
    cprint(f"  ⚖️  object mass OVERRIDE: {obj_mass_kg} kg (default {grasp_physics.GRASP_OBJECT_MASS_KG})", "yellow")
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
# We solve ONE qpos sequence (state[0] + trajectory) offline with cuRobo's
# 1024-seed GPU IK + minimax-DP continuity-chain selection.
# At sim run time, we just apply_action(qpos[k]) per step. Zero IK in the hot loop.

ARM_DOF = 7  # panda_joint1..7

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


# ── build target sequence ────────────────────────────────────────────────────
# Training data IS the human trajectory. Franka is set directly to IK(state[0]) at
# scene init (no in-sim pre-position phase). cuRobo's 1024-seed solver chooses the
# IK branch globally, so no Lula-style seeding chain is needed.
state0_pos_W = states[0, :3] + sim_origin_W
state0_quat  = states[0, 3:7]
targets_traj = [(actions[t, :3] + sim_origin_W, actions[t, 3:7], f"T_{t}") for t in range(n_steps)]

# ── precompute IK chain: state[0] + trajectory (offline, no sim drive) ───────
# HDF5 stores action[t]=state[t+1] → targets_traj[0] is state[1]. The combined chain
# [state0]+traj: qpos[0] becomes Franka's spawn pose, qpos[1:] drives the replay.
gripper_q = franka.get_joint_positions()[ARM_DOF:].copy()
state0_target = (state0_pos_W, state0_quat, "state0")
# --dp3 mode does NOT replay the recorded trajectory → solve ONLY qpos_state0.
targets_combined = [state0_target] if args.dp3 else ([state0_target] + targets_traj)

def _ok_str(oks): return f"{sum(oks)}/{len(oks)} ({100*sum(oks)/max(len(oks),1):.0f}%)"

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

# ── object physics: frozen visual reference, or dynamic+collidable for a real grasp ──
if args.grasp_collision and not args.dp3:
    # Real grasp eval (--grasp-collision): object is a DYNAMIC rigid body; the collider
    # + friction materials come from the shared sim/grasp_physics.py helper — identical
    # physics to run_grasp_sim.py. It rests on the table under gravity.
    _rb = UsdPhysics.RigidBodyAPI.Get(stage, obj_prim.GetPath())
    if _rb:
        _ke = _rb.GetKinematicEnabledAttr()
        (_ke if _ke else _rb.CreateKinematicEnabledAttr()).Set(False)   # dynamic
    _n_col = grasp_physics.setup_object_grasp_physics(
        stage, obj.rigid_prim_path, log=lambda m: cprint(m, "green"))
    grasp_physics.setup_finger_friction(stage, log=lambda m: cprint(m, "green"))
    obj.rigid.set_world_pose(np.asarray(obj_place_pos, dtype=np.float64), obj_quat_G_wxyz)
    for _ in range(SETTLE_AT_STATE0): world.step(render=True)        # object settles on the table
    obj_pos_chk, obj_quat_chk = obj.get_obj_pos()
    cprint(f"   object DYNAMIC + collision-ON ({_n_col} mesh collider(s)) — settled at "
           f"pos={obj_pos_chk.round(3)} (placed {np.asarray(obj_place_pos).round(3)})", "cyan")
else:
    # Frozen non-colliding visual reference: kinematic (gravity-immune) + collision-off so
    # the replay drives past without the open gripper ramming it (a collision impulse
    # blows up the Franka articulation → joint NaN).
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
    for _ in range(SETTLE_AT_STATE0): world.step(render=True)        # let Franka PD lock; object stays put
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
    """Drive Franka to qpos. --drive pd: PhysX implicit-PD position control. --drive
    kinematic: set_joint_positions teleport (no PD dynamics) — a diagnostic mode that
    isolates 'PD can't track' from 'cuRobo↔sim frame mismatch'. Skip if qpos is None."""
    if qpos is not None:
        if args.drive == "kinematic":
            full = np.concatenate([qpos, gripper_q])
            # PD target = qpos too, so the drive doesn't fight the teleport during steps
            franka._articulation_controller.apply_action(
                ArticulationAction(joint_positions=np.concatenate([qpos, np.array([np.nan, np.nan])])))
            franka.set_joint_positions(full)
            franka.set_joint_velocities(np.zeros(9))
            for _ in range(n_phys_steps): world.step(render=True)
            franka.set_joint_positions(full)            # re-assert: joints exact at qpos for measurement
            franka.set_joint_velocities(np.zeros(9))
            return
        franka._articulation_controller.apply_action(
            ArticulationAction(joint_positions=np.concatenate([qpos, np.array([np.nan, np.nan])])))
    for _ in range(n_phys_steps): world.step(render=True)


# ── DP3 PHASE 1: collect the closed-loop trajectory (opt-in via --dp3) ───────
# Two-phase DP3 eval. Phase 1 (this block) runs the trained DP3 policy closed-loop
# ONLY to COLLECT the trajectory it produces — it writes that trajectory to an HDF5
# and exits. Phase 2 (a separate invocation: --traj <that hdf5>, no --dp3) replays
# the collected trajectory through cuRobo's whole-trajectory continuity IK.
#
# Phase 1 is a PURE EE-SPACE rollout — there is NO Franka and NO IK in the loop.
# DP3 is an object-centric EE-space policy (obs = object point cloud + EE pose;
# action = future EE poses); its closed-loop semantics are "policy output → fed
# back as the next observation's EE". So the EE state fed back is the policy's OWN
# last action (state[0] for the first frame), not a robot FK. This keeps Phase 1
# free of robot-specific IK error; reachability/execution is assessed in Phase 2.
# This branch ends the script (sim_app.close()); the default replay path is untouched.
if args.dp3:
    cprint(f"\n🤖 DP3 PHASE 1 (collect) — server {args.dp3_server}, max_steps={args.dp3_max_steps}", "green")

    # ── server handshake: obs/action window sizes ────────────────────────────
    _info = dp3_get_policy_info(args.dp3_server)
    n_obs_steps    = int(_info["n_obs_steps"])
    n_action_steps = int(_info["n_action_steps"])
    cprint(f"   policy info: horizon={_info.get('horizon')} n_obs={n_obs_steps} "
           f"n_action={n_action_steps} action_dim={_info.get('action_dim')}", "cyan")

    # ── G-frame origin: pure translation between sim world and the training G-frame ──
    # Sample the object's CAD mesh at its current (frozen) sim world pose. The object is
    # frozen, so origin_world is computed ONCE. (G-frame rotation = identity: sim gravity
    # is -Z_world and the policy was trained with yaw augmentation → yaw-invariant.)
    dp3_cad_pts = load_cad_points(obj_meta["ycb_class_id"], n_points=4096)
    _obj_pos_now, _obj_quat_now = obj.get_obj_pos()
    origin_world = compute_origin_world(dp3_cad_pts, _obj_pos_now, _obj_quat_now)
    obj_centroid_W = cad_points_to_world(dp3_cad_pts, _obj_pos_now, _obj_quat_now).mean(axis=0)
    cprint(f"   CAD pts {dp3_cad_pts.shape}  origin_world={origin_world.round(3)}  "
           f"obj_centroid_W={obj_centroid_W.round(3)}", "cyan")

    def dp3_build_observation(ee_vec):
        """One (point_cloud, agent_pos) observation frame in the G-frame.
        point_cloud (4096,3): CAD points at the object's frozen sim pose, minus origin_world.
        agent_pos (8,): [xyz, qw,qx,qy,qz, gripper] — EE in G-frame, retarget orientation
        convention, gripper=0.0 (the training-data state keeps the gripper open through
        the whole approach — feed 0.0, NOT the policy's commanded gripper).
        ee_vec is the VIRTUAL EE state (8,): state[0] for the first frame, then the
        policy's own last action. No IK, no Franka FK — a pure EE-space rollout."""
        _op, _oq = obj.get_obj_pos()
        pc_G = (cad_points_to_world(dp3_cad_pts, _op, _oq) - origin_world).astype(np.float32)
        ee_vec = np.asarray(ee_vec, dtype=np.float32)
        agent = np.concatenate([ee_vec[:3], ee_vec[3:7],
                                [np.float32(0.0)]]).astype(np.float32)
        return pc_G, agent

    # ── virtual EE state: raw state[0] (RETARGET convention). gt_replay's in-memory
    # states[0] is axis-swapped to Franka convention, so re-read it from the HDF5. ──
    with h5py.File(TRAJ, "r") as _hsrc:
        raw_state0 = np.asarray(_hsrc["state"][0], dtype=np.float64)

    # Collected DP3 trajectory: each entry is a raw policy action sub-step (shape (8,)),
    # G-frame, RETARGET quaternion convention — the policy's raw output, verbatim.
    dp3_traj = []
    arrived_idx = None                                          # index into dp3_traj of the "arrived" step
    dp3_n_queries = 0

    # sliding-window obs buffer: filled with the frame-0 observation (virtual EE = state[0]).
    obs_window = [dp3_build_observation(raw_state0)] * n_obs_steps   # [(pc, agent)] * n_obs

    while len(dp3_traj) < args.dp3_max_steps:
        dp3_n_queries += 1
        pc_obs = np.stack([o[0] for o in obs_window])           # (n_obs, 4096, 3)
        ap_obs = np.stack([o[1] for o in obs_window])           # (n_obs, 8)
        try:
            dp3_action = dp3_query_policy(args.dp3_server, pc_obs, ap_obs)   # (n_action, 8)
        except Exception as e:
            cprint(f"   ❌ DP3 server error @ query {dp3_n_queries-1}: {e}", "red")
            break

        stop = False
        for sub in range(dp3_action.shape[0]):
            if len(dp3_traj) >= args.dp3_max_steps:
                stop = True
                break
            a = dp3_action[sub]
            # COLLECT the raw policy output (G-frame, RETARGET quat convention) verbatim.
            # This IS the virtual EE state — no IK, no Franka: the policy's own last
            # action becomes the EE fed back in the next observation.
            dp3_traj.append(np.asarray(a, dtype=np.float64).copy())
            # "arrived": the policy's gripper channel first crosses 0.5
            if float(a[7]) >= 0.5:
                arrived_idx = len(dp3_traj) - 1
                stop = True
                break

        # ── refresh the sliding window: EE fed back = the policy's own last action ──
        obs_window = obs_window[1:] + [dp3_build_observation(dp3_traj[-1])]

        _ee_g = dp3_traj[-1][:3]
        cprint(f"   query {dp3_n_queries-1:3d}: ee_G={_ee_g.round(3)}  collected={len(dp3_traj)}  "
               f"grip={dp3_action[-1,7]:.2f}", "cyan")
        if stop:
            if arrived_idx is not None:
                cprint(f"   policy signalled 'arrived' @ collected idx {arrived_idx}", "magenta")
            break

    if arrived_idx is None:
        cprint(f"   reached --dp3-max-steps ({args.dp3_max_steps}) without an 'arrived' signal",
               "yellow")

    # ── write the collected trajectory as an HDF5 for Phase 2 to replay ──────
    # Quat convention: the policy output AND raw_state0 are RETARGET-convention. Phase 2's
    # gt_replay load applies retarget_to_franka_quat itself — so we store RETARGET quats
    # here and do NOT pre-apply the axis swap. (raw_state0 was read at rollout init.)
    N = len(dp3_traj)
    dp3_action_arr = np.asarray(dp3_traj, dtype=np.float64).reshape(N, 8)
    dp3_state_arr  = np.vstack([raw_state0[None, :], dp3_action_arr])   # (N+1, 8)
    dp3_out_path = f"/tmp/dp3_traj_{OBJECT}.hdf5"
    with h5py.File(dp3_out_path, "w") as _hout:
        _hout.create_dataset("state", data=dp3_state_arr)               # (N+1, 8) RETARGET conv
        _hout.create_dataset("action", data=dp3_action_arr)             # (N, 8)   RETARGET conv
        _hout.attrs["n_steps"] = N
        _hout.attrs["grasp_onset_idx"] = arrived_idx if arrived_idx is not None else N
        _hout.attrs["obj_quat_G_wxyz"] = obj_quat_G_wxyz
        # obj_origin_G may be None for legacy source HDF5s; HDF5 attrs can't hold None,
        # so only write it when present (Phase 2 falls back to its old behaviour if absent).
        if obj_origin_G is not None:
            _hout.attrs["obj_origin_G"] = obj_origin_G

    cprint(f"\nDP3 trajectory written -> {dp3_out_path}  "
           f"({N} steps, arrived@{arrived_idx})", "green")
    sim_app.close()
    sys.exit(0)


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


# ── replay-fidelity metrics — measured at the LAST trajectory frame, BEFORE the
# grasp-lift gesture. Pre-lift is essential: the hardcoded +15cm lift would otherwise
# inflate final_track_mm and false-fail Gate 3. "Did the replay reach the human grasp
# pose" and "was the object lifted" are kept as SEPARATE metrics.
ee_final_W, q_final = measure_ee_W()
obj_final_W, _ = obj.get_obj_pos()
final_target_W = actions[n_steps - 1, :3] + sim_origin_W       # last action = human grasp pose
final_track_mm = float(np.linalg.norm(ee_final_W - final_target_W) * 1000.0)
# Fingertip-center → object distance: "is the gripper beside the object?" (also pre-lift)
lf_W, _ = ik._kinematics_solver.compute_forward_kinematics("panda_leftfingertip", franka.get_joint_positions()[:7])
rf_W, _ = ik._kinematics_solver.compute_forward_kinematics("panda_rightfingertip", franka.get_joint_positions()[:7])
fingertip_mid = (np.asarray(lf_W) + np.asarray(rf_W)) / 2
ft_to_obj_cm = float(np.linalg.norm(fingertip_mid - obj_final_W) * 100.0)


# ── optional: close the gripper + lift the EE +15cm (hard-coded grasp gesture) ───
# Opt-in via --grasp-lift. With --grasp-collision the object is dynamic + collidable, so
# a successful close+lift physically picks it up — object dz is the grasp signal,
# reported SEPARATELY from the pre-lift replay-fidelity metrics above.
obj_lift_dz = None
if args.grasp_lift:
    cprint(f"\n🤏 grasp+lift — close gripper, raise EE +15cm over 12 steps", "green")
    franka.close_gripper()
    for _ in range(HOLD_AFTER_GRIP): world.step(render=True)     # let the gripper close
    _lift_pos0, _lift_quat = measure_ee_W()                      # hold this orientation
    _lift_n = 12
    for i in range(1, _lift_n + 1):
        _lift_target = _lift_pos0 + np.array([0.0, 0.0, 0.15 * i / _lift_n])
        kw = dict(target_position=np.asarray(_lift_target, dtype=np.float64),
                  position_tolerance=args.ik_pos_tol)
        if not args.position_only:
            kw["target_orientation"] = np.asarray(_lift_quat, dtype=np.float64)
            kw["orientation_tolerance"] = args.ik_ori_tol
        _lift_action, _lift_ok = ik.compute_inverse_kinematics(**kw)
        if _lift_ok:
            drive_qpos(np.asarray(_lift_action.joint_positions[:ARM_DOF], dtype=np.float64),
                       args.phys_per_action)
        else:
            drive_qpos(None, args.phys_per_action)               # IK miss → hold
    _lift_ee, _ = measure_ee_W()
    _obj_post_lift, _ = obj.get_obj_pos()
    obj_lift_dz = float(_obj_post_lift[2] - obj_final_W[2])       # vs object pose at replay end
    cprint(f"   grasp+lift done — EE {_lift_pos0.round(3)} → {_lift_ee.round(3)}  "
           f"object dz = {obj_lift_dz*100:+.1f}cm "
           f"({'GRASPED + LIFTED' if obj_lift_dz > 0.03 else 'not lifted'})", "cyan")

for _ in range(30): world.step(render=True)   # brief idle hold for the video viewer — NOT measured

cprint(f"\n{'=' * 60}", "yellow")
cprint(f"=== Gate 3 RESULT (session {args.session}, object {obj_meta['name']}) ===", "yellow")
cprint(f"  Offline IK — Traj: {_ok_str(ok_traj)}", "yellow")
cprint(f"  Initial qpos jump — pos_err={dist_init*100:.2f}cm  quat_err={quat_err_init:.1f}°", "yellow")
cprint(f"  Replay unreachable frames (skipped): {ik_fail_count}/{n_steps}", "yellow")
if ee_track_errs_mm:
    cprint(f"  Replay EE tracking: avg {np.mean(ee_track_errs_mm):.0f}mm  "
           f"max {np.max(ee_track_errs_mm):.0f}mm  p50 {np.median(ee_track_errs_mm):.0f}mm", "cyan")
cprint(f"  Final EE vs human grasp pose: track_err={final_track_mm:.0f}mm  (replay end, pre-lift)  ← fidelity metric", "cyan")
cprint(f"  Final fingertip-center → object centroid: {ft_to_obj_cm:.1f}cm  (info only; large for tall "
       f"objects since the human grasps the body, not the centroid)", "cyan")
if obj_lift_dz is not None:
    cprint(f"  Grasp-lift: object dz = {obj_lift_dz*100:+.1f}cm  →  "
           f"{'GRASPED + LIFTED' if obj_lift_dz > 0.03 else 'not lifted'}  (grasp success, separate from fidelity)", "cyan")

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
