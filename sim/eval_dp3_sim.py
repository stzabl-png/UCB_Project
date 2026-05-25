#!/usr/bin/env python3
"""
sim/eval_dp3_sim.py
===================
Closed-loop IsaacSim evaluation for the Baseline1 DP3 policy ("Human Retarget DP").

What it does
------------
For each test YCB object:
  1. Spawn the object USD on the table as a REAL graspable physics body (convex-hull
     collision + friction), let physics settle it into a stable resting pose.
  2. Run the trained DP3 policy CLOSED-LOOP: at every DP3 step, build an observation,
     POST it to the inference server, receive `n_action_steps` future EE poses, and
     drive the Franka end-effector through them with online Lula IK (one IK / sub-step).
  3. The policy only learned the APPROACH (it ends "beside the object"; the training
     action gripper channel is 0 during approach and 1.0 on the final "arrived" frame).
     So we stop the closed loop the first time the returned action's gripper channel
     reaches >= 0.5 (the trained "arrived" signal), then run a HARDCODED grasp:
     close the gripper, lift the EE straight up ~0.15 m, settle.
  4. Success = the object was lifted more than 3 cm (obj_z_final - obj_z_initial > 0.03).

The G-frame (the most important coordinate detail)
---------------------------------------------------
Training (Baseline1/build_gt_replay.py) expresses BOTH the object point cloud and the
EE pose in the **G-frame**: gravity-aligned (+Z = -gravity) and object-translated
(origin = the object xy-centroid, z = object bottom). The eval observation MUST be in
that same frame or the policy sees garbage.

In sim, gravity is exactly -Z_world, so G-frame +Z == +Z_world. The policy was trained
with yaw augmentation -> it is yaw-invariant -> the G-frame rotation relative to the sim
world can be taken as the IDENTITY. Therefore **G-frame <-> world is a pure TRANSLATION**
(no rotation matrix). The translation is `origin_world`, the G-frame origin expressed in
world. We replicate `build_gt_replay.compute_session_origin_G`:

    origin_world = [ pts_world[:,0].mean(),                # object x-centroid
                     pts_world[:,1].mean(),                # object y-centroid
                     numpy.percentile(pts_world[:,2], 1) ] # object bottom (1st pct z)

where `pts_world` are the object's CAD surface points placed at its current sim world
pose. The object is static during the approach, so we compute `origin_world` ONCE at
rollout start. Conversion is then simply:

    v_G     = v_world - origin_world
    v_world = v_G     + origin_world

Observation construction (must match training exactly)
------------------------------------------------------
* point_cloud (n_obs, 4096, 3): sample 4096 surface points from the object's DexYCB CAD
  mesh (`models/{cad_name}/textured.obj`, already in metres + CAD frame -- no scaling, no
  extra rotation), transform them by the object's current sim world pose, then subtract
  `origin_world` -> G-frame point cloud. This is the SAME CAD-mesh source as
  `build_gt_replay.get_object_points` (Phase 1 is the CAD-first path).
* agent_pos (n_obs, 8): Franka EE pose in the G-frame, [x,y,z, qw,qx,qy,qz, gripper].
  - EE position: panda_hand world pose via Lula FK (exactly like gt_replay.measure_ee_W),
    minus origin_world.
  - EE orientation: training uses the RETARGET convention (local +X = gripper opening
    axis, +Z = approach). Franka's panda_hand uses +Y = opening, +Z = approach. So the
    panda_hand quaternion is converted to the retarget convention before being fed to
    the policy (`franka_to_retarget_quat`, post-multiply by Rz(+90 deg)). This is the
    exact INVERSE of gt_replay's `retarget_to_franka_quat` (post-multiply by Rz(-90)).
  - gripper channel: 0.0 while the gripper is open (the whole approach).

Architecture
------------
The DP3 policy runs in a SEPARATE process (Baseline1/eval/dp3_inference_server.py, conda
env `dp3`, HTTP on port 8765) because IsaacSim (`env_isaaclab`) and DP3 cannot share a
Python. This script assumes the server is already running and talks to it over HTTP.

Usage (in env_isaaclab; DP3 server must already be up in env dp3):
  /home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/eval_dp3_sim.py \\
      --objects 2 4 5 --n-rollouts 3 --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys, json, time, glob

# ── the 16 Baseline1 "KEEP" YCB class ids (all have a CAD USD in output/obj_usd_cad) ──
KEEP_CLASS_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 18, 20, 21]

parser = argparse.ArgumentParser(description="Baseline1 DP3 closed-loop IsaacSim eval")
parser.add_argument("--objects", nargs="+", type=int, default=KEEP_CLASS_IDS,
                    help="YCB class ids to test (default = the 16 KEEP ids)")
parser.add_argument("--n-rollouts", type=int, default=3, help="rollouts per object")
parser.add_argument("--max-steps", type=int, default=60,
                    help="max DP3 query steps per rollout (each yields n_action_steps sub-actions)")
parser.add_argument("--server-url", default="http://127.0.0.1:8765",
                    help="DP3 inference server base URL")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--result-dir", default="output/dp3_eval",
                    help="directory for the JSON results file")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", default=None,
                    help="if set, record the run: PNG frames to this dir + an mp4 in replay_video_check/")
parser.add_argument("--video-every", type=int, default=1,
                    help="capture one viewport frame every N sim steps")
args, _ = parser.parse_known_args()

sim_app = SimulationApp({"headless": args.headless})

import numpy as np
import trimesh
import requests
import h5py
from scipy.spatial.transform import Rotation
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
import omni.replicator.core as rep

SIM_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SIM_DIR)
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject

# ── constants ────────────────────────────────────────────────────────────────
PROJ_ROOT = "/home/accelerator/UCB_Project"
# DexYCB CAD models — same root build_gt_replay.py samples point clouds from.
RAW       = f"{PROJ_ROOT}/data_hub/RawData/ThirdPersonRawData/dexycb"

# Scene geometry — copied verbatim from gt_replay_ikpd_v2.py for cross-comparability.
TABLE_POS   = [0.0, 1.0, 0.75]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80
SETTLE_INIT = 50      # physics settle steps after Franka spawn / object re-spawn

# Where the object is dropped on the table (xy); same idea as gt_replay's sim_origin.
OBJECT_XY        = (0.0, 0.30)
SETTLE_OBJ_STEPS = 50      # let the spawned object fall + settle into a resting pose
N_PC_POINTS      = 4096    # DP3 point-cloud size (matches build_gt_replay --n-points)
ARM_DOF          = 7       # panda_joint1..7
SUCCESS_DZ       = 0.03    # 3 cm lift threshold
GRIP_ARRIVE_THR  = 0.5     # action gripper channel >= this => "arrived", stop the loop
APPROACH_OK_DIST = 0.12    # m — fingertip-centre within this of the object centroid = "reached"
LIFT_HEIGHT      = 0.15    # m — hardcoded straight-up lift distance
LIFT_STEPS       = 40      # sim steps over which to perform the lift
LIFT_SETTLE      = 30      # settle steps after the lift before measuring success

# YCB class id -> DexYCB CAD model folder (textured.obj). Identical mapping to
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


# ── quaternion helpers (IsaacSim/USD use wxyz; scipy uses xyzw) ───────────────
def quat_wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def quat_xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

# CONVENTION FIX (see header). The retarget convention has local +X = gripper opening
# axis and +Z = approach; Franka's panda_hand has local +Y = opening and +Z = approach.
# They differ by a 90 deg rotation about the local +Z (approach) axis.
#
#   retarget -> Franka : post-multiply by Rz(-90 deg)   (gt_replay's retarget_to_franka_quat)
#   Franka -> retarget : post-multiply by Rz(+90 deg)   (the inverse — used to build obs)
_RETARGET_TO_FRANKA_R = Rotation.from_euler("z", -90, degrees=True)
_FRANKA_TO_RETARGET_R = Rotation.from_euler("z", +90, degrees=True)


def retarget_to_franka_quat(q_wxyz_retarget):
    """RETARGET-convention quat -> Franka panda_hand convention quat (both wxyz).
    Used when EXECUTING actions: the policy outputs retarget-convention orientations,
    Lula IK expects panda_hand-convention orientations. Identical to gt_replay's helper."""
    r_retarget = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q_wxyz_retarget)))
    r_franka = r_retarget * _RETARGET_TO_FRANKA_R     # right-multiply = local-frame post-rotation
    return quat_xyzw_to_wxyz(r_franka.as_quat())


def franka_to_retarget_quat(q_wxyz_franka):
    """Franka panda_hand convention quat -> RETARGET-convention quat (both wxyz).
    The exact INVERSE of retarget_to_franka_quat. Used when BUILDING observations: FK
    gives the panda_hand quaternion, but the policy was trained on retarget-convention
    orientations, so we swap the axis convention before feeding the policy."""
    r_franka = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q_wxyz_franka)))
    r_retarget = r_franka * _FRANKA_TO_RETARGET_R     # post-multiply by Rz(+90 deg)
    return quat_xyzw_to_wxyz(r_retarget.as_quat())


def auto_robot_pose(obj_xy, reach_dist=0.40, base_z=0.80):
    """Place the Franka base `reach_dist` outward from the object along the world->object
    direction (in xy), facing the object. Mirrors gt_replay_ikpd_v2.auto_robot_pose but
    keyed on the object's sim world xy instead of a G-frame state[0]."""
    out = np.array([obj_xy[0], obj_xy[1]], dtype=np.float64)
    n = np.linalg.norm(out)
    out = out / n if n > 1e-3 else np.array([1.0, 0.0])
    robot_xy = (obj_xy[0] + reach_dist * out[0], obj_xy[1] + reach_dist * out[1])
    yaw_deg = float(np.degrees(np.arctan2(-out[1], -out[0])))   # face back toward the object
    return [robot_xy[0], robot_xy[1], base_z], [0.0, 0.0, yaw_deg]


# ── object CAD point cloud (matches build_gt_replay.get_object_points) ───────
def load_cad_points(ycb_class_id, n_points=N_PC_POINTS):
    """Surface-sample the DexYCB CAD model textured.obj. Returns local CAD-frame points.
    The CAD mesh is already in metres and in the pose_y/object frame — NO scaling, NO
    extra rotation (Phase 1 is the CAD-first path; identical to build_gt_replay)."""
    cad_name = YCB_CLASS_TO_CAD.get(ycb_class_id)
    if cad_name is None:
        raise ValueError(f"no CAD mapping for ycb_class_id {ycb_class_id}")
    mesh_path = f"{RAW}/models/{cad_name}/textured.obj"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"CAD mesh missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float64)          # (N, 3) in CAD local frame


def cad_points_to_world(cad_pts_local, obj_pos, obj_quat_wxyz):
    """Transform local CAD points to world frame using the object's current sim pose
    (rotation + translation). obj_quat is wxyz (USD convention)."""
    R = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(obj_quat_wxyz))).as_matrix()
    return (R @ cad_pts_local.T).T + np.asarray(obj_pos, dtype=np.float64)


def compute_origin_world(cad_pts_local, obj_pos, obj_quat_wxyz):
    """Replicate build_gt_replay.compute_session_origin_G in the sim world frame.

    The G-frame origin = the object xy-centroid + object-bottom z, all measured from the
    object's CAD surface points placed at its current sim world pose. Because the sim
    G-frame == world rotated by IDENTITY (gravity is -Z_world, policy is yaw-invariant),
    this origin IS the pure translation between the G-frame and the world frame."""
    pts_world = cad_points_to_world(cad_pts_local, obj_pos, obj_quat_wxyz)
    return np.array([pts_world[:, 0].mean(),                  # object x-centroid
                     pts_world[:, 1].mean(),                  # object y-centroid
                     np.percentile(pts_world[:, 2], 1)],      # object bottom (1st pct z)
                    dtype=np.float64)


# ── reference object poses from real DexYCB sessions ────────────────────────
def load_ref_poses(ycb_class_id, n, rng):
    """Pick n reference object poses from real DexYCB session HDF5s for this object.
    Returns [(obj_quat_G_wxyz, obj_origin_G, session_name), ...].

    Placing the sim object at a recorded DexYCB pose keeps the eval IN-DISTRIBUTION:
    the policy was trained on the object at exactly these orientations. (Dropping the
    object at identity + physics-settle instead gives an arbitrary resting pose — e.g.
    a cracker box stands tall, which DexYCB rarely has — so the policy never saw it.)"""
    ycb_dex_id = "ycb_dex_%02d" % ycb_class_id
    files = sorted(glob.glob(
        f"{PROJ_ROOT}/Baseline1/data/episodes_g/dexycb__*__{ycb_dex_id}.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"no reference HDF5 for {ycb_dex_id} under Baseline1/data/episodes_g/")
    idx = rng.choice(len(files), size=n, replace=(n > len(files)))
    out = []
    for i in idx:
        with h5py.File(files[int(i)], "r") as h:
            q  = np.array(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)   # CAD->G, wxyz
            oo = np.array(h.attrs["obj_origin_G"],   dtype=np.float64)    # object origin in G
            s0 = np.array(h["state"][0],             dtype=np.float64)    # EE start pose, G-frame (retarget conv)
        sess = os.path.basename(files[int(i)]).split("__")[2]
        out.append((q, oo, s0, sess))
    return out


# ── DP3 inference server HTTP client (reused from eval_dp3_policy.py) ─────────
def get_policy_info(server_url):
    """GET /info -> horizon, n_obs_steps, n_action_steps, action_dim, ..."""
    return requests.get(f"{server_url}/info", timeout=10).json()


def query_policy(server_url, pc_obs, ap_obs, timeout=10.0):
    """POST observation window to the DP3 server.
    pc_obs: (n_obs, N, 3)  ap_obs: (n_obs, 8).  Returns action (n_action, 8)."""
    r = requests.post(f"{server_url}/predict",
                      json={"point_cloud": pc_obs.tolist(),
                            "agent_pos":   ap_obs.tolist()},
                      timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return np.asarray(d["action"], dtype=np.float32)        # (n_action, 8)


# ── scene + IK globals (built once, objects swapped per loop) ─────────────────
world  = None
franka = None
ik     = None
ROBOT_POS = None
ROBOT_ORI = None
HOME_QPOS = None      # Franka home joint config — snapshot once, reused to reset per rollout


def measure_ee_W():
    """EE pose at the panda_hand frame in WORLD coords, via Lula forward kinematics —
    exactly like gt_replay_ikpd_v2.measure_ee_W. Returns (pos, quat_wxyz). Note: this is
    the panda_hand frame (the IK frame), NOT panda_rightfinger (Franka.get_cur_ee_pos)."""
    p, R = ik._kinematics_solver.compute_forward_kinematics(
        "panda_hand", franka.get_joint_positions()[:ARM_DOF])
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.asarray(p), quat_xyzw_to_wxyz(q_xyzw)


def setup_scene():
    """Build the persistent scene ONCE: physics context, lighting, ground, table, Franka,
    and the Lula KinematicsSolver. The object is spawned per-rollout (see spawn_object)."""
    global world, franka, ik, ROBOT_POS, ROBOT_ORI, HOME_QPOS

    world = World(backend="numpy")
    phys = world.get_physics_context()
    phys.enable_ccd(True)
    phys.enable_gpu_dynamics(True)
    phys.set_broadphase_type("gpu")
    phys.enable_stablization(True)
    phys.set_solver_type("TGS")
    set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.4, 0.85],
                    camera_prim_path="/OmniverseKit_Persp")

    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")
    GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0,
                physics_material=PhysicsMaterial(prim_path="/World/PM/g",
                                                 static_friction=0.5, dynamic_friction=0.5,
                                                 restitution=0.8),
                color=np.array([0.08, 0.08, 0.10]))   # near-black floor (high contrast)
    delete_prim("/World/Table")
    FixedCuboid(prim_path="/World/Table", name="table", position=TABLE_POS,
                orientation=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True)

    # Franka base — auto-placed so the object on the table is reachable.
    ROBOT_POS, ROBOT_ORI = auto_robot_pose(OBJECT_XY)
    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POS), np.array(ROBOT_ORI))
    world.reset()
    for _ in range(SETTLE_INIT):
        world.step(render=True)
    franka.open_gripper()
    HOME_QPOS = franka.get_joint_positions().copy()   # snapshot home pose for per-rollout reset

    ik = KinematicsSolver(franka, end_effector_frame_name="panda_hand")
    # CRITICAL: Lula IK uses world-frame targets but does NOT auto-discover the robot's
    # world base pose (default = origin/identity, wrong when ROBOT_POS != origin). We must
    # explicitly register where the robot is. (Same as gt_replay_ikpd_v2.py.)
    base_quat_wxyz = euler_angles_to_quat(np.array(ROBOT_ORI), degrees=True)
    ik._kinematics.set_robot_base_pose(np.array(ROBOT_POS, dtype=np.float64),
                                       np.asarray(base_quat_wxyz, dtype=np.float64))
    cprint(f"scene ready — Franka base_W={[round(v,3) for v in ROBOT_POS]} "
           f"ori={ROBOT_ORI} deg, Lula ee_frame=panda_hand", "green")


def spawn_object(ycb_class_id):
    """Spawn the object USD and FREEZE it (kinematic + collision OFF — see the freeze block
    below). It is then placed per-rollout at a recorded DexYCB pose; being kinematic it
    cannot fall or topple, and with collision off the gripper passes through it (no physics
    explosion). The eval measures the APPROACH against this fixed, in-distribution pose."""
    ycb_dex_id = "ycb_dex_%02d" % ycb_class_id
    usd_path = f"{PROJ_ROOT}/output/obj_usd_cad/ycb/{ycb_dex_id}.usd"
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"object USD missing: {usd_path}")

    # clean any object from a previous rollout
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")

    # spawn slightly above the table at identity orientation, then let it settle
    spawn_pos = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + 0.12])
    obj = RigidObject(world, usd_path=usd_path, pos=spawn_pos,
                      ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=0.4)

    # ── FREEZE the object ────────────────────────────────────────────────────
    # kinematic     -> gravity / contacts cannot move it: it cannot fall or topple
    #                  (the old "drop + physics-settle" toppled standing boxes before
    #                   the rollout even started).
    # collision off -> the gripper passes straight through: no contact, no physics
    #                  explosion. The eval measures the APPROACH, not the grasp.
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = world.stage
    obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
    UsdPhysics.RigidBodyAPI.Apply(obj_prim).CreateKinematicEnabledAttr().Set(True)
    for prim in Usd.PrimRange(obj_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    return obj


# ── initialize the Franka EE at a recorded session start pose ────────────────
def init_franka_to_pose(pos_world, q_franka_wxyz):
    """Initialize the Franka so panda_hand starts AT the given world pose (the recorded
    session's EE start state[0]). IK to it, teleport the arm joints there, hold via PD.
    Returns True on IK success."""
    action_obj, success = ik.compute_inverse_kinematics(
        target_position=np.asarray(pos_world, dtype=np.float64),
        target_orientation=np.asarray(q_franka_wxyz, dtype=np.float64))
    if not success:
        return False
    qpos7 = np.asarray(action_obj.joint_positions[:ARM_DOF], dtype=np.float64)
    grip  = np.asarray(franka.get_joint_positions()[ARM_DOF:], dtype=np.float64)
    full  = np.concatenate([qpos7, grip])
    franka.set_joint_positions(full)
    franka._articulation_controller.apply_action(ArticulationAction(joint_positions=full))
    for _ in range(15):
        world.step(render=True)
    return True


# ── online IK step (warm-started; mirrors gt_replay.precompute_ik_sequence) ──
def drive_ee_to(pos_world, q_franka_wxyz, render=True):
    """Drive the Franka EE to (pos_world, q_franka) with ONE online Lula IK call,
    warm-started from the current arm joints, then step physics once.

    Warm-start technique (copied from gt_replay.precompute_ik_sequence): Lula warm-starts
    its IK from the current articulation joint positions, so to bias toward a continuous
    solution we just leave the arm where it is and call compute_inverse_kinematics. The
    2 gripper joints are commanded with np.nan = "do not command them" (so a separate
    open/close_gripper call retains control of the fingers).

    Returns True if the IK solve succeeded."""
    action_obj, success = ik.compute_inverse_kinematics(
        target_position=np.asarray(pos_world, dtype=np.float64),
        target_orientation=np.asarray(q_franka_wxyz, dtype=np.float64))
    if success:
        qpos7 = np.asarray(action_obj.joint_positions[:ARM_DOF], dtype=np.float64)
        franka._articulation_controller.apply_action(
            ArticulationAction(joint_positions=np.concatenate(
                [qpos7, np.array([np.nan, np.nan])])))   # nan,nan = leave gripper alone
    world.step(render=render)
    return bool(success)


def build_observation(cad_pts_local, obj, origin_world, gripper_state):
    """Build ONE (point_cloud, agent_pos) observation frame in the G-frame.

    point_cloud (N,3): CAD points at the object's current sim world pose, minus origin_world.
    agent_pos   (8,) : [x,y,z, qw,qx,qy,qz, gripper] — EE pose in the G-frame, retarget
                       orientation convention, gripper channel = gripper_state."""
    obj_pos, obj_quat = obj.get_obj_pos()                       # world pose, quat wxyz
    pts_world = cad_points_to_world(cad_pts_local, obj_pos, obj_quat)
    pc_G = (pts_world - origin_world).astype(np.float32)        # world -> G = subtract origin

    ee_pos_world, ee_quat_franka = measure_ee_W()               # panda_hand FK (world)
    ee_pos_G = (ee_pos_world - origin_world).astype(np.float32) # world -> G
    # panda_hand orientation -> retarget convention (what the policy was trained on)
    ee_quat_retarget = franka_to_retarget_quat(ee_quat_franka).astype(np.float32)
    agent = np.concatenate([ee_pos_G, ee_quat_retarget,
                            [np.float32(gripper_state)]]).astype(np.float32)
    return pc_G, agent


# ── one closed-loop rollout ───────────────────────────────────────────────────
def rollout_one(obj, cad_pts_local, info, server_url, max_steps, origin_world):
    """One closed-loop DP3 rollout against the FROZEN (kinematic, collision-free) object.
    Runs the policy's approach and measures whether it brings the gripper to a graspable
    pose beside the object. No grasp/lift — the object is a ghost. Returns dict(...)."""
    n_obs = int(info["n_obs_steps"])

    obj_pos0, obj_quat0 = obj.get_obj_pos()
    obj_pos0 = np.asarray(obj_pos0, dtype=np.float64)
    obj_ctr  = cad_points_to_world(cad_pts_local, obj_pos0, obj_quat0).mean(axis=0)
    ee0, _   = measure_ee_W()
    cprint(f"    obj(frozen) @ {np.round(obj_pos0,3)}  centroid {np.round(obj_ctr,3)}  "
           f"EE@state0 {np.round(ee0,3)}  |EE-obj| {np.linalg.norm(ee0-obj_ctr)*100:.1f}cm", "cyan")

    # ── sliding-window observation buffer: fill with the frame-0 obs repeated ──
    obs0 = build_observation(cad_pts_local, obj, origin_world, gripper_state=0.0)
    obs_window = [obs0] * n_obs                                 # [(pc,agent)] * n_obs
    min_dist = 1e9

    arrived = False
    n_dp3_steps = 0
    last_grip_signal = 0.0
    ik_fail = 0

    for step in range(max_steps):
        n_dp3_steps += 1
        # stack the last n_obs observations -> (n_obs, N, 3) and (n_obs, 8)
        pc_obs = np.stack([o[0] for o in obs_window])
        ap_obs = np.stack([o[1] for o in obs_window])

        try:
            action = query_policy(server_url, pc_obs, ap_obs)   # (n_action, 8) G-frame, retarget
        except Exception as e:
            cprint(f"    policy server error @ step {step}: {e}", "red")
            return {"success": False, "dz": 0.0, "n_dp3_steps": step,
                    "error": str(e)}

        # ── execute each action sub-step via online warm-started IK ──
        for sub in range(action.shape[0]):
            a = action[sub]
            # action position is in the G-frame -> add origin_world to get world coords
            pos_world = (a[:3].astype(np.float64) + origin_world)
            # action orientation is retarget convention -> swap to Franka panda_hand conv.
            q_franka  = retarget_to_franka_quat(a[3:7].astype(np.float64))
            grip_sig  = float(a[7])
            last_grip_signal = grip_sig

            if not drive_ee_to(pos_world, q_franka, render=True):
                ik_fail += 1

            # the trained "arrived" signal: gripper channel first reaches >= 0.5
            if grip_sig >= GRIP_ARRIVE_THR:
                arrived = True
                break

        # ── refresh the sliding window with the newest world observation ──
        new_obs = build_observation(cad_pts_local, obj, origin_world, gripper_state=0.0)
        obs_window = obs_window[1:] + [new_obs]

        # ── per-step diagnostics: fingertip-centre vs the (frozen) object centroid ──
        ee_w, ee_q = measure_ee_W()
        approach = Rotation.from_quat(quat_wxyz_to_xyzw(ee_q)).as_matrix()[:, 2]
        tip = ee_w + 0.10 * approach                       # ~ fingertip centre (panda_hand +Z)
        d_tip = float(np.linalg.norm(tip - obj_ctr))
        min_dist = min(min_dist, d_tip)
        cprint(f"    step {step}: EE_w={np.round(ee_w,3)}  tip-vs-obj {d_tip*100:5.1f}cm  "
               f"grip={last_grip_signal:.2f}", "white")

        if arrived:
            cprint(f"    step {step}: policy 'arrived' (grip {last_grip_signal:.2f})", "magenta")
            break

    if not arrived:
        cprint(f"    max-steps ({max_steps}) reached without an 'arrived' signal "
               f"(last grip {last_grip_signal:.2f})", "yellow")

    # ── approach quality: did the gripper reach a graspable pose beside the object? ──
    ee_w, ee_q = measure_ee_W()
    approach = Rotation.from_quat(quat_wxyz_to_xyzw(ee_q)).as_matrix()[:, 2]
    tip = ee_w + 0.10 * approach
    final_dist = float(np.linalg.norm(tip - obj_ctr))
    reached = final_dist < APPROACH_OK_DIST
    cprint(f"    result: fingertip-vs-obj  final {final_dist*100:.1f}cm  min {min_dist*100:.1f}cm  "
           f"arrived={arrived}  ik_fail={ik_fail}  {'REACHED' if reached else 'MISS'}",
           "green" if reached else "red")
    return {"success": bool(reached), "final_dist": final_dist, "min_dist": min_dist,
            "n_dp3_steps": n_dp3_steps, "arrived": arrived, "ik_fail": ik_fail}


# ── video recording (optional, --video) — mirrors gt_replay_ikpd_v2.py's hook ──
_video_idx = 0
_video_n   = 0
_viewport  = None

def setup_video():
    """If --video is set, wrap world.step so every Nth sim step captures the viewport
    to a PNG. Called once, right after setup_scene() builds `world`."""
    global _viewport
    if not args.video:
        return
    os.makedirs(args.video, exist_ok=True)
    for p in os.listdir(args.video):
        if p.endswith(".png"):
            os.remove(os.path.join(args.video, p))
    import omni.kit.viewport.utility as vu
    _viewport = vu.get_active_viewport()
    _orig_step = world.step
    def step_with_capture(render=True):
        global _video_idx, _video_n
        _orig_step(render=render)
        _video_n += 1
        if _video_n % args.video_every == 0:
            vu.capture_viewport_to_file(
                _viewport, os.path.join(args.video, f"f_{_video_idx:05d}.png"))
            _video_idx += 1
    world.step = step_with_capture
    cprint(f"video recording -> {args.video}  (1 frame / {args.video_every} sim steps)", "magenta")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(args.seed)
    result_dir = os.path.join(PROJ_DIR, args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # query the DP3 server once for the obs/action shapes
    cprint(f"DP3 server: {args.server_url}", "cyan")
    info = get_policy_info(args.server_url)
    cprint(f"  horizon={info['horizon']}  n_obs={info['n_obs_steps']}  "
           f"n_action={info['n_action_steps']}  action_dim={info['action_dim']}", "cyan")

    setup_scene()
    setup_video()

    all_results = {}
    for class_id in args.objects:
        ycb_dex_id = "ycb_dex_%02d" % class_id
        cad_name = YCB_CLASS_TO_CAD.get(class_id, "?")
        cprint(f"\n=== object {ycb_dex_id} (class {class_id}, CAD {cad_name}) ===", "yellow")
        try:
            cad_pts_local = load_cad_points(class_id, N_PC_POINTS)
        except Exception as e:
            cprint(f"  skip {ycb_dex_id}: {e}", "red")
            continue
        cprint(f"  CAD pts: {cad_pts_local.shape}  "
               f"bbox {np.round(cad_pts_local.min(0),3)} -> {np.round(cad_pts_local.max(0),3)}",
               "cyan")

        try:
            obj = spawn_object(class_id)
            ref_poses = load_ref_poses(class_id, args.n_rollouts, rng)
        except Exception as e:
            cprint(f"  skip {ycb_dex_id}: {e}", "red")
            continue

        n_succ = 0
        runs = []
        for k in range(args.n_rollouts):
            q_G, oo_G, state0, sess = ref_poses[k]
            cprint(f"  --- rollout {k+1}/{args.n_rollouts}  (DexYCB session {sess}) ---", "yellow")
            franka.open_gripper()
            # ── place the object FROZEN at the recorded DexYCB pose: orientation obj_quat_G,
            #    object origin obj_origin_G[2] above the table. Kinematic → it stays put. ──
            place_pos = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + oo_G[2]])
            obj.rigid.set_world_pose(place_pos, q_G)
            for _ in range(5):
                world.step(render=True)                        # register the kinematic pose
            op0, oq0 = obj.get_obj_pos()
            origin_world = compute_origin_world(cad_pts_local, op0, oq0)
            # ── initialize the Franka EE at the session's recorded start pose state[0] ──
            ee0_world = state0[:3] + origin_world
            ee0_quat  = retarget_to_franka_quat(state0[3:7])
            if not init_franka_to_pose(ee0_world, ee0_quat):
                cprint(f"    IK failed for state[0] start pose -> skip rollout {k}", "red")
                runs.append({"rollout": k, "success": False, "error": "init-IK-fail"})
                continue
            try:
                result = rollout_one(obj, cad_pts_local, info, args.server_url,
                                     args.max_steps, origin_world)
            except Exception as ex:
                cprint(f"    rollout {k} crashed: {type(ex).__name__}: {ex}", "red")
                result = {"success": False, "n_dp3_steps": -1, "error": repr(ex)}
            runs.append({"rollout": k, **result})
            n_succ += int(result.get("success", False))

        rate = n_succ / max(args.n_rollouts, 1)
        all_results[ycb_dex_id] = {"class_id": class_id, "n_total": args.n_rollouts,
                                   "n_success": n_succ, "rate": rate, "runs": runs}
        cprint(f"  {ycb_dex_id}: {n_succ}/{args.n_rollouts} = {rate*100:.0f}% success",
               "green" if n_succ > 0 else "red")

    # ── overall summary + JSON results file ──────────────────────────────────
    total = sum(r["n_total"] for r in all_results.values())
    succ  = sum(r["n_success"] for r in all_results.values())
    cprint(f"\n=== overall: {succ}/{total} = "
           f"{(succ/total*100) if total else 0:.0f}% success ===",
           "green" if succ > 0 else "red")

    out_path = os.path.join(result_dir, f"eval_dp3_sim_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "policy_info": info,
                   "overall": {"n_total": total, "n_success": succ,
                               "rate": (succ / total) if total else 0.0},
                   "results": all_results}, f, indent=2)
    cprint(f"wrote results -> {out_path}", "cyan")

    if args.video and _video_idx > 0:
        vdir = os.path.join(PROJ_DIR, "replay_video_check")
        os.makedirs(vdir, exist_ok=True)
        mp4 = os.path.join(vdir, "eval_dp3_sim.mp4")
        os.system(f"ffmpeg -y -framerate 20 -i '{args.video}/f_%05d.png' "
                  f"-c:v libx264 -pix_fmt yuv420p '{mp4}' >/dev/null 2>&1")
        cprint(f"video ({_video_idx} frames) -> {mp4}", "magenta")

    sim_app.close()


if __name__ == "__main__":
    main()
