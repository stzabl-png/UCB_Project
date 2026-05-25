#!/usr/bin/env python3
"""
sim/eval_dp3_baseline3.py
=========================
Closed-loop DP3 evaluation for the baseline_3 cuRobo policy.

Scene
-----
Identical to sim/run_grasp_sim_baseline3.py:
  - Franka at ROBOT_POSITION / ROBOT_ORIENTATION (constants below).
  - Table at TABLE_POSITION / TABLE_SCALE.
  - Object spawned at OBJECT_XY (fixed; xy-position of the object is the same
    every episode). Only the object's orientation (obj_quat_G) is varied; it is
    pulled from a training episode in Baseline1/data/episodes_b3_curobo/ to
    guarantee the policy has seen this exact pose.

Rollout: chunked receding horizon (the DP3 / Diffusion Policy paper default)
-----
At every chunk:
  1. Read sim observation: object point cloud (CAD-sampled, G-frame) +
     panda_hand pose (G-frame, retarget convention) + gripper state.
  2. Append to the n_obs_steps observation window.
  3. POST /predict to the DP3 inference server → returns (n_action_steps, 8)
     future EE waypoints in the G-frame, retarget convention.
  4. cuRobo IK chain (sim/curobo_ik.py subprocess; NO Lula) on the 8 waypoints
     → 8 joint targets. Drive Franka through them with set_joint_positions
     (the object is pinned at its retarget pose throughout, like baseline_3).
  5. Stop when the policy emits gripper >= 0.5 or after --max-chunks chunks.

After rollout, close + lift (works around the baseline_3 grip-schedule bug
that fires the close signal only at the lift apex):
  - Pick the EE pose with the smallest distance to the object centroid across
    the whole rollout → that is the "grasp candidate".
  - Call sim/run_grasp_sim_baseline3.py's cuRobo plan_grasp pipeline
    (solve_plan_grasp; copied locally) to plan a fresh approach + grasp + lift
    from the *current* Franka joint state to the grasp candidate.
  - Drive only the grasp segment (final straight-in along tool z, fingers
    auto-disabled by cuRobo) + close gripper + drive the lift segment.

Success: object z-displacement > 3 cm (the same dz>0.03m criterion baseline_3
uses).

No Lula anywhere: init IK, action-chain IK, close+lift all go through cuRobo
out-of-process subprocesses.

Usage
-----
First start the DP3 inference server (separate `dp3` env):
    /home/accelerator/miniforge3/envs/dp3/bin/python \\
        Baseline1/eval/dp3_inference_server.py \\
        --ckpt Baseline1/dp3_runs/b3_curobo_tuna_v1/checkpoints/<ckpt>.ckpt \\
        --port 8765

Then in env_isaaclab:
    /home/accelerator/miniforge3/envs/env_isaaclab/bin/python \\
        sim/eval_dp3_baseline3.py \\
        --episodes-glob 'Baseline1/data/episodes_b3_curobo/*ycb_dex_06*.hdf5' \\
        --n-rollouts 5 --headless \\
        --video replay_video_check/eval_b3_curobo_tuna \\
        --result-dir output/dp3_eval_b3curobo
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

parser = argparse.ArgumentParser(description="baseline_3 cuRobo policy DP3 eval")
parser.add_argument("--episodes-glob", type=str,
                    default=os.path.join(PROJ_ROOT,
                        "Baseline1/data/episodes_b3_curobo/*ycb_dex_06*.hdf5"),
                    help="glob of TRAINING episodes — eval uses each one's "
                         "(obj_quat_G, obj_origin_G, state[0]) as its scene")
parser.add_argument("--n-rollouts", type=int, default=5,
                    help="how many episodes from the glob to eval (random sample if "
                         "fewer than available; deterministic via --seed)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-chunks", type=int, default=5,
                    help="receding-horizon chunks. Each chunk = n_action_steps "
                         "(=8) policy actions. Default 5 → up to 40 waypoints.")
parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8765",
                    help="DP3 inference server")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--video", type=str, default=None,
                    help="if set, capture per-episode mp4s into this dir (only "
                         "successful grasps kept)")
parser.add_argument("--video-every", type=int, default=3,
                    help="capture 1 frame per N sim steps")
parser.add_argument("--video-all", action="store_true",
                    help="keep video for ALL episodes (default: only successful)")
parser.add_argument("--result-dir", type=str,
                    default=os.path.join(PROJ_ROOT, "output/dp3_eval_b3curobo"),
                    help="where to write the per-run JSON summary")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

# now IsaacSim is up, we can import everything else
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject
import grasp_physics

# ============================================================
# Constants — IDENTICAL to sim/run_grasp_sim_baseline3.py
# ============================================================
ROBOT_POSITION    = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION    = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE       = [2.0, 2.0, 0.1]
TABLE_TOP_Z       = 0.80
OBJECT_XY         = np.array([0.0, 0.55])
LIFT_HEIGHT       = 0.15
PREGRASP_BACKOFF  = 0.12

HOME_JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
IK_POS_TOL, IK_ORI_TOL, IK_SEEDS = 0.005, 0.05, 1024
CIK_SCRIPT  = os.path.join(SIM_DIR, "curobo_ik.py")
CPLAN_SCRIPT = os.path.join(SIM_DIR, "curobo_plan.py")
PLAN_POS_TOL, PLAN_ORI_TOL = 0.01, 0.10

# Eval-only
GRIP_ARRIVE_THR = 0.5
SUCCESS_DZ_M    = 0.03                                 # same as baseline_3
N_PC_POINTS     = 4096

_VID = {"on": False, "i": 0, "n": 0, "vp": None}
_VID_FRAMES = "/tmp/b3eval_video_frames"


# ============================================================
# Quat helpers (copied from run_grasp_sim_baseline3.py)
# ============================================================
def _xyzw(q_wxyz): return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
def _wxyz(q_xyzw): return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
_RZ_NEG90 = Rotation.from_euler("z", -90, degrees=True)
_RZ_POS90 = Rotation.from_euler("z", +90, degrees=True)


def retarget_to_franka_quat(q_wxyz):
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _RZ_NEG90).as_quat())


def franka_to_retarget_quat(q_wxyz):
    return _wxyz((Rotation.from_quat(_xyzw(q_wxyz)) * _RZ_POS90).as_quat())


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
# cuRobo IK (out-of-process) — copied + adapted
# ============================================================
def solve_ik_chain(waypoints, start_qpos=None):
    """waypoints: [(pos_world(3), quat_wxyz_franka(4)), ...]
    start_qpos: optional (7,) seed for warm-start IK chain DP — pass last executed
        qpos of previous chunk so frame-0 IK candidate is continuity-gated against
        it. Prevents elbow-flip jumps across chunks.
    Returns (qpos (N,7) NaN-rows-for-fail, ok (N,) bool), or (None,None) on subprocess err."""
    pos = np.array([w[0] for w in waypoints], dtype=np.float64)
    quat = np.array([w[1] for w in waypoints], dtype=np.float64)
    tag = f"/tmp/b3eval_cik_{os.getpid()}"
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
    """One-pose IK. Returns 7-D qpos or None."""
    qpos, ok = solve_ik_chain([(pos_world, quat_franka_wxyz)])
    if qpos is None or not bool(ok[0]):
        return None
    return qpos[0]


# ============================================================
# cuRobo plan_grasp (for close+lift)
# ============================================================
def _T_robot_world():
    R_rw = Rotation.from_euler("z", -ROBOT_ORIENTATION[2], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_rw
    T[:3, 3] = -R_rw @ np.array(ROBOT_POSITION, dtype=np.float64)
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
    tag = f"/tmp/b3eval_plan_{os.getpid()}"
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


# ============================================================
# Scene setup (copied/adapted from baseline_3 collector)
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
                color=np.array([0.08, 0.08, 0.10]))   # near-black floor (high contrast vs white Franka + tan table)
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
                    os.path.join(_VID_FRAMES, f"f_{_VID['i']:05d}.png"))
                _VID["i"] += 1
        world.step = _step_capture
        cprint(f"📹 per-episode video → {args.video}/", "magenta")

    cprint("✅ World + Franka ready", "green")
    return {"world": world, "franka": franka, "obj": None}


def load_object(world, usd_path, mass):
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    spawn = np.array([OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + 0.08])
    obj = RigidObject(world, usd_path=usd_path, pos=spawn,
                      ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=mass)
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
# Per-episode video helpers
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
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", "20",
         "-i", os.path.join(_VID_FRAMES, "f_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
        capture_output=True, text=True)
    if r.returncode == 0:
        cprint(f"  📹 {out}", "magenta")
    else:
        cprint(f"  📹 ffmpeg failed: {r.stderr[-200:]}", "yellow")
    shutil.rmtree(_VID_FRAMES, ignore_errors=True)


# ============================================================
# DP3 inference server HTTP client
# ============================================================
def get_policy_info(url):
    import requests
    return requests.get(f"{url}/info", timeout=5).json()


def query_policy(url, pc_obs, agent_obs):
    """pc_obs: (n_obs, N, 3) float32. agent_obs: (n_obs, 8) float32.
    Returns (n_action_steps, 8) float32 — EE poses in G-frame retarget conv."""
    import requests
    r = requests.post(f"{url}/predict", json={
        "point_cloud": pc_obs.tolist(),
        "agent_pos":   agent_obs.tolist(),
    }, timeout=30).json()
    return np.asarray(r["action"], dtype=np.float32)


# ============================================================
# Eval rollout (chunked receding horizon)
# ============================================================
def build_observation(scene, pc0_G, origin_world, gripper_state):
    """Read sim state. pc0_G is the static training PC (we don't re-sample — object
    is pinned to its training pose, PC stays exactly as recorded). EE pose is read
    LIVE from sim → G-frame, retarget convention."""
    ee_pos_w, ee_q_w = read_panda_hand_pose(scene["world"].stage)
    ee_pos_G = (ee_pos_w - origin_world).astype(np.float32)
    ee_q_G_retarget = franka_to_retarget_quat(ee_q_w).astype(np.float32)
    agent_pos = np.concatenate([ee_pos_G, ee_q_G_retarget,
                                [np.float32(gripper_state)]]).astype(np.float32)
    return pc0_G.astype(np.float32), agent_pos


def rollout_chunked(scene, server_url, info, pc0_G, origin_world,
                    obj_pos_w, obj_quat, max_chunks):
    """End-to-end DP3 rollout: chunked receding horizon + DP3-driven close+lift.

    Sequence per executed waypoint:
      - while gripper still OPEN:
          set_joint_positions (kinematic), object PINNED at retarget pose
      - first action with grip>=0.5: STOP pinning + close_gripper() + 80 settle steps,
          then read initial_z (object's z when released)
      - while gripper CLOSED:
          apply_action (PD), franka.close_gripper() reasserted each step, no pin
      - keep stepping through all remaining DP3-predicted waypoints (these are the
          lift segment per training schedule). NO cuRobo plan_grasp — DP3 drives.

    Returns dict with success, dz, n_chunks, grip_signal_idx, executed (for debug),
    or None on subprocess / server error.
    """
    from omni.isaac.core.utils.types import ArticulationAction
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]
    n_obs    = int(info["n_obs_steps"])
    n_action = int(info["n_action_steps"])

    def _pin():
        obj.rigid.set_world_pose(obj_pos_w, obj_quat)
        try:
            obj.rigid.set_linear_velocity(np.zeros(3))
            obj.rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    obj_centroid_w = np.asarray(obj_pos_w, dtype=np.float64)
    executed = []
    grip_signal_idx = None
    gripper_closed = False
    initial_z = None

    franka.open_gripper()
    for _ in range(5):
        world.step(render=True); _pin()

    obs0 = build_observation(scene, pc0_G, origin_world, gripper_state=0.0)
    obs_window = [obs0] * n_obs
    last_qpos = np.asarray(franka.get_joint_positions()[:7], dtype=np.float64)

    for chunk in range(max_chunks):
        pc_obs = np.stack([o[0] for o in obs_window])
        # Update obs's gripper-state channel to reflect current physics (0 open, 1 closed)
        cur_obs = build_observation(scene, pc0_G, origin_world,
                                    gripper_state=(1.0 if gripper_closed else 0.0))
        obs_window[-1] = cur_obs
        ap_obs = np.stack([o[1] for o in obs_window])

        try:
            action = query_policy(server_url, pc_obs, ap_obs)
        except Exception as e:
            cprint(f"  ❌ DP3 server error chunk {chunk}: {e}", "red")
            return None

        chunk_wps, chunk_grips = [], []
        for a in action:
            pos_w = (a[:3].astype(np.float64) + origin_world)
            q_franka = retarget_to_franka_quat(a[3:7].astype(np.float64))
            chunk_wps.append((pos_w, q_franka))
            chunk_grips.append(float(a[7]))

        qpos, ok = solve_ik_chain(chunk_wps, start_qpos=last_qpos)
        if qpos is None:
            cprint(f"  ❌ cuRobo IK chain failed for chunk {chunk}", "red")
            return None
        n_ok = int(ok.sum())
        first_ok = int(np.argmax(ok)) if n_ok > 0 else -1
        seed_jump = (np.abs(qpos[first_ok] - last_qpos).max()
                     if first_ok >= 0 else float("nan"))
        cprint(f"  [chunk {chunk}] IK {n_ok}/{n_action} reachable, grip "
               f"[{min(chunk_grips):.2f}, {max(chunk_grips):.2f}], "
               f"seed→f0 Δ={np.rad2deg(seed_jump):.0f}°, "
               f"closed={gripper_closed}", "cyan")

        for k in range(n_action):
            if not ok[k]:
                continue

            # First time grip crosses 0.5 → close gripper, release pin, settle
            if chunk_grips[k] >= GRIP_ARRIVE_THR and not gripper_closed:
                obj_init_pos, _ = obj.get_obj_pos()
                initial_z = float(obj_init_pos[2])
                grip_signal_idx = len(executed)
                ee_now, _ = read_panda_hand_pose(world.stage)
                cprint(f"  ◉ DP3 grip≥0.5 @ chunk {chunk} step {k}: "
                       f"close+lift via DP3. EE={np.round(ee_now,3).tolist()}, "
                       f"obj_init_z={initial_z:.3f}", "magenta")
                franka.close_gripper()
                for _ in range(80):
                    world.step(render=True)         # let gripper close (no pin)
                gripper_closed = True

            # Drive arm: kinematic+pin while open, PD after close
            if gripper_closed:
                franka.close_gripper()              # re-assert closed each step
                franka.apply_action(ArticulationAction(
                    joint_positions=np.concatenate([qpos[k], np.array([None, None])])))
                for _ in range(3):
                    world.step(render=True)
            else:
                grip_finger = franka.get_joint_positions()[7:9]
                franka.set_joint_positions(np.concatenate([qpos[k], grip_finger]))
                for _ in range(2):
                    world.step(render=True)
                _pin()

            executed.append((chunk_wps[k][0].copy(), chunk_wps[k][1].copy()))
            last_qpos = qpos[k].copy()

        # roll obs window (gripper-state channel updated next iter)
        new_obs = build_observation(scene, pc0_G, origin_world,
                                    gripper_state=(1.0 if gripper_closed else 0.0))
        obs_window = obs_window[1:] + [new_obs]

    # End of all chunks — settle
    for _ in range(80):
        world.step(render=True)

    # Measure dz
    if initial_z is None:
        # Policy never emitted grip≥0.5 across max_chunks; no close fired → fail with dz=0
        cprint(f"  ⚠️ policy never emitted grip≥0.5 across {max_chunks} chunks "
               f"({len(executed)} wp) — fail by default", "yellow")
        return {"success": False, "dz": 0.0, "n_chunks": max_chunks,
                "grip_signal_idx": None,
                "n_executed": len(executed), "stage": "no_grip_signal"}

    obj_after, _ = obj.get_obj_pos()
    dz = float(obj_after[2]) - initial_z
    success = dz > SUCCESS_DZ_M
    cprint(f"  object Z Δ = {dz*100:+.1f}cm → "
           f"{'GRASPED + LIFTED' if success else 'not lifted'}",
           "green" if success else "red")

    # Diagnostic: distance from executed waypoint nearest to object centroid
    dists = np.array([np.linalg.norm(p - obj_centroid_w) for p, _ in executed]) \
            if executed else np.array([np.nan])
    return {"success": bool(success), "dz": dz,
            "n_chunks": chunk + 1,
            "grip_signal_idx": grip_signal_idx,
            "n_executed": len(executed),
            "min_dist_to_obj_cm": float(dists.min()) * 100,
            "stage": "dp3_close_lift"}


# ============================================================
# Close + lift via cuRobo plan_grasp (drive grasp + lift segments only)
# ============================================================
def close_and_lift(scene, grasp_pos_w, grasp_quat_w, obj_pos_w, obj_quat):
    """Returns (success, dz_m, status_str)."""
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
    plan = solve_plan_grasp(scene, start_qpos7, grasp_pos_w, grasp_quat_w,
                            obj_pos_w, obj_quat)
    if plan is None or not plan["success"]:
        st = (plan or {}).get("status", "no plan")[:160]
        cprint(f"  ❌ close+lift plan_grasp failed: {st}", "red")
        return False, 0.0, f"plan_grasp_failed: {st}"

    g_q, l_q = plan["grasp_qpos"], plan["lift_qpos"]
    cprint(f"  ✅ close+lift plan: grasp={len(g_q)} lift={len(l_q)} wp "
           f"(plan {plan['plan_seconds']:.2f}s)", "cyan")

    # Drive grasp segment (final straight-in along tool z, fingers free)
    for qpos7 in g_q:
        grip = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([qpos7, grip]))
        for _ in range(2):
            world.step(render=True)
        _pin()

    # Settle, then release the pin and read baseline z
    for _ in range(15):
        world.step(render=True)
    obj_init, _ = obj.get_obj_pos()
    initial_z = float(obj_init[2])

    # Close gripper
    franka.close_gripper()
    for _ in range(80):
        world.step(render=True)

    # Drive lift segment via PD
    franka.close_gripper()
    for qpos7 in l_q:
        franka.apply_action(ArticulationAction(
            joint_positions=np.concatenate([qpos7, np.array([None, None])])))
        for _ in range(3):
            world.step(render=True)
    for _ in range(80):
        world.step(render=True)

    obj_after, _ = obj.get_obj_pos()
    dz = float(obj_after[2]) - initial_z
    success = dz > SUCCESS_DZ_M
    cprint(f"  object Z Δ = {dz*100:+.1f}cm  →  "
           f"{'GRASPED + LIFTED' if success else 'not lifted'}",
           "green" if success else "red")
    return success, dz, "ok"


# ============================================================
# Per-episode eval (one rollout)
# ============================================================
def eval_one_episode(scene, ep_path, info):
    """Open the episode hdf5; place obj; init Franka; rollout; close+lift; return result."""
    name = os.path.basename(ep_path)
    with h5py.File(ep_path, "r") as h:
        obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64)
        obj_quat_G   = np.array(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)
        cid          = int(h.attrs["ycb_class_id"])
        state0       = np.array(h["state"][0], dtype=np.float64)
        pc0          = np.array(h["point_cloud"][0], dtype=np.float32)         # (N, 3) G-frame

    # ── Place object at FIXED OBJECT_XY + training orientation ─────────────
    sim_origin_W = np.array([OBJECT_XY[0] - obj_origin_G[0],
                             OBJECT_XY[1] - obj_origin_G[1], TABLE_TOP_Z])
    obj_pos_w    = obj_origin_G + sim_origin_W
    origin_world = sim_origin_W                                                # for G↔world

    obj = scene["obj"]
    scene["franka"].set_joint_positions(HOME_JOINTS)
    for _ in range(40):                                                       # pin while initialising
        obj.rigid.set_world_pose(obj_pos_w, obj_quat_G)
        try:
            obj.rigid.set_linear_velocity(np.zeros(3))
            obj.rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        scene["world"].step(render=True)

    # ── Init Franka at the episode's state[0] via cuRobo single-pose IK ────
    ee0_pos_w = state0[:3] + origin_world
    ee0_q_franka = retarget_to_franka_quat(state0[3:7])
    qpos0 = solve_single_ik(ee0_pos_w, ee0_q_franka)
    if qpos0 is None:
        cprint(f"  ❌ init IK failed for state[0] of {name}", "red")
        return {"name": name, "success": False, "dz": 0.0, "stage": "init_ik_fail"}
    scene["franka"].set_joint_positions(np.concatenate([qpos0, [0.04, 0.04]]))
    for _ in range(15):
        obj.rigid.set_world_pose(obj_pos_w, obj_quat_G)
        scene["world"].step(render=True)
    ee_after, _ = read_panda_hand_pose(scene["world"].stage)
    init_err = float(np.linalg.norm(ee_after - ee0_pos_w))
    cprint(f"  init Franka → state[0]: |EE err|={init_err*1000:.1f}mm", "cyan")

    # ── End-to-end DP3 rollout (chunked receding horizon + DP3 close+lift) ──
    rollout = rollout_chunked(scene, args.server_url, info, pc0, origin_world,
                              obj_pos_w, obj_quat_G, args.max_chunks)
    if rollout is None:
        return {"name": name, "success": False, "dz": 0.0, "stage": "rollout_fail"}
    rollout["name"] = name
    rollout["init_ee_err_mm"] = init_err * 1000
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

    # Sample n-rollouts episodes (with replacement if asking for more than available)
    n = min(args.n_rollouts, len(eps))
    if args.n_rollouts > len(eps):
        idx = list(range(len(eps))) + list(rng.integers(0, len(eps),
                                                        size=args.n_rollouts - len(eps)))
    else:
        idx = rng.choice(len(eps), size=n, replace=False)
    chosen = [eps[int(i)] for i in idx]

    cprint("=" * 64, "cyan")
    cprint(f"DP3 baseline_3 eval — {len(chosen)} rollouts", "cyan")
    cprint(f"  server: {args.server_url}", "cyan")
    info = get_policy_info(args.server_url)
    cprint(f"  policy: horizon={info['horizon']} n_obs={info['n_obs_steps']} "
           f"n_action={info['n_action_steps']}", "cyan")
    cprint(f"  scene : OBJECT_XY={OBJECT_XY.tolist()} TABLE_TOP_Z={TABLE_TOP_Z}", "cyan")
    cprint("=" * 64, "cyan")

    scene = setup_world_b3_eval()

    # Load USD for the target object (use the first episode's class id; assume single class)
    with h5py.File(chosen[0], "r") as h:
        cid = int(h.attrs["ycb_class_id"])
    usd = os.path.join(PROJ_ROOT, "output/obj_usd_cad/ycb", f"ycb_dex_{cid:02d}.usd")
    mass = grasp_physics.object_mass_kg(cid)
    cprint(f"\n=== object ycb_dex_{cid:02d}  mass={mass}kg ===", "cyan")
    if not os.path.exists(usd):
        cprint(f"  ❌ USD missing: {usd}", "red")
        simulation_app.close(); return
    scene["obj"] = load_object(scene["world"], usd, mass)
    scene["obj_mesh"] = getattr(scene["obj"], "_curobo_mesh", None)

    results = []
    for i, ep in enumerate(chosen):
        cprint(f"\n[{i+1}/{len(chosen)}] {os.path.basename(ep)}", "yellow")
        video_begin()
        try:
            r = eval_one_episode(scene, ep, info)
        except Exception as e:
            import traceback
            cprint(f"  ❌ episode crashed: {type(e).__name__}: {e}", "red")
            cprint(traceback.format_exc()[-600:], "red")
            r = {"name": os.path.basename(ep), "success": False, "dz": 0.0,
                 "stage": f"crash_{type(e).__name__}"}
        results.append(r)
        video_end(scene["world"], r["name"].replace(".hdf5", ".mp4"),
                  keep=(r["success"] or args.video_all))

    # ── Summary ────────────────────────────────────────────────────────────
    n_succ = sum(int(r["success"]) for r in results)
    cprint("\n" + "=" * 64, "cyan")
    cprint(f"  EVAL SUMMARY  ycb_dex_{cid:02d}", "cyan")
    cprint("=" * 64, "cyan")
    for r in results:
        tag = "✓" if r["success"] else "✗"
        cprint(f"  {tag} dz={r['dz']*100:+.1f}cm  stage={r.get('stage','')}  "
               f"min_dist={r.get('min_dist_to_obj_cm','?')}cm  {r['name']}",
               "green" if r["success"] else "red")
    cprint("-" * 64, "cyan")
    cprint(f"  TOTAL: {n_succ}/{len(results)} grasped "
           f"({100*n_succ/max(len(results),1):.0f}%)",
           "green" if n_succ > 0 else "red")
    cprint("=" * 64, "cyan")

    out_json = os.path.join(args.result_dir, f"eval_{int(time.time())}.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "policy_info": info,
                   "n_success": n_succ, "n_total": len(results),
                   "results": results}, f, indent=2)
    cprint(f"wrote → {out_json}", "cyan")

    simulation_app.close()


main()
