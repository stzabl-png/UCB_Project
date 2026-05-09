#!/usr/bin/env python3
"""
Baseline2 — Robot DP (Sim) Data Collection
==========================================
For each object:
  1. Generate N random grasp candidates (no human prior)
  2. Isaac Sim + cuRobo: execute each candidate
  3. Record per-step: EE state(8D) + action(8D) + point_cloud(4096,3)
  4. Check success (object lifted > 3cm)
  5. Save successful episodes to HDF5

DP3 format per episode:
    point_cloud: (T, 4096, 3)  object pts in robot base frame
    state:       (T, 8)        [x,y,z, qw,qx,qy,qz, gripper]
    action:      (T, 8)        state shifted by 1 (next EE pose)

Usage:
    sim45 Baseline2/collect_sim_trajectories.py --obj_id mug --n_candidates 50
    sim45 Baseline2/collect_sim_trajectories.py --all_objects --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys, json

parser = argparse.ArgumentParser()
parser.add_argument("--obj_id",       type=str,   default=None)
parser.add_argument("--all_objects",  action="store_true")
parser.add_argument("--n_candidates", type=int,   default=20,
                    help="Max candidates per object (default 20, consistent with main pipeline)")
parser.add_argument("--n_points",     type=int,   default=4096)
parser.add_argument("--headless",     action="store_true")
parser.add_argument("--output_dir",   type=str,   default="Baseline2/data/episodes")
parser.add_argument("--object_scale", type=float, default=1.0)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import h5py
import torch
import trimesh
from scipy.spatial.transform import Rotation
from termcolor import cprint
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "sim"))
from env_config.robot.Franka import Franka
from env_config.room.Real_Ground import Real_Ground
from env_config.rigid.RigidObject import RigidObject

# Load project config directly by path to avoid collision with Isaac Sim's cv2/config.py.
# Also inject into sys.modules['config'] so transitive imports (e.g. random_grasp_sampler)
# that do `import config` also get the project config instead of cv2/config.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("config", os.path.join(PROJ, "config.py"))
config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(config)
sys.modules["config"] = config   # ← prevents cv2/config.py collision in all sub-imports

# ── Scene constants ────────────────────────────────────────
ROBOT_POS  = [0.2, -0.05, 0.8]
ROBOT_ORI  = [0.0, 0.0, 90.0]
TABLE_POS  = [0.0, 1.0, 0.75]
TABLE_ORI  = [0.0, 0.0, 0.0]
TABLE_SCL  = [2.0, 2.0, 0.1]
TABLE_TOP  = 0.80
OBJ_POS    = [0.0, 0.55, TABLE_TOP]
OBJ_ORI    = [0.0, 0.0, 0.0]
LIFT_H     = 0.15
_MG        = None   # cuRobo singleton

# ── Coordinate helpers ─────────────────────────────────────
def robot_transforms():
    yaw = np.deg2rad(ROBOT_ORI[2])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    T[:3, 3]  = ROBOT_POS
    return T, np.linalg.inv(T)

def world_to_robot(pos_w, quat_wxyz_w):
    _, Trw = robot_transforms()
    pos_r = (Trw @ np.append(pos_w, 1.0))[:3]
    Rw = Rotation.from_quat([quat_wxyz_w[1],quat_wxyz_w[2],
                              quat_wxyz_w[3],quat_wxyz_w[0]])
    Rrw = Rotation.from_matrix(Trw[:3,:3])
    q   = (Rrw * Rw).as_quat()
    return pos_r, np.array([q[3],q[0],q[1],q[2]])

# ── Mesh & point cloud ─────────────────────────────────────
SAM3D_MESH_ROOT = os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "meshes")
SAM3D_DATASETS  = ["egodex", "oakink", "ycb"]   # search order


def load_mesh(obj_id, scale=1.0):
    """Load mesh for obj_id.

    Primary (SAM3DMesh): data_hub/meshes/SAM3DMesh/meshes/{dataset}/{obj_id}/mesh.ply
    Fallback (legacy):   data_hub/meshes/v1/{obj_id}.obj  etc.
    """
    # ── 1. SAM3DMesh (primary) ────────────────────────────────
    for ds in SAM3D_DATASETS:
        p = os.path.join(SAM3D_MESH_ROOT, ds, obj_id, "mesh.ply")
        if os.path.exists(p):
            cprint(f"  Mesh found (SAM3D/{ds}): {p}", "cyan")
            m = trimesh.load(p, force="mesh")
            m.apply_scale(scale)
            return m

    # ── 2. Legacy fallback ────────────────────────────────────
    legacy_dirs = [
        os.path.join(PROJ, "data_hub", "meshes", "v1"),
        os.path.join(PROJ, "sim", "assets"),
        os.path.join(PROJ, "assets", "usd"),
        os.path.join(PROJ, "data_hub", "meshes", "contactpose"),
        os.path.join(PROJ, "data_hub", "meshes", "grab"),
    ]
    for d in legacy_dirs:
        for ext in [".obj", ".ply", ".stl"]:
            for p in [os.path.join(d, f"{obj_id}{ext}"),
                      os.path.join(d, obj_id, f"{obj_id}{ext}")]:
                if os.path.exists(p):
                    cprint(f"  Mesh found (legacy): {p}", "cyan")
                    m = trimesh.load(p, force="mesh")
                    sj = os.path.join(os.path.dirname(p), "scale.json")
                    if os.path.exists(sj):
                        sc = json.load(open(sj)).get(obj_id, 1.0)
                        m.apply_scale(sc)
                        cprint(f"  Applied scale.json scale: {sc}", "cyan")
                    m.apply_scale(scale)
                    return m

    cprint(f"  ❌ Mesh not found for '{obj_id}'.", "red")
    cprint(f"     SAM3D searched: {[os.path.join(SAM3D_MESH_ROOT, ds, obj_id) for ds in SAM3D_DATASETS]}", "red")
    return None

def sample_pc(mesh, obj_pos_w, obj_quat_wxyz, n=4096):
    """Sample mesh surface → transform to robot base frame."""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    R = Rotation.from_quat([obj_quat_wxyz[1],obj_quat_wxyz[2],
                             obj_quat_wxyz[3],obj_quat_wxyz[0]]).as_matrix()
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=obj_pos_w
    ones = np.ones((n,1))
    pts_w = (T @ np.c_[pts,ones].T).T[:,:3]
    _, Trw = robot_transforms()
    pts_r = (Trw @ np.c_[pts_w,ones].T).T[:,:3]
    return pts_r.astype(np.float32)

# ── EE state ───────────────────────────────────────────────
def get_ee_state(franka):
    """[x,y,z, qw,qx,qy,qz, gripper(0-1)] in robot base frame."""
    pos_w, quat_w = franka.get_world_pose()
    pos_r, quat_r = world_to_robot(pos_w, quat_w)
    fingers = franka.get_joint_positions()[7:9]
    gripper = float(np.clip(1.0 - np.mean(fingers)/0.04, 0, 1))
    return np.array([*pos_r, *quat_r, gripper], dtype=np.float32)

# ── Scene setup (split into two parts to avoid Franka crash) ─
def setup_world_once(headless=True):
    """Create World + robot + table exactly once. Returns (world, franka)."""
    world = World(backend="numpy")
    ph = world.get_physics_context()
    ph.enable_ccd(True); ph.enable_gpu_dynamics(True)
    ph.set_broadphase_type("gpu"); ph.enable_stablization(True)
    ph.set_solver_type("TGS")
    set_camera_view(eye=[0,4.5,3.5], target=[0,0,0],
                    camera_prim_path="/OmniverseKit_Persp")
    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0,0,0], light_type="dome")
    Real_Ground(world.scene, visual_material_usd=None)
    delete_prim("/World/Table")
    FixedCuboid(prim_path="/World/Table", name="table",
                position=TABLE_POS,
                orientation=euler_angles_to_quat(np.array(TABLE_ORI),degrees=True),
                scale=TABLE_SCL, size=1.0, visible=True)
    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POS), np.array(ROBOT_ORI))
    world.reset()
    for _ in range(50): world.step(render=True)
    franka.open_gripper()
    for _ in range(10): world.step(render=True)
    cprint("✅ World + Franka ready", "green")
    return world, franka


def load_object(world, obj_id, scale=1.0):
    """Swap in a new rigid object (USD) without touching the robot."""
    usd = None
    for d in [os.path.join(PROJ,"output","assets"), os.path.join(PROJ,"sim","assets")]:
        p = os.path.join(d, f"{obj_id}.usd")
        if os.path.exists(p): usd=p; break
    if usd is None:
        cprint(f"❌ USD not found: {obj_id}", "red")
        return None

    # Remove any previous object prims
    for i in range(10): delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")

    obj_z = 0.075 * scale
    opos  = list(OBJ_POS); opos[2] += obj_z
    obj = RigidObject(world, usd_path=usd, pos=np.array(opos),
                      ori=np.array(OBJ_ORI), scale=np.array([scale]*3), mass=0.05)
    for _ in range(100): world.step(render=True)
    cprint(f"✅ Object loaded: {obj_id}", "green")
    return obj

# ── cuRobo ─────────────────────────────────────────────────
def init_mg():
    global _MG
    if _MG: return _MG
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    _, Trw = robot_transforms()
    tp = (Trw @ np.append(TABLE_POS,1.0))[:3]
    gp = (Trw @ np.array([0,0,-0.005,1.0]))[:3]
    wc = {"cuboid": {
        "table":  {"dims": TABLE_SCL, "pose": [*tp.tolist(),1,0,0,0]},
        "ground": {"dims": [5,5,0.01], "pose": [*gp.tolist(),1,0,0,0]},
    }}
    mg = MotionGen(MotionGenConfig.load_from_robot_config("franka.yml", wc, interpolation_dt=0.02))
    mg.warmup()
    _MG = mg
    cprint("✅ cuRobo ready", "green")
    return mg

def plan(mg, franka, pos_w, quat_wxyz_w):
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CJS
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    pos_r, quat_r = world_to_robot(pos_w, quat_wxyz_w)
    j = franka.get_joint_positions()[:7]
    st = CJS.from_position(torch.tensor(j,dtype=torch.float32).unsqueeze(0).cuda(),
                           joint_names=[f"panda_joint{i}" for i in range(1,8)])
    goal = Pose.from_list([*pos_r.tolist(), *quat_r.tolist()])
    res = mg.plan_single(st, goal, MotionGenPlanConfig(max_attempts=10, enable_graph=True))
    if res.success.item():
        return res.get_interpolated_plan().position.cpu().numpy()
    return None

# ── Execute & record ───────────────────────────────────────
def execute_and_record(scene, grasp_pt_obj, grasp_rot_obj, mesh, n_pts, scale):
    """Execute one grasp, record trajectory. Returns episode dict or None."""
    mg = init_mg()
    franka, world, obj = scene["franka"], scene["world"], scene["obj"]
    obj_pos, obj_quat = obj.get_obj_pos()

    # grasp pose: obj frame → world frame
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat([obj_quat[1],obj_quat[2],
                                   obj_quat[3],obj_quat[0]]).as_matrix()
    T[:3,3]  = obj_pos
    pos_w  = (T @ np.append(grasp_pt_obj*scale,1.0))[:3]
    rot_w  = T[:3,:3] @ grasp_rot_obj
    # coord adapt (same as run_grasp_sim.py)
    rot_w  = rot_w @ np.array([[0,1,0],[-1,0,0],[0,0,1]],dtype=np.float64)
    q_xyzw = Rotation.from_matrix(rot_w).as_quat()
    quat_w = np.array([q_xyzw[3],q_xyzw[0],q_xyzw[1],q_xyzw[2]])

    approach = rot_w[:,2]
    pre_pos  = pos_w - approach*0.15

    franka.open_gripper()
    for _ in range(20): world.step(render=True)

    states, pcs = [], []

    def record():
        states.append(get_ee_state(franka))
        pcs.append(sample_pc(mesh, obj_pos, obj_quat, n_pts))

    # Phase 1: pre-grasp
    traj = plan(mg, franka, pre_pos, quat_w) or plan(mg, franka, pos_w, quat_w)
    if traj is None:
        cprint("      cuRobo: pre-grasp plan FAILED", "red")
        return None
    for jp in traj:
        g = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([jp,g]))
        world.step(render=True)
        record()
    for _ in range(5): world.step(render=True)

    # Phase 2: final approach
    traj2 = plan(mg, franka, pos_w, quat_w)
    if traj2 is None:
        cprint("      cuRobo: final approach plan FAILED", "red")
        return None
    for jp in traj2:
        g = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([jp,g]))
        for _ in range(3): world.step(render=True)
        record()

    # Phase 3: close gripper
    franka.close_gripper()
    for i in range(60):
        world.step(render=True)
        if i % 10 == 0: record()

    # Phase 4: lift
    lift_pos = pos_w.copy(); lift_pos[2] += LIFT_H
    traj3 = plan(mg, franka, lift_pos, quat_w)
    if traj3 is not None:
        from omni.isaac.core.utils.types import ArticulationAction
        franka.close_gripper()
        for jp in traj3:
            franka.apply_action(ArticulationAction(
                joint_positions=np.concatenate([jp, np.array([None,None])])))
            for _ in range(2): world.step(render=True)
            record()

    for _ in range(60): world.step(render=True)
    record()

    # Success check
    pos_after, _ = obj.get_obj_pos()
    z_delta = pos_after[2] - obj_pos[2]
    cprint(f"      z_delta={z_delta*100:.1f}cm (need >3cm)", "cyan")
    if z_delta <= 0.03:
        cprint("      Grasp FAILED (object not lifted)", "red")
        return None

    s = np.array(states, dtype=np.float32)  # (T,8)
    p = np.array(pcs, dtype=np.float32)     # (T,N,3)
    if len(s) < 2: return None
    return {"point_cloud": p[:-1], "state": s[:-1], "action": s[1:]}

# ── Main ───────────────────────────────────────────────────
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    if args.obj_id:
        objs = [args.obj_id]
    else:
        # Enumerate all objects from SAM3DMesh directory (egodex + oakink + ycb)
        objs = []
        for ds in SAM3D_DATASETS:
            ds_dir = os.path.join(SAM3D_MESH_ROOT, ds)
            if os.path.isdir(ds_dir):
                for obj_name in sorted(os.listdir(ds_dir)):
                    if os.path.exists(os.path.join(ds_dir, obj_name, "mesh.ply")):
                        objs.append(obj_name)
        cprint(f"Found {len(objs)} objects in SAM3DMesh (egodex/oakink/ycb)", "cyan")

    sys.path.insert(0, os.path.join(PROJ, "tools"))
    from random_grasp_sampler import generate_candidates_iterative

    # ── Phase 1: pre-filter (mesh + candidates, no Isaac Sim needed) ──
    valid_objs = []   # [(obj_id, mesh, cands)]
    for obj_id in objs:
        cprint(f"\n{'='*55}\n  {obj_id} [pre-filter]\n{'='*55}", "cyan")
        mesh = load_mesh(obj_id, args.object_scale)
        if mesh is None:
            cprint(f"  ⚠️  Skip: mesh not found", "yellow"); continue
        cprint(f"  Mesh: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}", "green")

        # top 20 candidates — consistent with main pipeline (TARGET_HIGH_QUALITY=20)
        cands = generate_candidates_iterative(mesh, obj_id, hp_dir=None)[:args.n_candidates]
        if not cands:
            cprint(f"  ⚠️  Skip: no graspable candidates (object too large?)", "yellow"); continue

        # Check USD exists before bothering with Isaac Sim
        usd = None
        for d in [os.path.join(PROJ,"output","assets"), os.path.join(PROJ,"sim","assets")]:
            p = os.path.join(d, f"{obj_id}.usd")
            if os.path.exists(p): usd=p; break
        if usd is None:
            cprint(f"  ⚠️  Skip: USD not found (run: sim45 sim/convert_batch_usd.py --sam3d-only)", "yellow")
            continue

        valid_objs.append((obj_id, mesh, cands))
        cprint(f"  ✅ {len(cands)} candidates, USD ready", "green")

    if not valid_objs:
        cprint("\n❌ No valid objects found. Exiting.", "red")
        simulation_app.close()
        return

    cprint(f"\n✅ {len(valid_objs)} objects ready for Isaac Sim", "cyan")

    # ── Phase 2: Isaac Sim — init ONCE, swap objects ──────────
    world, franka = setup_world_once(headless=args.headless)

    total = 0
    for obj_id, mesh, cands in valid_objs:
        cprint(f"\n{'='*55}\n  {obj_id}\n{'='*55}", "cyan")

        obj = load_object(world, obj_id, args.object_scale)
        if obj is None:
            cprint(f"  ⚠️  Skipping {obj_id}: object load failed", "yellow")
            continue

        scene = {"world": world, "franka": franka, "obj": obj}
        obj_total = 0

        for i, c in enumerate(cands):
            cprint(f"  [{i+1}/{len(cands)}]", "yellow", end=" ")
            try:
                ep = execute_and_record(scene, c["grasp_point"], c["rotation"],
                                        mesh, args.n_points, args.object_scale)
            except Exception as e:
                import traceback
                cprint(f"error: {e}", "red")
                traceback.print_exc()
                ep = None

            if ep:
                path = os.path.join(args.output_dir, f"{obj_id}_ep{total:04d}.hdf5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("point_cloud", data=ep["point_cloud"])
                    f.create_dataset("state",       data=ep["state"])
                    f.create_dataset("action",      data=ep["action"])
                    f.attrs["obj_id"] = obj_id
                total += 1; obj_total += 1
                cprint(f"✅ saved ({len(ep['state'])} steps)", "green")
            else:
                cprint("failed", "red")

            # Reset object to initial position for next candidate
            obj_init = list(OBJ_POS)
            obj_init[2] += 0.075 * args.object_scale
            scene["obj"].set_obj_pose(np.array(obj_init), np.array(OBJ_ORI))
            for _ in range(30): world.step(render=True)
            franka.open_gripper()
            for _ in range(10): world.step(render=True)

        cprint(f"  {obj_id}: {obj_total} episodes saved", "cyan")

    cprint(f"\nDone. Total: {total} episodes → {args.output_dir}", "green")
    simulation_app.close()

if __name__ == "__main__":
    main()

