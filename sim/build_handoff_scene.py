#!/usr/bin/env python3
"""Build the IsaacSim scene PURELY from dp3_aug_handoff/sugar_scene_config.json
(+ the object USD it references) and render screenshots — a self-contained check
that the handoff JSON reproduces the scene. Run with env_isaaclab python."""
import sys, os, json, numpy as np
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.kit.viewport.utility as vu
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from pxr import Usd, UsdGeom, UsdPhysics

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject

PROJ    = os.path.dirname(SIM_DIR)
HANDOFF = os.path.join(PROJ, "dp3_aug_handoff")
OUT     = os.path.join(PROJ, "replay_video_check")
os.makedirs(OUT, exist_ok=True)

cfg    = json.load(open(os.path.join(HANDOFF, "sugar_scene_config.json")))
tds    = cfg["training_data_scene"]
table  = cfg["sim_scaffolding"]["table"]
ground = cfg["sim_scaffolding"]["ground_plane"]
print(f"[handoff scene] loaded sugar_scene_config.json — object={cfg['session']['object']}")

# ── world ────────────────────────────────────────────────────────────────────
world = World(backend="numpy")
world.get_physics_context().set_solver_type("TGS")
delete_prim("/Replicator/DomeLight_Xform")
rep.create.light(position=[0, 0, 0], light_type="dome")

# ── ground + table  (sim_scaffolding) ────────────────────────────────────────
GroundPlane(prim_path="/World/defaultGroundPlane",
            z_position=ground["z_position"], visual_material=None)
delete_prim("/World/Table")
FixedCuboid(prim_path="/World/Table", name="table",
            position=np.array(table["position"], dtype=float),
            orientation=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
            scale=np.array(table["scale"], dtype=float), size=float(table["size"]),
            visible=True)
print(f"[scene] table top z={table['top_z']}  ground z={ground['z_position']}")

# ── Franka at base pose  (training_data_scene.franka) ────────────────────────
fr        = tds["franka"]
ROBOT_POS = np.array(fr["base_position_W"], dtype=float)
ROBOT_ORI = np.array(fr["base_orientation_euler_deg"], dtype=float)
franka = Franka(world, ROBOT_POS, ROBOT_ORI)
world.reset()
for _ in range(30): world.step(render=True)
franka.open_gripper()
print(f"[scene] Franka base={ROBOT_POS.tolist()} yaw={ROBOT_ORI[2]:.1f}deg")

# ── pose the arm at state[0] via Lula IK  (training_data_scene.arm_initial_pose) ──
ik = KinematicsSolver(franka, end_effector_frame_name="panda_hand")
ik._kinematics.set_robot_base_pose(
    ROBOT_POS.astype(np.float64),
    np.asarray(euler_angles_to_quat(ROBOT_ORI, degrees=True), dtype=np.float64))
aip      = tds["arm_initial_pose"]
ee_pos_W = np.array(aip["ee_position_W"], dtype=np.float64)
ee_quat  = np.array(aip["ee_quat_wxyz_panda_hand"], dtype=np.float64)
_act, _ok = ik.compute_inverse_kinematics(
    target_position=ee_pos_W, target_orientation=ee_quat,
    position_tolerance=0.015, orientation_tolerance=0.1)
gripper_q = franka.get_joint_positions()[7:].copy()
if _ok:
    arm_q  = np.asarray(_act.joint_positions[:7], dtype=np.float64)
    full_q = np.concatenate([arm_q, gripper_q])
    print(f"[scene] IK OK — arm posed at state[0] EE={ee_pos_W.tolist()}")
else:
    full_q = franka.get_joint_positions().copy()
    print("[scene] IK FAILED — arm left at home pose")

def hold_franka():
    """Re-assert the arm pose (kinematic) so it doesn't sag between renders."""
    franka.set_joint_positions(full_q)
    franka.set_joint_velocities(np.zeros(9))
    franka._articulation_controller.apply_action(
        ArticulationAction(joint_positions=full_q))

# ── object — placed EXACTLY like gt_replay_ikpd_v2.py (RigidObject + set_world_pose
#    + kinematic freeze). The CAD USD is Y-up; this path is the one the eval pipeline
#    actually uses, so it reproduces dp3_twophase_sugar.mp4 faithfully. ───────────
obj_c   = tds["object"]
obj_usd = os.path.join(HANDOFF, obj_c["usd"])
obj_pos = np.array(obj_c["position_W"], dtype=np.float64)
obj_q   = np.array(obj_c["orientation_quat_wxyz"], dtype=np.float64)
for i in range(10): delete_prim(f"/World/Rigid/rigid_{i}")
delete_prim("/World/Rigid/rigid")
obj = RigidObject(world, usd_path=obj_usd, pos=np.array(obj_pos),
                  ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=0.1)
obj.rigid.set_world_pose(np.asarray(obj_pos, dtype=np.float64), obj_q)
obj_prim = world.stage.GetPrimAtPath(obj.rigid_prim_path)
for _ in range(5): world.step(render=True)
obj.rigid.set_world_pose(np.asarray(obj_pos, dtype=np.float64), obj_q)
# freeze: kinematic + collision-off (gt_replay's visual-reference setup)
_rb = UsdPhysics.RigidBodyAPI.Get(world.stage, obj_prim.GetPath())
if _rb:
    _ke = _rb.GetKinematicEnabledAttr()
    (_ke if _ke else _rb.CreateKinematicEnabledAttr()).Set(True)
for _p in Usd.PrimRange(obj_prim):
    if _p.IsA(UsdGeom.Mesh):
        _ca = UsdPhysics.CollisionAPI.Get(world.stage, _p.GetPath())
        if _ca:
            _ce = _ca.GetCollisionEnabledAttr()
            (_ce if _ce else _ca.CreateCollisionEnabledAttr()).Set(False)
obj.rigid.set_world_pose(np.asarray(obj_pos, dtype=np.float64), obj_q)
hold_franka()
for _ in range(50): world.step(render=True)
_rp, _rq = obj.get_obj_pos()
print(f"[scene] object readback pos={np.round(_rp,4).tolist()} quat={np.round(_rq,4).tolist()}")
print(f"        wanted  quat={np.round(obj_q,4).tolist()}   usd={obj_c['usd']}")

# ── render screenshots from a few angles ─────────────────────────────────────
viewport = vu.get_active_viewport()
VIEWS = [
    ("view1_overview",     [1.5,  1.5,  1.50], [0.00, 0.40, 0.85]),  # gt_replay 3/4 camera
    ("view2_franka_side",  [2.5, -0.7,  1.60], [0.18, 0.22, 0.90]),
    ("view3_top",          [0.45, 0.30, 2.70], [0.30, 0.15, 0.80]),
    ("view4_front",        [0.25,-1.55, 1.25], [0.20, 0.20, 0.92]),
    ("view5_object_close", [0.40, 0.80, 1.15], [0.00, 0.30, 0.86]),
]
saved = []
for name, eye, tgt in VIEWS:
    set_camera_view(eye=eye, target=tgt, camera_prim_path="/OmniverseKit_Persp")
    hold_franka()
    for _ in range(25): world.step(render=True)
    path = os.path.join(OUT, f"handoff_scene_{name}.png")
    vu.capture_viewport_to_file(viewport, path)
    for _ in range(20): world.step(render=True)
    saved.append(path)
    print(f"[render] saved {path}")

print("\n=== handoff scene built from sugar_scene_config.json ===")
for p in saved:
    print("  ", p, "exists" if os.path.exists(p) else "MISSING")
sim_app.close()
