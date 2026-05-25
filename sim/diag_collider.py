#!/usr/bin/env python3
"""Diagnose --grasp-collision: place sugar/tomato exactly as gt_replay does
(dynamic rigid body + convexHull collider, at obj_quat_G), render the PLACED pose,
let physics settle, render the SETTLED pose. If a flat/stable object reorients on
its own under gravity, its convexHull COLLIDER does not match the VISUAL mesh.
Readback prints confirm the object actually starts at obj_quat_G."""
import sys, os, numpy as np, h5py
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.kit.viewport.utility as vu
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdGeom, UsdPhysics

SIM_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SIM_DIR)
from env_config.rigid.RigidObject import RigidObject

PROJ = os.path.dirname(SIM_DIR)
TT = 0.80
SIM_ORIGIN_W = np.array([0.0, 0.3, TT])
OUT = os.path.join(PROJ, "replay_video_check")

world = World(backend="numpy"); world.get_physics_context().set_solver_type("TGS")
delete_prim("/Replicator/DomeLight_Xform"); rep.create.light(position=[0, 0, 0], light_type="dome")
GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0.0,
            color=np.array([0.08, 0.08, 0.10]))   # near-black floor (high contrast)
delete_prim("/World/Table")
FixedCuboid(prim_path="/World/Table", name="table", position=np.array([0., 1., 0.75]),
            scale=np.array([2., 2., 0.1]), size=1.0, visible=True)
vp = vu.get_active_viewport()

def qdeg(a, b):
    return float(np.rad2deg(2 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))))

CASES = [
    ("sugar",  "ycb_dex_03", "Baseline1/data/episodes_g/"
     "dexycb__20200709-subject-01__20200709_142517__840412060917__ycb_dex_03.hdf5"),
    ("tomato", "ycb_dex_04", "Baseline1/data/episodes_g/"
     "dexycb__20201015-subject-09__20201015_143403__840412060917__ycb_dex_04.hdf5"),
]
for name, objid, h5 in CASES:
    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    with h5py.File(os.path.join(PROJ, h5), "r") as f:
        oqG = np.array(f.attrs["obj_quat_G_wxyz"], dtype=float)
        place = np.array(f.attrs["obj_origin_G"], dtype=float) + SIM_ORIGIN_W
    usd = f"{PROJ}/output/obj_usd_cad/ycb/{objid}.usd"

    for i in range(10): delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    obj = RigidObject(world, usd_path=usd, pos=np.array(place),
                      ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=0.1)
    world.reset()
    objp = world.stage.GetPrimAtPath(obj.rigid_prim_path)

    # dynamic + convexHull collider (exactly what gt_replay --grasp-collision does)
    _rb = UsdPhysics.RigidBodyAPI.Get(world.stage, objp.GetPath())
    if _rb:
        _ke = _rb.GetKinematicEnabledAttr()
        (_ke if _ke else _rb.CreateKinematicEnabledAttr()).Set(False)
    for p in Usd.PrimRange(objp):
        if p.IsA(UsdGeom.Mesh):
            _ca = UsdPhysics.CollisionAPI.Apply(p)
            _ce = _ca.GetCollisionEnabledAttr()
            (_ce if _ce else _ca.CreateCollisionEnabledAttr()).Set(True)
            _mc = UsdPhysics.MeshCollisionAPI.Apply(p)
            _ap = _mc.GetApproximationAttr()
            (_ap if _ap else _mc.CreateApproximationAttr()).Set("convexHull")

    obj.rigid.set_world_pose(np.asarray(place, float), oqG)
    for _ in range(3): world.step(render=True)
    p0, q0 = obj.get_obj_pos()
    print(f"  PLACED   pos={p0.round(4)}  quat={q0.round(3)}")
    print(f"           target obj_quat_G={oqG.round(3)}  →  placement off by {qdeg(q0, oqG):.1f}°")

    set_camera_view(eye=[place[0]+0.45, place[1]+0.45, place[2]+0.33],
                    target=[float(place[0]), float(place[1]), float(place[2])],
                    camera_prim_path="/OmniverseKit_Persp")
    for _ in range(15): world.step(render=True)
    vu.capture_viewport_to_file(vp, os.path.join(OUT, f"diag2_{name}_placed.png"))
    for _ in range(12): world.step(render=True)

    for _ in range(220): world.step(render=True)              # settle under gravity
    p1, q1 = obj.get_obj_pos()
    print(f"  SETTLED  pos={p1.round(4)}  quat={q1.round(3)}")
    print(f"           >>> reoriented {qdeg(q0, q1):.1f}°  |  moved {np.linalg.norm(p1-p0)*100:.1f} cm "
          f"|  Δz {(p1[2]-p0[2])*100:+.1f} cm")
    print(f"           verdict: {'STABLE — collider matches visual' if qdeg(q0,q1)<10 else 'REORIENTED — collider misaligned with visual mesh'}")
    vu.capture_viewport_to_file(vp, os.path.join(OUT, f"diag2_{name}_settled.png"))
    for _ in range(12): world.step(render=True)

print("\n=== done ===")
sim_app.close()
