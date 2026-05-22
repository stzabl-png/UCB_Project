#!/usr/bin/env python3
"""Shared grasp-physics setup — single source of truth for the physics of the
grasp eval. Called by BOTH sim/run_grasp_sim.py and sim/gt_replay_ikpd_v2.py
(--grasp-collision) so the two evals run under identical physics (collider +
friction materials + mass) and never drift apart.

Extracted verbatim from run_grasp_sim.py's collision/friction block. Friction is
intentionally high (rubber-fingertip anti-slip — run_grasp_sim's chosen values).

NOTE: the functions use `pxr.PhysxSchema`, which IsaacSim registers only after
SimulationApp() is created — so call them from within a running IsaacSim app.
pxr is imported lazily inside each function so this module itself imports fine
anywhere.
"""

# ── canonical grasp-physics parameters (the single source of truth) ──────────
OBJECT_STATIC_FRICTION  = 1.0
OBJECT_DYNAMIC_FRICTION = 0.8
FINGER_STATIC_FRICTION  = 1.2
FINGER_DYNAMIC_FRICTION = 1.0
RESTITUTION             = 0.0
CONTACT_OFFSET          = 0.02
REST_OFFSET             = 0.001
COLLISION_APPROXIMATION = "convexHull"
GRASP_OBJECT_MASS_KG    = 0.05      # canonical object mass for the grasp eval

OBJECT_MATERIAL_PATH = "/World/PhysicsMaterials/BottleMaterial"
FINGER_MATERIAL_PATH = "/World/PhysicsMaterials/FingerMaterial"


def _make_physics_material(stage, path, mu_s, mu_d, restitution=RESTITUTION):
    from pxr import UsdPhysics, UsdShade
    UsdShade.Material.Define(stage, path)
    mat_prim = stage.GetPrimAtPath(path)
    pm = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pm.CreateStaticFrictionAttr(mu_s)
    pm.CreateDynamicFrictionAttr(mu_d)
    pm.CreateRestitutionAttr(restitution)
    return mat_prim


def _bind_physics_material(prim, mat_prim):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")


def setup_object_grasp_physics(stage, obj_prim_path, log=None):
    """Object: convexHull collider + contact/rest offsets + a bound high-friction
    physics material (static/dynamic = 1.0/0.8). Returns the mesh-collider count."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    mat_prim = _make_physics_material(stage, OBJECT_MATERIAL_PATH,
                                      OBJECT_STATIC_FRICTION, OBJECT_DYNAMIC_FRICTION)
    if log:
        log(f"   ✅ Physics material: friction={OBJECT_STATIC_FRICTION}/{OBJECT_DYNAMIC_FRICTION}")
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(obj_prim_path)):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.GetApproximationAttr().Set(COLLISION_APPROXIMATION)
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col_api.GetContactOffsetAttr().Set(CONTACT_OFFSET)
            col_api.GetRestOffsetAttr().Set(REST_OFFSET)
            _bind_physics_material(prim, mat_prim)
            n += 1
    return n


def setup_finger_friction(stage, franka_prim_path="/World/Franka", log=None):
    """Franka fingertips: a bound high-friction physics material (1.2/1.0)."""
    from pxr import Usd, UsdGeom
    finger_mat = _make_physics_material(stage, FINGER_MATERIAL_PATH,
                                        FINGER_STATIC_FRICTION, FINGER_DYNAMIC_FRICTION)
    for finger_name in ["panda_leftfinger", "panda_rightfinger"]:
        finger_prim = stage.GetPrimAtPath(f"{franka_prim_path}/{finger_name}")
        if finger_prim.IsValid():
            for child in Usd.PrimRange(finger_prim):
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
                    _bind_physics_material(child, finger_mat)
            if log:
                log(f"   ✅ Finger friction on {finger_name}")
