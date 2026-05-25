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
GRASP_OBJECT_MASS_KG    = 0.05      # fallback mass for objects NOT in YCB_REAL_MASS_KG

OBJECT_MATERIAL_PATH = "/World/PhysicsMaterials/BottleMaterial"
FINGER_MATERIAL_PATH = "/World/PhysicsMaterials/FingerMaterial"

# ── real YCB object masses (kg) ──────────────────────────────────────────────
# YCB-benchmark measured masses, keyed by the DexYCB object index 1-21
# (== gt_replay's obj_meta["ycb_class_id"]). Source: the official YCB object
# list (ycbbenchmarks.com .../object-list-Sheet1.pdf); sugar/tomato/mustard
# cross-checked against the Drake RobotLocomotion/models YCB SDFs (exact match).
# These are the real physical masses — use them, not the 0.05kg fallback (too
# light → a glancing gripper contact flings the object → false grasp failures).
YCB_REAL_MASS_KG = {
    1:  0.414,   # master_chef_can
    2:  0.411,   # cracker_box
    3:  0.514,   # sugar_box
    4:  0.349,   # tomato_soup_can
    5:  0.603,   # mustard_bottle
    6:  0.171,   # tuna_fish_can
    7:  0.187,   # pudding_box
    8:  0.097,   # gelatin_box
    9:  0.370,   # potted_meat_can
    10: 0.066,   # banana
    11: 0.178,   # pitcher_base
    12: 1.131,   # bleach_cleanser
    13: 0.147,   # bowl
    14: 0.118,   # mug
    15: 0.895,   # power_drill
    16: 0.729,   # wood_block
    17: 0.082,   # scissors
    18: 0.0158,  # large_marker
    19: 0.125,   # large_clamp
    20: 0.202,   # extra_large_clamp
    21: 0.028,   # foam_brick
}


def object_mass_kg(ycb_class_id, default=GRASP_OBJECT_MASS_KG):
    """Real YCB object mass (kg) for a DexYCB object index (1-21).
    Falls back to `default` for unknown ids."""
    try:
        return YCB_REAL_MASS_KG.get(int(ycb_class_id), default)
    except (TypeError, ValueError):
        return default


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
