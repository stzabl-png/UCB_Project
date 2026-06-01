#!/usr/bin/env python3
"""Convert GraspVLA-playground's franka_with_extended_finger.urdf → USD via IsaacSim."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import os, sys
import omni.kit.commands

URDF = "/home/accelerator/UCB_Project/third_party/GraspVLA-playground/assets/franka_with_extended_finger/franka_with_extended_finger.urdf"
OUT_USD = "/home/accelerator/UCB_Project/sim/assets_franka/franka_extended_finger.usd"

print(f"[convert] URDF: {URDF}")
print(f"[convert] OUT:  {OUT_USD}", flush=True)

# Use URDFParseAndImport command (high-level wrapper)
from isaacsim.asset.importer.urdf import _urdf
iface = _urdf.acquire_urdf_interface()

# Configure import
cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.make_default_prim = True
cfg.distance_scale = 1.0
cfg.density = 0.0
cfg.create_physics_scene = False
cfg.self_collision = False
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength = 1e7
cfg.default_position_drive_damping = 1e5

print("[convert] parsing URDF...", flush=True)
asset_path = os.path.dirname(URDF)
asset_name = os.path.basename(URDF)
robot = iface.parse_urdf(asset_path, asset_name, cfg)

print("[convert] computing dest_path", flush=True)
dest = OUT_USD
print(f"[convert] calling import_robot with dest={dest}", flush=True)
try:
    result = iface.import_robot(asset_path, asset_name, robot, cfg, dest, "")
    print(f"[convert] result: {result}")
except TypeError as e:
    # Try alternate signature
    print(f"[convert] sig1 failed: {e}")
    try:
        result = iface.import_robot(asset_path, asset_name, robot, cfg, dest)
        print(f"[convert] result: {result}")
    except Exception as e2:
        print(f"[convert] sig2 failed: {e2}")
        # Try via command
        print("[convert] trying omni.kit.commands path", flush=True)
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=URDF,
            import_config=cfg,
            dest_path=dest)
        print("[convert] command executed")

print(f"[convert] file exists: {os.path.exists(OUT_USD)}  size: "
      f"{os.path.getsize(OUT_USD) if os.path.exists(OUT_USD) else 'NONE'}")
app.close()
