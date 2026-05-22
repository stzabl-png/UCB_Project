#!/usr/bin/env python3
"""Re-tag a DexYCB CAD USD as Z-up — upAxis METADATA ONLY, geometry untouched.

Why this is the correct fix (verified against the training data):
  These CAD USDs hold the raw DexYCB `.obj` vertex coordinates plus an `upAxis=Y`
  tag. The retargeting pipeline (build_gt_replay) computed `obj_quat_G` directly
  against those raw `.obj` coordinates — confirmed: R(obj_quat_G) @ raw_obj_verts
  reproduces the training-data point cloud. So the raw coords ARE the canonical
  frame; obj_quat_G expects them as-is, with NO up-axis correction.

  The `upAxis=Y` tag only makes IsaacSim's metricsAssembler inject an UNWANTED
  up-axis correction. That correction reaches rendering/USD but NOT PhysX, so a
  dynamic rigid body ends up ~90deg split (visual vs collider/physics).

  Setting upAxis=Z to match the Z-up sim stage suppresses the metricsAssembler
  correction entirely → the raw geometry is used directly → obj_quat_G places the
  object correctly AND rendering + PhysX share one frame.

  Geometry is NOT rotated. Only the stage upAxis metadata changes.

Usage:  python sim/convert_usd_zup.py <a.usd> <b.usd> ...
Backs up each original to <path>.ybak (only if no backup exists yet).
"""
import sys, os, shutil
from pxr import Usd, UsdGeom


def convert(path):
    print(f"\n=== {path} ===")
    if not os.path.exists(path):
        print("  MISSING — skip"); return
    stage = Usd.Stage.Open(path)
    up = UsdGeom.GetStageUpAxis(stage)
    print(f"  upAxis before = {up}")
    if up == UsdGeom.Tokens.z:
        print("  already Z-up — skip"); return
    bak = path + ".ybak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()
    print(f"  upAxis after  = {UsdGeom.GetStageUpAxis(stage)}  — SAVED (geometry untouched)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: convert_usd_zup.py <usd> [<usd> ...]"); sys.exit(1)
    for p in sys.argv[1:]:
        convert(p)
    print("\n=== done ===")
