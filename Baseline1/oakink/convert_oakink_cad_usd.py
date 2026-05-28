#!/usr/bin/env python3
"""Convert OakInk CAD .obj → portable USD (CPU-only, no IsaacSim).

Why we can't use the existing sim/convert_batch_usd.py for OakInk:
  - It uses Omniverse asset_converter (needs IsaacSim + GPU)
  - More importantly, the existing output/obj_usd/oakink/*.usd were built from
    SAM3D meshes (unit-cube normalized → ~10× too large) which don't match the
    CAD-based obj_origin_G in our retargeted hdf5. See OakInk Phase 2a precheck
    output for the exact mismatch table.

This script uses `pxr.Usd` + `trimesh` directly — pure CPU, sub-second per USD,
~30 KB per USD (no textures, just geometry — sim collector only needs geom +
collision, which RigidObject wrapper adds at load time).

Writes to output/obj_usd_cad/oakink/{obj_id}.usd (parallel to DexYCB's
output/obj_usd_cad/ycb/). The v4 collector's `_usd_path_for_ep` was updated to
look here.

Usage:
    python Baseline1/oakink/convert_oakink_cad_usd.py            # all use=true in manifest
    python Baseline1/oakink/convert_oakink_cad_usd.py --object A01001
    python Baseline1/oakink/convert_oakink_cad_usd.py --overwrite
"""
import argparse
import json
import os
import sys

import trimesh
from pxr import Sdf, Usd, UsdGeom

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from Baseline1.oakink.oakink_paths import (
    CLASS_ID_MAP, OAKINK_OBJ_DIR, PROJ_ROOT,
)

OUT_DIR = os.path.join(PROJ_ROOT, "output", "obj_usd_cad", "oakink")


def convert_one(obj_id: str, overwrite: bool = False) -> tuple[bool, str]:
    src = os.path.join(OAKINK_OBJ_DIR, f"{obj_id}.obj")
    dst = os.path.join(OUT_DIR, f"{obj_id}.usd")
    if not os.path.exists(src):
        return False, f"src missing: {src}"
    if os.path.exists(dst) and not overwrite:
        return True, "exists (skip)"

    try:
        mesh = trimesh.load(src, force="mesh", process=False)
    except Exception as e:
        return False, f"trimesh load failed: {e}"
    if mesh is None or len(mesh.vertices) == 0:
        return False, "empty mesh"

    stage = Usd.Stage.CreateNew(dst)
    # Z-up + metric meters → matches DexYCB CAD USD convention. v4 collector
    # assumes this when placing the object via obj_origin_G / obj_quat_G_wxyz.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Hierarchy: /World/Mesh — keep simple so RigidObject wrapper finds the prim.
    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())
    mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/Mesh"))
    mesh_prim.CreatePointsAttr(mesh.vertices.astype(float).tolist())
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    mesh_prim.CreateFaceVertexIndicesAttr(mesh.faces.flatten().astype(int).tolist())

    stage.GetRootLayer().Save()
    return True, f"{len(mesh.vertices)}v, {len(mesh.faces)}f, {os.path.getsize(dst)/1024:.1f}KB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", type=str, default=None,
                    help="single obj_id (e.g. A01001). Default: all use=true in manifest.")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing USD")
    args = ap.parse_args()

    with open(CLASS_ID_MAP) as f:
        cmap = json.load(f)
    objects = {oid: info for oid, info in cmap["objects"].items() if info.get("use")}
    if args.object:
        if args.object not in cmap["objects"]:
            sys.exit(f"obj_id {args.object} not in manifest")
        objects = {args.object: cmap["objects"][args.object]}

    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_fail = 0
    failed = []
    for oid in sorted(objects):
        ok, msg = convert_one(oid, overwrite=args.overwrite)
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {oid}: {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed.append(oid)
    print(f"\n=== DONE: {n_ok} ok / {n_fail} fail ===")
    print(f"   output dir: {OUT_DIR}")
    if failed:
        print(f"   failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
