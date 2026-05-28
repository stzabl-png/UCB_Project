#!/usr/bin/env python3
"""OakInk Phase 2a — USD metadata precheck (CPU-only, no IsaacSim load).

For each `use=true` object in Baseline1/oakink/class_id_map.json, parses the
USD file at output/obj_usd/oakink/{obj_id}.usd via pxr.Usd directly (no scene
spawn) and checks:

  1. File exists
  2. upAxis = Z (else PhysX collider/visual split — see usd-zup-pitfall memory)
  3. metersPerUnit = 1.0 (no unit-mismatch surprises)
  4. At least one Mesh prim with non-zero verts
  5. Bbox roughly matches manifest's reported AABB (within 30%)

This is FAST (sub-second per USD). It catches gross mismatches before the
heavier PhysX settle test (Phase 2b) needs to spin up IsaacSim per object.

Outputs Baseline1/oakink/assets/phase2a_usd_precheck.json with per-obj report.
Usage:
    python Baseline1/oakink/phase2a_usd_precheck.py
"""
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from Baseline1.oakink.oakink_paths import CLASS_ID_MAP, OAKINK_USD_DIR


def check_one_usd(obj_id: str, manifest_aabb_cm: list[float]) -> dict:
    out = {"obj_id": obj_id, "checks": {}}
    usd_path = os.path.join(OAKINK_USD_DIR, f"{obj_id}.usd")
    out["usd_path"] = usd_path

    # G1: file exists
    if not os.path.exists(usd_path):
        out["checks"]["file_exists"] = False
        out["pass"] = False
        return out
    out["checks"]["file_exists"] = True
    out["size_kb"] = round(os.path.getsize(usd_path) / 1024, 1)

    # Use pxr.Usd to inspect (must come after IsaacSim import in caller? No —
    # pxr.Usd is available standalone via usd-core pip package, no SimulationApp).
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        out["checks"]["pxr_available"] = False
        out["pass"] = False
        return out
    out["checks"]["pxr_available"] = True

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        out["checks"]["stage_open"] = f"FAIL: {e}"
        out["pass"] = False
        return out
    out["checks"]["stage_open"] = True

    # G2: upAxis = Z
    up = str(UsdGeom.GetStageUpAxis(stage))
    out["upAxis"] = up
    out["checks"]["upAxis_Z"] = (up == "Z")

    # G3: metersPerUnit = 1.0
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    out["metersPerUnit"] = mpu
    out["checks"]["metersPerUnit_1"] = abs(mpu - 1.0) < 1e-6

    # G4: at least one mesh with verts
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if pts is not None and len(pts) > 0:
                meshes.append((str(prim.GetPath()), np.array(pts)))
    out["n_meshes"] = len(meshes)
    out["total_verts"] = sum(len(v) for _, v in meshes)
    out["checks"]["has_mesh"] = len(meshes) > 0

    # G5: bbox matches manifest within 30%
    if meshes:
        all_pts = np.vstack([v for _, v in meshes])
        bbox_cm = (all_pts.max(0) - all_pts.min(0)) * 100
        out["usd_bbox_cm"] = bbox_cm.round(2).tolist()
        out["manifest_aabb_cm"] = manifest_aabb_cm
        # sorted comparison (rotation-invariant)
        sorted_usd = sorted(bbox_cm.tolist())
        sorted_man = sorted(manifest_aabb_cm)
        rel_err = max(abs(u - m) / max(m, 1e-6) for u, m in zip(sorted_usd, sorted_man))
        out["bbox_rel_err"] = round(rel_err, 3)
        out["checks"]["bbox_match_30pct"] = rel_err <= 0.30

    out["pass"] = all(v is True for v in out["checks"].values())
    return out


def main():
    with open(CLASS_ID_MAP) as f:
        cmap = json.load(f)
    objects = {oid: info for oid, info in cmap["objects"].items() if info.get("use")}

    reports = []
    pass_n = fail_n = 0
    for oid, info in sorted(objects.items()):
        rep = check_one_usd(oid, info.get("aabb_cm", [0, 0, 0]))
        reports.append(rep)
        if rep["pass"]:
            pass_n += 1
            print(f"  PASS  {oid}  bbox={rep.get('usd_bbox_cm', 'n/a')}  {rep['total_verts']}v")
        else:
            fail_n += 1
            failed = [k for k, v in rep["checks"].items() if v is not True]
            print(f"  FAIL  {oid}  failed_checks={failed}")
            if "usd_bbox_cm" in rep:
                print(f"          usd_bbox={rep['usd_bbox_cm']} vs manifest={rep['manifest_aabb_cm']}")

    out_path = os.path.join(_THIS, "assets", "phase2a_usd_precheck.json")
    with open(out_path, "w") as f:
        json.dump({
            "_generated": "2026-05-25",
            "_n_objects": len(reports),
            "_n_pass": pass_n,
            "_n_fail": fail_n,
            "reports": reports,
        }, f, indent=2)

    print(f"\n=== {pass_n} pass / {fail_n} fail (of {len(reports)}) → {out_path}")
    sys.exit(0 if fail_n == 0 else 1)


if __name__ == "__main__":
    main()
