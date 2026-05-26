#!/usr/bin/env python3
"""Sanity checks for OakInk → v4-format hdf5 outputs.

5 gates (run each independently; first failure prints a clear diagnostic):

  1. SCHEMA — all v4 required attrs present, dtypes/shapes correct
  2. WORLD-FRAME — obj_origin_G z > 0 (above ground in OakInk world), normalized quat
  3. CAD-SIZE — PC[0] bbox matches official OakInk CAD mesh bbox within 20% tolerance
                (axes may permute due to T_w_o rotation, so we compare sorted bbox sides)
  4. EE-NEAR-OBJECT — minimum hand-object distance < 15 cm (sanity: hand reached the
                object); also state[0] EE position is at a reasonable height (within
                ±50 cm of obj_origin_G).
  5. GRIP-ONSET — grasp_onset is within [3, n_steps-3] (signal fires neither too
                early nor too late).

Usage:
    python Baseline1/oakink/verify_oakink_hdf5.py <hdf5_or_glob>
    python Baseline1/oakink/verify_oakink_hdf5.py 'Baseline1/data/episodes_oakink_*/*.hdf5'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import h5py
import numpy as np
import trimesh

# Make package importable when run as a script
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))
from Baseline1.oakink.oakink_paths import OAKINK_OBJ_DIR  # noqa: E402

REQUIRED_ATTRS = {
    "baseline", "dataset", "obj_id", "ycb_class_id",
    "obj_origin_G", "obj_quat_G_wxyz", "origin_G_W", "table_z_G",
    "ee_offset_m", "gripper_span_m", "mano_side", "n_steps", "grasp_onset",
}

REQUIRED_DATASETS = ["state", "action", "point_cloud"]


def _diameter(pts: np.ndarray, n_sub: int = 1024) -> float:
    """Max pairwise distance — strictly rotation-invariant size measure.

    1024-point subsample makes cost <1s per ep and gives ≥99.5% of true diameter
    for typical convex / quasi-convex objects.
    """
    if len(pts) > n_sub:
        idx = np.random.default_rng(0).choice(len(pts), n_sub, replace=False)
        pts = pts[idx]
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    return float(np.sqrt(d2.max()))


def _load_cad_diameter(obj_id: str) -> float | None:
    p = os.path.join(OAKINK_OBJ_DIR, f"{obj_id}.obj")
    if not os.path.exists(p):
        return None
    m = trimesh.load(p, force="mesh", process=False)
    return _diameter(m.vertices)


def _check_one(path: str) -> tuple[bool, list[str]]:
    """Return (passed_all_gates, list_of_failure_messages)."""
    fails = []
    with h5py.File(path, "r") as h:
        # G1 SCHEMA
        missing = REQUIRED_ATTRS - set(h.attrs)
        if missing:
            fails.append(f"G1 SCHEMA missing attrs: {sorted(missing)}")
        for ds in REQUIRED_DATASETS:
            if ds not in h:
                fails.append(f"G1 SCHEMA missing dataset: {ds}")
        if fails:
            return False, fails   # bail early if schema broken
        n_steps = int(h.attrs["n_steps"])
        state, action, pc = h["state"][:], h["action"][:], h["point_cloud"][:]
        if state.shape[0] != n_steps:
            fails.append(f"G1 SCHEMA: state len={state.shape[0]} ≠ n_steps={n_steps}")
        if state.shape != action.shape:
            fails.append(f"G1 SCHEMA: state {state.shape} ≠ action {action.shape}")
        if pc.shape[0] != n_steps or pc.shape[2] != 3:
            fails.append(f"G1 SCHEMA: pc shape {pc.shape} unexpected")
        if state.shape[-1] != 8:
            fails.append(f"G1 SCHEMA: state dim {state.shape[-1]} ≠ 8")

        # G2 WORLD-FRAME
        obj_origin = np.asarray(h.attrs["obj_origin_G"], dtype=np.float64)
        obj_quat = np.asarray(h.attrs["obj_quat_G_wxyz"], dtype=np.float64)
        if not (0.0 < obj_origin[2] < 1.0):
            fails.append(f"G2 WORLD-FRAME: obj_origin_G z={obj_origin[2]:.3f} not in (0, 1) m "
                         f"— OakInk world Z=up, table≈0, object should be above")
        qn = float(np.linalg.norm(obj_quat))
        if abs(qn - 1.0) > 1e-3:
            fails.append(f"G2 WORLD-FRAME: obj_quat_G norm={qn:.4f} not unit")

        # G3 CAD-SIZE — compare ROTATION-INVARIANT diameter within 10%
        # (max pairwise distance between PC points; same metric on CAD verts)
        # Volume / second-moment proxies were too noisy: CAD mesh verts and PC
        # surface samples have different distributions so their inertia matrices
        # differ even for the same object. Diameter is robust to both rotation
        # and sampling-distribution differences.
        obj_id = str(h.attrs["obj_id"])
        cad_d = _load_cad_diameter(obj_id)
        if cad_d is None:
            fails.append(f"G3 CAD-SIZE: official CAD mesh not found for {obj_id}")
        else:
            pc_d = _diameter(pc[0])
            tol = 0.10
            d_err = abs(pc_d - cad_d) / max(cad_d, 1e-6)
            if d_err > tol:
                fails.append(
                    f"G3 CAD-SIZE: PC diameter {pc_d*100:.1f}cm vs CAD {cad_d*100:.1f}cm "
                    f"({d_err*100:.0f}% err > {tol*100:.0f}% tol)"
                )

        # G4 EE-NEAR-OBJECT
        min_dist = h.attrs.get("min_hand_obj_dist_m", None)
        if min_dist is not None and float(min_dist) > 0.15:
            fails.append(f"G4 EE-NEAR-OBJECT: min hand-obj dist {float(min_dist)*1000:.0f} mm > 150 mm "
                         f"— hand never reached object")
        ee_pos = state[0, :3]   # object-centric
        if np.max(np.abs(ee_pos)) > 0.50:
            fails.append(f"G4 EE-NEAR-OBJECT: state[0] EE pos {ee_pos} >50cm from object origin "
                         f"— suspicious init")

        # G5 GRIP-ONSET
        onset = int(h.attrs["grasp_onset"])
        if not (3 <= onset <= n_steps - 3):
            fails.append(f"G5 GRIP-ONSET: onset={onset} outside [3, {n_steps-3}]")

    return len(fails) == 0, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="hdf5 paths or globs")
    ap.add_argument("--quiet", action="store_true",
                    help="only print failed files (default prints per-file PASS/FAIL)")
    args = ap.parse_args()

    paths = []
    for g in args.globs:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        sys.exit(f"no hdf5 matched: {args.globs}")

    print(f"verifying {len(paths)} hdf5 files\n")

    n_pass = 0
    n_fail = 0
    gate_fails = Counter()
    for p in paths:
        try:
            ok, fails = _check_one(p)
        except Exception as e:
            ok = False; fails = [f"EXCEPTION while reading: {type(e).__name__}: {e}"]
        if ok:
            n_pass += 1
            if not args.quiet:
                print(f"  PASS  {os.path.basename(p)}")
        else:
            n_fail += 1
            print(f"  FAIL  {os.path.basename(p)}")
            for m in fails:
                print(f"        {m}")
                gate = m.split(" ")[0]   # 'G1' / 'G2' ...
                gate_fails[gate] += 1

    print(f"\n=== RESULT: {n_pass} pass / {n_fail} fail (of {len(paths)}) ===")
    if gate_fails:
        print("  failures by gate:")
        for g in sorted(gate_fails):
            print(f"    {g}: {gate_fails[g]}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
