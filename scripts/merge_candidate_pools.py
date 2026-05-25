#!/usr/bin/env python3
"""
merge_candidate_pools.py — 合并两个 candidate pool 目录到单一 pool
====================================================================
按物体 ID 合并：两边都有则拼接全部 candidate（按 score 降序），重编号
candidate_0..N-1；仅一边有则直接采用该边。

默认:
  --dir-a  output/pool_500_all
  --dir-b  output/pool_500_threshold30
  --output-dir output/pool

用法:
    python3 scripts/merge_candidate_pools.py
    python3 scripts/merge_candidate_pools.py --dry-run
    python3 scripts/merge_candidate_pools.py --dir-a output/pool_500_all \\
        --dir-b output/pool_500_threshold30 --output-dir output/pool
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))

from mesh_utils import read_mesh_prerotation_hdf5_pose
from random_grasp_sampler import SAMPLING_METHOD_RAYCAST, save_candidates_hdf5

DEFAULT_DIR_A = os.path.join(PROJ, "output", "pool_500_all")
DEFAULT_DIR_B = os.path.join(PROJ, "output", "pool_500_threshold30")
DEFAULT_OUTPUT = os.path.join(PROJ, "output", "pool")
MANIFEST_NAME = "merge_pool_manifest.json"


def _attr_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _read_metadata(f: h5py.File) -> dict:
    if "metadata" not in f:
        return {}
    m = f["metadata"]
    meta = {}
    for key in m.attrs:
        v = m.attrs[key]
        if isinstance(v, (bytes, np.bytes_)):
            meta[key] = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        elif isinstance(v, (np.floating, np.integer)):
            meta[key] = v.item()
        elif isinstance(v, np.ndarray):
            meta[key] = v.tolist()
        else:
            meta[key] = v
    return meta


def load_candidates_from_pool(path: str) -> tuple[list[dict], dict]:
    """Load grasp candidates + file metadata from pool HDF5."""
    rows: list[dict] = []
    meta: dict = {}
    with h5py.File(path, "r") as f:
        meta = _read_metadata(f)
        if "candidates" not in f:
            return rows, meta
        cg = f["candidates"]
        n = int(cg.attrs.get("n_candidates", 0))
        for i in range(n):
            gname = f"candidate_{i}"
            if gname not in cg:
                continue
            g = cg[gname]
            pr = read_mesh_prerotation_hdf5_pose(g, f)
            rows.append(
                {
                    "position": np.array(g["position"][:], dtype=np.float64),
                    "grasp_point": np.array(g["grasp_point"][:], dtype=np.float64),
                    "rotation": np.array(g["rotation"][:], dtype=np.float64),
                    "name": _attr_str(g.attrs.get("name", gname)),
                    "score": float(g.attrs.get("score", 0.0)),
                    "gripper_width": float(g.attrs.get("gripper_width", 0.04)),
                    "cross_section_width": float(
                        g.attrs.get("cross_section_width", 0.0),
                    ),
                    "d_near": float(g.attrs.get("d_near", -1.0)),
                    "mesh_prerotation": pr,
                    "_source_file": os.path.abspath(path),
                    "_source_idx": i,
                }
            )
    return rows, meta


def scan_pool_objects(*pool_dirs: str) -> list[str]:
    ids: set[str] = set()
    for d in pool_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith("_grasp.hdf5"):
                ids.add(fn[: -len("_grasp.hdf5")])
    return sorted(ids)


def pool_path(pool_dir: str, obj_id: str) -> str:
    return os.path.join(pool_dir, f"{obj_id}_grasp.hdf5")


def merge_one_object(
    obj_id: str,
    dir_a: str,
    dir_b: str,
    output_dir: str,
    *,
    dry_run: bool,
) -> dict:
    path_a = pool_path(dir_a, obj_id)
    path_b = pool_path(dir_b, obj_id)
    has_a = os.path.isfile(path_a)
    has_b = os.path.isfile(path_b)

    if not has_a and not has_b:
        return {
            "obj_id": obj_id,
            "status": "missing",
            "n_out": 0,
            "n_a": 0,
            "n_b": 0,
        }

    rows: list[dict] = []
    meta: dict = {}
    n_a = n_b = 0

    if has_a:
        ra, ma = load_candidates_from_pool(path_a)
        n_a = len(ra)
        rows.extend(ra)
        meta = ma
    if has_b:
        rb, mb = load_candidates_from_pool(path_b)
        n_b = len(rb)
        rows.extend(rb)
        if len(mb) >= len(meta):
            meta = mb

    if not rows:
        return {
            "obj_id": obj_id,
            "status": "empty",
            "n_out": 0,
            "n_a": n_a,
            "n_b": n_b,
        }

    rows.sort(key=lambda r: (-float(r["score"]), r.get("_source_file", ""), r.get("_source_idx", 0)))
    for r in rows:
        r.pop("_source_file", None)
        r.pop("_source_idx", None)

    n_out = len(rows)
    if has_a and has_b:
        status = "merged"
    elif has_a:
        status = "from_a_only"
    else:
        status = "from_b_only"

    out_path = pool_path(output_dir, obj_id)
    if dry_run:
        return {
            "obj_id": obj_id,
            "status": status,
            "n_out": n_out,
            "n_a": n_a,
            "n_b": n_b,
            "output": out_path,
        }

    mesh_path = _attr_str(meta.get("mesh_path", ""))
    if not mesh_path:
        raise ValueError(f"{obj_id}: metadata.mesh_path missing in pool HDF5")

    dataset = _attr_str(meta.get("dataset", "oakink")) or "oakink"
    hp_path = meta.get("hp_path")
    if hp_path is not None:
        hp_path = _attr_str(hp_path) or None

    sampling = _attr_str(
        meta.get("sampling_method", meta.get("method", SAMPLING_METHOD_RAYCAST)),
    ) or SAMPLING_METHOD_RAYCAST

    save_candidates_hdf5(
        rows,
        obj_id,
        mesh_path,
        output_dir,
        no_rotation=bool(meta.get("no_rotation", True)),
        dataset=dataset,
        scale_factor=float(meta.get("scale_factor", 1.0)),
        apply_scale_to_mesh=bool(meta.get("scale_applied_to_mesh", True)),
        hp_scale_applied=bool(meta.get("hp_scale_applied_on_load", False)),
        hp_path=hp_path,
        sampling_method=sampling,
    )

    return {
        "obj_id": obj_id,
        "status": status,
        "n_out": n_out,
        "n_a": n_a,
        "n_b": n_b,
        "output": out_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two candidate pool directories into one (per-object concat by score)",
    )
    parser.add_argument(
        "--dir-a",
        default=DEFAULT_DIR_A,
        help=f"First pool directory (default: {DEFAULT_DIR_A})",
    )
    parser.add_argument(
        "--dir-b",
        default=DEFAULT_DIR_B,
        help=f"Second pool directory (default: {DEFAULT_DIR_B})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"Merged output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merge plan only, do not write HDF5",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"Write JSON manifest (default: {{output-dir}}/{MANIFEST_NAME})",
    )
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)
    out_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(dir_a) and not os.path.isdir(dir_b):
        print(f"❌ Neither pool dir exists:\n  {dir_a}\n  {dir_b}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    obj_ids = scan_pool_objects(dir_a, dir_b)
    if not obj_ids:
        print("❌ No *_grasp.hdf5 in input directories")
        sys.exit(1)

    print(f"dir_a:      {dir_a}")
    print(f"dir_b:      {dir_b}")
    print(f"output:     {out_dir}")
    print(f"objects:    {len(obj_ids)}")
    print(f"dry_run:    {args.dry_run}")
    print("-" * 60)

    results: list[dict] = []
    counts = {"merged": 0, "from_a_only": 0, "from_b_only": 0, "empty": 0, "missing": 0}

    for obj_id in obj_ids:
        try:
            row = merge_one_object(
                obj_id, dir_a, dir_b, out_dir, dry_run=args.dry_run,
            )
        except Exception as e:
            row = {
                "obj_id": obj_id,
                "status": "error",
                "error": str(e),
                "n_out": 0,
                "n_a": 0,
                "n_b": 0,
            }
        results.append(row)
        st = row.get("status", "error")
        counts[st] = counts.get(st, 0) + 1
        if st == "error":
            print(f"  {obj_id}: ERROR  {row.get('error', '')[:120]}")
        elif st in ("merged", "from_a_only", "from_b_only"):
            print(
                f"  {obj_id}: {st}  n={row['n_out']}  "
                f"(a={row['n_a']}, b={row['n_b']})",
            )
        else:
            print(f"  {obj_id}: {st}")

    total_cand = sum(r.get("n_out", 0) for r in results)
    print("-" * 60)
    print(
        f"Done. merged={counts.get('merged', 0)}  "
        f"a_only={counts.get('from_a_only', 0)}  "
        f"b_only={counts.get('from_b_only', 0)}  "
        f"errors={counts.get('error', 0)}",
    )
    print(f"Total candidates in output: {total_cand}")

    if not args.dry_run:
        manifest_path = args.manifest or os.path.join(out_dir, MANIFEST_NAME)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dir_a": dir_a,
            "dir_b": dir_b,
            "output_dir": out_dir,
            "n_objects": len(obj_ids),
            "total_candidates": total_cand,
            "counts": counts,
            "objects": results,
        }
        with open(manifest_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Manifest: {manifest_path}")

    if counts.get("error", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
