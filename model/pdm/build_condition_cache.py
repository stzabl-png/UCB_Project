#!/usr/bin/env python3
"""Build precomputed object condition cache for PDM.

This script prepares one fixed `(points, normals, affordance)` condition tensor
per object before training. It avoids repeated mesh loading/surface sampling in
DataLoader workers and mirrors the HDF5 layout used by the affordance v6 data:

  data/points      (M, N, 3)
  data/normals     (M, N, 3)
  data/affordance  (M, N)
  data/obj_ids     (M,)

If an affordance HDF5 is supplied, the cache reuses its aligned points/normals
and labels. Otherwise it samples mesh surfaces and fills affordance with zeros.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.pdm.dataset import (
    AffordanceStore,
    DEFAULT_MERGED_DIR,
    DEFAULT_ROTATED_MESH_DIR,
    PDMMergedDataset,
    sample_mesh_condition,
    _resample_rows,
)


def _condition_stats(points: np.ndarray, normals: np.ndarray) -> dict[str, float]:
    normal_norm = np.linalg.norm(normals, axis=1)
    return {
        "points_abs_max": float(np.nanmax(np.abs(points))),
        "normal_norm_min": float(np.nanmin(normal_norm)),
        "normal_norm_max": float(np.nanmax(normal_norm)),
        "normal_norm_mean": float(np.nanmean(normal_norm)),
        "finite": float(np.isfinite(points).all() and np.isfinite(normals).all()),
    }


def _is_condition_sane(points: np.ndarray, normals: np.ndarray, max_abs: float) -> tuple[bool, str]:
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        return False, "non_finite"
    if float(np.max(np.abs(points))) > max_abs:
        return False, "points_abs_too_large"
    n = np.linalg.norm(normals, axis=1)
    if not np.isfinite(n).all():
        return False, "normal_non_finite"
    if float(np.nanmean(n)) < 0.5:
        return False, "normal_norm_too_small"
    return True, ""


def object_ids_from_merged(merged_dir: str) -> list[str]:
    dataset = PDMMergedDataset(
        merged_dir=merged_dir,
        n_points=16,
        require_trusted_tips=False,
        cache_conditions=False,
    )
    return sorted({meta.obj_id for meta, _pose in dataset.rows})


def build_cache(args: argparse.Namespace) -> None:
    obj_ids = args.obj or object_ids_from_merged(args.merged_dir)
    if not obj_ids:
        raise RuntimeError(f"No objects found under {args.merged_dir}")

    aff_store = AffordanceStore(args.affordance_h5)
    points_out = []
    normals_out = []
    aff_out = []
    kept_ids = []
    stats_rows = []
    skipped: list[tuple[str, str]] = []

    print("=" * 72)
    print("Build PDM condition cache")
    print(f"  objects:    {len(obj_ids)}")
    print(f"  n_points:   {args.n_points}")
    print(f"  output:     {args.output}")
    print(f"  affordance: {args.affordance_h5 or '(none, zeros)'}")
    print("=" * 72)

    for i, obj_id in enumerate(obj_ids, 1):
        try:
            if aff_store.has(obj_id):
                cond = aff_store.load(obj_id)
                arr = _resample_rows(cond.points, args.n_points)
                source = "affordance_h5"
            else:
                cond = sample_mesh_condition(obj_id, args.n_points, mesh_root=args.mesh_root)
                arr = cond.points
                source = "mesh_sample"
            pts = np.ascontiguousarray(arr[:, :3].astype(np.float32))
            nrm = np.ascontiguousarray(arr[:, 3:6].astype(np.float32))
            aff = np.ascontiguousarray(arr[:, 6].astype(np.float32))
            ok, reason = _is_condition_sane(pts, nrm, args.max_abs_coord)
            if not ok:
                skipped.append((obj_id, reason))
                print(f"  [{i:03d}/{len(obj_ids)}] skip {obj_id}: {reason}")
                continue
            points_out.append(pts)
            normals_out.append(nrm)
            aff_out.append(aff)
            kept_ids.append(obj_id)
            row = _condition_stats(pts, nrm)
            row["source"] = source
            stats_rows.append(row)
            if i == 1 or i % args.log_every == 0:
                print(
                    f"  [{i:03d}/{len(obj_ids)}] {obj_id}  {source}  "
                    f"absmax={row['points_abs_max']:.3f}  "
                    f"|n|={row['normal_norm_mean']:.3f}"
                )
        except Exception as exc:
            skipped.append((obj_id, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i:03d}/{len(obj_ids)}] skip {obj_id}: {type(exc).__name__}: {exc}")

    if not kept_ids:
        raise RuntimeError("No valid object conditions generated")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["num_samples"] = len(kept_ids)
        meta.attrs["num_points"] = args.n_points
        meta.attrs["source"] = "model.pdm.build_condition_cache"
        meta.attrs["merged_dir"] = os.path.abspath(args.merged_dir)
        meta.attrs["mesh_root"] = os.path.abspath(args.mesh_root)
        meta.attrs["affordance_h5"] = os.path.abspath(args.affordance_h5) if args.affordance_h5 else ""
        meta.attrs["skipped_count"] = len(skipped)

        data = f.create_group("data")
        data.create_dataset(
            "points",
            data=np.stack(points_out, axis=0),
            compression="gzip",
            compression_opts=4,
        )
        data.create_dataset(
            "normals",
            data=np.stack(normals_out, axis=0),
            compression="gzip",
            compression_opts=4,
        )
        data.create_dataset(
            "affordance",
            data=np.stack(aff_out, axis=0),
            compression="gzip",
            compression_opts=4,
        )
        data.create_dataset("obj_ids", data=np.asarray(kept_ids, dtype=str_dt))
        data.create_dataset("source", data=np.asarray([r["source"] for r in stats_rows], dtype=str_dt))
        data.create_dataset(
            "points_abs_max",
            data=np.asarray([r["points_abs_max"] for r in stats_rows], dtype=np.float32),
        )
        data.create_dataset(
            "normal_norm_mean",
            data=np.asarray([r["normal_norm_mean"] for r in stats_rows], dtype=np.float32),
        )
        if skipped:
            sk = f.create_group("skipped")
            sk.create_dataset("obj_ids", data=np.asarray([s[0] for s in skipped], dtype=str_dt))
            sk.create_dataset("reasons", data=np.asarray([s[1] for s in skipped], dtype=str_dt))

    print("=" * 72)
    print(f"Saved {len(kept_ids)} object conditions -> {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} objects")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PDM object condition cache")
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--output", default=os.path.join(PROJ, "output", "pdm", "cache", "conditions_4096.h5"))
    parser.add_argument("--affordance-h5", default=None)
    parser.add_argument("--mesh-root", default=DEFAULT_ROTATED_MESH_DIR)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--max-abs-coord", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--obj", nargs="*", default=None, help="Optional subset of object ids")
    return parser


if __name__ == "__main__":
    build_cache(build_parser().parse_args())
