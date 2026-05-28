#!/usr/bin/env python3
"""Build precomputed object condition cache for PDM.

Each object stores one fixed ``(points, normals, affordance)`` tensor for training.
Affordance is **affordance v6 checkpoint prediction** on the metric rotated mesh
(same pipeline as ``glb_to_pdm_grasp`` / ``batch_pdm_candidates``), not GT soft labels.

HDF5 layout::

  data/points      (M, N, 3)
  data/normals     (M, N, 3)
  data/affordance  (M, N)   # v6 predicted heatmap
  data/obj_ids     (M,)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.inference_v6 import default_threshold, load_model
from model.pdm.dataset import DEFAULT_MERGED_DIR, DEFAULT_ROTATED_MESH_DIR, PDMMergedDataset
from model.pdm.mesh_points import prepare_metric_point_cloud, predict_affordance_v6

DEFAULT_AFF_CKPT = os.path.join(
    PROJ,
    "output",
    "affordance_no_rot_executed",
    "min20",
    "checkpoints_v6",
    "best_v6_model.pth",
)
DEFAULT_OUTPUT = os.path.join(PROJ, "output", "pdm", "cache", "conditions_4096_v6pred.h5")


def _condition_stats(points: np.ndarray, normals: np.ndarray, affordance: np.ndarray) -> dict:
    normal_norm = np.linalg.norm(normals, axis=1)
    return {
        "points_abs_max": float(np.nanmax(np.abs(points))),
        "normal_norm_mean": float(np.nanmean(normal_norm)),
        "aff_max": float(np.nanmax(affordance)),
        "aff_mean": float(np.nanmean(affordance)),
        "finite": float(
            np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(affordance).all()
        ),
    }


def _is_condition_sane(
    points: np.ndarray,
    normals: np.ndarray,
    affordance: np.ndarray,
    max_abs: float,
) -> tuple[bool, str]:
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        return False, "non_finite_xyz"
    if not np.isfinite(affordance).all():
        return False, "non_finite_affordance"
    if float(np.max(np.abs(points))) > max_abs:
        return False, "points_abs_too_large"
    n = np.linalg.norm(normals, axis=1)
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


def _object_seed(obj_id: str, base: int) -> int:
    return (int(base) + sum(ord(c) for c in obj_id)) % (2**31 - 1)


def build_cache(args: argparse.Namespace) -> None:
    obj_ids = args.obj or object_ids_from_merged(args.merged_dir)
    if not obj_ids:
        raise RuntimeError(f"No objects found under {args.merged_dir}")

    aff_ckpt = os.path.abspath(args.affordance_checkpoint)
    if not os.path.isfile(aff_ckpt):
        raise FileNotFoundError(f"Affordance checkpoint not found: {aff_ckpt}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset_dir = args.dataset_dir or os.path.dirname(os.path.dirname(aff_ckpt))
    threshold = default_threshold(aff_ckpt, os.path.abspath(dataset_dir))
    aff_model, _ = load_model(aff_ckpt, device)

    points_out: list[np.ndarray] = []
    normals_out: list[np.ndarray] = []
    aff_out: list[np.ndarray] = []
    mesh_paths: list[str] = []
    kept_ids: list[str] = []
    stats_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    print("=" * 72)
    print("Build PDM condition cache (affordance v6 predictions)")
    print(f"  objects:              {len(obj_ids)}")
    print(f"  n_points:             {args.n_points}")
    print(f"  affordance checkpoint: {aff_ckpt}")
    print(f"  mesh_root:            {args.mesh_root}")
    print(f"  output:               {args.output}")
    print(f"  device:               {device}")
    print("=" * 72)

    for i, obj_id in enumerate(obj_ids, 1):
        try:
            seed = _object_seed(obj_id, args.seed)
            points, normals, mesh_path = prepare_metric_point_cloud(
                obj_id,
                mesh_root=args.mesh_root,
                num_points=args.n_points,
                seed=seed,
                target_max_extent=args.target_max_extent,
                auto_extent_lo=args.auto_extent_lo,
                auto_extent_hi=args.auto_extent_hi,
                min_scale_factor=args.min_scale_factor,
            )
            pred = predict_affordance_v6(aff_model, points, normals, device)
            pts = np.ascontiguousarray(points.astype(np.float32))
            nrm = np.ascontiguousarray(normals.astype(np.float32))
            aff = np.ascontiguousarray(pred.astype(np.float32))
            ok, reason = _is_condition_sane(pts, nrm, aff, args.max_abs_coord)
            if not ok:
                skipped.append((obj_id, reason))
                print(f"  [{i:03d}/{len(obj_ids)}] skip {obj_id}: {reason}")
                continue
            points_out.append(pts)
            normals_out.append(nrm)
            aff_out.append(aff)
            mesh_paths.append(mesh_path)
            kept_ids.append(obj_id)
            row = _condition_stats(pts, nrm, aff)
            row["source"] = "v6_prediction"
            stats_rows.append(row)
            if i == 1 or i % args.log_every == 0:
                print(
                    f"  [{i:03d}/{len(obj_ids)}] {obj_id}  v6_pred  "
                    f"absmax={row['points_abs_max']:.3f}  aff_max={row['aff_max']:.3f}  "
                    f"aff_mean={row['aff_mean']:.4f}"
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
        meta.attrs["affordance_source"] = "v6_prediction"
        meta.attrs["affordance_checkpoint"] = aff_ckpt
        meta.attrs["affordance_threshold"] = float(threshold)
        meta.attrs["merged_dir"] = os.path.abspath(args.merged_dir)
        meta.attrs["mesh_root"] = os.path.abspath(args.mesh_root)
        meta.attrs["skipped_count"] = len(skipped)

        data = f.create_group("data")
        data.create_dataset("points", data=np.stack(points_out, axis=0), compression="gzip", compression_opts=4)
        data.create_dataset("normals", data=np.stack(normals_out, axis=0), compression="gzip", compression_opts=4)
        data.create_dataset("affordance", data=np.stack(aff_out, axis=0), compression="gzip", compression_opts=4)
        data.create_dataset("obj_ids", data=np.asarray(kept_ids, dtype=str_dt))
        data.create_dataset("mesh_paths", data=np.asarray(mesh_paths, dtype=str_dt))
        data.create_dataset("source", data=np.asarray([r["source"] for r in stats_rows], dtype=str_dt))
        data.create_dataset(
            "points_abs_max",
            data=np.asarray([r["points_abs_max"] for r in stats_rows], dtype=np.float32),
        )
        data.create_dataset(
            "aff_max",
            data=np.asarray([r["aff_max"] for r in stats_rows], dtype=np.float32),
        )
        if skipped:
            sk = f.create_group("skipped")
            sk.create_dataset("obj_ids", data=np.asarray([s[0] for s in skipped], dtype=str_dt))
            sk.create_dataset("reasons", data=np.asarray([s[1] for s in skipped], dtype=str_dt))

    print("=" * 72)
    print(f"Saved {len(kept_ids)} object conditions -> {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} objects (see data/skipped in HDF5)")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PDM condition cache with affordance v6 predictions (not GT soft labels)",
    )
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--affordance-checkpoint", default=DEFAULT_AFF_CKPT)
    parser.add_argument("--dataset-dir", default=None, help="For affordance default threshold lookup")
    parser.add_argument("--mesh-root", default=DEFAULT_ROTATED_MESH_DIR)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-object seed derived from obj_id")
    parser.add_argument("--max-abs-coord", type=float, default=2.0)
    parser.add_argument("--target-max-extent", type=float, default=0.28)
    parser.add_argument("--auto-extent-lo", type=float, default=0.02)
    parser.add_argument("--auto-extent-hi", type=float, default=0.80)
    parser.add_argument("--min-scale-factor", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--obj", nargs="*", default=None, help="Optional subset of object ids")
    return parser


if __name__ == "__main__":
    build_cache(build_parser().parse_args())
