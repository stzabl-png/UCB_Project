#!/usr/bin/env python3
"""
Export Gaussian soft affordance GT maps for affordance HDF5 datasets.

Uses the same heatmap logic as model/affordance/dataset.py (no augmentation).

Writes:
  <dataset_dir>/affordance_train_soft.h5
  <dataset_dir>/affordance_val_soft.h5
  <dataset_dir>/soft_gt_export_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import h5py
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.affordance.heatmap import (
    gaussian_soft_heatmap,
    object_bbox_diagonal,
    sigma_from_scale,
)


def _decode_obj_ids(raw) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def compute_soft_labels_for_file(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (soft_labels, per_sample_sigma) for all samples.
    soft_labels: (N, P) float32 in [0, 1]
    per_sample_sigma: (N,) meters
    """
    n = points.shape[0]
    soft = np.zeros_like(labels, dtype=np.float32)
    sigmas = np.zeros(n, dtype=np.float32)
    binary = labels > label_threshold

    for i in range(n):
        pts = points[i]
        obj_scale = object_bbox_diagonal(pts)
        sigma = sigma_from_scale(obj_scale, heatmap_sigma_ratio)
        sigmas[i] = sigma
        soft[i] = gaussian_soft_heatmap(pts, binary[i], sigma)
    return soft, sigmas


def export_split(
    src_path: str,
    dst_path: str,
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float,
    overwrite: bool,
) -> dict:
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"{dst_path} exists (use --overwrite)")

    with h5py.File(src_path, "r") as f:
        points = f["data/points"][:]
        labels = f["data/labels"][:]
        obj_ids = f["data/obj_ids"][:]
        extra_keys = []
        for key in ("normals", "human_priors", "force_centers", "categories", "intents"):
            path = f"data/{key}"
            if path in f:
                extra_keys.append(key)
        extra_data = {k: f[f"data/{k}"][:] for k in extra_keys}
        src_meta = dict(f["metadata"].attrs) if "metadata" in f else {}

    soft, sigmas = compute_soft_labels_for_file(
        points,
        labels,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
        label_threshold=label_threshold,
    )

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with h5py.File(dst_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["source_h5"] = os.path.abspath(src_path)
        meta.attrs["exported_at"] = datetime.now().isoformat(timespec="seconds")
        meta.attrs["heatmap_sigma_ratio"] = float(heatmap_sigma_ratio)
        meta.attrs["label_threshold"] = float(label_threshold)
        meta.attrs["object_scale"] = "bbox_diagonal(points)"
        meta.attrs["sigma_formula"] = "sigma = heatmap_sigma_ratio * object_scale"
        meta.attrs["soft_formula"] = (
            "soft[i]=exp(-d_i^2/(2*sigma^2)), d_i=min dist to contact; contact points=1.0"
        )
        meta.attrs["num_samples"] = int(points.shape[0])
        meta.attrs["num_points"] = int(points.shape[1])
        for k, v in src_meta.items():
            try:
                meta.attrs[f"source_{k}"] = v
            except TypeError:
                meta.attrs[f"source_{k}"] = str(v)

        grp = f.create_group("data")
        grp.create_dataset(
            "points", data=points, compression="gzip", compression_opts=4,
        )
        grp.create_dataset(
            "labels", data=labels, compression="gzip", compression_opts=4,
        )
        grp.create_dataset(
            "soft_labels", data=soft, compression="gzip", compression_opts=4,
        )
        grp.create_dataset(
            "soft_sigma", data=sigmas, compression="gzip", compression_opts=4,
        )
        grp.create_dataset("obj_ids", data=obj_ids)
        for k, arr in extra_data.items():
            grp.create_dataset(k, data=arr, compression="gzip", compression_opts=4)

    decoded = _decode_obj_ids(obj_ids)
    return {
        "source": os.path.basename(src_path),
        "output": os.path.basename(dst_path),
        "num_samples": int(points.shape[0]),
        "heatmap_sigma_ratio": heatmap_sigma_ratio,
        "sigma_m_min": float(sigmas.min()),
        "sigma_m_max": float(sigmas.max()),
        "sigma_m_mean": float(sigmas.mean()),
        "soft_max_mean": float(soft.max(axis=1).mean()),
        "soft_mean_mean": float(soft.mean(axis=1).mean()),
        "contact_frac_mean": float((labels > label_threshold).mean(axis=1).mean()),
        "objects": sorted(set(decoded)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Export soft affordance GT maps to HDF5")
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(PROJ, "output", "affordance_no_rot_executed"),
        help="Directory with affordance_train.h5 / affordance_val.h5",
    )
    p.add_argument(
        "--heatmap-sigma-ratio",
        type=float,
        default=0.03,
        help="σ = ratio × bbox_diagonal(points); training default often 0.03",
    )
    p.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Binary contact mask: labels > threshold",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    splits = [
        ("affordance_train.h5", "affordance_train_soft.h5"),
        ("affordance_val.h5", "affordance_val_soft.h5"),
    ]

    report = {
        "dataset_dir": dataset_dir,
        "heatmap_sigma_ratio": args.heatmap_sigma_ratio,
        "label_threshold": args.label_threshold,
        "note": (
            "Exported on canonical H5 points (no train-time augmentation). "
            "Training recomputes the same formula on augmented points each step."
        ),
        "splits": [],
    }

    for src_name, dst_name in splits:
        src = os.path.join(dataset_dir, src_name)
        dst = os.path.join(dataset_dir, dst_name)
        if not os.path.isfile(src):
            print(f"skip (missing): {src}")
            continue
        info = export_split(
            src,
            dst,
            heatmap_sigma_ratio=args.heatmap_sigma_ratio,
            label_threshold=args.label_threshold,
            overwrite=args.overwrite,
        )
        report["splits"].append(info)
        print(
            f"Wrote {dst_name}: {info['num_samples']} samples, "
            f"σ∈[{info['sigma_m_min']:.4f}, {info['sigma_m_max']:.4f}] m"
        )

    meta_path = os.path.join(dataset_dir, "soft_gt_export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
