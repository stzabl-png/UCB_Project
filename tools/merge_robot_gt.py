#!/usr/bin/env python3
"""
merge_robot_gt.py — 合并多轮 / 多文件 robot_gt HDF5 中的成功抓取
================================================================
用法:
    python3 tools/merge_robot_gt.py --obj A01001 \\
        --inputs output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5 \\
                 output/grasp_collect/robot_gt/round_0001/A01001_robot_gt.hdf5 \\
        --output output/grasp_collect/merged/A01001_robot_gt_merged.hdf5
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def _load_successful(path: str) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        if "successful_grasps" not in f:
            return rows
        sg = f["successful_grasps"]
        for key in sorted(sg.keys()):
            g = sg[key]
            rows.append({
                "name": g.attrs.get("name", key),
                "score": float(g.attrs.get("score", 0)),
                "gripper_width": float(g.attrs.get("gripper_width", 0.04)),
                "approach_type": g.attrs.get("approach_type", ""),
                "grasp_point": np.array(g["grasp_point"][:], dtype=np.float64),
                "rotation": np.array(g["rotation"][:], dtype=np.float64),
                "source_file": os.path.abspath(path),
                "round_key": key,
            })
            if "contact_points_local" in g:
                rows[-1]["contact_points_local"] = np.array(
                    g["contact_points_local"][:], dtype=np.float64
                )
                rows[-1]["finger_width_actual"] = float(
                    g.attrs.get("finger_width_actual", 0.0)
                )
    return rows


def _is_duplicate(a: dict, b: dict, pos_tol=0.005, rot_tol_deg=12.0) -> bool:
    if np.linalg.norm(a["grasp_point"] - b["grasp_point"]) > pos_tol:
        return False
    Ra = Rotation.from_matrix(a["rotation"])
    Rb = Rotation.from_matrix(b["rotation"])
    angle = Ra.inv() * Rb
    return np.linalg.norm(angle.as_rotvec()) <= np.deg2rad(rot_tol_deg)


def merge_grasps(all_rows: list[dict], pos_tol=0.005, rot_tol_deg=12.0) -> list[dict]:
    merged: list[dict] = []
    for row in sorted(all_rows, key=lambda r: -r["score"]):
        if any(_is_duplicate(row, m, pos_tol, rot_tol_deg) for m in merged):
            continue
        merged.append(row)
    return merged


def write_merged(path: str, obj_id: str, grasps: list[dict], source_files: list[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as rf:
        rf.attrs["obj_id"] = obj_id
        rf.attrs["success"] = len(grasps) > 0
        rf.attrs["n_successful"] = len(grasps)
        rf.attrs["n_source_files"] = len(source_files)
        rf.attrs["source_files"] = np.array(source_files, dtype=h5py.string_dtype())

        sg = rf.create_group("successful_grasps")
        sg.attrs["count"] = len(grasps)
        for i, cr in enumerate(grasps):
            gi = sg.create_group(f"grasp_{i}")
            gi.attrs["name"] = cr["name"]
            gi.attrs["score"] = cr["score"]
            gi.attrs["gripper_width"] = cr["gripper_width"]
            gi.attrs["approach_type"] = cr.get("approach_type", "")
            gi.attrs["source_file"] = cr.get("source_file", "")
            gi.create_dataset("grasp_point", data=cr["grasp_point"])
            gi.create_dataset("rotation", data=cr["rotation"])
            gi.create_dataset("approach_dir", data=cr["rotation"][:, 2])
            gi.create_dataset("finger_dir", data=cr["rotation"][:, 0])
            if "contact_points_local" in cr:
                gi.create_dataset("contact_points_local", data=cr["contact_points_local"])
                gi.attrs["finger_width_actual"] = cr.get("finger_width_actual", 0.0)
                gi.attrs["has_contact_points"] = True
            else:
                gi.attrs["has_contact_points"] = False


def main():
    parser = argparse.ArgumentParser(description="Merge robot_gt successful grasps")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--inputs", nargs="+", required=True, help="robot_gt HDF5 paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--rot-tol-deg", type=float, default=12.0)
    args = parser.parse_args()

    all_rows: list[dict] = []
    used_files: list[str] = []
    for p in args.inputs:
        if not os.path.isfile(p):
            print(f"  skip missing: {p}")
            continue
        rows = _load_successful(p)
        all_rows.extend(rows)
        used_files.append(os.path.abspath(p))
        print(f"  + {len(rows)} from {p}")

    merged = merge_grasps(all_rows, args.pos_tol, args.rot_tol_deg)
    write_merged(args.output, args.obj, merged, used_files)
    print(f"✅ merged {len(merged)} unique successes → {args.output}")


if __name__ == "__main__":
    main()
