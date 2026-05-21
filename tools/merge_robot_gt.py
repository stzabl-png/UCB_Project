#!/usr/bin/env python3
"""
merge_robot_gt.py — 合并多轮 / 多文件 robot_gt HDF5 中的成功抓取
================================================================
只合并各轮 robot_gt 里 **successful_grasps**（Sim 成功）；**不**读 candidate_results
里的失败条目。若无任何成功，则不写 merged 文件（并删除已有空占位文件）。

可单独运行（不依赖 batch）。默认 **不去重**，保留各轮全部成功条目。

用法:
    python3 tools/merge_robot_gt.py --obj A01001 \\
        --inputs output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5 \\
                 output/grasp_collect/robot_gt/round_0001/A01001_robot_gt.hdf5 \\
        --output output/grasp_collect/merged/A01001_robot_gt_merged.hdf5

    # 可选：去掉位置/朝向相近的重复抓取
    python3 tools/merge_robot_gt.py --obj A01001 --inputs ... --output ... \\
        --deduplicate --pos-tol 0.005 --rot-tol-deg 12
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from mesh_utils import (
    identity_mesh_prerotation_record,
    infer_dataset,
    read_mesh_prerotation_hdf5,
    read_mesh_prerotation_hdf5_pose,
    write_mesh_prerotation_hdf5,
)

_EXECUTED_SNAPSHOTS = ("at_close", "post_lift")


def _read_executed_snapshot(g: h5py.Group) -> dict | None:
    if g is None or "position" not in g or "rotation" not in g:
        return None
    snap = {
        "position": np.array(g["position"][:], dtype=np.float32),
        "rotation": np.array(g["rotation"][:], dtype=np.float32),
        "approach_dir": np.array(g["approach_dir"][:], dtype=np.float32),
        "finger_dir": np.array(g["finger_dir"][:], dtype=np.float32),
    }
    for k in ("frame", "ee_frame", "snapshot"):
        if k in g.attrs:
            snap[k] = g.attrs[k]
    return snap


def _write_executed_snapshot(parent: h5py.Group, snap: dict | None, label: str) -> None:
    if snap is None:
        return
    g = parent.create_group(f"executed_panda_hand_{label}")
    g.attrs["frame"] = snap.get("frame", "object_mesh")
    g.attrs["ee_frame"] = snap.get("ee_frame", "panda_hand")
    g.attrs["snapshot"] = snap.get("snapshot", label)
    g.create_dataset("position", data=snap["position"])
    g.create_dataset("rotation", data=snap["rotation"])
    g.create_dataset("approach_dir", data=snap["approach_dir"])
    g.create_dataset("finger_dir", data=snap["finger_dir"])


def _load_successful(path: str) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        if "successful_grasps" not in f:
            return rows
        sg = f["successful_grasps"]
        for key in sorted(sg.keys()):
            g = sg[key]
            row = {
                "name": g.attrs.get("name", key),
                "score": float(g.attrs.get("score", 0)),
                "gripper_width": float(g.attrs.get("gripper_width", 0.04)),
                "approach_type": g.attrs.get("approach_type", ""),
                "grasp_point": np.array(g["grasp_point"][:], dtype=np.float64),
                "rotation": np.array(g["rotation"][:], dtype=np.float64),
                "source_file": os.path.abspath(path),
                "round_key": key,
            }
            pr = read_mesh_prerotation_hdf5_pose(g, f)
            if pr is not None:
                row["mesh_prerotation"] = pr
            rows.append(row)
            if "contact_points_local" in g:
                rows[-1]["contact_points_local"] = np.array(
                    g["contact_points_local"][:], dtype=np.float64
                )
                rows[-1]["finger_width_actual"] = float(
                    g.attrs.get("finger_width_actual", 0.0)
                )
            for label in _EXECUTED_SNAPSHOTS:
                sub = f"executed_panda_hand_{label}"
                if sub in g:
                    snap = _read_executed_snapshot(g[sub])
                    if snap is not None:
                        rows[-1][f"executed_{label}"] = snap
    return rows


def _is_duplicate(a: dict, b: dict, pos_tol=0.005, rot_tol_deg=12.0) -> bool:
    if np.linalg.norm(a["grasp_point"] - b["grasp_point"]) > pos_tol:
        return False
    Ra = Rotation.from_matrix(a["rotation"])
    Rb = Rotation.from_matrix(b["rotation"])
    angle = Ra.inv() * Rb
    return np.linalg.norm(angle.as_rotvec()) <= np.deg2rad(rot_tol_deg)


def merge_grasps(
    all_rows: list[dict],
    *,
    deduplicate: bool = False,
    pos_tol: float = 0.005,
    rot_tol_deg: float = 12.0,
) -> list[dict]:
    ordered = sorted(all_rows, key=lambda r: -r["score"])
    if not deduplicate:
        return ordered
    merged: list[dict] = []
    for row in ordered:
        if any(_is_duplicate(row, m, pos_tol, rot_tol_deg) for m in merged):
            continue
        merged.append(row)
    return merged


def _default_pose_prerotation(obj_id: str) -> dict:
    return identity_mesh_prerotation_record(obj_id, infer_dataset(obj_id))


def _count_successful_in_merged(path: str) -> int:
    with h5py.File(path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        sg = f["successful_grasps"]
        n_attr = int(sg.attrs.get("count", -1))
        n_keys = len(sg.keys())
        n_root = int(f.attrs.get("n_successful", -1))
        if n_attr >= 0:
            return n_attr
        if n_keys > 0:
            return n_keys
        if n_root >= 0:
            return n_root
    return 0


def cleanup_empty_merged_files(merged_dir: str) -> int:
    """删除 merged 目录下无成功条目的占位 HDF5。"""
    import glob

    removed = 0
    for path in glob.glob(os.path.join(merged_dir, "*_merged.hdf5")):
        if _count_successful_in_merged(path) == 0:
            os.remove(path)
            removed += 1
            print(f"  removed empty: {path}")
    return removed


def merge_robot_gt_files(
    obj_id: str,
    inputs: list[str],
    output: str,
    *,
    deduplicate: bool = False,
    pos_tol: float = 0.005,
    rot_tol_deg: float = 12.0,
    verbose: bool = True,
) -> str | None:
    """
    合并 inputs 中的成功抓取并写入 output。
    无成功时删除 output（若存在）并返回 None。
    """
    all_rows: list[dict] = []
    used_files: list[str] = []
    for p in inputs:
        if not os.path.isfile(p):
            if verbose:
                print(f"  skip missing: {p}")
            continue
        rows = _load_successful(p)
        all_rows.extend(rows)
        used_files.append(os.path.abspath(p))
        if verbose:
            print(f"  + {len(rows)} successes from {p}")

    merged = merge_grasps(
        all_rows,
        deduplicate=deduplicate,
        pos_tol=pos_tol,
        rot_tol_deg=rot_tol_deg,
    )
    if not merged:
        if os.path.isfile(output):
            os.remove(output)
            if verbose:
                print(f"  ⬛ no successful grasps; removed {output}")
        elif verbose:
            print(f"  ⬛ no successful grasps; skip writing {output}")
        return None

    write_merged(
        output, obj_id, merged, used_files,
        deduplicated=deduplicate,
        n_before=len(all_rows),
    )
    if verbose:
        tag = "deduped" if deduplicate else "all"
        print(
            f"✅ merged {len(merged)} successes ({tag}, from {len(all_rows)} raw) → {output}"
        )
    return output


def write_merged(
    path: str,
    obj_id: str,
    grasps: list[dict],
    source_files: list[str],
    *,
    deduplicated: bool = False,
    n_before: int = 0,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    default_pr = _default_pose_prerotation(obj_id)
    with h5py.File(path, "w") as rf:
        rf.attrs["obj_id"] = obj_id
        rf.attrs["success"] = len(grasps) > 0
        rf.attrs["n_successful"] = len(grasps)
        rf.attrs["deduplicated"] = bool(deduplicated)
        rf.attrs["n_before_merge"] = int(n_before)
        rf.attrs["n_source_files"] = len(source_files)
        rf.attrs["source_files"] = np.array(source_files, dtype=h5py.string_dtype())
        rf.attrs["robot_gt_schema_version"] = 2
        rf.attrs["executed_pose_frame"] = "object_mesh"
        rf.attrs["executed_ee_frame"] = "panda_hand"
        rf.attrs["includes_executed_pose"] = True

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
            pr = cr.get("mesh_prerotation") or default_pr
            write_mesh_prerotation_hdf5(gi, pr)
            for label in _EXECUTED_SNAPSHOTS:
                _write_executed_snapshot(gi, cr.get(f"executed_{label}"), label)


def main():
    parser = argparse.ArgumentParser(description="Merge robot_gt successful grasps")
    parser.add_argument("--obj")
    parser.add_argument("--inputs", nargs="+", help="robot_gt HDF5 paths")
    parser.add_argument("--output")
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="去掉相近 pose（默认不去重，保留各轮全部成功）",
    )
    parser.add_argument("--pos-tol", type=float, default=0.005,
                        help="--deduplicate 时抓取点距离阈值 (m)")
    parser.add_argument("--rot-tol-deg", type=float, default=12.0,
                        help="--deduplicate 时旋转差阈值 (度)")
    parser.add_argument(
        "--cleanup-empty-merged-dir",
        metavar="DIR",
        help="删除 DIR 下 successful_grasps 为空的 *_merged.hdf5，然后退出",
    )
    args = parser.parse_args()

    if args.cleanup_empty_merged_dir:
        removed = cleanup_empty_merged_files(args.cleanup_empty_merged_dir)
        print(f"removed {removed} empty merged file(s) under {args.cleanup_empty_merged_dir}")
        return

    if not args.obj or not args.inputs or not args.output:
        parser.error("--obj, --inputs, and --output are required unless using --cleanup-empty-merged-dir")

    merge_robot_gt_files(
        args.obj,
        args.inputs,
        args.output,
        deduplicate=args.deduplicate,
        pos_tol=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
    )


if __name__ == "__main__":
    main()
