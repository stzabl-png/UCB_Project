#!/usr/bin/env python3
"""
**WORK IN PROGRESS** — stable_orientations.json format and dedup rules may change;
placement batch depends on this artifact but the pipeline is not production-ready.

estimate_stable_orientations.py
================================
为每个物体估计多个可稳定摆放在桌面上的旋转（方案 A：离散库）。

mesh 基准与当前 grasp pipeline 一致：raw SAM3D mesh.ply + scale.json，不应用已有 rotation.json。

方法:
  - trimesh.compute_stable_poses() 返回多个凸包稳定姿态
  - 按概率排序，去重（一般夹角 + 仅差绕竖直 Z 的等价类）
  - 默认包含 identity（raw+scale 基线）
  - 每物体最多保留 --max-poses 个 stable（默认 8，不含 identity）

关于「绕 Z 仍稳定」:
  同一稳定摆放绕世界竖直轴旋转后通常仍稳定；trimesh 会对对称物体返回多个
  仅差 R_z 的解。本脚本用 relative_rotation ≈ R_z 合并，只保留该族中概率最高的一条；
  后续 batch 若需转台多样性，可在抽样时再 @ R_z(φ)。

输出:
  data_hub/ProcessedData/obj_meshes/{dataset}/{obj_id}/stable_orientations.json

用法:
  python3 data/estimate_stable_orientations.py --dataset oakink --workers 8
  python3 data/estimate_stable_orientations.py --obj A01026 --dataset oakink
  # 默认跳过已有 stable_orientations.json；覆盖才加 --force
  python3 data/estimate_stable_orientations.py --dataset oakink --workers 8 --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from estimate_obj_rotation import (  # noqa: E402
    list_dataset_objs,
    load_mesh_scaled,
    pca_z_rotation,
)

PROJ = os.path.dirname(_SCRIPT_DIR)
OBJ_MESHES = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes")
DEFAULT_REPORT = os.path.join(PROJ, "output", "stable_orientations_report.csv")

def obj_mesh_dir(obj_id: str, dataset: str) -> str:
    return os.path.join(OBJ_MESHES, dataset, obj_id)


def missing_required_inputs(obj_id: str, dataset: str) -> str | None:
    """
    检查 estimate 所需输入是否齐全。
    返回 None 表示可处理；否则返回跳过原因（用于日志）。
    """
    obj_dir = obj_mesh_dir(obj_id, dataset)
    mesh_path = os.path.join(obj_dir, "mesh.ply")
    scale_path = os.path.join(obj_dir, "scale.json")

    if not os.path.isfile(mesh_path):
        return "missing mesh.ply"
    if not os.path.isfile(scale_path):
        return "missing scale.json"
    try:
        with open(scale_path) as f:
            data = json.load(f)
        scale_factor = float(data["scale_factor"])
        if scale_factor <= 0:
            return f"invalid scale_factor={scale_factor}"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        return f"bad scale.json: {e}"
    return None


def list_ready_dataset_objs(dataset: str) -> list[str]:
    """list_dataset_objs 中仅保留 mesh.ply + 有效 scale.json 的物体。"""
    return sorted(
        o for o in list_dataset_objs(dataset)
        if missing_required_inputs(o, dataset) is None
    )


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


def is_z_rotation_only(R: np.ndarray, tol: float = 0.08) -> bool:
    """判断 R 是否近似为绕固定 Z 轴的旋转（竖直轴朝上时的转台对称）。"""
    R = np.asarray(R, dtype=np.float64)
    if abs(R[2, 2] - 1.0) > tol:
        return False
    if np.linalg.norm(R[:2, 2]) > tol or np.linalg.norm(R[2, :2]) > tol:
        return False
    det2 = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(det2 - 1.0) > 0.15:
        return False
    return True


def relative_is_z_only(R_a: np.ndarray, R_b: np.ndarray, z_tol: float = 0.08) -> bool:
    return is_z_rotation_only(R_a @ R_b.T, tol=z_tol)


def collect_trimesh_stable_poses(mesh) -> list[tuple[np.ndarray, float | None, int]]:
    """返回 [(R, prob, source_index), ...]，R 为 3×3，左乘顶点 v' = R @ v。"""
    try:
        transforms, probs = mesh.compute_stable_poses()
    except Exception:
        return []

    if transforms is None or len(transforms) == 0:
        return []

    out: list[tuple[np.ndarray, float | None, int]] = []
    for i, T in enumerate(transforms):
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            continue
        R = T[:3, :3].copy()
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.05:
            if det < 0:
                R = -R
            else:
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt
        prob = None
        if probs is not None and i < len(probs):
            prob = float(probs[i])
        out.append((R, prob, i))
    return out


def deduplicate_stable_poses(
    candidates: list[tuple[np.ndarray, float | None, int]],
    *,
    max_poses: int = 8,
    min_prob: float = 0.02,
    dedup_deg: float = 8.0,
    z_tol: float = 0.08,
) -> tuple[list[tuple[np.ndarray, float | None, int]], dict]:
    """
    按概率降序保留至多 max_poses 条；合并一般重复与「仅差绕 Z」的重复。
    """
    stats = {"raw": len(candidates), "kept": 0, "skip_prob": 0, "skip_dedup": 0, "skip_z": 0}
    filtered = [(R, p, i) for R, p, i in candidates if p is None or p >= min_prob]
    stats["skip_prob"] = stats["raw"] - len(filtered)
    filtered.sort(key=lambda x: (-(x[1] if x[1] is not None else -1.0), x[2]))

    kept: list[tuple[np.ndarray, float | None, int]] = []
    for R, prob, src_i in filtered:
        if len(kept) >= max_poses:
            break
        reject = False
        for Rk, _, _ in kept:
            if rotation_angle_deg(R, Rk) < dedup_deg:
                stats["skip_dedup"] += 1
                reject = True
                break
            if relative_is_z_only(R, Rk, z_tol=z_tol):
                stats["skip_z"] += 1
                reject = True
                break
        if not reject:
            kept.append((R, prob, src_i))

    stats["kept"] = len(kept)
    return kept, stats


def validate_rotated_mesh(mesh, R: np.ndarray) -> dict:
    """粗检：旋转后 z_min 与正交性。"""
    import trimesh

    m = mesh.copy()
    m.vertices = (R @ m.vertices.T).T
    zmin = float(m.vertices[:, 2].min())
    det = float(np.linalg.det(R))
    return {
        "z_min_m": round(zmin, 6),
        "det_R": round(det, 6),
        "bbox_extent_cm": [round(float(x) * 100, 2) for x in (m.bounds[1] - m.bounds[0])],
    }


def orientation_record(
    oid: int,
    R: np.ndarray,
    *,
    method: str,
    probability: float | None = None,
    rank: int | None = None,
    source_index: int | None = None,
    notes: str = "",
    validation: dict | None = None,
) -> dict:
    euler = [float(x) for x in Rotation.from_matrix(R).as_euler("xyz", degrees=True)]
    rec = {
        "id": oid,
        "euler_xyz_deg": euler,
        "matrix": R.astype(np.float64).tolist(),
        "method": method,
        "probability": probability,
        "rank": rank,
        "source_index": source_index,
        "notes": notes,
    }
    if validation is not None:
        rec["validation"] = validation
    return rec


def estimate_one(
    obj_id: str,
    dataset: str,
    *,
    force: bool = False,
    max_poses: int = 8,
    min_prob: float = 0.02,
    dedup_deg: float = 8.0,
    z_tol: float = 0.08,
    include_identity: bool = True,
) -> dict | None:
    out_path = os.path.join(OBJ_MESHES, dataset, obj_id, "stable_orientations.json")
    if os.path.exists(out_path) and not force:
        existing = json.load(open(out_path))
        print(
            f"  ⏭  {obj_id}: 已存在 n={existing.get('n_orientations', '?')} "
            f"(primary_id={existing.get('primary_id')})"
        )
        return existing

    skip_reason = missing_required_inputs(obj_id, dataset)
    if skip_reason:
        print(f"  ⏭  {obj_id}: 跳过 ({skip_reason})")
        return None

    mesh, scale = load_mesh_scaled(obj_id, dataset)
    if mesh is None:
        print(f"  ⚠️  {obj_id}: mesh 加载失败")
        return None

    orientations: list[dict] = []
    warnings: list[str] = []
    next_id = 0

    if include_identity:
        R_i = np.eye(3, dtype=np.float64)
        orientations.append(
            orientation_record(
                next_id,
                R_i,
                method="identity",
                probability=None,
                rank=None,
                notes="raw_sam3d_scaled baseline (no_rotation pipeline)",
                validation=validate_rotated_mesh(mesh, R_i),
            )
        )
        next_id += 1

    raw_candidates = collect_trimesh_stable_poses(mesh)
    kept, dedup_stats = deduplicate_stable_poses(
        raw_candidates,
        max_poses=max_poses,
        min_prob=min_prob,
        dedup_deg=dedup_deg,
        z_tol=z_tol,
    )

    if not kept:
        warnings.append("no_trimesh_stable_pose")
        try:
            R_fb = pca_z_rotation(mesh.vertices)
            kept = [(R_fb, None, -1)]
            warnings.append("fallback_pca_z")
        except Exception as e:
            warnings.append(f"pca_z_failed:{e}")

    primary_id = 0 if include_identity else None
    stable_rank = 0
    for R, prob, src_i in kept:
        stable_rank += 1
        if primary_id is None or (prob is not None and stable_rank == 1):
            # 第一条 stable（概率最高）作为 primary，与旧 rotation.json 语义对齐
            primary_id = next_id
        rec = orientation_record(
            next_id,
            R,
            method="trimesh_stable_pose" if src_i >= 0 else "pca_z_fallback",
            probability=prob,
            rank=stable_rank,
            source_index=src_i if src_i >= 0 else None,
            validation=validate_rotated_mesh(mesh, R),
        )
        orientations.append(rec)
        next_id += 1

    if primary_id is None and orientations:
        primary_id = orientations[0]["id"]

    # 与旧 rotation.json 对比
    legacy = None
    rot_json = os.path.join(OBJ_MESHES, dataset, obj_id, "rotation.json")
    if os.path.exists(rot_json):
        legacy = json.load(open(rot_json))
        if primary_id is not None and legacy.get("euler_xyz_deg"):
            R_old = Rotation.from_euler(
                "xyz", legacy["euler_xyz_deg"], degrees=True
            ).as_matrix()
            R_pri = np.array(orientations[primary_id]["matrix"])
            legacy["angle_to_primary_deg"] = round(rotation_angle_deg(R_pri, R_old), 2)

    doc = {
        "obj": obj_id,
        "dataset": dataset,
        "mesh_frame": "raw_sam3d_scaled",
        "scale_factor": float(scale),
        "version": 1,
        "orientations": orientations,
        "n_orientations": len(orientations),
        "primary_id": primary_id,
        "dedup_threshold_deg": float(dedup_deg),
        "z_equivalence_tol": float(z_tol),
        "min_prob": float(min_prob),
        "max_stable_poses": int(max_poses),
        "include_identity": bool(include_identity),
        "dedup_stats": dedup_stats,
        "warnings": warnings,
        "legacy_rotation_json": legacy,
        "generated_by": "estimate_stable_orientations.py",
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    n_stable = len(orientations) - (1 if include_identity else 0)
    print(
        f"  ✅ {obj_id}: {len(orientations)} orientations "
        f"(identity={'yes' if include_identity else 'no'}, stable={n_stable}, "
        f"raw_trimesh={dedup_stats['raw']}, skip_z={dedup_stats['skip_z']}, "
        f"skip_dedup={dedup_stats['skip_dedup']}) -> primary_id={primary_id}"
    )
    return doc


def _estimate_job(task: tuple[str, str, dict]) -> dict | None:
    """ProcessPool 入口: (obj_id, dataset, kwargs) -> doc。"""
    obj_id, dataset, kwargs = task
    return estimate_one(obj_id, dataset, **kwargs)


def _run_dataset_parallel(
    obj_ids: list[str],
    dataset: str,
    kwargs: dict,
    workers: int,
) -> list[dict]:
    """并行估计；单 worker 时顺序执行便于调试。"""
    tasks = [(oid, dataset, kwargs) for oid in obj_ids]
    results: list[dict] = []
    n = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{n}]", end=" ")
            doc = _estimate_job(task)
            if doc is not None:
                results.append(doc)
        return results

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_estimate_job, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            obj_id = futures[fut]
            try:
                doc = fut.result()
            except Exception as e:
                print(f"  ❌ {obj_id}: {e}")
                continue
            if doc is not None:
                results.append(doc)
            if done % 10 == 0 or done == n:
                print(f"  … {done}/{n}  ({time.time() - t0:.0f}s)")
    print(f"  并行完成 {len(results)}/{n} 用时 {time.time() - t0:.0f}s")
    return results


def write_report(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="估计每物体多个稳定摆放 rotation (stable_orientations.json)"
    )
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument("--dataset", help="数据集: oakink / dexycb / …")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="覆盖已有 stable_orientations.json")
    parser.add_argument("--max-poses", type=int, default=8, help="每物体最多 stable 条数（不含 identity）")
    parser.add_argument("--min-prob", type=float, default=0.02)
    parser.add_argument("--dedup-deg", type=float, default=8.0, help="一般旋转去重角度阈值")
    parser.add_argument("--z-tol", type=float, default=0.08, help="判定「仅差绕 Z」的容差")
    parser.add_argument(
        "--no-identity", action="store_true", help="不写入 identity 基线条目",
    )
    parser.add_argument(
        "--report", default=DEFAULT_REPORT, help="汇总 CSV 路径",
    )
    default_workers = min(8, os.cpu_count() or 4)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"批量并行进程数 (默认 {default_workers}；单物体忽略)",
    )
    args = parser.parse_args()

    include_identity = not args.no_identity
    datasets = ["oakink", "dexycb"] if args.all else ([args.dataset] if args.dataset else [])

    if args.obj:
        for ds in ["oakink", "ycb", "arctic", "dexycb"]:
            if not os.path.isdir(os.path.join(OBJ_MESHES, ds, args.obj)):
                continue
            doc = estimate_one(
                args.obj,
                ds,
                force=args.force,
                max_poses=args.max_poses,
                min_prob=args.min_prob,
                dedup_deg=args.dedup_deg,
                z_tol=args.z_tol,
                include_identity=include_identity,
            )
            if doc and args.report:
                write_report(
                    [{
                        "obj_id": args.obj,
                        "dataset": ds,
                        "n_orientations": doc["n_orientations"],
                        "n_stable": doc["n_orientations"] - (1 if include_identity else 0),
                        "primary_id": doc["primary_id"],
                        "skip_z": doc["dedup_stats"]["skip_z"],
                        "warnings": ";".join(doc["warnings"]),
                    }],
                    args.report,
                )
            return
        print(f"❌ 未找到 {args.obj}")
        return

    if not datasets:
        parser.print_help()
        return

    estimate_kwargs = dict(
        force=args.force,
        max_poses=args.max_poses,
        min_prob=args.min_prob,
        dedup_deg=args.dedup_deg,
        z_tol=args.z_tol,
        include_identity=include_identity,
    )

    report_rows: list[dict] = []
    total = ok = 0
    for ds in datasets:
        all_ids = list_dataset_objs(ds)
        obj_ids = list_ready_dataset_objs(ds)
        skipped = len(all_ids) - len(obj_ids)
        skip_msg = f", 跳过 {skipped} 缺输入" if skipped else ""
        force_msg = "force=on" if args.force else "force=off (跳过已有)"
        print(
            f"\n=== {ds} ({len(obj_ids)} 可处理{skip_msg}) "
            f"workers={args.workers} {force_msg} "
            f"max_stable={args.max_poses} identity={include_identity} ==="
        )
        total += len(obj_ids)
        docs = _run_dataset_parallel(obj_ids, ds, estimate_kwargs, args.workers)
        for doc in docs:
            ok += 1
            report_rows.append({
                "obj_id": doc["obj"],
                "dataset": ds,
                "n_orientations": doc["n_orientations"],
                "n_stable": doc["n_orientations"] - (1 if include_identity else 0),
                "primary_id": doc["primary_id"],
                "raw_trimesh": doc["dedup_stats"]["raw"],
                "skip_z": doc["dedup_stats"]["skip_z"],
                "skip_dedup": doc["dedup_stats"]["skip_dedup"],
                "warnings": ";".join(doc["warnings"]),
            })

    if report_rows and args.report:
        write_report(report_rows, args.report)
        print(f"\n📄 report → {args.report}")
    print(f"\n完成: {ok}/{total}")


if __name__ == "__main__":
    main()
