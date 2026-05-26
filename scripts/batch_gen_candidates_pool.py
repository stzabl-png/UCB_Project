#!/usr/bin/env python3
"""
batch_gen_candidates_pool.py — 为 merged 成功数低于阈值的物体批量生成 candidate 池
==================================================================================
仅 Phase 1（random_grasp_sampler），不跑 sim。

物体筛选:
  - 只考虑 merged 目录里 **存在** `{obj}_robot_gt_merged.hdf5` 的文件
  - 无 merged 文件的物体 **不参与**（不视为 success=0）
  - merged 内 successful 条数 < --success-threshold 才生成

输出（默认）:
  output/grasp_collect_no_rot/candidates/pool/{obj_id}_grasp.hdf5

用法:
    python3 scripts/batch_gen_candidates_pool.py \\
        --success-threshold 20 \\
        --target 50 \\
        --sampler-workers 8

    # 锚定 merged 成功 pose（需 --min-merged-success 1）
    python3 scripts/batch_gen_candidates_pool.py \\
        --gen-mode anchored \\
        --min-merged-success 1 \\
        --success-threshold 20 --target 50

    # ~50% anchored + ~50% raycast（增加 exploration）
    python3 scripts/batch_gen_candidates_pool.py \\
        --gen-mode mixed --min-merged-success 1 \\
        --success-threshold 20 --target 50

    python3 scripts/batch_gen_candidates_pool.py \\
        --merged-dir output/grasp_collect_no_rot/merged \\
        --output-dir output/grasp_collect_no_rot/candidates/pool \\
        --success-threshold 20 --target 50 --resume
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))

DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
DEFAULT_POOL_DIR = os.path.join(
    PROJ, "output", "grasp_collect_no_rot", "candidates", "pool",
)
SAMPLER = os.path.join(PROJ, "tools", "random_grasp_sampler.py")
MANIFEST_NAME = "gen_pool_manifest.json"


@dataclass
class PoolJobConfig:
    output_dir: str
    target: int
    score_threshold: float
    no_rotation: bool
    python_bin: str
    resume: bool
    force: bool
    convert_usd: bool
    gen_mode: str = "raycast"  # raycast | anchored | mixed
    merged_dir: str = ""
    anchor_seed: int | None = None
    anchor_max_rot_deg: float = 15.0
    anchor_max_tip_jitter_mm: float = 8.0
    anchor_max_approach_jitter_mm: float = 1.0
    anchor_max_retry_per_slot: int | None = None
    write_anchor_meta: bool = False
    keep_gen_logs: bool = False


@dataclass
class PoolGenResult:
    obj_id: str
    dataset: str
    merged_success: int
    status: str
    n_candidates: int = 0
    grasp_hdf5: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    score_threshold: float = 0.0
    n_above_threshold: int = 0
    score_min: Optional[float] = None
    score_max: Optional[float] = None


def _summarize_grasp_hdf5(grasp_path: str, score_threshold: float) -> dict:
    """Read per-candidate scores from pool HDF5."""
    import h5py

    scores: list[float] = []
    with h5py.File(grasp_path, "r") as f:
        cg = f["candidates"]
        for key in cg.keys():
            scores.append(float(cg[key].attrs.get("score", 0.0)))
    if not scores:
        return {
            "n_candidates": 0,
            "n_above_threshold": 0,
            "score_min": None,
            "score_max": None,
        }
    return {
        "n_candidates": len(scores),
        "n_above_threshold": sum(1 for s in scores if s >= score_threshold),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
    }


def _format_pool_gen_line(res: dict) -> str:
    oid = res["obj_id"]
    st = res["status"]
    elapsed = res.get("elapsed_s", 0)
    n = int(res.get("n_candidates", 0))
    if st == "ok" and n > 0:
        thr = res.get("score_threshold", 0)
        na = int(res.get("n_above_threshold", 0))
        smin = res.get("score_min")
        smax = res.get("score_max")
        if smin is not None and smax is not None:
            return (
                f"  {oid}: ok  n={n}  ≥{thr:g}:{na}  "
                f"score {smax:.1f}~{smin:.1f}  {elapsed}s"
            )
        return f"  {oid}: ok  n={n}  {elapsed}s"
    if st == "skip":
        return f"  {oid}: skip  n={n}  {elapsed}s"
    err = res.get("error", "")
    if err:
        return f"  {oid}: {st}  n={n}  {elapsed}s  ({err[:80]})"
    return f"  {oid}: {st}  n={n}  {elapsed}s"


def _run_cmd(cmd: list[str], log_path: Optional[str] = None) -> tuple[int, str]:
    log_f = open(log_path, "w") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJ,
            stdout=log_f if log_path else subprocess.PIPE,
            stderr=subprocess.STDOUT if log_path else subprocess.PIPE,
            text=True,
        )
        out = ""
        if not log_path and proc.stdout:
            out = proc.stdout
        return proc.returncode, out
    finally:
        if log_path and log_f and not log_f.closed:
            log_f.close()


def _count_merged_success(merged_path: str) -> int:
    from merge_robot_gt import _count_successful_in_merged

    return _count_successful_in_merged(merged_path)


def scan_merged_objects(merged_dir: str) -> dict[str, int]:
    from grasp_pool_common import scan_merged_objects as _scan

    return _scan(merged_dir)


def resolve_dataset(obj_id: str) -> Optional[str]:
    """Return sampler --dataset (oakink / ycb) if object is in rotated_mesh pipeline."""
    from random_grasp_sampler import list_dataset_objs

    for ds in ("oakink", "ycb"):
        list_ds = "dexycb" if ds == "ycb" else ds
        if obj_id in list_dataset_objs(list_ds, use_legacy_assets=False):
            return ds
    return None


def select_target_objects(
    merged_dir: str,
    success_threshold: int,
    min_merged_success: int = 0,
) -> list[tuple[str, str, int]]:
    """
    Objects with merged file, success >= min_merged_success, and success < success_threshold.
    Returns [(obj_id, dataset, merged_success), ...] sorted by success then obj_id.
    """
    merged_counts = scan_merged_objects(merged_dir)
    selected: list[tuple[str, str, int]] = []
    skipped_no_dataset: list[str] = []

    for obj_id, success in merged_counts.items():
        if success < min_merged_success:
            continue
        if success >= success_threshold:
            continue
        dataset = resolve_dataset(obj_id)
        if dataset is None:
            skipped_no_dataset.append(obj_id)
            continue
        selected.append((obj_id, dataset, success))

    selected.sort(key=lambda x: (x[2], x[0]))
    if skipped_no_dataset:
        print(
            f"⚠️  {len(skipped_no_dataset)} merged object(s) not in sampler assets "
            f"(skipped): {skipped_no_dataset[:5]}"
            + (" ..." if len(skipped_no_dataset) > 5 else "")
        )
    return selected


def _append_anchor_sampler_args(
    gen_cmd: list[str],
    cfg: PoolJobConfig,
    obj_id: str,
    *,
    sampling_mode: str,
) -> None:
    merged_path = os.path.join(cfg.merged_dir, f"{obj_id}_robot_gt_merged.hdf5")
    gen_cmd.extend([
        "--sampling-mode", sampling_mode,
        "--anchor-merged-path", merged_path,
        "--anchor-max-rot-deg", str(cfg.anchor_max_rot_deg),
        "--anchor-max-tip-jitter-mm", str(cfg.anchor_max_tip_jitter_mm),
        "--anchor-max-approach-jitter-mm", str(cfg.anchor_max_approach_jitter_mm),
    ])
    if cfg.anchor_seed is not None:
        gen_cmd.extend(["--anchor-seed", str(cfg.anchor_seed)])
    if cfg.anchor_max_retry_per_slot is not None:
        gen_cmd.extend([
            "--anchor-max-retry-per-slot",
            str(cfg.anchor_max_retry_per_slot),
        ])
    if cfg.write_anchor_meta:
        gen_cmd.append("--write-anchor-meta")


def run_pool_gen_job(
    obj_id: str,
    dataset: str,
    merged_success: int,
    cfg_dict: dict,
) -> dict:
    cfg = PoolJobConfig(**cfg_dict)
    t0 = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_dir = os.path.join(cfg.output_dir, "logs") if cfg.keep_gen_logs else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(cfg.output_dir, f"{obj_id}_grasp.hdf5")
    result = PoolGenResult(
        obj_id=obj_id,
        dataset=dataset,
        merged_success=merged_success,
        status="pending",
        grasp_hdf5=out_path,
    )

    try:
        if os.path.isfile(out_path):
            if cfg.resume and not cfg.force:
                stats = _summarize_grasp_hdf5(out_path, cfg.score_threshold)
                result.n_candidates = stats["n_candidates"]
                result.n_above_threshold = stats["n_above_threshold"]
                result.score_min = stats["score_min"]
                result.score_max = stats["score_max"]
                result.score_threshold = cfg.score_threshold
                result.status = "skip"
                return asdict(result)
            if not cfg.force:
                result.status = "exists"
                result.error = "output exists (use --force or --resume)"
                return asdict(result)

        if cfg.convert_usd:
            convert = os.path.join(PROJ, "tools", "convert_obj_usd.py")
            conv_cmd = [
                cfg.python_bin, convert,
                "--obj", obj_id, "--dataset", dataset, "--force",
            ]
            if cfg.no_rotation:
                conv_cmd.append("--no-rotation")
            rc, out = _run_cmd(
                conv_cmd,
                log_path=os.path.join(log_dir, f"{obj_id}_convert.log") if log_dir else None,
            )
            if rc != 0:
                result.status = "convert_failed"
                result.error = (out or "convert failed")[:500]
                return asdict(result)

        gen_cmd = [
            cfg.python_bin, SAMPLER,
            "--obj", obj_id,
            "--dataset", dataset,
            "--force",
            "--output-dir", cfg.output_dir,
            "--target", str(cfg.target),
            "--score-threshold", str(cfg.score_threshold),
        ]
        if cfg.no_rotation:
            gen_cmd.append("--no-rotation")
        if cfg.gen_mode == "anchored":
            _append_anchor_sampler_args(
                gen_cmd, cfg, obj_id, sampling_mode="anchored_contact_v2",
            )
        elif cfg.gen_mode == "mixed":
            _append_anchor_sampler_args(
                gen_cmd, cfg, obj_id, sampling_mode="mixed_anchored_raycast_v1",
            )
        elif cfg.gen_mode != "raycast":
            result.status = "error"
            result.error = f"unknown gen_mode: {cfg.gen_mode}"
            return asdict(result)

        rc, out = _run_cmd(
            gen_cmd,
            log_path=os.path.join(log_dir, f"{obj_id}_gen.log") if log_dir else None,
        )
        if rc != 0:
            result.status = "gen_failed"
            result.error = (out or "sampler nonzero exit")[:500]
            return asdict(result)

        skip_path = os.path.join(cfg.output_dir, f"{obj_id}.skip")
        if not os.path.isfile(out_path):
            if os.path.isfile(skip_path):
                with open(skip_path) as sf:
                    result.error = sf.read().strip()[:500]
                result.status = "no_candidates"
            else:
                result.status = "no_output"
            return asdict(result)

        stats = _summarize_grasp_hdf5(out_path, cfg.score_threshold)
        result.n_candidates = stats["n_candidates"]
        result.n_above_threshold = stats["n_above_threshold"]
        result.score_min = stats["score_min"]
        result.score_max = stats["score_max"]
        result.score_threshold = cfg.score_threshold
        result.status = "ok" if result.n_candidates > 0 else "no_candidates"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
    finally:
        result.elapsed_s = round(time.time() - t0, 1)

    return asdict(result)


def _save_manifest(
    path: str,
    *,
    merged_dir: str,
    output_dir: str,
    merged_success_threshold: int,
    score_threshold: float,
    target: int,
    results: list[dict],
    gen_mode: str = "raycast",
    min_merged_success: int = 0,
):
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "merged_dir": os.path.abspath(merged_dir),
        "output_dir": os.path.abspath(output_dir),
        "gen_mode": gen_mode,
        "min_merged_success": min_merged_success,
        "merged_success_threshold": merged_success_threshold,
        "score_threshold": score_threshold,
        "target_per_object": target,
        "n_objects": len(results),
        "objects": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate pool for low-success merged objects",
    )
    parser.add_argument(
        "--merged-dir",
        default=DEFAULT_MERGED_DIR,
        help=f"merged robot_gt 目录 (default: {DEFAULT_MERGED_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_POOL_DIR,
        help=f"candidate 池输出目录 (default: {DEFAULT_POOL_DIR})",
    )
    parser.add_argument(
        "--success-threshold",
        type=int,
        required=True,
        help="只生成 merged 成功数 **严格小于** 该值的物体",
    )
    parser.add_argument(
        "--min-merged-success",
        type=int,
        default=0,
        help="只生成 merged 成功数 **>=** 该值的物体（默认 0 不限制下限）",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="每个目标物体生成的 candidate 数 (random_grasp_sampler --target)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=70.0,
        help="sampler 分数门槛",
    )
    parser.add_argument(
        "--sampler-workers",
        type=int,
        default=16,
        help="并行 sampler 进程数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 pool 里已有 {obj}_grasp.hdf5",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有 pool HDF5",
    )
    parser.add_argument(
        "--convert-usd",
        action="store_true",
        help="生成前 convert_obj_usd（默认跳过）",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="应用 rotation.json（默认 no-rotation）",
    )
    parser.add_argument(
        "--gen-mode",
        choices=("raycast", "anchored", "mixed"),
        default="raycast",
        help=(
            "raycast: 默认 random_grasp_sampler；"
            "anchored: 锚定 merged 成功 pose；"
            "mixed: ~50%% anchored + ~50%% raycast"
        ),
    )
    parser.add_argument(
        "--anchor-seed",
        type=int,
        default=None,
        help="anchored/mixed 扰动随机种子；默认不固定，每次调用使用系统熵",
    )
    parser.add_argument("--anchor-max-rot-deg", type=float, default=15.0)
    parser.add_argument("--anchor-max-tip-jitter-mm", type=float, default=8.0)
    parser.add_argument(
        "--anchor-max-approach-jitter-mm",
        type=float,
        default=1.0,
        help="沿 approach 方向扰动 grasp_point 的最大距离；默认 1mm",
    )
    parser.add_argument(
        "--anchor-max-retry-per-slot",
        type=int,
        default=None,
        help="覆盖每工位 retry；默认 (100×--target)/工位数",
    )
    parser.add_argument(
        "--write-anchor-meta",
        action="store_true",
        help="写入 {obj}_grasp_anchor_meta.json（默认不写；sim 不读）",
    )
    parser.add_argument(
        "--keep-gen-logs",
        action="store_true",
        help="在 pool 下写 logs/{obj}_gen.log（默认不写）",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--obj",
        help="只处理单个物体（须在 merged 中且 success < threshold）",
    )
    args = parser.parse_args()

    if args.success_threshold < 0:
        print("❌ --success-threshold 须 >= 0")
        sys.exit(1)
    if args.min_merged_success < 0:
        print("❌ --min-merged-success 须 >= 0")
        sys.exit(1)
    if args.min_merged_success >= args.success_threshold:
        print("❌ --min-merged-success 须 < --success-threshold")
        sys.exit(1)
    if args.target < 1:
        print("❌ --target 须 >= 1")
        sys.exit(1)
    if args.gen_mode in ("anchored", "mixed") and args.min_merged_success < 1:
        print(
            f"⚠️  --gen-mode {args.gen_mode} 建议 --min-merged-success 1；"
            "已自动设为 1",
        )
        args.min_merged_success = 1
    if not os.path.isdir(args.merged_dir):
        print(f"❌ merged dir not found: {args.merged_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    targets = select_target_objects(
        args.merged_dir,
        args.success_threshold,
        min_merged_success=args.min_merged_success,
    )
    if args.obj:
        targets = [t for t in targets if t[0] == args.obj]
        if not targets:
            merged_counts = scan_merged_objects(args.merged_dir)
            if args.obj not in merged_counts:
                print(f"❌ {args.obj}: no merged file in {args.merged_dir}")
            elif merged_counts[args.obj] < args.min_merged_success:
                print(
                    f"❌ {args.obj}: merged success={merged_counts[args.obj]} "
                    f"< min {args.min_merged_success}"
                )
            elif merged_counts[args.obj] >= args.success_threshold:
                print(
                    f"❌ {args.obj}: merged success={merged_counts[args.obj]} "
                    f">= threshold {args.success_threshold}"
                )
            else:
                print(f"❌ {args.obj}: not in sampler assets")
            sys.exit(1)

    if not targets:
        print(
            f"❌ no objects with {args.min_merged_success} <= merged success "
            f"< {args.success_threshold} (merged dir: {args.merged_dir})"
        )
        sys.exit(1)

    cfg = PoolJobConfig(
        output_dir=os.path.abspath(args.output_dir),
        target=args.target,
        score_threshold=args.score_threshold,
        no_rotation=not args.rotation,
        python_bin=args.python_bin,
        resume=args.resume,
        force=args.force,
        convert_usd=args.convert_usd,
        gen_mode=args.gen_mode,
        merged_dir=os.path.abspath(args.merged_dir),
        anchor_seed=args.anchor_seed,
        anchor_max_rot_deg=args.anchor_max_rot_deg,
        anchor_max_tip_jitter_mm=args.anchor_max_tip_jitter_mm,
        anchor_max_approach_jitter_mm=args.anchor_max_approach_jitter_mm,
        anchor_max_retry_per_slot=args.anchor_max_retry_per_slot,
        write_anchor_meta=args.write_anchor_meta,
        keep_gen_logs=args.keep_gen_logs,
    )
    cfg_dict = asdict(cfg)

    print(f"Merged:  {args.merged_dir}")
    print(f"Pool out: {args.output_dir}")
    print(
        f"Filter: {args.min_merged_success} <= merged success < {args.success_threshold}"
    )
    print(f"Target: {args.target} candidates / object")
    print(f"Gen mode: {args.gen_mode}")
    print(f"Objects: {len(targets)}")
    print(f"Workers: {args.sampler_workers}")
    print(f"Resume: {args.resume}  Force: {args.force}")
    print("-" * 60)
    for obj_id, dataset, success in targets[:10]:
        print(f"  {obj_id} ({dataset})  merged_success={success}")
    if len(targets) > 10:
        print(f"  ... and {len(targets) - 10} more")

    results: list[dict] = []
    workers = max(1, args.sampler_workers)
    t0 = time.time()

    if workers == 1:
        for obj_id, dataset, success in targets:
            print(f"\n>> {obj_id} (merged={success})")
            res = run_pool_gen_job(obj_id, dataset, success, cfg_dict)
            results.append(res)
            print(_format_pool_gen_line(res))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(run_pool_gen_job, oid, ds, sc, cfg_dict): oid
                for oid, ds, sc in targets
            }
            for fut in as_completed(futures):
                oid = futures[fut]
                res = fut.result()
                results.append(res)
                print(_format_pool_gen_line(res))

    manifest_path = os.path.join(args.output_dir, MANIFEST_NAME)
    _save_manifest(
        manifest_path,
        merged_dir=args.merged_dir,
        output_dir=args.output_dir,
        merged_success_threshold=args.success_threshold,
        score_threshold=args.score_threshold,
        target=args.target,
        results=sorted(results, key=lambda r: r["obj_id"]),
        gen_mode=args.gen_mode,
        min_merged_success=args.min_merged_success,
    )

    ok = sum(1 for r in results if r["status"] in ("ok", "skip"))
    failed = len(results) - ok
    total_cand = sum(r.get("n_candidates", 0) for r in results)
    print(f"\n✅ Done in {time.time() - t0:.0f}s")
    print(f"   ok/skip: {ok}  failed/other: {failed}")
    print(f"   candidates in pool (this run tally): {total_cand}")
    print(f"   manifest: {manifest_path}")


if __name__ == "__main__":
    main()
