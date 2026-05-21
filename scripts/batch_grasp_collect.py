#!/usr/bin/env python3
"""
batch_grasp_collect.py — 批量: 生成 candidate → Isaac Sim 验证 → 记录成功 GT
=============================================================================
每轮两阶段:
  1) 并行生成全部物体的 candidate pose (--sampler-workers)
  2) 并行 Isaac Sim 验证 (--sim-per-gpu × 每张 --sim-gpu-ids)

前置:
  - conda 环境含 scipy/trimesh/h5py/rtree (生成)
  - ISAAC_SIM_PATH 指向 Isaac Sim (sim)
  - 建议先: python3 tools/convert_obj_usd.py --dataset oakink|ycb --no-rotation --force

物体列表与 random_grasp_sampler 一致 (rotated_mesh + train_fp_rotated + scale.json)。
Sim 使用 run_grasp_sim.py (headless)；录屏请单独 run_grasp_sim_rec.py。

用法:
    export ISAAC_SIM_PATH=/path/to/isaac-sim

    python3 scripts/batch_grasp_collect.py --dataset oakink,ycb \\
        --sampler-workers 8 --sim-per-gpu 1 --sim-gpu-ids 0,1 --headless --no-convert

    # 默认 10 轮 (round_0000..0009)；续跑更多轮:
    python3 scripts/batch_grasp_collect.py --dataset all --max-rounds 5 --resume ...

输出目录（默认）: output/grasp_collect_no_rot/
  旧实验: output/grasp_collect_legacy/（原 grasp_collect，勿与新区混用 --resume）
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))

DEFAULT_OUT = os.path.join(PROJ, "output", "grasp_collect_no_rot")
SAMPLER = os.path.join(PROJ, "tools", "random_grasp_sampler.py")
CONVERT = os.path.join(PROJ, "tools", "convert_obj_usd.py")
MERGE = os.path.join(PROJ, "tools", "merge_robot_gt.py")
SIM_SCRIPT = os.path.join(PROJ, "sim", "run_grasp_sim.py")


@dataclass
class JobConfig:
    outdir: str
    dataset: str
    target: int
    score_threshold: float
    no_rotation: bool
    headless: bool
    isaac_python: str
    sim_timeout: int
    max_candidates: Optional[int]
    convert_usd: bool
    python_bin: str
    resume: bool
    sim_gpu_ids: tuple[int, ...]
    merge_deduplicate: bool


@dataclass
class GenResult:
    obj_id: str
    dataset: str
    round_idx: int
    worker_id: int
    status: str
    n_candidates: int = 0
    grasp_hdf5: str = ""
    error: str = ""
    elapsed_s: float = 0.0


@dataclass
class SimResult:
    obj_id: str
    dataset: str
    round_idx: int
    gpu_id: int
    status: str
    n_success: int = 0
    grasp_hdf5: str = ""
    gt_hdf5: str = ""
    merged_hdf5: str = ""
    error: str = ""
    elapsed_s: float = 0.0


# 当前 batch 支持的 dataset（--dataset all 等价于下列全部）
BATCH_DATASETS = ("oakink", "ycb")
ObjectJob = tuple[str, str]  # (obj_id, dataset)


def parse_datasets(spec: str) -> list[str]:
    """解析 --dataset：单集 / 逗号多集 / all → ['oakink', 'ycb', ...]。"""
    s = (spec or "oakink").strip().lower()
    if s in ("all", "oakink+ycb"):
        return list(BATCH_DATASETS)
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts:
        return ["oakink"]
    unknown = [p for p in parts if p not in BATCH_DATASETS]
    if unknown:
        raise ValueError(
            f"unknown dataset(s): {unknown}; supported: {', '.join(BATCH_DATASETS)} or all"
        )
    return parts


def list_object_jobs(dataset_spec: str) -> list[ObjectJob]:
    """与 random_grasp_sampler.list_dataset_objs 一致；返回 (obj_id, dataset) 列表。"""
    from random_grasp_sampler import list_dataset_objs

    jobs: list[ObjectJob] = []
    for ds in parse_datasets(dataset_spec):
        for obj_id in list_dataset_objs(ds, use_legacy_assets=False):
            jobs.append((obj_id, ds))
    return jobs


def resolve_object_jobs(obj_id: str, dataset_spec: str) -> list[ObjectJob]:
    """--obj 时在指定 dataset 集合中解析所属 dataset。"""
    from random_grasp_sampler import list_dataset_objs

    for ds in parse_datasets(dataset_spec):
        if obj_id in list_dataset_objs(ds, use_legacy_assets=False):
            return [(obj_id, ds)]
    return []


def _run(
    cmd: list[str],
    timeout: Optional[int] = None,
    log_path: Optional[str] = None,
    env: Optional[dict] = None,
) -> tuple[int, str]:
    log_f = open(log_path, "w") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJ,
            env=env,
            stdout=log_f if log_path else subprocess.PIPE,
            stderr=subprocess.STDOUT if log_path else subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        out = ""
        if not log_path and proc.stdout:
            out = proc.stdout
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return -9, "timeout"
    finally:
        if log_path and log_f and not log_f.closed:
            log_f.close()


def _round_tag(round_idx: int) -> str:
    return f"round_{round_idx:04d}"


def _paths(cfg: JobConfig, obj_id: str, round_idx: int):
    tag = _round_tag(round_idx)
    base = cfg.outdir
    return {
        "cand_dir": os.path.join(base, "candidates", tag),
        "gt_dir": os.path.join(base, "robot_gt", tag),
        "log_dir": os.path.join(base, "sim_logs", tag),
        "merged_dir": os.path.join(base, "merged"),
        "grasp_hdf5": os.path.join(base, "candidates", tag, f"{obj_id}_grasp.hdf5"),
        "gt_hdf5": os.path.join(base, "robot_gt", tag, f"{obj_id}_robot_gt.hdf5"),
        "merged_hdf5": os.path.join(base, "merged", f"{obj_id}_robot_gt_merged.hdf5"),
    }


def _merge_all_rounds(cfg: JobConfig, obj_id: str, up_to_round: int) -> str:
    paths = _paths(cfg, obj_id, 0)
    inputs = []
    for r in range(up_to_round + 1):
        p = os.path.join(cfg.outdir, "robot_gt", _round_tag(r), f"{obj_id}_robot_gt.hdf5")
        if os.path.isfile(p):
            inputs.append(p)
    if not inputs:
        return ""
    os.makedirs(paths["merged_dir"], exist_ok=True)
    cmd = [
        cfg.python_bin, MERGE,
        "--obj", obj_id,
        "--output", paths["merged_hdf5"],
        "--inputs", *inputs,
    ]
    if cfg.merge_deduplicate:
        cmd.append("--deduplicate")
    rc, _ = _run(cmd)
    if rc != 0:
        return ""
    return paths["merged_hdf5"] if os.path.isfile(paths["merged_hdf5"]) else ""


def run_gen_job(
    obj_id: str,
    dataset: str,
    round_idx: int,
    worker_id: int,
    cfg_dict: dict,
) -> dict:
    """Phase 1: optional USD convert + grasp candidate generation."""
    cfg = JobConfig(**cfg_dict)
    t0 = time.time()
    paths = _paths(cfg, obj_id, round_idx)
    for d in (paths["cand_dir"], paths["log_dir"]):
        os.makedirs(d, exist_ok=True)

    result = GenResult(
        obj_id=obj_id, dataset=dataset, round_idx=round_idx,
        worker_id=worker_id, status="pending",
    )

    try:
        if cfg.resume and os.path.isfile(paths["grasp_hdf5"]):
            import h5py
            with h5py.File(paths["grasp_hdf5"], "r") as f:
                result.n_candidates = int(f["candidates"].attrs.get("n_candidates", 0))
            result.grasp_hdf5 = paths["grasp_hdf5"]
            result.status = "gen_skip"
            return asdict(result)

        if cfg.convert_usd:
            conv_cmd = [
                cfg.python_bin, CONVERT,
                "--obj", obj_id, "--dataset", dataset, "--force",
            ]
            if cfg.no_rotation:
                conv_cmd.append("--no-rotation")
            rc, out = _run(conv_cmd, log_path=os.path.join(paths["log_dir"], f"{obj_id}_convert.log"))
            if rc != 0:
                result.status = "convert_failed"
                result.error = out[:500]
                return asdict(result)

        gen_cmd = [
            cfg.python_bin, SAMPLER,
            "--obj", obj_id,
            "--dataset", dataset,
            "--force",
            "--output-dir", paths["cand_dir"],
            "--target", str(cfg.target),
            "--score-threshold", str(cfg.score_threshold),
        ]
        if cfg.no_rotation:
            gen_cmd.append("--no-rotation")
        rc, out = _run(gen_cmd, log_path=os.path.join(paths["log_dir"], f"{obj_id}_gen.log"))
        if rc != 0:
            result.status = "gen_failed"
            result.error = (out or "sampler nonzero exit")[:500]
            return asdict(result)

        if not os.path.isfile(paths["grasp_hdf5"]):
            result.status = "no_candidates"
            return asdict(result)

        import h5py
        with h5py.File(paths["grasp_hdf5"], "r") as f:
            result.n_candidates = int(f["candidates"].attrs.get("n_candidates", 0))
        result.grasp_hdf5 = paths["grasp_hdf5"]
        result.status = "gen_ok" if result.n_candidates > 0 else "no_candidates"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
    finally:
        result.elapsed_s = round(time.time() - t0, 1)

    return asdict(result)


def run_sim_job(
    obj_id: str,
    dataset: str,
    round_idx: int,
    gpu_id: int,
    cfg_dict: dict,
) -> dict:
    """Phase 2: Isaac Sim verify + merge robot_gt across rounds."""
    cfg = JobConfig(**cfg_dict)
    t0 = time.time()
    paths = _paths(cfg, obj_id, round_idx)
    for d in (paths["gt_dir"], paths["log_dir"], paths["merged_dir"]):
        os.makedirs(d, exist_ok=True)

    result = SimResult(
        obj_id=obj_id, dataset=dataset, round_idx=round_idx,
        gpu_id=gpu_id, status="pending",
    )
    result.grasp_hdf5 = paths["grasp_hdf5"]

    try:
        if not os.path.isfile(paths["grasp_hdf5"]):
            result.status = "no_grasp_hdf5"
            return asdict(result)

        if cfg.resume and os.path.isfile(paths["gt_hdf5"]):
            import h5py
            with h5py.File(paths["gt_hdf5"], "r") as f:
                result.n_success = int(f.attrs.get("n_successful", 0))
            result.gt_hdf5 = paths["gt_hdf5"]
            merged = _merge_all_rounds(cfg, obj_id, round_idx)
            result.merged_hdf5 = merged
            result.status = "sim_skip"
            return asdict(result)

        sim_log = os.path.join(paths["log_dir"], f"{obj_id}_sim.log")
        sim_cmd = [
            cfg.isaac_python, SIM_SCRIPT,
            "--hdf5", paths["grasp_hdf5"],
            "--result-dir", paths["gt_dir"],
            "--save-result",
        ]
        if cfg.headless:
            sim_cmd.append("--headless")
        if cfg.max_candidates is not None:
            sim_cmd.extend(["--max-candidates", str(cfg.max_candidates)])

        sim_env = os.environ.copy()
        sim_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        rc, out = _run(sim_cmd, timeout=cfg.sim_timeout, log_path=sim_log, env=sim_env)
        if os.path.isfile(sim_log):
            with open(sim_log, "a") as lf:
                lf.write(f"\n# CUDA_VISIBLE_DEVICES={gpu_id}\n")
        if rc != 0:
            result.status = "sim_failed" if rc != -9 else "sim_timeout"
            result.error = (out or f"exit {rc}")[:500]
            return asdict(result)

        if not os.path.isfile(paths["gt_hdf5"]):
            result.status = "no_gt_file"
            return asdict(result)

        import h5py
        with h5py.File(paths["gt_hdf5"], "r") as f:
            result.n_success = int(f.attrs.get("n_successful", 0))
        result.gt_hdf5 = paths["gt_hdf5"]
        result.merged_hdf5 = _merge_all_rounds(cfg, obj_id, round_idx)
        result.status = "ok" if result.n_success > 0 else "all_failed"
    except Exception as e:
        result.status = "error"
        result.error = str(e)
    finally:
        result.elapsed_s = round(time.time() - t0, 1)

    return asdict(result)


def _run_parallel(
    fn,
    tasks: list[tuple],
    workers: int,
    label: str,
) -> list[dict]:
    """tasks: [(obj_id, dataset, round_idx, worker_id), ...]"""
    cfg_dict = tasks[0][-1] if tasks and len(tasks[0]) > 4 else None
    # tasks are (obj_id, dataset, round_idx, worker_id, cfg_dict)
    results = []
    n = len(tasks)
    if workers <= 1:
        for i, task in enumerate(tasks):
            obj_id, dataset, round_idx, worker_id, cfg = task[:5]
            print(f"  [{label} {i+1}/{n}] {obj_id}")
            res = fn(obj_id, dataset, round_idx, worker_id, cfg)
            results.append(res)
            _print_job_result(label, res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for i, task in enumerate(tasks):
                obj_id, dataset, round_idx, worker_id, cfg = task[:5]
                fut = ex.submit(fn, obj_id, dataset, round_idx, worker_id, cfg)
                futures[fut] = obj_id
            for fut in as_completed(futures):
                obj_id = futures[fut]
                res = fut.result()
                results.append(res)
                _print_job_result(label, res, obj_id)
    return results


def _print_job_result(label: str, res: dict, obj_id: str = ""):
    oid = obj_id or res.get("obj_id", "")
    if label == "gen":
        print(
            f"    {oid}: {res['status']}  "
            f"cand={res.get('n_candidates', 0)}  {res.get('elapsed_s', 0)}s"
        )
    else:
        gpu = res.get("gpu_id", "?")
        print(
            f"    {oid} (GPU {gpu}): {res['status']}  "
            f"success={res.get('n_success', 0)}  {res.get('elapsed_s', 0)}s"
        )


def _partition_objects_by_gpu(objects: list[str], gpu_ids: tuple[int, ...]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {g: [] for g in gpu_ids}
    for i, obj_id in enumerate(objects):
        buckets[gpu_ids[i % len(gpu_ids)]].append(obj_id)
    return buckets


def _run_sim_parallel_per_gpu(
    sim_jobs: list[ObjectJob],
    round_idx: int,
    cfg_dict: dict,
    gpu_ids: tuple[int, ...],
    sim_per_gpu: int,
) -> list[dict]:
    """
    每张 GPU 独立进程池，最多 sim_per_gpu 个 Isaac 同时占用该卡。
    总并行数 = len(gpu_ids) * sim_per_gpu。
    """
    obj_ids = [o for o, _ in sim_jobs]
    dataset_by_obj = {o: ds for o, ds in sim_jobs}
    buckets = _partition_objects_by_gpu(obj_ids, gpu_ids)
    results: list[dict] = []
    executors: dict[int, ProcessPoolExecutor] = {}
    futures: dict = {}

    try:
        for gpu_id in gpu_ids:
            objs = buckets[gpu_id]
            if not objs:
                continue
            executors[gpu_id] = ProcessPoolExecutor(max_workers=sim_per_gpu)
            for obj_id in objs:
                fut = executors[gpu_id].submit(
                    run_sim_job,
                    obj_id,
                    dataset_by_obj[obj_id],
                    round_idx,
                    gpu_id,
                    cfg_dict,
                )
                futures[fut] = obj_id

        for fut in as_completed(futures):
            obj_id = futures[fut]
            res = fut.result()
            results.append(res)
            _print_job_result("sim", res, obj_id)
    finally:
        for ex in executors.values():
            ex.shutdown(wait=True)

    return results


def _summary_row(
    obj_id: str,
    dataset: str,
    round_idx: int,
    gen: Optional[dict],
    sim: Optional[dict],
) -> dict:
    row = {
        "obj_id": obj_id,
        "dataset": dataset,
        "round": round_idx,
        "gen_status": gen.get("status", "missing") if gen else "missing",
        "n_candidates": gen.get("n_candidates", 0) if gen else 0,
        "sim_status": sim.get("status", "missing") if sim else "missing",
        "n_success": sim.get("n_success", 0) if sim else 0,
        "grasp_hdf5": (gen or {}).get("grasp_hdf5") or (sim or {}).get("grasp_hdf5", ""),
        "gt_hdf5": (sim or {}).get("gt_hdf5", ""),
        "merged_hdf5": (sim or {}).get("merged_hdf5", ""),
        "gen_elapsed_s": gen.get("elapsed_s", 0) if gen else 0,
        "sim_elapsed_s": sim.get("elapsed_s", 0) if sim else 0,
        "error": (gen or {}).get("error", "") or (sim or {}).get("error", ""),
    }
    row["status"] = _overall_status(row)
    return row


def _overall_status(row: dict) -> str:
    if row["sim_status"] in ("ok", "sim_skip") and row["n_success"] > 0:
        return "ok"
    if row["sim_status"] in ("ok", "all_failed", "sim_skip"):
        return row["sim_status"]
    if row["gen_status"] in ("gen_ok", "gen_skip") and row["sim_status"] == "missing":
        return "sim_pending"
    return row["gen_status"] if row["gen_status"] != "gen_ok" else row["sim_status"]


def run_round(
    object_jobs: list[ObjectJob],
    round_idx: int,
    cfg_dict: dict,
    sampler_workers: int,
    sim_gpu_ids: tuple[int, ...],
    sim_per_gpu: int,
    summary_path: str,
) -> tuple[list[dict], list[dict]]:
    tag = _round_tag(round_idx)
    total_sim_slots = len(sim_gpu_ids) * sim_per_gpu
    n_by_ds: dict[str, int] = {}
    for _, ds in object_jobs:
        n_by_ds[ds] = n_by_ds.get(ds, 0) + 1
    ds_summary = ", ".join(f"{ds}={n}" for ds, n in sorted(n_by_ds.items()))
    print(f"\n{'='*60}")
    print(f"Round {round_idx} ({tag})  objects={len(object_jobs)} ({ds_summary})")
    print(f"  Phase 1: sampler_workers={sampler_workers}")
    print(
        f"  Phase 2: sim_gpu_ids={sim_gpu_ids}  sim_per_gpu={sim_per_gpu}  "
        f"(max {total_sim_slots} concurrent Isaac)"
    )

    gen_tasks = [
        (obj_id, dataset, round_idx, i % max(sampler_workers, 1), cfg_dict)
        for i, (obj_id, dataset) in enumerate(object_jobs)
    ]
    t_gen = time.time()
    gen_results = _run_parallel(run_gen_job, gen_tasks, sampler_workers, "gen")
    print(f"  Phase 1 done in {time.time() - t_gen:.0f}s")

    gen_by_obj = {r["obj_id"]: r for r in gen_results}
    sim_jobs = [
        (obj_id, dataset)
        for obj_id, dataset in object_jobs
        if gen_by_obj.get(obj_id, {}).get("grasp_hdf5")
        and os.path.isfile(gen_by_obj[obj_id]["grasp_hdf5"])
    ]
    skipped = len(object_jobs) - len(sim_jobs)
    if skipped:
        print(f"  Sim queue: {len(sim_jobs)} objects ({skipped} skipped, no grasp HDF5)")

    sim_results: list[dict] = []
    if sim_jobs:
        for gpu_id, objs in _partition_objects_by_gpu(
            [o for o, _ in sim_jobs], sim_gpu_ids
        ).items():
            if objs:
                print(f"    GPU {gpu_id}: {len(objs)} objects")
        t_sim = time.time()
        sim_results = _run_sim_parallel_per_gpu(
            sim_jobs, round_idx, cfg_dict, sim_gpu_ids, sim_per_gpu,
        )
        print(f"  Phase 2 done in {time.time() - t_sim:.0f}s")

    sim_by_obj = {r["obj_id"]: r for r in sim_results}
    for obj_id, dataset in object_jobs:
        row = _summary_row(
            obj_id, dataset, round_idx, gen_by_obj.get(obj_id), sim_by_obj.get(obj_id),
        )
        _append_summary(summary_path, row)

    return gen_results, sim_results


def _load_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {"round": 0, "objects": {}, "updated_at": None}
    with open(path, "r") as f:
        return json.load(f)


def _save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(state, f, indent=2)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_summary(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="Batch grasp: two-phase gen then sim")
    parser.add_argument(
        "--dataset",
        default="oakink,ycb",
        help="数据集: oakink / ycb / oakink,ycb / all（默认 oakink+ycb 每轮一起走）",
    )
    parser.add_argument("--obj", help="只跑单个物体 (smoke test)")
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--target", type=int, default=20, help="每轮每物体 candidate 数")
    parser.add_argument("--score-threshold", type=float, default=70.0)
    parser.add_argument(
        "--sampler-workers", type=int, default=None,
        help="Phase 1 并行数 (candidate 生成, CPU)",
    )
    parser.add_argument(
        "--sim-per-gpu", type=int, default=1,
        help="每张 GPU 上同时跑的 Isaac 进程数 (默认 1，显存紧勿加大)",
    )
    parser.add_argument(
        "--sim-gpu-ids", type=str, default="0",
        help="物理 GPU 编号，逗号分隔，如 0,1；物体按序轮询分配到各卡",
    )
    parser.add_argument(
        "--sim-workers", type=int, default=None,
        help="(已弃用) 等同 --sim-per-gpu，请改用 --sim-per-gpu",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="(已弃用) 仅设置 sampler-workers",
    )
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="轮数 (每轮先全量 gen 再全量 sim；默认 10 → round_0000..0009)")
    parser.add_argument("--max-candidates", type=int, default=None, help="sim 最多尝试数")
    parser.add_argument("--sim-timeout", type=int, default=2400,
                        help="单次 sim 超时秒 (默认 2400 = 40 分钟)")
    parser.add_argument("--rotation", action="store_true",
                        help="应用 rotation.json (默认: 不旋转)")
    parser.add_argument("--no-convert", action="store_true", help="跳过 convert_obj_usd")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="跳过已有 grasp/gt HDF5")
    parser.add_argument(
        "--merge-deduplicate", action="store_true",
        help="合并 merged/ 时去掉相近 pose（默认不去重）",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--isaac-python", default=None)
    args = parser.parse_args()

    sampler_workers = args.sampler_workers
    if args.workers is not None:
        if sampler_workers is None:
            sampler_workers = args.workers
    sampler_workers = sampler_workers if sampler_workers is not None else 4

    sim_per_gpu = args.sim_per_gpu
    if args.sim_workers is not None:
        print("⚠️  --sim-workers 已弃用，当作 --sim-per-gpu 使用")
        sim_per_gpu = args.sim_workers
    if sim_per_gpu < 1:
        print("❌ --sim-per-gpu 必须 ≥ 1")
        sys.exit(1)

    sim_gpu_ids = tuple(
        int(x.strip()) for x in args.sim_gpu_ids.split(",") if x.strip()
    )
    if not sim_gpu_ids:
        print("❌ --sim-gpu-ids 为空")
        sys.exit(1)

    isaac_py = args.isaac_python
    if not isaac_py:
        base = os.environ.get("ISAAC_SIM_PATH", "").rstrip("/")
        isaac_py = os.path.join(base, "python.sh") if base else ""
    if not os.path.isfile(isaac_py):
        print(f"❌ Isaac python not found: {isaac_py}")
        print("   export ISAAC_SIM_PATH=/path/to/isaac-sim")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    state_path = os.path.join(args.outdir, "state.json")
    summary_path = os.path.join(args.outdir, "summary.csv")

    try:
        if args.obj:
            object_jobs = resolve_object_jobs(args.obj, args.dataset)
            if not object_jobs:
                print(f"❌ --obj {args.obj} not in dataset(s) {args.dataset}")
                sys.exit(1)
        else:
            object_jobs = list_object_jobs(args.dataset)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    if not object_jobs:
        print(f"❌ no objects for --dataset {args.dataset}")
        sys.exit(1)

    state = _load_state(state_path) if args.resume else {"round": 0, "objects": {}}
    start_round = int(state.get("round", 0)) if args.resume else 0

    cfg = JobConfig(
        outdir=os.path.abspath(args.outdir),
        dataset=args.dataset,
        target=args.target,
        score_threshold=args.score_threshold,
        no_rotation=not args.rotation,
        headless=args.headless,
        isaac_python=isaac_py,
        sim_timeout=args.sim_timeout,
        max_candidates=args.max_candidates or args.target,
        convert_usd=not args.no_convert,
        python_bin=args.python_bin,
        resume=args.resume,
        sim_gpu_ids=sim_gpu_ids,
        merge_deduplicate=args.merge_deduplicate,
    )
    cfg_dict = asdict(cfg)

    n_by_ds: dict[str, int] = {}
    for _, ds in object_jobs:
        n_by_ds[ds] = n_by_ds.get(ds, 0) + 1
    print(
        f"Objects: {len(object_jobs)} ({', '.join(f'{ds}={n}' for ds, n in sorted(n_by_ds.items()))})  "
        f"rounds: {args.max_rounds}  start_round: {start_round}"
    )
    print(f"Out: {args.outdir}")
    print(f"No rotation: {cfg.no_rotation}")
    total_sim = len(sim_gpu_ids) * sim_per_gpu
    print(
        f"sampler_workers={sampler_workers}  "
        f"sim: {len(sim_gpu_ids)} GPU × {sim_per_gpu}/GPU = {total_sim} max concurrent"
    )

    for r in range(start_round, start_round + args.max_rounds):
        run_round(
            object_jobs, r, cfg_dict,
            sampler_workers, sim_gpu_ids, sim_per_gpu, summary_path,
        )
        state["round"] = r + 1
        _save_state(state_path, state)

    print(f"\n✅ Done {args.max_rounds} round(s)")
    print(f"   summary: {summary_path}")
    print(f"   merged:  {os.path.join(args.outdir, 'merged')}")


if __name__ == "__main__":
    main()
