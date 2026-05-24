#!/usr/bin/env python3
"""
batch_sim_candidates_pool.py — 候选池加权 sim batch（固定 4 yaw）
================================================================
与 batch_grasp_collect.py 输出目录兼容: candidates/round_R, robot_gt/round_R,
merged/, state.json, summary.csv。

前置: batch_gen_candidates_pool.py 已生成 candidates/pool/{obj}_grasp.hdf5

用法:
    export ISAAC_SIM_PATH=/path/to/isaac-sim

    python3 scripts/batch_sim_candidates_pool.py \\
        --outdir output/grasp_collect_no_rot \\
        --resume --max-rounds 5 \\
        --sim-gpu-ids 0,1 --sim-per-gpu 5 --headless

    # 小规模 smoke:
    python3 scripts/batch_sim_candidates_pool.py \\
        --slots-per-round 2 --sim-gpu-ids 0 --sim-per-gpu 1 --max-rounds 1
"""
from __future__ import annotations

import argparse
import csv
import fcntl
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

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from grasp_pool_common import (  # noqa: E402
    DEFAULT_SLOTS_PER_ROUND,
    build_task_queue,
    compute_median_success_threshold,
    copy_slots_to_round_hdf5,
    is_queue_complete,
    load_registry,
    load_task_queue,
    mark_task_done,
    paths_for_outdir,
    pending_tasks,
    round_tag,
    save_registry,
    save_task_queue,
    scan_merged_objects,
    split_tasks_into_chunks,
    unique_slots_from_tasks,
    update_registry_from_results,
)
from batch_gen_candidates_pool import resolve_dataset  # noqa: E402

DEFAULT_OUT = os.path.join(PROJ, "output", "grasp_collect_no_rot")
GEN_POOL_SCRIPT = os.path.join(PROJ, "scripts", "batch_gen_candidates_pool.py")
MERGE = os.path.join(PROJ, "tools", "merge_robot_gt.py")
SIM_POOL_SCRIPT = os.path.join(PROJ, "sim", "run_grasp_sim_pool.py")
SAME_GPU_STAGGER_S = 10.0


@dataclass
class PoolSimConfig:
    outdir: str
    pool_dir: str
    merged_dir: str
    headless: bool
    isaac_python: str
    sim_timeout: int
    object_scale: float
    python_bin: str
    sim_gpu_ids: tuple[int, ...]
    sim_per_gpu: int
    merge_deduplicate: bool
    slots_per_round: int
    plan_seed: Optional[int]
    pool_target: int
    auto_refill: bool
    score_threshold: float
    no_rotation: bool


def _run_cmd(
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


def _load_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {"round": 0, "objects": {}, "updated_at": None}
    with open(path, "r") as f:
        return json.load(f)


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(state, f, indent=2)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_summary(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _merge_all_rounds(cfg: PoolSimConfig, obj_id: str, up_to_round: int) -> str:
    paths = paths_for_outdir(cfg.outdir, 0)
    inputs = []
    for r in range(up_to_round + 1):
        p = os.path.join(cfg.outdir, "robot_gt", round_tag(r), f"{obj_id}_robot_gt.hdf5")
        if os.path.isfile(p):
            inputs.append(p)
    if not inputs:
        return ""
    os.makedirs(paths["merged_dir"], exist_ok=True)
    out_path = os.path.join(paths["merged_dir"], f"{obj_id}_robot_gt_merged.hdf5")
    cmd = [
        cfg.python_bin, MERGE,
        "--obj", obj_id,
        "--output", out_path,
        "--inputs", *inputs,
    ]
    if cfg.merge_deduplicate:
        cmd.append("--deduplicate")
    rc, _ = _run_cmd(cmd)
    if rc != 0:
        return ""
    return out_path if os.path.isfile(out_path) else ""


def _summary_row(
    obj_id: str,
    dataset: str,
    round_idx: int,
    n_candidates: int,
    sim_status: str,
    n_success: int,
    grasp_hdf5: str,
    gt_hdf5: str,
    merged_hdf5: str,
    sim_elapsed_s: float,
    error: str = "",
) -> dict:
    row = {
        "obj_id": obj_id,
        "dataset": dataset,
        "round": round_idx,
        "gen_status": "pool_copy",
        "n_candidates": n_candidates,
        "sim_status": sim_status,
        "n_success": n_success,
        "grasp_hdf5": grasp_hdf5,
        "gt_hdf5": gt_hdf5,
        "merged_hdf5": merged_hdf5,
        "gen_elapsed_s": 0,
        "sim_elapsed_s": sim_elapsed_s,
        "error": error,
    }
    if sim_status in ("ok", "sim_skip") and n_success > 0:
        row["status"] = "ok"
    elif sim_status in ("ok", "all_failed", "sim_skip"):
        row["status"] = sim_status
    else:
        row["status"] = sim_status
    return row


def _dataset_by_obj(objs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for oid in objs:
        ds = resolve_dataset(oid)
        if ds:
            out[oid] = ds
    return out


def _chunk_startup_delays(gpu_for_chunk: list[int], stagger_s: float) -> list[float]:
    """Per-chunk sleep before Isaac launch; 2nd+ worker on same GPU waits stagger_s each."""
    launch_index_by_gpu: dict[int, int] = {}
    delays: list[float] = []
    for gpu in gpu_for_chunk:
        idx = launch_index_by_gpu.get(gpu, 0)
        delays.append(idx * stagger_s)
        launch_index_by_gpu[gpu] = idx + 1
    return delays


def run_sim_worker(
    chunk_path: str,
    gpu_id: int,
    cfg_dict: dict,
    startup_delay_s: float = 0.0,
) -> dict:
    cfg = PoolSimConfig(**cfg_dict)
    if startup_delay_s > 0:
        time.sleep(startup_delay_s)
    t0 = time.time()
    with open(chunk_path, "r") as cf:
        chunk_meta = json.load(cf)
    expected_tasks = len(chunk_meta.get("tasks", []))
    log_dir = os.path.join(
        cfg.outdir, "sim_logs",
        round_tag(chunk_meta["round_idx"]),
    )
    os.makedirs(log_dir, exist_ok=True)
    chunk_id = os.path.basename(chunk_path).replace(".json", "")
    log_path = os.path.join(log_dir, f"{chunk_id}_gpu{gpu_id}.log")

    sim_cmd = [
        cfg.isaac_python, SIM_POOL_SCRIPT,
        "--worker-chunk", chunk_path,
        "--outdir", cfg.outdir,
        "--object-scale", str(cfg.object_scale),
    ]
    if cfg.headless:
        sim_cmd.append("--headless")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    rc, out = _run_cmd(sim_cmd, timeout=cfg.sim_timeout, log_path=log_path, env=env)

    results_path = chunk_path.replace(".json", "_results.json")
    if results_path == chunk_path:
        results_path = chunk_path + "_results.json"

    payload = {
        "chunk_path": chunk_path,
        "status": "ok",
        "results": [],
        "error": "",
        "expected_tasks": expected_tasks,
    }
    if rc != 0:
        payload["status"] = "sim_timeout" if rc == -9 else "sim_failed"
        payload["error"] = (out or f"exit {rc}")[:500]
    elif os.path.isfile(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
        payload["results"] = data.get("results", [])
        n_got = len(payload["results"])
        if expected_tasks > 0 and n_got < expected_tasks:
            payload["status"] = "partial"
            payload["error"] = f"got {n_got}/{expected_tasks} task results"
    else:
        payload["status"] = "no_results"
        payload["error"] = f"missing {results_path}"

    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["gpu_id"] = gpu_id
    return payload


def _assign_chunks_to_gpus(n_chunks: int, gpu_ids: tuple[int, ...]) -> list[int]:
    """chunk index -> gpu_id (contiguous blocks per GPU)."""
    if n_chunks <= 0:
        return []
    gpus = list(gpu_ids)
    mapping: list[int] = []
    base, rem = divmod(n_chunks, len(gpus))
    idx = 0
    for gi, gpu in enumerate(gpus):
        count = base + (1 if gi < rem else 0)
        mapping.extend([gpu] * count)
        idx += count
    while len(mapping) < n_chunks:
        mapping.append(gpus[len(mapping) % len(gpus)])
    return mapping[:n_chunks]


def run_sim_phase(
    queue: dict,
    cfg: PoolSimConfig,
    round_idx: int,
) -> tuple[list[dict], list[dict]]:
    tasks = pending_tasks(queue)
    if not tasks:
        return [], []

    n_workers = len(cfg.sim_gpu_ids) * cfg.sim_per_gpu
    chunks = split_tasks_into_chunks(tasks, n_workers)
    chunk_dir = os.path.join(cfg.outdir, "sim_logs", round_tag(round_idx), "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_paths: list[str] = []
    for i, chunk_tasks in enumerate(chunks):
        cpath = os.path.join(chunk_dir, f"chunk_{i:03d}.json")
        payload = {
            "chunk_id": f"chunk_{i:03d}",
            "round_idx": round_idx,
            "outdir": cfg.outdir,
            "object_scale": cfg.object_scale,
            "tasks": chunk_tasks,
        }
        with open(cpath, "w") as f:
            json.dump(payload, f)
        chunk_paths.append(cpath)

    gpu_for_chunk = _assign_chunks_to_gpus(len(chunk_paths), cfg.sim_gpu_ids)
    startup_delays = _chunk_startup_delays(gpu_for_chunk, SAME_GPU_STAGGER_S)
    cfg_dict = asdict(cfg)

    print(
        f"  Sim: {len(tasks)} tasks → {len(chunk_paths)} chunks "
        f"({len(cfg.sim_gpu_ids)} GPU × {cfg.sim_per_gpu} workers)",
    )
    if any(d > 0 for d in startup_delays):
        print(f"  Same-GPU stagger: {SAME_GPU_STAGGER_S:.0f}s between workers on one GPU")

    all_results: list[dict] = []
    worker_outcomes: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for i in range(len(chunk_paths)):
            if startup_delays[i] > 0:
                print(
                    f"    chunk {i} GPU {gpu_for_chunk[i]}: "
                    f"startup delay +{startup_delays[i]:.0f}s",
                )
            fut = ex.submit(
                run_sim_worker,
                chunk_paths[i],
                gpu_for_chunk[i],
                cfg_dict,
                startup_delays[i],
            )
            futures[fut] = i
        for fut in as_completed(futures):
            ci = futures[fut]
            out = fut.result()
            worker_outcomes.append(out)
            all_results.extend(out.get("results", []))
            print(
                f"    chunk {ci} GPU {out.get('gpu_id', '?')}: {out['status']}  "
                f"{len(out.get('results', []))}/{out.get('expected_tasks', '?')} results  "
                f"{out.get('elapsed_s', 0)}s",
            )
            if out.get("error"):
                print(f"      {out['error'][:200]}")

    return all_results, worker_outcomes


def _count_round_candidates(cand_round_dir: str, obj_id: str) -> int:
    path = os.path.join(cand_round_dir, f"{obj_id}_grasp.hdf5")
    if not os.path.isfile(path):
        return 0
    import h5py

    with h5py.File(path, "r") as f:
        if "candidates" not in f:
            return 0
        return int(f["candidates"].attrs.get("n_candidates", 0))


def _count_round_success(gt_round_dir: str, obj_id: str) -> int:
    path = os.path.join(gt_round_dir, f"{obj_id}_robot_gt.hdf5")
    if not os.path.isfile(path):
        return 0
    import h5py

    with h5py.File(path, "r") as f:
        return int(f.attrs.get("n_successful", 0))


def auto_refill_pool(cfg: PoolSimConfig) -> bool:
    threshold = compute_median_success_threshold(cfg.outdir, cfg.merged_dir)
    print(f"\n🔄 Pool exhausted — auto refill (success_threshold=median={threshold})")
    cmd = [
        cfg.python_bin, GEN_POOL_SCRIPT,
        "--merged-dir", cfg.merged_dir,
        "--output-dir", cfg.pool_dir,
        "--success-threshold", str(threshold),
        "--target", str(cfg.pool_target),
        "--force",
        "--score-threshold", str(cfg.score_threshold),
    ]
    if cfg.no_rotation:
        pass  # default no-rotation in gen script
    else:
        cmd.append("--rotation")
    rc, out = _run_cmd(cmd)
    if rc != 0:
        print(f"❌ auto refill failed: {out[:400]}")
        return False
    print("✅ Pool refill finished")
    return True


def run_one_round(
    cfg: PoolSimConfig,
    round_idx: int,
    *,
    resume: bool,
) -> dict:
    """Run planning (if needed), sim workers, merge, summary. Returns round meta."""
    paths = paths_for_outdir(cfg.outdir, round_idx)
    os.makedirs(paths["cand_round_dir"], exist_ok=True)
    os.makedirs(paths["gt_round_dir"], exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)

    registry = load_registry(paths["registry"])
    queue = load_task_queue(paths["task_queue"]) if resume else None

    replanned = False
    if queue is None or queue.get("round_idx") != round_idx:
        rng = np.random.default_rng(cfg.plan_seed + round_idx if cfg.plan_seed is not None else None)
        merged_objs = list(scan_merged_objects(cfg.merged_dir).keys())
        dataset_by_obj = _dataset_by_obj(merged_objs)
        queue = build_task_queue(
            outdir=cfg.outdir,
            merged_dir=cfg.merged_dir,
            pool_dir=cfg.pool_dir,
            registry=registry,
            round_idx=round_idx,
            slots_per_round=cfg.slots_per_round,
            dataset_by_obj=dataset_by_obj,
            rng=rng,
        )
        save_task_queue(paths["task_queue"], queue)
        replanned = True

        slots = unique_slots_from_tasks(queue["tasks"])
        if slots:
            for fn in glob.glob(os.path.join(paths["cand_round_dir"], "*_grasp.hdf5")):
                os.remove(fn)
            copy_slots_to_round_hdf5(cfg.pool_dir, paths["cand_round_dir"], slots)
        print(
            f"  Planned {queue['slots_planned']}/{cfg.slots_per_round} slots  "
            f"({len(queue['tasks'])} sim tasks)  exhausted={queue['pool_exhausted']}",
        )
    else:
        print(f"  Resume task queue: {len(pending_tasks(queue))} pending tasks")

    pool_exhausted = bool(queue.get("pool_exhausted"))

    if not is_queue_complete(queue):
        sim_results, _ = run_sim_phase(queue, cfg, round_idx)
        update_registry_from_results(registry, sim_results, round_idx)
        for r in sim_results:
            if r.get("task_id"):
                mark_task_done(queue, r["task_id"])
        save_registry(paths["registry"], registry)
        save_task_queue(paths["task_queue"], queue)
    else:
        sim_results = []

    objs = sorted({t["obj_id"] for t in queue.get("tasks", [])})
    dataset_by_obj = _dataset_by_obj(objs)
    summary_rows: list[dict] = []

    for obj_id in objs:
        n_cand = _count_round_candidates(paths["cand_round_dir"], obj_id)
        n_ok = _count_round_success(paths["gt_round_dir"], obj_id)
        gt_path = os.path.join(paths["gt_round_dir"], f"{obj_id}_robot_gt.hdf5")
        grasp_path = os.path.join(paths["cand_round_dir"], f"{obj_id}_grasp.hdf5")
        merged = _merge_all_rounds(cfg, obj_id, round_idx)
        sim_status = "ok" if n_ok > 0 else "all_failed"
        if not os.path.isfile(grasp_path):
            sim_status = "no_grasp_hdf5"
        row = _summary_row(
            obj_id,
            dataset_by_obj.get(obj_id, ""),
            round_idx,
            n_cand,
            sim_status,
            n_ok,
            grasp_path,
            gt_path if os.path.isfile(gt_path) else "",
            merged,
            0,
        )
        _append_summary(paths["summary"], row)
        summary_rows.append(row)

    n_success_tasks = sum(1 for r in sim_results if r.get("success"))
    queue_complete = is_queue_complete(queue)
    n_pending = len(pending_tasks(queue))
    return {
        "round_idx": round_idx,
        "pool_exhausted": pool_exhausted,
        "slots_planned": queue.get("slots_planned", 0),
        "n_tasks": len(queue.get("tasks", [])),
        "n_sim_success": n_success_tasks,
        "n_pending_tasks": n_pending,
        "queue_complete": queue_complete,
        "replanned": replanned,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pool-based weighted sim (batch_sim_candidates_pool.py)",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument(
        "--pool-dir",
        default=None,
        help="candidate 池 (default: {outdir}/candidates/pool)",
    )
    parser.add_argument(
        "--merged-dir",
        default=None,
        help="merged robot_gt (default: {outdir}/merged)",
    )
    parser.add_argument(
        "--slots-per-round",
        type=int,
        default=DEFAULT_SLOTS_PER_ROUND,
        help="每轮 candidate 槽位数 (×4 = sim task 数)",
    )
    parser.add_argument("--max-rounds", type=int, default=1, help="本次最多跑几轮")
    parser.add_argument("--resume", action="store_true", help="从 state.json 的 round 续跑")
    parser.add_argument("--sim-gpu-ids", default="0")
    parser.add_argument("--sim-per-gpu", type=int, default=1)
    parser.add_argument("--sim-timeout", type=int, default=7200, help="单个 worker chunk 超时秒")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--plan-seed", type=int, default=None)
    parser.add_argument("--pool-target", type=int, default=50, help="auto-refill 时每物体 target")
    parser.add_argument("--score-threshold", type=float, default=70.0)
    parser.add_argument("--no-auto-refill", action="store_true")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument(
        "--merge-deduplicate", action="store_true",
        help="合并 merged 时去重 (默认不去重)",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--isaac-python", default=None)
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    pool_dir = os.path.abspath(args.pool_dir or os.path.join(outdir, "candidates", "pool"))
    merged_dir = os.path.abspath(args.merged_dir or os.path.join(outdir, "merged"))

    sim_gpu_ids = tuple(int(x.strip()) for x in args.sim_gpu_ids.split(",") if x.strip())
    if not sim_gpu_ids:
        print("❌ --sim-gpu-ids 为空")
        sys.exit(1)
    if args.sim_per_gpu < 1:
        print("❌ --sim-per-gpu 须 >= 1")
        sys.exit(1)

    isaac_py = args.isaac_python
    if not isaac_py:
        base = os.environ.get("ISAAC_SIM_PATH", "").rstrip("/")
        isaac_py = os.path.join(base, "python.sh") if base else ""
    if not os.path.isfile(isaac_py):
        print(f"❌ Isaac python not found: {isaac_py}")
        print("   export ISAAC_SIM_PATH=/path/to/isaac-sim")
        sys.exit(1)

    if not os.path.isdir(merged_dir):
        print(f"❌ merged dir not found: {merged_dir}")
        sys.exit(1)

    cfg = PoolSimConfig(
        outdir=outdir,
        pool_dir=pool_dir,
        merged_dir=merged_dir,
        headless=args.headless,
        isaac_python=isaac_py,
        sim_timeout=args.sim_timeout,
        object_scale=args.object_scale,
        python_bin=args.python_bin,
        sim_gpu_ids=sim_gpu_ids,
        sim_per_gpu=args.sim_per_gpu,
        merge_deduplicate=args.merge_deduplicate,
        slots_per_round=args.slots_per_round,
        plan_seed=args.plan_seed,
        pool_target=args.pool_target,
        auto_refill=not args.no_auto_refill,
        score_threshold=args.score_threshold,
        no_rotation=not args.rotation,
    )

    state_path = os.path.join(outdir, "state.json")
    state = _load_state(state_path) if args.resume else {"round": 0, "objects": {}}
    start_round = int(state.get("round", 0)) if args.resume else 0

    print(f"Out: {outdir}")
    print(f"Pool: {pool_dir}")
    print(f"Merged: {merged_dir}")
    print(f"Rounds: {args.max_rounds}  start_round: {start_round}")
    print(
        f"Sim: {len(sim_gpu_ids)} GPU × {args.sim_per_gpu}/GPU = "
        f"{len(sim_gpu_ids) * args.sim_per_gpu} workers",
    )

    for i in range(args.max_rounds):
        r = start_round + i
        print(f"\n{'='*60}")
        print(f"Round {r} ({round_tag(r)})")
        meta = run_one_round(cfg, r, resume=args.resume)
        if meta.get("queue_complete", False):
            state["round"] = r + 1
            _save_state(state_path, state)
        else:
            state["round"] = r
            _save_state(state_path, state)
            print(
                f"⚠️  Round {r} incomplete "
                f"({meta.get('n_pending_tasks', '?')} pending tasks); "
                f"state.round stays at {r} (re-run with --resume)",
            )

        if meta["pool_exhausted"] and cfg.auto_refill:
            auto_refill_pool(cfg)
        elif meta["pool_exhausted"]:
            print("⚠️  Pool exhausted (--no-auto-refill); stop after this round")

        if meta["slots_planned"] == 0:
            print("⚠️  No slots planned (no eligible candidates); stopping")
            break

    print(f"\n✅ Done. state.round={state['round']}")
    print(f"   summary: {os.path.join(outdir, 'summary.csv')}")


if __name__ == "__main__":
    main()
