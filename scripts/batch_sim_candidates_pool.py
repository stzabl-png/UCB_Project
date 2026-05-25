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

    # 每轮 slot 按 pool 内物体等概率抽取（默认按 success 加权）:
    python3 scripts/batch_sim_candidates_pool.py \\
        --equal-object-prob --max-rounds 1

    # merged 成功 ≥ 80 的物体不再被抽中:
    python3 scripts/batch_sim_candidates_pool.py \\
        --max-success-per-object 80 --max-rounds 1

    # 增量 merged（只需拷贝 merged/ + 本轮 robot_gt，不必搬历史 round_*）:
    python3 scripts/batch_sim_candidates_pool.py \\
        --incremental-merge --resume --max-rounds 1 ...

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
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(PROJ, "tools")
sys.path.insert(0, _TOOLS)
from merge_robot_gt import merge_robot_gt_files, merge_robot_gt_incremental
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from grasp_pool_common import (  # noqa: E402
    DEFAULT_SLOTS_PER_ROUND,
    archive_chunk_results_before_reshuffle,
    build_task_queue,
    clear_registry_for_objects,
    compute_median_success_threshold,
    copy_slots_to_round_hdf5,
    is_queue_complete,
    load_registry,
    load_task_queue,
    paths_for_outdir,
    pending_tasks,
    reset_gpu_startup_sems,
    round_tag,
    save_registry,
    save_task_queue,
    scan_merged_objects,
    split_tasks_into_chunks,
    sync_queue_and_registry_from_chunks,
    unique_slots_from_tasks,
)
from batch_gen_candidates_pool import resolve_dataset, select_target_objects  # noqa: E402

DEFAULT_OUT = os.path.join(PROJ, "output", "grasp_collect_no_rot")
GEN_POOL_SCRIPT = os.path.join(PROJ, "scripts", "batch_gen_candidates_pool.py")
SIM_POOL_SCRIPT = os.path.join(PROJ, "sim", "run_grasp_sim_pool.py")
SAME_GPU_STAGGER_S = 45.0
ISAAC_STARTUP_SLOTS_PER_GPU = 2
SIM_INGEST_POLL_S = 5.0
WORKER_CRASH_RETRY_S = 15.0
WORKER_MAX_RETRIES = 2


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
    same_gpu_stagger_s: float
    isaac_startup_slots_per_gpu: int
    merge_deduplicate: bool
    incremental_merge: bool
    slots_per_round: int
    plan_seed: Optional[int]
    equal_object_prob: bool
    max_success_per_object: Optional[int]
    pool_target: int
    auto_refill: bool
    score_threshold: float
    no_rotation: bool
    early_stop_on_candidate_success: bool


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


def _merge_for_round(cfg: PoolSimConfig, obj_id: str, round_idx: int) -> str:
    """Update merged/{obj}_robot_gt_merged.hdf5 after sim (incremental or full scan)."""
    paths = paths_for_outdir(cfg.outdir, 0)
    os.makedirs(paths["merged_dir"], exist_ok=True)
    out_path = os.path.join(paths["merged_dir"], f"{obj_id}_robot_gt_merged.hdf5")
    round_gt = os.path.join(
        cfg.outdir, "robot_gt", round_tag(round_idx), f"{obj_id}_robot_gt.hdf5",
    )
    try:
        if cfg.incremental_merge:
            if not os.path.isfile(round_gt) and not os.path.isfile(out_path):
                return ""
            merge_robot_gt_incremental(
                obj_id,
                out_path,
                existing_merged=out_path if os.path.isfile(out_path) else None,
                new_round_gt=round_gt,
                deduplicate=cfg.merge_deduplicate,
                verbose=False,
            )
        else:
            inputs = []
            for r in range(round_idx + 1):
                p = os.path.join(
                    cfg.outdir, "robot_gt", round_tag(r), f"{obj_id}_robot_gt.hdf5",
                )
                if os.path.isfile(p):
                    inputs.append(p)
            if not inputs:
                return ""
            merge_robot_gt_files(
                obj_id,
                inputs,
                out_path,
                deduplicate=cfg.merge_deduplicate,
                verbose=False,
            )
    except Exception:
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


def _chunk_sidecar_paths(chunk_path: str) -> tuple[str, str]:
    base = chunk_path[:-5] if chunk_path.endswith(".json") else chunk_path
    return f"{base}_results.json", f"{base}_progress.json"


def _read_chunk_task_ids(chunk_path: str) -> set[str]:
    try:
        with open(chunk_path, "r") as f:
            tasks = json.load(f).get("tasks", [])
            return {t["task_id"] for t in tasks if t.get("task_id")}
    except (OSError, json.JSONDecodeError, TypeError):
        return set()


def _load_chunk_results(chunk_path: str) -> list[dict]:
    """Results for tasks listed in the chunk file (ignores stale sidecar rows)."""
    task_ids = _read_chunk_task_ids(chunk_path)
    if not task_ids:
        return []
    results_path, _ = _chunk_sidecar_paths(chunk_path)
    try:
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                data = json.load(f)
            return [
                r for r in data.get("results", [])
                if r.get("task_id") in task_ids
            ]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _chunk_run_complete(chunk_path: str) -> bool:
    done, total, _ = _read_chunk_progress(chunk_path)
    return total > 0 and done >= total


def _read_chunk_progress(chunk_path: str) -> tuple[int, int, int]:
    """Return (completed, total, successes_in_chunk) for tasks in this chunk file."""
    task_ids = _read_chunk_task_ids(chunk_path)
    total = len(task_ids)
    if total == 0:
        return 0, 0, 0
    results_path, _ = _chunk_sidecar_paths(chunk_path)
    try:
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                data = json.load(f)
            results = [
                r for r in data.get("results", [])
                if r.get("task_id") in task_ids
            ]
            return (
                len(results),
                total,
                sum(1 for r in results if r.get("success")),
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return 0, total, 0


def _format_sim_progress(
    chunk_paths: list[str],
    gpu_for_chunk: list[int],
    *,
    batch_total: int,
) -> str:
    done_tasks = 0
    ok_tasks = 0
    parts: list[str] = []
    for i, cp in enumerate(chunk_paths):
        done, tot, ok = _read_chunk_progress(cp)
        done_tasks += done
        ok_tasks += ok
        gpu = gpu_for_chunk[i] if i < len(gpu_for_chunk) else "?"
        parts.append(f"c{i}:gpu{gpu} {done}/{tot}")
    chunk_summary = " ".join(parts)
    return (
        f"  [sim progress] {done_tasks}/{batch_total} tasks "
        f"({ok_tasks} ok in partial results) | {chunk_summary}"
    )


def _print_sync_progress(
    n_new: int,
    chunk_paths: list[str],
    gpu_for_chunk: list[int],
    *,
    batch_total: int,
    pending: Optional[int] = None,
) -> None:
    """Print chunk sync + sim progress when new task results were ingested."""
    if n_new <= 0:
        return
    if pending is not None:
        print(
            f"  [chunk sync] +{n_new} newly completed ({pending} pending sim)",
            flush=True,
        )
    else:
        print(f"  [chunk sync] +{n_new} newly completed", flush=True)
    print(
        _format_sim_progress(chunk_paths, gpu_for_chunk, batch_total=batch_total),
        flush=True,
    )


class _SimProgressReporter(threading.Thread):
    """Poll worker progress; periodically sync chunk results into task_queue/registry."""

    def __init__(
        self,
        chunk_paths: list[str],
        gpu_for_chunk: list[int],
        log_hint: str,
        *,
        batch_total: int,
        ingest_fn=None,
        persist_lock: threading.Lock | None = None,
        ingest_interval_s: float = SIM_INGEST_POLL_S,
    ):
        super().__init__(daemon=True)
        self._chunk_paths = chunk_paths
        self._gpu_for_chunk = gpu_for_chunk
        self._batch_total = batch_total
        self._log_hint = log_hint
        self._ingest_fn = ingest_fn
        self._persist_lock = persist_lock
        self._ingest_interval = ingest_interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        print(
            f"  Sim progress on newly completed tasks; "
            f"chunk→queue sync every {self._ingest_interval:.0f}s "
            f"(worker logs: {self._log_hint})",
            flush=True,
        )
        while not self._stop_event.wait(self._ingest_interval):
            if self._ingest_fn is not None and self._persist_lock is not None:
                with self._persist_lock:
                    n_new = self._ingest_fn()
                _print_sync_progress(
                    n_new,
                    self._chunk_paths,
                    self._gpu_for_chunk,
                    batch_total=self._batch_total,
                )

    def stop(self) -> None:
        self._stop_event.set()


def _chunk_startup_delays(gpu_for_chunk: list[int], stagger_s: float) -> list[float]:
    """Per-chunk sleep before Isaac launch; 2nd+ worker on same GPU waits stagger_s each."""
    launch_index_by_gpu: dict[int, int] = {}
    delays: list[float] = []
    for gpu in gpu_for_chunk:
        idx = launch_index_by_gpu.get(gpu, 0)
        delays.append(idx * stagger_s)
        launch_index_by_gpu[gpu] = idx + 1
    return delays


def _make_isaac_worker_env(
    base_env: dict,
    *,
    gpu_id: int,
    outdir: str,
    round_idx: int,
    chunk_path: str,
    isaac_startup_slots_per_gpu: int = 0,
) -> dict:
    """Isolate Kit/Omniverse state per worker for same-GPU parallel.

    HOME alone is insufficient: Isaac still shares install-dir kit/cache/DerivedDataCache,
    /tmp/hub-<user>.lock (all workers as root), and kvdb. We also set per-worker TMPDIR,
    OMNI_USER (unique /tmp/hub-<user>.lock), and per-worker TMPDIR.
    """
    env = base_env.copy()
    # Use physical GPU index for Isaac/cuRobo; do NOT mask with CUDA_VISIBLE_DEVICES
    # (PhysX on Isaac 5.0 fails with "no suitable CUDA GPU" when only GPU1 is visible).
    env["ISAAC_SIM_GPU_ID"] = str(gpu_id)
    chunk_id = os.path.basename(chunk_path).replace(".json", "")
    cache_root = os.path.join(
        outdir,
        "sim_logs",
        round_tag(round_idx),
        "kit_cache",
        f"gpu{gpu_id}_{chunk_id}",
    )
    hub_dir = os.path.join(cache_root, "hub")
    omni_cache = os.path.join(cache_root, "omni_cache")
    worker_tmp = os.path.join(cache_root, "tmp")
    os.makedirs(hub_dir, exist_ok=True)
    os.makedirs(omni_cache, exist_ok=True)
    os.makedirs(worker_tmp, exist_ok=True)
    env["OMNICLIENT_HUB_CACHE_DIR"] = hub_dir
    env["OMNI_CACHE_DIR"] = omni_cache
    env["XDG_CACHE_HOME"] = cache_root
    env["TMPDIR"] = worker_tmp
    env["TEMP"] = worker_tmp
    env["TMP"] = worker_tmp
    # Hub discovery lock defaults to /tmp/hub-<user>.lock; unique per worker.
    env["OMNI_USER"] = f"isaac_g{gpu_id}_{chunk_id}"
    if isaac_startup_slots_per_gpu > 0:
        env["ISAAC_STARTUP_SLOTS_PER_GPU"] = str(isaac_startup_slots_per_gpu)

    worker_home = os.path.join(cache_root, "home")
    for sub in (
        ".local/share/ov/data",
        ".nvidia-omniverse/logs",
        ".nvidia-omniverse/config",
        ".nv/ComputeCache",
        ".cache/ov",
        ".cache/pip",
        ".cache/nvidia/GLCache",
    ):
        os.makedirs(os.path.join(worker_home, sub), exist_ok=True)

    global_pkg = os.path.expanduser("~/.local/share/ov/pkg")
    worker_pkg = os.path.join(worker_home, ".local/share/ov/pkg")
    if os.path.isdir(global_pkg) and not os.path.lexists(worker_pkg):
        os.makedirs(os.path.dirname(worker_pkg), exist_ok=True)
        os.symlink(global_pkg, worker_pkg, target_is_directory=True)

    env["HOME"] = worker_home
    return env


def _submit_chunk_worker(
    executor: ProcessPoolExecutor,
    ci: int,
    chunk_paths: list[str],
    gpu_for_chunk: list[int],
    cfg_dict: dict,
    stagger_s: float,
    gpu_launch_idx: dict[int, int],
):
    gpu = gpu_for_chunk[ci]
    delay = gpu_launch_idx.get(gpu, 0) * stagger_s
    gpu_launch_idx[gpu] = gpu_launch_idx.get(gpu, 0) + 1
    if delay > 0:
        print(
            f"    chunk {ci} GPU {gpu}: startup delay +{delay:.0f}s",
            flush=True,
        )
    return executor.submit(
        run_sim_worker,
        chunk_paths[ci],
        gpu,
        cfg_dict,
        delay,
    )


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

    env = _make_isaac_worker_env(
        os.environ.copy(),
        gpu_id=gpu_id,
        outdir=cfg.outdir,
        round_idx=int(chunk_meta["round_idx"]),
        chunk_path=chunk_path,
        isaac_startup_slots_per_gpu=cfg.isaac_startup_slots_per_gpu,
    )
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
    *,
    registry: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    sim_paths = paths_for_outdir(cfg.outdir, round_idx)

    if registry is not None:
        status = sync_queue_and_registry_from_chunks(
            cfg.outdir,
            round_idx,
            registry,
            queue,
            registry_path=sim_paths["registry"],
            task_queue_path=sim_paths["task_queue"],
            persist=True,
        )
        n_on_disk = status.get("n_chunk_tasks", 0)
        if n_on_disk:
            print(
                f"  Chunk sync: {n_on_disk} task(s) on disk, "
                f"{status['n_pending']} pending sim",
                flush=True,
            )

    if registry is not None:
        tasks = pending_tasks(queue, cfg.outdir, round_idx, registry=registry)
    else:
        tasks = pending_tasks(queue)
    if not tasks:
        return [], []

    if cfg.isaac_startup_slots_per_gpu > 0:
        reset_gpu_startup_sems(cfg.outdir, round_idx, cfg.sim_gpu_ids)

    n_workers = len(cfg.sim_gpu_ids) * cfg.sim_per_gpu
    chunks = split_tasks_into_chunks(tasks, n_workers)
    chunk_dir = os.path.join(cfg.outdir, "sim_logs", round_tag(round_idx), "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    n_archived = archive_chunk_results_before_reshuffle(cfg.outdir, round_idx)
    if n_archived:
        print(
            f"  Archived {n_archived} sim result(s) before pending chunk reshuffle",
            flush=True,
        )

    batch_total = len(tasks)
    chunk_paths: list[str] = []
    for i, chunk_tasks in enumerate(chunks):
        cpath = os.path.join(chunk_dir, f"chunk_{i:03d}.json")
        payload = {
            "chunk_id": f"chunk_{i:03d}",
            "round_idx": round_idx,
            "outdir": cfg.outdir,
            "object_scale": cfg.object_scale,
            "early_stop_on_candidate_success": cfg.early_stop_on_candidate_success,
            "tasks": chunk_tasks,
        }
        with open(cpath, "w") as f:
            json.dump(payload, f)
        results_path, progress_path = _chunk_sidecar_paths(cpath)
        for sidecar in (results_path, progress_path):
            if os.path.isfile(sidecar):
                os.remove(sidecar)
        chunk_paths.append(cpath)

    gpu_for_chunk = _assign_chunks_to_gpus(len(chunk_paths), cfg.sim_gpu_ids)
    stagger_s = cfg.same_gpu_stagger_s
    cfg_dict = asdict(cfg)

    print(
        f"  Sim: {batch_total} tasks → {len(chunk_paths)} chunks "
        f"({len(cfg.sim_gpu_ids)} GPU × {cfg.sim_per_gpu} workers)"
        + (
            "; early-stop on candidate success"
            if cfg.early_stop_on_candidate_success
            else ""
        ),
    )
    if stagger_s > 0:
        print(
            f"  Same-GPU stagger: {stagger_s:.0f}s between Isaac launches on one GPU "
            f"(per-GPU process pools + isolated kit cache)",
            flush=True,
        )
    if cfg.isaac_startup_slots_per_gpu > 0:
        print(
            f"  Isaac cold-start semaphore: max {cfg.isaac_startup_slots_per_gpu} "
            f"concurrent Startup per GPU (others queue)",
            flush=True,
        )

    all_results: list[dict] = []
    worker_outcomes: list[dict] = []

    log_hint = os.path.join(
        cfg.outdir, "sim_logs", round_tag(round_idx), "chunk_*_gpu*.log",
    )
    persist_lock = threading.Lock()

    def _sync_from_chunks() -> int:
        st = sync_queue_and_registry_from_chunks(
            cfg.outdir,
            round_idx,
            registry,
            queue,
            registry_path=sim_paths["registry"],
            task_queue_path=sim_paths["task_queue"],
            persist=True,
        )
        return int(st.get("n_newly_completed", 0))

    reporter = _SimProgressReporter(
        chunk_paths,
        gpu_for_chunk,
        log_hint,
        batch_total=batch_total,
        ingest_fn=_sync_from_chunks if registry is not None else None,
        persist_lock=persist_lock if registry is not None else None,
    )
    reporter.start()
    interrupted = False
    executors: dict[int, ProcessPoolExecutor] = {}
    try:
        executors = {
            gpu: ProcessPoolExecutor(max_workers=cfg.sim_per_gpu)
            for gpu in cfg.sim_gpu_ids
        }
        futures: dict = {}
        chunk_attempts = [0] * len(chunk_paths)
        gpu_launch_idx = {gpu: 0 for gpu in cfg.sim_gpu_ids}
        for i in range(len(chunk_paths)):
            chunk_attempts[i] = 1
            gpu = gpu_for_chunk[i]
            fut = _submit_chunk_worker(
                executors[gpu],
                i,
                chunk_paths,
                gpu_for_chunk,
                cfg_dict,
                stagger_s,
                gpu_launch_idx,
            )
            futures[fut] = i

        while futures:
            done_set, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done_set:
                ci = futures.pop(fut)
                try:
                    out = fut.result()
                except Exception as exc:
                    out = {
                        "chunk_path": chunk_paths[ci],
                        "status": "worker_crashed",
                        "results": [],
                        "error": str(exc)[:500],
                        "expected_tasks": len(
                            _read_chunk_task_ids(chunk_paths[ci]),
                        ),
                        "elapsed_s": 0.0,
                        "gpu_id": gpu_for_chunk[ci],
                    }

                done_n, total_n, _ = _read_chunk_progress(chunk_paths[ci])
                if _chunk_run_complete(chunk_paths[ci]):
                    out["status"] = "ok"
                    out["results"] = _load_chunk_results(chunk_paths[ci])
                    worker_outcomes.append(out)
                    all_results.extend(out["results"])
                    print(
                        f"    chunk {ci} GPU {out.get('gpu_id', '?')}: ok  "
                        f"{done_n}/{total_n} results  "
                        f"{out.get('elapsed_s', 0)}s",
                        flush=True,
                    )
                else:
                    retries_used = chunk_attempts[ci] - 1
                    can_retry = retries_used < WORKER_MAX_RETRIES
                    print(
                        f"    chunk {ci} GPU {out.get('gpu_id', '?')}: "
                        f"{out.get('status', '?')}  {done_n}/{total_n} results  "
                        f"{out.get('elapsed_s', 0)}s"
                        + (
                            f" — retry in {WORKER_CRASH_RETRY_S:.0f}s "
                            f"({retries_used + 1}/{WORKER_MAX_RETRIES})"
                            if can_retry
                            else f" — max retries ({WORKER_MAX_RETRIES}) exhausted"
                        ),
                        flush=True,
                    )
                    if out.get("error"):
                        print(f"      {out['error'][:200]}", flush=True)
                    if can_retry:
                        time.sleep(WORKER_CRASH_RETRY_S)
                        if registry is not None:
                            with persist_lock:
                                _sync_from_chunks()
                        chunk_attempts[ci] += 1
                        print(
                            f"    chunk {ci} GPU {gpu_for_chunk[ci]}: "
                            f"restart (attempt {chunk_attempts[ci]})",
                            flush=True,
                        )
                        nf = _submit_chunk_worker(
                            executors[gpu_for_chunk[ci]],
                            ci,
                            chunk_paths,
                            gpu_for_chunk,
                            cfg_dict,
                            stagger_s,
                            gpu_launch_idx,
                        )
                        futures[nf] = ci
                    else:
                        out["results"] = _load_chunk_results(chunk_paths[ci])
                        worker_outcomes.append(out)
                        all_results.extend(out["results"])

                if registry is not None:
                    with persist_lock:
                        n_saved = _sync_from_chunks()
                    n_pending = len(
                        pending_tasks(
                            queue, cfg.outdir, round_idx, registry=registry,
                        ),
                    )
                    _print_sync_progress(
                        n_saved,
                        chunk_paths,
                        gpu_for_chunk,
                        batch_total=batch_total,
                        pending=n_pending,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\n  ⚠️  Sim interrupted; saving progress from disk...", flush=True)
        raise
    finally:
        for ex in executors.values():
            ex.shutdown(wait=False, cancel_futures=True)
        reporter.stop()
        reporter.join(timeout=1.0)
        if registry is not None:
            with persist_lock:
                sync_queue_and_registry_from_chunks(
                    cfg.outdir,
                    round_idx,
                    registry,
                    queue,
                    registry_path=sim_paths["registry"],
                    task_queue_path=sim_paths["task_queue"],
                    persist=True,
                )
            n_pending = len(
                pending_tasks(queue, cfg.outdir, round_idx, registry=registry),
            )
            if interrupted:
                print(
                    f"  Chunk sync saved ({n_pending} pending sim)",
                    flush=True,
                )

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


def auto_refill_pool(
    cfg: PoolSimConfig,
    round_idx: int,
    *,
    sim_complete: bool,
) -> bool:
    threshold = compute_median_success_threshold(cfg.merged_dir)
    refill_targets = [
        obj_id
        for obj_id, _, _ in select_target_objects(cfg.merged_dir, threshold)
    ]
    print(f"\n🔄 Pool exhausted — auto refill (success_threshold=median={threshold})")
    print(f"   refill {len(refill_targets)} object(s) → {cfg.pool_dir}")
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

    paths = paths_for_outdir(cfg.outdir, round_idx)
    registry = load_registry(paths["registry"])
    n_cleared = clear_registry_for_objects(registry, refill_targets)
    save_registry(paths["registry"], registry)
    print(
        f"✅ Pool refill finished; cleared registry for {n_cleared}/{len(refill_targets)} "
        f"refilled object(s)",
    )

    if not sim_complete:
        task_queue_path = paths["task_queue"]
        if os.path.isfile(task_queue_path):
            os.remove(task_queue_path)
            print(
                f"   removed {task_queue_path} (sim incomplete; --resume will replan "
                f"round {round_idx} from new pool)",
            )

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
            equal_object_prob=cfg.equal_object_prob,
            max_success_per_object=cfg.max_success_per_object,
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
            f"({len(queue['tasks'])} sim tasks)  exhausted={queue['pool_exhausted']}  "
            f"sampling={queue.get('object_sampling', 'weighted')}",
        )
    else:
        sync_queue_and_registry_from_chunks(
            cfg.outdir,
            round_idx,
            registry,
            queue,
            registry_path=paths["registry"],
            task_queue_path=paths["task_queue"],
            persist=True,
        )
        n_pending = len(
            pending_tasks(
                queue,
                cfg.outdir,
                round_idx,
                registry=registry,
            ),
        )
        print(f"  Resume task queue: {n_pending} pending sim (chunk has no record)")

    pool_exhausted = bool(queue.get("pool_exhausted"))

    sim_complete_before = is_queue_complete(
        queue,
        cfg.outdir,
        round_idx,
        registry=registry,
    )
    if not sim_complete_before:
        sim_results, _ = run_sim_phase(queue, cfg, round_idx, registry=registry)
    else:
        sim_results = []

    sync_status = sync_queue_and_registry_from_chunks(
        cfg.outdir,
        round_idx,
        registry,
        queue,
        registry_path=paths["registry"],
        task_queue_path=paths["task_queue"],
        persist=True,
    )
    if not sync_status.get("sync_ok", False):
        print(
            f"  ⚠️  Chunk↔queue sync not clean: ingest_gap={sync_status.get('n_ingest_gap')} "
            f"unreadable_chunk_files={sync_status.get('n_failed_chunk_files')}",
            flush=True,
        )
        if sync_status.get("failed_chunk_files"):
            for fp in sync_status["failed_chunk_files"][:5]:
                print(f"      {fp}", flush=True)
    else:
        print(
            f"  Chunk sync OK: {sync_status['n_chunk_tasks']} on disk, "
            f"{sync_status['n_pending']} pending sim, "
            f"{sync_status['n_sim_missing']} never written to chunk"
            + (
                f", {sync_status.get('n_synthesized_skipped', 0)} synthesized skipped"
                if sync_status.get("n_synthesized_skipped")
                else ""
            ),
            flush=True,
        )

    objs = sorted({t["obj_id"] for t in queue.get("tasks", [])})
    dataset_by_obj = _dataset_by_obj(objs)
    summary_rows: list[dict] = []

    for obj_id in objs:
        n_cand = _count_round_candidates(paths["cand_round_dir"], obj_id)
        n_ok = _count_round_success(paths["gt_round_dir"], obj_id)
        gt_path = os.path.join(paths["gt_round_dir"], f"{obj_id}_robot_gt.hdf5")
        grasp_path = os.path.join(paths["cand_round_dir"], f"{obj_id}_grasp.hdf5")
        merged = _merge_for_round(cfg, obj_id, round_idx)
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
    sim_complete = bool(sync_status.get("sim_complete", False))
    n_pending = int(sync_status.get("n_pending", 0))
    return {
        "round_idx": round_idx,
        "pool_exhausted": pool_exhausted,
        "slots_planned": queue.get("slots_planned", 0),
        "n_tasks": len(queue.get("tasks", [])),
        "n_sim_success": n_success_tasks,
        "n_pending_tasks": n_pending,
        "n_chunk_tasks": sync_status.get("n_chunk_tasks", 0),
        "sync_ok": bool(sync_status.get("sync_ok", False)),
        "sim_complete": sim_complete,
        "queue_complete": sim_complete,
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
    parser.add_argument(
        "--same-gpu-stagger-s",
        type=float,
        default=SAME_GPU_STAGGER_S,
        help="同 GPU 上相邻 Isaac worker 启动间隔秒数 (默认 45)",
    )
    parser.add_argument(
        "--isaac-startup-slots-per-gpu",
        type=int,
        default=ISAAC_STARTUP_SLOTS_PER_GPU,
        metavar="N",
        help="同 GPU 同时进行 Isaac 冷启动 (至 Startup Complete) 的上限; 0=不限制 (默认 2)",
    )
    parser.add_argument("--sim-timeout", type=int, default=7200, help="单个 worker chunk 超时秒")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--plan-seed", type=int, default=None)
    parser.add_argument(
        "--equal-object-prob",
        action="store_true",
        help="每轮抽 slot 时 pool 内每个 eligible 物体等概率（默认按 merged 成功 1/(n+1) 加权）",
    )
    parser.add_argument(
        "--max-success-per-object",
        type=int,
        default=None,
        metavar="N",
        help="merged 内成功数 ≥ N 的物体本轮规划 prob=0（读 merged n_successful；默认不限制）",
    )
    parser.add_argument("--pool-target", type=int, default=50, help="auto-refill 时每物体 target")
    parser.add_argument("--score-threshold", type=float, default=70.0)
    parser.add_argument("--no-auto-refill", action="store_true")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument(
        "--no-early-stop-yaw-on-success",
        action="store_true",
        help="禁用：candidate 任一 yaw 成功后仍 sim 其余 yaw（默认开启 early-stop）",
    )
    parser.add_argument(
        "--merge-deduplicate", action="store_true",
        help="合并 merged 时去重 (默认不去重)",
    )
    parser.add_argument(
        "--incremental-merge",
        action="store_true",
        help="merged 增量更新：读已有 merged + 仅本轮 robot_gt（无需历史 round_*）",
    )
    parser.add_argument(
        "--full-merge",
        action="store_true",
        help="强制全量扫描 robot_gt/round_* 合并（覆盖 --incremental-merge）",
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
    if args.isaac_startup_slots_per_gpu < 0:
        print("❌ --isaac-startup-slots-per-gpu 须 >= 0")
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

    if args.max_success_per_object is not None and args.max_success_per_object < 0:
        print("❌ --max-success-per-object 须 >= 0")
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
        same_gpu_stagger_s=args.same_gpu_stagger_s,
        isaac_startup_slots_per_gpu=args.isaac_startup_slots_per_gpu,
        merge_deduplicate=args.merge_deduplicate,
        incremental_merge=args.incremental_merge and not args.full_merge,
        slots_per_round=args.slots_per_round,
        plan_seed=args.plan_seed,
        equal_object_prob=args.equal_object_prob,
        max_success_per_object=args.max_success_per_object,
        pool_target=args.pool_target,
        auto_refill=not args.no_auto_refill,
        score_threshold=args.score_threshold,
        no_rotation=not args.rotation,
        early_stop_on_candidate_success=not args.no_early_stop_yaw_on_success,
    )

    state_path = os.path.join(outdir, "state.json")
    state = _load_state(state_path) if args.resume else {"round": 0, "objects": {}}

    print(f"Out: {outdir}")
    print(f"Pool: {pool_dir}")
    print(f"Merged: {merged_dir}")
    print(
        f"Merge mode: {'incremental (merged + this round gt)' if cfg.incremental_merge else 'full scan robot_gt/round_*'}",
    )
    print(f"Max rounds (sync_ok gate): {args.max_rounds}  state.round: {state.get('round', 0)}")
    print(
        f"Object sampling: {'equal' if args.equal_object_prob else 'weighted (1/(success+1))'}"
        + (
            f"  max_success_per_object={args.max_success_per_object}"
            if args.max_success_per_object is not None
            else ""
        ),
    )
    print(
        f"Sim: {len(sim_gpu_ids)} GPU × {args.sim_per_gpu}/GPU = "
        f"{len(sim_gpu_ids) * args.sim_per_gpu} workers",
    )

    rounds_advanced = 0
    while rounds_advanced < args.max_rounds:
        r = int(state.get("round", 0))
        print(f"\n{'='*60}")
        print(f"Round {r} ({round_tag(r)})")
        meta = run_one_round(cfg, r, resume=args.resume)
        if meta.get("sync_ok", False):
            state["round"] = r + 1
            _save_state(state_path, state)
            rounds_advanced += 1
        else:
            state["round"] = r
            _save_state(state_path, state)
            print(
                f"⚠️  Round {r}: chunk↔queue sync failed; "
                f"state.round stays at {r} (re-run with --resume to retry sync)",
                flush=True,
            )
            break

        if not meta.get("sim_complete", False):
            print(
                f"  Note: {meta.get('n_pending_tasks', '?')} task(s) have no chunk record "
                f"(sim not finished); advanced because chunk and queue match.",
                flush=True,
            )

        if meta["pool_exhausted"] and cfg.auto_refill:
            auto_refill_pool(
                cfg,
                r,
                sim_complete=bool(meta.get("sim_complete", False)),
            )
        elif meta["pool_exhausted"]:
            print("⚠️  Pool exhausted (--no-auto-refill); stop after this round")

        if meta["slots_planned"] == 0:
            print("⚠️  No slots planned (no eligible candidates); stopping")
            break

    print(f"\n✅ Done. state.round={state['round']}")
    print(f"   summary: {os.path.join(outdir, 'summary.csv')}")


if __name__ == "__main__":
    main()
