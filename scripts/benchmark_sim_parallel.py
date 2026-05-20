#!/usr/bin/env python3
"""
benchmark_sim_parallel.py — 探测单张 GPU 上最多能同时跑几个 Isaac sim
======================================================================
逐步增加并行数，用 nvidia-smi 监控显存；要求运行期间 **剩余显存 ≥ min_free_gb**
（默认 3GB），返回满足条件的最大并行数（可作为 --sim-per-gpu 参考）。

前置:
  - export ISAAC_SIM_PATH=...
  - 已有 A01001 候选: output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5
    或自行 --hdf5

用法:
    conda activate bundlesdf   # 仅用于启动本脚本；sim 子进程用 Isaac python.sh
    export ISAAC_SIM_PATH=/home/vision/isaacsim

    python3 scripts/benchmark_sim_parallel.py
    python3 scripts/benchmark_sim_parallel.py --start-n 6 --max-try 10
    python3 scripts/benchmark_sim_parallel.py --gpu 0 --min-free-gb 3
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_SCRIPT = os.path.join(PROJ, "sim", "run_grasp_sim.py")
DEFAULT_HDF5 = os.path.join(
    PROJ, "output", "grasp_collect", "candidates", "round_0000", "A01001_grasp.hdf5"
)
BENCH_OUT = os.path.join(PROJ, "output", "benchmark_sim_parallel")


@dataclass
class TrialResult:
    n_parallel: int
    ok: bool
    min_free_mb: float
    peak_used_mb: float
    baseline_free_mb: float
    baseline_used_mb: float
    all_running_at_peak: bool
    early_exit: bool
    stabilize_sec: float
    note: str = ""


def _isaac_python(explicit: Optional[str]) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    base = os.environ.get("ISAAC_SIM_PATH", "").rstrip("/")
    path = os.path.join(base, "python.sh") if base else ""
    if not os.path.isfile(path):
        sys.exit(
            "❌ Isaac python.sh 未找到。请 export ISAAC_SIM_PATH 或传 --isaac-python"
        )
    return path


def _query_gpu_mb(gpu_id: int) -> tuple[float, float, float]:
    """返回 (used_mb, free_mb, total_mb)。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=index,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"❌ nvidia-smi 失败: {e}")

    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 4:
        sys.exit(f"❌ 无法解析 nvidia-smi 输出: {out!r}")
    # index, used, free, total
    used = float(parts[1])
    free = float(parts[2])
    total = float(parts[3])
    return used, free, total


def _kill_procs(procs: list[subprocess.Popen]):
    for p in procs:
        if p.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for p in procs:
        if p.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _launch_sim(
    isaac_py: str,
    hdf5: str,
    result_dir: str,
    gpu_id: int,
    max_candidates: int,
    headless: bool,
) -> subprocess.Popen:
    os.makedirs(result_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        isaac_py, SIM_SCRIPT,
        "--hdf5", hdf5,
        "--result-dir", result_dir,
        "--save-result",
        "--max-candidates", str(max_candidates),
    ]
    if headless:
        cmd.append("--headless")
    return subprocess.Popen(
        cmd,
        cwd=PROJ,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _run_trial(
    n: int,
    isaac_py: str,
    hdf5: str,
    gpu_id: int,
    min_free_mb: float,
    stabilize_sec: float,
    poll_interval: float,
    max_candidates: int,
    headless: bool,
    work_dir: str,
) -> TrialResult:
    baseline_used, baseline_free, _ = _query_gpu_mb(gpu_id)
    result_dir = os.path.join(work_dir, f"n{n}")
    procs = [
        _launch_sim(
            isaac_py, hdf5,
            os.path.join(result_dir, f"w{i}"),
            gpu_id, max_candidates, headless,
        )
        for i in range(n)
    ]

    min_free = baseline_free
    peak_used = baseline_used
    deadline = time.time() + stabilize_sec
    early_exit = False
    all_running = True

    try:
        while time.time() < deadline:
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                early_exit = True
                all_running = False
            used, free, _ = _query_gpu_mb(gpu_id)
            min_free = min(min_free, free)
            peak_used = max(peak_used, used)
            time.sleep(poll_interval)
        else:
            all_running = all(p.poll() is None for p in procs)
    finally:
        _kill_procs(procs)
        time.sleep(2)  # 等待显存释放

    ok = min_free >= min_free_mb
    note = ""
    if early_exit:
        note = "部分 sim 提前退出，读数可能偏低"
    return TrialResult(
        n_parallel=n,
        ok=ok,
        min_free_mb=min_free,
        peak_used_mb=peak_used,
        baseline_free_mb=baseline_free,
        baseline_used_mb=baseline_used,
        all_running_at_peak=all_running,
        early_exit=early_exit,
        stabilize_sec=stabilize_sec,
        note=note,
    )


def main():
    parser = argparse.ArgumentParser(
        description="探测单 GPU 最大 Isaac sim 并行数（显存约束）",
    )
    parser.add_argument("--obj", default="A01001")
    parser.add_argument("--hdf5", default=None, help="候选 HDF5（默认 round_0000/{obj}_grasp.hdf5）")
    parser.add_argument("--gpu", type=int, default=0, help="物理 GPU 编号")
    parser.add_argument(
        "--min-free-gb", type=float, default=3.0,
        help="并行 sim 运行期间 GPU 至少保留的空闲显存 (GB)",
    )
    parser.add_argument(
        "--start-n", type=int, default=6,
        help="从该并行数开始向上探测 (默认 6，跳过 1~5 以节省时间)",
    )
    parser.add_argument("--max-try", type=int, default=10, help="最多尝试的并行数上限")
    parser.add_argument(
        "--no-probe-down", action="store_true",
        help="若 start-n 失败则不向下探测 1..start-n-1",
    )
    parser.add_argument(
        "--stabilize-sec", type=float, default=50.0,
        help="每个并行档位启动后观察显存的秒数",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=1,
                        help="每次 sim 只验证 1 个候选，加快压测")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--isaac-python", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    hdf5 = args.hdf5
    if not hdf5:
        hdf5 = os.path.join(
            PROJ, "output", "grasp_collect", "candidates", "round_0000",
            f"{args.obj}_grasp.hdf5",
        )
    if not os.path.isfile(hdf5):
        sys.exit(f"❌ HDF5 不存在: {hdf5}\n   请先跑 sampler 或指定 --hdf5")

    isaac_py = _isaac_python(args.isaac_python)
    min_free_mb = args.min_free_gb * 1024.0
    work_dir = os.path.join(BENCH_OUT, f"gpu{args.gpu}_{args.obj}")
    os.makedirs(work_dir, exist_ok=True)

    used0, free0, total0 = _query_gpu_mb(args.gpu)
    print(f"GPU {args.gpu}: total={total0:.0f} MB  idle used={used0:.0f} free={free0:.0f}")
    print(f"约束: 运行中 free >= {min_free_mb:.0f} MB ({args.min_free_gb} GB)")
    print(f"HDF5: {hdf5}")
    print(f"Isaac: {isaac_py}")
    print(f"探测: n={args.start_n}..{args.max_try}  观察 {args.stabilize_sec}s/档\n")

    if args.start_n < 1 or args.start_n > args.max_try:
        sys.exit(f"❌ --start-n 须在 1..--max-try 内 (got {args.start_n})")

    trials: list[TrialResult] = []
    max_ok = 0

    def run_one(n: int, label: str) -> TrialResult:
        print(f"--- {label} n={n} ---")
        tr = _run_trial(
            n, isaac_py, hdf5, args.gpu, min_free_mb,
            args.stabilize_sec, args.poll_interval,
            args.max_candidates, args.headless, work_dir,
        )
        trials.append(tr)
        status = "OK" if tr.ok else "FAIL"
        print(
            f"  {status}  min_free={tr.min_free_mb:.0f} MB  "
            f"peak_used={tr.peak_used_mb:.0f} MB  "
            f"(baseline free={tr.baseline_free_mb:.0f})"
        )
        if tr.note:
            print(f"  note: {tr.note}")
        time.sleep(5)
        return tr

    # 向上：start_n → max_try
    for n in range(args.start_n, args.max_try + 1):
        tr = run_one(n, "向上")
        if tr.ok:
            max_ok = n
        else:
            print(f"\n显存不足 (free < {args.min_free_gb} GB)，停止加开。")
            break

    # 若从 start_n 就失败，向下找最大可用档
    if max_ok == 0 and args.start_n > 1 and not args.no_probe_down:
        print(f"\n--- start-n={args.start_n} 失败，向下探测 ---")
        for n in range(args.start_n - 1, 0, -1):
            tr = run_one(n, "向下")
            if tr.ok:
                max_ok = n
                break

    print(f"\n{'='*50}")
    print(f"推荐 --sim-per-gpu (GPU {args.gpu}): {max_ok}")
    print(f"  条件: 并行时剩余显存 ≥ {args.min_free_gb} GB")
    if max_ok == 0:
        print("  ⚠️  无满足条件的并行数；请降低 --min-free-gb 或减小 --start-n")

    report = {
        "obj": args.obj,
        "hdf5": os.path.abspath(hdf5),
        "gpu_id": args.gpu,
        "min_free_gb": args.min_free_gb,
        "start_n": args.start_n,
        "max_sim_per_gpu": max_ok,
        "trials": [asdict(t) for t in trials],
    }
    out_json = args.out_json or os.path.join(
        work_dir, f"result_min_free_{args.min_free_gb}gb.json"
    )
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"报告: {out_json}")


if __name__ == "__main__":
    main()
