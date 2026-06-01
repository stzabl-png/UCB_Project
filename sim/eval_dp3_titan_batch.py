#!/usr/bin/env python3
"""Batch eval wrapper for DP3 with TITAN PROTOCOL on gate3 sim.

Calls sim/eval_dp3_titan_protocol.py once per (obj, trial) — one IsaacSim
subprocess each (mirrors titan eval_batch.py canonical protocol). Aggregates
per-episode JSONs into a batch_summary.json that uses the same schema as titan
eval_pool's _build_eval_summary (by_object + by_yaw + by_object_yaw + failure
stages + overall success_rate) so DP3 numbers can be directly compared to main
method results.

Per-episode JSON layout matches evaluation/results.py:write_episode_json schema:
    {schema_version, episode_id, obj_id, policy, success, failure_stage,
     z_delta_m, scene, policy_output, execution, video_path}

Usage:
    /home/accelerator/miniforge3/envs/env_isaaclab/bin/python \\
        sim/eval_dp3_titan_batch.py \\
        --obj-ids A02014,A02021 \\
        --trials-per-object 5 \\
        --z-yaw-deg 0 \\
        --server-url http://127.0.0.1:8765 \\
        --result-dir output/eval_dp3_titan_protocol/batch_$(date +%Y%m%d_%H%M)
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP3 titan-protocol batch eval (gate3 sim)")
    p.add_argument("--obj-ids", required=True,
                   help="Comma-separated obj_ids, e.g. A02014,A02021")
    p.add_argument("--trials-per-object", type=int, default=5,
                   help="trials per obj (titan canonical = 5). Each trial = 1 IsaacSim subprocess.")
    p.add_argument("--z-yaw-deg", type=float, default=0.0,
                   help="Fixed z-yaw for all trials (matches partner current protocol).")
    p.add_argument("--dataset", default="oakink")
    p.add_argument("--policy", default="dp3_titan_protocol")
    p.add_argument("--server-url", default="http://127.0.0.1:8765")
    p.add_argument("--max-chunks", type=int, default=5)
    p.add_argument("--success-dz-m", type=float, default=0.03)
    p.add_argument("--retry-physx", type=int, default=1)
    p.add_argument("--episodes-glob-template",
                   default="Baseline1/data/episodes_b3_v4_oakink89_2026-05-26/oakink__{OBJ}_*.hdf5",
                   help="glob pattern with {OBJ} placeholder per-obj.")
    p.add_argument("--titan-usd-dir",
                   default="/home/accelerator/UCB_Project_titan/output/obj_usd/oakink")
    p.add_argument("--titan-mesh-root",
                   default="/home/accelerator/UCB_Project_titan/data_hub/meshes/SAM3DMesh/rotated_mesh")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--python", default="/home/accelerator/miniforge3/envs/env_isaaclab/bin/python")
    p.add_argument("--record-video", action="store_true",
                   help="Record per-trial video (default OFF for batch speed)")
    p.add_argument("--ep-timeout-sec", type=int, default=600)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel IsaacSim workers. Each ep is one subprocess; "
                        "N workers run N eps concurrently. Limited by GPU memory "
                        "(~3-4 GB per worker). DP3 server queues requests so safe to scale.")
    p.add_argument("--skip-existing", action="store_true",
                   help="If set, any (obj, trial) whose per-ep JSON already exists in "
                        "--result-dir is skipped (its existing result is loaded into the "
                        "summary). Use to resume an interrupted batch without recomputing.")
    return p.parse_args()


def _success_rate(n_s: int, n_t: int) -> float:
    return (n_s / n_t * 100.0) if n_t else 0.0


def _aggregate_counts(rows: list[dict], key_fn) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        k = str(key_fn(r))
        item = out.setdefault(k, {"total": 0, "success": 0, "success_rate": 0.0})
        item["total"] += 1
        item["success"] += int(bool(r.get("success")))
    for it in out.values():
        it["success_rate"] = _success_rate(it["success"], it["total"])
    return dict(sorted(out.items()))


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("success"):
            continue
        k = str(r.get("failure_stage") or "unknown")
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _normalize_ep_row(gate3_result: dict, *, obj_id: str, episode_id: str,
                      policy: str, z_yaw_deg: float, args: argparse.Namespace,
                      titan_meta: dict | None,
                      result_dir: Path) -> dict:
    """Convert gate3 rollout output → titan-schema per-episode dict + write JSON."""
    success = bool(gate3_result.get("success", False))
    dz = gate3_result.get("dz")
    stage = gate3_result.get("stage")
    failure_stage = None if success else (stage or "unknown")

    rec = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_id": episode_id,
        "obj_id": obj_id,
        "policy": policy,
        "success": success,
        "failure_stage": failure_stage,
        "z_delta_m": float(dz) if dz is not None else None,
        "z_yaw_deg": float(z_yaw_deg),
        "scene": {
            "obj_id": obj_id,
            "dataset": args.dataset,
            "policy": policy,
            "z_yaw_deg": float(z_yaw_deg),
            "object_scale": 1.0,
            "placement": (titan_meta or {}),
        },
        "policy_output": {
            "kind": "closed_loop_actions",
            "metadata": {"policy_name": policy, "server_url": args.server_url,
                         "protocol": "titan-protocol (identity quat + SAM3D USD + live PC)"},
        },
        "execution": {
            "success": success,
            "z_delta_m": float(dz) if dz is not None else None,
            "failure_stage": failure_stage,
            "stage_raw": stage,
            "min_dist_to_obj_cm": gate3_result.get("min_dist_to_obj_cm"),
            "n_chunks": gate3_result.get("n_chunks"),
            "n_executed": gate3_result.get("n_executed"),
            "grip_signal_idx": gate3_result.get("grip_signal_idx"),
            "init_ee_err_mm": gate3_result.get("init_ee_err_mm"),
            "ycb_class_id": gate3_result.get("ycb_class_id"),
            "source_hdf5": gate3_result.get("name"),
        },
        "video_path": None,
    }
    out_path = result_dir / f"{episode_id}.json"
    with out_path.open("w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    return rec


def _build_titan_summary(*, args, result_dir, obj_ids, rows, n_success):
    n_total = len(rows)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir),
        "protocol": "dp3_titan_protocol (gate3 sim + titan placement/PC)",
        "success": {
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": _success_rate(n_success, n_total),
        },
        "inputs": {
            "objects": obj_ids,
            "trials_per_obj_yaw": int(args.trials_per_object),
            "z_yaw_deg": float(args.z_yaw_deg),
            "z_yaw_grid": None,
            "z_yaw_random": False,
            "policy": args.policy,
            "server_url": args.server_url,
            "record_video": bool(args.record_video),
        },
        "counts": {
            "n_objects": len(obj_ids),
            "n_tasks": int(len(obj_ids)) * int(args.trials_per_object),
            "n_results": n_total,
            "n_recorded": sum(1 for r in rows if r.get("video_path")),
        },
        "by_object": _aggregate_counts(rows, lambda r: r.get("obj_id", "unknown")),
        "by_yaw": _aggregate_counts(rows, lambda r: int(round(float(r.get("z_yaw_deg", 0.0)))) % 360),
        "by_object_yaw": _aggregate_counts(
            rows,
            lambda r: f"{r.get('obj_id','unknown')}_yaw{int(round(float(r.get('z_yaw_deg',0.0)))) % 360:03d}",
        ),
        "failure_stages": _failure_counts(rows),
        "episodes": rows,
    }


def main() -> None:
    args = _parse_args()
    obj_ids = [s.strip() for s in args.obj_ids.split(",") if s.strip()]
    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    sub_out_root = result_dir / "_subprocess_outputs"
    sub_out_root.mkdir(exist_ok=True)

    PROJ = Path(__file__).resolve().parents[1]
    EVAL = str(PROJ / "sim" / "eval_dp3_titan_protocol.py")
    TITAN_USD_DIR = args.titan_usd_dir

    yaw = float(args.z_yaw_deg)
    yaw_tag = f"yaw{int(round(yaw)) % 360:03d}"

    rows: list[dict] = []
    n_success = 0
    print(f"[batch] start  obj={len(obj_ids)}  trials/obj={args.trials_per_object}  "
          f"z_yaw={yaw}  workers={args.n_workers}  "
          f"total_ep={len(obj_ids)*args.trials_per_object}", flush=True)

    # Build job list
    jobs: list[dict] = []
    for obj_id in obj_ids:
        meta_path = os.path.join(TITAN_USD_DIR, f"{obj_id}_meta.json")
        titan_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else None
        ep_glob = args.episodes_glob_template.replace("{OBJ}", obj_id)
        matches = sorted(_glob.glob(os.path.join(str(PROJ), ep_glob)))
        if not matches:
            print(f"[batch] {obj_id}: NO training hdf5 matches {ep_glob} — skip", flush=True)
            continue
        ep_path = matches[0]
        for trial in range(args.trials_per_object):
            episode_id = f"{obj_id}_{args.policy}_t{trial}_{yaw_tag}"
            jobs.append({
                "obj_id": obj_id, "trial": trial, "episode_id": episode_id,
                "ep_path": ep_path, "titan_meta": titan_meta,
            })

    def _run_one_job(job: dict) -> dict:
        episode_id = job["episode_id"]
        # ★ skip-existing: if final per-ep JSON already in result_dir, load + return
        if args.skip_existing:
            existing = result_dir / f"{episode_id}.json"
            if existing.exists():
                try:
                    rec = json.load(open(existing))
                    rec["_rc"] = 0
                    rec["_skipped_existing"] = True
                    return rec
                except Exception:
                    pass  # fall through to re-run if existing is corrupt
        sub_out_dir = sub_out_root / episode_id
        sub_out_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{episode_id}.log"
        cmd = [
            args.python, "-u", EVAL,
            "--episodes-glob", job["ep_path"],
            "--n-rollouts", "1",
            "--seed", str(job["trial"]),
            "--max-chunks", str(args.max_chunks),
            "--server-url", args.server_url,
            "--result-dir", str(sub_out_dir),
            "--retry-physx", str(args.retry_physx),
            "--titan-protocol",
            "--titan-usd-dir", args.titan_usd_dir,
            "--titan-mesh-root", args.titan_mesh_root,
            "--headless",
        ]
        if args.record_video:
            cmd += ["--video", str(sub_out_dir), "--video-all"]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w") as flog:
            try:
                proc = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT,
                                      env=env, timeout=args.ep_timeout_sec)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = -9
        # Parse gate3 output JSON
        gate3_jsons = sorted(sub_out_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
        if gate3_jsons:
            gate3 = json.load(open(gate3_jsons[-1]))
            results = gate3.get("results", [])
            ep = results[0] if results else {"success": False, "dz": None, "stage": "no_result",
                                              "name": os.path.basename(job["ep_path"])}
        else:
            ep = {"success": False, "dz": None, "stage": "no_output",
                  "name": os.path.basename(job["ep_path"])}
        video_glob = list(sub_out_dir.glob("*.mp4"))
        video_path = str(video_glob[0]) if video_glob else None
        rec = _normalize_ep_row(ep, obj_id=job["obj_id"], episode_id=episode_id,
                                policy=args.policy, z_yaw_deg=yaw,
                                args=args, titan_meta=job["titan_meta"],
                                result_dir=result_dir)
        if video_path:
            rec["video_path"] = video_path
        rec["_rc"] = rc
        return rec

    # Execute jobs — sequential if n_workers=1, else thread-pool parallel
    if args.n_workers <= 1:
        for job in jobs:
            print(f"\n[batch] === {job['episode_id']} ===", flush=True)
            rec = _run_one_job(job)
            rows.append(rec)
            if rec["success"]:
                n_success += 1
            print(f"[batch] → {job['episode_id']}  success={rec['success']}  "
                  f"dz={rec['z_delta_m']}  stage={rec['failure_stage']}  rc={rec.pop('_rc')}", flush=True)
            summary = _build_titan_summary(args=args, result_dir=result_dir,
                                           obj_ids=obj_ids, rows=rows, n_success=n_success)
            with (result_dir / "batch_summary.json").open("w") as f:
                json.dump(summary, f, indent=2); f.write("\n")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[batch] launching {len(jobs)} jobs across {args.n_workers} workers ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {ex.submit(_run_one_job, j): j for j in jobs}
            for fut in as_completed(futs):
                rec = fut.result()
                rows.append(rec)
                if rec["success"]:
                    n_success += 1
                print(f"[batch] → {rec['episode_id']}  success={rec['success']}  "
                      f"dz={rec['z_delta_m']}  stage={rec['failure_stage']}  rc={rec.pop('_rc')}  "
                      f"({len(rows)}/{len(jobs)} done)", flush=True)
                summary = _build_titan_summary(args=args, result_dir=result_dir,
                                               obj_ids=obj_ids, rows=rows, n_success=n_success)
                with (result_dir / "batch_summary.json").open("w") as f:
                    json.dump(summary, f, indent=2); f.write("\n")

    print(f"\n[batch] DONE  {n_success}/{len(rows)} success ({_success_rate(n_success, len(rows)):.1f}%)", flush=True)
    print(f"[batch] summary: {result_dir / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
