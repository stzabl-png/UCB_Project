"""Generate DP3 evaluation "solution" JSONs for partner's task-queue infra.

Partner's evaluation pipeline (evaluation/eval_pool.py, eval_batch.py, etc.)
expects each task to have a ``solution_path`` pointing to a JSON file with a
``policy_output`` payload. For closed-loop online policies (like DP3) the
JSON is essentially a static "policy descriptor" — it carries the server
URL + chunk hyper-params; the actual policy is queried at execute time.

This helper writes one such JSON per (obj_id, yaw, trial) task. Use as:

    python -m evaluation.policies.dp3_solution_gen \\
        --out-dir /tmp/dp3_tasks \\
        --server-url http://127.0.0.1:8765 \\
        --tasks-spec my_tasks.json

The actual task-queue spec format is partner's; see evaluation/task_queue.py.
This module only generates the per-task ``solution_path`` JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def make_dp3_solution_dict(
    *,
    server_url: str = "http://127.0.0.1:8765",
    max_chunks: int = 5,
    success_dz_m: float = 0.03,
    retry_physx: int = 1,
    n_pc_points: int = 4096,
    request_timeout: int = 60,
    dataset: str = "",
    candidate_hdf5: str = "",
) -> dict:
    """Build a 'solution' dict that ``_policy_output_from_solution`` accepts.

    Returns a dict with:
      - policy: "dp3_online"
      - policy_output.kind: "closed_loop_actions"
      - policy_output.actions: HTTP server endpoint + chunk hyper-params
    """
    return {
        "policy":    "dp3_online",
        "dataset":   dataset,
        "candidate_hdf5": candidate_hdf5,
        "policy_output": {
            "kind":     "closed_loop_actions",
            "command":  None,
            "actions": {
                "server_url":      server_url,
                "max_chunks":      max_chunks,
                "success_dz_m":    success_dz_m,
                "retry_physx":     retry_physx,
                "n_pc_points":     n_pc_points,
                "request_timeout": request_timeout,
            },
            "metadata": {"policy_name": "dp3_online", "server_url": server_url},
        },
    }


def write_dp3_solution_json(path: str, **kwargs) -> str:
    """Write a single DP3 task JSON. Returns absolute path."""
    sol = make_dp3_solution_dict(**kwargs)
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(sol, f, indent=2, sort_keys=True)
        f.write("\n")
    return str(p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DP3 solution JSONs for partner task-queue")
    p.add_argument("--out-dir", required=True, help="directory to write JSONs")
    p.add_argument("--server-url", default="http://127.0.0.1:8765")
    p.add_argument("--max-chunks", type=int, default=5)
    p.add_argument("--success-dz-m", type=float, default=0.03)
    p.add_argument("--retry-physx", type=int, default=1)
    p.add_argument("--n-pc-points", type=int, default=4096)
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument(
        "--filename", default="dp3_solution.json",
        help="output filename (single solution file). Most callers will want to "
             "iterate per-task by overriding this.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out_path = os.path.join(args.out_dir, args.filename)
    abs_path = write_dp3_solution_json(
        out_path,
        server_url=args.server_url,
        max_chunks=args.max_chunks,
        success_dz_m=args.success_dz_m,
        retry_physx=args.retry_physx,
        n_pc_points=args.n_pc_points,
        request_timeout=args.request_timeout,
    )
    print(f"wrote DP3 solution → {abs_path}")


if __name__ == "__main__":
    main()
