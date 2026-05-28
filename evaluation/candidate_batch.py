"""Parallel candidate generation for eval_pool."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from evaluation.eval_single import default_candidate_python, resolve_generate_mesh

PROJ = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_gpu_ids(text: str | None) -> list[str]:
    if not text:
        return ["0"]
    return [part.strip() for part in str(text).split(",") if part.strip()] or ["0"]


def _split_even(items: list[dict], n_chunks: int) -> list[list[dict]]:
    chunks = [[] for _ in range(max(1, n_chunks))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [c for c in chunks if c]


def build_candidate_tasks(
    *,
    obj_ids: list[str],
    yaw_values_by_obj: dict[str, list[float]],
    trials_per_obj_yaw: int,
    result_dir: Path,
    mesh_root: str,
    dataset: str | None,
) -> list[dict]:
    tasks: list[dict] = []
    for obj_id in obj_ids:
        for yaw in yaw_values_by_obj[obj_id]:
            yaw_tag = int(round(float(yaw))) % 360
            out_dir = result_dir / "candidates" / obj_id
            out_path = out_dir / f"{obj_id}_yaw{yaw_tag:03d}_pool_grasp.hdf5"
            probe = type(
                "Probe",
                (),
                {
                    "obj_id": obj_id,
                    "dataset": dataset,
                    "mesh": None,
                    "mesh_root": mesh_root,
                    "sam3d_rotated_mesh": False,
                },
            )()
            mesh_path, _ = resolve_generate_mesh(probe)
            tasks.append(
                {
                    "obj_id": obj_id,
                    "z_yaw_deg": float(yaw),
                    "target_candidates": int(trials_per_obj_yaw),
                    "mesh_path": str(mesh_path),
                    "output_hdf5": str(out_path.resolve()),
                }
            )
    return tasks


def run_candidate_batch_generation(
    *,
    tasks: list[dict],
    result_dir: Path,
    mesh_root: str,
    dataset: str | None,
    candidate_python: str | None,
    candidate_gpu_ids: str | None,
    batch_multiplier: int,
    max_batches: int,
    object_scale: float,
    no_hard_gate: bool = False,
    pdm_checkpoint: str | Path | None = None,
    pose_stats: str | Path | None = None,
    affordance_checkpoint: str | Path | None = None,
    candidate_workers: int | None = None,
    candidate_per_gpu: int | None = None,
) -> dict[tuple[str, float], str]:
    if not tasks:
        return {}
    gpu_ids = parse_gpu_ids(candidate_gpu_ids)
    if candidate_workers is not None:
        n_workers = max(1, int(candidate_workers))
    elif candidate_per_gpu is not None:
        n_workers = max(1, len(gpu_ids) * max(1, int(candidate_per_gpu)))
    else:
        n_workers = max(1, len(gpu_ids))
    chunks = _split_even(tasks, n_workers)
    work_dir = result_dir / "candidate_generation"
    work_dir.mkdir(parents=True, exist_ok=True)
    python_cmd = candidate_python or default_candidate_python()

    procs = []
    for idx, chunk in enumerate(chunks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        chunk_path = work_dir / f"candidate_chunk_{idx:03d}.json"
        manifest_path = work_dir / f"candidate_chunk_{idx:03d}_manifest.json"
        log_path = work_dir / f"candidate_chunk_{idx:03d}.log"
        write_json(chunk_path, {"tasks": chunk})
        cmd = [
            *shlex.split(python_cmd),
            str(PROJ / "tools" / "batch_pdm_candidates.py"),
            "--tasks-json",
            str(chunk_path),
            "--output-manifest",
            str(manifest_path),
            "--mesh-root",
            str(mesh_root),
            "--batch-multiplier",
            str(int(batch_multiplier)),
            "--max-batches",
            str(int(max_batches)),
            "--object-scale",
            str(float(object_scale)),
        ]
        if dataset:
            cmd.extend(["--dataset", str(dataset)])
        if no_hard_gate:
            cmd.append("--no-hard-gate")
        if pdm_checkpoint:
            cmd.extend(["--pdm-checkpoint", str(Path(pdm_checkpoint).expanduser().resolve())])
        if pose_stats:
            cmd.extend(["--pose-stats", str(Path(pose_stats).expanduser().resolve())])
        if affordance_checkpoint:
            cmd.extend(
                ["--affordance-checkpoint", str(Path(affordance_checkpoint).expanduser().resolve())]
            )
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_f = log_path.open("w", encoding="utf-8")
        print(
            f"[candidate-batch] worker {idx} gpu={gpu_id} tasks={len(chunk)} log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJ),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((idx, proc, log_f, manifest_path, log_path))

    mapping: dict[tuple[str, float], str] = {}
    for idx, proc, log_f, manifest_path, log_path in procs:
        rc = proc.wait()
        log_f.close()
        if rc != 0:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(f"candidate batch worker {idx} failed rc={rc}\nLog tail:\n{tail}")
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        for row in manifest.get("tasks", []):
            mapping[(str(row["obj_id"]), float(row["z_yaw_deg"]))] = str(row["output_hdf5"])
            print(
                "[candidate-batch] done "
                f"obj={row['obj_id']} yaw={float(row['z_yaw_deg']):.0f} "
                f"selected={row['n_selected']} pass={row['hard_gate_pass_count']} "
                f"forced={row['forced_fill_count']} batches={row['n_batches_used']} "
                f"rejects={row.get('reject_counts', {})}",
                flush=True,
            )
    write_json(
        work_dir / "candidate_manifest.json",
        {
            "version": 1,
            "tasks": [
                {"obj_id": k[0], "z_yaw_deg": k[1], "candidate_hdf5": v}
                for k, v in sorted(mapping.items())
            ],
        },
    )
    return mapping

