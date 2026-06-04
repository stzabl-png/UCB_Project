#!/usr/bin/env python3
"""Batch random raycast candidate generation for eval_pool ablations (R1/R2/R3)."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import (  # noqa: E402
    add_affordance_checkpoint_args,
    resolve_affordance_checkpoint,
)
from evaluation.affordance_gate import make_both_contacts_v6_gate  # noqa: E402
from evaluation.random_candidate_backend import (  # noqa: E402
    CANDIDATE_BACKEND_RANDOM_HP,
    CANDIDATE_BACKEND_RANDOM_PURE,
    CANDIDATE_BACKEND_RANDOM_RP,
    RANDOM_CANDIDATE_BACKENDS,
    resolve_hp_affordance_for_backend,
    uses_v6_affordance_gate,
)
from model.inference_v6 import load_model, normalize_affordance_pred, predict_heatmap_batch  # noqa: E402
from model.pdm.mesh_points import resolve_metric_dataset as _resolve_metric_dataset_mp  # noqa: E402
from model.pdm.mesh_points import resolve_mesh_path as _resolve_mesh_path_mp  # noqa: E402
from tools.infer_mesh_v6 import load_triangle_mesh, rescale_mesh_for_v6, sample_mesh_points  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _resolve_metric_dataset(obj_id: str, dataset: str | None) -> str:
    return _resolve_metric_dataset_mp(obj_id, dataset)


def _resolve_mesh_path(obj_id: str, mesh_root: str, dataset: str | None) -> Path:
    return _resolve_mesh_path_mp(obj_id, mesh_root, dataset)


def _prepare_mesh(
    *,
    obj_id: str,
    mesh_path: Path,
    dataset: str | None,
    num_points: int,
    seed: int,
    target_max_extent: float,
    auto_extent_lo: float,
    auto_extent_hi: float,
    min_scale_factor: float,
) -> dict:
    from tools import random_grasp_sampler as rgs

    mesh = load_triangle_mesh(mesh_path)
    try:
        sf = rgs.read_scale_factor(obj_id, dataset or None)
        if rgs.apply_metric_scale_to_mesh(obj_id, dataset or None) and abs(sf - 1.0) > 1e-8:
            mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) * float(sf)).astype(np.float64)
    except Exception as exc:
        print(f"  WARNING: rotated SAM3D scale lookup failed for {obj_id}: {exc}", flush=True)

    mesh, scale_report = rescale_mesh_for_v6(
        mesh,
        target_max_extent=target_max_extent,
        scale_mode="never",
        extent_lo=auto_extent_lo,
        extent_hi=auto_extent_hi,
        min_scale_factor=min_scale_factor,
        center_mesh=False,
    )
    scale_report.mode = "sam3d_rotated_metric"
    points, normals = sample_mesh_points(mesh, num_points, seed)
    return {
        "mesh": mesh,
        "points": points,
        "normals": normals,
        "scale_report": scale_report,
    }


def _maybe_simplify_mesh(mesh, *, target_faces: int = 5000):
    from tools.random_grasp_sampler import _safe_mesh_repair

    if len(mesh.faces) <= target_faces * 2:
        _safe_mesh_repair(mesh, "mesh")
        return mesh, None
    mesh_rc = mesh.simplify_quadric_decimation(face_count=target_faces)
    _safe_mesh_repair(mesh_rc, "mesh_rc")
    return mesh, mesh_rc


def _task_eval_seed(args: argparse.Namespace, obj_id: str, yaw: float) -> int | None:
    if getattr(args, "eval_seed", None) is None:
        return None
    from evaluation.randomness import mix_eval_seed

    return mix_eval_seed(int(args.eval_seed), "random_task", obj_id, int(round(float(yaw))) % 360)


def _hard_gates_metadata() -> str:
    return "raycast_dual_contact,width_5mm_80mm,finger_depth_40mm"


def _generate_one(
    task: dict,
    args: argparse.Namespace,
    *,
    affordance_model,
    device: torch.device,
) -> dict:
    from tools import random_grasp_sampler as rgs

    obj_id = str(task["obj_id"])
    yaw = float(task["z_yaw_deg"])
    target = int(task["target_candidates"])
    out_path = Path(task["output_hdf5"]).expanduser().resolve()
    backend = str(args.candidate_backend)
    metric_ds = _resolve_metric_dataset(obj_id, args.dataset)
    mesh_path = Path(task.get("mesh_path") or _resolve_mesh_path(obj_id, args.mesh_root, metric_ds))
    task_seed = _task_eval_seed(args, obj_id, yaw)
    if task_seed is not None:
        np.random.seed(int(task_seed) % (2**31 - 1))

    t0 = time.perf_counter()
    prepared = _prepare_mesh(
        obj_id=obj_id,
        mesh_path=mesh_path,
        dataset=metric_ds,
        num_points=args.num_points,
        seed=int(task_seed) if task_seed is not None else secrets.randbits(31),
        target_max_extent=args.target_max_extent,
        auto_extent_lo=args.auto_extent_lo,
        auto_extent_hi=args.auto_extent_hi,
        min_scale_factor=args.min_scale_factor,
    )
    mesh = prepared["mesh"]
    mesh, mesh_rc = _maybe_simplify_mesh(mesh)

    affordance_gate = None
    gate_suffix = ""
    if uses_v6_affordance_gate(backend):
        if affordance_model is None:
            raise RuntimeError(f"backend {backend} requires affordance model")
        pred = predict_heatmap_batch(
            affordance_model,
            prepared["points"][None, ...],
            prepared["normals"][None, ...],
            device,
        )
        if pred.ndim == 2:
            pred = pred[0]
        pred_norm, norm_stats = normalize_affordance_pred(np.asarray(pred, dtype=np.float32))
        affordance_gate = make_both_contacts_v6_gate(prepared["points"], pred_norm)
        gate_suffix = ",v6_both_norm_0.3"
        aff_meta = norm_stats
    else:
        aff_meta = {}

    print(
        f"[batch-random] obj={obj_id} yaw={yaw:.0f} backend={backend} "
        f"target={target} pool_start",
        flush=True,
    )
    max_batches = int(args.max_batches)
    candidates, pool_stats = rgs.generate_candidates_eval_pool(
        mesh,
        mesh_rc=mesh_rc,
        target_n=target,
        affordance_gate=affordance_gate,
        max_batches=max_batches,
    )
    elapsed_s = time.perf_counter() - t0
    affordance_fallback = bool(pool_stats.get("affordance_gate_dropped"))
    geometry_fallback = bool(pool_stats.get("geometry_gates_dropped"))
    shortfall = int(pool_stats.get("pool_shortfall", 0))
    if not candidates:
        raise RuntimeError(
            f"no random candidates for {obj_id} yaw={yaw} backend={backend} "
            f"(gates={_hard_gates_metadata()}{gate_suffix}; fill phases exhausted)"
        )
    if affordance_fallback or shortfall > 0:
        print(
            f"[batch-random] obj={obj_id} yaw={yaw:.0f} "
            f"selected={len(candidates)}/{target} "
            f"gated={pool_stats.get('n_candidates_after_gated', '?')} "
            f"aff_drop={affordance_fallback} geom_drop={geometry_fallback} "
            f"shortfall={shortfall}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "candidate_backend": backend,
        "hard_gates": _hard_gates_metadata() + gate_suffix,
        "width_min_m": float(rgs.MIN_GRIPPER_WIDTH),
        "width_max_m": float(rgs.MAX_GRIPPER_OPEN),
        "finger_depth_max_m": 0.04,
        "z_yaw_deg": yaw,
        "n_target": target,
        "elapsed_s": round(elapsed_s, 2),
        **pool_stats,
        **{f"aff_{k}": v for k, v in aff_meta.items()},
    }
    if affordance_gate is not None and not affordance_fallback:
        extra["affordance_threshold_norm"] = 0.3
        extra["affordance_knn_max_m"] = 0.015
    if affordance_fallback:
        extra["affordance_gate_dropped_for_fill"] = True
    if geometry_fallback:
        extra["geometry_gates_dropped_for_fill"] = True
    if shortfall > 0:
        extra["pool_shortfall"] = shortfall
    rgs.save_candidates_hdf5(
        candidates,
        obj_id,
        str(mesh_path),
        str(out_path.parent),
        no_rotation=True,
        dataset=metric_ds,
        scale_factor=1.0,
        apply_scale_to_mesh=False,
        sampling_method=rgs.SAMPLING_METHOD_EVAL_RAYCAST,
        extra_metadata=extra,
        output_hdf5=str(out_path),
    )

    print(
        f"[batch-random] obj={obj_id} yaw={yaw:.0f} backend={backend} "
        f"selected={len(candidates)}/{target} "
        f"gated_batches={pool_stats.get('n_batches_gated')} "
        f"fill_batches={pool_stats.get('n_batches_fill_no_affordance', 0)} "
        f"elapsed_s={elapsed_s:.1f} -> {out_path}",
        flush=True,
    )
    return {
        "obj_id": obj_id,
        "z_yaw_deg": yaw,
        "output_hdf5": str(out_path),
        "candidate_backend": backend,
        "n_selected": len(candidates),
        "n_target": target,
        "n_batches_gated": pool_stats.get("n_batches_gated"),
        "n_batches_fill_no_affordance": pool_stats.get("n_batches_fill_no_affordance", 0),
        "affordance_gate_dropped": affordance_fallback,
        "reject_counts": {},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch random raycast candidates for eval_pool ablations")
    p.add_argument("--tasks-json", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--mesh-root", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument(
        "--candidate-backend",
        choices=sorted(RANDOM_CANDIDATE_BACKENDS),
        default=CANDIDATE_BACKEND_RANDOM_RP,
        help="random_rp / random_hp / random_pure",
    )
    add_affordance_checkpoint_args(p)
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--max-batches", type=int, default=10)
    p.add_argument("--target-max-extent", type=float, default=0.28)
    p.add_argument("--auto-extent-lo", type=float, default=0.02)
    p.add_argument("--auto-extent-hi", type=float, default=0.80)
    p.add_argument("--min-scale-factor", type=float, default=1e-6)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--eval-seed", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    backend = str(args.candidate_backend)
    if backend not in (CANDIDATE_BACKEND_RANDOM_RP, CANDIDATE_BACKEND_RANDOM_HP, CANDIDATE_BACKEND_RANDOM_PURE):
        raise SystemExit(
            f"--candidate-backend must be random_rp, random_hp, or random_pure (got {backend!r})"
        )

    tasks = _load_json(args.tasks_json).get("tasks", [])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    affordance_model = None
    if uses_v6_affordance_gate(backend):
        hp_flag = resolve_hp_affordance_for_backend(backend, bool(args.hp_affordance))
        aff_ckpt = resolve_affordance_checkpoint(
            hp_affordance=hp_flag,
            affordance_checkpoint=args.affordance_checkpoint,
        )
        affordance_model, _ = load_model(str(aff_ckpt), device)
        print(
            f"[batch-random] backend={backend} affordance={aff_ckpt} device={device} tasks={len(tasks)}",
            flush=True,
        )
    else:
        print(
            f"[batch-random] backend={backend} (no affordance) device={device} tasks={len(tasks)}",
            flush=True,
        )

    rows = []
    for task in tasks:
        rows.append(
            _generate_one(
                task,
                args,
                affordance_model=affordance_model,
                device=device,
            )
        )
    _write_json(args.output_manifest, {"version": 1, "tasks": rows})
    print(f"[batch-random] wrote {args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
