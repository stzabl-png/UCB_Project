"""
grasp_pool_common.py — 候选池 sim batch 规划 / registry / pool↔round 拷贝
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
from typing import Any, Optional

import h5py
import numpy as np

FIXED_Z_YAWS = (0.0, 90.0, 180.0, 270.0)
DEFAULT_SLOTS_PER_ROUND = 500
REGISTRY_NAME = "sim_pool_registry.json"
TASK_QUEUE_TEMPLATE = "round_{round:04d}_task_queue.json"


def round_tag(round_idx: int) -> str:
    return f"round_{round_idx:04d}"


def task_queue_name(round_idx: int) -> str:
    return TASK_QUEUE_TEMPLATE.format(round=round_idx)


def paths_for_outdir(outdir: str, round_idx: int) -> dict[str, str]:
    tag = round_tag(round_idx)
    base = os.path.abspath(outdir)
    return {
        "outdir": base,
        "pool_dir": os.path.join(base, "candidates", "pool"),
        "cand_round_dir": os.path.join(base, "candidates", tag),
        "gt_round_dir": os.path.join(base, "robot_gt", tag),
        "merged_dir": os.path.join(base, "merged"),
        "log_dir": os.path.join(base, "sim_logs", tag),
        "registry": os.path.join(base, REGISTRY_NAME),
        "task_queue": os.path.join(base, task_queue_name(round_idx)),
        "state": os.path.join(base, "state.json"),
        "summary": os.path.join(base, "summary.csv"),
    }


def pool_hdf5(pool_dir: str, obj_id: str) -> str:
    return os.path.join(pool_dir, f"{obj_id}_grasp.hdf5")


def round_grasp_hdf5(cand_round_dir: str, obj_id: str) -> str:
    return os.path.join(cand_round_dir, f"{obj_id}_grasp.hdf5")


def round_gt_hdf5(gt_round_dir: str, obj_id: str) -> str:
    return os.path.join(gt_round_dir, f"{obj_id}_robot_gt.hdf5")


def merged_hdf5(merged_dir: str, obj_id: str) -> str:
    return os.path.join(merged_dir, f"{obj_id}_robot_gt_merged.hdf5")


def _parse_round_from_path(path: str) -> Optional[int]:
    m = re.search(r"round_(\d+)", path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def count_success_in_gt_file(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with h5py.File(path, "r") as f:
        if "successful_grasps" in f:
            sg = f["successful_grasps"]
            n_attr = int(sg.attrs.get("count", -1))
            if n_attr >= 0:
                return n_attr
            return len(sg.keys())
        return int(f.attrs.get("n_successful", 0))


def scan_success_round_ge3(outdir: str, min_round: int = 3) -> dict[str, int]:
    """obj_id -> total successful_grasps in robot_gt/round_R for R >= min_round."""
    totals: dict[str, int] = {}
    gt_root = os.path.join(os.path.abspath(outdir), "robot_gt")
    if not os.path.isdir(gt_root):
        return totals
    for tag in sorted(os.listdir(gt_root)):
        if not tag.startswith("round_"):
            continue
        r = _parse_round_from_path(tag)
        if r is None or r < min_round:
            continue
        rd = os.path.join(gt_root, tag)
        for fn in os.listdir(rd):
            if not fn.endswith("_robot_gt.hdf5"):
                continue
            obj_id = fn[: -len("_robot_gt.hdf5")]
            n = count_success_in_gt_file(os.path.join(rd, fn))
            totals[obj_id] = totals.get(obj_id, 0) + n
    return totals


def compute_median_success_threshold(
    outdir: str,
    merged_dir: str,
    *,
    min_round: int = 3,
) -> int:
    """
    Median of success_round_ge3 over objects that have a merged file.
    Used when auto-refilling the candidate pool.
    """
    merged_counts = scan_merged_objects(merged_dir)
    success = scan_success_round_ge3(outdir, min_round=min_round)
    values = [success.get(obj_id, 0) for obj_id in merged_counts]
    if not values:
        return 0
    return int(np.median(np.array(values, dtype=np.float64)))


def scan_merged_objects(merged_dir: str) -> dict[str, int]:
    from merge_robot_gt import _count_successful_in_merged

    counts: dict[str, int] = {}
    pattern = os.path.join(merged_dir, "*_robot_gt_merged.hdf5")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        obj_id = base[: -len("_robot_gt_merged.hdf5")]
        counts[obj_id] = _count_successful_in_merged(path)
    return counts


def load_registry(path: str) -> dict:
    if not os.path.isfile(path):
        return {"version": 1, "candidates": {}}
    with open(path, "r") as f:
        data = json.load(f)
    data.setdefault("candidates", {})
    return data


def save_registry(path: str, registry: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _obj_registry(registry: dict, obj_id: str) -> dict:
    return registry.setdefault("candidates", {}).setdefault(obj_id, {})


def candidate_key(name: str, pool_idx: int) -> str:
    return f"{name}#{pool_idx}"


def parse_candidate_key(key: str) -> tuple[str, int]:
    if "#" in key:
        name, idx_s = key.rsplit("#", 1)
        return name, int(idx_s)
    return key, -1


def is_fully_simulated(registry: dict, obj_id: str, key: str) -> bool:
    rec = _obj_registry(registry, obj_id).get(key)
    if not rec:
        return False
    if rec.get("simulated"):
        return True
    done = set(float(y) for y in rec.get("yaws_done", []))
    return done >= set(FIXED_Z_YAWS)


def list_pool_candidates_sorted(pool_path: str) -> list[dict[str, Any]]:
    """Candidates from pool HDF5, highest score first."""
    if not os.path.isfile(pool_path):
        return []
    out: list[dict[str, Any]] = []
    with h5py.File(pool_path, "r") as f:
        if "candidates" not in f:
            return out
        cg = f["candidates"]
        n = int(cg.attrs.get("n_candidates", 0))
        for i in range(n):
            gname = f"candidate_{i}"
            if gname not in cg:
                continue
            ci = cg[gname]
            name = str(ci.attrs.get("name", gname))
            out.append(
                {
                    "pool_idx": i,
                    "name": name,
                    "key": candidate_key(name, i),
                    "score": float(ci.attrs.get("score", 0.0)),
                }
            )
    out.sort(key=lambda c: (-c["score"], c["pool_idx"]))
    return out


def available_pool_candidates(
    registry: dict,
    pool_path: str,
    obj_id: str,
) -> list[dict[str, Any]]:
    return [
        c
        for c in list_pool_candidates_sorted(pool_path)
        if not is_fully_simulated(registry, obj_id, c["key"])
    ]


def eligible_objects(
    merged_dir: str,
    pool_dir: str,
    registry: dict,
) -> list[str]:
    """Objects with merged file, pool HDF5, and ≥1 unsimulated candidate."""
    merged = scan_merged_objects(merged_dir)
    eligible: list[str] = []
    for obj_id in sorted(merged.keys()):
        pp = pool_hdf5(pool_dir, obj_id)
        if not os.path.isfile(pp):
            continue
        if available_pool_candidates(registry, pp, obj_id):
            eligible.append(obj_id)
    return eligible


def normalize_weights(obj_ids: list[str], success: dict[str, int]) -> dict[str, float]:
    weights = {oid: 1.0 / (success.get(oid, 0) + 1) for oid in obj_ids}
    total = sum(weights.values())
    if total <= 0:
        n = len(obj_ids)
        return {oid: 1.0 / n for oid in obj_ids}
    return {oid: w / total for oid, w in weights.items()}


def sample_object(
    rng: np.random.Generator,
    obj_ids: list[str],
    probs: dict[str, float],
) -> Optional[str]:
    if not obj_ids:
        return None
    p = np.array([probs[oid] for oid in obj_ids], dtype=np.float64)
    p /= p.sum()
    idx = int(rng.choice(len(obj_ids), p=p))
    return obj_ids[idx]


def plan_round_slots(
    *,
    outdir: str,
    merged_dir: str,
    pool_dir: str,
    registry: dict,
    round_idx: int,
    slots_per_round: int,
    min_success_round: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[dict], bool]:
    """
    Plan up to slots_per_round unique (obj, candidate) assignments.
    Returns (slot_list, exhausted) where exhausted=True if stopped early
    because no eligible candidate remained.
    """
    rng = rng or np.random.default_rng()
    success = scan_success_round_ge3(outdir, min_round=min_success_round)
    eligible = eligible_objects(merged_dir, pool_dir, registry)
    if not eligible:
        return [], True

    probs = normalize_weights(eligible, success)
    used_this_plan: set[tuple[str, str]] = set()
    slots: list[dict] = []
    exhausted = False

    for _ in range(slots_per_round):
        picked = None
        for _try in range(len(eligible) * 50):
            oid = sample_object(rng, eligible, probs)
            if oid is None:
                exhausted = True
                break
            avail = available_pool_candidates(
                registry, pool_hdf5(pool_dir, oid), oid,
            )
            for cand in avail:
                pair = (oid, cand["key"])
                if pair in used_this_plan:
                    continue
                picked = {
                    "obj_id": oid,
                    "candidate_key": cand["key"],
                    "candidate_name": cand["name"],
                    "pool_idx": cand["pool_idx"],
                    "score": cand["score"],
                }
                used_this_plan.add(pair)
                break
            if picked is not None:
                break
        if picked is None:
            exhausted = True
            break
        slots.append(picked)

    return slots, exhausted


def expand_slots_to_tasks(
    slots: list[dict],
    round_idx: int,
    dataset_by_obj: dict[str, str],
) -> list[dict]:
    tasks: list[dict] = []
    for si, slot in enumerate(slots):
        obj_id = slot["obj_id"]
        for yi, yaw in enumerate(FIXED_Z_YAWS):
            tasks.append(
                {
                    "task_id": f"r{round_idx:04d}_s{si:04d}_y{int(yaw):03d}",
                    "round_idx": round_idx,
                    "obj_id": obj_id,
                    "dataset": dataset_by_obj.get(obj_id, ""),
                    "candidate_key": slot["candidate_key"],
                    "candidate_name": slot["candidate_name"],
                    "pool_idx": slot["pool_idx"],
                    "score": slot["score"],
                    "z_yaw_deg": float(yaw),
                    "slot_index": si,
                    "yaw_index": yi,
                    "status": "pending",
                }
            )
    return tasks


def build_task_queue(
    *,
    outdir: str,
    merged_dir: str,
    pool_dir: str,
    registry: dict,
    round_idx: int,
    slots_per_round: int,
    dataset_by_obj: dict[str, str],
    rng: Optional[np.random.Generator] = None,
) -> dict:
    slots, exhausted = plan_round_slots(
        outdir=outdir,
        merged_dir=merged_dir,
        pool_dir=pool_dir,
        registry=registry,
        round_idx=round_idx,
        slots_per_round=slots_per_round,
        rng=rng,
    )
    tasks = expand_slots_to_tasks(slots, round_idx, dataset_by_obj)
    return {
        "version": 1,
        "round_idx": round_idx,
        "slots_planned": len(slots),
        "slots_target": slots_per_round,
        "pool_exhausted": exhausted,
        "tasks": tasks,
        "completed_task_ids": [],
    }


def load_task_queue(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_task_queue(path: str, queue: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(queue, f, indent=2)


def pending_tasks(queue: dict) -> list[dict]:
    done = set(queue.get("completed_task_ids", []))
    return [t for t in queue.get("tasks", []) if t["task_id"] not in done]


def is_queue_complete(queue: dict) -> bool:
    return len(pending_tasks(queue)) == 0


def mark_task_done(queue: dict, task_id: str) -> None:
    done = queue.setdefault("completed_task_ids", [])
    if task_id not in done:
        done.append(task_id)


def _copy_candidate_group(src_ci: h5py.Group, dst_cg: h5py.Group, dst_index: int) -> None:
    gname = f"candidate_{dst_index}"
    if gname in dst_cg:
        del dst_cg[gname]
    dst = dst_cg.create_group(gname)
    for key in src_ci.keys():
        src_ci.copy(src_ci[key], dst, key)
    for ak, av in src_ci.attrs.items():
        dst.attrs[ak] = av


def copy_slots_to_round_hdf5(
    pool_dir: str,
    cand_round_dir: str,
    slots: list[dict],
) -> dict[str, int]:
    """
    Merge this round's pool candidates into per-obj round grasp HDF5.
    Returns obj_id -> n_candidates in round file.
    """
    os.makedirs(cand_round_dir, exist_ok=True)
    by_obj: dict[str, list[dict]] = {}
    for slot in slots:
        by_obj.setdefault(slot["obj_id"], []).append(slot)

    counts: dict[str, int] = {}
    for obj_id, obj_slots in by_obj.items():
        pool_path = pool_hdf5(pool_dir, obj_id)
        round_path = round_grasp_hdf5(cand_round_dir, obj_id)
        keys_needed = {s["candidate_key"] for s in obj_slots}

        existing: dict[str, int] = {}
        next_idx = 0
        if os.path.isfile(round_path):
            with h5py.File(round_path, "r") as rf:
                if "candidates" in rf:
                    cg = rf["candidates"]
                    n = int(cg.attrs.get("n_candidates", 0))
                    for i in range(n):
                        gname = f"candidate_{i}"
                        if gname not in cg:
                            continue
                        ci = cg[gname]
                        name = str(ci.attrs.get("name", gname))
                        key = candidate_key(name, i)
                        existing[key] = i
                    next_idx = n

        with h5py.File(pool_path, "r") as pf:
            if "metadata" in pf:
                meta_attrs = dict(pf["metadata"].attrs.items())
            else:
                meta_attrs = dict(pf.attrs.items())
            pool_cg = pf["candidates"]

            mode = "a" if os.path.isfile(round_path) else "w"
            with h5py.File(round_path, mode) as wf:
                if "metadata" not in wf:
                    mg = wf.create_group("metadata")
                    for k, v in meta_attrs.items():
                        mg.attrs[k] = v
                if "candidates" not in wf:
                    wf.create_group("candidates")
                dst_cg = wf["candidates"]

                for slot in obj_slots:
                    key = slot["candidate_key"]
                    if key in existing:
                        continue
                    pi = slot["pool_idx"]
                    src_name = f"candidate_{pi}"
                    if src_name not in pool_cg:
                        continue
                    _copy_candidate_group(pool_cg[src_name], dst_cg, next_idx)
                    dst_cg[f"candidate_{next_idx}"].attrs["pool_candidate_key"] = key
                    dst_cg[f"candidate_{next_idx}"].attrs["pool_idx"] = pi
                    existing[key] = next_idx
                    next_idx += 1

                dst_cg.attrs["n_candidates"] = next_idx
                counts[obj_id] = next_idx

    return counts


def unique_slots_from_tasks(tasks: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    slots: list[dict] = []
    for t in tasks:
        pair = (t["obj_id"], t["candidate_key"])
        if pair in seen:
            continue
        seen.add(pair)
        slots.append(
            {
                "obj_id": t["obj_id"],
                "candidate_key": t["candidate_key"],
                "candidate_name": t["candidate_name"],
                "pool_idx": t["pool_idx"],
                "score": t.get("score", 0.0),
            }
        )
    return slots


def update_registry_from_results(
    registry: dict,
    results: list[dict],
    round_idx: int,
) -> None:
    """Group by (obj, candidate_key); mark simulated when all 4 yaws attempted."""
    by_cand: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        pair = (r["obj_id"], r["candidate_key"])
        by_cand.setdefault(pair, []).append(r)

    for (obj_id, key), rows in by_cand.items():
        rec = _obj_registry(registry, obj_id).setdefault(
            key,
            {"yaws_done": [], "simulated": False},
        )
        for row in rows:
            yaw = float(row.get("z_yaw_deg", 0.0))
            done = set(float(y) for y in rec.get("yaws_done", []))
            done.add(yaw)
            rec["yaws_done"] = sorted(done)
            if row.get("success"):
                rec.setdefault("success_yaws", []).append(yaw)
        rec["last_round"] = round_idx
        attempted = {float(r["z_yaw_deg"]) for r in rows if r.get("attempted", True)}
        if attempted >= set(FIXED_Z_YAWS):
            rec["simulated"] = True


def sort_tasks_for_workers(tasks: list[dict]) -> list[dict]:
    return sorted(
        tasks,
        key=lambda t: (t["obj_id"], t.get("slot_index", 0), t.get("yaw_index", 0)),
    )


def split_tasks_into_chunks(tasks: list[dict], n_chunks: int) -> list[list[dict]]:
    if n_chunks < 1:
        n_chunks = 1
    tasks = sort_tasks_for_workers(tasks)
    n = len(tasks)
    if n == 0:
        return []
    chunk_size = (n + n_chunks - 1) // n_chunks
    return [tasks[i : i + chunk_size] for i in range(0, n, chunk_size)]
