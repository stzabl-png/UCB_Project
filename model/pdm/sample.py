#!/usr/bin/env python3
"""Sample PDM poses and export candidate HDF5 for simulation/evaluation."""

from __future__ import annotations

import argparse
import os
import random
import sys

import h5py
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from model.pdm.dataset import (
    AffordanceStore,
    DEFAULT_ROTATED_MESH_DIR,
    PDMConditionStore,
    find_mesh_path,
    sample_mesh_condition,
    _resample_rows,
)
from model.pdm.model import PDM
from model.pdm.pose_codec import pose9_to_command


def _decode_obj_ids(raw) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def infer_dataset_for_obj(obj_id: str, fallback: str = "oakink") -> str:
    if obj_id.startswith("ycb_dex_"):
        return "dexycb"
    if obj_id.startswith("arctic_"):
        return "arctic"
    if obj_id and obj_id[0] in ("A", "C", "O", "S", "Y"):
        return "oakink"
    return fallback


def obj_ids_from_condition_h5(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        return _decode_obj_ids(f["data/obj_ids"][:])


def obj_ids_from_merged(merged_dir: str) -> list[str]:
    ids = []
    if not os.path.isdir(merged_dir):
        return ids
    for name in sorted(os.listdir(merged_dir)):
        if name.endswith("_robot_gt_merged.hdf5"):
            ids.append(name.replace("_robot_gt_merged.hdf5", ""))
    return ids


def resolve_obj_ids(args: argparse.Namespace) -> list[str]:
    if args.obj:
        ids = list(dict.fromkeys(args.obj))
    elif args.all or args.random:
        if args.condition_h5:
            ids = obj_ids_from_condition_h5(args.condition_h5)
        else:
            ids = obj_ids_from_merged(args.merged_dir)
    else:
        raise ValueError("provide --obj, --all, or --random N")

    if args.random:
        rng = random.Random(args.seed)
        ids = list(ids)
        rng.shuffle(ids)
        ids = ids[: args.random]
    return ids


def load_condition(
    obj_id: str,
    n_points: int,
    condition_h5: str | None,
    affordance_h5: str | None,
    mesh_root: str,
) -> torch.Tensor:
    cond_store = PDMConditionStore(condition_h5)
    if cond_store.has(obj_id):
        cond = cond_store.load(obj_id)
        points = _resample_rows(cond.points, n_points)
        return torch.from_numpy(points).unsqueeze(0)
    aff = AffordanceStore(affordance_h5)
    if aff.has(obj_id):
        cond = aff.load(obj_id)
        points = _resample_rows(cond.points, n_points)
    else:
        cond = sample_mesh_condition(obj_id, n_points=n_points, mesh_root=mesh_root)
        points = cond.points
    return torch.from_numpy(points).unsqueeze(0)


def _identity_prerotation(parent: h5py.Group, dataset: str = "unknown") -> None:
    g = parent.create_group("mesh_prerotation")
    g.attrs["method"] = "identity"
    g.attrs["dataset"] = dataset
    g.create_dataset("euler_xyz_deg", data=np.zeros(3, dtype=np.float32))
    g.create_dataset("matrix", data=np.eye(3, dtype=np.float32))


def write_candidates_hdf5(
    path: str,
    obj_id: str,
    poses: np.ndarray,
    *,
    mesh_path: str | None,
    gripper_width: float,
    dataset: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["obj_id"] = obj_id
        meta.attrs["method"] = "pdm"
        meta.attrs["sampling_method"] = "pdm_diffusion"
        meta.attrs["no_rotation"] = True
        meta.attrs["dataset"] = dataset
        meta.attrs["mesh_path"] = os.path.abspath(mesh_path) if mesh_path else ""
        meta.attrs["n_candidates"] = len(poses)
        _identity_prerotation(meta, dataset=dataset)

        cg = f.create_group("candidates")
        cg.attrs["n_candidates"] = len(poses)
        for i, pose9 in enumerate(poses):
            command = pose9_to_command(pose9)
            gi = cg.create_group(f"candidate_{i}")
            gi.attrs["name"] = f"pdm_{i:03d}"
            gi.attrs["score"] = float(len(poses) - i)
            gi.attrs["gripper_width"] = float(gripper_width)
            gi.attrs["approach_type"] = "pdm"
            gi.create_dataset("position", data=command.position.astype(np.float32))
            gi.create_dataset("grasp_point", data=command.position.astype(np.float32))
            gi.create_dataset("rotation", data=command.rotation.astype(np.float32))
            _identity_prerotation(gi, dataset=dataset)

        if len(poses) > 0:
            best = pose9_to_command(poses[0])
            g = f.create_group("grasp")
            g.attrs["gripper_width"] = float(gripper_width)
            g.create_dataset("position", data=best.position.astype(np.float32))
            g.create_dataset("grasp_point", data=best.position.astype(np.float32))
            g.create_dataset("rotation", data=best.rotation.astype(np.float32))
            _identity_prerotation(g, dataset=dataset)

        aff = f.create_group("affordance")
        aff.attrs["n_contact"] = 0


def sample_one(
    args: argparse.Namespace,
    model: PDM,
    stats: dict,
    obj_id: str,
    device: torch.device,
    *,
    use_explicit_output: bool = False,
) -> str:
    pose_mean = stats["pose_mean"].to(device)
    pose_std = stats["pose_std"].to(device)

    points = load_condition(
        obj_id,
        n_points=args.n_points,
        condition_h5=args.condition_h5,
        affordance_h5=args.affordance_h5,
        mesh_root=args.mesh_root,
    ).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pose_norm = model.sample(points, n_samples=args.n_samples, ddim_steps=args.ddim_steps)
        pose = pose_norm * pose_std.unsqueeze(0) + pose_mean.unsqueeze(0)
    poses_np = pose.cpu().numpy().astype(np.float32)

    if args.reject_upward:
        kept = []
        for p in poses_np:
            cmd = pose9_to_command(p)
            if cmd.rotation[:, 2][2] <= args.max_approach_z:
                kept.append(p)
        poses_np = np.asarray(kept, dtype=np.float32)

    if args.output and use_explicit_output:
        out_path = args.output
    else:
        out_path = os.path.join(args.output_dir, f"{obj_id}_grasp.hdf5")
    mesh_path = find_mesh_path(obj_id, args.mesh_root)
    write_candidates_hdf5(
        out_path,
        obj_id,
        poses_np,
        mesh_path=mesh_path,
        gripper_width=args.gripper_width,
        dataset=infer_dataset_for_obj(obj_id, args.dataset),
    )
    print(f"Saved {len(poses_np)} PDM candidates -> {out_path}")
    return out_path


def sample(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = PDM.load(args.checkpoint, device=device)
    stats = ckpt.get("pose_stats")
    if stats is None:
        if args.pose_stats is None:
            raise ValueError(
                "checkpoint has no pose_stats; provide --pose-stats pointing to pose_stats.pt"
            )
        stats = torch.load(args.pose_stats, map_location=device, weights_only=False)
    obj_ids = resolve_obj_ids(args)
    if not obj_ids:
        raise RuntimeError("no objects selected for PDM sampling")
    if args.output and len(obj_ids) > 1:
        raise ValueError("--output can only be used with a single object")
    print(f"Sampling {len(obj_ids)} object(s)")
    for i, obj_id in enumerate(obj_ids, 1):
        print(f"[{i}/{len(obj_ids)}] {obj_id}")
        sample_one(args, model, stats, obj_id, device, use_explicit_output=len(obj_ids) == 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample PDM grasp candidates")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obj", nargs="+", default=None, help="Object id(s) to sample")
    parser.add_argument("--all", action="store_true", help="Sample all objects from condition cache or merged dir")
    parser.add_argument("--random", type=int, default=0, metavar="N", help="Sample N random objects")
    parser.add_argument("--merged-dir", default=os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged"))
    parser.add_argument("--condition-h5", default=None)
    parser.add_argument("--affordance-h5", default=None)
    parser.add_argument("--pose-stats", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=os.path.join(PROJ, "output", "pdm", "candidates"))
    parser.add_argument("--mesh-root", default=DEFAULT_ROTATED_MESH_DIR)
    parser.add_argument("--dataset", default="oakink")
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--gripper-width", type=float, default=0.06)
    parser.add_argument("--reject-upward", action="store_true")
    parser.add_argument("--max-approach-z", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    sample(build_parser().parse_args())
