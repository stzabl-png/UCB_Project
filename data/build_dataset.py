#!/usr/bin/env python3
"""
Generate HDF5 Dataset for Affordance Pre-training

从 batch_output/ 的 contact map 数据 + 物体 mesh 生成训练数据集。
每个样本: 物体表面采样 1024 个点 + 每点接触/非接触标签。

用法:
    python generate_dataset.py
    python generate_dataset.py --num_points 2048 --contact_radius 0.003
"""

import os
import sys
import json
import glob
import argparse
import time
import numpy as np
import trimesh
import h5py
from scipy.spatial import cKDTree


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# Config
# ============================================================
OBJ_DIR = config.OAKINK_OBJ_DIR
BATCH_OUTPUT_DIR = config.CONTACTS_DIR
DATASET_DIR = config.DATASET_DIR


def load_object_mesh(obj_id):
    """加载物体 mesh (支持 v1 和 v2)."""
    # v2 物体: O02@xxxx@xxxxx 或其他 @ 开头的 ID
    if "@" in obj_id:
        v2_dir = getattr(config, 'OAKINK2_OBJ_DIR', None)
        if v2_dir:
            obj_dir = os.path.join(v2_dir, obj_id)
            if os.path.isdir(obj_dir):
                for f in os.listdir(obj_dir):
                    if f.endswith(('.ply', '.obj')):
                        return trimesh.load(os.path.join(obj_dir, f), force='mesh')

    # v1 物体: 直接在 OBJ_DIR 中查找
    for ext in [".obj", ".ply"]:
        path = os.path.join(OBJ_DIR, f"{obj_id}{ext}")
        if os.path.exists(path):
            return trimesh.load(path, force='mesh')
    return None


def process_sample(obj_mesh, contact_points_obj, num_points, contact_radius):
    """
    从物体 mesh 采样点云并生成接触标签。
    
    Args:
        obj_mesh: trimesh 物体网格
        contact_points_obj: (M, 3) 物体表面上的接触点
        num_points: 采样点数
        contact_radius: 接触标签半径阈值
    
    Returns:
        points: (num_points, 3) 采样点
        normals: (num_points, 3) 法线
        labels: (num_points,) 0/1 标签
    """
    # 采样点云 + 法线
    points, face_idx = obj_mesh.sample(num_points, return_index=True)
    normals = obj_mesh.face_normals[face_idx]
    
    # 计算每个采样点到最近接触点的距离
    if len(contact_points_obj) > 0:
        contact_tree = cKDTree(contact_points_obj)
        dists, _ = contact_tree.query(points)
        labels = (dists < contact_radius).astype(np.float32)
    else:
        labels = np.zeros(num_points, dtype=np.float32)
    
    return points.astype(np.float32), normals.astype(np.float32), labels


def main():
    parser = argparse.ArgumentParser(description="Generate HDF5 affordance dataset")
    parser.add_argument("--num_points", type=int, default=1024, help="Points to sample per object")
    parser.add_argument("--contact_radius", type=float, default=0.005, help="Radius for contact label (m)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--augment", type=int, default=3, help="Number of augmented samples per frame")
    parser.add_argument("--intent", type=str, default=None, help="Filter by intent (hold/use/liftup)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: dataset/)")
    args = parser.parse_args()

    output_dir = args.output_dir or DATASET_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Affordance Dataset Generation")
    print("=" * 60)
    print(f"  Batch output: {BATCH_OUTPUT_DIR}")
    print(f"  Dataset dir:  {output_dir}")
    print(f"  Points/sample: {args.num_points}")
    print(f"  Contact radius: {args.contact_radius}m")
    print(f"  Augmentations: {args.augment}x per frame")
    if args.intent:
        print(f"  Intent filter: {args.intent}")
    sys.stdout.flush()

    # Discover all NPZ files (v1 + v2)
    v1_files = sorted(glob.glob(os.path.join(BATCH_OUTPUT_DIR, "**", "*.npz"), recursive=True))
    v2_dir = getattr(config, 'CONTACTS_V2_DIR', None)
    v2_files = []
    if v2_dir and os.path.isdir(v2_dir):
        v2_files = sorted(glob.glob(os.path.join(v2_dir, "**", "*.npz"), recursive=True))
    npz_files = v1_files + v2_files
    print(f"  Found {len(v1_files)} v1 + {len(v2_files)} v2 = {len(npz_files)} contact frames")
    sys.stdout.flush()

    if not npz_files:
        print("  ⚠️  No OakInk NPZ files found, will check training_m5/ HDF5 files...")

    # Cache meshes
    mesh_cache = {}

    # Collect all samples
    all_points = []
    all_normals = []
    all_human_priors = []     # input feature: 人类抓哪里
    all_labels = []           # supervision: 机器人抓哪里能成功 (robot_gt)
    all_force_centers = []
    all_obj_ids = []
    all_categories = []
    all_intents = []

    total_start = time.time()
    skipped = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        obj_id = str(data["obj_id"])
        category = str(data["category"])
        intent = str(data.get("intent", "unknown"))
        contact_pts = data["contact_points_obj"]

        # Filter by intent
        if args.intent and intent != args.intent:
            continue

        # Load or cache mesh
        if obj_id not in mesh_cache:
            mesh = load_object_mesh(obj_id)
            if mesh is None:
                skipped += 1
                continue
            mesh_cache[obj_id] = mesh

        obj_mesh = mesh_cache[obj_id]

        # 读取受力中心 (如果存在)
        if 'force_center_obj' in data:
            force_center = data['force_center_obj'].astype(np.float32)
        else:
            force_center = np.zeros(3, dtype=np.float32)

        # Generate multiple augmented samples (different random samplings)
        for aug_i in range(args.augment):
            points, normals, labels = process_sample(
                obj_mesh, contact_pts, args.num_points, args.contact_radius
            )
            all_points.append(points)
            all_normals.append(normals)
            # OakInk: contact labels 来自人手 → 既是 human_prior 也是 labels
            all_human_priors.append(labels)
            all_labels.append(labels)
            all_force_centers.append(force_center)
            all_obj_ids.append(obj_id)
            all_categories.append(category)
            all_intents.append(intent)

        if (i + 1) % 500 == 0 or i == len(npz_files) - 1:
            n_pos = sum(l.sum() for l in all_labels)
            n_total = sum(len(l) for l in all_labels)
            print(f"  [{i+1}/{len(npz_files)}] samples={len(all_points)}, "
                  f"contact_ratio={n_pos/n_total*100:.1f}%")
            sys.stdout.flush()

    total_samples = len(all_points)
    print(f"\n  Total samples (from NPZ): {total_samples}")
    print(f"  Skipped: {skipped} frames (missing mesh)")
    sys.stdout.flush()

    # ============================================================
    # Ingest training_m5/ HDF5 files (video + gen_m5 data)
    # Each file has: human_prior (人类抓哪里) + robot_gt (机器人抓哪里能成功)
    # ============================================================
    m5_dir = getattr(config, 'TRAINING_M5_DIR', None)
    m5_count = 0
    if m5_dir and os.path.isdir(m5_dir):
        m5_files = sorted(glob.glob(os.path.join(m5_dir, "*.hdf5")))
        print(f"\n  Found {len(m5_files)} training_m5 HDF5 files")

        for m5_path in m5_files:
            try:
                with h5py.File(m5_path, 'r') as f:
                    pc = f['point_cloud'][()]   # (N, 3)
                    nrm = f['normals'][()]      # (N, 3)

                    # human_prior: 人类接触概率 (来自视频)
                    hp = f['human_prior'][()] if 'human_prior' in f else np.zeros(len(pc), dtype=np.float32)

                    # robot_gt: 机器人抓取成功标签 (来自 sim)
                    rgt = f['robot_gt'][()] if 'robot_gt' in f else np.zeros(len(pc), dtype=np.float32)

                    fc = f['force_center'][()] if 'force_center' in f else np.zeros(3, dtype=np.float32)

                # Derive obj_id from filename
                obj_id = os.path.splitext(os.path.basename(m5_path))[0]

                # Resample to match num_points if needed
                n_pts = len(pc)
                if n_pts != args.num_points:
                    if n_pts > args.num_points:
                        idx = np.random.choice(n_pts, args.num_points, replace=False)
                    else:
                        idx = np.random.choice(n_pts, args.num_points, replace=True)
                    pc = pc[idx]
                    nrm = nrm[idx]
                    hp = hp[idx]
                    rgt = rgt[idx]

                # Add augmented copies
                for _ in range(args.augment):
                    all_points.append(pc.astype(np.float32))
                    all_normals.append(nrm.astype(np.float32))
                    all_human_priors.append(hp.astype(np.float32))
                    all_labels.append(rgt.astype(np.float32))
                    all_force_centers.append(fc.astype(np.float32))
                    all_obj_ids.append(obj_id)
                    all_categories.append("video")
                    all_intents.append("grasp")

                m5_count += 1
            except Exception as e:
                print(f"    ⚠️  Failed to load {m5_path}: {e}")

        print(f"  Ingested {m5_count} m5 files → {m5_count * args.augment} samples")
    else:
        print(f"\n  No training_m5 directory found (skipping)")

    total_samples = len(all_points)
    print(f"\n  Grand total: {total_samples} samples")
    sys.stdout.flush()

    # Convert to arrays
    points_arr = np.array(all_points)           # (N, num_points, 3)
    normals_arr = np.array(all_normals)         # (N, num_points, 3)
    hp_arr = np.array(all_human_priors)         # (N, num_points) — 输入特征
    labels_arr = np.array(all_labels)           # (N, num_points) — 监督标签
    fc_arr = np.array(all_force_centers)        # (N, 3)

    # Encode string arrays
    obj_ids_arr = np.array(all_obj_ids, dtype='S20')
    categories_arr = np.array(all_categories, dtype='S20')
    intents_arr = np.array(all_intents, dtype='S20')

    # Split train/val
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    split = int(total_samples * args.train_ratio)
    train_idx = indices[:split]
    val_idx = indices[split:]

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    sys.stdout.flush()

    # Save HDF5
    for split_name, idx in [("train", train_idx), ("val", val_idx)]:
        h5_path = os.path.join(output_dir, f"affordance_{split_name}.h5")
        with h5py.File(h5_path, 'w') as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["num_samples"] = len(idx)
            meta.attrs["num_points"] = args.num_points
            meta.attrs["contact_radius"] = args.contact_radius
            meta.attrs["augmentations"] = args.augment

            # Data
            grp = f.create_group("data")
            grp.create_dataset("points", data=points_arr[idx], compression="gzip", compression_opts=4)
            grp.create_dataset("normals", data=normals_arr[idx], compression="gzip", compression_opts=4)
            grp.create_dataset("human_priors", data=hp_arr[idx], compression="gzip", compression_opts=4)
            grp.create_dataset("labels", data=labels_arr[idx], compression="gzip", compression_opts=4)
            grp.create_dataset("force_centers", data=fc_arr[idx], compression="gzip", compression_opts=4)
            grp.create_dataset("obj_ids", data=obj_ids_arr[idx])
            grp.create_dataset("categories", data=categories_arr[idx])
            grp.create_dataset("intents", data=intents_arr[idx])

        size_mb = os.path.getsize(h5_path) / 1024 / 1024
        print(f"  Saved: {h5_path} ({size_mb:.1f}MB)")
        sys.stdout.flush()

    # Contact statistics
    train_labels = labels_arr[train_idx]
    contact_ratio = train_labels.sum() / train_labels.size * 100
    avg_contacts = train_labels.sum(axis=1).mean()

    # Dataset info
    info = {
        "total_samples": total_samples,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "num_points": args.num_points,
        "contact_radius": args.contact_radius,
        "contact_ratio_percent": round(contact_ratio, 2),
        "avg_contact_points_per_sample": round(float(avg_contacts), 1),
        "augmentations": args.augment,
        "categories": sorted(set(all_categories)),
        "num_objects": len(mesh_cache),
        "generation_time_seconds": round(time.time() - total_start, 1)
    }
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {time.time() - total_start:.1f}s")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Contact ratio: {contact_ratio:.1f}%")
    print(f"  Avg contacts/sample: {avg_contacts:.1f}/{args.num_points}")
    print(f"  Dataset: {output_dir}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
