#!/usr/bin/env python3
"""
merge_robot_gt.py
─────────────────────────────────────────────────────────────
自动扫描 output/robot_gt_*/ 目录（R1, R2, R3, R4...），
对每个物体，合并所有轮次中 successful_grasps 里的成功候选，
输出到 robot_gt_merged_*/。

训练代码只需读 robot_gt_merged/，新增轮次后重新跑此脚本即可。

用法:
    python3 tools/merge_robot_gt.py                   # 合并 OakInk + DexYCB
    python3 tools/merge_robot_gt.py --dataset oakink
    python3 tools/merge_robot_gt.py --dataset dexycb
    python3 tools/merge_robot_gt.py --dry-run         # 预览不写文件
"""

import argparse
import glob
import os

import h5py
import numpy as np


OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

DATASET_PATTERNS = {
    "oakink": {
        "round_glob": "robot_gt_oakink*",
        "merged_dir": "robot_gt_merged_oakink",
    },
    "dexycb": {
        "round_glob": "robot_gt_dexycb*",
        "merged_dir": "robot_gt_merged_dexycb",
    },
}


def find_round_dirs(output_base: str, glob_pattern: str) -> list:
    dirs = sorted(glob.glob(os.path.join(output_base, glob_pattern)))
    dirs = [d for d in dirs if "merged" not in os.path.basename(d)]
    return dirs


def collect_all_successful(round_dirs: list, obj_id: str):
    """
    跨轮次收集同一物体的所有成功 grasp。
    返回: (list of grasp dicts, list of source round names)
    每个 grasp dict: {
        datasets: { key: np.array },
        attrs:    { key: value },
        source:   str  # 来自哪个轮次目录
    }
    """
    all_grasps = []
    sources = []

    for rdir in round_dirs:
        result_file = os.path.join(rdir, f"{obj_id}_robot_gt.hdf5")
        if not os.path.exists(result_file):
            continue
        try:
            with h5py.File(result_file, "r") as f:
                n_suc = int(f.attrs.get("n_successful", 0))
                if n_suc == 0:
                    continue
                rname = os.path.basename(rdir)
                if rname not in sources:
                    sources.append(rname)

                sg = f.get("successful_grasps")
                if sg is None:
                    continue

                for gkey in sorted(sg.keys()):
                    g = sg[gkey]
                    entry = {
                        "datasets": {k: g[k][()] for k in g.keys()},
                        "attrs":    dict(g.attrs),
                        "source":   rname,
                    }
                    all_grasps.append(entry)

        except Exception as e:
            print(f"    ⚠️  读取失败 {result_file}: {e}")

    return all_grasps, sources


def write_merged(obj_id: str, all_grasps: list, sources: list,
                 merged_dir: str, template_file: str):
    """写合并后的 HDF5，结构与原始文件一致。"""
    out_path = os.path.join(merged_dir, f"{obj_id}_robot_gt.hdf5")
    os.makedirs(merged_dir, exist_ok=True)

    # 读原始顶层 attrs 作为模板
    base_attrs = {}
    if template_file and os.path.exists(template_file):
        with h5py.File(template_file, "r") as f:
            base_attrs = dict(f.attrs)

    with h5py.File(out_path, "w") as out:
        # 顶层 attrs
        for k, v in base_attrs.items():
            out.attrs[k] = v
        out.attrs["n_successful"]       = len(all_grasps)
        out.attrs["n_candidates_total"] = len(all_grasps)
        out.attrs["success"]            = len(all_grasps) > 0
        out.attrs["merge_sources"]      = str(sources)
        out.attrs["obj_id"]             = obj_id

        # successful_grasps 组
        sg = out.create_group("successful_grasps")
        for i, grasp in enumerate(all_grasps):
            g = sg.create_group(f"grasp_{i}")
            for k, v in grasp["datasets"].items():
                g.create_dataset(k, data=v)
            for k, v in grasp["attrs"].items():
                g.attrs[k] = v
            g.attrs["merge_source"] = grasp["source"]


def merge_dataset(dataset: str, output_base: str, dry_run: bool):
    cfg = DATASET_PATTERNS[dataset]
    round_dirs = find_round_dirs(output_base, cfg["round_glob"])
    merged_dir = os.path.join(output_base, cfg["merged_dir"])

    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} 合并")
    print(f"  轮次目录: {[os.path.basename(d) for d in round_dirs]}")
    print(f"  输出:     {merged_dir}")
    print(f"{'='*60}")

    if not round_dirs:
        print("  ❌ 未找到任何轮次目录")
        return 0, 0

    # 收集所有物体 ID（跨轮次并集）
    all_obj_ids = set()
    for rdir in round_dirs:
        for fp in glob.glob(os.path.join(rdir, "*_robot_gt.hdf5")):
            all_obj_ids.add(os.path.basename(fp).replace("_robot_gt.hdf5", ""))

    merged_count = 0
    skipped_count = 0

    for obj_id in sorted(all_obj_ids):
        all_grasps, sources = collect_all_successful(round_dirs, obj_id)

        if not all_grasps:
            print(f"  ❌ {obj_id}: 所有轮次均失败，跳过")
            skipped_count += 1
            continue

        # 找 attrs 模板文件（任意一个有成功结果的）
        template_file = None
        for rdir in round_dirs:
            fp = os.path.join(rdir, f"{obj_id}_robot_gt.hdf5")
            if os.path.exists(fp):
                try:
                    with h5py.File(fp, "r") as f:
                        if int(f.attrs.get("n_successful", 0)) > 0:
                            template_file = fp
                            break
                except:
                    pass

        if dry_run:
            print(f"  [DRY] {obj_id}: {len(all_grasps)} 候选  来源: {sources}")
        else:
            write_merged(obj_id, all_grasps, sources, merged_dir, template_file)
            print(f"  ✅ {obj_id}: {len(all_grasps)} 候选  来源: {sources}")

        merged_count += 1

    print(f"\n  {'[DRY] ' if dry_run else ''}完成: ✅{merged_count} 有数据  ❌{skipped_count} 全轮失败  (共{len(all_obj_ids)}个物体)")
    return merged_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="合并多轮 robot_gt 结果")
    parser.add_argument("--dataset", choices=["oakink", "dexycb", "all"], default="all")
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    args = parser.parse_args()

    output_base = os.path.abspath(args.output_base or OUTPUT_BASE)
    datasets = ["oakink", "dexycb"] if args.dataset == "all" else [args.dataset]

    print(f"{'='*60}")
    print(f"  robot_gt 多轮合并  |  dry_run={args.dry_run}")
    print(f"  output_base: {output_base}")
    print(f"{'='*60}")

    total = 0
    for ds in datasets:
        m, _ = merge_dataset(ds, output_base, args.dry_run)
        total += m

    print(f"\n{'='*60}")
    print(f"  全部完成，共 {total} 个物体有合并数据")
    print(f"  训练时读取:")
    for ds in datasets:
        print(f"    output/{DATASET_PATTERNS[ds]['merged_dir']}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
