#!/usr/bin/env python3
"""
Step 1: 从 HuggingFace 下载 OakInk 中本地缺少的 mesh，并转换为 OBJ

HF 数据集: UCBProject/ObjMesh  meshes/oakink/<name>/mesh.ply
本地 USD 目录: assets/usd/ (检查 A/C/S/O/Y 开头的 .usd)
下载输出: assets/oakink_missing_obj/<name>.obj  (供 Isaac Sim 批量转 USD)

使用方式:
    python3 assets/download_missing_oakink.py
    python3 assets/download_missing_oakink.py --dry-run  # 只列出缺失项
"""
import os
import argparse
import trimesh
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID    = "UCBProject/ObjMesh"
REPO_TYPE  = "dataset"
HF_PREFIX  = "meshes/oakink"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR    = os.path.join(SCRIPT_DIR, "usd")
OBJ_OUT    = os.path.join(SCRIPT_DIR, "oakink_missing_obj")


def get_hf_names():
    """获取 HF 上 oakink 所有子目录名（即 mesh 名）"""
    files = list(list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=HF_PREFIX))
    # path 格式: "meshes/oakink/<name>"  (RepoFolder) 或 "meshes/oakink/<name>/mesh.ply" (RepoFile)
    names = set()
    for f in files:
        parts = f.path.split("/")
        if len(parts) >= 3:
            # 取第三层: meshes/oakink/<name>
            names.add(parts[2])
    return sorted(names)


def get_local_usd_names():
    """获取本地 usd/ 目录中所有 A/C/S/O/Y 开头的 USD 名（无后缀）"""
    if not os.path.isdir(USD_DIR):
        return set()
    return {f[:-4] for f in os.listdir(USD_DIR)
            if f.endswith(".usd") and f[0] in "ACOSY"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只列出缺失项，不下载")
    args = parser.parse_args()

    print("📡 查询 HuggingFace 目录...", flush=True)
    hf_names   = get_hf_names()
    local_usds = get_local_usd_names()

    missing = sorted(set(hf_names) - local_usds)
    print(f"HF 总数: {len(hf_names)}, 本地已有 USD: {len(local_usds)}, 需下载: {len(missing)}")

    if not missing:
        print("✅ 全部已转换，无需下载！")
        return

    print(f"缺失列表: {missing}\n")

    if args.dry_run:
        print("--dry-run 模式，退出。")
        return

    os.makedirs(OBJ_OUT, exist_ok=True)

    ok_cnt, skip_cnt, fail_list = 0, 0, []

    for i, name in enumerate(missing, 1):
        obj_path = os.path.join(OBJ_OUT, f"{name}.obj")
        prefix = f"[{i:3d}/{len(missing)}] {name}"

        # 已转好 OBJ 则跳过下载
        if os.path.exists(obj_path):
            print(f"{prefix}: OBJ exists, skip download ✓")
            skip_cnt += 1
            continue

        # 下载 mesh.ply
        hf_path = f"{HF_PREFIX}/{name}/mesh.ply"
        print(f"{prefix}: downloading...", end=" ", flush=True)
        try:
            ply_local = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=hf_path,
                local_dir=os.path.join(SCRIPT_DIR, ".hf_cache"),
            )
        except Exception as e:
            print(f"❌ download error: {e}")
            fail_list.append(name)
            continue

        # PLY → OBJ
        print("PLY→OBJ...", end=" ", flush=True)
        try:
            mesh = trimesh.load(ply_local, force="mesh")
            mesh.export(obj_path)
            print("✅")
            ok_cnt += 1
        except Exception as e:
            print(f"❌ trimesh error: {e}")
            fail_list.append(name)

    print(f"\n完成: {ok_cnt} 下载转换, {skip_cnt} 已有跳过, {len(fail_list)} 失败")
    if fail_list:
        print(f"失败: {fail_list}")
    if ok_cnt + skip_cnt > 0:
        print(f"\n▶ 下一步: 运行 Isaac Sim 批量转 USD")
        print(f"  sim45 assets/convert_missing_oakink_usd.py")


if __name__ == "__main__":
    main()
