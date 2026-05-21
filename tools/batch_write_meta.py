#!/usr/bin/env python3
"""
batch_write_meta.py — 批量为所有物体写入精确 z_offset 到 _meta.json

z_offset = mesh 质心 Z - mesh 最低点 Z（已 scale 到米制）

这保证：无论 SAM3D 原始姿态如何，物体底面都恰好落在桌面上。
物体若不稳定会自然落定，run_grasp_sim 里的 get_obj_pos()
会读取落定后的实际 pose 并正确 transform 抓取坐标。

无需重跑 Isaac Sim，纯 Python 操作，几秒完成。

用法:
    python3 tools/batch_write_meta.py              # 全部 oakink + ycb
    python3 tools/batch_write_meta.py --ds oakink  # 只处理 oakink
    python3 tools/batch_write_meta.py --obj A01026 # 只处理一个物体
"""
import os, sys, json, glob, argparse
import trimesh
import numpy as np

PROJ     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
USD_ROOT = os.path.join(PROJ, 'output', 'obj_usd')


def compute_z_offset_from_usd(usd_path: str, R_align: np.ndarray | None = None) -> float:
    """
    从 USD 文件的实际 mesh 顶点计算 z_offset。
    如果传入 R_align，先旋转顶点再计算（interactive_align 已对齐的物体用此路径）。
    """
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path)
    all_z = []
    for prim in stage.Traverse():
        m = UsdGeom.Mesh(prim)
        if m:
            pts = m.GetPointsAttr().Get()
            if pts:
                v = np.array(pts, dtype=np.float64)
                if R_align is not None:
                    v = (R_align @ v.T).T
                all_z.append(v[:, 2])
    if not all_z:
        return 0.075
    z_min = float(np.concatenate(all_z).min())
    return max(-z_min, 0.005)


def find_usd(obj_id: str) -> str | None:
    """在 output/obj_usd/ 下递归找 {obj_id}.usd。"""
    for root, _, files in os.walk(USD_ROOT):
        for f in files:
            if f == f'{obj_id}.usd':
                return os.path.join(root, f)
    return None


def write_meta(obj_id: str, mesh_path: str, scale_factor: float,
               usd_path: str, force: bool = False) -> tuple[bool, float | None]:
    meta_path = usd_path.replace('.usd', '_meta.json')

    # 读取已有 meta（保留 interactive_align 写入的 R_align 等字段）
    existing = {}
    if os.path.exists(meta_path):
        existing = json.load(open(meta_path))
        # 如果已经有 interactive_align 的旋转数据且 z_offset 也已计算，跳过
        if not force and 'z_offset_m' in existing and existing.get('source') == 'interactive_align':
            return False, None
        if not force and 'z_offset_m' in existing and existing.get('source') == 'usd_mesh_ralign':
            return False, None

    # 读取 R_align（如果有）
    R_align = None
    if 'R_align_matrix' in existing:
        R_align = np.array(existing['R_align_matrix'])

    z_off = compute_z_offset_from_usd(usd_path, R_align=R_align)

    # 合并：保留已有字段，更新 z_offset 和 source
    existing.update({
        'obj_id':       obj_id,
        'z_offset_m':   round(z_off, 6),
        'scale_factor': scale_factor,
        'source_mesh':  mesh_path,
        'source':       'usd_mesh_ralign' if R_align is not None else 'usd_mesh',
        'note':         'z_offset computed from USD vertices after R_align (if any)',
    })
    with open(meta_path, 'w') as f:
        json.dump(existing, f, indent=2)
    return True, z_off


def main():
    parser = argparse.ArgumentParser(
        description='批量写入精确 z_offset → _meta.json (SAM3D 原始姿态)')
    parser.add_argument('--ds',    default=None, help='只处理某个 dataset (oakink/ycb)')
    parser.add_argument('--obj',   default=None, help='只处理一个物体')
    parser.add_argument('--force', action='store_true', help='强制覆写已存在的 _meta.json')
    args = parser.parse_args()

    datasets = [args.ds] if args.ds else ['oakink', 'ycb']

    total = ok = skip = missing_usd = 0

    for ds in datasets:
        ds_dir = os.path.join(MESH_DIR, ds)
        if not os.path.isdir(ds_dir):
            continue

        obj_ids = [args.obj] if args.obj else sorted(os.listdir(ds_dir))

        for obj_id in obj_ids:
            obj_dir    = os.path.join(ds_dir, obj_id)
            mesh_path  = os.path.join(obj_dir, 'mesh.ply')
            scale_path = os.path.join(obj_dir, 'scale.json')

            if not os.path.exists(mesh_path):
                continue

            sf = float(json.load(open(scale_path)).get('scale_factor', 1.0)) \
                 if os.path.exists(scale_path) else 1.0

            usd_path = find_usd(obj_id)
            total += 1

            if usd_path is None:
                print(f'  ⚠️  {obj_id}: USD not found, skip')
                missing_usd += 1
                continue

            written, z_off = write_meta(obj_id, mesh_path, sf, usd_path, force=args.force)
            if written:
                print(f'  ✅ {obj_id:<16}  z_offset = {z_off*100:.2f} cm')
                ok += 1
            else:
                print(f'  ⏭️  {obj_id:<16}  already exists (use --force to overwrite)')
                skip += 1

    print(f'\n{"="*50}')
    print(f'Done: {ok} written  {skip} skipped  {missing_usd} missing USD  ({total} total)')


if __name__ == '__main__':
    main()
