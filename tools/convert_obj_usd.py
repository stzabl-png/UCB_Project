#!/usr/bin/env python3
"""
convert_obj_usd.py
==================
将 data_hub/ProcessedData/obj_meshes/{dataset}/{obj_id}/mesh.ply
转换为 Isaac Sim 兼容的 USD 文件，输出到:
    output/obj_usd/{dataset}/{obj_id}.usd

转换流程:
  1. 加载 mesh.ply
  2. 应用 scale_factor (来自 scale.json) → 米制单位
  3. 应用 canonical_rotation (来自 sim/canonical_rotation.json) → 正确朝向
  4. 导出 USD (upAxis=Z, metersPerUnit=1.0)

用法:
    # 单个物体 (自动找数据集)
    python3 tools/convert_obj_usd.py --obj A16013

    # 单个物体 + 指定数据集
    python3 tools/convert_obj_usd.py --obj A16013 --dataset oakink

    # 批量: 整个数据集
    python3 tools/convert_obj_usd.py --dataset oakink
    python3 tools/convert_obj_usd.py --dataset ycb

    # 全部数据集
    python3 tools/convert_obj_usd.py --all
"""
import os, sys, json, argparse
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_MESHES_DIR    = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
OBJ_USD_DIR       = os.path.join(PROJ, 'output', 'obj_usd')
CANONICAL_ROT_JSON = os.path.join(PROJ, 'sim', 'canonical_rotation.json')  # legacy, no longer used

DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_canonical_rotations():
    if os.path.exists(CANONICAL_ROT_JSON):
        with open(CANONICAL_ROT_JSON) as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith('_')}
    return {}


def find_obj_mesh(obj_id, dataset=None):
    """在 obj_meshes/ 中查找 mesh.ply、scale.json 和 rotation.json."""
    datasets = [dataset] if dataset else DATASETS
    for ds in datasets:
        mesh_path  = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'mesh.ply')
        scale_path = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
        if not os.path.exists(mesh_path):
            continue
        scale_factor = 1.0
        if os.path.exists(scale_path):
            with open(scale_path) as f:
                scale_factor = float(json.load(f)['scale_factor'])
        return mesh_path, scale_factor, ds
    return None, 1.0, None


def load_canonical_rotations():
    """Legacy: kept for backward compatibility only. No longer used in main pipeline."""
    return {}


def load_obj_rotation(obj_id, dataset, canonical_rotations_global):
    """
    读取物体的 canonical rotation (Euler XYZ, degrees).
    唯一来源: per-object rotation.json (estimate_obj_rotation.py 生成)
    """
    rot_path = os.path.join(OBJ_MESHES_DIR, dataset, obj_id, 'rotation.json')
    if os.path.exists(rot_path):
        data = json.load(open(rot_path))
        euler = data.get('euler_xyz_deg', None)
        if euler is not None:
            return euler, f'rotation.json ({data.get("method","?")})'
    return None, 'none'


def list_dataset_objs(dataset):
    ds_dir = os.path.join(OBJ_MESHES_DIR, dataset)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(
        o for o in os.listdir(ds_dir)
        if os.path.exists(os.path.join(ds_dir, o, 'mesh.ply'))
    )


def convert_one(obj_id, mesh_path, scale_factor, dataset, canonical_rotations, force=False, no_rotation=False):
    """将单个物体 PLY → USD."""
    out_dir  = os.path.join(OBJ_USD_DIR, dataset)
    out_path = os.path.join(out_dir, f'{obj_id}.usd')

    if os.path.exists(out_path) and not force:
        print(f'  ⏭️  已存在: {out_path}')
        return True

    os.makedirs(out_dir, exist_ok=True)

    try:
        import trimesh
        from pxr import Usd, UsdGeom, Vt, Gf
    except ImportError as e:
        print(f'  ❌ 缺少依赖: {e}')
        return False

    # ── 1. 加载 PLY ──────────────────────────────────────────────────────────
    mesh = trimesh.load(mesh_path, force='mesh')

    # ── 2. 应用 scale_factor → 米制 ─────────────────────────────────────────
    if abs(scale_factor - 1.0) > 1e-6:
        mesh.vertices = mesh.vertices * scale_factor

    # ── 3. 应用 canonical rotation ─────────────────────────────────────────────
    if no_rotation:
        rot_euler, rot_source = None, 'skipped (--no-rotation)'
        print(f'     [rotation: identity  source={rot_source}]')
    else:
        rot_euler, rot_source = load_obj_rotation(obj_id, dataset, canonical_rotations)
        if rot_euler is not None and any(abs(e) > 0.5 for e in rot_euler):
            from scipy.spatial.transform import Rotation as _R
            R_mat = _R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
            mesh.vertices = (R_mat @ mesh.vertices.T).T
            print(f'     [rotation {rot_euler}  source={rot_source}]')
        else:
            print(f'     [rotation: identity  source={rot_source}]')

    # ── 4. 写出 USD ──────────────────────────────────────────────────────────
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata('metersPerUnit', 1.0)

    root_xform = UsdGeom.Xform.Define(stage, '/Root')
    stage.SetDefaultPrim(root_xform.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, '/Root/Mesh')

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    mesh_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh_prim.GetFaceVertexCountsAttr().Set(
        Vt.IntArray([3] * len(faces))
    )
    mesh_prim.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(faces.flatten().tolist())
    )

    # 法线
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        normals = mesh.face_normals.astype(np.float32)
        mesh_prim.GetNormalsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals])
        )
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.uniform)

    stage.Save()

    # ── 5. 保存 companion meta JSON（z_offset 供 Sim 精确放置）────────────────
    vmin = verts.min(axis=0)   # (3,) x/y/z 最小值
    vmax = verts.max(axis=0)   # (3,) x/y/z 最大值
    z_offset = float(-vmin[2]) if vmin[2] < 0 else 0.0   # 抬起到 z_min=0
    meta = {
        'obj':          obj_id,
        'dataset':      dataset,
        'scale_factor': scale_factor,
        'z_offset_m':   round(z_offset, 6),   # 放置时 Z 轴偏移（米），使底面在桌面
        'bbox_min':     [round(float(v), 6) for v in vmin],
        'bbox_max':     [round(float(v), 6) for v in vmax],
        'bbox_extent_cm': [round(float(v)*100, 2) for v in (vmax - vmin)],
        'no_rotation': bool(no_rotation),
    }
    meta_path = os.path.join(out_dir, f'{obj_id}_meta.json')
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)

    ext = vmax - vmin
    print(f'  ✅  {out_path}')
    print(f'      尺寸 (cm): {ext[0]*100:.1f}×{ext[1]*100:.1f}×{ext[2]*100:.1f}   scale={scale_factor:.6f}   z_offset={z_offset*100:.1f}cm')
    return True


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='obj_meshes PLY → USD 转换器')
    parser.add_argument('--obj',     help='单个物体 ID (自动找数据集)')
    parser.add_argument('--dataset', help='指定数据集: oakink / ycb / arctic / dexycb / egocentric')
    parser.add_argument('--all',     action='store_true', help='转换所有数据集')
    parser.add_argument('--force',   action='store_true', help='覆盖已有 USD')
    parser.add_argument('--no-rotation', action='store_true',
                        help='不应用 rotation.json，导出 SAM3D 原始朝向')
    args = parser.parse_args()

    canonical_rotations = load_canonical_rotations()
    print(f'Canonical rotations loaded: {len(canonical_rotations)} entries')

    # ── 构建待处理列表 ───────────────────────────────────────────────────────
    todo = []  # [(obj_id, mesh_path, scale_factor, dataset), ...]

    if args.obj:
        mesh_path, sf, ds = find_obj_mesh(args.obj, dataset=args.dataset)
        if mesh_path is None:
            print(f'❌ 未在 obj_meshes/ 中找到: {args.obj}')
            return
        todo.append((args.obj, mesh_path, sf, ds))

    elif args.dataset or args.all:
        target_ds = DATASETS if args.all else [args.dataset]
        for ds in target_ds:
            for obj_id in list_dataset_objs(ds):
                mp, sf, _ = find_obj_mesh(obj_id, dataset=ds)
                if mp:
                    todo.append((obj_id, mp, sf, ds))
        print(f'待转换: {len(todo)} 个物体 (数据集: {target_ds})')

    else:
        parser.print_help()
        return

    # ── 执行转换 ─────────────────────────────────────────────────────────────
    ok = err = 0
    for i, (obj_id, mesh_path, sf, ds) in enumerate(todo):
        print(f'\n[{i+1}/{len(todo)}] {obj_id}  ({ds})')
        if convert_one(obj_id, mesh_path, sf, ds, canonical_rotations,
                       force=args.force, no_rotation=args.no_rotation):
            ok += 1
        else:
            err += 1

    print(f'\n{"="*50}')
    print(f'  完成: {ok} 成功  {err} 失败')
    print(f'  输出: {OBJ_USD_DIR}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
