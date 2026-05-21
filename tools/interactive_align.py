#!/usr/bin/env python3
"""
interactive_align.py — 人工旋转对齐工具
=========================================
对每个 OakInk 物体显示 USD mesh + 世界坐标轴，
通过键盘旋转到"正立于 XY 平面"，按 Enter 保存。

保存内容 (_meta.json 里的 R_align):
  - R_align_euler   : [rx, ry, rz] (度), 可读性好
  - R_align_matrix  : 3x3 矩阵 (精确)
  - z_offset_m      : 旋转后重新计算的底面偏移
  - T_ply_to_sim    : 4x4 完整变换 (供 HP 对齐使用)

键位:
  X / Y / Z    : 选择旋转轴 (高亮显示)
  ↑ / ↓        : 绕选中轴 +step° / -step°
  [ / ]        : 步长切换 1° / 5° / 10° / 45°
  R            : 重置旋转
  Enter        : 保存并进入下一个物体
  S            : 跳过（物体已经正立，不写入）
  B            : 回到上一个物体
  Q            : 退出

用法:
  python3 tools/interactive_align.py
  python3 tools/interactive_align.py --obj A01026        # 单个
  python3 tools/interactive_align.py --skip-done         # 跳过已有 R_align 的
  python3 tools/interactive_align.py --start-from A02011 # 从指定物体开始
"""
import os, sys, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

PROJ     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_ROOT = os.path.join(PROJ, 'output', 'obj_usd')

# ── 步长循环 ──────────────────────────────────────────────────────────────────
STEPS = [1.0, 5.0, 10.0, 45.0]

# ── 全局状态 ──────────────────────────────────────────────────────────────────
state = {
    'axis':      'Y',          # 当前旋转轴
    'step_idx':   1,           # STEPS 索引，默认 5°
    'R':          np.eye(3),   # 当前累积旋转
    'euler':      [0., 0., 0.],
    'obj_id':     '',
    'mesh':       None,        # open3d TriangleMesh（原始 USD mesh）
    'vis_mesh':   None,        # 当前显示的旋转后 mesh
    'coord':      None,        # 坐标轴
    'vis':        None,        # Visualizer
    'save_flag':  False,
    'skip_flag':  False,
    'back_flag':  False,
    'quit_flag':  False,
}


# ── USD mesh 加载 ─────────────────────────────────────────────────────────────
def load_usd_mesh_o3d(obj_id):
    from pxr import Usd, UsdGeom
    for root, _, files in os.walk(USD_ROOT):
        for f in files:
            if f == f'{obj_id}.usd':
                usd_path = os.path.join(root, f)
                stage = Usd.Stage.Open(usd_path)
                all_v, all_f, offset = [], [], 0
                for prim in stage.Traverse():
                    m = UsdGeom.Mesh(prim)
                    if not m:
                        continue
                    pts = m.GetPointsAttr().Get()
                    idx = m.GetFaceVertexIndicesAttr().Get()
                    if pts is None or idx is None:
                        continue
                    v  = np.array(pts, dtype=np.float64)
                    ff = np.array(idx, dtype=np.int32).reshape(-1, 3)
                    all_v.append(v)
                    all_f.append(ff + offset)
                    offset += len(v)
                if not all_v:
                    return None, None
                verts = np.concatenate(all_v)
                faces = np.concatenate(all_f)
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices  = o3d.utility.Vector3dVector(verts)
                mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
                mesh_o3d.compute_vertex_normals()
                mesh_o3d.paint_uniform_color([0.7, 0.7, 0.8])
                return mesh_o3d, usd_path
    return None, None


def meta_path_for(usd_path):
    return usd_path.replace('.usd', '_meta.json')


def load_meta(usd_path):
    mp = meta_path_for(usd_path)
    if os.path.exists(mp):
        return json.load(open(mp))
    return {}


def save_meta(usd_path, meta):
    mp = meta_path_for(usd_path)
    with open(mp, 'w') as f:
        json.dump(meta, f, indent=2)


# ── 旋转 mesh 并更新显示 ──────────────────────────────────────────────────────
def apply_rotation(vis, reset_camera=False):
    R   = state['R']
    src = state['mesh']

    # 旋转顶点
    verts = np.asarray(src.vertices).copy()
    verts = (R @ verts.T).T

    if state['vis_mesh'] is None or reset_camera:
        # ── 首次 / 重置：重新建立几何并设定相机 ─────────────────────────────
        rotated = o3d.geometry.TriangleMesh()
        rotated.vertices  = o3d.utility.Vector3dVector(verts)
        rotated.triangles = src.triangles
        rotated.compute_vertex_normals()
        rotated.paint_uniform_color([0.7, 0.7, 0.85])

        ext_max = float(np.ptp(verts, axis=0).max())
        frame   = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=ext_max * 0.60, origin=[0, 0, 0])

        vis.clear_geometries()
        vis.add_geometry(rotated, reset_bounding_box=True)
        vis.add_geometry(frame,   reset_bounding_box=False)

        state['vis_mesh'] = rotated
        state['coord']    = frame

        if reset_camera:
            ctr = vis.get_view_control()
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_front([0.5, -1.0, 0.6])
            ctr.set_zoom(0.65)
    else:
        # ── 后续旋转：原地更新顶点，不动相机 ────────────────────────────────
        m = state['vis_mesh']
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.compute_vertex_normals()
        vis.update_geometry(m)

    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    state['euler'] = list(euler)
    _print_status()


def _print_status():
    e = state['euler']
    step = STEPS[state['step_idx']]
    axis = state['axis']
    print(f"\r  [{state['obj_id']}]  轴={axis}  步长={step:.0f}°  "
          f"euler=[{e[0]:+.1f}, {e[1]:+.1f}, {e[2]:+.1f}]°    "
          "  Enter=保存  S=跳过  B=上一个  R=重置  Q=退出",
          end='', flush=True)


# ── 键盘回调 ──────────────────────────────────────────────────────────────────
def _rotate(delta_deg):
    axis = state['axis'].lower()
    vec  = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}[axis]
    dR   = Rotation.from_rotvec(np.deg2rad(delta_deg) * np.array(vec)).as_matrix()
    state['R'] = dR @ state['R']
    apply_rotation(state['vis'], reset_camera=False)  # 旋转时保持当前视角

def cb_x(vis):   state['axis'] = 'X'; _print_status()
def cb_y(vis):   state['axis'] = 'Y'; _print_status()
def cb_z(vis):   state['axis'] = 'Z'; _print_status()

def cb_up(vis):    _rotate(+STEPS[state['step_idx']])
def cb_down(vis):  _rotate(-STEPS[state['step_idx']])
def cb_left(vis):  _rotate(-STEPS[state['step_idx']])
def cb_right(vis): _rotate(+STEPS[state['step_idx']])

def cb_step(vis):
    state['step_idx'] = (state['step_idx'] + 1) % len(STEPS)
    _print_status()

def cb_reset(vis):
    state['R'] = np.eye(3)
    apply_rotation(vis, reset_camera=True)  # 重置时也重置视角

def cb_enter(vis):
    state['save_flag'] = True
    vis.close()

def cb_skip(vis):
    state['skip_flag'] = True
    vis.close()

def cb_back(vis):
    state['back_flag'] = True
    vis.close()

def cb_quit(vis):
    state['quit_flag'] = True
    vis.close()


# ── 计算 z_offset（旋转后的 mesh）────────────────────────────────────────────
def compute_z_offset_rotated(mesh_o3d, R):
    verts = (R @ np.asarray(mesh_o3d.vertices).T).T
    z_min = verts[:, 2].min()
    # z_offset = -z_min (把底面放在 z=0)
    return float(max(-z_min, 0.005))


# ── T_ply_to_sim = R_align @ T_YZ ────────────────────────────────────────────
def compute_T_ply_to_sim(R_align):
    # Y→Z 轴对换 (AssetConverter 做的): 绕 X 轴 -90°
    T_YZ = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    R_total = R_align @ T_YZ
    T = np.eye(4)
    T[:3, :3] = R_total
    return T.tolist()


# ── 主流程 ───────────────────────────────────────────────────────────────────
def process_object(obj_id, usd_path):
    mesh_o3d, _ = load_usd_mesh_o3d(obj_id)
    if mesh_o3d is None:
        print(f'  ❌ {obj_id}: USD mesh 读取失败')
        return 'skip'

    # 重置状态
    state.update({
        'R': np.eye(3), 'euler': [0.,0.,0.],
        'obj_id': obj_id, 'mesh': mesh_o3d,
        'vis_mesh': None, 'coord': None,           # 确保新物体重建几何
        'save_flag': False, 'skip_flag': False,
        'back_flag': False, 'quit_flag': False,
    })

    # 预加载已有旋转
    meta = load_meta(usd_path)
    if 'R_align_matrix' in meta:
        prev_R = np.array(meta['R_align_matrix'])
        state['R'] = prev_R
        print(f'\n  (已有旋转: {np.array(meta.get("R_align_euler",[0,0,0])).round(1)}°)')

    # 建立 Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Align: {obj_id}  |  X/Y/Z + ↑↓ 旋转  |  Enter保存  S跳过  B上一个  Q退出',
                      width=1100, height=700)
    state['vis'] = vis

    # 注册键位 (Open3D key codes)
    vis.register_key_callback(ord('X'), cb_x)
    vis.register_key_callback(ord('Y'), cb_y)
    vis.register_key_callback(ord('Z'), cb_z)
    vis.register_key_callback(ord('x'), cb_x)
    vis.register_key_callback(ord('y'), cb_y)
    vis.register_key_callback(ord('z'), cb_z)
    vis.register_key_callback(265, cb_up)       # ↑
    vis.register_key_callback(264, cb_down)     # ↓
    vis.register_key_callback(263, cb_left)     # ←
    vis.register_key_callback(262, cb_right)    # →
    vis.register_key_callback(ord('['), cb_step)
    vis.register_key_callback(ord(']'), cb_step)
    vis.register_key_callback(ord('R'), cb_reset)
    vis.register_key_callback(ord('r'), cb_reset)
    vis.register_key_callback(257, cb_enter)    # Enter
    vis.register_key_callback(ord('S'), cb_skip)
    vis.register_key_callback(ord('s'), cb_skip)
    vis.register_key_callback(ord('B'), cb_back)
    vis.register_key_callback(ord('b'), cb_back)
    vis.register_key_callback(ord('Q'), cb_quit)
    vis.register_key_callback(ord('q'), cb_quit)

    apply_rotation(vis, reset_camera=True)  # 初始显示：Z↑ XY水平视角
    vis.run()
    vis.destroy_window()

    # ── 处理结果 ──────────────────────────────────────────────────────────────
    if state['quit_flag']:
        return 'quit'
    if state['back_flag']:
        return 'back'
    if state['skip_flag']:
        print(f'\n  ⏭️  {obj_id} 跳过')
        return 'skip'

    if state['save_flag']:
        R     = state['R']
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        z_off = compute_z_offset_rotated(mesh_o3d, R)
        T_sim = compute_T_ply_to_sim(R)

        meta.update({
            'obj_id':        obj_id,
            'R_align_euler': [round(float(e), 4) for e in euler],
            'R_align_matrix': R.tolist(),
            'z_offset_m':    round(z_off, 6),
            'source':        'interactive_align',
            'T_ply_to_sim':  T_sim,
            'note':          'R_align: apply to USD mesh at runtime to get upright pose; z_offset recomputed after R_align',
        })
        save_meta(usd_path, meta)
        print(f'\n  ✅ {obj_id} 保存: euler={[round(float(e),1) for e in euler]}°  z_offset={z_off*100:.2f}cm')
        return 'saved'

    return 'skip'


def get_all_oakink():
    usd_dir = os.path.join(USD_ROOT, 'oakink')
    if not os.path.exists(usd_dir):
        return []
    ids = sorted(f.replace('.usd','') for f in os.listdir(usd_dir)
                 if f.endswith('.usd') and '_meta' not in f)
    paths = [os.path.join(usd_dir, f'{i}.usd') for i in ids]
    return list(zip(ids, paths))


def main():
    parser = argparse.ArgumentParser(description='Interactive mesh alignment tool')
    parser.add_argument('--obj',         default=None, help='只处理单个物体')
    parser.add_argument('--skip-done',   action='store_true',
                        help='跳过已有 R_align_matrix 的物体')
    parser.add_argument('--start-from',  default=None,
                        help='从指定物体 ID 开始')
    args = parser.parse_args()

    if args.obj:
        pairs = [(args.obj, None)]
        for root, _, files in os.walk(USD_ROOT):
            for f in files:
                if f == f'{args.obj}.usd':
                    pairs = [(args.obj, os.path.join(root, f))]
                    break
    else:
        pairs = get_all_oakink()

    if args.start_from:
        ids = [p[0] for p in pairs]
        if args.start_from in ids:
            pairs = pairs[ids.index(args.start_from):]

    if args.skip_done:
        filtered = []
        for obj_id, usd_path in pairs:
            if usd_path is None:
                continue
            meta = load_meta(usd_path)
            if 'R_align_matrix' not in meta:
                filtered.append((obj_id, usd_path))
        pairs = filtered
        print(f'  跳过已对齐，剩余 {len(pairs)} 个物体')

    print(f'\n{"="*60}')
    print(f'Interactive Align Tool — {len(pairs)} 个物体')
    print(f'  X/Y/Z + ↑↓ 旋转  |  [/] 切换步长')
    print(f'  Enter 保存  S 跳过  B 上一个  Q 退出')
    print(f'{"="*60}\n')

    i = 0
    saved = skipped = 0
    while i < len(pairs):
        obj_id, usd_path = pairs[i]
        if usd_path is None:
            print(f'  [{i+1}/{len(pairs)}] ❌ {obj_id}: USD 路径未找到')
            i += 1
            continue

        print(f'\n[{i+1}/{len(pairs)}] {obj_id}')
        result = process_object(obj_id, usd_path)

        if result == 'quit':
            break
        elif result == 'back':
            i = max(0, i - 1)
        elif result == 'saved':
            saved += 1
            i += 1
        else:   # skip
            skipped += 1
            i += 1

    print(f'\n\n{"="*60}')
    print(f'完成: {saved} 保存  {skipped} 跳过  {len(pairs)-saved-skipped} 未处理')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
