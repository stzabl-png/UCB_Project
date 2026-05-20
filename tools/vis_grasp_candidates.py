#!/usr/bin/env python3
"""
vis_grasp_candidates.py — 抓取候选精细可视化
==============================================
显示:
  ● 受力中心点  (CoM)         — 橙色星号
  ● 抓取点     (Grasp Point)  — 白色圆点
  ● 夹爪位置   (Finger tips)  — 青色方块 × 2
  ● 腕部位置   (Wrist / EE)   — 黄色菱形
  → 接近方向箭头 + 夹爪宽度线
"""
import os, sys, json, argparse
import numpy as np
import h5py
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

PROJ    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJ, 'output', 'grasp_vis')

# TCP 偏移: Franka panda_hand → 指尖中点距离 (m)
TCP_OFFSET = 0.105

# ── mesh 查找（与 random_grasp_sampler 一致）─────────────────────────────────
OBJ_MESHES_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']

def find_obj_mesh(obj_id):
    for ds in DATASETS:
        mp  = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'mesh.ply')
        sp  = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
        if not os.path.exists(mp):
            continue
        sf = 1.0
        if os.path.exists(sp):
            with open(sp) as f:
                sf = float(json.load(f)['scale_factor'])
        return mp, sf
    return None, 1.0


def load_mesh(obj_id, no_rotation=False):
    """从 obj_meshes/ 加载并应用 scale；可选 per-object rotation.json。"""
    mp, sf = find_obj_mesh(obj_id)
    if mp is None:
        sys.exit(f'❌ obj_meshes/ 中未找到 {obj_id}')
    mesh = trimesh.load(mp, force='mesh')
    if abs(sf - 1.0) > 1e-6:
        mesh.vertices *= sf
    if no_rotation:
        return mesh
    # ── per-object rotation.json (pca_z / stable_pose 等) ───────────────────
    rot_applied = False
    for ds in DATASETS:
        rot_path = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'rotation.json')
        if os.path.exists(rot_path):
            rot_data = json.load(open(rot_path))
            rot_euler = rot_data.get('euler_xyz_deg')
            if rot_euler and any(abs(e) > 0.5 for e in rot_euler):
                R = Rotation.from_euler('xyz', rot_euler, degrees=True).as_matrix()
                mesh.vertices = (R @ mesh.vertices.T).T
            rot_applied = True
            break
    # fallback: 旧全局 canonical_rotation.json
    if not rot_applied:
        cr_path = os.path.join(PROJ, 'sim', 'canonical_rotation.json')
        if os.path.exists(cr_path):
            cr = json.load(open(cr_path))
            rot_euler = cr.get(obj_id)
            if rot_euler and any(abs(e) > 0.5 for e in rot_euler):
                R = Rotation.from_euler('xyz', rot_euler, degrees=True).as_matrix()
                mesh.vertices = (R @ mesh.vertices.T).T
    return mesh


def hdf5_no_rotation_flag(hdf5_path):
    """从候选 HDF5 metadata 读取 no_rotation；无字段则返回 None。"""
    with h5py.File(hdf5_path, 'r') as f:
        if 'metadata' not in f:
            return None
        if 'no_rotation' not in f['metadata'].attrs:
            return None
        return bool(f['metadata'].attrs['no_rotation'])


def load_success_names_from_robot_gt(robot_gt_path):
    """从 sim 输出的 *_robot_gt.hdf5 读取 successful_grasps 的 name。"""
    names = []
    with h5py.File(robot_gt_path, 'r') as f:
        if 'successful_grasps' not in f:
            return names
        for key in sorted(f['successful_grasps'].keys()):
            g = f[f'successful_grasps/{key}']
            names.append(g.attrs.get('name', key))
    return names



# ── HDF5 读取 ─────────────────────────────────────────────────────────────────

def load_candidates(hdf5_path):
    cands = []
    with h5py.File(hdf5_path, 'r') as f:
        n = f['candidates'].attrs['n_candidates']
        for i in range(n):
            g     = f[f'candidates/candidate_{i}']
            pos   = g['position'][:]           # 指尖中点 (grasp point)
            rot   = g['rotation'][:]           # 3×3: col0=finger, col1=palm, col2=approach
            width = float(g.attrs.get('gripper_width', 0.05))
            score = float(g.attrs.get('score', 0.0))
            name  = g.attrs.get('name', f'raycast_{i}')
            approach = rot[:, 2]
            finger   = rot[:, 0]
            cands.append(dict(idx=i, name=name, pos=pos, rot=rot,
                              approach=approach, finger=finger,
                              width=width, score=score))
    return cands


# ── 单图绘制 ──────────────────────────────────────────────────────────────────

def draw_grasp(ax, mesh, cand, success_labels, n_mesh_faces=3000):
    # ── mesh 表面点云 ─────────────────────────────────────────────────────────
    pts, _ = trimesh.sample.sample_surface(mesh, 8000)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c='#5ab4d4', s=2.5, alpha=0.55,
               linewidths=0, zorder=2, depthshade=False)

    pos      = cand['pos']       # 抓取点 (指尖中点)
    approach = cand['approach']  # 接近方向单位向量
    finger   = cand['finger']    # 夹爪开合方向
    width    = cand['width']
    hw       = width / 2.0

    is_ok = cand['name'] in success_labels
    ac    = '#2ca02c' if is_ok else '#d62728'   # 主色 (成功=绿 失败=红)

    # ── 1. 受力中心点 (CoM) — 橙色星号 ──────────────────────────────────────
    com = mesh.center_mass
    ax.scatter(*com, c='#ff7f0e', s=120, marker='*', zorder=6,
               label='CoM (受力中心)', edgecolors='white', linewidths=0.4)

    # ── 2. 抓取点 (Grasp Point) — 白色圆点 ──────────────────────────────────
    ax.scatter(*pos, c='white', s=90, zorder=7,
               label='抓取点', edgecolors=ac, linewidths=1.2)

    # ── 3. 夹爪位置 (Finger Tips) — 两个青色方块 ────────────────────────────
    fl = pos - hw * finger    # 左指
    fr = pos + hw * finger    # 右指
    for fp, lbl in [(fl, '指L'), (fr, '指R')]:
        ax.scatter(*fp, c='#17becf', s=60, marker='s', zorder=6,
                   label=lbl, edgecolors='white', linewidths=0.4)
    # 夹爪横线
    ax.plot([fl[0], fr[0]], [fl[1], fr[1]], [fl[2], fr[2]],
            color='#17becf', linewidth=2.0, alpha=0.9)

    # ── 4. 腕部位置 (Wrist / EE) — 黄色菱形 ────────────────────────────────
    # 可视化用 pre-grasp 偏移 (3cm) 而非完整 TCP_OFFSET(10.5cm)，避免超出画框
    # 标注中注明真实 TCP 偏移
    viz_offset = 0.05   # 显示偏移 5cm
    wrist = pos - approach * viz_offset
    ax.scatter(*wrist, c='#ffdd57', s=100, marker='D', zorder=6,
               label=f'腕部 EE (TCP={TCP_OFFSET*100:.0f}cm)', edgecolors='#888', linewidths=0.4)

    # 腕部 → 抓取点 连线（虚线，表示 pre-grasp 方向）
    ax.plot([wrist[0], pos[0]], [wrist[1], pos[1]], [wrist[2], pos[2]],
            color='#ffdd57', linewidth=1.5, linestyle='--', alpha=0.7)

    # ── 5. 接近方向箭头 ──────────────────────────────────────────────────────
    arr_len = 0.025
    ax.quiver(*pos, *approach, length=arr_len, color=ac,
              arrow_length_ratio=0.4, linewidth=2.2)

    # ── 6. CoM → 抓取点 偏差线 ──────────────────────────────────────────────
    ax.plot([com[0], pos[0]], [com[1], pos[1]], [com[2], pos[2]],
            color='#ff7f0e', linewidth=1.0, linestyle=':', alpha=0.5)

    return com, pos, fl, fr, wrist


def make_fig(mesh, cand, success_labels):
    fig = plt.figure(figsize=(7, 7), facecolor='#1a1a2e')
    ax  = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    com, pos, fl, fr, wrist = draw_grasp(ax, mesh, cand, success_labels)

    # ── 坐标轴范围 ───────────────────────────────────────────────────────────
    b  = mesh.bounds
    cx, cy, cz = (b[0] + b[1]) / 2
    r  = max(b[1] - b[0]) * 0.85   # 稍大，确保腕部也在画框内
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(b[0][2] - r*0.2, b[0][2] + 2*r)

    ax.set_xlabel('X', color='#aaa', fontsize=9)
    ax.set_ylabel('Y', color='#aaa', fontsize=9)
    ax.set_zlabel('Z', color='#aaa', fontsize=9)
    ax.tick_params(colors='#555', labelsize=7)

    is_ok  = cand['name'] in success_labels
    status = 'SUCCESS' if is_ok else 'FAILED'
    clr    = '#2ca02c' if is_ok else '#d62728'

    com_dist = np.linalg.norm(pos - com) * 100  # cm

    title = (
        f"[{cand['idx']+1:02d}]  {cand['name']}  score={cand['score']:.1f}  [{status}]\n"
        f"approach=({cand['approach'][0]:+.2f},{cand['approach'][1]:+.2f},{cand['approach'][2]:+.2f})  "
        f"w={cand['width']*100:.1f}cm  CoM偏差={com_dist:.1f}cm"
    )
    ax.set_title(title, color=clr, fontsize=8.5, pad=10)

    # ── 图例 (右下) ──────────────────────────────────────────────────────────
    legend_items = [
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#ff7f0e',
                   markersize=10, label=f'受力中心 CoM {com*100}cm'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
                   markersize=8,  label=f'抓取点 {pos*100}cm'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#17becf',
                   markersize=7,  label=f'夹爪指尖 w={cand["width"]*100:.1f}cm'),
        plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#ffdd57',
                   markersize=7,  label=f'腕部 EE'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=6.5,
              facecolor='#2a2a3e', edgecolor='#555', labelcolor='#ccc')

    ax.view_init(elev=20, azim=130)
    plt.tight_layout()
    return fig


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5',    required=True, help='候选 HDF5 文件')
    parser.add_argument('--robot-gt', default=None,
                        help='sim 输出的 *_robot_gt.hdf5；自动标绿 successful_grasps')
    parser.add_argument('--success', nargs='*', default=[],
                        help='成功的候选 name（与 --robot-gt 可叠加）')
    parser.add_argument('--only-success', action='store_true',
                        help='只生成成功候选的 PNG（需 --robot-gt 或 --success）')
    parser.add_argument('--no-rotation', action='store_true',
                        help='不应用 rotation.json（与 sampler --no-rotation 一致）')
    parser.add_argument('--top',     type=int, default=None, help='只生成前 N 个候选')
    parser.add_argument('--outdir',  default=OUT_DIR)
    args = parser.parse_args()

    success_labels = set(args.success)
    if args.robot_gt:
        if not os.path.isfile(args.robot_gt):
            sys.exit(f'❌ robot_gt 不存在: {args.robot_gt}')
        from_gt = load_success_names_from_robot_gt(args.robot_gt)
        success_labels.update(from_gt)
        print(f'robot_gt: {len(from_gt)} 成功 → {from_gt}')

    obj_id = os.path.basename(args.hdf5).replace('_grasp.hdf5', '')
    no_rotation = args.no_rotation
    if not no_rotation:
        detected = hdf5_no_rotation_flag(args.hdf5)
        if detected is not None:
            no_rotation = detected
    mesh = load_mesh(obj_id, no_rotation=no_rotation)
    rot_note = 'raw SAM3D + scale' if no_rotation else 'scale + rotation.json'
    print(f'mesh ({rot_note}) 尺寸 (cm): {mesh.bounding_box.extents*100}')
    print(f'CoM: {mesh.center_mass*100} cm')

    cands = load_candidates(args.hdf5)
    if args.only_success:
        if not success_labels:
            sys.exit('❌ --only-success 需要 --robot-gt 或 --success')
        cands = [c for c in cands if c['name'] in success_labels]
        if not cands:
            sys.exit(f'❌ 候选 HDF5 中未找到成功 name: {success_labels}')
    print(f'物体: {obj_id}  绘制: {len(cands)}  成功标记: {success_labels}')

    outdir = os.path.join(args.outdir, obj_id)
    if args.only_success:
        outdir = os.path.join(outdir, 'success')
    os.makedirs(outdir, exist_ok=True)

    top_n = args.top or len(cands)
    for cand in cands[:top_n]:
        fig   = make_fig(mesh, cand, success_labels)
        fname = os.path.join(outdir, f"{cand['idx']+1:02d}_{cand['name']}.png")
        fig.savefig(fname, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        s = '✅' if cand['name'] in success_labels else '❌'
        print(f"  {s} [{cand['idx']+1:02d}] {cand['name']}  score={cand['score']:.1f}"
              f"  → {os.path.basename(fname)}")

    # 总览图 (4×5)
    imgs = sorted(f for f in os.listdir(outdir)
                  if f.endswith('.png') and f != 'overview.png')
    if len(imgs) >= 20:
        from PIL import Image
        tiles = [Image.open(os.path.join(outdir, f)) for f in imgs[:20]]
        w, h  = tiles[0].size
        ov    = Image.new('RGB', (w*5, h*4), (26, 26, 46))
        for i, t in enumerate(tiles):
            r, c = divmod(i, 5)
            ov.paste(t, (c*w, r*h))
        ov_path = os.path.join(outdir, 'overview.png')
        ov.save(ov_path)
        print(f'\n📊 总览图 → {ov_path}')

    print(f'\n✅ 完成: {outdir}')


if __name__ == '__main__':
    main()
