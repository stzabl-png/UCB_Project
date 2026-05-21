#!/usr/bin/env python3
"""
vis_pose_sanity.py — 3-panel 姿态一致性检查

Panel 布局:
  左  : OakInk 原始视频帧（物体在数据集里的自然位姿）
  中  : mesh + top-10 抓取候选（SAM3D 原始坐标系，世界 Z↑）
  右  : Isaac Sim 100步落定后截图（物体实际放置姿态）

目的: 目视确认三图里物体姿态一致，即 SAM3D 原始位姿贯穿全流程。

用法:
    python3 tools/vis_pose_sanity.py --obj A01026
    python3 tools/vis_pose_sanity.py --obj A01026 --seq 0002   # 指定 sequence
    python3 tools/vis_pose_sanity.py --obj A01026 --azim 45    # 调整 3D 视角
"""
import os, sys, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# 中文字体支持
import matplotlib.font_manager as _fm
_cjk_candidates = [
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
for _cjk_path in _cjk_candidates:
    if os.path.exists(_cjk_path):
        _fm.fontManager.addfont(_cjk_path)
        matplotlib.rcParams['font.family'] = _fm.FontProperties(fname=_cjk_path).get_name()
        break


PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
try:
    import config
    MESH_DIR = os.path.join(config.DATA_HUB, 'ProcessedData', 'obj_meshes')
except Exception:
    MESH_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')

OAKINK_DIR   = os.path.expanduser('~/Project/OakInk')
GRASP_DIR    = os.path.join(PROJ, 'output', 'grasps_candidate')
SNAPSHOT_DIR = os.path.join(PROJ, 'output', 'settle_snapshots')

BG  = '#111111'
BG2 = '#1a1a1a'


# ── 辅助：加载 mesh（优先 USD，和 Sim 坐标系一致）────────────────────────────
def load_mesh_raw(obj_id: str):
    """优先从 USD 加载（与 Sim/grasp 坐标系一致），返回 (trimesh, dataset_str)。"""
    import trimesh
    from pxr import Usd, UsdGeom

    usd_root = os.path.join(PROJ, 'output', 'obj_usd')
    usd_path = None
    for root, _, files in os.walk(usd_root):
        for f in files:
            if f == f'{obj_id}.usd':
                usd_path = os.path.join(root, f)
                break
        if usd_path:
            break

    if usd_path is not None:
        try:
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
                v = np.array(pts, dtype=np.float32)
                f = np.array(idx, dtype=np.int32).reshape(-1, 3)
                all_v.append(v)
                all_f.append(f + offset)
                offset += len(v)
            if all_v:
                verts = np.concatenate(all_v, axis=0)
                faces = np.concatenate(all_f, axis=0)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                ds = 'oakink' if 'oakink' in usd_path else 'ycb'
                return mesh, ds
        except Exception:
            pass

    # fallback: PLY + scale
    for ds in ['oakink', 'ycb']:
        mesh_path  = os.path.join(MESH_DIR, ds, obj_id, 'mesh.ply')
        scale_path = os.path.join(MESH_DIR, ds, obj_id, 'scale.json')
        if not os.path.exists(mesh_path):
            continue
        sf = float(json.load(open(scale_path)).get('scale_factor', 1.0)) \
             if os.path.exists(scale_path) else 1.0
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices * sf
        return mesh, ds
    return None, None


# ── Panel 1: OakInk 原始视频帧 ───────────────────────────────────────────────
def find_oakink_frame(obj_id: str, seq: str | None = None) -> str | None:
    """
    在 OakInk stream_release_v2 里找中间帧（north 视角，能看到物体正面）。
    优先 north_east，其次 north_west。
    """
    seq_pat = f'{obj_id}_{seq}*' if seq else f'{obj_id}_*'
    for cam in ['north_east', 'north_west', 'south_east']:
        pattern = os.path.join(
            OAKINK_DIR, 'image', 'stream_release_v2',
            seq_pat, '*', f'{cam}_color_*.png')
        files = sorted(glob.glob(pattern))
        if files:
            return files[len(files) // 2]   # 取中间帧
    return None


def draw_panel1(ax, obj_id: str, seq: str | None):
    frame_path = find_oakink_frame(obj_id, seq)
    if frame_path:
        img = plt.imread(frame_path)
        ax.imshow(img)
        seq_label = os.path.basename(os.path.dirname(os.path.dirname(frame_path)))
        ax.set_title(f'① OakInk 原始帧\n{seq_label}',
                     color='white', fontsize=10, pad=6)
    else:
        ax.text(0.5, 0.5, 'frame not found\n(check OAKINK_DIR)',
                ha='center', va='center', color='#ff5252',
                fontsize=9, transform=ax.transAxes)
        ax.set_title('① OakInk 原始帧', color='white', fontsize=10, pad=6)
    ax.set_facecolor(BG2)
    ax.axis('off')


# ── Panel 2: mesh + 抓取候选 ─────────────────────────────────────────────────
def draw_world_axes(ax, origin, axis_len):
    """
    在 3D 轴上画世界坐标轴箭头：
      X = 红  (Right)
      Y = 绿  (Forward)
      Z = 蓝  (Up ↑)  ← 抓取候选过滤的基准轴
    """
    ox, oy, oz = origin
    # X 轴（红）
    ax.quiver(ox, oy, oz, axis_len, 0, 0,
              color='#ff4444', linewidth=2.5, arrow_length_ratio=0.15)
    ax.text(ox + axis_len * 1.12, oy, oz, 'X', color='#ff4444',
            fontsize=13, fontweight='bold', ha='center', va='center')
    # Y 轴（绿）
    ax.quiver(ox, oy, oz, 0, axis_len, 0,
              color='#44ff44', linewidth=2.5, arrow_length_ratio=0.15)
    ax.text(ox, oy + axis_len * 1.12, oz, 'Y', color='#44ff44',
            fontsize=13, fontweight='bold', ha='center', va='center')
    # Z 轴（蓝，世界向上 = 抓取过滤基准）
    ax.quiver(ox, oy, oz, 0, 0, axis_len,
              color='#4488ff', linewidth=3.0, arrow_length_ratio=0.15)
    ax.text(ox, oy, oz + axis_len * 1.18, 'Z\n(world up)', color='#4488ff',
            fontsize=13, fontweight='bold', ha='center', va='bottom')


def draw_panel2(ax, obj_id: str, elev: int = 20, azim: int = 135):
    import trimesh, h5py

    mesh, ds = load_mesh_raw(obj_id)
    if mesh is None:
        ax.text2D(0.5, 0.5, 'mesh not found', ha='center', va='center',
                  color='red', transform=ax.transAxes)
        ax.set_title(f'② {obj_id} mesh', color='white', fontsize=10, pad=6)
        return

    # 物体包围盒参数
    ext     = mesh.bounding_box.extents          # [dx, dy, dz]
    ext_max = float(ext.max())
    bounds  = mesh.bounds                        # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    center  = mesh.centroid

    # 点云（灰色）
    import trimesh as _tri
    pts, _ = _tri.sample.sample_surface(mesh, 10000)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c='#888888', s=0.8, alpha=0.55, rasterized=True)

    # ── 世界坐标轴：放在 mesh 底面中心正下方 ─────────────────────────────────
    axis_len = ext_max * 0.50    # 轴长 = 物体最长边 50%
    # 原点：X/Y 居中，Z 在底面下方一点
    ax_origin = np.array([
        center[0],
        center[1],
        bounds[0, 2] - axis_len * 0.15,   # 底面往下 15%
    ])
    draw_world_axes(ax, ax_origin, axis_len)

    # 显式设置轴范围，保证坐标轴箭头在视图内
    pad = axis_len * 1.4
    cx, cy, cz = center
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_zlim(bounds[0, 2] - axis_len * 0.3,
                bounds[1, 2] + axis_len * 0.5)

    # ── 抓取候选 ─────────────────────────────────────────────────────────────
    grasp_hdf5 = os.path.join(GRASP_DIR, f'{obj_id}_grasp.hdf5')
    n_shown = 0
    if os.path.exists(grasp_hdf5):
        arr_scale = ext_max * 0.20   # approach 箭头长度

        with h5py.File(grasp_hdf5, 'r') as f:
            cg = f.get('candidates', {})
            shown_keys = list(cg.keys())[:10]
            for key in shown_keys:
                ci = cg[key]
                gp  = ci['position'][:]          # 抓取中心（USD坐标系，米制）
                rot = ci['rotation'][:]
                app = rot[:, 2]                  # approach dir：夹爪进入物体的方向
                fd  = rot[:, 0]                  # finger dir
                w   = float(ci.attrs.get('gripper_width', 0.05))

                # 预抓取点 = 抓取中心沿 approach 反方向退出
                pre = gp - app * arr_scale

                # 箭头：从预抓取点 → 抓取中心（真实接近方向）
                ax.quiver(pre[0], pre[1], pre[2],
                          app[0] * arr_scale,
                          app[1] * arr_scale,
                          app[2] * arr_scale,
                          color='#00e676', linewidth=1.8,
                          arrow_length_ratio=0.3)

                # finger 接触点（红/蓝）及连线
                cL = gp + fd * w / 2
                cR = gp - fd * w / 2
                ax.scatter(*cL, c='#ff5252', s=50, depthshade=False, zorder=5)
                ax.scatter(*cR, c='#448aff', s=50, depthshade=False, zorder=5)
                ax.plot([cL[0], cR[0]], [cL[1], cR[1]], [cL[2], cR[2]],
                        c='#ffffff', linewidth=0.8, alpha=0.5)
                n_shown += 1

        title2 = (f'② {obj_id}   top-{n_shown} grasps   dataset={ds}\n'
                  f'坐标系: USD/Sim frame  |  绿箭头 = approach 接近方向  🔴🔵 = finger')
    else:
        title2 = f'② {obj_id}  |  mesh only  (no grasp HDF5)'

    ax.set_title(title2, color='white', fontsize=10, pad=8)

    # 刻度和背景
    ax.tick_params(colors='#555', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2a2a2a')
    ax.grid(False)
    ax.set_facecolor(BG2)
    # 隐藏默认轴标签（用 draw_world_axes 代替）
    ax.set_xlabel('', color='#333')
    ax.set_ylabel('', color='#333')
    ax.set_zlabel('', color='#333')
    ax.view_init(elev=elev, azim=azim)


# ── Panel 3: Sim settling 截图 ───────────────────────────────────────────────
def find_settle_snapshot(obj_id: str) -> str | None:
    candidates = sorted(glob.glob(
        os.path.join(SNAPSHOT_DIR, f'{obj_id}*settle*.png')))
    return candidates[-1] if candidates else None


def draw_panel3(ax, obj_id: str):
    snap = find_settle_snapshot(obj_id)
    if snap:
        ax.imshow(plt.imread(snap))
        ts = os.path.basename(snap).replace(f'{obj_id}_settle', '').replace('.png', '')
        ax.set_title(f'③ Sim 落定（100步后）\n物体实际放置姿态{ts}',
                     color='white', fontsize=10, pad=6)
    else:
        ax.text(0.5, 0.5,
                '尚无截图\n先运行:\n'
                'sim45 sim/run_grasp_sim.py\n'
                f'--hdf5 output/grasps_candidate/{obj_id}_grasp.hdf5',
                ha='center', va='center', color='#aaaaaa',
                fontsize=8, transform=ax.transAxes)
        ax.set_title('③ Sim 落定截图', color='white', fontsize=10, pad=6)
    ax.set_facecolor(BG2)
    ax.axis('off')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='3-panel pose sanity check (SAM3D original frame)')
    parser.add_argument('--obj',  required=True, help='物体 ID，如 A01026')
    parser.add_argument('--seq',  default=None,  help='指定 OakInk sequence（如 0001）')
    parser.add_argument('--elev', type=int, default=20,  help='3D 视角仰角 (default 20)')
    parser.add_argument('--azim', type=int, default=135, help='3D 视角方位角 (default 135)')
    parser.add_argument('--out',  default=os.path.join(PROJ, 'output', 'vis_pose_sanity'),
                        help='输出目录')
    args = parser.parse_args()

    # 布局：左1份  中2.2份（放大）  右1.2份
    fig = plt.figure(figsize=(28, 9), facecolor=BG)
    fig.suptitle(
        f'{args.obj}  —  Pose Sanity Check  '
        '(SAM3D original frame  |  no rotation.json)',
        color='white', fontsize=15, fontweight='bold', y=1.01)

    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.2, 1.2],
                          wspace=0.05, left=0.02, right=0.98,
                          top=0.93, bottom=0.05)

    # Panel 1 (2D image)
    ax0 = fig.add_subplot(gs[0])
    draw_panel1(ax0, args.obj, args.seq)

    # Panel 2 (3D mesh+grasps)  ← 最大
    ax1 = fig.add_subplot(gs[1], projection='3d')
    ax1.set_facecolor(BG2)
    draw_panel2(ax1, args.obj, elev=args.elev, azim=args.azim)

    # Panel 3 (Sim screenshot)
    ax2 = fig.add_subplot(gs[2])
    draw_panel3(ax2, args.obj)

    plt.tight_layout(pad=1.5)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{args.obj}_pose_sanity.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'→ {out_path}')


if __name__ == '__main__':
    main()
