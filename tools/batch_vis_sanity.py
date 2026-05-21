#!/usr/bin/env python3
"""
batch_vis_sanity.py — 批量生成所有 OakInk 物体的 2-panel 姿态可视化

Panel 1: OakInk 原始视频帧（物体在数据集里的真实位姿）
Panel 2: USD mesh + 抓取候选（与 Sim 坐标系一致）

用法:
    python3 tools/batch_vis_sanity.py
    python3 tools/batch_vis_sanity.py --dataset oakink --out output/vis_batch
    python3 tools/batch_vis_sanity.py --obj A01026  # 单个物体
"""
import os, sys, argparse, glob, json, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

# CJK 字体
for _p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
           '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
        matplotlib.rcParams['font.family'] = _fm.FontProperties(fname=_p).get_name()
        break

BG  = '#111111'
BG2 = '#1a1a1a'

MESH_DIR     = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
USD_ROOT     = os.path.join(PROJ, 'output', 'obj_usd')
GRASP_DIR    = os.path.join(PROJ, 'output', 'grasps_candidate')
OAKINK_DIR   = os.path.expanduser('~/Project/OakInk')


def load_usd_mesh(obj_id):
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
                    v  = np.array(pts, dtype=np.float32)
                    ff = np.array(idx, dtype=np.int32).reshape(-1, 3)
                    all_v.append(v)
                    all_f.append(ff + offset)
                    offset += len(v)
                if not all_v:
                    return None
                import trimesh
                verts = np.concatenate(all_v).astype(np.float64)
                faces = np.concatenate(all_f)

                # ── 应用 R_align（与 grasp_sampler 完全一致）─────────────────
                meta_path = usd_path.replace('.usd', '_meta.json')
                if os.path.exists(meta_path):
                    meta = json.load(open(meta_path))
                    if 'R_align_matrix' in meta:
                        R = np.array(meta['R_align_matrix'], dtype=np.float64)
                        verts = (R @ verts.T).T

                return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return None


def find_oakink_frame(obj_id):
    for cam in ['north_east', 'north_west', 'south_east']:
        pat = os.path.join(OAKINK_DIR, 'image', 'stream_release_v2',
                           f'{obj_id}_*', '*', f'{cam}_color_*.png')
        files = sorted(glob.glob(pat))
        if files:
            return files[len(files) // 2]
    return None


def draw_world_axes(ax, origin, axis_len):
    ox, oy, oz = origin
    ax.quiver(ox, oy, oz, axis_len, 0, 0, color='#ff4444', linewidth=2.0, arrow_length_ratio=0.15)
    ax.text(ox + axis_len * 1.12, oy, oz, 'X', color='#ff4444', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.quiver(ox, oy, oz, 0, axis_len, 0, color='#44ff44', linewidth=2.0, arrow_length_ratio=0.15)
    ax.text(ox, oy + axis_len * 1.12, oz, 'Y', color='#44ff44', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.quiver(ox, oy, oz, 0, 0, axis_len, color='#4488ff', linewidth=2.5, arrow_length_ratio=0.15)
    ax.text(ox, oy, oz + axis_len * 1.18, 'Z\n(up)', color='#4488ff', fontsize=11, fontweight='bold', ha='center', va='bottom')


def make_panel_figure(obj_id, elev=20, azim=135):
    import trimesh, h5py

    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    fig.suptitle(f'{obj_id}  —  Pose Sanity  (USD/Sim frame | no rotation.json)',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.05,
                          left=0.02, right=0.98, top=0.93, bottom=0.05)

    # ── Panel 1: OakInk 原始帧 ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    frame = find_oakink_frame(obj_id)
    if frame:
        ax0.imshow(plt.imread(frame))
        seq = os.path.basename(os.path.dirname(os.path.dirname(frame)))
        ax0.set_title(f'① OakInk 原始帧\n{seq}', color='white', fontsize=9, pad=5)
    else:
        ax0.text(0.5, 0.5, 'frame not found', ha='center', va='center',
                 color='#ff5252', fontsize=9, transform=ax0.transAxes)
        ax0.set_title('① OakInk 原始帧', color='white', fontsize=9, pad=5)
    ax0.set_facecolor(BG2)
    ax0.axis('off')

    # ── Panel 2: USD mesh + 抓取候选 ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], projection='3d')
    ax1.set_facecolor(BG2)

    mesh = load_usd_mesh(obj_id)
    if mesh is None:
        ax1.text2D(0.5, 0.5, 'USD not found', ha='center', va='center',
                   color='red', transform=ax1.transAxes)
        ax1.set_title(f'② {obj_id} — USD missing', color='white', fontsize=9)
        plt.tight_layout(pad=1.0)
        return fig

    ext_max = float(mesh.bounding_box.extents.max())
    bounds  = mesh.bounds
    center  = mesh.centroid

    # 点云
    pts, _ = trimesh.sample.sample_surface(mesh, 8000)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                c='#888888', s=0.6, alpha=0.5, rasterized=True)

    # 世界坐标轴
    axis_len = ext_max * 0.50
    ax_origin = np.array([center[0], center[1], bounds[0, 2] - axis_len * 0.15])
    draw_world_axes(ax1, ax_origin, axis_len)

    # 轴范围
    pad = axis_len * 1.4
    ax1.set_xlim(center[0] - pad, center[0] + pad)
    ax1.set_ylim(center[1] - pad, center[1] + pad)
    ax1.set_zlim(bounds[0, 2] - axis_len * 0.3, bounds[1, 2] + axis_len * 0.5)

    # 抓取候选
    grasp_hdf5 = os.path.join(GRASP_DIR, f'{obj_id}_grasp.hdf5')
    n_shown = 0
    if os.path.exists(grasp_hdf5):
        arr_scale = ext_max * 0.20
        with h5py.File(grasp_hdf5, 'r') as f:
            cg = f.get('candidates', {})
            for key in list(cg.keys())[:10]:
                ci = cg[key]
                gp  = ci['position'][:]
                rot = ci['rotation'][:]
                app = rot[:, 2]
                fd  = rot[:, 0]
                w   = float(ci.attrs.get('gripper_width', 0.05))
                pre = gp - app * arr_scale
                ax1.quiver(pre[0], pre[1], pre[2],
                           app[0] * arr_scale, app[1] * arr_scale, app[2] * arr_scale,
                           color='#00e676', linewidth=1.5, arrow_length_ratio=0.3)
                cL = gp + fd * w / 2
                cR = gp - fd * w / 2
                ax1.scatter(*cL, c='#ff5252', s=40, depthshade=False, zorder=5)
                ax1.scatter(*cR, c='#448aff', s=40, depthshade=False, zorder=5)
                ax1.plot([cL[0], cR[0]], [cL[1], cR[1]], [cL[2], cR[2]],
                         c='#ffffff', linewidth=0.7, alpha=0.4)
                n_shown += 1
        title2 = f'② {obj_id}   {n_shown} grasps (USD/Sim frame)\n🟢 approach  🔴🔵 finger'
    else:
        title2 = f'② {obj_id}   no grasp HDF5 yet'

    ax1.set_title(title2, color='white', fontsize=9, pad=6)
    ax1.tick_params(colors='#444', labelsize=6)
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2a2a2a')
    ax1.grid(False)
    ax1.set_xlabel(''); ax1.set_ylabel(''); ax1.set_zlabel('')
    ax1.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=1.0)
    return fig


def get_all_oakink_ids():
    usd_dir = os.path.join(USD_ROOT, 'oakink')
    if not os.path.exists(usd_dir):
        return []
    return sorted(f.replace('.usd', '') for f in os.listdir(usd_dir)
                  if f.endswith('.usd') and '_meta' not in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',     default=None, help='单个物体 ID')
    parser.add_argument('--dataset', default='oakink')
    parser.add_argument('--elev',    type=int, default=20)
    parser.add_argument('--azim',    type=int, default=135)
    parser.add_argument('--out',     default=os.path.join(PROJ, 'output', 'vis_batch'))
    parser.add_argument('--only-with-grasp', action='store_true',
                        help='只处理已有 grasp HDF5 的物体')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.obj:
        obj_ids = [args.obj]
    else:
        obj_ids = get_all_oakink_ids()

    if args.only_with_grasp:
        obj_ids = [o for o in obj_ids
                   if os.path.exists(os.path.join(GRASP_DIR, f'{o}_grasp.hdf5'))]

    print(f'共 {len(obj_ids)} 个物体，输出到 {args.out}')

    done, failed = 0, []
    for i, obj_id in enumerate(obj_ids):
        out_path = os.path.join(args.out, f'{obj_id}_sanity.png')
        try:
            fig = make_panel_figure(obj_id, elev=args.elev, azim=args.azim)
            fig.savefig(out_path, dpi=120, bbox_inches='tight',
                        facecolor=BG, edgecolor='none')
            plt.close(fig)
            done += 1
            print(f'  [{i+1}/{len(obj_ids)}] ✅ {obj_id} → {os.path.basename(out_path)}')
        except Exception as e:
            failed.append(obj_id)
            print(f'  [{i+1}/{len(obj_ids)}] ❌ {obj_id}: {e}')
            traceback.print_exc()

    print(f'\n完成: {done} 成功  {len(failed)} 失败')
    if failed:
        print(f'失败: {failed}')


if __name__ == '__main__':
    main()
