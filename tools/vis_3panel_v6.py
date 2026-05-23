#!/usr/bin/env python3
"""
3-Panel 可视化 v6: Human Prior | Robot Posterior | Model Prediction

每个物体输出一张 3 列拼合图:
  左: Human Prior (连续 0~1, jet colormap) — 来自 ProcessedData/training_fp/oakink/
  中: Robot Posterior (连续 0~1, jet colormap) — 来自 training_m5/merged/
  右: Model Prediction (连续 0~1, jet colormap) — 来自 v6 模型

用法:
    # 单个物体
    python3 tools/vis_3panel_v6.py --obj A01001
    # 批量 OakInk
    python3 tools/vis_3panel_v6.py --batch --out output/vis_3panel_v6
"""
import os, sys, argparse, glob
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pointnet2_v6 import PointNet2AffordanceV6

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
HP_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'oakink')
RP_DIR = os.path.join(PROJ, 'data_hub', 'training_m5', 'merged')
CKPT = os.path.join(PROJ, 'output', 'checkpoints_v6_fix12', 'best_v6_model.pth')
N_POINTS = 4096


def load_human_prior(obj_id):
    """从 ProcessedData/training_fp/oakink/ 加载 Human Prior."""
    path = os.path.join(HP_DIR, f'{obj_id}.hdf5')
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as f:
        return {
            'pc': f['point_cloud'][()].astype(np.float32),
            'normals': f['normals'][()].astype(np.float32),
            'human_prior': f['human_prior'][()].astype(np.float32),
        }


def load_robot_posterior(obj_id):
    """从 affordance_*_soft.h5 加载 Robot Posterior (soft_labels).
    
    Soft labels are stored in the consolidated H5 files, indexed by obj_id.
    Falls back to training_fp's robot_gt if not found.
    """
    # Try affordance_*_soft.h5 files first
    for h5_name in ['affordance_train_soft.h5', 'affordance_val_soft.h5']:
        h5_path = os.path.join(PROJ, 'data_hub', 'training_m5', h5_name)
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, 'r') as f:
            obj_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f['data/obj_ids'][:]]
            if obj_id in obj_ids:
                idx = obj_ids.index(obj_id)
                return {
                    'pc': f['data/points'][idx].astype(np.float32),
                    'normals': f['data/normals'][idx].astype(np.float32),
                    'posterior': f['data/soft_labels'][idx].astype(np.float32),
                }

    # Fallback: use robot_gt from training_fp
    hp_path = os.path.join(HP_DIR, f'{obj_id}.hdf5')
    if os.path.exists(hp_path):
        with h5py.File(hp_path, 'r') as f:
            if 'robot_gt' in f:
                return {
                    'pc': f['point_cloud'][()].astype(np.float32),
                    'normals': f['normals'][()].astype(np.float32),
                    'posterior': f['robot_gt'][()].astype(np.float32),
                }
    return None


def predict_v6(model, pc, normals, device):
    """v6 模型推理 (xyz + normals, 6ch, no HP)."""
    features = np.concatenate([pc, normals], axis=-1)  # (N, 6)
    pts_t = torch.from_numpy(pc).unsqueeze(0).to(device)
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t, feat_t).squeeze(0).cpu().numpy()
    return pred


def render_pointcloud(ax, points, values, title, cmap_name='jet', vmin=0, vmax=1,
                      elev=25, azim=135):
    """在 matplotlib 3D subplot 上渲染连续热图点云."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.clip(values, vmin, vmax))

    # 排序: 低值先画, 高值后画 (高值覆盖)
    order = np.argsort(values)
    points = points[order]
    colors = colors[order]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=1.5, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)

    # 等比例
    extents = points.max(axis=0) - points.min(axis=0)
    max_ext = extents.max() * 0.6
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    ax.set_xlim(center[0] - max_ext, center[0] + max_ext)
    ax.set_ylim(center[1] - max_ext, center[1] + max_ext)
    ax.set_zlim(center[2] - max_ext, center[2] + max_ext)
    ax.set_axis_off()


def densify_values(pc_sparse, values_sparse, pc_dense):
    """KNN 插值: 从稀疏点云到密集点云."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pc_sparse)
    _, idx = tree.query(pc_dense, k=3)
    dists = np.linalg.norm(pc_dense[:, None, :] - pc_sparse[idx], axis=2)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(values_sparse[idx] * weights, axis=1)


def vis_single(obj_id, model, device, out_dir=None, show=False):
    """处理并可视化单个物体."""
    hp_data = load_human_prior(obj_id)
    rp_data = load_robot_posterior(obj_id)

    if hp_data is None and rp_data is None:
        print(f"  ⚠️ {obj_id}: 无 HP 也无 RP 数据, 跳过")
        return False

    # Use the point cloud from whichever source is available
    # Prefer RP since it has soft_labels (our training target)
    if rp_data is not None:
        pc = rp_data['pc']
        normals = rp_data['normals']
        posterior = rp_data['posterior']
    else:
        pc = hp_data['pc']
        normals = hp_data['normals']
        posterior = None

    hp = hp_data['human_prior'] if hp_data is not None else None

    # Model prediction
    pred = predict_v6(model, pc, normals, device)

    # Optional: densify using mesh
    import trimesh
    mesh_path = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        N_VIS = 20000
        vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
        vis_pc = vis_pc.astype(np.float32)

        pred_dense = densify_values(pc, pred, vis_pc)

        if posterior is not None:
            rp_dense = densify_values(pc, posterior, vis_pc)
        else:
            rp_dense = np.zeros(len(vis_pc))

        if hp is not None:
            hp_pc = hp_data['pc']
            hp_dense = densify_values(hp_pc, hp, vis_pc)
        else:
            hp_dense = np.zeros(len(vis_pc))
    else:
        vis_pc = pc
        pred_dense = pred
        rp_dense = posterior if posterior is not None else np.zeros(len(pc))
        hp_dense = hp if hp is not None else np.zeros(len(pc))

    # Compute Pearson between pred and posterior
    pearson_str = ""
    if posterior is not None:
        p_m = pred - pred.mean()
        g_m = posterior - posterior.mean()
        r = np.sum(p_m * g_m) / (np.sqrt(np.sum(p_m**2) * np.sum(g_m**2)) + 1e-8)
        pearson_str = f"  r={r:.3f}"

    # Plot
    fig = plt.figure(figsize=(20, 7), facecolor='white')

    hp_title = f'Human Prior\n(max={hp_dense.max():.2f})' if hp is not None else 'Human Prior (N/A)'
    rp_title = f'Robot Posterior\n(max={rp_dense.max():.2f})' if posterior is not None else 'Robot Posterior (N/A)'
    pred_title = f'Model Prediction\n(max={pred_dense.max():.2f}{pearson_str})'

    ax1 = fig.add_subplot(131, projection='3d')
    render_pointcloud(ax1, vis_pc, hp_dense, hp_title)

    ax2 = fig.add_subplot(132, projection='3d')
    render_pointcloud(ax2, vis_pc, rp_dense, rp_title)

    ax3 = fig.add_subplot(133, projection='3d')
    render_pointcloud(ax3, vis_pc, pred_dense, pred_title)

    fig.suptitle(f'{obj_id}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{obj_id}_3panel.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {obj_id}: saved → {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description='3-Panel v6 可视化: HP | RP | Pred')
    parser.add_argument('--obj', type=str, help='单个物体 ID')
    parser.add_argument('--batch', action='store_true', help='批量处理所有 OakInk 物体')
    parser.add_argument('--out', type=str,
                        default=os.path.join(PROJ, 'output', 'vis_3panel_v6'),
                        help='输出目录')
    parser.add_argument('--ckpt', type=str, default=CKPT, help='模型 checkpoint 路径')
    parser.add_argument('--show', action='store_true', help='交互显示')
    args = parser.parse_args()

    # Load v6 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2AffordanceV6(in_channel=6).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ v6 模型加载: epoch={ckpt['epoch']}, Pearson={ckpt.get('val_pearson', 'N/A')}")

    if args.obj:
        vis_single(args.obj, model, device, out_dir=args.out, show=args.show)
    elif args.batch:
        # Find all OakInk objects that have HP data
        hp_files = sorted(glob.glob(os.path.join(HP_DIR, '*.hdf5')))
        total = 0
        success = 0
        for f in hp_files:
            obj_id = os.path.splitext(os.path.basename(f))[0]
            # Skip non-OakInk
            if obj_id.startswith('ycb_'):
                continue
            total += 1
            if vis_single(obj_id, model, device, out_dir=args.out):
                success += 1
        print(f"\n{'='*50}")
        print(f"  完成! {success}/{total} 个 OakInk 物体可视化")
        print(f"  输出: {args.out}")
        print(f"{'='*50}")
    else:
        print("请指定 --obj 或 --batch")


if __name__ == '__main__':
    main()
