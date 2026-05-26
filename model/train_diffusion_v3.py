#!/usr/bin/env python3
"""
train_diffusion_v3.py — 接触点对 Diffusion 训练 v3

核心改动 (vs v2):
  ① 预测目标: delta = (R-L)/2  (3D)，而非 (position, rotation_6d)
  ② 条件:     global_feat(512) + p_norm(3) + hp_at_p(1) + label_at_p(1) = 517D
  ③ GT 来源: 从 executed_panda_hand_at_close 几何计算 (L,R) 接触点对
  ④ Loss:    MSE(delta) + 几何约束 (width/perp/aff)
  ⑤ 采样:    affordance×human_prior 联合得分采样接触中点 p

用法:
  python3 -m model.train_diffusion_v3 \\
      --v6_ckpt     data_hub/ProcessedData/RobotPosterior/best_v6_model.pth \\
      --aff_h5      data_hub/ProcessedData/RobotPosterior/affordance_all.h5 \\
      --merged_dir  data_hub/ProcessedData/RobotPosterior/merged \\
      --split_json  data_hub/ProcessedData/RobotPosterior/min20/objects_train_val_split.json \\
      --save_dir    output/checkpoints_diffusion_v3 \\
      --epochs 500 --batch_size 64 --lr 3e-4
"""

import os, sys, glob, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import h5py
from model.grasp_diffusion_v3 import (
    GraspDiffusionV3, contact_geometry_loss,
    TCP_OFFSET, MIN_WIDTH, MAX_WIDTH
)
from model.logger import TrainingLogger

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
GEO_WARMUP  = 50      # 前 N epoch 不加几何 loss
LAMBDA_GEO  = 0.5     # 几何 loss 权重
N_POINTS    = 4096
POS_MAX_ABS = 0.5     # 单位错误过滤阈值（米）


# ─────────────────────────────────────────────────────────────
# V6 全局特征预缓存
# ─────────────────────────────────────────────────────────────

def cache_global_feats(v6_ckpt, aff_h5, device, logger):
    from model.pointnet2_v6 import PointNet2AffordanceV6

    logger.section("Caching V6 global_feat")
    pn2 = PointNet2AffordanceV6().to(device)
    ckpt = torch.load(v6_ckpt, map_location=device, weights_only=False)
    pn2.load_state_dict(ckpt['model_state_dict'])
    pn2.eval()
    for p in pn2.parameters():
        p.requires_grad_(False)

    cache = {}
    with h5py.File(aff_h5, 'r') as hf:
        obj_ids = [x.decode() if isinstance(x, bytes) else x
                   for x in hf['data']['obj_ids'][:]]
        pts_all = hf['data']['points'][:]
        nrm_all = hf['data']['normals'][:]

    with torch.no_grad():
        for i, oid in enumerate(obj_ids):
            pts = torch.from_numpy(pts_all[i]).unsqueeze(0).to(device)
            nrm = torch.from_numpy(nrm_all[i]).unsqueeze(0).to(device)
            feat = pn2.extract_global_feat(pts, torch.cat([pts, nrm], -1))
            cache[oid] = feat.squeeze(0).cpu()
            if (i + 1) % 20 == 0:
                logger.info(f"  cached {i+1}/{len(obj_ids)}")

    logger.info(f"  Done: {len(cache)} objects")
    return cache


# ─────────────────────────────────────────────────────────────
# 几何计算：从 executed pose 推 (L, R) 接触点对
# ─────────────────────────────────────────────────────────────

def compute_contact_pair(ep_grp, prerot_mat, width):
    """
    从 executed_panda_hand_at_close 计算接触点对。
    返回 (L, R) in object local frame，若无效返回 None。

    ep_grp:    h5py group  executed_panda_hand_at_close
    prerot_mat: (3,3) mesh 预旋转矩阵（或 eye(3)）
    width:     float  夹爪宽度（米）
    """
    pos    = ep_grp['position'][:].astype(np.float64)    # wrist, world frame
    ap     = ep_grp['approach_dir'][:].astype(np.float64)
    fd     = ep_grp['finger_dir'][:].astype(np.float64)

    # 转到物体局部坐标系（undo prerotation）
    Rp     = prerot_mat.astype(np.float64)
    pos_o  = Rp.T @ pos
    ap_o   = Rp.T @ (ap / (np.linalg.norm(ap) + 1e-8))
    fd_o   = Rp.T @ (fd / (np.linalg.norm(fd) + 1e-8))

    # 过滤单位错误（mm vs m）
    if np.abs(pos_o).max() > POS_MAX_ABS:
        return None

    # 指尖中点（表面上）= wrist + approach * TCP_OFFSET
    finger_mid = pos_o + ap_o * TCP_OFFSET
    hw = width / 2.0
    L  = finger_mid - fd_o * hw
    R  = finger_mid + fd_o * hw

    # 检查宽度合理性
    if hw < MIN_WIDTH / 2 or hw > MAX_WIDTH:
        return None

    return L.astype(np.float32), R.astype(np.float32), finger_mid.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class ContactPairDataset(Dataset):
    """
    每条样本：
      global_feat (512,)
      p_norm      (3,)   归一化指尖中点
      hp_at_p     (1,)   human_prior value at p
      label_at_p  (1,)   affordance label at p
      delta_norm  (3,)   归一化 half-vector (target)
      pc_xyz      (N,3)  归一化点云（用于 geo loss）
      pc_normals  (N,3)
      aff_labels  (N,)
      pc_centroid (3,)
      pc_radius   (1,)
    """

    def __init__(self, obj_ids, aff_h5, merged_dir, feat_cache, logger=None):
        self.samples = []
        skipped_obj  = 0
        skipped_pose = 0

        with h5py.File(aff_h5, 'r') as hf:
            raw_ids = [x.decode() if isinstance(x, bytes) else x
                       for x in hf['data']['obj_ids'][:]]
            aff_idx = {o: i for i, o in enumerate(raw_ids)}
            all_pts = hf['data']['points'][:]
            all_nrm = hf['data']['normals'][:]
            all_lab = hf['data']['labels'][:]
            all_hp  = hf['data']['human_priors'][:]

        for oid in obj_ids:
            if oid not in aff_idx or oid not in feat_cache:
                skipped_obj += 1
                continue

            idx  = aff_idx[oid]
            pc   = all_pts[idx].astype(np.float32)
            nrm  = all_nrm[idx].astype(np.float32)
            lab  = all_lab[idx].astype(np.float32)
            hp   = all_hp[idx].astype(np.float32)
            gf   = feat_cache[oid]

            # 归一化
            centroid = pc.mean(0)
            radius   = float(np.linalg.norm(pc - centroid, axis=1).max()) + 1e-6
            pc_norm  = (pc - centroid) / radius

            # KDTree for hp/label lookup
            tree = cKDTree(pc_norm)

            mp = os.path.join(merged_dir, f'{oid}_robot_gt_merged.hdf5')
            if not os.path.exists(mp):
                skipped_obj += 1
                continue

            n_obj = 0
            with h5py.File(mp, 'r') as hf2:
                sg = hf2.get('successful_grasps', {})
                for gk in sg.keys():
                    g = sg[gk]
                    ep = g.get('executed_panda_hand_at_close')
                    if ep is None:
                        skipped_pose += 1
                        continue
                    pre = g.get('mesh_prerotation')
                    Rp  = pre['matrix'][:].astype(np.float64) \
                          if (pre and 'matrix' in pre) else np.eye(3)
                    width = float(g.attrs.get(
                        'finger_width_actual',
                        g.attrs.get('gripper_width', 0.04)))
                    if width <= 0:
                        skipped_pose += 1
                        continue

                    result = compute_contact_pair(ep, Rp, width)
                    if result is None:
                        skipped_pose += 1
                        continue

                    L, R, fmid = result
                    delta = 0.5 * (R - L)           # in metric object frame

                    # 归一化到点云尺度
                    p_norm_val  = (fmid - centroid) / radius
                    delta_norm  = delta / radius      # dimensionless

                    # 查 human_prior 和 label at p
                    _, nn_i    = tree.query(p_norm_val)
                    hp_val     = float(hp[nn_i])
                    label_val  = float(lab[nn_i])

                    self.samples.append({
                        'global_feat': gf,                # (512,)
                        'p_norm':      p_norm_val.astype(np.float32),  # (3,)
                        'hp_at_p':     np.float32(hp_val),
                        'label_at_p':  np.float32(label_val),
                        'delta_norm':  delta_norm.astype(np.float32),  # (3,)
                        'pc_xyz':      pc_norm,            # (N,3)
                        'pc_normals':  nrm,                # (N,3)
                        'aff_labels':  lab,                # (N,)
                        'pc_centroid': centroid.astype(np.float32),
                        'pc_radius':   np.float32(radius),
                    })
                    n_obj += 1

            if logger and n_obj == 0:
                logger.info(f'  ⚠  {oid}: 0 valid grasps')

        if logger:
            logger.info(f'  Dataset: {len(self.samples)} samples '
                        f'(skip_obj={skipped_obj} skip_pose={skipped_pose})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['global_feat'],
            torch.from_numpy(s['p_norm']),
            torch.tensor([s['hp_at_p'], s['label_at_p']]),  # (2,)
            torch.from_numpy(s['delta_norm']),
            torch.from_numpy(s['pc_xyz']),
            torch.from_numpy(s['pc_normals']),
            torch.from_numpy(s['aff_labels']),
            torch.from_numpy(s['pc_centroid']),
            torch.tensor(s['pc_radius']).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────────
# 统计 delta 归一化参数
# ─────────────────────────────────────────────────────────────

def compute_delta_stats(loader, device):
    print("  Computing delta statistics...")
    all_d = []
    for batch in loader:
        delta = batch[3].to(device)
        all_d.append(delta.cpu())
    all_d = torch.cat(all_d, 0)
    mean = all_d.mean(0)
    std  = all_d.std(0).clamp(min=1e-5)
    print(f"  delta_mean: {mean.numpy().round(4)}")
    print(f"  delta_std : {std.numpy().round(4)}")
    return mean.to(device), std.to(device)


# ─────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = TrainingLogger(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    logger.section("Grasp Contact-Pair Diffusion Training v3")
    logger.info(f"  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU:    {torch.cuda.get_device_name(0)}")

    # ── 1. 预缓存 V6 特征 ──────────────────────────────────────
    feat_cache = cache_global_feats(args.v6_ckpt, args.aff_h5, device, logger)

    # ── 2. 数据集 ──────────────────────────────────────────────
    split = json.load(open(args.split_json))
    train_ids = split['train']
    val_ids   = split['val']

    logger.section("Building datasets")
    train_ds = ContactPairDataset(train_ids, args.aff_h5,
                                  args.merged_dir, feat_cache, logger)
    val_ds   = ContactPairDataset(val_ids,   args.aff_h5,
                                  args.merged_dir, feat_cache, logger)

    logger.section("Dataset Statistics")
    logger.info(f"  Train: {len(train_ds)} samples")
    logger.info(f"  Val:   {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── 3. delta 归一化统计 ────────────────────────────────────
    delta_mean, delta_std = compute_delta_stats(train_loader, device)
    torch.save({'delta_mean': delta_mean, 'delta_std': delta_std},
               os.path.join(args.save_dir, 'delta_stats.pt'))

    # ── 4. 模型 ────────────────────────────────────────────────
    # cond_dim = global_feat(512) + p_norm(3) + hp_at_p(1) + label_at_p(1) = 517
    COND_DIM = 517
    diff = GraspDiffusionV3(T=args.T, hidden=args.hidden,
                             cond_dim=COND_DIM).to(device)
    optimizer = torch.optim.AdamW(diff.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    logger.section("Training Start")
    logger.info(f"  {'Ep':>4} | {'Train':>10} | {'Val':>10} | {'GeoLoss':>9} | LR")
    logger.info("  " + "-" * 55)

    best_val  = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────
        diff.train()
        t_losses, g_losses = [], []

        for batch in train_loader:
            gf, p_norm, hp_label, delta, pc_xyz, pc_nrm, aff_lab, ctr, rad = \
                [x.to(device) for x in batch]

            # 构建条件向量
            cond = torch.cat([gf, p_norm, hp_label], dim=-1)   # (B, 517)

            # 归一化 delta
            delta_n = (delta - delta_mean) / delta_std

            diff_loss, x0_pred = diff.training_loss(delta_n, cond, return_x0=True)

            if epoch > GEO_WARMUP and x0_pred is not None:
                # 反归一化回原始空间
                delta_pred = x0_pred.detach() * delta_std + delta_mean
                # pc_xyz 已归一化，centroid=0, radius=1 → 传入归一化形式
                pc_centroid_zero = torch.zeros_like(ctr)
                pc_radius_one    = torch.ones_like(rad)
                l_geo = contact_geometry_loss(
                    delta_pred, p_norm, pc_xyz, pc_nrm,
                    pc_centroid_zero, rad, aff_lab)
                loss = diff_loss + LAMBDA_GEO * l_geo
                g_losses.append(l_geo.item())
            else:
                loss = diff_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            optimizer.step()
            t_losses.append(diff_loss.item())

        scheduler.step()

        # ── Val ────────────────────────────────────────────────
        diff.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                gf, p_norm, hp_label, delta, _, _, _, _, _ = \
                    [x.to(device) for x in batch]
                cond    = torch.cat([gf, p_norm, hp_label], dim=-1)
                delta_n = (delta - delta_mean) / delta_std
                d_loss, _ = diff.training_loss(delta_n, cond)
                v_losses.append(d_loss.item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        g_loss = float(np.mean(g_losses)) if g_losses else 0.0
        lr_now = scheduler.get_last_lr()[0]
        dt     = int(time.time() - t0)

        is_best = v_loss < best_val
        if is_best:
            best_val   = v_loss
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'model_state_dict': diff.state_dict(),
                        'val_loss': best_val},
                       os.path.join(args.save_dir, 'best_model.pth'))

        if epoch % 10 == 0 or epoch == 1 or is_best:
            star = ' ★' if is_best else ''
            logger.info(
                f"  Ep {epoch:>4}/{args.epochs} | "
                f"train={t_loss:.6f} | val={v_loss:.6f}  "
                f"geo={g_loss:.5f} | lr={lr_now:.2e} | "
                f"{dt//60}m{dt%60:02d}s{star}")

    logger.section("Training Complete")
    logger.info(f"  Best epoch  : {best_epoch}")
    logger.info(f"  Best val    : {best_val:.6f}")
    logger.info(f"  Logs saved  : {args.save_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--v6_ckpt',    required=True)
    p.add_argument('--aff_h5',     required=True)
    p.add_argument('--merged_dir', required=True)
    p.add_argument('--split_json', required=True)
    p.add_argument('--save_dir',   default='output/checkpoints_diffusion_v3')
    p.add_argument('--epochs',     type=int,   default=500)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--hidden',     type=int,   default=512)
    p.add_argument('--T',          type=int,   default=1000)
    main(p.parse_args())
