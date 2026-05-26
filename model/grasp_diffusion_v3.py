#!/usr/bin/env python3
"""
grasp_diffusion_v3.py — 接触点对 Diffusion

核心思想:
  不直接预测 (position, rotation), 而是预测接触点对的半向量 delta:
    delta = (R - L) / 2    (3D, 物体归一化坐标系)
    L = p - delta,  R = p + delta   (p = 指尖中点, 由 affordance 采样)

从 (p, delta) 推导完整抓取 Pose:
  finger_dir   = normalize(delta)
  width        = 2 * ||delta|| * pc_radius
  finger_mid   = p * pc_radius + pc_centroid  (反归一化)
  approach_dir = local_normal(finger_mid)     (从点云 KNN 查)
  TCP position = finger_mid - approach_dir * TCP_OFFSET  (wrist 在外侧)
  R_mat[:,0] = finger_dir
  R_mat[:,1] = cross(approach_dir, finger_dir)
  R_mat[:,2] = approach_dir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TCP_OFFSET   = 0.105   # m，手腕到指尖中点的距离
GRIPPER_HW   = 0.030   # m，夹爪半宽名义值（用于 loss scaling）
MIN_WIDTH    = 0.010   # m
MAX_WIDTH    = 0.085   # m


# ─────────────────────────────────────────────────────────────
# 几何辅助函数
# ─────────────────────────────────────────────────────────────

def local_normal_at_point(p, pc, normals, k=5):
    """
    在点云 pc 中找 p 最近的 k 个点，返回法向加权平均。
    p:       (B, 3) or (3,) numpy
    pc:      (N, 3) numpy
    normals: (N, 3) numpy
    """
    d = np.linalg.norm(pc - p[None], axis=-1)   # (N,)
    idx = np.argsort(d)[:k]
    weights = 1.0 / (d[idx] + 1e-8)
    weights /= weights.sum()
    n = (normals[idx] * weights[:, None]).sum(0)
    return n / (np.linalg.norm(n) + 1e-8)


def derive_grasp_from_contact_pair(L, R, pc, normals):
    """
    从接触点对推导完整抓取 Pose (numpy, 物体坐标系).

    Returns dict:
      finger_mid  (3,)  指尖中点（在物体表面或附近）
      finger_dir  (3,)  手指张开方向
      approach    (3,)  接近方向（从外向内，朝向物体）
      wrist_pos   (3,)  panda_hand 位置（wrist）
      R_mat       (3,3) 旋转矩阵  cols = [finger_dir, up_dir, approach_dir]
      width_m     float  夹爪宽度（米）
    """
    finger_mid = 0.5 * (L + R)
    half_vec   = 0.5 * (R - L)
    width      = 2.0 * np.linalg.norm(half_vec)
    finger_dir = half_vec / (np.linalg.norm(half_vec) + 1e-8)

    # approach 从点云法向估计
    approach = local_normal_at_point(finger_mid, pc, normals, k=5)
    # 确保 approach 指向外（从物体表面出去）
    # 物体质心
    centroid = pc.mean(0)
    if np.dot(approach, finger_mid - centroid) < 0:
        approach = -approach

    # up_dir = approach × finger_dir（右手系）
    up_dir = np.cross(approach, finger_dir)
    up_dir /= np.linalg.norm(up_dir) + 1e-8

    # 重新正交化 finger_dir
    finger_dir = np.cross(up_dir, approach)
    finger_dir /= np.linalg.norm(finger_dir) + 1e-8

    # wrist = finger_mid - approach * TCP_OFFSET
    wrist_pos = finger_mid - approach * TCP_OFFSET

    R_mat = np.stack([finger_dir, up_dir, approach], axis=-1)   # (3,3)

    return {
        'finger_mid': finger_mid,
        'finger_dir': finger_dir,
        'approach':   approach,
        'wrist_pos':  wrist_pos,
        'R_mat':      R_mat,
        'width_m':    width,
    }


# ─────────────────────────────────────────────────────────────
# MLP Denoiser（去噪网络）
# ─────────────────────────────────────────────────────────────

class DeltaDenoiser(nn.Module):
    """
    输入:  [x_t(3), t_emb(128), cond(cond_dim)]
    输出:  predicted x_0 (3D delta, 归一化坐标系)
    """

    def __init__(self, cond_dim: int, hidden: int = 512):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 128),
        )
        in_dim = 3 + 128 + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x_t, t, cond):
        """
        x_t:  (B, 3)
        t:    (B,)  int timestep
        cond: (B, cond_dim)
        """
        t_f  = t.float().unsqueeze(-1) / 1000.0
        te   = self.time_emb(t_f)
        inp  = torch.cat([x_t, te, cond], dim=-1)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────
# Diffusion Model (DDPM, 3D delta)
# ─────────────────────────────────────────────────────────────

class GraspDiffusionV3(nn.Module):
    """
    预测 delta = (R - L) / 2 in normalized object frame.
    条件 cond = [global_feat(512), p_norm(3), local_hp(1), local_label(1)] = 517D
    """

    COND_DIM = 517   # 512 + 3 + 1 + 1

    def __init__(self, T: int = 1000, hidden: int = 512,
                 cond_dim: int = None):
        super().__init__()
        cd = cond_dim or self.COND_DIM
        self.T       = T
        self.denoiser = DeltaDenoiser(cond_dim=cd, hidden=hidden)

        # DDPM cosine schedule
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / T + s) / (1 + s)) * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)

        alphas     = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',     betas.float())
        self.register_buffer('alphas',    alphas.float())
        self.register_buffer('alpha_cumprod', alpha_cumprod.float())
        self.register_buffer('sqrt_alpha_cumprod',
                             torch.sqrt(alpha_cumprod).float())
        self.register_buffer('sqrt_one_minus_alpha_cumprod',
                             torch.sqrt(1 - alpha_cumprod).float())

    # ── 训练 ──────────────────────────────────────────────────

    def training_loss(self, x0, cond, return_x0=False):
        """
        x0:   (B, 3)  归一化 delta GT
        cond: (B, COND_DIM)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
        sqrt_oab = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_oab * noise

        x0_pred = self.denoiser(x_t, t, cond)
        loss = F.mse_loss(x0_pred, x0)

        if return_x0:
            return loss, x0_pred
        return loss, None

    # ── 推理 (DDIM) ───────────────────────────────────────────

    @torch.no_grad()
    def sample(self, cond, n_samples: int = 1, ddim_steps: int = 50):
        """
        cond: (1, COND_DIM) → 扩展为 (n_samples, COND_DIM)
        返回: (n_samples, 3) 预测的 delta（归一化坐标系）
        """
        cond = cond.expand(n_samples, -1)
        x = torch.randn(n_samples, 3, device=cond.device)

        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        for i, t_val in enumerate(timesteps):
            t = torch.full((n_samples,), t_val, dtype=torch.long,
                           device=cond.device)
            x0_pred = self.denoiser(x, t, cond)

            if t_val > 0:
                t_prev = t_val - step_size
                ab_t    = self.alpha_cumprod[t_val]
                ab_prev = self.alpha_cumprod[max(t_prev, 0)]
                sigma   = torch.sqrt((1 - ab_prev) / (1 - ab_t) *
                                     (1 - ab_t / ab_prev))
                x = (torch.sqrt(ab_prev) * x0_pred
                     + torch.sqrt(1 - ab_prev - sigma**2)
                       * (x - torch.sqrt(ab_t) * x0_pred)
                       / torch.sqrt(1 - ab_t)
                     + sigma * torch.randn_like(x))
            else:
                x = x0_pred

        return x   # (n_samples, 3)


# ─────────────────────────────────────────────────────────────
# 几何 Loss
# ─────────────────────────────────────────────────────────────

def contact_geometry_loss(delta_pred, p_norm, pc_xyz, pc_normals,
                          pc_centroid, pc_radius, aff_labels):
    """
    在预测的 delta 上施加几何约束：

    1. Width constraint: ||2*delta|| ∈ [MIN_WIDTH, MAX_WIDTH]
    2. Surface constraint: L 和 R 应在物体包围盒内（近表面）
    3. Approach alignment: delta ⊥ local_normal(p)   → finger 方向切于表面

    所有输入均已在 GPU 上。
    pc_xyz:     (B, N, 3) 已归一化
    pc_normals: (B, N, 3)
    p_norm:     (B, 3) 已归一化的指尖中点
    delta_pred: (B, 3) 预测
    """
    B = delta_pred.shape[0]
    losses = []

    # 1. Width constraint（在归一化空间）
    width_m   = 2.0 * torch.norm(delta_pred, dim=-1) * pc_radius.squeeze(-1)
    w_lo      = torch.clamp(MIN_WIDTH - width_m, min=0)
    w_hi      = torch.clamp(width_m - MAX_WIDTH, min=0)
    l_width   = (w_lo + w_hi).mean()
    losses.append(l_width)

    # 2. Finger ⊥ local_normal at p
    #    找 p 在点云中最近点的法向
    dists = torch.norm(pc_xyz - p_norm.unsqueeze(1), dim=-1)  # (B, N)
    nn_idx = dists.argmin(dim=-1)                              # (B,)
    local_n = pc_normals[torch.arange(B), nn_idx]             # (B, 3)
    local_n = F.normalize(local_n, dim=-1)
    fd      = F.normalize(delta_pred, dim=-1)
    # finger_dir 应垂直于法向（点积接近 0）
    l_perp  = (fd * local_n).sum(-1).abs().mean()
    losses.append(l_perp)

    # 3. Affordance: p 处 affordance 应高
    #    (已由采样策略保证，这里作为 soft penalty)
    aff_at_p = aff_labels[torch.arange(B), nn_idx]            # (B,)
    l_aff    = (1.0 - aff_at_p).clamp(min=0).mean()
    losses.append(l_aff * 0.5)

    return sum(losses)
