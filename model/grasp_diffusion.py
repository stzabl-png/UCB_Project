#!/usr/bin/env python3
"""
Grasp Pose Diffusion Model (DDPM) — v2

设计原则:
  PointNet++ 负责 WHERE: 预测 force_center (TCP 指尖中心位置)
  GraspDiffusion 负责 HOW: 预测 rotation_6d (夹爪朝向)

输出 (6D):
  rotation_6d (6): 旋转矩阵前两列 flatten
  → approach_dir = rotation[:, 2]  (第三列，Z轴)
  → finger_dir   = rotation[:, 0]  (第一列，X轴)

条件输入 (515D):
  global_feat (512D): PointNet++ SA3 max-pool 物体形状特征
  force_center  (3D): PointNet++ 预测的受力中心（TCP 指尖中心位置）

执行时:
  TCP 位置  = force_center（PointNet++ 给，Diffusion 不预测）
  TCP 姿态  = rotation_from_6d(rotation_6d)
  手腕位置  = force_center - approach_dir × TCP_OFFSET(0.105m)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TCP_OFFSET = 0.105  # Franka 手腕到指尖中心距离 (m)


# ─────────────────────────────────────────────────────────────
# 旋转工具函数
# ─────────────────────────────────────────────────────────────

def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    """(B,3,3) → (B,6): 取前两列 [col0, col1] flatten"""
    c1 = R[..., :, 0]   # (..., 3)
    c2 = R[..., :, 1]   # (..., 3)
    return torch.cat([c1, c2], dim=-1)  # (..., 6)


def rotation_from_6d(r6d: torch.Tensor) -> torch.Tensor:
    """(B,6) → (B,3,3): Gram-Schmidt 正交化还原旋转矩阵"""
    c1 = F.normalize(r6d[..., :3], dim=-1)
    c2 = r6d[..., 3:6]
    c2 = F.normalize(c2 - (c2 * c1).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)   # (..., 3, 3)


def get_approach_dir(r6d: torch.Tensor) -> torch.Tensor:
    """从 6D rotation 提取 approach_dir（旋转矩阵第三列 Z轴）"""
    R = rotation_from_6d(r6d)
    return R[..., :, 2]   # (..., 3)


def get_finger_dir(r6d: torch.Tensor) -> torch.Tensor:
    """从 6D rotation 提取 finger_dir（旋转矩阵第一列 X轴）"""
    R = rotation_from_6d(r6d)
    return R[..., :, 0]   # (..., 3)


def wrist_from_pose(force_center: torch.Tensor,
                    r6d: torch.Tensor) -> torch.Tensor:
    """
    计算手腕位置（EE）。
    force_center: (B,3)  TCP 指尖中心
    r6d:          (B,6)  rotation 6D
    → wrist:      (B,3)
    """
    approach_dir = get_approach_dir(r6d)              # (B,3)
    return force_center - approach_dir * TCP_OFFSET   # (B,3)


# ─────────────────────────────────────────────────────────────
# 时间步嵌入
# ─────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────
# 去噪网络
# ─────────────────────────────────────────────────────────────

class GraspDenoiser(nn.Module):
    """
    条件 MLP 去噪网络

    输入维度:
      rotation_6d (6) + t_emb (128) + global_feat (512) + force_center (3)
      = 6 + 128 + 515 = 649D

    输出: 预测噪声 ε (6D)
    """
    POSE_DIM   = 6
    T_EMB_DIM  = 128
    FEAT_DIM   = 512   # PointNet++ SA3 global feat
    FC_DIM     = 3     # force_center
    COND_DIM   = FEAT_DIM + FC_DIM   # 515

    def __init__(self, hidden: int = 512):
        super().__init__()
        in_dim = self.POSE_DIM + self.T_EMB_DIM + self.COND_DIM  # 649

        self.t_emb = SinusoidalEmbedding(self.T_EMB_DIM)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.POSE_DIM),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                global_feat: torch.Tensor,
                force_center: torch.Tensor) -> torch.Tensor:
        """
        x_t          : (B, 6)   noisy rotation_6d
        t            : (B,)     timestep
        global_feat  : (B, 512) PointNet++ 物体特征
        force_center : (B, 3)   PointNet++ 受力中心预测
        → (B, 6) predicted noise
        """
        t_emb = self.t_emb(t)                                        # (B,128)
        cond  = torch.cat([global_feat, force_center], dim=-1)       # (B,515)
        h     = torch.cat([x_t, t_emb, cond], dim=-1)               # (B,649)
        return self.net(h)


# ─────────────────────────────────────────────────────────────
# DDPM + DDIM
# ─────────────────────────────────────────────────────────────

class GraspDiffusion(nn.Module):
    """
    DDPM over 6D rotation，以 (global_feat + force_center) 为条件。

    训练:
        loss = model.training_loss(rot_6d_gt, global_feat, force_center)

    推理:
        rot_6d = model.sample(global_feat, force_center, n_samples=10)
        → rotation = rotation_from_6d(rot_6d)
        → TCP位置  = force_center（直接用，不预测）
        → 手腕位置 = wrist_from_pose(force_center, rot_6d)
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, hidden: int = 512):
        super().__init__()
        self.T = T
        self.denoiser = GraspDenoiser(hidden=hidden)

        betas       = torch.linspace(beta_start, beta_end, T)
        alphas      = 1.0 - betas
        alphas_bar  = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',      betas)
        self.register_buffer('alphas',     alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_ab',    alphas_bar.sqrt())
        self.register_buffer('sqrt_1m_ab', (1 - alphas_bar).sqrt())

    def q_sample(self, x0, t, noise):
        sa  = self.sqrt_ab[t].unsqueeze(-1)
        s1m = self.sqrt_1m_ab[t].unsqueeze(-1)
        return sa * x0 + s1m * noise

    def training_loss(self, rot_6d_gt: torch.Tensor,
                      global_feat: torch.Tensor,
                      force_center: torch.Tensor,
                      return_x0: bool = False):
        """
        rot_6d_gt    : (B, 6)   ground-truth rotation 6D (已归一化)
        global_feat  : (B, 512) 冻结 PointNet++ 特征
        force_center : (B, 3)   PointNet++ 预测的受力中心
        return_x0    : 若 True，额外返回 x0_pred 供 affordance loss 使用

        返回: loss  或  (loss, x0_pred)
        """
        B = rot_6d_gt.shape[0]
        t     = torch.randint(0, self.T, (B,), device=rot_6d_gt.device)
        noise = torch.randn_like(rot_6d_gt)
        x_t   = self.q_sample(rot_6d_gt, t, noise)
        pred  = self.denoiser(x_t, t, global_feat, force_center)
        loss  = F.mse_loss(pred, noise)

        if return_x0:
            # x0 估计: x0_hat = (x_t - sqrt(1-ab)*pred) / sqrt(ab)
            ab    = self.alphas_bar[t].unsqueeze(-1)       # (B,1)
            x0_pred = (x_t - (1 - ab).sqrt() * pred) / (ab.sqrt() + 1e-8)
            return loss, x0_pred.detach()
        return loss, None

    @torch.no_grad()
    def sample(self, global_feat: torch.Tensor,
               force_center: torch.Tensor,
               n_samples: int = 1,
               ddim_steps: int = 50) -> torch.Tensor:
        """
        global_feat  : (1,512) 或 (B,512)
        force_center : (1,3)   或 (B,3)
        → (n_samples, 6) rotation_6d
        """
        device = global_feat.device
        if global_feat.shape[0] == 1:
            global_feat  = global_feat.expand(n_samples, -1)
            force_center = force_center.expand(n_samples, -1)

        step_ids = torch.linspace(self.T - 1, 0, ddim_steps, dtype=torch.long)
        x = torch.randn(n_samples, GraspDenoiser.POSE_DIM, device=device)

        for i, t_cur in enumerate(step_ids):
            t_batch    = torch.full((n_samples,), t_cur, dtype=torch.long, device=device)
            pred_noise = self.denoiser(x, t_batch, global_feat, force_center)

            ab    = self.alphas_bar[t_cur]
            x0_hat = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
            x0_hat = x0_hat.clamp(-3, 3)

            if i < ddim_steps - 1:
                t_prev  = step_ids[i + 1]
                ab_prev = self.alphas_bar[t_prev]
                x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * pred_noise
            else:
                x = x0_hat

        return x   # (n_samples, 6)

    def save(self, path: str, epoch: int = 0, best_loss: float = 0.0):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.state_dict(),
                    'epoch': epoch, 'best_loss': best_loss}, path)

    def load(self, path: str, device=None):
        ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
        self.load_state_dict(ckpt['model_state_dict'])
        return ckpt.get('epoch', 0), ckpt.get('best_loss', 0.0)
