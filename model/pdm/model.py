#!/usr/bin/env python3
"""PDM network and diffusion wrapper."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PDMConfig:
    pose_dim: int = 9
    point_channels: int = 7
    point_feat_dim: int = 512
    time_dim: int = 128
    hidden_dim: int = 512
    use_yaw_condition: bool = False
    yaw_dim: int = 2
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class PDMPointEncoder(nn.Module):
    """Simple PointNet-style global encoder for xyz/normal/affordance points."""

    def __init__(self, in_channels: int = 7, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, C) -> global feature (B, out_dim)."""

        x = points.transpose(1, 2).contiguous()
        feat = self.net(x)
        return torch.max(feat, dim=-1)[0]


class PDMDenoiser(nn.Module):
    """Conditional MLP denoiser over packed 9D pose vectors."""

    def __init__(self, config: PDMConfig):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(config.time_dim)
        cond_dim = config.point_feat_dim + (config.yaw_dim if config.use_yaw_condition else 0)
        in_dim = config.pose_dim + config.time_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.pose_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        return self.net(torch.cat([x_t, t_emb, cond], dim=-1))


class PDM(nn.Module):
    """Conditional DDPM/DDIM over object-frame command poses."""

    def __init__(self, config: PDMConfig | None = None):
        super().__init__()
        self.config = config or PDMConfig()
        self.encoder = PDMPointEncoder(
            in_channels=self.config.point_channels,
            out_dim=self.config.point_feat_dim,
        )
        self.denoiser = PDMDenoiser(self.config)

        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.T,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_1m_ab", torch.sqrt(1.0 - alphas_bar))

    def encode_condition(self, points: torch.Tensor, yaw: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.encoder(points)
        if self.config.use_yaw_condition:
            if yaw is None:
                yaw = torch.zeros(cond.shape[0], self.config.yaw_dim, device=cond.device)
                yaw[:, 1] = 1.0
            yaw = yaw.to(device=cond.device, dtype=cond.dtype)
            if yaw.ndim == 1:
                yaw = yaw.unsqueeze(0)
            if yaw.shape[0] == 1 and cond.shape[0] > 1:
                yaw = yaw.expand(cond.shape[0], -1)
            cond = torch.cat([cond, yaw], dim=-1)
        return cond

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_ab[t].unsqueeze(-1) * x0 + self.sqrt_1m_ab[t].unsqueeze(-1) * noise

    def training_loss(
        self,
        pose_norm: torch.Tensor,
        points: torch.Tensor,
        yaw: torch.Tensor | None = None,
        return_pred: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Noise-prediction DDPM objective on normalized pose vectors."""

        batch = pose_norm.shape[0]
        t = torch.randint(0, self.config.T, (batch,), device=pose_norm.device)
        noise = torch.randn_like(pose_norm)
        x_t = self.q_sample(pose_norm, t, noise)
        cond = self.encode_condition(points, yaw=yaw)
        pred_noise = self.denoiser(x_t, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        if return_pred:
            ab = self.alphas_bar[t].unsqueeze(-1)
            x0_pred = (x_t - torch.sqrt(1.0 - ab) * pred_noise) / (torch.sqrt(ab) + 1e-8)
            return loss, x0_pred
        return loss, None

    @torch.no_grad()
    def sample(
        self,
        points: torch.Tensor,
        yaw: torch.Tensor | None = None,
        n_samples: int = 1,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """Sample normalized pose vectors conditioned on object points."""

        cond = self.encode_condition(points, yaw=yaw)
        if cond.shape[0] == 1:
            cond = cond.expand(n_samples, -1)
        elif n_samples != 1:
            raise ValueError("n_samples > 1 is only supported for a single condition")

        total = cond.shape[0]
        x = torch.randn(total, self.config.pose_dim, device=cond.device)
        step_ids = torch.linspace(
            self.config.T - 1,
            0,
            ddim_steps,
            dtype=torch.long,
            device=cond.device,
        )

        for i, t_cur in enumerate(step_ids):
            t_batch = torch.full((total,), int(t_cur.item()), dtype=torch.long, device=cond.device)
            pred_noise = self.denoiser(x, t_batch, cond)
            ab = self.alphas_bar[t_cur]
            x0_hat = (x - torch.sqrt(1.0 - ab) * pred_noise) / (torch.sqrt(ab) + 1e-8)
            x0_hat = x0_hat.clamp(-5.0, 5.0)
            if i < len(step_ids) - 1:
                t_prev = step_ids[i + 1]
                ab_prev = self.alphas_bar[t_prev]
                x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * pred_noise
            else:
                x = x0_hat
        return x

    def save(self, path: str, *, epoch: int, best_loss: float, pose_stats: dict) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": asdict(self.config),
                "epoch": epoch,
                "best_loss": best_loss,
                "pose_stats": pose_stats,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> tuple["PDM", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(PDMConfig(**ckpt.get("config", {}))).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt
