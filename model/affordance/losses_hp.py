#!/usr/bin/env python3
"""L1-only loss for human-prior supervision training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1HumanPriorLoss(nn.Module):
    """L = L1(pred_score, soft_gt) with soft_gt = human_prior."""

    def forward(
        self,
        prob: torch.Tensor,
        soft_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = prob.reshape(-1)
        target = soft_gt.reshape(-1).float()
        l_l1 = F.l1_loss(pred, target)
        return {"total": l_l1, "l1": l_l1}


# Backward-compatible alias
MseHumanPriorLoss = L1HumanPriorLoss
