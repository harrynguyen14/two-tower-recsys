"""BCE Loss cho Multi-Task/Multi-Action Ranking — bài toán KHÁC retrieval (xem"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BINARY_ACTION_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "is_profile_enter",
]
NUM_BINARY_ACTIONS = len(BINARY_ACTION_FIELDS)


class RankingLoss(nn.Module):
    """1 head tuyến tính riêng cho mỗi action nhị phân, chấm trên h_candidate — hidden"""

    def __init__(self, dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            action: nn.Linear(dim, 1) for action in BINARY_ACTION_FIELDS
        })

    def forward(
        self,
        h_candidate: torch.Tensor,
        action_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_action_loss = {}
        total = 0.0
        for idx, action in enumerate(BINARY_ACTION_FIELDS):
            logit = self.heads[action](h_candidate).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, action_labels[:, idx])
            per_action_loss[action] = loss
            total = total + loss

        return total / NUM_BINARY_ACTIONS, per_action_loss

    def forward_sequence(
        self,
        h_items: torch.Tensor,
        action_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """[THÊM 2026-09-14] Ranking TOÀN CHUỖI: p(a_t | Φ_0,a_0,…,Φ_t) tại MỌI vị trí —"""
        valid = valid_mask.float()
        denom = valid.sum().clamp(min=1)

        per_action_loss = {}
        total = 0.0
        for idx, action in enumerate(BINARY_ACTION_FIELDS):
            logit = self.heads[action](h_items).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logit, action_labels[..., idx], reduction="none"
            )
            loss = (loss * valid).sum() / denom
            per_action_loss[action] = loss
            total = total + loss

        return total / NUM_BINARY_ACTIONS, per_action_loss
