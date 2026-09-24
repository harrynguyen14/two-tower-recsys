"""Retrieval logit + sampled-softmax loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    def __init__(self, dim: int, t_base: float = 0.1):
        super().__init__()
        self.dim = dim
        self.t_base = t_base

    def scaled_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        """logit(u,i)/T_base − log_q — dùng CHUNG cho cross_entropy lúc train (forward) VÀ"""
        logit = torch.einsum("bd,bcd->bc", e_u_final, candidate_embeddings) / math.sqrt(self.dim)
        return logit / self.t_base - log_q

    def eval_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Logit dùng cho XẾP HẠNG lúc eval — dot-product THUẦN, KHÔNG trừ log_q."""
        return torch.einsum("bd,bcd->bc", e_u_final, candidate_embeddings) / math.sqrt(self.dim)

    def forward(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
        positive_idx: torch.Tensor,
    ) -> torch.Tensor:
        scaled_logit = self.scaled_logit(e_u_final, candidate_embeddings, log_q)
        return F.cross_entropy(scaled_logit, positive_idx)

    def forward_sequence(
        self,
        pred: torch.Tensor,
        positive_e_i: torch.Tensor,
        neg_e_i: torch.Tensor,
        positive_log_q: torch.Tensor,
        neg_log_q: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[THÊM 2026-09-14] Loss TỰ HỒI QUY TOÀN CHUỖI — dự đoán item kế tiếp tại MỌI vị"""
        pos_logit = (pred * positive_e_i).sum(-1) / math.sqrt(self.dim)
        pos_logit = pos_logit / self.t_base - positive_log_q

        neg_logit = torch.einsum("bkd,bcd->bkc", pred, neg_e_i) / math.sqrt(self.dim)
        neg_logit = neg_logit / self.t_base - neg_log_q.unsqueeze(1)

        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)
        target = torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).view_as(valid_mask)
        valid = valid_mask.float()
        return (loss * valid).sum() / valid.sum().clamp(min=1)
