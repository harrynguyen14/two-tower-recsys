"""Retrieval logit + sampled-softmax loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    """Sampled-softmax retrieval, MOT vector user = hidden[:,-1,:]. Xem formula.md §5.

    [XOA 2026-10-06] MULTI-VECTOR (num_user_vectors > 1) DA XOA HAN cung codes/code_scale/
    routing_regularization/user_vector_cosine. formula.md §5 ghi DA BAC: ca tien de ("nhom
    `both` ket vi output nen hai che do" — artefact phan bo n_u) lan lap luan ("max sinh
    phan hoa" — REMI/MIMA bao argmax routing VAN collapse) deu sai, va thu 2 cach thi cosine
    giua cac vector deu -> 0.98, HR tut 21%. Mac dinh M=1 nen nhanh nay chua bao gio chay.
    Muon thu lai thi DOC memory `gen-recsys-multi-interest-literature` truoc, dung dung lai
    code cu.
    """

    def __init__(self, dim: int, t_base: float = 0.8):
        super().__init__()
        self.dim = dim
        self.t_base = t_base

    def _score(self, e_u: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """e_u (B,d) -> diem (B,C): dot-product THUAN, KHONG chia sqrt(d).

        [SUA 2026-10-07] Bo `/sqrt(dim)` o day. Truoc day `_score` chia sqrt(d) ROI
        `scaled_logit`/`forward_sequence` chia tiep `t_base` => nhiet do hieu dung THAT la
        `sqrt(d) * t_base`, KHONG phai `t_base`. Do duoc voi d=64, t_base=0.1: logit =
        dot/0.80 chu khong phai dot/0.1 — lech 8x. Te hon: no DOI THEO dim (d=128 => 1.131),
        nen moi lan doi `dim` la vo tinh doi ca nhiet do. Gio chi con MOT cho dinh nghia
        nhiet do (`t_base`), mac dinh 0.8 = sqrt(64)*0.1 giu DUNG scale cu => checkpoint va
        moi so da do VAN so sanh duoc. Doi `dim` tu gio khong con keo theo nhiet do.
        """
        return torch.einsum("bd,bcd->bc", e_u, cand)

    def scaled_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        """logit(u,i)/T_base − log_q — dùng CHUNG cho cross_entropy lúc train (forward) VÀ"""
        return self._score(e_u_final, candidate_embeddings) / self.t_base - log_q

    def eval_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Logit XEP HANG luc eval — KHONG tru log_q. Cung ham cham diem nhu train."""
        return self._score(e_u_final, candidate_embeddings)

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
        # [SUA 2026-10-07] bo `/sqrt(dim)` — xem `_score`. Nhiet do chi con o `t_base`.
        pos_logit = (pred * positive_e_i).sum(-1)
        pos_logit = pos_logit / self.t_base - positive_log_q

        neg_logit = torch.einsum("bkd,bcd->bkc", pred, neg_e_i)
        neg_logit = neg_logit / self.t_base - neg_log_q.unsqueeze(1)

        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)
        target = torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).view_as(valid_mask)
        valid = valid_mask.float()
        return (loss * valid).sum() / valid.sum().clamp(min=1)
