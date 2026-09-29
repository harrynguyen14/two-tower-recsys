"""Retrieval logit + sampled-softmax loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    """Sampled-softmax retrieval. MULTI-VECTOR user khi num_user_vectors > 1 (formula.md §5).

    Vi sao multi-vector: delta_h(x_q) tach HAI che do nhin (gan/xa) o attention, nhung
    `hidden[:,-1,:]` nen ca hai vao MOT diem trong R^d, ma dot-product chi do duoc MOT huong
    tuong dong. Nhom `both` (48.8% test, co CA HAI loai tin hieu) ket o 0.0491 — thap hon
    `only_short` 1.7x va chi hon `neither` 11%. Xem formula.md §-1.

    `max` giu ngu nghia "HOAC": item hop che do nao thi che do do bat. Gradient THUA nen cac
    vector PHAN HOA; gop mem se lam chung troi ve giong nhau.
    """

    def __init__(self, dim: int, t_base: float = 0.1, num_user_vectors: int = 1):
        super().__init__()
        self.dim = dim
        self.t_base = t_base
        self.num_user_vectors = num_user_vectors
        if num_user_vectors > 1:
            # m=0 la identity => M=1 khoi dau Y HET ban cu (nhom doi chung hop le).
            # Cac m khac init NGAU NHIEN NHO — init giong nhau thi max khong pha duoc doi xung.
            self.user_proj = nn.Linear(dim, num_user_vectors * dim, bias=False)
            with torch.no_grad():
                w = self.user_proj.weight.view(num_user_vectors, dim, dim)
                w.zero_()
                w[0] = torch.eye(dim)
                w[1:].normal_(0, 0.02)
                w[1:] += torch.eye(dim)

    def _user_vectors(self, e_u_final: torch.Tensor) -> torch.Tensor:
        """(B,d) -> (B,M,d). M=1 tra ve chinh no, khong ton phep tinh nao."""
        if self.num_user_vectors == 1:
            return e_u_final.unsqueeze(1)
        B = e_u_final.shape[0]
        return self.user_proj(e_u_final).view(B, self.num_user_vectors, self.dim)

    def _score(self, e_u_final: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """max_m (e_u^(m) . e_i)/sqrt(d) — dung CHUNG cho train va eval."""
        eu = self._user_vectors(e_u_final)                       # (B,M,d)
        sim = torch.einsum("bmd,bcd->bmc", eu, cand) / math.sqrt(self.dim)
        return sim.amax(dim=1)                                   # (B,C)

    def user_vector_cosine(self, e_u_final: torch.Tensor) -> float:
        """CHI SO PHAI THEO DOI: cosine trung binh giua cac cap vector user.
        -> 1 nghia la co che DA THOAI HOA ve mot vector, M chi ton tham so."""
        if self.num_user_vectors == 1:
            return float("nan")
        with torch.no_grad():
            eu = F.normalize(self._user_vectors(e_u_final), dim=-1)  # (B,M,d)
            g = torch.einsum("bmd,bnd->bmn", eu, eu)
            M = self.num_user_vectors
            off = ~torch.eye(M, dtype=torch.bool, device=g.device)
            return float(g[:, off].mean())

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
