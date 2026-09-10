"""Retrieval logit + sampled-softmax loss — nơi DUY NHẤT mat_u/mat_i được gắn vào loss
(xem idea.md mục 4.5 điểm 3, "Chốt loss nền"). Đây là Sampled Softmax Loss / Multi-Class
Cross-Entropy cho bài toán retrieval (dự đoán item tiếp theo trong toàn catalog) — KHÔNG
phải BCE ranking loss (loss khác, chấm điểm nhị phân từng action trên shortlist đã chọn,
KHÔNG gắn mat_u/mat_i — xem idea.md, chưa implement ở đây).

Công thức đã chốt:
    logit(u,i) = h_u · E_i / √d − log_Q(i)
    T(u,i)     = T_base / (mat_u · mat_i)
    P(i | u)   = softmax_i( logit(u,i) / T(u,i) )

KHÔNG cộng thêm log(mat_i) trực tiếp vào logit — E_i đã "biết" mat_i từ trước (qua gate
g_i và content_branch_shrunk trong item_embedding.py), cộng thêm sẽ double-counting và
gây bias phạt candidate cold (đã phát hiện + sửa, xem idea.md mục 4.5 điểm 2).

log_Q(i): sampled-softmax correction cho negative sampling không đều (mục đích thống kê
thuần túy, KHÁC mat_u/mat_i — 2 cơ chế cộng ở 2 số hạng RIÊNG, tránh triệt tiêu/nhân đôi
tác dụng lẫn nhau, xem idea.md mục 4.5 điểm 3 "Quan trọng — tránh double-counting").

T_base: hyperparameter CỐ ĐỊNH (không học), tinh chỉnh qua grid search như mọi
temperature loss thông thường (xem idea.md "Việc CẦN LÀM TIẾP" #1).
"""

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

    def forward(
        self,
        h_u: torch.Tensor,  # (B, dim) — hidden state decoder tại vị trí dự đoán
        candidate_embeddings: torch.Tensor,  # (B, C, dim) — E_i của [positive, negatives...]
        log_q: torch.Tensor,  # (B, C) — log_Q(i), sampled-softmax correction mỗi candidate
        mat_u: torch.Tensor,  # (B,) — tanh(N_u/τ_u) tại thời điểm dự đoán
        mat_i: torch.Tensor,  # (B, C) — tanh(N_i/τ_i) của mỗi candidate (bao gồm cả positive)
        positive_idx: torch.Tensor,  # (B,) int64 — vị trí của positive trong trục C (thường 0)
    ) -> torch.Tensor:
        logit = torch.einsum("bd,bcd->bc", h_u, candidate_embeddings) / math.sqrt(self.dim)  # (B, C)
        logit = logit - log_q

        # T(u,i) = T_base / (mat_u · mat_i) — cặp càng tự tin (cả 2 phía) càng temperature
        # thấp (softmax sắc hơn). Clamp mẫu số tránh chia 0 khi mat_u hoặc mat_i = 0 tuyệt đối.
        temperature = self.t_base / (mat_u.unsqueeze(-1) * mat_i).clamp(min=1e-6)  # (B, C)

        scaled_logit = logit / temperature
        return F.cross_entropy(scaled_logit, positive_idx)
