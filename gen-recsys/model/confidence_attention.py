"""Confidence-Modulated Attention — self-attention causal chuẩn (softmax), cộng thêm
λ·log(mat_j) vào logit trước softmax (xem idea.md mục 4.5, "Đề xuất thiết kế cụ thể",
điểm 2 "Gate ở đúng nơi cần gate").

    attn_weight[i,j] = softmax_j( Q_i·K_j/√d + λ·log(mat_j) )

mat_j = confidence của TOKEN j trong chuỗi (không phải của query i) — token có ít lịch sử
quan sát phía sau nó (mat_j thấp -> log(mat_j) rất âm) bị giảm trọng số attention, token
đã "chín" (mat_j gần 1 -> log(mat_j) gần 0) không bị phạt. Đây là cách xử lý cold-USER:
không gate 2 nhánh (không có 2 nhánh nào ở phía user — sequence LÀ toàn bộ user
representation, xem idea.md mục 4.5 đầu "Đề xuất thiết kế cụ thể"), mà điều chỉnh ngay
trong attention.

Dùng standard causal self-attention (softmax chuẩn) — KHÔNG dùng HSTU pointwise
aggregation thật (paper gốc Meta không có softmax để cộng log(mat_j) vào theo đúng công
thức đã chốt). "Decoder-only" ở idea.md quyết định #4 chỉ yêu cầu KHÔNG two-tower/
encoder-decoder, không bắt buộc đúng attention kernel của HSTU.

λ: nn.Parameter học được, không cần khởi tạo đặc biệt (khác τ_u/τ_i/τ_c — xem idea.md
"Việc CẦN LÀM TIẾP" #1). Rủi ro λ tự triệt tiêu về 0 là 1 tín hiệu ablation hữu ích, không
phải lỗi cần né.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConfidenceModulatedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # λ: cường độ Confidence-Modulated Attention, học được, khởi tạo 0 (an toàn — bắt
        # đầu như standard attention thuần, model tự học tăng λ nếu cơ chế có ích).
        self.log_lambda_weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        mat_j: torch.Tensor,  # (B, L) float32 — tanh(N_j/τ_j) của mỗi token trong chuỗi, > 0
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding (bỏ qua)
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_h)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        logit = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L) — [b,h,i,j]

        # log(mat_j): mat_j > 0 luôn đúng vì tanh(N/τ) với N>=0, τ>0 -> mat_j trong [0, 1).
        # Trường hợp mat_j = 0 (token đầu tiên tuyệt đối, N=0) cho log(0) = -inf — CHỦ Ý,
        # đúng ngữ nghĩa "token chưa từng có lịch sử phía sau nó thì không đáng tin cậy để
        # các vị trí sau attend vào nó", clamp nhẹ để tránh NaN số học thuần túy.
        log_mat_j = torch.log(mat_j.clamp(min=1e-6))  # (B, L)
        logit = logit + self.log_lambda_weight * log_mat_j.view(B, 1, 1, L)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        logit = logit.masked_fill(causal_mask.view(1, 1, L, L), float("-inf"))
        if key_padding_mask is not None:
            logit = logit.masked_fill(key_padding_mask.view(B, 1, 1, L), float("-inf"))

        attn_weight = torch.softmax(logit, dim=-1)
        attn_weight = torch.nan_to_num(attn_weight, nan=0.0)  # dòng toàn -inf (padding) -> softmax NaN, ép 0
        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v  # (B, H, L, d_h)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)
