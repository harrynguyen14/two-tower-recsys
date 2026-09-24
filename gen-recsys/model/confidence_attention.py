"""Attention SIGMOID + pairwise bias, dùng trong SequenceDecoder. Xem formula.md §4.

KHÔNG phải softmax attention cộng bias. Hai tầng cùng phục vụ câu hỏi ở formula.md §−1:
  - kích hoạt sigmoid (thay softmax)  -> ngắn/dài hạn không còn tranh ngân sách
  - delta ĐỘNG theo query             -> mỗi bước tự chọn phạm vi nhìn
Cờ `use_softmax` / `static_delta` giữ lại để chạy ablation 4 ô (formula.md §4.4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-6


NUM_TS_BUCKETS = 48
MS_PER_SECOND = 1000.0
TS_BUCKET_DIVISOR = 0.1505


def compute_ts_bucket(hist_timestamps: torch.Tensor) -> torch.Tensor:
    """(B, L) epoch-ms -> (B, L, L) uint8, chỉ số bucket của |τ_i − τ_j|."""
    B, L = hist_timestamps.shape
    t = hist_timestamps - hist_timestamps.min()
    delta = (t.view(B, L, 1) - t.view(B, 1, L)).abs()
    bucket = torch.log10(delta.float() / MS_PER_SECOND + 1.0) / TS_BUCKET_DIVISOR
    return bucket.clamp(min=0, max=NUM_TS_BUCKETS).to(torch.uint8)


class AddTsBias(torch.autograd.Function):
    """logit += ts_w[h, bucket[b,i,j]] — CỘNG TẠI CHỖ, không materialize bias."""

    @staticmethod
    def forward(ctx, logit: torch.Tensor, ts_w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.ts_shape = ts_w.shape
        idx_flat = idx.reshape(-1).long()
        with torch.no_grad():
            for h in range(logit.shape[1]):
                logit[:, h].add_(ts_w[h].index_select(0, idx_flat).view(idx.shape).to(logit.dtype))
        ctx.mark_dirty(logit)
        return logit

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (idx,) = ctx.saved_tensors
        H, NB = ctx.ts_shape
        grad_w = grad_out.new_zeros(H, NB, dtype=torch.float32)
        flat_idx = idx.reshape(-1).long()
        for h in range(H):
            grad_w[h].index_add_(0, flat_idx, grad_out[:, h].reshape(-1).float())
        return grad_out, grad_w, None


class ConfidenceModulatedAttention(nn.Module):
    """Sigmoid attention + pairwise bias (xem docstring module)."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True,
        use_softmax: bool = False,
        static_delta: bool = False,
        use_pmi: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_beta = use_beta
        self.use_softmax = use_softmax
        self.static_delta = static_delta
        self.use_pmi = use_pmi

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.beta = nn.Parameter(torch.zeros(num_heads))
        self.delta = nn.Parameter(torch.zeros(num_heads))
        self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1))

        # delta ĐỘNG: δ_h(x_q) = δ_h + w_h·x_q. Init zero ⇒ khởi đầu Y HỆT bản tĩnh,
        # nên bản tĩnh vẫn là nhóm đối chứng hợp lệ (formula.md §4.1).
        self.delta_proj = nn.Linear(dim, num_heads)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # PMI: đo được đỉnh quanh w≈0.5 khi cộng với popularity (formula.md §4.1).
        self.mu = nn.Parameter(torch.full((num_heads,), 0.5))

        pos = torch.arange(max_seq_len)
        rel = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).float()
        self.register_buffer("log_rel_distance", torch.log1p(rel), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_u: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        ts_bucket: torch.Tensor | None = None,
        pmi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        logit = q @ k.transpose(-2, -1)
        logit.div_(math.sqrt(self.head_dim))

        if log_m is not None and self.use_beta:
            logit.add_(self.beta.view(1, -1, 1, 1) * log_m.view(B, 1, 1, L))

        log_rel = self.log_rel_distance[:L, :L].view(1, 1, L, L)
        if self.static_delta:
            logit.add_(self.delta.view(1, -1, 1, 1) * log_rel)
        else:
            # δ_h(x_q): (B,H,L) broadcast với (L,L) — không materialize thêm (B,H,L,L).
            delta_q = self.delta.view(1, -1, 1) + self.delta_proj(x).transpose(1, 2)
            logit.add_(delta_q.unsqueeze(-1) * log_rel)

        if ts_bucket is not None:
            logit = AddTsBias.apply(logit, self.ts_w, ts_bucket)

        if pmi_bias is not None and self.use_pmi:
            logit.add_(self.mu.view(1, -1, 1, 1) * pmi_bias.unsqueeze(1))

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        invalid = causal_mask.view(1, 1, L, L)
        if key_padding_mask is not None:
            invalid = invalid | key_padding_mask.view(B, 1, 1, L)

        if self.use_softmax:
            logit = logit.masked_fill(invalid, float("-inf"))
            attn_weight = torch.nan_to_num(torch.softmax(logit, dim=-1), nan=0.0)
        else:
            # Sigmoid KHÔNG chuẩn hoá theo hàng ⇒ tổng attention tăng theo độ dài chuỗi
            # ("large initial attention norms"). Trừ log(n_q) để chặn — BẮT BUỘC, xem
            # formula.md §4.2. n_q = số token hợp lệ trong tầm nhìn causal của q.
            n_q = (~invalid).sum(dim=-1, keepdim=True).clamp(min=1).to(logit.dtype)
            attn_weight = torch.sigmoid(logit - torch.log(n_q))
            attn_weight = attn_weight.masked_fill(invalid, 0.0)

        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)
