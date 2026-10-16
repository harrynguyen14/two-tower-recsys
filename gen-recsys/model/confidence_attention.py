"""Attention SIGMOID + pairwise bias, dùng trong SequenceDecoder. Xem formula.md §4.

KHÔNG phải softmax attention cộng bias:
  - kích hoạt sigmoid (thay softmax)  -> ngắn/dài hạn không còn tranh ngân sách
  - W_ts: bias theo khoảng cách THỜI GIAN (tĩnh hoặc động theo query)
Cờ ablation duy nhất còn lại: `use_qk` (cấu hình tốt nhất là BỎ q.k).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NUM_TS_BUCKETS = 48
MS_PER_SECOND = 1000.0
TS_BUCKET_DIVISOR = 0.1505

# [XOA 2026-10-15] age bias (AgeAwareBias) DA BO HAN — rut gon so hang
# ve W_ts (formula.md §4). age: Pure khong co upload_dt => code chet tren Pure. Muon khoi
# phuc: git show truoc 2026-10-15.


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
    """Sigmoid attention + pairwise bias: q.k (tuy chon) + W_ts (xem docstring module)."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0,
        use_qk: bool = True, dynamic_ts: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # ABLATION FuXi-beta (arXiv 2508.10615): ho bao BO q.k lai TOT HON tren MovieLens.
        # Tat q.k => attention chi con cac bias (chi con W_ts).
        self.use_qk = use_qk

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # W_ts DONG (nang cap 2): bang tra doi tu scalar sang VECTOR, roi dieu bien
        # theo query. Tinh: ts_w[h, bucket] la mot so. Dong: ts_w[h, bucket] la vector
        # d chieu, chieu voi g_h(x_q) => "query nay nen nhin o THANG NAO" (phut? gio?
        # ngay?), khong chi "nhin gan hay xa" nhu mot delta scalar (da xoa 10-16). Xem formula.md §4.1.
        self.dynamic_ts = dynamic_ts
        if dynamic_ts:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1, self.head_dim))
            # g_h(x_q): init zero => khoi dau logit ts = 0 Y HET ts_w=0 cua ban tinh,
            # nen ban tinh van la nhom doi chung hop le.
            # [SUA 2026-10-15 — W_ts dong CHET] Truoc day ts_gate VA ts_w CUNG zero-init:
            # logit += <g(x_q), ts_w[bucket]> => d/d ts_w = g = 0, d/d g = ts_w = 0 => SADDLE,
            # ca hai KHONG BAO GIO nhan gradient (audit 10-14: grad = 0 tuyet doi ca 4 layer;
            # log Kaggle B2 |ts|=0 gts=0 suot 1500 step => B2 thuc chat = "khong co time bias",
            # so sanh B1/B2 VO NGHIA). Cung bug AgeAwareBias da sua 10-09 nhung quen o day.
            # Quy tac LoRA (arXiv 2106.09685: A random, B zero): trong 1 TICH chi DUNG MOT
            # thua so = 0. Giu ts_w = 0 (=> logit ts luc init = 0 CHINH XAC, ablation sach),
            # ts_gate init MAC DINH (khac 0) => d/d ts_w = g(x) != 0 ngay buoc 1.
            self.ts_gate = nn.Linear(dim, num_heads * self.head_dim)
        else:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1))


    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        ts_bucket: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk:
            logit = q @ k.transpose(-2, -1)
            logit.div_(math.sqrt(self.head_dim))
        else:
            # khong co q.k: logit khoi dau = 0, moi tin hieu den TU BIAS.
            logit = torch.zeros(B, self.num_heads, L, L, dtype=q.dtype, device=q.device)

        if ts_bucket is not None:
            if self.dynamic_ts:
                # (B,H,L,d): thang thoi gian ma query q muon doc
                g = self.ts_gate(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                idx = ts_bucket.reshape(-1).long()
                for h in range(self.num_heads):
                    # (B,L,L,d) cho rieng head h — ts_w[h] la (NB,d)
                    vec = self.ts_w[h].index_select(0, idx).view(B, L, L, self.head_dim)
                    logit[:, h] = logit[:, h] + torch.einsum(
                        "bqd,bqpd->bqp", g[:, h].to(vec.dtype), vec
                    ).to(logit.dtype)
            else:
                logit = AddTsBias.apply(logit, self.ts_w, ts_bucket)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        invalid = causal_mask.view(1, 1, L, L)
        if key_padding_mask is not None:
            invalid = invalid | key_padding_mask.view(B, 1, 1, L)

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
