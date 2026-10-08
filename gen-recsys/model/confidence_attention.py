"""Attention SIGMOID + pairwise bias, dùng trong SequenceDecoder. Xem formula.md §4.

KHÔNG phải softmax attention cộng bias. Hai tầng cùng phục vụ câu hỏi ở formula.md §−1:
  - kích hoạt sigmoid (thay softmax)  -> ngắn/dài hạn không còn tranh ngân sách
  - delta ĐỘNG theo query             -> mỗi bước tự chọn phạm vi nhìn
[XOÁ 2026-10-06] Cờ `use_softmax` / `static_delta` / `use_beta` ĐÃ BỎ — bảng 2x3 ở
formula.md §4.4 đã chạy xong nên các nhánh đối chứng hết việc. Chỉ còn `use_qk` (§4 ghi
cấu hình tốt nhất là BỎ q.k, nên cờ này còn phải chạy được cả hai chiều).

[XOÁ 2026-10-05] Số hạng `mu_h * PPMI(i,j)` ĐÃ BỎ HẲN. Hai lý do:

1. **Model không học nó.** Trong checkpoint đã train đủ (`all-ckp-result.md`), `mu` ở cấu
   hình đề xuất = **0.521** — kẹt ở init 0.5. Cấu hình softmax thì mu chủ động giảm về
   0.14. Dưới sigmoid, các bias bớt quan trọng và model dựa vào q.k nhiều hơn.
2. **Không scale.** Bảng dense N^2 fp16: Pure 115 MB, 27K (N=32,038,693) = **2.05
   exabyte**. Đã đo cả 3 phương án thay thế, loại cả 3 (formula.md §9b.1): sparse top-128
   giữ 16% khối lượng mà tốn 24.6 GB; PMI cấp category-2 có R^2 = 0.025 với PMI item.

CẢNH BÁO khi đọc lại: `--no-pmi` CHƯA ai chạy, nên không có phép đo TRỰC TIẾP về việc bỏ
PMI mất gì. Bằng chứng là mu kẹt ở init — mạnh nhưng GIÁN TIẾP. Căn cứ thêm PMI ban đầu
(analyze.py:m6_pmi, giữ làm hồ sơ): PMI một mình THUA pop (median rank .284 vs .334) nhưng
pop+PMI = .557 (+66%) — hai tín hiệu trực giao. Muốn khôi phục thì đọc hai chỗ đó.
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
        use_qk: bool = True, dynamic_ts: bool = False, static_delta: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # ABLATION FuXi-beta (arXiv 2508.10615): ho bao BO q.k lai TOT HON tren MovieLens.
        # Tat q.k => attention chi con cac bias (rab tinh/dong + maturity + time).
        self.use_qk = use_qk

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.beta = nn.Parameter(torch.zeros(num_heads))
        self.delta = nn.Parameter(torch.zeros(num_heads))
        # W_ts DONG (nang cap 2): bang tra doi tu scalar sang VECTOR, roi dieu bien
        # theo query. Tinh: ts_w[h, bucket] la mot so. Dong: ts_w[h, bucket] la vector
        # d chieu, chieu voi g_h(x_q) => "query nay nen nhin o THANG NAO" (phut? gio?
        # ngay?), khong chi "nhin gan hay xa" nhu delta_h(x_q). Xem formula.md §4.1.
        self.dynamic_ts = dynamic_ts
        if dynamic_ts:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1, self.head_dim))
            # g_h(x_q): init zero => khoi dau logit ts = 0 Y HET ts_w=0 cua ban tinh,
            # nen ban tinh van la nhom doi chung hop le (cung ly le nhu delta_proj).
            self.ts_gate = nn.Linear(dim, num_heads * self.head_dim)
            nn.init.zeros_(self.ts_gate.weight)
            nn.init.zeros_(self.ts_gate.bias)
        else:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1))

        # delta ĐỘNG: δ_h(x_q) = δ_h + w_h·x_q. Init zero ⇒ khởi đầu Y HỆT bản tĩnh,
        # nên bản tĩnh vẫn là nhóm đối chứng hợp lệ (formula.md §4.1).
        self.delta_proj = nn.Linear(dim, num_heads)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # delta TINH (`--static-delta`): o A1/A3 cua bang 2x2 (formula.md §4.4 nhom A).
        # Cai lai 2026-10-08 — KHONG khoi phuc tu git (commit cu chua co 5 bug fix).
        # delta_proj da init ZERO, nen chi can dong bang: weight=0 + khong gradient
        # => delta_proj(x) luon tra 0 => delta_q = delta_h thuan = DUNG ban tinh.
        # Khong can nhanh if trong forward.
        self.static_delta = static_delta
        if static_delta:
            self.delta_proj.weight.requires_grad_(False)
            self.delta_proj.bias.requires_grad_(False)

        pos = torch.arange(max_seq_len)
        rel = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).float()
        self.register_buffer("log_rel_distance", torch.log1p(rel), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
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

        if log_m is not None:
            logit.add_(self.beta.view(1, -1, 1, 1) * log_m.view(B, 1, 1, L))

        log_rel = self.log_rel_distance[:L, :L].view(1, 1, L, L)
        # δ_h(x_q): (B,H,L) broadcast với (L,L) — không materialize thêm (B,H,L,L).
        delta_q = self.delta.view(1, -1, 1) + self.delta_proj(x).transpose(1, 2)
        logit.add_(delta_q.unsqueeze(-1) * log_rel)

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
