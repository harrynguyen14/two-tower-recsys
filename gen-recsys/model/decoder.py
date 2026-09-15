"""Decoder chuỗi — chỉ có 1 "chuỗi" duy nhất trong toàn bộ model: chuỗi hành vi USER (xem
idea.md mục 4.5 điểm 4, "CHỈ CÓ 1 CHUỖI DUY NHẤT"). Item KHÔNG có chuỗi riêng — chỉ xuất
hiện làm token (sau khi qua ItemEmbedding, xem item_embedding.py) hoặc làm candidate
(retrieval: ngoài chuỗi, so dot-product; ranking: NỐI vào cuối chuỗi, xem ranking_loss.py
và result.md "CHECKLIST CUỐI CÙNG" 2026-09-13).

User representation TẠI MỖI BƯỚC THỜI GIAN = hidden state của decoder sau khi encode
chuỗi hành vi (đúng HSTU: user representation derive hoàn toàn từ sequence encoding,
KHÔNG có "user tower" riêng — xem idea.md mục 4.5 đầu "Đề xuất thiết kế cụ thể").

[SỬA 2026-09-13] Đã bỏ tham số mat_j khỏi forward — attention không còn cơ chế
confidence-modulation nào (xem confidence_attention.py). Mỗi decoder block giờ chỉ còn:
standard causal self-attention + FFN, residual + pre-norm (chuẩn GPT-style).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from confidence_attention import ConfidenceModulatedAttention, compute_prod


class DecoderBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True, use_gamma: bool = True,  # công tắc ablation, xem confidence_attention.py
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConfidenceModulatedAttention(
            dim, num_heads, dropout, max_seq_len=max_seq_len, use_beta=use_beta, use_gamma=use_gamma,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding
        log_u: torch.Tensor | None = None,  # (B, L) — log(u_i + ε), xem confidence_attention.py
        log_m: torch.Tensor | None = None,  # (B, L) — log(m_j + ε)
        prod: torch.Tensor | None = None,  # (B,1,L,L) tính sẵn ở SequenceDecoder (tránh OOM)
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask, log_u=log_u, log_m=log_m, prod=prod)
        x = x + self.ffn(self.norm2(x))
        return x


class SequenceDecoder(nn.Module):
    """Encode chuỗi [token_1, ..., token_L] -> hidden state tại mỗi vị trí.

    L thường = K (retrieval, chỉ lịch sử) hoặc K+1 (ranking, có candidate nối vào cuối —
    xem sequence_model.py và ranking_loss.py). token_t đã là 1 vector (B, L, dim) trước khi
    vào decoder này — việc ghép (E_item_t, action_vector_t, positional encoding) thành 1
    token vector thuộc về bước gọi decoder (xem sequence_model.py), KHÔNG nằm trong module
    này — giữ SequenceDecoder chỉ làm đúng 1 việc: encode 1 chuỗi vector đã sẵn sàng.
    """

    def __init__(
        self, dim: int, num_heads: int, num_layers: int, ffn_dim: int,
        dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True, use_gamma: bool = True,  # công tắc ablation #3/#4
        use_checkpoint: bool = False,  # gradient checkpointing, xem forward()
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim, num_heads, ffn_dim, dropout, max_seq_len=max_seq_len,
                use_beta=use_beta, use_gamma=use_gamma,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,  # (B, L, dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding
        log_u: torch.Tensor | None = None,  # (B, L) — log(u_i + ε), xem confidence_attention.py
        log_m: torch.Tensor | None = None,  # (B, L) — log(m_j + ε)
    ) -> torch.Tensor:
        """log_u/log_m truyền cho MỌI layer (không chỉ layer đầu): chúng là thuộc tính của
        TOKEN, không phải của biểu diễn ở 1 tầng cụ thể — mỗi tầng attention đều cần biết
        token j đáng tin đến đâu và user đang ở giai đoạn nào.

        [SỬA 2026-09-15] prod = log(u_i)·log(m_j) tính MỘT LẦN ở đây rồi dùng chung cho mọi
        layer, thay vì để từng layer tự tính. prod chỉ phụ thuộc (log_u, log_m) — cả hai
        KHÔNG đổi qua các layer — nên tính lại mỗi layer là tạo num_layers bản (B,1,L,L)
        giống hệt nhau, autograd giữ hết cho backward: ở B=256, L=513, fp32 là 269 MB/bản,
        4 layer = 1.05 GB thừa -> CUDA OOM thật trên T4 (nhánh ablation #4/#5 chết ở
        loss.backward()). Cùng giá trị, cùng gradient — chỉ khác bộ nhớ."""
        prod = None
        if log_u is not None and log_m is not None and any(l.attn.use_gamma for l in self.layers):
            prod = compute_prod(log_u, log_m)

        x = token_embeddings
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # [THÊM 2026-09-15] Gradient checkpointing — KHÔNG lưu tensor trung gian của
                # layer, tính lại khi backward. Đổi ~30% thời gian lấy ~70% bộ nhớ.
                #
                # Vì sao cần: mỗi phép `logit = logit + bias` trong attention tạo một tensor
                # (B,H,L,L) MỚI mà autograd phải giữ. Ở B=256, H=4, L=513, fp32 = 1.00 GiB
                # MỖI BẢN, và có 9 phép như vậy mỗi layer (qk, +β, +γ, +δ, 2×masked_fill,
                # softmax, nan_to_num, dropout). Bật γ thêm đúng 1 bản -> vượt ngưỡng T4
                # 15GB. Khớp chính xác lỗi thật: "Tried to allocate 1.01 GiB".
                #
                # use_reentrant=False: bản mới, bắt buộc để hoạt động đúng với tham số
                # không nhận gradient ở một số nhánh ablation (β/γ bị tắt) — bản reentrant
                # cũ sẽ báo lỗi hoặc bỏ qua gradient trong trường hợp đó.
                # self.training: eval không cần checkpointing (torch.no_grad, không lưu gì).
                x = checkpoint(
                    layer, x, key_padding_mask, log_u, log_m, prod, use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask, log_u=log_u, log_m=log_m, prod=prod)
        return self.final_norm(x)  # (B, L, dim) — hidden state tại mọi vị trí (causal, vị trí t chỉ thấy <=t)
