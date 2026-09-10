"""Decoder chuỗi — chỉ có 1 "chuỗi" duy nhất trong toàn bộ model: chuỗi hành vi USER (xem
idea.md mục 4.5 điểm 4, "CHỈ CÓ 1 CHUỖI DUY NHẤT"). Item KHÔNG có chuỗi riêng — chỉ xuất
hiện làm token (sau khi qua ItemEmbedding, xem item_embedding.py) hoặc làm candidate ở
bước cuối (không đi qua decoder này, xem retrieval.py).

User representation TẠI MỖI BƯỚC THỜI GIAN = hidden state của decoder sau khi encode
chuỗi hành vi (đúng HSTU: user representation derive hoàn toàn từ sequence encoding,
KHÔNG có "user tower" riêng — xem idea.md mục 4.5 đầu "Đề xuất thiết kế cụ thể").

Mỗi decoder block: Confidence-Modulated Attention (xem confidence_attention.py) + FFN,
residual + pre-norm (chuẩn GPT-style, ổn định hơn post-norm khi xếp nhiều layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from confidence_attention import ConfidenceModulatedAttention


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConfidenceModulatedAttention(dim, num_heads, dropout)
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
        mat_j: torch.Tensor,  # (B, L)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mat_j, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SequenceDecoder(nn.Module):
    """Encode chuỗi [token_1, ..., token_K] -> hidden state h_u tại mỗi vị trí.

    token_t đã là 1 vector (B, L, dim) trước khi vào decoder này — việc ghép
    (E_{item_t}, action_vector_t, positional encoding theo t) thành 1 token vector thuộc
    về bước gọi decoder (xem sequence_model.py), KHÔNG nằm trong module này — giữ
    SequenceDecoder chỉ làm đúng 1 việc: encode 1 chuỗi vector đã sẵn sàng.
    """

    def __init__(self, dim: int, num_heads: int, num_layers: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,  # (B, L, dim)
        mat_j: torch.Tensor,  # (B, L) — confidence của mỗi token, dùng ở MỌI layer (xem idea.md)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding
    ) -> torch.Tensor:
        x = token_embeddings
        for layer in self.layers:
            x = layer(x, mat_j, key_padding_mask)
        return self.final_norm(x)  # (B, L, dim) — h_u tại mọi vị trí (causal, vị trí t chỉ thấy <=t)
