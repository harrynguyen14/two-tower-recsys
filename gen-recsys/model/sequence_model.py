"""Ghép 1 token trong chuỗi user: (E_{item_t}, action_vector_t, positional encoding) ->
1 vector (dim,), rồi đưa cả chuỗi K token vào SequenceDecoder (xem decoder.py).

token_t = (item_id_t, action_vector_t, timestamp_t) — item_id_t đã tra qua ItemEmbedding
(item_embedding.py) TRƯỚC KHI vào đây (giữ chuỗi nhẹ, nhất quán — xem idea.md mục 4.5
điểm 4). action_vector_t là 11 chiều multi-hot/multi-value (schema.py ACTION_VECTOR_FIELDS).
Positional encoding dùng learned embedding theo vị trí trong window K=200 (không dùng
sinusoidal cố định — cho phép model tự học độ quan trọng của "gần đây" vs "xa", phù hợp
domain short-video nơi hành vi gần đây quan trọng hơn hẳn, xem idea.md mục 4.5 điểm 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from decoder import SequenceDecoder

NUM_ACTION_DIMS = 11  # ACTION_VECTOR_FIELDS, xem schema.py


class SequenceModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 200,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # action_vector (11 chiều multi-hot/multi-value) -> dim, cộng trực tiếp vào E_i
        # (giống cách BERT cộng token+position+segment embedding, không phải concat+MLP
        # riêng — giữ đơn giản, 3 nguồn tín hiệu cùng "sống" trong 1 không gian dim chung).
        self.action_proj = nn.Linear(NUM_ACTION_DIMS, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        self.decoder = SequenceDecoder(dim, num_heads, num_layers, ffn_dim, dropout)

    def forward(
        self,
        item_embeddings: torch.Tensor,  # (B, L, dim) — E_{item_t}, đã tra qua ItemEmbedding
        action_vectors: torch.Tensor,  # (B, L, NUM_ACTION_DIMS)
        mat_j: torch.Tensor,  # (B, L) — confidence từng token, xem confidence_attention.py
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding
    ) -> torch.Tensor:
        B, L, _ = item_embeddings.shape
        positions = torch.arange(L, device=item_embeddings.device).unsqueeze(0).expand(B, L)  # (B, L)

        token = item_embeddings + self.action_proj(action_vectors) + self.position_embedding(positions)
        return self.decoder(token, mat_j, key_padding_mask)  # (B, L, dim) = h_u tại mọi vị trí
