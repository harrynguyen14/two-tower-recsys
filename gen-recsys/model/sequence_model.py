"""Ghép 1 token trong chuỗi user: (e_item_final, action_vector, positional encoding) ->"""

from __future__ import annotations

import torch
import torch.nn as nn

from decoder import SequenceDecoder
from action_encoder import ActionEncoder

class SequenceModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 513,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        use_qk: bool = True,
        dynamic_ts: bool = False,
        static_delta: bool = False,
        no_delta: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.action_encoder = ActionEncoder(dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        self.decoder = SequenceDecoder(
            dim, num_heads, num_layers, ffn_dim, dropout, max_seq_len=max_seq_len,
            use_checkpoint=use_checkpoint, use_qk=use_qk, dynamic_ts=dynamic_ts,
            static_delta=static_delta, no_delta=no_delta,
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,
        action_vectors: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        hist_timestamps: torch.Tensor | None = None,
        hist_video_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trả về hidden state TẠI VỊ TRÍ ITEM: (B, K, dim) — caller"""
        B, K, _ = item_embeddings.shape
        a = self.action_encoder(action_vectors, hist_timestamps)

        # Chuỗi XEN KẼ [Φ_0, a_0, Φ_1, a_1, ...], L = 2K (formula.md §3, HSTU Table 1).
        token = torch.stack([item_embeddings, a], dim=2).view(B, 2 * K, self.dim)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat_interleave(2, dim=1)

        L = token.shape[1]
        positions = torch.arange(L, device=token.device).unsqueeze(0).expand(B, L)
        token = token + self.position_embedding(positions)

        token_timestamps = None
        if hist_timestamps is not None:
            token_timestamps = hist_timestamps.repeat_interleave(2, dim=1)

        hidden = self.decoder(token, key_padding_mask, token_timestamps=token_timestamps)

        return hidden[:, ::2]
