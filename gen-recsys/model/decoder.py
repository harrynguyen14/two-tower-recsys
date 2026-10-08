"""Decoder chuỗi — chỉ có 1 "chuỗi" duy nhất trong toàn bộ model: chuỗi hành vi USER (xem"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from confidence_attention import ConfidenceModulatedAttention, compute_ts_bucket


class DecoderBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_qk: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConfidenceModulatedAttention(
            dim, num_heads, dropout, max_seq_len=max_seq_len, use_qk=use_qk,
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
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        ts_bucket: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask, log_m=log_m,
                          ts_bucket=ts_bucket)
        x = x + self.ffn(self.norm2(x))
        return x


class SequenceDecoder(nn.Module):
    """Encode chuỗi [token_1, ..., token_L] -> hidden state tại mỗi vị trí."""

    def __init__(
        self, dim: int, num_heads: int, num_layers: int, ffn_dim: int,
        dropout: float = 0.0, max_seq_len: int = 513,
        use_checkpoint: bool = False,
        use_qk: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim, num_heads, ffn_dim, dropout, max_seq_len=max_seq_len, use_qk=use_qk,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        token_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """log_m truyền cho MỌI layer (không chỉ layer đầu): nó là thuộc tính của TOKEN"""

        ts_bucket = compute_ts_bucket(token_timestamps) if token_timestamps is not None else None

        x = token_embeddings
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    layer, x, key_padding_mask, log_m, ts_bucket, use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask, log_m=log_m, ts_bucket=ts_bucket)
        return self.final_norm(x)
