"""Decoder chuỗi — chỉ có 1 "chuỗi" duy nhất trong toàn bộ model: chuỗi hành vi USER (xem"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from confidence_attention import ConfidenceModulatedAttention, compute_ts_bucket


class ConditionalFiLM(nn.Module):
    """FiLM modulation from the static user profile, GATED BY HOW COLD THE USER IS."""

    def __init__(self, dim: int, profile_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.to_gamma = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        self.to_beta = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        for mlp in (self.to_gamma, self.to_beta):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        e_profile: torch.Tensor,
        log_u: torch.Tensor,
    ) -> torch.Tensor:
        coldness = (-log_u).unsqueeze(-1)
        gamma = 1.0 + self.to_gamma(e_profile).unsqueeze(1) * coldness
        beta = self.to_beta(e_profile).unsqueeze(1) * coldness
        return gamma * x + beta


class DecoderBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True,
        profile_dim: int | None = None,
        use_softmax: bool = False,
        static_delta: bool = False,
        use_pmi: bool = True,
    ):
        super().__init__()
        self.film = ConditionalFiLM(dim, profile_dim) if profile_dim is not None else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConfidenceModulatedAttention(
            dim, num_heads, dropout, max_seq_len=max_seq_len, use_beta=use_beta,
            use_softmax=use_softmax, static_delta=static_delta, use_pmi=use_pmi,
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
        log_u: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        ts_bucket: torch.Tensor | None = None,
        e_profile: torch.Tensor | None = None,
        pmi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.film is not None and e_profile is not None and log_u is not None:
            x = self.film(x, e_profile, log_u)
        x = x + self.attn(self.norm1(x), key_padding_mask, log_u=log_u, log_m=log_m,
                          ts_bucket=ts_bucket, pmi_bias=pmi_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class SequenceDecoder(nn.Module):
    """Encode chuỗi [token_1, ..., token_L] -> hidden state tại mỗi vị trí."""

    def __init__(
        self, dim: int, num_heads: int, num_layers: int, ffn_dim: int,
        dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True,
        use_checkpoint: bool = False,
        profile_dim: int | None = None,
        use_softmax: bool = False,
        static_delta: bool = False,
        use_pmi: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim, num_heads, ffn_dim, dropout, max_seq_len=max_seq_len,
                use_beta=use_beta, profile_dim=profile_dim,
                use_softmax=use_softmax, static_delta=static_delta, use_pmi=use_pmi,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_u: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        e_profile: torch.Tensor | None = None,
        token_timestamps: torch.Tensor | None = None,
        pmi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """log_u/log_m truyền cho MỌI layer (không chỉ layer đầu): chúng là thuộc tính của"""

        ts_bucket = compute_ts_bucket(token_timestamps) if token_timestamps is not None else None

        x = token_embeddings
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    layer, x, key_padding_mask, log_u, log_m, ts_bucket, e_profile,
                    pmi_bias, use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask, log_u=log_u, log_m=log_m,
                          ts_bucket=ts_bucket, e_profile=e_profile, pmi_bias=pmi_bias)
        return self.final_norm(x)
