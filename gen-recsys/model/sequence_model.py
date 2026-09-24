"""Ghép 1 token trong chuỗi user: (e_item_final, action_vector, positional encoding) ->"""

from __future__ import annotations

import torch
import torch.nn as nn

from confidence_attention import EPS
from decoder import SequenceDecoder
from action_encoder import ActionEncoder, NUM_ACTION_DIMS

class SequenceModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 513,
        dropout: float = 0.0,
        interleave: bool = True,
        use_beta: bool = True,
        use_checkpoint: bool = False,
        use_softmax: bool = False,
        static_delta: bool = False,
        use_pmi: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.interleave = interleave

        self.action_encoder = ActionEncoder(dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        self.decoder = SequenceDecoder(
            dim, num_heads, num_layers, ffn_dim, dropout, max_seq_len=max_seq_len,
            use_beta=use_beta, use_checkpoint=use_checkpoint, profile_dim=dim,
            use_softmax=use_softmax, static_delta=static_delta, use_pmi=use_pmi,
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,
        action_vectors: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        profile_embedding: torch.Tensor | None = None,
        user_weight: torch.Tensor | None = None,
        log_user_maturity: torch.Tensor | None = None,
        item_weight: torch.Tensor | None = None,
        hist_timestamps: torch.Tensor | None = None,
        hist_video_ids: torch.Tensor | None = None,
        pmi_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trả về hidden state TẠI VỊ TRÍ ITEM: (B, K, dim) trong MỌI chế độ — caller"""
        B, K, _ = item_embeddings.shape
        a = self.action_encoder(action_vectors, hist_timestamps)

        if self.interleave:
            token = torch.stack([item_embeddings, a], dim=2).view(B, 2 * K, self.dim)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.repeat_interleave(2, dim=1)
        else:
            token = item_embeddings + a

        if profile_embedding is not None and user_weight is None:
            raise ValueError(
                "a profile_embedding requires user_weight: ConditionalFiLM is gated by "
                "log(u_i), so without it the profile silently drops out of the graph."
            )

        L = token.shape[1]
        positions = torch.arange(L, device=token.device).unsqueeze(0).expand(B, L)
        token = token + self.position_embedding(positions)

        log_u = log_m = None
        if user_weight is not None and item_weight is not None:
            if self.interleave:
                log_u_tok = user_weight.repeat_interleave(2, dim=1)
                log_m_tok = item_weight.repeat_interleave(2, dim=1)
            else:
                log_u_tok, log_m_tok = user_weight, item_weight


            if log_user_maturity is not None:
                lum = (log_user_maturity.repeat_interleave(2, dim=1)
                       if self.interleave else log_user_maturity)
                log_u = lum
            else:
                log_u = torch.log(log_u_tok + EPS)
            log_m = torch.log(log_m_tok + EPS)

        token_timestamps = None
        if hist_timestamps is not None:
            token_timestamps = (hist_timestamps.repeat_interleave(2, dim=1) if self.interleave
                                else hist_timestamps)

        # PMI là đại lượng CẶP nên tra ở đây rồi truyền xuống attention (formula.md §4.1).
        # Token lẻ (action) kế thừa item_id của token chẵn kề trước qua repeat_interleave(2),
        # nhất quán với log_m/log_u ở trên.
        pmi_bias = None
        if pmi_table is not None and hist_video_ids is not None:
            ids = (hist_video_ids.repeat_interleave(2, dim=1) if self.interleave
                   else hist_video_ids)
            pmi_bias = pmi_table[ids.unsqueeze(2), ids.unsqueeze(1)].to(token.dtype)

        hidden = self.decoder(token, key_padding_mask, log_u=log_u, log_m=log_m,
                              e_profile=profile_embedding,
                              token_timestamps=token_timestamps, pmi_bias=pmi_bias)

        stride = 2 if self.interleave else 1
        return hidden[:, ::stride]
