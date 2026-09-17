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

from confidence_attention import (ConfidenceModulatedAttention, compute_prod,
                                  compute_ts_bucket)


class ConditionalFiLM(nn.Module):
    """FiLM modulation from the static user profile, GATED BY HOW COLD THE USER IS.

        gamma(i) = 1 + s_g * MLP_g(e_profile) * (-log u_i)
        beta(i)  =     s_b * MLP_b(e_profile) * (-log u_i)
        h_i     <- gamma(i) * h_i + beta(i)

    WHY THIS EXISTS -- measured, not assumed. The static profile used to enter the model as
    token 0 of the sequence. Measurement on ckpt_tauc34.pt (measure_profile_attention.py)
    showed that token gets 28-33% of all attention, up to 40x the uniform share, and that
    the share barely moves as the sequence grows (0.331 at ~4 interactions -> 0.279 at ~64).
    A token genuinely carrying profile information would be consulted LESS once the user has
    real history. A flat share is the signature of an attention sink (arXiv 2309.17453): the
    token absorbs leftover softmax mass instead of contributing meaning. That matches the
    ablation, where adding the profile token made recall@5 WORSE (#2 0.0577 vs #1 0.0628).
    The problem was never that profile is ignored -- it is that profile CANNOT STEP BACK.

    So the profile stops competing for attention and modulates the representation instead:
    it reaches every position directly, and its strength is tied to u_i, which decays as the
    user accumulates history. Warm user -> log u_i -> 0 -> FiLM switches itself off and the
    model matches the baseline. Cold user -> profile dominates.

    WHY THE FORM IS HARD-CODED rather than "let an MLP learn when to apply it": that exact
    shortcut has failed twice in this codebase. The old 3-way user gate received NO gradient
    at all (the cold-user mechanism was white noise), and the item gate g_i got stuck at
    0.0367, deep in the region where ReLU's gradient is exactly zero. What did work was the
    explicit form beta + gamma*log(u_i) in confidence_attention.py: +26% recall@5 for
    per-position u_i over static u_i. An explicit form also makes the DEGENERACY CONDITION
    testable -- see test_film_conditioning.py.

    BLOCKING CONDITION (same shape as gamma's): if u_i is constant along the sequence then
    gamma(i) is constant too, and this collapses into ordinary FiLM -- the per-position claim
    evaporates while the model still trains happily and reports no error. Only the self-check
    catches that.

    s_g and s_b start at zero, so at step 0 the model is exactly the baseline and any
    deviation has to be earned by gradient -- the same discipline as beta/gamma/delta init 0.

    NOT the same as broadcast-adding side information to every token: NOVA (arXiv 2103.03578,
    AAAI 2021) showed that harms sequential recommenders ("information overwhelming"). This
    is a conditional MULTIPLICATION -- it changes how a representation is read, it does not
    inject content into it.
    """

    def __init__(self, dim: int, profile_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.to_gamma = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        self.to_beta = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        # ZERO-INIT THE LAST LAYER, not an outer scalar multiplier. Both give an exact
        # identity at step 0, but an outer scalar s also multiplies the gradient flowing
        # BACK to the profile: with s=0 the profile receives exactly 0 gradient, so the MLPs
        # and static_gmu sit frozen until s happens to move. Measured: scale_gamma.grad=9.79
        # (it does move) but e_profile.grad=0.000e+00. That is a fragile design -- whenever s
        # drifts near zero the profile stops learning again, silently.
        # Zeroing only the final Linear's weight+bias keeps d(out)/d(e_profile) = 0 at step 0
        # too, BUT the first Linear still receives gradient through the zeroed weight's own
        # grad path, so the mechanism starts learning immediately and cannot re-freeze.
        # This is the standard residual-branch zero-init (ADM / DiT adaLN-zero).
        for mlp in (self.to_gamma, self.to_beta):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        e_profile: torch.Tensor,  # (B, profile_dim)
        log_u: torch.Tensor,  # (B, L) -- log(u_i + eps) <= 0; magnitude grows as user gets colder
    ) -> torch.Tensor:
        coldness = (-log_u).unsqueeze(-1)  # (B, L, 1) >= 0, per position
        gamma = 1.0 + self.to_gamma(e_profile).unsqueeze(1) * coldness
        beta = self.to_beta(e_profile).unsqueeze(1) * coldness
        return gamma * x + beta


class DecoderBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True, use_gamma: bool = True,  # công tắc ablation, xem confidence_attention.py
        profile_dim: int | None = None,  # [2026-09-17] not None -> this block owns a ConditionalFiLM
        use_flex: bool = False,  # [2026-09-17] THỬ NGHIỆM: FlexAttention
    ):
        super().__init__()
        # Per-layer gamma/beta, NOT shared across layers: each layer represents at a
        # different level of abstraction, so one shared affine would be too strong low in
        # the stack or too weak high in it.
        self.film = ConditionalFiLM(dim, profile_dim) if profile_dim is not None else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConfidenceModulatedAttention(
            dim, num_heads, dropout, max_seq_len=max_seq_len, use_beta=use_beta, use_gamma=use_gamma,
            use_flex=use_flex,
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
        ts_bucket: torch.Tensor | None = None,  # (B,L,L) uint8, cũng tính sẵn ở SequenceDecoder
        e_profile: torch.Tensor | None = None,  # (B, profile_dim) -- for ConditionalFiLM
    ) -> torch.Tensor:
        # FiLM before attention, so the modulated representation is what attention reads.
        # Applied at EVERY layer rather than once at the input: a single application has to
        # survive 4 rounds of attention + FFN + residual mixing, which dilutes it -- the same
        # dilution that made the profile token ineffective. Re-stating the condition per
        # layer is how adaLN conditioning works in DiT.
        if self.film is not None and e_profile is not None and log_u is not None:
            x = self.film(x, e_profile, log_u)
        x = x + self.attn(self.norm1(x), key_padding_mask, log_u=log_u, log_m=log_m, prod=prod,
                          ts_bucket=ts_bucket)
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
        profile_dim: int | None = None,  # [2026-09-17] not None -> ConditionalFiLM in every block
        use_flex: bool = False,  # [2026-09-17] THỬ NGHIỆM: FlexAttention
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim, num_heads, ffn_dim, dropout, max_seq_len=max_seq_len,
                use_beta=use_beta, use_gamma=use_gamma, profile_dim=profile_dim,
                use_flex=use_flex,
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
        e_profile: torch.Tensor | None = None,  # (B, profile_dim) -- ConditionalFiLM condition
        token_timestamps: torch.Tensor | None = None,  # (B, L) epoch-ms THEO TOKEN, xem sequence_model.py
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

        # [THÊM 2026-09-17] bucket |τ_i−τ_j| — cũng tính MỘT LẦN ở đây, cùng lý do như prod:
        # chỉ phụ thuộc timestamps (không đổi qua layer). Khác prod ở chỗ đây là int64 và
        # KHÔNG nằm trong đồ thị autograd, nên chỉ tốn bộ nhớ forward chứ không giữ cho
        # backward. Xem compute_ts_bucket.
        ts_bucket = compute_ts_bucket(token_timestamps) if token_timestamps is not None else None

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
                    layer, x, key_padding_mask, log_u, log_m, prod, ts_bucket, e_profile,
                    use_reentrant=False
                )
            else:
                x = layer(x, key_padding_mask, log_u=log_u, log_m=log_m, prod=prod,
                          ts_bucket=ts_bucket, e_profile=e_profile)
        return self.final_norm(x)  # (B, L, dim) — hidden state tại mọi vị trí (causal, vị trí t chỉ thấy <=t)
