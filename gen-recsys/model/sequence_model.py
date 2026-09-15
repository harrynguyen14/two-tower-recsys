"""Ghép 1 token trong chuỗi user: (e_item_final, action_vector, positional encoding) ->
1 vector (dim,), rồi đưa cả chuỗi vào SequenceDecoder (xem decoder.py).

token_t = (item_id_t, action_vector_t, timestamp_t) — item_id_t đã tra qua ItemEmbedding
(item_embedding.py) TRƯỚC KHI vào đây (giữ chuỗi nhẹ, nhất quán — xem idea.md mục 4.5
điểm 4). action_vector_t là 11 chiều multi-hot/multi-value (schema.py ACTION_VECTOR_FIELDS).
Positional encoding dùng learned embedding theo vị trí trong window (không dùng sinusoidal
cố định — cho phép model tự học độ quan trọng của "gần đây" vs "xa", phù hợp domain
short-video nơi hành vi gần đây quan trọng hơn hẳn, xem idea.md mục 4.5 điểm 4).

[SỬA 2026-09-13] Bỏ tham số mat_j (attention không còn cơ chế confidence-modulation, xem
decoder.py/confidence_attention.py).

[SỬA 2026-09-14] Chuyển sang XEN KẼ item/action đúng HSTU (Zhai et al. 2024, arXiv
2402.17152) — chuỗi [Φ_0, a_0, Φ_1, a_1, ..., Φ_{K-1}, a_{K-1}], 2K token cho K lượt.
Lý do KHÔNG phải hình thức: paper nói rõ xen kẽ "enables the ranking task to be formulated
as p(a_{i+1}|Φ_0,a_0,...,Φ_{i+1})" và cho phép "target-aware cross-attention to all n_c
engagements IN ONE PASS" — chấm nhiều candidate 1 lần chạy decoder, thay vì 1 candidate/
lần như thiết kế K+1 cũ. Ngoài ra cộng gộp trộn item+action vào cùng 1 vector dim chiều:
e_A+proj(click) và e_B+proj(like) có thể va chạm — rủi ro thật với action 11 chiều đa nhãn
(SASRec cộng gộp được vì action của nó nhị phân ngầm).

Ràng buộc compute đã được BÁC BỎ bằng đo thật trên T4 — xem comment trong __init__.

forward() nhận (B, K, ...) và trả (B, K, dim) tại VỊ TRÍ ITEM trong CẢ hai chế độ, nên
caller không phải biết chuỗi nội bộ dài K hay 2K.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from confidence_attention import EPS
from decoder import SequenceDecoder

NUM_ACTION_DIMS = 11  # ACTION_VECTOR_FIELDS, xem schema.py


class SequenceModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 513,  # xen kẽ 2K token (K=256) + 1 token profile prepend
        dropout: float = 0.0,
        interleave: bool = True,  # True = xen kẽ [Φ_0,a_0,Φ_1,a_1,...] (HSTU); False = cộng gộp (ablation)
        use_beta: bool = True,   # ablation #3: tắt β·log m_j
        use_gamma: bool = True,  # ablation #4: tắt γ·log u_i·log m_j (số hạng TÍCH)
        use_checkpoint: bool = False,  # gradient checkpointing (tiết kiệm VRAM, xem decoder.py)
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.interleave = interleave

        # action_vector (11 chiều multi-hot/multi-value) -> dim.
        # [SỬA 2026-09-14] Quyết định 2026-09-13 "giữ cộng gộp vì attention rẻ hơn 4 lần,
        # compute là bottleneck (437 ngày/epoch)" đã BỊ BÁC BỎ bằng đo thật trên T4
        # (bench_t4.py). Hai sai lầm trong lập luận cũ:
        #   1. 437 ngày/epoch (review.md L360) là I/O ĐĨA trên KuaiRand-27K (caption 49.2GB
        #      random access) — KHÔNG có phép đo attention/GPU nào trong đó. Pure: caption
        #      ~11MB, vừa RAM -> bottleneck đó không tồn tại.
        #   2. Xen kẽ L=512 chỉ chậm hơn cộng gộp L=256 1.96× (KHÔNG phải 4×) — dim=64 quá
        #      nhỏ để attention O(L²) chiếm ưu thế; caption proj + FFN (tuyến tính theo số
        #      token) mới chiếm phần lớn thời gian.
        # Đo thật (T4, fp16, batch=256): cộng gộp L=256 = 63.8 ms/step (0.08 h/epoch);
        # xen kẽ L=512 = 125.3 ms/step (0.15 h/epoch = 9 phút), peak 1.73/15.6 GB.
        # Compute KHÔNG còn là ràng buộc -> chọn xen kẽ, đúng HSTU, giữ đủ 256 item lịch sử.
        # `interleave=False` giữ đường cộng gộp cũ để ablate.
        self.action_proj = nn.Linear(NUM_ACTION_DIMS, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        self.decoder = SequenceDecoder(
            dim, num_heads, num_layers, ffn_dim, dropout, max_seq_len=max_seq_len,
            use_beta=use_beta, use_gamma=use_gamma, use_checkpoint=use_checkpoint,
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,  # (B, K, dim) — e_item_final, đã tra qua ItemEmbedding
        action_vectors: torch.Tensor,  # (B, K, NUM_ACTION_DIMS)
        key_padding_mask: torch.Tensor | None = None,  # (B, K) bool, True = padding (theo LƯỢT, không phải token)
        profile_embedding: torch.Tensor | None = None,  # (B, dim) — e_profile, prepend làm token 0
        user_weight: torch.Tensor | None = None,  # (B, K) — u_i TẠI TỪNG LƯỢT, xem dataset.py hist_n_u
        item_weight: torch.Tensor | None = None,  # (B, K) — m_j của item mỗi lượt
    ) -> torch.Tensor:
        """Trả về hidden state TẠI VỊ TRÍ ITEM: (B, K, dim) trong MỌI chế độ — caller
        không cần biết chuỗi bên trong dài K, 2K hay 2K+1.

        Xen kẽ: chuỗi nội bộ là [Φ_0, a_0, Φ_1, a_1, ...] (2K token). Hidden tại Φ_t đã
        thấy (Φ_0..Φ_t, a_0..a_{t-1}) NHƯNG CHƯA thấy a_t nhờ causal mask — đúng thứ cần
        cho cả 2 việc:
          - dự đoán item t+1 (retrieval): không leak action của chính t
          - dự đoán a_t (ranking, p(a|Φ)): "nếu hiển thị item này user sẽ làm gì"
        Đây là lý do HSTU xen kẽ, không phải quy ước hình thức (xem docstring module).

        [THÊM 2026-09-14] profile_embedding != None -> PREPEND thành token 0:
            [e_profile, Φ_0, a_0, Φ_1, a_1, ...]   (2K+1 token)

        Vì sao prepend thay vì gate ngoài (thay hẳn nhánh user_embedding cũ):
          1. SỬA BUG THẬT. Loss tự hồi quy toàn chuỗi chấm thẳng trên `hidden`, còn
             e_u_final cũ chỉ dùng trong evaluate() -> static_gmu + 3 MLP gate KHÔNG NHẬN
             GRADIENT, khởi tạo ngẫu nhiên rồi đem xếp hạng. Cơ chế cold-user — trung tâm
             câu hỏi nghiên cứu — thực chất là nhiễu trắng. Prepend cho gradient chảy về
             e_profile từ MỌI vị trí của loss (K điểm, không phải 1).
          2. ĐÚNG NGỮ NGHĨA HƠN. Gate g_u trả lời "user này cold hay warm" — 1 câu trả lời
             cho cả chuỗi. Attention trả lời "tại vị trí i, profile đáng chú ý bao nhiêu so
             với 2i token lịch sử đang có" — RIÊNG cho từng i. Tại i=1 model cần profile;
             tại i=180 profile gần như thừa. Gate không phân biệt được vì user_weight cũ là
             1 scalar/chuỗi. Prepend biến "cold user" từ thuộc tính TĨNH của user thành đại
             lượng BIẾN THIÊN trong chuỗi — đúng hơn, vì mọi user đều cold ở token đầu.
          3. Đúng cách GPT/BERT điều kiện hóa (prefix token, không phải nhánh phụ) và đúng
             tinh thần HSTU: mọi tín hiệu thành token trong MỘT chuỗi thống nhất.
        Chi phí: 1 token (L: 512 -> 513, <0.2% so với 125.3 ms/step đo thật trên T4)."""
        B, K, _ = item_embeddings.shape
        a = self.action_proj(action_vectors)  # (B, K, dim)

        if self.interleave:
            # (B, K, 2, dim) -> (B, 2K, dim), thứ tự Φ_0, a_0, Φ_1, a_1, ...
            token = torch.stack([item_embeddings, a], dim=2).view(B, 2 * K, self.dim)
            if key_padding_mask is not None:
                # 1 lượt padding -> CẢ 2 token (item + action) của lượt đó là padding
                key_padding_mask = key_padding_mask.repeat_interleave(2, dim=1)  # (B, 2K)
        else:
            token = item_embeddings + a  # (B, K, dim) — đường cộng gộp cũ (ablation)

        if profile_embedding is not None:
            token = torch.cat([profile_embedding.unsqueeze(1), token], dim=1)  # (B, L+1, dim)
            if key_padding_mask is not None:
                # Token profile LUÔN là token thật, kể cả user không có lượt nào — đúng nhóm
                # cold user cần nó nhất. Mask nhầm chỗ này = mask mất chính token cứu họ.
                pad_false = torch.zeros(B, 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad_false, key_padding_mask], dim=1)

        L = token.shape[1]
        positions = torch.arange(L, device=token.device).unsqueeze(0).expand(B, L)
        token = token + self.position_embedding(positions)

        # --- log_u / log_m cho pairwise confidence bias (xem confidence_attention.py) ---
        # user_weight/item_weight vào đây theo LƯỢT (B, K); attention cần theo TOKEN (B, L).
        # Phép mở rộng phải khớp CHÍNH XÁC cách token được xếp ở trên, nếu không bias sẽ gắn
        # nhầm confidence cho nhầm token — lại một bug im lặng.
        log_u = log_m = None
        if user_weight is not None and item_weight is not None:
            if self.interleave:
                # Lượt t -> 2 token (Φ_t, a_t): cả hai cùng thuộc 1 lượt nên cùng u/m.
                log_u_tok = user_weight.repeat_interleave(2, dim=1)  # (B, 2K)
                log_m_tok = item_weight.repeat_interleave(2, dim=1)
            else:
                log_u_tok, log_m_tok = user_weight, item_weight  # (B, K)

            if profile_embedding is not None:
                # Token profile: u = giá trị của lượt ĐẦU (nó đứng trước mọi lượt), m = 1
                # -> log m = 0 -> token profile KHÔNG bị phạt bởi số hạng β/γ. Đúng ngữ
                # nghĩa: "độ hiếm" là khái niệm của item, profile không có item nào.
                log_u_tok = torch.cat([log_u_tok[:, :1], log_u_tok], dim=1)
                ones = torch.ones(B, 1, device=token.device, dtype=log_m_tok.dtype)
                log_m_tok = torch.cat([ones, log_m_tok], dim=1)

            log_u = torch.log(log_u_tok + EPS)
            log_m = torch.log(log_m_tok + EPS)

        hidden = self.decoder(token, key_padding_mask, log_u=log_u, log_m=log_m)  # (B, L, dim)

        # Vị trí item lệch 1 khi có profile token: [e_profile, Φ_0, a_0, Φ_1, ...] -> 1, 3, 5...
        # Sai chỗ này là bug IM LẶNG (không crash, chỉ kém) — mọi hidden lệch nửa bước.
        offset = 1 if profile_embedding is not None else 0
        stride = 2 if self.interleave else 1
        return hidden[:, offset::stride]  # (B, K, dim) — luôn tại vị trí item
