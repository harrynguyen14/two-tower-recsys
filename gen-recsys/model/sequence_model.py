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

[SỬA 2026-09-17] Profile KHÔNG còn là token của chuỗi. Trước đây e_profile được prepend
làm token 0; đo thật cho thấy token đó hút 28-33% attention ở MỌI độ dài chuỗi (share gần
như không đổi: 0.331 ở ~4 lượt -> 0.279 ở ~64 lượt) — dấu hiệu attention sink chứ không
phải token được tra cứu khi cần, và nhánh có nó cho recall@5 KÉM hơn (0.0577 vs 0.0628).
Profile giờ đi qua ConditionalFiLM (decoder.py), điều biến mọi layer và tự nhạt dần theo
u_i. Chuỗi trở lại đúng 2K token.

forward() nhận (B, K, ...) và trả (B, K, dim) tại VỊ TRÍ ITEM trong CẢ hai chế độ, nên
caller không phải biết chuỗi nội bộ dài K hay 2K.
"""

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
        max_seq_len: int = 513,  # xen kẽ 2K token (K=256); +1 slot dự phòng
        dropout: float = 0.0,
        interleave: bool = True,  # True = xen kẽ [Φ_0,a_0,Φ_1,a_1,...] (HSTU); False = cộng gộp (ablation)
        use_beta: bool = True,   # ablation #3: tắt β·log m_j
        use_gamma: bool = True,  # ablation #4: tắt γ·log u_i·log m_j (số hạng TÍCH)
        use_checkpoint: bool = False,  # gradient checkpointing (tiết kiệm VRAM, xem decoder.py)
        use_flex: bool = False,  # [2026-09-17] THỬ NGHIỆM: FlexAttention, xem confidence_attention.py
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
        # [THAY 2026-09-17] GMU 3 nhánh -> ActionEncoder (gated action-type embedding).
        # GMU là cơ chế cho MODALITY (biểu diễn thay thế được của cùng 1 vật, có cái
        # optional); reaction/intensity/rhythm không phải vậy — chúng cùng đúng ở cường
        # độ đầy đủ, mà softmax gate thì tổng = 1 nên buộc chúng CẠNH TRANH. Lập luận
        # đầy đủ + dẫn chứng HSTU/MBHT/DIF-SR: docstring action_encoder.py.
        self.action_encoder = ActionEncoder(dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # [ADDED 2026-09-17] The profile conditions EVERY decoder layer via ConditionalFiLM
        # (decoder.py); it is NOT a sequence token. Measured rationale in that class.
        self.decoder = SequenceDecoder(
            dim, num_heads, num_layers, ffn_dim, dropout, max_seq_len=max_seq_len,
            use_beta=use_beta, use_gamma=use_gamma, use_checkpoint=use_checkpoint,
            profile_dim=dim, use_flex=use_flex,
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,  # (B, K, dim) — e_item_final, đã tra qua ItemEmbedding
        action_vectors: torch.Tensor,  # (B, K, NUM_ACTION_DIMS)
        key_padding_mask: torch.Tensor | None = None,  # (B, K) bool, True = padding (theo LƯỢT, không phải token)
        profile_embedding: torch.Tensor | None = None,  # (B, dim) — e_profile, prepend làm token 0
        user_weight: torch.Tensor | None = None,  # (B, K) — u_i TẠI TỪNG LƯỢT, xem dataset.py hist_n_u
        log_user_maturity: torch.Tensor | None = None,  # (B, K) — thay log(u_i), xem learnable_thresholds.py
        item_weight: torch.Tensor | None = None,  # (B, K) — m_j của item mỗi lượt
        hist_timestamps: torch.Tensor | None = None,  # (B, K) int64 ms — cho nhánh nhipthoigian
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
        a = self.action_encoder(action_vectors, hist_timestamps)  # (B, K, dim)

        if self.interleave:
            # (B, K, 2, dim) -> (B, 2K, dim), thứ tự Φ_0, a_0, Φ_1, a_1, ...
            token = torch.stack([item_embeddings, a], dim=2).view(B, 2 * K, self.dim)
            if key_padding_mask is not None:
                # 1 lượt padding -> CẢ 2 token (item + action) của lượt đó là padding
                key_padding_mask = key_padding_mask.repeat_interleave(2, dim=1)  # (B, 2K)
        else:
            token = item_embeddings + a  # (B, K, dim) — đường cộng gộp cũ (ablation)

        # FiLM is gated by log(u_i), so without user_weight the profile would silently drop
        # out of the graph -- it would train, lose nothing visibly, and simply have no
        # effect. That is the failure mode this project has hit three times (the 3-way user
        # gate receiving no gradient, the dead ReLU item gate, tau_c off by 5000x), so it
        # fails loudly instead.
        if profile_embedding is not None and user_weight is None:
            raise ValueError(
                "a profile_embedding requires user_weight: ConditionalFiLM is gated by "
                "log(u_i), so without it the profile silently drops out of the graph."
            )

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


            # [SỬA 2026-09-22] log(u_i) = log(tanh(N_u/τ_u)) BÃO HOÀ VỀ 0 với user nhiều lịch
            # sử, làm số hạng γ·log(u_i)·log(m_j) tắt hẳn — đo được: |prod| yếu đi 1,000 lần
            # từ N_u 0-20 (1.50) xuống N_u 301+ (0.0015), và median rank xấu đi đơn điệu
            # 36 -> 48 theo đúng nhịp đó. Cùng lúc FiLM (gate profile) cũng dùng log_u nên
            # nó chịu chung số phận. Xem learnable_thresholds.py::log_user_maturity().
            #
            # Truyền log_user_maturity từ caller thì dùng nó; không thì giữ đường cũ (ablation
            # so sánh + không phá test đang có).
            if log_user_maturity is not None:
                lum = (log_user_maturity.repeat_interleave(2, dim=1)
                       if self.interleave else log_user_maturity)
                log_u = lum
            else:
                log_u = torch.log(log_u_tok + EPS)
            log_m = torch.log(log_m_tok + EPS)

        # [THÊM 2026-09-17] Timestamps cũng phải mở rộng THEO TOKEN cho relative time bias,
        # và phải khớp CHÍNH XÁC cùng phép xếp token như log_u/log_m ở trên — lệch một nhịp
        # là bias gắn nhầm khoảng cách thời gian cho nhầm cặp token, một bug hoàn toàn im
        # lặng. Lượt t -> 2 token (Φ_t, a_t) cùng xảy ra tại một thời điểm nên cùng τ_t.
        token_timestamps = None
        if hist_timestamps is not None:
            token_timestamps = (hist_timestamps.repeat_interleave(2, dim=1) if self.interleave
                                else hist_timestamps)  # (B, L)

        hidden = self.decoder(token, key_padding_mask, log_u=log_u, log_m=log_m,
                              e_profile=profile_embedding,
                              token_timestamps=token_timestamps)  # (B, L, dim)

        # Sequence is [Φ_0, a_0, Φ_1, a_1, ...] with no prepended token, so item positions
        # start at 0. Getting this wrong is a SILENT bug (no crash, just worse) -- every
        # hidden state would be off by half a step.
        stride = 2 if self.interleave else 1
        return hidden[:, ::stride]  # (B, K, dim) -- always at item positions
