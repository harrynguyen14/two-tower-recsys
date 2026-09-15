"""Causal self-attention + PAIRWISE CONFIDENCE BIAS, dùng trong SequenceDecoder.

[LỊCH SỬ 2026-09-13] Đã bỏ số hạng λ·log(mat_j) cộng vào logit (attention-bias theo
maturity của TOKEN). Lý do bỏ: `mat_j` là `item_weight` của token trong lịch sử, KHÔNG
phải tín hiệu về user — trùng vai trò với gate g_i (item_embedding.py), cả hai cùng dựa
vào item_weight làm cùng 1 việc. Đo thực nghiệm: gradient của λ ~1e-17 với item_weight≈1
(phần lớn dữ liệu warm) — cơ chế gần như không học được gì.

[THÊM 2026-09-14] Quay lại tầng attention, nhưng KHÁC HẲN về bản chất — nhìn CẶP (i, j)
thay vì chỉ j:

    bias[h,i,j] = (β_h + γ_h·log(u_i+ε))·log(m_j+ε)  +  δ_h·log(1 + i − j)

với u_i = user_weight TẠI VỊ TRÍ i (tanh(N_u tại t_i / τ_u), per-position — xem
dataset.py hist_n_u), m_j = item_weight của token j.

Ba số hạng, ba việc khác nhau. Ma trận 4 ô cold-start:

                    item warm          item cold
    user warm    ① attention chuẩn   ② β_h
    user cold    ③ profile token     ④ γ_h

  - β_h·log m_j — hệ số phạt token hiếm, MẶC ĐỊNH. Phủ ô ②. Đây đúng là cơ chế cũ.
  - γ_h·log(u_i)·log(m_j) — hệ số đó THAY ĐỔI theo giai đoạn của user. Phủ ô ④. Đây là
    số hạng duy nhất thật sự mới, và là chỗ g_i KHÔNG làm được: g_i chỉ thấy item, không
    bao giờ thấy user. Đọc thẳng: user warm (u_i→1, log u_i→0) -> hệ số = β_h; user cold
    (log u_i âm lớn) -> hệ số lệch hẳn. Dấu γ_h do model học — γ_h<0 nghĩa là user cold
    phạt item hiếm NẶNG hơn (an toàn, bám phổ biến), γ_h>0 là ngược lại (thăm dò).
  - Ô ③ KHÔNG cần tham số mới: profile token (sequence_model.py, token 0) lo — attention
    tự dồn trọng số về nó khi i nhỏ.
  - Ô ①: u_i→1, m_j→1 -> cả hai log →0 -> bias→0 -> attention chuẩn. Đúng: warm/warm
    không cần can thiệp.

VÌ SAO γ KHÔNG CHẾT NHƯ λ CŨ: λ chết vì log(m_j)≈0 trên phần lớn dữ liệu warm. γ nhân
THÊM với log(u_i), mà u_i biến thiên MẠNH trong chuỗi (mọi user đều cold ở token đầu của
chính mình) — tín hiệu không bị san phẳng. ĐIỀU KIỆN CHẶN: u_i phải per-position. Nếu
u_i hằng theo i thì γ_h·log(u_i) gộp thẳng vào β_h và ta được ĐÚNG cơ chế đã bỏ, lặp lại
nguyên thất bại cũ. Xem test_pairwise_bias.py kiểm đúng điều này.

SỐ HẠNG δ — TRỤC NGẮN/DÀI HẠN:
δ_h·log(1 + i − j) là relative position bias (họ ALiBi/T5), mỗi head 1 độ dốc HỌC ĐƯỢC:
    δ_h < 0  -> head phạt token xa   -> chuyên NGẮN hạn
    δ_h ≈ 0  -> head nhìn đều        -> chuyên DÀI hạn
    δ_h > 0  -> head ưu tiên token xa -> sở thích nền
Model TỰ phân công; ta chỉ cấp trục để nó phân chia trên đó. Khác hẳn head-level
local/global split (chia CỨNG bằng mask) — cái đó đã research và KHÔNG phải gap (LSAN,
patent US 12019671). Đây cũng là thứ thay cho e_short (ShortTermAttentionPool cũ, query
cố định + cửa sổ cứng 5 token) đã bỏ ở user_embedding.py: δ_h học ra độ dốc PHÙ HỢP VỚI
DỮ LIỆU thay vì cửa sổ do người chọn.

Positional encoding tuyệt đối (sequence_model.py) vẫn có, nhưng attention phải tự trừ
pos_i − pos_j qua tích q·k để suy ra khoảng cách — gián tiếp và yếu với dim=64. δ cấp
thẳng đại lượng đó.

Tổng tham số thêm: 3 × num_heads (với num_heads=4 là 12 số).
Chi phí runtime: ma trận log(1+i−j) là HẰNG SỐ, tính 1 lần trong __init__ (register_buffer).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-6  # chặn log(0) khi u_i hoặc m_j = 0 (item/user hoàn toàn mới)

# Kẹp tích log(u_i)·log(m_j) — xem lý do chi tiết trong forward(). Đo thật: p95=7.89,
# std=4.15, |max|=190.9 trên 18,936 token KuaiRand-Pure; logit gốc chỉ std≈1.02.
PROD_CLAMP = 10.0


class ConfidenceModulatedAttention(nn.Module):
    """Causal self-attention + pairwise confidence bias (xem docstring module).

    Không truyền log_u/log_m -> chạy như standard causal attention (2 số hạng đầu tắt),
    nhưng δ vẫn hoạt động — relative position bias không phụ thuộc confidence.
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True, use_gamma: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # [THÊM 2026-09-15] Công tắc ABLATION #3/#4 — tắt từng số hạng ĐỘC LẬP để trả lời
        # câu reviewer chắc chắn hỏi: "số hạng tích γ có đóng góp riêng, hay chỉ là β trá
        # hình?". Phải tắt được γ mà vẫn giữ β (và ngược lại) mới tách bạch được.
        # δ KHÔNG có công tắc: nó tương đương rab của HSTU (và trùng FIRE/ALiBi trong NLP)
        # — thuộc BASELINE, không phải đóng góp của ta, nên luôn bật.
        self.use_beta = use_beta
        self.use_gamma = use_gamma

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Khởi tạo 0 cho cả 3: bắt đầu từ ĐÚNG attention chuẩn, mọi sai lệch khỏi nó phải do
        # gradient kiếm được. Khởi tạo ngẫu nhiên sẽ nhiễu baseline ngay từ step 0 và không
        # còn đo được "bias có đóng góp gì không".
        self.beta = nn.Parameter(torch.zeros(num_heads))   # hệ số phạt token hiếm, mặc định
        self.gamma = nn.Parameter(torch.zeros(num_heads))  # điều biến hệ số đó theo u_i (ô ④)
        self.delta = nn.Parameter(torch.zeros(num_heads))  # độ dốc theo khoảng cách (ngắn/dài hạn)

        # log(1 + i − j) cho i >= j (causal), 0 ở nửa trên (bị mask -inf sau, giá trị không
        # dùng tới). Hằng số theo dữ liệu -> buffer, không phải parameter.
        pos = torch.arange(max_seq_len)
        rel = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).float()  # (L, L), i−j
        self.register_buffer("log_rel_distance", torch.log1p(rel), persistent=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding (bỏ qua)
        log_u: torch.Tensor | None = None,  # (B, L) — log(user_weight tại vị trí i + ε)
        log_m: torch.Tensor | None = None,  # (B, L) — log(item_weight của token j + ε)
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_h)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        logit = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L) — [b,h,i,j]

        # --- pairwise confidence bias ---
        # (β_h + γ_h·log u_i) · log m_j — hệ số phạt token hiếm PHỤ THUỘC giai đoạn của user
        if log_u is not None and log_m is not None:
            # β·log m_j: 1 chiều, thang đo lành (log(m) std=2.10 đo trên 18,936 token thật)
            if self.use_beta:
                logit = logit + self.beta.view(1, -1, 1, 1) * log_m.view(B, 1, 1, L)

            # γ·log(u_i)·log(m_j): TÍCH hai log -> đuôi nặng. Đo thật trên chuỗi KuaiRand-Pure
            # (L=67): std=4.15 nhưng |max|=63.6, và trên mẫu 18,936 token max=190.9 — trong
            # khi logit gốc q·k/√d_h chỉ có std≈1.02. Không kẹp thì 1 cặp (u,m) cực đoan đủ
            # ép softmax về one-hot khi γ rời khỏi 0.
            #
            # Ngưỡng 10 chọn theo phân vị ĐO ĐƯỢC, không phải số tròn tùy ý: p95 của tích là
            # 7.89 < 10, nên clamp giữ NGUYÊN toàn bộ dải tín hiệu hữu ích và chỉ chặn đuôi
            # bệnh lý. Đuôi đó sinh từ log(EPS)=-13.8 khi u hoặc m = 0 TUYỆT ĐỐI — tức token
            # hoàn toàn mới, đúng chỗ tín hiệu KÉM tin cậy nhất lại đang có ảnh hưởng lớn
            # nhất. Kẹp ở đây là sửa đúng nghịch lý đó, không phải che triệu chứng.
            if self.use_gamma:
                prod = (log_u.view(B, 1, L, 1) * log_m.view(B, 1, 1, L)).clamp(-PROD_CLAMP, PROD_CLAMP)
                logit = logit + self.gamma.view(1, -1, 1, 1) * prod

        # δ_h · log(1 + i − j) — relative position bias, trục ngắn/dài hạn
        logit = logit + self.delta.view(1, -1, 1, 1) * self.log_rel_distance[:L, :L].view(1, 1, L, L)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        logit = logit.masked_fill(causal_mask.view(1, 1, L, L), float("-inf"))
        if key_padding_mask is not None:
            logit = logit.masked_fill(key_padding_mask.view(B, 1, 1, L), float("-inf"))

        attn_weight = torch.softmax(logit, dim=-1)
        attn_weight = torch.nan_to_num(attn_weight, nan=0.0)  # dòng toàn -inf (padding) -> softmax NaN, ép 0
        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v  # (B, H, L, d_h)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)
