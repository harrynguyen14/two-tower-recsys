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

SỐ HẠNG ts — KHOẢNG CÁCH THỜI GIAN THẬT (THÊM 2026-09-17):
δ đo khoảng cách theo VỊ TRÍ (i−j). Hai chuỗi 30 tương tác — một trải 2 giờ, một trải 2
tuần — có cùng ma trận (i−j) nên đến giờ model KHÔNG phân biệt được. Số hạng mới cấp
khoảng cách THẬT:

    ts_bias[h,i,j] = W^ts_h[ bucket(τ_i − τ_j) ]

mỗi head 1 bảng scalar học được, tra theo bucket của hiệu timestamp. Đây đúng cơ chế
`RelativeBucketedTimeAndPositionBasedBias` của HSTU (arXiv:2402.17152, đọc source Meta
`generative-recommenders`): HSTU cộng rel_pos_bias + rel_ts_bias, cả hai đều là scalar
học được tra theo bucket, cộng thẳng vào logit. TiSASRec (WSDM 2020) xây cả mô hình quanh
ý này.

VÌ SAO Ở ĐÂY CHỨ KHÔNG PHẢI TRONG TOKEN: Δt là đại lượng CẶP (τ_i − τ_j). Bản 2026-09-17
sáng nhét log1p(gap) vào action vector như một scalar trên MỘT token — nhưng một scalar
gắn ở vị trí i không diễn đạt được "i và j gần nhau", nó chỉ nói "i cách thằng liền trước
nó bao xa". Thông tin cặp phải ở chỗ nhìn thấy cặp. DIF-SR (arXiv:2204.11046, SIGIR 2022)
lập luận cùng hướng: nhét side info vào embedding gây "rank bottleneck" trên ma trận
attention, nên đưa vào tầng attention.

Bucket hoá bằng log chứ không tuyến tính: phân phối gap lệch nặng (đo trên KuaiRand-Pure
p25=347s, p50=1724s, p75=14500s — hơn 40× giữa p25 và p75). Chỉ số bucket .detach() —
không có gradient chảy qua phép bucket hoá (giống HSTU), gradient chỉ vào bảng scalar.

Tổng tham số thêm: 3 × num_heads + num_heads × (num_ts_buckets+1).
Chi phí runtime: ma trận log(1+i−j) là HẰNG SỐ, tính 1 lần trong __init__ (register_buffer).
Ma trận bucket thời gian PHỤ THUỘC DỮ LIỆU nên phải tính mỗi batch, nhưng — giống `prod` —
chỉ tính MỘT LẦN ở SequenceDecoder rồi dùng chung mọi layer (xem compute_ts_bucket).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-6  # chặn log(0) khi u_i hoặc m_j = 0 (item/user hoàn toàn mới)

# Kẹp tích log(u_i)·log(m_j) — xem lý do chi tiết trong forward(). Đo thật: p95=7.89,
# std=4.15, |max|=190.9 trên 18,936 token KuaiRand-Pure; logit gốc chỉ std≈1.02.
PROD_CLAMP = 10.0


def compute_prod(log_u: torch.Tensor, log_m: torch.Tensor) -> torch.Tensor:
    """log(u_i)·log(m_j) đã clamp — (B, L) × (B, L) -> (B, 1, L, L).

    Tách ra hàm riêng để SequenceDecoder tính MỘT LẦN rồi truyền cho mọi layer: tensor này
    chỉ phụ thuộc (log_u, log_m), vốn không đổi qua các layer, nên tính lại mỗi layer là
    lãng phí 4× bộ nhớ và ĐÃ gây CUDA OOM thật ở B=256, L=513 (xem forward()).

    Trục 1 để size 1 (không phải num_heads): bias giống nhau cho mọi head, hệ số γ_h riêng
    theo head mới nhân vào sau — broadcast lo phần còn lại, không cần nhân bản theo head.
    """
    B, L = log_u.shape
    return (log_u.view(B, 1, L, 1) * log_m.view(B, 1, 1, L)).clamp(-PROD_CLAMP, PROD_CLAMP)


# [SỬA 2026-09-17 — SEV-3, tham số chết] Bản đầu: đơn vị MILI GIÂY, divisor log10(2)=0.301,
# 65 bucket. Đo trên gap THEO CẶP thật (872,844 cặp, lấy mẫu 400 chuỗi từ history_meta.npy):
#     dải thật 26 ms .. 29.4 ngày -> bucket 4.8 .. 31.2
#     tức chỉ ~21/65 bucket từng nhận gradient; 1-11 KHÔNG THỂ tới (gap ≥ 1s đã là bucket 10)
#     và 32-64 cần gap > 136 năm.
# Hai vấn đề: 44 tham số/head chết, và độ phân giải thực chỉ 21 mức với bước ×2 — 1h và 2h
# rơi chung một bucket.
#
# Lưu ý: docstring cũ viện dẫn p25=347s/p50=1724s để biện minh, nhưng đó là gap LIÊN TIẾP.
# Bias nhìn gap THEO CẶP (τ_i − τ_j với mọi j ≤ i), phân phối lệch cao hơn hẳn:
#     p1=549s  p25=23.3h  p50=70.6h  p75=226h  p99=639h
# Nay dùng GIÂY và bước ×1.41 (√2): dải 0.1 .. 42.6 trên 48 bucket -> dùng 42/48, và độ
# phân giải GẤP ĐÔI (1h và 2h giờ tách được).
NUM_TS_BUCKETS = 48
MS_PER_SECOND = 1000.0
TS_BUCKET_DIVISOR = 0.1505  # log10(2)/2 -> mỗi bucket ×√2 ≈ 1.41 lần khoảng thời gian


def compute_ts_bucket(hist_timestamps: torch.Tensor) -> torch.Tensor:
    """(B, L) epoch-ms -> (B, L, L) uint8, chỉ số bucket của |τ_i − τ_j|.

    Tính MỘT LẦN ở SequenceDecoder rồi truyền cho mọi layer, cùng lý do như `compute_prod`:
    chỉ phụ thuộc timestamps (không đổi qua layer), tính lại mỗi layer là nhân bản tensor
    (B,L,L) và giữ hết -> đã từng OOM thật trên T4 ở B=256, L=513.

    Bucket hoá LOG: phân phối gap lệch nặng (KuaiRand-Pure p25=347s, p50=1724s, p75=14500s
    — hơn 40× giữa p25 và p75), chia tuyến tính sẽ dồn gần hết dữ liệu vào bucket 0. Với
    divisor = log10(2), mỗi bucket tương ứng khoảng thời gian gấp đôi bucket trước.

    KHÔNG có gradient (chỉ số rời rạc) — gradient chỉ vào bảng scalar ts_w, giống HSTU
    (`.detach()` trên bucketized_timestamps).

    [SỬA 2026-09-17 — bộ nhớ] Bản đầu ép TOÀN BỘ sang float64 rồi mới trừ, vì epoch-ms
    (~1.65e12) vượt 24-bit mantissa của float32. Nhưng chỉ PHÉP TRỪ cần độ chính xác đó, và
    int64 làm việc đó CHÍNH XÁC TUYỆT ĐỐI; hiệu số thì đủ nhỏ cho float32. Bản cũ tạo 6-7
    tensor (B,1,L,L) float64 nối tiếp — mỗi cái 0.5 GB ở B=256,L=512, ~1.5-2 GB đỉnh sống.
    Bản này trừ bằng int64 rồi chỉ .float() một lần. Đã kiểm: bucket BIT-IDENTICAL với bản
    float64 trên timestamp thật cỡ 1.6e12.

    uint8 chứ không int64: bucket nằm trong [0, 64] nên vừa 1 byte. Tensor này được GIỮ suốt
    forward/backward của MỌI layer, nên 0.50 GB (int64) -> 0.0625 GB (uint8) là tiết kiệm
    THẬT chứ không phải tạm thời. Trả (B,L,L) chứ không (B,1,L,L): AddTsBias index theo head
    nên không cần trục head giả.

    Padding: dataset.py pad timestamp bằng GIÁ TRỊ THẬT ĐẦU TIÊN, nên hiệu giữa hai vị trí
    padding tự nhiên = 0 -> bucket 0. Không cần mask riêng; key_padding_mask đã loại các vị
    trí đó khỏi softmax.
    """
    B, L = hist_timestamps.shape
    # Trừ min trước để số nhỏ lại, rồi trừ theo cặp — cả hai đều int64, KHÔNG mất chính xác.
    t = hist_timestamps - hist_timestamps.min()
    delta = (t.view(B, L, 1) - t.view(B, 1, L)).abs()  # (B, L, L) int64, chính xác tuyệt đối
    # +1 để log10 xác định tại delta=0 (hai tương tác cùng mốc thời gian -> bucket 0)
    # Chia MS_PER_SECOND trước khi log: đơn vị giây, xem ghi chú NUM_TS_BUCKETS ở trên.
    bucket = torch.log10(delta.float() / MS_PER_SECOND + 1.0) / TS_BUCKET_DIVISOR
    return bucket.clamp(min=0, max=NUM_TS_BUCKETS).to(torch.uint8)  # (B, L, L)


class AddTsBias(torch.autograd.Function):
    """logit += ts_w[h, bucket[b,i,j]] — CỘNG TẠI CHỖ, không materialize bias.

    VÌ SAO CẦN HẲN MỘT autograd.Function. Cách viết thẳng `logit + ts_w[:, idx]` tạo ra
    (H,B,L,L) rồi phép cộng tạo thêm một (B,H,L,L) nữa. Ở B=256, L=512, H=4, fp32 mỗi cái
    ĐÚNG 1.00 GB, tức ~2 GB/layer × 4 layer = 8 GB — trên T4 15 GB thì đó là toàn bộ ngân
    sách. (Đo được 2.8 đơn vị (B,H,L,L) mỗi layer, tức ~11 GB chỉ riêng attention scores.)

    Mẹo nằm ở chỗ: bias là phép CỘNG, nên ∂(logit+b)/∂logit = 1 — grad_logit đi thẳng qua
    không đổi. Còn ∂L/∂ts_w[h,k] chỉ là TỔNG grad_output tại mọi (i,j) có bucket = k, tức
    đúng một index_add_. KHÔNG cần giữ bias forward để tính backward.

    VÌ SAO add_ TẠI CHỖ AN TOÀN Ở ĐÂY: backward của matmul chỉ cần q và k, KHÔNG cần output
    của chính nó. Nên sửa tại chỗ kết quả q@k^T là hợp lệ. Đã kiểm cả hai phía ranh giới:
        add_ trên output matmul  -> HỢP LỆ, q.grad bình thường
        add_ trên output softmax -> RuntimeError (softmax backward CẦN output của nó)
    Vì vậy mọi phép cộng bias phải nằm TRƯỚC softmax. Sau softmax là hỏng.

    Đã kiểm: forward BIT-IDENTICAL với `logit + ts_w[:, idx].permute(1,0,2,3)`, và grad ts_w
    lệch ĐÚNG 0.0.

    ĐIỀU KIỆN: `logit` phải là tensor do caller SỞ HỮU (vừa ra từ matmul, không ai giữ tham
    chiếu khác). Ở ConfidenceModulatedAttention.forward đúng như vậy.
    """

    @staticmethod
    def forward(ctx, logit: torch.Tensor, ts_w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.ts_shape = ts_w.shape
        idx_flat = idx.reshape(-1).long()
        with torch.no_grad():
            for h in range(logit.shape[1]):
                # index_select trên bảng (NB,) -> (B,L,L): chỉ 1/H kích thước bias đầy đủ,
                # và là tensor TẠM (giải phóng ngay sau add_). idx là uint8 -> .long().
                logit[:, h].add_(ts_w[h].index_select(0, idx_flat).view(idx.shape).to(logit.dtype))
        # mark_dirty: báo autograd rằng `logit` bị sửa TẠI CHỖ và Function này sở hữu việc đó.
        # Không có nó, output bị coi là VIEW của input, và mọi masked_fill_ phía sau sẽ báo
        # "Output 0 ... is a view and is being modified inplace" — autograd từ chối vì
        # view + inplace sẽ ghi đè custom backward và cho gradient SAI.
        ctx.mark_dirty(logit)
        return logit

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (idx,) = ctx.saved_tensors
        H, NB = ctx.ts_shape
        grad_w = grad_out.new_zeros(H, NB, dtype=torch.float32)
        flat_idx = idx.reshape(-1).long()
        for h in range(H):
            grad_w[h].index_add_(0, flat_idx, grad_out[:, h].reshape(-1).float())
        return grad_out, grad_w, None  # bias cộng -> grad_logit đi thẳng qua



# ---------------------------------------------------------------------------------------
# FlexAttention (THỬ NGHIỆM, mặc định TẮT) — xem ConfidenceModulatedAttention.use_flex
#
# FlexAttention (PyTorch 2.5+) sinh kernel flash-style hợp nhất từ một `score_mod` tuỳ ý, nên
# về nguyên tắc KHÔNG BAO GIỜ materialize ma trận (B,H,L,L) — đúng thứ đang ngốn ~9 GB ở đây.
# Nhưng:
#   - kernel là Triton, mà T4 là Turing (sm_75); Triton hỗ trợ sm_75 nhưng ÍT ĐƯỢC KIỂM CHỨNG
#     hơn Ampere+, và head_dim=16 (dim 64 / 4 head) là bất thường nhỏ với các kernel này;
#   - backward CHỈ có trên CUDA (trên CPU raise NotImplementedError);
#   - không bọc torch.compile thì nó chạy đường "unfused" và materialize ĐÚNG cái ta muốn
#     tránh — tức chậm hơn và KHÔNG tiết kiệm gì.
# Vì vậy đây là đường THỬ NGHIỆM có công tắc: bật bằng --flex-attention, và nếu lần gọi đầu
# lỗi thì TỰ ĐỘNG quay về đường thủ công (đã kiểm chứng) thay vì làm hỏng cả run.
_FLEX_AVAILABLE = False
_flex_attention = None
_create_block_mask = None
try:  # pragma: no cover - phụ thuộc phiên bản torch
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    _FLEX_AVAILABLE = True
except ImportError:
    pass


class ConfidenceModulatedAttention(nn.Module):
    """Causal self-attention + pairwise confidence bias (xem docstring module).

    Không truyền log_u/log_m -> chạy như standard causal attention (2 số hạng đầu tắt),
    nhưng δ vẫn hoạt động — relative position bias không phụ thuộc confidence.
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_beta: bool = True, use_gamma: bool = True,
        use_flex: bool = False,  # [2026-09-17] THỬ NGHIỆM: FlexAttention, xem ghi chú trên
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
        # use_flex chỉ là YÊU CẦU; _flex_ok là trạng thái THẬT — chuyển False vĩnh viễn ngay
        # lần gọi đầu nếu kernel không chạy được (Turing/Triton/head_dim nhỏ), để không thử
        # lại mỗi step và không làm hỏng run.
        self.use_flex = use_flex and _FLEX_AVAILABLE
        self._flex_ok = self.use_flex
        self._flex_compiled = None

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
        # [THÊM 2026-09-17] Bảng scalar theo bucket khoảng cách THỜI GIAN, mỗi head 1 bảng.
        # Zero-init cùng lý do như β/γ/δ: bắt đầu từ đúng attention chuẩn. Đây là bảng TRA
        # CỨU (không phải hệ số nhân) nên δ đo vị trí và ts đo thời gian thật hoàn toàn độc
        # lập — đúng như HSTU cộng rel_pos_bias + rel_ts_bias.
        self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1))

        # log(1 + i − j) cho i >= j (causal), 0 ở nửa trên (bị mask -inf sau, giá trị không
        # dùng tới). Hằng số theo dữ liệu -> buffer, không phải parameter.
        pos = torch.arange(max_seq_len)
        rel = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).float()  # (L, L), i−j
        self.register_buffer("log_rel_distance", torch.log1p(rel), persistent=False)

    def _forward_flex(self, q, k, v, B, L, key_padding_mask, log_u, log_m, prod, ts_bucket):
        """Trả về output nếu FlexAttention chạy được, None nếu phải quay về đường thủ công.

        Mọi bias gộp vào MỘT score_mod, nên kernel cộng chúng NGAY TRONG SRAM và không bao giờ
        ghi ma trận (B,H,L,L) ra HBM. Causal + padding đi qua mask_mod (BlockMask), vốn còn bỏ
        hẳn các khối bị mask -> ít việc hơn attention dày đặc.

        Bất kỳ lỗi nào (Triton không build được trên sm_75, head_dim=16 không hỗ trợ, backward
        thiếu) đều TẮT VĨNH VIỄN đường này và trả None. Không nuốt im lặng: in cảnh báo MỘT lần
        để người chạy biết mình đang ở đường nào."""
        beta, gamma, delta, ts_w = self.beta, self.gamma, self.delta, self.ts_w
        use_beta, use_gamma = self.use_beta, self.use_gamma
        log_rel = self.log_rel_distance[:L, :L]

        def score_mod(score, b, h, qi, ki):
            if log_u is not None and log_m is not None:
                if use_beta:
                    score = score + beta[h] * log_m[b, ki]
                if use_gamma and prod is not None:
                    score = score + gamma[h] * prod[b, 0, qi, ki]
            score = score + delta[h] * log_rel[qi, ki]
            if ts_bucket is not None:
                score = score + ts_w[h, ts_bucket[b, qi, ki].to(torch.int32)]
            return score

        def mask_mod(b, h, qi, ki):
            causal = qi >= ki
            if key_padding_mask is None:
                return causal
            return causal & (~key_padding_mask[b, ki])

        try:
            if self._flex_compiled is None:
                self._flex_compiled = torch.compile(_flex_attention, dynamic=False)
            block_mask = _create_block_mask(mask_mod, B, None, L, L, device=q.device)
            out = self._flex_compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)
        except Exception as exc:  # pragma: no cover - phụ thuộc phần cứng
            self._flex_ok = False
            # CHỈ dòng đầu của lỗi: InductorError kèm cả graph dump dài hàng trăm dòng, in
            # nguyên vẹn sẽ lấp hết log train. Dòng đầu đã đủ để phân biệt các nguyên nhân
            # thật (thiếu trình biên dịch / Triton không hỗ trợ sm_75 / head_dim quá nhỏ).
            first_line = str(exc).strip().splitlines()[0][:160]
            print(f"[flex] TẮT FlexAttention, dùng đường thủ công: {type(exc).__name__}: {first_line}")
            return None

        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padding (bỏ qua)
        log_u: torch.Tensor | None = None,  # (B, L) — log(user_weight tại vị trí i + ε)
        log_m: torch.Tensor | None = None,  # (B, L) — log(item_weight của token j + ε)
        prod: torch.Tensor | None = None,  # (B, 1, L, L) — log(u_i)·log(m_j) ĐÃ clamp, xem dưới
        ts_bucket: torch.Tensor | None = None,  # (B, 1, L, L) int64 — bucket |τ_i−τ_j|, xem compute_ts_bucket
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d_h)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- đường FlexAttention (thử nghiệm, tự tắt khi lỗi) ----
        if self._flex_ok:
            out = self._forward_flex(q, k, v, B, L, key_padding_mask, log_u, log_m, prod, ts_bucket)
            if out is not None:
                return out  # _flex_ok đã tự chuyển False nếu hỏng -> lần sau đi đường thủ công

        logit = q @ k.transpose(-2, -1)
        logit.div_(math.sqrt(self.head_dim))  # (B, H, L, L) — [b,h,i,j]

        # ===== MỌI PHÉP CỘNG BIAS DÙNG add_ TẠI CHỖ =====
        # Backward của matmul chỉ cần q và k, KHÔNG cần output của chính nó, nên sửa tại chỗ
        # kết quả q@k^T là hợp lệ. Đã kiểm cả hai phía ranh giới:
        #     add_ trên output matmul  -> HỢP LỆ, q.grad bình thường
        #     add_ trên output softmax -> RuntimeError (softmax backward CẦN output của nó)
        # Vì vậy mọi bias phải cộng TRƯỚC softmax; sau softmax là hỏng.
        #
        # [SỬA 2026-09-17 — bộ nhớ] Trước đây mỗi `logit = logit + ...` tạo MỘT tensor
        # (B,H,L,L) MỚI mà autograd giữ lại. Ở B=256, L=512, H=4, fp32 mỗi cái ĐÚNG 1.00 GB;
        # đo được 2.8 tensor/layer × 4 layer ≈ 11 GB chỉ riêng attention scores, trên T4
        # 15 GB. add_ tại chỗ bỏ hẳn các bản sao đó. Đã kiểm gradient khớp bản cũ.
        if log_u is not None and log_m is not None:
            # β·log m_j: 1 chiều, thang đo lành (log(m) std=2.10 đo trên 18,936 token thật)
            if self.use_beta:
                logit.add_(self.beta.view(1, -1, 1, 1) * log_m.view(B, 1, 1, L))

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
            #
            # [SỬA 2026-09-15 — OOM] `prod` TÍNH SẴN ở SequenceDecoder và truyền xuống, không
            # tính lại trong từng layer: nó chỉ phụ thuộc (log_u, log_m) — hai tensor GIỐNG
            # HỆT NHAU qua mọi layer — nên tính lại là tạo num_layers bản (B,1,L,L) y hệt và
            # autograd giữ cả 4 cho backward (269 MB/bản ở B=256 -> 1.05 GB thừa, đã CUDA OOM
            # thật trên T4: "Tried to allocate 1.01 GiB").
            # Fallback tự tính khi prod=None để module dùng độc lập được (test đơn vị).
            if self.use_gamma:
                if prod is None:
                    prod = compute_prod(log_u, log_m)
                logit.add_(self.gamma.view(1, -1, 1, 1) * prod)

        # δ_h · log(1 + i − j) — relative position bias, trục ngắn/dài hạn
        logit.add_(self.delta.view(1, -1, 1, 1) * self.log_rel_distance[:L, :L].view(1, 1, L, L))

        # [THÊM 2026-09-17] ts_w[h, bucket(|τ_i−τ_j|)] — relative TIME bias. Khác δ ở chỗ δ
        # đo khoảng cách VỊ TRÍ còn cái này đo khoảng cách THẬT: hai chuỗi cùng 30 tương tác,
        # một trải 2 giờ một trải 2 tuần, có cùng (i−j) nhưng khác hẳn ở đây.
        # AddTsBias cộng tại chỗ và tính grad ts_w bằng index_add_ trong backward, nên KHÔNG
        # bao giờ materialize bias (B,H,L,L) — xem docstring class, ~8 GB tiết kiệm.
        if ts_bucket is not None:
            logit = AddTsBias.apply(logit, self.ts_w, ts_bucket)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        logit.masked_fill_(causal_mask.view(1, 1, L, L), float("-inf"))
        if key_padding_mask is not None:
            logit.masked_fill_(key_padding_mask.view(B, 1, 1, L), float("-inf"))

        # Dòng toàn -inf (lượt padding) -> softmax NaN. Ép về 0 KHÔNG in-place được: softmax
        # backward cần chính output của nó, mà attn_weight còn đi tiếp vào `@ v`, nên sửa tại
        # chỗ sẽ báo "output 0 of SoftmaxBackward0 ... at version 1". (Thử nan_to_num_ riêng
        # lẻ thì "chạy được" chỉ vì tensor đó không được dùng tiếp — không đại diện.)
        # Đây chính là ranh giới add_: TRƯỚC softmax thì in-place an toàn, SAU thì không.
        attn_weight = torch.softmax(logit, dim=-1)
        attn_weight = torch.nan_to_num(attn_weight, nan=0.0)
        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v  # (B, H, L, d_h)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)
