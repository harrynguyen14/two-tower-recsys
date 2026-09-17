"""Mã hoá action_vector (11 chiều) + giờ-trong-ngày thành 1 token a_t.

[THAY 2026-09-17] Thay `GMU` 3 nhánh (reaction/intensity/rhythm) bằng gated action-type
embedding. Lý do bỏ GMU ở phía ACTION — KHÔNG áp dụng cho `content_gmu` phía item, chỗ đó
đúng là multimodal (categorical/numeric, sắp có caption) và giữ nguyên:

1. GMU sinh ra cho MODALITY — những biểu diễn THAY THẾ được cho nhau của CÙNG một vật
   (ảnh/text/audio của 1 item), có cái optional nên cần mask. reaction/intensity/rhythm
   không phải vậy: chúng cùng có mặt, cùng đúng, ở cường độ đầy đủ.
2. `gmu.py` dùng softmax -> tổng trọng số = 1 -> các nhánh CẠNH TRANH. "User like" chỉ
   được nặng thêm bằng cách bóp "user xem hết 95%". Sai về ngữ nghĩa, không phải thiếu
   capacity. (GMU gốc arXiv:1702.01992 với k>2 modality dùng k sigmoid ĐỘC LẬP rồi cộng;
   dạng z/(1-z) chỉ là biến thể 2-modality BỊ BUỘC CHUNG, mà chính bài báo nói nó
   "constrains the model, so that the units trade off between both modalities" — tức
   softmax ở đây tổng quát hoá nhầm nhánh. arXiv:2405.13997 (NeurIPS 2024) chứng minh
   softmax gating gây "unnecessary competition among experts, potentially causing
   representation collapse", sigmoid cần ít mẫu hơn để đạt cùng sai số.)
3. Không mô hình sequential recsys nào cùng lớp dùng gate chuẩn hoá ngang các nhóm hành vi.
   HSTU (arXiv:2402.17152) — đọc source Meta `generative-recommenders`,
   `modules/action_encoder.py` — cho mỗi action type MỘT embedding riêng, multi-hot
   zero-out, rồi CONCAT thành slice riêng biệt, không gate. MBHT (arXiv:2207.05584) cộng
   thuần item ⊕ position ⊕ behavior-type. MB-STR (SIGIR 2022) tương tự.

Công thức:

    a_t = Σ_k  m_k · s_k · E_k  +  W_h · [sin(2πh/24), cos(2πh/24)]

  E_k (dim,)  embedding học được của action type k
  m_k {0,1}   type k có xảy ra ở lượt này không
  s_k         CƯỜNG ĐỘ của chính type k (1.0 nếu type đó không có tín hiệu cường độ)

Khác HSTU ở một điểm, VÀ ĐÓ LÀ PHẦN CHƯA CÓ CÔNG BỐ KIỂM CHỨNG: HSTU biến watch-time liên
tục thành THÊM BIT NHỊ PHÂN bằng ngưỡng (`watchtime_to_action_thresholds_and_weights`), mỗi
bit một slice. Ở đây cường độ ĐIỀU BIẾN BIÊN ĐỘ của đúng embedding hành vi tương ứng. Lý do:
KuaiRand có sẵn cấu trúc CẶP mà cách chia reaction/intensity cũ đã phá vỡ —
`profile_stay_time_norm` là cường độ CỦA `is_profile_enter`; `comment_stay_time_norm` là
cường độ CỦA `is_comment`. Chúng không phải hai nhóm song song. Ngưỡng hoá sẽ vứt đi thứ bậc
(ở 0.9 khác ở 0.3) mà ở đây giữ được miễn phí.

Vì sao dạng này tốt hơn concat+MLP phẳng:
  - Hành vi VẮNG MẶT đóng góp đúng 0, không phải "số 0 đi qua MLP rồi ra bias khác 0". Với
    reaction flags rất thưa, khác biệt này là thật.
  - "like" và "xem hết 95%" có hướng riêng trong không gian dim NGAY TỪ ĐẦU; model không
    phải học lại từ một projection trộn chung rằng chúng là hai sự kiện khác nhau. Đúng thứ
    HSTU đảm bảo bằng slice, nhưng đạt bằng embedding riêng nên KHÔNG phải cắt ngân sách dim
    thủ công.
  - Không gate -> không nhánh nào phải thua để nhánh khác thắng.

NHỊP THỜI GIAN ở token này chỉ còn giờ-trong-ngày. `log1p(gap)` ĐÃ RÚT RA -> relative
attention bias (confidence_attention.py, số hạng W^ts), vì gap là đại lượng CẶP (t_j - t_i): một scalar gắn trên 1 token
không thể diễn đạt "hai tương tác NÀY gần nhau". HSTU đặt nó ở attention bias
(`RelativeBucketedTimeAndPositionBasedBias`), TiSASRec (WSDM 2020) xây cả mô hình quanh ý đó,
DIF-SR (arXiv:2204.11046, SIGIR 2022) lập luận nhét side info vào embedding gây "rank
bottleneck" trên ma trận attention. Giờ-trong-ngày thì ngược lại — đúng là thuộc tính của
TỪNG sự kiện, nên ở lại token, cộng thẳng như HSTU cộng time-bucket embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NUM_ACTION_DIMS = 11  # ACTION_VECTOR_FIELDS, xem schema.py

# Thứ tự ACTION_VECTOR_FIELDS (schema.py) — ĐÃ ĐÓNG BĂNG 2026-09-09, không đổi khi dữ liệu
# đã build:
#   0 is_click, 1 is_like, 2 is_follow, 3 is_comment, 4 is_forward, 5 is_hate,
#   6 long_view, 7 play_ratio, 8 profile_stay_time_norm, 9 comment_stay_time_norm,
#   10 is_profile_enter
#
# (tên, chỉ số m_k, chỉ số s_k hoặc None nếu type này không có tín hiệu cường độ).
# play_ratio (7) làm cường độ cho click: "bấm vào rồi xem bao nhiêu" đúng là cường độ của
# chính hành vi bấm. long_view (6) đứng riêng thành 1 type vì nó là NGƯỠNG DO NỀN TẢNG định
# nghĩa, không phải đại lượng liên tục — xem cảnh báo duration bias cuối file.
ACTION_TYPES: list[tuple[str, int, int | None]] = [
    ("click",         0, 7),     # is_click         × play_ratio
    ("like",          1, None),
    ("follow",        2, None),
    ("comment",       3, 9),     # is_comment       × comment_stay_time_norm
    ("forward",       4, None),
    ("hate",          5, None),
    ("long_view",     6, None),
    ("profile_enter", 10, 8),    # is_profile_enter × profile_stay_time_norm
]

HOURS_PER_DAY = 24
MS_PER_HOUR = 3_600_000
UTC_OFFSET_HOURS = 8  # KuaiRand lấy từ Kuaishou (Trung Quốc), UTC+8


def hour_of_day(hist_timestamps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, K) epoch-ms -> (B, K, 2) = [sin(2πh/24), cos(2πh/24)].

    PHẢI mã hoá dạng (sin, cos), không phải số 0-23 thô: 23:00 và 00:00 kề nhau về hành vi
    nhưng xa nhau tối đa về số. Đưa số thô vào Linear là dạy model rằng nửa đêm nằm ở một đầu
    thang đo — sai.

    Đo trên KuaiRand-Pure: biên độ 11.5× (22:00 chiếm 7.81% tương tác, 04:00 chỉ 0.68%), hai
    đỉnh — tối 20-23h và trưa 12-13h. Trước 2026-09-17 tín hiệu này BỊ BỎ HẲN (chỉ xuất hiện
    trong khai báo LOG_SCHEMA).
    """
    # epoch ms cỡ 1.65e12; float32 chỉ ~7 chữ số có nghĩa -> sẽ lượng tử hoá thành từng phút
    ts = hist_timestamps.to(torch.float64)
    hour = (ts / MS_PER_HOUR + UTC_OFFSET_HOURS) % HOURS_PER_DAY
    angle = (2 * math.pi * hour / HOURS_PER_DAY).to(dtype)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B, K, 2)


class ActionEncoder(nn.Module):
    """action_vector (B, K, 11) + timestamps (B, K) -> a_t (B, K, dim). Xem docstring module."""

    def __init__(self, dim: int, use_hour: bool = True):
        super().__init__()
        self.dim = dim
        self.use_hour = use_hour
        self.num_types = len(ACTION_TYPES)

        # E_k: 1 embedding cho mỗi action type. Parameter thẳng chứ không nn.Embedding vì tra
        # cứu ở đây là multi-hot (nhiều type cùng bật trong 1 lượt), không phải 1 chỉ số.
        self.type_embedding = nn.Parameter(torch.empty(self.num_types, dim))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)  # theo HSTU

        # Buffer chỉ số -> gather vector hoá, không vòng lặp Python trong forward.
        m_idx = [m for _, m, _ in ACTION_TYPES]
        s_idx = [s if s is not None else 0 for _, _, s in ACTION_TYPES]  # 0 = placeholder, bị mask
        has_s = [s is not None for _, _, s in ACTION_TYPES]
        self.register_buffer("m_idx", torch.tensor(m_idx, dtype=torch.long), persistent=False)
        self.register_buffer("s_idx", torch.tensor(s_idx, dtype=torch.long), persistent=False)
        self.register_buffer("has_s", torch.tensor(has_s, dtype=torch.bool), persistent=False)

        if use_hour:
            self.hour_proj = nn.Linear(2, dim)

    def forward(
        self,
        action_vectors: torch.Tensor,                  # (B, K, NUM_ACTION_DIMS)
        hist_timestamps: torch.Tensor | None = None,   # (B, K) int64 ms
    ) -> torch.Tensor:                                 # (B, K, dim)
        m = action_vectors[..., self.m_idx]  # (B, K, T) cờ nhị phân
        s = action_vectors[..., self.s_idx]  # (B, K, T) cường độ thô (cột placeholder là rác)
        # type không có cường độ -> hệ số 1.0. Dùng where() chứ không nhân mask, vì cột
        # placeholder có thể chứa giá trị bất kỳ.
        s = torch.where(self.has_s, s, torch.ones_like(s))

        weight = m * s                    # (B, K, T)
        a = weight @ self.type_embedding  # (B, K, T) @ (T, dim) -> (B, K, dim)

        if self.use_hour and hist_timestamps is not None:
            a = a + self.hour_proj(hour_of_day(hist_timestamps, a.dtype))
        return a


# ---------------------------------------------------------------------------------------
# DURATION BIAS — ĐÃ XỬ LÝ 2026-09-17 ở tầng dữ liệu (build_sequences.py)
#
# D2Q (arXiv:2206.06003, KDD 2022 — Kuaishou, đã chạy production trên Kuaishou App) chỉ ra
# thời lượng video là BIẾN GÂY NHIỄU: nó "concurrently affects video exposure and watch-time
# prediction". Đo trên chính KuaiRand-Pure (1,436,609 dòng): play_ratio THÔ giảm ĐƠN ĐIỆU
# 4.55× từ decile duration ngắn nhất (0.53, median 10.2s) xuống dài nhất (0.13, median
# 294.6s). Model học tín hiệu đó sẽ học "video ngắn = user thích".
#
# `play_ratio` giờ là THỨ HẠNG PHẦN TRĂM TRONG NHÓM duration, không phải tỉ lệ thô — sau khi
# sửa độ lệch còn 1.22×. Schema không đổi (vẫn 11 chiều, vẫn [0,1]), nên s_k của click ở
# trên dùng được y nguyên. Xem build_sequences._compute_derived_action_fields và
# test_duration_debias.py.
#
# `long_view` KHÔNG sửa: đo được nó gần như phẳng theo duration (0.31-0.37, không xu hướng)
# vì Kuaishou đã định nghĩa ngưỡng của nó theo duration sẵn rồi.
#
# CẦN BUILD LẠI DỮ LIỆU: checkpoint và output/ sinh trước 2026-09-17 mang play_ratio thô.
# ---------------------------------------------------------------------------------------
