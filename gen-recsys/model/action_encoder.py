"""Mã hoá action_vector (11 chiều) + giờ-trong-ngày + tuổi item thành 1 token a_t.

[THÊM 2026-10-05] W_age (formula.md §2): age_t = số ngày từ lúc item ra mắt tới lúc user
xem. Bảng tra, không phải scalar chiếu.

[SỬA 2026-10-05, ĐO LẠI] Bucket LOG, không phải tuyến tính như formula.md §2 viết ban đầu.
Lý do: đo trực tiếp trên log_standard 27K (3M dòng join upload_dt) cho age min=0 **max=1421
ngày**, p50=2, p90=48, p99=131 — chỉ 82% <= 30. Tiền đề "age trải 1.5 bậc" của spec là SAI
(thực tế 3.2 bậc), nên A_max=30 tuyến tính sẽ dồn 18% tương tác vào một bin và làm bẩn luôn
phép ablation ‖W_age‖ (không phân biệt được "age vô dụng" với "mã hoá cắt mất tín hiệu").
Chính §2 đã dự trù: "dataset có cửa sổ dài hơn thì đổi sang bucket log như b(Δ)" — 27K đúng
là trường hợp đó. Dùng CÙNG hằng số chia với b(Δ) ở confidence_attention để khỏi có hai
thang log khác nhau trong cùng model.

Đã đo: age ĐỘC LẬP với Delta t (Pearson log1p +0.041) nên nó mang một trục attention hiện
tại hoàn toàn không có; và nó ổn định theo user (corr nửa đầu/nửa sau +0.585) nên là tín
hiệu CÁ NHÂN HOÁ, không chỉ chống bias. Zero-init: khởi đầu y hệt bản không có age, và
‖W_age‖ sau train là ablation miễn phí.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NUM_ACTION_DIMS = 11

ACTION_TYPES: list[tuple[str, int, int | None]] = [
    ("click",         0, 7),
    ("like",          1, None),
    ("follow",        2, None),
    ("comment",       3, 9),
    ("forward",       4, None),
    ("hate",          5, None),
    ("long_view",     6, None),
    ("profile_enter", 10, 8),
]

HOURS_PER_DAY = 24
MS_PER_HOUR = 3_600_000
UTC_OFFSET_HOURS = 8

# Bucket log: bin = floor(log1p(age_ngày) / AGE_BUCKET_DIVISOR), clamp về NUM_AGE_BUCKETS-1.
# Divisor 0.1505 của compute_ts_bucket KHÔNG dùng lại được: nó chia thang GIÂY (9 bậc) nên
# ở thang NGÀY (3.2 bậc) thì 40 bin chỉ phủ tới age 365 — age 365 và 1421 rơi cùng bin, mất
# hẳn phân giải đuôi. Chọn divisor để max đo được (1421) nằm ở bin cuối cùng mà vẫn giữ
# phân giải vùng dày (p50=2): log1p(1500)/48 = 0.1525 -> 49 bin, khớp số bin của b(Δ).
AGE_BUCKET_DIVISOR = 0.1525
NUM_AGE_BUCKETS = 49


def hour_of_day(hist_timestamps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, K) epoch-ms -> (B, K, 2) = [sin(2πh/24), cos(2πh/24)]."""
    ts = hist_timestamps.to(torch.float64)
    hour = (ts / MS_PER_HOUR + UTC_OFFSET_HOURS) % HOURS_PER_DAY
    angle = (2 * math.pi * hour / HOURS_PER_DAY).to(dtype)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def age_bucket(hist_age_days: torch.Tensor) -> torch.Tensor:
    """(B, K) tuổi item theo NGÀY (>=0) -> (B, K) chỉ số bucket log, long.

    `.detach()`: chỉ số bucket không có gradient (giống W^ts_h ở confidence_attention) —
    gradient chỉ chảy vào bảng tra. age âm (item thiếu upload_dt, đánh dấu -1) về bin 0.
    """
    a = hist_age_days.detach().clamp(min=0).float()
    bucket = torch.log1p(a) / AGE_BUCKET_DIVISOR
    return bucket.clamp(min=0, max=NUM_AGE_BUCKETS - 1).long()


class ActionEncoder(nn.Module):
    """action_vector (B, K, 11) + timestamps (B, K) -> a_t (B, K, dim). Xem docstring module."""

    def __init__(self, dim: int, use_age: bool = True):
        super().__init__()
        self.dim = dim
        self.use_age = use_age
        self.num_types = len(ACTION_TYPES)

        self.type_embedding = nn.Parameter(torch.empty(self.num_types, dim))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

        m_idx = [m for _, m, _ in ACTION_TYPES]
        s_idx = [s if s is not None else 0 for _, _, s in ACTION_TYPES]
        has_s = [s is not None for _, _, s in ACTION_TYPES]
        self.register_buffer("m_idx", torch.tensor(m_idx, dtype=torch.long), persistent=False)
        self.register_buffer("s_idx", torch.tensor(s_idx, dtype=torch.long), persistent=False)
        self.register_buffer("has_s", torch.tensor(has_s, dtype=torch.bool), persistent=False)

        self.hour_proj = nn.Linear(2, dim)

        if use_age:
            # ZERO-init (formula.md §2): bước 0 giống y hệt bản không có age, và ‖W_age‖
            # sau train là ablation miễn phí.
            self.age_embedding = nn.Embedding(NUM_AGE_BUCKETS, dim)
            nn.init.zeros_(self.age_embedding.weight)

    def forward(
        self,
        action_vectors: torch.Tensor,
        hist_timestamps: torch.Tensor | None = None,
        hist_age_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = action_vectors[..., self.m_idx]
        s = action_vectors[..., self.s_idx]
        s = torch.where(self.has_s, s, torch.ones_like(s))

        weight = m * s
        a = weight @ self.type_embedding

        if hist_timestamps is not None:
            a = a + self.hour_proj(hour_of_day(hist_timestamps, a.dtype))

        if self.use_age and hist_age_days is not None:
            a = a + self.age_embedding(age_bucket(hist_age_days))
        return a




def _demo() -> None:
    """ponytail: check nhỏ nhất bắt được lỗi mã hoá age và tính zero-init."""
    enc = ActionEncoder(dim=8)

    # 1. zero-init: age KHÔNG được làm đổi gì ở bước 0 (ablation sạch, formula.md §2)
    av = torch.rand(2, 3, NUM_ACTION_DIMS)
    age = torch.tensor([[0.0, 30.0, 1421.0], [5.0, 131.0, 2.0]])
    assert torch.allclose(enc(av), enc(av, hist_age_days=age)), \
        "W_age zero-init nhưng age vẫn đổi output — ablation gián tiếp mất hiệu lực"

    # 2. bucket phải TÁCH được đuôi. Divisor của compute_ts_bucket (thang giây) làm
    #    age 365 và 1421 trùng bin — chính lỗi check này bắt được.
    b = {d: age_bucket(torch.tensor([[float(d)]])).item()
         for d in (0, 2, 30, 48, 131, 365, 1421)}
    assert b[365] != b[1421], f"age 365 và 1421 trùng bin {b[365]} — divisor sai thang"
    assert b[0] < b[2] < b[30] < b[48] < b[131] < b[365], f"bucket phải đơn điệu: {b}"
    assert b[1421] <= NUM_AGE_BUCKETS - 1, f"max đo được (1421) vượt bảng: bin {b[1421]}"
    # vùng DÀY (p50=2 ngày) phải còn phân giải theo ngày, không dồn chung với age 0
    assert b[2] - b[0] >= 4, f"age 0 vs 2 chỉ cách {b[2] - b[0]} bin — mất phân giải vùng dày"

    # 3. age âm (item thiếu upload_dt, đánh dấu -1) về bin 0, không nổ và không âm
    assert age_bucket(torch.tensor([[-1.0]])).item() == 0

    print("action_encoder self-check OK")


if __name__ == "__main__":
    _demo()
