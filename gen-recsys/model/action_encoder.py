"""Mã hoá action_vector (11 chiều) + giờ-trong-ngày thành 1 token a_t.

[XOÁ 2026-10-09] age_t ĐÃ CHUYỂN khỏi action token, sang confidence_attention.py như một
pairwise bias điều biến theo query (cùng pattern với delta_h(x_q)) — xem AgeAwareBias ở đó
và formula.md §2bis. Lý do: age_t cộng thẳng vào action token là đặc trưng TĨNH của riêng
item tại thời điểm xem, không có chỗ để biểu diễn "user này thích item mới hay cũ" — mọi
user thấy "item tuổi X" đều cộng đúng 1 vector giống hệt nhau. Đưa vào attention cho phép
hệ số bias tự học theo query (= theo chính user đang xem), mà vẫn giữ zero-init/ablation
miễn phí như thiết kế cũ.
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


def hour_of_day(hist_timestamps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, K) epoch-ms -> (B, K, 2) = [sin(2πh/24), cos(2πh/24)]."""
    ts = hist_timestamps.to(torch.float64)
    hour = (ts / MS_PER_HOUR + UTC_OFFSET_HOURS) % HOURS_PER_DAY
    angle = (2 * math.pi * hour / HOURS_PER_DAY).to(dtype)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


class ActionEncoder(nn.Module):
    """action_vector (B, K, 11) + timestamps (B, K) -> a_t (B, K, dim). Xem docstring module."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
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

    def forward(
        self,
        action_vectors: torch.Tensor,
        hist_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = action_vectors[..., self.m_idx]
        s = action_vectors[..., self.s_idx]
        s = torch.where(self.has_s, s, torch.ones_like(s))

        weight = m * s
        a = weight @ self.type_embedding

        if hist_timestamps is not None:
            a = a + self.hour_proj(hour_of_day(hist_timestamps, a.dtype))
        return a


def _demo() -> None:
    """ponytail: check nhỏ nhất bắt được shape/NaN cơ bản."""
    enc = ActionEncoder(dim=8)
    av = torch.rand(2, 3, NUM_ACTION_DIMS)
    out = enc(av)
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()
    print("action_encoder self-check OK")


if __name__ == "__main__":
    _demo()
