"""Mã hoá action_vector (11 chiều) + giờ-trong-ngày thành 1 token a_t."""

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

    def __init__(self, dim: int, use_hour: bool = True):
        super().__init__()
        self.dim = dim
        self.use_hour = use_hour
        self.num_types = len(ACTION_TYPES)

        self.type_embedding = nn.Parameter(torch.empty(self.num_types, dim))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

        m_idx = [m for _, m, _ in ACTION_TYPES]
        s_idx = [s if s is not None else 0 for _, _, s in ACTION_TYPES]
        has_s = [s is not None for _, _, s in ACTION_TYPES]
        self.register_buffer("m_idx", torch.tensor(m_idx, dtype=torch.long), persistent=False)
        self.register_buffer("s_idx", torch.tensor(s_idx, dtype=torch.long), persistent=False)
        self.register_buffer("has_s", torch.tensor(has_s, dtype=torch.bool), persistent=False)

        if use_hour:
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

        if self.use_hour and hist_timestamps is not None:
            a = a + self.hour_proj(hour_of_day(hist_timestamps, a.dtype))
        return a


