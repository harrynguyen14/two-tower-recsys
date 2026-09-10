"""Gated Multimodal Unit (GMU) — tổng quát, dùng cho content_branch của item embedding
module (xem idea.md mục 4.5, "Đánh đổi cần lưu ý khi chuyển từ Amazon sang KuaiRand").

Thiết kế: mỗi modality có 1 encoder riêng chiếu về cùng `dim`, rồi gate học được (dựa
trên chính các projection) quyết định trọng số mỗi modality có mặt. Modality thiếu ở 1
số dòng dữ liệu (vd. tương lai thêm text/image optional) truyền qua `mask` — dòng có
mask=0 bị loại khỏi softmax gate (không ảnh hưởng renormalize), không cần điền giá trị
giả 0 rồi để gate tự học "bỏ qua" (rủi ro nhiễu gradient không cần thiết).

Hiện tại (2026-09-10) chỉ có 2 modality bắt buộc luôn có mặt cho mọi item (không modality
nào optional): categorical (category/author/music/video_type/music_type) và numeric
(stat features). Text (caption) sẽ thêm sau — chỉ cần thêm 1 entry vào dict đầu vào của
ItemContentGMU, KHÔNG cần sửa module này.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Chiếu 1 modality (đã ở dạng vector) về `dim` chung qua 1 MLP nhỏ."""

    def __init__(self, in_dim: int, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GMU(nn.Module):
    """Gated fusion tổng quát cho N modality.

    forward nhận `inputs: dict[name, Tensor(B, in_dim_name)]` và `masks: dict[name, Tensor(B,)]
    | None` (None = modality luôn có mặt, không cần mask). Trả về `(B, dim)`.

    Gate: với mỗi sample, so trọng số giữa các modality CÓ MẶT (mask=1) qua softmax trên
    1 điểm số học được từ chính projection của modality đó — modality vắng mặt (mask=0)
    bị set -inf trước softmax nên không nhận trọng số (renormalize tự động qua softmax).
    """

    def __init__(self, in_dims: dict[str, int], dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.names = list(in_dims.keys())
        self.dim = dim
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(in_dim, dim, hidden_dim) for name, in_dim in in_dims.items()
        })
        self.gate_score = nn.ModuleDict({name: nn.Linear(dim, 1) for name in self.names})

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        masks = masks or {}
        projections = []  # mỗi phần tử (B, dim)
        scores = []  # mỗi phần tử (B, 1)
        for name in self.names:
            proj = self.encoders[name](inputs[name])
            projections.append(proj)
            score = self.gate_score[name](proj)
            if name in masks:
                mask = masks[name].unsqueeze(-1)  # (B, 1), 1=có mặt, 0=vắng mặt
                score = score.masked_fill(mask == 0, float("-inf"))
            scores.append(score)

        stacked_proj = torch.stack(projections, dim=1)  # (B, num_modalities, dim)
        stacked_score = torch.cat(scores, dim=1)  # (B, num_modalities)
        gate = torch.softmax(stacked_score, dim=1).unsqueeze(-1)  # (B, num_modalities, 1)
        return (gate * stacked_proj).sum(dim=1)  # (B, dim)
