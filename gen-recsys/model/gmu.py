"""Gated Multimodal Unit (GMU) — tổng quát, dùng cho content_branch của item embedding"""

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
    """Gated fusion tổng quát cho N modality."""

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
        projections = []
        scores = []
        for name in self.names:
            proj = self.encoders[name](inputs[name])
            projections.append(proj)
            score = self.gate_score[name](proj)
            if name in masks:
                mask = masks[name].unsqueeze(-1)
                score = score.masked_fill(mask == 0, float("-inf"))
            scores.append(score)

        stacked_proj = torch.stack(projections, dim=1)
        stacked_score = torch.cat(scores, dim=1)
        gate = torch.softmax(stacked_score, dim=1).unsqueeze(-1)
        return (gate * stacked_proj).sum(dim=1)
