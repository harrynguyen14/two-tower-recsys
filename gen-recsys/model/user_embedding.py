"""e_profile — nén đặc trưng TĨNH của user (register_days + onehot_feat0-17, xem"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

NUM_ONEHOT_FIELDS = 18


class UserProfileConfig:
    def __init__(
        self,
        onehot_num_categories: list[int],
        dim: int = 64,
        onehot_embed_dim: int = 8,
        use_profile_token: bool = True,
    ):
        assert len(onehot_num_categories) == NUM_ONEHOT_FIELDS
        self.onehot_num_categories = onehot_num_categories
        self.dim = dim
        self.onehot_embed_dim = onehot_embed_dim
        self.use_profile_token = use_profile_token


class UserProfileEmbedding(nn.Module):
    def __init__(self, config: UserProfileConfig):
        super().__init__()
        self.config = config

        if not config.use_profile_token:
            return

        self.onehot_embeddings = nn.ModuleList([
            nn.Embedding(n, config.onehot_embed_dim) for n in config.onehot_num_categories
        ])
        gmu_in_dims = {f"onehot_{i}": config.onehot_embed_dim for i in range(NUM_ONEHOT_FIELDS)}
        gmu_in_dims["register_days"] = 1
        self.static_gmu = GMU(gmu_in_dims, dim=config.dim)

    def forward(
        self,
        onehot: torch.Tensor,
        register_days: torch.Tensor,
    ) -> torch.Tensor | None:
        """Trả (B, dim) = e_profile, hoặc None nếu use_profile_token=False (ablation)."""
        if not self.config.use_profile_token:
            return None

        gmu_inputs = {
            f"onehot_{i}": self.onehot_embeddings[i](onehot[:, i]) for i in range(NUM_ONEHOT_FIELDS)
        }
        gmu_inputs["register_days"] = register_days.unsqueeze(-1)
        return self.static_gmu(gmu_inputs)
