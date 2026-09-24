"""τ_u, τ_i — ngưỡng độ trưởng thành trong công thức maturity, HỌC ĐƯỢC (nn.Parameter)."""

from __future__ import annotations

import torch
import torch.nn as nn

TAU_U_INIT = 35.0
TAU_I_INIT = 599.3


class LearnableThresholds(nn.Module):
    """Bọc τ_u/τ_i/τ_c thành nn.Parameter, cung cấp user_weight/item_weight/"""

    def __init__(self, tau_u_init: float = TAU_U_INIT, tau_i_init: float = TAU_I_INIT):
        super().__init__()
        self.tau_u = nn.Parameter(torch.tensor(tau_u_init))
        self.tau_i = nn.Parameter(torch.tensor(tau_i_init))

    def _safe_tau(self, tau: torch.Tensor) -> torch.Tensor:
        return tau.clamp(min=1e-3)

    def user_weight(self, n_u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_u / self._safe_tau(self.tau_u))

    def log_user_maturity(self, n_u: torch.Tensor) -> torch.Tensor:
        """log1p(N_u) - log1p(tau_u) — thay cho log(user_weight) trong attention bias."""
        return torch.log1p(n_u) - torch.log1p(self._safe_tau(self.tau_u))

    def item_weight(self, n_i: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_i / self._safe_tau(self.tau_i))


    def get_tau_snapshot(self) -> dict[str, float]:
        """Giá trị τ hiện tại — gọi mỗi epoch để log/vẽ đường giá trị, phát hiện sớm"""
        return {
            "tau_u": self.tau_u.item(),
            "tau_i": self.tau_i.item(),
        }
