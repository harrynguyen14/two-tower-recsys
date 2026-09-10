"""τ_u, τ_i, τ_c — ngưỡng độ trưởng thành trong tanh(N/τ), HỌC ĐƯỢC (nn.Parameter) nhưng
KHỞI TẠO từ thống kê thật của dataset (xem idea.md "Việc CẦN LÀM TIẾP" #1).

Khởi tạo:
    τ_u = 48   — median N_u/user (đo ở idea.md mục 0, KuaiRand-27K)
    τ_i = 10   — mean N_i/item (đo trực tiếp — median N_i=1 quá nhỏ, làm tanh(N_i/τ_i)
                 bão hòa gần như ngay lập tức mất hết độ phân giải ở vùng N nhỏ quan trọng
                 nhất, xem idea.md build_interactions.py TAU_I_INIT)
    τ_c        — CHƯA đo trực tiếp trong idea.md, khởi tạo tạm = τ_i (cùng bản chất "đếm
                 số thực thể đã warm", xem đo thực tế ở _measure_tau_c_init khi có dữ liệu)

CẢNH BÁO rủi ro "threshold-collapse" (xem idea.md): gradient descent có thể kéo τ về giá
trị tối ưu loss TRUNG BÌNH (bị chi phối bởi nhóm warm, đông hơn), vô tình "bỏ rơi" nhóm
cold ít ảnh hưởng loss tổng — BẮT BUỘC giám sát riêng: vẽ đường giá trị τ qua epoch + so
Recall/NDCG tách riêng theo nhóm cold (xem get_tau_snapshot()).
"""

from __future__ import annotations

import torch
import torch.nn as nn

TAU_U_INIT = 48.0  # median N_u/user, đo ở idea.md mục 0
TAU_I_INIT = 10.0  # mean N_i/item, đo trực tiếp 2026-09-09 (median N_i=1 quá nhỏ, không dùng)
TAU_C_INIT = 10.0  # chưa đo riêng — tạm dùng cùng giá trị τ_i (cùng bản chất đếm lũy kế)


class LearnableThresholds(nn.Module):
    """Bọc τ_u/τ_i/τ_c thành nn.Parameter, cung cấp mat_u/mat_i/conf_content tiện dụng và
    snapshot giá trị để giám sát threshold-collapse qua epoch."""

    def __init__(self, tau_u_init: float = TAU_U_INIT, tau_i_init: float = TAU_I_INIT, tau_c_init: float = TAU_C_INIT):
        super().__init__()
        self.tau_u = nn.Parameter(torch.tensor(tau_u_init))
        self.tau_i = nn.Parameter(torch.tensor(tau_i_init))
        self.tau_c = nn.Parameter(torch.tensor(tau_c_init))

    def _safe_tau(self, tau: torch.Tensor) -> torch.Tensor:
        # τ có thể bị gradient kéo về <=0 (vô nghĩa, N/τ đổi dấu) — clamp mềm giữ dương.
        return tau.clamp(min=1e-3)

    def mat_u(self, n_u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_u / self._safe_tau(self.tau_u))

    def mat_i(self, n_i: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_i / self._safe_tau(self.tau_i))

    def conf_content(self, n_category: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_category / self._safe_tau(self.tau_c))

    def get_tau_snapshot(self) -> dict[str, float]:
        """Giá trị τ hiện tại — gọi mỗi epoch để log/vẽ đường giá trị, phát hiện sớm
        threshold-collapse (xem cảnh báo ở docstring module)."""
        return {
            "tau_u": self.tau_u.item(),
            "tau_i": self.tau_i.item(),
            "tau_c": self.tau_c.item(),
        }
