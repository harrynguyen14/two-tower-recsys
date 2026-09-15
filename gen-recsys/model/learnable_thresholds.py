"""τ_u, τ_i, τ_c — ngưỡng độ trưởng thành trong tanh(N/τ), HỌC ĐƯỢC (nn.Parameter) nhưng
KHỞI TẠO từ thống kê thật của dataset (xem idea.md "Việc CẦN LÀM TIẾP" #1).

[SỬA 2026-09-13] Đổi tên phương thức cho tường minh (xem result.md "CHECKLIST CUỐI CÙNG"):
    mat_u        -> user_weight
    mat_i        -> item_weight
    conf_content -> category_confidence

Khởi tạo (đo lại cho KuaiRand-Pure, xem result.md 2026-09-13 — GIÁ TRỊ CŨ đo trên
KuaiRand-27K không áp dụng được cho Pure, N_u/N_i có phân phối khác hẳn):
    τ_u = 35.0     — median N_u tại thời điểm sample (đúng phương pháp idea.md mục 0)
    τ_i = 599.3    — mean N_i tại thời điểm sample (median N_i=238 vẫn dùng mean theo
                     đúng lý do gốc: tránh tanh bão hòa sớm ở vùng N nhỏ quan trọng nhất)
    τ_c = 53486.1  — mean N_category tại thời điểm sample, cùng phương pháp với τ_i

CẢNH BÁO rủi ro "threshold-collapse" (xem idea.md/review.md): gradient descent có thể kéo
τ về giá trị tối ưu loss TRUNG BÌNH (bị chi phối bởi nhóm warm, đông hơn), vô tình "bỏ rơi"
nhóm cold ít ảnh hưởng loss tổng — BẮT BUỘC giám sát riêng: vẽ đường giá trị τ qua epoch +
so Recall/NDCG tách riêng theo nhóm cold (xem get_tau_snapshot()). Cơ chế chủ động chống
threshold-collapse (class-balanced loss, anchor loss) đã CÂN NHẮC nhưng HOÃN — chờ đo
baseline này có thật sự cần không trước khi thêm (xem result.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

TAU_U_INIT = 35.0      # median N_u tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13
TAU_I_INIT = 599.3     # mean N_i tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13
TAU_C_INIT = 53486.1   # mean N_category tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13


class LearnableThresholds(nn.Module):
    """Bọc τ_u/τ_i/τ_c thành nn.Parameter, cung cấp user_weight/item_weight/
    category_confidence tiện dụng và snapshot giá trị để giám sát threshold-collapse qua
    epoch."""

    def __init__(self, tau_u_init: float = TAU_U_INIT, tau_i_init: float = TAU_I_INIT, tau_c_init: float = TAU_C_INIT):
        super().__init__()
        self.tau_u = nn.Parameter(torch.tensor(tau_u_init))
        self.tau_i = nn.Parameter(torch.tensor(tau_i_init))
        self.tau_c = nn.Parameter(torch.tensor(tau_c_init))

    def _safe_tau(self, tau: torch.Tensor) -> torch.Tensor:
        # τ có thể bị gradient kéo về <=0 (vô nghĩa, N/τ đổi dấu) — clamp mềm giữ dương.
        return tau.clamp(min=1e-3)

    def user_weight(self, n_u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_u / self._safe_tau(self.tau_u))

    def item_weight(self, n_i: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_i / self._safe_tau(self.tau_i))

    def category_confidence(self, n_category: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_category / self._safe_tau(self.tau_c))

    def get_tau_snapshot(self) -> dict[str, float]:
        """Giá trị τ hiện tại — gọi mỗi epoch để log/vẽ đường giá trị, phát hiện sớm
        threshold-collapse (xem cảnh báo ở docstring module)."""
        return {
            "tau_u": self.tau_u.item(),
            "tau_i": self.tau_i.item(),
            "tau_c": self.tau_c.item(),
        }
