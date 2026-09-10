"""BCE Loss cho Multi-Task/Multi-Action Ranking — bài toán KHÁC retrieval (xem
retrieval.py và idea.md mục 4.5 điểm 3, "Chốt loss nền"). Giả định đã có 1 shortlist
candidate nhỏ (không phải toàn catalog), chấm điểm NHỊ PHÂN riêng từng action
(click/like/share/comment/...) — trả lời "có nên đề xuất/user thích không", KHÁC câu
hỏi retrieval "item nào sẽ được hiển thị tiếp theo".

KHÔNG gắn mat_u/mat_i vào đây (đã CHỐT) — ranking giả định candidate đã được chọn rồi,
tách biệt khỏi câu hỏi "có nên đưa item cold vào xem xét không" (thuộc về retrieval loss
+ tầng candidate-filtering, xem idea.md "TRẠNG THÁI DỰ ÁN" quyết định #5).

Multi-action: mỗi action trong action_vector (is_click/like/follow/comment/forward/hate,
long_view, is_profile_enter — các field NHỊ PHÂN thật, KHÔNG gồm play_ratio/
profile_stay_time_norm/comment_stay_time_norm vì đó là giá trị liên tục [0,1], không phải
nhãn phân loại) có 1 đầu ra BCE riêng — multi-task, KHÔNG trộn chung 1 nhãn.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Field NHỊ PHÂN thật trong action_vector — dùng làm nhãn multi-task BCE. Loại play_ratio/
# profile_stay_time_norm/comment_stay_time_norm (giá trị liên tục, không phải nhãn 0/1).
BINARY_ACTION_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "is_profile_enter",
]
NUM_BINARY_ACTIONS = len(BINARY_ACTION_FIELDS)  # 8


class RankingLoss(nn.Module):
    """1 head tuyến tính riêng cho mỗi action nhị phân, chấm trên cặp (h_u, E_i) của 1
    candidate cụ thể đã có trong shortlist (KHÔNG so với toàn catalog như retrieval)."""

    def __init__(self, dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            action: nn.Linear(2 * dim, 1) for action in BINARY_ACTION_FIELDS
        })

    def forward(
        self,
        h_u: torch.Tensor,  # (B, dim) — hidden state decoder
        e_i: torch.Tensor,  # (B, dim) — E_i của candidate đang chấm điểm
        action_labels: torch.Tensor,  # (B, NUM_BINARY_ACTIONS) float32, 0/1 — nhãn thật
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pair = torch.cat([h_u, e_i], dim=-1)  # (B, 2*dim)

        per_action_loss = {}
        total = 0.0
        for idx, action in enumerate(BINARY_ACTION_FIELDS):
            logit = self.heads[action](pair).squeeze(-1)  # (B,)
            loss = F.binary_cross_entropy_with_logits(logit, action_labels[:, idx])
            per_action_loss[action] = loss
            total = total + loss

        return total / NUM_BINARY_ACTIONS, per_action_loss
