"""BCE Loss cho Multi-Task/Multi-Action Ranking — bài toán KHÁC retrieval (xem
retrieval.py và idea.md mục 4.5 điểm 3, "Chốt loss nền"). Trả lời "có nên đề xuất/user
thích không" cho 1 candidate cụ thể (thường là label thật), KHÁC câu hỏi retrieval "item
nào sẽ được hiển thị tiếp theo trong toàn catalog".

[SỬA 2026-09-13] THIẾT KẾ LẠI theo target-aware cross-attention — đúng cách HSTU thật làm
ranking (xem result.md "CHECKLIST CUỐI CÙNG", đọc trực tiếp mô tả + code
facebookresearch/generative-recommenders). KHÁC thiết kế cũ (chấm 1 candidate qua
concat([h_u, e_i]) HOÀN TOÀN TÁCH RỜI khỏi decoder):

    Chuỗi:  [x_1, ..., x_K, x_candidate]     -- candidate NỐI vào CUỐI chuỗi lịch sử
    Chạy decoder 1 LẦN cho chuỗi K+1 (causal attention tự động cho candidate "thấy" toàn
    bộ lịch sử — đây chính là target-aware cross-attention, không cần module attention
    riêng, chỉ cần đặt đúng vị trí trong chuỗi causal có sẵn)
    h_candidate = output[:, -1, :]           -- vị trí K+1 = vị trí candidate

    x_candidate = e_i_final(candidate) + action_proj(label_action) + position(K)
    (label_action = action THẬT tại vị trí label, dùng làm INPUT — nhãn để tính BCE loss
    cũng CHÍNH LÀ label_action đó: multi-task dự đoán "action nào đã thực sự xảy ra")

Việc NỐI candidate vào chuỗi + chạy decoder thuộc về train.py (nơi có quyền truy cập
SequenceModel/decoder) — module này CHỈ còn phần MLP đa nhiệm nhận sẵn h_candidate đã
tính xong, giữ đúng nguyên tắc "mỗi module 1 việc" (xem decoder.py/sequence_model.py).

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
    """1 head tuyến tính riêng cho mỗi action nhị phân, chấm trên h_candidate — hidden
    state của decoder TẠI ĐÚNG VỊ TRÍ candidate sau khi đã nối vào chuỗi (xem docstring
    module + train.py cho phần nối chuỗi/chạy decoder)."""

    def __init__(self, dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            action: nn.Linear(dim, 1) for action in BINARY_ACTION_FIELDS
        })

    def forward(
        self,
        h_candidate: torch.Tensor,  # (B, dim) — hidden state decoder TẠI VỊ TRÍ candidate (đã qua target-aware attention)
        action_labels: torch.Tensor,  # (B, NUM_BINARY_ACTIONS) float32, 0/1 — nhãn thật
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_action_loss = {}
        total = 0.0
        for idx, action in enumerate(BINARY_ACTION_FIELDS):
            logit = self.heads[action](h_candidate).squeeze(-1)  # (B,)
            loss = F.binary_cross_entropy_with_logits(logit, action_labels[:, idx])
            per_action_loss[action] = loss
            total = total + loss

        return total / NUM_BINARY_ACTIONS, per_action_loss

    def forward_sequence(
        self,
        h_items: torch.Tensor,  # (B, K, dim) — hidden tại Φ_t (CHƯA thấy a_t)
        action_labels: torch.Tensor,  # (B, K, NUM_BINARY_ACTIONS) float32 0/1 — a_t thật
        valid_mask: torch.Tensor,  # (B, K) bool/float — 1 = token thật
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """[THÊM 2026-09-14] Ranking TOÀN CHUỖI: p(a_t | Φ_0,a_0,…,Φ_t) tại MỌI vị trí —
        đúng công thức HSTU cho ranking (arXiv 2402.17152: "interleaving items and actions
        enables the ranking task to be formulated as p(a_{i+1}|Φ_0,a_0,…,Φ_{i+1})").

        An toàn về leak nhờ CẤU TRÚC, không nhờ hack: trong chuỗi xen kẽ, hidden tại Φ_t
        thấy (Φ_0..Φ_t, a_0..a_{t-1}) nhưng KHÔNG thấy a_t (causal mask) — đã kiểm chứng
        bằng test. Thiết kế K+1 cũ nối label_action vào chuỗi rồi dùng chính nó làm nhãn
        => model học được hàm đồng nhất; xen kẽ loại bỏ hẳn vấn đề đó.

        Tín hiệu ranking tăng từ 1 lên K điểm/sample, cùng 1 forward pass."""
        valid = valid_mask.float()
        denom = valid.sum().clamp(min=1)

        per_action_loss = {}
        total = 0.0
        for idx, action in enumerate(BINARY_ACTION_FIELDS):
            logit = self.heads[action](h_items).squeeze(-1)  # (B, K)
            loss = F.binary_cross_entropy_with_logits(
                logit, action_labels[..., idx], reduction="none"
            )  # (B, K)
            loss = (loss * valid).sum() / denom
            per_action_loss[action] = loss
            total = total + loss

        return total / NUM_BINARY_ACTIONS, per_action_loss
