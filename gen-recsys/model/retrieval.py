"""Retrieval logit + sampled-softmax loss.

Công thức (đã đơn giản hóa 2026-09-13 — xem result.md "CHECKLIST CUỐI CÙNG"):
    logit(u,i) = e_u_final · e_i_final / √d − log_Q(i)
    P(i | u)   = softmax_i( logit(u,i) / T_base )

T_base CỐ ĐỊNH — đã BỎ HẲN nhiệt độ biến thiên T(u,i) = T_base/(user_weight·item_weight)
từng dùng trước đây. Lý do bỏ (không phải sửa thứ tự phép toán): review.md L2 — hướng
T(u,i) tăng khi cặp càng cold đi NGƯỢC lý thuyết hardness-aware (Wang & Liu, CVPR 2021:
temperature lớn làm gradient phẳng đều, giảm tính phân biệt — đúng lúc nhóm cold cần
gradient sắc nhất). Đồng thời việc này tự động sửa luôn lỗi cũ (log_Q bị nhân với 1/T
biến thiên theo candidate, phá vỡ ý nghĩa importance-sampling correction — review.md L1):
với T_base cố định, (logit − log_Q)/T_base và logit/T_base − log_Q chỉ khác nhau hằng số,
không đổi rank.

log_Q(i): sampled-softmax correction cho negative sampling không đều (mục đích thống kê
thuần túy, KHÔNG liên quan đến item_weight/user_weight).

E_i đã "biết" item_weight từ trước (qua gate g_i và category_content_shrunk trong
item_embedding.py) — logit không cần cộng thêm số hạng maturity nào nữa, tránh
double-counting (xem item_embedding.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    def __init__(self, dim: int, t_base: float = 0.1):
        super().__init__()
        self.dim = dim
        self.t_base = t_base

    def scaled_logit(
        self,
        e_u_final: torch.Tensor,  # (B, dim) — user representation cuối cùng (sau gate 3 thành phần)
        candidate_embeddings: torch.Tensor,  # (B, C, dim) — e_i_final của [positive, negatives...]
        log_q: torch.Tensor,  # (B, C) — log_Q(i), sampled-softmax correction mỗi candidate
    ) -> torch.Tensor:
        """logit(u,i)/T_base − log_q — dùng CHUNG cho cross_entropy lúc train (forward) VÀ
        ranking lúc eval (train.py evaluate()), giữ đúng thứ tự rank giữa 2 nơi."""
        logit = torch.einsum("bd,bcd->bc", e_u_final, candidate_embeddings) / math.sqrt(self.dim)  # (B, C)
        return logit / self.t_base - log_q

    def eval_logit(
        self,
        e_u_final: torch.Tensor,  # (B, dim)
        candidate_embeddings: torch.Tensor,  # (B, C, dim)
    ) -> torch.Tensor:
        """Logit dùng cho XẾP HẠNG lúc eval — dot-product THUẦN, KHÔNG trừ log_q.

        [SỬA 2026-09-15] Trước đây eval dùng scaled_logit() (có trừ log_q) và cho ra
        recall@5 = 1.0000 ở ô item-cold — con số vô lý đã lộ ra bug này.

        VÌ SAO PHẢI BỎ log_q Ở EVAL. log_q là importance-sampling correction (Bengio &
        Senécal 2003): lúc TRAIN, negative được rút theo tần suất nên item phổ biến bị lấy
        mẫu quá nhiều; trừ log_q khử đúng thiên lệch đó để sampled-softmax xấp xỉ full
        softmax. Điều kiện để phép khử hợp lệ: MỌI candidate được so sánh phải đến từ CÙNG
        phân phối lấy mẫu.

        Lúc EVAL điều kiện đó KHÔNG thỏa: positive là item THẬT của user (rút từ phân phối
        dữ liệu), negative là item rút theo tần suất. Hai nguồn khác nhau -> trừ log_q
        không còn là hiệu chỉnh mà thành CỘNG ĐIỂM có hệ thống cho item hiếm, vì item hiếm
        có log_q thấp. Đo thật: log_q ≈ −11.92 cho item cold vs ≈ −8.02 cho item warm, tức
        cold được cộng thêm ~3.90 — trong khi tín hiệu dot-product thật chỉ dao động ±1.
        Thứ hạng khi đó do ĐỘ HIẾM quyết định, không phải do model. Đó là nguồn gốc của
        recall@5 = 1.0 ở ô item-cold: không phải model giỏi, mà là công thức tự thưởng.

        Trùng khớp với serving thật: lúc phục vụ, model xếp hạng bằng dot-product thuần
        trên catalog, không có khái niệm "phân phối lấy mẫu negative". Eval phải đo đúng
        thứ sẽ chạy lúc serving.

        T_base cũng bỏ luôn: chia một hằng số dương cho MỌI candidate không đổi thứ hạng,
        giữ lại chỉ gây hiểu nhầm là nó có vai trò.

        log_q VẪN GIỮ NGUYÊN trong forward()/forward_sequence() (lúc train) — ở đó điều
        kiện cùng phân phối được thỏa và nó đúng, cần thiết."""
        return torch.einsum("bd,bcd->bc", e_u_final, candidate_embeddings) / math.sqrt(self.dim)

    def forward(
        self,
        e_u_final: torch.Tensor,  # (B, dim)
        candidate_embeddings: torch.Tensor,  # (B, C, dim)
        log_q: torch.Tensor,  # (B, C)
        positive_idx: torch.Tensor,  # (B,) int64 — vị trí của positive trong trục C (thường 0)
    ) -> torch.Tensor:
        scaled_logit = self.scaled_logit(e_u_final, candidate_embeddings, log_q)
        return F.cross_entropy(scaled_logit, positive_idx)

    def forward_sequence(
        self,
        pred: torch.Tensor,  # (B, K-1, dim) — hidden tại Φ_t, dùng dự đoán item t+1
        positive_e_i: torch.Tensor,  # (B, K-1, dim) — e_i_final THẬT của item t+1
        neg_e_i: torch.Tensor,  # (B, C, dim) — negative dùng CHUNG cho mọi vị trí
        positive_log_q: torch.Tensor,  # (B, K-1)
        neg_log_q: torch.Tensor,  # (B, C)
        valid_mask: torch.Tensor,  # (B, K-1) bool/float — 1 = vị trí thật, 0 = padding
    ) -> torch.Tensor:
        """[THÊM 2026-09-14] Loss TỰ HỒI QUY TOÀN CHUỖI — dự đoán item kế tiếp tại MỌI vị
        trí, đúng HSTU ("a sequential transduction task maps this input sequence to the
        output tokens y_0, y_1, …, y_{n-1}", arXiv 2402.17152).

        Khác forward() cũ: forward() chỉ chấm 1 vị trí cuối -> 1 điểm giám sát/sample.
        forward_sequence() cho K-1 điểm -> tín hiệu học tăng ~255× từ CÙNG 1 forward pass
        (hidden tại mọi vị trí vốn đã tính sẵn, thiết kế cũ vứt đi 255/256).

        Negative dùng CHUNG cho mọi vị trí trong batch — (B,C,dim) thay vì (B,K,C,dim),
        tiết kiệm K lần bộ nhớ; đủ tốt vì negative lấy ngẫu nhiên theo tần suất, không phụ
        thuộc vị trí (chuẩn sampled-softmax).

        valid_mask loại vị trí padding khỏi loss — padding (video_id=0) vẫn chạy qua decoder
        nhưng KHÔNG đóng góp gradient, tránh dạy model dự đoán token giả."""
        pos_logit = (pred * positive_e_i).sum(-1) / math.sqrt(self.dim)  # (B, K-1)
        pos_logit = pos_logit / self.t_base - positive_log_q

        neg_logit = torch.einsum("bkd,bcd->bkc", pred, neg_e_i) / math.sqrt(self.dim)  # (B, K-1, C)
        neg_logit = neg_logit / self.t_base - neg_log_q.unsqueeze(1)

        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)  # (B, K-1, 1+C)
        target = torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).view_as(valid_mask)
        valid = valid_mask.float()
        return (loss * valid).sum() / valid.sum().clamp(min=1)
