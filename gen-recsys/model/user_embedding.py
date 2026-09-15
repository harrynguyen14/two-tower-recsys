"""e_profile — nén đặc trưng TĨNH của user (register_days + onehot_feat0-17, xem
build_user_static.py) thành 1 vector dim chiều, dùng làm TOKEN 0 prepend vào chuỗi
(xem sequence_model.py forward, tham số profile_embedding).

[VIẾT LẠI 2026-09-14] Module này TRƯỚC ĐÂY là "user gate": trộn e_behavior/e_profile/
e_short bằng 3 gate học được, rồi trả e_u_final cho loss. Đã BỎ HẲN cơ chế gate. Lý do —
không phải dọn dẹp, mà sửa bug + sửa ngữ nghĩa:

  1. BUG: 3 gate + static_gmu KHÔNG NHẬN GRADIENT. Sau khi chuyển sang loss tự hồi quy
     toàn chuỗi (retrieval.py forward_sequence), loss chấm THẲNG trên `hidden` của
     decoder tại mọi vị trí, còn e_u_final chỉ được dùng trong evaluate(). Nghĩa là toàn
     bộ nhánh này khởi tạo ngẫu nhiên rồi đem xếp hạng lúc eval — cơ chế cold-user, thứ
     đứng ở trung tâm câu hỏi nghiên cứu, thực chất là NHIỄU TRẮNG. Prepend token sửa
     triệt để: e_profile nằm trong chuỗi, gradient chảy về static_gmu từ MỌI vị trí của
     loss (K điểm/chuỗi, không phải 1).

  2. NGỮ NGHĨA: gate g_u trả lời "user này cold hay warm, tin profile bao nhiêu phần" —
     MỘT câu trả lời cho cả chuỗi (user_weight cũ là 1 scalar/chuỗi). Attention trả lời
     câu KHÁC: "tại vị trí i, profile đáng chú ý bao nhiêu so với 2i token lịch sử đang
     có" — riêng cho từng i. Tại i=1 model cần profile; tại i=180 profile gần như thừa.
     Gate không phân biệt được hai vị trí đó. Prepend biến "cold user" từ thuộc tính TĨNH
     của user thành đại lượng BIẾN THIÊN trong chuỗi — đúng hơn, vì mọi user đều cold ở
     token đầu của chính mình.

  3. e_short (ShortTermAttentionPool cũ, 5 token cuối, query cố định) cũng BỎ: attention
     trên 2i token lịch sử đã bao gồm 5 token cuối và tự học trọng số gần/xa tốt hơn một
     pool với query cố định + cửa sổ cứng. Trục ngắn/dài hạn giờ do δ_h·log(1+i-j)
     (relative position bias mỗi head 1 độ dốc học được) + u_i per-position đảm nhiệm —
     xem confidence_attention.py.

Giữ e_profile trong module riêng (không nhét thẳng vào train.py) vì nó vẫn là 1 khối
có cấu trúc: 18 bảng embedding onehot + register_days numeric, fuse qua GMU.

register_days: numeric (log1p + z-score, đã chuẩn hóa ở build_user_static.py) — đưa thẳng
vào GMU dạng vector 1 chiều, không cần embedding. onehot_feat0-17: mỗi field 1 nn.Embedding
riêng (range mỗi field khác nhau rất nhiều — 2 đến 1471 category, đã đo trực tiếp — xem
build_user_static.py, KHÔNG dùng chung 1 bảng).

`use_profile_token=False` tắt hẳn nhánh này (forward trả None, chuỗi không có token 0) —
dùng để ablate "có profile token vs không".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

NUM_ONEHOT_FIELDS = 18  # onehot_feat0-17, xem schema.py USER_STATIC_ONEHOT_FEATS


class UserProfileConfig:
    def __init__(
        self,
        onehot_num_categories: list[int],  # (18,) — từ onehot_num_categories.npy, xem build_user_static.py
        dim: int = 64,
        onehot_embed_dim: int = 8,
        use_profile_token: bool = True,  # False = không prepend token profile (ablation)
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
            return  # tắt cứng — forward() trả None, không khởi tạo tham số nào

        self.onehot_embeddings = nn.ModuleList([
            nn.Embedding(n, config.onehot_embed_dim) for n in config.onehot_num_categories
        ])
        gmu_in_dims = {f"onehot_{i}": config.onehot_embed_dim for i in range(NUM_ONEHOT_FIELDS)}
        gmu_in_dims["register_days"] = 1
        self.static_gmu = GMU(gmu_in_dims, dim=config.dim)

    def forward(
        self,
        onehot: torch.Tensor,  # (B, 18) int64 — onehot_feat0-17 đã factorize
        register_days: torch.Tensor,  # (B,) float32 — đã chuẩn hóa
    ) -> torch.Tensor | None:
        """Trả (B, dim) = e_profile, hoặc None nếu use_profile_token=False (ablation).
        Caller truyền thẳng kết quả vào SequenceModel.forward(profile_embedding=...) —
        None cũng hợp lệ ở đó (chuỗi không prepend token nào)."""
        if not self.config.use_profile_token:
            return None

        gmu_inputs = {
            f"onehot_{i}": self.onehot_embeddings[i](onehot[:, i]) for i in range(NUM_ONEHOT_FIELDS)
        }
        gmu_inputs["register_days"] = register_days.unsqueeze(-1)
        return self.static_gmu(gmu_inputs)  # (B, dim)
