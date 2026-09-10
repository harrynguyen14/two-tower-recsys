"""Item embedding module — biến 1 item thành 1 embedding E_i, dùng làm token trong chuỗi
user hoặc làm candidate lúc decode (KHÔNG phải "item tower" — xem lưu ý thuật ngữ ở
idea.md mục 4.5 đầu mục "Đề xuất thiết kế cụ thể").

Công thức đã chốt (idea.md mục 4.5 điểm 2 + điểm 3):
    conf_content(i)       = tanh(N_category(i) / τ_c)
    content_branch_shrunk = content_branch(i) · conf_content(i)
    g_i                   = σ(MLP([mat_i, collaborative_branch(i), content_branch(i)]))
    E_i                   = g_i · collaborative_branch(i) + (1-g_i) · content_branch_shrunk

collaborative_branch: nn.Embedding(num_items, dim) học từ đầu theo video_id — thuần ID
embedding, KHÔNG dùng feature nào khác (đã chốt 2026-09-10). Do catalog 32,038,725 item
(dim=64 -> ~8.2GB), cần chia model parallelism (item_id % 2 -> GPU0/GPU1, xem idea.md
mục "TRẠNG THÁI DỰ ÁN" quyết định #1) — CHƯA làm ở module này (single-GPU trước, xem
TODO ở EmbeddingConfig), thêm khi có 2 GPU thật để test.

content_branch: qua GMU (gmu.py) fuse các nhánh tĩnh của item — categorical (category
1 cấp cũ + category 4 cấp mới + author + music + video_type + music_type) và numeric
(8 feature thống kê đã chuẩn hóa). author_idx/music_idx CŨNG nằm trong content_branch
(đã CHỐT 2026-09-10 — xem hỏi-đáp: "author/music là 1 nhánh trong content_branch qua
GMU", KHÔNG tách riêng thành số hạng thứ 3 trong E_i, giữ đúng công thức 2 nhánh).

Input: item_static.npy (Pass 3, xem build_item_static.py) — mỗi field int32 đã factorize
sẵn thành index liên tục [0, n), category_id/video_type_id/music_type_id/cat_l1-l4_id
dùng nn.Embedding riêng từng field (KHÔNG concat one-hot — số category nhỏ nhưng tách
riêng để mỗi field có không gian embedding riêng, không ép chung 1 bảng).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

# Field categorical trong item_static.npy dùng embedding riêng (tên field -> tên tham số
# num_embeddings tương ứng khi khởi tạo ItemEmbeddingConfig) — KHÔNG gồm author_idx/
# music_idx (embedding lớn, tách riêng dict `id_embeddings` vì kích thước rất khác biệt
# so với category nhỏ, dù cùng "categorical" về bản chất).
CATEGORICAL_FIELDS = ["category_id", "video_type_id", "music_type_id", "cat_l1_id", "cat_l2_id", "cat_l3_id", "cat_l4_id"]
NUM_STAT_FEATURES = 8  # VIDEO_STATIC_STAT_FIELDS, xem schema.py


class ItemEmbeddingConfig:
    def __init__(
        self,
        num_items: int,
        num_authors: int,
        num_music: int,
        num_categories: dict[str, int],  # field name (trong CATEGORICAL_FIELDS) -> số lượng category
        dim: int = 64,
        cat_embed_dim: int = 16,
        id_embed_dim: int = 16,
    ):
        self.num_items = num_items
        self.num_authors = num_authors
        self.num_music = num_music
        self.num_categories = num_categories
        self.dim = dim
        self.cat_embed_dim = cat_embed_dim
        self.id_embed_dim = id_embed_dim


class ItemEmbedding(nn.Module):
    def __init__(self, config: ItemEmbeddingConfig):
        super().__init__()
        self.config = config

        # collaborative_branch — thuần ID embedding, KHÔNG dùng feature nào khác (đã chốt).
        # sparse=True: 32,038,725 item x dim -> dense Adam state (exp_avg+exp_avg_sq) sẽ cấp
        # phát ~8.2GB CHO TOÀN BỘ bảng dù mỗi batch chỉ chạm vài trăm dòng (đã xác nhận qua
        # RuntimeError thật khi test) — sparse gradient để optimizer chỉ giữ state cho các
        # dòng THỰC SỰ có gradient khác 0. Yêu cầu optimizer riêng (SparseAdam, xem train.py).
        # TODO: chia model parallelism (item_id % 2 -> GPU0/GPU1) khi có 2 GPU thật —
        # single-device trước, xem idea.md "TRẠNG THÁI DỰ ÁN" quyết định #1.
        self.collaborative_embedding = nn.Embedding(config.num_items, config.dim, sparse=True)

        # content_branch — categorical nhỏ (mỗi field 1 bảng embedding riêng, KHÔNG sparse vì
        # số lượng category nhỏ, dense Adam state không đáng kể) + author/music (embedding lớn
        # hơn nhiều — 8.8M/14.2M hàng, CŨNG cần sparse=True cùng lý do trên) + numeric (8 stat
        # feature, đưa thẳng vào GMU dạng vector liên tục, không cần embedding).
        self.category_embeddings = nn.ModuleDict({
            field: nn.Embedding(config.num_categories[field], config.cat_embed_dim) for field in CATEGORICAL_FIELDS
        })
        self.author_embedding = nn.Embedding(config.num_authors, config.id_embed_dim, sparse=True)
        self.music_embedding = nn.Embedding(config.num_music, config.id_embed_dim, sparse=True)

        gmu_in_dims = {field: config.cat_embed_dim for field in CATEGORICAL_FIELDS}
        gmu_in_dims["author"] = config.id_embed_dim
        gmu_in_dims["music"] = config.id_embed_dim
        gmu_in_dims["stat_features"] = NUM_STAT_FEATURES
        self.content_gmu = GMU(gmu_in_dims, dim=config.dim)

        # g_i = σ(MLP([mat_i, collaborative_branch, content_branch])) — mở rộng GMU-gate
        # thêm 1 tầng gate trộn collaborative/content (xem idea.md mục 4.5 điểm 2).
        self.gate_mlp = nn.Sequential(
            nn.Linear(1 + config.dim + config.dim, config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, 1),
        )

    def forward(
        self,
        video_idx: torch.Tensor,  # (B,) int64 — index vào collaborative_embedding (0..num_items-1)
        category_ids: dict[str, torch.Tensor],  # field -> (B,) int64
        author_idx: torch.Tensor,  # (B,) int64
        music_idx: torch.Tensor,  # (B,) int64
        stat_features: torch.Tensor,  # (B, NUM_STAT_FEATURES) float32
        mat_i: torch.Tensor,  # (B,) float32 — tanh(N_i / τ_i), đã tính sẵn ở Pass 5
        conf_content: torch.Tensor,  # (B,) float32 — tanh(N_category(i) / τ_c), đã tính sẵn
    ) -> torch.Tensor:
        collaborative_branch = self.collaborative_embedding(video_idx)  # (B, dim)

        gmu_inputs = {field: self.category_embeddings[field](category_ids[field]) for field in CATEGORICAL_FIELDS}
        gmu_inputs["author"] = self.author_embedding(author_idx)
        gmu_inputs["music"] = self.music_embedding(music_idx)
        gmu_inputs["stat_features"] = stat_features
        content_branch = self.content_gmu(gmu_inputs)  # (B, dim)

        content_branch_shrunk = content_branch * conf_content.unsqueeze(-1)  # (B, dim)

        gate_input = torch.cat([mat_i.unsqueeze(-1), collaborative_branch, content_branch], dim=-1)
        g_i = torch.sigmoid(self.gate_mlp(gate_input))  # (B, 1)

        return g_i * collaborative_branch + (1 - g_i) * content_branch_shrunk  # (B, dim) = E_i

    def sparse_parameters(self) -> list[nn.Parameter]:
        """3 bảng embedding lớn (sparse=True) — cần SparseAdam riêng, xem train.py."""
        return (
            list(self.collaborative_embedding.parameters())
            + list(self.author_embedding.parameters())
            + list(self.music_embedding.parameters())
        )

    def dense_parameters(self) -> list[nn.Parameter]:
        """Phần còn lại (category_embeddings nhỏ + GMU + gate_mlp) — Adam thường."""
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]
