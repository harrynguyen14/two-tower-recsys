"""Item embedding module — biến 1 item thành 1 embedding e_i_final, dùng làm token trong
chuỗi user hoặc làm candidate lúc decode (KHÔNG phải "item tower" — xem lưu ý thuật ngữ ở
idea.md mục 4.5 đầu mục "Đề xuất thiết kế cụ thể").

[SỬA 2026-09-13] Đổi tên biến cho tường minh (xem result.md "CHECKLIST CUỐI CÙNG"),
KHÔNG đổi công thức:
    mat_i             -> item_weight
    conf_content      -> category_confidence
    content_branch_shrunk -> category_content_shrunk
    collaborative_branch  -> e_collab
    content_branch        -> e_content
    E_i                   -> e_i_final

Công thức đã chốt (idea.md mục 4.5 điểm 2 + điểm 3):
    category_confidence(i)  = tanh(N_category(i) / τ_c)
    category_content_shrunk = e_content(i) · category_confidence(i)
    g_i                     = σ(MLP([item_weight, e_collab(i), e_content(i)]))
    e_i_final               = g_i · e_collab(i) + (1-g_i) · category_content_shrunk

e_collab: học từ đầu theo video_id — thuần ID embedding, KHÔNG dùng feature nào khác (đã
chốt 2026-09-10). Do catalog lớn, cần chia model parallelism (item_id % 2 -> GPU0/GPU1,
xem idea.md mục "TRẠNG THÁI DỰ ÁN" quyết định #1) — CHƯA làm ở module này (single-GPU
trước, xem TODO ở EmbeddingConfig), thêm khi có 2 GPU thật để test.

[THÊM 2026-09-11] `use_cuckoo_embedding=True` (mặc định) thay nn.Embedding cố định bằng
CuckooEmbedding (xem cuckoo_embedding.py, lấy cảm hứng ByteDance Monolith "Collisionless
Embedding Table") cho 3 bảng ID lớn (collaborative/author/music) — giải quyết đúng gap đã
tìm thấy khi review kiến trúc: nn.Embedding(num_items, dim) cố định kích thước lúc train
KHÔNG có hàng nào cho item/author/music HOÀN TOÀN MỚI xuất hiện lúc serving (không
hash-bucket/fallback). CuckooEmbedding dùng capacity NHỎ HƠN tổng ID tiềm năng (không cần
biết trước, đúng tinh thần Monolith cho hệ thống serving thật đang chạy liên tục) — ID mới
được cấp slot ngay (evict ID ít hoạt động nếu bảng đầy) thay vì crash/IndexError.
`use_cuckoo_embedding=False` giữ nn.Embedding cũ để so sánh ablation.

[CẢNH BÁO review.md] CuckooEmbedding (Python dict-based) đo được ~0.38s/step resolve trên
19,264 ID/batch — đây là bottleneck thật (dù nhỏ hơn I/O memmap ~9-18s/step). Với dataset
KuaiRand-Pure (7,583 item, không phải 32M), CÂN NHẮC `use_cuckoo_embedding=False` (dùng
nn.Embedding cố định thường) — bảng nhỏ, không cần collisionless, tiết kiệm hẳn 0.38s/step
này. Quyết định cụ thể để lại cho lúc build pipeline cho Pure, KHÔNG đổi mặc định ở đây.

e_content: qua GMU (gmu.py) fuse các nhánh tĩnh của item — categorical (category 1 cấp +
video_type + music_type) + author/music (ID embedding nhỏ). author_idx/music_idx CŨNG nằm
trong e_content (đã CHỐT 2026-09-10 — "author/music là 1 nhánh trong content qua GMU",
KHÔNG tách riêng thành số hạng thứ 3 trong e_i_final, giữ đúng công thức 2 nhánh).

[SỬA 2026-09-13] Chuyển sang KuaiRand-Pure (xem schema.py, result.md "CHECKLIST CUỐI
CÙNG"): BỎ 4 field category 4 cấp (cat_l1-l4_id — Pure không có file nguồn) và BỎ
stat_features (review.md L9 — snapshot cuối kỳ leak thời gian, corr≈0.92 với N_i cuối kỳ
qua mô phỏng) khỏi content_gmu. content_gmu giờ chỉ còn 5 modality: category_id,
video_type_id, music_type_id, author, music (+ caption optional).

[CHỐT 2026-09-11] Nhánh caption (text tiếng Trung, qua multilingual-e5-small, 384 chiều,
encode offline — xem encode_captions_kaggle.py) là nhánh OPTIONAL thứ 8 trong content_gmu
— đúng tinh thần "GMU tổng quát, text/image thêm sau chỉ cần thêm entry vào dict, không
sửa module" đã thiết kế từ đầu (xem gmu.py). Item KHÔNG có caption thật
(caption_has_caption.npy = False) truyền mask=0, GMU tự loại nhánh này khỏi softmax gate
cho đúng item đó.

[CẢNH BÁO review.md L9] `stat_features` (VIDEO_STATIC_STAT_FIELDS: play_cnt, like_cnt...)
là SNAPSHOT CUỐI KỲ (tổng cộng dồn tới lúc thu thập dataset) — nếu build lại cho Pure,
PHẢI kiểm tra/sửa để tránh leak thời gian tương tự đã phát hiện trên 27K (corr ước lượng
~0.92 với N_i cuối kỳ). Đây là việc của build_item_static.py (preprocess), KHÔNG sửa ở
module này — chỉ ghi chú lại để không quên khi build pipeline cho Pure.

Input: item_static.npy (Pass 3, xem build_item_static.py) — mỗi field int32 đã factorize
sẵn thành index liên tục [0, n), category_id/video_type_id/music_type_id/cat_l1-l4_id
dùng nn.Embedding riêng từng field (KHÔNG concat one-hot — số category nhỏ nhưng tách
riêng để mỗi field có không gian embedding riêng, không ép chung 1 bảng). caption_embedding
đọc riêng từ caption_embeddings.npy/caption_has_caption.npy (Pass 3.5, merge_caption_shards.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cuckoo_embedding import CuckooEmbedding
from gmu import GMU

# Field categorical trong item_static.npy dùng embedding riêng (tên field -> tên tham số
# num_embeddings tương ứng khi khởi tạo ItemEmbeddingConfig) — KHÔNG gồm author_idx/
# music_idx (embedding lớn, tách riêng dict `id_embeddings` vì kích thước rất khác biệt
# so với category nhỏ, dù cùng "categorical" về bản chất).
CATEGORICAL_FIELDS = ["category_id", "video_type_id", "music_type_id"]


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
        caption_dim: int = 384,  # multilingual-e5-small, xem encode_captions_kaggle.py
        use_cuckoo_embedding: bool = True,  # xem cuckoo_embedding.py — thay nn.Embedding cố định
        cuckoo_capacity_ratio: float = 0.25,  # capacity mỗi bảng con = ratio * num_ids — CỐ TÌNH
        # nhỏ hơn tổng ID (2 bảng con * ratio = 0.5x tổng số ID có slot ĐỒNG THỜI, không phải
        # 1x) để mô phỏng đúng vấn đề Monolith giải quyết: bảng KHÔNG đủ chỗ cho MỌI ID, cần cơ
        # chế evict thật. ratio=1.0 sẽ gần như không bao giờ evict (mất ý nghĩa mô phỏng).
    ):
        self.num_items = num_items
        self.num_authors = num_authors
        self.num_music = num_music
        self.num_categories = num_categories
        self.dim = dim
        self.cat_embed_dim = cat_embed_dim
        self.id_embed_dim = id_embed_dim
        self.caption_dim = caption_dim
        self.use_cuckoo_embedding = use_cuckoo_embedding
        self.cuckoo_capacity_ratio = cuckoo_capacity_ratio


class ItemEmbedding(nn.Module):
    def __init__(self, config: ItemEmbeddingConfig):
        super().__init__()
        self.config = config

        # e_collab — thuần ID embedding, KHÔNG dùng feature nào khác (đã chốt).
        if config.use_cuckoo_embedding:
            cap_items = max(1, int(config.num_items * config.cuckoo_capacity_ratio))
            cap_authors = max(1, int(config.num_authors * config.cuckoo_capacity_ratio))
            cap_music = max(1, int(config.num_music * config.cuckoo_capacity_ratio))
            self.collab_embedding = CuckooEmbedding(cap_items, config.dim, seed=1)
            self.author_embedding = CuckooEmbedding(cap_authors, config.id_embed_dim, seed=2)
            self.music_embedding = CuckooEmbedding(cap_music, config.id_embed_dim, seed=3)
        else:
            self.collab_embedding = nn.Embedding(config.num_items, config.dim, sparse=True)
            self.author_embedding = nn.Embedding(config.num_authors, config.id_embed_dim, sparse=True)
            self.music_embedding = nn.Embedding(config.num_music, config.id_embed_dim, sparse=True)

        # e_content — categorical nhỏ (mỗi field 1 bảng embedding riêng, KHÔNG sparse vì số
        # lượng category nhỏ, dense Adam state không đáng kể) + numeric (8 stat feature, đưa
        # thẳng vào GMU dạng vector liên tục, không cần embedding).
        self.category_embeddings = nn.ModuleDict({
            field: nn.Embedding(config.num_categories[field], config.cat_embed_dim) for field in CATEGORICAL_FIELDS
        })

        gmu_in_dims = {field: config.cat_embed_dim for field in CATEGORICAL_FIELDS}
        gmu_in_dims["author"] = config.id_embed_dim
        gmu_in_dims["music"] = config.id_embed_dim
        gmu_in_dims["caption"] = config.caption_dim  # nhánh OPTIONAL, xem docstring module
        self.content_gmu = GMU(gmu_in_dims, dim=config.dim)

        # g_i = σ(MLP([item_weight, e_collab, e_content])) — mở rộng GMU-gate thêm 1 tầng
        # gate trộn collab/content (xem idea.md mục 4.5 điểm 2).
        self.gate_mlp = nn.Sequential(
            nn.Linear(1 + config.dim + config.dim, config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, 1),
        )

        # Ảnh chụp thành phần nội bộ của forward() gần nhất — chỉ để chẩn đoán, xem forward().
        self._last_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        video_idx: torch.Tensor,  # (B,) int64 — index vào collab_embedding (0..num_items-1)
        category_ids: dict[str, torch.Tensor],  # field -> (B,) int64
        author_idx: torch.Tensor,  # (B,) int64
        music_idx: torch.Tensor,  # (B,) int64
        item_weight: torch.Tensor,  # (B,) float32 — tanh(N_i / τ_i), đã tính sẵn ở caller
        category_confidence: torch.Tensor,  # (B,) float32 — tanh(N_category(i) / τ_c), đã tính sẵn
        caption_embedding: torch.Tensor,  # (B, caption_dim) float32 — 0 nếu không có caption thật
        caption_mask: torch.Tensor,  # (B,) float32/bool — 1 nếu item CÓ caption thật, 0 nếu không
    ) -> torch.Tensor:
        e_collab = self.collab_embedding(video_idx)  # (B, dim)

        gmu_inputs = {field: self.category_embeddings[field](category_ids[field]) for field in CATEGORICAL_FIELDS}
        gmu_inputs["author"] = self.author_embedding(author_idx)
        gmu_inputs["music"] = self.music_embedding(music_idx)
        gmu_inputs["caption"] = caption_embedding
        e_content = self.content_gmu(gmu_inputs, masks={"caption": caption_mask})  # (B, dim)

        category_content_shrunk = e_content * category_confidence.unsqueeze(-1)  # (B, dim)

        gate_input = torch.cat([item_weight.unsqueeze(-1), e_collab, e_content], dim=-1)
        g_i = torch.sigmoid(self.gate_mlp(gate_input))  # (B, 1)

        # [CHẨN ĐOÁN 2026-09-16] Ghi lại thành phần nội bộ để evaluate() đọc, KHÔNG đổi giá
        # trị trả về. Cần trả lời 2 câu trước khi sửa công thức gate:
        #   - g_i có tiến về 0 khi item cold không? (gate làm đúng việc hay không)
        #   - category_confidence có THẬT SỰ biến thiên không, hay bão hoà ~1 ở mọi mẫu?
        #     τ_c=53486 không nhúc nhích sau 3000 step ở cả 6 lần chạy -> nghi tanh đang ở
        #     vùng phẳng. Nếu c ≈ hằng số thì đưa c vào gate cũng vô nghĩa, phải sửa τ_c.
        # detach(): thuần quan sát, không để lọt vào đồ thị gradient.
        self._last_stats = {
            "g_i": g_i.detach().squeeze(-1),
            "category_confidence": category_confidence.detach(),
            "item_weight": item_weight.detach(),
            "norm_collab": e_collab.detach().norm(dim=-1),
            "norm_content": e_content.detach().norm(dim=-1),
            "norm_content_shrunk": category_content_shrunk.detach().norm(dim=-1),
        }

        return g_i * e_collab + (1 - g_i) * category_content_shrunk  # (B, dim) = e_i_final

    def sparse_parameters(self) -> list[nn.Parameter]:
        """3 bảng embedding lớn (sparse=True) — cần SparseAdam riêng, xem train.py."""
        return (
            list(self.collab_embedding.parameters())
            + list(self.author_embedding.parameters())
            + list(self.music_embedding.parameters())
        )

    def dense_parameters(self) -> list[nn.Parameter]:
        """Phần còn lại (category_embeddings nhỏ + GMU + gate_mlp) — Adam thường."""
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]
