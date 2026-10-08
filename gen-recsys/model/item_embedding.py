"""Content token — biến 1 item thành 1 embedding, dùng làm token trong chuỗi user (HSTU gọi"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

# category ĐÃ RA KHỎI GMU (xem ItemEmbedding.forward). Chỉ còn 2 field nhỏ ở đây.
CATEGORICAL_FIELDS = ["video_type_id", "music_type_id"]

# Nhánh 3 (formula.md §1) — category 4 cấp, CHỈ có trên 27K (Pure không có file
# kuairand_video_categories.csv). Thiếu field trong item_static => nhánh tự tắt.
CATEGORY_LEVEL_FIELDS = ["cat_l1_id", "cat_l2_id", "cat_l3_id", "cat_l4_id"]

TAG_PAD = 0  # khớp build_item_static.TAG_PAD — index 0 = "không có tag"
CATEGORY_PAD = 0  # khớp build_item_static: UNKNOWN (-124 ở CSV) -> 0


class ItemEmbeddingConfig:
    def __init__(
        self,
        num_items: int,
        num_authors: int,
        num_music: int,
        num_categories: dict[str, int],
        num_tags: int,
        use_tag: bool = True,
        use_caption: bool = True,
        num_category_levels: dict[str, int] | None = None,
        dim: int = 64,
        cat_embed_dim: int = 16,
        id_embed_dim: int = 16,
        caption_dim: int = 384,
    ):
        self.num_items = num_items
        self.num_authors = num_authors
        self.num_music = num_music
        self.num_categories = num_categories
        self.num_tags = num_tags
        # False = ablation --no-tag: bo nhanh 2, W^fuse bot 1 nhanh (formula.md muc 1)
        self.use_tag = use_tag
        # [THEM 2026-10-08] False = ablation --no-caption: bo modality `caption` khoi GMU.
        # Dung de chay duoc KHI CHUA CO caption_embeddings.npy — tren 27K file do nang
        # 22.9 GB fp16 (32,038,725 x 384) va phai encode ~15.4h, nen can chay thu model
        # truoc khi cho encode xong. GMU con 4 modality (video_type, music_type, author,
        # music) + nhanh tag + nhanh category 4 cap, nen content token VAN phan biet duoc
        # item; gate softmax tu chuan hoa lai tren 4 modality (xem gmu.py).
        self.use_caption = use_caption
        # None = dataset KHÔNG có category 4 cấp (Pure) => nhánh 3 tắt, W^fuse về d x 2d
        self.num_category_levels = num_category_levels
        self.dim = dim
        self.cat_embed_dim = cat_embed_dim
        self.id_embed_dim = id_embed_dim
        self.caption_dim = caption_dim


class ItemEmbedding(nn.Module):
    def __init__(self, config: ItemEmbeddingConfig):
        super().__init__()
        self.config = config

        self.author_embedding = nn.Embedding(config.num_authors, config.id_embed_dim)
        self.music_embedding = nn.Embedding(config.num_music, config.id_embed_dim)

        self.category_embeddings = nn.ModuleDict({
            field: nn.Embedding(config.num_categories[field], config.cat_embed_dim) for field in CATEGORICAL_FIELDS
        })

        gmu_in_dims = {field: config.cat_embed_dim for field in CATEGORICAL_FIELDS}
        gmu_in_dims["author"] = config.id_embed_dim
        gmu_in_dims["music"] = config.id_embed_dim
        if config.use_caption:
            gmu_in_dims["caption"] = config.caption_dim
        self.content_gmu = GMU(gmu_in_dims, dim=config.dim)

        # tag KHÔNG vào GMU: gate GMU là softmax trên các modality (tổng = 1), nên caption
        # 384-d có thể ép tag về ~0 mà không có gì báo. Category là đơn vị phân tích của
        # câu hỏi nghiên cứu, nên nó đi đường riêng rồi concat — W tự học tỉ lệ giữa hai
        # nguồn thay vì phó mặc cho tỉ lệ norm.
        self.tag_embedding = (
            nn.Embedding(config.num_tags, config.dim, padding_idx=TAG_PAD)
            if config.use_tag else None
        )

        # Nhánh 3 — category 4 cấp (formula.md §1). Mỗi cấp một bảng riêng rồi concat +
        # chiếu: cấp 1 (39 nhãn) BÃO HOÀ nên một bảng gộp sẽ để cấp 1 lấn, còn cộng thì
        # ép mọi cấp cùng thang. 4 x d_c -> d.
        self.level_embeddings = None
        if config.num_category_levels is not None:
            self.level_embeddings = nn.ModuleDict({
                field: nn.Embedding(config.num_category_levels[field], config.cat_embed_dim,
                                    padding_idx=CATEGORY_PAD)
                for field in CATEGORY_LEVEL_FIELDS
            })
            self.cat_proj = nn.Linear(4 * config.cat_embed_dim, config.dim)

        # So nhanh = item (luon co) + tag? + cat?. Tat CA HAI thi content token chi con
        # GMU va `fuse` thanh phep chieu d->d, khong con gi de "hop nhat" => train.py chan.
        num_branches = (1 + int(self.tag_embedding is not None)
                          + int(self.level_embeddings is not None))
        self.fuse = nn.Linear(num_branches * config.dim, config.dim)

        self._last_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        video_idx: torch.Tensor,
        category_ids: dict[str, torch.Tensor],
        author_idx: torch.Tensor,
        music_idx: torch.Tensor,
        caption_embedding: torch.Tensor,
        caption_mask: torch.Tensor,
        tag_ids: torch.Tensor,
        category_level_ids: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """e_i = content token (HSTU gọi Φ_t). Xem formula.md §1."""
        gmu_inputs = {field: self.category_embeddings[field](category_ids[field]) for field in CATEGORICAL_FIELDS}
        gmu_inputs["author"] = self.author_embedding(author_idx)
        gmu_inputs["music"] = self.music_embedding(music_idx)
        # [SUA 2026-10-08] caption chi vao GMU khi use_caption — xem ItemEmbeddingConfig.
        # Tat thi KHONG dua key `caption` vao gmu_inputs (GMU hard-code stack theo dung tap
        # key da khai o __init__, them key la vo trong softmax sai truc).
        if self.config.use_caption:
            gmu_inputs["caption"] = caption_embedding
            e_item = self.content_gmu(gmu_inputs, masks={"caption": caption_mask})
        else:
            e_item = self.content_gmu(gmu_inputs)

        # `tag` là MULTI-LABEL (75% item 1 tag, 23% có 2, 0.4% có 3) — mean trên các tag
        # THẬT. Item chung tag ⇒ chung một nửa đầu vào của fuse, nên chúng gần nhau ngay
        # cả khi caption/author khác hẳn.
        branches = [e_item]
        e_tag = None
        if self.tag_embedding is not None:
            valid = (tag_ids != TAG_PAD).unsqueeze(-1).to(e_item.dtype)
            e_tag = ((self.tag_embedding(tag_ids) * valid).sum(dim=-2)
                     / valid.sum(dim=-2).clamp(min=1.0))
            branches.append(e_tag)

        e_cat = None
        if self.level_embeddings is not None:
            if category_level_ids is None:
                raise ValueError(
                    "model có nhánh category 4 cấp nhưng caller không truyền "
                    "category_level_ids — nhánh sẽ im lặng rơi khỏi graph."
                )
            e_cat = self.cat_proj(torch.cat(
                [self.level_embeddings[f](category_level_ids[f]) for f in CATEGORY_LEVEL_FIELDS],
                dim=-1,
            ))
            branches.append(e_cat)

        e_content = self.fuse(torch.cat(branches, dim=-1))

        self._last_stats = {
            "norm_content": e_content.detach().norm(dim=-1),
            "norm_item": e_item.detach().norm(dim=-1),
        }
        if e_tag is not None:
            self._last_stats["norm_tag"] = e_tag.detach().norm(dim=-1)
        if e_cat is not None:
            self._last_stats["norm_cat"] = e_cat.detach().norm(dim=-1)
        return e_content

    def dense_parameters(self) -> list[nn.Parameter]:
        """MỌI tham số — không còn chia sparse/dense."""
        return list(self.parameters())
