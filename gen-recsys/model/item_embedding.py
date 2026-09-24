"""Content token — biến 1 item thành 1 embedding, dùng làm token trong chuỗi user (HSTU gọi"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

# category ĐÃ RA KHỎI GMU (xem ItemEmbedding.forward). Chỉ còn 2 field nhỏ ở đây.
CATEGORICAL_FIELDS = ["video_type_id", "music_type_id"]

TAG_PAD = 0  # khớp build_item_static.TAG_PAD — index 0 = "không có tag"


class ItemEmbeddingConfig:
    def __init__(
        self,
        num_items: int,
        num_authors: int,
        num_music: int,
        num_categories: dict[str, int],
        num_tags: int = 47,
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
        gmu_in_dims["caption"] = config.caption_dim
        self.content_gmu = GMU(gmu_in_dims, dim=config.dim)

        # tag KHÔNG vào GMU: gate GMU là softmax trên các modality (tổng = 1), nên caption
        # 384-d có thể ép tag về ~0 mà không có gì báo. Category là đơn vị phân tích của
        # câu hỏi nghiên cứu, nên nó đi đường riêng rồi concat — W tự học tỉ lệ giữa hai
        # nguồn thay vì phó mặc cho tỉ lệ norm.
        self.tag_embedding = nn.Embedding(config.num_tags, config.dim, padding_idx=TAG_PAD)
        self.fuse = nn.Linear(2 * config.dim, config.dim)

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
    ) -> torch.Tensor:
        """e_i = content token (HSTU gọi Φ_t). Xem formula.md §1."""
        gmu_inputs = {field: self.category_embeddings[field](category_ids[field]) for field in CATEGORICAL_FIELDS}
        gmu_inputs["author"] = self.author_embedding(author_idx)
        gmu_inputs["music"] = self.music_embedding(music_idx)
        gmu_inputs["caption"] = caption_embedding
        e_item = self.content_gmu(gmu_inputs, masks={"caption": caption_mask})

        # `tag` là MULTI-LABEL (75% item 1 tag, 23% có 2, 0.4% có 3) — mean trên các tag
        # THẬT. Item chung tag ⇒ chung một nửa đầu vào của fuse, nên chúng gần nhau ngay
        # cả khi caption/author khác hẳn.
        valid = (tag_ids != TAG_PAD).unsqueeze(-1).to(e_item.dtype)
        e_tag = (self.tag_embedding(tag_ids) * valid).sum(dim=-2) / valid.sum(dim=-2).clamp(min=1.0)

        e_content = self.fuse(torch.cat([e_item, e_tag], dim=-1))

        self._last_stats = {
            "norm_content": e_content.detach().norm(dim=-1),
            "norm_item": e_item.detach().norm(dim=-1),
            "norm_tag": e_tag.detach().norm(dim=-1),
        }
        return e_content

    def dense_parameters(self) -> list[nn.Parameter]:
        """MỌI tham số — không còn chia sparse/dense."""
        return list(self.parameters())
