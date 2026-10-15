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
        num_tags: int,
        use_tag: bool = True,
        use_caption: bool = True,
        use_item_id: bool = True,
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
        # Dung de chay duoc KHI CHUA CO caption_embeddings.npy. GMU con 4 modality
        # (video_type, music_type, author, music) + nhanh tag, nen content token VAN phan
        # biet duoc item; gate softmax tu chuan hoa lai tren 4 modality (xem gmu.py).
        self.use_caption = use_caption
        # [THEM 2026-10-15] False = ablation --no-item-id: Phi_i content THUAN nhu truoc 10-15.
        self.use_item_id = use_item_id
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
        # [XOA 2026-10-15] TagPMA (Set Transformer, 2026-10-09) -> quay ve MEAN-POOLING. Pure
        # co toi da 3 tag/item, 75% chi 1 tag => attention tren tap 1 phan tu la phep tinh
        # thua; audit gradient 10-14: 19-30% unit ReLU trong rff_out khong co gradient.
        # Mean-pooling la chuan cho feature multi-hot (YouTube DNN), formula.md §1.

        # So nhanh = item (luon co) + tag?. Tat tag thi content token chi con GMU va
        # `fuse` thanh phep chieu d->d, khong con gi de "hop nhat" => train.py chan.
        num_branches = 1 + int(self.tag_embedding is not None)
        self.fuse = nn.Linear(num_branches * config.dim, config.dim)

        # [THEM 2026-10-15] ITEM-ID EMBEDDING (ID nguyen tu, SASRec/HSTU/FuXi deu co). Truoc
        # day Phi_i la content THUAN => 2 item cung author/tag/caption gan nhau gan nhu TRUNG
        # bieu dien du user phan ung khac han (85 item chung 1 vector caption) — thieu kha
        # nang GHI NHO (memorization), thu chi ID rieng moi tach duoc. Semantic ID KHONG giai
        # duoc: no la chinh content duoc luong tu hoa lai. Content-only chi hop ly khi theo
        # cold-start item, ma formula.md §−1 da bo huong do.
        # Phi_i = content_i + ID_i, ID ZERO-INIT => luc init model Y HET ban content-only
        # (ablation sach). Cong (khong nhan) nen KHONG vuong bug "tich 2 so 0" nhu W_ts dong:
        # d/d ID = gradient upstream, co ngay buoc 1. Pure 7,583 x 64 = 0.49M tham so, khong
        # can hash. Nhan weight decay (ndim=2) — dung y NormFace, chan norm phinh.
        self.id_embedding = None
        if config.use_item_id:
            self.id_embedding = nn.Embedding(config.num_items, config.dim)
            nn.init.zeros_(self.id_embedding.weight)

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
        # [SUA 2026-10-08] caption chi vao GMU khi use_caption — xem ItemEmbeddingConfig.
        # Tat thi KHONG dua key `caption` vao gmu_inputs (GMU hard-code stack theo dung tap
        # key da khai o __init__, them key la vo trong softmax sai truc).
        if self.config.use_caption:
            gmu_inputs["caption"] = caption_embedding
            e_item = self.content_gmu(gmu_inputs, masks={"caption": caption_mask})
        else:
            e_item = self.content_gmu(gmu_inputs)

        # `tag` là MULTI-LABEL (75% item 1 tag, 23% có 2, 0.4% có 3) — mean trên các tag
        # THẬT (PAD bị loại). Item chung tag ⇒ chung một phần đầu vào của fuse, nên chúng gần nhau hơn ngay
        # cả khi caption/author khác hẳn.
        branches = [e_item]
        e_tag = None
        if self.tag_embedding is not None:
            valid = (tag_ids != TAG_PAD).unsqueeze(-1).to(e_item.dtype)
            e_tag = (self.tag_embedding(tag_ids) * valid).sum(dim=-2) / valid.sum(dim=-2).clamp(min=1.0)
            branches.append(e_tag)

        e_content = self.fuse(torch.cat(branches, dim=-1))
        if self.id_embedding is not None:
            e_content = e_content + self.id_embedding(video_idx)

        self._last_stats = {
            "norm_content": e_content.detach().norm(dim=-1),
            "norm_item": e_item.detach().norm(dim=-1),
        }
        if e_tag is not None:
            self._last_stats["norm_tag"] = e_tag.detach().norm(dim=-1)
        return e_content

    def dense_parameters(self) -> list[nn.Parameter]:
        """MỌI tham số — không còn chia sparse/dense."""
        return list(self.parameters())
