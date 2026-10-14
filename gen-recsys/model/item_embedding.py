"""Content token — biến 1 item thành 1 embedding, dùng làm token trong chuỗi user (HSTU gọi"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

# category ĐÃ RA KHỎI GMU (xem ItemEmbedding.forward). Chỉ còn 2 field nhỏ ở đây.
CATEGORICAL_FIELDS = ["video_type_id", "music_type_id"]

TAG_PAD = 0  # khớp build_item_static.TAG_PAD — index 0 = "không có tag"


class TagPMA(nn.Module):
    """Pooling by Multihead Attention, k=1 (Lee et al. ICML 2019, arXiv:1810.00825).

    PMA_1(Z) = MAB(s, rFF(Z)), MAB(X,Y) = LayerNorm(H + rFF(H)),
    H = LayerNorm(X + Multihead(X,Y,Y)). Seed s hoc duoc, KHONG phu thuoc item — dong vai
    tro "query ngam dinh" hoi tap tag "tag nao quan trong nhat o day".

    So voi mean-pooling cu: trong so tag KHONG con chia deu 1/|T_t|, ma hoc qua attention.
    Dung cho |T_t| NHO (Pure: toi da 3 tag/item) — formula.md §1 da ghi ro rui ro CHUA
    kiem chung o quy mo set nho nay, khac boi canh goc cua paper (set lon, point cloud).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim phai chia het cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.seed = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.seed)

        self.rff_in = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.rff_out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, tags: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """tags: (N, T, dim) tag embedding (gom ca PAD). valid_mask: (N, T) bool, True=tag that.

        Item khong co tag that nao (valid_mask toan False) se khien softmax attention tren
        toan bo hang la -inf -> NaN. Xu ly: voi item do, bo qua key_padding_mask (attend
        deu moi slot, dieu bang PAD=0 cua tag_embedding) — ket qua la vector gan 0, khop
        hanh vi "item khong tag" cua ban mean-pooling cu.
        """
        n = tags.shape[0]
        z = self.rff_in(tags)
        seed = self.seed.expand(n, -1, -1)

        has_any_valid = valid_mask.any(dim=-1)
        key_padding_mask = ~valid_mask
        key_padding_mask = torch.where(
            has_any_valid.unsqueeze(-1), key_padding_mask, torch.zeros_like(key_padding_mask)
        )

        attn_out, _ = self.mha(seed, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(seed + attn_out)
        out = self.ln2(h + self.rff_out(h))
        return out.squeeze(1)


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
        # [THEM 2026-10-09] PMA (Lee et al. ICML 2019, arXiv:1810.00825 cong thuc 11/6/7),
        # thay mean-pooling cu — xem formula.md §1. k=1 seed vector hoc duoc, MAB = 2 lop
        # rFF + 2 LayerNorm boc quanh multihead attention (KHONG phai 1 attention tran).
        self.tag_pma = TagPMA(config.dim) if config.use_tag else None

        # So nhanh = item (luon co) + tag?. Tat tag thi content token chi con GMU va
        # `fuse` thanh phep chieu d->d, khong con gi de "hop nhat" => train.py chan.
        num_branches = 1 + int(self.tag_embedding is not None)
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

        # `tag` là MULTI-LABEL (75% item 1 tag, 23% có 2, 0.4% có 3) — PMA (TagPMA) thay
        # mean cũ: seed học được attend lên các tag THẬT, trọng số không còn chia đều.
        # Item chung tag ⇒ chung một phần đầu vào của fuse, nên chúng gần nhau hơn ngay
        # cả khi caption/author khác hẳn.
        branches = [e_item]
        e_tag = None
        if self.tag_embedding is not None:
            valid = tag_ids != TAG_PAD
            e_tag = self.tag_pma(self.tag_embedding(tag_ids), valid)
            branches.append(e_tag)

        e_content = self.fuse(torch.cat(branches, dim=-1))

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
