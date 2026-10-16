"""Content token Φ_i — biến 1 item thành 1 embedding, dùng làm token trong chuỗi user.

Hai cấu hình, chọn bằng cờ `--no-content`:
  ID + content (mặc định): Φ_i = ID_i + W_fuse [ MLP(caption_i) ; mean_tag_i ]
  ID-only (= HSTU):        Φ_i = ID_i

ID-only là đúng cấu hình HSTU (một bảng nn.Embedding duy nhất, xem
generative_recommenders/research/modeling/sequential/embedding_modules.py) nên so trực tiếp
được với HSTU gốc. Hai cấu hình dùng CHUNG đường ID, nên chênh lệch chỉ do phần content.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TAG_PAD = 0  # khớp build_item_static.TAG_PAD — index 0 = "không có tag"

# HSTU khởi tạo ID std 0.02 rồi nhân sqrt(d) (= 0.16 ở d=64) để cân với positional embedding
# std 1/sqrt(d) (= 0.125): tỉ lệ ID:pos ~ 1.3. SequenceModel của ta giữ positional mặc định
# N(0,1), nên ID khởi tạo std 1.0 để giữ đúng tỉ lệ đó. Dùng std 0.02 thì ID bị positional
# lấn át từ đầu (nhỏ hơn ~50 lần) và nhánh ID-only bị thiệt không công bằng.
ID_INIT_STD = 1.0


class ItemEmbeddingConfig:
    def __init__(
        self,
        num_items: int,
        num_tags: int,
        use_content: bool = True,
        dim: int = 64,
        caption_dim: int = 384,
    ):
        self.num_items = num_items
        self.num_tags = num_tags
        # False = ID-only (--no-content): Φ_i chỉ là bảng ID, giống HSTU.
        self.use_content = use_content
        self.dim = dim
        self.caption_dim = caption_dim


class ItemEmbedding(nn.Module):
    def __init__(self, config: ItemEmbeddingConfig):
        super().__init__()
        self.config = config
        d = config.dim

        self.id_embedding = nn.Embedding(config.num_items, d)
        nn.init.trunc_normal_(self.id_embedding.weight, std=ID_INIT_STD,
                              a=-2 * ID_INIT_STD, b=2 * ID_INIT_STD)

        if config.use_content:
            # caption: embedding câu CỐ ĐỊNH (đã trừ mean + L2-normalize ở dataset) -> MLP
            self.caption_mlp = nn.Sequential(
                nn.Linear(config.caption_dim, d), nn.ReLU(), nn.Linear(d, d),
            )
            # tag: multi-label (<=3/item), embedding riêng rồi mean trên tag THẬT
            self.tag_embedding = nn.Embedding(config.num_tags, d, padding_idx=TAG_PAD)
            self.fuse = nn.Linear(2 * d, d)

        self._last_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        video_idx: torch.Tensor,
        caption_embedding: torch.Tensor,
        tag_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Φ_i. ID-only: chỉ tra bảng ID; ID + content: cộng thêm nhánh content (xem docstring module)."""
        e = self.id_embedding(video_idx)
        if self.config.use_content:
            valid = (tag_ids != TAG_PAD).unsqueeze(-1).to(e.dtype)
            e_tag = (self.tag_embedding(tag_ids) * valid).sum(dim=-2) / valid.sum(dim=-2).clamp(min=1.0)
            e = e + self.fuse(torch.cat([self.caption_mlp(caption_embedding), e_tag], dim=-1))
        self._last_stats = {"norm_content": e.detach().norm(dim=-1)}
        return e

    def dense_parameters(self) -> list[nn.Parameter]:
        """MỌI tham số — không còn chia sparse/dense."""
        return list(self.parameters())
