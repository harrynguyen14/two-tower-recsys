"""
User Tower — xem Readme.md mục "User tower" + "Kiến trúc: 1 pipeline duy nhất với masking"
+ "Static/Pseudo-Static Features cho User Tower" + "max_seq_len cho pad/truncate".

1 pipeline duy nhất, không chia nhánh cứng theo độ dài sequence — thích ứng N=0,1,>1 qua
masking. max_seq_len=10 đã chốt (coverage 93.34%, xem Readme).

Static features theo đúng thứ tự ưu tiên đã chốt:
  - user_mean_rating, user_std_rating (ưu tiên cao nhất — autocorrelation r=0.341)
  - total_reviews_count, helpful_votes_mean (trung bình)
  - user_avg_page_count (trung bình, coverage 93.3%)
  - category_distribution (yếu, chỉ bổ trợ — 46.9% lặp lại, gần random)
Đã loại: verified_purchase_ratio, time-gap, hour-of-day, weekday, price/description-based
(xem Readme bảng "Static/Pseudo-Static Features").
"""

import torch
import torch.nn as nn

from utils import embedding_dim_rule_of_thumb


class SequenceEncoder(nn.Module):
    """Self-attention pooling có masking trên sequence (item_content_emb, rating).

    N=0 (toàn PAD)  -> output = 0 vector, static+context quyết định 100% E_U.
    N=1             -> attention trên 1 token = chính token đó (softmax 1 phần tử).
    N>1             -> attention thật, trích intent dài hạn.
    Cùng 1 forward path cho mọi N — không rẽ nhánh if/else theo độ dài.
    """

    def __init__(self, item_emb_dim, rating_emb_dim=8, hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.rating_proj = nn.Linear(1, rating_emb_dim)
        token_dim = item_emb_dim + rating_emb_dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # learned pooling query

    def forward(self, item_embs, ratings, mask):
        """
        item_embs : [batch, N, item_emb_dim]  (đã pad 0 ở vị trí PAD)
        ratings   : [batch, N]                (đã pad 0 ở vị trí PAD)
        mask      : [batch, N] bool            True = vị trí thật, False = PAD

        Trả về: [batch, hidden_dim]. Nếu N=0 toàn False -> trả vector 0.
        """
        batch_size, n, _ = item_embs.shape
        rating_emb = self.rating_proj(ratings.unsqueeze(-1))  # [batch, N, rating_emb_dim]
        tokens = self.input_proj(torch.cat([item_embs, rating_emb], dim=-1))  # [batch, N, hidden]

        has_any = mask.any(dim=1)  # [batch] — user có ít nhất 1 interaction thật
        # key_padding_mask: True = bỏ qua vị trí đó. Với user N=0 (mask toàn False),
        # key_padding_mask sẽ toàn True -> MultiheadAttention lỗi "no valid position";
        # tránh bằng cách mở 1 vị trí giả cho các user đó rồi zero-out kết quả sau.
        key_padding_mask = ~mask
        key_padding_mask[~has_any, 0] = False

        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, hidden]
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)  # [batch, hidden]

        pooled = pooled * has_any.unsqueeze(-1).float()  # N=0 -> ép về vector 0 thật
        return pooled


class UserTower(nn.Module):
    def __init__(
        self,
        item_emb_dim,
        n_static_features,
        category_vocab_size,
        seq_hidden_dim=128,
        n_heads=4,
        category_emb_dim=None,
        out_dim=128,
        mlp_hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(
            item_emb_dim, hidden_dim=seq_hidden_dim, n_heads=n_heads, dropout=dropout
        )

        # category_distribution là vector tần suất trên toàn vocab category (yếu, chỉ bổ
        # trợ — xem Readme), không phải 1 category-ID đơn -> không cần nn.Embedding lookup,
        # chỉ cần projection tuyến tính từ vector tần suất [category_vocab_size].
        if category_emb_dim is None:
            category_emb_dim = embedding_dim_rule_of_thumb(category_vocab_size)
        self.category_proj = nn.Linear(category_vocab_size, category_emb_dim)

        fusion_dim = seq_hidden_dim + n_static_features + category_emb_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_dim),
        )

    def forward(self, item_embs, ratings, mask, static_features, category_distribution):
        """
        item_embs             : [batch, N, item_emb_dim]
        ratings                : [batch, N]
        mask                   : [batch, N] bool
        static_features        : [batch, n_static_features]  (đã chuẩn hoá, xem preprocess.py)
        category_distribution  : [batch, category_vocab_size]
        """
        seq_vec = self.sequence_encoder(item_embs, ratings, mask)  # [batch, seq_hidden_dim]
        cat_vec = self.category_proj(category_distribution)  # [batch, category_emb_dim]
        fused = torch.cat([seq_vec, static_features, cat_vec], dim=-1)
        return self.fusion_mlp(fused)  # [batch, out_dim]
