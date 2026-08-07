"""
Two-Tower Model — ghép ItemTower + UserTower, xem Readme.md mục "Loss function — InfoNCE
multi-negative".

Score = user_vector . item_vector.
Loss  = InfoNCE multi-negative, K=8 mỗi positive (4 hard-negative rating<=2 thật +
        4 soft-negative cùng category/popularity-decile, xem preprocess.py).
Thay thế hoàn toàn BPR/BCE 1:1:1 trước đó — không tính loss pairwise riêng lẻ.
"""

import torch
import torch.nn.functional as F

from item_tower import ItemTower
from user_tower import UserTower


class TwoTowerModel(torch.nn.Module):
    def __init__(self, item_tower_kwargs, user_tower_kwargs):
        super().__init__()
        self.item_tower = ItemTower(**item_tower_kwargs)
        self.user_tower = UserTower(**user_tower_kwargs)

    def encode_user(self, item_embs, ratings, mask, static_features, category_distribution):
        return self.user_tower(item_embs, ratings, mask, static_features, category_distribution)

    def encode_item(self, text_emb, image_emb, has_image=None):
        return self.item_tower(text_emb, image_emb, has_image)

    def forward(self, user_batch, pos_item_emb, neg_item_emb):
        """
        user_batch    : dict với các key khớp tham số encode_user (item_embs, ratings, mask,
                        static_features, category_distribution)
        pos_item_emb  : tuple (text_emb, image_emb, has_image), mỗi cái [batch, dim] hoặc
                        [batch] — item_pos, 1 mỗi user
        neg_item_emb  : tuple (text_emb, image_emb, has_image), mỗi cái [batch, K, dim] hoặc
                        [batch, K] — K negative mỗi user (K=8 đã chốt)

        Trả về: user_vector [batch, out_dim], pos_vector [batch, out_dim],
                neg_vectors [batch, K, out_dim]
        """
        user_vector = self.encode_user(**user_batch)  # [batch, out_dim]

        pos_text, pos_image, pos_has_image = pos_item_emb
        pos_vector = self.encode_item(pos_text, pos_image, pos_has_image)  # [batch, out_dim]

        neg_text, neg_image, neg_has_image = neg_item_emb
        batch, k, dim_t = neg_text.shape
        dim_i = neg_image.shape[-1]
        neg_flat = self.encode_item(
            neg_text.reshape(batch * k, dim_t),
            neg_image.reshape(batch * k, dim_i),
            neg_has_image.reshape(batch * k),
        )
        neg_vectors = neg_flat.reshape(batch, k, -1)  # [batch, K, out_dim]

        return user_vector, pos_vector, neg_vectors


def info_nce_loss(user_vector, pos_vector, neg_vectors, temperature=0.1):
    """InfoNCE multi-negative đã chốt: 1 positive + K negative (hard+soft trộn sẵn ở
    neg_vectors, xem preprocess.py) trong cùng 1 softmax.

    user_vector : [batch, dim]
    pos_vector  : [batch, dim]
    neg_vectors : [batch, K, dim]
    """
    pos_score = (user_vector * pos_vector).sum(dim=-1, keepdim=True) / temperature  # [batch, 1]
    neg_scores = torch.einsum("bd,bkd->bk", user_vector, neg_vectors) / temperature  # [batch, K]

    logits = torch.cat([pos_score, neg_scores], dim=1)  # [batch, 1+K], vị trí 0 = positive
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
