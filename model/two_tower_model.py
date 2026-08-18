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

        # L2-normalize trước khi trả về — InfoNCE/contrastive chuẩn dùng cosine similarity
        # (dot product của vector ĐÃ normalize), không phải dot product thô trên vector MLP
        # output tuỳ ý norm. Thiếu normalize: (1) dot_product/temperature (temperature=0.1,
        # rất nhỏ) có thể vượt dải float16 (~65504) khi norm vector lớn -> NaN ngay cả
        # KHÔNG có in_batch_neg (bug gốc thật, độc lập với bug -inf ở info_nce_loss); (2) độ
        # lớn embedding không kiểm soát khiến loss scale không ổn định giữa các batch.
        #
        # eps=1e-6 THAY VÌ mặc định F.normalize (1e-12) — bug NaN thật thứ 3 đã gặp (grad_norm
        # vẫn nhỏ/bình thường ngay trước khi NaN, loại trừ gradient explosion): 1e-12 bị làm
        # tròn về 0.0 TUYỆT ĐỐI trong float16 (torch.finfo(float16).tiny ~6.1e-5) — dưới
        # torch.autocast, F.normalize(x, eps=1e-12) với x là vector gần-0 thật (vd
        # user_vector của user N=0 lịch sử, xem SequenceEncoder "pooled = pooled * has_any"
        # ép về 0) tính ra x / (norm + 0.0) = NaN. eps=1e-6 vẫn biểu diễn được an toàn ở fp16
        # (> tiny) nên phép chia luôn có mẫu số khác 0 thật.
        user_vector = F.normalize(user_vector, dim=-1, eps=1e-6)
        pos_vector = F.normalize(pos_vector, dim=-1, eps=1e-6)
        neg_vectors = F.normalize(neg_vectors, dim=-1, eps=1e-6)

        return user_vector, pos_vector, neg_vectors


def info_nce_loss(user_vector, pos_vector, neg_vectors, temperature=0.1, use_in_batch_neg=False,
                  log_q=None, pos_idx=None):
    """InfoNCE multi-negative đã chốt: 1 positive + K negative (hard+soft trộn sẵn ở
    neg_vectors, xem preprocess.py) trong cùng 1 softmax.

    user_vector : [batch, dim]
    pos_vector  : [batch, dim]
    neg_vectors : [batch, K, dim]

    use_in_batch_neg: thêm pos_vector CỦA SAMPLE KHÁC trong cùng batch làm negative bổ
    sung (kỹ thuật chuẩn của NVIDIA Merlin two-tower) — tận dụng item_pos đã encode sẵn
    trong forward pass, không tốn thêm CPU sample/GPU encode nào. Chỉ 1 phép nhân ma trận
    [batch,dim] @ [dim,batch] có sẵn, mask bỏ đường chéo (chính nó không phải negative
    của chính nó). Giữ nguyên hard/soft-negative hiện có — đây là TÍN HIỆU BỔ SUNG, không
    thay thế (in-batch neg thường "dễ" hơn hard-neg nên không đủ để học cold-start 1 mình).

    log_q, pos_idx: SAMPLED SOFTMAX CORRECTION cho in-batch negative (bắt buộc về mặt lý
    thuyết khi use_in_batch_neg=True — Bengio & Senécal 2008; Yi et al. 2019 "Sampling-Bias-
    Corrected Neural Modeling for Large Corpus Item Recommendations", chính là cách NVIDIA
    Merlin làm).
      log_q  : [n_items] float — log xác suất một item được lấy làm in-batch negative,
               ước lượng bằng log(count_train(i) / tổng_interaction). None = tắt correction.
      pos_idx: [batch] long — index item_pos của từng sample, CÙNG KHÔNG GIAN INDEX với
               log_q (xem cảnh báo index ở main()).

    VÌ SAO CẦN: in-batch negative lấy pos_vector của sample khác trong batch, mà batch được
    sample từ dòng interaction thật -> xác suất một item làm negative TỈ LỆ VỚI ĐỘ PHỔ BIẾN
    của nó. Sách bán chạy bị đẩy ra xa hàng nghìn lần mỗi epoch, sách hiếm gần như không bao
    giờ. Softmax khi đó xấp xỉ P(i|u) ∝ exp(s) * 1/Q(i), tức model bị dạy để DÌM item phổ
    biến — ngược hẳn mục tiêu retrieval. Trừ log Q(i) khỏi logit khử đúng phần thiên lệch đó:
        s'(u,i) = s(u,i)/T - log Q(i)
    Với cold-start còn có lợi phụ: item cold có Q rất nhỏ -> -log Q dương lớn -> logit của
    item cold được NÂNG lên, đúng nhóm item dự án này quan tâm.

    CHỈ áp cho in-batch. hard/soft-negative KHÔNG được sửa: soft-neg lấy bằng torch.randint
    UNIFORM nên Q là hằng số, trừ một hằng vào mọi logit thì softmax bất biến (cross_entropy
    chỉ phụ thuộc hiệu các logit) — sửa thêm chỉ là no-op tốn tính toán. hard-neg lấy từ
    lịch sử rating<=2 của chính user, không phải phân phối sampling nên log Q không định
    nghĩa được ở đó.

    Correction áp lên CỘT (item bị làm negative), không phải hàng: in_batch_scores[i][j] là
    score của user i với item_pos của sample j, nên phải trừ log_q của item j -> trừ theo
    vector hàng broadcast trên dim=1. Trừ theo hàng (log_q của item i) là sai và sẽ tự triệt
    tiêu trong softmax vì nó là hằng số trên mỗi hàng.

    Vị trí positive (cột 0) KHÔNG trừ: nó không phải mẫu sampled từ Q, nó là nhãn thật."""
    pos_score = (user_vector * pos_vector).sum(dim=-1, keepdim=True) / temperature  # [batch, 1]
    neg_scores = torch.einsum("bd,bkd->bk", user_vector, neg_vectors) / temperature  # [batch, K]

    logits = [pos_score, neg_scores]
    if use_in_batch_neg:
        batch = user_vector.size(0)
        in_batch_scores = (user_vector @ pos_vector.t()) / temperature  # [batch, batch]

        if log_q is not None:
            if pos_idx is None:
                raise ValueError("log_q cần pos_idx để biết item nào ứng với mỗi cột in-batch")
            # [batch] -> broadcast trên dim=1 (theo CỘT j = item_pos của sample j)
            lq = log_q.to(in_batch_scores.device, in_batch_scores.dtype)[pos_idx]
            in_batch_scores = in_batch_scores - lq.unsqueeze(0)

        diag_mask = torch.eye(batch, dtype=torch.bool, device=user_vector.device)
        # float("-inf") thật gây train_loss=nan dưới torch.autocast (float16 không biểu
        # diễn -inf ổn định qua log_softmax nội bộ của cross_entropy — bug thật đã gặp).
        # Dùng torch.finfo(dtype).min (số âm hữu hạn lớn nhất có thể) thay thế — vẫn đủ để
        # softmax gán xác suất ~0 cho vị trí bị mask, nhưng an toàn dưới cả fp16 lẫn fp32.
        neg_inf = torch.finfo(in_batch_scores.dtype).min
        in_batch_scores = in_batch_scores.masked_fill(diag_mask, neg_inf)
        logits.append(in_batch_scores)

    logits = torch.cat(logits, dim=1)  # [batch, 1+K(+batch)], vị trí 0 = positive
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
