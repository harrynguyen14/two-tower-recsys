"""
Ranking stage — chạy SAU retrieval (TwoTowerModel), KHÔNG thay thế nó.

Retrieval (two_tower_model.py): encode_user và encode_item ĐỘC LẬP, score = dot-product —
rẻ, ANN-friendly, xếp hạng được cả catalog. Nhưng vì 2 tower không nhìn thấy nhau lúc
encode, nó không học được tương tác bậc cao user×item cụ thể (vd "item hiếm + user thích
sách hiếm" là tín hiệu chỉ lộ ra khi so sánh TRỰC TIẾP 2 bên).

RankerModel giải quyết đúng chỗ đó: nhận thẳng sequence lịch sử THÔ (không phải user_vector
đã pool 1 lần như UserTower.sequence_encoder) + 1 candidate cụ thể, dùng TargetAttention
kiểu DIN (Zhou et al., KDD'18 "Deep Interest Network for CTR Prediction") — mỗi candidate
có 1 phép attention RIÊNG lên lịch sử, trọng số phụ thuộc độ liên quan giữa chính candidate
đó và từng item lịch sử. Khác SequenceEncoder (query cố định, học 1 cách pool DUY NHẤT
dùng chung cho mọi candidate).

Dùng LẠI ItemTower (encode candidate, fine-tune tiếp) + UserTower.category_proj (category
embedding) — không train from scratch toàn bộ, xem main() ghép loss.

Data: candidate list = retrieval top-N THẬT (topk_over_catalog, đã có sẵn) + ground-truth
nếu chưa nằm trong top-N — không phải sample ngẫu nhiên như negative của retrieval, vì
ranker phải học rerank ĐÚNG cái mà retrieval sẽ thực sự đưa lên (train/serve nhất quán).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAttention(nn.Module):
    """DIN activation unit: attention_score(candidate, seq[j]) học bằng MLP trên
    [candidate, seq[j], candidate*seq[j], candidate-seq[j]] — biểu diễn mạnh hơn dot-product
    thuần (SequenceEncoder dùng dot-product ngầm qua nn.MultiheadAttention chuẩn), vì nó có
    thể học quan hệ phi tuyến giữa 2 vector thay vì chỉ độ tương đồng hướng.

    Query LÀ candidate (thay đổi theo từng candidate) — khác SequenceEncoder.query
    (nn.Parameter cố định, học 1 cách pool chung cho mọi trường hợp)."""

    def __init__(self, item_emb_dim, rating_emb_dim=8, hidden_dim=64):
        super().__init__()
        self.rating_proj = nn.Linear(1, rating_emb_dim)
        token_dim = item_emb_dim + rating_emb_dim
        self.activation_mlp = nn.Sequential(
            nn.Linear(token_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_emb, seq_item_embs, seq_ratings, mask):
        """
        candidate_emb : [batch, N, dim]        N candidate mỗi user (listwise)
        seq_item_embs : [batch, L, dim]        L = max_seq_len, item_vec đã qua ItemTower
        seq_ratings   : [batch, L]
        mask          : [batch, L] bool         True = vị trí thật

        Trả về: [batch, N, dim] — 1 vector "interest" riêng cho MỖI candidate.
        """
        batch, n_cand, dim = candidate_emb.shape
        _, seq_len, _ = seq_item_embs.shape

        rating_emb = self.rating_proj(seq_ratings.unsqueeze(-1))  # [batch, L, rating_dim]
        seq_tokens = torch.cat([seq_item_embs, rating_emb], dim=-1)  # [batch, L, token_dim]

        # rating_emb của candidate không có ý nghĩa (candidate chưa được rate) -> pad 0,
        # giữ token_dim khớp seq_tokens để concat/subtract broadcast được.
        cand_rating_pad = torch.zeros(batch, n_cand, rating_emb.size(-1),
                                      device=candidate_emb.device, dtype=candidate_emb.dtype)
        cand_tokens = torch.cat([candidate_emb, cand_rating_pad], dim=-1)  # [batch, N, token_dim]

        # Broadcast N candidate x L seq -> [batch, N, L, token_dim] để tính activation unit
        # cho MỌI cặp (candidate, seq_item) cùng lúc, không vòng lặp Python.
        cand_exp = cand_tokens.unsqueeze(2).expand(-1, -1, seq_len, -1)
        seq_exp = seq_tokens.unsqueeze(1).expand(-1, n_cand, -1, -1)
        pair = torch.cat([cand_exp, seq_exp, cand_exp * seq_exp, cand_exp - seq_exp], dim=-1)

        scores = self.activation_mlp(pair).squeeze(-1)  # [batch, N, L]

        # Mask vị trí PAD — cùng kỹ thuật torch.finfo(dtype).min thay -inf đã chốt ở
        # two_tower_model.info_nce_loss (an toàn dưới cả fp16/fp32 autocast).
        neg_inf = torch.finfo(scores.dtype).min
        pad_mask = (~mask).unsqueeze(1).expand(-1, n_cand, -1)  # [batch, N, L]
        scores = scores.masked_fill(pad_mask, neg_inf)

        # User N=0 (mask toàn False): mọi vị trí bị mask -> softmax của toàn -inf là NaN.
        # Mở 1 vị trí giả (cùng thủ thuật SequenceEncoder.forward) rồi zero-out kết quả sau.
        has_any = mask.any(dim=1)  # [batch]
        scores = scores.clone()
        scores[~has_any, :, 0] = 0.0

        weights = F.softmax(scores, dim=-1)  # [batch, N, L]
        interest = torch.einsum("bnl,bld->bnd", weights, seq_item_embs)  # [batch, N, dim]
        interest = interest * has_any.view(batch, 1, 1).float()  # N=0 -> vector 0 thật
        return interest


class RankerModel(nn.Module):
    """Nhận seq thô + N candidate cùng lúc (listwise), trả về 1 score/candidate.

    item_tower/user_tower: TRUYỀN VÀO instance đã train của TwoTowerModel (không tạo mới) —
    fine-tune tiếp cùng ranker (gradient của loss ranking chảy ngược vào cả 2 tower qua
    candidate_vec và category_proj). Xem main() để biết cách cân bằng với loss retrieval
    gốc (multi-task loss, tránh catastrophic forgetting mục tiêu retrieval)."""

    def __init__(self, item_tower, user_tower, item_emb_dim, n_static_features,
                 category_emb_dim, rating_emb_dim=8, attn_hidden_dim=64,
                 mlp_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.target_attn = TargetAttention(item_emb_dim, rating_emb_dim, attn_hidden_dim)

        # fused = interest(dim) + candidate_vec(dim) + static(n_static) + category(cat_emb)
        fused_dim = item_emb_dim * 2 + n_static_features + category_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def forward(self, item_embs, ratings, mask, static_features,
                category_distribution, candidate_text, candidate_image, candidate_has_image):
        """
        item_embs, ratings, mask : CÙNG TÊN THAM SỐ với UserTower.forward/make_collate
            (item_embs=[batch,L,dim], ratings=[batch,L]) — sequence THÔ, không pool. Đặt
            trùng tên có chủ đích: user_batch từ make_ranker_collate() có thể forward
            thẳng bằng **user_batch mà không cần đổi tên key nào.
        static_features, category_distribution : như UserTower.
        candidate_text/image/has_image : [batch, N, dim] hoặc [batch, N] (has_image) — N
            candidate/user cùng lúc (từ retrieval top-N + ground-truth, xem main()).

        Trả về: [batch, N] logit thô (chưa softmax) — dùng trực tiếp với listwise_loss.
        """
        batch, n_cand, dim_t = candidate_text.shape
        dim_i = candidate_image.shape[-1]
        cand_flat = self.item_tower(
            candidate_text.reshape(batch * n_cand, dim_t),
            candidate_image.reshape(batch * n_cand, dim_i),
            candidate_has_image.reshape(batch * n_cand),
        )
        candidate_vec = cand_flat.reshape(batch, n_cand, -1)  # [batch, N, item_out_dim]

        interest = self.target_attn(candidate_vec, item_embs, ratings, mask)  # [batch, N, dim]

        cat_vec = self.user_tower.category_proj(category_distribution)  # [batch, cat_emb_dim]
        cat_vec = cat_vec.unsqueeze(1).expand(-1, n_cand, -1)
        static_exp = static_features.unsqueeze(1).expand(-1, n_cand, -1)

        fused = torch.cat([interest, candidate_vec, static_exp, cat_vec], dim=-1)
        return self.mlp(fused).squeeze(-1)  # [batch, N]


def listwise_loss(scores, gt_positions):
    """ListNet-style rút gọn: mỗi user có ĐÚNG 1 ground-truth trong candidate list ->
    softmax cross-entropy trên list, CÙNG CÔNG THỨC với two_tower_model.info_nce_loss
    (list ứng viên khác nguồn — retrieval top-N thật, không phải sample ngẫu nhiên — nhưng
    cơ chế toán học giống hệt: đẩy xác suất khối lượng về đúng 1 vị trí nhãn).

    scores      : [batch, N] logit thô
    gt_positions: [batch] long — index của ground-truth trong N candidate (xem main() để
                  biết cách chèn ground-truth vào candidate list khi nó không nằm sẵn
                  trong top-N của retrieval).
    """
    return F.cross_entropy(scores, gt_positions)


def _self_check():
    """Assert-based check cho TargetAttention/RankerModel/listwise_loss — logic non-trivial
    (broadcast 4 chiều, masking N=0, gradient chảy ngược vào tower dùng chung) nên cần ít
    nhất 1 test chạy được thay vì chỉ tin code.

    Chạy: python -X utf8 ranker.py"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from item_tower import ItemTower
    from user_tower import UserTower

    torch.manual_seed(0)
    ok = []

    BATCH, L, N_CAND, ITEM_DIM = 4, 6, 5, 16
    TEXT_DIM, IMG_DIM, N_STATIC, N_CAT, CAT_EMB = 8, 8, 4, 3, 6

    item_tower = ItemTower(text_dim=TEXT_DIM, image_dim=IMG_DIM, out_dim=ITEM_DIM,
                           gmu_hidden_dim=32, mlp_hidden_dim=32)
    user_tower = UserTower(item_emb_dim=ITEM_DIM, n_static_features=N_STATIC,
                           category_vocab_size=N_CAT, category_emb_dim=CAT_EMB,
                           seq_hidden_dim=32, mlp_hidden_dim=32)
    ranker = RankerModel(item_tower, user_tower, item_emb_dim=ITEM_DIM,
                        n_static_features=N_STATIC, category_emb_dim=CAT_EMB,
                        attn_hidden_dim=16, mlp_hidden_dim=32)

    def make_batch(mask_all_false=False):
        item_embs = torch.randn(BATCH, L, ITEM_DIM)
        ratings = torch.randint(1, 6, (BATCH, L)).float()
        if mask_all_false:
            mask = torch.zeros(BATCH, L, dtype=torch.bool)
        else:
            mask = torch.ones(BATCH, L, dtype=torch.bool)
            mask[0, 3:] = False  # user 0: chỉ 3 vị trí thật (N>1 nhưng có PAD)
            mask[1, 1:] = False  # user 1: chỉ 1 vị trí thật (N=1)
        static_features = torch.randn(BATCH, N_STATIC)
        category_distribution = torch.rand(BATCH, N_CAT)
        category_distribution = category_distribution / category_distribution.sum(-1, keepdim=True)
        candidate_text = torch.randn(BATCH, N_CAND, TEXT_DIM)
        candidate_image = torch.randn(BATCH, N_CAND, IMG_DIM)
        candidate_has_image = torch.ones(BATCH, N_CAND, dtype=torch.bool)
        return dict(item_embs=item_embs, ratings=ratings, mask=mask,
                    static_features=static_features, category_distribution=category_distribution,
                    candidate_text=candidate_text, candidate_image=candidate_image,
                    candidate_has_image=candidate_has_image)

    # ── 1. shape đúng, không NaN/Inf ─────────────────────────────────────────
    batch = make_batch()
    scores = ranker(**batch)
    assert scores.shape == (BATCH, N_CAND), f"shape sai: {scores.shape}"
    assert torch.isfinite(scores).all(), "scores chứa NaN/Inf"
    ok.append("forward shape [batch, N_CAND] đúng, không NaN/Inf")

    # ── 2. N=0 hoàn toàn (mọi user mask toàn False) không NaN, không crash ──
    batch_empty = make_batch(mask_all_false=True)
    scores_empty = ranker(**batch_empty)
    assert torch.isfinite(scores_empty).all(), "user N=0 gây NaN/Inf (softmax toàn -inf?)"
    ok.append("user N=0 (mask toàn False) không NaN/Inf")

    # ── 3. TargetAttention: seq bị mask KHÔNG được đóng góp vào interest ─────
    attn = TargetAttention(ITEM_DIM, rating_emb_dim=8, hidden_dim=16)
    candidate_emb = torch.randn(1, 2, ITEM_DIM)
    seq_item_embs = torch.randn(1, 4, ITEM_DIM)
    seq_ratings = torch.randint(1, 6, (1, 4)).float()
    mask_partial = torch.tensor([[True, True, False, False]])
    out_partial = attn(candidate_emb, seq_item_embs, seq_ratings, mask_partial)

    seq_item_embs_changed = seq_item_embs.clone()
    seq_item_embs_changed[0, 2:] = torch.randn(2, ITEM_DIM) * 100  # đổi mạnh vị trí PAD
    out_changed = attn(candidate_emb, seq_item_embs_changed, seq_ratings, mask_partial)
    assert torch.allclose(out_partial, out_changed, atol=1e-5), (
        "thay đổi giá trị ở vị trí PAD làm output đổi -> mask không chặn đúng")
    ok.append("TargetAttention: vị trí PAD không ảnh hưởng tới interest output")

    # ── 4. gradient chảy ngược vào item_tower/user_tower dùng chung ─────────
    ranker.zero_grad()
    batch2 = make_batch()
    scores2 = ranker(**batch2)
    labels = torch.zeros(BATCH, dtype=torch.long)
    loss = listwise_loss(scores2, labels)
    loss.backward()
    item_grad_norm = sum(p.grad.abs().sum().item() for p in item_tower.parameters() if p.grad is not None)
    user_grad_norm = sum(p.grad.abs().sum().item() for p in user_tower.category_proj.parameters()
                        if p.grad is not None)
    assert item_grad_norm > 0, "gradient KHÔNG chảy vào item_tower — fine-tune joint sẽ không hoạt động"
    assert user_grad_norm > 0, "gradient KHÔNG chảy vào user_tower.category_proj"
    ok.append("gradient chảy ngược vào item_tower và user_tower.category_proj dùng chung")

    # ── 5. listwise_loss: label đúng vị trí -> loss thấp hơn hẳn label sai ──
    scores_fixed = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    loss_correct = listwise_loss(scores_fixed, torch.tensor([0]))
    loss_wrong = listwise_loss(scores_fixed, torch.tensor([1]))
    assert loss_correct.item() < loss_wrong.item(), "loss không phạt đúng vị trí label sai"
    ok.append("listwise_loss: label khớp vị trí điểm cao nhất cho loss thấp hơn hẳn")

    for line in ok:
        print(f"  {line}  OK")
    print(f"ALL PASS ({len(ok)} nhóm assert)")


if __name__ == "__main__":
    _self_check()
