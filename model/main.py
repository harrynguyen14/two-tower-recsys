"""
Entrypoint — ghép preprocess -> TwoTowerModel -> training loop -> metrics.

Chạy: python main.py --item-emb-dir <path> [--epochs N --batch-size N ...]
(xem myargs.py cho toàn bộ options).

--item-emb-dir trỏ đến thư mục chứa text_embeddings.npy/image_embeddings.npy/
has_image.npy/asin_to_idx.json (xuất bởi model/preprocess_data/data.py, xem Readme mục
"Item tower — kiến trúc GMU"). Sequence trong user-tower dùng item_vector ĐÃ QUA ItemTower
(không phải raw text/image emb) — item_vector của mọi item được encode 1 lần qua ItemTower
lúc khởi động (xem PrecomputedItemVectors dưới), tương tự cách 1 embedding-table thường
được "freeze" trong lúc dùng làm input sequence cho tower khác.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from myargs import parse_args
from preprocess import (
    load_reviews_by_user,
    load_item_meta,
    compute_cutoffs,
    compute_static_features,
)
from item_tower import ItemEmbeddingStore
from two_tower_model import TwoTowerModel, info_nce_loss
from ranker import RankerModel, listwise_loss
from ranking_metrics import topk_over_catalog, ranking_metrics_at_k, format_ranking_report
from utils import (
    pad_and_mask,
    fit_static_scaler,
    transform_static,
    N_STATIC_FEATURES,
)

# Dùng cho slot exclude rỗng trong run_retrieval_eval — cấp phát 1 lần thay vì mỗi user.
EMPTY_EXCLUDE = np.empty(0, dtype=np.int64)


def raw_static_vec(static, category_vocab_size):
    """(static dict | None) -> (list[float] THÔ 4 phần tử, cat_dist).

    Thứ tự khớp utils.LOG1P_FEATURES: user_mean_rating, user_std_rating,
    user_total_reviews, user_avg_page_count.

    static=None (user N=0 hoàn toàn) -> vector 0 THÔ. Lưu ý: 0 thô sau khi transform sẽ
    KHÔNG còn là 0 (thành -mean/std) — đúng ngữ nghĩa "user này có total_reviews=0", vì
    sau chuẩn hoá giá trị 0 phải nằm đúng vị trí tương đối của nó trong phân phối."""
    if static is None:
        return [0.0] * N_STATIC_FEATURES, [0.0] * category_vocab_size
    return [
        static["user_mean_rating"],
        static["user_std_rating"],
        static["user_total_reviews"],
        static["user_avg_page_count"],
    ], static["category_distribution"]


class PrecomputedItemVectors:
    """Encode TOÀN BỘ item qua ItemTower (eval mode, không cần gradient) — dùng làm input
    sequence cho user-tower. Item_pos/hard-neg/soft-neg vẫn đi qua ItemTower "sống" trong
    forward pass thật (gradient chảy qua GMU) — chỉ sequence lịch sử là "đóng băng" theo
    tương tự cách một embedding-table thường được dùng làm input cho tower khác.

    BẮT BUỘC gọi .refresh(item_tower, item_emb_store) sau MỖI epoch (xem main()) —
    item_tower vẫn tiếp tục học suốt training trong khi self.vectors chỉ là snapshot tại 1
    thời điểm. Nếu không refresh, user_vector (dựa vào self.vectors cũ) và pos/neg_vector
    (dựa vào item_tower mới nhất) lệch pha ngày càng xa qua các epoch — model học cách "bù
    trừ" cho snapshot cố định thay vì học content thật (bug thật đã gặp: val AUC trong lúc
    train ~0.62 nhưng eval lại bằng snapshot MỚI tính từ checkpoint đã train thì rớt về
    ~0.52, gần random — xem thảo luận "PrecomputedItemVectors đóng băng")."""

    def __init__(self, item_tower, item_emb_store, device, batch_size=256):
        self.device = device
        self.asin_to_idx = {a: i for i, a in enumerate(sorted(item_emb_store.asin_to_idx.keys()))}
        self._all_asins = sorted(item_emb_store.asin_to_idx.keys())
        self._batch_size = batch_size
        self.refresh(item_tower, item_emb_store)

    def refresh(self, item_tower, item_emb_store):
        """Tính lại self.vectors bằng weight HIỆN TẠI của item_tower — gọi sau mỗi epoch."""
        was_training = item_tower.training
        item_tower.eval()
        vectors = []
        with torch.no_grad():
            for start in tqdm(range(0, len(self._all_asins), self._batch_size),
                               desc="Precompute item vectors", leave=False, position=1):
                batch_asins = self._all_asins[start:start + self._batch_size]
                text_emb, image_emb, has_image = item_emb_store.get(batch_asins)
                vec = item_tower(text_emb.to(self.device), image_emb.to(self.device),
                                  has_image.to(self.device))
                vectors.append(vec)  # giữ trên device — không .cpu() (xem docstring lớp)
        self.vectors = torch.cat(vectors, dim=0)  # [n_items, item_out_dim], trên device
        if was_training:
            item_tower.train()  # trả lại đúng mode — trước đây THIẾU dòng này, kẹt vĩnh
                                 # viễn ở eval() sau lần gọi đầu (bug thật, tắt luôn
                                 # modality_dropout suốt training dù model.train() đã gọi)

    @property
    def dim(self):
        return self.vectors.shape[1]

    def get(self, asins):
        idx = torch.tensor([self.asin_to_idx[a] for a in asins], dtype=torch.long, device=self.device)
        return self.vectors[idx]


class ColdStartDataset(Dataset):
    """1 sample = 1 positive (user, item_pos, rating>=4) + sequence lịch sử trước item_pos
    + N_HARD_NEG hard-negative (rating<=2 thật của chính user đó). Soft-negative KHÔNG còn
    sample ở đây nữa — chuyển sang uniform random trong collate_fn (xem make_collate), theo
    đúng cách NVIDIA Merlin/Mixed Negative Sampling (Yang et al., WWW'20) làm: uniform random
    negative + in-batch negative, KHÔNG dùng rejection sampling category/popularity-decile
    per-sample. Lý do đổi (bug hiệu năng thật đã gặp): sample_soft_negative dù đã tối ưu
    (rejection sampling, không sort/set) vẫn là Python loop per-sample chạy N_SOFT_NEG lần
    MỖI sample — với num_workers=0 bắt buộc (GPU embedding, xem ItemEmbeddingStore), đây là
    bottleneck chính khiến 1 epoch ~6h dù model rất nhẹ (GPU util chỉ ~6%). Merlin đạt tốc độ
    cao vì KHÔNG làm rejection sampling có điều kiện — chấp nhận negative "thô" hơn (uniform
    + in-batch) đổi lấy vector hoá hoàn toàn, không còn vòng lặp Python nào trong đường train.

    user_seq_fn(uid) -> list[(ts, asin, rating, helpful_vote)], static_features_fn(uid) ->
    dict|None: callable, nhận DatasetAccessor.user_seq/static_features_of khi có
    --dataset-dir, hoặc by_user.__getitem__/static_features.get khi tự quét JSONL — cùng 1
    Dataset dùng được cho cả 2 nguồn."""

    def __init__(self, positives, static_features_fn, category_vocab_size,
                 item_emb_store, precomputed_item_vectors, n_hard_neg, seed=0,
                 static_scaler=None):
        self.positives = positives  # list[(uid, item_pos, seq_before, hard_neg_pool, label)]
        self.static_features_fn = static_features_fn
        self.item_emb_store = item_emb_store
        self.precomputed_item_vectors = precomputed_item_vectors
        self.category_vocab_size = category_vocab_size
        self.n_hard_neg = n_hard_neg
        self.rng = random.Random(seed)
        self.static_scaler = static_scaler  # fit trên TRAIN, dùng cho mọi split (Lỗi 3)

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        uid, item_pos, seq_before, hard_neg_pool, label, _ts = self.positives[idx]

        seq_asins = [asin for _, asin, _, _ in seq_before]
        seq_items = list(self.precomputed_item_vectors.get(seq_asins)) if seq_asins else []
        seq_ratings = [r for _, _, r, _ in seq_before]

        raw, cat_dist = raw_static_vec(self.static_features_fn(uid), self.category_vocab_size)
        static_vec = transform_static(raw, self.static_scaler)

        # Chỉ lấy hard-negative THỰC CÓ, KHÔNG lấp thiếu bằng item_pos.
        #
        # BUG THẬT ĐÃ SỬA (nguyên nhân chính của AUC~0.52): code cũ lấp slot thiếu bằng
        # chính item_pos ("fallback cực hiếm"). Đo trên dữ liệu thật (Kindle_Store, 491k
        # user): 81.8% user sinh positive KHÔNG có review nào rating<=2, và sau khi lọc
        # hard-neg theo train_temporal_boundary (xem build_positives_from_array) tỷ lệ này lên 82.3%
        # — "cực hiếm" thực chất là đa số tuyệt đối. Phân phối rating giải thích: user
        # Amazon chấm điểm rất cao (5.0: 2.47M, 4.0: 0.91M, 2.0: 0.15M, 1.0: 0.11M).
        #
        # Hậu quả: neg_vectors chứa 4 BẢN SAO của pos_vector, nên softmax của InfoNCE có 5
        # logit giống hệt nhau trong khi label trỏ vào vị trí 0 -> gradient kéo user về
        # phía positive 1 lần và ĐẨY RA XA 4 lần. Loss không thể xuống dưới log(5)=1.609
        # (đo thực tế: 4.157), và trên 82% dữ liệu model bị dạy để QUÊN chính positive của
        # nó -> AUC gần random bất kể kiến trúc tốt đến đâu.
        #
        # Sửa: trả về số hard-neg thực có + số slot cần lấp; collate_fn lấp bằng uniform
        # random (torch.randint, cùng cơ chế soft-negative) nên vẫn vector hoá hoàn toàn,
        # không thêm vòng lặp Python nào vào đường train. K=8 giữ nguyên cố định -> shape
        # tensor không đổi, info_nce_loss không cần sửa gì. Đây đúng là Mixed Negative
        # Sampling của NVIDIA Merlin: hard-negative khi có, uniform random cho phần còn lại.
        hard_negs = self.rng.sample(hard_neg_pool, min(len(hard_neg_pool), self.n_hard_neg))
        n_random_fill = self.n_hard_neg - len(hard_negs)

        return {
            "seq_items": seq_items,
            "seq_ratings": seq_ratings,
            "static": static_vec,
            "cat_dist": cat_dist,
            "pos_asin": item_pos,
            "hard_neg_asins": hard_negs,  # có thể NGẮN HƠN n_hard_neg (xem n_random_fill)
            "n_random_fill": n_random_fill,  # số slot collate_fn phải lấp bằng random
            "label": label,  # "cold" | "warm" — chỉ dùng lúc eval (xem Readme Temporal split)
        }


def make_collate(item_emb_store, max_seq_len, item_out_dim, n_soft_neg, device=None):
    """n_soft_neg: số uniform-random negative thêm vào MỖI sample, sample VECTOR HOÁ 1 lần
    cho cả batch bằng torch.randint (xem lý do đổi từ sample_soft_negative per-sample sang
    uniform random ở docstring ColdStartDataset — kỹ thuật NVIDIA Merlin/Mixed Negative
    Sampling). Không loại trừ item_pos/user_seen khi sample (khác hẳn rejection sampling cũ)
    — chấp nhận hiếm khi trúng false negative, đổi lấy vector hoá hoàn toàn: xác suất trùng
    ~n_soft_neg/n_items cực nhỏ với catalog lớn (Merlin cũng chấp nhận đánh đổi này)."""
    def collate(batch):
        seq_items = [b["seq_items"] for b in batch]
        seq_ratings = [b["seq_ratings"] for b in batch]
        # vector_dim=item_out_dim tường minh (không đoán qua sample không rỗng) — batch có
        # thể toàn user N=0 lịch sử (mọi seq_items rỗng), lúc đó không còn sample nào để suy
        # luận dim, dẫn tới item_embs 2D thay vì 3D và crash ở SequenceEncoder (bug thật).
        # device=device: seq_items là Tensor GPU (precomputed_item_vectors giữ trên GPU),
        # padded/mask phải cùng device để torch.stack không lỗi mismatch.
        item_embs, mask = pad_and_mask(seq_items, max_seq_len, vector_dim=item_out_dim, device=device)
        ratings, _ = pad_and_mask(seq_ratings, max_seq_len, device=device)

        static_features = torch.tensor([b["static"] for b in batch], dtype=torch.float32, device=device)
        cat_dist = torch.tensor([b["cat_dist"] for b in batch], dtype=torch.float32, device=device)

        pos_asins = [b["pos_asin"] for b in batch]
        pos_text, pos_image, pos_has_image = item_emb_store.get(pos_asins)
        pos_item_emb = (pos_text, pos_image, pos_has_image)
        # pos_idx: index item_pos trong KHÔNG GIAN INDEX CỦA item_emb_store (asin_to_idx từ
        # JSON) — dùng cho sampled softmax correction, phải khớp không gian của log_q. Đây
        # cũng đúng không gian mà soft-negative randint(0, n_items) đang dùng.
        pos_idx = torch.as_tensor([item_emb_store.asin_to_idx[a] for a in pos_asins],
                                  dtype=torch.long, device=device)

        # Hard-negative: vẫn 1 lệnh get() cho toàn bộ batch*n_hard_neg (đã tối ưu trước đó,
        # xem comment gốc — mỗi get() riêng lẻ tạo tensor + index nhân với batch_size).
        #
        # hard_neg_asins giờ có thể NGẮN HƠN n_hard_neg (Dataset không lấp bằng item_pos
        # nữa — xem ColdStartDataset.__getitem__). Lấp phần thiếu bằng uniform random index
        # ở ĐÂY: gom mọi slot thiếu của cả batch vào 1 lệnh torch.randint duy nhất, rồi
        # ghép vào đúng vị trí -> vẫn không có vòng lặp Python nào trên đường tensor.
        batch_size = len(batch)
        n_hard_neg = len(batch[0]["hard_neg_asins"]) + batch[0]["n_random_fill"]

        flat_hard_asins = [a for b in batch for a in b["hard_neg_asins"]]
        total_fill = sum(b["n_random_fill"] for b in batch)

        if flat_hard_asins:
            h_text, h_image, h_has = item_emb_store.get(flat_hard_asins)
        else:
            h_text = h_image = h_has = None

        if total_fill > 0:
            fill_idx = torch.randint(0, item_emb_store.n_items, (total_fill,),
                                     device=item_emb_store.device)
            f_text, f_image, f_has = item_emb_store.get_by_idx(fill_idx)

        # Ghép hard thực + random fill theo ĐÚNG thứ tự từng sample: sample i chiếm đúng
        # n_hard_neg slot liên tiếp trong buffer phẳng, trong đó len(hard_neg_asins) slot
        # đầu là hard thực và n_random_fill slot sau là random. Ghép sai thứ tự ở đây sẽ
        # trộn negative giữa các user (đúng loại bug đã gặp với view(batch,k,-1), xem dưới).
        if n_hard_neg == 0:
            # --n-hard-neg 0 (thuần soft-negative): không có slot hard nào, cả h_* lẫn f_*
            # đều None -> phải bỏ qua hẳn phần hard, nếu không .view() trên None sẽ crash
            # (bug thật do chính bản sửa Lỗi 1 tạo ra — code cũ luôn lấp đủ slot nên không
            # bao giờ gặp nhánh này).
            hard_text = hard_image = hard_has_image = None
        elif total_fill == 0:
            hard_text, hard_image, hard_has_image = h_text, h_image, h_has
        elif not flat_hard_asins:
            hard_text, hard_image, hard_has_image = f_text, f_image, f_has
        else:
            dim_t, dim_i = h_text.shape[-1], h_image.shape[-1]
            hard_text = torch.empty(batch_size * n_hard_neg, dim_t,
                                    dtype=h_text.dtype, device=h_text.device)
            hard_image = torch.empty(batch_size * n_hard_neg, dim_i,
                                     dtype=h_image.dtype, device=h_image.device)
            hard_has_image = torch.empty(batch_size * n_hard_neg,
                                         dtype=h_has.dtype, device=h_has.device)
            h_pos = f_pos = 0
            for i, b in enumerate(batch):
                n_real = len(b["hard_neg_asins"])
                n_fill = b["n_random_fill"]
                base = i * n_hard_neg
                if n_real:
                    hard_text[base:base + n_real] = h_text[h_pos:h_pos + n_real]
                    hard_image[base:base + n_real] = h_image[h_pos:h_pos + n_real]
                    hard_has_image[base:base + n_real] = h_has[h_pos:h_pos + n_real]
                    h_pos += n_real
                if n_fill:
                    s, e = base + n_real, base + n_real + n_fill
                    hard_text[s:e] = f_text[f_pos:f_pos + n_fill]
                    hard_image[s:e] = f_image[f_pos:f_pos + n_fill]
                    hard_has_image[s:e] = f_has[f_pos:f_pos + n_fill]
                    f_pos += n_fill

        # Soft-negative: uniform random index vào TOÀN BỘ catalog, sample 1 lần cho cả batch
        # bằng torch.randint — không còn vòng lặp Python/rejection sampling nào (khác biệt
        # cốt lõi so với sample_soft_negative cũ, xem docstring hàm).
        soft_idx = torch.randint(0, item_emb_store.n_items, (batch_size * n_soft_neg,),
                                  device=item_emb_store.device)
        soft_text, soft_image, soft_has_image = item_emb_store.get_by_idx(soft_idx)

        # Reshape RIÊNG từng phần về (batch, n_x, dim) trước khi cat theo dim=1 (chiều K) —
        # cat trực tiếp theo dim=0 rồi view(batch, k, -1) sẽ sai thứ tự (buffer phẳng
        # [tất cả hard của mọi sample][tất cả soft của mọi sample] không tương ứng
        # 1-1 với [sample_i: hard rồi soft] mà view ngầm giả định — bug thật phát hiện khi
        # tự kiểm tra lại logic, chưa từng chạy).
        soft_text = soft_text.view(batch_size, n_soft_neg, -1)
        soft_image = soft_image.view(batch_size, n_soft_neg, -1)
        soft_has_image = soft_has_image.view(batch_size, n_soft_neg)

        if n_hard_neg == 0:
            neg_item_emb = (soft_text, soft_image, soft_has_image)
        else:
            hard_text = hard_text.view(batch_size, n_hard_neg, -1)
            hard_image = hard_image.view(batch_size, n_hard_neg, -1)
            hard_has_image = hard_has_image.view(batch_size, n_hard_neg)
            neg_item_emb = (
                torch.cat([hard_text, soft_text], dim=1),  # [batch, n_hard+n_soft, dim]
                torch.cat([hard_image, soft_image], dim=1),
                torch.cat([hard_has_image, soft_has_image], dim=1),
            )

        labels = [b["label"] for b in batch]

        user_batch = dict(
            item_embs=item_embs, ratings=ratings, mask=mask,
            static_features=static_features, category_distribution=cat_dist,
        )
        return user_batch, pos_item_emb, neg_item_emb, labels, pos_idx

    return collate


class RankerDataset(Dataset):
    """1 sample = 1 positive (giống ColdStartDataset) + candidate list = retrieval top-N
    THẬT (đã đóng băng cho cả epoch, xem build_ranker_candidates) + ground-truth chèn vào
    nếu chưa nằm sẵn trong top-N.

    KHÁC ColdStartDataset ở điểm cốt lõi: negative ở đây KHÔNG sample ngẫu nhiên/theo
    category — nó là chính những gì retrieval sẽ thực sự đưa lên hàng đầu, để ranker học
    rerank đúng cái nó sẽ thấy lúc serving (train/serve nhất quán). Nếu dùng sample ngẫu
    nhiên như ColdStartDataset, ranker sẽ học phân biệt "positive vs item ngẫu nhiên"
    (quá dễ, không giúp gì cho việc rerank trong top-N đã lọc).

    candidate_idx: dict[int -> np.ndarray[N]] — positive_row_idx -> N candidate index
    (không gian asin_to_item_idx, xem build_ranker_candidates), đã gán sẵn TRƯỚC khi
    Dataset này được tạo (đóng băng trong suốt epoch, giống PrecomputedItemVectors)."""

    def __init__(self, positives, candidate_idx, static_features_fn, category_vocab_size,
                 precomputed_item_vectors, asin_to_item_idx, all_asins, static_scaler=None):
        self.positives = positives
        self.candidate_idx = candidate_idx
        self.static_features_fn = static_features_fn
        self.category_vocab_size = category_vocab_size
        self.precomputed_item_vectors = precomputed_item_vectors
        # asin_to_item_idx (KHÔNG PHẢI item_emb_store.asin_to_idx) — candidate_idx được
        # build_ranker_candidates gán theo namespace SORTED của all_asins (namespace của
        # item_matrix/topk_over_catalog), phải tra ground-truth bằng ĐÚNG bảng này để khớp
        # (xem cảnh báo index ở main() — 2 bảng asin_to_idx khác nhau thật khi JSON không sorted).
        self.asin_to_item_idx = asin_to_item_idx
        self.all_asins = all_asins  # all_asins[i] = asin có index i trong candidate_idx
        self.static_scaler = static_scaler

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        uid, item_pos, seq_before, _hard_neg_pool, _label, _ts = self.positives[idx]

        seq_asins = [asin for _, asin, _, _ in seq_before]
        seq_items = list(self.precomputed_item_vectors.get(seq_asins)) if seq_asins else []
        seq_ratings = [r for _, _, r, _ in seq_before]

        raw, cat_dist = raw_static_vec(self.static_features_fn(uid), self.category_vocab_size)
        static_vec = transform_static(raw, self.static_scaler)

        cand_idx = self.candidate_idx[idx]  # np.ndarray[N], build_ranker_candidates đã
                                             # ĐẢM BẢO chứa ground-truth (chèn nếu thiếu)
        cand_asins = [self.all_asins[i] for i in cand_idx]
        gt_global_idx = self.asin_to_item_idx.get(item_pos, -1)
        match = np.nonzero(cand_idx == gt_global_idx)[0]
        assert len(match) > 0, (
            f"item_pos={item_pos!r} không nằm trong candidate list dù build_ranker_candidates "
            "đã đảm bảo chèn — kiểm tra build_ranker_candidates có dùng ĐÚNG asin_to_item_idx"
            " (không phải item_emb_store.asin_to_idx) để chèn ground-truth hay không")
        gt_position = int(match[0])

        return {
            "seq_items": seq_items, "seq_ratings": seq_ratings, "static": static_vec,
            "cat_dist": cat_dist, "cand_asins": cand_asins, "gt_position": gt_position,
        }


def build_ranker_candidates(positives, model, precomputed_item_vectors, item_matrix,
                            static_features_fn, category_vocab_size, static_scaler,
                            asin_to_item_idx, max_seq_len, item_out_dim,
                            n_candidates, device, batch_size=256):
    """Chạy retrieval THẬT cho mỗi positive trong `positives` -> top-N candidate, chèn
    ground-truth nếu retrieval chưa xếp nó vào top-N. Trả về list[np.ndarray[N]] (1 phần
    tử/positive, index theo asin_to_item_idx — cùng không gian item_matrix/all_asins).

    Gọi 1 LẦN MỖI EPOCH (đầu epoch, sau khi item_matrix đã encode lại với item_tower mới
    nhất, xem main()) — cùng nguyên tắc đóng băng như PrecomputedItemVectors: candidate
    list cố định trong suốt epoch, KHÔNG tính lại mỗi step (topk_over_catalog trên 1.59M
    item mỗi step sẽ làm ranker chậm ngang retrieval eval — quá đắt cho vòng train).

    Encode user dùng ĐÚNG cùng static/category pipeline như RetrievalEvalSet/make_collate
    (bỏ qua sẽ cho candidate SAI hẳn so với retrieval thật, vì fusion_mlp nhận
    cat([seq_vec, static_features, cat_vec]) — static/category ảnh hưởng trực tiếp
    user_vector, không thể bỏ qua)."""
    model.eval()
    candidate_idx = []
    with torch.no_grad():
        for start in tqdm(range(0, len(positives), batch_size), desc="Build ranker candidates",
                         leave=False, position=1):
            chunk = positives[start:start + batch_size]
            seq_items_list, seq_ratings_list, static_list, cat_list, seen_list = [], [], [], [], []
            for uid, _item_pos, seq_before, _hp, _lb, _ts in chunk:
                seq_asins = [asin for _, asin, _, _ in seq_before]
                seq_items_list.append(list(precomputed_item_vectors.get(seq_asins)) if seq_asins else [])
                seq_ratings_list.append([r for _, _, r, _ in seq_before])
                raw, cat_dist = raw_static_vec(static_features_fn(uid), category_vocab_size)
                static_list.append(transform_static(raw, static_scaler))
                cat_list.append(cat_dist)
                # Loại item đã xem khỏi candidate — NHẤT QUÁN với run_retrieval_eval (dùng
                # `exclude`, xem RetrievalEvalSet docstring): không thể "khuyến nghị lại" cái
                # user đã đọc, và ranker phải học rerank ĐÚNG cái retrieval thật sẽ đưa lên
                # lúc serving (cùng candidate distribution, xem docstring ranker.py).
                seen_list.append(np.array(
                    [asin_to_item_idx[a] for a in seq_asins if a in asin_to_item_idx],
                    dtype=np.int64))

            item_embs, mask = pad_and_mask(seq_items_list, max_seq_len, vector_dim=item_out_dim, device=device)
            ratings, _ = pad_and_mask(seq_ratings_list, max_seq_len, device=device)
            static_features = torch.tensor(static_list, dtype=torch.float32, device=device)
            cat_dist = torch.tensor(cat_list, dtype=torch.float32, device=device)

            user_vec = F.normalize(model.encode_user(
                item_embs=item_embs, ratings=ratings, mask=mask,
                static_features=static_features, category_distribution=cat_dist,
            ), dim=-1, eps=1e-6)

            topk = topk_over_catalog(user_vec, item_matrix, n_candidates, exclude=seen_list)
            topk_np = topk.cpu().numpy()

            for row, (_uid, item_pos, *_rest) in zip(topk_np, chunk):
                gt_idx = asin_to_item_idx.get(item_pos)
                # item_pos KHÔNG có trong asin_to_item_idx (embedding catalog thiếu ASIN
                # này — có thể do interactions/item embedding không đồng bộ 100%) phải FAIL
                # NGAY ở đây với thông báo rõ ràng. Nếu bỏ qua (gt_idx=None -> không chèn),
                # lỗi sẽ lan tới RankerDataset.__getitem__ và crash bằng AssertionError khó
                # truy nguồn gốc (không rõ vì sao ground-truth "biến mất", xem đối chiếu
                # cold_item_mask ở main() — cùng loại rủi ro dữ liệu thiếu đồng bộ).
                if gt_idx is None:
                    raise SystemExit(
                        f"item_pos={item_pos!r} không nằm trong item_emb_store (embedding "
                        "catalog) — kiểm tra --item-emb-dir có khớp với dataset interactions "
                        "hay không (build_ranker_candidates cần MỌI positive có embedding)."
                    )
                if gt_idx not in row:
                    # Ground-truth chưa lọt top-N retrieval — CHÈN vào thay vì bỏ, nếu không
                    # gt_position (RankerDataset) sẽ không tìm thấy nhãn và crash. Ghi đè
                    # slot CUỐI (thấp điểm nhất trong top-N) — mất 1 candidate "khó" nhất,
                    # chấp nhận được vì tần suất thấp (retrieval Recall@100 thường >0 sau
                    # vài epoch, xem README).
                    row = row.copy()
                    row[-1] = gt_idx
                candidate_idx.append(row)
    return candidate_idx


def gather_candidate_tensors(item_emb_store, cand_asins_per_row):
    """cand_asins_per_row: list[batch] của list[N] asin string, N candidate/row -> tensor
    (text, image, has_image) mỗi cái [batch, N, dim] hoặc [batch, N] (has_image).

    Dùng chung cho make_ranker_collate và run_ranker_eval — cả 2 đều cần đúng thao tác
    flatten asin -> item_emb_store.get -> reshape lại (batch, N, dim), khác nhau chỉ ở
    nguồn cand_asins_per_row (list asin sẵn có vs. tra ngược từ index topk_over_catalog)."""
    batch_size = len(cand_asins_per_row)
    n_candidates = len(cand_asins_per_row[0])
    flat_asins = [a for row in cand_asins_per_row for a in row]
    c_text, c_image, c_has = item_emb_store.get(flat_asins)
    return (
        c_text.view(batch_size, n_candidates, -1),
        c_image.view(batch_size, n_candidates, -1),
        c_has.view(batch_size, n_candidates),
    )


def make_ranker_collate(item_emb_store, max_seq_len, item_out_dim, device=None):
    """Collate cho RankerDataset — giống make_collate ở phần seq/static/category, khác ở
    phần candidate: N candidate/user (không phải 1 positive + K negative sample riêng)."""
    def collate(batch):
        seq_items = [b["seq_items"] for b in batch]
        seq_ratings = [b["seq_ratings"] for b in batch]
        item_embs, mask = pad_and_mask(seq_items, max_seq_len, vector_dim=item_out_dim, device=device)
        ratings, _ = pad_and_mask(seq_ratings, max_seq_len, device=device)

        static_features = torch.tensor([b["static"] for b in batch], dtype=torch.float32, device=device)
        cat_dist = torch.tensor([b["cat_dist"] for b in batch], dtype=torch.float32, device=device)

        candidate_text, candidate_image, candidate_has_image = gather_candidate_tensors(
            item_emb_store, [b["cand_asins"] for b in batch])

        gt_positions = torch.tensor([b["gt_position"] for b in batch], dtype=torch.long, device=device)

        user_batch = dict(
            item_embs=item_embs, ratings=ratings, mask=mask,
            static_features=static_features, category_distribution=cat_dist,
        )
        return user_batch, (candidate_text, candidate_image, candidate_has_image), gt_positions

    return collate


def build_positives(by_user, train_temporal_boundary, val_temporal_boundary, first_seen_by_item, split, max_seq_len):
    """split: "train" | "val" | "test". Trả về list[(uid, item_pos, seq_before,
    hard_neg_pool, cold_warm_label)]. cold_warm_label chỉ có ý nghĩa ngoài "train".

    Dùng khi KHÔNG có dataset build sẵn (--dataset-dir không set) — tự quét by_user từ đầu.
    Xem build_positives_from_array() cho version dùng dataset build sẵn (nhanh hơn, xem
    Readme)."""
    positives = []
    for uid, seq in by_user.items():
        for i, (ts, asin, rating, _) in enumerate(seq):
            if split == "train" and ts > train_temporal_boundary:
                continue
            if split == "val" and not (train_temporal_boundary < ts <= val_temporal_boundary):
                continue
            if split == "test" and ts <= val_temporal_boundary:
                continue
            if rating < 4:
                continue

            # Cùng 2 bản sửa như build_positives_from_array (Lỗi 4 + Lỗi 2, xem giải thích
            # đầy đủ ở đó): seq[:i] đã đảm bảo "trước positive" nên bỏ hẳn điều kiện
            # train_temporal_boundary; hard_neg_pool lọc theo split.
            seq_before = seq[:i][-max_seq_len:]
            neg_cutoff = train_temporal_boundary if split == "train" else ts
            hard_neg_pool = [a for (t, a, r, _) in seq if r <= 2 and t < neg_cutoff]

            label = None
            if split != "train":
                first_seen = first_seen_by_item.get(asin)
                label = "cold" if (first_seen is None or first_seen > train_temporal_boundary) else "warm"

            # ts đi kèm (phần tử thứ 6) để RetrievalEvalSet chọn được seq_before của
            # positive MUỘN NHẤT — xem giải thích ở đó (Lỗi 6).
            positives.append((uid, asin, seq_before, hard_neg_pool, label, ts))
    return positives


class DatasetAccessor:
    """Bọc toàn bộ dataset array-based (xuất bởi model/preprocess_data/build_dataset.py) sau
    1 interface giống hệt dict-based cũ (by_user[uid], category_leaf[asin], decile_of[asin],
    static_features[uid]) — để ColdStartDataset/sample_soft_negative dùng chung code cho cả
    2 nguồn (dataset build sẵn hoặc quét trực tiếp), không cần viết 2 phiên bản Dataset riêng.

    Không load gì vào dict Python — mọi lookup là slice/index trực tiếp trên mảng NumPy đã
    mmap-load, tránh lặp lại lỗi MemoryError khi chuyển ngược array -> dict cho 5.6M user."""

    def __init__(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        self.metadata = np.load(dataset_dir / "metadata.npy", allow_pickle=True).item()
        self.user_vocab = np.load(dataset_dir / "user_vocab.npy", allow_pickle=True)
        self.asin_vocab = np.load(dataset_dir / "asin_vocab.npy", allow_pickle=True)
        self.user_to_idx = {u: i for i, u in enumerate(self.user_vocab)}
        self.asin_to_idx = {a: i for i, a in enumerate(self.asin_vocab)}

        self.user_review_sequence = np.load(dataset_dir / "user_review_sequence.npy")
        self.user_review_offsets = np.load(dataset_dir / "user_review_offsets.npy")
        self.item_category_idx = np.load(dataset_dir / "item_category_idx.npy")
        self.item_popularity_decile = np.load(dataset_dir / "item_popularity_decile.npy")
        self.user_static_features = np.load(dataset_dir / "user_static_features.npy")
        self.user_category_distribution = np.load(dataset_dir / "user_category_distribution.npy")

        # Tương thích ngược: dataset build bằng bản build_dataset.py CŨ (trước khi đổi tên
        # field mean_rating/std_rating/total_reviews/avg_page_count -> thêm prefix "user_")
        # lưu user_static_features.npy KHÔNG có prefix, và còn thừa field
        # helpful_votes_mean (đã bỏ khỏi feature set — xem docstring user_tower.py, train/
        # serve skew). Dò field name có mặt thay vì giả định 1 kiểu cố định.
        names = self.user_static_features.dtype.names
        self._static_field = {
            "mean_rating": "user_mean_rating" if "user_mean_rating" in names else "mean_rating",
            "std_rating": "user_std_rating" if "user_std_rating" in names else "std_rating",
            "total_reviews": "user_total_reviews" if "user_total_reviews" in names else "total_reviews",
            "avg_page_count": "user_avg_page_count" if "user_avg_page_count" in names else "avg_page_count",
        }

        self.category_vocab = self.metadata["category_vocab"]
        # Tương thích ngược: dataset build bằng bản build_dataset.py CŨ (trước khi đổi tên
        # biến train_cutoff/val_cutoff -> train_temporal_boundary/val_temporal_boundary) lưu
        # metadata.npy với key cũ. Không build lại dataset (tốn hàng giờ) — chỉ đọc theo cả
        # 2 tên key, ưu tiên tên mới nếu có.
        self.train_temporal_boundary = self.metadata.get(
            "train_temporal_boundary", self.metadata.get("train_cutoff"))
        self.val_temporal_boundary = self.metadata.get(
            "val_temporal_boundary", self.metadata.get("val_cutoff"))
        if self.train_temporal_boundary is None or self.val_temporal_boundary is None:
            raise KeyError(
                "metadata.npy thiếu cả train_temporal_boundary/val_temporal_boundary lẫn "
                "train_cutoff/val_cutoff (tên key cũ) — dataset có thể build từ phiên bản "
                "build_dataset.py khác/hỏng."
            )

    def user_seq(self, uid):
        """Trả về list[(timestamp, asin, rating, helpful_vote)] — CÙNG ĐỊNH DẠNG với
        by_user[uid] cũ, để ColdStartDataset không cần biết nguồn dữ liệu."""
        uidx = self.user_to_idx[uid]
        start, end = self.user_review_offsets[uidx], self.user_review_offsets[uidx + 1]
        rows = self.user_review_sequence[start:end]
        return [(int(r["timestamp"]), self.asin_vocab[r["item_idx"]], float(r["rating"]),
                 int(r["helpful_vote"])) for r in rows]

    def first_seen_by_item_idx(self):
        """Trả về [n_asin] int64 — timestamp NHỎ NHẤT của mỗi item trên TOÀN BỘ lịch sử
        (kể cả sau cutoff), theo index của asin_vocab. Item chưa từng xuất hiện = iinfo.max.

        Dùng để dựng cold_item_mask ĐÚNG (first_seen > train_temporal_boundary) — xem main(). Trước đây
        mask được suy từ item_pos của val_cold/test_cold_pos, tức CHỈ gồm item cold mà có ai
        đó rate >=4; mọi item cold không ai thích bị loại khỏi pool, nên candidate pool toàn
        là ground-truth của một user nào đó — không còn distractor nào, metric cold-pool cao
        giả tạo và không so được với paper cold-start.

        np.minimum.at thay vòng Python: đã đo trên 2M dòng — 0.74s (dict) -> 0.01s (55x),
        và cho kết quả KHỚP TUYỆT ĐỐI với cách dict-based mà build_dataset.py dùng."""
        fs = np.full(len(self.asin_vocab), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(fs, self.user_review_sequence["item_idx"],
                      self.user_review_sequence["timestamp"])
        return fs

    def category_of(self, asin):
        return self.category_vocab[self.item_category_idx[self.asin_to_idx[asin]]]

    def decile_of_asin(self, asin):
        d = int(self.item_popularity_decile[self.asin_to_idx[asin]])
        return d if d > 0 else None

    def static_features_of(self, uid):
        """Trả về dict CÙNG SHAPE với static_features[uid] cũ, hoặc None nếu user N=0
        hoàn toàn (user_total_reviews=0, xem build_dataset.py build_user_static_features).

        Đọc field qua self._static_field (dò tên field thật ở __init__) — tương thích cả
        dataset build bằng bản build_dataset.py cũ (field không prefix "user_")."""
        uidx = self.user_to_idx[uid]
        row = self.user_static_features[uidx]
        f = self._static_field
        if int(row[f["total_reviews"]]) == 0:
            return None
        return {
            "user_mean_rating": float(row[f["mean_rating"]]),
            "user_std_rating": float(row[f["std_rating"]]),
            "user_total_reviews": int(row[f["total_reviews"]]),
            "user_avg_page_count": float(row[f["avg_page_count"]]),
            "category_distribution": self.user_category_distribution[uidx].tolist(),
        }


def build_positives_from_array(interactions_arr, dataset, max_seq_len, label=None,
                                split="train"):
    """Version dùng dataset build sẵn: interactions_arr là structured array đã lọc sẵn theo
    split (xuất bởi build_dataset.py — KHÔNG còn cột label, xem QUYẾT ĐỊNH (3) trong
    build_dataset.py). dataset: DatasetAccessor — mọi lookup qua slice/index NumPy, không
    dựng lại dict cho 5.6M user (tránh MemoryError, xem DatasetAccessor).

    label: "cold" | "warm" | None — gán CỐ ĐỊNH cho toàn bộ interactions_arr thay vì đọc
    từ mảng, vì giờ warm/cold đã được build_dataset.py tách sẵn thành 2 file riêng
    (val_warm_interactions.npy / val_cold_interactions.npy) — truyền None khi gọi cho
    train_interactions.npy hoặc val/test_interactions.npy đầy đủ (không cần label).

    Cache dataset.user_seq(uid) theo uid trong dict cục bộ — nhiều dòng trong interactions_arr
    thuộc cùng 1 user (mỗi user thường có nhiều review), nếu không cache thì user_seq() bị
    tính lại (slice + list-comprehension) mỗi dòng, biến vòng lặp 17M dòng train thành hàng
    giờ. seq bất biến trong suốt hàm này (dataset không thay đổi giữa các dòng) nên cache an
    toàn."""
    positives = []
    user_seq_cache = {}
    for row in tqdm(interactions_arr, desc="Building positives"):
        uid = dataset.user_vocab[int(row["user_idx"])]
        asin = dataset.asin_vocab[int(row["item_idx"])]
        ts = int(row["timestamp"])
        seq = user_seq_cache.get(uid)
        if seq is None:
            seq = dataset.user_seq(uid)
            user_seq_cache[uid] = seq
        # BUG THẬT ĐÃ SỬA (Lỗi 4 — sequence val/test bị cắt cụt tại train_temporal_boundary):
        # code cũ có thêm điều kiện "s[0] <= dataset.train_temporal_boundary", khiến với 1 positive ở
        # val (ts > train_temporal_boundary) mọi tương tác nằm GIỮA cutoff và ts bị loại — dù chúng
        # xảy ra TRƯỚC positive nên hoàn toàn hợp lệ về nhân quả (dùng chúng KHÔNG phải
        # leakage). Hệ quả: mọi user val/test bị đánh giá bằng lịch sử cũ, thường là rỗng
        # (N=0) -> user_vector sụp về chỉ còn static features. Giữ đúng 1 điều kiện nhân
        # quả duy nhất: s[0] < ts.
        #
        # Ràng buộc train_temporal_boundary VẪN GIỮ cho static_features (fit trên train, xem
        # compute_static_features) — chỗ đó chống leakage là đúng, không đổi.
        seq_before = [s for s in seq if s[0] < ts][-max_seq_len:]

        # BUG THẬT ĐÃ SỬA (Lỗi 2 — hard-negative rò rỉ qua ranh giới temporal split):
        # code cũ lấy "[a for (_,a,r,_) in seq if r <= 2]" trên TOÀN BỘ lịch sử user, không
        # lọc thời gian, nên sample train nhìn thấy cả item ở giai đoạn val/test. Kiểm
        # chứng: seq=[(100,'A',5),(200,'B',1),(900,'C',2)] với train_temporal_boundary=500 cho pool
        # ['B','C'] — 'C' là tương tác tương lai.
        #
        # Điều kiện đúng khác nhau theo split:
        #   train    : ts <= train_temporal_boundary (chỉ negative trong giai đoạn train)
        #   val/test : ts < ts_positive   (mọi thứ trước positive đều hợp lệ nhân quả,
        #              cùng nguyên tắc với seq_before ở trên)
        neg_cutoff = dataset.train_temporal_boundary if split == "train" else ts
        hard_neg_pool = [a for (t, a, r, _) in seq if r <= 2 and t < neg_cutoff]

        positives.append((uid, asin, seq_before, hard_neg_pool, label, ts))
    return positives


def to_device(item_emb_tuple, device):
    return tuple(t.to(device, non_blocking=True) for t in item_emb_tuple)


try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML_HANDLE = None


def gpu_util_str(device):
    """util%/VRAM dùng cho tqdm postfix — GPU util THẬT (nvidia-smi qua pynvml) khi có,
    ngược lại fallback về VRAM allocated (torch.cuda.memory_allocated, luôn sẵn có,
    không cần pynvml) — vẫn hữu ích để thấy pipeline có nghẽn CPU hay không (VRAM đứng
    yên trong lúc train = GPU đang chờ batch, xem thảo luận GPU 0% dù model đang chạy)."""
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    if _NVML_HANDLE is not None:
        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return f"{util.gpu}% {mem_gb:.2f}GB"
    return f"{mem_gb:.2f}GB (cài pynvml để xem % util)"


class RetrievalEvalSet(Dataset):
    """1 dòng = 1 USER (không phải 1 interaction) + ground-truth đã GOM của user đó.

    Khác hẳn ColdStartDataset (1 dòng = 1 positive + negative đi kèm): retrieval eval xếp
    hạng toàn catalog nên không cần negative nào, và phải gom mọi positive của cùng 1 user
    lại thành 1 ground-truth set. Đo thật trên val: 57% user chỉ có 1 positive nhưng max
    tới 630 (mean 4.61) — nếu giữ 1 dòng/interaction thì user 630-positive được tính 630
    lần trong macro-average, làm lệch hẳn metric về phía các user nặng.

    seen: item user đã có trong sequence lịch sử (loại khỏi candidate lúc xếp hạng — không
    thể "khuyến nghị lại" cái user đã đọc, xem ranking_metrics docstring)."""

    def __init__(self, positives, static_features_fn, category_vocab_size,
                 precomputed_item_vectors, asin_to_item_idx, static_scaler=None):
        by_uid = {}
        for uid, item_pos, seq_before, _hard_pool, _label, ts in positives:
            entry = by_uid.get(uid)
            if entry is None:
                by_uid[uid] = {"seq": seq_before, "ts": ts, "gt": set()}
                entry = by_uid[uid]
            elif ts > entry["ts"]:
                # BUG THẬT ĐÃ SỬA (Lỗi 6 — heuristic "seq dài nhất" là no-op): code cũ so
                # len(seq_before) để lấy "bản dài nhất, ứng với positive muộn nhất". Nhưng
                # seq_before đã bị cắt [-max_seq_len:] (mặc định 10), nên với user có >=10
                # tương tác trước positive ĐẦU TIÊN thì MỌI positive đều cho len == 10 ->
                # nhánh elif không bao giờ chạy, và bản được giữ là bản gặp đầu tiên = positive
                # SỚM NHẤT, tức ngược hẳn ý định. Đã đo: 3 positive -> len [10,10,10], seq
                # dừng ở A14 thay vì A24.
                #
                # So theo ts của chính positive mới đúng "muộn nhất" trong mọi trường hợp.
                entry["seq"] = seq_before
                entry["ts"] = ts
            idx = asin_to_item_idx.get(item_pos)
            if idx is not None:
                entry["gt"].add(idx)

        self.rows = [(uid, v["seq"], v["gt"]) for uid, v in by_uid.items()]
        self.static_features_fn = static_features_fn
        self.category_vocab_size = category_vocab_size
        self.precomputed_item_vectors = precomputed_item_vectors
        self.asin_to_item_idx = asin_to_item_idx
        self.static_scaler = static_scaler  # PHẢI là scaler fit trên train, giống train set

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        uid, seq_before, gt = self.rows[idx]

        seq_asins = [asin for _, asin, _, _ in seq_before]
        seq_items = list(self.precomputed_item_vectors.get(seq_asins)) if seq_asins else []
        seq_ratings = [r for _, _, r, _ in seq_before]

        raw, cat_dist = raw_static_vec(self.static_features_fn(uid), self.category_vocab_size)
        static_vec = transform_static(raw, self.static_scaler)

        seen = [self.asin_to_item_idx[a] for a in seq_asins if a in self.asin_to_item_idx]

        return {"seq_items": seq_items, "seq_ratings": seq_ratings, "static": static_vec,
                "cat_dist": cat_dist, "gt": gt, "seen": seen}


def make_retrieval_collate(max_seq_len, item_out_dim, device=None):
    """Collate cho RetrievalEvalSet — chỉ build user_batch, không pos/neg item nào
    (retrieval eval xếp hạng toàn catalog, xem RetrievalEvalSet)."""
    def collate(batch):
        item_embs, mask = pad_and_mask([b["seq_items"] for b in batch], max_seq_len,
                                       vector_dim=item_out_dim, device=device)
        ratings, _ = pad_and_mask([b["seq_ratings"] for b in batch], max_seq_len, device=device)
        static_features = torch.tensor([b["static"] for b in batch], dtype=torch.float32, device=device)
        cat_dist = torch.tensor([b["cat_dist"] for b in batch], dtype=torch.float32, device=device)
        user_batch = dict(item_embs=item_embs, ratings=ratings, mask=mask,
                          static_features=static_features, category_distribution=cat_dist)
        return user_batch, [b["gt"] for b in batch], [b["seen"] for b in batch]
    return collate


def encode_all_items(item_tower, item_emb_store, all_asins, device, batch_size=1024):
    """Encode toàn catalog qua ItemTower (eval mode) + L2-normalize — ma trận candidate cho
    retrieval eval. 1.59M x 128 fp32 = 0.81 GB, vừa VRAM thoải mái.

    L2-normalize ở ĐÂY là bắt buộc: two_tower_model.forward normalize cả user và item nên
    score lúc train là cosine. Nếu candidate matrix không normalize, thứ hạng sẽ bị chi phối
    bởi ĐỘ LỚN vector thay vì hướng -> metric không khớp mục tiêu train."""
    was_training = item_tower.training
    item_tower.eval()
    vectors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(all_asins), batch_size),
                          desc="Encode catalog", leave=False, position=1):
            text_emb, image_emb, has_image = item_emb_store.get(all_asins[start:start + batch_size])
            # .to(device) tường minh — item_emb_store thật (ItemEmbeddingStore) luôn tự load
            # sẵn lên device nên .get() vốn đã đúng device, nhưng tham số device của hàm này
            # trước đây "chết" (không dùng), gây hiểu lầm là hàm tự đảm bảo device-safety độc
            # lập với item_emb_store. Nếu ai tái sử dụng hàm với 1 store KHÔNG tự chuyển
            # device (như FakeStore trong test), .to(device) ở đây tránh device-mismatch âm
            # thầm thay vì phụ thuộc ngầm vào store.
            text_emb = text_emb.to(device)
            image_emb = image_emb.to(device)
            has_image = has_image.to(device)
            vec = item_tower(text_emb, image_emb, has_image)
            vectors.append(F.normalize(vec, dim=-1, eps=1e-6))
    out = torch.cat(vectors, dim=0)
    if was_training:
        item_tower.train()
    return out


def run_retrieval_eval(model, loader, item_matrix, device, ks=(10, 50, 100),
                       cold_item_mask=None, item_chunk=200_000):
    """Xếp hạng toàn catalog cho từng user trong loader, trả về dict metric.

    item_matrix    : [n_items, dim] đã normalize (xem encode_all_items)
    cold_item_mask : [n_items] bool hoặc None. Khi truyền, xếp hạng CHỈ trong tập item cold
                     (pool nhỏ hơn -> số cao hơn, đo "xếp hạng giữa các item mới"). Khi None,
                     xếp hạng trên TOÀN catalog (positive cold phải cạnh tranh với cả item
                     warm — đúng với hệ thống thật). Báo cáo CẢ HAI, xem main().

    Với cold_item_mask, item_matrix bị lọc còn tập con nên topk trả index CỤC BỘ — map lại
    về index toàn cục qua local_to_global trước khi so với ground-truth."""
    model.eval()
    max_k = max(ks)

    if cold_item_mask is not None:
        local_to_global = torch.nonzero(cold_item_mask, as_tuple=True)[0]
        candidates = item_matrix[local_to_global]
        # global -> local để dịch `seen` (index toàn cục) sang index của pool con. -1 =
        # item không nằm trong pool (đã bị mask loại sẵn, không cần exclude nữa).
        global_to_local = torch.full((item_matrix.size(0),), -1, dtype=torch.long,
                                     device=local_to_global.device)
        global_to_local[local_to_global] = torch.arange(local_to_global.numel(),
                                                        device=local_to_global.device)
    else:
        local_to_global = None
        global_to_local = None
        candidates = item_matrix

    all_topk, all_gt = [], []
    with torch.no_grad():
        for user_batch, gts, seens in tqdm(loader, desc="Retrieval eval", leave=False, position=1):
            user_batch = {k: v.to(device, non_blocking=True) for k, v in user_batch.items()}
            user_vec = F.normalize(model.encode_user(**user_batch), dim=-1, eps=1e-6)

            if local_to_global is not None:
                # BUG THẬT ĐÃ SỬA (Lỗi 7 — cold pool bỏ exclude với lý do sai): comment cũ
                # ghi "item user đã đọc nằm trong train nên gần như luôn là warm, đã bị mask
                # loại sẵn". Sai kể từ bản sửa Lỗi 4: seq_before giờ là mọi thứ có ts < ts_pos
                # và KHÔNG còn chặn ở train_temporal_boundary, nên tương tác trong khoảng
                # (train_temporal_boundary, ts_pos) nằm trong `seen` — item first_seen trong cửa sổ đó
                # đúng là COLD. Pool cold nhỏ hơn catalog rất nhiều nên một item đã đọc không
                # bị loại gần như chắc chắn chiếm slot top-K, làm metric cold-pool thấp giả.
                #
                # `seen` là index TOÀN CỤC -> phải dịch sang index pool con trước khi lọc;
                # bỏ các giá trị -1 (item không thuộc pool, mask đã loại rồi).
                seens_local = []
                for s in seens:
                    if not len(s):
                        seens_local.append(EMPTY_EXCLUDE)
                        continue
                    t = global_to_local[torch.as_tensor(s, dtype=torch.long,
                                                        device=global_to_local.device)]
                    seens_local.append(t[t >= 0].cpu().numpy())
                topk_local = topk_over_catalog(user_vec, candidates, max_k,
                                               exclude=seens_local, item_chunk=item_chunk)
                # slot -1 (không đủ candidate sau khi lọc) phải giữ nguyên -1, không map
                valid = topk_local >= 0
                topk = torch.full_like(topk_local, -1)
                topk[valid] = local_to_global[topk_local[valid]]
            else:
                topk = topk_over_catalog(user_vec, candidates, max_k, exclude=seens,
                                         item_chunk=item_chunk)

            all_topk.append(topk.cpu())
            all_gt.extend(gts)

    if not all_topk:
        return {"n_users": 0}
    return ranking_metrics_at_k(torch.cat(all_topk, dim=0), all_gt, ks=list(ks))


def run_ranker_eval(model, ranker, loader, item_matrix, item_emb_store, all_asins, device,
                    n_candidates, ks=(10, 50, 100), item_chunk=200_000):
    """So sánh retrieval THÔ (top-N theo cosine) với retrieval ĐÃ RERANK bởi ranker, TRÊN
    CÙNG 1 candidate list — đo ranker có thực sự cải thiện THỨ HẠNG BÊN TRONG top-N hay
    không (KHÔNG đo Recall trên full catalog, đó là việc của run_retrieval_eval/retrieval
    stage). Trả về (metrics_before, metrics_after) — cùng ks, so trực tiếp được.

    Cách làm: lấy top-N của retrieval (CÓ exclude `seen` — xem Lỗi 7 ở comment bên dưới),
    forward N candidate đó qua ranker để có score mới, sort lại theo score mới -> topk_after.
    topk_before giữ nguyên thứ tự retrieval gốc (đã sort theo cosine similarity).

    n_candidates phải KHỚP hoặc LỚN HƠN max(ks) — nếu không, metric @k > n_candidates sẽ
    luôn thiếu candidate (topk_over_catalog tự trả -1 cho slot thiếu, ranking_metrics tính
    đúng là miss, không crash, nhưng số sẽ thấp giả tạo — xem ràng buộc ở main())."""
    model.eval()
    ranker.eval()
    max_k = max(ks)
    assert n_candidates >= max_k, (
        f"n_candidates ({n_candidates}) phải >= max(ks) ({max_k}) để so sánh công bằng — "
        "rerank trong danh sách ngắn hơn max_k sẽ luôn thiếu candidate ở @{max_k}"
    )

    all_topk_before, all_topk_after, all_gt = [], [], []
    with torch.no_grad():
        for user_batch, gts, seens in tqdm(loader, desc="Ranker eval", leave=False, position=1):
            user_batch = {k: v.to(device, non_blocking=True) for k, v in user_batch.items()}
            user_vec = F.normalize(model.encode_user(**user_batch), dim=-1, eps=1e-6)

            # topk_before: retrieval THÔ, có exclude `seen` — cùng cách run_retrieval_eval
            # làm (Lỗi 7: seen phải được loại, xem comment ở đó). topk_after rerank CHÍNH
            # candidate list này — so sánh công bằng nghĩa là 2 bên phải cùng ứng viên.
            topk_before = topk_over_catalog(user_vec, item_matrix, n_candidates,
                                            exclude=seens, item_chunk=item_chunk)

            valid_mask = topk_before >= 0  # slot -1 = không đủ candidate sạch (hiếm, catalog lớn)
            # Candidate KHÔNG hợp lệ (-1) tạm thay bằng index 0 để gather không lỗi, rồi
            # ÉP SCORE VỀ -inf để chúng luôn rớt xuống cuối sau khi ranker rerank (không
            # được để lẫn vào top vì item_matrix[0] là item THẬT, không phải "rỗng").
            safe_idx = topk_before.clamp(min=0)
            cand_asins = [[all_asins[int(i)] for i in row] for row in safe_idx.cpu().numpy()]
            candidate_text, candidate_image, candidate_has_image = gather_candidate_tensors(
                item_emb_store, cand_asins)

            rank_scores = ranker(**user_batch, candidate_text=candidate_text,
                                candidate_image=candidate_image,
                                candidate_has_image=candidate_has_image)  # [batch, n_candidates]
            neg_inf = torch.finfo(rank_scores.dtype).min
            rank_scores = rank_scores.masked_fill(~valid_mask, neg_inf)

            order = torch.argsort(rank_scores, dim=1, descending=True)
            topk_after = torch.gather(topk_before, 1, order)

            all_topk_before.append(topk_before.cpu())
            all_topk_after.append(topk_after.cpu())
            all_gt.extend(gts)

    if not all_topk_before:
        empty = {"n_users": 0}
        return empty, empty
    metrics_before = ranking_metrics_at_k(torch.cat(all_topk_before, dim=0), all_gt, ks=list(ks))
    metrics_after = ranking_metrics_at_k(torch.cat(all_topk_after, dim=0), all_gt, ks=list(ks))
    return metrics_before, metrics_after


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # random + NumPy cũng phải seed: hard-negative dùng rng.sample() (random) và nhiều chỗ
    # dùng np.random. Chỉ torch.manual_seed thì 2 nguồn kia vẫn khác nhau mỗi lần chạy ->
    # không tái lập được thí nghiệm.
    random.seed(args.seed)
    np.random.seed(args.seed)

    using_dataset = args.dataset_dir is not None

    if using_dataset:
        print(f"Loading preprocess dataset from {args.dataset_dir} ...")
        dataset = DatasetAccessor(args.dataset_dir)
        # category_leaf dựng lại dạng dict (~1.59M entry, an toàn RAM — khác hẳn by_user
        # 5.6M user đã gây MemoryError, xem build_dataset.py) — chỉ còn dùng cho
        # category_vocab/category_distribution, KHÔNG còn cho soft-negative sampling nữa
        # (đã đổi sang uniform random trong make_collate, xem ColdStartDataset). decile_of/
        # cat_decile_index (chỉ phục vụ sample_soft_negative cũ) đã bỏ hẳn.
        category_leaf = {asin: dataset.category_of(asin) for asin in dataset.asin_vocab}
        category_vocab = dataset.category_vocab
        train_temporal_boundary, val_temporal_boundary = dataset.train_temporal_boundary, dataset.val_temporal_boundary
        static_features_fn = dataset.static_features_of
        train_arr = np.load(Path(args.dataset_dir) / "train_interactions.npy")
        val_arr = np.load(Path(args.dataset_dir) / "val_interactions.npy")
        test_arr = np.load(Path(args.dataset_dir) / "test_interactions.npy")
        # val_cold dùng cho eval MỖI EPOCH (mục tiêu chính là cold-start); val_warm/test_*
        # chỉ dùng cho final evaluation sau khi train xong (xem build_dataset.py QUYẾT ĐỊNH (3)).
        val_warm_arr = np.load(Path(args.dataset_dir) / "val_warm_interactions.npy")
        val_cold_arr = np.load(Path(args.dataset_dir) / "val_cold_interactions.npy")
        test_warm_arr = np.load(Path(args.dataset_dir) / "test_warm_interactions.npy")
        test_cold_arr = np.load(Path(args.dataset_dir) / "test_cold_interactions.npy")
        print(f"Train cutoff: {train_temporal_boundary}  Val cutoff: {val_temporal_boundary}")
        print(f"train={len(train_arr):,}  val={len(val_arr):,}  test={len(test_arr):,}  "
              f"(val_warm={len(val_warm_arr):,} val_cold={len(val_cold_arr):,}  "
              f"test_warm={len(test_warm_arr):,} test_cold={len(test_cold_arr):,})")
    else:
        print("Loading reviews + meta (no --dataset-dir, quét lại toàn bộ JSONL — chậm)...")
        by_user, sorted_ts, _ = load_reviews_by_user()
        category_leaf, page_count = load_item_meta()
        category_vocab = sorted(set(category_leaf.values()))

        train_temporal_boundary, val_temporal_boundary = compute_cutoffs(sorted_ts)
        print(f"Train cutoff: {train_temporal_boundary}  Val cutoff: {val_temporal_boundary}")

        first_seen_by_item = {}
        for seq in tqdm(by_user.values(), desc="Computing item first-seen"):
            for ts, asin, _, _ in seq:
                if asin not in first_seen_by_item or ts < first_seen_by_item[asin]:
                    first_seen_by_item[asin] = ts

        static_features = compute_static_features(
            by_user, train_temporal_boundary, category_leaf, page_count, category_vocab
        )
        static_features_fn = static_features.get

    if args.item_emb_dir is None:
        raise SystemExit(
            "Thiếu --item-emb-dir: cần thư mục chứa text_embeddings.npy/image_embeddings.npy/"
            "has_image.npy/asin_to_idx.json (Giai đoạn 1, xem model/preprocess_data/data.py). "
            "Pipeline này không tự encode ảnh/text."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.num_workers > 0:
        raise SystemExit(
            "--num-workers > 0 không tương thích với item embeddings trên GPU: mỗi "
            "DataLoader worker là process riêng, không thể chia sẻ CUDA tensor với main "
            "process (CUDA context không kế thừa qua fork/pickle). Chạy với --num-workers 0 "
            "khi dùng GPU (item_emb_store giờ luôn load thẳng lên device để loại bỏ "
            "CPU->GPU transfer mỗi batch — model nhẹ nên transfer PCIe là bottleneck chính, "
            "không phải compute)."
        )
    # item_emb_store load thẳng lên device (không phải CPU rồi .to() mỗi batch) — xem
    # docstring ItemEmbeddingStore. Yêu cầu num_workers=0 (enforce ở trên).
    item_emb_store = ItemEmbeddingStore(args.item_emb_dir, device=device)

    model = TwoTowerModel(
        item_tower_kwargs=dict(text_dim=item_emb_store.text_dim, image_dim=item_emb_store.image_dim,
                                out_dim=args.item_out_dim, mlp_hidden_dim=args.mlp_hidden_dim,
                                dropout=args.dropout),
        user_tower_kwargs=dict(
            item_emb_dim=args.item_out_dim, n_static_features=N_STATIC_FEATURES,
            category_vocab_size=len(category_vocab), seq_hidden_dim=args.seq_hidden_dim,
            n_heads=args.n_heads, out_dim=args.user_out_dim, mlp_hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
        ),
    ).to(device)

    # Ranker DÙNG LẠI item_tower/user_tower của model (không tạo tower mới) — fine-tune
    # joint: gradient của listwise_loss chảy ngược vào chính 2 tower đang phục vụ retrieval,
    # xem ranker.py. optimizer GỘP CHUNG param của cả model lẫn ranker (không phải 2
    # optimizer riêng) — đơn giản nhất cho fine-tune joint, không cần đồng bộ 2 lịch LR khác
    # nhau. args.enable_ranker=False (mặc định): ranker=None, mọi thứ chạy y hệt trước khi
    # có ranking stage.
    ranker = None
    if args.enable_ranker:
        category_emb_dim = model.user_tower.category_proj.out_features
        ranker = RankerModel(
            model.item_tower, model.user_tower, item_emb_dim=args.item_out_dim,
            n_static_features=N_STATIC_FEATURES, category_emb_dim=category_emb_dim,
            attn_hidden_dim=args.rank_attn_hidden_dim, mlp_hidden_dim=args.rank_mlp_hidden_dim,
            dropout=args.dropout,
        ).to(device)

    all_params = list(model.parameters())
    if ranker is not None:
        # target_attn + mlp là param MỚI của ranker; item_tower/user_tower.parameters() đã
        # nằm trong model.parameters() ở trên rồi — cộng thêm sẽ bị DUPLICATE, Adam sẽ update
        # 2 LẦN cho cùng 1 param mỗi step (bug thật nếu không tách riêng, xem set() dưới).
        seen_ids = {id(p) for p in all_params}
        all_params += [p for p in ranker.parameters() if id(p) not in seen_ids]
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    start_epoch = 0
    best_recall = -1.0  # metric chọn checkpoint: Recall@K trên full catalog (thay best_auc)
    resume_ckpt = None  # giữ lại để load scheduler.state_dict sau khi scheduler được tạo
                         # (scheduler cần len(train_loader), tạo sau đoạn resume model/optimizer)
    if args.resume and Path(args.checkpoint_path).exists():
        # weights_only=False: mặc định PyTorch 2.6+ đổi thành True (chặn pickle chứa object
        # tuỳ ý vì lý do bảo mật) — checkpoint của chính pipeline này chứa numpy scalar
        # (best_auc dạng numpy.float64) nên bị chặn nếu để mặc định. An toàn vì checkpoint
        # tự tạo ra (không phải tải từ nguồn ngoài không tin cậy).
        resume_ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        # BẮT BUỘC kiểm tra --enable-ranker KHỚP với lần chạy đã tạo checkpoint TRƯỚC khi
        # optimizer.load_state_dict: all_params (dòng ~941) được dựng theo args.enable_ranker
        # CỦA LẦN CHẠY HIỆN TẠI, còn resume_ckpt["optimizer"] là state_dict của LẦN CHẠY
        # TRƯỚC. Nếu 2 cấu hình lệch nhau (bật ranker rồi resume quên truyền cờ, hoặc ngược
        # lại), optimizer hiện tại có SỐ PARAM KHÁC checkpoint -> PyTorch raise ValueError
        # "loaded state dict contains a parameter group that doesn't match the size of
        # optimizer's group" — crash khó truy nguồn gốc đúng lúc cần resume nhất (sau khi
        # mất tiến trình train nhiều giờ). Raise SystemExit RÕ RÀNG ở đây thay vì để lan tới
        # lỗi PyTorch mù mờ.
        ckpt_enable_ranker = resume_ckpt.get("enable_ranker", False)
        if ckpt_enable_ranker != args.enable_ranker:
            raise SystemExit(
                f"Checkpoint được train với --enable-ranker={ckpt_enable_ranker} nhưng lần "
                f"chạy này dùng --enable-ranker={args.enable_ranker} — optimizer state_dict "
                f"sẽ KHÔNG khớp số lượng param (crash mù mờ nếu bỏ qua kiểm tra này). "
                f"Chạy resume với ĐÚNG --enable-ranker như lúc train checkpoint này."
            )
        model.load_state_dict(resume_ckpt["model"])
        if ranker is not None and resume_ckpt.get("ranker") is not None:
            ranker.load_state_dict(resume_ckpt["ranker"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        # checkpoint cũ (trước khi đổi sang retrieval metric) chỉ có best_auc — không so
        # sánh được với Recall nên bỏ qua, bắt đầu lại từ -1.0.
        best_recall = resume_ckpt.get("best_recall", -1.0)
        print(f"Resumed from {args.checkpoint_path} at epoch {start_epoch} "
              f"(best_recall={best_recall:.4f})")

    precomputed_item_vectors = PrecomputedItemVectors(model.item_tower, item_emb_store, device)

    if using_dataset:
        # split= quyết định cutoff của hard_neg_pool (Lỗi 2) — train dùng train_temporal_boundary,
        # val/test dùng timestamp của chính positive.
        train_pos = build_positives_from_array(train_arr, dataset, args.max_seq_len, split="train")
        val_warm_pos = build_positives_from_array(val_warm_arr, dataset, args.max_seq_len, label="warm", split="val")
        val_cold_pos = build_positives_from_array(val_cold_arr, dataset, args.max_seq_len, label="cold", split="val")
        test_warm_pos = build_positives_from_array(test_warm_arr, dataset, args.max_seq_len, label="warm", split="test")
        test_cold_pos = build_positives_from_array(test_cold_arr, dataset, args.max_seq_len, label="cold", split="test")
    else:
        train_pos = build_positives(by_user, train_temporal_boundary, val_temporal_boundary, first_seen_by_item, "train", args.max_seq_len)
        val_pos = build_positives(by_user, train_temporal_boundary, val_temporal_boundary, first_seen_by_item, "val", args.max_seq_len)
        test_pos = build_positives(by_user, train_temporal_boundary, val_temporal_boundary, first_seen_by_item, "test", args.max_seq_len)
        val_warm_pos = [p for p in val_pos if p[4] == "warm"]
        val_cold_pos = [p for p in val_pos if p[4] == "cold"]
        test_warm_pos = [p for p in test_pos if p[4] == "warm"]
        test_cold_pos = [p for p in test_pos if p[4] == "cold"]
    print(f"train={len(train_pos)}  val_warm={len(val_warm_pos)}  val_cold={len(val_cold_pos)}  "
          f"test_warm={len(test_warm_pos)}  test_cold={len(test_cold_pos)}")

    # ── Static feature scaler: fit CHỈ trên train (Lỗi 3) ────────────────────────
    # Nếu resume, dùng lại scaler đã lưu trong checkpoint — fit lại sẽ cho mean/std khác
    # (thứ tự user khác/subsample khác) khiến model nhận feature ở scale lệch so với lúc
    # train trước đó.
    static_scaler = resume_ckpt.get("static_scaler") if resume_ckpt is not None else None
    if static_scaler is None:
        # Fit trên UNIQUE user của train (không phải trên từng positive) — user có nhiều
        # positive sẽ được đếm nhiều lần nếu fit theo dòng, làm mean/std lệch về phía user
        # hoạt động nhiều.
        seen_uid = set()
        raw_rows = []
        for uid, *_rest in tqdm(train_pos, desc="Fit static scaler"):
            if uid in seen_uid:
                continue
            seen_uid.add(uid)
            raw, _ = raw_static_vec(static_features_fn(uid), len(category_vocab))
            raw_rows.append(raw)
        static_scaler = fit_static_scaler(raw_rows)
        print(f"Static scaler fit trên {len(raw_rows):,} unique train user")
        for name, m, s in zip(
            ("user_mean_rating", "user_std_rating", "user_total_reviews", "user_avg_page_count"),
            static_scaler["mean"], static_scaler["std"],
        ):
            print(f"  {name:<15} mean={m:.4f}  std={s:.4f}")
    else:
        print("Dùng lại static_scaler từ checkpoint (resume)")

    def make_dataset(positives):
        return ColdStartDataset(
            positives, static_features_fn, len(category_vocab),
            item_emb_store, precomputed_item_vectors, args.n_hard_neg, seed=args.seed,
            static_scaler=static_scaler,
        )

    collate = make_collate(item_emb_store, args.max_seq_len, args.item_out_dim,
                            args.n_soft_neg, device=device)
    # pin_memory=False: batch giờ được build thẳng trên GPU trong collate_fn (item_emb_store
    # + precomputed_item_vectors giữ trên device, num_workers=0 bắt buộc — xem enforce ở
    # trên) — không còn CPU tensor nào cần pin để transfer async nữa, pin_memory=True lúc
    # này vô nghĩa (hoặc lỗi khi PyTorch cố pin tensor đã ở GPU).
    loader_kwargs = dict(
        batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=False, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    # ColdStartDataset giờ CHỈ còn dùng cho train — mọi eval đã chuyển sang RetrievalEvalSet
    # (1 dòng/user + ground-truth gom, xếp hạng full catalog thay vì 1 pos + 8 neg).
    train_loader = DataLoader(make_dataset(train_pos), shuffle=True, **loader_kwargs)

    # ── Retrieval eval (Recall/Hit/NDCG/MRR@K trên full catalog) ─────────────────
    # Thay hẳn AUC/PR-AUC: AUC được tính trên pool [1 pos + 8 neg] nên lệch khỏi mục tiêu
    # InfoNCE (model xếp hạng HOÀN HẢO mọi user vẫn chỉ đạt pooled AUC ~0.76 — đã đo bằng
    # mô phỏng), và Hit@K trên 9 candidate là "sampled metric" mà Krichene & Rendle (KDD'20)
    # chứng minh xếp hạng model sai lệch. Xem ranking_metrics.py.
    # RÀNG BUỘC BẮT BUỘC (đã kiểm chứng, dễ vỡ nếu sửa bừa): asin_to_item_idx và item_matrix
    # phải cùng đánh index theo THỨ TỰ SORTED của all_asins, KHÔNG phải theo
    # item_emb_store.asin_to_idx (bảng đó là row index trong text_embeddings.npy, thứ tự do
    # data.py quyết định và KHÔNG đảm bảo sorted). Hai bảng này khác nhau thật khi JSON
    # không sorted. Chúng khớp được vì encode_all_items() duyệt đúng all_asins theo thứ tự
    # này -> item_matrix[k] là vector của all_asins[k] = asin có asin_to_item_idx = k, và
    # ground_truth trong RetrievalEvalSet cũng dùng asin_to_item_idx. Nếu đổi 1 trong 2 chỗ
    # mà không đổi chỗ kia, ground-truth sẽ trỏ sang item khác -> metric sai lặng lẽ.
    all_asins = sorted(item_emb_store.asin_to_idx.keys())
    asin_to_item_idx = {a: i for i, a in enumerate(all_asins)}

    # cold_item_mask: item có first_seen > train_temporal_boundary (chưa từng xuất hiện ở train).
    #
    # BUG THẬT ĐÃ SỬA (Lỗi 8 — cold pool không có distractor): code cũ suy mask từ item_pos
    # của val_cold_pos + test_cold_pos, tức chỉ gồm item cold mà CÓ NGƯỜI rate >=4. Hệ quả:
    # mọi candidate trong pool đều là ground-truth của một user nào đó, không còn item cold
    # "nhiễu" nào để model phải phân biệt -> Recall/NDCG cold-pool cao giả tạo và không so
    # được với paper cold-start. Giờ lấy ĐÚNG định nghĩa: quét first_seen trên toàn bộ lịch
    # sử rồi so với train_temporal_boundary, đúng cách build_dataset.split_warm_cold gán nhãn cold/warm.
    #
    # CẢNH BÁO INDEX (xem khối comment ngay trên): first_seen_by_item_idx() trả mảng theo
    # thứ tự dataset.asin_vocab, còn mask phải theo thứ tự all_asins (sorted). Hai bảng này
    # KHÁC NHAU — phải map qua dataset.asin_to_idx cho từng asin, không được gán thẳng.
    cold_item_mask = torch.zeros(len(all_asins), dtype=torch.bool)
    if using_dataset:
        fs = dataset.first_seen_by_item_idx()
        dataset_idx = np.array([dataset.asin_to_idx.get(a, -1) for a in all_asins])
        has = dataset_idx >= 0
        # Item không có trong dataset = chưa từng được review = cold (first_seen "vô cực").
        cold_np = np.ones(len(all_asins), dtype=bool)
        cold_np[has] = fs[dataset_idx[has]] > train_temporal_boundary
        cold_item_mask = torch.from_numpy(cold_np)
    else:
        for i, a in enumerate(all_asins):
            f = first_seen_by_item.get(a)
            cold_item_mask[i] = (f is None or f > train_temporal_boundary)
    cold_item_mask = cold_item_mask.to(device)
    # Đối chiếu: mọi item_pos trong val_cold/test_cold PHẢI nằm trong mask (chúng đã được
    # build_dataset gán nhãn cold bằng cùng tiêu chí). Lệch = 2 nguồn first_seen bất đồng.
    _cold_pos_idx = {asin_to_item_idx[p[1]] for p in list(val_cold_pos) + list(test_cold_pos)
                     if p[1] in asin_to_item_idx}
    _missing = [i for i in _cold_pos_idx if not bool(cold_item_mask[i])]
    print(f"cold item pool = {int(cold_item_mask.sum()):,} / {len(all_asins):,} item "
          f"(positive cold nằm ngoài mask: {len(_missing)} — phải là 0)")

    # ── Ranking stage: hạ tầng loader (chỉ dựng khi --enable-ranker) ─────────────
    # candidate_idx được build LẠI mỗi epoch (đóng băng trong epoch, xem
    # build_ranker_candidates) — train_ranker_loader tạo MỚI mỗi epoch từ candidate_idx
    # mới, không phải 1 DataLoader cố định như train_loader.
    ranker_collate = None
    if ranker is not None:
        ranker_collate = make_ranker_collate(item_emb_store, args.max_seq_len, args.item_out_dim,
                                             device=device)

    # ── log_q cho sampled softmax correction (chỉ dùng khi --in-batch-neg) ───────
    # Q(i) = xác suất item i được lấy làm in-batch negative. In-batch negative là item_pos
    # của sample khác trong batch, và batch sample uniform TRÊN DÒNG INTERACTION của train,
    # nên Q(i) = (số interaction train của i) / (tổng interaction train) — tần suất thực
    # nghiệm, đúng phân phối sinh ra negative. Xem info_nce_loss để biết vì sao phải trừ.
    #
    # CẢNH BÁO INDEX (cùng loại hazard với asin_to_item_idx, xem khối comment dưới):
    # log_q phải đánh index theo item_emb_store.asin_to_idx (bảng từ JSON), KHÔNG phải
    # asin_to_item_idx (sorted). Lý do: pos_idx trong collate và soft-negative
    # randint(0, item_emb_store.n_items) đều ở không gian của store. Dùng lẫn bảng -> trừ
    # log Q của item KHÁC, thiên lệch còn tệ hơn không sửa mà KHÔNG crash.
    log_q = None
    if args.in_batch_neg:
        counts = np.zeros(item_emb_store.n_items, dtype=np.int64)
        if using_dataset:
            # Đếm vector hoá trên train_arr (đã lọc theo split sẵn). item_idx của dataset là
            # không gian asin_vocab -> phải map sang không gian store qua asin.
            dataset_to_store = np.full(len(dataset.asin_vocab), -1, dtype=np.int64)
            for a, ci in dataset.asin_to_idx.items():
                si = item_emb_store.asin_to_idx.get(a)
                if si is not None:
                    dataset_to_store[ci] = si
            mapped = dataset_to_store[train_arr["item_idx"]]
            mapped = mapped[mapped >= 0]
            np.add.at(counts, mapped, 1)
        else:
            for _uid, asin, *_r in train_pos:
                si = item_emb_store.asin_to_idx.get(asin)
                if si is not None:
                    counts[si] += 1
        total = int(counts.sum())
        if total == 0:
            raise SystemExit("log_q: không đếm được interaction train nào — kiểm tra mapping asin")
        # Item chưa từng xuất hiện ở train (count=0) có Q=0 -> log Q = -inf, không dùng được.
        # Kẹp sàn ở 1 "lần giả" (add-one smoothing): các item này là COLD, Q thật của chúng
        # gần 0 nên -log Q phải là số dương LỚN, và sàn 1/total cho đúng hướng đó mà vẫn hữu
        # hạn. Không kẹp -> -inf lan vào logit và loss thành nan.
        q = np.maximum(counts, 1).astype(np.float64) / (total + item_emb_store.n_items)
        log_q = torch.from_numpy(np.log(q)).to(device=device, dtype=torch.float32)
        nz = int((counts > 0).sum())
        print(f"log_q (sampled softmax correction): {total:,} interaction train, "
              f"{nz:,}/{item_emb_store.n_items:,} item có count>0, "
              f"log Q ∈ [{float(log_q.min()):.2f}, {float(log_q.max()):.2f}]")

    retrieval_collate = make_retrieval_collate(args.max_seq_len, args.item_out_dim, device=device)
    retrieval_loader_kwargs = dict(
        batch_size=args.eval_batch_size, collate_fn=retrieval_collate,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    def make_retrieval_loader(positives, **override):
        kw = dict(retrieval_loader_kwargs)
        kw.update(override)
        return DataLoader(
            RetrievalEvalSet(positives, static_features_fn, len(category_vocab),
                             precomputed_item_vectors, asin_to_item_idx,
                             static_scaler=static_scaler),
            shuffle=False, **kw,
        )

    # val_cold chạy MỖI EPOCH trên TOÀN BỘ user (đã chốt) -> giữ worker sống.
    val_cold_ret_loader = make_retrieval_loader(val_cold_pos)
    # 3 loader còn lại chỉ dùng 1 lần ở final eval -> không persistent_workers (mỗi worker
    # giữ riêng 1 bản item_emb_store vài GB, cộng dồn là nguyên nhân OOM đã gặp trên Kaggle).
    _once = dict(num_workers=min(args.num_workers, 2), persistent_workers=False,
                 prefetch_factor=None)
    val_warm_ret_loader = make_retrieval_loader(val_warm_pos, **_once)
    test_warm_ret_loader = make_retrieval_loader(test_warm_pos, **_once)
    test_cold_ret_loader = make_retrieval_loader(test_cold_pos, **_once)
    ks = tuple(args.eval_ks)

    # LR schedule: warmup tuyến tính (0 -> args.lr trong args.warmup_steps step đầu) rồi
    # cosine decatuâny (args.lr -> 0 hết các step còn lại) — chuẩn contrastive learning
    # (CLIP/SimCLR). Bổ trợ cho clip_grad_norm_ (không thay thế): warmup tránh việc LR full
    # ngay từ step 0 khi weight còn random dễ gây gradient lớn (nguyên nhân NaN thật đã gặp,
    # loss giảm bất thường nhanh rồi NaN quanh step 185).
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_ckpt is not None and "scheduler" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler"])

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # torch.cuda.amp.* đã deprecated

    log_every = args.log_every

    # 1 progress bar DUY NHẤT xuyên suốt toàn bộ training (giống HuggingFace Trainer) —
    # thay vì tqdm mới mỗi epoch (nhìn như nhiều thanh bar nối tiếp nhau). Log (loss/lr/gpu)
    # in bằng tqdm.write() phía TRÊN bar thay vì print() thường — print() thường sẽ chen vào
    # giữa bar đang render và làm hỏng layout terminal (bar bị lặp/nhảy dòng lộn xộn).
    # total=total_steps là TOÀN BỘ training (tất cả epoch cộng lại) — ETA hiển thị trên bar
    # do đó là ETA cho HẾT 15 epoch, KHÔNG PHẢI ETA 1 epoch (dễ hiểu lầm: 8.97 it/s x 34295
    # step/epoch ~ 64 phút/epoch thật, ETA "~16h" là 15 epoch x 64 phút, không phải 1 epoch
    # chậm đi). desc mỗi epoch có ghi rõ "Epoch X/Y" để phân biệt.
    # position=0 tường minh: pbar chính PHẢI neo cố định ở dòng đầu tiên trong suốt training.
    # Mọi tqdm() lồng bên trong vòng lặp epoch (Precompute item vectors/Encode catalog/
    # Retrieval eval/Ranker eval/Build ranker candidates) đều dùng position=1 — nếu để mặc
    # định (cả 2 cùng position=0), chúng tranh nhau CÙNG 1 dòng terminal và mỗi lần bar con
    # mở/đóng sẽ ghi đè lên bar chính, gây hiện tượng "nhảy dòng" liên tục (bug thật đã gặp:
    # progress bar renders chồng chéo, layout terminal vỡ khi bar con xuất hiện xen giữa
    # bar chính đang chạy).
    pbar = tqdm(total=total_steps, desc="Training (ETA = toàn bộ epochs)",
                initial=start_epoch * len(train_loader), position=0)
    last_recall = None  # Recall@K val_cold gần nhất — chỉ tính cuối mỗi epoch (full-catalog
                         # ranking quá đắt để chạy theo step), log theo step vẫn hiển thị giá
                         # trị này để luôn thấy metric mới nhất cạnh InfoNCE loss (đã chốt:
                         # metric theo step = InfoNCE loss + Recall@K)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_ranker_iter = None
        if ranker is not None:
            ranker.train()
            # Encode catalog bằng weight ĐẦU epoch (kết quả epoch trước) để build candidate
            # list — KHÁC ma trận cuối epoch dùng cho eval (encode lại lần nữa ở đó, xem
            # dưới) vì item_tower thay đổi liên tục suốt epoch. Chấp nhận encode 2 lần/epoch
            # (đầu để build candidate cho ranker, cuối để eval) — cùng chi phí encode_all_items
            # đã có sẵn, không thêm thao tác mới nào.
            item_matrix_for_ranker = encode_all_items(model.item_tower, item_emb_store, all_asins, device)
            candidate_idx = build_ranker_candidates(
                train_pos, model, precomputed_item_vectors, item_matrix_for_ranker,
                static_features_fn, len(category_vocab), static_scaler, asin_to_item_idx,
                args.max_seq_len, args.item_out_dim, args.rank_candidate_n, device,
            )
            del item_matrix_for_ranker
            torch.cuda.empty_cache() if device.type == "cuda" else None
            model.train()  # build_ranker_candidates gọi model.eval(), trả lại đúng mode

            ranker_dataset = RankerDataset(
                train_pos, candidate_idx, static_features_fn, len(category_vocab),
                precomputed_item_vectors, asin_to_item_idx, all_asins, static_scaler=static_scaler,
            )
            train_ranker_loader = DataLoader(
                ranker_dataset, shuffle=True, batch_size=args.batch_size,
                collate_fn=ranker_collate, num_workers=0, pin_memory=False,
            )
            # cycle: ranker_dataset dùng CHUNG train_pos (cùng độ dài) nên số step 2 loader
            # khớp nhau tự nhiên qua từng epoch — cycle chỉ để an toàn nếu lệch (vd batch
            # cuối bị drop khác nhau giữa 2 DataLoader), không phải lệch thiết kế.
            from itertools import cycle
            train_ranker_iter = cycle(train_ranker_loader)

        total_loss = 0.0
        running_loss = 0.0  # tổng loss trong window hiện tại (reset mỗi lần log, xem dưới)
        n_in_window = 0  # số step đã cộng vào running_loss kể từ lần reset gần nhất
        pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
        for step, (user_batch, pos_item_emb, neg_item_emb, _, pos_idx) in enumerate(train_loader):
            user_batch = {k: v.to(device, non_blocking=True) for k, v in user_batch.items()}
            pos_item_emb = to_device(pos_item_emb, device)
            neg_item_emb = to_device(neg_item_emb, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                user_vec, pos_vec, neg_vecs = model(user_batch, pos_item_emb, neg_item_emb)
                loss = info_nce_loss(user_vec, pos_vec, neg_vecs, temperature=args.temperature,
                                      use_in_batch_neg=args.in_batch_neg,
                                      log_q=log_q, pos_idx=pos_idx.to(device))

                # Multi-task loss: total = info_nce_loss + lambda_rank * listwise_loss —
                # CỘNG DỒN trong CÙNG 1 backward (không train xen kẽ 2 optimizer.step()
                # riêng) để tránh catastrophic forgetting mục tiêu retrieval (xem docstring
                # ranker.py + lý do chọn lambda_rank nhỏ ở myargs.py). lambda_rank nhỏ giữ
                # retrieval là mục tiêu CHÍNH, ranking là tín hiệu BỔ SUNG lên cùng tower.
                rank_loss_value = None
                if ranker is not None:
                    rank_user_batch, rank_cand_emb, gt_positions = next(train_ranker_iter)
                    rank_user_batch = {k: v.to(device, non_blocking=True) for k, v in rank_user_batch.items()}
                    rank_cand_emb = to_device(rank_cand_emb, device)
                    rank_scores = ranker(**rank_user_batch,
                                        candidate_text=rank_cand_emb[0],
                                        candidate_image=rank_cand_emb[1],
                                        candidate_has_image=rank_cand_emb[2])
                    rank_loss = listwise_loss(rank_scores, gt_positions.to(device))
                    loss = loss + args.lambda_rank * rank_loss
                    rank_loss_value = rank_loss.item()

            scaler.scale(loss).backward()
            # Gradient clipping — thiếu bước này là nguyên nhân NaN thật đã gặp (loss giảm
            # rất nhanh vài chục step đầu rồi NaN đột ngột: dấu hiệu điển hình gradient
            # explosion, không phải lỗi numeric overflow như 2 bug trước ở info_nce_loss).
            # unscale_ TRƯỚC khi clip vì scaler.scale() đã nhân loss lên hệ số lớn cho AMP —
            # clip trên gradient CHƯA unscale sẽ clip sai ngưỡng (ngưỡng thật bị nhân theo
            # scale factor, max_norm=1.0 gần như luôn bị vi phạm/vô nghĩa).
            scaler.unscale_(optimizer)
            # clip trên all_params (model + ranker param mới, KHÔNG duplicate — xem set()
            # lúc dựng optimizer) — clip riêng model.parameters() sẽ bỏ sót gradient của
            # target_attn/mlp trong ranker, để chúng nổ tự do.
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.update(1)
            loss_value = loss.item()
            total_loss += loss_value
            running_loss += loss_value
            n_in_window += 1

            if not torch.isfinite(loss).item():
                # Dừng NGAY khi thấy NaN/Inf thay vì chạy hết epoch (có thể mất hàng giờ) rồi
                # mới biết qua train_loss trung bình cuối epoch — với epoch ~6h, phát hiện
                # sớm ở đúng step tiết kiệm rất nhiều thời gian debug (bug thật đã gặp:
                # train_loss=nan chỉ lộ ra ở cuối epoch 1, sau khi đã chạy xong toàn bộ).
                raise SystemExit(
                    f"Loss không hữu hạn ({loss_value}) ở epoch {epoch+1} step {step} "
                    f"— dừng ngay để debug thay vì chạy tiếp lãng phí thời gian."
                )

            # Log mỗi log_every step: postfix của tqdm (xem tại chỗ) + pbar.write() in dòng
            # thật PHÍA TRÊN bar (giữ log lại trong output cell Kaggle để xem sau) — dùng
            # pbar.write() thay vì print() thường vì print() sẽ chen ngang bar đang render và
            # làm hỏng layout terminal (bar bị lặp dòng/nhảy lộn xộn, xem comment ở pbar).
            # n_in_window đếm tường minh số step kể từ lần reset gần nhất — không suy luận
            # từ step % log_every (bug đã gặp khi viết: tại đúng step chia hết log_every,
            # step % log_every luôn = 0 nên biểu thức đó luôn ra 1, sai hoàn toàn cho các
            # window sau lần đầu).
            if n_in_window == log_every or step == 0:
                avg_running = running_loss / n_in_window
                rec_str = f"{last_recall:.4f}" if last_recall is not None else "n/a"
                postfix = dict(loss=f"{loss_value:.4f}", avg=f"{avg_running:.4f}",
                                grad_norm=f"{grad_norm:.4f}", recall=rec_str)
                if rank_loss_value is not None:
                    postfix["rank_loss"] = f"{rank_loss_value:.4f}"
                if device.type == "cuda":
                    postfix["gpu"] = gpu_util_str(device)
                pbar.set_postfix(postfix, refresh=False)
                rank_str = f"  rank_loss={rank_loss_value:.4f}" if rank_loss_value is not None else ""
                pbar.write(f"  Epoch {epoch+1} step {step}/{len(train_loader)}  "
                           f"loss={loss_value:.4f}  avg_loss={avg_running:.4f}  "
                           f"grad_norm={grad_norm:.4f}{rank_str}  val_cold_recall@{ks[0]}={rec_str}")
                running_loss = 0.0
                n_in_window = 0

        pbar.write(f"Epoch {epoch+1}/{args.epochs}  train_loss={total_loss/len(train_loader):.4f}")
        # Refresh precomputed_item_vectors bằng item_tower MỚI NHẤT trước khi eval — bắt
        # buộc (xem docstring PrecomputedItemVectors). Thiếu bước này: eval dùng đúng cùng
        # snapshot đóng băng như lúc train (item_tower cũ), cho AUC "hợp lý" nhưng KHÔNG
        # phản ánh khả năng học content thật — bug thật đã gặp, xác nhận bằng thực nghiệm
        # (tính lại vector bằng item_tower đã train làm AUC rớt từ ~0.62 xuống ~0.52).
        precomputed_item_vectors.refresh(model.item_tower, item_emb_store)

        # Encode toàn catalog bằng item_tower MỚI NHẤT — ma trận candidate cho retrieval.
        item_matrix = encode_all_items(model.item_tower, item_emb_store, all_asins, device)

        # Mỗi epoch: val_cold trên TOÀN BỘ user, cả 2 candidate pool (đã chốt).
        cold_full = run_retrieval_eval(model, val_cold_ret_loader, item_matrix, device,
                                       ks=ks, item_chunk=args.item_chunk)
        cold_pool = run_retrieval_eval(model, val_cold_ret_loader, item_matrix, device,
                                       ks=ks, cold_item_mask=cold_item_mask,
                                       item_chunk=args.item_chunk)
        del item_matrix  # 0.81 GB — giải phóng trước khi vào epoch sau
        torch.cuda.empty_cache() if device.type == "cuda" else None

        primary_k = ks[0]
        last_recall = cold_full.get(f"recall@{primary_k}")  # log per-step dùng đến epoch kế
        pbar.write(f"  val_cold [full catalog]:\n{format_ranking_report(cold_full, prefix='   ')}")
        pbar.write(f"  val_cold [cold pool]:\n{format_ranking_report(cold_pool, prefix='   ')}")

        # Ranker eval: so retrieval THÔ vs ĐÃ RERANK trên CÙNG candidate list — đo ranker có
        # thực sự cải thiện thứ hạng BÊN TRONG top-N hay không (khác run_retrieval_eval, đo
        # Recall trên FULL catalog — đó là việc của retrieval stage, không phải ranker).
        if ranker is not None:
            item_matrix_for_eval = encode_all_items(model.item_tower, item_emb_store, all_asins, device)
            eval_n_candidates = max(args.rank_candidate_n, max(ks))
            rank_before, rank_after = run_ranker_eval(
                model, ranker, val_cold_ret_loader, item_matrix_for_eval, item_emb_store,
                all_asins, device, n_candidates=eval_n_candidates, ks=ks, item_chunk=args.item_chunk,
            )
            del item_matrix_for_eval
            torch.cuda.empty_cache() if device.type == "cuda" else None
            pbar.write(f"  val_cold [ranker: retrieval THÔ, top-{eval_n_candidates}]:\n"
                       f"{format_ranking_report(rank_before, prefix='   ')}")
            pbar.write(f"  val_cold [ranker: SAU RERANK, top-{eval_n_candidates}]:\n"
                       f"{format_ranking_report(rank_after, prefix='   ')}")

        # Chọn best checkpoint theo Recall@K trên FULL catalog — nghiêm ngặt nhất, đúng với
        # hệ thống thật (item cold phải cạnh tranh với toàn bộ item warm).
        if last_recall is not None and last_recall > best_recall:
            best_recall = last_recall
            # static_scaler PHẢI đi cùng checkpoint: inference/eval lại bằng scaler khác
            # (hoặc không có scaler) sẽ đưa feature vào model ở scale hoàn toàn khác so với
            # lúc train -> kết quả vô nghĩa. Đây cũng là lý do resume đọc lại scaler này.
            torch.save({"model": model.state_dict(),
                        "ranker": ranker.state_dict() if ranker is not None else None,
                        "enable_ranker": args.enable_ranker,
                        "epoch": epoch, "best_recall": best_recall, "static_scaler": static_scaler},
                       args.best_checkpoint_path)
            pbar.write(f"  -> new best (Recall@{primary_k}={best_recall:.4f}), "
                       f"saved to {args.best_checkpoint_path}")

        torch.save({"model": model.state_dict(),
                    "ranker": ranker.state_dict() if ranker is not None else None,
                    "enable_ranker": args.enable_ranker,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "static_scaler": static_scaler, "epoch": epoch, "best_recall": best_recall},
                   args.checkpoint_path)

    pbar.close()

    # Final eval: encode catalog 1 lần rồi dùng cho cả 4 slice.
    item_matrix = encode_all_items(model.item_tower, item_emb_store, all_asins, device)
    for split_name, warm_loader, cold_loader in (
        ("val", val_warm_ret_loader, val_cold_ret_loader),
        ("test", test_warm_ret_loader, test_cold_ret_loader),
    ):
        print(f"\nFinal evaluation ({split_name}) — full catalog ({len(all_asins):,} item):")
        for slice_name, ldr in (("warm", warm_loader), ("cold", cold_loader)):
            r = run_retrieval_eval(model, ldr, item_matrix, device, ks=ks,
                                   item_chunk=args.item_chunk)
            print(f"  {slice_name}:\n{format_ranking_report(r, prefix='   ')}")
        print(f"\nFinal evaluation ({split_name}) — cold pool "
              f"({int(cold_item_mask.sum()):,} item):")
        r = run_retrieval_eval(model, cold_loader, item_matrix, device, ks=ks,
                               cold_item_mask=cold_item_mask, item_chunk=args.item_chunk)
        print(f"  cold:\n{format_ranking_report(r, prefix='   ')}")

        if ranker is not None:
            eval_n_candidates = max(args.rank_candidate_n, max(ks))
            for slice_name, ldr in (("warm", warm_loader), ("cold", cold_loader)):
                rb, ra = run_ranker_eval(model, ranker, ldr, item_matrix, item_emb_store,
                                        all_asins, device, n_candidates=eval_n_candidates,
                                        ks=ks, item_chunk=args.item_chunk)
                print(f"\nFinal ranker eval ({split_name}, {slice_name}) — "
                      f"retrieval THÔ top-{eval_n_candidates}:\n{format_ranking_report(rb, prefix='   ')}")
                print(f"Final ranker eval ({split_name}, {slice_name}) — "
                      f"SAU RERANK top-{eval_n_candidates}:\n{format_ranking_report(ra, prefix='   ')}")


if __name__ == "__main__":
    main()
