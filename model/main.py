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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from myargs import parse_args
from preprocess import (
    load_reviews_by_user,
    load_item_meta,
    compute_cutoffs,
    compute_popularity_deciles,
    compute_static_features,
    build_category_decile_index,
    sample_soft_negative,
)
from item_tower import ItemEmbeddingStore
from two_tower_model import TwoTowerModel, info_nce_loss
from metrics import evaluate_cold_warm, format_report, binary_metrics
from utils import pad_and_mask


class PrecomputedItemVectors:
    """Encode TOÀN BỘ item qua ItemTower 1 lần (eval mode, không cần gradient) — dùng làm
    input sequence cho user-tower. Item_pos/hard-neg/soft-neg vẫn đi qua ItemTower "sống"
    trong forward pass thật (gradient chảy qua GMU) — chỉ sequence lịch sử là "đóng băng"
    theo tương tự cách một embedding-table thường được dùng làm input cho tower khác."""

    def __init__(self, item_tower, item_emb_store, device, batch_size=256):
        item_tower.eval()
        all_asins = sorted(item_emb_store.asin_to_idx.keys())
        vectors = []
        with torch.no_grad():
            for start in tqdm(range(0, len(all_asins), batch_size), desc="Precompute item vectors"):
                batch_asins = all_asins[start:start + batch_size]
                text_emb, image_emb, has_image = item_emb_store.get(batch_asins)
                vec = item_tower(text_emb.to(device), image_emb.to(device), has_image.to(device))
                vectors.append(vec)  # giữ trên device — không .cpu() nữa (xem docstring lớp)
        self.vectors = torch.cat(vectors, dim=0)  # [n_items, item_out_dim], trên device
        self.device = device
        self.asin_to_idx = {a: i for i, a in enumerate(all_asins)}

    @property
    def dim(self):
        return self.vectors.shape[1]

    def get(self, asins):
        idx = torch.tensor([self.asin_to_idx[a] for a in asins], dtype=torch.long, device=self.device)
        return self.vectors[idx]


class ColdStartDataset(Dataset):
    """1 sample = 1 positive (user, item_pos, rating>=4) + sequence lịch sử trước item_pos
    + N_HARD_NEG hard-negative (rating<=2 thật của chính user đó) + N_SOFT_NEG soft-negative
    (thuật toán đã chốt, preprocess.sample_soft_negative).

    user_seq_fn(uid) -> list[(ts, asin, rating, helpful_vote)], static_features_fn(uid) ->
    dict|None: callable, nhận CacheAccessor.user_seq/static_features_of khi có --cache-dir,
    hoặc by_user.__getitem__/static_features.get khi tự quét JSONL — cùng 1 Dataset dùng
    được cho cả 2 nguồn. category_leaf/decile_of vẫn là dict thật (1.59M entry — đủ nhỏ để
    giữ trong RAM, không như by_user 5.6M user từng gây MemoryError, xem build_cache.py)."""

    def __init__(self, positives, static_features_fn, category_leaf, decile_of,
                 cat_decile_index, item_emb_store, precomputed_item_vectors,
                 category_vocab_size, n_hard_neg, n_soft_neg, seed=0):
        self.positives = positives  # list[(uid, item_pos, seq_before, hard_neg_pool, label)]
        self.static_features_fn = static_features_fn
        self.category_leaf = category_leaf
        self.decile_of = decile_of
        self.cat_decile_index = cat_decile_index
        self.item_emb_store = item_emb_store
        self.precomputed_item_vectors = precomputed_item_vectors
        self.category_vocab_size = category_vocab_size
        self.n_hard_neg = n_hard_neg
        self.n_soft_neg = n_soft_neg
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        uid, item_pos, seq_before, hard_neg_pool, label = self.positives[idx]

        seq_asins = [asin for _, asin, _, _ in seq_before]
        seq_items = list(self.precomputed_item_vectors.get(seq_asins)) if seq_asins else []
        seq_ratings = [r for _, _, r, _ in seq_before]

        static = self.static_features_fn(uid)
        if static is None:
            static_vec = [0.0, 0.0, 0.0, 0.0]
            cat_dist = [0.0] * self.category_vocab_size
        else:
            static_vec = [static["mean_rating"], static["std_rating"],
                          static["total_reviews"], static["avg_page_count"]]
            cat_dist = static["category_distribution"]

        user_seen = {asin for _, asin, _, _ in seq_before} | {item_pos}
        hard_negs = self.rng.sample(hard_neg_pool, min(len(hard_neg_pool), self.n_hard_neg))
        while len(hard_negs) < self.n_hard_neg and hard_neg_pool:
            hard_negs.append(self.rng.choice(hard_neg_pool))

        soft_negs = []
        for _ in range(self.n_soft_neg):
            neg = sample_soft_negative(
                item_pos, self.category_leaf, self.decile_of, self.cat_decile_index,
                user_seen, self.rng,
            )
            soft_negs.append(neg)

        neg_asins = [a for a in (hard_negs + soft_negs) if a is not None]
        while len(neg_asins) < self.n_hard_neg + self.n_soft_neg:
            neg_asins.append(item_pos)  # fallback cực hiếm — xem Readme mục Soft-negative

        return {
            "seq_items": seq_items,
            "seq_ratings": seq_ratings,
            "static": static_vec,
            "cat_dist": cat_dist,
            "pos_asin": item_pos,
            "neg_asins": neg_asins,
            "label": label,  # "cold" | "warm" — chỉ dùng lúc eval (xem Readme Temporal split)
        }


def make_collate(item_emb_store, max_seq_len, item_out_dim, device=None):
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

        pos_text, pos_image, pos_has_image = item_emb_store.get([b["pos_asin"] for b in batch])
        pos_item_emb = (pos_text, pos_image, pos_has_image)

        # 1 lệnh get() cho toàn bộ batch*K neg_asins thay vì loop Python gọi get() riêng mỗi
        # sample — cùng bug round-trip nhỏ lẻ như seq_items đã sửa trước đó (mỗi lệnh get()
        # tạo tensor + index riêng, nhân với batch_size chạy trong collate_fn/worker).
        k = len(batch[0]["neg_asins"])
        flat_neg_asins = [a for b in batch for a in b["neg_asins"]]
        neg_text, neg_image, neg_has_image = item_emb_store.get(flat_neg_asins)
        neg_item_emb = (
            neg_text.view(len(batch), k, -1), neg_image.view(len(batch), k, -1), neg_has_image.view(len(batch), k)
        )

        labels = [b["label"] for b in batch]

        user_batch = dict(
            item_embs=item_embs, ratings=ratings, mask=mask,
            static_features=static_features, category_distribution=cat_dist,
        )
        return user_batch, pos_item_emb, neg_item_emb, labels

    return collate


def build_positives(by_user, train_cutoff, val_cutoff, first_seen_by_item, split, max_seq_len):
    """split: "train" | "val" | "test". Trả về list[(uid, item_pos, seq_before,
    hard_neg_pool, cold_warm_label)]. cold_warm_label chỉ có ý nghĩa ngoài "train".

    Dùng khi KHÔNG có cache (--cache-dir không set) — tự quét by_user từ đầu. Xem
    build_positives_from_array() cho version dùng cache (nhanh hơn, xem Readme)."""
    positives = []
    for uid, seq in by_user.items():
        for i, (ts, asin, rating, _) in enumerate(seq):
            if split == "train" and ts > train_cutoff:
                continue
            if split == "val" and not (train_cutoff < ts <= val_cutoff):
                continue
            if split == "test" and ts <= val_cutoff:
                continue
            if rating < 4:
                continue

            seq_before = [s for s in seq[:i] if s[0] <= train_cutoff][-max_seq_len:]
            hard_neg_pool = [a for (_, a, r, _) in seq if r <= 2]

            label = None
            if split != "train":
                first_seen = first_seen_by_item.get(asin)
                label = "cold" if (first_seen is None or first_seen > train_cutoff) else "warm"

            positives.append((uid, asin, seq_before, hard_neg_pool, label))
    return positives


class CacheAccessor:
    """Bọc toàn bộ cache array-based (xuất bởi model/preprocess_data/build_cache.py) sau
    1 interface giống hệt dict-based cũ (by_user[uid], category_leaf[asin], decile_of[asin],
    static_features[uid]) — để ColdStartDataset/sample_soft_negative dùng chung code cho cả
    2 nguồn (cache hoặc quét trực tiếp), không cần viết 2 phiên bản Dataset riêng.

    Không load gì vào dict Python — mọi lookup là slice/index trực tiếp trên mảng NumPy đã
    mmap-load, tránh lặp lại lỗi MemoryError khi chuyển ngược array -> dict cho 5.6M user."""

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.metadata = np.load(cache_dir / "metadata.npy", allow_pickle=True).item()
        self.user_vocab = np.load(cache_dir / "user_vocab.npy", allow_pickle=True)
        self.asin_vocab = np.load(cache_dir / "asin_vocab.npy", allow_pickle=True)
        self.user_to_idx = {u: i for i, u in enumerate(self.user_vocab)}
        self.asin_to_idx = {a: i for i, a in enumerate(self.asin_vocab)}

        self.user_review_sequence = np.load(cache_dir / "user_review_sequence.npy")
        self.user_review_offsets = np.load(cache_dir / "user_review_offsets.npy")
        self.item_category_idx = np.load(cache_dir / "item_category_idx.npy")
        self.item_popularity_decile = np.load(cache_dir / "item_popularity_decile.npy")
        self.user_static_features = np.load(cache_dir / "user_static_features.npy")
        self.user_category_distribution = np.load(cache_dir / "user_category_distribution.npy")

        self.category_vocab = self.metadata["category_vocab"]
        self.train_cutoff = self.metadata["train_cutoff"]
        self.val_cutoff = self.metadata["val_cutoff"]

    def user_seq(self, uid):
        """Trả về list[(timestamp, asin, rating, helpful_vote)] — CÙNG ĐỊNH DẠNG với
        by_user[uid] cũ, để ColdStartDataset không cần biết nguồn dữ liệu."""
        uidx = self.user_to_idx[uid]
        start, end = self.user_review_offsets[uidx], self.user_review_offsets[uidx + 1]
        rows = self.user_review_sequence[start:end]
        return [(int(r["timestamp"]), self.asin_vocab[r["item_idx"]], float(r["rating"]),
                 int(r["helpful_vote"])) for r in rows]

    def category_of(self, asin):
        return self.category_vocab[self.item_category_idx[self.asin_to_idx[asin]]]

    def decile_of_asin(self, asin):
        d = int(self.item_popularity_decile[self.asin_to_idx[asin]])
        return d if d > 0 else None

    def static_features_of(self, uid):
        """Trả về dict CÙNG SHAPE với static_features[uid] cũ, hoặc None nếu user N=0
        hoàn toàn (total_reviews=0, xem build_cache.py build_user_static_features)."""
        uidx = self.user_to_idx[uid]
        row = self.user_static_features[uidx]
        if int(row["total_reviews"]) == 0:
            return None
        return {
            "mean_rating": float(row["mean_rating"]),
            "std_rating": float(row["std_rating"]),
            "total_reviews": int(row["total_reviews"]),
            "helpful_votes_mean": float(row["helpful_votes_mean"]),
            "avg_page_count": float(row["avg_page_count"]),
            "category_distribution": self.user_category_distribution[uidx].tolist(),
        }


def build_positives_from_array(interactions_arr, cache, max_seq_len, label=None):
    """Version dùng cache: interactions_arr là structured array đã lọc sẵn theo split
    (xuất bởi build_cache.py — KHÔNG còn cột label, xem QUYẾT ĐỊNH (3) trong build_cache.py).
    cache: CacheAccessor — mọi lookup qua slice/index NumPy, không dựng lại dict cho 5.6M
    user (tránh MemoryError, xem CacheAccessor).

    label: "cold" | "warm" | None — gán CỐ ĐỊNH cho toàn bộ interactions_arr thay vì đọc
    từ mảng, vì giờ warm/cold đã được build_cache.py tách sẵn thành 2 file riêng
    (val_warm_interactions.npy / val_cold_interactions.npy) — truyền None khi gọi cho
    train_interactions.npy hoặc val/test_interactions.npy đầy đủ (không cần label).

    Cache cache.user_seq(uid) theo uid trong dict cục bộ — nhiều dòng trong interactions_arr
    thuộc cùng 1 user (mỗi user thường có nhiều review), nếu không cache thì user_seq() bị
    tính lại (slice + list-comprehension) mỗi dòng, biến vòng lặp 17M dòng train thành hàng
    giờ. seq bất biến trong suốt hàm này (cache không thay đổi giữa các dòng) nên cache an toàn."""
    positives = []
    user_seq_cache = {}
    for row in tqdm(interactions_arr, desc="Building positives"):
        uid = cache.user_vocab[int(row["user_idx"])]
        asin = cache.asin_vocab[int(row["item_idx"])]
        ts = int(row["timestamp"])
        seq = user_seq_cache.get(uid)
        if seq is None:
            seq = cache.user_seq(uid)
            user_seq_cache[uid] = seq
        seq_before = [s for s in seq if s[0] <= cache.train_cutoff and s[0] < ts][-max_seq_len:]
        hard_neg_pool = [a for (_, a, r, _) in seq if r <= 2]
        positives.append((uid, asin, seq_before, hard_neg_pool, label))
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


def run_eval_scores(model, loader, device):
    """Chạy model qua loader, trả về (scores, labels) nhị phân gộp (positive=1, negative=0)
    — không tách slice, dùng chung cho cả eval 1-slice (cold-only mỗi epoch) và eval nhiều
    slice (full report sau khi train xong, xem run_eval_cold/run_eval_full)."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for user_batch, pos_item_emb, neg_item_emb, _ in tqdm(loader, desc="Eval", leave=False):
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            pos_item_emb = to_device(pos_item_emb, device)
            neg_item_emb = to_device(neg_item_emb, device)
            user_vec, pos_vec, neg_vecs = model(user_batch, pos_item_emb, neg_item_emb)

            pos_scores = (user_vec * pos_vec).sum(dim=-1).cpu().tolist()
            neg_scores = torch.einsum("bd,bkd->bk", user_vec, neg_vecs).cpu().tolist()

            for i in range(len(pos_scores)):
                scores.append(pos_scores[i])
                labels.append(1)
                for ns in neg_scores[i]:
                    scores.append(ns)
                    labels.append(0)
    return scores, labels


def run_eval_cold(model, cold_loader, device):
    """Eval mỗi epoch trong training loop — CHỈ cold slice (mục tiêu chính là cold-start,
    xem quyết định ở build_cache.py QUYẾT ĐỊNH (3)). Trả về dict {auc, pr_auc, n}."""
    scores, labels = run_eval_scores(model, cold_loader, device)
    return binary_metrics(scores, labels)


def run_eval_full(model, warm_loader, cold_loader, device):
    """Final evaluation sau khi train xong — đủ 2 slice {cold, warm} (xem metrics.py)."""
    warm_scores, warm_labels = run_eval_scores(model, warm_loader, device)
    cold_scores, cold_labels = run_eval_scores(model, cold_loader, device)
    return evaluate_cold_warm(
        {"warm": warm_scores, "cold": cold_scores},
        {"warm": warm_labels, "cold": cold_labels},
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    using_cache = args.cache_dir is not None

    if using_cache:
        print(f"Loading preprocess cache from {args.cache_dir} ...")
        cache = CacheAccessor(args.cache_dir)
        # category_leaf/decile_of dựng lại dạng dict (~1.59M entry, an toàn RAM — khác hẳn
        # by_user 5.6M user đã gây MemoryError, xem build_cache.py) để tái dùng đúng
        # preprocess.sample_soft_negative/build_category_decile_index không sửa API.
        category_leaf = {asin: cache.category_of(asin) for asin in cache.asin_vocab}
        decile_of = {asin: cache.decile_of_asin(asin) for asin in cache.asin_vocab}
        cat_decile_index = build_category_decile_index(category_leaf, decile_of)
        category_vocab = cache.category_vocab
        train_cutoff, val_cutoff = cache.train_cutoff, cache.val_cutoff
        static_features_fn = cache.static_features_of
        train_arr = np.load(Path(args.cache_dir) / "train_interactions.npy")
        val_arr = np.load(Path(args.cache_dir) / "val_interactions.npy")
        test_arr = np.load(Path(args.cache_dir) / "test_interactions.npy")
        # val_cold dùng cho eval MỖI EPOCH (mục tiêu chính là cold-start); val_warm/test_*
        # chỉ dùng cho final evaluation sau khi train xong (xem build_cache.py QUYẾT ĐỊNH (3)).
        val_warm_arr = np.load(Path(args.cache_dir) / "val_warm_interactions.npy")
        val_cold_arr = np.load(Path(args.cache_dir) / "val_cold_interactions.npy")
        test_warm_arr = np.load(Path(args.cache_dir) / "test_warm_interactions.npy")
        test_cold_arr = np.load(Path(args.cache_dir) / "test_cold_interactions.npy")
        print(f"Train cutoff: {train_cutoff}  Val cutoff: {val_cutoff}")
        print(f"train={len(train_arr):,}  val={len(val_arr):,}  test={len(test_arr):,}  "
              f"(val_warm={len(val_warm_arr):,} val_cold={len(val_cold_arr):,}  "
              f"test_warm={len(test_warm_arr):,} test_cold={len(test_cold_arr):,})")
    else:
        print("Loading reviews + meta (no --cache-dir, quét lại toàn bộ JSONL — chậm)...")
        by_user, sorted_ts, item_review_count = load_reviews_by_user()
        category_leaf, page_count = load_item_meta()
        decile_of = compute_popularity_deciles(item_review_count)
        cat_decile_index = build_category_decile_index(category_leaf, decile_of)
        category_vocab = sorted(set(category_leaf.values()))

        train_cutoff, val_cutoff = compute_cutoffs(sorted_ts)
        print(f"Train cutoff: {train_cutoff}  Val cutoff: {val_cutoff}")

        first_seen_by_item = {}
        for seq in tqdm(by_user.values(), desc="Computing item first-seen"):
            for ts, asin, _, _ in seq:
                if asin not in first_seen_by_item or ts < first_seen_by_item[asin]:
                    first_seen_by_item[asin] = ts

        static_features = compute_static_features(
            by_user, train_cutoff, category_leaf, page_count, category_vocab
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
            item_emb_dim=args.item_out_dim, n_static_features=4,
            category_vocab_size=len(category_vocab), seq_hidden_dim=args.seq_hidden_dim,
            n_heads=args.n_heads, out_dim=args.user_out_dim, mlp_hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_auc = -1.0
    if args.resume and Path(args.checkpoint_path).exists():
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", -1.0)
        print(f"Resumed from {args.checkpoint_path} at epoch {start_epoch} (best_auc={best_auc:.4f})")

    precomputed_item_vectors = PrecomputedItemVectors(model.item_tower, item_emb_store, device)

    if using_cache:
        train_pos = build_positives_from_array(train_arr, cache, args.max_seq_len)
        val_warm_pos = build_positives_from_array(val_warm_arr, cache, args.max_seq_len, label="warm")
        val_cold_pos = build_positives_from_array(val_cold_arr, cache, args.max_seq_len, label="cold")
        test_warm_pos = build_positives_from_array(test_warm_arr, cache, args.max_seq_len, label="warm")
        test_cold_pos = build_positives_from_array(test_cold_arr, cache, args.max_seq_len, label="cold")
    else:
        train_pos = build_positives(by_user, train_cutoff, val_cutoff, first_seen_by_item, "train", args.max_seq_len)
        val_pos = build_positives(by_user, train_cutoff, val_cutoff, first_seen_by_item, "val", args.max_seq_len)
        test_pos = build_positives(by_user, train_cutoff, val_cutoff, first_seen_by_item, "test", args.max_seq_len)
        val_warm_pos = [p for p in val_pos if p[4] == "warm"]
        val_cold_pos = [p for p in val_pos if p[4] == "cold"]
        test_warm_pos = [p for p in test_pos if p[4] == "warm"]
        test_cold_pos = [p for p in test_pos if p[4] == "cold"]
    print(f"train={len(train_pos)}  val_warm={len(val_warm_pos)}  val_cold={len(val_cold_pos)}  "
          f"test_warm={len(test_warm_pos)}  test_cold={len(test_cold_pos)}")

    def make_dataset(positives):
        return ColdStartDataset(
            positives, static_features_fn, category_leaf, decile_of, cat_decile_index,
            item_emb_store, precomputed_item_vectors, len(category_vocab),
            args.n_hard_neg, args.n_soft_neg, seed=args.seed,
        )

    collate = make_collate(item_emb_store, args.max_seq_len, args.item_out_dim, device=device)
    # pin_memory=False: batch giờ được build thẳng trên GPU trong collate_fn (item_emb_store
    # + precomputed_item_vectors giữ trên device, num_workers=0 bắt buộc — xem enforce ở
    # trên) — không còn CPU tensor nào cần pin để transfer async nữa, pin_memory=True lúc
    # này vô nghĩa (hoặc lỗi khi PyTorch cố pin tensor đã ở GPU).
    loader_kwargs = dict(
        batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=False, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    # val_warm/test_warm/test_cold chỉ dùng 1 lần (final eval) — num_workers thấp hơn,
    # KHÔNG persistent_workers/prefetch sâu. Mỗi worker của mỗi DataLoader giữ riêng 1 bản
    # item_emb_store (text+image embeddings toàn bộ item, vài GB) do collate_fn/Dataset
    # đóng closure quanh nó — 5 loader x persistent_workers cộng dồn RAM là nguyên nhân
    # OOM-restart đã gặp trên Kaggle. val_cold_loader (chạy mỗi epoch) vẫn dùng
    # loader_kwargs đầy đủ vì lặp lại nhiều lần, đáng giữ worker sống.
    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs.update(num_workers=min(args.num_workers, 2),
                               persistent_workers=False, prefetch_factor=None)

    train_loader = DataLoader(make_dataset(train_pos), shuffle=True, **loader_kwargs)
    val_cold_loader = DataLoader(make_dataset(val_cold_pos), shuffle=False, **loader_kwargs)
    val_warm_loader = DataLoader(make_dataset(val_warm_pos), shuffle=False, **eval_loader_kwargs)
    test_warm_loader = DataLoader(make_dataset(test_warm_pos), shuffle=False, **eval_loader_kwargs)
    test_cold_loader = DataLoader(make_dataset(test_cold_pos), shuffle=False, **eval_loader_kwargs)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for step, (user_batch, pos_item_emb, neg_item_emb, _) in enumerate(pbar):
            user_batch = {k: v.to(device, non_blocking=True) for k, v in user_batch.items()}
            pos_item_emb = to_device(pos_item_emb, device)
            neg_item_emb = to_device(neg_item_emb, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                user_vec, pos_vec, neg_vecs = model(user_batch, pos_item_emb, neg_item_emb)
                loss = info_nce_loss(user_vec, pos_vec, neg_vecs, temperature=args.temperature,
                                      use_in_batch_neg=args.in_batch_neg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if device.type == "cuda" and step % 20 == 0:
                pbar.set_postfix(gpu=gpu_util_str(device), refresh=False)

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={total_loss/len(train_loader):.4f}")
        cold_metrics = run_eval_cold(model, val_cold_loader, device)
        auc = f"{cold_metrics['auc']:.4f}" if cold_metrics["auc"] is not None else "n/a"
        pr_auc = f"{cold_metrics['pr_auc']:.4f}" if cold_metrics["pr_auc"] is not None else "n/a"
        print(f"  val_cold  AUC={auc}  PR-AUC={pr_auc}  (n={cold_metrics['n']})")

        if cold_metrics["auc"] is not None and cold_metrics["auc"] > best_auc:
            best_auc = cold_metrics["auc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_auc": best_auc},
                       args.best_checkpoint_path)
            print(f"  -> new best (AUC={best_auc:.4f}), saved to {args.best_checkpoint_path}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_auc": best_auc}, args.checkpoint_path)

    print("\nFinal evaluation (val, đủ warm+cold):")
    print(format_report(run_eval_full(model, val_warm_loader, val_cold_loader, device)))
    print("\nFinal evaluation (test, đủ warm+cold):")
    print(format_report(run_eval_full(model, test_warm_loader, test_cold_loader, device)))


if __name__ == "__main__":
    main()
