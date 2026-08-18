"""
Preprocess — xem Readme.md mục "Temporal split", "Static/Pseudo-Static Features cho User
Tower".

Pipeline (single streaming pass qua Kindle_Store.jsonl + meta_Kindle_Store.jsonl, theo
đúng pattern của các script analyze_*.py đã có ở ds-down/ — không load toàn bộ vào RAM
1 lúc trừ các Counter/dict cần thiết):

1. Đọc meta -> category_leaf (categories[2]) + popularity_decile (từ review count).
2. Đọc review -> group theo user_id, sort theo timestamp.
3. Temporal split: global timestamp cutoff percentile 80/90 (train/val/test).
4. Với mỗi positive (rating>=4) nằm ngoài train (val/test): gắn nhãn cold/warm theo
   first-seen của item_pos so với cutoff train.
5. Static features: user_mean_rating, user_std_rating, user_total_reviews,
   user_avg_page_count, category_distribution — CHỈ tính từ review TRƯỚC cutoff train
   của chính split đang xử lý (fit/transform tách biệt, xem Readme mục "Ghi chú kỹ thuật
   từ NVIDIA Merlin" — tránh leakage).

Soft-negative sampling (category+popularity_decile matched) đã bị bỏ khỏi module này —
main.py giờ dùng uniform random negative trong collate_fn (xem ColdStartDataset/
make_collate ở main.py để biết lý do đổi: rejection sampling per-sample là bottleneck
hiệu năng chính của pipeline train)."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

REVIEWS_FILE = Path(r"D:\amazon-datasets\Kindle_Store\Kindle_Store.jsonl")
META_FILE = Path(r"D:\amazon-datasets\Kindle_Store\meta_Kindle_Store.jsonl")

PAGE_RE = re.compile(r"([\d,]+)\s*pages", re.IGNORECASE)

TRAIN_PERCENTILE = 80
VAL_PERCENTILE = 90


# ── item meta: category_leaf + popularity decile ────────────────────────────────

def load_item_meta():
    """Trả về: category_leaf: dict[asin, str], page_count: dict[asin, int]."""
    category_leaf = {}
    page_count = {}
    with open(META_FILE, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading item meta"):
            r = json.loads(line)
            asin = r["parent_asin"]
            cats = r.get("categories") or []
            if len(cats) >= 3:
                category_leaf[asin] = cats[2]
            elif cats:
                category_leaf[asin] = cats[-1]

            details = r.get("details") or {}
            pl = details.get("Print length")
            if pl:
                m = PAGE_RE.search(pl)
                if m:
                    page_count[asin] = int(m.group(1).replace(",", ""))
    return category_leaf, page_count


def compute_popularity_deciles(item_review_count):
    """item_review_count: Counter[asin] -> count. Trả về dict[asin, decile] (1..10,
    1=ít phổ biến nhất, đúng quy ước đã chốt ở Readme mục Soft-negative sampling)."""
    items_by_pop = sorted(item_review_count.items(), key=lambda kv: kv[1])
    n = len(items_by_pop)
    decile_size = max(1, n // 10)
    decile_of = {}
    for d in range(10):
        lo = d * decile_size
        hi = (d + 1) * decile_size if d < 9 else n
        for asin, _ in items_by_pop[lo:hi]:
            decile_of[asin] = d + 1
    return decile_of


# ── reviews: load + temporal cutoff ──────────────────────────────────────────────

def load_reviews_by_user():
    by_user = defaultdict(list)
    all_ts = []
    item_review_count = Counter()
    with open(REVIEWS_FILE, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reviews"):
            r = json.loads(line)
            by_user[r["user_id"]].append(
                (r["timestamp"], r["parent_asin"], r["rating"], r.get("helpful_vote", 0))
            )
            all_ts.append(r["timestamp"])
            item_review_count[r["parent_asin"]] += 1
    for uid in by_user:
        by_user[uid].sort(key=lambda t: t[0])
    return by_user, sorted(all_ts), item_review_count


def compute_cutoffs(sorted_ts):
    """Global timestamp cutoff theo percentile 80/90 (đã chốt ở Readme mục Temporal split)."""
    n = len(sorted_ts)
    train_temporal_boundary = sorted_ts[int(n * TRAIN_PERCENTILE / 100) - 1]
    val_temporal_boundary = sorted_ts[int(n * VAL_PERCENTILE / 100) - 1]
    return train_temporal_boundary, val_temporal_boundary


# ── static features (fit CHỈ trên phần train, xem docstring module) ─────────────

def compute_static_features(by_user, train_temporal_boundary, category_leaf, page_count, category_vocab):
    """Trả về dict[user_id] -> {user_mean_rating, user_std_rating, user_total_reviews,
    user_avg_page_count, category_distribution (list[float], len=len(category_vocab))}.

    CHỈ dùng review có timestamp <= train_temporal_boundary — đây là "fit" trên train, áp dụng cho
    mọi split (train/val/test) để tránh leakage (nguyên tắc NVTabular fit/transform)."""
    cat_index = {c: i for i, c in enumerate(category_vocab)}
    features = {}

    for uid, seq in by_user.items():
        train_part = [s for s in seq if s[0] <= train_temporal_boundary]
        if not train_part:
            features[uid] = None  # user cold hoàn toàn (N=0), xử lý ở Dataset layer
            continue

        ratings = [s[2] for s in train_part]
        mean_rating = sum(ratings) / len(ratings)
        var = sum((x - mean_rating) ** 2 for x in ratings) / len(ratings)
        std_rating = var ** 0.5

        pages = [page_count[asin] for _, asin, _, _ in train_part if asin in page_count]
        avg_pages = sum(pages) / len(pages) if pages else 0.0

        cat_dist = [0.0] * len(category_vocab)
        n_with_cat = 0
        for _, asin, _, _ in train_part:
            leaf = category_leaf.get(asin)
            if leaf in cat_index:
                cat_dist[cat_index[leaf]] += 1
                n_with_cat += 1
        if n_with_cat > 0:
            cat_dist = [c / n_with_cat for c in cat_dist]

        features[uid] = {
            "user_mean_rating": mean_rating,
            "user_std_rating": std_rating,
            "user_total_reviews": len(train_part),
            "user_avg_page_count": avg_pages,
            "category_distribution": cat_dist,
        }
    return features
