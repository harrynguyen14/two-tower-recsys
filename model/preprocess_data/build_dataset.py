"""
Build dataset từ kết quả preprocess (xem Readme.md mục "Temporal split", "Soft-negative
sampling", "Static/Pseudo-Static Features") — chạy 1 LẦN, quét toàn bộ Kindle_Store.jsonl
(25.6M dòng) + meta_Kindle_Store.jsonl (1.59M item), xuất ra .npy để main.py load lại nhanh
thay vì quét lại từ đầu mỗi lần thử nghiệm hyperparameter.

Xuất (tất cả trong --out-dir):
  metadata.npy            dict (allow_pickle, NHỎ — an toàn pickle): n_users, n_items,
                           n_categories, category_vocab (list[str]), train_temporal_boundary,
                           val_temporal_boundary, train/val/test size
  user_vocab.npy           array[str], user_vocab[i] = user_id gốc — giải mã user_idx
                           (int32) trong các mảng dưới về lại string.
  asin_vocab.npy           array[str], asin_vocab[i] = asin gốc — tương tự cho item_idx.
  item_category_idx.npy    int32[n_items], item_category_idx[item_idx] = index vào
                           category_vocab (thay dict[asin]->category, xem QUYẾT ĐỊNH dưới).
  item_popularity_decile.npy int8[n_items], item_popularity_decile[item_idx] = popularity
                           decile (1..10).
  user_review_sequence.npy structured array TOÀN BỘ review (kể cả sau cutoff), SẮP XẾP
                           theo user_idx rồi theo timestamp: (user_idx, item_idx, timestamp,
                           rating, helpful_vote). Thay cho by_user dict — xem QUYẾT ĐỊNH dưới.
                           Input sequence cho User Tower.
  user_review_offsets.npy  int64[n_users+1]. Review của user_idx=i nằm ở
                           user_review_sequence[user_review_offsets[i]:user_review_offsets[i+1]]
                           (đã sort theo timestamp trong từng đoạn này).
  user_static_features.npy  structured array [n_users]: user_mean_rating, user_std_rating,
                           user_total_reviews, user_avg_page_count (index theo
                           user_idx). category_distribution KHÔNG nằm trong mảng này (kích
                           thước n_categories khác nhau/user quá lớn để flatten gọn) — lưu
                           riêng ở user_category_distribution.npy (float32[n_users, n_categories]).
  user_category_distribution.npy  float32[n_users, n_categories], hàng i = phân phối category
                           của user_idx=i (đã chuẩn hoá tổng=1, xem preprocess.py).
  train_interactions.npy / val_interactions.npy / test_interactions.npy
                           structured array: user_idx, item_idx (int32, tra qua vocab),
                           timestamp, rating (KHÔNG có cột label — train không có ý nghĩa
                           cold/warm; val/test đã tách sẵn ra warm/cold ở 4 file dưới nên
                           không cần cột label nữa, xem QUYẾT ĐỊNH (3)).
  val_warm_interactions.npy / val_cold_interactions.npy
  test_warm_interactions.npy / test_cold_interactions.npy
                           cùng dtype với val/test_interactions.npy — subset lọc sẵn theo
                           warm (item_pos đã xuất hiện ở train) / cold (chưa xuất hiện ở
                           train), xem preprocess.py mục "Temporal split". val_interactions/
                           test_interactions vẫn giữ ĐẦY ĐỦ (warm+cold gộp) cho final eval;
                           4 file *_warm/*_cold chỉ để tránh phải lọc lại theo label mỗi
                           lần dùng (đặc biệt val_cold_interactions.npy — dùng trong vòng
                           eval mỗi epoch của main.py vì mục tiêu chính là cold-start).

                           QUYẾT ĐỊNH QUAN TRỌNG (sửa 3 LẦN sau lỗi thật/yêu cầu thay đổi khi
                           chạy trên toàn bộ dữ liệu):
                           (1) train_interactions ban đầu dùng dtype Unicode cố định
                               (U32/U16) để lưu user_id/item_asin dạng string thô — với
                               17.5M dòng, mảng cần 3.6GB RAM (mỗi ký tự Unicode chiếm 4
                               byte) và crash ArrayMemoryError. Sửa: int32 index + vocab.
                           (2) by_user (dict[user_id] -> list[tuple]) lưu qua
                               np.save(..., allow_pickle=True) crash MemoryError khi
                               pickle.dump 5.6M user object riêng lẻ (pickle serialize
                               từng object Python, không nén được như mảng nhị phân đặc).
                               Sửa: bỏ hẳn dict, chuyển toàn bộ sang 1 structured array
                               phẳng (user_review_sequence.npy) + mảng offset
                               (user_review_offsets.npy) — tra cứu review của 1 user = slice
                               theo offset, O(1), không
                               cần dict/pickle nào cho dữ liệu lớn. category_leaf/decile_of
                               (dict[asin]->...) cũng đổi theo cùng nguyên tắc dù nhỏ hơn,
                               để tránh vá lại lần 3 khi n_items tăng.
                           (3) val/test bỏ cột "label" (cold/warm dạng string U4 trong
                               INTERACTION_DTYPE) — tách sẵn thành 4 file *_warm/*_cold
                               riêng thay vì lọc theo label lúc dùng. Lý do: vòng eval mỗi
                               epoch trong main.py CHỈ cần val_cold_interactions.npy (mục
                               tiêu chính là cold-start), không cần đọc/giữ cả warm trong
                               RAM lúc train; val_interactions/test_interactions đầy đủ vẫn
                               giữ lại (không xoá) để dùng cho final evaluation sau khi
                               train xong.

                           Lưu THÔ, không kèm sequence/negative — Dataset vẫn tự build lúc
                           train từ user_review_sequence.npy + item_category_idx.npy/
                           item_popularity_decile.npy (đã chốt: soft-negative cần random mỗi
                           epoch để đa dạng, không đóng băng negative cố định).

Chạy: python build_dataset.py [--out-dir <path>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # để import preprocess.py ở model/

from preprocess import (
    load_reviews_by_user,
    load_item_meta,
    compute_cutoffs,
    compute_popularity_deciles,
    compute_static_features,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build preprocess dataset (.npy) từ Kindle_Store")
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent))
    return p.parse_args()


INTERACTION_DTYPE = [
    ("user_idx", "i4"), ("item_idx", "i4"), ("timestamp", "i8"), ("rating", "f4"),
]

REVIEW_DTYPE = [
    ("user_idx", "i4"), ("item_idx", "i4"), ("timestamp", "i8"),
    ("rating", "f4"), ("helpful_vote", "i4"),
]

STATIC_FEATURE_DTYPE = [
    ("user_mean_rating", "f4"), ("user_std_rating", "f4"), ("user_total_reviews", "i4"),
    ("user_avg_page_count", "f4"),
]


def build_vocabs(by_user, category_leaf):
    user_vocab = sorted(by_user.keys())
    asin_vocab = sorted(category_leaf.keys())
    user_to_idx = {u: i for i, u in enumerate(user_vocab)}
    asin_to_idx = {a: i for i, a in enumerate(asin_vocab)}
    return user_vocab, asin_vocab, user_to_idx, asin_to_idx


def build_user_review_sequence(by_user, user_vocab, user_to_idx, asin_to_idx):
    """Structured array phẳng, SẮP XẾP theo user_idx rồi timestamp — thay cho
    dict[user_id] -> list (đã crash MemoryError khi pickle, xem docstring module).
    Trả về (user_review_sequence, user_review_offsets): user_review_offsets[i]:
    user_review_offsets[i+1] = review của user_idx=i trong user_review_sequence."""
    n_users = len(user_vocab)
    user_review_offsets = np.zeros(n_users + 1, dtype="i8")
    n_reviews = sum(len(seq) for seq in by_user.values())
    user_review_sequence = np.empty(n_reviews, dtype=REVIEW_DTYPE)  # pre-allocate: tránh
    # spike RAM của list[tuple] trung gian (Python box từng int/float) trước khi np.array()
    # nén lại, từng crash ArrayMemoryError dù mảng đích chỉ ~585MiB (xem docstring module).
    pos = 0
    for uid in tqdm(user_vocab, desc="Flattening reviews per user"):
        uidx = user_to_idx[uid]
        seq = by_user[uid]  # đã sort theo timestamp sẵn (xem load_reviews_by_user)
        for i, (ts, asin, rating, helpful_vote) in enumerate(seq):
            user_review_sequence[pos + i] = (uidx, asin_to_idx[asin], ts, rating, helpful_vote)
        pos += len(seq)
        user_review_offsets[uidx + 1] = user_review_offsets[uidx] + len(seq)

    return user_review_sequence, user_review_offsets


def _keep_interaction(ts, rating, split, train_temporal_boundary, val_temporal_boundary):
    if split == "train" and ts > train_temporal_boundary:
        return False
    if split == "val" and not (train_temporal_boundary < ts <= val_temporal_boundary):
        return False
    if split == "test" and ts <= val_temporal_boundary:
        return False
    return rating >= 4


def build_interactions_array(by_user, train_temporal_boundary, val_temporal_boundary, split, user_to_idx, asin_to_idx):
    """split: "train" | "val" | "test". Không gán cold/warm ở đây nữa — xem
    split_warm_cold() (val/test tách sẵn thành 2 file riêng, xem QUYẾT ĐỊNH (3)).

    Pre-allocate thay vì rows=[]+append+np.array(rows) — cùng lý do đã sửa ở
    build_user_review_sequence: list[tuple] trung gian (mỗi phần tử Python box riêng) tốn
    RAM gấp nhiều lần mảng đích, từng crash MemoryError ở đúng pattern này (xem docstring
    module) — quét 2 lần (đếm rồi ghi) rẻ hơn nhiều so với giữ list trung gian cho ~10-20M
    dòng."""
    n_matched = 0
    for seq in by_user.values():
        for ts, _, rating, _ in seq:
            if _keep_interaction(ts, rating, split, train_temporal_boundary, val_temporal_boundary):
                n_matched += 1

    arr = np.empty(n_matched, dtype=INTERACTION_DTYPE)
    pos = 0
    for uid, seq in by_user.items():
        uidx = user_to_idx[uid]
        for ts, asin, rating, _ in seq:
            if _keep_interaction(ts, rating, split, train_temporal_boundary, val_temporal_boundary):
                arr[pos] = (uidx, asin_to_idx[asin], ts, rating)
                pos += 1

    return arr


def split_warm_cold(interactions_arr, train_temporal_boundary, first_seen_by_item, asin_vocab):
    """cold = item_pos chưa từng xuất hiện ở train (first_seen None hoặc > train_temporal_boundary);
    warm = ngược lại. Trả về (warm_arr, cold_arr).

    Định nghĩa (trước đây nằm ở metrics.py, file đó đã xoá vì AUC/PR-AUC bị thay bằng
    retrieval metrics — xem model/ranking_metrics.py):
      cold = item_pos chưa từng xuất hiện ở train (first-seen nằm trong val/test).
      warm = item_pos đã xuất hiện ở train trước đó — dùng làm ĐỐI CHỨNG để phát hiện
             model "ăn gian" bằng memorization thay vì học content thật."""
    is_cold = np.empty(len(interactions_arr), dtype=bool)
    for i, row in enumerate(interactions_arr):
        asin = asin_vocab[int(row["item_idx"])]
        first_seen = first_seen_by_item.get(asin)
        is_cold[i] = first_seen is None or first_seen > train_temporal_boundary
    return interactions_arr[~is_cold], interactions_arr[is_cold]


def build_item_arrays(asin_vocab, category_leaf, decile_of, category_vocab):
    """item_category_idx[item_idx] = index vào category_vocab; item_popularity_decile[item_idx]
    = popularity decile (1..10, 0 nếu thiếu — item luôn có decile vì tính từ review count
    thật, xem preprocess.compute_popularity_deciles)."""
    cat_index = {c: i for i, c in enumerate(category_vocab)}
    n_items = len(asin_vocab)
    item_category_idx = np.zeros(n_items, dtype="i4")
    item_popularity_decile = np.zeros(n_items, dtype="i1")
    for i, asin in enumerate(asin_vocab):
        item_category_idx[i] = cat_index[category_leaf[asin]]
        item_popularity_decile[i] = decile_of.get(asin, 0)
    return item_category_idx, item_popularity_decile


def build_user_static_features(static_features, user_vocab, category_vocab):
    """Trả về (user_static_features, user_category_distribution) — index theo user_idx
    (thứ tự user_vocab). User không có static_features (N=0 hoàn toàn) -> hàng toàn 0,
    đúng ngữ nghĩa "Global Anchor Vector" đã chốt ở Readme."""
    n_users = len(user_vocab)
    n_categories = len(category_vocab)
    user_static_features_arr = np.zeros(n_users, dtype=STATIC_FEATURE_DTYPE)
    user_category_distribution = np.zeros((n_users, n_categories), dtype="f4")

    for i, uid in enumerate(tqdm(user_vocab, desc="Building static feature arrays")):
        feat = static_features.get(uid)
        if feat is None:
            continue
        user_static_features_arr[i] = (
            feat["user_mean_rating"], feat["user_std_rating"], feat["user_total_reviews"],
            feat["user_avg_page_count"],
        )
        user_category_distribution[i] = feat["category_distribution"]

    return user_static_features_arr, user_category_distribution


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reviews + meta (single pass, xem preprocess.py)...")
    by_user, sorted_ts, item_review_count = load_reviews_by_user()
    category_leaf, page_count = load_item_meta()
    decile_of = compute_popularity_deciles(item_review_count)
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

    print("Building user_id/asin vocab (int index thay vì string thô, xem docstring module)...")
    user_vocab, asin_vocab, user_to_idx, asin_to_idx = build_vocabs(by_user, category_leaf)

    print("Flattening by_user -> user_review_sequence.npy + user_review_offsets.npy (không dùng dict/pickle)...")
    user_review_sequence, user_review_offsets = build_user_review_sequence(by_user, user_vocab, user_to_idx, asin_to_idx)

    print("Building train/val/test interaction arrays...")
    train_arr = build_interactions_array(by_user, train_temporal_boundary, val_temporal_boundary, "train", user_to_idx, asin_to_idx)
    val_arr = build_interactions_array(by_user, train_temporal_boundary, val_temporal_boundary, "val", user_to_idx, asin_to_idx)
    test_arr = build_interactions_array(by_user, train_temporal_boundary, val_temporal_boundary, "test", user_to_idx, asin_to_idx)
    print(f"train={len(train_arr):,}  val={len(val_arr):,}  test={len(test_arr):,}")

    # by_user (dict 5.6M user -> list tuple) là cấu trúc nặng nhất trong toàn bộ pipeline —
    # giải phóng NGAY khi không còn chỗ nào dùng nữa (mọi thứ cần by_user đã build xong ở
    # trên), tránh nó chiếm RAM song song với việc cấp phát user_category_distribution
    # (1.39 GiB liên tục) ở build_user_static_features bên dưới — từng crash ArrayMemoryError
    # dù RAM tổng hệ thống vẫn còn trống, do process không tìm được khối liên tục đủ lớn.
    del by_user

    print("Building item arrays (item_category_idx, item_popularity_decile)...")
    item_category_idx, item_popularity_decile = build_item_arrays(asin_vocab, category_leaf, decile_of, category_vocab)

    print("Building static feature arrays...")
    user_static_features_arr, user_category_distribution = build_user_static_features(static_features, user_vocab, category_vocab)

    print("Splitting val/test thành warm/cold (xem split_warm_cold docstring)...")
    val_warm_arr, val_cold_arr = split_warm_cold(val_arr, train_temporal_boundary, first_seen_by_item, asin_vocab)
    test_warm_arr, test_cold_arr = split_warm_cold(test_arr, train_temporal_boundary, first_seen_by_item, asin_vocab)
    print(f"val: warm={len(val_warm_arr):,} cold={len(val_cold_arr):,}  "
          f"test: warm={len(test_warm_arr):,} cold={len(test_cold_arr):,}")

    metadata = {
        "n_users": len(user_vocab),
        "n_items": len(asin_vocab),
        "n_categories": len(category_vocab),
        "category_vocab": category_vocab,
        "train_temporal_boundary": train_temporal_boundary,
        "val_temporal_boundary": val_temporal_boundary,
        "train_size": len(train_arr),
        "val_size": len(val_arr),
        "test_size": len(test_arr),
    }

    print(f"Saving dataset to {out_dir} ...")
    np.save(out_dir / "metadata.npy", metadata, allow_pickle=True)
    np.save(out_dir / "user_vocab.npy", np.array(user_vocab, dtype=object))
    np.save(out_dir / "asin_vocab.npy", np.array(asin_vocab, dtype=object))
    np.save(out_dir / "user_review_sequence.npy", user_review_sequence)
    np.save(out_dir / "user_review_offsets.npy", user_review_offsets)
    np.save(out_dir / "item_category_idx.npy", item_category_idx)
    np.save(out_dir / "item_popularity_decile.npy", item_popularity_decile)
    np.save(out_dir / "user_static_features.npy", user_static_features_arr)
    np.save(out_dir / "user_category_distribution.npy", user_category_distribution)
    np.save(out_dir / "train_interactions.npy", train_arr)
    np.save(out_dir / "val_interactions.npy", val_arr)
    np.save(out_dir / "test_interactions.npy", test_arr)
    np.save(out_dir / "val_warm_interactions.npy", val_warm_arr)
    np.save(out_dir / "val_cold_interactions.npy", val_cold_arr)
    np.save(out_dir / "test_warm_interactions.npy", test_warm_arr)
    np.save(out_dir / "test_cold_interactions.npy", test_cold_arr)

    print("Done. metadata:", metadata)


if __name__ == "__main__":
    main()
