"""Pass 3 — item static features (content_branch input, KHÔNG dùng cho N_i).

Nguồn: video_features_basic_27k.csv (category/author/music/video_type) +
video_features_statistic_27k_part{1,2,3}.csv (thống kê engagement, bao gồm
report_cnt/reduce_similar_cnt bổ sung 2026-09-09) + kuairand_video_categories.csv
(category 4 cấp, bổ sung 2026-09-10 — GIỮ SONG SONG với tag 1 cấp cũ, không thay thế:
tag cũ tiếp tục dùng riêng cho N_category/conf_content ở build_n_cumulative.py).

Đây là snapshot cuối (tổng cộng dồn tới lúc dataset thu thập) — CHỈ dùng làm feature
tĩnh cho content_branch, KHÔNG dùng để tính mat_i (đã tách riêng ở Pass 1, xem
build_n_cumulative.py để tránh leak tương lai).

[CHỐT 2026-09-10] author_id (8,839,735 unique) và music_id (14,155,985 unique, range
tới ~9.48 tỷ) là ID lớn — factorize thành index liên tục [0, n) trước khi lưu (KHÔNG
lưu ID gốc trực tiếp làm chỉ số embedding, sẽ cần bảng embedding kích thước bằng ID lớn
nhất thay vì số lượng unique thật). video_type/music_type: categorical nhỏ (3/6 giá
trị), cũng factorize (null gộp thành 1 code riêng qua pandas.factorize mặc định).

Output: item_static.npy (num_items,) structured array — mọi field cùng độ dài/cùng index
gộp chung 1 file (tránh nhân file rời cho từng loại feature):
  video_id             int64          — id gốc, dùng để tra cứu ngược
  category_id          int32          — tag 1 cấp cũ, đã label-encode (GIỮ NGUYÊN)
  author_idx           int32          — author_id đã factorize thành index liên tục
  music_idx            int32          — music_id đã factorize thành index liên tục
  video_type_id        int32          — video_type đã label-encode (3 giá trị)
  music_type_id        int32          — music_type đã label-encode (6 giá trị + missing)
  cat_l1_id..cat_l4_id  int32 x4       — category 4 cấp mới, đã label-encode riêng từng cấp
                                        (-124/UNKNOWN gốc -> 1 code riêng, không lẫn category thật)
  features             float32, (F,)  — theo thứ tự VIDEO_STATIC_STAT_FIELDS (log1p + z-score)
"""

from pathlib import Path

import numpy as np
import polars as pl

from schema import (
    VIDEO_BASIC_CATEGORICAL_FIELDS,
    VIDEO_BASIC_CATEGORY_FIELD,
    VIDEO_BASIC_ID_FIELDS,
    VIDEO_CATEGORY_ID_FIELDS,
    VIDEO_STATIC_STAT_FIELDS,
)

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data")
VIDEO_BASIC_FILE = LOG_DIR / "video_features_basic_27k.csv"
VIDEO_STAT_FILES = [
    LOG_DIR / "video_features_statistic_27k_part1.csv",
    LOG_DIR / "video_features_statistic_27k_part2.csv",
    LOG_DIR / "video_features_statistic_27k_part3.csv",
]
VIDEO_CATEGORY_L4_FILE = LOG_DIR / "kuairand_video_categories.csv"
OUT_DIR = Path(__file__).parent / "output"


def _factorize(series: pl.Series) -> np.ndarray:
    """category id/name -> index liên tục [0, n), null/-124/UNKNOWN gộp thành 1 code riêng."""
    codes, _ = series.to_pandas().factorize()
    # factorize() gán -1 cho null/NaN — CỘNG 1 để mọi code >= 0 (nn.Embedding không chấp
    # nhận index âm). Code 0 giờ dành riêng cho null/UNKNOWN, category thật bắt đầu từ 1.
    return (codes + 1).astype(np.int32)


def build_item_static() -> None:
    basic_cols = ["video_id", VIDEO_BASIC_CATEGORY_FIELD, *VIDEO_BASIC_ID_FIELDS, *VIDEO_BASIC_CATEGORICAL_FIELDS]
    basic = pl.read_csv(VIDEO_BASIC_FILE, columns=basic_cols)
    stat = pl.concat([pl.read_csv(f, columns=["video_id", *VIDEO_STATIC_STAT_FIELDS]) for f in VIDEO_STAT_FILES])
    cat_l4 = pl.read_csv(VIDEO_CATEGORY_L4_FILE, columns=["final_video_id", *VIDEO_CATEGORY_ID_FIELDS])

    df = basic.join(stat, on="video_id", how="left").fill_null(0)
    df = df.join(cat_l4, left_on="video_id", right_on="final_video_id", how="left")
    df = df.sort("video_id")

    stat_raw = df.select(VIDEO_STATIC_STAT_FIELDS).to_numpy().astype(np.float64)
    stat_log = np.log1p(np.clip(stat_raw, a_min=0, a_max=None))
    mean = stat_log.mean(axis=0, keepdims=True)
    std = stat_log.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    stat_norm = ((stat_log - mean) / std).astype(np.float32)
    num_feats = stat_norm.shape[1]

    category_codes = _factorize(df[VIDEO_BASIC_CATEGORY_FIELD])
    author_idx = _factorize(df["author_id"])
    music_idx = _factorize(df["music_id"])
    video_type_id = _factorize(df["video_type"])
    music_type_id = _factorize(df["music_type"])
    cat_l1 = _factorize(df["first_level_category_id"])
    cat_l2 = _factorize(df["second_level_category_id"])
    cat_l3 = _factorize(df["third_level_category_id"])
    cat_l4 = _factorize(df["fourth_level_category_id"])

    item_static = np.empty(
        len(df),
        dtype=np.dtype([
            ("video_id", np.int64),
            ("category_id", np.int32),
            ("author_idx", np.int32),
            ("music_idx", np.int32),
            ("video_type_id", np.int32),
            ("music_type_id", np.int32),
            ("cat_l1_id", np.int32),
            ("cat_l2_id", np.int32),
            ("cat_l3_id", np.int32),
            ("cat_l4_id", np.int32),
            ("features", np.float32, (num_feats,)),
        ]),
    )
    item_static["video_id"] = df["video_id"].to_numpy().astype(np.int64)
    item_static["category_id"] = category_codes
    item_static["author_idx"] = author_idx
    item_static["music_idx"] = music_idx
    item_static["video_type_id"] = video_type_id
    item_static["music_type_id"] = music_type_id
    item_static["cat_l1_id"] = cat_l1
    item_static["cat_l2_id"] = cat_l2
    item_static["cat_l3_id"] = cat_l3
    item_static["cat_l4_id"] = cat_l4
    item_static["features"] = stat_norm

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "item_static.npy", item_static)
    print(
        f"[build_item_static] {len(df)} items -> item_static.npy "
        f"(author={author_idx.max()+1}, music={music_idx.max()+1}, "
        f"video_type={video_type_id.max()+1}, music_type={music_type_id.max()+1})"
    )


if __name__ == "__main__":
    build_item_static()
