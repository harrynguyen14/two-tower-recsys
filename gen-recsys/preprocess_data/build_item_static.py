"""Pass 3 — item static features (content_branch input, KHÔNG dùng cho N_i).

[SỬA 2026-09-13] Chuyển sang KuaiRand-Pure (xem schema.py, result.md "CHECKLIST CUỐI
CÙNG"). 2 thay đổi so với bản 27K cũ:
1. BỎ HẲN 4 field cat_l1-l4_id (category 4 cấp) — Pure KHÔNG có
   kuairand_video_categories.csv (chỉ 27K có file này). Chỉ còn category 1 cấp (`tag`,
   VIDEO_BASIC_CATEGORY_FIELD trong schema.py).
2. [SỬA review.md L9 — leak thời gian] BỎ HẲN `stat_features` (play_cnt/like_cnt/...)
   khỏi output — đây là SNAPSHOT CUỐI KỲ (tổng cộng dồn tới lúc thu thập dataset), leak
   thông tin tương lai vào content_branch cho MỌI sample ở MỌI timestamp (đo mô phỏng:
   corr(log1p(play_cnt), log1p(N_i cuối kỳ)) ≈ 0.92 — xem result.md). Đã cân nhắc tính
   lại theo time-window nhưng CHỐT bỏ hẳn (đơn giản, an toàn tuyệt đối) thay vì build
   thêm 1 pass cumulative riêng cho từng stat field.

Nguồn: video_features_basic_pure.csv (category/author/music/video_type) — KHÔNG còn dùng
video_features_statistic_pure.csv (đã bỏ stat_features) và KHÔNG còn
kuairand_video_categories.csv (không tồn tại cho Pure).

Đây KHÔNG dùng để tính item_weight (đã tách riêng ở Pass 1, xem build_n_cumulative.py để
tránh leak tương lai).

[CHỐT 2026-09-10] author_id và music_id là ID lớn — factorize thành index liên tục [0, n)
trước khi lưu (KHÔNG lưu ID gốc trực tiếp làm chỉ số embedding). video_type/music_type:
categorical nhỏ, cũng factorize (null gộp thành 1 code riêng qua pandas.factorize mặc định).

Output: item_static.npy (num_items,) structured array — mọi field cùng độ dài/cùng index
gộp chung 1 file:
  video_id             int64          — id gốc, dùng để tra cứu ngược
  category_id          int32          — tag (category 1 cấp), đã label-encode
  author_idx           int32          — author_id đã factorize thành index liên tục
  music_idx            int32          — music_id đã factorize thành index liên tục
  video_type_id        int32          — video_type đã label-encode (3 giá trị)
  music_type_id        int32          — music_type đã label-encode (6 giá trị + missing)
"""

from pathlib import Path

import numpy as np
import polars as pl

from schema import (
    VIDEO_BASIC_CATEGORICAL_FIELDS,
    VIDEO_BASIC_CATEGORY_FIELD,
    VIDEO_BASIC_ID_FIELDS,
)

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-Pure-extracted\KuaiRand-Pure\data")
VIDEO_BASIC_FILE = LOG_DIR / "video_features_basic_pure.csv"
OUT_DIR = Path(__file__).parent / "output"


MAX_TAGS_PER_ITEM = 3  # do duoc tu Pure: 1 tag 75.1%, 2 tag 23.2%, 3 tag 0.4%
TAG_PAD = 0  # index 0 = "khong co tag" (96 item null) -> padding_idx cua nn.Embedding


def _parse_multi_tag(series: pl.Series) -> tuple[np.ndarray, int]:
    """`tag` la MULTI-LABEL ("20,43") — tach thanh multi-hot thay vi factorize ca chuoi.

    factorize("20,43") cu tao ra 1 ma RIENG, khac ca "20" lan "43": 111 nhan gia thay vi
    46 tag that, va item chung tag 20 mat lien ket. Tra ve (n_item, MAX_TAGS_PER_ITEM)
    int32 da pad TAG_PAD, cong 1 de danh index 0 cho padding.
    """
    raw = series.cast(pl.Utf8).fill_null("").to_list()
    lists = [[int(x) for x in s.split(",") if x != ""] for s in raw]
    vocab = {t: i + 1 for i, t in enumerate(sorted({t for l in lists for t in l}))}
    out = np.full((len(lists), MAX_TAGS_PER_ITEM), TAG_PAD, dtype=np.int32)
    for i, l in enumerate(lists):
        for j, t in enumerate(l[:MAX_TAGS_PER_ITEM]):
            out[i, j] = vocab[t]
    return out, len(vocab) + 1


def _factorize(series: pl.Series) -> np.ndarray:
    """category id/name -> index liên tục [0, n), null gộp thành 1 code riêng."""
    codes, _ = series.to_pandas().factorize()
    # factorize() gán -1 cho null/NaN — CỘNG 1 để mọi code >= 0 (nn.Embedding không chấp
    # nhận index âm). Code 0 giờ dành riêng cho null/UNKNOWN, category thật bắt đầu từ 1.
    return (codes + 1).astype(np.int32)


def build_item_static() -> None:
    basic_cols = ["video_id", VIDEO_BASIC_CATEGORY_FIELD, *VIDEO_BASIC_ID_FIELDS, *VIDEO_BASIC_CATEGORICAL_FIELDS]
    df = pl.read_csv(VIDEO_BASIC_FILE, columns=basic_cols).sort("video_id")

    category_codes = _factorize(df[VIDEO_BASIC_CATEGORY_FIELD])
    tag_ids, num_tags = _parse_multi_tag(df[VIDEO_BASIC_CATEGORY_FIELD])
    author_idx = _factorize(df["author_id"])
    music_idx = _factorize(df["music_id"])
    video_type_id = _factorize(df["video_type"])
    music_type_id = _factorize(df["music_type"])

    item_static = np.empty(
        len(df),
        dtype=np.dtype([
            ("video_id", np.int64),
            ("category_id", np.int32),
            ("tag_ids", np.int32, (MAX_TAGS_PER_ITEM,)),
            ("author_idx", np.int32),
            ("music_idx", np.int32),
            ("video_type_id", np.int32),
            ("music_type_id", np.int32),
        ]),
    )
    item_static["video_id"] = df["video_id"].to_numpy().astype(np.int64)
    item_static["category_id"] = category_codes
    item_static["tag_ids"] = tag_ids
    item_static["author_idx"] = author_idx
    item_static["music_idx"] = music_idx
    item_static["video_type_id"] = video_type_id
    item_static["music_type_id"] = music_type_id

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "item_static.npy", item_static)
    print(
        f"[build_item_static] {len(df)} items -> item_static.npy "
        f"(author={author_idx.max()+1}, music={music_idx.max()+1}, "
        f"video_type={video_type_id.max()+1}, music_type={music_type_id.max()+1}, "
        f"tags={num_tags} multi-hot)"
    )


if __name__ == "__main__":
    build_item_static()
