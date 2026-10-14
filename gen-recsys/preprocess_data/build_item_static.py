"""Pass 3 — item static features (content_branch input, KHÔNG dùng cho N_i).

[SỬA 2026-09-13] Chuyển sang KuaiRand-Pure (xem schema.py, result.md "CHECKLIST CUỐI CÙNG"):
1. [SỬA review.md L9 — leak thời gian] BỎ HẲN `stat_features` (play_cnt/like_cnt/...)
   khỏi output — đây là SNAPSHOT CUỐI KỲ (tổng cộng dồn tới lúc thu thập dataset), leak
   thông tin tương lai vào content_branch cho MỌI sample ở MỌI timestamp (đo mô phỏng:
   corr(log1p(play_cnt), log1p(N_i cuối kỳ)) ≈ 0.92 — xem result.md). Đã cân nhắc tính
   lại theo time-window nhưng CHỐT bỏ hẳn (đơn giản, an toàn tuyệt đối) thay vì build
   thêm 1 pass cumulative riêng cho từng stat field.

Nguồn: video_features_basic_pure.csv (tag/author/music/video_type/upload_dt). KHÔNG dùng
video_features_statistic_*.csv (đã bỏ stat_features vì leak tương lai).

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
  upload_day           int32          — ngày ra mắt (số ngày từ epoch), dùng tính age_t
                                        (CHỈ khi upload_dt có >= 10 giá trị — xem _upload_day)
"""

import os
from pathlib import Path

import numpy as np
import polars as pl

from schema import (
    VIDEO_BASIC_CATEGORICAL_FIELDS,
    VIDEO_BASIC_CATEGORY_FIELD,
    VIDEO_BASIC_FILE,
    VIDEO_BASIC_ID_FIELDS,
    VIDEO_BASIC_UPLOAD_FIELD,
)

OUT_DIR = Path(os.environ.get("GEN_RECSYS_OUT_DIR") or (Path(__file__).parent / "output"))


MAX_TAGS_PER_ITEM = 3  # do duoc tu Pure: 1 tag 75.1%, 2 tag 23.2%, 3 tag 0.4%
TAG_PAD = 0  # index 0 = "khong co tag" (96 item null) -> padding_idx cua nn.Embedding

# Duoi nguong nay thi age gan nhu HANG SO -> khong ghi upload_day, ActionEncoder tu bo
# nhanh age. Pure co DUNG 3 gia tri upload_dt (2022-04-09/10/11) nen roi vao day.
MIN_UPLOAD_DATES_FOR_AGE = 10


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
    n_truncated = 0
    for i, l in enumerate(lists):
        if len(l) > MAX_TAGS_PER_ITEM:
            n_truncated += 1
        for j, t in enumerate(l[:MAX_TAGS_PER_ITEM]):
            out[i, j] = vocab[t]
    if n_truncated:
        # Canh bao chu khong tu nang MAX_TAGS_PER_ITEM: nang len la doi shape cua
        # item_static, phai build lai toan bo.
        print(f"[build_item_static] CANH BAO: {n_truncated:,} item "
              f"({n_truncated/len(lists):.3%}) co >{MAX_TAGS_PER_ITEM} tag, da cat bot")
    return out, len(vocab) + 1


def _upload_day(series: pl.Series) -> np.ndarray | None:
    """`upload_dt` ("2022-04-10") -> so ngay tu epoch (int32). None neu age vo nghia.

    Luu dang NGAY-tu-epoch chu khong phai age: age phu thuoc timestamp cua tung tuong tac
    nen phai tinh runtime (dataset.py), con day la thuoc tinh TINH cua item.
    """
    n_unique = series.n_unique()
    if n_unique < MIN_UPLOAD_DATES_FOR_AGE:
        print(f"[build_item_static] upload_dt chi co {n_unique} gia tri -> age gan nhu "
              f"hang so, bo field upload_day (nhanh age tu tat)")
        return None
    days = (series.cast(pl.Utf8).str.to_date(strict=False).cast(pl.Int32))
    # item khong co ngay upload -> -1, action_encoder.age_bucket cho ve bin 0
    return days.fill_null(-1).to_numpy().astype(np.int32)


def _factorize(series: pl.Series) -> np.ndarray:
    """category id/name -> index liên tục [0, n), null gộp thành 1 code riêng."""
    codes, _ = series.to_pandas().factorize()
    # factorize() gán -1 cho null/NaN — CỘNG 1 để mọi code >= 0 (nn.Embedding không chấp
    # nhận index âm). Code 0 giờ dành riêng cho null/UNKNOWN, category thật bắt đầu từ 1.
    return (codes + 1).astype(np.int32)


def build_item_static() -> None:
    basic_cols = ["video_id", VIDEO_BASIC_CATEGORY_FIELD, VIDEO_BASIC_UPLOAD_FIELD,
                  *VIDEO_BASIC_ID_FIELDS, *VIDEO_BASIC_CATEGORICAL_FIELDS]
    df = pl.read_csv(VIDEO_BASIC_FILE, columns=basic_cols).sort("video_id")

    category_codes = _factorize(df[VIDEO_BASIC_CATEGORY_FIELD])
    tag_ids, num_tags = _parse_multi_tag(df[VIDEO_BASIC_CATEGORY_FIELD])
    author_idx = _factorize(df["author_id"])
    music_idx = _factorize(df["music_id"])
    video_type_id = _factorize(df["video_type"])
    music_type_id = _factorize(df["music_type"])
    upload_day = _upload_day(df[VIDEO_BASIC_UPLOAD_FIELD])

    video_ids = df["video_id"].to_numpy().astype(np.int64)

    fields = [
        ("video_id", np.int64),
        ("category_id", np.int32),
        ("tag_ids", np.int32, (MAX_TAGS_PER_ITEM,)),
        ("author_idx", np.int32),
        ("music_idx", np.int32),
        ("video_type_id", np.int32),
        ("music_type_id", np.int32),
    ]
    if upload_day is not None:
        fields.append(("upload_day", np.int32))

    item_static = np.empty(len(df), dtype=np.dtype(fields))
    item_static["video_id"] = video_ids
    item_static["category_id"] = category_codes
    item_static["tag_ids"] = tag_ids
    item_static["author_idx"] = author_idx
    item_static["music_idx"] = music_idx
    item_static["video_type_id"] = video_type_id
    item_static["music_type_id"] = music_type_id
    if upload_day is not None:
        item_static["upload_day"] = upload_day

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "item_static.npy", item_static)
    print(
        f"[build_item_static] {len(df)} items -> item_static.npy "
        f"(author={author_idx.max()+1}, music={music_idx.max()+1}, "
        f"video_type={video_type_id.max()+1}, music_type={music_type_id.max()+1}, "
        f"tags={num_tags} multi-hot, "
        f"age={'co' if upload_day is not None else 'khong'})"
    )


if __name__ == "__main__":
    build_item_static()
