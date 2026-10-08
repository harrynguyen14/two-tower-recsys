"""Pass 2 — build lịch sử hành vi user (KHÔNG lưu sequence đầy đủ cho mỗi sample).

[VIẾT LẠI HOÀN TOÀN 2026-09-09 — bug thiết kế nghiêm trọng đã xác nhận qua chạy thật]:
Cách làm ban đầu (lưu sequences.npy shape (N_sample, K=200)) gây "nhân bản" dữ liệu ~200
lần — với ~322,251,100 sample thật, ước tính cần ~515GB (sequences) + ~2.8TB
(action_vectors) — ĐÃ XÁC NHẬN OOM ổ đĩa thật khi chạy full ("No space left on device").

Thiết kế MỚI: chỉ lưu 1 BẢN DUY NHẤT của lịch sử (đã sort theo user_id, time_ms) +
1 mảng nhỏ (user_idx, position) cho mỗi sample hợp lệ. Build sequence ON-THE-FLY lúc
train (Dataset.__getitem__ tự cắt K=200 token từ lịch sử gốc dựa trên position) — không
nhân bản gì, dung lượng ~bằng CSV gốc (~19-22GB) thay vì hàng trăm GB/TB.

Output:
  history_meta.npy           (M,) structured [("video_id", int64), ("t", int64)]
                              — M = tổng số dòng log (~322M), đã sort. video_id + timestamp
                              gộp chung 1 file (luôn cùng độ dài, luôn đọc song song) —
                              history_video_ids.npy/history_timestamps.npy cũ đã gộp lại.
  history_action_vectors.npy (M, 11)  float32 — GIỮ RIÊNG (2D, không gộp được vào meta)

Index i trong history_meta.npy và history_action_vectors.npy LUÔN tương ứng cùng 1 dòng
log — cả 2 được ghi bởi cùng 1 vòng lặp, cùng write_pos, không cần ánh xạ gì thêm.
  user_ids_sorted.npy        (num_users,) int64 — user_id thật, sort tăng dần
  user_offsets.npy           (num_users+1,) int64 — CSR-style: user thứ i chiếm
                                history_*[user_offsets[i] : user_offsets[i+1]]
  sample_user_idx.npy   (N,) int32 — index vào user_ids_sorted/user_offsets (N = tổng sample)
  sample_position.npy   (N,) int64 — vị trí TUYỆT ĐỐI trong history_* (label tại đây,
                                input = history_*[max(user_start, position-K) : position])

KHÔNG lọc token nào khi build lịch sử (giữ cả hate/skip).
"""

import os
from pathlib import Path

import numpy as np
import polars as pl

from schema import (  # [SỬA 2026-10-07] đường dẫn gom về schema.py (env GEN_RECSYS_DATASET)
    ACTION_VECTOR_FIELDS,
    DATASET,
    LOG_STANDARD_FILES,
)
# [SỬA 2026-10-07] Bỏ import `MAX_SEQ_LEN` — KHÔNG dùng ở đâu trong file này. Preprocess ghi
# lịch sử ĐẦY ĐỦ (CSR qua user_offsets); việc cắt cửa sổ K là của `dataset.py` lúc train
# (`--max-seq-len`). Nghĩa là output .npy KHÔNG phụ thuộc K: đổi K không cần build lại.

# [SỬA 2026-10-07] Theo env `GEN_RECSYS_OUT_DIR` như build_n_cumulative.py đã làm. Trước đây
# hard-code cạnh file code, nên build 27K sẽ GHI ĐÈ output Pure (cùng tên file .npy) — mất
# dữ liệu Pure mà không có cảnh báo nào.
OUT_DIR = Path(os.environ.get("GEN_RECSYS_OUT_DIR") or (Path(__file__).parent / "output"))
# [SỬA 2026-09-13] Pure chỉ ~1.4M dòng (so với 322M của 27K) — 100 bucket không cần thiết
# nữa (mỗi bucket chỉ ~14K dòng, overhead chia bucket > lợi ích tránh OOM). Giảm xuống 10
# để giảm số lần quét lại CSV (mỗi bucket phải scan lại toàn bộ LOG_STANDARD_FILES).
#
# [SỬA 2026-10-07] Theo dataset: 27K có ~322M dòng / 23 GB CSV, 10 bucket nghĩa là ~32M
# dòng/bucket phải sort trong RAM cùng lúc => OOM trên máy 31.7 GB. 100 bucket đưa về
# ~3.2M dòng/bucket, đổi lại quét CSV 100 lần thay vì 10. Đây là đánh đổi RAM/IO, và với
# 27K thì RAM là thứ hết trước.
#
# LƯU Ý: đổi NUM_BUCKETS làm đổi play_ratio của MỌI dòng nếu rank được tính trong bucket —
# xem _compute_duration_edges_and_rank_table, nó tính TOÀN CỤC đúng vì lý do này.
NUM_BUCKETS = 100 if DATASET == "27k" else 10


def _compute_global_log1p_max() -> tuple[float, float]:
    """max(log1p(profile_stay_time)), max(log1p(comment_stay_time)) — TOÀN CỤC, 1 lần.

    [SỬA 2026-09-09 — OOM đã xác nhận qua test cụ thể]: tính .max() lồng trong
    with_columns trên TOÀN BỘ base (mọi cột log) rồi mới filter theo bucket buộc Polars
    vật chất hóa hết 322M dòng x mọi cột MỖI LẦN trong 100 bucket — verified crash
    "memory allocation of 1048576 bytes failed" ngay ở bucket 0. Tách riêng 1 pass
    streaming CHỈ select 2 cột cần thiết, tính max 1 lần duy nhất, truyền vào các bucket
    làm hằng số — giữ đúng ý nghĩa "chuẩn hóa theo max TOÀN CỤC" (không phải theo từng
    bucket, tránh lệch thang đo giữa các bucket).
    """
    scans = [pl.scan_csv(f).select(["profile_stay_time", "comment_stay_time"]) for f in LOG_STANDARD_FILES]
    lazy = pl.concat(scans)
    row = lazy.select(
        pl.col("profile_stay_time").log1p().max().alias("profile_max"),
        pl.col("comment_stay_time").log1p().max().alias("comment_max"),
    ).collect(streaming=True)
    return float(row["profile_max"][0]), float(row["comment_max"][0])


NUM_DURATION_GROUPS = 10  # số nhóm duration cho D2Q; đo được 35x giữa p10 và p90, xem dưới

# Số chữ số thập phân làm tròn `play_raw` TRƯỚC khi dựng bảng mid-rank (xem
# _compute_duration_edges_and_rank_table). Đây là thứ làm cho bảng tra scale được.
#
# Đo trên Pure thật (1,436,609 dòng). spread = max/min của trung bình theo decile duration;
# corr = tương quan decile vs trung bình. spread KHÔNG thấy CHIỀU nên phải xem cả hai:
#
#   làm tròn        spread    corr     số khoá
#   thô             4.551x   -0.982        —      <- bias gốc
#   round(3)        1.000x   -0.056     10,010    <- CHỌN: tốt nhất CẢ HAI trục
#   round(4)        1.000x   +0.632     96,293
#   round(5)        1.000x   +0.054    421,189
#   không làm tròn  1.000x   +0.506    944,664    <- dict cũ
#
# round(3) khử bias tốt NHẤT (corr -0.056, gần 0 nhất) với ít khoá nhất. Nghịch lý biểu kiến
# "làm tròn thô hơn lại chính xác hơn" có lý do thật: play_raw là thương hai số nguyên nên
# gần như LIÊN TỤC (94.6% giá trị duy nhất ở Pure, 88.3% ở 27K). Không làm tròn thì mỗi giá
# trị thành một khoá riêng, mid-rank không còn ties để lấy trung bình, và thứ hạng biến thành
# chỉ số thứ tự thuần — đó là lý do dict cũ ĐẢO DẤU (+0.506). Làm tròn gộp ties lại cho
# mid-rank làm đúng việc của nó.
#
# Scale: 10 nhóm x tối đa 1,001 giá trị = <= 10,010 khoá, BẤT KỂ dataset lớn cỡ nào — thay cho
# ~284M khoá (~28 GB) mà 27K sẽ cần nếu không làm tròn.
PLAY_RAW_DECIMALS = 3


def _compute_duration_edges_and_rank_table() -> tuple[list[float], list[dict[float, float]]]:
    """Biên decile duration + BẢNG TRA mid-rank percentile cho từng nhóm — TOÀN CỤC, 1 lần.

    PHẢI toàn cục, không được tính trong từng bucket. Bucket chia theo `user_id % NUM_BUCKETS`
    nên mỗi bucket chỉ là một MẪU: xếp hạng trong bucket cho kết quả PHỤ THUỘC NUM_BUCKETS,
    và đổi NUM_BUCKETS sẽ lặng lẽ đổi play_ratio của mọi dòng.

    [VIẾT LẠI 2026-09-17, lần 2] Bản đầu dùng pl.cut với biên percentile và một hàm ép biên
    tăng nghiêm ngặt (+1e-9 cho biên trùng). BẢN ĐÓ ĐẢO NGƯỢC chính cái bias nó phải khử, đã
    đo trên dữ liệu thật:

      play_raw bị clip về [0,1] và DỒN CỤC ở đúng 1.0 (41.5% số dòng ở decile duration ngắn
      nhất, chỉ 3.6% ở dài nhất -- lệch 8.6x). Quantile trùng nhau ở 1.0 -> bị nudge lên
      +1e-9 -> BIÊN VƯỢT QUÁ 1.0 -> không dữ liệu nào tới được các bin đó. Số bin chết tỉ lệ
      với tỉ lệ bão hòa, tức tỉ lệ với duration:

        decile        0      3      6      9
        biên > 1.0   40     18      9      2
        trần thật  0.58   0.80   0.89   0.96
        mean       0.401  0.472  0.481  0.490

      Thô: 4.55x GIẢM theo duration. Sau "sửa": 1.22x TĂNG -- đảo dấu, model học ngược lại.

    Mid-rank (`rank(method="average")`) không có vấn đề đó: mọi giá trị bằng nhau nhận CÙNG
    thứ hạng trung bình, nên khối bão hòa ở 1.0 nằm đúng giữa phần đuôi của nó thay vì bị
    dồn vào một bin. Đo được spread = 1.0000x, mỗi nhóm mean đúng 0.500.

    Trả về:
      duration_edges — NUM_DURATION_GROUPS-1 biên chia duration_ms thành decile
      rank_tables[g] — dict {play_raw đã làm tròn -> percentile mid-rank} cho nhóm duration g

    [SỬA 2026-10-05 — CHẶN trên 27K] Bảng vẫn là dict tra trực tiếp (chính xác tuyệt đối trên
    thang đã làm tròn), nhưng `play_raw` được **làm tròn PLAY_RAW_DECIMALS=3 chữ số** trước khi
    dựng. Không làm tròn thì nó không scale: `play_raw` là thương hai số nguyên nên gần như
    liên tục — 27K cho 88.3% giá trị duy nhất (đo: 1,047,291 trên 1,186,059 dòng), ngoại suy
    322M dòng là ~284M khoá, dict Python ~**28 GB**.

    Làm tròn còn khử bias TỐT HƠN, không chỉ rẻ hơn — xem bảng đo ở PLAY_RAW_DECIMALS.
    """
    scans = [pl.scan_csv(f).select(["duration_ms", "play_time_ms"]) for f in LOG_STANDARD_FILES]
    qs = [i / NUM_DURATION_GROUPS for i in range(1, NUM_DURATION_GROUPS)]
    df = pl.concat(scans).select(
        pl.col("duration_ms"),
        (pl.col("play_time_ms") / pl.col("duration_ms").clip(lower_bound=1))
        .clip(0.0, 1.0).alias("play_raw"),
    ).collect(streaming=True)

    # Biên duration cũng phải tăng nghiêm ngặt cho pl.cut, nhưng ở đây BỎ biên trùng đi
    # (nhóm rỗng) chứ không nudge: duration không bị chặn trên nên nudge không gây trần giả,
    # song bỏ hẳn vẫn đơn giản và an toàn hơn.
    duration_edges = sorted({float(df["duration_ms"].quantile(q)) for q in qs})

    groups = df["duration_ms"].cut(
        duration_edges, labels=[str(i) for i in range(len(duration_edges) + 1)]
    )
    df = df.with_columns(groups.alias("_g"))

    rank_tables = []
    for g in range(len(duration_edges) + 1):
        sub = df.filter(pl.col("_g") == str(g))["play_raw"].to_numpy()
        if len(sub) == 0:
            rank_tables.append({})      # nhom rong -> _to_percentile de nguyen NaN
            continue
        rank_tables.append(_mid_rank_table(np.round(sub, PLAY_RAW_DECIMALS)))
    return duration_edges, rank_tables


def _mid_rank_table(values: np.ndarray) -> dict[float, float]:
    """{gia tri -> percentile mid-rank} cho MOT nhom duration.

    MID-rank chu khong phai rank thuong: cac gia tri BANG NHAU nhan CUNG thu hang trung binh.
    Day la yeu cau cua D2Q, va la ly do phai lam tron truoc — khong lam tron thi moi gia tri
    thanh mot khoa rieng, khong con ties de lay trung binh (xem PLAY_RAW_DECIMALS).
    """
    uniq, counts = np.unique(values, return_counts=True)
    csum = np.concatenate([[0], np.cumsum(counts)])
    mid = (csum[:-1] + csum[1:] - 1) / 2.0
    pct = mid / max(len(values) - 1, 1)
    return dict(zip(uniq.tolist(), pct.tolist()))


def _compute_derived_action_fields(
    df: pl.DataFrame, profile_max: float, comment_max: float,
    duration_edges: list[float], rank_tables: list[dict[float, float]],
) -> pl.DataFrame:
    """play_ratio, profile_stay_time_norm, comment_stay_time_norm — derived.

    profile/comment: chuẩn hóa theo max TOÀN CỤC (xem _compute_global_log1p_max).

    play_ratio [SỬA 2026-09-17 — D2Q]: KHÔNG còn là play_time/duration thô. Thời lượng video
    là BIẾN GÂY NHIỄU — nó ảnh hưởng ĐỒNG THỜI lên việc video được hiển thị và lên watch
    time dự đoán (D2Q, arXiv:2206.06003, KDD 2022, Kuaishou, đã chạy production; báo cáo
    +0.57% tổng watch time so với WLR, +0.75% so với VR trong A/B online).

    ĐO TRÊN CHÍNH KuaiRand-Pure (1,436,609 dòng), play_ratio thô theo decile duration:

        decile   median duration   mean play_ratio   % bão hòa ở 1.0
          0           10.2 s           0.529             0.311
          3           41.6 s           0.375             0.177
          6          106.4 s           0.270             0.104
          9          294.6 s           0.131             0.036

    Giảm ĐƠN ĐIỆU 4× từ decile 0 xuống decile 9. Model học play_ratio thô sẽ học "video
    ngắn = user thích", trong khi thực chất chỉ là "video ngắn dễ xem hết". duration trải
    35× giữa p10 (11.9 s) và p90 (237.9 s), nên đây không phải hiệu ứng biên.

    Cách sửa theo D2Q: chia video theo nhóm duration, rồi đổi watch time thành MID-RANK
    PERCENTILE TRONG NHÓM đó (các giá trị bằng nhau nhận cùng thứ hạng trung bình). "Xem lâu hơn 80% số lần xem các video cùng độ dài" là đại lượng
    so sánh được giữa video 10 giây và video 5 phút; "xem hết 80%" thì không.

    ĐÁNH ĐỔI ĐÃ BIẾT: mất thông tin TUYỆT ĐỐI (0.9 và 0.3 giờ chỉ còn phân biệt qua thứ hạng
    trong nhóm). Chấp nhận vì bản thô cũng đã mất một phần rồi — 16.7% số dòng bão hòa đúng
    ở 1.0 do clip, và tỉ lệ bão hòa đó lệch 8.6× theo duration (31.1% ở decile 0 vs 3.6% ở
    decile 9), tức phần bị mất CHÍNH LÀ phần nhiễu duration.

    `long_view` KHÔNG sửa: đo được nó gần như phẳng theo duration (0.31-0.37, không xu
    hướng) vì Kuaishou đã định nghĩa ngưỡng của nó theo duration sẵn rồi.

    duration_ms <= 0 (28,874 dòng, ~2%) rơi vào nhóm 0 và được xếp hạng trong đó — không
    tách riêng, vì play_time trên video duration=0 vốn không diễn giải được, và tách ra
    thành nhóm riêng sẽ cho chúng thứ hạng "cao" giả tạo.
    """
    play_raw = (pl.col("play_time_ms") / pl.col("duration_ms").clip(lower_bound=1)).clip(0.0, 1.0)
    # Tra BẢNG đã tính toàn cục, không rank tại chỗ: hàm này chạy trên TỪNG BUCKET user nên
    # rank cục bộ sẽ phụ thuộc NUM_BUCKETS (xem _compute_duration_edges_and_rank_table).
    group_expr = pl.col("duration_ms").cut(
        duration_edges, labels=[str(i) for i in range(len(duration_edges) + 1)]
    )
    tagged = df.with_columns(play_raw.alias("_play_raw"), group_expr.alias("_g"))

    # [SỬA 2026-10-05] Tra dict trên play_raw ĐÃ LÀM TRÒN, thay cho replace_strict trên
    # play_raw thô (không scale: 27K cần ~284M khoá ~28 GB, xem PLAY_RAW_DECIMALS).
    #
    # map_batches giữ được LAZY (hàm này nhận cả LazyFrame của pipeline lẫn DataFrame của
    # test) và chạy một lần trên cả cột thay vì NUM_DURATION_GROUPS lần when/then.
    def _to_percentile(cols: pl.Series) -> pl.Series:
        st = cols.struct.unnest()
        raw = np.round(st["_play_raw"].to_numpy(), PLAY_RAW_DECIMALS)
        grp = st["_g"].cast(pl.Int32).to_numpy()
        # NaN mac dinh: gia tri KHONG co trong bang phai LO RA, khong duoc thanh 0.0 — mot
        # play_ratio=0 gia se tat han nhanh click cua token do ma khong bao gi. Cho ghi
        # history_action_mm bat NaN thanh ValueError.
        out = np.full(len(raw), np.nan, dtype=np.float64)
        for g, table in enumerate(rank_tables):
            m = grp == g
            if not m.any() or not table:
                continue
            out[m] = [table.get(v, np.nan) for v in raw[m]]
        return pl.Series("play_ratio", out, dtype=pl.Float64)

    play_expr = pl.struct(["_play_raw", "_g"]).map_batches(
        _to_percentile, return_dtype=pl.Float64
    )

    return tagged.with_columns(play_expr.alias("play_ratio")).with_columns(
        (pl.col("profile_stay_time").log1p() / profile_max)
        .fill_nan(0.0)
        .alias("profile_stay_time_norm"),
        (pl.col("comment_stay_time").log1p() / comment_max)
        .fill_nan(0.0)
        .alias("comment_stay_time_norm"),
    ).drop(["_play_raw", "_g"])


def _iter_log_buckets():
    """Chia user_id thành NUM_BUCKETS nhóm, xử lý tuần tự — mỗi bucket chỉ lọc + sort
    phần dữ liệu thuộc về nó (nhỏ hơn NUM_BUCKETS lần), không giữ quá 1 bucket trong RAM.

    [SỬA 2026-09-09]: filter theo bucket TRƯỚC, tính derived fields SAU (đã verified qua
    test cụ thể: filter-trước-derived-sau chạy OK; derived-trước-filter OOM ngay bucket 0
    vì .max() buộc vật chất hóa toàn bộ 322M dòng mỗi bucket).
    """
    profile_max, comment_max = _compute_global_log1p_max()
    duration_edges, rank_tables = _compute_duration_edges_and_rank_table()

    scans_base = [pl.scan_csv(f) for f in LOG_STANDARD_FILES]
    base = pl.concat(scans_base)

    for bucket_idx in range(NUM_BUCKETS):
        bucket_filtered = base.filter(pl.col("user_id") % NUM_BUCKETS == bucket_idx)
        bucket = (
            _compute_derived_action_fields(bucket_filtered, profile_max, comment_max,
                                           duration_edges, rank_tables)
            .sort(["user_id", "time_ms"])
            .collect(streaming=True)
        )
        yield bucket


def _count_total_users_and_rows() -> tuple[int, int]:
    """Đếm trước tổng số user + tổng số dòng — rẻ, chỉ đọc cột user_id."""
    scans = [pl.scan_csv(f).select("user_id") for f in LOG_STANDARD_FILES]
    lazy = pl.concat(scans)
    counts = lazy.group_by("user_id").agg(pl.len().alias("count")).collect(streaming=True)
    return len(counts), int(counts["count"].sum())


def build_sequences() -> None:
    """Ghi lịch sử (history_*) + index sample (sample_*), KHÔNG lưu sequence đầy đủ.

    2 pass qua dữ liệu (chấp nhận đọc lại NUM_BUCKETS lần, đã đổi từ thiết kế 1-pass ban
    đầu do sort toàn cục OOM thật — xem note ở _iter_log_buckets):
      Pass A: build history_* (mọi dòng log, sort theo user) + user_offsets (CSR).
      Pass B: build sample_* (chỉ (user_idx, position) cho vị trí có count>=2, tức có
              ít nhất 1 token lịch sử trước nó — vị trí ĐẦU TIÊN của mỗi user KHÔNG hợp
              lệ làm sample, giữ đúng bug đã sửa trước đó "N=count-1").
    """
    num_actions = len(ACTION_VECTOR_FIELDS)

    print("[build_sequences] đếm trước tổng số user/dòng (streaming)...")
    num_users, total_rows = _count_total_users_and_rows()
    print(f"[build_sequences] tổng: {num_users} user, {total_rows} dòng log")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_dtype = np.dtype([("video_id", np.int64), ("t", np.int64)])
    history_meta_mm = np.lib.format.open_memmap(OUT_DIR / "history_meta.npy", mode="w+", dtype=meta_dtype, shape=(total_rows,))
    history_action_mm = np.lib.format.open_memmap(OUT_DIR / "history_action_vectors.npy", mode="w+", dtype=np.float32, shape=(total_rows, num_actions))

    # [SỬA 2026-09-09 — BUG NGHIÊM TRỌNG đã xác nhận qua test cụ thể]: KHÔNG được dùng
    # np.unique(all_user_ids, return_index=True) để build offsets sau khi ghi xong —
    # np.unique SẮP XẾP LẠI theo GIÁ TRỊ user_id tăng dần, nhưng vị trí THẬT trong
    # history_* lại theo THỨ TỰ BUCKET (user_id % NUM_BUCKETS), hoàn toàn khác thứ tự
    # giá trị user_id — gây offsets sai hoàn toàn (không đơn điệu tăng), đã tái hiện cụ
    # thể: user_offsets nhảy lùi giữa các bucket. Đã CHỐT: track offset TRỰC TIẾP ngay
    # trong vòng lặp ghi, theo đúng thứ tự vật lý, KHÔNG sort lại sau.
    user_ids_ordered: list[int] = []
    user_offset_start_ordered: list[int] = []
    user_count_ordered: list[int] = []

    write_pos = 0
    for bucket_idx, bucket_df in enumerate(_iter_log_buckets()):
        n = len(bucket_df)
        if n == 0:
            print(f"[build_sequences] bucket {bucket_idx}/{NUM_BUCKETS}: empty, skip")
            continue

        history_meta_mm["video_id"][write_pos:write_pos + n] = bucket_df["video_id"].to_numpy().astype(np.int64)
        history_meta_mm["t"][write_pos:write_pos + n] = bucket_df["time_ms"].to_numpy().astype(np.int64)
        av = bucket_df.select(ACTION_VECTOR_FIELDS).to_numpy().astype(np.float32)
        # [THÊM 2026-09-17] Chặn NaN TẠI CHỖ GHI. play_ratio tra bảng mid-rank với
        # default=None, nên một giá trị play_raw không có trong bảng thành null -> NaN ở đây.
        # Không bắt thì nó chảy thẳng vào token và tắt âm thầm nhánh click của lượt đó.
        if not np.isfinite(av).all():
            bad = np.argwhere(~np.isfinite(av))
            cols = sorted({ACTION_VECTOR_FIELDS[c] for _, c in bad})
            raise ValueError(
                f"{len(bad)} giá trị NaN/Inf trong action_vector ở bucket này, cột: {cols}. "
                f"Với play_ratio nghĩa là bảng mid-rank không phủ hết giá trị play_raw."
            )
        history_action_mm[write_pos:write_pos + n] = av

        # bucket_df đã sort theo (user_id, time_ms) -> np.unique ở ĐÂY an toàn (chỉ dùng
        # để tìm ranh giới user TRONG bucket này, không ảnh hưởng thứ tự ghi toàn cục)
        uids, starts_local, counts_local = np.unique(
            bucket_df["user_id"].to_numpy().astype(np.int64), return_index=True, return_counts=True
        )
        user_ids_ordered.extend(uids.tolist())
        user_offset_start_ordered.extend((starts_local + write_pos).tolist())
        user_count_ordered.extend(counts_local.tolist())

        write_pos += n
        print(f"[build_sequences] bucket {bucket_idx}/{NUM_BUCKETS}: {n} rows written, {write_pos}/{total_rows}")

    history_meta_mm.flush()
    history_action_mm.flush()

    assert write_pos == total_rows, f"write_pos={write_pos} != total_rows={total_rows}"

    print("[build_sequences] build user_offsets (CSR, theo đúng thứ tự ghi thật)...")
    user_ids_sorted = np.array(user_ids_ordered, dtype=np.int64)
    user_offsets_start = np.array(user_offset_start_ordered, dtype=np.int64)
    user_counts_final = np.array(user_count_ordered, dtype=np.int64)
    user_offsets = np.append(user_offsets_start, total_rows).astype(np.int64)

    np.save(OUT_DIR / "user_ids_sorted.npy", user_ids_sorted)
    np.save(OUT_DIR / "user_offsets.npy", user_offsets)
    print(f"[build_sequences] {len(user_ids_sorted)} users -> user_ids_sorted.npy, user_offsets.npy")

    print("[build_sequences] build sample index (user_idx, position)...")
    valid_user_mask = user_counts_final >= 2
    sample_user_idx_list = []
    sample_position_list = []
    for u_idx in np.where(valid_user_mask)[0]:
        start, end = user_offsets[u_idx], user_offsets[u_idx + 1]
        positions = np.arange(start + 1, end)  # bỏ vị trí đầu tiên (start) — không đủ lịch sử
        sample_user_idx_list.append(np.full(len(positions), u_idx, dtype=np.int32))
        sample_position_list.append(positions.astype(np.int64))

    sample_user_idx = np.concatenate(sample_user_idx_list)
    sample_position = np.concatenate(sample_position_list)

    np.save(OUT_DIR / "sample_user_idx.npy", sample_user_idx)
    np.save(OUT_DIR / "sample_position.npy", sample_position)
    print(f"[build_sequences] {len(sample_position)} samples -> sample_user_idx.npy, sample_position.npy")


if __name__ == "__main__":
    build_sequences()
