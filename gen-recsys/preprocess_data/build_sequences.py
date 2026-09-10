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

from pathlib import Path

import numpy as np
import polars as pl

from schema import ACTION_VECTOR_FIELDS, MAX_SEQ_LEN

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data")
LOG_STANDARD_FILES = [
    LOG_DIR / "log_standard_4_08_to_4_21_27k_part1.csv",
    LOG_DIR / "log_standard_4_08_to_4_21_27k_part2.csv",
    LOG_DIR / "log_standard_4_22_to_5_08_27k_part1.csv",
    LOG_DIR / "log_standard_4_22_to_5_08_27k_part2.csv",
]
OUT_DIR = Path(__file__).parent / "output"
NUM_BUCKETS = 100  # số bucket user_id — tránh OOM khi sort (đã xác nhận sort toàn cục OOM thật)


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


def _compute_derived_action_fields(df: pl.DataFrame, profile_max: float, comment_max: float) -> pl.DataFrame:
    """play_ratio, profile_stay_time_norm, comment_stay_time_norm — derived, chuẩn hóa
    theo max TOÀN CỤC (profile_max/comment_max tính 1 lần trước, xem _compute_global_log1p_max)."""
    return df.with_columns(
        (pl.col("play_time_ms") / pl.col("duration_ms").clip(lower_bound=1))
        .clip(0.0, 1.0)
        .alias("play_ratio"),
        (pl.col("profile_stay_time").log1p() / profile_max)
        .fill_nan(0.0)
        .alias("profile_stay_time_norm"),
        (pl.col("comment_stay_time").log1p() / comment_max)
        .fill_nan(0.0)
        .alias("comment_stay_time_norm"),
    )


def _iter_log_buckets():
    """Chia user_id thành NUM_BUCKETS nhóm, xử lý tuần tự — mỗi bucket chỉ lọc + sort
    phần dữ liệu thuộc về nó (nhỏ hơn NUM_BUCKETS lần), không giữ quá 1 bucket trong RAM.

    [SỬA 2026-09-09]: filter theo bucket TRƯỚC, tính derived fields SAU (đã verified qua
    test cụ thể: filter-trước-derived-sau chạy OK; derived-trước-filter OOM ngay bucket 0
    vì .max() buộc vật chất hóa toàn bộ 322M dòng mỗi bucket).
    """
    profile_max, comment_max = _compute_global_log1p_max()

    scans_base = [pl.scan_csv(f) for f in LOG_STANDARD_FILES]
    base = pl.concat(scans_base)

    for bucket_idx in range(NUM_BUCKETS):
        bucket_filtered = base.filter(pl.col("user_id") % NUM_BUCKETS == bucket_idx)
        bucket = (
            _compute_derived_action_fields(bucket_filtered, profile_max, comment_max)
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
        history_action_mm[write_pos:write_pos + n] = bucket_df.select(ACTION_VECTOR_FIELDS).to_numpy().astype(np.float32)

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
