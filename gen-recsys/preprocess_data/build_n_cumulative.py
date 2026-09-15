"""Pass 1 — tính N_i và N_category lũy kế THEO THỜI GIAN (không leak tương lai).

Vì sao pass riêng: N_i (số tương tác 1 item đã nhận) và N_category (số item RIÊNG
BIỆT cùng tag đã có >=1 tương tác) đều thay đổi theo thời gian và cần tra cứu được
TẠI BẤT KỲ thời điểm t nào khi build sequence/interactions (Pass 2, Pass 5) — nên phải
tính xong và lưu dạng lũy kế TRƯỚC, thay vì tính lại mỗi lần cần dùng.

KHÔNG dùng video_features_statistic_*.csv (play_cnt tổng...) cho N_i — đó là snapshot
cuối, sẽ leak thông tin tương lai (idea.md mục 4.5, điểm 4).

Output: CSR-style flat arrays (KHÔNG dict theo key — đã đo thực nghiệm: 88,905 item
trên chỉ 100k dòng subset mất ~10s chỉ để build dict do vòng lặp Python group_by; với
32,038,725 item thật, cách này KHÔNG scale được). Mỗi nhóm gồm 3 file:
  {name}_ids.npy         — id (item_id hoặc category) đã unique + sort tăng dần
  {name}_offsets.npy     — offsets[i] = vị trí bắt đầu của ids[i] trong mảng phẳng dưới
  {name}_events.npy      — structured array [("t", int64), ("count", int32)], đã sort
                           theo (id, t), nối liền tất cả các id (timestamps + counts
                           gộp chung 1 file vì luôn cùng độ dài, luôn đọc/cắt song song)

Tra cứu N(id, t): 2 lần np.searchsorted liên tiếp (id trong ids, rồi t trong đoạn con
timestamps[offsets[i]:offsets[i+1]]) — hoàn toàn vector hóa, KHÔNG vòng lặp Python.
"""

from pathlib import Path

import numpy as np
import polars as pl

from schema import LOG_SCHEMA, VIDEO_BASIC_CATEGORY_FIELD

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-Pure-extracted\KuaiRand-Pure\data")
LOG_STANDARD_FILES = [
    LOG_DIR / "log_standard_4_08_to_4_21_pure.csv",
    LOG_DIR / "log_standard_4_22_to_5_08_pure.csv",
]
VIDEO_BASIC_FILE = LOG_DIR / "video_features_basic_pure.csv"
OUT_DIR = Path(__file__).parent / "output"


def _first_interaction_per_item() -> pl.DataFrame:
    """Với mỗi video_id, thời điểm nó XUẤT HIỆN LẦN ĐẦU trong log (dùng để đếm N_category)."""
    scans = [
        pl.scan_csv(f, schema_overrides={"video_id": pl.Int64, "user_id": pl.Int64, "time_ms": pl.Int64})
        for f in LOG_STANDARD_FILES
    ]
    lazy = pl.concat(scans).select(["video_id", "time_ms"])
    return (
        lazy.group_by("video_id")
        .agg(pl.col("time_ms").min().alias("first_seen_ms"))
        .collect(streaming=True)
    )


def _save_csr(name: str, id_col: pl.Series, time_col: pl.Series, count_col: pl.Series) -> None:
    """Ghi 3 file CSR-style — hoàn toàn vector hóa, KHÔNG vòng lặp Python.

    timestamps + counts gộp thành 1 structured array ({name}_events.npy) vì luôn
    cùng độ dài và luôn được đọc/cắt slice song song (xem lookup_n_at_t*).
    """
    ids_np = id_col.to_numpy()
    ts_np = time_col.to_numpy()
    cnt_np = count_col.to_numpy().astype(np.int32)

    # ids_np đã sort theo (id, time) từ trước khi gọi hàm này (xem 2 hàm build_* dưới)
    unique_ids, first_idx = np.unique(ids_np, return_index=True)
    offsets = np.append(first_idx, len(ids_np)).astype(np.int64)

    events = np.empty(len(ts_np), dtype=np.dtype([("t", ts_np.dtype), ("count", np.int32)]))
    events["t"] = ts_np
    events["count"] = cnt_np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}_ids.npy", unique_ids)
    np.save(OUT_DIR / f"{name}_offsets.npy", offsets)
    np.save(OUT_DIR / f"{name}_events.npy", events)
    print(f"[_save_csr] {name}: {len(unique_ids)} unique ids, {len(ts_np)} total rows")


def build_item_n_cumulative() -> None:
    """N_i(item, t) = số dòng log có video_id=item và time_ms < t, lũy kế theo thời gian."""
    scans = [
        pl.scan_csv(f, schema_overrides={"video_id": pl.Int64, "time_ms": pl.Int64})
        for f in LOG_STANDARD_FILES
    ]
    lazy = pl.concat(scans).select(["video_id", "time_ms"]).sort(["video_id", "time_ms"])
    df = lazy.collect(streaming=True)

    # cumulative count trong từng nhóm video_id, theo đúng thứ tự thời gian đã sort
    # (vector hóa qua window function của polars, KHÔNG vòng lặp Python)
    df = df.with_columns(pl.int_range(1, pl.len() + 1).over("video_id").alias("cum_count"))

    _save_csr("item_N", df["video_id"], df["time_ms"], df["cum_count"])


def build_user_n_cumulative() -> None:
    """N_u(user, t) = số dòng log có user_id=user và time_ms < t, lũy kế theo thời gian.

    [THÊM 2026-09-14] Đối xứng hoàn toàn với build_item_n_cumulative, nhưng dùng cho
    user_weight THEO TỪNG VỊ TRÍ trong chuỗi (u_i), KHÔNG phải 1 scalar/chuỗi như trước.

    Vì sao cần: `user_weight` cũ tra 1 lần tại thời điểm dự đoán -> hằng theo i. Khi đưa
    vào attention dạng cặp (u_i, m_j), log(u_i) hằng theo hàng i sẽ GỘP THẲNG vào hệ số
    β — tức thoái hóa về đúng cơ chế λ·log(mat_j) đã bỏ 2026-09-13 (gradient đo được
    ~1e-17). u_i per-position mới tạo được biến thiên thật: mọi user đều cold ở token đầu
    chuỗi của mình và warm dần về cuối — đây đồng thời là trục ngắn/dài hạn.
    """
    scans = [
        pl.scan_csv(f, schema_overrides={"user_id": pl.Int64, "time_ms": pl.Int64})
        for f in LOG_STANDARD_FILES
    ]
    lazy = pl.concat(scans).select(["user_id", "time_ms"]).sort(["user_id", "time_ms"])
    df = lazy.collect(streaming=True)

    df = df.with_columns(pl.int_range(1, pl.len() + 1).over("user_id").alias("cum_count"))

    _save_csr("user_N", df["user_id"], df["time_ms"], df["cum_count"])


def build_category_n_cumulative() -> None:
    """N_category(tag, t) = số video_id RIÊNG BIỆT cùng tag có first_seen_ms < t, lũy kế theo thời gian."""
    video_basic = pl.read_csv(VIDEO_BASIC_FILE, columns=["video_id", VIDEO_BASIC_CATEGORY_FIELD])
    first_seen = _first_interaction_per_item()

    joined = first_seen.join(video_basic, on="video_id", how="inner").sort(
        [VIDEO_BASIC_CATEGORY_FIELD, "first_seen_ms"]
    )
    joined = joined.with_columns(
        pl.int_range(1, pl.len() + 1).over(VIDEO_BASIC_CATEGORY_FIELD).alias("cum_count")
    )
    # category là string (tag) — encode thành int trước khi lưu (CSR cần id dạng số để sort/searchsorted)
    tag_codes, tag_uniques = joined[VIDEO_BASIC_CATEGORY_FIELD].to_pandas().factorize(sort=True)
    joined = joined.with_columns(pl.Series("tag_code", tag_codes.astype(np.int64)))

    _save_csr("category_N", joined["tag_code"], joined["first_seen_ms"], joined["cum_count"])
    np.save(OUT_DIR / "category_N_tag_names.npy", np.array(tag_uniques, dtype=object), allow_pickle=True)


def lookup_n_at_t(name: str, entity_id: int, t: int) -> int:
    """Tra cứu N tại thời điểm t — 2 lần searchsorted, KHÔNG vòng lặp Python.

    `name` = "item_N" hoặc "category_N" (tag_code, không phải tag string — dùng
    category_N_tag_names.npy để map ngược nếu cần).
    """
    ids = np.load(OUT_DIR / f"{name}_ids.npy")
    offsets = np.load(OUT_DIR / f"{name}_offsets.npy")
    events = np.load(OUT_DIR / f"{name}_events.npy")

    id_pos = np.searchsorted(ids, entity_id)
    if id_pos >= len(ids) or ids[id_pos] != entity_id:
        return 0  # entity chưa từng xuất hiện trước thời điểm t (hoặc không tồn tại) -> N=0

    start, end = offsets[id_pos], offsets[id_pos + 1]
    seg = events[start:end]
    # [SỬA 2026-09-11] side="left" — vị trí = số phần tử có t < query_t, tức N TRƯỚC thời
    # điểm t, KHÔNG tính chính sự kiện tại t. Bug đã xác nhận: side="right" - 1 trỏ đúng vào
    # chính sự kiện có event_t == query_t (khi trùng), đếm LUÔN sự kiện đó -> leak 1 đơn vị
    # thông tin tương lai (biết trước label/candidate này sẽ xảy ra tại t). count[j] (0-indexed)
    # = j+1 (cộng dồn từ đầu, xem build_item_n_cumulative) nên count TRƯỚC t = count tại vị
    # trí (side="left" - 1).
    idx = np.searchsorted(seg["t"], t, side="left")
    return int(seg["count"][idx - 1]) if idx > 0 else 0


def build_n_cache(name: str) -> dict:
    """Build 1 LẦN DUY NHẤT flat_struct (322M phần tử, cố định theo dữ liệu — KHÔNG phụ
    thuộc entity_ids/ts của bất kỳ query nào) — dùng cho training loop gọi lookup lặp lại
    hàng nghìn lần (mỗi step 2 lần: hist + candidate). KHÔNG gọi lại _build bên trong
    lookup_n_at_t_batch cho mỗi lần tra — đã xác nhận OOM thật khi build lại mỗi lần gọi
    trong vòng lặp training (ArrayMemoryError 2.40 GiB ở np.arange(322,278,385), do RAM
    không kịp giải phóng giữa các lần gọi liên tiếp).

    Trả về dict {ids, offsets, flat_struct, counts} — truyền vào lookup_n_at_t_batch_cached.
    """
    ids = np.load(OUT_DIR / f"{name}_ids.npy")
    offsets = np.load(OUT_DIR / f"{name}_offsets.npy")
    events = np.load(OUT_DIR / f"{name}_events.npy")
    timestamps = events["t"]
    counts = events["count"]

    entity_idx_per_row = np.searchsorted(offsets, np.arange(len(timestamps)), side="right") - 1
    entity_id_per_row = ids[entity_idx_per_row]

    dtype = np.dtype([("id", entity_id_per_row.dtype), ("t", timestamps.dtype)])
    flat_struct = np.empty(len(timestamps), dtype=dtype)
    flat_struct["id"] = entity_id_per_row
    flat_struct["t"] = timestamps
    # flat_struct đã sort đúng theo (id, t) vì dữ liệu gốc được sort trước khi build CSR

    return {"ids": ids, "offsets": offsets, "flat_struct": flat_struct, "counts": counts, "dtype": dtype}


def lookup_n_at_t_batch_cached(
    cache: dict, entity_ids: np.ndarray, ts: np.ndarray, batch_size: int = 20_000_000
) -> np.ndarray:
    """Tra N(entity_id, t) dùng cache đã build sẵn (xem build_n_cache) — KHÔNG đọc lại file
    hay build lại flat_struct, chỉ làm phần searchsorted phụ thuộc query."""
    ids, offsets, flat_struct, counts, dtype = (
        cache["ids"], cache["offsets"], cache["flat_struct"], cache["counts"], cache["dtype"]
    )

    result = np.zeros(len(entity_ids), dtype=np.int32)
    for start in range(0, len(entity_ids), batch_size):
        end = min(start + batch_size, len(entity_ids))
        batch_ids = entity_ids[start:end]
        batch_ts = ts[start:end]

        id_pos = np.searchsorted(ids, batch_ids)
        id_pos_clipped = np.clip(id_pos, 0, len(ids) - 1)
        valid = ids[id_pos_clipped] == batch_ids

        query_struct = np.empty(len(batch_ids), dtype=dtype)
        query_struct["id"] = batch_ids
        query_struct["t"] = batch_ts

        # [SỬA 2026-09-11] side="left" (KHÔNG trừ 1) — searchsorted trên structured array so
        # sánh từ điển (id, t): vị trí trả về = số phần tử flat_struct có (id, t) < (query_id,
        # query_t) theo thứ tự đó, tức đúng "N TRƯỚC thời điểm t" (không tính chính sự kiện có
        # t == query_t nếu trùng). Bug đã xác nhận: side="right" - 1 trỏ đúng vào chính sự kiện
        # trùng (id, t) với query (khi query_t == 1 event_t thật, ví dụ chính label/candidate
        # đang tra) -> đếm LUÔN sự kiện đó -> leak 1 đơn vị thông tin tương lai. Trừ đi 1 SAU
        # khi lấy seg_pos (không phải trừ trong searchsorted) để tra counts[] tại đúng dòng
        # TRƯỚC nó.
        seg_pos = np.searchsorted(flat_struct, query_struct, side="left") - 1

        starts = offsets[id_pos_clipped]
        ends = offsets[id_pos_clipped + 1]
        ok = valid & (seg_pos >= starts) & (seg_pos < ends)

        batch_result = np.zeros(len(batch_ids), dtype=np.int32)
        batch_result[ok] = counts[seg_pos[ok]]
        result[start:end] = batch_result
    return result


def lookup_n_at_t_batch(name: str, entity_ids: np.ndarray, ts: np.ndarray, batch_size: int = 20_000_000) -> np.ndarray:
    """Phiên bản vector hóa của lookup_n_at_t cho nhiều (entity_id, t) cùng lúc — dùng ở
    Pass 5 (build_interactions.py, gọi ĐÚNG 1 LẦN cho mỗi name) để tránh vòng lặp Python
    trên hàng chục triệu sample.

    [SỬA 2026-09-09] Kỹ thuật namespace-offset ban đầu (nhân entity_index với 1 hằng số
    lớn rồi cộng vào timestamp) bị TRÀN SỐ int64 thật (đã xác nhận qua test cụ thể: với
    88,905 item test — chưa tới 32,038,725 item thật — entity_idx_max * NAMESPACE đã vượt
    np.iinfo(np.int64).max, gây searchsorted trả về vị trí sai hoàn toàn). Đã CHỐT thay
    bằng structured array (entity_id, timestamp) — searchsorted trên structured/void array
    so sánh đúng thứ tự từ điển (trước theo id, sau theo t trong cùng id) mà KHÔNG cần
    phép cộng/nhân số học nào, không có rủi ro tràn số.

    [SỬA 2026-09-10 — OOM đã xác nhận: ArrayMemoryError 4.80 GiB khi build query_struct
    cho toàn bộ 322,251,100 sample cùng lúc, trên máy 31.7GB RAM]: flat_struct (dựng từ
    item_N/category_N, cố định theo dữ liệu — KHÔNG phụ thuộc entity_ids/ts của query) chỉ
    build 1 LẦN; query_struct (phụ thuộc entity_ids/ts, có thể rất lớn — 322M+ sample) chia
    thành từng batch nhỏ (batch_size phần tử/lần) để tránh giữ 2 mảng ~4.8GB cùng lúc.

    [SỬA 2026-09-11 — dùng trong TRAINING LOOP thì KHÔNG dùng hàm này]: hàm này build lại
    flat_struct MỖI LẦN GỌI (phù hợp Pass 5 — chỉ gọi 1 lần/name) — nếu gọi lặp lại hàng
    nghìn lần (training loop, mỗi step 2 lần) sẽ OOM vì build lại 322M-phần-tử liên tục,
    không kịp giải phóng RAM giữa các lần gọi (đã xác nhận qua traceback thật). Training
    loop PHẢI dùng build_n_cache() 1 lần lúc khởi tạo + lookup_n_at_t_batch_cached() mỗi
    step, xem train.py.
    """
    cache = build_n_cache(name)
    return lookup_n_at_t_batch_cached(cache, entity_ids, ts, batch_size)


if __name__ == "__main__":
    build_item_n_cumulative()
    build_user_n_cumulative()
    build_category_n_cumulative()
