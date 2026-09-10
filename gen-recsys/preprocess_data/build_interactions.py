"""Pass 5 — split train/val/test theo temporal boundary + gán cờ is_user_cold/is_item_cold.

[VIẾT LẠI 2026-09-09 — khớp thiết kế mới của build_sequences.py (history + index, KHÔNG
còn sequences.npy/attention_mask.npy/labels.npy/label_timestamps.npy nữa)]:

Input: history_meta.npy/user_offsets.npy/sample_user_idx.npy/sample_position.npy (từ Pass
2, history_meta.npy = video_id + timestamp gộp chung, xem build_sequences.py), item_N_ids.npy...
(từ Pass 1).

label = history_meta["video_id"][sample_position]; label_timestamp = history_meta["t"][sample_position]
N_u (tại thời điểm sample) = sample_position - user_offsets[sample_user_idx] — số token
lịch sử user đã tích lũy TRƯỚC vị trí này (không tính chính label).

Split: 80/10/10 theo percentile của timestamp LABEL — sample mà label rơi vào khoảng
thời gian sau cùng thuộc test, giữa thuộc val, đầu thuộc train — đã CHỐT 2026-09-09
(idea.md mục 4.5 điểm 4), đúng bản chất "dự đoán tương lai", không leak.

is_user_cold/is_item_cold: CHỈ dùng để gán nhãn khi EVAL (filter runtime, xem idea.md
mục 4.5 — "1 file duy nhất, filter bằng cột, không tách file"), KHÔNG ảnh hưởng model
(model luôn dùng mat_u/mat_i liên tục). Ngưỡng N<5 = cold (đã CHỐT 2026-09-09) — thô,
chỉ để so sánh nhóm dễ đọc trong report.

τ_u, τ_i: 2 hằng số RIÊNG BIỆT, khởi tạo từ thống kê thật đo trực tiếp trên toàn bộ
log_standard:
  - τ_u = 48 (MEDIAN N_u/user, đo ở idea.md mục 0).
  - τ_i = 10 (MEAN N_i/item, đo trực tiếp — median N_i=1 quá nhỏ, sẽ làm tanh(N_i/τ_i)
    bão hòa gần như ngay lập tức mất hết độ phân giải ở vùng N nhỏ quan trọng nhất).
Đây là giá trị KHỞI TẠO cho nn.Parameter học được trong model thật (không phải hằng
số cố định vĩnh viễn) — xem idea.md "TRẠNG THÁI DỰ ÁN", việc CẦN LÀM TIẾP #1.

Output: {split}.npy (split = train/val/test) — structured array, 5 field cùng độ dài/cùng
index gộp chung 1 file (trước đây 5 file rời mỗi split: {split}_indices/_mat_u/_mat_i/
_is_user_cold/_is_item_cold.npy — luôn đọc/dùng song song nên gộp lại):
  index         int64    — vị trí gốc trong sample_user_idx/sample_position (mảng đầy đủ)
  mat_u         float32  — tanh(N_u / τ_u)
  mat_i         float32  — tanh(N_i / τ_i)
  is_user_cold  bool     — N_u < COLD_THRESHOLD_N
  is_item_cold  bool     — N_i < COLD_THRESHOLD_N
"""

from pathlib import Path

import numpy as np

from build_n_cumulative import lookup_n_at_t_batch

OUT_DIR = Path(__file__).parent / "output"
COLD_THRESHOLD_N = 5  # CHỐT 2026-09-09 — chỉ dùng gán nhãn eval, không ảnh hưởng model
TAU_U_INIT = 48.0  # median N_u, đo ở idea.md mục 0
TAU_I_INIT = 10.0  # mean N_i, đo trực tiếp 2026-09-09 (median N_i=1 quá nhỏ, không dùng)


def _mat(n: np.ndarray, tau: float) -> np.ndarray:
    return np.tanh(n / tau)


def build_interactions() -> None:
    history_meta = np.load(OUT_DIR / "history_meta.npy")
    user_offsets = np.load(OUT_DIR / "user_offsets.npy")
    sample_user_idx = np.load(OUT_DIR / "sample_user_idx.npy")
    sample_position = np.load(OUT_DIR / "sample_position.npy")

    labels = history_meta["video_id"][sample_position]
    label_timestamps = history_meta["t"][sample_position].astype(np.int64)

    # N_u tại thời điểm sample = số token lịch sử TRƯỚC vị trí này, tính từ đầu user
    # (sample_position - user_offsets[user] = số token đã đi qua, KHÔNG tính chính label
    # vì label nằm TẠI sample_position, token lịch sử là [user_start, sample_position)).
    n_u = (sample_position - user_offsets[sample_user_idx]).astype(np.float64)
    mat_u = _mat(n_u, TAU_U_INIT)
    is_user_cold = n_u < COLD_THRESHOLD_N

    n_i = lookup_n_at_t_batch("item_N", labels, label_timestamps).astype(np.float64)
    mat_i = _mat(n_i, TAU_I_INIT)
    is_item_cold = n_i < COLD_THRESHOLD_N

    # temporal split 80/10/10 theo percentile của label_timestamps
    p80, p90 = np.percentile(label_timestamps, [80, 90])
    train_mask = label_timestamps <= p80
    val_mask = (label_timestamps > p80) & (label_timestamps <= p90)
    test_mask = label_timestamps > p90

    split_dtype = np.dtype([
        ("index", np.int64),
        ("mat_u", np.float32),
        ("mat_i", np.float32),
        ("is_user_cold", np.bool_),
        ("is_item_cold", np.bool_),
    ])

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        idx = np.where(mask)[0]
        split_arr = np.empty(len(idx), dtype=split_dtype)
        split_arr["index"] = idx
        split_arr["mat_u"] = mat_u[idx].astype(np.float32)
        split_arr["mat_i"] = mat_i[idx].astype(np.float32)
        split_arr["is_user_cold"] = is_user_cold[idx]
        split_arr["is_item_cold"] = is_item_cold[idx]

        np.save(OUT_DIR / f"{split_name}.npy", split_arr)
        print(f"[build_interactions] {split_name}: {len(idx)} samples -> {split_name}.npy")


if __name__ == "__main__":
    build_interactions()
