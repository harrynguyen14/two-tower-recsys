"""Pass 5 — split train/val/test theo temporal boundary + gán cờ is_user_cold/is_item_cold.

[SỬA 2026-09-14 — QUAN TRỌNG, sửa lỗi nguyên tắc split cho cold-start] Phiên bản trước
(2026-09-13) định nghĩa is_user_cold/is_item_cold theo NGƯỠNG N tại từng sample riêng lẻ
(N_u/N_i < COLD_THRESHOLD_N) — nhưng đây KHÔNG phải strict holdout: CÙNG 1 user/item có
thể xuất hiện ở CẢ train lẫn test (chỉ khác N tại mỗi thời điểm). Model ĐÃ được train trực
tiếp trên chính user/item đó (ở giai đoạn N nhỏ) trước khi bị đánh giá "cold" ở test — đây
đo "warm-start với ít tương tác" (low-interaction regime), KHÔNG phải "cold-start thật"
(model dự đoán cho thực thể CHƯA TỪNG THẤY).

Đã sửa is_user_cold theo STRICT HOLDOUT thật: user được coi là cold-user-holdout NẾU VÀ
CHỈ NẾU user đó có `first_seen_ms > boundary` (mọi tương tác của user này rơi hoàn toàn
SAU mốc train/test — user KHÔNG BAO GIỜ xuất hiện trong train). is_user_cold giờ là thuộc
tính của USER_ID (cố định), KHÔNG phải của từng sample riêng lẻ.

is_item_cold GIỮ NGUYÊN threshold-based (N_i < COLD_THRESHOLD_N tại thời điểm sample) —
ĐÃ ĐO TRỰC TIẾP (xem result.md 2026-09-14): strict item-holdout KHÔNG khả thi trên
KuaiRand-Pure ở BẤT KỲ mốc train/test nào — 99% trong 7,551 item "ra mắt" (first_seen)
chỉ trong 3 NGÀY ĐẦU của cửa sổ log 29.5 ngày (2,771+3,328+1,380 item ở ngày 0-2), sau đó
gần như KHÔNG có item mới nào xuất hiện nữa (chỉ vài đơn vị/ngày). Ở mốc p80 hiện tại,
strict item-holdout chỉ cho ĐÚNG 11 item / 27 dòng tương tác — quá mỏng để đo bất kỳ metric
nào có ý nghĩa. Đây là ĐẶC ĐIỂM CỦA CHÍNH DATASET (Pure là catalog "chính" đã lọc sẵn, khác
27K có 32M item và luồng item mới liên tục), KHÔNG phải lỗi chọn mốc.

**GIỚI HẠN CẦN GHI RÕ TRONG BÁO CÁO**: is_item_cold trên Pure đo "item ít tương tác tính
đến thời điểm đó" (low-interaction), KHÔNG phải "item hoàn toàn mới, chưa từng huấn luyện"
(true zero-shot cold-start). Kết luận về cold-ITEM rút ra từ is_item_cold cần diễn giải
đúng phạm vi này — khác is_user_cold (giờ ĐÃ là true zero-shot, strict holdout thật).

Khởi tạo τ (xem learnable_thresholds.py, đo lại cho Pure — không đổi so với 2026-09-13):
  COLD_THRESHOLD_N = 10 (từ schema.py, chỉ còn dùng cho is_item_cold)
  TAU_U_INIT = 35.0    — median N_u tại thời điểm sample
  TAU_I_INIT = 599.3   — mean N_i tại thời điểm sample

Input: history_meta.npy/user_offsets.npy/sample_user_idx.npy/sample_position.npy (từ Pass
2), item_N_ids.npy... (từ Pass 1), user_ids_sorted.npy (để tra first_seen theo user_id).

label = history_meta["video_id"][sample_position]; label_timestamp = history_meta["t"][sample_position]
N_u (tại thời điểm sample) = sample_position - user_offsets[sample_user_idx] — số token
lịch sử user đã tích lũy TRƯỚC vị trí này (không tính chính label).

Split: 80/10/10 theo percentile của timestamp LABEL — sample mà label rơi vào khoảng
thời gian sau cùng thuộc test, giữa thuộc val, đầu thuộc train — đúng bản chất "dự đoán
tương lai", không leak.

**Ràng buộc mới sau khi sửa**: mọi sample của 1 user cold-holdout PHẢI rơi vào val/test
(không được lọt vào train) — vì user này chưa từng có mặt trong train. Điều này TỰ ĐỘNG
đúng vì user cold-holdout được định nghĩa là user có first_seen > boundary train/test
CHÍNH LÀ mốc p80 — token đầu tiên của user này đã sau p80, nên MỌI token/sample của user
đó (kể cả các vị trí sau) cũng sau p80, tự động rơi vào val/test.

τ_u, τ_i CHỈ dùng để gán is_item_cold + user_weight/item_weight tham khảo — KHÔNG phải
giá trị model thực sự dùng (model học τ riêng qua nn.Parameter — xem learnable_thresholds.py).

Output: {split}.npy (split = train/val/test) — structured array:
  index         int64    — vị trí gốc trong sample_user_idx/sample_position (mảng đầy đủ)
  user_weight   float32  — tanh(N_u / τ_u) — CHỈ để tham khảo, KHÔNG dùng trong model
  item_weight   float32  — tanh(N_i / τ_i) — tương tự
  is_user_cold  bool     — STRICT HOLDOUT: user có first_seen_ms > boundary train/test
  is_item_cold  bool     — THRESHOLD-BASED (giới hạn, xem cảnh báo trên): N_i < COLD_THRESHOLD_N
"""

from pathlib import Path

import numpy as np

from build_n_cumulative import lookup_n_at_t_batch
from schema import COLD_THRESHOLD_N, LOW_HISTORY_N

OUT_DIR = Path(__file__).parent / "output"
TAU_U_INIT = 35.0    # median N_u tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13
TAU_I_INIT = 599.3   # mean N_i tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13


def _mat(n: np.ndarray, tau: float) -> np.ndarray:
    return np.tanh(n / tau)


def build_interactions() -> None:
    history_meta = np.load(OUT_DIR / "history_meta.npy")
    user_offsets = np.load(OUT_DIR / "user_offsets.npy")
    user_ids_sorted = np.load(OUT_DIR / "user_ids_sorted.npy")
    sample_user_idx = np.load(OUT_DIR / "sample_user_idx.npy")
    sample_position = np.load(OUT_DIR / "sample_position.npy")

    labels = history_meta["video_id"][sample_position]
    label_timestamps = history_meta["t"][sample_position].astype(np.int64)

    # temporal split 80/10/10 theo percentile của label_timestamps — TÍNH TRƯỚC vì
    # boundary này CŨNG dùng làm mốc strict user-holdout (xem docstring module).
    p80, p90 = np.percentile(label_timestamps, [80, 90])
    train_mask = label_timestamps <= p80
    val_mask = (label_timestamps > p80) & (label_timestamps <= p90)
    test_mask = label_timestamps > p90

    # N_u tại thời điểm sample = số token lịch sử TRƯỚC vị trí này, tính từ đầu user
    # (sample_position - user_offsets[user] = số token đã đi qua, KHÔNG tính chính label
    # vì label nằm TẠI sample_position, token lịch sử là [user_start, sample_position)).
    n_u = (sample_position - user_offsets[sample_user_idx]).astype(np.float64)
    user_weight = _mat(n_u, TAU_U_INIT)

    # [SỬA 2026-09-14] STRICT HOLDOUT cho user — first_seen_ms của MỖI user (token đầu
    # tiên trong lịch sử của họ, tại user_offsets[u]) so với boundary p80. User có
    # first_seen > p80 nghĩa là CHƯA TỪNG xuất hiện trong train — mọi sample của user này
    # tự động rơi vào val/test (vì token đầu tiên đã sau p80, nên mọi token sau đó cũng
    # sau p80). Đây là thuộc tính CỦA USER (không đổi giữa các sample cùng user), khác
    # is_item_cold vẫn theo threshold N tại từng thời điểm.
    user_first_seen_ms = history_meta["t"][user_offsets[:-1]].astype(np.int64)  # (num_users,) — token đầu tiên mỗi user
    is_user_cold_by_user = user_first_seen_ms > p80  # (num_users,) bool
    is_user_cold = is_user_cold_by_user[sample_user_idx]  # broadcast theo sample

    # [THÊM 2026-09-15] Cờ THỨ HAI cho user, SONG SONG với is_user_cold (không thay thế).
    #
    # Vì sao cần — đo trực tiếp trên dữ liệu đã build: strict holdout khiến ô cold/cold có
    # 0 sample trong TRAIN và 18 sample trong val+test. Cơ chế γ (confidence_attention.py)
    # sinh ra riêng cho ô đó nên KHÔNG có ví dụ nào để học, và không đủ mẫu để đo. Với
    # N_u < LOW_HISTORY_N: train 31,538 / val 81 / test 65 — học được và đo được xu hướng.
    #
    # Hai cờ trả lời HAI câu hỏi khác nhau, cả hai đều hợp lệ, phải báo cáo RIÊNG:
    #   is_user_cold        (strict) — "user CHƯA TỪNG xuất hiện" = zero-shot thật. Nghiêm
    #                                  nhất, n nhỏ. Giữ nguyên, không đụng vào.
    #   is_user_lowhistory          — "user ít lịch sử TẠI thời điểm dự đoán" = few-shot.
    #                                  Đây là định nghĩa cold-start phổ biến trong văn liệu.
    #
    # Đây KHÔNG phải nới ngưỡng để ra kết quả đẹp: γ vốn làm việc trên u_i = tanh(N_u/τ_u),
    # một đại lượng LIÊN TỤC theo N_u. Cờ few-shot khớp đúng trục đó; cờ strict holdout
    # (thuộc tính của USER, không phải của thời điểm) mới là cái lệch khỏi cơ chế.
    is_user_lowhistory = n_u < LOW_HISTORY_N

    n_i = lookup_n_at_t_batch("item_N", labels, label_timestamps).astype(np.float64)
    item_weight = _mat(n_i, TAU_I_INIT)
    is_item_cold = n_i < COLD_THRESHOLD_N  # threshold-based — xem giới hạn ở docstring module

    # Kiểm tra ràng buộc: KHÔNG sample nào của user cold-holdout được lọt vào train
    # (nếu vi phạm, có nghĩa boundary tính sai hoặc user_offsets không đúng thứ tự thời gian)
    violating = is_user_cold & train_mask
    if violating.any():
        raise AssertionError(
            f"{violating.sum()} sample của user cold-holdout lọt vào train — "
            "vi phạm strict holdout, kiểm tra lại user_offsets/boundary."
        )

    split_dtype = np.dtype([
        ("index", np.int64),
        ("user_weight", np.float32),
        ("item_weight", np.float32),
        ("is_user_cold", np.bool_),
        ("is_item_cold", np.bool_),
        ("is_user_lowhistory", np.bool_),  # [THÊM 2026-09-15] few-shot, song song strict holdout
    ])

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        idx = np.where(mask)[0]
        split_arr = np.empty(len(idx), dtype=split_dtype)
        split_arr["index"] = idx
        split_arr["user_weight"] = user_weight[idx].astype(np.float32)
        split_arr["item_weight"] = item_weight[idx].astype(np.float32)
        split_arr["is_user_cold"] = is_user_cold[idx]
        split_arr["is_item_cold"] = is_item_cold[idx]
        split_arr["is_user_lowhistory"] = is_user_lowhistory[idx]

        np.save(OUT_DIR / f"{split_name}.npy", split_arr)
        n_cc_strict = int((is_user_cold[idx] & is_item_cold[idx]).sum())
        n_cc_low = int((is_user_lowhistory[idx] & is_item_cold[idx]).sum())
        print(
            f"[build_interactions] {split_name}: {len(idx)} samples -> {split_name}.npy "
            f"(cold_user_STRICT={is_user_cold[idx].mean()*100:.2f}%, "
            f"lowhistory={is_user_lowhistory[idx].mean()*100:.2f}%, "
            f"cold_item_threshold={is_item_cold[idx].mean()*100:.2f}%) "
            f"| o COLD/COLD: strict n={n_cc_strict}, lowhistory n={n_cc_low}"
        )

    n_cold_users = int(is_user_cold_by_user.sum())
    print(
        f"[build_interactions] strict user-holdout: {n_cold_users}/{len(user_ids_sorted)} "
        f"user ({n_cold_users/len(user_ids_sorted)*100:.1f}%) chưa từng xuất hiện trong train"
    )


if __name__ == "__main__":
    build_interactions()
