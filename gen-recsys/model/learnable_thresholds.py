"""Độ trưởng thành của item: N_i(t) -> percentile. Xem formula.md §0.

[SỬA 2026-10-05] `item_weight` đổi từ `tanh(N_i/τ_i)` sang **percentile** (formula.md §0).
`N_i` là đếm TUYỆT ĐỐI, không có đơn vị bất biến — nó phụ thuộc cỡ dataset (Pure 1.44M vs
27K 322M tương tác = 224x), độ dài cửa sổ log, VÀ thời điểm serving (cùng một item, sau 6
tháng N_i phình lên). Cái cuối nghiêm trọng nhất vì xảy ra NGAY TRONG PRODUCTION: tanh bão
hoà về 1 => log m_p -> 0 => số hạng `β_h log m_p` TẮT DẦN theo thời gian chạy. Đúng cơ chế
đã đo với u_t (|prod| yếu đi 1000x).

Percentile bất biến theo cả ba. Cùng kỹ thuật đã dùng cho play_ratio (D2Q,
build_sequences._compute_derived_action_fields) — ở đó giá trị thô phụ thuộc duration.

Đo trên Pure (7,551 item có N_i>0, median 57, max 10,424), N_i 10->100 là vùng DÀY quanh
median: tanh chỉ +0.149 còn percentile +0.512 (3.4x). tanh tiêu phí dải động cho đuôi hiếm
(N_i 999->4887 chỉ +0.069, gần chết).

[XOÁ 2026-10-05] τ_i đã bỏ HẲN cùng các caller (train.py, eval.py, test_thresholds_scale).
percentile không cần ngưỡng nên không có gì để học.

[XOÁ 2026-10-06] Cả class `LearnableThresholds` đã bỏ cùng τ_u. formula.md §0 bỏ u_t
(27K không có cold user, N_u min=100) nên `user_weight` hết đối tượng phục vụ — và
attention đã KHÔNG đọc `log_u` ở bất kỳ số hạng nào, nên đó là đường ống chết: τ_u là
nn.Parameter nằm trong optimizer mà không bao giờ nhận gradient. Bỏ τ_u xong thì class
rỗng tham số (item_weight chỉ là identity), nên gọi `lookup_percentile` trực tiếp.
"""

from __future__ import annotations

import numpy as np


def item_percentile_table(n_i_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bảng tra N_i -> percentile [0,1], dựng 1 LẦN trên train.

    Mid-rank BẮT BUỘC (không phải rank thường): 94.6% item trên Pure có N_i trùng item khác
    (nhóm lớn nhất 117 item cùng N_i=13). Rank thường sẽ gán thứ hạng khác nhau cho các item
    GIỐNG HỆT nhau — cùng lỗi mà D2Q đã phải sửa bằng mid-rank ở play_ratio.

    Trả về (uniq, pct) đã sort theo N_i; tra bằng `lookup_percentile`.
    """
    uniq, counts = np.unique(np.asarray(n_i_all), return_counts=True)
    csum = np.concatenate([[0], np.cumsum(counts)])
    # mid-rank của mỗi giá trị duy nhất = trung bình các thứ hạng mà nó chiếm
    mid = (csum[:-1] + csum[1:] - 1) / 2.0
    denom = max(len(n_i_all) - 1, 1)
    return uniq.astype(np.int64), (mid / denom).astype(np.float32)


def lookup_percentile(uniq: np.ndarray, pct: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    """Tra percentile cho N_i bất kỳ (kể cả giá trị chưa từng thấy -> giá trị gần nhất bên
    dưới). O(log n), dùng được cả lúc serving: giữ sẵn (uniq, pct), không cần tính lại
    phân phối."""
    idx = np.searchsorted(uniq, np.asarray(n_i), side="right") - 1
    return pct[np.clip(idx, 0, len(pct) - 1)]


def _demo() -> None:
    """ponytail: self-check nhỏ nhất bắt được lỗi mid-rank và tính bất biến."""
    # 1. mid-rank: item CÙNG N_i phải nhận CÙNG percentile
    n_i = np.array([5, 5, 5, 10, 20, 20, 100])
    uniq, pct = item_percentile_table(n_i)
    got = lookup_percentile(uniq, pct, n_i)
    assert got[0] == got[1] == got[2], f"cùng N_i=5 phải cùng percentile, được {got[:3]}"
    assert got[4] == got[5], f"cùng N_i=20 phải cùng percentile, được {got[4:6]}"
    assert got[0] < got[3] < got[4] < got[6], f"phải đơn điệu theo N_i, được {got}"
    assert 0.0 <= got.min() and got.max() <= 1.0, "percentile phải trong [0,1]"

    # 2. BẤT BIẾN theo thang N_i — lý do chính bỏ tanh. Nhân đôi mọi N_i (mô phỏng dataset
    #    lớn hơn / hệ chạy lâu hơn): percentile KHÔNG đổi, tanh thì đổi hẳn.
    uniq2, pct2 = item_percentile_table(n_i * 2)
    got2 = lookup_percentile(uniq2, pct2, n_i * 2)
    assert np.allclose(got, got2), f"percentile phải bất biến theo thang N_i: {got} vs {got2}"
    tau_i_cu = 599.3  # τ_i cũ đã bỏ — giữ tại chỗ để check chứng minh được lý do bỏ
    t1, t2 = np.tanh(n_i / tau_i_cu), np.tanh(n_i * 2 / tau_i_cu)
    assert not np.allclose(t1, t2), "tanh LẼ RA phải đổi khi nhân đôi N_i (chứng minh lý do bỏ)"

    # 3. N_i chưa từng thấy -> clamp, không nổ
    assert lookup_percentile(uniq, pct, np.array([0, 10 ** 9])).shape == (2,)

    print("learnable_thresholds self-check OK")


if __name__ == "__main__":
    _demo()
