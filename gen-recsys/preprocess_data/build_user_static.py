"""Pass 4 — user static features (static_branch input, trộn qua gate g_u với mat_u,
xem user_embedding.py).

Nguồn: user_features_27k.csv — chỉ dùng register_days + onehot_feat0-17 (đã CHỐT
2026-09-09, xem idea.md mục 4.5 điểm 4). Nhóm follow_user_num/fans_user_num/
friend_user_num/user_active_degree/is_lowactive_period — bỏ qua theo quyết định.

onehot_feat0-17: categorical id đã label-encode, "encrypted" theo dataset gốc
(kuairand.com/GitHub tác giả) — bản chất không công bố. Vẫn đưa vào static_branch,
không cần cơ chế riêng vì gate g_u (thiết kế ở model, không phải ở đây) tự động giảm
trọng số static_branch khi mat_u tăng.

[SỬA 2026-09-11] Range mỗi field RẤT KHÁC NHAU (đã đo trực tiếp: onehot_feat0 chỉ 2 giá
trị, onehot_feat3 tới 1471 giá trị) và 6/18 field CÓ NULL thật (onehot_feat4, 12-17 —
714-874 dòng null/tổng ~27,285 user, đã đo trực tiếp) — bản CŨ dùng fill_null(-1) rồi
lưu category id GỐC trực tiếp, để lại 2 rủi ro nếu dùng làm index nn.Embedding:
(a) index -1 âm sẽ crash (IndexError/CUDA device-side assert), (b) giữ category id GỐC
(không factorize liên tục [0,n)) lãng phí embedding table nếu id không liên tục (giống
lý do build_item_static.py đã factorize). Đã CHỐT: factorize TỪNG FIELD riêng (giống hệt
_factorize() ở build_item_static.py) — null -> code 0 riêng, category thật bắt đầu từ 1.

Output: user_static.npy (num_users,) structured array — 3 field cùng độ dài/cùng index
gộp chung 1 file (trước đây 3 file rời: user_static_onehot.npy/user_register_days.npy/
user_id_map.npy — luôn đọc/dùng song song nên gộp lại):
  user_id        int64          — id gốc, dùng để tra cứu ngược
  register_days  float32        — log1p + z-score
  onehot         int32, (18,)   — onehot_feat0-17, ĐÃ factorize riêng từng field thành
                                  index liên tục [0, n_field), null -> code 0 riêng
onehot_num_categories.npy (18,) int32 — số category (đã +1 cho null) mỗi field, dùng để
                                khởi tạo nn.Embedding riêng từng field ở user_embedding.py.
"""

from pathlib import Path

import numpy as np
import polars as pl

from schema import USER_STATIC_FIELD, USER_STATIC_ONEHOT_FEATS

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-Pure-extracted\KuaiRand-Pure\data")
USER_FEATURES_FILE = LOG_DIR / "user_features_pure.csv"
OUT_DIR = Path(__file__).parent / "output"


def _factorize(series: pl.Series) -> np.ndarray:
    """category id -> index liên tục [0, n), null gộp thành 1 code riêng (0) — khớp
    _factorize() ở build_item_static.py, tránh index -1 âm (không hợp lệ cho nn.Embedding)."""
    codes, _ = series.to_pandas().factorize()
    return (codes + 1).astype(np.int32)


def build_user_static() -> None:
    df = pl.read_csv(
        USER_FEATURES_FILE,
        columns=["user_id", USER_STATIC_FIELD, *USER_STATIC_ONEHOT_FEATS],
    ).sort("user_id")

    onehot_cols = [_factorize(df[field]) for field in USER_STATIC_ONEHOT_FEATS]
    onehot = np.stack(onehot_cols, axis=1)  # (num_users, 18)
    onehot_num_categories = np.array([col.max() + 1 for col in onehot_cols], dtype=np.int32)  # (18,)

    register_days = df[USER_STATIC_FIELD].fill_null(0).to_numpy().astype(np.float64)
    register_days_log = np.log1p(np.clip(register_days, a_min=0, a_max=None))
    mean, std = register_days_log.mean(), register_days_log.std()
    register_days_norm = ((register_days_log - mean) / (std if std > 0 else 1.0)).astype(np.float32)

    num_onehot = onehot.shape[1]
    user_static = np.empty(
        len(df),
        dtype=np.dtype([("user_id", np.int64), ("register_days", np.float32), ("onehot", np.int32, (num_onehot,))]),
    )
    user_static["user_id"] = df["user_id"].to_numpy().astype(np.int64)
    user_static["register_days"] = register_days_norm
    user_static["onehot"] = onehot

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "user_static.npy", user_static)
    np.save(OUT_DIR / "onehot_num_categories.npy", onehot_num_categories)
    print(f"[build_user_static] {len(df)} users -> user_static.npy (num_categories per field: {onehot_num_categories.tolist()})")


if __name__ == "__main__":
    build_user_static()
