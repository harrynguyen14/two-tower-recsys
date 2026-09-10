"""Pass 4 — user static features (static_branch input, trộn qua gate g_u với mat_u).

Nguồn: user_features_27k.csv — chỉ dùng register_days + onehot_feat0-17 (đã CHỐT
2026-09-09, xem idea.md mục 4.5 điểm 4). Nhóm follow_user_num/fans_user_num/
friend_user_num/user_active_degree/is_lowactive_period — bỏ qua theo quyết định.

onehot_feat0-17: categorical id đã label-encode, "encrypted" theo dataset gốc
(kuairand.com/GitHub tác giả) — bản chất không công bố. Vẫn đưa vào static_branch,
không cần cơ chế riêng vì gate g_u (thiết kế ở model, không phải ở đây) tự động giảm
trọng số static_branch khi mat_u tăng.

Output: user_static.npy (num_users,) structured array — 3 field cùng độ dài/cùng index
gộp chung 1 file (trước đây 3 file rời: user_static_onehot.npy/user_register_days.npy/
user_id_map.npy — luôn đọc/dùng song song nên gộp lại):
  user_id        int64          — id gốc, dùng để tra cứu ngược
  register_days  float32        — log1p + z-score
  onehot         int32, (18,)   — onehot_feat0-17, giữ nguyên category id gốc (KHÔNG cần
                                  factorize lại vì dataset đã tự encode sẵn)
"""

from pathlib import Path

import numpy as np
import polars as pl

from schema import USER_STATIC_FIELD, USER_STATIC_ONEHOT_FEATS

LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data")
USER_FEATURES_FILE = LOG_DIR / "user_features_27k.csv"
OUT_DIR = Path(__file__).parent / "output"


def build_user_static() -> None:
    df = pl.read_csv(
        USER_FEATURES_FILE,
        columns=["user_id", USER_STATIC_FIELD, *USER_STATIC_ONEHOT_FEATS],
    ).sort("user_id")

    onehot = df.select(USER_STATIC_ONEHOT_FEATS).fill_null(-1).to_numpy().astype(np.int32)

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
    print(f"[build_user_static] {len(df)} users -> user_static.npy")


if __name__ == "__main__":
    build_user_static()
