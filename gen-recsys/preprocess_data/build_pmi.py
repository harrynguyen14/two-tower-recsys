"""Pass 6 — bảng PMI item-item cho attention bias. Xem formula.md §4.1.

PMI(i,j) = log[ c_ij * |U| / (df_i * df_j) ],  lưu PPMI = max(PMI, 0).

TÍNH TRÊN TRAIN-ONLY (cắt tại timestamp lớn nhất của train split) — dùng toàn log là
LEAK, vì bảng này tham gia dự đoán.

Đo được trên KuaiRand-Pure: 26.6% của N^2 có co-occurrence, median PMI +1.404, 88.6%
dương, corr(PMI, log popularity) = -0.498. Dense fp16 = 115 MB, vừa GPU — không cần
sparse (gather thưa trong attention inner loop đắt hơn phần tiết kiệm được).

Output: pmi_table.npy (num_items, num_items) float16.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

OUT_DIR = Path(__file__).parent / "output"


def build_pmi() -> None:
    hm = np.load(OUT_DIR / "history_meta.npy")
    vid, ts = hm["video_id"], hm["t"]
    item_static = np.load(OUT_DIR / "item_static.npy", mmap_mode="r")
    n_items = len(item_static)

    sample_position = np.load(OUT_DIR / "sample_position.npy")
    train = np.load(OUT_DIR / "train.npy")
    cutoff = ts[sample_position[train["index"]]].max()

    user_offsets = np.load(OUT_DIR / "user_offsets.npy")
    n_users = len(user_offsets) - 1
    uid = np.zeros(len(vid), dtype=np.int64)
    for u in range(n_users):
        uid[user_offsets[u]:user_offsets[u + 1]] = u

    keep = ts <= cutoff
    # Nhị phân: user CÓ xem item hay không. Đếm số lần xem sẽ làm popularity lấn át.
    M = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (uid[keep], vid[keep])),
        shape=(n_users, n_items),
    )
    M.data[:] = 1.0

    C = (M.T @ M).tocoo()
    diag = C.row == C.col
    df = np.zeros(n_items)
    df[C.row[diag]] = C.data[diag]

    off = ~diag
    r, c, v = C.row[off], C.col[off], C.data[off]
    pmi = np.log(v * n_users / (df[r] * df[c] + 1e-12) + 1e-12)

    table = np.zeros((n_items, n_items), dtype=np.float16)
    table[r, c] = np.maximum(pmi, 0).astype(np.float16)

    np.save(OUT_DIR / "pmi_table.npy", table)
    print(
        f"[build_pmi] {n_items} items, train cutoff={cutoff} -> pmi_table.npy "
        f"({table.nbytes / 1e6:.1f} MB fp16, {100 * (table > 0).mean():.1f}% nnz, "
        f"median PMI={np.median(pmi):+.3f})"
    )


if __name__ == "__main__":
    build_pmi()
