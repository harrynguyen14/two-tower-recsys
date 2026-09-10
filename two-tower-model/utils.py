"""Hàm dùng chung cho item_tower.py / user_tower.py / two_tower_model.py / preprocess.py."""

import numpy as np
import torch

# Static features theo ĐÚNG thứ tự đưa vào UserTower (xem ColdStartDataset.__getitem__):
#   0 user_mean_rating, 1 user_std_rating, 2 user_total_reviews, 3 user_avg_page_count
#
# LOG1P_FEATURES: index các feature lệch phải NẶNG cần log1p trước khi standardize. Đo trên
# dữ liệu thật (Kindle_Store, 2M review đầu):
#   user_total_reviews  p50=2    p99=98   max=2208   <- max gấp ~1100x p50
#   user_mean_rating    p50=4.67 p99=5.00 max=5.00   <- vốn bounded 1-5, KHÔNG log1p
#   user_std_rating     p50=0.00 p99=1.89 max=2.00   <- vốn bounded, KHÔNG log1p
# Chỉ standardize (trừ mean/chia std) mà không log1p trước thì outlier 2208 vẫn kéo lệch
# toàn bộ phân phối: p50 và p99 bị nén vào một dải rất hẹp quanh 0.
LOG1P_FEATURES = (2, 3)

N_STATIC_FEATURES = 4


def fit_static_scaler(static_vectors):
    """Tính (mean, std) cho static features — CHỈ gọi trên tập TRAIN.

    static_vectors: array-like [n_samples, N_STATIC_FEATURES] giá trị THÔ.

    Trả về dict {"mean": list, "std": list, "log1p": list} — lưu được vào checkpoint bằng
    torch.save (thuần Python list, không phụ thuộc numpy version).

    BUG THẬT ĐÃ SỬA (Lỗi 3): docstring preprocess.py ghi static_features "đã chuẩn hoá"
    nhưng KHÔNG có code nào chuẩn hoá — giá trị thô đi thẳng vào fusion_mlp. user_total_reviews
    (~O(100)) và user_avg_page_count (~O(400)) áp đảo seq_vec (128-d, ~N(0,1)) và cat_vec
    (16-d), tức 2 chiều nhấn chìm 144 chiều còn lại. user_mean_rating — feature tín hiệu
    mạnh nhất (autocorrelation r=0.341 theo Readme) — bị số trang sách dìm.

    Fit CHỈ trên train rồi transform cho mọi split: đúng nguyên tắc fit/transform của
    NVTabular, tránh leakage thống kê từ val/test vào quá trình chuẩn hoá."""
    arr = np.asarray(static_vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_STATIC_FEATURES:
        raise ValueError(f"static_vectors phải là [n, {N_STATIC_FEATURES}], nhận {arr.shape}")

    arr = arr.copy()
    for j in LOG1P_FEATURES:
        # clip(min=0) trước log1p: các feature này về ngữ nghĩa không thể âm (count/mean của
        # count), nhưng phòng dữ liệu lỗi có giá trị âm -> log1p(-1) = -inf.
        arr[:, j] = np.log1p(np.clip(arr[:, j], 0, None))

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # std=0 (feature hằng số trên train, vd std_rating khi mọi user chỉ có 1 review) -> chia
    # cho 0 ra inf/nan. Thay bằng 1.0: feature đó thành hằng số 0 sau transform, vô hại.
    std = np.where(std < 1e-8, 1.0, std)

    return {"mean": mean.tolist(), "std": std.tolist(), "log1p": list(LOG1P_FEATURES)}


def transform_static(static_vec, scaler):
    """Áp scaler (từ fit_static_scaler) lên 1 vector static thô -> list[float] đã chuẩn hoá.

    scaler=None -> trả về nguyên trạng (đường code chưa fit scaler, vd test nhanh)."""
    if scaler is None:
        return [float(v) for v in static_vec]
    mean, std, log1p_idx = scaler["mean"], scaler["std"], set(scaler["log1p"])
    out = []
    for j, v in enumerate(static_vec):
        v = float(v)
        if j in log1p_idx:
            v = float(np.log1p(max(v, 0.0)))
        out.append((v - mean[j]) / std[j])
    return out


def embedding_dim_rule_of_thumb(n_categories):
    """Công thức NVTabular/fast.ai đã chốt trong Readme (mục "Ghi chú kỹ thuật từ NVIDIA
    Merlin"): dim = min(max(16, round(1.6 * n_cat^0.56)), 512)."""
    return min(max(16, round(1.6 * n_categories ** 0.56)), 512)


def pad_and_mask(seq_list, max_len, pad_value=0.0, vector_dim=None, device=None):
    """Pad 1 list các sequence về max_len, trả về (padded_tensor, mask). mask[i, j] = True
    nếu vị trí j là dữ liệu thật (không phải PAD).

    Mỗi sequence là list[float] (ratings) hoặc list[Tensor 1D] (item vectors, đã stack sẵn
    dạng tensor ở __getitem__ — KHÔNG dùng list[list[float]]/.tolist() nữa, tránh round-trip
    tensor->list->tensor vô ích trên CPU khiến GPU rảnh trong lúc train, xem ColdStartDataset).

    Cắt phần ĐẦU (cũ nhất) khi sequence dài hơn max_len — giữ lại các tương tác gần nhất,
    đúng quyết định "recency quan trọng hơn lịch sử xa" đã chốt ở Readme (mục max_seq_len).

    vector_dim: bắt buộc truyền khi seq_list chứa Tensor (item vectors) — trước đây suy luận
    dim từ sample KHÔNG rỗng đầu tiên trong batch (first_nonempty), nhưng khi CẢ BATCH đều
    rỗng (mọi user trong batch N=0 lịch sử — có thể xảy ra thật, đặc biệt ở val_cold) thì
    first_nonempty=None -> tạo nhầm tensor 2D [batch, max_len] thay vì 3D
    [batch, max_len, dim] -> crash "not enough values to unpack" ở SequenceEncoder.forward
    (bug thật đã gặp). Truyền vector_dim tường minh loại bỏ hẳn việc phải đoán shape.

    device: truyền khi seq_list chứa Tensor GPU (precomputed_item_vectors giờ giữ trên GPU,
    xem item_tower.ItemEmbeddingStore) — padded/mask phải cùng device với các tensor được
    stack vào, nếu không torch.stack sẽ lỗi mismatch device."""
    batch = len(seq_list)
    is_vector = vector_dim is not None
    if is_vector:
        padded = torch.full((batch, max_len, vector_dim), pad_value, dtype=torch.float32, device=device)
    else:
        padded = torch.full((batch, max_len), pad_value, dtype=torch.float32, device=device)
    mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)

    for i, seq in enumerate(seq_list):
        seq = seq[-max_len:]  # cắt phần đầu (cũ nhất) nếu dài hơn max_len
        n = len(seq)
        if n == 0:
            continue
        padded[i, :n] = torch.stack(list(seq)) if is_vector else torch.tensor(seq, dtype=torch.float32, device=device)
        mask[i, :n] = True

    return padded, mask
