"""Hàm dùng chung cho item_tower.py / user_tower.py / two_tower_model.py / preprocess.py."""

import torch


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
