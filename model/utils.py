"""Hàm dùng chung cho item_tower.py / user_tower.py / two_tower_model.py / preprocess.py."""

import torch


def embedding_dim_rule_of_thumb(n_categories):
    """Công thức NVTabular/fast.ai đã chốt trong Readme (mục "Ghi chú kỹ thuật từ NVIDIA
    Merlin"): dim = min(max(16, round(1.6 * n_cat^0.56)), 512)."""
    return min(max(16, round(1.6 * n_categories ** 0.56)), 512)


def pad_and_mask(seq_list, max_len, pad_value=0.0):
    """Pad 1 list các sequence (list[float] hoặc list[list[float]]) về max_len, trả về
    (padded_tensor, mask). mask[i, j] = True nếu vị trí j là dữ liệu thật (không phải PAD).

    Cắt phần ĐẦU (cũ nhất) khi sequence dài hơn max_len — giữ lại các tương tác gần nhất,
    đúng quyết định "recency quan trọng hơn lịch sử xa" đã chốt ở Readme (mục max_seq_len).
    """
    batch = len(seq_list)
    is_vector = len(seq_list) > 0 and len(seq_list[0]) > 0 and isinstance(seq_list[0][0], (list, tuple))
    if is_vector:
        dim = len(seq_list[0][0])
        padded = torch.full((batch, max_len, dim), pad_value, dtype=torch.float32)
    else:
        padded = torch.full((batch, max_len), pad_value, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)

    for i, seq in enumerate(seq_list):
        seq = seq[-max_len:]  # cắt phần đầu (cũ nhất) nếu dài hơn max_len
        n = len(seq)
        if n == 0:
            continue
        padded[i, :n] = torch.tensor(seq, dtype=torch.float32)
        mask[i, :n] = True

    return padded, mask
