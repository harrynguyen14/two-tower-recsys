"""Hàm dùng chung cho item_tower.py / user_tower.py / two_tower_model.py / preprocess.py."""

import torch


def embedding_dim_rule_of_thumb(n_categories):
    """Công thức NVTabular/fast.ai đã chốt trong Readme (mục "Ghi chú kỹ thuật từ NVIDIA
    Merlin"): dim = min(max(16, round(1.6 * n_cat^0.56)), 512)."""
    return min(max(16, round(1.6 * n_categories ** 0.56)), 512)


def pad_and_mask(seq_list, max_len, pad_value=0.0):
    """Pad 1 list các sequence về max_len, trả về (padded_tensor, mask). mask[i, j] = True
    nếu vị trí j là dữ liệu thật (không phải PAD).

    Mỗi sequence là list[float] (ratings) hoặc list[Tensor 1D] (item vectors, đã stack sẵn
    dạng tensor ở __getitem__ — KHÔNG dùng list[list[float]]/.tolist() nữa, tránh round-trip
    tensor->list->tensor vô ích trên CPU khiến GPU rảnh trong lúc train, xem ColdStartDataset).

    Cắt phần ĐẦU (cũ nhất) khi sequence dài hơn max_len — giữ lại các tương tác gần nhất,
    đúng quyết định "recency quan trọng hơn lịch sử xa" đã chốt ở Readme (mục max_seq_len).
    """
    batch = len(seq_list)
    # Không chỉ nhìn seq_list[0]: sample đầu batch có thể rỗng (user chưa có lịch sử trước
    # cutoff — bình thường với cold-start) trong khi sample khác có vector thật, nên phải
    # quét tìm sample KHÔNG rỗng đầu tiên mới xác định đúng is_vector (bug thật đã gặp:
    # batch[0] rỗng -> is_vector=False -> shape mismatch khi gặp sample có Tensor(dim) sau đó).
    first_nonempty = next((s for s in seq_list if len(s) > 0), None)
    is_vector = first_nonempty is not None and torch.is_tensor(first_nonempty[0])
    if is_vector:
        dim = first_nonempty[0].shape[0]
        padded = torch.full((batch, max_len, dim), pad_value, dtype=torch.float32)
    else:
        padded = torch.full((batch, max_len), pad_value, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)

    for i, seq in enumerate(seq_list):
        seq = seq[-max_len:]  # cắt phần đầu (cũ nhất) nếu dài hơn max_len
        n = len(seq)
        if n == 0:
            continue
        padded[i, :n] = torch.stack(list(seq)) if is_vector else torch.tensor(seq, dtype=torch.float32)
        mask[i, :n] = True

    return padded, mask
