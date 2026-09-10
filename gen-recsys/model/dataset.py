"""Dataset đọc trực tiếp output của preprocessing pipeline (gen-recsys/preprocess_data) —
build sequence ON-THE-FLY lúc train, KHÔNG lưu sequence đầy đủ sẵn (xem
build_sequences.py, thiết kế chống nhân bản dữ liệu ~200 lần đã sửa 2026-09-09).

1 sample = (window K token lịch sử TRƯỚC vị trí dự đoán, label tại vị trí đó, mat_u/mat_i/
is_user_cold/is_item_cold đã tính sẵn ở Pass 5). Toàn bộ dùng memmap (np.load mmap_mode="r")
— không load hết vào RAM, phù hợp 322M+ dòng.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 200  # xem schema.py
NUM_STAT_FEATURES = 8

# Field categorical trong item_static.npy — khớp CATEGORICAL_FIELDS ở item_embedding.py
ITEM_CATEGORICAL_FIELDS = ["category_id", "video_type_id", "music_type_id", "cat_l1_id", "cat_l2_id", "cat_l3_id", "cat_l4_id"]

# 11 field action_vector — khớp ACTION_VECTOR_FIELDS ở schema.py, thứ tự CỐ ĐỊNH
ACTION_VECTOR_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_ratio", "profile_stay_time_norm", "comment_stay_time_norm", "is_profile_enter",
]


class GenRecsysDataset(Dataset):
    """split = "train" | "val" | "test" — đọc {split}.npy (Pass 5) làm danh sách sample,
    tra cứu history_meta/history_action_vectors (Pass 2) + item_static (Pass 3) khi cần.
    """

    def __init__(self, output_dir: str | Path, split: str, max_seq_len: int = MAX_SEQ_LEN):
        self.output_dir = Path(output_dir)
        self.max_seq_len = max_seq_len

        self.split_data = np.load(self.output_dir / f"{split}.npy")  # index, mat_u, mat_i, is_*_cold
        self.sample_user_idx = np.load(self.output_dir / "sample_user_idx.npy", mmap_mode="r")
        self.sample_position = np.load(self.output_dir / "sample_position.npy", mmap_mode="r")
        self.user_offsets = np.load(self.output_dir / "user_offsets.npy")  # nhỏ, load hết vào RAM

        self.history_meta = np.load(self.output_dir / "history_meta.npy", mmap_mode="r")  # video_id, t
        self.history_action = np.load(self.output_dir / "history_action_vectors.npy", mmap_mode="r")  # (M, 11)

        self.item_static = np.load(self.output_dir / "item_static.npy", mmap_mode="r")  # video_id == index

    def __len__(self) -> int:
        return len(self.split_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.split_data[idx]
        sample_idx = int(row["index"])  # vị trí trong sample_user_idx/sample_position (mảng đầy đủ)

        user_idx = int(self.sample_user_idx[sample_idx])
        position = int(self.sample_position[sample_idx])
        user_start = int(self.user_offsets[user_idx])

        window_start = max(user_start, position - self.max_seq_len)
        window_len = position - window_start  # số token lịch sử thật (chưa padding)

        # video_id lịch sử == index trực tiếp vào item_static (đã xác nhận identity mapping)
        hist_video_ids = self.history_meta["video_id"][window_start:position].astype(np.int64)
        hist_action = np.array(self.history_action[window_start:position], dtype=np.float32)  # (window_len, 11), copy khỏi memmap

        hist_timestamps = self.history_meta["t"][window_start:position].astype(np.int64)

        pad_len = self.max_seq_len - window_len
        if pad_len > 0:
            hist_video_ids = np.pad(hist_video_ids, (pad_len, 0), constant_values=0)
            hist_action = np.pad(hist_action, ((pad_len, 0), (0, 0)), constant_values=0.0)
            # timestamp padding = timestamp token thật đầu tiên (an toàn cho lookup_n_at_t_batch:
            # N tại thời điểm padding không được dùng vì key_padding_mask loại các vị trí này
            # khỏi attention, giá trị chỉ cần hợp lệ về mặt số học, không cần đúng ngữ nghĩa).
            fill_ts = hist_timestamps[0] if window_len > 0 else 0
            hist_timestamps = np.pad(hist_timestamps, (pad_len, 0), constant_values=fill_ts)

        key_padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        if pad_len > 0:
            key_padding_mask[:pad_len] = True  # padding ở ĐẦU chuỗi (token gần nhất luôn ở cuối)

        label_video_id = int(self.history_meta["video_id"][position])
        label_timestamp = int(self.history_meta["t"][position])
        # action_vector THẬT tại vị trí label — dùng cho RankingLoss (KHÁC action_vector
        # của các token TRONG history, vốn là hành vi user đã làm TRƯỚC đó).
        label_action = np.array(self.history_action[position], dtype=np.float32)

        return {
            "hist_video_ids": torch.from_numpy(hist_video_ids),  # (K,) int64 — index vào item_static
            "hist_action": torch.from_numpy(hist_action),  # (K, 11) float32
            "hist_timestamps": torch.from_numpy(hist_timestamps),  # (K,) int64 — dùng cho lookup_n_at_t_batch
            "key_padding_mask": torch.from_numpy(key_padding_mask),  # (K,) bool
            "label_video_id": torch.tensor(label_video_id, dtype=torch.int64),
            "label_timestamp": torch.tensor(label_timestamp, dtype=torch.int64),
            "label_action": torch.from_numpy(label_action),  # (11,) float32
            "mat_u": torch.tensor(float(row["mat_u"]), dtype=torch.float32),
            "mat_i": torch.tensor(float(row["mat_i"]), dtype=torch.float32),
            "is_user_cold": torch.tensor(bool(row["is_user_cold"])),
            "is_item_cold": torch.tensor(bool(row["is_item_cold"])),
        }

    def get_category_ids(self, video_ids: torch.Tensor) -> torch.Tensor:
        """category_id (tag cũ, đã factorize) của mỗi video_id — dùng làm entity_id khi
        tra category_N qua lookup_n_at_t_batch (Pass 1) để tính conf_content."""
        idx = video_ids.numpy()
        return torch.from_numpy(self.item_static["category_id"][idx].astype(np.int64))

    def get_item_features(self, video_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tra item_static cho 1 tensor video_id bất kỳ (dùng cho cả token lịch sử lẫn
        candidate set) — video_id == index trực tiếp, KHÔNG cần searchsorted."""
        idx = video_ids.numpy()
        rows = self.item_static[idx]  # fancy-index trên memmap -> numpy array thường
        return {
            "category_ids": {
                field: torch.from_numpy(rows[field].astype(np.int64)) for field in ITEM_CATEGORICAL_FIELDS
            },
            "author_idx": torch.from_numpy(rows["author_idx"].astype(np.int64)),
            "music_idx": torch.from_numpy(rows["music_idx"].astype(np.int64)),
            "stat_features": torch.from_numpy(np.asarray(rows["features"], dtype=np.float32)),
        }
