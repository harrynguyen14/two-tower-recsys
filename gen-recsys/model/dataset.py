"""Dataset đọc trực tiếp output của preprocessing pipeline (gen-recsys/preprocess_data) —"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 256

# category_id (tag da factorize CA CHUOI) KHONG con vao content token — xem tag_ids.
# Giu lai trong item_static vi category_N/N_category van tra cuu theo no.
ITEM_CATEGORICAL_FIELDS = ["video_type_id", "music_type_id"]

ACTION_VECTOR_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_ratio", "profile_stay_time_norm", "comment_stay_time_norm", "is_profile_enter",
]


class GenRecsysDataset(Dataset):
    """split = "train" | "val" | "test" — đọc {split}.npy (Pass 5) làm danh sách sample,"""

    def __init__(self, output_dir: str | Path, split: str, max_seq_len: int = MAX_SEQ_LEN):
        self.output_dir = Path(output_dir)
        self.max_seq_len = max_seq_len

        self.split_data = np.load(self.output_dir / f"{split}.npy")
        self.sample_user_idx = np.load(self.output_dir / "sample_user_idx.npy", mmap_mode="r")
        self.sample_position = np.load(self.output_dir / "sample_position.npy", mmap_mode="r")
        self.user_offsets = np.load(self.output_dir / "user_offsets.npy")
        self.user_ids_sorted = np.load(self.output_dir / "user_ids_sorted.npy")

        self.history_meta = np.load(self.output_dir / "history_meta.npy", mmap_mode="r")
        self.history_action = np.load(self.output_dir / "history_action_vectors.npy", mmap_mode="r")

        self.item_static = np.load(self.output_dir / "item_static.npy", mmap_mode="r")
        self.caption_embeddings = np.load(self.output_dir / "caption_embeddings.npy", mmap_mode="r")
        self.caption_has_caption = np.load(self.output_dir / "caption_has_caption.npy", mmap_mode="r")

        self.user_static = np.load(self.output_dir / "user_static.npy", mmap_mode="r")
        self.onehot_num_categories = np.load(self.output_dir / "onehot_num_categories.npy").tolist()

    def __len__(self) -> int:
        return len(self.split_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.split_data[idx]
        sample_idx = int(row["index"])

        user_idx = int(self.sample_user_idx[sample_idx])
        position = int(self.sample_position[sample_idx])
        user_start = int(self.user_offsets[user_idx])

        window_start = max(user_start, position - self.max_seq_len)
        window_len = position - window_start

        hist_video_ids = self.history_meta["video_id"][window_start:position].astype(np.int64)
        hist_action = np.array(self.history_action[window_start:position], dtype=np.float32)

        hist_timestamps = self.history_meta["t"][window_start:position].astype(np.int64)

        pad_len = self.max_seq_len - window_len
        if pad_len > 0:
            hist_video_ids = np.pad(hist_video_ids, (pad_len, 0), constant_values=0)
            hist_action = np.pad(hist_action, ((pad_len, 0), (0, 0)), constant_values=0.0)
            fill_ts = hist_timestamps[0] if window_len > 0 else 0
            hist_timestamps = np.pad(hist_timestamps, (pad_len, 0), constant_values=fill_ts)

        key_padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        if pad_len > 0:
            key_padding_mask[:pad_len] = True

        hist_valid_mask = ~key_padding_mask

        label_video_id = int(self.history_meta["video_id"][position])
        label_timestamp = int(self.history_meta["t"][position])
        label_action = np.array(self.history_action[position], dtype=np.float32)

        n_u = position - user_start

        hist_n_u = (window_start - user_start) + np.arange(window_len, dtype=np.float32)
        if pad_len > 0:
            hist_n_u = np.pad(hist_n_u, (pad_len, 0), constant_values=0.0)

        user_id = int(self.user_ids_sorted[user_idx])

        return {
            "hist_video_ids": torch.from_numpy(hist_video_ids),
            "hist_action": torch.from_numpy(hist_action),
            "hist_timestamps": torch.from_numpy(hist_timestamps),
            "key_padding_mask": torch.from_numpy(key_padding_mask),
            "hist_valid_mask": torch.from_numpy(hist_valid_mask),
            "label_video_id": torch.tensor(label_video_id, dtype=torch.int64),
            "label_timestamp": torch.tensor(label_timestamp, dtype=torch.int64),
            "label_action": torch.from_numpy(label_action),
            "n_u": torch.tensor(float(n_u), dtype=torch.float32),
            "hist_n_u": torch.from_numpy(hist_n_u),
            "user_id": torch.tensor(user_id, dtype=torch.int64),
            "is_user_cold": torch.tensor(bool(row["is_user_cold"])),
            "is_item_cold": torch.tensor(bool(row["is_item_cold"])),
            "is_user_lowhistory": torch.tensor(bool(row["is_user_lowhistory"])),
        }

    def get_category_ids(self, video_ids: torch.Tensor) -> torch.Tensor:
        """category_id (tag cũ, đã factorize) của mỗi video_id — dùng làm entity_id khi"""
        idx = video_ids.numpy()
        return torch.from_numpy(self.item_static["category_id"][idx].astype(np.int64))

    def get_item_features(self, video_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tra item_static cho 1 tensor video_id bất kỳ (dùng cho cả token lịch sử lẫn"""
        idx = video_ids.numpy()
        rows = self.item_static[idx]
        caption_embedding = np.array(self.caption_embeddings[idx], dtype=np.float32)
        caption_mask = np.array(self.caption_has_caption[idx], dtype=np.float32)
        return {
            "category_ids": {
                field: torch.from_numpy(rows[field].astype(np.int64)) for field in ITEM_CATEGORICAL_FIELDS
            },
            "author_idx": torch.from_numpy(rows["author_idx"].astype(np.int64)),
            "music_idx": torch.from_numpy(rows["music_idx"].astype(np.int64)),
            "caption_embedding": torch.from_numpy(caption_embedding),
            "caption_mask": torch.from_numpy(caption_mask),
            "tag_ids": torch.from_numpy(rows["tag_ids"].astype(np.int64)),
        }

    def get_user_features(self, user_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tra user_static cho 1 tensor user_id — PHẢI searchsorted (KHÁC get_item_features):"""
        ids = self.user_static["user_id"]
        query = user_ids.numpy()
        idx = np.clip(np.searchsorted(ids, query), 0, len(ids) - 1)
        mismatched = ids[idx] != query
        if mismatched.any():
            bad_ids = query[mismatched][:5].tolist()
            raise KeyError(
                f"user_id không tồn tại trong user_static.npy (ví dụ: {bad_ids}) — "
                "user_features_27k.csv có thể thiếu user này so với log tương tác."
            )
        rows = self.user_static[idx]
        return {
            "onehot": torch.from_numpy(rows["onehot"].astype(np.int64)),
            "register_days": torch.from_numpy(rows["register_days"].astype(np.float32)),
        }
