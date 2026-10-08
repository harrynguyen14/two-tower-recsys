"""Dataset đọc trực tiếp output của preprocessing pipeline (gen-recsys/preprocess_data) —"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 256  # mac dinh cho Pure; 27K phai dung --max-seq-len 1024 (xem duoi)

# category_id (tag da factorize CA CHUOI) KHONG con vao content token — xem tag_ids.
# Giu lai trong item_static vi category_N/N_category van tra cuu theo no.
ITEM_CATEGORICAL_FIELDS = ["video_type_id", "music_type_id"]

# Nhanh 3 (formula.md §1) va age (§2) — CHI co khi build_item_static ghi duoc chung
# (27K co ca hai; Pure khong co file categories va upload_dt chi 3 gia tri). Dataset tu
# phat hien qua ten field trong item_static thay vi co cau hinh rieng.
CATEGORY_LEVEL_FIELDS = ["cat_l1_id", "cat_l2_id", "cat_l3_id", "cat_l4_id"]
MS_PER_DAY = 86_400_000

ACTION_VECTOR_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_ratio", "profile_stay_time_norm", "comment_stay_time_norm", "is_profile_enter",
]


class GenRecsysDataset(Dataset):
    """split = "train" | "val" | "test" — đọc {split}.npy (Pass 5) làm danh sách sample,"""

    def __init__(self, output_dir: str | Path, split: str, max_seq_len: int = MAX_SEQ_LEN,
                 random_window: bool = False, seed: int = 0):
        self.output_dir = Path(output_dir)
        self.max_seq_len = max_seq_len
        # CUA SO TRUOT (chi train): xem docstring _window_start.
        self.random_window = random_window
        self.seed = seed
        self.epoch = 0

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

        names = self.item_static.dtype.names
        self.has_category_levels = all(f in names for f in CATEGORY_LEVEL_FIELDS)
        self.has_age = "upload_day" in names

    def __len__(self) -> int:
        return len(self.split_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.split_data[idx]
        sample_idx = int(row["index"])

        user_idx = int(self.sample_user_idx[sample_idx])
        position = int(self.sample_position[sample_idx])
        user_start = int(self.user_offsets[user_idx])

        window_start = self._window_start(user_start, position, idx)
        # Cat do dai CO DINH = max_seq_len: `_window_start` chi chon CHO, con cua so luon
        # dai dung K. Thieu dong nay thi cua so truot cho window_len > K => pad_len am =>
        # tensor dai hon K va collate vo (bug 2026-10-05, test_random_window bat duoc).
        window_end = min(position, window_start + self.max_seq_len)
        window_len = window_end - window_start

        hist_video_ids = self.history_meta["video_id"][window_start:window_end].astype(np.int64)
        hist_action = np.array(self.history_action[window_start:window_end], dtype=np.float32)

        hist_timestamps = self.history_meta["t"][window_start:window_end].astype(np.int64)

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

        # Label = item NGAY SAU cua so, khong phai item o `position`. Voi cua so co dinh
        # (eval) hai cai la MOT (window_end == position). Voi cua so truot (train) thi
        # `position` cach cuoi cua so median 29 tuong tac — ghep no voi cua so da truot se
        # bien bai toan thanh "du doan item sau 29 luot nua", KHAC han next-item ma eval do.
        label_pos = window_end
        label_video_id = int(self.history_meta["video_id"][label_pos])
        label_timestamp = int(self.history_meta["t"][label_pos])
        label_action = np.array(self.history_action[label_pos], dtype=np.float32)

        n_u = label_pos - user_start

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

    def set_epoch(self, epoch: int) -> None:
        """Doi cua so moi epoch. PHAI goi tu train loop, neu khong moi epoch cho CUNG cua
        so va tac dung "qua nhieu epoch thay het chuoi" mat han."""
        self.epoch = int(epoch)

    def _window_start(self, user_start: int, position: int, idx: int) -> int:
        """Diem bat dau cua so lich su.

        EVAL (random_window=False): LUON la `max_seq_len` item NGAY TRUOC diem du doan —
        dung y nhu luc serving.

        TRAIN (random_window=True): lay cua so o vi tri NGAU NHIEN trong qua khu. Ly do la
        so do tren 27K: median 1,744 tuong tac/user (p99 12,915, max 67,647), nen cua so
        co dinh K=1024 chi phu 34% tuong tac va vung "xa" bi CAT CUNG — dung thu ma
        delta_h(x_q) (confidence_attention, §4.1) can de hoc pham vi nhin. Lay ngau nhien
        thi qua nhieu epoch model thay gan het chuoi, va moi epoch van chi tra O((2K)^2).

        KHONG lay mau thua dan ve qua khu (log-spaced): nhu vay `log_rel_distance` khong
        con la khoang cach THAT giua hai token, va delta_h se hoc tren mot thang bi bop meo.
        """
        latest = max(user_start, position - self.max_seq_len)
        if not self.random_window or latest <= user_start:
            return latest
        # `latest` la diem bat dau MUON NHAT (cua so ke diem du doan). Truot ve QUA KHU:
        # chon trong [user_start, latest] — cua so van dai DUNG max_seq_len, chi doi cho.
        #
        # RNG dung tu (seed, epoch, idx) chu KHONG phai mot `self._rng` dung chung: DataLoader
        # voi num_workers>0 fork dataset nen moi worker se giu mot BAN SAO rng va sinh cung
        # mot chuoi so => cac worker cho cung cua so. Cach nay cho ket qua giong nhau bat ke
        # so worker, va tai lap duoc khi debug.
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        return int(rng.integers(user_start, latest + 1))

    def num_tags(self) -> int:
        """So hang bang E^tag = ma tag LON NHAT + 1, khong phai so tag duy nhat.

        Tag id THUA (27K: 58 tag duy nhat nhung max id 68) nen dem so tag duy nhat se cho
        bang QUA NHO va nn.Embedding nem IndexError. Truoc 2026-10-05 cho nay la hang so
        47 (cua Pure) trong ItemEmbeddingConfig va train.py khong truyen gi — tren 27K se
        vo ngay o forward dau tien.
        """
        return int(self.item_static["tag_ids"].max()) + 1

    def num_category_levels(self) -> dict[str, int] | None:
        """So nhan moi cap — truyen vao ItemEmbeddingConfig. None neu dataset khong co."""
        if not self.has_category_levels:
            return None
        return {f: int(self.item_static[f].max()) + 1 for f in CATEGORY_LEVEL_FIELDS}

    def get_age_days(self, video_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """age_t = so ngay tu luc item ra mat toi luc xem (formula.md §2).

        Tinh runtime chu khong luu san: age phu thuoc timestamp cua TUNG tuong tac, luu
        san se la (n_item x n_timestamp). item thieu upload_dt (upload_day=-1) cho age am,
        action_encoder.age_bucket day ve bin 0.
        """
        upload = self.item_static["upload_day"][video_ids.numpy()].astype(np.int64)
        view_day = timestamps.numpy() // MS_PER_DAY
        return torch.from_numpy((view_day - upload).astype(np.float32))

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
            "category_level_ids": {
                field: torch.from_numpy(rows[field].astype(np.int64))
                for field in CATEGORY_LEVEL_FIELDS
            } if self.has_category_levels else None,
        }

