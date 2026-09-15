"""Dataset đọc trực tiếp output của preprocessing pipeline (gen-recsys/preprocess_data) —
build sequence ON-THE-FLY lúc train, KHÔNG lưu sequence đầy đủ sẵn (xem
build_sequences.py, thiết kế chống nhân bản dữ liệu ~200 lần đã sửa 2026-09-09).

1 sample = (window K token lịch sử TRƯỚC vị trí dự đoán, label tại vị trí đó,
is_user_cold/is_item_cold đã tính sẵn ở Pass 5). Toàn bộ dùng memmap (np.load mmap_mode="r")
— không load hết vào RAM, phù hợp dataset lớn.

[SỬA 2026-09-13] Chuyển sang KuaiRand-Pure (xem schema.py, result.md "CHECKLIST CUỐI
CÙNG"): bỏ 4 field cat_l1-l4_id (category 4 cấp — Pure không có file nguồn) khỏi
ITEM_CATEGORICAL_FIELDS, bỏ stat_features khỏi get_item_features (review.md L9 — leak
thời gian, đã sửa tận gốc ở build_item_static.py: không còn ghi field "features" vào
item_static.npy nữa).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 256  # xem schema.py — đo phân phối thật trên Pure, 256 phủ p99 (2026-09-14)

# Field categorical trong item_static.npy — khớp CATEGORICAL_FIELDS ở item_embedding.py
ITEM_CATEGORICAL_FIELDS = ["category_id", "video_type_id", "music_type_id"]

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
        self.user_ids_sorted = np.load(self.output_dir / "user_ids_sorted.npy")  # nhỏ, index -> user_id thật

        self.history_meta = np.load(self.output_dir / "history_meta.npy", mmap_mode="r")  # video_id, t
        self.history_action = np.load(self.output_dir / "history_action_vectors.npy", mmap_mode="r")  # (M, 11)

        self.item_static = np.load(self.output_dir / "item_static.npy", mmap_mode="r")  # video_id == index
        # nhánh caption OPTIONAL (Pass 3.5, xem encode_captions_kaggle.py + merge_caption_shards.py)
        self.caption_embeddings = np.load(self.output_dir / "caption_embeddings.npy", mmap_mode="r")
        self.caption_has_caption = np.load(self.output_dir / "caption_has_caption.npy", mmap_mode="r")

        # [THÊM 2026-09-11] user_static (Pass 4, xem build_user_static.py + user_embedding.py)
        # — KHÔNG dùng identity mapping user_id==index (khác item_static): user_ids_sorted.npy
        # (build_sequences.py) sort theo THỨ TỰ BUCKET GHI (user_id % 100), KHÔNG monotonic
        # theo giá trị user_id (đã xác nhận trực tiếp: user_ids_sorted[:10] =
        # [0,100,200,...,900], rõ ràng là thứ tự bucket, không phải giá trị tăng dần), trong
        # khi user_static.npy .sort("user_id") monotonic tăng dần — 2 thứ tự KHÁC NHAU, phải
        # tra qua searchsorted trên user_static["user_id"] (đã sort), không được giả định
        # trùng index như item_static.
        self.user_static = np.load(self.output_dir / "user_static.npy", mmap_mode="r")
        self.onehot_num_categories = np.load(self.output_dir / "onehot_num_categories.npy").tolist()

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

        # [THÊM 2026-09-14] Loss tự hồi quy TOÀN CHUỖI (retrieval.py forward_sequence) cần
        # biết vị trí nào dự đoán được: tại Φ_t dự đoán item t+1, cặp (t, t+1) chỉ hợp lệ
        # khi CẢ HAI là token thật. hist_valid_mask[t]=True nghĩa là t là token thật; train.py
        # dùng hist_valid_mask[:, :-1] & hist_valid_mask[:, 1:] cho các cặp trong window, và
        # hist_valid_mask[:, -1] cho cặp (token cuối -> label).
        hist_valid_mask = ~key_padding_mask  # (K,) True = token thật

        label_video_id = int(self.history_meta["video_id"][position])
        label_timestamp = int(self.history_meta["t"][position])
        # action_vector THẬT tại vị trí label — dùng cho RankingLoss (KHÁC action_vector
        # của các token TRONG history, vốn là hành vi user đã làm TRƯỚC đó).
        label_action = np.array(self.history_action[position], dtype=np.float32)

        # [SỬA 2026-09-11] n_u THÔ (không phải mat_u tính sẵn) — n_u = số token lịch sử TRƯỚC
        # vị trí dự đoán, TOÀN BỘ từ đầu user (KHÔNG cắt bởi max_seq_len như window_len), khớp
        # đúng cách build_interactions.py tính. Dùng để train.py tính mat_u = thresholds.mat_u(n_u)
        # RUNTIME (có gradient chảy qua tau_u) thay vì đọc row["mat_u"] tính sẵn bằng TAU_U_INIT
        # hằng số ở Pass 5 (build_interactions.py) — bug đã xác nhận: dùng giá trị tính sẵn khiến
        # tau_u KHÔNG BAO GIỜ nhận gradient dù được đăng ký vào optimizer, làm cơ chế giám sát
        # threshold-collapse cho tau_u vô nghĩa (giá trị đứng yên ở khởi tạo, không phải hội tụ).
        n_u = position - user_start

        # [THÊM 2026-09-14] n_u THEO TỪNG VỊ TRÍ trong window (KHÔNG chỉ 1 scalar tại điểm
        # dự đoán). Token thứ t của window nằm ở vị trí tuyệt đối (window_start + t) trong
        # lịch sử user, nên số lượt TRƯỚC nó = window_start + t - user_start.
        #
        # Vì sao cần, không phải tiện tay thêm: attention sắp dùng cặp (u_i, m_j) —
        # (β + γ·log u_i)·log m_j, xem confidence_attention.py. Nếu u_i HẰNG theo i thì
        # log(u_i) gộp thẳng vào β và công thức thoái hóa về đúng λ·log(mat_j) đã bỏ
        # 2026-09-13 (gradient đo được ~1e-17). Biến thiên của u_i TRONG chuỗi mới là thứ
        # tạo tín hiệu: mọi user đều cold ở token đầu của chính mình và warm dần về cuối —
        # đây đồng thời là trục ngắn/dài hạn.
        #
        # Tính bằng SỐ HỌC, không tra user_N CSR: n_u vốn đã là position - user_start (số
        # token lịch sử), nên per-position chỉ là cùng phép trừ dịch theo t. Tra CSR ở đây
        # vừa thừa vừa chậm (2 searchsorted × K/step) và còn LỆCH ngữ nghĩa — CSR đếm trên
        # log gốc, còn cái model thật sự thấy là lịch sử trong dataset.
        hist_n_u = (window_start - user_start) + np.arange(window_len, dtype=np.float32)
        if pad_len > 0:
            # Vị trí padding: 0 (user chưa có lượt nào) — an toàn vì key_padding_mask loại
            # các vị trí này khỏi attention, giá trị chỉ cần hợp lệ về số học.
            hist_n_u = np.pad(hist_n_u, (pad_len, 0), constant_values=0.0)

        user_id = int(self.user_ids_sorted[user_idx])  # id thật, dùng để tra user_static (get_user_features)

        return {
            "hist_video_ids": torch.from_numpy(hist_video_ids),  # (K,) int64 — index vào item_static
            "hist_action": torch.from_numpy(hist_action),  # (K, 11) float32
            "hist_timestamps": torch.from_numpy(hist_timestamps),  # (K,) int64 — dùng cho lookup_n_at_t_batch
            "key_padding_mask": torch.from_numpy(key_padding_mask),  # (K,) bool
            "hist_valid_mask": torch.from_numpy(hist_valid_mask),  # (K,) bool — True = token thật, xem loss toàn chuỗi
            "label_video_id": torch.tensor(label_video_id, dtype=torch.int64),
            "label_timestamp": torch.tensor(label_timestamp, dtype=torch.int64),
            "label_action": torch.from_numpy(label_action),  # (11,) float32
            "n_u": torch.tensor(float(n_u), dtype=torch.float32),  # (,) — n_u tại điểm dự đoán
            "hist_n_u": torch.from_numpy(hist_n_u),  # (K,) float32 — n_u TẠI TỪNG vị trí, xem trên
            "user_id": torch.tensor(user_id, dtype=torch.int64),  # dùng để tra user_static (get_user_features)
            "is_user_cold": torch.tensor(bool(row["is_user_cold"])),  # STRICT holdout (zero-shot)
            "is_item_cold": torch.tensor(bool(row["is_item_cold"])),
            # [THÊM 2026-09-15] few-shot (N_u < LOW_HISTORY_N) — SONG SONG, không thay thế.
            # Strict holdout cho ô cold/cold chỉ 0/5/13 sample (train/val/test); cờ này cho
            # 31,538/81/65. Xem build_interactions.py + schema.py LOW_HISTORY_N.
            "is_user_lowhistory": torch.tensor(bool(row["is_user_lowhistory"])),
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
        caption_embedding = np.array(self.caption_embeddings[idx], dtype=np.float32)  # (N, 384)
        caption_mask = np.array(self.caption_has_caption[idx], dtype=np.float32)  # (N,) 1.0/0.0
        return {
            "category_ids": {
                field: torch.from_numpy(rows[field].astype(np.int64)) for field in ITEM_CATEGORICAL_FIELDS
            },
            "author_idx": torch.from_numpy(rows["author_idx"].astype(np.int64)),
            "music_idx": torch.from_numpy(rows["music_idx"].astype(np.int64)),
            "caption_embedding": torch.from_numpy(caption_embedding),
            "caption_mask": torch.from_numpy(caption_mask),
        }

    def get_user_features(self, user_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tra user_static cho 1 tensor user_id — PHẢI searchsorted (KHÁC get_item_features):
        user_static.npy sort theo GIÁ TRỊ user_id tăng dần, nhưng user_id thật không liên tục
        [0,n) như video_id (identity mapping không áp dụng được ở đây, xem __init__).

        [SỬA 2026-09-11] np.searchsorted KHÔNG báo lỗi nếu user_id không tồn tại trong
        user_static — nó trả về vị trí CHÈN gần nhất (bug đã xác nhận qua audit: nếu
        user_static.npy (từ user_features_27k.csv) thiếu 1 số user_id có mặt trong log
        tương tác — 2 nguồn dữ liệu ĐỘC LẬP, không đảm bảo khớp 100% — searchsorted âm thầm
        trả về feature của MỘT USER KHÁC (silent-wrong) hoặc IndexError nếu user_id lớn hơn
        mọi giá trị trong bảng (crash mơ hồ, khó truy vết). Assert khớp chính xác để lỗi bung
        ra RÕ RÀNG ngay tại đây thay vì lan xuống model như dữ liệu sai lặng lẽ."""
        ids = self.user_static["user_id"]  # đã sort tăng dần
        query = user_ids.numpy()
        idx = np.clip(np.searchsorted(ids, query), 0, len(ids) - 1)
        mismatched = ids[idx] != query
        if mismatched.any():
            bad_ids = query[mismatched][:5].tolist()
            raise KeyError(
                f"user_id không tồn tại trong user_static.npy (ví dụ: {bad_ids}) — "
                "user_features_27k.csv có thể thiếu user này so với log tương tác."
            )
        rows = self.user_static[idx]  # fancy-index trên memmap -> numpy array thường
        return {
            "onehot": torch.from_numpy(rows["onehot"].astype(np.int64)),  # (N, 18)
            "register_days": torch.from_numpy(rows["register_days"].astype(np.float32)),  # (N,)
        }
