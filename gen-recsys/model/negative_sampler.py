"""Negative sampler cho sampled-softmax retrieval loss — sample theo TẦN SUẤT xuất hiện
thật trong log (chuẩn word2vec/HSTU negative sampling), KHÔNG uniform (đã CHỐT 2026-09-10
— item phổ biến làm negative "khó" hơn, đúng tinh thần item cold không bị lấn át bởi việc
so sánh không công bằng với item phổ biến).

Tần suất lấy từ item_N_ids/item_N_offsets (Pass 1, build_n_cumulative.py) — số lượt tương
tác TOÀN CỤC mỗi item đã nhận (không phải theo thời điểm, chỉ dùng cho sampling, KHÔNG
dùng để tính mat_i — mat_i vẫn phải tính theo timestamp để tránh leak tương lai, xem
build_n_cumulative.py lookup_n_at_t_batch). 32 item hoàn toàn chưa từng xuất hiện trong
log (N_i=0 tuyệt đối) KHÔNG có trong item_N_ids — gán tần suất tối thiểu (1) để vẫn có
cơ hội được sample làm negative, không bị loại trừ hoàn toàn.

log_Q(i) = log(freq(i) / sum(freq)) — sampled-softmax correction chuẩn (xem retrieval.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class NegativeSampler:
    def __init__(self, output_dir: str | Path, num_items: int):
        output_dir = Path(output_dir)
        item_ids = np.load(output_dir / "item_N_ids.npy")
        item_offsets = np.load(output_dir / "item_N_offsets.npy")
        freq_observed = np.diff(item_offsets).astype(np.float64)

        # Full-length tần suất theo đúng index video_id (0..num_items-1) — item không xuất
        # hiện trong item_N_ids (chưa từng có tương tác) được gán tần suất tối thiểu = 1.
        freq = np.ones(num_items, dtype=np.float64)
        freq[item_ids] = freq_observed

        self.probs = freq / freq.sum()
        self.log_probs = np.log(self.probs)
        self.num_items = num_items

    def sample(
        self,
        batch_size: int,
        num_negatives: int,
        device: torch.device,
        exclude: torch.Tensor | None = None,  # (batch_size,) video_id positive cần loại
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về (negative_video_ids, log_q) — shape (batch_size, num_negatives).

        [SỬA 2026-09-14] Thêm `exclude` — loại FALSE NEGATIVE. Bug cũ: sample theo tần suất
        KHÔNG loại positive, nên positive thường xuyên nằm trong tập negative của chính nó.
        Cross-entropy khi đó phạt model vì xếp positive lên cao — hại trực tiếp metric. Rủi
        ro cao trên Pure: chỉ 7,583 item với 100 negative/sample, sampling theo tần suất còn
        dồn về item phổ biến.

        Resample lặp (tối đa 10 vòng) thay vì 1 lần: lần resample vẫn có thể trúng lại
        positive. 10 vòng đủ để xác suất còn sót không đáng kể với mọi phân phối thực tế."""
        neg_ids = np.random.choice(self.num_items, size=(batch_size, num_negatives), p=self.probs)

        if exclude is not None:
            pos = exclude.detach().cpu().numpy().reshape(-1, 1)  # (B, 1) broadcast theo cột
            for _ in range(10):
                collide = neg_ids == pos
                n_collide = int(collide.sum())
                if n_collide == 0:
                    break
                neg_ids[collide] = np.random.choice(self.num_items, size=n_collide, p=self.probs)

        log_q = self.log_probs[neg_ids]
        return (
            torch.from_numpy(neg_ids.astype(np.int64)).to(device),
            torch.from_numpy(log_q.astype(np.float32)).to(device),
        )

    def log_q_for(self, video_ids: torch.Tensor) -> torch.Tensor:
        """log_Q(i) cho 1 tensor video_id cụ thể (dùng cho positive, không cần sample)."""
        idx = video_ids.detach().cpu().numpy()
        return torch.from_numpy(self.log_probs[idx].astype(np.float32)).to(video_ids.device)
