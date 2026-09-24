"""Negative sampler cho sampled-softmax retrieval loss — sample theo TẦN SUẤT xuất hiện"""

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
        exclude: torch.Tensor | None = None,
        uniform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về (negative_video_ids, log_q) — shape (batch_size, num_negatives)."""
        probs = None if uniform else self.probs
        neg_ids = np.random.choice(self.num_items, size=(batch_size, num_negatives), p=probs)

        if exclude is not None:
            pos = exclude.detach().cpu().numpy().reshape(-1, 1)
            for _ in range(10):
                collide = neg_ids == pos
                n_collide = int(collide.sum())
                if n_collide == 0:
                    break
                neg_ids[collide] = np.random.choice(self.num_items, size=n_collide, p=probs)

        log_q = self.log_probs[neg_ids]
        return (
            torch.from_numpy(neg_ids.astype(np.int64)).to(device),
            torch.from_numpy(log_q.astype(np.float32)).to(device),
        )

    def log_q_for(self, video_ids: torch.Tensor) -> torch.Tensor:
        """log_Q(i) cho 1 tensor video_id cụ thể (dùng cho positive, không cần sample)."""
        idx = video_ids.detach().cpu().numpy()
        return torch.from_numpy(self.log_probs[idx].astype(np.float32)).to(video_ids.device)
