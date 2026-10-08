"""Negative sampler cho sampled-softmax retrieval loss — sample theo TẦN SUẤT xuất hiện.

[SỬA 2026-10-05 — ALIAS TABLE, bắt buộc cho 27K] `np.random.choice(N, p=...)` KHÔNG dùng
alias table: mỗi lần gọi nó dựng CDF rồi searchsorted, O(N) trên mỗi batch. Đo trực tiếp
(batch 32 x 512 negative, đúng cấu hình train):

    N = 7,583 (Pure)   :   2.2 ms/batch
    N = 1,000,000      :  19.7 ms/batch
    N = 32,038,693     : 423.7 ms/batch  => 1 epoch ~ 1,177 GIỜ chỉ để sample negative

Pure chạy được nên lỗi chưa bao giờ lộ. Alias table (Walker 1977) dựng 1 lần rồi sample
O(1): 41.8s dựng + 3.71 ms/batch = NHANH HƠN 114x, 1 epoch còn ~10.3 giờ. Bảng 256 MB
(prob fp32 + alias int32) cho 32M item.

KHÔNG chuyển sang uniform (0.08 ms/batch) để tránh việc này: phân phối negative là thứ đã
đo được ảnh hưởng trực tiếp lên gradient — pop-neg + logQ hỏng từng làm gap sup +0.084 ->
+0.001 (xem memory gen-recsys-ngan-sach-gradient). [XOÁ 2026-10-06] Cờ `--uniform-negatives`
(chẩn đoán cold item) đã bỏ cùng các nhánh ablation khác — cold-start không còn theo đuổi
(formula.md §−1).
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

        freq = np.ones(num_items, dtype=np.float64)
        freq[item_ids] = freq_observed

        self.probs = freq / freq.sum()
        self.log_probs = np.log(self.probs)
        self.num_items = num_items
        self._alias_prob, self._alias_idx = _build_alias_table(self.probs)

    def sample(
        self,
        batch_size: int,
        num_negatives: int,
        device: torch.device,
        exclude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về (negative_video_ids, log_q) — shape (batch_size, num_negatives).

        `exclude` nhận (B,) HOẶC (B, K): negative dùng CHUNG cho mọi vị trí t của chuỗi nên
        phải loại trùng với TOÀN BỘ target_ids của hàng đó, không chỉ label ở vị trí cuối.
        Đo trên Pure (C=512): nếu chỉ loại label thì P(một positive lọt vào tập negative) =
        0.270 theo phân bố positive (positive thiên về popular, negative cũng sample theo
        popularity) ⇒ CE kẹt ở sàn log(2)=0.693 tại ~69/255 vị trí mỗi batch.
        """
        neg_ids = self._draw((batch_size, num_negatives))

        if exclude is not None:
            pos = exclude.detach().cpu().numpy()
            if pos.ndim == 1:
                pos = pos[:, None]
            pos = pos.reshape(batch_size, -1)
            for _ in range(10):
                # (B, C, 1) vs (B, 1, P) -> trùng với BẤT KỲ id nào trong hàng
                collide = (neg_ids[:, :, None] == pos[:, None, :]).any(axis=-1)
                n_collide = int(collide.sum())
                if n_collide == 0:
                    break
                neg_ids[collide] = self._draw((n_collide,))

        log_q = self.log_probs[neg_ids]
        return (
            torch.from_numpy(neg_ids.astype(np.int64)).to(device),
            torch.from_numpy(log_q.astype(np.float32)).to(device),
        )

    def _draw(self, shape: tuple[int, ...]) -> np.ndarray:
        """Rút negative — alias table O(1)/mẫu, KHÔNG phải np.random.choice O(N)/lần gọi."""
        i = np.random.randint(0, self.num_items, size=shape)
        u = np.random.random_sample(shape)
        return np.where(u < self._alias_prob[i], i, self._alias_idx[i])

    def log_q_for(self, video_ids: torch.Tensor) -> torch.Tensor:
        """log_Q(i) cho 1 tensor video_id cụ thể (dùng cho positive, không cần sample)."""
        idx = video_ids.detach().cpu().numpy()
        return torch.from_numpy(self.log_probs[idx].astype(np.float32)).to(video_ids.device)


def _build_alias_table(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alias table của Walker — dựng O(N) MỘT LẦN, sau đó mỗi mẫu O(1).

    Trả về (prob, alias): rút i ~ Uniform{0..N-1}, u ~ U[0,1); kết quả là i nếu
    u < prob[i], ngược lại alias[i]. Dùng hàng đợi list thay vì vòng while trên mảng để
    giữ O(N) — trên 32M item chênh lệch này là phút vs giờ.
    """
    n = len(probs)
    scaled = probs * n
    prob = np.ones(n, dtype=np.float32)
    alias = np.arange(n, dtype=np.int32)

    small = list(np.flatnonzero(scaled < 1.0))
    large = list(np.flatnonzero(scaled >= 1.0))
    q = scaled.astype(np.float64)

    while small and large:
        s_i = small.pop()
        l_i = large.pop()
        prob[s_i] = q[s_i]
        alias[s_i] = l_i
        # phần dư của l_i sau khi "lấp" cho s_i
        q[l_i] = q[l_i] - (1.0 - q[s_i])
        (small if q[l_i] < 1.0 else large).append(l_i)

    # còn sót do sai số dấu phẩy động -> xác suất 1.0, tự trỏ về chính nó
    for rest in (small, large):
        for i in rest:
            prob[i] = 1.0
    return prob, alias


def _demo() -> None:
    """ponytail: check alias table tái tạo ĐÚNG phân phối, và nhanh hơn choice."""
    import time

    rng = np.random.default_rng(0)
    p = rng.random(1000)
    p /= p.sum()
    prob, alias = _build_alias_table(p)

    # 1. phân phối thực nghiệm phải khớp p
    N = 2_000_000
    i = rng.integers(0, 1000, size=N)
    u = rng.random(N)
    draws = np.where(u < prob[i], i, alias[i])
    emp = np.bincount(draws, minlength=1000) / N
    max_err = float(np.abs(emp - p).max())
    assert max_err < 5e-4, f"alias table lech phan phoi: max |emp-p| = {max_err:.2e}"

    # 2. tổng xác suất bảo toàn
    assert abs(emp.sum() - 1.0) < 1e-9

    # 3. nhanh hơn np.random.choice ở N lớn — chính lý do thay nó
    big = rng.random(200_000)
    big /= big.sum()
    pb, ab = _build_alias_table(big)
    t = time.perf_counter()
    for _ in range(3):
        np.random.choice(200_000, size=(32, 512), p=big)
    t_choice = (time.perf_counter() - t) / 3
    t = time.perf_counter()
    for _ in range(3):
        k = np.random.randint(0, 200_000, size=(32, 512))
        np.where(np.random.random_sample((32, 512)) < pb[k], k, ab[k])
    t_alias = (time.perf_counter() - t) / 3
    assert t_alias < t_choice, f"alias ({t_alias*1e3:.2f}ms) khong nhanh hon choice ({t_choice*1e3:.2f}ms)"

    print(f"negative_sampler self-check OK "
          f"(max |emp-p| = {max_err:.1e}; alias {t_alias*1e3:.2f}ms vs choice {t_choice*1e3:.2f}ms "
          f"o N=200k)")


if __name__ == "__main__":
    _demo()
