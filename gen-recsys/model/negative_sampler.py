"""Negative sampler cho sampled-softmax retrieval loss — sample theo TẦN SUẤT xuất hiện.

[SỬA 2026-10-05 — ALIAS TABLE] `np.random.choice(N, p=...)` KHÔNG dùng alias table: mỗi lần
gọi nó dựng CDF rồi searchsorted, O(N) trên mỗi batch. Đo trực tiếp (batch 32 x 512 negative,
đúng cấu hình train):

    N = 7,583 (Pure)   :   2.2 ms/batch
    N = 1,000,000      :  19.7 ms/batch

Alias table (Walker 1977) dựng 1 lần rồi sample O(1): 41.8s dựng + 3.71 ms/batch = NHANH HƠN
114x so với cách cũ ở N lớn. Bảng 256 MB
(prob fp32 + alias int32) cho 32M item.

KHÔNG chuyển sang uniform (0.08 ms/batch) để tránh việc này: phân phối negative là thứ đã
đo được ảnh hưởng trực tiếp lên gradient — pop-neg + logQ hỏng từng làm gap sup +0.084 ->
+0.001 (xem memory gen-recsys-ngan-sach-gradient). [XOÁ 2026-10-06] Cờ `--uniform-negatives`
(chẩn đoán cold item) đã bỏ cùng các nhánh ablation khác — cold-start không còn theo đuổi
(formula.md §−1).

[SỬA 2026-10-13 — error.md #9] TRƯỚC ĐÂY `freq_observed = np.diff(item_offsets)` dùng N_i
CUỐI KỲ (tổng toàn bộ log train+val+test, vì item_N_*.npy build từ TOÀN BỘ
LOG_STANDARD_FILES không filter split) — log_q và alias table (phân phối negative) bị FIT
TRÊN CẢ VAL+TEST. Sửa: nếu có `p80`, tra N_i của MỖI item TẠI đúng mốc đó (qua
`lookup_n_at_t_batch`) thay vì N_i cuối kỳ. `p80=None` (ví dụ gọi trực tiếp không qua train.py) giữ hành vi CŨ + cảnh báo.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess_data"))
from build_n_cumulative import lookup_n_at_t_batch  # noqa: E402


class NegativeSampler:
    def __init__(self, output_dir: str | Path, num_items: int, p80: int | None = None):
        output_dir = Path(output_dir)
        item_ids = np.load(output_dir / "item_N_ids.npy")

        if p80 is not None:
            # N_i TAI MOC p80 cho MOI item — khong leak val/test.
            ts_at_p80 = np.full(item_ids.shape[0], p80, dtype=np.int64)
            freq_observed = lookup_n_at_t_batch("item_N", item_ids, ts_at_p80).astype(np.float64)
        else:
            warnings.warn(
                "NegativeSampler khong nhan p80 — dung N_i CUOI KY (fit tren CA "
                "val+test, error.md #9). Truyen p80 (tu train_val_boundary_p80.npy) "
                "de tranh leak.",
                stacklevel=2,
            )
            item_offsets = np.load(output_dir / "item_N_offsets.npy")
            freq_observed = np.diff(item_offsets).astype(np.float64)

        freq = np.ones(num_items, dtype=np.float64)
        freq[item_ids] = freq_observed

        # [SUA 2026-10-14 — phat hien khi chay smoke-test sau fix #9] Item ra mat SAU
        # p80 co N_i(p80)=0 THAT (0.15% tren Pure, 11/7551 item) — dung ve y nghia (item
        # nay chua "pho bien" trong giai doan train), nhung log(0)=-inf gay NaN ha luu:
        # log_q_for() co the duoc goi cho CHINH item nay neu no la POSITIVE cua 1 sample
        # val/test (dung vi val/test o SAU p80) — scaled_logit = s/temp - (-inf) = +inf.
        # Dung san nho thay 0 tuyet doi: item van gan nhu KHONG BAO GIO duoc sample lam
        # negative (prob cuc nho), nhung log_q_for() khong con tra -inf.
        #
        # [SUA 2026-10-14 — code review] Floor nay AP DUNG DONG LOAT cho moi item co
        # freq=0, nen neu MOT bug alignment TUONG LAI (vd lech item_ids voi freq_observed
        # do off-by-one trong join/lookup) tao ra freq=0 cho NHIEU item hon han 11/7551 da
        # biet, floor se AM THAM hap thu thanh xac suat nho hop le thay vi crash ro qua
        # NaN nhu code CU — mat kha nang phat hien bug moi. Canh bao ro khi ty le vuot han
        # muc da biet (nguong 1% = ~7x muc 0.15% da do, du rong de khong bao dong gia).
        n_zero = int((freq_observed == 0).sum())
        frac_zero = n_zero / max(len(freq_observed), 1)
        if frac_zero > 0.01:
            warnings.warn(
                f"NegativeSampler: {n_zero}/{len(freq_observed)} item co N_i=0 "
                f"({frac_zero*100:.2f}%) — VUOT XA muc da biet (0.15%, 11/7551 tren "
                f"Pure). Co the la bug alignment MOI (item_ids lech voi freq_observed), "
                f"khong chi item ra mat sau p80. Kiem tra TRUOC khi tin ket qua.",
                stacklevel=2,
            )
        freq = np.maximum(freq, 1e-8)

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
            else:
                # [THEM 2026-10-13, NANG CAP 2026-10-14 — code review] error.md #19:
                # truoc day het 10 vong ma van con collide thi CHI warn roi AM THAM giu
                # nguyen gia tri trung — van di thang vao loss, van gay "CE ket o log(2)"
                # dung nhu docstring da canh bao, warning khong ngan duoc gi ca. Sua THAT:
                # voi dung cac vi tri con collide, thay bang uniform random tren catalog,
                # LOAI TRU dung set pos cua CHINH sample do (np.setdiff1d dam bao TUYET
                # DOI khong trung, khac het 10 vong resample theo phan phoi popularity
                # von co the mai khong thoat duoc neu pos chiem ty trong lon trong phan
                # phoi). Hy sinh: cac vi tri HIEM nay nhan negative THEO UNIFORM thay vi
                # popularity — chap nhan duoc vi day la fallback cuc hiem, khong phai
                # hanh vi chinh cua sampler.
                collide = (neg_ids[:, :, None] == pos[:, None, :]).any(axis=-1)
                n_collide_final = int(collide.sum())
                if n_collide_final > 0:
                    warnings.warn(
                        f"NegativeSampler.sample: con {n_collide_final} negative trung "
                        f"positive SAU 10 lan resample — thay THEO UNIFORM (loai tru "
                        f"dung) de dam bao KHONG con trung, thay vi giu nguyen gia tri "
                        f"trung nhu truoc. Kiem tra exclude co dung khong, hoac exclude "
                        f"qua lon so catalog.",
                        stacklevel=2,
                    )
                    rows, cols = np.where(collide)
                    for r in np.unique(rows):
                        mask_r = rows == r
                        n_r = int(mask_r.sum())
                        excluded_r = pos[r]
                        # setdiff1d dam bao TUYET DOI khong trung — khong con vong lap
                        valid_ids = np.setdiff1d(
                            np.arange(self.num_items), excluded_r, assume_unique=False
                        )
                        replacement = np.random.choice(valid_ids, size=n_r, replace=True)
                        neg_ids[r, cols[mask_r]] = replacement

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
