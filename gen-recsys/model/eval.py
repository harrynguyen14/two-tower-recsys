"""Đánh giá model — Recall@K, NDCG@K TÁCH RIÊNG theo 4 nhóm cold-start (user×item:
warm/warm, cold/warm, warm/cold, cold/cold) — xem idea.md "Câu hỏi nghiên cứu": phải
chứng minh quyết định đề xuất THỰC SỰ đổi theo đúng bên đang cold, không chỉ cải thiện
metric tổng không phân biệt được nguồn gốc.

Cách tính: với mỗi sample trong tập eval, xếp hạng label thật (positive) trong 1 tập
candidate (positive + N negative theo tần suất, giống lúc train — retrieval, KHÔNG phải
ranking full-catalog vì 32M item không thể tính hết mỗi sample). Recall@K = tỉ lệ label
nằm trong top-K theo logit. NDCG@K = DCG chuẩn hóa (chỉ 1 positive/sample nên
NDCG@K = 1/log2(rank+2) nếu rank<K, else 0 — công thức rút gọn cho single-relevant-item).

Cũng theo dõi gate g_i theo N_interactions (idea.md: "vẽ gate value theo N_interactions —
kỳ vọng thấy đường cong đơn điệu tăng giống DropoutNet/GateSID") và τ snapshot mỗi lần
gọi evaluate() để phát hiện threshold-collapse (xem learnable_thresholds.py).
"""

from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def compute_recall_ndcg_at_k(
    logit: torch.Tensor,  # (B, C) — logit(u, candidate), candidate[:, 0] LUÔN là positive
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, torch.Tensor]:
    """Trả về {"recall@5": (B,) bool, "ndcg@5": (B,) float, ...} — PER-SAMPLE, chưa
    average, để gọi nơi khác tách nhóm cold/warm rồi mới .mean()."""
    B, C = logit.shape
    # rank của positive (cột 0) trong thứ tự giảm dần theo logit — rank=0 nghĩa là đứng đầu
    order = torch.argsort(logit, dim=1, descending=True)  # (B, C), order[b, 0] = idx logit lớn nhất
    positive_rank = (order == 0).float().argmax(dim=1)  # (B,) — vị trí của cột 0 sau khi sort

    metrics = {}
    for k in k_values:
        hit = (positive_rank < k).float()  # (B,) — 1.0 nếu positive lọt top-K
        metrics[f"recall@{k}"] = hit
        # NDCG rút gọn cho đúng 1 relevant item/sample: DCG = 1/log2(rank+2) nếu hit, IDCG=1
        ndcg = hit * (1.0 / torch.log2(positive_rank.float() + 2.0))
        metrics[f"ndcg@{k}"] = ndcg
    return metrics


def aggregate_by_cold_group(
    per_sample_metrics: dict[str, torch.Tensor],  # tên metric -> (N,) tensor, N = tổng sample đã eval
    is_user_cold: torch.Tensor,  # (N,) bool
    is_item_cold: torch.Tensor,  # (N,) bool
) -> dict[str, dict[str, float]]:
    """Tách kết quả theo 4 nhóm: warm_warm, cold_user_warm_item, warm_user_cold_item,
    cold_cold — xem idea.md "Cách test bám sát câu hỏi".

    `is_user_cold` nhận ĐƯỢC CẢ HAI định nghĩa (strict holdout hoặc few-shot) — caller
    quyết định, gọi 2 lần để có 2 bảng. Xem print_eval_report."""
    groups = {
        "warm_warm": (~is_user_cold) & (~is_item_cold),
        "cold_user_warm_item": is_user_cold & (~is_item_cold),
        "warm_user_cold_item": (~is_user_cold) & is_item_cold,
        "cold_cold": is_user_cold & is_item_cold,
    }

    result = {}
    for group_name, mask in groups.items():
        n = int(mask.sum().item())
        result[group_name] = {"n_samples": n}
        for metric_name, values in per_sample_metrics.items():
            result[group_name][metric_name] = float(values[mask].mean().item()) if n > 0 else float("nan")
    return result


def _ci95(p: float, n: int) -> float:
    """Nửa khoảng tin cậy 95% cho tỉ lệ (xấp xỉ Wald). In kèm mọi recall để KHÔNG ai đọc
    chênh lệch nhỏ trên nhóm n bé là hiệu ứng thật — ô cold/cold chỉ có 65-81 sample."""
    if n <= 0:
        return float("nan")
    return 1.96 * (max(p * (1 - p), 1e-12) / n) ** 0.5


def _print_table(title: str, result: dict[str, dict[str, float]]) -> None:
    print(f"\n  --- {title} ---")
    for group_name, metrics in result.items():
        n = metrics["n_samples"]
        parts = []
        for k, v in metrics.items():
            if k == "n_samples":
                continue
            # chỉ recall là tỉ lệ nhị phân -> CI có nghĩa; ndcg in trần
            parts.append(f"{k}={v:.4f}+-{_ci95(v, n):.3f}" if k.startswith("recall") else f"{k}={v:.4f}")
        print(f"    [{group_name}] n={n} " + " ".join(parts))


def print_eval_report(
    result: dict[str, dict[str, float]],
    tau_snapshot: dict[str, float],
    result_lowhistory: dict[str, dict[str, float]] | None = None,
) -> None:
    """[SỬA 2026-09-15] In HAI bảng 4 ô — hai định nghĩa cold user trả lời hai câu hỏi
    khác nhau, gộp lại thành một bảng là đánh tráo:

      strict holdout — "user CHƯA TỪNG xuất hiện" (zero-shot thật). Nghiêm nhất, nhưng ô
                       cold/cold chỉ 5 (val) / 13 (test) sample -> KHÔNG kết luận được gì,
                       CI rộng hơn chính giá trị đo. Train có 0 sample.
      few-shot       — "user ít lịch sử tại thời điểm dự đoán" (N_u < LOW_HISTORY_N). Ô
                       cold/cold 81 (val) / 65 (test), và train có 31,538 ví dụ để γ HỌC.

    Cả hai in kèm CI95 để chênh lệch nhỏ trên nhóm n bé không bị đọc nhầm là thật."""
    print("\n=== EVAL REPORT (tách theo nhóm cold-start) ===")
    print(f"τ_u={tau_snapshot['tau_u']:.2f} τ_i={tau_snapshot['tau_i']:.2f} τ_c={tau_snapshot['tau_c']:.2f}")
    _print_table("user-cold = STRICT HOLDOUT (zero-shot; n nho, doc than trong)", result)
    if result_lowhistory is not None:
        _print_table("user-cold = FEW-SHOT N_u<LOW_HISTORY_N (noi gamma thuc su hoc duoc)", result_lowhistory)
