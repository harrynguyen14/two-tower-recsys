"""Đánh giá model — HR@K, Recall@K, NDCG@K.

TÁCH THEO VỊ TRÍ TÍN HIỆU (formula.md §−1) là trục CHÍNH: tag của item đích nằm ở lịch
sử gần, xa, cả hai, hay không đâu. Đây là thứ trả lời câu hỏi nghiên cứu — chưa benchmark
nào báo cáo. Cold user giữ lại làm lát cắt phụ.

Bảng 4 ô cold-start (user×item) ĐÃ BỎ: định vị cold-start không còn theo đuổi
(formula.md §−1 "Hướng đã bỏ").
"""

from __future__ import annotations

import torch

SHORT_WINDOW = 10   # "gần" = 10 item cuối, khớp mọi phép đo ở formula.md §−1

SIGNAL_GROUPS = ("only_long", "only_short", "both", "neither")


@torch.no_grad()
def compute_metrics_at_k(
    logit: torch.Tensor,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, torch.Tensor]:
    """{"hr@10": (B,) float, "recall@10": ..., "ndcg@10": ...} — PER-SAMPLE, chưa gộp.

    Positive luôn ở cột 0. Với 1 positive/sample thì HR@K == Recall@K về giá trị; giữ cả
    hai vì literature báo cáo lẫn lộn hai tên và ta cần đối chiếu trực tiếp.
    """
    order = torch.argsort(logit, dim=1, descending=True)
    positive_rank = (order == 0).float().argmax(dim=1)

    metrics: dict[str, torch.Tensor] = {}
    for k in k_values:
        hit = (positive_rank < k).float()
        metrics[f"hr@{k}"] = hit
        metrics[f"recall@{k}"] = hit
        metrics[f"ndcg@{k}"] = hit * (1.0 / torch.log2(positive_rank.float() + 2.0))
    metrics["mrr"] = 1.0 / (positive_rank.float() + 1.0)
    return metrics


@torch.no_grad()
def classify_signal_position(
    hist_video_ids: torch.Tensor,
    label_video_id: torch.Tensor,
    hist_valid_mask: torch.Tensor,
    item_tag_table: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Tag của item đích xuất hiện ở vùng nào của lịch sử — trục chính của câu hỏi.

    `item_tag_table`: (num_items, num_tags) bool, True nếu item mang tag đó.
    Trả về 4 mask loại trừ nhau, cộng lại đúng B.
    """
    B, K = hist_video_ids.shape
    target_tags = item_tag_table[label_video_id]                 # (B, T)
    hist_tags = item_tag_table[hist_video_ids]                   # (B, K, T)

    shares = (hist_tags & target_tags.unsqueeze(1)).any(dim=-1) & hist_valid_mask  # (B, K)

    pos = torch.arange(K, device=hist_video_ids.device).expand(B, K)
    near_zone = pos >= (K - SHORT_WINDOW)                        # SHORT_WINDOW item cuối
    near = (shares & near_zone).any(dim=1)
    far = (shares & ~near_zone).any(dim=1)

    return {
        "only_long": far & ~near,     # ngắn hạn MÙ — 37.9% ở ngưỡng lịch sử >=200
        "only_short": near & ~far,
        "both": near & far,
        "neither": ~near & ~far,
    }


def aggregate_by_group(
    per_sample_metrics: dict[str, torch.Tensor],
    group_masks: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Gộp per-sample theo các mask đã cho. Nhóm rỗng trả nan, không phải 0."""
    result: dict[str, dict[str, float]] = {}
    for name, mask in group_masks.items():
        n = int(mask.sum().item())
        result[name] = {"n_samples": n}
        for metric_name, values in per_sample_metrics.items():
            result[name][metric_name] = (
                float(values[mask].mean().item()) if n > 0 else float("nan")
            )
    return result


def _ci95(p: float, n: int) -> float:
    """Nửa khoảng tin cậy 95% (xấp xỉ Wald). In kèm mọi hr/recall để KHÔNG ai đọc một
    con số lẻ trên nhóm n nhỏ rồi kết luận."""
    if n <= 0:
        return float("nan")
    return 1.96 * (max(p * (1 - p), 1e-12) / n) ** 0.5


def _print_table(title: str, result: dict[str, dict[str, float]]) -> None:
    print(f"\n  --- {title} ---")
    for name, metrics in result.items():
        n = metrics["n_samples"]
        parts = []
        for key, value in metrics.items():
            if key == "n_samples":
                continue
            if key.startswith(("hr", "recall")):
                parts.append(f"{key}={value:.4f}+-{_ci95(value, n):.3f}")
            else:
                parts.append(f"{key}={value:.4f}")
        print(f"    [{name:<12}] n={n:<6} " + " ".join(parts))


def print_eval_report(
    result_overall: dict[str, dict[str, float]],
    tau_snapshot: dict[str, float],
    result_signal: dict[str, dict[str, float]] | None = None,
    result_cold: dict[str, dict[str, float]] | None = None,
) -> None:
    print("\n=== EVAL REPORT ===")
    print(f"τ_u={tau_snapshot['tau_u']:.2f} τ_i={tau_snapshot['tau_i']:.2f}")
    _print_table("TONG THE", result_overall)
    if result_signal is not None:
        _print_table(
            "VI TRI TIN HIEU (truc chinh — formula.md §−1): only_long = ngan han MU",
            result_signal,
        )
    if result_cold is not None:
        _print_table("COLD USER (lat cat phu)", result_cold)
