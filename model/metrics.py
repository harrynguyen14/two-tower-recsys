"""
Metrics — xem Readme.md mục "Metric đánh giá — đã chốt".

AUC + PR-AUC trên cặp (user, item, label), KHÔNG dùng HR@10/NDCG@10 (ranking toàn catalog)
— bài toán đã chốt là phân loại nhị phân (rating>=4 vs <=2), không phải xếp hạng.

Báo cáo tách theo 4 slice: {AUC, PR-AUC} x {cold, warm} (xem Readme mục "Temporal split").
cold = item_pos chưa từng xuất hiện ở train (first-seen nằm trong val/test).
warm = item_pos đã xuất hiện ở train trước đó (đối chứng, phát hiện model có "ăn gian"
       bằng memorization thay vì học content thật hay không).
"""

from sklearn.metrics import roc_auc_score, average_precision_score


def binary_metrics(scores, labels):
    """scores, labels: array-like cùng độ dài, labels in {0,1}.
    Trả về dict {auc, pr_auc}. Nếu chỉ có 1 class (không thể tính AUC), trả về None cho
    metric đó thay vì raise — để caller quyết định cách báo cáo (vd slice quá nhỏ)."""
    labels = list(labels)
    if len(set(labels)) < 2:
        return {"auc": None, "pr_auc": None, "n": len(labels)}
    return {
        "auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "n": len(labels),
    }


def evaluate_cold_warm(scores_by_slice, labels_by_slice):
    """
    scores_by_slice, labels_by_slice: dict với đúng 2 key "cold" và "warm"
    (mỗi giá trị là array-like scores/labels của slice đó, đã gộp positive+hard-neg+soft-neg).

    Trả về dict lồng: {"cold": {"auc":..., "pr_auc":..., "n":...},
                        "warm": {"auc":..., "pr_auc":..., "n":...}}
    """
    result = {}
    for slice_name in ("cold", "warm"):
        result[slice_name] = binary_metrics(
            scores_by_slice[slice_name], labels_by_slice[slice_name]
        )
    return result


def format_report(result):
    lines = []
    for slice_name, m in result.items():
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "n/a"
        pr_auc = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "n/a"
        lines.append(f"  {slice_name:<5} AUC={auc}  PR-AUC={pr_auc}  (n={m['n']})")
    return "\n".join(lines)
    print(format_report(result))
