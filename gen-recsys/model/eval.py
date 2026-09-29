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


def cold_vs_warm_matched(
    per_sample_metrics: dict[str, torch.Tensor],
    is_user_cold: torch.Tensor,
    n_u: torch.Tensor,
    metric: str = "hr@10",
) -> dict[str, dict[str, float]]:
    """So cold vs warm KHOP THEO n_u — lat cat cold DUY NHAT so sanh duoc.

    Vi sao can (do 2026-09-29, `probes/probe_cold_vs_warm_matched.py`): lat cat
    `cold_user`/`warm_user` tho KHONG so sanh duoc vi hai nhom cham o VI TRI CHUOI khac
    nhau — median n_u cold=20 vs warm=69; 51% cold co n_u<=20 so voi 7% warm (lech 6.9x).
    Du doan tuong tac thu 20 DE HON thu 69, nen cold cao hon warm la artefact cua thiet ke
    lat cat, khong phai ket qua.

    Cach sua: chia tang n_u, tinh metric trong TUNG tang, roi gop lai bang trung binh co
    trong so theo so sample WARM cua tang do (direct standardisation — chuan hoa cold ve
    dung phan bo n_u cua warm). Ket qua tra ve co khoa `cold_std` so sanh truc tiep duoc
    voi `warm`.
    """
    bins = [(0, 5), (6, 20), (21, 50), (51, 100), (101, 300), (301, 10 ** 9)]
    out: dict[str, dict[str, float]] = {}
    vals = per_sample_metrics[metric]
    num = num_w = den = 0.0
    for lo, hi in bins:
        in_bin = (n_u >= lo) & (n_u <= hi)
        c, w = in_bin & is_user_cold, in_bin & ~is_user_cold
        nc, nw = int(c.sum()), int(w.sum())
        out[f"n_u {lo}-{hi if hi < 10 ** 9 else '+'}"] = {
            "n_samples": nc + nw,
            "n_cold": nc, "n_warm": nw,
            f"cold_{metric}": float(vals[c].mean()) if nc else float("nan"),
            f"warm_{metric}": float(vals[w].mean()) if nw else float("nan"),
        }
        # Chuan hoa: chi cong tang co DU sample CA HAI ben (n>=30). CA HAI phia deu phai
        # cong tren CUNG tap tang, neu khong thi lai so hai phan bo khac nhau (bug da bat
        # duoc bang kiem dinh tong hop 2026-09-29).
        if nc >= 30 and nw >= 30:
            num += float(vals[c].mean()) * nw
            num_w += float(vals[w].mean()) * nw
            den += nw
    n_cold_tot, n_warm_tot = int(is_user_cold.sum()), int((~is_user_cold).sum())
    out["TONG"] = {
        "n_samples": n_cold_tot + n_warm_tot,
        "n_cold": n_cold_tot, "n_warm": n_warm_tot,
        f"cold_{metric}": float(vals[is_user_cold].mean()) if n_cold_tot else float("nan"),
        f"warm_{metric}": float(vals[~is_user_cold].mean()) if n_warm_tot else float("nan"),
    }
    out["cold_CHUAN_HOA"] = {
        "n_samples": int(den),
        f"cold_{metric}": num / den if den else float("nan"),
        f"warm_{metric}": num_w / den if den else float("nan"),
    }
    return out


def print_cold_matched(res: dict[str, dict[str, float]], metric: str = "hr@10") -> None:
    print()
    print("  --- COLD vs WARM KHOP n_u (lat cat tho KHONG so sanh duoc, xem eval.py) ---")
    print(f"    {'tang n_u':<14} {'n_cold':>7} {'n_warm':>8} {'cold':>8} {'warm':>8}   ghi chu")
    for k, v in res.items():
        if k in ("TONG", "cold_CHUAN_HOA"):
            continue
        c, w = v[f"cold_{metric}"], v[f"warm_{metric}"]
        note = "n nho, bo qua" if (v["n_cold"] < 30 or v["n_warm"] < 30) else ""
        print(f"    {k:<14} {v['n_cold']:>7,} {v['n_warm']:>8,} {c:>8.4f} {w:>8.4f}   {note}")
    t, std = res["TONG"], res["cold_CHUAN_HOA"]
    print(f"    {'TONG (tho)':<14} {t['n_cold']:>7,} {t['n_warm']:>8,} "
          f"{t[f'cold_{metric}']:>8.4f} {t[f'warm_{metric}']:>8.4f}   <- KHONG so sanh duoc")
    cs, ws = std[f"cold_{metric}"], std[f"warm_{metric}"]
    # So CO TINH SAI SO: nhom cold nho (n~2000 -> +-0.012 o hr@10) nen chenh lech vai %
    # KHONG phan biet duoc voi 0. So cung `ws > cs` se gan nhan "VI PHAM" cho nhieu.
    ci = _ci95(cs, max(t["n_cold"], 1))
    if abs(ws - cs) <= ci:
        verdict = f"KHONG PHAN BIET DUOC (lech {abs(ws - cs):.4f} <= CI95 {ci:.4f})"
    elif ws > cs:
        verdict = "WARM > COLD (dung quy luat)"
    else:
        verdict = "COLD > WARM (vuot sai so — can dieu tra)"
    print(f"    {'CHUAN HOA':<14} {'':>7} {std['n_samples']:>8,} "
          f"{cs:>8.4f} {ws:>8.4f}   <- {verdict}")
    print("    (chuan hoa chi dung tang co n>=30 CA HAI ben; cold hau nhu khong ton tai "
          "o n_u>50 nen hai nhom chi chong nhau o vung n_u nho)")


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
