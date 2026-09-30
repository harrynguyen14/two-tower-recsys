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


def matched_by_n_u(
    per_sample_metrics: dict[str, torch.Tensor],
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    n_u: torch.Tensor,
    metric: str = "hr@10",
    name_a: str = "cold",
    name_b: str = "warm",
) -> dict[str, dict[str, float]]:
    """So hai nhom KHOP THEO n_u — cach DUY NHAT so sanh duoc khi hai nhom cham o
    vi tri chuoi khac nhau. Dung cho cold/warm VA cho nhom tin hieu (`both` median n_u=68
    vs `only_short`=17, lech 4.0x — do 2026-09-30, `probes/probe_both_confound.py`).

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
    out: dict[str, dict[str, float]] = {"_names": {"a": name_a, "b": name_b}}
    vals = per_sample_metrics[metric]
    num_a = num_b = den = 0.0
    for lo, hi in bins:
        in_bin = (n_u >= lo) & (n_u <= hi)
        a, b = in_bin & mask_a, in_bin & mask_b
        na, nb = int(a.sum()), int(b.sum())
        out[f"n_u {lo}-{hi if hi < 10 ** 9 else '+'}"] = {
            "n_samples": na + nb,
            "n_a": na, "n_b": nb,
            f"a_{metric}": float(vals[a].mean()) if na else float("nan"),
            f"b_{metric}": float(vals[b].mean()) if nb else float("nan"),
        }
        # Chuan hoa: chi cong tang co DU sample CA HAI ben (n>=30). CA HAI phia deu phai
        # cong tren CUNG tap tang, neu khong thi lai so hai phan bo khac nhau (bug da bat
        # duoc bang kiem dinh tong hop 2026-09-29). Trong so = n cua nhom B (nhom THAM CHIEU).
        if na >= 30 and nb >= 30:
            num_a += float(vals[a].mean()) * nb
            num_b += float(vals[b].mean()) * nb
            den += nb
    na_tot, nb_tot = int(mask_a.sum()), int(mask_b.sum())
    out["TONG"] = {
        "n_samples": na_tot + nb_tot,
        "n_a": na_tot, "n_b": nb_tot,
        f"a_{metric}": float(vals[mask_a].mean()) if na_tot else float("nan"),
        f"b_{metric}": float(vals[mask_b].mean()) if nb_tot else float("nan"),
    }
    out["CHUAN_HOA"] = {
        "n_samples": int(den),
        f"a_{metric}": num_a / den if den else float("nan"),
        f"b_{metric}": num_b / den if den else float("nan"),
    }
    return out


def print_matched(res: dict[str, dict[str, float]], metric: str = "hr@10",
                  expect: str = "b>a") -> None:
    """In bang khop n_u. `expect` = quan he KY VONG neu quy luat dung ("b>a" hoac "a>b")."""
    na_, nb_ = res["_names"]["a"], res["_names"]["b"]
    print()
    print(f"  --- {na_.upper()} vs {nb_.upper()} KHOP n_u "
          f"(lat cat tho KHONG so sanh duoc, xem eval.py) ---")
    print(f"    {'tang n_u':<14} {f'n {na_}':>9} {f'n {nb_}':>9} {na_:>9} {nb_:>9}   ghi chu")
    for k, v in res.items():
        if k in ("TONG", "CHUAN_HOA", "_names"):
            continue
        a, b = v[f"a_{metric}"], v[f"b_{metric}"]
        note = "n nho, bo qua" if (v["n_a"] < 30 or v["n_b"] < 30) else ""
        print(f"    {k:<14} {v['n_a']:>9,} {v['n_b']:>9,} {a:>9.4f} {b:>9.4f}   {note}")
    t, std = res["TONG"], res["CHUAN_HOA"]
    print(f"    {'TONG (tho)':<14} {t['n_a']:>9,} {t['n_b']:>9,} "
          f"{t[f'a_{metric}']:>9.4f} {t[f'b_{metric}']:>9.4f}   <- KHONG so sanh duoc")
    av, bv = std[f"a_{metric}"], std[f"b_{metric}"]
    # So CO TINH SAI SO: nhom nho (n~2000 -> +-0.012 o hr@10) nen chenh lech vai %
    # KHONG phan biet duoc voi 0. So cung se gan nhan "VI PHAM" cho nhieu.
    ci = _ci95(av, max(min(t["n_a"], t["n_b"]), 1))
    hi, lo = (bv, av) if expect == "b>a" else (av, bv)
    hi_n, lo_n = (nb_, na_) if expect == "b>a" else (na_, nb_)
    if abs(bv - av) <= ci:
        verdict = f"KHONG PHAN BIET DUOC (lech {abs(bv - av):.4f} <= CI95 {ci:.4f})"
    elif hi > lo:
        verdict = f"{hi_n.upper()} > {lo_n.upper()} (dung ky vong)"
    else:
        verdict = f"{lo_n.upper()} > {hi_n.upper()} (NGUOC ky vong — vuot sai so, can dieu tra)"
    print(f"    {'CHUAN HOA':<14} {'':>9} {std['n_samples']:>9,} "
          f"{av:>9.4f} {bv:>9.4f}   <- {verdict}")
    print(f"    (chuan hoa chi dung tang co n>=30 CA HAI ben, trong so theo n cua {nb_})")


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
