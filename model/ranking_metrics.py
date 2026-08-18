"""
Ranking metrics — Recall@K / Hit@K / NDCG@K / MRR@K trên FULL CATALOG.

Thay thế AUC/PR-AUC (metrics.py) làm metric chính. Lý do đổi: AUC được tính trên pool
gộp [1 positive + 8 negative] mỗi sample, nhưng InfoNCE chỉ tối ưu thứ hạng TRONG TỪNG
DÒNG và bất biến với việc cộng hằng số riêng mỗi user — nên một model xếp hạng hoàn hảo
cho MỌI user vẫn chỉ đạt pooled AUC ~0.76 (đã đo bằng mô phỏng). Metric bị lệch khỏi
mục tiêu train.

Ngoài ra Hit@K trên 9 candidate là "sampled metric": Krichene & Rendle (KDD'20) chứng minh
loại metric này xếp hạng model SAI LỆCH so với metric thật. Vì vậy ở đây xếp hạng positive
trên TOÀN BỘ catalog (1.59M item) — đúng cách NVIDIA Merlin đánh giá retrieval.

QUY ƯỚC ĐÃ CHỐT:
  - Ground-truth GOM THEO USER, không theo từng dòng interaction. Đo thật trên val:
    57% user chỉ có 1 positive nhưng max lên tới 630 (mean 4.61) — tính Recall theo từng
    dòng riêng lẻ sẽ đếm trùng user nhiều positive và làm lệch metric về phía user đó.
  - Recall@K = |hit| / |GT|  (chuẩn học thuật). User có |GT| > K bị chặn trần
    (|GT|=630, K=10 -> Recall tối đa 0.016) — đó là lý do BẮT BUỘC báo cáo kèm Hit@K
    (có ít nhất 1 hit) để không bị các user nhiều positive làm méo bức tranh.
  - Item đã xuất hiện trong lịch sử TRAIN của user bị LOẠI khỏi candidate trước khi xếp
    hạng (không thể "khuyến nghị lại" cái user đã đọc). Ảnh hưởng tới catalog rất nhỏ
    (~30/1.59M = 0.002%) nhưng nếu không loại, item đã xem sẽ chiếm slot top-K thật.
  - NDCG@K dùng IDCG = tổng của min(|GT|, K) vị trí đầu (không phải K vị trí) — nếu chia
    theo K, user có ít positive hơn K sẽ không bao giờ đạt NDCG=1.0 dù xếp hạng hoàn hảo.

KHÔNG materialize ma trận score [n_users, 1.59M]: batch=256 x 1.59M fp16 = 0.81 GB mỗi
batch, và fp32 gấp đôi. Thay vào đó chunk theo ITEM và giữ top-K running (xem
topk_over_catalog) — chỉ 0.1 GB/chunk với item_chunk=200k.
"""

import numpy as np
import torch


def topk_over_catalog(user_vectors, item_vectors, k, exclude=None, item_chunk=200_000):
    """Xếp hạng TOÀN BỘ item_vectors cho từng user, trả về index top-k.

    user_vectors : [B, dim]   (đã L2-normalize, xem two_tower_model.forward)
    item_vectors : [N, dim]   (đã L2-normalize)
    exclude      : list[B] của array-like item index cần loại (lịch sử train của user),
                   hoặc None.
    item_chunk   : số item xử lý mỗi lần — chunk để không bao giờ giữ [B, N] trong VRAM.

    Trả về: [B, k] long tensor — index item theo thứ tự score giảm dần (-1 = slot rỗng
    sau khi lọc exclude).

    Duyệt từng chunk item, giữ top-k running rồi merge: sau mỗi chunk ta có
    (k cũ + k mới) ứng viên, topk lại trên 2k phần tử đó. Đúng vì top-k toàn cục luôn là
    tập con của hợp các top-k từng chunk.

    Khi có exclude, lấy top-(k + max_exclude) trong lúc chunk rồi mới lọc — nếu chỉ lấy
    top-k thô rồi lọc, user có nhiều item bị loại sẽ còn ÍT HƠN k item sạch (slot rỗng
    tính là miss, làm Recall thấp giả tạo).
    """
    b = user_vectors.size(0)
    n_items = item_vectors.size(0)

    k_internal = k
    if exclude is not None:
        max_ex = max((len(e) if e is not None else 0) for e in exclude) if b else 0
        k_internal = k + max_ex
    k_internal = min(k_internal, n_items)

    best_scores = None
    best_idx = None

    for start in range(0, n_items, item_chunk):
        end = min(start + item_chunk, n_items)
        chunk_scores = user_vectors @ item_vectors[start:end].t()  # [B, chunk]

        kk = min(k_internal, end - start)
        s, i = torch.topk(chunk_scores, kk, dim=1)
        i = i + start  # đưa về index toàn cục
        del chunk_scores

        if best_scores is None:
            best_scores, best_idx = s, i
        else:
            cat_s = torch.cat([best_scores, s], dim=1)
            cat_i = torch.cat([best_idx, i], dim=1)
            kk2 = min(k_internal, cat_s.size(1))
            best_scores, sel = torch.topk(cat_s, kk2, dim=1)
            best_idx = torch.gather(cat_i, 1, sel)

    if exclude is not None:
        best_idx = _filter_excluded(best_idx, exclude, k)

    return best_idx


def _filter_excluded(topk_idx, exclude, k):
    """Loại các item trong exclude[i] khỏi topk_idx[i], giữ thứ tự, cắt về k.
    Slot không đủ điền -1 (ranking_metrics_at_k coi -1 là miss)."""
    out = torch.full((topk_idx.size(0), k), -1, dtype=torch.long, device=topk_idx.device)
    idx_cpu = topk_idx.cpu().numpy()
    for i, row in enumerate(idx_cpu):
        ex = exclude[i]
        if ex is None or len(ex) == 0:
            keep = [int(x) for x in row[:k]]
        else:
            ex_set = {int(x) for x in ex}
            keep = [int(x) for x in row if int(x) not in ex_set][:k]
        if keep:
            out[i, :len(keep)] = torch.as_tensor(keep, dtype=torch.long, device=topk_idx.device)
    return out


def ranking_metrics_at_k(topk_idx, ground_truth, ks):
    """Tính Recall/Hit/NDCG/MRR @ mỗi k trong ks, gom theo USER.

    topk_idx     : [n_users, max_k] index item đã xếp hạng giảm dần (-1 = slot rỗng)
    ground_truth : list[n_users] của set/array item index thật của user đó
    ks           : list[int], phải <= topk_idx.shape[1]

    Trả về dict {f"recall@{k}", f"hit@{k}", f"ndcg@{k}", f"mrr@{k}"} + "n_users"
    — mỗi giá trị là TRUNG BÌNH TRÊN USER (macro-average), không phải trên interaction.
    """
    topk = topk_idx.cpu().numpy() if torch.is_tensor(topk_idx) else np.asarray(topk_idx)
    n_users = len(ground_truth)
    max_k = max(ks)

    # discount[j] = 1/log2(j+2) — dùng chung cho DCG và IDCG
    discount = 1.0 / np.log2(np.arange(max_k) + 2.0)

    acc = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "hit", "ndcg", "mrr")}
    n_valid = 0

    for u in range(n_users):
        gt = ground_truth[u]
        if gt is None or len(gt) == 0:
            continue  # user không có positive nào trong slice này -> bỏ khỏi macro-average
        n_valid += 1
        gt_set = gt if isinstance(gt, set) else {int(x) for x in gt}
        n_gt = len(gt_set)

        row = topk[u][:max_k]
        rel = np.fromiter((1.0 if int(x) in gt_set else 0.0 for x in row),
                          dtype=np.float64, count=len(row))

        for k in ks:
            rel_k = rel[:k]
            n_hit = rel_k.sum()

            acc[f"recall@{k}"] += n_hit / n_gt
            acc[f"hit@{k}"] += 1.0 if n_hit > 0 else 0.0

            dcg = float((rel_k * discount[:len(rel_k)]).sum())
            # IDCG = min(n_gt, k) vị trí đầu — không phải k vị trí (xem docstring module)
            idcg = float(discount[:min(n_gt, k)].sum())
            acc[f"ndcg@{k}"] += dcg / idcg if idcg > 0 else 0.0

            nz = np.flatnonzero(rel_k)
            acc[f"mrr@{k}"] += 1.0 / (nz[0] + 1.0) if len(nz) else 0.0

    if n_valid == 0:
        out = {key: None for key in acc}
        out["n_users"] = 0
        return out

    result = {key: val / n_valid for key, val in acc.items()}
    result["n_users"] = n_valid
    return result


def format_ranking_report(result, prefix=""):
    """In gọn theo từng k: 1 dòng mỗi k, 4 metric trên cùng dòng."""
    if not result or result.get("n_users", 0) == 0:
        return f"{prefix}  (không có user nào có ground-truth)"

    ks = sorted({int(key.split("@")[1]) for key in result if "@" in key})
    lines = [f"{prefix}  (n_users={result['n_users']:,})"]
    for k in ks:
        lines.append(
            f"{prefix}  @{k:<4} Recall={result[f'recall@{k}']:.4f}  "
            f"Hit={result[f'hit@{k}']:.4f}  "
            f"NDCG={result[f'ndcg@{k}']:.4f}  "
            f"MRR={result[f'mrr@{k}']:.4f}"
        )
    return "\n".join(lines)


def _self_check():
    """Chốt lại 4 QUY ƯỚC ở docstring module + tính đúng đắn của chunking/exclude.

    Chạy: python -X utf8 ranking_metrics.py
    Mỗi nhóm assert ứng với một quy ước có thể vỡ lặng lẽ (metric sai mà không crash)."""
    import torch.nn.functional as _F
    ok = []

    # ── 1. chunking cho kết quả GIỐNG HỆT torch.topk một lần ───────────────────
    # Đây là quy ước dễ vỡ nhất: merge top-k running sai thì thứ hạng lệch nhẹ, metric
    # vẫn "hợp lý" nên không ai phát hiện. Thử nhiều chunk size, kể cả chunk < k.
    torch.manual_seed(0)
    items = _F.normalize(torch.randn(500, 16), dim=-1)
    users = _F.normalize(torch.randn(8, 16), dim=-1)
    ref = torch.topk(users @ items.t(), 10, dim=1).indices
    for ch in (5000, 500, 137, 50, 7, 1):
        got = topk_over_catalog(users, items, 10, item_chunk=ch)
        assert torch.equal(got, ref), f"chunk={ch} cho thứ hạng khác torch.topk"
    ok.append("chunking khớp torch.topk với chunk=5000/500/137/50/7/1")

    # ── 2. exclude: lọc đúng + lấy top-(k+max_ex) TRƯỚC khi lọc ────────────────
    # Nếu lọc sau khi lấy top-k thô, user bị loại nhiều item sẽ còn ÍT HƠN k item sạch
    # -> slot rỗng tính là miss -> Recall thấp giả tạo.
    full = users @ items.t()
    exclude = [torch.topk(full[i], 5).indices.numpy() for i in range(8)]  # loại đúng top-5
    got = topk_over_catalog(users, items, 10, exclude=exclude, item_chunk=137)
    assert int((got < 0).sum()) == 0, "còn slot -1 dù catalog dư item sạch"
    for i in range(8):
        order = torch.argsort(full[i], descending=True).tolist()
        want = [x for x in order if x not in set(int(e) for e in exclude[i])][:10]
        assert [int(x) for x in got[i]] == want, f"user {i}: exclude lọc sai"
    ok.append("exclude lọc đúng thứ tự, không để lại slot rỗng khi catalog còn dư")

    # catalog nhỏ hơn k+max_ex -> thiếu slot là ĐÚNG, phải điền -1 (miss), không crash
    small = _F.normalize(torch.randn(12, 16), dim=-1)
    got_s = topk_over_catalog(users, small, 10,
                              exclude=[np.arange(8)] * 8, item_chunk=5)
    assert int((got_s[0] >= 0).sum()) == 4, "12 item loại 8, K=10 -> phải còn đúng 4 slot"
    ok.append("catalog < k+max_ex: điền -1 thay vì crash")

    # ── 3. Recall = hits/|GT| nên user có |GT| > K bị CHẶN TRẦN ────────────────
    # Đây là lý do BẮT BUỘC báo cáo kèm Hit@K. Nếu ai đó "sửa" thành hits/min(|GT|,K)
    # thì Recall của user nặng nhảy lên 1.0 và bức tranh bị bóp méo.
    topk = torch.arange(10).unsqueeze(0)          # đoán đúng 10 item đầu
    gt_big = [set(range(100))]                    # user có 100 positive
    r = ranking_metrics_at_k(topk, gt_big, ks=[10])
    assert abs(r["recall@10"] - 0.10) < 1e-9, f"Recall phải = 10/100 = 0.10, được {r['recall@10']}"
    assert r["hit@10"] == 1.0, "Hit@K phải = 1.0 (có ít nhất 1 hit)"
    ok.append("Recall@K = hits/|GT| (user |GT|=100, K=10 -> 0.10, bị chặn trần)")

    # ── 4. NDCG dùng IDCG = min(|GT|,K) vị trí đầu, KHÔNG phải K vị trí ────────
    # Chia theo K thì user có ít positive hơn K không bao giờ đạt 1.0 dù xếp hạng hoàn hảo.
    topk_p = torch.tensor([[7, 3, 99, 42, 5]])
    r1 = ranking_metrics_at_k(topk_p, [{7}], ks=[5])          # 1 positive, đúng hạng 1
    assert abs(r1["ndcg@5"] - 1.0) < 1e-9, f"xếp hạng hoàn hảo phải NDCG=1.0, được {r1['ndcg@5']}"
    r2 = ranking_metrics_at_k(torch.tensor([[7, 3, 1, 2, 4]]), [{7, 3}], ks=[5])
    assert abs(r2["ndcg@5"] - 1.0) < 1e-9, "2 positive ở hạng 1-2 cũng phải NDCG=1.0"
    # và xếp hạng KÉM phải < 1.0 (nếu không test mất tác dụng)
    r3 = ranking_metrics_at_k(torch.tensor([[9, 9, 9, 9, 7]]), [{7}], ks=[5])
    assert r3["ndcg@5"] < 0.5, "positive ở hạng 5 mà NDCG vẫn cao -> IDCG sai"
    ok.append("NDCG: IDCG=min(|GT|,K) -> xếp hạng hoàn hảo đạt đúng 1.0")

    # ── 5. MRR = 1/hạng của hit ĐẦU TIÊN ──────────────────────────────────────
    assert abs(ranking_metrics_at_k(torch.tensor([[1, 2, 7]]), [{7}], ks=[3])["mrr@3"]
               - 1.0 / 3.0) < 1e-9
    assert ranking_metrics_at_k(torch.tensor([[7, 2, 1]]), [{7}], ks=[3])["mrr@3"] == 1.0
    # BẮT BUỘC có case NHIỀU hit để phân biệt "hit đầu" với "hit cuối" — chỉ dùng case
    # 1-hit thì nz[0] và nz[-1] trùng nhau, test mất khả năng phát hiện lỗi hoán đầu/cuối
    # (đã kiểm bằng mutation test: bản chỉ có case 1-hit để lọt mutation nz[0]->nz[-1]).
    r_multi = ranking_metrics_at_k(torch.tensor([[9, 7, 3, 8]]), [{7, 3}], ks=[4])
    assert abs(r_multi["mrr@4"] - 0.5) < 1e-9, (
        f"MRR phải = 1/2 (hit ĐẦU ở hạng 2), được {r_multi['mrr@4']} "
        "— nếu = 1/3 thì đang dùng hit CUỐI")
    ok.append("MRR = 1/hạng hit đầu tiên (có case nhiều hit để phân biệt đầu/cuối)")

    # ── 6. -1 tính là MISS, không bao giờ khớp ground-truth ───────────────────
    r = ranking_metrics_at_k(torch.full((1, 5), -1), [{0}], ks=[5])
    assert r["hit@5"] == 0.0 and r["recall@5"] == 0.0, "slot -1 bị tính thành hit"
    ok.append("slot -1 tính là miss")

    # ── 7. macro-average THEO USER, user không có GT bị loại khỏi mẫu ─────────
    # Gom theo user chứ không theo interaction: user 630-positive không được đếm 630 lần.
    topk_m = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    r = ranking_metrics_at_k(topk_m, [{0}, set(), {5}], ks=[3])
    assert r["n_users"] == 2, f"user GT rỗng phải bị loại, n_users={r['n_users']}"
    assert abs(r["hit@3"] - 0.5) < 1e-9, "macro-average trên 2 user hợp lệ = 0.5"
    ok.append("macro-average theo user; user không có GT bị loại khỏi mẫu")

    # toàn bộ GT rỗng -> trả None thay vì chia cho 0
    r = ranking_metrics_at_k(topk_m, [set(), set(), set()], ks=[3])
    assert r["n_users"] == 0 and r["hit@3"] is None, "không có user hợp lệ phải trả None"
    ok.append("không user nào có GT -> trả None, không ZeroDivisionError")

    # ── 8. nhiều k cùng lúc phải nhất quán (monotonic theo k) ─────────────────
    r = ranking_metrics_at_k(torch.tensor([[9, 9, 7, 9, 9]]), [{7}], ks=[1, 3, 5])
    assert r["hit@1"] == 0.0 and r["hit@3"] == 1.0 and r["hit@5"] == 1.0
    assert r["recall@1"] <= r["recall@3"] <= r["recall@5"], "Recall phải không giảm theo k"
    ok.append("nhiều k cùng lúc: Hit/Recall không giảm theo k")

    # ── 9. format_ranking_report không crash ở cả 2 nhánh ─────────────────────
    assert "n_users=2" in format_ranking_report(
        ranking_metrics_at_k(topk_m, [{0}, {1}], ks=[3]))
    assert "không có user" in format_ranking_report({"n_users": 0})
    ok.append("format_ranking_report chạy cả khi rỗng")

    for line in ok:
        print(f"  {line}  OK")
    print(f"ALL PASS ({len(ok)} nhóm assert)")


if __name__ == "__main__":
    _self_check()
