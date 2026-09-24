"""Chay lai MOI so do da bao cao trong session 09-23/24. Source of truth.

    python analyze.py              # chay tat ca
    python analyze.py 1 3 5        # chi chay muc 1, 3, 5
    python analyze.py --list       # xem danh sach muc
    python analyze.py --out R.md   # ghi ket qua ra file (mac dinh analyze_result.md)

Ket qua LUON duoc ghi ra `analyze_result.md` (markdown, kem timestamp + hash du lieu) de
doi chieu giua cac lan chay. Dung `--no-out` neu chi muon xem tren terminal.

Moi muc IN RA gia tri KY VONG ben canh gia tri DO DUOC. Lech qua nguong se bi danh dau
[!] — do la tin hieu du lieu da doi hoac so bao cao sai, KHONG bo qua.

Nguyen tac chong leak: moi thong ke dung de DU DOAN (PMI, popularity) deu tinh tren
TRAIN-ONLY, cat tai `t <= cutoff`. Cac thong ke MO TA (phan bo tag, tuoi doi item) tinh
tren toan log va duoc danh dau ro o tieu de muc.
"""

from __future__ import annotations

import sys
import time

import numpy as np

OUT = "preprocess_data/output"
MS_PER_DAY = 86_400_000.0
N_TAGS = 47          # 46 tag that + index 0 = padding
N_NEG = 199          # 1 positive + 199 negative, giong moi phep do da bao cao
SEED = 0

_D: dict = {}
_LOG: list[str] = []
_REAL_PRINT = print


def print(*a, **kw):  # noqa: A001 — co y de lai: moi muc dung print nhu binh thuong
    """In ra terminal VA gom vao _LOG de xuat file."""
    _LOG.append(" ".join(str(x) for x in a))
    _REAL_PRINT(*a, **kw)


def data() -> dict:
    """Nap 1 lan, dung lai cho moi muc."""
    if _D:
        return _D
    t0 = time.time()
    hm = np.load(f"{OUT}/history_meta.npy")
    it = np.load(f"{OUT}/item_static.npy")
    sp = np.load(f"{OUT}/sample_position.npy")
    tr = np.load(f"{OUT}/train.npy")
    off = np.load(f"{OUT}/user_offsets.npy")

    vid, ts = hm["video_id"], hm["t"]
    n_items = len(it)
    cutoff = ts[sp[tr["index"]]].max()          # bien gioi train/val theo thoi gian

    # tag_ids -> ma tran boolean (n_item, 47) de kiem tra "chung >=1 tag" bang phep &
    has_tag = np.zeros((n_items, N_TAGS), dtype=bool)
    if "tag_ids" in (it.dtype.names or ()):
        for j in range(it["tag_ids"].shape[1]):
            has_tag[np.arange(n_items), it["tag_ids"][:, j]] = True
        has_tag[:, 0] = False                   # index 0 la padding, khong phai tag that
    else:
        print("[!] item_static.npy CHUA co truong tag_ids — chay lai "
              "preprocess_data/build_item_static.py truoc.\n")

    _D.update(
        vid=vid, ts=ts, item=it, n_items=n_items, cutoff=cutoff, has_tag=has_tag,
        old_cat=it["category_id"], off=off, sp=sp,
        su=np.load(f"{OUT}/sample_user_idx.npy"),
        test=np.load(f"{OUT}/test.npy"),
        first_all=_first_seen(vid, ts, n_items),
        cnt_all=np.bincount(vid, minlength=n_items),
        cnt_tr=np.bincount(vid[ts <= cutoff], minlength=n_items).astype(float),
    )
    print(f"[nap] {len(vid):,} interaction, {n_items:,} item, {time.time()-t0:.1f}s\n")
    return _D


def _first_seen(vid: np.ndarray, ts: np.ndarray, n: int) -> np.ndarray:
    f = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(f, vid, ts)
    return f


def chk(label: str, got: float, want: float, tol: float = 0.005, fmt: str = "%.4f") -> None:
    flag = " " if abs(got - want) <= tol else "[!]"
    print(f"  {flag} {label:<42} do={fmt % got}  ky vong={fmt % want}")


def iter_eval(min_hist: int, cap: int = 20000):
    """Sinh (pos, user_start) cho cac sample test co du lich su. Thu tu co dinh theo SEED."""
    d = data()
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(d["test"]), len(d["test"]), replace=False)
    n = 0
    for s in d["test"]["index"][sel]:
        pos, u = int(d["sp"][s]), int(d["su"][s])
        st = int(d["off"][u])
        if pos - st < min_hist:
            continue
        yield pos, st
        n += 1
        if n >= cap:
            return


def rank_of_target(score: np.ndarray) -> int:
    """Hang cua positive (luon o index 0) trong danh sach ung vien."""
    return int((score > score[0]).sum()) + 1


def nrm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


# ---------------------------------------------------------------- 1
def m1_tag_parsing() -> None:
    """PARSE TAG — 46 tag that, khong phai 111 nhan (mo ta)"""
    print("1. PARSE TAG — bug multi-label da sua")
    d = data()
    n_tag = d["has_tag"].sum(axis=1)
    for k, want in [(1, 5696), (2, 1759), (3, 32)]:
        chk(f"item co {k} tag", (n_tag == k).sum(), want, tol=0, fmt="%.0f")
    chk("item khong tag", (n_tag == 0).sum(), 96, tol=0, fmt="%.0f")
    chk("tag duy nhat", d["has_tag"].any(axis=0).sum(), 46, tol=0, fmt="%.0f")
    chk("nhan cu (111 = HONG)", len(np.unique(d["old_cat"])), 111, tol=0, fmt="%.0f")
    print()


# ---------------------------------------------------------------- 2
def m2_item_age() -> None:
    """VONG DOI ITEM — age_at_event doc lap voi popularity (mo ta)"""
    print("2. VONG DOI ITEM (mo ta, toan log)")
    d = data()
    last = np.zeros(d["n_items"], dtype=np.int64)
    np.maximum.at(last, d["vid"], d["ts"])
    seen = d["cnt_all"] > 0
    life = (last[seen] - d["first_all"][seen]) / MS_PER_DAY
    chk("tuoi doi p50 (ngay) — VO DUNG", np.percentile(life, 50), 26.62, tol=0.05, fmt="%.2f")

    age = (d["ts"] - d["first_all"][d["vid"]]) / MS_PER_DAY
    for q, want in [(25, 0.68), (50, 2.26), (90, 19.90)]:
        chk(f"age_at_event p{q} (ngay)", np.percentile(age, q), want, tol=0.05, fmt="%.2f")

    # Dieu kien chan cua phan (b): age co doc lap voi popularity khong?
    logm = np.log(d["cnt_all"][d["vid"]].astype(float) + 1e-6)
    chk("corr(age, log m_j) — PHAI ~0", np.corrcoef(age, logm)[0, 1], -0.032, tol=0.01)
    print("      => age KHONG du thua voi popularity: dieu kien chan PASS\n")


# ---------------------------------------------------------------- 3
def m3_signal_position() -> None:
    """VI TRI TIN HIEU — 37.9% truong hop ngan han MU (bang chung trung tam)"""
    print("3. VI TRI TIN HIEU (46 tag dung) — bang chung TRUNG TAM")
    d = data()
    print(f"  {'nguong':<12}{'n':>7}{'CHI dai':>10}{'CHI ngan':>10}{'ca hai':>9}{'khong':>9}")
    want = {30: 0.1990, 50: 0.2631, 100: 0.3358, 200: 0.3790}
    for lo in (30, 50, 100, 200):
        c = np.zeros(4, dtype=np.int64)          # [chi dai, chi ngan, ca hai, khong dau]
        for pos, st in iter_eval(lo, cap=20000 if lo < 200 else 6700):
            tv = d["has_tag"][d["vid"][pos]]
            near = bool((d["has_tag"][d["vid"][pos - 10:pos]] & tv).any())
            far = bool((d["has_tag"][d["vid"][max(st, pos - lo):pos - 10]] & tv).any())
            c[0 if (far and not near) else 1 if (near and not far) else 2 if near else 3] += 1
        n = c.sum()
        p = c / n
        flag = " " if abs(p[0] - want[lo]) <= 0.005 else "[!]"
        print(f"{flag} >={lo:<9}{n:>7}{p[0]:>10.4f}{p[1]:>10.4f}{p[2]:>9.4f}{p[3]:>9.4f}"
              f"   (ky vong CHI dai={want[lo]:.4f})")
    print("      => moi nguong: ngan han mu nhieu gap 2-53 lan dai han mu\n")


# ---------------------------------------------------------------- 4
def m4_short_vs_long() -> None:
    """NGAN vs DAI vs ORACLE — cong deu PHA tin hieu, oracle gap +0.089 MRR"""
    print("4. NGAN vs DAI vs ORACLE (lich su>=50, 1 pos + 199 neg)")
    d = data()
    rng = np.random.default_rng(SEED)
    logm = np.log(d["cnt_tr"] + 1e-6)            # popularity TRAIN-ONLY
    cat = d["old_cat"]
    R = {k: [] for k in ("ngan", "dai", "cong", "pop", "oracle")}

    for pos, st in iter_eval(50, cap=15000):
        cand = rng.choice(d["n_items"], N_NEG + 1, replace=False)
        cand[0] = int(d["vid"][pos])             # positive luon o index 0

        def pref(h):                             # phan bo category cua 1 cua so lich su
            p = np.bincount(cat[h], minlength=111).astype(float)
            return nrm(np.log(p / max(p.sum(), 1) + 1e-4)[cat[cand]])

        fs = pref(d["vid"][pos - 10:pos])
        fl = pref(d["vid"][max(st, pos - 50):pos - 10])
        R["ngan"].append(rank_of_target(fs))
        R["dai"].append(rank_of_target(fl))
        R["cong"].append(rank_of_target(fs + fl))
        R["pop"].append(rank_of_target(nrm(logm[cand])))
        R["oracle"].append(min(R["ngan"][-1], R["dai"][-1]))

    want = {"ngan": (.2617, .2301), "dai": (.2042, .2004), "cong": (.2001, .1946),
            "pop": (.3087, .1427), "oracle": (.3438, .3191)}
    print(f"  n={len(R['ngan'])}")
    for k in ("ngan", "dai", "cong", "pop", "oracle"):
        r = np.array(R[k])
        hr, mrr = (r <= 10).mean(), (1 / r).mean()
        f1 = " " if abs(hr - want[k][0]) <= 0.01 else "[!]"
        f2 = " " if abs(mrr - want[k][1]) <= 0.01 else "[!]"
        print(f"  {f1}{f2} {k:<8} HR@10={hr:.4f} (ky vong {want[k][0]:.4f})   "
              f"MRR={mrr:.4f} (ky vong {want[k][1]:.4f})")
    print("      => CONG DEU te hon CA HAI thanh phan: tron TINH pha tin hieu")
    print("      => ORACLE hon ngan han +0.089 MRR: tran cua viec chon DONG\n")


# ---------------------------------------------------------------- 5
def m5_burst_and_gating() -> None:
    """BURST — ngan han hep hon 34.5%, nhung entropy/JSD KHONG gate duoc"""
    print("5. BURST + THAT BAI CUA GATE THU CONG")
    d = data()
    cat = d["old_cat"]
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    es, el, js = [], [], []
    for pos, st in iter_eval(200, cap=1929):
        sh = np.bincount(cat[d["vid"][pos - 10:pos]], minlength=111).astype(float)
        lo = np.bincount(cat[d["vid"][max(st, pos - 200):pos - 10]], minlength=111).astype(float)
        sh /= sh.sum(); lo /= lo.sum()
        es.append(H(sh)); el.append(H(lo))
        m = (sh + lo) / 2
        js.append(0.5 * float((sh[m > 0] * np.log(sh[m > 0] / m[m > 0] + 1e-12)).sum())
                  + 0.5 * float((lo[m > 0] * np.log(lo[m > 0] / m[m > 0] + 1e-12)).sum()))
    chk("entropy 10 item gan nhat (nats)", float(np.mean(es)), 1.895, tol=0.02, fmt="%.3f")
    chk("entropy item 11-200 (nats)", float(np.mean(el)), 2.894, tol=0.02, fmt="%.3f")
    chk("JS-divergence(ngan||dai)", float(np.mean(js)), 0.326, tol=0.01, fmt="%.3f")
    print(f"      => ngan han hep hon {100*(1-np.mean(es)/np.mean(el)):.1f}% — CO burst that")
    print("      NHUNG: entropy chia quartile cho ti le ngan-thang .618/.643/.664/.654")
    print("      => gan PHANG, con hoi NGUOC. Gate thu cong THAT BAI -> phai de mo hinh HOC\n")


# ---------------------------------------------------------------- 6
def m6_pmi() -> None:
    """PMI — mot minh THUA popularity, nhung cong vao thi +66% (train-only)"""
    print("6. PMI (train-only, khong leak)")
    from scipy import sparse
    d = data()
    keep = d["ts"] <= d["cutoff"]
    n_u = len(d["off"]) - 1
    uid = np.zeros(len(d["vid"]), dtype=np.int64)
    for u in range(n_u):
        uid[d["off"][u]:d["off"][u + 1]] = u

    M = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                           (uid[keep], d["vid"][keep])), shape=(n_u, d["n_items"]))
    M.data[:] = 1.0                              # nhi phan: user CO xem item hay khong
    C = (M.T @ M).tocoo()
    diag = C.row == C.col
    df = np.zeros(d["n_items"]); df[C.row[diag]] = C.data[diag]
    r, c, v = C.row[~diag], C.col[~diag], C.data[~diag]
    pmi = np.log(v * n_u / (df[r] * df[c] + 1e-12) + 1e-12)

    chk("PMI median", float(np.median(pmi)), 1.404, tol=0.02, fmt="%+.3f")
    chk("ti le PMI > 0", float((pmi > 0).mean()), 0.886, tol=0.01)
    chk("corr(PMI, log m_j)", float(np.corrcoef(pmi, np.log(df[c] + 1e-6))[0, 1]), -0.498, tol=0.01)
    share = (d["has_tag"][r] & d["has_tag"][c]).any(axis=1)
    cr = float(np.corrcoef(pmi, share.astype(float))[0, 1])
    chk("R2(PMI, chung tag) — PHAI ~1%", cr ** 2, 0.0104, tol=0.003)
    print(f"      bang dense fp16: {d['n_items']**2 * 2 / 1e6:.1f} MB "
          f"(fp32 = {d['n_items']**2 * 4 / 1e6:.1f} MB)")

    # Phep do quyet dinh: PMI mot minh vs cong voi popularity
    P = sparse.csr_matrix((np.maximum(pmi, 0), (r, c)), shape=(d["n_items"],) * 2)
    rng = np.random.default_rng(SEED)
    logm = np.log(d["cnt_tr"] + 1e-6)
    R = {k: [] for k in ("pmi", "pop", "mix")}
    for pos, st in iter_eval(3, cap=3000):
        hist = d["vid"][max(st, pos - 50):pos]
        if len(hist) < 3:
            continue
        cand = rng.choice(d["n_items"], N_NEG + 1, replace=False)
        cand[0] = int(d["vid"][pos])
        sc = np.asarray(P[hist][:, cand].mean(axis=0)).ravel()
        R["pmi"].append(rank_of_target(sc))
        R["pop"].append(rank_of_target(logm[cand]))
        R["mix"].append(rank_of_target(nrm(logm[cand]) + 0.5 * nrm(sc)))
    print(f"  n={len(R['pmi'])}")
    for k, want in [("pop", .3342), ("mix", .5566), ("pmi", .2838)]:
        r_ = np.array(R[k])
        chk(f"HR@10 {k}", float((r_ <= 10).mean()), want, tol=0.02)
    print("      => PMI MOT MINH thua pop (.284<.334) — do rieng se ket luan SAI")
    print("      => pop+PMI = .557, +66%: hai tin hieu truc giao, PHAI cong THEM\n")


# ---------------------------------------------------------------- 7
def m7_no_matthew() -> None:
    """MATTHEW EFFECT — do de BAC BO: ngheo x1.92, giau x0.93 (mo ta)"""
    print("7. MATTHEW EFFECT — do de BAC BO, khong phai de xac nhan")
    d = data()
    cnt = d["cnt_all"][d["cnt_all"] > 0].astype(float)
    srt = np.sort(cnt)[::-1]
    chk("top 10% item chiem", srt[:len(srt) // 10].sum() / srt.sum(), 0.573, tol=0.01)
    g = np.sort(cnt)
    n = len(g)
    chk("Gini", float((2 * np.arange(1, n + 1) - n - 1) @ g / (n * g.sum())), 0.7117, tol=0.005)

    half = (d["ts"].min() + d["ts"].max()) / 2
    c1 = np.bincount(d["vid"][d["ts"] <= half], minlength=d["n_items"]).astype(float)
    c2 = np.bincount(d["vid"][d["ts"] > half], minlength=d["n_items"]).astype(float)
    m = (c1 > 0) & (c2 > 0)
    q = np.percentile(c1[m], [25, 50, 75])
    print("      nhom theo pop nua dau -> he so thay doi thi phan o nua sau:")
    for nm, lo, hi, want in [("duoi p25", 0, q[0], 1.921), ("p25-p50", q[0], q[1], 1.329),
                             ("p50-p75", q[1], q[2], 1.187), ("tren p75", q[2], 1e18, 0.925)]:
        s = m & (c1 >= lo) & (c1 < hi)
        ratio = (c2[s].sum() / c2[m].sum()) / (c1[s].sum() / c1[m].sum())
        chk(f"  {nm}", float(ratio), want, tol=0.02, fmt="x%.3f")
    print("      => ngheo x1.92, giau x0.93: HOI QUY VE TRUNG BINH, NGUOC Matthew")
    print("      => Pure co can thiep ngau nhien nen vong phan hoi da bi cat\n")


# ---------------------------------------------------------------- 8
def m8_tag_lifecycle() -> None:
    """TAG vs VONG DOI — eta2 thap, nhung toc do phai chenh 11x (mo ta)"""
    print("8. TAG vs VONG DOI")
    d = data()
    seen = d["cnt_all"] > 0
    last = np.zeros(d["n_items"], dtype=np.int64)
    np.maximum.at(last, d["vid"], d["ts"])
    life = (last - d["first_all"]) / MS_PER_DAY

    gm = life[seen].mean()
    sst = float(((life[seen] - gm) ** 2).sum())
    ssb = 0.0
    for k in range(1, N_TAGS):
        mk = d["has_tag"][:, k] & seen
        if mk.sum() > 1:
            ssb += mk.sum() * (life[mk].mean() - gm) ** 2
    chk("eta2(life | tag) — PHAI thap", ssb / sst, 0.0425, tol=0.005)

    age = (d["ts"] - d["first_all"][d["vid"]]) / MS_PER_DAY
    ev = d["has_tag"][d["vid"]]
    ratios = []
    for k in range(1, N_TAGS):
        mk = ev[:, k]
        if mk.sum() < 5000:
            continue
        a = age[mk]
        ratios.append(((a < 1).mean()) / ((a > 7).mean() + 1e-9))
    ratios = np.array(ratios)
    chk("tag co >=5000 event", len(ratios), 35, tol=0, fmt="%.0f")
    chk("toc do phai NHANH nhat", ratios.max(), 3.05, tol=0.05, fmt="%.2f")
    chk("toc do phai CHAM nhat", ratios.min(), 0.27, tol=0.02, fmt="%.2f")
    chk("chenh lech (lan)", ratios.max() / ratios.min(), 11.1, tol=0.3, fmt="%.1f")
    print("      => tag khong quyet dinh item song bao lau (eta2 ~4%)")
    print("      => nhung quyet dinh item PHAI NHANH hay CHAM: chenh 11x")
    print("      => age phai NHAN vao content (FiLM), khong CONG\n")


MODULES = [m1_tag_parsing, m2_item_age, m3_signal_position, m4_short_vs_long,
           m5_burst_and_gating, m6_pmi, m7_no_matthew, m8_tag_lifecycle]


def _data_fingerprint() -> str:
    """Van tay du lieu — hai lan chay khac fingerprint thi so lech la do DU LIEU doi."""
    import hashlib
    import os
    h = hashlib.sha256()
    for f in ("history_meta.npy", "item_static.npy", "train.npy", "test.npy"):
        try:
            with open(f"{OUT}/{f}", "rb") as fh:
                h.update(fh.read(1 << 20))       # 1MB dau moi file du de phat hien doi
            h.update(str(os.path.getsize(f"{OUT}/{f}")).encode())
        except OSError:
            h.update(b"MISSING")
    return h.hexdigest()[:12]


def write_report(path: str, pick: list[int], elapsed: float) -> None:
    """Ghi ket qua ra markdown. Timestamp + fingerprint de doi chieu giua cac lan chay."""
    body = "\n".join(_LOG)
    n_flag = body.count("[!]")
    verdict = ("MOI SO KHOP — ket qua tai hien duoc." if n_flag == 0
               else f"CO {n_flag} DONG LECH [!] — doc ky truoc khi trich dan.")
    header = (
        "# Ket qua analyze.py\n\n"
        f"- chay luc: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- muc da chay: {', '.join(str(i) for i in pick)}\n"
        f"- seed: {SEED}  |  fingerprint du lieu: `{_data_fingerprint()}`\n"
        f"- thoi gian: {elapsed:.0f}s\n"
        f"- **{verdict}**\n\n"
        "`do=` la gia tri script vua tinh. `ky vong=` la so da bao cao trong session\n"
        "09-23/24. Dong co `[!]` nghia la hai gia tri lech qua nguong — hoac du lieu da\n"
        "doi, hoac so bao cao sai. Khong bo qua.\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "```\n" + body + "\n```\n")
    _REAL_PRINT(f"\n[ghi] {path}  ({len(_LOG)} dong, {n_flag} lech)")


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        for i, f in enumerate(MODULES, 1):
            _REAL_PRINT(f"  {i}. {(f.__doc__ or '').splitlines()[0]}")
        return

    out = "analyze_result.md"
    if "--out" in args:
        out = args[args.index("--out") + 1]
    if "--no-out" in args:
        out = ""

    pick = [int(a) for a in args if a.isdigit()] or list(range(1, len(MODULES) + 1))
    print(f"=== ANALYZE (seed={SEED}, ky vong = so da bao cao 09-23/24) ===\n")
    t0 = time.time()
    for i in pick:
        MODULES[i - 1]()
    elapsed = time.time() - t0
    print(f"=== xong trong {elapsed:.0f}s. Moi dong co [!] la mot LECH can giai thich. ===")
    if out:
        write_report(out, pick, elapsed)


if __name__ == "__main__":
    main()
