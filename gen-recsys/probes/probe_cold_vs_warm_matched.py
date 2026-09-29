"""Cold user CAO HON warm — la do KHO hon, hay do CHAM O DAU CHUOI?

Nghi phan #2 (neu tu 2026-09-23, CHUA ai kiem): cold_user duoc cham o vi tri dau chuoi
(du doan tuong tac thu 2-3) con warm_user o cuoi chuoi (thu 500). Neu dung thi hai nhom
KHONG SO SANH DUOC va moi so cold_user da bao cao khong co nghia nhu ten goi.

Phep kiem: so cold vs warm KHOP THEO n_u (so tuong tac tai thoi diem du doan).
Chay tu gen-recsys/: python probes/probe_cold_vs_warm_matched.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, analyze as A

d = A.data()
OUT = A.OUT
test = d["test"]
sp, su, off = d["sp"], d["su"], d["off"]
ts = d["ts"]

# --- dung lai DUNG dinh nghia cua build_interactions.py ---
n_users = len(off) - 1 if len(off) > int(su.max()) + 1 else int(su.max()) + 1
first_seen = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
for u in range(n_users):
    a, b = int(off[u]), int(off[u + 1]) if u + 1 < len(off) else len(ts)
    if b > a:
        first_seen[u] = ts[a]
seen = first_seen[first_seen < np.iinfo(np.int64).max]
p80 = np.percentile(seen, 80)
is_cold_user = first_seen > p80                       # strict holdout, theo USER

idx = test["index"]
pos, u = sp[idx], su[idx]
n_u = pos - off[u]                                    # so tuong tac TAI thoi diem du doan
cold = is_cold_user[u]

print(f"\nn_test={len(idx):,}  cold={cold.sum():,} ({cold.mean()*100:.2f}%)  warm={(~cold).sum():,}")
print(f"p80 first_seen = {p80:.0f}\n")

# ============ (1) n_u hai nhom co khac nhau khong? ============
print("(1) PHAN BO n_u — neu khac nhau thi hai nhom cham o VI TRI CHUOI khac nhau")
for name, m in (("cold", cold), ("warm", ~cold)):
    v = n_u[m]
    print(f"  {name:5} n={m.sum():7,}  median={np.median(v):6.0f}  "
          f"p25={np.percentile(v,25):6.0f}  p75={np.percentile(v,75):6.0f}  mean={v.mean():7.1f}")

# ============ (2) KHOP n_u roi so lai ============
# eval that dung model; o day dung mot proxy KHONG CAN GPU: do kho do duoc bang
# popularity rank cua item dich (pop la baseline manh nhat, HR@10=0.3085 o m4).
logm = np.log(d["cnt_tr"] + 1e-6)
tgt = d["vid"][pos]
pop_pct = (logm[tgt] - logm.min()) / (logm.max() - logm.min())   # 1 = sieu pho bien

print("\n(2) DO KHO NOI DUNG (popularity cua item dich; CAO = de hon)")
for name, m in (("cold", cold), ("warm", ~cold)):
    print(f"  {name:5} pop_pct median={np.median(pop_pct[m]):.4f}  mean={pop_pct[m].mean():.4f}")

print("\n(3) KHOP THEO n_u — so cold vs warm TRONG tung tang n_u")
bins = [(0,5),(6,20),(21,50),(51,100),(101,300),(301,10**9)]
print(f"  {'tang n_u':<12} {'n cold':>7} {'n warm':>8} {'pop cold':>9} {'pop warm':>9}  ghi chu")
tot_c = tot_w = 0
for lo, hi in bins:
    m = (n_u >= lo) & (n_u <= hi)
    c, w = m & cold, m & ~cold
    tot_c += c.sum(); tot_w += w.sum()
    pc = np.median(pop_pct[c]) if c.sum() else float("nan")
    pw = np.median(pop_pct[w]) if w.sum() else float("nan")
    note = "cold BIEN MAT" if c.sum() < 20 else ""
    print(f"  {f'{lo}-{hi if hi<10**9 else chr(43)}':<12} {c.sum():>7,} {w.sum():>8,} "
          f"{pc:>9.4f} {pw:>9.4f}  {note}")
print(f"  {'TONG':<12} {tot_c:>7,} {tot_w:>8,}")

# ============ (4) ket luan dinh luong ============
print("\n(4) KET LUAN")
frac_cold_low = (n_u[cold] <= 20).mean()
frac_warm_low = (n_u[~cold] <= 20).mean()
print(f"  ti le sample co n_u<=20:   cold={frac_cold_low:.4f}   warm={frac_warm_low:.4f}"
      f"   -> lech {frac_cold_low/max(frac_warm_low,1e-9):.1f}x")
overlap = ((n_u >= np.percentile(n_u[cold],25)) & (n_u <= np.percentile(n_u[cold],75)))
print(f"  warm nam trong khoang IQR cua cold: {(overlap & ~cold).sum():,} / {(~cold).sum():,}"
      f" = {(overlap & ~cold).sum()/(~cold).sum():.4f}")
