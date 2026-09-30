"""Nhom `both` ket (0.0491) — la NGHEN OUTPUT, hay chi la CONFOUND vi tri chuoi + do kho item?

Nghi phan: `both = near & far` doi hoi tag trung o CA HAI vung. `far` chi ton tai khi chuoi
dai hon SHORT_WINDOW=10, va cang nhieu item thi cang de trung => `both` THIEN VE user chuoi
DAI theo cau truc, khong phai theo so thich. Day dung la confound da giet ket luan cold>warm
(94% nghich ly do vi tri chuoi) va "lich su dai hai".

Neu `both` chi la "chuoi dai + item pho bien thap" thi gia thuyet nghen output SAI DE, va ca
huong multi-vector la sai dia chi.

Chay tu gen-recsys/: python probes/probe_both_confound.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, analyze as A

SHORT_WINDOW = 10          # khop eval.py:15
d = A.data()
sp, su, off, ts, vid = d["sp"], d["su"], d["off"], d["ts"], d["vid"]
has_tag = d["has_tag"]
idx = d["test"]["index"]
pos, u = sp[idx], su[idx]
start = off[u]
n_u = pos - start                                  # so tuong tac TAI thoi diem du doan

# --- phan nhom Y HET eval.classify_signal_position, nhung tren toan bo test ---
# eval.py cat lich su thanh K item cuoi; o day dung toan bo tien su de khong them confound moi.
near = np.zeros(len(idx), dtype=bool)
far = np.zeros(len(idx), dtype=bool)
for i in range(len(idx)):
    p, st = int(pos[i]), int(start[i])
    tv = has_tag[vid[p]]
    lo_near = max(st, p - SHORT_WINDOW)
    near[i] = bool((has_tag[vid[lo_near:p]] & tv).any()) if p > lo_near else False
    far[i] = bool((has_tag[vid[st:lo_near]] & tv).any()) if lo_near > st else False

G = {"only_long": far & ~near, "only_short": near & ~far,
     "both": near & far, "neither": ~near & ~far}

logm = np.log(d["cnt_tr"] + 1e-6)
tgt = vid[pos]
pop_pct = (logm[tgt] - logm.min()) / (logm.max() - logm.min())   # 1 = sieu pho bien

print(f"\nn_test={len(idx):,}  SHORT_WINDOW={SHORT_WINDOW}\n")

# ============ (1) n_u cua 4 nhom — neu lech thi KHONG so sanh duoc ============
print("(1) VI TRI CHUOI (n_u) theo nhom — day la confound so 1")
print(f"  {'nhom':<12}{'n':>9}{'%':>7}{'median':>9}{'p25':>8}{'p75':>8}{'mean':>9}")
for name in ("only_short", "both", "neither", "only_long"):
    m = G[name]; v = n_u[m]
    print(f"  {name:<12}{m.sum():>9,}{m.mean()*100:>6.1f}%{np.median(v):>9.0f}"
          f"{np.percentile(v,25):>8.0f}{np.percentile(v,75):>8.0f}{v.mean():>9.1f}")

# ============ (2) do kho item dich — confound so 2 ============
print("\n(2) DO KHO ITEM DICH (pop_pct; CAO = de hon)")
for name in ("only_short", "both", "neither", "only_long"):
    m = G[name]
    print(f"  {name:<12} median={np.median(pop_pct[m]):.4f}  mean={pop_pct[m].mean():.4f}")

# ============ (3) KHOP n_u: trong CUNG tang n_u, `both` con khac `only_short` khong? ============
print("\n(3) KHOP THEO n_u — cot then chot. Trong cung tang, neu pop cua `both` va")
print("    `only_short` GIONG nhau thi hai nhom chi khac nhau o NHAN, khong o do kho.")
bins = [(0,10),(11,20),(21,50),(51,100),(101,300),(301,10**9)]
hdr = f"  {'tang n_u':<11}" + "".join(f"{n[:9]:>11}" for n in ("only_short","both","neither","only_long"))
print(hdr)
for lo, hi in bins:
    mb = (n_u >= lo) & (n_u <= hi)
    row = f"  {f'{lo}-{hi if hi<10**9 else chr(43)}':<11}"
    for name in ("only_short", "both", "neither", "only_long"):
        m = mb & G[name]
        row += f"{m.sum():>11,}" if m.sum() else f"{'-':>11}"
    print(row)
print("  --- pop_pct median trong tung tang ---")
print(hdr)
for lo, hi in bins:
    mb = (n_u >= lo) & (n_u <= hi)
    row = f"  {f'{lo}-{hi if hi<10**9 else chr(43)}':<11}"
    for name in ("only_short", "both", "neither", "only_long"):
        m = mb & G[name]
        row += f"{np.median(pop_pct[m]):>11.4f}" if m.sum() >= 20 else f"{'-':>11}"
    print(row)

# ============ (4) `both` co the nao KHAC ngoai chuoi dai? ============
print("\n(4) KET LUAN DINH LUONG")
ns, nb = n_u[G["only_short"]], n_u[G["both"]]
print(f"  median n_u: only_short={np.median(ns):.0f}  both={np.median(nb):.0f}"
      f"  -> both dai gap {np.median(nb)/max(np.median(ns),1):.1f}x")
frac_short_le10 = (ns <= SHORT_WINDOW).mean()
print(f"  ti le only_short co n_u<={SHORT_WINDOW}: {frac_short_le10:.4f}"
      f"   (n_u<={SHORT_WINDOW} thi vung `far` RONG => KHONG THE vao `both`)")
imposs = (n_u <= SHORT_WINDOW).mean()
print(f"  ti le TOAN test co n_u<={SHORT_WINDOW} (bi loai khoi `both` theo CAU TRUC): {imposs:.4f}")
# `both` chi thuc su khac `only_short` neu trong CUNG tang n_u van con lech do kho
tang = (n_u >= 101) & (n_u <= 300)
a, b = tang & G["only_short"], tang & G["both"]
if a.sum() >= 20 and b.sum() >= 20:
    print(f"  trong tang n_u 101-300: only_short n={a.sum():,} pop={np.median(pop_pct[a]):.4f}"
          f" | both n={b.sum():,} pop={np.median(pop_pct[b]):.4f}")
