import sys; sys.path.insert(0, ".")
import numpy as np, analyze as A

d = A.data()
rng = np.random.default_rng(A.SEED)
has_tag, cat = d["has_tag"], d["old_cat"]
N_NEG = A.N_NEG

GRID = [0.0, -0.25, -0.5, -1.0, -2.0]
keys = [f"pair_{abs(g)}" for g in GRID] + ["static_0.5", "static_1.0", "ngan", "dai", "oracle"]
R = {k: [] for k in keys}

for pos, st in A.iter_eval(50, cap=15000):
    cand = rng.choice(d["n_items"], N_NEG + 1, replace=False)
    cand[0] = int(d["vid"][pos])
    hist = d["vid"][max(st, pos - 50):pos]
    if len(hist) == 0:
        continue
    log_rel = np.log1p(np.arange(len(hist), 0, -1))          # (L,) 1=gan nhat

    # share[p,c]: token p chung >=1 tag voi candidate c  -> DIEM THO (proxy q.k)
    share = ((has_tag[hist] @ has_tag[cand].T) > 0).astype(np.float64)   # (L,C)
    # diem tho co ca phan am: chung tag = +1, khong chung = -1 (de token khong lien quan
    # con duong anh huong, nen delta moi co cho tac dong)
    base = 2.0 * share - 1.0

    for g in GRID:
        # delta(x_q,x_p) = 0 neu chung tag, g neu khong -> DIEU BIEN THEO NOI DUNG
        w = np.exp(np.where(share > 0, 0.0, g) * log_rel[:, None])       # (L,C)
        R[f"pair_{abs(g)}"].append(A.rank_of_target(A.nrm((base * w).sum(0))))

    for s in (0.5, 1.0):
        # delta TINH: cung mot he so cho moi token, khong biet tag
        w0 = np.exp(-s * log_rel)[:, None]
        R[f"static_{s}"].append(A.rank_of_target(A.nrm((base * w0).sum(0))))

    def pref(h):
        p = np.bincount(cat[h], minlength=111).astype(float)
        return A.nrm(np.log(p / max(p.sum(), 1) + 1e-4)[cat[cand]])
    fs = pref(d["vid"][pos - 10:pos]); fl = pref(d["vid"][max(st, pos - 50):pos - 10])
    R["ngan"].append(A.rank_of_target(fs)); R["dai"].append(A.rank_of_target(fl))
    R["oracle"].append(min(R["ngan"][-1], R["dai"][-1]))

print(f"\n  n={len(R['ngan'])}   (moc: cong=0.1997  max=0.2543  ngan=0.2627  oracle=0.3449)\n")
for k in keys:
    r = np.array(R[k]); print(f"  {k:<12} HR@10={(r<=10).mean():.4f}  MRR={(1/r).mean():.4f}")
