import sys; sys.path.insert(0, ".")
import numpy as np, analyze as A
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d = A.data()
rng = np.random.default_rng(A.SEED)
has_tag, cat = d["has_tag"], d["old_cat"]
logm = np.log(d["cnt_tr"] + 1e-6)          # popularity TRAIN-ONLY (chong leak)
N_NEG = A.N_NEG

X, y, r_s, r_l = [], [], [], []

def pref(h, cand):
    p = np.bincount(cat[h], minlength=111).astype(float)
    return A.nrm(np.log(p / max(p.sum(), 1) + 1e-4)[cat[cand]])

def ent(h):                                 # entropy category cua mot cua so
    p = np.bincount(cat[h], minlength=111).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())

for pos, st in A.iter_eval(50, cap=15000):
    cand = rng.choice(d["n_items"], N_NEG + 1, replace=False)
    cand[0] = int(d["vid"][pos])
    hs = d["vid"][pos - 10:pos]                      # cua so NGAN
    hl = d["vid"][max(st, pos - 50):pos - 10]        # cua so DAI
    if len(hs) == 0 or len(hl) == 0:
        continue

    rs = A.rank_of_target(pref(hs, cand))
    rl = A.rank_of_target(pref(hl, cand))
    if rs == rl:                                     # khong phan biet -> bo, tranh nhan nhieu
        continue
    r_s.append(rs); r_l.append(rl)
    y.append(1 if rl < rs else 0)                    # 1 = cua so DAI tot hon

    # ==== DAC TRUNG: chi tu LICH SU, khong nhin target/candidate ====
    ts_s, ts_l = has_tag[hs], has_tag[hl]
    tag_s, tag_l = ts_s.any(0), ts_l.any(0)          # tap tag moi cua so
    inter = float((tag_s & tag_l).sum())
    union = float((tag_s | tag_l).sum())
    X.append([
        ent(hs), ent(hl), ent(hs) - ent(hl),         # burst: §-1(iii) da do la PHANG
        len(hl), pos - st,                            # do dai lich su
        float(tag_s.sum()), float(tag_l.sum()),       # so tag rieng biet moi cua so
        inter / max(union, 1),                        # Jaccard hai cua so
        inter / max(float(tag_s.sum()), 1),           # bao nhieu tag ngan CO trong dai
        float(len(np.unique(cat[hs]))), float(len(np.unique(cat[hl]))),
        float(np.unique(hs).size) / len(hs),          # ti le item khac nhau (repeat rate)
        float(np.unique(hl).size) / len(hl),
        float(logm[hs].mean()), float(logm[hl].mean()),   # do pho bien trung binh
        float(logm[hs].std()), float(logm[hl].std()),
        float(np.isin(hs, hl).mean()),                # item ngan co xuat hien o dai
    ])

X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
n = len(y); cut = int(n * 0.7)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

print(f"\n  n={n}  (bo {15000-n} sample rank bang nhau)   ty le nhan 'DAI tot hon' = {y.mean():.4f}")
print(f"  train={cut} test={n-cut}   so dac trung={X.shape[1]}\n")

for name, clf in (("logistic ", LogisticRegression(max_iter=2000, C=1.0)),
                  ("gbdt     ", GradientBoostingClassifier(random_state=0))):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf.fit((Xtr - mu) / sd, ytr)
    p = clf.predict_proba((Xte - mu) / sd)[:, 1]
    print(f"  {name} AUC={roc_auc_score(yte, p):.4f}   acc={( (p>0.5).astype(int)==yte ).mean():.4f}")

print(f"\n  moc: doan bua AUC=0.5000, acc={max(y.mean(),1-y.mean()):.4f} (chon nhan da so)")

# Tran thuc te: gate hoan hao vs gate hoc duoc, do bang MRR tren TEST
rs, rl = np.array(r_s[cut:]), np.array(r_l[cut:])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
clf = GradientBoostingClassifier(random_state=0).fit((Xtr - mu) / sd, ytr)
pick_l = clf.predict_proba((Xte - mu) / sd)[:, 1] > 0.5
print(f"\n  --- MRR tren cung tap test ---")
print(f"  chi ngan han        {(1/rs).mean():.4f}")
print(f"  chi dai han         {(1/rl).mean():.4f}")
print(f"  gate HOC DUOC       {(1/np.where(pick_l, rl, rs)).mean():.4f}")
print(f"  gate HOAN HAO       {(1/np.minimum(rs, rl)).mean():.4f}  <- oracle")
