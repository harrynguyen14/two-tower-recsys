# Probes — ba phep do AM TINH (2026-09-28)

Chay tu `gen-recsys/`: `python probes/<file>.py`. Ket luan: memory
`gen-recsys-truc-khoang-cach-can`.

| file | do gi | ket qua |
|---|---|---|
| `probe_delta_pairwise.py` | delta(x_q,x_p) dieu bien khoang cach theo tag, quet 5 gia tri | 0.2111 < delta tinh 0.2149 < KHONG dung khoang cach 0.2219 |
| `probe_gate_auc.py` | bit gate "cua so nao tot hon" co hoc duoc khong (18 dac trung, GBDT) | AUC 0.5275 vs 0.5 doan bua; gate hoc duoc MRR 0.1602 vs khong gate 0.1605 |

O `max` (OR) va `nhan` (AND) nam trong `analyze.py` muc 4: max=0.2543, nhan=0.1993,
so voi cong deu 0.1997 / ngan han 0.2627 / oracle 0.3449.

**oracle = min(rank_ngan, rank_dai) la TRAN GIA** — no xem dap an roi moi chon.
