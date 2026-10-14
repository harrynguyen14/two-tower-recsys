"""D4a — Popularity baseline (formula.md §4.4 nhom D). KHONG train gi, chi xep hang item
theo N_i (do pho bien) tai DUNG moc train (p80), danh gia tren full catalog.

CONG GAC truoc khi chay bang ablation nhom A: neu model KHONG hon duoc baseline nay,
40h train nhom A la vo nghia (formula.md: "D4a la CONG GAC — chay TRUOC bang 2x2").

Chay (val, tune):
    python popularity_baseline.py --output-dir <path> --eval-split val

Chay (test, DUNG 1 LAN sau khi da chot moi thu tren val):
    python popularity_baseline.py --output-dir <path> --eval-split test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# [SUA] build_n_cumulative.OUT_DIR duoc doc tai IMPORT-time tu env GEN_RECSYS_OUT_DIR
# (xem train.py:25-27, cung pattern). Phai set bien env NAY truoc khi import module do,
# neu khong no se roi ve default "./output" canh file thay vi --output-dir CLI truyen vao.
def _peek_output_dir() -> str | None:
    for i, a in enumerate(sys.argv):
        if a == "--output-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--output-dir="):
            return a.split("=", 1)[1]
    return None


_out = _peek_output_dir()
if _out and not os.environ.get("GEN_RECSYS_OUT_DIR"):
    os.environ["GEN_RECSYS_OUT_DIR"] = str(Path(_out).resolve())

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess_data"))
from build_n_cumulative import build_n_cache, lookup_n_at_t_batch_cached  # noqa: E402

from dataset import GenRecsysDataset  # noqa: E402
from eval import compute_metrics_at_k, aggregate_by_group  # noqa: E402
from train import load_or_compute_p80  # noqa: E402


def popularity_scores(output_dir: Path, p80: np.int64, num_items: int) -> torch.Tensor:
    """N_i cua MOI item TAI moc p80 (khong leak val/test) — xem error.md #9, cung co che
    voi NegativeSampler. Item chua xuat hien truoc p80 nhan N_i=0 (dung — chua pho bien)."""
    item_n_cache = build_n_cache("item_N")
    ids = np.arange(num_items, dtype=np.int64)
    ts = np.full(num_items, p80, dtype=np.int64)
    n_i = lookup_n_at_t_batch_cached(item_n_cache, ids, ts)
    return torch.from_numpy(n_i.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = GenRecsysDataset(output_dir, split=args.eval_split, max_seq_len=args.max_seq_len)
    num_items = dataset.item_static.shape[0]

    p80 = load_or_compute_p80(dataset, output_dir)
    pop_score = popularity_scores(output_dir, p80, num_items)  # (num_items,)

    label_video_ids = []
    for idx in range(len(dataset)):
        row = dataset.split_data[idx]
        sample_idx = int(row["index"])
        user_idx = int(dataset.sample_user_idx[sample_idx])
        position = int(dataset.sample_position[sample_idx])
        label_pos = position  # xem dataset.py: window_end == position cho eval (cua so co dinh)
        label_video_ids.append(int(dataset.history_meta["video_id"][label_pos]))
    label_video_ids = torch.tensor(label_video_ids, dtype=torch.int64)

    # Xep hang TOAN BO catalog bang pop_score, dat positive vao cot 0 (khop quy uoc
    # compute_metrics_at_k) bang cach HOAN VI cot cua label ve dau — giong cach
    # evaluate() lam voi full_ranking trong train.py.
    B = len(label_video_ids)
    logit = pop_score.unsqueeze(0).expand(B, -1).clone()  # (B, num_items)
    pos_col = logit.gather(1, label_video_ids.view(-1, 1))
    logit = logit.scatter(1, label_video_ids.view(-1, 1), logit[:, :1])
    logit = torch.cat([pos_col, logit[:, 1:]], dim=1)

    metrics = compute_metrics_at_k(logit)
    n_all = B
    result = aggregate_by_group(metrics, {"all": torch.ones(n_all, dtype=torch.bool)})

    print(f"=== D4a POPULARITY BASELINE — split={args.eval_split}, n={n_all} ===")
    for name, m in result.items():
        parts = [f"{k}={v:.4f}" for k, v in m.items() if k != "n_samples"]
        print(f"  [{name}] n={m['n_samples']} " + " ".join(parts))


if __name__ == "__main__":
    main()
