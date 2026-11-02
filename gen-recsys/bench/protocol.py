"""Shared protocol for the benchmark (single source of truth for samples + metrics).

Samples and metric code come from the gen-recsys repo (pinned via GEN_RECSYS_DIR), so every
model is scored on EXACTLY the same samples with EXACTLY the same metric code:
  - full ranking over ALL items (no sampled negatives); seen items are NOT filtered
  - the positive logit is swapped into column 0, tie-safe rank (gr_eval.compute_positive_rank)
  - same slices: all / warm_user / cold_user / lowhistory + n_u-matched cold vs warm
Do NOT reimplement the metrics here: any drift makes numbers incomparable with gen-recsys.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

GEN_DIR = Path(os.environ.get("GEN_RECSYS_DIR", Path(__file__).resolve().parents[1] / "model"))
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))
from dataset import GenRecsysDataset  # noqa: E402  (gen-recsys)
import eval as gr_eval  # noqa: E402  (gen-recsys eval.py, NOT the builtin)


def make_dataset(output_dir, split, max_seq_len=256, random_window=False, seed=0,
                 use_caption=False, caption_file="caption_embeddings.npy"):
    """Same window/label logic as gen-recsys. Captions are only needed by our own model."""
    return GenRecsysDataset(Path(output_dir), split=split, max_seq_len=max_seq_len,
                            random_window=random_window, seed=seed, use_caption=use_caption,
                            caption_file=caption_file)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(score_fn, dataset, batch_size=128, device="cpu", max_batches=None, num_workers=0):
    """score_fn(batch) -> (B, num_items) logits; column j is item id j (0-based, identity).

    Prints the same report as gen-recsys train.evaluate (so the `[all ` line can be grepped)
    and returns the overall metric dict {'hr@10': ..., 'mrr': ..., 'n_samples': ...}.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per, cold, lowh, n_u = {}, [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        logit = score_fn(batch)
        lbl = batch["label_video_id"].view(-1, 1)
        pos_col = logit.gather(1, lbl)
        logit = logit.scatter(1, lbl, logit[:, :1])
        logit = torch.cat([pos_col, logit[:, 1:]], dim=1)
        for k, v in gr_eval.compute_metrics_at_k(logit).items():
            per.setdefault(k, []).append(v.cpu())
        cold.append(batch["is_user_cold"].cpu())
        lowh.append(batch["is_user_lowhistory"].cpu())
        n_u.append(batch["hist_n_u"][:, -1].cpu())
    per = {k: torch.cat(v) for k, v in per.items()}
    cold, lowh, n_u = torch.cat(cold), torch.cat(lowh), torch.cat(n_u)
    n = next(iter(per.values())).shape[0]
    overall = gr_eval.aggregate_by_group(per, {"all": torch.ones(n, dtype=torch.bool)})
    cold_r = gr_eval.aggregate_by_group(per, {"warm_user": ~cold, "cold_user": cold, "lowhistory": lowh})
    gr_eval.print_eval_report(overall, result_cold=cold_r)
    gr_eval.print_matched(
        gr_eval.matched_by_n_u(per, cold, ~cold, n_u, name_a="cold", name_b="warm"), expect="b>a")
    return overall["all"]
