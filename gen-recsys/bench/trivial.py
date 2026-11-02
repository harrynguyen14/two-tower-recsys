"""Reference rows for the table: popularity and 'repeat last item', through the SAME protocol.

  python trivial.py --eval-split val
popularity = train label counts (no val/test leakage). repeat = last history item first, popularity breaks ties.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import protocol

p = argparse.ArgumentParser()
p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "preprocess_data" / "output"))
p.add_argument("--eval-split", default="val", choices=["val", "test"])
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--eval-max-batches", type=int, default=None)
a = p.parse_args()

train = protocol.make_dataset(a.output_dir, "train")
ev = protocol.make_dataset(a.output_dir, a.eval_split)
V = int(train.item_static.shape[0])
labels = train.history_meta["video_id"][np.asarray(train.sample_position)[train.split_data["index"]]]
pop = torch.tensor(np.bincount(labels, minlength=V), dtype=torch.float32)
pop = pop / pop.sum()

for name in ("popularity", "repeat_last"):
    def score(batch, name=name):
        s = pop.expand(batch["label_video_id"].shape[0], -1).clone()
        if name == "repeat_last":
            s.scatter_add_(1, batch["hist_video_ids"][:, -1:].cpu(), torch.full((s.shape[0], 1), 10.0))
        return s.to(batch["label_video_id"].device)
    print(f"##### {name}")
    res = protocol.evaluate(score, ev, batch_size=a.batch_size, max_batches=a.eval_max_batches)
    print(f"model={name} split={a.eval_split} " + " ".join(f"{k}={float(v):.4f}" for k, v in res.items()))
