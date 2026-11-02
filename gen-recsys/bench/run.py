"""Train one baseline with its published recipe (see models.RECIPES) and score it with the shared protocol.

  python run.py --model sasrec  --steps 6000 --device cuda
  python run.py --model gru4rec --steps 6000 --device cuda
  python run.py --model hstu    --steps 6000 --device cuda
Training samples = gen-recsys train split with random sliding windows; every window position
is a next-item training example. Constant learning rate, no warmup, no grad clipping (as in the recipes).
Prints a TOM TAT block.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import models
import protocol

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=list(models.MODELS))
p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "preprocess_data" / "output"))
p.add_argument("--loss", default=None, choices=["ssm", "gru_xe", "ce", "bce", "bprmax"],
               help="default = the model's published recipe")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--dim", type=int, default=None)
p.add_argument("--layers", type=int, default=None)
p.add_argument("--heads", type=int, default=None)
p.add_argument("--dropout", type=float, default=None)
p.add_argument("--n-neg", type=int, default=128, help="ssm negatives (HSTU/SASRec recipe: 128)")
p.add_argument("--sample-alpha", type=float, default=0.5, help="GRU4Rec popularity^alpha negatives")
p.add_argument("--max-seq-len", type=int, default=256)
p.add_argument("--eval-split", default="val", choices=["val", "test"])
p.add_argument("--eval-max-batches", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--num-workers", type=int, default=0)
a = p.parse_args()

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
recipe = models.RECIPES[a.model]
loss_kind = a.loss or recipe["loss"]
train_ds = protocol.make_dataset(a.output_dir, "train", a.max_seq_len, random_window=True, seed=a.seed)
eval_ds = protocol.make_dataset(a.output_dir, a.eval_split, a.max_seq_len)
num_items = int(train_ds.item_static.shape[0])

kw = dict(recipe["model"], max_seq_len=a.max_seq_len)
for k in ("dim", "layers", "heads", "dropout"):
    if getattr(a, k) is not None:
        kw[k] = getattr(a, k)
model = models.build(a.model, num_items, **kw).to(a.device)
lr = a.lr or recipe["lr"]
opt = (torch.optim.Adagrad if recipe["opt"] == "adagrad" else torch.optim.AdamW)(
    model.parameters(), lr=lr, **({} if recipe["opt"] == "adagrad" else {"weight_decay": 0.0}))

# popularity from TRAIN labels only (no val/test leakage), for GRU4Rec's sampled negatives + logQ
idx = train_ds.split_data["index"]
labels = train_ds.history_meta["video_id"][np.asarray(train_ds.sample_position)[idx]]
pop_log_q = models.item_log_q(np.bincount(labels, minlength=num_items), a.sample_alpha)

n_par = sum(x.numel() for x in model.parameters())
print(f"[bench] {a.model} loss={loss_kind} opt={recipe['opt']} lr={lr} params={n_par:,} cfg={kw} "
      f"steps={a.steps} bs={a.batch_size} seed={a.seed}", flush=True)

step, epoch, t0, run = 0, 0, time.time(), []
model.train()
while step < a.steps:
    train_ds.set_epoch(epoch)
    for batch in DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                            num_workers=a.num_workers):
        batch = protocol.to_device(batch, a.device)
        loss = models.loss_fn(model, batch, loss_kind, n_neg=a.n_neg, pop_log_q=pop_log_q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append(loss.item())
        step += 1
        if step % 200 == 0 or step == a.steps:
            print(f"[bench] step {step}/{a.steps} loss {np.mean(run):.4f} "
                  f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
            run = []
        if step >= a.steps:
            break
    epoch += 1

model.eval()
res = protocol.evaluate(lambda b: models.last_scores(model, b), eval_ds, batch_size=a.batch_size,
                        device=a.device, max_batches=a.eval_max_batches, num_workers=a.num_workers)
print("===== TOM TAT =====")
print(f"model={a.model} loss={loss_kind} steps={a.steps} seed={a.seed} split={a.eval_split} "
      + " ".join(f"{k}={float(v):.4f}" for k, v in res.items()))
