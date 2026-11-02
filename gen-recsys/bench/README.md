# gen-recsys/bench -- baselines on the same protocol

Baselines (SASRec, GRU4Rec, HSTU) scored with the **same samples and the same metric code**
as `gen-recsys` (KuaiRand-Pure, 7,583 items, global temporal split).

| file | role |
|---|---|
| `protocol.py` | one evaluation: full ranking over ALL items, no seen-item filtering, tie-safe rank, cold/warm/n_u-matched slices. Imports `dataset.py` / `eval.py` from gen-recsys (`GEN_RECSYS_DIR`) — never reimplement the metrics. |
| `models.py` | SASRec, GRU4Rec, HSTU (torch-rechub `HSTUModel`) with the authors' published recipes, see below. |
| `run.py` | train one baseline, then evaluate; prints a `===== TOM TAT =====` block |
| `ours.py` | score a gen-recsys checkpoint through the same `protocol.evaluate` |

```bash
pip install torch numpy pytest
pip install --no-deps torch-rechub==0.9.0        # HSTUModel only needs torch
python run.py --model sasrec  --steps 6000 --device cuda --seed 0
python run.py --model gru4rec --steps 6000 --device cuda --seed 0
python run.py --model hstu    --steps 6000 --device cuda --seed 0
python ours.py --ckpt ckpt_C2.pt --eval-split val --device cuda
python -m pytest tests -q
```

Notes
- `--eval-split test` only for the final number; tune on `val`.
- Item `j` is token `j+1` (0 = PAD). HSTU `time_diffs[i]` = `label_time - ts[i]` in seconds
  (one shared anchor, as torch-rechub's rab^t requires; a test checks the pairwise property), 0 on padding.
- 2.7% of val labels equal the last history item, so any model that just repeats it scores hr@10 ~0.027.

## Recipes (fetched from the authors' repos, 2026-10-31)
| model | source | recipe |
|---|---|---|
| HSTU | facebookresearch/generative-recommenders `configs/ml-1m/hstu-sampled-softmax-n128-final.gin` | d=50, 2 blocks, 1 head, dqk=dv=50, dropout 0.2, AdamW lr 1e-3 wd 0, constant lr, sampled softmax with 128 negatives, temperature 0.05, user+item l2-normalised, rab^{p,t} only (no input time embedding) |
| SASRec | same repo `sasrec-sampled-softmax-n128-final.gin` (the HSTU paper trains its SASRec baseline with the SAME recipe) | d=50, 2 blocks, 1 head, ffn 50, dropout 0.2, rest identical to HSTU |
| GRU4Rec | hidasib/GRU4Rec `paramfiles/rees46_xe_shared_best.py` | 1 GRU layer 512, tied (constrained) embedding, embed dropout 0.45, Adagrad lr 0.065, cross-entropy over 1024 in-batch targets + 2048 items ~ popularity^0.5, logQ 1.0 |

Original-paper variants stay selectable: `--loss bce` (SASRec 2018), `--loss bprmax` (GRU4Rec 2018).

Deviations from the recipes (deliberate, none verified to be neutral):
- Window = gen-recsys `max_seq_len` 256 (Meta: 200) so every model sees the same history.
- Training is step-based, not epoch-based. `ssm` negatives are uniform over the catalog = Meta's `sampling_strategy=local`
  (checked in `autoregressive_losses.py`), except that I share 128 negatives across the batch while Meta draws 128 per position.
  GRU4Rec in-batch negatives are a random subset of 1024 batch targets, not the session-parallel batch.
- Initialisation follows the authors (SASRec/HSTU: embeddings trunc-normal std 0.02, other matrices xavier; GRU4Rec: Glorot-uniform).
  The first Kaggle run used PyTorch's default N(0,1) embeddings and SASRec/GRU4Rec stayed at popularity level (hr@10 0.024/0.027); discard it.
- `trivial.py` gives the reference rows (popularity, repeat-last-item) through the same protocol.
- HSTU time buckets = floor(ln(max(|dt|,1 s)) / 0.301), 128 buckets: copied from Meta's `research/modeling/sequential/hstu.py`
  (lines ~611-613). torch-rechub's rab has one bias per head, Meta's is shared across heads (identical with 1 head).
- GRU4Rec has no output bias, and its hyper-parameters come from another dataset (the repo says to tune per dataset;
  run a short Optuna search before quoting it).
- Checked: `ssm` with all items as negatives equals the full-softmax CE exactly (SASRec and HSTU).
- NOT used: torch-rechub's eval example has `filter_seen=True`; that would make numbers
  incomparable with gen-recsys.
- Calibration (done): the same gen-recsys checkpoint gives identical n / hr@10 / mrr
  from `train.py --eval-only --full-ranking` and `ours.py` (smoke checkpoint, 320 val samples).
  Redo it on a trained checkpoint before quoting results.
- Budget: gen-recsys numbers are 6000 steps (0.68 epoch). Give baselines the same `--steps`
  (and also a longer run) before claiming a ranking; seed SD in gen-recsys is ~0.001.
