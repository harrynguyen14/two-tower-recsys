"""Score a gen-recsys checkpoint with the SHARED bench protocol (protocol.evaluate).

  python ours.py --ckpt ckpt_C2.pt --eval-split val --device cuda
Uses gen-recsys' own modules/forward (train.run_batch_forward) only to produce the user vector;
ranking + metrics go through protocol.py, exactly like the baselines. Calibration = this must
print the same hr@10 as `train.py --eval-only --full-ranking` on the same checkpoint.
"""
import argparse
from pathlib import Path

import torch

import protocol  # noqa: F401  (also puts gen-recsys/model on sys.path)
import train as gr_train  # gen-recsys/model/train.py
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from negative_sampler import NegativeSampler
from retrieval import RetrievalLoss
from sequence_model import SequenceModel


def load_ours(ckpt, output_dir, device, split="val", max_seq_len=None):
    """-> (score_fn, dataset). Architecture comes from the checkpoint's own config block."""
    dev = torch.device(device)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = blob["config"]
    K = max_seq_len or c["max_seq_len"]
    ds = protocol.make_dataset(output_dir, split, K, use_caption=c["use_content"],
                               caption_file=c["caption_file"])
    item_embed = ItemEmbedding(ItemEmbeddingConfig(
        num_items=len(ds.item_static), num_tags=ds.num_tags(), use_content=c["use_content"],
        dim=c["dim"], caption_dim=ds.caption_dim or 384)).to(dev)
    seq_model = SequenceModel(
        dim=c["dim"], num_heads=c["num_heads"], num_layers=c["num_layers"], ffn_dim=c["ffn_dim"],
        max_seq_len=2 * c["max_seq_len"] + 1, use_checkpoint=False, use_qk=c["use_qk"],
        dynamic_ts=c["dynamic_ts"]).to(dev)
    retr = RetrievalLoss(dim=c["dim"], t_base=c["t_base"], learnable_temp=c["learnable_temp"]).to(dev)
    item_embed.load_state_dict(blob["item_embed"])
    seq_model.load_state_dict(blob["seq_model"])
    retr.load_state_dict(blob["retrieval_loss_fn"])
    for m in (item_embed, seq_model, retr):
        m.eval()
    # the sampler is only needed because run_batch_forward draws (unused) eval negatives
    p80 = gr_train.load_or_compute_p80(ds, Path(output_dir))
    sampler = NegativeSampler(Path(output_dir), num_items=len(ds.item_static), p80=p80)
    cat = gr_train.catalog_features(ds, dev)
    with torch.no_grad():
        all_e = gr_train.embed_catalog(item_embed, cat, dev)

    @torch.no_grad()
    def score_fn(batch):
        cpu = {k: v.cpu() for k, v in batch.items()}   # run_batch_forward expects CPU id tensors
        fwd = gr_train.run_batch_forward(cpu, cat, item_embed, seq_model, sampler, 8, dev,
                                         all_item_e=all_e)
        cand = all_e.unsqueeze(0).expand(fwd["e_user_eval"].shape[0], -1, -1)
        return retr.eval_logit(fwd["e_user_eval"], cand)

    return score_fn, ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "preprocess_data" / "output"))
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-max-batches", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    fn, ds = load_ours(a.ckpt, a.output_dir, a.device, a.eval_split)
    res = protocol.evaluate(fn, ds, batch_size=a.batch_size, device=a.device,
                            max_batches=a.eval_max_batches)
    print("===== TOM TAT =====")
    print(f"model=ours ckpt={Path(a.ckpt).name} split={a.eval_split} "
          + " ".join(f"{k}={float(v):.4f}" for k, v in res.items()))
