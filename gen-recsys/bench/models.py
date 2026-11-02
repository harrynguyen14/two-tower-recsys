"""SASRec, GRU4Rec and HSTU (torch-rechub) with the recipes their authors publish.

Item j (0-based, as in gen-recsys) is token j+1; token 0 = PAD. Windows are LEFT-padded.
  - SASRec/GRU4Rec: forward(batch) -> hidden (B, L, d);  item_table() -> (num_items, d)
  - HSTU:           forward(batch) -> logits (B, L, num_items)  (already l2-normalised / temperature)

RECIPES (sources fetched 2026-10-31):
  hstu, sasrec : facebookresearch/generative-recommenders configs/ml-1m/{hstu,sasrec}-sampled-softmax-n128-final.gin
                 d=50, 2 blocks, 1 head, dropout 0.2, lr 1e-3, wd 0, sampled softmax over 128 negatives,
                 temperature 0.05, user AND item embeddings l2-normalised, no warmup.
                 (HSTU paper trains its SASRec baseline with the SAME recipe as HSTU.)
  gru4rec      : hidasib/GRU4Rec paramfiles/rees46_xe_shared_best.py: 1 GRU layer 512, tied (constrained) item
                 embedding, dropout on embedding 0.45, Adagrad lr 0.065, cross-entropy over in-batch targets +
                 2048 items sampled ~ popularity^0.5, logQ correction 1.0. The repo says to tune per dataset.
Deviations are listed in README.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_of(batch):
    """(B, L) tokens: item+1 where valid, 0 on padding."""
    return (batch["hist_video_ids"] + 1) * batch["hist_valid_mask"].long()


def targets_of(batch):
    """Next-item target (0-based) at every position; the last one is the real label."""
    return torch.cat([batch["hist_video_ids"][:, 1:], batch["label_video_id"][:, None]], dim=1)


def time_diffs_of(batch):
    """torch-rechub convention: time_diffs[i] = anchor - ts[i] in seconds, ONE anchor per sample,
    0 on padding. Then td[i] - td[j] = ts[j] - ts[i], which is what the pairwise rab^t needs.
    Anchor = label (query) time, known at serving and available to our own model as well."""
    ts = batch["hist_timestamps"]
    return ((batch["label_timestamp"][:, None] - ts).clamp(min=0) / 1000.0) * batch["hist_valid_mask"]


class SASRec(nn.Module):
    """Kang & McAuley 2018 blocks (learned positions, pre-LN causal self-attention, ReLU FFN)."""
    returns_logits = False
    l2, scale = True, 1.0 / 0.05          # HSTU-paper recipe: cosine / temperature 0.05

    def __init__(self, num_items, dim=50, layers=2, heads=1, max_seq_len=256, dropout=0.2):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.item = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim, dropout, activation="relu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def item_table(self):
        return self.item.weight[1:]

    def forward(self, batch):
        x = tokens_of(batch)
        valid = x.ne(0)
        B, L = x.shape
        h = self.item(x) * self.dim ** 0.5 + self.pos(torch.arange(L, device=x.device))
        h = self.drop(h) * valid[..., None]
        # causal AND key-not-padding; the diagonal stays open so padded rows never go NaN
        allow = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))[None] & valid[:, None, :]
        allow = allow | torch.eye(L, dtype=torch.bool, device=x.device)[None]
        mask = torch.zeros(B, L, L, device=x.device).masked_fill(~allow, float("-inf"))
        mask = mask.repeat_interleave(self.heads, dim=0)
        h = self.blocks(h, mask=mask)
        return self.norm(h) * valid[..., None]


class GRU4Rec(nn.Module):
    """Hidasi et al.: 1 GRU layer, constrained embedding (the item table is also the output layer)."""
    returns_logits = False
    l2, scale = False, 1.0

    def __init__(self, num_items, dim=512, dropout_embed=0.45, **_):
        super().__init__()
        self.item = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.drop = nn.Dropout(dropout_embed)

    def item_table(self):
        return self.item.weight[1:]

    def forward(self, batch):
        x = tokens_of(batch)
        valid = x.ne(0)
        B, L = x.shape
        n = valid.sum(1)
        ar = torch.arange(L, device=x.device)[None]
        shift = (L - n)[:, None]
        right = x.gather(1, (ar + shift).clamp(max=L - 1)) * (ar < n[:, None])   # left-pad -> right-pad
        pk = nn.utils.rnn.pack_padded_sequence(self.drop(self.item(right)), n.clamp(min=1).cpu(),
                                               batch_first=True, enforce_sorted=False)
        h, _ = self.gru(pk)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=L)
        back = (ar - shift).clamp(min=0)                                            # right-pad -> left-pad
        return h.gather(1, back[..., None].expand(-1, -1, h.shape[-1])) * valid[..., None]


class HSTU(nn.Module):
    """torch-rechub HSTUModel set up like Meta's ml-1m config: no input time-embedding (rab^{p,t} only),
    l2-normalised user/item, temperature 0.05, tied embeddings, no output bias."""
    returns_logits = True

    def __init__(self, num_items, dim=50, layers=2, heads=1, max_seq_len=256, dropout=0.2):
        super().__init__()
        from torch_rechub.models.generative.hstu import HSTUModel
        self.net = HSTUModel(vocab_size=num_items + 1, d_model=dim, n_heads=heads, n_layers=layers,
                             dqk=dim, dv=dim, max_seq_len=max_seq_len, dropout=dropout,
                             use_time_embedding=False, num_time_buckets=128,
                             time_bucket_fn="log", time_bucket_divisor=0.301, time_bucket_unit="seconds",
                             tie_embeddings=True, score_norm="l2", temperature=0.05, use_output_bias=False)

    def forward(self, batch):
        return self.net(tokens_of(batch), time_diffs_of(batch))[..., 1:]


MODELS = {"sasrec": SASRec, "gru4rec": GRU4Rec, "hstu": HSTU}
RECIPES = {
    "sasrec": dict(model=dict(dim=50, layers=2, heads=1, dropout=0.2), loss="ssm", opt="adam", lr=1e-3),
    "hstu": dict(model=dict(dim=50, layers=2, heads=1, dropout=0.2), loss="ssm", opt="adam", lr=1e-3),
    "gru4rec": dict(model=dict(dim=512, dropout_embed=0.45), loss="gru_xe", opt="adagrad", lr=0.065),
}
DEFAULT_LOSS = {k: v["loss"] for k, v in RECIPES.items()}


def build(name, num_items, **kw):
    return MODELS[name](num_items, **kw)


def user_item_scores(model, h, E):
    """(N, V) logits from hidden states: l2-normalise + temperature where the recipe says so."""
    if model.l2:
        h, E = F.normalize(h, dim=-1), F.normalize(E, dim=-1)
    return model.scale * (h @ E.T)


def last_scores(model, batch):
    """(B, num_items) scores for the next item = the eval entry point."""
    out = model(batch)[:, -1]
    return out if model.returns_logits else user_item_scores(model, out, model.item_table())


def item_log_q(counts, alpha):
    """log of the sampling distribution ~ popularity^alpha (GRU4Rec `sample_alpha`)."""
    p = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1) ** alpha
    return (p / p.sum()).log()


def loss_fn(model, batch, kind, n_neg=128, pop_log_q=None, n_sample=2048, n_inbatch=1024, logq=1.0):
    """kind:
      ssm    sampled softmax, n_neg uniform negatives shared by the batch (HSTU/SASRec recipe)
      gru_xe GRU4Rec cross-entropy: n_inbatch batch targets + n_sample popularity-sampled items, logQ-corrected
      ce     full softmax;  bce  original SASRec (1 negative);  bprmax  GRU4Rec 2018 (needs pop_log_q)
    """
    out = model(batch)
    sel = batch["hist_valid_mask"]
    tgt = targets_of(batch)[sel]
    dev, V = tgt.device, (out.shape[-1] if model.returns_logits else model.item_table().shape[0])
    if model.returns_logits:
        lg = out[sel]
        full, pos_of, neg_of = (lambda: lg), (lambda: lg.gather(1, tgt[:, None])[:, 0]), (lambda ids: lg[:, ids])
    else:
        h, E = out[sel], model.item_table()
        full = lambda: user_item_scores(model, h, E)
        pos_of = lambda: model.scale * (F.normalize(h, dim=-1) * F.normalize(E[tgt], dim=-1)).sum(-1) \
            if model.l2 else (h * E[tgt]).sum(-1)
        neg_of = lambda ids: user_item_scores(model, h, E[ids])

    def sampled_ce(ids, log_q=None):
        pos, neg = pos_of(), neg_of(ids)
        if log_q is not None:
            pos, neg = pos - logq * log_q[tgt], neg - logq * log_q[ids][None]
        neg = neg.masked_fill(ids[None] == tgt[:, None], float("-inf"))      # drop accidental positives
        return F.cross_entropy(torch.cat([pos[:, None], neg], 1), torch.zeros_like(tgt))

    if kind == "ce":
        return F.cross_entropy(full(), tgt)
    if kind == "ssm":
        return sampled_ce(torch.randint(0, V, (n_neg,), device=dev))
    if kind == "gru_xe":
        q = pop_log_q.to(dev)
        samp = torch.multinomial(q.exp(), n_sample, replacement=True)
        inb = tgt[torch.randperm(len(tgt), device=dev)[:n_inbatch]]
        return sampled_ce(torch.cat([samp, inb]), q)
    assert not model.returns_logits, f"{kind} needs the hidden state; not available for HSTU"
    if kind == "bce":
        pos = pos_of()
        neg = (h * E[torch.randint(0, V, tgt.shape, device=dev)]).sum(-1)
        return F.softplus(-pos).mean() + F.softplus(neg).mean()
    if kind == "bprmax":
        pos = pos_of()
        ids = torch.multinomial(pop_log_q.to(dev).exp(), n_sample, replacement=True)
        s_neg = neg_of(ids)
        w = s_neg.softmax(-1)
        main = -torch.log((w * torch.sigmoid(pos[:, None] - s_neg)).sum(-1) + 1e-24)
        return (main + 1.95 * (w * s_neg ** 2).sum(-1)).mean()
    raise ValueError(kind)
