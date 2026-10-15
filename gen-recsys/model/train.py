"""Training loop — nối toàn bộ module: Dataset -> ItemEmbedding -> SequenceModel ->"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from dataset import ACTION_VECTOR_FIELDS, GenRecsysDataset, ITEM_CATEGORICAL_FIELDS, MAX_SEQ_LEN
from eval import (aggregate_by_group, matched_by_n_u, compute_metrics_at_k,
                  compute_positive_rank, print_matched, print_eval_report)
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from negative_sampler import NegativeSampler
from ranking_loss import BINARY_ACTION_FIELDS, RankingLoss
from retrieval import RetrievalLoss
from sequence_model import SequenceModel

BINARY_ACTION_INDICES = [ACTION_VECTOR_FIELDS.index(f) for f in BINARY_ACTION_FIELDS]


def build_category_counts(item_static_path: Path) -> dict[str, int]:
    item_static = np.load(item_static_path, mmap_mode="r")
    return {field: int(item_static[field].max()) + 1 for field in ITEM_CATEGORICAL_FIELDS}


def item_features_to_device(features: dict, device: torch.device) -> dict:
    return {
        "category_ids": {k: v.to(device, non_blocking=True) for k, v in features["category_ids"].items()},
        "tag_ids": features["tag_ids"].to(device, non_blocking=True),
        "author_idx": features["author_idx"].to(device, non_blocking=True),
        "music_idx": features["music_idx"].to(device, non_blocking=True),
        "caption_embedding": features["caption_embedding"].to(device, non_blocking=True),
        "caption_mask": features["caption_mask"].to(device, non_blocking=True),
    }


def catalog_features(dataset: GenRecsysDataset, device: torch.device) -> dict:
    """Feature cua TOAN catalog, dua len GPU 1 LAN luc khoi tao.

    [THEM 2026-10-15 — tang toc] Truoc day moi step gom feature tren CPU cho ~100K luot tra
    (128x256 lich su + 128x512 negative) roi copy sang GPU: CPU 107%, GPU 73% tren Kaggle T4.
    Pure chi co 7,583 item va embed item gio CHI phu thuoc video_id (bang maturity da xoa)
    => giu feature catalog tren GPU (~12 MB), xem embed_catalog.
    """
    ids = torch.arange(len(dataset.item_static))
    return item_features_to_device(dataset.get_item_features(ids), device)


def embed_catalog(item_embed: ItemEmbedding, cat_feats: dict, device: torch.device) -> torch.Tensor:
    """(num_items, d): chay item tower MOT LAN cho ca catalog — moi step goi 1 lan roi gather.

    Toan hoc Y HET embed tung id rieng (item tower khong dropout, khong phu thuoc batch):
    gradient cua 1 item = tong gradient moi lan no xuat hien, dung nhu truoc. Khoi luong tinh
    giam ~13x (7,583 thay vi ~100K hang). ponytail: chi hop catalog NHO (Pure); catalog
    lon (27K: 32M item) phai quay lai embed theo id cua batch.
    """
    n = cat_feats["author_idx"].shape[0]
    return item_embed(
        torch.arange(n, device=device), cat_feats["category_ids"],
        cat_feats["author_idx"], cat_feats["music_idx"],
        cat_feats["caption_embedding"], cat_feats["caption_mask"], cat_feats["tag_ids"],
    )


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """He so LR tai global step: warmup TUYEN TINH roi COSINE ve 0 (Loshchilov & Hutter, SGDR).

    [THEM 2026-10-15] Tinh truc tiep tu global step => khong co state scheduler phai luu,
    --resume tu dung dung cho. Warmup on dinh giai doan dau (loss cao, ID/delta/W_ts vua roi
    khoi 0); cosine giam LR cuoi ky de hoi tu sau hon voi cung ngan sach step.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def load_or_compute_p80(
    dataset: GenRecsysDataset, output_dir: Path | None,
) -> np.int64:
    """Doc mốc p80 tu `train_val_boundary_p80.npy` (ghi boi build_interactions.py).

    [SỬA 2026-10-13 — error.md #15] p80 ĐỌC từ file (CÙNG lúc tính split train/val/test
    ở build_interactions.py) nếu file tồn tại — trước đây TÍNH LẠI độc lập bằng
    `np.percentile` ngay tại đây, 2 nguồn không có gì đảm bảo mãi khớp nếu 1 trong 2 công
    thức đổi. Fallback tính lại + CẢNH BÁO nếu file chưa có (dữ liệu cũ build trước khi
    thêm dòng lưu file này).

    [2026-10-15] Gio CHI con NegativeSampler dung (tranh leak val/test, error.md #9) — bang
    maturity tung dung chung da xoa.
    """
    p80_path = Path(output_dir) / "train_val_boundary_p80.npy" if output_dir else None
    if p80_path is not None and p80_path.exists():
        return np.int64(np.load(p80_path))
    import warnings
    warnings.warn(
        "train_val_boundary_p80.npy khong ton tai — TINH LAI p80 doc lap tai day "
        "(co the LECH voi build_interactions.py neu khong cung du lieu/cong thuc). "
        "Build lai du lieu de dung file nguon duy nhat.",
        stacklevel=2,
    )
    label_t = dataset.history_meta["t"][dataset.sample_position].astype(np.int64)
    return np.int64(np.percentile(label_t, 80))


def setup_ddp() -> tuple[bool, int, int, int]:
    """Khởi tạo DDP từ biến môi trường của torchrun. Trả về (bật, rank, local_rank,..."""
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False, 0, 0, 1
    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def run_batch_forward(
    batch: dict,
    cat_feats: dict,
    item_embed: ItemEmbedding,
    seq_model: SequenceModel,
    neg_sampler: NegativeSampler,
    num_negatives: int,
    device: torch.device,
    all_item_e: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """1 forward pass đầy đủ cho 1 batch — dùng CHUNG bởi train() và evaluate().

    `all_item_e`: embedding catalog tinh SAN (eval, no_grad — tinh 1 lan cho ca split).
    None (train) => tinh lai MOI step de gradient chay vao item tower.
    """
    if all_item_e is None:
        all_item_e = embed_catalog(item_embed, cat_feats, device)
    hist_video_ids_cpu = batch["hist_video_ids"]
    hist_timestamps_cpu = batch["hist_timestamps"]
    hist_action = batch["hist_action"].to(device, non_blocking=True)
    key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
    hist_valid_mask = batch["hist_valid_mask"].to(device, non_blocking=True)
    label_video_id_cpu = batch["label_video_id"]
    label_action = batch["label_action"].to(device, non_blocking=True)

    B, K = hist_video_ids_cpu.shape

    hist_e_i = all_item_e[hist_video_ids_cpu.to(device, non_blocking=True)]   # (B, K, d)

    hist_n_u = batch["hist_n_u"].to(device, non_blocking=True)

    label_e_i = all_item_e[label_video_id_cpu.to(device, non_blocking=True)]

    hidden = seq_model(
        hist_e_i, hist_action, key_padding_mask,
        hist_timestamps=hist_timestamps_cpu.to(device, non_blocking=True),
        hist_video_ids=hist_video_ids_cpu.to(device, non_blocking=True),
    )

    # [SUA 2026-10-07] .detach() o NHANH TARGET — bat buoc, khong phai toi uu.
    # hist_e_i la CUNG tensor da dua vao seq_model, nen khong detach thi gradient chay vao
    # Phi_i qua HAI duong: (1) input, (2) target. Duong (2) la shortcut: model giam loss bang
    # cach CO Phi_i ve phia pred thay vi hoc pred du doan Phi_i — dung co che representation
    # collapse ma SimSiam/BYOL phai chan bang stop-gradient. Do duoc: |grad| duong target
    # = 22.8 (khac 0 ro rang). NGHI VAN: ket luan "multi-vector that bai" (cosine->0.98) co
    # the la HE QUA cua shortcut nay chu khong phai cua multi-vector — chay lai truoc khi chot.
    #
    # [BO .detach() 2026-10-14 — ROOT CAUSE regression "model khong hoc duoc gi"] Lap luan
    # SimSiam/BYOL o tren SAI cho bai nay: SimSiam/BYOL KHONG co negative nen moi can
    # stop-gradient; o day sampled-softmax CO 512 negative lay tu CUNG bang Phi_i, chinh mau
    # so softmax chan collapse (SASRec/HSTU deu de gradient chay ca 2 nhanh). Te hon,
    # detach CHI nhanh positive con `neg_e_i` VAN co gradient => Phi_i chi nhan luc DAY ra
    # xa pred, khong bao gio duoc KEO ve => |e_pos| troi 84-500 (khong phai do thieu
    # normalize — cosine 10-09 chi che trieu chung) va item token la content THUAN (khong
    # co ID embedding) nen khong con duong nao khac de can chinh. A/B cung data p80, cung
    # code HEAD, chi khac dong nay (CPU, batch 64, seq 64, 600 step, 3840 mau val
    # full-ranking):
    #     .detach()  : retrieval 7.69 -> 7.42 (dung yen), hr@10 0.0016 (random 0.0013),
    #                  median rank 3736/7583, |e_pos| 265
    #     bo detach  : retrieval 7.76 -> 5.87,            hr@10 0.0336 (> pop 0.0239),
    #                  median rank 707,                   |e_pos| 3.6
    # Moc tot cu (hr@10 0.0469, 10-02, commit 212cb7d) cung KHONG co detach.
    target_e_i = torch.cat([hist_e_i[:, 1:, :], label_e_i.unsqueeze(1)], dim=1)
    target_ids_cpu = torch.cat([hist_video_ids_cpu[:, 1:], label_video_id_cpu.unsqueeze(1)], dim=1)
    pair_valid = hist_valid_mask.clone()
    pair_valid[:, :-1] &= hist_valid_mask[:, 1:]

    # [SUA 2026-10-07] exclude TOAN BO target_ids, khong chi label.
    # Loss tinh tren MOI vi tri t voi target = [hist[1:], label], nhung negative dung CHUNG
    # cho ca chuoi va truoc day chi loai trung voi label (vi tri K-1). Do tren phan bo thuc
    # cua Pure (C=512): P(mot positive lot vao tap negative) = 0.270 theo phan bo positive
    # (positive thien ve popular, negative cung sample theo popularity) => ~69/255 vi tri moi
    # batch bi day "item dung ra xa". Khi positive trung negative, CE KHONG the xuong 0 — san
    # log(2)=0.693 (do duoc: loss 0.0047 -> 0.6954). Nhieu label CO HE THONG, khong ngau nhien.
    neg_ids, neg_log_q = neg_sampler.sample(
        B, num_negatives, device, exclude=target_ids_cpu,
    )
    neg_e_i = all_item_e[neg_ids.to(device, non_blocking=True)]               # (B, C, d)
    target_log_q = neg_sampler.log_q_for(target_ids_cpu.reshape(-1).to(device, non_blocking=True)).view(B, K)

    e_user_eval = hidden[:, -1, :]

    cand_e_i = torch.cat([label_e_i.unsqueeze(1), neg_e_i], dim=1)
    positive_log_q = neg_sampler.log_q_for(label_video_id_cpu.to(device, non_blocking=True)).unsqueeze(1)
    log_q = torch.cat([positive_log_q, neg_log_q], dim=1)

    return {
        "pred": hidden,
        "target_e_i": target_e_i,
        "target_log_q": target_log_q,
        "neg_e_i": neg_e_i,
        "neg_log_q": neg_log_q,
        "pair_valid": pair_valid,
        "hist_action": hist_action,
        "hist_valid_mask": hist_valid_mask,
        "e_user_eval": e_user_eval,
        # Poly-encoder (M>1) pool tu TOAN chuoi, khong chi vi tri cuoi — can ca hai o day.
        "key_padding_mask": key_padding_mask,
        "cand_e_i": cand_e_i,
        "log_q": log_q,
        "label_action": label_action,
        "is_user_cold": batch["is_user_cold"],
        "is_item_cold": batch["is_item_cold"],
        "is_user_lowhistory": batch["is_user_lowhistory"],
        "hist_video_ids": hist_video_ids_cpu,
        "label_video_id": label_video_id_cpu,
        "n_u_at_pred": hist_n_u[:, -1].detach().cpu(),
    }


def train(
    output_dir: str = "../preprocess_data/output",
    dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    ffn_dim: int = 256,
    batch_size: int = 64,
    num_negatives: int = 512,
    eval_negatives: int = 1024,
    # [SUA 2026-10-09] 0.8 -> 0.07 + hoc duoc. `_score` gio L2-NORMALIZE ca hai phia nen
    # diem la COSINE trong [-1,1]: 0.8 cu cho logit toi da 1.25 — qua phang de phan biet
    # 7,583 item. 0.07 ~ init CLIP, va `learnable_temp` de model tu tim. Chi tiet nguyen nhan
    # (|e_pos| troi len 473 sau fix #1 => CE saturate => hr@10 0.0018): retrieval.py.
    # CANH BAO: doi thang logit => checkpoint TRUOC 10-09 KHONG nap lai de so sanh duoc.
    t_base: float = 0.07,
    learnable_temp: bool = True,
    lr: float = 1e-3,
    # [THEM 2026-10-13] error.md #21: da co scaler.unscale_(optimizer) truoc buoc nay
    # (de clip o DUNG thang gradient that, khong phai thang da nhan fp16 scale) nhung
    # truoc day KHONG clip gi ca — mat chinh ly do thuong lam unscale_ som. 1.0 la gia
    # tri mac dinh pho bien (PyTorch tutorial, nhieu paper NLP/recsys), chua tune rieng
    # cho model nay.
    grad_clip_norm: float = 1.0,
    # [THEM 2026-10-15] NormFace (arXiv 1704.06369 §3.2): logit COSINE => gradient VUONG GOC
    # voi vector => moi buoc update ||x|| chi TANG ("weight decay is necessary"). Da gap
    # that: |e_pos| troi 84-500 (luc con .detach(), xem run_batch_forward). Bo detach xong
    # con 3.6 nhung train dai hon van troi => effective lr tu giam (Weight Norm 1602.07868).
    # AdamW (decay TACH khoi gradient) — Adam + L2 thi decay bi chia cho sqrt(v), yeu that
    # thuong o tham so gradient lon. 0.01 = mac dinh AdamW, chua tune.
    weight_decay: float = 0.01,
    # [THEM 2026-10-15] warmup tuyen tinh + cosine ve 0, xem lr_factor. "constant" = hanh vi cu.
    lr_schedule: str = "cosine",
    warmup_frac: float = 0.05,
    ranking_loss_weight: float = 0.5,
    num_epochs: int = 1,
    max_steps_per_epoch: int | None = None,
    # [THEM 2026-10-13] error.md #22: truoc day evaluate() dung CHUNG max_steps_per_epoch
    # (gioi han so BATCH TRAIN moi epoch cho debug/chay nhanh) de cat luon so batch EVAL —
    # cung loai rui ro "tune va report tren cung split" da tung tra gia (xem loi #4 dau
    # formula.md). Tach rieng: None = eval tren TOAN BO (hanh vi mac dinh cu neu
    # max_steps_per_epoch=None, nhung tu nay KHONG con bi anh huong boi gia tri train).
    eval_max_batches: int | None = None,
    amp: bool = True,
    use_qk: bool = True,
    dynamic_ts: bool = False,
    static_delta: bool = False,
    full_ranking: bool = False,
    use_checkpoint: bool = False,
    save_every: int | None = None,
    save_path: str | None = None,
    resume: str | None = None,
    eval_only: bool = False,
    # [THEM 2026-10-07] Split de eval. Mac dinh "val" — dung de TUNE/chon nhanh ablation.
    # Truoc day ca hai cho goi evaluate() deu hardcode "val", nen `test.npy` (140,954 sample,
    # da build tu 2026-09-17) CHUA BAO GIO duoc cham. Hau qua: ~11 nhanh ablation chon tren
    # val roi BAO CAO cung bang val => moi so da cong bo (0.0376/0.0433/0.0465/0.0469, bang
    # 2x3) bi bias boi model selection. Quy trinh dung: tune tren val, chot cau hinh, roi
    # chay DUNG MOT LAN `--eval-split test` va bao so do.
    eval_split: str = "val",
    max_seq_len: int = MAX_SEQ_LEN,
    random_window: bool = True,
    no_tag: bool = False,
    # [THEM 2026-10-08] Bo nhanh caption khoi GMU. Dung de chay thu model KHI CHUA CO
    # caption_embeddings.npy. Dataset cung TU DONG tat khi thieu file, nen co nay chi can
    # khi muon tat DU DA co file (ablation that).
    no_caption: bool = False,
    no_item_id: bool = False,
    num_workers: int = 4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)

    ddp, rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}") if ddp else torch.device(device_str)

    def log(*a, **kw):
        """In CHỈ ở rank 0 — nếu không, mọi dòng log bị nhân world_size lần."""
        if is_main:
            print(*a, **kw)

    # random_window CHI o train (xem dataset._window_start): eval phai dung cua so ngay
    # truoc diem du doan de khop serving.
    train_dataset = GenRecsysDataset(output_dir, split="train", max_seq_len=max_seq_len,
                                     random_window=random_window,
                                     use_caption=not no_caption)
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        if ddp else None
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    num_items = len(train_dataset.item_static)
    num_authors = int(train_dataset.item_static["author_idx"].max()) + 1
    num_music = int(train_dataset.item_static["music_idx"].max()) + 1
    num_categories = build_category_counts(output_dir / "item_static.npy")

    item_config = ItemEmbeddingConfig(
        num_items=num_items, num_authors=num_authors, num_music=num_music,
        num_categories=num_categories,
        num_tags=train_dataset.num_tags(), use_tag=not no_tag, dim=dim,
        # `train_dataset.use_caption` da gop ca co --no-caption LAN viec file co ton tai
        # khong (dataset tu tat khi thieu), nen lay tu day thay vi tu `no_caption`.
        use_caption=train_dataset.use_caption,
        use_item_id=not no_item_id,
        caption_dim=train_dataset.caption_dim or 384,
    )
    item_embed = ItemEmbedding(item_config).to(device, non_blocking=True)

    seq_model = SequenceModel(
        dim=dim, num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim,
        max_seq_len=2 * max_seq_len + 1,
        use_checkpoint=use_checkpoint, use_qk=use_qk, dynamic_ts=dynamic_ts,
        static_delta=static_delta,
    ).to(device, non_blocking=True)
    retrieval_loss_fn = RetrievalLoss(
        dim=dim, t_base=t_base, learnable_temp=learnable_temp,
    ).to(device, non_blocking=True)
    ranking_loss_fn = RankingLoss(dim=dim).to(device, non_blocking=True)
    # [THEM 2026-10-13 — error.md #9] Doc p80 MOT LAN cho NegativeSampler (tranh leak
    # val+test vao phan phoi negative).
    p80 = load_or_compute_p80(train_dataset, output_dir)
    neg_sampler = NegativeSampler(output_dir, num_items=num_items, p80=p80)

    dense_params = (
        item_embed.dense_parameters()
        + list(seq_model.parameters())
        + list(retrieval_loss_fn.parameters())
        + list(ranking_loss_fn.parameters())
    )
    # Chi decay MA TRAN trong so + bang embedding (ndim>=2). KHONG decay: bias/LayerNorm
    # (ndim<2, thong le), delta/log_temp (vo huong — decay keo ve 0 = keo ve NHOM DOI
    # CHUNG cua ablation; log_temp ve 0 = temp ve 1.0, phang het logit cosine), va bang
    # bias attention ts_w (2 chieu nhung cung la bias zero-init, cung ly do).
    _no_decay_names = ("ts_w",)
    decay, no_decay = [], []
    for mod in (item_embed, seq_model, retrieval_loss_fn, ranking_loss_fn):
        for pname, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or pname.split(".")[-1] in _no_decay_names:
                no_decay.append(p)
            else:
                decay.append(p)
    assert len(decay) + len(no_decay) == sum(p.requires_grad for p in dense_params)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, fused=(device.type == "cuda"),
    )
    use_amp = (device.type == "cuda") and amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        item_embed_ddp = DDP(item_embed, device_ids=[local_rank], find_unused_parameters=True)
        seq_model_ddp = DDP(seq_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        item_embed_ddp, seq_model_ddp = item_embed, seq_model

    if no_tag:
        raise ValueError(
            "--no-tag: content token chi con GMU va `fuse` thanh phep chieu d->d, khong "
            "con gi de hop nhat. Gan nhu chac la nham co."
        )

    log(f"[train] nhanh tag: {'TAT' if no_tag else 'BAT'}"
        f" | item-ID: {'TAT' if no_item_id else 'BAT'}")

    steps_per_epoch = min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch else len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_frac * total_steps))
    cat_feats = catalog_features(train_dataset, device)

    ckpt_modules = {
        "item_embed": item_embed, "seq_model": seq_model,
        "retrieval_loss_fn": retrieval_loss_fn, "ranking_loss_fn": ranking_loss_fn,
    }
    ckpt_config = {
        "dim": dim, "num_heads": num_heads, "num_layers": num_layers, "ffn_dim": ffn_dim,
        "amp": amp, "use_qk": use_qk, "dynamic_ts": dynamic_ts,
        "static_delta": static_delta,
        "t_base": t_base, "learnable_temp": learnable_temp,
        "max_seq_len": max_seq_len, "eval_negatives": eval_negatives,
        "random_window": random_window,
        "no_tag": no_tag, "no_item_id": no_item_id,
        # Lay tu dataset (gop ca co --no-caption lan viec file co ton tai), de checkpoint
        # ghi dung cau hinh THUC TE da train, khong phai y dinh tren CLI.
        "use_caption": train_dataset.use_caption,
    }
    # [THEM 2026-10-13] error.md #18: hyperparameter HUAN LUYEN (duoc phep doi giua cac
    # lan resume, vd ha lr) — KHONG dua vao ckpt_config (cai do RAISE khi lech, dung cho
    # KIEN TRUC). Thieu field nay thi checkpoint khong tu du de biet da chay voi gia tri
    # gi, phai nho/doc lai log CLI (neu con).
    run_metadata = {
        "lr": lr, "grad_clip_norm": grad_clip_norm, "weight_decay": weight_decay,
        "lr_schedule": lr_schedule, "warmup_frac": warmup_frac,
        "ranking_loss_weight": ranking_loss_weight,
        "batch_size": batch_size, "num_negatives": num_negatives,
        "eval_negatives": eval_negatives,
        "full_ranking": full_ranking, "eval_split": eval_split,
    }
    ckpt_path = Path(save_path) if save_path else output_dir / "ckpt.pt"

    start_epoch, start_step = 0, 0
    if resume or eval_only:
        src = resume or save_path or str(ckpt_path)
        state = load_checkpoint(
            src, modules=ckpt_modules, config=ckpt_config,
            optimizer=None if eval_only else optimizer,
        )
        start_epoch, start_step = state["epoch"], state["step"]
        log(f"[resume] nạp {src} — epoch={start_epoch} step={start_step}")
        # [SUA 2026-10-14 — code review] In run_metadata cua LAN CHAY TRUOC — truoc day
        # save_checkpoint ghi field nay nhung khong noi nao doc lai, nguoi resume van
        # phai tu nho/doc log CLI cu (dung dieu #18 dinh sua).
        if state.get("run_metadata"):
            log(f"[resume] hyperparameter lan chay truoc: {state['run_metadata']}")

    if eval_only:
        evaluate(
            eval_split, output_dir, item_embed, seq_model, neg_sampler,
            retrieval_loss_fn, eval_negatives, batch_size, device,
            use_caption=train_dataset.use_caption,
            max_seq_len=max_seq_len, max_batches=eval_max_batches,
            num_workers=num_workers,
            full_ranking=full_ranking,
        )
        return

    forward_time_acc = 0.0
    step_time_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Cua so truot phai DOI moi epoch, neu khong moi epoch cho cung cua so va tac dung
        # "qua nhieu epoch thay het chuoi" mat han (dataset._window_start).
        train_dataset.set_epoch(epoch)
        progress = tqdm(
            enumerate(train_loader), total=steps_per_epoch, desc=f"epoch {epoch}", unit="step",
            disable=not is_main,
        )
        for step, batch in progress:
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            if epoch == start_epoch and step < start_step:
                continue

            step_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fwd = run_batch_forward(
                    batch, cat_feats, item_embed_ddp, seq_model_ddp, neg_sampler,
                    num_negatives, device,
                )
                forward_time_acc += time.perf_counter() - step_start

                r_loss = retrieval_loss_fn.forward_sequence(
                    fwd["pred"], fwd["target_e_i"], fwd["neg_e_i"],
                    fwd["target_log_q"], fwd["neg_log_q"], fwd["pair_valid"],
                )

                binary_labels = fwd["hist_action"][..., BINARY_ACTION_INDICES]
                k_loss, _ = ranking_loss_fn.forward_sequence(
                    fwd["pred"], binary_labels, fwd["hist_valid_mask"],
                )

                loss = r_loss + ranking_loss_weight * k_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # [THEM 2026-10-13] error.md #21: da unscale_ (dung de clip o DUNG thang
            # gradient that) nhung truoc day KHONG clip gi — bo lo chinh ly do thuong
            # goi unscale_ som. Thieu clip => mot batch co gradient dot bien (vd item
            # cuc hiem, hoac delta_proj troi xa — xem formula.md §4.1ter) co the day
            # optimizer di xa.
            torch.nn.utils.clip_grad_norm_(dense_params, max_norm=grad_clip_norm)

            grad_norms = {}
            for pname in ("delta", "ts_w"):
                total = 0.0
                for layer in seq_model.decoder.layers:
                    g = getattr(layer.attn, pname).grad
                    if g is not None:
                        total += g.norm().item() ** 2
                grad_norms[pname] = total ** 0.5

            if lr_schedule == "cosine":
                cur_lr = lr * lr_factor(epoch * steps_per_epoch + step, total_steps, warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
            scaler.step(optimizer)
            scaler.update()
            step_time_acc += time.perf_counter() - step_start

            progress.set_postfix(loss=f"{loss.item():.4f}", retrieval=f"{r_loss.item():.4f}", ranking=f"{k_loss.item():.4f}")

            if is_main and save_every and step > 0 and step % save_every == 0:
                save_checkpoint(
                    ckpt_path, epoch=epoch, step=step + 1,
                    global_step=epoch * steps_per_epoch + step + 1, modules=ckpt_modules,
                    optimizer=optimizer, scaler=scaler,
                    config=ckpt_config, run_metadata=run_metadata,
                )
                tqdm.write(f"[ckpt] đã lưu {ckpt_path} (epoch={epoch} step={step + 1})")

            if step % 50 == 0:
                forward_pct = 100.0 * forward_time_acc / step_time_acc if step_time_acc > 0 else 0.0
                fwd_msg = f" | fwd%={forward_pct:.0f}"
                forward_time_acc = 0.0
                step_time_acc = 0.0
                with torch.no_grad():
                    vals = {}
                    for pname in ("delta", "ts_w"):
                        ps = [getattr(l.attn, pname).abs().mean().item() for l in seq_model.decoder.layers]
                        vals[pname] = sum(ps) / len(ps)
                tqdm.write(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"retrieval={r_loss.item():.4f} ranking={k_loss.item():.4f}"
                    f" | |d|={vals['delta']:.2e}"
                    f" |ts|={vals['ts_w']:.2e}"
                    f" gd={grad_norms['delta']:.2e}"
                    f" gts={grad_norms['ts_w']:.2e}"
                    f" lr={optimizer.param_groups[0]['lr']:.2e}"
                    f"{fwd_msg}"
                )

        if is_main:
            save_checkpoint(
                ckpt_path, epoch=epoch + 1, step=0,
                global_step=(epoch + 1) * steps_per_epoch, modules=ckpt_modules,
                optimizer=optimizer, scaler=scaler,
                config=ckpt_config, run_metadata=run_metadata,
            )
            print(f"[ckpt] đã lưu {ckpt_path} (hết epoch {epoch})")
        if ddp:
            torch.distributed.barrier()

        if not is_main:
            continue
        evaluate(
            eval_split, output_dir, item_embed, seq_model, neg_sampler, retrieval_loss_fn,
            eval_negatives, batch_size, device,
            use_caption=train_dataset.use_caption,
            max_seq_len=max_seq_len, max_batches=eval_max_batches,
            num_workers=num_workers, full_ranking=full_ranking,
        )

    cleanup_ddp(ddp)


_CKPT_MODULES = ("item_embed", "seq_model", "retrieval_loss_fn", "ranking_loss_fn")


def cleanup_ddp(ddp: bool) -> None:
    """Đóng process group. Bỏ qua thì tiến trình có thể treo lúc thoát."""
    if ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def save_checkpoint(path: Path, *, epoch: int, step: int, global_step: int,
                    modules: dict, optimizer, scaler, config: dict,
                    run_metadata: dict | None = None) -> None:
    """Lưu đủ để train tiếp ĐÚNG chỗ đã dừng, không chỉ để eval.

    [THEM 2026-10-13] error.md #18: `config` CHỈ chứa tham số KIẾN TRÚC (dim, num_heads,
    ...) — `load_checkpoint` so khớp nó và RAISE nếu lệch (đúng, vì đổi kiến trúc giữa
    chừng ra model lai). `run_metadata` là field RIÊNG, KHÔNG đi qua so khớp đó, dùng để
    ghi lại hyperparameter huấn luyện (lr, batch_size, ranking_loss_weight...) — các giá
    trị NÀY được phép đổi giữa các lần resume (ví dụ hạ lr) nên không thể nhồi vào
    `config`. Thiếu field này thì checkpoint không tự đủ để reproduce — trước đây chỉ có
    thể suy luận ngược từ lệnh CLI đã chạy (nếu còn nhớ/còn log).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "epoch": epoch, "step": step, "global_step": global_step, "config": config,
        "run_metadata": run_metadata or {},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    for name in _CKPT_MODULES:
        blob[name] = modules[name].state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, modules: dict, optimizer=None, scaler=None,
                    config: dict | None = None) -> dict:
    """Nạp checkpoint; trả về {"epoch", "step", "global_step", "run_metadata"}.

    [SUA 2026-10-14] Doc lai docstring: them "run_metadata" (hyperparameter huan luyen
    cua lan chay truoc, xem save_checkpoint) — truoc day chi tra 3 key dau, dung trong
    khi code da them key thu 4 (error.md #18 nang cap)."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)

    if config is not None:
        saved = blob.get("config", {})
        lech = {
            k: (saved.get(k), v)
            for k, v in config.items()
            if k in saved and saved[k] != v
        }
        if lech:
            raise SystemExit(
                "[resume] CẤU HÌNH LỆCH so với checkpoint — train tiếp sẽ ra model lai, "
                "không đọc được kết quả:\n"
                + "\n".join(f"    {k}: checkpoint={a!r} nhưng lần chạy này={b!r}"
                             for k, (a, b) in lech.items())
            )

    thieu = [n for n in _CKPT_MODULES if n not in blob]
    if thieu:
        raise SystemExit(f"[resume] checkpoint THIẾU module: {thieu} — không nạp được")
    for name in _CKPT_MODULES:
        modules[name].load_state_dict(blob[name])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    if scaler is not None and "scaler" in blob:
        scaler.load_state_dict(blob["scaler"])
    return {"epoch": blob.get("epoch", 0), "step": blob.get("step", 0),
            "global_step": blob.get("global_step", 0),
            # [SUA 2026-10-14 — code review] Truoc day save_checkpoint ghi run_metadata
            # (lr, grad_clip_norm, ranking_loss_weight...) nhung load_checkpoint KHONG
            # doc lai field nay — write-only, khong giai quyet dung muc tieu #18 (muc
            # tieu la checkpoint TU DU de reproduce/kiem tra khi resume).
            "run_metadata": blob.get("run_metadata", {})}


def _report_history_diagnostic(
    rank: torch.Tensor,
    n_u: torch.Tensor,
    n_cand: torch.Tensor,
) -> None:
    """LỊCH SỬ DÀI có giúp hay hại? — câu hỏi gốc của chất lượng, không liên quan cold-start."""
    print("\n--- CHẨN ĐOÁN: lịch sử DÀI giúp hay hại? (rank thấp = tốt) ---")
    C = int(n_cand.float().median().item())
    print(f"    ngẫu nhiên -> median≈{C // 2}; N_u = số tương tác của user TẠI vị trí dự đoán")
    bins = [(0, 5), (6, 20), (21, 50), (51, 100), (101, 300), (301, 10 ** 9)]
    print(f"    {'N_u':<14}{'n':>8}{'median rank':>13}{'p25':>7}{'p75':>7}{'top-10%':>10}")
    for lo, hi in bins:
        m = (n_u >= lo) & (n_u <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        r = rank[m].float()
        top = (r < C * 0.1).float().mean().item()
        name = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"    {name:<14}{n:>8}{r.median():>13.0f}{r.quantile(0.25):>7.0f}"
              f"{r.quantile(0.75):>7.0f}{top:>10.3f}")
    ru = torch.argsort(torch.argsort(n_u.float())).float()
    rr = torch.argsort(torch.argsort(rank.float())).float()
    rho = torch.corrcoef(torch.stack([ru, rr]))[0, 1].item()
    print(f"    tương quan hạng N_u <-> rank: rho={rho:+.4f}")
    print("    CẢNH BÁO: rho>0 KHÔNG có nghĩa lịch sử dài hại — đã chứng minh 2026-09-22 nó")
    print("    là CONFOUND (user warm xem đồ ngách hơn -> bài khó hơn) — memory 2026-09-22.")


def _report_rank_diagnostic(
    rank: torch.Tensor,
    norm_user: torch.Tensor,
    norm_pos: torch.Tensor,
    n_cand: torch.Tensor,
    is_user_cold: torch.Tensor,
    is_item_cold: torch.Tensor,
) -> None:
    """In phân bố THỨ HẠNG positive theo nhóm — phân định nguyên nhân recall=0."""
    groups = {
        "warm_warm": (~is_user_cold) & (~is_item_cold),
        "cold_user_warm_item": is_user_cold & (~is_item_cold),
        "warm_user_cold_item": (~is_user_cold) & is_item_cold,
        "cold_cold": is_user_cold & is_item_cold,
    }
    C = int(n_cand[0].item()) if n_cand.numel() else 0
    print(f"\n  --- CHẨN ĐOÁN thứ hạng positive (C={C} candidate; "
          f"random -> median≈{C//2}, đáy={C-1}) ---")
    for name, mask in groups.items():
        n = int(mask.sum().item())
        if n == 0:
            print(f"    [{name}] n=0")
            continue
        r = rank[mask].float()
        q = torch.quantile(r, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        at_bottom = (r >= C - 1).float().mean().item()
        print(
            f"    [{name}] n={n} rank min={q[0]:.0f} p25={q[1]:.0f} median={q[2]:.0f} "
            f"p75={q[3]:.0f} max={q[4]:.0f} | ở đáy={at_bottom:.1%} "
            f"| ‖e_user‖={norm_user[mask].mean():.3f} ‖e_pos‖={norm_pos[mask].mean():.4f}"
        )


@torch.no_grad()
def evaluate(
    split: str,
    output_dir: Path,
    item_embed: ItemEmbedding,
    seq_model: SequenceModel,
    neg_sampler: NegativeSampler,
    retrieval_loss_fn: RetrievalLoss,
    num_negatives: int,
    batch_size: int,
    device: torch.device,
    use_caption: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
    max_batches: int | None = None,
    num_workers: int = 4,
    full_ranking: bool = False,
) -> None:
    """Đánh giá Recall/NDCG@K trên split (val/test), tách theo 4 nhóm cold-start — xem"""
    dataset = GenRecsysDataset(output_dir, split=split, max_seq_len=max_seq_len,
                               use_caption=use_caption)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    item_embed.eval()
    seq_model.eval()

    all_metrics: dict[str, list[torch.Tensor]] = {}
    all_user_cold, all_item_cold, all_user_lowhist = [], [], []
    all_n_u = []
    all_rank, all_norm_user, all_norm_pos, all_n_cand = [], [], [], []

    # FULL RANKING -- CHI dung duoc tren catalog NHO (Pure: 7,583 item).
    #
    # [CHOT 2026-10-05] Co nay GIU LAI vi no la bang chung cho bang 2x3 TREN PURE: Krichene &
    # Rendle (KDD 2020) chung minh sampled metric khong bao toan phat bieu so sanh ("A tot
    # hon B" co the LAT khi do full), nen bang 2x3 cua Pure can mot lan do full de xac nhan
    # thu tu.
    # Da chay: hr@10 full-ranking 0.0469 vs sampled 0.0469 => thu tu GIU.
    # Item embedding KHONG phu thuoc timestamp => nhung ca catalog MOT LAN, dung lai moi batch.
    cat_feats = catalog_features(dataset, device)
    with torch.no_grad():
        all_item_e = embed_catalog(item_embed, cat_feats, device)     # (num_items, d), 1 LAN
    if full_ranking:
        print(f"  [full-ranking] xep hang tren TOAN BO {all_item_e.shape[0]:,} item "
              f"(thay vi {num_negatives + 1} mau)")

    total = min(len(loader), max_batches) if max_batches else len(loader)
    for i, batch in enumerate(tqdm(loader, total=total, desc=f"eval[{split}]", unit="batch")):
        if max_batches is not None and i >= max_batches:
            break

        fwd = run_batch_forward(
            batch, cat_feats, item_embed, seq_model, neg_sampler,
            num_negatives, device, all_item_e=all_item_e,
        )
        e_u = fwd["e_user_eval"]
        if full_ranking:
            # Dat positive vao COT 0 nhu cand_e_i de compute_metrics_at_k khong doi:
            # cham diem toan catalog, roi hoan vi cot cua item dich ve dau.
            B_ = e_u.shape[0]
            cand = all_item_e.unsqueeze(0).expand(B_, -1, -1)
            eval_logit = retrieval_loss_fn.eval_logit(e_u, cand)      # (B, num_items)
            lbl = fwd["label_video_id"].to(eval_logit.device).view(-1, 1)
            pos_col = eval_logit.gather(1, lbl)
            eval_logit = eval_logit.scatter(1, lbl, eval_logit[:, :1])
            eval_logit = torch.cat([pos_col, eval_logit[:, 1:]], dim=1)
        else:
            eval_logit = retrieval_loss_fn.eval_logit(e_u, fwd["cand_e_i"])

        batch_metrics = compute_metrics_at_k(eval_logit)
        # [SUA 2026-10-14 — code review] Truoc day tinh LAI rank bang argsort KHONG
        # stable ngay ben canh loi goi compute_metrics_at_k (von da dung cong thuc
        # tie-safe) — lam 2 bo so lieu (headline metric vs diagnostics all_rank) LECH
        # NHAU tren chinh batch co tie. Dung chung ham compute_positive_rank.
        rank = compute_positive_rank(eval_logit)
        all_rank.append(rank.cpu())
        all_norm_user.append(fwd["e_user_eval"].norm(dim=-1).cpu())
        all_norm_pos.append(fwd["cand_e_i"][:, 0].norm(dim=-1).cpu())
        all_n_cand.append(torch.full_like(rank.cpu(), eval_logit.shape[1]))
        for k, v in batch_metrics.items():
            all_metrics.setdefault(k, []).append(v.cpu())
        all_user_cold.append(fwd["is_user_cold"])
        all_item_cold.append(fwd["is_item_cold"])
        all_user_lowhist.append(fwd["is_user_lowhistory"])
        all_n_u.append(fwd["n_u_at_pred"])

    per_sample_metrics = {k: torch.cat(v) for k, v in all_metrics.items()}
    is_user_cold = torch.cat(all_user_cold)
    is_item_cold = torch.cat(all_item_cold)

    is_user_lowhist = torch.cat(all_user_lowhist)
    _report_rank_diagnostic(
        torch.cat(all_rank), torch.cat(all_norm_user), torch.cat(all_norm_pos),
        torch.cat(all_n_cand), is_user_cold, is_item_cold,
    )
    _report_history_diagnostic(torch.cat(all_rank), torch.cat(all_n_u), torch.cat(all_n_cand))
    n_all = next(iter(per_sample_metrics.values())).shape[0]
    result_overall = aggregate_by_group(
        per_sample_metrics, {"all": torch.ones(n_all, dtype=torch.bool)})

    # Lat cat phu: cold user theo hai dinh nghia (strict holdout / few-shot).
    result_cold = aggregate_by_group(per_sample_metrics, {
        "warm_user": ~is_user_cold,
        "cold_user": is_user_cold,
        "lowhistory": is_user_lowhist,
    })
    print_eval_report(result_overall, result_cold=result_cold)

    # Lat cat cold DUY NHAT so sanh duoc: khop n_u (do 2026-09-29, xem eval.py).
    n_u_all = torch.cat(all_n_u)
    print_matched(matched_by_n_u(
        per_sample_metrics, is_user_cold, ~is_user_cold, n_u_all,
        name_a="cold", name_b="warm"), expect="b>a")

    item_embed.train()
    seq_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../preprocess_data/output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None,
                        help="[error.md #22] So batch EVAL toi da, TACH RIENG khoi "
                             "--max-steps-per-epoch (von chi gioi han batch TRAIN). "
                             "Mac dinh None = eval tren TOAN BO split.")
    parser.add_argument("--full-ranking", action="store_true",
                        help="Eval xep hang tren TOAN BO catalog thay vi 1 positive + "
                             "--num-negatives mau. Krichene & Rendle (KDD 2020) chung minh "
                             "sampled metric KHONG bao toan phat bieu so sanh: 'A tot hon B' "
                             "co the LAT khi do full (BERT4Rec vs SASRec la vi du that). "
                             "7583 item du nho de xep hang toan bo. SO SE TUT MANH — day la "
                             "so THAT va so duoc voi literature.")
    parser.add_argument("--no-qk", action="store_true",
                        help="ABLATION FuXi-beta (arXiv 2508.10615): bo han q.k, attention chi "
                             "con bias. Ho bao bo q.k TOT HON tren MovieLens — kiem tren KuaiRand.")
    parser.add_argument("--dynamic-ts", action="store_true",
                        help="W_ts DONG: bang tra vector dieu bien theo query (formula.md §4.1). "
                             "Mac dinh TINH (scalar moi bucket).")
    parser.add_argument("--t-base", type=float, default=0.07,
                        help="Nhiet do retrieval (thang COSINE). Neu --learnable-temp (mac dinh) "
                             "thi day la gia tri KHOI TAO; neu --fixed-temp thi la hang so.")
    parser.add_argument("--fixed-temp", action="store_true",
                        help="Nhiet do CO DINH o --t-base thay vi hoc duoc (ablation).")
    parser.add_argument("--static-delta", action="store_true",
                        help="delta TINH: dong bang delta_proj o zero => delta_h hang so qua moi "
                             "query. O A1/A3 cua bang 2x2 (formula.md §4.4 nhom A).")
    parser.add_argument("--no-amp", action="store_true", help="Tắt fp16 autocast + GradScaler (mặc định BẬT trên CUDA). Dùng khi nghi ngờ vấn đề độ chính xác")
    parser.add_argument("--eval-negatives", type=int, default=1024,
                        help="So candidate luc EVAL = 1 positive + (N-1) negative. KHAC "
                             "--num-negatives (512, dung cho sampled-softmax luc train: do "
                             "la ngan sach gradient, da do la du). 1024 la quy uoc nganh cho "
                             "catalog lon: full-ranking tren 32M item khong ai lam. CHU Y: "
                             "sampled metric KHONG bao toan phat bieu so sanh (Krichene & "
                             "Rendle KDD 2020) nen moi bang so sanh phai bao cao CUNG mot N.")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help="Do dai cua so lich su K. 256 la so cua Pure (p99=234). "
                             "Attention O((2K)^2): K=1024 ton 16x, K=2048 ton 64x so voi 256.")
    parser.add_argument("--no-random-window", action="store_true",
                        help="Tat cua so truot ngau nhien luc train (xem "
                             "dataset._window_start). Mac dinh BAT: cua so co dinh cat cung "
                             "vung 'xa', dung thu ma delta_h(x_q) can de hoc pham vi nhin.")
    parser.add_argument("--no-tag", action="store_true",
                        help="ABLATION (formula.md muc 1): bo nhanh 2 (tag) cua content "
                             "token — content token chi con GMU, khong con gi de hop nhat.")
    parser.add_argument("--save-every", type=int, default=None, help="Lưu checkpoint mỗi N step (ngoài lần lưu cuối mỗi epoch, vốn LUÔN chạy). Dùng khi session hay bị ngắt giữa chừng")
    parser.add_argument("--save-path", default=None, help="Đường dẫn checkpoint; mặc định <output-dir>/ckpt.pt. Trên Kaggle nên trỏ vào /kaggle/working (output-dir có thể read-only)")
    parser.add_argument("--resume", default=None, help="Nạp checkpoint và train TIẾP từ đúng step đã dừng (gồm cả optimizer state)")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker nạp dữ liệu (mặc định 4). Nghẽn là CPU chứ không phải GPU — đặt 0 để debug hoặc khi môi trường không cho fork")
    parser.add_argument("--eval-only", action="store_true", help="Không train, chỉ nạp checkpoint (--resume/--save-path) và chạy evaluate() — dùng để chạy lại chẩn đoán trên CÙNG model, ~8 phút thay vì train lại ~2 giờ")
    parser.add_argument("--no-caption", action="store_true",
                        help="ABLATION (formula.md §1): bo modality `caption` khoi GMU. "
                             "Dataset TU DONG tat khi thieu caption_embeddings.npy, nen co "
                             "nay chi can khi muon tat DU DA co file. GMU con 4 modality "
                             "(video_type, music_type, author, music) + nhanh tag.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                        help="Split để eval. 'val' (mặc định) dùng để TUNE và chọn nhánh "
                             "ablation. 'test' chỉ chạy ĐÚNG MỘT LẦN sau khi đã chốt cấu "
                             "hình — mọi số báo cáo phải lấy từ đây. Trước 2026-10-07 cả "
                             "hai chỗ gọi evaluate() đều hardcode 'val' nên test.npy chưa "
                             "bao giờ được chấm, và tune/report dùng chung một split.")
    parser.add_argument("--checkpoint", action="store_true", help="Gradient checkpointing: chậm ~30%%, tiết kiệm ~70%% VRAM. BẮT BUỘC cho nhánh có γ ở batch=256 trên T4 15GB (nếu không sẽ CUDA OOM ở loss.backward)")
    # [THEM 2026-10-13] error.md #20: lr/ranking_loss_weight la tham so cua train() tu
    # lau nhung chua bao gio co co CLI — chi doi duoc bang cach goi train() truc tiep
    # tu Python, khong qua dong lenh.
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5,
                        help="He so lambda nhan vao ranking loss truoc khi cong voi "
                             "retrieval loss (L = L_ret + lambda*L_rank)")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="[error.md #21] Max norm cho clip_grad_norm_, ap dung "
                             "SAU scaler.unscale_ (dung thang gradient that)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW decoupled weight decay cho ma tran/embedding (NormFace "
                             "1704.06369: logit cosine lam norm phinh). 0 = tat")
    parser.add_argument("--no-item-id", action="store_true",
                        help="Ablation: tat item-ID embedding, Phi_i = content THUAN (hanh vi truoc 2026-10-15)")
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine",
                        help="cosine = warmup tuyen tinh + cosine ve 0 (mac dinh); constant = LR co dinh (hanh vi truoc 2026-10-15)")
    parser.add_argument("--warmup-frac", type=float, default=0.05,
                        help="Ti le tong so step dung de warmup (mac dinh 5%%)")
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        eval_max_batches=args.eval_max_batches,
        amp=not args.no_amp,
        use_qk=not args.no_qk,
        dynamic_ts=args.dynamic_ts,
        static_delta=args.static_delta,
        t_base=args.t_base,
        learnable_temp=not args.fixed_temp,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        ranking_loss_weight=args.ranking_loss_weight,
        full_ranking=args.full_ranking,
        use_checkpoint=args.checkpoint,
        save_every=args.save_every,
        save_path=args.save_path,
        resume=args.resume,
        eval_only=args.eval_only,
        no_caption=args.no_caption,
        eval_split=args.eval_split,
        eval_negatives=args.eval_negatives,
        max_seq_len=args.max_seq_len,
        random_window=not args.no_random_window,
        no_tag=args.no_tag,
        no_item_id=args.no_item_id,
        num_workers=args.num_workers,
    )
