"""Training loop — nối toàn bộ module: Dataset -> ItemEmbedding -> SequenceModel ->"""

from __future__ import annotations

import argparse
import os
import sys
import time
import os
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
from build_n_cumulative import build_n_cache, lookup_n_at_t_batch_cached

from dataset import ACTION_VECTOR_FIELDS, GenRecsysDataset, ITEM_CATEGORICAL_FIELDS, MAX_SEQ_LEN
from eval import (aggregate_by_group, classify_signal_position, compute_metrics_at_k,
                  print_eval_report)
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from learnable_thresholds import LearnableThresholds
from negative_sampler import NegativeSampler
from ranking_loss import BINARY_ACTION_FIELDS, RankingLoss
from retrieval import RetrievalLoss
from sequence_model import SequenceModel
from user_embedding import UserProfileConfig, UserProfileEmbedding

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


def compute_n_i_n_category(
    dataset: GenRecsysDataset,
    item_n_cache: dict,
    category_n_cache: dict,
    video_ids_cpu: torch.Tensor,
    timestamps_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """N_i(video_id, t) qua item_N, N_category(category_id, t) qua category_N — CẢ HAI"""
    video_ids_np = video_ids_cpu.numpy()
    ts_np = timestamps_cpu.numpy()
    n_i = lookup_n_at_t_batch_cached(item_n_cache, video_ids_np, ts_np)

    category_ids_np = dataset.get_category_ids(video_ids_cpu).numpy()
    n_category = lookup_n_at_t_batch_cached(category_n_cache, category_ids_np, ts_np)

    return torch.from_numpy(n_i.astype(np.float32)), torch.from_numpy(n_category.astype(np.float32))


def embed_items(
    video_ids_cpu: torch.Tensor,
    timestamps_cpu: torch.Tensor,
    dataset: GenRecsysDataset,
    item_embed: ItemEmbedding,
    thresholds: LearnableThresholds,
    item_n_cache: dict,
    category_n_cache: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tra N_i theo đúng timestamp, tính item_weight runtime (gradient chảy về τ_i), rồi"""
    n_i, _ = compute_n_i_n_category(dataset, item_n_cache, category_n_cache, video_ids_cpu, timestamps_cpu)
    item_weight = thresholds.item_weight(n_i.to(device, non_blocking=True))

    features = item_features_to_device(dataset.get_item_features(video_ids_cpu), device)
    e_i_final = item_embed(
        video_ids_cpu.to(device, non_blocking=True), features["category_ids"], features["author_idx"], features["music_idx"],
        features["caption_embedding"], features["caption_mask"], features["tag_ids"],
    )
    return e_i_final, item_weight


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
    dataset: GenRecsysDataset,
    item_embed: ItemEmbedding,
    seq_model: SequenceModel,
    profile_embed: UserProfileEmbedding,
    thresholds: LearnableThresholds,
    neg_sampler: NegativeSampler,
    item_n_cache: dict,
    category_n_cache: dict,
    num_negatives: int,
    device: torch.device,
    static_user_weight: bool = False,
    uniform_negatives: bool = False,
    log_user_maturity: bool = False,
    pmi_table: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """1 forward pass đầy đủ cho 1 batch — dùng CHUNG bởi train() và evaluate()."""
    hist_video_ids_cpu = batch["hist_video_ids"]
    hist_timestamps_cpu = batch["hist_timestamps"]
    hist_action = batch["hist_action"].to(device, non_blocking=True)
    key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
    hist_valid_mask = batch["hist_valid_mask"].to(device, non_blocking=True)
    label_video_id_cpu = batch["label_video_id"]
    label_timestamp_cpu = batch["label_timestamp"]
    label_action = batch["label_action"].to(device, non_blocking=True)

    B, K = hist_video_ids_cpu.shape

    hist_e_i, hist_item_weight = embed_items(
        hist_video_ids_cpu.reshape(-1), hist_timestamps_cpu.reshape(-1),
        dataset, item_embed, thresholds, item_n_cache, category_n_cache, device,
    )
    hist_e_i = hist_e_i.view(B, K, -1)
    hist_item_weight = hist_item_weight.view(B, K)

    hist_n_u = batch["hist_n_u"].to(device, non_blocking=True)
    if static_user_weight:
        hist_n_u = hist_n_u[:, -1:].expand_as(hist_n_u)
    hist_user_weight = thresholds.user_weight(hist_n_u)
    hist_log_u = thresholds.log_user_maturity(hist_n_u) if log_user_maturity else None

    label_e_i, _ = embed_items(
        label_video_id_cpu, label_timestamp_cpu, dataset, item_embed, thresholds,
        item_n_cache, category_n_cache, device,
    )
    _ie = getattr(item_embed, "module", item_embed)
    label_embed_stats = {k: v.clone() for k, v in _ie._last_stats.items()}

    user_features = dataset.get_user_features(batch["user_id"])
    e_profile = profile_embed(
        user_features["onehot"].to(device, non_blocking=True), user_features["register_days"].to(device, non_blocking=True)
    )

    hidden = seq_model(
        hist_e_i, hist_action, key_padding_mask, profile_embedding=e_profile,
        user_weight=hist_user_weight, item_weight=hist_item_weight,
        log_user_maturity=hist_log_u,
        hist_timestamps=hist_timestamps_cpu.to(device, non_blocking=True),
        hist_video_ids=hist_video_ids_cpu.to(device, non_blocking=True),
        pmi_table=pmi_table,
    )

    target_e_i = torch.cat([hist_e_i[:, 1:, :], label_e_i.unsqueeze(1)], dim=1)
    target_ids_cpu = torch.cat([hist_video_ids_cpu[:, 1:], label_video_id_cpu.unsqueeze(1)], dim=1)
    pair_valid = hist_valid_mask.clone()
    pair_valid[:, :-1] &= hist_valid_mask[:, 1:]

    neg_ids, neg_log_q = neg_sampler.sample(
        B, num_negatives, device, exclude=label_video_id_cpu, uniform=uniform_negatives,
    )
    neg_ts_cpu = label_timestamp_cpu.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
    neg_e_i, _ = embed_items(
        neg_ids.cpu().reshape(-1), neg_ts_cpu, dataset, item_embed, thresholds,
        item_n_cache, category_n_cache, device,
    )
    neg_e_i = neg_e_i.view(B, num_negatives, -1)
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
        "cand_e_i": cand_e_i,
        "log_q": log_q,
        "label_action": label_action,
        "is_user_cold": batch["is_user_cold"],
        "is_item_cold": batch["is_item_cold"],
        "label_embed_stats": label_embed_stats,
        "is_user_lowhistory": batch["is_user_lowhistory"],
        "hist_video_ids": hist_video_ids_cpu,
        "label_video_id": label_video_id_cpu,
        "n_u_at_pred": hist_n_u[:, -1].detach().cpu(),
        "m_saturated_frac": (
            (hist_item_weight > 0.95).float() * hist_valid_mask.float()
        ).sum(1).div(hist_valid_mask.float().sum(1).clamp(min=1)).detach().cpu(),
    }


def train(
    output_dir: str = "../preprocess_data/output",
    dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    ffn_dim: int = 256,
    batch_size: int = 64,
    num_negatives: int = 512,
    t_base: float = 0.1,
    lr: float = 1e-3,
    ranking_loss_weight: float = 0.5,
    num_epochs: int = 1,
    max_steps_per_epoch: int | None = None,
    amp: bool = True,
    use_softmax: bool = False,
    static_delta: bool = False,
    use_pmi: bool = True,
    use_profile_token: bool = True,
    interleave: bool = True,
    static_user_weight: bool = False,
    use_beta: bool = True,
    use_checkpoint: bool = False,
    save_every: int | None = None,
    save_path: str | None = None,
    resume: str | None = None,
    eval_only: bool = False,
    uniform_negatives: bool = False,
    log_user_maturity: bool = False,
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

    train_dataset = GenRecsysDataset(output_dir, split="train")
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
        num_categories=num_categories, dim=dim,
    )
    item_embed = ItemEmbedding(item_config).to(device, non_blocking=True)

    # Bảng PPMI (formula.md §4.1) — hằng số, KHÔNG có gradient, chỉ mu_h học được.
    # 115 MB fp16 nên nạp thẳng lên device. Chạy build_pmi.py nếu chưa có.
    pmi_table = None
    if use_pmi:
        pmi_path = Path(output_dir) / "pmi_table.npy"
        if pmi_path.exists():
            pmi_table = torch.from_numpy(np.load(pmi_path)).to(device)
            print(f"[train] nạp pmi_table {tuple(pmi_table.shape)} ({pmi_table.element_size() * pmi_table.nelement() / 1e6:.0f} MB)")
        else:
            print(f"[train] KHÔNG thấy {pmi_path} — chạy preprocess_data/build_pmi.py. Tắt PMI bias.")

    seq_model = SequenceModel(
        dim=dim, num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim,
        max_seq_len=(2 * MAX_SEQ_LEN if interleave else MAX_SEQ_LEN) + 1, interleave=interleave,
        use_beta=use_beta, use_checkpoint=use_checkpoint,
        use_softmax=use_softmax, static_delta=static_delta, use_pmi=use_pmi,
    ).to(device, non_blocking=True)
    profile_config = UserProfileConfig(
        onehot_num_categories=train_dataset.onehot_num_categories, dim=dim,
        use_profile_token=use_profile_token,
    )
    profile_embed = UserProfileEmbedding(profile_config).to(device, non_blocking=True)
    thresholds = LearnableThresholds().to(device, non_blocking=True)
    retrieval_loss_fn = RetrievalLoss(dim=dim, t_base=t_base).to(device, non_blocking=True)
    ranking_loss_fn = RankingLoss(dim=dim).to(device, non_blocking=True)
    neg_sampler = NegativeSampler(output_dir, num_items=num_items)

    dense_params = (
        item_embed.dense_parameters()
        + list(seq_model.parameters())
        + list(profile_embed.parameters())
        + list(thresholds.parameters())
        + list(retrieval_loss_fn.parameters())
        + list(ranking_loss_fn.parameters())
    )
    optimizer = torch.optim.Adam(dense_params, lr=lr, fused=(device.type == "cuda"))
    use_amp = (device.type == "cuda") and amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        item_embed_ddp = DDP(item_embed, device_ids=[local_rank], find_unused_parameters=True)
        seq_model_ddp = DDP(seq_model, device_ids=[local_rank], find_unused_parameters=True)
        profile_embed_ddp = DDP(profile_embed, device_ids=[local_rank], find_unused_parameters=True)
    else:
        item_embed_ddp, seq_model_ddp, profile_embed_ddp = item_embed, seq_model, profile_embed

    log("[train] building N_i/N_category cache (1 lần, dùng xuyên suốt training)...")

    item_n_cache = build_n_cache("item_N")
    category_n_cache = build_n_cache("category_N")

    steps_per_epoch = min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch else len(train_loader)

    ckpt_modules = {
        "item_embed": item_embed, "seq_model": seq_model, "profile_embed": profile_embed,
        "thresholds": thresholds, "retrieval_loss_fn": retrieval_loss_fn,
        "ranking_loss_fn": ranking_loss_fn,
    }
    ckpt_config = {
        "dim": dim, "num_heads": num_heads, "num_layers": num_layers, "ffn_dim": ffn_dim,
        "amp": amp, "use_softmax": use_softmax, "static_delta": static_delta,
        "use_pmi": use_pmi, "use_profile_token": use_profile_token,
        "interleave": interleave, "static_user_weight": static_user_weight,
        "log_user_maturity": log_user_maturity,
        "use_beta": use_beta,
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

    if eval_only:
        evaluate(
            "val", output_dir, item_embed, seq_model, profile_embed, thresholds, neg_sampler,
            retrieval_loss_fn, item_n_cache, category_n_cache, num_negatives, batch_size, device,
            max_batches=max_steps_per_epoch, static_user_weight=static_user_weight,
            num_workers=num_workers, uniform_negatives=uniform_negatives, pmi_table=pmi_table,
        )
        return

    forward_time_acc = 0.0
    step_time_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
            B = batch["hist_video_ids"].shape[0]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fwd = run_batch_forward(
                    batch, train_dataset, item_embed_ddp, seq_model_ddp, profile_embed_ddp,
                    thresholds, neg_sampler,
                    item_n_cache, category_n_cache, num_negatives, device,
                    static_user_weight=static_user_weight,
                    log_user_maturity=log_user_maturity, pmi_table=pmi_table,
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

            grad_norms = {}
            for pname in ("beta", "delta", "ts_w"):
                total = 0.0
                for layer in seq_model.decoder.layers:
                    g = getattr(layer.attn, pname).grad
                    if g is not None:
                        total += g.norm().item() ** 2
                grad_norms[pname] = total ** 0.5

            scaler.step(optimizer)
            scaler.update()
            step_time_acc += time.perf_counter() - step_start

            progress.set_postfix(loss=f"{loss.item():.4f}", retrieval=f"{r_loss.item():.4f}", ranking=f"{k_loss.item():.4f}")

            if is_main and save_every and step > 0 and step % save_every == 0:
                save_checkpoint(
                    ckpt_path, epoch=epoch, step=step + 1,
                    global_step=epoch * steps_per_epoch + step + 1, modules=ckpt_modules,
                    optimizer=optimizer, scaler=scaler,
                    config=ckpt_config,
                )
                tqdm.write(f"[ckpt] đã lưu {ckpt_path} (epoch={epoch} step={step + 1})")

            if step % 50 == 0:
                snap = thresholds.get_tau_snapshot()
                forward_pct = 100.0 * forward_time_acc / step_time_acc if step_time_acc > 0 else 0.0
                fwd_msg = f" | fwd%={forward_pct:.0f}"
                forward_time_acc = 0.0
                step_time_acc = 0.0
                with torch.no_grad():
                    vals = {}
                    for pname in ("beta", "delta", "ts_w"):
                        ps = [getattr(l.attn, pname).abs().mean().item() for l in seq_model.decoder.layers]
                        vals[pname] = sum(ps) / len(ps)
                tqdm.write(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"retrieval={r_loss.item():.4f} ranking={k_loss.item():.4f} "
                    f"tau_u={snap['tau_u']:.2f} tau_i={snap['tau_i']:.2f}"
                    f" | |b|={vals['beta']:.2e} |d|={vals['delta']:.2e}"
                    f" |ts|={vals['ts_w']:.2e}"
                    f" gb={grad_norms['beta']:.2e} gd={grad_norms['delta']:.2e}"
                    f" gts={grad_norms['ts_w']:.2e}"
                    f"{fwd_msg}"
                )

        if is_main:
            save_checkpoint(
                ckpt_path, epoch=epoch + 1, step=0,
                global_step=(epoch + 1) * steps_per_epoch, modules=ckpt_modules,
                optimizer=optimizer, scaler=scaler,
                config=ckpt_config,
            )
            print(f"[ckpt] đã lưu {ckpt_path} (hết epoch {epoch})")
        if ddp:
            torch.distributed.barrier()

        if not is_main:
            continue
        evaluate(
            "val", output_dir, item_embed, seq_model, profile_embed, thresholds, neg_sampler, retrieval_loss_fn,
            item_n_cache, category_n_cache, num_negatives, batch_size, device,
            max_batches=max_steps_per_epoch, static_user_weight=static_user_weight,
            num_workers=num_workers, log_user_maturity=log_user_maturity,
            pmi_table=pmi_table,
        )

    cleanup_ddp(ddp)


_CKPT_MODULES = ("item_embed", "seq_model", "profile_embed", "thresholds",
                 "retrieval_loss_fn", "ranking_loss_fn")


def cleanup_ddp(ddp: bool) -> None:
    """Đóng process group. Bỏ qua thì tiến trình có thể treo lúc thoát."""
    if ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def save_checkpoint(path: Path, *, epoch: int, step: int, global_step: int,
                    modules: dict, optimizer, scaler, config: dict) -> None:
    """Lưu đủ để train tiếp ĐÚNG chỗ đã dừng, không chỉ để eval."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "epoch": epoch, "step": step, "global_step": global_step, "config": config,
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
    """Nạp checkpoint; trả về {"epoch", "step", "global_step"} để train() chạy tiếp."""
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
            "global_step": blob.get("global_step", 0)}


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
    print("    là CONFOUND (user warm xem đồ ngách hơn -> bài khó hơn). Đọc bảng tiếp theo.")


def _report_msat_diagnostic(
    rank: torch.Tensor,
    m_sat: torch.Tensor,
    n_cand: torch.Tensor,
) -> None:
    """m_j BÃO HOÀ có hại không? — phép thử trước khi quyết định có sửa hay không."""
    print("\n  --- CHẨN ĐOÁN: m_j bão hoà có hại không? ---")
    C = int(n_cand.float().median().item())
    print(f"    m_j > 0.95 = vùng log(m_j) -> 0, β/γ mất tín hiệu; ngẫu nhiên -> median≈{C // 2}")
    print(f"    {'% token bão hoà':<20}{'n':>8}{'median rank':>13}{'top-10%':>10}")
    for lo, hi, ten in [(0.0, 0.15, "0-15%"), (0.15, 0.30, "15-30%"),
                        (0.30, 0.45, "30-45%"), (0.45, 1.01, "45%+")]:
        k = (m_sat >= lo) & (m_sat < hi)
        n = int(k.sum())
        if n == 0:
            continue
        r = rank[k].float()
        print(f"    {ten:<20}{n:>8}{r.median():>13.0f}{(r < C * 0.1).float().mean():>10.3f}")
    ra = torch.argsort(torch.argsort(m_sat.float())).float()
    rr = torch.argsort(torch.argsort(rank.float())).float()
    rho = torch.corrcoef(torch.stack([ra, rr]))[0, 1].item()
    print(f"    tương quan hạng (% bão hoà) <-> rank: rho={rho:+.4f}  "
          f"(>0 = bão hoà m_j HẠI; ≈0 = vô hại, ĐỪNG sửa)")


def _report_confound_diagnostic(
    rank: torch.Tensor,
    n_u: torch.Tensor,
    m_sat: torch.Tensor,
) -> None:
    """"Lịch sử dài gây hại" có THẬT không, hay chỉ là user warm gặp bài KHÓ hơn?"""
    def _rho(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() < 30:
            return float("nan")
        ra = torch.argsort(torch.argsort(a.float())).float()
        rb = torch.argsort(torch.argsort(b.float())).float()
        return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()

    print("\n  --- CHẨN ĐOÁN: 'lịch sử dài hại' là THẬT hay chỉ là ĐỘ KHÓ? ---")
    print(f"    rho tổng thể (N_u <-> rank) = {_rho(n_u, rank):+.4f}")
    print(f"    rho (N_u <-> m_sat)         = {_rho(n_u, m_sat):+.4f}  "
          f"(<0 = user warm xem đồ ÍT phổ biến -> bài khó hơn)")
    print(f"\n    Khống chế độ khó — rho(N_u <-> rank) TRONG từng tầng m_sat:")
    print(f"    {'tầng m_sat':<16}{'n':>9}{'rho':>10}{'median rank':>13}")
    for lo, hi, ten in [(0.0, 0.15, "0-15%"), (0.15, 0.30, "15-30%"),
                        (0.30, 0.45, "30-45%"), (0.45, 1.01, "45%+")]:
        k = (m_sat >= lo) & (m_sat < hi)
        n = int(k.sum())
        if n < 30:
            continue
        print(f"    {ten:<16}{n:>9}{_rho(n_u[k], rank[k]):>10.4f}{rank[k].float().median():>13.0f}")
    print("    -> rho sụp về ~0 trong mọi tầng = CONFOUND, không có gì để sửa;")
    print("       rho giữ nguyên = lịch sử dài hại thật, độc lập độ khó.")


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
    profile_embed: UserProfileEmbedding,
    thresholds: LearnableThresholds,
    neg_sampler: NegativeSampler,
    retrieval_loss_fn: RetrievalLoss,
    item_n_cache: dict,
    category_n_cache: dict,
    num_negatives: int,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
    static_user_weight: bool = False,
    num_workers: int = 4,
    uniform_negatives: bool = False,
    log_user_maturity: bool = False,
    pmi_table: torch.Tensor | None = None,
) -> None:
    """Đánh giá Recall/NDCG@K trên split (val/test), tách theo 4 nhóm cold-start — xem"""
    dataset = GenRecsysDataset(output_dir, split=split)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    item_embed.eval()
    seq_model.eval()
    profile_embed.eval()

    all_metrics: dict[str, list[torch.Tensor]] = {}
    all_user_cold, all_item_cold, all_user_lowhist = [], [], []
    all_hist_ids, all_label_id, all_valid = [], [], []

    # (num_items, num_tags) bool — dùng phân loại vị trí tín hiệu. Dựng 1 lần mỗi eval.
    tag_ids = torch.from_numpy(dataset.item_static["tag_ids"].astype("int64"))
    item_tag_table = torch.zeros(tag_ids.shape[0], int(tag_ids.max()) + 1, dtype=torch.bool)
    item_tag_table.scatter_(1, tag_ids, True)
    item_tag_table[:, 0] = False          # index 0 là padding, không phải tag thật
    all_n_u = []
    all_m_sat = []
    all_rank, all_norm_user, all_norm_pos, all_n_cand = [], [], [], []

    total = min(len(loader), max_batches) if max_batches else len(loader)
    for i, batch in enumerate(tqdm(loader, total=total, desc=f"eval[{split}]", unit="batch")):
        if max_batches is not None and i >= max_batches:
            break

        fwd = run_batch_forward(
            batch, dataset, item_embed, seq_model, profile_embed, thresholds, neg_sampler,
            item_n_cache, category_n_cache, num_negatives, device,
            static_user_weight=static_user_weight, uniform_negatives=uniform_negatives,
            log_user_maturity=log_user_maturity, pmi_table=pmi_table,
        )
        eval_logit = retrieval_loss_fn.eval_logit(fwd["e_user_eval"], fwd["cand_e_i"])

        batch_metrics = compute_metrics_at_k(eval_logit)
        order = torch.argsort(eval_logit, dim=1, descending=True)
        rank = (order == 0).float().argmax(dim=1)
        all_rank.append(rank.cpu())
        all_norm_user.append(fwd["e_user_eval"].norm(dim=-1).cpu())
        all_norm_pos.append(fwd["cand_e_i"][:, 0].norm(dim=-1).cpu())
        all_n_cand.append(torch.full_like(rank.cpu(), eval_logit.shape[1]))
        for k, v in batch_metrics.items():
            all_metrics.setdefault(k, []).append(v.cpu())
        all_user_cold.append(fwd["is_user_cold"])
        all_item_cold.append(fwd["is_item_cold"])
        all_user_lowhist.append(fwd["is_user_lowhistory"])
        all_hist_ids.append(fwd["hist_video_ids"])
        all_label_id.append(fwd["label_video_id"])
        all_valid.append(fwd["hist_valid_mask"].cpu())
        all_n_u.append(fwd["n_u_at_pred"])
        all_m_sat.append(fwd["m_saturated_frac"])

    per_sample_metrics = {k: torch.cat(v) for k, v in all_metrics.items()}
    is_user_cold = torch.cat(all_user_cold)
    is_item_cold = torch.cat(all_item_cold)

    is_user_lowhist = torch.cat(all_user_lowhist)
    _report_rank_diagnostic(
        torch.cat(all_rank), torch.cat(all_norm_user), torch.cat(all_norm_pos),
        torch.cat(all_n_cand), is_user_cold, is_item_cold,
    )
    _report_history_diagnostic(torch.cat(all_rank), torch.cat(all_n_u), torch.cat(all_n_cand))
    _report_msat_diagnostic(torch.cat(all_rank), torch.cat(all_m_sat), torch.cat(all_n_cand))
    _report_confound_diagnostic(torch.cat(all_rank), torch.cat(all_n_u), torch.cat(all_m_sat))
    n_all = next(iter(per_sample_metrics.values())).shape[0]
    result_overall = aggregate_by_group(
        per_sample_metrics, {"all": torch.ones(n_all, dtype=torch.bool)})

    # Truc CHINH: tag item dich nam o lich su gan / xa / ca hai / khong dau (formula.md §−1).
    signal_masks = classify_signal_position(
        torch.cat(all_hist_ids), torch.cat(all_label_id), torch.cat(all_valid), item_tag_table)
    result_signal = aggregate_by_group(per_sample_metrics, signal_masks)

    # Lat cat phu: cold user theo hai dinh nghia (strict holdout / few-shot).
    result_cold = aggregate_by_group(per_sample_metrics, {
        "warm_user": ~is_user_cold,
        "cold_user": is_user_cold,
        "lowhistory": is_user_lowhist,
    })
    print_eval_report(result_overall, thresholds.get_tau_snapshot(),
                      result_signal=result_signal, result_cold=result_cold)

    item_embed.train()
    seq_model.train()
    profile_embed.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../preprocess_data/output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--softmax-attn", action="store_true", help="ABLATION (formula.md §4.4): softmax thay cho sigmoid")
    parser.add_argument("--static-delta", action="store_true", help="ABLATION (formula.md §4.4): delta hằng số thay cho delta_h(x_q)")
    parser.add_argument("--no-pmi", action="store_true", help="ABLATION: tắt PMI bias")
    parser.add_argument("--no-amp", action="store_true", help="Tắt fp16 autocast + GradScaler (mặc định BẬT trên CUDA). Dùng khi nghi ngờ vấn đề độ chính xác")
    parser.add_argument("--no-profile-token", action="store_true", help="Không prepend e_profile làm token 0 của chuỗi — ablation (xem user_embedding.py)")
    parser.add_argument("--no-interleave", action="store_true", help="Cộng gộp item+action vào 1 token (chuỗi K) thay vì xen kẽ [Φ,a,Φ,a,...] (chuỗi 2K, đúng HSTU) — ablation")
    parser.add_argument("--static-user-weight", action="store_true", help="ABLATION #5: u_i TĨNH per-user (N_u tại điểm dự đoán, broadcast ra K vị trí) thay vì per-position. Thí nghiệm tách bạch đóng góp 'cold-start là đại lượng per-position' — xem run_batch_forward()")
    parser.add_argument("--no-beta", action="store_true", help="ABLATION #3: tắt số hạng β·log m_j trong attention bias")
    parser.add_argument("--save-every", type=int, default=None, help="Lưu checkpoint mỗi N step (ngoài lần lưu cuối mỗi epoch, vốn LUÔN chạy). Dùng khi session hay bị ngắt giữa chừng")
    parser.add_argument("--save-path", default=None, help="Đường dẫn checkpoint; mặc định <output-dir>/ckpt.pt. Trên Kaggle nên trỏ vào /kaggle/working (output-dir có thể read-only)")
    parser.add_argument("--resume", default=None, help="Nạp checkpoint và train TIẾP từ đúng step đã dừng (gồm cả optimizer state)")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker nạp dữ liệu (mặc định 4). Nghẽn là CPU chứ không phải GPU — đặt 0 để debug hoặc khi môi trường không cho fork")
    parser.add_argument("--eval-only", action="store_true", help="Không train, chỉ nạp checkpoint (--resume/--save-path) và chạy evaluate() — dùng để chạy lại chẩn đoán trên CÙNG model, ~8 phút thay vì train lại ~2 giờ")
    parser.add_argument("--log-user-maturity", action="store_true", help="[ĐÃ THỬ, CÓ HẠI — ĐỪNG BẬT] log1p(N_u)-log1p(τ_u) thay log(tanh(N_u/τ_u)) cho log_u. Bão hoà của cái cũ CÓ THẬT (|prod| yếu đi 1,000 lần ở user warm) nhưng KHÔNG phải nguyên nhân rank xấu: train lại cho warm_warm 0.2574->0.2175 (-16%%) và rho(N_u,rank) không nhúc nhích (+0.081->+0.0755, còn tệ hơn: +0.120). Nguyên nhân thật là CONFOUND độ khó — xem _report_confound_diagnostic(). Giữ lại để ablation")
    parser.add_argument("--uniform-negatives", action="store_true", help="CHẨN ĐOÁN [2026-09-21]: eval với candidate rút UNIFORM thay vì theo tần suất. Trả lời: item cold recall=0 vì embedding rác, hay vì luôn phải đấu 100 item warm? Chỉ dùng với --eval-only")
    parser.add_argument("--checkpoint", action="store_true", help="Gradient checkpointing: chậm ~30%%, tiết kiệm ~70%% VRAM. BẮT BUỘC cho nhánh có γ ở batch=256 trên T4 15GB (nếu không sẽ CUDA OOM ở loss.backward)")
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        amp=not args.no_amp,
        use_softmax=args.softmax_attn,
        static_delta=args.static_delta,
        use_pmi=not args.no_pmi,
        use_profile_token=not args.no_profile_token,
        interleave=not args.no_interleave,
        static_user_weight=args.static_user_weight,
        use_beta=not args.no_beta,
        use_checkpoint=args.checkpoint,
        save_every=args.save_every,
        save_path=args.save_path,
        resume=args.resume,
        eval_only=args.eval_only,
        uniform_negatives=args.uniform_negatives,
        log_user_maturity=args.log_user_maturity,
        num_workers=args.num_workers,
    )
