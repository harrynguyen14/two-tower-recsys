"""Training loop — nối toàn bộ module: Dataset -> ItemEmbedding -> SequenceModel ->
RetrievalLoss (+ RankingLoss phụ) -> optimizer step. LearnableThresholds tính mat_u/mat_i/
conf_content RUNTIME từ N_i/N_category THẬT (tra qua lookup_n_at_t_batch, Pass 1, ĐÚNG
THEO TIMESTAMP của từng token/candidate — KHÔNG leak tương lai, xem build_n_cumulative.py).

Retrieval loss: 1 batch = (chuỗi lịch sử user, label, N sample negative theo tần suất
thật trong log — xem negative_sampler.py).
Ranking loss: dùng label làm candidate positive, action_vector THẬT tại vị trí label
(dataset.py trả riêng field label_action, KHÁC hist_action là hành vi TRƯỚC label).

CHƯA CHIA model parallelism (item_id % 2 -> GPU0/GPU1, xem idea.md quyết định #1) — chạy
single-device trước, thêm khi có 2 GPU T4 thật để test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess_data"))
from build_n_cumulative import lookup_n_at_t_batch  # noqa: E402

from dataset import ACTION_VECTOR_FIELDS, GenRecsysDataset, ITEM_CATEGORICAL_FIELDS
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from learnable_thresholds import LearnableThresholds
from negative_sampler import NegativeSampler
from ranking_loss import BINARY_ACTION_FIELDS, RankingLoss
from retrieval import RetrievalLoss
from sequence_model import SequenceModel

# Index của mỗi field nhị phân trong ACTION_VECTOR_FIELDS — dùng để cắt nhãn cho RankingLoss
BINARY_ACTION_INDICES = [ACTION_VECTOR_FIELDS.index(f) for f in BINARY_ACTION_FIELDS]


def build_category_counts(item_static_path: Path) -> dict[str, int]:
    item_static = np.load(item_static_path, mmap_mode="r")
    return {field: int(item_static[field].max()) + 1 for field in ITEM_CATEGORICAL_FIELDS}


def item_features_to_device(features: dict, device: torch.device) -> dict:
    return {
        "category_ids": {k: v.to(device) for k, v in features["category_ids"].items()},
        "author_idx": features["author_idx"].to(device),
        "music_idx": features["music_idx"].to(device),
        "stat_features": features["stat_features"].to(device),
    }


def compute_n_i_n_category(
    dataset: GenRecsysDataset,
    video_ids_cpu: torch.Tensor,  # (N,) int64, CPU
    timestamps_cpu: torch.Tensor,  # (N,) int64, CPU
) -> tuple[torch.Tensor, torch.Tensor]:
    """N_i(video_id, t) qua item_N, N_category(category_id, t) qua category_N — CẢ HAI
    tính ĐÚNG THEO TIMESTAMP truyền vào (không leak tương lai, xem lookup_n_at_t_batch)."""
    video_ids_np = video_ids_cpu.numpy()
    ts_np = timestamps_cpu.numpy()
    n_i = lookup_n_at_t_batch("item_N", video_ids_np, ts_np)

    category_ids_np = dataset.get_category_ids(video_ids_cpu).numpy()
    n_category = lookup_n_at_t_batch("category_N", category_ids_np, ts_np)

    return torch.from_numpy(n_i.astype(np.float32)), torch.from_numpy(n_category.astype(np.float32))


def train(
    output_dir: str = "../preprocess_data/output",
    dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    ffn_dim: int = 256,
    batch_size: int = 64,
    num_negatives: int = 100,
    t_base: float = 0.1,
    lr: float = 1e-3,
    ranking_loss_weight: float = 0.5,
    num_epochs: int = 1,
    max_steps_per_epoch: int | None = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)
    device = torch.device(device_str)

    train_dataset = GenRecsysDataset(output_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_items = len(train_dataset.item_static)
    num_authors = int(train_dataset.item_static["author_idx"].max()) + 1
    num_music = int(train_dataset.item_static["music_idx"].max()) + 1
    num_categories = build_category_counts(output_dir / "item_static.npy")

    item_config = ItemEmbeddingConfig(
        num_items=num_items, num_authors=num_authors, num_music=num_music,
        num_categories=num_categories, dim=dim,
    )
    item_embed = ItemEmbedding(item_config).to(device)
    seq_model = SequenceModel(dim=dim, num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim).to(device)
    thresholds = LearnableThresholds().to(device)
    retrieval_loss_fn = RetrievalLoss(dim=dim, t_base=t_base).to(device)
    ranking_loss_fn = RankingLoss(dim=dim).to(device)
    neg_sampler = NegativeSampler(output_dir, num_items=num_items)

    # 3 bảng embedding lớn (collaborative/author/music, sparse=True) cần SparseAdam riêng —
    # Adam thường sẽ cấp phát optimizer state (exp_avg/exp_avg_sq) cho TOÀN BỘ bảng dù mỗi
    # batch chỉ chạm vài trăm dòng (đã xác nhận RuntimeError thật: cố cấp phát ~8.2GB cho
    # riêng collaborative_embedding 32M item x dim=64). Phần còn lại (category nhỏ, decoder,
    # loss heads, τ) dùng Adam thường.
    dense_params = (
        item_embed.dense_parameters()
        + list(seq_model.parameters())
        + list(thresholds.parameters())
        + list(retrieval_loss_fn.parameters())
        + list(ranking_loss_fn.parameters())
    )
    sparse_optimizer = torch.optim.SparseAdam(item_embed.sparse_parameters(), lr=lr)
    dense_optimizer = torch.optim.Adam(dense_params, lr=lr)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break

            hist_video_ids_cpu = batch["hist_video_ids"]  # (B, K) — GIỮ bản CPU để tra N_i
            hist_timestamps_cpu = batch["hist_timestamps"]  # (B, K)
            hist_action = batch["hist_action"].to(device)  # (B, K, 11)
            key_padding_mask = batch["key_padding_mask"].to(device)  # (B, K)
            label_video_id_cpu = batch["label_video_id"]  # (B,)
            label_timestamp_cpu = batch["label_timestamp"]  # (B,)
            label_action = batch["label_action"].to(device)  # (B, 11)
            mat_u_precomputed = batch["mat_u"].to(device)  # (B,) — τ_u_init cố định từ Pass 5

            B, K = hist_video_ids_cpu.shape

            # --- N_i/N_category THẬT theo timestamp cho mọi token lịch sử ---
            hist_n_i, hist_n_cat = compute_n_i_n_category(
                train_dataset, hist_video_ids_cpu.reshape(-1), hist_timestamps_cpu.reshape(-1)
            )
            hist_mat_i = thresholds.mat_i(hist_n_i.to(device))
            hist_conf_content = thresholds.conf_content(hist_n_cat.to(device))

            hist_features = item_features_to_device(
                train_dataset.get_item_features(hist_video_ids_cpu.reshape(-1)), device
            )
            hist_e_i = item_embed(
                hist_video_ids_cpu.reshape(-1).to(device), hist_features["category_ids"], hist_features["author_idx"],
                hist_features["music_idx"], hist_features["stat_features"], hist_mat_i, hist_conf_content,
            ).view(B, K, -1)

            h_u_all = seq_model(hist_e_i, hist_action, hist_mat_i.view(B, K), key_padding_mask)
            h_u = h_u_all[:, -1, :]  # (B, dim) — hidden state tại vị trí dự đoán (token cuối cùng, causal)

            # --- candidate set: 1 positive (label) + num_negatives negative theo tần suất ---
            neg_ids, neg_log_q = neg_sampler.sample(B, num_negatives, device)  # (B, num_negatives)
            candidate_ids_cpu = torch.cat([label_video_id_cpu.unsqueeze(1), neg_ids.cpu()], dim=1)  # (B, 1+neg)
            # Negative không có timestamp thật (chưa từng được sample tại thời điểm cụ thể nào) —
            # dùng CHUNG timestamp của label cho mọi candidate trong cùng hàng (đúng ngữ cảnh: N tại
            # đúng THỜI ĐIỂM đang dự đoán, kể cả cho item chưa từng liên quan tới sample này).
            candidate_ts_cpu = label_timestamp_cpu.unsqueeze(1).expand(-1, 1 + num_negatives)

            cand_n_i, cand_n_cat = compute_n_i_n_category(
                train_dataset, candidate_ids_cpu.reshape(-1), candidate_ts_cpu.reshape(-1)
            )
            cand_mat_i = thresholds.mat_i(cand_n_i.to(device))
            cand_conf_content = thresholds.conf_content(cand_n_cat.to(device))

            positive_log_q = neg_sampler.log_q_for(label_video_id_cpu.to(device)).unsqueeze(1)
            log_q = torch.cat([positive_log_q, neg_log_q], dim=1)

            cand_features = item_features_to_device(
                train_dataset.get_item_features(candidate_ids_cpu.reshape(-1)), device
            )
            cand_e_i = item_embed(
                candidate_ids_cpu.reshape(-1).to(device), cand_features["category_ids"], cand_features["author_idx"],
                cand_features["music_idx"], cand_features["stat_features"], cand_mat_i, cand_conf_content,
            ).view(B, 1 + num_negatives, -1)

            positive_idx = torch.zeros(B, dtype=torch.int64, device=device)  # positive luôn ở vị trí 0
            r_loss = retrieval_loss_fn(
                h_u, cand_e_i, log_q, mat_u_precomputed, cand_mat_i.view(B, 1 + num_negatives), positive_idx,
            )

            # --- ranking loss phụ: label làm candidate positive, action THẬT tại vị trí label ---
            binary_labels = label_action[:, BINARY_ACTION_INDICES]
            positive_e_i = cand_e_i[:, 0, :]
            k_loss, _ = ranking_loss_fn(h_u, positive_e_i, binary_labels)

            loss = r_loss + ranking_loss_weight * k_loss

            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

            if step % 50 == 0:
                snap = thresholds.get_tau_snapshot()
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"retrieval={r_loss.item():.4f} ranking={k_loss.item():.4f} "
                    f"tau_u={snap['tau_u']:.2f} tau_i={snap['tau_i']:.2f} tau_c={snap['tau_c']:.2f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../preprocess_data/output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
