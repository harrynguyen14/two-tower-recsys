"""Training loop — nối toàn bộ module: Dataset -> ItemEmbedding -> SequenceModel ->
RetrievalLoss (+ RankingLoss phụ) -> optimizer step. LearnableThresholds tính
user_weight/item_weight/category_confidence RUNTIME từ N_i/N_category THẬT (tra qua
lookup_n_at_t_batch, Pass 1, ĐÚNG THEO TIMESTAMP của từng token/candidate — KHÔNG leak
tương lai, xem build_n_cumulative.py).

[SỬA 2026-09-13] Thiết kế lại toàn bộ theo result.md "CHECKLIST CUỐI CÙNG":
- Bỏ T(u,i) biến thiên (retrieval.py), bỏ λ·log(mat_j) (confidence_attention.py) — không
  còn truyền mat_u/mat_i/mat_j vào các module đó.
- [SỬA 2026-09-14] Gate user 3 thành phần đã BỎ HẲN: e_profile giờ là TOKEN 0 prepend vào
  chuỗi (xem user_embedding.py + sequence_model.py). Gate cũ không nhận gradient từ loss
  tự hồi quy toàn chuỗi — bug thật, không phải tối ưu hình thức.
- RankingLoss thiết kế lại theo target-aware cross-attention: candidate NỐI vào cuối
  chuỗi lịch sử (K -> K+1), decoder chạy 1 LẦN cho chuỗi K+1 — dùng CHUNG cho cả retrieval
  (đọc vị trí K-1, 0-indexed — không bị ảnh hưởng bởi token K+1 phía sau nhờ causal mask)
  và ranking (đọc vị trí K, chính là vị trí candidate) — TIẾT KIỆM 1 lần chạy decoder so
  với chạy riêng 2 lần.

Retrieval loss: 1 batch = (chuỗi lịch sử user, label, N sample negative theo tần suất
thật trong log — xem negative_sampler.py).
Ranking loss: dùng label làm candidate positive, action_vector THẬT tại vị trí label
(dataset.py trả riêng field label_action, KHÁC hist_action là hành vi TRƯỚC label).

CHƯA CHIA model parallelism (item_id % 2 -> GPU0/GPU1, xem idea.md quyết định #1) — chạy
single-device trước, thêm khi có 2 GPU T4 thật để test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# [SỬA 2026-09-15] Set GEN_RECSYS_OUT_DIR TRƯỚC khi import build_n_cumulative — module đó
# đọc env var ở cấp module (biến OUT_DIR), nên set sau khi import là VÔ TÁC DỤNG.
#
# Vì sao phải làm ở đây, trước cả argparse: build_n_cumulative.OUT_DIR trước đây hard-code
# thành thư mục cạnh file code, bỏ qua --output-dir hoàn toàn. Trên máy dev hai đường dẫn
# trùng nhau nên không lộ; trên Kaggle (code /kaggle/working, dữ liệu /kaggle/input) thì
# FileNotFoundError: item_N_ids.npy, cả 5 nhánh ablation cùng chết ở step 0.
# Đọc --output-dir bằng tay từ sys.argv vì argparse chỉ chạy ở __main__, sau import.
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

from dataset import ACTION_VECTOR_FIELDS, GenRecsysDataset, ITEM_CATEGORICAL_FIELDS, MAX_SEQ_LEN
from eval import aggregate_by_cold_group, compute_recall_ndcg_at_k, print_eval_report
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from learnable_thresholds import LearnableThresholds
from negative_sampler import NegativeSampler
from ranking_loss import BINARY_ACTION_FIELDS, RankingLoss
from retrieval import RetrievalLoss
from sequence_model import SequenceModel
from user_embedding import UserProfileConfig, UserProfileEmbedding

# Index của mỗi field nhị phân trong ACTION_VECTOR_FIELDS — dùng để cắt nhãn cho RankingLoss
BINARY_ACTION_INDICES = [ACTION_VECTOR_FIELDS.index(f) for f in BINARY_ACTION_FIELDS]


def reset_optimizer_state_for_evicted(item_embed: ItemEmbedding, sparse_optimizer: torch.optim.SparseAdam) -> None:
    """Reset SparseAdam state (exp_avg/exp_avg_sq) tại các slot vừa bị force-evict (LRU)
    trong step NÀY — xem cuckoo_embedding.py docstring: nếu không reset, ID mới chiếm slot
    sẽ "thừa kế" nhầm momentum của ID cũ đã bị đá khỏi hệ thống hẳn, học sai hướng ngay từ
    bước đầu. Gọi SAU sparse_optimizer.step() mỗi step.

    PHẢI dùng drain_pending_evicted_slots() (đọc + xóa), KHÔNG đọc trực tiếp
    .pending_evicted_slots — ItemEmbedding.forward gọi resolve() nhiều LẦN/step (hist +
    label + negatives) trên CÙNG 1 CuckooEmbedding instance; resolve() TÍCH LŨY, chỉ hàm
    này (gọi 1 LẦN/step, sau MỌI lần resolve) mới drain."""
    from cuckoo_embedding import CuckooEmbedding

    for table in (item_embed.collab_embedding, item_embed.author_embedding, item_embed.music_embedding):
        if not isinstance(table, CuckooEmbedding):
            continue
        for evicted_table, evicted_slot in table.drain_pending_evicted_slots():
            param = table.table_a if evicted_table == "a" else table.table_b
            state = sparse_optimizer.state.get(param)
            if state is None or "exp_avg" not in state:
                continue  # optimizer chưa từng step trên param này (batch đầu tiên) -> không có state để reset
            state["exp_avg"][evicted_slot].zero_()
            state["exp_avg_sq"][evicted_slot].zero_()


def build_category_counts(item_static_path: Path) -> dict[str, int]:
    item_static = np.load(item_static_path, mmap_mode="r")
    return {field: int(item_static[field].max()) + 1 for field in ITEM_CATEGORICAL_FIELDS}


def item_features_to_device(features: dict, device: torch.device) -> dict:
    return {
        "category_ids": {k: v.to(device) for k, v in features["category_ids"].items()},
        "author_idx": features["author_idx"].to(device),
        "music_idx": features["music_idx"].to(device),
        "caption_embedding": features["caption_embedding"].to(device),
        "caption_mask": features["caption_mask"].to(device),
    }


def compute_n_i_n_category(
    dataset: GenRecsysDataset,
    item_n_cache: dict,
    category_n_cache: dict,
    video_ids_cpu: torch.Tensor,  # (N,) int64, CPU
    timestamps_cpu: torch.Tensor,  # (N,) int64, CPU
) -> tuple[torch.Tensor, torch.Tensor]:
    """N_i(video_id, t) qua item_N, N_category(category_id, t) qua category_N — CẢ HAI
    tính ĐÚNG THEO TIMESTAMP truyền vào (không leak tương lai). Dùng cache đã build 1 lần
    (build_n_cache, xem build_n_cumulative.py) — KHÔNG gọi lookup_n_at_t_batch() thẳng vì
    hàm đó build lại flat_struct MỖI LẦN GỌI, đã xác nhận OOM thật khi gọi lặp lại nhiều
    lần/step trong training loop."""
    video_ids_np = video_ids_cpu.numpy()
    ts_np = timestamps_cpu.numpy()
    n_i = lookup_n_at_t_batch_cached(item_n_cache, video_ids_np, ts_np)

    category_ids_np = dataset.get_category_ids(video_ids_cpu).numpy()
    n_category = lookup_n_at_t_batch_cached(category_n_cache, category_ids_np, ts_np)

    return torch.from_numpy(n_i.astype(np.float32)), torch.from_numpy(n_category.astype(np.float32))


def embed_items(
    video_ids_cpu: torch.Tensor,  # (N,) CPU, flatten
    timestamps_cpu: torch.Tensor,  # (N,) CPU, flatten
    dataset: GenRecsysDataset,
    item_embed: ItemEmbedding,
    thresholds: LearnableThresholds,
    item_n_cache: dict,
    category_n_cache: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tra N_i/N_category theo đúng timestamp, tính item_weight/category_confidence
    runtime (có gradient chảy về τ_i/τ_c), rồi chạy ItemEmbedding — dùng CHUNG cho token
    lịch sử, candidate label, và negative (giữ logic 1 chỗ, tránh trùng lặp)."""
    n_i, n_cat = compute_n_i_n_category(dataset, item_n_cache, category_n_cache, video_ids_cpu, timestamps_cpu)
    item_weight = thresholds.item_weight(n_i.to(device))
    category_confidence = thresholds.category_confidence(n_cat.to(device))

    features = item_features_to_device(dataset.get_item_features(video_ids_cpu), device)
    e_i_final = item_embed(
        video_ids_cpu.to(device), features["category_ids"], features["author_idx"], features["music_idx"],
        item_weight, category_confidence,
        features["caption_embedding"], features["caption_mask"],
    )
    return e_i_final, item_weight


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
    static_user_weight: bool = False,  # ablation #5: u_i tĩnh per-user thay vì per-position
) -> dict[str, torch.Tensor]:
    """1 forward pass đầy đủ cho 1 batch — dùng CHUNG bởi train() và evaluate().

    [SỬA 2026-09-14] Thiết kế lại theo HSTU thật (arXiv 2402.17152):
    - Chuỗi XEN KẼ [Φ_0,a_0,…,Φ_{K-1},a_{K-1}] thay cho cộng gộp; BỎ việc nối candidate
      vào cuối (thiết kế K+1 cũ) — xen kẽ đã cho target-aware mà KHÔNG leak action.
    - Loss TỰ HỒI QUY toàn chuỗi: dự đoán item t+1 tại MỌI vị trí (retrieval) và a_t tại
      MỌI vị trí (ranking) — tín hiệu ×K so với thiết kế cũ chỉ dùng 1 vị trí, cùng 1
      forward pass (hidden mọi vị trí vốn đã tính, trước đây vứt đi 255/256).

    Trả về (xem chú thích từng key ở cuối hàm): pred/target_e_i/target_log_q/neg_e_i/
    neg_log_q/pair_valid (retrieval toàn chuỗi), hist_action/hist_valid_mask (ranking toàn
    chuỗi), e_user_eval/cand_e_i/log_q (eval), label_action, is_user_cold, is_item_cold."""
    hist_video_ids_cpu = batch["hist_video_ids"]
    hist_timestamps_cpu = batch["hist_timestamps"]
    hist_action = batch["hist_action"].to(device)
    key_padding_mask = batch["key_padding_mask"].to(device)
    hist_valid_mask = batch["hist_valid_mask"].to(device)  # (B, K) True = token thật
    label_video_id_cpu = batch["label_video_id"]
    label_timestamp_cpu = batch["label_timestamp"]
    label_action = batch["label_action"].to(device)

    B, K = hist_video_ids_cpu.shape

    hist_e_i, hist_item_weight = embed_items(
        hist_video_ids_cpu.reshape(-1), hist_timestamps_cpu.reshape(-1),
        dataset, item_embed, thresholds, item_n_cache, category_n_cache, device,
    )
    hist_e_i = hist_e_i.view(B, K, -1)
    hist_item_weight = hist_item_weight.view(B, K)  # m_j cho pairwise bias (confidence_attention.py)

    # u_i TẠI TỪNG VỊ TRÍ (không phải 1 scalar/chuỗi như trước) — điều kiện CHẶN để số hạng
    # γ·log(u_i)·log(m_j) không thoái hóa về đúng cơ chế λ·log(mat_j) đã bỏ 2026-09-13.
    # Xem dataset.py hist_n_u + confidence_attention.py docstring.
    hist_n_u = batch["hist_n_u"].to(device)  # (B, K) — N_u tại TỪNG vị trí
    if static_user_weight:
        # [ABLATION #5, THÊM 2026-09-15] Thay u_i per-position bằng u TĨNH per-user: lấy
        # N_u tại điểm dự đoán (vị trí cuối) rồi broadcast ra cả K vị trí. Đây là thí
        # nghiệm DUY NHẤT tách bạch đóng góp "cold-start là đại lượng per-position" khỏi
        # đóng góp "bias dạng tích": cả hai nhánh đều có γ·log(u)·log(m_j), chỉ khác u
        # biến thiên hay hằng. Nếu nhánh tĩnh ngang nhánh per-position -> luận điểm
        # per-position KHÔNG có cơ sở thực nghiệm và phải rút khỏi paper.
        hist_n_u = hist_n_u[:, -1:].expand_as(hist_n_u)
    hist_user_weight = thresholds.user_weight(hist_n_u)  # (B, K)

    label_e_i, _ = embed_items(
        label_video_id_cpu, label_timestamp_cpu, dataset, item_embed, thresholds,
        item_n_cache, category_n_cache, device,
    )
    # [CHẨN ĐOÁN 2026-09-16] Chụp NGAY SAU lần embed label — _last_stats bị ghi đè ở mỗi
    # lần gọi embed_items (history/label/negative), nên phải lấy đúng ở đây mới là stats của
    # candidate POSITIVE, thứ ta cần đối chiếu với thứ hạng của nó.
    label_embed_stats = {k: v.clone() for k, v in item_embed._last_stats.items()}

    # --- 1 lần chạy decoder trên chuỗi XEN KẼ [Φ_0,a_0,...,Φ_{K-1},a_{K-1}] ---
    # KHÔNG nối label vào chuỗi nữa (bỏ thiết kế K+1 cũ): xen kẽ khiến hidden tại Φ_t đã
    # thấy (Φ_0..Φ_t, a_0..a_{t-1}) nhưng CHƯA thấy a_t — đúng thứ cần cho cả 2 việc, và
    # tự loại bỏ leak label_action của thiết kế cũ (đã xác nhận bằng test causal, xem
    # sequence_model.py forward docstring).
    # [SỬA 2026-09-14] e_profile PREPEND làm token 0 của chuỗi (thay nhánh user gate cũ —
    # xem user_embedding.py docstring: gate cũ không nhận gradient từ loss toàn chuỗi).
    user_features = dataset.get_user_features(batch["user_id"])
    e_profile = profile_embed(
        user_features["onehot"].to(device), user_features["register_days"].to(device)
    )  # (B, dim), hoặc None nếu use_profile_token=False (ablation)

    hidden = seq_model(
        hist_e_i, hist_action, key_padding_mask, profile_embedding=e_profile,
        user_weight=hist_user_weight, item_weight=hist_item_weight,
    )  # (B, K, dim) tại vị trí item

    # --- Retrieval: loss TỰ HỒI QUY toàn chuỗi (mọi vị trí dự đoán item kế tiếp) ---
    # target[t] = item t+1: item trong window dịch trái 1, vị trí cuối là LABEL.
    target_e_i = torch.cat([hist_e_i[:, 1:, :], label_e_i.unsqueeze(1)], dim=1)  # (B, K, dim)
    target_ids_cpu = torch.cat([hist_video_ids_cpu[:, 1:], label_video_id_cpu.unsqueeze(1)], dim=1)
    # Cặp (t -> t+1) hợp lệ khi t thật VÀ t+1 thật; vị trí cuối dự đoán label (luôn thật)
    # nên chỉ cần t thật.
    pair_valid = hist_valid_mask.clone()
    pair_valid[:, :-1] &= hist_valid_mask[:, 1:]

    neg_ids, neg_log_q = neg_sampler.sample(B, num_negatives, device, exclude=label_video_id_cpu)
    neg_ts_cpu = label_timestamp_cpu.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
    neg_e_i, _ = embed_items(
        neg_ids.cpu().reshape(-1), neg_ts_cpu, dataset, item_embed, thresholds,
        item_n_cache, category_n_cache, device,
    )
    neg_e_i = neg_e_i.view(B, num_negatives, -1)
    target_log_q = neg_sampler.log_q_for(target_ids_cpu.reshape(-1).to(device)).view(B, K)

    # --- Biểu diễn user tại vị trí dự đoán CUỐI (= label), dùng cho eval ---
    # [SỬA 2026-09-14] Là hidden THUẦN từ decoder, KHÔNG qua gate nào nữa: e_profile đã nằm
    # trong chuỗi (token 0) nên hidden tại Φ_{K-1} đã "thấy" nó qua attention. Gate ngoài
    # giờ vừa thừa (đếm e_profile 2 lần) vừa sai (train chấm trên hidden, eval chấm trên
    # e_u_final -> 2 đại lượng KHÁC NHAU, đúng bug đã phát hiện).
    e_user_eval = hidden[:, -1, :]  # hidden tại Φ_{K-1} — chưa thấy a_{K-1}, dự đoán label

    # Candidate set cho EVAL (xếp hạng label giữa positive + negative) — giữ đúng cách đo
    # cũ để so sánh được với baseline.
    cand_e_i = torch.cat([label_e_i.unsqueeze(1), neg_e_i], dim=1)  # (B, 1+num_negatives, dim)
    positive_log_q = neg_sampler.log_q_for(label_video_id_cpu.to(device)).unsqueeze(1)
    log_q = torch.cat([positive_log_q, neg_log_q], dim=1)

    return {
        # loss retrieval toàn chuỗi
        "pred": hidden,  # (B, K, dim) — hidden tại Φ_t
        "target_e_i": target_e_i,  # (B, K, dim) — e_i_final của item t+1
        "target_log_q": target_log_q,  # (B, K)
        "neg_e_i": neg_e_i,  # (B, C, dim)
        "neg_log_q": neg_log_q,  # (B, C)
        "pair_valid": pair_valid,  # (B, K) bool
        # ranking p(a_t | Φ_t) — hidden tại Φ_t CHƯA thấy a_t nên KHÔNG leak
        "hist_action": hist_action,  # (B, K, NUM_ACTION_DIMS) — nhãn action tại mọi vị trí
        "hist_valid_mask": hist_valid_mask,  # (B, K)
        # eval
        "e_user_eval": e_user_eval,
        "cand_e_i": cand_e_i,
        "log_q": log_q,
        "label_action": label_action,
        "is_user_cold": batch["is_user_cold"],
        "is_item_cold": batch["is_item_cold"],
        "label_embed_stats": label_embed_stats,  # chẩn đoán gate, xem item_embedding.py
        "is_user_lowhistory": batch["is_user_lowhistory"],  # few-shot, xem eval.py print_eval_report
    }


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
    use_cuckoo_embedding: bool = True,
    cuckoo_capacity_ratio: float = 0.25,
    use_profile_token: bool = True,  # prepend e_profile làm token 0 (xem user_embedding.py)
    interleave: bool = True,  # True = chuỗi xen kẽ [Φ,a,Φ,a,...] (HSTU); False = cộng gộp (ablation)
    static_user_weight: bool = False,  # ablation #5: u_i tĩnh per-user (xem run_batch_forward)
    use_beta: bool = True,   # ablation #3: tắt β·log m_j (xem confidence_attention.py)
    use_gamma: bool = True,  # ablation #4: tắt γ·log u_i·log m_j — số hạng TÍCH, đóng góp chính
    use_checkpoint: bool = False,  # gradient checkpointing — BẮT BUỘC cho nhánh có γ ở batch lớn
    save_every: int | None = None,   # lưu checkpoint mỗi N step (None = chỉ lưu cuối epoch)
    save_path: str | None = None,    # nơi lưu; mặc định <output_dir>/ckpt.pt
    resume: str | None = None,       # nạp checkpoint rồi train tiếp từ đúng step đã dừng
    eval_only: bool = False,         # bỏ qua train, chỉ nạp checkpoint và chạy evaluate()
    num_workers: int = 4,            # worker nạp dữ liệu — xem DataLoader dưới (nghẽn là CPU, không phải GPU)
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)
    device = torch.device(device_str)

    train_dataset = GenRecsysDataset(output_dir, split="train")
    # [SỬA 2026-09-16] num_workers=4 + pin_memory: step đo được 1.59 s nhưng compute GPU chỉ
    # ~125 ms (bench_t4.py, batch=256) và peak VRAM 1.73/15.6 GB — nghẽn nằm ở CPU, trong
    # compute_n_i_n_category() (tra N_i/N_category theo timestamp, .numpy()/from_numpy mỗi
    # step). Với num_workers=0 phần đó chạy tuần tự, GPU ngồi chờ. Worker process làm nó
    # chồng lấn với forward/backward.
    # Dataset dùng memmap read-only (dataset.py) nên fork sang worker an toàn: mỗi worker tự
    # map lại, không chia sẻ file handle.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,  # không dựng lại 4 process mỗi epoch
    )

    num_items = len(train_dataset.item_static)
    num_authors = int(train_dataset.item_static["author_idx"].max()) + 1
    num_music = int(train_dataset.item_static["music_idx"].max()) + 1
    num_categories = build_category_counts(output_dir / "item_static.npy")

    item_config = ItemEmbeddingConfig(
        num_items=num_items, num_authors=num_authors, num_music=num_music,
        num_categories=num_categories, dim=dim,
        use_cuckoo_embedding=use_cuckoo_embedding, cuckoo_capacity_ratio=cuckoo_capacity_ratio,
    )
    item_embed = ItemEmbedding(item_config).to(device)
    # [SỬA 2026-09-14] Xen kẽ: chuỗi nội bộ 2K token cho K=256 lượt -> max_seq_len=512.
    # Đo thật trên T4 (bench_t4.py, fp16, batch=256): 125.3 ms/step = 0.15 h/epoch, peak
    # 1.73/15.6 GB — compute KHÔNG phải ràng buộc. Bỏ thiết kế K+1 cũ (nối candidate vào
    # cuối) vì xen kẽ đã cho target-aware mà không leak action.
    seq_model = SequenceModel(
        dim=dim, num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim,
        # +1 cho token profile prepend (xem sequence_model.py forward) — thiếu 1 slot ở đây
        # là IndexError trong position_embedding ngay step đầu.
        max_seq_len=(2 * MAX_SEQ_LEN if interleave else MAX_SEQ_LEN) + 1, interleave=interleave,
        use_beta=use_beta, use_gamma=use_gamma, use_checkpoint=use_checkpoint,
    ).to(device)
    profile_config = UserProfileConfig(
        onehot_num_categories=train_dataset.onehot_num_categories, dim=dim,
        use_profile_token=use_profile_token,
    )
    profile_embed = UserProfileEmbedding(profile_config).to(device)
    thresholds = LearnableThresholds().to(device)
    retrieval_loss_fn = RetrievalLoss(dim=dim, t_base=t_base).to(device)
    ranking_loss_fn = RankingLoss(dim=dim).to(device)
    neg_sampler = NegativeSampler(output_dir, num_items=num_items)

    # 3 bảng embedding lớn (collaborative/author/music, sparse=True) cần SparseAdam riêng —
    # Adam thường sẽ cấp phát optimizer state (exp_avg/exp_avg_sq) cho TOÀN BỘ bảng dù mỗi
    # batch chỉ chạm vài trăm dòng. Phần còn lại (category nhỏ, decoder, loss heads, τ)
    # dùng Adam thường.
    dense_params = (
        item_embed.dense_parameters()
        + list(seq_model.parameters())
        + list(profile_embed.parameters())
        + list(thresholds.parameters())
        + list(retrieval_loss_fn.parameters())
        + list(ranking_loss_fn.parameters())
    )
    sparse_optimizer = torch.optim.SparseAdam(item_embed.sparse_parameters(), lr=lr)
    dense_optimizer = torch.optim.Adam(dense_params, lr=lr)

    # Build 1 LẦN DUY NHẤT trước vòng lặp — KHÔNG build lại mỗi step.
    print("[train] building N_i/N_category cache (1 lần, dùng xuyên suốt training)...")
    item_n_cache = build_n_cache("item_N")
    category_n_cache = build_n_cache("category_N")

    steps_per_epoch = min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch else len(train_loader)

    # Gom module vào 1 dict cho save/load — tránh lặp 6 tên ở 3 chỗ gọi khác nhau.
    ckpt_modules = {
        "item_embed": item_embed, "seq_model": seq_model, "profile_embed": profile_embed,
        "thresholds": thresholds, "retrieval_loss_fn": retrieval_loss_fn,
        "ranking_loss_fn": ranking_loss_fn,
    }
    # Cờ ĐỔI CƠ CHẾ model — lưu kèm để resume/eval-only không âm thầm chạy nhánh ablation
    # khác với lúc train. Không gồm lr/batch_size: đổi chúng vẫn là cùng model, hợp lệ.
    ckpt_config = {
        "dim": dim, "num_heads": num_heads, "num_layers": num_layers, "ffn_dim": ffn_dim,
        "use_cuckoo_embedding": use_cuckoo_embedding, "use_profile_token": use_profile_token,
        "interleave": interleave, "static_user_weight": static_user_weight,
        "use_beta": use_beta, "use_gamma": use_gamma,
    }
    ckpt_path = Path(save_path) if save_path else output_dir / "ckpt.pt"

    start_epoch, start_step = 0, 0
    if resume or eval_only:
        src = resume or save_path or str(ckpt_path)
        state = load_checkpoint(
            src, modules=ckpt_modules, config=ckpt_config,
            sparse_optimizer=None if eval_only else sparse_optimizer,
            dense_optimizer=None if eval_only else dense_optimizer,
        )
        start_epoch, start_step = state["epoch"], state["step"]
        print(f"[resume] nạp {src} — epoch={start_epoch} step={start_step}")

    if eval_only:
        # Chỉ đo lại, không train: dùng khi cần chạy chẩn đoán nhiều lần trên CÙNG model
        # (eval ~8 phút so với ~2 giờ train lại — và train lại còn ra model KHÁC vì seed
        # dataloader khác, nên số đo sẽ không so sánh được với lần trước).
        evaluate(
            "val", output_dir, item_embed, seq_model, profile_embed, thresholds, neg_sampler,
            retrieval_loss_fn, item_n_cache, category_n_cache, num_negatives, batch_size, device,
            max_batches=max_steps_per_epoch, static_user_weight=static_user_weight,
            num_workers=num_workers,
        )
        return

    forward_time_acc = 0.0
    step_time_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        progress = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"epoch {epoch}", unit="step")
        for step, batch in progress:
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            # Resume giữa epoch: bỏ qua các step ĐÃ train. Quay vòng dataloader vẫn tốn
            # thời gian đọc nhưng không forward/backward — chấp nhận được, và giữ thứ tự
            # batch nhất quán với lần chạy trước (shuffle dùng cùng seed torch mặc định).
            if epoch == start_epoch and step < start_step:
                continue

            step_start = time.perf_counter()
            B = batch["hist_video_ids"].shape[0]
            fwd = run_batch_forward(
                batch, train_dataset, item_embed, seq_model, profile_embed, thresholds, neg_sampler,
                item_n_cache, category_n_cache, num_negatives, device,
                static_user_weight=static_user_weight,
            )
            forward_time_acc += time.perf_counter() - step_start

            # --- retrieval TỰ HỒI QUY toàn chuỗi: dự đoán item t+1 tại MỌI vị trí ---
            r_loss = retrieval_loss_fn.forward_sequence(
                fwd["pred"], fwd["target_e_i"], fwd["neg_e_i"],
                fwd["target_log_q"], fwd["neg_log_q"], fwd["pair_valid"],
            )

            # --- ranking p(a_t | Φ_t) tại MỌI vị trí — không leak nhờ chuỗi xen kẽ ---
            binary_labels = fwd["hist_action"][..., BINARY_ACTION_INDICES]  # (B, K, 8)
            k_loss, _ = ranking_loss_fn.forward_sequence(
                fwd["pred"], binary_labels, fwd["hist_valid_mask"],
            )

            loss = r_loss + ranking_loss_weight * k_loss

            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss.backward()

            # [THÊM 2026-09-15] Đo ‖∇β‖/‖∇γ‖/‖∇δ‖ NGAY SAU backward, TRƯỚC step() — sau
            # step() gradient vẫn còn nhưng đã bị optimizer dùng, và zero_grad() đầu vòng
            # sau sẽ xóa. Đây là phép đo QUYẾT ĐỊNH của cả nghiên cứu: λ cũ chết vì gradient
            # ~1.1e-17 (log(m)≈0 trên dữ liệu warm). Nếu ∇γ cũng ở bậc 1e-15 thì γ chết y
            # hệt và phải dừng, không xây tiếp. Cộng ‖·‖ qua MỌI layer vì mỗi layer có β/γ/δ
            # riêng — nhìn 1 layer có thể bỏ sót layer khác đang học.
            grad_norms = {}
            for pname in ("beta", "gamma", "delta"):
                total = 0.0
                for layer in seq_model.decoder.layers:
                    g = getattr(layer.attn, pname).grad
                    if g is not None:
                        total += g.norm().item() ** 2
                grad_norms[pname] = total ** 0.5

            sparse_optimizer.step()
            dense_optimizer.step()
            reset_optimizer_state_for_evicted(item_embed, sparse_optimizer)
            step_time_acc += time.perf_counter() - step_start

            progress.set_postfix(loss=f"{loss.item():.4f}", retrieval=f"{r_loss.item():.4f}", ranking=f"{k_loss.item():.4f}")

            if save_every and step > 0 and step % save_every == 0:
                save_checkpoint(
                    ckpt_path, epoch=epoch, step=step + 1,  # +1: step này ĐÃ xong
                    global_step=epoch * steps_per_epoch + step + 1, modules=ckpt_modules,
                    sparse_optimizer=sparse_optimizer, dense_optimizer=dense_optimizer,
                    config=ckpt_config,
                )
                tqdm.write(f"[ckpt] đã lưu {ckpt_path} (epoch={epoch} step={step + 1})")

            if step % 50 == 0:
                snap = thresholds.get_tau_snapshot()
                cuckoo_msg = ""
                if item_config.use_cuckoo_embedding:
                    forward_pct = 100.0 * forward_time_acc / step_time_acc if step_time_acc > 0 else 0.0
                    cuckoo_msg = (
                        f" | cuckoo[item] load={item_embed.collab_embedding.load_factor():.2f} "
                        f"evict={item_embed.collab_embedding.num_evictions} "
                        f"forward%={forward_pct:.0f}"
                    )
                    forward_time_acc = 0.0
                    step_time_acc = 0.0
                # Giá trị |β|/|γ|/|δ| trung bình qua mọi head & layer — cặp với grad norm:
                # grad cho biết "có tín hiệu học không", giá trị cho biết "đã học được gì
                # chưa". γ→0 kèm ∇γ→0 = chết (như λ cũ); γ→0 kèm ∇γ lớn = chưa hội tụ.
                with torch.no_grad():
                    vals = {}
                    for pname in ("beta", "gamma", "delta"):
                        ps = [getattr(l.attn, pname).abs().mean().item() for l in seq_model.decoder.layers]
                        vals[pname] = sum(ps) / len(ps)
                tqdm.write(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"retrieval={r_loss.item():.4f} ranking={k_loss.item():.4f} "
                    f"tau_u={snap['tau_u']:.2f} tau_i={snap['tau_i']:.2f} tau_c={snap['tau_c']:.2f}"
                    f" | |b|={vals['beta']:.2e} |g|={vals['gamma']:.2e} |d|={vals['delta']:.2e}"
                    f" gb={grad_norms['beta']:.2e} gg={grad_norms['gamma']:.2e} gd={grad_norms['delta']:.2e}"
                    f"{cuckoo_msg}"
                )

        # Lưu CUỐI epoch không phụ thuộc --save-every: đây là checkpoint ta thật sự muốn
        # giữ (train xong đủ số step), và là thứ --eval-only sẽ nạp lại.
        save_checkpoint(
            ckpt_path, epoch=epoch + 1, step=0,
            global_step=(epoch + 1) * steps_per_epoch, modules=ckpt_modules,
            sparse_optimizer=sparse_optimizer, dense_optimizer=dense_optimizer,
            config=ckpt_config,
        )
        print(f"[ckpt] đã lưu {ckpt_path} (hết epoch {epoch})")

        # --- eval trên val sau MỖI epoch — Recall/NDCG@K tách theo 4 nhóm cold-start ---
        evaluate(
            "val", output_dir, item_embed, seq_model, profile_embed, thresholds, neg_sampler, retrieval_loss_fn,
            item_n_cache, category_n_cache, num_negatives, batch_size, device,
            max_batches=max_steps_per_epoch, static_user_weight=static_user_weight,
            num_workers=num_workers,
        )


# Mọi module có tham số học được + cả 2 optimizer. THIẾU bất kỳ cái nào là resume ra
# model khác: τ (LearnableThresholds) quyết định item_weight/user_weight, retrieval/ranking
# head có bảng riêng — bỏ sót chúng thì nạp lại xong metric lệch mà không báo lỗi gì.
_CKPT_MODULES = ("item_embed", "seq_model", "profile_embed", "thresholds",
                 "retrieval_loss_fn", "ranking_loss_fn")


def save_checkpoint(path: Path, *, epoch: int, step: int, global_step: int,
                    modules: dict, sparse_optimizer, dense_optimizer, config: dict) -> None:
    """Lưu đủ để train tiếp ĐÚNG chỗ đã dừng, không chỉ để eval.

    Lưu cả optimizer state: Adam mang exp_avg/exp_avg_sq: bỏ đi thì resume xong momentum
    reset về 0 và loss nhảy vọt vài trăm step — nhìn như model hỏng, thực ra chỉ là mất
    trạng thái optimizer.

    `config` giữ các cờ ablation (use_gamma/use_beta/static_user_weight/...). Nạp lại mà
    cấu hình lệch thì state_dict vẫn khớp shape nhưng model chạy CƠ CHẾ KHÁC — đúng loại
    lỗi âm thầm đã tốn của ta nhiều lần chạy. load_checkpoint() so và báo.

    Ghi ra .tmp rồi rename: Kaggle ngắt session giữa lúc ghi sẽ để lại file hỏng, và ta chỉ
    phát hiện lúc nạp — sau khi đã mất phiên train."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "epoch": epoch, "step": step, "global_step": global_step, "config": config,
        "sparse_optimizer": sparse_optimizer.state_dict(),
        "dense_optimizer": dense_optimizer.state_dict(),
    }
    for name in _CKPT_MODULES:
        blob[name] = modules[name].state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, modules: dict, sparse_optimizer=None,
                    dense_optimizer=None, config: dict | None = None) -> dict:
    """Nạp checkpoint; trả về {"epoch", "step", "global_step"} để train() chạy tiếp.

    optimizer=None -> chỉ nạp trọng số (đủ cho --eval-only, không cần optimizer state."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)

    if config is not None:
        saved = blob.get("config", {})
        lech = {k: (saved.get(k), v) for k, v in config.items() if saved.get(k) != v}
        if lech:
            raise SystemExit(
                "[resume] CẤU HÌNH LỆCH so với checkpoint — train tiếp sẽ ra model lai, "
                "không đọc được kết quả:\n"
                + "\n".join(f"    {k}: checkpoint={a!r} nhưng lần chạy này={b!r}"
                             for k, (a, b) in lech.items())
            )

    # Thiếu module trong checkpoint = nạp lại chỉ MỘT PHẦN model, phần còn lại giữ trọng
    # số khởi tạo ngẫu nhiên. Im lặng bỏ qua thì metric sai mà không có dấu hiệu gì — báo
    # thẳng còn hơn chẩn đoán ngược từ một con số vô lý.
    thieu = [n for n in _CKPT_MODULES if n not in blob]
    if thieu:
        raise SystemExit(f"[resume] checkpoint THIẾU module: {thieu} — không nạp được")
    for name in _CKPT_MODULES:
        modules[name].load_state_dict(blob[name])
    if sparse_optimizer is not None and "sparse_optimizer" in blob:
        sparse_optimizer.load_state_dict(blob["sparse_optimizer"])
    if dense_optimizer is not None and "dense_optimizer" in blob:
        dense_optimizer.load_state_dict(blob["dense_optimizer"])
    return {"epoch": blob.get("epoch", 0), "step": blob.get("step", 0),
            "global_step": blob.get("global_step", 0)}


def _report_gate_diagnostic(
    stats: dict,  # g_i, category_confidence, item_weight, norm_collab/content/content_shrunk
    is_user_cold: torch.Tensor,
    is_item_cold: torch.Tensor,
) -> None:
    """In thành phần nội bộ của ItemEmbedding theo nhóm — trả lời 2 câu TRƯỚC khi sửa gate.

    Công thức hiện tại (item_embedding.py):
        e_i = g_i·e_collab + (1−g_i)·e_content·c        với g_i = σ(MLP([m, e_collab, e_content]))

    Nghi vấn 1 — GATE có làm đúng việc không? Item cold (m→0) LẼ RA phải đẩy g_i→0 để dựa
    vào content. Nếu g_i vẫn ≈0.5 hoặc cao ở nhóm item-cold thì gate hỏng, và việc thêm c
    vào gate không cứu được gì.

    Nghi vấn 2 — c có THẬT SỰ biến thiên không? c = tanh(N_cat/τ_c) với τ_c=53486; N_category
    gộp hàng nghìn item nên có thể bão hoà ≈1 ở MỌI mẫu. Bằng chứng gián tiếp: τ_c không
    nhúc nhích sau 3000 step ở cả 6 lần chạy (53486.10 → 53486.10) — dấu hiệu tanh ở vùng
    phẳng, không nhận gradient. Nếu std(c) ≈ 0 thì đưa c vào gate cũng như đưa một hằng số,
    và vấn đề thật nằm ở τ_c chứ không ở công thức gate.

    Nghi vấn 3 — nhánh content bị co bao nhiêu? So ‖e_content‖ với ‖e_content·c‖. Ở item
    cold, nếu CẢ g_i·e_collab lẫn (1−g_i)·e_content·c đều nhỏ thì e_i chỉ còn phần e_collab
    chưa được train — khớp với ‖e_pos‖ cold > warm đã đo ở bảng thứ hạng."""
    groups = {
        "warm_warm": (~is_user_cold) & (~is_item_cold),
        "cold_user_warm_item": is_user_cold & (~is_item_cold),
        "warm_user_cold_item": (~is_user_cold) & is_item_cold,
        "cold_cold": is_user_cold & is_item_cold,
    }
    print(f"\n  --- CHẨN ĐOÁN gate ItemEmbedding (candidate positive) ---")
    print("    g_i→1 = tin collab, g_i→0 = tin content; c = category_confidence")
    for name, mask in groups.items():
        n = int(mask.sum().item())
        if n == 0:
            print(f"    [{name}] n=0")
            continue
        g = stats["g_i"][mask]
        c = stats["category_confidence"][mask]
        m = stats["item_weight"][mask]
        print(
            f"    [{name}] n={n} "
            f"g_i={g.mean():.4f}±{g.std():.4f} "
            f"c={c.mean():.4f}±{c.std():.4f} "
            f"m={m.mean():.4f}±{m.std():.4f} | "
            f"‖collab‖={stats['norm_collab'][mask].mean():.4f} "
            f"‖content‖={stats['norm_content'][mask].mean():.4f} "
            f"->co={stats['norm_content_shrunk'][mask].mean():.4f}"
        )
    c_all = stats["category_confidence"]
    print(f"    [toàn bộ] c: min={c_all.min():.4f} p50={c_all.median():.4f} "
          f"max={c_all.max():.4f} std={c_all.std():.4f}"
          f"  <- std≈0 nghĩa là c vô dụng trong gate, phải sửa τ_c")


def _report_rank_diagnostic(
    rank: torch.Tensor,        # (N,) thứ hạng positive, 0 = đứng đầu
    norm_user: torch.Tensor,   # (N,) ‖e_user_eval‖
    norm_pos: torch.Tensor,    # (N,) ‖e_i của positive‖
    n_cand: torch.Tensor,      # (N,) số candidate = 1 + num_negatives
    is_user_cold: torch.Tensor,
    is_item_cold: torch.Tensor,
) -> None:
    """In phân bố THỨ HẠNG positive theo nhóm — phân định nguyên nhân recall=0.

    Recall@K chỉ là (rank < K), nó bằng 0 trong HAI tình huống hoàn toàn khác nhau và
    không cho biết là tình huống nào:

      (a) rank dồn về ĐÁY (median ≈ C−1): positive bị đẩy xuống hệ thống. Nguyên nhân
          thường là embedding positive bị triệt tiêu (‖e_pos‖→0) hoặc logit NaN/-inf —
          tức LỖI, không phải chất lượng model.
      (b) rank RẢI ĐỀU (median ≈ C/2): model đoán mò, không biết gì về item cold. Đó là
          giới hạn học được thật, không phải bug.

    Cột random= là mốc đối chiếu: median của model ngẫu nhiên hoàn toàn. So median quan
    sát với nó là đọc được ngay (a) hay (b). ‖e_pos‖ tách riêng để bắt trường hợp
    embedding cold bị nhân về 0 (xem item_embedding.py: g_i·e_collab + (1−g_i)·e_content
    ·category_confidence — cả hai nhánh đều co lại khi N nhỏ)."""
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
    static_user_weight: bool = False,  # PHẢI khớp cấu hình lúc train, nếu không train/eval lệch
    num_workers: int = 4,
) -> None:
    """Đánh giá Recall/NDCG@K trên split (val/test), tách theo 4 nhóm cold-start — xem
    eval.py. Dùng đúng candidate-sampling như lúc train (positive + N negative theo tần
    suất) — KHÔNG rank full-catalog mỗi sample (không khả thi)."""
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
    all_rank, all_norm_user, all_norm_pos, all_n_cand = [], [], [], []  # chẩn đoán, xem dưới
    all_embed_stats: dict[str, list[torch.Tensor]] = {}  # g_i, c, ‖e_collab‖... của candidate positive

    total = min(len(loader), max_batches) if max_batches else len(loader)
    for i, batch in enumerate(tqdm(loader, total=total, desc=f"eval[{split}]", unit="batch")):
        if max_batches is not None and i >= max_batches:
            break

        fwd = run_batch_forward(
            batch, dataset, item_embed, seq_model, profile_embed, thresholds, neg_sampler,
            item_n_cache, category_n_cache, num_negatives, device,
            static_user_weight=static_user_weight,
        )
        # [SỬA 2026-09-15] eval_logit = dot-product THUẦN, KHÔNG trừ log_q. Trước đây dùng
        # scaled_logit() (có log_q) -> recall@5 = 1.0000 ở ô item-cold, vì log_q cộng ~3.90
        # điểm cho item hiếm trong khi tín hiệu thật chỉ ±1. log_q chỉ hợp lệ khi mọi
        # candidate cùng phân phối lấy mẫu — đúng lúc train, SAI lúc eval (positive từ dữ
        # liệu thật, negative từ tần suất). Xem retrieval.py::eval_logit() docstring.
        eval_logit = retrieval_loss_fn.eval_logit(fwd["e_user_eval"], fwd["cand_e_i"])

        batch_metrics = compute_recall_ndcg_at_k(eval_logit)
        # [CHẨN ĐOÁN 2026-09-16] Vì sao ô item-cold ra recall=0.0000 TUYỆT ĐỐI? Recall chỉ
        # nói "có lọt top-K không", không phân biệt HAI nguyên nhân rất khác nhau:
        #   rank dồn ở đáy (~C-1) -> positive bị ĐẨY xuống hệ thống (embedding hỏng/triệt tiêu)
        #   rank rải đều 0..C-1   -> model chỉ đơn giản KHÔNG BIẾT item cold (đoán mò)
        # Giữ rank thô + norm embedding để phân định; in 1 lần ở cuối, không spam mỗi batch.
        order = torch.argsort(eval_logit, dim=1, descending=True)
        rank = (order == 0).float().argmax(dim=1)  # thứ hạng positive, 0 = đứng đầu
        all_rank.append(rank.cpu())
        all_norm_user.append(fwd["e_user_eval"].norm(dim=-1).cpu())
        all_norm_pos.append(fwd["cand_e_i"][:, 0].norm(dim=-1).cpu())
        all_n_cand.append(torch.full_like(rank.cpu(), eval_logit.shape[1]))
        for k, v in fwd["label_embed_stats"].items():
            all_embed_stats.setdefault(k, []).append(v.cpu())
        for k, v in batch_metrics.items():
            all_metrics.setdefault(k, []).append(v.cpu())
        all_user_cold.append(fwd["is_user_cold"])
        all_item_cold.append(fwd["is_item_cold"])
        all_user_lowhist.append(fwd["is_user_lowhistory"])

    per_sample_metrics = {k: torch.cat(v) for k, v in all_metrics.items()}
    is_user_cold = torch.cat(all_user_cold)
    is_item_cold = torch.cat(all_item_cold)

    # [SỬA 2026-09-15] HAI bảng: strict holdout (zero-shot, n nhỏ) + few-shot (nơi γ thật
    # sự học được). Không gộp — hai định nghĩa trả lời hai câu hỏi khác nhau, xem eval.py.
    is_user_lowhist = torch.cat(all_user_lowhist)
    _report_rank_diagnostic(
        torch.cat(all_rank), torch.cat(all_norm_user), torch.cat(all_norm_pos),
        torch.cat(all_n_cand), is_user_cold, is_item_cold,
    )
    _report_gate_diagnostic(
        {k: torch.cat(v) for k, v in all_embed_stats.items()}, is_user_cold, is_item_cold,
    )
    result = aggregate_by_cold_group(per_sample_metrics, is_user_cold, is_item_cold)
    result_low = aggregate_by_cold_group(per_sample_metrics, is_user_lowhist, is_item_cold)
    print_eval_report(result, thresholds.get_tau_snapshot(), result_lowhistory=result_low)

    item_embed.train()
    seq_model.train()
    profile_embed.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../preprocess_data/output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--no-cuckoo-embedding", action="store_true", help="Tắt CuckooEmbedding, dùng nn.Embedding cố định (ablation, khuyến nghị cho catalog nhỏ như KuaiRand-Pure)")
    parser.add_argument("--cuckoo-capacity-ratio", type=float, default=0.25, help="Capacity mỗi bảng con cuckoo = ratio * num_ids")
    parser.add_argument("--no-profile-token", action="store_true", help="Không prepend e_profile làm token 0 của chuỗi — ablation (xem user_embedding.py)")
    parser.add_argument("--no-interleave", action="store_true", help="Cộng gộp item+action vào 1 token (chuỗi K) thay vì xen kẽ [Φ,a,Φ,a,...] (chuỗi 2K, đúng HSTU) — ablation")
    parser.add_argument("--static-user-weight", action="store_true", help="ABLATION #5: u_i TĨNH per-user (N_u tại điểm dự đoán, broadcast ra K vị trí) thay vì per-position. Thí nghiệm tách bạch đóng góp 'cold-start là đại lượng per-position' — xem run_batch_forward()")
    parser.add_argument("--no-beta", action="store_true", help="ABLATION #3: tắt số hạng β·log m_j trong attention bias")
    parser.add_argument("--no-gamma", action="store_true", help="ABLATION #4: tắt số hạng TÍCH γ·log(u_i)·log(m_j) — đóng góp chính, ablation quan trọng nhất")
    parser.add_argument("--save-every", type=int, default=None, help="Lưu checkpoint mỗi N step (ngoài lần lưu cuối mỗi epoch, vốn LUÔN chạy). Dùng khi session hay bị ngắt giữa chừng")
    parser.add_argument("--save-path", default=None, help="Đường dẫn checkpoint; mặc định <output-dir>/ckpt.pt. Trên Kaggle nên trỏ vào /kaggle/working (output-dir có thể read-only)")
    parser.add_argument("--resume", default=None, help="Nạp checkpoint và train TIẾP từ đúng step đã dừng (gồm cả optimizer state)")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker nạp dữ liệu (mặc định 4). Nghẽn là CPU chứ không phải GPU — đặt 0 để debug hoặc khi môi trường không cho fork")
    parser.add_argument("--eval-only", action="store_true", help="Không train, chỉ nạp checkpoint (--resume/--save-path) và chạy evaluate() — dùng để chạy lại chẩn đoán trên CÙNG model, ~8 phút thay vì train lại ~2 giờ")
    parser.add_argument("--checkpoint", action="store_true", help="Gradient checkpointing: chậm ~30%%, tiết kiệm ~70%% VRAM. BẮT BUỘC cho nhánh có γ ở batch=256 trên T4 15GB (nếu không sẽ CUDA OOM ở loss.backward)")
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        use_cuckoo_embedding=not args.no_cuckoo_embedding,
        cuckoo_capacity_ratio=args.cuckoo_capacity_ratio,
        use_profile_token=not args.no_profile_token,
        interleave=not args.no_interleave,
        static_user_weight=args.static_user_weight,
        use_beta=not args.no_beta,
        use_gamma=not args.no_gamma,
        use_checkpoint=args.checkpoint,
        save_every=args.save_every,
        save_path=args.save_path,
        resume=args.resume,
        eval_only=args.eval_only,
        num_workers=args.num_workers,
    )
