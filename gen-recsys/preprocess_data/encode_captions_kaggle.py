"""Pass 3.5 (chạy trên Kaggle, KHÔNG chạy local) — encode caption tiếng Trung qua
multilingual-e5-small (frozen, không fine-tune) thành embedding 384 chiều, dùng làm nhánh
text OPTIONAL trong content_branch (GMU, xem item_embedding.py).

[CHỐT 2026-09-10] Đã thử Alibaba-NLP/gte-multilingual-base (tối ưu tiếng Trung tốt hơn)
nhưng gặp lỗi CUDA "index out of bounds" trong custom code của kiến trúc đó khi chạy qua
multi-GPU pool trên Kaggle thật (2xT4) — lỗi không tái hiện trên CPU đơn, chỉ xảy ra khi
kết hợp custom RoPE/token_type_ids code + multi-process pool. QUAY LẠI multilingual-e5
(kiến trúc chuẩn, không cần trust_remote_code, đã verify multi-GPU pool hoạt động đúng) —
đổi từ base (278M, 768-dim) sang small (118M, 384-dim) để nhanh hơn ~2-3x. Benchmark CPU
local (bản base): 67.3 caption/s -> ~5.5 ngày cho 32M caption -> CHẠY TRÊN KAGGLE (GPU
T4/P100 miễn phí) để rút ngắn còn vài giờ.

Cách dùng trên Kaggle:
1. Tạo Kaggle Dataset chỉ chứa kuairand_video_captions.csv (944MB, tải từ
   https://zenodo.org/records/18159199/files/kuairand_video_captions.csv).
2. Tạo Notebook mới, add dataset vừa tạo, BẬT GPU (Settings -> Accelerator -> GPU T4 x2
   hoặc P100), copy nội dung file này vào 1 cell, chạy với đúng --input-csv/--output-dir
   (xem ví dụ lệnh ở cuối file).
3. Sau khi chạy xong, tải TỪNG file caption_embeddings_shard{i}.npy từ /kaggle/working/ về
   (xem giải thích sharding bên dưới), đặt cùng vào D:\\ama-rs\\gen-recsys\\preprocess_data\\
   output\\ (cùng chỗ với item_static.npy).

[CHỐT 2026-09-10 — SHARDING] 1 file caption_embeddings.npy duy nhất (num_items=32,038,725 x
384 x float32) nặng ~49.2GB — VƯỢT giới hạn output 20GB của Kaggle notebook. Chia thành
`num_shards` file nhỏ hơn (mặc định 4, mỗi file ~12.3GB), mỗi shard chứa 1 dải video_id
LIÊN TỤC: shard k chứa video_id trong [k*shard_size, (k+1)*shard_size). Tải về từng file
riêng (Kaggle cho phép tải nhiều lần, không giới hạn tổng dung lượng tải xuống — chỉ giới
hạn dung lượng LƯU trong 1 lần commit output).

Output: caption_embeddings_shard{k}.npy (shard_size, 384) float32 với k=0..num_shards-1,
mỗi shard tự chứa cả video_id range của nó — dùng chung has_caption (1 file, nhỏ, không cần
sharding). Item không có trong file caption gốc (nếu có) được điền vector 0 (GMU sẽ tự mask
qua has_caption, không dùng nhánh text cho các dòng này).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NUM_ITEMS = 32_038_725  # tổng số item KuaiRand-27K, đã xác nhận video_id liên tục 0..N-1
MODEL_NAME = "intfloat/multilingual-e5-small"


# Model nào cần prefix "passage: "/"query: " trước mỗi câu (chuẩn E5-style asymmetric
# embedding) — GTE/BGE/text2vec KHÔNG cần, tự thêm sai prefix sẽ làm giảm chất lượng.
MODELS_REQUIRE_PASSAGE_PREFIX = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"}
# Model cần trust_remote_code=True để tải code custom (đã XÁC NHẬN qua test trực tiếp:
# Alibaba-NLP/gte-multilingual-base tự tải thêm code từ Alibaba-NLP/new-impl trên HF Hub —
# chấp nhận rủi ro supply-chain vì là model chính thức của Alibaba, đã CHỐT 2026-09-10).
MODELS_REQUIRE_TRUST_REMOTE_CODE = {"Alibaba-NLP/gte-multilingual-base"}


class ShardedEmbeddingWriter:
    """N file .npy nhỏ thay vì 1 file khổng lồ (49.2GB cho num_items=32,038,725 x 384 x
    float32 — VƯỢT giới hạn output 20GB của Kaggle notebook). Shard k chứa video_id trong
    [k*shard_size, (k+1)*shard_size) — route ghi bằng __setitem__ giống 1 mảng thống nhất,
    người gọi (encode_all_captions) không cần biết chi tiết sharding.
    """

    def __init__(self, output_dir: Path, num_items: int, embed_dim: int, num_shards: int, mode: str):
        self.num_items = num_items
        self.num_shards = num_shards
        self.shard_size = (num_items + num_shards - 1) // num_shards
        self.shards = []
        for k in range(num_shards):
            path = output_dir / f"caption_embeddings_shard{k}.npy"
            start = k * self.shard_size
            this_shard_len = min(self.shard_size, num_items - start)
            if mode == "w+":
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(this_shard_len, embed_dim))
            else:
                mm = np.lib.format.open_memmap(path, mode="r+")
            self.shards.append(mm)

    def __setitem__(self, video_ids: np.ndarray, values: np.ndarray) -> None:
        shard_idx = video_ids // self.shard_size
        local_idx = video_ids % self.shard_size
        for k in np.unique(shard_idx):
            mask = shard_idx == k
            self.shards[k][local_idx[mask]] = values[mask]

    def flush(self) -> None:
        for mm in self.shards:
            mm.flush()


def encode_all_captions(
    input_csv: Path,
    output_dir: Path,
    num_items: int = NUM_ITEMS,
    model_name: str = MODEL_NAME,
    batch_size: int = 256,
    chunk_size: int = 1_000_000,
    multi_gpu: bool = True,
    num_shards: int = 4,
) -> None:
    """Checkpoint/resume: sau MỖI chunk, ghi số dòng đã xử lý vào {output_dir}/
    encode_progress.json. Nếu file này đã tồn tại lúc bắt đầu (session Kaggle trước bị
    ngắt do giới hạn 12h), tự động mở lại 2 file .npy đã có (mode="r+", KHÔNG tạo mới —
    tránh mất tiến độ cũ) và pd.read_csv(..., skiprows=...) bỏ qua phần đã encode."""
    import json

    import torch

    num_gpus = torch.cuda.device_count()
    use_pool = multi_gpu and num_gpus > 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[encode_captions] device={device} num_gpus={num_gpus} multi_gpu_pool={use_pool} model={model_name}")

    needs_prefix = model_name in MODELS_REQUIRE_PASSAGE_PREFIX
    needs_remote_code = model_name in MODELS_REQUIRE_TRUST_REMOTE_CODE
    model_kwargs = {"trust_remote_code": True} if needs_remote_code else {}
    model = SentenceTransformer(model_name, device=device, **model_kwargs)
    embed_dim = model.get_sentence_embedding_dimension()

    # start_multi_process_pool: chia batch qua từng process/GPU riêng (API chính thức
    # sentence-transformers cho multi-GPU) — CHỈ có lợi khi >1 GPU, vì mỗi process tốn thời
    # gian khởi động riêng, không đáng với 1 GPU (xem cảnh báo GPU thứ 2 rảnh 0% ở Kaggle).
    pool = model.start_multi_process_pool() if use_pool else None

    output_dir.mkdir(parents=True, exist_ok=True)
    has_caption_path = output_dir / "caption_has_caption.npy"
    progress_path = output_dir / "encode_progress.json"

    if progress_path.exists():
        rows_done = json.loads(progress_path.read_text())["rows_done"]
        print(f"[encode_captions] resume: {rows_done} dòng đã xử lý ở session trước, bỏ qua.")
        embeddings_mm = ShardedEmbeddingWriter(output_dir, num_items, embed_dim, num_shards, mode="r+")
        has_caption_mm = np.lib.format.open_memmap(has_caption_path, mode="r+")
    else:
        rows_done = 0
        embeddings_mm = ShardedEmbeddingWriter(output_dir, num_items, embed_dim, num_shards, mode="w+")
        has_caption_mm = np.lib.format.open_memmap(has_caption_path, mode="w+", dtype=np.bool_, shape=(num_items,))
        has_caption_mm[:] = False  # mặc định KHÔNG có caption, chỉ bật True cho dòng thật xử lý được

    try:
        # skiprows bỏ qua đúng số dòng ĐÃ encode ở session trước — CSV có header nên
        # pandas tự hiểu skiprows áp dụng SAU header (không cần cộng thêm 1).
        reader = pd.read_csv(input_csv, chunksize=chunk_size, skiprows=range(1, rows_done + 1) if rows_done else None)
        # chunk_size LỚN (mặc định 1 triệu) — mỗi lần gọi encode(pool=...) qua process
        # boundary có overhead khởi tạo/serialize đáng kể (đã đo thực tế: chunk_size=20_000
        # cho tốc độ ~277 caption/s, CHẬM HƠN CPU đơn luồng 67 caption/s — vì số lần gọi pool
        # quá nhiều, 1602 lần, overhead lấn át lợi ích multi-GPU). Chunk lớn hơn nhiều giảm số
        # lần gọi pool xuống ~32 lần, để mỗi lần pool xử lý đủ khối lượng bù overhead khởi tạo.
        # KHÔNG bọc `reader` bằng tqdm ở đây — model.encode(show_progress_bar=True) đã tự vẽ
        # 1 thanh tqdm "Batches" bên trong mỗi chunk. 2 thanh tqdm lồng nhau (chunk-level +
        # batch-level) tranh nhau dòng terminal (\r) gây giật/đè lên nhau, KHÔNG mượt hơn —
        # đã xác nhận qua log thật. Chỉ in mốc chunk đơn giản, để tqdm nội bộ là nguồn tiến
        # độ chi tiết duy nhất.
        remaining_chunks = (num_items - rows_done + chunk_size - 1) // chunk_size
        for chunk_idx, chunk in enumerate(reader):
            print(f"[encode_captions] chunk {chunk_idx + 1}/{remaining_chunks} ({rows_done} dòng đã xong)")
            video_ids = chunk["final_video_id"].to_numpy()
            captions = chunk["caption"].fillna("").astype(str).tolist()
            if needs_prefix:
                # chỉ E5-style: prefix "passage: " cho document embedding (khác "query: ")
                captions = [f"passage: {c}" for c in captions]

            if pool is not None:
                # sentence-transformers >=3.2 gộp multi-process vào encode(pool=...), bản cũ
                # hơn (đã xác nhận local đang có 3.0.1) chỉ có encode_multi_process() riêng —
                # Kaggle có thể cài bản khác local, thử API mới trước, rơi về API cũ nếu
                # TypeError (tham số pool không tồn tại).
                try:
                    emb = model.encode(captions, pool=pool, batch_size=batch_size, show_progress_bar=True)
                except TypeError:
                    emb = model.encode_multi_process(captions, pool, batch_size=batch_size)
            else:
                emb = model.encode(captions, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            embeddings_mm[video_ids] = emb.astype(np.float32)

            nonempty = chunk["caption"].fillna("").astype(str).str.len() > 0
            has_caption_mm[video_ids[nonempty.to_numpy()]] = True

            rows_done += len(chunk)
            embeddings_mm.flush()
            has_caption_mm.flush()
            progress_path.write_text(json.dumps({"rows_done": rows_done}))
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    progress_path.unlink(missing_ok=True)  # xóa checkpoint khi đã xong HẲN, tránh resume nhầm lần sau
    print(f"[encode_captions] DONE -> {num_shards} shard(s) trong {output_dir}")


if __name__ == "__main__":
    # Ví dụ chạy trên Kaggle:
    #   python encode_captions_kaggle.py \
    #     --input-csv /kaggle/input/kuairand-video-captions/kuairand_video_captions.csv \
    #     --output-dir /kaggle/working
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True, help="Đường dẫn kuairand_video_captions.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Thư mục ghi caption_embeddings.npy")
    parser.add_argument("--num-items", type=int, default=NUM_ITEMS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--single-gpu", action="store_true", help="Tắt multi-GPU pool, chỉ dùng 1 GPU/CPU")
    parser.add_argument("--num-shards", type=int, default=4, help="Chia output thành N file .npy nhỏ (mặc định 4, ~12.3GB/shard)")
    args = parser.parse_args()

    encode_all_captions(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_items=args.num_items,
        model_name=args.model_name,
        batch_size=args.batch_size,
        multi_gpu=not args.single_gpu,
        chunk_size=args.chunk_size,
        num_shards=args.num_shards,
    )
