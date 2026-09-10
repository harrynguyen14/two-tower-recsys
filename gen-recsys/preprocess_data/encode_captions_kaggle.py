"""Pass 3.5 (chạy trên Kaggle, KHÔNG chạy local) — encode caption tiếng Trung qua
multilingual-e5-base (frozen, không fine-tune) thành embedding 768 chiều, dùng làm nhánh
text OPTIONAL trong content_branch (GMU, xem item_embedding.py). Benchmark CPU local:
67.3 caption/s -> ~5.5 ngày cho 32M caption -> CHẠY TRÊN KAGGLE (GPU T4/P100 miễn phí)
để rút ngắn còn vài giờ.

Cách dùng trên Kaggle:
1. Tạo Kaggle Dataset chỉ chứa kuairand_video_captions.csv (944MB, tải từ
   https://zenodo.org/records/18159199/files/kuairand_video_captions.csv).
2. Tạo Notebook mới, add dataset vừa tạo, BẬT GPU (Settings -> Accelerator -> GPU T4 x2
   hoặc P100), copy nội dung file này vào 1 cell, chạy với đúng --input-csv/--output-dir
   (xem ví dụ lệnh ở cuối file).
3. Sau khi chạy xong, tải file caption_embeddings.npy từ /kaggle/working/ về, đặt vào
   D:\\ama-rs\\gen-recsys\\preprocess_data\\output\\ (cùng chỗ với item_static.npy).

Output: caption_embeddings.npy (num_items, 768) float32 — index i tương ứng video_id=i
(giống item_static.npy, identity mapping, ĐÃ XÁC NHẬN video_id liên tục 0..N-1). Item
không có trong file caption gốc (nếu có) được điền vector 0 (GMU sẽ tự mask qua
has_caption, không dùng nhánh text cho các dòng này).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NUM_ITEMS = 32_038_725  # tổng số item KuaiRand-27K, đã xác nhận video_id liên tục 0..N-1
MODEL_NAME = "intfloat/multilingual-e5-base"


def encode_all_captions(
    input_csv: Path,
    output_dir: Path,
    num_items: int = NUM_ITEMS,
    model_name: str = MODEL_NAME,
    batch_size: int = 256,
    chunk_size: int = 200_000,
) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[encode_captions] device = {device}")

    model = SentenceTransformer(model_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "caption_embeddings.npy"
    embeddings_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(num_items, embed_dim))
    has_caption_mm = np.lib.format.open_memmap(
        output_dir / "caption_has_caption.npy", mode="w+", dtype=np.bool_, shape=(num_items,)
    )
    has_caption_mm[:] = False  # mặc định KHÔNG có caption, chỉ bật True cho dòng thật xử lý được

    reader = pd.read_csv(input_csv, chunksize=chunk_size)
    total_processed = 0
    for chunk_idx, chunk in enumerate(reader):
        video_ids = chunk["final_video_id"].to_numpy()
        captions = chunk["caption"].fillna("").astype(str).tolist()
        # multilingual-e5 yêu cầu prefix "passage: " cho document embedding (khác "query: ")
        captions = [f"passage: {c}" for c in captions]

        emb = model.encode(captions, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        embeddings_mm[video_ids] = emb.astype(np.float32)

        nonempty = chunk["caption"].fillna("").astype(str).str.len() > 0
        has_caption_mm[video_ids[nonempty.to_numpy()]] = True

        total_processed += len(chunk)
        print(f"[encode_captions] chunk {chunk_idx}: {total_processed}/{num_items} processed")

    embeddings_mm.flush()
    has_caption_mm.flush()
    print(f"[encode_captions] DONE -> {out_path}")


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
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    encode_all_captions(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_items=args.num_items,
        model_name=args.model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
