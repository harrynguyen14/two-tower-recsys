"""Pass 3.5 (chạy trên Kaggle, KHÔNG chạy local) — encode caption tiếng Trung qua
multilingual-e5-base (frozen, không fine-tune) thành embedding 768 chiều, dùng làm nhánh
text OPTIONAL trong content_branch (GMU, xem item_embedding.py). Benchmark CPU local:
67.3 caption/s -> ~5.5 ngày cho 32M caption -> CHẠY TRÊN KAGGLE (GPU T4/P100 miễn phí)
để rút ngắn còn vài giờ.

Cách dùng trên Kaggle:
1. Tạo Kaggle Dataset chỉ chứa kuairand_video_captions.csv (944MB, tải từ
   https://zenodo.org/records/18159199/files/kuairand_video_captions.csv).
2. Tạo Notebook mới, add dataset vừa tạo, BẬT GPU (Settings -> Accelerator -> GPU T4 x2
   hoặc P100), copy nội dung file này vào 1 cell, chạy.
3. Sau khi chạy xong, tải file caption_embeddings.npy từ /kaggle/working/ về, đặt vào
   D:\\ama-rs\\gen-recsys\\preprocess_data\\output\\ (cùng chỗ với item_static.npy).

Output: caption_embeddings.npy (num_items, 768) float32 — index i tương ứng video_id=i
(giống item_static.npy, identity mapping, ĐÃ XÁC NHẬN video_id liên tục 0..N-1). Item
không có trong file caption gốc (nếu có) được điền vector 0 (GMU sẽ tự mask qua
has_caption, không dùng nhánh text cho các dòng này).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==== ĐỔI 2 DÒNG NÀY THEO TÊN DATASET/NOTEBOOK THẬT TRÊN KAGGLE ====
INPUT_CSV = Path("/kaggle/input/kuairand-video-captions/kuairand_video_captions.csv")
OUTPUT_DIR = Path("/kaggle/working")
# ====================================================================

NUM_ITEMS = 32_038_725  # tổng số item KuaiRand-27K, đã xác nhận video_id liên tục 0..N-1
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 256  # GPU T4/P100 xử lý batch lớn hơn CPU nhiều, tăng throughput
CHUNK_SIZE = 200_000  # đọc CSV theo chunk, tránh load hết 944MB + embedding cùng lúc vào RAM


def encode_all_captions() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[encode_captions] device = {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    embed_dim = model.get_sentence_embedding_dimension()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "caption_embeddings.npy"
    embeddings_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(NUM_ITEMS, embed_dim))
    has_caption_mm = np.lib.format.open_memmap(
        OUTPUT_DIR / "caption_has_caption.npy", mode="w+", dtype=np.bool_, shape=(NUM_ITEMS,)
    )
    has_caption_mm[:] = False  # mặc định KHÔNG có caption, chỉ bật True cho dòng thật xử lý được

    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    total_processed = 0
    for chunk_idx, chunk in enumerate(reader):
        video_ids = chunk["final_video_id"].to_numpy()
        captions = chunk["caption"].fillna("").astype(str).tolist()
        # multilingual-e5 yêu cầu prefix "passage: " cho document embedding (khác "query: ")
        captions = [f"passage: {c}" for c in captions]

        emb = model.encode(captions, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        embeddings_mm[video_ids] = emb.astype(np.float32)

        nonempty = chunk["caption"].fillna("").astype(str).str.len() > 0
        has_caption_mm[video_ids[nonempty.to_numpy()]] = True

        total_processed += len(chunk)
        print(f"[encode_captions] chunk {chunk_idx}: {total_processed}/{NUM_ITEMS} processed")

    embeddings_mm.flush()
    has_caption_mm.flush()
    print(f"[encode_captions] DONE -> {out_path}")


if __name__ == "__main__":
    encode_all_captions()
