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


# Model nào cần prefix "passage: "/"query: " trước mỗi câu (chuẩn E5-style asymmetric
# embedding) — GTE/BGE/text2vec KHÔNG cần, tự thêm sai prefix sẽ làm giảm chất lượng.
MODELS_REQUIRE_PASSAGE_PREFIX = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"}
# Model cần trust_remote_code=True để tải code custom (đã XÁC NHẬN qua test trực tiếp:
# Alibaba-NLP/gte-multilingual-base tự tải thêm code từ Alibaba-NLP/new-impl trên HF Hub —
# chấp nhận rủi ro supply-chain vì là model chính thức của Alibaba, đã CHỐT 2026-09-10).
MODELS_REQUIRE_TRUST_REMOTE_CODE = {"Alibaba-NLP/gte-multilingual-base"}


def encode_all_captions(
    input_csv: Path,
    output_dir: Path,
    num_items: int = NUM_ITEMS,
    model_name: str = MODEL_NAME,
    batch_size: int = 256,
    chunk_size: int = 1_000_000,
    multi_gpu: bool = True,
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
    out_path = output_dir / "caption_embeddings.npy"
    has_caption_path = output_dir / "caption_has_caption.npy"
    progress_path = output_dir / "encode_progress.json"

    if progress_path.exists():
        rows_done = json.loads(progress_path.read_text())["rows_done"]
        print(f"[encode_captions] resume: {rows_done} dòng đã xử lý ở session trước, bỏ qua.")
        embeddings_mm = np.lib.format.open_memmap(out_path, mode="r+")
        has_caption_mm = np.lib.format.open_memmap(has_caption_path, mode="r+")
    else:
        rows_done = 0
        embeddings_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(num_items, embed_dim))
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
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--single-gpu", action="store_true", help="Tắt multi-GPU pool, chỉ dùng 1 GPU/CPU")
    args = parser.parse_args()

    encode_all_captions(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_items=args.num_items,
        model_name=args.model_name,
        batch_size=args.batch_size,
        multi_gpu=not args.single_gpu,
        chunk_size=args.chunk_size,
    )
