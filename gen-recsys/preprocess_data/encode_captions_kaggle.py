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

Cách dùng trên Kaggle (chạy TUẦN TỰ từng shard — bắt buộc, xem lý do ở mục SHARDING dưới):
1. Tạo Kaggle Dataset chỉ chứa kuairand_video_captions.csv (944MB, tải từ
   https://zenodo.org/records/18159199/files/kuairand_video_captions.csv).
2. Tạo Notebook mới, add dataset vừa tạo, BẬT GPU (Settings -> Accelerator -> GPU T4 x2
   hoặc P100), copy nội dung file này vào 1 cell.
3. Chạy VỚI --shard-index 0 trước. Khi xong (chỉ 1 file caption_embeddings_shard0.npy
   ~12.3GB xuất hiện, KHÔNG có shard khác), tải file đó về, XÓA nó khỏi /kaggle/working/
   (giải phóng dung lượng output), rồi chạy lại với --shard-index 1, rồi 2, rồi 3.
4. Đặt cả 4 file shard đã tải về D:\\ama-rs\\gen-recsys\\preprocess_data\\output\\ (cùng
   chỗ với item_static.npy).

[CHỐT 2026-09-10 — SHARDING TUẦN TỰ] 1 file caption_embeddings.npy duy nhất (num_items=
32,038,725 x 384 x float32) nặng ~49.2GB — VƯỢT giới hạn output 20GB của Kaggle notebook.
[SỬA — bản đầu tiên mở CẢ 4 shard SONG SONG (vì dữ liệu CSV không sort theo video_id, mỗi
chunk chứa video_id rải rác khắp mọi shard) — nghĩa là dù chia file, TỔNG dung lượng 4 file
cộng lại vẫn ~49GB tại cùng 1 thời điểm, KHÔNG giải quyết được giới hạn output. Đã xác nhận
qua ảnh chụp Kaggle thật: cả 4 shard0-3.npy xuất hiện ngay từ đầu, cộng dồn tới gần
19.5GB limit dù mới chạy được vài % tiến độ]. ĐÃ SỬA: mỗi lần chạy CHỈ xử lý 1 shard
(--shard-index), filter dòng CSV theo đúng dải video_id của shard đó TRƯỚC khi encode (bỏ
qua dòng ngoài dải — phải quét lại toàn bộ CSV mỗi lần, chấp nhận đánh đổi vì mục tiêu
chính là không vượt dung lượng, không phải tốc độ tối đa). Ghi ra ĐÚNG 1 file
caption_embeddings_shard{k}.npy rồi dừng hẳn — cho phép tải về + xóa trước khi chạy shard
tiếp theo, tại mọi thời điểm CHỈ có tối đa 1 shard (~12.3GB) tồn tại trên output.

Output: caption_embeddings_shard{k}.npy (shard_size, 384) float32 — 1 file/lần chạy, ĐÃ
XÁC NHẬN video_id liên tục 0..N-1. caption_has_caption.npy cũng chia theo shard cùng cách
(caption_has_caption_shard{k}.npy). Item không có trong file caption gốc (nếu có) được
điền vector 0 (GMU sẽ tự mask qua has_caption, không dùng nhánh text cho các dòng này).
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


def encode_shard(
    input_csv: Path,
    output_dir: Path,
    shard_index: int,
    num_items: int = NUM_ITEMS,
    num_shards: int = 4,
    model_name: str = MODEL_NAME,
    batch_size: int = 256,
    chunk_size: int = 1_000_000,
    multi_gpu: bool = True,
) -> None:
    """Encode CHỈ 1 shard (video_id trong [shard_index*shard_size, (shard_index+1)*shard_size)),
    quét TOÀN BỘ CSV nhưng bỏ qua dòng ngoài dải (đánh đổi: chậm hơn vì đọc lại CSV mỗi lần
    gọi cho 1 shard khác nhau, nhưng đảm bảo tại mọi thời điểm chỉ có ĐÚNG 1 shard tồn tại
    trên đĩa — bắt buộc vì Kaggle giới hạn output 20GB, 1 shard ~12.3GB, 4 shard cộng dồn
    ~49GB nếu ghi song song sẽ vượt giới hạn, xem docstring module).

    Checkpoint/resume: {output_dir}/encode_progress_shard{shard_index}.json — riêng theo
    từng shard, không lẫn giữa các lần chạy --shard-index khác nhau.
    """
    import json

    import torch

    shard_size = (num_items + num_shards - 1) // num_shards
    shard_start = shard_index * shard_size
    shard_len = min(shard_size, num_items - shard_start)
    shard_end = shard_start + shard_len

    num_gpus = torch.cuda.device_count()
    use_pool = multi_gpu and num_gpus > 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[encode_captions] shard={shard_index}/{num_shards} range=[{shard_start},{shard_end}) "
        f"device={device} num_gpus={num_gpus} multi_gpu_pool={use_pool} model={model_name}"
    )

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
    out_path = output_dir / f"caption_embeddings_shard{shard_index}.npy"
    has_caption_path = output_dir / f"caption_has_caption_shard{shard_index}.npy"
    progress_path = output_dir / f"encode_progress_shard{shard_index}.json"

    if progress_path.exists():
        rows_done = json.loads(progress_path.read_text())["rows_done"]
        print(f"[encode_captions] resume: {rows_done} dòng CSV đã quét ở session trước, bỏ qua.")
        embeddings_mm = np.lib.format.open_memmap(out_path, mode="r+")
        has_caption_mm = np.lib.format.open_memmap(has_caption_path, mode="r+")
    else:
        rows_done = 0
        embeddings_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(shard_len, embed_dim))
        has_caption_mm = np.lib.format.open_memmap(has_caption_path, mode="w+", dtype=np.bool_, shape=(shard_len,))
        has_caption_mm[:] = False  # mặc định KHÔNG có caption, chỉ bật True cho dòng thật xử lý được

    try:
        # skiprows bỏ qua đúng số DÒNG CSV (không phải video_id) đã QUÉT ở session trước —
        # rows_done đếm theo vị trí CSV, KHÔNG phải số dòng đã encode (vì phần lớn dòng CSV
        # bị filter bỏ, không thuộc shard này) — cần quét lại đúng chỗ đã dừng, không phải
        # đếm số item đã encode.
        reader = pd.read_csv(input_csv, chunksize=chunk_size, skiprows=range(1, rows_done + 1) if rows_done else None)
        # chunk_size LỚN (mặc định 1 triệu dòng CSV/lần đọc, KHÔNG phải 1 triệu dòng
        # encode — sau filter theo shard chỉ còn ~1/num_shards số dòng thật cần encode mỗi
        # chunk) — mỗi lần gọi encode(pool=...) qua process boundary có overhead khởi tạo/
        # serialize đáng kể (đã đo thực tế: chunk_size=20_000 cho tốc độ CHẬM HƠN CPU đơn
        # luồng vì số lần gọi pool quá nhiều, overhead lấn át lợi ích multi-GPU).
        for chunk_idx, chunk in enumerate(reader):
            print(f"[encode_captions] shard {shard_index}: CSV chunk {chunk_idx + 1} ({rows_done} dòng CSV đã quét)")
            all_video_ids = chunk["final_video_id"].to_numpy()
            in_shard = (all_video_ids >= shard_start) & (all_video_ids < shard_end)
            video_ids = all_video_ids[in_shard]

            if len(video_ids) > 0:
                captions = chunk["caption"].fillna("").astype(str).to_numpy()[in_shard].tolist()
                if needs_prefix:
                    # chỉ E5-style: prefix "passage: " cho document embedding (khác "query: ")
                    captions = [f"passage: {c}" for c in captions]

                if pool is not None:
                    # sentence-transformers >=3.2 gộp multi-process vào encode(pool=...), bản
                    # cũ hơn chỉ có encode_multi_process() riêng — thử API mới trước, rơi về
                    # API cũ nếu TypeError (tham số pool không tồn tại).
                    try:
                        emb = model.encode(captions, pool=pool, batch_size=batch_size, show_progress_bar=True)
                    except TypeError:
                        emb = model.encode_multi_process(captions, pool, batch_size=batch_size)
                else:
                    emb = model.encode(captions, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

                local_idx = video_ids - shard_start
                embeddings_mm[local_idx] = emb.astype(np.float32)

                nonempty = chunk["caption"].fillna("").astype(str).to_numpy()[in_shard] != ""
                has_caption_mm[local_idx[nonempty]] = True

            rows_done += len(chunk)
            embeddings_mm.flush()
            has_caption_mm.flush()
            progress_path.write_text(json.dumps({"rows_done": rows_done}))
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    progress_path.unlink(missing_ok=True)  # xóa checkpoint khi đã xong HẲN, tránh resume nhầm lần sau
    print(f"[encode_captions] DONE shard {shard_index} -> {out_path} (~{shard_len * embed_dim * 4 / 1e9:.1f}GB)")
    print(f"[encode_captions] TẢI FILE NÀY VỀ + XÓA khỏi Kaggle trước khi chạy --shard-index {shard_index + 1}")


if __name__ == "__main__":
    # Ví dụ chạy trên Kaggle — LẶP LẠI 4 LẦN, mỗi lần đổi --shard-index (0,1,2,3), tải file
    # + xóa khỏi /kaggle/working/ giữa mỗi lần (xem docstring module):
    #   python encode_captions_kaggle.py \
    #     --input-csv /kaggle/input/kuairand-video-captions/kuairand_video_captions.csv \
    #     --output-dir /kaggle/working --shard-index 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True, help="Đường dẫn kuairand_video_captions.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Thư mục ghi caption_embeddings_shard{k}.npy")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard cần encode (0..num_shards-1) — CHỈ 1 shard/lần chạy")
    parser.add_argument("--num-items", type=int, default=NUM_ITEMS)
    parser.add_argument("--num-shards", type=int, default=4, help="Tổng số shard (mặc định 4, ~12.3GB/shard)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--single-gpu", action="store_true", help="Tắt multi-GPU pool, chỉ dùng 1 GPU/CPU")
    args = parser.parse_args()

    encode_shard(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_items=args.num_items,
        num_shards=args.num_shards,
        model_name=args.model_name,
        batch_size=args.batch_size,
        multi_gpu=not args.single_gpu,
        chunk_size=args.chunk_size,
    )
