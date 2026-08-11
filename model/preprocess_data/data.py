"""
Giai đoạn 1 (xem Readme.md mục "Item tower — kiến trúc GMU"): nhúng text và ảnh bìa QUA
CÙNG 1 CLIP model nhưng xuất RIÊNG 2 vector (không average/concat thô ở đây) — quyết định
cách kết hợp 2 modal đẩy xuống ItemTower (GMU gate học per-sample, xem item_tower.py).

Lý do tách riêng (đã đo trên 10 item, xem Readme): text_embedding phân biệt category tốt
(same-category sim ~0.57-0.75), image_embedding KHÔNG (bìa sách phản ánh phong cách thiết
kế/marketing hơn nội dung — vd Romance vs SciFi cover ~0.54, gần bằng same-category) — nếu
average cố định 50/50 ngay ở bước nhúng sẽ pha loãng tín hiệu tốt của text bằng nhiễu ảnh
không đồng đều giữa các item. Gate học được (GMU) cần 2 vector riêng làm input.

Xuất:
  text_embeddings.npy  [n_items, dim]
  image_embeddings.npy [n_items, dim]  (vector 0 cho item thiếu ảnh — không giả lập ảnh)
  has_image.npy        [n_items] bool — đánh dấu item nào có ảnh THẬT (khác vector 0 thật
                        do CLIP encode) để GMU/modality-dropout phân biệt "thiếu ảnh" vs
                        "có ảnh nhưng embedding tình cờ gần 0".
  asin_to_idx.json     mapping asin -> row index, dùng chung cho cả 3 mảng trên.

Model: openai/clip-vit-base-patch32 (qua HuggingFace transformers, chạy CPU hoặc GPU tự
động — CPU dùng để test cục bộ, GPU dùng khi chạy thật trên cloud, xem Readme).

Ảnh phải tải về trước bằng ds-down/download_images.py (đã sửa trỏ Kindle_Store, có
MAX_IMAGES để giới hạn khi test cục bộ).

Chạy 1 GPU/CPU: python data.py --meta-file <path> --image-dir <path> --out-dir <path> [--limit N]
Chạy nhiều GPU (Kaggle T4x2): thêm --gpus "0,1" — tự spawn 1 process/GPU (multiprocessing,
mỗi process 1 CUDA context riêng biệt), chạy song song rồi TỰ GỘP kết quả vào --out-dir.
Không cần mở nhiều lệnh shell tay (Kaggle notebook chỉ chạy được 1 cell tuần tự).
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DEFAULT_META_FILE = r"D:\amazon-datasets\Kindle_Store\meta_Kindle_Store.jsonl"
DEFAULT_IMAGE_DIR = r"D:\amazon-datasets\Kindle_Store\images"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def parse_args():
    p = argparse.ArgumentParser(description="Nhúng item (text + image riêng) qua CLIP, xuất .npy")
    p.add_argument("--meta-file", type=str, default=DEFAULT_META_FILE)
    p.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent))
    p.add_argument("--batch-size", type=int, default=16,
                   help="16 hợp lý cho CPU; tăng lên 128-256 khi chạy GPU (T4 16GB dư sức "
                        "với CLIP-ViT-B/32 ~150M params)")
    p.add_argument("--limit", type=int, default=None, help="Giới hạn số item xử lý (test cục bộ)")
    p.add_argument("--device", type=str, default=None, help="Mặc định: cuda nếu có, ngược lại cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Số worker đọc/decode ảnh song song (overlap I/O với GPU compute)")
    p.add_argument("--gpus", type=str, default=None,
                   help="VD '0,1' — tự spawn 1 process/GPU, mỗi process 1 shard rời rạc, chạy "
                        "song song rồi TỰ GỘP kết quả vào --out-dir (không cần chạy tay 2 lệnh "
                        "shell, dùng khi Kaggle T4x2 có 2 GPU). Bỏ trống = chạy 1 device duy "
                        "nhất (--device, mặc định cuda nếu có).")
    return p.parse_args()


def build_text(record):
    """title + features + category + author gộp thành 1 chuỗi (đã chốt ở Readme mục
    Item tower: nhét category/author dạng text vào cùng prompt nhúng chung với CLIP)."""
    parts = [record.get("title") or ""]

    features = record.get("features") or []
    if features:
        parts.append(" ".join(features))

    cats = record.get("categories") or []
    if cats:
        category_leaf = cats[2] if len(cats) >= 3 else cats[-1]
        parts.append(f"category: {category_leaf}")

    author = record.get("author")
    author_name = author.get("name") if isinstance(author, dict) else None
    parts.append(f"author: {author_name or 'unknown'}")

    return " | ".join(p for p in parts if p)


def find_image_path(image_dir, asin):
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidate = image_dir / f"{asin}.{ext}"
        if candidate.exists():
            return candidate
    return None


def load_items(meta_file, limit=None):
    items = []
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            items.append({
                "asin": record["parent_asin"],
                "text": build_text(record),
            })
            if limit and len(items) >= limit:
                break
    return items


def load_image(path):
    """Đọc + decode 1 ảnh (I/O-bound) — chạy trong ThreadPoolExecutor để overlap với GPU
    compute, xem ImageLoaderPool bên dưới."""
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


class ImageLoaderPool:
    """Đọc ảnh của batch N+1 song song TRONG LÚC GPU đang encode batch N — tránh CPU I/O
    (mở/decode file ảnh) làm nghẽn GPU vốn rất nhanh với CLIP-ViT-B/32 (~150M params).

    submit_many() trả về ngay (không block) — caller gọi nó TRƯỚC khi encode batch hiện
    tại, rồi .result() ở batch kế tiếp mới thật sự đợi (lúc đó ảnh thường đã đọc xong rồi
    vì GPU encode mất nhiều thời gian hơn đọc ảnh). Bug đã sửa: bản trước dùng load_many()
    gọi executor.map(...).list() ngay lập tức — dù đọc song song bên trong nhưng caller vẫn
    BLOCK đợi xong toàn bộ trước khi encode, nên GPU vẫn phải chờ CPU tuần tự y hệt không
    có prefetch gì cả (lỗi thật: GPU 0%/40% dù batch-size=128, VRAM chỉ dùng ~6.6%)."""

    def __init__(self, num_workers):
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))

    def submit_many(self, image_paths):
        futures = [self.executor.submit(load_image, p) for p in image_paths]
        return futures

    def resolve(self, futures):
        return [f.result() for f in futures]

    def close(self):
        self.executor.shutdown(wait=True)


def _as_tensor(output):
    """model.get_text_features/get_image_features NÊN trả thẳng Tensor theo API CLIP chuẩn,
    nhưng một số phiên bản/config transformers trả về object output (BaseModelOutputWithPooling)
    thay vì Tensor trực tiếp — lỗi thật gặp trên Kaggle (AttributeError: ... has no attribute
    'float'). Tự trích đúng tensor bất kể dạng trả về nào, ưu tiên pooler_output."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]  # CLS token, fallback hiếm khi cần tới
    raise TypeError(f"Không nhận diện được kiểu output từ CLIP: {type(output)}")


def embed_batch(model, processor, device, use_amp, texts, images):
    """Trả về (text_emb, image_emb, has_image) — 3 mảng riêng biệt, KHÔNG trộn.
    image_emb là vector 0 cho item thiếu ảnh (không suy đoán/giả lập ảnh nào).
    use_amp: bật torch.autocast (mixed precision) — chỉ có tác dụng thật trên CUDA, không
    ảnh hưởng độ chính xác kết quả cuối theo cách đáng kể cho embedding (chuẩn thực hành
    inference), giảm ~1 nửa VRAM và tăng tốc trên GPU có Tensor Core (T4)."""
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
        text_emb = model.get_text_features(**text_inputs)  # [batch, dim]
    text_emb = _as_tensor(text_emb).float()

    dim = text_emb.shape[1]
    image_emb = torch.zeros(len(texts), dim, device=device)
    has_image = [False] * len(texts)

    real_images = [img for img in images if img is not None]
    if real_images:
        image_inputs = processor(images=real_images, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            image_feats = model.get_image_features(**image_inputs)  # [n_real, dim]
        image_feats = _as_tensor(image_feats).float()
        j = 0
        for i, img in enumerate(images):
            if img is not None:
                image_emb[i] = image_feats[j]
                has_image[i] = True
                j += 1

    return text_emb.cpu().numpy(), image_emb.cpu().numpy(), has_image


def run_shard(shard_items, meta_args, device, out_dir):
    """Embed đúng shard_items (list[(global_idx, item)]) trên 1 device, lưu .npy/.json vào
    out_dir. Dùng chung cho cả đường chạy 1-device (shard = toàn bộ items) và multi-GPU
    (mỗi process con gọi hàm này với 1 shard rời rạc, xem run_multi_gpu)."""
    use_amp = device.startswith("cuda")
    print(f"[{device}] Loading CLIP model...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    image_dir = Path(meta_args["image_dir"])
    all_text_emb, all_image_emb, all_has_image = [], [], []
    asin_to_idx = {}

    batch_size = meta_args["batch_size"]
    batches = [shard_items[i:i + batch_size] for i in range(0, len(shard_items), batch_size)]

    loader_pool = ImageLoaderPool(meta_args["num_workers"])
    try:
        def image_paths_of(batch):
            return [find_image_path(image_dir, it["asin"]) for _, it in batch]

        # Pipeline thật: submit đọc ảnh batch 0 trước vòng lặp, rồi ở MỖI lần lặp submit
        # tiếp batch kế trước khi .resolve() (đợi) ảnh của batch hiện tại — trong lúc GPU
        # encode batch hiện tại (embed_batch bên dưới), ThreadPoolExecutor đã đang đọc ảnh
        # batch kế tiếp song song. Khác bản cũ (đã sửa): gọi load_many() ngay lập tức luôn
        # block caller đợi xong hết mới encode, GPU không có gì overlap (lỗi thật: GPU
        # 0%/40% dù batch-size=128, VRAM chỉ ~6.6%).
        pending_futures = loader_pool.submit_many(image_paths_of(batches[0])) if batches else []

        desc = f"Embedding items [{device}]"
        for i in tqdm(range(len(batches)), desc=desc):
            batch = batches[i]
            images = loader_pool.resolve(pending_futures)  # ảnh batch này (đã đọc song song ở vòng trước)

            if i + 1 < len(batches):
                pending_futures = loader_pool.submit_many(image_paths_of(batches[i + 1]))  # prefetch batch kế

            texts = [it["text"] for _, it in batch]
            text_emb, image_emb, has_image = embed_batch(model, processor, device, use_amp, texts, images)

            for global_idx, it in batch:
                asin_to_idx[it["asin"]] = global_idx

            all_text_emb.append(text_emb)
            all_image_emb.append(image_emb)
            all_has_image.extend(has_image)
    finally:
        loader_pool.close()

    text_embeddings = np.concatenate(all_text_emb, axis=0) if all_text_emb else np.zeros((0, 512), dtype="f4")
    image_embeddings = np.concatenate(all_image_emb, axis=0) if all_image_emb else np.zeros((0, 512), dtype="f4")
    has_image_arr = np.array(all_has_image, dtype=bool)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "text_embeddings.npy", text_embeddings)
    np.save(out_dir / "image_embeddings.npy", image_embeddings)
    np.save(out_dir / "has_image.npy", has_image_arr)
    with open(out_dir / "asin_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_idx, f)
    print(f"[{device}] Done: {len(shard_items):,} items -> {out_dir}")


def _shard_worker(shard_items, meta_args, device, shard_out_dir):
    """Entry point cho multiprocessing.Process — mỗi process con có CUDA context riêng
    biệt gắn với đúng 1 GPU (an toàn hơn threading, vốn không tách CUDA context tốt cho
    nhiều GPU trong cùng 1 process)."""
    run_shard(shard_items, meta_args, device, shard_out_dir)


def run_multi_gpu(items, meta_args, gpu_ids, out_dir):
    """Tự spawn 1 process/GPU (--gpus '0,1'), mỗi process nhận 1 shard rời rạc
    (items[i::n] theo global index gốc), chạy song song, rồi GỘP kết quả lại thành đúng 1
    bộ .npy/.json ở out_dir — không cần người dùng tự mở nhiều shell (Kaggle notebook chỉ
    chạy được 1 cell tuần tự, xem lý do đổi hướng ở đầu module)."""
    n = len(gpu_ids)
    out_dir = Path(out_dir)
    shard_dirs = [out_dir / f"_shard{shard_id}" for shard_id in range(n)]

    # QUAN TRỌNG: PyTorch CUDA không cho phép re-init context trong process con tạo bằng
    # fork (mặc định trên Linux/Kaggle) nếu CUDA đã init ở process cha trước đó — crash
    # "Cannot re-initialize CUDA in forked subprocess" (lỗi thật đã gặp khi chạy trên
    # Kaggle T4x2). Bắt buộc dùng context "spawn" (tạo process hoàn toàn mới, import lại
    # module từ đầu, không kế thừa CUDA context của cha).
    ctx = mp.get_context("spawn")
    procs = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        shard_items = [(i, it) for i, it in enumerate(items) if i % n == shard_id]
        device = f"cuda:{gpu_id}"
        p = ctx.Process(target=_shard_worker, args=(shard_items, meta_args, device, shard_dirs[shard_id]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)}/{n} shard process(es) thất bại — xem log ở trên")

    print("Merging shard outputs...")
    text_parts, image_parts, has_image_parts = [], [], []
    asin_to_idx = {}
    for shard_dir in shard_dirs:
        text_parts.append(np.load(shard_dir / "text_embeddings.npy"))
        image_parts.append(np.load(shard_dir / "image_embeddings.npy"))
        has_image_parts.append(np.load(shard_dir / "has_image.npy"))
        with open(shard_dir / "asin_to_idx.json", encoding="utf-8") as f:
            asin_to_idx.update(json.load(f))

    # asin_to_idx dùng GLOBAL index gốc (xem run_multi_gpu/run_shard) nên ghép trực tiếp
    # theo đúng thứ tự index đó, không phụ thuộc thứ tự các shard hoàn thành trước/sau.
    n_items = len(items)
    dim = text_parts[0].shape[1]
    text_embeddings = np.zeros((n_items, dim), dtype="f4")
    image_embeddings = np.zeros((n_items, dim), dtype="f4")
    has_image_arr = np.zeros(n_items, dtype=bool)
    for shard_id in range(n):
        shard_items = [(i, it) for i, it in enumerate(items) if i % n == shard_id]
        for row, (global_idx, _) in enumerate(shard_items):
            text_embeddings[global_idx] = text_parts[shard_id][row]
            image_embeddings[global_idx] = image_parts[shard_id][row]
            has_image_arr[global_idx] = has_image_parts[shard_id][row]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "text_embeddings.npy", text_embeddings)
    np.save(out_dir / "image_embeddings.npy", image_embeddings)
    np.save(out_dir / "has_image.npy", has_image_arr)
    with open(out_dir / "asin_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_idx, f)

    n_with_image = int(has_image_arr.sum())
    print(f"Items with real image used: {n_with_image}/{n_items}")
    print(f"text_embeddings shape: {text_embeddings.shape}  image_embeddings shape: {image_embeddings.shape}")
    print(f"Saved -> {out_dir / 'text_embeddings.npy'}")
    print(f"Saved -> {out_dir / 'image_embeddings.npy'}")
    print(f"Saved -> {out_dir / 'has_image.npy'}")
    print(f"Saved -> {out_dir / 'asin_to_idx.json'}")


def main():
    args = parse_args()

    print(f"Loading items from {args.meta_file} (limit={args.limit})...")
    items = load_items(args.meta_file, limit=args.limit)
    print(f"Total items: {len(items):,}")

    meta_args = {
        "image_dir": args.image_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        print(f"Multi-GPU: {len(gpu_ids)} process(es) trên GPU {gpu_ids} — tự điều phối, không "
              f"cần chạy tay nhiều lệnh shell.")
        run_multi_gpu(items, meta_args, gpu_ids, args.out_dir)
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        shard_items = list(enumerate(items))
        run_shard(shard_items, meta_args, device, args.out_dir)


if __name__ == "__main__":
    main()
