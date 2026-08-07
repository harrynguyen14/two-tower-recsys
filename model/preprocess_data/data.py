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

Chạy: python data.py --meta-file <path> --image-dir <path> --out-dir <path> [--limit N]
"""

import argparse
import json
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
    p.add_argument("--num-shards", type=int, default=1,
                   help="Chia items thành N shard để chạy song song nhiều GPU (mỗi process 1 shard)")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard xử lý bởi process này (0-indexed, dùng với --num-shards trên "
                        "Kaggle T4x2: 2 process --device cuda:0 --shard-id 0 --num-shards 2 và "
                        "--device cuda:1 --shard-id 1 --num-shards 2 chạy song song, out-dir "
                        "khác nhau rồi gộp lại sau)")
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
    """Đọc ảnh của batch N+1 song song trong lúc GPU đang encode batch N — tránh CPU I/O
    (mở/decode file ảnh) làm nghẽn GPU vốn rất nhanh với CLIP-ViT-B/32 (~150M params)."""

    def __init__(self, num_workers):
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))

    def load_many(self, image_paths):
        return list(self.executor.map(load_image, image_paths))

    def close(self):
        self.executor.shutdown(wait=True)


def embed_batch(model, processor, device, use_amp, texts, images):
    """Trả về (text_emb, image_emb, has_image) — 3 mảng riêng biệt, KHÔNG trộn.
    image_emb là vector 0 cho item thiếu ảnh (không suy đoán/giả lập ảnh nào).
    use_amp: bật torch.autocast (mixed precision) — chỉ có tác dụng thật trên CUDA, không
    ảnh hưởng độ chính xác kết quả cuối theo cách đáng kể cho embedding (chuẩn thực hành
    inference), giảm ~1 nửa VRAM và tăng tốc trên GPU có Tensor Core (T4)."""
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
        text_emb = model.get_text_features(**text_inputs)  # [batch, dim]
    text_emb = text_emb.float()

    dim = text_emb.shape[1]
    image_emb = torch.zeros(len(texts), dim, device=device)
    has_image = [False] * len(texts)

    real_images = [img for img in images if img is not None]
    if real_images:
        image_inputs = processor(images=real_images, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            image_feats = model.get_image_features(**image_inputs)  # [n_real, dim]
        image_feats = image_feats.float()
        j = 0
        for i, img in enumerate(images):
            if img is not None:
                image_emb[i] = image_feats[j]
                has_image[i] = True
                j += 1

    return text_emb.cpu().numpy(), image_emb.cpu().numpy(), has_image


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda")
    print(f"Device: {device}  AMP: {use_amp}")

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    print(f"Loading items from {args.meta_file} (limit={args.limit})...")
    items = load_items(args.meta_file, limit=args.limit)
    print(f"Total items: {len(items):,}")

    # Sharding: mỗi process chỉ xử lý items[shard_id::num_shards] — chạy 2 process song
    # song (--device cuda:0 --shard-id 0 --num-shards 2 và cuda:1 --shard-id 1 --num-shards 2)
    # trên Kaggle T4x2. global_idx GIỮ NGUYÊN vị trí gốc trong toàn bộ items (không phải vị
    # trí cục bộ trong shard) để asin_to_idx.json của các shard gộp lại không bị trùng/lệch.
    shard_items = [(i, it) for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_items):,} items")

    image_dir = Path(args.image_dir)
    all_text_emb, all_image_emb, all_has_image = [], [], []
    asin_to_idx = {}

    loader_pool = ImageLoaderPool(args.num_workers)
    try:
        for start in tqdm(range(0, len(shard_items), args.batch_size), desc="Embedding items"):
            batch = shard_items[start:start + args.batch_size]
            texts = [it["text"] for _, it in batch]
            image_paths = [find_image_path(image_dir, it["asin"]) for _, it in batch]
            images = loader_pool.load_many(image_paths)  # overlap I/O với GPU compute batch trước

            text_emb, image_emb, has_image = embed_batch(model, processor, device, use_amp, texts, images)

            for global_idx, it in batch:
                asin_to_idx[it["asin"]] = global_idx

            all_text_emb.append(text_emb)
            all_image_emb.append(image_emb)
            all_has_image.extend(has_image)
    finally:
        loader_pool.close()

    text_embeddings = np.concatenate(all_text_emb, axis=0)
    image_embeddings = np.concatenate(all_image_emb, axis=0)
    has_image_arr = np.array(all_has_image, dtype=bool)

    n_with_image = int(has_image_arr.sum())
    print(f"Items with real image used: {n_with_image}/{len(items)}")
    print(f"text_embeddings shape: {text_embeddings.shape}  image_embeddings shape: {image_embeddings.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "text_embeddings.npy", text_embeddings)
    np.save(out_dir / "image_embeddings.npy", image_embeddings)
    np.save(out_dir / "has_image.npy", has_image_arr)
    with open(out_dir / "asin_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(asin_to_idx, f)

    print(f"Saved -> {out_dir / 'text_embeddings.npy'}")
    print(f"Saved -> {out_dir / 'image_embeddings.npy'}")
    print(f"Saved -> {out_dir / 'has_image.npy'}")
    print(f"Saved -> {out_dir / 'asin_to_idx.json'}")


if __name__ == "__main__":
    main()
