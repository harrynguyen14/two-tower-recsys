"""Pass 3.5 (LOCAL, CPU) — encode caption 27K -> caption_embeddings.npy + caption_has_caption.npy.

[THEM 2026-10-08] `encode_captions_kaggle.py` noi "chay tren Kaggle, KHONG chay local" vi
benchmark cu do 67.3 caption/s => 5.5 ngay. Do LAI tren may nay: **577.6 caption/s**
(nhanh hon 8.6x) => 32,038,725 caption ~ **15.4 gio**, chay local duoc, khong can Kaggle +
khong can sharding (Kaggle gioi han output 20GB; o local chi can disk).

KHAC ban Kaggle o 3 diem:
1. **float16** thay vi float32: 45.8 GB -> **22.9 GB**. Embedding chi di vao GMU
   (item_embedding.py) qua mot Linear, fp16 du chinh xac; `dataset.py` doc bang mmap roi
   torch tu cast. Giam mot nua dung luong la thu quyet dinh viec co the dua len Kaggle.
2. **Khong sharding**: ghi 1 file duy nhat bang np.lib.format.open_memmap, nen RAM khong
   bao gio giu qua 1 batch (ban Kaggle phai chia 4 shard vi gioi han output 20GB).
3. **Ghi theo video_id truc tiep**: CSV khong sort theo video_id, nhung memmap cho phep
   gan ngau nhien `emb[vid] = vec` nen khong can filter theo dai id nhu ban Kaggle.

Do duoc: CSV co 32,038,725 dong, **24,757,607 (77.3%) co caption that**, 0 item thieu dong.
Item khong co caption -> vector 0 + has_caption=False (GMU tu mask, xem gmu.py:51).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CAPTION_CSV = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data\kuairand_video_captions.csv")
NUM_ITEMS = 32_038_725          # video_id lien tuc 0..N-1, da xac nhan
MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384
CHUNK = 200_000                 # dong CSV moi lan doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="chi encode N dong dau (smoke test)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb_path, has_path = out / "caption_embeddings.npy", out / "caption_has_caption.npy"

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(NUM_ITEMS, DIM))
    has = np.zeros(NUM_ITEMS, dtype=bool)

    done = 0
    t0 = time.time()
    for ch in pd.read_csv(CAPTION_CSV, usecols=["final_video_id", "caption"], chunksize=CHUNK):
        ch = ch[ch["caption"].notna()]
        if len(ch):
            vids = ch["final_video_id"].to_numpy(dtype=np.int64)
            # E5 doi prefix "passage: " (asymmetric embedding) — bo prefix se giam chat luong
            txt = ["passage: " + s for s in ch["caption"].astype(str).tolist()]
            vecs = model.encode(txt, batch_size=args.batch_size, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
            ok = (vids >= 0) & (vids < NUM_ITEMS)
            emb[vids[ok]] = vecs[ok].astype(np.float16)
            has[vids[ok]] = True
        done += CHUNK
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        print(f"[encode] {done:,}/{NUM_ITEMS:,} | {rate:.0f} cap/s | "
              f"da {el/3600:.2f}h | con ~{(NUM_ITEMS-done)/max(rate,1)/3600:.2f}h", flush=True)
        if args.limit and done >= args.limit:
            print(f"[encode] dung som do --limit {args.limit}", flush=True)
            break

    emb.flush()
    np.save(has_path, has)
    print(f"[encode] XONG: {emb_path.name} {emb.shape} fp16, has_caption={has.sum():,}", flush=True)


if __name__ == "__main__":
    main()
