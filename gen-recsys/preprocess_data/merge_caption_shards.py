"""Ghép 4 shard caption_embeddings_shard{0-3}.npy (tải từ Kaggle, xem
encode_captions_kaggle.py) thành 1 file caption_embeddings.npy + caption_has_caption.npy
duy nhất trong output/ — dùng memmap cả đọc lẫn ghi, không load hết ~49GB vào RAM.

video_id == index trực tiếp (đã xác nhận identity mapping ở item_static.npy) nên ghép chỉ
đơn giản là nối các shard theo đúng thứ tự 0,1,2,3 (không cần remap gì thêm).
"""

from pathlib import Path

import numpy as np

SHARD_DIR = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\caption-embeding")
OUT_DIR = Path(__file__).parent / "output"
NUM_SHARDS = 4


def merge_shards() -> None:
    shard_embs = [np.load(SHARD_DIR / f"caption_embeddings_shard{k}.npy", mmap_mode="r") for k in range(NUM_SHARDS)]
    shard_caps = [np.load(SHARD_DIR / f"caption_has_caption_shard{k}.npy", mmap_mode="r") for k in range(NUM_SHARDS)]

    total_items = sum(s.shape[0] for s in shard_embs)
    embed_dim = shard_embs[0].shape[1]
    print(f"[merge_caption_shards] tổng {total_items} item, embed_dim={embed_dim}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_out = np.lib.format.open_memmap(
        OUT_DIR / "caption_embeddings.npy", mode="w+", dtype=np.float32, shape=(total_items, embed_dim)
    )
    cap_out = np.lib.format.open_memmap(
        OUT_DIR / "caption_has_caption.npy", mode="w+", dtype=np.bool_, shape=(total_items,)
    )

    write_pos = 0
    for k in range(NUM_SHARDS):
        n = shard_embs[k].shape[0]
        emb_out[write_pos:write_pos + n] = shard_embs[k]
        cap_out[write_pos:write_pos + n] = shard_caps[k]
        write_pos += n
        print(f"[merge_caption_shards] shard {k}: {n} item -> {write_pos}/{total_items}")

    emb_out.flush()
    cap_out.flush()
    assert write_pos == total_items, f"write_pos={write_pos} != total_items={total_items}"
    print(f"[merge_caption_shards] DONE -> {OUT_DIR / 'caption_embeddings.npy'}")


if __name__ == "__main__":
    merge_shards()
