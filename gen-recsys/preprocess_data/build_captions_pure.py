"""Pass 3.5 (KuaiRand-Pure) — build caption_embeddings.npy/caption_has_caption.npy từ
embedding đã encode sẵn (xem result.md 2026-09-13 "Luận điểm: vì sao KHÔNG dùng cầu nối
content->collaborative" — caption đã được encode 1 lần cho toàn bộ 7,583 item Pure bằng
multilingual-e5-small trong lúc kiểm chứng thực nghiệm R²).

[KHÁC encode_captions_kaggle.py] Pure chỉ 7,583 item — KHÔNG cần sharding/Kaggle GPU như
27K (32M item, ~49GB). Script này chỉ REMAP embedding đã encode sẵn (lưu tạm ở scratchpad)
về đúng thứ tự index của item_static.npy (video_id 0..N-1, identity mapping — xem
dataset.py), KHÔNG encode lại.

Nếu chưa có embedding đã encode sẵn, hàm _encode_if_missing() sẽ tự encode trực tiếp local
(nhanh, chỉ 7,583 caption — không cần GPU/Kaggle).

Output: caption_embeddings.npy (num_items, 384) float32, caption_has_caption.npy
(num_items,) float32 (1.0/0.0) — cùng index với item_static.npy.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "output"
CAPTION_SOURCE_27K = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data\kuairand_video_captions.csv")
CAPTION_DIM = 384


def _encode_if_missing(video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode trực tiếp local (7,583 item, nhanh) — dùng khi chưa có embedding cache sẵn."""
    from sentence_transformers import SentenceTransformer

    cap = pd.read_csv(CAPTION_SOURCE_27K).rename(columns={"final_video_id": "video_id"})
    cap = cap[cap["video_id"].isin(set(video_ids.tolist()))].copy()
    cap["caption"] = cap["caption"].fillna("")

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    emb = model.encode(cap["caption"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    id_to_row = {int(vid): i for i, vid in enumerate(cap["video_id"].tolist())}
    return emb.astype(np.float32), np.array([id_to_row.get(int(v), -1) for v in video_ids])


def build_captions_pure(cache_emb_path: Path | None = None, cache_ids_path: Path | None = None) -> None:
    item_static = np.load(OUT_DIR / "item_static.npy", mmap_mode="r")
    video_ids = item_static["video_id"]  # index i == video_id (identity mapping, xem dataset.py)
    num_items = len(video_ids)

    if cache_emb_path and cache_emb_path.exists() and cache_ids_path and cache_ids_path.exists():
        print(f"[build_captions_pure] dùng cache đã encode sẵn: {cache_emb_path}")
        cached_emb = np.load(cache_emb_path)
        cached_ids = pd.read_csv(cache_ids_path)["video_id"].to_numpy()
        id_to_row = {int(vid): i for i, vid in enumerate(cached_ids.tolist())}
        row_map = np.array([id_to_row.get(int(v), -1) for v in video_ids])
        emb_by_video = cached_emb
    else:
        print("[build_captions_pure] không có cache, encode trực tiếp local...")
        emb_by_video, row_map = _encode_if_missing(video_ids)

    caption_embeddings = np.zeros((num_items, CAPTION_DIM), dtype=np.float32)
    caption_has_caption = np.zeros(num_items, dtype=np.float32)

    valid = row_map >= 0
    caption_embeddings[valid] = emb_by_video[row_map[valid]]
    caption_has_caption[valid] = 1.0

    np.save(OUT_DIR / "caption_embeddings.npy", caption_embeddings)
    np.save(OUT_DIR / "caption_has_caption.npy", caption_has_caption)
    print(
        f"[build_captions_pure] {num_items} items -> caption_embeddings.npy, "
        f"caption_has_caption.npy ({valid.sum()}/{num_items} có caption thật = {valid.mean()*100:.1f}%)"
    )


if __name__ == "__main__":
    import sys

    cache_emb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cache_ids = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    build_captions_pure(cache_emb, cache_ids)
