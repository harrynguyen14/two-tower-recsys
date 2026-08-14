"""CLI args cho main.py — mọi default lấy đúng con số đã chốt trong Readme.md."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Two-Tower cold-start recsys (Kindle_Store)")

    # data paths
    p.add_argument("--reviews-file", type=str,
                    default=r"D:\amazon-datasets\Kindle_Store\Kindle_Store.jsonl")
    p.add_argument("--meta-file", type=str,
                    default=r"D:\amazon-datasets\Kindle_Store\meta_Kindle_Store.jsonl")
    p.add_argument("--item-emb-dir", type=str, default=None,
                    help="Thư mục chứa text_embeddings.npy/image_embeddings.npy/has_image.npy/"
                         "asin_to_idx.json (Giai đoạn 1, xem model/preprocess_data/data.py)")
    p.add_argument("--cache-dir", type=str, default=None,
                    help="Thư mục chứa metadata.npy/by_user.npy/train_interactions.npy/... "
                         "(xuất bởi model/preprocess_data/build_cache.py). None = tự quét lại "
                         "toàn bộ JSONL mỗi lần chạy (chậm, chỉ dùng khi chưa có cache).")
    p.add_argument("--checkpoint-path", type=str, default="checkpoint.pt",
                    help="File lưu model/optimizer state_dict + epoch sau mỗi epoch (để resume).")
    p.add_argument("--best-checkpoint-path", type=str, default="best_model.pt",
                    help="File lưu model state_dict mỗi khi val_cold AUC cải thiện.")
    p.add_argument("--resume", action="store_true",
                    help="Load lại --checkpoint-path (nếu tồn tại) và tiếp tục train từ epoch kế tiếp.")

    # sequence / user-tower (Readme: max_seq_len=10 đã chốt)
    p.add_argument("--max-seq-len", type=int, default=10)
    p.add_argument("--seq-hidden-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)

    # towers output dim
    p.add_argument("--item-out-dim", type=int, default=128)
    p.add_argument("--user-out-dim", type=int, default=128)
    p.add_argument("--mlp-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # loss (Readme: InfoNCE multi-negative K=8 = 4 hard + 4 soft, đã chốt)
    p.add_argument("--n-hard-neg", type=int, default=4)
    p.add_argument("--n-soft-neg", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--in-batch-neg", action="store_true",
                    help="Thêm pos_vector của sample khác trong cùng batch làm negative bổ "
                         "sung (kỹ thuật NVIDIA Merlin two-tower) — miễn phí về CPU/GPU vì "
                         "item_pos đã encode sẵn trong forward pass, chỉ thêm 1 phép nhân "
                         "ma trận. Không thay thế hard/soft-negative hiện có.")

    # temporal split (Readme: global cutoff percentile 80/90, đã chốt)
    p.add_argument("--train-percentile", type=int, default=80)
    p.add_argument("--val-percentile", type=int, default=90)

    # training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker process — song song hoá __getitem__ (soft-negative "
                         "sampling + sequence lookup) trên CPU trong lúc GPU train batch trước.")
    p.add_argument("--prefetch-factor", type=int, default=4,
                    help="Số batch mỗi worker prefetch trước — tăng để worker luôn có sẵn "
                         "batch chờ GPU, giảm thời gian GPU idle giữa các step.")
    p.add_argument("--amp", action="store_true",
                    help="Bật mixed precision (autocast + GradScaler) cho vòng train chính "
                         "— tận dụng Tensor Core trên GPU (T4/V100+), không ảnh hưởng "
                         "data loading (bottleneck riêng, xem --num-workers).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--log-every", type=int, default=50,
                    help="In loss (+ GPU util nếu cuda) mỗi N step trong lúc train, thay vì "
                         "chỉ biết train_loss trung bình ở cuối epoch (epoch dài hàng giờ thì "
                         "quá chậm để debug NaN/theo dõi tiến độ theo cách đó).")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()
