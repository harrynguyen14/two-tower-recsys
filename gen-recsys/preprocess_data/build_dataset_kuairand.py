"""Orchestrator — chạy toàn bộ 5 pass tiền xử lý KuaiRand-27K theo đúng thứ tự phụ thuộc.

Thứ tự BẮT BUỘC (không đổi):
  Pass 1 (build_n_cumulative)  — N_i/N_category lũy kế, KHÔNG phụ thuộc pass nào khác
  Pass 2 (build_sequences)     — chuỗi user + label_timestamps, KHÔNG phụ thuộc Pass 1
  Pass 3 (build_item_static)   — item static features, độc lập
  Pass 4 (build_user_static)   — user static features, độc lập
  Pass 5 (build_interactions)  — CẦN Pass 1 (tra cứu N_i) VÀ Pass 2 (sequences/labels)
                                  đã chạy xong trước, nên luôn chạy CUỐI CÙNG.

Xem D:\\ama-rs\\idea.md mục "TRẠNG THÁI DỰ ÁN" + mục 4.5 cho toàn bộ lý do thiết kế,
D:\\ama-rs\\gen-recsys (plan file lexical-hopping-lobster.md) cho đặc tả đầy đủ.
"""

from build_interactions import build_interactions
from build_item_static import build_item_static
from build_n_cumulative import build_category_n_cumulative, build_item_n_cumulative
from build_sequences import build_sequences
from build_user_static import build_user_static


def main() -> None:
    print("=== Pass 1: N_i / N_category lũy kế ===")
    build_item_n_cumulative()
    build_category_n_cumulative()

    print("=== Pass 2: chuỗi hành vi user ===")
    build_sequences()

    print("=== Pass 3: item static features ===")
    build_item_static()

    print("=== Pass 4: user static features ===")
    build_user_static()

    print("=== Pass 5: split train/val/test + cờ cold/warm ===")
    build_interactions()

    print("Hoàn tất — xem output/ cho toàn bộ .npy/.npz đã build.")


if __name__ == "__main__":
    main()
