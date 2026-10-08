"""Field mapping cho KuaiRand-Pure -> gen-recsys pipeline.

[SỬA 2026-09-13] Chuyển từ KuaiRand-27K sang KuaiRand-Pure (xem result.md
"CHECKLIST CUỐI CÙNG"/2026-09-13 — 27K có 32M item long-tail (54% N_i=1, không phải
cold-start thật mà là noise-floor vĩnh viễn) và KHÔNG có cold-user (N_u min=100); Pure có
cả cold-user thật (N_u min=1, 5.33% dưới ngưỡng cold=10) lẫn cold-item mang đúng ý nghĩa
(N_i min=1, chỉ 4.15% dưới ngưỡng, không lẫn noise). Chi tiết đo đạc đầy đủ trong
result.md 2026-09-13.

Nguồn field đã xác nhận qua đọc trực tiếp header CSV thật của Pure (không suy đoán).
"""

import os
from pathlib import Path

# [THÊM 2026-10-07] Chọn dataset bằng env `GEN_RECSYS_DATASET` = "pure" (mặc định) | "27k".
#
# Trước đây `LOG_DIR` + tên file hard-code Pure ở BA chỗ (build_n_cumulative.py:35,
# build_sequences.py:39, build_item_static.py:56) nên muốn chạy 27K phải sửa 3 file và dễ
# sót. Gom về đây làm MỘT nguồn, cùng cách đã dùng cho `GEN_RECSYS_OUT_DIR`.
#
# Header 27K đã đối chiếu trực tiếp: KHỚP TỪNG CỘT, đúng thứ tự với Pure (19 cột
# user_id..tab) nên `LOG_SCHEMA` dùng chung, không cần nhánh riêng.
#
# Khác biệt thật giữa hai bộ (đã kiểm trên đĩa):
#   - 27K có 4 file log_standard (part1/part2 x 2 khoảng thời gian), Pure có 2.
#   - 27K có `kuairand_video_categories.csv` (category 4 cấp); Pure KHÔNG có => nhánh
#     category trong build_item_static tự tắt (nó đã check .exists()).
#   - 27K `upload_dt` có 347 giá trị (age_t dùng được); Pure chỉ 3 => nhánh age tự tắt.
DATASET = os.environ.get("GEN_RECSYS_DATASET", "pure").lower()
if DATASET not in ("pure", "27k"):
    raise ValueError(f"GEN_RECSYS_DATASET phải là 'pure' hoặc '27k', nhận: {DATASET!r}")

if DATASET == "27k":
    LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-27K-extracted\KuaiRand-27K\data")
    LOG_STANDARD_FILES = [
        LOG_DIR / "log_standard_4_08_to_4_21_27k_part1.csv",
        LOG_DIR / "log_standard_4_08_to_4_21_27k_part2.csv",
        LOG_DIR / "log_standard_4_22_to_5_08_27k_part1.csv",
        LOG_DIR / "log_standard_4_22_to_5_08_27k_part2.csv",
    ]
    LOG_RANDOM_FILES = [LOG_DIR / "log_random_4_22_to_5_08_27k.csv"]
    VIDEO_BASIC_FILE = LOG_DIR / "video_features_basic_27k.csv"
    # 27K: median 1,744 tương tác/user, p99 12,915 => K=256 chỉ giữ 9.8% tương tác.
    DEFAULT_MAX_SEQ_LEN = 1024
else:
    LOG_DIR = Path(r"D:\amazon-datasets\KuaiRand-Pure-extracted\KuaiRand-Pure\data")
    LOG_STANDARD_FILES = [
        LOG_DIR / "log_standard_4_08_to_4_21_pure.csv",
        LOG_DIR / "log_standard_4_22_to_5_08_pure.csv",
    ]
    LOG_RANDOM_FILES = [LOG_DIR / "log_random_4_22_to_5_08_pure.csv"]
    VIDEO_BASIC_FILE = LOG_DIR / "video_features_basic_pure.csv"
    DEFAULT_MAX_SEQ_LEN = 256

# Chỉ 27K có file này; Pure không => nhánh category 4 cấp tự tắt (đã check .exists()).
VIDEO_CATEGORIES_FILE = LOG_DIR / "kuairand_video_categories.csv"

# log_standard_*.csv / log_random_*.csv — CÙNG schema, phân biệt qua cột is_rand
LOG_SCHEMA = [
    "user_id", "video_id", "date", "hourmin", "time_ms",
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
]

# action_vector: multi-hot/multi-value, đã chốt 2026-09-09 (idea.md mục 4.5 điểm 4)
# Thứ tự cố định — dùng xuyên suốt pipeline, KHÔNG đổi thứ tự sau khi đã build dữ liệu.
ACTION_VECTOR_FIELDS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view",            # đã là 0/1 trong dataset gốc
    "play_ratio",           # derived [SỬA 2026-09-17 — D2Q]: THỨ HẠNG PHẦN TRĂM của
                            # play_time/duration TRONG NHÓM duration (decile), không
                            # phải tỉ lệ thô. Thời lượng video là biến gây nhiễu — đo
                            # được play_ratio thô giảm đơn điệu 4.55x từ decile ngắn
                            # sang decile dài. Sau khi sửa còn 1.22x. Xem
                            # build_sequences._compute_derived_action_fields.
    "profile_stay_time_norm",  # derived, chuẩn hóa (log1p + scale)
    "comment_stay_time_norm",  # derived, chuẩn hóa (log1p + scale)
    "is_profile_enter",
]
NUM_ACTION_DIMS = len(ACTION_VECTOR_FIELDS)  # 11

# user_features_pure.csv
# [XOA 2026-10-06] USER_STATIC_ONEHOT_FEATS (onehot_feat0-17) va USER_STATIC_FIELD
# (register_days) DA BO cung `build_user_static.py`. Chuoi chet: chung nuoi static_branch cua
# `user_embedding.py`, file do xoa cung FiLM 2026-10-05 (formula.md §3bis).

# video_features_basic_pure.csv
VIDEO_BASIC_CATEGORY_FIELD = "tag"  # category 1 cấp DUY NHẤT có trong Pure — dùng cho cả N_category/category_confidence VÀ content_branch
# [CHỐT 2026-09-10, xác nhận lại 2026-09-13] ID lớn — cần nn.Embedding riêng (không phải
# feature thống kê), giống e_collab của video_id.
VIDEO_BASIC_ID_FIELDS = ["author_id", "music_id"]
# categorical nhỏ — video_type: 3 giá trị (NORMAL/AD/UNKNOWN); music_type: 6 giá trị (có NaN, cần fillna trước)
VIDEO_BASIC_CATEGORICAL_FIELDS = ["video_type", "music_type"]

# [SỬA 2026-10-05] Category 4 cấp — nhánh 3 của content token (formula.md §1).
# CHỈ 27K có kuairand_video_categories.csv; Pure không có file này. build_item_static.py
# tự phát hiện: thiếu file => bỏ 4 field, ItemEmbedding về 2 nhánh (W^fuse d x 2d).
#
# Đo trên 27K: cấp 1 có 39 nhãn và BÃO HOÀ (user chạm median 39/39 = 100% catalog), cấp 2
# có 176 nhãn thì không (76%) — nên mọi phép đo NHÓM phải dùng cấp 2, không phải cấp 1.
VIDEO_CATEGORY_LEVELS = [1, 2, 3, 4]
VIDEO_CATEGORY_JOIN_KEY = "final_video_id"  # khoá join của kuairand_video_categories.csv
VIDEO_CATEGORY_SRC_FIELDS = [
    "first_level_category_id", "second_level_category_id",
    "third_level_category_id", "fourth_level_category_id",
]
VIDEO_CATEGORY_ID_FIELDS = ["cat_l1_id", "cat_l2_id", "cat_l3_id", "cat_l4_id"]
VIDEO_CATEGORY_MISSING_ID = -124  # mã UNKNOWN của dataset gốc -> factorize về CATEGORY_PAD=0

# [THÊM 2026-10-05] Ngày ra mắt item — cần cho age_t (formula.md §2). Có ở CẢ Pure và 27K.
VIDEO_BASIC_UPLOAD_FIELD = "upload_dt"

# video_features_statistic_pure.csv — dùng cho content_branch (KHÔNG dùng để tính N_i, sẽ leak tương lai)
VIDEO_STATIC_STAT_FIELDS = [
    "play_cnt", "like_cnt", "share_cnt", "comment_cnt", "follow_cnt", "collect_cnt",
    "report_cnt", "reduce_similar_cnt",  # tín hiệu tiêu cực cụ thể hơn is_hate
]

# Sliding window — chốt lại 2026-09-14: đo trực tiếp trên output/user_offsets.npy (Pure),
# p99 độ dài chuỗi/user = 234, 256 phủ 99.26% user + là power-of-2 (thân thiện GPU/attention
# kernel hơn 200). Không dùng 512/2048 của HSTU paper — số đó đo trên dataset Meta production
# (user hàng chục nghìn interaction), ở Pure (max=910, mean=53) sẽ toàn padding, tốn compute
# vô ích (attention O(K^2)).
MAX_SEQ_LEN = 256  # mac dinh cho Pure; 27K phai dung --max-seq-len 1024 (xem duoi)

# [DO LAI 2026-10-05 cho 27K] 256 la so cua PURE (p99 do dai chuoi = 234, phu 99.26% user).
# Tren 27K so nay SAI HAN: do tren log_standard part1 (26,858 user, 1/4 du lieu nen chuoi
# day du con dai ~4x) cho median 1,744 | p75 3,292 | p90 5,520 | p99 12,915 | max 67,647.
#
#   K      %user phu tron   %tuong tac giu   compute O((2K)^2)
#   256          6.6%             9.8%            1x
#   1024        30.5%            34.4%           16x
#   2048        56.3%            56.9%           64x
#   4096        82.3%            79.9%          256x
#
# CHOT: K=1024 + CUA SO TRUOT NGAU NHIEN luc train (dataset._window_start). Cua so co dinh
# cat cung vung "xa" — dung thu ma delta_h(x_q) can de hoc pham vi nhin; truot thi qua
# nhieu epoch model thay gan het chuoi ma moi epoch van chi tra 16x.
MAX_SEQ_LEN_27K = 1024

# [THÊM 2026-09-13] Ngưỡng cold-start cho KuaiRand-Pure — đo trực tiếp trên dữ liệu thật
# (xem result.md 2026-09-13 "Chốt COLD_THRESHOLD_N = 10"): tại threshold=10, cold_cold
# chiếm 1.34% (18,932 sample trên ~1.4M) — đủ mẫu để đo Recall/NDCG ổn định, trong khi
# warm_warm vẫn giữ 80.4% làm baseline đáng tin. KHÁC hằng số cũ (=5) đo trên 27K, không
# áp dụng được cho Pure (phân phối N_u/N_i khác hẳn).
COLD_THRESHOLD_N = 10

# [THÊM 2026-09-15] Ngưỡng few-shot cho user — is_user_lowhistory (build_interactions.py).
#
# CẢNH BÁO về comment ngay trên: con số "cold_cold chiếm 1.34% (18,932 sample)" đo khi
# is_user_cold CÒN LÀ threshold-based (N_u < 10). Ngày 2026-09-14 is_user_cold đã đổi sang
# STRICT HOLDOUT (user first_seen > p80) và ô cold_cold SỤP VỀ 0 trong train, 18 trong
# val+test — comment cũ không được cập nhật theo. Đã đo lại trực tiếp 2026-09-15:
#
#   định nghĩa user-cold    | train ô cold/cold | val | test
#   strict holdout (p80)    |          0        |   5 |   13
#   N_u < 10                |     18,883        |  37 |   22
#   N_u < 20                |     31,538        |  81 |   65
#   N_u < 50                |     50,811        | 276 |  170
#
# Chọn 20, KHÔNG phải 10 hay 50:
#   - 10 cho val/test chỉ 37/22 — vẫn quá mỏng để metric ổn định.
#   - 50 cho n lớn nhất nhưng 67.9% train thành "cold user" — mất ý nghĩa phân nhóm,
#     baseline warm không còn đáng tin.
#   - 20 giữ cold_user ở 35.6% train / 15.1% val / 13.0% test — phân nhóm còn ý nghĩa,
#     và train có 31,538 ví dụ cold/cold để γ (confidence_attention.py) HỌC được.
# Val/test vẫn mỏng (81/65): đo được XU HƯỚNG, chưa đo được hiệu ứng nhỏ. Phải ghi rõ
# khoảng tin cậy khi báo cáo ô này, đừng đọc chênh lệch nhỏ là thật.
LOW_HISTORY_N = 20
