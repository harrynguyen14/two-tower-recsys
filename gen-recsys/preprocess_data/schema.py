"""Field mapping cho KuaiRand-Pure -> gen-recsys pipeline.

[SỬA 2026-09-13] Chuyển từ KuaiRand-27K sang KuaiRand-Pure (xem result.md
"CHECKLIST CUỐI CÙNG"/2026-09-13 — 27K có 32M item long-tail (54% N_i=1, không phải
cold-start thật mà là noise-floor vĩnh viễn) và KHÔNG có cold-user (N_u min=100); Pure có
cả cold-user thật (N_u min=1, 5.33% dưới ngưỡng cold=10) lẫn cold-item mang đúng ý nghĩa
(N_i min=1, chỉ 4.15% dưới ngưỡng, không lẫn noise). Chi tiết đo đạc đầy đủ trong
result.md 2026-09-13.

Nguồn field đã xác nhận qua đọc trực tiếp header CSV thật của Pure (không suy đoán).
"""

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
    "play_ratio",           # derived = play_time_ms / duration_ms, clip [0, 1]
    "profile_stay_time_norm",  # derived, chuẩn hóa (log1p + scale)
    "comment_stay_time_norm",  # derived, chuẩn hóa (log1p + scale)
    "is_profile_enter",
]
NUM_ACTION_DIMS = len(ACTION_VECTOR_FIELDS)  # 11

# user_features_pure.csv
USER_STATIC_ONEHOT_FEATS = [f"onehot_feat{i}" for i in range(18)]  # categorical id đã encode, "encrypted" theo dataset gốc
USER_STATIC_FIELD = "register_days"  # static feature phụ, KHÔNG dùng làm N_u

# video_features_basic_pure.csv
VIDEO_BASIC_CATEGORY_FIELD = "tag"  # category 1 cấp DUY NHẤT có trong Pure — dùng cho cả N_category/category_confidence VÀ content_branch
# [CHỐT 2026-09-10, xác nhận lại 2026-09-13] ID lớn — cần nn.Embedding riêng (không phải
# feature thống kê), giống e_collab của video_id.
VIDEO_BASIC_ID_FIELDS = ["author_id", "music_id"]
# categorical nhỏ — video_type: 3 giá trị (NORMAL/AD/UNKNOWN); music_type: 6 giá trị (có NaN, cần fillna trước)
VIDEO_BASIC_CATEGORICAL_FIELDS = ["video_type", "music_type"]

# [SỬA 2026-09-13] KuaiRand-Pure KHÔNG có file category 4 cấp (kuairand_video_categories.csv
# — chỉ tồn tại ở bản 27K). Pure chỉ có `tag` (category 1 cấp, VIDEO_BASIC_CATEGORY_FIELD ở
# trên) — đã CHỐT bỏ hẳn 4 field cat_l1-l4_id thay vì giả lập giá trị rỗng (xem result.md).
VIDEO_CATEGORY_ID_FIELDS: list[str] = []
VIDEO_CATEGORY_MISSING_ID = -124  # giữ lại hằng số (không dùng khi VIDEO_CATEGORY_ID_FIELDS rỗng) để không phá vỡ import ở nơi khác

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
MAX_SEQ_LEN = 256

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
