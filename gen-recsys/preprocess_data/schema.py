"""Field mapping cho KuaiRand-27K -> gen-recsys pipeline.

Nguồn field đã xác nhận qua đọc trực tiếp header CSV thật (không suy đoán) —
xem D:\\ama-rs\\idea.md mục 0 và mục 4.5 cho lý do/quyết định đầy đủ.
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

# user_features_27k.csv
USER_STATIC_ONEHOT_FEATS = [f"onehot_feat{i}" for i in range(18)]  # categorical id đã encode, "encrypted" theo dataset gốc
USER_STATIC_FIELD = "register_days"  # static feature phụ, KHÔNG dùng làm N_u

# video_features_basic_27k.csv
VIDEO_BASIC_CATEGORY_FIELD = "tag"  # category 1 cấp cũ — GIỮ NGUYÊN, vẫn dùng cho N_category/conf_content
# [CHỐT 2026-09-10] ID lớn — cần nn.Embedding riêng (không phải feature thống kê), giống collaborative_branch
# của video_id. author_id: 8,839,735 unique. music_id: 14,155,985 unique (range tới ~9.48 tỷ, PHẢI dùng int64).
VIDEO_BASIC_ID_FIELDS = ["author_id", "music_id"]
# categorical nhỏ — video_type: 3 giá trị (NORMAL/AD/UNKNOWN); music_type: 6 giá trị (có NaN, cần fillna trước)
VIDEO_BASIC_CATEGORICAL_FIELDS = ["video_type", "music_type"]

# kuairand_video_categories.csv — bổ sung 2026-09-10, category 4 cấp phân cấp, chi tiết hơn `tag` đơn cấp ở
# trên. GIỮ SONG SONG với tag cũ (không thay thế) — tag cũ tiếp tục dùng cho N_category/conf_content, 4 cấp
# mới dùng làm feature bổ sung riêng trong content_branch. Missing đánh dấu id=-124/name="UNKNOWN" (tần suất
# missing tăng dần theo cấp sâu: level1 0%, level2 35%, level3 69%, level4 91.8% — đã đo trực tiếp).
VIDEO_CATEGORY_ID_FIELDS = [
    "first_level_category_id",
    "second_level_category_id",
    "third_level_category_id",
    "fourth_level_category_id",
]
VIDEO_CATEGORY_MISSING_ID = -124

# video_features_statistic_27k_part{1,2,3}.csv — dùng cho content_branch (KHÔNG dùng để tính N_i, sẽ leak tương lai)
VIDEO_STATIC_STAT_FIELDS = [
    "play_cnt", "like_cnt", "share_cnt", "comment_cnt", "follow_cnt", "collect_cnt",
    "report_cnt", "reduce_similar_cnt",  # bổ sung 2026-09-09 — tín hiệu tiêu cực cụ thể hơn is_hate
]

# Sliding window — đã chốt 2026-09-09 (idea.md mục 4.5 điểm 4)
MAX_SEQ_LEN = 200
