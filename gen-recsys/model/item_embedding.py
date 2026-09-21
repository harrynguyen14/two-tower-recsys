"""Item embedding module — biến 1 item thành 1 embedding e_i_final, dùng làm token trong
chuỗi user hoặc làm candidate lúc decode (KHÔNG phải "item tower" — xem lưu ý thuật ngữ ở
idea.md mục 4.5 đầu mục "Đề xuất thiết kế cụ thể").

[SỬA 2026-09-13] Đổi tên biến cho tường minh (xem result.md "CHECKLIST CUỐI CÙNG"),
KHÔNG đổi công thức:
    mat_i             -> item_weight
    conf_content      -> category_confidence
    content_branch_shrunk -> category_content_shrunk
    collaborative_branch  -> e_collab
    content_branch        -> e_content
    E_i                   -> e_i_final

Công thức [SỬA 2026-09-16 — xem lý do đầy đủ ở __init__/forward]:
    category_confidence(i) = tanh(N_category(i) / τ_c)
    w                      = sigmoid(MLP([item_weight, category_confidence, e_collab, e_content]))
    e_i_final              = w_collab · e_collab(i) + w_content · e_content(i)

Khác công thức cũ ở hai chỗ, cả hai đều từ chẩn đoán ĐO ĐƯỢC:
  - BỎ `category_content_shrunk = e_content · category_confidence`. Phép nhân vô điều kiện
    này đã vô hiệu hoá nhánh content: ‖e_content‖=18.57 -> ‖e_content·c‖=0.0238 (co ~780
    lần), vì c kẹt ~0.0006 do τ_c sai thang đo. c giờ vào GATE, không nhân thẳng vào vector.
  - g_i (1 kênh + sigmoid) -> w (2 kênh + sigmoid ĐỘC LẬP), và ReLU -> SiLU trong MLP ẩn.

e_collab: học từ đầu theo video_id — thuần ID embedding, KHÔNG dùng feature nào khác (đã
chốt 2026-09-10). Do catalog lớn, cần chia model parallelism (item_id % 2 -> GPU0/GPU1,
xem idea.md mục "TRẠNG THÁI DỰ ÁN" quyết định #1) — CHƯA làm ở module này (single-GPU
trước, xem TODO ở EmbeddingConfig), thêm khi có 2 GPU thật để test.

[GỠ 2026-09-17] CuckooEmbedding đã GỠ khỏi đường chạy. Nó nén bảng ID lớn (ByteDance
Monolith, "Collisionless Embedding Table") — cần thiết khi catalog hàng chục triệu ID.
KuaiRand-Pure chỉ có 7,583 item: toàn bộ bảng embedding 64 chiều là 1.85 MB. Nén 1.85 MB
không giải quyết vấn đề gì, trong khi resolve() dựa trên dict Python đo được ~0.38 s/step —
tức CHÍNH LÀ nút thắt CPU mà train.py than phiền. File `cuckoo_embedding.py` giữ nguyên,
import lại được khi chuyển sang catalog lớn (KuaiRand-27K).

[SỬA 2026-09-17] `sparse=False` (trước là True). Gradient thưa buộc phải dùng SparseAdam,
mà SparseAdam không gộp được với optimizer dày đặc, không dùng được `fused=True`, và buộc
GradScaler đi đường coalesce chậm khi bật AMP. Với 7,583 item thì Adam dày đặc chỉ tốn thêm
~3.7 MB state — đổi lại được MỘT optimizer duy nhất cho toàn model.

e_content: qua GMU (gmu.py) fuse các nhánh tĩnh của item — categorical (category 1 cấp +
video_type + music_type) + author/music (ID embedding nhỏ). author_idx/music_idx CŨNG nằm
trong e_content (đã CHỐT 2026-09-10 — "author/music là 1 nhánh trong content qua GMU",
KHÔNG tách riêng thành số hạng thứ 3 trong e_i_final, giữ đúng công thức 2 nhánh).

[SỬA 2026-09-13] Chuyển sang KuaiRand-Pure (xem schema.py, result.md "CHECKLIST CUỐI
CÙNG"): BỎ 4 field category 4 cấp (cat_l1-l4_id — Pure không có file nguồn) và BỎ
stat_features (review.md L9 — snapshot cuối kỳ leak thời gian, corr≈0.92 với N_i cuối kỳ
qua mô phỏng) khỏi content_gmu. content_gmu giờ chỉ còn 5 modality: category_id,
video_type_id, music_type_id, author, music (+ caption optional).

[CHỐT 2026-09-11] Nhánh caption (text tiếng Trung, qua multilingual-e5-small, 384 chiều,
encode offline — xem encode_captions_kaggle.py) là nhánh OPTIONAL thứ 8 trong content_gmu
— đúng tinh thần "GMU tổng quát, text/image thêm sau chỉ cần thêm entry vào dict, không
sửa module" đã thiết kế từ đầu (xem gmu.py). Item KHÔNG có caption thật
(caption_has_caption.npy = False) truyền mask=0, GMU tự loại nhánh này khỏi softmax gate
cho đúng item đó.

[CẢNH BÁO review.md L9] `stat_features` (VIDEO_STATIC_STAT_FIELDS: play_cnt, like_cnt...)
là SNAPSHOT CUỐI KỲ (tổng cộng dồn tới lúc thu thập dataset) — nếu build lại cho Pure,
PHẢI kiểm tra/sửa để tránh leak thời gian tương tự đã phát hiện trên 27K (corr ước lượng
~0.92 với N_i cuối kỳ). Đây là việc của build_item_static.py (preprocess), KHÔNG sửa ở
module này — chỉ ghi chú lại để không quên khi build pipeline cho Pure.

Input: item_static.npy (Pass 3, xem build_item_static.py) — mỗi field int32 đã factorize
sẵn thành index liên tục [0, n), category_id/video_type_id/music_type_id/cat_l1-l4_id
dùng nn.Embedding riêng từng field (KHÔNG concat one-hot — số category nhỏ nhưng tách
riêng để mỗi field có không gian embedding riêng, không ép chung 1 bảng). caption_embedding
đọc riêng từ caption_embeddings.npy/caption_has_caption.npy (Pass 3.5, merge_caption_shards.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmu import GMU

# Field categorical trong item_static.npy dùng embedding riêng (tên field -> tên tham số
# num_embeddings tương ứng khi khởi tạo ItemEmbeddingConfig) — KHÔNG gồm author_idx/
# music_idx (embedding lớn, tách riêng dict `id_embeddings` vì kích thước rất khác biệt
# so với category nhỏ, dù cùng "categorical" về bản chất).
CATEGORICAL_FIELDS = ["category_id", "video_type_id", "music_type_id"]


class ItemEmbeddingConfig:
    def __init__(
        self,
        num_items: int,
        num_authors: int,
        num_music: int,
        num_categories: dict[str, int],  # field name (trong CATEGORICAL_FIELDS) -> số lượng category
        dim: int = 64,
        cat_embed_dim: int = 16,
        id_embed_dim: int = 16,
        caption_dim: int = 384,  # multilingual-e5-small, xem encode_captions_kaggle.py
    ):
        self.num_items = num_items
        self.num_authors = num_authors
        self.num_music = num_music
        self.num_categories = num_categories
        self.dim = dim
        self.cat_embed_dim = cat_embed_dim
        self.id_embed_dim = id_embed_dim
        self.caption_dim = caption_dim


class ItemEmbedding(nn.Module):
    def __init__(self, config: ItemEmbeddingConfig):
        super().__init__()
        self.config = config

        # e_collab — thuần ID embedding, KHÔNG dùng feature nào khác (đã chốt).
        # sparse=False: xem docstring module. Bảng nhỏ (7,583 item = 1.85 MB) nên Adam dày
        # đặc rẻ, và nó cho phép MỘT optimizer duy nhất + fused + AMP không phải đi đường
        # coalesce của GradScaler.
        self.collab_embedding = nn.Embedding(config.num_items, config.dim)
        self.author_embedding = nn.Embedding(config.num_authors, config.id_embed_dim)
        self.music_embedding = nn.Embedding(config.num_music, config.id_embed_dim)

        # e_content — categorical nhỏ (mỗi field 1 bảng embedding riêng, KHÔNG sparse vì số
        # lượng category nhỏ, dense Adam state không đáng kể) + numeric (8 stat feature, đưa
        # thẳng vào GMU dạng vector liên tục, không cần embedding).
        self.category_embeddings = nn.ModuleDict({
            field: nn.Embedding(config.num_categories[field], config.cat_embed_dim) for field in CATEGORICAL_FIELDS
        })

        gmu_in_dims = {field: config.cat_embed_dim for field in CATEGORICAL_FIELDS}
        gmu_in_dims["author"] = config.id_embed_dim
        gmu_in_dims["music"] = config.id_embed_dim
        gmu_in_dims["caption"] = config.caption_dim  # nhánh OPTIONAL, xem docstring module
        self.content_gmu = GMU(gmu_in_dims, dim=config.dim)

        # w = softmax(MLP([item_weight, category_confidence, e_collab, e_content])) — trộn
        # collab/content (xem idea.md mục 4.5 điểm 2).
        #
        # [SỬA 2026-09-16] Ba thay đổi, đều từ chẩn đoán đo được — xem forward().
        #
        # 1. THÊM category_confidence vào đầu vào. Trước đây gate chỉ thấy `item_weight`,
        #    tức nó biết "item này có đủ dữ liệu riêng chưa" nhưng KHÔNG biết "content có
        #    đáng tin không". Nên với item cold nó đẩy trọng số sang content kể cả khi
        #    content cũng vô giá trị (category mới, chưa có video nào cùng tag). Ba tình
        #    huống gate PHẢI phân biệt được, và chỉ phân biệt được khi thấy cả (m, c):
        #        m cao, c cao  -> collab   (đã đủ dữ liệu riêng)
        #        m thấp, c cao -> content  (item mới nhưng category quen)
        #        m cao, c thấp -> collab
        #    Tình huống thứ tư (m thấp, c thấp) — item hoàn toàn mới, không metadata — KHÔNG
        #    có lời giải bằng kiến trúc: không có thông tin nào để xếp hạng. Nó nằm ngoài
        #    phạm vi model, cần cơ chế exploration ở tầng hệ thống cấp cho item vài tương
        #    tác đầu tiên trước khi vào recsys. Ghi nhận, không xử lý ở đây.
        #
        # 2. ReLU -> SiLU. Đo thật: g_i trung bình = 0.0367, tức pre-activation nằm sâu ở
        #    vùng âm, nơi ReLU cho gradient ĐÚNG BẰNG 0. SiLU vẫn dẫn gradient ở đó. Cũng
        #    nhất quán với HSTU (dùng SiLU).
        #
        # 3. Đầu ra 2 kênh. [SỬA 2026-09-18] softmax -> SIGMOID ĐỘC LẬP cho mỗi kênh.
        #
        #    Lập luận cũ ("giữ tổ hợp lồi vì công thức trộn chỉ đúng khi trọng số tổng 1")
        #    đã BỊ BÁC BỎ bằng đo thật sau 3000 step:
        #        warm_warm:           w_collab=0.276  ‖collab‖=7.94  ‖content‖=10.22
        #        warm_user_cold_item: w_collab=0.414  ‖collab‖=7.89  ‖content‖=11.81
        #    ‖collab‖ KHÔNG ĐỔI giữa item warm và item cold. Item có hàng trăm tương tác
        #    đáng lẽ phải cho biểu diễn ID sắc nét hơn hẳn item mới — ở đây nó phẳng. Và
        #    ngay ở warm_warm (96% dữ liệu) gate chỉ đặt 27.6% trọng số vào collab, tức
        #    model KHÔNG TIN nhánh ID của chính nó.
        #
        #    Nguyên nhân là ràng buộc tổng = 1: collab và content CẠNH TRANH. Nhưng một item
        #    vừa LÀ CHÍNH NÓ (ID) vừa THUỘC thể loại/tác giả nào đó (content) — cả hai cùng
        #    đúng, cùng đầy đủ, KHÔNG loại trừ nhau. Softmax buộc phải chọn, nên gradient về
        #    collab luôn bị nhân 0.276, học chậm ~3.6× so với content. Hệ quả đo được:
        #    warm_warm recall@20 = 0.2246 THẤP HƠN cold_user_warm_item = 0.3190, tức cơ chế
        #    cold-start đang che lấp một nền biểu diễn yếu chứ không phải nền đó tốt.
        #
        #    Đây ĐÚNG lỗi đã sửa ở action vector ngày 2026-09-17 (xem action_encoder.py):
        #    softmax gate trên các nguồn KHÔNG thay thế được cho nhau. Cùng dẫn chứng:
        #    arXiv:2405.13997 (NeurIPS 2024) — softmax gating gây "unnecessary competition
        #    among experts, potentially causing representation collapse".
        #
        #    Sigmoid giữ nguyên miền [0,1] (không có trọng số âm hay >1, không "trừ"
        #    e_content), chỉ BỎ ràng buộc tổng = 1 -> hai nhánh lên xuống độc lập. Vẫn KHÔNG
        #    dùng SiLU/GELU (không chặn miền) hay hardsigmoid (gradient 0 cứng ngoài [-3,3],
        #    đúng bệnh đã chẩn ở τ_c).
        #
        #    ĐÁNH ĐỔI: ‖e_i_final‖ giờ có thể tới 2× thay vì bị chuẩn hoá về 1× — theo dõi
        #    ‖e_user‖ vs ‖e_pos‖ trong chẩn đoán (trước khi sửa: 10.20 vs 7.66). Nếu loss nổ
        #    ở vài trăm step đầu, đây là chỗ nhìn trước tiên.
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 + config.dim + config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, 2),  # [score_collab, score_content]
        )

        # Ảnh chụp thành phần nội bộ của forward() gần nhất — chỉ để chẩn đoán, xem forward().
        self._last_stats: dict[str, torch.Tensor] = {}

        # [THÊM 2026-09-21] COLLAB WARM-UP. Bật cờ này thì gate bị BỎ QUA: w_collab=1,
        # w_content=0 cứng, nhánh content không chạy. Dùng cho vài trăm step đầu để
        # collab_embedding học trong môi trường KHÔNG có đối thủ.
        #
        # VÌ SAO CẦN. Đo 2026-09-21: collab_embedding chưa bao giờ học được gì qua CẢ 6
        # cấu hình đã thử — cosine-std của bảng = 0.1254 trong khi vector ngẫu nhiên độc
        # lập ở dim=64 cho 0.1250, tức bảng vẫn nguyên nhiễu khởi tạo. Nguyên nhân là cuộc
        # đua không cân sức giữa hai nhánh:
        #
        #   nhánh      tham số                    nhận gradient    norm init -> sau 1 epoch
        #   collab     bảng 7583x64, hàng riêng   vài lần/epoch    7.83 -> 8.03 (đứng yên)
        #   content    MLP DÙNG CHUNG mọi item    MỖI step         0.89 -> 8.58 (học được)
        #
        # Content học nhanh gấp bội vì trọng số dùng chung. Gate — vốn làm ĐÚNG việc của nó
        # — thấy collab vô dụng nên hạ w_collab từ 0.498 xuống 0.052 chỉ trong 25 step, đáy
        # 0.0030 ở step 50. Từ đó gradient tới e_collab bị nhân ~0.01, collab mất cơ hội
        # vĩnh viễn, và 2,975 step còn lại của epoch chạy không tải: sau 200 step
        # cos(W, W_0) = 0.9998, bảng gần như không xoay.
        #
        # ĐÃ LOẠI TRỪ bằng đo đạc (không phải phỏng đoán): SparseAdam->Adam (mô phỏng cho
        # thấy dense dịch chuyển NHIỀU hơn, 4.02 vs 2.95); gate bias lệch (±0.05 ở cả 6
        # ckpt); item_weight nhân vào output (không có, nó chỉ là input của gate); thiếu dữ
        # liệu (item N>200 nhận 10,480 lượt chạm/epoch — thừa cho 64 chiều); gradient triệt
        # tiêu (tỉ lệ kéo/đẩy 20:1, lành mạnh); gradient bị chặn (2,120 hàng có grad != 0).
        #
        # VÌ SAO KHÔNG DÙNG WARM-UP GATE (ép w_collab >= 0.5 rồi thả). Cách đó bắt model
        # dùng NHIỄU trong giai đoạn warm-up, làm hỏng luôn nhánh content đang học tốt.
        # Ở đây content bị TẮT hẳn nên nó không bị kéo theo, và khi gate mở ra thì collab
        # đã có nội dung thật — gate không còn lý do đóng nó xuống 0.003.
        #
        # Giai đoạn 1 nên NGẮN (vài trăm step): chỉ cần collab thoát nhiễu, không cần hội
        # tụ. Theo dõi bằng cos.std của bảng — rời khỏi 0.125 là có tác dụng. Kéo dài sẽ
        # khiến e_user (từ decoder) bị kéo về chỗ chỉ hợp với collab rồi phải học lại.
        self.collab_warmup = False

    def forward(
        self,
        video_idx: torch.Tensor,  # (B,) int64 — index vào collab_embedding (0..num_items-1)
        category_ids: dict[str, torch.Tensor],  # field -> (B,) int64
        author_idx: torch.Tensor,  # (B,) int64
        music_idx: torch.Tensor,  # (B,) int64
        item_weight: torch.Tensor,  # (B,) float32 — tanh(N_i / τ_i), đã tính sẵn ở caller
        category_confidence: torch.Tensor,  # (B,) float32 — tanh(N_category(i) / τ_c), đã tính sẵn
        caption_embedding: torch.Tensor,  # (B, caption_dim) float32 — 0 nếu không có caption thật
        caption_mask: torch.Tensor,  # (B,) float32/bool — 1 nếu item CÓ caption thật, 0 nếu không
    ) -> torch.Tensor:
        e_collab = self.collab_embedding(video_idx)  # (B, dim)

        # [THÊM 2026-09-21] Giai đoạn warm-up: collab là tín hiệu DUY NHẤT. Bỏ qua gate và
        # bỏ luôn việc chạy content_gmu (nhánh nặng nhất: GMU 6 modality + caption MLP
        # 384->64), nên giai đoạn này còn RẺ HƠN train thường. Xem __init__ self.collab_warmup.
        if self.collab_warmup:
            self._last_stats = {
                "g_i": torch.ones_like(item_weight),
                "category_confidence": category_confidence.detach(),
                "item_weight": item_weight.detach(),
                "norm_collab": e_collab.detach().norm(dim=-1),
                "norm_content": torch.zeros_like(item_weight),
                "norm_content_shrunk": torch.zeros_like(item_weight),
            }
            return e_collab

        gmu_inputs = {field: self.category_embeddings[field](category_ids[field]) for field in CATEGORICAL_FIELDS}
        gmu_inputs["author"] = self.author_embedding(author_idx)
        gmu_inputs["music"] = self.music_embedding(music_idx)
        gmu_inputs["caption"] = caption_embedding
        e_content = self.content_gmu(gmu_inputs, masks={"caption": caption_mask})  # (B, dim)

        # [SỬA 2026-09-16] BỎ `e_content * category_confidence`. Phép nhân này co nhánh
        # content một cách VÔ ĐIỀU KIỆN và đã vô hiệu hoá nó hoàn toàn: đo thật
        # ‖e_content‖=18.57 -> ‖e_content·c‖=0.0238, co ~780 lần, vì c kẹt ở ~0.0006 do
        # τ_c=53486 sai thang đo (đã sửa về 34.0, xem learnable_thresholds.py).
        #
        # Nhưng kể cả khi τ_c đúng, nhân thẳng vào embedding vẫn là ràng buộc CỨNG do người
        # áp đặt: nó ép "category ít video => vector content phải ngắn lại", trộn hai việc
        # khác nhau — ĐỘ TIN CẬY của content và ĐỘ LỚN của nó. c giờ vào gate làm đầu vào,
        # để model tự học dùng nó thế nào. Cùng tinh thần đã dùng cho δ: thay cửa sổ cứng
        # 5 token bằng độ dốc học được.
        gate_input = torch.cat(
            [item_weight.unsqueeze(-1), category_confidence.unsqueeze(-1), e_collab, e_content],
            dim=-1,
        )
        w = torch.sigmoid(self.gate_mlp(gate_input))  # (B, 2), mỗi kênh [0,1] ĐỘC LẬP
        w_collab, w_content = w[:, 0:1], w[:, 1:2]  # mỗi cái (B, 1)

        # [CHẨN ĐOÁN 2026-09-16] Ghi lại thành phần nội bộ để evaluate() đọc, KHÔNG đổi giá
        # trị trả về. Hai câu hỏi ban đầu ĐÃ được trả lời bằng bảng này:
        #   - gate có phản ứng với m không? CÓ: w_collab 0.0367 (warm) -> 0.1857 (item cold).
        #   - c có biến thiên không? KHÔNG, và đó là bug τ_c đã sửa.
        # Giữ lại để XÁC NHẬN sau khi sửa: kỳ vọng c p50≈0.26 std≈0.39 (thay vì 0.0007/0.0016)
        # và ‖content‖ không còn bị co.
        # detach(): thuần quan sát, không để lọt vào đồ thị gradient.
        self._last_stats = {
            "g_i": w_collab.detach().squeeze(-1),  # giữ tên cũ: vẫn là "trọng số cho collab"
            "category_confidence": category_confidence.detach(),
            "item_weight": item_weight.detach(),
            "norm_collab": e_collab.detach().norm(dim=-1),
            "norm_content": e_content.detach().norm(dim=-1),
            "norm_content_shrunk": (w_content.detach() * e_content.detach()).norm(dim=-1),
        }

        return w_collab * e_collab + w_content * e_content  # (B, dim) = e_i_final

    def dense_parameters(self) -> list[nn.Parameter]:
        """MỌI tham số — không còn chia sparse/dense.

        [GỠ 2026-09-17] Trước đây 3 bảng ID lớn dùng sparse=True + SparseAdam riêng, phần còn
        lại dùng Adam. Với 7,583 item (bảng 64 chiều = 1.85 MB) việc chia đó không đáng: nó
        buộc phải nuôi 2 optimizer, chặn `fused=True`, và bắt GradScaler đi đường coalesce khi
        bật AMP. Giữ tên `dense_parameters` để caller không phải đổi."""
        return list(self.parameters())
