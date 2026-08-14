"""
Item Tower — xem Readme.md mục "Item tower — kiến trúc GMU (Gated Multimodal Unit)".

Giai đoạn 1 (nhúng text và ảnh bìa QUA CÙNG 1 CLIP model nhưng xuất RIÊNG 2 vector, không
average/concat thô) chạy ở model/preprocess_data/data.py, OFFLINE trước khi train.

File này nhận 2 vector đã nhúng sẵn (text_emb, image_emb) + cờ has_image, rồi TỰ HỌC cách
kết hợp 2 modal qua GMU gate — per-sample, không phải trọng số cố định cho toàn dataset.
Lý do: đo thử cho thấy text_embedding phân biệt content tốt nhưng image_embedding nhiễu
không đồng đều giữa các item (bìa sách phản ánh phong cách thiết kế hơn nội dung) — GMU
cho phép model tự hạ tỷ trọng ảnh xuống gần 0 với item mà ảnh không hữu ích, thay vì áp đặt
1 công thức trộn cứng (xem Readme).

Modality dropout (random zero-out image trong lúc train) giúp gate học phân biệt thật thay
vì luôn dựa vào cả 2 modal, và giúp model robust khi item cold-start thiếu ảnh chất lượng.

Item cold-start (item hoàn toàn mới) dùng được ngay: chỉ cần text_emb/image_emb của item đó
(chạy qua CLIP 1 lần khi item xuất hiện), không cần bất kỳ thống kê interaction nào.
"""

import json

import numpy as np
import torch
import torch.nn as nn


class GatedMultimodalUnit(nn.Module):
    """GMU (Arevalo et al., 2017) — gate học được, per-sample, quyết định tỷ trọng mỗi
    modal thay vì trộn cố định (vd average 50/50).

    h_text = tanh(Linear_t(text_emb))
    h_img  = tanh(Linear_i(image_emb))
    z      = sigmoid(Linear_gate([text_emb, image_emb]))   # per-sample, per-dim
    fused  = z * h_text + (1 - z) * h_img
    """

    def __init__(self, text_dim, image_dim, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.gate = nn.Linear(text_dim + image_dim, hidden_dim)

    def forward(self, text_emb, image_emb):
        h_text = torch.tanh(self.text_proj(text_emb))
        h_img = torch.tanh(self.image_proj(image_emb))
        z = torch.sigmoid(self.gate(torch.cat([text_emb, image_emb], dim=-1)))
        return z * h_text + (1 - z) * h_img  # [batch, hidden_dim]


class ItemTower(nn.Module):
    """(text_emb, image_emb, has_image) -> GMU -> MLP -> item_vector.

    modality_dropout_p: xác suất zero-out image_emb mỗi sample TRONG LÚC TRAIN (chỉ áp
    dụng khi self.training=True và has_image=True — không "dropout" item vốn đã thiếu
    ảnh, vì image_emb của chúng đã là vector 0 sẵn)."""

    def __init__(self, text_dim, image_dim, out_dim=128, gmu_hidden_dim=256,
                 mlp_hidden_dim=256, dropout=0.1, modality_dropout_p=0.2):
        super().__init__()
        self.gmu = GatedMultimodalUnit(text_dim, image_dim, gmu_hidden_dim)
        self.modality_dropout_p = modality_dropout_p
        self.net = nn.Sequential(
            nn.Linear(gmu_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_dim),
        )

    def forward(self, text_emb, image_emb, has_image=None):
        """
        text_emb, image_emb : [batch, dim]
        has_image            : [batch] bool hoặc None (None = giả định mọi item đều có ảnh
                                thật, dùng khi gọi rời rạc ngoài training loop)
        """
        if self.training and has_image is not None:
            drop_mask = (torch.rand(image_emb.size(0), device=image_emb.device) < self.modality_dropout_p)
            drop_mask = drop_mask & has_image  # chỉ drop item vốn CÓ ảnh thật
            image_emb = image_emb.clone()
            image_emb[drop_mask] = 0.0

        fused = self.gmu(text_emb, image_emb)
        return self.net(fused)  # [batch, out_dim]


class ItemEmbeddingStore:
    """Load text_embeddings.npy + image_embeddings.npy + has_image.npy (Giai đoạn 1,
    xem model/preprocess_data/data.py) + mapping asin -> row index.

    device: khi truyền (vd "cuda"), giữ luôn text/image_embeddings TRÊN GPU thay vì CPU
    — .get() trả về gather trực tiếp trên GPU, loại bỏ hẳn round-trip CPU->GPU transfer
    mỗi batch cho pos/neg item embeddings (tương tự cách NVTabular giữ toàn bộ tensor trên
    GPU xuyên suốt training). Với model nhẹ, transfer PCIe mỗi batch chiếm tỷ trọng lớn
    hơn hẳn compute — đây là chỗ đáng tối ưu nhất.

    RÀNG BUỘC: chỉ dùng được với num_workers=0 — CUDA tensor không thể chia sẻ an toàn
    qua DataLoader worker process khác (mỗi worker có CUDA context riêng, không kế thừa
    được tensor GPU từ main process qua fork/pickle). main.py tự enforce điều này."""

    def __init__(self, npy_dir, asin_to_idx=None, device=None):
        npy_dir = str(npy_dir)
        self.device = device
        self.text_embeddings = torch.from_numpy(np.load(f"{npy_dir}/text_embeddings.npy")).float().to(device)
        self.image_embeddings = torch.from_numpy(np.load(f"{npy_dir}/image_embeddings.npy")).float().to(device)
        self.has_image = torch.from_numpy(np.load(f"{npy_dir}/has_image.npy")).to(device)

        if asin_to_idx is None:
            with open(f"{npy_dir}/asin_to_idx.json", encoding="utf-8") as f:
                asin_to_idx = json.load(f)
        self.asin_to_idx = asin_to_idx  # dict[str, int]

    @property
    def text_dim(self):
        return self.text_embeddings.shape[1]

    @property
    def image_dim(self):
        return self.image_embeddings.shape[1]

    def get(self, asins):
        idx = torch.tensor([self.asin_to_idx[a] for a in asins], dtype=torch.long, device=self.device)
        return self.text_embeddings[idx], self.image_embeddings[idx], self.has_image[idx]
