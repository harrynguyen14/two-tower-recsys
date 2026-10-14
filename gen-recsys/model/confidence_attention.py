"""Attention SIGMOID + pairwise bias, dùng trong SequenceDecoder. Xem formula.md §4.

KHÔNG phải softmax attention cộng bias. Hai tầng cùng phục vụ câu hỏi ở formula.md §−1:
  - kích hoạt sigmoid (thay softmax)  -> ngắn/dài hạn không còn tranh ngân sách
  - delta ĐỘNG theo query             -> mỗi bước tự chọn phạm vi nhìn
[XOÁ 2026-10-06] Cờ `use_softmax` / `static_delta` / `use_beta` ĐÃ BỎ — bảng 2x3 ở
formula.md §4.4 đã chạy xong nên các nhánh đối chứng hết việc. Chỉ còn `use_qk` (§4 ghi
cấu hình tốt nhất là BỎ q.k, nên cờ này còn phải chạy được cả hai chiều).

[XOÁ 2026-10-05] Số hạng `mu_h * PPMI(i,j)` ĐÃ BỎ HẲN. Hai lý do:

1. **Model không học nó.** Trong checkpoint đã train đủ (`all-ckp-result.md`), `mu` ở cấu
   hình đề xuất = **0.521** — kẹt ở init 0.5. Cấu hình softmax thì mu chủ động giảm về
   0.14. Dưới sigmoid, các bias bớt quan trọng và model dựa vào q.k nhiều hơn.
2. **Không scale.** Bảng dense N^2 fp16: Pure 115 MB, nhưng không scale lên catalog lớn hơn.
   Đã đo cả 3 phương án thay thế, loại cả 3 (formula.md §9b.1): sparse top-128 giữ 16% khối
   lượng mà tốn 24.6 GB; PMI cấp category-2 có R^2 = 0.025 với PMI item.

CẢNH BÁO khi đọc lại: `--no-pmi` CHƯA ai chạy, nên không có phép đo TRỰC TIẾP về việc bỏ
PMI mất gì. Bằng chứng là mu kẹt ở init — mạnh nhưng GIÁN TIẾP. Căn cứ thêm PMI ban đầu
(analyze.py:m6_pmi, giữ làm hồ sơ): PMI một mình THUA pop (median rank .284 vs .334) nhưng
pop+PMI = .557 (+66%) — hai tín hiệu trực giao. Muốn khôi phục thì đọc hai chỗ đó.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-6


NUM_TS_BUCKETS = 48
MS_PER_SECOND = 1000.0
TS_BUCKET_DIVISOR = 0.1505

# [SUA 2026-10-13] delta_proj(x_q) CHAN DO LON (KHONG chan dau) — xem formula.md §4.1ter.
# Ly do: delta_h(x_q) = delta_h + w_h^T x_q PHAI giu kha nang doi dau (am=nhin gan,
# duong=nhin xa — test_delta_shifts_attention_by_distance), nen KHONG the dung Softplus
# (CABLE) hay sigmoid (FoX) — ca hai XOA dau. Nhung ||w_h|| KHONG co gi chan tang vo han
# qua gradient descent (x_q da qua LayerNorm nen bi chan, w_h thi khong). Da do thuc te
# (formula.md §4.1): |delta| bi "ep bu" len 4.8x (dat 0.0851) khi thieu delta dong — dau
# hieu dung huong phinh, du CHUA toi muc nghiem trong. Sua: bound BANG tanh (giu dau,
# chan bien do), KHONG sigmoid/softplus. DELTA_PROJ_LIMIT=1.0 chon de vung da do (0.0851)
# gan tuyen tinh (giu lai 99.8%, hau nhu khong doi hanh vi hien tai), chi bop khi vuot
# xa (vd tho=5.0 -> con 1.0) — phong ngua BUNG NO trong tuong lai (train lau hon, scale
# lon hon), khong phai sua loi dang xay ra bay gio.
DELTA_PROJ_LIMIT = 1.0

# [SUA 2026-10-13] beta*log(m_p) CHUYEN khoi logit sang "value-confidence gate" — xem
# formula.md §4.1bis. Ly do: log(m_p) co range toi ~-9 (item hiem tren Pure, N_i min),
# RONG HON HAN range cua delta*log(dist) (~0-0.6) hay W_ts/W_age moi bucket (~0.2-0.3) SAU
# TRAIN — khi cong chung 1 logit truoc sigmoid, mot minh beta*log(m_p) co the AP CHE ca 3
# bias con lai cho item hiem, bat ke query dang hoi gi. Sua: KHONG con cong vao logit; thay
# vao do r_p = m_p**beta_h = exp(beta_h*log(m_p)) nhan truc tiep vao attn_weight (SAU
# sigmoid(logit), TRUOC khi nhan v) — dung vai tro "do tin cay cua value" (item hiem ->
# embedding it duoc hoc -> gate thap khi beta>0) thay vi "do lien quan giua query-key" (logit).
# Dung power-law (khong dung sigmoid(..+const)) vi beta=0 cho r_p = m_p**0 = 1.0 CHINH XAC
# (tuyet doi, khong xap xi) voi MOI m_p — giu dung pattern zero-init/ablation sach (diff
# tuyet doi = 0.000e+00) ma delta/W_ts/age da dung, thay vi sigmoid(0+C) chi xap xi 1.
# Clamp max=1 chi can khi beta hoc AM (m_p**beta > 1 voi m_p<1) — khong anh huong tai init.
# beta_h TAI DUNG tham so da co (khong them tham so hoc moi).

# [THEM 2026-10-09] age_t chuyen tu action token (action_encoder.py) sang day — xem
# AgeAwareBias ben duoi va formula.md §2bis. Bucket LOG tren thang NGAY, do truc tiep tren
# Pure: age min=0 max=29 (dataset chi thu thap 29.5 ngay, item ra mat gan nhu cung luc dau
# ky). Divisor chon de 29 nam o bin cuoi ma van giu phan giai vung day (median=3):
# log1p(29)/0.38 ~ 9 bin — SO VOI 49 bin cu (hieu chinh theo dataset da xoa, max=1421 ngay,
# qua thua cho thang 0-29 that cua Pure).
NUM_AGE_BUCKETS = 10
AGE_BUCKET_DIVISOR = 0.38


def age_bucket(hist_age_days: torch.Tensor) -> torch.Tensor:
    """(B, L) tuổi item theo NGÀY (>=0) -> (B, L) chỉ số bucket log, long.

    `.detach()`: chỉ số bucket không có gradient — gradient chỉ chảy vào bảng tra.
    age âm (item thiếu upload_dt, đánh dấu -1) về bin 0.
    """
    a = hist_age_days.detach().clamp(min=0).float()
    bucket = torch.log1p(a) / AGE_BUCKET_DIVISOR
    return bucket.clamp(min=0, max=NUM_AGE_BUCKETS - 1).long()


def compute_ts_bucket(hist_timestamps: torch.Tensor) -> torch.Tensor:
    """(B, L) epoch-ms -> (B, L, L) uint8, chỉ số bucket của |τ_i − τ_j|."""
    B, L = hist_timestamps.shape
    t = hist_timestamps - hist_timestamps.min()
    delta = (t.view(B, L, 1) - t.view(B, 1, L)).abs()
    bucket = torch.log10(delta.float() / MS_PER_SECOND + 1.0) / TS_BUCKET_DIVISOR
    return bucket.clamp(min=0, max=NUM_TS_BUCKETS).to(torch.uint8)


class AddTsBias(torch.autograd.Function):
    """logit += ts_w[h, bucket[b,i,j]] — CỘNG TẠI CHỖ, không materialize bias."""

    @staticmethod
    def forward(ctx, logit: torch.Tensor, ts_w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.ts_shape = ts_w.shape
        idx_flat = idx.reshape(-1).long()
        with torch.no_grad():
            for h in range(logit.shape[1]):
                logit[:, h].add_(ts_w[h].index_select(0, idx_flat).view(idx.shape).to(logit.dtype))
        ctx.mark_dirty(logit)
        return logit

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (idx,) = ctx.saved_tensors
        H, NB = ctx.ts_shape
        grad_w = grad_out.new_zeros(H, NB, dtype=torch.float32)
        flat_idx = idx.reshape(-1).long()
        for h in range(H):
            grad_w[h].index_add_(0, flat_idx, grad_out[:, h].reshape(-1).float())
        return grad_out, grad_w, None


class AgeAwareBias(nn.Module):
    """logit[b,h,q,j] += W_age[h, age_bucket[b,j]] + gamma_proj_h(x_q) * W_age[h, bucket[b,j]]
    — xem formula.md §2bis.

    Cung pattern voi delta_h(x_q) (ca hai deu bias PAIRWISE dieu bien theo QUERY, khong
    phai token tinh): W_age la bang tra THEO KEY j (tuoi cua item dang duoc attend toi) —
    SO HANG TINH, moi user deu thay. gamma_proj(x_q) la PHAN DIEU BIEN THEM tu chinh token
    dang hoi (query) — neu user nay (bieu dien qua x_q, tuc la lich su gan nhat) co xu
    huong thich item moi, gamma se hoc am khi age nho; nguoc lai duong khi thich item cu.

    [SUA 2026-10-09 — BUG tu phat hien qua test] KHONG duoc viet thanh TICH
    `gamma_q(x_q) * W_age(age)` voi ca hai zero-init: d(logit)/d(W_age) = gamma_q, ma
    gamma_q=0 luc init => W_age khong bao gio nhan gradient (tich hai so zero luon cho
    gradient zero qua mot trong hai nhanh). Sua thanh TONG hai so hang: (1) W_age cong
    THANG vao logit (nhan gradient TRUC TIEP, khong qua gamma), (2) gamma_proj(x_q) nhan
    VOI W_age roi cong them — day la phan DIEU BIEN, zero-init rieng. Khoi dau gamma=0 nen
    so hang (2) = 0 nhung so hang (1) van dung W_age=0 (chua hoc) => logit age = 0 y het
    ban khong co age (ablation mien phi), va CA HAI tham so deu nhan gradient tu buoc dau.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.W_age = nn.Parameter(torch.zeros(num_heads, NUM_AGE_BUCKETS))
        self.gamma_proj = nn.Linear(dim, num_heads)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

    def forward(self, x: torch.Tensor, age_bucket_idx: torch.Tensor) -> torch.Tensor:
        """x: (B,L,dim) token (dung lam query). age_bucket_idx: (B,L) bucket tuoi tai KEY.

        Tra ve (B,H,L,L) de cong truc tiep vao logit — KHONG materialize (B,H,L,L,head_dim)
        nhu nhanh dynamic_ts, vi o day chi can 1 scalar/head moi (q,j), khong phai vector.
        """
        B, L, _ = x.shape
        # (B,L) -> gia tri W_age theo KEY j, cho tung head: (H,B,L) -> (B,H,1,L)
        age_val = self.W_age[:, age_bucket_idx]  # (H,B,L)
        age_val = age_val.permute(1, 0, 2).unsqueeze(2)  # (B,H,1,L)

        # so hang TINH: moi query thay CUNG gia tri W_age(age_j) — nhan gradient truc tiep.
        static_term = age_val.expand(B, self.num_heads, L, L)

        # so hang DIEU BIEN: (B,L,H) -> (B,H,L,1), nhan voi age_val (B,H,1,L) -> (B,H,L,L).
        gamma_q = self.gamma_proj(x).transpose(1, 2).unsqueeze(-1)  # (B,H,L,1)
        dynamic_term = gamma_q * age_val  # (B,H,L,1)*(B,H,1,L) -> (B,H,L,L)

        return static_term + dynamic_term


class ConfidenceModulatedAttention(nn.Module):
    """Sigmoid attention + pairwise bias (xem docstring module).

    [SUA 2026-10-13] beta*log(m_p) KHONG con trong logit — xem formula.md §4.1bis. No la
    gate r_p = m_p**beta_h nhan vao attn_weight SAU sigmoid(logit), dong vai "do tin cay
    cua value" (item hiem -> gate thap khi beta>0), khac voi delta/W_ts/W_age van o trong
    logit (dong vai "do lien quan giua query-key").
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 513,
        use_qk: bool = True, dynamic_ts: bool = False, static_delta: bool = False,
        use_age: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # ABLATION FuXi-beta (arXiv 2508.10615): ho bao BO q.k lai TOT HON tren MovieLens.
        # Tat q.k => attention chi con cac bias (rab tinh/dong + maturity + time).
        self.use_qk = use_qk

        self.use_age = use_age
        self.age_bias = AgeAwareBias(dim, num_heads) if use_age else None

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.beta = nn.Parameter(torch.zeros(num_heads))
        self.delta = nn.Parameter(torch.zeros(num_heads))
        # W_ts DONG (nang cap 2): bang tra doi tu scalar sang VECTOR, roi dieu bien
        # theo query. Tinh: ts_w[h, bucket] la mot so. Dong: ts_w[h, bucket] la vector
        # d chieu, chieu voi g_h(x_q) => "query nay nen nhin o THANG NAO" (phut? gio?
        # ngay?), khong chi "nhin gan hay xa" nhu delta_h(x_q). Xem formula.md §4.1.
        self.dynamic_ts = dynamic_ts
        if dynamic_ts:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1, self.head_dim))
            # g_h(x_q): init zero => khoi dau logit ts = 0 Y HET ts_w=0 cua ban tinh,
            # nen ban tinh van la nhom doi chung hop le (cung ly le nhu delta_proj).
            self.ts_gate = nn.Linear(dim, num_heads * self.head_dim)
            nn.init.zeros_(self.ts_gate.weight)
            nn.init.zeros_(self.ts_gate.bias)
        else:
            self.ts_w = nn.Parameter(torch.zeros(num_heads, NUM_TS_BUCKETS + 1))

        # delta ĐỘNG: δ_h(x_q) = δ_h + w_h·x_q. Init zero ⇒ khởi đầu Y HỆT bản tĩnh,
        # nên bản tĩnh vẫn là nhóm đối chứng hợp lệ (formula.md §4.1).
        self.delta_proj = nn.Linear(dim, num_heads)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # delta TINH (`--static-delta`): o A1/A3 cua bang 2x2 (formula.md §4.4 nhom A).
        # Cai lai 2026-10-08 — KHONG khoi phuc tu git (commit cu chua co 5 bug fix).
        # delta_proj da init ZERO, nen chi can dong bang: weight=0 + khong gradient
        # => delta_proj(x) luon tra 0 => delta_q = delta_h thuan = DUNG ban tinh.
        # Khong can nhanh if trong forward.
        self.static_delta = static_delta
        if static_delta:
            self.delta_proj.weight.requires_grad_(False)
            self.delta_proj.bias.requires_grad_(False)

        pos = torch.arange(max_seq_len)
        rel = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).float()
        self.register_buffer("log_rel_distance", torch.log1p(rel), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        log_m: torch.Tensor | None = None,
        ts_bucket: torch.Tensor | None = None,
        hist_age_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk:
            logit = q @ k.transpose(-2, -1)
            logit.div_(math.sqrt(self.head_dim))
        else:
            # khong co q.k: logit khoi dau = 0, moi tin hieu den TU BIAS.
            logit = torch.zeros(B, self.num_heads, L, L, dtype=q.dtype, device=q.device)

        if self.age_bias is not None and hist_age_days is not None:
            logit = logit + self.age_bias(x, age_bucket(hist_age_days))

        log_rel = self.log_rel_distance[:L, :L].view(1, 1, L, L)
        # δ_h(x_q): (B,H,L) broadcast với (L,L) — không materialize thêm (B,H,L,L).
        # [SUA 2026-10-13] phan dieu bien qua tanh*DELTA_PROJ_LIMIT (chan DO LON, giu DAU)
        # — xem DELTA_PROJ_LIMIT o tren va formula.md §4.1ter. tanh(0)=0 CHINH XAC nen
        # delta_proj init zero van cho delta_q = delta_h thuan (ablation sach, khong doi).
        delta_adj = DELTA_PROJ_LIMIT * torch.tanh(self.delta_proj(x) / DELTA_PROJ_LIMIT)
        delta_q = self.delta.view(1, -1, 1) + delta_adj.transpose(1, 2)
        logit.add_(delta_q.unsqueeze(-1) * log_rel)

        if ts_bucket is not None:
            if self.dynamic_ts:
                # (B,H,L,d): thang thoi gian ma query q muon doc
                g = self.ts_gate(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                idx = ts_bucket.reshape(-1).long()
                for h in range(self.num_heads):
                    # (B,L,L,d) cho rieng head h — ts_w[h] la (NB,d)
                    vec = self.ts_w[h].index_select(0, idx).view(B, L, L, self.head_dim)
                    logit[:, h] = logit[:, h] + torch.einsum(
                        "bqd,bqpd->bqp", g[:, h].to(vec.dtype), vec
                    ).to(logit.dtype)
            else:
                logit = AddTsBias.apply(logit, self.ts_w, ts_bucket)

        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        invalid = causal_mask.view(1, 1, L, L)
        if key_padding_mask is not None:
            invalid = invalid | key_padding_mask.view(B, 1, 1, L)

        # Sigmoid KHÔNG chuẩn hoá theo hàng ⇒ tổng attention tăng theo độ dài chuỗi
        # ("large initial attention norms"). Trừ log(n_q) để chặn — BẮT BUỘC, xem
        # formula.md §4.2. n_q = số token hợp lệ trong tầm nhìn causal của q.
        n_q = (~invalid).sum(dim=-1, keepdim=True).clamp(min=1).to(logit.dtype)
        attn_weight = torch.sigmoid(logit - torch.log(n_q))

        if log_m is not None:
            # r_p = exp(beta_h*log(m_p)) = m_p**beta_h — "do tin cay cua value", nhan SAU
            # sigmoid(logit), KHONG cong vao logit. beta=0 => r_p=1.0 CHINH XAC (ablation
            # sach). Xem formula.md §4.1bis.
            r_p = torch.exp(
                self.beta.view(1, -1, 1) * log_m.view(B, 1, L)
            ).clamp(max=1.0)  # (B,H,L) theo KEY p; clamp chi tac dung khi beta hoc am
            attn_weight = attn_weight * r_p.unsqueeze(2)  # broadcast (B,H,1,L) -> (B,H,L,L)

        attn_weight = attn_weight.masked_fill(invalid, 0.0)

        attn_weight = self.dropout(attn_weight)

        out = attn_weight @ v
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.out_proj(out)
