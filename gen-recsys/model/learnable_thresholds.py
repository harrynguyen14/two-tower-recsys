"""τ_u, τ_i, τ_c — ngưỡng độ trưởng thành trong tanh(N/τ), HỌC ĐƯỢC (nn.Parameter) nhưng
KHỞI TẠO từ thống kê thật của dataset (xem idea.md "Việc CẦN LÀM TIẾP" #1).

[SỬA 2026-09-13] Đổi tên phương thức cho tường minh (xem result.md "CHECKLIST CUỐI CÙNG"):
    mat_u        -> user_weight
    mat_i        -> item_weight
    conf_content -> category_confidence

Khởi tạo (đo lại cho KuaiRand-Pure, xem result.md 2026-09-13 — GIÁ TRỊ CŨ đo trên
KuaiRand-27K không áp dụng được cho Pure, N_u/N_i có phân phối khác hẳn):
    τ_u = 35.0     — median N_u tại thời điểm sample (đúng phương pháp idea.md mục 0)
    τ_i = 599.3    — mean N_i tại thời điểm sample (median N_i=238 vẫn dùng mean theo
                     đúng lý do gốc: tránh tanh bão hòa sớm ở vùng N nhỏ quan trọng nhất)
    τ_c = 34.0     — mean N_category tại thời điểm sample, ĐO LẠI 2026-09-16 (xem dưới)

[SỬA 2026-09-16] τ_c: 53486.1 -> 34.0. Giá trị cũ SAI THANG ĐO, không phải sai số nhỏ.
Nguồn gốc: 2026-09-13 đo τ_c bằng `groupby(tag).cumcount()` = số TƯƠNG TÁC của tag, trong
khi runtime (`build_n_cumulative.py::build_category_n_cumulative`) định nghĩa N_category =
số VIDEO RIÊNG BIỆT cùng tag có first_seen_ms < t. Hai đại lượng khác nhau hàng trăm lần.
Ghi chú cũ "τ_c = cùng phương pháp với τ_i (cùng bản chất đếm lũy kế)" chính là chỗ nhầm.

Đo lại trên 905,334 token thật của tập train, qua ĐÚNG hàm lookup mà train.py gọi:
    min=0  p25=1  median=9  p75=39  p90=98  p99=306  max=805  mean=34.0
τ_c=53486 lớn hơn GIÁ TRỊ LỚN NHẤT CÓ THỂ (805) 66 lần -> không mẫu nào đưa tanh ra khỏi
vùng phẳng -> c = tanh(N_cat/τ_c) kẹt ở ~0.0006 trên TOÀN dataset.

Hậu quả đo được (bảng chẩn đoán gate, `train.py::_report_gate_diagnostic`):
    ‖e_content‖ = 18.57  ->  ‖e_content · c‖ = 0.0238     (co ~780 lần)
tức nhánh content trong `item_embedding.py` bị vô hiệu hoá gần như hoàn toàn — đúng cơ chế
lẽ ra phải cứu item cold. Và τ_c đứng yên tuyệt đối (53486.10 -> 53486.10) qua cả 6 lần
chạy ablation vì gradient của tanh ở x≈0.0006 gần bằng 0.

Chọn MEAN (34.0) chứ không MEDIAN (9.0), cùng lý do đã dùng cho τ_i: median quá nhỏ làm
tanh bão hoà sớm ở vùng N nhỏ — vùng quan trọng nhất cho cold-start. Với τ_c=34:
c p50=0.2590, std=0.3860, trải đều [0,1]; với τ_c=9: c p50=0.7616, 50% mẫu đã vượt 0.76.

CẢNH BÁO rủi ro "threshold-collapse" (xem idea.md/review.md): gradient descent có thể kéo
τ về giá trị tối ưu loss TRUNG BÌNH (bị chi phối bởi nhóm warm, đông hơn), vô tình "bỏ rơi"
nhóm cold ít ảnh hưởng loss tổng — BẮT BUỘC giám sát riêng: vẽ đường giá trị τ qua epoch +
so Recall/NDCG tách riêng theo nhóm cold (xem get_tau_snapshot()). Cơ chế chủ động chống
threshold-collapse (class-balanced loss, anchor loss) đã CÂN NHẮC nhưng HOÃN — chờ đo
baseline này có thật sự cần không trước khi thêm (xem result.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

TAU_U_INIT = 35.0      # median N_u tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13
TAU_I_INIT = 599.3     # mean N_i tại thời điểm sample, đo trên KuaiRand-Pure 2026-09-13
TAU_C_INIT = 34.0      # mean N_category tại thời điểm sample, ĐO LẠI 2026-09-16 (xem docstring)


class LearnableThresholds(nn.Module):
    """Bọc τ_u/τ_i/τ_c thành nn.Parameter, cung cấp user_weight/item_weight/
    category_confidence tiện dụng và snapshot giá trị để giám sát threshold-collapse qua
    epoch."""

    def __init__(self, tau_u_init: float = TAU_U_INIT, tau_i_init: float = TAU_I_INIT, tau_c_init: float = TAU_C_INIT):
        super().__init__()
        self.tau_u = nn.Parameter(torch.tensor(tau_u_init))
        self.tau_i = nn.Parameter(torch.tensor(tau_i_init))
        self.tau_c = nn.Parameter(torch.tensor(tau_c_init))

    def _safe_tau(self, tau: torch.Tensor) -> torch.Tensor:
        # τ có thể bị gradient kéo về <=0 (vô nghĩa, N/τ đổi dấu) — clamp mềm giữ dương.
        return tau.clamp(min=1e-3)

    def user_weight(self, n_u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_u / self._safe_tau(self.tau_u))

    def log_user_maturity(self, n_u: torch.Tensor) -> torch.Tensor:
        """log1p(N_u) - log1p(tau_u) — thay cho log(user_weight) trong attention bias.

        [SUA 2026-09-22] VI SAO KHONG DUNG log(tanh(N_u/tau_u)). tanh bao hoa ve 1 theo cap
        so nhan, va log(1) = 0 — nen so hang gamma*log(u_i)*log(m_j) TAT HAN voi user nhieu
        lich su. Do that tren du lieu val (2,490 chuoi, tau_u=35.3):

            N_u       |prod|    |log_u|    median rank (C=101)
            0-20      1.5009     3.7876     36
            21-50     0.7427     1.5592     40
            51-100    0.4121     0.8290     44
            101-300   0.2383     0.4186     45
            301+      0.0015     0.0015     48   <- gan nhu doan mo

        So hang gamma yeu di 1,000 LAN tu nhom it lich su den nhom nhieu, va thu hang xau di
        DON DIEU theo dung nhip do. |log_m| thi KHONG giam (0.57 -> 1.09), nen su sup do den
        hoan toan tu phia log_u. Nhom warm_warm (138k/141k mau) nam gan nhu tron trong vung
        chet — day la thu quyet dinh gan het metric, khong phai o cold-start.

        Tang tau_u KHONG sua duoc: tanh luon co tiem can ngang tai 1 nen moi tau chi DOI tran
        chu khong bo duoc no. Phai doi DANG HAM.

        log1p(N_u) - log1p(tau_u) = log((1+N_u)/(1+tau_u)): giu nguyen y nghia "N_u so voi
        nguong tau_u", khong bao hoa, va tau_u van hoc duoc (kha vi). Them mot cai loi: gia
        tri gio co CA HAI DAU (am khi N_u < tau_u, duong khi lon hon), trong khi
        log(tanh(.)) luon am — model phan biet duoc "duoi nguong" va "tren nguong" thay vi
        chi "gan hay xa nguong".

            N_u     cu: log(tanh(N/35.3))    moi: log1p(N)-log1p(35.3)
            1              -3.56                    -2.89
            35             -0.28                    -0.02
            100            -0.0069                  +1.03
            300            -0.0000                  +2.13
            1000            0.0000                  +3.33

        Vung N_u < 35 gan nhu khong doi (nhom cold_user, dang cho ket qua tot nhat, it bi
        anh huong); vung N_u > 100 tu cho CHET thanh co tin hieu that va tang don dieu."""
        return torch.log1p(n_u) - torch.log1p(self._safe_tau(self.tau_u))

    def item_weight(self, n_i: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_i / self._safe_tau(self.tau_i))

    def category_confidence(self, n_category: torch.Tensor) -> torch.Tensor:
        return torch.tanh(n_category / self._safe_tau(self.tau_c))

    def get_tau_snapshot(self) -> dict[str, float]:
        """Giá trị τ hiện tại — gọi mỗi epoch để log/vẽ đường giá trị, phát hiện sớm
        threshold-collapse (xem cảnh báo ở docstring module)."""
        return {
            "tau_u": self.tau_u.item(),
            "tau_i": self.tau_i.item(),
            "tau_c": self.tau_c.item(),
        }
