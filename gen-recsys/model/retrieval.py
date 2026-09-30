"""Retrieval logit + sampled-softmax loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    """Sampled-softmax retrieval. MULTI-VECTOR kieu POLY-ENCODER khi num_user_vectors > 1.

    Vi sao (formula.md §-1): delta_h(x_q) tach HAI che do nhin (gan/xa) o attention, nhung
    `hidden[:,-1,:]` nen ca hai vao MOT diem trong R^d. Nhom `both` (48.8% test) ket o 0.0491.

    CACH LAM — theo dong MULTI-INTEREST (MIND/ComiRec-SA/REMI), KHONG phai ColBERT:
        A     = softmax_j(c_m . h_j / sqrt d)   c_m = code hoc duoc, A = routing (B,M,L)
        e_u^m = sum_j A[m,j] * h_j              moi vector pool tu vung khac nhau cua chuoi
        score = max_m (e_u^m . e_i / sqrt d)    hard readout (argmax), chuan cua ComiRec

    LICH SU THAT BAI (doc truoc khi thu lai — xem memory gen-recsys-muc-tieu-va-multivector):
    - lan 8: chieu tuyen tinh M vector tu MOT hidden[:,-1,:] => tat ca la ham cua cung mot
      vector 64 chieu, khong them thong tin. cosine 0.9822, HR@10 tut 21%.
    - lan 9: Poly-encoder + orthogonal + code_scale=8 + softmax-gop. Init cosine 0.3814
      (tach tot) nhung 200 step sau ve 0.9812.

    CHONG COLLAPSE la viec cua RR/IHN, KHONG phai cua toan tu gop hay cua init. Literature
    (REMI RecSys 2023, MIMA 2026) da chi ro argmax routing tu no VAN collapse. Co HAI loai:
      - routing collapse: mot interest chi gom 1 item (attention sparse) -> RR chua, do bang
        phuong sai routing.
      - interest collapse: M vector trung nhau -> do bang `user_vector_cosine`.
    Phai theo doi CA HAI.
    """

    def __init__(self, dim: int, t_base: float = 0.1, num_user_vectors: int = 1):
        super().__init__()
        self.dim = dim
        self.t_base = t_base
        self.num_user_vectors = num_user_vectors
        if num_user_vectors > 1:
            # Code init ORTHOGONAL: moi code bat dau o mot huong khac han => attention cua
            # chung pool tu cac vung khac nhau cua chuoi ngay tu step 0.
            self.codes = nn.Parameter(torch.empty(num_user_vectors, dim))
            nn.init.orthogonal_(self.codes)
            # `code_scale`: GIU, nhung ly do da doi (2026-09-30). Hai loai collapse keo
            # NGUOC nhau, do duoc tren hidden ngau nhien (B=64,L=50,M=4):
            #   scale   cosine   entropy   RR_pen
            #     1.0   0.9732    0.998   9.6e-11   <- vector TRUNG nhau (interest collapse)
            #     4.0   0.6882    0.969   3.1e-08
            #     8.0   0.3441    0.882   1.1e-06   <- tach tot, RR con RAT nho
            #    16.0   0.1190    0.633   4.1e-05
            #   (moi code bam DUNG 1 item: RR = 9.1e-4, gap ~800 lan muc scale=8)
            # => cosine thap DOI routing sac, con RR PHAT routing sac. Nhung thang do cho
            # thay chung KHONG loai tru nhau: co vung o giua (scale ~8) du sac de tach
            # vector ma RR van gan khong. Vai tro cua RR la canh GIOI HAN TREN — chan
            # routing truot den muc moi code chi lay 1 item.
            self.code_scale = nn.Parameter(torch.tensor(8.0))

    def user_vectors(
        self, hidden: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B,L,d) -> (e_u (B,M,d), routing (B,M,L)). M=1 tra ve hidden[:,-1,:], routing None.

        `routing` la ma tran A cua REMI — can tra ra ngoai de tinh routing_regularization().
        """
        if self.num_user_vectors == 1:
            return hidden[:, -1, :].unsqueeze(1), None
        att = torch.einsum("md,bld->bml", self.codes, hidden) / math.sqrt(self.dim)
        att = att * self.code_scale
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(att, dim=-1)                       # (B,M,L) = A
        return torch.einsum("bml,bld->bmd", w, hidden), w

    def routing_regularization(self, routing: torch.Tensor) -> torch.Tensor:
        """RR cua REMI (arXiv 2302.14532 eq.13) — PHAT phuong sai routing de bo sparsity.

            C     = (A - mean_L(A))^T (A - mean_L(A))
            L_reg = || diag(C) ||_F^2

        Chua routing collapse: mot interest chi gom 1 item thay vi MOI item lien quan.
        Trong Table 4 cua REMI, RR (+28.5~47.1% HR@50) manh hon HAN regularization trong
        khong gian bieu dien (ho orthogonality penalty, chi +1.1~7.3%).

        LUU Y: day la loai collapse KHAC voi cai `user_vector_cosine` do (M vector trung
        nhau). Phai theo doi CA HAI chi so — chua co bang chung RR mot minh ha duoc cosine.
        """
        a = routing - routing.mean(dim=2, keepdim=True)       # tru trung binh theo CHUOI
        c = torch.bmm(a, a.transpose(1, 2)) / self.dim        # (B,M,M), theo code goc
        dr = torch.diagonal(c, dim1=-2, dim2=-1)              # (B,M) phuong sai tung hang
        return (dr.norm(dim=1) ** 2).mean()                   # mean theo batch (goc: sum)

    def _score(self, e_u: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """e_u (B,M,d) hoac (B,d) -> diem (B,C). M=1 la dot-product thuan."""
        if e_u.dim() == 2:
            e_u = e_u.unsqueeze(1)
        sim = torch.einsum("bmd,bcd->bmc", e_u, cand) / math.sqrt(self.dim)
        if sim.shape[1] == 1:
            return sim.squeeze(1)
        # ARGMAX (hard readout) — CHUAN cua MIND/ComiRec/REMI: chon vector khop nhat voi
        # positive. KHONG phai vi "max sinh phan hoa" (lap luan do DA BI BAC 2026-09-30:
        # REMI noi argmax routing "inherently collapses"; MIMA chi ro mot interest thang
        # lap lai se HUT het gradient cac intent khac roi troi ve huong trung binh).
        # Dung argmax chi vi day la duong da duoc kiem chung cua dong multi-interest, va
        # de RR (phat phuong sai routing) ap len dung kien truc ma no duoc thiet ke cho.
        # Chong collapse la viec cua RR + IHN, KHONG phai cua toan tu gop.
        # Xem memory gen-recsys-multi-interest-literature.
        return sim.amax(dim=1)

    def user_vector_cosine(self, e_u: torch.Tensor) -> float:
        """CHI SO THOAI HOA: cosine trung binh giua cac cap vector user.
        -> 1 nghia la M vector da hoi tu ve mot => co che vo nghia (bat duoc o lan chay 8)."""
        if self.num_user_vectors == 1 or e_u.dim() == 2:
            return float("nan")
        with torch.no_grad():
            eu = F.normalize(e_u, dim=-1)
            g = torch.einsum("bmd,bnd->bmn", eu, eu)
            off = ~torch.eye(self.num_user_vectors, dtype=torch.bool, device=g.device)
            return float(g[:, off].mean())

    def scaled_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        """logit(u,i)/T_base − log_q — dùng CHUNG cho cross_entropy lúc train (forward) VÀ"""
        return self._score(e_u_final, candidate_embeddings) / self.t_base - log_q

    def eval_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Logit XEP HANG luc eval — KHONG tru log_q. Cung ham cham diem nhu train."""
        return self._score(e_u_final, candidate_embeddings)

    def forward(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
        positive_idx: torch.Tensor,
    ) -> torch.Tensor:
        scaled_logit = self.scaled_logit(e_u_final, candidate_embeddings, log_q)
        return F.cross_entropy(scaled_logit, positive_idx)

    def forward_sequence(
        self,
        pred: torch.Tensor,
        positive_e_i: torch.Tensor,
        neg_e_i: torch.Tensor,
        positive_log_q: torch.Tensor,
        neg_log_q: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[THÊM 2026-09-14] Loss TỰ HỒI QUY TOÀN CHUỖI — dự đoán item kế tiếp tại MỌI vị"""
        pos_logit = (pred * positive_e_i).sum(-1) / math.sqrt(self.dim)
        pos_logit = pos_logit / self.t_base - positive_log_q

        neg_logit = torch.einsum("bkd,bcd->bkc", pred, neg_e_i) / math.sqrt(self.dim)
        neg_logit = neg_logit / self.t_base - neg_log_q.unsqueeze(1)

        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)
        target = torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).view_as(valid_mask)
        valid = valid_mask.float()
        return (loss * valid).sum() / valid.sum().clamp(min=1)
