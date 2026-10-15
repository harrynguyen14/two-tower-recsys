"""Retrieval logit + sampled-softmax loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    """Sampled-softmax retrieval, MOT vector user = hidden[:,-1,:]. Xem formula.md §5.

    [XOA 2026-10-06] MULTI-VECTOR (num_user_vectors > 1) DA XOA HAN cung codes/code_scale/
    routing_regularization/user_vector_cosine. formula.md §5 ghi DA BAC: ca tien de ("nhom
    `both` ket vi output nen hai che do" — artefact phan bo n_u) lan lap luan ("max sinh
    phan hoa" — REMI/MIMA bao argmax routing VAN collapse) deu sai, va thu 2 cach thi cosine
    giua cac vector deu -> 0.98, HR tut 21%. Mac dinh M=1 nen nhanh nay chua bao gio chay.
    Muon thu lai thi DOC memory `gen-recsys-multi-interest-literature` truoc, dung dung lai
    code cu.
    """

    def __init__(
        self, dim: int, t_base: float = 0.07, learnable_temp: bool = True,
    ):
        super().__init__()
        self.dim = dim
        # [XOA 2026-10-15] CCR (Zhou et al. AAAI'24) da bo: chua tune, mac dinh tat, va bat
        # len la DOI ham muc tieu => moi o ablation khong con so duoc. Xem formula.md §5bis.

        # [SUA 2026-10-09] L2-NORMALIZE + NHIET DO HOC DUOC. Sua bug model khong hoc:
        # do duoc |e_user|=6.78 nhung |e_pos|=473.57 (lech 70x) => |logit| ~ 4013 =>
        # cross-entropy SATURATE, gradient ~ 0 => hr@10 0.0018 (chance 0.0013).
        #
        # NGUYEN NHAN CAU TRUC (khong phai nhiet do): `pred` = output seq_model nen bi
        # `final_norm` LayerNorm chan o |pred| ~ sqrt(d) = 8; con `target_e_i` la item
        # embedding THUAN + `.detach()` (fix #1) nen KHONG qua norm nao va KHONG nhan
        # gradient tu nhanh target. Truoc fix #1 duong target CO gradient (|grad|=22.8,
        # formula.md) va chinh no dieu hoa norm. Fix #1 dung ve nguyen ly (chan shortcut
        # collapse kieu SimSiam/BYOL) nhung GO LUON co che dieu hoa norm ma khong thay
        # bang gi => norm item troi tu do len 473.
        #
        # Dot THUAN khong chan duoc: nhiet do co dinh chi nhan logit len 1/t_base, con do
        # lon THAT do norm quyet dinh. L2-normalize ca hai phia => dot thanh COSINE trong
        # [-1,1] => logit bi chan trong +-1/temp, KHONG THE saturate du embedding hoc the nao.
        #
        # Nhiet do hoc duoc (cach CLIP): mot vo huong, hoc trong LOG-SPACE cho on dinh,
        # co chan tren. 0.8 cu KHONG dung cho thang cosine (cosine/0.8 => logit toi da 1.25,
        # qua phang de phan biet 7,583 item). Default 0.07 ~ CLIP init; model tu tim.
        self.learnable_temp = learnable_temp
        # [SUA 2026-10-14 — code review] self.t_base truoc day CHI duoc gan o nhanh
        # learnable_temp=False — goi .t_base truc tiep (bo qua property .temp) tren
        # instance mac dinh (learnable_temp=True) se AttributeError, de nham vi t_base
        # cung la ten constructor argument. Luu LUON, ca 2 nhanh — o nhanh learnable,
        # day la gia tri INIT (khong doi theo train), khong phai nhiet do hieu dung
        # (dung .temp cho cai do).
        self.t_base = t_base
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(float(math.log(t_base))))

    # Chan duoi 0.01 (= scale 100, muc CLIP hoi tu ve) de khong sinh logit qua sac.
    MIN_TEMP = 0.01
    # [THEM 2026-10-13] Chan TREN — error.md #11. CLIP goc clip SCALE (1/temp) o 100 de
    # chong logit qua sac (=> MIN_TEMP=0.01 da co). Nhung KHONG co chan cho chieu NGUOC:
    # temp qua LON => scale=1/temp qua NHO => cosine/temp gan 0 voi MOI candidate =>
    # phan phoi softmax gan UNIFORM => gradient TOAN BATCH suy yeu (doi xung voi loi #6
    # da sua o chieu nguoc: logit qua SAC => CE saturate). MAX_TEMP=100.0 doi xung voi
    # MIN_TEMP=0.01 qua log-space (scale trong [0.01, 100]).
    MAX_TEMP = 100.0

    @property
    def temp(self) -> torch.Tensor | float:
        """Nhiet do hieu dung. MOT dinh nghia duy nhat, dung cho CA train VA eval."""
        if self.learnable_temp:
            return self.log_temp.exp().clamp(min=self.MIN_TEMP, max=self.MAX_TEMP)
        return self.t_base

    def _score(self, e_u: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """e_u (B,d) -> diem (B,C): COSINE (L2-normalize ca hai phia), KHONG chia sqrt(d).

        [SUA 2026-10-07] Bo `/sqrt(dim)` o day. Truoc day `_score` chia sqrt(d) ROI
        `scaled_logit`/`forward_sequence` chia tiep `t_base` => nhiet do hieu dung THAT la
        `sqrt(d) * t_base`, KHONG phai `t_base`. Do duoc voi d=64, t_base=0.1: logit =
        dot/0.80 chu khong phai dot/0.1 — lech 8x. Te hon: no DOI THEO dim (d=128 => 1.131),
        nen moi lan doi `dim` la vo tinh doi ca nhiet do. Gio chi con MOT cho dinh nghia
        nhiet do (`t_base`), mac dinh 0.8 = sqrt(64)*0.1 giu DUNG scale cu => checkpoint va
        moi so da do VAN so sanh duoc. Doi `dim` tu gio khong con keo theo nhiet do.
        """
        # [SUA 2026-10-09] L2-normalize CA HAI phia => cosine, khong con dot thuan.
        e_u = F.normalize(e_u, dim=-1)
        cand = F.normalize(cand, dim=-1)
        return torch.einsum("bd,bcd->bc", e_u, cand)

    def scaled_logit(
        self,
        e_u_final: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        """logit(u,i)/T_base − log_q — dùng CHUNG cho cross_entropy lúc train (forward) VÀ"""
        return self._score(e_u_final, candidate_embeddings) / self.temp - log_q

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
        # [SUA 2026-10-07] bo `/sqrt(dim)` — xem `_score`. Nhiet do chi con o `t_base`.
        # [SUA 2026-10-09] normalize truoc khi cham diem — CUNG thang voi `_score`.
        pred = F.normalize(pred, dim=-1)
        positive_e_i = F.normalize(positive_e_i, dim=-1)
        neg_e_i = F.normalize(neg_e_i, dim=-1)
        temp = self.temp

        pos_logit = (pred * positive_e_i).sum(-1)
        pos_logit = pos_logit / temp - positive_log_q

        neg_logit = torch.einsum("bkd,bcd->bkc", pred, neg_e_i)
        neg_logit = neg_logit / temp - neg_log_q.unsqueeze(1)

        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)
        target = torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device)

        flat_logits = logits.reshape(-1, logits.shape[-1])
        loss = F.cross_entropy(
            flat_logits, target.reshape(-1), reduction="none"
        ).view_as(valid_mask)
        valid = valid_mask.float()
        return (loss * valid).sum() / valid.sum().clamp(min=1)
