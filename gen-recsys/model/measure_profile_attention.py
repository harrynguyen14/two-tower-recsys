"""ĐO: token profile (vị trí 0) thực sự nhận bao nhiêu attention, và δ đè nó bao nhiêu.

Vì sao cần script này — xem result.md mục 2026-09-17 "Ô ③ KHÔNG CÓ CƠ CHẾ". Tóm tắt:
`sequence_model.py` gán m_profile=1 -> log m_0 = 0 -> CẢ β lẫn γ triệt tiêu tại cột j=0
(bias = (β + γ·log u_i)·0). Thứ duy nhất còn tác động lên token profile là δ_h·log(1+i),
mà δ chỉ biết KHOẢNG CÁCH, không biết user cold hay warm. Trên ckpt_tauc34.pt có 12/16
head học δ<0 — phần lớn head chủ động PHẠT token xa, và profile luôn là token xa nhất.

Ba câu hỏi script trả lời bằng attention weight THẬT, không phải suy luận từ công thức:

  [1] Attention về token 0 có KHÁC nhau giữa user cold vs warm không?
      Công thức dự đoán: KHÔNG khác (β/γ triệt tiêu -> chỉ còn δ, mà δ mù với user).
      Nếu đo ra khác -> dự đoán sai, phải xem lại. Nếu đo ra giống -> xác nhận ô ③ mù.

  [2] Nó suy giảm thế nào theo độ dài chuỗi?

  [3] Bao nhiêu phần của suy giảm đó là DO δ gây ra? Đo lại với δ=0, giữ nguyên mọi
      trọng số khác. Đây là phần quan trọng nhất: nó tách "softmax pha loãng tự nhiên khi
      chuỗi dài" (không phải lỗi của ai) khỏi "chính cơ chế của ta đè profile xuống" (là
      lỗi thiết kế, sửa được). Hai nguyên nhân đó cần biện pháp khác hẳn nhau.

CÁCH ĐỌC KẾT QUẢ — share thô một mình KHÔNG kết luận được gì:
Attention sink (arXiv 2309.17453) chỉ ra token đầu chuỗi thường hút nhiều attention mà
vô nghĩa ngữ nghĩa — softmax buộc tổng = 1 nên model đổ phần thừa vào đó. Vì vậy script
in CẢ share thô LẪN tỉ lệ so với mức chia đều 1/(L_i). Chỉ con số thứ hai mới trả lời
"có được ưu tiên hay không". Cùng lý do, đừng đọc "share lớn" thành "profile đang có tác
dụng" — ô ③ vẫn có thể chết trong khi share cao.

Chạy:  python measure_profile_attention.py --ckpt C:/Users/minhn/Downloads/ckpt_tauc34.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _peek_output_dir() -> str | None:
    for i, a in enumerate(sys.argv):
        if a == "--output-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--output-dir="):
            return a.split("=", 1)[1]
    return None


# PHẢI set env + sys.path TRƯỚC khi import build_n_cumulative: module đó đọc
# GEN_RECSYS_OUT_DIR ở CẤP MODULE, set sau khi import là vô tác dụng. Đây đúng là lỗi đã
# làm chết cả 5 nhánh ablation ở step 0 trên Kaggle — xem train.py dòng 41-63.
_out = _peek_output_dir() or "../preprocess_data/output"
os.environ.setdefault("GEN_RECSYS_OUT_DIR", str(Path(_out).resolve()))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess_data"))

from build_n_cumulative import build_n_cache  # noqa: E402
from confidence_attention import compute_prod
from dataset import GenRecsysDataset, MAX_SEQ_LEN
from item_embedding import ItemEmbedding, ItemEmbeddingConfig
from learnable_thresholds import LearnableThresholds
from negative_sampler import NegativeSampler
from ranking_loss import RankingLoss
from retrieval import RetrievalLoss
from sequence_model import SequenceModel
from train import build_category_counts, load_checkpoint, run_batch_forward
from user_embedding import UserProfileConfig, UserProfileEmbedding


def build_modules(output_dir: Path, cfg: dict, device: torch.device) -> dict:
    """Dựng module ĐÚNG như train.py. Lệch một tham số là state_dict lệch shape -> nạp lỗi,
    hoặc tệ hơn: nạp được nhưng chạy cơ chế khác (đúng lỗi mà load_checkpoint chặn)."""
    ds = GenRecsysDataset(output_dir, split="train")
    num_items = len(ds.item_static)
    item_embed = ItemEmbedding(ItemEmbeddingConfig(
        num_items=num_items,
        num_authors=int(ds.item_static["author_idx"].max()) + 1,
        num_music=int(ds.item_static["music_idx"].max()) + 1,
        num_categories=build_category_counts(output_dir / "item_static.npy"),
        dim=cfg["dim"], 
    )).to(device)
    seq_model = SequenceModel(
        dim=cfg["dim"], num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
        max_seq_len=(2 * MAX_SEQ_LEN if cfg["interleave"] else MAX_SEQ_LEN) + 1,
        interleave=cfg["interleave"], use_beta=cfg["use_beta"], use_gamma=cfg["use_gamma"],
    ).to(device)
    profile_embed = UserProfileEmbedding(UserProfileConfig(
        onehot_num_categories=ds.onehot_num_categories, dim=cfg["dim"],
        use_profile_token=cfg["use_profile_token"],
    )).to(device)
    return {
        "dataset": ds,
        "item_embed": item_embed,
        "seq_model": seq_model,
        "profile_embed": profile_embed,
        "thresholds": LearnableThresholds().to(device),
        "retrieval_loss_fn": RetrievalLoss(dim=cfg["dim"], t_base=0.1).to(device),
        "ranking_loss_fn": RankingLoss(dim=cfg["dim"]).to(device),
        "neg_sampler": NegativeSampler(output_dir, num_items=num_items),
    }


class AttnProbe:
    """Chộp attention weight bằng forward hook.

    `ConfidenceModulatedAttention.forward()` chỉ trả output đã qua out_proj, không trả
    attn_weight — không sửa module thì không lấy được. Dùng hook thay vì sửa forward để
    đường train không mang thêm rủi ro nào: đo xong gỡ sạch handle.

    Hook tính LẠI attn_weight theo đúng các bước của forward() (cùng thứ tự cộng bias,
    cùng mask). Tốn thêm một lần q@k nhưng đảm bảo đo đúng cái model dùng. Nếu forward()
    đổi công thức mà quên sửa chỗ này, số đo sẽ sai IM LẶNG — nên self_check() dưới cùng
    đối chiếu output tái dựng với output thật của module.
    """

    def __init__(self, seq_model: SequenceModel):
        self.seq_model = seq_model
        self.captured: list[torch.Tensor] = []  # mỗi phần tử (B, H, L, L), đã .cpu()
        self._handles: list = []

    @staticmethod
    def _attn_weight(module, x, kpm, log_u, log_m, prod) -> torch.Tensor:
        B, L, _ = x.shape
        H, dh = module.num_heads, module.head_dim
        q = module.q_proj(x).view(B, L, H, dh).transpose(1, 2)
        k = module.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        logit = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if log_u is not None and log_m is not None:
            if module.use_beta:
                logit = logit + module.beta.view(1, -1, 1, 1) * log_m.view(B, 1, 1, L)
            if module.use_gamma:
                p = prod if prod is not None else compute_prod(log_u, log_m)
                logit = logit + module.gamma.view(1, -1, 1, 1) * p
        logit = logit + module.delta.view(1, -1, 1, 1) * \
            module.log_rel_distance[:L, :L].view(1, 1, L, L)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        logit = logit.masked_fill(causal.view(1, 1, L, L), float("-inf"))
        if kpm is not None:
            logit = logit.masked_fill(kpm.view(B, 1, 1, L), float("-inf"))
        return torch.nan_to_num(torch.softmax(logit, dim=-1), nan=0.0)

    def _hook(self, module, args, kwargs, output):
        x = args[0] if args else kwargs["x"]
        w = self._attn_weight(module, x, kwargs.get("key_padding_mask"),
                              kwargs.get("log_u"), kwargs.get("log_m"), kwargs.get("prod"))
        self.captured.append(w.detach().cpu())

    def __enter__(self):
        for layer in self.seq_model.decoder.layers:
            self._handles.append(layer.attn.register_forward_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _bucket(n_hist: int) -> int:
    """Gom độ dài chuỗi thành bucket để mỗi ô đủ mẫu."""
    return 4 if n_hist < 8 else 16 if n_hist < 32 else 64 if n_hist < 128 else 256


def measure(ckpt_path: Path, output_dir: Path, *, max_batches: int, batch_size: int,
            device: torch.device, zero_delta: bool, item_n_cache: dict,
            category_n_cache: dict) -> dict:
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    mods = build_modules(output_dir, cfg, device)
    load_checkpoint(ckpt_path, modules={
        k: mods[k] for k in ("item_embed", "seq_model", "profile_embed", "thresholds",
                             "retrieval_loss_fn", "ranking_loss_fn")})

    if zero_delta:
        # ĐỐI CHỨNG: tắt δ, giữ NGUYÊN mọi trọng số khác. Chênh lệch so với lượt δ-thật
        # chính là phần suy giảm DO δ, tách khỏi pha loãng softmax tự nhiên.
        with torch.no_grad():
            for layer in mods["seq_model"].decoder.layers:
                layer.attn.delta.zero_()

    ds = GenRecsysDataset(output_dir, split="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for name in ("item_embed", "seq_model", "profile_embed"):
        mods[name].eval()

    acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    n_user = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            with AttnProbe(mods["seq_model"]) as probe:
                fwd = run_batch_forward(
                    batch, ds, mods["item_embed"], mods["seq_model"], mods["profile_embed"],
                    mods["thresholds"], mods["neg_sampler"], item_n_cache, category_n_cache,
                    100, device, static_user_weight=cfg["static_user_weight"])
            is_cold = fwd["is_user_cold"].cpu()
            n_hist = batch["hist_valid_mask"].sum(1)  # (B,) số lượt thật
            # Trung bình trên 4 layer: mỗi layer cho 1 tensor (B,H,L,L) trong probe.captured
            share_layers = [w[:, :, :, 0].mean(1) for w in probe.captured]  # mỗi cái (B, L)
            share = torch.stack(share_layers).mean(0)  # (B, L) — trung bình layer & head
            for b in range(share.shape[0]):
                nh = int(n_hist[b])
                if nh < 2:
                    continue
                # vị trí truy vấn CUỐI của user này: chuỗi [profile, Φ_0, a_0, ...] nên
                # lượt cuối nằm ở 2*nh-1 (token action) — lấy min để không vượt L.
                i_last = min(2 * nh - 1, share.shape[1] - 1)
                grp = "cold" if bool(is_cold[b]) else "warm"
                acc[(grp, _bucket(nh))].append(float(share[b, i_last]))
                acc[("all", _bucket(nh))].append(float(share[b, i_last]))
                acc[(grp + "_L", _bucket(nh))].append(float(i_last + 1))  # L thật, để tính chia đều
                acc[("all_L", _bucket(nh))].append(float(i_last + 1))
            n_user += int(n_hist.shape[0])
    return {"acc": acc, "n": n_user}


def report(res_on: dict, res_off: dict) -> None:
    acc_on, acc_off = res_on["acc"], res_off["acc"]
    buckets = sorted({b for (g, b) in acc_on if not g.endswith("_L")})
    print(f"\n{'=' * 84}")
    print("ATTENTION VỀ TOKEN PROFILE (cột j=0), đo tại vị trí truy vấn CUỐI của mỗi user")
    print(f"trung bình trên 4 layer × 4 head | n user = {res_on['n']}")
    print(f"{'=' * 84}")
    print(f"{'bucket':>8} {'nhóm':>6} {'n':>6} {'share δ thật':>14} {'share δ=0':>12} "
          f"{'chia đều':>10} {'tỉ lệ/đều':>10} {'δ làm giảm':>11}")
    print("-" * 84)
    for b in buckets:
        for grp in ("cold", "warm", "all"):
            v_on = acc_on.get((grp, b), [])
            if not v_on:
                continue
            v_off = acc_off.get((grp, b), [])
            L = acc_on.get((grp + "_L", b), [1.0])
            m_on = sum(v_on) / len(v_on)
            m_off = (sum(v_off) / len(v_off)) if v_off else float("nan")
            unif = 1.0 / (sum(L) / len(L))
            drop = (1 - m_on / m_off) * 100 if v_off and m_off > 0 else float("nan")
            print(f"{b:>8} {grp:>6} {len(v_on):>6} {m_on:>14.5f} {m_off:>12.5f} "
                  f"{unif:>10.5f} {m_on / unif:>10.3f} {drop:>10.1f}%")
    print("-" * 84)
    print("tỉ lệ/đều < 1 -> profile được ưu tiên THẤP HƠN mức chia đều (bị bỏ quên)")
    print("cold ≈ warm   -> cơ chế MÙ với user cold — đúng dự đoán β/γ triệt tiêu tại j=0")
    print("δ làm giảm >0 -> phần suy giảm do CHÍNH δ, không phải softmax pha loãng tự nhiên")
    print("\nLưu ý: share thô cao KHÔNG có nghĩa profile đang cấp thông tin — xem attention")
    print("sink (arXiv 2309.17453). Chỉ cột 'tỉ lệ/đều' mới nói lên ưu tiên.")


def self_check() -> None:
    """Chặn lỗi IM LẶNG của AttnProbe: nếu forward() đổi công thức mà hook không đổi theo,
    số đo sai mà không có dấu hiệu. Đối chiếu output tái dựng từ attn_weight của hook với
    output THẬT của module — phải trùng khít."""
    from confidence_attention import ConfidenceModulatedAttention
    torch.manual_seed(0)
    B, L, dim, H = 2, 9, 16, 2
    attn = ConfidenceModulatedAttention(dim=dim, num_heads=H, max_seq_len=16).eval()
    with torch.no_grad():
        attn.beta.copy_(torch.tensor([0.3, -0.2]))
        attn.gamma.copy_(torch.tensor([0.1, 0.4]))
        attn.delta.copy_(torch.tensor([-0.5, 0.2]))
    x = torch.randn(B, L, dim)
    log_u = torch.log(torch.rand(B, L) * 0.8 + 0.1)
    log_m = torch.log(torch.rand(B, L) * 0.8 + 0.1)

    with torch.no_grad():
        out_that = attn(x, None, log_u=log_u, log_m=log_m)
        w = AttnProbe._attn_weight(attn, x, None, log_u, log_m, None)
        v = attn.v_proj(x).view(B, L, H, dim // H).transpose(1, 2)
        out_dung = attn.out_proj((w @ v).transpose(1, 2).reshape(B, L, dim))
    lech = (out_that - out_dung).abs().max().item()
    assert lech < 1e-5, f"hook tái dựng SAI: lệch {lech:.3e} — attn_weight đo được không khớp forward()"
    print(f"  [self-check] OK — attn_weight của hook khớp forward() (lệch {lech:.2e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output-dir", default="../preprocess_data/output")
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    self_check()
    out_dir = Path(args.output_dir)
    dev = torch.device(args.device)
    item_n_cache = build_n_cache("item_N")
    category_n_cache = build_n_cache("category_N")

    print("[1/2] đo với δ ĐÃ HỌC ...")
    r_on = measure(Path(args.ckpt), out_dir, max_batches=args.max_batches,
                   batch_size=args.batch_size, device=dev, zero_delta=False,
                   item_n_cache=item_n_cache, category_n_cache=category_n_cache)
    print("[2/2] đo lại với δ=0 (đối chứng) ...")
    r_off = measure(Path(args.ckpt), out_dir, max_batches=args.max_batches,
                    batch_size=args.batch_size, device=dev, zero_delta=True,
                    item_n_cache=item_n_cache, category_n_cache=category_n_cache)
    report(r_on, r_off)
