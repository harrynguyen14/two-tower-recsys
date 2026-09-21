"""CHẨN ĐOÁN [2026-09-21]: e_collab có học được gì trong những step ĐẦU không?

Bối cảnh: 6 checkpoint qua 6 cấu hình đều cho cosine-std của collab_embedding = 0.125,
đúng bằng vector ngẫu nhiên độc lập ở dim=64 — tức bảng collab KHÔNG học được gì sau cả
epoch. Câu hỏi còn lại: w_collab (gate) tụt xuống ~0.12 từ lúc nào?

  - Nếu tụt trong vài trăm step đầu -> vòng lặp tự sát: collab chưa kịp học thì gate đã
    bóp gradient của nó xuống, và nó vĩnh viễn không học được. Train thêm epoch vô ích.
  - Nếu tụt từ từ -> gate chỉ đang phản ánh việc collab vô dụng vì lý do khác.

Script KHÔNG sửa train.py: monkey-patch ItemEmbedding.forward để ghi lại w_collab mỗi
step, và chụp collab_embedding.weight tại các mốc để đo dịch chuyển thật.
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F

import train as T
from item_embedding import ItemEmbedding


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="../preprocess_data/output")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args()

    traj: list[tuple[int, float, float]] = []   # (step, w_collab, w_content)
    snaps: dict[int, torch.Tensor] = {}
    state = {"step": 0, "emb": None}

    orig_forward = ItemEmbedding.forward

    def patched(self, *a, **kw):
        out = orig_forward(self, *a, **kw)
        st = self._last_stats
        if state["emb"] is None:
            state["emb"] = self.collab_embedding
            snaps[0] = self.collab_embedding.weight.detach().clone()
        # mỗi step gọi forward nhiều lần (hist/label/neg) — chỉ ghi lần đầu của step
        if len(traj) == 0 or traj[-1][0] != state["step"]:
            wc = st["g_i"].mean().item()
            wct = st["norm_content_shrunk"].mean().item() / max(st["norm_content"].mean().item(), 1e-9)
            traj.append((state["step"], wc, wct))
        return out

    ItemEmbedding.forward = patched

    # đếm step qua optimizer.step
    orig_opt_step = torch.optim.Adam.step

    def patched_opt(self, *a, **kw):
        r = orig_opt_step(self, *a, **kw)
        state["step"] += 1
        s = state["step"]
        if state["emb"] is not None and s % args.every == 0:
            snaps[s] = state["emb"].weight.detach().clone()
        return r

    torch.optim.Adam.step = patched_opt

    T.train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=1,
        max_steps_per_epoch=args.steps,
        num_workers=0,
        save_path=None,
        save_every=10**9,   # không ghi đè ckpt thật
    )

    print("\n=== QUỸ ĐẠO w_collab (gate) ===")
    print(f"{'step':>6}{'w_collab':>11}{'w_content/|c|':>15}")
    for s, wc, wct in traj:
        if s % args.every == 0 or s < 5:
            print(f"{s:>6}{wc:>11.4f}{wct:>15.4f}")

    print("\n=== DỊCH CHUYỂN e_collab so voi step 0 ===")
    W0 = snaps[0]
    W0n = F.normalize(W0, dim=-1)
    print(f"{'step':>6}{'|dW| mean':>12}{'cos(W,W0)':>12}{'W.std':>9}{'cos.std':>10}")
    for s in sorted(snaps):
        W = snaps[s]
        d = (W - W0).norm(dim=-1).mean().item()
        c = (F.normalize(W, dim=-1) * W0n).sum(-1).mean().item()
        Wn = F.normalize(W, dim=-1)
        g = torch.Generator().manual_seed(0)
        i = torch.randint(0, W.shape[0], (30000,), generator=g)
        j = torch.randint(0, W.shape[0], (30000,), generator=g)
        m = i != j
        cs = (Wn[i[m]] * Wn[j[m]]).sum(-1).std().item()
        print(f"{s:>6}{d:>12.4f}{c:>12.4f}{W.std():>9.4f}{cs:>10.4f}")
    print(f"\n  cos.std cho vector ngau nhien doc lap (dim={W0.shape[1]}): "
          f"{1/ W0.shape[1] ** 0.5:.4f}")


if __name__ == "__main__":
    main()
