"""Chạy bộ ablation ĐẦY ĐỦ cho paper — dành cho GPU (T4/A100), KHÔNG chạy trên CPU.

VÌ SAO CẦN FILE NÀY thay vì gõ tay 5 lệnh: mỗi nhánh phải chạy CÙNG số step, CÙNG batch,
CÙNG dữ liệu; lệch 1 tham số là kết quả không so sánh được và không ai phát hiện ra. File
này ép điều đó bằng code, và gom log về 1 chỗ để đọc.

CÁCH DÙNG (Colab/Kaggle T4):
    python run_ablation.py --output-dir ../preprocess_data/output --steps 3000
    python run_ablation.py --only 4,5              # chỉ 2 nhánh quyết định, chạy trước
    python run_ablation.py --steps 500 --dry-run   # in lệnh, không chạy

Đo thật trên T4 (fp16, batch=256): 125.3 ms/step -> 3000 step ~ 6 phút/nhánh.
Trên CPU: ~1.9 s/step ở batch=32 -> 3000 step ~ 95 phút/nhánh, KHÔNG nên.

HAI SỐ QUYẾT ĐỊNH CẢ NGHIÊN CỨU:
  1. ‖∇γ‖ (`gg=` trong log). λ cũ chết vì gradient ~1.1e-17 (log(m)≈0 trên dữ liệu warm).
     Nếu ∇γ cũng ở bậc 1e-15 thì γ chết y hệt -> DỪNG, không xây tiếp. Cần bậc >= 1e-4.
  2. #4 vs #5 (γ per-position vs γ với u TĨNH). Cả hai đều có số hạng tích; chỉ khác u biến
     thiên hay hằng. Nếu #5 ngang #4 -> luận điểm "cold-start là đại lượng per-position"
     KHÔNG có cơ sở thực nghiệm và phải rút khỏi paper. Đây là thí nghiệm DUY NHẤT chứng
     minh đóng góp 2.

Ô TRỌNG TÂM khi đọc bảng eval: **cold_user_warm_item ở bảng FEW-SHOT** (n=21,212 val).
KHÔNG phải cold_cold (n=81 — không đủ mẫu, xem idea.md mục 4.6.5b).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Mỗi nhánh: (id, tên, cờ thêm, câu hỏi nó trả lời)
# Thứ tự CỘNG DỒN: mỗi nhánh thêm ĐÚNG 1 cơ chế so với nhánh trước -> quy kết được đóng góp.
ABLATIONS = [
    (1, "hstu_thuan", ["--no-profile-token", "--no-beta", "--no-gamma"],
     "baseline: HSTU xen kẽ, không profile token, không count-bias"),
    (2, "profile_token", ["--no-beta", "--no-gamma"],
     "e_profile prepend có đáng giá không"),
    (3, "beta", ["--no-gamma"],
     "beta*log(m_j): bias phía item một mình đủ chưa"),
    (4, "gamma_full", [],
     "gamma*log(u_i)*log(m_j): số hạng TÍCH có đóng góp RIÊNG không  <-- QUYẾT ĐỊNH"),
    (5, "gamma_static_u", ["--static-user-weight"],
     "gamma nhưng u TĨNH per-user: per-position có phải mấu chốt không  <-- QUYẾT ĐỊNH"),
]


def build_cmd(flags: list[str], args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "train.py",
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--num-epochs", "1",
        "--max-steps-per-epoch", str(args.steps),
    ]
    if args.no_cuckoo:
        cmd.append("--no-cuckoo-embedding")
    return cmd + flags


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", default="../preprocess_data/output")
    p.add_argument("--steps", type=int, default=3000, help="step mỗi nhánh (T4: 3000 ~ 6 phút)")
    p.add_argument("--batch-size", type=int, default=256, help="256 cho T4 16GB (đo thật: peak 1.73GB)")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--only", default="", help="chỉ chạy các nhánh này, vd '4,5'")
    p.add_argument("--with-cuckoo", dest="no_cuckoo", action="store_false", default=True,
                   help="bật CuckooEmbedding (mặc định TẮT — khuyến nghị cho catalog nhỏ như Pure)")
    p.add_argument("--dry-run", action="store_true", help="in lệnh rồi thoát, không chạy")
    args = p.parse_args()

    selected = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    todo = [a for a in ABLATIONS if selected is None or a[0] in selected]
    if not todo:
        sys.exit(f"--only={args.only} không khớp nhánh nào (hợp lệ: 1-5)")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    # UTF-8 BẮT BUỘC: stdout Windows mặc định cp1252, gặp tiếng Việt trong log là
    # UnicodeEncodeError và cả run chết giữa chừng (đã xảy ra thật 2026-09-15).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    print(f"Chạy {len(todo)} nhánh × {args.steps} step, batch={args.batch_size}, log -> {log_dir}/\n")
    for aid, name, flags, question in todo:
        print(f"  [{aid}] {name:16s} {' '.join(flags) or '(mặc định)':45s} {question}")
    print()

    if args.dry_run:
        for _, _, flags, _ in todo:
            print(" ".join(build_cmd(flags, args)))
        return

    results = []
    for aid, name, flags, question in todo:
        log_path = log_dir / f"abl{aid}_{name}.log"
        cmd = build_cmd(flags, args)
        print(f"=== [{aid}] {name} -> {log_path}")
        print(f"    {question}")

        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# ablation {aid}: {name}\n# {question}\n# cmd: {' '.join(cmd)}\n\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        elapsed = time.perf_counter() - start

        print(f"    {'OK' if rc == 0 else f'LỖI (rc={rc})'} — {elapsed / 60:.1f} phút\n")
        results.append((aid, name, rc, elapsed, log_path))

    print("\n=== TỔNG KẾT ===")
    for aid, name, rc, elapsed, log_path in results:
        print(f"  [{aid}] {name:16s} {'OK ' if rc == 0 else 'LỖI'} {elapsed / 60:6.1f} phút  {log_path}")

    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f"\n{len(failed)} nhánh LỖI — đọc log trước khi so sánh kết quả.")
        sys.exit(1)

    parse_logs(log_dir, [r[0] for r in results])


def parse_logs(log_dir: Path, ids: list[int]) -> None:
    """Rút 2 con số quyết định từ log: ‖∇γ‖ cuối cùng và recall@5 ở ô trọng tâm."""
    print("\n=== ‖∇γ‖ (dòng log cuối) — λ cũ chết ở 1.1e-17, cần thấy >= 1e-4 ===")
    for aid in ids:
        for log_path in sorted(log_dir.glob(f"abl{aid}_*.log")):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            grads = re.findall(r"gg=([\d.e+-]+)", text)
            vals = re.findall(r"\|g\|=([\d.e+-]+)", text)
            g = f"grad={grads[-1]}" if grads else "grad=?"
            v = f"|gamma|={vals[-1]}" if vals else "|gamma|=?"
            print(f"  [{aid}] {log_path.stem:24s} {g:20s} {v}")

    print("\n=== recall@5 ô TRỌNG TÂM (cold_user_warm_item, bảng FEW-SHOT, n≈21k) ===")
    for aid in ids:
        for log_path in sorted(log_dir.glob(f"abl{aid}_*.log")):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            # Bảng few-shot là bảng THỨ HAI trong report -> lấy match CUỐI CÙNG
            hits = re.findall(r"\[cold_user_warm_item\] n=(\d+) recall@5=([\d.]+)\+-([\d.]+)", text)
            if hits:
                n, r5, ci = hits[-1]
                print(f"  [{aid}] {log_path.stem:24s} n={n:>6s} recall@5={r5} +-{ci}")
            else:
                print(f"  [{aid}] {log_path.stem:24s} (không đọc được — xem log)")

    print("\nĐỌC KẾT QUẢ:")
    print("  - ∇γ ở bậc 1e-15  -> γ chết như λ cũ, DỪNG nghiên cứu ở đây.")
    print("  - #4 ngang #5      -> luận điểm per-position không có cơ sở, rút khỏi paper.")
    print("  - #4 > #5 rõ rệt   -> đóng góp 2 (per-position) được chứng minh.")
    print("  - #4 > #3 rõ rệt   -> đóng góp 1 (số hạng tích) được chứng minh.")
    print("  CI không chồng nhau mới gọi là 'rõ rệt'.")


if __name__ == "__main__":
    main()
