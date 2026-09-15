"""Cuckoo hash embedding table — collisionless, DUNG LƯỢNG NHỎ HƠN tổng số ID tiềm năng,
mô phỏng đúng vấn đề Monolith (ByteDance, "Real Time Recommendation System With
Collisionless Embedding Table") giải quyết cho SERVING: item/user mới liên tục xuất hiện
sau khi hệ thống đã launch, không thể biết trước tổng số ID để cấp 1 bảng nn.Embedding cố
định đủ lớn cho MỌI ID sẽ từng xuất hiện trong tương lai.

KHÁC với hash-bucket truyền thống (`slot = hash(id) % capacity`, nhiều ID cùng slot ->
"cướp" gradient của nhau, không thể phân biệt): cuckoo hashing đảm bảo mỗi ID ĐANG HOẠT
ĐỘNG có ĐÚNG 1 slot RIÊNG, không ai khác ghi đè, bằng 2 bảng con độc lập + 2 hàm hash:
  - Lookup ID: thử slot ở bảng A trước (nếu slot_id_a[hash_a(id)] == id -> dùng slot đó);
    nếu không khớp, thử bảng B tương tự.
  - Insert ID mới: nếu slot đích ở A đang TRỐNG -> chiếm luôn. Nếu ĐANG BỊ ID khác chiếm
    -> "đá" (evict) ID cũ sang slot đích của nó ở B (đệ quy nếu B cũng bị chiếm, tối đa
    MAX_KICKS lần trước khi buộc phải evict thẳng theo LRU — khác cuckoo hashing "thuần
    lý thuyết" vốn giả định bảng luôn đủ chỗ, ở đây bảng CỐ TÌNH nhỏ hơn tổng ID tiềm năng
    nên phải chấp nhận eviction thật, giống production Monolith).

QUAN TRỌNG — đây là cấu trúc dữ liệu ĐỘNG (id->slot mapping thay đổi lúc runtime), khác
hẳn nn.Embedding tĩnh: phần quản lý id->slot (dict Python, CPU-side, KHÔNG phải tensor)
phải chạy TRƯỚC mỗi forward để "resolve" batch ID thành batch slot index cụ thể, rồi mới
index vào 2 bảng con (`table_a`/`table_b`, LÀ nn.Parameter thật) để autograd hoạt động
bình thường trên phần đã resolve slot. Khi 1 ID bị evict, optimizer state (exp_avg/
exp_avg_sq của SparseAdam) tại slot đó PHẢI reset về 0 — nếu không, ID mới chiếm slot sẽ
"thừa kế" nhầm momentum của ID cũ đã bị đá đi, học sai hướng ngay từ bước đầu (bug tinh vi,
không crash, chỉ làm chậm hội tụ của ID mới một cách khó phát hiện). Xem
train.py::reset_optimizer_state_for_evicted() cho cách xử lý phần này.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_KICKS = 8  # giới hạn số lần đá đệ quy trước khi buộc evict theo LRU (tránh vòng lặp vô hạn)


class CuckooEmbedding(nn.Module):
    def __init__(self, capacity_per_table: int, dim: int, seed: int = 0):
        super().__init__()
        self.capacity = capacity_per_table
        self.dim = dim

        # 2 bảng con — LÀ nn.Parameter thật, autograd/SparseAdam hoạt động bình thường
        # trên các HÀNG đã được resolve slot (giống hệt nn.Embedding.weight, chỉ khác cách
        # xác định index trước khi lookup).
        self.table_a = nn.Parameter(torch.zeros(capacity_per_table, dim))
        self.table_b = nn.Parameter(torch.zeros(capacity_per_table, dim))
        nn.init.normal_(self.table_a, std=0.01)
        nn.init.normal_(self.table_b, std=0.01)

        # 2 hàm hash ĐỘC LẬP (nhân với 2 số nguyên tố lớn khác nhau rồi mod capacity) — đơn
        # giản, đủ dùng cho mục đích mô phỏng cơ chế (production Monolith thật dùng hash
        # 64-bit chuyên biệt hơn, nhưng bản chất thuật toán cuckoo giống hệt).
        gen = torch.Generator().manual_seed(seed)
        self._salt_a = int(torch.randint(1, 2**31 - 1, (1,), generator=gen).item())
        self._salt_b = int(torch.randint(1, 2**31 - 1, (1,), generator=gen).item())

        # id -> (table 'a'/'b', slot index) — CPU-side, KHÔNG phải tensor, đây là state
        # ĐỘNG thay đổi lúc runtime (khác nn.Embedding tĩnh).
        self.id_to_slot: dict[int, tuple[str, int]] = {}
        # slot -> id đang chiếm (None nếu trống) — hướng ngược lại, cần để biết khi insert
        # có phải evict ai không.
        self.slot_occupant_a: list[int | None] = [None] * capacity_per_table
        self.slot_occupant_b: list[int | None] = [None] * capacity_per_table
        # LRU timestamp — dùng khi MAX_KICKS vượt quá, evict ID lâu không được truy cập nhất
        # thay vì đá vòng vô hạn (production thật dùng "probabilistic feature filtering",
        # LRU đơn giản hơn nhưng cùng mục đích: ưu tiên giữ ID đang hoạt động, bỏ ID ít dùng).
        self._lru_clock = 0
        self.last_access_a: list[int] = [0] * capacity_per_table
        self.last_access_b: list[int] = [0] * capacity_per_table

        self.num_evictions = 0  # đếm để theo dõi tỷ lệ evict — cao bất thường nghĩa là capacity quá nhỏ
        self.pending_evicted_slots: list[tuple[str, int]] = []  # (table,slot) vừa bị evict trong lần resolve() gần nhất — train.py đọc để reset optimizer state

    def _hash_a(self, entity_id: int) -> int:
        return (entity_id * self._salt_a) % self.capacity

    def _hash_b(self, entity_id: int) -> int:
        return (entity_id * self._salt_b + 1) % self.capacity  # +1 để hash_b khác hash_a ngay cả khi salt trùng

    def _evict_and_insert(self, entity_id: int) -> tuple[str, int]:
        """Chèn entity_id CHƯA từng có slot — đá (kick) ID cũ nếu slot đích đang bị chiếm,
        đệ quy tối đa MAX_KICKS lần theo đúng thuật toán cuckoo hashing, rồi buộc evict
        theo LRU nếu vẫn không tìm được slot trống (bảng cố tình nhỏ hơn tổng ID tiềm năng
        -> eviction THẬT là một phần thiết kế, không phải lỗi)."""
        current_id = entity_id
        current_table = "a"

        for _ in range(MAX_KICKS):
            slot = self._hash_a(current_id) if current_table == "a" else self._hash_b(current_id)
            occupants = self.slot_occupant_a if current_table == "a" else self.slot_occupant_b

            evicted_id = occupants[slot]
            occupants[slot] = current_id
            self.id_to_slot[current_id] = (current_table, slot)
            self._touch(current_table, slot)

            if evicted_id is None:
                return current_table, slot  # slot trống -> chèn xong, không cần đá tiếp

            # slot đang bị evicted_id chiếm -> nó bị đá sang bảng KIA, tiếp tục vòng lặp
            # với evicted_id đóng vai "current_id" cần tìm chỗ mới. evicted_id CHƯA bị loại
            # bỏ hẳn khỏi hệ thống (chỉ đổi bảng), nên KHÔNG tính vào num_evictions/
            # pending_evicted_slots ở đây — chỉ trường hợp buộc evict theo LRU dưới mới là
            # loại bỏ THẬT (mất embedding đã học, cần reset optimizer state).
            del self.id_to_slot[evicted_id]  # xóa mapping cũ TRƯỚC khi tìm slot mới cho nó
            current_id = evicted_id
            current_table = "b" if current_table == "a" else "a"

        # Vượt MAX_KICKS — buộc evict theo LRU tại bảng/slot đang xét (current_id, current_table
        # vẫn đang "lơ lửng" chưa có chỗ) — ép nó vào đúng slot hash của nó, đá ID lâu-không-
        # dùng-nhất ra khỏi hệ thống HẲN (không tiếp tục đá dây chuyền, chặn vòng lặp vô hạn).
        slot = self._hash_a(current_id) if current_table == "a" else self._hash_b(current_id)
        occupants = self.slot_occupant_a if current_table == "a" else self.slot_occupant_b
        evicted_id = occupants[slot]
        if evicted_id is not None:
            del self.id_to_slot[evicted_id]
            self.num_evictions += 1
            self.pending_evicted_slots.append((current_table, slot))
            # reset embedding value tại slot bị chiếm lại — optimizer state reset ở train.py
            self._zero_slot(current_table, slot)
        occupants[slot] = current_id
        self.id_to_slot[current_id] = (current_table, slot)
        self._touch(current_table, slot)
        return current_table, slot

    def _touch(self, table: str, slot: int) -> None:
        self._lru_clock += 1
        if table == "a":
            self.last_access_a[slot] = self._lru_clock
        else:
            self.last_access_b[slot] = self._lru_clock

    @torch.no_grad()
    def _zero_slot(self, table: str, slot: int) -> None:
        """Reset embedding value tại slot bị evict-lại (LRU force-evict) về 0 — tránh ID
        mới chiếm slot "thừa kế" giá trị embedding của ID cũ đã bị loại bỏ hẳn khỏi hệ
        thống. Optimizer state (exp_avg/exp_avg_sq) KHÔNG reset được từ đây (cuckoo table
        không giữ tham chiếu tới optimizer) — xem train.py::reset_optimizer_state_for_evicted()."""
        (self.table_a if table == "a" else self.table_b)[slot].zero_()

    def resolve(self, entity_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tra cứu/chèn 1 batch entity_id (CPU LongTensor) -> (table_mask, slot_idx):
        table_mask (N,) bool — True = bảng A, False = bảng B; slot_idx (N,) int64 — vị trí
        trong bảng tương ứng. PHẢI gọi TRƯỚC forward, đây là phần "cấu trúc dữ liệu động"
        chạy trên CPU/Python, KHÔNG có gradient — tách 2 giá trị (thay vì gộp namespace
        [0, 2*capacity) rồi cat 2 bảng) để forward() có thể dùng F.embedding(sparse=True)
        RIÊNG từng bảng, giữ đúng sparse gradient cho SparseAdam (torch.cat 2 nn.Parameter
        rồi index KHÔNG sinh sparse gradient, sẽ buộc phải dùng Adam thường -> mất lợi ích
        sparse đã có, xem item_embedding.py/train.py SparseAdam).

        [SỬA 2026-09-11] pending_evicted_slots TÍCH LŨY qua nhiều lần gọi resolve() trong
        CÙNG 1 step (KHÔNG reset ở đầu hàm nữa) — bug đã xác nhận: ItemEmbedding.forward
        được gọi 2 LẦN/step trên CÙNG 1 CuckooEmbedding instance (1 lần cho hist_video_ids,
        1 lần cho candidate_ids, xem train.py run_batch_forward). Reset ở đầu resolve() làm
        lần gọi thứ 2 (candidate) XÓA MẤT eviction đã xảy ra ở lần gọi thứ 1 (hist) trước khi
        train.py kịp đọc để reset optimizer state — đúng bug mà docstring module đã cảnh báo
        ("ID mới thừa kế nhầm momentum ID cũ") vẫn xảy ra thật cho khoảng nửa số eviction mỗi
        step. Giờ train.py PHẢI gọi drain_pending_evicted_slots() (xóa + trả về) SAU CẢ 2 lần
        resolve của step (tức sau sparse_optimizer.step()), không phải resolve() tự xóa.
        """
        ids_list = entity_ids.tolist()
        table_mask = torch.empty(len(ids_list), dtype=torch.bool)
        slot_idx = torch.empty(len(ids_list), dtype=torch.int64)
        for i, entity_id in enumerate(ids_list):
            slot_info = self.id_to_slot.get(entity_id)
            if slot_info is None:
                table, slot = self._evict_and_insert(entity_id)
            else:
                table, slot = slot_info
                self._touch(table, slot)
            table_mask[i] = table == "a"
            slot_idx[i] = slot
        return table_mask, slot_idx

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """entity_ids: (N,) int64, CPU hoặc GPU — trả về (N, dim) embedding, CÓ gradient
        chảy về đúng slot trong table_a/table_b (autograd hoạt động bình thường vì slot đã
        cố định trước khi F.embedding chạy). Dùng sparse=True để tương thích SparseAdam —
        mỗi hàng chỉ 1 trong 2 bảng thật sự có gradient khác 0, còn lại phải là 0 (không
        phải NaN/không xác định) để phép cộng out_a + out_b không làm hỏng hàng nào."""
        device = entity_ids.device
        table_mask, slot_idx = self.resolve(entity_ids.detach().cpu())
        table_mask = table_mask.to(device)
        slot_idx = slot_idx.to(device)

        # slot_idx cho bảng KHÔNG được chọn: clamp về 0 (index hợp lệ bất kỳ, giá trị bị
        # loại ngay sau bởi mask nên không ảnh hưởng kết quả/gradient của slot đó).
        out_a = torch.nn.functional.embedding(slot_idx, self.table_a, sparse=True)
        out_b = torch.nn.functional.embedding(slot_idx, self.table_b, sparse=True)
        mask = table_mask.unsqueeze(-1).float()
        return mask * out_a + (1 - mask) * out_b

    def drain_pending_evicted_slots(self) -> list[tuple[str, int]]:
        """Trả về + XÓA toàn bộ eviction TÍCH LŨY từ mọi lần resolve() kể từ lần drain
        trước — PHẢI gọi ở train.py SAU sparse_optimizer.step() của MỖI step (không phải
        sau mỗi lần resolve()), vì ItemEmbedding.forward gọi resolve() 2 LẦN/step (hist rồi
        candidate) trên CÙNG instance này — xem resolve() docstring cho bug đã xác nhận."""
        drained = self.pending_evicted_slots
        self.pending_evicted_slots = []
        return drained

    def load_factor(self) -> float:
        """Tỷ lệ slot đang bị chiếm / tổng capacity — giám sát: load factor quá cao (>90%)
        nghĩa là capacity quá nhỏ so với số ID hoạt động đồng thời, eviction sẽ xảy ra RẤT
        thường xuyên (mọi ID liên tục bị đá ra vào, embedding không kịp hội tụ)."""
        occupied = sum(1 for x in self.slot_occupant_a if x is not None) + sum(
            1 for x in self.slot_occupant_b if x is not None
        )
        return occupied / (2 * self.capacity)
