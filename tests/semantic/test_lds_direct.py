"""ds_write2_b32, ds_read2_b32, ds_write2st64_b32 and ds_read2st64_b32 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": the 2-address forms access two dwords at
    ADDR0 = vaddr + offset0 * 4,  ADDR1 = vaddr + offset1 * 4          (the st64 forms: offset * 4 * 64)
  * ds_write2_b32 vaddr, vdata0, vdata1:  MEM[ADDR0] = vdata0, MEM[ADDR1] = vdata1
  * ds_read2_b32  vdst[0:1], vaddr:       vdst[0] = MEM[ADDR0], vdst[1] = MEM[ADDR1]
(offset0 and offset1 are 8-bit unit counts; the b64 forms are in test_lds_read2_write2_b64.py.)

Each case is a compute shader translated by tools/rdna2spirv.py (support.translate_lines) and run for 32 lanes in
tests/semantic/spirv_mini_interp.py. Lane i uses vaddr = 4 i, so with offsets that are multiples of 32 dwords
(non-st64) or of 64 dwords (st64) the lanes never share a dword. A plain-Python LDS model, written from the
formula above, gives the expected words for seeded random offsets; the hand-written cases pin the unit.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

RSRC2_8K = 0x80088           # 4 user SGPRs, workgroup id in s4, LDS_SIZE = 16 x 512 bytes
LDS_BYTES = 8192


def lane_program(write, read, unit_bytes, off0, off1, roff0, roff1):
    """Lane i writes (0x5000 + i, 0x6000 + i) with `write`, reads two dwords with `read`, and reads the dwords at
    the write offsets back with plain ds_read_b32 (byte offsets off * unit_bytes)."""
    body = ["v_lshlrev_b32_e32 v1, 2, v0", "v_add_nc_u32_e32 v2, 0x5000, v0", "v_add_nc_u32_e32 v3, 0x6000, v0",
            f"{write} v1, v2, v3 offset0:{off0} offset1:{off1}", "s_waitcnt lgkmcnt(0)",
            f"{read} v[4:5], v1 offset0:{roff0} offset1:{roff1}",
            f"ds_read_b32 v6, v1 offset:{off0 * unit_bytes}", f"ds_read_b32 v7, v1 offset:{off1 * unit_bytes}",
            "s_waitcnt lgkmcnt(0)"]
    for k, r in enumerate(("v4", "v5", "v6", "v7")):
        body += L.store(r, k)
    return body


def model(lane, unit_bytes, off0, off1, roff0, roff1):
    lds = {(4 * lane + off0 * unit_bytes) % LDS_BYTES: 0x5000 + lane,
           (4 * lane + off1 * unit_bytes) % LDS_BYTES: 0x6000 + lane}       # off1 wins a tie, as the later store
    rd = lambda off: lds.get((4 * lane + off * unit_bytes) % LDS_BYTES, 0)   # noqa: E731
    return [rd(roff0), rd(roff1), rd(off0), rd(off1)]


class TestRead2Write2(unittest.TestCase):
    def run_case(self, write, read, unit_bytes, off0, off1, roff0, roff1):
        mem = L.run(lane_program(write, read, unit_bytes, off0, off1, roff0, roff1), rsrc2=RSRC2_8K)
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(4)], model(i, unit_bytes, off0, off1, roff0, roff1),
                             f"lane {i}")

    def test_b32_by_hand(self):
        # write at +0 and +128 bytes (offset 32 dwords); read2 with the offsets swapped returns [second, first].
        mem = L.run(lane_program("ds_write2_b32", "ds_read2_b32", 4, 0, 32, 32, 0), rsrc2=RSRC2_8K)
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(4)], [0x6000 + i, 0x5000 + i, 0x5000 + i, 0x6000 + i])

    def test_st64_by_hand(self):
        # st64: offsets count 256-byte units; writes at +0 and +256 (offset1:1), read2st64 with (1, 0).
        mem = L.run(lane_program("ds_write2st64_b32", "ds_read2st64_b32", 256, 0, 1, 1, 0), rsrc2=RSRC2_8K)
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(4)], [0x6000 + i, 0x5000 + i, 0x5000 + i, 0x6000 + i])

    def test_equal_offsets_read_the_same_dword_twice(self):
        mem = L.run(lane_program("ds_write2_b32", "ds_read2_b32", 4, 0, 32, 32, 32), rsrc2=RSRC2_8K)
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(2)], [0x6000 + i, 0x6000 + i])

    def test_seeded_offsets(self):
        rng = random.Random(0xD1EC7)
        for _ in range(8):
            a, b = rng.sample(range(0, 255, 32), 2)
            ra, rb = rng.choice((a, b)), rng.choice((a, b))
            with self.subTest(kind="b32", off0=a, off1=b, r0=ra, r1=rb):
                self.run_case("ds_write2_b32", "ds_read2_b32", 4, a, b, ra, rb)
            c, d = rng.sample(range(0, 8), 2)
            rc, rd = rng.choice((c, d)), rng.choice((c, d))
            with self.subTest(kind="st64", off0=c, off1=d, r0=rc, r1=rd):
                self.run_case("ds_write2st64_b32", "ds_read2st64_b32", 256, c, d, rc, rd)


class TestOutsideCompute(unittest.TestCase):
    def test_pixel_shader_refuses_read2_and_write2(self):
        for line in ("ds_read2_b32 v[0:1], v2 offset1:1", "ds_write2_b32 v0, v1, v2 offset1:1",
                     "ds_read2st64_b32 v[0:1], v2 offset1:1", "ds_write2st64_b32 v0, v1, v2 offset1:1"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines("ps_lds_read2", [line, "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
