"""ds_write_b32/b64/b96/b128, ds_read_b32/b64/b96/b128, ds_write_b8/b16 and ds_read_u8/i8/u16/i16 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions": the address is vaddr + offset (bytes), the dword at ADDR / 4; the b64/b96/b128
forms move 2/3/4 consecutive dwords, low first (the dword at ADDR + 4 k goes to / comes from VGPR k of the group).
LDS is `LDS_SIZE` bytes and the address wraps at that size. The 8/16-bit forms address a byte of the dword at ADDR / 4:
byte position (ADDR & 3), halfword (ADDR & 2), zero or sign extended on a read; a write replaces only those bits.

Each case is a compute shader translated by tools/rdna2spirv.py (support.translate_lines) and run for 32 lanes in
tests/semantic/spirv_mini_interp.py; every lane works in its own LDS words and stores what it read to an output
buffer. The expected words are written out by hand from the rules above, and a plain-Python LDS model gives the same
words for seeded random data. Until this file, the LDS tests of this name ran a self-built interpreter that never
reached the translator.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

M32 = L.M32
LDS_BYTES = 4096


class Model:
    """Plain-Python LDS: a dict of dwords by index, addresses wrapping at LDS_BYTES."""

    def __init__(self):
        self.w = {}

    def write32(self, addr, v):
        self.w[(addr % LDS_BYTES) // 4] = v & M32

    def read32(self, addr):
        return self.w.get((addr % LDS_BYTES) // 4, 0)

    def write_bits(self, addr, v, bits):
        sh = (addr & 3) * 8
        m = ((1 << bits) - 1) << sh
        self.w[(addr % LDS_BYTES) // 4] = (self.read32(addr) & ~m & M32) | ((v << sh) & m)

    def read_bits(self, addr, bits, signed):
        v = (self.read32(addr) >> ((addr & 3) * 8)) & ((1 << bits) - 1)
        if signed and v >> (bits - 1):
            v = (v | (M32 << bits)) & M32
        return v


class TestDwordForms(unittest.TestCase):
    def test_b32_offset_by_hand(self):
        # lane i: v1 = 8 i; write 0x1000 + i at v1 + 0x10, read it back at v1 + 0x10 through a different split.
        body = ["v_lshlrev_b32_e32 v1, 3, v0", "v_add_nc_u32_e32 v2, 0x1000, v0",
                "ds_write_b32 v1, v2 offset:0x10", "s_waitcnt lgkmcnt(0)",
                "v_add_nc_u32_e32 v3, 0x10, v1", "ds_read_b32 v4, v3", "s_waitcnt lgkmcnt(0)",
                "ds_read_b32 v5, v1 offset:0x10", "s_waitcnt lgkmcnt(0)"] + L.store("v4", 0) + L.store("v5", 1)
        mem = L.run(body)
        for i in range(32):
            self.assertEqual((L.out(mem, i, 0), L.out(mem, i, 1)), (0x1000 + i, 0x1000 + i), f"lane {i}")

    def test_wide_forms_by_hand(self):
        # lane i: v1 = 16 i; v[4:7] = four known words; write as b128, read back as b128, b96, b64 and b32 + offsets.
        body = ["v_lshlrev_b32_e32 v1, 4, v0", "v_add_nc_u32_e32 v4, 0x100, v0", "v_add_nc_u32_e32 v5, 0x200, v0",
                "v_add_nc_u32_e32 v6, 0x300, v0", "v_add_nc_u32_e32 v7, 0x400, v0",
                "ds_write_b128 v1, v[4:7]", "s_waitcnt lgkmcnt(0)",
                "ds_read_b128 v[8:11], v1", "ds_read_b96 v[12:14], v1", "ds_read_b64 v[16:17], v1 offset:8",
                "ds_read_b32 v18, v1 offset:12", "s_waitcnt lgkmcnt(0)"]
        for k, r in enumerate(("v8", "v9", "v10", "v11", "v12", "v13", "v14")):
            body += L.store(r, k)
        body += L.store("v16", 7)
        mem = L.run(body)
        for i in range(32):
            got = [L.out(mem, i, k) for k in range(8)]
            want = [0x100 + i, 0x200 + i, 0x300 + i, 0x400 + i, 0x100 + i, 0x200 + i, 0x300 + i, 0x300 + i]
            self.assertEqual(got, want, f"lane {i}")

    def test_b64_and_b96_write(self):
        body = ["v_lshlrev_b32_e32 v1, 4, v0", "v_add_nc_u32_e32 v4, 0x11, v0", "v_add_nc_u32_e32 v5, 0x22, v0",
                "v_add_nc_u32_e32 v6, 0x33, v0",
                "ds_write_b64 v1, v[4:5] offset:4", "ds_write_b96 v1, v[4:6] offset:0x800", "s_waitcnt lgkmcnt(0)",
                "ds_read_b32 v8, v1 offset:4", "ds_read_b32 v9, v1 offset:8", "ds_read_b32 v10, v1 offset:0x800",
                "ds_read_b32 v11, v1 offset:0x808", "s_waitcnt lgkmcnt(0)"]
        for k, r in enumerate(("v8", "v9", "v10", "v11")):
            body += L.store(r, k)
        mem = L.run(body)
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(4)], [0x11 + i, 0x22 + i, 0x11 + i, 0x33 + i], f"lane {i}")

    def test_address_wraps_at_lds_size(self):
        # LDS_SIZE is 4 KiB here (and no offset grows it): byte 4096 + 8 is byte 8, with the offset or without.
        body = ["v_mov_b32_e32 v1, 0x1008", "v_add_nc_u32_e32 v2, 0x7000, v0",
                "v_lshlrev_b32_e32 v3, 6, v0",                 # lane i uses its own word: 64 i
                "v_add_nc_u32_e32 v1, v1, v3",
                "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)",
                "v_add_nc_u32_e32 v4, 8, v3", "ds_read_b32 v5, v4", "ds_read_b32 v6, v3 offset:8",
                "s_waitcnt lgkmcnt(0)"] + L.store("v5", 0) + L.store("v6", 1)
        mem = L.run(body)
        for i in range(32):
            self.assertEqual((L.out(mem, i, 0), L.out(mem, i, 1)), (0x7000 + i, 0x7000 + i), f"lane {i}")


class TestSubDwordForms(unittest.TestCase):
    WORD = 0x7F80FF01        # bytes 01 FF 80 7F

    def test_reads_by_hand(self):
        body = ["v_lshlrev_b32_e32 v1, 3, v0", "v_mov_b32_e32 v2, 0x7f80ff01",
                "ds_write_b32 v1, v2 offset:0x40", "s_waitcnt lgkmcnt(0)",
                "ds_read_u8 v4, v1 offset:0x41", "ds_read_i8 v5, v1 offset:0x41", "ds_read_i8 v6, v1 offset:0x43",
                "ds_read_u16 v7, v1 offset:0x42", "ds_read_i16 v8, v1 offset:0x40", "ds_read_i16 v9, v1 offset:0x42",
                "s_waitcnt lgkmcnt(0)"]
        for k, r in enumerate(("v4", "v5", "v6", "v7", "v8", "v9")):
            body += L.store(r, k)
        mem = L.run(body)
        want = [0xFF, 0xFFFFFFFF, 0x7F, 0x7F80, 0xFFFFFF01, 0x7F80]
        # i16 at byte 0 is 0xFF01 -> sign extended 0xFFFFFF01; at byte 2 is 0x7F80 (positive).
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(6)], want, f"lane {i}")

    def test_writes_replace_only_their_bits(self):
        body = ["v_lshlrev_b32_e32 v1, 3, v0", "v_mov_b32_e32 v2, 0x11223344", "v_mov_b32_e32 v3, 0xaabbccdd",
                "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)",
                "ds_write_b8 v1, v3 offset:1", "ds_write_b16 v1, v3 offset:2", "s_waitcnt lgkmcnt(0)",
                "ds_read_b32 v4, v1", "s_waitcnt lgkmcnt(0)",
                "ds_write_b8_d16_hi v1, v3", "ds_write_b16_d16_hi v1, v3 offset:4", "s_waitcnt lgkmcnt(0)",
                "ds_read_b32 v5, v1", "ds_read_b32 v6, v1 offset:4", "s_waitcnt lgkmcnt(0)"]
        for k, r in enumerate(("v4", "v5", "v6")):
            body += L.store(r, k)
        mem = L.run(body)
        # 0x11223344: b8 0xdd at byte 1 -> 0x1122dd44; b16 0xccdd at byte 2 -> 0xccdddd44;
        # b8_d16_hi stores bits 23:16 of v3 (0xbb) at byte 0 -> 0xccdddbb... = 0xccddddbb;
        # b16_d16_hi stores bits 31:16 of v3 (0xaabb) at byte 4: the neighbouring dword was never written (0) -> 0xaabb.
        for i in range(32):
            self.assertEqual([L.out(mem, i, k) for k in range(3)], [0xCCDDDD44, 0xCCDDDDBB, 0xAABB], f"lane {i}")

    def test_seeded_against_the_model(self):
        rng = random.Random(0x1D5F)
        for trial in range(6):
            ops = []
            body = ["v_lshlrev_b32_e32 v1, 4, v0"]
            model = Model()
            base = 0                    # the model is lane 0's words; every lane runs the same program on its own
            word, wr = rng.getrandbits(32), rng.getrandbits(32)
            body += [f"v_mov_b32_e32 v2, {word:#x}", f"v_mov_b32_e32 v3, {wr:#x}", "ds_write_b32 v1, v2",
                     "s_waitcnt lgkmcnt(0)"]
            model.write32(base, word)
            for _ in range(4):
                kind = rng.choice(("b8", "b16"))
                off = rng.randrange(0, 4 if kind == "b8" else 3, 1 if kind == "b8" else 2)
                body += [f"ds_write_{kind} v1, v3 offset:{off}", "s_waitcnt lgkmcnt(0)"]
                model.write_bits(base + off, wr, 8 if kind == "b8" else 16)
                wr = rng.getrandbits(32)
                body += [f"v_mov_b32_e32 v3, {wr:#x}"]
            for k, (op, bits, sg) in enumerate((("u8", 8, 0), ("i8", 8, 1), ("u16", 16, 0), ("i16", 16, 1))):
                off = rng.randrange(0, 4, bits // 8)
                body += [f"ds_read_{op} v{4 + k}, v1 offset:{off}", "s_waitcnt lgkmcnt(0)"] + L.store(f"v{4 + k}", k)
                ops.append(model.read_bits(base + off, bits, sg))
            body += ["ds_read_b32 v9, v1", "s_waitcnt lgkmcnt(0)"] + L.store("v9", 4)
            ops.append(model.read32(base))
            mem = L.run(body)
            for i in (0, 5, 31):
                self.assertEqual([L.out(mem, i, k) for k in range(5)], ops, f"trial {trial} lane {i}")


class TestOutsideCompute(unittest.TestCase):
    def test_pixel_shader_refuses_lds(self):
        for line in ("ds_read_b32 v1, v0", "ds_write_b32 v0, v1", "ds_add_u32 v0, v1"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines("ps_lds_refused", [line, "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
