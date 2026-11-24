"""Semantic tests for LDS sub-dword and d16 read/write instructions.

Standard library only; no GPU, no LLVM, no game files.

    python -I tests/semantic/test_lds_subword_d16.py -v

Covers RDNA2 Data Share (DS) sub-dword and d16 operations translated by tools/rdna2spirv.py:
  * ds_write_b8 (all 4 byte lanes 0..3) and ds_write_b16 (byte lanes 0 and 2),
    verifying via ds_read_b32 that adjacent bytes/halfwords in the same dword stay intact.
  * ds_write_b8_d16_hi (extracts source bits [23:16]) and ds_write_b16_d16_hi
    (extracts source bits [31:16]), verifying low source bits are ignored and adjacent LDS bytes stay intact.
  * ds_read_i8, ds_read_u8, ds_read_i16, ds_read_u16 on positive (0x7F, 0x7FFF) and
    negative (0x80, 0xFF, 0x8000, 0xFFFF) values, verifying sign-extension vs zero-extension.
  * ds_read_u16_d16 (writes [15:0], preserves dest [31:16]) and
    ds_read_u16_d16_hi (writes [31:16], preserves dest [15:0]).
  * In-memory SPIR-V negative controls (OpBitFieldSExtract <-> OpBitFieldUExtract,
    d16 insert offset 0 <-> 16, and d16_hi source shift).
"""
import os
import re
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

# 4 user SGPRs (the output V# in s[0:3]), workgroup id in s4, 512 bytes of LDS, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def translate_cs(lines, name="cs_lds_subword_d16"):
    return support.translate_lines(name, lines, fssm=FSSM)


def run_cs(spv, out_words=1):
    memory = {}
    si.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0), "%sg_lane": 0},
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )
    return [memory.get(4 * k) for k in range(out_words)]


def emit_stores(regs):
    lines = []
    for k, reg in enumerate(regs):
        lines.append(f"v_mov_b32_e32 v21, {4 * k}")
        lines.append(f"buffer_store_dword {reg}, v21, s[0:3], 0 offen")
    lines.append("s_endpgm")
    return lines


class LdsSubwordD16Tests(unittest.TestCase):
    def test_ds_write_b8_all_positions_preserve_neighbors(self):
        # Start with dword 0 = 0xA1B2C3D4, overwrite byte 0..3 one by one, reading b32 after each write.
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xa1b2c3d4",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            # Byte 0 via vaddr=0
            "v_mov_b32_e32 v3, 0xffffff11",
            "ds_write_b8 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v10, v1",
            # Byte 1 via vaddr=1
            "v_mov_b32_e32 v4, 1",
            "v_mov_b32_e32 v3, 0xffffff22",
            "ds_write_b8 v4, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v11, v1",
            # Byte 2 via vaddr=0, offset:2
            "v_mov_b32_e32 v3, 0xffffff33",
            "ds_write_b8 v1, v3 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v12, v1",
            # Byte 3 via vaddr=2, offset:1
            "v_mov_b32_e32 v4, 2",
            "v_mov_b32_e32 v3, 0xffffff44",
            "ds_write_b8 v4, v3 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v13, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11", "v12", "v13"])
        spv = translate_cs(lines, "cs_ds_write_b8")
        got = run_cs(spv, out_words=4)
        self.assertEqual(got, [0xA1B2C311, 0xA1B22211, 0xA1332211, 0x44332211])

    def test_ds_write_b16_positions_0_and_2_preserve_neighbors(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            # Halfword 0 via vaddr=0 (high source bits 0xFFFF must be masked off)
            "v_mov_b32_e32 v3, 0xffffaabb",
            "ds_write_b16 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v10, v1",
            # Halfword 2 via vaddr=0, offset:2
            "v_mov_b32_e32 v3, 0xffffccdd",
            "ds_write_b16 v1, v3 offset:2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v11, v1",
            # Halfword 2 via vaddr=2 on fresh dword 1 = 0x89abcdef
            "v_mov_b32_e32 v4, 4",
            "v_mov_b32_e32 v2, 0x89abcdef",
            "ds_write_b32 v4, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v5, 6",
            "v_mov_b32_e32 v3, 0xdead0055",
            "ds_write_b16 v5, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v12, v4",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11", "v12"])
        spv = translate_cs(lines, "cs_ds_write_b16")
        got = run_cs(spv, out_words=3)
        self.assertEqual(got, [0x1234AABB, 0xCCDDAABB, 0x0055CDEF])

    def test_ds_write_b8_d16_hi_extracts_bits_23_16(self):
        # Source 0xDE55BEEF: bits [23:16] = 0x55; [7:0]=0xEF, [15:8]=0xBE, [31:24]=0xDE must not be written.
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xa1b2c3d4",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xde55beef",
            "ds_write_b8_d16_hi v1, v3 offset:1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v10, v1",
            "v_mov_b32_e32 v3, 0xde77beef",
            "ds_write_b8_d16_hi v1, v3 offset:3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v11, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11"])
        spv = translate_cs(lines, "cs_ds_write_b8_d16_hi")
        got = run_cs(spv, out_words=2)
        self.assertEqual(got, [0xA1B255D4, 0x77B255D4])

    def test_ds_write_b16_d16_hi_extracts_bits_31_16(self):
        # Source 0xCAFEBEEF: bits [31:16] = 0xCAFE; [15:0] = 0xBEEF must not be written.
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xcafebeef",
            "ds_write_b16_d16_hi v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v10, v1",
            "v_mov_b32_e32 v4, 2",
            "v_mov_b32_e32 v3, 0xbabef00d",
            "ds_write_b16_d16_hi v4, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v11, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11"])
        spv = translate_cs(lines, "cs_ds_write_b16_d16_hi")
        got = run_cs(spv, out_words=2)
        self.assertEqual(got, [0x1234CAFE, 0xBABECAFE])

    def test_ds_read_i8_u8_sign_vs_zero_extension(self):
        # Dword 0 = 0x00FF807F -> byte 0=0x7F, byte 1=0x80, byte 2=0xFF, byte 3=0x00.
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x00ff807f",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v10, v1 offset:0",
            "ds_read_u8 v11, v1 offset:0",
            "ds_read_i8 v12, v1 offset:1",
            "ds_read_u8 v13, v1 offset:1",
            "ds_read_i8 v14, v1 offset:2",
            "ds_read_u8 v15, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11", "v12", "v13", "v14", "v15"])
        spv = translate_cs(lines, "cs_ds_read_8")
        got = run_cs(spv, out_words=6)
        self.assertEqual(got, [
            0x0000007F,  # i8(0x7F)
            0x0000007F,  # u8(0x7F)
            0xFFFFFF80,  # i8(0x80)
            0x00000080,  # u8(0x80)
            0xFFFFFFFF,  # i8(0xFF)
            0x000000FF,  # u8(0xFF)
        ])

    def test_ds_read_i16_u16_sign_vs_zero_extension(self):
        # Dword 0 = 0x80007FFF (half0=0x7FFF, half2=0x8000); Dword 1 = 0x0001FFFF (half0=0xFFFF).
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x80007fff",
            "ds_write_b32 v1, v2",
            "v_mov_b32_e32 v2, 0x0001ffff",
            "ds_write_b32 v1, v2 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v10, v1 offset:0",
            "ds_read_u16 v11, v1 offset:0",
            "ds_read_i16 v12, v1 offset:2",
            "ds_read_u16 v13, v1 offset:2",
            "ds_read_i16 v14, v1 offset:4",
            "ds_read_u16 v15, v1 offset:4",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11", "v12", "v13", "v14", "v15"])
        spv = translate_cs(lines, "cs_ds_read_16")
        got = run_cs(spv, out_words=6)
        self.assertEqual(got, [
            0x00007FFF,  # i16(0x7FFF)
            0x00007FFF,  # u16(0x7FFF)
            0xFFFF8000,  # i16(0x8000)
            0x00008000,  # u16(0x8000)
            0xFFFFFFFF,  # i16(0xFFFF)
            0x0000FFFF,  # u16(0xFFFF)
        ])

    def test_ds_read_u16_d16_and_d16_hi_preserve_opposite_half(self):
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xcafebeef",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, 0x12345678",
            "v_mov_b32_e32 v11, 0x12345678",
            "v_mov_b32_e32 v12, 0x12345678",
            "v_mov_b32_e32 v13, 0x12345678",
            "ds_read_u16_d16 v10, v1 offset:0",
            "ds_read_u16_d16 v11, v1 offset:2",
            "ds_read_u16_d16_hi v12, v1 offset:0",
            "ds_read_u16_d16_hi v13, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10", "v11", "v12", "v13"])
        spv = translate_cs(lines, "cs_ds_read_u16_d16")
        got = run_cs(spv, out_words=4)
        self.assertEqual(got, [
            0x1234BEEF,  # u16_d16(0xBEEF) into [15:0], [31:16] preserved
            0x1234CAFE,  # u16_d16(0xCAFE) into [15:0], [31:16] preserved
            0xBEEF5678,  # u16_d16_hi(0xBEEF) into [31:16], [15:0] preserved
            0xCAFE5678,  # u16_d16_hi(0xCAFE) into [31:16], [15:0] preserved
        ])

    def test_negative_controls_spirv_in_memory(self):
        # 1) Signedness swap on ds_read_i8 (0x80 -> 0xFFFFFF80 must break if OpBitFieldUExtract is used).
        spv_i8 = translate_cs([
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x00000080",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v10, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10"]), "cs_neg_i8")
        self.assertEqual(run_cs(spv_i8), [0xFFFFFF80])
        self.assertIn("OpBitFieldSExtract", spv_i8)
        bad_i8 = spv_i8.replace("OpBitFieldSExtract", "OpBitFieldUExtract")
        self.assertEqual(run_cs(bad_i8), [0x00000080])
        self.assertNotEqual(run_cs(bad_i8), [0xFFFFFF80])

        # 2) Signedness swap on ds_read_u16 (0x8000 -> 0x00008000 must break if OpBitFieldSExtract is used).
        spv_u16 = translate_cs([
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x00008000",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v10, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10"]), "cs_neg_u16")
        self.assertEqual(run_cs(spv_u16), [0x00008000])
        m_c16_u = re.search(r"^\s*(%\w+) = OpConstant %uint 16$", spv_u16, re.MULTILINE)
        self.assertIsNotNone(m_c16_u)
        m_uext = re.search(
            rf"^(\s*%\w+ = )OpBitFieldUExtract( %uint %\w+ %\w+ {re.escape(m_c16_u.group(1))})$",
            spv_u16,
            re.MULTILINE,
        )
        self.assertIsNotNone(m_uext)
        bad_u16 = spv_u16.replace(m_uext.group(0), f"{m_uext.group(1)}OpBitFieldSExtract{m_uext.group(2)}")
        self.assertEqual(run_cs(bad_u16), [0xFFFF8000])
        self.assertNotEqual(run_cs(bad_u16), [0x00008000])

        # 3) d16 insert offset swap (0 <-> 16) on ds_read_u16_d16 and ds_read_u16_d16_hi.
        base_d16 = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x0000beef",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, 0x12345678",
        ]
        spv_lo = translate_cs(base_d16 + [
            "ds_read_u16_d16 v10, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10"]), "cs_neg_d16_lo")
        self.assertEqual(run_cs(spv_lo), [0x1234BEEF])
        m_lo = re.search(r"^(\s*%\w+ = OpBitFieldInsert %uint %\w+ %\w+) (%\w+) (%\w+)$", spv_lo, re.MULTILINE)
        self.assertIsNotNone(m_lo)
        # Swap offset=0 to offset=16 (reusing the 16-bit count constant).
        bad_lo = spv_lo.replace(m_lo.group(0), f"{m_lo.group(1)} {m_lo.group(3)} {m_lo.group(3)}")
        self.assertEqual(run_cs(bad_lo), [0xBEEF5678])

        spv_hi = translate_cs(base_d16 + [
            "ds_read_u16_d16_hi v10, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10"]), "cs_neg_d16_hi")
        self.assertEqual(run_cs(spv_hi), [0xBEEF5678])
        m_c0 = re.search(r"^\s*(%\w+) = OpConstant %uint 0$", spv_hi, re.MULTILINE)
        m_hi = re.search(r"^(\s*%\w+ = OpBitFieldInsert %uint %\w+ %\w+) (%\w+) (%\w+)$", spv_hi, re.MULTILINE)
        self.assertIsNotNone(m_c0)
        self.assertIsNotNone(m_hi)
        bad_hi = spv_hi.replace(m_hi.group(0), f"{m_hi.group(1)} {m_c0.group(1)} {m_hi.group(3)}")
        self.assertEqual(run_cs(bad_hi), [0x1234BEEF])

        # 4) d16_hi source shift swap (16 -> 0) on ds_write_b16_d16_hi.
        spv_w_hi = translate_cs([
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xcafebeef",
            "ds_write_b16_d16_hi v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v10, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + emit_stores(["v10"]), "cs_neg_w_hi")
        self.assertEqual(run_cs(spv_w_hi), [0x1234CAFE])
        m_c0_w = re.search(r"^\s*(%\w+) = OpConstant %uint 0$", spv_w_hi, re.MULTILINE)
        m_c16_w = re.search(r"^\s*(%\w+) = OpConstant %uint 16$", spv_w_hi, re.MULTILINE)
        self.assertIsNotNone(m_c0_w)
        self.assertIsNotNone(m_c16_w)
        m_shr = re.search(
            rf"^(\s*%\w+ = OpShiftRightLogical %uint %\w+) {re.escape(m_c16_w.group(1))}$",
            spv_w_hi,
            re.MULTILINE,
        )
        self.assertIsNotNone(m_shr)
        bad_w_hi = spv_w_hi.replace(m_shr.group(0), f"{m_shr.group(1)} {m_c0_w.group(1)}")
        self.assertEqual(run_cs(bad_w_hi), [0x1234BEEF])


if __name__ == "__main__":
    unittest.main()
