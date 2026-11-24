"""Known-answer tests for tools/cpu_ref.py.

These use hand-written .s listings, so they need no llvm-mc, SPIR-V tools, GPU or game data.

Run from the repository root:
    python -I tests/test_cpu_ref.py
    python -m unittest discover -s tests -p "test_cpu_ref.py"
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cpu_ref  # noqa: E402

QNAN = 0x7FC00000
SIGN = 0x80000000


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class CpuRefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cpu-ref-")
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def write(self, body):
        self.n += 1
        path = os.path.join(self.tmp.name, f"t{self.n}.s")
        lines = "; test  code 0x0..0x1000\n" + "\n".join("\t" + line for line in body.strip().splitlines()) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(lines)
        return path

    def execute(self, body, **inputs):
        return cpu_ref.run(self.write(body), inputs)

    def vgpr(self, result, index):
        return result["registers"]["vgpr"][index]

    def sgpr(self, result, index):
        return result["registers"]["sgpr"][index]

    def test_integer_add_sub_mul(self):
        r = self.execute("""
            v_mov_b32 v0, 0xffffffff
            v_mov_b32 v1, 2
            v_add_nc_u32 v2, v0, v1
            v_sub_nc_u32 v3, v0, v1
            v_subrev_nc_u32 v4, v0, v1
            v_mul_u32_u24 v5, v0, v1
            v_mul_i32_i24 v6, 0x800000, 2
            v_mul_lo_u32 v7, v0, v1
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 2), 1)             # wraps
        self.assertEqual(self.vgpr(r, 3), 0xFFFFFFFD)    # dst = src0 - src1
        self.assertEqual(self.vgpr(r, 4), 3)             # dst = src1 - src0
        self.assertEqual(self.vgpr(r, 5), 0x1FFFFFE)     # low 24 bits of each source
        self.assertEqual(self.vgpr(r, 6), 0xFF000000)    # -2**23 * 2
        self.assertEqual(self.vgpr(r, 7), 0xFFFFFFFE)    # low 32 bits

    def test_integer_min_max_med3(self):
        r = self.execute("""
            v_mov_b32 v0, 0xfffffffb
            v_mov_b32 v1, 3
            v_mov_b32 v2, 0
            v_min_i32 v3, v0, v1
            v_max_i32 v4, v0, v1
            v_min_u32 v5, v0, v1
            v_max_u32 v6, v0, v1
            v_med3_i32 v7, v0, v1, v2
            v_med3_u32 v8, v0, v1, v2
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 3), 0xFFFFFFFB)    # -5
        self.assertEqual(self.vgpr(r, 4), 3)
        self.assertEqual(self.vgpr(r, 5), 3)
        self.assertEqual(self.vgpr(r, 6), 0xFFFFFFFB)
        self.assertEqual(self.vgpr(r, 7), 0)             # median(-5, 3, 0)
        self.assertEqual(self.vgpr(r, 8), 3)             # median(4294967291, 3, 0)

    def test_bitfield_extract_and_insert(self):
        r = self.execute("""
            v_mov_b32 v0, 0xabcdef12
            v_bfe_u32 v1, v0, 28, 8
            v_bfe_i32 v2, v0, 28, 8
            v_bfe_u32 v3, v0, 4, 0
            v_bfe_i32 v4, v0, 0, 4
            v_mov_b32 v5, 0xff00ff00
            v_mov_b32 v6, 0x12345678
            v_mov_b32 v7, 0xaabbccdd
            v_bfi_b32 v8, v5, v6, v7
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 1), 0xA)           # width cut to 32 - 28
        self.assertEqual(self.vgpr(r, 2), 0xFFFFFFFA)    # sign-extended 0xA
        self.assertEqual(self.vgpr(r, 3), 0)             # a zero width extracts nothing
        self.assertEqual(self.vgpr(r, 4), 2)
        self.assertEqual(self.vgpr(r, 8), 0x12BB56DD)    # (x & y) | (~x & z)

    def test_align_and_permute(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11223344
            v_mov_b32 v1, 0xaabbccdd
            v_mov_b32 v2, 8
            v_alignbit_b32 v3, v0, v1, v2
            v_mov_b32 v2, 3
            v_alignbyte_b32 v4, v0, v1, v2
            v_mov_b32 v2, 0
            v_alignbit_b32 v5, v0, v1, v2
            v_mov_b32 v2, 0x0c0b0400
            v_perm_b32 v6, v0, v1, v2
            v_mov_b32 v2, 0x0d080000
            v_perm_b32 v7, v0, v1, v2
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 3), 0x44AABBCC)    # {v0, v1} >> 8
        self.assertEqual(self.vgpr(r, 4), 0x223344AA)    # {v0, v1} >> 24
        self.assertEqual(self.vgpr(r, 5), 0xAABBCCDD)    # a zero shift keeps src1
        self.assertEqual(self.vgpr(r, 6), 0x000044DD)    # bytes 0, 4; sign 0; zero
        self.assertEqual(self.vgpr(r, 7), 0xFFFFDDDD)    # byte 0, sign of src1 half, 0xff

    def test_perm_sign_bit_selectors(self):
        r = self.execute("""
            v_mov_b32 v0, 0x80000000
            v_mov_b32 v1, 0x80000000
            v_mov_b32 v2, 0x0b0a0908
            v_perm_b32 v3, v0, v1, v2
            v_mov_b32 v0, 0x00008000
            v_mov_b32 v1, 0x00008000
            v_perm_b32 v4, v0, v1, v2
            s_endpgm
        """)
        # Selectors 8/9 read bit 7 of src1 bytes 1/3 (bits 15/31), 10/11 the same of src0.
        self.assertEqual(self.vgpr(r, 3), 0xFF00FF00)
        self.assertEqual(self.vgpr(r, 4), 0x00FF00FF)

    def test_mul_hi_and_mad(self):
        r = self.execute("""
            v_mul_hi_u32 v0, 0xffffffff, 0xffffffff
            v_mul_hi_i32 v1, 0xffffffff, 1
            v_mul_hi_i32 v2, 0x40000000, 0x40000000
            v_mul_hi_i32 v3, 0x80000000, 0x40000000
            v_mad_u32_u24 v4, 0xffffff, 0xffffff, 5
            v_mad_i32_i24 v5, 0xffffff, 0xffffff, 5
            v_mad_i32_i24 v6, 0x800000, 0x800000, 1
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 0), 0xFFFFFFFE)
        self.assertEqual(self.vgpr(r, 1), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 2), 0x10000000)
        self.assertEqual(self.vgpr(r, 3), 0xE0000000)
        self.assertEqual(self.vgpr(r, 4), 0xFE000006)    # 0xffffff**2 + 5, low 32 bits
        self.assertEqual(self.vgpr(r, 5), 6)
        self.assertEqual(self.vgpr(r, 6), 1)             # 2**46 + 1, low 32 bits

    def test_mov_inline_constants_and_literals(self):
        r = self.execute("""
            v_mov_b32 v0, 1.0
            v_mov_b32 v1, 1
            v_mov_b32 v2, 0x3f800000
            v_mov_b32 v3, -1
            s_mov_b32 s0, 1.0
            s_mov_b32 s1, -1
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 0), fbits(1.0))
        self.assertEqual(self.vgpr(r, 1), 1)          # inline integer constant 1, not 1.0
        self.assertEqual(self.vgpr(r, 2), fbits(1.0))
        self.assertEqual(self.vgpr(r, 3), 0xFFFFFFFF)
        self.assertEqual(self.sgpr(r, 0), fbits(1.0))
        self.assertEqual(self.sgpr(r, 1), 0xFFFFFFFF)

    def test_float_add_sub_mul(self):
        r = self.execute("""
            v_mov_b32 v0, 1.0
            v_mov_b32 v1, 2.0
            v_add_f32 v2, v0, v1
            v_sub_f32 v3, v0, v1
            v_mul_f32 v4, v0, v1
            v_add_f32 v5, v1, 2
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 2), fbits(3.0))
        self.assertEqual(self.vgpr(r, 3), fbits(-1.0))
        self.assertEqual(self.vgpr(r, 4), fbits(2.0))
        self.assertEqual(self.vgpr(r, 5), fbits(2.0))   # integer inline 2 is the bits 0x2, a denormal

    def test_integer_inline_constant_in_float_op_is_raw_bits(self):
        # llvm-mc prints float inline constants with a point (2.0, -1.0); a bare integer is its bit pattern,
        # as in LLVM's AMDGPU operand syntax example "v_add_f32 v0, -1, 0 // src0 = 0xFFFFFFFF (NaN)".
        r = self.execute("""
            v_mov_b32 v0, 2.0
            v_mul_f32 v1, -1, v0
            v_add_f32 v2, 1, 0
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 1), 0x7FC00000)   # 0xffffffff is a NaN, canonicalised
        self.assertEqual(self.vgpr(r, 2), 1)            # the denormal 0x1, unflushed

    def test_nan_and_signed_zero(self):
        r = self.execute("""
            v_mov_b32 v0, 0x7f800000
            v_mov_b32 v1, 0xff800000
            v_add_f32 v2, v0, v1
            v_mov_b32 v3, 0
            v_mul_f32 v4, v0, v3
            v_mov_b32 v5, 0x80000000
            v_add_f32 v6, v5, v3
            v_add_f32 v7, v5, v5
            v_mul_f32 v8, v5, 1.0
            v_sub_f32 v9, v5, v3
            v_sub_f32 v10, v3, v3
            v_mov_b32 v11, 0x7fc00001
            v_add_f32 v12, v11, 1.0
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 2), QNAN)         # inf + -inf
        self.assertEqual(self.vgpr(r, 4), QNAN)         # inf * 0
        self.assertEqual(self.vgpr(r, 6), 0)            # -0 + +0 = +0
        self.assertEqual(self.vgpr(r, 7), SIGN)         # -0 + -0 = -0
        self.assertEqual(self.vgpr(r, 8), SIGN)         # -0 * 1 = -0
        self.assertEqual(self.vgpr(r, 9), SIGN)         # -0 - +0 = -0
        self.assertEqual(self.vgpr(r, 10), 0)           # +0 - +0 = +0
        self.assertEqual(self.vgpr(r, 12), QNAN)        # NaN payload is canonicalized

    def test_denormals_are_preserved(self):
        r = self.execute("""
            v_mov_b32 v0, 0x00000001
            v_add_f32 v1, v0, v0
            v_mul_f32 v2, v0, 0.5
            v_mul_f32 v3, v0, 1.5
            v_mov_b32 v4, 0x007fffff
            v_add_f32 v5, v4, v0
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 1), 0x00000002)   # min subnormal + itself
        self.assertEqual(self.vgpr(r, 2), 0x00000000)   # half of min subnormal, tie to even
        self.assertEqual(self.vgpr(r, 3), 0x00000002)   # 1.5 * min subnormal, tie to even
        self.assertEqual(self.vgpr(r, 5), 0x00800000)   # max subnormal + min subnormal = min normal

    def test_fma_is_fused_not_double_rounded(self):
        # 97/64 * 172961/262144 = 1 + 2**-24 exactly, a binary32 midpoint. Adding 2**-60
        # decides the direction; a host double fma would lose it and round to 1.0.
        r = self.execute("""
            v_mov_b32 v0, 0x3fc20000
            v_mov_b32 v1, 0x3f28e840
            v_mov_b32 v2, 0x21800000
            v_fma_f32 v3, v0, v1, v2
            v_mov_b32 v4, 2.0
            v_mov_b32 v5, 3.0
            v_mov_b32 v6, 4.0
            v_fma_f32 v7, v4, v5, v6
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 3), fbits(1.0 + 2.0 ** -23))
        self.assertEqual(self.vgpr(r, 7), fbits(10.0))

    def test_fma_special_values(self):
        r = self.execute("""
            v_mov_b32 v0, 0x7f800000
            v_mov_b32 v1, 0
            v_fma_f32 v2, v0, v1, 1.0
            v_mov_b32 v3, 0x80000000
            v_fma_f32 v4, v3, 0, v3
            v_fma_f32 v5, 1.0, 1.0, -1.0
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 2), QNAN)         # inf * 0 + 1
        self.assertEqual(self.vgpr(r, 4), SIGN)         # -0 * +0 + -0 = -0
        self.assertEqual(self.vgpr(r, 5), 0)            # exact cancellation, +0

    def test_cndmask_wave64(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            v_mov_b32 v1, 0x22222222
            v_cndmask_b32 v2, v0, v1, vcc_lo
            s_endpgm
        """, lanes=64, vcc=(1 << 1) | (1 << 3) | (1 << 40))
        # Lane 40 checks that VOP2's implicit condition uses the full 64-bit VCC, not only vcc_lo.
        expected = [0x22222222 if lane in (1, 3, 40) else 0x11111111 for lane in range(64)]
        self.assertEqual(self.vgpr(r, 2), expected)

    def test_cndmask_implicit_vcc_one_lane(self):
        body = """
            v_mov_b32 v0, 0x11111111
            v_mov_b32 v1, 0x22222222
            v_cndmask_b32 v2, v0, v1
            s_endpgm
        """
        set_bit = self.execute(body, lane=1, vcc=1 << 1)
        clear_bit = self.execute(body, lane=2, vcc=1 << 1)
        self.assertEqual(self.vgpr(set_bit, 2), 0x22222222)
        self.assertEqual(self.vgpr(clear_bit, 2), 0x11111111)

    def test_cndmask_sgpr_pair_wave64(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            v_mov_b32 v1, 0x22222222
            v_cndmask_b32_e64 v2, v0, v1, s[2:3]
            s_endpgm
        """, lanes=64, sgpr={2: 0x00000002, 3: 0x00000001})
        # Low half selects lane 1; high half selects lane 32.
        self.assertEqual(self.vgpr(r, 2), [0x11111111] + [0x22222222] +
                         [0x11111111] * 30 + [0x22222222] + [0x11111111] * 31)

    def test_cndmask_sgpr_wave32(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            v_mov_b32 v1, 0x22222222
            v_cndmask_b32_e64 v2, v0, v1, s2
            s_endpgm
        """, lane=1, sgpr={2: 0x00000002})
        self.assertEqual(self.vgpr(r, 2), 0x22222222)

    def test_cndmask_e64_abs_and_neg_modify_bit31(self):
        # FUZZ-20, measured natively on RDNA2 (gfx1035): VOP3 abs/neg on v_cndmask_b32_e64 clear/flip bit 31
        # (float sign-bit moves, abs first then neg), not two's-complement integer abs/negate.
        r = self.execute("""
            v_mov_b32 v0, 0x80000005
            v_mov_b32 v1, 0x00000005
            v_mov_b32 v2, 0x12345678
            s_mov_b32 s2, -1
            v_cndmask_b32_e64 v3, v2, -v0, s2
            v_cndmask_b32_e64 v4, v2, |v0|, s2
            v_cndmask_b32_e64 v5, v2, -|v0|, s2
            v_cndmask_b32_e64 v6, v2, -v1, s2
            v_cndmask_b32_e64 v7, -v0, v2, 0
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 3), 0x00000005)
        self.assertEqual(self.vgpr(r, 4), 0x00000005)
        self.assertEqual(self.vgpr(r, 5), 0x80000005)
        self.assertEqual(self.vgpr(r, 6), 0x80000005)
        self.assertEqual(self.vgpr(r, 7), 0x00000005)

    def test_cndmask_respects_exec(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            v_mov_b32 v1, 0x22222222
            s_mov_b32 exec_lo, 0x00000002
            s_mov_b32 exec_hi, 0
            v_cndmask_b32 v2, v0, v1, vcc_lo
            s_endpgm
        """, lanes=64, vcc=0xFFFFFFFFFFFFFFFF)
        # Only lane 1 is active, and it selects src1.
        self.assertEqual(self.vgpr(r, 2), [0] + [0x22222222] + [0] * 62)

    def test_exec_lo_does_not_touch_the_high_half(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            s_mov_b32 exec_lo, 0x00000000
            v_mov_b32 v0, 0x22222222
            s_endpgm
        """, lanes=64)
        # Writing exec_lo leaves exec_hi as it was, so lanes 32-63 still run the second write.
        self.assertEqual(self.vgpr(r, 0), [0x11111111] * 32 + [0x22222222] * 32)

    def test_shifts(self):
        r = self.execute("""
            v_mov_b32 v0, 7
            v_lshlrev_b32 v1, 2, v0
            v_mov_b32 v2, 33
            v_lshlrev_b32 v3, v2, v0
            v_mov_b32 v4, 0x80000000
            v_lshrrev_b32 v5, 31, v4
            v_ashrrev_i32 v6, 31, v4
            v_mov_b32 v7, 0xfffffffe
            v_ashrrev_i32 v8, 1, v7
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 1), 28)
        self.assertEqual(self.vgpr(r, 3), 14)           # amount 33 is masked to 1
        self.assertEqual(self.vgpr(r, 5), 1)
        self.assertEqual(self.vgpr(r, 6), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 8), 0xFFFFFFFF)

    def test_scalar_add_carry_and_shifts(self):
        r = self.execute("""
            s_mov_b32 s0, 0xffffffff
            s_add_u32 s1, s0, 1
            s_mov_b32 s2, scc
            s_mov_b32 s3, 0x80000000
            s_lshr_b32 s4, s3, 31
            s_mov_b32 s5, scc
            s_mov_b32 s6, 0
            s_lshl_b32 s7, s6, 1
            s_mov_b32 s8, scc
            s_endpgm
        """)
        self.assertEqual(self.sgpr(r, 1), 0)
        self.assertEqual(self.sgpr(r, 2), 1)            # carry out of s_add_u32
        self.assertEqual(self.sgpr(r, 4), 1)
        self.assertEqual(self.sgpr(r, 5), 1)            # s_lshr_b32: SCC = result != 0
        self.assertEqual(self.sgpr(r, 7), 0)
        self.assertEqual(self.sgpr(r, 8), 0)            # s_lshl_b32: zero result clears SCC

    def test_exec_mask_keeps_inactive_lanes(self):
        r = self.execute("""
            v_mov_b32 v0, 0x11111111
            s_mov_b32 exec_lo, 0xffff0000
            s_mov_b32 exec_hi, 0
            v_mov_b32 v0, 0x22222222
            s_endpgm
        """, lanes=64)
        expected = [0x11111111] * 16 + [0x22222222] * 16 + [0x11111111] * 32
        self.assertEqual(self.vgpr(r, 0), expected)

    def test_exec_mask_one_lane(self):
        body = """
            v_mov_b32 v0, 0x11111111
            s_mov_b32 exec_lo, 0xffff0000
            v_mov_b32 v0, 0x22222222
            s_endpgm
        """
        active = self.execute(body, lane=20)
        inactive = self.execute(body, lane=4)
        self.assertEqual(self.vgpr(active, 0), 0x22222222)
        self.assertEqual(self.vgpr(inactive, 0), 0x11111111)
        self.assertTrue(active["active"])
        self.assertFalse(inactive["active"])            # "active" is the lane's final EXEC bit

    def test_16_bit_literals_and_packed_halves(self):
        r = self.execute("""
            v_mov_b32 v0, 0x3c003c00
            v_and_b32 v1, v0, 0xffff
            v_lshrrev_b32 v2, 16, v0
            v_or_b32 v3, v1, v2
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 0), 0x3C003C00)
        self.assertEqual(self.vgpr(r, 1), 0x3C00)
        self.assertEqual(self.vgpr(r, 2), 0x3C00)
        self.assertEqual(self.vgpr(r, 3), 0x3C00)

    def test_64_lane_per_lane_inputs(self):
        values = [fbits(float(i)) for i in range(64)]
        r = self.execute("""
            v_add_f32 v1, v0, 1.0
            s_endpgm
        """, lanes=64, vgpr={0: values})
        self.assertEqual(self.vgpr(r, 1), [fbits(float(i) + 1.0) for i in range(64)])

    def test_clamp(self):
        r = self.execute("""
            v_add_f32 v0, 2.0, 2.0 clamp
            v_add_f32 v1, -1.0, -1.0 clamp
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 0), fbits(1.0))
        self.assertEqual(self.vgpr(r, 1), 0)

    def test_unsupported_instruction_is_reported(self):
        path = self.write("""
            v_pk_fma_f16 v0, v1, v2, v3
            s_endpgm
        """)
        with self.assertRaises(cpu_ref.Unsupported) as cm:
            cpu_ref.run(path)
        self.assertIn("v_pk_fma_f16", str(cm.exception))

    def test_missing_endpgm_is_reported(self):
        path = self.write("""
            v_mov_b32 v0, 1
        """)
        with self.assertRaises(cpu_ref.ProgramError):
            cpu_ref.run(path)

    def test_missing_header_is_reported(self):
        path = os.path.join(self.tmp.name, "raw.s")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\tv_mov_b32 v0, 1\n\ts_endpgm\n")
        with self.assertRaises(cpu_ref.ProgramError):
            cpu_ref.run(path)

    def test_unknown_input_is_rejected(self):
        path = self.write("""
            s_endpgm
        """)
        with self.assertRaises(ValueError):
            cpu_ref.run(path, {"sgprs": {0: 1}})

    def test_bit_and_vop3_integer_ops(self):
        r = self.execute("""
            v_mov_b32 v0, 0x00ff0001
            v_not_b32 v1, v0
            v_bfrev_b32 v2, 0x12345678
            v_ffbl_b32 v3, 0x00010100
            v_ffbh_u32 v4, 0x00010000
            v_ffbh_i32 v5, 0xfffffff0
            v_mov_b32 v15, 0x00ff00ff
            v_xnor_b32 v6, 0x0f0f0f0f, v15
            v_bcnt_u32_b32 v7, 0xf0f0f0f0, 3
            v_add3_u32 v8, 10, 20, 30
            v_lshl_add_u32 v9, 3, 2, 5
            v_add_lshl_u32 v10, 3, 5, 2
            v_lshl_or_b32 v11, 3, 4, 1
            v_and_or_b32 v12, 0xf0, 0x3c, 1
            v_or3_b32 v13, 1, 2, 4
            v_xad_u32 v14, 0x5, 0x3, 10
            ; Edge cases: zero/all-ones, positive ffbh_i32, shift > 31, 32-bit overflow, overlapping or3/lshl_or
            v_ffbl_b32 v16, 0
            v_ffbh_u32 v17, 0
            v_ffbh_i32 v18, 0
            v_ffbh_i32 v19, -1
            v_ffbh_i32 v20, 0x00100000
            v_lshl_add_u32 v21, 3, 33, 5
            v_add_lshl_u32 v22, 3, 5, 34
            v_lshl_or_b32 v23, 3, 33, 2
            v_add3_u32 v24, 0xffffffff, 2, 3
            v_bcnt_u32_b32 v25, 0xffffffff, 0xfffffffe
            v_or3_b32 v26, 3, 6, 12
            s_endpgm
        """)
        self.assertEqual(self.vgpr(r, 1), 0xFF00FFFE)
        self.assertEqual(self.vgpr(r, 2), 0x1E6A2C48)
        self.assertEqual(self.vgpr(r, 3), 8)
        self.assertEqual(self.vgpr(r, 4), 15)
        self.assertEqual(self.vgpr(r, 5), 28)
        self.assertEqual(self.vgpr(r, 6), 0xF00FF00F)
        self.assertEqual(self.vgpr(r, 7), 19)
        self.assertEqual(self.vgpr(r, 8), 60)
        self.assertEqual(self.vgpr(r, 9), 17)
        self.assertEqual(self.vgpr(r, 10), 32)
        self.assertEqual(self.vgpr(r, 11), 49)
        self.assertEqual(self.vgpr(r, 12), 0x31)
        self.assertEqual(self.vgpr(r, 13), 7)
        self.assertEqual(self.vgpr(r, 14), 16)
        self.assertEqual(self.vgpr(r, 16), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 17), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 18), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 19), 0xFFFFFFFF)
        self.assertEqual(self.vgpr(r, 20), 11)
        self.assertEqual(self.vgpr(r, 21), 11)   # (3 << (33 & 31)) + 5
        self.assertEqual(self.vgpr(r, 22), 32)   # 8 << (34 & 31)
        self.assertEqual(self.vgpr(r, 23), 6)    # (3 << 1) | 2 (overlapping bit 1, differs from +)
        self.assertEqual(self.vgpr(r, 24), 4)    # wrapped 32-bit sum
        self.assertEqual(self.vgpr(r, 25), 30)   # 32 + 0xfffffffe mod 2^32 (differs from 32 ^ 0xfffffffe)
        self.assertEqual(self.vgpr(r, 26), 15)   # overlapping 3 | 6 | 12 (differs from 3 ^ 6 ^ 12)

    def test_malformed_register_assignments_are_cli_errors(self):
        path = self.write("s_endpgm")
        for option, assignment, flags in (
            ("--sgpr", "3", []),
            ("--vgpr", "3", []),
            ("--sgpr", "x=0", []),
            ("--sgpr", "3=bad", []),
            ("--vgpr", "3=0,1", []),
            ("--sgpr", "3", ["--parse-only"]),
        ):
            with self.subTest(option=option, assignment=assignment, flags=flags):
                result = subprocess.run([sys.executable, "-I", cpu_ref.__file__, path, option, assignment, *flags],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(len(result.stderr.splitlines()), 1)
                self.assertIn(option, result.stderr)
                self.assertIn("N=VALUE", result.stderr)

    def test_out_of_range_lane_is_a_cli_error(self):
        path = self.write("s_endpgm")
        for lane in ("100", "-1", "64"):
            with self.subTest(lane=lane):
                result = subprocess.run([sys.executable, "-I", cpu_ref.__file__, path, "--lane", lane],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--lane", result.stderr)
                self.assertIn("0..63", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
