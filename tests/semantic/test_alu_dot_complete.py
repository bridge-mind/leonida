"""Semantic interpreter tests for RDNA2 v_dot2c/4c/8c integer forms and v_dot2_f32_f16 / v_dot2c_f32_f16.

Covers the 7 remaining RDNA2 dot-product instructions:
  - v_dot2c_i32_i16, v_dot2c_u32_u16, v_dot4c_u32_u8, v_dot8c_i32_i4, v_dot8c_u32_u4 (VOP2 accumulating)
  - v_dot2_f32_f16 (VOP3P half-to-float dot product with op_sel, op_sel_hi, neg_lo, neg_hi, clamp)
  - v_dot2c_f32_f16 (VOP2 accumulating half-to-float dot product with optional clamp)

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import math
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

INT_DOTC_OPS = (
    "v_dot2c_i32_i16",
    "v_dot2c_u32_u16",
    "v_dot4c_u32_u8",
    "v_dot8c_i32_i4",
    "v_dot8c_u32_u4",
)


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x >> 31 else x


def ref_dot_int(op: str, x: int, y: int, acc: int, clamp: bool = False) -> int:
    width = 16 if ("_dot2_" in op or "_dot2c_" in op) else (8 if "_dot4" in op else 4)
    signed = op.endswith(("i16", "i4"))
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    total = s32(acc) if signed else (acc & MASK32)
    for off in range(0, 32, width):
        xe = (x >> off) & mask
        ye = (y >> off) & mask
        if signed:
            if xe & sign_bit:
                xe -= 1 << width
            if ye & sign_bit:
                ye -= 1 << width
        total += xe * ye
    if clamp:
        if signed:
            total = max(INT32_MIN, min(INT32_MAX, total))
        else:
            total = max(0, min(MASK32, total))
    return total & MASK32


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32(val: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", val))[0]
    except OverflowError:
        return math.inf if val > 0 else -math.inf


def f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def pack_f16x2(lo: float, hi: float) -> int:
    return f16_bits(lo) | (f16_bits(hi) << 16)


def flush_f32_py(bits: int) -> float:
    f = bits_f32(bits)
    if math.isfinite(f) and abs(f) < 2.0 ** -126:
        return math.copysign(0.0, f)
    return f


def ref_dot2_f32_f16(
    s0: int,
    s1: int,
    acc_bits: int,
    sel=(0, 0, 0),
    selhi=(1, 1, 1),
    nlo=(0, 0, 0),
    nhi=(0, 0, 0),
    clamp: bool = False,
) -> int:
    a0 = bits_f16((s0 >> (16 * sel[0])) & 0xFFFF)
    a1 = bits_f16((s0 >> (16 * selhi[0])) & 0xFFFF)
    b0 = bits_f16((s1 >> (16 * sel[1])) & 0xFFFF)
    b1 = bits_f16((s1 >> (16 * selhi[1])) & 0xFFFF)
    if nlo[0]:
        a0 = -a0
    if nhi[0]:
        a1 = -a1
    if nlo[1]:
        b0 = -b0
    if nhi[1]:
        b1 = -b1
    acc = flush_f32_py(acc_bits)
    if nlo[2]:
        acc = -acc
    p0 = f32(a0 * b0)
    p1 = f32(a1 * b1)
    s = f32(p0 + p1)
    r = f32(s + acc)
    if clamp:
        r = max(0.0, min(1.0, r))
    return f32_bits(r)


def run_ps(lines) -> tuple:
    spv = translate_lines("ps_dot_complete", list(lines) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return tuple(mrt0_words(spv))


class IntDotAccumTests(unittest.TestCase):
    def test_v_dotc_values_and_clamp(self):
        cases = [
            # (op, x, y, acc)
            ("v_dot2c_i32_i16", 0xFFFC0003, 0x0005FFFE, 100),
            ("v_dot2c_i32_i16", 0x80008000, 0x80008000, 0),            # 2*(-32768)^2 = 0x80000000 -> clamps to INT32_MAX
            ("v_dot2c_i32_i16", 0x80008000, 0x7FFF7FFF, 0x80000000),   # negative overflow -> INT32_MIN
            ("v_dot2c_i32_i16", 0x80008000, 0x80008000, 0x80000000),   # cancellation edge: 2^31 - 2^31 = 0
            ("v_dot2c_u32_u16", 0x80000001, 0x00010003, 5),
            ("v_dot2c_u32_u16", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF0000),   # unsigned wrap / clamp
            ("v_dot4c_u32_u8", 0x80FF0102, 0x02010304, 10),
            ("v_dot4c_u32_u8", 0xFFFFFFFF, 0x01010101, 0xFFFFFF00),    # unsigned wrap / clamp
            ("v_dot8c_i32_i4", 0x8F123456, 0x12121212, 10),
            ("v_dot8c_i32_i4", 0x88888888, 0x88888888, 0x7FFFFFF0),    # signed overflow -> INT32_MAX
            ("v_dot8c_i32_i4", 0x88888888, 0x77777777, 0x80000010),    # signed underflow -> INT32_MIN
            ("v_dot8c_u32_u4", 0xF8000001, 0x12000003, 4),
            ("v_dot8c_u32_u4", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFF00),    # unsigned wrap / clamp
        ]
        for op, x, y, acc in cases:
            for clamp in (False, True):
                mnem = f"{op}_e64" if clamp else op
                mod = " clamp" if clamp else ""
                with self.subTest(op=mnem, x=hex(x), y=hex(y), acc=hex(acc)):
                    got = run_ps([
                        f"v_mov_b32 v0, {acc:#x}",
                        f"v_mov_b32 v1, {x:#x}",
                        f"v_mov_b32 v2, {y:#x}",
                        "v_mov_b32 v3, 0",
                        f"{mnem} v0, v1, v2{mod}",
                    ])[0]
                    self.assertEqual(got, ref_dot_int(op, x, y, acc, clamp=clamp))

    def test_v_dotc_sgpr_literal_and_seeded(self):
        got = run_ps([
            "s_mov_b32 s4, 0x00020003",
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 0x00040005",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "v_dot2c_i32_i16 v0, s4, v1",
            "v_mov_b32 v2, 7",
            "v_dot4c_u32_u8 v2, 0x01020304, v1",
        ])
        self.assertEqual(got[0], ref_dot_int("v_dot2c_i32_i16", 0x00020003, 0x00040005, 10))
        self.assertEqual(got[2], ref_dot_int("v_dot4c_u32_u8", 0x01020304, 0x00040005, 7))

        rng = random.Random(0xD07C)
        for op in INT_DOTC_OPS:
            for _ in range(5):
                x, y, acc = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
                clamp = bool(rng.getrandbits(1))
                with self.subTest(op=op, clamp=clamp):
                    got_w = run_ps([
                        f"v_mov_b32 v0, {acc:#x}",
                        f"v_mov_b32 v1, {x:#x}",
                        f"v_mov_b32 v2, {y:#x}",
                        "v_mov_b32 v3, 0",
                        f"{op}{'_e64' if clamp else ''} v0, v1, v2{' clamp' if clamp else ''}",
                    ])[0]
                    self.assertEqual(got_w, ref_dot_int(op, x, y, acc, clamp=clamp))

    def test_v_dotc_rejects_unsupported_forms(self):
        bad_lines = [
            "v_dot2c_i32_i16 v0, v1, v2 op_sel:[0,0,0]",
            "v_dot2c_u32_u16 v0, v1, v2 neg_lo:[1,0,0]",
            "v_dot4c_u32_u8_dpp v0, v1, v2 quad_perm:[0,1,2,3]",
            "v_dot8c_i32_i4_sdwa v0, v1, v2 dst_sel:DWORD",
            "v_dot8c_u32_u4_e64 v0, -v1, v2",
            "v_dot2c_i32_i16 s0, v1, v2",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_dotc_bad", [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


class HalfDotF32Tests(unittest.TestCase):
    def test_v_dot2_f32_f16_and_v_dot2c_f32_f16_values(self):
        s0 = pack_f16x2(1.5, -2.0)
        s1 = pack_f16x2(4.0, 0.5)
        acc = f32_bits(0.25)
        got = run_ps([
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {acc:#x}",
            "v_dot2_f32_f16 v0, v1, v2, v3",
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v3, {acc:#x}",
            "v_dot2c_f32_f16 v3, v1, v2",
        ])
        want = ref_dot2_f32_f16(s0, s1, acc)
        self.assertEqual(got[0], want)
        self.assertEqual(got[3], want)

    def test_v_dot2_f32_f16_opsel_neg_and_clamp(self):
        s0 = pack_f16x2(1.5, -2.0)
        s1 = pack_f16x2(4.0, 0.5)
        acc = f32_bits(0.75)
        cases = [
            ("op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), False),
            ("neg_lo:[1,0,0] neg_hi:[0,1,0]", (0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), False),
            ("neg_lo:[0,0,1]", (0, 0, 0), (1, 1, 1), (0, 0, 1), (0, 0, 0), False),
            ("clamp", (0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0), True),
            ("neg_lo:[1,1,1] clamp", (0, 0, 0), (1, 1, 1), (1, 1, 1), (0, 0, 0), True),
        ]
        for mods, sel, selhi, nlo, nhi, clamp in cases:
            with self.subTest(mods=mods):
                got = run_ps([
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {acc:#x}",
                    f"v_dot2_f32_f16 v0, v1, v2, v3 {mods}",
                ])[0]
                self.assertEqual(got, ref_dot2_f32_f16(s0, s1, acc, sel, selhi, nlo, nhi, clamp=clamp))

        # v_dot2c_f32_f16 clamp
        got_c = run_ps([
            f"v_mov_b32 v0, {f32_bits(0.5):#x}",
            f"v_mov_b32 v1, {pack_f16x2(2.0, 1.0):#x}",
            f"v_mov_b32 v2, {pack_f16x2(1.0, 1.0):#x}",
            "v_mov_b32 v3, 0",
            "v_dot2c_f32_f16_e64 v0, v1, v2 clamp",
        ])[0]
        self.assertEqual(got_c, f32_bits(1.0))

    def test_subnormals_inf_and_nan(self):
        # f16 subnormal input (2^-24) * 2.0 + 0.0 -> 2^-23 (exact normal f32, kept)
        s0_sub = 0x0001
        s1_two = pack_f16x2(2.0, 0.0)
        got_sub = run_ps([
            f"v_mov_b32 v1, {s0_sub:#x}",
            f"v_mov_b32 v2, {s1_two:#x}",
            "v_mov_b32 v3, 0",
            "v_dot2_f32_f16 v0, v1, v2, v3",
        ])[0]
        self.assertEqual(got_sub, f32_bits(2.0 ** -23))

        # f32 subnormal acc is flushed to signed zero by flush_f32 before adding
        got_flush = run_ps([
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x80000001",  # negative f32 subnormal
            "v_dot2_f32_f16 v0, v1, v2, v3",
        ])[0]
        self.assertEqual(got_flush, 0x00000000)

        # +inf and NaN propagation
        got_inf = run_ps([
            f"v_mov_b32 v1, {pack_f16x2(math.inf, 1.0):#x}",
            f"v_mov_b32 v2, {pack_f16x2(1.0, 1.0):#x}",
            "v_mov_b32 v3, 0",
            "v_dot2_f32_f16 v0, v1, v2, v3",
        ])[0]
        self.assertEqual(got_inf, f32_bits(math.inf))

        got_nan = run_ps([
            "v_mov_b32 v1, 0x7e01",
            f"v_mov_b32 v2, {pack_f16x2(1.0, 0.0):#x}",
            "v_mov_b32 v3, 0",
            "v_dot2c_f32_f16 v3, v1, v2",
        ])[3]
        self.assertTrue(math.isnan(bits_f32(got_nan)))

    def test_seeded_finite(self):
        rng = random.Random(0xF16F32)
        pool = [-4.0, -1.5, -0.5, -0.0, 0.0, 0.25, 0.5, 1.25, 2.0, 3.5]
        for _ in range(15):
            a0, a1 = rng.choice(pool), rng.choice(pool)
            b0, b1 = rng.choice(pool), rng.choice(pool)
            acc = rng.choice(pool)
            s0, s1, acc_b = pack_f16x2(a0, a1), pack_f16x2(b0, b1), f32_bits(acc)
            want = ref_dot2_f32_f16(s0, s1, acc_b)
            got = run_ps([
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {acc_b:#x}",
                "v_dot2_f32_f16 v0, v1, v2, v3",
            ])[0]
            self.assertEqual(got, want)

    def test_rejects_unsupported_forms(self):
        bad_lines = [
            "v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0,1]",
            "v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[1,1,0]",
            "v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,0,1]",
            "v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0]",
            "v_dot2_f32_f16 v0, v1, v2, v3 mul:2",
            "v_dot2_f32_f16 v0, -v1, v2, v3",
            "v_dot2_f32_f16_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3]",
            "v_dot2_f32_f16 s0, v1, v2, v3",
            "v_dot2c_f32_f16 v0, v1, v2 op_sel:[0,0,0]",
            "v_dot2c_f32_f16 v0, v1, v2 neg_lo:[1,0,0]",
            "v_dot2c_f32_f16_sdwa v0, v1, v2 dst_sel:DWORD",
            "v_dot2c_f32_f16_e64 v0, |v1|, v2",
            "v_dot2c_f32_f16 s0, v1, v2",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_dotf_bad", [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


class RmwPrepassTests(unittest.TestCase):
    def test_accumulating_ops_marked_read_before_write(self):
        for op in (*INT_DOTC_OPS, "v_dot2c_f32_f16"):
            with self.subTest(op=op):
                insns = [shader_ir.parse_line(ln) for ln in (
                    f"{op} v7, v1, v2",
                    "exp mrt0 v7, v7, v7, v7 done vm",
                    "s_endpgm",
                )]
                rbw = rdna2spirv.read_before_write_vgprs(insns)
                rr, ww = rdna2spirv.reg_reads_writes(insns)
                self.assertIn(7, rbw)
                self.assertIn(7, rr)
                self.assertIn(7, ww)


if __name__ == "__main__":
    unittest.main()
