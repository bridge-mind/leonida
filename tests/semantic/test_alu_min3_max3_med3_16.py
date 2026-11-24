"""Semantic interpreter tests for RDNA2 16-bit three-operand min/max/median instructions:
v_min3_i16, v_min3_u16, v_max3_i16, v_max3_u16, v_med3_i16, v_med3_u16,
v_min3_f16, and v_max3_f16.

Each test translates a synthetic pixel shader listing with tools/rdna2spirv.py, executes it in
tests/semantic/spirv_mini_interp.py, and compares the exported mrt0 words against plain-Python references.

Covers:
- A multi-instruction hand-written .asm fixture exercising all eight opcodes into low and high VGPR halves.
- Signed vs unsigned 16-bit integer comparisons across boundary values (0x8000, 0xFFFF, 0x7FFF, 0x0000, 0x0001)
  in all 3! = 6 operand permutations.
- All 16 op_sel:[s0,s1,s2,dst] bit combinations for every opcode.
- Unwritten destination-half preservation and destination/source register aliasing.
- Inline integer/float constants and SGPR operands.
- Floating-point 16-bit NaN skipping (NMin/NMax), signed-zero ordering (-0.0 < +0.0), neg/abs source modifiers,
  and VOP3 output modifiers (mul:2, mul:4, div:2, clamp).
- Unsupported rejections for clamp on integer min3/max3/med3 and non-VGPR op_sel destination.
- Seeded random sweep over all eight opcodes.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xA55A5AA5

INT_OPS = (
    "v_min3_i16",
    "v_min3_u16",
    "v_max3_i16",
    "v_max3_u16",
    "v_med3_i16",
    "v_med3_u16",
)
FLOAT_OPS = ("v_min3_f16", "v_max3_f16", "v_med3_f16")
ALL_OPS = INT_OPS + FLOAT_OPS

# Hand-written multi-instruction RDNA2 pixel shader fixture covering all 8 opcodes into lo/hi VGPR halves.
# Register inputs:
#   v10 = 0x80007FFF -> lo = 0x7FFF (+32767 i16 / 32767 u16), hi = 0x8000 (-32768 i16 / 32768 u16)
#   v11 = 0xFFFF0001 -> lo = 0x0001 (+1),                     hi = 0xFFFF (-1 i16 / 65535 u16)
#   v12 = 0x0005FFFE -> lo = 0xFFFE (-2 i16 / 65534 u16),     hi = 0x0005 (+5)
#   v13 = 0x3C00BC00 -> lo = 0xBC00 (-1.0 f16),               hi = 0x3C00 (+1.0 f16)
#   v14 = 0x80000000 -> lo = 0x0000 (+0.0 f16),               hi = 0x8000 (-0.0 f16)
#   v15 = 0x7E004000 -> lo = 0x4000 (+2.0 f16),               hi = 0x7E00 (qNaN f16)
# Expected mrt0 words:
#   v0: v_min3_i16(32767, -1, -2) = -2 (0xFFFE) | v_max3_i16(-32768, 1, 5) = 5 (0x0005)       -> 0x0005FFFE
#   v1: v_med3_i16(32767, -1, 5)  = 5  (0x0005) | v_min3_u16(32768, 65535, 65534) = 0x8000    -> 0x80000005
#   v2: v_max3_u16(32767, 65535, 65534) = 0xFFFF | v_med3_u16(32768, 1, 65534) = 0x8000       -> 0x8000FFFF
#   v3: v_min3_f16(+1.0, -0.0, qNaN) = -0.0 (0x8000) | v_max3_f16(-1.0, +0.0, +2.0) = 0x4000  -> 0x40008000
PS_MIN3_MAX3_MED3_16_ASM = """\
; test_alu_min3_max3_med3_16 fixture
\tv_mov_b32 v0, 0xdeadbef0
\tv_mov_b32 v1, 0xdeadbef1
\tv_mov_b32 v2, 0xdeadbef2
\tv_mov_b32 v3, 0xdeadbef3
\tv_mov_b32 v10, 0x80007fff
\tv_mov_b32 v11, 0xffff0001
\tv_mov_b32 v12, 0x0005fffe
\tv_mov_b32 v13, 0x3c00bc00
\tv_mov_b32 v14, 0x80000000
\tv_mov_b32 v15, 0x7e004000
\tv_min3_i16 v0, v10, v11, v12 op_sel:[0,1,0,0]
\tv_max3_i16 v0, v10, v11, v12 op_sel:[1,0,1,1]
\tv_med3_i16 v1, v10, v11, v12 op_sel:[0,1,1,0]
\tv_min3_u16 v1, v10, v11, v12 op_sel:[1,1,0,1]
\tv_max3_u16 v2, v10, v11, v12 op_sel:[0,1,0,0]
\tv_med3_u16 v2, v10, v11, v12 op_sel:[1,0,0,1]
\tv_min3_f16 v3, v13, v14, v15 op_sel:[1,1,1,0]
\tv_max3_f16 v3, v13, v14, v15 op_sel:[0,0,0,1]
\texp mrt0 v0, v1, v2, v3 done vm
\ts_endpgm
"""
PS_MIN3_MAX3_MED3_16_EXPECTED = (0x0005FFFE, 0x80000005, 0x8000FFFF, 0x40008000)

INT_TRIPLES = (
    (0x0005, 0x8000, 0x0007),
    (0xFFFF, 0x0000, 0x0001),
    (0x7FFF, 0x8000, 0xFFFF),
    (0x8001, 0x8000, 0x7FFE),
    (0x0001, 0xFFFE, 0xFFFF),
    (0x0009, 0x0009, 0x8000),
    (0xFFFF, 0xFFFF, 0x0000),
    (0x0003, 0x0003, 0x0003),
)

# f16 bit patterns: -inf, -2.0, -1.0, -0.5, smallest neg subnormal, -0.0, +0.0, subnormal, +0.5, +1.0, +2.0, +inf
F16_VALS = (
    0xFC00,  # -inf
    0xC000,  # -2.0
    0xBC00,  # -1.0
    0xB800,  # -0.5
    0x8001,  # -2^-24 (negative subnormal)
    0x8000,  # -0.0
    0x0000,  # +0.0
    0x0001,  # +2^-24 (positive subnormal)
    0x3800,  # +0.5
    0x3C00,  # +1.0
    0x4000,  # +2.0
    0x7C00,  # +inf
)


def s16(h: int) -> int:
    h &= MASK16
    return h - 0x10000 if h & 0x8000 else h


def f16_to_float(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def half_bits(word: int, hi: int) -> int:
    return ((word >> 16) if hi else word) & MASK16


def merge_half(res16: int, dst_hi: int, old: int = OLD) -> int:
    res16 &= MASK16
    return ((res16 << 16) | (old & MASK16)) if dst_hi else ((old & 0xFFFF0000) | res16)


def ref_int16(op: str, a: int, b: int, c: int) -> int:
    key = s16 if op.endswith("_i16") else (lambda v: v & MASK16)
    xs = [a & MASK16, b & MASK16, c & MASK16]
    if "min3" in op:
        return min(xs, key=key) & MASK16
    if "max3" in op:
        return max(xs, key=key) & MASK16
    if "med3" in op:
        return sorted(xs, key=key)[1] & MASK16
    raise ValueError(op)


def ref_f16(op: str, a_bits: int, b_bits: int, c_bits: int, mods: str = "") -> int:
    vals = [f16_to_float(h) for h in (a_bits, b_bits, c_bits)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00
    if "med3" in op:
        if len(non_nan) < 3:
            # With any NaN operand, RDNA2 hardware returns min3(x, y, z)
            chosen = min(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
        else:
            chosen = sorted(non_nan, key=lambda v: (v, math.copysign(1.0, v)))[1]
    elif "min3" in op:
        chosen = min(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    elif "max3" in op:
        chosen = max(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    else:
        raise ValueError(op)
    if "mul:2" in mods:
        chosen *= 2.0
    elif "mul:4" in mods:
        chosen *= 4.0
    elif "div:2" in mods:
        chosen *= 0.5
    if "clamp" in mods:
        if math.isnan(chosen):
            chosen = 0.0
        else:
            chosen = max(0.0, min(1.0, chosen))
            if chosen == 0.0:
                chosen = 0.0
    return float_to_f16_bits(chosen)


def ref_word(op: str, w1: int, w2: int, w3: int, sel=(0, 0, 0), dst_hi: int = 0, old: int = OLD, mods: str = "") -> int:
    h1, h2, h3 = half_bits(w1, sel[0]), half_bits(w2, sel[1]), half_bits(w3, sel[2])
    r16 = ref_f16(op, h1, h2, h3, mods=mods) if op in FLOAT_OPS else ref_int16(op, h1, h2, h3)
    return merge_half(r16, dst_hi, old=old)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_min3_16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_min3_16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_asm(asm: str) -> str:
    if _TMP_DIR is None:
        setUpModule()
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_insn(
    op: str,
    operands_words: tuple[int | str, int | str, int | str],
    *,
    inline: tuple[int, ...] = (),
    sgpr_idx: tuple[int, ...] = (),
    src_mods: dict[int, str] | None = None,
    tail_mods: str = "",
    old: int = OLD,
    dst: str = "v0",
    alias_dst: int | None = None,
) -> int:
    lines = [f"\tv_mov_b32 v0, {old & MASK32:#x}"]
    op_strs = []
    sgpr_vals: dict[int, int] = {}
    for i, val in enumerate(operands_words):
        fmt = (src_mods or {}).get(i, "{}")
        if i in inline:
            val_str = str(val) if isinstance(val, str) else (f"{val:#x}" if isinstance(val, int) and val >= 0 else str(val))
            op_strs.append(fmt.format(val_str))
        elif i in sgpr_idx:
            reg = 4 + i
            sgpr_vals[reg] = int(val) & MASK32
            lines.append(f"\ts_mov_b32 s{reg}, {sgpr_vals[reg]:#x}")
            op_strs.append(fmt.format(f"s{reg}"))
        elif alias_dst == i:
            lines[0] = f"\tv_mov_b32 v0, {int(val) & MASK32:#x}"
            op_strs.append(fmt.format("v0"))
        else:
            vreg = f"v{i + 1}"
            lines.append(f"\tv_mov_b32 {vreg}, {int(val) & MASK32:#x}")
            op_strs.append(fmt.format(vreg))
    mod_suffix = f" {tail_mods}" if tail_mods else ""
    lines.append(f"\t{op} {dst}, {', '.join(op_strs)}{mod_suffix}")
    asm = "; test_alu_min3_max3_med3_16\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    spv = translate_asm(asm)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    """Cross-check plain-Python reference calculations against hand-worked edge cases."""

    def test_int_reference(self):
        self.assertEqual(ref_int16("v_min3_i16", 0x7FFF, 0xFFFF, 0xFFFE), 0xFFFE)
        self.assertEqual(ref_int16("v_min3_u16", 0x7FFF, 0xFFFF, 0xFFFE), 0x7FFF)
        self.assertEqual(ref_int16("v_max3_i16", 0x8000, 0x0001, 0x0005), 0x0005)
        self.assertEqual(ref_int16("v_max3_u16", 0x8000, 0x0001, 0x0005), 0x8000)
        self.assertEqual(ref_int16("v_med3_i16", 0x7FFF, 0xFFFF, 0x0005), 0x0005)
        self.assertEqual(ref_int16("v_med3_u16", 0x8000, 0x0001, 0xFFFE), 0x8000)

    def test_f16_reference(self):
        self.assertEqual(ref_f16("v_min3_f16", 0x3C00, 0x8000, 0x7E00), 0x8000)
        self.assertEqual(ref_f16("v_max3_f16", 0xBC00, 0x0000, 0x8000), 0x0000)
        self.assertEqual(ref_f16("v_min3_f16", 0x0000, 0x8000, 0x3C00), 0x8000)
        self.assertEqual(ref_f16("v_med3_f16", 0x3C00, 0x8000, 0xBC00), 0x8000)
        self.assertEqual(ref_f16("v_med3_f16", 0x3C00, 0x7E00, 0xBC00), 0xBC00)
        self.assertEqual(ref_f16("v_med3_f16", 0x3C00, 0x7E00, 0xBC00, mods="clamp"), 0x0000)


class FixtureTests(unittest.TestCase):
    """Hand-written multi-instruction RDNA2 .asm fixture for all 8 instructions."""

    def test_ps_min3_max3_med3_16_fixture(self):
        spv = translate_asm(PS_MIN3_MAX3_MED3_16_ASM)
        got = si.run_fragment(spv)["%out_mrt0"]
        self.assertEqual(got, PS_MIN3_MAX3_MED3_16_EXPECTED)


class Int16PermutationsTests(unittest.TestCase):
    """Verify signed vs unsigned ordering across all 3! = 6 operand permutations."""

    def test_boundary_permutations(self):
        for op in INT_OPS:
            for triple in INT_TRIPLES:
                for perm in sorted(set(itertools.permutations(triple))):
                    w1, w2, w3 = (0xBEEF0000 | x for x in perm)
                    with self.subTest(op=op, perm=[hex(x) for x in perm]):
                        self.assertEqual(run_insn(op, (w1, w2, w3)), ref_word(op, w1, w2, w3))


class OpselAllPatternsTests(unittest.TestCase):
    """Test all 16 op_sel:[s0,s1,s2,dst] bit combinations for every opcode."""

    def test_all_sixteen_opsel_patterns_int(self):
        # Distinct lo/hi values in every register, mixing signed-negative and unsigned-high patterns.
        w1, w2, w3 = 0x80007FFF, 0xFFFF0001, 0x0005FFFE
        for op in INT_OPS:
            for bits in itertools.product((0, 1), repeat=4):
                mods = f"op_sel:[{bits[0]},{bits[1]},{bits[2]},{bits[3]}]"
                with self.subTest(op=op, mods=mods):
                    got = run_insn(op, (w1, w2, w3), tail_mods=mods)
                    want = ref_word(op, w1, w2, w3, bits[:3], bits[3])
                    self.assertEqual(got, want)

    def test_all_sixteen_opsel_patterns_f16(self):
        # v1 = (-1.0 | 4.0), v2 = (0.5 | 2.0), v3 = (1.5 | -3.0)
        w1 = float_to_f16_bits(4.0) << 16 | float_to_f16_bits(-1.0)
        w2 = float_to_f16_bits(2.0) << 16 | float_to_f16_bits(0.5)
        w3 = float_to_f16_bits(-3.0) << 16 | float_to_f16_bits(1.5)
        for op in FLOAT_OPS:
            for bits in itertools.product((0, 1), repeat=4):
                mods = f"op_sel:[{bits[0]},{bits[1]},{bits[2]},{bits[3]}]"
                with self.subTest(op=op, mods=mods):
                    got = run_insn(op, (w1, w2, w3), tail_mods=mods)
                    want = ref_word(op, w1, w2, w3, bits[:3], bits[3])
                    self.assertEqual(got, want)


class HalfPreservationAndAliasingTests(unittest.TestCase):
    """Unwritten half of dst is preserved, including when dst aliases src0, src1, or src2."""

    def test_preserves_unwritten_half(self):
        for op in ALL_OPS:
            w1, w2, w3 = 0x3C003800, 0x4000B800, 0xBC000001
            for dst_hi in (0, 1):
                for sentinel in (0x12345678, 0xFEDCBA98):
                    with self.subTest(op=op, dst_hi=dst_hi, sentinel=hex(sentinel)):
                        got = run_insn(op, (w1, w2, w3), tail_mods=f"op_sel:[0,0,0,{dst_hi}]", old=sentinel)
                        self.assertEqual(got, ref_word(op, w1, w2, w3, (0, 0, 0), dst_hi, old=sentinel))

    def test_dst_source_aliasing(self):
        for op in ALL_OPS:
            w = [0x80003C00, 0x4000FFFF, 0x38000002]
            for alias_idx in (0, 1, 2):
                for dst_hi in (0, 1):
                    mods = f"op_sel:[1,0,1,{dst_hi}]"
                    with self.subTest(op=op, alias_idx=alias_idx, dst_hi=dst_hi):
                        got = run_insn(
                            op,
                            (w[0], w[1], w[2]),
                            tail_mods=mods,
                            alias_dst=alias_idx,
                        )
                        want = ref_word(op, w[0], w[1], w[2], (1, 0, 1), dst_hi, old=w[alias_idx])
                        self.assertEqual(got, want)


class ConstantsAndSgprTests(unittest.TestCase):
    """Inline constants (imm / fimm) and SGPR sources."""

    def test_int_inline_constants_and_sgpr(self):
        # -16 inline is 0xFFFFFFF0 (lo = 0xFFF0 = -16 i16 / 65520 u16, hi = 0xFFFF = -1 i16 / 65535 u16).
        cases = [
            ("v_min3_i16", (-16, 64, 0x00030005), (0, 1), (), "op_sel:[0,0,0,0]"),
            ("v_min3_u16", (-16, 64, 0x00030005), (0, 1), (), "op_sel:[0,0,0,1]"),
            ("v_max3_i16", (-1, 0x80000002, 10), (0, 2), (1,), "op_sel:[0,1,0,0]"),
            ("v_max3_u16", (-16, 0x00010002, 0x80000003), (0,), (1,), "op_sel:[1,0,1,1]"),
            ("v_med3_i16", (0x7FFF8000, -1, 16), (1, 2), (0,), "op_sel:[0,0,0,0]"),
            ("v_med3_u16", (0x7FFF8000, -16, 64), (1, 2), (0,), "op_sel:[1,0,0,1]"),
        ]
        for op, words, inl, sgprs, mods in cases:
            with self.subTest(op=op, words=words, mods=mods):
                sel_parts = [int(x) for x in mods.split("[")[1].rstrip("]").split(",")]
                norm_words = [
                    (w & MASK32) if isinstance(w, int) else w
                    for w in words
                ]
                got = run_insn(op, words, inline=inl, sgpr_idx=sgprs, tail_mods=mods)
                want = ref_word(op, norm_words[0], norm_words[1], norm_words[2], sel_parts[:3], sel_parts[3])
                self.assertEqual(got, want)

    def test_f16_fimm_constants_and_sgpr(self):
        # Inline float constants populate the low half as f16 with high half = +0.0 (docs/design.md).
        # 1.0 -> 0x00003C00; -2.0 -> 0x0000C000; 0.5 -> 0x00003800
        got_min = run_insn(
            "v_min3_f16",
            ("1.0", 0x40003C00, "-2.0"),
            inline=(0, 2),
            sgpr_idx=(1,),
            tail_mods="op_sel:[0,1,0,0]",
        )
        self.assertEqual(got_min, merge_half(0xC000, 0))

        # Selecting the high half of an inline float constant reads +0.0.
        got_hi_fimm = run_insn(
            "v_max3_f16",
            ("1.0", 0xBC00BC00, "-0.5"),
            inline=(0, 2),
            tail_mods="op_sel:[1,0,0,1]",
        )
        self.assertEqual(got_hi_fimm, merge_half(0x0000, 1))


class Float16SpecialsAndModsTests(unittest.TestCase):
    """f16 NaNs, signed zeros (-0.0 < +0.0), neg/abs, omod, and clamp."""

    def test_signed_zeros_all_orders(self):
        # -0.0 (0x8000) orders strictly below +0.0 (0x0000) regardless of operand position.
        for triple in (
            (0x8000, 0x0000, 0x3C00),   # -0, +0, +1 -> min=-0, max=+1
            (0x8000, 0x0000, 0xBC00),   # -0, +0, -1 -> min=-1, max=+0
            (0x8000, 0x0000, 0x0000),   # -0, +0, +0 -> min=-0, max=+0
            (0x8000, 0x8000, 0x0000),   # -0, -0, +0 -> min=-0, max=+0
        ):
            for perm in sorted(set(itertools.permutations(triple))):
                for op in FLOAT_OPS:
                    with self.subTest(op=op, perm=[hex(x) for x in perm]):
                        got = run_insn(op, perm)
                        self.assertEqual(got, ref_word(op, *perm))

    def test_nan_operands(self):
        for op in FLOAT_OPS:
            # One quiet or signaling NaN: returns min/max of the remaining finite values.
            for nan_bits in (0x7E00, 0x7C01, 0xFE01):
                for perm in sorted(set(itertools.permutations((nan_bits, 0x4000, 0xBC00)))):
                    with self.subTest(op=op, perm=[hex(x) for x in perm]):
                        got = run_insn(op, perm)
                        self.assertEqual(got, ref_word(op, *perm))
            # Two NaNs: returns the single non-NaN operand.
            for perm in sorted(set(itertools.permutations((0x7E00, 0xFE01, 0x3800)))):
                with self.subTest(op=op, two_nans=[hex(x) for x in perm]):
                    got = run_insn(op, perm)
                    self.assertEqual(got, ref_word(op, *perm))
            # Three NaNs: result is NaN and unwritten high half is preserved.
            got_all_nan = run_insn(op, (0x7E00, 0x7E01, 0xFE01))
            self.assertEqual(got_all_nan >> 16, OLD >> 16)
            self.assertTrue(math.isnan(f16_to_float(got_all_nan & MASK16)))

    def test_source_and_output_modifiers(self):
        # Input neg/abs apply to the selected half; on RDNA2 omod is ignored for f16 with denorms kept; clamp applies [0, 1].
        w1 = float_to_f16_bits(-3.0) << 16 | float_to_f16_bits(0.5)
        w2 = float_to_f16_bits(0.25) << 16 | float_to_f16_bits(-2.0)
        w3 = float_to_f16_bits(-0.125) << 16 | float_to_f16_bits(4.0)

        # neg(|v1.hi|) = -3.0, |v2.lo| = +2.0, v3.hi = -0.125 -> min = -3.0 (div:2 ignored on RDNA2)
        got_div2 = run_insn(
            "v_min3_f16",
            (w1, w2, w3),
            src_mods={0: "-|{}|", 1: "|{}|"},
            tail_mods="op_sel:[1,0,1,0] div:2",
        )
        self.assertEqual(got_div2, merge_half(float_to_f16_bits(-3.0), 0))

        # min(|v1.lo|=0.5, v2.hi=0.25, -v3.hi=0.125) = 0.125 (mul:4 ignored on RDNA2, clamp keeps 0.125) (hi half)
        got_mul4_clamp = run_insn(
            "v_min3_f16",
            (w1, w2, w3),
            src_mods={0: "|{}|", 2: "-{}"},
            tail_mods="op_sel:[0,1,1,1] mul:4 clamp",
        )
        self.assertEqual(got_mul4_clamp, merge_half(float_to_f16_bits(0.125), 1))

        # max(v1.lo=0.5, v2.hi=0.25, v3.lo=4.0) = 4.0 -> div:2 ignored on RDNA2 -> clamp = 1.0
        got_max_clamp = run_insn(
            "v_max3_f16",
            (w1, w2, w3),
            tail_mods="op_sel:[0,1,0,0] div:2 clamp",
        )
        self.assertEqual(got_max_clamp, merge_half(float_to_f16_bits(1.0), 0))

        # max(v1.hi=-3.0, v2.lo=-2.0, v3.hi=-0.125) = -0.125 -> mul:2 = -0.25 -> clamp = +0.0
        got_neg_clamp = run_insn(
            "v_max3_f16",
            (w1, w2, w3),
            tail_mods="op_sel:[1,0,1,1] mul:2 clamp",
        )
        self.assertEqual(got_neg_clamp, merge_half(0x0000, 1))


class ClampSupportTests(unittest.TestCase):
    """Verify that integer and float min3/max3/med3 accept VOP3 clamp."""

    def test_int_clamp_supported(self):
        for op in INT_OPS:
            with self.subTest(op=op):
                w1, w2, w3 = (0x0005, 0x0010, 0xFFFF)
                got_noclamp = run_insn(op, (w1, w2, w3))
                got_clamp = run_insn(op, (w1, w2, w3), tail_mods="clamp")
                self.assertEqual(got_noclamp, got_clamp)
                got_dhi_clamp = run_insn(op, (w1, w2, w3), tail_mods="op_sel:[0,0,0,1] clamp")
                self.assertEqual(got_dhi_clamp, merge_half(got_clamp & MASK16, 1))

    def test_int_clamp_dirty_sentinels(self):
        # Verify clamp range-invariance and destination half preservation with dirty sentinel values.
        for op in INT_OPS:
            with self.subTest(op=op):
                w1, w2, w3 = (0xDEAD0005, 0xBEEF0010, 0x5A5AFFFF)
                # Low half destination preserves 0xDEAD in high half
                got_lo = run_insn(op, (w1, w2, w3), tail_mods="clamp", old=0xDEAD3800)
                ref_lo = ref_word(op, w1, w2, w3, (0, 0, 0), dst_hi=0, old=0xDEAD3800)
                self.assertEqual(got_lo, ref_lo)
                self.assertEqual(got_lo >> 16, 0xDEAD)

                # High half destination preserves 0x3800 in low half
                got_hi = run_insn(op, (w1, w2, w3), tail_mods="op_sel:[0,0,0,1] clamp", old=0xDEAD3800)
                ref_hi = ref_word(op, w1, w2, w3, (0, 0, 0), dst_hi=1, old=0xDEAD3800)
                self.assertEqual(got_hi, ref_hi)
                self.assertEqual(got_hi & MASK16, 0x3800)

    def test_f16_clamp_all_ops(self):
        # In-range [0.0, 1.0], upper saturation (> 1.0 -> 1.0), lower saturation (< 0.0 -> +0.0)
        # across all three f16 opcodes.
        for op in FLOAT_OPS:
            with self.subTest(op=op):
                # in-range: 0.25, 0.5, 0.75
                w_mid = (float_to_f16_bits(0.25), float_to_f16_bits(0.5), float_to_f16_bits(0.75))
                got_mid = run_insn(op, w_mid, tail_mods="clamp")
                self.assertEqual(got_mid, ref_word(op, *w_mid, mods="clamp"))

                # upper saturation: 1.5, 2.0, 3.0
                w_hi = (float_to_f16_bits(1.5), float_to_f16_bits(2.0), float_to_f16_bits(3.0))
                got_hi = run_insn(op, w_hi, tail_mods="clamp")
                self.assertEqual(got_hi, ref_word(op, *w_hi, mods="clamp"))
                self.assertEqual(got_hi & MASK16, float_to_f16_bits(1.0))

                # lower saturation: -2.0, -1.0, -0.5
                w_lo = (float_to_f16_bits(-2.0), float_to_f16_bits(-1.0), float_to_f16_bits(-0.5))
                got_lo = run_insn(op, w_lo, tail_mods="clamp")
                self.assertEqual(got_lo, ref_word(op, *w_lo, mods="clamp"))
                self.assertEqual(got_lo & MASK16, 0x0000)

                # negative zero clamps to +0.0
                w_neg_zero = (0x8000, 0x8000, 0x8000)
                got_nz = run_insn(op, w_neg_zero, tail_mods="clamp")
                self.assertEqual(got_nz & MASK16, 0x0000)

    def test_f16_clamp_nan_and_sentinels(self):
        # NaN operands with DX10 clamp and dirty sentinel destination halves
        w_nan = (0x7E00, float_to_f16_bits(2.0), float_to_f16_bits(-1.0))
        # v_min3_f16: min(2.0, -1.0) = -1.0 -> clamp = +0.0
        got_min = run_insn("v_min3_f16", w_nan, tail_mods="clamp", old=0xDEAD3800)
        self.assertEqual(got_min, 0xDEAD0000)

        # v_max3_f16: max(2.0, -1.0) = 2.0 -> clamp = 1.0 (0x3C00)
        got_max = run_insn("v_max3_f16", w_nan, tail_mods="clamp", old=0xDEAD3800)
        self.assertEqual(got_max, 0xDEAD3C00)

        # v_med3_f16: NaN fallback to min3 = -1.0 -> clamp = +0.0
        got_med = run_insn("v_med3_f16", w_nan, tail_mods="clamp", old=0xDEAD3800)
        self.assertEqual(got_med, 0xDEAD0000)

        # High half write with op_sel:[1,1,1,1] clamp leaves 0xBEEF in low half
        w1_packed = float_to_f16_bits(0.5) << 16 | 0x7E00
        w2_packed = float_to_f16_bits(-2.0) << 16 | 0x4000
        w3_packed = float_to_f16_bits(0.5) << 16 | 0xBC00
        # median of high halves (0.5, -2.0, 0.5) is 0.5, clamped to 0.5 (0x3800)
        got_med_hi = run_insn(
            "v_med3_f16",
            (w1_packed, w2_packed, w3_packed),
            tail_mods="op_sel:[1,1,1,1] clamp",
            old=0x1234BEEF,
        )
        self.assertEqual(got_med_hi, (float_to_f16_bits(0.5) << 16) | 0xBEEF)

class UnsupportedTests(unittest.TestCase):
    """Reject non-VGPR op_sel destinations."""

    def test_non_vgpr_opsel_dst_rejected(self):
        for op in ALL_OPS:
            with self.subTest(op=op):
                asm = (
                    "; test_unsupported_dst\n"
                    "\tv_mov_b32 v1, 1\n\tv_mov_b32 v2, 2\n\tv_mov_b32 v3, 3\n"
                    f"\t{op} s0, v1, v2, v3 op_sel:[0,0,0,1]\n"
                    "\texp mrt0 v1, v1, v1, v1 done vm\n\ts_endpgm\n"
                )
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_asm(asm)


class RandomSweepTests(unittest.TestCase):
    """Seeded random sweep over all eight opcodes, halves, and destination halves."""

    def test_seeded_sweep(self):
        rng = random.Random(0x1633)
        int_pool = [0, 1, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF]
        for op in INT_OPS:
            for _ in range(18):
                w1, w2, w3 = (
                    ((rng.choice(int_pool) if rng.random() < 0.4 else rng.randrange(0x10000)) << 16)
                    | (rng.choice(int_pool) if rng.random() < 0.4 else rng.randrange(0x10000))
                    for _ in range(3)
                )
                sel = tuple(rng.randrange(2) for _ in range(3))
                dst_hi = rng.randrange(2)
                mods = f"op_sel:[{sel[0]},{sel[1]},{sel[2]},{dst_hi}]"
                with self.subTest(op=op, w=(hex(w1), hex(w2), hex(w3)), mods=mods):
                    self.assertEqual(run_insn(op, (w1, w2, w3), tail_mods=mods), ref_word(op, w1, w2, w3, sel, dst_hi))

        for op in FLOAT_OPS:
            for _ in range(18):
                w1, w2, w3 = ((rng.choice(F16_VALS) << 16) | rng.choice(F16_VALS) for _ in range(3))
                sel = tuple(rng.randrange(2) for _ in range(3))
                dst_hi = rng.randrange(2)
                mods = f"op_sel:[{sel[0]},{sel[1]},{sel[2]},{dst_hi}]"
                with self.subTest(op=op, w=(hex(w1), hex(w2), hex(w3)), mods=mods):
                    self.assertEqual(run_insn(op, (w1, w2, w3), tail_mods=mods), ref_word(op, w1, w2, w3, sel, dst_hi))


if __name__ == "__main__":
    unittest.main()
