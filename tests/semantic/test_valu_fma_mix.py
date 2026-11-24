"""Semantic interpreter tests for RDNA2 v_fma_mix_f32, v_fma_mixlo_f16 and v_fma_mixhi_f16.

Each VOP3P mixed-precision FMA computes fma(src0, src1, src2) in f32:
- op_sel_hi[i] == 0: src[i] is an f32 word.
- op_sel_hi[i] == 1: src[i] is an f16 packed in the low half (op_sel[i] == 0) or high half (op_sel[i] == 1),
  widened to f32 before applying neg/abs modifiers.
- v_fma_mix_f32 writes the f32 result; v_fma_mixlo_f16 narrows to f16 in the low half (keeping bits [31:16]);
  v_fma_mixhi_f16 narrows to f16 in the high half (keeping bits [15:0]).

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference.
"""
from fractions import Fraction
import itertools
import math
import os
import re
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xCAFE55AA


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def pack_h(lo: float, hi: float) -> int:
    return (f2h(hi) << 16) | f2h(lo)


def unpack_src(word: int, is_f16: int, sel_hi_half: int, neg: bool = False, abs_: bool = False) -> float:
    v = h2f((word >> 16) if sel_hi_half else word) if is_f16 else u2f(word)
    if abs_:
        v = math.copysign(v, 1.0)
    if neg:
        v = -v
    return v


def ref_fma(w0: int, w1: int, w2: int, sel_hi, sel, mods_src=((False, False),) * 3, clamp: bool = False) -> float:
    x = unpack_src(w0, sel_hi[0], sel[0], *mods_src[0])
    y = unpack_src(w1, sel_hi[1], sel[1], *mods_src[1])
    z = unpack_src(w2, sel_hi[2], sel[2], *mods_src[2])
    r = u2f(f2u(float(Fraction(x) * Fraction(y) + Fraction(z))))
    if clamp:
        r = min(max(r, 0.0), 1.0)
    return r


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_fma_mix.s")
    with open(os.path.join(_TMP_DIR.name, "ps_fma_mix.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_and_run(line: str, v1: int, v2: int, v3: int, old: int = OLD):
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"v_mov_b32 v3, {v3:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ]
    asm = (
        "; test_valu_fma_mix\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return spv, si.run_fragment(spv)["%out_mrt0"][0]


def run(line: str, v1: int, v2: int, v3: int, old: int = OLD) -> int:
    return translate_and_run(line, v1, v2, v3, old=old)[1]


class FmaMixF32Tests(unittest.TestCase):
    """v_fma_mix_f32 across op_sel_hi/op_sel combinations, source modifiers and clamp."""

    # Each word is valid both as a finite f32 and as two distinct finite f16 halves.
    W0 = pack_h(1.5, -3.25)    # f16: (1.5, -3.25), also finite f32
    W1 = pack_h(2.25, 0.75)    # f16: (2.25, 0.75)
    W2 = pack_h(-0.5, 4.5)     # f16: (-0.5, 4.5)

    def test_all_f32_default_no_modifiers(self):
        a, b, c = f2u(2.5), f2u(-1.25), f2u(0.75)
        got = run("v_fma_mix_f32 v0, v1, v2, v3", a, b, c)
        want = f2u(ref_fma(a, b, c, (0, 0, 0), (0, 0, 0)))
        self.assertEqual(got, want)

    def test_all_64_op_sel_and_op_sel_hi_combinations(self):
        for sel_hi in itertools.product((0, 1), repeat=3):
            for sel in itertools.product((0, 1), repeat=3):
                # When op_sel_hi[i] == 0, supply a clean f32; when 1, supply packed f16 lo/hi.
                w0 = self.W0 if sel_hi[0] else f2u(1.75)
                w1 = self.W1 if sel_hi[1] else f2u(-2.5)
                w2 = self.W2 if sel_hi[2] else f2u(0.375)
                line = (
                    f"v_fma_mix_f32 v0, v1, v2, v3 "
                    f"op_sel:[{sel[0]},{sel[1]},{sel[2]}] "
                    f"op_sel_hi:[{sel_hi[0]},{sel_hi[1]},{sel_hi[2]}]"
                )
                with self.subTest(sel_hi=sel_hi, sel=sel):
                    got = run(line, w0, w1, w2)
                    want = f2u(ref_fma(w0, w1, w2, sel_hi, sel))
                    self.assertEqual(got, want)

    def test_source_neg_abs_modifiers(self):
        w0 = pack_h(-2.0, 3.0)
        w1 = f2u(-1.5)
        w2 = pack_h(0.25, -4.0)
        line = "v_fma_mix_f32 v0, -|v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1]"
        got = run(line, w0, w1, w2)
        want = f2u(ref_fma(w0, w1, w2, (1, 0, 1), (0, 0, 1), mods_src=((True, True), (True, False), (False, True))))
        self.assertEqual(got, want)

    def test_clamp_modifier(self):
        cases = [
            (pack_h(2.0, 0.0), f2u(3.0), pack_h(1.0, 0.0), 1.0),     # > 1.0 -> 1.0
            (pack_h(-2.0, 0.0), f2u(3.0), pack_h(-1.0, 0.0), 0.0),   # < 0.0 -> 0.0
            (pack_h(0.5, 0.0), f2u(0.5), pack_h(0.25, 0.0), 0.5),    # in [0, 1] -> unchanged
        ]
        for w0, w1, w2, expected in cases:
            with self.subTest(expected=expected):
                got = run("v_fma_mix_f32 v0, v1, v2, v3 op_sel_hi:[1,0,1] clamp", w0, w1, w2)
                self.assertEqual(got, f2u(expected))


class FmaMixF16Tests(unittest.TestCase):
    """v_fma_mixlo_f16 and v_fma_mixhi_f16: half preservation, mixed inputs, modifiers, clamp and aliasing."""

    def test_mixlo_and_mixhi_preserve_untouched_half(self):
        w0 = pack_h(1.5, -2.5)
        w1 = f2u(2.0)
        w2 = pack_h(0.25, 3.0)
        for op, dst_hi in (("v_fma_mixlo_f16", 0), ("v_fma_mixhi_f16", 1)):
            for sel in ((0, 0, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1)):
                line = f"{op} v0, v1, v2, v3 op_sel:[{sel[0]},{sel[1]},{sel[2]}] op_sel_hi:[1,0,1]"
                with self.subTest(op=op, sel=sel):
                    got = run(line, w0, w1, w2, old=OLD)
                    h = f2h(ref_fma(w0, w1, w2, (1, 0, 1), sel))
                    want = ((h << 16) | (OLD & 0xFFFF)) if dst_hi else ((OLD & 0xFFFF0000) | h)
                    self.assertEqual(got, want)

    def test_mixlo_and_mixhi_neg_abs_and_clamp(self):
        w0 = pack_h(-3.0, 1.0)
        w1 = pack_h(2.0, -0.5)
        w2 = f2u(0.5)
        # -|-3.0| * (-0.5) + 0.5 = -3.0 * -0.5 + 0.5 = 2.0 -> clamped to 1.0 (0x3c00)
        got_lo = run("v_fma_mixlo_f16 v0, -|v1|, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] clamp", w0, w1, w2, old=OLD)
        self.assertEqual(got_lo, (OLD & 0xFFFF0000) | f2h(1.0))

        # -|-3.0| * 2.0 + 0.5 = -5.5 -> clamped to 0.0 (0x0000)
        got_hi = run("v_fma_mixhi_f16 v0, -|v1|, v2, v3 op_sel:[0,0,0] op_sel_hi:[1,1,0] clamp", w0, w1, w2, old=OLD)
        self.assertEqual(got_hi, (f2h(0.0) << 16) | (OLD & 0xFFFF))

    def test_destination_aliases_source(self):
        # v0 starts as pack_h(1.5, 4.0); compute mixhi using v0.lo (1.5) * 2.0 + 0.5 = 3.5 -> keep lo (1.5)
        old = pack_h(1.5, 4.0)
        got = run("v_fma_mixhi_f16 v0, v0, v1, v2 op_sel:[0,0,0] op_sel_hi:[1,0,1]", f2u(2.0), pack_h(0.5, 9.0), 0, old=old)
        self.assertEqual(got, pack_h(1.5, 3.5))

    def test_single_rounding_discriminating_cases(self):
        # Discriminating test cases A through G from tools/fma_mix_probe.py where
        # hardware single rounding gives 1 ulp difference from double rounding.
        # Case A: 1 * 2^-40 + (1 + 2^-11) -> single: 0x3c01, double: 0x3c00
        got_a = run("v_fma_mixlo_f16 v0, v1, v2, v3", 0x3F800000, 0x2B800000, 0x3F801000, old=OLD)
        self.assertEqual(got_a, (OLD & 0xFFFF0000) | 0x3C01)

        # Case B: 1 * -2^-40 + (1 + 3 * 2^-11) -> single: 0x3c01, double: 0x3c02
        got_b = run("v_fma_mixhi_f16 v0, v1, v2, v3", 0x3F800000, 0xAB800000, 0x3F803000, old=OLD)
        self.assertEqual(got_b, (0x3C01 << 16) | (OLD & 0xFFFF))

        # Case C: -1 * 2^-40 - (1 + 2^-11) -> single: 0xbc01, double: 0xbc00
        got_c = run("v_fma_mixlo_f16 v0, v1, v2, v3", 0xBF800000, 0x2B800000, 0xBF801000, old=OLD)
        self.assertEqual(got_c, (OLD & 0xFFFF0000) | 0xBC01)

        # Case G (f16 operands): (1 + 2^-7) * (2^-11 - 2^-18) + (1 + 2^-10) -> single: 0x3c01, double: 0x3c02
        got_g = run("v_fma_mixlo_f16 v0, v1, v2, v3 op_sel_hi:[1,1,1]", 0x3C08, 0x0FF0, 0x3C01, old=OLD)
        self.assertEqual(got_g, (OLD & 0xFFFF0000) | 0x3C01)


class NegativeControls(unittest.TestCase):
    """Sabotaging the translated SPIR-V must change the observed output."""

    def test_sabotage_mixhi_half_mask_detected(self):
        w0, w1, w2 = pack_h(1.5, 2.0), f2u(2.0), pack_h(0.5, 1.0)
        spv, good = translate_and_run("v_fma_mixhi_f16 v0, v1, v2, v3 op_sel_hi:[1,0,1]", w0, w1, w2, old=OLD)
        self.assertEqual(good, (f2h(3.5) << 16) | (OLD & 0xFFFF))
        # Replacing the preserved-low-half mask 0xFFFF with 0 wipes OLD's low half.
        bad, n = re.subn(r"(OpConstant %uint )65535\b", r"\g<1>0", spv)
        self.assertGreaterEqual(n, 1)
        self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], good)


class FmaMixF32DenormalOperandTests(unittest.TestCase):
    """SHADER-298: an f32 operand of v_fma_mix* that is denormal reads as a zero of its sign (the game's MODE has
    denorm_mode_32 = 0), measured on gfx1036 by @Sanonymous01 for 0x00000001, 0x807fffff and the integer inline
    constants 1 and 64. The result is not flushed here (not measured), so 0 * y + denormal keeps the denormal only
    when the denormal is not an operand, which cannot happen; the checks below use operands only."""

    BIG = f2u(1.0e30)
    ZERO = 0x00000000
    NEG_ZERO = 0x80000000

    def test_denormal_multiplicand_reads_as_zero(self):
        # 1.4e-45 * 1e30 = 1.4e-15 if the denormal is kept; 0 if it is flushed.
        for d in (0x00000001, 0x007FFFFF):
            with self.subTest(denormal=hex(d), slot="src0"):
                self.assertEqual(run("v_fma_mix_f32 v0, v1, v2, v3", d, self.BIG, self.ZERO), 0x00000000)
            with self.subTest(denormal=hex(d), slot="src1"):
                self.assertEqual(run("v_fma_mix_f32 v0, v1, v2, v3", self.BIG, d, self.ZERO), 0x00000000)

    def test_negative_denormal_reads_as_negative_zero(self):
        # (-0) * 1e30 + (-0) = -0; a kept denormal would give -1.1e-8-ish tiny non-zero.
        for d in (0x80000001, 0x807FFFFF):
            with self.subTest(denormal=hex(d)):
                self.assertEqual(run("v_fma_mix_f32 v0, v1, v2, v3", d, self.BIG, self.NEG_ZERO), 0x80000000)

    def test_denormal_addend_reads_as_zero(self):
        # 0 * y + denormal: the denormal addend is flushed, so the result is +0 (kept it would be the denormal).
        self.assertEqual(run("v_fma_mix_f32 v0, v1, v2, v3", self.ZERO, self.BIG, 0x00000001), 0x00000000)

    def test_integer_inline_constants_are_denormal_f32_operands(self):
        # `1` and `64` in an f32 operand slot are the words 0x00000001 and 0x00000040: denormal, flushed to +0.
        for const in ("1", "64"):
            with self.subTest(const=const):
                self.assertEqual(run(f"v_fma_mix_f32 v0, {const}, v2, v3", self.ZERO, self.BIG, self.ZERO), 0x00000000)

    def test_normal_operands_are_untouched(self):
        a, b, c = f2u(2.5), f2u(-1.25), f2u(0.75)
        self.assertEqual(run("v_fma_mix_f32 v0, v1, v2, v3", a, b, c), f2u(ref_fma(a, b, c, (0, 0, 0), (0, 0, 0))))
        smallest_normal = 0x00800000      # 2**-126 is the first value that is not flushed
        got = run("v_fma_mix_f32 v0, v1, v2, v3", smallest_normal, f2u(1.0), self.ZERO)
        self.assertEqual(got, smallest_normal)

    def test_the_flush_reaches_the_f16_result(self):
        # 1.4e-45 * 3e37 = 4.2e-8 rounds to the f16 subnormal 0x0001 if the denormal is kept, and is 0 flushed.
        got = run("v_fma_mixlo_f16 v0, v1, v2, v3", 0x00000001, f2u(3.0e37), self.ZERO, old=0xAAAA0000)
        self.assertEqual(got, 0xAAAA0000)


if __name__ == "__main__":
    unittest.main()
