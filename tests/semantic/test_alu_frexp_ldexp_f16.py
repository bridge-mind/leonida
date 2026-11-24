"""Semantic interpreter tests for RDNA2 v_frexp_mant_f16, v_frexp_exp_i16_f16 and v_ldexp_f16.

The binary16 versions of the f32 ops covered by test_alu_frexp_ldexp.py: v_frexp_mant_f16 returns the mantissa of S0 in
[0.5, 1) with its sign, v_frexp_exp_i16_f16 the matching exponent (S0 = mant * 2^exp, as a 16-bit integer) and
v_ldexp_f16 computes S0 * 2^S1 with S1 a signed 16-bit integer (RDNA2 ISA). Zero gives a signed zero mantissa and
exponent 0; infinities and NaNs pass through the mantissa and ldexp, and give exponent 0. op_sel picks the high half of
a source and the last bit sends the result to the high half of the destination; the other half of the destination is
kept. Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference (math.frexp /
math.ldexp on the double value of the half; struct's binary16 packing rounds to nearest even).

f16 denormals are kept, as in the MODE inferred for the game (denorm_mode_16_64 = 3): a denormal operand is normalized
(mantissa in [0.5, 1), exponent down to -23) and an ldexp result below the smallest normal is a denormal rounded to
nearest even. That is what an RDNA2 GPU (gfx1036) gives natively in that mode, for every one of the 65,536 f16 inputs
of both frexp ops and for all of them under 16 exponents of v_ldexp_f16 (docs/design.md); the MEASURED cases below
are words read back from it. The high half of v_frexp_exp_i16_f16's destination is kept on the hardware too.
"""
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
INF, NINF, QNAN = 0x7C00, 0xFC00, 0x7E00
MIN_NORMAL = 0x0400
MAX_FINITE = 0x7BFF


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def is_normal(h: int) -> bool:
    return 0 < ((h >> 10) & 0x1F) < 31


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_frexp_ldexp_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_frexp_ldexp_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int = 0, v0: int = 0) -> int:
    """Seed v0, v1 and v2, run ``line``, return v0 (the whole 32-bit register)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {v1 & 0xFFFFFFFF:#x}", f"v_mov_b32 v2, {v2 & 0xFFFFFFFF:#x}",
             f"v_mov_b32 v0, {v0 & 0xFFFFFFFF:#x}", line]
    asm = "; test_frexp_ldexp_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def is_finite_nonzero(h: int) -> bool:
    return (h >> 10) & 0x1F != 31 and h & 0x7FFF != 0


def ref_mant(h: int) -> int:
    if not is_finite_nonzero(h):
        return h                                     # signed zero, infinity, NaN
    return f2h(math.frexp(h2f(h))[0])


def ref_exp(h: int) -> int:
    return math.frexp(h2f(h))[1] & 0xFFFF if is_finite_nonzero(h) else 0


def ref_ldexp(h: int, n: int) -> int:
    if not is_finite_nonzero(h):
        return h
    n &= 0xFFFF
    n = n - (1 << 16) if n & 0x8000 else n
    try:
        return f2h(math.ldexp(h2f(h), max(-64, min(64, n))))   # exact, then rounded to nearest even
    except OverflowError:
        return (h & 0x8000) | INF


# Read back from gfx1036 (RDNA2), IEEE 0, f16 denormals kept: (input, mantissa, exponent) and (input, n, ldexp).
MEASURED_FREXP = ((0x0001, 0x3800, 0xFFE9), (0x8001, 0xB800, 0xFFE9), (0x0003, 0x3A00, 0xFFEA),
                  (0x0200, 0x3800, 0xFFF2), (0x03FF, 0x3BFE, 0xFFF2), (0x83FF, 0xBBFE, 0xFFF2),
                  (0x0400, 0x3800, 0xFFF3), (0x0010, 0x3800, 0xFFED))
MEASURED_LDEXP = ((0x3C00, -23, 0x0002), (0x3C00, -15, 0x0200), (0x3C00, -24, 0x0001), (0x3E00, -23, 0x0003),
                  (0x3E00, -15, 0x0300), (0x3E00, -24, 0x0002), (0x0001, 1, 0x0002), (0x0200, 1, 0x0400),
                  (0x8001, 10, 0x8400), (0x0001, 30, 0x5400), (0x0400, -1, 0x0200), (0x0400, -10, 0x0001),
                  (0x0400, -25, 0x0000), (0x7BFF, -24, 0x1BFF), (0x7BFF, -30, 0x0400), (0x3C00, -25, 0x0000))


def sample_normals(seed: int, count: int):
    rng = random.Random(seed)
    out = [f2h(1.0), f2h(-1.0), f2h(0.5), f2h(0.75), f2h(8.0), f2h(-0.1), f2h(3.14159), MIN_NORMAL, MAX_FINITE,
           MAX_FINITE | 0x8000, MIN_NORMAL | 0x8000]
    out += [(ex << 10) | rng.getrandbits(10) for ex in (1, 2, 14, 15, 16, 29, 30)]   # every end of the exponent range
    while len(out) < count:
        h = rng.getrandbits(16)
        if is_normal(h):
            out.append(h)
    return out


class FrexpTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_mant(f2h(8.0)), f2h(0.5))
        self.assertEqual(ref_exp(f2h(8.0)), 4)
        self.assertEqual(ref_exp(f2h(0.75)), 0)
        self.assertEqual(ref_exp(f2h(-0.1)), (-3) & 0xFFFF)
        self.assertEqual(ref_exp(MAX_FINITE), 16)
        self.assertEqual(ref_exp(MIN_NORMAL), (-13) & 0xFFFF)

    def test_known_values(self):
        for x, m, e in ((1.0, 0.5, 1), (-1.0, -0.5, 1), (0.5, 0.5, 0), (3.0, 0.75, 2), (-0.1875, -0.75, -2)):
            with self.subTest(x=x):
                self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", f2h(x)), f2h(m))
                self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", f2h(x)), e & 0xFFFF)

    def test_matches_python_on_normals(self):
        for h in sample_normals(0x89, 60):
            with self.subTest(x=hex(h)):
                self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", h), ref_mant(h))
                self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", h), ref_exp(h))

    def test_mantissa_times_two_to_the_exponent_is_the_input(self):
        for h in sample_normals(0x90, 20):
            m, e = h2f(run("v_frexp_mant_f16_e32 v0, v1", h)), run("v_frexp_exp_i16_f16_e32 v0, v1", h)
            e = e - (1 << 16) if e & 0x8000 else e
            self.assertEqual(f2h(math.ldexp(m, e)), h)
            self.assertTrue(0.5 <= abs(m) < 1.0)

    def test_zero_and_specials(self):
        for h in (0x0000, 0x8000):
            self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", h), h)       # signed zero
            self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", h), 0)
        for h in (INF, NINF, QNAN, QNAN | 0x8000):
            self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", h), h)       # passed through
            self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", h), 0)

    def test_denormal_is_normalized_as_measured(self):
        for h, mant, exp in MEASURED_FREXP:
            with self.subTest(x=hex(h)):
                self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", h), mant)
                self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", h), exp)
                self.assertEqual((ref_mant(h), ref_exp(h)), (mant, exp))

    def test_matches_python_on_denormals(self):
        for h in list(range(1, 0x400, 37)) + [0x8000 | h for h in range(3, 0x400, 101)]:
            with self.subTest(x=hex(h)):
                self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", h), ref_mant(h))
                self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", h), ref_exp(h))

    def test_high_half_of_the_source_is_ignored_and_the_destination_high_half_kept(self):
        h = f2h(8.0)
        self.assertEqual(run("v_frexp_mant_f16_e32 v0, v1", 0xABCD0000 | h, v0=0x12345678), 0x12340000 | f2h(0.5))
        self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, v1", 0xABCD0000 | h, v0=0x12345678), 0x12340000 | 4)

    def test_op_sel(self):
        lo, hi = f2h(8.0), f2h(-0.1875)
        v1 = (hi << 16) | lo
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, v1 op_sel:[1,0]", v1) & 0xFFFF, f2h(-0.75))
        self.assertEqual(run("v_frexp_exp_i16_f16_e64 v0, v1 op_sel:[1,0]", v1) & 0xFFFF, (-2) & 0xFFFF)
        # destination bit: the result goes to the high half and the low half is kept
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, v1 op_sel:[0,1]", v1, v0=0x1111_2222), (f2h(0.5) << 16) | 0x2222)
        self.assertEqual(run("v_frexp_exp_i16_f16_e64 v0, v1 op_sel:[1,1]", v1, v0=0x1111_2222), (((-2) & 0xFFFF) << 16) | 0x2222)

    def test_source_modifiers_and_inline_constants(self):
        one = f2h(1.0)
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, -v1", one), f2h(-0.5))
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, |v1|", f2h(-1.0)), f2h(0.5))
        self.assertEqual(run("v_frexp_exp_i16_f16_e64 v0, -v1", f2h(8.0)), 4)
        self.assertEqual(run("v_frexp_mant_f16_e32 v0, 8.0", 0), f2h(0.5))
        self.assertEqual(run("v_frexp_exp_i16_f16_e32 v0, 8.0", 0), 4)

    def test_clamp(self):
        cases = [
            (f2h(1.0), f2h(0.5)),      # +1.0 -> mant 0.5 (in range, kept)
            (f2h(-1.0), 0),            # -1.0 -> mant -0.5 -> clamped to +0.0
            (0x8000, 0),               # -0.0 -> clamped to +0.0
            (f2h(1.5), f2h(0.75)),     # +1.5 -> mant 0.75 (in range, kept)
            (0x0001, f2h(0.5)),        # +min_sub -> normalized mant 0.5 (kept)
            (0x8001, 0),               # -min_sub -> clamped to +0.0
            (0x7C00, f2h(1.0)),        # +Inf -> clamped to 1.0 (0x3c00)
            (0xFC00, 0),               # -Inf -> clamped to +0.0
            (0x7E00, 0),               # +qNaN -> clamped to +0.0
            (0x7C01, 0),               # +sNaN -> clamped to +0.0
        ]
        for val, want in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(run("v_frexp_mant_f16_e64 v0, v1 clamp", val), want)
        # Dirty upper-16-bit source VGPR values must not leak into clamp or destination high half
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, v1 clamp", 0xDEAD0000, v0=0xBEEF1234), 0xBEEF0000)
        self.assertEqual(run("v_frexp_mant_f16_e64 v0, v1 clamp", 0xDEAD0000 | f2h(0.5), v0=0xBEEF1234), 0xBEEF0000 | f2h(0.5))

    def test_output_modifiers(self):
        # Under console MODE (denorm_mode_16_64 = 3), mul:2/mul:4/div:2 are hardware no-ops on frexp_mant_f16,
        # clamp still saturates to [0.0, 1.0], and frexp_exp_i16_f16 ignores both clamp and omod.
        for mod in ("mul:2", "mul:4", "div:2"):
            with self.subTest(op="v_frexp_mant_f16", mod=mod):
                self.assertEqual(run(f"v_frexp_mant_f16_e64 v0, v1 {mod}", f2h(1.0), v0=0xBEEF1234), 0xBEEF0000 | f2h(0.5))
                self.assertEqual(run(f"v_frexp_mant_f16_e64 v0, v1 clamp {mod}", f2h(-1.0), v0=0xBEEF1234), 0xBEEF0000)
                self.assertEqual(run(f"v_frexp_mant_f16_e64 v0, v1 clamp {mod}", 0x7C00, v0=0xBEEF1234), 0xBEEF3C00)
        for mod in ("clamp", "mul:2", "mul:4", "div:2", "clamp mul:2", "clamp mul:4", "clamp div:2"):
            with self.subTest(op="v_frexp_exp_i16_f16", mod=mod):
                self.assertEqual(run(f"v_frexp_exp_i16_f16_e64 v0, v1 {mod}", f2h(8.0), v0=0xBEEF1234), 0xBEEF0004)
        for line in ("v_frexp_mant_f16_e64 v0, v1 mul:8", "v_frexp_mant_f16_e64 v0, v1 op_sel_hi:[1,0]",
                     "v_frexp_exp_i16_f16_e64 v0, v1 div:4", "v_frexp_exp_i16_f16_e64 v0, v1 neg_lo:[1]"):
            with self.subTest(rejected=line), self.assertRaises(rdna2spirv.Unsupported):
                run(line, f2h(1.0))


class LdexpTests(unittest.TestCase):
    def test_known_values(self):
        for x, n, want in ((1.0, 3, 8.0), (1.0, -1, 0.5), (-3.0, 2, -12.0), (0.75, 0, 0.75), (6.0, -2, 1.5)):
            with self.subTest(x=x, n=n):
                self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", f2h(x), n), f2h(want))

    def test_matches_python_on_normals(self):
        rng = random.Random(0x91)
        for h in sample_normals(0x92, 40):
            for n in (rng.randrange(-20, 20), rng.randrange(-300, 300)):
                with self.subTest(x=hex(h), n=n):
                    self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), ref_ldexp(h, n))

    def test_every_exponent_step_at_the_edges(self):
        for n in range(-32, 33):
            for h in (f2h(1.0), f2h(-1.0), MIN_NORMAL, MAX_FINITE, MAX_FINITE | 0x8000):
                with self.subTest(x=hex(h), n=n):
                    self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), ref_ldexp(h, n))

    def test_overflow_and_underflow_saturate(self):
        L = "v_ldexp_f16_e64 v0, v1, v2"
        self.assertEqual(run(L, f2h(1.0), 15), f2h(2.0 ** 15))
        self.assertEqual(run(L, f2h(1.0), 16), INF)
        self.assertEqual(run(L, f2h(-1.0), 16), NINF)
        self.assertEqual(run(L, f2h(1.0), 0x7FFF), INF)
        self.assertEqual(run(L, f2h(1.0), -14), MIN_NORMAL)
        self.assertEqual(run(L, f2h(1.0), -15), 0x0200)              # a denormal, not zero
        self.assertEqual(run(L, f2h(-1.0), -15), 0x8200)
        self.assertEqual(run(L, f2h(1.0), -25), 0)                   # half the smallest denormal: ties to even 0
        self.assertEqual(run(L, f2h(-1.0), 0x8000), 0x8000)          # INT16_MIN
        self.assertEqual(run(L, MAX_FINITE, 1), INF)
        self.assertEqual(run(L, MIN_NORMAL, -1), 0x0200)

    def test_the_exponent_is_a_signed_16_bit_integer(self):
        L = "v_ldexp_f16_e64 v0, v1, v2"
        # only the low 16 bits of S1 count: 0x00010003 is 3, 0xFFFF is -1
        self.assertEqual(run(L, f2h(1.0), 0x00010003), f2h(8.0))
        self.assertEqual(run(L, f2h(1.0), 0xFFFF), f2h(0.5))
        self.assertEqual(run(L, f2h(1.0), 0x8000), 0)
        self.assertEqual(run(L, f2h(1.0), 0x7FFF), INF)

    def test_zero_and_specials_ignore_the_exponent(self):
        for h in (0x0000, 0x8000):
            for n in (0, 5, -5, 0x7FFF, 0x8000):
                self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), h)
        for h in (INF, NINF, QNAN):
            for n in (0, 5, -200, 0x7FFF):
                self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), h)

    def test_denormals_as_measured(self):
        for h, n, want in MEASURED_LDEXP:
            with self.subTest(x=hex(h), n=n):
                self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), want)
                self.assertEqual(ref_ldexp(h, n), want)

    def test_denormal_inputs_and_results_match_python(self):
        for h in (0x0001, 0x0003, 0x0155, 0x03FF, 0x8001, 0x82AB, f2h(1.5), f2h(-1.75), MAX_FINITE):
            for n in (-30, -26, -25, -24, -20, -16, -14, 0, 3, 9, 10, 11, 24):
                with self.subTest(x=hex(h), n=n):
                    self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", h, n), ref_ldexp(h, n))

    def test_op_sel(self):
        L = "v_ldexp_f16_e64 v0, v1, v2"
        v1 = (f2h(3.0) << 16) | f2h(1.0)
        v2 = (2 << 16) | 1                       # high half 2, low half 1
        self.assertEqual(run(L + " op_sel:[0,0,0]", v1, v2) & 0xFFFF, f2h(2.0))     # 1.0 * 2^1
        self.assertEqual(run(L + " op_sel:[1,0,0]", v1, v2) & 0xFFFF, f2h(6.0))     # 3.0 * 2^1
        self.assertEqual(run(L + " op_sel:[0,1,0]", v1, v2) & 0xFFFF, f2h(4.0))     # 1.0 * 2^2
        self.assertEqual(run(L + " op_sel:[1,1,0]", v1, v2) & 0xFFFF, f2h(12.0))    # 3.0 * 2^2
        # destination bit: the result goes to the high half and the low half is kept
        self.assertEqual(run(L + " op_sel:[1,1,1]", v1, v2, v0=0x1111_2222), (f2h(12.0) << 16) | 0x2222)
        self.assertEqual(run(L + " op_sel:[0,0,1]", v1, v2, v0=0x1111_2222), (f2h(2.0) << 16) | 0x2222)

    def test_destination_high_half_kept(self):
        self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2", f2h(1.0), 3, v0=0x12345678), 0x12340000 | f2h(8.0))

    def test_source_modifiers_and_inline_constants(self):
        self.assertEqual(run("v_ldexp_f16_e64 v0, -v1, v2", f2h(1.0), 1), f2h(-2.0))
        self.assertEqual(run("v_ldexp_f16_e64 v0, |v1|, v2", f2h(-1.0), 1), f2h(2.0))
        self.assertEqual(run("v_ldexp_f16_e32 v0, 1.0, v2", 0, 3), f2h(8.0))
        self.assertEqual(run("v_ldexp_f16_e64 v0, v1, 5", f2h(1.0)), f2h(32.0))

    def test_clamp(self):
        cases = [
            (f2h(1.0), 1, f2h(1.0)),     # 1.0 * 2 = 2.0 -> clamped to 1.0
            (f2h(1.0), -1, f2h(0.5)),    # 1.0 * 0.5 = 0.5 -> in range, kept
            (f2h(-1.0), 0, 0),           # -1.0 -> clamped to +0.0
            (0x8000, 0, 0),              # -0.0 -> clamped to +0.0
            (0x7C00, 0, f2h(1.0)),       # +Inf -> clamped to 1.0 (0x3c00)
            (0xFC00, 0, 0),              # -Inf -> clamped to +0.0
            (0x7E00, 0, 0),              # +qNaN -> clamped to +0.0
            (f2h(1.0), 16, f2h(1.0)),    # overflow -> clamped to 1.0
            (0x0001, 0, 0x0001),         # +min_sub -> in range, kept
        ]
        for s0, s1, want in cases:
            with self.subTest(s0=hex(s0), s1=s1):
                self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2 clamp", s0, s1), want)
        # Dirty upper-16-bit source VGPR values must not leak into clamp or destination high half
        self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2 clamp", 0xDEAD0000, 0, v0=0xBEEF1234), 0xBEEF0000)
        self.assertEqual(run("v_ldexp_f16_e64 v0, v1, v2 clamp", 0xDEAD0000 | f2h(0.5), 0, v0=0xBEEF1234), 0xBEEF0000 | f2h(0.5))

    def test_output_modifiers(self):
        # Under console MODE (denorm_mode_16_64 = 3), mul:2/mul:4/div:2 are hardware no-ops on v_ldexp_f16,
        # while clamp applies [0.0, 1.0] saturation alone or combined with mul:*/div:*.
        for mod in ("mul:2", "mul:4", "div:2"):
            with self.subTest(mod=mod):
                self.assertEqual(run(f"v_ldexp_f16_e64 v0, v1, v2 {mod}", f2h(0.5), 1, v0=0xBEEF1234), 0xBEEF3C00)
                self.assertEqual(run(f"v_ldexp_f16_e64 v0, v1, v2 {mod}", f2h(1.0), 1, v0=0xBEEF1234), 0xBEEF4000)
                self.assertEqual(run(f"v_ldexp_f16_e64 v0, v1, v2 clamp {mod}", f2h(1.0), 1, v0=0xBEEF1234), 0xBEEF3C00)
                self.assertEqual(run(f"v_ldexp_f16_e64 v0, v1, v2 clamp {mod}", f2h(-1.0), 1, v0=0xBEEF1234), 0xBEEF0000)
        for line in ("v_ldexp_f16_e64 v0, v1, v2 mul:8", "v_ldexp_f16_e64 v0, v1, v2 div:4",
                     "v_ldexp_f16_e64 v0, v1, v2 neg_lo:[1,0]"):
            with self.subTest(rejected=line), self.assertRaises(rdna2spirv.Unsupported):
                run(line, f2h(1.0), 1)


if __name__ == "__main__":
    unittest.main()
