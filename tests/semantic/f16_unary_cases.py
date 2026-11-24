"""Shared harness for the one-source binary16 semantic tests (v_fract/rcp/rsq/sqrt_f16), standard library only.

A test file subclasses `UnaryF16` together with unittest.TestCase and sets OP, ref (a plain-Python reference taking
and returning binary16 words), CASES (hand-written (S0, D) pairs from the ISA), MOD_INPUTS and INLINE. Every case is
translated by tools/rdna2spirv.py and run on tests/semantic/spirv_mini_interp.py; the reference in f16_ref.py rounds
exactly once, to nearest even, which is what the hardware does for these ops (docs/design.md, f16 transcendentals).
"""
import math
import os
import random
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
from f16_ref import OLD, clamp01, f16_value, is_nan, round_f16, same_f16, scale  # noqa: E402

ONE, TWO, HALF, QUARTER, FOUR = 0x3C00, 0x4000, 0x3800, 0x3400, 0x4400
MAXH, INF, NINF, QNAN = 0x7BFF, 0x7C00, 0xFC00, 0x7E00
MIN_NORMAL, MIN_SUB, MAX_SUB = 0x0400, 0x0001, 0x03FF
SPECIAL = [0x0000, 0x8000, ONE, 0xBC00, TWO, 0xC000, HALF, 0xB800, 0x4200, 0x3E00, MAXH, 0xFBFF, INF, NINF, QNAN,
           0xFE00, 0x7E01, MIN_NORMAL, 0x8400, MIN_SUB, 0x8001, MAX_SUB, 0x83FF, 0x3C01, 0x3BFF, 0x1000, 0x6800,
           0x6801, 0x4C00, 0x4400, 0x4800, 0x3555, 0x7400, 0x0400]


def run(line, seeds, old=OLD, hi=0x5A5A):
    """Run one instruction on a one-lane pixel shader and return v0 as exported to mrt0.

    `seeds` maps a register ("v1", "s2", ...) to the word it starts with; a value that fits in 16 bits gets `hi` in its
    upper half, so reading the wrong half of a source shows. v0 starts as `old`.
    """
    prog = [f"v_mov_b32 v0, {old:#x}"]
    for reg, w in seeds.items():
        w = w if w > 0xFFFF else (hi << 16) | w
        prog.append(f"{'s' if reg[0] == 's' else 'v'}_mov_b32 {reg}, {w:#x}")
    prog += [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_unary_f16", prog))[0]


def exact_sqrt_ratio(num, den, k=64):
    """floor(sqrt(num/den) * 2^k) / 2^k as a Fraction: far finer than binary16, and exact when the root is rational."""
    return Fraction(math.isqrt((num << (2 * k)) // den), 1 << k) if num else Fraction(0)


def ref_rcp(h):
    if is_nan(h):
        return h | 0x7E00
    if h & 0x7FFF == 0:
        return 0xFC00 if h & 0x8000 else 0x7C00
    if h & 0x7FFF == 0x7C00:
        return h & 0x8000
    return round_f16(1 / Fraction(f16_value(h)))


def ref_sqrt(h):
    if is_nan(h):
        return h | 0x7E00
    if h & 0x7FFF == 0:
        return h
    if h & 0x8000:
        return 0xFE00
    if h == 0x7C00:
        return h
    v = Fraction(f16_value(h))
    return round_f16(exact_sqrt_ratio(v.numerator, v.denominator))


def ref_rsq(h):
    if is_nan(h):
        return h | 0x7E00
    if h & 0x7FFF == 0:
        return 0xFC00 if h & 0x8000 else 0x7C00
    if h & 0x8000:
        return 0xFE00
    if h == 0x7C00:
        return 0
    v = Fraction(f16_value(h))
    return round_f16(exact_sqrt_ratio(v.denominator, v.numerator))


def ref_fract(h):
    """x - floor(x), rounded once; 1.0 (a tiny negative x) becomes the largest word below 1; fract(+-0) = +0."""
    if is_nan(h):
        return h | 0x7E00
    if h & 0x7FFF == 0x7C00:
        return 0xFE00
    x = Fraction(f16_value(h))
    r = round_f16(x - math.floor(x))
    return 0x3BFF if r == 0x3C00 else r


class UnaryF16:
    """Tests shared by the one-source f16 ops; a subclass names OP and the reference and supplies the tables."""
    OP = ""
    CASES = []          # (S0, D), by hand
    MOD_INPUTS = []     # finite, ordinary operands for the modifier tests
    INLINE = []         # (inline constant text, its binary16 word, D), by hand
    OP_SEL = []         # (S0 word with a different half each side, D for the low half, D for the high half)
    NAN_OK = True

    @staticmethod
    def ref(h):
        raise NotImplementedError

    def line(self, suffix="", ops="v0, v1"):
        return f"{self.OP}{suffix} {ops}"

    def test_hand_written_table_matches_reference(self):
        for a, want in self.CASES:
            self.assertTrue(same_f16(self.ref(a), want), (hex(a), hex(want), hex(self.ref(a))))

    def test_e32_writes_low_half_and_keeps_high_half(self):
        for a, want in self.CASES:
            with self.subTest(a=hex(a)):
                got = run(self.line(), {"v1": a})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_vop3_sgpr_and_literal_sources(self):
        for a, want in self.CASES:
            forms = [
                ("e64 vgpr", self.line("_e64"), {"v1": a}),
                ("e64 sgpr", self.line("_e64", "v0, s1"), {"s1": a}),
                ("e32 sgpr", self.line("", "v0, s1"), {"s1": a}),
                ("e32 literal", self.line("", f"v0, {a:#x}"), {}),
                ("e64 literal", self.line("_e64", f"v0, {a:#x}"), {}),
            ]
            for form, line, seeds in forms:
                with self.subTest(a=hex(a), form=form):
                    got = run(line, seeds)
                    self.assertEqual(got >> 16, OLD >> 16)
                    self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_inline_constants(self):
        for text, c, want in self.INLINE:
            with self.subTest(const=text):
                got = run(self.line("", f"v0, {text}"), {})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))
                self.assertTrue(same_f16(self.ref(c), want))

    def test_op_sel_selects_source_and_destination_halves(self):
        for word, want_lo_src, want_hi_src in self.OP_SEL:
            for sh in (0, 1):
                for dh in (0, 1):
                    want = want_hi_src if sh else want_lo_src
                    expect = ((want << 16) | (OLD & 0xFFFF)) if dh else ((OLD & 0xFFFF0000) | want)
                    with self.subTest(word=hex(word), op_sel=(sh, dh)):
                        got = run(self.line("_e64", "v0, v1") + f" op_sel:[{sh},{dh}]", {"v1": word})
                        self.assertEqual(got, expect)

    def test_neg_and_abs_source_modifiers(self):
        for a in self.MOD_INPUTS:
            for ops, src in (("-v1", a ^ 0x8000), ("|v1|", a & 0x7FFF), ("-|v1|", a | 0x8000)):
                with self.subTest(a=hex(a), ops=ops):
                    got = run(self.line("_e64", f"v0, {ops}"), {"v1": a})
                    self.assertEqual(got >> 16, OLD >> 16)
                    self.assertTrue(same_f16(got & 0xFFFF, self.ref(src)), (hex(got), hex(self.ref(src))))

    def test_output_modifiers_and_clamp(self):
        for a in self.MOD_INPUTS:
            base = self.ref(a)
            # `clamp` of a NaN result is not modelled (the translator emits FClamp; docs/design.md, float min/max/med3 with
            # NaN), so a NaN result is only checked with the output multipliers.
            # #4542, measured on gfx1036: an f16 destination ignores omod (mul:2/mul:4/div:2) under the game's MODE
            # (IEEE=0, f16 denormals kept) and keeps clamp (docs/design.md, f16 output modifiers).
            for suffix, want in ((" mul:2", base), (" mul:4", base),
                                 (" div:2", base), (" clamp", clamp01(base)),
                                 (" mul:2 clamp", clamp01(base))):
                if is_nan(base) and "clamp" in suffix:
                    continue
                with self.subTest(a=hex(a), mod=suffix):
                    got = run(self.line("_e64") + suffix, {"v1": a})
                    self.assertEqual(got >> 16, OLD >> 16)
                    self.assertTrue(same_f16(got & 0xFFFF, want), (hex(got), hex(want)))

    def test_every_finite_normal_exponent_boundary_and_its_neighbours(self):
        # 1-ulp neighbours of each power of two, both signs, plus the subnormal range edges: against the reference
        words = []
        for e in range(0, 31):
            for d in (-1, 0, 1):
                w = (e << 10) + d
                if 0 <= w < 0x7C00:
                    words += [w, w | 0x8000]
        for a in words:
            with self.subTest(a=hex(a)):
                got = run(self.line(), {"v1": a})
                self.assertTrue(same_f16(got & 0xFFFF, self.ref(a)), (hex(a), hex(got), hex(self.ref(a))))

    def test_random_operands(self):
        rng = random.Random(f"{self.OP}-seed")
        pool = SPECIAL + [rng.getrandbits(16) for _ in range(40)]
        for _ in range(150):
            a = rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(16)
            with self.subTest(a=hex(a)):
                got = run(self.line(), {"v1": a})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, self.ref(a)), (hex(a), hex(got), hex(self.ref(a))))
