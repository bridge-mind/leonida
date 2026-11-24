"""v_subrev_f16 (RDNA2 VOP2 / VOP3): D.f16 = S1.f16 - S0.f16 (v_sub_f16 with its operands swapped), binary16.

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand (from the ISA) next to each case, rechecked against the plain-Python binary16 reference in
tests/semantic/f16_ref.py, and extended with seeded random operands. Covers the _e32 form (the high half of the
destination is kept), the _e64 form with op_sel, SGPR / literal / inline-constant sources, neg/abs, omod and clamp.
Standard library only: no GPU, no LLVM, no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
from f16_ref import OLD, ref_sub, same_f16, scale  # noqa: E402

ONE, TWO, THREE, HALF, QUARTER = 0x3C00, 0x4000, 0x4200, 0x3800, 0x3400
MAXH, INF, NINF, QNAN = 0x7BFF, 0x7C00, 0xFC00, 0x7E00
MIN_NORMAL, MIN_SUB, MAX_SUB = 0x0400, 0x0001, 0x03FF


def ref_subrev(a, b):
    return ref_sub(b, a)


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
    return support.mrt0_words(support.translate_lines("ps_subrev_f16", prog))[0]


# (src0, src1, D), written by hand.
CASES = [
    (ONE, THREE, TWO),                      # 3 - 1 = 2
    (TWO, ONE, 0xBC00),                     # 1 - 2 = -1
    (QUARTER, HALF, QUARTER),               # 0.5 - 0.25
    (0x3E00, 0x3E00, 0x0000),               # x - x = +0
    (0x0000, 0x8000, 0x8000),               # -0 - +0 = -0
    (0x0000, 0x0000, 0x0000),               # +0 - +0 = +0
    (0x8000, 0x0000, 0x0000),               # +0 - -0 = +0
    (0x8000, 0x8000, 0x0000),               # -0 - -0 = -0 + +0 = +0
    (0xFBFF, MAXH, INF),                    # 65504 - -65504 overflows
    (0xCC00, MAXH, INF),                    # 65504 - -16 = 65520: tie, even is the overflow
    (MIN_SUB, MIN_NORMAL, MAX_SUB),         # 2^-14 - 2^-24 = the largest subnormal
    (0x8001, MIN_SUB, 0x0002),              # 2^-24 - -2^-24
    (0x9000, ONE, ONE),                     # 1 - -2^-11: half an ulp, ties to even, stays 1
    (0x9001, ONE, 0x3C01),                  # a little over half an ulp rounds up
    (ONE, 0x3C01, 0x1400),                  # (1 + 2^-10) - 1 = 2^-10, exact
    (INF, INF, QNAN),                       # inf - inf = NaN
    (NINF, INF, INF),
    (QNAN, ONE, QNAN),
]

SPECIAL = [0x0000, 0x8000, ONE, 0xBC00, TWO, 0xC000, HALF, 0xB800, THREE, 0x3E00, MAXH, 0xFBFF, INF, NINF, QNAN,
           MIN_NORMAL, MIN_SUB, 0x8001, MAX_SUB, 0x3C01, 0x3BFF, 0x1000, 0x6800, 0x6801, 0x4C00]


class SubrevF16(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for a, b, want in CASES:
            self.assertTrue(same_f16(ref_subrev(a, b), want), (hex(a), hex(b), hex(ref_subrev(a, b))))

    def test_e32_writes_low_half_and_keeps_high_half(self):
        for a, b, want in CASES:
            with self.subTest(a=hex(a), b=hex(b)):
                got = run("v_subrev_f16 v0, v1, v2", {"v1": a, "v2": b})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_vop3_sgpr_and_literal_sources(self):
        for a, b, want in CASES[:14]:
            forms = [
                ("e64 vgpr", "v_subrev_f16_e64 v0, v1, v2", {"v1": a, "v2": b}),
                ("e64 sgpr", "v_subrev_f16_e64 v0, s1, s2", {"s1": a, "s2": b}),
                ("e32 sgpr src0", "v_subrev_f16 v0, s1, v2", {"s1": a, "v2": b}),
                ("e32 literal src0", f"v_subrev_f16 v0, {a:#x}, v2", {"v2": b}),
                ("e64 literal src1", f"v_subrev_f16_e64 v0, v1, {b:#x}", {"v1": a}),
            ]
            for form, line, seeds in forms:
                with self.subTest(a=hex(a), b=hex(b), form=form):
                    got = run(line, seeds)
                    self.assertEqual(got >> 16, OLD >> 16)
                    self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_inline_constants(self):
        # (inline constant as src0, its binary16 value, v1 low half, D), by hand
        for text, c, b, want in (("2.0", TWO, HALF, 0xBE00), ("-1.0", 0xBC00, HALF, 0x3E00), ("0.5", HALF, QUARTER, 0xB400),
                              ("4.0", 0x4400, 0xB800, 0xC480), ("1.0", ONE, ONE, 0x0000)):
            with self.subTest(const=text):
                self.assertEqual(run(f"v_subrev_f16 v0, {text}, v1", {"v1": b}) & 0xFFFF, want)
                self.assertEqual(ref_subrev(c, b), want)

    def test_op_sel_selects_source_and_destination_halves(self):
        # src0 = (lo 1.0, hi 3.0), src1 = (lo 0.5, hi -0.25); D by hand for each (src0 half, src1 half).
        s0, s1 = (THREE << 16) | ONE, (0xB400 << 16) | HALF
        expect_lo = {(0, 0): 0xB800, (1, 0): 0xC100, (0, 1): 0xBD00, (1, 1): 0xC280}      # -0.5, -2.5, -1.25, -3.25
        for h0 in (0, 1):
            for h1 in (0, 1):
                for dh in (0, 1):
                    want = expect_lo[(h0, h1)]
                    expect = ((want << 16) | (OLD & 0xFFFF)) if dh else ((OLD & 0xFFFF0000) | want)
                    with self.subTest(op_sel=(h0, h1, dh)):
                        got = run(f"v_subrev_f16_e64 v0, v1, v2 op_sel:[{h0},{h1},{dh}]", {"v1": s0, "v2": s1})
                        self.assertEqual(got, expect)

    def test_neg_and_abs_source_modifiers(self):
        # v1 = 1.0, v2 = -0.5; (operand text, D) by hand
        for ops, want in (("v1, v2", 0xBE00), ("-v1, v2", HALF), ("v1, -v2", 0xB800), ("|v1|, |v2|", 0xB800),
                          ("-|v1|, v2", HALF), ("v1, -|v2|", 0xBE00)):
            with self.subTest(ops=ops):
                got = run(f"v_subrev_f16_e64 v0, {ops}", {"v1": ONE, "v2": 0xB800})
                self.assertEqual(got & 0xFFFF, want)

    def test_output_modifiers_and_clamp(self):
        # #4542, measured on gfx1036: an f16 destination ignores omod (mul:2/mul:4/div:2) under the game's MODE
        # (IEEE=0, f16 denormals kept) and keeps clamp (docs/design.md, f16 output modifiers).
        # unmodified result: 2.5 - 0.5 = 2.0 (src1 - src0); clamp limits it to [0, 1] (negatives and NaN become +0)
        for suffix, want in ((" mul:2", 0x4000), (" mul:4", 0x4000), (" div:2", 0x4000), (" clamp", ONE)):
            with self.subTest(mod=suffix):
                got = run("v_subrev_f16_e64 v0, v1, v2" + suffix, {"v1": HALF, "v2": 0x4100})
                self.assertEqual(got & 0xFFFF, want)
        for (a, b), suffix, want in (((QUARTER, HALF), " clamp", QUARTER), ((ONE, HALF), " clamp", 0x0000),
                                  ((0xB800, ONE), " div:2 clamp", ONE), ((QUARTER, ONE), " mul:2 clamp", 0x3A00)):
            with self.subTest(a=hex(a), b=hex(b), mod=suffix):
                self.assertEqual(run("v_subrev_f16_e64 v0, v1, v2" + suffix, {"v1": a, "v2": b}) & 0xFFFF, want)
        for suffix, f in ((" mul:2", 2.0), (" mul:4", 4.0), (" div:2", 0.5)):
            got = run("v_subrev_f16_e64 v0, v1, v2" + suffix, {"v1": 0x4900, "v2": 0xC100})      # 10 and -2.5
            self.assertEqual(got & 0xFFFF, ref_subrev(0x4900, 0xC100))

    def test_random_operands(self):
        rng = random.Random(0x5B2E5)
        pool = SPECIAL + [rng.getrandbits(16) for _ in range(40)]
        for _ in range(120):
            a = rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(16)
            b = rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(16)
            with self.subTest(a=hex(a), b=hex(b)):
                got = run("v_subrev_f16 v0, v1, v2", {"v1": a, "v2": b})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, ref_subrev(a, b)), (hex(got), hex(ref_subrev(a, b))))


if __name__ == "__main__":
    unittest.main()
