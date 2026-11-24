"""v_fma_f16 (RDNA2 VOP3): D.f16 = S0.f16 * S1.f16 + S2.f16 with a single rounding (binary16, nearest even).

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand (from the ISA) next to each case, rechecked against the plain-Python binary16 reference in
tests/semantic/f16_ref.py, and extended with seeded random operands. Covers the fused (not double-rounded) result, op_sel
for all three sources and the destination, SGPR / literal / inline-constant sources, neg/abs, omod and clamp.
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
from f16_ref import OLD, ref_fma, same_f16, scale  # noqa: E402

ONE, TWO, THREE, HALF, QUARTER = 0x3C00, 0x4000, 0x4200, 0x3800, 0x3400
MAXH, INF, NINF, QNAN = 0x7BFF, 0x7C00, 0xFC00, 0x7E00
MIN_NORMAL, MIN_SUB, MAX_SUB = 0x0400, 0x0001, 0x03FF


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
    return support.mrt0_words(support.translate_lines("ps_fma_f16", prog))[0]


# (src0, src1, src2, D), written by hand.
CASES = [
    (TWO, THREE, ONE, 0x4700),              # 2 * 3 + 1 = 7
    (0x3E00, TWO, 0xC200, 0x0000),          # 1.5 * 2 + -3 = +0
    (HALF, HALF, QUARTER, HALF),            # 0.5 * 0.5 + 0.25 = 0.5
    (0x3C01, 0x3C01, 0xBC02, 0x0010),       # (1+2^-10)^2 - (1+2^-9) = 2^-20 exactly; unfused it would be 0
    (0x3C01, 0x3C01, 0xBC00, 0x1800),       # 2^-9 + 2^-20: half a quantum of 2^-9 ties to even
    (MAXH, TWO, 0xFBFF, MAXH),              # 131008 - 65504: the product alone would overflow, the fused sum does not
    (0x8000, ONE, 0x8000, 0x8000),          # -0 * 1 + -0 = -0
    (0x0000, ONE, 0x8000, 0x0000),          # +0 * 1 + -0 = +0
    (0x8000, ONE, 0x0000, 0x0000),          # -0 * 1 + +0 = +0
    (ONE, ONE, MIN_SUB, ONE),               # 1 + 2^-24 rounds back to 1
    (MIN_SUB, HALF, MIN_SUB, 0x0002),       # 1.5 quanta ties to even: 2 quanta
    (MIN_NORMAL, MIN_NORMAL, 0x0000, 0x0000),   # 2^-28 underflows to +0
    (INF, 0x0000, ONE, QNAN),               # inf * 0 = NaN
    (INF, ONE, NINF, QNAN),                 # inf - inf = NaN
    (INF, TWO, ONE, INF),
    (ONE, ONE, QNAN, QNAN),
]

SPECIAL = [0x0000, 0x8000, ONE, 0xBC00, TWO, 0xC000, HALF, 0xB800, THREE, 0x3E00, MAXH, 0xFBFF, INF, NINF, QNAN,
           MIN_NORMAL, MIN_SUB, 0x8001, MAX_SUB, 0x3C01, 0x3BFF, 0xBC02, 0x6800, 0x6801, 0x4C00]


class FmaF16(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for a, b, c, want in CASES:
            self.assertTrue(same_f16(ref_fma(a, b, c), want), (hex(a), hex(b), hex(c), hex(ref_fma(a, b, c))))

    def test_vgpr_sources_write_low_half_and_keep_high_half(self):
        for a, b, c, want in CASES:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                got = run("v_fma_f16 v0, v1, v2, v3", {"v1": a, "v2": b, "v3": c})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_sgpr_and_literal_sources(self):
        for a, b, c, want in CASES[:12]:
            forms = [
                ("sgpr all", "v_fma_f16 v0, s1, s2, s3", {"s1": a, "s2": b, "s3": c}),
                ("sgpr src0 and src2", "v_fma_f16 v0, s1, v2, s3", {"s1": a, "v2": b, "s3": c}),
                ("literal src0", f"v_fma_f16 v0, {a:#x}, v2, v3", {"v2": b, "v3": c}),
                ("literal src2", f"v_fma_f16 v0, v1, v2, {c:#x}", {"v1": a, "v2": b}),
            ]
            for form, line, seeds in forms:
                with self.subTest(a=hex(a), b=hex(b), c=hex(c), form=form):
                    got = run(line, seeds)
                    self.assertEqual(got >> 16, OLD >> 16)
                    self.assertTrue(same_f16(got & 0xFFFF, want), hex(got))

    def test_inline_constants(self):
        # (src0 inline constant, its binary16 value, v1 = src1, v2 = src2, D), by hand
        for text, a, b, c, want in (("2.0", TWO, 0x3E00, ONE, 0x4400),          # 2 * 1.5 + 1 = 4
                                    ("-1.0", 0xBC00, HALF, HALF, 0x0000),       # -0.5 + 0.5
                                    ("0.5", HALF, 0x4400, 0xBC00, ONE),         # 0.5 * 4 - 1 = 1
                                    ("4.0", 0x4400, QUARTER, QUARTER, 0x3D00)):     # 4 * 0.25 + 0.25 = 1.25
            with self.subTest(const=text):
                self.assertEqual(run(f"v_fma_f16 v0, {text}, v1, v2", {"v1": b, "v2": c}) & 0xFFFF, want)
                self.assertEqual(ref_fma(a, b, c), want)
        self.assertEqual(run("v_fma_f16 v0, v1, v2, 1.0", {"v1": TWO, "v2": THREE}) & 0xFFFF, 0x4700)

    def test_op_sel_selects_source_and_destination_halves(self):
        # src0 = (lo 1.0, hi 3.0), src1 = (lo 0.5, hi -0.25), src2 = (lo 0.25, hi -1.0); D by hand for each
        # (src0 half, src1 half, src2 half): a * b + c.
        s0, s1, s2 = (THREE << 16) | ONE, (0xB400 << 16) | HALF, (0xBC00 << 16) | QUARTER
        expect_lo = {
            (0, 0, 0): 0x3A00,      # 1 * 0.5 + 0.25 = 0.75
            (0, 0, 1): 0xB800,      # 1 * 0.5 - 1 = -0.5
            (0, 1, 0): 0x0000,      # 1 * -0.25 + 0.25 = +0
            (0, 1, 1): 0xBD00,      # -0.25 - 1 = -1.25
            (1, 0, 0): 0x3F00,      # 3 * 0.5 + 0.25 = 1.75
            (1, 0, 1): 0x3800,      # 1.5 - 1 = 0.5
            (1, 1, 0): 0xB800,      # -0.75 + 0.25 = -0.5
            (1, 1, 1): 0xBF00,      # -0.75 - 1 = -1.75
        }
        for (h0, h1, h2), want in expect_lo.items():
            for dh in (0, 1):
                expect = ((want << 16) | (OLD & 0xFFFF)) if dh else ((OLD & 0xFFFF0000) | want)
                with self.subTest(op_sel=(h0, h1, h2, dh)):
                    got = run(f"v_fma_f16 v0, v1, v2, v3 op_sel:[{h0},{h1},{h2},{dh}]", {"v1": s0, "v2": s1, "v3": s2})
                    self.assertEqual(got, expect)

    def test_neg_and_abs_source_modifiers(self):
        # v1 = 2.0, v2 = 0.5, v3 = -0.25; (operand text, D) by hand
        for ops, want in (("v1, v2, v3", 0x3A00),                   # 1 - 0.25 = 0.75
                          ("-v1, v2, v3", 0xBD00),                  # -1 - 0.25 = -1.25
                          ("v1, -v2, v3", 0xBD00),                  # -1 - 0.25
                          ("|v1|, |v2|, |v3|", 0x3D00),             # 1 + 0.25
                          ("v1, v2, -v3", 0x3D00),                  # 1 + 0.25
                          ("v1, v2, -|v3|", 0x3A00),                # 1 - 0.25
                          ("-|v1|, v2, |v3|", 0xBA00)):             # -1 + 0.25 = -0.75
            with self.subTest(ops=ops):
                got = run(f"v_fma_f16 v0, {ops}", {"v1": TWO, "v2": HALF, "v3": 0xB400})
                self.assertEqual(got & 0xFFFF, want)

    def test_output_modifiers_and_clamp(self):
        # #4542, measured on gfx1036: an f16 destination ignores omod (mul:2/mul:4/div:2) under the game's MODE
        # (IEEE=0, f16 denormals kept) and keeps clamp (docs/design.md, f16 output modifiers).
        # unmodified result: 1.5 * 2 + -1 = 2.0; clamp limits it to [0, 1]
        for suffix, want in ((" mul:2", 0x4000), (" mul:4", 0x4000), (" div:2", 0x4000), (" clamp", ONE)):
            with self.subTest(mod=suffix):
                got = run("v_fma_f16 v0, v1, v2, v3" + suffix, {"v1": 0x3E00, "v2": TWO, "v3": 0xBC00})
                self.assertEqual(got & 0xFFFF, want)
        for (a, b, c), suffix, want in (((HALF, HALF, 0x0000), " clamp", QUARTER),      # inside [0, 1]: unchanged
                                        ((HALF, 0xB800, 0x0000), " clamp", 0x0000),     # -0.25 -> +0
                                        ((ONE, 0x3E00, 0x0000), " div:2 clamp", ONE),     # 1.5 -> 1
                                        ((0x3A00, ONE, 0x0000), " mul:2 clamp", 0x3A00)):  # 0.75, unscaled
            with self.subTest(a=hex(a), b=hex(b), c=hex(c), mod=suffix):
                got = run("v_fma_f16 v0, v1, v2, v3" + suffix, {"v1": a, "v2": b, "v3": c})
                self.assertEqual(got & 0xFFFF, want)
        for suffix, f in ((" mul:2", 2.0), (" mul:4", 4.0), (" div:2", 0.5)):
            got = run("v_fma_f16 v0, v1, v2, v3" + suffix, {"v1": 0x4900, "v2": 0xC100, "v3": 0x4C00})  # 10 * -2.5 + 16
            self.assertEqual(got & 0xFFFF, ref_fma(0x4900, 0xC100, 0x4C00))

    def test_random_operands(self):
        rng = random.Random(0xF3A16)
        pool = SPECIAL + [rng.getrandbits(16) for _ in range(40)]
        for _ in range(160):
            a, b, c = (rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(16) for _ in range(3))
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                got = run("v_fma_f16 v0, v1, v2, v3", {"v1": a, "v2": b, "v3": c})
                self.assertEqual(got >> 16, OLD >> 16)
                self.assertTrue(same_f16(got & 0xFFFF, ref_fma(a, b, c)), (hex(got), hex(ref_fma(a, b, c))))


if __name__ == "__main__":
    unittest.main()
