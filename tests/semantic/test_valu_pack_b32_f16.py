"""v_pack_b32_f16 (RDNA2 VOP3): D = (S1.half << 16) | S0.half, where op_sel[0] / op_sel[1] pick the low (0) or high (1)
16-bit half of S0 / S1. All 32 bits of the destination are written (nothing of the old value survives). The halves are
moved as bits: a VOP3 abs clears bit 15 of the half, neg flips it, abs first (docs/design.md, `v_pack_b32_f16` source
modifiers; measured on gfx1036 with IEEE mode 0, where a signalling NaN keeps its payload).

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand next to each case, then compared against a few lines of plain Python, with SGPR / literal /
inline-constant sources, op_sel and seeded random operands. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f16_unary_cases import run as _run  # noqa: E402

LINE = "v_pack_b32_f16 v0, v1, v2"


def run(line, seeds):
    """The destination starts as 0xdeadbeef; seeds are whole 32-bit words (no filler in the high half)."""
    return _run(line, seeds, hi=0)


def half(w, sel, ab=False, ng=False):
    h = (w >> (16 * sel)) & 0xFFFF
    if ab:
        h &= 0x7FFF
    if ng:
        h ^= 0x8000
    return h


def ref(s0, s1, sel=(0, 0), mods=((False, False), (False, False))):
    return half(s0, sel[0], *mods[0]) | (half(s1, sel[1], *mods[1]) << 16)


# (src0 dword, src1 dword, D with op_sel [0,0]), by hand: D = (S1[15:0] << 16) | S0[15:0]
CASES = [
    (0x00003C00, 0x00004000, 0x40003C00),                   # 1.0, 2.0
    (0xAAAA3C00, 0xBBBB4000, 0x40003C00),                   # the high halves of the sources are ignored
    (0x00000000, 0x00008000, 0x80000000),                   # +0, -0
    (0x00008000, 0x00000000, 0x00008000),
    (0x00007C00, 0x0000FC00, 0xFC007C00),                   # +inf, -inf
    (0x00007E00, 0x00007E01, 0x7E017E00),                   # quiet NaNs keep their bits
    (0x00007C01, 0x0000FC01, 0xFC017C01),                   # signalling NaNs are moved, not quieted (IEEE mode 0)
    (0x00000001, 0x000003FF, 0x03FF0001),                   # subnormals
    (0x00007BFF, 0x0000FBFF, 0xFBFF7BFF),                   # +-65504
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    (0x12345678, 0x9ABCDEF0, 0xDEF05678),
]

SPECIAL = [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFC01, 0x7DFF, 0x0001, 0x03FF,
           0x0400, 0x7BFF, 0xFBFF, 0xFFFF, 0x5A5A]
# Operand words whose halves differ, so op_sel and the modifiers show: low 1.5 / high -3.0, low sNaN / high -0, ...
MIXED = [0xC2003E00, 0x80007C01, 0x7E00FC01, 0x03FF8001, 0x3C000000]


class PackB32F16(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for a, b, want in CASES:
            self.assertEqual(ref(a, b), want, (hex(a), hex(b)))

    def test_vgpr_pair_writes_the_whole_destination(self):
        for a, b, want in CASES:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(run(LINE, {"v1": a, "v2": b}), want)       # OLD = 0xdeadbeef leaves no trace

    def test_sgpr_and_literal_sources(self):
        for a, b, want in CASES:
            forms = [
                ("sgpr sgpr", "v_pack_b32_f16 v0, s1, s2", {"s1": a, "s2": b}),
                ("sgpr vgpr", "v_pack_b32_f16 v0, s1, v2", {"s1": a, "v2": b}),
                ("vgpr sgpr", "v_pack_b32_f16 v0, v1, s2", {"v1": a, "s2": b}),
                ("literal src0", f"v_pack_b32_f16 v0, {a:#x}, v2", {"v2": b}),
                ("literal src1", f"v_pack_b32_f16 v0, v1, {b:#x}", {"v1": a}),
            ]
            for form, line, seeds in forms:
                with self.subTest(a=hex(a), b=hex(b), form=form):
                    self.assertEqual(run(line, seeds), want)

    def test_inline_constants_are_f16_in_the_low_half_over_a_zero_high_half(self):
        # docs/design.md "f16 inline constants": 1.0 is 0x3c00, not the f32 pattern; its high half reads +0.0
        for text, c in (("1.0", 0x3C00), ("-1.0", 0xBC00), ("0.5", 0x3800), ("2.0", 0x4000), ("-4.0", 0xC400),
                        ("0", 0), ("1", 1), ("64", 64)):
            with self.subTest(const=text):
                self.assertEqual(run(f"v_pack_b32_f16 v0, {text}, v2", {"v2": 0x4400}), (0x4400 << 16) | c)
                self.assertEqual(run(f"v_pack_b32_f16 v0, v1, {text}", {"v1": 0x4400}), (c << 16) | 0x4400)
        self.assertEqual(run("v_pack_b32_f16 v0, 1.0, 2.0", {}), 0x40003C00)

    def test_op_sel_selects_each_source_half(self):
        for a in MIXED:
            for b in MIXED:
                for s0 in (0, 1):
                    for s1 in (0, 1):
                        with self.subTest(a=hex(a), b=hex(b), op_sel=(s0, s1)):
                            got = run(f"v_pack_b32_f16 v0, v1, v2 op_sel:[{s0},{s1},0]", {"v1": a, "v2": b})
                            self.assertEqual(got, ref(a, b, (s0, s1)))

    def test_op_sel_by_hand(self):
        a, b = (0xC200 << 16) | 0x3E00, (0x8000 << 16) | 0x7C01          # a: 1.5 | -3.0, b: sNaN | -0
        for sel, want in (((0, 0), 0x7C013E00), ((1, 0), 0x7C01C200), ((0, 1), 0x80003E00), ((1, 1), 0x8000C200)):
            with self.subTest(op_sel=sel):
                self.assertEqual(run(f"v_pack_b32_f16 v0, v1, v2 op_sel:[{sel[0]},{sel[1]},0]", {"v1": a, "v2": b}), want)

    def test_source_modifiers_act_on_the_selected_half_as_bit_moves(self):
        # a = low 1.5 (0x3e00) / high -3.0 (0xc200), b = low sNaN (0x7c01) / high -0 (0x8000); D by hand
        a, b = 0xC2003E00, 0x80007C01
        cases = (
            ("-v1, v2", (0, 0), 0x7C01BE00), ("|v1|, v2", (0, 0), 0x7C013E00), ("-|v1|, v2", (0, 0), 0x7C01BE00),
            ("v1, -v2", (0, 0), 0xFC013E00), ("v1, |v2|", (0, 0), 0x7C013E00),
            ("|v1|, |v2|", (1, 1), 0x00004200),        # |-3.0| = 0x4200 in the low half, |-0| = +0 in the high half
            ("-v1, -v2", (1, 1), 0x0000_4200),          # -(-3.0) = 0x4200, -(-0) = +0
            ("-v1, -v2", (0, 1), 0x0000000 | 0xBE00),   # -(1.5) = 0xbe00, -(-0) = +0
            ("-|v1|, -|v2|", (1, 0), 0xFC01C200),       # bit 15 set on both: 0xc200 and 0xfc01
        )
        for ops, sel, want in cases:
            with self.subTest(ops=ops, op_sel=sel):
                got = run(f"v_pack_b32_f16 v0, {ops} op_sel:[{sel[0]},{sel[1]},0]", {"v1": a, "v2": b})
                self.assertEqual(got, want)

    def test_modifier_grid_against_reference(self):
        flags = [(False, False), (True, False), (False, True), (True, True)]
        for w0 in (0x7C01FC01, 0xC2003E00, 0x03FF8001):
            for w1 in (0x80007E00, 0x00007C01):
                for m0 in flags:
                    for m1 in flags:
                        for sel in ((0, 0), (1, 1), (1, 0)):
                            ops = []
                            for r, (ab, ng) in (("v1", m0), ("v2", m1)):
                                ops.append(("-" if ng else "") + (f"|{r}|" if ab else r))
                            with self.subTest(w0=hex(w0), w1=hex(w1), ops=ops, op_sel=sel):
                                got = run(f"v_pack_b32_f16 v0, {ops[0]}, {ops[1]} op_sel:[{sel[0]},{sel[1]},0]",
                                          {"v1": w0, "v2": w1})
                                self.assertEqual(got, ref(w0, w1, sel, (m0, m1)))

    def test_same_register_twice_and_destination_aliasing_a_source(self):
        self.assertEqual(run("v_pack_b32_f16 v0, v1, v1", {"v1": 0xC2003E00}), 0x3E003E00)
        self.assertEqual(run("v_pack_b32_f16 v0, v1, v1 op_sel:[1,0,0]", {"v1": 0xC2003E00}), 0x3E00C200)
        # v0 = (-3.0 | 1.5) read and written by the same instruction: the halves swap
        self.assertEqual(run("v_pack_b32_f16 v0, v0, v0 op_sel:[1,0,0]", {"v0": 0xC2003E00}), 0x3E00C200)

    def test_random_operands(self):
        rng = random.Random(0x9AC4B32)
        pool = SPECIAL + [rng.getrandbits(16) for _ in range(30)]
        for _ in range(120):
            a = (rng.choice(pool) << 16 | rng.choice(pool)) if rng.random() < 0.5 else rng.getrandbits(32)
            b = (rng.choice(pool) << 16 | rng.choice(pool)) if rng.random() < 0.5 else rng.getrandbits(32)
            sel = (rng.getrandbits(1), rng.getrandbits(1))
            m = ((rng.getrandbits(1) == 1, rng.getrandbits(1) == 1), (rng.getrandbits(1) == 1, rng.getrandbits(1) == 1))
            ops = [("-" if ng else "") + (f"|{r}|" if ab else r) for r, (ab, ng) in (("v1", m[0]), ("v2", m[1]))]
            with self.subTest(a=hex(a), b=hex(b), sel=sel, ops=ops):
                got = run(f"v_pack_b32_f16 v0, {ops[0]}, {ops[1]} op_sel:[{sel[0]},{sel[1]},0]", {"v1": a, "v2": b})
                self.assertEqual(got, ref(a, b, sel, m))


if __name__ == "__main__":
    unittest.main()
