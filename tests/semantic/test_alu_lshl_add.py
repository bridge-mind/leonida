"""v_lshl_add_u32 (RDNA2 VOP3) through tools/rdna2spirv.py; v_lshl_add_u64 is pinned as Unsupported.

The previous version of this file ran a private Python model of both instructions and never called the translator
(it only fed the interpreter hand-written SPIR-V), so it passed whatever the translator did.  Every case here is a
synthetic one-lane pixel shader translated by the translator and executed by tests/semantic/spirv_mini_interp.py.

ISA (RDNA2): v_lshl_add_u32 D.u32 = (S0.u32 << S1.u32[4:0]) + S2.u32, wrapping at 32 bits.  The shift count is the low
five bits of S1, so a count of 32 shifts by 0.  v_lshl_add_u64 does not exist on RDNA2 (llvm-mc -mcpu=gfx1030: "instruction
not supported on this GPU"; it appeared with gfx940/gfx12), so the translator rejecting it by name is correct; the
test fails if that ever changes, so the new lowering gets a real test.
Standard library only: no GPU, no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF


def ref(a, b, c):
    return (((a & M32) << (b & 31)) + (c & M32)) & M32


def run(a, b, c, forms=("v", "v", "v"), dst="v0"):
    """v_lshl_add_u32 on one lane. Each source form is v (VGPR), s (SGPR), i (inline constant or literal)."""
    setup, srcs = [], []
    for i, (val, form) in enumerate(zip((a, b, c), forms), start=1):
        if form == "v":
            setup.append(f"v_mov_b32 v{i}, {val & M32:#x}")
            srcs.append(f"v{i}")
        elif form == "s":
            setup.append(f"s_mov_b32 s{i}, {val & M32:#x}")
            srcs.append(f"s{i}")
        else:
            srcs.append(f"{val & M32:#x}")
    lines = setup + [f"v_lshl_add_u32 {dst}, {', '.join(srcs)}", f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_lshl_add", lines))[0]


# (S0, S1, S2, D), written by hand from (S0 << (S1 & 31)) + S2 mod 2^32
CASES = [
    (1, 0, 0, 0x00000001),
    (1, 4, 0, 0x00000010),
    (1, 31, 0, 0x80000000),
    (1, 32, 5, 0x00000006),                 # the count wraps: 32 & 31 == 0
    (1, 33, 0, 0x00000002),
    (0xFFFFFFFF, 1, 1, 0xFFFFFFFF),         # 0xFFFFFFFE + 1
    (0x80000000, 1, 0, 0x00000000),         # the shifted-out bit is lost
    (0x80000001, 31, 0x80000000, 0x00000000),   # 0x80000000 + 0x80000000 wraps
    (0x12345678, 8, 0x000000AB, 0x345678AB),
    (0x0000FFFF, 16, 0xFFFF0000, 0xFFFE0000),   # 0xFFFF0000 + 0xFFFF0000 wraps
    (0xDEADBEEF, 0xFFFFFFE4, 0x11111111, 0xFBED0001),   # count 0xFFFFFFE4 & 31 = 4
    (7, 3, 0xFFFFFFF8, 0x00000030),
]


class LshlAddU32(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for a, b, c, d in CASES:
            self.assertEqual(ref(a, b, c), d, (hex(a), hex(b), hex(c)))

    def test_translated_cases(self):
        for a, b, c, d in CASES:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(run(a, b, c), d)

    def test_every_shift_count(self):
        for n in range(0, 40):
            with self.subTest(n=n):
                self.assertEqual(run(0x80000001, n, 0x01010101), ref(0x80000001, n, 0x01010101))

    def test_sgpr_inline_and_literal_sources(self):
        for a, b, c, d in CASES[:8]:
            for forms in (("s", "v", "v"), ("v", "s", "v"), ("v", "v", "s"), ("s", "s", "s"), ("v", "i", "v"),
                          ("v", "v", "i"), ("i", "v", "v")):
                with self.subTest(a=hex(a), b=hex(b), c=hex(c), forms="".join(forms)):
                    self.assertEqual(run(a, b, c, forms), d)

    def test_destination_may_alias_a_source(self):
        # v1 = v_lshl_add_u32 v1, v2, v3 reads every source before writing.
        lines = ["v_mov_b32 v1, 0x12345678", "v_mov_b32 v2, 8", "v_mov_b32 v3, 0xab",
                 "v_lshl_add_u32 v1, v1, v2, v3", "v_lshl_add_u32 v2, v2, v2, v2",
                 "exp mrt0 v1, v2, v1, v2 done vm", "s_endpgm"]
        out = support.mrt0_words(support.translate_lines("ps_lshl_add_alias", lines))
        self.assertEqual(out[0], 0x345678AB)
        self.assertEqual(out[1], ref(8, 8, 8))      # (8 << 8) + 8

    def test_random_operands(self):
        rng = random.Random(0x6C73686C)
        for _ in range(80):
            a, b, c = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(run(a, b, c), ref(a, b, c))

    def test_a_shift_is_not_a_rotate_and_the_add_is_not_an_or(self):
        self.assertEqual(run(0x80000000, 1, 0), 0)                 # not a rotate (a rotate would give 1)
        self.assertEqual(run(1, 1, 2), 4)                          # 2 + 2 = 4, an OR would give 2


class LshlAddU64Unsupported(unittest.TestCase):
    def test_refused_by_name(self):
        for ins in ("v_lshl_add_u64 v[0:1], v[2:3], v4, v[6:7]", "v_lshl_add_u64 v[0:1], v[0:1], 3, v[0:1]",
                    "v_lshl_add_u64 v[0:1], s[2:3], v4, 0"):
            with self.subTest(ins=ins):
                lines = [ins, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_lshl_add_u64", lines)
                self.assertEqual(str(refused.exception), "v_lshl_add_u64")


if __name__ == "__main__":
    unittest.main()
