"""v_sad_u32 (RDNA2 VOP3): D = |S0 - S1| + S2 on unsigned 32-bit operands, modulo 2^32.

Each case is translated by tools/rdna2spirv.py (support.translate_lines) and executed with
tests/semantic/spirv_mini_interp.py; the expected words are written by hand from the ISA formula and also recomputed
by a plain-Python reference, with VGPR, SGPR and literal/inline-constant sources and seeded random operands.
Standard library only: no GPU, no LLVM, no game data. (The old file modelled the instruction in its own classes and
never ran the translator. The `clamp` modifier is the business of test_sad_clamp.py / test_alu_sad_u32.py.)
"""
import os
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

M32 = 0xFFFFFFFF


def ref(s0, s1, s2):
    return (abs(s0 - s1) + s2) & M32


def run(s0, s1, s2, form="vgpr"):
    if form == "vgpr":
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {s2:#x}", "v_sad_u32 v0, v1, v2, v3"]
    elif form == "sgpr":
        lines = [f"s_mov_b32 s1, {s0:#x}", f"s_mov_b32 s2, {s1:#x}", f"s_mov_b32 s3, {s2:#x}", "v_sad_u32 v0, s1, s2, s3"]
    elif form == "literal":      # one literal (S1) between VGPRs: the VOP3 literal slot of GFX10
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v3, {s2:#x}", f"v_sad_u32 v0, v1, {s1:#x}, v3"]
    else:
        raise ValueError(form)
    lines += ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_sad_u32", lines))[0]


# (S0, S1, S2, expected word), written by hand.
CASES = [
    (5, 3, 0, 2), (3, 5, 0, 2), (7, 7, 0, 0), (10, 4, 100, 106), (4, 10, 100, 106),
    (0, 0xFFFFFFFF, 0, 0xFFFFFFFF), (0xFFFFFFFF, 0, 0, 0xFFFFFFFF),
    (0x80000000, 0x7FFFFFFF, 0, 1),                 # unsigned: not a signed wrap
    (0, 0x80000000, 0, 0x80000000),
    (0xFFFFFFFF, 0, 1, 0),                          # 2^32 - 1 + 1 wraps
    (0xFFFFFFFF, 0, 0xFFFFFFFF, 0xFFFFFFFE),
    (0x12345678, 0x12345600, 0x10, 0x88),
    (0x00000001, 0xFFFFFFFF, 0x00000003, 0x00000001),   # 0xFFFFFFFE + 3 mod 2^32
]


class SadU32(unittest.TestCase):
    def test_hand_table_matches_reference(self):
        for s0, s1, s2, want in CASES:
            self.assertEqual(ref(s0, s1, s2), want, (hex(s0), hex(s1), hex(s2)))

    def test_translated_forms(self):
        for form in ("vgpr", "sgpr", "literal"):
            for s0, s1, s2, want in CASES:
                with self.subTest(form=form, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(run(s0, s1, s2, form), want)

    def test_symmetric_in_s0_s1(self):
        rng = random.Random(0x5AD)
        for _ in range(20):
            a, b, c = (rng.getrandbits(32) for _ in range(3))
            self.assertEqual(run(a, b, c), run(b, a, c))

    def test_random_operands(self):
        rng = random.Random(0x5AD32)
        for _ in range(60):
            s0, s1, s2 = (rng.getrandbits(32) for _ in range(3))
            for form in ("vgpr", "sgpr", "literal"):
                with self.subTest(form=form, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(run(s0, s1, s2, form), ref(s0, s1, s2))


if __name__ == "__main__":
    unittest.main()
