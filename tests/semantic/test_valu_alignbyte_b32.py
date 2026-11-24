"""v_alignbyte_b32 (RDNA2 VOP3): D = low 32 bits of ({S0, S1} >> (8 * (S2 & 3))), S0 the high dword.

Each case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
words are written by hand and also recomputed by a plain-Python reference. Standard library only: no GPU, no LLVM,
no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_alignbyte(s0, s1, s2):
    return ((((s0 & MASK32) << 32) | (s1 & MASK32)) >> (8 * (s2 & 3))) & MASK32


def run(s0, s1, s2, form="vgpr", dst=None):
    """dst: None for a fresh v0, or 0/1/2 to write the result over S0/S1/S2 (vgpr form) and export that register."""
    if dst is not None:
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {s2 & MASK32:#x}",
                 f"v_alignbyte_b32 v{dst + 1}, v1, v2, v3", f"exp mrt0 v{dst + 1}, v{dst + 1}, v{dst + 1}, v{dst + 1} done vm",
                 "s_endpgm"]
        return support.mrt0_words(support.translate_lines("ps_alignbyte", lines))[0]
    if form == "vgpr":
        setup = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {s2 & MASK32:#x}"]
        srcs = "v1, v2, v3"
    elif form == "sgpr":
        setup = [f"s_mov_b32 s1, {s0:#x}", f"s_mov_b32 s2, {s1:#x}", f"s_mov_b32 s3, {s2 & MASK32:#x}"]
        srcs = "s1, s2, s3"
    else:   # the shift as an inline constant
        setup = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}"]
        srcs = f"v1, v2, {s2}"
    lines = setup + [f"v_alignbyte_b32 v0, {srcs}", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_alignbyte", lines))[0]


# (name, S0, S1, S2, expected word), written by hand.
CASES = [
    ("shift 0 returns S1", 0xDEADBEEF, 0xCAFEBABE, 0, 0xCAFEBABE),
    ("shift 1", 0x88776655, 0x44332211, 1, 0x55443322),
    ("shift 2", 0x88776655, 0x44332211, 2, 0x66554433),
    ("shift 3", 0x88776655, 0x44332211, 3, 0x77665544),
    ("rotate idiom, shift 1", 0x12345678, 0x12345678, 1, 0x78123456),
    ("rotate idiom, shift 2", 0x12345678, 0x12345678, 2, 0x56781234),
    ("rotate idiom, shift 3", 0x12345678, 0x12345678, 3, 0x34567812),
    ("S0 = 0 is a logical shift right", 0, 0x89ABCDEF, 1, 0x0089ABCD),
    ("S1 = 0 is a shift left", 0x89ABCDEF, 0, 1, 0xEF000000),
    ("S1 = 0, shift 3", 0x89ABCDEF, 0, 3, 0xABCDEF00),
    ("top byte of S0 never reaches D", 0xFF000000, 0, 3, 0x00000000),
    ("shift uses only the low two bits (4+1)", 0x9ABCDEF0, 0x12345678, 5, 0xF0123456),
    ("shift uses only the low two bits (0xFFFFFFFC+2)", 0x9ABCDEF0, 0x12345678, 0xFFFFFFFE, 0xDEF01234),
    ("negative shift -1 is 3", 0x11223344, 0xAABBCCDD, 0xFFFFFFFF, 0x223344AA),
]


class AlignByte(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for name, s0, s1, s2, want in CASES:
            self.assertEqual(ref_alignbyte(s0, s1, s2), want, name)

    def test_translated_cases(self):
        for name, s0, s1, s2, want in CASES:
            with self.subTest(name):
                self.assertEqual(run(s0, s1, s2), want)

    def test_sgpr_and_inline_constant_shift(self):
        for name, s0, s1, s2, want in CASES[:11]:
            with self.subTest(name, form="sgpr"):
                self.assertEqual(run(s0, s1, s2, "sgpr"), want)
            with self.subTest(name, form="inline"):
                self.assertEqual(run(s0, s1, s2, "inline"), want)

    def test_inline_shift_above_three(self):
        # Immediate shifts 4 and 5 use only the low two bits (0 and 1); a mask of 7 would give other words.
        for shift, want in ((4, 0x12345678), (5, 0xF0123456), (6, 0xDEF01234), (7, 0xBCDEF012)):
            self.assertEqual(ref_alignbyte(0x9ABCDEF0, 0x12345678, shift), want)
            with self.subTest(shift=shift):
                self.assertEqual(run(0x9ABCDEF0, 0x12345678, shift, "inline"), want)

    def test_destination_overlaps_each_source(self):
        # Distinct operand values, so a destination that is cleared or overwritten before a source is read changes D.
        for name, s0, s1, s2, want in CASES:
            for dst, what in ((0, "S0"), (1, "S1"), (2, "S2 (the shift)")):
                with self.subTest(name, dst=what):
                    self.assertEqual(run(s0, s1, s2, dst=dst), want)

    def test_random_operands(self):
        rng = random.Random(0x616C676E)
        for _ in range(60):
            s0, s1, s2 = (rng.getrandbits(32) for _ in range(3))
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                self.assertEqual(run(s0, s1, s2), ref_alignbyte(s0, s1, s2))


if __name__ == "__main__":
    unittest.main()
