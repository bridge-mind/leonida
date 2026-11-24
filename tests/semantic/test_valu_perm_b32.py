"""v_perm_b32 (RDNA2 VOP3): result byte i is picked by selector byte i from the 8 bytes {S0, S1} (S1 is the low half).

    0..3  byte 0..3 of S1       4..7  byte 0..3 of S0
    8..11 0xFF or 0x00 from the sign bit of S1[15], S1[31], S0[15], S0[31] (bits 15, 31, 47, 63 of {S0, S1})
    12    0x00                  >=13  0xFF

Each case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
words are written by hand. (An earlier copy of this file checked a private model that picked other sign bits for
selectors 8..11 than the ISA and the translator do.) Standard library only: no GPU, no LLVM, no game data.
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


def sel(b0, b1, b2, b3):
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def ref_perm(s0, s1, selector):
    """Plain-Python reference of the table in the module docstring."""
    pool = [(s1 >> (8 * i)) & 0xFF for i in range(4)] + [(s0 >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        c = (selector >> (8 * i)) & 0xFF
        if c < 8:
            v = pool[c]
        elif c < 12:
            w, b = [(s1, 15), (s1, 31), (s0, 15), (s0, 31)][c - 8]
            v = 0xFF if (w >> b) & 1 else 0
        else:
            v = 0 if c == 12 else 0xFF
        out |= v << (8 * i)
    return out


def run(s0, s1, selector, form="vgpr", dst=None):
    """dst: None for a fresh v0, or 0/1/2 to write the result over S0/S1/the selector and export that register."""
    if dst is not None:
        lines = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {selector:#x}",
                 f"v_perm_b32 v{dst + 1}, v1, v2, v3", f"exp mrt0 v{dst + 1}, v{dst + 1}, v{dst + 1}, v{dst + 1} done vm",
                 "s_endpgm"]
        return support.mrt0_words(support.translate_lines("ps_perm", lines))[0]
    if form == "vgpr":
        setup = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {selector:#x}"]
        srcs = "v1, v2, v3"
    else:
        setup = [f"s_mov_b32 s1, {s0:#x}", f"s_mov_b32 s2, {s1:#x}", f"s_mov_b32 s3, {selector:#x}"]
        srcs = "s1, s2, s3"
    lines = setup + [f"v_perm_b32 v0, {srcs}", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_perm", lines))[0]


S0, S1 = 0xAABBCCDD, 0x11223344

# (name, S0, S1, selector, expected word), written by hand.
CASES = [
    ("identity S1", S0, S1, sel(0, 1, 2, 3), 0x11223344),
    ("identity S0", S0, S1, sel(4, 5, 6, 7), 0xAABBCCDD),
    ("interleave", S0, S1, sel(0, 4, 1, 5), 0xCC33DD44),
    ("bswap S1", 0x9ABCDEF0, 0x12345678, sel(3, 2, 1, 0), 0x78563412),
    ("bswap S0", 0x9ABCDEF0, 0x12345678, sel(7, 6, 5, 4), 0xF0DEBC9A),
    ("broadcast S1 byte 0", 0, 0x01020304, sel(0, 0, 0, 0), 0x04040404),
    ("broadcast S0 byte 3", 0x88776655, 0, sel(7, 7, 7, 7), 0x88888888),
    ("zero fill", 0xFFFFFFFF, 0xFFFFFFFF, sel(12, 12, 12, 12), 0x00000000),
    ("zero mixed", S0, S1, sel(0, 12, 4, 12), 0x00DD0044),
    ("ones fill 13..255", 0, 0, sel(13, 14, 15, 255), 0xFFFFFFFF),
    ("ones mixed", 0x12345678, 0x9ABCDEF0, sel(0, 13, 4, 255), 0xFF78FFF0),
    # 8: S1 bit 15, 9: S1 bit 31, 10: S0 bit 15, 11: S0 bit 31
    ("sel 8 set", 0, 0x00008000, sel(8, 12, 12, 12), 0x000000FF),
    ("sel 8 clear (S1 bit 31 does not count)", 0, 0x80000000, sel(8, 12, 12, 12), 0x00000000),
    ("sel 9 set", 0, 0x80000000, sel(12, 9, 12, 12), 0x0000FF00),
    ("sel 9 clear", 0, 0x7FFFFFFF, sel(12, 9, 12, 12), 0x00000000),
    ("sel 10 set", 0x00008000, 0, sel(12, 12, 10, 12), 0x00FF0000),
    ("sel 10 clear (S0 bit 31 does not count)", 0x80000000, 0, sel(12, 12, 10, 12), 0x00000000),
    ("sel 11 set", 0x80000000, 0, sel(12, 12, 12, 11), 0xFF000000),
    ("sel 11 clear (S1 bit 7 does not count)", 0, 0x00000080, sel(12, 12, 12, 11), 0x00000000),
    ("all four sign bits set", 0x80008000, 0x80008000, sel(8, 9, 10, 11), 0xFFFFFFFF),
    ("selector bytes use all 8 bits", S0, S1, sel(0x10, 0x7F, 0x80, 0xFF), 0xFFFFFFFF),
]


class PermB32(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for name, s0, s1, selector, want in CASES:
            self.assertEqual(ref_perm(s0, s1, selector), want, name)

    def test_translated_cases(self):
        for name, s0, s1, selector, want in CASES:
            with self.subTest(name):
                self.assertEqual(run(s0, s1, selector), want)

    def test_sgpr_sources(self):
        for name, s0, s1, selector, want in CASES[:8]:
            with self.subTest(name):
                self.assertEqual(run(s0, s1, selector, "sgpr"), want)

    def test_destination_overlaps_each_source(self):
        # The destination is S0, S1 or the selector: every source must be read before the destination is written.
        for name, s0, s1, selector, want in CASES:
            for dst, what in ((0, "S0"), (1, "S1"), (2, "selector")):
                with self.subTest(name, dst=what):
                    self.assertEqual(run(s0, s1, selector, dst=dst), want)

    def test_random_operands(self):
        rng = random.Random(0x7065726D)
        for _ in range(60):
            s0, s1, selector = (rng.getrandbits(32) for _ in range(3))
            with self.subTest(s0=hex(s0), s1=hex(s1), selector=hex(selector)):
                self.assertEqual(run(s0, s1, selector), ref_perm(s0, s1, selector))


if __name__ == "__main__":
    unittest.main()
