"""Translator tests for the scalar bitfield-insert sequence.

RDNA2's SALU has no bitfield-insert instruction; shaders insert `insert` into `base` at [offset +: width] with

    s_bfm_b32   mask,  width, offset     mask  = ((1 << (width & 31)) - 1) << (offset & 31), SCC unchanged
    s_lshl_b32  field, insert, offset    field = insert << (offset & 31)
    s_and_b32   field, field, mask
    s_andn2_b32 kept,  base, mask        kept  = base & ~mask
    s_or_b32    dst,   kept, field       SCC   = dst != 0

Each case translates that listing through tools/rdna2spirv.py and executes it in tests/semantic/spirv_mini_interp.py,
with the operands as push constants (run-time values) and as s_mov_b32 immediates (folded at translation time).
Standard library only; no GPU, no game files.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_insert(base, insert, offset, width):
    """(dst, scc, mask) of the sequence above, from the RDNA2 ISA pseudocode of each instruction."""
    mask = (((1 << (width & 31)) - 1) << (offset & 31)) & MASK32
    field = ((insert << (offset & 31)) & MASK32) & mask
    dst = (base & ~mask & MASK32) | field
    return dst, 1 if dst else 0, mask


SEQUENCE = [
    "s_bfm_b32 s4, s2, s1",
    "s_lshl_b32 s5, s3, s1",
    "s_and_b32 s5, s5, s4",
    "s_andn2_b32 s6, s0, s4",
    "s_or_b32 s7, s6, s5",
]
EXPORT = [
    "s_cselect_b32 s8, 1, 0",
    "v_mov_b32 v0, s7",
    "v_mov_b32 v1, s8",
    "v_mov_b32 v2, s4",
    "v_mov_b32 v3, 0",
    "exp mrt0 v0, v1, v2, v3 done vm",
    "s_endpgm",
]

OFFSETS = (0, 1, 7, 16, 31, 33)
WIDTHS = (0, 1, 5, 16, 31, 32, 40)
VALUES = (0, MASK32, 0xA5A5A5A5, 0x12345678)


class BitfieldInsertSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = support.translate_lines("ps_salu_bitfield_insert", SEQUENCE + EXPORT)

    def run_case(self, base, insert, offset, width):
        out = support.mrt0_words(self.spv, push_constants=[base, offset, width, insert])
        return out[0], out[1], out[2]

    def test_reference_examples(self):
        self.assertEqual(ref_insert(0xFFFFFFFF, 0, 4, 8), (0xFFFFF00F, 1, 0x00000FF0))
        self.assertEqual(ref_insert(0, 0xFF, 28, 8), (0xF0000000, 1, 0xF0000000))
        self.assertEqual(ref_insert(0x12345678, 0xABCD, 8, 0), (0x12345678, 1, 0))
        self.assertEqual(ref_insert(0, 0, 0, 31), (0, 0, 0x7FFFFFFF))

    def test_offset_width_grid(self):
        for offset in OFFSETS:
            for width in WIDTHS:
                for base in VALUES:
                    for insert in VALUES:
                        with self.subTest(base=hex(base), insert=hex(insert), offset=offset, width=width):
                            self.assertEqual(self.run_case(base, insert, offset, width),
                                             ref_insert(base, insert, offset, width))

    def test_scc_is_zero_only_for_a_zero_result(self):
        # base cleared by the mask and nothing inserted: dst == 0, SCC == 0
        self.assertEqual(self.run_case(0x0000FF00, 0, 8, 8), (0, 0, 0x0000FF00))
        # s_bfm_b32 leaves SCC alone, so SCC is s_or_b32's even when the mask is zero
        self.assertEqual(self.run_case(0, 0xFFFFFFFF, 3, 0), (0, 0, 0))
        self.assertEqual(self.run_case(1, 0, 3, 0), (1, 1, 0))

    def test_seeded_random(self):
        rng = random.Random(0xBF1)
        for i in range(200):
            base, insert = rng.getrandbits(32), rng.getrandbits(32)
            offset, width = rng.getrandbits(6), rng.getrandbits(6)
            with self.subTest(case=i):
                self.assertEqual(self.run_case(base, insert, offset, width),
                                 ref_insert(base, insert, offset, width))

    def test_constant_operands_fold(self):
        cases = (
            (0xFFFFFFFF, 0, 4, 8),
            (0x12345678, 0xABCD, 8, 16),
            (0x0000FF00, 0, 8, 8),
            (0xDEADBEEF, 0x5, 30, 4),
            (0x80000000, 0x7FFFFFFF, 1, 31),
        )
        for base, insert, offset, width in cases:
            with self.subTest(base=hex(base), insert=hex(insert), offset=offset, width=width):
                movs = [f"s_mov_b32 s0, {base:#x}", f"s_mov_b32 s1, {offset}", f"s_mov_b32 s2, {width}",
                        f"s_mov_b32 s3, {insert:#x}"]
                spv = support.translate_lines("ps_salu_bitfield_insert_const", movs + SEQUENCE + EXPORT)
                self.assertEqual(tuple(support.mrt0_words(spv)[:3]), ref_insert(base, insert, offset, width))

    def test_destination_aliases_base(self):
        # dst written over base, as compilers emit it in place
        spv = support.translate_lines("ps_salu_bitfield_insert_alias", SEQUENCE[:3] + [
            "s_andn2_b32 s0, s0, s4",
            "s_or_b32 s0, s0, s5",
            "s_cselect_b32 s8, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s8",
            "v_mov_b32 v2, s4",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        for base, insert, offset, width in ((0xFFFFFFFF, 0x3, 4, 2), (0x0F0F0F0F, 0xF, 28, 4)):
            with self.subTest(base=hex(base)):
                out = support.mrt0_words(spv, push_constants=[base, offset, width, insert])
                self.assertEqual(tuple(out[:3]), ref_insert(base, insert, offset, width))


if __name__ == "__main__":
    unittest.main()
