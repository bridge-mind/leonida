"""Value and lowering checks for known v_bfm_b32 operands."""

from pathlib import Path
import unittest

from support import HERE, mrt0_words, translate_listing, translate_lines


class KnownBfmOperands(unittest.TestCase):
    def test_known_width_and_offset(self):
        fixture = Path(HERE) / "fixtures" / "bfm_known_operands.asm"
        text = translate_listing("ps_bfm_known_operands", fixture.read_bytes())
        self.assertEqual(mrt0_words(text), [0x3F800008, 0xBF800000, 0x3F800000, 0x3F800038])
        self.assertEqual(text.count("OpShiftLeftLogical %uint"), 1)
        self.assertEqual(text.count("OpISub %uint"), 0)
        self.assertEqual(text.count("OpBitwiseAnd %uint"), 1)

    def test_known_offset_with_runtime_width(self):
        text = translate_lines("ps_bfm_known_offset", [
            "v_mov_b32 v0, 35",
            "v_bfm_b32 v1, v0, 32",
            "v_bfm_b32 v2, v0, 35",
            "v_bfm_b32 v3, v0, v0",
            "exp mrt0 v1, v2, v3, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(text), [7, 56, 56, 35])
        self.assertEqual(text.count("OpShiftLeftLogical %uint"), 5)
        self.assertEqual(text.count("OpBitwiseAnd %uint"), 4)
