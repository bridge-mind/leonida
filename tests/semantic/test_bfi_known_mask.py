"""Known v_bfi_b32 masks use a shorter lowering with the same result words."""

from pathlib import Path
import unittest

from support import HERE, mrt0_words, translate_listing, translate_lines


class KnownBfiMask(unittest.TestCase):
    def test_literal_and_sgpr_masks(self):
        fixture = Path(HERE) / "fixtures" / "bfi_known_mask.asm"
        text = translate_listing("ps_bfi_known_mask", fixture.read_bytes())
        self.assertEqual(mrt0_words(text), [0x3F123456, 0x4055AA7E, 0x3F553456, 0x3F523476])
        self.assertEqual(text.count("OpBitFieldInsert %uint"), 1)
        self.assertNotIn("OpNot %uint", text)
        self.assertEqual(text.count("OpBitwiseOr %uint"), 1)

    def test_contiguous_masks_at_both_ends(self):
        text = translate_lines("ps_bfi_mask_edges", [
            "v_mov_b32 v0, 0xc055aa7f",
            "v_mov_b32 v1, 0x3f123456",
            "v_bfi_b32 v2, 1, v0, v1",
            "v_bfi_b32 v3, 0x80000000, v0, v1",
            "v_bfi_b32 v4, 0x7fffffff, v0, v1",
            "v_bfi_b32 v5, 0xfffffffe, v0, v1",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(text), [0x3F123457, 0xBF123456, 0x4055AA7F, 0xC055AA7E])
        self.assertEqual(text.count("OpBitFieldInsert %uint"), 4)
        self.assertEqual(text.count("OpShiftRightLogical %uint"), 2)

    def test_sgpr_overwritten_by_dynamic_source(self):
        text = translate_lines("ps_bfi_mask_overwritten", [
            "s_mov_b32 s8, -1",
            "s_mov_b32 m0, 0x00f000f0",
            "s_mov_b32 s8, m0",
            "v_mov_b32 v1, 0x4055aa7e",
            "v_mov_b32 v2, 0x3f123456",
            "v_bfi_b32 v2, s8, v1, v2",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(text), [0x3F523476] * 4)
        self.assertIn("OpNot %uint", text)
        self.assertNotIn("OpBitFieldInsert %uint", text)
