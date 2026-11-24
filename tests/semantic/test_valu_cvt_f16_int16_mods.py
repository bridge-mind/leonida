"""Value tests for v_cvt_f16_{u16,i16} output modifiers (clamp, mul:2, mul:4, div:2) and SDWA clamp.

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/semantic/spirv_mini_interp.py.
Standard library only.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)
import rdna2spirv  # noqa: E402


def run_ps(lines):
    return mrt0_words(translate_lines("ps_cvt_f16_mods", lines))


class CvtF16Int16OutputMods(unittest.TestCase):
    def test_vop3_clamp_u16_and_i16(self):
        words = run_ps([
            "v_mov_b32 v0, 0xdead0005",
            "v_cvt_f16_u16 v0, v0 clamp",
            "v_mov_b32 v1, 0xbeef0000",
            "v_cvt_f16_u16 v1, v1 clamp",
            "v_mov_b32 v2, 0xcafeffff",
            "v_cvt_f16_i16 v2, v2 clamp",
            "v_mov_b32 v3, 0x0003abcd",
            "v_cvt_f16_i16_sdwa v3, v3 clamp dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words[0], 0xDEAD3C00)  # 5 -> clamped to 1.0 (0x3c00), high half kept
        self.assertEqual(words[1], 0xBEEF0000)  # 0 -> 0.0
        self.assertEqual(words[2], 0xCAFE0000)  # -1 -> clamped to +0.0
        self.assertEqual(words[3], 0x00033C00)  # SDWA src0_sel:WORD_1 (3) -> clamped to 1.0 in WORD_0, WORD_1 kept

    def test_vop3_omod_scaling_and_clamp(self):
        words = run_ps([
            "v_mov_b32 v0, 3",
            "v_cvt_f16_u16 v0, v0 mul:2",
            "v_mov_b32 v1, 0xffff",
            "v_cvt_f16_i16 v1, v1 mul:4",
            "v_mov_b32 v2, 3",
            "v_cvt_f16_i16 v2, v2 div:2",
            "v_mov_b32 v3, 1",
            "v_cvt_f16_u16 v3, v3 clamp div:2",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        # Under default MODE (denorm16_64=3), RDNA2 silicon ignores omod on f16 destinations while clamp applies.
        self.assertEqual(words[0] & 0xFFFF, 0x4200)  # 3.0 (omod ignored)
        self.assertEqual(words[1] & 0xFFFF, 0xBC00)  # -1.0 (omod ignored)
        self.assertEqual(words[2] & 0xFFFF, 0x4200)  # 3.0 (omod ignored)
        self.assertEqual(words[3] & 0xFFFF, 0x3C00)  # 1.0 (clamped, omod ignored)

    def test_sdwa_clamp(self):
        words = run_ps([
            "v_mov_b32 v0, 0x1234fffe",
            "v_cvt_f16_i16_sdwa v0, v0 clamp dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0",
            "v_mov_b32 v1, 0x00020000",
            "v_cvt_f16_u16_sdwa v1, v1 clamp dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ])
        self.assertEqual(words[0], 0x0000FFFE)  # -2 clamped to +0.0 in WORD_1, WORD_0 preserved
        self.assertEqual(words[1], 0x00003C00)  # 2 clamped to 1.0 in WORD_0, WORD_1 padded with 0

    def test_f16_to_int16_output_modifiers_still_rejected(self):
        for op in ("v_cvt_u16_f16", "v_cvt_i16_f16"):
            with self.subTest(op=op), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines("ps_cvt_f16_reject", [f"{op} v0, v0 clamp", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
