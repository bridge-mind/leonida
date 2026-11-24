"""v_sin_f32/v_cos_f32: range reduction in turns, not after scaling by 2*pi.

The hardware takes the operand in turns and reduces to [0, 1) before evaluating. Scaling first loses bits for large
inputs and leaves Sin outside [-pi, pi], where Vulkan's precision is not guaranteed. The old translation is up to
1e-4 off near 255 turns; reducing first is within an f32 ulp of the double result.
"""
import math
import struct
import unittest

import support


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


class SinCosTurns(unittest.TestCase):
    def test_large_turns_match_the_turn_reduced_reference(self):
        x, y = 100.3, 254.51
        lines = [
            f"v_mov_b32 v0, 0x{_bits(x):08x}",
            "v_sin_f32 v1, v0",
            "v_cos_f32 v2, v0",
            f"v_mov_b32 v3, 0x{_bits(y):08x}",
            "v_sin_f32 v4, v3",
            "v_cos_f32 v5, v3",
            "exp mrt0 v1, v2, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sin_cos_turns", lines)
        got = [_float(w) for w in support.mrt0_words(spv)]
        xf, yf = _float(_bits(x)), _float(_bits(y))   # the shader sees the binary32 input, not the decimal
        want = [math.sin(2 * math.pi * xf), math.cos(2 * math.pi * xf),
                math.sin(2 * math.pi * yf), math.cos(2 * math.pi * yf)]
        for i, (g, w) in enumerate(zip(got, want)):
            self.assertLess(abs(g - w), 1e-6, (i, g, w))

    def test_f16_large_turns(self):
        half = struct.unpack("<H", struct.pack("<e", 100.0))[0]
        lines = [
            f"v_mov_b32 v0, 0x{half:04x}",
            "v_sin_f16 v1, v0",
            "v_cos_f16 v2, v0",
            "exp mrt0 v1, v2, v0, v0 done vm",
            "s_endpgm",
        ]
        words = support.mrt0_words(support.translate_lines("ps_sin_cos_f16_turns", lines))
        self.assertEqual(words[0] & 0xFFFF, 0x0000)      # sin(100 turns) = +0
        self.assertEqual(words[1] & 0xFFFF, 0x3C00)      # cos(100 turns) = 1.0

    def test_negative_zero_keeps_its_sign_through_sin(self):
        lines = [
            "v_mov_b32 v0, 0x80000000",
            "v_sin_f32 v1, v0",
            "v_cos_f32 v2, v0",
            "exp mrt0 v1, v2, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sin_cos_zero", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80000000)      # sin(-0) = -0
        self.assertEqual(words[1], 0x3f800000)      # cos(-0) = 1


if __name__ == "__main__":
    unittest.main()
