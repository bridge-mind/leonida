"""Semantic regression tests for v_cvt_f16_f32 VOP3 output modifiers across RNE and MODE RTZ.

Standard library only; no GPU, no LLVM, no game files.

Covers the native RDNA2 (gfx1036) measurements via tools/rdna2spirv.py + spirv_mini_interp:
  * div:2 underflowing a negative exponent-1 normal (-2^-14 .. -(2 - 2^-10) * 2^-14) preserves the sign bit (-0.0,
    0x8000) in both RNE and MODE toward-zero, while exponent-0 inputs (-0.0, -2^-15) give +0.0 (0x0000) and clamp
    still turns -0.0 into +0.0;
  * under dynamic MODE toward-zero (s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3), finite mul:2/mul:4 overflow
    saturates to +-65504 (0x7bff / 0xfbff), true +-inf stays +-inf, +-65520 / FLT_MAX with div:2 convert to +-65504
    before halving to +-32752 (0x77ff / 0xf7ff), and switching back to RNE (0) restores +-inf;
  * clamp runs after omod, preserving the seeded high half of vdst.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _run_cvt_quad(cases: list[tuple[int, str]], *, prefix: list[str] = ()) -> tuple[str, list[int]]:
    """Translate and run up to four (f32_bits, mod_suffix) cases on one fragment, returning (spvasm, mrt0_words)."""
    assert len(cases) == 4
    lines = list(prefix)
    for i, (bits, mod) in enumerate(cases):
        suffix = f" {mod}" if mod else ""
        lines += [
            f"v_mov_b32 v{10 + i}, {bits:#010x}",
            f"v_mov_b32 v{i}, 0xaaaa5555",
            f"v_cvt_f16_f32_e64 v{i}, v{10 + i}{suffix}",
        ]
    lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    spv = support.translate_lines("ps_cvt_f16_omod", lines)
    return spv, support.mrt0_words(spv)


class CvtF16OmodSemanticTests(unittest.TestCase):
    def test_div2_exp1_underflow_preserves_sign_rne_and_rtz(self):
        exp1_cases = [
            (_f32_bits(-(2.0 ** -14)), "div:2"),         # -2^-14 -> 0x8400 -> 0x8000
            (_f32_bits(-1.5 * (2.0 ** -14)), "div:2"),   # -1.5*2^-14 -> 0x8600 -> 0x8000
            (0xB8FFE000, "div:2"),                       # -(2 - 2^-10) * 2^-14 -> 0x87ff -> 0x8000
            (0xB8FFFE00, "div:2"),                       # RNE rounds to -2^-13 (0x8800 -> 0x8400); RTZ keeps 0x87ff (-> 0x8000)
        ]
        other_cases = [
            (_f32_bits(2.0 ** -14), "div:2"),            # +2^-14 -> +0.0
            (0x80000000, "div:2"),                       # -0.0 (exp 0) -> +0.0
            (_f32_bits(-(2.0 ** -15)), "div:2"),         # -2^-15 (exp 0) -> +0.0
            (_f32_bits(-(2.0 ** -14)), "clamp div:2"),   # -0.0 clamped -> +0.0
        ]
        for mode_prefix, exp4 in (([], 0xAAAA8400), ([RTZ], 0xAAAA8000)):
            with self.subTest(rtz=bool(mode_prefix)):
                _, got_exp1 = _run_cvt_quad(exp1_cases, prefix=mode_prefix)
                self.assertEqual(got_exp1, [0xAAAA8000, 0xAAAA8000, 0xAAAA8000, exp4])
                _, got_other = _run_cvt_quad(other_cases, prefix=mode_prefix)
                self.assertEqual(got_other, [0xAAAA0000, 0xAAAA0000, 0xAAAA0000, 0xAAAA0000])

    def test_rtz_omod_overflow_and_div2_order(self):
        rtz_overflow = [
            (_f32_bits(32768.0), "mul:2"),
            (_f32_bits(-32768.0), "mul:2"),
            (_f32_bits(16384.0), "mul:4"),
            (_f32_bits(-16384.0), "mul:4"),
        ]
        _, got_ov = _run_cvt_quad(rtz_overflow, prefix=[RTZ])
        self.assertEqual(got_ov, [0xAAAA7BFF, 0xAAAAFBFF, 0xAAAA7BFF, 0xAAAAFBFF])

        rtz_div2_and_inf = [
            (_f32_bits(65504.0), "mul:2"),
            (_f32_bits(65520.0), "div:2"),
            (_f32_bits(-65520.0), "div:2"),
            (0x7F7FFFFF, "div:2"),                       # FLT_MAX -> 65504 -> 32752
        ]
        _, got_div = _run_cvt_quad(rtz_div2_and_inf, prefix=[RTZ])
        self.assertEqual(got_div, [0xAAAA7BFF, 0xAAAA77FF, 0xAAAAF7FF, 0xAAAA77FF])

        rtz_true_inf = [
            (0x7F800000, "mul:2"),
            (0xFF800000, "mul:4"),
            (0x7F800000, "div:2"),
            (0xFF800000, "div:2"),
        ]
        _, got_inf = _run_cvt_quad(rtz_true_inf, prefix=[RTZ])
        self.assertEqual(got_inf, [0xAAAA7C00, 0xAAAAFC00, 0xAAAA7C00, 0xAAAAFC00])

        # Switching RTZ -> RNE in one program matches the dynamic MODE write semantics.
        switch_lines = [
            RTZ,
            f"v_mov_b32 v10, {_f32_bits(32768.0):#010x}",
            f"v_mov_b32 v11, {_f32_bits(65520.0):#010x}",
            "v_mov_b32 v0, 0xaaaa5555",
            "v_mov_b32 v1, 0xaaaa5555",
            "v_mov_b32 v2, 0xaaaa5555",
            "v_mov_b32 v3, 0xaaaa5555",
            "v_cvt_f16_f32_e64 v0, v10 mul:2",
            "v_cvt_f16_f32_e64 v1, v11 div:2",
            RNE,
            "v_cvt_f16_f32_e64 v2, v10 mul:2",
            "v_cvt_f16_f32_e64 v3, v11 div:2",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_mode_switch", switch_lines)
        self.assertEqual(support.mrt0_words(spv), [0xAAAA7BFF, 0xAAAA77FF, 0xAAAA7C00, 0xAAAA7C00])

    def test_clamp_with_omod_order(self):
        c1 = [
            (_f32_bits(0.75), "clamp mul:2"),
            (_f32_bits(0.75), "clamp div:2"),
            (_f32_bits(2.0), "clamp mul:2"),
            (_f32_bits(-0.5), "clamp mul:2"),
        ]
        c2 = [
            (_f32_bits(65520.0), "clamp mul:2"),
            (0xFF7FFFFF, "clamp div:2"),                 # -FLT_MAX
            (0x7F800000, "clamp div:2"),                 # +inf
            (0xFF800000, "clamp mul:2"),                 # -inf
        ]
        for mode_prefix in ([], [RTZ]):
            with self.subTest(rtz=bool(mode_prefix)):
                self.assertEqual(_run_cvt_quad(c1, prefix=mode_prefix)[1], [0xAAAA3C00, 0xAAAA3600, 0xAAAA3C00, 0xAAAA0000])
                self.assertEqual(_run_cvt_quad(c2, prefix=mode_prefix)[1], [0xAAAA3C00, 0xAAAA0000, 0xAAAA3C00, 0xAAAA0000])

    def test_negative_control_div2_signed_zero(self):
        spv, got = _run_cvt_quad([
            (_f32_bits(-(2.0 ** -14)), "div:2"),
            (_f32_bits(1.0), ""),
            (_f32_bits(1.0), ""),
            (_f32_bits(1.0), ""),
        ])
        self.assertEqual(got[0], 0xAAAA8000)
        # Sabotage the e == 1 underflow select to drop sign (reverting to +0.0).
        bad, n = re.subn(
            r"(%\w+ = OpIEqual %bool %\w+ %c1\n\s*"
            r"%\w+ = OpBitwiseAnd %uint %\w+ %c8000\n\s*"
            r"%\w+ = OpSelect %uint %\w+ )(%\w+)( %\w+)",
            r"\1%c0\3",
            spv,
        )
        self.assertEqual(n, 1)
        self.assertEqual(support.mrt0_words(bad)[0], 0xAAAA0000)

    def test_negative_control_rtz_saturation(self):
        spv, got = _run_cvt_quad(
            [
                (_f32_bits(32768.0), "mul:2"),
                (_f32_bits(1.0), ""),
                (_f32_bits(1.0), ""),
                (_f32_bits(1.0), ""),
            ],
            prefix=[RTZ],
        )
        self.assertEqual(got[0], 0xAAAA7BFF)
        bad, n = re.subn(r"OpBitwiseOr %uint (%\w+) %c7bff", r"OpBitwiseOr %uint \1 %c7c00", spv)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(support.mrt0_words(bad)[0], 0xAAAA7C00)


if __name__ == "__main__":
    unittest.main()
