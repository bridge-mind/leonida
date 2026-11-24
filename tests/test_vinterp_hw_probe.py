"""Unit tests for tools/vinterp_hw_probe.py (SHADER-430). Standard library only; no GPU or game files needed."""
from contextlib import redirect_stdout
from fractions import Fraction
import io
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import vinterp_hw_probe as probe  # noqa: E402
from hw_oracle import runner as hw_runner  # noqa: E402


class VinterpHwProbeTests(unittest.TestCase):
    def test_lds_param_offset_layout(self):
        # Confirmed on AMD Radeon 680M (gfx1035): 48 bytes per attr, P0/P10 at 8*comp + {0, 4}, P20 at 32 + 4*comp.
        self.assertEqual(probe.lds_param_offset(0, 0, "p0"), 0)
        self.assertEqual(probe.lds_param_offset(0, 0, "p10"), 4)
        self.assertEqual(probe.lds_param_offset(0, 0, "p20"), 32)
        self.assertEqual(probe.lds_param_offset(0, 1, "p0"), 8)
        self.assertEqual(probe.lds_param_offset(0, 1, "p10"), 12)
        self.assertEqual(probe.lds_param_offset(0, 1, "p20"), 36)
        self.assertEqual(probe.lds_param_offset(0, 2, "p20"), 40)
        self.assertEqual(probe.lds_param_offset(0, 3, "p20"), 44)
        self.assertEqual(probe.lds_param_offset(1, 0, "p0"), 48)
        self.assertEqual(probe.lds_param_offset(1, 0, "p20"), 80)
        with self.assertRaises(ValueError):
            probe.lds_param_offset(0, 4, "p0")
        with self.assertRaises(ValueError):
            probe.lds_param_offset(0, 0, "p30")

    def test_hand_derivation_of_mixed_f16_and_f32_channel(self):
        # Analytical derivation of CASE_MIXED_F16_F32 matching RDNA2 gfx1035 hardware words:
        # P0 = 0xc0003e00 (f16 lo = 1.5, hi = -2.0), P10 = 0x49003c00 (f16 lo = 1.0, hi = 10.0),
        # P20 = 0x4c804000 (f16 lo = 2.0, hi = 18.0), i = 0.25, j = 0.50.
        i = Fraction(1, 4)
        j = Fraction(1, 2)
        f16_lo = probe.hbits_to_fraction(0x3E00) + i * probe.hbits_to_fraction(0x3C00) + j * probe.hbits_to_fraction(0x4000)
        f16_hi = probe.hbits_to_fraction(0xC000) + i * probe.hbits_to_fraction(0x4900) + j * probe.hbits_to_fraction(0x4C80)
        self.assertEqual(f16_lo, Fraction(11, 4))  # 2.75
        self.assertEqual(f16_hi, Fraction(19, 2))  # 9.5
        self.assertEqual(probe.SENTINEL_HI | probe.fraction_to_f16_rne(f16_lo), probe.CASE_MIXED_F16_F32.expected_hw[0])
        self.assertEqual(probe.SENTINEL_HI | probe.fraction_to_f16_rne(f16_hi), probe.CASE_MIXED_F16_F32.expected_hw[1])
        self.assertEqual(0xC0003E00, probe.CASE_MIXED_F16_F32.expected_hw[2])

        p0_f32 = probe.fbits_to_fraction(0xC0003E00)
        p10_f32 = probe.fbits_to_fraction(0x49003C00)
        p20_f32 = probe.fbits_to_fraction(0x4C804000)
        partial = probe.round_rne_fraction(p0_f32 + i * p10_f32, 24, -126)
        full = probe.fraction_to_f32_rne(partial + j * p20_f32)
        self.assertEqual(full, probe.CASE_MIXED_F16_F32.expected_hw[3])

    def test_single_vs_double_rounding_ties(self):
        # Verify the 4 near-tie cases in CASE_SINGLE_ROUND_TIES separate single rounding from f32-then-f16 rounding.
        # Tie A: 1 + 2^-11 + (-2^-40)*(-1) = 1 + 2^-11 + 2^-40
        exact_a = Fraction(1) + Fraction(1, 1 << 11) + Fraction(1, 1 << 40)
        self.assertEqual(probe.fraction_to_f16_rne(exact_a), 0x3C01)
        self.assertEqual(probe.fraction_to_f16_rne(probe.round_rne_fraction(exact_a, 24, -126)), 0x3C00)

        # Tie B: 1 + 3*2^-11 + (2^-30)*(-1) = 1 + 3*2^-11 - 2^-30
        exact_b = Fraction(1) + Fraction(3, 1 << 11) - Fraction(1, 1 << 30)
        self.assertEqual(probe.fraction_to_f16_rne(exact_b), 0x3C01)
        self.assertEqual(probe.fraction_to_f16_rne(probe.round_rne_fraction(exact_b, 24, -126)), 0x3C02)

        # Tie E (overflow): 65504 + 16 - 2^-20 = 65520 - 2^-20
        exact_e = Fraction(65520) - Fraction(1, 1 << 20)
        self.assertEqual(probe.fraction_to_f16_rne(exact_e), 0x7BFF)
        self.assertEqual(probe.fraction_to_f16_rne(probe.round_rne_fraction(exact_e, 24, -126)), 0x7C00)

        # Tie D (subnormal): 2^-25 + 2^-60
        exact_d = Fraction(1, 1 << 25) + Fraction(1, 1 << 60)
        self.assertEqual(probe.fraction_to_f16_rne(exact_d), 0x0001)
        self.assertEqual(probe.fraction_to_f16_rne(probe.round_rne_fraction(exact_d, 24, -126)), 0x0000)

    def test_translator_cli_and_mocked_hw_oracle(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = probe.main(["--oracle", "translator"])
        self.assertEqual(rc, 0)
        self.assertIn("translator checks: ok", out.getvalue())

        # Mocked hardware oracle returning exact expected words in wave32 and wave64
        per_wave = {
            32: {c.name: c.expected_hw for c in probe.CASES},
            64: {c.name: c.expected_hw for c in probe.CASES},
        }
        out_hw = io.StringIO()
        with unittest.mock.patch.object(probe, "run_hw", return_value=("mock-gfx1035", "gfx1035", per_wave)):
            with redirect_stdout(out_hw):
                rc_hw = probe.main(["--oracle", "both"])
        self.assertEqual(rc_hw, 0)
        self.assertIn("mixed_f16_and_f32_same_channel", out_hw.getvalue())

        # Mismatch on wave64 returns exit code 1
        bad_wave = {
            32: {c.name: c.expected_hw for c in probe.CASES},
            64: {c.name: (0, 0, 0, 0) for c in probe.CASES},
        }
        out_bad = io.StringIO()
        with unittest.mock.patch.object(probe, "run_hw", return_value=("mock-gfx1035", "gfx1035", bad_wave)):
            with redirect_stdout(out_bad):
                rc_bad = probe.main(["--oracle", "hw"])
        self.assertEqual(rc_bad, 1)
        self.assertIn("MISMATCH", out_bad.getvalue())

    def test_hw_unavailable_returns_77(self):
        out = io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", side_effect=hw_runner.Unavailable("no AMD GPU")):
            with redirect_stdout(out):
                rc = probe.main(["--oracle", "hw"])
        self.assertEqual(rc, 77)
        self.assertIn("no AMD GPU", out.getvalue())


if __name__ == "__main__":
    unittest.main()
