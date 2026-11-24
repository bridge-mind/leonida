"""tools/smem_neg_probe.py without a GPU: encodings, translator expectations, kernel builder, and oracle lifecycle.

    python -I tests/test_smem_neg_probe.py
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hw_oracle"))

import runner  # noqa: E402
import smem_neg_probe as p  # noqa: E402


class SmemNegProbeTest(unittest.TestCase):
    def test_seed_bounds(self):
        self.assertEqual(p.seed(-0x40), (p.SEED - 0x40) & 0xFFFFFFFF)
        self.assertEqual(p.seed(0), p.SEED)
        self.assertEqual(p.seed(0xBC), p.SEED + 0xBC)
        self.assertIsNone(p.seed(-0x44))
        self.assertIsNone(p.seed(0xC0))

    def test_sbuf_raw_encoding(self):
        lines = p.sbuf_raw(48, 12, 0x7D, -4)
        self.assertEqual(lines, [".long 0xf4200c06", ".long 0xfa1ffffc"])

    def test_translator_sbuf_bounds(self):
        self.assertEqual(p.translator_sbuf(0, 0), [0])
        self.assertEqual(p.translator_sbuf(0xFC, 0), [0xFC])
        self.assertEqual(p.translator_sbuf(0x100, 0), [None])
        self.assertEqual(p.translator_sbuf(0xFFFFFFFC, 8), [4])
        self.assertEqual(p.translator_sbuf(0xFFFFFFFC, 0), [None])

    def test_expected_and_kernel_lines(self):
        want = p.expected()
        self.assertEqual(list(want.keys()), [r[0] for r in p.ROWS])
        self.assertEqual(want["sload_imm_m4"], [(p.SEED - 4) & 0xFFFFFFFF])
        self.assertEqual(want["sload_x2_soff4_m10"], [(p.SEED - 0xC) & 0xFFFFFFFF, (p.SEED - 8) & 0xFFFFFFFF])
        self.assertEqual(want["sbuf_soffm4_imm8"], [p.SEED + 4])
        self.assertEqual(want["sbuf_soffm4"], [None])

        lines, order = p.kernel_lines()
        self.assertEqual(len(order), sum(len(r[2]) for r in p.ROWS))
        self.assertEqual(lines[-1], "s_endpgm")

    def test_translator_mode_prints_every_row(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = p.main(["--oracle", "translator"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        for name, *_ in p.ROWS:
            self.assertIn(name, out)


class OracleLifecycleTest(unittest.TestCase):
    def _expected_flat(self):
        want = p.expected()
        flat = []
        for name, *_ in p.ROWS:
            flat.extend(0 if v is None else v for v in want[name])
        return flat

    def _run_hw(self, fake_oracle):
        buf = io.StringIO()
        with (
            mock.patch.object(runner, "open_oracle", return_value=fake_oracle) as open_mock,
            mock.patch.object(runner, "Oracle", side_effect=AssertionError("Oracle() called directly")),
            contextlib.redirect_stdout(buf),
        ):
            rc = p.main(["--oracle", "hw"])
        open_mock.assert_called_once_with()
        return rc, buf.getvalue()

    def test_hw_uses_open_oracle_and_closes_on_match(self):
        flat = self._expected_flat()

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return [list(flat)]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 0)
        self.assertTrue(fake.closed)
        self.assertIn("device: fake-gpu (gfx1036)", out)
        self.assertNotIn("DIFFERS", out)

    def test_hw_closes_on_mismatch(self):
        flat = self._expected_flat()

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return [[0] * len(flat)]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 1)
        self.assertTrue(fake.closed)
        self.assertIn("DIFFERS", out)

    def test_hw_closes_on_kernel_rejection(self):
        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return ["assembler rejected"]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 2)
        self.assertTrue(fake.closed)
        self.assertIn("kernel rejected", out)

    def test_hw_unavailable_returns_77(self):
        buf = io.StringIO()
        with (
            mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no device")),
            mock.patch.object(runner, "Oracle", side_effect=AssertionError("Oracle() called directly")),
            contextlib.redirect_stdout(buf),
        ):
            rc = p.main(["--oracle", "hw"])
        self.assertEqual(rc, 77)
        self.assertIn("no device", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
