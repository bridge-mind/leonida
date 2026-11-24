"""tools/texoff_probe.py without a GPU: the rows tell a signed field from an unsigned one, and the kernels build.

    python -I tests/test_texoff_probe.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hw_oracle"))

import texoff_probe as p  # noqa: E402


def unsigned_reading(dim, start, off):
    """The same sample if the fields were read unsigned (the rule the rows must rule out)."""
    x0, b0 = start
    x = p.clamp(x0 + (off & 0x3F), p.W)
    shift, n = (8, p.H2D) if dim == "2d" else (16, p.D3D)
    b = p.clamp(b0 + ((off >> shift) & 0x3F), n)
    return float(b * p.W + x)


class RowsTest(unittest.TestCase):
    def test_sext6(self):
        self.assertEqual([p.sext6(v) for v in (0, 0x1F, 0x20, 0x3F, 0x40, 0x7F)], [0, 31, -32, -1, 0, -1])

    def test_rows_with_a_negative_field_tell_signed_from_unsigned(self):
        for name, dim, start, off in p.ROWS:
            field = (off >> (8 if dim == "2d" else 16)) & 0x3F
            if field & 0x20 and field != 0x3F:
                with self.subTest(name):
                    self.assertNotEqual(p.expected(dim, start, off), unsigned_reading(dim, start, off))

    def test_both_axes_are_covered(self):
        self.assertEqual({d for _n, d, _s, _o in p.ROWS}, {"2d", "3d"})
        self.assertEqual(len({n for n, *_ in p.ROWS}), len(p.ROWS))

    def test_expected_values(self):
        self.assertEqual(p.expected("2d", (5, 40), 0x20 << 8), 8 * 64 + 5)       # 40 - 32
        self.assertEqual(p.expected("2d", (5, 10), 0x20 << 8), 0 * 64 + 5)       # clamps at row 0
        self.assertEqual(p.expected("2d", (5, 40), 0xC0 << 8), 40 * 64 + 5)      # bits 14-15 ignored
        self.assertEqual(p.expected("3d", (20, 40), (0x3C << 16) | (0x3F << 8) | 0x3E), 36 * 64 + 18)

    def test_kernels_build_for_rdna2_and_rdna3(self):
        for arch in ("gfx1036", "gfx1150"):
            for dim in ("2d", "3d"):
                rows = [r for r in p.ROWS if r[1] == dim]
                lines = p.kernel(arch, dim, rows)
                with self.subTest(arch=arch, dim=dim):
                    self.assertEqual(sum("image_sample_lz_o" in ln for ln in lines), len(rows))
                    self.assertIn(f"dim:SQ_RSRC_IMG_{dim.upper()}", "\n".join(lines))
                    self.assertEqual(lines[-1].startswith("  .long"), True)

    def test_3d_descriptor_type_and_depth(self):
        t = p.words("gfx1036", "3d")
        self.assertEqual(t[3] >> 28, p.SQ_RSRC_IMG_3D)
        self.assertEqual(t[4], p.D3D - 1)
        self.assertEqual(p.sampler_words()[0] & 0x1FF, 2 | 2 << 3 | 2 << 6)


class OracleLifecycleTest(unittest.TestCase):
    def _run_hw(self, fake_oracle):
        import contextlib
        import io
        from unittest import mock
        import runner

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
        import struct

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                text = "\n".join(kernels[0].lines)
                dim = "2d" if "SQ_RSRC_IMG_2D" in text else "3d"
                rows = [r for r in p.ROWS if r[1] == dim]
                return [[struct.unpack("<I", struct.pack("<f", p.expected(dim, s, o)))[0] for _n, _d, s, o in rows]]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 0)
        self.assertTrue(fake.closed)
        self.assertIn("device: fake-gpu", out)

    def test_hw_closes_on_mismatch(self):
        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return [[0] * kernels[0].words]

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
        import contextlib
        import io
        from unittest import mock
        import runner

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
