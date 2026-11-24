"""scripts/run_gpu_tests.py: the pixel checkers fail a short frame or non-finite values, as check.py does (no device needed)."""
from pathlib import Path
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt
import check

NAN = float("nan")


class PixelChecks(unittest.TestCase):
    def raw(self, data: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".raw")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def test_good_frames_pass(self):
        grad = b"".join(struct.pack("<4f", (i % 64 + 0.5) / 64, (i // 64 + 0.5) / 64, 0.5, 1.0) for i in range(4096))
        self.assertTrue(rgt.check_grad(self.raw(grad), 0.5, 1.0)[0])

    def test_bary_checks_both_interpolation_paths(self) -> None:
        triangle = []
        for y in range(64):
            for x in range(64):
                u, v = (x + 0.5) / 64, (y + 0.5) / 64
                triangle.append((u, v, u / 2, v / 2))
        wrong_raw = triangle.copy()
        u, v, i, _ = wrong_raw[1337]
        wrong_raw[1337] = (u, v, i, 0.0)
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "ps_bary.spv").touch()
            for name, frame, expected in (
                ("triangle gradient", triangle, True),
                ("both paths return zero", [(0.0, 0.0, 0.0, 0.0)] * 4096, False),
                ("one raw barycentric is wrong", wrong_raw, False),
            ):
                with self.subTest(frame=name):
                    raw = self.raw(b"".join(struct.pack("<4f", *pixel) for pixel in frame))
                    with patch.object(check, "pixels", return_value=(frame, "")):
                        standalone = check.pixel_test(work, work, "ps_bary")
                    self.assertEqual((rgt.check_bary(raw)[0], standalone == ""), (expected, expected))

    def test_d16_gradient_rejects_swapped_halves(self):
        # Binary16 encodings of UNORM8 texels 0, 32, ... 224.
        words = (0x0000, 0x3004, 0x3404, 0x3606, 0x3804, 0x3905, 0x3a06, 0x3b07)
        levels = [struct.unpack("<e", struct.pack("<H", word))[0] for word in words]
        frame = [(levels[x // 8], levels[y // 8], levels[4], 1.0) for y in range(64) for x in range(64)]
        good = self.raw(b"".join(struct.pack("<4f", *pixel) for pixel in frame))
        self.assertTrue(rgt.check_d16_texture(good)[0])
        swapped = self.raw(b"".join(struct.pack("<4f", g, r, b, a) for r, g, b, a in frame))
        ok, detail = rgt.check_d16_texture(swapped)
        self.assertFalse(ok)
        self.assertIn("3584", detail)
        self.assertIn("(8,0)", detail)

    def test_packed_interpolation_checks_each_decoded_channel(self):
        def half_word(value: float) -> int:
            # Only the low 16 bits of the first three exports carry an interpolation result.
            return 0xffff0000 | struct.unpack("<H", struct.pack("<e", value))[0]

        frame = []
        for y in range(64):
            v = (y + 0.5) / 64
            for x in range(64):
                u = (x + 0.5) / 64
                frame.append([half_word(u), half_word(v), half_word(u), struct.unpack("<I", struct.pack("<f", v))[0]])
        self.assertEqual(rgt.interp_f16_high_problem(frame), "")
        for pixel in (0, 2071):
            for channel in range(4):
                for bits in ((0x7fc00000, 0xff800000) if channel == 3 else (0x7e00, 0xfc00)):
                    with self.subTest(pixel=pixel, channel=channel, bits=bits):
                        saved = frame[pixel][channel]
                        frame[pixel][channel] = bits
                        problem = rgt.interp_f16_high_problem(frame)
                        frame[pixel][channel] = saved
                        self.assertIn("non-finite", problem)

        frame[2071][1] = half_word(0.0)
        self.assertIn("halves off", rgt.interp_f16_high_problem(frame))

    def test_computed_ij_requires_a_full_finite_frame(self):
        def words(v_over_two):
            return struct.unpack("<4I", struct.pack("<4f", 0.5, 0.0, v_over_two, 0.5))

        frame = [words((y + 0.5) / 128) for y in range(64) for _ in range(64)]
        self.assertEqual(rgt.ij_computed_problem(frame), "")
        for wrong_size in ([], [words(0.25)], frame[:-1], frame + frame[:1]):
            with self.subTest(pixels=len(wrong_size)):
                self.assertTrue(rgt.ij_computed_problem(wrong_size))

        for pixel in (0, 1337):
            for channel in range(4):
                with self.subTest(pixel=pixel, channel=channel):
                    corrupted = frame.copy()
                    bad_pixel = list(corrupted[pixel])
                    bad_pixel[channel] = 0x7FC00000
                    corrupted[pixel] = tuple(bad_pixel)
                    self.assertTrue(rgt.ij_computed_problem(corrupted))

    def test_nan_frame_fails(self):
        nan = self.raw(struct.pack("<4f", NAN, NAN, NAN, NAN) * 4096)
        self.assertFalse(rgt.check_bary(nan)[0])
        self.assertFalse(rgt.check_grad(nan, 0.5, 1.0)[0])

    def test_empty_frame_fails(self):
        empty = self.raw(b"")
        for name, check in (("bary", rgt.check_bary), ("grad", lambda p: rgt.check_grad(p, 0.0, 0.0)),
                            ("wqm", rgt.check_wqm)):
            with self.subTest(check=name):
                self.assertFalse(check(empty)[0])

    def test_partial_pixel_is_no_frame(self):
        # TEST-29: a readback that ends mid-pixel is malformed, even with 4096 whole pixels before the tail.
        odd = self.raw(struct.pack("<4f", 0.5, 0.25, 0.25, 0.125) * 4096 + b"\0" * 8)
        self.assertEqual(rgt.read_raw_rgba32f(odd), [])
        self.assertEqual(rgt.read_raw_u32(odd), [])
        self.assertFalse(rgt.check_bary(odd)[0])

    def test_remaining_checks_reject_short_and_nan_frames(self):
        # TEST-29: the checks 662752cc did not reach.
        def uniform(i_value):   # (s5, bit, i, w) with all three (s5, bit) kinds present
            kinds = ((8, 1, 0.5), (8, 0, 0.1), (7, 0, 0.1))
            return b"".join(struct.pack("<2I2f", s5, bit, i_value if i_value is not None else i, 0.0)
                            for s5, bit, i in (kinds[k % 3] for k in range(4096)))
        self.assertTrue(rgt.check_uniform_branch(self.raw(uniform(None)))[0])
        self.assertIn("non-finite", rgt.check_uniform_branch(self.raw(uniform(NAN)))[1])
        self.assertFalse(rgt.check_uniform_branch(self.raw(uniform(None)[:16 * 300]))[0])

        flat = b"".join(struct.pack("<2I2f", 0x101, 0x102, (i % 64 + 0.5) / 64, (i // 64 + 0.5) / 64)
                        for i in range(4096))
        self.assertTrue(rgt.check_flat(self.raw(flat))[0])
        self.assertIn("non-finite", rgt.check_flat(self.raw(struct.pack("<2I2f", 0x101, 0x102, NAN, NAN) * 4096))[1])
        self.assertFalse(rgt.check_flat(self.raw(b""))[0])   # used to raise ValueError (max of nothing)

        nan = self.raw(struct.pack("<4f", NAN, NAN, NAN, NAN) * 4096)
        self.assertIn("non-finite", rgt.check_clip_half(nan, 0.5, 1.0)[1])

        short = self.raw(b"".join(struct.pack("<4I", i % 32, (i % 32) ^ 9, (((i % 32) & 5) | 16) ^ 1, 0)
                                  for i in range(64)))
        ok, detail = rgt.check_bitmask_perm(short, "device: NVIDIA (subgroup 32..32)\nok 64x64\n")
        self.assertIs(ok, False)
        self.assertIn("4096", detail)

    def test_bitmask_perm_narrow_fragment_subgroup_skip(self):
        import check
        self.assertIs(rgt.bitmask_perm_narrow_skip, check.bitmask_perm_narrow_skip)
        simd16 = self.raw(b"".join(struct.pack("<4I", i % 16, (i % 16) ^ 9, 0, 0) for i in range(64 * 64)))
        ok, detail = rgt.check_bitmask_perm(simd16, "device: Intel ANV (subgroup 8..32)\nok 64x64\n")
        self.assertIsNone(ok)
        self.assertIn("16 lanes", detail)

        # Same narrow output on a fixed 32..32 device is a real failure, not a skip.
        ok, detail = rgt.check_bitmask_perm(simd16, "device: NVIDIA (subgroup 32..32)\nok 64x64\n")
        self.assertIs(ok, False)
        self.assertIn("want (9,17,0)", detail)

        # Full 32-lane output on an 8..32 device is still judged.
        simd32 = self.raw(b"".join(
            struct.pack("<4I", i % 32, (i % 32) ^ 9, (((i % 32) & 5) | 16) ^ 1, 0) for i in range(64 * 64)))
        ok, detail = rgt.check_bitmask_perm(simd32, "device: Intel ANV (subgroup 8..32)\nok 64x64\n")
        self.assertIs(ok, True)
        self.assertIn("4096/4096", detail)

    def test_vcmpx_forms_narrow_fragment_subgroup_skip(self):
        import check
        self.assertIs(rgt.vcmpx_forms_narrow_skip, check.vcmpx_forms_narrow_skip)
        # Intel 8..32 cannot pin fragment (#1318); skip before judging the constant mrt0 words (TEST-205).
        skip = check.vcmpx_forms_narrow_skip("device: Intel UHD 770 (subgroup 8..32)\nok 64x64\n")
        self.assertIsInstance(skip, check.Skip)
        self.assertIn("8..32", str(skip))
        self.assertIn("needs 32", str(skip))

        # Fixed 32-wide devices still run the const check.
        self.assertIsNone(check.vcmpx_forms_narrow_skip("device: NVIDIA (subgroup 32..32)\nok 64x64\n"))
        # AMD 32..64 can pin to 32 (min is already 32).
        self.assertIsNone(check.vcmpx_forms_narrow_skip("device: AMD (subgroup 32..64)\nok 64x64\n"))

        # max < 32: skip like the other wave32 pixel tests.
        skip = check.vcmpx_forms_narrow_skip("device: Mesa llvmpipe (subgroup 8..8)\nok 64x64\n")
        self.assertIsInstance(skip, check.Skip)
        self.assertIn("narrower than 32", str(skip))

        # Missing width line: skip rather than false-fail.
        skip = check.vcmpx_forms_narrow_skip("device: unknown\nok 64x64\n")
        self.assertIsInstance(skip, check.Skip)
        self.assertIn("no subgroup width", str(skip))


if __name__ == "__main__":
    unittest.main()
