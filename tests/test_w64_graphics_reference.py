"""CPU replay of the wave64 graphics runner's saved-reference comparisons."""

import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import w64_graphics


class TierCReferenceTests(unittest.TestCase):
    def test_partition_invariant_word_must_match_reference(self) -> None:
        name = "ps_w64_c_lane_id"
        # The fixture writes lane ID, lane ID within a half, pixel X and zero.
        wave64 = [(i % 64, i % 32, i % 64, 0) for i in range(64 * 64)]
        wave32 = [(i % 32, i % 32, i % 64, 0) for i in range(64 * 64)]
        wrong_x = wave32.copy()
        lane, half, x, zero = wrong_x[73]
        wrong_x[73] = (lane, half, x + 1, zero)

        with tempfile.TemporaryDirectory() as work:
            (Path(work) / f"{name}.spv").touch()
            (Path(work) / f"{name}.ref64.raw").write_bytes(
                b"".join(struct.pack("<4I", *pixel) for pixel in wave64)
            )
            with patch.dict(os.environ, W64G_REF=work, W64G_SAVE_REF=""):
                for frame, expected in ((wave32, ""), (wrong_x, "partition-invariant word")):
                    with self.subTest(expected=expected), patch.object(
                        w64_graphics, "draw",
                        side_effect=lambda _tools, _spirv, _name, size, _tmp: (frame, None, size == 32),
                    ):
                        result = w64_graphics.check(work, work, name)
                        if expected:
                            self.assertIn(expected, result)
                        else:
                            self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
