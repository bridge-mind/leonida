"""TEST-20: check.py's partial-wave test asks vk_compute_test for subgroup size 64 natively and for 32 under
LEONIDA_W64_EMULATE=1, so a device that defaults to 64 (AMD) also runs the emulated path at 32."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import check

NAME, WANT = check.W64X_PARTIAL[0]


class PartialWaveSubgroupSize(unittest.TestCase):
    def run_partial(self, emulated, max_subgroup=64):
        asked = []

        def harness(_tool, _shader, *_args):
            asked.append(os.environ.get("VKC_SUBGROUP_SIZE"))
            return 0, f"device: gpu (subgroup 32..{max_subgroup})\nout[0..7] = " + " ".join(map(str, WANT)) + "\n"

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"{NAME}.spv").touch()
            with (patch.object(check, "run", side_effect=harness), patch.object(check, "W64_EMULATED", emulated),
                  patch.dict(os.environ, {"VKC_SUBGROUP_SIZE": "16"})):
                result = check.partial_wave_test(tmp, tmp, NAME, WANT)
                restored = os.environ.get("VKC_SUBGROUP_SIZE")
        return result, asked, restored

    def test_native_asks_for_64(self):
        result, asked, restored = self.run_partial(emulated=False)
        self.assertEqual((result, asked, restored), ("", ["64"], "16"))

    def test_emulated_asks_for_32(self):
        result, asked, restored = self.run_partial(emulated=True)
        self.assertEqual((result, asked, restored), ("", ["32"], "16"))

    def test_emulated_runs_on_a_32_wide_device(self):
        result, asked, _ = self.run_partial(emulated=True, max_subgroup=32)
        self.assertEqual((result, asked), ("", ["32"]))

    def test_native_skips_without_64_wide_subgroups(self):
        result, _, _ = self.run_partial(emulated=False, max_subgroup=32)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
