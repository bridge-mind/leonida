"""Exercise the texture synchronization CMake check with a synthetic cube harness."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECK = Path(__file__).resolve().parents[1] / "runtime/test/check_texture_readback_sync.cmake"
CMAKE = shutil.which("cmake")
FRAME_BYTES = 64 * 64 * 16
HARNESS = """\
import os
from pathlib import Path
import sys

assert sys.argv[1] == "cube"
assert os.environ["VK_INSTANCE_LAYERS"] == "VK_LAYER_KHRONOS_validation"
mode = os.environ["LEONIDA_SYNTHETIC_READBACK_MODE"]
output = Path(sys.argv[2])
if mode in ("full", "hazard"):
    output.write_bytes(bytes(64 * 64 * 16))
elif mode == "short":
    output.write_bytes(b"short")
if mode == "hazard":
    print("SYNC-HAZARD", file=sys.stderr)
print('Insert instance layer "VK_LAYER_KHRONOS_validation"', file=sys.stderr)
print("vk_texture_test cube: PASSED (4096 pixels)")
"""


@unittest.skipUnless(CMAKE, "CMake is required to exercise the texture synchronization check")
class TextureReadbackSyncTests(unittest.TestCase):
    def test_requires_fresh_complete_readback(self):
        cases = (
            ("fresh", "full", False, 0),
            ("valid rerun", "full", True, 0),
            ("missing", "none", False, 1),
            ("stale full-size file", "none", True, 1),
            ("short", "short", False, 1),
            ("sync hazard", "hazard", False, 1),
        )
        for name, mode, preseeded, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shader_dir = root / "shaders"
                shader_dir.mkdir()
                output = shader_dir / "cube readback.raw"
                if preseeded:
                    output.write_bytes(b"x" * FRAME_BYTES)
                harness = root / "harness.py"
                harness.write_text(HARNESS, encoding="utf-8", newline="\n")

                result = subprocess.run(
                    [CMAKE, f"-DRUNNER={sys.executable}", f"-DTOOL={harness}",
                     f"-DSPIRV={shader_dir.name}", f"-DOUTPUT={output.name}", "-P", str(CHECK)],
                    cwd=root, env=dict(os.environ, LEONIDA_SYNTHETIC_READBACK_MODE=mode),
                    capture_output=True, encoding="utf-8", errors="replace",
                    timeout=10, check=False,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if expected == 0:
                    self.assertEqual(output.read_bytes(), bytes(FRAME_BYTES))
                if mode == "hazard":
                    self.assertIn("SYNC-HAZARD", result.stderr)


if __name__ == "__main__":
    unittest.main()
