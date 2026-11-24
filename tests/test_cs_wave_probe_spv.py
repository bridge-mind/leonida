"""runtime/test/cs_wave_probe_spv.h is the SPIR-V of runtime/test/glsl/cs_wave_probe.comp (vk_compute_test's
subgroup probe). Run: python -I tests/test_cs_wave_probe_spv.py"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "runtime", "test", "cs_wave_probe_spv.h")
SOURCE = os.path.join(ROOT, "runtime", "test", "glsl", "cs_wave_probe.comp")


def header_words():
    with open(HEADER, encoding="utf-8") as f:
        return [int(x, 16) for x in re.findall(r"0x([0-9a-f]{8})u", f.read())]


class WaveProbeSpirv(unittest.TestCase):
    def test_header_is_spirv_1_5(self):
        # Below 1.6, so varying subgroup sizes are not allowed in the probe either (as in the translator's output).
        w = header_words()
        self.assertEqual(w[0], 0x07230203)
        self.assertEqual(w[1], 0x00010500)

    def test_header_matches_the_glsl(self):
        glslang = shutil.which("glslangValidator")
        if not glslang:
            self.skipTest("glslangValidator is not on PATH")
        version = subprocess.run([glslang, "--version"], capture_output=True, text=True).stdout
        if "16.6.0" not in version:   # another glslang may emit other (equally valid) code
            self.skipTest("the header was generated with glslang 16.6.0")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "cs_wave_probe.spv")
            subprocess.run([glslang, "-V", "--target-env", "vulkan1.2", "-Os", "-o", out, SOURCE], check=True,
                           stdout=subprocess.DEVNULL)
            with open(out, "rb") as f:
                data = f.read()
        words = list(struct.unpack(f"<{len(data) // 4}I", data))
        # Word 2 is the generator (glslang's version); the rest must be what the header holds.
        self.assertEqual(words[:2] + words[3:], header_words()[:2] + header_words()[3:])


if __name__ == "__main__":
    unittest.main()
