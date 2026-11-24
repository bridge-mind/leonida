"""Input checks for tools/tile_probe.py, on a synthetic blob (no game data).

    python -I tests/test_tile_probe.py

The tool needs numpy, so the tests skip without it.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "tile_probe.py")
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
# the tool is documented as `python -I`, which does not see a --user install of Pillow
HAVE_PIL_ISOLATED = subprocess.run([sys.executable, "-I", "-c", "import PIL"],
                                   capture_output=True, check=False).returncode == 0


@unittest.skipUnless(HAVE_NUMPY, "tile_probe.py needs numpy")
class TileProbeInputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.blob = os.path.join(self.tmp.name, "blob.bin")
        with open(self.blob, "wb") as f:
            f.write(bytes(range(256)) * 64)  # 16 KiB: smaller blobs give no candidate width

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, timeout=120)

    def test_missing_file_is_a_message_not_a_traceback(self):
        p = self.run_tool(os.path.join(self.tmp.name, "missing.bin"))
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("missing.bin", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_offset_outside_the_file_is_an_error(self):
        # Before: -5 quietly probed the last five bytes, the file size and more printed nothing, and both exited 0.
        for offset in ("-5", "16384", "99999"):
            with self.subTest(offset=offset):
                p = self.run_tool(self.blob, "--offset", offset)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn("outside", p.stderr)
                self.assertEqual(p.stdout, "")

    def test_empty_file_is_an_error(self):
        empty = os.path.join(self.tmp.name, "empty.bin")
        open(empty, "wb").close()
        p = self.run_tool(empty)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("0 bytes", p.stderr)

    def test_a_sample_with_no_candidate_layout_is_an_error(self):
        # Before: an 8-byte or 4 KiB blob printed nothing, wrote no preview and exited 0.
        tiny = os.path.join(self.tmp.name, "tiny.bin")
        with open(tiny, "wb") as f:
            f.write(bytes(4096))
        preview = os.path.join(self.tmp.name, "preview.png")
        for extra in ((), ("--out", preview)):
            with self.subTest(extra=extra):
                p = self.run_tool(tiny, *extra)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn("no layout fits", p.stderr)
                self.assertIn("tiny.bin", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertEqual(p.stdout, "")
                self.assertFalse(os.path.exists(preview), "a run with no layout writes no preview")

    def test_out_cannot_overwrite_the_input(self):
        # Before: --out naming the blob saved the preview over it after the probe, destroying the sample.
        with open(self.blob, "rb") as f:
            before = f.read()
        p = self.run_tool(self.blob, "--out", self.blob)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("overwrite the input", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        with open(self.blob, "rb") as f:
            self.assertEqual(f.read(), before, "the sample is left as it was")

    def test_out_directory_is_a_message_not_a_traceback(self):
        p = self.run_tool(self.blob, "--out", self.tmp.name)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("is a directory", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_out_under_a_regular_file_is_a_message_not_a_traceback(self):
        # Before: os.makedirs raised FileExistsError (NotADirectoryError on Windows) out of check_preview_path.
        p = self.run_tool(self.blob, "--out", os.path.join(self.blob, "sub", "p.png"))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("--out", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_out_creates_a_missing_folder(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import tile_probe
        preview = os.path.join(self.tmp.name, "previews", "deep", "best.png")
        tile_probe.check_preview_path(preview, self.blob)
        self.assertTrue(os.path.isdir(os.path.dirname(preview)), "the preview's folder is created")

    @unittest.skipIf(HAVE_PIL_ISOLATED, "this interpreter's -I can import Pillow")
    def test_preview_without_pillow_is_a_message_not_a_traceback(self):
        # Before: the probe ran, then render()'s lazy PIL import raised after all the work was done.
        p = self.run_tool(self.blob, "--out", os.path.join(self.tmp.name, "preview.png"))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("Pillow", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_offset_inside_the_file_still_probes(self):
        p = self.run_tool(self.blob, "--offset", "0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("score", p.stdout)

    def test_block_feature_rejects_unsupported_bpe(self):
        import numpy as np
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import tile_probe

        for bad_bpe in (1, 2, 4, 12):
            with self.subTest(bpe=bad_bpe), self.assertRaises(ValueError):
                tile_probe.block_feature(np.zeros((4, bad_bpe), np.uint8), bad_bpe)

    def test_render_packs_ddscaps_texture_and_rejects_unfitting_layout(self):
        import struct
        import types
        from unittest import mock

        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import tile_probe

        captured = []

        class _FakeImage:
            size = (256, 64)

            def convert(self, _mode):
                return self

            def save(self, _out):
                pass

        fake_image_mod = types.SimpleNamespace(open=lambda bio: (captured.append(bio.getvalue()), _FakeImage())[1])
        fake_pil = types.SimpleNamespace(Image=fake_image_mod)
        blob = bytes(256 * 256)
        with mock.patch.dict(sys.modules, {"PIL": fake_pil, "PIL.Image": fake_image_mod}):
            tile_probe.render(blob, 0, 8, 256, "4k", os.path.join(self.tmp.name, "ok.png"))
            self.assertEqual(len(captured), 1)
            caps = struct.unpack_from("<I", captured[0], 108)[0]
            self.assertEqual(caps, 0x1000)

            with self.assertRaises(ValueError):
                tile_probe.render(blob, 0, 8, 256, "64k", os.path.join(self.tmp.name, "bad.png"))


if __name__ == "__main__":
    unittest.main()
