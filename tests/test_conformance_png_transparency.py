"""Synthetic PNG transparency regressions; no renderer, GPU or game files."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_compare_transparency", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)


class TransparencyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def png(self, name, rgba=False, transparent=False):
        path = self.root / name
        colour = (10, 20, 30, 0) if rgba else (10, 20, 30)
        C.write_png(str(path), C.make_solid_raster(1, 1, 6 if rgba else 2, colour, name))
        if transparent:
            chunks = C.split_png_chunks(path.read_bytes(), str(path))
            # Truecolor tRNS is three big-endian 16-bit samples (PNG 11.3.1.1).
            key = b"\x00\x0a\x00\x14\x00\x1e"
            path.write_bytes(C.PNG_SIGNATURE + b"".join(
                C.make_chunk(kind, data) + (C.make_chunk(b"tRNS", key) if kind == b"IHDR" else b"")
                for kind, data in chunks))
        return path

    def test_rgb_transparency_is_not_silently_decoded_as_opaque(self):
        path = self.png("transparent.png", transparent=True)
        with self.assertRaisesRegex(C.ToolError, "tRNS.*RGBA"):
            C.read_png(str(path))

    def test_trns_on_rgba_is_also_rejected(self):
        # tRNS is forbidden on RGBA, which already stores alpha.
        path = self.png("invalid.png", rgba=True, transparent=True)
        with self.assertRaisesRegex(C.ToolError, "tRNS is forbidden.*RGBA"):
            C.read_png(str(path))

    def test_opaque_rgb_and_explicit_rgba_remain_supported(self):
        rgb = C.read_png(str(self.png("rgb.png")))
        rgba = C.read_png(str(self.png("rgba.png", rgba=True)))
        self.assertEqual(rgb.pixel_rgba(0, 0), (10, 20, 30, 255))
        self.assertEqual(rgba.pixel_rgba(0, 0), (10, 20, 30, 0))
        self.assertEqual(C.compare_rasters(rgb, rgba).differing_pixels, 1)

    def test_diff_refuses_transparency_in_either_input_without_a_heatmap(self):
        for transparent_side in ("reference", "candidate"):
            with self.subTest(side=transparent_side):
                manifests = []
                for side in ("reference", "candidate"):
                    image = self.png(side + ".png", transparent=side == transparent_side)
                    manifest = self.root / (side + ".json")
                    C.write_selftest_manifest(str(manifest), side, "synthetic", [{
                        "name": "scene", "png": image.name,
                        "checks": [{"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255}],
                    }])
                    manifests.append(str(manifest))
                before = {p.name: p.read_bytes() for p in self.root.iterdir()}
                errors = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
                    rc = C.main(["diff", "--reference", manifests[0], "--candidate", manifests[1]])
                self.assertEqual(rc, C.EXIT_ERROR)
                self.assertIn("tRNS", errors.getvalue())
                self.assertIn("RGBA", errors.getvalue())
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
