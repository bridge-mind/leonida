"""tools/mask_channels.py on synthetic images: every texel is built here, no game data."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mask_channels as mc  # noqa: E402
import report_paths  # noqa: E402

W = H = 16


def image():
    """R: one 8x8 square (a broad layer). G: constant 200. B: isolated texels on a 4-texel grid (spots). A: 255."""
    px = bytearray()
    for y in range(H):
        for x in range(W):
            r = 255 if 4 <= x < 12 and 4 <= y < 12 else 0
            b = 255 if x % 4 == 1 and y % 4 == 1 else 0
            px += bytes((r, 200, b, 255))
    return bytes(px)


def rgba8_dds(width, height, payload):
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<IIII", header, 4, 124, 0x1007, height, width)
    struct.pack_into("<II", header, 76, 32, 0x4)                       # DDPF_FOURCC
    header[84:88] = b"DX10"
    return bytes(header) + struct.pack("<IIIII", 28, 3, 0, 1, 0) + payload  # R8G8B8A8_UNORM


class MaskChannelTests(unittest.TestCase):
    def test_blobs(self):
        mask = bytearray(9)
        for i in (0, 1, 3, 8):                                            # 3x3: an L of three and a corner
            mask[i] = 1
        self.assertEqual(mc.blobs(mask, 3, 3), [3, 1])
        row_end = bytearray([0, 0, 1, 1, 0, 0])                         # 3 wide: (2,0) and (0,1) don't touch
        self.assertEqual(mc.blobs(row_end, 3, 2), [1, 1])
        row_start = bytearray([1, 0, 1, 1, 0, 0, 0, 0, 0])               # fill from (0,0) reaches (0,1) first;
        self.assertEqual(mc.blobs(row_start, 3, 3), [2, 1])             # its left step must not wrap to (2,0)

    def test_channel_counts(self):
        c = mc.channel_counts(W, H, image(), (128,))
        self.assertEqual((c["R"]["thresholds"][128]["blobs"], c["R"]["thresholds"][128]["largest_share"]), (1, 1.0))
        self.assertEqual(c["R"]["thresholds"][128]["cover"], 0.25)
        self.assertEqual(c["B"]["thresholds"][128]["blobs"], 16)
        self.assertEqual(c["B"]["thresholds"][128]["largest_share"], 0.0625)
        self.assertEqual((c["G"]["min"], c["G"]["max"], c["G"]["distinct"]), (200, 200, 1))
        self.assertEqual(c["G"]["thresholds"][128], {"cover": 1.0, "blobs": 1, "largest_share": 1.0})
        self.assertEqual((c["A"]["min"], c["A"]["max"]), (255, 255))
        self.assertEqual(mc.channel_counts(W, H, image(), (250,))["G"]["thresholds"][250]["blobs"], 0)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = os.path.join(tmp, "m.dds"), os.path.join(tmp, "m.json")
            with open(path, "wb") as fh:
                fh.write(rgba8_dds(W, H, image()))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(mc.main([path, "--thresholds", "128", "--json", out]), 0)
            self.assertIn("16x16", buf.getvalue())
            with open(out, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["channels"]["B"]["thresholds"]["128"]["blobs"], 16)
            for bad in ("0", "300", "x"):
                with self.subTest(bad=bad), redirect_stdout(io.StringIO()), \
                        self.assertRaises(SystemExit) as cm:
                    sys.stderr, saved_err = io.StringIO(), sys.stderr
                    try:
                        mc.main([path, "--thresholds", bad])
                    finally:
                        sys.stderr = saved_err
                self.assertEqual(cm.exception.code, 2)

    def test_json_folder_is_created(self):
        """--json under a folder that does not exist yet gets the report, not a traceback, and nothing else."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.dds")
            with open(path, "wb") as fh:
                fh.write(rgba8_dds(W, H, image()))
            reports = os.path.join(tmp, "reports")
            out = os.path.join(reports, "m.json")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(mc.main([path, "--json", out]), 0)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["width"], W)
            self.assertEqual(os.listdir(reports), ["m.json"])                # no staging file left behind

    def test_json_write_error_is_one_line(self):
        """A --json path the tool cannot write is one ERROR line and exit 2, as the other report tools do."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.dds")
            with open(path, "wb") as fh:
                fh.write(rgba8_dds(W, H, image()))
            out = os.path.join(path, "r.json")                              # its folder is the input file
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(mc.main([path, "--json", out]), 2)
            self.assertEqual(err.getvalue().count("\n"), 1)
            self.assertTrue(err.getvalue().startswith(f"ERROR: {out}: "), err.getvalue())

    def test_json_cannot_replace_the_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.dds")
            original = rgba8_dds(W, H, image())
            with open(path, "wb") as fh:
                fh.write(original)
            aliases = {"same path": path, "other spelling": os.path.join(tmp, ".", "m.dds")}
            case_alias = os.path.join(tmp, "M.dds")
            if report_paths.same_report_path(path, case_alias):
                aliases["case-only alias"] = case_alias
            link = os.path.join(tmp, "hard.json")
            try:
                os.link(path, link)
                aliases["hard link"] = link
            except OSError:
                pass                                                    # file system without hard links
            sym = os.path.join(tmp, "sym.json")
            try:
                os.symlink(path, sym)
                aliases["symlink"] = sym
            except (OSError, NotImplementedError):
                pass                                                    # symlinks may require privileges on Windows
            for name, out in aliases.items():
                with self.subTest(alias=name), redirect_stdout(io.StringIO()), \
                        self.assertRaises(SystemExit) as cm:
                    sys.stderr, saved_err = io.StringIO(), sys.stderr
                    try:
                        mc.main([path, "--json", out])
                    finally:
                        sys.stderr = saved_err
                self.assertEqual(cm.exception.code, 2)
                with open(path, "rb") as fh:
                    self.assertEqual(fh.read(), original)

    def test_an_unreadable_input_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "fake.dds")
            with open(fake, "wb") as fh:
                fh.write(b"garbage")
            stderr, stdout = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(mc.main([fake]), 2)
            self.assertIn("error: cannot read", stderr.getvalue())
            self.assertIn("DDS header", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(mc.main([os.path.join(tmp, "no_such.dds")]), 2)
            self.assertIn("No such file or directory", stderr.getvalue())

    def test_the_json_guard_runs_before_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.dds")
            with open(path, "wb") as fh:
                fh.write(rgba8_dds(W, H, image()))
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(mc.main([path, "--json", tmp]), 2)               # tmp is a directory
            self.assertIn("is a directory", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")            # refused before any channel work
            out = os.path.join(tmp, "new", "folder", "m.json")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(mc.main([path, "--json", out]), 0)
            self.assertTrue(os.path.isfile(out))

    def test_uses_shared_same_report_path(self):
        self.assertIs(mc.same_report_path, report_paths.same_report_path)
        self.assertFalse(hasattr(mc, "same_file"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.dds")
            original = rgba8_dds(W, H, image())
            with open(path, "wb") as fh:
                fh.write(original)
            out = os.path.join(tmp, "M.dds")
            with mock.patch.object(mc, "same_report_path", return_value=True) as guard, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as cm:
                mc.main([path, "--json", out])
            self.assertEqual(cm.exception.code, 2)
            guard.assert_called_once_with(path, out)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), original)


if __name__ == "__main__":
    unittest.main()
