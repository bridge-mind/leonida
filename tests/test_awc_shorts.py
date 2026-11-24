"""Tests for tools/awc_shorts.py, built from synthetic AWC data (no game files).

    python -I tests/test_awc_shorts.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "awc_shorts.py")
    spec = importlib.util.spec_from_file_location("awc_shorts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shorts = load_tool()


def tag(typ, offset, size):
    return (typ << 56) | (size << 28) | offset


def build_music_bank(toc: int, samples: int, rate: int = 48000, *, duplicate_toc: int | None = None) -> bytes:
    """One descriptor and one block, with one packet subblock per channel."""
    tocs = (toc,) if duplicate_toc is None else (toc, duplicate_toc)
    desc = bytearray(12 + 16 * len(tocs))
    struct.pack_into("<III", desc, 0, 1, 2048 * (1 + len(tocs)), len(tocs))
    data = bytearray(2048 * (1 + len(tocs)))
    for i, packet in enumerate(tocs):
        channel = 12 + 16 * i
        struct.pack_into("<II", desc, channel, 0x1234, samples)
        struct.pack_into("<H", desc, channel + 10, rate)
        desc[channel + 12] = 0x0B
        struct.pack_into("<I", data, 24 * i + 4, 1)
        subblock = 2048 * (i + 1)
        struct.pack_into("<H", data, subblock, 1)
        data[subblock + 2] = packet
    header = 16 + 4 + 2 * 8
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header))
    out += struct.pack("<I", 0x1234 | (2 << 29))
    out += struct.pack("<Q", tag(0x48, header, len(desc)))
    out += struct.pack("<Q", tag(0x55, header + len(desc), len(data)))
    return bytes(out + desc + data)


def ogg_head_and_last_granule(data):
    if data[:4] != b"OggS":
        raise AssertionError("not Ogg")
    p = 0
    granule = None
    first_body = None
    while p < len(data):
        if data[p:p + 4] != b"OggS":
            raise AssertionError("bad Ogg page")
        g, = struct.unpack_from("<q", data, p + 6)
        nseg = data[p + 26]
        body = p + 27 + nseg
        body_size = sum(data[p + 27:p + 27 + nseg])
        if first_body is None:
            first_body = body
        granule = g
        p = body + body_size
    if first_body is None:
        raise AssertionError("empty Ogg")
    return data[first_body + 9], granule


class ShortsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def write(self, name, toc, samples):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(build_music_bank(toc, samples))
        return path

    def test_stereo_selection_and_granule(self):
        path = self.write("stereo.awc", 0xFC, 480000)   # 5 s interleaved
        rows = shorts.export_bank(path, self.out, 4, 6)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["stream"], rows[0]["channels"]), ("00001234", 2))
        self.assertAlmostEqual(rows[0]["duration_s"], 5.0, places=6)
        with open(os.path.join(self.out, "stereo", "00001234.ogg"), "rb") as f:
            channels, granule = ogg_head_and_last_granule(f.read())
        self.assertEqual(channels, 2)
        self.assertEqual(granule, 960)   # one 20 ms packet; the declared count exceeds the payload

    def test_mono_selection_and_filter(self):
        path = self.write("mono.awc", 0x78, 480000)      # 10 s mono
        self.assertEqual(shorts.export_bank(path, self.out, 9, 11)[0]["channels"], 1)
        self.assertEqual(shorts.export_bank(path, self.out, 4, 6), [])

    def test_duplicate_selected_channels_keep_both_packet_streams(self) -> None:
        path = os.path.join(self.tmp.name, "duplicate.awc")
        with open(path, "wb") as bank:
            bank.write(build_music_bank(0x78, 960, duplicate_toc=0xFC))

        rows = shorts.export_bank(path, self.out, 0, 1)
        self.assertEqual([row["stream"] for row in rows], ["00001234", "00001234"])
        self.assertEqual([row["file"] for row in rows], [
            "duplicate/00001234.ogg", "duplicate/00001234~2.ogg",
        ])
        for row, toc, channels in zip(rows, (0x78, 0xFC), (1, 2)):
            with open(os.path.join(self.out, row["file"]), "rb") as clip:
                ogg = clip.read()
            self.assertEqual(ogg[-1:], bytes([toc]))
            self.assertEqual(ogg_head_and_last_granule(ogg)[0], channels)

    def test_cli_writes_csv(self):
        path = self.write("cli.awc", 0xFC, 480000)
        argv = [path, self.out, "--min-seconds", "4", "--max-seconds", "6", "--jobs", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(shorts.main(argv), 0)
        fields, rows = None, []
        with open(os.path.join(self.out, "shorts.csv"), encoding="utf-8") as f:
            import csv
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        self.assertEqual(fields, list(shorts.FIELDS))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["channels"], "2")
        self.assertTrue(rows[0]["file"].endswith("00001234.ogg"))

    def test_bank_extensions_cannot_collapse_two_banks_into_one_export(self):
        first = self.write("clip.awc", 0x78, 480000)
        second = os.path.join(self.tmp.name, "clip.AWC")
        if os.path.exists(second):
            self.skipTest("input volume does not support case-distinct files")
        self.write("clip.AWC", 0xFC, 480000)
        os.makedirs(self.out)
        catalog = os.path.join(self.out, "shorts.csv")
        with open(catalog, "wb") as f:
            f.write(b"previous catalog\n")
        for jobs in ("1", "2"):
            with self.subTest(jobs=jobs), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(shorts.main([self.tmp.name, self.out, "--jobs", jobs]), 2)
            self.assertIn(first, err.getvalue())
            self.assertIn(second, err.getvalue())
            self.assertEqual(os.listdir(self.out), ["shorts.csv"])
            with open(catalog, "rb") as f:
                self.assertEqual(f.read(), b"previous catalog\n")

    def test_rejects_nonpositive_job_counts(self):
        for jobs in ("0", "-1"):
            with self.subTest(jobs=jobs):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(shorts.main([self.tmp.name, self.out, "--jobs", jobs]), 2)
                self.assertIn("--jobs must be at least 1", stderr.getvalue())
                self.assertFalse(os.path.exists(self.out))

    def test_bad_range(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(shorts.main(["x", self.out, "--min-seconds", "6", "--max-seconds", "5"]), 2)

    def test_missing_source_preserves_csv(self):
        path = self.write("good.awc", 0x78, 960)
        args = [self.out, "--jobs", "1", "--min-seconds", "0"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(shorts.main([path, *args]), 0)
        index = os.path.join(self.out, "shorts.csv")
        with open(index, "rb") as f:
            previous = f.read()

        missing = os.path.join(self.tmp.name, "missing.awc")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            result = shorts.main([missing, *args])
        with open(index, "rb") as f:
            self.assertEqual(f.read(), previous)
        self.assertEqual(result, 2)
        self.assertIn("missing.awc", err.getvalue())
        self.assertEqual(len(err.getvalue().splitlines()), 1)

    def test_missing_source_names_the_path_as_typed(self):
        """The path is printed as typed: repr() doubles every Windows backslash, so `C:\\a\\b` came out `C:\\\\a\\\\b`."""
        missing = os.path.join(self.tmp.name, "missing.awc")
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.assertEqual(shorts.main([missing, self.out]), 2)
        self.assertIn(missing, stderr.getvalue())

    def test_clip_write_error_preserves_catalog_in_both_worker_modes(self):
        path = self.write("blocked.awc", 0xFC, 480000)
        os.makedirs(os.path.join(self.out, "blocked", "00001234.ogg"))
        index = os.path.join(self.out, "shorts.csv")
        with open(index, "wb") as f:
            f.write(b"previous catalog\n")
        tool = os.path.join(ROOT, "tools", "awc_shorts.py")
        for jobs in ("1", "2"):
            with self.subTest(jobs=jobs):
                result = subprocess.run([sys.executable, "-I", tool, path, self.out, "--jobs", jobs],
                                        capture_output=True, text=True, errors="replace", timeout=60)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("blocked", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                with open(index, "rb") as f:
                    self.assertEqual(f.read(), b"previous catalog\n")

    def test_partial_csv_write_preserves_catalog(self):
        path = self.write("good.awc", 0xFC, 480000)
        os.makedirs(self.out)
        index = os.path.join(self.out, "shorts.csv")
        with open(index, "wb") as f:
            f.write(b"previous catalog\n")
        args = [path, self.out, "--jobs", "1"]
        with mock.patch.object(shorts.csv.DictWriter, "writerows", side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(shorts.main(args), 1)
        self.assertIn("shorts.csv", err.getvalue())
        with open(index, "rb") as f:
            self.assertEqual(f.read(), b"previous catalog\n")
        self.assertEqual(sorted(os.listdir(self.out)), ["good", "shorts.csv"])

    def test_nan_duration_bounds(self):
        path = self.write("good.awc", 0x78, 960)
        for option in ("--min-seconds", "--max-seconds"):
            with self.subTest(option=option):
                args = [path, self.out, "--jobs", "1", option, "nan"]
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(shorts.main(args), 2)
                self.assertFalse(os.path.exists(self.out))
                self.assertIn(option, err.getvalue())
                self.assertEqual(len(err.getvalue().splitlines()), 1)

    def test_nested_banks_keep_both_clips(self):
        for folder, toc in (("left", 0x78), ("right", 0xFC)):
            os.mkdir(os.path.join(self.tmp.name, folder))
            self.write(os.path.join(folder, "shared.awc"), toc, 480000)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(shorts.main([self.tmp.name, self.out, "--jobs", "1"]), 0)
        for folder, expected in (("left", 1), ("right", 2)):
            with open(os.path.join(self.out, folder, "shared", "00001234.ogg"), "rb") as f:
                channels, _ = ogg_head_and_last_granule(f.read())
            self.assertEqual(channels, expected)

    def check_truncated_bank_skipped(self, tool, jobs):
        os.mkdir(os.path.join(self.tmp.name, "banks"))
        self.write(os.path.join("banks", "good.awc"), 0xFC, 480000)
        # Header says 5 streams but the bank ends after 16 bytes: parse_awc raises struct.error.
        with open(os.path.join(self.tmp.name, "banks", "bad.awc"), "wb") as f:
            f.write(b"ADAT" + struct.pack("<III", 0, 5, 16))
        argv = [os.path.join(self.tmp.name, "banks"), self.out, "--min-seconds", "4",
                "--max-seconds", "6", "--jobs", str(jobs)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(tool.main(argv), 0)
        self.assertIn("  skip bad: ", err.getvalue())
        self.assertEqual(len(err.getvalue().splitlines()), 1)
        import csv
        with open(os.path.join(self.out, "shorts.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["bank"], r["stream"]) for r in rows], [("good", "00001234")])

    def test_truncated_bank_skipped_serial(self):
        self.check_truncated_bank_skipped(shorts, 1)

    def test_truncated_bank_skipped_pool(self):
        # A spawned worker must import the tool by name, so use the real module, not the spec-loaded copy.
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        self.addCleanup(sys.path.remove, os.path.join(ROOT, "tools"))
        import awc_shorts
        self.check_truncated_bank_skipped(awc_shorts, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
