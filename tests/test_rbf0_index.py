"""Synthetic-header tests for tools/rbf0_index.py.

    python -I tests/test_rbf0_index.py

Every byte here is built by the test. No game file, game asset path or game name is used.
"""
import contextlib
import csv
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rbf0_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_index  # noqa: E402


@contextlib.contextmanager
def tracked_mmaps():
    """Record every mmap.mmap opened by rbf0_index and close any leak on exit."""
    real_mmap = rbf0_index.mmap.mmap
    created = []

    def wrap(*args, **kwargs):
        mapping = real_mmap(*args, **kwargs)
        created.append(mapping)
        return mapping

    try:
        with mock.patch.object(rbf0_index.mmap, "mmap", side_effect=wrap):
            yield created
    finally:
        for mapping in created:
            if not mapping.closed:
                mapping.close()

WIDE = b"RBF0" + rbf0_index.PROLOGUE


def _name(name):
    return struct.pack("<H", len(name)) + name


def _open(index, name, attrs=0):
    return bytes([index, 0x00]) + _name(name) + struct.pack("<HHH", 0, 0, attrs)


def _value(index, kind, payload=b"", name=None):
    out = bytes([index, kind])
    if name is not None:
        out += _name(name)
    return out + payload


def _text(text):
    raw = text + b"\x00"
    return b"\xfd\xff" + struct.pack("<I", len(raw)) + raw


def wide_blob():
    """One root with an integer, a float, a string, a bool and text; all descriptors are introduced in-line."""
    b = bytearray(WIDE)
    b += _open(0, b"TestRoot")
    b += _value(1, 0x10, struct.pack("<Q", 7), b"count")
    b += _value(2, 0x40, struct.pack("<d", 0.5), b"ratio")
    b += _value(3, 0x60, struct.pack("<H", 5) + b"alpha", b"label")
    b += _value(4, 0x20, b"", b"flag")
    b += _text(b"body")
    b += b"\xff\xff"
    return bytes(b)


def narrow_blob():
    """The 32-bit layout: no prologue, 4-byte integers and floats."""
    b = bytearray(b"RBF0")
    b += _open(0, b"TestRoot")
    b += _value(1, 0x10, struct.pack("<I", 7), b"count")
    b += _value(2, 0x40, struct.pack("<f", 0.5), b"ratio")
    b += b"\xff\xff"
    return bytes(b)


def large_blob():
    """A blob bigger than one 64 KiB chunk, with a string that crosses the boundary."""
    b = bytearray(WIDE)
    b += _open(0, b"BigRoot")
    filler = b"x" * 4000
    first = True
    while len(b) < 70000:
        b += _value(1, 0x60, struct.pack("<H", len(filler)) + filler, b"filler" if first else None)
        first = False
    if len(b) > 64000:
        b += _value(2, 0x60, struct.pack("<H", 40) + b"y" * 40, b"tail")
    b += b"\xff\xff"
    return bytes(b)


class ParseTest(unittest.TestCase):
    def test_wide_header_and_counts(self):
        r = rbf0_index.parse_rbf0(wide_blob())
        self.assertEqual(r["layout"], "wide-64")
        self.assertEqual(r["prologue"], "fbff0100")
        self.assertEqual(r["root"], "TestRoot")
        self.assertEqual((r["descriptors"], r["elements"], r["values"], r["texts"]),
                         (5, 1, 4, 1))
        self.assertEqual((r["ints"], r["floats"], len(r["strings"])), ([7], [0.5], 2))
        self.assertEqual(r["depth"], 1)
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["complete"])
        self.assertEqual(r["end"], len(wide_blob()))
        self.assertEqual(r["descriptor_names"],
                         ["TestRoot", "count", "ratio", "label", "flag"])
        self.assertEqual(r["attributes"], 0)
        self.assertEqual(r["types"], {"open": 1, "int": 1, "float": 1, "string": 1,
                                      "true": 1, "text": 1})

    def test_narrow_header_and_widths(self):
        r = rbf0_index.parse_rbf0(narrow_blob())
        self.assertEqual(r["layout"], "narrow-32")
        self.assertEqual(r["prologue"], "")
        self.assertEqual((r["ints"], r["floats"]), ([7], [0.5]))
        self.assertEqual(r["status"], "ok")

    def test_truncated(self):
        r = rbf0_index.parse_rbf0(wide_blob()[:26])
        self.assertEqual(r["status"], "truncated")
        self.assertFalse(r["complete"])
        self.assertEqual(r["root"], "TestRoot")
        self.assertIn("truncated", r["error"])
    def test_trailing_bytes(self):
        r = rbf0_index.parse_rbf0(wide_blob() + b"\x00\x00")
        self.assertEqual(r["status"], "trailing")
        self.assertIn("after the root closed", r["error"])

    def test_bad_descriptor_index(self):
        b = WIDE + bytes([9, 0x10]) + struct.pack("<Q", 1)
        r = rbf0_index.parse_rbf0(b)
        self.assertEqual(r["status"], "bad")
        self.assertIn("descriptor index", r["error"])

    def test_bad_element_word(self):
        b = WIDE + bytes([0, 0]) + _name(b"Root")
        b += struct.pack("<HHH", 1, 0, 0)  # word 0 must be zero
        b += b"\xff\xff"
        r = rbf0_index.parse_rbf0(b)
        self.assertEqual(r["status"], "bad")
        self.assertIn("word is not zero", r["error"])

    def test_not_rbf0(self):
        with self.assertRaises(rbf0_index.RbfError):
            rbf0_index.parse_rbf0(b"PSIN" + bytes(16))

    def test_attribute_mismatch_counted_once_per_element(self):
        # One element declares 1 attribute, has 0 attributes, three child elements, and a trailing value:
        # must report 1 attribute mismatch at the first content transition (not 3 on main).
        b = bytearray(b"RBF0")
        b += _open(0, b"Root", attrs=1)
        b += _open(1, b"A") + b"\xff\xff"
        b += _open(2, b"B") + b"\xff\xff"
        b += _open(3, b"C") + b"\xff\xff"
        b += _value(4, 0x10, struct.pack("<I", 9), b"after")
        b += b"\xff\xff"
        r = rbf0_index.parse_rbf0(bytes(b))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["attributes"], 1)

        # Truncated after content starts must still record the already-observed attribute mismatch once.
        trunc = bytearray(b"RBF0") + _open(0, b"Root", attrs=1) + _text(b"hi")
        rt = rbf0_index.parse_rbf0(bytes(trunc))
        self.assertEqual(rt["status"], "truncated")
        self.assertEqual(rt["attributes"], 1)

    def test_attribute_count(self):
        b = bytearray(WIDE)
        b += _open(0, b"Root", attrs=2)
        b += _value(1, 0x10, struct.pack("<Q", 1), b"first")
        b += _value(2, 0x20, b"", b"second")
        b += _open(3, b"Child")
        b += b"\xff\xff\xff\xff"
        r = rbf0_index.parse_rbf0(bytes(b))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["elements"], 2)
        self.assertEqual(r["attributes"], 0)
        # A close before both declared attributes is a mismatch, not a parse failure.
        broken = WIDE + _open(0, b"Root", attrs=1) + b"\xff\xff"
        self.assertEqual(rbf0_index.parse_rbf0(broken)["attributes"], 1)


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_join_at_64k(self):
        blob = large_blob()
        self.assertGreater(len(blob), 65536)
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(blob[:65536])
        with open(os.path.join(self.root, "00000001.bin"), "wb") as f:
            f.write(blob[65536:])
        rows, records, summary = rbf0_index.scan_dir(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["chunks"], 2)
        self.assertEqual(records[0]["size"], len(blob))
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["root"], "BigRoot")
        self.assertEqual(summary["rbf0"]["multi_chunk"], 1)

    def test_archive_join_at_64k(self):
        blob = large_blob()
        path = os.path.join(self.root, "synthetic.rpf")
        header = bytearray(0x100)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = zlib.compress(blob[:65536]) + zlib.compress(blob[65536:]) + bytes(64)
        with open(path, "wb") as f:
            f.write(header + body)
        rows, records, summary = rbf0_index.scan_rpf(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["chunks"], 2)
        self.assertEqual(records[0]["size"], len(blob))
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(summary["streams"], 2)
        self.assertGreater(summary["readable_fraction"], 0.5)

    def test_single_stream_directory(self):
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        with open(os.path.join(self.root, "00000001.bin"), "wb") as f:
            f.write(b"PSIN" + bytes(60))
        rows, records, summary = rbf0_index.scan_dir(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["chunks"], 1)
        self.assertEqual(summary["streams"], 2)

    def test_directory_skips_the_carvers_index_and_reports(self):
        """rpf9_carve.py writes index.json next to the streams; it and a report are not payloads."""
        with open(os.path.join(self.root, "00000000.bin"), "wb") as f:
            f.write(wide_blob())
        with open(os.path.join(self.root, "00000001.bin"), "wb") as f:
            f.write(b"PSIN" + bytes(60))
        with open(os.path.join(self.root, "index.json"), "w", encoding="utf-8") as f:
            f.write('[{"off": 0, "stored": 16}]')
        with open(os.path.join(self.root, "report.csv"), "w", encoding="utf-8") as f:
            f.write("offset,size\n")
        rows, records, summary = rbf0_index.scan_dir(self.root)
        self.assertEqual(summary["streams"], 2)
        self.assertEqual(sorted(r["offset"] for r in rows), [0, 1])


class ArgumentTest(unittest.TestCase):
    """Out-of-range scan arguments are refused before anything is written."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rpf = os.path.join(self.root, "one.rpf")
        with open(self.rpf, "wb") as f:
            f.write(b"RBF0" + struct.pack("<I", 0x20) + bytes(0x20))

    def tearDown(self):
        self.tmp.cleanup()

    def test_bad_scan_arguments_fail_before_an_index_is_written(self):
        for flag, value, message in (("--start", "-1", "must be 0 or more"),
                                     ("--limit", "0", "must be at least 1"),
                                     ("--limit", "-100", "must be at least 1")):
            out = os.path.join(self.root, "bad.json")
            p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet", flag, value],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(message, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_rejects_archive_ranges(self):
        out = os.path.join(self.root, "index.json")
        for flag in ("--start", "--limit"):
            with self.subTest(flag=flag):
                result = self.run_tool(self.root, out, flag, "1")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("apply to a single archive", result.stderr)
                self.assertFalse(os.path.exists(out))

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args, "--quiet"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_negative_string_count_fails_before_an_index_is_written(self):
        carve = os.path.join(self.root, "carve")
        os.mkdir(carve)
        blob = WIDE + _open(0, b"TestRoot")
        blob += _value(1, 0x60, _name(b"a"), b"label")
        blob += _value(1, 0x60, _name(b"b"))
        blob += _value(1, 0x60, _name(b"c"))
        blob += b"\xff\xff"
        with open(os.path.join(carve, "00000000.bin"), "wb") as f:
            f.write(blob)
        out = os.path.join(self.root, "out.json")

        p = self.run_tool(carve, out, "--strings", "-1")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("--strings", p.stderr)
        self.assertIn("must be 0 or more", p.stderr)
        self.assertEqual(len(p.stderr.splitlines()), 1)
        self.assertFalse(os.path.exists(out))

    def test_missing_source_is_an_error_not_a_traceback(self):
        out = os.path.join(self.root, "missing.json")
        p = self.run_tool(os.path.join(self.root, "nope.rpf"), out)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("nope.rpf", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertFalse(os.path.exists(out))

    def test_directory_report_cannot_replace_a_carve_input(self):
        manifest = os.path.join(self.root, "index.json")
        stream = os.path.join(self.root, "00000001.bin")
        for path, original in ((manifest, b'{"files": []}'), (stream, b"synthetic stream")):
            with self.subTest(path=path):
                with open(path, "wb") as f:
                    f.write(original)
                result = self.run_tool(self.root, path)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("would overwrite", result.stderr)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), original)

    def test_directory_report_cannot_replace_a_stream_via_a_hard_link(self):
        stream = os.path.join(self.root, "00000001.bin")
        out = os.path.join(self.root, "report.csv")
        with open(stream, "wb") as f:
            f.write(b"synthetic stream")
        try:
            os.link(stream, out)
        except (OSError, NotImplementedError) as e:
            self.skipTest(str(e))
        result = self.run_tool(self.root, out)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("would overwrite", result.stderr)
        with open(stream, "rb") as f:
            self.assertEqual(f.read(), b"synthetic stream")

    def test_unwritable_output_path_fails_before_the_scan(self):
        for out in (os.path.join(self.root, "no_such_folder", "x.json"), self.root):
            p = self.run_tool(self.rpf, out)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "no_such_folder")))

    def test_scan_rpf_closes_mmap_on_invalid_start(self):
        with tracked_mmaps() as created:
            with self.assertRaisesRegex(ValueError, "outside the archive"):
                rbf0_index.scan_rpf(self.rpf, start=os.path.getsize(self.rpf) + 1)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(self.rpf)
            self.assertFalse(os.path.exists(self.rpf))


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.records = [{
            "offset": 0x1000, "stored": 60, "size": 98, "chunks": 1, "layout": "wide-64",
            "prologue": "fbff0100", "root": "TestRoot", "descriptors": 5, "elements": 1,
            "values": 5, "texts": 1, "strings": 2, "ints": 1, "floats": 1, "depth": 1,
            "attributes": 0, "types": {"open": 1}, "status": "ok", "error": "",
            "descriptor_names": ["TestRoot"], "sample": ["alpha", "body"],
            "_all_strings": ["alpha", "body"],
        }]
        self.summary = {"mode": "directory", "rbf0": {"resources": 1}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv(self):
        path = os.path.join(self.root, "out.csv")
        rbf0_index.write_output(path, self.records, self.summary)
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["offset"], "0x1000")
        self.assertEqual(rows[0]["root"], "TestRoot")
        self.assertEqual(rows[0]["sample"], "alpha; body")

    def test_json(self):
        path = os.path.join(self.root, "out.json")
        rbf0_index.write_output(path, self.records, self.summary)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["rbf0"]["resources"], 1)
        self.assertEqual(doc["files"][0]["string_count"], 2)
        self.assertEqual(doc["files"][0]["sample_strings"], ["alpha", "body"])
        self.assertFalse(doc["files"][0]["sample_truncated"])

    def test_json_all_strings(self):
        path = os.path.join(self.root, "out.json")
        self.records[0]["sample"] = ["alpha"]
        rbf0_index.write_output(path, self.records, self.summary, all_strings=True)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["files"][0]["sample_strings"], ["alpha", "body"])


class ScanDirCollisionAndTrailingTest(unittest.TestCase):
    def _make_rbf0(self, root_name):
        raw = root_name.encode("utf-8")
        return b"RBF0\x00\x00" + struct.pack("<H", len(raw)) + raw + struct.pack("<HHH", 0, 0, 0) + b"\xff\xff"

    def test_scan_dir_keeps_each_file_on_offset_collisions(self):
        with tempfile.TemporaryDirectory() as d:
            # Non-hex name used to get offset=len(entries)=1, colliding with 00000001.bin, and
            # duplicate hex prefixes (00001000.bin + 00001000_alt.bin) used to overwrite in by_offset.
            with open(os.path.join(d, "00000001.bin"), "wb") as f:
                f.write(self._make_rbf0("Alpha"))
            with open(os.path.join(d, "extra.bin"), "wb") as f:
                f.write(b"NOT_RBF0_PAYLOAD" * 8)
            with open(os.path.join(d, "00001000.bin"), "wb") as f:
                f.write(self._make_rbf0("First"))
            with open(os.path.join(d, "00001000_alt.bin"), "wb") as f:
                f.write(self._make_rbf0("Second"))
            _, records, _ = rbf0_index.scan_dir(d)
        self.assertEqual([r["root"] for r in records], ["Alpha", "First", "Second"])

    def test_join_resource_stops_on_trailing_status(self):
        hdr = b"RBF0\x00\x00" + struct.pack("<H", 4) + b"Root" + struct.pack("<HHH", 0, 0, 0)
        full = hdr + b"\xfd\xff" + struct.pack("<I", 70000) + (b"A" * 70000) + b"\xff\xff"
        c1, c2 = full[:65536], full[65536:].ljust(65536, b"\x00")
        c3 = b"UNRELATED_NON_RBF0_CHUNK" + b"\x00" * 100
        rows = [
            {"offset": 0, "stored": len(c1), "size": len(c1), "head": c1[:8]},
            {"offset": len(c1), "stored": len(c2), "size": len(c2), "head": c2[:8]},
            {"offset": len(c1) + len(c2), "stored": len(c3), "size": len(c3), "head": c3[:8]},
        ]
        blobs = {0: c1, len(c1): c2, len(c1) + len(c2): c3}
        blob, chunks, parsed = rbf0_index.join_resource(rows, 0, lambda r: blobs[r["offset"]], check_gap=False)
        self.assertEqual(chunks, 2)
        self.assertEqual(len(blob), 131072)
        self.assertEqual(parsed["status"], "trailing")


if __name__ == "__main__":
    unittest.main()
