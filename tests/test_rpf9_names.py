"""Synthetic tests for tools/rpf9_names.py.

    python -I tests/test_rpf9_names.py

Every byte is built by the test; no game file or game asset name is used.
"""
import contextlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_names.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rpf9_names  # noqa: E402


@contextlib.contextmanager
def tracked_mmaps():
    """Record every mmap.mmap opened by rpf9_names and close any leak on exit."""
    real_mmap = rpf9_names.mmap.mmap
    created = []

    def wrap(*args, **kwargs):
        mapping = real_mmap(*args, **kwargs)
        created.append(mapping)
        return mapping

    try:
        with mock.patch.object(rpf9_names.mmap, "mmap", side_effect=wrap):
            yield created
    finally:
        for mapping in created:
            if not mapping.closed:
                mapping.close()


def stream(text):
    """A zlib stream of at least 64 bytes so the scanner accepts it."""
    data = text + b" " * max(0, 64 - len(text))
    return zlib.compress(data)


def make_archive(path, magic=b"RPF9"):
    a = stream(b"prefix alpha.#dr suffix")
    b = stream(b"bravo.#ft")
    c = stream(b"charlie.#dd")
    d = stream(b"no resource names here")
    body = a + b + c + d
    data = bytearray(0x40 + len(body) + 4096)
    data[0:4] = magic
    struct.pack_into("<I", data, 4, 0x40)
    data[0x40:0x40 + len(body)] = body
    with open(path, "wb") as f:
        f.write(data)
    return [a, b, c, d]


def bink2(inner=b"", total=96, w=64, h=48, rev=b"i"):
    """A synthetic Bink 2 video: 'KB2' + revision, u32 size-8, width/height at +0x14, then ``inner``."""
    v = bytearray(total)
    v[0:4] = b"KB2" + rev
    struct.pack_into("<I", v, 4, total - 8)
    struct.pack_into("<II", v, 0x14, w, h)
    v[0x20:0x20 + len(inner)] = inner
    return bytes(v)


def make_blob(path, *parts):
    """An RPF9 archive whose data area (at 0x40) is ``parts`` laid end to end, then 4096 zero bytes."""
    body = b"".join(parts)
    data = bytearray(0x40 + len(body) + 4096)
    data[0:4] = b"RPF9"
    struct.pack_into("<I", data, 4, 0x40)
    data[0x40:0x40 + len(body)] = body
    with open(path, "wb") as f:
        f.write(data)


class NamesTest(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args],
                              capture_output=True, text=True)

    def test_finds_names_and_counts_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc)
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout, "")
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            summary = doc["summary"]
            self.assertEqual(summary["streams"], 4, summary)
            self.assertEqual(summary["streams_with_names"], 3, summary)
            self.assertEqual(summary["unique_names"], 3, summary)
            self.assertEqual(summary["by_type"], {"dd": 1, "dr": 1, "ft": 1}, summary)
            names = sorted(n for row in doc["files"] for n in row["names"])
            self.assertEqual(names, ["alpha.#dr", "bravo.#ft", "charlie.#dd"])
            self.assertEqual(len(doc["files"]), 3)

    def test_accepts_reversed_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc, magic=b"9FPR")
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout, "")
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["unique_names"], 3, doc["summary"])

    def test_types_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc)
            p = self.run_tool(arc, out, "--types", "ft", "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            names = [n for row in doc["files"] for n in row["names"]]
            self.assertEqual(names, ["bravo.#ft"])
            self.assertEqual(doc["summary"]["by_type"], {"ft": 1})

    def test_max_inflate_hides_late_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            data = b"x" * 40 + b"late.#dr" + b"y" * 40
            a = zlib.compress(data)
            blob = bytearray(0x40 + len(a) + 4096)
            blob[0:4] = b"RPF9"
            struct.pack_into("<I", blob, 4, 0x40)
            blob[0x40:0x40 + len(a)] = a
            with open(arc, "wb") as f:
                f.write(blob)
            p = self.run_tool(arc, out, "--max-inflate", "32", "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["streams_with_names"], 0, doc["summary"])
            self.assertEqual(doc["files"], [])

    def test_invalid_arguments_do_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            cases = [
                ("--start", "-1"),
                ("--start", "nope"),
                ("--limit", "0"),
                ("--limit", "nope"),
                ("--max-inflate", "0"),
                ("--max-inflate", "nope"),
                ("--types", "bad suffix"),
            ]
            for i, args in enumerate(cases):
                out = os.path.join(tmp, f"out{i}.json")
                p = self.run_tool(arc, out, *args)
                self.assertEqual(p.returncode, 2, (args, p.stdout, p.stderr))
                self.assertNotIn("<lambda>", p.stderr)
                self.assertFalse(os.path.exists(out), args)

    def test_scan_limit_excludes_crossing_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            streams = make_archive(arc)
            stored = len(streams[0])
            rows, summary = rpf9_names.scan_archive(arc, start=0x40, limit=stored)
            self.assertEqual(len(rows), 1)
            self.assertEqual(summary["streams"], 1)
            self.assertEqual(summary["unread_bytes"], 0)

            rows, summary = rpf9_names.scan_archive(arc, start=0x40, limit=stored - 1)
            self.assertEqual(rows, [])
            self.assertEqual(summary["streams"], 0)
            self.assertEqual(summary["unread_bytes"], stored - 1)
            self.assertEqual(summary["span"], stored - 1)

    def test_missing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            p = self.run_tool(os.path.join(tmp, "missing.rpf"), out, "--quiet")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_empty_archive_raises_not_rpf9(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            with tracked_mmaps() as created:
                with self.assertRaisesRegex(ValueError, "not an RPF9 archive"):
                    rpf9_names.scan_archive(empty)
                self.assertEqual(created, [])

    def test_scan_archive_closes_mmap_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            with tracked_mmaps() as created:
                rows, summary = rpf9_names.scan_archive(arc)
                self.assertEqual(len(rows), 3)
                self.assertEqual(summary["streams"], 4)
                self.assertEqual(len(created), 1)
                self.assertTrue(created[0].closed)
                os.remove(arc)
                self.assertFalse(os.path.exists(arc))

    def test_bink2_video_is_skipped_with_its_embedded_stream(self):
        before = stream(b"before.#dr")
        inner = stream(b"hidden.#ft")
        video = bink2(inner, total=0x20 + len(inner) + 16)
        after = stream(b"after.#dd")
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "v.rpf")
            make_blob(arc, before, bytes(5), video, after)
            rows, summary = rpf9_names.scan_archive(arc)
        names = sorted(n for row in rows for n in row["names"])
        self.assertEqual(names, ["after.#dd", "before.#dr"], summary)
        self.assertEqual(summary["streams"], 2, summary)
        self.assertEqual(summary["bink2_files"], 1, summary)
        self.assertEqual(summary["bink2_bytes"], len(video), summary)
        self.assertEqual(sum(r["stored"] for r in rows), len(before) + len(after))
        self.assertEqual(summary["zero_padding"] + summary["unread_bytes"] + len(before) + len(after)
                         + summary["bink2_bytes"], summary["span"], summary)
        self.assertEqual(summary["zero_padding"], 5 + 4096, summary)
        self.assertEqual(summary["unread_bytes"], 0, summary)

    def test_bink2_gap_before_video_counts_as_unread(self):
        video = bink2()
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "v.rpf")
            make_blob(arc, b"junk!", video)
            rows, summary = rpf9_names.scan_archive(arc)
        self.assertEqual(summary["bink2_files"], 1, summary)
        self.assertEqual(summary["unread_bytes"], 5, summary)
        self.assertEqual(summary["zero_padding"], 4096, summary)
        self.assertEqual(summary["unread_bytes"] + summary["zero_padding"] + summary["bink2_bytes"],
                         summary["span"], summary)

    def test_implausible_bink2_header_is_not_a_video(self):
        good = stream(b"real.#dr")
        bad_headers = {
            "zero_width": bink2(w=0),
            "huge_height": bink2(h=9000),
            "runs_past_end": bink2()[:4] + struct.pack("<I", 1 << 20) + bink2()[8:],
            "too_short": bink2()[:4] + struct.pack("<I", 8) + bink2()[8:],
        }
        for label, bad in bad_headers.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                arc = os.path.join(tmp, "v.rpf")
                make_blob(arc, bad, good)
                rows, summary = rpf9_names.scan_archive(arc)
                self.assertEqual(summary["bink2_files"], 0, summary)
                self.assertEqual(summary["bink2_bytes"], 0, summary)
                self.assertEqual([n for r in rows for n in r["names"]], ["real.#dr"], summary)
                self.assertEqual(summary["zero_padding"] + summary["unread_bytes"] + len(good),
                                 summary["span"], summary)

    def test_limit_bounds_inflation_to_the_window(self):
        first = stream(b"first.#dr")
        second = stream(b"second.#ft")
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "l.rpf")
            make_blob(arc, first, second)
            rows, summary = rpf9_names.scan_archive(arc, limit=len(first) - 1)
            self.assertEqual(rows, [])
            self.assertEqual(summary["streams"], 0, summary)
            self.assertEqual(summary["span"], len(first) - 1)
            self.assertEqual(summary["unread_bytes"], len(first) - 1, summary)
            self.assertEqual(summary["zero_padding"], 0, summary)
            rows, summary = rpf9_names.scan_archive(arc, limit=len(first))
            self.assertEqual([n for r in rows for n in r["names"]], ["first.#dr"], summary)
            self.assertEqual(rows[0]["stored"], len(first))
            self.assertEqual(summary["streams"], 1, summary)
            self.assertEqual(summary["span"], len(first), summary)
            self.assertEqual(summary["zero_padding"] + summary["unread_bytes"], 0, summary)

    def _assert_validation_error_closes_mmap(self, arc, start, message):
        """Assert scan_archive raises ValueError and closes its mmap before returning."""
        with tracked_mmaps() as created:
            with self.assertRaisesRegex(ValueError, message):
                rpf9_names.scan_archive(arc, start=start)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(arc)
            self.assertFalse(os.path.exists(arc))

    def test_positive_max_length_and_shared_read_offsets(self):
        # The first stream inflates to 65 bytes and, with a 64-byte feed cap, both compressed
        # streams fit in one read; at completed EOF unconsumed_tail and unused_data both hold
        # the next stream, so a flush would double unused_data and misplace the next offset.
        real_decompressobj = rpf9_names.zlib.decompressobj
        seen = []

        class Rec:
            def __init__(self, obj):
                self._obj = obj

            def decompress(self, data, max_length=0):
                seen.append(max_length)
                return self._obj.decompress(data, max_length)

            def __getattr__(self, name):
                return getattr(self._obj, name)

        first = zlib.compress(b"alpha.#dr " + b"x" * 55)  # 65 bytes inflated
        nxt = stream(b"bravo.#ft")
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "t.rpf")
            make_blob(arc, first, nxt)
            with mock.patch.object(rpf9_names.zlib, "decompressobj",
                                   lambda: Rec(real_decompressobj())):
                with mock.patch.object(rpf9_names, "DECOMPRESS_MAX", 64, create=True):
                    rows, summary = rpf9_names.scan_archive(arc)
        self.assertTrue(seen, "decompress should have been called")
        self.assertTrue(all(m > 0 for m in seen), seen)
        self.assertEqual(summary["streams"], 2, summary)
        self.assertEqual(rows[0]["size"], 65, rows)
        self.assertEqual(rows[0]["offset"], 0x40, rows)
        self.assertEqual(rows[0]["names"], ["alpha.#dr"], rows)
        self.assertEqual(rows[1]["offset"], 0x40 + len(first), rows)
        self.assertEqual(rows[1]["names"], ["bravo.#ft"], rows)

    def test_name_across_first_feed_boundary(self):
        first = stream(b"alpha.#dr")
        # With level 0 this arrives almost verbatim: the second name straddles
        # the first compressed feed and the final stream shares its last read.
        text = b" " * (rpf9_names.INITIAL_FEED - 10) + b"bravo.#ft " + b" " * 9000
        second = zlib.compress(text, level=0)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "t.rpf")
            make_blob(arc, first, second, stream(b"charlie.#dd"))
            rows, summary = rpf9_names.scan_archive(arc)
        self.assertEqual(summary["streams"], 3)
        self.assertEqual([r["names"] for r in rows], [["alpha.#dr"], ["bravo.#ft"], ["charlie.#dd"]])
        self.assertEqual(rows[1]["size"], len(text))
        self.assertEqual(rows[1]["stored"], len(second))
        self.assertEqual(rows[2]["offset"], 0x40 + len(first) + len(second))

    def test_scan_archive_closes_mmap_on_validation_errors(self):
        cases = (
            ("bad_magic", b"NOPE" + bytes(60), None, "not an RPF9 archive"),
            ("bad_data_start", b"RPF9" + struct.pack("<I", 0x1000) + bytes(56), None,
             "RPF9 data start is outside the archive"),
            ("bad_start", None, 99999, "--start is outside the archive"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload, start, message in cases:
                arc = os.path.join(tmp, f"{name}.rpf")
                if payload is None:
                    make_archive(arc)
                else:
                    with open(arc, "wb") as f:
                        f.write(payload)
                with self.subTest(case=name):
                    self._assert_validation_error_closes_mmap(arc, start, message)

    def test_refuses_output_that_is_the_input_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            with open(arc, "rb") as f:
                original = f.read()
            p = self.run_tool(arc, arc, "--quiet")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("input archive", p.stderr)
            with open(arc, "rb") as f:
                self.assertEqual(f.read(), original)

    def test_refusal_names_the_output_path_as_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            p = self.run_tool(arc, arc, "--quiet")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(f"output {arc} is the input archive", p.stderr)
            self.assertNotIn(repr(arc), p.stderr)   # a repr would quote the path and double every backslash

    def test_write_error_names_the_report_path_as_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            out_dir = os.path.join(tmp, "outdir")
            os.mkdir(out_dir)                      # a directory the report cannot replace
            p = self.run_tool(arc, out_dir, "--quiet")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(f"cannot write report {out_dir}:", p.stderr)
            self.assertNotIn(repr(out_dir), p.stderr)

    def test_publish_creates_nested_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            make_archive(arc)
            out = os.path.join(tmp, "nested", "missing", "index.json")
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["unique_names"], 3, doc["summary"])

    def test_publish_leaves_existing_tmp_sibling_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "index.json")
            make_archive(arc)
            sibling = out + ".tmp"
            with open(sibling, "wb") as f:
                f.write(b"pre-existing sibling")
            p = self.run_tool(arc, out, "--quiet")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["unique_names"], 3, doc["summary"])
            with open(sibling, "rb") as f:
                self.assertEqual(f.read(), b"pre-existing sibling")


if __name__ == "__main__":
    unittest.main()
