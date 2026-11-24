"""Synthetic tests for tools/page_index.py. No game files are read or needed."""
import contextlib
import csv
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "page_index.py")
sys.path.insert(0, os.path.dirname(TOOL))
import page_index

BASE = 0x50000000


def header(size, q28=0, q30=0, q38=0, q40=0, counts=(0, 0, 0, 0), page_count=2):
    data = bytearray(size)
    data[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, page_count)
    for at, value in ((0x28, q28), (0x30, q30), (0x38, q38), (0x40, q40)):
        struct.pack_into("<Q", data, at, BASE + value if value else 0)
    for i, value in enumerate(counts):
        struct.pack_into("<H", data, 0x48 + 2 * i, value)
    return data


def put_q38(data, table, entries):
    for i, (hash_value, size, target) in enumerate(entries):
        struct.pack_into("<IIQ", data, table + 16 * i, hash_value, size, BASE + target)


def make_rpf(path, streams):
    data = bytearray(0x40)
    data[0:4] = b"RPF9"
    struct.pack_into("<I", data, 4, 0x40)
    offsets = []
    for stream in streams:
        offsets.append(len(data))
        data += zlib.compress(stream)
    with open(path, "wb") as f:
        f.write(data)
    return offsets


def run_tool(source, out, extra=()):
    proc = subprocess.run([sys.executable, "-I", TOOL, source, out, "--quiet", *extra],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)


class PageIndexTest(unittest.TestCase):
    def test_cli_reports_output_write_errors(self):
        with tempfile.TemporaryDirectory() as source:
            for extension, writer in (("json", "write_json"), ("csv", "write_csv")):
                with self.subTest(extension=extension):
                    out = os.path.join(source, f"pages.{extension}")
                    stderr = io.StringIO()
                    with (mock.patch.object(sys, "argv", [TOOL, source, out, "--quiet"]),
                          mock.patch.object(page_index, writer, side_effect=OSError("disk full")),
                          contextlib.redirect_stderr(stderr)):
                        self.assertEqual(page_index.main(), 2)
                    self.assertIn(f"{out}: disk full", stderr.getvalue())

    def test_simple_page_and_q28_raw(self):
        page = header(0x200, q28=0x180, q38=0x100, counts=(1, 0, 1, 0))
        page[0x60:0xA0] = b"\xaa" * 0x40
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        struct.pack_into("<IIQQII", page, 0x180, 0x1234, 0x999, 0x300, BASE + 0x60, 0x10, 0x30000)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "tiny.rpf")
            out = os.path.join(tmp, "pages.json")
            offsets = make_rpf(arc, [page])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            self.assertEqual(doc["summary"]["complete_pages"], 1)
            self.assertEqual(doc["summary"]["q38_records"], 1)
            self.assertEqual(doc["summary"]["payload_at_0x60"], 1)
            page_doc = doc["pages"][0]
            self.assertEqual(page_doc["offset"], offsets[0])
            self.assertEqual(page_doc["stored"], len(zlib.compress(page)))
            self.assertEqual(page_doc["inflated"], 0x200)
            self.assertEqual(page_doc["version"], 122)
            self.assertEqual(page_doc["page_count"], 2)
            self.assertEqual(page_doc["tables"], {"q28": 0x180, "q30": None, "q38": 0x100, "q40": None})
            rec = page_doc["q38_records"][0]
            self.assertEqual((rec["hash"], rec["size"], rec["target"], rec["target_stream"]),
                             (0x1234, 0x40, 0x60, 0))
            q28 = page_doc["q28_records"][0]
            self.assertEqual((q28["hash_a"], q28["hash_b"], q28["field_08"], q28["target"],
                              q28["count_or_size"], q28["flags"]),
                             (0x1234, 0x999, 0x300, 0x60, 0x10, 0x30000))
            csv_path = os.path.join(tmp, "pages.csv")
            run_tool(arc, csv_path)
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["hash"], "0x00001234")
            self.assertEqual(rows[0]["target"], "0x60")
            self.assertEqual(rows[0]["target_stream"], "0")

    def test_q28_null_pointer_record_is_kept_and_the_table_goes_on(self):
        # The shape 3,511 anim.rpf pages have: a record with a null pointer and its other words set, then more.
        page = header(0x200, q28=0x140, q38=0x100, counts=(3, 0, 1, 0))
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        struct.pack_into("<IIQQII", page, 0x140, 0x1234, 0x11, 0x100, BASE + 0x60, 0x10, 0x10000)
        struct.pack_into("<IIQQII", page, 0x160, 0x2222, 0x22, 0x200, 0, 0x08, 0x20000)
        struct.pack_into("<IIQQII", page, 0x180, 0x3333, 0x33, 0x300, BASE + 0x80, 0x04, 0x30000)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "null.rpf")
            out = os.path.join(tmp, "pages.json")
            make_rpf(arc, [page])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
        page_doc = doc["pages"][0]
        self.assertEqual(page_doc["q28_error"], "")
        self.assertEqual([(r["hash_a"], r["target"], r["count_or_size"]) for r in page_doc["q28_records"]],
                         [(0x1234, 0x60, 0x10), (0x2222, None, 0x08), (0x3333, 0x80, 0x04)])
        self.assertEqual(doc["summary"]["q28_records"], 3)
        self.assertEqual(doc["summary"]["q28_declared"], 3)
        self.assertEqual(doc["summary"]["q28_null_targets"], 1)
        self.assertEqual(doc["summary"]["q28_tables_unread"], 0)

    def test_q28_pointer_with_another_base_still_stops_the_table(self):
        page = header(0x200, q28=0x140, q38=0x100, counts=(2, 0, 1, 0))
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        struct.pack_into("<IIQQII", page, 0x140, 0x1234, 0x11, 0x100, 0x40000060, 0x10, 0x10000)
        struct.pack_into("<IIQQII", page, 0x160, 0x2222, 0x22, 0x200, BASE + 0x60, 0x08, 0x20000)
        parsed, complete = page_index.parse_page(bytes(page), [])
        self.assertTrue(complete)
        self.assertEqual(parsed["q28_records"], [])
        self.assertEqual(parsed["q28_error"], "q28 entry 0 has a non-0x50000000 pointer")

    def test_q28_declared_without_a_table_pointer_is_named(self):
        # n28 = 2 but +0x28 holds no 0x50000000 pointer: the records are declared and cannot be read.
        page = header(0x200, q28=0, q38=0x100, counts=(2, 0, 1, 0))
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        parsed, complete = page_index.parse_page(bytes(page), [])
        self.assertTrue(complete)
        self.assertIsNone(parsed["tables"]["q28"])
        self.assertEqual(parsed["q28_records"], [])
        self.assertEqual(parsed["q28_error"], "q28 is not a 0x50000000 pointer")
        # No records declared: nothing is missing, so nothing is reported.
        page = header(0x200, q28=0, q38=0x100, counts=(0, 0, 1, 0))
        put_q38(page, 0x100, [(0x1234, 0x40, 0x60)])
        self.assertEqual(page_index.parse_page(bytes(page), [])[0]["q28_error"], "")

    def test_q28_table_past_the_streams_read_is_named(self):
        # q38 and its record fit in the directory stream, so no continuation is read; the q28 table is after it.
        page = header(0x100, q28=0x100, q38=0xC0, counts=(2, 0, 1, 0))
        put_q38(page, 0xC0, [(0x5678, 0x20, 0x60)])
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "late.rpf")
            out = os.path.join(tmp, "pages.json")
            make_rpf(arc, [page, b"\xcc" * 0x100])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
        page_doc = doc["pages"][0]
        self.assertTrue(page_doc["complete"])
        self.assertEqual(page_doc["continuations"], [])
        self.assertEqual(page_doc["q28_records"], [])
        self.assertEqual(page_doc["q28_error"], page_index.Q28_UNREAD)
        self.assertEqual(doc["summary"]["q28_declared"], 2)
        self.assertEqual(doc["summary"]["q28_records"], 0)
        self.assertEqual(doc["summary"]["q28_tables_unread"], 1)

    def test_continuation_stream_is_appended(self):
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(page, 0xC0, [(0x5678, 0x80, 0x100)])
        cont = b"\xbb" * 0x100
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "cont.rpf")
            out = os.path.join(tmp, "pages.json")
            offsets = make_rpf(arc, [page, cont])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            self.assertEqual(doc["summary"]["pages_with_continuations"], 1)
            page_doc = doc["pages"][0]
            self.assertTrue(page_doc["complete"])
            self.assertEqual(len(page_doc["continuations"]), 1)
            self.assertEqual(page_doc["continuations"][0]["offset"], offsets[1])
            self.assertEqual(page_doc["continuations"][0]["inflated"], 0x100)
            rec = page_doc["q38_records"][0]
            self.assertEqual((rec["target"], rec["size"], rec["target_stream"]), (0x100, 0x80, 1))
            self.assertEqual(page_doc["span"], 0x180)

    def test_resource_grouping_reports_streams_and_declared_pages(self):
        # The resource group is the directory plus the streams needed to reach the q38 table and its
        # last record. The +0x20 count is reported next to the 64 KiB units, never used as a boundary.
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0), page_count=1)
        put_q38(page, 0xC0, [(0x1234, 0x40, 0x60)])
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "single.rpf")
            out = os.path.join(tmp, "single.json")
            offsets = make_rpf(arc, [page])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            resource = doc["pages"][0]["resource"]
            self.assertEqual([s["offset"] for s in resource["streams"]], offsets)
            self.assertEqual(resource["bytes"], 0x100)
            self.assertEqual(resource["page_units"], 1)
            self.assertEqual(resource["declared_pages"], 1)
            self.assertTrue(resource["declared_matches_units"])
            self.assertEqual(doc["summary"]["declared_matches_units"], 1)
            self.assertEqual(doc["summary"]["declared_differs_from_units"], 0)

        # A continuation is part of the same resource; a declared count of 2 over 0x200 bytes is two
        # 64 KiB units short, so the mismatch is reported instead of being trusted as a boundary.
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0), page_count=2)
        put_q38(page, 0xC0, [(0x5678, 0x80, 0x100)])
        cont = b"\xbb" * 0x100
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "pair.rpf")
            out = os.path.join(tmp, "pair.json")
            offsets = make_rpf(arc, [page, cont])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            resource = doc["pages"][0]["resource"]
            self.assertEqual([s["offset"] for s in resource["streams"]], offsets)
            self.assertEqual([s["inflated"] for s in resource["streams"]], [0x100, 0x100])
            self.assertEqual(resource["bytes"], 0x200)
            self.assertEqual(resource["page_units"], 1)
            self.assertEqual(resource["declared_pages"], 2)
            self.assertFalse(resource["declared_matches_units"])
            self.assertEqual(doc["summary"]["declared_differs_from_units"], 1)
            self.assertEqual(doc["summary"]["resource_streams"], {"2": 1})

    def test_q38_table_straddles_stream(self):
        page = header(0x100, q38=0xF8, counts=(0, 0, 2, 0))
        entries = [(0x1111, 0x10, 0x60), (0x2222, 0x20, 0x120)]
        table = b"".join(struct.pack("<IIQ", h, s, BASE + t) for h, s, t in entries)
        page[0xF8:0x100] = table[:8]
        page[0x60:0x70] = b"\xaa" * 0x10
        cont = bytearray(0x140)
        cont[:8] = table[8:16]
        cont[8:24] = table[16:32]
        cont[0x20:0x40] = b"\xcc" * 0x20
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "straddle.rpf")
            out = os.path.join(tmp, "pages.json")
            make_rpf(arc, [page, bytes(cont)])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            page_doc = doc["pages"][0]
            self.assertTrue(page_doc["complete"])
            self.assertEqual(page_doc["tables"]["q38"], 0xF8)
            self.assertEqual([(r["hash"], r["size"], r["target"], r["target_stream"])
                              for r in page_doc["q38_records"]],
                             [(0x1111, 0x10, 0x60, 0), (0x2222, 0x20, 0x120, 1)])
            self.assertEqual(page_doc["span"], 0x140)

    def test_new_page_after_a_continuation_is_reported_with_both_streams(self):
        # The first page's q38 target stays out of reach even after both following streams, so the
        # second page is consumed as a continuation before it is recognised as a new directory page.
        first = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(first, 0xC0, [(0x1234, 0x10, 0x10000)])
        cont = b"\xbb" * 0x100
        second = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(second, 0xC0, [(0x5678, 0x10, 0x60)])
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "nextpage.rpf")
            out = os.path.join(tmp, "pages.json")
            offsets = make_rpf(arc, [first, cont, second])
            run_tool(arc, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
        self.assertEqual(doc["summary"]["pages"], 2)
        reported, fresh = doc["pages"]
        self.assertEqual([c["offset"] for c in reported["continuations"]], [offsets[1], offsets[2]])
        self.assertFalse(reported["complete"])
        self.assertEqual(reported["error"], "resource ended before q38 was complete")
        self.assertEqual(reported["q38_records"], [])
        self.assertEqual(reported["offset"], offsets[0])
        self.assertTrue(fresh["complete"])
        self.assertEqual(fresh["offset"], offsets[2])
        self.assertEqual([(r["hash"], r["target"]) for r in fresh["q38_records"]], [(0x5678, 0x60)])

    def test_carve_directory_mode(self):
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(page, 0xC0, [(0x9999, 0x20, 0x100)])
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "000000000040.bin"), "wb") as f:
                f.write(page)
            with open(os.path.join(tmp, "000000000200.bin"), "wb") as f:
                f.write(b"\xdd" * 0x40)
            out = os.path.join(tmp, "pages.json")
            run_tool(tmp, out)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["mode"], "directory")
            self.assertEqual(doc["summary"]["pages"], 1)
            page_doc = doc["pages"][0]
            self.assertEqual(page_doc["offset"], 0x40)
            self.assertIsNone(page_doc["stored"])
            self.assertEqual(page_doc["continuations"][0]["offset"], 0x200)
            self.assertEqual(page_doc["q38_records"][0]["target_stream"], 1)

    def test_bad_arguments_fail_before_the_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "a.rpf")
            make_rpf(rpf, [header(0x100)])
            cases = ((os.path.join(tmp, "out.txt"), (), "must end in .json or .csv"),
                     (os.path.join(tmp, "no_such_folder", "out.json"), (), "is not a folder"),
                     (tmp, (), "is a folder"),
                     (os.path.join(tmp, "out.json"), ("--limit", "0"), "must be at least 1"),
                     (os.path.join(tmp, "out.json"), ("--limit", "-5"), "must be at least 1"),
                     (os.path.join(tmp, "out.json"), ("--start", "-1"), "must be 0 or more"))
            for out, extra, message in cases:
                proc = subprocess.run([sys.executable, "-I", TOOL, rpf, out, "--quiet", *extra],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn(message, proc.stderr)
                if out != tmp:
                    self.assertFalse(os.path.exists(out))

    def test_scan_limit_excludes_crossing_stream(self):
        page = header(0x100, q38=0xC0, counts=(0, 0, 1, 0))
        put_q38(page, 0xC0, [(0x1234, 0x20, 0x60)])
        stored = len(zlib.compress(page))
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "limit.rpf")
            offsets = make_rpf(arc, [page])
            exact = os.path.join(tmp, "exact.json")
            run_tool(arc, exact, ("--start", str(offsets[0]), "--limit", str(stored)))
            with open(exact, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 1)
            self.assertEqual(doc["summary"]["streams"], 1)

            short = os.path.join(tmp, "short.json")
            run_tool(arc, short, ("--start", str(offsets[0]), "--limit", str(stored - 1)))
            with open(short, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["pages"], 0)
            self.assertEqual(doc["summary"]["streams"], 0)

    def test_start_outside_the_archive_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "a.rpf")
            make_rpf(rpf, [header(0x100)])
            proc = subprocess.run([sys.executable, "-I", TOOL, rpf, os.path.join(tmp, "o.json"), "--quiet",
                                   "--start", "99999"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("start offset is outside the archive", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_empty_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "empty.rpf")
            open(rpf, "wb").close()
            out = os.path.join(tmp, "pages.json")
            run_tool(rpf, out)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["summary"]["pages"], 0)


    def test_directory_rejects_start_or_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            for extra in (("--start", "0"), ("--limit", "100")):
                proc = subprocess.run([sys.executable, "-I", TOOL, tmp, out, "--quiet", *extra],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("single archive", proc.stderr)
                self.assertEqual(proc.stdout, "")
                self.assertFalse(os.path.exists(out))

    def test_output_cannot_overwrite_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "archive.json")
            make_rpf(arc, [header(0x100)])
            with open(arc, "rb") as f:
                original = f.read()

            # Same path
            proc = subprocess.run([sys.executable, "-I", TOOL, arc, arc, "--quiet"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("output must not overwrite the source", proc.stderr)
            with open(arc, "rb") as f:
                self.assertEqual(f.read(), original)

            # Hard-link alias
            if hasattr(os, "link"):
                alias = os.path.join(tmp, "alias.json")
                try:
                    os.link(arc, alias)
                except OSError:
                    alias = None
                if alias:
                    proc2 = subprocess.run([sys.executable, "-I", TOOL, arc, alias, "--quiet"],
                                           capture_output=True, text=True)
                    self.assertEqual(proc2.returncode, 2)
                    self.assertIn("output must not overwrite the source", proc2.stderr)
                    with open(arc, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_output_cannot_create_a_report_in_place_of_the_carve_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "index.json")
            proc = subprocess.run([sys.executable, "-I", TOOL, tmp, manifest, "--quiet"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("output would overwrite the carve manifest", proc.stderr)
            self.assertFalse(os.path.exists(manifest))

    def test_output_cannot_overwrite_carve_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "index.json")
            with open(manifest, "w", encoding="utf-8") as f:
                f.write('{"carve_manifest": true}')
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(header(0x100))

            proc = subprocess.run([sys.executable, "-I", TOOL, tmp, manifest, "--quiet"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("output would overwrite the carve manifest", proc.stderr)
            with open(manifest, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"carve_manifest": true}')

    def test_output_cannot_overwrite_carved_input(self):
        if not hasattr(os, "link"):
            return
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "00000001.bin")
            with open(bin_path, "wb") as f:
                f.write(header(0x100))
            linked_json = os.path.join(tmp, "linked.json")
            try:
                os.link(bin_path, linked_json)
            except OSError:
                return
            proc = subprocess.run([sys.executable, "-I", TOOL, tmp, linked_json, "--quiet"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("output would overwrite an input file", proc.stderr)

    def test_page_counts_histogram_matches_naive_reference(self):
        def page(count):
            return {
                "complete": True,
                "counts": (0, 0, 0, 0),
                "q38_records": [],
                "q28_records": [],
                "q28_error": "",
                "continuations": [],
                "span": 0,
                "payload_at_0x60": False,
                "resource": {"streams": [], "page_units": 1, "declared_matches_units": True},
                "version": 122,
                "page_count": count,
            }

        def naive(pages):
            return {str(v): sum(1 for p in pages if p["page_count"] == v)
                    for v in sorted({p["page_count"] for p in pages if p["page_count"] is not None})}

        cases = {
            "high-cardinality": [page(v) for v in range(1200)],
            "mixed": [page(None if i % 3 == 0 else i % 97) for i in range(1500)],
            "all-none": [page(None) for _ in range(50)],
            "zero-count": [page(0) for _ in range(7)],
            "no-pages": [],
        }
        for name, pages in cases.items():
            with self.subTest(name):
                actual = page_index.summarise(pages, "directory", "synthetic.rpf")["page_counts"]
                expected = naive(pages)
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), list(expected))


if __name__ == "__main__":
    unittest.main()
