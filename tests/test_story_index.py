"""Synthetic-header tests for tools/story_index.py.

    python -I tests/test_story_index.py

Every byte here is built by the test. No game file or game asset name is used.
"""
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
TOOL = os.path.join(ROOT, "tools", "story_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import story_index  # noqa: E402


def psin():
    data = bytearray(0x200)
    data[:4] = b"PSIN"
    tags = [b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS"]
    sizes = [0x40, 0x30, 0x28, 0x10, 0x18, 0x14]
    pos = 0x40
    for tag, size in zip(tags, sizes):
        data[pos:pos + 4] = tag
        data[pos + 4:pos + 8] = size.to_bytes(4, "big")
        pos += size
    return bytes(data)


def page():
    data = bytearray(0x60)
    struct.pack_into("<Q", data, 8, 0x50100420)
    data[16:20] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, 3)
    return bytes(data)


def rsc8():
    data = bytearray(0x40)
    data[:4] = b"RSC8"
    struct.pack_into("<I", data, 4, 8001)
    struct.pack_into("<I", data, 8, 0x1234)
    struct.pack_into("<I", data, 12, 1)
    return bytes(data)


def nested():
    data = bytearray(0x40)
    data[:4] = b"9FPR"
    struct.pack_into("<I", data, 4, 0x8A0)
    struct.pack_into("<I", data, 8, 0x810)
    struct.pack_into("<I", data, 12, 0x140)
    return bytes(data)


def rbf0():
    data = b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1) + struct.pack("<H", 21)
    data += b"test_schema\x00"
    return data + bytes(64 - len(data))


def chunks():
    return struct.pack("<III", 0x88560988, 8, 7) + bytes(52)


def aux(entries, payload=None, trailing=b""):
    """A synthetic auxiliary table: u32 count, {u32 tag, u32 size} pairs, payloads, then trailing bytes."""
    data = struct.pack("<I", len(entries)) + b"".join(struct.pack("<II", t, n) for t, n in entries)
    if payload is None:
        payload = b"".join(bytes([i + 1]) * n for i, (_, n) in enumerate(entries))
    return data + payload + trailing


def clip():
    return aux([(0x101, 8), (0x113, 12), (0x207, 4)], trailing=b"\xee" * 36)


def xml():
    data = b"\xef\xbb\xbf<?xml version=\"1.0\"?><X/>"
    return data + bytes(max(0, 64 - len(data)))


def raw():
    return b"\x11\x22\x33\x44" + bytes(60)


def zero():
    return bytes(64)


def text():
    return b"A" * 80


def bink():
    data = bytearray(0x40)
    data[:4] = b"KB2n"
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 8, 1)
    struct.pack_into("<I", data, 0x14, 64)
    struct.pack_into("<I", data, 0x18, 64)
    return bytes(data)


PAYLOADS = [psin(), page(), rsc8(), nested(), rbf0(), chunks(), clip(), aux([(0x101, 40), (0x207, 20)]),
            xml(), raw(), zero(), text(), bink()]
EXPECTED = ["psin", "page-directory", "rsc8", "rpf9-nested", "rbf0", "typehash", "clip-stream",
            "clip-stream", "xml", "raw:11223344", "zero-prefix", "text", "bink2"]


class ClassifyTest(unittest.TestCase):
    def test_signatures(self):
        got = [story_index.classify(p, 0, len(p))["signature"] for p in PAYLOADS]
        self.assertEqual(got, EXPECTED)

    def test_clip_header(self):
        r = story_index.classify(clip(), 0x600, 70)
        self.assertEqual((r["signature"], r["version"], r["count"], r["detail"]),
                         ("clip-stream", None, 3, "aux-table tags=0x101,0x113,0x207 payload_end=52"))

    def test_clip_record_matches_anim_index(self):
        import anim_index
        for data in (clip(), aux([(0x101, 40), (0x207, 20)])):
            mine, theirs = story_index.classify(data, 0, len(data)), anim_index.classify(data, 0, len(data))
            for key in ("signature", "version", "count", "detail"):
                self.assertEqual(mine[key], theirs[key], key)

    def test_aux_exact_fit_and_trailing_bytes(self):
        exact = aux([(0x101, 40), (0x207, 20)])
        r = story_index.classify(exact, 0, len(exact))
        self.assertEqual((r["signature"], r["count"], r["detail"]),
                         ("clip-stream", 2, f"aux-table tags=0x101,0x207 payload_end={len(exact)}"))
        trailing = aux([(0x101, 16), (0x204, 8), (0x207, 8)], trailing=bytes(range(64)))
        r = story_index.classify(trailing, 0, len(trailing))
        self.assertEqual((r["signature"], r["detail"]),
                         ("clip-stream", "aux-table tags=0x101,0x204,0x207 payload_end=60"))

    def test_aux_payload_overrun_is_generic(self):
        short = aux([(0x101, 40), (0x207, 30)])[:-1]   # last payload one byte short
        self.assertEqual(story_index.classify(short, 0, len(short))["signature"], "raw:02000000")
        big = aux([(0x101, 8), (0x113, 0xFFFFFFF0), (0x207, 8)], payload=bytes(64))
        self.assertNotEqual(story_index.classify(big, 0, len(big))["signature"], "clip-stream")

    def test_aux_old_prefix_alone_is_generic(self):
        # The old eight-byte prefix check called each of these a clip stream with a version and a 24-byte header.
        prefix = struct.pack("<I", 3) + b"\x01\x01\x00\x00"
        for data in (prefix, prefix + bytes(12), prefix + b"\xff" * 56):   # bare, directory cut, sizes overrun
            r = story_index.classify(data, 0, len(data))
            self.assertEqual((r["signature"], r["version"], r["count"]), ("raw:03000000", None, None))
            self.assertNotIn("24-byte", r["detail"])

    def test_aux_wrong_count_or_first_tag_is_generic(self):
        for entries in ([(0x101, 4)], [(0x101, 4)] * 4, [(0x113, 4), (0x207, 4)], [(0x207, 4), (0x101, 4)]):
            data = aux(entries, trailing=b"\x5a" * 64)
            self.assertNotEqual(story_index.classify(data, 0, len(data))["signature"], "clip-stream", entries)

    def test_aux_table_in_an_archive_scan(self):
        data = clip()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.rpf")
            with open(path, "wb") as fh:
                fh.write(struct.pack("<II", 0, 16) + bytes(8) + zlib.compress(data))
            rows, _ = story_index.scan_rpf(path)
        self.assertEqual([(r["signature"], r["count"]) for r in rows], [("clip-stream", 3)])

    def test_psin_section_chain(self):
        chain = story_index.psin_sections(psin())
        self.assertEqual([t for t, _, _ in chain], ["LSCH", "LENM", "LSNS", "LGID", "LRSC", "CHKS"])
        self.assertEqual([o for _, o, _ in chain], [0x40, 0x80, 0xB0, 0xD8, 0xE8, 0x100])
        self.assertEqual([s for _, _, s in chain], [0x40, 0x30, 0x28, 0x10, 0x18, 0x14])
        r = story_index.classify(psin(), 0x100, 90)
        self.assertEqual((r["signature"], r["count"]), ("psin", 6))
        self.assertEqual(r["detail"], "sections=LSCH,LENM,LSNS,LGID,LRSC,CHKS")

    def test_psin_header_offset(self):
        data = bytearray(psin())
        data[4:8] = (0x80).to_bytes(4, "big")
        chain = story_index.psin_sections(bytes(data))
        self.assertEqual([t for t, _, _ in chain], ["LENM", "LSNS", "LGID", "LRSC", "CHKS"])
        self.assertEqual(chain[0][1], 0x80)

    def test_page_header(self):
        r = story_index.classify(page(), 0x200, 40)
        self.assertEqual((r["version"], r["count"], r["detail"]), (122, 3, "0DRP at +16"))

    def test_rsc8_header(self):
        r = story_index.classify(rsc8(), 0x300, 50)
        self.assertEqual((r["signature"], r["version"]), ("rsc8", 8001))
        self.assertEqual(r["detail"], "declared_size=4660 flags=1")

    def test_nested_rpf9_header(self):
        r = story_index.classify(nested(), 0x400, 60)
        self.assertEqual(r["signature"], "rpf9-nested")
        self.assertEqual(r["detail"], "data_off=2208 toc=2064")

    def test_xml(self):
        self.assertEqual(story_index.classify(xml(), 0x500, 70)["signature"], "xml")


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rpf = os.path.join(self.root, "synthetic.rpf")
        header = bytearray(0x100)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = bytearray()
        self.offsets = []
        for payload in PAYLOADS:
            self.offsets.append(len(header) + len(body))
            body += payload if payload[:3] == b"KB2" else zlib.compress(payload)
            body += bytes((-len(body)) % 16)
        body += bytes(64)
        with open(self.rpf, "wb") as f:
            f.write(header + body)
        self.span = len(body)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_offsets_and_summary(self):
        rows, summary = story_index.scan_rpf(self.rpf)
        self.assertEqual([r["offset"] for r in rows], self.offsets)
        self.assertEqual([r["signature"] for r in rows], EXPECTED)
        self.assertEqual(summary["files"], len(PAYLOADS))
        self.assertEqual(summary["bink2_files"], 1)
        self.assertEqual(summary["unread_bytes"], 0)
        self.assertEqual(summary["span"], self.span)
        self.assertEqual(summary["inflated_bytes"], sum(len(p) for p in PAYLOADS))
        # A stored length must stop at the zlib end, not the end of the read block.
        zlibs = [r for r in rows if r["signature"] != "bink2"]
        self.assertEqual(sum(r["stored"] for r in zlibs), sum(len(zlib.compress(p)) for p in PAYLOADS[:-1]))
        self.assertEqual(rows[-1]["stored"], 0x40)
        self.assertGreater(summary["readable_fraction"], 0.5)

    def test_scan_limit(self):
        limit = self.offsets[1] - 0x100 - 1
        rows, summary = story_index.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signature"], "psin")
        self.assertEqual(summary["span"], limit)

    def test_scan_limit_excludes_crossing_streams(self):
        for start, stored, limit in ((self.offsets[0], len(zlib.compress(psin())), 2),
                                     (self.offsets[-1], len(bink()), len(bink()) - 1)):
            with self.subTest(start=start):
                rows, summary = story_index.scan_rpf(self.rpf, start=start, limit=stored)
                self.assertEqual([r["stored"] for r in rows], [stored])
                self.assertEqual(summary["readable_fraction"], 1.0)

                rows, summary = story_index.scan_rpf(self.rpf, start=start, limit=limit)
                self.assertEqual(summary["readable_fraction"], 0.0)
                self.assertEqual(summary["span"], limit)
                self.assertEqual(summary["unread_bytes"], limit)
                self.assertEqual(summary["compressed_bytes"], 0)
                self.assertEqual(rows, [])

    def test_csv_and_json(self):
        rows, summary = story_index.scan_rpf(self.rpf)
        csv_path = os.path.join(self.root, "out.csv")
        json_path = os.path.join(self.root, "out.json")
        story_index.write_output(csv_path, rows, summary)
        story_index.write_output(json_path, rows, summary)
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        self.assertEqual(table[0], story_index.COLUMNS)
        self.assertEqual(len(table), len(PAYLOADS) + 1)
        self.assertEqual(table[1][3], "psin")
        self.assertEqual(table[1][5], "6")
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], len(PAYLOADS))
        self.assertEqual(len(doc["files"]), len(PAYLOADS))

    def test_csv_partial_staged_write_keeps_prior_bytes_and_cleans_up(self):
        out = os.path.join(self.root, "out.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("prior,bytes\r\n")
        with open(out, "rb") as f:
            prior = f.read()
        rows = [{"offset": 0x100, "stored": 8, "size": 9, "signature": "psin",
                 "version": None, "count": 1, "detail": ""},
                {"offset": 0x200, "stored": 8, "size": 9, "signature": "text",
                 "version": None, "count": None, "detail": "printable text"}]
        real_csv_writer = csv.writer  # captured before story_index.csv (the csv module) is patched

        class _FailingWriter:
            """A csv writer that fails once the header and first row have hit the staged file."""

            def __init__(self, fileobj, **kwargs):
                self._w = real_csv_writer(fileobj, **kwargs)
                self._n = 0

            def writerow(self, row):
                self._w.writerow(row)
                self._n += 1
                if self._n == 2:
                    raise OSError("disk full")

        before = sorted(os.listdir(self.root))
        with self.assertRaises(OSError):
            with mock.patch.object(story_index.csv, "writer", side_effect=_FailingWriter):
                story_index.write_output(out, rows, {"files": 2})
        with open(out, "rb") as f:
            self.assertEqual(f.read(), prior)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_json_partial_staged_write_keeps_prior_bytes_and_cleans_up(self):
        out = os.path.join(self.root, "out.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write('{"prior": true}')
        with open(out, "rb") as f:
            prior = f.read()

        def failing_dump(obj, f, **kwargs):
            f.write('{"partial": ')
            raise OSError("disk full")

        before = sorted(os.listdir(self.root))
        with self.assertRaises(OSError):
            with mock.patch.object(story_index.json, "dump", side_effect=failing_dump):
                story_index.write_output(out, [], {"files": 0})
        with open(out, "rb") as f:
            self.assertEqual(f.read(), prior)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_cli(self):
        out = os.path.join(self.root, "cli.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(p.stdout, "")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["signatures"]["psin"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["page-directory"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["rsc8"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["clip-stream"]["files"], 2)
        clips = [row for row in doc["files"] if row["signature"] == "clip-stream"]
        self.assertEqual([(row["count"], row["version"]) for row in clips], [(3, None), (2, None)])

    def test_cli_reports_a_report_write_error(self):
        out = os.path.join(self.root, "index.json")
        err = io.StringIO()
        with (mock.patch.object(sys, "argv", [TOOL, self.rpf, out, "--quiet"]),
              mock.patch.object(story_index, "write_output", side_effect=OSError("disk full")),
              contextlib.redirect_stderr(err)):
            self.assertEqual(story_index.main(), 2)
        self.assertIn(f"{out}: disk full", err.getvalue())

    def test_unwritable_or_bad_output_path_fails_before_the_scan(self):
        for out in (os.path.join(self.root, "no_such_folder", "x.json"),
                    os.path.join(self.root, "out.txt"),
                    self.root):
            with self.subTest(out=out):
                p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertEqual(p.stdout, "")
        self.assertFalse(os.path.exists(os.path.join(self.root, "no_such_folder")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "out.txt")))

    def test_directory(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((psin(), page(), rsc8())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        rows, summary = story_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])
        self.assertEqual([r["signature"] for r in rows], ["psin", "page-directory", "rsc8"])
        self.assertEqual(summary["mode"], "directory")

    def test_directory_skips_the_carvers_index_and_reports(self):
        """rpf9_carve.py writes index.json next to the streams; it and a report are not payloads."""
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((psin(), page(), rsc8())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
            f.write('[{"off": 4096, "stored": 16}]')
        with open(os.path.join(folder, "report.csv"), "w", encoding="utf-8") as f:
            f.write("offset,size\n")
        rows, summary = story_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])

    def test_cli_directory_excludes_its_report_on_rerun(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        with open(os.path.join(folder, "000000000100.bin"), "wb") as f:
            f.write(b"A" * 64)
        out = os.path.join(folder, ".", "report.json")
        for run in (1, 2):
            with self.subTest(run=run):
                p = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                with open(out, encoding="utf-8") as f:
                    doc = json.load(f)
                self.assertEqual(doc["summary"]["files"], 1)
                self.assertEqual(doc["summary"]["compressed_bytes"], 64)
                self.assertEqual([(r["offset"], r["stored"]) for r in doc["files"]], [(0x100, 64)])

    def test_empty_archive(self):
        empty = os.path.join(self.root, "empty.rpf")
        open(empty, "wb").close()
        rows, summary = story_index.scan_rpf(empty)
        self.assertEqual(rows, [])
        self.assertEqual((summary["files"], summary["span"]), (0, 0))

    def test_cli_rejects_bad_ranges(self):
        out = os.path.join(self.root, "bad.json")
        for extra, code, message in ((("--limit", "0"), 2, "must be at least 1"),
                                     (("--limit", "-1"), 2, "must be at least 1"),
                                     (("--start", "-1"), 2, "must be 0 or more"),
                                     (("--start", "0x7fffffff"), 2, "start offset is outside the archive")):
            p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet", *extra],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, code, p.stderr)
            self.assertIn(message, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))


    def test_cli_rejects_start_or_limit_for_a_directory(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        out = os.path.join(self.root, "dir.json")
        for extra in (("--start", "0"), ("--limit", "100")):
            p = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet", *extra],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("single archive", p.stderr)
            self.assertEqual(p.stdout, "")
            self.assertFalse(os.path.exists(out))

    def test_output_cannot_overwrite_source_archive(self):
        with open(self.rpf, "rb") as f:
            rpf_data = f.read()
        arc = os.path.join(self.root, "archive.json")
        with open(arc, "wb") as f:
            f.write(rpf_data)

        # Same path
        p = subprocess.run([sys.executable, "-I", TOOL, arc, arc, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 2)
        self.assertIn("output must not overwrite the source", p.stderr)
        with open(arc, "rb") as f:
            self.assertEqual(f.read(), rpf_data)

        # Hard-link alias
        if hasattr(os, "link"):
            alias = os.path.join(self.root, "alias.json")
            try:
                os.link(arc, alias)
            except OSError:
                alias = None
            if alias:
                p2 = subprocess.run([sys.executable, "-I", TOOL, arc, alias, "--quiet"],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(p2.returncode, 2)
                self.assertIn("output must not overwrite the source", p2.stderr)
                with open(arc, "rb") as f:
                    self.assertEqual(f.read(), rpf_data)

    def test_output_cannot_create_a_report_in_place_of_the_carve_manifest(self):
        folder = os.path.join(self.root, "no_manifest")
        os.mkdir(folder)
        manifest = os.path.join(folder, "index.json")
        p = subprocess.run([sys.executable, "-I", TOOL, folder, manifest, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 2)
        self.assertIn("output would overwrite the carve manifest", p.stderr)
        self.assertFalse(os.path.exists(manifest))

    def test_output_cannot_overwrite_carve_manifest(self):
        folder = os.path.join(self.root, "carved_manifest")
        os.makedirs(folder)
        manifest = os.path.join(folder, "index.json")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write('{"carve_manifest": true}')
        with open(os.path.join(folder, "00000001.bin"), "wb") as f:
            f.write(psin())

        p = subprocess.run([sys.executable, "-I", TOOL, folder, manifest, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 2)
        self.assertIn("output would overwrite the carve manifest", p.stderr)
        with open(manifest, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"carve_manifest": true}')

    def test_output_cannot_overwrite_carved_input(self):
        if not hasattr(os, "link"):
            return
        folder = os.path.join(self.root, "carved_linked")
        os.makedirs(folder)
        bin_path = os.path.join(folder, "00000001.bin")
        with open(bin_path, "wb") as f:
            f.write(psin())
        linked_json = os.path.join(folder, "linked.json")
        try:
            os.link(bin_path, linked_json)
        except OSError:
            return
        p = subprocess.run([sys.executable, "-I", TOOL, folder, linked_json, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 2)
        self.assertIn("output would overwrite an input file", p.stderr)


if __name__ == "__main__":
    unittest.main()
