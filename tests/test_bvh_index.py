"""Hand-built headings for tools/bvh_index.py. No game data is read.

    python -I tests/test_bvh_index.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bvh_index  # noqa: E402


def bvh_payload(n=10, tag=bvh_index.TAG, end_delta=0, root=0x136AC):
    b = bytearray(0x140 + n * 64)
    struct.pack_into("<I", b, 0x100, tag)
    struct.pack_into("<I", b, 0x104, n)
    end = 0x140 + n * 64 + end_delta
    struct.pack_into("<I", b, 0x20, end)
    struct.pack_into("<I", b, 0x28, end - 0x100)
    struct.pack_into("<I", b, 0x108, 0x9ABCDEF0)
    struct.pack_into("<I", b, 0x10C, 7)
    struct.pack_into("<6f", b, 0x110, -1.5, -2.0, -3.0, 4.5, 5.0, 6.0)
    struct.pack_into("<I", b, 0x128, 0x1234)
    struct.pack_into("<i", b, 0x12C, -5)
    struct.pack_into("<I", b, 0x130, root)
    struct.pack_into("<I", b, 0x134, 0x80)
    struct.pack_into("<I", b, 0x138, 0xFFFFAB40)
    struct.pack_into("<I", b, 0x13C, 0x808C8A89)
    b[0x140:0x180] = b"\x81" * 64
    return bytes(b)


def instance_payload():
    b = bytearray(208)
    for off, val in bvh_index.INSTANCE_WORDS:
        struct.pack_into("<Q", b, off, val)
    struct.pack_into("<Q", b, 0x40, 0x0123456789ABCDEF)
    struct.pack_into("<12f", b, 0x50, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    struct.pack_into("<I", b, 0xA0, 0x1234)
    struct.pack_into("<I", b, 0xA4, 0x00200010)
    return bytes(b)


def xml_payload():
    return b"\xef\xbb\xbf<?xml version=\"1.0\"?><SSetupData><deviceName>x</deviceName></SSetupData>"


def make_rpf(payloads):
    """An RPF9-shaped buffer: 4-byte magic, u32 data start at 4, then zlib streams separated by zeros."""
    body = b""
    for p in payloads:
        body += zlib.compress(p, 9) + b"\0" * 16
    head = bytearray(0x40)
    head[0:4] = b"9FPR"
    struct.pack_into("<I", head, 4, len(head))
    return bytes(head) + body


class ClassifyTest(unittest.TestCase):
    def test_bvh4(self):
        row = bvh_index.classify(bvh_payload(12), 0x140 + 12 * 64)
        self.assertEqual(row["kind"], "bvh4")
        self.assertEqual(row["node_count"], 12)
        self.assertEqual(row["tail_offset"], 0x140 + 12 * 64 - 0x100)
        self.assertEqual(row["bounds_min"], [-1.5, -2.0, -3.0])
        self.assertEqual(row["bounds_max"], [4.5, 5.0, 6.0])
        self.assertEqual(row["flags"], bvh_index.TAG)
        self.assertEqual(row["version"], 6)
        self.assertEqual(row["crc32"], 0x9ABCDEF0)
        self.assertEqual(row["num_elements"], 7)
        self.assertEqual(row["offset_to_nodes"], 0x1234)
        self.assertEqual(row["offset_to_prim_remap"], -5)
        self.assertEqual(row["root_node_ptr"], 0x136AC)
        self.assertEqual((row["root_node_index"], row["root_node_type"]), (9941, 4))
        self.assertEqual(row["root_node_type_name"], "box16")
        self.assertFalse(row["root_node_invalid"])
        self.assertEqual(row["offset_to_elements"], 0x80)
        self.assertEqual(row["offset_to_parents"], 0xFFFFAB40)
        self.assertEqual(row["compression_params"], 0x808C8A89)
        self.assertTrue(row["dehydrated"])

    def test_header_fields(self):
        hdr = bvh_index.parse_header(bvh_payload(4))
        self.assertIsNotNone(hdr)
        self.assertEqual(hdr["version"], 6)
        self.assertEqual(hdr["offset_to_prim_remap"], -5)
        self.assertEqual(hdr["offset_to_parents"], 0xFFFFAB40)
        self.assertEqual(hdr["node_count"], 4)
        self.assertEqual(hdr["root_node_index"], 9941)
        self.assertEqual(hdr["root_node_type"], 4)
        self.assertEqual(hdr["root_node_type_name"], "box16")
        self.assertFalse(hdr["root_node_invalid"])
        self.assertTrue(hdr["bottom_level"])
        self.assertFalse(hdr["compressed"])

    def test_node_pointer(self):
        self.assertEqual(bvh_index.decode_node_pointer(0x136AC),
                         {"ptr": 0x136AC, "index": 9941, "type": 4, "name": "box16", "invalid": False})
        self.assertEqual(bvh_index.decode_node_pointer(bvh_index.INVALID_NODE),
                         {"ptr": 0xFFFFFFFF, "index": 536870911, "type": 7,
                          "name": "invalid", "invalid": True})

    def test_invalid_root_node_pointer(self):
        payload = bvh_payload(4, root=bvh_index.INVALID_NODE)
        hdr = bvh_index.parse_header(payload)
        self.assertTrue(hdr["root_node_invalid"])
        self.assertEqual(hdr["root_node_type_name"], "invalid")
        row = bvh_index.classify(payload, len(payload))
        self.assertEqual(row["kind"], "bvh4")
        self.assertTrue(row["root_node_invalid"])
        self.assertEqual(row["root_node_type_name"], "invalid")

    def test_bvh4_chunk(self):
        b = bytearray(65536)
        struct.pack_into("<I", b, 0x100, bvh_index.TAG)
        struct.pack_into("<I", b, 0x104, 1200)
        end = 0x140 + 1200 * 64
        struct.pack_into("<I", b, 0x20, end)
        struct.pack_into("<I", b, 0x28, end - 0x100)
        row = bvh_index.classify(bytes(b[:0x140]), 65536)
        self.assertEqual(row["kind"], "bvh4-chunk")
        self.assertTrue(row["chunked"])
        self.assertEqual(row["declared_bytes"], end)

    def test_bvh4_requires_the_size_formula(self):
        self.assertEqual(bvh_index.classify(bvh_payload(12, end_delta=64), 0x140 + 12 * 64 + 64)["kind"], "raw")
        self.assertEqual(bvh_index.classify(bvh_payload(12, tag=0x81020007), 0x140 + 12 * 64)["kind"], "raw")

    def test_instance208(self):
        row = bvh_index.classify(instance_payload(), 208)
        self.assertEqual(row["kind"], "instance208")
        self.assertEqual(row["hash"], "0123456789abcdef")
        self.assertEqual((row["hi16"], row["lo16"]), (0x20, 0x10))
        self.assertEqual(row["off_a0"], 0x1234)

    def test_descriptor(self):
        b = bytearray(160)
        struct.pack_into("<I", b, 8, 0x50000060)
        struct.pack_into("<I", b, 0x20, 0x50000090)
        self.assertEqual(bvh_index.classify(bytes(b), 160)["kind"], "descriptor")

    def test_xml(self):
        row = bvh_index.classify(xml_payload(), len(xml_payload()))
        self.assertEqual(row["kind"], "xml")
        self.assertFalse(row["fragment"])
        self.assertEqual(row["root_element"], "SSetupData")
        with_enc = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\n<CMapData><name>x</name></CMapData>'
        row_enc = bvh_index.classify(with_enc, len(with_enc))
        self.assertEqual(row_enc["kind"], "xml")
        self.assertFalse(row_enc["fragment"])
        self.assertEqual(row_enc["root_element"], "CMapData")

    def test_xml_fragment(self):
        text = b"/americas/map/x_bvh.rpf</filename><Item/>" + b" " * 40
        row = bvh_index.classify(text, len(text))
        self.assertEqual(row["kind"], "xml")
        self.assertTrue(row["fragment"])


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_streams_are_both_found(self):
        # The old carver advanced by the 1 MiB read size, so a small first stream hid the second one.
        payloads = [b"a" * 5000, bvh_payload(5)]
        path = os.path.join(self.dir, "two.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf(payloads))
        rows, summary = bvh_index.scan_rpf(path)
        self.assertEqual(summary[0]["streams"], 2)
        self.assertEqual([r["kind"] for r in rows], ["raw", "bvh4"])
        self.assertEqual(rows[1]["node_count"], 5)
        self.assertEqual(summary[0]["inflated"], 5000 + 0x140 + 5 * 64)

    def test_fill_requires_all_bvh4_node_bytes(self):
        path = os.path.join(self.dir, "one.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([bvh_payload(1)]))
        rows, _ = bvh_index.scan_rpf(path, max_inflate=384)
        self.assertEqual(rows[0]["fill"], {"81": 64, "total": 64})
        for cap in (320, 352):
            with self.subTest(cap=cap):
                rows, _ = bvh_index.scan_rpf(path, max_inflate=cap)
                self.assertEqual(rows[0]["kind"], "bvh4")
                self.assertEqual(rows[0]["node_bytes"], 64)
                self.assertNotIn("fill", rows[0])

    def test_directory_with_an_unreadable_archive_keeps_the_others_but_exits_nonzero(self):
        good = os.path.join(self.dir, "good.rpf")
        with open(good, "wb") as f:
            f.write(make_rpf([bvh_payload(3)]))
        bad = bytearray(make_rpf([bvh_payload(2)]))
        struct.pack_into("<I", bad, 4, len(bad) + 1000)       # the data area starts past the end of the file
        with open(os.path.join(self.dir, "bad.rpf"), "wb") as f:
            f.write(bad)
        out = os.path.join(self.dir, "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = bvh_index.main([self.dir, "--out", out])
        self.assertEqual(code, 1)
        self.assertIn("bad.rpf: skipped", err.getvalue())
        self.assertIn("incomplete", err.getvalue())
        with open(out) as f:
            self.assertEqual(json.load(f)["summary"]["streams"], 1)    # the readable archive is still reported

    def test_directory_with_an_unlistable_subfolder_exits_nonzero_and_names_it(self):
        with open(os.path.join(self.dir, "good.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(3)]))
        locked = os.path.join(self.dir, "locked")
        os.makedirs(locked)
        with open(os.path.join(locked, "hidden.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        real = os.scandir

        def scandir(path="."):   # chmod does not deny root, so the listing failure is made here
            if os.path.abspath(os.fspath(path)) == os.path.abspath(locked):
                raise PermissionError(13, "Permission denied", os.fspath(path))
            return real(path)

        out = os.path.join(self.dir, "index.json")
        err = io.StringIO()
        with mock.patch.object(os, "scandir", scandir), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            code = bvh_index.main([self.dir, "--out", out])
        self.assertEqual(code, 1)
        self.assertIn("locked: skipped", err.getvalue())
        self.assertIn("incomplete", err.getvalue())
        with open(out) as f:
            self.assertEqual(json.load(f)["summary"]["streams"], 1)    # the readable archive is still reported

    def test_hidden_folders_and_files_are_not_scanned(self):
        os.makedirs(os.path.join(self.dir, ".cache"))
        for rel in (os.path.join(".cache", "a.rpf"), ".b.rpf"):
            with open(os.path.join(self.dir, rel), "wb") as f:
                f.write(make_rpf([bvh_payload(2)]))
        with open(os.path.join(self.dir, "c.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(3)]))
        rows, _ = bvh_index.scan_dir(self.dir)
        self.assertEqual([os.path.basename(r["source"]) for r in rows], ["c.rpf"])

    def test_index_outputs(self):
        path = os.path.join(self.dir, "pack.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([instance_payload(), bvh_payload(3), xml_payload()]))
        out = os.path.join(self.dir, "index.json")
        csv_path = os.path.join(self.dir, "index.csv")
        self.assertEqual(bvh_index.main([path, "--out", out, "--csv", csv_path]), 0)
        with open(out) as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 3)
        self.assertEqual(doc["summary"]["kinds"], {"instance208": 1, "bvh4": 1, "xml": 1})
        self.assertNotIn("head", doc["streams"][0])
        self.assertEqual(doc["streams"][1]["version"], 6)
        self.assertIsInstance(doc["streams"][1]["version"], int)
        self.assertFalse(doc["streams"][1]["root_node_invalid"])
        with open(csv_path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("signature", text.splitlines()[0])
        self.assertIn("root_node_invalid", text.splitlines()[0])
        self.assertIn("bvh4:0x81020006", text)

    def test_cli_rejects_colliding_output_paths(self):
        out = os.path.join(self.dir, "index.json")
        alias = os.path.join(self.dir, ".", "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bvh_index.main([self.dir, "--out", out, "--csv", alias])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--out and --csv must use different paths", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_report_under_a_file_is_refused_before_the_scan(self):
        # Before: the scan ran to its end and os.makedirs then died in a NotADirectoryError traceback.
        with open(os.path.join(self.dir, "a.bin"), "wb") as f:
            f.write(bvh_payload(2))
        blocker = os.path.join(self.dir, "blocker")
        with open(blocker, "wb") as f:
            f.write(b"not a folder")
        out = os.path.join(blocker, "sub", "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(bvh_index, "scan_dir", side_effect=AssertionError("scanned anyway")):
            with self.assertRaises(SystemExit) as cm:
                bvh_index.main([self.dir, "--out", out])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--out cannot be written to", err.getvalue())
        self.assertIn("cannot create its folder", err.getvalue())

    def test_write_error_after_the_scan_is_a_message_and_exit_2(self):
        with open(os.path.join(self.dir, "a.bin"), "wb") as f:
            f.write(bvh_payload(2))
        out = os.path.join(self.dir, "index.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(bvh_index, "_publish", side_effect=OSError(28, "No space left on device")):
            self.assertEqual(bvh_index.main([self.dir, "--out", out]), 2)
        self.assertIn("cannot write the report", err.getvalue())
        self.assertIn("No space left on device", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_directory_with_glob_metacharacters_in_its_name(self):
        # Before: os.path.join(path, "**", "*.bin"/"*.rpf") fed a root like "carve[1]" straight to glob, which
        # read "[1]" as a character class and matched nothing.
        folder = os.path.join(self.dir, "carve[1]")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.bin"), "wb") as f:
            f.write(bvh_payload(4))
        with open(os.path.join(folder, "b.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        rows, summary = bvh_index.scan_dir(folder)
        self.assertEqual([r["kind"] for r in rows], ["bvh4"])
        self.assertEqual(summary[0]["streams"], 1)

    def test_directory_of_payloads(self):
        for name, payload in (("a.bin", bvh_payload(4)), ("b.bin", instance_payload()), ("c.bin", b"z" * 80)):
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(payload)
        rows, summary = bvh_index.scan_dir(self.dir)
        self.assertEqual({r["kind"] for r in rows}, {"bvh4", "instance208", "raw"})
        self.assertEqual(summary[0]["streams"], 3)

    def test_carved_bvh_fill_requires_all_declared_node_bytes(self) -> None:
        n = 1200
        complete = bvh_payload(n)
        chunk = complete[:65536]
        for name, payload in (("complete.bin", complete), ("chunk.bin", chunk)):
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(payload)

        rows, _ = bvh_index.scan_dir(self.dir)
        by_kind = {row["kind"]: row for row in rows}
        self.assertEqual(by_kind["bvh4"]["fill"]["total"], n * 64)
        self.assertEqual(by_kind["bvh4-chunk"]["node_count"], n)
        self.assertNotIn("fill", by_kind["bvh4-chunk"])

        archive = os.path.join(self.dir, "chunks.rpf")
        with open(archive, "wb") as f:
            f.write(make_rpf([chunk]))
        (archived,), _ = bvh_index.scan_rpf(archive)
        self.assertEqual(archived["kind"], "bvh4-chunk")
        self.assertNotIn("fill", archived)

    def test_folder_name_with_glob_characters_is_a_path(self):
        # glob read "[v1]" as a character class, so a folder named like this indexed nothing and exited 0.
        folder = os.path.join(self.dir, "carve [v1]")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.bin"), "wb") as f:
            f.write(bvh_payload(4))
        rows, summary = bvh_index.scan_dir(folder)
        self.assertEqual([r["kind"] for r in rows], ["bvh4"])
        archives = os.path.join(self.dir, "pack [1]")
        os.makedirs(archives)
        with open(os.path.join(archives, "a.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        rows, summary = bvh_index.scan_dir(archives)
        self.assertEqual([r["kind"] for r in rows], ["bvh4"])

    def test_empty_and_short_archives_have_no_streams(self):
        for name, data in (("empty.rpf", b""), ("short.rpf", b"9FPR")):
            path = os.path.join(self.dir, name)
            with open(path, "wb") as f:
                f.write(data)
            rows, summary = bvh_index.scan_rpf(path)
            self.assertEqual(rows, [])
            self.assertEqual((summary[0]["streams"], summary[0]["span"]), (0, 0))

    def test_start_outside_the_archive_is_refused(self):
        path = os.path.join(self.dir, "one.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        with self.assertRaises(ValueError):
            bvh_index.scan_rpf(path, start=1 << 20)

    def test_directory_keeps_good_archives_next_to_a_bad_one(self):
        with open(os.path.join(self.dir, "a_good.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        with open(os.path.join(self.dir, "b_bad.rpf"), "wb") as f:
            f.write(b"9FPR" + struct.pack("<I", 1 << 20))   # data start past the end of the file
        rows, summaries = bvh_index.scan_dir(self.dir)
        self.assertEqual([r["kind"] for r in rows], ["bvh4"])
        self.assertEqual(len(summaries), 1)
        self.assertIsInstance(summaries[0], dict)
        self.assertEqual(summaries[0]["streams"], 1)

    def test_directory_of_rpfs_writes_flat_summaries(self):
        one = os.path.join(self.dir, "one")
        two = os.path.join(self.dir, "two")
        os.makedirs(one)
        os.makedirs(two)
        with open(os.path.join(one, "a.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        for name, n in (("a.rpf", 2), ("b.rpf", 3)):
            with open(os.path.join(two, name), "wb") as f:
                f.write(make_rpf([bvh_payload(n)]))

        out1 = os.path.join(self.dir, "one.json")
        out2 = os.path.join(self.dir, "two.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bvh_index.main([one, "--out", out1]), 0)
            self.assertEqual(bvh_index.main([two, "--out", out2]), 0)
        with open(out1, encoding="utf-8") as f:
            doc1 = json.load(f)
        self.assertIsInstance(doc1["summary"], dict)
        self.assertEqual(doc1["summary"]["streams"], 1)
        self.assertEqual(doc1["summary"]["bvh4_nodes"], 2)
        with open(out2, encoding="utf-8") as f:
            doc2 = json.load(f)
        self.assertIsInstance(doc2["summary"], list)
        self.assertEqual(len(doc2["summary"]), 2)
        self.assertTrue(all(isinstance(item, dict) for item in doc2["summary"]))
        self.assertEqual([item["bvh4_nodes"] for item in doc2["summary"]], [2, 3])

    def test_cli_rejects_a_limit_below_one(self):
        path = os.path.join(self.dir, "one.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        for value in ("0", "-100"):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                bvh_index.main([path, "--limit", value])
            self.assertEqual(cm.exception.code, 2)


    def test_directory_scan_honours_max_inflate(self):
        with open(os.path.join(self.dir, "one.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(4)]))
        full = 0x140 + 4 * 64
        rows, _ = bvh_index.scan_dir(self.dir)
        self.assertEqual(rows[0]["kind"], "bvh4")
        out = os.path.join(self.dir, "capped.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bvh_index.main([self.dir, "--max-inflate", "128", "--out", out]), 0)
        with open(out) as f:
            doc = json.load(f)
        # Only 128 bytes were kept, so the 0x140-byte header cannot be read; the real size is still reported.
        self.assertEqual(doc["streams"][0]["kind"], "raw")
        self.assertEqual(doc["streams"][0]["size"], full)

    def test_cli_rejects_start_or_limit_for_a_directory(self):
        with open(os.path.join(self.dir, "one.rpf"), "wb") as f:
            f.write(make_rpf([bvh_payload(2)]))
        out = os.path.join(self.dir, "index.json")
        for extra in (("--start", "0"), ("--limit", "100")):
            err, so = io.StringIO(), io.StringIO()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
                bvh_index.main([self.dir, "--out", out, *extra])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("single archive", err.getvalue())
            self.assertEqual(so.getvalue(), "")
            self.assertFalse(os.path.exists(out))


    def test_report_cannot_overwrite_carve_manifest(self):
        manifest = os.path.join(self.dir, "index.json")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write('{"carve_manifest": true}')
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bvh_index.main([self.dir, "--out", manifest])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("would overwrite the carve manifest", err.getvalue())
        with open(manifest, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"carve_manifest": true}')

    def test_report_cannot_overwrite_a_carved_input(self):
        payload = os.path.join(self.dir, "a.bin")
        with open(payload, "wb") as f:
            f.write(bvh_payload(4))
        alias = os.path.join(self.dir, "alias.json")
        try:
            os.link(payload, alias)
        except OSError:
            self.skipTest("hard links are not available")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bvh_index.main([self.dir, "--out", alias])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("would overwrite an input file", err.getvalue())

    def test_missing_report_folder_is_created(self):
        out = os.path.join(self.dir, "missing", "sub", "index.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bvh_index.main([self.dir, "--out", out]), 0)
        self.assertTrue(os.path.isfile(out))

    def test_report_folder_is_rejected(self):
        outdir = os.path.join(self.dir, "outdir")
        os.makedirs(outdir, exist_ok=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                bvh_index.main([self.dir, "--out", outdir])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("is a folder, not a report file", err.getvalue())


class MissingTargetTest(unittest.TestCase):
    """A target that cannot be opened is one line naming the path once; the OSError text repeated it and repr'd it."""

    def _stderr(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bvh_index.main(argv), 2)
        return err.getvalue()

    def test_cli_names_a_missing_target_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-such.rpf")
            text = self._stderr([missing])
        self.assertIn(f"{missing}: No such file or directory", text)
        self.assertNotIn("[Errno", text)
        self.assertEqual(text.count(missing), 1)

    def test_cli_keeps_the_reason_of_a_refused_archive(self):
        """A ValueError from the scan still reads as its own sentence, not as an empty line."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "one.rpf")
            with open(path, "wb") as f:
                f.write(make_rpf([bvh_payload(2)]))
            text = self._stderr([path, "--start", str(1 << 20)])
        self.assertIn("start offset is outside the archive", text)


class _Unprintable:
    """A cell whose str() fails part-way through the CSV, standing in for a write that dies mid-report."""

    def __str__(self):
        raise OSError("simulated write failure")


class PublishTest(unittest.TestCase):
    """write_outputs stages each report next to its destination and swaps it in: a run that fails while writing
    keeps the previous report instead of leaving it truncated (before: open(dest, "w") emptied it first)."""

    OLD_JSON = '{"summary": {"previous": true}, "streams": []}'
    OLD_CSV = "source,offset\nprevious.rpf,0\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.out = os.path.join(self.dir, "index.json")
        self.csv = os.path.join(self.dir, "index.csv")
        with open(self.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.OLD_JSON)
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            f.write(self.OLD_CSV)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, path):
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()

    def assert_previous_reports_kept(self):
        self.assertEqual(self.read(self.out), self.OLD_JSON)
        self.assertEqual(self.read(self.csv), self.OLD_CSV)
        self.assertEqual(sorted(os.listdir(self.dir)), ["index.csv", "index.json"])

    def test_json_serialization_failure_keeps_previous_reports(self):
        rows = [{"source": "a.rpf", "offset": 0, "bad": object()}]
        with self.assertRaises(TypeError):
            bvh_index.write_outputs(rows, [{"streams": 1}], self.out, self.csv)
        self.assert_previous_reports_kept()

    def test_csv_write_failure_keeps_previous_csv(self):
        rows = [{"source": _Unprintable(), "offset": 0}]
        with self.assertRaises(OSError):
            bvh_index.write_outputs(rows, [{"streams": 1}], None, self.csv)
        self.assert_previous_reports_kept()

    def test_interrupted_write_keeps_previous_report(self):
        with mock.patch.object(bvh_index.json, "dump", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                bvh_index.write_outputs([], [{"streams": 0}], self.out, self.csv)
        self.assert_previous_reports_kept()

    def test_reports_replaced_with_published_mode(self):
        rows = [{"source": "a.rpf", "offset": 16, "bounds_min": [1, 2, 3]}]
        with mock.patch.object(bvh_index, "published_mode", wraps=bvh_index.published_mode) as published, \
                mock.patch.object(bvh_index.os, "chmod", wraps=os.chmod) as chmod:
            bvh_index.write_outputs(rows, [{"streams": 1}], self.out, self.csv)
        self.assertEqual([c.args for c in published.call_args_list], [(self.out,), (self.csv,)])
        self.assertEqual(len(chmod.call_args_list), 2)
        for call, dest in zip(chmod.call_args_list, (self.out, self.csv)):
            staged, mode = call.args
            self.assertEqual(os.path.dirname(staged), self.dir)
            self.assertTrue(os.path.basename(staged).startswith(".bvh_index-"))
            self.assertTrue(staged.endswith(".tmp"))
            self.assertEqual(mode, bvh_index.published_mode(dest))
        with open(self.out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["streams"][0]["offset"], 16)
        self.assertEqual(self.read(self.csv).splitlines()[1].split(",")[:2], ["a.rpf", "16"])
        self.assertEqual(sorted(os.listdir(self.dir)), ["index.csv", "index.json"])

    def test_failed_replace_of_read_only_report_removes_writable_temp(self):
        os.chmod(self.out, 0o444)
        self.addCleanup(lambda: os.path.exists(self.out) and os.chmod(self.out, 0o600))
        made_writable = []
        real_chmod = os.chmod

        def chmod(path, mode, *args, **kwargs):
            if isinstance(path, str) and os.path.basename(path).startswith(".bvh_index-") and mode == 0o600:
                made_writable.append(path)
            return real_chmod(path, mode, *args, **kwargs)

        with mock.patch.object(bvh_index.os, "replace", side_effect=OSError("replace denied")), \
                mock.patch.object(bvh_index.os, "chmod", chmod):
            with self.assertRaises(OSError):
                bvh_index.write_outputs([], [{"streams": 0}], self.out, None)
        self.assertEqual(len(made_writable), 1)
        self.assertEqual(self.read(self.out), self.OLD_JSON)
        self.assertEqual([n for n in os.listdir(self.dir) if n.startswith(".bvh_index-")], [])

    @unittest.skipUnless(os.name == "posix", "permission bits are POSIX")
    def test_replaced_report_keeps_its_permission_bits(self):
        os.chmod(self.out, 0o640)
        bvh_index.write_outputs([], [{"streams": 0}], self.out, None)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o640)


def setUpModule():
    # The tools' main() prints its summary JSON; keep it out of the test log (tests that check it redirect again).
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    unittest.addModuleCleanup(quiet.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
