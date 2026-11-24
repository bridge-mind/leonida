"""navmesh_index must find the measured 0DRP page directory in an RPF9 stream and in carved files.

    python -I tests/test_navmesh_index.py

Everything is synthetic: a hand-built RPF9 header, zlib streams and a fake page. No game data is used.
"""
import importlib.util
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
TOOL = os.path.join(ROOT, "tools", "navmesh_index.py")
KEY = 0x8A2D3FF3DF663960
FIELD68 = 141492


def load_tool_module():
    spec = importlib.util.spec_from_file_location("navmesh_index_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def page(version=122, pages=3, key=KEY, field68=FIELD68):
    """A 0x80-byte synthetic page matching the measured layout: 0DRP at 16, key at 0x60."""
    b = bytearray(0x80)
    struct.pack_into("<Q", b, 8, 0x50004000)       # pointer-like word, unused by the index
    b[16:20] = b"0DRP"
    struct.pack_into("<I", b, 20, version)
    struct.pack_into("<I", b, 32, pages)
    struct.pack_into("<Q", b, 0x60, key)
    struct.pack_into("<I", b, 0x68, field68)
    return bytes(b)


def make_rpf(path):
    """RPF9-like header (start word at 4) followed by three zlib streams."""
    header = bytearray(0x780)
    header[0:4] = b"9FPR"                           # bytes of little-endian 'RPF9'
    struct.pack_into("<I", header, 4, 0x780)
    streams = [zlib.compress(page(), 9),
               zlib.compress(b"not a page directory" * 100, 9),
               zlib.compress(b"0DRP" + bytes(124), 9)]   # marker at 0, must not match
    with open(path, "wb") as f:
        f.write(bytes(header) + b"".join(streams) + bytes(64))
    return streams


class NavmeshIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        r = subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r

    def test_rpf_stream(self):
        rpf = os.path.join(self.root, "test.rpf")
        streams = make_rpf(rpf)
        out, csv_path = os.path.join(self.root, "index.json"), os.path.join(self.root, "index.csv")
        self.run_tool(rpf, "--out", out, "--csv", csv_path)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        s = doc["summary"]
        self.assertEqual(s["mode"], "rpf")
        self.assertEqual(s["streams"], 3)
        self.assertEqual(s["matched"], 1)
        self.assertEqual(s["versions"], {"122": 1})
        self.assertEqual(s["page_counts"], {"3": 1})
        self.assertEqual(s["keys_present"], 1)
        self.assertEqual(s["compressed_bytes"], sum(len(z) for z in streams))
        row = doc["pages"][0]
        self.assertEqual(row["offset"], 0x780)
        self.assertEqual(row["stored"], len(streams[0]))
        self.assertEqual(row["size"], 0x80)
        self.assertEqual(row["signature"], "0DRP@16")
        self.assertEqual(row["version"], 122)
        self.assertEqual(row["pages"], 3)
        self.assertEqual(row["key64"], KEY.to_bytes(8, "little").hex())
        self.assertEqual(row["u32_68"], FIELD68)
        with open(csv_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], "source,offset,stored,size,signature,version,pages,key64,u32_68")
        self.assertEqual(len(lines), 2)
        self.assertIn("0DRP@16", lines[1])

    def test_directory(self):
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=2))
        with open(os.path.join(folder, "other.bin"), "wb") as f:
            f.write(b"plain" * 100)
        with open(os.path.join(folder, "ignored.txt"), "wb") as f:
            f.write(page())
        out = os.path.join(self.root, "dir.json")
        self.run_tool(folder, "--out", out, "--pattern", "*.bin")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 2)
        self.assertEqual(doc["summary"]["matched"], 1)
        self.assertEqual(doc["summary"]["page_counts"], {"2": 1})
        row = doc["pages"][0]
        self.assertEqual(row["source"], "page.bin")
        self.assertIsNone(row["offset"])
        self.assertIsNone(row["stored"])
        self.assertEqual(row["size"], 0x80)

    def test_recursive_pattern_counts_each_nested_path_once(self):
        folder = os.path.join(self.root, "carve")
        nested = os.path.join(folder, "group", "tiles")
        os.makedirs(nested)
        for directory in (folder, nested):
            with open(os.path.join(directory, "page.bin"), "wb") as f:
                f.write(page(pages=2))
        out = os.path.join(self.root, "recursive.json")
        self.run_tool(folder, "--out", out, "--pattern", "**/*.bin")
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        self.assertCountEqual([r["source"] for r in report["pages"]],
                              ["page.bin", os.path.join("group", "tiles", "page.bin")])
        self.assertEqual(report["summary"]["streams"], 2)
        self.assertEqual(report["summary"]["span"], 2 * 0x80)
        self.assertEqual(report["summary"]["page_counts"], {"2": 2})

    def test_explicit_recursive_prefix_does_not_rescan_directories(self):
        nav = load_tool_module()
        folder = os.path.join(self.root, "carve")
        leaf = os.path.join(folder, "tiles", "nested", "tiles")
        os.makedirs(leaf)
        with open(os.path.join(leaf, "page.bin"), "wb") as f:
            f.write(page())
        scans = []
        for pattern in ("*.bin", "**/*.bin", "**/tiles/**/*.bin"):
            with mock.patch.object(os, "scandir", wraps=os.scandir) as calls:
                rows, summary = nav.scan_directory(folder, pattern)
            self.assertEqual(summary["streams"], 1)
            self.assertEqual(rows[0]["source"], os.path.join("tiles", "nested", "tiles", "page.bin"))
            scans.append(calls.call_count)
        self.assertEqual(scans[1], scans[0])

    def test_directory_with_glob_metacharacters_in_its_name(self):
        # Before: os.path.join(root, "**", pattern) fed a root like "carve[1]" straight to glob, which read
        # the "[1]" as a character class and matched nothing -- the index came back empty.
        folder = os.path.join(self.root, "carve[1]")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=6))
        with open(os.path.join(folder, "other.bin"), "wb") as f:
            f.write(b"plain" * 100)
        out = os.path.join(self.root, "dir_bracket.json")
        self.run_tool(folder, "--out", out, "--pattern", "*.bin")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 2)
        self.assertEqual(doc["summary"]["matched"], 1)
        self.assertEqual(doc["summary"]["page_counts"], {"6": 1})
        self.assertEqual(doc["pages"][0]["source"], "page.bin")

    def test_folder_name_with_glob_characters_is_a_path(self):
        # glob read "[v1]" as a character class, so a folder named like this indexed nothing and exited 0.
        folder = os.path.join(self.root, "carve [v1]")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=2))
        out = os.path.join(self.root, "glob.json")
        self.run_tool(folder, "--out", out)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 1)
        self.assertEqual(doc["summary"]["matched"], 1)

    def test_a_non_archive_input_is_a_usage_error(self):
        # Before: a non-RPF7/8/9 file (or a truncated header) raised SystemExit -> exit 1 with a bare message;
        # the convention is one error line and exit 2 (the tool's own ap.error cases above).
        bad = os.path.join(self.root, "notrpf.bin")
        with open(bad, "wb") as f:
            f.write(b"\x00" * 64)
        r = subprocess.run([sys.executable, "-I", TOOL, bad], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("error:", r.stderr)
        self.assertIn("not an RPF7/8/9 archive", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        short = os.path.join(self.root, "short.rpf")
        with open(short, "wb") as f:
            f.write(b"7")
        r = subprocess.run([sys.executable, "-I", TOOL, short], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("not an RPF7/8/9 archive", r.stderr)

    def test_hard_link_to_a_scanned_file_in_a_glob_named_input_is_refused(self):
        folder = os.path.join(self.root, "carve [v1]")
        os.makedirs(folder)
        carved = os.path.join(folder, "page.bin")
        with open(carved, "wb") as f:
            f.write(page())
        alias = os.path.join(folder, "alias.dat")   # not matched by *.bin: only the scan's own glob finds the file
        try:
            os.link(carved, alias)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("hard links are not available here")
        r = subprocess.run([sys.executable, "-I", TOOL, folder, "--out", alias], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("would overwrite the input", r.stderr)
        with open(carved, "rb") as f:
            self.assertEqual(f.read(), page())

    def test_out_and_csv_in_missing_folders_are_created(self):
        # Before: write_outputs opened both paths directly, so a missing folder ended a finished scan in a
        # FileNotFoundError traceback and lost the index.
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=5))
        out = os.path.join(self.root, "no", "such", "index.json")
        csv_path = os.path.join(self.root, "also", "missing", "index.csv")
        self.run_tool(folder, "--out", out, "--csv", csv_path, "--pattern", "*.bin")
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["page_counts"], {"5": 1})
        with open(csv_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("0DRP@16", lines[1])

    def assert_output_collision(self, *args):
        r = subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--out and --csv must refer to different files", r.stderr)

    def test_explicit_output_collision_before_scan(self):
        # A missing input would give a different error if the scan started.
        self.assert_output_collision("missing.rpf", "--out", "new.json", "--csv", "./new.json")
        self.assertFalse(os.path.exists(os.path.join(self.root, "new.json")))

    def test_default_output_collision_preserves_report(self):
        make_rpf(os.path.join(self.root, "nav.rpf"))
        report = os.path.join(self.root, "nav.navmesh-index.json")
        with open(report, "wb") as f:
            f.write(b"previous report")
        self.assert_output_collision("nav.rpf", "--csv", report)
        with open(report, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_hard_link_output_collision(self):
        report = os.path.join(self.root, "out.json")
        with open(report, "wb") as f:
            f.write(b"previous report")
        try:
            os.link(report, os.path.join(self.root, "out.csv"))
        except OSError as exc:
            self.skipTest(f"cannot create a hard link: {exc}")
        self.assert_output_collision(self.root, "--out", "out.json", "--csv", "out.csv")
        with open(report, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_report_path_cannot_be_scanned_input(self):
        # Before: --out/--csv naming a file the scan reads replaced that file with the report, exit 0.
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        carved = os.path.join(folder, "page.bin")
        with open(carved, "wb") as f:
            f.write(page(pages=4))
        rpf = os.path.join(self.root, "nav.rpf")
        make_rpf(rpf)
        originals = {}
        for path in (carved, rpf):
            with open(path, "rb") as f:
                originals[path] = f.read()
        cases = [((folder, "--out", carved), carved), ((folder, "--csv", carved), carved),
                 ((rpf, "--out", rpf), rpf), ((rpf, "--csv", rpf), rpf)]
        for args, path in cases:
            r = subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT, capture_output=True, text=True,
                               check=False)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), originals[path], f"{args} touched the input")

    def test_differently_cased_report_path_cannot_be_scanned_input(self):
        # Before: on a case-insensitive filesystem, --out/--csv spelling the input folder in another case
        # passed the containment check and replaced the carved file with the report, exit 0.
        folder = os.path.join(self.root, "carve")
        os.makedirs(os.path.join(folder, "sub"))
        carved = os.path.join(folder, "page.bin")
        nested = os.path.join(folder, "sub", "deep.bin")
        for path in (carved, nested):
            with open(path, "wb") as f:
                f.write(page(pages=4))
        cased = os.path.join(self.root, "CARVE")
        if not os.path.isdir(cased):
            self.skipTest("the temporary folder is on a case-sensitive filesystem")
        module = load_tool_module()
        for spelled in (os.path.join(cased, "page.bin"), os.path.join(cased, "PAGE.BIN"),
                        os.path.join(cased, "sub", "deep.bin")):
            self.assertTrue(module.is_scanned_input(spelled, folder, None), spelled)
            self.assertTrue(module.is_scanned_input(spelled, cased, None), spelled)
        for flag in ("--out", "--csv"):
            r = subprocess.run([sys.executable, "-I", TOOL, folder, flag, os.path.join(cased, "page.bin")],
                               cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            with open(carved, "rb") as f:
                self.assertEqual(f.read(), page(pages=4), f"{flag} touched the input")

    def test_report_path_linked_to_a_scanned_file_is_scanned_input(self):
        # Before: a report path the pattern doesn't match was let through even when it is the same file as one the
        # scan reads. A hard link does that on any filesystem, and so does PAGE.BIN for page.bin on a
        # case-insensitive macOS or Linux volume, where fnmatch doesn't fold case: the report truncated the input.
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        carved = os.path.join(folder, "page.bin")
        with open(carved, "wb") as f:
            f.write(page(pages=4))
        linked = os.path.join(self.root, "report.json")
        try:
            os.link(carved, linked)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"no hard links here: {e}")
        module = load_tool_module()
        self.assertTrue(module.is_scanned_input(linked, folder, None))
        for flag in ("--out", "--csv"):
            r = subprocess.run([sys.executable, "-I", TOOL, folder, flag, linked],
                               cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            with open(carved, "rb") as f:
                self.assertEqual(f.read(), page(pages=4), f"{flag} touched the input")

    def test_unscanned_file_in_input_folder_is_not_scanned_input(self):
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page(pages=4))
        report = os.path.join(folder, "report.json")
        with open(report, "wb") as f:
            f.write(b"previous report")
        module = load_tool_module()
        self.assertFalse(module.is_scanned_input(report, folder, None))

    def test_sibling_folder_with_shared_prefix_is_not_scanned_input(self):
        folder = os.path.join(self.root, "carve")
        sibling = os.path.join(self.root, "carve2")
        os.makedirs(folder)
        os.makedirs(sibling)
        report = os.path.join(sibling, "page.bin")
        with open(report, "wb") as f:
            f.write(b"previous report")
        module = load_tool_module()
        self.assertFalse(module.is_scanned_input(report, folder, None))
        self.assertFalse(module.is_scanned_input(os.path.join(folder, "missing.bin"), folder, None))

    def test_not_an_rpf(self):
        path = os.path.join(self.root, "not.rpf")
        with open(path, "wb") as f:
            f.write(b"nope" * 64)
        r = subprocess.run([sys.executable, "-I", TOOL, path, "--out", os.path.join(self.root, "x.json")],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not an RPF7/8/9 archive", r.stderr)

    def test_limit_below_one_is_refused(self):
        path = os.path.join(self.root, "nav.rpf")
        make_rpf(path)
        for extra in (("--limit", "0"), ("--limit", "-100"), ("--start", "-1")):
            out = os.path.join(self.root, "x.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--out", out, *extra],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_refuses_start_and_limit(self):
        folder = os.path.join(self.root, "carve")
        os.makedirs(folder)
        with open(os.path.join(folder, "page.bin"), "wb") as f:
            f.write(page())
        for extra in (("--start", "0"), ("--limit", "128"), ("--start", "0", "--limit", "128")):
            out = os.path.join(self.root, "dir_bad.json")
            r = subprocess.run([sys.executable, "-I", TOOL, folder, "--out", out, *extra],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--start and --limit apply to a single archive, not a directory", r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_archive_refuses_pattern(self):
        rpf = os.path.join(self.root, "nav.rpf")
        make_rpf(rpf)
        for pat in ("*.bin", "*.dat"):
            out = os.path.join(self.root, "rpf_bad.json")
            r = subprocess.run([sys.executable, "-I", TOOL, rpf, "--out", out, "--pattern", pat],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--pattern applies to a directory, not an archive", r.stderr)
            self.assertFalse(os.path.exists(out))

    def test_scan_archive_closes_mmap_on_success(self):
        mod = load_tool_module()
        rpf = os.path.join(self.root, "close_ok.rpf")
        make_rpf(rpf)
        real_mmap = mod.mmap.mmap
        created = []

        def tracking_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        try:
            with mock.patch.object(mod.mmap, "mmap", side_effect=tracking_mmap):
                rows, summary = mod.scan_archive(rpf)
            self.assertEqual(len(rows), 1)
            self.assertEqual(summary["matched"], 1)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(rpf)
            self.assertFalse(os.path.exists(rpf))
        finally:
            for m in created:
                if not m.closed:
                    m.close()

    def test_scan_archive_closes_mmap_on_start_out_of_range(self):
        mod = load_tool_module()
        rpf = os.path.join(self.root, "close_err.rpf")
        make_rpf(rpf)
        size = os.path.getsize(rpf)
        real_mmap = mod.mmap.mmap
        created = []

        def tracking_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        try:
            with mock.patch.object(mod.mmap, "mmap", side_effect=tracking_mmap):
                # main() turns this into one usage error line (ap.error, exit 2); the mmap must still close.
                with self.assertRaises(ValueError):
                    mod.scan_archive(rpf, start=size)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(rpf)
            self.assertFalse(os.path.exists(rpf))
        finally:
            for m in created:
                if not m.closed:
                    m.close()


class MissingInputTest(unittest.TestCase):
    def test_missing_input_is_a_message_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "n.json")
            p = subprocess.run([sys.executable, "-I", TOOL, os.path.join(tmp, "missing.rpf"), "--out", out],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("missing.rpf", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))


class LimitWindowTest(unittest.TestCase):
    """--limit must bound the bytes read, not just where signatures are searched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rpf = os.path.join(self.tmp.name, "window.rpf")
        self.streams = make_rpf(self.rpf)
        self.mod = load_tool_module()

    def summary(self, limit):
        _rows, summary = self.mod.scan_archive(self.rpf, limit=limit)
        self.assertLessEqual(summary["compressed_bytes"], summary["span"])
        self.assertLessEqual(summary["readable_fraction"], 1.0)
        self.assertEqual(summary["compressed_bytes"] + summary["zero_padding"] + summary["unread_bytes"],
                         summary["span"])
        return summary

    def test_stream_crossing_the_limit_is_not_counted(self):
        # The zlib header is inside the window but its body is not: counting it would make the
        # compressed bytes exceed the span and readable_fraction greater than 1.
        summary = self.summary(10)
        self.assertEqual(summary["span"], 10)
        self.assertEqual(summary["streams"], 0)
        self.assertEqual(summary["compressed_bytes"], 0)

    def test_stream_ending_at_the_limit_is_counted(self):
        summary = self.summary(len(self.streams[0]))
        self.assertEqual(summary["span"], len(self.streams[0]))
        self.assertEqual(summary["streams"], 1)
        self.assertEqual(summary["compressed_bytes"], len(self.streams[0]))
        self.assertEqual(summary["readable_fraction"], 1.0)

    def test_window_too_short_for_a_signature_has_no_streams(self):
        summary = self.summary(1)
        self.assertEqual(summary["streams"], 0)
        self.assertEqual(summary["span"], 1)


class _Unprintable:
    """A cell whose str() fails part-way through the CSV, standing in for a write that dies mid-report."""

    def __str__(self):
        raise OSError("simulated write failure")


class PublishTest(unittest.TestCase):
    """write_outputs stages each report next to its destination and swaps it in: a run that fails while writing
    keeps the previous report instead of leaving it truncated (before: open(dest, "w") emptied it first)."""

    OLD_JSON = '{"summary": {"previous": true}, "pages": []}'
    OLD_CSV = "source,offset\nprevious.rpf,0\n"

    def setUp(self):
        self.tool = load_tool_module()
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
            self.tool.write_outputs(rows, {"streams": 1}, self.out, self.csv)
        self.assert_previous_reports_kept()

    def test_csv_write_failure_keeps_previous_csv(self):
        good = os.path.join(self.dir, "new.json")
        rows = [{"source": _Unprintable(), "offset": 0}]
        with mock.patch.object(self.tool.json, "dump"):
            with self.assertRaises(OSError):
                self.tool.write_outputs(rows, {"streams": 1}, good, self.csv)
        self.assertEqual(self.read(self.csv), self.OLD_CSV)
        self.assertEqual(sorted(os.listdir(self.dir)), ["index.csv", "index.json", "new.json"])

    def test_interrupted_write_keeps_previous_report(self):
        with mock.patch.object(self.tool.json, "dump", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.tool.write_outputs([], {"streams": 0}, self.out, self.csv)
        self.assert_previous_reports_kept()

    def test_reports_replaced_with_published_mode(self):
        rows = [{"source": "a.rpf", "offset": 16, "signature": "0DRP@16"}]
        with mock.patch.object(self.tool, "published_mode", wraps=self.tool.published_mode) as published, \
                mock.patch.object(self.tool.os, "chmod", wraps=os.chmod) as chmod:
            written = self.tool.write_outputs(rows, {"streams": 1}, self.out, self.csv)
        self.assertEqual(written, [self.csv])
        self.assertEqual([c.args for c in published.call_args_list], [(self.out,), (self.csv,)])
        self.assertEqual(len(chmod.call_args_list), 2)
        for call, dest in zip(chmod.call_args_list, (self.out, self.csv)):
            staged, mode = call.args
            self.assertEqual(os.path.dirname(staged), self.dir)
            self.assertTrue(os.path.basename(staged).startswith(".navmesh_index-"))
            self.assertTrue(staged.endswith(".tmp"))
            self.assertEqual(mode, self.tool.published_mode(dest))
        with open(self.out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["pages"][0]["offset"], 16)
        self.assertEqual(self.read(self.csv).splitlines()[1].split(",")[:2], ["a.rpf", "16"])
        self.assertEqual(sorted(os.listdir(self.dir)), ["index.csv", "index.json"])

    @unittest.skipUnless(os.name == "posix", "permission bits are POSIX")
    def test_replaced_report_keeps_its_permission_bits(self):
        os.chmod(self.out, 0o640)
        self.tool.write_outputs([], {"streams": 0}, self.out, None)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o640)

    def test_failed_replace_of_read_only_report_removes_writable_temp(self):
        os.chmod(self.out, 0o444)
        self.addCleanup(lambda: os.path.exists(self.out) and os.chmod(self.out, 0o600))
        made_writable = []
        real_chmod = os.chmod

        def chmod(path, mode, *args, **kwargs):
            if isinstance(path, str) and os.path.basename(path).startswith(".navmesh_index-") and mode == 0o600:
                made_writable.append(path)
            return real_chmod(path, mode, *args, **kwargs)

        with mock.patch.object(self.tool.os, "replace", side_effect=OSError("replace denied")), \
                mock.patch.object(self.tool.os, "chmod", chmod):
            with self.assertRaises(OSError):
                self.tool.write_outputs([], {"streams": 0}, self.out, None)
        self.assertEqual(len(made_writable), 1)
        self.assertEqual(self.read(self.out), self.OLD_JSON)
        self.assertEqual([n for n in os.listdir(self.dir) if n.startswith(".navmesh_index-")], [])


class OutputPathTest(unittest.TestCase):
    """A report path that is already a folder is a usage error, not a PermissionError traceback."""

    def test_out_that_is_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = os.path.join(tmp, "carved"), os.path.join(tmp, "reports")
            os.makedirs(source)
            os.makedirs(out)
            r = subprocess.run([sys.executable, "-I", TOOL, source, "--out", out],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--out is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
