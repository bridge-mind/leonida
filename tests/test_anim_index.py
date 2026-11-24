"""Synthetic-header tests for tools/anim_index.py.

    python -I tests/test_anim_index.py

Every byte here is built by the test. No game file or game asset name is used.
"""
import contextlib
import csv
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
TOOL = os.path.join(ROOT, "tools", "anim_index.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_index  # noqa: E402


@contextlib.contextmanager
def tracked_mmaps():
    """Record every mmap.mmap opened by anim_index and close any leak on exit."""
    real_mmap = anim_index.mmap.mmap
    created = []

    def wrap(*args, **kwargs):
        mapping = real_mmap(*args, **kwargs)
        created.append(mapping)
        return mapping

    try:
        with mock.patch.object(anim_index.mmap, "mmap", side_effect=wrap):
            yield created
    finally:
        for mapping in created:
            if not mapping.closed:
                mapping.close()


def move():
    data = bytearray(72)
    data[:4] = b"MoVE"
    struct.pack_into("<I", data, 4, 19)
    struct.pack_into("<I", data, 0x14, 852)
    return bytes(data)


def page():
    data = bytearray(0x60)
    struct.pack_into("<Q", data, 8, 0x50100420)
    data[16:20] = b"0DRP"
    struct.pack_into("<I", data, 0x14, 122)
    struct.pack_into("<Q", data, 0x20, 3)
    return bytes(data)


def names():
    data = b"pack:/alpha.clip\x00pack:/bravo.anim\x00"
    return data + bytes(64 - len(data))


def skeleton():
    data = b"SKEL_Spine\x00SKEL_Neck\x00"
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


def rbf0():
    data = b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1) + struct.pack("<H", 21)
    data += b"rage__crRigSetManager\x00"
    return data + bytes(64 - len(data))


def raw():
    return b"\x11\x22\x33\x44" + bytes(60)


def text():
    data = b"<AnimData></AnimData>" * 4
    return data + bytes(max(0, 64 - len(data)))


def bink():
    data = bytearray(0x40)
    data[:4] = b"KB2n"
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 8, 1)
    struct.pack_into("<I", data, 0x14, 64)
    struct.pack_into("<I", data, 0x18, 64)
    return bytes(data)


def _u16s(text):
    return struct.pack("<H", len(text)) + text


def _fd(text):
    return b"\xfd\xff" + struct.pack("<I", len(text) + 1) + text + b"\x00"


def _entry(source, rig, group):
    return _fd(source) + b"\xff\xff\x06\x00" + bytes(6) + _fd(rig) + b"\xff\xff\x07\x00" + bytes(6) + _fd(group)


CHUNK = 65536   # anim_index.MAX_CHUNK: the 64 KiB stream size of a split resource


def rbf0_blob():
    """Synthetic RBF0 with three rig sets (one empty), three rig entries and three source kinds."""
    b = bytearray(b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1) + _u16s(b"rage__crRigSetManager") + b"\x00")
    b += b"\x03\x60\x03\x00key" + _u16s(b"anim/rigset/synthetic/one")
    b += b"\x04\x00" + bytes(6) + b"\x02\x00" + bytes(6) + b"\x05\x00" + bytes(6)
    b += _entry(b"canonical@source", b"anim/rig/canonical/source/synthetic/one", b"pose")
    b += _entry(b"generated@source", b"anim/rig/generated/source/synthetic/two", b"deformation")
    b += b"\xff" * 6
    b += b"\x02\x00\x00\x00\x00\x00\x01\x00\x03\x60" + _u16s(b"anim/rigset/synthetic/empty")
    b += b"\x04\x00" + bytes(6) + b"\xff" * 4
    b += b"\x02\x00\x00\x00\x00\x00\x01\x00\x03\x60" + _u16s(b"anim/rigset/synthetic/three")
    b += b"\x04\x00" + bytes(6) + b"\x02\x00" + bytes(6) + b"\x05\x00" + bytes(6)
    b += _entry(b"pipelined@source", b"anim/rig/pipelined/source/synthetic/three", b"facial")
    b += b"\xff" * 6
    return bytes(b)


PAYLOADS = [move(), page(), names(), skeleton(), chunks(), clip(), rbf0(), raw(), text(), bink()]
EXPECTED = ["move", "page-directory", "name-table", "skeleton-names", "typehash", "clip-stream",
            "rbf0", "raw:11223344", "text", "bink2"]


class ClassifyTest(unittest.TestCase):
    def test_signatures(self):
        got = [anim_index.classify(p, 0, len(p))["signature"] for p in PAYLOADS]
        self.assertEqual(got, EXPECTED)

    def test_move_header(self):
        r = anim_index.classify(move(), 0x100, 90)
        self.assertEqual((r["version"], r["detail"]), (19, "size_hint=852"))

    def test_page_header(self):
        r = anim_index.classify(page(), 0x200, 40)
        self.assertEqual((r["version"], r["count"], r["detail"]), (122, 3, "0DRP at +16"))

    def test_names(self):
        r = anim_index.classify(names(), 0x300, 30)
        self.assertEqual((r["name"], r["count"]), ("pack:/alpha.clip", 2))

    def test_skeleton(self):
        r = anim_index.classify(skeleton(), 0x400, 20)
        self.assertEqual((r["name"], r["count"]), ("SKEL_Spine", 2))

    def test_typehash(self):
        r = anim_index.classify(chunks(), 0x500, 70)
        self.assertEqual((r["version"], r["count"], r["detail"]), (8, 7, "type=0x88560988"))

    def test_clip_header(self):
        r = anim_index.classify(clip(), 0x600, 70)
        self.assertEqual((r["signature"], r["version"], r["count"], r["detail"]),
                         ("clip-stream", None, 3, "aux-table tags=0x101,0x113,0x207 payload_end=52"))

    def test_aux_two_entries_exact_fit(self):
        data = aux([(0x101, 40), (0x207, 20)])   # payloads end exactly at the end of the stream
        self.assertEqual(anim_index.aux_table(data), ([(0x101, 40), (0x207, 20)], len(data)))
        r = anim_index.classify(data, 0, len(data))
        self.assertEqual((r["signature"], r["count"]), ("clip-stream", 2))

    def test_aux_trailing_schema_bytes_kept(self):
        data = aux([(0x101, 16), (0x204, 8), (0x207, 8)], trailing=bytes(range(64)))
        self.assertEqual(anim_index.aux_table(data)[1], 4 + 24 + 32)
        self.assertEqual(anim_index.classify(data, 0, len(data))["signature"], "clip-stream")

    def test_aux_payload_overruns_stream(self):
        short = aux([(0x101, 40), (0x207, 30)])[:-1]   # last payload one byte short
        self.assertIsNone(anim_index.aux_table(short))
        self.assertEqual(anim_index.classify(short, 0, len(short))["signature"], "raw:02000000")

    def test_aux_declared_size_too_large(self):
        data = aux([(0x101, 8), (0x113, 0xFFFFFFF0), (0x207, 8)], payload=bytes(64))
        self.assertIsNone(anim_index.aux_table(data))
        self.assertNotEqual(anim_index.classify(data, 0, len(data))["signature"], "clip-stream")

    def test_aux_truncated_directory(self):
        data = aux([(0x101, 0), (0x113, 0), (0x207, 0)])[:20]   # the matching 8-byte prefix, directory cut
        self.assertIsNone(anim_index.aux_table(data))
        self.assertIsNone(anim_index.aux_table(data[:8]))
        self.assertEqual(anim_index.classify(data[:8], 0, 8)["signature"], "raw:03000000")

    def test_aux_wrong_count_or_first_tag(self):
        for entries in ([(0x101, 4)], [(0x101, 4)] * 4, [(0x113, 4), (0x207, 4)], [(0x207, 4), (0x101, 4)]):
            data = aux(entries, trailing=b"\x5a" * 64)
            self.assertIsNone(anim_index.aux_table(data), entries)
            self.assertNotEqual(anim_index.classify(data, 0, len(data))["signature"], "clip-stream")

    def test_aux_prefix_with_overrunning_size_is_generic(self):
        # The old 8-byte prefix check called this a clip stream; its declared payload does not fit.
        data = struct.pack("<III", 2, 0x101, 1 << 20) + bytes(60)
        self.assertEqual(anim_index.classify(data, 0, len(data))["signature"], "raw:02000000")

    def test_aux_valid_page_stops_rbf0_continuation(self):
        self.assertIn(anim_index.classify(clip(), 0, len(clip()))["signature"], anim_index.STOP_SIGNATURES)

    def test_rbf0(self):
        r = anim_index.classify(rbf0(), 0x700, 30)
        self.assertEqual((r["name"], r["detail"]), ("rage__crRigSetManager", "rig metadata"))


class RigParseTest(unittest.TestCase):
    def test_joaat_known_value(self):
        self.assertEqual(anim_index.joaat32("adder"), 0xB779A091)

    def test_rbf0_rigsets_and_entries(self):
        r = anim_index.parse_rbf0(rbf0_blob())
        self.assertEqual(r["header"]["schema"], "rage__crRigSetManager")
        self.assertEqual((r["summary"]["rigsets"], r["summary"]["entries"], r["summary"]["distinct_rigs"]),
                         (3, 3, 3))
        self.assertEqual(r["summary"]["empty_rigsets"], 1)
        self.assertEqual(r["summary"]["sources"]["canonical@source"]["entries"], 1)
        self.assertEqual(r["summary"]["sources"]["generated@source"]["distinct_rigs"], 1)
        self.assertEqual(r["summary"]["sources"]["pipelined@source"]["entries"], 1)
        self.assertEqual(r["summary"]["groups"], {"pose": 1, "deformation": 1, "facial": 1})
        self.assertEqual(r["rigsets"][0]["entries"][0]["source_offset"],
                         rbf0_blob().find(b"canonical@source") - 6)
        self.assertEqual(r["rigsets"][0]["distinct_rigs"], 2)

    def test_rbf0_rejects_non_rbf0(self):
        with self.assertRaises(ValueError):
            anim_index.parse_rbf0(b"not an RBF0 blob")

    def test_skel_names_and_hashes(self):
        name = b"SKEL_Synthetic"
        data = name + b"\x00" + struct.pack("<I", anim_index.joaat32(name.decode())) + b"\x00SKEL_Second\x00"
        r = anim_index.parse_skel_names(data, 0x123)
        self.assertEqual((r["offset"], r["occurrences"], r["distinct"], r["hashes_present"]),
                         (0x123, 2, 2, 1))


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
        rows, summary = anim_index.scan_rpf(self.rpf)
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
        rows, summary = anim_index.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signature"], "move")
        self.assertEqual(summary["span"], limit)

    def test_csv_and_json(self):
        rows, summary = anim_index.scan_rpf(self.rpf)
        csv_path = os.path.join(self.root, "out.csv")
        json_path = os.path.join(self.root, "out.json")
        anim_index.write_output(csv_path, rows, summary)
        anim_index.write_output(json_path, rows, summary)
        with open(csv_path, newline="", encoding="utf-8") as f:
            table = list(csv.reader(f))
        self.assertEqual(table[0], anim_index.COLUMNS)
        self.assertEqual(len(table), len(PAYLOADS) + 1)
        self.assertEqual(table[1][3], "move")
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], len(PAYLOADS))
        self.assertEqual(len(doc["files"]), len(PAYLOADS))

    def test_bad_scan_arguments_fail_before_an_index_is_written(self):
        for flag, value, message in (("--start", "-1", "must be 0 or more"),
                                     ("--limit", "0", "must be at least 1"),
                                     ("--limit", "-100", "must be at least 1")):
            out = os.path.join(self.root, "bad.json")
            p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet", flag, value],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(message, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_missing_source_or_out_folder_is_one_line(self):
        missing = os.path.join(self.root, "missing.rpf")
        out_in_missing = os.path.join(self.root, "no_dir", "out.json")
        for source, out, path in ((missing, os.path.join(self.root, "m.json"), missing),
                                  (self.rpf, out_in_missing, out_in_missing)):
            with self.subTest(path=os.path.relpath(path, self.root)):
                p = subprocess.run([sys.executable, "-I", TOOL, source, out, "--quiet"],
                                   capture_output=True, text=True, errors="replace")
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertTrue(p.stderr.startswith(path + ": "), p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertEqual(p.stdout, "")
        self.assertFalse(os.path.exists(os.path.join(self.root, "m.json")))

    def test_cli(self):
        out = os.path.join(self.root, "cli.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["signatures"]["move"]["files"], 1)
        self.assertEqual(doc["summary"]["signatures"]["page-directory"]["files"], 1)

    def test_cli_rigs_without_rigset_keys_keeps_existing_file(self):
        # PAYLOADS' RBF0 stream has a schema name but no anim/rigset/ keys, so parse_rbf0 rejects it.
        out = os.path.join(self.root, "index.json")
        rigs = os.path.join(self.root, "rigs.json")
        with open(rigs, "w", encoding="utf-8") as f:
            f.write('{"previous": true}')
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--rigs", rigs, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("no anim/rigset/ keys found", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        with open(rigs, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"previous": True})

    def test_cli_rejects_folder_destination(self):
        folder_out = os.path.join(self.root, "out_folder")
        os.makedirs(folder_out)
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, folder_out, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("is a folder, not a .csv or .json file", p.stderr)

    def test_cli_rejects_invalid_extension(self):
        txt_out = os.path.join(self.root, "out.txt")
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, txt_out, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("output must end in .json or .csv", p.stderr)

    def test_cli_cannot_create_a_report_in_place_of_the_carve_manifest(self):
        # <source>/index.json is the carve manifest's name even before a carve has written it.
        carve = os.path.join(self.root, "carved_no_manifest")
        os.makedirs(carve)
        manifest = os.path.join(carve, "index.json")
        out_csv = os.path.join(self.root, "no_manifest.csv")
        for args, flag in (([manifest], "output"), ([out_csv, "--rigs", manifest], "--rigs")):
            with self.subTest(flag=flag):
                p = subprocess.run([sys.executable, "-I", TOOL, carve, *args, "--quiet"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn(f"{flag} would overwrite the carve manifest", p.stderr)
                self.assertFalse(os.path.exists(manifest))

    def test_cli_carve_dir_protects_manifest_and_input_files(self):
        carve = os.path.join(self.root, "carved_manifest_test")
        os.makedirs(carve)
        manifest = os.path.join(carve, "index.json")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write('{"manifest": true}')
        bin_file = os.path.join(carve, "000000001000.bin")
        with open(bin_file, "wb") as f:
            f.write(b"A" * 64)
        out_csv = os.path.join(self.root, "valid.csv")

        # Refuse to overwrite carve manifest via out
        p = subprocess.run([sys.executable, "-I", TOOL, carve, manifest, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("output would overwrite the carve manifest", p.stderr)

        # Refuse to overwrite carve manifest via --rigs
        p = subprocess.run([sys.executable, "-I", TOOL, carve, out_csv, "--rigs", manifest, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("--rigs would overwrite the carve manifest", p.stderr)

        # Refuse to overwrite carved input file via out
        p = subprocess.run([sys.executable, "-I", TOOL, carve, bin_file, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("output would overwrite an input file", p.stderr)

        # Refuse to overwrite carved input file via --rigs
        p = subprocess.run([sys.executable, "-I", TOOL, carve, out_csv, "--rigs", bin_file, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("--rigs would overwrite an input file", p.stderr)

    def test_directory(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((move(), page(), names())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        rows, summary = anim_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])
        self.assertEqual([r["signature"] for r in rows], ["move", "page-directory", "name-table"])
        self.assertEqual(summary["mode"], "directory")

    def test_directory_classifies_each_payload_before_opening_the_next(self):
        folder = os.path.join(self.root, "streamed")
        os.mkdir(folder)
        payloads = (move(), page(), names())
        for name, payload in zip(("00001000.bin", "a.bin", "b.bin"), payloads):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(payload)

        classify = mock.Mock(wraps=anim_index.classify)
        reads = []

        def tracked_open(path, mode):
            self.assertEqual(classify.call_count, len(reads))
            reads.append(os.path.basename(path))
            return open(path, mode)

        with mock.patch.object(anim_index, "classify", classify), \
                mock.patch.object(anim_index, "open", tracked_open, create=True):
            rows, summary = anim_index.scan_dir(folder)
        self.assertEqual(reads, ["00001000.bin", "a.bin", "b.bin"])
        self.assertEqual([row["offset"] for row in rows], [0x1000, 1, 2])
        self.assertEqual(summary["span"], sum(map(len, payloads)))

    def test_directory_rejects_start_and_limit(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for offset in (0x100, 0x200):
            with open(os.path.join(folder, f"{offset:012x}.bin"), "wb") as f:
                f.write(b"A" * 64)
        cases = [("--start", "0"), ("--limit", "1"), ("--limit", "1", "--start", "0x200")]
        for i, flags in enumerate(cases):
            with self.subTest(flags=flags):
                out = os.path.join(self.root, f"bad{i}.json")
                p = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet", *flags],
                                   capture_output=True, text=True, errors="replace")
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertEqual(len(p.stderr.splitlines()), 1, p.stderr)
                for flag in flags[::2]:
                    self.assertIn(flag, p.stderr)
                self.assertFalse(os.path.exists(out))

    def test_directory_skips_the_carvers_index_and_reports(self):
        """rpf9_carve.py writes index.json next to the streams; it and a report are not payloads."""
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        for i, payload in enumerate((move(), page(), names())):
            with open(os.path.join(folder, f"{0x1000 + i * 0x100:012x}.bin"), "wb") as f:
                f.write(payload)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
            f.write('[{"off": 4096, "stored": 16}]')
        with open(os.path.join(folder, "report.csv"), "w", encoding="utf-8") as f:
            f.write("offset,size\n")
        rows, summary = anim_index.scan_dir(folder)
        self.assertEqual([r["offset"] for r in rows], [0x1000, 0x1100, 0x1200])

    def test_scan_rpf_closes_mmap_on_invalid_start(self):
        with tracked_mmaps() as created:
            with self.assertRaisesRegex(ValueError, "outside the archive"):
                anim_index.scan_rpf(self.rpf, start=os.path.getsize(self.rpf) + 1)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].closed)
            os.remove(self.rpf)
            self.assertFalse(os.path.exists(self.rpf))


class RigArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rpf = os.path.join(self.root, "rigs.rpf")
        # A real continuation follows a full 64 KiB stream: pad the rig table (parse_rbf0 ignores the tail)
        # and cut it there.
        blob = rbf0_blob()
        blob += bytes(CHUNK + 4096 - len(blob))
        self.parts = [blob[:CHUNK], blob[CHUNK:]]
        self.skel = b"SKEL_Synthetic\x00SKEL_Second\x00"
        self.skel += bytes(64 - len(self.skel))
        self.front = b"\x0f\x00\xe0\x00\x0c\x00\x16\x00SKEL_Front\x00"
        self.front += struct.pack("<I", anim_index.joaat32("SKEL_Front")) + bytes(64)
        header = bytearray(0x100)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = bytearray()
        for payload in self.parts + [self.skel, self.front]:
            body += zlib.compress(payload)
            body += bytes((-len(body)) % 16)
        with open(self.rpf, "wb") as f:
            f.write(header + body)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_and_parse_continuation(self):
        rows, _ = anim_index.scan_rpf(self.rpf)
        blob = anim_index.collect_rbf0(self.rpf, rows)
        self.assertIsNotNone(blob)
        r = anim_index.parse_rbf0(blob)
        self.assertEqual(r["summary"]["rigsets"], 3)
        self.assertEqual(r["summary"]["entries"], 3)

    def test_collect_skeletons_dir_matches_archive_classification(self):
        carved = os.path.join(self.root, "carved_skels")
        os.makedirs(carved)
        big_skel = b"SKEL_Large\x00" + b"SKEL_Other\x00" + b"\x00" * 20000
        rbf0_with_skel = b"RBF0" + b"\x00" * 16 + b"SKEL_NotASkeleton\x00" + b"\x00" * 32
        with open(os.path.join(carved, "000000000100.bin"), "wb") as f:
            f.write(self.skel)
        with open(os.path.join(carved, "000000000200.bin"), "wb") as f:
            f.write(big_skel)
        with open(os.path.join(carved, "000000000300.bin"), "wb") as f:
            f.write(self.front)
        with open(os.path.join(carved, "000000000400.bin"), "wb") as f:
            f.write(rbf0_with_skel)
        got = anim_index.collect_skeletons_dir(carved)
        self.assertEqual(
            [(r["offset"], r["kind"], r["names"]) for r in got],
            [
                (0x100, "plain-bone-table", ["SKEL_Second", "SKEL_Synthetic"]),
                (0x200, "plain-bone-table", ["SKEL_Large", "SKEL_Other"]),
                (0x300, "bone-table", ["SKEL_Front"]),
            ],
        )

    def test_cli_rejects_a_shared_rig_destination(self):
        out = os.path.join(self.root, "index.json")
        alias = os.path.join(self.root, ".", "index.json")
        result = subprocess.run(
            [sys.executable, "-I", TOOL, self.rpf, out, "--rigs", alias, "--quiet"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("index and --rigs must use different paths", result.stderr)
        self.assertFalse(os.path.exists(out))

    def test_cli_rejects_rigs_folder_and_invalid_extension(self):
        out = os.path.join(self.root, "index.json")
        folder = os.path.join(self.root, "rig_folder")
        os.makedirs(folder)
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--rigs", folder, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("is a folder, not a JSON file", p.stderr)

        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--rigs", os.path.join(self.root, "rigs.csv"), "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("--rigs must end in .json", p.stderr)

    def test_cli_rigs(self):
        out = os.path.join(self.root, "index.json")
        rigs = os.path.join(self.root, "rigs.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--rigs", rigs, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(rigs, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["rbf0"]["summary"]["rigsets"], 3)
        self.assertEqual(doc["skeletons"][0]["distinct"], 2)
        self.assertEqual(doc["skeletons"][0]["kind"], "plain-bone-table")
        self.assertEqual(doc["skeletons"][1]["hashes_present"], 1)

    def test_cli_rigs_stored_name_hashes(self):
        hashed = b"\x3b\x39\x77\x99"     # JOAAT32("anim/rig/a") = 0x9977393b, little-endian
        base = bytearray(b"RBF0" + struct.pack("<H", 0xFFFB) + struct.pack("<I", 1)
                         + _u16s(b"rage__crRigSetManager") + b"\x00")
        base += b"\x03\x60" + _u16s(b"anim/rigset/synthetic/one")
        base += _entry(b"canonical@source", b"anim/rig/a", b"pose") + b"\xff" * 6
        base += bytes((1 - len(base)) % 4)
        self.assertEqual(len(base) % 4, 1)
        cases = {"absent": (hashed[::-1], 0),
                 "unaligned": (hashed, 1),
                 "duplicate": (hashed + bytes(3) + hashed, 1)}
        for label, (tail, expected) in cases.items():
            with self.subTest(label):
                carved = os.path.join(self.root, label)
                os.mkdir(carved)
                with open(os.path.join(carved, "000000.bin"), "wb") as f:
                    f.write(bytes(base) + tail)
                out = os.path.join(self.root, label + "-index.json")
                rigs = os.path.join(self.root, label + "-rigs.json")
                p = subprocess.run([sys.executable, "-I", TOOL, carved, out, "--rigs", rigs, "--quiet"],
                                   capture_output=True, text=True, errors="replace")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                with open(rigs, encoding="utf-8") as f:
                    summary = json.load(f)["rbf0"]["summary"]
                self.assertEqual((summary["distinct_names"], summary["stored_name_hashes"]), (2, expected))


class ShortStreamEndsTheResource(unittest.TestCase):
    """A stream shorter than 64 KiB is the last one of its resource: the adjacent stream is not joined."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rbf0 = rbf0_blob()
        self.other = b"\x5a" * 300   # raw, adjacent, no stop signature: only the size rule ends the join

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive(self):
        rpf = os.path.join(self.root, "rigs.rpf")
        header = bytearray(0x100)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = bytearray()
        for payload in (self.rbf0, self.other):
            body += zlib.compress(payload)
            body += bytes((-len(body)) % 16)
        with open(rpf, "wb") as f:
            f.write(header + body)
        rows, _ = anim_index.scan_rpf(rpf)
        self.assertEqual(anim_index.collect_rbf0(rpf, rows), self.rbf0)

    def test_directory(self):
        with open(os.path.join(self.root, "001000.bin"), "wb") as f:
            f.write(self.rbf0)
        with open(os.path.join(self.root, f"{0x1000 + len(self.rbf0):06x}.bin"), "wb") as f:
            f.write(self.other)
        self.assertEqual(anim_index.collect_rbf0_dir(self.root), self.rbf0)

    def test_directory_full_stream_still_joins(self):
        blob = self.rbf0 + bytes(CHUNK + 100 - len(self.rbf0))
        with open(os.path.join(self.root, "001000.bin"), "wb") as f:
            f.write(blob[:CHUNK])
        with open(os.path.join(self.root, f"{0x1000 + CHUNK:06x}.bin"), "wb") as f:
            f.write(blob[CHUNK:])
        self.assertEqual(anim_index.collect_rbf0_dir(self.root), blob)

    def test_directory_rigs_skips_json_and_csv_reports(self):
        blob = self.rbf0 + bytes(CHUNK + 100 - len(self.rbf0))
        with open(os.path.join(self.root, "000000.bin"), "wb") as f:
            f.write(blob[:CHUNK])
        with open(os.path.join(self.root, "010000.bin"), "wb") as f:
            f.write(blob[CHUNK:])
        with open(os.path.join(self.root, "020000.bin"), "wb") as f:
            f.write(skeleton())
        for report in ("index.json", "report.CSV"):
            with open(os.path.join(self.root, report), "w", encoding="utf-8", newline="\n") as f:
                f.write("{\"name\": \"SKEL_FakeReportBone\"}\n" * 4)
        self.assertEqual(anim_index.collect_rbf0_dir(self.root), blob)
        skels = anim_index.collect_skeletons_dir(self.root)
        self.assertEqual([s["offset"] for s in skels], [0x20000])
        self.assertEqual(skels[0]["names"], ["SKEL_Neck", "SKEL_Spine"])


if __name__ == "__main__":
    unittest.main()
