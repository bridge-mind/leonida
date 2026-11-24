"""Tests for tools/text_tables.py, built from synthetic resources and conversation data (no game data).

    python -I tests/test_text_tables.py
"""
import contextlib
import csv
import errno
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "text_tables.py")
    spec = importlib.util.spec_from_file_location("text_tables_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_module()


def ptr(off):
    return struct.pack("<II", 0x50000000 | off, 0)


def make_resource(strings, pad=0):
    """A logical resource: 0DRP at +0x10, a descriptor, then hash table, index array and string blob.

    ``strings`` maps label -> text. ``pad`` adds bytes before the tables so they can cross a 64 KiB boundary.
    """
    labels = list(strings)
    blob = bytearray()
    blob_offsets = []
    for label in labels:
        blob_offsets.append(len(blob))
        blob += strings[label].encode("utf-8") + b"\0"
    head = bytearray(0x10) + b"0DRP" + bytes(0x0C)
    desc_at = len(head)
    desc_len = 8 + 4 + 4 + 8 + 4 + 4 + 8 + 4 + 4 + 8
    index_at = desc_at + desc_len + pad
    hash_at = index_at + 16 * len(labels)
    blob_at = hash_at + 16 * len(labels)
    desc = (ptr(index_at) + struct.pack("<II", 0, 0x10) + ptr(0) + b"\x01\x01\x00\x00"
            + struct.pack("<I", 16 * len(labels)) + ptr(hash_at) + struct.pack("<II", 0x10, len(blob))
            + ptr(blob_at))
    assert len(desc) == desc_len
    index = b"".join(struct.pack("<QQ", 0, off << 20) for off in blob_offsets)
    hashes = bytearray()
    for i, label in enumerate(labels):
        entry = bytearray(16)
        struct.pack_into("<II", entry, 0, T.joaat(label), 0)
        entry[8] = 1
        entry[11:14] = i.to_bytes(3, "little")
        hashes += entry
    return bytes(head + desc + bytes(pad) + index + hashes + blob)


def make_chunked_resource(chunks, between=(), gap=b"GAP-BETWEEN-CHUNKS\0"):
    """A string table whose blob is split into chunks, as tables with more than 1 MiB of text are.

    ``chunks`` is a list of {label: text}, one per chunk. The descriptor lists one tag-0x10 entry per chunk, with the
    ``between`` (tag, bytes) entries after the first one. ``gap`` sits between the chunks so they are not contiguous.
    An index entry's second word is ``offset << 20 | 4 | chunk``. Bytes that are not an entry end the descriptor.
    """
    labels, refs, blobs = [], [], []
    for k, strings in enumerate(chunks):
        blob = bytearray()
        for label, text in strings.items():
            labels.append(label)
            refs.append(len(blob) << 20 | 4 | k)
            blob += text.encode("utf-8") + b"\0"
        blobs.append(bytes(blob))
    head = bytearray(0x10) + b"0DRP" + bytes(0x0C)
    desc_at = len(head)
    desc_len = 0x28 + 16 * (len(chunks) + len(between)) + 16
    index_at = desc_at + desc_len
    hash_at = index_at + 16 * len(labels)
    blob_at, at = [], hash_at + 16 * len(labels)
    for blob in blobs:
        blob_at.append(at)
        at += len(blob) + len(gap)
    entries = [(0x10, len(blobs[0]), blob_at[0])] + [(tag, size, 0) for tag, size in between]
    entries += [(0x10, len(blob), off) for blob, off in zip(blobs[1:], blob_at[1:])]
    desc = (ptr(index_at) + struct.pack("<II", 0, 0x10) + ptr(0) + b"\x01\x01\x00\x00"
            + struct.pack("<I", 16 * len(labels)) + ptr(hash_at))
    desc += b"".join(struct.pack("<II", tag, size) + ptr(off) for tag, size, off in entries)
    desc += struct.pack("<IIII", 0x10, 0x10, 0x10, 0x10)      # not an entry: its pointer word is not 0x5xxxxxxx
    assert len(desc) == desc_len
    index = b"".join(struct.pack("<QQ", 0, ref) for ref in refs)
    hashes = bytearray()
    for i, label in enumerate(labels):
        entry = bytearray(16)
        struct.pack_into("<II", entry, 0, T.joaat(label), 0)
        entry[8] = 1
        entry[11:14] = i.to_bytes(3, "little")
        hashes += entry
    return bytes(head + desc + index + hashes + b"".join(blob + gap for blob in blobs))


def write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)
    return path


EN = {"ZZ_LINE_A_1": "This is the first line and you know it", "ZZ_LINE_B_2": "What is that, the second?",
      "ZZ_MENU_TITLE": "the menu"}
DE = {"ZZ_LINE_A_1": "Das ist die erste Zeile und du weißt es", "ZZ_LINE_B_2": "Was ist das, die zweite?",
      "ZZ_MENU_TITLE": "das Menü"}
JA = {"ZZ_LINE_A_1": "これは最初の行です"}
CONV = b"head\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_B_2\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A_1\x00"


class TestTables(unittest.TestCase):
    def test_decode(self):
        tables = list(T.string_tables(make_resource(EN)))
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], {T.joaat(k): v for k, v in EN.items()})

    def test_bad_bounds_skipped(self):
        res = bytearray(make_resource(EN))
        at = res.index(b"\x01\x01\x00\x00") + 4
        struct.pack_into("<I", res, at, len(res) * 2)      # hash table larger than the resource
        self.assertEqual(list(T.string_tables(bytes(res))), [])

    def test_blob_chunks_named_by_the_index_entry(self):
        chunks = [{"ZZ_FIRST": "first chunk text"}, {"ZZ_SECOND": "second chunk"}, {"ZZ_THIRD": "third", "ZZ_LAST": "x"}]
        tables = list(T.string_tables(make_chunked_resource(chunks)))
        self.assertEqual(tables, [{T.joaat(k): v for c in chunks for k, v in c.items()}])

    def test_other_descriptor_entries_between_chunks_are_skipped(self):
        chunks = [{"ZZ_FIRST": "first chunk text"}, {"ZZ_SECOND": "second chunk"}]
        res = make_chunked_resource(chunks, between=[(0x956A9DF5, 0x20), (0x10075946, 0x40)])
        self.assertEqual(list(T.string_tables(res)), [{T.joaat(k): v for c in chunks for k, v in c.items()}])

    def test_one_chunk_reads_as_before(self):
        res = make_chunked_resource([EN])
        self.assertEqual(list(T.string_tables(res)), [{T.joaat(k): v for k, v in EN.items()}])
        first_blob_entry = 0x20 + 0x28
        self.assertEqual(len(T.blob_chunks(res, first_blob_entry)), 1)   # the bytes after the entry end the run

    def test_index_entry_naming_a_missing_chunk_is_skipped(self):
        res = bytearray(make_chunked_resource([{"ZZ_FIRST": "first"}, {"ZZ_SECOND": "second"}]))
        index_at = struct.unpack_from("<I", res, 0x20)[0] & 0x0FFFFFFF
        struct.pack_into("<Q", res, index_at + 16 + 8, 4 | 3)  # second string: chunk 3 of 2
        self.assertEqual(list(T.string_tables(bytes(res))), [{T.joaat("ZZ_FIRST"): "first"}])

    def test_language(self):
        self.assertEqual(T.language(EN.values()), "en")
        self.assertEqual(T.language(DE.values()), "de")
        self.assertEqual(T.language(JA.values()), "ja")
        self.assertEqual(T.language(["Привет"]), "ru")
        self.assertEqual(T.language(["안녕하세요"]), "ko")
        self.assertEqual(T.language(["你好世界"]), "zh")

    def test_group(self):
        self.assertEqual(T.table_group(["BIL1_A_1", "BIL1_B_2", "BIL1_C_3", "OTHER"]), "BIL1")
        self.assertEqual(T.table_group(["AB_CD_1", "AB_CD_2", "AB_CD_3"]), "AB_CD")
        self.assertEqual(T.table_group(["A_1", "B_2"]), "")


class TestChains(unittest.TestCase):
    def carve(self, tmp, streams):
        """rpf9_carve.py layout: index.json plus one <offset>.bin per inflated stream."""
        files, off = [], 0x1000
        for data in streams:
            write(os.path.join(tmp, f"{off:012x}.bin"), data)
            files.append({"off": off, "stored": len(data), "size": len(data), "pre": "", "head": data[:16].hex()})
            off += 0x20000
        with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"files": files, "summary": {}}, fh)

    def test_resource_across_64k_streams(self):
        res = make_resource(EN, pad=T.BLOCK)            # tables start past the first 64 KiB
        streams = [res[:T.BLOCK], res[T.BLOCK:], make_resource(DE), CONV]
        with tempfile.TemporaryDirectory() as tmp:
            self.carve(tmp, streams)
            scan = T.Scan()
            for label, off, data in T.blobs(tmp):
                scan.add(label, off, data)
        self.assertEqual(sorted(lang for _s, _o, lang, _t in scan.tables), ["de", "en"])
        self.assertEqual(scan.tables[0][1], 0x1000)

    def test_resource_across_a_128k_first_stream(self):
        # A resource's first stream can be any whole number of 64 KiB blocks (docs/page_directory.md): 41 data.rpf
        # resources start with a 128 or 192 KiB stream whose descriptors end in the next stream.
        res = make_resource(EN, pad=2 * T.BLOCK)        # tables start past the first 128 KiB
        streams = [res[:2 * T.BLOCK], res[2 * T.BLOCK:], make_resource(DE), CONV]
        with tempfile.TemporaryDirectory() as tmp:
            self.carve(tmp, streams)
            scan = T.Scan()
            for label, off, data in T.blobs(tmp):
                scan.add(label, off, data)
        self.assertEqual(sorted(lang for _s, _o, lang, _t in scan.tables), ["de", "en"])

    def test_whole_block_streams_continue_and_others_end_the_chain(self):
        for first, joined in ((T.BLOCK, True), (2 * T.BLOCK, True), (3 * T.BLOCK, True),
                              (T.BLOCK + 16, False), (T.BLOCK - 16, False)):
            with self.subTest(first=first):
                rows = [{"off": 0, "size": first}, {"off": 1, "size": 100}]
                data = {0: bytes(first), 1: b"b" * 100}
                offsets = [o for o, _d in T._chains(rows, lambda r: data[r["off"]])]
                self.assertEqual(offsets, [0] if joined else [0, 1])

    def test_short_stream_ends_the_chain(self):
        rows = [{"off": 0, "size": 100}, {"off": 1, "size": 50}]
        data = {0: b"a" * 100, 1: b"b" * 50}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])

    def test_new_resource_ends_the_chain(self):
        rows = [{"off": 0, "size": T.BLOCK}, {"off": 1, "size": 10}]
        data = {0: bytes(T.BLOCK), 1: bytes(0x10) + b"0DRP"}
        self.assertEqual([o for o, _d in T._chains(rows, lambda r: data[r["off"]])], [0, 1])


class TestDialogue(unittest.TestCase):
    def test_waves_and_text(self):
        scan = T.Scan()
        scan.add("a", 0, make_resource(EN))
        scan.add("b", 0, make_resource(DE))
        scan.add("c", 0, CONV)
        rows = list(scan.dialogue_rows())
        self.assertEqual([(r["wave"], r["label"], r["speaker"]) for r in rows],
                         [("zz_rootau_abcd_01", "ZZ_LINE_A_1", "A"), ("zz_rootau_abcd_02", "ZZ_LINE_B_2", "B")])
        self.assertEqual(rows[0]["text_en"], EN["ZZ_LINE_A_1"])
        self.assertEqual(rows[1]["text_de"], DE["ZZ_LINE_B_2"])
        self.assertEqual(scan.languages(), ["en", "de"])

    def test_lowercase_in_label(self):
        scan = T.Scan()
        scan.add("c", 0, b"ZZ_KEYAU_ABCD\x00ZZ_TIMELEFT_90s_1\x00")
        self.assertEqual(scan.pairs["ZZ_KEYAU_ABCD"], ["ZZ_TIMELEFT_90s_1"])


class TestCli(unittest.TestCase):
    def test_unreadable_subdirectory_preserves_previous_tables_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            blocked = os.path.join(source, "blocked")
            os.makedirs(blocked)
            write(os.path.join(source, "visible.bin"), make_resource(EN))
            write(os.path.join(blocked, "hidden.bin"), make_resource(DE))
            report = os.path.join(tmp, "tables.csv")
            previous = b"previous complete report\n"
            write(report, previous)
            scandir = os.scandir

            def scan(path):
                if path == blocked:
                    raise PermissionError(errno.EACCES, "Permission denied", path)
                return scandir(path)

            with (patch.object(os, "scandir", side_effect=scan),
                  contextlib.redirect_stderr(io.StringIO()) as error,
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(T.main([source, "--tables", report]), 2)

            self.assertEqual(output.getvalue(), "")
            self.assertEqual(len(error.getvalue().splitlines()), 1)
            self.assertIn("cannot read input", error.getvalue())
            self.assertIn(repr(blocked), error.getvalue())
            with open(report, "rb") as fh:
                self.assertEqual(fh.read(), previous)

    def test_csv_exports_explicitly_empty_strings(self) -> None:
        strings = {"ZZ_SYNTHETIC_BLANK": "", "ZZ_SYNTHETIC_LINE": "This is the text you know"}
        with tempfile.TemporaryDirectory() as tmp:
            source = write(os.path.join(tmp, "strings.bin"), make_resource(strings))
            tables_csv = os.path.join(tmp, "tables.csv")
            strings_csv = os.path.join(tmp, "strings.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(T.main([source, "--tables", tables_csv, "--strings", strings_csv]), 0)
            with open(tables_csv, encoding="utf-8", newline="") as f:
                self.assertEqual([row["strings"] for row in csv.DictReader(f)], ["2"])
            with open(strings_csv, encoding="utf-8", newline="") as f:
                self.assertEqual({row["hash"]: row["text"] for row in csv.DictReader(f)},
                                 {f"{T.joaat(label):08x}": text for label, text in strings.items()})

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "en.bin"), make_resource(EN))
            write(os.path.join(src, "de.bin"), make_resource(DE))
            write(os.path.join(src, "conv.pso"), CONV)
            labels = os.path.join(tmp, "labels.json")
            with open(labels, "w", encoding="utf-8") as fh:
                json.dump({f"{T.joaat('zz_menu_title'):08x}": "zz_menu_title"}, fh)
            outs = {k: os.path.join(tmp, f"{k}.csv") for k in ("dialogue", "tables", "strings")}
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(T.main([src, "--labels", labels, "--dialogue", outs["dialogue"],
                                         "--tables", outs["tables"], "--strings", outs["strings"]]), 0)
            self.assertIn("2 string tables", log.getvalue())
            self.assertIn("1 dialogue keys, 2 lines", log.getvalue())
            with open(outs["tables"], encoding="utf-8", newline="") as fh:
                tables = list(csv.DictReader(fh))
            self.assertEqual({(t["lang"], t["strings"], t["resolved"], t["group"]) for t in tables},
                             {("en", "3", "3", "ZZ_LINE"), ("de", "3", "3", "ZZ_LINE")})  # 2 of 3 share ZZ_LINE
            with open(outs["strings"], encoding="utf-8", newline="") as fh:
                strings = list(csv.DictReader(fh))
            self.assertIn(("ZZ_MENU_TITLE", "das Menü"), {(s["label"], s["text"]) for s in strings})
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging files must not be left behind")

    def test_failed_later_write_leaves_earlier_reports_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            old = "OLD GOOD REPORT\n"
            with open(dialogue, "w", encoding="utf-8") as fh:
                fh.write(old)
            tables = os.path.join(tmp, "tables.csv")
            os.mkdir(tables)                        # a later output that cannot be written
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")])

    @unittest.skipUnless(os.name == "nt", "Windows denies os.replace on a file held open by a reader")
    def test_reader_holding_a_later_report_keeps_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            dialogue = os.path.join(tmp, "dialogue.csv")
            tables = os.path.join(tmp, "tables.csv")
            old_dialogue, old_tables = "OLD DIALOGUE\n", "OLD TABLES\n"
            for path, text in ((dialogue, old_dialogue), (tables, old_tables)):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            with open(tables, "rb"):       # a reader keeps the second output open
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(T.main([src, "--dialogue", dialogue, "--tables", tables]), 2)
            self.assertIn("cannot write", err.getvalue())
            with open(dialogue, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_dialogue)
            with open(tables, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old_tables)
            self.assertFalse([f for f in os.listdir(tmp) if f.startswith(".text_tables-")],
                             "staging and backup files must not be left behind")

    def test_report_cannot_replace_the_carve_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "carve")
            os.mkdir(source)
            index = write(os.path.join(source, "index.json"), b'{"files": []}')
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as code:
                T.main([source, "--tables", index])
            self.assertEqual(code.exception.code, 2)
            self.assertIn("must not overwrite an input", err.getvalue())
            with open(index, "rb") as f:
                self.assertEqual(f.read(), b'{"files": []}')

    def test_two_outputs_naming_one_file_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "conv.pso"), CONV)
            out = os.path.join(tmp, "same.csv")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as cm:
                    T.main([src, "--dialogue", out, "--tables", out])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("distinct", err.getvalue())
            self.assertFalse(os.path.exists(out), "no report may be written before the check")

    def test_missing_input(self):
        # "/nonexistent" is a drive-root path on Windows (C:\nonexistent), which really exists on some hosts.
        # Use a path inside a fresh temp directory so the input is guaranteed to be missing on every platform.
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing-input")
            self.assertFalse(os.path.exists(missing))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(T.main([missing]), 2)

    def test_missing_parent_and_hardlinked_outputs_rejected_upfront(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "en.bin"), make_resource(EN))
            missing_parent = os.path.join(tmp, "no_such_dir", "out.csv")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                T.main([src, "--tables", missing_parent])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("parent directory does not exist", err.getvalue())

            first = write(os.path.join(tmp, "first.csv"), b"old\n")
            alias = os.path.join(tmp, "alias.csv")
            try:
                os.link(first, alias)
            except OSError:
                self.skipTest("hard links unavailable")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                T.main([src, "--dialogue", first, "--tables", alias])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("distinct", err.getvalue())

    def test_invalid_json_labels_is_clean_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "en.bin"), make_resource(EN))
            bad_labels = os.path.join(tmp, "bad.json")
            with open(bad_labels, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = T.main([src, "--labels", bad_labels])
            self.assertEqual(rc, 2)
            self.assertIn("cannot read input", err.getvalue())

    def test_wrong_shape_labels_json_is_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write(os.path.join(tmp, "en.bin"), make_resource(EN))
            for text in ("[1, 2]", '"x"', "null"):
                with self.subTest(text):
                    labels = os.path.join(tmp, "labels.json")
                    with open(labels, "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(text)
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err):
                        rc = T.main([src, "--labels", labels])
                    self.assertEqual(rc, 2)
                    self.assertIn("cannot read input", err.getvalue())

    def test_carve_folder_with_wrong_shape_index_is_clean_error(self):
        for name, text in (("list", "[]"), ("nofiles", "{}"), ("badrow", '{"files": [1]}')):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rc = T.main([tmp])
                self.assertEqual(rc, 2)
                self.assertIn("cannot read input", err.getvalue())


UNDECIDED = {"ZZ_CODE_1": "qqq zzz", "ZZ_CODE_2": "xkcd 42"}        # no script, letter or word decides: xx


def rpf9_header(data_off=0x800):
    return b"9FPR" + struct.pack("<I", data_off)


class TestNestedArchiveLanguage(unittest.TestCase):
    """A table inside a nested RPF9 archive takes the language most of that archive's tables share."""

    def test_nested_archives_are_aligned_headers_with_a_sane_data_offset(self):
        data = bytearray(0x5000)
        data[0:8] = rpf9_header(0x100)                    # the archive itself: not nested
        data[0x1000:0x1008] = rpf9_header()               # nested
        data[0x2010:0x2018] = rpf9_header()               # not 4 KiB-aligned
        data[0x3000:0x3008] = rpf9_header(0x7FFFFFFF)     # data offset past the end of the file
        data[0x4000:0x4008] = rpf9_header(4)              # data offset inside its own header
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "a.rpf"), bytes(data))
            self.assertEqual(T.nested_archives(path), [0x1000])
        self.assertEqual(T.archive_of([0x1000, 0x3000], 0x0FFF), None)
        self.assertEqual(T.archive_of([0x1000, 0x3000], 0x2000), 0x1000)
        self.assertEqual(T.archive_of([0x1000, 0x3000], 0x3000), 0x3000)

    def test_the_archive_majority_replaces_strays_and_xx(self):
        scan = T.Scan()
        en, de = make_resource(EN), make_resource(DE)
        for off, data in ((0x100, en), (0x200, en), (0x300, make_resource(UNDECIDED))):
            scan.add("s", off, data, ("s", 0x1000))       # archive A: en, en, xx
        for off, data in ((0x2100, de), (0x2200, de), (0x2300, en)):
            scan.add("s", off, data, ("s", 0x2000))       # archive B: de, de and one en stray
        scan.add("s", 0x9000, make_resource(UNDECIDED))   # outside any nested archive
        self.assertEqual(scan.archive_lang, {("s", 0x1000): "en", ("s", 0x2000): "de"})
        self.assertEqual([lang for _s, _o, lang, _t in scan.tables], ["en", "en", "en", "de", "de", "de", "xx"])
        self.assertEqual([f[2] for f in scan.found], ["en", "en", "xx", "de", "de", "en", "xx"])  # own guesses kept
        self.assertEqual(set(scan.texts), {"en", "de", "xx"})

    def test_no_majority_keeps_each_guess(self):
        scan = T.Scan()
        scan.add("s", 0x100, make_resource(EN), ("s", 0x1000))
        scan.add("s", 0x200, make_resource(DE), ("s", 0x1000))
        scan.add("s", 0x300, make_resource(UNDECIDED), ("s", 0x1000))  # xx does not vote
        self.assertEqual(scan.archive_lang, {})
        self.assertEqual([lang for _s, _o, lang, _t in scan.tables], ["en", "de", "xx"])

    def test_carve_directory_finds_the_archive_from_its_index(self):
        """End to end: an rpf9_carve.py directory whose index.json names the archive on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            arc = bytearray(0x4000)
            arc[0:8] = rpf9_header(0x40)
            arc[0x1000:0x1008] = rpf9_header()
            arc[0x2000:0x2008] = rpf9_header()
            arc_path = write(os.path.join(tmp, "pack.rpf"), bytes(arc))
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            rows = []
            for off, strings in ((0x1100, EN), (0x1200, UNDECIDED), (0x1300, EN),
                                 (0x2100, DE), (0x2200, DE), (0x2300, UNDECIDED)):
                data = make_resource(strings)
                write(os.path.join(carve, f"{off:012x}.bin"), data)
                rows.append({"off": off, "stored": len(data), "size": len(data), "pre": "", "head": data[:16].hex()})
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8") as fh:
                json.dump({"summary": {"archive": arc_path}, "files": rows}, fh)
            out = os.path.join(tmp, "tables.csv")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(T.main([carve, "--tables", out]), 0)
            self.assertIn("language from the nested archive: 6 tables in 2 archives; 2 differ", log.getvalue())
            with open(out, encoding="utf-8", newline="") as fh:
                got = [(r["offset"], r["lang"], r["guess"], r["archive"]) for r in csv.DictReader(fh)]
            self.assertEqual(got, [("4352", "en", "en", "0x1000"), ("4608", "en", "xx", "0x1000"),
                                   ("4864", "en", "en", "0x1000"), ("8448", "de", "de", "0x2000"),
                                   ("8704", "de", "de", "0x2000"), ("8960", "de", "xx", "0x2000")])

    def test_without_the_archive_each_table_keeps_its_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            write(os.path.join(src, "a.bin"), make_resource(UNDECIDED))
            out = os.path.join(tmp, "tables.csv")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(T.main([src, "--tables", out]), 0)
            self.assertNotIn("nested archive", log.getvalue())
            with open(out, encoding="utf-8", newline="") as fh:
                self.assertEqual([(r["lang"], r["guess"], r["archive"]) for r in csv.DictReader(fh)],
                                 [("xx", "xx", "")])


if __name__ == "__main__":
    unittest.main()
