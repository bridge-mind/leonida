"""The FSSM pixel-input decoder reads documented record fields, the directory identity and the fp16 mask.

All blobs here are built by hand; no shader from the game is used.

    python -I tests/test_fssm_header.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "fssm_header.py")
sys.path.insert(0, TOOLS)

import fssm_header  # noqa: E402


def blob(records=(), directory=(0x100, 0x20, 0x120), size=0x200, first=None, wave_flags=0):
    """Build a synthetic FSSM header.  records are complete 8-byte input records."""
    data = bytearray(size)
    data[0:4] = b"FSSM"
    if size >= 0xD8:
        struct.pack_into("<I", data, 0xD4, wave_flags)
    struct.pack_into("<I", data, 0x120, directory[0])
    struct.pack_into("<I", data, 0x12C, directory[1])
    struct.pack_into("<I", data, 0x130, directory[2])
    for i, record in enumerate(records):
        start = 0x138 + 8 * i
        data[start:start + 8] = bytes(record)
    if first is not None:
        data[0x138] = first
    return bytes(data)


def record(semantic, mask, index, slot, flags, kind=6):
    return (kind, semantic, mask, index, 0, 0, slot, flags)


def tail_blob(records=(), name=b"shader", header=3, declared_end=None, tail_at=0x120,
              filler=0xBF9F0000, pad_word=0, name_length=None):
    """Build an FSSM blob with a post-code binding table at tail_at."""
    data = bytearray(0x400)
    data[0:4] = b"FSSM"
    struct.pack_into("<Q", data, 0xC0, 0x100)
    struct.pack_into("<Q", data, 0xC8, 0x20)
    struct.pack_into("<Q", data, 0xE8, tail_at if declared_end is None else declared_end)
    for i in range(4):
        struct.pack_into("<I", data, tail_at + 4 * i, filler)
    struct.pack_into("<I", data, tail_at + 16, header)
    offset = tail_at + 24
    for words in records:
        struct.pack_into("<5I", data, offset, *words)
        offset += 20
    struct.pack_into("<I", data, offset, 0xFFFFFFFF)
    struct.pack_into("<I", data, offset + 4, 0)
    struct.pack_into("<H", data, offset + 8, pad_word)
    struct.pack_into("<H", data, offset + 10, len(name) if name_length is None else name_length)
    data[offset + 12:offset + 12 + len(name)] = name
    return bytes(data)


class PixelInputTest(unittest.TestCase):
    def test_two_records_keep_their_fields(self):
        b = blob([record(4, 0x7, 1, 9, 0x12), record(7, 0x3, 2, 10, 0x08)])
        self.assertEqual(fssm_header.pixel_inputs(b), [
            {"semantic": 4, "mask": 0x7, "index": 1, "slot": 9, "flags": 0x12},
            {"semantic": 7, "mask": 0x3, "index": 2, "slot": 10, "flags": 0x08},
        ])

    def test_table_stops_at_a_non_six_kind(self):
        b = blob([record(1, 0xF, 0, 3, 0), record(2, 0xF, 1, 4, 0, kind=5)])
        self.assertEqual(len(fssm_header.pixel_inputs(b)), 1)

    def test_no_table_when_first_byte_is_not_six(self):
        b = blob([record(1, 0xF, 0, 3, 0)], first=5)
        self.assertEqual(fssm_header.pixel_inputs(b), [])
        self.assertEqual(fssm_header.interpolants(b), (0, 0))

    def test_short_blob_has_no_records(self):
        self.assertEqual(fssm_header.pixel_inputs(b""), [])
        self.assertEqual(fssm_header.pixel_inputs(b"\x06" * 0x139), [])

    def test_fp16_mask_uses_table_position(self):
        b = blob([record(1, 0xF, 0, 0, 0x10),
                  record(2, 0xF, 1, 1, 0x00),
                  record(3, 0x3, 2, 2, 0x10)])
        self.assertEqual(fssm_header.interpolants(b), (3, 0b101))

    def test_fp16_mask_is_empty_without_the_flag(self):
        b = blob([record(1, 0xF, 0, 0, 0x02), record(2, 0xF, 1, 1, 0x08)])
        self.assertEqual(fssm_header.interpolants(b), (2, 0))


class ReadMethodTest(unittest.TestCase):
    """Flags bits 0-1, the read method: 1 = v_interp_mov (flat), 2 = v_interp_p1/p2 (interpolated); SHADER-181."""

    def test_read_methods_by_table_position(self):
        b = blob([record(1, 0xF, 0, 0, 0x02),
                  record(2, 0xF, 1, 1, 0x01),
                  record(3, 0x3, 2, 2, 0x1A),   # the corpus' 16-bit records: fp16 | bit 3 | linear
                  record(4, 0x1, 3, 3, 0x09)])
        self.assertEqual(fssm_header.read_methods(b), [2, 1, 2, 1])

    def test_read_bits_do_not_touch_the_fp16_mask(self):
        b = blob([record(1, 0xF, 0, 0, 0x03), record(2, 0xF, 1, 1, 0x13)])
        self.assertEqual(fssm_header.read_methods(b), [3, 3])
        self.assertEqual(fssm_header.interpolants(b), (2, 0b10))

    def test_no_table_no_read_methods(self):
        self.assertEqual(fssm_header.read_methods(blob(first=0)), [])

    def test_names(self):
        self.assertEqual([fssm_header.read_name(m) for m in range(4)], ["read0", "mov", "interp", "read3"])

    def test_tally_counts_ps_read_methods_only(self):
        rows = [{"stage": "ps", "directory_ok": True, "entries": 2, "fp16_mask": 0,
                 "inputs": [{"flags": 0x02}, {"flags": 0x00}]},
                {"stage": "vs", "directory_ok": True, "entries": 1, "fp16_mask": 0, "inputs": [{"flags": 0x01}]}]
        self.assertEqual(fssm_header.tally(rows, True)["read_methods"], {"interp": 1, "read0": 1})


class DirectoryTest(unittest.TestCase):
    def test_identity_holds(self):
        self.assertTrue(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0x120))))

    def test_identity_breaks(self):
        self.assertFalse(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0x121))))
        self.assertFalse(fssm_header.directory_ok(blob(directory=(0x100, 0x20, 0xDEADBEEF))))

    def test_short_blob_is_not_ok(self):
        self.assertFalse(fssm_header.directory_ok(b"FSSM"))


class WaveSizeTest(unittest.TestCase):
    def test_wave64_when_bit5_clear(self):
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0)), 64)
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0x00110000)), 64)
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0x040C0004)), 64)
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0x000F0001)), 64)

    def test_wave32_when_bit5_set(self):
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=fssm_header.FLAG_WAVE32)), 32)
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0x00110020)), 32)
        self.assertEqual(fssm_header.wave_size(blob(wave_flags=0x000C0020)), 32)

    def test_exact_minimum_length(self):
        data = bytearray(0xD8)
        data[0:4] = b"FSSM"
        struct.pack_into("<I", data, 0xD4, 0x20)
        self.assertEqual(fssm_header.wave_size(bytes(data)), 32)
        struct.pack_into("<I", data, 0xD4, 0)
        self.assertEqual(fssm_header.wave_size(bytes(data)), 64)
        self.assertIsNone(fssm_header.wave_size(bytes(data[:0xD7])))

    def test_short_or_non_fssm_blob_returns_none(self):
        self.assertIsNone(fssm_header.wave_size(b""))
        self.assertIsNone(fssm_header.wave_size(b"FSSM"))
        # Zero-prefixed synthetic compute headers lack the FSSM magic and must return None.
        self.assertIsNone(fssm_header.wave_size(bytes(0x148)))
        bad_magic = bytearray(0x100)
        bad_magic[0:4] = b"NOPE"
        struct.pack_into("<I", bad_magic, 0xD4, 0x20)
        self.assertIsNone(fssm_header.wave_size(bytes(bad_magic)))

    def test_tally_counts_wave_sizes(self):
        rows = [
            {"stage": "cs", "directory_ok": True, "wave_size": 32, "entries": 0, "fp16_mask": 0, "inputs": []},
            {"stage": "ps", "directory_ok": True, "wave_size": 64, "entries": 0, "fp16_mask": 0, "inputs": []},
            {"stage": "vs", "directory_ok": True, "wave_size": 64, "entries": 0, "fp16_mask": 0, "inputs": []},
            {"stage": "cs", "directory_ok": False, "wave_size": None, "entries": 0, "fp16_mask": 0, "inputs": []},
        ]
        self.assertEqual(fssm_header.tally(rows, True)["wave_sizes"], {"wave32": 1, "wave64": 2})


class BindingTableTest(unittest.TestCase):
    def test_records_header_and_name(self):
        table = fssm_header.binding_table(
            tail_blob([(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)], b"shader_ps"))
        self.assertEqual(table["code_end"], 0x120)
        self.assertEqual(table["header"], 3)
        self.assertEqual(table["records"], [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)])
        self.assertEqual(table["name"], "shader_ps")
        self.assertEqual(table["trailing"], 0x400 - (0x120 + 24 + 40 + 12 + 9))

    def test_empty_table_still_has_a_name(self):
        table = fssm_header.binding_table(tail_blob(name=b"x"))
        self.assertEqual(table["records"], [])
        self.assertEqual(table["name"], "x")

    def test_unset_code_end_falls_back_to_code_size(self):
        table = fssm_header.binding_table(
            tail_blob([(1, 2, 3, 4, 5)], b"x", declared_end=0xDEADBEEFDEADBEEF))
        self.assertEqual(table["code_end"], 0x120)
        self.assertEqual(table["name"], "x")

    def test_missing_filler_is_rejected(self):
        self.assertIsNone(fssm_header.binding_table(tail_blob(filler=0)))

    def test_nonzero_pad_after_sentinel_is_rejected(self):
        self.assertIsNone(fssm_header.binding_table(tail_blob(pad_word=1)))

    def test_non_printable_name_is_rejected(self):
        self.assertIsNone(fssm_header.binding_table(tail_blob(name_length=0x100)))

    def test_name_past_end_is_rejected(self):
        self.assertIsNone(fssm_header.binding_table(tail_blob(name_length=0x400)))

    def test_missing_sentinel_is_rejected(self):
        data = bytearray(tail_blob())
        struct.pack_into("<I", data, 0x120 + 24, 1)
        self.assertIsNone(fssm_header.binding_table(bytes(data)))

    def test_short_blob_is_rejected(self):
        self.assertIsNone(fssm_header.binding_table(b"FSSM"))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        fssm_dir = os.path.join(self.dir, "fssm")
        os.makedirs(fssm_dir)
        with open(os.path.join(fssm_dir, "a.fssm"), "wb") as f:
            f.write(blob([record(1, 0xF, 0, 0, 0x12), record(2, 0x3, 1, 1, 0x01)]))
        with open(os.path.join(fssm_dir, "b.fssm"), "wb") as f:
            f.write(blob(first=0))
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": "a.dxbc", "stage": "ps"}, {"file": "b.dxbc", "stage": "ps"}], f)
        self.fssm_dir = fssm_dir

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_path_is_one_line(self):
        missing = os.path.join(self.dir, "missing")
        p = subprocess.run([sys.executable, "-I", TOOL, missing], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn(missing, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stdout, "")

    def test_limit_counts_from_the_start(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json", "--limit", "1"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([row["name"] for row in json.loads(p.stdout)["files"]], ["a.fssm"])
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json", "--limit", "0x1"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([row["name"] for row in json.loads(p.stdout)["files"]], ["a.fssm"])
        # A negative limit sliced from the end (paths[:-1] dropped b.fssm) instead of being refused.
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--limit", "-1"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("must be 0 or more", p.stderr)
        self.assertEqual(p.stdout, "")
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--limit", "nope"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("invalid int value: 'nope'", p.stderr)
        self.assertNotIn("non_negative", p.stderr)

    def test_ignores_fssm_subdirectories_and_reports_read_errors_cleanly(self):
        os.makedirs(os.path.join(self.fssm_dir, "nested.fssm"))
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([row["name"] for row in json.loads(p.stdout)["files"]], ["a.fssm", "b.fssm"])

        err = PermissionError(13, "Permission denied", os.path.join(self.fssm_dir, "a.fssm"))
        with mock.patch("fssm_header.summarize", side_effect=err):
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
                orig_stderr = sys.stderr
                try:
                    sys.stderr = stderr_file
                    rc = fssm_header.main([self.fssm_dir])
                finally:
                    sys.stderr = orig_stderr
                stderr_file.seek(0)
                err_out = stderr_file.read()
        self.assertEqual(rc, 2)
        self.assertIn("ERROR: cannot read", err_out)

    def test_text_tally(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("2 files, directory_ok 2, PS with table 1, entries 2", p.stdout)
        self.assertIn("a.fssm: stage=ps directory_ok=True entries=2 fp16=0x1", p.stdout)
        self.assertIn("read methods: interp:1, mov:1", p.stdout)

    def test_json_tally(self):
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)
        self.assertEqual(doc["tally"]["ps_with_table"], 1)
        self.assertEqual(doc["tally"]["entries"], 2)
        self.assertEqual(doc["tally"]["read_methods"], {"interp": 1, "mov": 1})
        self.assertEqual(doc["tally"]["wave_sizes"], {"wave64": 2})

    def test_wave_size_in_text_and_json_cli(self):
        with open(os.path.join(self.fssm_dir, "b.fssm"), "wb") as f:
            f.write(blob(first=0, wave_flags=fssm_header.FLAG_WAVE32))
        p_txt = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir], capture_output=True, text=True)
        self.assertEqual(p_txt.returncode, 0, p_txt.stderr)
        self.assertIn("a.fssm: stage=ps directory_ok=True entries=2 fp16=0x1 wave=64", p_txt.stdout)
        self.assertIn("b.fssm: stage=ps directory_ok=True entries=0 fp16=0x0 wave=32", p_txt.stdout)
        self.assertIn("wave sizes: wave32:1, wave64:1", p_txt.stdout)

        p_json = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json"],
                                capture_output=True, text=True)
        self.assertEqual(p_json.returncode, 0, p_json.stderr)
        doc = json.loads(p_json.stdout)
        self.assertEqual({r["name"]: r["wave_size"] for r in doc["files"]}, {"a.fssm": 64, "b.fssm": 32})
        self.assertEqual(doc["tally"]["wave_sizes"], {"wave32": 1, "wave64": 1})

    def test_table_only_entry_counts_exclude_blobs_without_a_table(self):
        os.remove(os.path.join(self.dir, "index.json"))
        p = subprocess.run([sys.executable, "-I", TOOL, self.fssm_dir, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        tally = json.loads(p.stdout)["tally"]
        self.assertEqual(tally["stage_source"], "table-only")
        self.assertEqual(tally["ps_with_table"], 1)
        self.assertEqual(tally["entries"], 2)
        self.assertEqual(tally["entry_counts"], {"2": 1})
        self.assertEqual(sum(tally["entry_counts"].values()), tally["ps_with_table"])

    def test_table_only_tally_with_only_non_ps_blobs_has_an_empty_histogram(self):
        rows = [{"stage": None, "directory_ok": True, "entries": 0, "fp16_mask": 0, "inputs": []}]
        tally = fssm_header.tally(rows, False)
        self.assertEqual(tally["entry_counts"], {})
        self.assertEqual(tally["entries"], 0)

    def test_local_stage_index_works_with_an_unrelated_parent_index(self):
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"unrelated": []}, f)
        with open(os.path.join(self.fssm_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": "a.dxbc", "stage": "ps"}], f)
        self.assertEqual(fssm_header.stage_index(self.fssm_dir), {"a": "ps"})

    def test_parent_stage_index_is_used_when_local_index_is_empty_or_unrelated(self):
        for data in (
            {"unrelated": []},
            [],
            {},
            [{"file": "a.bin"}],
            [{"file": "a.dxbc", "stage": None}],
            [{"file": "a.dxbc", "stage": ""}],
        ):
            with self.subTest(data=data):
                with open(os.path.join(self.fssm_dir, "index.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f)
                self.assertEqual(fssm_header.stage_index(self.fssm_dir), {"a": "ps", "b": "ps"})

    def test_stage_index_skips_malformed_or_conflicting_rows_and_normalizes_subpaths(self):
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump([
                {"missing_file": True, "stage": "ps"},
                {"file": None, "stage": "ps"},
                {"file": "a.dxbc", "stage": None},
                {"file": "dxbc/a.dxbc", "stage": "ps"},
                {"file": "other/a.dxbc", "stage": "ps"},
                {"file": "dxbc\\b.dxbc", "stage": "vs"},
                {"file": "x/conflict.dxbc", "stage": "vs"},
                {"file": "y/conflict.dxbc", "stage": "ps"},
            ], f)
        self.assertEqual(fssm_header.stage_index(self.fssm_dir), {"a": "ps", "b": "vs"})

    def test_unindexed_blob_falls_back_to_table_only_tally(self):
        unlisted = os.path.join(self.fssm_dir, "unlisted.fssm")
        with open(unlisted, "wb") as f:
            f.write(blob([record(1, 0xF, 0, 0, 0x12), record(2, 0x3, 1, 1, 0x01)]))
        p = subprocess.run([sys.executable, "-I", TOOL, unlisted, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        tally = json.loads(p.stdout)["tally"]
        self.assertEqual(tally["stage_source"], "table-only")
        self.assertEqual(tally["ps_with_table"], 1)
        self.assertEqual(tally["entries"], 2)
        self.assertEqual(tally["entry_counts"], {"2": 1})
        self.assertEqual(tally["read_methods"], {"interp": 1, "mov": 1})
        p_txt = subprocess.run([sys.executable, "-I", TOOL, unlisted],
                               capture_output=True, text=True)
        self.assertEqual(p_txt.returncode, 0, p_txt.stderr)
        self.assertIn("(no inspected blob in index.json: counted every blob with a non-empty table as a PS)",
                      p_txt.stdout)



class BindingCliTest(unittest.TestCase):
    def test_bindings_flag_reports_the_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            fssm_dir = os.path.join(tmp, "fssm")
            os.makedirs(fssm_dir)
            with open(os.path.join(fssm_dir, "c.fssm"), "wb") as f:
                f.write(tail_blob([(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)], b"x"))
            p = subprocess.run([sys.executable, "-I", TOOL, fssm_dir, "--bindings"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("bindings header=3 records=2 name_len=1 trailing=", p.stdout)
            self.assertIn("binding tables 1, records 2, tail name equals file stem 0", p.stdout)

    def test_bindings_in_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            fssm_dir = os.path.join(tmp, "fssm")
            os.makedirs(fssm_dir)
            with open(os.path.join(fssm_dir, "c.fssm"), "wb") as f:
                f.write(tail_blob([(1, 2, 3, 4, 5)], b"x"))
            p = subprocess.run([sys.executable, "-I", TOOL, fssm_dir, "--json", "--bindings"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            self.assertEqual(doc["files"][0]["bindings"]["records"], 1)
            self.assertEqual(doc["files"][0]["bindings"]["name"], "x")


if __name__ == "__main__":
    unittest.main()
