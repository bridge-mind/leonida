"""Synthetic-data tests for tools/fragment_children.py (ASSET-261, the fragment child-drawable census).

    python -I tests/test_fragment_children.py

Every byte here is built by the test: synthetic fragment system pages inside a synthetic RPF9-style archive
made of page-aligned zlib streams. No game file, value or name is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "fragment_children.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fragment_children as fc  # noqa: E402

S = 0x50000000


def drawable(b, at, shader=S + 0x10, lod=True):
    """A minimal drawable record at `at` that passes the tool's body test (and the decoder's header test)."""
    struct.pack_into("<Q", b, at + 8, 0)
    struct.pack_into("<Q", b, at + 16, shader)
    struct.pack_into("<f", b, at + 44, 2.0)
    struct.pack_into("<3f", b, at + 48, -1.0, -1.0, -1.0)
    struct.pack_into("<3f", b, at + 64, 1.0, 1.0, 1.0)
    struct.pack_into("<Q", b, at + 80, S + 0x880 if lod else S + 0xF00)   # a LOD list: one model pointer
    struct.pack_into("<Q", b, 0x880, S + 0x8A0)
    struct.pack_into("<HH", b, 0x888, 1, 1)
    struct.pack_into("<Q", b, 0x8A0, S + 0x900)
    struct.pack_into("<HHH", b, 0x900 + 16, 1, 1, 1)             # the model's geometry count, three copies
    struct.pack_into("<H", b, 0x900 + 54, 1)


MAIN = 0x200
RECORD = 0x110


def fragment(size=0x2000, drawables=1, strays=(), null_optional=False, elements=0, abcd=None, broken=()):
    """A fragment system page: header at 0, a drawable record array at 0x200 (stride 0x110), strays elsewhere.

    `broken` lists record indices whose LOD list is missing, so the array walk rejects them.
    """
    b = bytearray(size)
    struct.pack_into("<Q", b, 8, S + 0x100)           # page directory
    b[0x100 + 8], b[0x100 + 9] = 1, 1                 # one system page, one graphics page
    struct.pack_into("<Q", b, 24, S + MAIN)           # main drawable
    struct.pack_into("<Q", b, 168, S + 0x180)         # name
    if not null_optional:
        for at in (16, 32, 40):
            struct.pack_into("<Q", b, at, S + 0x800)
    for i, value in enumerate(abcd or (drawables, 1, 1, 1)):
        struct.pack_into("<H", b, 48 + 2 * i, value)
    for i in range(drawables):
        drawable(b, MAIN + RECORD * i, lod=i not in broken)
    for at in strays:
        drawable(b, at)
    if elements:
        struct.pack_into("<Q", b, 40, S + 0xC00)
        struct.pack_into("<HH", b, 52, elements, elements)
    return bytes(b)


def archive(*resources, page_gap=0x1000):
    """One zlib stream per resource, each at a page boundary + 16 and preceded by 16 zero bytes."""
    out = bytearray(page_gap)
    for data in resources:
        at = len(out) + 16
        out += b"\0" * 16 + zlib.compress(data, 9)
        out += b"\0" * (-len(out) % 0x1000)
    out += b"\0" * 0x1000
    return bytes(out)


class TestFragmentChildren(unittest.TestCase):
    def scan(self, blob, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.rpf")
            with open(path, "wb") as fh:
                fh.write(blob)
            return fc.scan_archive(path, **kwargs)

    def test_single_drawable_fragment_has_no_children(self):
        stats = self.scan(archive(fragment()))
        self.assertEqual(stats["fragments"], 1)
        self.assertEqual(stats["main_drawable_invalid"], 0)
        self.assertEqual(stats["multi_drawable_fragments"], 0)
        self.assertEqual(stats["child_drawables"], 0)
        self.assertEqual(stats["child_arrays"], 0)
        self.assertEqual(stats["stray_drawables"], 0)

    def test_declared_children_are_walked_at_the_record_stride(self):
        stats = self.scan(archive(fragment(drawables=3)))
        self.assertEqual(stats["multi_drawable_fragments"], 1)
        self.assertEqual(stats["declared_child_drawables"], 2)
        self.assertEqual(stats["child_drawables"], 2)
        self.assertEqual(stats["child_arrays"], 1)
        self.assertEqual(stats["max_children"], 2)
        self.assertEqual(stats["stray_drawables"], 0)

    def test_declared_child_without_a_body_is_not_found(self):
        stats = self.scan(archive(fragment(drawables=3, broken=(2,))))
        self.assertEqual(stats["declared_child_drawables"], 2)
        self.assertEqual(stats["child_drawables"], 1)

    def test_undeclared_drawable_is_a_stray(self):
        stats = self.scan(archive(fragment(strays=(0x1000,))))
        self.assertEqual(stats["child_drawables"], 0)
        self.assertEqual(stats["stray_drawables"], 1)
        self.assertEqual(stats["fragments_with_stray"], 1)

    def test_stray_at_the_last_aligned_offset_is_found(self):
        # 0x1f50 + 176 == 0x2000: the body ends exactly at the end of the page, with or without trailing padding
        for padding in (0, 16):
            with self.subTest(padding=padding):
                stats = self.scan(archive(fragment(strays=(0x1f50,)) + bytes(padding)))
                self.assertEqual(stats["stray_drawables"], 1)

    def test_null_shader_group_is_still_a_body(self):
        b = bytearray(fragment(drawables=2))
        struct.pack_into("<Q", b, MAIN + RECORD + 16, 0)
        stats = self.scan(archive(bytes(b)))
        self.assertEqual(stats["child_drawables"], 1)

    def test_pointer_table_over_zeros_is_not_a_drawable(self):
        b = bytearray(fragment(drawables=2))
        struct.pack_into("<f", b, MAIN + RECORD + 44, 0.0)
        struct.pack_into("<Q", b, MAIN + RECORD + 80, 0)        # no LOD list
        stats = self.scan(archive(bytes(b)))
        self.assertEqual(stats["child_drawables"], 0)
        self.assertEqual(stats["declared_child_drawables"], 1)

    def test_non_finite_bounds_are_not_a_body(self):
        b = bytearray(fragment(drawables=2))
        struct.pack_into("<f", b, MAIN + RECORD + 48, float("nan"))
        self.assertEqual(self.scan(archive(bytes(b)))["child_drawables"], 0)

    def test_model_with_mismatched_geometry_counts_is_not_a_body(self):
        b = bytearray(fragment(drawables=2))
        struct.pack_into("<H", b, 0x900 + 18, 2)                 # the copies of the geometry count disagree
        self.assertEqual(self.scan(archive(bytes(b)))["child_drawables"], 0)

    def test_nonzero_second_header_word_is_not_a_body(self):
        b = bytearray(fragment(drawables=2))
        struct.pack_into("<Q", b, MAIN + RECORD + 8, S + 0x100)  # a top-level drawable, not an array record
        stats = self.scan(archive(bytes(b)))
        self.assertEqual(stats["child_drawables"], 0)

    def test_null_optional_pointers_are_still_a_fragment(self):
        stats = self.scan(archive(fragment(null_optional=True)))
        self.assertEqual(stats["fragments"], 1)
        self.assertEqual(stats["optional_pointers_null"], 1)

    def test_reversed_count_fields_are_not_a_fragment(self):
        stats = self.scan(archive(fragment(abcd=(1, 1, 9, 3))))
        self.assertEqual(stats["fragments"], 0)

    def test_a_plain_drawable_is_not_a_fragment(self):
        b = bytearray(0x2000)
        struct.pack_into("<Q", b, 8, S + 0x100)
        b[0x108], b[0x109] = 1, 1
        drawable(b, 0)
        struct.pack_into("<Q", b, 8, S + 0x100)                  # a top-level drawable's +0x08 is a pointer
        struct.pack_into("<Q", b, 168, S + 0x180)
        stats = self.scan(archive(bytes(b)))
        self.assertEqual(stats["fragments"], 0)

    def test_part_elements(self):
        stats = self.scan(archive(fragment(elements=4), fragment(elements=1)))
        self.assertEqual(stats["part_elements"], 5)
        self.assertEqual(stats["multi_part_fragments"], 1)

    def test_directory_past_the_first_page(self):
        b = bytearray(fragment(size=0x4000))
        struct.pack_into("<Q", b, 8, S + 0x2100)       # the directory sits in the second system page
        b[0x2100 + 8], b[0x2100 + 9] = 2, 1
        page0, page1 = bytes(b[:0x2000]), bytes(b[0x2000:])
        stats = self.scan(archive(page0, page1, b"\x01" * 64))
        self.assertEqual(stats["fragments"], 1)
        self.assertEqual(stats["unreadable"], 0)

    def test_two_fragments_and_a_range_split_agree(self):
        blob = archive(fragment(), fragment(drawables=3), fragment(null_optional=True))
        whole = self.scan(blob)
        self.assertEqual(whole["fragments"], 3)
        self.assertEqual(whole["child_drawables"], 2)
        self.assertEqual(self.scan(blob, jobs=2), whole)

    def test_start_skips_earlier_fragments(self):
        blob = archive(fragment(), fragment(drawables=3))
        later = self.scan(blob, start=0x2000)
        self.assertEqual(later["fragments"], 1)
        self.assertEqual(later["child_drawables"], 2)

    def test_empty_archive(self):
        self.assertEqual(self.scan(b"")["fragments"], 0)


class TestCommandLine(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_json_and_records_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.rpf")
            with open(path, "wb") as fh:
                fh.write(archive(fragment(drawables=3), fragment()))
            totals, records = os.path.join(tmp, "t.json"), os.path.join(tmp, "r.jsonl")
            result = self.run_tool(path, "--json", totals, "--records", records, "--jobs", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(totals, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual((data["fragments"], data["child_drawables"], data["child_arrays"]), (2, 2, 1))
            with open(records, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh]
            self.assertEqual([line["children"] for line in lines], [2, 0])

    def test_missing_input_and_output_over_input_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_tool(os.path.join(tmp, "nope.rpf")).returncode, 2)
            path = os.path.join(tmp, "a.rpf")
            with open(path, "wb") as fh:
                fh.write(archive(fragment()))
            self.assertEqual(self.run_tool(path, "--json", path).returncode, 2)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), archive(fragment()))

    def test_output_in_a_missing_folder_is_refused_before_the_scan(self):
        # a typo in the folder used to raise at write time, after a scan that can run for hours
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.rpf")
            with open(path, "wb") as fh:
                fh.write(archive(fragment(drawables=3)))
            for flag in ("--json", "--records"):
                with self.subTest(flag=flag):
                    out = os.path.join(tmp, "no-such-folder", "out")
                    result = self.run_tool(path, flag, out)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("no-such-folder", result.stderr)
                    self.assertNotIn("a.rpf:", result.stdout)   # the scan did not start
                    self.assertFalse(os.path.exists(os.path.dirname(out)))


if __name__ == "__main__":
    unittest.main()
