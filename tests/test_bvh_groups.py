"""Hand-built descriptor and BVH streams for tools/bvh_groups.py. No game data is read.

    python -I tests/test_bvh_groups.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bvh_groups  # noqa: E402
import bvh_index  # noqa: E402


def a16(n):
    return (n + 15) & ~15


def blas(n, pad=0):
    """A BVH4 stream wrapper: 0x100 bytes, the 64-byte header at 0x100, n nodes, then `pad` zero bytes."""
    end = 0x140 + 64 * n
    b = bytearray(end + pad)
    struct.pack_into("<I", b, 0x20, end)
    struct.pack_into("<I", b, 0x28, end - 0x100)
    struct.pack_into("<II", b, 0x100, bvh_index.TAG, n)
    struct.pack_into("<6f", b, 0x110, -1, -1, -1, 1, 1, 1)
    return bytes(b)


def buffer_for(node_counts):
    """The BLAS of `node_counts` back to back, each padded to 256 bytes; returns (bytes, entry sizes)."""
    out, sizes = b"", []
    for n in node_counts:
        end = 0x140 + 64 * n
        size = (end + 255) & ~255
        out += blas(n, size - end)
        sizes.append(size)
    return out, sizes


def instance_desc(entry_sizes, map_first=True, high=0x0B000340):
    """Instance template: transforms at 0x50, then map/entries/two arrays, each region padded to 16 bytes."""
    n = len(entry_sizes)
    xf = 0x50
    order = ["map", "ent", "a28", "a30"] if map_first else ["ent", "a28", "a30", "map"]
    lengths = {"map": 32, "ent": 8 * n, "a28": 8 * n, "a30": 8 * n}
    pos, at = xf + 48 * n, {}
    for name in order:
        at[name] = pos
        pos += a16(lengths[name])
    b = bytearray(pos)
    struct.pack_into("<Q", b, 0x08, 0x50000000 | at["map"])
    struct.pack_into("<Q", b, 0x10, n)
    struct.pack_into("<Q", b, 0x18, 0x50000000 | at["ent"])
    struct.pack_into("<Q", b, 0x20, 0x60000000)
    struct.pack_into("<Q", b, 0x28, 0x50000000 | at["a28"])
    struct.pack_into("<Q", b, 0x30, 0x50000000 | at["a30"])
    struct.pack_into("<Q", b, 0x38, 0x50000000 | xf)
    struct.pack_into("<Q", b, 0x40, 0x1111222233334444)
    for i in range(n):
        struct.pack_into("<12f", b, xf + 48 * i, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        e = entry_sizes[i]
        struct.pack_into("<Q", b, at["ent"] + 8 * i, (high << 32 | e) if e else 0)
    struct.pack_into("<I", b, at["map"] + 8, 0x101)
    return bytes(b)


def table_desc(values, items_first=True):
    """Table template: K items of 32 bytes, the map, K pointers and K u32 values, each region padded to 16."""
    k = len(values)
    lengths = {"items": 32 * k, "map": 32, "ptrs": 8 * k, "vals": 4 * k}
    order = ["items", "map", "ptrs", "vals"] if items_first else ["ptrs", "items", "map", "vals"]
    pos, at = 0x40, {}
    for name in order:
        at[name] = pos
        pos += a16(lengths[name])
    b = bytearray(pos)
    struct.pack_into("<Q", b, 0x08, 0x50000000 | at["map"])
    struct.pack_into("<Q", b, 0x18, 1)
    struct.pack_into("<Q", b, 0x20, 0x50000000 | at["vals"])
    struct.pack_into("<HH", b, 0x28, k, k)
    struct.pack_into("<Q", b, 0x30, 0x50000000 | at["ptrs"])
    struct.pack_into("<HH", b, 0x38, k, k)
    struct.pack_into("<I", b, at["map"] + 8, 1)
    for i, v in enumerate(values):
        struct.pack_into("<I", b, at["vals"] + 4 * i, v)
        struct.pack_into("<Q", b, at["ptrs"] + 8 * i, 0x50000000 | (at["items"] + 32 * i))
        struct.pack_into("<Q", b, at["items"] + 32 * i + 0x10, 0x0102030405060708 + i)
    return bytes(b)


def make_rpf(payloads):
    body = b""
    for p in payloads:
        body += zlib.compress(p, 9) + b"\0" * 16
    head = bytearray(0x40)
    head[0:4] = b"9FPR"
    struct.pack_into("<I", head, 4, len(head))
    return bytes(head) + body


class ParseTest(unittest.TestCase):
    def test_single_entry_instance_is_the_208_byte_record(self):
        buf, sizes = buffer_for([20])
        d = instance_desc(sizes)
        self.assertEqual(len(d), 208)
        self.assertEqual(bvh_index.classify(d, len(d))["kind"], "instance208")
        p = bvh_groups.parse_descriptor(d)
        self.assertEqual((p["family"], p["count"], p["page_word"], p["matrix_offset"]), ("instance", 1, 0x101, 0x50))
        self.assertTrue(p["tiles"] and p["arrays_zero"] and p["zero_0x48"] and p["entry_high_low16_mod64"])
        self.assertEqual(p["entry_sizes"], sizes)
        self.assertNotIn(0x1111222233334444, p.values())

    def test_multi_entry_instance_with_zero_slot(self):
        _, sizes = buffer_for([3, 40])
        d = instance_desc([sizes[0], 0, sizes[1]], map_first=False)
        self.assertEqual(bvh_index.classify(d, len(d))["kind"], "descriptor")
        p = bvh_groups.parse_descriptor(d)
        self.assertEqual((p["count"], p["zero_entries"]), (3, 1))
        self.assertTrue(p["tiles"])

    def test_instance_layout_gap_does_not_tile(self):
        d = bytearray(instance_desc([256, 512]))
        d += bytes(16)   # a trailing region no pointer covers
        self.assertFalse(bvh_groups.parse_descriptor(bytes(d))["tiles"])

    def test_table(self):
        d = table_desc([5, 9, 0x80000000])
        p = bvh_groups.parse_descriptor(d)
        self.assertEqual((p["family"], p["count"], p["page_word"]), ("table", 3, 1))
        for f in ("count_fields_equal", "values_sorted_distinct", "items_contiguous", "item_words", "tiles"):
            self.assertTrue(p[f], f)
        p = bvh_groups.parse_descriptor(table_desc([1, 2, 3, 4, 5, 6], items_first=False))
        self.assertTrue(p["tiles"])

    def test_table_checks_fail(self):
        self.assertFalse(bvh_groups.parse_descriptor(table_desc([9, 5]))["values_sorted_distinct"])
        self.assertFalse(bvh_groups.parse_descriptor(table_desc([5, 5]))["values_sorted_distinct"])
        d = bytearray(table_desc([1, 2]))
        struct.pack_into("<HH", d, 0x38, 1, 2)
        self.assertFalse(bvh_groups.parse_descriptor(bytes(d))["count_fields_equal"])
        d = bytearray(table_desc([1, 2]))
        struct.pack_into("<Q", d, 0x40, 7)   # first item gets a non-zero word outside +0x10
        self.assertFalse(bvh_groups.parse_descriptor(bytes(d))["item_words"])

    def test_not_a_descriptor(self):
        self.assertIsNone(bvh_groups.parse_descriptor(b"\x01" + bytes(0x9F)))
        d = bytearray(table_desc([1]))
        struct.pack_into("<Q", d, 0x08, 0x40000060)
        self.assertIsNone(bvh_groups.parse_descriptor(bytes(d)))
        d = bytearray(table_desc([1]))
        struct.pack_into("<Q", d, 0x18, 2)
        self.assertIsNone(bvh_groups.parse_descriptor(bytes(d)))


class BufferTest(unittest.TestCase):
    def test_blas_at_cumulative_offsets(self):
        buf, sizes = buffer_for([3, 40, 1])
        d = bvh_groups.parse_descriptor(instance_desc([sizes[0], 0, sizes[1], sizes[2]]))
        res = bvh_groups.check_buffer(d, buf + bytes(4096))
        self.assertTrue(res["ok"], res["reason"])
        self.assertEqual([o for o, _ in res["blas"]], [0, sizes[0], sizes[0] + sizes[1]])
        self.assertEqual([e for _, e in res["blas"]], [0x140 + 64 * n for n in (3, 40, 1)])

    def test_entry_not_rounded_end_fails(self):
        buf, sizes = buffer_for([3, 40])
        d = bvh_groups.parse_descriptor(instance_desc([sizes[0] + 256, sizes[1]]))
        self.assertFalse(bvh_groups.check_buffer(d, buf)["ok"])
        d = bvh_groups.parse_descriptor(instance_desc([sizes[0], sizes[1] - 256]))
        self.assertIn("rounded", bvh_groups.check_buffer(d, buf)["reason"])

    def test_missing_second_blas_fails(self):
        buf, sizes = buffer_for([3])
        d = bvh_groups.parse_descriptor(instance_desc([sizes[0], 512]))
        res = bvh_groups.check_buffer(d, buf + bytes(1024))
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["blas"]), 1)

    def test_all_zero_entries_fail(self):
        d = bvh_groups.parse_descriptor(instance_desc([0, 0]))
        self.assertFalse(bvh_groups.check_buffer(d, blas(2))["ok"])


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        one, s1 = buffer_for([8])
        two, s2 = buffer_for([3, 5])
        bad, s3 = buffer_for([4])
        self.rpf = os.path.join(self.tmp.name, "t.rpf")
        with open(self.rpf, "wb") as f:
            f.write(make_rpf([
                table_desc([1, 2]), table_desc([3]),
                instance_desc(s2), two + bytes(1000),
                instance_desc(s1), one,
                instance_desc([s3[0] + 256]), bad,
                table_desc([7]), instance_desc(s1),
            ]))

    def test_summary(self):
        rows, summary = bvh_groups.scan(self.rpf)
        c = summary["counts"]
        self.assertEqual(c["table/descriptor"], 3)
        self.assertEqual(c["table/descriptor items"], 4)
        self.assertEqual(c["table/descriptor next=descriptor"], 2)
        self.assertEqual(c["table/descriptor next=instance208"], 1)
        self.assertEqual(c["table/descriptor page_word=0x1"], 3)
        self.assertEqual(c["instance/descriptor buffer_ok=True"], 1)
        self.assertEqual(c["instance/descriptor blas tied"], 2)
        self.assertEqual(c["instance/instance208"], 3)
        self.assertEqual(c["instance/instance208 buffer_ok=True"], 1)
        self.assertEqual(c["instance/instance208 buffer_ok=False"], 2)
        self.assertEqual(summary["bvh_preceded_by"], {"instance/descriptor": 1, "instance/instance208": 2})
        last = [r for r in rows if r.get("family") == "instance"][-1]
        self.assertEqual(last["reason"], "last stream of the scan")
        wrong = [r for r in rows if r.get("buffer_ok") is False][0]
        self.assertIn("rounded", wrong["reason"])

    def test_main_writes_json_and_refuses_the_archive(self):
        out = os.path.join(self.tmp.name, "r.json")
        with contextlib.redirect_stdout(io.StringIO()) as so:
            self.assertEqual(bvh_groups.main([self.rpf, "--json", out]), 0)
        self.assertIn("table/descriptor", so.getvalue())
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 10)
        self.assertNotIn("1111222233334444", json.dumps(doc))
        with contextlib.redirect_stderr(io.StringIO()) as se:
            self.assertEqual(bvh_groups.main([self.rpf, "--json", self.rpf]), 2)
            self.assertEqual(bvh_groups.main([self.rpf, "--start", str(10 ** 9)]), 2)
        self.assertIn("bvh_groups:", se.getvalue())

    def test_empty_archive(self):
        empty = os.path.join(self.tmp.name, "e.rpf")
        open(empty, "wb").close()
        rows, summary = bvh_groups.scan(empty)
        self.assertEqual((rows, summary["streams"]), ([], 0))


if __name__ == "__main__":
    unittest.main()
