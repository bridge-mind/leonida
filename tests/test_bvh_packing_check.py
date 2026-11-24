"""Synthetic dehydrated BLASes for tools/bvh_packing_check.py: a clean two-level tree passes, and each broken
invariant is reported under its own reason. Also scans a small fake archive with a split stream. No game data.

    python -I tests/test_bvh_packing_check.py
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

import bvh_packing_check as C  # noqa: E402
import bvh_rehydrate as R  # noqa: E402

FLAGS = 0x80000000 | R.FLAG_DEHYDRATED | 0x00020006
ELEMENTS, NODES = 0x100, 0x200
BOUNDS = (0.0, 0.0, 0.0, 4.0, 2.0, 1.0)
GOOD_ORDER = 0x09


def ptr(offset, kind):
    return (offset >> 6) << 3 | kind


def leaf(i):
    return ptr(ELEMENTS + 64 * i, R.TRI0)


def group(nodes, wide=False):
    """nodes: [(children, {bit: value})] -> group bytes (node words, child pointers, stored values)."""
    words, kids, values = [], [], []
    for children, overrides in nodes:
        mask = 0
        for bit in sorted(overrides):
            mask |= 1 << bit
            values.append(overrides[bit])
        words.append(len(children) << 24 | mask)
        kids += children
    body = struct.pack("<%dI" % len(words), *words) + struct.pack("<%dI" % len(kids), *kids)
    if wide:
        return body + struct.pack("<%df" % len(values), *values)
    return body + struct.pack("<%dH" % len(values), *(R.float_to_half(v) for v in values))


def build(level0=None, level1=None, orders=None, n_leaves=4, flags=FLAGS, root=None, total=0x300, table=None):
    """Four leaves; level 0: two BOX16 nodes over leaves (0, 1) and (2, 3); level 1: the BOX16 root over both."""
    a, b = ptr(NODES, R.BOX16), ptr(NODES + 64, R.BOX16)
    root = ptr(NODES + 128, R.BOX16) if root is None else root
    level0 = [([leaf(0), leaf(1)], {}), ([leaf(2), leaf(3)], {})] if level0 is None else level0
    level1 = [([a, b], {})] if level1 is None else level1
    src = bytearray(total)
    src[0:64] = struct.pack("<4I6f6I", flags, total // 64 - 1, 0, n_leaves, *BOUNDS, NODES, 0, root, ELEMENTS,
                            0x40, R.NO_TRANSFORM)
    orders = [GOOD_ORDER] * n_leaves if orders is None else orders
    chunk = bytes(3 * n_leaves) + bytes([1] * n_leaves) + bytes(3 * n_leaves) + b"\xff" * (3 * n_leaves) + bytes(orders)
    src[ELEMENTS:ELEMENTS + len(chunk)] = chunk
    g0, g1 = group(level0), group(level1)
    src[NODES:NODES + len(g0)] = g0
    src[NODES + 128:NODES + 128 + len(g1)] = g1
    table = [(0, 128), (128, 192)] if table is None else table
    lv = b"".join(struct.pack("<2I", x, y) for x, y in table) + struct.pack("<I", len(table))
    src[total - len(lv):] = lv
    return bytes(src)


def fails(src):
    return dict(C.check_stream(src).fails)


class TestCleanTree(unittest.TestCase):
    def test_passes_with_stats(self):
        res = C.check_stream(build())
        self.assertEqual(dict(res.fails), {})
        self.assertEqual(res.stats["box_nodes"], 3)
        self.assertEqual(res.stats["child_type_0"], 4)
        self.assertEqual(res.stats["child_type_4"], 2)
        self.assertEqual(res.stats["levels_2"], 1)
        for key in ("root_in_top_level", "root_is_last_node", "top_level_single_node"):
            self.assertEqual(res.stats[key], 1, key)

    def test_header_offset(self):
        self.assertEqual(dict(C.check_stream(bytes(256) + build(), 256).fails), {})

    def test_overrides_inside_bounds_pass(self):
        src = build(level0=[([leaf(0), leaf(1)], {0: 0.5, 6 + 3: 3.5}), ([leaf(2), leaf(3)], {})])
        res = C.check_stream(src)
        self.assertEqual(dict(res.fails), {})
        self.assertEqual(res.stats["overrides"], 2)

    def test_root_union_equal_to_bounds(self):
        # both root slots store max.x; their union is the bound 4.0
        src = build(level1=[([ptr(NODES, R.BOX16), ptr(NODES + 64, R.BOX16)], {3: 2.0, 6 + 3: 4.0})])
        res = C.check_stream(src)
        self.assertEqual(dict(res.fails), {})
        self.assertEqual((res.stats["root_components_stored"], res.stats["root_components_equal"]), (1, 1))


class TestBroken(unittest.TestCase):
    def test_not_dehydrated(self):
        self.assertEqual(fails(build(flags=FLAGS & ~R.FLAG_DEHYDRATED)), {"not_dehydrated": 1})

    def test_short_input(self):
        self.assertEqual(fails(build()[:0x200]), {"short_input": 1})

    def test_shared_and_unreferenced_leaf(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {}), ([leaf(1), leaf(3)], {})]))
        self.assertEqual(f, {"leaf_shared": 1, "leaf_unreferenced": 1})

    def test_child_not_a_leaf(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {}), ([leaf(2), leaf(9)], {})]))
        self.assertIn("child_not_a_leaf", f)

    def test_child_wrong_width(self):
        f = fails(build(level1=[([ptr(NODES, R.BOX32), ptr(NODES + 64, R.BOX16)], {})]))
        self.assertIn("child_wrong_width", f)
        self.assertIn("node_unreferenced", f)

    def test_child_not_below(self):
        # a level-0 node pointing at its sibling on the same level
        f = fails(build(level0=[([leaf(0), leaf(1), ptr(NODES + 64, R.BOX16)], {}), ([leaf(2), leaf(3)], {})]))
        self.assertIn("child_not_below", f)
        self.assertIn("node_shared", f)

    def test_root_referenced(self):
        f = fails(build(level0=[([leaf(0), leaf(1), ptr(NODES + 128, R.BOX16)], {}), ([leaf(2), leaf(3)], {})]))
        self.assertIn("root_referenced", f)

    def test_too_many_children(self):
        # a node word claiming five children (the fifth pointer is read from the group as well)
        level0 = [([leaf(0), leaf(1), leaf(2), leaf(3), ptr(NODES + 64, R.BOX16)], {}), ([], {})]
        self.assertIn("too_many_children", fails(build(level0=level0)))

    def test_child_not_a_node(self):
        f = fails(build(level1=[([ptr(NODES, R.BOX16), ptr(NODES + 192, R.BOX16)], {})]))
        self.assertIn("child_not_a_node", f)
        self.assertNotIn("child_wrong_width", f)

    def test_invalid_child(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {}), ([leaf(2), R.INVALID], {})]))
        self.assertIn("invalid_child", f)
        self.assertIn("leaf_unreferenced", f)

    def test_root_not_a_box(self):
        # a TRI0 root over four leaves is not the single-leaf form
        self.assertEqual(fails(build(root=leaf(0))), {"root_not_a_box": 1})

    def test_leaves_past_end(self):
        self.assertEqual(fails(build(n_leaves=20, orders=[GOOD_ORDER] * 20)), {"leaves_past_end": 1})

    def test_other_child_type(self):
        f = fails(build(level0=[([leaf(0), leaf(1) | 6], {}), ([leaf(2), leaf(3)], {})]))
        self.assertIn("child_type_6", f)

    def test_no_children(self):
        self.assertIn("no_children", fails(build(level0=[([], {}), ([leaf(2), leaf(3)], {})], n_leaves=4)))

    def test_mask_past_children(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {12: 0.5}), ([leaf(2), leaf(3)], {})]))
        self.assertEqual(f, {"mask_past_children": 1})

    def test_override_outside_bounds(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {3: 9.0}), ([leaf(2), leaf(3)], {})]))
        self.assertEqual(f, {"override_outside_bounds": 1})

    def test_infinite_box16_override_inside_half_range(self):
        f = fails(build(level0=[([leaf(0), leaf(1)], {3: float("inf")}), ([leaf(2), leaf(3)], {})]))
        self.assertEqual(f, {"override_outside_bounds": 1})

    def test_root_union_not_bounds(self):
        src = build(level1=[([ptr(NODES, R.BOX16), ptr(NODES + 64, R.BOX16)], {3: 2.0, 6 + 3: 3.0})])
        self.assertEqual(fails(src), {"root_union_not_bounds": 1})

    def test_bad_vertex_order(self):
        # 0x05: first = 1, second = 1 -> two vertices in one slot
        self.assertEqual(fails(build(orders=[GOOD_ORDER, 0x05, GOOD_ORDER, 0x05])), {"leaf_vertex_order": 2})

    def test_level_table_out_of_order(self):
        self.assertEqual(fails(build(table=[(0, 100), (128, 192)])), {"level_table": 1})

    def test_levels_past_table(self):
        self.assertEqual(fails(build(table=[(0, 128), (128, 0x300)])), {"levels_past_table": 1})

    def test_group_overflow(self):
        # a single BOX16 node (64 bytes) cannot hold a word, four children and 24 halves (4 + 16 + 48 bytes)
        many = {bit: 0.5 for bit in range(24)}
        level0 = [([leaf(0), leaf(1), leaf(2), leaf(3)], many)]
        src = build(level0=level0, level1=[([ptr(NODES, R.BOX16)], {})], table=[(0, 64), (64, 128)],
                    root=ptr(NODES + 64, R.BOX16))
        self.assertIn("group_overflow", fails(src))

    def test_root_not_a_node(self):
        self.assertEqual(fails(build(root=ptr(NODES + 192, R.BOX16))), {"root_not_a_node": 1})

    def test_root_leaf(self):
        src = build(n_leaves=1, root=leaf(0), level0=[], level1=[], table=[])
        res = C.check_stream(src)
        self.assertEqual(dict(res.fails), {})
        self.assertEqual(res.stats["root_leaf"], 1)


def payload(structure):
    """A whole bvh4 payload: the 0x100 prefix with the wrapper words, then the structure."""
    end = 0x100 + len(structure)
    prefix = bytearray(0x100)
    struct.pack_into("<I", prefix, 0x20, end)
    struct.pack_into("<I", prefix, 0x28, end - 0x100)
    return bytes(prefix) + structure


class TestArchive(unittest.TestCase):
    def write_rpf(self, streams):
        body = b"".join(zlib.compress(s) + bytes(16) for s in streams)
        fd, path = tempfile.mkstemp(suffix=".rpf")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RPF9" + struct.pack("<I", 16) + bytes(8) + body)
        self.addCleanup(os.unlink, path)
        return path

    def test_scan_counts_and_split_stream(self):
        good = payload(build())
        bad = payload(build(orders=[GOOD_ORDER, 0x05, GOOD_ORDER, GOOD_ORDER]))
        big = payload(build(total=0x20000))          # 0x20100 bytes: split into a 64 KiB chunk and the rest
        path = self.write_rpf([good, b"<?xml version='1.0'?><a/>" + bytes(64), bad, big[:0x10000], big[0x10000:]])
        for workers, window in ((1, 1 << 20), (2, 64)):
            tally, _ = C.scan_archive(path, workers, window, progress=False)
            rep = tally.report()
            self.assertEqual((rep["checked"], rep["passed"]), (3, 2), (workers, window))
            self.assertEqual(rep["fail_streams"], {"leaf_vertex_order": 1})
            self.assertEqual(rep["kinds"].get("bvh4-chunk-joined"), 1)
            self.assertEqual(rep["kinds"].get("continuation"), 1)
            self.assertNotIn("raw", rep["kinds"])

    def test_stream_starting_at_window_boundary_is_not_dropped(self):
        good_z = zlib.compress(payload(build()))
        bad_z = zlib.compress(payload(build(orders=[GOOD_ORDER, 0x05, GOOD_ORDER, GOOD_ORDER])))
        # Header has data_start = 16; with window = 64, window 0 is [16, 80).
        # Place good_z at offset 79 (end - 1) so its 2-byte zlib header straddles 79..80,
        # and place bad_z on an exact later window boundary to verify no double-counting.
        window = 64
        pad_before_good = (16 + window - 1) - 16
        after_good = 16 + pad_before_good + len(good_z)
        next_boundary = 16 + ((after_good - 16 + window - 1) // window) * window
        if next_boundary == after_good:
            next_boundary += window
        pad_before_bad = next_boundary - after_good
        raw = b"RPF9" + struct.pack("<I", 16) + bytes(8) + bytes(pad_before_good) + good_z + bytes(pad_before_bad) + bad_z + bytes(16)
        fd, path = tempfile.mkstemp(suffix=".rpf")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        self.addCleanup(os.unlink, path)

        for workers in (1, 2):
            tally, _ = C.scan_archive(path, workers, window, progress=False)
            rep = tally.report()
            self.assertEqual((rep["checked"], rep["passed"]), (2, 1), workers)
            self.assertEqual(rep["fail_streams"], {"leaf_vertex_order": 1}, workers)

        # Direct window check: stream at end - 1 belongs to window 0, stream at next_boundary belongs only to its window.
        w0 = C.scan_window((path, 16, 16 + window))
        self.assertEqual((w0.checked, w0.passed), (1, 1))
        w_before_bad = C.scan_window((path, next_boundary - window, next_boundary))
        self.assertEqual(w_before_bad.checked, 0)
        w_bad = C.scan_window((path, next_boundary, next_boundary + window))
        self.assertEqual((w_bad.checked, w_bad.passed), (1, 0))

    def test_cli_json_and_exit_status(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = os.path.join(d, "good.bin"), os.path.join(d, "bad.bin")
            with open(good, "wb") as f:
                f.write(build())
            with open(bad, "wb") as f:
                f.write(build(orders=[0x05] * 4))
            out = os.path.join(d, "r.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main([good, "--json", out]), 0)
                self.assertEqual(C.main([good, bad, "--json", out]), 1)
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual((rep["checked"], rep["failed"]), (2, 1))
            self.assertEqual(rep["fail_examples"]["leaf_vertex_order"], [bad])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(C.main([good, "--json", good]), 2)          # never overwrite an input
                self.assertEqual(C.main([good, "--limit", "5"]), 2)          # --limit is for an archive
                self.assertEqual(C.main([os.path.join(d, "missing.bin")]), 2)
                self.assertEqual(C.main([good, "--json", d]), 2)             # --json is a directory
                self.assertEqual(C.main([good, "--json", os.path.join(d, "no_such_dir", "r.json")]), 2)
                self.assertEqual(C.main([good, "--json", os.path.join(d, ".", os.path.basename(good))]), 2)
                if sys.platform == "win32":
                    self.assertEqual(C.main([good, "--json", good.upper()]), 2)  # case-distinct spelling on Windows


if __name__ == "__main__":
    unittest.main()
