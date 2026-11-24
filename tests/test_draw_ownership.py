"""Synthetic-data tests for tools/draw_ownership.py.

    python -I tests/test_draw_ownership.py

Builds a ~64 KiB system segment with five drawable scenarios, then walks it and
reverses it:

  A1, A2 - regular chains: the walk MUST find the geometries, and classify_parent
           answers "model-compatible" for the walked entries tables.
  B      - the LOD+8 used/capacity pair has used != capacity (H3): the walk rejects it.
  C      - the LOD pointer sits at +72, not at the +80 the schema reads (H1
           simulation): the walk cannot find it, only the reverse scan shows the
           parent offset.
  D      - an orphan geometry-entries table that no drawable/model points at, only a
           foreign structure: classify_parent answers "unknown parent", with a
           32-byte hex dump.

Every byte here is built by the test. No game file, value or name is used.
"""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import draw_ownership as dwo  # noqa: E402

SIZE = 64 * 1024


class Builder:
    """8-byte-aligned bump allocator and field writers for the synthetic segment.

    Every system-pointer write is recorded as a (source position, target offset) edge,
    so the tests can check the reverse index against exactly the edges that were built.
    """

    def __init__(self, size=SIZE):
        self.buf = bytearray(size)
        self.pos = 0
        self.edges = []

    def alloc(self, nbytes):
        self.pos = (self.pos + 7) & ~7
        at = self.pos
        self.pos += nbytes
        if self.pos > len(self.buf):
            raise OverflowError("out of space in the synthetic segment")
        return at

    def u16(self, at, value):
        struct.pack_into("<H", self.buf, at, value)

    def u32(self, at, value):
        struct.pack_into("<I", self.buf, at, value)

    def ptr(self, at, target):
        struct.pack_into("<Q", self.buf, at, dwo.PTR_BASE + target)
        self.edges.append((at, target))

    def graphics_ptr(self, at, offset):
        struct.pack_into("<Q", self.buf, at, 0x60000000 + offset)


def build_geometry(b, vertices=192):
    """A 160B geometry record with its system-side companions: an 80B vertex buffer, a
    320B vertex layout, a 16B meshlet descriptor and a 64B index-buffer record. The
    graphics pointers point outside the segment (0x60000000-based), as in a real pair."""
    geo = b.alloc(160)
    vb = b.alloc(80)
    layout = b.alloc(320)
    desc = b.alloc(16)
    ib = b.alloc(64)
    b.u32(vb + 8, vertices)            # count u32
    b.u16(vb + 12, 12)                 # stride u16
    b.graphics_ptr(vb + 24, 0x1000)    # graphics vertex-block address
    b.ptr(vb + 64, layout)
    b.u32(layout, 128)                 # layout+0 = 128
    b.buf[layout + 0xD0] = 8           # layout+0xD0 = 8
    b.buf[layout + 0x104] = 16         # layout+0x104 = 16
    b.u32(layout + 0x138, (vertices << 10) | 1)  # flags: count=flags>>10, lsb=1
    b.ptr(desc, ib)
    b.u16(desc + 8, 2)                 # meshlets u16
    b.u16(desc + 10, 8)                # triangles u16
    b.u32(desc + 12, 16 << 8)          # binding u32: refs=(binding>>8)&0x7FFFFF
    b.u32(ib + 8, 64)                  # index word count
    b.u16(ib + 12, 4)                  # stride u16 = 4
    b.graphics_ptr(ib + 24, 0x2000)
    b.ptr(geo + 8, vb)
    b.ptr(geo + 64, desc)
    return geo


def build_model(b, geometry_offsets):
    """A 56B model record plus a geometry-entries table of 8B slots (used == capacity)."""
    model = b.alloc(56)
    entries = b.alloc(8 * len(geometry_offsets))
    for j, geo in enumerate(geometry_offsets):
        b.ptr(entries + 8 * j, geo)
    b.ptr(model + 8, entries)
    b.u16(model + 16, len(geometry_offsets))
    b.u16(model + 18, len(geometry_offsets))
    b.u16(model + 54, len(geometry_offsets))
    return model, entries


def build_lod(b, model_offsets, used=None, capacity=None):
    """A 16B LOD record plus a model-pointer table of 8B slots. used/capacity can be
    overridden for the H3 case; the table holds `capacity` slots and the first
    len(model_offsets) are live."""
    n_slots = capacity if capacity is not None else len(model_offsets)
    lod = b.alloc(16)
    table = b.alloc(8 * n_slots)
    for i, model in enumerate(model_offsets):
        b.ptr(table + 8 * i, model)
    b.ptr(lod, table)
    b.u16(lod + 8, used if used is not None else len(model_offsets))
    b.u16(lod + 10, n_slots)
    return lod, table


def build_drawable(b, lod, link=80):
    """A 176B drawable record; the LOD pointer goes to the `link` offset (+80 is the
    schema, +72 the H1 simulation)."""
    drawable = b.alloc(176)
    b.ptr(drawable + link, lod)
    return drawable


def build_segment():
    """Build the whole synthetic segment. Returns (system bytes, recorded edges,
    dict of the offsets the tests assert on)."""
    b = Builder()

    # --- A1: two models (3 + 1 geometries) ---
    a1_geos = [build_geometry(b) for _ in range(3)] + [build_geometry(b)]
    a1_m1, a1_e1 = build_model(b, a1_geos[:3])
    a1_m2, a1_e2 = build_model(b, a1_geos[3:])
    a1_lod, _ = build_lod(b, [a1_m1, a1_m2])
    a1 = build_drawable(b, a1_lod)

    # --- A2: one model (2 geometries) ---
    a2_geos = [build_geometry(b, vertices=128) for _ in range(2)]
    a2_m, a2_e = build_model(b, a2_geos)
    a2_lod, _ = build_lod(b, [a2_m])
    a2 = build_drawable(b, a2_lod)

    # --- B: used != capacity at LOD+8 (H3) ---
    b_geos = [build_geometry(b) for _ in range(2)]
    b_m, _ = build_model(b, b_geos)
    b_lod, _ = build_lod(b, [b_m], used=2, capacity=5)
    b_drawable = build_drawable(b, b_lod)

    # --- C: the LOD pointer sits at drawable+72, the schema's +80 stays zero (H1 simulation) ---
    c_geos = [build_geometry(b) for _ in range(2)]
    c_m, _ = build_model(b, c_geos)
    c_lod, _ = build_lod(b, [c_m])
    c_drawable = build_drawable(b, c_lod, link=72)

    # --- D: an orphan entries table (no model or drawable above it); its parent is a
    #        foreign, non-model-shaped 64B record whose +16 field points at it ---
    d_geos = [build_geometry(b) for _ in range(2)]
    d_entries = b.alloc(8 * len(d_geos))
    for j, geo in enumerate(d_geos):
        b.ptr(d_entries + 8 * j, geo)
    d_x = b.alloc(64)
    d_scratch = b.alloc(32)
    b.ptr(d_x, d_scratch)              # another pointer field of the foreign structure
    b.ptr(d_x + 16, d_entries)         # the single edge to the orphan table
    b.u32(d_x + 40, 0x7F3A00C1)        # foreign type tag / count-like fields

    system = bytes(b.buf)
    offsets = {"a1": a1, "a1_geos": a1_geos, "a1_m1": a1_m1, "a1_e1": a1_e1,
               "a1_m2": a1_m2, "a1_e2": a1_e2, "a1_lod": a1_lod,
               "a2": a2, "a2_lod": a2_lod, "a2_geos": a2_geos, "a2_m": a2_m, "a2_e": a2_e,
               "b_drawable": b_drawable, "c_drawable": c_drawable, "c_lod": c_lod,
               "d_geos": d_geos, "d_entries": d_entries, "d_x": d_x}
    return system, b.edges, offsets


class WalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system, _, cls.off = build_segment()

    def test_a1_two_model_chain(self):
        found = list(dwo.walk_drawable(self.system, self.off["a1"]))
        self.assertEqual(found, self.off["a1_geos"])
        chain = dwo.walk_chain(self.system, self.off["a1"])
        self.assertEqual(chain["drawable"], self.off["a1"])
        self.assertEqual(chain["lod"], self.off["a1_lod"])
        self.assertEqual([m["model"] for m in chain["models"]], [self.off["a1_m1"], self.off["a1_m2"]])
        self.assertEqual([m["entries"] for m in chain["models"]], [self.off["a1_e1"], self.off["a1_e2"]])
        self.assertEqual([m["geometries"] for m in chain["models"]],
                         [self.off["a1_geos"][:3], self.off["a1_geos"][3:]])

    def test_a2_single_model_chain(self):
        found = list(dwo.walk_drawable(self.system, self.off["a2"]))
        self.assertEqual(found, self.off["a2_geos"])
        chain = dwo.walk_chain(self.system, self.off["a2"])
        self.assertEqual(chain["lod"], self.off["a2_lod"])
        self.assertEqual([m["model"] for m in chain["models"]], [self.off["a2_m"]])
        self.assertEqual([m["entries"] for m in chain["models"]], [self.off["a2_e"]])

    def test_b_used_capacity_mismatch_is_rejected(self):
        # H3: the walk must reject used != capacity, with the exact pair in the message
        with self.assertRaises(dwo.CountError) as ctx:
            list(dwo.walk_drawable(self.system, self.off["b_drawable"]))
        message = str(ctx.exception)
        self.assertIn("used=2", message)
        self.assertIn("capacity=5", message)

    def test_drawable_outside_segment_is_rejected(self):
        with self.assertRaises(dwo.OutOfSegmentError):
            list(dwo.walk_drawable(self.system, len(self.system)))


class ReverseScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system, cls.edges, cls.off = build_segment()
        cls.index = dwo.reverse_pointers(cls.system)

    def test_c_walk_misses_the_lod_pointer_at_72(self):
        with self.assertRaises(dwo.OutOfSegmentError):
            list(dwo.walk_drawable(self.system, self.off["c_drawable"]))

    def test_c_reverse_scan_finds_the_parent(self):
        # H1 simulation: the only incoming edge of the LOD record is the +72 field
        sources = self.index.get(self.off["c_lod"], [])
        self.assertEqual(sources, [self.off["c_drawable"] + 72])

    def test_reverse_index_has_no_false_positives(self):
        # The reverse index is exactly the set of pointers the builder wrote: every edge
        # in it is a real pointer field, and no count/stride/tag value became an edge.
        expected = {}
        for at, target in self.edges:
            expected.setdefault(target, []).append(at)
        for sources in expected.values():
            sources.sort()
        self.assertEqual(self.index, expected)

    def test_walked_geometries_have_their_entries_edge(self):
        for drawable in (self.off["a1"], self.off["a2"]):
            chain = dwo.walk_chain(self.system, drawable)
            for model in chain["models"]:
                for j, geo in enumerate(model["geometries"]):
                    self.assertIn(model["entries"] + 8 * j, self.index.get(geo, []))

    def test_walked_entries_tables_classify_model_compatible(self):
        for entries_key, model_key, geo_n in (("a1_e1", "a1_m1", 3), ("a1_e2", "a1_m2", 1),
                                              ("a2_e", "a2_m", 2)):
            with self.subTest(entries=hex(self.off[entries_key])):
                parents = dwo.classify_parent(self.system, self.off[entries_key], self.index)
                self.assertEqual(len(parents), 1)
                self.assertEqual(parents[0]["classification"], "model-compatible")
                self.assertEqual(parents[0]["model"], self.off[model_key])
                self.assertEqual(parents[0]["geometry_count"], geo_n)
                self.assertTrue(parents[0]["model_referenced"])

    def test_orphan_entries_table_classifies_unknown_parent(self):
        parents = dwo.classify_parent(self.system, self.off["d_entries"], self.index)
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        self.assertEqual(parent["source"], self.off["d_x"] + 16)
        self.assertEqual(parent["classification"], "unknown parent")
        self.assertEqual(parent["model_candidate"], self.off["d_x"] + 8)
        self.assertEqual(len(parent["context_dump"].splitlines()), 2)

    def test_orphan_geometries_pass_the_shape_check(self):
        # The D table's geometry records are geometry-shaped on the system side;
        # only their parent is unknown.
        for geo in self.off["d_geos"]:
            dwo.geometry_shape_check(self.system, geo)


if __name__ == "__main__":
    unittest.main()
