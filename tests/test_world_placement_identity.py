"""Placement identity through the split-stream --entities-out path (synthetic data only)."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(TESTS), str(TESTS.parent / "tools")]
import test_world_index as twi  # noqa: E402


STRIDE = 0x90
PARENT_MAPS = (0xABCD, 0xCDEF0123)
UNSUPPORTED = 5  # one-based block ID of the unsupported class


def entity_ref(index):
    return ((index * STRIDE) << 20) | 1  # block 1, byte offset in the upper bits


def make_map(links, pointers, parent_maps=PARENT_MAPS, *, map_name=0xABCDEF, first_x=10):
    """Extend the stock fixture's CMapData root and its template CEntityDef."""
    blob = bytearray(twi.make_resource())
    template = blob[0x400:0x400 + STRIDE]

    def extend_members(header, additions):
        old_table = struct.unpack_from("<Q", blob, header + 0x10)[0] & 0x0FFFFFFF
        count = struct.unpack_from("<H", blob, header + 0x1E)[0]
        members = blob[old_table:old_table + 16 * count]
        struct.pack_into("<Q", blob, header + 0x10, twi.ptr(len(blob)))
        struct.pack_into("<H", blob, header + 0x1E, count + len(additions))
        blob.extend(members + b"".join(twi.member(*args) for args in additions))

    def put_block(index, tag, data):
        struct.pack_into("<IIQ", blob, 0x200 + 16 * (index - 1), tag, len(data), twi.ptr(len(blob)))
        blob.extend(data)

    extend_members(0x100, [(twi.WI.FIELD["parentIndex"], 0x5C, 0x12),
                           (twi.WI.FIELD["parentInstance"], 0x5E, 0x12),
                           (0x84799847, 0x7F, 0x10)])
    extend_members(0x120, [(0xCCAB9307, 0x20, 0x52), (0x75D16ADF, 0x58, 0x52)])
    blob[0x140:0x160] = twi.struct_def(0x12345678, 16, [], 0x140, 0)
    struct.pack_into("<I", blob, 0x20, 2)  # the stock fixture's CMapData is block 2
    struct.pack_into("<H", blob, 0x48, 3)
    struct.pack_into("<H", blob, 0x4C, 5)
    struct.pack_into("<I", blob, 0x508, map_name)
    struct.pack_into("<QHH", blob, 0x520, 3, len(pointers), len(pointers))
    struct.pack_into("<QHH", blob, 0x558, 4, len(parent_maps), len(parent_maps))

    records = bytearray()
    for index, (selector, parent_index, instance) in enumerate(links):
        row = template.copy()
        struct.pack_into("<f", row, 0x38, first_x + index)
        struct.pack_into("<hh", row, 0x5C, parent_index, instance)
        struct.pack_into("<b", row, 0x7F, selector)
        records.extend(row)
    put_block(1, twi.WI.ENTITY, records)
    put_block(3, 7, b"".join(struct.pack("<Q", ref) for ref in pointers))
    put_block(5, 0x12345678, bytes(16))
    put_block(4, 0x4A, b"".join(struct.pack("<I", name) for name in parent_maps))
    return blob


class PlacementIdentityTest(unittest.TestCase):
    def export(self, *maps, aggregate_count=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, out, rows = root / "fixture.rpf", root / "maps.json", root / "entities.jsonl"
            parts = [part for blob in maps for part in (blob[:0x150], blob[0x150:])]
            archive.write_bytes(twi.make_rpf_streams(parts))
            args = [str(archive), "--out", str(out), "--entities-out", str(rows)]
            aggregates = root / "archetypes.json"
            if aggregate_count is not None:
                args.extend(["--archetypes-out", str(aggregates)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(twi.WI.main(args), 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["summary"]["streams"], len(parts))
            if aggregate_count is not None:
                report = json.loads(aggregates.read_text(encoding="utf-8"))
                self.assertEqual(report["archetypes"], {"deadbeef": aggregate_count})
                self.assertEqual(report["keys"], {"deadbeef": "1122334455667788"})
            return [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]

    def test_authored_order_includes_unsupported_class_and_parent_maps_follow_array_order(self):
        order = [UNSUPPORTED, entity_ref(1), entity_ref(0)]
        links = [(1, 3, -1), (0, 3, -1)]
        for hashes, selected in [(PARENT_MAPS, ["cdef0123", "0000abcd"]),
                                 (PARENT_MAPS[::-1], ["0000abcd", "cdef0123"])]:
            with self.subTest(hashes=hashes):
                rows = self.export(make_map(links, order, hashes))
                self.assertEqual([r["map_name"] for r in rows], ["00abcdef", "00abcdef"])
                self.assertEqual([r["entity_index"] for r in rows], [2, 1])
                self.assertEqual([r["parent_map_name"] for r in rows], selected)
                for i, row in enumerate(rows):
                    legacy = {"archetype": "deadbeef", "key": "1122334455667788",
                              "position": [10 + i, 2.5, 3.5], "rotation": [0, 0, 0, 1],
                              "scale_xy": 1, "scale_z": 1, "lod_level": 1, "parent_index": 3,
                              "num_children": None, "lod_dist": 250, "child_lod_dist": None}
                    self.assertEqual({key: row[key] for key in legacy}, legacy)

    def test_parent_map_requires_valid_signed_selector_ordinary_parent_and_index(self):
        links = [(-1, 0, -1), (2, 0, -1), (0, -1, -1), (0, 0, 0)]
        rows = self.export(make_map(links, [entity_ref(i) for i in range(len(links))]))
        self.assertEqual([r["entity_index"] for r in rows], [0, 1, 2, 3])
        self.assertEqual([r["parent_index"] for r in rows], [0, 0, -1, 0])
        for row in rows:
            self.assertNotIn("parent_map_name", row)

    def test_duplicate_pointer_and_unreferenced_row_have_no_authored_index(self):
        pointers = [entity_ref(0), entity_ref(0), entity_ref(2)]
        rows = self.export(make_map([(0, 0, -1)] * 3, pointers))
        self.assertEqual([r["position"][0] for r in rows], [10, 11, 12])
        for row in rows[:2]:
            self.assertNotIn("entity_index", row)
            self.assertNotIn("parent_map_name", row)
        self.assertEqual(rows[2]["entity_index"], 2)

    def test_cross_map_join_selects_the_authored_parent_from_combined_export(self):
        child = make_map([(1, 2, -1)], [entity_ref(0)])
        target = make_map([(-1, -1, -1)] * 2, [UNSUPPORTED, entity_ref(1), entity_ref(0)],
                          map_name=PARENT_MAPS[1], first_x=100)
        rows = self.export(child, target, aggregate_count=3)
        self.assertEqual([r["position"][0] for r in rows], [10, 100, 101])
        by_identity = {(r["map_name"], r["entity_index"]): r for r in rows}
        source = by_identity["00abcdef", 0]
        parent = by_identity[source["parent_map_name"], source["parent_index"]]
        self.assertEqual(parent["position"][0], 100)
        self.assertEqual(by_identity["cdef0123", 1]["position"][0], 101)

    def test_truncated_references_do_not_read_neighboring_bytes(self):
        links = [(0, 2, -1), (1, 2, -1)]
        for damaged in ("pointer block", "hash block", "hash payload"):
            with self.subTest(damaged=damaged):
                blob = make_map(links, [entity_ref(0), entity_ref(1)])
                if damaged == "pointer block":
                    struct.pack_into("<I", blob, 0x224, 8)  # second pointer outside declared block
                elif damaged == "hash block":
                    struct.pack_into("<I", blob, 0x234, 4)  # second hash outside declared block
                else:
                    del blob[-1:]  # block is declared complete, but the last hash is short
                rows = self.export(blob)
                self.assertEqual([r["position"][0] for r in rows], [10, 11])
                field = "entity_index" if damaged == "pointer block" else "parent_map_name"
                for row in rows:
                    self.assertNotIn(field, row)


if __name__ == "__main__":
    unittest.main()
