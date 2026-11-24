"""Synthetic controls for multi-slot BVH geometry checks."""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")]

import bvh_slot_check as S
from test_bvh_geometry_check import BOUNDS, LEAVES, TRUE, build
from test_bvh_rehydrate import box_group, header, leaf_chunk, level_table, ptr


def translation(dx):
    return ((1, 0, 0, dx), (0, 1, 0, 0), (0, 0, 1, 0))


def tiled_block(dx):
    """32 pairs of triangles, eight boxes of leaves, two boxes of boxes, then the root."""
    elements, nodes = 0x400, 0x1000
    body = bytearray(0x1800)
    leaves = [(0, 0x81, 2 * i, 2 * i + 1, 0x09) for i in range(32)]
    packed = leaf_chunk(leaves)
    body[elements:elements + len(packed)] = packed
    children = [ptr(elements + 64 * i, S.R.TRI0) for i in range(32)]
    end = 0
    levels = []
    while len(children) > 1:
        boxes = [(children[i:i + 4], {}) for i in range(0, len(children), 4)]
        packed = box_group(boxes, True)
        base = nodes + end
        body[base:base + len(packed)] = packed
        children = [ptr(base + 128 * i, S.R.BOX32) for i in range(len(boxes))]
        end += 128 * len(boxes)
        levels.append((end, end))
    body[:64] = header(len(body), 32, (dx, 0, 0, dx + 32, 1, 0), nodes, children[0], elements, 64)
    table = level_table(levels)
    body[-len(table):] = table
    wrapper = bytearray(256)
    struct.pack_into("<I", wrapper, 32, 256 + len(body))
    struct.pack_into("<I", wrapper, 40, len(body))
    return wrapper + body


def export_fixture(root, swapped=False):
    manifest = root / "manifest.json"
    item = {"File": "tiles.obj", "MeshTris": [64]}
    manifest.write_text(json.dumps({"Items": [item]}), encoding="utf-8", newline="\n")
    with (root / "tiles.obj").open("w", encoding="ascii", newline="\n") as f:
        f.write("o tiles\n")
        for i in range(32):
            for x, y in ((i, 0), (i + 1, 0), (i, 1), (i + 1, 1)):
                f.write(f"v {x} {y} 0\n")
            j = 4 * i + 1
            f.write(f"f {j} {j+1} {j+2}\nf {j+2} {j+1} {j+3}\n")
    blocks = [tiled_block(0), tiled_block(100)]
    desc = bytearray(256)
    struct.pack_into("<4Q", desc, 8, 0x500000d0, 2, 0x500000b0, 0x60000000)
    struct.pack_into("<Q", desc, 56, 0x50000050)
    for i, dx in enumerate((0, 100)):
        struct.pack_into("<12f", desc, 80 + 48 * i, *(v for row in translation(dx) for v in row))
        struct.pack_into("<Q", desc, 176 + 8 * i, len(blocks[i]))
    order = blocks[::-1] if swapped else blocks
    archive = root / "tiles.rpf"
    index = root / "index.json"
    data = bytearray(struct.pack("<4sI", b"RPF9", 8))
    rows = []
    for stream in (desc, b"".join(order)):
        compressed = zlib.compress(stream)
        rows.append(dict(off=len(data), stored=len(compressed), size=len(stream), head=stream[:16].hex()))
        data.extend(compressed)
    archive.write_bytes(data)
    index.write_text(json.dumps({"files": rows, "summary": {"files": 2, "span": len(data) - 8}}),
                     encoding="utf-8", newline="\n")
    return archive, index, manifest


class SlotControls(unittest.TestCase):
    def test_census_with_known_source_and_swapped_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for swapped, outcome in ((False, "own_only"), (True, "alternative_only")):
                archive, index, manifest = export_fixture(root, swapped)
                report = S.census(archive, index, [manifest])
                self.assertEqual(report[outcome], 2)
                self.assertEqual(report["occupied_slots"], 2)

    def test_incremental_cli_matches_joint_scan_through_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, index, manifest = export_fixture(root)

            def run(*args):
                output = subprocess.check_output([sys.executable, "-I", S.__file__, archive, index, *args])
                return json.loads(output)

            first = root / "first.json"
            overlap = root / "overlap.json"
            conflict = root / "conflict.json"
            run(manifest, "--save-evidence", first)
            other = root / "other"
            other.mkdir()
            second = other / manifest.name
            shutil.copyfile(manifest, second)
            obj = other / "tiles.obj"
            shutil.copyfile(root / obj.name, obj)
            run(second, "--save-evidence", overlap)
            joint = run(manifest, second)
            self.assertEqual(joint["own_only"], 2)
            self.assertEqual(run("--prior", first, "--prior", overlap), joint)
            self.assertEqual(run(second, "--prior", first), joint)

            obj.write_bytes(obj.read_bytes().replace(b"v 2 0 0", b"v 2.000001 0 0"))
            self.assertEqual(run(second, "--save-evidence", conflict)["own_only"], 2)
            joint = run(manifest, second, manifest)
            self.assertEqual(joint["unknown_multiple_sources"], 2)
            for left, right in ((first, conflict), (conflict, first)):
                merged = root / "merged.json"
                run("--prior", left, "--prior", right, "--save-evidence", merged)
                self.assertEqual(run("--prior", merged, "--prior", overlap), joint)

            export_fixture(root, swapped=True)
            with self.assertRaisesRegex(ValueError, "archive/index differs"):
                S.census(archive, index, [], [first])

    def test_swapping_two_matrices_changes_both_verdicts(self):
        body = build(LEAVES, bounds=BOUNDS)[0]
        shifted = build(LEAVES, bounds=(4, 0, 0, 5, 1, 0))[0]
        matrices = [translation(0), translation(4)]
        for k, src in enumerate([body, shifted]):
            self.assertEqual(S.compare(src, [TRUE], matrices, k), "own_only")
            self.assertEqual(S.compare(src, [TRUE], matrices[::-1], k), "alternative_only")

    def test_duplicate_symmetric_and_neither(self):
        body = build(LEAVES, bounds=BOUNDS)[0]
        identity = translation(0)
        swap_axes = ((0, 1, 0, 0), (1, 0, 0, 0), identity[2])
        for alternative in (identity, swap_axes):
            self.assertEqual(S.compare(body, [TRUE], [identity, alternative], 0), "non_discriminating")
        self.assertEqual(S.compare(body, [TRUE], [translation(2), translation(4)], 0), "unknown_no_matrix_passed")

    def test_signature_filters_short_and_sparse_primitive_runs(self):
        leaves = [(0, 0, 0, 2 * i, 2 * i + 1, 0x29) for i in range(30)]
        signature = S.signature(leaves)
        self.assertEqual(signature[0][0], 60)
        self.assertEqual(list(S.PAIR.iter_unpack(signature[0][1])), [(2*i, 2*i+1, 0) for i in range(30)])
        self.assertIsNone(S.signature(leaves[:-1]))
        leaves[-1] = (0, 0, 0, 58, 60, 0x29)
        self.assertIsNone(S.signature(leaves))

    def test_topology_requires_recorded_fourth_corner(self):
        faces = [(0, 1, 2), (2, 1, 3), (4, 5, 6)]
        self.assertTrue(S.topology_matches(faces, [(0, 1, 2)]))
        self.assertFalse(S.topology_matches(faces, [(0, 1, 0)]))
        self.assertFalse(S.topology_matches(faces, [(0, 2, 2)]))

    def test_conflicting_positions_exclude_a_topology_match(self):
        sig = ((2, S.PAIR.pack(0, 1, 2)),)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = "o g\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 3 2 4\n"
            item = {"File": "first.obj", "MeshTris": [2]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"Items": [item, dict(item, File="other.obj")]}),
                                encoding="utf-8", newline="\n")
            (root / "first.obj").write_text(obj, encoding="ascii", newline="\n")
            other = root / "other.obj"
            other.write_text(obj, encoding="ascii", newline="\n")
            self.assertIsNotNone(S.match_exports([sig], [manifest], Counter())[sig])
            other.write_text(obj.replace("v 1 1 0", "v 4 1 0"), encoding="ascii", newline="\n")
            self.assertIsNone(S.match_exports([sig], [manifest], Counter())[sig])

    def test_model_groups_are_selected_in_order(self):
        shared = [(0, 1, 2), (2, 1, 3)]
        separate = [(0, 1, 2), (3, 4, 5)]
        groups = [(None, faces) for faces in (shared, separate, shared)]
        sig = ((2, S.PAIR.pack(0, 1, 2)),) * 2
        self.assertEqual(list(S.source_candidates(sig, [0, 1, 2], groups)), [(0, 2)])
        self.assertEqual(list(S.source_candidates(sig, [0, 1], groups)), [])

    def test_empty_entry_keeps_matrix_index(self):
        body = build(LEAVES, bounds=BOUNDS)[0]
        wrapper = bytearray(256)
        struct.pack_into("<I", wrapper, 32, 256 + len(body))
        struct.pack_into("<I", wrapper, 40, len(body))
        block = wrapper + body
        desc = bytearray(256)
        struct.pack_into("<QQ", desc, 16, 3, 0x500000e0)
        struct.pack_into("<Q", desc, 56, 0x50000050)
        for i in range(3):
            struct.pack_into("<12f", desc, 80 + 48 * i, *(v for row in translation(i) for v in row))
        struct.pack_into("<3Q", desc, 224, len(block), 1 << 32, len(block))
        matrices, bodies = S.slots(desc, block * 2)
        self.assertEqual(matrices, [translation(i) for i in range(3)])
        self.assertEqual(bodies, [body, None, body])
        with self.assertRaises(ValueError):
            S.slots(desc, block)


if __name__ == "__main__":
    unittest.main()
