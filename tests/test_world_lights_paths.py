"""Tests for tools/world_lights_paths.py, built from synthetic 0x7a resources (no game data).

    python -I tests/test_world_lights_paths.py
"""
import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Imported by name (not from a spec) so that --jobs worker processes can import it too, also under spawn.
sys.path.insert(0, os.path.join(ROOT, "tools"))
import world_lights_paths as T  # noqa: E402
M = T.M
PTR = 0x50000000


def pool(block, offset=0):
    """A pool reference: 1-based block number in bits 0..11, byte offset from bit 20."""
    return block | (offset << 20)


class Resource:
    """Lay out a 0x7a resource: header, struct table, member tables, block table, then the blocks."""

    def __init__(self):
        self.structs = []   # (hash, size, [(member hash, offset, type)])
        self.blocks = []    # (hash, bytes)

    def struct(self, hash_, size, members):
        self.structs.append((hash_, size, members))

    def block(self, hash_, data):
        self.blocks.append((hash_, bytes(data)))
        return len(self.blocks)          # 1-based block number

    def build(self):
        st, it = 0x60, 0x60 + 32 * len(self.structs)
        mt = it + 16 * len(self.blocks)
        b = bytearray(mt + 16 * sum(len(m) for _, _, m in self.structs))
        b[0x10:0x14] = b"0DRP"
        b[0x14] = 0x7A
        struct.pack_into("<I", b, 0x20, 1)
        struct.pack_into("<Q", b, 0x28, PTR | st)
        struct.pack_into("<Q", b, 0x38, PTR | it)
        struct.pack_into("<HHHH", b, 0x48, len(self.structs), 0, len(self.blocks), 0)
        at = mt
        for i, (h, size, members) in enumerate(self.structs):
            struct.pack_into("<IIIIQIHH", b, st + 32 * i, h, 0, 0, 0, PTR | at, size, 0, len(members))
            for mh, off, ty in members:
                struct.pack_into("<IIBBHI", b, at, mh, off, ty, 0, 0, 0)
                at += 16
        data_at = (len(b) + 15) & ~15
        b.extend(bytes(data_at - len(b)))
        for i, (h, data) in enumerate(self.blocks):
            struct.pack_into("<IIQ", b, it + 16 * i, h, len(data), PTR | len(b))
            b.extend(data)
            b.extend(bytes((-len(b)) % 16))
        return bytes(b)


LIGHT_SIZE = 0x40
LIGHT_MEMBERS = [
    (M["position120"], 0x00, 0x32), (M["posn"], 0x0C, 0x50), (M["direction"], 0x18, 0x50),
    (M["intensity"], 0x24, 0x21), (M["falloff"], 0x28, 0x21), (M["coneInnerAngle"], 0x2C, 0x21),
    (M["coneOuterAngle"], 0x30, 0x21), (M["timeFlags"], 0x34, 0x15), (M["lightType"], 0x38, 0x11),
    (M["colour"], 0x39, 0x50),
]
EXT_SIZE = 0x30
EXT_MEMBERS = [(M["name"], 0x00, 0x4A), (M["offsetPosition"], 0x04, 0x33), (M["instances"], 0x10, 0x52),
               (M["instances2"], 0x20, 0x52)]


def light(pos, kind=2, direction=(0.0, -1.0, 0.0), time=0xFFFFFF, colour=b"\xff\x80\x26", posn=(0.0, 0.0, 0.0)):
    d = bytearray(LIGHT_SIZE)
    struct.pack_into("<3f", d, 0x00, *pos)
    struct.pack_into("<3f", d, 0x0C, *posn)
    struct.pack_into("<3f", d, 0x18, *direction)
    struct.pack_into("<4f", d, 0x24, 50.0, 6.0, 30.0, 60.0)
    struct.pack_into("<I", d, 0x34, time)
    d[0x38] = kind
    d[0x39:0x3C] = colour
    return d


def light_resource(archetype=0xA1B2C3D4, *, posn_only: bool = False):
    """An archetype resource: one archetype with a light effect holding two lights (one per list)."""
    r = Resource()
    arch_members = [(M["name"], 0x00, 0x4A), (M["bbMin"], 0x04, 0x32), (M["bbMax"], 0x10, 0x32),
                    (M["extensions"], 0x20, 0x52)]
    r.struct(T.ARCHETYPE, 0x30, arch_members)
    r.struct(T.LIGHT_EXT, EXT_SIZE, EXT_MEMBERS)
    light_members = [m for m in LIGHT_MEMBERS if not posn_only or m[0] != M["position120"]]
    r.struct(T.LIGHT, LIGHT_SIZE, light_members)
    # blocks: 1 archetype, 2 pointer list, 3 effect, 4 lights
    arch = bytearray(0x30)
    struct.pack_into("<I3f3f", arch, 0, archetype, -1.0, -1.0, 0.0, 1.0, 1.0, 3.0)
    struct.pack_into("<QHH", arch, 0x20, pool(2), 1, 1)
    ptrs = struct.pack("<Q", pool(3))
    eff = bytearray(EXT_SIZE)
    struct.pack_into("<I3f", eff, 0, 0x12345678, 0.0, 0.0, 0.0)
    struct.pack_into("<QHH", eff, 0x10, pool(4, 0), 1, 1)
    struct.pack_into("<QHH", eff, 0x20, pool(4, LIGHT_SIZE), 1, 1)
    lights = light((0.0, 0.5, 2.5)) + light((0.0, 0.0, 1.0), kind=1, time=0x00007F, posn=(0.0, 0.1, 0.0))
    for h, data in ((T.ARCHETYPE, arch), (7, ptrs), (T.LIGHT_EXT, eff), (T.LIGHT, lights)):
        r.block(h, data)
    return r.build()


def map_resource(scale_xy: float = 2.0, scale_z: float = 3.0, *, posn_only: bool = False) -> bytes:
    """A map resource with one entity that owns a light effect (one light, offset 1 m up)."""
    r = Resource()
    ent_members = [(M["position"], 0x00, 0x32), (M["rotation"], 0x10, 0x34), (M["scaleXY"], 0x20, 0x21),
                   (M["scaleZ"], 0x24, 0x21), (M["archetypeName"], 0x28, 0x4A), (M["extensions"], 0x30, 0x52)]
    r.struct(T.ENTITY, 0x40, ent_members)
    r.struct(T.MAPDATA, 0x10, [(M["name"], 0x00, 0x4A)])
    r.struct(T.LIGHT_EXT, EXT_SIZE, EXT_MEMBERS)
    light_members = [m for m in LIGHT_MEMBERS if not posn_only or m[0] != M["position120"]]
    r.struct(T.LIGHT, LIGHT_SIZE, light_members)
    ent = bytearray(0x40)
    s = math.sqrt(0.5)
    struct.pack_into("<3f", ent, 0x00, 100.0, 200.0, 10.0)
    struct.pack_into("<4f", ent, 0x10, 0.0, 0.0, s, s)          # stored +90 degrees about Z: the entity turns -90
    struct.pack_into("<2f", ent, 0x20, scale_xy, scale_z)
    struct.pack_into("<I", ent, 0x28, 0xCAFEF00D)
    struct.pack_into("<QHH", ent, 0x30, pool(3), 1, 1)
    eff = bytearray(EXT_SIZE)
    struct.pack_into("<I3f", eff, 0, 0, 0.0, 0.0, 1.0)
    struct.pack_into("<QHH", eff, 0x10, pool(5), 1, 1)
    posn = (1.0, 0.0, 0.0) if posn_only else (0.0, 0.0, 0.0)
    placed_light = light((1.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), posn=posn)
    for h, data in ((T.ENTITY, ent), (T.MAPDATA, bytes(0x10)), (7, struct.pack("<Q", pool(4))),
                    (T.LIGHT_EXT, eff), (T.LIGHT, placed_light)):
        r.block(h, data)
    return r.build()


NODE_SIZE, NL_SIZE, LINK_SIZE = 0x30, 0x3C, 0x14


def node_resource(nodes):
    """nodes: [(address index, key index, first link, num links, flags, speed mph)]."""
    r = Resource()
    r.struct(T.NODE_LIST, NL_SIZE, [(M["node"], 0x00, 0x05), (M["list_address"], 0x34, 0x05),
                                     (M["first_link"], 0x38, 0x15)])
    r.struct(T.NODE, NODE_SIZE, [
        (M["node_position"], 0x00, 0x32), (M["node_flags"], 0x0C, 0x65), (M["address"], 0x14, 0x05),
        (M["key"], 0x18, 0x05), (M["streetNameHash"], 0x20, 0x4A), (M["speed"], 0x24, 0x20),
        (M["kind"], 0x2A, 0x60), (M["road_class"], 0x2B, 0x60), (M["num_links"], 0x2C, 0x11),
        (M["density"], 0x2D, 0x11)])
    data = bytearray()
    for i, (addr, key, first, num, flags, mph) in enumerate(nodes):
        rec = bytearray(NL_SIZE)
        struct.pack_into("<3fQ", rec, 0, float(i), 0.0, 0.0, flags)
        struct.pack_into("<HHHH", rec, 0x14, 0, addr, 0, key)
        struct.pack_into("<e", rec, 0x24, mph * 0.44704)
        rec[0x2A], rec[0x2B], rec[0x2C], rec[0x2D] = 1, 2, num, 3
        struct.pack_into("<I", rec, 0x38, first)
        data += rec
    r.block(T.NODE_LIST, data)
    return r.build()


def link_resource(links):
    """links: [(target address index, lanes to, lanes from, flags, length)]."""
    r = Resource()
    r.struct(T.LINK, LINK_SIZE, [(M["target"], 0x00, 0x05), (M["length"], 0x04, 0x13),
                                 (M["lanes_to"], 0x06, 0x11), (M["lanes_from"], 0x07, 0x11),
                                 (M["link_flags"], 0x0C, 0x65)])
    data = bytearray()
    for to, lt, lf, flags, length in links:
        rec = bytearray(LINK_SIZE)
        struct.pack_into("<HHHBB", rec, 0, 0, to, length, lt, lf)
        struct.pack_into("<I", rec, 0x0C, flags)
        data += rec
    r.block(T.LINK, data)
    return r.build()


def junction_resource(node_key, entrance_keys, phases):
    r = Resource()
    r.struct(T.JUNCTION, 0x30, [(M["junction_node"], 0x00, 0x05), (M["entrances"], 0x08, 0x5C),
                                (M["phases"], 0x18, 0x5C)])
    r.struct(T.ENTRANCE, 0x20, [(M["turns"], 0x00, 0x5C), (M["entrance_node"], 0x10, 0x05),
                                (M["bIsStop"], 0x14, 0x01), (M["bIsYield"], 0x15, 0x01)])
    r.struct(T.TURN, 0x10, [(M["entrance_node"], 0x00, 0x05), (M["fPathLength"], 0x04, 0x21)])
    r.struct(T.PHASE, 0x18, [(M["fDuration"], 0x00, 0x21), (M["lightBoxes"], 0x08, 0x5C)])
    j = bytearray(0x30)
    struct.pack_into("<HH", j, 0, 0, node_key)
    struct.pack_into("<IIQ", j, 0x08, len(entrance_keys), len(entrance_keys), pool(2))
    struct.pack_into("<IIQ", j, 0x18, len(phases), len(phases), pool(4))
    ents = bytearray()
    for i, k in enumerate(entrance_keys):
        e = bytearray(0x20)
        struct.pack_into("<IIQ", e, 0, 1, 1, pool(3, 0x10 * i))
        struct.pack_into("<HHBB", e, 0x10, 0, k, i == 0, 0)
        ents += e
    turns = b"".join(struct.pack("<HHf8x", 0, node_key, 12.5) for _ in entrance_keys)
    ph = b"".join(struct.pack("<fIIIQ", d, 0, 0, 0, 0) for d in phases)
    for h, data in ((T.JUNCTION, j), (T.ENTRANCE, ents), (T.TURN, turns), (T.PHASE, ph)):
        r.block(h, data)
    return r.build()


def write_rpf(path, resources):
    """An RPF9-like file: '9FPR' header with the data start at +4, then 16-aligned zlib streams."""
    out = bytearray(b"9FPR" + struct.pack("<I", 0x40))
    out.extend(bytes(0x40 - len(out)))
    for data in resources:
        out.extend(zlib.compress(data, 9))
        out.extend(bytes((-len(out)) % 16))
        out.extend(bytes(16))
    with open(path, "wb") as f:
        f.write(out)


GRAPH_NODES = [  # address, key, first link, num links, flags, mph
    (0, 100, 0, 1, 1 << 4, 25),
    (1, 101, 1, 2, 0, 25),
    (2, 102, 3, 1, 0, 40),
    (3, 103, 4, 0, 0, 10),
]
GRAPH_LINKS = [(1, 1, 1, 1 << 10, 10), (0, 1, 1, 1 << 10, 20), (2, 1, 0, 0, 30), (9, 1, 0, 0, 40)]


class MetaTest(unittest.TestCase):
    def test_pool_reference_and_system_pointer(self):
        meta = T.Meta(light_resource())
        name, at = meta.ref(pool(4, LIGHT_SIZE))
        self.assertEqual(name, T.LIGHT)
        self.assertEqual(at, meta.blocks[3][2] + LIGHT_SIZE)
        self.assertIsNone(meta.ref(pool(4, 0x10000)))                # past the block
        self.assertIsNone(meta.ref(pool(9)))                         # no such block
        self.assertEqual(meta.ref(PTR | 0x60), (None, 0x60))

    def test_pool_offset_above_32_bits(self):
        r = Resource()
        r.struct(T.LIGHT, 1, [])
        r.block(T.LIGHT, bytes(0x2000))
        meta = T.Meta(r.build())
        self.assertEqual(meta.ref(pool(1, 0x1040))[1], meta.blocks[0][2] + 0x1040)
        self.assertGreater(pool(1, 0x1040), 0xFFFFFFFF)

    def test_pool_reference_shaped_like_a_system_pointer(self):
        # Block 30, offset 0x500: the value is 0x5000001e, which also reads as a 0x5xxxxxxx pointer to offset 0x1e.
        r = Resource()
        r.struct(T.LIGHT, 1, [])
        for _ in range(30):
            r.block(T.LIGHT, bytes(0x600))
        meta = T.Meta(r.build())
        v = pool(30, 0x500)
        self.assertEqual(v, 0x5000001E)
        self.assertEqual(meta.ref(v), (T.LIGHT, meta.blocks[29][2] + 0x500))
        self.assertEqual(meta.system_shaped, 1)

    def test_not_a_resource(self):
        with self.assertRaises(ValueError):
            T.Meta(bytes(0x60))


class LightTest(unittest.TestCase):
    def test_archetype_lights_from_both_lists(self):
        got = T.inspect_resource(light_resource())
        st = got["lights"]
        self.assertEqual((st["effects"], st["effects_archetype"], st["lights"]), (1, 1, 2))
        self.assertEqual((st["list_instances"], st["list_instances2"]), (1, 1))
        self.assertEqual((st["type_spot"], st["type_point"]), (1, 1))
        self.assertEqual((st["time_always"], st["time_never_daytime"]), (1, 1))
        self.assertEqual(st["position120_inside_bbox"], 2)
        self.assertEqual(st["posn_equals_position120"], 0)
        lights = got["archetype_lights"][0xA1B2C3D4]
        self.assertEqual([L["position120"] for L in lights], [[0.0, 0.5, 2.5], [0.0, 0.0, 1.0]])
        self.assertEqual(lights[0]["colour"], "ff8026")
        self.assertEqual(lights[0]["coneOuterAngle"], 60.0)

    def test_entity_light_is_placed_with_the_conjugated_rotation_and_scale(self):
        got = T.inspect_resource(map_resource(), lights_out=True)
        self.assertEqual(got["lights"]["lights_entity_in_map_resources"], 1)
        (row,) = got["world_lights"]
        # local (1, 0, 0) + offset (0, 0, 1), scaled (2, 0, 3), rotated -90 degrees about Z -> (0, -2, 3).
        for a, b in zip(row["position"], (100.0, 198.0, 13.0)):
            self.assertAlmostEqual(a, b, places=3)
        for a, b in zip(row["direction"], (0.0, -1.0, 0.0)):
            self.assertAlmostEqual(a, b, places=4)
        self.assertEqual(row["archetype"], "cafef00d")

    def test_export_normalizes_the_stored_light_rotation(self) -> None:
        resource = bytearray(map_resource())
        meta = T.Meta(resource)
        (entity_offset,) = meta.objects(T.ENTITY)
        rotation_offset = meta.off(T.ENTITY, "rotation")
        assert rotation_offset is not None
        struct.pack_into("<4f", resource, entity_offset + rotation_offset, 0.0, 0.0, 0.5, 0.5)

        (row,) = T.inspect_resource(resource, lights_out=True)["world_lights"]
        self.assertEqual(row["position"], [100.0, 198.0, 13.0])
        self.assertEqual(row["direction"], [0.0, -1.0, 0.0])

    def test_place_identity(self):
        p, d = T.place([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0, 1.0)
        self.assertEqual(p, [11.0, 2.0, 3.0])
        self.assertEqual(d, [0.0, 0.0, 1.0])

    def test_unreferenced_effect_is_counted(self):
        r = Resource()
        r.struct(T.LIGHT_EXT, EXT_SIZE, EXT_MEMBERS)
        r.struct(T.LIGHT, LIGHT_SIZE, LIGHT_MEMBERS)
        eff = bytearray(EXT_SIZE)
        struct.pack_into("<QHH", eff, 0x10, pool(2), 1, 1)
        r.block(T.LIGHT_EXT, eff)
        r.block(T.LIGHT, light((0.0, 0.0, 0.0)))
        st = T.inspect_resource(r.build())["lights"]
        self.assertEqual((st["effects_unreferenced"], st["lights_unreferenced"]), (1, 1))


class GraphTest(unittest.TestCase):
    def graph(self):
        nodes = T.read_nodes(T.Meta(node_resource(GRAPH_NODES)))
        links = T.read_links(T.Meta(link_resource(GRAPH_LINKS)))
        junctions = T.read_junctions(T.Meta(junction_resource(100, [101, 102, 555], [8.0, 30.0])))
        return nodes, links, junctions

    def test_records(self):
        nodes, links, junctions = self.graph()
        self.assertEqual([n["address"] for n in nodes], [(0, 0), (0, 1), (0, 2), (0, 3)])
        self.assertAlmostEqual(nodes[2]["speed"], 40 * 0.44704, delta=0.01)   # f16 precision
        self.assertEqual(links[0]["lanes_to"], 1)
        self.assertEqual(len(junctions[0]["entrances"]), 3)
        self.assertTrue(junctions[0]["entrances"][0]["stop"])
        self.assertEqual([p["duration"] for p in junctions[0]["phases"]], [8.0, 30.0])

    def test_stats(self):
        s, hist = T.graph_stats(*self.graph())
        self.assertEqual((s["nodes"], s["links"], s["links_owned"]), (4, 4, 4))
        self.assertEqual(s["links_unresolved"], 1)                    # target address 9 does not exist
        self.assertEqual(s["links_with_reverse"], 2)                  # 0 -> 1 and 1 -> 0
        self.assertEqual(s["num_links_matches_range"], 4)
        self.assertEqual((s["components"], s["largest_component_nodes"]), (2, 3))
        self.assertEqual(s["isolated_nodes"], 1)
        self.assertEqual((s["junction_nodes_resolved"], s["junction_entrances"]), (1, 3))
        self.assertEqual(s["junction_entrances_resolved"], 2)
        self.assertEqual((s["signalised_junctions"], s["phases"], s["entrances_stop"]), (1, 2, 1))
        self.assertEqual(s["junction_nodes_flagged_junction"], 1)
        self.assertEqual(hist["link_flag_bits"][10], 2)
        self.assertEqual(hist["lanes_from"][0], 2)
        # Even-sized samples take the stdlib median (the mean of the two middle
        # values), not the upper-middle element lengths[len // 2] used before.
        self.assertEqual(hist["link_length_m"], {"min": 10, "median": 25.0, "max": 40})
        self.assertEqual(hist["phase_duration_s"], {"min": 8.0, "median": 19.0, "max": 30.0})


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rpf = os.path.join(self.tmp.name, "levels.rpf")
        write_rpf(self.rpf, [light_resource(), map_resource(), node_resource(GRAPH_NODES),
                             link_resource(GRAPH_LINKS), junction_resource(100, [101], [5.0])])
        self.entities = os.path.join(self.tmp.name, "entities.jsonl")
        with open(self.entities, "w", encoding="utf-8", newline="\n") as f:
            for x in (0.0, 50.0):
                f.write(json.dumps({"archetype": "a1b2c3d4", "position": [x, 0.0, 0.0],
                                    "rotation": [0.0, 0.0, 0.0, 1.0], "scale_xy": 1.0, "scale_z": 1.0}) + "\n")
            f.write(json.dumps({"archetype": "0badbeef", "position": [0, 0, 0], "rotation": [0, 0, 0, 1]}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_truncated_rpf9_header_is_one_error_line(self) -> None:
        # Before: scan() unpacked the data-start word from an unchecked 8-byte head, so a 5-byte "9FPR" file
        # died in a struct.error traceback. A truncated header is a usage error: one ERROR line, exit 2.
        short = os.path.join(self.tmp.name, "short.rpf")
        with open(short, "wb") as f:
            f.write(b"9FPR\x01")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = T.main([short])
        self.assertEqual(rc, 2)
        self.assertIn(f"ERROR: {short}: the RPF9 header is 5 bytes, the format needs 8", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        self.assertNotIn("struct.error", err.getvalue())

    def test_summary_write_failure_is_one_error_line(self) -> None:
        # --out naming a directory: IsADirectoryError/PermissionError on most systems. The input half
        # of main already answers with one ERROR line; the write half must too.
        target = os.path.join(self.tmp.name, "as-a-directory")
        os.makedirs(target)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = T.main([self.rpf, "--entities", self.entities, "--out", target])
        self.assertEqual(rc, 2)
        self.assertIn("ERROR: cannot write", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        self.assertNotIn("IsADirectoryError", err.getvalue())
        self.assertNotIn("PermissionError", err.getvalue())

    def test_lights_out_uses_posn_when_position120_is_absent(self) -> None:
        archive = os.path.join(self.tmp.name, "posn-only.rpf")
        write_rpf(archive, [light_resource(posn_only=True), map_resource(posn_only=True)])
        output = os.path.join(self.tmp.name, "posn-only.jsonl")
        summary = os.path.join(self.tmp.name, "posn-only-summary.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(T.main([archive, "--entities", self.entities, "--out", summary,
                                     "--lights-out", output]), 0)
        with open(output, encoding="utf-8") as handle:
            placements = [json.loads(line) for line in handle]
        self.assertEqual([row["position"] for row in placements if row["owner"] == "entity"],
                         [[100.0, 198.0, 13.0]])
        self.assertEqual([row["position"] for row in placements if row["owner"] == "archetype"],
                         [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [50.0, 0.0, 0.0], [50.0, 0.1, 0.0]])

    def test_scan_and_summary(self):
        total = T.scan(self.rpf)
        summary = T.summarize(total, self.entities)
        self.assertEqual(summary["light_resources"], 2)
        self.assertEqual(summary["lights"]["lights"], 3)
        self.assertEqual(summary["archetypes_with_lights"], 1)
        self.assertEqual(summary["placement"]["entities"], 3)
        self.assertEqual(summary["placement"]["entities_with_lights"], 2)
        self.assertEqual(summary["placement"]["archetype_lights_placed"], 4)
        self.assertEqual(summary["lane_graph"]["nodes"], 4)
        self.assertEqual(summary["lane_graph"]["junctions"], 1)
        self.assertEqual(summary["graph_resources"], {"junctions": 1, "links": 1, "nodes": 1})

    def test_duplicate_light_definitions_compare_values_not_just_counts(self):
        total = T.scan(self.rpf)
        total["archetype_lights"] = {
            1: [[{"posn": [0, 1, 0]}], [{"posn": [0, 1, 0]}]],
            2: [[{"posn": [0, 1, 0]}], [{"posn": [1, 0, 0]}]],
            3: [[], [{"intensity": 2}]],
            4: [[{"intensity": 2}]],
        }
        summary = T.summarize(total)
        self.assertEqual(summary["archetypes_defined_more_than_once"], 3)
        self.assertEqual(summary["archetype_light_definitions_differ"], 2)

    def test_jobs_give_the_same_summary(self):
        one = T.summarize(T.scan(self.rpf, jobs=1))
        two = T.summarize(T.scan(self.rpf, jobs=2))
        self.assertEqual(one, two)

    def test_cli_rejects_negative_start_before_scanning(self):
        for start in ("-1", "-0x10"):
            with self.subTest(start=start):
                out = os.path.join(self.tmp.name, "summary.json")
                err = io.StringIO()
                with mock.patch.object(T, "scan", side_effect=AssertionError("unexpected scan")) as scan, \
                        contextlib.redirect_stderr(err):
                    rc = T.main([self.rpf, f"--start={start}", "--out", out])
                scan.assert_not_called()
                self.assertEqual(rc, 2)
                self.assertIn("--start must be non-negative", err.getvalue())
                self.assertFalse(os.path.exists(out))

    def test_cli_rejects_shared_report_destination(self):
        out = os.path.join(self.tmp.name, "lights.json")
        alias = os.path.join(self.tmp.name, ".", "lights.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            T.main([self.rpf, "--out", out, "--lights-out", alias])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--out and --lights-out must use different paths", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_cli_rejects_case_alias_before_scan_on_a_case_insensitive_volume(self):
        marker = os.path.join(self.tmp.name, "case-check")
        with open(marker, "w", encoding="utf-8"):
            pass
        if not os.path.exists(os.path.join(self.tmp.name, "CASE-CHECK")):
            self.skipTest("test directory is case-sensitive")
        original_files = set(os.listdir(self.tmp.name))

        out = os.path.join(self.tmp.name, "lights.json")
        alias = os.path.join(self.tmp.name, "LIGHTS.JSON")
        err = io.StringIO()
        # Exercise case-insensitive lookup without Windows's lexical normalization.
        with mock.patch.object(T.os.path, "normcase", side_effect=lambda path: path), \
                mock.patch.object(T, "scan", side_effect=AssertionError("scan started before rejecting the alias")) as scan, \
                contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as cm:
            T.main([self.rpf, "--out", out, "--lights-out", alias])
        scan.assert_not_called()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--out and --lights-out must use different paths", err.getvalue())
        self.assertEqual(set(os.listdir(self.tmp.name)), original_files)

    def test_cli_writes_summary_and_placed_lights(self):
        out = os.path.join(self.tmp.name, "summary.json")
        lights = os.path.join(self.tmp.name, "lights.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = T.main([self.rpf, "--out", out, "--entities", self.entities, "--lights-out", lights])
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as f:
            summary = json.load(f)
        limited = os.path.join(self.tmp.name, "limited.json")
        limited_lights = os.path.join(self.tmp.name, "limited.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(T.main([self.rpf, "--start", "0", "--limit", "0x1000", "--out", limited,
                                     "--entities", self.entities, "--lights-out", limited_lights]), 0)
        with open(limited, encoding="utf-8") as f:
            self.assertEqual(json.load(f), summary)
        with open(lights, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1 + 4)                            # one entity light, 2 x 2 archetype lights
        self.assertEqual(sorted(r["owner"] for r in rows), ["archetype"] * 4 + ["entity"])
        self.assertEqual(summary["placement"]["archetype_lights_placed"], 4)
        xs = sorted(r["position"][0] for r in rows if r["owner"] == "archetype")
        self.assertEqual(xs, [0.0, 0.0, 50.0, 50.0])

    def test_cli_rejects_report_aliasing_an_input(self):
        with open(self.rpf, "rb") as f:
            rpf_bytes = f.read()
        with open(self.entities, "rb") as f:
            entities_bytes = f.read()
        cases = [
            (["--out", self.rpf], "--out must use a different path than archive"),
            (["--out", self.entities], "--out must use a different path than --entities"),
            (["--lights-out", self.rpf], "--lights-out must use a different path than archive"),
            (["--lights-out", self.entities], "--lights-out must use a different path than --entities"),
        ]
        for args, msg in cases:
            err = io.StringIO()
            with mock.patch.object(T, "scan", side_effect=AssertionError("scan started before rejecting the alias")) as scan, \
                    contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                T.main([self.rpf, "--entities", self.entities] + args)
            scan.assert_not_called()
            self.assertEqual(cm.exception.code, 2)
            self.assertIn(msg, err.getvalue())
        with open(self.rpf, "rb") as f:
            self.assertEqual(f.read(), rpf_bytes)
        with open(self.entities, "rb") as f:
            self.assertEqual(f.read(), entities_bytes)

    def test_light_export_preserves_zero_entity_scale(self) -> None:
        write_rpf(self.rpf, [light_resource(), map_resource(scale_xy=0.0, scale_z=0.0)])
        lights = os.path.join(self.tmp.name, "zero-scale.jsonl")
        with open(self.entities, "w", encoding="utf-8", newline="\n") as f:
            for x, scales in ((10, {"scale_xy": 0.0, "scale_z": 0.0}), (20, {})):
                f.write(json.dumps({"archetype": "a1b2c3d4", "position": [x, 0, 0],
                                    "rotation": [0, 0, 0, 1], **scales}) + "\n")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(T.main([self.rpf, "--entities", self.entities, "--lights-out", lights]), 0)
        with open(lights, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        self.assertEqual([row["owner"] for row in rows], ["entity", "archetype", "archetype",
                                                         "archetype", "archetype"])
        for row, position in zip(rows, ([100.0, 200.0, 10.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                                        [20.0, 0.5, 2.5], [20.0, 0.0, 1.0])):
            with self.subTest(owner=row["owner"], expected=position):
                self.assertEqual(row["position"], position)

    def test_bad_entities_leaves_the_previous_lights_report_alone(self):
        # --lights-out used to be opened (truncating it) before --entities was parsed.
        lights = os.path.join(self.tmp.name, "lights.jsonl")
        with open(lights, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"old": true}\n')
        bad = os.path.join(self.tmp.name, "bad.jsonl")
        with open(bad, "w", encoding="utf-8", newline="\n") as f:
            f.write("{not json\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = T.main([self.rpf, "--entities", bad, "--lights-out", lights])
        self.assertEqual(rc, 2)
        with open(lights, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"old": true}\n')
        self.assertFalse(os.path.exists(lights + ".partial"), "a failed run left its staged report behind")
        lines = err.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("bad.jsonl", lines[0])
        self.assertNotIn("Traceback", err.getvalue())

    def test_lights_report_that_is_a_directory_leaves_no_partial(self):
        lights = os.path.join(self.tmp.name, "lights.jsonl")
        os.makedirs(lights)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = T.main([self.rpf, "--entities", self.entities, "--lights-out", lights])
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(lights + ".partial"), "a failed replace left its staged report behind")
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_bad_input_is_one_line(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(T.main([os.path.join(self.tmp.name, "missing.rpf")]), 2)
            self.assertEqual(T.main([self.rpf, "--jobs", "0"]), 2)
            self.assertEqual(T.main([self.rpf, "--entities", os.path.join(self.tmp.name, "none.jsonl")]), 2)
            self.assertEqual(T.main([self.rpf, "--limit", "0"]), 2)
            self.assertEqual(T.main([self.rpf, "--limit", "-1"]), 2)
        lines = err.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[3], "ERROR: --limit must be at least 1")
        self.assertEqual(lines[4], "ERROR: --limit must be at least 1")


    def test_cli_refuses_malformed_entities_rows_before_the_scan(self):
        good = '{"archetype": "a1b2c3d4", "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}\n'
        cases = [
            ("badjson", '{"archetype": "a1b2c3d4", "position": [0,0,0]\n', "not valid JSON"),
            ("notjson", "garbage line\n", "not valid JSON"),
            ("nonobject", "[1, 2, 3]\n", "not a JSON object"),
            ("scalar", "123\n", "not a JSON object"),
            ("badhex", '{"archetype": "zzz"}\n', "archetype is not a hex hash"),
            ("unhashable", '{"archetype": {"x": 1}}\n', "archetype is not a hash:"),
            ("shortrot", good.replace("[0.0, 0.0, 0.0, 1.0]", "[0.0, 0.0]"), "rotation is not 4 numbers"),
            ("scalarpos", good.replace('"position": [0.0, 0.0, 0.0]', '"position": 5'),
             "position is not 3 numbers"),
            ("boolpos", good.replace('"position": [0.0, 0.0, 0.0]', '"position": [true, 0.0, 0.0]'),
             "position is not 3 numbers"),
            ("emptypos", good.replace('"position": [0.0, 0.0, 0.0]', '"position": []'),
             "position is not 3 numbers"),
            ("strscale", good.replace('1.0]}\n', '1.0], "scale_xy": "2"}\n'), "scale_xy is not a number"),
        ]
        for name, line, expected in cases:
            with self.subTest(case=name):
                ent = os.path.join(self.tmp.name, name + ".jsonl")
                with open(ent, "w", encoding="utf-8", newline="\n") as f:
                    f.write(line)
                err = io.StringIO()
                with mock.patch.object(T, "scan",
                                       side_effect=AssertionError("scan started before refusing the entities")
                                       ) as scan, contextlib.redirect_stderr(err):
                    self.assertEqual(T.main([self.rpf, "--entities", ent]), 2)
                scan.assert_not_called()
                lines = err.getvalue().strip().splitlines()
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("ERROR: %s:1:" % ent), lines[0])
                self.assertIn(expected, lines[0])

    def test_cli_refuses_non_utf8_entities_file(self):
        ent = os.path.join(self.tmp.name, "binary.jsonl")
        with open(ent, "wb") as f:
            f.write(b"\xff\xfe binary garbage\n")
        err = io.StringIO()
        with mock.patch.object(T, "scan",
                               side_effect=AssertionError("scan started before refusing the entities")) as scan, \
                contextlib.redirect_stderr(err):
            self.assertEqual(T.main([self.rpf, "--entities", ent]), 2)
        scan.assert_not_called()
        self.assertEqual(err.getvalue().strip().splitlines(), ["ERROR: %s: not UTF-8 text" % ent])

    def test_malformed_entities_write_no_lights_file_or_summary(self):
        ent = os.path.join(self.tmp.name, "late-bad.jsonl")
        with open(ent, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"archetype": "a1b2c3d4", "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}\n')
            f.write('{"archetype": "zzz"}\n')
        out = os.path.join(self.tmp.name, "never.json")
        lights = os.path.join(self.tmp.name, "never.jsonl")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(T.main([self.rpf, "--out", out, "--entities", ent, "--lights-out", lights]), 2)
        self.assertIn("archetype is not a hex hash", err.getvalue())
        self.assertFalse(os.path.exists(lights))
        self.assertFalse(os.path.exists(out))

    def test_entities_rows_with_null_fields_and_blank_lines_are_accepted(self):
        ent = os.path.join(self.tmp.name, "nulls.jsonl")
        with open(ent, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n")
            f.write(json.dumps({"archetype": "a1b2c3d4", "position": None, "rotation": None,
                                "scale_xy": None, "scale_z": None}) + "\n")
        out = os.path.join(self.tmp.name, "nulls.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(T.main([self.rpf, "--out", out, "--entities", ent]), 0)
        with open(out, encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["placement"]["entities"], 1)
        self.assertEqual(summary["placement"]["entities_with_lights"], 1)
        self.assertEqual(summary["placement"]["archetype_lights_placed"], 2)


class GraphStatsFirstLinkTest(unittest.TestCase):
    @staticmethod
    def _node(idx, first_link):
        return {
            "address": (0, idx), "key": 100 + idx, "first_link": first_link, "num_links": 0,
            "flags": 0, "road_class": 0, "kind": 0, "density": 0,
            "speed": 10.0, "street": "", "position": [0.0, 0.0, 0.0],
        }

    def test_graph_stats_handles_none_first_link_without_type_error(self):
        # Integer followed by None used to crash with TypeError in firsts[i] <= firsts[i + 1].
        s_mixed, _ = T.graph_stats([self._node(0, 0), self._node(1, None)], [], [])
        self.assertEqual(s_mixed["first_link_monotonic"], 0)
        # A single node with None first_link must not vacuously report monotonic=1.
        s_single_none, _ = T.graph_stats([self._node(0, None)], [], [])
        self.assertEqual(s_single_none["first_link_monotonic"], 0)
        s_leading_none, _ = T.graph_stats([self._node(0, None), self._node(1, 0)], [], [])
        self.assertEqual(s_leading_none["first_link_monotonic"], 0)
        s_ok, _ = T.graph_stats([self._node(0, 0), self._node(1, 0), self._node(2, 2)], [], [])
        self.assertEqual(s_ok["first_link_monotonic"], 1)


if __name__ == "__main__":
    unittest.main()
