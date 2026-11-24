"""Count and resolve hand-built SGD5 effects with python -I tests/test_sgd5_effects.py."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sgd5_effects
import shader_carve


def u32(value):
    return struct.pack("<I", value)


def label(text):
    value = text.encode("ascii") + b"\0"
    return bytes([len(value)]) + value


def program(name):
    # A synthetic DXBC with a single DXIL part and no instructions.
    payload = struct.pack("<II4s", 0x60, 3, b"DXIL")
    dxbc = b"DXBC" + bytes(16) + struct.pack("<IIII", 1, 44 + len(payload), 1, 36)
    dxbc += b"DXIL" + u32(len(payload)) + payload
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(dxbc) + compressor.flush()
    stored = 12 + len(compressed)
    packed = struct.pack("<III", 0xDDCE7, stored, len(dxbc)) + compressed
    return label(name) + b"\x02" + u32(stored) + packed + bytes(152) + u32(2) + b"\xaa\xbb"


def post_pass(words=(), blocks=(b"", b""), packed=0, entries=b""):
    data = u32(len(words)) + b"".join(u32(word) for word in words) + u32(0x80000000)
    return data + b"".join(u32(len(block)) + block for block in blocks) + u32(packed) + entries


def effect(name, ordinals=None, techniques=(), pool_tail=b"", suffix=None):
    data = bytearray(label(name) + u32(0xDEADBEEF))
    for group in range(6):
        entries = (ordinals or {}).get(group, ())
        data += u32(len(entries)) + b"".join(u32(i) for i in entries)
    data += bytes(80)
    for width in (16, 24, 68, 36):
        data += u32(1) + bytes(width)

    pool = bytearray(b"\0")
    headers = bytearray()
    rows = []
    for technique, passes in techniques:
        headers += struct.pack("<HBB", len(pool) | 0x8000, 0, len(passes))
        pool += technique.encode("ascii") + b"\0"
        rows.extend(passes)
    data += u32(len(techniques)) + headers + u32(len(rows))
    for row in rows:
        data += bytes(row) + bytes(18)
    pool += pool_tail
    return data + (post_pass() if suffix is None else suffix) + u32(len(pool)) + pool


def database(groups, effects, base=0):
    data = bytearray(b"SGD5" + bytes(4))
    for group in range(6):
        names = groups.get(group, ())
        data += u32(len(names))
        for name in names:
            data += program(name)
    data += u32(len(effects))
    for item in effects:
        data += item
    data += u32(1)
    data += bytes(-(base + len(data)) % 4096)
    return data, (base, base + 8, base + len(data))


class EffectsTest(unittest.TestCase):
    def test_discover_and_resolve_via_cli(self):
        row = (1, 0, 0, 0, 0, 0)
        data, layout = database({0: ("Light",)}, [
            effect("Bulb", {0: (0,)}, [("Glow", [row]), ("Glow", [row, row])]),
            effect("Bulb"), effect("Dark"),
        ])
        with tempfile.TemporaryDirectory() as folder, patch.object(sgd5_effects, "DATABASES", [layout]):
            archive = Path(folder) / "mock.rpf"
            archive.write_bytes(data)

            def query(*args):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(sgd5_effects.main([str(archive), *args]), 0)
                return json.loads(output.getvalue())[0]["matches"]

            self.assertEqual(query(), [])
            effects = query("--list-effects")
            self.assertEqual([entry["effect"] for entry in effects], ["Bulb", "Bulb", "Dark"])
            self.assertEqual(effects[0]["shader_hash"], "0CF39D0D")
            named, empty = query("--effect", effects[0]["effect"])
            self.assertEqual(query("--shader-hash", "0CF39D0D"), [named, empty])
            self.assertEqual([named["effect_offset"], empty["effect_offset"]],
                             [entry["effect_offset"] for entry in effects[:2]])
            self.assertEqual(empty["techniques"], [])
            self.assertEqual([(t["technique_index"], t["pass_count"]) for t in named["techniques"]], [(0, 1), (1, 2)])
            resolved = query("--effect", named["effect"], "--technique", named["techniques"][0]["technique"])
            self.assertEqual(query("--shader-hash", "0xcf39d0d", "--technique", "Glow"), resolved)
            self.assertEqual([len(match["passes"]) for match in resolved], [1, 2])
            self.assertEqual(resolved[0]["passes"][0][0]["record_offset"], data.index(label("Light")))
            for bad in [("--technique", "Glow"), ("--list-effects", "--effect", "Bulb"), ("--effect", "Absent"),
                        ("--shader-hash", "00000000"), ("--shader-hash", "100000000"), ("--shader-hash", "oops")]:
                with self.subTest(bad=bad), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    query(*bad)

    def test_material_hash_preserves_all_matching_effect_records(self):
        data, layout = database({}, [effect("Bulb"), effect("bulb")])
        report = sgd5_effects.read_database(io.BytesIO(data), layout, shader_hash=0x0CF39D0D)
        self.assertEqual([m["effect"] for m in report["matches"]], ["Bulb", "bulb"])
        self.assertNotEqual(report["matches"][0]["effect_offset"], report["matches"][1]["effect_offset"])

    def test_duplicate_names_resolve_per_effect_technique_and_pass(self):
        # Both length bytes are printable. The carver includes the program's
        # length byte in its match, and the effect's length looks like a letter.
        name = "MockLamp" * 8
        bright = "Glow" * 8
        none = (0, 0, 0, 0, 0, 0)
        both = (1, 0, 0, 0, 0, 1)
        spare = (2, 0, 0, 0, 0, 0)
        first = effect(name, {0: (1, 0), 5: (0,)}, [("Hue", [both, none]), ("Hue", [spare])])
        second = effect(name, {5: (0,)}, [("Hue", [(0, 0, 0, 0, 0, 1)])])
        data, layout = database({0: ("Spare", bright), 5: (bright,)}, [first, effect("Vacant"), second])
        report = sgd5_effects.read_database(io.BytesIO(data), layout, name, "Hue")
        self.assertEqual(report["groups"], [2, 0, 0, 0, 0, 1])
        self.assertEqual((report["effects"], report["named_effects"], report["techniques"], report["passes"]),
                         (3, 2, 3, 4))
        first_at = data.index(first)
        second_at = data.index(second)
        carve = [m.start() for m in shader_carve.ENTRY.finditer(data, 0, first_at - 4)]
        self.assertEqual(len(carve), 3)
        selected = (0, 1, data.index(label(bright)), carve[1])
        other_group = (5, 0, data.index(label(bright), selected[2] + 1), carve[2])
        spare_program = (0, 0, data.index(label("Spare")), carve[0])
        self.assertEqual(carve[1], selected[2])
        self.assertEqual(carve[0], spare_program[2] + 1)
        results = []
        for match in report["matches"]:
            passes = [[(p["group"], p["ordinal"], p["record_offset"], p["carve_offset"])
                       for p in row] for row in match["passes"]]
            results.append((match["effect_offset"], match["technique_index"], passes))
        self.assertEqual(results, [(first_at, 0, [[selected, other_group], []]),
                                   (first_at, 1, [[spare_program]]), (second_at, 0, [[other_group]])])
        for missing in [(name, "Absent"), ("Absent", "Hue")]:
            with self.subTest(missing=missing):
                self.assertEqual(sgd5_effects.read_database(io.BytesIO(data), layout, *missing)["matches"], [])

    def test_ordinals_restart_in_each_database(self):
        item = effect("Mock", {5: (0,)}, [("Wave", [(0, 0, 0, 0, 0, 1)])])
        first, layout1 = database({5: ("One",)}, [item])
        second, layout2 = database({5: ("Two",)}, [item], base=len(first))
        data = first + second
        source = io.BytesIO(data)
        for layout, local in [(layout2, "Two"), (layout1, "One")]:
            with self.subTest(local=local):
                report = sgd5_effects.read_database(source, layout, "Mock", "Wave")
                program_ref, = report["matches"][0]["passes"][0]
                self.assertEqual(program_ref["ordinal"], 0)
                self.assertEqual(program_ref["record_offset"], data.index(label(local)))

    def test_malformed_effect_rejected_even_when_query_misses(self):
        for reason, item in [
            ("ordinal outside group", effect("Mock", {0: (1,)})),
            ("selector outside effect list", effect("Mock", {0: (0,)}, [("Hue", [(2, 0, 0, 0, 0, 0)])])),
            ("invalid pool text", effect("Mock", pool_tail=b"\xff\0")),
        ]:
            with self.subTest(reason=reason):
                data, layout = database({0: ("Glow",)}, [item])
                with self.assertRaises(ValueError):
                    sgd5_effects.read_database(io.BytesIO(data), layout, "Absent", "Absent")

    def test_counted_suffix_skips_header_and_pool_lookalikes(self):
        decoy = label("Other") + u32(0) + u32(0) * 6
        tags = u32(0x80000000) + b"ABCD" + u32(24) + bytes(24)
        suffix = post_pass((0, 17), (decoy, b"test"), 0x14000002, tags)
        inner_pool = u32(32) + b"\0" + b"X" * 30 + b"\0"
        first = effect("First", pool_tail=inner_pool, suffix=suffix)
        second = effect("Next")
        data, layout = database({}, [first, second])
        matches = sgd5_effects.read_database(io.BytesIO(data), layout, list_effects=True)["matches"]
        self.assertEqual([(m["effect"], m["effect_offset"]) for m in matches],
                         [("First", data.index(first)), ("Next", data.index(second))])

    def test_counted_suffix_refuses_invalid_framing(self):
        invalid = {
            "oversize word array": u32(0xffffffff),
            "marker": u32(0) + u32(7),
            "first block length": u32(0) + u32(0x80000000) + u32(0xffffffff),
            "second block length": u32(0) + u32(0x80000000) + u32(0) + u32(0xffffffff),
            "packed count": post_pass(packed=0x00000100),
            "missing tag payload": post_pass(packed=1, entries=u32(0x80000000)),
            "oversize variable payload": post_pass(packed=1, entries=u32(0xffffffff)),
            "oversize pool": post_pass() + u32(0xffffffff),
        }
        for reason, suffix in invalid.items():
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                data, layout = database({}, [effect("First", suffix=suffix)])
                sgd5_effects.read_database(io.BytesIO(data), layout)

    def test_declared_effect_count_must_consume_region(self):
        item = effect("First")
        original, layout = database({}, [item])
        offset = original.index(item) - 4
        for count in (0, 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                data = bytearray(original)
                data[offset:offset + 4] = u32(count)
                sgd5_effects.read_database(io.BytesIO(data), layout)

    def test_bad_framing_and_truncation(self):
        data, layout = database({0: ("Glow",)}, [effect("Vacant")])
        record = data.index(label("Glow"))
        end = data.rfind(u32(1))
        for reason, offset, replacement in [
            ("magic", 0, b"MOCK"),
            ("stored size", record + len(label("Glow")) + 1, u32(0xffffffff)),
            ("end marker", end, u32(2)),
            ("extra marker in padding", end + 4, u32(1)),
        ]:
            with self.subTest(reason=reason):
                damaged = bytearray(data)
                damaged[offset:offset + len(replacement)] = replacement
                with self.assertRaises(ValueError):
                    sgd5_effects.read_database(io.BytesIO(damaged), layout)
        for cut in (record + 2, end + 2):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                sgd5_effects.read_database(io.BytesIO(data[:cut]), layout)

    @unittest.skipUnless(os.environ.get("LEONIDA_GAME_DIR"), "set LEONIDA_GAME_DIR to check package counts")
    def test_xbox_package_counts(self):
        archive = Path(os.environ["LEONIDA_GAME_DIR"]) / "shaders.rpf"
        fields = ("effects", "named_effects", "techniques", "passes")
        with archive.open("rb") as source:
            counts = []
            for layout in sgd5_effects.DATABASES:
                result = sgd5_effects.read_database(source, layout)
                counts.append(tuple(result[key] for key in fields))
        self.assertEqual(counts, [(910, 891, 23938, 27536), (22, 22, 1099, 1262)])


if __name__ == "__main__":
    unittest.main()
