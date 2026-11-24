"""gamedata_maps.py must decode the measured MHIU and PAMH (HMAP) layouts.

    python -I tests/test_gamedata_maps.py

Every fixture is synthetic. MHIU is a 44-byte header, a row table and packed
u8 heights with a default fill. PAMH is HMAP v1.2: a row table, a max u8 array
and a min u8 array. The archive tests build a carve-shaped directory and check
the CLI, including a MHIU split across two 64 KiB streams.
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
TOOL = os.path.join(TOOLS, "gamedata_maps.py")
sys.path.insert(0, TOOLS)

import gamedata_maps  # noqa: E402


def mhiu_payload(width: int, height: int, rows, pixels: bytes,
                 default: int = 23) -> bytes:
    """`rows` is a list of (start, count, values); offsets are assigned in order."""
    table = bytearray()
    packed = bytearray()
    for start, count, values in rows:
        assert len(values) == count
        offset = len(packed) - start
        table += struct.pack("<HHi", start, count, offset)
        packed += bytes(values)
    table += b"\0" * (8 * (height - len(rows)))
    data = bytes(table) + bytes(packed) + pixels
    head = bytearray(44)
    head[:4] = b"MHIU"
    struct.pack_into("<H", head, 0x04, 1)
    struct.pack_into("<HH", head, 0x06, width, height)
    struct.pack_into("<6f", head, 0x0C, -10.0, -20.0, -30.0, 40.0, 50.0, 60.0)
    struct.pack_into("<I", head, 0x24, len(data))
    head[0x28] = default
    head[0x29:0x2C] = bytes.fromhex("a55a3c")
    return bytes(head) + data


def pamh_payload(width: int, height: int, maximum, minimum,
                 padding: int = 0x1BBC8CED,
                 version: tuple = (1, 2)) -> bytes:
    rows = b"".join(struct.pack("<HHi", 0, width, i * width)
                    for i in range(height))
    data = rows + bytes(maximum) + bytes(minimum)
    head = bytearray(44)
    head[:4] = b"PAMH"
    head[0x04] = version[0]
    head[0x05] = version[1]
    struct.pack_into("<I", head, 0x08, padding)
    struct.pack_into("<HH", head, 0x0C, width, height)
    struct.pack_into("<6f", head, 0x10, -10.0, -20.0, -30.0, 40.0, 50.0, 60.0)
    struct.pack_into("<I", head, 0x28, len(data))
    return bytes(head) + data


class GamedataMapsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload: bytes, chunks=None):
        parts = chunks if chunks is not None else [payload]
        for part in parts:
            off = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({"off": off, "stored": len(part), "size": len(part),
                              "head": part[:16].hex(), "pre": ""})

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)},
                       "files": self.rows}, f)

    def run_tool(self):
        out = os.path.join(self.root, "maps.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.carve, "--json", out],
                           cwd=self.root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        data = None
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        return p, data

    def test_mhiu_row_table_and_default_fill(self):
        payload = mhiu_payload(4, 3, [
            (0, 0, []),
            (1, 2, [7, 8]),
            (0, 4, [9, 10, 11, 12]),
        ], pixels=b"")
        p = gamedata_maps.parse_mhiu(payload)
        self.assertEqual(p["header"]["width"], 4)
        self.assertEqual(p["header"]["height"], 3)
        self.assertEqual(p["header"]["bounds"]["min"], [-10.0, -20.0, -30.0])
        self.assertEqual(p["header"]["default_height"], 23)
        self.assertEqual(p["header"]["pad_29"], "a55a3c")
        self.assertEqual(p["grid"]["filled_rows"], 2)
        self.assertEqual(p["grid"]["empty_rows"], 1)
        self.assertEqual(p["grid"]["filled_cells"], 6)
        self.assertEqual(p["grid"]["default"], 23)
        self.assertEqual(p["grid"]["span_ends_at_default"], 0)
        self.assertEqual(list(p["_grid"]), [23, 23, 23, 23,
                                            23, 7, 8, 23,
                                            9, 10, 11, 12])

    def test_mhiu_default_comes_from_the_header_byte(self):
        payload = mhiu_payload(3, 2, [(0, 0, []), (1, 1, [40])], pixels=b"",
                               default=99)
        p = gamedata_maps.parse_mhiu(payload)
        self.assertEqual(p["header"]["default_height"], 99)
        self.assertEqual(p["grid"]["default"], 99)
        self.assertEqual(list(p["_grid"]), [99, 99, 99, 99, 40, 99])
        # z = -30 + 99/255 * 90
        self.assertAlmostEqual(p["grid"]["z_at_default"], 4.94, places=2)

    def test_mhiu_counts_span_ends_on_the_default(self):
        payload = mhiu_payload(4, 2, [(0, 3, [23, 5, 6]), (1, 3, [7, 8, 23])],
                               pixels=b"")
        p = gamedata_maps.parse_mhiu(payload)
        self.assertEqual(p["grid"]["span_ends_at_default"], 2)

    def test_mhiu_rejects_a_row_past_the_pixels(self):
        payload = mhiu_payload(4, 1, [(0, 4, [1, 2, 3, 4])], pixels=b"")
        head = bytearray(payload)
        struct.pack_into("<i", head, 0x2C + 4, 1000)
        with self.assertRaisesRegex(ValueError, "pixel data"):
            gamedata_maps.parse_mhiu(bytes(head))

    def test_pamh_splits_max_and_min_arrays(self):
        payload = pamh_payload(3, 2,
                               [1, 2, 3, 4, 5, 6],
                               [0, 1, 2, 3, 4, 7])
        p = gamedata_maps.parse_pamh(payload)
        self.assertEqual(p["header"]["version_major"], 1)
        self.assertEqual(p["header"]["version_minor"], 2)
        self.assertEqual(p["header"]["padding08"], 0x1BBC8CED)
        self.assertEqual(p["header"]["bounds"]["max"], [40.0, 50.0, 60.0])
        self.assertEqual(list(p["_max"]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(p["_min"]), [0, 1, 2, 3, 4, 7])
        self.assertEqual(p["grid"]["min_above_max"], 1)

    def test_pamh_padding_word_does_not_change_the_arrays(self):
        args = (3, 2, [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 7])
        zero = gamedata_maps.parse_pamh(pamh_payload(*args, padding=0))
        measured = gamedata_maps.parse_pamh(pamh_payload(*args, padding=0x1BBC8CED))
        ascii_tail = gamedata_maps.parse_pamh(pamh_payload(*args, padding=0x74680001))
        self.assertEqual((zero["header"]["padding08"], measured["header"]["padding08"],
                          ascii_tail["header"]["padding08"]),
                         (0, 0x1BBC8CED, 0x74680001))
        self.assertEqual(zero["_max"], measured["_max"])
        self.assertEqual(zero["_min"], ascii_tail["_min"])
        self.assertEqual(measured["grid"], ascii_tail["grid"])

    def test_pamh_census_groups_tiles_by_grid(self):
        self.add(pamh_payload(4, 3, [1] * 12, [0] * 12, padding=0, version=(1, 1)))
        self.add(pamh_payload(4, 3, [2] * 12, [0] * 12,
                              padding=0x74680001, version=(1, 2)))
        self.add(pamh_payload(2, 2, [3] * 4, [0] * 4, version=(1, 2)))
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        census = data["pamh_census"]
        self.assertEqual([(g["width"], g["height"], g["tiles"]) for g in census],
                         [(4, 3, 2), (2, 2, 1)])
        first = census[0]
        self.assertEqual((first["full_width_row_tiles"], first["padding08_zero"],
                          first["padding08_distinct"]), (2, 1, 2))
        self.assertEqual(first["version_minor"], {"1": 1, "2": 1})
        self.assertEqual(first["bounds_min"], [-10.0, -20.0, -30.0])
        self.assertIn("PAMH census 4x3: 2 tiles", p.stdout)

    def test_pamh_row_table_full_width_count(self):
        payload = pamh_payload(3, 2, [1] * 6, [0] * 6)
        head = bytearray(payload)
        struct.pack_into("<HH", head, 44, 0, 2)
        p = gamedata_maps.parse_pamh(bytes(head))
        self.assertEqual(p["grid"]["full_width_rows"], 1)

    def test_bounds_reject_nan_and_infinity(self):
        for off, args, field in (
                (0x0C, (3, 1, [(0, 3, [1, 2, 3])], b""), "mhiu"),
                (0x10, (3, 1, [1, 2, 3], [0, 0, 0]), "pamh")):
            for bad in (float("nan"), float("inf"), float("-inf")):
                payload = (mhiu_payload if field == "mhiu" else pamh_payload)(*args)
                head = bytearray(payload)
                struct.pack_into("<f", head, off, bad)
                parser = gamedata_maps.parse_mhiu if field == "mhiu" \
                    else gamedata_maps.parse_pamh
                with self.assertRaisesRegex(ValueError, "non-finite bound"):
                    parser(bytes(head))

    def test_pamh_rejects_a_short_row_body(self):
        payload = pamh_payload(2, 2, [1, 2, 3, 4], [0, 1, 2, 3])
        head = bytearray(payload[:-1])
        struct.pack_into("<I", head, 0x28, len(head) - 44)
        with self.assertRaisesRegex(ValueError, "max and min rows"):
            gamedata_maps.parse_pamh(bytes(head))

    def test_decodes_mhiu_and_pamh_from_a_carve(self):
        self.add(mhiu_payload(4, 3, [(0, 0, []), (1, 2, [7, 8]), (0, 4, [9, 10, 11, 12])],
                              pixels=b""))
        self.add(pamh_payload(3, 2, [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 7]))
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        mhiu, pamh = data["payloads"]
        self.assertEqual(mhiu["type"], "MHIU")
        self.assertEqual((mhiu["grid"]["width"], mhiu["grid"]["height"]), (4, 3))
        self.assertEqual(mhiu["grid"]["filled_rows"], 2)
        self.assertEqual(pamh["type"], "PAMH")
        self.assertEqual((pamh["grid"]["width"], pamh["grid"]["height"]), (3, 2))
        self.assertEqual(pamh["grid"]["cells"], 6)
        self.assertIn("filled rows", p.stdout)
        self.assertIn("min>max", p.stdout)

    def test_groups_mhiu_continuation_stream(self):
        payload = mhiu_payload(300, 1, [(0, 300, [i % 256 for i in range(300)])], pixels=b"\x07" * 90000)
        self.assertGreater(len(payload), 65536)
        self.add(payload, [payload[:65536], payload[65536:]])
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        mhiu = data["payloads"][0]
        self.assertEqual(mhiu["chunks"], 2)
        self.assertEqual(mhiu["grid"]["width"], 300)
        self.assertEqual(mhiu["grid"]["height"], 1)

    def test_groups_a_continuation_after_a_128k_first_stream(self):
        # The header's size bounds the join, so a first stream of any whole number of 64 KiB blocks continues.
        payload = mhiu_payload(300, 1, [(0, 300, [i % 256 for i in range(300)])], pixels=b"\x07" * 150000)
        self.assertGreater(len(payload), 2 * 65536)
        self.add(payload, [payload[:2 * 65536], payload[2 * 65536:]])
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        mhiu = data["payloads"][0]
        self.assertEqual(mhiu["chunks"], 2)
        self.assertEqual(mhiu["grid"]["width"], 300)

    def test_reports_an_unwritable_json_destination(self):
        self.finish()
        blocker = os.path.join(self.root, "blocker")
        with open(blocker, "wb") as f:
            f.write(b"not a directory")
        out = os.path.join(blocker, "maps.json")
        result = subprocess.run(
            [sys.executable, "-I", TOOL, self.carve, "--json", out],
            capture_output=True, text=True, errors="replace",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"cannot write {out}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("wrote", result.stdout)

    def test_creates_missing_parent_directories_for_json_with_lf_endings(self):
        self.add(pamh_payload(3, 2, [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]))
        self.finish()
        out = os.path.join(self.root, "missing", "nested", "maps.json")
        result = subprocess.run(
            [sys.executable, "-I", TOOL, self.carve, "--json", out],
            cwd=self.root, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(os.path.isfile(out))
        with open(out, "rb") as f:
            raw = f.read()
        self.assertIn(b"\n", raw)
        self.assertNotIn(b"\r\n", raw)
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(len(data["payloads"]), 1)
        self.assertEqual(data["payloads"][0]["type"], "PAMH")

    def test_refuses_json_colliding_with_index_json_and_case_variants(self):
        self.add(pamh_payload(2, 2, [1, 2, 3, 4], [0, 1, 2, 3]))
        self.finish()
        index_path = os.path.join(self.carve, "index.json")
        with open(index_path, "rb") as f:
            original = f.read()

        targets = [index_path, os.path.join(self.carve, ".", "index.json")]
        cased_index = os.path.join(self.carve, "INDEX.JSON")
        if os.path.exists(cased_index):
            targets.append(cased_index)
            targets.append(os.path.join(self.root, "CARVE", "INDEX.JSON"))

        for out in targets:
            with self.subTest(out=out):
                result = subprocess.run(
                    [sys.executable, "-I", TOOL, self.carve, "--json", out],
                    cwd=self.root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertIn("would overwrite", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("wrote", result.stdout)
                with open(index_path, "rb") as f:
                    self.assertEqual(f.read(), original)

    def test_refuses_json_colliding_with_carved_bin_file(self):
        self.add(pamh_payload(2, 2, [1, 2, 3, 4], [0, 1, 2, 3]))
        self.finish()
        carved = os.path.join(self.carve, "000000001000.bin")
        with open(carved, "rb") as f:
            original = f.read()

        targets = [carved]
        cased_bin = os.path.join(self.carve, "000000001000.BIN")
        if os.path.exists(cased_bin):
            targets.append(cased_bin)

        hardlink = os.path.join(self.root, "hardlink.json")
        try:
            os.link(carved, hardlink)
        except (OSError, NotImplementedError):
            hardlink = None
        if hardlink is not None:
            targets.append(hardlink)

        symlink = os.path.join(self.root, "symlink.json")
        try:
            os.symlink(carved, symlink)
        except (OSError, NotImplementedError):
            symlink = None
        if symlink is not None:
            targets.append(symlink)

        for out in targets:
            with self.subTest(out=out):
                result = subprocess.run(
                    [sys.executable, "-I", TOOL, self.carve, "--json", out],
                    cwd=self.root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertIn("would overwrite", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("wrote", result.stdout)
                with open(carved, "rb") as f:
                    self.assertEqual(f.read(), original)

    def test_refuses_json_colliding_with_index_only_carved_stream(self):
        self.add(pamh_payload(2, 2, [1, 2, 3, 4], [0, 1, 2, 3]))
        self.finish()
        carved = os.path.join(self.carve, "000000001000.bin")
        os.remove(carved)

        result = subprocess.run(
            [sys.executable, "-I", TOOL, self.carve, "--json", carved],
            cwd=self.root, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("error:", result.stderr)
        self.assertIn("would overwrite", result.stderr)
        self.assertFalse(os.path.exists(carved))

    def test_refuses_json_pointing_to_existing_directory_or_carve_dir(self):
        self.finish()
        existing_dir = os.path.join(self.root, "reports_dir")
        os.makedirs(existing_dir)

        for out, expected in ((existing_dir, "directory"), (self.carve, "would overwrite")):
            with self.subTest(out=out):
                result = subprocess.run(
                    [sys.executable, "-I", TOOL, self.carve, "--json", out],
                    cwd=self.root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("wrote", result.stdout)

    def test_validates_json_before_scanning_carve_dir(self):
        # Leave index.json corrupt so a scan would fail with a JSON error instead of the path guard.
        carved = os.path.join(self.carve, "000000001000.bin")
        with open(carved, "wb") as f:
            f.write(b"raw carved stream")
        index_path = os.path.join(self.carve, "index.json")
        with open(index_path, "wb") as f:
            f.write(b"{corrupt json")
        out_dir = os.path.join(self.root, "existing_dir")
        os.makedirs(out_dir)

        for out, expected in ((out_dir, "directory"),
                              (index_path, "would overwrite"),
                              (carved, "would overwrite")):
            with self.subTest(out=out):
                result = subprocess.run(
                    [sys.executable, "-I", TOOL, self.carve, "--json", out],
                    cwd=self.root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_allows_non_input_json_report_inside_carve_dir(self):
        self.add(pamh_payload(2, 2, [1, 2, 3, 4], [0, 1, 2, 3]))
        self.finish()
        out = os.path.join(self.carve, "maps.json")
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-I", TOOL, self.carve, "--json", out],
                cwd=self.root, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["payloads"]), 1)

    def test_macos_case_insensitive_volume_folds_case_and_nfd(self):
        with mock.patch.object(gamedata_maps.os, "name", "posix"), \
             mock.patch.object(gamedata_maps.sys, "platform", "darwin"), \
             mock.patch.object(gamedata_maps.os, "pathconf", return_value=0, create=True):
            self.assertTrue(gamedata_maps.same_report_path(
                os.path.join(self.carve, "INDEX.JSON"),
                os.path.join(self.carve, "index.json"),
            ))
            self.assertTrue(gamedata_maps.same_report_path(
                os.path.join(self.carve, "caf\u00e9.json"),
                os.path.join(self.carve, "CAFE\u0301.json"),
            ))

    def test_complete_64k_map_leaves_the_next_stream_alone(self):
        payload = mhiu_payload(255, 1, [(0, 255, [7] * 255)], pixels=b"\x00" * 65229)
        self.assertEqual(len(payload), 65536)
        self.add(payload)
        self.add(b"unrelated resource")
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        row, = data["payloads"]
        self.assertEqual((row["size"], row["chunks"]), (65536, 1))
        self.assertEqual(row["grid"]["width"], 255)


class MalformedCarveIndex(unittest.TestCase):
    """A carve index that parses but has no usable files list names the index, not a bare KeyError."""

    def run_tool(self, index):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            result = subprocess.run([sys.executable, "-I", TOOL, root],
                                    capture_output=True, text=True, errors="replace")
        return result

    def test_empty_index_names_the_file_and_the_missing_key(self):
        result = self.run_tool({})
        self.assertEqual(result.returncode, 1)
        self.assertIn("index.json has no 'files' list", result.stderr)
        self.assertNotIn("error: 'files'", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_row_without_an_offset_is_named(self):
        result = self.run_tool({"files": [{"stored": 12}]})
        self.assertEqual(result.returncode, 1)
        self.assertIn("row 0 needs an 'off'", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
