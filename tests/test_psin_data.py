"""Synthetic tests for tools/psin_data.py (no game data: every hash is the joaat of an invented name)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psin_data.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psin_data as D  # noqa: E402
from test_psin_index import joaat, make_lenm, make_lsch, make_section  # noqa: E402


def ptr(block, offset=0):
    """A PSIN pointer: block number in the low 20 bits, byte offset above."""
    return block | (offset << 20)


def make_container(blocks, structs, enums=(), root=1, end_row=True, fill=b"\xaa" * 8):
    """blocks: [(type, bytes)]. Returns a complete PSIN container with LSCH, LENM, LRSC (root at +0x10) and CHKS."""
    rows = b"".join(struct.pack("<II", t, len(data)) + fill for t, data in blocks)
    rows += b"\x70" * 8 if end_row else b"\0" * 8      # the table ends 8 bytes short of a 16-byte boundary
    head = 8 + len(rows)
    data = bytearray()
    for _, payload in blocks:
        data += payload + b"\0" * ((-len(payload)) % 16)
    first = head + len(data)
    lrsc = b"0DRP" + struct.pack("<I", 122) + b"\0" * 8 + struct.pack("<I", root) + b"\0" * 12
    sections = (make_section("LSCH", make_lsch(structs)) + make_section("LENM", make_lenm(list(enums))) +
                make_section("LGID") + make_section("LRSC", lrsc))
    sections += make_section("CHKS", struct.pack(">I", first + len(sections) + 16) + b"\0" * 4)
    return b"PSIN" + struct.pack(">I", first) + rows + bytes(data) + sections


ROOT_T, ITEM_T, STR_T = joaat("RootThing"), joaat("ItemThing"), 0x10
KIND = joaat("KindEnum")

# RootThing (0x90 bytes): every container shape the decoder reads.
ROOT_MEMBERS = [
    (0x100, 0, 0x05, 0, 0, ITEM_T),                  # 0: element descriptor, ItemThing
    (0x100, 0, 0x15, 0, 0, 0),                       # 1: element descriptor, u32
    (0x100, 0, 0x4A, 0, 0, 0),                       # 2: element descriptor, u32 hash (map key)
    (joaat("items"), 0x00, 0x5C, 0, 0, 0),           # {u32 count, u32 capacity, ptr} -> ItemThing[]
    (joaat("values"), 0x10, 0x52, 0, 1, 0),          # {ptr, u16 count, u16 capacity} -> u32[]
    (joaat("sorted"), 0x20, 0x71, 0, 0, 2 | (1 << 16)),   # binary map u32 hash -> u32
    (joaat("hashed"), 0x38, 0x75, 0, 0, 2 | (1 << 16)),   # SwissTable u32 hash -> u32
    (joaat("open"), 0x48, 0x73, 0, 0, 2 | (1 << 16)),     # open addressing u32 hash -> u32
    (joaat("child"), 0x58, 0x07, 0, 0, 0),           # pointer to an ItemThing block
    (joaat("label"), 0x60, 0x42, 0, 0, 0),           # string pointer
    (joaat("short"), 0x68, 0x44, 0, 0, 0),           # 16-byte string, stored in place
    (joaat("kind"), 0x78, 0x62, 0, 0, KIND),         # 32-bit enum
    (joaat("fixed"), 0x7C, 0x50, 0, 1, 3),           # u32[3] in place
    (joaat("odd"), 0x88, 0x77, 0, 0, 0),             # a type this tool does not decode
]
SCHEMA = [
    (ROOT_T, 0x1, 0x90, 3, ROOT_MEMBERS),
    (ITEM_T, 0x2, 0x8, 2, [(joaat("x"), 0, 0x21, 0, 0, 0), (joaat("n"), 4, 0x15, 0, 0, 0)]),
]
ENUMS = [(KIND, 0x3, [(0, joaat("KIND_A")), (1, joaat("KIND_B"))])]


def sample(kind=1):
    """Blocks: 1 root, 2 ItemThing[2], 3 u32[3], 4 map pairs, 5 SwissTable group, 6 open slots, 7 one ItemThing,
    8 strings."""
    k1, k2 = joaat("alpha"), joaat("beta")
    lo, hi = sorted((k1, k2))
    root = bytearray(0x90)
    struct.pack_into("<IIQ", root, 0x00, 2, 2, ptr(2))
    struct.pack_into("<QHH", root, 0x10, ptr(3), 3, 3)
    struct.pack_into("<QQHH", root, 0x20, 1, ptr(4), 2, 2)
    struct.pack_into("<IIQ", root, 0x38, 1, 2, ptr(5))
    struct.pack_into("<IIQ", root, 0x48, 4, 1, ptr(6))
    struct.pack_into("<Q", root, 0x58, ptr(7))
    struct.pack_into("<Q", root, 0x60, ptr(8, 0))
    root[0x68:0x6B] = b"abc"
    root[0x68 + 15] = 0x80 | 3
    struct.pack_into("<i", root, 0x78, kind)
    struct.pack_into("<3I", root, 0x7C, 7, 8, 9)
    items = struct.pack("<fIfI", 1.5, 10, 2.5, 20)
    values = struct.pack("<3I", 1, 2, 3)
    pairs = struct.pack("<II", lo, 100) + struct.pack("<II", hi, 200)
    ctrl = bytearray(b"\x80" * 16)
    ctrl[0], ctrl[5] = k1 & 0x7F, k2 & 0x7F
    slots = bytearray(16 * 8)
    struct.pack_into("<II", slots, 0, k1, 11)
    struct.pack_into("<II", slots, 5 * 8, k2, 22)
    open_slots = bytearray(4 * 8)
    struct.pack_into("<II", open_slots, (k1 % 4) * 8, k1, 33)
    item = struct.pack("<fI", 3.5, 30)
    strings = b"hello\0"
    blocks = [(ROOT_T, bytes(root)), (ITEM_T, items), (0x15, values), (0x100, pairs), (0x101, bytes(ctrl) + slots),
              (0x102, bytes(open_slots)), (ITEM_T, item), (STR_T, strings)]
    return make_container(blocks, SCHEMA, ENUMS)


class TestBlockTable(unittest.TestCase):
    def test_blocks_follow_the_table_and_fill_the_region(self):
        blob = make_container([(0x15, b"\1\0\0\0"), (0x16, b"\0" * 17)], [])
        first = struct.unpack_from(">I", blob, 4)[0]
        blocks, fills = D.block_table(blob, first)
        self.assertTrue(fills)
        self.assertEqual([(t, s) for t, _, s in blocks], [(0x15, 4), (0x16, 17)])
        self.assertEqual(blocks[0][1] % 16, 0)
        self.assertEqual(blocks[1][1], blocks[0][1] + 16)

    def test_missing_end_row(self):
        blob = make_container([(0x15, b"\0" * 4)], [], end_row=False)
        with self.assertRaisesRegex(ValueError, "fill|padding"):
            D.block_table(blob, struct.unpack_from(">I", blob, 4)[0])

    def test_bad_fill(self):
        blob = make_container([(0x15, b"\0" * 4)], [], fill=b"\0" * 8)
        with self.assertRaisesRegex(ValueError, "fill"):
            D.block_table(blob, struct.unpack_from(">I", blob, 4)[0])

    def test_blocks_past_the_region_are_rejected(self):
        # A damaged size must not make the walker allocate gigabytes (review on #1077).
        blob = bytearray(make_container([(0x15, b"\0" * 4), (0x16, b"\0" * 32)], []))
        struct.pack_into("<I", blob, 8 + 16 + 4, 0xA2000000)       # second row's size
        with self.assertRaisesRegex(ValueError, "past the first section"):
            D.block_table(bytes(blob), struct.unpack_from(">I", blob, 4)[0])
        stats = __import__("collections").Counter()
        self.assertFalse(D.measure(bytes(blob), stats))
        self.assertEqual(stats["containers"], 0)

    def test_first_row_is_what_psin_index_calls_version(self):
        blob = make_container([(0x16, b"\0" * 0x230)], [])
        record = D.psin_index.parse(blob)
        self.assertEqual((record["version"], record["table_bytes"]), (0x16, 0x230))


class TestWalk(unittest.TestCase):
    def setUp(self):
        self.container = D.Container(sample())
        self.stats = __import__("collections").Counter()
        self.tree = D.Walker(self.container, self.stats, keep=True).run()

    def test_every_byte_is_reached(self):
        self.assertEqual(self.stats["bytes reached"], self.stats["bytes"])
        self.assertEqual(self.stats["blocks reached"], 8)

    def test_arrays(self):
        self.assertEqual(self.tree[f"{joaat('items'):08x}"],
                         [{f"{joaat('x'):08x}": 1.5, f"{joaat('n'):08x}": 10},
                          {f"{joaat('x'):08x}": 2.5, f"{joaat('n'):08x}": 20}])
        self.assertEqual(self.tree[f"{joaat('values'):08x}"], [1, 2, 3])
        self.assertEqual(self.tree[f"{joaat('fixed'):08x}"], [7, 8, 9])
        self.assertEqual(self.stats["array 5c block type matches element"], 1)
        self.assertEqual(self.stats["array 52 block type matches element"], 1)

    def test_maps(self):
        lo, hi = sorted((joaat("alpha"), joaat("beta")))
        self.assertEqual(self.tree[f"{joaat('sorted'):08x}"], [[f"{lo:08x}", 100], [f"{hi:08x}", 200]])
        self.assertEqual(sorted(v for _, v in self.tree[f"{joaat('hashed'):08x}"]), [11, 22])
        self.assertEqual(self.tree[f"{joaat('open'):08x}"], [[f"{joaat('alpha'):08x}", 33]])
        for check in ("map 71 first word 1", "map 71 keys ascending", "map 75 used slots == count",
                      "map 75 groups a power of two", "map 73 used slots == count"):
            self.assertEqual(self.stats[check], 1, check)

    def test_pointers_and_strings(self):
        self.assertEqual(self.tree[f"{joaat('child'):08x}"], {f"{joaat('x'):08x}": 3.5, f"{joaat('n'):08x}": 30})
        self.assertEqual(self.tree[f"{joaat('label'):08x}"], "hello")
        self.assertEqual(self.tree[f"{joaat('short'):08x}"], "abc")
        self.assertEqual(self.stats["short strings in place"], 1)

    def test_struct_pointer_past_block_end_is_rejected(self):
        # ItemThing is 8 bytes; pointing at offset 6 of an 8-byte block leaves only 2 bytes inside the block.
        root = bytearray(0x90)
        struct.pack_into("<Q", root, 0x58, ptr(2, 6))
        blob = make_container([(ROOT_T, bytes(root)), (ITEM_T, struct.pack("<fI", 3.5, 30))], SCHEMA, ENUMS)
        st = __import__("collections").Counter()
        out = D.Walker(D.Container(blob), st, keep=True).run()
        self.assertEqual(st["struct pointer resolved"], 0)
        self.assertEqual(st["struct pointer past block end"], 1)
        self.assertIsNone(out[f"{joaat('child'):08x}"])

    def test_enum_and_undecoded_type(self):
        self.assertEqual(self.tree[f"{joaat('kind'):08x}"], 1)
        self.assertEqual(self.stats["enum value listed"], 1)
        self.assertEqual(self.stats["type 77 not decoded"], 1)
        self.assertIsNone(self.tree[f"{joaat('odd'):08x}"])

    def test_enum_value_not_listed(self):
        stats = __import__("collections").Counter()
        D.Walker(D.Container(sample(kind=5)), stats).run()
        self.assertEqual(stats["enum value not listed"], 1)

    def test_32_bit_enum_matches_negative_and_high_values(self):
        # LENM lists both negative values and values >= 2**31 for 0x62 enums.
        for listed, stored in ((-1, -1), (0xFFFFFFF0, -16)):
            c = D.Container(sample(kind=stored))
            c.enums[KIND]["values"].append({"value": listed, "name": joaat("KIND_X")})
            stats = __import__("collections").Counter()
            tree = D.Walker(c, stats, keep=True).run()
            self.assertEqual(stats["enum value listed"], 1)
            self.assertEqual(stats["enum value not listed"], 0)
            self.assertEqual(tree[f"{joaat('kind'):08x}"], listed)

    def test_bad_pointer_is_counted_not_followed(self):
        blob = bytearray(sample())
        root_at = D.Container(bytes(blob)).blocks[0][1]
        struct.pack_into("<Q", blob, root_at + 0x58, ptr(99))
        stats = __import__("collections").Counter()
        D.Walker(D.Container(bytes(blob)), stats).run()
        self.assertEqual(stats["struct pointer block out of range"], 1)
        self.assertLess(stats["bytes reached"], stats["bytes"])


class TestMeasure(unittest.TestCase):
    def test_skips_incomplete_and_schema_less_containers(self):
        stats = __import__("collections").Counter()
        self.assertFalse(D.measure(sample()[:-5], stats))
        self.assertEqual(stats["skipped: incomplete container"], 1)
        blob = b"PSIN" + struct.pack(">I", 16) + b"\x70" * 8 + make_section("CHKS", b"\0" * 8)
        self.assertFalse(D.measure(blob, stats))
        self.assertEqual(stats["skipped: no LSCH or LRSC section"], 1)

    def test_root_out_of_range(self):
        blob = make_container([(ROOT_T, b"\0" * 0x90)], SCHEMA, ENUMS, root=3)
        stats = __import__("collections").Counter()
        self.assertTrue(D.measure(blob, stats))
        self.assertEqual(stats["root out of range"], 1)

    def test_inline_and_fixed_arrays_are_bounded_to_enclosing_block(self):
        # 16-byte block with 0x54 inline array of char[4] claiming count=capacity=50000:
        # must be rejected as out of bounds rather than performing 50,000 OOB reads.
        inline_schema = [(0xAAAA0002, 0, 16, 2, [
            (0x22222222, 0, 0x54, 0, 1, 0),
            (0x00000100, 0, 0x40, 0, 0, 4),
        ])]
        blob = make_container([(0xAAAA0002, struct.pack("<II", 50000, 50000) + bytes(8))], inline_schema, [])
        stats = __import__("collections").Counter()
        self.assertTrue(D.measure(blob, stats))
        self.assertEqual(stats["inline array out of bounds"], 1)
        self.assertEqual(stats["char arrays"], 0)

        # 16-byte block with 0x50 fixed array of 100 char[4] elements (400 bytes > 16).
        fixed_schema = [(0xAAAA0003, 0, 16, 2, [
            (0x33333333, 0, 0x50, 0, 1, 100),
            (0x00000100, 0, 0x40, 0, 0, 4),
        ])]
        blob2 = make_container([(0xAAAA0003, bytes(16))], fixed_schema, [])
        stats2 = __import__("collections").Counter()
        self.assertTrue(D.measure(blob2, stats2))
        self.assertEqual(stats2["fixed array out of bounds"], 1)
        self.assertEqual(stats2["char arrays"], 0)

    def test_map_75_zero_groups_is_not_power_of_two(self):
        map_schema = [(0xAAAA0001, 0, 16, 2, [
            (0x11111111, 0, 0x75, 0, 0, (2 << 16) | 1),
            (0x00000100, 0, 0x15, 0, 0, 0),
            (0x00000100, 0, 0x15, 0, 0, 0),
        ])]
        blob = make_container([(0xAAAA0001, struct.pack("<IIQ", 0, 1, 1))], map_schema, [])
        stats = __import__("collections").Counter()
        self.assertTrue(D.measure(blob, stats))
        self.assertEqual(stats["map 75 groups a power of two"], 0)
        self.assertEqual(stats["map 75 groups not a power of two"], 1)

    def test_root_block_shorter_than_struct_is_rejected(self):
        # 32-byte struct backed by a 16-byte root block (followed immediately by LSCH)
        # must not read the LSCH header as struct fields.
        schema = [(0xAAAA0010, 1, 32, 2, [
            (joaat("a"), 0, 0x15, 0, 0, 0),
            (joaat("b"), 16, 0x15, 0, 0, 0),
        ])]
        blob = make_container([(0xAAAA0010, struct.pack("<4I", 42, 0, 0, 0))], schema, [])
        stats = __import__("collections").Counter()
        tree = D.Walker(D.Container(blob), stats, keep=True).run()
        self.assertIsNone(tree)
        self.assertEqual(stats["root past block end"], 1)
        self.assertEqual(stats["structs"], 0)
        self.assertEqual(stats["bytes reached"], 0)

        # A zero-byte root block shares its base offset with block 2; it must still be rejected.
        blob_zero = make_container([(0xAAAA0010, b""), (0x15, b"\x11" * 32)], schema, [])
        stats_zero = __import__("collections").Counter()
        self.assertIsNone(D.Walker(D.Container(blob_zero), stats_zero, keep=True).run())
        self.assertEqual(stats_zero["root past block end"], 1)
        self.assertEqual(stats_zero["structs"], 0)

    def test_embedded_struct_past_block_end_is_rejected(self):
        schema = [
            (0xAAAA0011, 1, 16, 2, [(joaat("sub"), 0, 0x05, 0, 0, 0xAAAA0012)]),
            (0xAAAA0012, 2, 32, 2, [(joaat("x"), 16, 0x15, 0, 0, 0)]),
        ]
        blob = make_container([(0xAAAA0011, bytes(16))], schema, [])
        stats = __import__("collections").Counter()
        tree = D.Walker(D.Container(blob), stats, keep=True).run()
        self.assertEqual(tree, {f"{joaat('sub'):08x}": None})
        self.assertEqual(stats["structs"], 1)
        self.assertEqual(stats["struct past block end"], 1)

    def test_char_array_past_block_end_is_rejected(self):
        schema = [(0xAAAA0013, 1, 16, 2, [(joaat("name"), 0, 0x40, 0, 0, 32)])]
        blob = make_container([(0xAAAA0013, b"A" * 16)], schema, [])
        stats = __import__("collections").Counter()
        tree = D.Walker(D.Container(blob), stats, keep=True).run()
        self.assertEqual(tree, {f"{joaat('name'):08x}": None})
        self.assertEqual(stats["char arrays"], 0)
        self.assertEqual(stats["member past block end"], 1)

    def test_scalar_and_container_headers_past_block_end_are_rejected(self):
        schema = [(0xAAAA0014, 1, 16, 2, [
            (0x00000100, 0, 0x15, 0, 0, 0),                    # element 0: u32
            (joaat("qword"), 12, 0x17, 0, 0, 0),                # 8 bytes at +12 -> past 16
            (joaat("arr"), 8, 0x5C, 0, 0, 0),                   # 16-byte header at +8 -> past 16
            (joaat("smap"), 4, 0x71, 0, 0, 0 | (0 << 16)),      # 20-byte header at +4 -> past 16
            (joaat("inline"), 12, 0x54, 0, 0, 0),               # 8-byte header at +12 -> past 16
        ])]
        blob = make_container([(0xAAAA0014, bytes(16))], schema, [])
        stats = __import__("collections").Counter()
        self.assertTrue(D.measure(blob, stats))
        self.assertEqual(stats["walk error"], 0)
        self.assertEqual(stats["member past block end"], 3)
        self.assertEqual(stats["inline array out of bounds"], 1)

    def test_short_string_length_mismatch_and_oob_slot(self):
        # Length byte claims 64 bytes in a 16-byte short-string slot: must return None, not read into LSCH.
        schema = [(0xAAAA0015, 1, 16, 2, [(joaat("short"), 0, 0x44, 0, 0, 0)])]
        payload = bytearray(b"A" * 15 + bytes([0x80 | 64]))
        blob = make_container([(0xAAAA0015, bytes(payload))], schema, [])
        stats = __import__("collections").Counter()
        tree = D.Walker(D.Container(blob), stats, keep=True).run()
        self.assertEqual(tree, {f"{joaat('short'):08x}": None})
        self.assertEqual(stats["short string length mismatch"], 1)
        self.assertEqual(stats["short strings in place"], 0)

        # Short-string slot starting at +8 in a 16-byte block extends past the block end.
        oob_schema = [(0xAAAA0016, 1, 16, 2, [(joaat("short"), 8, 0x44, 0, 0, 0)])]
        blob2 = make_container([(0xAAAA0016, bytes(16))], oob_schema, [])
        stats2 = __import__("collections").Counter()
        tree2 = D.Walker(D.Container(blob2), stats2, keep=True).run()
        self.assertEqual(tree2, {f"{joaat('short'):08x}": None})
        self.assertEqual(stats2["member past block end"], 1)


class TestCli(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_counts_on_a_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "000000001000.bin"), "wb") as f:
                f.write(sample())
            with open(os.path.join(tmp, "000000002000.bin"), "wb") as f:
                f.write(b"\x78\xda not psin")
            r = self.run_tool(tmp, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(r.stdout)
            self.assertEqual((got["PSIN files"], got["containers"], got["containers fully reached"]), (1, 1, 1))
            self.assertEqual(got["bytes reached"], got["bytes"])

    def test_dump_prints_the_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "one.bin")
            with open(path, "wb") as f:
                f.write(sample())
            r = self.run_tool(path, "--dump")
            self.assertEqual(r.returncode, 0, r.stderr)
            tree = json.loads(r.stdout)[f"{ROOT_T:08x}"]
            self.assertEqual(tree[f"{joaat('label'):08x}"], "hello")
            self.assertEqual(self.run_tool(tmp, "--dump").returncode, 2)

    def test_missing_source_is_one_line(self):
        r = self.run_tool(os.path.join(tempfile.gettempdir(), "no-such-psin-folder"))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)

    def test_dump_of_a_bad_container_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as f:
                f.write(sample()[:-5])
            r = self.run_tool(path, "--dump")
            self.assertEqual(r.returncode, 2)
            self.assertIn("incomplete", r.stderr)

    def test_a_single_file_without_the_magic_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not-psin.bin")
            with open(path, "wb") as f:
                f.write(b"\x78\xda not psin")
            r = self.run_tool(path, "--json")
            self.assertEqual(r.returncode, 2)
            self.assertEqual(r.stdout, "")
            self.assertEqual(len(r.stderr.strip().splitlines()), 1)
            self.assertIn("not a PSIN container", r.stderr)

    def test_a_folder_still_skips_a_file_without_the_magic(self):
        # the negative control: the same bytes inside a folder are skipped, not refused
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "000000003000.bin"), "wb") as f:
                f.write(b"\x78\xda not psin")
            r = self.run_tool(tmp, "--json")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout), {})

    def test_dump_missing_file_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.bin")
            r = self.run_tool(missing, "--dump")
            self.assertEqual(r.returncode, 2)
            self.assertIn("ERROR:", r.stderr)
            self.assertEqual(len(r.stderr.strip().splitlines()), 1)

    def test_dump_oserror_returns_2_without_traceback(self):
        import io
        import unittest.mock
        from contextlib import redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "locked.bin")
            with open(path, "wb") as f:
                f.write(sample())
            err = io.StringIO()
            with unittest.mock.patch.object(sys, "argv", ["psin_data.py", path, "--dump"]), \
                 unittest.mock.patch("builtins.open", side_effect=OSError("I/O error")), \
                 redirect_stderr(err):
                rc = D.main()
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", err.getvalue())
            self.assertIn("I/O error", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
