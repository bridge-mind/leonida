"""CPU test for tools/psocache_inspect.py on a synthetic cache built by this test, not the game's.

The synthetic file follows the measured layout: a 0x80 header with the four counts and two sizes, two graphics
records of 0x70, one compute record of 0x18, two 8-byte meta-index entries, two 8-byte shader-index entries, a
meta blob of chunk records and a code blob with two s_endpgm markers padded with s_code_end.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psocache_inspect.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import psocache_inspect  # noqa: E402

HEADER = 0x80
CHUNK = 32
ENTRY0 = 0x18 + 5 * CHUNK  # header 0x18, two 0x41 and two 0x42 and one 0x43 chunk
ENTRY1 = 0x10 + 2 * CHUNK  # header 0x10, one 0x41 and one 0x43 chunk
META_SIZE = ENTRY0 + ENTRY1
CODE_SIZE = 0x200  # two 0x100-byte programs
GRAPHICS = 2
COMPUTE = 1
META_INDEX = 2
SHADER_INDEX = 2
FILE_SIZE = (
    HEADER
    + GRAPHICS * 0x70
    + COMPUTE * 0x18
    + META_INDEX * 8
    + SHADER_INDEX * 8
    + META_SIZE
    + CODE_SIZE
)
END_PGM = b"\x00\x00\x81\xbf"  # s_endpgm 0xBF810000 little-endian
CODE_END = b"\x00\x00\x9f\xbf"  # s_code_end 0xBF9F0000 little-endian


def build():
    """Return the bytes of a tiny cache that satisfies the model."""
    data = bytearray(FILE_SIZE)
    data[0:4] = b"PS\x00\x09"
    data[0x04:0x0A] = b"test\x00\x00"
    data[0x24:0x29] = b"v1\x00\x00"
    data[0x48:0x50] = bytes.fromhex("0011223344556677")
    struct.pack_into("<4I", data, 0x50, GRAPHICS, COMPUTE, META_INDEX, SHADER_INDEX)
    struct.pack_into("<2I", data, 0x60, META_SIZE, CODE_SIZE)

    off = HEADER
    for i in range(GRAPHICS):
        struct.pack_into("<Q", data, off, 0x1000 + i)
        struct.pack_into("<2I", data, off + 8, i, META_INDEX - 1)
        data[off + 0x60 : off + 0x70] = b"\xff" * 16
        off += 0x70
    for i in range(COMPUTE):
        struct.pack_into("<Q", data, off, 0x2000 + i)
        struct.pack_into("<4I", data, off + 8, 0, i, 0, i + 1)
        off += 0x18
    meta_index_off = off
    meta_off = meta_index_off + META_INDEX * 8 + SHADER_INDEX * 8
    struct.pack_into("<2I", data, meta_index_off, 0, ENTRY0)
    struct.pack_into("<2I", data, meta_index_off + 8, ENTRY0, ENTRY1)

    def chunk(pos, *words):
        packed = list(words) + [0] * (8 - len(words))
        struct.pack_into("<8I", data, meta_off + pos, *packed)
        return pos + CHUNK

    pos = chunk(0x18, 0x41, 0x300, 0x00, 0x11, 0x04, 0x00000408, 0xFFFFFFFF, 0)
    pos = chunk(pos, 0x41, 0x303, 0x08, 0x12, 0x02, 0x00010504, 0xFFFFFFFF, 0)
    pos = chunk(pos, 0x42, 0x00, 0x00070001, 0x11, 0x01, 0, 0xFFFFFFFF, 0)
    pos = chunk(pos, 0x42, 0x00, 0x00020101, 0x12, 0x01, 0, 0xFFFFFFFF, 0)
    pos = chunk(pos, 0x43, 0xABCDEF01, 0x12345678, 0x9ABCDEF0, 0x13579BDF)
    assert pos == ENTRY0
    pos = chunk(ENTRY0 + 0x10, 0x41, 0x002, 0x00, 0x1F4, 0x00, 0x00040D04, 0xFFFFFFFF, 0)
    pos = chunk(pos, 0x43, 0x0FEDCBA9, 0x76543210, 0x0FEDCBA9, 0x2468ACE0)
    assert pos == META_SIZE

    shader_index_off = meta_index_off + META_INDEX * 8
    struct.pack_into("<2I", data, shader_index_off, 0, 0x100)
    struct.pack_into("<2I", data, shader_index_off + 8, 0x100, 0x100)
    code_off = meta_off + META_SIZE
    for base in (0, 0x100):
        data[code_off + base + 0x20 : code_off + base + 0x24] = END_PGM
        for pos in range(base + 0x24, base + 0x100, 4):
            data[code_off + pos : code_off + pos + 4] = CODE_END
    return bytes(data)


def run(*args):
    return subprocess.run(
        [sys.executable, "-I", TOOL, *args], capture_output=True, text=True
    )


class PsocacheInspectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.cache")
        with open(self.path, "wb") as f:
            f.write(build())

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_matches_the_model(self):
        r = psocache_inspect.inspect(self.path)
        self.assertEqual(r["size"], FILE_SIZE)
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["verdict"], "model holds")
        self.assertEqual(
            r["counts"],
            {"graphics": 2, "compute": 1, "meta_index": 2, "shader_index": 2},
        )
        self.assertEqual(r["sizes"], {"meta": META_SIZE, "code": CODE_SIZE})
        self.assertTrue(all(g["fits"] for g in r["regions"].values()))
        self.assertEqual(r["regions"]["graphics"]["end"], 0x80 + 2 * 0x70)
        self.assertEqual(r["regions"]["compute"]["end"], 0x80 + 2 * 0x70 + 0x18)
        self.assertEqual(
            r["regions"]["meta_blob"]["offset"], 0x80 + 2 * 0x70 + 0x18 + 2 * 8 + 2 * 8
        )
        self.assertEqual(r["regions"]["code_blob"]["end"], FILE_SIZE)
        self.assertEqual(r["slack"], 0)

    def test_index_checks(self):
        r = psocache_inspect.inspect(self.path)
        self.assertTrue(r["meta_index"]["contiguous"])
        self.assertEqual(r["meta_index"]["coverage"], META_SIZE)
        self.assertTrue(r["shader_index"]["ascending"])
        self.assertEqual(r["shader_index"]["max_end"], CODE_SIZE)
        self.assertEqual(r["shader_index"]["slack"], 0)
        self.assertEqual(
            r["references"],
            {
                "limit": META_INDEX,
                "graphics_out_of_range": 0,
                "compute_out_of_range": 0,
            },
        )

    def test_chunk_histogram(self):
        r = psocache_inspect.inspect(self.path)
        ch = r["chunks"]
        self.assertEqual(ch["chunk_lists"], 2)
        self.assertEqual(ch["unparsed"], 0)
        self.assertEqual(ch["types"], {"0x41": 3, "0x42": 2, "0x43": 2})
        self.assertEqual(ch["user_tags"], {"0x2": 1, "0x300": 1, "0x303": 1})
        self.assertEqual(ch["entry_headers"], {"0x10": 1, "0x18": 1})

    def test_meta_decode(self):
        r = psocache_inspect.inspect(self.path, with_meta=True)
        md = r["meta"]
        self.assertEqual(md["records"], 2)
        self.assertEqual(md["ordered"], 2)
        self.assertEqual(md["one_hash"], 2)
        self.assertEqual(md["exact_pairs"], 2)
        self.assertEqual(md["records_with_extras"], 0)
        self.assertEqual(md["tag_middle"], {"0x300/0": 1, "0x303/1": 1})
        self.assertEqual(md["slot_space"], {"2": 1, "7": 1})
        self.assertEqual(md["slot_middle"], {"0": 1, "1": 1})
        self.assertEqual(md["slot_width"], {"1": 2})
        self.assertEqual(md["slot_w4"], {"1": 2})
        self.assertEqual(
            md["tags"]["0x300"],
            {
                "count": 1,
                "offset_min": 0,
                "offset_max": 0,
                "offset_values": 1,
                "aligned8": True,
                "sizes": {"8": 1},
                "a": {"0x0": 1},
                "b": {"0x4": 1},
                "count_values": 1,
            },
        )
        self.assertEqual(md["tags"]["0x303"]["offset_max"], 8)
        self.assertEqual(md["tags"]["0x303"]["sizes"], {"4": 1})
        self.assertEqual(md["tags"]["0x303"]["b"], {"0x5": 1})

    def test_meta_entry_listing(self):
        r = psocache_inspect.inspect(self.path, with_meta=True, entry=0)
        lines = r["meta_entry"]
        self.assertIn("entry offset 0x0 size %d: 5 chunks" % ENTRY0, lines[0])
        self.assertTrue(any("0x41 kind=0x300" in line for line in lines))
        self.assertTrue(any("0x42 slot=0x070001 space=7 middle=0 width=1" in line for line in lines))
        self.assertTrue(any("0x43 16-byte value, not printed" in line for line in lines))
        self.assertFalse(any("0xABCDEF01" in line for line in lines))

    def test_meta_entry_out_of_range(self):
        with self.assertRaises(ValueError):
            psocache_inspect.inspect(self.path, entry=99)

    def test_meta_cli(self):
        p = run(self.path, "--meta", "--entry", "0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("meta decode: 2 records", p.stdout)
        self.assertIn("0x41 kind=0x300", p.stdout)
        self.assertIn("0x42 slot=0x070001 space=7 middle=0 width=1", p.stdout)
        self.assertNotIn("0xABCDEF01", p.stdout)
        p = run(self.path, "--meta", "--limit", "1")
        self.assertIn("meta decode: 1 records", p.stdout)
        doc = json.loads(run(self.path, "--meta", "--json").stdout)
        self.assertEqual(doc["meta"]["records"], 2)

    def test_markers(self):
        r = psocache_inspect.inspect(self.path)
        mk = r["markers"]
        self.assertEqual(mk["s_endpgm"], 2)
        self.assertEqual(mk["s_code_end"], 2 * ((0x100 - 0x24) // 4))
        self.assertEqual(mk["entries_with_one"], 2)
        self.assertEqual(mk["entries_with_none"], 0)

    def test_cli_reports_and_exits_zero(self):
        p = run(self.path)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("verdict: model holds", p.stdout)
        self.assertIn("graphics=2 compute=1 meta_index=2 shader_index=2", p.stdout)
        self.assertIn("s_endpgm 2", p.stdout)
        doc = json.loads(run(self.path, "--json").stdout)
        self.assertEqual(doc["size"], FILE_SIZE)

    def test_truncated_file_disagrees_without_crashing(self):
        with open(self.path, "r+b") as f:
            f.truncate(FILE_SIZE - 16)
        p = run(self.path)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("DOES NOT FIT", p.stdout)
        self.assertIn("code_blob", p.stdout)
        self.assertIn("disagrees with the model", p.stdout)
        self.assertNotIn("Traceback", p.stderr)
        doc = json.loads(run(self.path, "--json").stdout)
        self.assertFalse(doc["regions"]["code_blob"]["fits"])
        self.assertEqual(doc["regions"]["code_blob"]["shortfall"], 16)

    def test_out_of_range_reference_disagrees(self):
        with open(self.path, "r+b") as f:
            f.seek(0x80 + 8)
            f.write(struct.pack("<I", META_INDEX + 3))
        p = run(self.path)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("record references", p.stdout)
        self.assertIn("graphics out of range 1", p.stdout)
        self.assertNotIn("Traceback", p.stderr)

    def test_bad_magic_exits_two(self):
        with open(self.path, "r+b") as f:
            f.write(b"NOPE")
        p = run(self.path)
        self.assertEqual(p.returncode, 2)
        self.assertIn("magic", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_short_file_exits_two(self):
        short = os.path.join(self.tmp.name, "short.bin")
        with open(short, "wb") as f:
            f.write(b"PS\x00\x09")
        p = run(short)
        self.assertEqual(p.returncode, 2)
        self.assertIn("shorter than", p.stderr)

    def test_missing_file_exits_two(self):
        p = run(os.path.join(self.tmp.name, "absent.bin"))
        self.assertEqual(p.returncode, 2)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
