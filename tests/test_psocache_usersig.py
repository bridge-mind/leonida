"""CPU test for tools/psocache_usersig.py on a synthetic cache built by this test.

The cache has two graphics PSOs. Their stage records name three chunk records: one with SRV/UAV/CBV/sampler
entries and constants (fits), one with extra slots (fits), and one with 40 constant dwords (does not fit).
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "psocache_usersig.py")

HEADER = 0x80
GRAPHICS = 2
COMPUTE = 0
META_INDEX = 6
SHADER_INDEX = 1
CODE_SIZE = 0x100
END_PGM = b"\x00\x00\x81\xbf"
CODE_END = b"\x00\x00\x9f\xbf"
CHUNK = 32

STAGE0 = 0x30
CHUNK_A = 0x18 + 9 * CHUNK
CHUNK_B = 0x18 + 3 * CHUNK
CHUNK_C = 0x18 + 5 * CHUNK
LAYOUT = 0x20
STAGE1 = 0x30
META_SIZE = STAGE0 + CHUNK_A + CHUNK_B + CHUNK_C + LAYOUT + STAGE1
FILE_SIZE = (
    HEADER
    + GRAPHICS * 0x70
    + COMPUTE * 0x18
    + META_INDEX * 8
    + SHADER_INDEX * 8
    + META_SIZE
    + CODE_SIZE
)


def chunk(data, off, *words):
    struct.pack_into("<8I", data, off, *(list(words) + [0] * (8 - len(words))))


def build():
    data = bytearray(FILE_SIZE)
    data[0:4] = b"PS\x00\x09"
    struct.pack_into("<4I", data, 0x50, GRAPHICS, COMPUTE, META_INDEX, SHADER_INDEX)
    struct.pack_into("<2I", data, 0x60, META_SIZE, CODE_SIZE)

    off = HEADER
    for stage in (0, 5):
        struct.pack_into("<Q", data, off, 0x1000 + stage)
        struct.pack_into("<2I", data, off + 8, stage, 4)
        data[off + 0x60 : off + 0x70] = b"\xff" * 16
        off += 0x70

    meta_index_off = off
    shader_index_off = meta_index_off + META_INDEX * 8
    meta_off = shader_index_off + SHADER_INDEX * 8

    entries = []
    pos = 0
    entries.append((pos, STAGE0))
    struct.pack_into("<I", data, meta_off + pos + 4, 1)
    struct.pack_into("<I", data, meta_off + pos + 0x2C, 3)
    pos += STAGE0
    entries.append((pos, CHUNK_A))
    chunk(
        data,
        meta_off + pos + 0x18,
        0x41,
        0x401,
        0x00,
        0x03,
        0x00,
        0x000B0004,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + CHUNK,
        0x41,
        0x300,
        0x00,
        0x10,
        0x04,
        0x00000408,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 2 * CHUNK,
        0x41,
        0x303,
        0x08,
        0x11,
        0x04,
        0x00000408,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 3 * CHUNK,
        0x41,
        0x302,
        0x10,
        0x12,
        0x00,
        0x00000404,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 4 * CHUNK,
        0x41,
        0x002,
        0x00,
        0x1F4,
        0x00,
        0x00040D04,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 5 * CHUNK,
        0x42,
        0x00,
        0x00070001,
        0x10,
        0x01,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 6 * CHUNK,
        0x42,
        0x00,
        0x00070101,
        0x11,
        0x01,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 7 * CHUNK,
        0x42,
        0x00,
        0x000D0201,
        0x12,
        0x01,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 8 * CHUNK,
        0x43,
        0xDEADBEEF,
        0x11223344,
        0x55667788,
        0x99AABBCC,
    )
    pos += CHUNK_A
    entries.append((pos, CHUNK_B))
    chunk(
        data,
        meta_off + pos + 0x18,
        0x41,
        0x401,
        0x00,
        0x01,
        0x00,
        0x00140004,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + CHUNK,
        0x41,
        0x401,
        0x00,
        0x02,
        0x00,
        0x00140004,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 2 * CHUNK,
        0x43,
        0x01020304,
        0x05060708,
        0x090A0B0C,
        0x0D0E0F10,
    )
    pos += CHUNK_B
    entries.append((pos, CHUNK_C))
    chunk(
        data,
        meta_off + pos + 0x18,
        0x41,
        0x300,
        0x00,
        0x01,
        0x01,
        0x00000408,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + CHUNK,
        0x42,
        0x00,
        0x00070001,
        0x01,
        0x01,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 2 * CHUNK,
        0x42,
        0x00,
        0x000E0302,
        0x00,
        0x00,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 3 * CHUNK,
        0x42,
        0x00,
        0x000A0002,
        0x00,
        0x00,
        0,
        0xFFFFFFFF,
        0,
    )
    chunk(
        data,
        meta_off + pos + 0x18 + 4 * CHUNK,
        0x43,
        0x0F1E2D3C,
        0x4B5A6978,
        0x8796A5B4,
        0xC3D2E1F0,
    )
    pos += CHUNK_C
    entries.append((pos, LAYOUT))
    pos += LAYOUT
    entries.append((pos, STAGE1))
    struct.pack_into("<I", data, meta_off + pos + 4, 2)
    struct.pack_into("<I", data, meta_off + pos + 0x2C, 4)
    pos += STAGE1
    assert pos == META_SIZE
    for i, (start, size) in enumerate(entries):
        struct.pack_into("<2I", data, meta_index_off + i * 8, start, size)
    struct.pack_into("<2I", data, shader_index_off, 0, CODE_SIZE)

    code_off = meta_off + META_SIZE
    data[code_off + 0x20 : code_off + 0x24] = END_PGM
    for p in range(code_off + 0x24, code_off + CODE_SIZE, 4):
        data[p : p + 4] = CODE_END
    return bytes(data)


def run(*args):
    return subprocess.run(
        [sys.executable, "-I", TOOL, *args], capture_output=True, text=True
    )


class UsersigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.cache")
        with open(self.path, "wb") as f:
            f.write(build())

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary(self):
        p = run(self.path)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("graphics PSOs sampled: 2 of 2", p.stdout)
        self.assertIn("distinct chunk records resolved: 3", p.stdout)
        self.assertIn("fit 32 dwords: 1 of 2", p.stdout)
        self.assertIn("SRV=2", p.stdout)
        self.assertIn("UAV=1", p.stdout)
        self.assertIn("CBV=1", p.stdout)
        self.assertIn("Sampler=1", p.stdout)
        self.assertIn("constants=3", p.stdout)
        self.assertIn("extra slots: 2 in 1 records", p.stdout)
        self.assertIn("pso 1 record 2: constants=40", p.stdout)

    def test_entry_lists_kinds_slots_and_fit(self):
        p = run(self.path, "--entry", "0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("graphics PSO 0", p.stdout)
        self.assertIn("chunk record 1", p.stdout)
        self.assertIn("kind=0x401(constants)", p.stdout)
        self.assertIn(
            "kind=0x300(SRV) offset=0x0 key=0x10 count=0x4 space(b)=4 base(a)=0 size=8",
            p.stdout,
        )
        self.assertIn("kind=0x303(UAV)", p.stdout)
        self.assertIn("kind=0x302(CBV)", p.stdout)
        self.assertIn("kind=0x002(Sampler)", p.stdout)
        self.assertIn("slot table(SS)=7 range(II)=0(SRV) width(RR)=1", p.stdout)
        self.assertIn("slot table(SS)=13 range(II)=2(CBV) width(RR)=1", p.stdout)
        self.assertIn("extra slot table(SS)=14 range(II)=3 width(RR)=2", p.stdout)
        self.assertIn("PSO fits: True", p.stdout)
        self.assertNotIn("DEADBEEF", p.stdout)

    def test_entry_reports_a_non_fitting_record(self):
        p = run(self.path, "--entry", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("chunk record 2", p.stdout)
        self.assertIn("40 constants", p.stdout)
        self.assertIn("PSO fits: False", p.stdout)

    def test_limit(self):
        p = run(self.path, "--limit", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("graphics PSOs sampled: 1 of 2", p.stdout)

    def test_json(self):
        p = run(self.path, "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)
        s = doc["summary"]
        self.assertEqual(s["sampled"], 2)
        self.assertEqual(s["fit_two_per_table"], 1)
        self.assertEqual(s["fit_one_per_table"], 1)
        self.assertEqual(s["extras"], 2)
        self.assertEqual(len(s["failing"]), 1)
        self.assertEqual(s["failing"][0]["constants"], 40)
        self.assertEqual(s["kind_counts"]["SRV"], 2)
        self.assertIsNone(doc["pso"])
        doc = json.loads(run(self.path, "--entry", "0", "--json").stdout)
        self.assertEqual(doc["pso"]["stage"], 0)
        self.assertEqual(len(doc["pso"]["records"]), 2)

    def test_bad_entry_exits_two(self):
        p = run(self.path, "--entry", "99")
        self.assertEqual(p.returncode, 2)
        self.assertIn("outside the 2 graphics records", p.stderr)

    def test_not_a_cache_exits_two(self):
        bad = os.path.join(self.tmp.name, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + bytes(200))
        p = run(bad)
        self.assertEqual(p.returncode, 2)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
