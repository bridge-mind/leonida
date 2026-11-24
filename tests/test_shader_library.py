"""Unit tests for tools/shader_library.py (the pipeline library in the zlib part of shaders.rpf).

    python -I tests/test_shader_library.py

Zero game data: every library and archive is built here from synthetic bytes.
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

import shader_library as sl  # noqa: E402

ENDPGM = struct.pack("<I", 0xBF810000)
CODE_END = struct.pack("<I", 0xBF9F0000)
V_MOV = struct.pack("<I", 0x7E000280)
NONE = 0xFFFFFFFF


def prologue(tag):
    """Three 12-byte VOP3 opcode-0x180 words with literals: the shape every library program starts with."""
    out = b""
    for lo in (tag & 0xFFFF, (tag >> 16) & 0xFFFF, 0):
        out += struct.pack("<III", 0xD5800000 | lo, 0x03FE0100, 0x00000AC0)
    return out


def prog(tag, body=2):
    code = prologue(tag) + V_MOV * body + ENDPGM
    return code + CODE_END * ((-len(code)) % 256 // 4)


def kind0(threads=(8, 8, 1), rsrc2=0xB8A):
    x, y, z = threads
    return struct.pack("<7I", 0x4005CF00, 0x6C090102, 0, x | y << 16, z, 0x402C0343, rsrc2)


def layout_blob(kind, digest, records=1):
    """A resource-layout blob: head (16 bytes for a graphics stage, 24 for compute) ending in a 0x40-tagged word,
    32-byte 0x41/0x42 records, then 0x43, the 16-byte digest and 12 bytes of padding."""
    head = struct.pack("<I", kind) + bytes(sl.STAGE_HEAD[kind] - 8) + struct.pack("<I", 0x00030440)
    recs = b"".join(struct.pack("<I", 0x41 + (r & 1)) + bytes(16) + struct.pack("<I", 0x43) + bytes(8)
                    for r in range(records))    # a 0x43 inside a record must not end the walk
    return head + recs + struct.pack("<I", 0x43) + digest + bytes(12)


def dxbc(digest, extra=b""):
    """A DXBC container with a HASH part (flags + 16-byte digest) and one more part."""
    parts = [(b"HASH", struct.pack("<I", 0) + digest), (b"SFI0", extra or bytes(8))]
    off = 32 + 4 * len(parts)
    table, body = b"", b""
    for tag, data in parts:
        table += struct.pack("<I", off + len(body))
        body += tag + struct.pack("<I", len(data)) + data
    total = off + len(body)
    return b"DXBC" + bytes(16) + struct.pack("<III", 1, total, len(parts)) + table + body


D_PS, D_VS, D_MS, D_CS = (bytes([n]) * 16 for n in (1, 2, 3, 4))


def build(graphics, compute, blobs, programs, name=b"0x00000000", version=b"1.0"):
    """graphics: [(state, layout, {slot: (program, blob)})]; compute: [(state, blob, program, layout)]."""
    head = bytearray(sl.HEADER)
    head[:4] = sl.MAGIC
    head[4:4 + len(name)] = name
    head[0x24:0x24 + len(version)] = version
    struct.pack_into("<Q", head, 0x48, 0x0123456789ABCDEF)
    blob_data, blob_tab = b"", b""
    for x in blobs:
        blob_tab += struct.pack("<II", len(blob_data), len(x))
        blob_data += x
    code, prog_tab = b"", b""
    for x in programs:
        code += b"\0" * ((-len(code)) % 256)
        prog_tab += struct.pack("<II", len(code), len(x))
        code += x
    struct.pack_into("<6I", head, 0x50, len(graphics), len(compute), len(blobs), len(programs), len(blob_data),
                     len(code))
    gfx = b""
    for i, (state, layout, stages) in enumerate(graphics):
        pairs = []
        for slot in range(7):
            pairs += list(stages.get(slot, (NONE, NONE)))
        gfx += struct.pack("<QII", 0x1000 + i, state, layout) + bytes(40) + struct.pack("<14I", *pairs)
    cs = b"".join(struct.pack("<QIIII", 0x2000 + i, *c) for i, c in enumerate(compute))
    return bytes(head) + gfx + cs + blob_tab + prog_tab + blob_data + code


def sample():
    blobs = [kind0(), b"\x01" * 20, layout_blob(3, D_CS, 3), b"\x03" * 388, b"POSITION\0",
             layout_blob(2, D_PS), layout_blob(2, D_VS, 2), layout_blob(2, D_MS, 0)]
    programs = [prog(0x11112222), prog(0x33334444, 5), prog(0x55556666), prog(0x77778888, 3)]
    graphics = [(3, 4, {0: (0, 5), 1: (1, 6)}),      # VS + PS
                (3, 4, {0: (0, 5), 2: (2, 7)})]      # MS + PS, the PS shared
    compute = [(0, 1, 3, 2)]
    return build(graphics, compute, blobs, programs)


def archive(lib, junk=b"\x78\x9c not a stream " * 50):
    """An archive-like file: unrelated bytes (with false zlib headers), then the library as 64 KiB zlib blocks with
    small gaps between them, as in shaders.rpf."""
    out = bytearray(b"9FPR" + bytes(60) + junk)
    for k in range(0, len(lib), sl.BLOCK):
        out += zlib.compress(lib[k:k + sl.BLOCK], 9)
        out += b"\0" * (k // sl.BLOCK % 13)
    return bytes(out)


class Parse(unittest.TestCase):
    def test_header_and_tables(self):
        b = sample()
        lib = sl.parse(b)
        h = lib["header"]
        self.assertEqual((h["graphics"], h["compute"], h["blobs"], h["programs"]), (2, 1, 8, 4))
        self.assertEqual(h["name"], "0x00000000")
        self.assertEqual(lib["layout"]["end"], len(b))
        g0, g1 = lib["graphics"]
        self.assertEqual(g0["stages"], {"ps": {"program": 0, "blob": 5}, "vs": {"program": 1, "blob": 6}})
        self.assertEqual(set(g1["stages"]), {"ps", "ms"})
        self.assertEqual(lib["compute"][0]["program"], 3)
        self.assertEqual({k: sorted(v) for k, v in lib["stage"].items()},
                         {0: ["ps"], 1: ["vs"], 2: ["ms"], 3: ["cs"]})
        self.assertEqual(sl.program(b, lib, 1)[:36], prologue(0x33334444))
        self.assertEqual(sl.blob(b, lib, 4), b"POSITION\0")
        self.assertEqual(lib["digest"], {0: {D_PS}, 1: {D_VS}, 2: {D_MS}, 3: {D_CS}})

    def test_stage_digest(self):
        self.assertEqual(sl.stage_digest(layout_blob(2, D_PS, 4)), D_PS)
        self.assertEqual(sl.stage_digest(layout_blob(3, D_CS, 0)), D_CS)
        self.assertIsNone(sl.stage_digest(b""))
        self.assertIsNone(sl.stage_digest(layout_blob(2, D_PS)[:-20]))        # cut before the digest ends
        bad = bytearray(layout_blob(2, D_PS))
        bad[12] = 0x3F                                                         # head not closed by a 0x40 word
        self.assertIsNone(sl.stage_digest(bytes(bad)))
        bad = bytearray(layout_blob(3, D_CS, 1))
        bad[24] = 0x44                                                         # unknown record tag
        self.assertIsNone(sl.stage_digest(bytes(bad)))
        self.assertIsNone(sl.stage_digest(b"\x05" + layout_blob(2, D_PS)[1:]))  # unknown kind

    def test_rejects(self):
        b = sample()
        with self.assertRaises(sl.LibraryError):
            sl.parse(b"XX" + b[2:])
        with self.assertRaises(sl.LibraryError):
            sl.parse(b + b"\0")                      # size must match the header exactly
        bad = bytearray(b)                           # a stage naming a program past the table
        struct.pack_into("<I", bad, sl.HEADER + 56, 99)
        with self.assertRaises(sl.LibraryError):
            sl.parse(bytes(bad))
        bad = bytearray(b)                           # a program entry ending past the code section
        lay = sl.layout(sl.header(b))
        struct.pack_into("<I", bad, lay["program_table"] + 4, 1 << 20)
        with self.assertRaises(sl.LibraryError):
            sl.parse(bytes(bad))

    def test_compute_with_missing_program_or_blob(self):
        # A compute record may name no program or no layout blob (0xFFFFFFFF, "a missing stage is program
        # 0xFFFFFFFF"): it must not crash parse(), take a "cs" stage, or link a digest to a missing program.
        blobs = [kind0(), layout_blob(3, D_CS)]
        programs = [prog(0x11112222)]
        b = build([], [(0, 0, NONE, NONE), (0, NONE, 0, NONE)], blobs, programs)
        lib = sl.parse(b)
        self.assertEqual([c["program"] for c in lib["compute"]], [None, 0])
        self.assertEqual(lib["compute"][0]["layout"], None)
        self.assertEqual(lib["stage"], {0: {"cs"}})
        self.assertEqual(lib["digest"], {})

    def test_find_in_archive(self):
        lib = sample() + b""
        # make it span several 64 KiB blocks
        big = build([], [(0, 0, 0, 0)], [kind0()], [prog(1, 40000)])
        for b in (lib, big):
            self.assertEqual(sl.find_library(archive(b)), b)
        with self.assertRaises(sl.LibraryError):
            sl.find_library(archive(big)[:-200])    # the last block cut off: size no longer matches
        with self.assertRaises(sl.LibraryError):
            sl.find_library(b"9FPR" + bytes(4096))


class Main(unittest.TestCase):
    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = sl.main(list(args))
        return rc, out.getvalue()

    def test_unpack_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shaders.rpf")
            with open(src, "wb") as f:
                f.write(archive(sample()))
            out = os.path.join(tmp, "out")
            rc, text = self.run_main(src, "--out", out, "--save", os.path.join(tmp, "lib.bin"))
            self.assertEqual(rc, 0)
            self.assertIn("2 graphics + 1 compute pipelines, 4 programs; 4 written", text)
            with open(os.path.join(tmp, "lib.bin"), "rb") as f:
                self.assertEqual(f.read(), sample())
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual([e["file"] for e in index],
                             ["lib_ps_00000.dxbc", "lib_vs_00001.dxbc", "lib_ms_00002.dxbc", "lib_cs_00003.dxbc"])
            cs = index[3]
            self.assertEqual((cs["stage"], cs["threads"], cs["wave32"]), ("cs", [8, 8, 1], False))
            with open(os.path.join(out, "fspx", "lib_cs_00003.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual((meta["rsrc2"], meta["user_sgprs"], meta["code"]["start"]), (0xB8A, 5, 36))
            self.assertTrue(meta["code"]["prologue"])
            with open(os.path.join(out, "fspx", "lib_vs_00001.code"), "rb") as f:
                code = f.read()
            self.assertEqual(code[36:36 + 20], V_MOV * 5)
            self.assertEqual(meta["code"]["end"], 36 + 3 * 4 + 4)
            with open(os.path.join(out, "library.json"), encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["graphics"][1]["stages"], {"ps": "lib_ps_00000", "ms": "lib_ms_00002"})
            self.assertEqual(summary["compute"][0]["program"], "lib_cs_00003")

    def test_stage_filter_and_saved_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "lib.bin")
            with open(src, "wb") as f:
                f.write(sample())
            out = os.path.join(tmp, "out")
            rc, text = self.run_main(src, "--out", out, "--stages", "cs,vs")
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(os.listdir(os.path.join(out, "fspx"))),
                             ["lib_cs_00003.code", "lib_cs_00003.json", "lib_vs_00001.code", "lib_vs_00001.json"])

    def test_carve_names_sources_and_copies_containers(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            index = []
            for name, d in (("PS_A", D_PS), ("PS_A~1", D_PS), ("CS_B", D_CS), ("VS_C", D_VS)):
                with open(os.path.join(carve, name + ".dxbc"), "wb") as f:
                    f.write(dxbc(d))
                index.append({"name": name, "file": name + ".dxbc"})
            index.append({"name": "gone", "file": "gone.dxbc"})       # listed but missing: skipped
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f)
            src = os.path.join(tmp, "lib.bin")
            with open(src, "wb") as f:
                f.write(sample())
            out = os.path.join(tmp, "out")
            rc, text = self.run_main(src, "--out", out, "--carve", carve)
            self.assertEqual(rc, 0)
            self.assertIn("3 of 4 named by a carved shader's HASH digest, 1 not in the carve", text)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                by = {e["file"]: e for e in json.load(f)}
            self.assertEqual(by["lib_ps_00000.dxbc"]["source"], ["PS_A", "PS_A~1"])
            self.assertEqual(by["lib_cs_00003.dxbc"]["source"], ["CS_B"])
            self.assertEqual(by["lib_ms_00002.dxbc"]["source"], [])
            with open(os.path.join(out, "lib_vs_00001.dxbc"), "rb") as f:
                self.assertEqual(f.read(), dxbc(D_VS))
            self.assertFalse(os.path.exists(os.path.join(out, "lib_ms_00002.dxbc")))
            with open(os.path.join(out, "fspx", "lib_ps_00000.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["digest"], [D_PS.hex()])

    def test_translator_reads_compute_sidecar(self):
        import rdna2spirv
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "lib.bin")
            with open(src, "wb") as f:
                f.write(sample())
            out = os.path.join(tmp, "out")
            self.run_main(src, "--out", out, "--stages", "cs")
            s_path = os.path.join(out, "disasm", "lib_cs_00003.s")
            self.assertEqual(rdna2spirv.cs_dispatch_state(s_path, "lib_cs_00003"), (0xB8A, 8, 8, 1))
            self.assertEqual(rdna2spirv.native_blob(s_path, "lib_cs_00003")[:36], prologue(0x77778888))

    def test_translator_rejects_pixel_program_without_container(self):
        """A pixel program whose source isn't in the carve has no <name>.dxbc for its input signature: Unsupported for
        that shader, not a FileNotFoundError that stops the run."""
        import rdna2spirv
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "lib.bin")
            with open(src, "wb") as f:
                f.write(sample())
            out = os.path.join(tmp, "out")
            self.run_main(src, "--out", out, "--stages", "ps")
            os.makedirs(os.path.join(out, "disasm"))
            s_path = os.path.join(out, "disasm", "lib_ps_00000.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; lib_ps_00000  code 0x24..0x30 (fspx), gfx1013, 0 invalid words\n\ts_endpgm\n")
            rdna2spirv._STAGES = None                    # stage_of caches the first index.json it reads
            try:
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "no shader container"):
                    rdna2spirv.translate(s_path)
            finally:
                rdna2spirv._STAGES = None


if __name__ == "__main__":
    unittest.main()
