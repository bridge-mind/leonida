"""Authoritative FSSM wave-size detection in tools/rdna2spirv.py (stdlib only, no game data).

Tests that rdna2spirv.translate() reads u32[0xD4] bit 5 (0x20 = wave32, 0 = wave64) from the
sidecar fssm/<name>.fssm blob via fssm_header.wave_size(), cross-checks against code markers,
and falls back to the code-marker heuristic when no FSSM sidecar is present.

All blobs, DXBC containers and disassembly listings here are built by hand; nothing needs
llvm-mc, spirv-as/val or a GPU.

    python -I tests/test_fssm_wave_size.py
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fssm_header  # noqa: E402
import rdna2spirv  # noqa: E402


def make_dxbc(chunks=()):
    head = 32 + 4 * len(chunks)
    offsets, body = [], b""
    for fourcc, payload in chunks:
        offsets.append(head + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(body), len(chunks))
            + b"".join(struct.pack("<I", o) for o in offsets) + body)


def sig1_chunk(fourcc, elements):
    """Build an ISG1 or OSG1 chunk: elements = [(name, sidx, reg, mask, rw_mask, sysv, ctype)]."""
    cnt = len(elements)
    str_off = 8 + 32 * cnt
    table = bytearray(struct.pack("<II", cnt, 8))
    strings = bytearray()
    for name, sidx, reg, mask, rw_mask, sysv, ctype in elements:
        noff = str_off + len(strings) if name else 0
        if name:
            strings.extend(name.encode("ascii") + b"\x00")
        table.extend(struct.pack("<6I", 0, noff, sidx, sysv, ctype, reg))
        table.extend(bytes([mask, rw_mask, 0, 0]))
        table.extend(struct.pack("<I", 0))
    return fourcc, bytes(table + strings)


def make_fssm(wave_size=64, cs_header=None, magic=b"FSSM"):
    """Build a synthetic FSSM blob with u32[0xD4] bit 5 set for wave32 or clear for wave64."""
    data = bytearray(0x148)
    data[0:len(magic)] = magic
    flags = 0x00110020 if wave_size == 32 else 0x00110000
    struct.pack_into("<I", data, fssm_header.WAVE_FLAGS_OFFSET, flags)
    if cs_header is not None:
        rsrc2, (lx, ly, lz) = cs_header
        struct.pack_into("<4I", data, 0x138, rsrc2, lx, ly, lz)
    return bytes(data)


def translate_case(name, lines, fssm=None, dxbc=None, emulate=False):
    """Write synthetic disasm/<name>.s + <name>.dxbc + optional fssm/<name>.fssm and translate."""
    prev_emulate = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            s_path = os.path.join(disasm, name + ".s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(f"; {name}  code 0x100..0x200 (synthetic)\n")
                for ln in lines:
                    f.write("\t" + ln + "\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(make_dxbc() if dxbc is None else dxbc)
            if fssm is not None:
                fdir = os.path.join(tmp, "fssm")
                os.mkdir(fdir)
                with open(os.path.join(fdir, name + ".fssm"), "wb") as f:
                    f.write(fssm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gen = rdna2spirv.translate.last_gen
                meta = rdna2spirv.shader_meta(gen, spv)
                gc.collect()
            return spv, gen, meta
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emulate


CS_HEADER_64 = (0x8088, (64, 1, 1))
CS_HEADER_32 = (0x8088, (32, 1, 1))

STRAIGHT_CS = [
    "v_mov_b32_e32 v1, 1 ; encoding: [0x81,0x02,0x02,0x7e]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]

READLANE_CS = [
    "v_mov_b32_e32 v1, 40 ; encoding: [0xa8,0x02,0x02,0x7e]",
    "v_readlane_b32 s4, v1, 40 ; encoding: [0x04,0x00,0x60,0xd7,0x01,0x51,0x01,0x00]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]

STRAIGHT_PS = [
    "v_mov_b32_e32 v2, 1.0 ; encoding: [0xf2,0x02,0x04,0x7e]",
    "exp mrt0 v2, v2, v2, v2 done vm ; encoding: [0x0f,0x18,0x00,0xf8,0x02,0x02,0x02,0x02]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]

STRAIGHT_VS = [
    "v_mov_b32_e32 v1, 1.0 ; encoding: [0xf2,0x02,0x02,0x7e]",
    "exp pos0 v1, v1, v1, v1 done ; encoding: [0xcf,0x08,0x00,0xf8,0x01,0x01,0x01,0x01]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]

STRAIGHT_GS = [
    "s_mov_b32 m0, 0 ; encoding: [0x80,0x03,0xfc,0xbe]",
    "s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0) ; encoding: [0x22,0x00,0x90,0xbf]",
    "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP) ; encoding: [0x03,0x00,0x90,0xbf]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]

GS_DXBC = make_dxbc([
    sig1_chunk(b"ISG1", [("SV_Position", 0, 0, 0xF, 0xF, 1, 3)]),
    sig1_chunk(b"OSG1", [("SV_Position", 0, 0, 0xF, 0xF, 1, 3)]),
])


class FssmWaveSizeTest(unittest.TestCase):
    def test_straight_line_cs_wave64_from_fssm(self):
        _, gen, meta = translate_case("cs_w64", STRAIGHT_CS, fssm=make_fssm(64, CS_HEADER_64))
        self.assertTrue(gen.wave64)
        self.assertTrue(meta["wave64"])
        self.assertEqual(meta["subgroup_size"], 64)

    def test_straight_line_cs_readlane_uses_64_lane_mask(self):
        # 11 markerless wave64 CS in the corpus use v_readlane_b32; wave64 must mask with 63 (%c3f), not 31 (%c1f).
        spv64, gen64, meta64 = translate_case("cs_rl64", READLANE_CS, fssm=make_fssm(64, CS_HEADER_64))
        self.assertTrue(gen64.wave64)
        self.assertEqual(meta64["subgroup_size"], 64)
        self.assertIn("%c3f = OpConstant %uint 63", spv64)
        self.assertIn("OpBitwiseAnd %uint %c28 %c3f", spv64)

        spv32, gen32, meta32 = translate_case("cs_rl32", READLANE_CS, fssm=make_fssm(32, CS_HEADER_32))
        self.assertFalse(gen32.wave64)
        self.assertEqual(meta32["subgroup_size"], 32)
        self.assertIn("%c1f = OpConstant %uint 31", spv32)
        self.assertIn("OpBitwiseAnd %uint %c28 %c1f", spv32)
        self.assertNotIn("%c3f = OpConstant %uint 63", spv32)

    def test_straight_line_ps_vs_gs_wave64_from_fssm(self):
        for name, lines, dxbc in (
            ("ps_w64", STRAIGHT_PS, None),
            ("vs_w64", STRAIGHT_VS, None),
            ("gs_w64", STRAIGHT_GS, GS_DXBC),
        ):
            with self.subTest(stage=name[:2]):
                _, gen, meta = translate_case(name, lines, fssm=make_fssm(64), dxbc=dxbc)
                self.assertTrue(gen.wave64)
                self.assertTrue(meta["wave64"])
                self.assertEqual(meta["subgroup_size"], 64)
                self.assertTrue(meta["wave64_emulated"])
                self.assertEqual(meta["wave64_graphics"], "A")

    def test_wave32_from_fssm_for_cs_and_ps(self):
        _, gen_cs, meta_cs = translate_case("cs_w32", STRAIGHT_CS, fssm=make_fssm(32, CS_HEADER_32))
        self.assertFalse(gen_cs.wave64)
        self.assertFalse(meta_cs["wave64"])
        self.assertEqual(meta_cs["subgroup_size"], 32)

        _, gen_ps, meta_ps = translate_case("ps_w32", STRAIGHT_PS, fssm=make_fssm(32))
        self.assertFalse(gen_ps.wave64)
        self.assertFalse(meta_ps["wave64"])
        self.assertEqual(meta_ps["subgroup_size"], 32)
        self.assertNotIn("wave64_graphics", meta_ps)

    def test_missing_or_non_fssm_sidecar_falls_back_to_wave64_re(self):
        # PS with no sidecar: straight-line falls back to wave32; WAVE64_RE falls back to wave64.
        _, gen_ps32, meta_ps32 = translate_case("ps_nosidecar", STRAIGHT_PS, fssm=None)
        self.assertFalse(gen_ps32.wave64)
        self.assertEqual(meta_ps32["subgroup_size"], 32)

        w64_ps = ["s_mov_b64 exec, -1 ; encoding: [0xc1,0x04,0xfe,0xbe]"] + STRAIGHT_PS
        _, gen_ps64, meta_ps64 = translate_case("ps_nosidecar64", w64_ps, fssm=None)
        self.assertTrue(gen_ps64.wave64)
        self.assertEqual(meta_ps64["subgroup_size"], 64)

        # CS with zero-prefixed synthetic header (no FSSM magic, as tests/conftest.py writes):
        # wave_size() returns None, so translation falls back to WAVE64_RE.
        zero_cs = make_fssm(64, CS_HEADER_64, magic=b"\x00\x00\x00\x00")
        _, gen_cs32, _ = translate_case("cs_zero_hdr", STRAIGHT_CS, fssm=zero_cs)
        self.assertFalse(gen_cs32.wave64)

        w64_cs = ["s_mov_b64 exec, -1 ; encoding: [0xc1,0x04,0xfe,0xbe]"] + STRAIGHT_CS
        _, gen_cs64, _ = translate_case("cs_zero_hdr64", w64_cs, fssm=zero_cs)
        self.assertTrue(gen_cs64.wave64)

    def test_contradicting_fssm_and_code_is_refused(self):
        w64_code = ["s_and_saveexec_b64 s[0:1], vcc ; encoding: [0x6a,0x24,0x80,0xbe]"] + STRAIGHT_PS
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "FSSM wave32 header contradicts wave64 code"):
            translate_case("ps_bad32", w64_code, fssm=make_fssm(32))

        w32_code = ["s_and_saveexec_b32 s0, vcc_lo ; encoding: [0x6a,0x3c,0x80,0xbe]"] + STRAIGHT_PS
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "FSSM wave64 header contradicts wave32 code"):
            translate_case("ps_bad64", w32_code, fssm=make_fssm(64))

    def test_wave64_emulate_runs_on_straight_line_wave64_cs_from_fssm(self):
        spv, gen, meta = translate_case("cs_emul64", READLANE_CS,
                                        fssm=make_fssm(64, CS_HEADER_64), emulate=True)
        self.assertTrue(gen.wave64)
        self.assertTrue(gen.emulate)
        self.assertTrue(meta["wave64"])
        self.assertTrue(meta["wave64_emulated"])
        self.assertEqual(meta["wave64_exchange"], "workgroup")
        self.assertIsNone(meta["subgroup_size"])
        self.assertEqual(meta["control_flow"], "emulated wave64 rounds")
        self.assertIn("%x_buf", spv)
        self.assertNotIn("OpGroupNonUniform", spv)

        # Wave32 CS with --wave64-emulate must stay native wave32.
        spv32, gen32, meta32 = translate_case("cs_emul32", READLANE_CS,
                                              fssm=make_fssm(32, CS_HEADER_32), emulate=True)
        self.assertFalse(gen32.wave64)
        self.assertFalse(gen32.emulate)
        self.assertEqual(meta32["subgroup_size"], 32)
        self.assertNotIn("%x_buf", spv32)


if __name__ == "__main__":
    unittest.main()
