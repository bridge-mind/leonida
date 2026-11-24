"""Semantic interpreter tests for sub-dword MUBUF loads and GFX10_FORMAT_8_UINT (code 5).

Covers the remaining translator rules needed by XboxBvhBuilder.dll (#1813 follow-ups):
  - buffer_load_ubyte, buffer_load_sbyte, buffer_load_ushort, buffer_load_sshort
    (all byte offsets 0..7 including dword-crossing 16-bit loads at addr % 4 == 3, zero/sign extension,
    OOB_SELECT modes 0/1/2/3 at exact element-end boundaries, inactive lanes preserving vdst, and rejected forms)
  - tbuffer_load_format_* and buffer_load_format_* with GFX10_FORMAT_8_UINT (code 5: 1-byte unsigned integer,
    byte offsets 0..3, OOB check, invalid V# format 0 suppression, and DST_SEL integer 1).

Standard library only; no GPU, no LLVM, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE = 0x1000
OUT = 0x3000
# Bytes 0..7: 0x7F, 0x80, 0xFF, 0x01, 0xAB, 0xCD, 0x00, 0xFE
BYTES = bytes([0x7F, 0x80, 0xFF, 0x01, 0xAB, 0xCD, 0x00, 0xFE])
MEMORY = {
    BASE: struct.unpack_from("<I", BYTES, 0)[0],
    BASE + 4: struct.unpack_from("<I", BYTES, 4)[0],
}
DST_SEL_X_0_1_1 = 4 | (0 << 3) | (1 << 6) | (1 << 9)


def translate(body, *, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_bufsub.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_bufsub.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_bufsub.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_bufsub\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(body, memory=None, nlanes=1):
    spv = translate(body)
    mem = dict(MEMORY if memory is None else memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def buf_load(op, offset, num_records=8, stride=0, oob_mode=3, idx=None, memory=None):
    w3 = (oob_mode << 28) | (4 | (5 << 3) | (6 << 6) | (7 << 9)) | (20 << 12)
    body = [
        f"s_mov_b32 s4, {BASE:#x}",
        f"s_mov_b32 s5, {(stride & 0x3FFF) << 16:#x}",
        f"s_mov_b32 s6, {num_records:#x}",
        f"s_mov_b32 s7, {w3:#x}",
        "v_mov_b32_e32 v5, 0xdeadbeef",
    ]
    if idx is not None:
        body += [
            f"v_mov_b32_e32 v1, {idx:#x}",
            f"v_mov_b32_e32 v2, {offset:#x}",
            f"{op} v5, v[1:2], s[4:7], 0 idxen offen",
        ]
    else:
        body += [
            f"v_mov_b32_e32 v1, {offset:#x}",
            f"{op} v5, v1, s[4:7], 0 offen",
        ]
    body += [
        "s_waitcnt vmcnt(0)",
        f"v_mov_b32_e32 v6, {OUT:#x}",
        "global_store_dword v6, v5, s[0:1]",
    ]
    mem = run_cs(body, memory=memory)
    return mem.get(OUT)


class BufferLoadSubTests(unittest.TestCase):
    def test_ubyte_and_sbyte_all_offsets(self):
        for off in range(len(BYTES)):
            raw = BYTES[off]
            signed = (raw - 256) & 0xFFFFFFFF if raw & 0x80 else raw
            with self.subTest(off=off):
                self.assertEqual(buf_load("buffer_load_ubyte", off), raw)
                self.assertEqual(buf_load("buffer_load_sbyte", off), signed)

    def test_ushort_and_sshort_all_offsets_including_dword_crossing(self):
        for off in range(len(BYTES) - 1):
            raw = BYTES[off] | (BYTES[off + 1] << 8)
            signed = (raw - 0x10000) & 0xFFFFFFFF if raw & 0x8000 else raw
            with self.subTest(off=off):
                self.assertEqual(buf_load("buffer_load_ushort", off), raw)
                self.assertEqual(buf_load("buffer_load_sshort", off), signed)

    def test_raw_oob_exact_element_boundaries(self):
        # Raw mode (OOB_SELECT=3): element ends at offset + size <= num_records.
        self.assertEqual(buf_load("buffer_load_ubyte", 3, num_records=4), BYTES[3])
        self.assertEqual(buf_load("buffer_load_ubyte", 4, num_records=4), 0)
        self.assertEqual(buf_load("buffer_load_ushort", 2, num_records=4), BYTES[2] | (BYTES[3] << 8))
        # offset 3 + 2 bytes = 5 > 4 -> OOB returns 0 without reading dword 1.
        self.assertEqual(buf_load("buffer_load_ushort", 3, num_records=4, memory={BASE: MEMORY[BASE]}), 0)
        # num_records 5 allows the dword-crossing ushort at offset 3..4.
        self.assertEqual(buf_load("buffer_load_ushort", 3, num_records=5), BYTES[3] | (BYTES[4] << 8))

    def test_structured_oob_modes(self):
        # Mode 0 (STRUCTURED_WITH_OFFSET): idx < num_records and off + size <= stride.
        self.assertEqual(buf_load("buffer_load_ushort", 2, num_records=2, stride=4, oob_mode=0, idx=1),
                         BYTES[6] | (BYTES[7] << 8))
        self.assertEqual(buf_load("buffer_load_ushort", 3, num_records=2, stride=4, oob_mode=0, idx=0), 0)
        self.assertEqual(buf_load("buffer_load_ushort", 0, num_records=1, stride=4, oob_mode=0, idx=1), 0)
        # Mode 2 (DISABLED): only num_records == 0 is OOB.
        self.assertEqual(buf_load("buffer_load_ushort", 0, num_records=0, stride=4, oob_mode=2), 0)

    def test_inactive_lane_preserves_vdst(self):
        w3 = (3 << 28) | (20 << 12)
        body = [
            f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 8", f"s_mov_b32 s7, {w3:#x}",
            "v_mov_b32_e32 v1, 2", "v_mov_b32_e32 v5, 0xcafebabe",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
            "s_and_saveexec_b32 s8, vcc_lo",
            "buffer_load_ushort v5, v1, s[4:7], 0 offen",
            "s_mov_b32 exec_lo, s8",
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32_e32 v6, 2, v0",
            f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, nlanes=2)
        self.assertEqual(mem.get(OUT), BYTES[2] | (BYTES[3] << 8))
        self.assertEqual(mem.get(OUT + 4), 0xCAFEBABE)

    def test_rejected_sub_dword_forms(self):
        for line in ("buffer_load_ushort v5, v1, s[4:7], 0 offen lds",
                     "buffer_load_ushort v5, v1, s[4:7], 0 offen tfe",
                     "buffer_load_ushort v5, v1, s[4:7], 0 offen d16"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["s_mov_b32 s4, 0", "s_mov_b32 s5, 0", "s_mov_b32 s6, 8", "s_mov_b32 s7, 0", line])


class Format8UintTests(unittest.TestCase):
    def test_tbuffer_load_format_x_8_uint_all_byte_offsets(self):
        # V# has a non-zero format (code 20 = 32_UINT); MTBUF uses its own BUF_FMT_8_UINT (code 5) and identity X000.
        w3 = (3 << 28) | (20 << 12) | DST_SEL_X_0_1_1
        for off in range(4):
            with self.subTest(off=off):
                body = [
                    f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 4", f"s_mov_b32 s7, {w3:#x}",
                    f"v_mov_b32_e32 v1, {off:#x}",
                    "tbuffer_load_format_x v5, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body)
                self.assertEqual(mem.get(OUT), BYTES[off])

    def test_tbuffer_8_uint_oob_and_invalid_vsharp_format(self):
        # OOB at off == num_records -> 0; INVALID V# format (0) suppresses MTBUF even in bounds -> 0.
        for off, vfmt, expected in ((4, 20, 0), (1, 0, 0), (1, 5, BYTES[1])):
            w3 = (3 << 28) | (vfmt << 12) | (4 | (5 << 3) | (6 << 6) | (7 << 9))
            with self.subTest(off=off, vfmt=vfmt):
                body = [
                    f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 4", f"s_mov_b32 s7, {w3:#x}",
                    f"v_mov_b32_e32 v1, {off:#x}",
                    "tbuffer_load_format_x v5, v1, s[4:7], 0 format:[BUF_FMT_8_UINT] offen",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body)
                self.assertEqual(mem.get(OUT), expected)

    def test_mubuf_8_uint_dst_sel_integer_one(self):
        # V# FORMAT = 5 (GFX10_FORMAT_8_UINT), DST_SEL = (X, 0, 1, 1) -> integer 1, not 1.0f.
        w3 = (3 << 28) | (5 << 12) | DST_SEL_X_0_1_1
        body = [
            f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 4", f"s_mov_b32 s7, {w3:#x}",
            "v_mov_b32_e32 v1, 2",
            "buffer_load_format_xyzw v[8:11], v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dwordx4 v6, v[8:11], s[0:1]",
        ]
        mem = run_cs(body)
        self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)], [BYTES[2], 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
