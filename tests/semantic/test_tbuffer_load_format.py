"""Semantic interpreter tests for 32-bit (non-d16) tbuffer_load_format_{x,xy,xyz,xyzw}.

Covers:
- All 4 component widths (x, xy, xyz, xyzw) with 32_32_32_32_FLOAT and 16_16_16_16_FLOAT,
  verifying that unloaded trailing VGPRs retain their pre-loaded sentinel values.
- Normalized and integer formats: 8_8_8_8_UNORM, 8_8_8_8_SNORM (with negatives),
  16_16_16_16_UINT, and 16_16_16_16_SINT (with negatives).
- Precedence of the MTBUF format:[BUF_FMT_...] modifier (and identity DST_SEL) over
  an incompatible V# FORMAT / DST_SEL.
- Out-of-bounds behavior on num_records (returns 0 without touching trailing VGPRs).
- In-memory negative controls on the translated SPIR-V text.
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import spirv_mini_interp  # noqa: E402
import support  # noqa: E402


def fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    """Build the compute-shader FSSM header (COMPUTE_PGM_RSRC2 and thread-group size)."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# 4 user SGPRs (s[0:3] output V#), workgroup id in s4, 64x1x1 threads.
CS_FSSM = fssm(0x8088, (64, 1, 1))
# Output V# passed via push constants in s[0:3]: base 0, num_records 0xffffffff, 32_FLOAT, OOB_SELECT 3.
OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

IN_BASE = 0x1000
OUT_BASE = 0x200
SENTINELS = (0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD)

# V# word3 fields (bits 0..11 DST_SEL, bits 12..18 FORMAT, bits 28..29 OOB_SELECT).
DST_SEL_XYZW = 4 | (5 << 3) | (6 << 6) | (7 << 9)
DST_SEL_ZERO_ONE = 0 | (0 << 3) | (1 << 6) | (1 << 9)
FMT_U32 = 20
FMT_UINT8X4 = 60
FMT_U32X4 = 75


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def vdst_for_width(ncomp: int) -> str:
    return "v2" if ncomp == 1 else f"v[2:{1 + ncomp}]"


def build_tbuffer_load_spv(
    ncomp: int,
    fmt_mod: str,
    *,
    offset: int = 0,
    num_records: int = 0x100,
    vsharp_word3: int = (FMT_U32 << 12) | DST_SEL_XYZW | 0x30000000,
) -> str:
    """Translate a compute shader that pre-loads v[2:5] with sentinels, runs tbuffer_load_format_*,
    and stores v[2:5] to OUT_BASE."""
    suffix = "xyzw"[:ncomp]
    vdst = vdst_for_width(ncomp)
    mod_str = f" format:[{fmt_mod}]" if fmt_mod else ""
    lines = [
        f"s_mov_b32 s8, {IN_BASE:#x}",
        "s_mov_b32 s9, 0",
        f"s_mov_b32 s10, {num_records:#x}",
        f"s_mov_b32 s11, {vsharp_word3:#x}",
        *(f"v_mov_b32 v{2 + i}, {val:#x}" for i, val in enumerate(SENTINELS)),
        f"v_mov_b32 v1, {offset:#x}",
        f"tbuffer_load_format_{suffix} {vdst}, v1, s[8:11], 0{mod_str} offen",
        "s_waitcnt vmcnt(0)",
        f"v_mov_b32 v0, {OUT_BASE:#x}",
        "buffer_store_dwordx4 v[2:5], v0, s[0:3], 0 offen",
        "s_endpgm",
    ]
    return support.translate_lines(f"cs_tbuf_load_{suffix}", lines, fssm=CS_FSSM)


def run_spv(spv: str, memory: dict[int, int]) -> list[int]:
    mem = dict(memory)
    outputs = spirv_mini_interp.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=OUT_VSHARP + [0] * 28,
        memory=mem,
    )
    if outputs:
        raise AssertionError(f"unexpected shader outputs: {outputs}")
    return [mem.get(OUT_BASE + 4 * i, 0) for i in range(4)]


class TbufferLoadWidthsFloatTests(unittest.TestCase):
    def test_f32x4_all_widths_preserve_trailing_vgprs(self):
        vals = (1.5, -2.25, 0.125, 42.0)
        words = [f32_bits(v) for v in vals]
        mem = {IN_BASE + 4 * i: w for i, w in enumerate(words)}
        for ncomp in (1, 2, 3, 4):
            with self.subTest(ncomp=ncomp):
                spv = build_tbuffer_load_spv(ncomp, "BUF_FMT_32_32_32_32_FLOAT")
                got = run_spv(spv, mem)
                want = words[:ncomp] + list(SENTINELS[ncomp:])
                self.assertEqual(got, want)

    def test_f16x4_all_widths_preserve_trailing_vgprs(self):
        vals = (1.0, -0.5, 3.5, -16.0)
        halves = [f16_bits(v) for v in vals]
        mem = {
            IN_BASE: halves[0] | (halves[1] << 16),
            IN_BASE + 4: halves[2] | (halves[3] << 16),
        }
        expected = [f32_bits(v) for v in vals]
        for ncomp in (1, 2, 3, 4):
            with self.subTest(ncomp=ncomp):
                spv = build_tbuffer_load_spv(ncomp, "BUF_FMT_16_16_16_16_FLOAT")
                got = run_spv(spv, mem)
                want = expected[:ncomp] + list(SENTINELS[ncomp:])
                self.assertEqual(got, want)


class TbufferLoadNormAndIntTests(unittest.TestCase):
    def test_unorm8x4(self):
        # 0x00 -> 0.0, 0x33 -> 51/255 = 0.2, 0xCC -> 204/255 = 0.8, 0xFF -> 1.0
        packed = 0x00 | (0x33 << 8) | (0xCC << 16) | (0xFF << 24)
        mem = {IN_BASE: packed}
        spv = build_tbuffer_load_spv(4, "BUF_FMT_8_8_8_8_UNORM")
        got = run_spv(spv, mem)
        want = [f32_bits(0.0), f32_bits(51.0 / 255.0), f32_bits(204.0 / 255.0), f32_bits(1.0)]
        self.assertEqual(got, want)

    def test_snorm8x4_with_negatives(self):
        # 0x80 (-128) -> -1.0, 0x81 (-127) -> -1.0, 0xC0 (-64) -> -64/127, 0x7F (127) -> 1.0
        packed = 0x80 | (0x81 << 8) | (0xC0 << 16) | (0x7F << 24)
        mem = {IN_BASE: packed}
        spv = build_tbuffer_load_spv(4, "BUF_FMT_8_8_8_8_SNORM")
        got = run_spv(spv, mem)
        want = [f32_bits(-1.0), f32_bits(-1.0), f32_bits(-64.0 / 127.0), f32_bits(1.0)]
        self.assertEqual(got, want)

    def test_uint16x4(self):
        mem = {
            IN_BASE: 0x0000 | (0x0001 << 16),
            IN_BASE + 4: 0x7FFF | (0xFFFF << 16),
        }
        spv = build_tbuffer_load_spv(4, "BUF_FMT_16_16_16_16_UINT")
        got = run_spv(spv, mem)
        self.assertEqual(got, [0x00000000, 0x00000001, 0x00007FFF, 0x0000FFFF])

    def test_sint16x4_with_negatives(self):
        mem = {
            IN_BASE: 0x8000 | (0xFFFF << 16),
            IN_BASE + 4: 0x0000 | (0x7FFF << 16),
        }
        spv = build_tbuffer_load_spv(4, "BUF_FMT_16_16_16_16_SINT")
        got = run_spv(spv, mem)
        self.assertEqual(got, [0xFFFF8000, 0xFFFFFFFF, 0x00000000, 0x00007FFF])


class TbufferPrecedenceAndOobTests(unittest.TestCase):
    def test_modifier_overrides_incompatible_vsharp_format_and_dst_sel(self):
        # V# has incompatible FORMAT (FMT_U32X4) and non-identity DST_SEL (0, 0, 1, 1).
        # MTBUF instruction format:[BUF_FMT_16_16_16_16_SINT] and identity XYZW must win.
        mem = {
            IN_BASE: 0xFFF0 | (0x0002 << 16),
            IN_BASE + 4: 0xFFFE | (0x1234 << 16),
        }
        incompatible_w3 = (FMT_U32X4 << 12) | DST_SEL_ZERO_ONE | 0x30000000
        spv = build_tbuffer_load_spv(4, "BUF_FMT_16_16_16_16_SINT", vsharp_word3=incompatible_w3)
        got = run_spv(spv, mem)
        self.assertEqual(got, [0xFFFFFFF0, 0x00000002, 0xFFFFFFFE, 0x00001234])

    def test_out_of_bounds_returns_zero_and_preserves_trailing_vgpr(self):
        mem = {IN_BASE + 4 * i: 0x12345670 + i for i in range(4)}
        # num_records = 16 bytes; offset = 16 is OOB, and offset = 4 with 16-byte element straddles OOB.
        for offset in (16, 4):
            with self.subTest(offset=offset):
                spv = build_tbuffer_load_spv(
                    3, "BUF_FMT_32_32_32_32_FLOAT", offset=offset, num_records=16
                )
                got = run_spv(spv, mem)
                self.assertEqual(got, [0, 0, 0, SENTINELS[3]])


class TbufferNegativeControlsTests(unittest.TestCase):
    def test_mutated_swizzle_selector_breaks_result(self):
        mem = {
            IN_BASE: 0x11111111,
            IN_BASE + 4: 0x22222222,
            IN_BASE + 8: 0x33333333,
            IN_BASE + 12: 0x44444444,
        }
        spv = build_tbuffer_load_spv(4, "BUF_FMT_32_32_32_32_FLOAT")
        self.assertEqual(run_spv(spv, mem), [0x11111111, 0x22222222, 0x33333333, 0x44444444])

        # Mutate the DST_SEL selector check for code 5 (Y) so it matches code 4 (X) instead.
        mutated, count = re.subn(
            r"(OpIEqual %bool %\S+ )%c5\b",
            r"\1%c4",
            spv,
        )
        self.assertGreater(count, 0, "expected DST_SEL selector comparison against %c5 in SPIR-V")
        got_mutated = run_spv(mutated, mem)
        self.assertNotEqual(got_mutated, [0x11111111, 0x22222222, 0x33333333, 0x44444444])

    def test_mutated_oob_guard_breaks_oob_zero_result(self):
        mem = {IN_BASE + 4 * i: f32_bits(float(i + 1)) for i in range(4)}
        spv = build_tbuffer_load_spv(
            4, "BUF_FMT_32_32_32_32_FLOAT", offset=16, num_records=16
        )
        self.assertEqual(run_spv(spv, mem), [0, 0, 0, 0])

        # Mutate the final OOB OpSelect (%uint <in_b> <val> %c0) to return %c1 (1) instead of 0 on OOB.
        mutated, count = re.subn(
            r"(OpSelect %uint %\S+ %\S+ )%c0\b",
            r"\1%c1",
            spv,
        )
        self.assertGreater(count, 0, "expected OOB OpSelect against %c0 in SPIR-V")
        got_mutated = run_spv(mutated, mem)
        self.assertNotEqual(got_mutated, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
