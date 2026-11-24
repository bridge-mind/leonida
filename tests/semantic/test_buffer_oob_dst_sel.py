"""Semantic regression tests for out-of-bounds typed buffer loads with DST_SEL=1 (SHADER-429).

Measured natively on RDNA2 (gfx1036, wave32 and wave64, 746 kernels via tools/hw_oracle):
  * When a MUBUF buffer_load_format_{x,xy,xyz,xyzw} or buffer_load_format_d16_{x,xy,xyz,xyzw} is out of bounds
    (any OOB_SELECT mode, num_records == 0, partial-element spill, or swizzled OOB), memory is not read and
    data channels (DST_SEL 4..7) decode as 0, while V# DST_SEL still runs: DST_SEL=0 produces 0 and DST_SEL=1
    produces 1.0 (0x3F800000, d16 0x3C00) iff V# FORMAT[6:0] is a valid float/normalized/scaled code (1..77
    ending in _UNORM, _SNORM, _USCALED, _SSCALED, or _FLOAT), and integer 1 (0x00000001, d16 0x0001) for
    UINT/SINT, INVALID (0), and reserved codes (78..127).
  * d16_x and d16_xyz preserve the unaddressed upper 16 bits of the final destination VGPR.
  * MTBUF tbuffer_load_format_* ignores V# DST_SEL on both in-bounds and out-of-bounds / INVALID loads, using
    identity selectors X000/XY00/XYZ0/XYZW so out-of-bounds and FORMAT=0 loads return 0 in every addressed channel.
  * Inactive EXEC lanes leave destination VGPRs untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

BASE = 0x1000
OUT_V = 0x31016FAC  # RAW output V# at base 0
DST_X_0_1_1 = 4 | (0 << 3) | (1 << 6) | (1 << 9)
DST_1_Y_Z_W = 1 | (5 << 3) | (6 << 6) | (7 << 9)
DST_1_1_1_1 = 1 | (1 << 3) | (1 << 6) | (1 << 9)
SEED0, SEED1, SEED2, SEED3 = 0x12345678, 0xA9BCDEF0, 0xCAFEBABE, 0x0F0E0D0C
FLOAT_SUFFIXES = ("_UNORM", "_SNORM", "_USCALED", "_SSCALED", "_FLOAT")

with open(os.path.join(TOOLS_DIR, "gfx10_formats.json"), encoding="utf-8") as _f:
    GFX10_FORMAT_NAMES = {int(k): v for k, v in json.load(_f).items()}

# 12 user SGPRs (s[0:11] from push constants), no TGID SGPRs, 64x1x1 threads.
CS_12_USER_FSSM = tv._fssm(12 << 1, (64, 1, 1))


def word3(fmt_code: int, dst_sel: int, oob_select: int = 3, swz_idx_stride: int = 0) -> int:
    return (
        (dst_sel & 0xFFF)
        | ((fmt_code & 0x7F) << 12)
        | ((swz_idx_stride & 0x3) << 21)
        | (1 << 24)
        | ((oob_select & 0x3) << 28)
    )


class OobTypedLoadDstSelTests(unittest.TestCase):
    def _run_cs(
        self,
        lines: list[str],
        *,
        push_constants: list[int] | None = None,
        preload: dict[int, int] | None = None,
        wave64: bool = False,
        name: str = "cs_oob_dst_sel",
    ) -> tuple[tv.ReadLog, str]:
        spv = support.translate_lines(name, lines, fssm=CS_12_USER_FSSM)
        mem = tv.ReadLog(preload if preload is not None else {
            BASE: 0x3F000000,
            BASE + 4: 0x40000000,
            BASE + 8: 0x00030002,
            BASE + 12: 0x04030201,
        })
        pcs = push_constants if push_constants is not None else ([0, 0, 0x100, OUT_V] + [0] * 28)
        if wave64:
            si.run_workgroup(
                spv,
                64,
                64,
                inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                push_constants=pcs,
                memory=mem,
            )
        else:
            support.run_fragment(
                spv,
                inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                push_constants=pcs,
                memory=mem,
            )
        return mem, spv

    def test_all_supported_formats_preserve_dst_sel_one_when_oob(self):
        """Runtime V# sweep across every format in Gen.FORMATS (xyzw, d16_xyzw, d16_xyz, d16_x)."""
        lines = [
            "v_cmpx_eq_u32_e32 0, v0",
            # s[4:7] and s[8:11] come from push constants; offset v1 = 0x100 is past num_records = 16.
            "v_mov_b32 v1, 0x100",
            f"v_mov_b32 v4, {SEED0:#x}",
            f"v_mov_b32 v5, {SEED1:#x}",
            f"v_mov_b32 v6, {SEED2:#x}",
            f"v_mov_b32 v7, {SEED3:#x}",
            "buffer_load_format_xyzw v[4:7], v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[4:7], off, s[0:3], 0",
            f"v_mov_b32 v8, {SEED0:#x}",
            f"v_mov_b32 v9, {SEED1:#x}",
            "buffer_load_format_d16_xyzw v[8:9], v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx2 v[8:9], off, s[0:3], 0 offset:16",
            f"v_mov_b32 v10, {SEED0:#x}",
            f"v_mov_b32 v11, {SEED1:#x}",
            "buffer_load_format_d16_xyz v[10:11], v1, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx2 v[10:11], off, s[0:3], 0 offset:24",
            f"v_mov_b32 v12, {SEED0:#x}",
            "buffer_load_format_d16_x v12, v1, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dword v12, off, s[0:3], 0 offset:32",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_oob_all_fmts", lines, fssm=CS_12_USER_FSSM)
        for code, (_, kind, _) in sorted(rdna2spirv.Gen.FORMATS.items()):
            is_int = kind in ("u32", "i32", "uint", "sint")
            one32 = 1 if is_int else 0x3F800000
            one16 = 1 if is_int else 0x3C00
            pcs = (
                [0, 0, 0x100, OUT_V]
                + [BASE, 0, 16, word3(code, DST_X_0_1_1, 3)]
                + [BASE, 0, 16, word3(code, DST_1_Y_Z_W, 3)]
                + [0] * 20
            )
            mem = tv.ReadLog({BASE: 0x3F000000, BASE + 4: 0x40000000, BASE + 8: 0x00030002, BASE + 12: 0x04030201})
            support.run_fragment(
                spv,
                inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                push_constants=pcs,
                memory=mem,
            )
            with self.subTest(code=code, kind=kind):
                self.assertEqual(mem.reads, [])
                self.assertEqual([dict.get(mem, 4 * i, 0) for i in range(4)], [0, 0, one32, one32])
                self.assertEqual([dict.get(mem, 16 + 4 * i, 0) for i in range(2)], [0, (one16 << 16) | one16])
                self.assertEqual(
                    [dict.get(mem, 24 + 4 * i, 0) for i in range(2)],
                    [one16, SEED1 & 0xFFFF0000],
                )
                self.assertEqual(dict.get(mem, 32, 0), (SEED0 & 0xFFFF0000) | one16)

    def test_all_128_gfx10_7bit_format_codes_match_rdna2_hardware_table(self):
        """All 128 7-bit V# FORMAT codes (0..127) match the native gfx1036 sweep for both f32 and d16."""
        lines = [
            "v_cmpx_eq_u32_e32 0, v0",
            "v_mov_b32 v1, 0x100",
            f"v_mov_b32 v4, {SEED0:#x}",
            f"v_mov_b32 v5, {SEED0:#x}",
            "buffer_load_format_x v4, v1, s[4:7], 0 offen",
            "buffer_load_format_d16_x v5, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx2 v[4:5], off, s[0:3], 0",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_oob_128_fmts", lines, fssm=CS_12_USER_FSSM)
        for code in range(128):
            is_flt = GFX10_FORMAT_NAMES.get(code, "").endswith(FLOAT_SUFFIXES)
            want32 = 0x3F800000 if is_flt else 1
            want16 = (SEED0 & 0xFFFF0000) | (0x3C00 if is_flt else 1)
            pcs = [0, 0, 0x100, OUT_V, BASE, 0, 16, word3(code, 1, 3)] + [0] * 24
            mem = tv.ReadLog({BASE: 0xDEADBEEF})
            support.run_fragment(
                spv,
                inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                push_constants=pcs,
                memory=mem,
            )
            with self.subTest(code=code, name=GFX10_FORMAT_NAMES.get(code, "RESERVED")):
                self.assertEqual(mem.reads, [])
                self.assertEqual([dict.get(mem, 0, 0), dict.get(mem, 4, 0)], [want32, want16])

    def test_oob_select_modes_partial_spill_unaligned_and_swizzled(self):
        """Literal V# checks across OOB_SELECT 0..3, partial element spill, mid-dword OOB offset, and swizzled OOB."""
        fmt_f32x4, fmt_u32, fmt_i32 = 77, 20, 21
        cases = [
            # Partial element spill: 16-byte RGBA32F at offset 8 of a 16-byte raw buffer.
            ("partial_spill", 0, 16, word3(fmt_f32x4, DST_X_0_1_1, 3), "v1", [8], "offen", [0, 0, 0x3F800000, 0x3F800000]),
            # Mid-dword OOB byte offset (bsh = 24): shifts on zero words stay defined and produce 0 / 1.0f.
            ("unaligned_oob", 0, 16, word3(fmt_f32x4, DST_X_0_1_1, 3), "v1", [0x103], "offen", [0, 0, 0x3F800000, 0x3F800000]),
            # Mode 0 structured: idx 4 >= num_records 4 (stride 16).
            ("mode0_idx", 16 << 16, 4, word3(fmt_f32x4, DST_X_0_1_1, 0), "v[1:2]", [4, 0], "idxen offen", [0, 0, 0x3F800000, 0x3F800000]),
            # Mode 0 structured: idx 0 < num_records 4, off 16 + 4 > stride 16.
            ("mode0_off", 16 << 16, 4, word3(fmt_u32, DST_X_0_1_1, 0), "v[1:2]", [0, 16], "idxen offen", [0, 0, 1, 1]),
            # Mode 1 index-only: idx 4 >= num_records 4.
            ("mode1_idx", 4 << 16, 4, word3(fmt_i32, DST_X_0_1_1, 1), "v1", [4], "idxen", [0, 0, 1, 1]),
            # Mode 2 disabled except when num_records == 0.
            ("mode2_nr0", 0, 0, word3(fmt_f32x4, DST_X_0_1_1, 2), "v1", [0], "offen", [0, 0, 0x3F800000, 0x3F800000]),
            # Swizzled OOB: SWIZZLE_ENABLE = 1, stride 16, idx 4 >= num_records 4.
            ("swizzled_oob", (1 << 31) | (16 << 16), 4, word3(fmt_f32x4, DST_X_0_1_1, 0), "v[1:2]", [4, 0], "idxen offen", [0, 0, 0x3F800000, 0x3F800000]),
        ]
        for tag, w1, nr, w3, vaddr, vals, mods, want in cases:
            setup = [f"v_mov_b32 v{1 + i}, {v:#x}" for i, v in enumerate(vals)]
            lines = [
                "v_cmpx_eq_u32_e32 0, v0",
                f"s_mov_b32 s4, {BASE:#x}",
                f"s_mov_b32 s5, {w1:#x}",
                f"s_mov_b32 s6, {nr:#x}",
                f"s_mov_b32 s7, {w3:#x}",
                *setup,
                "buffer_load_format_xyzw v[4:7], " + vaddr + f", s[4:7], 0 {mods}",
                "s_waitcnt vmcnt(0)",
                "buffer_store_dwordx4 v[4:7], off, s[0:3], 0",
                "s_endpgm",
            ]
            with self.subTest(tag=tag):
                mem, _ = self._run_cs(lines, name=f"cs_{tag}")
                self.assertEqual(mem.reads, [])
                self.assertEqual([dict.get(mem, 4 * i, 0) for i in range(4)], want)

    def test_mtbuf_oob_and_invalid_descriptor_ignore_dst_sel_one(self):
        """MTBUF ignores V# DST_SEL=1 on both OOB and FORMAT=0 loads, returning 0 in addressed channels."""
        for tag, fmt_code, off, d16, want in (
            ("mtbuf_oob_u32", 20, 0x100, False, [0, 0, 0, 0]),
            ("mtbuf_oob_f32_d16_xyz", 22, 0x100, True, [0, SEED1 & 0xFFFF0000]),
            ("mtbuf_fmt0_f32", 0, 0, False, [0, 0, 0, 0]),
            ("mtbuf_fmt0_u8_d16_xyz", 0, 0, True, [0, SEED1 & 0xFFFF0000]),
        ):
            op = "tbuffer_load_format_d16_xyz" if d16 else "tbuffer_load_format_xyzw"
            vdata = "v[4:5]" if d16 else "v[4:7]"
            store = "buffer_store_dwordx2 v[4:5]" if d16 else "buffer_store_dwordx4 v[4:7]"
            inst_fmt = "BUF_FMT_32_32_32_32_FLOAT" if "f32" in tag else "BUF_FMT_8_8_8_8_UINT"
            lines = [
                "v_cmpx_eq_u32_e32 0, v0",
                f"s_mov_b32 s4, {BASE:#x}",
                "s_mov_b32 s5, 0",
                "s_mov_b32 s6, 16",
                f"s_mov_b32 s7, {word3(fmt_code, DST_1_1_1_1, 3):#x}",
                f"v_mov_b32 v1, {off:#x}",
                f"v_mov_b32 v4, {SEED0:#x}",
                f"v_mov_b32 v5, {SEED1:#x}",
                f"v_mov_b32 v6, {SEED2:#x}",
                f"v_mov_b32 v7, {SEED3:#x}",
                f"{op} {vdata}, v1, s[4:7], 0 format:[{inst_fmt}] offen",
                "s_waitcnt vmcnt(0)",
                f"{store}, off, s[0:3], 0",
                "s_endpgm",
            ]
            with self.subTest(tag=tag):
                mem, _ = self._run_cs(lines, name=f"cs_{tag}")
                self.assertEqual(mem.reads, [])
                self.assertEqual([dict.get(mem, 4 * i, 0) for i in range(len(want))], want)

    def test_inactive_lanes_and_wave64_emulate(self):
        """Wave64 active lanes in both halves get DST_SEL=1 on OOB while masked-off lanes keep seeds; --wave64-emulate validates."""
        lines = [
            # Enable only lane 0 (low half) and lane 35 (high half).
            "v_cmp_eq_u32_e64 s[12:13], 0, v0",
            "v_cmp_eq_u32_e64 s[14:15], 35, v0",
            "s_or_b64 s[12:13], s[12:13], s[14:15]",
            "s_mov_b32 s4, 0x1000",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 16",
            f"s_mov_b32 s7, {word3(56, DST_X_0_1_1, 3):#x}",  # 8_8_8_8_UNORM
            f"s_mov_b32 s8, {BASE:#x}",
            "s_mov_b32 s9, 0",
            "s_mov_b32 s10, 16",
            f"s_mov_b32 s11, {word3(60, DST_1_Y_Z_W, 3):#x}",  # 8_8_8_8_UINT
            "v_mov_b32 v1, 0x100",
            f"v_mov_b32 v4, {SEED0:#x}",
            f"v_mov_b32 v5, {SEED1:#x}",
            f"v_mov_b32 v6, {SEED2:#x}",
            f"v_mov_b32 v7, {SEED3:#x}",
            f"v_mov_b32 v8, {SEED0:#x}",
            "s_and_saveexec_b64 s[16:17], s[12:13]",
            "buffer_load_format_xyzw v[4:7], v1, s[4:7], 0 offen",
            "buffer_load_format_d16_x v8, v1, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
            "s_mov_b64 exec, s[16:17]",
            # Store lane 0, lane 1 (inactive), and lane 35 (high half active) into output buffer.
            "v_lshlrev_b32 v2, 5, v0",
            "buffer_store_dwordx4 v[4:7], v2, s[0:3], 0 offen",
            "buffer_store_dword v8, v2, s[0:3], 0 offen offset:16",
            "s_endpgm",
        ]
        pcs = [0, 0, 0x1000, OUT_V] + [0] * 28
        mem, spv_w64 = self._run_cs(lines, push_constants=pcs, wave64=True, name="cs_w64_oob_dst_sel")
        self.assertEqual(mem.reads, [])
        for active_lane in (0, 35):
            base_off = active_lane * 32
            self.assertEqual(
                [dict.get(mem, base_off + 4 * i, 0) for i in range(5)],
                [0, 0, 0x3F800000, 0x3F800000, (SEED0 & 0xFFFF0000) | 1],
                active_lane,
            )
        for inactive_lane in (1, 34):
            base_off = inactive_lane * 32
            self.assertEqual(
                [dict.get(mem, base_off + 4 * i, 0) for i in range(5)],
                [SEED0, SEED1, SEED2, SEED3, SEED0],
                inactive_lane,
            )

        old_emu = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            spv_emu = support.translate_lines("cs_w64_emu_oob_dst_sel", lines, fssm=CS_12_USER_FSSM)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        self.assertIn("%x_buf", spv_emu)
        self.assertNotIn("OpGroupNonUniform", spv_emu)

        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if spirv_as and spirv_val:
            with tempfile.TemporaryDirectory() as td:
                for idx, mod_text in enumerate((spv_w64, spv_emu)):
                    spv_bin = os.path.join(td, f"mod_{idx}.spv")
                    subprocess.run(
                        [spirv_as, "--target-env", "vulkan1.2", "-", "-o", spv_bin],
                        input=mod_text.encode("utf-8"),
                        check=True,
                    )
                    subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_bin], check=True)


if __name__ == "__main__":
    unittest.main()
