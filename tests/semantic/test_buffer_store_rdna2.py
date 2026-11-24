"""RDNA2 (gfx1036) buffer format store semantics (SHADER-364):
1. UNORM/SNORM format stores round the exact product clamp(x) * scale once (ties away from zero),
   matching native gfx1036 (wave32 and wave64, MUBUF and MTBUF): e.g. UNORM16(0.3) == 19661 (not 19660).
2. 2-component MUBUF buffer_store_format_* stores infer COMP_SWAP from V# DST_SEL_X/Y/Z:
   XY01 -> (x, y), YX01 -> (y, x), XYZW -> (x, w) (so _xy writes (x, 0)), YXZW/ZYXW -> (w, x);
   MTBUF tbuffer_store_format_* ignores DST_SEL and always stores (x, y).
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import buffer_store_probe as probe  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

BASE = 0x1000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def _translate(lines):
    with tempfile.TemporaryDirectory() as td:
        dxbc_dir = os.path.join(td, "dxbc")
        dis_dir = os.path.join(dxbc_dir, "disasm")
        os.makedirs(dis_dir)
        with open(os.path.join(dxbc_dir, "ps_test.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        src = os.path.join(dis_dir, "ps_test.s")
        with open(src, "w", encoding="ascii") as f:
            f.write("; ps_test\n" + "\n".join("\t" + ln for ln in lines) + "\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            asm_p = os.path.join(td, "s.spvasm")
            bin_p = os.path.join(td, "s.spv")
            with open(asm_p, "w", encoding="utf-8") as f:
                f.write(spv)
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm_p, "-o", bin_p], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", bin_p], check=True)
    return spv


def _run_store(inst, code, sel, vals, fixed_fmt=False, d16=False):
    n_comp = len(inst.rsplit("_", 1)[1])
    n_regs = (n_comp + 1) // 2 if d16 else n_comp
    vreg = "v4" if n_regs == 1 else f"v[4:{3 + n_regs}]"
    d3 = probe.vsharp_d3(code, sel)
    lines = [
        f"s_mov_b32 s0, {BASE:#x}",
        "s_mov_b32 s1, 0",
        "s_mov_b32 s2, 0x100",
        f"s_mov_b32 s3, {d3:#x}",
        "v_mov_b32 v1, 0",
    ]
    for i in range(n_regs):
        lines.append(f"v_mov_b32 v{4 + i}, {vals[i]:#x}")
    if fixed_fmt:
        fmt_name = rdna2spirv.GFX10_FORMATS[code].replace("GFX10_FORMAT_", "BUF_FMT_")
        lines.append(f"{inst} {vreg}, v1, s[0:3], 0 format:[{fmt_name}] offen")
    else:
        lines.append(f"{inst} {vreg}, v1, s[0:3], 0 offen")
    lines += ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    spv = _translate(lines)
    mem = {BASE: 0xDEAD0000, BASE + 4: 0xDEAD0000}
    si.run_fragment(spv, memory=mem)
    return mem[BASE], mem[BASE + 4]


class NormStoreRoundingTests(unittest.TestCase):
    def test_unorm16_0_3_single_rounds_to_19661(self):
        u03 = probe.f32_to_u32(0.3)
        for inst, fixed in (("buffer_store_format_x", False), ("tbuffer_store_format_x", True)):
            w0, _ = _run_store(inst, 7, probe.SEL_X001, (u03,), fixed_fmt=fixed)
            self.assertEqual(w0 & 0xFFFF, 19661, inst)

    def test_double_rounding_witnesses_across_norm_formats(self):
        # One positive and one negative (where signed) double-rounding witness per format, plus specials.
        cases = [
            (7, "unorm", 16, [probe.f32_to_u32(0.3), 0x37C000C0, 0x00000000, 0x80000000,
                              0x00000001, 0x3F000000, 0x3F800000, 0x3FC00000, 0x7F800000, 0xFF800000, 0x7FC00000]),
            (8, "snorm", 16, [0x38400180, 0xB8400180, 0x00000000, 0x80000000,
                              0x3F000000, 0xBF000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0xFFC00000]),
            (1, "unorm", 8, [0x3C20A0A1, 0x3C909091, 0x3F000000, 0xBF800000, 0x7FC00000]),
            (2, "snorm", 8, [0x3C418306, 0xBC418306, 0x3F000000, 0xBF000000, 0xFF800000]),
            (50, "unorm", 10, [0x3AC0300C, 0x3B60380E, 0x3F000000, 0x3F800000]),
        ]
        for code, kind, bits, inputs in cases:
            mask = (1 << bits) - 1
            sel = probe.SEL_XYZW if bits == 10 else probe.SEL_X001
            for u in inputs:
                want = probe.model_norm(u, kind, bits)
                w0, _ = _run_store("buffer_store_format_x", code, sel, (u,))
                self.assertEqual(w0 & mask, want, f"code={code} u={u:#010x}")


class TwoCompDstSelTests(unittest.TestCase):
    def test_mubuf_2comp_comp_swap_modes(self):
        vals = (0x11111111, 0x22222222, 0x33333333, 0x44444444)
        # XY01 -> (x, y) for both _xy and _xyzw
        self.assertEqual(_run_store("buffer_store_format_xy", 62, (4, 5, 0, 1), vals), (0x11111111, 0x22222222))
        self.assertEqual(_run_store("buffer_store_format_xyzw", 62, (4, 5, 0, 1), vals), (0x11111111, 0x22222222))
        # YX01 -> (y, x)
        self.assertEqual(_run_store("buffer_store_format_xyzw", 62, (5, 4, 0, 1), vals), (0x22222222, 0x11111111))
        # XYZW -> (x, w) for _xyzw and (x, 0) for _xy
        self.assertEqual(_run_store("buffer_store_format_xyzw", 62, (4, 5, 6, 7), vals), (0x11111111, 0x44444444))
        self.assertEqual(_run_store("buffer_store_format_xy", 62, (4, 5, 6, 7), vals), (0x11111111, 0x00000000))
        # YXZW -> (w, x)
        self.assertEqual(_run_store("buffer_store_format_xyzw", 62, (5, 4, 6, 7), vals), (0x44444444, 0x11111111))

    def test_mubuf_2comp_packed_and_norm_formats(self):
        # 8_8_UINT (18): XY01 -> 0xdead2211, XYZW xyzw -> 0xdead4411, XYZW xy -> 0xdead0011
        w0, _ = _run_store("buffer_store_format_xyzw", 18, (4, 5, 0, 1), (0x11, 0x22, 0x33, 0x44))
        self.assertEqual(w0, 0xDEAD2211)
        w0, _ = _run_store("buffer_store_format_xyzw", 18, (4, 5, 6, 7), (0x11, 0x22, 0x33, 0x44))
        self.assertEqual(w0, 0xDEAD4411)
        w0, _ = _run_store("buffer_store_format_xy", 18, (4, 5, 6, 7), (0x11, 0x22, 0x33, 0x44))
        self.assertEqual(w0, 0xDEAD0011)
        # 16_16_UNORM (23): (0.25, 0.5, 0.75, 1.0)
        flt = tuple(probe.f32_to_u32(x) for x in (0.25, 0.5, 0.75, 1.0))
        w0, _ = _run_store("buffer_store_format_xyzw", 23, (4, 5, 0, 1), flt)
        self.assertEqual(w0, 0x80004000)
        w0, _ = _run_store("buffer_store_format_xyzw", 23, (4, 5, 6, 7), flt)
        self.assertEqual(w0, 0xFFFF4000)
        # d16_xyzw and d16_xy on 16_16_UINT (27)
        d16_vals = (0x22221111, 0x44443333)
        w0, _ = _run_store("buffer_store_format_d16_xyzw", 27, (4, 5, 0, 1), d16_vals, d16=True)
        self.assertEqual(w0, 0x22221111)
        w0, _ = _run_store("buffer_store_format_d16_xyzw", 27, (4, 5, 6, 7), d16_vals, d16=True)
        self.assertEqual(w0, 0x44441111)
        w0, _ = _run_store("buffer_store_format_d16_xy", 27, (4, 5, 6, 7), d16_vals, d16=True)
        self.assertEqual(w0, 0x00001111)

    def test_mtbuf_2comp_ignores_dst_sel(self):
        vals = (0x11111111, 0x22222222, 0x33333333, 0x44444444)
        self.assertEqual(_run_store("tbuffer_store_format_xyzw", 62, (4, 5, 6, 7), vals, fixed_fmt=True),
                         (0x11111111, 0x22222222))
        self.assertEqual(_run_store("tbuffer_store_format_xyzw", 62, (5, 4, 6, 7), vals, fixed_fmt=True),
                         (0x11111111, 0x22222222))


if __name__ == "__main__":
    unittest.main()
