"""v_floor_f32, v_ceil_f32, v_fract_f32 and v_cvt_flr/rpi_i32_f32 against words an RDNA2 GPU returned for the same instruction.

Every row ran natively on an AMD Radeon RDNA2 iGPU (gfx1036, Ryzen 9 9950X3D, Windows AMD driver 32.0.21045.16010)
through tools/hw_oracle with AMD clang 23 (rocm-sdk-core 10.0.0), in the MODE inferred for the game (IEEE 0,
DX10_CLAMP 1, f32 denormals flushed, f16 denormals kept): v1 holds the operand, v4 starts as 0xDEADBEEF and the word
after the instruction is the expected one. No model in this test.

What the rows pin: a denormal operand is a zero of its sign (floor(-1e-45) = -0, ceil(+1e-45) = +0, fract of either
0), which a translation that uses the denormal as it is gets wrong by a whole 1.0; a NaN operand passes through
unchanged, a signalling one too (IEEE 0); fract(+-inf) is the negative quiet NaN 0xffc00000; and the ordinary values
around them (fract of a tiny negative normal is 0x3f7fffff, docs/design.md). v_cvt_flr_i32_f32 of a
negative denormal is 0 for the same reason (floor of the kept denormal gives -1); v_cvt_rpi_i32_f32 rows
are the control.
"""
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
BASE = 0x1000

# (instruction, v1, the word gfx1036 left in v4)
MEASURED = [
    ('v_fract_f32 v4, v1', 0x00000001, 0x00000000),
    ('v_fract_f32 v4, v1', 0x80000001, 0x00000000),
    ('v_fract_f32 v4, v1', 0x00400000, 0x00000000),
    ('v_fract_f32 v4, v1', 0x807FFFFF, 0x00000000),
    ('v_fract_f32 v4, v1', 0x007FFFFF, 0x00000000),
    ('v_fract_f32 v4, v1', 0x80000000, 0x00000000),
    ('v_fract_f32 v4, v1', 0x00000000, 0x00000000),
    ('v_fract_f32 v4, v1', 0x00800000, 0x00800000),
    ('v_fract_f32 v4, v1', 0x80800000, 0x3F7FFFFF),
    ('v_fract_f32 v4, v1', 0x7F800000, 0xFFC00000),
    ('v_fract_f32 v4, v1', 0xFF800000, 0xFFC00000),
    ('v_fract_f32 v4, v1', 0x7FA00000, 0x7FA00000),
    ('v_fract_f32 v4, v1', 0xFFC00001, 0xFFC00001),
    ('v_fract_f32 v4, v1', 0x7FC00000, 0x7FC00000),
    ('v_fract_f32 v4, v1', 0xBF000000, 0x3F000000),
    ('v_fract_f32 v4, v1', 0x3FC00000, 0x3F000000),
    ('v_fract_f32 v4, v1', 0xB3800000, 0x3F7FFFFF),
    ('v_fract_f32 v4, v1', 0xC0490FDB, 0x3F5BC094),
    ('v_fract_f32 v4, v1', 0x4B800001, 0x00000000),
    ('v_fract_f32 v4, v1', 0xCB000001, 0x00000000),
    ('v_fract_f32 v4, v1', 0x3F7FFFFF, 0x3F7FFFFF),
    ('v_fract_f32 v4, v1', 0xBF7FFFFF, 0x33800000),
    ('v_floor_f32 v4, v1', 0x00000001, 0x00000000),
    ('v_floor_f32 v4, v1', 0x80000001, 0x80000000),
    ('v_floor_f32 v4, v1', 0x00400000, 0x00000000),
    ('v_floor_f32 v4, v1', 0x807FFFFF, 0x80000000),
    ('v_floor_f32 v4, v1', 0x007FFFFF, 0x00000000),
    ('v_floor_f32 v4, v1', 0x80000000, 0x80000000),
    ('v_floor_f32 v4, v1', 0x00000000, 0x00000000),
    ('v_floor_f32 v4, v1', 0x00800000, 0x00000000),
    ('v_floor_f32 v4, v1', 0x80800000, 0xBF800000),
    ('v_floor_f32 v4, v1', 0x7F800000, 0x7F800000),
    ('v_floor_f32 v4, v1', 0xFF800000, 0xFF800000),
    ('v_floor_f32 v4, v1', 0x7FA00000, 0x7FA00000),
    ('v_floor_f32 v4, v1', 0xFFC00001, 0xFFC00001),
    ('v_floor_f32 v4, v1', 0x7FC00000, 0x7FC00000),
    ('v_floor_f32 v4, v1', 0xBF000000, 0xBF800000),
    ('v_floor_f32 v4, v1', 0x3FC00000, 0x3F800000),
    ('v_floor_f32 v4, v1', 0xB3800000, 0xBF800000),
    ('v_floor_f32 v4, v1', 0xC0490FDB, 0xC0800000),
    ('v_floor_f32 v4, v1', 0x4B800001, 0x4B800001),
    ('v_floor_f32 v4, v1', 0xCB000001, 0xCB000001),
    ('v_floor_f32 v4, v1', 0x3F7FFFFF, 0x00000000),
    ('v_floor_f32 v4, v1', 0xBF7FFFFF, 0xBF800000),
    ('v_ceil_f32 v4, v1', 0x00000001, 0x00000000),
    ('v_ceil_f32 v4, v1', 0x80000001, 0x80000000),
    ('v_ceil_f32 v4, v1', 0x00400000, 0x00000000),
    ('v_ceil_f32 v4, v1', 0x807FFFFF, 0x80000000),
    ('v_ceil_f32 v4, v1', 0x007FFFFF, 0x00000000),
    ('v_ceil_f32 v4, v1', 0x80000000, 0x80000000),
    ('v_ceil_f32 v4, v1', 0x00000000, 0x00000000),
    ('v_ceil_f32 v4, v1', 0x00800000, 0x3F800000),
    ('v_ceil_f32 v4, v1', 0x80800000, 0x80000000),
    ('v_ceil_f32 v4, v1', 0x7F800000, 0x7F800000),
    ('v_ceil_f32 v4, v1', 0xFF800000, 0xFF800000),
    ('v_ceil_f32 v4, v1', 0x7FA00000, 0x7FA00000),
    ('v_ceil_f32 v4, v1', 0xFFC00001, 0xFFC00001),
    ('v_ceil_f32 v4, v1', 0x7FC00000, 0x7FC00000),
    ('v_ceil_f32 v4, v1', 0xBF000000, 0x80000000),
    ('v_ceil_f32 v4, v1', 0x3FC00000, 0x40000000),
    ('v_ceil_f32 v4, v1', 0xB3800000, 0x80000000),
    ('v_ceil_f32 v4, v1', 0xC0490FDB, 0xC0400000),
    ('v_ceil_f32 v4, v1', 0x4B800001, 0x4B800001),
    ('v_ceil_f32 v4, v1', 0xCB000001, 0xCB000001),
    ('v_ceil_f32 v4, v1', 0x3F7FFFFF, 0x3F800000),
    ('v_ceil_f32 v4, v1', 0xBF7FFFFF, 0x80000000),
    ('v_fract_f32_e64 v4, |v1|', 0x80000001, 0x00000000),
    ('v_fract_f32_e64 v4, |v1|', 0x807FFFFF, 0x00000000),
    ('v_fract_f32_e64 v4, |v1|', 0xFF800000, 0xFFC00000),
    ('v_floor_f32_e64 v4, |v1|', 0x80000001, 0x00000000),
    ('v_floor_f32_e64 v4, |v1|', 0x807FFFFF, 0x00000000),
    ('v_floor_f32_e64 v4, |v1|', 0xFF800000, 0x7F800000),
    ('v_ceil_f32_e64 v4, |v1|', 0x80000001, 0x00000000),
    ('v_ceil_f32_e64 v4, |v1|', 0x807FFFFF, 0x00000000),
    ('v_ceil_f32_e64 v4, |v1|', 0xFF800000, 0x7F800000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x00000001, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x80000001, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x00400000, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x807FFFFF, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x007FFFFF, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x80000000, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x00000000, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x00800000, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x80800000, 0xFFFFFFFF),
    ('v_cvt_flr_i32_f32 v4, v1', 0xBF000000, 0xFFFFFFFF),
    ('v_cvt_flr_i32_f32 v4, v1', 0x3F000000, 0x00000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0xBF800001, 0xFFFFFFFE),
    ('v_cvt_flr_i32_f32 v4, v1', 0x3FC00000, 0x00000001),
    ('v_cvt_flr_i32_f32 v4, v1', 0xBFC00000, 0xFFFFFFFE),
    ('v_cvt_flr_i32_f32 v4, v1', 0x4EFFFFFF, 0x7FFFFF80),
    ('v_cvt_flr_i32_f32 v4, v1', 0xCF000000, 0x80000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0xCF000001, 0x80000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x7F800000, 0x7FFFFFFF),
    ('v_cvt_flr_i32_f32 v4, v1', 0xFF800000, 0x80000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0x7FC00000, 0x7FFFFFFF),
    ('v_cvt_flr_i32_f32 v4, v1', 0xFFC00000, 0x80000000),
    ('v_cvt_flr_i32_f32 v4, v1', 0xB3800000, 0xFFFFFFFF),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x00000001, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x80000001, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x00400000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x807FFFFF, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x007FFFFF, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x80000000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x00000000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x00800000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x80800000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xBF000000, 0x00000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x3F000000, 0x00000001),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xBF800001, 0xFFFFFFFF),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x3FC00000, 0x00000002),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xBFC00000, 0xFFFFFFFF),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x4EFFFFFF, 0x7FFFFF80),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xCF000000, 0x80000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xCF000001, 0x80000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x7F800000, 0x7FFFFFFF),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xFF800000, 0x80000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0x7FC00000, 0x7FFFFFFF),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xFFC00000, 0x80000000),
    ('v_cvt_rpi_i32_f32 v4, v1', 0xB3800000, 0x00000000),
]


def run_rows(rows):
    """Translate one pixel shader that runs every row and stores v4 at BASE + 4k; return the stored words."""
    body = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000", "s_mov_b32 s3, 0x31016fac"]
    for k, (line, x, _) in enumerate(rows):
        body += [f"v_mov_b32 v1, {x:#x}", "v_mov_b32 v4, 0xdeadbeef", line,
                 f"buffer_store_dword v4, off, s[0:3], 0 offset:{4 * k}"]
    body += ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_floor_ceil_fract.s")
        with open(os.path.join(tmp, "ps_floor_ceil_fract.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8") as f:
            f.write("; test_alu_floor_ceil_fract_hw\n" + "".join(f"\t{x}\n" for x in body))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    memory = {}
    si.run_fragment(spv, memory=memory)
    return [memory.get(BASE + 4 * k) for k in range(len(rows))]


class MeasuredWords(unittest.TestCase):
    def test_rows(self):
        for row, word in zip(MEASURED, run_rows(MEASURED)):
            with self.subTest(insn=row[0], v1=hex(row[1])):
                self.assertEqual(word, row[2])


if __name__ == "__main__":
    unittest.main()
