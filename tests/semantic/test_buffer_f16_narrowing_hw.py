"""f32 -> f16 narrowing in typed buffer stores to f16 formats and in d16 loads of 32-bit float formats (SHADER-358).

Every expected word was read back from an AMD Radeon RDNA2 iGPU (gfx1036, Ryzen 9 9950X3D, Windows AMD driver
32.0.21045.16010) through tools/hw_oracle with AMD clang 23 (rocm-sdk-core 10.0.0), wave32 and wave64 alike, with
the same instruction and a V# of the same format (the runtime's selectors: X001, XY01). The hardware rounds toward
zero and gives +-65504 for a finite value past the f16 range (0x3f803000 stores 0x3c01, not 0x3c02; 65520 stores
0x7bff, not infinity); infinities stay infinite, f32 denormals give a signed zero, and a NaN keeps its sign and its
top payload bits. UNORM/SNORM d16 loads are different (nearest even, tests/semantic/test_buffer_load_format.py).

MEASURED rows: (f32 word, buffer_store_format_x into 16_FLOAT, buffer_store_format_xy into 16_16_FLOAT of the
same value twice, buffer_load_format_d16_x from 32_FLOAT, buffer_load_format_d16_xy from 32_32_FLOAT of the same
value twice), the store words over a 0xdead0000 fill, the load words over a 0xdeadbeef destination.
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
OUT = 0x8000
SEL_R, SEL_RG = 4 | 0 << 3 | 0 << 6 | 1 << 9, 4 | 5 << 3 | 0 << 6 | 1 << 9

MEASURED = [
    (0x3F801000, 0xDEAD3C00, 0x3C003C00, 0xDEAD3C00, 0x3C003C00),
    (0x3F803000, 0xDEAD3C01, 0x3C013C01, 0xDEAD3C01, 0x3C013C01),
    (0x3F801800, 0xDEAD3C00, 0x3C003C00, 0xDEAD3C00, 0x3C003C00),
    (0x3F800800, 0xDEAD3C00, 0x3C003C00, 0xDEAD3C00, 0x3C003C00),
    (0xBF801000, 0xDEADBC00, 0xBC00BC00, 0xDEADBC00, 0xBC00BC00),
    (0xBF803000, 0xDEADBC01, 0xBC01BC01, 0xDEADBC01, 0xBC01BC01),
    (0xBF801800, 0xDEADBC00, 0xBC00BC00, 0xDEADBC00, 0xBC00BC00),
    (0x477FE000, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0x477FEFFF, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0x477FF000, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0x477FF001, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0x47800000, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0x501502F9, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0xD01502F9, 0xDEADFBFF, 0xFBFFFBFF, 0xDEADFBFF, 0xFBFFFBFF),
    (0x7F800000, 0xDEAD7C00, 0x7C007C00, 0xDEAD7C00, 0x7C007C00),
    (0xFF800000, 0xDEADFC00, 0xFC00FC00, 0xDEADFC00, 0xFC00FC00),
    (0x7FC00000, 0xDEAD7E00, 0x7E007E00, 0xDEAD7E00, 0x7E007E00),
    (0x7FA00000, 0xDEAD7D00, 0x7D007D00, 0xDEAD7D00, 0x7D007D00),
    (0xFFC12345, 0xDEADFE09, 0xFE09FE09, 0xDEADFE09, 0xFE09FE09),
    (0x33C00000, 0xDEAD0001, 0x00010001, 0xDEAD0001, 0x00010001),
    (0x33A00000, 0xDEAD0001, 0x00010001, 0xDEAD0001, 0x00010001),
    (0x33000000, 0xDEAD0000, 0x00000000, 0xDEAD0000, 0x00000000),
    (0x33000001, 0xDEAD0000, 0x00000000, 0xDEAD0000, 0x00000000),
    (0x38800000, 0xDEAD0400, 0x04000400, 0xDEAD0400, 0x04000400),
    (0x387FC000, 0xDEAD03FF, 0x03FF03FF, 0xDEAD03FF, 0x03FF03FF),
    (0x387FE000, 0xDEAD03FF, 0x03FF03FF, 0xDEAD03FF, 0x03FF03FF),
    (0x00000001, 0xDEAD0000, 0x00000000, 0xDEAD0000, 0x00000000),
    (0x80000000, 0xDEAD8000, 0x80008000, 0xDEAD8000, 0x80008000),
    (0x3EAAAAAB, 0xDEAD3555, 0x35553555, 0xDEAD3555, 0x35553555),
    (0x40490FDB, 0xDEAD4248, 0x42484248, 0xDEAD4248, 0x42484248),
    (0xC0490FDB, 0xDEADC248, 0xC248C248, 0xDEADC248, 0xC248C248),
    (0x3DCCCCCD, 0xDEAD2E66, 0x2E662E66, 0xDEAD2E66, 0x2E662E66),
    (0x477FE001, 0xDEAD7BFF, 0x7BFF7BFF, 0xDEAD7BFF, 0x7BFF7BFF),
    (0xC77FF000, 0xDEADFBFF, 0xFBFFFBFF, 0xDEADFBFF, 0xFBFFFBFF),
    (0x337FFFFF, 0xDEAD0000, 0x00000000, 0xDEAD0000, 0x00000000),
    (0x32800000, 0xDEAD0000, 0x00000000, 0xDEAD0000, 0x00000000),
    (0x7F800001, 0xDEAD7C01, 0x7C017C01, 0xDEAD7C01, 0x7C017C01),
    (0x7F801FFF, 0xDEAD7C01, 0x7C017C01, 0xDEAD7C01, 0x7C017C01),
    (0x7F802000, 0xDEAD7C01, 0x7C017C01, 0xDEAD7C01, 0x7C017C01),
    (0xFF800001, 0xDEADFC01, 0xFC01FC01, 0xDEADFC01, 0xFC01FC01),
    (0x7FBFFFFF, 0xDEAD7DFF, 0x7DFF7DFF, 0xDEAD7DFF, 0x7DFF7DFF),
    (0x7FC00001, 0xDEAD7E00, 0x7E007E00, 0xDEAD7E00, 0x7E007E00),
    (0xFFFFFFFF, 0xDEADFFFF, 0xFFFFFFFF, 0xDEADFFFF, 0xFFFFFFFF),
    (0x7FE00000, 0xDEAD7F00, 0x7F007F00, 0xDEAD7F00, 0x7F007F00),
    (0x7F900000, 0xDEAD7C80, 0x7C807C80, 0xDEAD7C80, 0x7C807C80),
]


def run(lines, fmt, sel, memory):
    """Translate a pixel shader: s[0:3] a raw V# at BASE with this format, s[4:7] a raw 32-bit V# at OUT."""
    pre = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x1000",
           f"s_mov_b32 s3, {fmt << 12 | sel | 1 << 24 | 3 << 28:#x}", f"s_mov_b32 s4, {OUT:#x}", "s_mov_b32 s5, 0",
           "s_mov_b32 s6, 0x1000", "s_mov_b32 s7, 0x31016fac"]
    post = ["exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_f16_narrowing.s")
        with open(os.path.join(tmp, "ps_f16_narrowing.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8") as f:
            f.write("; test_buffer_f16_narrowing_hw\n" + "".join(f"\t{x}\n" for x in pre + lines + post))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    si.run_fragment(spv, memory=memory)


def store(fmt, sel, n):
    """Store every row's value (n components of it) at BASE + 8k; return the first dword of each element."""
    memory = {BASE + 4 * i: 0xDEAD0000 for i in range(2 * len(MEASURED))}
    lines = []
    for k, row in enumerate(MEASURED):
        regs = "v2" if n == 1 else "v[2:3]"
        lines += [f"v_mov_b32 v2, {row[0]:#x}", f"v_mov_b32 v3, {row[0]:#x}", f"v_mov_b32 v1, {8 * k}",
                  f"buffer_store_format_{'xy'[:n]} {regs}, v1, s[0:3], 0 offen"]
    run(lines, fmt, sel, memory)
    return [memory[BASE + 8 * k] for k in range(len(MEASURED))]


def load(fmt, sel, n):
    """d16 load of every row's value (n copies at BASE + 8k); return the destination register of each."""
    memory = {}
    for k, row in enumerate(MEASURED):
        memory[BASE + 8 * k] = memory[BASE + 8 * k + 4] = row[0]
    lines = []
    for k in range(len(MEASURED)):
        lines += [f"v_mov_b32 v1, {8 * k}", "v_mov_b32 v2, 0xdeadbeef",
                  f"buffer_load_format_d16_{'xy'[:n]} v2, v1, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
                  f"buffer_store_dword v2, off, s[4:7], 0 offset:{4 * k}"]
    run(lines, fmt, sel, memory)
    return [memory[OUT + 4 * k] for k in range(len(MEASURED))]


class MeasuredNarrowing(unittest.TestCase):
    def check(self, got, column, what):
        for row, word in zip(MEASURED, got):
            with self.subTest(what=what, value=hex(row[0])):
                self.assertEqual(word, row[column])

    def test_store_16_float_x(self):
        self.check(store(13, SEL_R, 1), 1, "store 16_FLOAT x")

    def test_store_16_16_float_xy(self):
        self.check(store(29, SEL_RG, 2), 2, "store 16_16_FLOAT xy")

    def test_d16_load_32_float_x(self):
        self.check(load(22, SEL_R, 1), 3, "d16_x from 32_FLOAT")

    def test_d16_load_32_32_float_xy(self):
        self.check(load(64, SEL_RG, 2), 4, "d16_xy from 32_32_FLOAT")


if __name__ == "__main__":
    unittest.main()
