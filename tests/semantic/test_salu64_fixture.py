"""cs_w32_salu64: the GPU test of the 64-bit scalar data ops and v_mad_u64_u32, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_salu64.asm has lane 0 of a wave32 group store one word of each of s_brev_b64,
s_bfe_u64, s_bfe_i64, s_bitset1_b64, s_flbit_i32_b64, s_flbit_i32_i64, s_cmov_b32 of s_sext_i32_i16 and
v_mad_u64_u32 (derivations in its header). These ops were checked on the CPU only. This runs the fixture
(tests/semantic/regen_fixtures.py) through spirv_mini_interp, checks the eight words against values computed here
from the ISA definitions, and checks that check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def brev64(x: int) -> int:
    return int(f"{x & M64:064b}"[::-1], 2)


def bfe64(x: int, s1: int, signed: bool) -> int:
    off, width = s1 & 63, (s1 >> 16) & 127
    v = (x >> off) & ((1 << width) - 1)
    if signed and width and v >> (width - 1):
        v -= 1 << width
    return v & M64


def flbit_b64(x: int) -> int:
    return 64 - x.bit_length() if x else M32


def flbit_i64(x: int) -> int:
    sign = x >> 63
    for i in range(62, -1, -1):
        if (x >> i) & 1 != sign:
            return 63 - i
    return M32


def sext16(x: int) -> int:
    x &= 0xFFFF
    return (x - 0x10000 if x >> 15 else x) & M32


EXPECTED = [
    brev64(0x12345678_0000000F) & M32,                                  # s_brev_b64, low word
    bfe64(0x00000123_45678000, 0x10001C, False) & M32,                  # s_bfe_u64, low word
    bfe64(0x00000001_80000000, 0x2001F, True) >> 32,                    # s_bfe_i64, high word
    (1 << 40) >> 32,                                                    # s_bitset1_b64, high word
    flbit_b64(0x00000000_00010000),                                     # s_flbit_i32_b64
    flbit_i64(0xFFFFFFFF_FFFF0000),                                     # s_flbit_i32_i64
    sext16(0x00018000),                                                 # s_cmov_b32 (SCC = 1) of s_sext_i32_i16
    ((M32 * M32 + M64) & M64) >> 32,                                    # v_mad_u64_u32, high word
]


class Salu64Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x1E6A2C48, 0x1234, 0xFFFFFFFF, 0x100, 47, 48, 0xFFFF8000, 0xFFFFFFFE])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_salu64", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_salu64"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
