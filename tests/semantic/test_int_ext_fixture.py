"""cs_w32_int_ext: the GPU test of integer ops that until now had CPU-only checks, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_int_ext.asm has lane 0 of a wave32 group store one result of each of s_absdiff_i32,
s_lshl2_add_u32, the SCC of s_lshl1_add_u32, s_pack_lh_b32_b16, s_mul_hi_i32, v_mul_hi_i32_i24, v_sad_u8 and v_lerp_u8
(derivations in its header). This runs its fixture (tests/semantic/regen_fixtures.py) as one invocation through
spirv_mini_interp and checks the eight words against values computed here from the ISA definitions, and that
check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
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

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def i24(x: int) -> int:
    x &= 0xFFFFFF
    return x - (1 << 24) if x >> 23 else x


def lerp_u8(a: int, b: int, c: int) -> int:
    return sum((((a >> s) & 0xFF) + ((b >> s) & 0xFF) + ((c >> s) & 1)) >> 1 << s for s in (0, 8, 16, 24))


EXPECTED = [
    abs(s32(0x7FFFFFFF - 0x80000000)) & M32,                                   # s_absdiff_i32: wraps to -1 first
    ((5 << 2) + 7) & M32,                                                     # s_lshl2_add_u32
    int((0x80000000 << 1) + 0 >= 1 << 32),                                    # s_lshl1_add_u32 SCC
    (0x33334444 & 0xFFFF0000) | (0x11112222 & 0xFFFF),                        # s_pack_lh_b32_b16
    ((s32(0x80000000) * s32(0x7FFFFFFF)) >> 32) & M32,                        # s_mul_hi_i32
    ((i24(0x800000) * i24(0x800000)) >> 32) & M32,                            # v_mul_hi_i32_i24
    10 + sum(abs(((0x01020304 >> s) & 0xFF) - ((0x04030201 >> s) & 0xFF)) for s in (0, 8, 16, 24)),   # v_sad_u8
    lerp_u8(0x10203040, 0x30405060, 0x01010101),                              # v_lerp_u8
]


class IntExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [1, 27, 1, 0x33332222, 0xC0000000, 0x4000, 18, 0x20304050])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_int_ext", fssm=FSSM)
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
            m = re.search(r'"cs_w32_int_ext"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
