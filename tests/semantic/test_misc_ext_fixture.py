"""cs_w32_misc_ext: the GPU test of more ops that had CPU-only checks, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_misc_ext.asm has lane 0 of a wave32 group store one word of each of s_nand_b64,
s_xnor_b64, v_cvt_pk_u16_u32, v_cvt_pk_i16_i32, v_mad_u32_u16, v_mad_i32_i16, ds_cmpst_rtn_b32 (old value and the
LDS word after it) and ds_write2_b64 / ds_read2_b64 (derivations in its header). This runs the fixture
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

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS (LDS_SIZE 1), 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def s16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if x >> 15 else x


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sat_u16(x: int) -> int:
    return min(x & M32, 0xFFFF)


def sat_i16(x: int) -> int:
    return max(-32768, min(32767, s32(x))) & 0xFFFF


A, B_NAND, B_XNOR = 0xF0F0F0F0_12345678, 0xFF00FF00_FFFFFFFF, 0xFF00FF00_0F0F0F0F
LDS_INIT, CMP, NEW = 5, 5, 9
EXPECTED = [
    (~(A & B_NAND) & M64) >> 32,                                        # s_nand_b64, high word
    ~(A ^ B_XNOR) & M32,                                                # s_xnor_b64, low word
    sat_u16(0x12345) | sat_u16(7) << 16,                                # v_cvt_pk_u16_u32
    sat_i16(-40000) | sat_i16(40000) << 16,                             # v_cvt_pk_i16_i32
    (0xFFFF * 0xFFFF + 1) & M32,                                        # v_mad_u32_u16
    (s16(0x8000) * 2 + 5) & M32,                                        # v_mad_i32_i16
    LDS_INIT << 8 | (NEW if LDS_INIT == CMP else LDS_INIT),             # ds_cmpst_rtn_b32: old, then LDS
    0xCAFEF00D,                                                         # ds_read2_b64: element 2, high dword
]


class MiscExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x0FFF0FFF, 0xE2C4A688, 0x0007FFFF, 0x7FFF8000, 0xFFFE0002, 0xFFFF0005, 0x509,
                                    0xCAFEF00D])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_misc_ext", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_lds_ops_run_in_lane_0_only(self):
        # spirv_mini_interp gives each invocation its own LDS, so lanes racing on one address are invisible here;
        # the .asm must cut EXEC to lane 0 before its first ds_ op (#1850).
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_misc_ext.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertLess(lines.index("v_cmpx_eq_u32_e32 0, v0"),
                        min(i for i, ln in enumerate(lines) if ln.startswith("ds_")))

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_misc_ext"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
