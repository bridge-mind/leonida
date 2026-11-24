"""cs_w32_new_ops: the GPU test of ops merged on 2026-11-22 with CPU-only checks, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_new_ops.asm has lane 0 of a wave32 group store one word of each of s_quadmask_b32,
s_quadmask_b64, s_bitreplicate_b64_b32, v_cvt_pknorm_i16_f32, v_cvt_pknorm_u16_f32, ds_wrxchg2_rtn_b32 (old word and
the LDS word after it) and a VGPR kept across ds_nop / v_clrexcp (derivations in its header). This runs the fixture
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

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS (LDS_SIZE 1), 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def quadmask(x: int, quads: int) -> int:
    return sum(1 << i for i in range(quads) if (x >> (4 * i)) & 0xF)


def bitreplicate(x: int) -> int:
    return sum(3 << (2 * i) for i in range(32) if (x >> i) & 1)


def norm16(x: float, snorm: bool) -> int:
    x = min(max(x, -1.0 if snorm else 0.0), 1.0)
    return round(x * (32767 if snorm else 65535)) & 0xFFFF      # round() is round-half-even


EXPECTED = [
    quadmask(0x01020408, 8),                                            # s_quadmask_b32
    quadmask(0x80000000_00000001, 16) & M32,                            # s_quadmask_b64, low word
    bitreplicate(0x80000001) >> 32,                                     # s_bitreplicate_b64_b32, high word
    norm16(0.5, True) | norm16(-1.0, True) << 16,                       # v_cvt_pknorm_i16_f32
    norm16(1.0, False) | norm16(0.25, False) << 16,                     # v_cvt_pknorm_u16_f32
    0x77,                                                               # ds_wrxchg2_rtn_b32: old word at offset0
    0x99,                                                               # LDS at offset1 after: DATA2
    0x1234,                                                             # kept across ds_nop / v_clrexcp
]


class NewOpsFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x55, 0x8001, 0xC0000000, 0x80014000, 0x4000FFFF, 0x77, 0x99, 0x1234])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_new_ops", fssm=FSSM)
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
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_new_ops.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertLess(lines.index("v_cmpx_eq_u32_e32 0, v0"),
                        min(i for i, ln in enumerate(lines) if ln.startswith("ds_")))

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_new_ops"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
