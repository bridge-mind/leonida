"""cs_w32_fcmpswap: the GPU test of ds_cmpst_f32 and buffer_atomic_fcmpswap, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_fcmpswap.asm (derivations in its header) stores eight words from lane 0. This runs its
fixture as one wave32 group through spirv_mini_interp with the buffer vk_compute_test gives it (every byte 0xEE, a V#
with num_records 0xFFFFFFFF), checks the words against a float-compare reference computed here, that check.py and
scripts/run_gpu_tests.py expect the same words, and that EXEC is cut to lane 0 before the first memory op.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)      # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE
PZ, NZ, ONE, TWO, THREE, NAN = 0, 0x80000000, 0x3F800000, 0x40000000, 0x40400000, 0x7FC00000


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def cmpst(m, cmp, data):
    """(old, new) of a float compare-and-store: -0 == +0, NaN matches nothing."""
    return m, (data if f32(m) == f32(cmp) else m)


EXPECTED = [
    cmpst(PZ, NZ, ONE)[0], cmpst(PZ, NZ, ONE)[1],           # ds_cmpst_rtn_f32: old, then the LDS word
    cmpst(NAN, NAN, ONE)[1],                                # NaN against its own bits: unchanged
    cmpst(ONE, TWO, THREE)[1],                              # ds_cmpst_f32, no match
    cmpst(NZ, PZ, THREE)[0], cmpst(NZ, PZ, THREE)[1],       # buffer_atomic_fcmpswap glc: old, then memory
    cmpst(ONE, 0x3F800001, TWO)[1],                         # one ulp off: no match
    cmpst(NAN, NAN, ONE)[0],                                # glc old value of the NaN case
]


class FcmpswapFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0, 0x3F800000, 0x7FC00000, 0x3F800000, 0x80000000, 0x40400000, 0x3F800000,
                                    0x7FC00000])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_fcmpswap", fssm=FSSM)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=VSHARP + [0] * 28, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_fcmpswap"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)

    def test_memory_ops_run_in_lane_0_only(self):
        with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_fcmpswap.asm"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(";")]
        self.assertEqual(lines[0], "v_cmpx_eq_u32_e32 0, v0")


if __name__ == "__main__":
    unittest.main()
