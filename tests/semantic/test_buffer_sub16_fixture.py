"""cs_w32_buffer_sub16: the GPU test of the 8/16-bit MUBUF d16 loads and stores (#1864), run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_buffer_sub16.asm (derivations in its header) stores eight words from lane 0. This runs its
fixture as one wave32 group through spirv_mini_interp with vk_compute_test's buffer (every byte 0xEE, a V# with
num_records 0xFFFFFFFF), checks the words against a byte-level model of the d16 rules, and that check.py and
scripts/run_gpu_tests.py expect the same words.
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
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
KEEP, D40, D50, D60 = 0xBEEF1234, 0x8899AABB, 0x11223344, 0x55667788


def byte(w, i):
    return (w >> (8 * i)) & 0xFF


def lo16(v):
    return (KEEP & 0xFFFF0000) | v


def hi16(v):
    return (v << 16) | (KEEP & 0xFFFF)


def sext8(v):
    return v | 0xFF00 if v & 0x80 else v


UB, SH = byte(D40, 1), byte(D40, 2) | byte(D40, 3) << 8
EXPECTED = [lo16(UB), hi16(UB), lo16(sext8(UB)), hi16(sext8(UB)), lo16(SH), hi16(SH),
            (D50 & ~0xFF00) | ((0x00CD0000 >> 16) & 0xFF) << 8,            # store_byte_d16_hi: bits 23:16
            (D60 & 0xFFFF) | ((0xABCD0000 >> 16) & 0xFFFF) << 16]          # store_short_d16_hi: bits 31:16


class BufferSub16Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0xBEEF00AA, 0x00AA1234, 0xBEEFFFAA, 0xFFAA1234, 0xBEEF8899, 0x88991234,
                                    0x1122CD44, 0xABCD7788])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_buffer_sub16", fssm=FSSM)
        mem = {4 * k: 0xEEEEEEEE for k in range(64)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                m = re.search(r'"cs_w32_buffer_sub16"\)?[,:]\s*\[([^\]]*)\]', f.read())
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
