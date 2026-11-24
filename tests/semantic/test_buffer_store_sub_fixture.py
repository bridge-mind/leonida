"""cs_w32_buffer_store_sub: the GPU test of the plain 8/16-bit MUBUF stores (#1864), run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_buffer_store_sub.asm (derivations in its header) seeds six dwords, stores bytes and shorts
into them (one short crossing a dword) and reads them back, from lane 0. This runs it as one wave32 group through
spirv_mini_interp with vk_compute_test's buffer, checks the words against a byte-level model, and that check.py and
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
SEED = {0x40: 0x11223344, 0x44: 0x55667788, 0x48: 0x99AABBCC, 0x4C: 0xDDDDDDDD, 0x50: 0x01020304, 0x54: 0x05060708}


def model():
    mem = dict(SEED)

    def store_byte(addr, v):
        d, sh = addr & ~3, 8 * (addr & 3)
        mem[d] = (mem[d] & ~(0xFF << sh)) | ((v & 0xFF) << sh)

    def store_short(addr, v):                     # one byte at a time: a short at byte 3 crosses into the next dword
        store_byte(addr, v)
        store_byte(addr + 1, v >> 8)

    store_byte(0x41, 0xAB)
    store_short(0x46, 0xCDEF)
    store_short(0x4B, 0x1234)
    store_byte(0x50, 0xFFFFFF5A)
    store_short(0x54, 0xABCD1234)
    byte = lambda a: (mem[a & ~3] >> (8 * (a & 3))) & 0xFF
    sh = byte(0x46) | byte(0x47) << 8
    return [mem[0x40], mem[0x44], mem[0x48], mem[0x4C], mem[0x50], mem[0x54], byte(0x41),
            sh | 0xFFFF0000 if sh & 0x8000 else sh]


EXPECTED = model()


class BufferStoreSubFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x1122AB44, 0xCDEF7788, 0x34AABBCC, 0xDDDDDD12, 0x0102035A, 0x05061234, 0xAB,
                                    0xFFFFCDEF])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_buffer_store_sub", fssm=FSSM)
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
                m = re.search(r'"cs_w32_buffer_store_sub"\)?[,:]\s*\[([^\]]*)\]', f.read())
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
