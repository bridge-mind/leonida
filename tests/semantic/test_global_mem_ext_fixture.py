"""cs_global_mem_ext: the GPU test of the global/buffer atomics, 8/16-bit global stores and d16 loads, on the CPU.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_global_mem_ext.asm (header: derivations) has group 0 thread 0 seed a heap with MUBUF stores,
run global_atomic_add / cmpswap / or_x2, buffer_atomic_xor_x2, a global_store_short that crosses a dword and a
global_load_sbyte_d16_hi, and store eight words. This runs its fixture (tests/semantic/regen_fixtures.py) as one
64-thread group through spirv_mini_interp, checks the words against values computed here, and checks that
check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
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
M32 = 0xFFFFFFFF

# build.py's default compute header: 4 user SGPRs (the V#), workgroup id in s4, 512 bytes of LDS, 64x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x2800, 0x31016FAC]          # base 0 (the heap), 0x2800 bytes, raw bounds


def short_store(lo_word, hi_word, at, value):
    b = bytearray(lo_word.to_bytes(4, "little") + hi_word.to_bytes(4, "little"))
    b[at:at + 2] = value.to_bytes(2, "little")
    return int.from_bytes(b[:4], "little"), int.from_bytes(b[4:], "little")


SHORT = short_store(0x44332211, 0x88776655, 3, 0xBEEF)
EXPECTED = [
    5,                                                  # global_atomic_add glc: old value
    5 + 7,                                              # the dword after it
    9 << 8 | 0x77,                                      # global_atomic_cmpswap: old << 8 | new (9 == CMP)
    0x00F00000 | 0xFF000000,                            # global_atomic_or_x2: high dword after
    0x12345678 ^ 0xFFFF0000,                            # buffer_atomic_xor_x2: low dword after
    SHORT[0], SHORT[1],                                 # global_store_short across 0x2020 / 0x2024
    0xFF80 << 16 | 0x1234,                              # global_load_sbyte_d16_hi of 0x80, low half kept
]


class GlobalMemExtFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x5, 0xC, 0x977, 0xFFF00000, 0xEDCB5678, 0xEF332211, 0x887766BE, 0xFF801234])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_global_mem_ext", fssm=FSSM)
        mem = {}
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                         push_constants=VSHARP + [0] * 32, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem.get(4 * k), want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_global_mem_ext"[,:]\s*(?:"40",\s*)?\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
