"""Value tests for RDNA2 ds_mskor_b32 / ds_mskor_rtn_b32, run through spirv_mini_interp on the CPU.

RDNA2 ISA, "Data Share Instructions": tmp = M[A]; M[A] = (tmp & ~DATA) | DATA2; the _rtn_ form returns tmp in vdst.
The translator does an atomic AND with ~DATA (whose result is tmp) and then an atomic OR with DATA2. Each case runs
one invocation (lane 0): it seeds the dword, runs the op and reads the dword back. Standard library only.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_atomic  # noqa: E402  (also puts tools/ on sys.path)

M32 = 0xFFFFFFFF
VALUES = [0, M32, 0x12345678, 0xF0F0F0F0, 0x0000FFFF, 0x80000001]


def mskor(m, mask, data):
    return (m & ~mask | data) & M32


class LdsMskor(unittest.TestCase):
    def test_returning_form(self):
        for m in VALUES:
            for mask in VALUES[::2]:
                for data in VALUES[1::2]:
                    with self.subTest(m=hex(m), mask=hex(mask), data=hex(data)):
                        got = run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", m, data=(mask, data))
                        self.assertEqual(got, (m, mskor(m, mask, data)))

    def test_non_returning_form_leaves_vdst_alone(self):
        for m, mask, data in ((0x12345678, 0x0000FF00, 0x0000AB00), (M32, M32, 0), (0, 0, 0x55)):
            with self.subTest(m=hex(m)):
                self.assertEqual(run_atomic("ds_mskor_b32 v1, v3, v4", m, data=(mask, data)),
                                 (0xDEAD, mskor(m, mask, data)))

    def test_data_outside_the_mask_and_offset(self):
        # DATA2 bits outside DATA are ORed in too: (tmp & ~DATA) | DATA2, not a masked insert.
        self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", 0x00FF00FF, data=(0x0000FFFF, 0xFFFF0000)),
                         (0x00FF00FF, 0xFFFF0000 | 0x00FF0000))
        self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", 7, addr=0x20, offset=0x24, data=(3, 8)),
                         (7, 12))

    def test_seeded(self):
        rng = random.Random(0x3502)
        for _ in range(20):
            m, mask, data = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(m=hex(m), mask=hex(mask), data=hex(data)):
                self.assertEqual(run_atomic("ds_mskor_rtn_b32 v5, v1, v3, v4", m, data=(mask, data)),
                                 (m, mskor(m, mask, data)))


if __name__ == "__main__":
    unittest.main()
