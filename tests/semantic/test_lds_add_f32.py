"""ds_add_f32 / ds_add_rtn_f32, run through spirv_mini_interp on the CPU.

RDNA2 ISA, "Data Share Instructions": M = M + DATA as binary32 (round to nearest even); the _rtn_ form returns the old
M. The translator emits an OpAtomicCompareExchange loop with an OpFAdd. Each case runs one invocation: it seeds an LDS
dword, runs the op, reads the dword back and stores (vdst, M) as integer words, so no float round trip touches them.
The reference adds in double precision (exact for two binary32 values) and rounds once to binary32. Denormal and NaN
behaviour of the LDS adder is not measured and not tested here. Standard library only; no GPU, no game files.
"""
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_atomic  # noqa: E402  (also puts tools/ on sys.path)


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def add_ref(m, d):
    """Python floats are binary64: the sum is exact, the pack rounds once (RNE); past FLT_MAX it is +-inf."""
    total = f32(m) + f32(d)
    try:
        return bits(total)
    except OverflowError:
        return 0x7F800000 if total > 0 else 0xFF800000


VALUES = [bits(v) for v in (0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 3.0, 1e10, -1e10, 1e-10, 65504.0, 16777216.0,
                             0.1, -2.75)] + [0x3F800001, 0x4B7FFFFF]


class LdsAddF32(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(add_ref(bits(1.0), bits(2.0)), bits(3.0))
        self.assertEqual(add_ref(bits(16777216.0), bits(1.0)), bits(16777216.0))        # tie to even
        self.assertEqual(add_ref(bits(-0.0), bits(-0.0)), bits(-0.0))
        self.assertEqual(add_ref(bits(3e38), bits(3e38)), 0x7F800000)

    def test_returning_form(self):
        for m in VALUES:
            for d in VALUES[::3]:
                with self.subTest(m=hex(m), d=hex(d)):
                    self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", m, data=(d,)), (m, add_ref(m, d)))

    def test_non_returning_form_and_offset(self):
        self.assertEqual(run_atomic("ds_add_f32 v1, v3", bits(1.25), data=(bits(2.5),)), (0xDEAD, bits(3.75)))
        self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", bits(4.0), addr=0x20, offset=0x24,
                                    data=(bits(-1.0),)), (bits(4.0), bits(3.0)))

    def test_rounding_and_signed_zero(self):
        cases = ((16777216.0, 1.0), (16777218.0, 1.0), (1.0, 2.0 ** -24), (1.0, 3 * 2.0 ** -25), (-0.0, 0.0),
                 (3e38, 3e38), (-3e38, -3e38))                         # ties to even, -0 + +0 = +0, overflow to +-inf
        for m, d in cases:
            with self.subTest(m=m, d=d):
                self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", bits(m), data=(bits(d),))[1],
                                 add_ref(bits(m), bits(d)))

    def test_signed_zeros_and_infinities(self):
        # Hardware-verified on RDNA2 (gfx1036): signed zero preservation, infs, and NaN propagation
        # -0.0 + -0.0 -> -0.0
        self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x80000000, data=(0x80000000,)), (0x80000000, 0x80000000))
        # +0.0 + -0.0 -> +0.0
        self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x00000000, data=(0x80000000,)), (0x00000000, 0x00000000))
        # +0.0 + +0.0 -> +0.0
        self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x00000000, data=(0x00000000,)), (0x00000000, 0x00000000))
        # +Inf + 1.0 -> +Inf
        self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x7F800000, data=(bits(1.0),)), (0x7F800000, 0x7F800000))
        # +Inf + -Inf -> qNaN
        rtn, res = run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x7F800000, data=(0xFF800000,))
        self.assertEqual(rtn, 0x7F800000)
        self.assertTrue((res & 0x7F800000) == 0x7F800000 and (res & 0x007FFFFF) != 0, f"Expected NaN, got {res:#x}")
        # qNaN + 1.0 -> qNaN (verifies CAS loop terminates cleanly on NaN without hang)
        rtn_nan, res_nan = run_atomic("ds_add_rtn_f32 v5, v1, v3", 0x7FC00000, data=(bits(1.0),))
        self.assertEqual(rtn_nan, 0x7FC00000)
        self.assertTrue((res_nan & 0x7F800000) == 0x7F800000 and (res_nan & 0x007FFFFF) != 0, f"Expected NaN, got {res_nan:#x}")

    def test_seeded(self):
        rng = random.Random(0xADDF)
        for _ in range(25):
            m = bits(rng.uniform(-1e6, 1e6))
            d = bits(rng.uniform(-1e6, 1e6))
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(run_atomic("ds_add_rtn_f32 v5, v1, v3", m, data=(d,)), (m, add_ref(m, d)))


if __name__ == "__main__":
    unittest.main()
