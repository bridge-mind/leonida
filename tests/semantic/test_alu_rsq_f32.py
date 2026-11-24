"""Semantic translation tests for RDNA2 32-bit inverse square root v_rsq_f32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: 1.0 / math.sqrt(x) for finite positive x.

Covers:
  - Exact values: rsq(1) = 1, rsq(4) = 0.5, rsq(16) = 0.25, rsq(0.25) = 2
  - rsq(2) with tolerance 1e-6
  - rsq(+inf) -> +0
  - Reference comparison across varied positive bit patterns
  - 50 randomized positive cases vs reference
  - Negative control: result differs from input for non-trivial cases
"""
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def bits32(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def ref_rsq32(x_bits):
    """Plain-Python reference for v_rsq_f32 (finite positive inputs)."""
    x = f32(x_bits)
    if x == float("inf"):
        return bits32(0.0)  # rsq(+inf) = +0
    if x <= 0.0 or math.isnan(x):
        return x_bits  # out of domain for this test
    return bits32(1.0 / math.sqrt(x))


def run_op(x_bits):
    """Translate and run a shader that computes v_rsq_f32 v1, v0 with v0=x_bits."""
    asm = "; test_rsq_f32\n"
    asm += "\tv_mov_b32_e32 v0, %#x\n" % (x_bits & M32)
    asm += "\tv_rsq_f32 v1, v0\n"
    asm += "\ts_waitcnt vmcnt(0)\n"
    asm += "\texp mrt0 v1, v1, v1, v1 done vm\n"
    asm += "\ts_endpgm\n"
    tmp = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(tmp.name, "disasm")
    os.mkdir(disasm_dir)
    src_path = os.path.join(disasm_dir, "ps_rsq_f32.s")
    with open(os.path.join(tmp.name, "ps_rsq_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
    w = si.run_fragment(spv)["%out_mrt0"]
    tmp.cleanup()
    return w[0] & M32


class ALUVRsqF32Tests(unittest.TestCase):
    def test_rsq_f32_exact_values(self):
        cases = [
            (bits32(1.0), bits32(1.0)),
            (bits32(4.0), bits32(0.5)),
            (bits32(16.0), bits32(0.25)),
            (bits32(0.25), bits32(2.0)),
        ]
        for x_bits, expected in cases:
            with self.subTest(x=repr(f32(x_bits))):
                res = run_op(x_bits)
                self.assertEqual(res, expected)
                self.assertEqual(res, ref_rsq32(x_bits))

    def test_rsq_f32_rsq2(self):
        x_bits = bits32(2.0)
        res = run_op(x_bits)
        self.assertEqual(res, ref_rsq32(x_bits))
        res_f = f32(res)
        self.assertAlmostEqual(res_f, 1.0 / math.sqrt(2.0), delta=1e-6)

    def test_rsq_f32_inf(self):
        res = run_op(bits32(float("inf")))
        self.assertEqual(res, bits32(0.0))
        self.assertEqual(res, ref_rsq32(bits32(float("inf"))))

    def test_rsq_f32_reference(self):
        cases = [
            bits32(1.0), bits32(2.0), bits32(3.0), bits32(10.0),
            bits32(100.0), bits32(0.5), bits32(0.01), bits32(65536.0),
            bits32(1e-10), bits32(1e10),
        ]
        for x_bits in cases:
            with self.subTest(x=repr(f32(x_bits))):
                res = run_op(x_bits)
                self.assertEqual(res, ref_rsq32(x_bits))

    def test_rsq_f32_random_vs_reference(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x_bits = rng.getrandbits(31)  # non-negative, clear sign bit
            x = f32(x_bits)
            if x <= 0.0 or math.isinf(x):
                continue
            with self.subTest(x=repr(x)):
                res = run_op(x_bits)
                self.assertEqual(res, ref_rsq32(x_bits))

    def test_rsq_f32_negative_control(self):
        # Result differs from input for non-trivial inputs.
        x_bits = bits32(4.0)
        res = run_op(x_bits)
        self.assertEqual(res, bits32(0.5))
        self.assertNotEqual(res, x_bits)


if __name__ == "__main__":
    unittest.main()
