"""Semantic translation tests for RDNA2 32-bit reciprocal v_rcp_f32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: 1.0 / x rounded to binary32 for finite non-zero x,
+infinity for +inf, -0.0 for -inf, and NaN for NaN.

v_rcp_f32(0) is undefined: the translator lowers it to OpFDiv 1.0/x and SPIR-V
leaves division by zero undefined, so the interpreter refuses it (that case is
asserted separately, not fed through the reference).

Covers:
  - Exact reciprocals: rcp(1)=1, rcp(2)=0.5, rcp(4)=0.25, rcp(0.5)=2, rcp(0.25)=4
  - Negative inputs: rcp(-2)=-0.5, rcp(-0.5)=-2
  - rcp(3) within 1e-6 tolerance
  - rcp(+inf)=+0.0 and rcp(-inf)=-0.0 (signed zeros preserved)
  - rcp(NaN)=NaN
  - Random cases vs the reference
  - Negative control: result differs from the input for non-trivial cases
  - rcp(0) is undefined and refused by the interpreter
"""
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

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


def bits32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def is_nan(b: int) -> bool:
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    return e == 0xFF and m != 0


def ref_rcp32(x_bits: int) -> int:
    """Plain-Python reference for v_rcp_f32: 1.0 / x rounded to binary32."""
    x = f32(x_bits)
    if math.isnan(x):
        return bits32(float("nan"))
    if x == float("inf"):
        return bits32(0.0)
    if x == float("-inf"):
        return bits32(-0.0)
    return bits32(1.0 / x)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_rcp_f32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_rcp_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x_bits: int) -> int:
    """Translate and run a shader that computes v_rcp_f32 v1, v0 with v0=x_bits."""
    x = f32(x_bits)
    asm = (
        "; test_rcp_f32\n"
        f"\tv_mov_b32_e32 v0, {x_bits & M32:#x}   ; {x!r}\n"
        "\tv_rcp_f32 v1, v0\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v1, v1, v1 done vm\n"
        "\ts_endpgm\n"
    )
    tmp = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(tmp.name, "disasm")
    os.mkdir(disasm_dir)
    src_path = os.path.join(disasm_dir, "ps_rcp_f32.s")
    with open(os.path.join(tmp.name, "ps_rcp_f32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
    w = si.run_fragment(spv)["%out_mrt0"]
    tmp.cleanup()
    return w[0] & M32


class ALUVRcpF32Tests(unittest.TestCase):
    def test_rcp_f32_exact_reciprocals(self):
        cases = [
            (bits32(1.0), bits32(1.0)),
            (bits32(2.0), bits32(0.5)),
            (bits32(4.0), bits32(0.25)),
            (bits32(0.5), bits32(2.0)),
            (bits32(0.25), bits32(4.0)),
            (bits32(8.0), bits32(0.125)),
        ]
        for x_bits, expected in cases:
            with self.subTest(x=f32(x_bits)):
                res = run_op(x_bits)
                self.assertEqual(res, expected)
                self.assertEqual(res, ref_rcp32(x_bits))

    def test_rcp_f32_negative_inputs(self):
        cases = [
            (bits32(-2.0), bits32(-0.5)),
            (bits32(-0.5), bits32(-2.0)),
            (bits32(-4.0), bits32(-0.25)),
            (bits32(-0.25), bits32(-4.0)),
        ]
        for x_bits, expected in cases:
            with self.subTest(x=f32(x_bits)):
                res = run_op(x_bits)
                self.assertEqual(res, expected)
                self.assertEqual(res, ref_rcp32(x_bits))

    def test_rcp_f32_reciprocal_of_3(self):
        x_bits = bits32(3.0)
        res = run_op(x_bits)
        self.assertEqual(res, ref_rcp32(x_bits))
        res_f = f32(res)
        self.assertAlmostEqual(res_f, 1.0 / 3.0, delta=1e-6)

    def test_rcp_f32_signed_zeros_and_special(self):
        cases = [
            (bits32(float("inf")), bits32(0.0), False),    # rcp(+inf) = +0.0
            (bits32(float("-inf")), bits32(-0.0), False),  # rcp(-inf) = -0.0
            (bits32(float("nan")), bits32(float("nan")), True),  # rcp(NaN) = NaN
        ]
        for x_bits, expected, expect_nan in cases:
            with self.subTest(x=f32(x_bits)):
                res = run_op(x_bits)
                self.assertEqual(res, ref_rcp32(x_bits))
                self.assertEqual(is_nan(res), expect_nan)

    def test_rcp_f32_random_vs_reference(self):
        import random
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x_bits = rng.getrandbits(32)
            if x_bits == 0 or x_bits == M32:
                continue  # +0/-0: division by zero is undefined
            x = f32(x_bits)
            if math.isinf(x) and x < 0:
                continue  # covered above
            with self.subTest(x=x):
                res = run_op(x_bits)
                self.assertEqual(res, ref_rcp32(x_bits))

    def test_rcp_f32_negative_control(self):
        # Result differs from the input for a non-trivial input.
        x_bits = bits32(2.0)
        res = run_op(x_bits)
        self.assertEqual(res, bits32(0.5))
        self.assertNotEqual(res, x_bits)

    def test_rcp_f32_zero_is_undefined(self):
        # v_rcp_f32(0) lowers to OpFDiv 1.0/0, which SPIR-V leaves undefined.
        with self.assertRaises(si.UnsupportedConstruct):
            run_op(bits32(0.0))


if __name__ == "__main__":
    unittest.main()
