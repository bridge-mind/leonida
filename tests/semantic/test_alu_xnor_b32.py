"""Semantic translation tests for RDNA2 32-bit bitwise XNOR v_xnor_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: ~(x ^ y) & 0xFFFFFFFF.

Covers:
  - Basic XNOR: all-zeros/all-ones, complementary inputs
  - Identical inputs -> all-ones
  - Reference comparison across varied bit patterns
  - Negative control: result differs from both inputs for non-trivial cases
"""
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


def ref_xnor32(x: int, y: int) -> int:
    """Plain-Python reference for v_xnor_b32: bitwise NOT of XOR, masked to 32 bits."""
    return (~(x ^ y)) & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_xnor_b32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_xnor_b32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x: int, y: int) -> int:
    """Translate and run a shader that computes v_xnor_b32 v1, v2, v3 with v2=x, v3=y."""
    asm = (
        "; test_xnor_b32\n"
        f"\tv_mov_b32_e32 v2, {x & M32:#x}\n"
        f"\tv_mov_b32_e32 v3, {y & M32:#x}\n"
        "\tv_xnor_b32 v1, v2, v3\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v1, v1, v1 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    return w[0] & M32


class ALUVXnorB32Tests(unittest.TestCase):
    def test_xnor_b32_basic(self):
        # ~(0 ^ 0) = ~0 = 0xFFFFFFFF
        self.assertEqual(run_op(0, 0), M32)
        # ~(0xFFFFFFFF ^ 0xFFFFFFFF) = ~0 = 0xFFFFFFFF
        self.assertEqual(run_op(M32, M32), M32)
        # ~(0 ^ 0xFFFFFFFF) = ~0xFFFFFFFF = 0
        self.assertEqual(run_op(0, M32), 0)
        # ~(0xFFFFFFFF ^ 0) = ~0xFFFFFFFF = 0
        self.assertEqual(run_op(M32, 0), 0)

    def test_xnor_b32_reference(self):
        cases = [
            (0x12345678, 0x12345678, M32),          # x ^ x = 0 -> ~0 = all ones
            (0x12345678, 0x87654321, 0x6AAEEAA6),   # ~(0x95511559)
            (0x00000001, 0x00000001, M32),
            (0x00000001, 0x00000000, 0xFFFFFFFE),   # ~(1 ^ 0) = ~1 = 0xFFFFFFFE
            (0xAAAAAAAA, 0x55555555, 0),            # ~(0xFFFFFFFF) = 0
            (0xAAAAAAAA, 0xAAAAAAAA, M32),
            (0x0F0F0F0F, 0xF0F0F0F0, 0),            # ~(0xFFFFFFFF) = 0
        ]
        for x, y, expected in cases:
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y), expected)
                self.assertEqual(run_op(x, y), ref_xnor32(x, y))

    def test_xnor_b32_random_vs_reference(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x = rng.getrandbits(32)
            y = rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y), ref_xnor32(x, y))

    def test_xnor_b32_negative_control(self):
        # For non-trivial inputs, result differs from both inputs.
        x, y = 0x12345678, 0x9ABCDEF0
        res = run_op(x, y)
        expected = ref_xnor32(x, y)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, x)
        self.assertNotEqual(res, y)


if __name__ == "__main__":
    unittest.main()
