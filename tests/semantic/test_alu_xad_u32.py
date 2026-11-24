"""Semantic translation tests for RDNA2 32-bit xor-add v_xad_u32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: ((x ^ y) + z) & 0xFFFFFFFF.

Covers:
  - Basic xor-add: (x ^ y) + z
  - z = 0: pure XOR
  - x = y: XOR cancels, result = z
  - x = 0: result = y + z
  - 32-bit wraparound on overflow
  - Reference comparison across varied bit patterns
  - Negative control: result differs from each input for non-trivial cases
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


def ref_xad32(x: int, y: int, z: int) -> int:
    """Plain-Python reference for v_xad_u32: ((x ^ y) + z) mod 2^32."""
    return ((x ^ y) + z) & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_xad_u32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_xad_u32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x: int, y: int, z: int) -> int:
    """Translate and run a shader that computes v_xad_u32 v1, v2, v3, v4 with v2=x, v3=y, v4=z."""
    asm = (
        "; test_xad_u32\n"
        f"\tv_mov_b32_e32 v2, {x & M32:#x}\n"
        f"\tv_mov_b32_e32 v3, {y & M32:#x}\n"
        f"\tv_mov_b32_e32 v4, {z & M32:#x}\n"
        "\tv_xad_u32 v1, v2, v3, v4\n"
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


class ALUVXadU32Tests(unittest.TestCase):
    def test_xad_u32_basic(self):
        # (5 ^ 3) + 2 = 6 + 2 = 8
        self.assertEqual(run_op(5, 3, 2), 8)
        # z = 0: pure XOR
        self.assertEqual(run_op(0x12345678, 0x87654321, 0), 0x95511559)
        # x = y: XOR cancels, result = z
        self.assertEqual(run_op(0xDEADBEEF, 0xDEADBEEF, 0x12345678), 0x12345678)
        # x = 0: result = y + z
        self.assertEqual(run_op(0, 0x1000, 0x2000), 0x3000)

    def test_xad_u32_wraparound(self):
        # 32-bit overflow wraps
        self.assertEqual(run_op(0, 0, M32 + 1 & M32), 0)  # z = 0 (M32+1 masked to 0)
        self.assertEqual(run_op(0, 0, M32), M32)
        self.assertEqual(run_op(0, 0, 1), 1)
        # (0xFFFFFFFF ^ 0) + 1 = M32 + 1 wraps to 0
        self.assertEqual(run_op(0, M32, 1), 0)

    def test_xad_u32_reference(self):
        cases = [
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000),  # x==y -> 0 + z
            (0xFFFFFFFF, 0x00000000, 0x00000001, 0x00000000),  # M32 + 1 wraps to 0
            (0x12345678, 0x87654321, 0x00000000, 0x95511559),
            (0xAAAAAAAA, 0x55555555, 0x00000000, 0xFFFFFFFF),
            (0xAAAAAAAA, 0x55555555, 0x00000001, 0x00000000),  # 0xFFFFFFFF + 1 wraps
            (0x0F0F0F0F, 0xF0F0F0F0, 0x12345678, 0x12345677),  # x^y = M32, +z wraps to z-1
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(x, y, z), expected)
                self.assertEqual(run_op(x, y, z), ref_xad32(x, y, z))

    def test_xad_u32_random_vs_reference(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x = rng.getrandbits(32)
            y = rng.getrandbits(32)
            z = rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(x, y, z), ref_xad32(x, y, z))

    def test_xad_u32_negative_control(self):
        # For non-trivial inputs, result differs from each input.
        x, y, z = 0x12345678, 0x9ABCDEF0, 0x01234567
        res = run_op(x, y, z)
        expected = ref_xad32(x, y, z)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, x)
        self.assertNotEqual(res, y)
        self.assertNotEqual(res, z)


if __name__ == "__main__":
    unittest.main()
