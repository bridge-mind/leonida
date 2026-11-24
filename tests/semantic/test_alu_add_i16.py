"""Semantic translation tests for RDNA2 16-bit signed add v_add_i16.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: sign-extend both operands to 16 bits, add, mask to 16 bits
(equivalently (x + y) & 0xFFFF for modular addition).

Covers:
  - Basic 16-bit signed add
  - Positive overflow (0x7FFF + 1 wraps to 0x8000)
  - Negative overflow (-1 + -1 wraps to 0xFFFE)
  - Mixed sign addition
  - Reference comparison across varied 16-bit patterns
  - 50 randomized cases vs reference
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

M16 = 0xFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s16(v: int) -> int:
    """Sign-extend a 16-bit value."""
    v &= M16
    return v - 0x10000 if v >= 0x8000 else v


def ref_add_i16(x: int, y: int) -> int:
    """Plain-Python reference for v_add_i16: sign-extend, add, mask to 16 bits."""
    return (s16(x) + s16(y)) & M16


def run_op(x: int, y: int) -> int:
    """Translate and run a shader that computes v_add_i16 v1, v2, v3 with v2=x, v3=y.

    Uses a unique temp directory per call to avoid file-sharing issues.
    """
    asm = (
        "; test_add_i16\n"
        f"\tv_mov_b32_e32 v2, {x & M16:#x}\n"
        f"\tv_mov_b32_e32 v3, {y & M16:#x}\n"
        "\tv_add_i16 v1, v2, v3\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v1, v1, v1 done vm\n"
        "\ts_endpgm\n"
    )
    tmp = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(tmp.name, "disasm")
    os.mkdir(disasm_dir)
    src_path = os.path.join(disasm_dir, "ps_add_i16.s")
    with open(os.path.join(tmp.name, "ps_add_i16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
    w = si.run_fragment(spv)["%out_mrt0"]
    tmp.cleanup()
    return w[0] & M16


class ALUVAddI16Tests(unittest.TestCase):
    def test_add_i16_basic(self):
        self.assertEqual(run_op(0, 0), 0)
        self.assertEqual(run_op(1, 2), 3)
        self.assertEqual(run_op(100, 200), 300)
        self.assertEqual(run_op(0x7FFF, 1), 0x8000)  # positive overflow

    def test_add_i16_overflow(self):
        # -1 + -1 = -2 = 0xFFFE
        self.assertEqual(run_op(M16, M16), 0xFFFE)
        # 0x7FFF + 1 = 0x8000 (positive overflow to negative)
        self.assertEqual(run_op(0x7FFF, 0x0001), 0x8000)
        # -32768 + -1 = 0x7FFF (negative overflow to positive)
        self.assertEqual(run_op(0x8000, M16), 0x7FFF)

    def test_add_i16_reference(self):
        cases = [
            (0, 0, 0),
            (1, 1, 2),
            (0x7FFF, 1, 0x8000),
            (M16, M16, 0xFFFE),    # -1 + -1 = -2
            (0x8000, M16, 0x7FFF), # -32768 + -1 = 32767
            (0x1234, 0x5678, 0x68AC),
        ]
        for x, y, expected in cases:
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y), expected)
                self.assertEqual(run_op(x, y), ref_add_i16(x, y))

    def test_add_i16_random_vs_reference(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x = rng.getrandbits(16)
            y = rng.getrandbits(16)
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y), ref_add_i16(x, y))

    def test_add_i16_negative_control(self):
        # For non-trivial inputs, result differs from each input.
        x, y = 0x1234, 0x5678
        res = run_op(x, y)
        expected = ref_add_i16(x, y)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, x)
        self.assertNotEqual(res, y)


if __name__ == "__main__":
    unittest.main()
