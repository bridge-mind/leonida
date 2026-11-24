"""Semantic translation tests for RDNA2 32-bit bit count v_bcnt_u32_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: (popcount(x) + acc) & 0xFFFFFFFF.

Covers:
  - Basic popcount: x=0, x=1, x=all-ones
  - Accumulator: popcount(x) + acc
  - 32-bit wraparound on overflow
  - Reference comparison across varied bit patterns
  - Negative control: result differs from inputs for non-trivial cases
"""
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


def ref_bcnt32(x: int, acc: int) -> int:
    """Plain-Python reference for v_bcnt_u32_b32: popcount(x) + acc, masked to 32 bits."""
    return (bin(x & M32).count("1") + acc) & M32


def run_op(x: int, acc: int) -> int:
    """Translate and run a shader that computes v_bcnt_u32_b32 v1, v2, v3 with v2=x, v3=acc.

    Uses a unique temp directory per call to avoid file-sharing issues.
    """
    asm = (
        "; test_bcnt_u32\n"
        f"\tv_mov_b32_e32 v2, {x & M32:#x}\n"
        f"\tv_mov_b32_e32 v3, {acc & M32:#x}\n"
        "\tv_bcnt_u32_b32 v1, v2, v3\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v1, v1, v1 done vm\n"
        "\ts_endpgm\n"
    )
    tmp = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(tmp.name, "disasm")
    os.mkdir(disasm_dir)
    src_path = os.path.join(disasm_dir, "ps_bcnt_u32.s")
    with open(os.path.join(tmp.name, "ps_bcnt_u32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
    w = si.run_fragment(spv)["%out_mrt0"]
    tmp.cleanup()
    return w[0] & M32


class ALUVBitcntU32Tests(unittest.TestCase):
    def test_bcnt_u32_basic(self):
        # popcount(0) = 0
        self.assertEqual(run_op(0, 0), 0)
        # popcount(1) = 1
        self.assertEqual(run_op(1, 0), 1)
        # popcount(0xFFFFFFFF) = 32
        self.assertEqual(run_op(M32, 0), 32)
        # popcount(0xAAAAAAAA) = 16
        self.assertEqual(run_op(0xAAAAAAAA, 0), 16)
        # popcount(0x55555555) = 16
        self.assertEqual(run_op(0x55555555, 0), 16)

    def test_bcnt_u32_accumulator(self):
        # popcount(0xFF) = 8, acc = 10 -> 18
        self.assertEqual(run_op(0xFF, 10), 18)
        # popcount(0) = 0, acc = 0x12345678
        self.assertEqual(run_op(0, 0x12345678), 0x12345678)

    def test_bcnt_u32_wraparound(self):
        # popcount(0xFFFFFFFF) = 32, acc = 0xFFFFFFFE -> 32 + 0xFFFFFFFE = 0x10000001E -> 0x1E
        self.assertEqual(run_op(M32, M32 - 1), 30)
        # popcount(1) = 1, acc = M32 -> 1 + M32 = 0x100000000 -> 0
        self.assertEqual(run_op(1, M32), 0)

    def test_bcnt_u32_reference(self):
        cases = [
            (0, 0, 0),
            (1, 0, 1),
            (2, 0, 1),       # 0b10 -> 1 bit
            (3, 0, 2),       # 0b11 -> 2 bits
            (0xFF, 0, 8),
            (0xFFFF, 0, 16),
            (0xFFFFFFFF, 0, 32),
            (0x00000000, 0xABCD, 0xABCD),
            (0x12345678, 0, 13),  # popcount(0x12345678) = 13
        ]
        for x, acc, expected in cases:
            with self.subTest(x=hex(x), acc=hex(acc)):
                self.assertEqual(run_op(x, acc), expected)
                self.assertEqual(run_op(x, acc), ref_bcnt32(x, acc))

    def test_bcnt_u32_random_vs_reference(self):
        import random
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x = rng.getrandbits(32)
            acc = rng.getrandbits(32)
            with self.subTest(x=hex(x), acc=hex(acc)):
                self.assertEqual(run_op(x, acc), ref_bcnt32(x, acc))

    def test_bcnt_u32_negative_control(self):
        # For non-trivial inputs, result differs from each input.
        x, acc = 0x12345678, 0x9ABCDEF0
        res = run_op(x, acc)
        expected = ref_bcnt32(x, acc)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, x)
        self.assertNotEqual(res, acc)


if __name__ == "__main__":
    unittest.main()
