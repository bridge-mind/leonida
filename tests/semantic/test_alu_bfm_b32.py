"""Semantic translation tests for RDNA2 32-bit bitfield mask v_bfm_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: ((1 << (width & 31)) - 1) << (offset & 31).

Covers:
  - Basic mask generation for small widths/offsets
  - Width 0 -> all zeros
  - Width 31 -> all ones below bit 31
  - Offset up to 31 (mask shifted to top of dword)
  - Width and offset masked to 5 bits (values >= 32 wrap)
  - Negative control: result must not equal the input for non-trivial cases
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


def ref_bfm32(width: int, offset: int) -> int:
    """Plain-Python reference for v_bfm_b32."""
    w = width & 31
    o = offset & 31
    if w == 0:
        return 0
    return (((1 << w) - 1) << o) & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bfm_b32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bfm_b32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(width: int, offset: int) -> int:
    """Translate and run a shader that computes v_bfm_b32 v1, v2, v3 with v2=width, v3=offset."""
    asm = (
        "; test_bfm_b32\n"
        f"\tv_mov_b32_e32 v2, {width & M32:#x}\n"
        f"\tv_mov_b32_e32 v3, {offset & M32:#x}\n"
        "\tv_bfm_b32 v1, v2, v3\n"
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


class ALUVBitfieldMaskB32Tests(unittest.TestCase):
    def test_bfm_b32_basic(self):
        # width=4, offset=0 -> 0b1111 = 0xF
        self.assertEqual(run_op(4, 0), 0x0000000F)
        # width=4, offset=4 -> 0b11110000 = 0xF0
        self.assertEqual(run_op(4, 4), 0x000000F0)
        # width=8, offset=16 -> 0xFF0000
        self.assertEqual(run_op(8, 16), 0x00FF0000)

    def test_bfm_b32_width_zero(self):
        # width=0 -> all zeros regardless of offset
        for offset in (0, 1, 16, 31, 100):
            with self.subTest(offset=offset):
                self.assertEqual(run_op(0, offset), 0)

    def test_bfm_b32_reference(self):
        cases = [
            (1, 0, 0x00000001),
            (1, 31, 0x80000000),
            (31, 0, 0x7FFFFFFF),
            (32, 0, 0x00000000),   # width masked to 0 -> all zeros
            (33, 1, 0x00000002),   # width masked to 1 -> bit 1
            (5, 31, 0x80000000),   # width=5, offset=31 -> 31<<31 in 32 bits = bit 31 only
            (16, 16, 0xFFFF0000),  # width=16, offset=16
        ]
        for width, offset, expected in cases:
            with self.subTest(width=width, offset=offset):
                self.assertEqual(run_op(width, offset), expected)
                self.assertEqual(run_op(width, offset), ref_bfm32(width, offset))

    def test_bfm_b32_masking(self):
        # width and offset are masked to 5 bits (values >= 32 wrap)
        self.assertEqual(run_op(32, 0), ref_bfm32(32, 0))   # width->0
        self.assertEqual(run_op(33, 0), ref_bfm32(33, 0))   # width->1
        self.assertEqual(run_op(0, 32), ref_bfm32(0, 32))   # offset->0
        self.assertEqual(run_op(1, 33), ref_bfm32(1, 33))   # offset->1

    def test_bfm_b32_negative_control(self):
        # For non-trivial width/offset, result differs from both inputs.
        width, offset = 8, 4
        res = run_op(width, offset)
        expected = ref_bfm32(width, offset)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, width)
        self.assertNotEqual(res, offset)


if __name__ == "__main__":
    unittest.main()
