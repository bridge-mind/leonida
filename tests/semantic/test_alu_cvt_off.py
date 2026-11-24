"""Semantic interpreter tests for RDNA2 v_cvt_off_f32_i4.

RDNA ISA: D.f32 = CVT_OFF_TABLE[S0[3:0]]: the low 4 bits of S0 as a signed integer, divided by 16, so 0..7 give
0.0 .. 0.4375 and 8..15 give -0.5 .. -0.0625 (a sample-offset conversion). Bits 4-31 are ignored. Output modifiers
(mul:2 / mul:4 / div:2, then clamp) apply to the result. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the exported word with a
plain-Python reference (every result is an exact, normal binary32 value or zero).
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# The ISA's table, written out: index -> value.
TABLE = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
         -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625]


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cvt_off.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cvt_off.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x: int, line: str = "v_cvt_off_f32_i4 v0, v1") -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"v_mov_b32 v1, {x:#x}", line]
    asm = "; test_cvt_off\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class CvtOffTests(unittest.TestCase):
    def test_table(self):
        for i, want in enumerate(TABLE):
            with self.subTest(i=i):
                self.assertEqual(run_op(i), f2u(want))

    def test_upper_bits_are_ignored(self):
        for x in (0xFFFFFFF0, 0x12345678, 0x80000009, 0xABCDEF0F):
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(x), f2u(TABLE[x & 15]))

    def test_output_modifiers(self):
        self.assertEqual(run_op(7, "v_cvt_off_f32_i4_e64 v0, v1 mul:4"), f2u(1.75))
        self.assertEqual(run_op(7, "v_cvt_off_f32_i4_e64 v0, v1 mul:4 clamp"), f2u(1.0))
        self.assertEqual(run_op(8, "v_cvt_off_f32_i4_e64 v0, v1 clamp"), f2u(0.0))
        self.assertEqual(run_op(8, "v_cvt_off_f32_i4_e64 v0, v1 div:2"), f2u(-0.25))


if __name__ == "__main__":
    unittest.main()
