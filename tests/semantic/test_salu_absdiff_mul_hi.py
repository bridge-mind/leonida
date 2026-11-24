"""Semantic interpreter tests for RDNA2 s_absdiff_i32 and s_mul_hi_i32.

s_absdiff_i32 (RDNA2 ISA): D = S0 - S1 as a signed 32-bit value, negated if negative; SCC = (D != 0). The
subtraction wraps first, so 0x7FFFFFFF - 0x80000000 is -1 and gives 1, and a difference of -2^31 stays 0x80000000.
s_mul_hi_i32: D = the high 32 bits of the signed 64-bit product S0 * S1; SCC is not written. Each case translates a
synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the
exported words (D, and SCC through s_cselect_b32) with a plain-Python reference.
"""
import os
import random
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678, 0xEDCBA988, 0x10000]


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x >> 31 else x


def ref_absdiff(x: int, y: int) -> int:
    d = s32(x - y)                   # the 32-bit difference, wrapped
    return (-d if d < 0 else d) & MASK32


def ref_mul_hi_i32(x: int, y: int) -> int:
    return ((s32(x) * s32(y)) >> 32) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_absdiff.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_absdiff.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(op: str, x: int, y: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = op(s2, s3); return (s0, SCC after)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}", f"s_mov_b32 s4, {scc_in}", "s_cmp_eq_u32 s4, 1",
             f"{op} s0, s2, s3", "s_cselect_b32 s1, 1, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"]
    asm = "; test_salu_absdiff\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class AbsdiffTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_absdiff(1, 3), 2)
        self.assertEqual(ref_absdiff(0x7FFFFFFF, 0x80000000), 1)            # wraps to -1 first
        self.assertEqual(ref_absdiff(0, 0x80000000), 0x80000000)            # -2^31 is its own absolute value
        self.assertEqual(ref_absdiff(0xFFFFFFFF, 1), 2)                     # -1 - 1

    def test_values_and_scc(self):
        for x in VALUES:
            for y in VALUES:
                want = ref_absdiff(x, y)
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertEqual(run_op("s_absdiff_i32", x, y, 1 - int(want != 0)), (want, int(want != 0)))

    def test_seeded(self):
        rng = random.Random(0xAD)
        for _ in range(30):
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op("s_absdiff_i32", x, y, 0), (ref_absdiff(x, y), int(x != y)))


class MulHiI32Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_mul_hi_i32(0xFFFFFFFF, 1), 0xFFFFFFFF)         # -1 * 1: high word -1
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x80000000), 0x40000000)
        self.assertEqual(ref_mul_hi_i32(0x80000000, 0x7FFFFFFF), 0xC0000000)

    def test_values_and_scc_kept(self):
        for x in VALUES:
            for y in VALUES[::2]:
                for scc_in in (0, 1):
                    with self.subTest(x=hex(x), y=hex(y), scc=scc_in):
                        self.assertEqual(run_op("s_mul_hi_i32", x, y, scc_in), (ref_mul_hi_i32(x, y), scc_in))

    def test_differs_from_unsigned(self):
        # Same operands, s_mul_hi_u32 gives the unsigned high word.
        self.assertEqual(run_op("s_mul_hi_u32", 0xFFFFFFFF, 2, 0)[0], 1)
        self.assertEqual(run_op("s_mul_hi_i32", 0xFFFFFFFF, 2, 0)[0], 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
