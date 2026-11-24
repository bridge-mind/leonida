"""Semantic interpreter tests for RDNA2 v_alignbyte_b32.

v_alignbyte_b32 D, S0, S1, S2: the low 32 bits of ({S0, S1} >> (8 * (S2 & 3))), S1 being the low word (RDNA2 ISA
reference, V_ALIGNBYTE_B32). Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the mrt0 word with a plain-Python reference. The interpreter
rejects shifts of 32 or more (undefined in SPIR-V), so S2 & 3 == 0 also checks that the translation never emits one,
including on the side of the OpSelect that is discarded.
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

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_alignbyte(src0: int, src1: int, sel: int) -> int:
    return ((((src0 & MASK32) << 32) | (src1 & MASK32)) >> (8 * (sel & 3))) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_alignbyte.s")
    with open(os.path.join(_TMP_DIR.name, "ps_alignbyte.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_alignbyte(src0: int, src1: int, sel: int, inline: bool = False) -> int:
    if _TMP_DIR is None:
        setUpModule()
    lines = [f"\tv_mov_b32 v1, {src0 & MASK32:#x}", f"\tv_mov_b32 v2, {src1 & MASK32:#x}"]
    if inline:
        lines.append(f"\tv_alignbyte_b32 v0, v1, v2, {sel}")
    else:
        lines += [f"\tv_mov_b32 v3, {sel & MASK32:#x}", "\tv_alignbyte_b32 v0, v1, v2, v3"]
    asm = "; test_alignbyte\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class AlignbyteTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_alignbyte(0x8899AABB, 0x11223344, 0), 0x11223344)
        self.assertEqual(ref_alignbyte(0x8899AABB, 0x11223344, 1), 0xBB112233)
        self.assertEqual(ref_alignbyte(0x8899AABB, 0x11223344, 3), 0x99AABB11)
        self.assertEqual(ref_alignbyte(0x8899AABB, 0x11223344, 4), 0x11223344)

    def test_every_byte_shift(self):
        for src0, src1 in ((0x8899AABB, 0x11223344), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF), (0x80000001, 0x7FFFFFFE)):
            for sel in range(4):
                with self.subTest(src0=hex(src0), src1=hex(src1), sel=sel):
                    self.assertEqual(run_alignbyte(src0, src1, sel), ref_alignbyte(src0, src1, sel))

    def test_shift_zero_returns_low_word(self):
        for sel in (0, 4, 8, 0x100, 0xFFFFFFFC):
            with self.subTest(sel=hex(sel)):
                self.assertEqual(run_alignbyte(0xDEADBEEF, 0x12345678, sel), 0x12345678)

    def test_selector_is_masked_to_2_bits(self):
        for sel in (5, 6, 7, 0xFFFFFFFF, 0x80000002):
            with self.subTest(sel=hex(sel)):
                self.assertEqual(run_alignbyte(0x8899AABB, 0x11223344, sel), ref_alignbyte(0x8899AABB, 0x11223344, sel))

    def test_inline_selector(self):
        for sel in range(4):
            with self.subTest(sel=sel):
                self.assertEqual(run_alignbyte(0x8899AABB, 0x11223344, sel, inline=True),
                                 ref_alignbyte(0x8899AABB, 0x11223344, sel))


if __name__ == "__main__":
    unittest.main()
