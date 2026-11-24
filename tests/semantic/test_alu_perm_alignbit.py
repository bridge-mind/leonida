"""Semantic interpreter tests for RDNA2 byte permute and funnel shift (v_perm_b32, v_alignbit_b32).

Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares the mrt0 word with a plain-Python reference. The interpreter rejects shifts of 32 or more (undefined in
SPIR-V), so these tests also check that the translation never emits one, including on the discarded side of an
OpSelect.

v_perm_b32 D, S0, S1, SEL: result byte i is chosen by byte i of SEL from the 64-bit value {S0, S1} (S1 is bytes 0-3,
S0 bytes 4-7). Selector 0x0c gives 0x00 and anything above 0x0c gives 0xff. Reference: LLVM
llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp, computeKnownBitsForTargetNode, case AMDGPUISD::PERM. Selectors 8-11
repeat the sign bit of a 16-bit half of {S0, S1}: bit 15, 31, 47 or 63 (RDNA2 ISA reference, V_PERM_B32; LLVM's
known-bits code treats them as unknown; #211).

v_alignbit_b32 D, S0, S1, S2: the low 32 bits of ({S0, S1} >> (S2 & 31)), S1 being the low word.
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

# Selector byte values with a defined, documented result.
PERM_SELECTORS = tuple(range(12)) + (0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x7F, 0x80, 0xFF)


def ref_perm(src0: int, src1: int, sel: int) -> int:
    """v_perm_b32 reference for every selector byte."""
    data = ((src0 & MASK32) << 32) | (src1 & MASK32)
    out = 0
    for i in range(4):
        s = (sel >> (8 * i)) & 0xFF
        if s < 8:
            byte = (data >> (8 * s)) & 0xFF
        elif s < 0x0C:   # 8-11: sign bit of the 16-bit half s - 8
            byte = 0xFF if (data >> (16 * (s - 8) + 15)) & 1 else 0x00
        elif s == 0x0C:
            byte = 0x00
        else:
            byte = 0xFF
        out |= byte << (8 * i)
    return out


def ref_alignbit(src0: int, src1: int, shift: int) -> int:
    """v_alignbit_b32 reference: low 32 bits of ({src0, src1} >> (shift & 31))."""
    data = ((src0 & MASK32) << 32) | (src1 & MASK32)
    return (data >> (shift & 31)) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_perm.s")
    with open(os.path.join(_TMP_DIR.name, "ps_perm.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_vop3(op: str, src0: int, src1: int, src2: int, src2_inline: bool = False) -> int:
    """Translate and run `op v0, v1, v2, v3` (or with src2 as an inline constant), return the mrt0 word."""
    if _TMP_DIR is None:
        setUpModule()
    lines = [
        f"\tv_mov_b32 v1, {src0 & MASK32:#x}",
        f"\tv_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if src2_inline:
        lines.append(f"\t{op} v0, v1, v2, {src2}")
    else:
        lines.append(f"\tv_mov_b32 v3, {src2 & MASK32:#x}")
        lines.append(f"\t{op} v0, v1, v2, v3")
    asm = "; test_perm_alignbit\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceTests(unittest.TestCase):
    """The references against hand-worked values, so a test failure points at the translator, not the oracle."""

    def test_perm_reference(self):
        self.assertEqual(ref_perm(0x8899AABB, 0x11223344, 0x03020100), 0x11223344)
        self.assertEqual(ref_perm(0x8899AABB, 0x11223344, 0x07060504), 0x8899AABB)
        self.assertEqual(ref_perm(0x8899AABB, 0x11223344, 0x00010203), 0x44332211)
        self.assertEqual(ref_perm(0x8899AABB, 0x11223344, 0x07050301), 0x88AA1133)
        self.assertEqual(ref_perm(0x8899AABB, 0x11223344, 0x0C0D0C0D), 0x00FF00FF)
        # #211's probe values: S0 = 0x8C7A06F5, S1 = 0x13E49B80; selectors 8 and 11 hit set sign bits.
        self.assertEqual(ref_perm(0x8C7A06F5, 0x13E49B80, 0x0B0A0908), 0xFF0000FF)
        self.assertEqual(ref_perm(0x00008000, 0x80000000, 0x0B0A0908), 0x00FFFF00)

    def test_alignbit_reference(self):
        self.assertEqual(ref_alignbit(0x12345678, 0x9ABCDEF0, 0), 0x9ABCDEF0)
        self.assertEqual(ref_alignbit(0x12345678, 0x9ABCDEF0, 8), 0x789ABCDE)
        self.assertEqual(ref_alignbit(0x12345678, 0x9ABCDEF0, 16), 0x56789ABC)
        self.assertEqual(ref_alignbit(0x12345678, 0x9ABCDEF0, 31), 0x2468ACF1)
        self.assertEqual(ref_alignbit(0x12345678, 0x9ABCDEF0, 32), 0x9ABCDEF0)


class PermTests(unittest.TestCase):
    S0, S1 = 0x8899AABB, 0x11223344

    def check(self, sel, src0=None, src1=None):
        src0 = self.S0 if src0 is None else src0
        src1 = self.S1 if src1 is None else src1
        self.assertEqual(run_vop3("v_perm_b32", src0, src1, sel), ref_perm(src0, src1, sel),
                         f"src0={src0:#010x} src1={src1:#010x} sel={sel:#010x}")

    def test_whole_words(self):
        self.check(0x03020100)  # src1 unchanged
        self.check(0x07060504)  # src0 unchanged

    def test_byte_swaps_and_mixes(self):
        for sel in (0x00010203, 0x04050607, 0x07050301, 0x06040200, 0x01000504, 0x03070206):
            with self.subTest(sel=hex(sel)):
                self.check(sel)

    def test_constant_selectors(self):
        for sel in (0x0C0C0C0C, 0x0D0D0D0D, 0xFFFFFFFF, 0x800C7F0D, 0x0C030C07):
            with self.subTest(sel=hex(sel)):
                self.check(sel)

    def test_every_selector_in_every_byte(self):
        # Each defined selector value in each result byte, the other bytes fixed to 0x0c (zero).
        for pos in range(4):
            for s in PERM_SELECTORS:
                sel = (0x0C0C0C0C & ~(0xFF << (8 * pos))) | (s << (8 * pos))
                with self.subTest(pos=pos, s=hex(s)):
                    self.check(sel)

    def test_inline_selector(self):
        # Inline-constant selector 0 picks byte 0 of src1 into every byte.
        self.assertEqual(run_vop3("v_perm_b32", self.S0, self.S1, 0, src2_inline=True), 0x44444444)

    def test_other_data(self):
        for src0, src1 in ((0, MASK32), (MASK32, 0), (0xDEADBEEF, 0x01234567)):
            for sel in (0x03020100, 0x07060504, 0x04000501, 0x0D0C0702):
                with self.subTest(src0=hex(src0), src1=hex(src1), sel=hex(sel)):
                    self.check(sel, src0, src1)


class AlignbitTests(unittest.TestCase):
    S0, S1 = 0x12345678, 0x9ABCDEF0

    def test_every_shift(self):
        for shift in range(32):
            with self.subTest(shift=shift):
                self.assertEqual(run_vop3("v_alignbit_b32", self.S0, self.S1, shift),
                                 ref_alignbit(self.S0, self.S1, shift))

    def test_shift_zero_returns_low_word(self):
        self.assertEqual(run_vop3("v_alignbit_b32", self.S0, self.S1, 0), self.S1)
        self.assertEqual(run_vop3("v_alignbit_b32", self.S0, self.S1, 0, src2_inline=True), self.S1)

    def test_shift_is_masked_to_5_bits(self):
        for shift in (32, 33, 40, 63, 0xFFFFFFE8):
            with self.subTest(shift=hex(shift)):
                self.assertEqual(run_vop3("v_alignbit_b32", self.S0, self.S1, shift),
                                 ref_alignbit(self.S0, self.S1, shift))

    def test_inline_shift(self):
        for shift in (1, 8, 24, 31):
            with self.subTest(shift=shift):
                self.assertEqual(run_vop3("v_alignbit_b32", self.S0, self.S1, shift, src2_inline=True),
                                 ref_alignbit(self.S0, self.S1, shift))

    def test_same_word_is_rotate_right(self):
        # v_alignbit_b32 with src0 == src1 is a 32-bit rotate right, a common compiler idiom.
        x = 0x80000001
        for shift in (1, 4, 31):
            with self.subTest(shift=shift):
                want = ((x >> shift) | (x << (32 - shift))) & MASK32
                self.assertEqual(run_vop3("v_alignbit_b32", x, x, shift), want)


if __name__ == "__main__":
    unittest.main()
