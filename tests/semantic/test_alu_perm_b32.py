"""Semantic interpreter tests for RDNA2 byte permutation (v_perm_b32).

Exercises shader translation via tools/rdna2spirv.py and execution via
tests/semantic/spirv_mini_interp.py against a plain-Python reference oracle.

RDNA2 VALU Specification (VOP3 opcode 836, v_perm_b32):
  v_perm_b32 D, S0, S1, S2
    Source: 64-bit packed container {S0, S1}, where S1 is the low 32 bits (bytes 0..3)
    and S0 is the high 32 bits (bytes 4..7).
    Destination: 32-bit register D, containing 4 bytes (i = 0..3).
    Selector: 32-bit register or inline constant S2, containing 4 byte selectors:
      sel_byte_i = (S2 >> (8 * i)) & 0xFF

  Selector code rules for each result byte i:
    - 0..7: Selects byte (sel_byte) directly from {S0, S1}:
        0..3 pick bytes 0..3 from S1 (bits 0..31)
        4..7 pick bytes 0..3 from S0 (bits 32..63)
    - 8..11: Sign-extension of 16-bit halves:
        Repeats the sign bit (bit 15 or 31) across the entire result byte:
        yielding 0xFF if sign bit is 1, or 0x00 if sign bit is 0.
        8  -> sign bit of S1[15:0]  (bit 15 of S1, i.e. bit 7 of byte 1)
        9  -> sign bit of S1[31:16] (bit 31 of S1, i.e. bit 7 of byte 3)
        10 -> sign bit of S0[15:0]  (bit 15 of S0, i.e. bit 7 of byte 5)
        11 -> sign bit of S0[31:16] (bit 31 of S0, i.e. bit 7 of byte 7)
    - 12 (0x0C): Writes constant 0x00.
    - >= 13 (0x0D..0xFF): Writes constant 0xFF.

  4-Nibble Selector Masks:
    Because each byte's primary selector code (0x0 through 0xF) fits in 4 bits (a nibble),
    common compiler and shader patterns express selector words as 4 packed nibbles:
      make_sel(n0, n1, n2, n3) = (n3 << 24) | (n2 << 16) | (n1 << 8) | n0

Standard library only; zero game data, zero proprietary assets.
"""
from __future__ import annotations

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


def make_sel(b0: int, b1: int, b2: int, b3: int) -> int:
    """Construct 32-bit selector mask from 4 selector byte codes (nibbles 0x0..0xF or bytes 0x00..0xFF)."""
    return ((b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24)) & MASK32


def ref_perm_b32(s0: int, s1: int, sel: int) -> int:
    """RDNA2 v_perm_b32 reference oracle."""
    s0 &= MASK32
    s1 &= MASK32
    sel &= MASK32
    packed = (s0 << 32) | s1
    out = 0
    for i in range(4):
        sb = (sel >> (8 * i)) & 0xFF
        if sb < 8:
            byte_val = (packed >> (8 * sb)) & 0xFF
        elif sb < 12:
            # 8..11: sign bit of 16-bit halfword (sb - 8)
            # halfword 0: s1[0..15] -> bit 15
            # halfword 1: s1[16..31] -> bit 31
            # halfword 2: s0[0..15] -> bit 47
            # halfword 3: s0[16..31] -> bit 63
            sign_bit = (packed >> (16 * (sb - 8) + 15)) & 1
            byte_val = 0xFF if sign_bit else 0x00
        elif sb == 12:
            byte_val = 0x00
        else:
            byte_val = 0xFF
        out |= byte_val << (8 * i)
    return out & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_perm_b32.s")
    dxbc_path = os.path.join(_TMP_DIR.name, "ps_perm_b32.dxbc")
    with open(dxbc_path, "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        try:
            _TMP_DIR.cleanup()
        except Exception:
            pass
        _TMP_DIR = None


def run_rdna_perm(s0: int, s1: int, sel: int, inline_sel: bool = False) -> int:
    """Translate and run `v_perm_b32 v0, v1, v2, v3` (or inline selector), return mrt0."""
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is None:
        setUpModule()

    lines = [
        f"\tv_mov_b32 v1, {s0 & MASK32:#x}",
        f"\tv_mov_b32 v2, {s1 & MASK32:#x}",
    ]
    if inline_sel:
        lines.append(f"\tv_perm_b32 v0, v1, v2, {sel & MASK32:#x}")
    else:
        lines.append(f"\tv_mov_b32 v3, {sel & MASK32:#x}")
        lines.append("\tv_perm_b32 v0, v1, v2, v3")

    asm = "; test_perm_b32\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)

    out = si.run_fragment(spv)
    return out["%out_mrt0"][0]


class ReferenceOracleTests(unittest.TestCase):
    """Test the Python reference oracle against known hand-computed values."""

    S0 = 0xAABBCCDD  # Bytes: 4=0xDD, 5=0xCC, 6=0xBB, 7=0xAA
    S1 = 0x11223344  # Bytes: 0=0x44, 1=0x33, 2=0x22, 3=0x11

    def test_identity_and_endian_swaps(self):
        # Identity on src1 and src0
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(0, 1, 2, 3)), 0x11223344)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(4, 5, 6, 7)), 0xAABBCCDD)
        # Endian byte swap
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(3, 2, 1, 0)), 0x44332211)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(7, 6, 5, 4)), 0xDDCCBBAA)

    def test_constant_zero_and_ff_injections(self):
        # Selector 12 -> 0x00, Selector 13..15 -> 0xFF
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(12, 12, 12, 12)), 0x00000000)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(13, 13, 13, 13)), 0xFFFFFFFF)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(14, 15, 13, 14)), 0xFFFFFFFF)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(12, 13, 12, 13)), 0xFF00FF00)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(0x10, 0x7F, 0x80, 0xFF)), 0xFFFFFFFF)

    def test_sign_extension_rules(self):
        # S1 low half: bit 15 -> selector 8
        # S1 high half: bit 31 -> selector 9
        # S0 low half: bit 15 (bit 47) -> selector 10
        # S0 high half: bit 31 (bit 63) -> selector 11
        # Test all sign bits = 0
        self.assertEqual(ref_perm_b32(0x7FFF7FFF, 0x7FFF7FFF, make_sel(8, 9, 10, 11)), 0x00000000)
        # Test all sign bits = 1
        self.assertEqual(ref_perm_b32(0x80008000, 0x80008000, make_sel(8, 9, 10, 11)), 0xFFFFFFFF)
        # Test individual sign bits
        self.assertEqual(ref_perm_b32(0x00000000, 0x00008000, make_sel(8, 9, 10, 11)), 0x000000FF)
        self.assertEqual(ref_perm_b32(0x00000000, 0x80000000, make_sel(8, 9, 10, 11)), 0x0000FF00)
        self.assertEqual(ref_perm_b32(0x00008000, 0x00000000, make_sel(8, 9, 10, 11)), 0x00FF0000)
        self.assertEqual(ref_perm_b32(0x80000000, 0x00000000, make_sel(8, 9, 10, 11)), 0xFF000000)

    def test_broadcast_and_interleave(self):
        # Broadcast byte 0 of S1 (0x44)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(0, 0, 0, 0)), 0x44444444)
        # Broadcast byte 3 of S0 (byte 7 = 0xAA)
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(7, 7, 7, 7)), 0xAAAAAAAA)
        # Interleave low bytes of S1 and S0: 0, 4, 1, 5 -> 0xCC, 0x33, 0xDD, 0x44
        self.assertEqual(ref_perm_b32(self.S0, self.S1, make_sel(0, 4, 1, 5)), 0xCC33DD44)


class AluPermB32SemanticTests(unittest.TestCase):
    """End-to-end interpreter tests verifying rdna2spirv translation and SPIR-V execution."""

    S0 = 0xAABBCCDD  # Bytes: 4=0xDD, 5=0xCC, 6=0xBB, 7=0xAA
    S1 = 0x11223344  # Bytes: 0=0x44, 1=0x33, 2=0x22, 3=0x11

    def _check(self, s0: int, s1: int, sel: int, inline: bool = False):
        expected = ref_perm_b32(s0, s1, sel)
        actual = run_rdna_perm(s0, s1, sel, inline_sel=inline)
        self.assertEqual(
            actual,
            expected,
            f"Mismatch: s0={s0:#010x} s1={s1:#010x} sel={sel:#010x} (inline={inline}) "
            f"expected={expected:#010x} got={actual:#010x}",
        )

    def test_identity_and_word_moves(self):
        # Identity S1
        self._check(self.S0, self.S1, make_sel(0, 1, 2, 3))
        # Identity S0
        self._check(self.S0, self.S1, make_sel(4, 5, 6, 7))

    def test_endianness_byte_reversal(self):
        # 32-bit endianness reverse of S1
        self._check(self.S0, self.S1, make_sel(3, 2, 1, 0))
        # 32-bit endianness reverse of S0
        self._check(self.S0, self.S1, make_sel(7, 6, 5, 4))
        # 16-bit halfword endianness reverse of S1
        self._check(self.S0, self.S1, make_sel(1, 0, 3, 2))
        # 16-bit halfword swap (words exchanged)
        self._check(self.S0, self.S1, make_sel(2, 3, 0, 1))

    def test_64bit_endian_swap_pair(self):
        # Full 64-bit endian swap across {S0, S1} using two v_perm_b32 instructions:
        # High dword of result: S1 reversed -> make_sel(3, 2, 1, 0)
        # Low dword of result:  S0 reversed -> make_sel(7, 6, 5, 4)
        self._check(self.S0, self.S1, make_sel(7, 6, 5, 4))
        self._check(self.S0, self.S1, make_sel(3, 2, 1, 0))

    def test_four_nibble_mask_packing(self):
        # Explicit test verifying 4-nibble selector masks (n0, n1, n2, n3 in 0x0..0xF):
        # e.g., nibbles (0x0, 0xC, 0x8, 0xD) -> byte0=0x00, byte1=0x0C, byte2=0x08, byte3=0x0D
        sel = make_sel(0x0, 0xC, 0x8, 0xD)
        # With S1=0x00008044 (byte 0=0x44, halfword 0 has sign bit set):
        # byte 0 (code 0) -> S1 byte 0 = 0x44
        # byte 1 (code 12 / 0xC) -> 0x00
        # byte 2 (code 8) -> sign bit of S1[15:0] = 1 -> 0xFF
        # byte 3 (code 13 / 0xD) -> 0xFF
        self._check(0x00000000, 0x00008044, sel)

    def test_byte_broadcast_all_lanes(self):
        # Broadcast each of the 8 input bytes across all 4 result byte positions
        for b in range(8):
            with self.subTest(byte=b):
                self._check(self.S0, self.S1, make_sel(b, b, b, b))

    def test_every_nibble_selector_code_in_every_position(self):
        # Test all 16 selector nibbles (0x0 through 0xF) in each byte position 0..3,
        # with remaining positions filled with selector 12 (0x00 zero injection).
        # We test with a mixed operand pair where sign bits are set so selectors 8..11
        # and 13..15 can be distinctly validated.
        s0 = 0x8C7A06F5
        s1 = 0x13E49B80
        for pos in range(4):
            for code in range(16):
                nibbles = [12, 12, 12, 12]
                nibbles[pos] = code
                sel = make_sel(*nibbles)
                with self.subTest(pos=pos, code=hex(code)):
                    self._check(s0, s1, sel)

    def test_sign_extension_rules_truth_table(self):
        # Selectors 8, 9, 10, 11 inspect the sign bits of 4 halfwords:
        #   8  -> S1[15]
        #   9  -> S1[31]
        #   10 -> S0[15]
        #   11 -> S0[31]
        # The selector 0x0B0A0908 maps (byte0<-8, byte1<-9, byte2<-10, byte3<-11).
        # We test all 16 combinations of (sign_s1_lo, sign_s1_hi, sign_s0_lo, sign_s0_hi).
        sel = make_sel(8, 9, 10, 11)
        for s1_lo in (0, 1):
            for s1_hi in (0, 1):
                for s0_lo in (0, 1):
                    for s0_hi in (0, 1):
                        s1 = (s1_lo << 15) | (s1_hi << 31)
                        s0 = (s0_lo << 15) | (s0_hi << 31)
                        expected_bytes = [
                            0xFF if s1_lo else 0x00,
                            0xFF if s1_hi else 0x00,
                            0xFF if s0_lo else 0x00,
                            0xFF if s0_hi else 0x00,
                        ]
                        expected = (
                            expected_bytes[0]
                            | (expected_bytes[1] << 8)
                            | (expected_bytes[2] << 16)
                            | (expected_bytes[3] << 24)
                        )
                        with self.subTest(s1_lo=s1_lo, s1_hi=s1_hi, s0_lo=s0_lo, s0_hi=s0_hi):
                            actual = run_rdna_perm(s0, s1, sel)
                            self.assertEqual(actual, expected)

    def test_sign_extension_with_realistic_values(self):
        # 16-bit signed boundary values:
        # Positive max: 0x7FFF (sign bit 0 -> 0x00)
        # Negative min: 0x8000 (sign bit 1 -> 0xFF)
        # -1 as i16:    0xFFFF (sign bit 1 -> 0xFF)
        # 0 as i16:     0x0000 (sign bit 0 -> 0x00)
        sel = make_sel(8, 9, 10, 11)
        s1 = 0x80007FFF  # S1 low is +0x7FFF (sign=0), S1 high is -0x8000 (sign=1)
        s0 = 0x0000FFFF  # S0 low is -1 (sign=1), S0 high is 0 (sign=0)
        self._check(s0, s1, sel)

    def test_constant_zero_injection_selector_12(self):
        # Selector 12 (0x0C) injects 0x00 regardless of operands
        self._check(0xFFFFFFFF, 0xFFFFFFFF, make_sel(12, 12, 12, 12))
        self._check(self.S0, self.S1, make_sel(0, 12, 1, 12))
        self._check(self.S0, self.S1, make_sel(12, 4, 12, 5))

    def test_constant_ff_injection_selectors_ge_13(self):
        # Selector >= 13 injects 0xFF regardless of operands
        self._check(0x00000000, 0x00000000, make_sel(13, 14, 15, 13))
        self._check(self.S0, self.S1, make_sel(0, 13, 1, 14))
        # Byte values with top bits set (0x10, 0x7F, 0x80, 0xFF)
        for code in (0x10, 0x20, 0x7F, 0x80, 0xFE, 0xFF):
            with self.subTest(code=hex(code)):
                self._check(self.S0, self.S1, make_sel(code, 0, code, 1))

    def test_inline_constant_selectors(self):
        # Direct immediate constant in place of VGPR selector
        # Identity S1
        self._check(self.S0, self.S1, make_sel(0, 1, 2, 3), inline=True)
        # Identity S0
        self._check(self.S0, self.S1, make_sel(4, 5, 6, 7), inline=True)
        # All zeros
        self._check(self.S0, self.S1, make_sel(12, 12, 12, 12), inline=True)
        # All 0xFF
        self._check(self.S0, self.S1, make_sel(13, 13, 13, 13), inline=True)
        # Sign extension
        self._check(0x80000000, 0x00008000, make_sel(8, 9, 10, 11), inline=True)

    def test_common_shader_idioms(self):
        # 1. 16-bit sign extension to 32-bit (i16 to i32)
        # Low half of S1 sign-extended: byte 0, byte 1, then sign extension from half 0 (sel 8)
        sel_i16_lo = make_sel(0, 1, 8, 8)
        self._check(0, 0x00007FFF, sel_i16_lo)  # +32767 -> 0x00007FFF
        self._check(0, 0x00008000, sel_i16_lo)  # -32768 -> 0xFFFF8000
        self._check(0, 0x0000FFFF, sel_i16_lo)  # -1 -> 0xFFFFFFFF

        # High half of S1 sign-extended: byte 2, byte 3, then sign extension from half 1 (sel 9)
        sel_i16_hi = make_sel(2, 3, 9, 9)
        self._check(0, 0x7FFF0000, sel_i16_hi)  # +32767 -> 0x00007FFF
        self._check(0, 0x80000000, sel_i16_hi)  # -32768 -> 0xFFFF8000

        # 2. 16-bit zero extension to 32-bit (u16 to u32)
        # Low half zero-extended: byte 0, byte 1, sel 12, sel 12
        sel_u16_lo = make_sel(0, 1, 12, 12)
        self._check(0, 0x1234FFFF, sel_u16_lo)  # -> 0x0000FFFF

        # 3. Channel swizzles:
        # Assume S1 is RGBA (R=byte0, G=byte1, B=byte2, A=byte3)
        # Swizzle to BGRA: byte 2, byte 1, byte 0, byte 3
        self._check(0, 0x11223344, make_sel(2, 1, 0, 3))
        # Swizzle to ARGB: byte 3, byte 0, byte 1, byte 2
        self._check(0, 0x11223344, make_sel(3, 0, 1, 2))
        # Swizzle to ABGR: byte 3, byte 2, byte 1, byte 0
        self._check(0, 0x11223344, make_sel(3, 2, 1, 0))

        # 4. Packing two 16-bit halves into one word:
        # Low 16 bits of S1 and low 16 bits of S0: 0, 1, 4, 5
        self._check(0x12345678, 0x9ABCDEF0, make_sel(0, 1, 4, 5))
        # High 16 bits of S1 and high 16 bits of S0: 2, 3, 6, 7
        self._check(0x12345678, 0x9ABCDEF0, make_sel(2, 3, 6, 7))

        # 5. Interleaving bytes (zip):
        self._check(0x12345678, 0x9ABCDEF0, make_sel(0, 4, 1, 5))
        self._check(0x12345678, 0x9ABCDEF0, make_sel(2, 6, 3, 7))

    def test_boundary_operands(self):
        # All zeros
        self._check(0x00000000, 0x00000000, make_sel(0, 1, 2, 3))
        # All ones
        self._check(0xFFFFFFFF, 0xFFFFFFFF, make_sel(0, 1, 2, 3))
        # Alternating bit patterns
        s0 = 0xAAAAAAAA
        s1 = 0x55555555
        self._check(s0, s1, make_sel(0, 4, 1, 5))
        self._check(s0, s1, make_sel(8, 9, 10, 11))

    def test_negative_controls(self):
        # Mutating selector changes output
        sel_good = make_sel(0, 1, 2, 3)
        sel_bad = make_sel(0, 1, 2, 2)
        res_good = run_rdna_perm(self.S0, self.S1, sel_good)
        res_bad = run_rdna_perm(self.S0, self.S1, sel_bad)
        self.assertNotEqual(res_good, res_bad)

        # Mutating active source byte changes output
        res1 = run_rdna_perm(self.S0, 0x11223344, make_sel(0, 1, 2, 3))
        res2 = run_rdna_perm(self.S0, 0x11223355, make_sel(0, 1, 2, 3))
        self.assertNotEqual(res1, res2)

        # Mutating inactive source byte does not change output
        # Selector only picks bytes 0..3 from S1, so modifying S0 must not change result
        res_s0_a = run_rdna_perm(0x11111111, self.S1, make_sel(0, 1, 2, 3))
        res_s0_b = run_rdna_perm(0x22222222, self.S1, make_sel(0, 1, 2, 3))
        self.assertEqual(res_s0_a, res_s0_b)

        # Toggling sign bit flips result between 0x00 and 0xFF for sign selectors
        res_pos = run_rdna_perm(0, 0x00007FFF, make_sel(8, 12, 12, 12))
        res_neg = run_rdna_perm(0, 0x00008000, make_sel(8, 12, 12, 12))
        self.assertEqual(res_pos & 0xFF, 0x00)
        self.assertEqual(res_neg & 0xFF, 0xFF)


if __name__ == "__main__":
    unittest.main()
