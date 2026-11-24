"""Semantic interpreter tests for AMD RDNA2 16-bit integer shift operations.

Instructions covered:
  - v_lshlrev_b16: D.u16 = (S1.u16 << (S0.u16 & 15)) & 0xFFFF
    Logical shift left on 16-bit operands with reversed operand order (S0 is shift count, S1 is value).
  - v_lshrrev_b16: D.u16 = (S1.u16 >> (S0.u16 & 15)) & 0xFFFF
    Logical shift right on 16-bit unsigned operands with zero extension.
  - v_ashrrev_i16: D.i16 = (S1.i16 >> (S0.u16 & 15)) & 0xFFFF
    Arithmetic shift right on 16-bit signed operands with sign extension.

Key architectural rules (AMD RDNA2 ISA reference, docs/design.md):
  1. Reversed operand order: S0 holds the shift amount, S1 holds the value to shift.
  2. Shift amount masking: only the low 4 bits of S0 are used (S0 & 15, range 0..15).
     Shift amounts >= 16 (e.g. 16, 17, 31, 0x1234) wrap modulo 16.
  3. Logical vs arithmetic right shift:
     - v_lshrrev_b16 shifts in zeroes (e.g. 0x8000 >> 1 -> 0x4000; 0x8000 >> 15 -> 0x0001).
     - v_ashrrev_i16 sign-extends bit 15 (e.g. 0x8000 >> 1 -> 0xC000; 0x8000 >> 15 -> 0xFFFF;
       0xFFFF >> n -> 0xFFFF).
  4. op_sel (VOP3 / _e64):
     - op_sel:[s0, s1, dst] selects either the low half (0) or high half (1) for each source.
     - Destination half selection: dst=0 writes to bits [15:0] and preserves old bits [31:16];
       dst=1 writes to bits [31:16] and preserves old bits [15:0].
  5. VOP2 (_e32) vs VOP3 (_e64):
     - VOP2 defines the entire 32-bit register (high 16 bits [31:16] are zeroed).
     - VOP3 (or disassembled with 8-byte encoding comment) is a partial write that preserves the untouched half.

Standard library only: zero game data, no GPU hardware required.
"""
from __future__ import annotations

import gc
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD_DEST = 0xCAFE1234

# 8-byte VOP3 encoding comment (LLVM 21 disassembles VOP3 16-bit ops without _e64 suffix).
VOP3_ENC8 = "; encoding: [0x00,0x00,0x00,0xd5,0x01,0x05,0x02,0x00]"

OPS = ("v_lshlrev_b16", "v_lshrrev_b16", "v_ashrrev_i16")


def u16(x: int) -> int:
    return x & MASK16


def s16(x: int) -> int:
    v = x & MASK16
    return v - 0x10000 if (v & 0x8000) else v


def half16(word: int, hi: int) -> int:
    return ((word >> 16) & MASK16) if hi else (word & MASK16)


def merge16(old: int, res16: int, dst_hi: int = 0, preserve_high: bool = True) -> int:
    res16 &= MASK16
    if dst_hi:
        return ((res16 << 16) | (old & MASK16)) & MASK32
    if preserve_high:
        return ((old & 0xFFFF0000) | res16) & MASK32
    return res16


def ref_shift16(op: str, shift_amount: int, value: int) -> int:
    """ISA reference for RDNA2 16-bit shift operations."""
    sh = shift_amount & 15
    uval = u16(value)
    if op == "v_lshlrev_b16":
        return (uval << sh) & MASK16
    if op == "v_lshrrev_b16":
        return (uval >> sh) & MASK16
    if op == "v_ashrrev_i16":
        sval = s16(uval)
        return (sval >> sh) & MASK16
    raise ValueError(f"unknown op: {op}")


def ref_shift_word(op: str, s0: int, s1: int, old: int = OLD_DEST,
                   sel=(0, 0, 0), preserve_high: bool = True) -> int:
    """Full 32-bit destination word reference accounting for op_sel and half preservation."""
    sh = half16(s0, sel[0])
    val = half16(s1, sel[1])
    res16 = ref_shift16(op, sh, val)
    return merge16(old, res16, dst_hi=sel[2], preserve_high=preserve_high)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_alu_shift16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_shift16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def translate_lines(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_shift16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_mrt0(lines, regs=("v0", "v0", "v0", "v0")) -> list[int]:
    spv = translate_lines(list(lines) + [f"exp mrt0 {', '.join(regs)} done vm"])
    return list(si.run_fragment(spv)["%out_mrt0"])


def run_one(line: str, v1: int, v2: int, old: int = OLD_DEST) -> int:
    return run_mrt0([
        f"v_mov_b32 v1, {v1 & MASK32:#x}",
        f"v_mov_b32 v2, {v2 & MASK32:#x}",
        f"v_mov_b32 v0, {old & MASK32:#x}",
        line,
    ])[0]


class ReferenceOracleTests(unittest.TestCase):
    """Verify the Python reference model against known hand-calculated constants."""

    def test_hand_calculated_shifts(self):
        cases = [
            # v_lshlrev_b16 (S0=shift, S1=value)
            ("v_lshlrev_b16", 0, 0x1234, 0x1234),
            ("v_lshlrev_b16", 1, 0x1234, 0x2468),
            ("v_lshlrev_b16", 4, 0x0005, 0x0050),
            ("v_lshlrev_b16", 15, 0x0001, 0x8000),
            ("v_lshlrev_b16", 15, 0x0003, 0x8000),  # bit 1 drops off
            ("v_lshlrev_b16", 16, 0x1234, 0x1234),  # 16 & 15 == 0 -> no shift
            ("v_lshlrev_b16", 17, 0x1234, 0x2468),  # 17 & 15 == 1 -> shift by 1
            ("v_lshlrev_b16", 31, 0x0001, 0x8000),  # 31 & 15 == 15

            # v_lshrrev_b16 (zero extension)
            ("v_lshrrev_b16", 0, 0x8000, 0x8000),
            ("v_lshrrev_b16", 1, 0x8000, 0x4000),
            ("v_lshrrev_b16", 4, 0x0050, 0x0005),
            ("v_lshrrev_b16", 15, 0x8000, 0x0001),
            ("v_lshrrev_b16", 15, 0xFFFF, 0x0001),
            ("v_lshrrev_b16", 16, 0x8000, 0x8000),  # 16 & 15 == 0

            # v_ashrrev_i16 (sign extension)
            ("v_ashrrev_i16", 0, 0x8000, 0x8000),
            ("v_ashrrev_i16", 1, 0x8000, 0xC000),
            ("v_ashrrev_i16", 2, 0x8000, 0xE000),
            ("v_ashrrev_i16", 15, 0x8000, 0xFFFF),
            ("v_ashrrev_i16", 4, 0x7FFF, 0x07FF),   # positive sign preserved
            ("v_ashrrev_i16", 1, 0xFFFF, 0xFFFF),   # -1 >> 1 == -1
            ("v_ashrrev_i16", 15, 0xFFFF, 0xFFFF),  # -1 >> 15 == -1
            ("v_ashrrev_i16", 16, 0x8000, 0x8000),  # 16 & 15 == 0
            ("v_ashrrev_i16", 17, 0x8000, 0xC000),  # 17 & 15 == 1
        ]
        for op, sh, val, expected in cases:
            with self.subTest(op=op, sh=sh, val=hex(val)):
                self.assertEqual(ref_shift16(op, sh, val), expected)


class ShiftBoundaryTests(unittest.TestCase):
    """Exhaustive boundary testing across all 16 shift counts and key 16-bit values."""

    BOUNDARY_VALUES = (
        0x0000, 0x0001, 0x0002, 0x007F, 0x0080, 0x00FF,
        0x0100, 0x5555, 0x7FFF, 0x8000, 0x8001, 0xAAAA,
        0xFF00, 0xFFFE, 0xFFFF,
    )

    def test_all_shift_counts_e64(self):
        """Test shift counts 0..15 on VOP3 form for all three instructions."""
        for op in OPS:
            mnemonic = f"{op}_e64"
            for sh in range(16):
                for val in self.BOUNDARY_VALUES:
                    # Place non-zero noise in the high halves of inputs
                    v1 = (0xBEEF << 16) | sh
                    v2 = (0xDEAD << 16) | val
                    line = f"{mnemonic} v0, v1, v2"
                    got = run_one(line, v1, v2, old=OLD_DEST)
                    want = ref_shift_word(op, v1, v2, old=OLD_DEST, preserve_high=True)
                    self.assertEqual(
                        got, want,
                        f"{op} sh={sh} val={val:#06x}: got {got:#010x}, want {want:#010x}"
                    )

    def test_shift_amount_masking(self):
        """Test shift counts >= 16 to confirm masking with 15 (bits [3:0])."""
        high_shift_counts = (16, 17, 20, 31, 32, 45, 63, 64, 127, 255, 0x1234)
        test_val = 0x8000
        for op in OPS:
            mnemonic = f"{op}_e64"
            for sh in high_shift_counts:
                v1 = (0x1111 << 16) | sh
                v2 = (0x2222 << 16) | test_val
                line = f"{mnemonic} v0, v1, v2"
                got = run_one(line, v1, v2, old=OLD_DEST)
                want = ref_shift_word(op, v1, v2, old=OLD_DEST, preserve_high=True)
                self.assertEqual(
                    got, want,
                    f"{op} large_sh={sh} (sh&15={sh&15}): got {got:#010x}, want {want:#010x}"
                )

    def test_arithmetic_vs_logical_contrast(self):
        """Direct contrast: v_lshrrev_b16 vs v_ashrrev_i16 on negative 16-bit values."""
        negative_val = 0x8000
        for sh in (1, 2, 4, 8, 14, 15):
            v1 = sh
            v2 = negative_val
            got_lshr = run_one(f"v_lshrrev_b16_e64 v0, v1, v2", v1, v2, old=OLD_DEST)
            got_ashr = run_one(f"v_ashrrev_i16_e64 v0, v1, v2", v1, v2, old=OLD_DEST)

            want_lshr = ref_shift_word("v_lshrrev_b16", v1, v2, old=OLD_DEST)
            want_ashr = ref_shift_word("v_ashrrev_i16", v1, v2, old=OLD_DEST)

            self.assertEqual(got_lshr, want_lshr)
            self.assertEqual(got_ashr, want_ashr)
            # Arithmetic right shift must have sign bit replicated, logical right shift must not
            self.assertNotEqual(got_lshr & MASK16, got_ashr & MASK16)
            self.assertEqual(got_ashr & 0x8000, 0x8000, "ashr must maintain sign bit 15")
            self.assertEqual(got_lshr & 0x8000, 0x0000, "lshr must clear bit 15 on sh >= 1")


class OpSelAndPreservationTests(unittest.TestCase):
    """Test all op_sel combinations and high/low half destination preservation."""

    # Distinct words: low half and high half differ distinctly
    WORD_A = 0x12348001  # hi: 0x1234, lo: 0x8001
    WORD_B = 0xABCD0004  # hi: 0xABCD, lo: 0x0004

    def test_all_op_sel_combinations(self):
        """Exhaustively verify all 8 combinations of op_sel:[s0, s1, dst]."""
        for op in OPS:
            for s0_hi in (0, 1):
                for s1_hi in (0, 1):
                    for dst_hi in (0, 1):
                        sel = (s0_hi, s1_hi, dst_hi)
                        line = f"{op}_e64 v0, v1, v2 op_sel:[{s0_hi},{s1_hi},{dst_hi}]"
                        got = run_one(line, self.WORD_A, self.WORD_B, old=OLD_DEST)
                        want = ref_shift_word(op, self.WORD_A, self.WORD_B, old=OLD_DEST,
                                              sel=sel, preserve_high=True)
                        self.assertEqual(
                            got, want,
                            f"{op} op_sel:{list(sel)} got {got:#010x}, want {want:#010x}"
                        )

    def test_destination_high_half_written(self):
        """Verify op_sel:[0,0,1] writes to destination bits [31:16] and preserves [15:0]."""
        old = 0xCAFE1234
        for op in OPS:
            line = f"{op}_e64 v0, v1, v2 op_sel:[0,0,1]"
            got = run_one(line, 4, 0x0005, old=old)
            # 5 << 4 = 0x50 in high half, low half 0x1234 preserved
            expected_res16 = ref_shift16(op, 4, 5)
            want = (expected_res16 << 16) | 0x1234
            self.assertEqual(got, want)
            self.assertEqual(got & 0xFFFF, 0x1234, "low half must be preserved")

    def test_vop3_encoding_comment_preservation(self):
        """Verify LLVM disassembly format (bare mnemonic with 8-byte encoding comment)."""
        for op in OPS:
            # LLVM disassembles VOP3 16-bit integer ops as bare mnemonic + encoding comment
            line = f"{op} v0, v1, v2 {VOP3_ENC8}"
            got = run_one(line, 3, 0x0008, old=OLD_DEST)
            want = ref_shift_word(op, 3, 0x0008, old=OLD_DEST, preserve_high=True)
            self.assertEqual(got, want)
            self.assertEqual(got & 0xFFFF0000, OLD_DEST & 0xFFFF0000, "high half must be preserved")


class Vop2ZeroHighTests(unittest.TestCase):
    """Verify VOP2 (_e32) forms zero out destination bits [31:16]."""

    def test_vop2_zeroes_high_half(self):
        for op in OPS:
            # Bare mnemonic without encoding comment models VOP2
            line = f"{op} v0, v1, v2"
            got = run_one(line, 2, 0x0007, old=OLD_DEST)
            want = ref_shift_word(op, 2, 0x0007, old=OLD_DEST, preserve_high=False)
            self.assertEqual(
                got, want,
                f"VOP2 {op} failed: got {got:#010x}, want {want:#010x}"
            )
            self.assertEqual(got >> 16, 0, f"VOP2 {op} must zero destination high half")


class OperandFormsTests(unittest.TestCase):
    """Test scalar SGPRs, inline integer constants, and literal operands."""

    def test_sgpr_shift_amount(self):
        """Scalar SGPR s0 as the shift amount operand."""
        for op in OPS:
            spv = translate_lines([
                "s_mov_b32 s0, 3",
                "v_mov_b32 v1, 0x0004",
                "v_mov_b32 v0, 0x12345678",
                f"{op}_e64 v0, s0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
            ])
            got = list(si.run_fragment(spv)["%out_mrt0"])[0]
            want = ref_shift_word(op, 3, 4, old=0x12345678, preserve_high=True)
            self.assertEqual(got, want, f"SGPR shift failed for {op}")

    def test_inline_constants(self):
        """Inline constants: 0, 1, 4, 15, 16."""
        for op in OPS:
            for c in (0, 1, 4, 15, 16):
                spv = translate_lines([
                    "v_mov_b32 v1, 0x0080",
                    "v_mov_b32 v0, 0xCAFE0000",
                    f"{op}_e64 v0, {c}, v1",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                ])
                got = list(si.run_fragment(spv)["%out_mrt0"])[0]
                want = ref_shift_word(op, c, 0x0080, old=0xCAFE0000, preserve_high=True)
                self.assertEqual(got, want, f"constant {c} failed for {op}")


class RandomizedFuzzSweepTests(unittest.TestCase):
    """Deterministic pseudo-random sweep verifying consistency against reference."""

    def test_randomized_sweep_50_iterations(self):
        rng = random.Random(0x16541F7)  # deterministic seed
        for _ in range(50):
            op = rng.choice(OPS)
            s0 = rng.randint(0, 0xFFFFFFFF)
            s1 = rng.randint(0, 0xFFFFFFFF)
            old = rng.randint(0, 0xFFFFFFFF)
            sel = (rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 1))

            line = f"{op}_e64 v0, v1, v2 op_sel:[{sel[0]},{sel[1]},{sel[2]}]"
            got = run_one(line, s0, s1, old=old)
            want = ref_shift_word(op, s0, s1, old=old, sel=sel, preserve_high=True)
            self.assertEqual(
                got, want,
                f"Random sweep failed for {op} with s0={s0:#x}, s1={s1:#x}, sel={sel}"
            )


if __name__ == "__main__":
    unittest.main()
