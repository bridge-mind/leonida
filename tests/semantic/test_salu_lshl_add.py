"""Semantic interpreter tests for RDNA2 s_lshl1_add_u32 ... s_lshl4_add_u32.

RDNA2 scalar shift-and-add instructions:
  - s_lshl1_add_u32 D, S0, S1: D = ((S0 << 1) + S1) & 0xFFFFFFFF
  - s_lshl2_add_u32 D, S0, S1: D = ((S0 << 2) + S1) & 0xFFFFFFFF
  - s_lshl3_add_u32 D, S0, S1: D = ((S0 << 3) + S1) & 0xFFFFFFFF
  - s_lshl4_add_u32 D, S0, S1: D = ((S0 << 4) + S1) & 0xFFFFFFFF

Semantics and Condition Code (SCC):
  - D is the lower 32 bits of the scaled addition: ((S0 << N) + S1) mod 2^32.
  - SCC represents unsigned overflow of the sum taken in 64 bits (RDNA ISA pseudocode):
      tmp = (64'U(S0) << N) + 64'U(S1)
      SCC = 1 if tmp >= 2^32 else 0
    Bits shifted out of S0 (upper N bits of S0) count as carry just like an addition
    carry out of the 32-bit addition.
  - Pre-existing SCC state is ignored and unconditionally overwritten.

Each test case verifies mathematical semantics against pure-Python reference models
and executes synthetic shaders with tools/rdna2spirv.py in tests/semantic/spirv_mini_interp.py.
"""
from __future__ import annotations

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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
TWO_POW_32 = 1 << 32
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [
    0,
    1,
    0x0FFFFFFF,
    0x10000000,
    0x1FFFFFFF,
    0x3FFFFFFF,
    0x40000000,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x12345678,
    0xF0000000,
]


def ref_s_lshl_add_u32(n: int, s0: int, s1: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 s_lshlN_add_u32 (N in {1, 2, 3, 4}).

    Computes 64-bit unsigned temporary sum:
      tmp = (u64(s0) << n) + u64(s1)
      result = tmp & 0xFFFFFFFF
      scc = 1 if tmp >= 2^32 else 0

    Returns:
      (result, scc): 32-bit unsigned result and SCC overflow flag.
    """
    if n not in (1, 2, 3, 4):
        raise ValueError(f"Shift amount N must be in {1, 2, 3, 4}, got {n}")
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    tmp = (s0_u32 << n) + s1_u32
    res = tmp & MASK32
    scc = 1 if (tmp >= TWO_POW_32) else 0

    # Cross-check against hardware decomposition:
    # shifted_out_bits | addition_carry == scc
    out_bits = s0_u32 >> (32 - n)
    sh_low = (s0_u32 << n) & MASK32
    carry = 1 if (sh_low + s1_u32 > MASK32) else 0
    assert (1 if (out_bits != 0 or carry != 0) else 0) == scc

    return res, scc


def ref_s_lshl1_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl1_add_u32: D = ((S0 << 1) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(1, s0, s1)


def ref_s_lshl2_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl2_add_u32: D = ((S0 << 2) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(2, s0, s1)


def ref_s_lshl3_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl3_add_u32: D = ((S0 << 3) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(3, s0, s1)


def ref_s_lshl4_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl4_add_u32: D = ((S0 << 4) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(4, s0, s1)


def ref(n: int, x: int, y: int) -> tuple[int, int]:
    """Backwards-compatible reference wrapper."""
    return ref_s_lshl_add_u32(n, x, y)


class ScalarAluInterpreter:
    """Minimal deterministic software emulator for scalar ALU state."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.scc: int = 0

    def reset(self) -> None:
        for i in range(len(self.sgprs)):
            self.sgprs[i] = 0
        self.scc = 0

    def get_sgpr(self, reg: int | str) -> int:
        idx = int(reg[1:]) if isinstance(reg, str) and reg.startswith("s") else int(reg)
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, reg: int | str, val: int) -> None:
        idx = int(reg[1:]) if isinstance(reg, str) and reg.startswith("s") else int(reg)
        self.sgprs[idx] = val & MASK32

    def parse_operand(self, op: str | int) -> int:
        if isinstance(op, int):
            return op & MASK32
        op = op.strip()
        if op.startswith("s"):
            return self.get_sgpr(op)
        if op.startswith("0x") or op.startswith("0X"):
            return int(op, 16) & MASK32
        return int(op) & MASK32

    def step(self, op: str, *args: str | int) -> tuple[int, int]:
        """Execute a single SALU instruction and update register state."""
        if op == "s_mov_b32":
            dst, src = args[0], args[1]
            val = self.parse_operand(src)
            self.set_sgpr(dst, val)
            return val, self.scc

        if op == "s_cmp_eq_u32":
            s0, s1 = self.parse_operand(args[0]), self.parse_operand(args[1])
            self.scc = 1 if s0 == s1 else 0
            return 0, self.scc

        if op == "s_cselect_b32":
            dst, s0, s1 = args[0], self.parse_operand(args[1]), self.parse_operand(args[2])
            val = s0 if self.scc != 0 else s1
            self.set_sgpr(dst, val)
            return val, self.scc

        if op == "s_addc_u32":
            dst, s0, s1 = args[0], self.parse_operand(args[1]), self.parse_operand(args[2])
            tot = s0 + s1 + self.scc
            res = tot & MASK32
            self.scc = 1 if tot > MASK32 else 0
            self.set_sgpr(dst, res)
            return res, self.scc

        if op in ("s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32"):
            n = int(op[6])
            dst = args[0]
            s0 = self.parse_operand(args[1])
            s1 = self.parse_operand(args[2])
            res, scc = ref_s_lshl_add_u32(n, s0, s1)
            self.set_sgpr(dst, res)
            self.scc = scc
            return res, self.scc

        raise ValueError(f"Unsupported instruction in ScalarAluInterpreter: {op}")


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_lshl_add.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_lshl_add.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, export v0 and v1, return those two words."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_salu_lshl_add\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def run_op(n: int, x: int, y: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = s_lshlN_add_u32(s2, s3); return (s0, SCC after)."""
    return run_lines([
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s3, {y:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"s_lshl{n}_add_u32 s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ])


class ReferenceOracleTests(unittest.TestCase):
    """Validation of pure-Python reference oracles and mathematical invariants."""

    def test_reference_oracle_zero_identity(self):
        for n in (1, 2, 3, 4):
            with self.subTest(n=n):
                res, scc = ref_s_lshl_add_u32(n, 0, 0)
                self.assertEqual(res, 0)
                self.assertEqual(scc, 0)

    def test_reference_oracle_address_scaling(self):
        base_addr = 0x2000
        index = 25
        # Scale 1: 2 bytes (halfword) -> 2000 + 50 = 2050
        self.assertEqual(ref_s_lshl1_add_u32(index, base_addr), (base_addr + 50, 0))
        # Scale 2: 4 bytes (dword) -> 2000 + 100 = 2100
        self.assertEqual(ref_s_lshl2_add_u32(index, base_addr), (base_addr + 100, 0))
        # Scale 3: 8 bytes (qword) -> 2000 + 200 = 2200
        self.assertEqual(ref_s_lshl3_add_u32(index, base_addr), (base_addr + 200, 0))
        # Scale 4: 16 bytes (dqword) -> 2000 + 400 = 2400
        self.assertEqual(ref_s_lshl4_add_u32(index, base_addr), (base_addr + 400, 0))

    def test_reference_oracle_shifted_out_decomposition(self):
        # Bit shifted out of S0 alone sets SCC even with zero addend
        self.assertEqual(ref_s_lshl1_add_u32(0x80000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl2_add_u32(0x40000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl3_add_u32(0x20000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl4_add_u32(0x10000000, 0), (0, 1))

    def test_reference_oracle_add_carry_alone(self):
        # Addition overflow alone sets SCC even without shifted-out bits
        self.assertEqual(ref_s_lshl1_add_u32(1, 0xFFFFFFFE), (0, 1))
        self.assertEqual(ref_s_lshl2_add_u32(1, 0xFFFFFFFC), (0, 1))
        self.assertEqual(ref_s_lshl3_add_u32(1, 0xFFFFFFF8), (0, 1))
        self.assertEqual(ref_s_lshl4_add_u32(1, 0xFFFFFFF0), (0, 1))

    def test_reference_oracle_both_shift_and_add_carry(self):
        # Both shifted out bits and addition carry present
        res1, scc1 = ref_s_lshl1_add_u32(0x80000001, 0xFFFFFFFF)
        self.assertEqual(res1, 1)
        self.assertEqual(scc1, 1)

        res4, scc4 = ref_s_lshl4_add_u32(0xF0000001, 0xFFFFFFFF)
        self.assertEqual(res4, 0x0000000F)
        self.assertEqual(scc4, 1)

    def test_reference_oracle_invalid_shift_raises(self):
        with self.assertRaises(ValueError):
            ref_s_lshl_add_u32(0, 10, 20)
        with self.assertRaises(ValueError):
            ref_s_lshl_add_u32(5, 10, 20)

    def test_scalar_interpreter_basic_pipeline(self):
        interp = ScalarAluInterpreter()
        interp.step("s_mov_b32", "s2", 0x10)
        interp.step("s_mov_b32", "s3", 0x20)
        res, scc = interp.step("s_lshl2_add_u32", "s0", "s2", "s3")
        self.assertEqual(res, 0x10 * 4 + 0x20)
        self.assertEqual(scc, 0)
        self.assertEqual(interp.get_sgpr("s0"), 0x60)
        self.assertEqual(interp.scc, 0)


class TestSccCarryAndWrapping(unittest.TestCase):
    """Validation of SCC carry generation, overflow mechanisms, and 32-bit modulo wrapping."""

    def test_scc_from_shifted_out_bits_exact_boundaries(self):
        # High bits shifted out of 32-bit range set SCC even when 32-bit sum has no carry.
        # Boundary cases:
        # N=1: bit 31 (0x80000000) shifts out -> SCC=1; bit 30 (0x40000000) does not -> SCC=0.
        # N=2: bit 30 (0x40000000) shifts out -> SCC=1; bit 29 (0x20000000) does not -> SCC=0.
        # N=3: bit 29 (0x20000000) shifts out -> SCC=1; bit 28 (0x10000000) does not -> SCC=0.
        # N=4: bit 28 (0x10000000) shifts out -> SCC=1; bit 27 (0x08000000) does not -> SCC=0.
        for n in (1, 2, 3, 4):
            boundary_bit = 1 << (32 - n)
            sub_boundary_bit = boundary_bit >> 1

            # Threshold bit shifted out:
            want_ovf = ref_s_lshl_add_u32(n, boundary_bit, 0)
            self.assertEqual(want_ovf, (0, 1))
            got_ovf = run_op(n, boundary_bit, 0, scc_in=0)
            self.assertEqual(got_ovf, want_ovf)

            # One bit below threshold (no bits shifted out, so SCC=0):
            want_no_ovf = ref_s_lshl_add_u32(n, sub_boundary_bit, 0)
            self.assertEqual(want_no_ovf, (0x80000000, 0))
            got_no_ovf = run_op(n, sub_boundary_bit, 0, scc_in=1)
            self.assertEqual(got_no_ovf, want_no_ovf)

            # Max value that does not shift out any bits:
            max_no_shift = (1 << (32 - n)) - 1
            want_max_no_shift = ref_s_lshl_add_u32(n, max_no_shift, 0)
            self.assertEqual(want_max_no_shift[1], 0)
            got_max_no_shift = run_op(n, max_no_shift, 0, scc_in=1)
            self.assertEqual(got_max_no_shift, want_max_no_shift)

    def test_scc_from_addition_overflow_exact_boundaries(self):
        # Shifted value fits completely in 32 bits, but addition with S1 overflows 2^32.
        for n in (1, 2, 3, 4):
            sh_val = 1 << n  # s0 = 1, so (s0 << n) = 1 << n

            # Exact wrap to 0: s1 = 2^32 - (1 << n) -> sum is exactly 2^32
            s1_wrap = (1 << 32) - sh_val
            want_wrap = ref_s_lshl_add_u32(n, 1, s1_wrap)
            self.assertEqual(want_wrap, (0, 1))
            got_wrap = run_op(n, 1, s1_wrap, scc_in=0)
            self.assertEqual(got_wrap, want_wrap)

            # Maximum sum without carry: s1 = (2^32 - 1) - (1 << n) -> sum is 0xFFFFFFFF
            s1_no_wrap = ((1 << 32) - 1) - sh_val
            want_no_wrap = ref_s_lshl_add_u32(n, 1, s1_no_wrap)
            self.assertEqual(want_no_wrap, (MASK32, 0))
            got_no_wrap = run_op(n, 1, s1_no_wrap, scc_in=1)
            self.assertEqual(got_no_wrap, want_no_wrap)

    def test_both_shift_out_and_addition_overflow(self):
        # Both mechanisms fire simultaneously (S0 shifts out bits AND addition carries)
        for n in (1, 2, 3, 4):
            s0 = MASK32
            s1 = MASK32
            want = ref_s_lshl_add_u32(n, s0, s1)
            # tmp = (MASK32 << n) + MASK32 = MASK32 * ((1 << n) + 1)
            expected_res = ((MASK32 << n) + MASK32) & MASK32
            self.assertEqual(want, (expected_res, 1))
            got = run_op(n, s0, s1, scc_in=0)
            self.assertEqual(got, want)

    def test_scc_input_independence(self):
        # SCC must be completely overwritten regardless of whether incoming SCC is 0 or 1.
        test_cases = [
            (1, 0x80000000, 0),         # SCC becomes 1 (shift-out)
            (1, 0x1000, 0x2000),         # SCC becomes 0 (no carry)
            (2, 0x40000000, 0),         # SCC becomes 1
            (2, 1, 0xFFFFFFFC),         # SCC becomes 1 (add overflow)
            (3, 0x20000000, 0),         # SCC becomes 1
            (3, 0x10, 0x20),            # SCC becomes 0
            (4, 0x10000000, 0),         # SCC becomes 1
            (4, 0x01, 0x02),            # SCC becomes 0
        ]
        for n, s0, s1 in test_cases:
            want = ref_s_lshl_add_u32(n, s0, s1)
            with self.subTest(n=n, s0=hex(s0), s1=hex(s1)):
                # Run with SCC=0 before
                got0 = run_op(n, s0, s1, scc_in=0)
                self.assertEqual(got0, want)
                # Run with SCC=1 before
                got1 = run_op(n, s0, s1, scc_in=1)
                self.assertEqual(got1, want)

    def test_scc_chaining_s_addc_all_scales(self):
        # Multi-word 64-bit pointer/index arithmetic across all four shift scales:
        # scale index into base pointer with s_lshlN_add_u32, feed carry into high dword via s_addc_u32.
        for n in (1, 2, 3, 4):
            idx = 1 << (32 - n)  # generates carry via shift-out
            base_lo = 0x00000100
            base_hi = 0x00000005
            lines = [
                f"s_mov_b32 s2, {idx:#x}",
                f"s_mov_b32 s3, {base_lo:#x}",
                f"s_mov_b32 s5, {base_hi:#x}",
                f"s_lshl{n}_add_u32 s0, s2, s3",
                "s_addc_u32 s1, s5, 0",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
            ]
            got = run_lines(lines)
            expected_lo = base_lo  # idx << n wraps low bits to 0
            expected_hi = base_hi + 1  # carry added
            self.assertEqual(got, (expected_lo, expected_hi))

    def test_deterministic_differential_fuzzing(self):
        # 500 pure-Python random vectors cross-checked against 64-bit mathematical definition
        rng = random.Random(0x42A5C)
        for _ in range(500):
            n = rng.choice((1, 2, 3, 4))
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            res, scc = ref_s_lshl_add_u32(n, s0, s1)
            tmp = (s0 << n) + s1
            self.assertEqual(res, tmp & MASK32)
            self.assertEqual(scc, int(tmp >= TWO_POW_32))

        # 20 diverse random vectors translated and executed in SPIR-V mini interpreter
        for _ in range(20):
            n = rng.choice((1, 2, 3, 4))
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            scc_in = rng.getrandbits(1)
            want = ref_s_lshl_add_u32(n, s0, s1)
            with self.subTest(n=n, s0=hex(s0), s1=hex(s1), scc_in=scc_in):
                got = run_op(n, s0, s1, scc_in)
                self.assertEqual(got, want)


class TestShiftBoundaryAndAliasing(unittest.TestCase):
    """Validation of register aliasing, literal constants, and extreme bit boundary patterns."""

    def test_register_aliasing_dst_src0(self):
        # Destination register aliases source 0 (D == S0): s_lshlN_add_u32 s0, s0, s1
        for n in (1, 2, 3, 4):
            x = 0x1234
            y = 0x5678
            lines = [
                f"s_mov_b32 s0, {x:#x}",
                f"s_mov_b32 s1, {y:#x}",
                f"s_lshl{n}_add_u32 s0, s0, s1",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
            ]
            got_d, got_s1 = run_lines(lines)
            expected_d, _ = ref_s_lshl_add_u32(n, x, y)
            self.assertEqual(got_d, expected_d)
            self.assertEqual(got_s1, y)  # S1 must remain unmodified

    def test_register_aliasing_dst_src1(self):
        # Destination register aliases source 1 (D == S1): s_lshlN_add_u32 s0, s1, s0
        for n in (1, 2, 3, 4):
            x = 0x2000
            y = 0x0042
            lines = [
                f"s_mov_b32 s1, {x:#x}",
                f"s_mov_b32 s0, {y:#x}",
                f"s_lshl{n}_add_u32 s0, s1, s0",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
            ]
            got_d, got_s1 = run_lines(lines)
            expected_d, _ = ref_s_lshl_add_u32(n, x, y)
            self.assertEqual(got_d, expected_d)
            self.assertEqual(got_s1, x)  # S1 must remain unmodified

    def test_register_aliasing_src0_src1(self):
        # Source 0 aliases source 1 (S0 == S1): s_lshlN_add_u32 s0, s1, s1
        # Mathematically computes s1 * ((1 << n) + 1):
        # N=1: * 3; N=2: * 5; N=3: * 9; N=4: * 17
        factors = {1: 3, 2: 5, 3: 9, 4: 17}
        for n in (1, 2, 3, 4):
            val = 0x1111
            lines = [
                f"s_mov_b32 s1, {val:#x}",
                f"s_lshl{n}_add_u32 s0, s1, s1",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
            ]
            got_d, got_s1 = run_lines(lines)
            expected_d = (val * factors[n]) & MASK32
            self.assertEqual(got_d, expected_d)
            self.assertEqual(got_s1, val)

    def test_register_aliasing_three_way(self):
        # Full three-way aliasing (D == S0 == S1): s_lshlN_add_u32 s0, s0, s0
        factors = {1: 3, 2: 5, 3: 9, 4: 17}
        for n in (1, 2, 3, 4):
            val = 0x01020304
            lines = [
                f"s_mov_b32 s0, {val:#x}",
                f"s_lshl{n}_add_u32 s0, s0, s0",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, 0",
            ]
            got_d, _ = run_lines(lines)
            expected_d = (val * factors[n]) & MASK32
            self.assertEqual(got_d, expected_d)

    def test_inline_constants_and_literals(self):
        # SALU supports immediate/literal constants in S0 and S1
        # Literal in S0:
        got = run_lines([
            "s_mov_b32 s1, 100",
            "s_lshl1_add_u32 s0, 7, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ])
        self.assertEqual(got[0], 7 * 2 + 100)

        # Literal in S1:
        got2 = run_lines([
            "s_mov_b32 s0, 50",
            "s_lshl2_add_u32 s0, s0, 42",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ])
        self.assertEqual(got2[0], 50 * 4 + 42)

        # Negative inline constant (-1):
        got3 = run_lines([
            "s_mov_b32 s1, 5",
            "s_lshl3_add_u32 s0, -1, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ])
        expected3, _ = ref_s_lshl_add_u32(3, -1, 5)
        self.assertEqual(got3[0], expected3)

    def test_walking_bit_patterns(self):
        # Walk a 1-bit through all 32 positions of S0 across all four shift scales
        for n in (1, 2, 3, 4):
            for k in range(32):
                s0 = 1 << k
                res, scc = ref_s_lshl_add_u32(n, s0, 0)
                expected_shifted = (s0 << n) & MASK32
                expected_scc = 1 if (k >= (32 - n)) else 0
                self.assertEqual(res, expected_shifted)
                self.assertEqual(scc, expected_scc)

    def test_alternating_bit_patterns(self):
        for n in (1, 2, 3, 4):
            # 0xAAAAAAAA (1010...) and 0x55555555 (0101...)
            s0 = 0xAAAAAAAA
            s1 = 0x55555555
            want = ref_s_lshl_add_u32(n, s0, s1)
            got = run_op(n, s0, s1, scc_in=0)
            self.assertEqual(got, want)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated arithmetic or missing overflow checks fail assertions."""

    def test_bitwise_or_mutation_is_detected(self):
        # Mutating (shifted + S1) to (shifted | S1) must be detected when bits collide
        s0 = 3
        s1 = 4
        # For N=2: (3 << 2) = 12 (0b1100). S1 = 4 (0b0100).
        # ADD: 12 + 4 = 16 (0b10000)
        # OR:  12 | 4 = 12 (0b01100)
        correct_res, _ = ref_s_lshl_add_u32(2, s0, s1)
        mutated_or = ((s0 << 2) | s1) & MASK32
        self.assertEqual(correct_res, 16)
        self.assertEqual(mutated_or, 12)
        self.assertNotEqual(correct_res, mutated_or)

    def test_missing_shift_out_carry_mutation_is_detected(self):
        # If SCC only checks (shifted + s1) > 0xFFFFFFFF and ignores shifted-out bits:
        s0 = 0x80000000
        s1 = 0
        _, correct_scc = ref_s_lshl_add_u32(1, s0, s1)
        # Mutated: only checks 32-bit addition carry
        shifted_low = (s0 << 1) & MASK32
        mutated_scc = 1 if (shifted_low + s1 > MASK32) else 0
        self.assertEqual(correct_scc, 1)
        self.assertEqual(mutated_scc, 0)
        self.assertNotEqual(correct_scc, mutated_scc)

    def test_missing_addition_carry_mutation_is_detected(self):
        # If SCC only checks shifted-out bits and ignores addition carry:
        s0 = 1
        s1 = 0xFFFFFFFF
        _, correct_scc = ref_s_lshl_add_u32(1, s0, s1)
        # Mutated: only checks shifted-out bits
        shifted_out = (s0 & MASK32) >> 31
        mutated_scc = 1 if (shifted_out != 0) else 0
        self.assertEqual(correct_scc, 1)
        self.assertEqual(mutated_scc, 0)
        self.assertNotEqual(correct_scc, mutated_scc)

    def test_shift_amount_mutation_is_detected(self):
        # Mutating shift amount N to N+1 produces wrong scaled address
        s0 = 10
        s1 = 5
        correct_n2, _ = ref_s_lshl_add_u32(2, s0, s1)  # 10 * 4 + 5 = 45
        mutated_n3, _ = ref_s_lshl_add_u32(3, s0, s1)  # 10 * 8 + 5 = 85
        self.assertEqual(correct_n2, 45)
        self.assertEqual(mutated_n3, 85)
        self.assertNotEqual(correct_n2, mutated_n3)

    def test_unmasked_modulo_overflow_is_detected(self):
        # Failing to mask with & 0xFFFFFFFF yields values >= 2^32
        s0 = 0x80000000
        s1 = 1
        correct_res, _ = ref_s_lshl_add_u32(1, s0, s1)
        unmasked = (s0 << 1) + s1
        self.assertEqual(correct_res, 1)
        self.assertEqual(unmasked, (1 << 32) + 1)
        self.assertNotEqual(correct_res, unmasked)


class LshlAddTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(2, 3, 4), (16, 0))
        self.assertEqual(ref(1, 0x80000000, 0), (0, 1))                 # the shifted-out bit alone sets SCC
        self.assertEqual(ref(4, 0x0FFFFFFF, 0x10), (0, 1))              # carry out of the addition alone
        self.assertEqual(ref(3, 0x1FFFFFFF, 7), (0xFFFFFFFF, 0))        # fits exactly

    def test_values_and_scc(self):
        for n in (1, 2, 3, 4):
            for x in VALUES:
                for y in VALUES[::3]:
                    want = ref(n, x, y)
                    with self.subTest(n=n, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(n, x, y, 1 - want[1]), want)

    def test_seeded(self):
        rng = random.Random(0x5A)
        for _ in range(40):
            n, x, y = rng.randint(1, 4), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(n=n, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(n, x, y, rng.getrandbits(1)), ref(n, x, y))

    def test_scc_drives_s_addc(self):
        # A 64-bit index scaled by 4 into a base: the carry from s_lshl2_add_u32 feeds the high word.
        got = run_lines(["s_mov_b32 s2, 0x40000001", "s_mov_b32 s3, 0xfffffffc", "s_mov_b32 s5, 7",
                         "s_lshl2_add_u32 s0, s2, s3", "s_addc_u32 s1, s5, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
        self.assertEqual(got, (0, 8))


if __name__ == "__main__":
    unittest.main()
