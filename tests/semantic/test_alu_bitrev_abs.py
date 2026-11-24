"""Semantic interpreter tests for RDNA2 vector bit manipulation and absolute value:
  - v_bfrev_b32: 32-bit vector bitfield reverse (D = reverse_bits(S0))
  - s_brev_b32: 32-bit scalar bitfield reverse
  - v_abs_i32: 32-bit vector integer absolute value (D = |S0|)
  - s_abs_i32: 32-bit scalar integer absolute value with SCC flag

Features covered:
  - RDNA2 vector bit manipulation reference oracles and algebraic identities:
      * Involution identity: bfrev(bfrev(x)) == x (bijective self-inverse permutation).
      * Popcount preservation: popcount(bfrev(x)) == popcount(x).
      * Bit boundary reflection: bit i maps to bit (31 - i) (e.g. bit 0 <-> bit 31, bit 15 <-> bit 16).
      * Algebraic homomorphism: bfrev(x ^ y) == bfrev(x) ^ bfrev(y) and bfrev(~x) == ~bfrev(x).
      * Extreme and canonical bit patterns: 0x00000000, 0xFFFFFFFF, 0x55555555 <-> 0xAAAAAAAA,
        0x33333333 <-> 0xCCCCCCCC, 0x0F0F0F0F <-> 0xF0F0F0F0, 0x00FF00FF <-> 0xFF00FF00,
        0x0000FFFF <-> 0xFFFF0000, 0x80000000 <-> 0x00000001.
      * Cross-oracle validation between string-reversal, parallel SWAR bit-swaps, and byte-lookup tables.
  - Two's-complement integer absolute value semantics:
      * D = |S0| for signed 32-bit integer in range [-2^31, 2^31 - 1].
      * Non-negative values are identity: |x| == x for x >= 0.
      * Strictly negative values are negated: |x| == -x for x < 0.
      * Two's-complement overflow: |-2147483648| == -2147483648 (0x80000000).
  - Synthetic shader translation via tools/rdna2spirv.py and execution via tests/semantic/spirv_mini_interp.py.

Zero game data, standard library Python only, Python 3.9+ compatible.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CANONICAL_VALUES = [
    0x00000000,
    0x00000001,
    0x00000002,
    0x00000003,
    0x00000080,
    0x00000100,
    0x0000FFFF,
    0x00010000,
    0x12345678,
    0x55555555,
    0xAAAAAAAA,
    0x33333333,
    0xCCCCCCCC,
    0x0F0F0F0F,
    0xF0F0F0F0,
    0x00FF00FF,
    0xFF00FF00,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
]
VALUES = CANONICAL_VALUES


def u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit dword as signed two's complement integer."""
    v = val & MASK32
    return v - (1 << 32) if (v & 0x80000000) else v


def popcount(x: int) -> int:
    """Count number of set bits in 32-bit dword (Python 3.9+ compatible)."""
    return bin(x & MASK32).count("1")


def ref_brev32(x: int) -> int:
    """Reference oracle for 32-bit bitfield reverse (string-based)."""
    return int(f"{x & MASK32:032b}"[::-1], 2)


def ref_brev32_swar(x: int) -> int:
    """Hacker's Delight parallel SWAR (SIMD within a register) 32-bit bit reversal."""
    v = x & MASK32
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
    v = ((v >> 16) & 0x0000FFFF) | ((v & 0x0000FFFF) << 16)
    return v & MASK32


_REV_TABLE_8 = tuple(int(f"{b:08b}"[::-1], 2) for b in range(256))


def ref_brev32_table(x: int) -> int:
    """Table-based 32-bit bit reversal oracle across 4 lookup bytes."""
    v = x & MASK32
    b0 = _REV_TABLE_8[v & 0xFF]
    b1 = _REV_TABLE_8[(v >> 8) & 0xFF]
    b2 = _REV_TABLE_8[(v >> 16) & 0xFF]
    b3 = _REV_TABLE_8[(v >> 24) & 0xFF]
    return ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) & MASK32


# Alias for backwards compatibility
ref_brev = ref_brev32


def ref_abs32(x: int) -> int:
    """Reference oracle for signed 32-bit absolute value with two's-complement overflow."""
    s = s32(x)
    return u32(-s if s < 0 else s)


def ref_absdiff32(x: int, y: int) -> int:
    """Reference oracle for 32-bit signed absolute difference: |x - y|."""
    diff = (u32(x) - u32(y)) & MASK32
    return ref_abs32(diff)


# Alias for backwards compatibility
ref_abs = ref_abs32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bitrev_abs.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bitrev_abs.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines: list[str]) -> str:
    """Assemble synthetic shader lines with rdna2spirv and return the SPIR-V assembly text."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_bitrev_abs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines: list[str]) -> tuple[int, ...]:
    """Assemble and run synthetic pixel shader lines, returning (out_mrt0.x, out_mrt0.y)."""
    spv = translate_lines(lines)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class VectorBitrevOracleTests(unittest.TestCase):
    """Rigorous algebraic and property-based verification of bit-reversal reference models."""

    def test_oracle_equivalence_across_algorithms(self):
        """String-based, SWAR bit-swap, and byte-lookup algorithms must be 100% equivalent."""
        for val in CANONICAL_VALUES:
            with self.subTest(val=hex(val)):
                r1 = ref_brev32(val)
                r2 = ref_brev32_swar(val)
                r3 = ref_brev32_table(val)
                self.assertEqual(r1, r2)
                self.assertEqual(r1, r3)

    def test_single_bit_boundary_reflections(self):
        """Bit i must reflect strictly to bit (31 - i) for all 0 <= i <= 31."""
        for i in range(32):
            src = 1 << i
            expected = 1 << (31 - i)
            with self.subTest(bit=i, src=hex(src)):
                self.assertEqual(ref_brev32(src), expected)
                self.assertEqual(ref_brev32_swar(src), expected)
                self.assertEqual(ref_brev32_table(src), expected)

    def test_popcount_preservation_property(self):
        """Bit reversal must preserve hamming weight (popcount) identically: popcount(brev(x)) == popcount(x)."""
        test_inputs = list(CANONICAL_VALUES)
        # Add random-like deterministic multi-bit patterns
        for shift in range(32):
            test_inputs.append((0x12345678 << shift) & MASK32)
            test_inputs.append((0xCAFEBABE >> shift) & MASK32)
            test_inputs.append((1 << shift) | (1 << ((shift + 7) % 32)))

        for x in test_inputs:
            with self.subTest(x=hex(x)):
                rev = ref_brev32(x)
                self.assertEqual(popcount(rev), popcount(x))

    def test_involution_identity(self):
        """Bit reversal is an involution: bfrev(bfrev(x)) == x for all inputs."""
        for x in CANONICAL_VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(ref_brev32(ref_brev32(x)), x)
                self.assertEqual(ref_brev32_swar(ref_brev32_swar(x)), x)
                self.assertEqual(ref_brev32_table(ref_brev32_table(x)), x)

    def test_algebraic_homomorphism_xor_and_not(self):
        """bfrev distributes over XOR: bfrev(x ^ y) == bfrev(x) ^ bfrev(y) and bfrev(~x) == ~bfrev(x)."""
        pairs = [
            (0x55555555, 0xAAAAAAAA),
            (0x12345678, 0x87654321),
            (0x00FF00FF, 0xF0F0F0F0),
            (0x80000001, 0x7FFFFFFE),
            (0xFFFFFFFF, 0x12345678),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_brev32(a ^ b), ref_brev32(a) ^ ref_brev32(b))
                self.assertEqual(ref_brev32(~a & MASK32), (~ref_brev32(a)) & MASK32)
                self.assertEqual(ref_brev32(~b & MASK32), (~ref_brev32(b)) & MASK32)

    def test_palindromic_self_reversing_bit_patterns(self):
        """Bit patterns that are symmetric across the center must equal themselves under bit reversal."""
        palindromes = [
            0x00000000,
            0xFFFFFFFF,
            0x80000001,  # bit 0 and bit 31 set
            0x00018000,  # bit 15 and bit 16 set
            0x5555AAAA,  # 0101...1010
            0x0FF00FF0,
            0x3C3C3C3C,
            0xA55A5AA5,
        ]
        for p in palindromes:
            with self.subTest(p=hex(p)):
                self.assertEqual(ref_brev32(p), p)

    def test_canonical_reflections(self):
        """Canonical test pairs must match hand-derived mathematical reflections."""
        expected_pairs = [
            (0x00000001, 0x80000000),
            (0x00000003, 0xC0000000),
            (0x55555555, 0xAAAAAAAA),
            (0x33333333, 0xCCCCCCCC),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x00FF00FF, 0xFF00FF00),
            (0x0000FFFF, 0xFFFF0000),
            (0x12345678, 0x1E6A2C48),
        ]
        for src, expected in expected_pairs:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_brev32(src), expected)
                self.assertEqual(ref_brev32(expected), src)


class VectorAluInterpreter:
    """Deterministic SIMD wave interpreter modeling vector and scalar ALU with execution masking.

    Supports Wave32 (32 lanes) and Wave64 (64 lanes).
    Models:
      - EXEC mask: per-lane active bit controlling vector instruction side-effects.
      - Inactive lanes strictly retain their previous VGPR contents.
      - SGPRs: uniform scalar registers shared across all lanes in the wave.
      - SCC: scalar condition code flag (updated by scalar ops like s_abs_i32).
      - VGPRs: independent per-lane vector registers.
    """

    def __init__(self, wave_size: int = 32, initial_exec: int = 0xFFFFFFFF) -> None:
        assert wave_size in (32, 64)
        self.wave_size = wave_size
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask = initial_exec & self.mask_all
        self.sgprs = [0] * 128
        self.scc = 0
        self.vgprs = [[0] * 256 for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & self.mask_all

    def resolve_src(self, lane: int, opnd: int | str) -> int:
        if isinstance(opnd, int):
            return opnd & MASK32
        if isinstance(opnd, str):
            if opnd.startswith("v"):
                reg = int(opnd[1:])
                return self.vgprs[lane][reg] & MASK32
            if opnd.startswith("s"):
                reg = int(opnd[1:])
                return self.sgprs[reg] & MASK32
            try:
                return int(opnd, 0) & MASK32
            except ValueError:
                pass
        raise ValueError(f"Unknown operand: {opnd}")

    def execute_instruction(self, op: str, *args) -> None:
        if op == "v_mov_b32":
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    self.vgprs[lane][dst_reg] = self.resolve_src(lane, src)

        elif op in ("v_bfrev_b32", "v_bfrev_b32_e32"):
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    val = self.resolve_src(lane, src)
                    self.vgprs[lane][dst_reg] = ref_brev32(val)

        elif op in ("v_abs_i32", "v_abs_i32_e32"):
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    val = self.resolve_src(lane, src)
                    self.vgprs[lane][dst_reg] = ref_abs32(val)

        elif op in ("v_absdiff_i32", "v_absdiff_i32_e32"):
            dst, src0, src1 = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    v0 = self.resolve_src(lane, src0)
                    v1 = self.resolve_src(lane, src1)
                    self.vgprs[lane][dst_reg] = ref_absdiff32(v0, v1)

        elif op == "s_mov_b32":
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            self.sgprs[dst_reg] = self.resolve_src(0, src)

        elif op == "s_abs_i32":
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            res = ref_abs32(self.resolve_src(0, src))
            self.sgprs[dst_reg] = res
            self.scc = 1 if res != 0 else 0

        elif op == "s_absdiff_i32":
            dst, src0, src1 = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            res = ref_absdiff32(self.resolve_src(0, src0), self.resolve_src(0, src1))
            self.sgprs[dst_reg] = res
            self.scc = 1 if res != 0 else 0

        elif op == "s_brev_b32":
            dst, src = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            res = ref_brev32(self.resolve_src(0, src))
            self.sgprs[dst_reg] = res

        elif op in ("s_and_saveexec_b32", "s_and_saveexec_b64"):
            dst, mask = args
            dst_reg = int(dst[1:]) if isinstance(dst, str) else dst
            mask_val = mask if isinstance(mask, int) else self.resolve_src(0, mask)
            self.sgprs[dst_reg] = self.exec_mask & MASK32
            if self.wave_size == 64 and op.endswith("64"):
                self.sgprs[dst_reg + 1] = (self.exec_mask >> 32) & MASK32
            self.exec_mask = (self.exec_mask & mask_val) & self.mask_all

        elif op in ("s_or_exec_b32", "s_or_exec_b64"):
            mask = args[0]
            mask_val = mask if isinstance(mask, int) else self.resolve_src(0, mask)
            self.exec_mask = (self.exec_mask | mask_val) & self.mask_all
        else:
            raise NotImplementedError(f"Unsupported op: {op}")


class VectorAbsPropertiesTests(unittest.TestCase):
    """Algebraic verification of signed 32-bit absolute value and difference semantics."""

    def test_non_negative_identity(self):
        """For non-negative signed integers, abs(x) == x."""
        non_negatives = [0, 1, 2, 42, 1000, 0x12345678, 0x7FFFFFFE, 0x7FFFFFFF]
        for x in non_negatives:
            with self.subTest(x=hex(x)):
                self.assertEqual(ref_abs32(x), x)

    def test_strictly_negative_inversion(self):
        """For strictly negative integers (excluding INT_MIN), abs(x) == -x."""
        negatives = [-1, -2, -42, -1000, -0x12345678, -0x7FFFFFFF]
        for s in negatives:
            u = u32(s)
            with self.subTest(s=s, u=hex(u)):
                self.assertEqual(ref_abs32(u), abs(s))

    def test_two_complement_overflow_int_min(self):
        """Two's-complement overflow: |-2147483648| wraps to -2147483648 (0x80000000)."""
        int_min = 0x80000000
        self.assertEqual(ref_abs32(int_min), int_min)
        self.assertEqual(s32(ref_abs32(int_min)), -2147483648)

    def test_abs_idempotency(self):
        """Absolute value is idempotent: abs(abs(x)) == abs(x)."""
        for x in CANONICAL_VALUES:
            with self.subTest(x=hex(x)):
                a1 = ref_abs32(x)
                a2 = ref_abs32(a1)
                self.assertEqual(a1, a2)

    def test_absdiff_symmetry(self):
        """Absolute difference is symmetric: |x - y| == |y - x| in 32-bit modular arithmetic."""
        test_pairs = [
            (0, 0),
            (0, 1),
            (1, 0),
            (10, 42),
            (42, 10),
            (0xFFFFFFFF, 1),
            (1, 0xFFFFFFFF),
            (0x7FFFFFFF, 0x80000000),
            (0x80000000, 0x7FFFFFFF),
            (0x12345678, 0x87654321),
            (0x55555555, 0xAAAAAAAA),
        ]
        for x, y in test_pairs:
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(ref_absdiff32(x, y), ref_absdiff32(y, x))

    def test_absdiff_zero_identity(self):
        """|x - x| == 0 for all 32-bit values."""
        for x in CANONICAL_VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(ref_absdiff32(x, x), 0)

    def test_absdiff_with_zero(self):
        """|x - 0| == |x| and |0 - x| == |x|."""
        for x in CANONICAL_VALUES:
            with self.subTest(x=hex(x)):
                want = ref_abs32(x)
                self.assertEqual(ref_absdiff32(x, 0), want)
                self.assertEqual(ref_absdiff32(0, x), want)

    def test_absdiff_overflow_wrap(self):
        """Difference wraps before absolute value: 0x7FFFFFFF - 0x80000000 = -1 -> abs is 1."""
        self.assertEqual(ref_absdiff32(0x7FFFFFFF, 0x80000000), 1)
        self.assertEqual(ref_absdiff32(0x80000000, 0x7FFFFFFF), 1)


class SimdMultiLaneDivergenceTests(unittest.TestCase):
    """Multi-lane SIMD divergence verification under diverse EXEC execution masks."""

    def test_full_exec_all_lanes_updated(self):
        """Wave32 with EXEC=0xFFFFFFFF updates all 32 lanes."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        sentinel = 0xDEADBEEF
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = u32(-(lane + 1))

        interp.execute_instruction("v_abs_i32", "v0", "v1")

        for lane in range(32):
            expected = lane + 1
            self.assertEqual(interp.vgprs[lane][0], expected, f"Lane {lane} failed")

    def test_zero_exec_preserves_all_vgprs(self):
        """Wave32 with EXEC=0x00000000 preserves all destination VGPRs across the wave."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0x00000000)
        sentinel = 0xDEADBEEF
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = u32(-(lane + 1))

        interp.execute_instruction("v_abs_i32", "v0", "v1")

        for lane in range(32):
            self.assertEqual(interp.vgprs[lane][0], sentinel, f"Lane {lane} was incorrectly overwritten")

    def test_checkerboard_divergence_v_abs(self):
        """Wave32 with EXEC=0x55555555 updates even lanes and preserves odd lanes."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0x55555555)
        sentinel = 0xBAADF00D
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = u32(-((lane + 1) * 10))

        interp.execute_instruction("v_abs_i32", "v0", "v1")

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(interp.vgprs[lane][0], (lane + 1) * 10)
            else:
                self.assertEqual(interp.vgprs[lane][0], sentinel)

    def test_checkerboard_divergence_v_bfrev(self):
        """Wave32 with EXEC=0xAAAAAAAA updates odd lanes and preserves even lanes with bit-reversal."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0xAAAAAAAA)
        sentinel = 0xCAFEBABE
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = 1 << lane

        interp.execute_instruction("v_bfrev_b32", "v0", "v1")

        for lane in range(32):
            if lane % 2 == 1:
                self.assertEqual(interp.vgprs[lane][0], 1 << (31 - lane))
            else:
                self.assertEqual(interp.vgprs[lane][0], sentinel)

    def test_single_lane_active_boundaries(self):
        """Test isolated active lane at boundaries (lane 0, lane 15, lane 31)."""
        boundary_lanes = [0, 15, 31]
        for active_lane in boundary_lanes:
            with self.subTest(active_lane=active_lane):
                exec_mask = 1 << active_lane
                interp = VectorAluInterpreter(wave_size=32, initial_exec=exec_mask)
                sentinel = 0x11223344
                for lane in range(32):
                    interp.vgprs[lane][0] = sentinel
                    interp.vgprs[lane][1] = 0x80000000 if lane == 0 else u32(-(lane * 100))

                interp.execute_instruction("v_abs_i32", "v0", "v1")

                for lane in range(32):
                    if lane == active_lane:
                        expected = ref_abs32(interp.vgprs[lane][1])
                        self.assertEqual(interp.vgprs[lane][0], expected)
                    else:
                        self.assertEqual(interp.vgprs[lane][0], sentinel)

    def test_wave64_divergence_halves(self):
        """Wave64 with low half active (EXEC=0x00000000FFFFFFFF) updates lanes 0..31, preserves 32..63."""
        interp = VectorAluInterpreter(wave_size=64, initial_exec=0x00000000FFFFFFFF)
        sentinel = 0x55AA55AA
        for lane in range(64):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = u32(-(lane + 1))

        interp.execute_instruction("v_abs_i32", "v0", "v1")

        for lane in range(64):
            if lane < 32:
                self.assertEqual(interp.vgprs[lane][0], lane + 1)
            else:
                self.assertEqual(interp.vgprs[lane][0], sentinel)

    def test_vector_absdiff_simd_multi_lane(self):
        """v_absdiff_i32 across SIMD lanes with pairwise vector operands."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0x0000FFFF)
        sentinel = 0x01020304
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = 50 + lane
            interp.vgprs[lane][2] = 100

        interp.execute_instruction("v_absdiff_i32", "v0", "v1", "v2")

        for lane in range(32):
            if lane < 16:
                expected = ref_absdiff32(50 + lane, 100)
                self.assertEqual(interp.vgprs[lane][0], expected)
            else:
                self.assertEqual(interp.vgprs[lane][0], sentinel)

    def test_saveexec_restore_simd_branch_pattern(self):
        """Simulates divergent branch pattern using s_and_saveexec and s_or_exec."""
        interp = VectorAluInterpreter(wave_size=32, initial_exec=0xFFFFFFFF)
        # Initialize lanes: even lanes get negative, odd lanes get positive
        for lane in range(32):
            interp.vgprs[lane][0] = 0
            interp.vgprs[lane][1] = u32(-(lane + 1)) if (lane % 2 == 0) else (lane + 1)

        # Step 1: Branch taken by even lanes only (EXEC & 0x55555555)
        # s_and_saveexec_b32 s2, 0x55555555 -> saves 0xFFFFFFFF to s2, sets EXEC=0x55555555
        interp.execute_instruction("s_and_saveexec_b32", "s2", 0x55555555)
        self.assertEqual(interp.sgprs[2], 0xFFFFFFFF)
        self.assertEqual(interp.exec_mask, 0x55555555)

        # In then-block: v_abs_i32 v0, v1 (only even lanes update)
        interp.execute_instruction("v_abs_i32", "v0", "v1")

        # Step 2: Invert mask for else-block: EXEC = (saved ~EXEC) & saved
        # s_or_exec_b32 restores saved or complements
        interp.set_exec((~0x55555555) & 0xFFFFFFFF)
        # In else-block: v_bfrev_b32 v0, v1 (only odd lanes update)
        interp.execute_instruction("v_bfrev_b32", "v0", "v1")

        # Step 3: Reconverge
        interp.execute_instruction("s_mov_b32", "s0", "s2")
        interp.set_exec(interp.sgprs[0])
        self.assertEqual(interp.exec_mask, 0xFFFFFFFF)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(interp.vgprs[lane][0], lane + 1)
            else:
                self.assertEqual(interp.vgprs[lane][0], ref_brev32(lane + 1))


class BitReverseTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_brev(1), 0x80000000)
        self.assertEqual(ref_brev(0x12345678), 0x1E6A2C48)
        self.assertEqual(ref_brev(0x0000FFFF), 0xFFFF0000)

    def test_v_bfrev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v0, v2", "v_mov_b32 v1, 0"])[0],
                                 ref_brev(x))

    def test_s_brev_b32(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"s_mov_b32 s2, {x:#x}", "s_brev_b32 s0, s2", "v_mov_b32 v0, s0",
                                            "v_mov_b32 v1, 0"])[0], ref_brev(x))

    def test_twice_is_identity(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_lines([f"v_mov_b32 v2, {x:#x}", "v_bfrev_b32_e32 v3, v2", "v_bfrev_b32_e32 v0, v3",
                                            "v_mov_b32 v1, 0"])[0], x)


class AbsTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_abs(0xFFFFFFFF), 1)
        self.assertEqual(ref_abs(0x80000000), 0x80000000)
        self.assertEqual(ref_abs(0x7FFFFFFF), 0x7FFFFFFF)

    def test_s_abs_i32_and_scc(self):
        for x in VALUES:
            with self.subTest(x=hex(x)):
                want = ref_abs(x)
                got = run_lines([f"s_mov_b32 s2, {x:#x}", "s_abs_i32 s0, s2", "s_cselect_b32 s1, 1, 0",
                                 "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
                self.assertEqual(got, (want, int(want != 0)))

    def test_s_absdiff_i32_and_scc(self):
        test_pairs = [
            (0, 0),
            (5, 12),
            (12, 5),
            (10, 10),
            (0xFFFFFFFF, 1),
            (1, 0xFFFFFFFF),
            (0x7FFFFFFF, 0x80000000),
            (0x80000000, 0x7FFFFFFF),
            (0x80000000, 0),
            (0, 0x80000000),
            (0x12345678, 0x12345678),
            (0x12345678, 0x87654321),
        ]
        for x, y in test_pairs:
            with self.subTest(x=hex(x), y=hex(y)):
                want = ref_absdiff32(x, y)
                got = run_lines([f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}",
                                 "s_absdiff_i32 s0, s2, s3", "s_cselect_b32 s1, 1, 0",
                                 "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
                self.assertEqual(got, (want, int(want != 0)))


class SpirvEmissionTests(unittest.TestCase):
    """Verify that tools/rdna2spirv emits expected SPIR-V instructions and structures."""

    def test_v_bfrev_b32_emission(self):
        spv = translate_lines(["v_mov_b32 v2, 0x12345678", "v_bfrev_b32_e32 v0, v2"])
        self.assertIn("OpBitReverse %uint", spv)

    def test_s_brev_b32_emission(self):
        spv = translate_lines(["s_mov_b32 s2, 0x12345678", "s_brev_b32 s0, s2", "v_mov_b32 v0, s0"])
        self.assertIn("OpBitReverse %uint", spv)

    def test_s_abs_i32_emission(self):
        # A loaded operand, not a literal: SHADER-315 folds scalar ALU ops on constants at translate time.
        spv = translate_lines(["s_load_dword s2, s[0:1], 0x0", "s_waitcnt lgkmcnt(0)", "s_abs_i32 s0, s2",
                               "v_mov_b32 v0, s0"])
        self.assertIn('OpExtInstImport "GLSL.std.450"', spv)
        self.assertIn("SAbs", spv)
        self.assertIn("OpStore %scc", spv)

    def test_s_absdiff_i32_emission(self):
        spv = translate_lines(["s_load_dwordx2 s[2:3], s[0:1], 0x0", "s_waitcnt lgkmcnt(0)",
                               "s_absdiff_i32 s0, s2, s3", "v_mov_b32 v0, s0"])
        self.assertIn("OpISub %uint", spv)
        self.assertIn("SAbs", spv)
        self.assertIn("OpStore %scc", spv)


class DirectSpirvInterpreterEdgeCaseTests(unittest.TestCase):
    """Direct verification of SPIR-V OpBitReverse and SAbs via mini-interpreter and wave-interpreter."""

    def test_spirv_opbitreverse_all_single_bits(self):
        for bit in range(32):
            val = 1 << bit
            expected = 1 << (31 - bit)
            spv = f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%_ptr_Output_v4uint = OpTypePointer Output %v4uint
%out_mrt0 = OpVariable %_ptr_Output_v4uint Output
%val = OpConstant %uint {val}
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%rev = OpBitReverse %uint %val
%out = OpCompositeConstruct %v4uint %rev %c0 %c0 %c0
OpStore %out_mrt0 %out
OpReturn
OpFunctionEnd
"""
            out = si.run_fragment(spv)["%out_mrt0"]
            self.assertEqual(out[0], expected)

    def test_spirv_sabs_two_complement_wrap(self):
        spv = """OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%_ptr_Output_v4uint = OpTypePointer Output %v4uint
%out_mrt0 = OpVariable %_ptr_Output_v4uint Output
%int_min = OpConstant %uint 2147483648
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%i = OpBitcast %int %int_min
%ab = OpExtInst %int %glsl SAbs %i
%res = OpBitcast %uint %ab
%out = OpCompositeConstruct %v4uint %res %c0 %c0 %c0
OpStore %out_mrt0 %out
OpReturn
OpFunctionEnd
"""
        out = si.run_fragment(spv)["%out_mrt0"]
        self.assertEqual(out[0], 0x80000000)

    def test_spirv_wave_bitreverse_all_lanes(self):
        spv = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %lane %out
OpDecorate %lane BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%pin = OpTypePointer Input %uint
%pout = OpTypePointer Output %v4uint
%lane = OpVariable %pin Input
%out = OpVariable %pout Output
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%l = OpLoad %uint %lane
%rev = OpBitReverse %uint %l
%v = OpCompositeConstruct %v4uint %rev %c0 %c0 %c0
OpStore %out %v
OpReturn
OpFunctionEnd
"""
        res = wi.run_wave(spv, wave=32)
        outputs = res["outputs"]
        for lane in range(32):
            self.assertEqual(outputs[lane]["%out"][0], ref_brev32(lane))

    def test_spirv_wave_sabs_all_lanes(self):
        spv = """OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %lane %out
OpDecorate %lane BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%pin = OpTypePointer Input %uint
%pout = OpTypePointer Output %v4uint
%lane = OpVariable %pin Input
%out = OpVariable %pout Output
%c0 = OpConstant %uint 0
%c0_i = OpConstant %int 0
%main = OpFunction %void None %fn
%entry = OpLabel
%l = OpLoad %uint %lane
%i = OpBitcast %int %l
%neg = OpISub %int %c0_i %i
%ab = OpExtInst %int %glsl SAbs %neg
%u = OpBitcast %uint %ab
%v = OpCompositeConstruct %v4uint %u %c0 %c0 %c0
OpStore %out %v
OpReturn
OpFunctionEnd
"""
        res = wi.run_wave(spv, wave=32)
        outputs = res["outputs"]
        for lane in range(32):
            self.assertEqual(outputs[lane]["%out"][0], lane)


class NegativeMutationControlTests(unittest.TestCase):
    """Negative controls verifying that mutated or corrupt behaviors are strictly detected."""

    def test_mutation_bitrev_shift_rejected(self):
        def bad_brev32_shift(x: int) -> int:
            return (ref_brev32(x) >> 1) & MASK32

        self.assertNotEqual(bad_brev32_shift(1), ref_brev32(1))
        self.assertNotEqual(bad_brev32_shift(0x12345678), ref_brev32(0x12345678))

    def test_mutation_bitrev_inverted_rejected(self):
        def bad_brev32_invert(x: int) -> int:
            return (~ref_brev32(x)) & MASK32

        self.assertNotEqual(bad_brev32_invert(0), ref_brev32(0))
        self.assertNotEqual(bad_brev32_invert(0xFFFFFFFF), ref_brev32(0xFFFFFFFF))

    def test_mutation_abs_unwrapped_overflow_rejected(self):
        def bad_abs32(x: int) -> int:
            return 0 if x == 0x80000000 else ref_abs32(x)

        self.assertNotEqual(bad_abs32(0x80000000), 0x80000000)

    def test_mutation_simd_inactive_lanes_written_rejected(self):
        class MutatedInterpreterNoMask(VectorAluInterpreter):
            def is_lane_active(self, lane: int) -> bool:
                return True

        interp = MutatedInterpreterNoMask(wave_size=32, initial_exec=0x00000000)
        sentinel = 0xDEADBEEF
        for lane in range(32):
            interp.vgprs[lane][0] = sentinel
            interp.vgprs[lane][1] = 100

        interp.execute_instruction("v_abs_i32", "v0", "v1")

        overwritten_lanes = [lane for lane in range(32) if interp.vgprs[lane][0] != sentinel]
        self.assertEqual(len(overwritten_lanes), 32)

    def test_mutation_absdiff_unsigned_sub_rejected(self):
        def bad_absdiff_unsigned(x: int, y: int) -> int:
            return abs(u32(x) - u32(y)) & MASK32

        self.assertNotEqual(bad_absdiff_unsigned(0xFFFFFFFF, 1), ref_absdiff32(0xFFFFFFFF, 1))

    def test_mutation_spirv_opcode_corruption_rejected(self):
        corrupted_spv = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%_ptr_Output_v4uint = OpTypePointer Output %v4uint
%out_mrt0 = OpVariable %_ptr_Output_v4uint Output
%val = OpConstant %uint 1
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpInvalidNonExistentOpcode %uint %val
%out = OpCompositeConstruct %v4uint %res %c0 %c0 %c0
OpStore %out_mrt0 %out
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(corrupted_spv)


if __name__ == "__main__":
    unittest.main()
