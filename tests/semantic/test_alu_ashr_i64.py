from __future__ import annotations

"""Semantic interpreter tests for RDNA2 64-bit vector arithmetic shift right (v_ashrrev_i64).

RDNA2 ISA definition:
  v_ashrrev_i64 dst, src0, src1
  - dst: 64-bit VGPR pair v[d:d+1]
  - src0: 32-bit shift count (VGPR, SGPR, or inline constant). Bits [5:0] define the shift amount.
  - src1: 64-bit input operand (VGPR pair v[s:s+1], SGPR pair s[s:s+1], or inline constant).
  - Operation: dst = (int64)src1 >> (src0 & 63), sign-extended with bit 63 copied into upper bits.
"""

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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN32 = 0x80000000
SIGN64 = 1 << 63
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_ashr_i64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_ashr_i64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        try:
            _TMP_DIR.cleanup()
        except Exception:
            pass
        _TMP_DIR = None
        _SRC_PATH = None


def u32(x: int) -> int:
    """Normalize integer to 32-bit unsigned."""
    return x & MASK32


def s32(x: int) -> int:
    """Interpret 32-bit unsigned integer as signed two's complement."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN32) else x


def u64(x: int) -> int:
    """Normalize integer to 64-bit unsigned."""
    return x & MASK64


def s64(x: int) -> int:
    """Interpret 64-bit unsigned integer as signed two's complement."""
    x &= MASK64
    return x - (1 << 64) if (x & SIGN64) else x


def decompose_u64(val: int) -> tuple[int, int]:
    """Decompose a 64-bit unsigned integer into (lo, hi) 32-bit dwords."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def compose_u64(lo: int, hi: int) -> int:
    """Compose two 32-bit dwords into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def ref_ashr_i64(shift: int, val: int) -> int:
    """Ideal reference oracle for v_ashrrev_i64: (int64)val >> (shift & 63)."""
    sh = shift & 63
    return (s64(val) >> sh) & MASK64


def emulate_ashr_i64_dwords(shift: int, lo: int, hi: int) -> tuple[int, int]:
    """Emulate 64-bit arithmetic shift right using 32-bit register pair datapath."""
    sh = shift & 63
    lo &= MASK32
    hi &= MASK32

    if sh == 0:
        return lo, hi
    if sh < 32:
        res_lo = ((lo >> sh) | ((hi << (32 - sh)) & MASK32)) & MASK32
        res_hi = (s32(hi) >> sh) & MASK32
        return res_lo, res_hi
    if sh == 32:
        res_lo = hi
        res_hi = MASK32 if (hi & SIGN32) else 0
        return res_lo, res_hi

    sh_hi = sh - 32
    res_lo = (s32(hi) >> sh_hi) & MASK32
    res_hi = MASK32 if (hi & SIGN32) else 0
    return res_lo, res_hi


EXPORT_HALVES = [
    "v_and_b32 v5, 0xffff, v0",
    "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1",
    "v_lshrrev_b32 v8, 16, v1",
    "v_or_b32 v5, 0x40000000, v5",
    "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7",
    "v_or_b32 v8, 0x40000000, v8",
]


def run_lines(lines: list[str]) -> int:
    """Run assembly lines, export v[0:1] in 16-bit halves, return as one 64-bit value."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None

    body = list(lines) + EXPORT_HALVES
    asm = "; test_alu_ashr_i64\n" + "".join(f"\t{ln}\n" for ln in body) + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    words = si.run_fragment(spv)["%out_mrt0"]
    assert all((x >> 16) == 0x4000 for x in words), [hex(x) for x in words]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(words))


def run_ashr_op(shift: int, val: int, src_reg: str = "v[2:3]", dst_reg: str = "v[0:1]") -> int:
    """Helper to load operands into registers, execute v_ashrrev_i64, and export result."""
    lo, hi = decompose_u64(val)
    lines = [
        f"v_mov_b32 v4, {shift:#x}",
        f"v_mov_b32 v2, {lo:#x}",
        f"v_mov_b32 v3, {hi:#x}",
        f"v_ashrrev_i64 {dst_reg}, v4, {src_reg}",
    ]
    if dst_reg != "v[0:1]":
        parts = dst_reg.strip("v[]").split(":")
        d0, d1 = int(parts[0]), int(parts[1])
        lines.extend([
            f"v_mov_b32 v0, v{d0}",
            f"v_mov_b32 v1, v{d1}",
        ])
    return run_lines(lines)


class TestRefOracle(unittest.TestCase):
    """Unit tests validating reference oracle, 32-bit dword pair emulation, and basic shader execution."""

    def test_u64_s64_conversions(self) -> None:
        self.assertEqual(u64(0), 0)
        self.assertEqual(s64(0), 0)
        self.assertEqual(s64(1), 1)
        self.assertEqual(s64(0x7FFFFFFFFFFFFFFF), (1 << 63) - 1)
        self.assertEqual(s64(0x8000000000000000), -(1 << 63))
        self.assertEqual(s64(MASK64), -1)

    def test_decompose_compose_roundtrip(self) -> None:
        values = [
            0,
            1,
            MASK64,
            SIGN64,
            SIGN64 - 1,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            0x00000000FFFFFFFF,
            0xFFFFFFFF00000000,
        ]
        for val in values:
            lo, hi = decompose_u64(val)
            self.assertEqual(compose_u64(lo, hi), val)

    def test_dword_emulation_matches_ref_oracle(self) -> None:
        test_values = [
            0,
            1,
            -1 & MASK64,
            SIGN64,
            SIGN64 - 1,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            0x8000000000000001,
            0x7FFFFFFFFFFFFFFF,
            0xAAAAAAAA55555555,
            0x55555555AAAAAAAA,
        ]
        for val in test_values:
            lo, hi = decompose_u64(val)
            for shift in range(64):
                ref = ref_ashr_i64(shift, val)
                elo, ehi = emulate_ashr_i64_dwords(shift, lo, hi)
                got = compose_u64(elo, ehi)
                self.assertEqual(got, ref, f"Mismatch at val={hex(val)}, shift={shift}")

    def test_identity_shift_zero(self) -> None:
        cases = [0, 1, MASK64, SIGN64, 0x0123456789ABCDEF, 0x8000000000000001]
        for val in cases:
            self.assertEqual(ref_ashr_i64(0, val), val)
            self.assertEqual(ref_ashr_i64(64, val), val)

    def test_basic_positive_values(self) -> None:
        val = 0x4000000000000000
        self.assertEqual(ref_ashr_i64(1, val), 0x2000000000000000)
        self.assertEqual(ref_ashr_i64(2, val), 0x1000000000000000)
        self.assertEqual(ref_ashr_i64(62, val), 1)
        self.assertEqual(ref_ashr_i64(63, val), 0)

    def test_basic_negative_values(self) -> None:
        val = SIGN64
        self.assertEqual(ref_ashr_i64(1, val), 0xC000000000000000)
        self.assertEqual(ref_ashr_i64(2, val), 0xE000000000000000)
        self.assertEqual(ref_ashr_i64(31, val), 0xFFFFFFFF00000000)
        self.assertEqual(ref_ashr_i64(32, val), 0xFFFFFFFF80000000)
        self.assertEqual(ref_ashr_i64(63, val), MASK64)

    def test_shader_execution_basic(self) -> None:
        val = 0x8000000000000000
        got = run_ashr_op(1, val)
        self.assertEqual(got, 0xC000000000000000)
        got = run_ashr_op(32, val)
        self.assertEqual(got, 0xFFFFFFFF80000000)

    def test_shader_immediate_shift_operand(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_ashrrev_i64 v[0:1], 4, v[2:3]",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(4, val))

    def test_shader_scalar_pair_operand(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)
        lines = [
            f"s_mov_b32 s2, {lo:#x}",
            f"s_mov_b32 s3, {hi:#x}",
            "v_ashrrev_i64 v[0:1], 8, s[2:3]",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(8, val))

    def test_shader_vector_pair_operand(self) -> None:
        val = 0xFEDCBA9876543210
        lo, hi = decompose_u64(val)
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_mov_b32 v4, 16",
            "v_ashrrev_i64 v[0:1], v4, v[2:3]",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(16, val))


class TestSimdMultiLaneVgprPairs(unittest.TestCase):
    """Multi-lane SIMD wave simulation and VGPR register pair allocation tests."""

    def test_wave64_divergent_shifts_uniform_data(self) -> None:
        val = 0x8000000000000000
        lanes = 64
        for lane_id in range(lanes):
            shift_count = lane_id
            expected = ref_ashr_i64(shift_count, val)
            lo, hi = decompose_u64(val)
            elo, ehi = emulate_ashr_i64_dwords(shift_count, lo, hi)
            got = compose_u64(elo, ehi)
            self.assertEqual(got, expected, f"Lane {lane_id} shift {shift_count} failed")

    def test_wave64_uniform_shift_divergent_data(self) -> None:
        rng = random.Random(0x42)
        lanes = 64
        shift = 16
        for lane_id in range(lanes):
            lane_val = rng.getrandbits(64)
            if lane_id % 2 == 0:
                lane_val |= SIGN64
            else:
                lane_val &= ~SIGN64
            expected = ref_ashr_i64(shift, lane_val)
            lo, hi = decompose_u64(lane_val)
            elo, ehi = emulate_ashr_i64_dwords(shift, lo, hi)
            got = compose_u64(elo, ehi)
            self.assertEqual(got, expected, f"Lane {lane_id} val {hex(lane_val)} failed")

    def test_wave64_fully_divergent_lanes(self) -> None:
        rng = random.Random(0x1337)
        lanes = 64
        for lane_id in range(lanes):
            lane_shift = rng.getrandbits(32)
            lane_val = rng.getrandbits(64)
            expected = ref_ashr_i64(lane_shift, lane_val)
            lo, hi = decompose_u64(lane_val)
            elo, ehi = emulate_ashr_i64_dwords(lane_shift, lo, hi)
            got = compose_u64(elo, ehi)
            self.assertEqual(got, expected, f"Lane {lane_id} divergent pair failed")

    def test_shader_non_overlapping_vgpr_pairs(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)

        pairs = [
            ("v[4:5]", "v[6:7]"),
            ("v[10:11]", "v[12:13]"),
            ("v[20:21]", "v[22:23]"),
            ("v[30:31]", "v[14:15]"),
        ]
        for src_reg, dst_reg in pairs:
            s0 = int(src_reg.strip("v[]").split(":")[0])
            s1 = s0 + 1
            d0 = int(dst_reg.strip("v[]").split(":")[0])
            d1 = d0 + 1
            lines = [
                f"v_mov_b32 v{s0}, {lo:#x}",
                f"v_mov_b32 v{s1}, {hi:#x}",
                f"v_ashrrev_i64 {dst_reg}, 4, {src_reg}",
                f"v_mov_b32 v0, v{d0}",
                f"v_mov_b32 v1, v{d1}",
            ]
            got = run_lines(lines)
            self.assertEqual(got, ref_ashr_i64(4, val), f"Failed for pair {src_reg} -> {dst_reg}")

    def test_shader_in_place_destination_aliasing(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)
        lines = [
            f"v_mov_b32 v0, {lo:#x}",
            f"v_mov_b32 v1, {hi:#x}",
            "v_ashrrev_i64 v[0:1], 8, v[0:1]",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(8, val))

        lines_v2 = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_ashrrev_i64 v[2:3], 12, v[2:3]",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, v3",
        ]
        got_v2 = run_lines(lines_v2)
        self.assertEqual(got_v2, ref_ashr_i64(12, val))

    def test_shader_partially_overlapping_register_pairs(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_ashrrev_i64 v[1:2], 4, v[2:3]",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, v2",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(4, val))

    def test_shader_chained_vgpr_pair_shifts(self) -> None:
        val = 0x8000000000000000
        lo, hi = decompose_u64(val)
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            "v_ashrrev_i64 v[4:5], 16, v[2:3]",
            "v_ashrrev_i64 v[0:1], 16, v[4:5]",
        ]
        got = run_lines(lines)
        self.assertEqual(got, ref_ashr_i64(32, val))

    def test_shader_shift_count_from_different_vgprs(self) -> None:
        val = 0xFEDCBA9876543210
        lo, hi = decompose_u64(val)
        for count_vgpr in ("v4", "v7", "v15", "v25"):
            lines = [
                f"v_mov_b32 {count_vgpr}, 24",
                f"v_mov_b32 v2, {lo:#x}",
                f"v_mov_b32 v3, {hi:#x}",
                f"v_ashrrev_i64 v[0:1], {count_vgpr}, v[2:3]",
            ]
            got = run_lines(lines)
            self.assertEqual(got, ref_ashr_i64(24, val), f"Failed with count in {count_vgpr}")


class TestSignPropagationExtremes(unittest.TestCase):
    """Sign-propagation extremes and cross-dword transition tests."""

    def test_negative_one_all_ones_invariance(self) -> None:
        neg_one = MASK64
        for shift in range(64):
            self.assertEqual(ref_ashr_i64(shift, neg_one), neg_one)

    def test_most_negative_int64_sign_expansion(self) -> None:
        min_int64 = SIGN64
        self.assertEqual(ref_ashr_i64(0, min_int64), min_int64)
        self.assertEqual(ref_ashr_i64(1, min_int64), 0xC000000000000000)
        self.assertEqual(ref_ashr_i64(2, min_int64), 0xE000000000000000)
        self.assertEqual(ref_ashr_i64(3, min_int64), 0xF000000000000000)
        self.assertEqual(ref_ashr_i64(31, min_int64), 0xFFFFFFFF00000000)
        self.assertEqual(ref_ashr_i64(32, min_int64), 0xFFFFFFFF80000000)
        self.assertEqual(ref_ashr_i64(33, min_int64), 0xFFFFFFFFC0000000)
        self.assertEqual(ref_ashr_i64(62, min_int64), 0xFFFFFFFFFFFFFFFE)
        self.assertEqual(ref_ashr_i64(63, min_int64), MASK64)

        for shift in range(64):
            expected = (MASK64 << (63 - shift)) & MASK64
            self.assertEqual(ref_ashr_i64(shift, min_int64), expected)

    def test_max_positive_int64_zero_expansion(self) -> None:
        max_int64 = 0x7FFFFFFFFFFFFFFF
        self.assertEqual(ref_ashr_i64(0, max_int64), max_int64)
        self.assertEqual(ref_ashr_i64(1, max_int64), 0x3FFFFFFFFFFFFFFF)
        self.assertEqual(ref_ashr_i64(32, max_int64), 0x000000007FFFFFFF)
        self.assertEqual(ref_ashr_i64(62, max_int64), 1)
        self.assertEqual(ref_ashr_i64(63, max_int64), 0)

    def test_zero_expansion(self) -> None:
        for shift in range(64):
            self.assertEqual(ref_ashr_i64(shift, 0), 0)

    def test_positive_one_drain(self) -> None:
        self.assertEqual(ref_ashr_i64(0, 1), 1)
        for shift in range(1, 64):
            self.assertEqual(ref_ashr_i64(shift, 1), 0)

    def test_cross_dword_transition_31_32_33(self) -> None:
        patterns = [
            0x8000000000000000,
            0x7FFFFFFFFFFFFFFF,
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
            0x00000000FFFFFFFF,
            0xFFFFFFFF00000000,
        ]
        for val in patterns:
            lo, hi = decompose_u64(val)
            for shift in (31, 32, 33):
                ref = ref_ashr_i64(shift, val)
                elo, ehi = emulate_ashr_i64_dwords(shift, lo, hi)
                self.assertEqual(compose_u64(elo, ehi), ref, f"Mismatch at val={hex(val)}, shift={shift}")

    def test_shader_sign_propagation_extremes(self) -> None:
        val = SIGN64
        self.assertEqual(run_ashr_op(1, val), 0xC000000000000000)
        self.assertEqual(run_ashr_op(31, val), 0xFFFFFFFF00000000)
        self.assertEqual(run_ashr_op(32, val), 0xFFFFFFFF80000000)
        self.assertEqual(run_ashr_op(63, val), MASK64)


class TestShiftCountModulo64(unittest.TestCase):
    """Modulo-64 shift count masking tests."""

    def test_modulo_64_masking_comprehensive(self) -> None:
        test_values = [
            0x8000000000000000,
            0x7FFFFFFFFFFFFFFF,
            0xFEDCBA9876543210,
            0x0123456789ABCDEF,
        ]
        base_shifts = [0, 1, 15, 31, 32, 47, 63]
        multipliers = [1, 2, 3, 10, 100]

        for val in test_values:
            for base in base_shifts:
                ref = ref_ashr_i64(base, val)
                for mult in multipliers:
                    overflowed_shift = base + 64 * mult
                    self.assertEqual(ref_ashr_i64(overflowed_shift, val), ref)

    def test_upper_bits_ignored(self) -> None:
        val = 0x8000000000000000
        self.assertEqual(ref_ashr_i64(0xFFFFFFFF, val), ref_ashr_i64(63, val))
        self.assertEqual(ref_ashr_i64(0x80000000, val), ref_ashr_i64(0, val))
        self.assertEqual(ref_ashr_i64(0x80000020, val), ref_ashr_i64(32, val))
        self.assertEqual(ref_ashr_i64(0x12345600, val), ref_ashr_i64(0, val))
        self.assertEqual(ref_ashr_i64(0x1234563F, val), ref_ashr_i64(63, val))

    def test_shader_shift_count_modulo_64(self) -> None:
        val = 0x8000000000000000
        self.assertEqual(run_ashr_op(64, val), val)
        self.assertEqual(run_ashr_op(65, val), ref_ashr_i64(1, val))
        self.assertEqual(run_ashr_op(0x80000020, val), ref_ashr_i64(32, val))
        self.assertEqual(run_ashr_op(0xFFFFFFFF, val), ref_ashr_i64(63, val))


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated semantics and missing masks fail assertions."""

    def test_logical_shift_mutation_rejected(self) -> None:
        val = 0x8000000000000000
        shift = 1
        arithmetic_correct = ref_ashr_i64(shift, val)
        mutated_logical = (u64(val) >> (shift & 63)) & MASK64
        self.assertEqual(arithmetic_correct, 0xC000000000000000)
        self.assertEqual(mutated_logical, 0x4000000000000000)
        self.assertNotEqual(arithmetic_correct, mutated_logical)

    def test_unmasked_shift_mutation_rejected(self) -> None:
        val = 0x8000000000000000
        shift = 64
        masked_correct = ref_ashr_i64(shift, val)
        mutated_unmasked = (s64(val) >> shift) & MASK64
        self.assertEqual(masked_correct, val)
        self.assertEqual(mutated_unmasked, MASK64)
        self.assertNotEqual(masked_correct, mutated_unmasked)

    def test_left_shift_mutation_rejected(self) -> None:
        val = 0x0000000000000001
        shift = 4
        arithmetic_correct = ref_ashr_i64(shift, val)
        mutated_left = (s64(val) << (shift & 63)) & MASK64
        self.assertEqual(arithmetic_correct, 0)
        self.assertEqual(mutated_left, 0x10)
        self.assertNotEqual(arithmetic_correct, mutated_left)

    def test_independent_dword_shift_mutation_rejected(self) -> None:
        val = 0x0000000100000000
        shift = 1
        correct = ref_ashr_i64(shift, val)
        lo, hi = decompose_u64(val)
        mutated_independent_lo = (lo >> shift) & MASK32
        mutated_independent_hi = (hi >> shift) & MASK32
        mutated = compose_u64(mutated_independent_lo, mutated_independent_hi)
        self.assertEqual(correct, 0x0000000080000000)
        self.assertEqual(mutated, 0x0000000000000000)
        self.assertNotEqual(correct, mutated)

    def test_sign_bit_inversion_mutation_rejected(self) -> None:
        val = 0x8000000000000000
        correct = ref_ashr_i64(1, val)
        mutated = correct ^ SIGN64
        self.assertNotEqual(correct, mutated)

    def test_off_by_one_shift_count_mutation_rejected(self) -> None:
        val = 0x8000000000000000
        correct = ref_ashr_i64(4, val)
        mutated_plus = ref_ashr_i64(5, val)
        mutated_minus = ref_ashr_i64(3, val)
        self.assertNotEqual(correct, mutated_plus)
        self.assertNotEqual(correct, mutated_minus)


if __name__ == "__main__":
    unittest.main()
