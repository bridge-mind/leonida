"""Semantic interpreter tests for AMD RDNA vector integer subtraction.

Instructions covered:
  - v_sub_u32: D.u32 = (S0.u32 - S1.u32) & 0xFFFFFFFF
  - v_subrev_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF
  - v_sub_nc_u32: D.u32 = (S0.u32 - S1.u32) & 0xFFFFFFFF (RDNA non-carry vector subtract)
  - v_subrev_nc_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF (RDNA non-carry vector reverse subtract)

Models VALU two's complement integer subtraction, reverse subtraction, SIMD lane
masking via EXEC, inline constant operands, SGPR source scalar broadcast, borrow
propagation for multi-word math, and lowering to SPIR-V OpISub via tools/rdna2spirv.py
executed in tests/semantic/spirv_mini_interp.py.

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

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_v_sub_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_sub_u32 / v_sub_nc_u32: S0 - S1 (mod 2^32)."""
    return ((s0 & MASK32) - (s1 & MASK32)) & MASK32


def ref_v_subrev_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_subrev_u32 / v_subrev_nc_u32: S1 - S0 (mod 2^32)."""
    return ((s1 & MASK32) - (s0 & MASK32)) & MASK32


def ref_v_sub_nc_u32(s0: int, s1: int) -> int:
    """Alias for ref_v_sub_u32."""
    return ref_v_sub_u32(s0, s1)


def ref_v_subrev_nc_u32(s0: int, s1: int) -> int:
    """Alias for ref_v_subrev_u32."""
    return ref_v_subrev_u32(s0, s1)


def ref_borrow_u32(s0: int, s1: int) -> int:
    """Borrow out flag for S0 - S1 (1 if S0 < S1 unsigned, else 0)."""
    return 1 if (s0 & MASK32) < (s1 & MASK32) else 0


def ref_borrow_rev_u32(s0: int, s1: int) -> int:
    """Borrow out flag for reverse subtraction S1 - S0 (1 if S1 < S0 unsigned, else 0)."""
    return 1 if (s1 & MASK32) < (s0 & MASK32) else 0


def to_signed_i32(val: int) -> int:
    """Interpret 32-bit bit pattern as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


class VectorALUInterpreter:
    """Deterministic semantic interpreter for AMD RDNA vector integer subtraction.

    Models:
      - 64-lane SIMD vector registers (VGPRs)
      - Scalar registers (SGPRs)
      - EXEC mask gating active lanes (inactive lanes preserve prior contents)
      - Support for inline constants and register operands
    """

    def __init__(self, num_vgprs: int = 64, num_sgprs: int = 106, wave_size: int = 64):
        self.num_vgprs = num_vgprs
        self.num_sgprs = num_sgprs
        self.wave_size = wave_size
        self.vgprs = [[0] * wave_size for _ in range(num_vgprs)]
        self.sgprs = [0] * num_sgprs
        self.exec_mask = (1 << wave_size) - 1

    def reset(self) -> None:
        self.vgprs = [[0] * self.wave_size for _ in range(self.num_vgprs)]
        self.sgprs = [0] * self.num_sgprs
        self.exec_mask = (1 << self.wave_size) - 1

    def set_vgpr(self, reg_idx: int, lane: int, val: int) -> None:
        self.vgprs[reg_idx][lane] = val & MASK32

    def get_vgpr(self, reg_idx: int, lane: int) -> int:
        return self.vgprs[reg_idx][lane] & MASK32

    def set_sgpr(self, reg_idx: int, val: int) -> None:
        self.sgprs[reg_idx] = val & MASK32

    def get_sgpr(self, reg_idx: int) -> int:
        return self.sgprs[reg_idx] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def resolve_src(self, op: str | int, lane: int) -> int:
        if isinstance(op, int):
            return op & MASK32
        op_str = op.strip().lower()
        if op_str.startswith("v"):
            idx = int(op_str[1:])
            return self.vgprs[idx][lane] & MASK32
        if op_str.startswith("s"):
            idx = int(op_str[1:])
            return self.sgprs[idx] & MASK32
        return int(op_str, 0) & MASK32

    def execute_v_sub_u32(self, dst_idx: int, src0: str | int, src1: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_v_sub_u32(s0, s1)

    def execute_v_subrev_u32(self, dst_idx: int, src0: str | int, src1: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_v_subrev_u32(s0, s1)

    def execute_v_sub_nc_u32(self, dst_idx: int, src0: str | int, src1: str | int) -> None:
        self.execute_v_sub_u32(dst_idx, src0, src1)

    def execute_v_subrev_nc_u32(self, dst_idx: int, src0: str | int, src1: str | int) -> None:
        self.execute_v_subrev_u32(dst_idx, src0, src1)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sub.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sub.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_sub_shader(
    op: str,
    s0: int,
    s1: int,
    inline_s0: bool = False,
    inline_s1: bool = False,
    s0_is_sgpr: bool = False,
) -> int:
    """Translates and runs a synthetic pixel shader invoking the subtraction operation."""
    if _TMP_DIR is None:
        setUpModule()
    lines = []
    if s0_is_sgpr:
        lines.append(f"\ts_mov_b32 s0, {s0 & MASK32:#x}")
        op0 = "s0"
    elif inline_s0:
        op0 = f"{s0 & MASK32:#x}"
    else:
        lines.append(f"\tv_mov_b32 v1, {s0 & MASK32:#x}")
        op0 = "v1"

    if inline_s1:
        op1 = f"{s1 & MASK32:#x}"
    else:
        lines.append(f"\tv_mov_b32 v2, {s1 & MASK32:#x}")
        op1 = "v2"

    lines.append(f"\t{op} v0, {op0}, {op1}")
    asm = (
        "; test_alu_sub\n"
        + "\n".join(lines)
        + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class VectorSubSemanticReferencesTest(unittest.TestCase):
    """Tests the mathematical references for vector integer subtraction."""

    def test_basic_subtraction(self):
        self.assertEqual(ref_v_sub_u32(100, 30), 70)
        self.assertEqual(ref_v_subrev_u32(100, 30), (30 - 100) & MASK32)
        self.assertEqual(ref_v_subrev_u32(100, 30), 0xFFFFFFBA)

    def test_identity_properties(self):
        values = [0, 1, 42, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        for v in values:
            self.assertEqual(ref_v_sub_u32(v, 0), v)
            self.assertEqual(ref_v_subrev_u32(0, v), v)
            self.assertEqual(ref_v_sub_u32(v, v), 0)
            self.assertEqual(ref_v_subrev_u32(v, v), 0)
            self.assertEqual(ref_v_sub_u32(0, v), (-v) & MASK32)
            self.assertEqual(ref_v_subrev_u32(v, 0), (-v) & MASK32)

    def test_anti_symmetry_and_reversal_duality(self):
        pairs = [
            (50, 20),
            (20, 50),
            (0, 1),
            (0xFFFFFFFF, 1),
            (0x80000000, 0x7FFFFFFF),
        ]
        for a, b in pairs:
            fwd = ref_v_sub_u32(a, b)
            rev = ref_v_subrev_u32(a, b)
            self.assertEqual((fwd + rev) & MASK32, 0)
            self.assertEqual(ref_v_sub_u32(a, b), ref_v_subrev_u32(b, a))
            self.assertEqual(ref_v_sub_nc_u32(a, b), ref_v_sub_u32(a, b))
            self.assertEqual(ref_v_subrev_nc_u32(a, b), ref_v_subrev_u32(a, b))

    def test_signed_two_complement_equivalence(self):
        pairs = [
            (10, 25),
            (0x7FFFFFFF, 1),
            (0x80000000, 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0, 0xFFFFFFFF),
        ]
        for a, b in pairs:
            u_diff = ref_v_sub_u32(a, b)
            sa = to_signed_i32(a)
            sb = to_signed_i32(b)
            s_diff = (sa - sb) & MASK32
            self.assertEqual(u_diff, s_diff)

    def test_borrow_flags(self):
        self.assertEqual(ref_borrow_u32(10, 5), 0)
        self.assertEqual(ref_borrow_u32(5, 10), 1)
        self.assertEqual(ref_borrow_u32(0, 0), 0)
        self.assertEqual(ref_borrow_u32(0, 1), 1)
        self.assertEqual(ref_borrow_u32(0xFFFFFFFF, 0xFFFFFFFF), 0)
        self.assertEqual(ref_borrow_u32(0xFFFFFFFF, 0), 0)
        self.assertEqual(ref_borrow_u32(0, 0xFFFFFFFF), 1)

        self.assertEqual(ref_borrow_rev_u32(10, 5), 1)
        self.assertEqual(ref_borrow_rev_u32(5, 10), 0)


class VectorSubBoundaryValuesTest(unittest.TestCase):
    """Tests extreme and boundary cases for 32-bit vector subtraction."""

    def test_extremal_differences(self):
        self.assertEqual(ref_v_sub_u32(0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_v_sub_u32(1, 0), 1)
        self.assertEqual(ref_v_sub_u32(0x7FFFFFFF, 0x80000000), 0xFFFFFFFF)
        self.assertEqual(ref_v_sub_u32(0x80000000, 0x7FFFFFFF), 1)
        self.assertEqual(ref_v_sub_u32(0x80000000, 1), 0x7FFFFFFF)
        self.assertEqual(ref_v_sub_u32(0x7FFFFFFF, 0xFFFFFFFF), 0x80000000)
        self.assertEqual(ref_v_sub_u32(0xFFFFFFFF, 0x7FFFFFFF), 0x80000000)
        self.assertEqual(ref_v_sub_u32(0, 0xFFFFFFFF), 1)

    def test_reverse_extremal_differences(self):
        self.assertEqual(ref_v_subrev_u32(1, 0), 0xFFFFFFFF)
        self.assertEqual(ref_v_subrev_u32(0, 1), 1)
        self.assertEqual(ref_v_subrev_u32(0x80000000, 0x7FFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_v_subrev_u32(0x7FFFFFFF, 0x80000000), 1)
        self.assertEqual(ref_v_subrev_u32(1, 0x80000000), 0x7FFFFFFF)


class VectorALUInterpreterTest(unittest.TestCase):
    """Tests the stateful VectorALUInterpreter modeling multi-lane SIMD execution and EXEC gating."""

    def test_multi_lane_subtraction(self):
        interp = VectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(1, lane, lane * 10)
            interp.set_vgpr(2, lane, lane * 3)

        interp.execute_v_sub_u32(0, "v1", "v2")
        for lane in range(64):
            expected = (lane * 10 - lane * 3) & MASK32
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_multi_lane_subrev(self):
        interp = VectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(1, lane, lane * 10)
            interp.set_vgpr(2, lane, lane * 3)

        interp.execute_v_subrev_u32(0, "v1", "v2")
        for lane in range(64):
            expected = (lane * 3 - lane * 10) & MASK32
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_exec_mask_inactive_lanes_preserved(self):
        interp = VectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0xDEADBEEF)
            interp.set_vgpr(1, lane, 100)
            interp.set_vgpr(2, lane, 40)

        # Alternating mask: only even lanes active
        interp.set_exec(0x5555555555555555)
        interp.execute_v_sub_nc_u32(0, "v1", "v2")

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), 60)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0xDEADBEEF)

    def test_scalar_and_constant_operands(self):
        interp = VectorALUInterpreter(wave_size=32)
        interp.set_sgpr(0, 500)
        for lane in range(32):
            interp.set_vgpr(1, lane, lane)

        # v_sub_nc_u32 v0, s0, v1 -> 500 - lane
        interp.execute_v_sub_nc_u32(0, "s0", "v1")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 500 - lane)

        # v_subrev_nc_u32 v0, 10, v1 -> lane - 10
        interp.execute_v_subrev_nc_u32(0, 10, "v1")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), (lane - 10) & MASK32)


class VectorSubSpirvLoweringTest(unittest.TestCase):
    """Tests shader assembly translation through rdna2spirv and execution in spirv_mini_interp."""

    def test_v_sub_nc_u32_default_vop2(self):
        res = run_sub_shader("v_sub_nc_u32", 100, 30)
        self.assertEqual(res, 70)

    def test_v_subrev_nc_u32_default_vop2(self):
        res = run_sub_shader("v_subrev_nc_u32", 100, 30)
        self.assertEqual(res, (30 - 100) & MASK32)

    def test_v_sub_nc_u32_e32(self):
        res = run_sub_shader("v_sub_nc_u32_e32", 200, 75)
        self.assertEqual(res, 125)

    def test_v_subrev_nc_u32_e32(self):
        res = run_sub_shader("v_subrev_nc_u32_e32", 200, 75)
        self.assertEqual(res, (75 - 200) & MASK32)

    def test_v_sub_nc_u32_e64(self):
        res = run_sub_shader("v_sub_nc_u32_e64", 0x80000000, 1)
        self.assertEqual(res, 0x7FFFFFFF)

    def test_v_subrev_nc_u32_e64(self):
        res = run_sub_shader("v_subrev_nc_u32_e64", 1, 0x80000000)
        self.assertEqual(res, 0x7FFFFFFF)

    def test_inline_constants(self):
        res = run_sub_shader("v_sub_nc_u32", 5, 20, inline_s0=True)
        self.assertEqual(res, (5 - 20) & MASK32)

        res = run_sub_shader("v_subrev_nc_u32", 5, 20, inline_s0=True)
        self.assertEqual(res, (20 - 5) & MASK32)

        res = run_sub_shader("v_sub_nc_u32", 0, 1, inline_s0=True)
        self.assertEqual(res, 0xFFFFFFFF)

        res = run_sub_shader("v_subrev_nc_u32", 0, 1, inline_s0=True)
        self.assertEqual(res, 1)

    def test_sgpr_scalar_source(self):
        res = run_sub_shader("v_sub_nc_u32", 50, 15, s0_is_sgpr=True)
        self.assertEqual(res, 35)

        res = run_sub_shader("v_subrev_nc_u32", 50, 15, s0_is_sgpr=True)
        self.assertEqual(res, (15 - 50) & MASK32)

    def test_emitted_spirv_contains_op_isub(self):
        if _TMP_DIR is None:
            setUpModule()
        lines = [
            "\tv_mov_b32 v1, 80",
            "\tv_mov_b32 v2, 35",
            "\tv_sub_nc_u32 v0, v1, v2",
            "\texp mrt0 v0, v0, v0, v0 done vm",
            "\ts_endpgm",
        ]
        asm = "; test_op_check\n" + "\n".join(lines) + "\n"
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        self.assertIn("OpISub %uint", spv)


class VectorSub64BitBorrowChainingTest(unittest.TestCase):
    """Tests 64-bit multi-word vector integer subtraction using borrow tracking."""

    def emulate_v_sub_u64(self, a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> tuple[int, int]:
        diff_lo = ref_v_sub_nc_u32(a_lo, b_lo)
        borrow = ref_borrow_u32(a_lo, b_lo)
        diff_hi = (ref_v_sub_nc_u32(a_hi, b_hi) - borrow) & MASK32
        return diff_lo, diff_hi

    def test_chained_64bit_subtraction(self):
        test_cases = [
            (0x00000000_00000005, 0x00000000_00000002),
            (0x00000001_00000000, 0x00000000_00000001),
            (0x00000000_00000000, 0x00000000_00000001),
            (0x12345678_00000000, 0x00000000_FFFFFFFF),
            (0xFFFFFFFF_FFFFFFFF, 0x00000000_00000001),
            (0x80000000_00000000, 0x7FFFFFFF_FFFFFFFF),
        ]
        for a64, b64 in test_cases:
            a_lo, a_hi = a64 & MASK32, (a64 >> 32) & MASK32
            b_lo, b_hi = b64 & MASK32, (b64 >> 32) & MASK32
            d_lo, d_hi = self.emulate_v_sub_u64(a_lo, a_hi, b_lo, b_hi)
            res64 = ((d_hi & MASK32) << 32) | (d_lo & MASK32)
            expected64 = (a64 - b64) & MASK64
            self.assertEqual(res64, expected64)


class VectorSubDifferentialFuzzTest(unittest.TestCase):
    """Deterministic fuzz testing against reference models and SPIR-V pipeline."""

    def test_differential_random_pairs(self):
        rng = random.Random(0x50B)
        interp = VectorALUInterpreter(wave_size=1)

        for _ in range(120):
            a = rng.randint(0, MASK32)
            b = rng.randint(0, MASK32)

            expected_sub = (a - b) & MASK32
            expected_subrev = (b - a) & MASK32

            self.assertEqual(ref_v_sub_u32(a, b), expected_sub)
            self.assertEqual(ref_v_sub_nc_u32(a, b), expected_sub)
            self.assertEqual(ref_v_subrev_u32(a, b), expected_subrev)
            self.assertEqual(ref_v_subrev_nc_u32(a, b), expected_subrev)

            interp.set_vgpr(1, 0, a)
            interp.set_vgpr(2, 0, b)
            interp.execute_v_sub_nc_u32(0, "v1", "v2")
            self.assertEqual(interp.get_vgpr(0, 0), expected_sub)

            interp.execute_v_subrev_nc_u32(0, "v1", "v2")
            self.assertEqual(interp.get_vgpr(0, 0), expected_subrev)

        # Cross-check a sample subset through rdna2spirv and spirv_mini_interp
        sample_pairs = [
            (rng.randint(0, MASK32), rng.randint(0, MASK32))
            for _ in range(8)
        ]
        for a, b in sample_pairs:
            spv_sub = run_sub_shader("v_sub_nc_u32", a, b)
            self.assertEqual(spv_sub, (a - b) & MASK32)
            spv_subrev = run_sub_shader("v_subrev_nc_u32", a, b)
            self.assertEqual(spv_subrev, (b - a) & MASK32)


if __name__ == "__main__":
    unittest.main()
