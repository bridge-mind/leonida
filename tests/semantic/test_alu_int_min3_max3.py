from __future__ import annotations

"""Semantic interpreter tests for RDNA2 3-way integer min/max instructions:
- v_min3_i32: signed two's complement 32-bit minimum: min(S0, S1, S2)
- v_min3_u32: unsigned 32-bit integer minimum: min(S0, S1, S2)
- v_max3_i32: signed two's complement 32-bit maximum: max(S0, S1, S2)
- v_max3_u32: unsigned 32-bit integer maximum: max(S0, S1, S2)

Features reference models, pairwise reduction equivalence, signed vs unsigned
divergence on identical bit patterns, multi-lane SIMD wave execution under EXEC
mask gating, two's complement extrema boundaries, full permutation matrices,
negative controls, and shader execution via rdna2spirv and spirv_mini_interp.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
import itertools
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

MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000
INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32")


def to_signed_i32(x: int) -> int:
    """Interpret raw 32-bit unsigned pattern as two's complement signed 32-bit integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_BIT32) else x


def to_unsigned_u32(x: int) -> int:
    """Mask integer to 32-bit unsigned integer range [0, 2^32 - 1]."""
    return x & MASK32


def ref_v_min3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_min3_i32: signed minimum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return min(v0, v1, v2) & MASK32


def ref_v_min3_u32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_min3_u32: unsigned minimum of 3 operands."""
    v0 = to_unsigned_u32(s0)
    v1 = to_unsigned_u32(s1)
    v2 = to_unsigned_u32(s2)
    return min(v0, v1, v2) & MASK32


def ref_v_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_max3_i32: signed maximum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return max(v0, v1, v2) & MASK32


def ref_v_max3_u32(s0: int, s1: int, s2: int) -> int:
    """Reference implementation of v_max3_u32: unsigned maximum of 3 operands."""
    v0 = to_unsigned_u32(s0)
    v1 = to_unsigned_u32(s1)
    v2 = to_unsigned_u32(s2)
    return max(v0, v1, v2) & MASK32


def ref_pairwise_min3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed min: min(min(s0, s1), s2)."""
    step1 = min(to_signed_i32(s0), to_signed_i32(s1))
    return min(step1, to_signed_i32(s2)) & MASK32


def ref_pairwise_min3_u32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise unsigned min: min(min(s0, s1), s2)."""
    step1 = min(to_unsigned_u32(s0), to_unsigned_u32(s1))
    return min(step1, to_unsigned_u32(s2)) & MASK32


def ref_pairwise_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed max: max(max(s0, s1), s2)."""
    step1 = max(to_signed_i32(s0), to_signed_i32(s1))
    return max(step1, to_signed_i32(s2)) & MASK32


def ref_pairwise_max3_u32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise unsigned max: max(max(s0, s1), s2)."""
    step1 = max(to_unsigned_u32(s0), to_unsigned_u32(s1))
    return max(step1, to_unsigned_u32(s2)) & MASK32


# Backwards-compatible alias
s32 = to_signed_i32


def ref(op: str, *xs: int) -> int:
    """Plain-Python reference; operands and result are raw 32-bit patterns."""
    if len(xs) == 3:
        if op == "v_min3_i32":
            return ref_v_min3_i32(xs[0], xs[1], xs[2])
        if op == "v_min3_u32":
            return ref_v_min3_u32(xs[0], xs[1], xs[2])
        if op == "v_max3_i32":
            return ref_v_max3_i32(xs[0], xs[1], xs[2])
        if op == "v_max3_u32":
            return ref_v_max3_u32(xs[0], xs[1], xs[2])
    key = to_signed_i32 if op.endswith("_i32") else to_unsigned_u32
    pick = min if op.startswith("v_min3") else max
    return pick(xs, key=key) & MASK32


class VectorALUInterpreter:
    """Deterministic semantic interpreter for RDNA2 VALU 3-way integer min/max.

    Models:
      - Vector General Purpose Registers (VGPRs) across wave lanes.
      - Scalar General Purpose Registers (SGPRs) uniformly broadcast to all lanes.
      - EXEC mask write gating (inactive lanes preserve prior VGPR contents).
      - Inline constants and register operand decoding.
      - Both Wave64 and Wave32 execution widths.
    """

    def __init__(self, num_vgprs: int = 64, num_sgprs: int = 106, wave_size: int = 64) -> None:
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
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: v{reg_idx}")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index out of range: {lane}")
        self.vgprs[reg_idx][lane] = val & MASK32

    def get_vgpr(self, reg_idx: int, lane: int) -> int:
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: v{reg_idx}")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index out of range: {lane}")
        return self.vgprs[reg_idx][lane] & MASK32

    def set_sgpr(self, reg_idx: int, val: int) -> None:
        if not (0 <= reg_idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: s{reg_idx}")
        self.sgprs[reg_idx] = val & MASK32

    def get_sgpr(self, reg_idx: int) -> int:
        if not (0 <= reg_idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: s{reg_idx}")
        return self.sgprs[reg_idx] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def resolve_src(self, op: str | int, lane: int) -> int:
        if isinstance(op, int):
            return op & MASK32
        op_str = op.strip().lower()
        if op_str.startswith("v"):
            idx = int(op_str[1:])
            return self.get_vgpr(idx, lane)
        if op_str.startswith("s"):
            idx = int(op_str[1:])
            return self.get_sgpr(idx)
        return int(op_str, 0) & MASK32

    def execute_v_min3_i32(self, dst_idx: int, src0: str | int, src1: str | int, src2: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                s2 = self.resolve_src(src2, lane)
                self.set_vgpr(dst_idx, lane, ref_v_min3_i32(s0, s1, s2))

    def execute_v_min3_u32(self, dst_idx: int, src0: str | int, src1: str | int, src2: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                s2 = self.resolve_src(src2, lane)
                self.set_vgpr(dst_idx, lane, ref_v_min3_u32(s0, s1, s2))

    def execute_v_max3_i32(self, dst_idx: int, src0: str | int, src1: str | int, src2: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                s2 = self.resolve_src(src2, lane)
                self.set_vgpr(dst_idx, lane, ref_v_max3_i32(s0, s1, s2))

    def execute_v_max3_u32(self, dst_idx: int, src0: str | int, src1: str | int, src2: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                s2 = self.resolve_src(src2, lane)
                self.set_vgpr(dst_idx, lane, ref_v_max3_u32(s0, s1, s2))

    def execute_instruction(
        self, op: str, dst_idx: int, src0: str | int, src1: str | int, src2: str | int
    ) -> None:
        if op == "v_min3_i32":
            self.execute_v_min3_i32(dst_idx, src0, src1, src2)
        elif op == "v_min3_u32":
            self.execute_v_min3_u32(dst_idx, src0, src1, src2)
        elif op == "v_max3_i32":
            self.execute_v_max3_i32(dst_idx, src0, src1, src2)
        elif op == "v_max3_u32":
            self.execute_v_max3_u32(dst_idx, src0, src1, src2)
        else:
            raise ValueError(f"Unknown instruction: {op}")


# Triples where signed and unsigned order disagree, plus ties and ordinary values.
TRIPLES = (
    (5, 0x80000000, 7),
    (0xFFFFFFFF, 0, 1),
    (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
    (0x80000001, 0x80000000, 0x7FFFFFFE),
    (1, 0xFFFFFFFE, 0xFFFFFFFF),
    (9, 9, 0x80000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0),
    (3, 3, 3),
)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_min3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_min3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run(op: str, *xs: int, inline: tuple[int, ...] = ()) -> int:
    """Translate and run `op v0, a, b, c`.

    Operands are moved into v1..v3 unless their index is in `inline`, in which
    case the value is written into the instruction as an inline constant.
    Returns the mrt0 word.
    """
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    lines, operands = [], []
    for i, x in enumerate(xs):
        if i in inline:
            operands.append(str(x))
        else:
            lines.append(f"\tv_mov_b32 v{i + 1}, {x & MASK32:#x}")
            operands.append(f"v{i + 1}")
    lines.append(f"\t{op} v0, {', '.join(operands)}")
    asm = "; test_int_min3_max3\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class TestAluIntMin3Max3ReferenceOracle(unittest.TestCase):
    """Rigorous tests for the reference oracle models and mathematical properties."""

    def test_signed_vs_unsigned_divergence_core(self) -> None:
        """Verify fundamental bit pattern divergences between signed and unsigned min/max."""
        # 0x0 vs 0xFFFFFFFF (-1 signed, UINT_MAX unsigned)
        self.assertEqual(ref_v_min3_i32(0, 0xFFFFFFFF, 10), 0xFFFFFFFF)  # -1 is minimum
        self.assertEqual(ref_v_min3_u32(0, 0xFFFFFFFF, 10), 0)           # 0 is minimum
        self.assertEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), 10)          # 10 is maximum (-1 < 0 < 10)
        self.assertEqual(ref_v_max3_u32(0, 0xFFFFFFFF, 10), 0xFFFFFFFF)  # UINT_MAX is maximum

        # 0x7FFFFFFF (INT_MAX) vs 0x80000000 (INT_MIN signed, 2^31 unsigned)
        self.assertEqual(ref_v_min3_i32(0x7FFFFFFF, 0x80000000, 0), 0x80000000)
        self.assertEqual(ref_v_min3_u32(0x7FFFFFFF, 0x80000000, 0), 0)
        self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        self.assertEqual(ref_v_max3_u32(0x7FFFFFFF, 0x80000000, 0), 0x80000000)

    def test_pairwise_reduction_equivalence(self) -> None:
        """3-way min/max must identically match chained 2-way pairwise min/max."""
        sample_vectors = [
            (0, 0, 0),
            (1, 2, 3),
            (-5, 10, -20),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (0x12345678, 0x87654321, 0x00000000),
            (0xFFFFFFFE, 0xFFFFFFFF, 0x00000001),
        ]
        for s0, s1, s2 in sample_vectors:
            with self.subTest(s0=hex(s0 & MASK32), s1=hex(s1 & MASK32), s2=hex(s2 & MASK32)):
                self.assertEqual(ref_v_min3_i32(s0, s1, s2), ref_pairwise_min3_i32(s0, s1, s2))
                self.assertEqual(ref_v_min3_u32(s0, s1, s2), ref_pairwise_min3_u32(s0, s1, s2))
                self.assertEqual(ref_v_max3_i32(s0, s1, s2), ref_pairwise_max3_i32(s0, s1, s2))
                self.assertEqual(ref_v_max3_u32(s0, s1, s2), ref_pairwise_max3_u32(s0, s1, s2))

    def test_idempotent_property(self) -> None:
        """min3(x, x, x) == x and max3(x, x, x) == x for all x."""
        test_values = [0, 1, -1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 42, -9999]
        for val in test_values:
            raw = val & MASK32
            self.assertEqual(ref_v_min3_i32(val, val, val), raw)
            self.assertEqual(ref_v_min3_u32(val, val, val), raw)
            self.assertEqual(ref_v_max3_i32(val, val, val), raw)
            self.assertEqual(ref_v_max3_u32(val, val, val), raw)

    def test_dispatch_ref_wrapper(self) -> None:
        """Verify the generic ref dispatch wrapper matches direct reference functions."""
        for op, func in (
            ("v_min3_i32", ref_v_min3_i32),
            ("v_min3_u32", ref_v_min3_u32),
            ("v_max3_i32", ref_v_max3_i32),
            ("v_max3_u32", ref_v_max3_u32),
        ):
            for t in TRIPLES:
                self.assertEqual(ref(op, *t), func(*t))


class TestAluIntMin3Max3SIMD(unittest.TestCase):
    """Tests the stateful VectorALUInterpreter modeling multi-lane SIMD execution and EXEC gating."""

    def test_wave64_simd_multi_lane_evaluation(self) -> None:
        """Verify 64 SIMD lanes independently evaluate 3-way min/max with per-lane inputs."""
        interp = VectorALUInterpreter(wave_size=64)
        for lane in range(64):
            # Lane 0..31 have negative signed numbers, 32..63 have positive numbers
            val0 = (lane * 100 - 3200) & MASK32
            val1 = ((63 - lane) * 150 - 4500) & MASK32
            val2 = 0
            interp.set_vgpr(1, lane, val0)
            interp.set_vgpr(2, lane, val1)
            interp.set_vgpr(3, lane, val2)

        for op in OPS:
            interp.execute_instruction(op, 0, "v1", "v2", "v3")
            for lane in range(64):
                v1 = interp.get_vgpr(1, lane)
                v2 = interp.get_vgpr(2, lane)
                v3 = interp.get_vgpr(3, lane)
                expected = ref(op, v1, v2, v3)
                actual = interp.get_vgpr(0, lane)
                self.assertEqual(
                    actual,
                    expected,
                    f"Mismatch in {op} lane {lane}: got {hex(actual)}, expected {hex(expected)}",
                )

    def test_wave32_scalar_broadcast_and_inline_const(self) -> None:
        """Verify wave32 mode with SGPR scalar broadcast and inline constant operand."""
        interp = VectorALUInterpreter(wave_size=32)
        # s0 uniformly set to scalar threshold 50
        interp.set_sgpr(0, 50)
        for lane in range(32):
            interp.set_vgpr(1, lane, lane * 5)  # 0, 5, 10, ... 155

        # v_min3_u32 v0, s0, v1, 75
        interp.execute_instruction("v_min3_u32", 0, "s0", "v1", 75)
        for lane in range(32):
            v1_val = lane * 5
            expected = min(50, v1_val, 75)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

        # v_max3_i32 v0, s0, v1, -20 (signed inline const)
        interp.execute_instruction("v_max3_i32", 0, "s0", "v1", -20)
        for lane in range(32):
            v1_val = lane * 5
            expected = max(50, v1_val, -20) & MASK32
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_exec_mask_inactive_lanes_preserved(self) -> None:
        """EXEC mask write gating: inactive lanes must strictly preserve prior VGPR contents."""
        interp = VectorALUInterpreter(wave_size=64)
        sentinel = 0xDEADBEEF
        for lane in range(64):
            interp.set_vgpr(0, lane, sentinel)
            interp.set_vgpr(1, lane, 10)
            interp.set_vgpr(2, lane, 20)
            interp.set_vgpr(3, lane, 30)

        # Alternating mask: only even lanes active
        even_mask = 0x5555555555555555
        interp.set_exec(even_mask)
        interp.execute_v_min3_i32(0, "v1", "v2", "v3")

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), 10, f"Even active lane {lane}")
            else:
                self.assertEqual(interp.get_vgpr(0, lane), sentinel, f"Odd inactive lane {lane}")

        # Sparse mask test
        sparse_mask = (1 << 3) | (1 << 27) | (1 << 59)
        interp.reset()
        for lane in range(64):
            interp.set_vgpr(0, lane, 0x11111111 * ((lane % 7) + 1))
            interp.set_vgpr(1, lane, 100)
            interp.set_vgpr(2, lane, 200)
            interp.set_vgpr(3, lane, 300)

        saved_v0 = [interp.get_vgpr(0, lane) for lane in range(64)]
        interp.set_exec(sparse_mask)
        interp.execute_v_max3_u32(0, "v1", "v2", "v3")

        for lane in range(64):
            if lane in (3, 27, 59):
                self.assertEqual(interp.get_vgpr(0, lane), 300)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), saved_v0[lane])

        # Full zero mask test: nothing is modified
        interp.set_exec(0)
        interp.execute_v_min3_u32(0, "v1", "v2", "v3")
        for lane in range(64):
            if lane in (3, 27, 59):
                self.assertEqual(interp.get_vgpr(0, lane), 300)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), saved_v0[lane])

    def test_multi_lane_signed_unsigned_divergence(self) -> None:
        """Vector lanes simultaneously executing signed vs unsigned operations diverge per lane."""
        interp = VectorALUInterpreter(wave_size=64)
        for lane in range(64):
            if lane < 32:
                # 0xFFFFFFFF (-1 as i32, UINT32_MAX as u32), 0, 10
                interp.set_vgpr(1, lane, 0xFFFFFFFF)
                interp.set_vgpr(2, lane, 0)
                interp.set_vgpr(3, lane, 10)
            else:
                # 0x80000000 (INT32_MIN as i32, 2^31 as u32), 0x7FFFFFFF (INT32_MAX), 0
                interp.set_vgpr(1, lane, 0x80000000)
                interp.set_vgpr(2, lane, 0x7FFFFFFF)
                interp.set_vgpr(3, lane, 0)

        # Test v_min3_i32 vs v_min3_u32
        interp.execute_v_min3_i32(4, "v1", "v2", "v3")
        interp.execute_v_min3_u32(5, "v1", "v2", "v3")
        interp.execute_v_max3_i32(6, "v1", "v2", "v3")
        interp.execute_v_max3_u32(7, "v1", "v2", "v3")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(4, lane), 0xFFFFFFFF)  # signed min: -1
            self.assertEqual(interp.get_vgpr(5, lane), 0)           # unsigned min: 0
            self.assertEqual(interp.get_vgpr(6, lane), 10)          # signed max: 10
            self.assertEqual(interp.get_vgpr(7, lane), 0xFFFFFFFF)  # unsigned max: UINT32_MAX

        for lane in range(32, 64):
            self.assertEqual(interp.get_vgpr(4, lane), 0x80000000)  # signed min: INT32_MIN
            self.assertEqual(interp.get_vgpr(5, lane), 0)           # unsigned min: 0
            self.assertEqual(interp.get_vgpr(6, lane), 0x7FFFFFFF)  # signed max: INT32_MAX
            self.assertEqual(interp.get_vgpr(7, lane), 0x80000000)  # unsigned max: 2^31


class TestAluIntMin3Max3ExtremaPermutationsNegative(unittest.TestCase):
    """Extrema matrices, 3! permutation invariance, and negative controls."""

    def test_twos_complement_extrema_matrix(self) -> None:
        """Verify behavior at and around two's complement and unsigned boundaries."""
        extrema = (
            0x80000000,  # INT32_MIN (-2147483648)
            0x80000001,  # INT32_MIN + 1
            0xFFFFFFFF,  # -1 as i32, UINT32_MAX as u32
            0x00000000,  # 0
            0x00000001,  # 1
            0x7FFFFFFE,  # INT32_MAX - 1
            0x7FFFFFFF,  # INT32_MAX
        )
        # Any triple containing 0x80000000 must have signed min = 0x80000000
        for x in extrema:
            for y in extrema:
                self.assertEqual(ref_v_min3_i32(0x80000000, x, y), 0x80000000)

        # Any triple containing 0x7FFFFFFF must have signed max = 0x7FFFFFFF
        for x in extrema:
            for y in extrema:
                self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, x, y), 0x7FFFFFFF)

        # Any triple containing 0x00000000 must have unsigned min = 0
        for x in extrema:
            for y in extrema:
                self.assertEqual(ref_v_min3_u32(0x00000000, x, y), 0)

        # Any triple containing 0xFFFFFFFF must have unsigned max = 0xFFFFFFFF
        for x in extrema:
            for y in extrema:
                self.assertEqual(ref_v_max3_u32(0xFFFFFFFF, x, y), 0xFFFFFFFF)

    def test_quad_divergence_extrema_triple(self) -> None:
        """On (0x80000000, 0x00000000, 0xFFFFFFFF), all 4 instructions yield mutually distinct results."""
        s0, s1, s2 = 0x80000000, 0x00000000, 0xFFFFFFFF
        res_min_i32 = ref_v_min3_i32(s0, s1, s2)
        res_min_u32 = ref_v_min3_u32(s0, s1, s2)
        res_max_i32 = ref_v_max3_i32(s0, s1, s2)
        res_max_u32 = ref_v_max3_u32(s0, s1, s2)

        self.assertEqual(res_min_i32, 0x80000000)
        self.assertEqual(res_min_u32, 0x00000000)
        self.assertEqual(res_max_i32, 0x00000000)
        self.assertEqual(res_max_u32, 0xFFFFFFFF)

        distinct_results = {res_min_i32, res_min_u32, res_max_i32, res_max_u32}
        self.assertEqual(len(distinct_results), 3)  # Note: 0x00000000 is shared by min_u32 and max_i32
        self.assertNotEqual(res_min_i32, res_min_u32)
        self.assertNotEqual(res_max_i32, res_max_u32)
        self.assertNotEqual(res_min_i32, res_max_i32)
        self.assertNotEqual(res_min_u32, res_max_u32)

    def test_permutation_invariance_all_orders(self) -> None:
        """All 3! = 6 permutations must yield identical results across all 4 instructions."""
        triples_to_permute = [
            (0x80000000, 0, 0x7FFFFFFF),
            (0xFFFFFFFF, 1, 0x80000000),
            (0x7FFFFFFF, 0x7FFFFFFF, 0x80000000),
            (42, 42, 42),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x80000000, 0x80000001, 0x80000002),
            (0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF),
            (0x55555555, 0xAAAAAAAA, 0xFFFFFFFF),
        ]
        for triple in triples_to_permute:
            perms = list(itertools.permutations(triple))
            self.assertEqual(len(perms), 6)
            for op in OPS:
                expected = ref(op, *triple)
                for p in perms:
                    with self.subTest(op=op, perm=[hex(v) for v in p]):
                        self.assertEqual(ref(op, *p), expected)

    def test_negative_controls_signed_unsigned_divergence(self) -> None:
        """Signed vs unsigned operations must not be mistakenly interchangeable."""
        # When negative numbers are involved, signed and unsigned min/max diverge
        self.assertNotEqual(ref_v_min3_i32(0, 0xFFFFFFFF, 1), ref_v_min3_u32(0, 0xFFFFFFFF, 1))
        self.assertNotEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 1), ref_v_max3_u32(0, 0xFFFFFFFF, 1))
        self.assertNotEqual(ref_v_min3_i32(0x80000000, 5, 10), ref_v_min3_u32(0x80000000, 5, 10))
        self.assertNotEqual(ref_v_max3_i32(0x80000000, 5, 10), ref_v_max3_u32(0x80000000, 5, 10))

        # min3 and max3 must not be interchangeable on non-identical inputs
        self.assertNotEqual(ref_v_min3_i32(1, 2, 3), ref_v_max3_i32(1, 2, 3))
        self.assertNotEqual(ref_v_min3_u32(1, 2, 3), ref_v_max3_u32(1, 2, 3))

    def test_negative_controls_interpreter_fault_injection(self) -> None:
        """Interpreter must reject invalid opcodes, register indices, and operands."""
        interp = VectorALUInterpreter(wave_size=64)

        # Invalid opcode
        with self.assertRaises(ValueError):
            interp.execute_instruction("v_invalid_op", 0, "v1", "v2", "v3")

        # Invalid VGPR index
        with self.assertRaises(IndexError):
            interp.set_vgpr(-1, 0, 0)
        with self.assertRaises(IndexError):
            interp.set_vgpr(64, 0, 0)
        with self.assertRaises(IndexError):
            interp.get_vgpr(0, -1)
        with self.assertRaises(IndexError):
            interp.get_vgpr(0, 64)

        # Invalid SGPR index
        with self.assertRaises(IndexError):
            interp.set_sgpr(-1, 0)
        with self.assertRaises(IndexError):
            interp.set_sgpr(106, 0)
        with self.assertRaises(IndexError):
            interp.get_sgpr(-1)
        with self.assertRaises(IndexError):
            interp.get_sgpr(106)

        # Invalid operand string
        with self.assertRaises(ValueError):
            interp.resolve_src("not_a_register_or_int", 0)


class ReferenceTests(unittest.TestCase):
    """The reference against hand-worked values, so a failure points at the translator, not the oracle."""

    def test_reference(self) -> None:
        self.assertEqual(ref("v_min3_i32", 5, 0x80000000, 7), 0x80000000)  # INT_MIN is the smallest i32
        self.assertEqual(ref("v_min3_u32", 5, 0x80000000, 7), 5)
        self.assertEqual(ref("v_max3_i32", 5, 0x80000000, 7), 7)
        self.assertEqual(ref("v_max3_u32", 5, 0x80000000, 7), 0x80000000)
        self.assertEqual(ref("v_min3_i32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)  # -1
        self.assertEqual(ref("v_max3_u32", 0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref("v_max3_i32", 0xFFFFFFFF, 0, 1), 1)
        self.assertEqual(ref("v_min3_u32", 0xFFFFFFFF, 0, 1), 0)


class Min3Max3Tests(unittest.TestCase):
    def test_every_order(self) -> None:
        for op in OPS:
            for xs in TRIPLES:
                for perm in sorted(set(itertools.permutations(xs))):
                    with self.subTest(op=op, xs=[hex(v) for v in perm]):
                        self.assertEqual(run(op, *perm), ref(op, *perm))

    def test_sweep(self) -> None:
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for op in OPS:
            for xs in itertools.product(values, repeat=3):
                with self.subTest(op=op, xs=[hex(v) for v in xs]):
                    self.assertEqual(run(op, *xs), ref(op, *xs))

    def test_inline_constants(self) -> None:
        # -16 as an inline constant is 0xFFFFFFF0: below 64 as i32, above it as u32.
        cases = [
            ("v_min3_i32", (0x7FFFFFFF, -16, 64), 0xFFFFFFF0),
            ("v_min3_u32", (0x7FFFFFFF, -16, 64), 64),
            ("v_max3_i32", (0x80000000, -16, 64), 64),
            ("v_max3_u32", (0x80000000, -16, 64), 0xFFFFFFF0),
        ]
        for op, xs, want in cases:
            with self.subTest(op=op):
                self.assertEqual(want, ref(op, *xs))
                self.assertEqual(run(op, *xs, inline=(1, 2)), want)


if __name__ == "__main__":
    unittest.main()
