"""Semantic interpreter tests for RDNA2 funnel shift bit alignment (v_alignbit_b32).

v_alignbit_b32 D, S0, S1, S2:
Concatenates S0 and S1 into a 64-bit value {S0, S1} (where S0 is the high 32 bits
and S1 is the low 32 bits), shifts right logically by the 5-bit shift count
(S2 & 31), and extracts the low 32 bits of the result:
    D = (({S0, S1} >> (S2 & 31)) & 0xFFFFFFFF)

Properties verified:
  * Full 5-bit shift range (shifts 0 through 31).
  * Register boundary crossing: bit propagation from S0 across the 32-bit dword
    boundary into the upper bits of the destination word as S1 bits shift out.
  * Shift amount masking: hardware masks shift amounts with 0x1F (bits 4:0), so
    shifts of 32, 33, 64, or negative / large integers wrap modulo 32.
  * Shift count zero: returns S1 (the low word) exactly without invoking undefined
    SPIR-V shift operations.
  * Common compiler idioms:
      - S0 == S1: 32-bit rotate right (ror).
      - S0 == 0:  logical shift right of S1 (lsr).
      - S1 == 0:  logical shift left of S0 by (32 - shift) for shift in 1..31.
  * Operand encoding forms: VGPR, SGPR, and inline integer constant shift counts.
  * Dual execution paths:
      1. RDNA2 assembly parsed and translated to SPIR-V via tools/rdna2spirv.py,
         then evaluated in tests/semantic/spirv_mini_interp.py.
      2. Isolated SPIR-V instruction sequence emitted for v_alignbit_b32 to check
         shift guard safety ((32 - m) & 31) against SPIR-V undefined shift rules.
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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_alignbit(src0: int, src1: int, shift: int) -> int:
    """RDNA2 v_alignbit_b32 reference oracle: low 32 bits of ({src0, src1} >> (shift & 31))."""
    data = ((src0 & MASK32) << 32) | (src1 & MASK32)
    return (data >> (shift & 31)) & MASK32


def interp_v_alignbit_b32(s0: int, s1: int, s2: int) -> int:
    """Bit-level model of funnel shift alignment decomposing high/low word contributions."""
    shift = s2 & 31
    s0 &= MASK32
    s1 &= MASK32
    if shift == 0:
        return s1
    lo_contrib = s1 >> shift
    hi_contrib = (s0 << (32 - shift)) & MASK32
    return lo_contrib | hi_contrib


def run_spirv_alignbit(s0: int, s1: int, s2: int) -> int:
    """Executes the exact SPIR-V instruction sequence emitted by rdna2spirv for v_alignbit_b32."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_hi = OpConstant %uint {s0 & MASK32}
%c_lo = OpConstant %uint {s1 & MASK32}
%c_sh = OpConstant %uint {s2 & MASK32}
%c31 = OpConstant %uint 31
%c32 = OpConstant %uint 32
%c0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpBitwiseAnd %uint %c_sh %c31
%inv0 = OpISub %uint %c32 %m
%inv = OpBitwiseAnd %uint %inv0 %c31
%a1 = OpShiftRightLogical %uint %c_lo %m
%b1 = OpShiftLeftLogical %uint %c_hi %inv
%r0 = OpBitwiseOr %uint %a1 %b1
%nz = OpINotEqual %bool %m %c0
%r = OpSelect %uint %nz %r0 %c_lo
OpStore %out %r
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_alignbit.s")
    with open(os.path.join(_TMP_DIR.name, "ps_alignbit.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run_alignbit(
    src0: int,
    src1: int,
    shift: int,
    inline_shift: bool = False,
    src0_sgpr: bool = False,
    src1_sgpr: bool = False,
    shift_sgpr: bool = False,
) -> int:
    """Translate and run synthetic RDNA2 shader with v_alignbit_b32, returning mrt0."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None

    lines: list[str] = []

    # Emit src0 operand
    if src0_sgpr:
        lines.append(f"\ts_mov_b32 s0, {src0 & MASK32:#x}")
        op_src0 = "s0"
    else:
        lines.append(f"\tv_mov_b32 v1, {src0 & MASK32:#x}")
        op_src0 = "v1"

    # Emit src1 operand
    if src1_sgpr:
        lines.append(f"\ts_mov_b32 s1, {src1 & MASK32:#x}")
        op_src1 = "s1"
    else:
        lines.append(f"\tv_mov_b32 v2, {src1 & MASK32:#x}")
        op_src1 = "v2"

    # Emit shift operand
    if inline_shift:
        op_shift = str(shift)
    elif shift_sgpr:
        lines.append(f"\ts_mov_b32 s2, {shift & MASK32:#x}")
        op_shift = "s2"
    else:
        lines.append(f"\tv_mov_b32 v3, {shift & MASK32:#x}")
        op_shift = "v3"

    lines.append(f"\tv_alignbit_b32 v0, {op_src0}, {op_src1}, {op_shift}")
    asm = "; test_alignbit\n" + "\n".join(lines) + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"

    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)

    return si.run_fragment(spv)["%out_mrt0"][0]


class ReferenceOracleTests(unittest.TestCase):
    """Verifies reference functions against hand-worked values and mathematical identities."""

    def test_reference_known_vectors(self) -> None:
        s0, s1 = 0x12345678, 0x9ABCDEF0
        # shift 0 -> low word unmodified
        self.assertEqual(ref_alignbit(s0, s1, 0), 0x9ABCDEF0)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 0), 0x9ABCDEF0)

        # shift 1 -> low word shifted right by 1, bit 0 of s0 enters bit 31
        # s1 >> 1 = 0x4D5E6F78, s0 & 1 = 0, so bit 31 is 0 -> 0x4D5E6F78
        self.assertEqual(ref_alignbit(s0, s1, 1), 0x4D5E6F78)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 1), 0x4D5E6F78)

        # shift 4 -> one hex nibble shifted across
        # {0x12345678, 0x9ABCDEF0} >> 4 = 0x0123456789ABCDEF
        self.assertEqual(ref_alignbit(s0, s1, 4), 0x89ABCDEF)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 4), 0x89ABCDEF)

        # shift 8 -> one byte shifted across
        self.assertEqual(ref_alignbit(s0, s1, 8), 0x789ABCDE)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 8), 0x789ABCDE)

        # shift 16 -> two bytes shifted across
        self.assertEqual(ref_alignbit(s0, s1, 16), 0x56789ABC)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 16), 0x56789ABC)

        # shift 24 -> three bytes shifted across
        self.assertEqual(ref_alignbit(s0, s1, 24), 0x3456789A)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 24), 0x3456789A)

        # shift 31 -> 31 bits from s0, 1 bit from s1
        # {0x12345678, 0x9ABCDEF0} >> 31 = 0x2468ACF1
        self.assertEqual(ref_alignbit(s0, s1, 31), 0x2468ACF1)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 31), 0x2468ACF1)

        # shift 32 -> masked to 0
        self.assertEqual(ref_alignbit(s0, s1, 32), 0x9ABCDEF0)
        self.assertEqual(interp_v_alignbit_b32(s0, s1, 32), 0x9ABCDEF0)

    def test_interp_matches_ref_all_shifts(self) -> None:
        vectors = [
            (0x12345678, 0x9ABCDEF0),
            (0xAAAAAAAA, 0x55555555),
            (0xFFFFFFFF, 0x00000000),
            (0x00000000, 0xFFFFFFFF),
            (0xDEADBEEF, 0xCAFEBABE),
            (0x80000001, 0x7FFFFFFE),
        ]
        for s0, s1 in vectors:
            for shift in range(64):
                with self.subTest(s0=hex(s0), s1=hex(s1), shift=shift):
                    self.assertEqual(
                        interp_v_alignbit_b32(s0, s1, shift),
                        ref_alignbit(s0, s1, shift),
                    )


class AlignbitSemanticInterpreterTests(unittest.TestCase):
    """Semantic interpreter tests for RDNA2 v_alignbit_b32 translation and execution."""

    def test_every_5bit_shift_count(self) -> None:
        """Every shift count 0..31 across diverse bit patterns."""
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
            (0xFFFFFFFF, 0x00000000),
            (0x00000000, 0xFFFFFFFF),
            (0x80000001, 0x7FFFFFFE),
            (0xDEADBEEF, 0xCAFEBABE),
            (0x13579BDF, 0x2468ACE0),
            (0x8899AABB, 0x11223344),
        ]
        for s0, s1 in test_pairs:
            for shift in range(32):
                with self.subTest(s0=hex(s0), s1=hex(s1), shift=shift):
                    got = run_alignbit(s0, s1, shift)
                    want = ref_alignbit(s0, s1, shift)
                    self.assertEqual(got, want)

    def test_register_boundary_crossing_single_bit(self) -> None:
        """Validates bit movement across the 32-bit register boundary."""
        # Test a single bit set in S0 crossing into the MSB of the result
        # S0 = 1 (bit 0 set), S1 = 0
        # For shift = 1: bit 0 of S0 moves to bit 31 of result (0x80000000)
        # For shift = 2: bit 0 of S0 moves to bit 30 of result (0x40000000)
        # For shift = k (1..31): bit 0 of S0 moves to bit (32 - k)
        for shift in range(1, 32):
            with self.subTest(direction="s0_to_result", shift=shift):
                got = run_alignbit(1, 0, shift)
                want = 1 << (32 - shift)
                self.assertEqual(got, want)

        # When shift is 0, S0 contribution is completely discarded
        self.assertEqual(run_alignbit(1, 0, 0), 0)

        # Test single bit in S1 crossing out towards the LSB
        # S0 = 0, S1 = 0x80000000 (bit 31 set)
        # For shift = 1: bit 31 of S1 moves to bit 30 (0x40000000)
        # For shift = 31: bit 31 of S1 moves to bit 0 (0x00000001)
        for shift in range(32):
            with self.subTest(direction="s1_shifted", shift=shift):
                got = run_alignbit(0, 0x80000000, shift)
                want = (0x80000000 >> shift) & MASK32
                self.assertEqual(got, want)

        # Test full 64-bit single-bit walk across boundary
        # For bit index 0..63 in {s0, s1}:
        # shift by 16: bits 16..47 should map to bits 0..31 of output
        shift = 16
        for bit_pos in range(64):
            if bit_pos < 32:
                s0_val, s1_val = 0, 1 << bit_pos
            else:
                s0_val, s1_val = 1 << (bit_pos - 32), 0

            with self.subTest(bit_pos=bit_pos, shift=shift):
                got = run_alignbit(s0_val, s1_val, shift)
                expected_bit = bit_pos - shift
                want = (1 << expected_bit) if (0 <= expected_bit < 32) else 0
                self.assertEqual(got, want)

    def test_shift_zero_returns_low_word(self) -> None:
        """Shift 0 (and multiples of 32) must preserve S1 exactly."""
        test_words = [
            (0xDEADBEEF, 0x12345678),
            (0xFFFFFFFF, 0x00000000),
            (0x00000000, 0xFFFFFFFF),
            (0x13579BDF, 0x2468ACE0),
            (0x80000000, 0x7FFFFFFF),
        ]
        for s0, s1 in test_words:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_alignbit(s0, s1, 0), s1)
                self.assertEqual(run_alignbit(s0, s1, 0, inline_shift=True), s1)

    def test_shift_amount_masking_5bit(self) -> None:
        """Shift amounts >= 32 must be masked with 0x1F (bits 4:0)."""
        s0, s1 = 0x8899AABB, 0x11223344
        high_shifts = [
            32,           # 32 & 31 == 0
            33,           # 33 & 31 == 1
            36,           # 36 & 31 == 4
            40,           # 40 & 31 == 8
            48,           # 48 & 31 == 16
            63,           # 63 & 31 == 31
            64,           # 64 & 31 == 0
            65,           # 65 & 31 == 1
            127,          # 127 & 31 == 31
            128,          # 128 & 31 == 0
            255,          # 255 & 31 == 31
            1024,         # 1024 & 31 == 0
            0x80000000,   # bit 31 set, low 5 bits 0
            0x80000007,   # low 5 bits 7
            0xFFFFFFFF,   # all bits set, low 5 bits 31
            0xFFFFFFE0,   # low 5 bits 0
            0xFFFFFFE1,   # low 5 bits 1
        ]
        for shift in high_shifts:
            with self.subTest(shift=hex(shift)):
                got = run_alignbit(s0, s1, shift)
                want = ref_alignbit(s0, s1, shift)
                self.assertEqual(got, want)

    def test_inline_constant_shifts(self) -> None:
        """Inline literal constants for the shift operand."""
        s0, s1 = 0xA1B2C3D4, 0xE5F60718
        for shift in (0, 1, 2, 5, 8, 15, 16, 23, 24, 30, 31, 32, 33, 63):
            with self.subTest(shift=shift):
                got = run_alignbit(s0, s1, shift, inline_shift=True)
                want = ref_alignbit(s0, s1, shift)
                self.assertEqual(got, want)

    def test_sgpr_operands(self) -> None:
        """VOP3 instruction encoding with scalar register sources."""
        s0, s1, shift = 0x12345678, 0x9ABCDEF0, 12
        want = ref_alignbit(s0, s1, shift)

        # S0 in SGPR
        self.assertEqual(run_alignbit(s0, s1, shift, src0_sgpr=True), want)
        # S1 in SGPR
        self.assertEqual(run_alignbit(s0, s1, shift, src1_sgpr=True), want)
        # Shift in SGPR
        self.assertEqual(run_alignbit(s0, s1, shift, shift_sgpr=True), want)

    def test_rotate_right_compiler_idiom(self) -> None:
        """v_alignbit_b32 with S0 == S1 is identical to 32-bit rotate right."""
        rot_patterns = [
            0x80000001,
            0x12345678,
            0xA5A5A5A5,
            0xDEADBEEF,
            0x00000001,
            0xFFFFFFFF,
        ]
        for x in rot_patterns:
            for shift in range(32):
                with self.subTest(x=hex(x), shift=shift):
                    got = run_alignbit(x, x, shift)
                    if shift == 0:
                        want_ror = x & MASK32
                    else:
                        want_ror = ((x >> shift) | (x << (32 - shift))) & MASK32
                    self.assertEqual(got, want_ror)

    def test_logical_shift_equivalences(self) -> None:
        """Validates funnel shift degenerating to LSR or LSL."""
        val = 0xA7B8C9D0

        # S0 == 0 -> equivalent to LSR (logical shift right of S1)
        for shift in range(32):
            with self.subTest(op="lsr_equiv", shift=shift):
                got = run_alignbit(0, val, shift)
                want = (val >> shift) & MASK32
                self.assertEqual(got, want)

        # S1 == 0 -> equivalent to LSL (logical shift left of S0 by (32 - shift))
        for shift in range(1, 32):
            with self.subTest(op="lsl_equiv", shift=shift):
                got = run_alignbit(val, 0, shift)
                want = (val << (32 - shift)) & MASK32
                self.assertEqual(got, want)

        # When shift == 0 and S1 == 0, output must be 0
        self.assertEqual(run_alignbit(val, 0, 0), 0)

    def test_wave_lane_vector_simulation(self) -> None:
        """Simulates SIMD wave-64 lane execution with lane-dependent shift amounts."""
        for lane_id in range(64):
            # Vary inputs and shifts deterministically across lanes
            s0 = (0x10000000 + (lane_id * 0x01010101)) & MASK32
            s1 = (0xF0000000 - (lane_id * 0x02020202)) & MASK32
            shift = lane_id  # tests shifts 0..63 (triggering modulo 32 above 31)

            with self.subTest(lane=lane_id, shift=shift):
                got = interp_v_alignbit_b32(s0, s1, shift)
                want = ref_alignbit(s0, s1, shift)
                self.assertEqual(got, want)


class DirectSpirvSequenceTests(unittest.TestCase):
    """Direct execution of the SPIR-V sequence emitted for v_alignbit_b32."""

    def test_spirv_sequence_all_shifts(self) -> None:
        """Tests the standalone SPIR-V instruction sequence against the oracle."""
        test_pairs = [
            (0xFEEDBEEF, 0x01234567),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x00FF00FF, 0xFF00FF00),
        ]
        for s0, s1 in test_pairs:
            for shift in range(32):
                with self.subTest(s0=hex(s0), s1=hex(s1), shift=shift):
                    got = run_spirv_alignbit(s0, s1, shift)
                    want = ref_alignbit(s0, s1, shift)
                    self.assertEqual(got, want)

    def test_spirv_shift_guard_at_zero(self) -> None:
        """Ensures shift=0 does not trigger SPIR-V undefined shift (>= 32 bits).

        In SPIR-V, OpShiftLeftLogical / OpShiftRightLogical with shift operand >= bit width
        is undefined. For shift=0, (32 - m) is 32, but rdna2spirv masks it with 31:
        inv = (32 - m) & 31 = 0, safely shifting by 0 and letting OpSelect pick lo.
        """
        for s0, s1 in ((0x12345678, 0x9ABCDEF0), (0xFFFFFFFF, 0x00000000), (0, 0xFFFFFFFF)):
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_spirv_alignbit(s0, s1, 0)
                self.assertEqual(got, s1)

    def test_spirv_masked_shift_counts(self) -> None:
        """Direct SPIR-V sequence evaluation with out-of-range shift amounts."""
        s0, s1 = 0x13579BDF, 0x2468ACE0
        for shift in (32, 33, 40, 63, 64, 127, 0xFFFFFFFF):
            with self.subTest(shift=hex(shift)):
                got = run_spirv_alignbit(s0, s1, shift)
                want = ref_alignbit(s0, s1, shift)
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
