from __future__ import annotations

"""Semantic interpreter tests for 3-operand integer add (v_add3_u32).

Verifies unsigned 32-bit addition semantics with modulo 2^32 overflow behavior,
intermediate and final carry tracking (yielding carries in {0, 1, 2}),
equivalence with sequential scalar fallback additions, 64-bit multi-precision
extension with carry propagation, SPIR-V translation lowering to chained
OpIAdd instructions, and execution under spirv_mini_interp.
"""
import gc
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
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_add3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_add3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def ref_add3_u32(a: int, b: int, c: int) -> int:
    """Ideal 3-operand 32-bit unsigned addition: (a + b + c) mod 2^32."""
    return (a + b + c) & MASK32


def ref_add3_u32_with_carry(a: int, b: int, c: int) -> tuple[int, int]:
    """3-operand 32-bit unsigned addition with full carry-out.

    Returns (truncated_sum, carry_out) where carry_out in {0, 1, 2}.
    """
    total = (a & MASK32) + (b & MASK32) + (c & MASK32)
    return total & MASK32, total >> 32


def scalar_fallback_add3(a: int, b: int, c: int) -> int:
    """Sequential 2-operand scalar addition fallback simulating SALU or lowering.

    RDNA2 Scalar ALU (SALU) provides 2-operand additions (s_add_u32) rather than
    a native 3-operand scalar add. A scalar fallback lowers the 3-input operation
    into two sequential additions with 32-bit truncation at each stage.
    """
    step1 = (a + b) & MASK32
    step2 = (step1 + c) & MASK32
    return step2


def scalar_fallback_add3_with_carry(a: int, b: int, c: int) -> tuple[int, int]:
    """Sequential scalar fallback modeling carry accumulation.

    Simulates RDNA2 scalar ALU addition sequence:
      s_add_u32  d1, a, b    # SCC = carry1
      s_add_u32  d2, d1, c   # SCC = carry2
    The total carry out of the 3-input addition is the sum of both stage carries:
      total_carry = carry1 + carry2
    where total_carry in {0, 1, 2}.
    """
    a &= MASK32
    b &= MASK32
    c &= MASK32

    sum1 = a + b
    d1 = sum1 & MASK32
    carry1 = 1 if sum1 > MASK32 else 0

    sum2 = d1 + c
    d2 = sum2 & MASK32
    carry2 = 1 if sum2 > MASK32 else 0

    return d2, carry1 + carry2


def decompose_u64(val: int) -> tuple[int, int]:
    """Decompose a 64-bit unsigned integer into (lo, hi) 32-bit dwords."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def compose_u64(lo: int, hi: int) -> int:
    """Compose two 32-bit dwords into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def interp_v_add3_u64(a: int, b: int, c: int) -> int:
    """64-bit 3-operand addition reference."""
    return (a + b + c) & MASK64


def interp_v_add3_u64_dwords(
    a_lo: int, a_hi: int, b_lo: int, b_hi: int, c_lo: int, c_hi: int
) -> tuple[int, int]:
    """64-bit multi-precision 3-operand addition using 32-bit dwords and carry.

    Low dword is evaluated with v_add3_u32.
    Carry out (0, 1, or 2) propagates into high dword addition.
    """
    res_lo, carry_lo = scalar_fallback_add3_with_carry(a_lo, b_lo, c_lo)
    res_hi = (a_hi + b_hi + c_hi + carry_lo) & MASK32
    return res_lo, res_hi


def run_add3_interp(a: int, b: int, c: int) -> int:
    """Execute chained OpIAdd instructions via spirv_mini_interp."""
    module = (
        'OpCapability Shader\n'
        '%glsl = OpExtInstImport "GLSL.std.450"\n'
        'OpMemoryModel Logical GLSL450\n'
        'OpEntryPoint Fragment %main "main" %out\n'
        '%void = OpTypeVoid\n'
        '%fn = OpTypeFunction %void\n'
        '%uint = OpTypeInt 32 0\n'
        '%_ptr_Output_uint = OpTypePointer Output %uint\n'
        '%out = OpVariable %_ptr_Output_uint Output\n'
        f'%ca = OpConstant %uint {a & MASK32}\n'
        f'%cb = OpConstant %uint {b & MASK32}\n'
        f'%cc = OpConstant %uint {c & MASK32}\n'
        '%main = OpFunction %void None %fn\n'
        '%entry = OpLabel\n'
        '%s = OpIAdd %uint %ca %cb\n'
        '%res = OpIAdd %uint %s %cc\n'
        'OpStore %out %res\n'
        'OpReturn\n'
        'OpFunctionEnd\n'
    )
    return si.run_fragment(module)["%out"][0]


def run_synthetic_shader(asm_instructions: list[str]) -> tuple[list[int], str]:
    """Translate synthetic RDNA2 disassembly to SPIR-V and execute it."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    assert _SRC_PATH is not None

    lines = ["; synthetic shader for alu_add3 test"]
    for ins in asm_instructions:
        lines.append("\t" + ins.strip())
    content = "\n".join(lines) + "\n"

    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spvasm = rdna2spirv.translate(_SRC_PATH)
    words = support.mrt0_words(spvasm)
    return words, spvasm


class TestAluAdd3Semantics(unittest.TestCase):
    """Unit tests for v_add3_u32 execution semantics, carry handling, and scalar fallback."""

    def test_scalar_fallback_equivalence(self) -> None:
        """Scalar fallback must strictly match the 3-operand reference across edge cases."""
        test_values = [
            0,
            1,
            2,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFE,
            0xFFFFFFFF,
            0x55555555,
            0xAAAAAAAA,
            0x12345678,
            0x87654321,
            0x0000FFFF,
            0xFFFF0000,
        ]
        for a in test_values:
            for b in test_values:
                for c in (0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF):
                    ref = ref_add3_u32(a, b, c)
                    fallback = scalar_fallback_add3(a, b, c)
                    self.assertEqual(ref, fallback, f"Mismatch for inputs ({hex(a)}, {hex(b)}, {hex(c)})")

    def test_scalar_fallback_carry_tracking_all_levels(self) -> None:
        """Scalar fallback carry accumulation must match mathematical sum >> 32 for carries 0, 1, and 2."""
        cases = [
            # carry = 0
            (10, 20, 30, 60, 0),
            (0, 0, 0, 0, 0),
            (0x10000000, 0x20000000, 0x30000000, 0x60000000, 0),
            # carry = 1 (stage 1 overflow, stage 2 no overflow)
            (0xFFFFFFFF, 1, 5, 5, 1),
            (0x80000000, 0x80000000, 100, 100, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 1, 0xFFFFFFFF, 1),
            # carry = 1 (stage 1 no overflow, stage 2 overflow)
            (1, 4, 0xFFFFFFFF, 4, 1),
            (0x40000000, 0x40000000, 0x80000000, 0, 1),
            # carry = 2 (both stages overflow)
            (0xFFFFFFFF, 0xFFFFFFFF, 2, 0, 2),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFD, 2),
            (0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFA, 2),
            (0x80000001, 0x80000001, 0xFFFFFFFF, 1, 2),
            (0xC0000000, 0xC0000000, 0xC0000000, 0x40000000, 2),
        ]
        for a, b, c, expected_res, expected_carry in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                ref_res, ref_carry = ref_add3_u32_with_carry(a, b, c)
                fb_res, fb_carry = scalar_fallback_add3_with_carry(a, b, c)

                self.assertEqual(ref_res, expected_res)
                self.assertEqual(ref_carry, expected_carry)
                self.assertEqual(fb_res, expected_res)
                self.assertEqual(fb_carry, expected_carry)
                # Verify exact mathematical reconstruction: (a + b + c) == res + carry * 2^32
                self.assertEqual(a + b + c, fb_res + (fb_carry << 32))

    def test_scalar_fallback_commutativity_and_associativity(self) -> None:
        """Scalar fallback must preserve permutation invariance under mod 2^32."""
        triplets = [
            (0xFFFFFFFF, 1, 42),
            (0x80000000, 0x80000000, 0x80000000),
            (0x7FFFFFFF, 0x7FFFFFFF, 2),
            (0xFFFFFF00, 0x000000F0, 0x00000020),
            (0x10000000, 0x20000000, 0x30000000),
        ]
        for a, b, c in triplets:
            expected = ref_add3_u32(a, b, c)
            perms = [
                (a, b, c),
                (a, c, b),
                (b, a, c),
                (b, c, a),
                (c, a, b),
                (c, b, a),
            ]
            for p1, p2, p3 in perms:
                self.assertEqual(scalar_fallback_add3(p1, p2, p3), expected)
                self.assertEqual(run_add3_interp(p1, p2, p3), expected)

    def test_interp_basic_arithmetic(self) -> None:
        """Basic additions without overflow produce exact mathematical sums."""
        cases = [
            (1, 2, 3, 6),
            (100, 200, 300, 600),
            (0, 0, 0, 0),
            (0x1000, 0x2000, 0x3000, 0x6000),
            (0x1234, 0x5678, 0x9ABC, 0x10368),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(run_add3_interp(a, b, c), expected)
                self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_interp_identity_operands(self) -> None:
        """Zero operand acts as identity in any of the three operand positions."""
        val = 0xDEADBEEF
        self.assertEqual(run_add3_interp(val, 0, 0), val)
        self.assertEqual(run_add3_interp(0, val, 0), val)
        self.assertEqual(run_add3_interp(0, 0, val), val)

    def test_interp_single_overflow_intermediate(self) -> None:
        """Overflow occurs on the first addition (a + b >= 2^32) and wraps cleanly."""
        cases = [
            (0xFFFFFFFF, 1, 5, 5),
            (0x80000000, 0x80000000, 0x1234, 0x1234),
            (0xFFFFFFFE, 2, 0x42, 0x42),
            (0xFFFFFFFF, 0x10, 0x20, 0x2F),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(run_add3_interp(a, b, c), expected)
                self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_interp_single_overflow_final(self) -> None:
        """First addition does not overflow, but second addition wraps modulo 2^32."""
        cases = [
            (1, 4, 0xFFFFFFFF, 4),
            (10, 20, 0xFFFFFFF0, 0xE),
            (0x40000000, 0x40000000, 0x80000000, 0),
            (0x100, 0x200, 0xFFFFFF00, 0x200),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(run_add3_interp(a, b, c), expected)
                self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_interp_double_overflow(self) -> None:
        """Both additions overflow 2^32, wrapping twice."""
        cases = [
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFD),
            (0xFFFFFFFF, 0xFFFFFFFF, 1, 0xFFFFFFFF),
            (0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFA),
            (0x80000001, 0x80000001, 0xFFFFFFFF, 1),
            (0xC0000000, 0xC0000000, 0xC0000000, 0x40000000),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(run_add3_interp(a, b, c), expected)
                self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_interp_signed_extrema_and_boundaries(self) -> None:
        """Signed boundary patterns (0x7FFFFFFF, 0x80000000) are evaluated as unsigned 32-bit."""
        cases = [
            (0x7FFFFFFF, 0x7FFFFFFF, 2, 0),
            (0x80000000, 0x80000000, 0, 0),
            (0x7FFFFFFF, 1, 0, 0x80000000),
            (0x80000000, 0x7FFFFFFF, 1, 0),
            (0xFFFFFFFF, 0, 1, 0),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(run_add3_interp(a, b, c), expected)
                self.assertEqual(ref_add3_u32(a, b, c), expected)

    def test_wave_lane_vector_simulation(self) -> None:
        """Simulate 64 SIMD wave lanes independently evaluating v_add3_u32."""
        base_address = 0x80000000
        stride = 0x40
        lanes = 64
        for lane_id in range(lanes):
            lane_offset = lane_id * stride
            tag = 0x1234
            res = ref_add3_u32(base_address, lane_offset, tag)
            expected = (base_address + lane_offset + tag) & MASK32
            self.assertEqual(res, expected)
            self.assertEqual(scalar_fallback_add3(base_address, lane_offset, tag), expected)


class TestAluAdd3MultiPrecision64(unittest.TestCase):
    """Multi-precision 64-bit addition emulated via 32-bit dwords and carry propagation."""

    def test_64bit_decomposition_reconstruction(self) -> None:
        val = 0x123456789ABCDEF0
        lo, hi = decompose_u64(val)
        self.assertEqual(lo, 0x9ABCDEF0)
        self.assertEqual(hi, 0x12345678)
        self.assertEqual(compose_u64(lo, hi), val)

    def test_multi_precision_chained_carry(self) -> None:
        test_triplets = [
            (0x100000000, 0x200000000, 0x300000000),
            (0x00000000FFFFFFFF, 1, 2),  # low carry = 1
            (0x00000000FFFFFFFF, 0x00000000FFFFFFFF, 0x00000000FFFFFFFF),  # low carry = 2
            (0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 2),
            (0xFFFFFFFFFFFFFFFF, 1, 0),  # full 64-bit wrap
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),  # double 64-bit wrap
        ]
        for a, b, c in test_triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                expected_u64 = interp_v_add3_u64(a, b, c)
                exp_lo, exp_hi = decompose_u64(expected_u64)

                a_lo, a_hi = decompose_u64(a)
                b_lo, b_hi = decompose_u64(b)
                c_lo, c_hi = decompose_u64(c)

                got_lo, got_hi = interp_v_add3_u64_dwords(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)
                self.assertEqual((got_lo, got_hi), (exp_lo, exp_hi))
                self.assertEqual(compose_u64(got_lo, got_hi), expected_u64)


class TestAluAdd3TranslationAndExecution(unittest.TestCase):
    """End-to-end translation of v_add3_u32 disassembly and SPIR-V execution."""

    def test_shader_vector_operands_basic(self) -> None:
        """v_add3_u32 with all VGPR operands calculates expected sum."""
        asm = [
            "v_mov_b32 v0, 0x10",
            "v_mov_b32 v1, 0x20",
            "v_mov_b32 v2, 0x30",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [0x60, 0x60, 0x60, 0x60])

    def test_shader_vector_operands_double_wrap(self) -> None:
        """v_add3_u32 correctly wraps modulo 2^32 twice in full shader execution."""
        asm = [
            "v_mov_b32 v0, 0xffffffff",
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 0xffffffff",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [0xFFFFFFFD, 0xFFFFFFFD, 0xFFFFFFFD, 0xFFFFFFFD])

    def test_shader_mixed_sgpr_and_vgpr_operands(self) -> None:
        """v_add3_u32 supports mixed scalar SGPR and vector VGPR inputs."""
        asm = [
            "s_mov_b32 s0, 100",
            "s_mov_b32 s1, 200",
            "v_mov_b32 v0, 50",
            "v_add3_u32 v1, s0, s1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [350, 350, 350, 350])

    def test_shader_all_scalar_sources(self) -> None:
        """v_add3_u32 with all SGPR sources writing to vector register destination."""
        asm = [
            "s_mov_b32 s0, 0x7fffffff",
            "s_mov_b32 s1, 0x7fffffff",
            "s_mov_b32 s2, 2",
            "v_add3_u32 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [0, 0, 0, 0])

    def test_shader_immediate_operands(self) -> None:
        """v_add3_u32 accepts integer immediate operands in source positions."""
        asm = [
            "v_mov_b32 v0, 1000",
            "v_add3_u32 v1, v0, 15, 25",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [1040, 1040, 1040, 1040])

    def test_shader_equivalence_with_sequential_add_nc(self) -> None:
        """v_add3_u32 must yield identical output to sequential 2-operand v_add_nc_u32."""
        asm_add3 = [
            "v_mov_b32 v0, 0x90000000",
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x70000000",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        asm_seq = [
            "v_mov_b32 v0, 0x90000000",
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x70000000",
            "v_add_nc_u32 v4, v0, v1",
            "v_add_nc_u32 v3, v4, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        words_add3, _ = run_synthetic_shader(asm_add3)
        words_seq, _ = run_synthetic_shader(asm_seq)
        expected = ref_add3_u32(0x90000000, 0x80000000, 0x70000000)
        self.assertEqual(words_add3, [expected] * 4)
        self.assertEqual(words_seq, [expected] * 4)
        self.assertEqual(words_add3, words_seq)

    def test_spirv_lowering_structure(self) -> None:
        """v_add3_u32 must lower to exactly two chained OpIAdd instructions."""
        asm = [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 3",
            "v_add3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        _, spv = run_synthetic_shader(asm)
        iadd_lines = [line.strip() for line in spv.splitlines() if "= OpIAdd %uint" in line]
        self.assertEqual(len(iadd_lines), 2, f"Expected 2 OpIAdd instructions, got: {iadd_lines}")


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated arithmetic or missing masks fail assertions."""

    def test_subtraction_mutation_is_detected(self) -> None:
        ref = ref_add3_u32(100, 200, 300)
        sabotaged = (100 - 200 + 300) & MASK32
        self.assertNotEqual(ref, sabotaged)

    def test_bitwise_or_mutation_is_detected(self) -> None:
        # Compare v_add3_u32 with v_or3_b32
        a, b, c = 4, 8, 12
        # ADD: 4 + 8 + 12 = 24
        # OR:  4 | 8 | 12 = 12
        correct_add = ref_add3_u32(a, b, c)
        mutated_or = (a | b | c) & MASK32
        self.assertNotEqual(correct_add, mutated_or)

    def test_bitwise_xor_mutation_is_detected(self) -> None:
        # Compare v_add3_u32 with v_xor3_b32
        a, b, c = 1, 1, 1
        # ADD: 1 + 1 + 1 = 3
        # XOR: 1 ^ 1 ^ 1 = 1
        correct_add = ref_add3_u32(a, b, c)
        mutated_xor = (a ^ b ^ c) & MASK32
        self.assertNotEqual(correct_add, mutated_xor)

    def test_missing_truncation_mutation_is_detected(self) -> None:
        a, b, c = 0xFFFFFFFF, 1, 0
        correct_truncated = ref_add3_u32(a, b, c)
        untruncated = a + b + c
        self.assertEqual(correct_truncated, 0)
        self.assertEqual(untruncated, 0x100000000)
        self.assertNotEqual(correct_truncated, untruncated)

    def test_carry_loss_mutation_is_detected(self) -> None:
        # Double overflow generates carry = 2; clamping to 1 must be detected
        a, b, c = 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        _, true_carry = ref_add3_u32_with_carry(a, b, c)
        sabotaged_carry = 1
        self.assertEqual(true_carry, 2)
        self.assertNotEqual(true_carry, sabotaged_carry)


if __name__ == "__main__":
    unittest.main()
