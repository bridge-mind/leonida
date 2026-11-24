from __future__ import annotations

"""Semantic interpreter tests for AMD RDNA2 / GFX10 Vector ALU sum of absolute
differences instructions: v_sad_u32 and v_sad_hi_u32.

ISA Specifications (AMD RDNA2 / GFX10 Instruction Set Architecture):
- v_sad_u32 (VOP3 3-operand vector ALU instruction, opcode 349):
    Calculates the 32-bit unsigned sum of absolute differences with accumulation:
        diff = |src0.u32 - src1.u32|
        dst.u32 = (diff + src2.u32) mod 2^32
    Note on Unsigned Semantics (documented in AMD GCN/RDNA ISA errata):
    The absolute difference is computed in unsigned arithmetic:
        diff = (src0 >= src1) ? (src0 - src1) : (src1 - src0)
    All inputs and distances operate on unsigned 32-bit integers in [0, 2^32 - 1].
    Treating operands as signed integers (e.g. 0xFFFFFFFF as -1) is an architectural
    deviation that produces incorrect results (e.g. |0 - 0xFFFFFFFF| is 0xFFFFFFFF, not 1).

- v_sad_hi_u32 (High 32-bit carry / accumulation for multi-precision SAD):
    Models the high 32-bit carry/overflow word of the full 64-bit sum:
        diff = |src0.u32 - src1.u32|
        full_sum = diff + src2.u32
        dst.u32 = (full_sum >> 32) & 0xFFFFFFFF
    Since diff <= 2^32 - 1 and src2 <= 2^32 - 1, full_sum <= 2^33 - 2, so the high
    carry word is strictly in {0, 1}.
    Pairing (v_sad_u32, v_sad_hi_u32) synthesizes full 64-bit precision sum of
    absolute differences with accumulation.
    In RDNA2 ISA, v_sad_hi_u32 is not a hardware opcode (the hardware provides
    v_sad_hi_u8 for media extensions, while 32-bit multi-precision carry uses
    OpIAddCarry / carry chaining). Translator integration confirms that unknown
    opcodes raise Unsupported rather than silently miscompiling.

Standard library only: no GPU, no LLVM, no external tools, zero game data.
Verified against tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py.
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
    _SRC_PATH = os.path.join(disasm_dir, "ps_sad.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sad.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Calculate the strictly unsigned absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u32(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_u32: |s0 - s1| + s2 wrapped to 32 bits."""
    diff = ref_abs_diff_u32(s0, s1)
    return (diff + (s2 & MASK32)) & MASK32


def ref_sad_u32_clamp(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_u32 with clamp modifier: clamped to 0xFFFFFFFF on overflow."""
    diff = ref_abs_diff_u32(s0, s1)
    full_sum = diff + (s2 & MASK32)
    return min(full_sum, MASK32)


def ref_sad_hi_u32(s0: int, s1: int, s2: int) -> int:
    """Calculate v_sad_hi_u32: high 32 bits (carry out) of (|s0 - s1| + s2)."""
    diff = ref_abs_diff_u32(s0, s1)
    full_sum = diff + (s2 & MASK32)
    return (full_sum >> 32) & MASK32


def ref_sad_64(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Synthesize 64-bit result (lo, hi) from the (v_sad_u32, v_sad_hi_u32) pair."""
    return ref_sad_u32(s0, s1, s2), ref_sad_hi_u32(s0, s1, s2)


def ref_sad_hi_u32_shift16(s0: int, s1: int, s2: int) -> int:
    """Shifted variant (analogous to v_sad_hi_u8 in media extensions): (|s0 - s1| << 16) + s2."""
    diff = ref_abs_diff_u32(s0, s1)
    return (((diff & 0xFFFF) << 16) + (s2 & MASK32)) & MASK32


def ref_sad_hi_u32_upper_halfwords(s0: int, s1: int, s2: int) -> int:
    """High halfword SAD variant: |s0[31:16] - s1[31:16]| + s2."""
    h0 = (s0 >> 16) & 0xFFFF
    h1 = (s1 >> 16) & 0xFFFF
    diff = (h0 - h1) if h0 >= h1 else (h1 - h0)
    return (diff + (s2 & MASK32)) & MASK32


def run_spirv_sad_u32(s0: int, s1: int, s2: int) -> int:
    """Executes the canonical SPIR-V instruction sequence for v_sad_u32 via spirv_mini_interp."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c_s2 = OpConstant %uint {s2 & MASK32}
%main = OpFunction %void None %fn
%entry = OpLabel
%cond = OpUGreaterThanEqual %bool %c_s0 %c_s1
%d0 = OpISub %uint %c_s0 %c_s1
%d1 = OpISub %uint %c_s1 %c_s0
%diff = OpSelect %uint %cond %d0 %d1
%res = OpIAdd %uint %diff %c_s2
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


def run_spirv_sad_hi_u32(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Executes the canonical SPIR-V lowering for full 64-bit SAD (v_sad_u32, v_sad_hi_u32)
    using OpIAddCarry via spirv_mini_interp. Returns (lo, hi).
    """
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_lo %out_hi
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%struct_u2 = OpTypeStruct %uint %uint
%_ptr_Output_uint = OpTypePointer Output %uint
%out_lo = OpVariable %_ptr_Output_uint Output
%out_hi = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c_s2 = OpConstant %uint {s2 & MASK32}
%main = OpFunction %void None %fn
%entry = OpLabel
%cond = OpUGreaterThanEqual %bool %c_s0 %c_s1
%d0 = OpISub %uint %c_s0 %c_s1
%d1 = OpISub %uint %c_s1 %c_s0
%diff = OpSelect %uint %cond %d0 %d1
%sum = OpIAddCarry %struct_u2 %diff %c_s2
%lo = OpCompositeExtract %uint %sum 0
%hi = OpCompositeExtract %uint %sum 1
OpStore %out_lo %lo
OpStore %out_hi %hi
OpReturn
OpFunctionEnd
"""
    res = si.run_fragment(module)
    return res["%out_lo"][0], res["%out_hi"][0]


def run_spirv_sad_u32_clamp(s0: int, s1: int, s2: int) -> int:
    """Executes the canonical SPIR-V lowering for v_sad_u32 with clamp modifier."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%struct_u2 = OpTypeStruct %uint %uint
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c_s2 = OpConstant %uint {s2 & MASK32}
%c_zero = OpConstant %uint 0
%c_max = OpConstant %uint 4294967295
%main = OpFunction %void None %fn
%entry = OpLabel
%cond = OpUGreaterThanEqual %bool %c_s0 %c_s1
%d0 = OpISub %uint %c_s0 %c_s1
%d1 = OpISub %uint %c_s1 %c_s0
%diff = OpSelect %uint %cond %d0 %d1
%sum = OpIAddCarry %struct_u2 %diff %c_s2
%lo = OpCompositeExtract %uint %sum 0
%hi = OpCompositeExtract %uint %sum 1
%overflow = OpINotEqual %bool %hi %c_zero
%clamped = OpSelect %uint %overflow %c_max %lo
OpStore %out %clamped
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


def run_spirv_sad_hi_shift16(s0: int, s1: int, s2: int) -> int:
    """Executes the media extension shift-16 variant in SPIR-V: (|s0 - s1| << 16) + s2."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c_s2 = OpConstant %uint {s2 & MASK32}
%c_16 = OpConstant %uint 16
%c_mask16 = OpConstant %uint 65535
%main = OpFunction %void None %fn
%entry = OpLabel
%cond = OpUGreaterThanEqual %bool %c_s0 %c_s1
%d0 = OpISub %uint %c_s0 %c_s1
%d1 = OpISub %uint %c_s1 %c_s0
%diff = OpSelect %uint %cond %d0 %d1
%diff_masked = OpBitwiseAnd %uint %diff %c_mask16
%shifted = OpShiftLeftLogical %uint %diff_masked %c_16
%res = OpIAdd %uint %shifted %c_s2
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


def run_synthetic_shader(asm_instructions: list[str]) -> tuple[list[int], str]:
    """Translate synthetic RDNA2 disassembly to SPIR-V and execute it via spirv_mini_interp."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    assert _SRC_PATH is not None

    lines = ["; synthetic shader for alu_sad_u32 test"]
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


def run_synthetic_sad_u32(s0: int, s1: int, s2: int) -> int:
    """Executes a translated v_sad_u32 shader and returns the resulting 32-bit output."""
    asm = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        "v_sad_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    words, _ = run_synthetic_shader(asm)
    return words[0]


class TestSadU32ReferenceSemantics(unittest.TestCase):
    """Unit tests verifying mathematical properties of 32-bit unsigned SAD."""

    def test_identity_zero_difference(self) -> None:
        """When s0 == s1, the absolute difference is 0 and result is exactly s2."""
        for val in (0, 1, 42, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_abs_diff_u32(val, val), 0)
                self.assertEqual(ref_sad_u32(val, val, 12345), 12345)

    def test_zero_accumulator(self) -> None:
        """When s2 == 0, result equals |s0 - s1|."""
        self.assertEqual(ref_sad_u32(20, 5, 0), 15)
        self.assertEqual(ref_sad_u32(5, 20, 0), 15)
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 0), 0xFFFFFFFF)

    def test_operand_commutativity(self) -> None:
        """The absolute difference is commutative: |s0 - s1| == |s1 - s0|."""
        values = (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678)
        for s0 in values:
            for s1 in values:
                with self.subTest(s0=hex(s0), s1=hex(s1)):
                    diff_a = ref_abs_diff_u32(s0, s1)
                    diff_b = ref_abs_diff_u32(s1, s0)
                    self.assertEqual(diff_a, diff_b)
                    self.assertEqual(ref_sad_u32(s0, s1, 99), ref_sad_u32(s1, s0, 99))

    def test_strictly_unsigned_distance(self) -> None:
        """Confirms that distances operate on unsigned values.

        |0 - 0xFFFFFFFF| must equal 0xFFFFFFFF (unsigned magnitude), NOT 1.
        Signed interpretation would treat 0xFFFFFFFF as -1, giving |0 - (-1)| = 1,
        which is an architectural deviation from AMD GPU ISA.
        """
        self.assertEqual(ref_abs_diff_u32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_abs_diff_u32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        self.assertEqual(ref_sad_u32(0, 0xFFFFFFFF, 0), 0xFFFFFFFF)

        self.assertEqual(ref_abs_diff_u32(0x7FFFFFFF, 0x80000000), 1)
        self.assertEqual(ref_abs_diff_u32(0, 0x80000000), 0x80000000)

    def test_modulo_2_32_wrap_around(self) -> None:
        """Addition of the absolute difference and accumulator wraps modulo 2^32."""
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 1), 0)
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 2), 1)
        self.assertEqual(ref_sad_u32(0x80000000, 0, 0x80000000), 0)
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 0xFFFFFFFF), 0xFFFFFFFE)

    def test_triplet_combinations(self) -> None:
        """Known vectors representing common shader ALU sequences."""
        triplets = [
            (10, 25, 100, 115),
            (25, 10, 100, 115),
            (0, 0, 0, 0),
            (0x1000, 0x2000, 0x3000, 0x4000),
            (0x80000000, 0x7FFFFFFE, 10, 12),
            (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0),
        ]
        for s0, s1, s2, expected in triplets:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                self.assertEqual(ref_sad_u32(s0, s1, s2), expected)


class TestSadHiU32AndMultiPrecision(unittest.TestCase):
    """Unit tests for high 32-bit carry / accumulation (v_sad_hi_u32) and multi-precision SAD."""

    def test_carry_generation_exact_threshold(self) -> None:
        """Carry is 0 when sum < 2^32, and 1 when sum >= 2^32."""
        self.assertEqual(ref_sad_hi_u32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_sad_hi_u32(0xFFFFFFFE, 0, 1), 0)

        self.assertEqual(ref_sad_hi_u32(0xFFFFFFFF, 0, 1), 1)
        self.assertEqual(ref_sad_hi_u32(0x80000000, 0, 0x80000000), 1)

        self.assertEqual(ref_sad_hi_u32(0xFFFFFFFF, 0, 0xFFFFFFFF), 1)
        self.assertEqual(ref_sad_hi_u32(0x80000000, 0, 0x80000001), 1)

    def test_carry_out_maximum_bound(self) -> None:
        """Since diff <= 2^32 - 1 and s2 <= 2^32 - 1, full_sum <= 2^33 - 2, so carry is always in {0, 1}."""
        max_sum = ref_abs_diff_u32(0xFFFFFFFF, 0) + 0xFFFFFFFF
        self.assertEqual(max_sum, 0x1FFFFFFFE)
        self.assertEqual(max_sum >> 32, 1)
        self.assertIn(ref_sad_hi_u32(0xFFFFFFFF, 0, 0xFFFFFFFF), (0, 1))

    def test_64bit_synthesis(self) -> None:
        """The (v_sad_u32, v_sad_hi_u32) pair synthesizes full 64-bit precision sum of absolute differences."""
        vectors = [
            (0, 0, 0),
            (100, 50, 25),
            (0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0, 1),
            (0xFFFFFFFF, 0, 0xFFFFFFFF),
            (0x80000000, 0, 0x80000000),
            (0x12345678, 0x87654321, 0xCAFEBABE),
        ]
        for s0, s1, s2 in vectors:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                lo, hi = ref_sad_64(s0, s1, s2)
                reconstructed_64 = ((hi & MASK32) << 32) | (lo & MASK32)
                expected_64 = ref_abs_diff_u32(s0, s1) + (s2 & MASK32)
                self.assertEqual(reconstructed_64, expected_64)

    def test_shift16_media_variant(self) -> None:
        """Shift-16 media extension variant: (|s0 - s1| << 16) + s2."""
        self.assertEqual(ref_sad_hi_u32_shift16(10, 5, 0), 5 << 16)
        self.assertEqual(ref_sad_hi_u32_shift16(5, 10, 100), (5 << 16) + 100)
        self.assertEqual(ref_sad_hi_u32_shift16(0x0010, 0x0000, 0x0005), 0x00100005)

    def test_upper_halfwords_variant(self) -> None:
        """High halfword SAD variant: |s0[31:16] - s1[31:16]| + s2."""
        s0 = (0x0010 << 16) | 0xFFFF
        s1 = (0x0005 << 16) | 0x0000
        self.assertEqual(ref_sad_hi_u32_upper_halfwords(s0, s1, 100), 111)


class TestClampModifier(unittest.TestCase):
    """Unit tests for the clamp modifier saturation behavior."""

    def test_clamp_saturation_on_overflow(self) -> None:
        """When full_sum >= 2^32, the result is clamped to 0xFFFFFFFF."""
        self.assertEqual(ref_sad_u32_clamp(0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_sad_u32_clamp(0xFFFFFFFF, 0, 10), 0xFFFFFFFF)
        self.assertEqual(ref_sad_u32_clamp(0x80000000, 0, 0x80000000), 0xFFFFFFFF)

    def test_clamp_preserves_in_range_values(self) -> None:
        """When full_sum < 2^32, clamp does not modify the output."""
        self.assertEqual(ref_sad_u32_clamp(100, 20, 5), 85)
        self.assertEqual(ref_sad_u32_clamp(0xFFFFFFFE, 0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_sad_u32_clamp(0, 0, 0), 0)


class TestSpirvMiniInterpEquivalence(unittest.TestCase):
    """Executes canonical SPIR-V lowerings under spirv_mini_interp and checks against reference oracle."""

    def test_spirv_sad_u32_basic_cases(self) -> None:
        test_cases = [
            (10, 25, 100),
            (25, 10, 100),
            (0, 0, 0),
            (1000, 500, 250),
            (0x1234, 0x5678, 0x9ABC),
        ]
        for s0, s1, s2 in test_cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = ref_sad_u32(s0, s1, s2)
                got = run_spirv_sad_u32(s0, s1, s2)
                self.assertEqual(got, expected)

    def test_spirv_sad_u32_unsigned_boundary(self) -> None:
        """Confirms that SPIR-V lowering computes unsigned distance correctly."""
        got = run_spirv_sad_u32(0, 0xFFFFFFFF, 0)
        self.assertEqual(got, 0xFFFFFFFF)

        got = run_spirv_sad_u32(0xFFFFFFFF, 0, 0)
        self.assertEqual(got, 0xFFFFFFFF)

        got = run_spirv_sad_u32(0x7FFFFFFF, 0x80000000, 0)
        self.assertEqual(got, 1)

    def test_spirv_sad_u32_overflow_wrapping(self) -> None:
        """Confirms that SPIR-V OpIAdd correctly models modulo 2^32 wrap-around."""
        self.assertEqual(run_spirv_sad_u32(0xFFFFFFFF, 0, 1), 0)
        self.assertEqual(run_spirv_sad_u32(0xFFFFFFFF, 0, 2), 1)

    def test_spirv_sad_hi_u32_and_carry(self) -> None:
        """Confirms that SPIR-V OpIAddCarry produces exact 64-bit SAD output."""
        vectors = [
            (0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0, 1),
            (0xFFFFFFFF, 0, 0xFFFFFFFF),
            (0x80000000, 0, 0x80000000),
            (20, 5, 100),
        ]
        for s0, s1, s2 in vectors:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                expected_lo, expected_hi = ref_sad_64(s0, s1, s2)
                got_lo, got_hi = run_spirv_sad_hi_u32(s0, s1, s2)
                self.assertEqual(got_lo, expected_lo)
                self.assertEqual(got_hi, expected_hi)

    def test_spirv_clamp_execution(self) -> None:
        """Confirms that SPIR-V clamp lowering saturates upon overflow."""
        self.assertEqual(run_spirv_sad_u32_clamp(100, 20, 5), 85)
        self.assertEqual(run_spirv_sad_u32_clamp(0xFFFFFFFF, 0, 1), 0xFFFFFFFF)
        self.assertEqual(run_spirv_sad_u32_clamp(0xFFFFFFFF, 0, 10), 0xFFFFFFFF)

    def test_spirv_shift16_execution(self) -> None:
        """Confirms that SPIR-V shift-16 media lowering evaluates correctly."""
        self.assertEqual(run_spirv_sad_hi_shift16(20, 5, 3), (15 << 16) + 3)
        self.assertEqual(run_spirv_sad_hi_shift16(5, 20, 100), (15 << 16) + 100)

    def test_randomized_differential_sweep(self) -> None:
        """Deterministic pseudorandom sweep comparing reference oracle with SPIR-V execution."""
        rng = random.Random(0x5AD_032)
        for _ in range(50):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            exp_lo, exp_hi = ref_sad_64(s0, s1, s2)
            spv_lo = run_spirv_sad_u32(s0, s1, s2)
            spv_pair_lo, spv_pair_hi = run_spirv_sad_hi_u32(s0, s1, s2)

            self.assertEqual(spv_lo, exp_lo)
            self.assertEqual(spv_pair_lo, exp_lo)
            self.assertEqual(spv_pair_hi, exp_hi)


class TestTranslatedShaderExecution(unittest.TestCase):
    """Executes synthetic RDNA2 disassembly shaders translated via rdna2spirv and run in spirv_mini_interp."""

    def test_shader_v_sad_u32_basic(self) -> None:
        """v_sad_u32 computes unsigned sum of absolute differences in translated shader."""
        self.assertEqual(run_synthetic_sad_u32(10, 25, 100), 115)
        self.assertEqual(run_synthetic_sad_u32(25, 10, 100), 115)
        self.assertEqual(run_synthetic_sad_u32(0, 0, 0), 0)
        self.assertEqual(run_synthetic_sad_u32(1000, 200, 50), 850)

    def test_shader_strictly_unsigned_distance(self) -> None:
        """v_sad_u32 treats operands as unsigned: |0 - 0xFFFFFFFF| == 0xFFFFFFFF."""
        self.assertEqual(run_synthetic_sad_u32(0, 0xFFFFFFFF, 0), 0xFFFFFFFF)
        self.assertEqual(run_synthetic_sad_u32(0xFFFFFFFF, 0, 0), 0xFFFFFFFF)
        self.assertEqual(run_synthetic_sad_u32(0x7FFFFFFF, 0x80000000, 0), 1)

    def test_shader_overflow_wrap(self) -> None:
        """v_sad_u32 wraps modulo 2^32 on overflow in translated shader."""
        self.assertEqual(run_synthetic_sad_u32(0xFFFFFFFF, 0, 1), 0)
        self.assertEqual(run_synthetic_sad_u32(0xFFFFFFFF, 0, 2), 1)
        self.assertEqual(run_synthetic_sad_u32(0x80000000, 0, 0x80000000), 0)

    def test_shader_edge_values_sweep(self) -> None:
        """Verifies v_sad_u32 across edge words."""
        edge_words = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678)
        for s0 in edge_words:
            for s1 in (0, 0xFFFFFFFF, 0x80000000):
                s2 = 100
                expected = ref_sad_u32(s0, s1, s2)
                got = run_synthetic_sad_u32(s0, s1, s2)
                self.assertEqual(got, expected)

    def test_shader_mixed_sgpr_and_vgpr(self) -> None:
        """v_sad_u32 accepts SGPR source registers alongside VGPRs."""
        asm = [
            "s_mov_b32 s0, 50",
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 5",
            "v_sad_u32 v0, s0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [35, 35, 35, 35])

    def test_shader_all_sgpr_sources(self) -> None:
        """v_sad_u32 with all SGPR sources writing to vector destination."""
        asm = [
            "s_mov_b32 s0, 200",
            "s_mov_b32 s1, 50",
            "s_mov_b32 s2, 1000",
            "v_sad_u32 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [1150, 1150, 1150, 1150])

    def test_shader_inline_immediate_operands(self) -> None:
        """v_sad_u32 accepts integer literal constants in source positions."""
        asm = [
            "v_mov_b32 v1, 100",
            "v_sad_u32 v0, v1, 30, 10",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [80, 80, 80, 80])

    def test_shader_multi_instruction_manhattan_block_matching(self) -> None:
        """Simulates 4D vector L1 Manhattan distance block matching in translated shader.

        Vector A = (10, 25, 40, 80)
        Vector B = (15, 20, 50, 70)
        Accumulator in v0 initialized to 0.
        Expected distance: |10-15| + |25-20| + |40-50| + |80-70| = 5 + 5 + 10 + 10 = 30.
        """
        asm = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 15",
            "v_sad_u32 v0, v1, v2, v0",
            "v_mov_b32 v1, 25",
            "v_mov_b32 v2, 20",
            "v_sad_u32 v0, v1, v2, v0",
            "v_mov_b32 v1, 40",
            "v_mov_b32 v2, 50",
            "v_sad_u32 v0, v1, v2, v0",
            "v_mov_b32 v1, 80",
            "v_mov_b32 v2, 70",
            "v_sad_u32 v0, v1, v2, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        words, _ = run_synthetic_shader(asm)
        self.assertEqual(words, [30, 30, 30, 30])

    def test_clamp_modifier_rejected(self) -> None:
        """v_sad_u32 with clamp modifier raises Unsupported rather than guessing."""
        asm = [
            "v_mov_b32 v1, 1",
            "v_sad_u32 v0, v1, v1, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            run_synthetic_shader(asm)
        self.assertIn("v_sad_u32 clamp", str(cm.exception))

    def test_v_sad_hi_u32_opcode_unsupported(self) -> None:
        """v_sad_hi_u32 is not a valid RDNA2 opcode and must raise Unsupported."""
        asm = [
            "v_mov_b32 v1, 1",
            "v_sad_hi_u32 v0, v1, v1, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            run_synthetic_shader(asm)
        self.assertIn("v_sad_hi_u32", str(cm.exception))


class TestMutationSensitivity(unittest.TestCase):
    """Ensure that subtle semantic mutations in SAD logic are detected."""

    def test_detects_signed_distance_mutation(self) -> None:
        """A buggy implementation treating operands as signed two's complement fails."""
        def buggy_signed_sad(s0: int, s1: int, s2: int) -> int:
            i0 = s0 - (1 << 32) if (s0 >> 31) else s0
            i1 = s1 - (1 << 32) if (s1 >> 31) else s1
            diff = abs(i0 - i1)
            return (diff + s2) & MASK32

        correct = ref_sad_u32(0, 0xFFFFFFFF, 0)
        wrong = buggy_signed_sad(0, 0xFFFFFFFF, 0)
        self.assertEqual(correct, 0xFFFFFFFF)
        self.assertEqual(wrong, 1)
        self.assertNotEqual(correct, wrong)

    def test_detects_missing_accumulator_mutation(self) -> None:
        """A buggy implementation ignoring src2 fails."""
        def buggy_no_acc(s0: int, s1: int, s2: int) -> int:
            return ref_abs_diff_u32(s0, s1)

        self.assertNotEqual(ref_sad_u32(10, 5, 20), buggy_no_acc(10, 5, 20))

    def test_detects_unwrapped_sum_mutation(self) -> None:
        """A buggy implementation without modulo 2^32 fails."""
        def buggy_unwrapped(s0: int, s1: int, s2: int) -> int:
            return ref_abs_diff_u32(s0, s1) + s2

        self.assertNotEqual(ref_sad_u32(0, 0xFFFFFFFF, 1), buggy_unwrapped(0, 0xFFFFFFFF, 1))

    def test_detects_subtraction_without_abs_mutation(self) -> None:
        """A buggy implementation using (s0 - s1 + s2) instead of absolute difference fails."""
        def buggy_no_abs(s0: int, s1: int, s2: int) -> int:
            return ((s0 - s1) + s2) & MASK32

        self.assertNotEqual(ref_sad_u32(5, 10, 0), buggy_no_abs(5, 10, 0))

    def test_detects_carry_loss_mutation(self) -> None:
        """A buggy implementation that drops carry on overflow fails."""
        s0, s1, s2 = 0xFFFFFFFF, 0, 1
        true_hi = ref_sad_hi_u32(s0, s1, s2)
        buggy_hi = 0
        self.assertEqual(true_hi, 1)
        self.assertNotEqual(true_hi, buggy_hi)


if __name__ == "__main__":
    unittest.main()
