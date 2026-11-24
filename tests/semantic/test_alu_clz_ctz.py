"""Semantic interpreter tests for bit counting instructions:
  - v_clz_i32_u32 / v_ffbh_u32 / s_flbit_i32_b32: count leading zeros
  - v_ctz_i32_b32 / v_ffbl_b32 / s_ff1_i32_b32: count trailing zeros
  - v_cls_i32 / v_ffbh_i32: count leading sign bits / first differing sign bit

Covers:
  * Bit-exact reference oracles for CLZ, CTZ, and CLS
  * Distinction between mathematical/architectural (CLZ=32, CTZ=32 on 0) vs
    AMD RDNA hardware/SPIR-V conventions (0xFFFFFFFF on 0 or all-identical sign bits)
  * Direct SPIR-V GLSL.std.450 FindUMsb, FindILsb, FindSMsb interpreter validation
  * Full translation and execution through rdna2spirv and spirv_mini_interp
  * Exhaustive single-bit sweeps across all 32 bit positions (1 << i)
  * Boundary values: 0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000001, 0x7FFFFFFE
  * Inverted single-bit masks, alternating bit patterns, and byte-boundary values
  * Bit reversal identity: clz(bit_reverse(x)) == ctz(x) for all non-zero x
  * Multi-lane SIMD wave simulation across 64 lanes
Standard library Python only; zero game data.
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
SIGN32 = 0x80000000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    """Interpret raw 32-bit integer as signed two's complement."""
    x &= MASK32
    return x - (1 << 32) if x & SIGN32 else x


def u32(x: int) -> int:
    """Interpret raw integer as unsigned 32-bit word."""
    return x & MASK32


def bit_reverse32(x: int) -> int:
    """32-bit bit reversal helper."""
    return int(f"{x & MASK32:032b}"[::-1], 2)


def interp_v_clz_i32_u32(val: int, zero_value: int = 32) -> int:
    """Reference interpreter for Count Leading Zeros (v_clz_i32_u32).

    Counts the number of leading zero bits from MSB (bit 31) down to the first set bit.
    zero_value defaults to 32 (standard arithmetic/LLVM ctlz); AMD hardware v_ffbh_u32 returns 0xFFFFFFFF (-1).
    """
    val &= MASK32
    if val == 0:
        return zero_value
    return 32 - val.bit_length()


def interp_v_ctz_i32_b32(val: int, zero_value: int = 32) -> int:
    """Reference interpreter for Count Trailing Zeros (v_ctz_i32_b32).

    Counts the number of trailing zero bits from LSB (bit 0) up to the first set bit.
    zero_value defaults to 32 (standard arithmetic/LLVM cttz); AMD hardware v_ffbl_b32 returns 0xFFFFFFFF (-1).
    """
    val &= MASK32
    if val == 0:
        return zero_value
    return (val & -val).bit_length() - 1


def interp_v_cls_i32(val: int, all_match_value: int = 31) -> int:
    """Reference interpreter for Count Leading Sign bits (v_cls_i32).

    Counts consecutive bits following bit 31 that match the sign bit.
    Returns count in range 0..30 (or all_match_value when all bits match sign bit).
    all_match_value defaults to 31 (architectural count of redundant sign bits).
    """
    val &= MASK32
    sign = (val >> 31) & 1
    match_count = 0
    for i in range(30, -1, -1):
        bit = (val >> i) & 1
        if bit == sign:
            match_count += 1
        else:
            return match_count
    return all_match_value


def interp_v_ffbh_u32(val: int) -> int:
    """RDNA2 hardware find first bit high (unsigned): 31 - FindUMsb(val), or 0xFFFFFFFF if 0."""
    return interp_v_clz_i32_u32(val, zero_value=MASK32)


def interp_v_ffbl_b32(val: int) -> int:
    """RDNA2 hardware find first bit low: FindILsb(val), or 0xFFFFFFFF if 0."""
    return interp_v_ctz_i32_b32(val, zero_value=MASK32)


def interp_v_ffbh_i32(val: int) -> int:
    """RDNA2 hardware find first bit high (signed): bit position from MSB of first differing bit, or 0xFFFFFFFF."""
    val &= MASK32
    s = s32(val)
    if s in (0, -1):
        return MASK32
    f = s.bit_length() - 1 if s > 0 else (~s).bit_length() - 1
    return (31 - f) & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_clz_ctz.s")
    with open(os.path.join(_TMP_DIR.name, "ps_clz_ctz.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_asm(lines: list[str]) -> tuple[int, int]:
    """Assemble and run shader lines, exporting v0 and v1 to mrt0."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_clz_ctz\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    res = si.run_fragment(spv)["%out_mrt0"]
    return res[0], res[1]


def run_spirv_clz(val: int, hw_zero: bool = True) -> int:
    """Directly executes SPIR-V Count Leading Zeros sequence using GLSL FindUMsb."""
    zero_ret = MASK32 if hw_zero else 32
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint {val & MASK32}
%c0 = OpConstant %uint 0
%c31 = OpConstant %uint 31
%c_zero_ret = OpConstant %uint {zero_ret}
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpExtInst %int %glsl FindUMsb %c_in
%mu = OpBitcast %uint %m
%d = OpISub %uint %c31 %mu
%z = OpIEqual %bool %c_in %c0
%res = OpSelect %uint %z %c_zero_ret %d
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


def run_spirv_ctz(val: int, hw_zero: bool = True) -> int:
    """Directly executes SPIR-V Count Trailing Zeros sequence using GLSL FindILsb."""
    zero_ret = MASK32 if hw_zero else 32
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint {val & MASK32}
%c0 = OpConstant %uint 0
%c_zero_ret = OpConstant %uint {zero_ret}
%main = OpFunction %void None %fn
%entry = OpLabel
%ri = OpExtInst %int %glsl FindILsb %c_in
%ru = OpBitcast %uint %ri
%z = OpIEqual %bool %c_in %c0
%res = OpSelect %uint %z %c_zero_ret %ru
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


def run_spirv_cls(val: int, hw_semantics: bool = True) -> int:
    """Directly executes SPIR-V Count Leading Sign sequence using GLSL FindSMsb."""
    base_val = 31 if hw_semantics else 30
    none_val = MASK32 if hw_semantics else 31
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%bool = OpTypeBool
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_in = OpConstant %uint {val & MASK32}
%c0_i = OpConstant %int 0
%c31_i = OpConstant %int 31
%c30_i = OpConstant %int 30
%c_none = OpConstant %uint {none_val}
%base = OpConstant %int {base_val}
%main = OpFunction %void None %fn
%entry = OpLabel
%i = OpBitcast %int %c_in
%f = OpExtInst %int %glsl FindSMsb %i
%none = OpSLessThan %bool %f %c0_i
%sub = OpISub %int %base %f
%subu = OpBitcast %uint %sub
%res = OpSelect %uint %none %c_none %subu
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


class TestBitCountingReferenceSemantics(unittest.TestCase):
    """Unit tests verifying mathematical and architectural reference models."""

    def test_clz_powers_of_two(self):
        for bit in range(32):
            val = 1 << bit
            expected_zeros = 31 - bit
            with self.subTest(bit=bit, val=hex(val)):
                self.assertEqual(interp_v_clz_i32_u32(val), expected_zeros)
                self.assertEqual(interp_v_ffbh_u32(val), expected_zeros)

    def test_ctz_powers_of_two(self):
        for bit in range(32):
            val = 1 << bit
            expected_zeros = bit
            with self.subTest(bit=bit, val=hex(val)):
                self.assertEqual(interp_v_ctz_i32_b32(val), expected_zeros)
                self.assertEqual(interp_v_ffbl_b32(val), expected_zeros)

    def test_clz_ctz_zero_handling(self):
        # Standard arithmetic convention
        self.assertEqual(interp_v_clz_i32_u32(0, zero_value=32), 32)
        self.assertEqual(interp_v_ctz_i32_b32(0, zero_value=32), 32)
        # Hardware RDNA / SPIR-V convention
        self.assertEqual(interp_v_clz_i32_u32(0, zero_value=MASK32), MASK32)
        self.assertEqual(interp_v_ctz_i32_b32(0, zero_value=MASK32), MASK32)
        self.assertEqual(interp_v_ffbh_u32(0), MASK32)
        self.assertEqual(interp_v_ffbl_b32(0), MASK32)

    def test_clz_ctz_all_ones(self):
        self.assertEqual(interp_v_clz_i32_u32(MASK32), 0)
        self.assertEqual(interp_v_ctz_i32_b32(MASK32), 0)
        self.assertEqual(interp_v_ffbh_u32(MASK32), 0)
        self.assertEqual(interp_v_ffbl_b32(MASK32), 0)

    def test_cls_positive_values(self):
        # 0x40000000: sign=0, bit 30=1 -> 0 redundant sign bits; distance from MSB = 1
        self.assertEqual(interp_v_cls_i32(0x40000000), 0)
        self.assertEqual(interp_v_ffbh_i32(0x40000000), 1)

        # 0x20000000: sign=0, bit 30=0, bit 29=1 -> 1 redundant sign bit; distance from MSB = 2
        self.assertEqual(interp_v_cls_i32(0x20000000), 1)
        self.assertEqual(interp_v_ffbh_i32(0x20000000), 2)

        # 0x00000001: bit 0=1, bits 30..1=0 -> 30 redundant sign bits; distance from MSB = 31
        self.assertEqual(interp_v_cls_i32(0x00000001), 30)
        self.assertEqual(interp_v_ffbh_i32(0x00000001), 31)

    def test_cls_negative_values(self):
        # 0x80000000 (-2147483648): sign=1, bit 30=0 -> 0 redundant sign bits; distance = 1
        self.assertEqual(interp_v_cls_i32(0x80000000), 0)
        self.assertEqual(interp_v_ffbh_i32(0x80000000), 1)

        # 0xC0000000 (-1073741824): sign=1, bit 30=1, bit 29=0 -> 1 redundant sign bit; distance = 2
        self.assertEqual(interp_v_cls_i32(0xC0000000), 1)
        self.assertEqual(interp_v_ffbh_i32(0xC0000000), 2)

        # 0xFFFFFFFE (-2): sign=1, bit 0=0, bits 30..1=1 -> 30 redundant sign bits; distance = 31
        self.assertEqual(interp_v_cls_i32(0xFFFFFFFE), 30)
        self.assertEqual(interp_v_ffbh_i32(0xFFFFFFFE), 31)

    def test_cls_uniform_values(self):
        # 0x00000000: all bits 0 (all match sign bit)
        self.assertEqual(interp_v_cls_i32(0, all_match_value=31), 31)
        self.assertEqual(interp_v_ffbh_i32(0), MASK32)

        # 0xFFFFFFFF: all bits 1 (all match sign bit)
        self.assertEqual(interp_v_cls_i32(MASK32, all_match_value=31), 31)
        self.assertEqual(interp_v_ffbh_i32(MASK32), MASK32)

    def test_bit_reverse_symmetry(self):
        test_patterns = [
            0x1, 0x2, 0x12345678, 0x80000000, 0x0F000000, 0x000000F0,
            0x55555555, 0xAAAAAAAA, 0x7FFFFFFF, 0x0000FFFF, 0xFFFF0000,
            0x00000100, 0x00800000, 0x01010101, 0x80808080,
        ]
        for p in test_patterns:
            with self.subTest(pattern=hex(p)):
                rev = bit_reverse32(p)
                # clz of reversed == ctz of original
                self.assertEqual(interp_v_clz_i32_u32(rev), interp_v_ctz_i32_b32(p))
                self.assertEqual(interp_v_clz_i32_u32(p), interp_v_ctz_i32_b32(rev))


class TestSpirvMiniInterpBitCounting(unittest.TestCase):
    """Direct SPIR-V mini interpreter verification for FindUMsb, FindILsb, and FindSMsb."""

    def test_spirv_clz_single_bits(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                expected = 31 - bit
                self.assertEqual(run_spirv_clz(val, hw_zero=True), expected)
                self.assertEqual(run_spirv_clz(val, hw_zero=False), expected)

    def test_spirv_ctz_single_bits(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                expected = bit
                self.assertEqual(run_spirv_ctz(val, hw_zero=True), expected)
                self.assertEqual(run_spirv_ctz(val, hw_zero=False), expected)

    def test_spirv_clz_ctz_zero(self):
        self.assertEqual(run_spirv_clz(0, hw_zero=True), MASK32)
        self.assertEqual(run_spirv_clz(0, hw_zero=False), 32)
        self.assertEqual(run_spirv_ctz(0, hw_zero=True), MASK32)
        self.assertEqual(run_spirv_ctz(0, hw_zero=False), 32)

    def test_spirv_cls_patterns(self):
        patterns = [
            (0x40000000, 1, 0),
            (0x20000000, 2, 1),
            (0x10000000, 3, 2),
            (0x00000001, 31, 30),
            (0x80000000, 1, 0),
            (0xC0000000, 2, 1),
            (0xE0000000, 3, 2),
            (0xFFFFFFFE, 31, 30),
            (0x00000000, MASK32, 31),
            (0xFFFFFFFF, MASK32, 31),
        ]
        for val, hw_exp, arch_exp in patterns:
            with self.subTest(val=hex(val)):
                self.assertEqual(run_spirv_cls(val, hw_semantics=True), hw_exp)
                self.assertEqual(run_spirv_cls(val, hw_semantics=False), arch_exp)

    def test_spirv_composite_masks(self):
        cases = [
            (0xF0F0F0F0, 0, 4),
            (0x0F0F0F0F, 4, 0),
            (0x00FF0000, 8, 16),
            (0x0000FF00, 16, 8),
            (0x000000FF, 24, 0),
            (0xFF000000, 0, 24),
        ]
        for val, exp_clz, exp_ctz in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(run_spirv_clz(val), exp_clz)
                self.assertEqual(run_spirv_ctz(val), exp_ctz)


class TestRdna2SpirvEndToEndExecution(unittest.TestCase):
    """End-to-end translation of RDNA assembly via rdna2spirv and execution in spirv_mini_interp."""

    def test_v_ffbh_u32_and_v_ffbl_b32_all_bits(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit, val=hex(val)):
                lines = [
                    f"v_mov_b32 v2, {val:#x}",
                    "v_ffbh_u32 v0, v2",
                    "v_ffbl_b32 v1, v2",
                ]
                clz_res, ctz_res = run_asm(lines)
                self.assertEqual(clz_res, 31 - bit)
                self.assertEqual(ctz_res, bit)

    def test_v_ffbh_u32_and_v_ffbl_b32_inverted_bits(self):
        for bit in range(32):
            val = (~(1 << bit)) & MASK32
            with self.subTest(bit=bit, val=hex(val)):
                lines = [
                    f"v_mov_b32 v2, {val:#x}",
                    "v_ffbh_u32 v0, v2",
                    "v_ffbl_b32 v1, v2",
                ]
                clz_res, ctz_res = run_asm(lines)
                self.assertEqual(clz_res, interp_v_ffbh_u32(val))
                self.assertEqual(ctz_res, interp_v_ffbl_b32(val))

    def test_v_ffbh_u32_and_v_ffbl_b32_boundaries(self):
        values = [
            0, MASK32, 0x7FFFFFFF, 0x80000000, 0x80000001, 0x7FFFFFFE,
            0x12345678, 0xEDB88320, 0x55555555, 0xAAAAAAAA, 0x0000FFFF, 0xFFFF0000,
        ]
        for val in values:
            with self.subTest(val=hex(val)):
                lines = [
                    f"v_mov_b32 v2, {val:#x}",
                    "v_ffbh_u32 v0, v2",
                    "v_ffbl_b32 v1, v2",
                ]
                clz_res, ctz_res = run_asm(lines)
                self.assertEqual(clz_res, interp_v_ffbh_u32(val))
                self.assertEqual(ctz_res, interp_v_ffbl_b32(val))

    def test_v_ffbh_i32_signed_boundaries(self):
        values = [
            0, MASK32, 0x40000000, 0x20000000, 0x10000000, 0x00000001,
            0x80000000, 0xC0000000, 0xE0000000, 0xFFFFFFFE,
            0x7FFFFFFF, 0x0000FFFF, 0xFFFF0000, 0x55555555, 0xAAAAAAAA,
        ]
        for val in values:
            with self.subTest(val=hex(val)):
                lines = [
                    f"v_mov_b32 v2, {val:#x}",
                    "v_ffbh_i32 v0, v2",
                    "v_mov_b32 v1, 0",
                ]
                cls_res, _ = run_asm(lines)
                self.assertEqual(cls_res, interp_v_ffbh_i32(val))

    def test_scalar_s_flbit_and_s_ff1(self):
        values = [0, 1, 2, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 0x00008000]
        for val in values:
            with self.subTest(val=hex(val)):
                lines = [
                    f"s_mov_b32 s2, {val:#x}",
                    "s_flbit_i32_b32 s0, s2",
                    "s_ff1_i32_b32 s1, s2",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ]
                clz_res, ctz_res = run_asm(lines)
                self.assertEqual(clz_res, interp_v_ffbh_u32(val))
                self.assertEqual(ctz_res, interp_v_ffbl_b32(val))


class TestWaveLaneSimulation(unittest.TestCase):
    """Simulate 64-lane wave vector execution of bit-counting operations."""

    def test_64_lane_simd_clz_ctz(self):
        lane_values = []
        for lane in range(64):
            # Alternating bit allocations across lanes
            if lane < 32:
                lane_values.append(1 << lane)
            else:
                lane_values.append((1 << (63 - lane)) | (1 << ((lane * 3) % 32)))

        for lane, val in enumerate(lane_values):
            with self.subTest(lane=lane, val=hex(val)):
                clz_want = interp_v_clz_i32_u32(val)
                ctz_want = interp_v_ctz_i32_b32(val)
                cls_want = interp_v_cls_i32(val)

                clz_got = run_spirv_clz(val, hw_zero=False)
                ctz_got = run_spirv_ctz(val, hw_zero=False)
                cls_got = run_spirv_cls(val, hw_semantics=False)

                self.assertEqual(clz_got, clz_want)
                self.assertEqual(ctz_got, ctz_want)
                self.assertEqual(cls_got, cls_want)


if __name__ == "__main__":
    unittest.main()
