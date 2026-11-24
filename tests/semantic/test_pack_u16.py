"""Semantic interpreter tests for AMD RDNA vector integer packing (v_pack_b32_u16).

AMD RDNA VALU Instruction Set Architecture (ISA):
  v_pack_b32_u16 dst, src0, src1 [op_sel:[s0, s1]] [clamp]
  Operation: Packs two 16-bit integer values into a 32-bit vector register (VGPR).
    - dst[15:0]  = half_word(src0, op_sel[0]) [clamped to [0, 0xFFFF] if clamp]
    - dst[31:16] = half_word(src1, op_sel[1]) [clamped to [0, 0xFFFF] if clamp]

Modifiers and Semantics:
  - op_sel:[s0, s1]:
      s0 = 0: select low half-word (bits 15:0) of src0
      s0 = 1: select high half-word (bits 31:16) of src0
      s1 = 0: select low half-word (bits 15:0) of src1
      s1 = 1: select high half-word (bits 31:16) of src1
  - Half-word Truncation:
      When clamp is not asserted, bits outside the selected 16-bit window are
      discarded (masked to 0xFFFF).
  - Clamping / Saturation:
      When clamp is asserted on integer packing, values outside [0, 65535] saturate
      to the unsigned 16-bit limits:
        - Unsigned saturation (UClamp): values > 0xFFFF saturate to 0xFFFF.
        - Signed saturation (SClamp): values < 0 saturate to 0; values > 0xFFFF
          saturate to 0xFFFF.

In SPIR-V (as executed in spirv_mini_interp.py):
  1. Logical shift/mask/clamp formulation:
       %s0_val = OpShiftRightLogical %uint %src0 %c16  (if op_sel[0])
       %h0 = OpExtInst %uint %glsl UClamp %s0_val %c0 %c65535 (or OpBitwiseAnd %s0_val %c65535)
       %s1_val = OpShiftRightLogical %uint %src1 %c16  (if op_sel[1])
       %h1 = OpExtInst %uint %glsl UClamp %s1_val %c0 %c65535 (or OpBitwiseAnd %s1_val %c65535)
       %h1_hi = OpShiftLeftLogical %uint %h1 %c16
       %res = OpBitwiseOr %uint %h0 %h1_hi
  2. Bitfield extract/insert formulation:
       %h0 = OpBitFieldUExtract %uint %src0 %off0 %c16
       %h1 = OpBitFieldUExtract %uint %src1 %off1 %c16
       %res = OpBitFieldInsert %uint %h0 %h1 %c16 %c16

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import itertools
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

import rdna2spirv
import spirv_mini_interp as si

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def ref_extract_half(val: int, hi: bool = False) -> int:
    """Extract the low or high 16-bit half-word with 16-bit truncation."""
    val &= MASK32
    return ((val >> 16) if hi else val) & MASK16


def ref_clamp_u16(val: int, signed: bool = False) -> int:
    """Clamp integer value to unsigned 16-bit range [0, 65535]."""
    if signed:
        s = s32(val)
        if s < 0:
            return 0
        if s > MASK16:
            return MASK16
        return s & MASK16
    u = val & MASK32
    return min(u, MASK16)


def ref_pack_b32_u16(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    clamp: bool = False,
    signed: bool = False,
) -> int:
    """Pure-Python reference oracle for RDNA2 vector integer packing v_pack_b32_u16."""
    s0 &= MASK32
    s1 &= MASK32
    val0 = (s0 >> 16) if op_sel[0] else s0
    val1 = (s1 >> 16) if op_sel[1] else s1

    if clamp:
        h0 = ref_clamp_u16(val0, signed=signed)
        h1 = ref_clamp_u16(val1, signed=signed)
    else:
        h0 = val0 & MASK16
        h1 = val1 & MASK16

    return (h0 & MASK16) | ((h1 & MASK16) << 16)


def ref_decompose_pack(dword: int) -> tuple[int, int]:
    """Decompose a 32-bit packed word into (lo_u16, hi_u16)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


class PackU16Interpreter:
    """A lightweight semantic interpreter for VALU integer packing instructions.

    Maintains vector (VGPR) and scalar (SGPR) register banks and executes
    tokenized or text assembly instructions modeling v_pack_b32_u16.
    """

    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def parse_operand(self, token: str) -> int:
        """Resolve a register or literal operand to its current 32-bit unsigned value."""
        token = token.strip().rstrip(",")
        if not token:
            raise ValueError("Empty operand token")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        try:
            return int(token) & MASK32
        except ValueError:
            raise ValueError(f"Invalid operand token: {token}")

    def execute_line(self, line: str) -> None:
        """Parse and execute a single RDNA assembly line."""
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return

        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        # Tokenize args by commas/whitespace while preserving brackets like op_sel:[x,y]
        tokens = []
        buf = []
        bracket_depth = 0
        for char in args_str:
            if char == "[":
                bracket_depth += 1
                buf.append(char)
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
                buf.append(char)
            elif char in (",", " ", "\t") and bracket_depth == 0:
                if buf:
                    tokens.append("".join(buf).strip())
                    buf = []
            else:
                buf.append(char)
        if buf:
            tokens.append("".join(buf).strip())
        tokens = [t for t in tokens if t]

        if opcode in ("v_mov_b32", "s_mov_b32"):
            if len(tokens) < 2:
                raise ValueError(f"Instruction {opcode} requires at least dst and src")
            dst_token = tokens[0].lower()
            val = self.parse_operand(tokens[1])
            if dst_token.startswith("v") and dst_token[1:].isdigit():
                self.vgprs[int(dst_token[1:])] = val
            elif dst_token.startswith("s") and dst_token[1:].isdigit():
                self.sgprs[int(dst_token[1:])] = val
            else:
                raise ValueError(f"Invalid destination register: {dst_token}")

        elif opcode == "v_pack_b32_u16":
            if len(tokens) < 3:
                raise ValueError("v_pack_b32_u16 requires at least dst, src0, src1")
            dst_token = tokens[0].lower()
            if not (dst_token.startswith("v") and dst_token[1:].isdigit()):
                raise ValueError(f"Destination must be a VGPR: {dst_token}")
            dst_idx = int(dst_token[1:])
            if not (0 <= dst_idx < 256):
                raise IndexError(f"VGPR index out of range: {dst_token}")

            s0_val = self.parse_operand(tokens[1])
            s1_val = self.parse_operand(tokens[2])

            op_sel = (0, 0)
            clamp = False

            for mod in tokens[3:]:
                mod_lower = mod.lower()
                if mod_lower == "clamp":
                    clamp = True
                elif mod_lower.startswith("op_sel:[") and mod_lower.endswith("]"):
                    inner = mod_lower[8:-1].split(",")
                    if len(inner) != 2:
                        raise ValueError(f"Malformed op_sel modifier: {mod}")
                    op_sel = (int(inner[0].strip()), int(inner[1].strip()))
                else:
                    raise ValueError(f"Unrecognized modifier: {mod}")

            res = ref_pack_b32_u16(s0_val, s1_val, op_sel=op_sel, clamp=clamp)
            self.vgprs[dst_idx] = res

        else:
            raise NotImplementedError(f"Unsupported opcode: {opcode}")

    def execute_program(self, program: str) -> None:
        """Execute a multiline assembly sequence."""
        for line in program.strip().splitlines():
            self.execute_line(line)


def run_spirv_pack_logical(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    clamp: bool = False,
    signed: bool = False,
) -> int:
    """Execute SPIR-V shader sequence modeling v_pack_b32_u16 via shift/bitwise/clamp ops."""
    s0_u = s0 & MASK32
    s1_u = s1 & MASK32

    h0_extract = (
        "%s0_val = OpShiftRightLogical %uint %c_s0 %c16\n"
        if op_sel[0]
        else "%s0_val = OpCopyObject %uint %c_s0\n"
    )
    h1_extract = (
        "%s1_val = OpShiftRightLogical %uint %c_s1 %c16\n"
        if op_sel[1]
        else "%s1_val = OpCopyObject %uint %c_s1\n"
    )

    if clamp:
        clamp_op = "SClamp" if signed else "UClamp"
        h0_proc = f"%h0 = OpExtInst %uint %glsl {clamp_op} %s0_val %c0 %c65535\n"
        h1_proc = f"%h1 = OpExtInst %uint %glsl {clamp_op} %s1_val %c0 %c65535\n"
    else:
        h0_proc = "%h0 = OpBitwiseAnd %uint %s0_val %c65535\n"
        h1_proc = "%h1 = OpBitwiseAnd %uint %s1_val %c65535\n"

    spv = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0_u}
%c_s1 = OpConstant %uint {s1_u}
%c0 = OpConstant %uint 0
%c16 = OpConstant %uint 16
%c65535 = OpConstant %uint 65535
%main = OpFunction %void None %fn
%entry = OpLabel
{h0_extract}{h0_proc}{h1_extract}{h1_proc}%h1_hi = OpShiftLeftLogical %uint %h1 %c16
%res = OpBitwiseOr %uint %h0 %h1_hi
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(spv)["%out"][0]


def run_spirv_pack_bitfield(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
) -> int:
    """Execute SPIR-V shader sequence modeling unclamped v_pack_b32_u16 via bitfield ops."""
    off0 = 16 if op_sel[0] else 0
    off1 = 16 if op_sel[1] else 0
    spv = f"""OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c16 = OpConstant %uint 16
%c_off0 = OpConstant %uint {off0}
%c_off1 = OpConstant %uint {off1}
%main = OpFunction %void None %fn
%entry = OpLabel
%h0 = OpBitFieldUExtract %uint %c_s0 %c_off0 %c16
%h1 = OpBitFieldUExtract %uint %c_s1 %c_off1 %c16
%res = OpBitFieldInsert %uint %h0 %h1 %c16 %c16
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(spv)["%out"][0]


class TestPackU16ReferenceOracle(unittest.TestCase):
    """Unit tests for the independent Python reference oracle modeling v_pack_b32_u16."""

    def test_basic_packing_identity(self):
        self.assertEqual(ref_pack_b32_u16(0, 0), 0)
        self.assertEqual(ref_pack_b32_u16(0x1234, 0x5678), 0x56781234)
        self.assertEqual(ref_pack_b32_u16(0x00FF, 0xFF00), 0xFF0000FF)
        self.assertEqual(ref_pack_b32_u16(0xAAAA, 0x5555), 0x5555AAAA)

    def test_low_half_truncation_without_clamp(self):
        self.assertEqual(ref_pack_b32_u16(0xDEAD1234, 0xBEEF5678), 0x56781234)
        self.assertEqual(ref_pack_b32_u16(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_pack_b32_u16(0xFFFF0000, 0xFFFF0000), 0x00000000)
        self.assertEqual(ref_pack_b32_u16(0x12340001, 0x56780002), 0x00020001)

    def test_op_sel_permutations(self):
        s0 = 0xA1B2C3D4
        s1 = 0xE5F60718

        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 0)), 0x0718C3D4)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 0)), 0x0718A1B2)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 1)), 0xE5F6C3D4)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 1)), 0xE5F6A1B2)

    def test_unsigned_clamping(self):
        self.assertEqual(ref_pack_b32_u16(100, 200, clamp=True), (200 << 16) | 100)
        self.assertEqual(ref_pack_b32_u16(65535, 65535, clamp=True), 0xFFFFFFFF)

        self.assertEqual(ref_pack_b32_u16(65536, 10, clamp=True), (10 << 16) | 0xFFFF)
        self.assertEqual(ref_pack_b32_u16(10, 70000, clamp=True), (0xFFFF << 16) | 10)
        self.assertEqual(ref_pack_b32_u16(0xFFFFFFFF, 0xFFFFFFFF, clamp=True), 0xFFFFFFFF)
        self.assertEqual(ref_pack_b32_u16(0x80000000, 0x7FFFFFFF, clamp=True), 0xFFFFFFFF)

    def test_signed_clamping(self):
        self.assertEqual(ref_pack_b32_u16(-1, 500, clamp=True, signed=True), 500 << 16)
        self.assertEqual(ref_pack_b32_u16(500, -100, clamp=True, signed=True), 500)
        self.assertEqual(ref_pack_b32_u16(0x80000000, 0x80000000, clamp=True, signed=True), 0)

        self.assertEqual(ref_pack_b32_u16(70000, -5, clamp=True, signed=True), 0x0000FFFF)
        self.assertEqual(ref_pack_b32_u16(-5, 0x7FFFFFFF, clamp=True, signed=True), 0xFFFF0000)

    def test_clamping_with_op_sel(self):
        s0 = 50
        s1 = 100000

        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 0), clamp=True), (0xFFFF << 16) | 50)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 0), clamp=True), 0xFFFF0000)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(0, 1), clamp=True), (1 << 16) | 50)
        self.assertEqual(ref_pack_b32_u16(s0, s1, op_sel=(1, 1), clamp=True), 1 << 16)

    def test_self_packing_broadcast_and_swap(self):
        v = 0x12345678

        self.assertEqual(ref_pack_b32_u16(v, v, op_sel=(0, 0)), 0x56785678)
        self.assertEqual(ref_pack_b32_u16(v, v, op_sel=(1, 1)), 0x12341234)
        self.assertEqual(ref_pack_b32_u16(v, v, op_sel=(1, 0)), 0x56781234)
        self.assertEqual(ref_pack_b32_u16(v, v, op_sel=(0, 1)), 0x12345678)

    def test_decomposition_oracle(self):
        self.assertEqual(ref_decompose_pack(0x56781234), (0x1234, 0x5678))
        self.assertEqual(ref_decompose_pack(0xFFFFFFFF), (0xFFFF, 0xFFFF))
        self.assertEqual(ref_decompose_pack(0x00000000), (0, 0))

        test_pairs = [(0x1111, 0x2222), (0xDEAD, 0xBEEF), (0x0001, 0x8000), (0xFFFF, 0x0000)]
        for lo, hi in test_pairs:
            dword = ref_pack_b32_u16(lo, hi)
            self.assertEqual(ref_decompose_pack(dword), (lo, hi))


class TestPackU16InterpreterExecution(unittest.TestCase):
    """Unit tests for the PackU16Interpreter executing assembly programs."""

    def setUp(self):
        self.interp = PackU16Interpreter()

    def test_single_pack_instruction(self):
        program = """
        v_mov_b32 v1, 0x1234
        v_mov_b32 v2, 0x5678
        v_pack_b32_u16 v0, v1, v2
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0x56781234)

    def test_op_sel_modifier_in_assembly(self):
        program = """
        v_mov_b32 v1, 0xA1B2C3D4
        v_mov_b32 v2, 0xE5F60718
        v_pack_b32_u16 v0, v1, v2 op_sel:[1,0]
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0x0718A1B2)

    def test_clamp_modifier_in_assembly(self):
        program = """
        v_mov_b32 v1, 100000
        v_mov_b32 v2, 20
        v_pack_b32_u16 v0, v1, v2 clamp
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], (20 << 16) | 0xFFFF)

    def test_vertex_attribute_packing_pipeline(self):
        """Pack 4 normalized 16-bit halfword vertex attributes into 2 32-bit registers."""
        program = """
        v_mov_b32 v0, 0x0001
        v_mov_b32 v1, 0x0002
        v_mov_b32 v2, 0x0003
        v_mov_b32 v3, 0x0004
        v_pack_b32_u16 v4, v0, v1
        v_pack_b32_u16 v5, v2, v3
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[4], 0x00020001)
        self.assertEqual(self.interp.vgprs[5], 0x00040003)

    def test_high_halfword_swap(self):
        program = """
        v_mov_b32 v1, 0x12345678
        v_pack_b32_u16 v0, v1, v1 op_sel:[1,0]
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0x56781234)

    def test_chained_packing_with_scalar_and_literals(self):
        program = """
        s_mov_b32 s0, 0xABCD
        v_mov_b32 v1, 0x1234
        v_pack_b32_u16 v0, v1, s0
        v_pack_b32_u16 v2, v0, 0x9999 op_sel:[1,0]
        """
        self.interp.execute_program(program)
        self.assertEqual(self.interp.vgprs[0], 0xABCD1234)
        # v0 high half is 0xABCD, literal low half is 0x9999
        self.assertEqual(self.interp.vgprs[2], 0x9999ABCD)

    def test_invalid_syntax_and_bounds_handling(self):
        with self.assertRaises(ValueError):
            self.interp.execute_line("v_pack_b32_u16 v0, v1")
        with self.assertRaises(NotImplementedError):
            self.interp.execute_line("v_unknown_op v0, v1")
        with self.assertRaises(IndexError):
            self.interp.execute_line("v_pack_b32_u16 v999, v0, v1")


class TestPackU16SpirvInterp(unittest.TestCase):
    """Semantic tests executing SPIR-V code in spirv_mini_interp and cross-checking oracles."""

    VECTORS = (
        (0, 0),
        (1, 1),
        (0x1234, 0x5678),
        (0xA1B2C3D4, 0xE5F60718),
        (0xFFFFFFFF, 0x00000000),
        (0x00000000, 0xFFFFFFFF),
        (0x80000000, 0x7FFFFFFF),
        (0x55555555, 0xAAAAAAAA),
        (0x0000FFFF, 0xFFFF0000),
        (0x00010000, 0x00020000),
    )

    def test_spirv_unclamped_all_op_sel_modes(self):
        for s0, s1 in self.VECTORS:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=False)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=False)
                    self.assertEqual(got, expected)

    def test_spirv_unsigned_clamping(self):
        clamp_vectors = (
            (0, 0),
            (100, 200),
            (65535, 65535),
            (65536, 100),
            (100, 70000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x7FFFFFFF),
            (0x00010000, 0x00020000),
        )
        for s0, s1 in clamp_vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=True, signed=False)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=True, signed=False)
                    self.assertEqual(got, expected)

    def test_spirv_signed_clamping(self):
        signed_vectors = (
            (0, 0),
            (-1, 500),
            (500, -100),
            (-5000, -5000),
            (70000, -1),
            (-1, 0x7FFFFFFF),
            (0x80000000, 0x7FFFFFFF),
        )
        for s0, s1 in signed_vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=s0, s1=s1, op_sel=op_sel):
                    expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=True, signed=True)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=True, signed=True)
                    self.assertEqual(got, expected)

    def test_spirv_bitfield_equivalence(self):
        for s0, s1 in self.VECTORS:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    logical_res = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=False)
                    bitfield_res = run_spirv_pack_bitfield(s0, s1, op_sel=op_sel)
                    self.assertEqual(bitfield_res, logical_res)

    def test_deterministic_fuzzing_matches_oracle(self):
        rng = random.Random(0x1616)
        for _ in range(40):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            op_sel = (rng.choice((0, 1)), rng.choice((0, 1)))
            clamp = rng.choice((True, False))

            expected = ref_pack_b32_u16(s0, s1, op_sel=op_sel, clamp=clamp)
            got = run_spirv_pack_logical(s0, s1, op_sel=op_sel, clamp=clamp)
            self.assertEqual(got, expected)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated packing or missing masks fail assertions."""

    def test_operand_order_reversal_detected(self):
        s0, s1 = 0x11112222, 0x33334444
        correct = ref_pack_b32_u16(s0, s1, op_sel=(0, 0))
        reversed_operands = ref_pack_b32_u16(s1, s0, op_sel=(0, 0))
        self.assertNotEqual(correct, reversed_operands)

    def test_missing_high_word_shift_detected(self):
        s0, s1 = 0x1234, 0x5678
        correct = ref_pack_b32_u16(s0, s1)
        sabotaged_no_shift = (s0 & MASK16) | (s1 & MASK16)
        self.assertNotEqual(correct, sabotaged_no_shift)

    def test_off_by_one_shift_detected(self):
        s0, s1 = 0x1234, 0x5678
        correct = ref_pack_b32_u16(s0, s1)
        sabotaged_shift_15 = (s0 & MASK16) | ((s1 & MASK16) << 15)
        self.assertNotEqual(correct, sabotaged_shift_15)

    def test_missing_mask_truncation_detected(self):
        s0, s1 = 0xDEAD1234, 0xBEEF5678
        correct = ref_pack_b32_u16(s0, s1)
        sabotaged_no_truncation = (s0 | (s1 << 16)) & MASK32
        self.assertNotEqual(correct, sabotaged_no_truncation)

    def test_modulo_wrap_instead_of_clamp_detected(self):
        s0, s1 = 70000, 20
        clamped = ref_pack_b32_u16(s0, s1, clamp=True)
        wrapped = ref_pack_b32_u16(s0, s1, clamp=False)
        self.assertNotEqual(clamped, wrapped)

    def test_signed_vs_unsigned_clamp_discrepancy_detected(self):
        s0, s1 = -10, 10
        uclamped = ref_pack_b32_u16(s0, s1, clamp=True, signed=False)
        sclamped = ref_pack_b32_u16(s0, s1, clamp=True, signed=True)
        self.assertNotEqual(uclamped, sclamped)

    def test_inverted_op_sel_detected(self):
        s0, s1 = 0x12345678, 0x9ABCDEF0
        correct = ref_pack_b32_u16(s0, s1, op_sel=(0, 0))
        inverted = ref_pack_b32_u16(s0, s1, op_sel=(1, 1))
        self.assertNotEqual(correct, inverted)


class TestRdna2SpirvTranslatorStatus(unittest.TestCase):
    """Verify translator behavior on synthetic RDNA assembly containing v_pack_b32_u16."""

    def test_v_pack_b32_u16_assembly_translation_status(self):
        shader_text = "; ps_pack_test\n\tv_pack_b32_u16 v0, v1, v2\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            disasm_dir = os.path.join(tmp, "disasm")
            os.mkdir(disasm_dir)
            src_path = os.path.join(disasm_dir, "ps_pack_test.s")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(shader_text)
            with open(os.path.join(tmp, "ps_pack_test.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                try:
                    spv = rdna2spirv.translate(src_path)
                    self.assertIsInstance(spv, str)
                except rdna2spirv.Unsupported as e:
                    self.assertIn("v_pack_b32_u16", str(e))


if __name__ == "__main__":
    unittest.main()
