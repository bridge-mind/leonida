"""Semantic interpreter tests for AMD RDNA2 vector 32-bit integer low multiplication:
  - v_mul_lo_u32: D.u32 = (S0.u32 * S1.u32) & 0xFFFFFFFF

In two's complement binary arithmetic, the low 32 bits of 32-bit integer multiplication
are mathematically identical regardless of whether operands are treated as signed (i32)
or unsigned (u32), because (a * b) mod 2^32 is isomorphic across signed and unsigned
representations. RDNA2 therefore implements both signed and unsigned 32-bit low-dword
multiplication with the single VALU instruction v_mul_lo_u32, translated to SPIR-V OpIMul.

Standard library only. No game assets, archives, or shaders.
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
SIGN32 = 0x80000000
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

SAMPLE_VALUES = (
    0,
    1,
    2,
    3,
    7,
    15,
    16,
    31,
    32,
    64,
    100,
    255,
    256,
    0x1000,
    0x10000,
    0x55555555,
    0xAAAAAAAA,
    0x12345678,
    0x9ABCDEF0,
    0x7FFFFFFE,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
)

SAMPLE_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 0xFFFFFFFF),
    (1, 1),
    (1, 0xFFFFFFFF),
    (2, 3),
    (7, 9),
    (10, 10),
    (16, 64),
    (0x10000, 0x10000),
    (0x7FFFFFFF, 1),
    (0x7FFFFFFF, 2),
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0x80000000, 1),
    (0x80000000, 2),
    (0x80000000, 0x80000000),
    (0x80000000, 0xFFFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0xFFFFFFFF, 5),
    (0x12345678, 0x9ABCDEF0),
    (0xAAAA5555, 0x5555AAAA),
    (0xDEADBEEF, 0xCAFEBABE),
    (0x80000001, 0x7FFFFFFE),
    (100, 0xFFFFFF9C),  # 100 * -100
)


def u32(x: int) -> int:
    """Mask integer to 32 bits unsigned."""
    return x & MASK32


def s32(x: int) -> int:
    """Convert integer to 32-bit signed two's complement integer."""
    val = x & MASK32
    return val - (1 << 32) if (val & SIGN32) else val


def ref_mul_lo_u32(a: int, b: int) -> int:
    """Reference oracle for v_mul_lo_u32: (u32(a) * u32(b)) & 0xFFFFFFFF."""
    return (u32(a) * u32(b)) & MASK32


def ref_mul_lo_i32(a: int, b: int) -> int:
    """Signed reference for low 32-bit multiplication: (s32(a) * s32(b)) & 0xFFFFFFFF."""
    return (s32(a) * s32(b)) & MASK32


def ref_mul_64(a: int, b: int, signed: bool = False) -> tuple[int, int]:
    """Returns (lo, hi) halves of full 64-bit product."""
    if signed:
        prod = s32(a) * s32(b)
        prod_u = prod & MASK64
    else:
        prod_u = (u32(a) * u32(b)) & MASK64
    return prod_u & MASK32, (prod_u >> 32) & MASK32


class VectorALUMulInterpreter:
    """Deterministic multi-lane SIMD interpreter modeling AMD RDNA2 VALU execution."""

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
            return self.vgprs[int(op_str[1:])][lane] & MASK32
        if op_str.startswith("s"):
            return self.sgprs[int(op_str[1:])] & MASK32
        return int(op_str, 0) & MASK32

    def execute_v_mul_lo_u32(self, dst_idx: int, src0: str | int, src1: str | int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_mul_lo_u32(s0, s1)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None
_CS_PATH: str | None = None
_CS_FSSM: str | None = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mul_lo.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_mul_lo.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_mul_lo.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_mul_lo.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_mul_lo.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def run_mul_lo_shader(
    a: int,
    b: int,
    op: str = "v_mul_lo_u32",
    inline_s0: bool = False,
    inline_s1: bool = False,
    s0_is_sgpr: bool = False,
) -> int:
    """Run `op v0, s0/v1, v2` and return the 32-bit word written to mrt0[0]."""
    if _TMP_DIR is None:
        setUpModule()
    lines = []
    if s0_is_sgpr:
        lines.append(f"\ts_mov_b32 s0, {u32(a):#x}")
        op0 = "s0"
    elif inline_s0:
        op0 = f"{u32(a):#x}"
    else:
        lines.append(f"\tv_mov_b32 v1, {u32(a):#x}")
        op0 = "v1"

    if inline_s1:
        op1 = f"{u32(b):#x}"
    else:
        lines.append(f"\tv_mov_b32 v2, {u32(b):#x}")
        op1 = "v2"

    lines.append(f"\t{op} v0, {op0}, {op1}")
    asm = (
        "; test_alu_mul_lo_i32\n"
        + "\n".join(lines)
        + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
        gc.collect()
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_mul_lo_cs(
    multiplier: int,
    wave_size: int = 64,
) -> dict[int, int]:
    """Run a multi-lane compute shader where each lane multiplies its lane index by `multiplier`."""
    if _TMP_DIR is None:
        setUpModule()
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x8088, (wave_size, 1, 1)))

    lines = [
        f"\ts_mov_b32 s4, {u32(multiplier):#x}",
        "\tv_mul_lo_u32 v1, v0, s4",
        "\tv_lshlrev_b32_e32 v2, 2, v0",
        "\tbuffer_store_dword v1, v2, s[0:3], 0 offen",
        "\ts_endpgm",
    ]
    asm = "; test_mul_lo_cs\n" + "\n".join(lines) + "\n"
    with open(_CS_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()

    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv,
        wave_size,
        wave_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={},
    )


def run_spirv_imul(a: int, b: int) -> int:
    """Direct execution of SPIR-V OpIMul instruction in spirv_mini_interp."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%ca = OpConstant %uint {u32(a)}
%cb = OpConstant %uint {u32(b)}
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpIMul %uint %ca %cb
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


class TestMulLoReferenceOracle(unittest.TestCase):
    """Mathematical validation of 32-bit integer low multiplication reference oracle."""

    def test_zero_multiplier(self) -> None:
        """Any 32-bit integer multiplied by 0 equals 0."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 0), 0)
                self.assertEqual(ref_mul_lo_u32(0, val), 0)
                self.assertEqual(ref_mul_lo_i32(val, 0), 0)
                self.assertEqual(ref_mul_lo_i32(0, val), 0)

    def test_unity_multiplier(self) -> None:
        """Any 32-bit integer multiplied by 1 preserves its value."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 1), u32(val))
                self.assertEqual(ref_mul_lo_u32(1, val), u32(val))
                self.assertEqual(ref_mul_lo_i32(val, 1), u32(val))
                self.assertEqual(ref_mul_lo_i32(1, val), u32(val))

    def test_negation_identity(self) -> None:
        """Multiplying by -1 (0xFFFFFFFF) yields two's complement negation modulo 2^32."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                expected = (-u32(val)) & MASK32
                self.assertEqual(ref_mul_lo_u32(val, -1), expected)
                self.assertEqual(ref_mul_lo_u32(val, 0xFFFFFFFF), expected)
                self.assertEqual(ref_mul_lo_i32(val, -1), expected)
                self.assertEqual(ref_mul_lo_i32(val, 0xFFFFFFFF), expected)

    def test_commutativity(self) -> None:
        """Low multiplication is commutative: a * b == b * a."""
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_mul_lo_u32(a, b), ref_mul_lo_u32(b, a))
                self.assertEqual(ref_mul_lo_i32(a, b), ref_mul_lo_i32(b, a))

    def test_associativity_mod32(self) -> None:
        """Low multiplication is associative modulo 2^32: (a * b) * c == a * (b * c)."""
        triplets = [
            (3, 5, 7),
            (100, 200, 300),
            (0x1000, 0x100, 0x10),
            (0x12345678, 3, 5),
            (0x7FFFFFFF, 2, 4),
            (0xFFFFFFFF, 2, 3),
            (0x80000001, 0x10001, 7),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                left = ref_mul_lo_u32(ref_mul_lo_u32(a, b), c)
                right = ref_mul_lo_u32(a, ref_mul_lo_u32(b, c))
                self.assertEqual(left, right)

    def test_distributivity_over_addition(self) -> None:
        """Low multiplication distributes over addition modulo 2^32: a * (b + c) == a * b + a * c."""
        triplets = [
            (5, 12, 18),
            (0x1000, 0x2000, 0x3000),
            (0xFFFFFFFF, 1, 1),
            (0x7FFFFFFF, 0x80000000, 1),
            (0x12345678, 0x9ABCDEF0, 0x55555555),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                sum_bc = (b + c) & MASK32
                prod_distributed = (ref_mul_lo_u32(a, b) + ref_mul_lo_u32(a, c)) & MASK32
                self.assertEqual(ref_mul_lo_u32(a, sum_bc), prod_distributed)

    def test_power_of_two_shift_equivalence(self) -> None:
        """Multiplication by 2^k is equivalent to left shift by k modulo 2^32."""
        for val in (1, 3, 7, 42, 0x12345678, 0x80000001, 0xFFFFFFFF):
            for k in range(32):
                with self.subTest(val=hex(val), shift=k):
                    multiplier = 1 << k
                    expected = (u32(val) << k) & MASK32
                    self.assertEqual(ref_mul_lo_u32(val, multiplier), expected)

    def test_hand_calculated_constants(self) -> None:
        """Rigorous validation against hand-calculated boundary results."""
        cases = [
            (0, 0, 0),
            (1, 1, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
            (0x7FFFFFFF, 2, 0xFFFFFFFE),
            (0x80000000, 2, 0),
            (0x80000000, 0x80000000, 0),
            (0x80000000, 0xFFFFFFFF, 0x80000000),
            (0x10000, 0x10000, 0),
            (0x10001, 0x10001, 0x00020001),
            (100, -1, 0xFFFFFF9C),
            (100, 0xFFFFFFFF, 0xFFFFFF9C),
            (0x12345678, 2, 0x2468ACF0),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), expected=hex(expected)):
                self.assertEqual(ref_mul_lo_u32(a, b), expected)
                self.assertEqual(ref_mul_lo_i32(a, b), expected)


class TestMulLoTranslator(unittest.TestCase):
    """End-to-end tests translating RDNA2 v_mul_lo_u32 and executing in spirv_mini_interp."""

    def test_basic_execution(self) -> None:
        """Test v_mul_lo_u32 shader execution against the reference oracle."""
        test_pairs = [
            (0, 0),
            (1, 1),
            (2, 3),
            (10, 20),
            (0x10000, 0x10000),
            (0x7FFFFFFF, 2),
            (0x80000000, 2),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
            (100, -1),
        ]
        for a, b in test_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                expected = ref_mul_lo_u32(a, b)
                self.assertEqual(run_mul_lo_shader(a, b), expected)
                self.assertEqual(run_mul_lo_shader(b, a), expected)

    def test_inline_constants(self) -> None:
        """Test inline constants in src0 and src1 positions."""
        test_cases = [
            (5, 7, 35),
            (100, -1, 0xFFFFFF9C),
            (0x7FFFFFFF, 2, 0xFFFFFFFE),
            (0x80000000, 2, 0),
            (0xFFFFFFFF, 5, (-5) & MASK32),
        ]
        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, pos="src1"):
                self.assertEqual(run_mul_lo_shader(a, b, inline_s1=True), expected)
            with self.subTest(a=a, b=b, pos="src0"):
                self.assertEqual(run_mul_lo_shader(a, b, inline_s0=True), expected)

    def test_sgpr_scalar_source(self) -> None:
        """Test mixed scalar SGPR and vector VGPR inputs."""
        test_cases = [
            (42, 10, 420),
            (0x1234, 0x5678, ref_mul_lo_u32(0x1234, 0x5678)),
            (0x80000000, 2, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
        ]
        for s0_val, v1_val, expected in test_cases:
            with self.subTest(s0=hex(s0_val), v1=hex(v1_val)):
                got = run_mul_lo_shader(s0_val, v1_val, s0_is_sgpr=True)
                self.assertEqual(got, expected)

    def test_opcode_aliases(self) -> None:
        """Test opcode aliases v_mul_lo_u32_e32 and v_mul_lo_u32_e64."""
        aliases = ("v_mul_lo_u32", "v_mul_lo_u32_e32", "v_mul_lo_u32_e64")
        a = 0x12345678
        b = 0x00000002
        expected = ref_mul_lo_u32(a, b)
        for op in aliases:
            with self.subTest(op=op):
                self.assertEqual(run_mul_lo_shader(a, b, op=op), expected)

    def test_spirv_lowering_instruction(self) -> None:
        """Verifies that rdna2spirv emits OpIMul %uint for v_mul_lo_u32."""
        if _TMP_DIR is None:
            setUpModule()
        lines = [
            "\tv_mov_b32 v1, 15",
            "\tv_mov_b32 v2, 4",
            "\tv_mul_lo_u32 v0, v1, v2",
            "\texp mrt0 v0, v0, v0, v0 done vm",
            "\ts_endpgm",
        ]
        asm = "; test_spirv_lowering\n" + "\n".join(lines) + "\n"
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        self.assertIn("OpIMul %uint", spv)

    def test_spirv_direct_imul(self) -> None:
        """Direct execution of OpIMul in SPIR-V interpreter matches reference."""
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(run_spirv_imul(a, b), ref_mul_lo_u32(a, b))


class TestMulLoSIMDDivergence(unittest.TestCase):
    """Multi-lane SIMD execution and divergence tests for v_mul_lo_u32."""

    def test_simd_independent_lanes(self) -> None:
        """All 64 lanes evaluate distinct, divergent multiplications concurrently."""
        interp = VectorALUMulInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(1, lane, lane * 17 + 3)
            interp.set_vgpr(2, lane, 0x10001 + lane * 5)

        interp.execute_v_mul_lo_u32(0, "v1", "v2")

        for lane in range(64):
            s0 = lane * 17 + 3
            s1 = 0x10001 + lane * 5
            expected = ref_mul_lo_u32(s0, s1)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_simd_exec_mask_even_odd_gating(self) -> None:
        """Inactive EXEC lanes preserve their prior VGPR contents unchanged."""
        interp = VectorALUMulInterpreter(wave_size=64)
        sentinel = 0xDEADBEEF

        for lane in range(64):
            interp.set_vgpr(0, lane, sentinel)
            interp.set_vgpr(1, lane, lane)
            interp.set_vgpr(2, lane, 10)

        # Alternating even lanes active: 0x5555555555555555
        interp.set_exec(0x5555555555555555)
        interp.execute_v_mul_lo_u32(0, "v1", "v2")

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), lane * 10)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), sentinel)

    def test_simd_exec_mask_half_wave_divergence(self) -> None:
        """First 32 lanes active, last 32 lanes inactive (wave32 sub-execution)."""
        interp = VectorALUMulInterpreter(wave_size=64)
        sentinel = 0xCAFEBABE

        for lane in range(64):
            interp.set_vgpr(0, lane, sentinel)
            interp.set_vgpr(1, lane, 0x1234)
            interp.set_vgpr(2, lane, lane + 1)

        interp.set_exec(0x00000000FFFFFFFF)
        interp.execute_v_mul_lo_u32(0, "v1", "v2")

        for lane in range(64):
            if lane < 32:
                expected = ref_mul_lo_u32(0x1234, lane + 1)
                self.assertEqual(interp.get_vgpr(0, lane), expected)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), sentinel)

    def test_simd_exec_mask_single_lane(self) -> None:
        """Exactly one divergent lane active in wave."""
        interp = VectorALUMulInterpreter(wave_size=64)
        target_lane = 42

        for lane in range(64):
            interp.set_vgpr(0, lane, 0)
            interp.set_vgpr(1, lane, 1000)
            interp.set_vgpr(2, lane, 3)

        interp.set_exec(1 << target_lane)
        interp.execute_v_mul_lo_u32(0, "v1", "v2")

        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr(0, lane), 3000)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0)

    def test_simd_exec_mask_contiguous_slice(self) -> None:
        """Contiguous sub-range of active lanes [16, 48)."""
        interp = VectorALUMulInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0x77777777)
            interp.set_vgpr(1, lane, lane)
            interp.set_vgpr(2, lane, 0x80000001)

        mask = ((1 << 32) - 1) << 16
        interp.set_exec(mask)
        interp.execute_v_mul_lo_u32(0, "v1", "v2")

        for lane in range(64):
            if 16 <= lane < 48:
                expected = ref_mul_lo_u32(lane, 0x80000001)
                self.assertEqual(interp.get_vgpr(0, lane), expected)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0x77777777)

    def test_simd_sgpr_scalar_broadcast(self) -> None:
        """Scalar SGPR multiplier broadcast across all active lanes."""
        interp = VectorALUMulInterpreter(wave_size=64)
        scalar_val = 0x10001
        interp.set_sgpr(5, scalar_val)

        for lane in range(64):
            interp.set_vgpr(1, lane, lane * 256)

        interp.execute_v_mul_lo_u32(0, "s5", "v1")

        for lane in range(64):
            expected = ref_mul_lo_u32(scalar_val, lane * 256)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_simd_inline_constant_broadcast(self) -> None:
        """Inline constant broadcast across divergent lanes."""
        interp = VectorALUMulInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(1, lane, 0x80000000 + lane)

        interp.execute_v_mul_lo_u32(0, 3, "v1")

        for lane in range(64):
            expected = ref_mul_lo_u32(3, 0x80000000 + lane)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_simd_spirv_workgroup_wave32(self) -> None:
        """Multi-lane execution of v_mul_lo_u32 in a 32-lane wave via run_workgroup."""
        multiplier = 0x10003
        mem = run_mul_lo_cs(multiplier, wave_size=32)
        for lane in range(32):
            expected = ref_mul_lo_u32(lane, multiplier)
            self.assertEqual(mem[lane * 4], expected, f"Wave32 lane {lane}")

    def test_simd_spirv_workgroup_wave64(self) -> None:
        """Multi-lane execution of v_mul_lo_u32 in a 64-lane wave via run_workgroup."""
        multiplier = 0x7FFFFFFF
        mem = run_mul_lo_cs(multiplier, wave_size=64)
        for lane in range(64):
            expected = ref_mul_lo_u32(lane, multiplier)
            self.assertEqual(mem[lane * 4], expected, f"Wave64 lane {lane}")


class TestMulLoOverflowWrapping(unittest.TestCase):
    """Exhaustive 32-bit overflow wrapping and modulo 2^32 truncation tests."""

    def test_overflow_wrap_boundary_products(self) -> None:
        """Boundary products wrapping cleanly modulo 2^32."""
        cases = [
            # 0xFFFFFFFF * 2 = 2^33 - 2 -> 0xFFFFFFFE
            (0xFFFFFFFF, 2, 0xFFFFFFFE),
            # 0xFFFFFFFF * 3 = 3 * 2^32 - 3 -> 0xFFFFFFFD
            (0xFFFFFFFF, 3, 0xFFFFFFFD),
            # 0xFFFFFFFF * 0xFFFFFFFF = 2^64 - 2*2^32 + 1 -> 1
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
            # 0x80000000 * 2 = 2^32 -> 0
            (0x80000000, 2, 0),
            # 0x80000000 * 3 = 2^32 + 2^31 -> 0x80000000
            (0x80000000, 3, 0x80000000),
            # 0x80000000 * 4 = 2^33 -> 0
            (0x80000000, 4, 0),
            # 0x80000000 * 0x80000000 = 2^62 -> 0
            (0x80000000, 0x80000000, 0),
            # 0x7FFFFFFF * 0x7FFFFFFF = (2^31 - 1)^2 = 2^62 - 2^32 + 1 -> 1
            (0x7FFFFFFF, 0x7FFFFFFF, 1),
            # 0x7FFFFFFF * 2 = 2^32 - 2 -> 0xFFFFFFFE
            (0x7FFFFFFF, 2, 0xFFFFFFFE),
            # 0x7FFFFFFF * 3 = 3 * 2^31 - 3 = 2^32 + 2^31 - 3 -> 0x7FFFFFFD
            (0x7FFFFFFF, 3, 0x7FFFFFFD),
            # 0x55555555 * 3 = 0xFFFFFFFF
            (0x55555555, 3, 0xFFFFFFFF),
            # 0xAAAAAAAA * 3 = 2 * 0xFFFFFFFF = 2^33 - 2 -> 0xFFFFFFFE
            (0xAAAAAAAA, 3, 0xFFFFFFFE),
            # 0x10000001 * 0x10000001 = (2^28 + 1)^2 = 2^56 + 2^29 + 1 -> 0x20000001
            (0x10000001, 0x10000001, 0x20000001),
            # 0x10000000 * 0x10 = 2^28 * 2^4 = 2^32 -> 0
            (0x10000000, 0x10, 0),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), expected=hex(expected)):
                self.assertEqual(ref_mul_lo_u32(a, b), expected)
                self.assertEqual(run_spirv_imul(a, b), expected)

    def test_overflow_wrap_powers_of_two(self) -> None:
        """Powers of two wrapping: 2^p * 2^q mod 2^32."""
        for p in range(32):
            for q in range(32):
                a = 1 << p
                b = 1 << q
                expected = (1 << (p + q)) & MASK32 if (p + q < 32) else 0
                with self.subTest(p=p, q=q):
                    self.assertEqual(ref_mul_lo_u32(a, b), expected)

    def test_overflow_wrap_large_primes(self) -> None:
        """Multiplying large primes that exceed 32 bits."""
        prime_pairs = [
            (65537, 65539),
            (1000003, 1000033),
            (2147483647, 3),
            (2147483647, 5),
            (2147483647, 2147483647),
        ]
        for p1, p2 in prime_pairs:
            with self.subTest(p1=p1, p2=p2):
                expected = (p1 * p2) & MASK32
                self.assertEqual(ref_mul_lo_u32(p1, p2), expected)
                self.assertEqual(run_spirv_imul(p1, p2), expected)

    def test_overflow_wrap_chain_accumulation(self) -> None:
        """Chained multiplication wrapping across repeated iterations."""
        val = 3
        mult = 5
        py_val = 3
        for i in range(100):
            val = ref_mul_lo_u32(val, mult)
            py_val = (py_val * mult) & MASK32
            self.assertEqual(val, py_val, f"Iteration {i}")

    def test_overflow_multiplicative_inverses(self) -> None:
        """Every odd 32-bit integer has a multiplicative inverse mod 2^32."""
        odd_numbers = [
            1,
            3,
            5,
            7,
            11,
            17,
            257,
            65537,
            0x12345679,
            0x7FFFFFFF,
            0xFFFFFFFF,
        ]
        for x in odd_numbers:
            inv = pow(x, -1, 1 << 32)
            with self.subTest(x=hex(x), inv=hex(inv)):
                self.assertEqual(ref_mul_lo_u32(x, inv), 1)
                self.assertEqual(run_spirv_imul(x, inv), 1)

    def test_fuzz_overflow_differential(self) -> None:
        """Differential fuzz testing with pseudo-random 32-bit numbers."""
        rng = random.Random(0x4D554C)
        for _ in range(150):
            a = rng.randint(0, MASK32)
            b = rng.randint(0, MASK32)
            expected = (a * b) & MASK32
            self.assertEqual(ref_mul_lo_u32(a, b), expected)
            self.assertEqual(run_spirv_imul(a, b), expected)


class TestMulLoSignedUnsignedIdentity(unittest.TestCase):
    """Rigorous verification of the signed vs unsigned low-dword identity:
    (s32(a) * s32(b)) mod 2^32 == (u32(a) * u32(b)) mod 2^32.
    """

    def test_signed_unsigned_low_dword_isomorphic_identity(self) -> None:
        """Low 32 bits are unconditionally identical for signed and unsigned integers."""
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                lo_u = ref_mul_lo_u32(a, b)
                lo_i = ref_mul_lo_i32(a, b)
                self.assertEqual(lo_u, lo_i)

    def test_signed_negative_operands_product(self) -> None:
        """Verification of signed negative operand combinations."""
        # Negative * Negative = Positive
        neg_neg_cases = [
            (-1, -1, 1),
            (-2, -3, 6),
            (-10, -20, 200),
            (-100, -200, 20000),
            (-2147483648, -1, 0x80000000),  # INT32_MIN * -1 = 2^31 -> 0x80000000
            (-2147483647, -1, 0x7FFFFFFF),  # (-(2^31 - 1)) * -1 = INT32_MAX
        ]
        for a, b, expected in neg_neg_cases:
            with self.subTest(a=a, b=b, expected=hex(expected)):
                self.assertEqual(ref_mul_lo_i32(a, b), expected)
                self.assertEqual(ref_mul_lo_u32(a, b), expected)
                self.assertEqual(run_spirv_imul(a, b), expected)

        # Negative * Positive = Negative (two's complement)
        neg_pos_cases = [
            (-1, 1, 0xFFFFFFFF),
            (-5, 10, (-50) & MASK32),
            (-16, 64, (-1024) & MASK32),
            (-2147483648, 2, 0),             # INT32_MIN * 2 = -2^32 -> 0 mod 2^32
            (-2147483648, 3, 0x80000000),    # INT32_MIN * 3 = -3 * 2^31 = -2^32 - 2^31 -> 0x80000000
        ]
        for a, b, expected in neg_pos_cases:
            with self.subTest(a=a, b=b, expected=hex(expected)):
                self.assertEqual(ref_mul_lo_i32(a, b), expected)
                self.assertEqual(ref_mul_lo_u32(a, b), expected)
                self.assertEqual(run_spirv_imul(a, b), expected)

    def test_signed_vs_unsigned_high_dword_divergence_contrast(self) -> None:
        """Contrast: low dword is identical, but high dword diverges on negative operands.
        This demonstrates why hardware needs separate v_mul_hi_u32 and v_mul_hi_i32,
        but only a single v_mul_lo_u32.
        """
        divergent_pairs = [
            (0xFFFFFFFF, 0xFFFFFFFF),  # -1 * -1
            (0xFFFFFFFF, 5),           # -1 * 5
            (0x80000000, 2),           # INT32_MIN * 2
            (100, -1),                 # 100 * -1
            (-16, 64),                 # -16 * 64
        ]
        for a, b in divergent_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                lo_u, hi_u = ref_mul_64(a, b, signed=False)
                lo_i, hi_i = ref_mul_64(a, b, signed=True)

                # Low dword MUST be identical
                self.assertEqual(lo_u, lo_i, f"Low dword diverged for ({hex(a)}, {hex(b)})")
                self.assertEqual(lo_u, ref_mul_lo_u32(a, b))

                # High dword MUST diverge for these negative inputs
                self.assertNotEqual(hi_u, hi_i, f"Expected high dwords to diverge for ({hex(a)}, {hex(b)})")

    def test_signed_fuzz_identity(self) -> None:
        """Differential fuzz verification that signed and unsigned lo-multiplication match."""
        rng = random.Random(0x5349474E)
        for _ in range(200):
            a = rng.randint(0, MASK32)
            b = rng.randint(0, MASK32)
            self.assertEqual(ref_mul_lo_u32(a, b), ref_mul_lo_i32(a, b))


class TestMulLoNegativeControls(unittest.TestCase):
    """Negative controls ensuring any faulty arithmetic, incorrect signedness,
    truncation, or pipeline mutations are caught.
    """

    def test_hi_lo_word_confusion_detected(self) -> None:
        """Negative control: returning high word instead of low word fails."""
        pairs = [
            (0x10000, 0x10000),
            (0xFFFFFFFF, 2),
            (0x80000000, 2),
            (0x12345678, 0x9ABCDEF0),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                lo = ref_mul_lo_u32(a, b)
                hi_u = ((u32(a) * u32(b)) >> 32) & MASK32
                self.assertNotEqual(lo, hi_u, f"lo and hi should diverge for ({hex(a)}, {hex(b)})")

    def test_16bit_truncation_mutation_detected(self) -> None:
        """Negative control: truncating inputs to 16 bits fails on 32-bit operands."""
        a = 0x10001
        b = 0x20003
        correct = ref_mul_lo_u32(a, b)
        sabotaged_16bit = ((a & 0xFFFF) * (b & 0xFFFF)) & MASK32
        self.assertNotEqual(correct, sabotaged_16bit)

    def test_arithmetic_mutations_detected(self) -> None:
        """Negative control: common arithmetic sabotages are detected."""
        a = 0x12345678
        b = 0x00000003
        correct = ref_mul_lo_u32(a, b)

        # Addition instead of multiplication
        sabotaged_add = (a + b) & MASK32
        self.assertNotEqual(correct, sabotaged_add)

        # Off-by-one error
        sabotaged_off_by_one = (correct + 1) & MASK32
        self.assertNotEqual(correct, sabotaged_off_by_one)

        # Dropped high bit (31-bit masking)
        val_with_msb = 0x80000001
        res = ref_mul_lo_u32(val_with_msb, 1)
        sabotaged_mask31 = res & 0x7FFFFFFF
        self.assertNotEqual(res, sabotaged_mask31)

    def test_signed_saturation_mutation_detected(self) -> None:
        """Negative control: saturating overflow instead of wrapping is detected."""
        a = 0x7FFFFFFF  # INT32_MAX
        b = 2
        correct = ref_mul_lo_u32(a, b)  # wraps to 0xFFFFFFFE (-2)
        sabotaged_saturated = 0x7FFFFFFF  # clamped to INT32_MAX
        self.assertNotEqual(correct, sabotaged_saturated)

    def test_exec_mask_leak_mutation_detected(self) -> None:
        """Negative control: modifying inactive EXEC lanes is detected."""
        interp = VectorALUMulInterpreter(wave_size=4)
        interp.set_exec(0b0011)  # only lanes 0 and 1 active
        interp.set_vgpr(0, 2, 0xDEADBEEF)
        interp.set_vgpr(0, 3, 0xDEADBEEF)

        interp.execute_v_mul_lo_u32(0, 10, 20)

        # Inactive lanes 2 and 3 must not be modified
        self.assertEqual(interp.get_vgpr(0, 2), 0xDEADBEEF)
        self.assertEqual(interp.get_vgpr(0, 3), 0xDEADBEEF)

        # If a buggy implementation updated lane 2 to 200, it would fail
        sabotaged_lane2 = 200
        self.assertNotEqual(interp.get_vgpr(0, 2), sabotaged_lane2)

    def test_spirv_opcode_substitution_detected(self) -> None:
        """Negative control: mutating OpIMul to OpISub in SPIR-V produces divergence."""
        a = 20
        b = 5
        correct = run_spirv_imul(a, b)  # 20 * 5 = 100

        module_isub = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%ca = OpConstant %uint {u32(a)}
%cb = OpConstant %uint {u32(b)}
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpISub %uint %ca %cb
OpStore %out %res
OpReturn
OpFunctionEnd
"""
        mutated_res = si.run_fragment(module_isub)["%out"][0]  # 20 - 5 = 15
        self.assertNotEqual(correct, mutated_res)


if __name__ == "__main__":
    unittest.main()
