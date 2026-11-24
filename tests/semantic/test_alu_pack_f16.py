"""Semantic interpreter and translator tests for AMD RDNA2 v_pack_b32_f16 float packing.

v_pack_b32_f16 (VOP3 opcode 0x272 on GFX10 / RDNA2):
  dst[15:0]  = half_word(src0, op_sel[0])
  dst[31:16] = half_word(src1, op_sel[1])

Semantics:
  - Packs two 16-bit half-precision float values (or raw 16-bit bit patterns) from src0 and src1
    into a single 32-bit vector register (VGPR).
  - Unlike 16-bit ALU instructions that preserve the unwritten destination half, v_pack_b32_f16
    completely overwrites all 32 bits of the destination register.
  - Operand selection (op_sel:[s0, s1, s2]):
      s0 = 0: low 16 bits of src0 (bits 15:0)
      s0 = 1: high 16 bits of src0 (bits 31:16)
      s1 = 0: low 16 bits of src1 (bits 15:0)
      s1 = 1: high 16 bits of src1 (bits 31:16)
  - Source modifiers:
      abs (|src|): clears bit 15 (sign bit) of the selected half-word (evaluated first).
      neg (-src): flips bit 15 (sign bit) of the selected half-word (evaluated second).
  - Operands:
      VGPRs, SGPRs, literal constants, and inline constants (e.g. 1.0, 2.0, 0.5, 0.0, -1.0).
      Inline float constants place the f16 bit pattern in the low half and 0 in the high half;
      selecting the high half of an inline float constant therefore reads +0.0.
  - Bit-exact float representation:
      Preserves all IEEE 754 binary16 values without rounding or flushing:
      +0.0, -0.0, subnormals (denorms), normalized numbers, +inf, -inf, quiet NaNs (qNaN),
      and signaling NaNs (sNaN) with custom payload bits preserved intact.

Each test case verifies:
  1. Pure-Python reference oracle behavior.
  2. Synthetic RDNA2 pixel shader translation via tools/rdna2spirv.py and execution in
     tests/semantic/spirv_mini_interp.py.
  3. Direct SPIR-V shader sequence execution in spirv_mini_interp.py.
  4. Negative controls confirming fault sensitivity to operand swaps, off-by-one shifts,
     missing masks, and modifier bugs.

Standard library Python only; zero game data; Python 3.9+ compatible.
"""
from __future__ import annotations

import itertools
import math
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
MASK16 = 0xFFFF
SIGN16 = 0x8000
ABS16 = 0x7FFF
POISON_DST = 0xDEADBEEF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# IEEE 754-2008 binary16 float bit constants
POS_ZERO = 0x0000              # +0.0
NEG_ZERO = 0x8000              # -0.0
MIN_SUBNORMAL = 0x0001         # Smallest positive subnormal (2^-24)
MAX_SUBNORMAL = 0x03FF         # Largest positive subnormal
MIN_SUBNORMAL_NEG = 0x8001     # Smallest negative subnormal
MAX_SUBNORMAL_NEG = 0x83FF     # Largest negative subnormal
MIN_NORMAL = 0x0400            # Smallest positive normal (2^-14)
QUARTER = 0x3400               # 0.25
HALF = 0x3800                  # 0.5
THREE_QUARTERS = 0x3A00        # 0.75
ONE = 0x3C00                   # 1.0
ONE_POINT_FIVE = 0x3E00        # 1.5
TWO = 0x4000                   # 2.0
FOUR = 0x4400                  # 4.0
MAX_FINITE = 0x7BFF            # 65504.0 (largest finite half float)
NEG_QUARTER = 0xB400           # -0.25
NEG_HALF = 0xB800              # -0.5
NEG_ONE = 0xBC00               # -1.0
NEG_TWO = 0xC000               # -2.0
MIN_FINITE_NEG = 0xFBFF        # -65504.0
POS_INF = 0x7C00               # +Infinity
NEG_INF = 0xFC00               # -Infinity
POS_QNAN_DEFAULT = 0x7E00      # Quiet NaN (+qNaN default)
NEG_QNAN_DEFAULT = 0xFE00      # Quiet NaN (-qNaN default)
POS_QNAN_PAYLOAD = 0x7E01      # Quiet NaN with payload 1
POS_QNAN_HIGH = 0x7FFF         # Quiet NaN with all payload bits set
NEG_QNAN_PAYLOAD = 0xFE55      # Negative Quiet NaN with payload 0x55
POS_SNAN_DEFAULT = 0x7C01      # Signaling NaN (+sNaN)
POS_SNAN_PAYLOAD = 0x7DFF      # Signaling NaN with custom payload
NEG_SNAN_DEFAULT = 0xFC01      # Negative Signaling NaN (-sNaN)
NEG_SNAN_PAYLOAD = 0xFDAB      # Negative Signaling NaN with custom payload


def h2f(h: int) -> float:
    """Convert a 16-bit half precision integer bit pattern into Python float."""
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def f2h(x: float) -> int:
    """Convert a Python float into a 16-bit half precision integer bit pattern."""
    if math.isnan(x):
        return POS_QNAN_DEFAULT
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def is_nan_f16(h: int) -> bool:
    """Check if 16-bit integer represents an IEEE 754 binary16 NaN."""
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_qnan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a Quiet NaN (MSB of mantissa is 1)."""
    return is_nan_f16(h) and bool(h & 0x0200)


def is_snan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a Signaling NaN (MSB of mantissa is 0, payload != 0)."""
    return is_nan_f16(h) and not bool(h & 0x0200)


def ref_extract_half(val: int, hi: bool = False) -> int:
    """Extract low (bits 15:0) or high (bits 31:16) half-word with 16-bit masking."""
    val &= MASK32
    return ((val >> 16) if hi else val) & MASK16


def ref_apply_mods(val16: int, abs_val: bool = False, neg: bool = False) -> int:
    """Apply RDNA2 VOP3 abs and neg modifiers to a 16-bit half-word: abs first, neg second."""
    m = val16 & MASK16
    if abs_val:
        m &= ABS16
    if neg:
        m ^= SIGN16
    return m


def ref_pack_b32_f16(
    src0: int,
    src1: int,
    op_sel: tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Pure-Python reference oracle for AMD RDNA2 v_pack_b32_f16 float packing."""
    h0 = ref_extract_half(src0, hi=bool(op_sel[0]))
    h1 = ref_extract_half(src1, hi=bool(op_sel[1]))

    h0 = ref_apply_mods(h0, abs_val=abs0, neg=neg0)
    h1 = ref_apply_mods(h1, abs_val=abs1, neg=neg1)

    return (h0 & MASK16) | ((h1 & MASK16) << 16)


def ref_decompose_pack(dword: int) -> tuple[int, int]:
    """Decompose a 32-bit packed word into (lo16, hi16)."""
    dword &= MASK32
    return dword & MASK16, (dword >> 16) & MASK16


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_pack_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_pack_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_shader_lines(lines: list[str]) -> int:
    """Assemble and run synthetic pixel shader lines, returning exported mrt0 v0 word."""
    if _TMP_DIR is None:
        setUpModule()
    assert _SRC_PATH is not None
    asm = (
        "; test_alu_pack_f16\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_pack_instruction(
    instruction_line: str,
    v1: int = 0,
    v2: int = 0,
    old_v0: int = POISON_DST,
) -> int:
    """Seed v1, v2 and destination v0, execute instruction_line, return result in v0."""
    lines = [
        f"v_mov_b32 v0, {old_v0 & MASK32:#x}",
        f"v_mov_b32 v1, {v1 & MASK32:#x}",
        f"v_mov_b32 v2, {v2 & MASK32:#x}",
        instruction_line,
    ]
    return run_shader_lines(lines)


def run_spirv_pack_logical(
    src0: int,
    src1: int,
    op_sel: tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Execute direct SPIR-V shader sequence modeling v_pack_b32_f16 via logical shifts and bitwise ops."""
    lines = [
        "OpCapability Shader",
        '%glsl = OpExtInstImport "GLSL.std.450"',
        "OpMemoryModel Logical GLSL450",
        'OpEntryPoint Fragment %main "main" %out',
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%_ptr_Output_uint = OpTypePointer Output %uint",
        "%out = OpVariable %_ptr_Output_uint Output",
        f"%c_s0 = OpConstant %uint {src0 & MASK32}",
        f"%c_s1 = OpConstant %uint {src1 & MASK32}",
        "%c16 = OpConstant %uint 16",
        "%c0xFFFF = OpConstant %uint 65535",
        "%c0x7FFF = OpConstant %uint 32767",
        "%c0x8000 = OpConstant %uint 32768",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
    ]

    var_idx = 0

    def get_var(prefix: str) -> str:
        nonlocal var_idx
        var_idx += 1
        return f"%{prefix}_{var_idx}"

    def build_half(c_src: str, op_sel_bit: int, abs_f: bool, neg_f: bool) -> str:
        u_var = c_src
        if op_sel_bit:
            shifted = get_var("shift")
            lines.append(f"{shifted} = OpShiftRightLogical %uint {u_var} %c16")
            u_var = shifted
        masked = get_var("mask")
        lines.append(f"{masked} = OpBitwiseAnd %uint {u_var} %c0xFFFF")
        cur = masked
        if abs_f:
            abs_var = get_var("abs")
            lines.append(f"{abs_var} = OpBitwiseAnd %uint {cur} %c0x7FFF")
            cur = abs_var
        if neg_f:
            neg_var = get_var("neg")
            lines.append(f"{neg_var} = OpBitwiseXor %uint {cur} %c0x8000")
            cur = neg_var
        return cur

    h0 = build_half("%c_s0", op_sel[0], abs0, neg0)
    h1 = build_half("%c_s1", op_sel[1], abs1, neg1)

    h1_hi = get_var("h1_hi")
    res = get_var("res")
    lines.extend([
        f"{h1_hi} = OpShiftLeftLogical %uint {h1} %c16",
        f"{res} = OpBitwiseOr %uint {h0} {h1_hi}",
        f"OpStore %out {res}",
        "OpReturn",
        "OpFunctionEnd",
    ])

    spv = "\n".join(lines) + "\n"
    return si.run_fragment(spv)["%out"][0]


class TestPackF16ReferenceOracle(unittest.TestCase):
    """Verify pure-Python reference oracle functions and IEEE 754 bit-level representation."""

    def test_reference_basic_identities(self):
        self.assertEqual(ref_pack_b32_f16(0, 0), 0)
        self.assertEqual(ref_pack_b32_f16(ONE, TWO), (TWO << 16) | ONE)
        self.assertEqual(ref_pack_b32_f16(HALF, ONE_POINT_FIVE), (ONE_POINT_FIVE << 16) | HALF)
        self.assertEqual(ref_pack_b32_f16(0x1234, 0x5678), 0x56781234)

    def test_reference_truncation_of_unselected_bits(self):
        self.assertEqual(ref_pack_b32_f16(0xDEAD1234, 0xBEEF5678), 0x56781234)
        self.assertEqual(ref_pack_b32_f16(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_pack_b32_f16(0xFFFF0000, 0xFFFF0000), 0x00000000)

    def test_reference_op_sel_modes(self):
        v1 = 0xA1B2C3D4
        v2 = 0xE5F60718

        self.assertEqual(ref_pack_b32_f16(v1, v2, op_sel=(0, 0)), 0x0718C3D4)
        self.assertEqual(ref_pack_b32_f16(v1, v2, op_sel=(1, 0)), 0x0718A1B2)
        self.assertEqual(ref_pack_b32_f16(v1, v2, op_sel=(0, 1)), 0xE5F6C3D4)
        self.assertEqual(ref_pack_b32_f16(v1, v2, op_sel=(1, 1)), 0xE5F6A1B2)

    def test_reference_modifiers_abs_and_neg(self):
        # abs clears bit 15: 0xBC00 (-1.0) -> 0x3C00 (+1.0)
        self.assertEqual(ref_apply_mods(NEG_ONE, abs_val=True), ONE)
        self.assertEqual(ref_apply_mods(ONE, abs_val=True), ONE)

        # neg inverts bit 15: 0x3C00 (+1.0) -> 0xBC00 (-1.0)
        self.assertEqual(ref_apply_mods(ONE, neg=True), NEG_ONE)
        self.assertEqual(ref_apply_mods(NEG_ONE, neg=True), ONE)

        # abs then neg: -|x| always sets sign bit
        self.assertEqual(ref_apply_mods(ONE, abs_val=True, neg=True), NEG_ONE)
        self.assertEqual(ref_apply_mods(NEG_ONE, abs_val=True, neg=True), NEG_ONE)

        # Packing with modifiers
        packed = ref_pack_b32_f16(NEG_ONE, TWO, abs0=True, neg1=True)
        self.assertEqual(packed, (NEG_TWO << 16) | ONE)

    def test_reference_decomposition(self):
        self.assertEqual(ref_decompose_pack(0x40003C00), (ONE, TWO))
        self.assertEqual(ref_decompose_pack(0x56781234), (0x1234, 0x5678))
        self.assertEqual(ref_decompose_pack(0), (0, 0))

    def test_nan_classification_helpers(self):
        self.assertFalse(is_nan_f16(ONE))
        self.assertFalse(is_nan_f16(POS_INF))
        self.assertFalse(is_nan_f16(NEG_INF))

        self.assertTrue(is_nan_f16(POS_QNAN_DEFAULT))
        self.assertTrue(is_qnan_f16(POS_QNAN_DEFAULT))
        self.assertFalse(is_snan_f16(POS_QNAN_DEFAULT))

        self.assertTrue(is_nan_f16(POS_SNAN_DEFAULT))
        self.assertTrue(is_snan_f16(POS_SNAN_DEFAULT))
        self.assertFalse(is_qnan_f16(POS_SNAN_DEFAULT))


class TestPackF16RdnaTranslation(unittest.TestCase):
    """Semantic tests executing translated RDNA2 v_pack_b32_f16 pixel shaders via spirv_mini_interp."""

    def test_basic_register_pack(self):
        line = "v_pack_b32_f16 v0, v1, v2"
        v1 = (TWO << 16) | ONE
        v2 = (FOUR << 16) | HALF
        got = run_pack_instruction(line, v1=v1, v2=v2)
        want = ref_pack_b32_f16(v1, v2, op_sel=(0, 0))
        self.assertEqual(got, want)
        self.assertEqual(got, (HALF << 16) | ONE)

    def test_full_destination_overwrite(self):
        """Unlike 16-bit ALU instructions, v_pack_b32_f16 overwrites all 32 bits of the destination."""
        for poison in (0xDEADBEEF, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF):
            with self.subTest(poison=hex(poison)):
                got = run_pack_instruction("v_pack_b32_f16 v0, v1, v2", v1=ONE, v2=TWO, old_v0=poison)
                self.assertEqual(got, (TWO << 16) | ONE)
                self.assertNotEqual(got & 0xFFFF0000, poison & 0xFFFF0000)
                self.assertNotEqual(got & 0x0000FFFF, poison & 0x0000FFFF)

    def test_op_sel_permutations(self):
        v1 = (ONE_POINT_FIVE << 16) | ONE       # hi: 1.5, lo: 1.0
        v2 = (FOUR << 16) | TWO                 # hi: 4.0, lo: 2.0

        op_sel_cases = [
            ("[0,0,0]", (0, 0), (TWO << 16) | ONE),
            ("[1,0,0]", (1, 0), (TWO << 16) | ONE_POINT_FIVE),
            ("[0,1,0]", (0, 1), (FOUR << 16) | ONE),
            ("[1,1,0]", (1, 1), (FOUR << 16) | ONE_POINT_FIVE),
        ]
        for sel_str, sel_tuple, expected in op_sel_cases:
            with self.subTest(sel=sel_str):
                line = f"v_pack_b32_f16 v0, v1, v2 op_sel:{sel_str}"
                got = run_pack_instruction(line, v1=v1, v2=v2)
                self.assertEqual(got, expected)
                self.assertEqual(got, ref_pack_b32_f16(v1, v2, op_sel=sel_tuple))

    def test_source_modifiers_neg(self):
        v1 = ONE  # +1.0
        v2 = TWO  # +2.0

        # -v1 on src0
        got_neg0 = run_pack_instruction("v_pack_b32_f16 v0, -v1, v2", v1=v1, v2=v2)
        self.assertEqual(got_neg0, (TWO << 16) | NEG_ONE)
        self.assertEqual(got_neg0, ref_pack_b32_f16(v1, v2, neg0=True))

        # -v2 on src1
        got_neg1 = run_pack_instruction("v_pack_b32_f16 v0, v1, -v2", v1=v1, v2=v2)
        self.assertEqual(got_neg1, (NEG_TWO << 16) | ONE)
        self.assertEqual(got_neg1, ref_pack_b32_f16(v1, v2, neg1=True))

        # -v1, -v2 on both
        got_neg_both = run_pack_instruction("v_pack_b32_f16 v0, -v1, -v2", v1=v1, v2=v2)
        self.assertEqual(got_neg_both, (NEG_TWO << 16) | NEG_ONE)
        self.assertEqual(got_neg_both, ref_pack_b32_f16(v1, v2, neg0=True, neg1=True))

    def test_source_modifiers_abs(self):
        v1 = NEG_ONE  # -1.0
        v2 = NEG_TWO  # -2.0

        # |v1| on src0
        got_abs0 = run_pack_instruction("v_pack_b32_f16 v0, |v1|, v2", v1=v1, v2=v2)
        self.assertEqual(got_abs0, (NEG_TWO << 16) | ONE)
        self.assertEqual(got_abs0, ref_pack_b32_f16(v1, v2, abs0=True))

        # |v2| on src1
        got_abs1 = run_pack_instruction("v_pack_b32_f16 v0, v1, |v2|", v1=v1, v2=v2)
        self.assertEqual(got_abs1, (TWO << 16) | NEG_ONE)
        self.assertEqual(got_abs1, ref_pack_b32_f16(v1, v2, abs1=True))

        # |v1|, |v2| on both
        got_abs_both = run_pack_instruction("v_pack_b32_f16 v0, |v1|, |v2|", v1=v1, v2=v2)
        self.assertEqual(got_abs_both, (TWO << 16) | ONE)
        self.assertEqual(got_abs_both, ref_pack_b32_f16(v1, v2, abs0=True, abs1=True))

    def test_source_modifiers_abs_and_neg_combined(self):
        """Abs clears bit 15 first, then neg inverts bit 15, yielding negative value."""
        v1 = NEG_ONE  # -1.0
        v2 = TWO      # +2.0

        got = run_pack_instruction("v_pack_b32_f16 v0, -|v1|, |v2|", v1=v1, v2=v2)
        self.assertEqual(got, (TWO << 16) | NEG_ONE)
        self.assertEqual(got, ref_pack_b32_f16(v1, v2, abs0=True, neg0=True, abs1=True))

    def test_modifiers_combined_with_op_sel(self):
        """Modifiers operate on the half selected by op_sel."""
        v1 = (NEG_ONE << 16) | ONE       # hi: -1.0, lo: 1.0
        v2 = (NEG_TWO << 16) | TWO       # hi: -2.0, lo: 2.0

        # Select high half of v1 with abs -> | -1.0 | = +1.0; low half of v2 with neg -> - (+2.0) = -2.0
        line = "v_pack_b32_f16 v0, |v1|, -v2 op_sel:[1,0,0]"
        got = run_pack_instruction(line, v1=v1, v2=v2)
        want = (NEG_TWO << 16) | ONE
        self.assertEqual(got, want)
        self.assertEqual(got, ref_pack_b32_f16(v1, v2, op_sel=(1, 0), abs0=True, neg1=True))

    def test_self_pack_broadcast_and_swap(self):
        """Packing a register with itself can duplicate halves or swap low and high halves."""
        v = (TWO << 16) | ONE  # hi: 2.0, lo: 1.0

        # Duplicate low half: [0, 0] -> (1.0, 1.0)
        got_lo = run_pack_instruction("v_pack_b32_f16 v0, v1, v1 op_sel:[0,0,0]", v1=v)
        self.assertEqual(got_lo, (ONE << 16) | ONE)

        # Duplicate high half: [1, 1] -> (2.0, 2.0)
        got_hi = run_pack_instruction("v_pack_b32_f16 v0, v1, v1 op_sel:[1,1,0]", v1=v)
        self.assertEqual(got_hi, (TWO << 16) | TWO)

        # Swap low and high halves: [1, 0] -> (hi, lo) -> dst[15:0]=hi(2.0), dst[31:16]=lo(1.0)
        got_swap = run_pack_instruction("v_pack_b32_f16 v0, v1, v1 op_sel:[1,0,0]", v1=v)
        self.assertEqual(got_swap, (ONE << 16) | TWO)

    def test_sgpr_operands(self):
        lines = [
            "s_mov_b32 s0, 0x1111",
            "s_mov_b32 s1, 0x2222",
            "v_pack_b32_f16 v0, s0, s1",
        ]
        self.assertEqual(run_shader_lines(lines), 0x22221111)

        # Mixed VGPR and SGPR
        lines_mixed = [
            "v_mov_b32 v1, 0x3333",
            "s_mov_b32 s0, 0x4444",
            "v_pack_b32_f16 v0, v1, s0",
        ]
        self.assertEqual(run_shader_lines(lines_mixed), 0x44443333)

    def test_literal_constants(self):
        lines = [
            "v_pack_b32_f16 v0, 0x1234, 0x5678",
        ]
        self.assertEqual(run_shader_lines(lines), 0x56781234)

    def test_inline_float_constants(self):
        test_pairs = [
            ("1.0", "2.0", (TWO << 16) | ONE),
            ("0.5", "1.5", (ONE_POINT_FIVE << 16) | HALF),
            ("0.0", "4.0", (FOUR << 16) | POS_ZERO),
            ("-1.0", "1.0", (ONE << 16) | NEG_ONE),
            ("0.5", "-0.5", (NEG_HALF << 16) | HALF),
            ("-2.0", "-1.0", (NEG_ONE << 16) | NEG_TWO),
        ]
        for c0, c1, expected in test_pairs:
            with self.subTest(c0=c0, c1=c1):
                line = f"v_pack_b32_f16 v0, {c0}, {c1}"
                self.assertEqual(run_shader_lines([line]), expected)

    def test_inline_float_constant_high_half_zero(self):
        """Selecting the high half of an inline float constant reads +0.0."""
        line = "v_pack_b32_f16 v0, 1.0, 2.0 op_sel:[1,0,0]"
        # src0 is 1.0 with op_sel 1 -> high half is 0x0000 (+0.0)
        # src1 is 2.0 with op_sel 0 -> low half is 0x4000 (2.0)
        got = run_shader_lines([line])
        self.assertEqual(got, (TWO << 16) | POS_ZERO)

        line_both = "v_pack_b32_f16 v0, 1.0, 2.0 op_sel:[1,1,0]"
        # both high halves are 0x0000
        got_both = run_shader_lines([line_both])
        self.assertEqual(got_both, 0x00000000)

    def test_nan_preservation_bit_exact(self):
        """All NaN variations preserve quiet/signaling bits and payload bits exactly."""
        nan_cases = [
            (POS_QNAN_DEFAULT, "positive default quiet NaN"),
            (NEG_QNAN_DEFAULT, "negative default quiet NaN"),
            (POS_QNAN_PAYLOAD, "positive quiet NaN with payload 1"),
            (POS_QNAN_HIGH, "positive quiet NaN with high payload"),
            (NEG_QNAN_PAYLOAD, "negative quiet NaN with custom payload"),
            (POS_SNAN_DEFAULT, "positive default signaling NaN"),
            (NEG_SNAN_DEFAULT, "negative default signaling NaN"),
            (POS_SNAN_PAYLOAD, "positive signaling NaN with custom payload"),
            (NEG_SNAN_PAYLOAD, "negative signaling NaN with custom payload"),
        ]
        for nan0, name0 in nan_cases:
            for nan1, name1 in nan_cases:
                with self.subTest(lo=name0, hi=name1):
                    expected = (nan1 << 16) | nan0
                    got = run_pack_instruction("v_pack_b32_f16 v0, v1, v2", v1=nan0, v2=nan1)
                    self.assertEqual(got, expected)

                    lo, hi = ref_decompose_pack(got)
                    self.assertEqual(lo, nan0)
                    self.assertEqual(hi, nan1)
                    self.assertEqual(is_qnan_f16(lo), is_qnan_f16(nan0))
                    self.assertEqual(is_snan_f16(lo), is_snan_f16(nan0))
                    self.assertEqual(is_qnan_f16(hi), is_qnan_f16(nan1))
                    self.assertEqual(is_snan_f16(hi), is_snan_f16(nan1))

    def test_subnormal_and_boundary_values(self):
        """Subnormal half floats and boundary floats are preserved without flushing."""
        special_values = [
            POS_ZERO,
            NEG_ZERO,
            MIN_SUBNORMAL,
            MAX_SUBNORMAL,
            MIN_SUBNORMAL_NEG,
            MAX_SUBNORMAL_NEG,
            MIN_NORMAL,
            MAX_FINITE,
            MIN_FINITE_NEG,
            POS_INF,
            NEG_INF,
        ]
        for s0 in special_values:
            for s1 in special_values:
                with self.subTest(s0=hex(s0), s1=hex(s1)):
                    expected = (s1 << 16) | s0
                    got = run_pack_instruction("v_pack_b32_f16 v0, v1, v2", v1=s0, v2=s1)
                    self.assertEqual(got, expected)
                    lo, hi = ref_decompose_pack(got)
                    self.assertEqual(lo, s0)
                    self.assertEqual(hi, s1)


class TestPackF16SpirvInterp(unittest.TestCase):
    """Semantic tests executing direct SPIR-V code in spirv_mini_interp and cross-checking oracles."""

    def test_direct_spirv_logical_all_op_sel_modes(self):
        test_vectors = [
            (0, 0),
            (0x12345678, 0x9ABCDEF0),
            ((TWO << 16) | ONE, (FOUR << 16) | HALF),
            (0xFFFFFFFF, 0x00000000),
            (0x80000000, 0x7FFFFFFF),
            (0x55555555, 0xAAAAAAAA),
            (POS_QNAN_PAYLOAD, POS_SNAN_PAYLOAD),
            (MIN_SUBNORMAL, MAX_FINITE),
        ]
        for s0, s1 in test_vectors:
            for op_sel in itertools.product((0, 1), repeat=2):
                with self.subTest(s0=hex(s0), s1=hex(s1), op_sel=op_sel):
                    expected = ref_pack_b32_f16(s0, s1, op_sel=op_sel)
                    got = run_spirv_pack_logical(s0, s1, op_sel=op_sel)
                    self.assertEqual(got, expected)

    def test_direct_spirv_logical_with_modifiers(self):
        s0 = (NEG_ONE << 16) | ONE
        s1 = (NEG_TWO << 16) | TWO

        for abs0, neg0, abs1, neg1 in itertools.product((True, False), repeat=4):
            with self.subTest(abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1):
                expected = ref_pack_b32_f16(s0, s1, abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1)
                got = run_spirv_pack_logical(s0, s1, abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1)
                self.assertEqual(got, expected)

    def test_spirv_glsl_pack_half2x16_builtins(self):
        """Cross-check with GLSL.std.450 PackHalf2x16 and UnpackHalf2x16 builtins."""
        header = """OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%f1 = OpConstant %float 1.0
%f2 = OpConstant %float 2.0
%f70000 = OpConstant %float 70000.0
%ch = OpConstant %uint 1073757184
%main = OpFunction %void None %fn
%entry = OpLabel
"""
        footer = "OpReturn\nOpFunctionEnd\n"

        def run_body(body: str) -> int:
            return si.run_fragment(header + body + "OpStore %out %res\n" + footer)["%out"][0]

        # Pack 1.0 and 2.0 -> 0x40003C00
        pack_res = run_body(
            "%v = OpCompositeConstruct %v2float %f1 %f2\n"
            "%res = OpExtInst %uint %glsl PackHalf2x16 %v\n"
        )
        self.assertEqual(pack_res, (TWO << 16) | ONE)

        # Overflow: 70000.0 overflows half float to +inf (0x7C00)
        inf_res = run_body(
            "%v_inf = OpCompositeConstruct %v2float %f70000 %f1\n"
            "%res = OpExtInst %uint %glsl PackHalf2x16 %v_inf\n"
        )
        self.assertEqual(inf_res, (ONE << 16) | POS_INF)

        # Unpack roundtrip: %ch is 0x40003C00 -> low half is 1.0 -> 0x3F800000 in binary32
        unpack_res = run_body(
            "%u = OpExtInst %v2float %glsl UnpackHalf2x16 %ch\n"
            "%x = OpCompositeExtract %float %u 0\n"
            "%res = OpBitcast %uint %x\n"
        )
        self.assertEqual(unpack_res, 0x3F800000)

    def test_deterministic_fuzzing_matches_oracle_and_rdna(self):
        """Cross-check 60 random inputs across reference oracle, direct SPIR-V, and RDNA shader translation."""
        rng = random.Random(0xF16F16)
        special_halves = [
            POS_ZERO, NEG_ZERO, MIN_SUBNORMAL, MAX_SUBNORMAL,
            MIN_NORMAL, ONE, TWO, HALF, MAX_FINITE,
            POS_INF, NEG_INF, POS_QNAN_DEFAULT, POS_SNAN_PAYLOAD,
        ]

        for idx in range(60):
            # Seed 32-bit registers with either random bit words or special f16 combinations
            if rng.random() < 0.4:
                s0 = (rng.choice(special_halves) << 16) | rng.choice(special_halves)
                s1 = (rng.choice(special_halves) << 16) | rng.choice(special_halves)
            else:
                s0 = rng.getrandbits(32)
                s1 = rng.getrandbits(32)

            op_sel = (rng.choice((0, 1)), rng.choice((0, 1)))
            abs0 = rng.choice((True, False))
            neg0 = rng.choice((True, False))
            abs1 = rng.choice((True, False))
            neg1 = rng.choice((True, False))

            with self.subTest(iteration=idx, s0=hex(s0), s1=hex(s1)):
                expected = ref_pack_b32_f16(
                    s0, s1, op_sel=op_sel,
                    abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1,
                )

                # Direct SPIR-V execution
                got_spv = run_spirv_pack_logical(
                    s0, s1, op_sel=op_sel,
                    abs0=abs0, neg0=neg0, abs1=abs1, neg1=neg1,
                )
                self.assertEqual(got_spv, expected)

                # Translated RDNA shader execution
                src0_str = "v1"
                if abs0 and neg0:
                    src0_str = "-|v1|"
                elif abs0:
                    src0_str = "|v1|"
                elif neg0:
                    src0_str = "-v1"

                src1_str = "v2"
                if abs1 and neg1:
                    src1_str = "-|v2|"
                elif abs1:
                    src1_str = "|v2|"
                elif neg1:
                    src1_str = "-v2"

                line = f"v_pack_b32_f16 v0, {src0_str}, {src1_str} op_sel:[{op_sel[0]},{op_sel[1]},0]"
                got_rdna = run_pack_instruction(line, v1=s0, v2=s1)
                self.assertEqual(got_rdna, expected)


class TestPackF16NegativeControls(unittest.TestCase):
    """Negative controls ensuring tests fail when intentional bugs or regressions are injected."""

    def test_operand_order_reversal_detected(self):
        s0, s1 = ONE, TWO
        correct = ref_pack_b32_f16(s0, s1)
        reversed_order = ref_pack_b32_f16(s1, s0)
        self.assertNotEqual(correct, reversed_order)

    def test_missing_high_half_shift_detected(self):
        s0, s1 = ONE, TWO
        correct = ref_pack_b32_f16(s0, s1)
        missing_shift = (s0 & MASK16) | (s1 & MASK16)
        self.assertNotEqual(correct, missing_shift)

    def test_off_by_one_shift_detected(self):
        s0, s1 = ONE, TWO
        correct = ref_pack_b32_f16(s0, s1)
        shift_15 = (s0 & MASK16) | ((s1 & MASK16) << 15)
        self.assertNotEqual(correct, shift_15)

    def test_missing_mask_truncation_detected(self):
        s0, s1 = 0xDEAD1234, 0xBEEF5678
        correct = ref_pack_b32_f16(s0, s1)
        unmasked = (s0 | (s1 << 16)) & MASK32
        self.assertNotEqual(correct, unmasked)

    def test_inverted_op_sel_detected(self):
        s0 = (ONE_POINT_FIVE << 16) | ONE
        s1 = (FOUR << 16) | TWO
        correct = ref_pack_b32_f16(s0, s1, op_sel=(0, 0))
        inverted = ref_pack_b32_f16(s0, s1, op_sel=(1, 1))
        self.assertNotEqual(correct, inverted)

    def test_wrong_sign_bit_in_modifier_detected(self):
        s = ONE
        correct_neg = ref_apply_mods(s, neg=True)
        # Inverting bit 14 instead of bit 15
        wrong_bit = s ^ (1 << 14)
        self.assertNotEqual(correct_neg, wrong_bit)

    def test_modifier_evaluation_order_detected(self):
        # In RDNA2 ISA, abs clears bit 15 first, then neg inverts bit 15: -|x| always sets sign bit.
        # If order were inverted (neg first, then abs), | -x | would clear sign bit to positive!
        val = ONE
        correct_abs_then_neg = ref_apply_mods(val, abs_val=True, neg=True)
        buggy_neg_then_abs = ((val & MASK16) ^ SIGN16) & ABS16
        self.assertEqual(correct_abs_then_neg, NEG_ONE)
        self.assertEqual(buggy_neg_then_abs, ONE)
        self.assertNotEqual(correct_abs_then_neg, buggy_neg_then_abs)


if __name__ == "__main__":
    unittest.main()
