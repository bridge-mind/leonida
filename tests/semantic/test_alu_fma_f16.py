from __future__ import annotations

"""Semantic interpreter tests for packed half-precision fused multiply-add (v_fma_mix_f32, v_pk_fma_f16).

Covers RDNA2 instruction translation and mini-interpreter execution for:
- v_pk_fma_f16 (VOP3P): Packed 16-bit floating point FMA computing:
    D.lo = S0.lo * S1.lo + S2.lo
    D.hi = S0.hi * S1.hi + S2.hi
  Supports source half selection via op_sel (for lo half) and op_sel_hi (for hi half),
  negation modifiers neg_lo and neg_hi, and subnormal arithmetic.
- v_fma_mix_f32 (VOP3P): Mixed-precision FMA producing a single-precision binary32 float:
    D.f32 = S0 * S1 + S2
  where op_sel_hi[i] selects whether operand i is f16 (1) or f32 (0), and op_sel[i] selects
  low (0) or high (1) 16-bit half when operand i is f16. Supports operand modifiers (neg, abs),
  clamp, and subnormal handling.
- Subnormal handling: IEEE 754 binary16 subnormals (exponent 0, fraction != 0, value (-1)^s * fraction * 2^-24).
  Smallest positive subnormal: 0x0001 (2^-24 ~= 5.96e-8).
  Largest positive subnormal: 0x03FF ((1023/1024)*2^-14 ~= 6.098e-5).
  Tests gradual underflow, subnormal inputs, subnormal scaling, underflow to zero, and
  boundary transitions between subnormal and normal representations.
- Direct SPIR-V interpreter validation for GLSL.std.450 Fma, PackHalf2x16, and UnpackHalf2x16.
- Cross-checked against independent plain-Python reference implementations and negative controls.
"""

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

sys.dont_write_bytecode = True

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Notable IEEE 754 half-precision bit patterns
F16_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_MIN_SUBNORMAL = 0x0001       # 2^-24
F16_MAX_SUBNORMAL = 0x03FF       # (1023/1024) * 2^-14
F16_NEG_MIN_SUBNORMAL = 0x8001   # -2^-24
F16_NEG_MAX_SUBNORMAL = 0x83FF   # -(1023/1024) * 2^-14
F16_MIN_NORMAL = 0x0400          # 2^-14
F16_ONE = 0x3C00                 # 1.0
F16_TWO = 0x4000                 # 2.0
F16_NEG_ONE = 0xBC00             # -1.0


def h2f(h: int) -> float:
    """Convert a 16-bit unsigned integer bit pattern to a Python float via IEEE 754 half precision."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    """Convert a Python float to a 16-bit IEEE 754 half-precision bit pattern with round-to-nearest-even."""
    try:
        return struct.unpack("<H", struct.pack("<e", float(x)))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def f32_to_u32(x: float) -> int:
    """Convert a float to 32-bit IEEE 754 single-precision integer bits."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def u32_to_f32(bits: int) -> float:
    """Convert 32-bit integer bits to float via IEEE 754 single precision."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def pack2x16(lo: int, hi: int) -> int:
    """Pack two 16-bit half words into one 32-bit dword (hi in [31:16], lo in [15:0])."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def unpack2x16(dword: int) -> tuple[int, int]:
    """Unpack a 32-bit dword into (lo, hi) 16-bit half words."""
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def ref_pk_fma_f16(
    v0: int,
    v1: int,
    v2: int,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (1, 1, 1),
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Reference calculation for RDNA2 v_pk_fma_f16.

    Result lo: source i takes its half from op_sel[i] (0=lo, 1=hi), negated by neg_lo[i].
    Result hi: source i takes its half from op_sel_hi[i] (0=lo, 1=hi), negated by neg_hi[i].
    """
    def sel_half(word: int, hi: int, neg: int) -> float:
        h = (word >> 16) if hi else (word & 0xFFFF)
        val = h2f(h)
        return -val if neg else val

    s0_lo = sel_half(v0, op_sel[0], neg_lo[0])
    s1_lo = sel_half(v1, op_sel[1], neg_lo[1])
    s2_lo = sel_half(v2, op_sel[2], neg_lo[2])
    res_lo_f = s0_lo * s1_lo + s2_lo
    if clamp:
        res_lo_f = max(0.0, min(1.0, res_lo_f))
    res_lo = f2h(res_lo_f)

    s0_hi = sel_half(v0, op_sel_hi[0], neg_hi[0])
    s1_hi = sel_half(v1, op_sel_hi[1], neg_hi[1])
    s2_hi = sel_half(v2, op_sel_hi[2], neg_hi[2])
    res_hi_f = s0_hi * s1_hi + s2_hi
    if clamp:
        res_hi_f = max(0.0, min(1.0, res_hi_f))
    res_hi = f2h(res_hi_f)

    return pack2x16(res_lo, res_hi)


def ref_fma_mix_f32(
    v0: int,
    v1: int,
    v2: int,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (0, 0, 0),
    neg: tuple[int, int, int] = (0, 0, 0),
    abs_mask: tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Reference calculation for RDNA2 v_fma_mix_f32.

    Each source is f32 (op_sel_hi[i]==0) or f16 (op_sel_hi[i]==1) unpacked from
    low (op_sel[i]==0) or high (op_sel[i]==1) half.
    """
    def get_src(word: int, is_f16: int, sel: int, is_neg: int, is_abs: int) -> float:
        if is_f16:
            h = (word >> 16) if sel else (word & 0xFFFF)
            val = h2f(h)
        else:
            val = u32_to_f32(word)
        if is_abs:
            val = abs(val)
        if is_neg:
            val = -val
        return val

    s0 = get_src(v0, op_sel_hi[0], op_sel[0], neg[0], abs_mask[0])
    s1 = get_src(v1, op_sel_hi[1], op_sel[1], neg[1], abs_mask[1])
    s2 = get_src(v2, op_sel_hi[2], op_sel[2], neg[2], abs_mask[2])

    res = s0 * s1 + s2
    if clamp:
        res = max(0.0, min(1.0, res))
    return f32_to_u32(res)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_fma_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_fma_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_assembly(lines: list[str]) -> int:
    """Translate straight-line assembly instructions to SPIR-V and return the first mrt0 word."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    asm = "; test_alu_fma_f16\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    return out["%out_mrt0"][0]


def run_pk_fma(
    v0: int,
    v1: int,
    v2: int,
    op_sel: tuple[int, int, int] | None = None,
    op_sel_hi: tuple[int, int, int] | None = None,
    neg_lo: tuple[int, int, int] | None = None,
    neg_hi: tuple[int, int, int] | None = None,
) -> int:
    """Seed v1..v3 with values, execute v_pk_fma_f16 into v0, return v0."""
    mods: list[str] = []
    if op_sel is not None:
        mods.append(f"op_sel:[{op_sel[0]},{op_sel[1]},{op_sel[2]}]")
    if op_sel_hi is not None:
        mods.append(f"op_sel_hi:[{op_sel_hi[0]},{op_sel_hi[1]},{op_sel_hi[2]}]")
    if neg_lo is not None:
        mods.append(f"neg_lo:[{neg_lo[0]},{neg_lo[1]},{neg_lo[2]}]")
    if neg_hi is not None:
        mods.append(f"neg_hi:[{neg_hi[0]},{neg_hi[1]},{neg_hi[2]}]")

    mod_str = (" " + " ".join(mods)) if mods else ""
    lines = [
        f"v_mov_b32 v1, {v0:#010x}",
        f"v_mov_b32 v2, {v1:#010x}",
        f"v_mov_b32 v3, {v2:#010x}",
        f"v_pk_fma_f16 v0, v1, v2, v3{mod_str}",
    ]
    return run_assembly(lines)


def run_fma_mix(
    v0: int,
    v1: int,
    v2: int,
    op_sel: tuple[int, int, int] | None = None,
    op_sel_hi: tuple[int, int, int] | None = None,
    neg: tuple[int, int, int] | None = None,
    abs_mask: tuple[int, int, int] | None = None,
    clamp: bool = False,
) -> int:
    """Seed v1..v3 with values, execute v_fma_mix_f32 into v0, return v0."""
    def format_operand(reg_name: str, idx: int) -> str:
        s = reg_name
        if abs_mask and abs_mask[idx]:
            s = f"|{s}|"
        if neg and neg[idx]:
            s = f"-{s}"
        return s

    mods: list[str] = []
    if op_sel is not None:
        mods.append(f"op_sel:[{op_sel[0]},{op_sel[1]},{op_sel[2]}]")
    if op_sel_hi is not None:
        mods.append(f"op_sel_hi:[{op_sel_hi[0]},{op_sel_hi[1]},{op_sel_hi[2]}]")
    if clamp:
        mods.append("clamp")

    mod_str = (" " + " ".join(mods)) if mods else ""
    s0_str = format_operand("v1", 0)
    s1_str = format_operand("v2", 1)
    s2_str = format_operand("v3", 2)

    lines = [
        f"v_mov_b32 v1, {v0:#010x}",
        f"v_mov_b32 v2, {v1:#010x}",
        f"v_mov_b32 v3, {v2:#010x}",
        f"v_fma_mix_f32 v0, {s0_str}, {s1_str}, {s2_str}{mod_str}",
    ]
    return run_assembly(lines)


class OracleConsistencyTests(unittest.TestCase):
    """Verify that reference calculation functions agree with independently hand-computed values."""

    def test_pk_fma_reference_against_hand_values(self):
        # v0: lo=1.0 (0x3c00), hi=2.0 (0x4000)
        # v1: lo=3.0 (0x4200), hi=4.0 (0x4400)
        # v2: lo=0.5 (0x3800), hi=1.0 (0x3c00)
        # lo: 1.0 * 3.0 + 0.5 = 3.5 -> 0x4300
        # hi: 2.0 * 4.0 + 1.0 = 9.0 -> 0x4880
        # result: 0x48804300
        v0 = pack2x16(f2h(1.0), f2h(2.0))
        v1 = pack2x16(f2h(3.0), f2h(4.0))
        v2 = pack2x16(f2h(0.5), f2h(1.0))
        got = ref_pk_fma_f16(v0, v1, v2)
        self.assertEqual(got, 0x48804300)

    def test_fma_mix_reference_against_hand_values(self):
        # v0: lo=1.5 (0x3e00), hi=2.5 (0x4100)
        # v1: lo=2.0 (0x4000), hi=0.5 (0x3800)
        # v2: single-precision 10.0f (0x41200000)
        # op_sel_hi=[1, 1, 0] (f16, f16, f32), op_sel=[0, 0, 0] (lo, lo, lo)
        # 1.5 * 2.0 + 10.0 = 13.0f -> 0x41500000
        v0 = pack2x16(f2h(1.5), f2h(2.5))
        v1 = pack2x16(f2h(2.0), f2h(0.5))
        v2 = f32_to_u32(10.0)
        got = ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 0))
        self.assertEqual(got, 0x41500000)

    def test_subnormal_constants_and_properties(self):
        # 0x0001: 2^-24
        val_min = h2f(F16_MIN_SUBNORMAL)
        self.assertAlmostEqual(val_min, 2.0**-24, delta=1e-15)
        self.assertEqual(f2h(val_min), F16_MIN_SUBNORMAL)

        # 0x03FF: (1023/1024) * 2^-14
        val_max = h2f(F16_MAX_SUBNORMAL)
        self.assertAlmostEqual(val_max, (1023.0 / 1024.0) * (2.0**-14), delta=1e-15)
        self.assertEqual(f2h(val_max), F16_MAX_SUBNORMAL)

        # 0x0400: smallest normal 2^-14
        val_norm = h2f(F16_MIN_NORMAL)
        self.assertAlmostEqual(val_norm, 2.0**-14, delta=1e-15)
        self.assertEqual(f2h(val_norm), F16_MIN_NORMAL)


class SpirvInterpFmaDirectTests(unittest.TestCase):
    """Direct tests for the mini-interpreter's GLSL.std.450 Fma, PackHalf2x16, and UnpackHalf2x16."""

    HEADER = """\
OpCapability Shader
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
%f0 = OpConstant %float 0
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%f3 = OpConstant %float 3
%f4 = OpConstant %float 4
%f5 = OpConstant %float 5
%fn2 = OpConstant %float -2
%fn3 = OpConstant %float -3
%main = OpFunction %void None %fn
%entry = OpLabel
"""
    FOOTER = "OpReturn\nOpFunctionEnd\n"

    def run_word(self, body: str) -> int:
        text = self.HEADER + body + "OpStore %out %res\n" + self.FOOTER
        return si.run_fragment(text)["%out"][0]

    def test_vector_fma_arithmetic(self):
        # v0 = (2.0, 3.0), v1 = (4.0, 5.0), v2 = (1.0, 2.0)
        # res = v0 * v1 + v2 = (2*4+1, 3*5+2) = (9.0, 17.0)
        # PackHalf2x16: 9.0 (0x4880), 17.0 (0x4c40) -> 0x4C404880
        body = """\
%v0 = OpCompositeConstruct %v2float %f2 %f3
%v1 = OpCompositeConstruct %v2float %f4 %f5
%v2 = OpCompositeConstruct %v2float %f1 %f2
%r = OpExtInst %v2float %glsl Fma %v0 %v1 %v2
%res = OpExtInst %uint %glsl PackHalf2x16 %r
"""
        self.assertEqual(self.run_word(body), 0x4C404880)

    def test_vector_fma_negative_and_zero(self):
        # v0 = (-2.0, 0.0), v1 = (3.0, 5.0), v2 = (1.0, 4.0)
        # res = (-2*3+1, 0*5+4) = (-5.0, 4.0)
        # f16(-5.0) = 0xc500, f16(4.0) = 0x4400 -> 0x4400c500
        body = """\
%v0 = OpCompositeConstruct %v2float %fn2 %f0
%v1 = OpCompositeConstruct %v2float %f3 %f5
%v2 = OpCompositeConstruct %v2float %f1 %f4
%r = OpExtInst %v2float %glsl Fma %v0 %v1 %v2
%res = OpExtInst %uint %glsl PackHalf2x16 %r
"""
        self.assertEqual(self.run_word(body), 0x4400C500)

    def test_scalar_fma_f32_precision(self):
        # 2.0f * -3.0f + 1.0f = -5.0f (0xC0A00000)
        body = """\
%r = OpExtInst %float %glsl Fma %f2 %fn3 %f1
%res = OpBitcast %uint %r
"""
        self.assertEqual(self.run_word(body), 0xC0A00000)


class PackedFmaF16Tests(unittest.TestCase):
    """Tests for RDNA2 v_pk_fma_f16 instruction translation and interpretation."""

    def test_pk_fma_basic_lanes(self):
        # lo: 1.0 * 2.0 + 3.0 = 5.0 (0x4500)
        # hi: 0.5 * 4.0 + 1.5 = 3.5 (0x4300)
        v0 = pack2x16(f2h(1.0), f2h(0.5))
        v1 = pack2x16(f2h(2.0), f2h(4.0))
        v2 = pack2x16(f2h(3.0), f2h(1.5))
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)

    def test_pk_fma_zeroes_and_identity(self):
        # lo: 0.0 * 10.0 + 7.5 = 7.5
        # hi: 3.5 * 1.0 + 0.0 = 3.5
        v0 = pack2x16(f2h(0.0), f2h(3.5))
        v1 = pack2x16(f2h(10.0), f2h(1.0))
        v2 = pack2x16(f2h(7.5), f2h(0.0))
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)

    def test_pk_fma_negative_numbers(self):
        # lo: -1.5 * 2.0 + 4.0 = 1.0
        # hi: -2.0 * -3.0 + -1.0 = 5.0
        v0 = pack2x16(f2h(-1.5), f2h(-2.0))
        v1 = pack2x16(f2h(2.0), f2h(-3.0))
        v2 = pack2x16(f2h(4.0), f2h(-1.0))
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)

    def test_pk_fma_fractional_precision(self):
        # lo: 0.25 * 0.5 + 0.125 = 0.25
        # hi: 1.5 * 1.5 + 0.25 = 2.5
        v0 = pack2x16(f2h(0.25), f2h(1.5))
        v1 = pack2x16(f2h(0.5), f2h(1.5))
        v2 = pack2x16(f2h(0.125), f2h(0.25))
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)

    def test_pk_fma_op_sel_permutations(self):
        v0 = pack2x16(f2h(1.0), f2h(10.0))
        v1 = pack2x16(f2h(2.0), f2h(20.0))
        v2 = pack2x16(f2h(3.0), f2h(30.0))

        # 10.0 * 2.0 + 3.0 = 23.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 1, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 1, 1)))
        # 1.0 * 20.0 + 3.0 = 23.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 1, 0), op_sel_hi=(1, 1, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 1, 0), op_sel_hi=(1, 1, 1)))
        # 1.0 * 2.0 + 30.0 = 32.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 0, 1), op_sel_hi=(1, 1, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 1), op_sel_hi=(1, 1, 1)))
        # 10.0 * 20.0 + 30.0 = 230.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(1, 1, 1), op_sel_hi=(1, 1, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(1, 1, 1), op_sel_hi=(1, 1, 1)))

    def test_pk_fma_op_sel_hi_permutations(self):
        v0 = pack2x16(f2h(2.0), f2h(5.0))
        v1 = pack2x16(f2h(3.0), f2h(4.0))
        v2 = pack2x16(f2h(1.0), f2h(6.0))

        # 2.0 * 4.0 + 6.0 = 14.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 1)))
        # 5.0 * 3.0 + 6.0 = 21.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 1)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 1)))
        # 5.0 * 4.0 + 1.0 = 21.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 0)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 0)))
        # 2.0 * 3.0 + 1.0 = 7.0
        self.assertEqual(run_pk_fma(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0)),
                         ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0)))

    def test_pk_fma_neg_lo_and_neg_hi(self):
        # v0: lo=1.0, hi=2.0
        # v1: lo=2.0, hi=3.0
        # v2: lo=3.0, hi=4.0
        # neg_lo:[1, 0, 0]: lo = -(1.0) * 2.0 + 3.0 = 1.0
        # neg_hi:[0, 1, 0]: hi = 2.0 * -(3.0) + 4.0 = -2.0
        v0 = pack2x16(f2h(1.0), f2h(2.0))
        v1 = pack2x16(f2h(2.0), f2h(3.0))
        v2 = pack2x16(f2h(3.0), f2h(4.0))
        expected = ref_pk_fma_f16(v0, v1, v2, neg_lo=(1, 0, 0), neg_hi=(0, 1, 0))
        got = run_pk_fma(v0, v1, v2, neg_lo=(1, 0, 0), neg_hi=(0, 1, 0))
        self.assertEqual(got, expected)

    def test_pk_fma_cross_lane_independent(self):
        v0 = pack2x16(f2h(1.5), f2h(3.0))
        v1 = pack2x16(f2h(2.0), f2h(4.0))
        v2 = pack2x16(f2h(0.5), f2h(2.5))
        expected = ref_pk_fma_f16(v0, v1, v2, op_sel=(1, 0, 1), op_sel_hi=(0, 1, 0))
        got = run_pk_fma(v0, v1, v2, op_sel=(1, 0, 1), op_sel_hi=(0, 1, 0))
        self.assertEqual(got, expected)


class PackedFmaF16SubnormalTests(unittest.TestCase):
    """Audits packed 16-bit float FMA arithmetic with IEEE 754 half-precision subnormals."""

    def test_subnormal_identity_and_scaling(self):
        # lo: 0x0001 (2^-24) * 1.0 + 0.0 = 0x0001
        # hi: 0x0001 (2^-24) * 2.0 + 0.0 = 0x0002
        v0 = pack2x16(F16_MIN_SUBNORMAL, F16_MIN_SUBNORMAL)
        v1 = pack2x16(F16_ONE, F16_TWO)
        v2 = pack2x16(F16_ZERO, F16_ZERO)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        lo, hi = unpack2x16(got)
        self.assertEqual(lo, F16_MIN_SUBNORMAL)
        self.assertEqual(hi, 0x0002)

    def test_subnormal_addition_and_accumulation(self):
        # lo: 0x0001 * 1.0 + 0x0001 = 0x0002
        # hi: 0x0002 * 1.0 + 0x0003 = 0x0005
        v0 = pack2x16(0x0001, 0x0002)
        v1 = pack2x16(F16_ONE, F16_ONE)
        v2 = pack2x16(0x0001, 0x0003)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        self.assertEqual(got, pack2x16(0x0002, 0x0005))

    def test_subnormal_largest_and_negative(self):
        # lo: 0x03FF (largest subnormal) * 1.0 + 0.0 = 0x03FF
        # hi: 0x8001 (-2^-24) * 1.0 + 0.0 = 0x8001
        v0 = pack2x16(F16_MAX_SUBNORMAL, F16_NEG_MIN_SUBNORMAL)
        v1 = pack2x16(F16_ONE, F16_ONE)
        v2 = pack2x16(F16_ZERO, F16_ZERO)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)

    def test_boundary_crossing_subnormal_to_normal(self):
        # Adding 0x0001 (2^-24) to 0x03FF (largest subnormal) crosses the threshold into smallest normal 0x0400 (2^-14).
        # lo: 0x03FF * 1.0 + 0x0001 = 0x0400
        # hi: 0x03FF * 1.0 + 0x0002 = 0x0401
        v0 = pack2x16(F16_MAX_SUBNORMAL, F16_MAX_SUBNORMAL)
        v1 = pack2x16(F16_ONE, F16_ONE)
        v2 = pack2x16(0x0001, 0x0002)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        lo, hi = unpack2x16(got)
        self.assertEqual(lo, F16_MIN_NORMAL)
        self.assertEqual(hi, 0x0401)

    def test_boundary_crossing_normal_to_subnormal(self):
        # Subtracting 0x0001 from 0x0400 (smallest normal) crosses down into subnormal space 0x03FF.
        # lo: 0x0400 * 1.0 - 0x0001 = 0x03FF
        # hi: 0x0400 * 1.0 - 0x0002 = 0x03FE
        v0 = pack2x16(F16_MIN_NORMAL, F16_MIN_NORMAL)
        v1 = pack2x16(F16_ONE, F16_ONE)
        v2 = pack2x16(0x0001, 0x0002)
        expected = ref_pk_fma_f16(v0, v1, v2, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))
        got = run_pk_fma(v0, v1, v2, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))
        self.assertEqual(got, expected)
        lo, hi = unpack2x16(got)
        self.assertEqual(lo, F16_MAX_SUBNORMAL)
        self.assertEqual(hi, 0x03FE)

    def test_gradual_underflow_from_normal_multiplication(self):
        # 2^-8 (0x1C00) * 2^-8 (0x1C00) = 2^-16.
        # In f16, 2^-16 is subnormal: (2^-16 / 2^-24) = 256 = 0x0100.
        # 2^-10 (0x1400) * 2^-10 (0x1400) = 2^-20 -> 16 * 2^-24 = 0x0010.
        v0 = pack2x16(0x1C00, 0x1400)
        v1 = pack2x16(0x1C00, 0x1400)
        v2 = pack2x16(F16_ZERO, F16_ZERO)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        lo, hi = unpack2x16(got)
        self.assertEqual(lo, 0x0100)
        self.assertEqual(hi, 0x0010)

    def test_underflow_below_subnormal_to_zero(self):
        # 2^-13 (0x0800) * 2^-13 (0x0800) = 2^-26.
        # Smallest subnormal is 2^-24; half least significant bit is 2^-25.
        # 2^-26 < 2^-25, so IEEE 754 round-to-nearest-even underflows to zero.
        v0 = pack2x16(0x0800, 0x0800)
        v1 = pack2x16(0x0800, 0x0800)
        v2 = pack2x16(F16_ZERO, F16_ZERO)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        self.assertEqual(got, 0x00000000)

    def test_normals_multiplying_to_min_subnormal(self):
        # 2^-12 (0x0C00) * 2^-12 (0x0C00) = 2^-24 (0x0001), exact minimum subnormal.
        v0 = pack2x16(0x0C00, 0x0C00)
        v1 = pack2x16(0x0C00, 0x0C00)
        v2 = pack2x16(F16_ZERO, F16_ZERO)
        expected = ref_pk_fma_f16(v0, v1, v2)
        got = run_pk_fma(v0, v1, v2)
        self.assertEqual(got, expected)
        self.assertEqual(got, pack2x16(F16_MIN_SUBNORMAL, F16_MIN_SUBNORMAL))

    def test_subnormal_cancellation_to_zero(self):
        # 0x0005 * 1.0 - 0x0005 = 0.0
        v0 = pack2x16(0x0005, 0x0005)
        v1 = pack2x16(F16_ONE, F16_ONE)
        v2 = pack2x16(0x0005, 0x0005)
        expected = ref_pk_fma_f16(v0, v1, v2, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))
        got = run_pk_fma(v0, v1, v2, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))
        self.assertEqual(got, expected)
        lo, hi = unpack2x16(got)
        self.assertEqual(lo & 0x7FFF, 0)
        self.assertEqual(hi & 0x7FFF, 0)


class FmaMixF32Tests(unittest.TestCase):
    """Tests for RDNA2 v_fma_mix_f32 instruction translation and interpretation."""

    def test_fma_mix_all_three_f16(self):
        # op_sel_hi=[1, 1, 1]: all three sources are f16
        v0 = pack2x16(f2h(1.5), f2h(2.5))
        v1 = pack2x16(f2h(2.0), f2h(4.0))
        v2 = pack2x16(f2h(0.25), f2h(1.0))

        # All low halves: 1.5 * 2.0 + 0.25 = 3.25f
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1)))
        # All high halves: 2.5 * 4.0 + 1.0 = 11.0f
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(1, 1, 1), op_sel_hi=(1, 1, 1)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(1, 1, 1), op_sel_hi=(1, 1, 1)))
        # Mixed halves: s0 hi (2.5), s1 lo (2.0), s2 hi (1.0) -> 2.5 * 2.0 + 1.0 = 6.0f
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(1, 0, 1), op_sel_hi=(1, 1, 1)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(1, 0, 1), op_sel_hi=(1, 1, 1)))

    def test_fma_mix_two_f16_one_f32(self):
        # [1, 1, 0]: s0(f16), s1(f16), s2(f32) -> 2.5 * 4.0 + 10.5 = 20.5f
        v0 = pack2x16(f2h(2.5), f2h(5.0))
        v1 = pack2x16(f2h(4.0), f2h(1.0))
        v2 = f32_to_u32(10.5)
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 0)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 0)))

        # [1, 0, 1]: s0(f16), s1(f32), s2(f16) -> 3.0 * 5.0 + (-2.0) = 13.0f
        v0_b = pack2x16(f2h(3.0), f2h(6.0))
        v1_b = f32_to_u32(5.0)
        v2_b = pack2x16(f2h(-2.0), f2h(8.0))
        self.assertEqual(run_fma_mix(v0_b, v1_b, v2_b, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 1)),
                         ref_fma_mix_f32(v0_b, v1_b, v2_b, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 1)))

        # [0, 1, 1]: s0(f32), s1(f16), s2(f16) -> -2.5 * 6.0 + 1.0 = -14.0f
        v0_c = f32_to_u32(-2.5)
        v1_c = pack2x16(f2h(6.0), f2h(1.0))
        v2_c = pack2x16(f2h(1.0), f2h(2.0))
        self.assertEqual(run_fma_mix(v0_c, v1_c, v2_c, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 1)),
                         ref_fma_mix_f32(v0_c, v1_c, v2_c, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 1)))

    def test_fma_mix_one_f16_two_f32(self):
        # [1, 0, 0]: s0(f16), s1(f32), s2(f32) -> 0.5 * 8.0 + 3.0 = 7.0f
        v0 = pack2x16(f2h(0.5), f2h(9.0))
        v1 = f32_to_u32(8.0)
        v2 = f32_to_u32(3.0)
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0)))

        # [0, 1, 0]: s0(f32), s1(f16), s2(f32) -> 7.0 * 2.0 + (-5.0) = 9.0f
        v0_b = f32_to_u32(7.0)
        v1_b = pack2x16(f2h(2.0), f2h(10.0))
        v2_b = f32_to_u32(-5.0)
        self.assertEqual(run_fma_mix(v0_b, v1_b, v2_b, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 0)),
                         ref_fma_mix_f32(v0_b, v1_b, v2_b, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 0)))

        # [0, 0, 1]: s0(f32), s1(f32), s2(f16) -> 3.0 * 4.0 + 0.5 = 12.5f
        v0_c = f32_to_u32(3.0)
        v1_c = f32_to_u32(4.0)
        v2_c = pack2x16(f2h(0.5), f2h(10.0))
        self.assertEqual(run_fma_mix(v0_c, v1_c, v2_c, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 1)),
                         ref_fma_mix_f32(v0_c, v1_c, v2_c, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 1)))

    def test_fma_mix_all_three_f32(self):
        # [0, 0, 0]: all three sources are f32 -> 1.25 * 4.0 + 2.75 = 7.75f
        v0 = f32_to_u32(1.25)
        v1 = f32_to_u32(4.0)
        v2 = f32_to_u32(2.75)
        self.assertEqual(run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0)),
                         ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0)))

    def test_fma_mix_modifiers_neg_abs_clamp(self):
        # Test operand negation and absolute value modifiers
        # s0: -|v1|, s1: v2, s2: v3
        v0 = f32_to_u32(2.0)
        v1 = f32_to_u32(3.0)
        v2 = f32_to_u32(10.0)
        # -|2.0| * 3.0 + 10.0 = -6.0 + 10.0 = 4.0f
        expected = ref_fma_mix_f32(v0, v1, v2, neg=(1, 0, 0), abs_mask=(1, 0, 0))
        got = run_fma_mix(v0, v1, v2, neg=(1, 0, 0), abs_mask=(1, 0, 0))
        self.assertEqual(got, expected)

        # Clamp positive overflow: 2.0 * 2.0 + 0.0 = 4.0 -> clamp -> 1.0f (0x3F800000)
        got_clamp_hi = run_fma_mix(f32_to_u32(2.0), f32_to_u32(2.0), f32_to_u32(0.0), clamp=True)
        self.assertEqual(got_clamp_hi, 0x3F800000)

        # Clamp negative underflow: 2.0 * -2.0 + 0.0 = -4.0 -> clamp -> 0.0f (0x00000000)
        got_clamp_lo = run_fma_mix(f32_to_u32(2.0), f32_to_u32(-2.0), f32_to_u32(0.0), clamp=True)
        self.assertEqual(got_clamp_lo, 0x00000000)


class FmaMixF32SubnormalTests(unittest.TestCase):
    """Audits mixed-precision FMA handling of f16 subnormals unpacked to single-precision float."""

    def test_f16_subnormal_unpacked_to_f32_exact(self):
        # An f16 subnormal 0x0001 (2^-24) unpacked to f32 is an exact normal single-precision float (0x33800000).
        # In v_fma_mix_f32: 2^-24 * 1.0 + 0.0 = 2^-24
        v0 = pack2x16(F16_MIN_SUBNORMAL, F16_ZERO)
        v1 = f32_to_u32(1.0)
        v2 = f32_to_u32(0.0)
        expected = ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        got = run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        self.assertEqual(got, expected)
        self.assertEqual(got, 0x33800000)
        self.assertAlmostEqual(u32_to_f32(got), 2.0**-24, delta=1e-15)

    def test_f16_largest_subnormal_in_mixed_fma(self):
        # 0x03FF unpacked to f32: (1023/1024) * 2^-14
        # Multiplied by 2.0f: 0x03FF * 2.0 + 0.0
        v0 = pack2x16(F16_MAX_SUBNORMAL, F16_ZERO)
        v1 = f32_to_u32(2.0)
        v2 = f32_to_u32(0.0)
        expected = ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        got = run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        self.assertEqual(got, expected)

    def test_f16_negative_subnormal_in_mixed_fma(self):
        # 0x8001 (-2^-24) * 4.0 + 1.0 = 1.0 - 2^-22
        v0 = pack2x16(F16_NEG_MIN_SUBNORMAL, F16_ZERO)
        v1 = f32_to_u32(4.0)
        v2 = f32_to_u32(1.0)
        expected = ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        got = run_fma_mix(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        self.assertEqual(got, expected)

    def test_mixed_f16_subnormals_with_high_half_selection(self):
        # v0: lo=normal 1.0, hi=subnormal 0x0001
        # op_sel:[1, 0, 0] selects high half (subnormal 2^-24)
        v0 = pack2x16(F16_ONE, F16_MIN_SUBNORMAL)
        v1 = pack2x16(F16_TWO, F16_ZERO)
        v2 = f32_to_u32(0.5)
        # 2^-24 * 2.0 + 0.5 = 2^-23 + 0.5
        expected = ref_fma_mix_f32(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 1, 0))
        got = run_fma_mix(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 1, 0))
        self.assertEqual(got, expected)


class NegativeControlTests(unittest.TestCase):
    """Verify that mutations in operands, selections, or precision are detected as failures."""

    def test_pk_fma_corruption_detected(self):
        v0 = pack2x16(f2h(1.0), f2h(2.0))
        v1 = pack2x16(f2h(3.0), f2h(4.0))
        v2 = pack2x16(f2h(0.5), f2h(1.0))
        good = ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1))

        # Inverting op_sel or op_sel_hi should produce a different result
        bad_sel = ref_pk_fma_f16(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 1, 1))
        self.assertNotEqual(good, bad_sel)

        bad_hi = ref_pk_fma_f16(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(0, 1, 1))
        self.assertNotEqual(good, bad_hi)

    def test_fma_mix_corruption_detected(self):
        v0 = pack2x16(f2h(1.0), f2h(50.0))
        v1 = f32_to_u32(2.0)
        v2 = f32_to_u32(3.0)
        # f16(1.0) * 2.0 + 3.0 = 5.0
        good = ref_fma_mix_f32(v0, v1, v2, op_sel=(0, 0, 0), op_sel_hi=(1, 0, 0))
        # Corrupted to high half: f16(50.0) * 2.0 + 3.0 = 103.0
        corrupted = ref_fma_mix_f32(v0, v1, v2, op_sel=(1, 0, 0), op_sel_hi=(1, 0, 0))
        self.assertNotEqual(good, corrupted)

    def test_subnormal_precision_loss_detected(self):
        # Adding smallest subnormal to 0x03FF should strictly yield 0x0400, not 0x03FF
        res = ref_pk_fma_f16(
            pack2x16(F16_MAX_SUBNORMAL, F16_MAX_SUBNORMAL),
            pack2x16(F16_ONE, F16_ONE),
            pack2x16(F16_MIN_SUBNORMAL, F16_MIN_SUBNORMAL),
        )
        lo, _ = unpack2x16(res)
        self.assertEqual(lo, F16_MIN_NORMAL)
        self.assertNotEqual(lo, F16_MAX_SUBNORMAL)


if __name__ == "__main__":
    unittest.main()
