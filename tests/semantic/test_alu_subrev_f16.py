"""Semantic interpreter tests for RDNA2 v_subrev_f16, v_subrev_nc_u32, and packed half-precision reverse subtraction.

v_subrev_f16: D.f16 = S1.f16 - S0.f16 (VOP2 0x34 on GFX10), the reversed form of v_sub_f16. Like the other f16
VOP2/VOP3 ops it writes the low half of the destination and keeps the high half; in the VOP3 form op_sel bit 0 / 1
reads the high half of src0 / src1 and bit 2 writes the high half of the destination instead (docs/design.md,
"16-bit results"). Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py and compares the exported word with a plain-Python reference: the difference
of the two f16 values computed in double precision and rounded once to f16 (round to nearest even).

Also covers v_subrev_nc_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF, the non-carry 32-bit reverse subtract,
packed 16-bit lane execution (modeling v_pk_subrev_f16 on lo and hi halves), SIMD divergence under EXEC mask,
and comprehensive IEEE 754 edge cases (signed zeros, infinities, NaN quieting and propagation).

Standard library only: zero game data, no GPU hardware required.
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xAAAA5555
MASK32 = 0xFFFFFFFF

# Half-precision constants
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_SUB_MIN = 0x0001     # Smallest positive subnormal (~5.96e-8)
H_SUB_MAX = 0x03FF     # Largest positive subnormal (~6.097e-5)
H_MIN_NORMAL = 0x0400  # Smallest positive normalized half (2^-14)
H_HALF = 0x3800        # 0.5
H_ONE = 0x3C00         # 1.0
H_ONE_P_FIVE = 0x3E00  # 1.5
H_TWO = 0x4000         # 2.0
H_THREE = 0x4200       # 3.0
H_FOUR = 0x4400        # 4.0
H_NEG_HALF = 0xB800    # -0.5
H_NEG_ONE = 0xBC00     # -1.0
H_NEG_TWO = 0xC000     # -2.0
H_POS_INF = 0x7C00     # +inf
H_NEG_INF = 0xFC00     # -inf
H_QNAN = 0x7E00        # quiet NaN
H_QNAN_PAYLOAD = 0x7E01  # quiet NaN with payload
H_SNAN = 0x7C01        # signaling NaN
H_SNAN_PAYLOAD = 0x7DFF  # signaling NaN with payload
H_NEG_QNAN = 0xFE00    # negative quiet NaN
H_NEG_SNAN = 0xFC01    # negative signaling NaN
H_MAX_FINITE = 0x7BFF  # 65504.0
H_NEG_MAX = 0xFBFF     # -65504.0

A = 0x44003E00         # v1: lo 1.5, hi 4.0
B = 0x40003800         # v2: lo 0.5, hi 2.0


def h2f(h: int) -> float:
    """Unpack 16-bit integer as IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    """Pack float to 16-bit IEEE 754 half-precision integer with round-to-nearest-even."""
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def is_nan_f16(h: int) -> bool:
    """Check if 16-bit integer represents an IEEE 754 half-precision NaN."""
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_qnan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a quiet NaN."""
    return is_nan_f16(h) and bool(h & 0x0200)


def is_snan_f16(h: int) -> bool:
    """Check if 16-bit integer represents a signaling NaN."""
    return is_nan_f16(h) and not bool(h & 0x0200)


def is_inf_f16(h: int) -> bool:
    """Check if 16-bit integer represents an IEEE 754 infinity."""
    return (h & 0x7FFF) == 0x7C00


def half(word: int, hi: int) -> float:
    """Extract float from low (hi=0) or high (hi=1) 16 bits of a 32-bit word."""
    return h2f(word >> 16 if hi else word)


def merge_half(old: int, h: int, dst_hi: int) -> int:
    """Merge 16-bit result into low (dst_hi=0) or high (dst_hi=1) half of old word."""
    h &= 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def pack16(lo: int, hi: int) -> int:
    """Pack two 16-bit values into a 32-bit register word."""
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def ref_subrev_f16(s0: int, s1: int) -> int:
    """Reference implementation of v_subrev_f16: D.f16 = S1.f16 - S0.f16."""
    s0 &= 0xFFFF
    s1 &= 0xFFFF
    if is_nan_f16(s0) or is_nan_f16(s1):
        return H_QNAN
    inf0, inf1 = is_inf_f16(s0), is_inf_f16(s1)
    if inf0 and inf1:
        if (s0 >> 15) == (s1 >> 15):
            return H_QNAN
        return s1
    if inf1:
        return s1
    if inf0:
        return H_NEG_INF if (s0 >> 15) == 0 else H_POS_INF
    f0, f1 = h2f(s0), h2f(s1)
    diff = f1 - f0
    if diff == 0.0:
        if s1 == H_NEG_ZERO and s0 == H_POS_ZERO:
            return H_NEG_ZERO
        return H_POS_ZERO
    return f2h(diff)


def ref_sub_f16(s0: int, s1: int) -> int:
    """Reference implementation of v_sub_f16: D.f16 = S0.f16 - S1.f16."""
    return ref_subrev_f16(s1, s0)


def ref_subrev_nc_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_subrev_nc_u32: D.u32 = (S1.u32 - S0.u32) & 0xFFFFFFFF."""
    return ((s1 & MASK32) - (s0 & MASK32)) & MASK32


def ref_sub_nc_u32(s0: int, s1: int) -> int:
    """Reference implementation of v_sub_nc_u32: D.u32 = (S0.u32 - S1.u32) & 0xFFFFFFFF."""
    return ((s0 & MASK32) - (s1 & MASK32)) & MASK32


def ref_pk_subrev_f16(s0: int, s1: int, clamp: bool = False) -> int:
    """Reference implementation of packed 16-bit reverse subtraction (v_pk_subrev_f16):
    D.lo = S1.lo - S0.lo
    D.hi = S1.hi - S0.hi
    """
    s0_lo, s0_hi = s0 & 0xFFFF, (s0 >> 16) & 0xFFFF
    s1_lo, s1_hi = s1 & 0xFFFF, (s1 >> 16) & 0xFFFF
    d_lo = ref_subrev_f16(s0_lo, s1_lo)
    d_hi = ref_subrev_f16(s0_hi, s1_hi)
    if clamp:
        if not is_nan_f16(d_lo):
            d_lo = f2h(max(0.0, min(1.0, h2f(d_lo))))
        if not is_nan_f16(d_hi):
            d_hi = f2h(max(0.0, min(1.0, h2f(d_hi))))
    return pack16(d_lo, d_hi)


def ref_pk_sub_f16(s0: int, s1: int, clamp: bool = False) -> int:
    """Reference implementation of packed 16-bit forward subtraction: D = S0 - S1 on both halves."""
    return ref_pk_subrev_f16(s1, s0, clamp=clamp)


def ref_subrev_f16_opsel(
    w0: int,
    w1: int,
    old: int,
    sel0: int,
    sel1: int,
    dst_hi: int,
    neg0: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Compute VOP3 v_subrev_f16 result with op_sel and input/output modifiers."""
    h0 = (w0 >> 16 if sel0 else w0) & 0xFFFF
    h1 = (w1 >> 16 if sel1 else w1) & 0xFFFF
    if neg0:
        h0 ^= 0x8000
    if neg1:
        h1 ^= 0x8000
    res = ref_subrev_f16(h0, h1)
    if clamp and not is_nan_f16(res):
        f = h2f(res)
        f_clamped = max(0.0, min(1.0, f))
        res = f2h(f_clamped)
    return merge_half(old, res, dst_hi)


class HalfVectorALUInterpreter:
    """Deterministic SIMD vector ALU interpreter for half-precision and 32-bit operations.

    Models:
      - 32-lane (Wave32) or 64-lane (Wave64) SIMD execution
      - VGPR file: 32-bit registers containing packed binary16 halves or u32 integers
      - SGPR file: 32-bit scalar registers with automatic broadcast to vector lanes
      - EXEC mask: lane gating where inactive lanes preserve pre-existing register state
      - Inline constant and register operand resolution
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

    def execute_v_subrev_f16(
        self,
        dst_idx: int,
        src0: str | int,
        src1: str | int,
        op_sel: tuple[int, int, int] = (0, 0, 0),
        clamp: bool = False,
    ) -> None:
        sel0, sel1, dst_hi = op_sel
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                old = self.vgprs[dst_idx][lane]
                self.vgprs[dst_idx][lane] = ref_subrev_f16_opsel(
                    s0, s1, old, sel0, sel1, dst_hi, clamp=clamp
                )

    def execute_v_sub_f16(
        self,
        dst_idx: int,
        src0: str | int,
        src1: str | int,
        op_sel: tuple[int, int, int] = (0, 0, 0),
        clamp: bool = False,
    ) -> None:
        sel0, sel1, dst_hi = op_sel
        self.execute_v_subrev_f16(
            dst_idx, src1, src0, op_sel=(sel1, sel0, dst_hi), clamp=clamp
        )

    def execute_v_pk_subrev_f16(
        self,
        dst_idx: int,
        src0: str | int,
        src1: str | int,
        clamp: bool = False,
    ) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_pk_subrev_f16(s0, s1, clamp=clamp)

    def execute_v_subrev_nc_u32(
        self,
        dst_idx: int,
        src0: str | int,
        src1: str | int,
    ) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_subrev_nc_u32(s0, s1)

    def execute_v_sub_nc_u32(
        self,
        dst_idx: int,
        src0: str | int,
        src1: str | int,
    ) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.resolve_src(src0, lane)
                s1 = self.resolve_src(src1, lane)
                self.vgprs[dst_idx][lane] = ref_sub_nc_u32(s0, s1)


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_subrev_f16.s")
    with open(os.path.join(_TMP_DIR.name, "ps_subrev_f16.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run(line: str, v1: int, v2: int, old: int = OLD) -> int:
    """Seed v1, v2 and destination v0, run assembly line, return exported v0."""
    return run_shader([line], v1=v1, v2=v2, old=old)


def run_shader(lines: list[str], v1: int = 0, v2: int = 0, old: int = OLD) -> int:
    """Seed v1, v2 and destination v0, run lines, return exported v0."""
    if _TMP_DIR is None:
        setUpModule()
    preamble = [
        f"v_mov_b32 v1, {v1 & MASK32:#x}",
        f"v_mov_b32 v2, {v2 & MASK32:#x}",
        f"v_mov_b32 v0, {old & MASK32:#x}",
    ]
    asm = (
        "; test_subrev_f16\n"
        + "".join(f"\t{ln}\n" for ln in preamble + list(lines))
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


class SubrevF16ReferenceOracleTests(unittest.TestCase):
    """Mathematical properties and correctness of the half-precision reverse subtraction oracle."""

    def test_formula_s1_minus_s0(self):
        self.assertEqual(ref_subrev_f16(H_ONE_P_FIVE, H_HALF), H_NEG_ONE)
        self.assertEqual(ref_subrev_f16(H_HALF, H_ONE_P_FIVE), H_ONE)
        self.assertEqual(ref_subrev_f16(H_ONE, H_THREE), H_TWO)
        self.assertEqual(ref_subrev_f16(H_TWO, H_ONE), H_NEG_ONE)

    def test_non_commutativity(self):
        pairs = [
            (H_ONE_P_FIVE, H_HALF),
            (H_ONE, H_THREE),
            (H_TWO, H_FOUR),
            (H_NEG_ONE, H_ONE),
        ]
        for s0, s1 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d_rev = ref_subrev_f16(s0, s1)
                d_rev_swapped = ref_subrev_f16(s1, s0)
                self.assertNotEqual(d_rev, d_rev_swapped)

    def test_anti_symmetry(self):
        pairs = [
            (H_ONE_P_FIVE, H_HALF),
            (H_ONE, H_THREE),
            (H_TWO, H_FOUR),
            (H_HALF, H_TWO),
            (H_NEG_ONE, H_TWO),
        ]
        for s0, s1 in pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                r1 = ref_subrev_f16(s0, s1)
                r2 = ref_subrev_f16(s1, s0)
                self.assertEqual(r1 ^ 0x8000, r2)
                self.assertAlmostEqual(h2f(r1), -h2f(r2), places=3)

    def test_identity_properties(self):
        values = [H_HALF, H_ONE, H_ONE_P_FIVE, H_TWO, H_THREE, H_FOUR, H_NEG_ONE, H_NEG_TWO]
        for v in values:
            with self.subTest(v=hex(v)):
                self.assertEqual(ref_subrev_f16(H_POS_ZERO, v), v)
                self.assertEqual(ref_subrev_f16(v, v), H_POS_ZERO)
                self.assertEqual(ref_subrev_f16(v, H_POS_ZERO), v ^ 0x8000)

    def test_reversal_duality_with_sub(self):
        rng = random.Random(0x3416)
        samples = [H_POS_ZERO, H_HALF, H_ONE, H_ONE_P_FIVE, H_TWO, H_THREE, H_FOUR, H_NEG_ONE, H_NEG_TWO]
        for _ in range(50):
            s0 = rng.choice(samples)
            s1 = rng.choice(samples)
            self.assertEqual(ref_subrev_f16(s0, s1), ref_sub_f16(s1, s0))

    def test_u32_reverse_subtraction(self):
        self.assertEqual(ref_subrev_nc_u32(30, 100), 70)
        self.assertEqual(ref_subrev_nc_u32(100, 30), (30 - 100) & MASK32)
        self.assertEqual(ref_subrev_nc_u32(0, 42), 42)
        self.assertEqual(ref_subrev_nc_u32(42, 0), (-42) & MASK32)
        self.assertEqual(ref_subrev_nc_u32(0x80000000, 0x7FFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_subrev_nc_u32(0x7FFFFFFF, 0x80000000), 1)

        for a, b in [(10, 50), (100, 25), (0x12345678, 0x87654321)]:
            fwd = ref_sub_nc_u32(a, b)
            rev = ref_subrev_nc_u32(a, b)
            self.assertEqual((fwd + rev) & MASK32, 0)
            self.assertEqual(ref_subrev_nc_u32(a, b), ref_sub_nc_u32(b, a))


class SubrevF16ShaderTranslationTests(unittest.TestCase):
    """End-to-end pixel shader execution comparing rdna2spirv output against reference oracle."""

    def test_vop2_is_src1_minus_src0_and_keeps_the_high_half(self):
        got = run("v_subrev_f16_e32 v0, v1, v2", A, B)
        want = merge_half(OLD, ref_subrev_f16(0x3E00, 0x3800), 0)
        self.assertEqual(got, want)

    def test_vop3_op_sel(self):
        for bits in itertools.product((0, 1), repeat=3):
            line = f"v_subrev_f16_e64 v0, v1, v2 op_sel:[{bits[0]},{bits[1]},{bits[2]}]"
            with self.subTest(line=line):
                want = ref_subrev_f16_opsel(A, B, OLD, bits[0], bits[1], bits[2])
                self.assertEqual(run(line, A, B), want)

    def test_neg_on_src0_and_clamp(self):
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2", A, B), merge_half(OLD, f2h(2.0), 0))
        self.assertEqual(run("v_subrev_f16_e64 v0, -v1, v2 clamp", A, B), merge_half(OLD, f2h(1.0), 0))

    def test_matches_v_sub_f16_with_swapped_operands(self):
        rng = random.Random(0x34)
        special = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x3555]
        for _ in range(40):
            x, y = (rng.choice(special) if rng.random() < 0.3 else rng.randrange(0x10000) for _ in range(2))
            if (x & 0x7C00) == 0x7C00 or (y & 0x7C00) == 0x7C00:
                continue
            v1, v2 = 0x12340000 | x, 0x56780000 | y
            with self.subTest(x=hex(x), y=hex(y)):
                got = run("v_subrev_f16_e32 v0, v1, v2", v1, v2)
                self.assertEqual(got, run("v_sub_f16_e32 v0, v2, v1", v1, v2))
                want = ref_subrev_f16(x, y)
                if want in (0x0000, 0x8000):
                    self.assertEqual(got & 0xFFFF7FFF, OLD & 0xFFFF0000)
                else:
                    self.assertEqual(got, merge_half(OLD, want, 0))

    def test_v_subrev_nc_u32_shader_execution(self):
        lines = ["v_subrev_nc_u32 v0, v1, v2"]
        self.assertEqual(run_shader(lines, v1=30, v2=100), 70)
        self.assertEqual(run_shader(lines, v1=100, v2=30), (30 - 100) & MASK32)
        self.assertEqual(run_shader(lines, v1=0x80000000, v2=0x7FFFFFFF), 0xFFFFFFFF)

    def test_inline_constants_and_sgpr_sources(self):
        lines = ["v_subrev_f16_e32 v0, 0x3800, v2"]
        self.assertEqual(run_shader(lines, v2=H_TWO), merge_half(OLD, H_ONE_P_FIVE, 0))

        lines_sgpr = [
            "s_mov_b32 s0, 50",
            "v_subrev_nc_u32 v0, s0, v2",
        ]
        self.assertEqual(run_shader(lines_sgpr, v2=120), 70)


class PackedSubrevF16OracleTests(unittest.TestCase):
    """Reference oracle tests for packed 16-bit lane execution modeling v_pk_subrev_f16."""

    def test_packed_subrev_formula(self):
        s0 = pack16(H_HALF, H_ONE)
        s1 = pack16(H_TWO, H_THREE)
        res = ref_pk_subrev_f16(s0, s1)
        self.assertEqual(res & 0xFFFF, H_ONE_P_FIVE)
        self.assertEqual((res >> 16) & 0xFFFF, H_TWO)

    def test_packed_non_commutativity(self):
        s0 = pack16(H_HALF, H_ONE)
        s1 = pack16(H_TWO, H_THREE)
        res_fwd = ref_pk_subrev_f16(s0, s1)
        res_swapped = ref_pk_subrev_f16(s1, s0)
        self.assertNotEqual(res_fwd, res_swapped)
        self.assertNotEqual(res_fwd & 0xFFFF, res_swapped & 0xFFFF)
        self.assertNotEqual((res_fwd >> 16) & 0xFFFF, (res_swapped >> 16) & 0xFFFF)

    def test_packed_anti_symmetry(self):
        s0 = pack16(H_HALF, H_ONE)
        s1 = pack16(H_TWO, H_THREE)
        r1 = ref_pk_subrev_f16(s0, s1)
        r2 = ref_pk_subrev_f16(s1, s0)
        self.assertEqual((r1 & 0xFFFF) ^ 0x8000, r2 & 0xFFFF)
        self.assertEqual(((r1 >> 16) & 0xFFFF) ^ 0x8000, (r2 >> 16) & 0xFFFF)

    def test_packed_lane_isolation(self):
        base_s0 = pack16(H_ONE, H_ONE)
        base_s1 = pack16(H_TWO, H_THREE)
        res_base = ref_pk_subrev_f16(base_s0, base_s1)

        mod_s0 = pack16(H_HALF, H_ONE)
        res_mod = ref_pk_subrev_f16(mod_s0, base_s1)
        self.assertNotEqual(res_base & 0xFFFF, res_mod & 0xFFFF)
        self.assertEqual((res_base >> 16) & 0xFFFF, (res_mod >> 16) & 0xFFFF)

        mod_s1 = pack16(H_TWO, H_FOUR)
        res_mod2 = ref_pk_subrev_f16(base_s0, mod_s1)
        self.assertEqual(res_base & 0xFFFF, res_mod2 & 0xFFFF)
        self.assertNotEqual((res_base >> 16) & 0xFFFF, (res_mod2 >> 16) & 0xFFFF)

    def test_packed_reversal_duality(self):
        s0 = pack16(H_ONE, H_TWO)
        s1 = pack16(H_THREE, H_FOUR)
        self.assertEqual(ref_pk_subrev_f16(s0, s1), ref_pk_sub_f16(s1, s0))


class HalfVectorALUInterpreterTests(unittest.TestCase):
    """Multi-lane SIMD interpreter tests modeling packed subrev, EXEC divergence, and state preservation."""

    def test_simd_wave64_packed_subrev(self):
        interp = HalfVectorALUInterpreter(wave_size=64)
        for lane in range(64):
            s0 = pack16(f2h(1.0 + lane * 0.1), f2h(2.0 + lane * 0.2))
            s1 = pack16(f2h(5.0 + lane * 0.5), f2h(10.0 + lane * 1.0))
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)

        interp.execute_v_pk_subrev_f16(0, "v1", "v2")

        for lane in range(64):
            expected = ref_pk_subrev_f16(interp.get_vgpr(1, lane), interp.get_vgpr(2, lane))
            self.assertEqual(interp.get_vgpr(0, lane), expected, f"lane {lane}")

    def test_simd_wave32_packed_subrev(self):
        interp = HalfVectorALUInterpreter(wave_size=32)
        for lane in range(32):
            s0 = pack16(H_HALF, H_ONE)
            s1 = pack16(H_TWO, H_THREE)
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)

        interp.execute_v_pk_subrev_f16(0, "v1", "v2")

        expected = ref_pk_subrev_f16(pack16(H_HALF, H_ONE), pack16(H_TWO, H_THREE))
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), expected, f"lane {lane}")

    def test_simd_exec_mask_partial_divergence(self):
        interp = HalfVectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0xDEADBEEF)
            interp.set_vgpr(1, lane, pack16(H_HALF, H_ONE))
            interp.set_vgpr(2, lane, pack16(H_TWO, H_FOUR))

        interp.set_exec(0x5555555555555555)
        interp.execute_v_pk_subrev_f16(0, "v1", "v2")

        expected = ref_pk_subrev_f16(pack16(H_HALF, H_ONE), pack16(H_TWO, H_FOUR))
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), expected, f"active lane {lane}")
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0xDEADBEEF, f"inactive lane {lane}")

    def test_simd_exec_mask_single_lane(self):
        interp = HalfVectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0xCAFEBABE)
            interp.set_vgpr(1, lane, pack16(H_ONE, H_TWO))
            interp.set_vgpr(2, lane, pack16(H_THREE, H_FOUR))

        interp.set_exec(1 << 7)
        interp.execute_v_pk_subrev_f16(0, "v1", "v2")

        expected = ref_pk_subrev_f16(pack16(H_ONE, H_TWO), pack16(H_THREE, H_FOUR))
        for lane in range(64):
            if lane == 7:
                self.assertEqual(interp.get_vgpr(0, lane), expected)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0xCAFEBABE)

    def test_simd_exec_mask_all_inactive(self):
        interp = HalfVectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0x12345678)
            interp.set_vgpr(1, lane, pack16(H_HALF, H_HALF))
            interp.set_vgpr(2, lane, pack16(H_TWO, H_TWO))

        interp.set_exec(0)
        interp.execute_v_pk_subrev_f16(0, "v1", "v2")

        for lane in range(64):
            self.assertEqual(interp.get_vgpr(0, lane), 0x12345678)

    def test_simd_scalar_broadcast_and_inline_constants(self):
        interp = HalfVectorALUInterpreter(wave_size=32)
        interp.set_sgpr(0, pack16(H_ONE, H_TWO))
        for lane in range(32):
            interp.set_vgpr(1, lane, pack16(H_THREE, H_FOUR))

        interp.execute_v_pk_subrev_f16(0, "s0", "v1")
        expected = ref_pk_subrev_f16(pack16(H_ONE, H_TWO), pack16(H_THREE, H_FOUR))
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_simd_v_subrev_f16_half_preservation(self):
        interp = HalfVectorALUInterpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(0, lane, OLD)
            interp.set_vgpr(1, lane, pack16(H_HALF, H_TWO))
            interp.set_vgpr(2, lane, pack16(H_ONE_P_FIVE, H_THREE))

        interp.set_exec(0x55555555)
        interp.execute_v_subrev_f16(0, "v1", "v2", op_sel=(0, 0, 0))

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), merge_half(OLD, H_ONE, 0))
            else:
                self.assertEqual(interp.get_vgpr(0, lane), OLD)

    def test_simd_v_subrev_nc_u32_exec_divergence(self):
        interp = HalfVectorALUInterpreter(wave_size=64)
        for lane in range(64):
            interp.set_vgpr(0, lane, 0x11111111)
            interp.set_vgpr(1, lane, 50)
            interp.set_vgpr(2, lane, 200 + lane)

        interp.set_exec(0xFFFFFFFF00000000)
        interp.execute_v_subrev_nc_u32(0, "v1", "v2")

        for lane in range(64):
            if lane >= 32:
                self.assertEqual(interp.get_vgpr(0, lane), (200 + lane) - 50)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 0x11111111)


class ShaderPackedAndExecExecutionTests(unittest.TestCase):
    """Shader execution tests for packed operations and EXEC mask gating."""

    def test_packed_execution_via_dual_subrev_f16(self):
        lines = [
            "v_subrev_f16_e64 v0, v1, v2 op_sel:[0,0,0]",
            "v_subrev_f16_e64 v0, v1, v2 op_sel:[1,1,1]",
        ]
        v1 = pack16(H_HALF, H_ONE)
        v2 = pack16(H_TWO, H_THREE)
        got = run_shader(lines, v1=v1, v2=v2, old=OLD)
        want = ref_pk_subrev_f16(v1, v2)
        self.assertEqual(got, want)

    def test_packed_execution_via_v_pk_add_f16_with_negated_src0(self):
        lines = [
            "v_pk_add_f16 v0, v1, v2 neg_lo:[1,0] neg_hi:[1,0]",
        ]
        v1 = pack16(H_HALF, H_ONE)
        v2 = pack16(H_TWO, H_THREE)
        got = run_shader(lines, v1=v1, v2=v2, old=OLD)
        want = ref_pk_subrev_f16(v1, v2)
        self.assertEqual(got, want)

    def test_exec_masking_preserves_destination(self):
        lines = [
            "s_mov_b32 exec_lo, 0",
            "v_subrev_f16_e32 v0, v1, v2",
            "s_mov_b32 exec_lo, 1",
        ]
        got = run_shader(lines, v1=H_HALF, v2=H_TWO, old=OLD)
        self.assertEqual(got, OLD)

    def test_exec_masking_toggled_execution(self):
        lines = [
            "s_mov_b32 exec_lo, 0",
            "v_subrev_f16_e64 v0, v1, v2 op_sel:[0,0,0]",
            "s_mov_b32 exec_lo, 1",
            "v_subrev_f16_e64 v0, v1, v2 op_sel:[1,1,1]",
        ]
        v1 = pack16(H_HALF, H_ONE)
        v2 = pack16(H_TWO, H_THREE)
        got = run_shader(lines, v1=v1, v2=v2, old=OLD)
        expected = merge_half(OLD, H_TWO, 1)
        self.assertEqual(got, expected)


class Ieee754SpecialValuesOracleTests(unittest.TestCase):
    """IEEE 754 half-precision special values: signed zeros, infinities, NaN quieting and propagation."""

    def test_signed_zero_reverse_subtraction(self):
        # S1 - S0 in half precision:
        # (+0.0) - (+0.0) == +0.0
        self.assertEqual(ref_subrev_f16(H_POS_ZERO, H_POS_ZERO), H_POS_ZERO)
        # (-0.0) - (+0.0) == -0.0
        self.assertEqual(ref_subrev_f16(H_POS_ZERO, H_NEG_ZERO), H_NEG_ZERO)
        # (+0.0) - (-0.0) == +0.0
        self.assertEqual(ref_subrev_f16(H_NEG_ZERO, H_POS_ZERO), H_POS_ZERO)
        # (-0.0) - (-0.0) == +0.0 (IEEE 754 round to nearest: -0 - (-0) = -0 + 0 = +0)
        self.assertEqual(ref_subrev_f16(H_NEG_ZERO, H_NEG_ZERO), H_POS_ZERO)

    def test_infinity_arithmetic(self):
        # (+inf) - (+inf) == NaN
        self.assertTrue(is_nan_f16(ref_subrev_f16(H_POS_INF, H_POS_INF)))
        # (-inf) - (-inf) == NaN
        self.assertTrue(is_nan_f16(ref_subrev_f16(H_NEG_INF, H_NEG_INF)))
        # (+inf) - (-inf) == +inf (S1=+inf, S0=-inf)
        self.assertEqual(ref_subrev_f16(H_NEG_INF, H_POS_INF), H_POS_INF)
        # (-inf) - (+inf) == -inf (S1=-inf, S0=+inf)
        self.assertEqual(ref_subrev_f16(H_POS_INF, H_NEG_INF), H_NEG_INF)

        # Finite S1 minus +/-inf S0
        self.assertEqual(ref_subrev_f16(H_POS_INF, H_ONE), H_NEG_INF)
        self.assertEqual(ref_subrev_f16(H_NEG_INF, H_ONE), H_POS_INF)
        self.assertEqual(ref_subrev_f16(H_POS_INF, H_NEG_ONE), H_NEG_INF)
        self.assertEqual(ref_subrev_f16(H_NEG_INF, H_NEG_ONE), H_POS_INF)

        # +/-inf S1 minus finite S0
        self.assertEqual(ref_subrev_f16(H_ONE, H_POS_INF), H_POS_INF)
        self.assertEqual(ref_subrev_f16(H_ONE, H_NEG_INF), H_NEG_INF)
        self.assertEqual(ref_subrev_f16(H_NEG_ONE, H_POS_INF), H_POS_INF)
        self.assertEqual(ref_subrev_f16(H_NEG_ONE, H_NEG_INF), H_NEG_INF)

    def test_nan_propagation_and_quieting(self):
        # Quiet NaN in S0 or S1 propagates as quiet NaN
        res_q0 = ref_subrev_f16(H_QNAN, H_ONE)
        self.assertTrue(is_qnan_f16(res_q0))
        res_q1 = ref_subrev_f16(H_ONE, H_QNAN)
        self.assertTrue(is_qnan_f16(res_q1))

        # Signaling NaN in S0 or S1 is quieted to qNaN
        res_s0 = ref_subrev_f16(H_SNAN, H_ONE)
        self.assertTrue(is_qnan_f16(res_s0))
        res_s1 = ref_subrev_f16(H_ONE, H_SNAN)
        self.assertTrue(is_qnan_f16(res_s1))

        # Both operands NaN
        self.assertTrue(is_qnan_f16(ref_subrev_f16(H_QNAN, H_QNAN)))
        self.assertTrue(is_qnan_f16(ref_subrev_f16(H_SNAN, H_QNAN)))
        self.assertTrue(is_qnan_f16(ref_subrev_f16(H_SNAN, H_SNAN)))
        self.assertTrue(is_qnan_f16(ref_subrev_f16(H_SNAN_PAYLOAD, H_QNAN_PAYLOAD)))

    def test_finite_overflow_to_infinity(self):
        # 65504.0 - (-65504.0) overflows to +inf
        self.assertEqual(ref_subrev_f16(H_NEG_MAX, H_MAX_FINITE), H_POS_INF)
        # -65504.0 - 65504.0 overflows to -inf
        self.assertEqual(ref_subrev_f16(H_MAX_FINITE, H_NEG_MAX), H_NEG_INF)

    def test_subnormal_boundaries(self):
        # Subnormal subtraction
        self.assertEqual(ref_subrev_f16(H_SUB_MIN, H_SUB_MAX), H_SUB_MAX - H_SUB_MIN)
        # Minimum normal minus smallest subnormal transitions to largest subnormal
        diff = ref_subrev_f16(H_SUB_MIN, H_MIN_NORMAL)
        self.assertEqual(diff, H_SUB_MAX)


class Ieee754SpecialValuesShaderExecutionTests(unittest.TestCase):
    """End-to-end pixel shader execution for special values through rdna2spirv and spirv_mini_interp."""

    def test_signed_zero_shader_execution(self):
        # (+0.0) - (+0.0) == +0.0
        got_pos_pos = run("v_subrev_f16_e32 v0, v1, v2", H_POS_ZERO, H_POS_ZERO)
        self.assertEqual(got_pos_pos & 0xFFFF, H_POS_ZERO)

        # (-0.0) - (+0.0) == -0.0
        got_neg_pos = run("v_subrev_f16_e32 v0, v1, v2", H_POS_ZERO, H_NEG_ZERO)
        self.assertEqual(got_neg_pos & 0xFFFF, H_NEG_ZERO)

        # (+0.0) - (-0.0) == +0.0
        got_pos_neg = run("v_subrev_f16_e32 v0, v1, v2", H_NEG_ZERO, H_POS_ZERO)
        self.assertEqual(got_pos_neg & 0xFFFF, H_POS_ZERO)

        # (-0.0) - (-0.0) == +0.0
        got_neg_neg = run("v_subrev_f16_e32 v0, v1, v2", H_NEG_ZERO, H_NEG_ZERO)
        self.assertEqual(got_neg_neg & 0xFFFF, H_POS_ZERO)

    def test_infinity_subtraction_shader_execution(self):
        # (+inf) - (+inf) == NaN
        got_inf_nan = run("v_subrev_f16_e32 v0, v1, v2", H_POS_INF, H_POS_INF) & 0xFFFF
        self.assertTrue(is_nan_f16(got_inf_nan))

        # (-inf) - (-inf) == NaN
        got_ninf_nan = run("v_subrev_f16_e32 v0, v1, v2", H_NEG_INF, H_NEG_INF) & 0xFFFF
        self.assertTrue(is_nan_f16(got_ninf_nan))

        # (+inf) - (-inf) == +inf
        got_pinf = run("v_subrev_f16_e32 v0, v1, v2", H_NEG_INF, H_POS_INF) & 0xFFFF
        self.assertEqual(got_pinf, H_POS_INF)

        # (-inf) - (+inf) == -inf
        got_ninf = run("v_subrev_f16_e32 v0, v1, v2", H_POS_INF, H_NEG_INF) & 0xFFFF
        self.assertEqual(got_ninf, H_NEG_INF)

    def test_nan_propagation_shader_execution(self):
        # Quiet NaN in S0 produces NaN
        got_q0 = run("v_subrev_f16_e32 v0, v1, v2", H_QNAN, H_ONE) & 0xFFFF
        self.assertTrue(is_nan_f16(got_q0))

        # Quiet NaN in S1 produces NaN
        got_q1 = run("v_subrev_f16_e32 v0, v1, v2", H_ONE, H_QNAN) & 0xFFFF
        self.assertTrue(is_nan_f16(got_q1))

        # Signaling NaN in S0 is quieted to qNaN
        got_s0 = run("v_subrev_f16_e32 v0, v1, v2", H_SNAN, H_ONE) & 0xFFFF
        self.assertTrue(is_qnan_f16(got_s0))

        # Signaling NaN in S1 is quieted to qNaN
        got_s1 = run("v_subrev_f16_e32 v0, v1, v2", H_ONE, H_SNAN) & 0xFFFF
        self.assertTrue(is_qnan_f16(got_s1))

        # Both NaNs produce NaN
        got_both = run("v_subrev_f16_e32 v0, v1, v2", H_SNAN, H_QNAN) & 0xFFFF
        self.assertTrue(is_nan_f16(got_both))

    def test_overflow_to_infinity_shader_execution(self):
        # 65504.0 - (-65504.0) produces +inf
        got_overflow_pos = run("v_subrev_f16_e32 v0, v1, v2", H_NEG_MAX, H_MAX_FINITE) & 0xFFFF
        self.assertEqual(got_overflow_pos, H_POS_INF)

        # -65504.0 - 65504.0 produces -inf
        got_overflow_neg = run("v_subrev_f16_e32 v0, v1, v2", H_MAX_FINITE, H_NEG_MAX) & 0xFFFF
        self.assertEqual(got_overflow_neg, H_NEG_INF)

    def test_vop3_clamping_with_special_values(self):
        # Negative result clamped to +0.0: 0.5 - 2.0 = -1.5 -> 0.0
        got_neg_clamp = run("v_subrev_f16_e64 v0, v1, v2 clamp", H_TWO, H_HALF) & 0xFFFF
        self.assertEqual(got_neg_clamp, H_POS_ZERO)

        # Positive result > 1.0 clamped to 1.0: 3.0 - 1.0 = 2.0 -> 1.0
        got_pos_clamp = run("v_subrev_f16_e64 v0, v1, v2 clamp", H_ONE, H_THREE) & 0xFFFF
        self.assertEqual(got_pos_clamp, H_ONE)

        # Overflow to +inf clamped to 1.0
        got_inf_clamp = run("v_subrev_f16_e64 v0, v1, v2 clamp", H_NEG_INF, H_POS_INF) & 0xFFFF
        self.assertEqual(got_inf_clamp, H_ONE)

        # Overflow to -inf clamped to 0.0
        got_ninf_clamp = run("v_subrev_f16_e64 v0, v1, v2 clamp", H_POS_INF, H_NEG_INF) & 0xFFFF
        self.assertEqual(got_ninf_clamp, H_POS_ZERO)

    def test_vop3_omod_and_clamp(self):
        # An f16 destination ignores omod under the game's MODE (IEEE=0, f16 denormals kept) and keeps clamp:
        # measured on gfx1036 (#4542, docs/design.md "`v_fmac_f16`, `v_interp_p2/mov_f32`, and f16 output modifiers").
        # 1.5 - 1.0 = 0.5; mul:2 leaves it.
        got = run("v_subrev_f16_e64 v0, v1, v2 mul:2", H_ONE, f2h(1.5)) & 0xFFFF
        self.assertEqual(got, H_HALF)
        # 2.0 - 1.0 = 1.0; div:2 leaves it.
        got_div = run("v_subrev_f16_e64 v0, v1, v2 div:2", H_ONE, H_TWO) & 0xFFFF
        self.assertEqual(got_div, H_ONE)
        # 1.5 - 1.0 = 0.5; mul:4 is ignored, clamp keeps 0.5. 3.0 - 1.0 = 2.0 clamps to 1.0.
        got_clamp = run("v_subrev_f16_e64 v0, v1, v2 mul:4 clamp", H_ONE, f2h(1.5)) & 0xFFFF
        self.assertEqual(got_clamp, H_HALF)
        got_sat = run("v_subrev_f16_e64 v0, v1, v2 mul:4 clamp", H_ONE, f2h(3.0)) & 0xFFFF
        self.assertEqual(got_sat, H_ONE)


if __name__ == "__main__":
    unittest.main()
