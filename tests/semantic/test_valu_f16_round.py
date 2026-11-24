"""Semantic interpreter tests for RDNA2 16-bit float rounding/fraction VALU ops.

Covers:
  - v_floor_f16: floor(S0.f16)
  - v_ceil_f16:  ceil(S0.f16)
  - v_trunc_f16: trunc(S0.f16)
  - v_rndne_f16: roundToEven(S0.f16)
  - v_fract_f16: S0.f16 - floor(S0.f16)

Per docs/design.md ("16-bit results"), all five write one 16-bit half of the destination VGPR while preserving the
other 16-bit half (VOP2 _e32 writes the low half; VOP3 _e64 op_sel:[src,dst] selects source/destination halves).

All five ops are executed end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py across
finite values, subnormals, +-0.0, .5 ties, +-inf, NaNs, op_sel:[src,dst], VOP3 source/output modifiers, inline f16
constants, SGPR sources, destination aliasing, and EXEC masking.

tools/rdna2spirv.py lowers v_fract_f16 to GLSL.std.450 Fract, which tests/semantic/spirv_mini_interp.py computes
exactly. A Fract result that rounded up to 1.0 is mapped to 0x3bff (the largest f16 below 1.0); ref_fract_f16 below
models that fix-up for the end-to-end cases.
"""
from __future__ import annotations

import itertools
import math
import os
import re
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
OLD = 0xBEEF1234

# Representative binary16 encodings.
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_SUB_MIN = 0x0001          # smallest positive subnormal (~5.96e-8)
H_NEG_SUB_MIN = 0x8001      # smallest-magnitude negative subnormal
H_SUB_MAX = 0x03FF          # largest positive subnormal (~6.097e-5)
H_NEG_SUB_MAX = 0x83FF      # largest-magnitude negative subnormal
H_0P25 = 0x3400             # 0.25
H_NEG_0P25 = 0xB400         # -0.25
H_0P5 = 0x3800              # 0.5
H_NEG_0P5 = 0xB800          # -0.5
H_0P75 = 0x3A00             # 0.75
H_NEG_0P75 = 0xBA00         # -0.75
H_1P0 = 0x3C00              # 1.0
H_NEG_1P0 = 0xBC00          # -1.0
H_1P25 = 0x3D00             # 1.25
H_NEG_1P25 = 0xBD00         # -1.25
H_1P5 = 0x3E00              # 1.5
H_NEG_1P5 = 0xBE00          # -1.5
H_1P75 = 0x3F00             # 1.75
H_NEG_1P75 = 0xBF00         # -1.75
H_2P0 = 0x4000              # 2.0
H_NEG_2P0 = 0xC000          # -2.0
H_2P5 = 0x4100              # 2.5
H_NEG_2P5 = 0xC100          # -2.5
H_3P0 = 0x4200              # 3.0
H_NEG_3P0 = 0xC200          # -3.0
H_3P5 = 0x4300              # 3.5
H_NEG_3P5 = 0xC300          # -3.5
H_4P0 = 0x4400              # 4.0
H_NEG_4P0 = 0xC400          # -4.0
H_1022P5 = 0x63FD           # 1022.5 (even integer tie)
H_1023P5 = 0x63FF           # 1023.5 (odd integer tie -> 1024.0)
H_NEG_1022P5 = 0xE3FD       # -1022.5
H_NEG_1023P5 = 0xE3FF       # -1023.5
H_1024 = 0x6400             # 1024.0
H_MAX = 0x7BFF              # 65504.0
H_NEG_MAX = 0xFBFF          # -65504.0
H_POS_INF = 0x7C00          # +inf
H_NEG_INF = 0xFC00          # -inf
H_QNAN = 0x7E01             # +qNaN
H_NEG_QNAN = 0xFE01         # -qNaN
H_SNAN = 0x7C01             # +sNaN
H_NEAR_ONE = 0x3BFF         # largest f16 below 1.0 (the translator's Fract fix-up result)


def h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f2h(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return H_POS_INF if x > 0 else H_NEG_INF


def ref_integral(fn, h: int) -> int:
    x = h2f(h)
    if not math.isfinite(x):
        return h & 0xFFFF
    return f2h(math.copysign(float(fn(x)), x))


def ref_floor(h: int) -> int:
    return ref_integral(math.floor, h)


def ref_ceil(h: int) -> int:
    return ref_integral(math.ceil, h)


def ref_trunc(h: int) -> int:
    return ref_integral(math.trunc, h)


def ref_rndne(h: int) -> int:
    return ref_integral(round, h)


def ref_fract(h: int) -> int:
    x = h2f(h)
    if not math.isfinite(x):
        return H_QNAN
    return f2h(x - math.copysign(float(math.floor(x)), x))


def ref_fract_f16(h: int) -> int:
    """Oracle for end-to-end v_fract_f16: ref_fract plus the translator's 0x3c00 -> 0x3bff fix-up."""
    r = ref_fract(h)
    return H_NEAR_ONE if r == H_1P0 else r


def merge_half(old: int, half: int, dst_hi: int) -> int:
    h = half & 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


def is_f16_nan(half: int) -> bool:
    return (half & 0x7FFF) > 0x7C00


def is_f16_inf(half: int) -> bool:
    return (half & 0x7FFF) == 0x7C00


def is_f16_subnormal(half: int) -> bool:
    return (half & 0x7C00) == 0 and (half & 0x03FF) != 0


def is_f16_zero(half: int) -> bool:
    return (half & 0x7FFF) == 0


f16_to_float = h2f
float_to_f16 = f2h


def pack_f16x2(lo: int, hi: int) -> int:
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def unpack_f16x2(dword: int) -> tuple[int, int]:
    return dword & 0xFFFF, (dword >> 16) & 0xFFFF


def ref_pk_floor(dword: int) -> int:
    lo, hi = unpack_f16x2(dword)
    return pack_f16x2(ref_floor(lo), ref_floor(hi))


def ref_pk_ceil(dword: int) -> int:
    lo, hi = unpack_f16x2(dword)
    return pack_f16x2(ref_ceil(lo), ref_ceil(hi))


def ref_pk_trunc(dword: int) -> int:
    lo, hi = unpack_f16x2(dword)
    return pack_f16x2(ref_trunc(lo), ref_trunc(hi))


def ref_pk_rndne(dword: int) -> int:
    lo, hi = unpack_f16x2(dword)
    return pack_f16x2(ref_rndne(lo), ref_rndne(hi))


class SimdWavefront:
    """RDNA2 SIMD Wavefront state holding per-lane VGPRs, SGPRs, and 64-bit EXEC mask."""

    def __init__(self, wave_size: int = 64, initial_exec: int = 0xFFFFFFFFFFFFFFFF) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size: {wave_size}")
        self.wave_size = wave_size
        self.vgpr: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgpr: list[int] = [0] * 106
        self.mask_all = (1 << wave_size) - 1
        self.exec_mask: int = initial_exec & self.mask_all

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & self.mask_all

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgpr[lane][reg & 0xFF] = val & 0xFFFFFFFF

    def get_vgpr(self, lane: int, reg: int) -> int:
        return self.vgpr[lane][reg & 0xFF]

    def set_sgpr(self, reg: int, val: int) -> None:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range")
        self.sgpr[reg] = val & 0xFFFFFFFF

    def get_sgpr(self, reg: int) -> int:
        if not (0 <= reg < 106):
            raise IndexError(f"SGPR index {reg} out of range")
        return self.sgpr[reg]


class F16RoundingInterpreter:
    """Deterministic instruction interpreter for RDNA2 VALU f16 rounding and SIMD wavefront ops."""

    def __init__(self, wave_size: int = 64) -> None:
        self.wave = SimdWavefront(wave_size=wave_size)

    def parse_reg(self, token: str) -> tuple[str, int]:
        token = token.strip().lower()
        if token.startswith("v") and token[1:].isdigit():
            return "vgpr", int(token[1:])
        if token.startswith("s") and token[1:].isdigit():
            return "sgpr", int(token[1:])
        raise ValueError(f"Unrecognized register name: {token}")

    def parse_imm(self, token: str) -> int:
        token = token.strip()
        if token.startswith("0x") or token.startswith("-0x"):
            return int(token, 16) & 0xFFFFFFFF
        try:
            return int(token) & 0xFFFFFFFF
        except ValueError:
            f_val = float(token)
            return f2h(f_val)

    def read_operand_half(self, token: str, lane: int, hi: int) -> int:
        token = token.strip()
        is_neg = token.startswith("-")
        if is_neg:
            token = token[1:].strip()
        is_abs = token.startswith("|") and token.endswith("|")
        if is_abs:
            token = token[1:-1].strip()

        if token.startswith("v") and token[1:].isdigit():
            reg_val = self.wave.get_vgpr(lane, int(token[1:]))
            half = (reg_val >> 16) & 0xFFFF if hi else reg_val & 0xFFFF
        elif token.startswith("s") and token[1:].isdigit():
            reg_val = self.wave.get_sgpr(int(token[1:]))
            half = (reg_val >> 16) & 0xFFFF if hi else reg_val & 0xFFFF
        else:
            imm = self.parse_imm(token)
            half = imm & 0xFFFF

        if is_abs:
            half &= 0x7FFF
        if is_neg:
            half ^= 0x8000
        return half

    def execute_line(self, line: str) -> None:
        line = line.split(";")[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if mnemonic == "s_mov_b32":
            dst, src = [t.strip() for t in rest.split(",")]
            dst_type, dst_idx = self.parse_reg(dst)
            if dst_type == "sgpr":
                val = self.wave.get_sgpr(int(src[1:])) if (src.startswith("s") and src[1:].isdigit()) else self.parse_imm(src)
                self.wave.set_sgpr(dst_idx, val)
            return

        if mnemonic in ("s_mov_b64", "s_mov_b32_exec"):
            dst, src = [t.strip() for t in rest.split(",")]
            if dst.lower() in ("exec", "exec_lo"):
                val = int(src, 16) if (src.startswith("0x") or src.startswith("-0x")) else int(src)
                self.wave.set_exec(val)
            return

        if mnemonic == "v_mov_b32":
            dst, src = [t.strip() for t in rest.split(",")]
            dst_type, dst_idx = self.parse_reg(dst)
            for lane in range(self.wave.wave_size):
                if not self.wave.is_lane_active(lane):
                    continue
                if src.startswith("v") and src[1:].isdigit():
                    val = self.wave.get_vgpr(lane, int(src[1:]))
                elif src.startswith("s") and src[1:].isdigit():
                    val = self.wave.get_sgpr(int(src[1:]))
                else:
                    val = self.parse_imm(src)
                self.wave.set_vgpr(lane, dst_idx, val)
            return

        base_op = mnemonic.replace("_e32", "").replace("_e64", "")
        if base_op in OPS:
            _, ref_fn = OPS[base_op]
            m_opsel = re.search(r"op_sel:\[(\d+),(\d+)\]", rest, re.IGNORECASE)
            src_hi = int(m_opsel.group(1)) if m_opsel else 0
            dst_hi = int(m_opsel.group(2)) if m_opsel else 0
            has_clamp = bool(re.search(r"\bclamp\b", rest, re.IGNORECASE))
            has_mul2 = bool(re.search(r"\bmul:2\b", rest, re.IGNORECASE))
            has_mul4 = bool(re.search(r"\bmul:4\b", rest, re.IGNORECASE))
            has_div2 = bool(re.search(r"\bdiv:2\b", rest, re.IGNORECASE))

            clean = re.sub(r"(?:op_sel:\[\d+,\d+\]|clamp|mul:2|mul:4|div:2)", "", rest, flags=re.IGNORECASE)
            operands = [t.strip() for t in clean.split(",") if t.strip()]
            dst_tok, src_tok = operands[0], operands[1]
            dst_type, dst_idx = self.parse_reg(dst_tok)

            for lane in range(self.wave.wave_size):
                if not self.wave.is_lane_active(lane):
                    continue
                src_half = self.read_operand_half(src_tok, lane, src_hi)
                res_half = ref_fn(src_half)
                if has_mul2 or has_mul4 or has_div2 or has_clamp:
                    val = h2f(res_half)
                    if has_mul2:
                        val *= 2.0
                    elif has_mul4:
                        val *= 4.0
                    elif has_div2:
                        val *= 0.5
                    if has_clamp:
                        val = max(0.0, min(1.0, val)) if not math.isnan(val) else 0.0
                    res_half = f2h(val)

                old_val = self.wave.get_vgpr(lane, dst_idx)
                new_val = merge_half(old_val, res_half, dst_hi)
                self.wave.set_vgpr(lane, dst_idx, new_val)
            return

        raise ValueError(f"Unsupported mnemonic: {mnemonic}")

    def execute_program(self, lines: list[str]) -> None:
        for line in lines:
            self.execute_line(line)


OPS = {
    "v_floor_f16": ("Floor", ref_floor),
    "v_ceil_f16": ("Ceil", ref_ceil),
    "v_trunc_f16": ("Trunc", ref_trunc),
    "v_rndne_f16": ("RoundEven", ref_rndne),
}

_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_f16_round.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f16_round.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_valu_f16_round\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def run_ps(lines: list[str]) -> int:
    return si.run_fragment(translate_ps(lines))["%out_mrt0"][0]


def run_op(line: str, v1: int, old: int = OLD, s4: int = 0) -> int:
    lines = [f"s_mov_b32 s4, {s4:#x}", f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v0, {old:#x}", line]
    return run_ps(lines)


class ReferenceOracleTests(unittest.TestCase):
    def test_integral_and_fract_edge_values(self):
        self.assertEqual(ref_floor(H_NEG_0P25), H_NEG_1P0)
        self.assertEqual(ref_floor(H_NEG_ZERO), H_NEG_ZERO)
        self.assertEqual(ref_ceil(H_NEG_0P25), H_NEG_ZERO)
        self.assertEqual(ref_ceil(H_0P25), H_1P0)
        self.assertEqual(ref_trunc(H_NEG_0P75), H_NEG_ZERO)
        self.assertEqual(ref_trunc(H_1P75), H_1P0)
        self.assertEqual(ref_rndne(H_0P5), H_POS_ZERO)
        self.assertEqual(ref_rndne(H_NEG_0P5), H_NEG_ZERO)
        self.assertEqual(ref_rndne(H_1P5), H_2P0)
        self.assertEqual(ref_rndne(H_2P5), H_2P0)
        self.assertEqual(ref_rndne(H_1022P5), f2h(1022.0))
        self.assertEqual(ref_rndne(H_1023P5), H_1024)
        self.assertEqual(ref_fract(H_1P25), H_0P25)
        self.assertEqual(ref_fract(H_NEG_0P25), H_0P75)
        self.assertEqual(ref_fract(H_NEG_1P75), H_0P25)
        self.assertEqual(ref_fract(H_NEG_ZERO), H_POS_ZERO)
        self.assertTrue(is_f16_nan(ref_fract(H_POS_INF)))
        self.assertEqual(merge_half(OLD, H_1P0, 0), 0xBEEF3C00)
        self.assertEqual(merge_half(OLD, H_1P0, 1), 0x3C001234)

    def test_exact_integers_identity(self):
        """Exact integer representations in binary16 are unchanged by all 4 rounding modes."""
        for i in range(-100, 101):
            h = f2h(float(i))
            self.assertEqual(ref_floor(h), h, f"floor({i}) failed")
            self.assertEqual(ref_ceil(h), h, f"ceil({i}) failed")
            self.assertEqual(ref_trunc(h), h, f"trunc({i}) failed")
            self.assertEqual(ref_rndne(h), h, f"rndne({i}) failed")

    def test_exponent_ge_10_no_fractional_bits(self):
        """Values with magnitude >= 2^10 = 1024 have no fractional bits in binary16 mantissa.

        In IEEE 754 binary16 (5 exponent bits, 10 mantissa bits), when unbiased exponent >= 10,
        the unit in the last place (ulp) is >= 1.0. Therefore, all representable binary16 numbers
        with |x| >= 1024 are already exact integers and rounding is guaranteed to be an identity.
        """
        large_magnitudes = [
            1024.0, 1025.0, 1026.0, 1200.0, 1536.0, 2048.0, 4096.0,
            8192.0, 16384.0, 32768.0, 65504.0,
        ]
        for val in large_magnitudes:
            for sign in (1.0, -1.0):
                x = sign * val
                h = f2h(x)
                self.assertEqual(ref_floor(h), h, f"floor({x}) failed")
                self.assertEqual(ref_ceil(h), h, f"ceil({x}) failed")
                self.assertEqual(ref_trunc(h), h, f"trunc({x}) failed")
                self.assertEqual(ref_rndne(h), h, f"rndne({x}) failed")

    def test_signed_zeros_round_to_integral_rules(self):
        """IEEE 754-2008 / 2019 roundToIntegral rules for negative fractional numbers and signed zeros.

        Strict rules:
          - floor(-0.5) = -1.0, floor(-0.0) = -0.0, floor(+0.0) = +0.0
          - ceil(-0.5)  = -0.0, ceil(-0.0)  = -0.0, ceil(+0.0)  = +0.0
          - trunc(-0.5) = -0.0, trunc(-0.0) = -0.0, trunc(+0.0) = +0.0
          - rndne(-0.4) = -0.0, rndne(-0.5) = -0.0
        """
        self.assertEqual(ref_floor(H_NEG_0P5), H_NEG_1P0)
        self.assertEqual(ref_ceil(H_NEG_0P5), H_NEG_ZERO)
        self.assertEqual(ref_trunc(H_NEG_0P5), H_NEG_ZERO)
        self.assertEqual(ref_rndne(f2h(-0.4)), H_NEG_ZERO)
        self.assertEqual(ref_rndne(H_NEG_0P5), H_NEG_ZERO)

        # Signed zero inputs produce identical signed zero outputs
        for h_zero in (H_POS_ZERO, H_NEG_ZERO):
            self.assertEqual(ref_floor(h_zero), h_zero)
            self.assertEqual(ref_ceil(h_zero), h_zero)
            self.assertEqual(ref_trunc(h_zero), h_zero)
            self.assertEqual(ref_rndne(h_zero), h_zero)

    def test_tie_breaking_to_even_halfway(self):
        """Halfway cases (n + 0.5) must tie-break to the nearest even integer."""
        cases = [
            (H_0P5, H_POS_ZERO),       # 0.5 -> 0.0 (0 is even)
            (H_1P5, H_2P0),            # 1.5 -> 2.0 (2 is even)
            (H_2P5, H_2P0),            # 2.5 -> 2.0 (2 is even)
            (H_3P5, H_4P0),            # 3.5 -> 4.0 (4 is even)
            (f2h(4.5), H_4P0),         # 4.5 -> 4.0
            (f2h(5.5), f2h(6.0)),      # 5.5 -> 6.0
            (H_NEG_0P5, H_NEG_ZERO),   # -0.5 -> -0.0 (-0 is even)
            (H_NEG_1P5, H_NEG_2P0),    # -1.5 -> -2.0
            (H_NEG_2P5, H_NEG_2P0),    # -2.5 -> -2.0
            (H_NEG_3P5, H_NEG_4P0),    # -3.5 -> -4.0
            (f2h(-4.5), H_NEG_4P0),    # -4.5 -> -4.0
            (f2h(-5.5), f2h(-6.0)),    # -5.5 -> -6.0
            (H_1022P5, f2h(1022.0)),   # 1022.5 -> 1022.0 (even)
            (H_1023P5, H_1024),        # 1023.5 -> 1024.0 (odd -> even)
            (H_NEG_1022P5, f2h(-1022.0)),
            (H_NEG_1023P5, f2h(-1024.0)),
        ]
        for inp, expected in cases:
            with self.subTest(inp=hex(inp), expected=hex(expected)):
                self.assertEqual(ref_rndne(inp), expected)

    def test_subnormals_and_special_values(self):
        """Subnormals (< 2^-14) and infinities/NaNs handling in reference oracle."""
        for sub in (H_SUB_MIN, H_SUB_MAX):
            self.assertEqual(ref_floor(sub), H_POS_ZERO)
            self.assertEqual(ref_ceil(sub), H_1P0)
            self.assertEqual(ref_trunc(sub), H_POS_ZERO)
            self.assertEqual(ref_rndne(sub), H_POS_ZERO)

        for neg_sub in (H_NEG_SUB_MIN, H_NEG_SUB_MAX):
            self.assertEqual(ref_floor(neg_sub), H_NEG_1P0)
            self.assertEqual(ref_ceil(neg_sub), H_NEG_ZERO)
            self.assertEqual(ref_trunc(neg_sub), H_NEG_ZERO)
            self.assertEqual(ref_rndne(neg_sub), H_NEG_ZERO)

        for inf in (H_POS_INF, H_NEG_INF):
            self.assertEqual(ref_floor(inf), inf)
            self.assertEqual(ref_ceil(inf), inf)
            self.assertEqual(ref_trunc(inf), inf)
            self.assertEqual(ref_rndne(inf), inf)

        for nan in (H_QNAN, H_NEG_QNAN, H_SNAN):
            self.assertTrue(is_f16_nan(ref_floor(nan)))
            self.assertTrue(is_f16_nan(ref_ceil(nan)))
            self.assertTrue(is_f16_nan(ref_trunc(nan)))
            self.assertTrue(is_f16_nan(ref_rndne(nan)))


class F16RoundingExecutionTests(unittest.TestCase):
    def test_edge_values_across_e32_e64_and_bare(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_0P25, H_NEG_0P25, H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_1P0, H_NEG_1P0, H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P0, H_NEG_3P0, H_3P5, H_NEG_3P5,
            H_1022P5, H_1023P5, H_NEG_1022P5, H_NEG_1023P5,
            H_MAX, H_NEG_MAX, H_POS_INF, H_NEG_INF,
        ]
        for op, (_, ref) in OPS.items():
            for suffix in ("_e32", "_e64", ""):
                for h in samples:
                    v1 = 0x55550000 | h
                    with self.subTest(op=op + suffix, h=hex(h)):
                        got = run_op(f"{op}{suffix} v0, v1", v1)
                        self.assertEqual(got, merge_half(OLD, ref(h), 0))

    def test_signed_zero_and_subnormal_signs_preserved(self):
        # IEEE-754 roundToIntegral keeps -0.0 when rounding values in (-1.0, -0.0] to zero.
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_NEG_ZERO), merge_half(OLD, H_NEG_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_POS_ZERO), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_SUB_MIN), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, v1", 0xABCD0000 | H_NEG_SUB_MIN), merge_half(OLD, H_NEG_1P0, 0))
        for op in ("v_ceil_f16", "v_trunc_f16", "v_rndne_f16"):
            for h in (H_NEG_ZERO, H_NEG_SUB_MIN, H_NEG_SUB_MAX, H_NEG_0P25):
                with self.subTest(op=op, h=hex(h)):
                    self.assertEqual(run_op(f"{op}_e32 v0, v1", 0xABCD0000 | h), merge_half(OLD, H_NEG_ZERO, 0))

    def test_nan_preserves_nan_and_other_half(self):
        for op in OPS:
            for h in (H_QNAN, H_NEG_QNAN, H_SNAN):
                with self.subTest(op=op, h=hex(h)):
                    got = run_op(f"{op}_e32 v0, v1", 0xCAFE0000 | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(is_f16_nan(got & 0xFFFF), f"expected NaN in low half, got {got:#x}")

    def test_op_sel_all_combinations(self):
        # v1 = (hi: -2.5, lo: 3.5) -> clear, distinct results across every op and half.
        v1 = (H_NEG_2P5 << 16) | H_3P5
        for op, (_, ref) in OPS.items():
            for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
                h_in = H_NEG_2P5 if src_hi else H_3P5
                line = f"{op}_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
                with self.subTest(line=line):
                    self.assertEqual(run_op(line, v1), merge_half(OLD, ref(h_in), dst_hi))

    def test_vop3_source_and_output_modifiers(self):
        v1 = (H_NEG_1P5 << 16) | H_2P5
        # Source modifiers apply to the selected half before the rounding op.
        self.assertEqual(run_op("v_floor_f16_e64 v0, -v1 op_sel:[0,0]", v1), merge_half(OLD, f2h(-3.0), 0))
        self.assertEqual(run_op("v_ceil_f16_e64 v0, |v1| op_sel:[1,1]", v1), merge_half(OLD, H_2P0, 1))
        self.assertEqual(run_op("v_rndne_f16_e64 v0, -|v1| op_sel:[1,0]", v1), merge_half(OLD, H_NEG_2P0, 0))

        # Output modifiers: on RDNA2 with denorms kept, omod is ignored on f16; clamp applies to [0.0, 1.0].
        # trunc(2.5) = 2.0; mul:2/mul:4/div:2 ignored; clamp -> 1.0.
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:2", v1), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:4", v1), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 div:2", v1), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 mul:2 clamp", v1), merge_half(OLD, H_1P0, 0))
        # floor(-1.5) = -2.0 -> clamp -> 0.0 (preserving low half when dst_hi=1).
        self.assertEqual(run_op("v_floor_f16_e64 v0, v1 op_sel:[1,1] clamp", v1), merge_half(OLD, H_POS_ZERO, 1))
        # rndne(0.5) = 0.0 -> div:2 = 0.0; trunc(1.25) = 1.0 (div:2 ignored).
        self.assertEqual(run_op("v_trunc_f16_e64 v0, v1 div:2", H_1P25), merge_half(OLD, H_1P0, 0))

    def test_inline_constants_sgpr_and_aliasing(self):
        # Inline float constants populate the low half as f16 (docs/design.md).
        self.assertEqual(run_op("v_ceil_f16_e32 v0, 0.5", 0), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_floor_f16_e32 v0, -0.5", 0), merge_half(OLD, H_NEG_1P0, 0))
        self.assertEqual(run_op("v_rndne_f16_e64 v0, -1.0", 0), merge_half(OLD, H_NEG_1P0, 0))

        # SGPR source (reading high half via op_sel).
        s4 = (H_NEG_1P25 << 16) | H_0P75
        self.assertEqual(run_op("v_floor_f16_e32 v0, s4", 0, s4=s4), merge_half(OLD, H_POS_ZERO, 0))
        self.assertEqual(run_op("v_floor_f16_e64 v0, s4 op_sel:[1,1]", 0, s4=s4), merge_half(OLD, H_NEG_2P0, 1))

        # In-place aliasing: dst == src reading high half and writing low half keeps the original high half.
        v0_init = (H_3P5 << 16) | H_0P25
        self.assertEqual(run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_rndne_f16_e64 v0, v0 op_sel:[1,0]"]),
                         (H_3P5 << 16) | H_4P0)
        self.assertEqual(run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_trunc_f16_e32 v0, v0"]),
                         (H_3P5 << 16) | H_POS_ZERO)

    def test_exec_masking(self):
        # Inactive single lane preserves pre-existing v0.
        got = run_ps([
            f"v_mov_b32 v1, {H_1P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_ceil_f16_e32 v0, v1",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, OLD)

        # Disabled ops leave both halves untouched; re-enabling EXEC runs the subsequent op.
        got_toggled = run_ps([
            f"v_mov_b32 v1, {(H_2P5 << 16) | H_NEG_1P25:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
            "v_ceil_f16_e64 v0, v1 op_sel:[1,1]",
            "s_mov_b32 exec_lo, 1",
            "v_trunc_f16_e64 v0, v1 op_sel:[1,0]",
        ])
        self.assertEqual(got_toggled, merge_half(OLD, H_2P0, 0))

    def test_negative_control_sabotaged_spirv(self):
        spv = translate_ps([f"v_mov_b32 v1, {H_2P5:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_floor_f16_e32 v0, v1"])
        sabotaged, n = re.subn(r"\bFloor\b", "Ceil", spv, count=1)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_2P0, 0))
        self.assertEqual(got, merge_half(OLD, H_3P0, 0))


class PackedF16SimdTests(unittest.TestCase):
    """Packed half-precision floating-point rounding and multi-lane SIMD tests.

    Models:
      - Packed pairs: v_pk_floor_f16 and v_pk_ceil_f16 behavior processing two 16-bit
        halfwords packed into a 32-bit dword.
      - SIMD Wavefront execution across 32 and 64 lanes.
      - Inactive lane preservation under EXEC execution masks.
      - End-to-end RDNA2 packed SIMD emulation via op_sel:[0,0] and op_sel:[1,1].
      - Negative rejection test for unsupported packed pseudo-opcodes in translator.
    """

    def test_packed_floor_and_ceil_reference_oracle(self):
        """Packed half-precision oracle validates low and high halves independently."""
        pairs = [
            (H_1P5, H_2P5),
            (H_NEG_0P5, H_0P5),
            (H_SUB_MIN, H_NEG_SUB_MIN),
            (H_1022P5, H_1023P5),
            (H_POS_ZERO, H_NEG_ZERO),
            (H_POS_INF, H_NEG_INF),
            (H_0P25, H_NEG_0P75),
        ]
        for lo, hi in pairs:
            dword = pack_f16x2(lo, hi)
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                exp_floor = pack_f16x2(ref_floor(lo), ref_floor(hi))
                exp_ceil = pack_f16x2(ref_ceil(lo), ref_ceil(hi))
                exp_trunc = pack_f16x2(ref_trunc(lo), ref_trunc(hi))
                exp_rndne = pack_f16x2(ref_rndne(lo), ref_rndne(hi))
                self.assertEqual(ref_pk_floor(dword), exp_floor)
                self.assertEqual(ref_pk_ceil(dword), exp_ceil)
                self.assertEqual(ref_pk_trunc(dword), exp_trunc)
                self.assertEqual(ref_pk_rndne(dword), exp_rndne)

    def test_simd_wavefront_uniform_packed_execution(self):
        """Simulate packed floor and ceil across all 64 lanes of a SIMD wavefront."""
        interp = F16RoundingInterpreter(wave_size=64)
        v1 = pack_f16x2(H_1P5, H_NEG_2P5)
        # Low half floor(1.5)=1.0, high half floor(-2.5)=-3.0
        interp.execute_program([
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
            "v_floor_f16_e64 v0, v1 op_sel:[1,1]",
        ])
        expected_floor = ref_pk_floor(v1)
        for lane in range(64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), expected_floor, f"Lane {lane} floor mismatch")

        # Ceil sequence: ceil(1.5)=2.0, ceil(-2.5)=-2.0
        interp.execute_program([
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_ceil_f16_e64 v0, v1 op_sel:[0,0]",
            "v_ceil_f16_e64 v0, v1 op_sel:[1,1]",
        ])
        expected_ceil = ref_pk_ceil(v1)
        for lane in range(64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), expected_ceil, f"Lane {lane} ceil mismatch")

    def test_simd_wavefront_divergent_lanes(self):
        """Each lane in a 64-lane wave processes distinct packed half-precision pairs."""
        interp = F16RoundingInterpreter(wave_size=64)
        per_lane_inputs = []
        for lane in range(64):
            lo_val = float(lane % 8) + 0.5
            hi_val = -(float(lane // 8) + 0.5)
            h_lo = f2h(lo_val)
            h_hi = f2h(hi_val)
            dword = pack_f16x2(h_lo, h_hi)
            per_lane_inputs.append(dword)
            interp.wave.set_vgpr(lane, 1, dword)
            interp.wave.set_vgpr(lane, 0, OLD)

        interp.execute_program([
            "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
            "v_floor_f16_e64 v0, v1 op_sel:[1,1]",
        ])
        for lane in range(64):
            expected = ref_pk_floor(per_lane_inputs[lane])
            self.assertEqual(interp.wave.get_vgpr(lane, 0), expected, f"Divergent lane {lane} floor mismatch")

    def test_simd_wavefront_exec_mask_inactive_lanes_preserved(self):
        """Inactive lanes under EXEC mask must strictly retain their previous values."""
        interp = F16RoundingInterpreter(wave_size=64)
        v1 = pack_f16x2(H_1P5, H_NEG_1P5)
        # Enable only lower 32 lanes (exec = 0x00000000FFFFFFFF)
        for lane in range(64):
            interp.wave.set_vgpr(lane, 0, OLD)
            interp.wave.set_vgpr(lane, 1, v1)

        interp.execute_program([
            "s_mov_b64 exec, 0x00000000FFFFFFFF",
            "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
            "v_floor_f16_e64 v0, v1 op_sel:[1,1]",
        ])

        expected_floor = ref_pk_floor(v1)
        for lane in range(32):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), expected_floor, f"Active lane {lane} failed")
        for lane in range(32, 64):
            self.assertEqual(interp.wave.get_vgpr(lane, 0), OLD, f"Inactive lane {lane} was modified")

    def test_end_to_end_rdna2_simd_packed_sequence(self):
        """End-to-end translation and execution of RDNA2 packed f16 floor/ceil sequences."""
        samples = [
            pack_f16x2(H_1P5, H_2P5),
            pack_f16x2(H_NEG_0P5, H_0P75),
            pack_f16x2(H_1022P5, H_1023P5),
            pack_f16x2(H_SUB_MIN, H_NEG_SUB_MIN),
        ]
        for dword in samples:
            with self.subTest(dword=hex(dword)):
                got_floor = run_ps([
                    f"v_mov_b32 v1, {dword:#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    "v_floor_f16_e64 v0, v1 op_sel:[0,0]",
                    "v_floor_f16_e64 v0, v1 op_sel:[1,1]",
                ])
                self.assertEqual(got_floor, ref_pk_floor(dword))

                got_ceil = run_ps([
                    f"v_mov_b32 v1, {dword:#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    "v_ceil_f16_e64 v0, v1 op_sel:[0,0]",
                    "v_ceil_f16_e64 v0, v1 op_sel:[1,1]",
                ])
                self.assertEqual(got_ceil, ref_pk_ceil(dword))

    def test_unsupported_packed_pseudo_opcodes_rejected(self):
        """In RDNA2 ISA, packed rounding is achieved via scalar half ops; packed mnemonics are unsupported."""
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_ps(["v_pk_floor_f16 v0, v1"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_ps(["v_pk_ceil_f16 v0, v1"])


class FractF16TranslationTests(unittest.TestCase):
    """v_fract_f16 lowers to GLSL.std.450 Fract (spirv_mini_interp.py computes it exactly, rounded once)."""

    def test_fract_f16_emits_glsl_fract_op_sel_mods_and_half_keep(self):
        spv = translate_ps([
            f"v_mov_b32 v1, {(H_NEG_1P25 << 16) | H_2P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "v_fract_f16_e64 v0, v1 op_sel:[1,1] clamp",
        ])
        self.assertIn("UnpackHalf2x16", spv)
        self.assertRegex(spv, r"OpExtInst %float %glsl Fract %\w+")
        self.assertRegex(spv, r"OpExtInst %float %glsl FClamp %\w+ %f0 %f1")
        # op_sel:[1,1]: fract of the high half (-1.25 -> 0.75 = 0x3a00) into the high half; the low half is kept.
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], (0x3A00 << 16) | (OLD & 0xFFFF))

    def test_edge_values_across_e32_e64_and_bare(self):
        samples = [
            H_POS_ZERO, H_NEG_ZERO,
            H_SUB_MIN, H_NEG_SUB_MIN, H_SUB_MAX, H_NEG_SUB_MAX,
            H_0P25, H_NEG_0P25, H_0P5, H_NEG_0P5, H_0P75, H_NEG_0P75,
            H_1P0, H_NEG_1P0, H_1P25, H_NEG_1P25, H_1P5, H_NEG_1P5, H_1P75, H_NEG_1P75,
            H_2P0, H_NEG_2P0, H_2P5, H_NEG_2P5, H_3P5, H_NEG_3P5,
            H_1022P5, H_1023P5, H_NEG_1022P5, H_NEG_1023P5,
            H_MAX, H_NEG_MAX,
        ]
        for suffix in ("_e32", "_e64", ""):
            for h in samples:
                v1 = 0x55550000 | h
                with self.subTest(op="v_fract_f16" + suffix, h=hex(h)):
                    got = run_op(f"v_fract_f16{suffix} v0, v1", v1)
                    self.assertEqual(got, merge_half(OLD, ref_fract_f16(h), 0))

    def test_inf_and_nan_give_nan_and_keep_other_half(self):
        for h in (H_POS_INF, H_NEG_INF, H_QNAN, H_NEG_QNAN, H_SNAN):
            for suffix in ("_e32", "_e64"):
                with self.subTest(h=hex(h), suffix=suffix):
                    got = run_op(f"v_fract_f16{suffix} v0, v1", 0xCAFE0000 | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(is_f16_nan(got & 0xFFFF), f"expected NaN in low half, got {got:#x}")

    def test_op_sel_all_combinations(self):
        # v1 = (hi: -1.25, lo: 2.5) -> distinct fractions 0.75 and 0.5 across every src/dst half.
        v1 = (H_NEG_1P25 << 16) | H_2P5
        for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
            h_in = H_NEG_1P25 if src_hi else H_2P5
            line = f"v_fract_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
            with self.subTest(line=line):
                self.assertEqual(run_op(line, v1), merge_half(OLD, ref_fract_f16(h_in), dst_hi))

    def test_vop3_source_and_output_modifiers(self):
        v1 = (H_NEG_1P25 << 16) | H_2P5
        # Source modifiers apply to the selected half before Fract.
        self.assertEqual(run_op("v_fract_f16_e64 v0, -v1 op_sel:[0,0]", v1), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, |v1| op_sel:[1,1]", v1), merge_half(OLD, H_0P25, 1))
        self.assertEqual(run_op("v_fract_f16_e64 v0, -|v1| op_sel:[1,0]", v1), merge_half(OLD, H_0P75, 0))

        # On RDNA2 with denorms kept, omod is ignored on f16 results; clamp applies.
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:2", v1), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:4", v1), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 div:2", v1), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 mul:2 clamp", v1), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, v1 op_sel:[1,1] clamp", v1), merge_half(OLD, H_0P75, 1))

    def test_inline_constants_sgpr_and_aliasing(self):
        self.assertEqual(run_op("v_fract_f16_e32 v0, 0.5", 0), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e32 v0, -0.5", 0), merge_half(OLD, H_0P5, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, -1.0", 0), merge_half(OLD, H_POS_ZERO, 0))

        s4 = (H_NEG_1P25 << 16) | H_0P25
        self.assertEqual(run_op("v_fract_f16_e32 v0, s4", 0, s4=s4), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_fract_f16_e64 v0, s4 op_sel:[1,1]", 0, s4=s4), merge_half(OLD, H_0P75, 1))

        # In-place aliasing: dst == src reading the high half keeps the original high half.
        v0_init = (H_3P5 << 16) | H_0P25
        self.assertEqual(run_ps([f"v_mov_b32 v0, {v0_init:#x}", "v_fract_f16_e64 v0, v0 op_sel:[1,0]"]),
                         (H_3P5 << 16) | H_0P5)

    def test_exec_masking(self):
        # Inactive single lane preserves pre-existing v0.
        got = run_ps([
            f"v_mov_b32 v1, {H_1P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f16_e32 v0, v1",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, OLD)

        # Disabled ops leave both halves untouched; re-enabling EXEC runs the subsequent op.
        got_toggled = run_ps([
            f"v_mov_b32 v1, {(H_NEG_1P25 << 16) | H_2P5:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_fract_f16_e64 v0, v1 op_sel:[0,0]",
            "v_fract_f16_e64 v0, v1 op_sel:[1,1]",
            "s_mov_b32 exec_lo, 1",
            "v_fract_f16_e64 v0, v1 op_sel:[1,0]",
        ])
        self.assertEqual(got_toggled, merge_half(OLD, H_0P75, 0))


class TieBreakingSignedZerosAndNegativeControlTests(unittest.TestCase):
    """End-to-end and negative control tests for tie-breaking to even, signed zeros, and mutants."""

    def test_e2e_rndne_halfway_tie_breaking_even_matrix(self):
        """End-to-end v_rndne_f16 execution strictly enforces round-to-nearest-even on halfway ties."""
        ties = [
            (H_0P5, H_POS_ZERO),       # 0.5 -> 0.0 (0 is even)
            (H_1P5, H_2P0),            # 1.5 -> 2.0 (2 is even)
            (H_2P5, H_2P0),            # 2.5 -> 2.0 (2 is even)
            (H_3P5, H_4P0),            # 3.5 -> 4.0 (4 is even)
            (f2h(4.5), H_4P0),         # 4.5 -> 4.0 (4 is even)
            (f2h(5.5), f2h(6.0)),      # 5.5 -> 6.0 (6 is even)
            (H_NEG_0P5, H_NEG_ZERO),   # -0.5 -> -0.0 (-0 is even, preserving sign)
            (H_NEG_1P5, H_NEG_2P0),    # -1.5 -> -2.0
            (H_NEG_2P5, H_NEG_2P0),    # -2.5 -> -2.0
            (H_NEG_3P5, H_NEG_4P0),    # -3.5 -> -4.0
            (f2h(-4.5), H_NEG_4P0),    # -4.5 -> -4.0
            (f2h(-5.5), f2h(-6.0)),    # -5.5 -> -6.0
            (H_1022P5, f2h(1022.0)),   # 1022.5 -> 1022.0
            (H_1023P5, H_1024),        # 1023.5 -> 1024.0 (odd -> even)
            (H_NEG_1022P5, f2h(-1022.0)),
            (H_NEG_1023P5, f2h(-1024.0)),
        ]
        for inp, expected in ties:
            with self.subTest(inp=hex(inp), expected=hex(expected)):
                got = run_op("v_rndne_f16_e32 v0, v1", 0x12340000 | inp)
                self.assertEqual(got, merge_half(OLD, expected, 0))

    def test_e2e_rndne_opsel_high_half_destination(self):
        """v_rndne_f16_e64 with op_sel:[0,1] writes to the high half while preserving the low half."""
        v1 = (0x9999 << 16) | H_2P5  # low half is 2.5 (ties to 2.0)
        got = run_op("v_rndne_f16_e64 v0, v1 op_sel:[0,1]", v1)
        self.assertEqual(got, merge_half(OLD, H_2P0, 1))

        v1_hi = (H_1P5 << 16) | 0x9999  # high half is 1.5 (ties to 2.0)
        got_hi = run_op("v_rndne_f16_e64 v0, v1 op_sel:[1,1]", v1_hi)
        self.assertEqual(got_hi, merge_half(OLD, H_2P0, 1))

    def test_e2e_signed_zeros_exact_bit_patterns(self):
        """Strict IEEE 754-2008 / 2019 roundToIntegral signed zero bit preservation (0x8000 vs 0x0000)."""
        cases = [
            ("v_floor_f16_e32", H_NEG_0P5, H_NEG_1P0),
            ("v_ceil_f16_e32",  H_NEG_0P5, H_NEG_ZERO),
            ("v_trunc_f16_e32", H_NEG_0P5, H_NEG_ZERO),
            ("v_rndne_f16_e32", f2h(-0.4),  H_NEG_ZERO),
            ("v_rndne_f16_e32", H_NEG_0P5, H_NEG_ZERO),
            ("v_floor_f16_e32", H_NEG_ZERO, H_NEG_ZERO),
            ("v_ceil_f16_e32",  H_NEG_ZERO, H_NEG_ZERO),
            ("v_trunc_f16_e32", H_NEG_ZERO, H_NEG_ZERO),
            ("v_rndne_f16_e32", H_NEG_ZERO, H_NEG_ZERO),
            ("v_floor_f16_e32", H_POS_ZERO, H_POS_ZERO),
            ("v_ceil_f16_e32",  H_POS_ZERO, H_POS_ZERO),
            ("v_trunc_f16_e32", H_POS_ZERO, H_POS_ZERO),
            ("v_rndne_f16_e32", H_POS_ZERO, H_POS_ZERO),
        ]
        for op, inp, expected in cases:
            with self.subTest(op=op, inp=hex(inp), expected=hex(expected)):
                got = run_op(f"{op} v0, v1", 0xFACE0000 | inp)
                self.assertEqual(got, merge_half(OLD, expected, 0))
                # Verify exact bitwise sign bit 0x8000
                got_half = got & 0xFFFF
                self.assertEqual(got_half & 0x8000, expected & 0x8000, f"Sign bit mismatch for {op} on {inp:#x}")

    def test_negative_control_roundeven_sabotaged_to_trunc(self):
        """Sabotaging RoundEven to Trunc in SPIR-V fails halfway tie-breaking."""
        spv = translate_ps([f"v_mov_b32 v1, {H_1P5:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_rndne_f16_e32 v0, v1"])
        sabotaged, n = re.subn(r"\bRoundEven\b", "Trunc", spv, count=1)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_2P0, 0))
        self.assertEqual(got, merge_half(OLD, H_1P0, 0))

    def test_negative_control_roundeven_sabotaged_to_ceil(self):
        """Sabotaging RoundEven to Ceil in SPIR-V fails halfway tie-breaking on odd rounds."""
        spv = translate_ps([f"v_mov_b32 v1, {H_2P5:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_rndne_f16_e32 v0, v1"])
        sabotaged, n = re.subn(r"\bRoundEven\b", "Ceil", spv, count=1)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_2P0, 0))
        self.assertEqual(got, merge_half(OLD, H_3P0, 0))

    def test_negative_control_ceil_sabotaged_to_floor(self):
        """Sabotaging Ceil to Floor in SPIR-V fails ceiling assertion."""
        spv = translate_ps([f"v_mov_b32 v1, {H_1P25:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_ceil_f16_e32 v0, v1"])
        sabotaged, n = re.subn(r"\bCeil\b", "Floor", spv, count=1)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_2P0, 0))
        self.assertEqual(got, merge_half(OLD, H_1P0, 0))

    def test_negative_control_clamp_omitted(self):
        """Omitting FClamp in SPIR-V produces out-of-bounds unclamped float."""
        spv = translate_ps([f"v_mov_b32 v1, {H_2P5:#x}", f"v_mov_b32 v0, {OLD:#x}", "v_trunc_f16_e64 v0, v1 mul:2 clamp"])
        sabotaged, n = re.subn(r"OpExtInst %float %glsl FClamp (%\w+) %f0 %f1", r"OpCopyObject %float \1", spv)
        self.assertEqual(n, 1)
        got = si.run_fragment(sabotaged)["%out_mrt0"][0]
        self.assertNotEqual(got, merge_half(OLD, H_1P0, 0))
        self.assertEqual(got, merge_half(OLD, H_2P0, 0))

    def test_negative_control_dropped_signed_zero_detected(self):
        """A mutant that strips sign bits from -0.0 results is detected by bitwise check."""
        for op, inp in [("v_ceil_f16_e32", H_NEG_0P5), ("v_trunc_f16_e32", H_NEG_0P5), ("v_rndne_f16_e32", f2h(-0.4))]:
            got = run_op(f"{op} v0, v1", 0x11110000 | inp)
            mutated = got & 0xFFFF7FFF
            self.assertNotEqual(mutated, got, f"Sign bit mutation must differ for {op} on {inp:#x}")


if __name__ == "__main__":
    unittest.main()
