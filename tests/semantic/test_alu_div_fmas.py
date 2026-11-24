"""Semantic interpreter and SPIR-V validation tests for RDNA2 v_div_fmas_f32 and v_div_fmas_f64.

Measured natively on RDNA2 (AMD Radeon Graphics, gfx1036, Windows driver 32.0.21042.62,
wave32 and wave64, game MODE ieee=0 / dx10_clamp=1 / denorm32=0 / denorm16_64=3):
  VCC == 0: fma(S0, S1, S2)
  VCC == 1: (exp(S2) > bias ? 2^+S : 2^-S) * fma_exact(S0, S1, S2)
where (bias, S) = (127, 64) for f32 and (1023, 128) for f64 (exp(S2) > bias <=> |S2| >= 2.0).

For f32 under denorm32=0, subnormal inputs and intermediate products are kept, while the
scaled FMA output flushes subnormals to signed zero before (and after) output modifiers.
For f64 under denorm16_64=3, subnormal inputs and outputs are preserved with single rounding.
Both accept VOP3 input (abs/neg) and output (mul:2, mul:4, div:2, clamp) modifiers.
"""
from fractions import Fraction
import math
import os
import re
import shutil
import struct
import subprocess
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
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def dwords(x: float):
    bits = d2u(x)
    return bits & 0xFFFFFFFF, bits >> 32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_div_fmas.s")
    with open(os.path.join(_TMP_DIR.name, "ps_div_fmas.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_div_fmas\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spv_text: str):
    # Always assemble with the in-repo SPIR-V assembler so malformed opcodes/IDs fail everywhere.
    binary = spirv_asm.assemble(spv_text)
    assert len(binary) >= 20 and binary[:4] == b"\x03\x02\x23\x07"
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_f32_u(ux: int, uy: int, uz: int, vcc: bool, insn: str = "v_div_fmas_f32 v0, v1, v2, v3") -> int:
    vcc_set = "v_cmp_eq_u32_e32 vcc_lo, 1, 1" if vcc else "v_cmp_eq_u32_e32 vcc_lo, 0, 1"
    spv = translate_lines([
        f"v_mov_b32 v1, {ux & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v2, {uy & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {uz & 0xFFFFFFFF:#x}",
        "v_mov_b32 v0, 0",
        vcc_set,
        insn,
    ])
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_f32(x: float, y: float, z: float, vcc: bool, insn: str = "v_div_fmas_f32 v0, v1, v2, v3") -> int:
    return run_f32_u(f2u(x), f2u(y), f2u(z), vcc, insn)


def _eval_spv_f64_mrt0(spv_text: str):
    """Straight-line SSA evaluator for translated %double/%uint/%bool fragment shaders."""
    def u64_to_d(lo, hi):
        return struct.unpack("<d", struct.pack("<II", lo & 0xFFFFFFFF, hi & 0xFFFFFFFF))[0]

    def d_to_u64(d):
        if math.isnan(d):
            return 0, 0x7FF80000
        q = struct.unpack("<Q", struct.pack("<d", d))[0]
        return q & 0xFFFFFFFF, q >> 32

    def safe_fma(a, b, c):
        # Exact IEEE-754 binary64 FMA compatible with Python 3.12 (math.fma was added in 3.13).
        if math.isnan(a) or math.isnan(b) or math.isnan(c):
            return float("nan")
        if math.isinf(a) or math.isinf(b):
            if a == 0.0 or b == 0.0:
                return float("nan")
            prod_inf = math.copysign(math.inf, math.copysign(1.0, a) * math.copysign(1.0, b))
            if math.isinf(c) and math.copysign(1.0, c) != math.copysign(1.0, prod_inf):
                return float("nan")
            return prod_inf
        if math.isinf(c):
            return c
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        if exact == 0:
            prod_neg = (math.copysign(1.0, a) * math.copysign(1.0, b)) < 0.0
            c_neg = math.copysign(1.0, c) < 0.0
            if a * b == 0.0 and c == 0.0 and prod_neg and c_neg:
                return -0.0
            return 0.0
        try:
            r = float(exact)
        except OverflowError:
            return -math.inf if exact < 0 else math.inf
        return -0.0 if (r == 0.0 and exact < 0) else r

    consts, env, mem = {}, {}, {}
    in_fn = False
    for raw in spv_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("%main = OpFunction"):
            in_fn = True
            continue
        if line == "OpFunctionEnd":
            in_fn = False
            continue
        m = re.match(r"^(%\S+)\s*=\s*(\S+)(.*)$", line)
        if not in_fn:
            if m:
                res, op, rest = m.group(1), m.group(2), m.group(3).strip().split()
                if op == "OpConstant":
                    ty, val = rest[0], rest[1]
                    if ty == "%uint":
                        consts[res] = int(val) & 0xFFFFFFFF
                    elif ty == "%int":
                        consts[res] = int(val)
                    elif ty == "%float":
                        consts[res] = float(val)
                elif op == "OpConstantTrue":
                    consts[res] = True
                elif op == "OpConstantFalse":
                    consts[res] = False
                elif op == "OpConstantNull":
                    ty = rest[0]
                    if ty == "%uint":
                        consts[res] = 0
                    elif ty == "%float":
                        consts[res] = 0.0
                    elif ty == "%bool":
                        consts[res] = False
                    else:
                        consts[res] = (0.0, 0.0, 0.0, 0.0)
                elif op == "OpVariable":
                    consts[res] = res
                    mem[res] = (0.0, 0.0, 0.0, 0.0) if "v4float" in rest[0] else 0
            continue
        if not m:
            toks = line.split()
            if toks[0] == "OpStore":
                dst = env.get(toks[1], consts.get(toks[1], toks[1]))
                mem[dst] = env[toks[2]] if toks[2] in env else consts[toks[2]]
            continue
        res, op, rest = m.group(1), m.group(2), m.group(3).strip().split()

        def get(tok):
            return env[tok] if tok in env else consts[tok]

        if op == "OpVariable":
            mem[res] = get(rest[2]) if len(rest) >= 3 else (True if res == "%active" else 0)
            env[res] = res
        elif op == "OpAccessChain":
            key = (rest[1], tuple(get(t) for t in rest[2:]))
            env[res] = key
            mem.setdefault(key, 0)
        elif op == "OpLoad":
            env[res] = mem.get(get(rest[1]), 0)
        elif op == "OpCompositeConstruct":
            env[res] = tuple(get(t) for t in rest[1:])
        elif op == "OpCompositeExtract":
            env[res] = get(rest[1])[int(rest[2])]
        elif op == "OpBitcast":
            ty, v = rest[0], get(rest[1])
            if ty == "%double":
                env[res] = u64_to_d(v[0], v[1])
            elif ty == "%v2uint_d":
                env[res] = d_to_u64(v)
            elif ty == "%float":
                env[res] = struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]
            elif ty == "%uint":
                env[res] = struct.unpack("<I", struct.pack("<f", v))[0] if isinstance(v, float) else (v & 0xFFFFFFFF)
            elif ty == "%int":
                env[res] = v - 0x100000000 if v >= 0x80000000 else v
            else:
                raise NotImplementedError(f"OpBitcast {ty}")
        elif op == "OpExtInst":
            fn, args = rest[2], [get(t) for t in rest[3:]]
            if fn == "FAbs":
                env[res] = abs(args[0])
            elif fn == "Fma":
                env[res] = safe_fma(args[0], args[1], args[2])
            elif fn == "FClamp":
                x, lo, hi = args
                env[res] = float("nan") if math.isnan(x) else max(lo, min(hi, x))
            else:
                raise NotImplementedError(f"OpExtInst {fn}")
        elif op == "OpFNegate":
            env[res] = -get(rest[1])
        elif op == "OpFMul":
            env[res] = get(rest[1]) * get(rest[2])
        elif op == "OpFAdd":
            env[res] = get(rest[1]) + get(rest[2])
        elif op == "OpFSub":
            env[res] = get(rest[1]) - get(rest[2])
        elif op == "OpIAdd":
            env[res] = (get(rest[1]) + get(rest[2])) & 0xFFFFFFFF
        elif op == "OpISub":
            env[res] = (get(rest[1]) - get(rest[2])) & 0xFFFFFFFF
        elif op == "OpBitwiseAnd":
            env[res] = (get(rest[1]) & get(rest[2])) & 0xFFFFFFFF
        elif op == "OpBitwiseOr":
            env[res] = (get(rest[1]) | get(rest[2])) & 0xFFFFFFFF
        elif op == "OpBitwiseXor":
            env[res] = (get(rest[1]) ^ get(rest[2])) & 0xFFFFFFFF
        elif op == "OpShiftRightLogical":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) >> (get(rest[2]) & 31)
        elif op == "OpShiftLeftLogical":
            env[res] = ((get(rest[1]) & 0xFFFFFFFF) << (get(rest[2]) & 31)) & 0xFFFFFFFF
        elif op == "OpIEqual":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) == (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpINotEqual":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) != (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpUGreaterThan":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) > (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpUGreaterThanEqual":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) >= (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpULessThan":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) < (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpULessThanEqual":
            env[res] = (get(rest[1]) & 0xFFFFFFFF) <= (get(rest[2]) & 0xFFFFFFFF)
        elif op == "OpFOrdGreaterThanEqual":
            a, b = get(rest[1]), get(rest[2])
            env[res] = (not math.isnan(a)) and (not math.isnan(b)) and (a >= b)
        elif op == "OpFOrdGreaterThan":
            a, b = get(rest[1]), get(rest[2])
            env[res] = (not math.isnan(a)) and (not math.isnan(b)) and (a > b)
        elif op == "OpFOrdLessThan":
            a, b = get(rest[1]), get(rest[2])
            env[res] = (not math.isnan(a)) and (not math.isnan(b)) and (a < b)
        elif op == "OpFOrdEqual":
            a, b = get(rest[1]), get(rest[2])
            env[res] = (not math.isnan(a)) and (not math.isnan(b)) and (a == b)
        elif op == "OpFOrdNotEqual":
            a, b = get(rest[1]), get(rest[2])
            env[res] = (not math.isnan(a)) and (not math.isnan(b)) and (a != b)
        elif op == "OpLogicalAnd":
            env[res] = bool(get(rest[1])) and bool(get(rest[2]))
        elif op == "OpLogicalOr":
            env[res] = bool(get(rest[1])) or bool(get(rest[2]))
        elif op == "OpLogicalNot":
            env[res] = not bool(get(rest[1]))
        elif op == "OpLogicalEqual":
            env[res] = bool(get(rest[1])) == bool(get(rest[2]))
        elif op == "OpSelect":
            env[res] = get(rest[2]) if get(rest[1]) else get(rest[3])
        elif op == "OpCopyObject":
            env[res] = get(rest[1])
        elif op in ("OpLabel", "OpFunctionParameter"):
            pass
        else:
            raise NotImplementedError(f"Unhandled SPIR-V op: {op} ({line})")
    return [struct.unpack("<I", struct.pack("<f", x))[0] for x in mem["%out_mrt0"]]


def run_f64_u(ux: int, uy: int, uz: int, vcc: bool, insn: str = "v_div_fmas_f64 v[0:1], v[2:3], v[4:5], v[6:7]") -> int:
    vcc_set = "v_cmp_eq_u32_e32 vcc_lo, 1, 1" if vcc else "v_cmp_eq_u32_e32 vcc_lo, 0, 1"
    spv = translate_lines([
        f"v_mov_b32 v2, {ux & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v3, {(ux >> 32) & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v4, {uy & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v5, {(uy >> 32) & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v6, {uz & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v7, {(uz >> 32) & 0xFFFFFFFF:#x}",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0",
        vcc_set,
        insn,
    ])
    mrt0 = _eval_spv_f64_mrt0(spv)
    return ((mrt0[1] & 0xFFFFFFFF) << 32) | (mrt0[0] & 0xFFFFFFFF)


class DivFmasF32Tests(unittest.TestCase):
    def test_vcc_clear_and_s2_exponent_threshold(self):
        # Measured on gfx1036: VCC=1 scales by 2^+64 when exp(S2) > 127 (|S2| >= 2.0), else 2^-64.
        cases = (
            (f2u(1.5), f2u(2.0), f2u(0.5), 0x40600000, 0x20600000),      # |S2|=0.5 < 2.0 -> 3.5 * 2^-64
            (f2u(1.0), f2u(2.0), f2u(1.0), 0x40400000, 0x20400000),      # |S2|=1.0 (exp=127) -> 3.0 * 2^-64
            (0, 0, 0x3FFFFFFF, 0x3FFFFFFF, 0x1FFFFFFF),                  # nextbelow(2.0) (exp=127) -> * 2^-64
            (0, 0, 0x40000000, 0x40000000, 0x60000000),                  # 2.0 (exp=128) -> * 2^+64
            (0, 0, 0xBFFFFFFF, 0xBFFFFFFF, 0x9FFFFFFF),                  # -nextbelow(2.0) (exp=127) -> * 2^-64
            (0, 0, 0xC0000000, 0xC0000000, 0xE0000000),                  # -2.0 (exp=128) -> * 2^+64
            (f2u(1.0), f2u(2.0), f2u(2.0), 0x40800000, 0x60800000),      # |S2|=2.0 -> 4.0 * 2^+64
            (f2u(1.0), f2u(2.0), f2u(-4.0), 0xC0000000, 0xE0000000),     # |S2|=4.0 -> -2.0 * 2^+64
        )
        for ux, uy, uz, want0, want1 in cases:
            with self.subTest(uz=hex(uz)):
                self.assertEqual(run_f32_u(ux, uy, uz, False), want0)
                self.assertEqual(run_f32_u(ux, uy, uz, True), want1)

    def test_overflow_prevention_and_sticky_bit(self):
        # Scale-down prevents intermediate product overflow (measured on gfx1036).
        self.assertEqual(run_f32(2.0 ** 80, 2.0 ** 80, 0.5, False), 0x7F800000)
        self.assertEqual(run_f32(2.0 ** 80, 2.0 ** 80, 0.5, True), 0x6F800000)   # 2^96
        self.assertEqual(run_f32(2.0 ** 120, 2.0 ** 20, 1.0, True), 0x65800000)  # 2^76
        # Exact cancellation and mixed-magnitude scale-up (no premature pre-scale overflow).
        self.assertEqual(run_f32(2.0 ** 80, -(2.0 ** -10), 2.0 ** 70, True), 0x00000000)
        self.assertEqual(run_f32(2.0 ** -50, 2.0 ** 100, 2.0, True), 0x78800000)  # 2^114
        # Scale-up overflow to +/-inf.
        self.assertEqual(run_f32(2.0 ** 50, 2.0 ** 50, 2.0 ** 50, True), 0x7F800000)
        self.assertEqual(run_f32(-(2.0 ** 50), 2.0 ** 50, 2.0 ** 50, True), 0xFF800000)
        # Half-ULP tie breaking when z*2^-64 underflows normal range (including subnormal z).
        fx = f2u(2.0 ** 120 * (1.0 + 3.0 * 2.0 ** -23))
        fy = f2u(2.0 ** 20 * 1.5)
        self.assertEqual(run_f32_u(fx, fy, 0, True), 0x65C00004)
        self.assertEqual(run_f32_u(fx, fy, f2u(2.0 ** -100), True), 0x65C00005)
        self.assertEqual(run_f32_u(fx, fy, 0x00000010, True), 0x65C00005)
        self.assertEqual(run_f32_u(fx, fy, f2u(-(2.0 ** -100)), True), 0x65C00004)

    def test_denormal_inputs_kept_and_output_flushed_denorm32_0(self):
        # Measured on gfx1036 under game MODE (denorm32=0): subnormal inputs/intermediates are kept,
        # while subnormal outputs flush to signed zero before and after omod.
        self.assertEqual(run_f32_u(0x00080000, f2u(2.0 ** 80), 0, False), 0x26800000)  # 2^-130 * 2^80 = 2^-50
        self.assertEqual(run_f32_u(0x00080000, f2u(1.0), 0, False), 0x00000000)        # flushed output
        self.assertEqual(run_f32(2.0 ** -70, 2.0 ** -70, 2.0 ** -126, False), 0x00800200)
        self.assertEqual(run_f32(2.0 ** -40, 2.0 ** -40, 0.0, True), 0x00000000)       # 2^-144 -> flushed
        self.assertEqual(run_f32(2.0 ** -31, 2.0 ** -31, 0.0, True), 0x00800000)       # 2^-126 -> min normal
        # Signed zero preserved on flush/scale.
        self.assertEqual(run_f32_u(0x80000000, f2u(1.0), 0x80000000, True), 0x80000000)
        # Flush applies before mul:2 and after div:2.
        self.assertEqual(run_f32_u(0x00400000, f2u(1.0), 0, False, "v_div_fmas_f32 v0, v1, v2, v3 mul:2"), 0)
        self.assertEqual(run_f32_u(0x00800000, f2u(1.0), 0, False, "v_div_fmas_f32 v0, v1, v2, v3 div:2"), 0)

    def test_input_and_output_modifiers(self):
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3"), f2u(1.0))
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3 mul:2"), f2u(2.0))
        self.assertEqual(run_f32(-2.0, 1.5, 4.0, False, "v_div_fmas_f32 v0, |v1|, -v2, v3 mul:2 clamp"), f2u(1.0))
        self.assertEqual(run_f32(1.0, 2.0, 2.0, True, "v_div_fmas_f32 v0, v1, v2, v3 mul:2"), 0x61000000)
        self.assertEqual(run_f32(1.0, 2.0, 2.0, True, "v_div_fmas_f32 v0, v1, v2, v3 div:2"), 0x60000000)
        self.assertEqual(run_f32(2.0 ** 64, 1.0, 2.0, True, "v_div_fmas_f32 v0, v1, v2, v3 div:2"), 0x7F800000)

    def test_spirv_val(self):
        spv = translate_lines([
            "v_mov_b32 v1, 0x3fc00000",
            "v_mov_b32 v2, 0x40000000",
            "v_mov_b32 v3, 0x3f000000",
            "v_cmp_eq_u32_e32 vcc_lo, 1, 1",
            "v_div_fmas_f32 v0, |v1|, -v2, v3 clamp div:2",
        ])
        self.assertNotIn("OpCapability Float64", spv)
        validate_spirv(spv)


class DivFmasF64Tests(unittest.TestCase):
    def test_vcc_clear_and_s2_exponent_threshold(self):
        # Measured on gfx1036 (wave32 and wave64): VCC=1 scales by 2^+128 when exp(S2) > 1023, else 2^-128.
        cases = (
            (d2u(1.0), d2u(2.0), d2u(1.0), 0x4008000000000000, 0x3808000000000000),
            (0, 0, 0x3FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF, 0x37FFFFFFFFFFFFFF),
            (0, 0, 0x4000000000000000, 0x4000000000000000, 0x4800000000000000),
            (0, 0, 0xBFFFFFFFFFFFFFFF, 0xBFFFFFFFFFFFFFFF, 0xB7FFFFFFFFFFFFFF),
            (0, 0, 0xC000000000000000, 0xC000000000000000, 0xC800000000000000),
            (d2u(1.0), d2u(2.0), d2u(2.0), 0x4010000000000000, 0x4810000000000000),
        )
        for ux, uy, uz, want0, want1 in cases:
            with self.subTest(uz=hex(uz)):
                self.assertEqual(run_f64_u(ux, uy, uz, False), want0)
                self.assertEqual(run_f64_u(ux, uy, uz, True), want1)

    def test_overflow_prevention_sticky_and_subnormal_single_rounding(self):
        # All expected 64-bit words measured natively on gfx1036.
        self.assertEqual(run_f64_u(d2u(2.0 ** 530), d2u(2.0 ** 530), d2u(0.5), False), 0x7FF0000000000000)
        self.assertEqual(run_f64_u(d2u(2.0 ** 530), d2u(2.0 ** 530), d2u(0.5), True), 0x7A30000000000000)
        self.assertEqual(run_f64_u(d2u(2.0 ** 1000), d2u(2.0 ** 100), d2u(1.0), True), 0x7CB0000000000000)
        self.assertEqual(run_f64_u(d2u(2.0 ** 600), d2u(-(2.0 ** 350)), d2u(2.0 ** 950), True), 0x0000000000000000)
        self.assertEqual(run_f64_u(d2u(2.0 ** -500), d2u(2.0 ** 900), d2u(2.0), True), 0x60F0000000000000)
        # Single-rounded subnormal half-ULP ties (double rounding gives 0x...0000 and 0x...0002 instead of 0x...0001).
        x1 = d2u(2.0 ** -450)
        y1 = d2u(2.0 ** -450 * (1.0 + 2.0 ** -47))
        y2 = d2u(2.0 ** -450 * (1.0 + 2.0 ** -46 + 2.0 ** -47))
        self.assertEqual(run_f64_u(x1, y1, d2u(2.0 ** -953), True), 0x0000400000000001)
        self.assertEqual(run_f64_u(x1, y1, 0, True), 0x0000400000000000)
        self.assertEqual(run_f64_u(x1, y2, d2u(-(2.0 ** -953)), True), 0x0000400000000001)
        self.assertEqual(run_f64_u(x1, y2, 0, True), 0x0000400000000002)
        # Half-of-min-subnormal boundary (2^-947 * 2^-128 = 2^-1075) and tiny product sticky bit.
        self.assertEqual(run_f64_u(d2u(2.0 ** -473), d2u(2.0 ** -474), d2u(2.0 ** -1074), True), 0x0000000000000001)
        self.assertEqual(run_f64_u(d2u(2.0 ** -473), d2u(2.0 ** -474), d2u(-(2.0 ** -1074)), True), 0x0000000000000000)
        self.assertEqual(run_f64_u(d2u(2.0 ** -600), d2u(2.0 ** -600), d2u(2.0 ** -947), True), 0x0000000000000001)
        self.assertEqual(run_f64_u(d2u(-(2.0 ** -600)), d2u(2.0 ** -600), d2u(2.0 ** -947), True), 0x0000000000000000)
        # k=52 boundary (2^-895 * (1 + 2^-52) * 2^-128 = 2^-1023 + 2^-1075).
        xa = d2u(2.0 ** -447)
        ya = d2u(2.0 ** -448 * (1.0 + 2.0 ** -52))
        self.assertEqual(run_f64_u(xa, ya, d2u(2.0 ** -1000), True), 0x0008000000000001)
        self.assertEqual(run_f64_u(xa, ya, d2u(-(2.0 ** -1000)), True), 0x0008000000000000)
        # big_prod sticky bit in f64.
        bx = d2u(2.0 ** 900 * (1.0 + 3.0 * 2.0 ** -52))
        by = d2u(2.0 ** 150 * 1.5)
        self.assertEqual(run_f64_u(bx, by, 0, True), 0x7998000000000004)
        self.assertEqual(run_f64_u(bx, by, d2u(2.0 ** -950), True), 0x7998000000000005)
        self.assertEqual(run_f64_u(bx, by, d2u(-(2.0 ** -950)), True), 0x7998000000000004)

    def test_f64_lowering_modifiers_and_validation(self):
        up_lo, up_hi = dwords(2.0 ** 128)
        dn_lo, dn_hi = dwords(2.0 ** -128)
        for mods, want_clamp in (
            ("", False),
            ("mul:2", False),
            ("div:2", False),
            ("clamp", True),
            ("mul:4 clamp", True),
        ):
            with self.subTest(mods=mods):
                spv = translate_lines([
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0x3ff80000",
                    "s_mov_b32 s4, 0",
                    "s_mov_b32 s5, 0x40000000",
                    "v_cmp_eq_u32_e32 vcc_lo, 1, 1",
                    f"v_div_fmas_f64 v[0:1], |v[2:3]|, -s[4:5], 0.5 {mods}".rstrip(),
                ])
                self.assertIn("OpCapability Float64", spv)
                self.assertNotIn("OpCapability Int64", spv)
                self.assertRegex(spv, r"OpExtInst %double %glsl Fma %\w+ %\w+ %\w+")
                self.assertRegex(spv, r"OpExtInst %double %glsl FAbs %\w+")
                self.assertRegex(spv, r"OpFNegate %double %\w+")
                self.assertIn(f"OpConstant %uint {up_hi}", spv)
                self.assertIn(f"OpConstant %uint {up_lo}", spv)
                self.assertIn(f"OpConstant %uint {dn_hi}", spv)
                self.assertIn(f"OpConstant %uint {dn_lo}", spv)
                if want_clamp:
                    self.assertIn("FClamp", spv)
                else:
                    self.assertNotIn("FClamp", spv)
                validate_spirv(spv)

    def test_f64_rejects_non_pair_destination(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_div_fmas_f64 v0, v[2:3], v[4:5], v[6:7]"])


if __name__ == "__main__":
    unittest.main()
