"""Differential checks for the 47 binary64 VALU instruction forms translated by main.

The reference below decodes IEEE-754 words itself. Finite arithmetic is performed with
Fractions and rounded to binary64; it shares no implementation with the translator, its
CPU interpreter, cpu_ref, or another fuzzer. The second path translates generated listings
and executes their SPIR-V with the semantic one-lane interpreter. Only synthetic programs
are created; no game files are read.

The GPU path is deliberately optional. It translates the same fragment shader, runs it
through vk_harness, and reads back RGBA32F words. ``--oracle hw`` runs the programs natively
on an AMD GPU through tools/hw_oracle (``--hw-arch`` selects the architecture) and compares
them with the reference; it returns 77 only when that oracle is unavailable. ``--known`` runs
known divergence probes; a reproduced mismatch intentionally returns 1.
Usage: `python -I tools/f64_diff.py --help` lists the options.
"""
import argparse
import gc
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEM = os.path.join(ROOT, "tests", "semantic")
for path in (HERE, SEM, os.path.join(ROOT, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)
sys.dont_write_bytecode = True

import rdna2spirv  # noqa: E402
from spirv_mini_interp import UnsupportedConstruct, run_fragment  # noqa: E402
import spirv_interp  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
EXP = 0x7FF0000000000000
FRAC = 0x000FFFFFFFFFFFFF
QNAN = 0x7FF8000000000000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
CONDS = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")
ROUND = ("v_trunc_f64", "v_ceil_f64", "v_rndne_f64", "v_floor_f64")
BASE = (*ROUND, "v_add_f64", "v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64",
        "v_div_fixup_f64", "v_div_fmas_f64", "v_frexp_mant_f64", "v_frexp_exp_i32_f64",
        *(f"v_cmp_{c}_f64" for c in CONDS), *(f"v_cmpx_{c}_f64" for c in CONDS),
        "v_cmp_class_f64", "v_cmpx_class_f64")
OP_FORMS = {op: "round" if op in ROUND else "unary" if op in ("v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64", "v_frexp_mant_f64", "v_frexp_exp_i32_f64") else "binary" if op == "v_add_f64" else "ternary" if op in ("v_div_fixup_f64", "v_div_fmas_f64") else "compare" if "cmp_" in op and "class" not in op else "class" for op in BASE}
EXPORT = "exp mrt0 v8, v9, v10, v11 done vm"
KNOWN_DIVERGENCES = {
    "f64-omod": "SHADER-721: f64 output modifiers; owned by @fiestorra.",
    "double-neg-abs": "TEST-912 (#4172): spirv_mini_interp rounds binary64 OpFNegate/FAbs through binary32, changing values as well as NaN payloads.",
    "fclamp-nan-zero": "FClamp NaN and signed-zero results are driver-dependent after f64 clamp/omod.",
    "div-fixup-nan-s0": "RDNA2 ISA maps NaN(S0) to signed infinity; current translator preserves the NaN payload.",
}


class HarnessError(Exception):
    pass


class SkipCase(Exception):
    pass


class GPUUnavailable(Exception):
    pass


class HWUnavailable(Exception):
    """No configured AMD hw_oracle on this host (tools/hw_oracle/); the CLI turns it into exit 77."""


def words_to_bits(lo, hi):
    return ((hi & M32) << 32) | (lo & M32)


def bits_to_words(bits):
    return bits & M32, (bits >> 32) & M32


def is_nan(x):
    return (x & EXP) == EXP and (x & FRAC) != 0


def is_inf(x):
    return x & 0x7FFFFFFFFFFFFFFF == EXP


def is_zero(x):
    return x & 0x7FFFFFFFFFFFFFFF == 0


def is_subnormal(x):
    return (x & EXP) == 0 and (x & FRAC) != 0


def is_f32_subnormal(x):
    return (x & 0x7F800000) == 0 and (x & 0x7FFFFF) != 0


def to_float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits & M64))[0]


def from_float(value):
    try:
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    except OverflowError:
        return 0xFFF0000000000000 if value < 0 else EXP


def to_fraction(bits):
    e, m = (bits >> 52) & 0x7FF, bits & FRAC
    if e == 0:
        n, p = m, -1074
    else:
        n, p = m | (1 << 52), e - 1075
    f = Fraction(n) * (Fraction(2) ** p)
    return -f if bits >> 63 else f


def round_fraction(value, mode=0, negative_zero=False):
    """Round a finite rational to binary64: nearest-even, +inf, -inf, or toward zero."""
    if not value:
        return 0x8000000000000000 if negative_zero else 0
    sign = 0x8000000000000000 if value < 0 else 0
    a = abs(value)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    def overflow():
        negative = bool(sign)
        if mode == 3 or (mode == 1 and negative) or (mode == 2 and not negative):
            return sign | 0x7FEFFFFFFFFFFFFF
        return sign | EXP

    if e > 1023: return overflow()
    if e < -1022:
        scaled = a * Fraction(2 ** 1074)
        q, r = divmod(scaled.numerator, scaled.denominator)
        increment = (2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1)) if mode == 0 else bool(r) and ((mode == 1 and not sign) or (mode == 2 and bool(sign)))
        if increment:
            q += 1
        return sign | q
    scaled = a / Fraction(2) ** (e - 52)
    q, r = divmod(scaled.numerator, scaled.denominator)
    increment = (2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1)) if mode == 0 else bool(r) and ((mode == 1 and not sign) or (mode == 2 and bool(sign)))
    if increment:
        q += 1
    if q == 1 << 53:
        q >>= 1
        e += 1
    if e > 1023:
        return overflow()
    return sign | ((e + 1023) << 52) | (q & FRAC)


def add_bits(a, b, mode=0):
    """IEEE sum; MODE 3 (toward zero) steps back if nearest-even rounded away from zero."""
    if is_nan(a) or is_nan(b):
        return QNAN
    if is_inf(a) or is_inf(b):
        af, bf = to_float(a), to_float(b)
        return from_float(af + bf)
    exact = to_fraction(a) + to_fraction(b)
    # IEEE 754-2019 6.3: an exact zero sum of two zeros of the same sign keeps that sign in every mode; any other
    # exact zero (opposite-signed zeros, or x + (-x)) is -0 only when rounding toward negative, +0 otherwise.
    if is_zero(a) and is_zero(b) and (a ^ b) >> 63 == 0:
        negzero = bool(a >> 63)
    else:
        negzero = mode == 2
    return round_fraction(exact, mode, negzero)


def class_index(bits):
    sign, e, m = bits >> 63, (bits >> 52) & 0x7FF, bits & FRAC
    if e == 0x7FF:
        if m:
            return 0 if not (m & (1 << 51)) else 1
        return 2 if sign else 9
    if e == 0:
        if not m:
            return 5 if sign else 6
        return 4 if sign else 7
    return 3 if sign else 8


def compare(cond, a, b):
    unordered = is_nan(a) or is_nan(b)
    if cond == "f": return 0
    if cond == "tru": return 1
    if cond == "o": return int(not unordered)
    if cond == "u": return int(unordered)
    if unordered:
        return int({"nge": 0, "nlg": 1, "ngt": 2, "nle": 3, "neq": 4, "nlt": 5}.get(cond, -1) >= 0)
    x, y = to_float(a), to_float(b)
    result = {"lt": x < y, "eq": x == y, "le": x <= y, "gt": x > y, "lg": x != y,
              "ge": x >= y, "nge": x < y, "nlg": x == y, "ngt": x <= y,
              "nle": x > y, "neq": x != y, "nlt": x >= y}
    return int(result[cond])


def fixup(s0, s1, s2):
    sign = (s1 ^ s2) & 0x8000000000000000
    if is_nan(s2): return s2 | (1 << 51)
    if is_nan(s1): return s1 | (1 << 51)
    z1, z2, i1, i2 = is_zero(s1), is_zero(s2), is_inf(s1), is_inf(s2)
    if (z1 and z2) or (i1 and i2): return 0xFFF8000000000000
    e1, e2 = (s1 >> 52) & 0x7FF, (s2 >> 52) & 0x7FF
    ediff = e2 - e1
    if i1 or z2 or ediff < -1074: return sign
    if z1 or i2 or ediff >= 1024 or is_nan(s0): return sign | EXP
    return (s0 & 0x7FFFFFFFFFFFFFFF) | sign


def frexp(bits):
    sign = bits & 0x8000000000000000
    e, m = (bits >> 52) & 0x7FF, bits & FRAC
    if e == 0x7FF or (e == 0 and m == 0): return bits, 0
    if e:
        return sign | (0x3FE << 52) | m, e - 1022
    top = m.bit_length() - 1
    exp = top - 1073
    mantissa = m << (52 - top)
    return sign | (0x3FE << 52) | (mantissa & FRAC), exp


def rounded_unary(op, bits):
    if is_nan(bits) or is_inf(bits) or is_zero(bits): return bits
    x = to_float(bits)
    value = {"v_trunc_f64": math.trunc, "v_ceil_f64": math.ceil,
             "v_rndne_f64": round, "v_floor_f64": math.floor}[op](x)
    if value == 0 and bits >> 63: return 0x8000000000000000
    return from_float(float(value))


def read_source(tok, vgpr, sgpr):
    tok = tok.strip()
    neg = tok.startswith("neg(") or tok.startswith("-|") or (tok.startswith("-") and tok[1:].startswith(("v[", "s[", "v", "s", "|")))
    absmod = "|" in tok
    tok = tok.replace("|", "").replace("neg(", "").replace(")", "").strip()
    if neg and tok.startswith("-"):
        tok = tok[1:].strip()
    m = re.fullmatch(r"([sv])\[(\d+):(\d+)\]", tok)
    if m:
        regs = vgpr if m.group(1) == "v" else sgpr
        lo, hi = int(m.group(2)), int(m.group(3))
        value = words_to_bits(regs.get(lo, 0), regs.get(hi, 0))
    elif tok.startswith(("v", "s")) and tok[1:].isdigit():
        raise HarnessError("binary64 source must be an even register pair")
    elif tok.startswith("0x"):
        # A printed 32-bit ISA literal occupies bits 63:32 in forms marked by shader_ir;
        # a full 64-bit hexadecimal operand is already the complete IEEE word.
        value = int(tok, 16)
        if value <= M32: value <<= 32
        value &= M64
    elif re.fullmatch(r"[-+]?\d+", tok):
        n = int(tok)
        value = n & M64
    else:
        try: value = from_float(float(tok))
        except (ValueError, OverflowError) as exc: raise HarnessError(f"bad source {tok!r}") from exc
    if absmod: value &= 0x7FFFFFFFFFFFFFFF
    if neg: value ^= 0x8000000000000000
    return value


def parse_source_args(line):
    head, *args = line.split(",")
    bits = head.split()
    return bits[0], bits[1].rstrip(","), [x.strip() for x in args]


def ref_program(lines, include_known=False):
    vgpr, sgpr, mode, vcc, active = {}, {}, 0, 0, M32
    for line in lines:
        s = line.strip()
        if not s or s.startswith((";", "exp ", "s_endpgm")): continue
        if s.startswith("v_mov_b32") or s.startswith("s_mov_b32"):
            op, dst, srcs = parse_source_args(s)
            src = srcs[0]
            if dst == "exec_lo": active = sgpr.get(int(src[1:]), 0) & M32
            elif src == "exec_lo": sgpr[int(dst[1:])] = active
            else:
                if re.fullmatch(r"[sv]\d+", src):
                    regs = vgpr if src.startswith("v") else sgpr
                    word = regs.get(int(src[1:]), 0)
                else: word = int(src, 0)
                (vgpr if dst.startswith("v") else sgpr)[int(dst[1:])] = word & M32
            continue
        if s.startswith("s_round_mode "):
            mode = (int(s.split(None, 1)[1], 0) >> 2) & 3
            continue
        if "s_setreg_imm32_b32" in s:
            mode = (int(s.rsplit(",", 1)[1], 0) >> 2) & 3
            continue
        if s.startswith("v_cmp_eq_i32_e32 vcc_lo,"):
            vcc = int(s.rsplit(",", 2)[1].strip(), 0) == int(s.rsplit(",", 1)[1].strip(), 0)
            continue
        op, dst, a = parse_source_args(s)
        if op in ROUND:
            out = rounded_unary(op, read_source(a[0], vgpr, sgpr))
            lo, hi = bits_to_words(out); vgpr[int(dst[2:-1].split(":")[0])] = lo
            vgpr[int(dst[2:-1].split(":")[0]) + 1] = hi
        elif op == "v_add_f64":
            modifiers = re.findall(r"(?<!\S)(mul:2|mul:4|div:2|clamp)(?=\s|$)", a[1])
            right = re.sub(r"(?<!\S)(?:mul:2|mul:4|div:2|clamp)(?=\s|$)", "", a[1]).strip()
            out = add_bits(read_source(a[0], vgpr, sgpr), read_source(right, vgpr, sgpr), mode)
            for modifier in modifiers:
                scale = {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}.get(modifier)
                if scale is not None:
                    out = QNAN if is_nan(out) else from_float(to_float(out) * scale)
            if "clamp" in modifiers and not is_nan(out):
                out = from_float(min(1.0, max(0.0, to_float(out))))
            d = int(dst[2:-1].split(":")[0]); vgpr[d], vgpr[d + 1] = bits_to_words(out)
        elif op in ("v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64"):
            x = read_source(a[0], vgpr, sgpr)
            if op == "v_cvt_f32_f64":
                try: out = struct.unpack("<I", struct.pack("<f", to_float(x)))[0]
                except OverflowError: out = 0xFF800000 if x >> 63 else 0x7F800000
            elif op == "v_rcp_f64": out = from_float(1.0 / to_float(x)) if not is_zero(x) else ((x ^ 0x3FF0000000000000) & 0x8000000000000000) | EXP
            elif op == "v_sqrt_f64":
                if is_zero(x): out = x
                elif x >> 63: out = QNAN
                else: out = from_float(math.sqrt(to_float(x)))
            elif is_zero(x): out = (x & 0x8000000000000000) | EXP
            else: out = QNAN if x >> 63 else from_float(1.0 / math.sqrt(to_float(x)))
            if op in ("v_rcp_f64", "v_rsq_f64", "v_sqrt_f64") and (is_subnormal(x) or is_subnormal(out)):
                raise SkipCase(f"{op} subnormal operand/result is driver-dependent (FTZ)")
            if op == "v_cvt_f32_f64" and is_f32_subnormal(out):
                raise SkipCase("v_cvt_f32_f64 subnormal f32 result is driver-dependent (FTZ)")
            if op == "v_cvt_f32_f64": vgpr[int(dst[1:])] = out
            else:
                d = int(dst[2:-1].split(":")[0]); vgpr[d], vgpr[d + 1] = bits_to_words(out)
        elif op == "v_div_fixup_f64":
            args = [read_source(t, vgpr, sgpr) for t in a[:3]]
            s0, s1, s2 = args
            ediff = ((s2 >> 52) & 0x7FF) - ((s1 >> 52) & 0x7FF)
            normal_division = (not is_nan(s1) and not is_inf(s1) and not is_zero(s1)
                               and not is_nan(s2) and not is_inf(s2) and not is_zero(s2)
                               and -1074 <= ediff < 1024)
            if not include_known and is_nan(s0) and normal_division:
                raise SkipCase(f"KNOWN_DIVERGENCE div-fixup-nan-s0: {KNOWN_DIVERGENCES['div-fixup-nan-s0']}")
            d = int(dst[2:-1].split(":")[0]); vgpr[d], vgpr[d + 1] = bits_to_words(fixup(*args))
        elif op == "v_frexp_mant_f64":
            d = int(dst[2:-1].split(":")[0]); vgpr[d], vgpr[d + 1] = bits_to_words(frexp(read_source(a[0], vgpr, sgpr))[0])
        elif op == "v_frexp_exp_i32_f64":
            vgpr[int(dst[1:])] = frexp(read_source(a[0], vgpr, sgpr))[1] & M32
        elif "class" in op:
            operands = [dst] + a if op.startswith("v_cmpx") else a
            src = read_source(operands[-2], vgpr, sgpr); mask = int(operands[-1], 0) & 0x3FF
            hit = int(bool(mask & (1 << class_index(src))))
            sgpr[20] = hit
            if op.startswith("v_cmpx"): active &= M32 if hit else 0
        elif op.startswith("v_cmp_") or op.startswith("v_cmpx_"):
            cond = op.split("_")[2]
            operands = [dst] + a if op.startswith("v_cmpx") else a
            hit = compare(cond, read_source(operands[-2], vgpr, sgpr), read_source(operands[-1], vgpr, sgpr))
            sgpr[20] = hit
            if op.startswith("v_cmpx"): active &= M32 if hit else 0
        elif op == "v_cndmask_b32_e64":
            _, dst, a = parse_source_args(s)
            vgpr[int(dst[1:])] = int(a[1], 0) & M32 if sgpr.get(20, 0) else int(a[0], 0) & M32
        elif op == "v_div_fmas_f64":
            x, y, z = (read_source(t, vgpr, sgpr) for t in a)
            if any(is_nan(t) or is_inf(t) for t in (x, y, z)):
                raise SkipCase("non-finite v_div_fmas_f64 operands are not judged")
            exact = to_fraction(x) * to_fraction(y) + to_fraction(z)
            if vcc: exact *= 1 << 64
            d = int(dst[2:-1].split(":")[0]); vgpr[d], vgpr[d + 1] = bits_to_words(round_fraction(exact))
        else:
            raise HarnessError(f"reference does not implement {op}")
    return [vgpr.get(r, 0) & M32 for r in range(8, 12)]


def write_listing(path, lines, name):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name} code 0x0..0x1000\n")
        for line in lines: f.write("\t" + line + "\n")
        f.write("\t" + EXPORT + "\n\ts_endpgm\n")


def translate_assembly(lines, name, tmpdir):
    dis = os.path.join(tmpdir, "disasm"); os.makedirs(dis, exist_ok=True)
    src = os.path.join(dis, name + ".s")
    write_listing(src, lines, name)
    with open(os.path.join(tmpdir, name + ".dxbc"), "wb") as f: f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def translate(lines, name, tmpdir):
    asm = translate_assembly(lines, name, tmpdir)
    gc.collect()
    if "OpCapability Float64" in asm:
        if "OpFDiv %double" in asm:
            raise SkipCase("the available SPIR-V CPU interpreters do not model binary64 OpFDiv")
        if "OpFConvert" in asm:
            raise SkipCase("the available SPIR-V CPU interpreters do not model binary64 OpFConvert")
        try: return [int(x) & M32 for x in spirv_interp.run(asm).vgpr[8:12]]
        except (NotImplementedError, ZeroDivisionError, OverflowError, ValueError) as exc:
            raise SkipCase(f"binary64 SPIR-V interpreter limitation: {type(exc).__name__}: {exc}") from exc
    try: out = run_fragment(asm)
    except UnsupportedConstruct as exc: raise SkipCase(str(exc)) from exc
    if list(out) != ["%out_mrt0"]: raise HarnessError(f"unexpected outputs: {list(out)}")
    return [int(x) & M32 for x in out["%out_mrt0"]]


def sdk_tool(name):
    exe = name + (".exe" if os.name == "nt" else "")
    found = shutil.which(exe) or shutil.which(name)
    if found: return found
    sdk = os.environ.get("VULKAN_SDK")
    candidate = os.path.join(sdk, "Bin", exe) if sdk else ""
    return candidate if candidate and os.path.isfile(candidate) else None


def prepare_gpu(exe, tmpdir):
    if not os.path.isfile(exe): raise GPUUnavailable(f"vk_harness not found: {exe}")
    glslc, spirv_as = sdk_tool("glslc"), sdk_tool("spirv-as")
    if not glslc or not spirv_as: raise GPUUnavailable("Vulkan SDK glslc and spirv-as are required for --oracle gpu")
    vertex = os.path.join(ROOT, "runtime", "test", "glsl", "vs_fullscreen.vert")
    if not os.path.isfile(vertex): raise HarnessError(f"full-screen vertex shader is missing: {vertex}")
    vs_spv = os.path.join(tmpdir, "f64_diff_vs.spv")
    try:
        r = subprocess.run([glslc, vertex, "-o", vs_spv], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired: raise HarnessError("glslc timed out while compiling the full-screen vertex shader") from None
    if r.returncode: raise HarnessError(f"glslc failed: {(r.stdout + r.stderr).strip()}")
    return {"exe": exe, "glslc": glslc, "spirv_as": spirv_as, "vs_spv": vs_spv}


def run_gpu(lines, name, tmpdir, gpu):
    asm = translate_assembly(lines, name, tmpdir)
    asm_path = os.path.join(tmpdir, name + ".spvasm")
    fs_spv = os.path.join(tmpdir, name + ".spv")
    raw = os.path.join(tmpdir, name + ".raw")
    with open(asm_path, "w", encoding="utf-8", newline="\n") as f: f.write(asm)
    try:
        r = subprocess.run([gpu["spirv_as"], "--target-env", "vulkan1.2", asm_path, "-o", fs_spv],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired: raise HarnessError("spirv-as timed out while assembling the translated shader") from None
    if r.returncode: raise HarnessError(f"spirv-as failed: {(r.stdout + r.stderr).strip()}")
    try:
        r = subprocess.run([gpu["exe"], gpu["vs_spv"], fs_spv, raw, "1", "1"],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired: raise HarnessError("vk_harness timed out after 60 s") from None
    detail = " ".join((r.stdout + r.stderr).split())
    if r.returncode == 77: raise GPUUnavailable(detail or "vk_harness reports that the Vulkan device cannot run this shader")
    if r.returncode: raise HarnessError(f"vk_harness exited {r.returncode}: {detail}")
    if not os.path.isfile(raw) or os.path.getsize(raw) != 16:
        raise HarnessError("vk_harness did not write exactly four RGBA32F dwords")
    with open(raw, "rb") as f: return list(struct.unpack("<4I", f.read()))


# The float MODE the hardware kernels start with: the one docs/design.md infers for the game's shaders (the same as
# tools/fmisc_diff.py): IEEE off so output modifiers apply, DX10_CLAMP on. Directives an architecture cannot set make
# the program a counted skip (runner.Oracle.run returns the reason).
HW_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0}


def hw_kernel(lines):
    """A raw tools/hw_oracle kernel: the program on one lane, then v8..v11 stored at the output pointer."""
    from hw_oracle import asm
    return asm.Kernel(list(lines) + ["v_mov_b32 v0, 0", "global_store_b128 v0, v[8:11], s[2:3]", "s_endpgm"],
                      wave=32, threads=1, words=4, raw=True, mode=HW_MODE)


def open_hw_oracle(arch=None):
    """The AMD hardware oracle, or HWUnavailable when its tools or device are missing."""
    try:
        from hw_oracle import runner
    except ImportError as exc:
        raise HWUnavailable(f"tools/hw_oracle cannot be imported: {exc}") from exc
    try:
        return runner.open_oracle(arch=arch)
    except runner.Unavailable as exc:
        raise HWUnavailable(str(exc)) from exc


def rand_bits(rng):
    # Mostly finite normals, with signed zeros, infinities and NaNs for predicate/class coverage.
    if rng.random() < .2:
        return rng.choice((0, 0x8000000000000000, EXP, 0xFFF0000000000000, QNAN, 0xFFF0000000000001, 1, 0x8000000000000001))
    return rng.getrandbits(64) & 0xFFEFFFFFFFFFFFFF | (rng.getrandbits(1) << 63) | (rng.randrange(1, 2047) << 52)


def gen_source(rng, idx, allow_literal=True, bus=None):
    choices = ("v", "s", "fimm", "inline", "literal") if allow_literal else ("v", "s", "fimm", "inline")
    kind = rng.choice(choices)
    if bus is not None:
        if kind == "s" and len(bus[0]) >= 2: kind = "v"
        if kind == "literal" and bus[1]: kind = "v"
        if kind == "s": bus[0].add(idx)
        if kind == "literal": bus[1] = True
    if kind == "v": return f"v[{8 + 2*idx}:{9 + 2*idx}]"
    if kind == "s": return f"s[{4 + 2*(idx % 2)}:{5 + 2*(idx % 2)}]"
    if kind == "fimm": return rng.choice(("0.0", "-0.0", "1.0", "-1.0", "1.5", "2.0", "0.5", "3.25"))
    if kind == "inline": return str(rng.choice((-16, -1, 0, 1, 2, 16, 64)))
    if bus is not None and bus[2] == 64:
        return f"0x{rng.getrandbits(64):016x}"
    return f"0x{rng.getrandbits(32):08x}"


def maybe_word_modifier(rng, src):
    """Only add modifiers to integer-word paths (round, compare and class), not %double math."""
    if not src.startswith(("v[", "s[")) or rng.random() >= 0.25: return src
    choice = rng.randrange(3)
    return f"|{src}|" if choice == 0 else f"-{src}" if choice == 1 else f"-|{src}|"


def make_case(rng, op, allow_literal=True, known_case=None):
    setup_lines, body_lines = [], []
    vg, sg = {}, {}
    for r in range(4, 8):
        word = rng.getrandbits(32); sg[r] = word; setup_lines.append(f"s_mov_b32 s{r}, 0x{word:08x}")
    # Replace both data lanes with selected edge cases for interpretable differential coverage.
    for d in range(4):
        bits = rand_bits(rng); lo, hi = bits_to_words(bits); r = 8 + 2*d
        vg[r], vg[r + 1] = lo, hi
        setup_lines += [f"v_mov_b32 v{r}, 0x{lo:08x}", f"v_mov_b32 v{r+1}, 0x{hi:08x}"]
    if op == "v_add_f64": body_lines.append(f"s_round_mode {0 if known_case else rng.choice((0, 4, 8, 12))}")
    high_literal = op in ROUND or op in ("v_add_f64", "v_cvt_f32_f64") or "cmp" in op or "class" in op
    bus = [set(), False, 32 if high_literal else 64]
    if op in ROUND:
        src = maybe_word_modifier(rng, gen_source(rng, 0, allow_literal, bus))
        body_lines.append(f"{op} v[8:9], {src}")
    elif op in ("v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64", "v_frexp_mant_f64", "v_frexp_exp_i32_f64"):
        src = gen_source(rng, 0, allow_literal, bus)
        dst = "v8" if op in ("v_cvt_f32_f64", "v_frexp_exp_i32_f64") else "v[8:9]"
        body_lines.append(f"{op} {dst}, {src}")
    elif op == "v_add_f64":
        left, right = gen_source(rng, 0, allow_literal, bus), gen_source(rng, 1, allow_literal, bus)
        if known_case == "double-neg-abs": left = "-|v[8:9]|"
        suffix = " mul:2" if known_case == "f64-omod" else " clamp" if known_case == "fclamp-nan-zero" else ""
        if known_case == "fclamp-nan-zero": left, right = "0x7ff80001", "1.0"
        body_lines.append(f"{op} v[8:9], {left}, {right}{suffix}")
    elif op == "v_div_fixup_f64":
        if known_case == "div-fixup-nan-s0":
            setup_lines += ["v_mov_b32 v12, 0x00000000", "v_mov_b32 v13, 0x7ff80000"]
            body_lines.append(f"{op} v[8:9], v[12:13], 1.0, 2.0")
        else:
            body_lines.append(f"{op} v[8:9], {gen_source(rng, 0, allow_literal, bus)}, {gen_source(rng, 1, allow_literal, bus)}, {gen_source(rng, 2, allow_literal, bus)}")
    elif op == "v_div_fmas_f64":
        # VCC is initialized true so the f64 scaled FMA path is exercised deterministically.
        body_lines += ["v_cmp_eq_i32_e32 vcc_lo, 0, 0", f"{op} v[8:9], {gen_source(rng, 0, allow_literal, bus)}, {gen_source(rng, 1, allow_literal, bus)}, {gen_source(rng, 2, allow_literal, bus)}"]
    elif "class" in op:
        src, mask = maybe_word_modifier(rng, gen_source(rng, 2 if op.startswith("v_cmpx") else 0, allow_literal, bus)), f"0x{rng.randrange(1024):x}"
        if op.startswith("v_cmpx"):
            body_lines += ["s_mov_b32 s10, exec_lo", f"{op}_e64 {src}, {mask}", "s_mov_b32 s22, exec_lo", "s_mov_b32 exec_lo, s10", "v_mov_b32 v9, s22"]
            body_lines += [f"v_cmp_class_f64_e64 s20, {src}, {mask}"]
        else: body_lines.append(f"{op}_e64 s20, {src}, {mask}")
        body_lines.append("v_cndmask_b32_e64 v8, 0, 1, s20")
    else:
        cond = op.split("_")[2]
        source_index = 2 if op.startswith("v_cmpx") else 0
        s0 = maybe_word_modifier(rng, gen_source(rng, source_index, allow_literal, bus))
        s1 = maybe_word_modifier(rng, gen_source(rng, 3 if op.startswith("v_cmpx") else 1, allow_literal, bus))
        if op.startswith("v_cmpx"):
            body_lines += ["s_mov_b32 s10, exec_lo", f"{op}_e64 {s0}, {s1}", "s_mov_b32 s22, exec_lo", "s_mov_b32 exec_lo, s10", "v_mov_b32 v9, s22"]
        body_lines += [f"v_cmp_{cond}_f64_e64 s20, {s0}, {s1}", "v_cndmask_b32_e64 v8, 0, 1, s20"]
    return setup_lines, body_lines, vg, sg


def output_kind(lines):
    low, high = "int", "int"
    for line in lines:
        s = line.strip()
        if s.startswith(("v_cndmask_b32_e64 v8,", "v_mov_b32 v8,")): low = "int"
        elif s.startswith("v_mov_b32 v9,"): high = "int"
        elif s.startswith("v_cvt_f32_f64 v8,"): low = "f32"
        elif s.startswith("v_frexp_exp_i32_f64 v8,"): low = "int"
        elif any(s.startswith(op + " v[8:9],") for op in (*ROUND, "v_add_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64", "v_div_fixup_f64", "v_div_fmas_f64", "v_frexp_mant_f64")):
            low, high = "f64", "f64"
    return low, high


def equivalent(a, b, kind="int"):
    if len(a) != len(b): return False
    if kind == "f64":
        x, y = words_to_bits(a[0], a[1]), words_to_bits(b[0], b[1])
        if x != y and not (is_nan(x) and is_nan(y)): return False
        return a[2:] == b[2:]
    if kind == "f32":
        x, y = a[0], b[0]
        if x != y and not (((x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF) and
                           ((y & 0x7F800000) == 0x7F800000 and y & 0x7FFFFF)): return False
        return a[1:] == b[1:]
    if isinstance(kind, tuple):
        low, high = kind
        if low == "f64":
            x, y = words_to_bits(a[0], a[1]), words_to_bits(b[0], b[1])
            if x != y and not (is_nan(x) and is_nan(y)): return False
        elif low == "f32":
            x, y = a[0], b[0]
            if x != y and not (((x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF) and
                               ((y & 0x7F800000) == 0x7F800000 and y & 0x7FFFFF)): return False
        elif a[0] != b[0]: return False
        if high == "f64" and low != "f64":
            x, y = words_to_bits(a[0], a[1]), words_to_bits(b[0], b[1])
            if not (is_nan(x) and is_nan(y)) and a[1] != b[1]: return False
        elif high != "f64" and a[1] != b[1]: return False
        return a[2:] == b[2:]
    return a == b


def run_one(lines, name, tmp, oracle="translator", gpu=None, include_known=False, hw_words=None):
    """Compare one program with the reference. ``gpu`` is the prepared Vulkan context for --oracle gpu, or the open
    hw_oracle for --oracle hw (whose result may be passed in as ``hw_words``: four words or a skip reason)."""
    if oracle == "hw":
        # The hardware runs every form natively, so the translator-only exclusions do not apply: compare it with the
        # whole reference, the known cases included.
        include_known = True
    if not include_known:
        for line in lines:
            if " mul:" in line or " div:" in line:
                return "skip", f"KNOWN_DIVERGENCE f64-omod: {KNOWN_DIVERGENCES['f64-omod']}", None, None
            if " clamp" in line and "f64" in line:
                return "skip", f"KNOWN_DIVERGENCE fclamp-nan-zero: {KNOWN_DIVERGENCES['fclamp-nan-zero']}", None, None
            if line.split() and line.split()[0] in ("v_add_f64", "v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64", "v_div_fmas_f64") and ("|" in line or "-v[" in line or "-s[" in line):
                return "skip", f"KNOWN_DIVERGENCE double-neg-abs: {KNOWN_DIVERGENCES['double-neg-abs']}", None, None
    try: expected = ref_program(lines, include_known)
    except SkipCase as exc: return "skip", str(exc), None, None
    except (ArithmeticError, IndexError, KeyError, ValueError) as exc:
        return "skip", f"reference limitation: {type(exc).__name__}: {exc}", None, None
    if oracle == "gpu" and any(line.lstrip().startswith(("v_rsq_f64 ", "v_sqrt_f64 ")) for line in lines):
        return "skip", "GLSL.std.450 InverseSqrt/Sqrt double-precision result precision is implementation-defined", expected, None
    if oracle == "hw":
        if any(line.lstrip().startswith(("v_rcp_f64 ", "v_rsq_f64 ", "v_sqrt_f64 ")) for line in lines):
            return "skip", ("native v_rcp_f64/v_rsq_f64/v_sqrt_f64 are approximations (about 26 bits, PROC-216) and "
                            "the reference is exact, so their result is not judged"), expected, None
        actual = hw_words if hw_words is not None else gpu.run([hw_kernel(lines)])[0]
        if isinstance(actual, str): return "skip", f"hardware: {actual}", expected, None
        actual = [int(x) & M32 for x in actual]
        return ("match", "", expected, actual) if equivalent(expected, actual, output_kind(lines)) else ("mismatch", "word comparison", expected, actual)
    try: actual = run_gpu(lines, name, tmp, gpu) if oracle == "gpu" else translate(lines, name, tmp)
    except (rdna2spirv.Unsupported, UnsupportedConstruct, SkipCase) as exc: return "skip", f"{type(exc).__name__}: {exc}", expected, None
    except (IndexError, KeyError, ValueError, OverflowError) as exc: return "skip", f"translator/interpreter limitation: {type(exc).__name__}: {exc}", expected, None
    return ("match", "", expected, actual) if equivalent(expected, actual, output_kind(lines)) else ("mismatch", "word comparison", expected, actual)


def shrink(lines, expected, oracle="translator", gpu=None, gpu_ctx=None, tmpdir=None, include_known=False):
    """Greedy instruction deletion minimizer, retaining only instructions needed for the mismatch."""
    current = list(lines)
    def reduce(tmp, context):
        i = 0
        while i < len(current):
            trial = current[:i] + current[i + 1:]
            try:
                status, _, _, _ = run_one(trial, "ps_f64_shrink", tmp, oracle, context, include_known)
                if status == "mismatch":
                    current[:] = trial
                    continue
            except GPUUnavailable:
                raise
            except Exception:
                pass
            i += 1
        return current
    if tmpdir is not None:
        context = gpu_ctx if gpu_ctx is not None else prepare_gpu(gpu, tmpdir) if oracle == "gpu" else None
        return reduce(tmpdir, context)
    with tempfile.TemporaryDirectory(prefix="f64-shrink-") as tmp:
        context = gpu_ctx if gpu_ctx is not None else prepare_gpu(gpu, tmp) if oracle == "gpu" else None
        return reduce(tmp, context)


def run_batch(ops, programs, seed, max_ops=1, sweep=True, allow_literal=True, known=False, all_failures=False,
              oracle="translator", gpu=None, hw=None):
    counts = Counter(); skips = Counter(); failures = []
    rng = random.Random(seed)
    sequence = [(op, True, None) for op in ops] if sweep else []
    sequence += [(rng.choice(ops), False, None) for _ in range(programs)]
    if known:
        sequence += [("v_div_fixup_f64" if label == "div-fixup-nan-s0" else "v_add_f64", True, label)
                     for label in KNOWN_DIVERGENCES]
    cases = []
    for idx, (op, is_sweep, known_case) in enumerate(sequence):
        setup_lines, body_lines, _, _ = make_case(rng, op, allow_literal, known_case)
        lines = setup_lines + body_lines
        if not is_sweep and not known_case:
            n_ops = rng.randint(1, max_ops)
            for _ in range(n_ops - 1):
                _, extra_body, _, _ = make_case(rng, rng.choice(ops), allow_literal)
                lines.extend(extra_body)
        cases.append((idx, lines, known_case))
    # One launch for the whole campaign; a failing program is re-run alone while it is shrunk.
    hw_results = hw.run([hw_kernel(lines) for _, lines, _ in cases]) if oracle == "hw" else None
    with tempfile.TemporaryDirectory(prefix="f64-diff-") as tmp:
        gpu_ctx = prepare_gpu(gpu, tmp) if oracle == "gpu" else hw if oracle == "hw" else None
        for idx, lines, known_case in cases:
            status, reason, want, got = run_one(lines, f"ps_f64_{idx:05d}", tmp, oracle, gpu_ctx,
                                                include_known=bool(known_case),
                                                hw_words=hw_results[idx] if hw_results is not None else None)
            if status == "skip": skips[reason] += 1
            else: counts[status] += 1
            if status == "mismatch":
                reduced = shrink(lines, want, oracle, gpu, gpu_ctx, tmp, bool(known_case))
                failures.append((reduced, want, got, reason))
            if status == "mismatch" and not all_failures: break
    return counts, skips, failures


def self_test():
    tests = [round_fraction(Fraction(1, 3) + Fraction(2, 3)) == 0x3FF0000000000000,
             round_fraction(Fraction(1, 1 << 1075)) == 0,
             add_bits(0x3FF0000000000000, 0x3CA0000000000000) == 0x3FF0000000000000,
             class_index(0x7FF0000000000001) == 0,
             class_index(0xFFF8000000000000) == 1,
             compare("nlt", 0x7FF8000000000000, 0) == 1,
             round_fraction(Fraction(1) + Fraction(1, 1 << 53), 1) == 0x3FF0000000000001,
             round_fraction(-(Fraction(1) + Fraction(1, 1 << 53)), 2) == 0xBFF0000000000001,
             round_fraction(Fraction(1 << 1024), 3) == 0x7FEFFFFFFFFFFFFF,
             len(BASE) == 47]
    for i, ok in enumerate(tests, 1): print(f"{'ok' if ok else 'FAIL'} reference invariant {i}")
    # A planted output error must be reported and remain observable after shrinking.
    lines = ["v_mov_b32 v12, 0x00000000", "v_mov_b32 v13, 0x3ff80000",
             "v_trunc_f64 v[8:9], v[12:13]"]
    original = translate
    def planted_bug(listing, name, tmp):
        out = original(listing, name, tmp)
        if any(line.startswith("v_trunc_f64 ") for line in listing): out[0] ^= 1
        return out
    try:
        globals()["translate"] = planted_bug
        with tempfile.TemporaryDirectory(prefix="f64-self-test-") as tmp:
            status, _, expected, actual = run_one(lines, "ps_f64_self_test_negative", tmp)
        reduced = shrink(lines, expected)
        caught = (status == "mismatch" and not equivalent(expected, actual, output_kind(lines)) and len(reduced) == 1
                  and reduced[0].startswith("v_trunc_f64 "))
        print(f"{'ok' if caught else 'FAIL'} planted translator mismatch is caught and shrinkable")
    finally:
        globals()["translate"] = original
    return all(tests) and caught


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", dest="sweep", action="store_true", default=True)
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=1)
    ap.add_argument("--ops")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--no-literals", action="store_true")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--all-failures", action="store_true")
    ap.add_argument("--oracle", choices=("translator", "gpu", "hw"), default="translator",
                    help="translator (default), Vulkan GPU via vk_harness, or AMD hardware oracle")
    ap.add_argument("--exe", help="vk_harness executable for --oracle gpu")
    ap.add_argument("--hw-arch", help="GPU architecture hw_oracle assembles for (default: detected)")
    args = ap.parse_args(argv)
    if args.programs < 0 or args.max_ops < 1: print("programs must be nonnegative and max-ops positive", file=sys.stderr); return 2
    ops = [x.strip() for x in args.ops.split(",")] if args.ops else list(BASE)
    bad = sorted(set(ops) - set(BASE))
    if bad: print(f"unknown opcode(s): {','.join(bad)}", file=sys.stderr); return 2
    if args.list_ops:
        for op in BASE: print(op)
        print("Excluded (not translated by main): v_mul_f64, v_min_f64, v_max_f64, v_fract_f64, v_cvt_f64_f32, v_ldexp_f64, v_mad_f64, v_cvt_i32_f64, v_cvt_u32_f64, v_div_scale_f64")
        return 0
    if args.self_test: return 0 if self_test() else 1
    gpu = hw = None
    if args.oracle == "hw":
        try: hw = open_hw_oracle(args.hw_arch)
        except HWUnavailable as exc:
            print(f"hardware oracle unavailable: {exc}", file=sys.stderr); return 77
        print(f"hardware oracle: {getattr(hw, 'device', 'AMD GPU')}")
    if args.oracle == "gpu":
        ext = ".exe" if os.name == "nt" else ""
        gpu = args.exe or os.path.join(ROOT, "runtime", "build", "vk_harness" + ext)
    try:
        counts, skips, failures = run_batch(ops, args.programs, args.seed, args.max_ops, args.sweep,
                                            not args.no_literals, args.known, args.all_failures,
                                            args.oracle, gpu, hw)
    except GPUUnavailable as exc:
        print(f"GPU oracle unavailable: {exc}", file=sys.stderr); return 77
    except (HarnessError, RuntimeError) as exc:
        print(f"harness error: {exc}", file=sys.stderr); return 2
    finally:
        if hw is not None: hw.close()
    total = sum(counts.values()) + sum(skips.values())
    print(f"{args.oracle}: {counts['match']} match, {counts['mismatch']} mismatch; {total} total, {dict(skips)} skips")
    if failures:
        for lines, want, got, reason in failures:
            print(f"MISMATCH: {reason}\nminimal program:")
            print("\n".join("    " + x for x in lines))
            if want is not None and got is not None:
                print("expected: " + " ".join(f"{x:08x}" for x in want))
                print("actual:   " + " ".join(f"{x:08x}" for x in got))
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
