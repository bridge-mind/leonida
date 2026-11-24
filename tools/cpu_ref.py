"""CPU reference for a small, straight-line RDNA2 subset.

This is a reference model for differential shader validation, not a recompiler. It executes the
instructions listed below from a listing produced by tools/shader_disasm.py (the same ``.s`` format
that tools/rdna2spirv.py consumes). Anything outside the subset raises Unsupported with the mnemonic
and PC; it is never guessed.

Model:
  * SGPRs are 32-bit wave-uniform values.
  * VGPRs are one 32-bit value per lane.
  * VCC, EXEC and SCC are modelled; vector writes are masked by EXEC.
  * One lane or 64 lanes. One-lane mode is a view of one hardware lane: scalar reads of
    exec_lo/exec_hi and writes of the full masks still use the 64-bit wave state.
  * Float registers keep raw bits. f32 add/sub/mul use IEEE binary32. FMA is evaluated exactly with
    rational arithmetic and rounded once, so it does not inherit a host double-rounding error.
  * NaNs are canonicalized to 0x7fc00000. Denormals are preserved.

Implemented: s_mov_b32, v_mov_b32, v_add_f32, v_sub_f32, v_mul_f32, v_fma_f32, v_cndmask_b32,
v_and_b32, v_or_b32, v_xor_b32, v_lshlrev_b32, v_lshrrev_b32, v_ashrrev_i32, s_add_u32,
s_lshl_b32, s_lshr_b32, s_endpgm. Also the integer/bit VALU block: v_add_nc_u32, v_sub_nc_u32,
v_subrev_nc_u32, v_mul_u32_u24, v_mul_i32_i24, v_mul_lo_u32, v_mul_hi_u32/v_mul_hi_i32,
v_mad_u32_u24/v_mad_i32_i24, v_min_u32/v_max_u32, v_min_i32/v_max_i32,
v_bfe_u32/v_bfe_i32, v_bfi_b32, v_alignbit_b32, v_alignbyte_b32, v_perm_b32 and
v_med3_i32/v_med3_u32. Inline constants and literals for those operands are handled.
``clamp`` is accepted on the f32 ops (NaN -> 0). Registers not initialized by the caller read as
zero, matching the translator's deterministic register arrays.

Not modelled: branches and loops, memory/LDS/images, wave operations, 16-bit arithmetic, SDWA/DPP,
and any modifier except ``clamp``. An explicit 32-bit SGPR condition on v_cndmask is a 32-bit lane
mask, so lanes 32-63 read zero; VOP2's implicit VCC uses the full 64-bit mask.

A listing must start with the usual header line (``; name ...``): shader_ir.parse_file treats the
first line as a header and parses tab-indented instruction lines after it.

API:
    run(listing_path, inputs) -> {
        "lanes": 1 | 64,
        "lane": <hardware lane, one-lane mode only>,
        "active": <bool, one-lane mode only>,
        "registers": {
            "sgpr": {index: u32, ...},
            "vgpr": {index: u32} or {index: [u32, ...]},   # 64-lane mode: one list per VGPR
            "vcc": u64, "exec": u64, "scc": 0 | 1,
        },
        "steps": int, "pc": int, "halted": bool,
        "source": listing name,
    }

inputs:
    lanes     1 (default) or 64
    lane      hardware lane for one-lane mode (default 0)
    sgpr      {index: u32}
    vgpr      {index: u32} or {index: [u32, ...]}; a list must have one entry per lane
    vcc       u64 lane mask (default 0)
    exec      u64 lane mask (default all ones)
    scc       0 or 1 (default 0)
    max_steps instruction issue limit (default 100000)

The result's register maps contain the registers that were initialized or written. In one-lane mode
a VGPR value is the selected lane's value; VCC/EXEC remain full 64-bit masks.

CLI:
    python -I tools/cpu_ref.py <listing.s> [--lanes 64] [--parse-only] [--json]
    python -I tools/cpu_ref.py <listing.s> --sgpr 0=0x3f800000 --vgpr 1=0x40000000
Usage: `python -I tools/cpu_ref.py --help` lists the options.
"""
import argparse
import json
import math
import os
import struct
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shader_ir import parse_file  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
SIGN32 = 0x80000000
QNAN = 0x7FC00000
INF = 0x7F800000

FLOAT_OPS = {"v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32"}
SUPPORTED_OPS = {
    "s_mov_b32", "s_add_u32", "s_lshl_b32", "s_lshr_b32",
    "v_mov_b32", "v_cndmask_b32", "v_and_b32", "v_or_b32", "v_xor_b32",
    "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
    "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_mul_u32_u24", "v_mul_i32_i24",
    "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_hi_i32", "v_mad_u32_u24", "v_mad_i32_i24",
    "v_min_u32", "v_max_u32", "v_min_i32", "v_max_i32",
    "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_alignbit_b32", "v_alignbyte_b32",
    "v_perm_b32", "v_med3_i32", "v_med3_u32",
    "v_not_b32", "v_bfrev_b32", "v_ffbl_b32", "v_ffbh_u32", "v_ffbh_i32",
    "v_xnor_b32", "v_bcnt_u32_b32",
    "v_add3_u32", "v_lshl_add_u32", "v_add_lshl_u32", "v_lshl_or_b32", "v_and_or_b32", "v_or3_b32", "v_xad_u32",
    "v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32",
    "s_endpgm",
}
SPECIAL_WRITE = {"vcc", "vcc_lo", "vcc_hi", "exec", "exec_lo", "exec_hi", "scc", "null"}


def _ffbh_i32(a):
    if a in (0, MASK32):
        return MASK32
    return 32 - (~a & MASK32).bit_length() if a & SIGN32 else 32 - a.bit_length()


_INT_VOP = {
    "v_not_b32": (1, lambda a: ~a),
    "v_bfrev_b32": (1, lambda a: int(f"{a:032b}"[::-1], 2)),
    "v_ffbl_b32": (1, lambda a: MASK32 if a == 0 else (a & -a).bit_length() - 1),
    "v_ffbh_u32": (1, lambda a: MASK32 if a == 0 else 32 - a.bit_length()),
    "v_ffbh_i32": (1, _ffbh_i32),
    "v_xnor_b32": (2, lambda a, b: ~(a ^ b)),
    "v_bcnt_u32_b32": (2, lambda a, b: a.bit_count() + b),
    "v_add3_u32": (3, lambda a, b, c: a + b + c),
    "v_lshl_add_u32": (3, lambda a, b, c: (a << (b & 31)) + c),
    "v_add_lshl_u32": (3, lambda a, b, c: ((a + b) & MASK32) << (c & 31)),
    "v_lshl_or_b32": (3, lambda a, b, c: (a << (b & 31)) | c),
    "v_and_or_b32": (3, lambda a, b, c: (a & b) | c),
    "v_or3_b32": (3, lambda a, b, c: a | b | c),
    "v_xad_u32": (3, lambda a, b, c: (a ^ b) + c),
}


class CpuRefError(Exception):
    """Base class for a program the model cannot execute as written."""


class Unsupported(CpuRefError):
    def __init__(self, op, pc=None, reason=""):
        self.op, self.pc, self.reason = op, pc, reason
        where = f" at pc 0x{pc:x}" if pc is not None else ""
        why = f": {reason}" if reason else ""
        super().__init__(f"unsupported instruction {op}{where}{why}")


class ProgramError(CpuRefError):
    pass


# ---- float helpers ---------------------------------------------------------------------------

def bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def float_to_bits(value):
    """Round a Python float to binary32 and return its bits. NaN is canonicalized; overflow is inf."""
    if math.isnan(value):
        return QNAN
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if value < 0 else INF


def _is_nan_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def _is_inf_bits(bits):
    return (bits & 0x7FFFFFFF) == INF


def _is_zero_bits(bits):
    return (bits & 0x7FFFFFFF) == 0


def _fraction_to_f32(value, zero_sign=0):
    """Round an exact Fraction to binary32 (round to nearest, ties to even)."""
    if value == 0:
        return (zero_sign & 1) << 31
    sign = 1 if value < 0 else 0
    a = -value if sign else value

    # Find e with 2**e <= a < 2**(e+1). The loops run at most once or twice after the estimate.
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a < Fraction(2) ** e:
        e -= 1
    while a >= Fraction(2) ** (e + 1):
        e += 1

    if e >= -126:
        scaled = a / (Fraction(2) ** (e - 23))
        m = _round_half_even(scaled)
        if m >= 1 << 24:
            m >>= 1
            e += 1
        if e > 127:
            return (sign << 31) | INF
        return (sign << 31) | ((e + 127) << 23) | (m & 0x7FFFFF)

    # Subnormal: units of 2**-149.
    m = _round_half_even(a * (Fraction(2) ** 149))
    if m == 0:
        return sign << 31
    if m >= 1 << 23:
        return (sign << 31) | (1 << 23)  # rounded up to the smallest normal
    return (sign << 31) | m


def _round_half_even(value):
    """Round a non-negative Fraction to the nearest integer, ties to even."""
    q, r = divmod(value.numerator, value.denominator)
    twice = 2 * r
    if twice > value.denominator or (twice == value.denominator and (q & 1)):
        q += 1
    return q


def add_f32_bits(a, b):
    return float_to_bits(bits_to_float(a) + bits_to_float(b))


def sub_f32_bits(a, b):
    return float_to_bits(bits_to_float(a) - bits_to_float(b))


def mul_f32_bits(a, b):
    return float_to_bits(bits_to_float(a) * bits_to_float(b))


def fma_f32_bits(a, b, c):
    """Exact fused multiply-add, rounded once to binary32.

    The product of two binary32 values always fits in the exact rational. Adding the third operand
    and rounding once is the ISA's FMA; rounding a host double first can lose a tie-breaking addend.
    """
    if _is_nan_bits(a) or _is_nan_bits(b) or _is_nan_bits(c):
        return QNAN
    a_inf, b_inf, c_inf = _is_inf_bits(a), _is_inf_bits(b), _is_inf_bits(c)
    a_zero, b_zero = _is_zero_bits(a), _is_zero_bits(b)

    if (a_inf and b_zero) or (b_inf and a_zero):
        return QNAN
    product_inf = a_inf or b_inf
    product_sign = (a >> 31) ^ (b >> 31)
    if product_inf:
        if c_inf and (c >> 31) != product_sign:
            return QNAN
        return (product_sign << 31) | INF
    if c_inf:
        return c

    exact = Fraction(*bits_to_float(a).as_integer_ratio()) * \
        Fraction(*bits_to_float(b).as_integer_ratio()) + \
        Fraction(*bits_to_float(c).as_integer_ratio())
    if exact == 0:
        # IEEE FMA zero sign: the product's sign when both terms are zero with that sign, otherwise
        # +0 for cancellation under round-to-nearest.
        if a_zero or b_zero:
            psign = product_sign
            if _is_zero_bits(c) and (c >> 31) == psign:
                return psign << 31
        return 0
    return _fraction_to_f32(exact)


def clamp_f32_bits(bits):
    """AMD clamp: min(max(x, 0), 1); a NaN clamps to 0."""
    if _is_nan_bits(bits):
        return 0
    value = bits_to_float(bits)
    if value < 0.0:
        return 0
    if value > 1.0:
        return float_to_bits(1.0)
    return bits


# ---- listing helpers -------------------------------------------------------------------------

def base_op(op):
    """Drop the VOP encoding suffix. Only _e32/_e64 are transparent here; SDWA/DPP are not modelled."""
    for suffix in ("_e32", "_e64"):
        if op.endswith(suffix):
            return op[: -len(suffix)]
    return op


def check_supported(ins):
    """Return None if the mnemonic and modifiers are in the modelled subset, else a printable reason.

    Operand kinds and arity are checked when the instruction executes, not here.
    """
    op = ins["op"]
    if op == "s_endpgm":
        return None
    base = base_op(op)
    if base not in SUPPORTED_OPS:
        return f"{op} is not modelled"
    mods = ins.get("mods") or []
    allowed = {"clamp"} if base in FLOAT_OPS else set()
    bad = [m for m in mods if m not in allowed]
    if bad:
        return f"modifier(s) {', '.join(bad)} are not modelled"
    return None


# ---- machine ---------------------------------------------------------------------------------

class Machine:
    def __init__(self, info, inputs=None):
        inputs = {} if inputs is None else dict(inputs)
        allowed = {"lanes", "lane", "sgpr", "vgpr", "vcc", "exec", "scc", "max_steps"}
        unknown = sorted(set(inputs) - allowed)
        if unknown:
            raise ValueError("unknown input key(s): " + ", ".join(unknown))

        self.info = info
        self.insns = info["insns"]
        self.lanes = int(inputs.get("lanes", 1))
        if self.lanes not in (1, 64):
            raise ValueError("lanes must be 1 or 64")
        self.lane = int(inputs.get("lane", 0))
        if self.lanes == 64 and "lane" in inputs:
            raise ValueError("lane is only valid in one-lane mode")
        if not 0 <= self.lane < 64:
            raise ValueError("lane must be in 0..63")

        self.sgpr = {}
        for k, v in (inputs.get("sgpr") or {}).items():
            self.sgpr[int(k)] = int(v) & MASK32
        self.vgpr = {}
        for k, v in (inputs.get("vgpr") or {}).items():
            k = int(k)
            if isinstance(v, (list, tuple)):
                if len(v) != self.lanes:
                    raise ValueError(f"vgpr[{k}] has {len(v)} values, expected {self.lanes}")
                self.vgpr[k] = [int(x) & MASK32 for x in v]
            else:
                self.vgpr[k] = [int(v) & MASK32] * self.lanes

        self.vcc = int(inputs.get("vcc", 0)) & MASK64
        self.exec = int(inputs.get("exec", MASK64)) & MASK64
        self.scc = 1 if inputs.get("scc", 0) else 0
        self.max_steps = int(inputs.get("max_steps", 100000))
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.steps = 0
        self.pc = info.get("code_start", 0)
        self.next_pc = self.pc
        self.halted = False

    # -- register access --
    def _hw_lane(self, lane):
        return self.lane if self.lanes == 1 else lane

    def _vgpr(self, index, lane):
        values = self.vgpr.get(index)
        return 0 if values is None else values[lane]

    def _write_vgpr(self, dst, lane, value):
        if dst["k"] != "vgpr":
            raise Unsupported(dst.get("t", "destination"), self.pc, "expected a VGPR destination")
        if dst.get("n", 1) != 1:
            raise Unsupported("v[%d:%d]" % (dst["r"], dst["r"] + dst.get("n", 1) - 1), self.pc,
                              "multi-dword VGPR destination is not modelled")
        self.vgpr.setdefault(dst["r"], [0] * self.lanes)[lane] = value & MASK32

    def _read_special(self, name):
        if name == "scc":
            return self.scc
        if name in ("vcc", "vcc_lo"):
            return self.vcc & MASK32
        if name == "vcc_hi":
            return (self.vcc >> 32) & MASK32
        if name in ("exec", "exec_lo"):
            return self.exec & MASK32
        if name == "exec_hi":
            return (self.exec >> 32) & MASK32
        if name == "null":
            return 0
        raise Unsupported(name, self.pc, "special register is not modelled")

    def _write_scalar_dst(self, dst, value):
        value &= MASK32
        if dst["k"] == "sgpr":
            if dst.get("n", 1) != 1:
                raise Unsupported("s[%d:%d]" % (dst["r"], dst["r"] + dst.get("n", 1) - 1), self.pc,
                                  "multi-dword SGPR destination is not modelled")
            self.sgpr[dst["r"]] = value
            return
        if dst["k"] != "special" or dst["t"] not in SPECIAL_WRITE:
            raise Unsupported(dst.get("t", "destination"), self.pc, "expected an SGPR or special destination")
        name = dst["t"]
        if name == "scc":
            self.scc = 1 if value else 0
        elif name == "null":
            return
        elif name in ("vcc", "vcc_lo"):
            self.vcc = (self.vcc & ~MASK32) | value
        elif name == "vcc_hi":
            self.vcc = (self.vcc & MASK32) | (value << 32)
        elif name in ("exec", "exec_lo"):
            self.exec = (self.exec & ~MASK32) | value
        elif name == "exec_hi":
            self.exec = (self.exec & MASK32) | (value << 32)

    def _int_src(self, arg, lane, scalar=False):
        kind = arg["k"]
        if kind == "vgpr":
            if scalar:
                raise Unsupported("vgpr source", self.pc, "a scalar instruction cannot read a VGPR directly")
            value = self._vgpr(arg["r"], lane)
        elif kind == "sgpr":
            if arg.get("n", 1) != 1:
                raise Unsupported("s[%d:%d]" % (arg["r"], arg["r"] + arg.get("n", 1) - 1), self.pc,
                                  "multi-dword SGPR source is not modelled")
            value = self.sgpr.get(arg["r"], 0)
        elif kind == "imm":
            value = arg["v"] & MASK32
        elif kind == "fimm":
            value = float_to_bits(arg["v"])
        elif kind == "special":
            value = self._read_special(arg["t"])
        else:
            raise Unsupported(arg.get("t", kind), self.pc, "operand kind is not modelled")
        if arg.get("abs"):
            value &= 0x7FFFFFFF
        if arg.get("neg"):
            value ^= SIGN32
        return value & MASK32

    def _float_src(self, arg, lane):
        kind = arg["k"]
        if kind == "vgpr":
            bits = self._vgpr(arg["r"], lane)
        elif kind == "sgpr":
            if arg.get("n", 1) != 1:
                raise Unsupported("s[%d:%d]" % (arg["r"], arg["r"] + arg.get("n", 1) - 1), self.pc,
                                  "multi-dword SGPR source is not modelled")
            bits = self.sgpr.get(arg["r"], 0)
        elif kind == "imm":
            bits = arg["v"] & MASK32  # integer inline constants are bit patterns; float ones parse as "fimm"
        elif kind == "fimm":
            bits = float_to_bits(arg["v"])
        elif kind == "special":
            bits = self._read_special(arg["t"])
        else:
            raise Unsupported(arg.get("t", kind), self.pc, "operand kind is not modelled")
        if arg.get("abs"):
            bits &= 0x7FFFFFFF
        if arg.get("neg"):
            bits ^= SIGN32
        return bits

    def _mask_bit(self, arg, lane):
        kind = arg["k"]
        if kind == "sgpr":
            low = self.sgpr.get(arg["r"], 0)
            if arg.get("n", 1) >= 2:
                mask = (self.sgpr.get(arg["r"] + 1, 0) << 32) | low
            else:
                mask = low
        elif kind == "special":
            name = arg["t"]
            if name in ("vcc", "vcc_lo"):
                mask = self.vcc if name == "vcc" else self.vcc & MASK32
            elif name == "vcc_hi":
                mask = self.vcc >> 32
            elif name == "exec":
                mask = self.exec
            elif name == "exec_lo":
                mask = self.exec & MASK32
            elif name == "exec_hi":
                mask = self.exec >> 32
            elif name == "scc":
                mask = self.scc
            elif name == "null":
                mask = 0
            else:
                raise Unsupported(name, self.pc, "condition register is not modelled")
        elif kind == "imm":
            mask = arg["v"] & MASK32
            if arg["v"] < 0:
                mask |= MASK32 << 32
        elif kind == "fimm":
            mask = float_to_bits(arg["v"])
        else:
            raise Unsupported(arg.get("t", kind), self.pc, "condition operand is not modelled")
        return (mask >> lane) & 1

    def _active_lanes(self):
        if self.lanes == 1:
            lanes = [0]
        else:
            lanes = range(64)
        return [lane for lane in lanes if (self.exec >> self._hw_lane(lane)) & 1]

    # -- execution --
    def run(self):
        if not self.insns:
            raise ProgramError("listing has no instructions")
        for ins in self.insns:
            if self.halted:
                break
            if self.steps >= self.max_steps:
                raise ProgramError(f"step limit {self.max_steps} reached at pc 0x{ins['pc']:x}")
            self.steps += 1
            self.step(ins)
            enc = ins.get("enc")
            self.next_pc = ins["pc"] + (len(enc) // 2 if enc else 0)
            self.pc = self.next_pc
        if not self.halted:
            raise ProgramError("listing ended without s_endpgm")
        return self.result()

    def step(self, ins):
        op = ins["op"]
        reason = check_supported(ins)
        if reason:
            raise Unsupported(op, ins["pc"], reason)
        if op == "s_endpgm":
            self.halted = True
            return
        base = base_op(op)
        args = ins["args"]
        clamp = "clamp" in (ins.get("mods") or [])

        if base == "v_mov_b32":
            self._expect(ins, 2)
            for lane in self._active_lanes():
                self._write_vgpr(args[0], lane, self._int_src(args[1], lane))
        elif base in FLOAT_OPS:
            self._expect(ins, 4 if base == "v_fma_f32" else 3)
            for lane in self._active_lanes():
                a = self._float_src(args[1], lane)
                b = self._float_src(args[2], lane)
                if base == "v_add_f32":
                    r = add_f32_bits(a, b)
                elif base == "v_sub_f32":
                    r = sub_f32_bits(a, b)
                elif base == "v_mul_f32":
                    r = mul_f32_bits(a, b)
                else:
                    r = fma_f32_bits(a, b, self._float_src(args[3], lane))
                if clamp:
                    r = clamp_f32_bits(r)
                self._write_vgpr(args[0], lane, r)
        elif base == "v_cndmask_b32":
            if len(args) not in (3, 4):
                raise Unsupported(op, ins["pc"], f"expected 3 or 4 operands, got {len(args)}")
            cond = args[3] if len(args) == 4 else None
            # VOP2 cndmask has an implicit VCC condition; LLVM prints it as vcc_lo, but on a wave64
            # the hardware operand is the full VCC. VOP3 (_e64) names a 32-bit half explicitly.
            implicit = cond is None or (cond["k"] == "special" and cond["t"] in ("vcc", "vcc_lo")
                                        and not op.endswith("_e64"))
            for lane in self._active_lanes():
                if implicit:
                    bit = (self.vcc >> self._hw_lane(lane)) & 1
                else:
                    bit = self._mask_bit(cond, self._hw_lane(lane))
                src0 = self._int_src(args[1], lane)
                src1 = self._int_src(args[2], lane)
                self._write_vgpr(args[0], lane, src1 if bit else src0)
        elif base in ("v_and_b32", "v_or_b32", "v_xor_b32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                a = self._int_src(args[1], lane)
                b = self._int_src(args[2], lane)
                r = a & b if base == "v_and_b32" else (a | b if base == "v_or_b32" else a ^ b)
                self._write_vgpr(args[0], lane, r)
        elif base in ("v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                shift = self._int_src(args[1], lane) & 31
                value = self._int_src(args[2], lane)
                if base == "v_lshlrev_b32":
                    r = (value << shift) & MASK32
                elif base == "v_lshrrev_b32":
                    r = value >> shift
                else:
                    signed = value - (1 << 32) if value & SIGN32 else value
                    r = (signed >> shift) & MASK32
                self._write_vgpr(args[0], lane, r)
        elif base in ("v_add_nc_u32", "v_sub_nc_u32", "v_mul_u32_u24", "v_mul_lo_u32",
                      "v_min_u32", "v_max_u32", "v_min_i32", "v_max_i32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                a = self._int_src(args[1], lane)
                b = self._int_src(args[2], lane)
                if base == "v_add_nc_u32":
                    r = (a + b) & MASK32
                elif base == "v_sub_nc_u32":
                    r = (a - b) & MASK32
                elif base == "v_mul_u32_u24":
                    r = ((a & 0xFFFFFF) * (b & 0xFFFFFF)) & MASK32
                elif base == "v_mul_lo_u32":
                    r = (a * b) & MASK32
                elif base in ("v_min_u32", "v_max_u32"):
                    r = (max if base == "v_max_u32" else min)(a, b)
                else:
                    sa = a - (1 << 32) if a & SIGN32 else a
                    sb = b - (1 << 32) if b & SIGN32 else b
                    r = (max if base == "v_max_i32" else min)(sa, sb) & MASK32
                self._write_vgpr(args[0], lane, r)
        elif base == "v_subrev_nc_u32":
            self._expect(ins, 3)
            for lane in self._active_lanes():
                a = self._int_src(args[1], lane)
                b = self._int_src(args[2], lane)
                self._write_vgpr(args[0], lane, (b - a) & MASK32)
        elif base in ("v_mul_hi_u32", "v_mul_hi_i32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                a = self._int_src(args[1], lane)
                b = self._int_src(args[2], lane)
                if base == "v_mul_hi_i32":
                    a = a - (1 << 32) if a & SIGN32 else a
                    b = b - (1 << 32) if b & SIGN32 else b
                self._write_vgpr(args[0], lane, ((a * b) >> 32) & MASK32)
        elif base in ("v_mad_u32_u24", "v_mad_i32_i24"):
            self._expect(ins, 4)
            for lane in self._active_lanes():
                x = self._int_src(args[1], lane) & 0xFFFFFF
                y = self._int_src(args[2], lane) & 0xFFFFFF
                if base == "v_mad_i32_i24":
                    x = x - (1 << 24) if x & 0x800000 else x
                    y = y - (1 << 24) if y & 0x800000 else y
                z = self._int_src(args[3], lane)
                self._write_vgpr(args[0], lane, (x * y + z) & MASK32)
        elif base == "v_mul_i32_i24":
            self._expect(ins, 3)
            for lane in self._active_lanes():
                x = self._int_src(args[1], lane) & 0xFFFFFF
                y = self._int_src(args[2], lane) & 0xFFFFFF
                sx = x - (1 << 24) if x & 0x800000 else x
                sy = y - (1 << 24) if y & 0x800000 else y
                self._write_vgpr(args[0], lane, (sx * sy) & MASK32)
        elif base in ("v_bfe_u32", "v_bfe_i32"):
            self._expect(ins, 4)
            for lane in self._active_lanes():
                value = self._int_src(args[1], lane)
                off = self._int_src(args[2], lane) & 31
                width = min(self._int_src(args[3], lane) & 31, 32 - off)
                if width == 0:
                    r = 0
                else:
                    r = (value >> off) & ((1 << width) - 1)
                    if base == "v_bfe_i32" and r & (1 << (width - 1)):
                        r = (r - (1 << width)) & MASK32
                self._write_vgpr(args[0], lane, r)
        elif base == "v_bfi_b32":
            self._expect(ins, 4)
            for lane in self._active_lanes():
                x = self._int_src(args[1], lane)
                y = self._int_src(args[2], lane)
                z = self._int_src(args[3], lane)
                self._write_vgpr(args[0], lane, (x & y) | (~x & z))
        elif base in ("v_alignbit_b32", "v_alignbyte_b32"):
            self._expect(ins, 4)
            for lane in self._active_lanes():
                hi = self._int_src(args[1], lane)
                lo = self._int_src(args[2], lane)
                shift = self._int_src(args[3], lane)
                shift = (shift & 31) if base == "v_alignbit_b32" else (shift & 3) * 8
                self._write_vgpr(args[0], lane, (((hi << 32) | lo) >> shift) & MASK32)
        elif base == "v_perm_b32":
            self._expect(ins, 4)
            for lane in self._active_lanes():
                hi = self._int_src(args[1], lane)
                lo = self._int_src(args[2], lane)
                sel = self._int_src(args[3], lane)
                packed = (hi << 32) | lo
                r = 0
                for i in range(4):
                    sb = (sel >> (8 * i)) & 0xFF
                    if sb < 8:
                        val = (packed >> (8 * (sb & 7))) & 0xFF
                    elif sb < 12:
                        k = sb & 3
                        src = lo if k < 2 else hi
                        # Selector 8/9 pick bit 7 of bytes 1/3 of src1 (bits 15/31);
                        # 10/11 the same bytes of src0 (bits 47/63, i.e. 15/31 again).
                        val = 0xFF if (src >> (15 + 16 * (k & 1))) & 1 else 0
                    else:
                        val = 0 if sb == 12 else 0xFF
                    r |= val << (8 * i)
                self._write_vgpr(args[0], lane, r)
        elif base in ("v_med3_i32", "v_med3_u32"):
            self._expect(ins, 4)
            for lane in self._active_lanes():
                vals = [self._int_src(args[i], lane) for i in (1, 2, 3)]
                if base == "v_med3_i32":
                    vals = [v - (1 << 32) if v & SIGN32 else v for v in vals]
                x, y, z = vals
                self._write_vgpr(args[0], lane, max(min(x, y), min(max(x, y), z)) & MASK32)
        elif base in _INT_VOP:
            arity, fn = _INT_VOP[base]
            self._expect(ins, 1 + arity)
            for lane in self._active_lanes():
                vals = [self._int_src(args[i], lane) for i in range(1, 1 + arity)]
                self._write_vgpr(args[0], lane, fn(*vals) & MASK32)
        elif base == "s_mov_b32":
            self._expect(ins, 2)
            self._write_scalar_dst(args[0], self._int_src(args[1], 0, scalar=True))
        elif base == "s_add_u32":
            self._expect(ins, 3)
            a = self._int_src(args[1], 0, scalar=True)
            b = self._int_src(args[2], 0, scalar=True)
            total = a + b
            self.scc = 1 if total >> 32 else 0
            self._write_scalar_dst(args[0], total & MASK32)
        elif base in ("s_lshl_b32", "s_lshr_b32"):
            self._expect(ins, 3)
            value = self._int_src(args[1], 0, scalar=True)
            shift = self._int_src(args[2], 0, scalar=True) & 31
            r = ((value << shift) & MASK32) if base == "s_lshl_b32" else (value >> shift)
            self.scc = 1 if r else 0
            self._write_scalar_dst(args[0], r)
        else:  # check_supported should have caught this
            raise Unsupported(op, ins["pc"], "not modelled")

    @staticmethod
    def _expect(ins, n):
        if len(ins["args"]) != n:
            raise Unsupported(ins["op"], ins["pc"], f"expected {n} operands, got {len(ins['args'])}")

    def result(self):
        regs = {
            "sgpr": {k: self.sgpr[k] for k in sorted(self.sgpr)},
            "vgpr": {k: (self.vgpr[k][0] if self.lanes == 1 else list(self.vgpr[k]))
                     for k in sorted(self.vgpr)},
            "vcc": self.vcc,
            "exec": self.exec,
            "scc": self.scc,
        }
        out = {
            "lanes": self.lanes,
            "registers": regs,
            "steps": self.steps,
            "pc": self.pc,
            "halted": self.halted,
            "source": self.info.get("name"),
        }
        if self.lanes == 1:
            out["lane"] = self.lane
            out["active"] = bool((self.exec >> self.lane) & 1)
        return out


def _require_header(listing_path):
    """shader_ir.parse_file treats line 1 as the header; a raw llvm-mc dump would lose an instruction."""
    with open(listing_path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if first.startswith("\t"):
        raise ProgramError("listing has no header line; its first instruction would be skipped")


def run(listing_path, inputs=None):
    """Parse a .s listing and execute the modelled subset. Raises CpuRefError on anything else."""
    _require_header(listing_path)
    info = parse_file(listing_path)
    return Machine(info, inputs).run()


# ---- CLI -------------------------------------------------------------------------------------

def _value(text):
    return int(text, 0)


def _vgpr_value(text, lanes):
    if "," in text:
        values = [int(x, 0) for x in text.split(",")]
        if len(values) != lanes:
            raise argparse.ArgumentTypeError(f"expected {lanes} comma-separated values")
        return values
    return int(text, 0)


def _print_parse_report(path):
    _require_header(path)
    info = parse_file(path)
    print(f"{path}: {len(info['insns'])} instructions parsed (code_start=0x{info['code_start']:x}, "
          f"{info['invalid']} invalid words)")
    supported = 0
    first_bad = None
    for ins in info["insns"]:
        reason = check_supported(ins)
        if reason is None:
            supported += 1
            state = "supported mnemonic"
        else:
            state = "UNSUPPORTED: " + reason
            if first_bad is None:
                first_bad = (ins, reason)
        print(f"  0x{ins['pc']:04x}  {ins['op']:<24} {state}")
    print(f"{supported}/{len(info['insns'])} instructions have a modelled mnemonic and modifiers")
    if first_bad:
        ins, reason = first_bad
        print(f"first unsupported: {ins['op']} at pc 0x{ins['pc']:x} ({reason})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Execute a small RDNA2 subset from a shader_ir listing.")
    ap.add_argument("listing", help=".s listing (tools/shader_disasm.py or runtime/test/shaders/build.py)")
    ap.add_argument("--lanes", type=int, choices=(1, 64), default=1)
    ap.add_argument("--lane", type=int, default=0, help="hardware lane for one-lane mode")
    ap.add_argument("--sgpr", action="append", default=[], metavar="N=VALUE")
    ap.add_argument("--vgpr", action="append", default=[], metavar="N=VALUE[,VALUE...]")
    ap.add_argument("--vcc", type=_value, default=0)
    ap.add_argument("--exec", type=_value, default=None)
    ap.add_argument("--parse-only", action="store_true", help="report parsed and modelled instructions only")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    if not os.path.exists(args.listing):
        print(f"ERROR: {args.listing}: does not exist", file=sys.stderr)
        return 2

    inputs = {"lanes": args.lanes, "vcc": args.vcc}
    if args.lanes == 1:
        if not 0 <= args.lane < 64:
            ap.error("--lane must be in 0..63")
        inputs["lane"] = args.lane
    if args.exec is not None:
        inputs["exec"] = args.exec
    for kind, assignments, syntax in (
        ("sgpr", args.sgpr, "N=VALUE"),
        ("vgpr", args.vgpr, "N=VALUE[,VALUE...]"),
    ):
        for item in assignments:
            try:
                register, text = item.split("=", 1)
                index = _value(register)
                value = _value(text) if kind == "sgpr" else _vgpr_value(text, args.lanes)
            except (ValueError, argparse.ArgumentTypeError):
                ap.exit(2, f"error: --{kind} expects {syntax}, got {item!r}\n")
            inputs.setdefault(kind, {})[index] = value

    if args.parse_only:
        try:
            return _print_parse_report(args.listing)
        except (CpuRefError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    try:
        result = run(args.listing, inputs)
    except (CpuRefError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{args.listing}: {result['steps']} instructions, halted={result['halted']}")
    if result["lanes"] == 1:
        print(f"lane {result['lane']} active={result['active']}")
    for name, value in result["registers"]["sgpr"].items():
        print(f"  s{name} = 0x{value:08x}")
    for name, value in result["registers"]["vgpr"].items():
        if isinstance(value, list):
            print(f"  v{name} = [" + ", ".join(f"0x{x:08x}" for x in value) + "]")
        else:
            print(f"  v{name} = 0x{value:08x}")
    print(f"  vcc = 0x{result['registers']['vcc']:016x}  exec = 0x{result['registers']['exec']:016x}  "
          f"scc = {result['registers']['scc']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
