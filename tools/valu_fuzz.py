r"""Differential fuzzer for the translator, two instruction families on the GPU (FUZZ-2, FUZZ-5), standard
library only.

    python -I tools/valu_fuzz.py --family f16  --programs 200 --seed 1 [--wave 64|32|both] [--emulate] [--asm-jobs N]
    python -I tools/valu_fuzz.py --family mem  ...
    python -I tools/valu_fuzz.py --family f16 --show 1:7:32      # print program 7 of seed 1 (wave32) and its model
    python -I tools/valu_fuzz.py --family f16 --model-only ...   # generate and model only (no GPU)

Families:
  f16   16-bit and packed VALU: f16 add/sub/mul/min/max/fma/fmac/med3, conversions, v_pk_* with op_sel/op_sel_hi,
        16-bit integer ops and v_pack_b32_f16, in the encodings GFX10 has (VOP2, VOP3, op_sel where llvm-mc
        -mcpu=gfx1013 accepts it). Which forms keep the other half of the destination follows the 16-bit row of the
        contract in docs/design.md (from Mesa ACO), so a mismatch means the translator disagrees with that contract.
  mem   memory: buffer loads and stores of dword..dwordx4 (the byte and short forms are not generated) with offen,
        idxen, idxen+offen and offset-only addressing through V#s built in SGPRs with a random OOB_SELECT, stride and
        num_records (any byte count, so accesses straddle the end; per-mode rules of docs/design.md "Buffer bounds
        checking"), and s_load / s_buffer_load (dword, x2, x4) of what the stores wrote.

Oracles (--oracle translator|hw|translator,hw): translator (default) assembles for gfx1013, translates and runs
vk_compute_test. hw runs the same programs natively on the AMD GPU through tools/hw_oracle/ (wave32 and native wave64,
64 threads) and compares them with the model, so it tests the model, not the translator; it exits 77 when the tools
are missing. The forms are the ones with the same meaning on GFX10 and GFX11; what differs is listed in
docs/shader_validation.md ("16-bit and memory ops on the GPU"). --negative-control checks that wrong models are caught.

Each program is one 64-thread group (wave64, or two wave32s). Every thread stores 16 result VGPRs (v20..v35); the
mem family also dumps its scratch area. The model of each family is written here from the RDNA2 ISA and does not
import the translator. Values whose model result is NaN, infinite or denormal are marked uncertain and not
compared, so float rounding conventions can't masquerade as findings.

A mismatch is printed with the thread, register, the two values and the op that last wrote the register, plus a
--show line. --skip-op OP leaves an op out without changing the rest of the program, to set a known finding aside.
Exit code: 0 when every program that ran matches, 1 on a mismatch or an untranslated program, 2 on a harness error
or when nothing ran. Needs vk_compute_test's VKC_DUMP / VKC_OUT_WORDS (from FUZZ-6, #368). A wave32 program is
skipped where vk_compute_test can't pin 32-lane subgroups ("compute subgroup size: 32 not available", e.g. Mesa
llvmpipe at 8 lanes); wave64 programs translated with --emulate use no subgroup operations and run at any size.

Needs llvm-mc 21 or newer for --family f16 (docs/testing.md): llvm-mc 18 rejects op_sel on the 16-bit integer VOP3
forms, so no f16 program assembles and the run stops with exit code 2. --family mem assembles with 18.
Usage: `python -I tools/valu_fuzz.py --help` lists the options.
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import islice
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402
from hw_oracle import asm  # noqa: E402
from check import gpu_test_timeout, timed_out  # noqa: E402
from compute_fixture import compute_header, find_test_program, unpinned_subgroup  # noqa: E402

THREADS = 64
PERTURB = set()   # deliberately wrong models for --negative-control (see PERTURBATIONS)
POOL = list(range(20, 36))  # result registers, stored at the end
ADDR, TMP, TMP2 = 6, 7, 8
M32 = 0xFFFFFFFF
M16 = 0xFFFF
RSRC2 = 0x8088  # 4 user SGPRs (the V#), workgroup id in s4, 512 bytes of LDS: as runtime/test/shaders/build.py


# =============================================================================================== numbers

def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def f16_value(h):
    """Exact value of a finite f16 as a Fraction, or None for NaN/inf."""
    h &= M16
    sign = -1 if h & 0x8000 else 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0x1F:
        return None
    if e == 0:
        return sign * Fraction(m, 1 << 24)
    return sign * Fraction(1024 + m, 1 << 10) * (Fraction(2) ** (e - 15))


def f32_value(b):
    b &= M32
    sign = -1 if b & 0x80000000 else 1
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    if e == 0xFF:
        return None
    if e == 0:
        return sign * Fraction(m, 1 << 149)
    return sign * Fraction((1 << 23) + m, 1 << 23) * (Fraction(2) ** (e - 127))


def round_fraction(q, mant_bits, emin, emax, rtz=False):
    """Round an exact value to a binary format with `mant_bits` fraction bits, normal exponents [emin, emax].
    Returns (sign, exponent field, mantissa field, flags) where flags marks 'denorm' or 'overflow' results, which
    the fuzzer treats as uncertain."""
    if q == 0:
        return 0, 0, 0, set()
    sign = 1 if q < 0 else 0
    a = -q if q < 0 else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < emin:
        return sign, 0, 0, {"denorm"}
    scaled = a / (Fraction(2) ** (e - mant_bits))  # in [2^mant_bits, 2^(mant_bits+1))
    whole = scaled.numerator // scaled.denominator
    rem = scaled - whole
    if not rtz and (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and whole & 1)):
        whole += 1
    if whole == 1 << (mant_bits + 1):
        whole >>= 1
        e += 1
    if e > emax:
        return sign, 0, 0, {"overflow"}
    return sign, e, whole - (1 << mant_bits), set()


def to_f16(q, rtz=False):
    """(bits, uncertain) for an exact value rounded to f16 (round to nearest even, or toward zero). A zero comes back
    as +0; callers that know the IEEE sign of an exact zero set it."""
    if q is None:
        return 0x7E00, True
    if q == 0:
        return 0, False
    s, e, m, flags = round_fraction(q, 10, -14, 15, rtz)
    if flags:
        return 0x7E00, True
    return (s << 15) | ((e + 15) << 10) | m, False


def to_f32(q):
    if q is None:
        return 0x7FC00000, True
    if q == 0:
        return 0, False
    s, e, m, flags = round_fraction(q, 23, -126, 127)
    if flags:
        return 0x7FC00000, True
    return (s << 31) | ((e + 127) << 23) | m, False


def s16(x):
    x &= M16
    return x - 0x10000 if x & 0x8000 else x


# ============================================================================================ framework

class Reg:
    """A 32-bit register per thread, with an 'uncertain' flag per 16-bit half."""
    __slots__ = ("v", "ulo", "uhi")

    def __init__(self, v, ulo=False, uhi=False):
        self.v, self.ulo, self.uhi = v & M32, ulo, uhi


class Family:
    """One instruction family: generator, assembly and model. Subclasses fill these in."""
    name = ""
    short = ""
    extra_words = 0   # dumped after the per-thread registers

    def __init__(self, seed, index, wave, skip=()):
        self.rng = random.Random(f"{self.short}:{seed}:{index}:{wave}")
        self.name = f"cs_{self.short}fuzz_s{seed}_{index}_w{wave}"
        self.wave = wave
        self.skip = set(skip)
        self.ops = []
        self.build()

    # -- to implement
    def build(self):
        raise NotImplementedError

    def asm_body(self):
        raise NotImplementedError

    def model(self):
        """Returns (regs, extra): regs[t][r] = Reg, extra = list of words or uncertain None."""
        raise NotImplementedError

    def describe(self, i):
        return f"op {i}: {self.ops[i]['text']}" if i is not None and i >= 0 else "the initial value"

    # -- shared
    def asm(self):
        w64 = self.wave == 64
        out = [f"; {self.name}: tools/valu_fuzz.py ({self.title})", "; group: 64,1,1", f"; rsrc2: {RSRC2:#x}"]
        if w64:
            out.append("; mattr: +wavefrontsize64,-wavefrontsize32")
        out.append("s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1")
        out += self.asm_body()
        out.append("s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1")
        out.append(f"v_mul_u32_u24_e32 v{ADDR}, {4 * len(POOL)}, v0")
        for i, r in enumerate(POOL):
            out.append(f"buffer_store_dword v{r}, v{ADDR}, s[0:3], 0 offen offset:{4 * i}")
        out += self.asm_tail()
        out.append("s_endpgm")
        return "\n".join(out) + "\n"

    def asm_tail(self):
        return []

    def hw_lines(self, fam):
        """The program as a native kernel body for tools/hw_oracle (a raw kernel: s[2:3] is the output pointer, v0 the
        thread id, EXEC all ones). Every thread stores its result registers, the mem family also dumps its scratch."""
        store = "global_store_dword" if fam == "gfx10" else "global_store_b32"
        out = self.hw_body(fam)
        out.append(f"v_mul_u32_u24_e32 v{ADDR}, {4 * len(POOL)}, v0")
        for i, r in enumerate(POOL):
            out.append(f"{store} v{ADDR}, v{r}, s[2:3] offset:{4 * i}")
        out += self.hw_tail(fam)
        out.append("s_endpgm")
        return out

    def hw_body(self, fam):
        return self.asm_body()

    def hw_tail(self, fam):
        return []

    def out_words(self):
        return THREADS * len(POOL) + self.extra_words


# ===================================================================================== FUZZ-2: f16/packed

F16_BIN = {"v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16"}
F16_TRI = {"v_fma_f16", "v_med3_f16"}
PK_F16 = {"v_pk_add_f16", "v_pk_mul_f16", "v_pk_max_f16", "v_pk_min_f16", "v_pk_fma_f16"}
PK_SHIFT = {"v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16"}
I16_VOP3 = {"v_add_nc_u16", "v_sub_nc_u16"}
I16_BOTH = {"v_mul_lo_u16", "v_lshlrev_b16", "v_lshrrev_b16", "v_ashrrev_i16", "v_max_u16", "v_min_u16",
            "v_max_i16", "v_min_i16"}
F16_OPS = sorted(F16_BIN | F16_TRI | PK_F16 | PK_SHIFT | I16_VOP3 | I16_BOTH |
                 {"v_fmac_f16", "v_cvt_f16_f32", "v_cvt_f32_f16", "v_pack_b32_f16", "v_cvt_pkrtz_f16_f32"})


def is_f16_denorm(h):
    return (h & 0x7C00) == 0 and (h & 0x3FF) != 0


def is_f16_snan(h):
    """A signaling NaN: exponent all ones, quiet bit (9) clear, payload non-zero."""
    return (h & 0x7E00) == 0x7C00 and (h & 0x3FF) != 0


def f16_sign(h):
    return 1 if h & 0x8000 else 0


def zero_sign_add(sa, sb):
    """IEEE round-to-nearest: an exact zero sum is -0 only when both addends are -0 (x + -x is +0)."""
    return 1 if sa and sb else 0


def f16_op(name, a, b, c=None):
    """f16 arithmetic on half bit patterns: (result bits, uncertain). Denormal inputs are uncertain too: whether a
    Vulkan driver keeps fp16 denormals is a device property (shaderDenormPreserveFloat16), not a translator rule.
    Exact zero results get their IEEE sign."""
    if any(x is not None and is_f16_denorm(x) for x in (a, b, c)):
        return 0x7E00, True
    va, vb = f16_value(a), f16_value(b)
    vc = f16_value(c) if c is not None else Fraction(0)
    if va is None or vb is None or vc is None:
        return 0x7E00, True
    sa, sb = f16_sign(a), f16_sign(b)
    sc = f16_sign(c) if c is not None else 0

    def rounded(q, zero_sign):
        bits, u = to_f16(q)
        if not u and q == 0:
            bits = zero_sign << 15
        return bits, u

    if "min_is_max" in PERTURB and name == "min":
        name = "max"
    if name == "add":
        return rounded(va + vb, zero_sign_add(sa, sb) if va == 0 and vb == 0 else 0)
    if name == "sub":
        return rounded(va - vb, zero_sign_add(sa, 1 - sb) if va == 0 and vb == 0 else 0)
    if name == "mul":
        return rounded(va * vb, sa ^ sb)
    if name == "fma":
        prod_zero = va == 0 or vb == 0
        zs = zero_sign_add(sa ^ sb, sc) if prod_zero and vc == 0 else 0
        return rounded(va * vb + vc, zs)
    if name in ("max", "min"):
        if va == vb and (a & 0x8000) != (b & 0x8000):
            return 0, True  # +0 vs -0: which one min/max returns is not promised
        pick = (va >= vb) if name == "max" else (va <= vb)
        return (a if pick else b) & M16, False
    if name == "med3":
        # Half decoding loses the sign of zero. The documented med3 ordering puts -0 before +0.
        vals = sorted(((va, a), (vb, b), (vc, c)), key=lambda item: (item[0], -f16_sign(item[1])))
        return vals[1][1] & M16, False
    raise ValueError(name)


def i16_op(name, a, b):
    a &= M16
    b &= M16
    if name == "v_add_nc_u16":
        return (a + b) & M16
    if name == "v_sub_nc_u16":
        return (a - b) & M16
    if name == "v_mul_lo_u16":
        return (a * b) & M16
    if name == "v_lshlrev_b16":
        return (b << (a & 15)) & M16
    if name == "v_lshrrev_b16":
        return b >> (a & 15)
    if name == "v_ashrrev_i16":
        return (s16(b) >> (a & 15)) & M16
    if name == "v_max_u16":
        return max(a, b)
    if name == "v_min_u16":
        return min(a, b)
    if name == "v_max_i16":
        return max(a, b, key=s16)
    if name == "v_min_i16":
        return min(a, b, key=s16)
    raise ValueError(name)


# LLVM's gfx11 assembler (fake16 mode, the only one gfx1150 has) takes op_sel on v_add_nc_u16/v_sub_nc_u16 but rejects it
# on the other 16-bit integer VOP3 ops ("operands are not valid for this GPU or mode"; the instructions are VOP3 with an
# op_sel field on GFX11 as on GFX10, only the assembler has no spelling for it). With a non-zero op_sel they are emitted
# as the instruction words, so the hardware sees what a true16 compiler would write. GFX11 renumbered the three shifts
# (GFX10: 0x314, 0x315, 0x316), the others kept their opcode; the numbers are from llvm-mc -mcpu=gfx1150 and
# tests/test_valu_fuzz.py checks them against it when it is installed.
HW_NO_OPSEL = set(I16_BOTH)
GFX11_VOP3_OPCODE = {"v_mul_lo_u16": 0x305, "v_lshlrev_b16": 0x338, "v_lshrrev_b16": 0x339, "v_ashrrev_i16": 0x33A,
                     "v_max_u16": 0x309, "v_max_i16": 0x30A, "v_min_u16": 0x30B, "v_min_i16": 0x30C}


def vop3_words(opcode, vdst, srcs, opsel):
    """The two instruction dwords of a VOP3 op with VGPR operands, no abs/neg/clamp/omod (the GFX10 and GFX11 layouts
    agree on the fields used: vdst [7:0], op_sel [14:11], opcode [25:16], src0..2 9 bits each from bit 0 of dword 1)."""
    w0 = 0xD7000000 | (opcode << 16) | (opsel << 11) | vdst
    w1 = 0
    for i, v in enumerate(srcs):
        w1 |= (256 + v) << (9 * i)
    if len(srcs) < 3:
        w1 |= 128 << 18          # the unused src2 is the inline constant 0, as the assembler writes it
    return w0, w1


class F16Family(Family):
    title = "FUZZ-2"
    short = "f16"

    def nice_half(self):
        rng = self.rng
        if rng.random() < 0.2:
            return rng.choice((0x3C00, 0x4000, 0x3800, 0xBC00, 0x4200, 0x3E00))  # 1, 2, 0.5, -1, 3, 1.5
        e = rng.randrange(-3, 7) + 15   # keeps sums and differences away from denormals
        return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)

    def nice_f32(self):
        rng = self.rng
        e = rng.randrange(-3, 7) + 127
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)

    def build(self):
        rng = self.rng
        self.init = {}
        for r in POOL:
            kind = rng.random()
            if kind < 0.15:
                self.init[r] = ("f32", self.nice_f32())
            elif kind < 0.3:
                self.init[r] = ("int", rng.getrandbits(32))
            else:
                self.init[r] = ("halves", self.nice_half() | (self.nice_half() << 16))
        ops = [o for o in F16_OPS if o not in self.skip]
        for _ in range(rng.randrange(6, 17)):
            self.ops.append(self.random_op(rng.choice(ops)))

    def random_op(self, name):
        rng = self.rng
        d = rng.choice(POOL)
        src = lambda: rng.choice(POOL)  # noqa: E731
        sel = lambda n: [rng.randrange(2) for _ in range(n)]  # noqa: E731
        op = {"name": name, "d": d}
        # GFX10 encodings, checked with llvm-mc -mcpu=gfx1013/gfx1030: the f16 VOP2 ops' VOP3 forms take no op_sel
        # (only the native VOP3 ops v_fma/v_med3_f16 do), and the 16-bit integer ops exist only as op_sel VOP3.
        if name in F16_BIN:
            op["enc"] = rng.choice(("e32", "e64"))
            op["s"] = [src(), src()]
            op["opsel"] = [0, 0, 0]
        elif name in I16_BOTH or name in I16_VOP3:
            op["enc"] = "e64"
            op["s"] = [src(), src()]
            op["opsel"] = sel(3)
        elif name in F16_TRI:
            op["enc"] = "e64"
            op["s"] = [src(), src(), src()]
            op["opsel"] = sel(4)
        elif name == "v_fmac_f16":
            op["enc"] = "e32"
            op["s"] = [src(), src()]
        elif name in ("v_cvt_f16_f32", "v_cvt_f32_f16"):
            op["enc"] = "e32"
            op["s"] = [src()]
        elif name in PK_F16 or name in PK_SHIFT:
            n = 3 if name == "v_pk_fma_f16" else 2
            op["s"] = [src() for _ in range(n)]
            op["opsel"], op["opsel_hi"] = sel(n), sel(n)
        elif name == "v_pack_b32_f16":
            op["s"] = [src(), src()]
            op["opsel"] = sel(2)
        elif name == "v_cvt_pkrtz_f16_f32":
            op["enc"] = "e32"
            op["s"] = [src(), src()]
        op["text"] = self.op_text(op)
        return op

    def op_text(self, op):
        n, d, s = op["name"], op["d"], op["s"]
        vs = ", ".join(f"v{x}" for x in s)
        if op.get("enc") == "e32":
            return f"{n}_e32 v{d}, {vs}"
        if n in F16_BIN:
            return f"{n}_e64 v{d}, {vs}"
        if n in PK_F16 or n in PK_SHIFT:
            return (f"{n} v{d}, {vs} op_sel:[{','.join(map(str, op['opsel']))}]"
                    f" op_sel_hi:[{','.join(map(str, op['opsel_hi']))}]")
        if n == "v_pack_b32_f16":
            return f"{n} v{d}, {vs} op_sel:[{op['opsel'][0]},{op['opsel'][1]}]"
        return f"{n}_e64 v{d}, {vs} op_sel:[{','.join(map(str, op['opsel']))}]"

    def hw_body(self, fam):
        """The same program for a GFX11 assembler, which has no op_sel operand on most 16-bit integer ops: those with
        a non-zero op_sel are written as instruction words (see GFX11_VOP3_OPCODE)."""
        def text(op):
            if fam != "gfx11" or op["name"] not in HW_NO_OPSEL:
                return op["text"]
            if not any(op["opsel"]):
                return op["text"].split(" op_sel:")[0]
            sel = op["opsel"][0] | op["opsel"][1] << 1 | op["opsel"][2] << 3
            w0, w1 = vop3_words(GFX11_VOP3_OPCODE[op["name"]], op["d"], op["s"], sel)
            return f".long {w0:#x}\n.long {w1:#x}   // {op['text']}"
        return self.asm_body(text)

    def asm_body(self, text=lambda op: op["text"]):
        out = []
        for r, (_, v) in self.init.items():
            out.append(f"v_mov_b32_e32 v{r}, {v:#x}")
        # One register differs per thread, so thread mix-ups show: low half = a half derived from the thread id.
        out += [f"v_and_b32_e32 v{TMP}, 7, v0", f"v_lshlrev_b32_e32 v{TMP}, 7, v{TMP}",
                f"v_or_b32_e32 v{POOL[0]}, 0x3c00, v{TMP}"]   # 1.0 + thread-dependent mantissa bits
        out += [text(op) for op in self.ops]
        return out

    def model(self):
        regs = []
        writers = []
        for t in range(THREADS):
            R = {r: Reg(v) for r, (_, v) in self.init.items()}
            R[POOL[0]] = Reg(0x3C00 | ((t & 7) << 7))  # the v_or_b32 below writes the whole register
            W = {r: -1 for r in POOL}
            for i, op in enumerate(self.ops):
                self.step(op, R)
                W[op["d"]] = i
            regs.append(R)
            writers.append(W)
        self.writer = writers
        return regs, []

    def step(self, op, R):
        n, d, s = op["name"], op["d"], op["s"]
        half = lambda r, hi: ((R[r].v >> 16) if hi else R[r].v) & M16  # noqa: E731
        unc = lambda r, hi: R[r].uhi if hi else R[r].ulo  # noqa: E731
        old = R[d]

        def write_half(value, hi, uncertain):
            if "opsel_dst_ignored" in PERTURB:
                hi = 0
            if "hi_zero" in PERTURB:
                R[d] = Reg(value << 16 if hi else value, uncertain, uncertain)
            elif hi:
                R[d] = Reg((old.v & 0xFFFF) | (value << 16), old.ulo, uncertain)
            else:
                R[d] = Reg((old.v & 0xFFFF0000) | value, uncertain, old.uhi)

        if n in F16_BIN or n in F16_TRI:
            sel = op["opsel"]
            k = len(s)
            vals = [half(s[j], sel[j]) for j in range(k)]
            u = any(unc(s[j], sel[j]) for j in range(k))
            res, u2 = f16_op(n[2:-4], *vals)
            write_half(res, sel[k], u or u2)        # f16 ops keep the other half (contract: partial writes)
        elif n == "v_fmac_f16":
            res, u2 = f16_op("fma", half(s[0], 0), half(s[1], 0), half(d, 0))
            write_half(res, 0, u2 or unc(s[0], 0) or unc(s[1], 0) or unc(d, 0))
        elif n in I16_BOTH or n in I16_VOP3:
            sel = [0, 0, 0] if "i16_opsel_ignored" in PERTURB and n in HW_NO_OPSEL else op["opsel"]
            res = i16_op(n, half(s[0], sel[0]), half(s[1], sel[1]))
            write_half(res, sel[2], unc(s[0], sel[0]) or unc(s[1], sel[1]))  # op_sel VOP3: keeps the other half
        elif n == "v_cvt_f16_f32":
            x = R[s[0]].v
            res, u2 = to_f16(f32_value(x)) if (x & 0x7F800000) else (0, True)  # f32 zero/denormal: uncertain
            write_half(res, 0, u2 or R[s[0]].ulo or R[s[0]].uhi)
        elif n == "v_cvt_f32_f16":
            h = half(s[0], 0)
            v = None if is_f16_denorm(h) else f16_value(h)
            bits, u2 = to_f32(v) if v is not None else (0x7FC00000, True)
            if v == 0:
                bits = f16_sign(h) << 31
            R[d] = Reg(bits, u2 or unc(s[0], 0), u2 or unc(s[0], 0))
        elif n == "v_cvt_pkrtz_f16_f32":
            fa, fb = R[s[0]].v, R[s[1]].v
            lo, ul = to_f16(f32_value(fa), rtz=True) if fa & 0x7F800000 else (0, True)
            hi, uh = to_f16(f32_value(fb), rtz=True) if fb & 0x7F800000 else (0, True)
            R[d] = Reg(lo | (hi << 16), ul or R[s[0]].ulo or R[s[0]].uhi, uh or R[s[1]].ulo or R[s[1]].uhi)
        elif n == "v_pack_b32_f16":
            sel = op["opsel"]
            lo, hi = half(s[0], sel[0]), half(s[1], sel[1])
            # a signaling NaN half is uncertain (a quiet NaN stays compared): gfx1150 quiets a signaling NaN here (0x7c96 -> 0x7e96); the model moves
            # bits, which is what RDNA2 does under IEEE mode 0 and not under mode 1 (0x7c01 -> 0xfe01 on gfx1036, SHADER-260)
            if "pack_swapped" in PERTURB:
                lo, hi = hi, lo
            R[d] = Reg(lo | (hi << 16), unc(s[0], sel[0]) or is_f16_snan(lo), unc(s[1], sel[1]) or is_f16_snan(hi))
        elif n in PK_F16 or n in PK_SHIFT:
            k = len(s)
            out, us = [], []
            for lane_sel in ((op["opsel_hi"], op["opsel"]) if "pk_swapped" in PERTURB else (op["opsel"], op["opsel_hi"])):
                vals = [half(s[j], lane_sel[j]) for j in range(k)]
                u = any(unc(s[j], lane_sel[j]) for j in range(k))
                if n in PK_SHIFT:
                    a, b = vals
                    base = {"v_pk_lshlrev_b16": "v_lshlrev_b16", "v_pk_lshrrev_b16": "v_lshrrev_b16",
                            "v_pk_ashrrev_i16": "v_ashrrev_i16"}[n]
                    res, u2 = i16_op(base, a, b), False
                else:
                    res, u2 = f16_op(n[5:-4], *vals)
                out.append(res)
                us.append(u or u2)
            R[d] = Reg(out[0] | (out[1] << 16), us[0], us[1])
        else:
            raise ValueError(n)



# ============================================================================================== FUZZ-5: memory

SCR_BASE = 4096          # byte address (console VA) of the scratch area, right after the per-thread results
SLOT = 64                # bytes per thread in the scratch area
SCR_BYTES = THREADS * SLOT
FILL_VK = 0xEEEEEEEE     # what vk_compute_test pre-fills the output buffer with
FILL_HW = 0xDEAD0000     # what hw_oracle.c pre-fills it with (runner.SENTINEL)
PERTURBATIONS = {
    "f16": ("hi_zero", "opsel_dst_ignored", "pk_swapped", "min_is_max", "pack_swapped",
            "i16_opsel_ignored"),
    "mem": ("oob_ignore_mode", "mode0_no_stride", "idx_unchecked", "soffset_unchecked", "whole_access", "oob_load_keeps",
            "sbuf_no_bound", "sbuf_stride_ignored"),
}
VA, VB, VS = 44, 48, 40  # s[44:47] V# "A" (offen), s[48:51] V# "B" (idxen), s[40:43] V# "S" (s_buffer_load)
SOFF, SWAVE, SPTR, SDST = 52, 53, 56, 60   # s52 constant soffset, s53 slot offset of the wave's first thread,
                                           # s[56:57] pointer to the scratch area, s[60:63] scalar load destinations
GLD = ("load_off", "load_idx", "load_io", "load_wf", "load_untouched")
GST = ("store_off", "store_idx", "store_io")
MEM_OPS = GLD + GST + ("sload", "sbuf")
MODE_NAMES = ("structured_with_offset", "structured", "disabled", "raw")


def word3(mode, fam):
    """V# word 3 for a 32_FLOAT buffer with identity DST_SEL and OOB_SELECT = mode. GFX10 and GFX11 keep DST_SEL
    [11:0] and OOB_SELECT [29:28] where they were; FORMAT is 7 bits [18:12] on GFX10 (22 = 32_FLOAT) and 6 bits
    [17:12] on GFX11 (22 = 32_FLOAT too), and GFX10 has RESOURCE_LEVEL in bit 24, which GFX11 dropped (Mesa
    gfx10-rsrc.json and gfx11-rsrc.json). TYPE [31:30] must be 0 on GFX11."""
    w = 0xFAC | (22 << 12) | (mode << 28)
    return w | (1 << 24) if fam == "gfx10" else w


def buffer_dword_ok(mode, n_rec, stride, idx, off, offs, k):
    """Is dword k (4*k bytes into the access) of a MUBUF access inside the buffer? off = inst_offset + vaddr offset,
    offs = off + soffset. The per-mode rule of docs/design.md "Buffer bounds checking" (measured on RDNA2 and on
    gfx1150): 0 STRUCTURED_WITH_OFFSET idx >= num_records or off + 4k + 4 > stride; 1 STRUCTURED idx >= num_records;
    2 DISABLED num_records == 0; 3 RAW offs + 4k + 4 > num_records. Modes 1 and 2 check once for the whole access."""
    if "oob_ignore_mode" in PERTURB:
        mode = 3
    if "soffset_unchecked" in PERTURB:
        offs = off
    if mode == 0:
        return not ((idx >= n_rec and "idx_unchecked" not in PERTURB) or
                    (off + 4 * k + 4 > stride and "mode0_no_stride" not in PERTURB))
    if mode == 1:
        return idx < n_rec or "idx_unchecked" in PERTURB
    if mode == 2:
        return n_rec != 0
    return offs + 4 * k + 4 <= n_rec


class MemFamily(Family):
    """MUBUF loads/stores, s_load and s_buffer_load. Every thread owns a 64-byte slot of the scratch area and only
    stores to its own slot, so the result doesn't depend on thread order. Two V#s with a random OOB_SELECT, stride and
    num_records (any byte value, so an access can straddle the end) are used for MUBUF ops: A for `offen` (vaddr =
    the slot's byte offset) and offset-only ops, B (stride = SLOT) for `idxen` (vindex = thread id) and `idxen offen`
    (voffset = 4 * (thread id & 3)). The scalar ops read the slot of the first thread of their wave (a wave-uniform
    soffset), which only that thread writes, so they never race with another wave. s_buffer_load has its own V# S."""
    title = "FUZZ-5"
    short = "mem"
    extra_words = SCR_BYTES // 4
    fill = FILL_VK

    # ---- generation
    def build(self):
        rng = self.rng
        self.init = {r: (rng.randrange(1, 1 << 24), rng.getrandbits(32)) for r in POOL}
        self.soff = rng.choice((0, 0, 4, 8, 16))
        self.a_mode, self.b_mode = rng.choice((3, 3, 3, 0, 1, 2)), rng.choice((0, 0, 1, 1, 2, 3))
        edge = lambda: rng.choice((0, 1, rng.randrange(2, 64), 64, 64, 64))           # noqa: E731  slots in range
        part = lambda: SLOT * edge() + 4 * rng.randrange(0, 16) * rng.choice((0, 1, 1))  # noqa: E731  ends mid-slot
        # V# A. stride matters only in mode 0 (where `offen` exceeds it); N is a byte count in mode 3, else just 0 or not
        self.a_stride = part() if self.a_mode == 0 else rng.choice((0, 0, SLOT))
        self.a_n = part() if self.a_mode == 3 else rng.choice((0, 1, 7, 64, 100000))
        # V# B, stride = SLOT: N is records in modes 0 and 1, bytes in mode 3, any in mode 2
        self.b_stride = SLOT
        self.b_n = (edge() if self.b_mode in (0, 1) else rng.choice((0, 4, 8, 20, 36, 60, 64, 80, 100000)) if self.b_mode == 3
                    else rng.choice((0, 5)))
        # V# S for s_buffer_load: the limit is N bytes (stride 0) or N * stride bytes
        self.s_stride = rng.choice((0, 0, 4, SLOT))
        limit = rng.choice((0, 4, 20, SLOT, 2 * SLOT + 8, 32 * SLOT, 32 * SLOT + 24, 33 * SLOT, SCR_BYTES, 1 << 20))
        self.s_n = limit if self.s_stride == 0 else limit // self.s_stride
        self.s_mode = rng.randrange(4)               # OOB_SELECT: ignored by s_buffer_load (docs/design.md)
        self.a_rt = self.a_mode == 3 and rng.random() < 0.4    # translator path: word 3 of A is the run-time V#'s (s3)
        kinds = [k for k in MEM_OPS if k not in self.skip]
        weights = [{"sload": 1, "sbuf": 2}.get(k, 3 if k in GST else 2) for k in kinds]
        for _ in range(rng.randrange(4, 13)):
            self.ops.append(self.random_op(rng.choices(kinds, weights)[0]))

    def random_op(self, kind):
        rng = self.rng
        n = rng.choice((1, 1, 2, 3, 4))
        soff = self.soff if kind in ("store_off", "load_off", "store_idx", "load_idx", "store_io", "load_io") and \
            rng.random() < 0.6 else 0
        voff = 12 if kind in ("store_io", "load_io") else 0      # room for the largest voffset, 4 * (tid & 3) = 12
        top = SLOT - 4 * n - soff - voff
        off = 4 * rng.randrange(0, top // 4 + 1)
        reg = rng.choice(POOL[: len(POOL) - n + 1])
        op = {"kind": kind, "n": n, "off": off, "r": reg, "soff": soff}
        if kind == "load_untouched":
            op.update(n=1, off=4 * rng.randrange(0, SLOT // 4), soff=0)
        if kind in ("sload", "sbuf"):
            op["n"] = n = rng.choice((1, 1, 2, 4))
            op["reg_dst"] = SDST
            op["off"] = 4 * rng.randrange(0, (SLOT - 4 * n) // 4 + 1)
            op["r"] = rng.choice(POOL[: len(POOL) - n + 1])
            op["imm"] = op["off"]
        op["text"] = f"{kind} n={op['n']} offset={op['off']} soffset={op['soff']} v{op['r']}"
        return op

    # ---- assembly
    def vsharp_lines(self, hw, fam):
        """The three V#s and the scalar pointer. On the hardware the base is the output pointer s[2:3] plus SCR_BASE;
        in the translator the output buffer sits at address 0 (console VA), so the base is SCR_BASE itself."""
        out = []

        def vsharp(base, stride, n, mode, w3=None):
            if hw:
                out.extend([f"s_add_u32 s{base}, s2, {SCR_BASE:#x}", f"s_addc_u32 s{base + 1}, s3, 0",
                            f"s_and_b32 s{base + 1}, s{base + 1}, 0xffff"])
                if stride:
                    out.append(f"s_or_b32 s{base + 1}, s{base + 1}, {stride << 16:#x}")
            else:
                out.extend([f"s_mov_b32 s{base}, {SCR_BASE:#x}", f"s_mov_b32 s{base + 1}, {stride << 16:#x}"])
            out.extend([f"s_mov_b32 s{base + 2}, {n:#x}",
                        f"s_mov_b32 s{base + 3}, {w3 if w3 else f'{word3(mode, (W3_LAYOUT or fam) if hw else fam):#x}'}"])

        vsharp(VA, self.a_stride, self.a_n, self.a_mode, "s3" if (self.a_rt and not hw) else None)
        vsharp(VB, self.b_stride, self.b_n, self.b_mode)
        vsharp(VS, self.s_stride, self.s_n, self.s_mode)
        out += [f"s_add_u32 s{SPTR}, s2, {SCR_BASE:#x}", f"s_addc_u32 s{SPTR + 1}, s3, 0"] if hw else \
               [f"s_mov_b32 s{SPTR}, {SCR_BASE:#x}", f"s_mov_b32 s{SPTR + 1}, 0"]
        return out

    def body(self, fam, hw):
        gfx10 = fam == "gfx10"
        sfx = {1: "dword", 2: "dwordx2", 3: "dwordx3", 4: "dwordx4"} if gfx10 else {1: "b32", 2: "b64", 3: "b96", 4: "b128"}
        ssfx = {1: "dword", 2: "dwordx2", 4: "dwordx4"} if gfx10 else {1: "b32", 2: "b64", 4: "b128"}
        out = []
        for r, (k, c) in self.init.items():
            out += [f"v_mul_u32_u24_e32 v{r}, {k:#x}, v0", f"v_xor_b32_e32 v{r}, {c:#x}, v{r}"]
        out += self.vsharp_lines(hw, fam)
        out += [f"s_mov_b32 s{SOFF}, {self.soff:#x}",
                f"v_mul_u32_u24_e32 v{ADDR}, {SLOT}, v0",            # offen: byte offset of the thread's slot
                "v_mov_b32_e32 v8, v0", "v_and_b32_e32 v9, 3, v0", "v_lshlrev_b32_e32 v9, 2, v9",   # idxen offen: (index, offset)
                f"v_readfirstlane_b32 s{SWAVE}, v0", f"s_mul_i32 s{SWAVE}, s{SWAVE}, {SLOT}"]
        sync = HW_SYNC if hw else "translator"
        for op in self.ops:
            n, off, r, kind = op["n"], op["off"], op["r"], op["kind"]
            regs = f"v{r}" if n == 1 else f"v[{r}:{r + n - 1}]"
            soff = f"s{SOFF}" if op["soff"] else "0"
            if kind in ("sload", "sbuf"):
                dst = f"s{SDST}" if n == 1 else f"s[{SDST}:{SDST + n - 1}]"
                if hw and sync in ("min", "scalar", "full"):
                    out.append("s_waitcnt_vscnt null, 0x0")      # earlier stores have left the vector memory pipeline
                if hw and sync in ("sinv", "scalar", "full"):
                    out.append("s_dcache_inv")                  # the scalar cache is not coherent with the vector stores
                if hw and sync == "full":
                    out += ["buffer_gl0_inv", "buffer_gl1_inv"]
                if kind == "sload":
                    out.append(f"s_load_{ssfx[n]} {dst}, s[{SPTR}:{SPTR + 1}], s{SWAVE} offset:{op['imm']:#x}")
                else:
                    out.append(f"s_buffer_load_{ssfx[n]} {dst}, s[{VS}:{VS + 3}], s{SWAVE} offset:{op['imm']:#x}")
                out.append("s_waitcnt lgkmcnt(0)")
                out += [f"v_mov_b32_e32 v{r + j}, s{SDST + j}" for j in range(n)]
                continue
            sj = sfx[n]
            vsh = f"s[{VB}:{VB + 3}]" if kind.endswith(("_idx", "_io")) else f"s[{VA}:{VA + 3}]"
            if kind.endswith("_off") or kind == "load_untouched":
                addr, flags = f"v{ADDR}", "offen"
            elif kind.endswith("_idx"):
                addr, flags = "v8", "idxen"
            elif kind.endswith("_io"):
                addr, flags = "v[8:9]", "idxen offen"
            else:                                                   # load_wf: offset-only, soffset = the wave's slot
                addr, flags, soff = "off", "", f"s{SWAVE}"
            verb = "store" if kind in GST else "load"
            out.append(f"buffer_{verb}_{sj} {regs}, {addr}, {vsh}, {soff} {flags} offset:{off}".replace("  ", " "))
            out.append("s_waitcnt vmcnt(0)")
            if verb == "store" and hw and sync in ("min", "scalar", "full"):
                out.append("s_waitcnt_vscnt null, 0x0")
        return out

    def asm_body(self):
        return self.body("gfx10", False)

    def hw_body(self, fam):
        return self.body(fam, True)

    def hw_tail(self, fam):
        # the scratch area is in the output buffer already; the results are stored by hw_lines
        return []

    # ---- model
    def model(self):
        wave = self.wave
        fill = self.fill.to_bytes(4, "little") * (SCR_BYTES // 4)
        mem = bytearray(fill)
        regs = [{r: Reg(((t * (k & 0xFFFFFF)) & M32) ^ c) for r, (k, c) in self.init.items()} for t in range(THREADS)]
        writer = [{r: -1 for r in POOL} for _ in range(THREADS)]
        rd = lambda a: struct.unpack_from("<I", mem, a)[0]          # noqa: E731
        for i, op in enumerate(self.ops):
            n, off, r, kind = op["n"], op["off"], op["r"], op["kind"]
            for t in range(THREADS):
                f = t - t % wave                                     # the first thread of t's wave
                if kind in ("sload", "sbuf"):
                    for j in range(n):
                        a = f * SLOT + op["imm"] + 4 * j
                        if kind == "sbuf":
                            limit = self.s_n * (1 if "sbuf_stride_ignored" in PERTURB else (self.s_stride or 1))
                            ok = ((f * SLOT & ~3) + (op["imm"] & ~3) + 4 * j + 4 <= limit    # num_records bound, no OOB_SELECT
                                  or "sbuf_no_bound" in PERTURB)
                        else:
                            ok = True
                        regs[t][r + j] = Reg(rd(a) if ok else 0)
                        writer[t][r + j] = i
                    continue
                if kind.endswith("_off") or kind == "load_untouched":
                    mode, n_rec, stride, idx, voff, soff, base = self.a_mode, self.a_n, self.a_stride, 0, t * SLOT, op["soff"], 0
                elif kind.endswith("_idx"):
                    mode, n_rec, stride, idx, voff, soff, base = self.b_mode, self.b_n, self.b_stride, t, 0, op["soff"], 0
                elif kind.endswith("_io"):
                    mode, n_rec, stride, idx, voff, soff, base = (self.b_mode, self.b_n, self.b_stride, t, 4 * (t & 3),
                                                                  op["soff"], 0)
                else:                                                # load_wf
                    mode, n_rec, stride, idx, voff, soff, base = self.a_mode, self.a_n, self.a_stride, 0, 0, f * SLOT, 0
                inst = off
                oks = [buffer_dword_ok(mode, n_rec, stride, idx, inst + voff, inst + voff + soff, j) for j in range(n)]
                if "whole_access" in PERTURB:
                    oks = [all(oks)] * n
                for j in range(n):
                    ok = oks[j]
                    # the address: base + soffset + inst + voffset + idx * stride, which the generator keeps in the
                    # thread's slot (or, for load_wf, the wave's first slot)
                    a = voff + soff + inst + 4 * j + (t * SLOT if kind.endswith(("_idx", "_io")) else 0)
                    if kind.startswith("store"):
                        if ok:
                            struct.pack_into("<I", mem, a, regs[t][r + j].v)
                    else:
                        if ok or "oob_load_keeps" not in PERTURB:
                            regs[t][r + j] = Reg(rd(a) if ok else 0)
                        writer[t][r + j] = i
        self.writer = writer
        extra = list(struct.unpack(f"<{SCR_BYTES // 4}I", bytes(mem)))
        return regs, extra


# ========================================================================================== runner

FAMILIES = {"f16": F16Family, "mem": MemFamily}


def programs(family, seed, count, waves, skip=()):
    cls = FAMILIES[family]
    return [cls(seed, i, w, skip) for i in range(count) for w in waves]


def compare(p, data):
    """Differences between the model and the dumped words. p.uncertain_words is set to how many result words had a
    half the model does not judge (so a summary can show how many comparisons were skipped)."""
    words = struct.unpack(f"<{len(data) // 4}I", data)
    want, extra = p.model()
    diffs = []
    p.uncertain_words = sum(1 for t in range(THREADS) for r in POOL if want[t][r].ulo or want[t][r].uhi)
    for t in range(THREADS):
        for i, r in enumerate(POOL):
            got = words[t * len(POOL) + i]
            w = want[t][r]
            mask = (0 if w.ulo else 0xFFFF) | (0 if w.uhi else 0xFFFF0000)
            if (got ^ w.v) & mask:
                diffs.append(f"thread {t} v{r}: got {got:#010x}, want {w.v:#010x} (compared bits {mask:#010x}; "
                             f"last written by {p.describe(p.writer[t][r])})")
    base = THREADS * len(POOL)
    for i, w in enumerate(extra):
        if w is not None and words[base + i] != w:
            diffs.append(f"extra word {i}: got {words[base + i]:#010x}, want {w:#010x}")
    return diffs


def pc_tail_env():
    """True when LEONIDA_PC_TAIL_UBO asks for rdna2spirv.py --pc-tail-ubo (128-byte push constants)."""
    return (os.environ.get("LEONIDA_PC_TAIL_UBO") or "").strip().lower() in ("1", "true", "yes", "on")


def translate(progs, work, emulate, pc_tail=False, *, asm_jobs=1):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    for d in (disasm, fssm, spirv):
        if d.exists():
            shutil.rmtree(d)
    for d in (disasm, fssm):
        d.mkdir(parents=True)
    mc = mc_command.mc_command()

    def assemble(p):
        text = p.asm()
        asm_path = work / (p.name + ".asm")
        asm_path.write_text(text, encoding="utf-8", newline="\n")
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = mc_command.run(cmd, input=text, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            raise RuntimeError(f"{p.name} did not assemble:\n{r.stderr}")
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (p.name + ".s")).write_text(f"; {p.name}\n{code}", encoding="utf-8", newline="\n")
        (fssm / (p.name + ".fssm")).write_bytes(compute_header(asm_path))

    if asm_jobs == 1:
        for p in progs:
            assemble(p)
    else:
        with ThreadPoolExecutor(max_workers=asm_jobs) as pool:
            # Refill a small window in input order to bound the queue and report the first failure by program order.
            remaining = iter(progs)
            window = min(2 * asm_jobs, sys.maxsize)
            pending = deque(pool.submit(assemble, p) for p in islice(remaining, window))
            for p in remaining:
                pending.popleft().result()
                pending.append(pool.submit(assemble, p))
            for job in pending:
                job.result()

    cmd = [sys.executable, "-I", os.path.join(HERE, "rdna2spirv.py"), str(disasm), "--out", str(spirv)]
    if emulate:
        cmd.append("--wave64-emulate")
    if pc_tail:
        cmd.append("--pc-tail-ubo")
    r = subprocess.run(cmd, capture_output=True, text=True)
    index = spirv / "index.json"
    if not index.exists():
        raise RuntimeError(f"rdna2spirv.py exited {r.returncode} without an index.json:\n{(r.stdout + r.stderr)[-2000:]}")
    return json.loads(index.read_text(encoding="utf-8")), (r.stdout + r.stderr)


def run_gpu(p, spv, build, index_entry, emulate, dump, pc_tail=False):
    exe = os.path.join(build, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(exe):
        sys.exit(f"vk_compute_test not found in {build} or its RelWithDebInfo/Release folder; "
                 "build it first (cmake --build <build dir> --target vk_compute_test)")
    env = dict(os.environ, VKC_OUT_WORDS=str(p.out_words()), VKC_DUMP=str(dump))
    if pc_tail:
        env["LEONIDA_PC_TAIL_UBO"] = "1"
    args = [exe, str(spv), "1", "dump"]
    if p.wave == 64 and not emulate:
        args.append("--require-wave64")
    elif p.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    if index_entry.get("subgroup_size") and not emulate:
        env["VKC_SUBGROUP_SIZE"] = str(index_entry["subgroup_size"])
    limit = gpu_test_timeout()
    try:
        r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=limit)
    except subprocess.TimeoutExpired:
        return 1, timed_out(exe, limit)
    return r.returncode, (r.stdout + r.stderr).strip()


def show(family, seed, index, wave, skip=(), hw_fam=None):
    p = FAMILIES[family](seed, index, wave, skip)
    if hw_fam:
        print("\n".join(p.hw_lines(hw_fam)))
    else:
        print(p.asm())
    regs, _ = p.model()
    print("; model, thread 0:", " ".join(f"v{r}={regs[0][r].v:#x}" for r in POOL))


HW_VGPRS, HW_SGPRS = 40, 72
W3_LAYOUT = None   # --gfx10-word3: build the V#s of the hardware kernels with the GFX10 word 3 (informational)
HW_SYNC = "scalar"    # none, min (wait for the stores), scalar (and invalidate the scalar cache) or full (the vector caches too)


def hw_kernel(p, fam):
    return asm.Kernel(p.hw_lines(fam), p.wave, THREADS, p.out_words(), raw=True, vgprs=HW_VGPRS, sgprs=HW_SGPRS)


def run_hw(oracle, progs, args, tag="hw", quiet=False):
    """Run every program natively (wave32 or native wave64, 64 threads, one dword per result word) and compare with
    the model. Returns (stats, failed programs); prints the first differences of each mismatch."""
    fam = asm.family(oracle.arch)
    stats = {}
    kernels = [hw_kernel(p, fam) for p in progs]
    skip_args = "".join(f" --skip-op {k}" for k in args.skip_op)
    failed = []
    for p, got in zip(progs, oracle.run(kernels)):
        st = stats.setdefault(p.wave, {"programs": 0, "match": 0, "mismatch": 0, "rejected": 0, "uncertain_words": 0})
        st["programs"] += 1
        if isinstance(got, str):
            st["rejected"] += 1
            if not quiet:
                print(f"REJECTED {p.name} [{tag}]: {got}")
            continue
        p.fill = FILL_HW
        diffs = compare(p, struct.pack(f"<{len(got)}I", *got))
        st["uncertain_words"] += p.uncertain_words
        if not diffs:
            st["match"] += 1
            continue
        st["mismatch"] += 1
        failed.append(p)
        if quiet:
            continue
        print(f"MISMATCH {p.name} [{tag}] ({len(diffs)} words); print it with --family {args.family} --oracle hw "
              f"--show {args.seed}:{p.name.split('_')[3]}:{p.wave}{skip_args}")
        for d in diffs[:args.max_report]:
            print("   ", d)
    return stats, failed


def negative_control(oracle, a, waves):
    """Every deliberately wrong model must be caught by the hardware comparison: the real model passes, each perturbed
    one has to mismatch on at least one program. Exit 0 when all are caught, 1 when one slips through."""
    progs = programs(a.family, a.seed, a.programs, waves, tuple(a.skip_op))
    fam = asm.family(oracle.arch)
    got = oracle.run([hw_kernel(p, fam) for p in progs])
    ok_progs = [(p, g) for p, g in zip(progs, got) if not isinstance(g, str)]

    def mismatches():
        n = 0
        for p, g in ok_progs:
            p.fill = FILL_HW
            n += bool(compare(p, struct.pack(f"<{len(g)}I", *g)))
        return n
    base = mismatches()
    print(f"unperturbed model: {base} of {len(ok_progs)} programs mismatch")
    code = 1 if base else 0
    for name in PERTURBATIONS[a.family]:
        PERTURB.add(name)
        try:
            n = mismatches()
        finally:
            PERTURB.discard(name)
        print(f"perturbation {name}: caught by {n} of {len(ok_progs)} programs" + ("" if n else "   <-- NOT CAUGHT"))
        if not n:
            code = 1
    return code


def run_translator(progs, a):
    """The translator path: assemble for gfx1013, translate, run with vk_compute_test. Returns an exit code."""
    a.build = find_test_program(a.build, "vk_compute_test") or a.build   # a Visual Studio config subfolder
    try:
        parent = Path(a.work).resolve() if a.work else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f"{a.family}_fuzz_", dir=parent))
    except OSError as e:
        print(f"ERROR: cannot create work directory: {e}", file=sys.stderr)
        return 2
    print(f"work dir {work}")
    pc_tail = bool(getattr(a, "pc_tail_ubo", False))
    translate_options = {"pc_tail": True} if pc_tail else {}
    asm_jobs = getattr(a, "asm_jobs", 1)
    if asm_jobs != 1:
        translate_options["asm_jobs"] = asm_jobs
    try:
        index, log = translate(progs, work, a.emulate, **translate_options)
    except RuntimeError as e:
        print(e)
        return 2
    failed = untranslated = skipped = errors = 0
    skip_args = "".join(f" --skip-op {k}" for k in a.skip_op)
    uncertain = 0
    for p in progs:
        spv = work / "spirv" / (p.name + ".spv")
        if p.name not in index or not spv.is_file():
            untranslated += 1
            reason = next((ln.strip() for ln in log.splitlines() if p.name in ln), "")
            print(f"NOT TRANSLATED {p.name} {reason[:160]}")
            continue
        dump = work / (p.name + ".bin")
        dump.unlink(missing_ok=True)
        rc, msg = (run_gpu(p, spv, a.build, index[p.name], a.emulate, dump, pc_tail=True)
                   if pc_tail else run_gpu(p, spv, a.build, index[p.name], a.emulate, dump))
        if rc == 77:
            skipped += 1
            print(f"SKIP {p.name}: {msg.splitlines()[-1] if msg else 'unsupported'}")
            continue
        # A wave32 program needs 32-lane subgroups. Where vk_compute_test couldn't pin them, its words come from
        # another wave size and are no evidence either way. --wave64-emulate output uses no subgroup operations.
        lanes = unpinned_subgroup(msg) if p.wave == 32 else None
        if lanes:
            skipped += 1
            print(f"SKIP {p.name}: the device can't run it at {lanes} lanes "
                  f"(vk_compute_test: compute subgroup size: {lanes} not available)")
            continue
        data = dump.read_bytes() if dump.is_file() else b""
        if rc != 0 or len(data) != 4 * p.out_words():
            errors += 1
            print(f"ERROR {p.name}: exit {rc}, {len(data)} of {4 * p.out_words()} dump bytes: {msg[-400:]}")
            continue
        p.fill = FILL_VK
        diffs = compare(p, data)
        uncertain += p.uncertain_words
        if diffs:
            failed += 1
            idx = p.name.split("_")[3]
            print(f"MISMATCH {p.name} ({len(diffs)} words); print it with --family {a.family} --show "
                  f"{a.seed}:{idx}:{p.wave}{skip_args}")
            for d in diffs[:a.max_report]:
                print("   ", d)
    mode = "emulated" if a.emulate else "native"
    ran = len(progs) - untranslated - skipped - errors
    print(f"{a.family}: {ran - failed}/{ran} programs that ran match ({mode}); {failed} mismatched, {skipped} skipped, "
          f"{untranslated} not translated, {errors} harness errors; {uncertain} result words not judged; work dir {work}")
    if errors or ran == 0:
        return 2
    return 1 if failed or untranslated else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--programs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--asm-jobs", type=int, default=1, metavar="N",
                    help="assemble translator fixtures with N concurrent llvm-mc processes (default: 1)")
    ap.add_argument("--wave", choices=("64", "32", "both"), default="both")
    ap.add_argument("--emulate", action="store_true", help="translate with --wave64-emulate, run at subgroup 32")
    ap.add_argument("--pc-tail-ubo", action="store_true",
                    help="translate with --pc-tail-ubo and run vk_compute_test with LEONIDA_PC_TAIL_UBO=1 "
                         "(128-byte push constants; also set by LEONIDA_PC_TAIL_UBO=1)")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"))
    ap.add_argument("--work", default=None, help="keep each run's generated .asm and SPIR-V in a new subfolder here")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--show", default=None, metavar="SEED:INDEX[:WAVE]")
    ap.add_argument("--max-report", type=int, default=8)
    ap.add_argument("--skip-op", action="append", default=[], help="leave this op out (a known finding)")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated: translator (default; llvm-mc, rdna2spirv.py, vk_compute_test), hw (natively on "
                         "the AMD GPU through tools/hw_oracle/, wave32 and native wave64; exit 77 when it is not there), "
                         "or translator,hw. The same programs run on both; they only use forms with the same meaning on GFX10 and GFX11")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--negative-control", action="store_true",
                    help="hw only: perturbed models must be caught (exit 0 when every perturbation is)")
    ap.add_argument("--hw-sync", choices=("none", "min", "sinv", "scalar", "full"), default="scalar",
                    help="mem on the hardware: none = only data waits, min = wait for the stores before a scalar load, "
                         "sinv = only s_dcache_inv, scalar (default) = both, full = also buffer_gl0_inv/buffer_gl1_inv")
    ap.add_argument("--gfx10-word3", action="store_true",
                    help="mem on the hardware: V# word 3 in the GFX10 layout (RESOURCE_LEVEL set); informational")
    ap.add_argument("--keep", default=None, help="hw: write the native kernel of each mismatching program here")
    a = ap.parse_args()
    a.pc_tail_ubo = a.pc_tail_ubo or pc_tail_env()
    global HW_SYNC, W3_LAYOUT
    HW_SYNC = a.hw_sync
    W3_LAYOUT = "gfx10" if a.gfx10_word3 else None
    if a.programs < 1 or a.max_report < 1:
        ap.error("--programs and --max-report must be at least 1")
    if a.asm_jobs < 1:
        ap.error("--asm-jobs must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    known = set(F16_OPS) | set(MEM_OPS)
    unknown = sorted(set(a.skip_op) - known)
    if unknown:
        ap.error(f"--skip-op: unknown op {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    if a.negative_control and "hw" not in backends:
        ap.error("--negative-control needs --oracle hw")
    hw = "hw" in backends
    oracle = None
    if hw and not a.model_only:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        from hw_oracle import asm
        why = asm.rdna23_skip_reason(oracle.arch)
        if why:
            oracle.close()
            print(f"hardware oracle unavailable: {why}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    if a.show is not None:
        parts = a.show.split(":")
        try:
            if len(parts) not in (2, 3):
                raise ValueError
            seed, index = int(parts[0]), int(parts[1])
            wave = int(parts[2]) if len(parts) == 3 else 64
        except ValueError:
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        if index < 0 or wave not in (32, 64):
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        hw_fam = (asm.family(oracle.arch) if oracle else "gfx11") if hw else None
        show(a.family, seed, index, wave, tuple(a.skip_op), hw_fam)
        if oracle:
            oracle.close()
        return 0
    waves = (64, 32) if a.wave == "both" else (int(a.wave),)
    progs = programs(a.family, a.seed, a.programs, waves, tuple(a.skip_op))
    if a.model_only:
        for p in progs:
            p.model()
        print(f"{len(progs)} {a.family} programs generated and modelled")
        return 0
    code = 0
    try:
        if oracle:
            if a.negative_control:
                return negative_control(oracle, a, waves)
            stats, failed = run_hw(oracle, progs, a)
            if a.keep:
                os.makedirs(a.keep, exist_ok=True)
                for p in failed:
                    with open(os.path.join(a.keep, p.name + "_hw.s"), "w", encoding="utf-8", newline="\n") as f:
                        f.write("\n".join(p.hw_lines(asm.family(oracle.arch))) + "\n")
            for wave, st in sorted(stats.items()):
                print(f"{a.family} wave{wave} hw: " + ", ".join(f"{k} {v}" for k, v in st.items()))
            if any(st["mismatch"] or st["rejected"] for st in stats.values()):
                code = 1
            if sum(st["match"] for st in stats.values()) == 0:
                code = max(code, 2)
        if "translator" in backends:
            code = max(code, run_translator(progs, a))
    finally:
        if oracle:
            oracle.close()
    return code


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
