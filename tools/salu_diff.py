r"""Differential fuzzing of the scalar ALU and SCC (FUZZ-4 in #156). Standard library only.

Generates seeded straight-line programs of SOP1/SOP2/SOPC/SOPK instructions, runs them through

  * the reference below, written from the RDNA2 ISA guide's pseudocode for each opcode, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py,

and compares four exported words bit for bit: three SGPRs and SCC (through s_cselect_b32).
The reference is separate from tools/cpu_ref.py on purpose: it models only the scalar unit, keeps
SCC explicit for every opcode, and does not share code with the translator.

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/) and the hardware's four
words are compared with the reference: that validates the reference itself, not the translator. The GPU is
whatever the machine has (gfx11 on an RDNA3 laptop), so a difference is checked against the ISA text before it
counts as a reference bug. Without the oracle tools the run exits 77.

With --branches the programs also consume SCC in control flow: after an SCC-producing instruction comes either
a chain of 1..3 SCC consumers (s_cselect_b32, s_addc_u32, s_subb_u32) or s_cbranch_scc0/scc1 forward over the
next 1..3 scalar instructions (no loops, no nesting, one lane). In the program lines a branch is written
`s_cbranch_scc0 +N` (skip the next N instructions); the listing handed to the translator carries the real offset
in dwords and `; encoding:` placeholders of each instruction's size, which is what places the branch targets.
Half of the instructions are then drawn from the SCC producers among the selected opcodes, so over a few thousand
programs every producer is followed by every consumer and by both branch senses. The hardware oracle does not
take branch programs (its kernels are straight-line).

Usage:
    python -I tools/salu_diff.py [--programs N] [--seed S] [--max-ops N] [--ops s_min_i32,s_cselect_b32]
    python -I tools/salu_diff.py --branches [--programs N] [--seed S]   # SCC consumers and forward branches
    python -I tools/salu_diff.py --oracle hw             # hardware vs reference only
    python -I tools/salu_diff.py --oracle translator,hw  # both
    python -I tools/salu_diff.py --list-ops
    python -I tools/salu_diff.py --self-test

Exit 0 when every program matches, 1 on a mismatch (the shortest failing program is printed after
removing instructions that do not matter), 2 on a harness error, 77 when the requested hardware oracle is
unavailable. Skips are counted with their reasons.

By default the opcodes in LANE_MASK_OPS are left out (a hardware-only run keeps them: the GPU has no wave-mask
view of a 64-bit SGPR pair): the translator models those 64-bit values (and
s_wqm_b32) as per-lane wave masks, so a 64-bit integer reference disagrees with them by design (--ops
names them explicitly). s_bfe_* control words stay inside offset + width <= 32 unless --bfe-any: past
that the field stops at bit 31 (width = min(width, 32 - offset), so s_bfe_i32 sign-extends from the top bit
that exists). gfx1150 hardware and the translator both do that; the ISA pseudocode does not say so, and RDNA2
itself is unmeasured (#156). s_ff0/ff1_i32_b32 with a constant source diverged until SHADER-46 (#349).
"""
import argparse
import gc
import os
import random
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Data SGPRs. 64-bit operands use the even-aligned pairs inside this range.
REGS = tuple(range(4, 20))
PAIRS = tuple(range(4, 20, 2))
OUT_SCC = 30   # s_cselect_b32 s30, 1, 0 observes SCC at the end
BFE_ANY = False    # any s_bfe_* control word (--bfe-any), not only offset + width <= 32
INLINE64 = False   # inline constants in 64-bit positions (--inline64); the translator reads those as lane masks

LANE_MASK_OPS = ("s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64", "s_nand_b64", "s_nor_b64",
                 "s_xnor_b64", "s_not_b64", "s_wqm_b64",
                 "s_wqm_b32", "s_cselect_b64", "s_bfm_b64")

INTERESTING = (0, 1, 2, 3, 31, 32, 33, 63, 64, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE,
               0x0000FFFF, 0xFFFF0000, 0x00008000, 0xFFFF8000, 0x12345678, 0xDEADBEEF, 0x55555555, 0xAAAAAAAA)


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def s64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def sext8(x):
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def sext16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def flbit32(x):
    x &= M32
    sign = x >> 31
    for n in range(1, 32):
        if (x >> (31 - n)) & 1 != sign:
            return n
    return M32


def is_nan(w):
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0


def popcount(x):
    return bin(x).count("1")


def first_set(x, width):
    """Index of the lowest set bit, or -1 (0xFFFFFFFF)."""
    for i in range(width):
        if (x >> i) & 1:
            return i
    return M32


def wqm(x, width):
    out = 0
    for q in range(0, width, 4):
        if (x >> q) & 0xF:
            out |= 0xF << q
    return out


def quadmask(x, width):
    out = 0
    for q in range(width // 4):
        if (x >> (4 * q)) & 0xF:
            out |= 1 << q
    return out


def bitreplicate(x):
    out = 0
    for i in range(32):
        if (x >> i) & 1:
            out |= 3 << (2 * i)
    return out


def flbit_signed(x, width):
    mask = (1 << width) - 1
    x &= mask
    sign = (x >> (width - 1)) & 1
    v = (~x & mask) if sign else x
    return (width - v.bit_length()) if v else M32


def bfe64(src, ctl, signed):
    off, width = ctl & 63, (ctl >> 16) & 0x7F
    width = min(width, 64 - off)
    if width == 0:
        return 0
    field = (src >> off) & ((1 << width) - 1)
    if signed and (field >> (width - 1)) & 1:
        field -= 1 << width
    return field & M64


def bfe(src, ctl, signed):
    off, width = ctl & 31, (ctl >> 16) & 0x7F
    # A field that runs past bit 31 stops there: s_bfe_i32 sign-extends from the top bit that exists (LLVM
    # constantFoldBFE does for v_bfe; 5-bit width there). The s_bfe rule itself rests on the gfx1150 measurement
    # (4000 random control words, --oracle hw), and RDNA2 agrees: gfx1036 (counts in the s_bfe row of docs/design.md).
    width = min(width, 32 - off)
    if width == 0:
        return 0
    field = (src >> off) & ((1 << width) - 1)
    if signed and (field >> (width - 1)) & 1:
        field -= 1 << width
    return field & M32


# ---- the reference: one entry per opcode ---------------------------------------------------------
# form: "sop2" d, a, b (32-bit); "sop2_64" d, a, b (64-bit pairs); "sop2_64s" d64, a64, b32 (shift amount);
# "sop2_bfm64" d64, a32, b32; "sop1" d, a; "sop1_64" d64, a64; "sop1_64to32" d32, a64; "sopc" a, b;
# "sopc_64" a64, b64; "sopk" d, simm16 (d is read too); "movk"; "cmpk" a, simm16.
# fn(a, b, scc[, d]) -> (result, new_scc or None). None leaves SCC as it was.

def _sop2():
    ops = {}

    def add(name, fn):
        ops[name] = ("sop2", fn)

    add("s_add_u32", lambda a, b, c: ((a + b) & M32, (a + b) >> 32))
    add("s_sub_u32", lambda a, b, c: ((a - b) & M32, int(b > a)))
    add("s_addc_u32", lambda a, b, c: ((a + b + c) & M32, (a + b + c) >> 32))
    add("s_subb_u32", lambda a, b, c: ((a - b - c) & M32, int(b + c > a)))
    add("s_add_i32", lambda a, b, c: ((a + b) & M32, int(not -2**31 <= s32(a) + s32(b) < 2**31)))
    add("s_sub_i32", lambda a, b, c: ((a - b) & M32, int(not -2**31 <= s32(a) - s32(b) < 2**31)))
    add("s_min_i32", lambda a, b, c: (a if s32(a) < s32(b) else b, int(s32(a) < s32(b))))
    add("s_min_u32", lambda a, b, c: (a if a < b else b, int(a < b)))
    add("s_max_i32", lambda a, b, c: (a if s32(a) > s32(b) else b, int(s32(a) > s32(b))))
    add("s_max_u32", lambda a, b, c: (a if a > b else b, int(a > b)))
    add("s_cselect_b32", lambda a, b, c: (a if c else b, None))
    for name, f in (("and", lambda a, b: a & b), ("or", lambda a, b: a | b), ("xor", lambda a, b: a ^ b),
                    ("andn2", lambda a, b: a & ~b), ("orn2", lambda a, b: a | ~b), ("nand", lambda a, b: ~(a & b)),
                    ("nor", lambda a, b: ~(a | b)), ("xnor", lambda a, b: ~(a ^ b))):
        add(f"s_{name}_b32", lambda a, b, c, f=f: (f(a, b) & M32, int(f(a, b) & M32 != 0)))
    add("s_lshl_b32", lambda a, b, c: ((a << (b & 31)) & M32, int((a << (b & 31)) & M32 != 0)))
    add("s_lshr_b32", lambda a, b, c: (a >> (b & 31), int(a >> (b & 31) != 0)))
    add("s_ashr_i32", lambda a, b, c: ((s32(a) >> (b & 31)) & M32, int((s32(a) >> (b & 31)) & M32 != 0)))
    add("s_bfm_b32", lambda a, b, c: ((((1 << (a & 31)) - 1) << (b & 31)) & M32, None))
    add("s_mul_i32", lambda a, b, c: ((s32(a) * s32(b)) & M32, None))
    add("s_mul_hi_u32", lambda a, b, c: ((a * b) >> 32, None))
    add("s_mul_hi_i32", lambda a, b, c: (((s32(a) * s32(b)) >> 32) & M32, None))
    add("s_absdiff_i32", lambda a, b, c: (abs(s32((a - b) & M32)) & M32, int(abs(s32((a - b) & M32)) & M32 != 0)))
    add("s_lshl1_add_u32", lambda a, b, c: (((a << 1) + b) & M32, int(((a << 1) + b) >= (1 << 32))))
    add("s_lshl2_add_u32", lambda a, b, c: (((a << 2) + b) & M32, int(((a << 2) + b) >= (1 << 32))))
    add("s_lshl3_add_u32", lambda a, b, c: (((a << 3) + b) & M32, int(((a << 3) + b) >= (1 << 32))))
    add("s_lshl4_add_u32", lambda a, b, c: (((a << 4) + b) & M32, int(((a << 4) + b) >= (1 << 32))))
    add("s_bfe_u32", lambda a, b, c: (bfe(a, b, False), int(bfe(a, b, False) != 0)))
    add("s_bfe_i32", lambda a, b, c: (bfe(a, b, True), int(bfe(a, b, True) != 0)))
    add("s_pack_ll_b32_b16", lambda a, b, c: (((b & 0xFFFF) << 16) | (a & 0xFFFF), None))
    add("s_pack_lh_b32_b16", lambda a, b, c: ((b & 0xFFFF0000) | (a & 0xFFFF), None))
    add("s_pack_hh_b32_b16", lambda a, b, c: ((b & 0xFFFF0000) | ((a >> 16) & 0xFFFF), None))
    return ops


def _sop2_64():
    ops = {}
    for name, f in (("and", lambda a, b: a & b), ("or", lambda a, b: a | b), ("xor", lambda a, b: a ^ b),
                    ("andn2", lambda a, b: a & ~b), ("orn2", lambda a, b: a | ~b),
                    ("nand", lambda a, b: ~(a & b)), ("nor", lambda a, b: ~(a | b)),
                    ("xnor", lambda a, b: ~(a ^ b))):
        ops[f"s_{name}_b64"] = ("sop2_64", lambda a, b, c, f=f: (f(a, b) & M64, int(f(a, b) & M64 != 0)))
    ops["s_cselect_b64"] = ("sop2_64", lambda a, b, c: (a if c else b, None))
    ops["s_lshl_b64"] = ("sop2_64s", lambda a, b, c: ((a << (b & 63)) & M64, int((a << (b & 63)) & M64 != 0)))
    ops["s_lshr_b64"] = ("sop2_64s", lambda a, b, c: (a >> (b & 63), int(a >> (b & 63) != 0)))
    ops["s_ashr_i64"] = ("sop2_64s", lambda a, b, c: ((s64(a) >> (b & 63)) & M64, int((s64(a) >> (b & 63)) & M64 != 0)))
    ops["s_bfe_u64"] = ("sop2_64s", lambda a, b, c: (bfe64(a, b, False), int(bfe64(a, b, False) != 0)))
    ops["s_bfe_i64"] = ("sop2_64s", lambda a, b, c: (bfe64(a, b, True), int(bfe64(a, b, True) != 0)))
    ops["s_bfm_b64"] = ("sop2_bfm64", lambda a, b, c: ((((1 << (a & 63)) - 1) << (b & 63)) & M64, None))
    return ops


def _sop1():
    ops = {
        "s_mov_b32": ("sop1", lambda a, b, c: (a, None)),
        "s_not_b32": ("sop1", lambda a, b, c: (~a & M32, int(~a & M32 != 0))),
        "s_brev_b32": ("sop1", lambda a, b, c: (int(f"{a:032b}"[::-1], 2), None)),
        "s_bcnt0_i32_b32": ("sop1", lambda a, b, c: (32 - popcount(a), int(32 - popcount(a) != 0))),
        "s_bcnt1_i32_b32": ("sop1", lambda a, b, c: (popcount(a), int(popcount(a) != 0))),
        "s_ff0_i32_b32": ("sop1", lambda a, b, c: (first_set(~a & M32, 32), None)),
        "s_ff1_i32_b32": ("sop1", lambda a, b, c: (first_set(a, 32), None)),
        "s_flbit_i32_b32": ("sop1", lambda a, b, c: (32 - a.bit_length() if a else M32, None)),
        "s_flbit_i32": ("sop1", lambda a, b, c: (flbit32(a), None)),
        "s_sext_i32_i8": ("sop1", lambda a, b, c: (sext8(a) & M32, None)),
        "s_sext_i32_i16": ("sop1", lambda a, b, c: (sext16(a) & M32, None)),
        "s_quadmask_b32": ("sop1", lambda a, b, c: (quadmask(a, 32), int(quadmask(a, 32) != 0))),
        "s_abs_i32": ("sop1", lambda a, b, c: (abs(s32(a)) & M32, int(abs(s32(a)) & M32 != 0))),
        "s_wqm_b32": ("sop1", lambda a, b, c: (wqm(a, 32), int(wqm(a, 32) != 0))),
        # D is read: bit S0[4:0] (or S0[5:0] for b64) of the destination is cleared or set.
        "s_bitset0_b32": ("sop1_rmw", lambda a, b, c, d: (d & ~(1 << (a & 31)) & M32, None)),
        "s_bitset1_b32": ("sop1_rmw", lambda a, b, c, d: (d | (1 << (a & 31)), None)),
        "s_bitset0_b64": ("sop1_64_rmw", lambda a, b, c, d: (d & ~(1 << (a & 63)) & M64, None)),
        "s_bitset1_b64": ("sop1_64_rmw", lambda a, b, c, d: (d | (1 << (a & 63)), None)),
        "s_mov_b64": ("sop1_64", lambda a, b, c: (a, None)),
        "s_not_b64": ("sop1_64", lambda a, b, c: (~a & M64, int(~a & M64 != 0))),
        "s_brev_b64": ("sop1_64", lambda a, b, c: (int(f"{a:064b}"[::-1], 2), None)),
        "s_quadmask_b64": ("sop1_64", lambda a, b, c: (quadmask(a, 64), int(quadmask(a, 64) != 0))),
        "s_wqm_b64": ("sop1_64", lambda a, b, c: (wqm(a, 64), int(wqm(a, 64) != 0))),
        "s_bitreplicate_b64_b32": ("sop1_32to64", lambda a, b, c: (bitreplicate(a), None)),
        "s_bcnt0_i32_b64": ("sop1_64to32", lambda a, b, c: (64 - popcount(a), int(64 - popcount(a) != 0))),
        "s_bcnt1_i32_b64": ("sop1_64to32", lambda a, b, c: (popcount(a), int(popcount(a) != 0))),
        "s_ff0_i32_b64": ("sop1_64to32", lambda a, b, c: (first_set(~a & M64, 64), None)),
        "s_ff1_i32_b64": ("sop1_64to32", lambda a, b, c: (first_set(a, 64), None)),
        "s_flbit_i32_b64": ("sop1_64to32", lambda a, b, c: (64 - a.bit_length() if a else M32, None)),
        "s_flbit_i32_i64": ("sop1_64to32", lambda a, b, c: (flbit_signed(a, 64), None)),
    }
    return ops


def _sopc():
    ops = {}
    for cmp, f in (("eq", lambda x, y: x == y), ("lg", lambda x, y: x != y), ("gt", lambda x, y: x > y),
                   ("ge", lambda x, y: x >= y), ("lt", lambda x, y: x < y), ("le", lambda x, y: x <= y)):
        ops[f"s_cmp_{cmp}_i32"] = ("sopc", lambda a, b, c, f=f: (None, int(f(s32(a), s32(b)))))
        ops[f"s_cmp_{cmp}_u32"] = ("sopc", lambda a, b, c, f=f: (None, int(f(a, b))))
        # SOPK compares: SIMM16 sign-extended for i32, zero-extended for u32.
        ops[f"s_cmpk_{cmp}_i32"] = ("cmpk", lambda a, k, c, f=f: (None, int(f(s32(a), sext16(k)))))
        ops[f"s_cmpk_{cmp}_u32"] = ("cmpk", lambda a, k, c, f=f: (None, int(f(a, k & 0xFFFF))))
    ops["s_bitcmp0_b32"] = ("sopc", lambda a, b, c: (None, int(not (a >> (b & 31)) & 1)))
    ops["s_bitcmp1_b32"] = ("sopc", lambda a, b, c: (None, int((a >> (b & 31)) & 1)))
    ops["s_cmp_eq_u64"] = ("sopc_64", lambda a, b, c: (None, int(a == b)))
    ops["s_cmp_lg_u64"] = ("sopc_64", lambda a, b, c: (None, int(a != b)))
    ops["s_movk_i32"] = ("movk", lambda a, k, c: (sext16(k) & M32, None))
    ops["s_addk_i32"] = ("sopk", lambda d, k, c: ((d + sext16(k)) & M32,
                                                  int(not -2**31 <= s32(d) + sext16(k) < 2**31)))
    ops["s_mulk_i32"] = ("sopk", lambda d, k, c: ((s32(d) * sext16(k)) & M32, None))
    return ops


OPS = {**_sop2(), **_sop2_64(), **_sop1(), **_sopc()}

SCC_CONSUMERS = ("s_cselect_b32", "s_addc_u32", "s_subb_u32")
SCC_BRANCHES = ("s_cbranch_scc0", "s_cbranch_scc1")


def writes_scc(name):
    """Whether the opcode writes SCC (each one always does or never does)."""
    form, fn = OPS[name]
    return (fn(1, 2, 0, 3) if form in ("sop1_rmw", "sop1_64_rmw") else fn(1, 2, 0))[1] is not None


def branch_skip(line):
    """For `s_cbranch_scc0/1 +N`: (mnemonic, N); None for any other line."""
    op, _, rest = line.partition(" ")
    if op not in SCC_BRANCHES:
        return None
    if not rest.startswith("+"):
        raise RefError(f"branch {line!r}: expected +N (instructions skipped)")
    return op, int(rest[1:])


# ---- program generation ---------------------------------------------------------------------------

def _value(rng):
    return rng.choice(INTERESTING) if rng.random() < 0.6 else rng.getrandbits(32)


def _src32(rng):
    """An SGPR most of the time, sometimes an inline constant or a literal."""
    r = rng.random()
    if r < 0.75:
        return f"s{rng.choice(REGS)}"
    if r < 0.9:
        return str(rng.randrange(-16, 65))
    return f"0x{_value(rng):x}"


def _pair(r):
    return f"s[{r}:{r + 1}]"


def _src64(rng):
    if rng.random() < 0.85 or not INLINE64:
        return _pair(rng.choice(PAIRS))
    return str(rng.randrange(-16, 65))   # an inline constant, sign-extended to 64 bits


def _bfe_ctl(rng):
    off = rng.randrange(0, 32)
    width = rng.randrange(0, 33 - off)   # offset + width <= 32: the documented range
    return f"0x{(width << 16) | off:x}"


def _shared_literal_sources(src0: str, src1: str, register: int) -> str:
    # A scalar instruction can carry only one literal dword. Equal literals can share it.
    if src0 != src1 and _is_literal(src0) and _is_literal(src1):
        src0 = f"s{register}"
    return f"{src0}, {src1}"


def emit(rng, name):
    form = OPS[name][0]
    d, p = rng.choice(REGS), rng.choice(PAIRS)
    if form == "sop2":
        b = _src32(rng) if name.startswith("s_bfe") and BFE_ANY and rng.random() < 0.3 else             _bfe_ctl(rng) if name.startswith("s_bfe") else _src32(rng)
        return f"{name} s{d}, {_shared_literal_sources(_src32(rng), b, d)}"
    if form == "sop2_64":
        return f"{name} {_pair(p)}, {_src64(rng)}, {_src64(rng)}"
    if form == "sop2_64s":
        return f"{name} {_pair(p)}, {_src64(rng)}, {_src32(rng)}"
    if form == "sop2_bfm64":
        return f"{name} {_pair(p)}, {_shared_literal_sources(_src32(rng), _src32(rng), d)}"
    if form in ("sop1", "sop1_rmw"):
        return f"{name} s{d}, {_src32(rng)}"
    if form == "sop1_64":
        return f"{name} {_pair(p)}, {_src64(rng)}"
    if form in ("sop1_32to64", "sop1_64_rmw"):
        return f"{name} {_pair(p)}, {_src32(rng)}"
    if form == "sop1_64to32":
        return f"{name} s{d}, {_src64(rng)}"
    if form == "sopc":
        return f"{name} {_shared_literal_sources(_src32(rng), _src32(rng), d)}"
    if form == "sopc_64":
        return f"{name} {_src64(rng)}, {_src64(rng)}"
    k = rng.choice((0, 1, 0x7FFF, 0x8000, 0xFFFF, 0xFFFE, rng.getrandbits(16)))
    if form == "cmpk":
        return f"{name} s{d}, 0x{k:x}"
    return f"{name} s{d}, 0x{k:x}"   # movk, sopk


def generate(rng, n_ops, ops, branches=False):
    lines = [f"s_mov_b32 s{r}, 0x{_value(rng):x}" for r in REGS]
    lines.append(f"s_cmp_eq_u32 s{rng.choice(REGS)}, s{rng.choice(REGS)}")   # a known SCC to start with
    if branches:
        lines += _cf_body(rng, n_ops, ops)
    else:
        for _ in range(n_ops):
            lines.append(emit(rng, rng.choice(ops)))
    lines += observe(rng.sample(REGS, 3))
    return lines


def _cf_body(rng, n_ops, ops):
    """n_ops instructions, half of them SCC producers. After a producer: sometimes a chain of SCC consumers,
    sometimes s_cbranch_scc0/1 forward over the next 1..3 instructions (which never branch themselves)."""
    producers = [o for o in ops if writes_scc(o)] or ops
    body = []
    for _ in range(n_ops):
        op = rng.choice(producers if rng.random() < 0.5 else ops)
        body.append(emit(rng, op))
        if not writes_scc(op):
            continue
        r = rng.random()
        if r < 0.35:
            body += [emit(rng, rng.choice(SCC_CONSUMERS)) for _ in range(rng.randint(1, 3))]
        elif r < 0.8:
            k = rng.randint(1, 3)
            body.append(f"{rng.choice(SCC_BRANCHES)} +{k}")
            body += [emit(rng, rng.choice(ops + list(SCC_CONSUMERS))) for _ in range(k)]
    return body


def observe(regs):
    return [f"s_cselect_b32 s{OUT_SCC}, 1, 0",
            f"v_mov_b32 v8, s{regs[0]}", f"v_mov_b32 v9, s{regs[1]}", f"v_mov_b32 v10, s{regs[2]}",
            f"v_mov_b32 v11, s{OUT_SCC}"]


# ---- the reference interpreter -------------------------------------------------------------------

class RefError(Exception):
    pass


def _parse_src(tok, sgpr, width):
    tok = tok.strip()
    if tok.startswith("s["):
        lo = int(tok[2:tok.index(":")])
        if width == 32:
            raise RefError(f"pair {tok} in a 32-bit position")
        return sgpr.get(lo, 0) | sgpr.get(lo + 1, 0) << 32
    if tok.startswith("s"):
        if width == 64:
            raise RefError(f"{tok} in a 64-bit position")
        return sgpr.get(int(tok[1:]), 0)
    v = int(tok, 0)
    if width == 64:
        # inline constants are sign-extended; a 32-bit literal is zero-extended (not generated here)
        return v & M64 if -16 <= v <= 64 else v & M32
    return v & M32


def _write(sgpr, tok, value, width):
    tok = tok.strip()
    if width == 64:
        lo = int(tok[2:tok.index(":")])
        sgpr[lo], sgpr[lo + 1] = value & M32, (value >> 32) & M32
    else:
        sgpr[int(tok[1:])] = value & M32


def reference(lines):
    """Run the scalar program; returns the four exported words."""
    sgpr, scc, vgpr = {}, 0, {}
    skip = 0
    for line in lines:
        if skip:
            skip -= 1
            continue
        br = branch_skip(line)
        if br:
            # SOPP: if SCC == 0 (scc0) / SCC == 1 (scc1) then PC = PC + SIMM16 * 4 + 4; SCC is not written.
            if scc == (0 if br[0] == "s_cbranch_scc0" else 1):
                skip = br[1]
            continue
        op, _, rest = line.partition(" ")
        args = [a.strip() for a in rest.split(", ")] if rest else []
        if op == "v_mov_b32":
            vgpr[int(args[0][1:])] = _parse_src(args[1], sgpr, 32)
            continue
        if op not in OPS:
            raise RefError(f"{op} is not modelled")
        form, fn = OPS[op]
        if form == "sop2":
            r, c = fn(_parse_src(args[1], sgpr, 32), _parse_src(args[2], sgpr, 32), scc)
            _write(sgpr, args[0], r, 32)
        elif form == "sop2_64":
            r, c = fn(_parse_src(args[1], sgpr, 64), _parse_src(args[2], sgpr, 64), scc)
            _write(sgpr, args[0], r, 64)
        elif form == "sop2_64s":
            r, c = fn(_parse_src(args[1], sgpr, 64), _parse_src(args[2], sgpr, 32), scc)
            _write(sgpr, args[0], r, 64)
        elif form == "sop2_bfm64":
            r, c = fn(_parse_src(args[1], sgpr, 32), _parse_src(args[2], sgpr, 32), scc)
            _write(sgpr, args[0], r, 64)
        elif form == "sop1":
            r, c = fn(_parse_src(args[1], sgpr, 32), None, scc)
            _write(sgpr, args[0], r, 32)
        elif form == "sop1_rmw":
            r, c = fn(_parse_src(args[1], sgpr, 32), None, scc, _parse_src(args[0], sgpr, 32))
            _write(sgpr, args[0], r, 32)
        elif form == "sop1_64":
            r, c = fn(_parse_src(args[1], sgpr, 64), None, scc)
            _write(sgpr, args[0], r, 64)
        elif form == "sop1_64_rmw":
            r, c = fn(_parse_src(args[1], sgpr, 32), None, scc, _parse_src(args[0], sgpr, 64))
            _write(sgpr, args[0], r, 64)
        elif form == "sop1_32to64":
            r, c = fn(_parse_src(args[1], sgpr, 32), None, scc)
            _write(sgpr, args[0], r, 64)
        elif form == "sop1_64to32":
            r, c = fn(_parse_src(args[1], sgpr, 64), None, scc)
            _write(sgpr, args[0], r, 32)
        elif form == "sopc":
            _, c = fn(_parse_src(args[0], sgpr, 32), _parse_src(args[1], sgpr, 32), scc)
        elif form == "sopc_64":
            _, c = fn(_parse_src(args[0], sgpr, 64), _parse_src(args[1], sgpr, 64), scc)
        elif form == "cmpk":
            _, c = fn(_parse_src(args[0], sgpr, 32), int(args[1], 0), scc)
        elif form == "movk":
            r, c = fn(None, int(args[1], 0), scc)
            _write(sgpr, args[0], r, 32)
        elif form == "sopk":
            r, c = fn(_parse_src(args[0], sgpr, 32), int(args[1], 0), scc)
            _write(sgpr, args[0], r, 32)
        else:
            raise RefError(f"form {form}")
        if c is not None:
            scc = int(c)
    return [vgpr.get(r, 0) for r in (8, 9, 10, 11)]


# ---- the translator side -------------------------------------------------------------------------

def _is_literal(tok):
    """A 32-bit source that needs a literal dword: not a register or an integer inline constant (-16..64)."""
    tok = tok.strip()
    if tok[0] in "sv":
        return False
    return not -16 <= s32(int(tok, 0)) <= 64


def insn_size(line):
    """Encoded bytes of one generated line: 4, or 8 with a literal (a SOPK immediate sits in SIMM16)."""
    op, _, rest = line.partition(" ")
    if op == "exp":
        return 8
    if op in SCC_BRANCHES or op == "s_endpgm" or (op in OPS and OPS[op][0] in ("sopk", "movk", "cmpk")):
        return 4
    args = rest.split(", ") if rest else []
    srcs = args if op in OPS and OPS[op][0] in ("sopc", "sopc_64") else args[1:]
    return 8 if any(_is_literal(a) for a in srcs) else 4


def listing(lines):
    """(text, size) per line, with `s_cbranch_scc0/1 +N` turned into the dword offset over the next N lines."""
    sizes = [insn_size(line) for line in lines]
    out = []
    for i, line in enumerate(lines):
        br = branch_skip(line)
        if br:
            if i + 1 + br[1] > len(lines):
                raise RefError(f"branch {line!r} past the end of the program")
            line = f"{br[0]} {sum(sizes[i + 1:i + 1 + br[1]]) // 4}"
        out.append((line, sizes[i]))
    return out


def has_branches(lines):
    return any(branch_skip(line) for line in lines)

def translated(lines, name="ps_salu"):
    """Translate and execute; returns the four exported words, or raises Skip."""
    import rdna2spirv
    from spirv_mini_interp import run_fragment
    with tempfile.TemporaryDirectory(prefix="salu-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            if has_branches(lines):
                # The translator places branch targets from the encoding sizes; it does not read the bytes.
                body = listing(lines + ["exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"])
                f.write(f"; {name}  code 0x0..0x{sum(n for _, n in body):x}\n")
                for text, n in body:
                    f.write("\t" + text + " ; encoding: [" + ",".join(["0x00"] * n) + "]\n")
            else:
                f.write(f"; {name}  salu_diff case\n")
                for line in lines:
                    f.write("\t" + line + "\n")
                f.write("\texp mrt0 v8, v9, v10, v11 done vm\n\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spvasm = rdna2spirv.translate(path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            raise Skip(f"translator unsupported: {e}") from None
        try:
            out = run_fragment(spvasm)
        except Exception as e:  # noqa: BLE001 - an interpreter gap is a counted skip, never a pass
            raise Skip(f"mini_interp: {type(e).__name__}: {str(e)[:80]}") from None
    return [w & M32 for w in out["%out_mrt0"]]


class Skip(Exception):
    pass


def _diff(lines, want, got, label, quiet_nan):
    # A signalling-NaN pattern comes back quiet through the interpreter's Python floats; hardware is exact.
    bad = [k for k in range(4) if want[k] != got[k] and not (quiet_nan and is_nan(want[k]) and is_nan(got[k]))]
    if not bad:
        return None
    i = bad[0]
    what = "SCC" if i == 3 else lines[-4 + i].split(", ")[1]
    return f"word {i} ({what}): reference 0x{want[i]:08x}, {label} 0x{got[i]:08x}"


def check(lines, backends=("translator",), hw=None):
    """None on a match, else a description. Raises Skip (every backend skipped) / RefError.
    hw is the hardware result for this program (four words or a str reason), or a callable lines -> that."""
    want = reference(lines)
    problems, skips = [], []
    if "translator" in backends:
        try:
            problems.append(_diff(lines, want, translated(lines), "translated", True))
        except Skip as e:
            skips.append(e)
    if "hw" in backends:
        got = hw(lines) if callable(hw) else hw
        if isinstance(got, str):
            skips.append(Skip(f"hardware: {got}"))
        else:
            problems.append(_diff(lines, want, got, "hardware", False))
    if skips and len(skips) == len(backends):
        raise skips[0]
    return "; ".join(p for p in problems if p) or None


def drop(body, i):
    """body without line i; a branch that skipped over that line now skips one line fewer."""
    out = []
    for j, line in enumerate(body):
        if j == i:
            continue
        br = branch_skip(line)
        if br and j < i <= j + br[1]:
            line = f"{br[0]} +{br[1] - 1}"
        out.append(line)
    return out


def shrink(lines, check_fn=check):
    """Drop instructions (not seeds or the observation) while the program still mismatches."""
    n_seed = len(REGS)
    body = lines[n_seed:-5]
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            trial = lines[:n_seed] + drop(body, i) + lines[-5:]
            try:
                if check_fn(trial):
                    body = drop(body, i)
                    changed = True
                    break
            except (Skip, RefError):
                continue
    return lines[:n_seed] + body + lines[-5:]


def hw_prepare(oracle, programs):
    """Run every program on the hardware in batches. Returns per program the four words, or a str reason it was skipped."""
    from hw_oracle import asm
    out = [None] * len(programs)
    todo = []
    for i, lines in enumerate(programs):
        why = asm.skip_reason(lines, oracle.arch)
        if why:
            out[i] = why
        else:
            todo.append(i)
    for i, r in zip(todo, oracle.run([programs[i] for i in todo])):
        out[i] = r
    return out


def run_suite(programs, seed, max_ops, ops, backends=("translator",), oracle=None, branches=False):
    counts = {"pass": 0, "mismatch": 0, "skip": {}}
    failures = []
    rng = random.Random(seed)
    progs = [generate(rng, rng.randrange(1, max_ops + 1), ops, branches) for _ in range(programs)]
    hw_words = hw_prepare(oracle, progs) if "hw" in backends else [None] * programs
    for i, lines in enumerate(progs):
        try:
            problem = check(lines, backends, hw_words[i])
        except Skip as e:
            counts["skip"][str(e)] = counts["skip"].get(str(e), 0) + 1
            continue
        if problem is None:
            counts["pass"] += 1
            continue
        counts["mismatch"] += 1
        failures.append((i, problem, lines))
    return counts, failures


def self_test():
    ok = check(["s_mov_b32 s4, 5", "s_mov_b32 s5, 7", "s_add_u32 s6, s4, s5", "s_cmp_eq_u32 s4, s4"]
               + observe([4, 5, 6])) is None
    caught = reference(["s_mov_b32 s4, 0xffffffff", "s_add_u32 s5, s4, 1"] + observe([4, 5, 6]))[3] == 1
    try:
        translated(["s_rfe_b64 s[4:5]"] + observe([4, 5, 6]))
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, carry modelled {caught}, unsupported op skipped {skipped}")
    return ok and caught and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scalar ALU / SCC differential fuzzing (FUZZ-4).")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=6)
    ap.add_argument("--ops", default=None, help="comma-separated opcodes (default: all)")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--all-failures", action="store_true", help="print every failing program, not just the first")
    ap.add_argument("--bfe-any", action="store_true",
                    help="also give s_bfe_* control words past offset + width = 32 (measured on gfx1150 and gfx1036: the field stops at bit 31)")
    ap.add_argument("--inline64", action="store_true",
                    help="also use inline constants as 64-bit sources (the translator reads them as lane masks)")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated comparison targets: translator (default), hw (natively on the AMD GPU; see "
                         "tools/hw_oracle/). 'hw' alone compares the hardware with the reference only")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--branches", action="store_true",
                    help="SCC in control flow: consumer chains and s_cbranch_scc0/1 forward over 1..3 instructions "
                         "after SCC producers (translator oracle only)")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    if a.branches and "hw" in backends:
        ap.error("--branches runs against the translator only (the hardware kernels are straight-line)")
    global INLINE64, BFE_ANY
    INLINE64, BFE_ANY = a.inline64, a.bfe_any
    if a.list_ops:
        for name in sorted(OPS):
            print(f"{name:20} {OPS[name][0]}")
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.ops is not None:
        tokens = [t.strip() for t in a.ops.split(",")]
        if not any(tokens) or any(not t for t in tokens):
            ap.error("--ops must be a comma-separated list of opcodes")
        ops = sorted(set(tokens))
    else:
        ops = sorted(o for o in OPS if backends == ("hw",) or o not in LANE_MASK_OPS)
    unknown = [o for o in ops if o not in OPS]
    if unknown:
        ap.error("unknown ops: " + ", ".join(unknown))
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=a.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    try:
        counts, failures = run_suite(a.programs, a.seed, a.max_ops, ops, backends, oracle, a.branches)
        skips = sum(counts["skip"].values())
        mode = "+".join(backends) + (", branches" if a.branches else "")
        print(f"{a.programs} programs (seed {a.seed}, {mode}): {counts['pass']} pass, "
              f"{counts['mismatch']} mismatch, {skips} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
            print(f"  skip x{n}: {reason}")

        def check_fn(lines):
            return check(lines, backends, (lambda l: oracle.run([l])[0]) if oracle else None)
        for i, problem, lines in failures[: None if a.all_failures else 1]:
            small = shrink(lines, check_fn)
            print(f"MISMATCH program {i}: {check_fn(small) or problem}")
            shown = [text for text, _ in listing(small)] if has_branches(small) else small
            print("\n".join("\t" + line for line in shown))
    finally:
        if oracle:
            oracle.close()
    if failures:
        return 1
    if not counts["pass"]:   # nothing was compared: every program was skipped
        print("no program ran: every one was skipped")
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
