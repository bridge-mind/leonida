r"""Differential fuzzing of 32-bit integer VALU in every encoding and source kind. Standard library only.

Generates seeded straight-line programs of integer vector instructions, one lane, runs them through

  * the reference below, written from the RDNA2 ISA guide's pseudocode for each opcode, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py (lane 0 of a wave32),

and compares four exported words bit for bit (v8..v11). The reference does not share code with the translator, with
tools/cpu_ref.py or with the other fuzzers. tools/shader_diff.py covers many of the same opcodes with VGPR sources in
their e32 form; this one puts every legal source kind in every legal position and adds the VOP3-only opcodes.

What is modelled, per instruction:
  * encodings: VOP1 and VOP2 opcodes as e32 and as e64, the VOP3-only opcodes (OPS, --list-ops), and VOPC e64
    into an SGPR mask (s20, s21) or VCC followed by v_cndmask_b32_e64 on it (one lane: the mask is that lane's bit).
  * sources: VGPRs v0..v7 and v12..v19, SGPRs s4..s19 (seeded with s_mov_b32, so the translator knows them as
    constants), integer inline constants -16..64, float inline constants (0.5, 1.0, 2.0, 4.0, their negatives and
    1/(2*pi)) as their f32 bits, and 32-bit literals. An e32 VOP2 has a VGPR in src1 and a literal only in src0; a
    VOP3 form has at most one literal and at most two scalar values (SGPRs and the literal: the GFX10 constant bus).
  * v_mad_u32_u16 / v_mad_i32_i16 with op_sel on the two 16-bit sources (the high halves). A literal there has 16
    bits (zero-extended: its high half is 0); a float inline constant is its f32 bits (high half 0x3f00 for 0.5).
  * v_bfe_u32 / v_bfe_i32: offset and width are their low 5 bits, and the field stops at bit 31
    (width = min(width, 32 - offset), as for s_bfe in tools/salu_diff.py).
  * v_perm_b32: the selector bytes 0..7 pick a byte of {src0, src1}, 8..11 replicate a sign bit, 12 is 0x00,
    13 and up 0xff.
Every rule was checked against an RDNA2 GPU (AMD Radeon iGPU, gfx1036, Windows driver 26.10.1) with --oracle hw.

Not generated: VOP3 neg/abs/clamp/omod on integer ops, DPP and SDWA forms (tools/sdwa_diff.py), 16-bit and packed
ops, carries and anything that depends on other lanes (tools/wint_diff.py), 64-bit ops.

Known divergences (KNOWN_DIVERGENCES): by default the generator leaves out the forms where the translator is known to
differ from the hardware, so that a run shows new problems only; --known puts them back (a hardware-only run always
has them):
  * mask-as-data: a lane mask written by a compare (s20, s21, vcc_lo) read as a 32-bit VALU source. The hardware
    reads the wave's bits (here 0 or 1), the translator its per-lane view of the mask, 0 or 0xffffffff (#293).

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/), one lane of a wave32 compute
dispatch, and the four words are compared with the reference: that validates the reference itself. Without the
oracle tools the run exits 77.

Usage:
    python -I tools/vint_diff.py [--programs N] [--seed S] [--max-ops N] [--ops v_add3_u32,v_perm_b32,cmp]
    python -I tools/vint_diff.py --no-literals          # sources without 32-bit literals
    python -I tools/vint_diff.py --known                # also the forms in KNOWN_DIVERGENCES
    python -I tools/vint_diff.py --oracle hw            # hardware vs reference only
    python -I tools/vint_diff.py --oracle translator,hw
    python -I tools/vint_diff.py --list-ops
    python -I tools/vint_diff.py --self-test

Exit 0 when every program matches, 1 on a mismatch (the shortest failing program is printed after removing
instructions that do not matter), 2 on a harness error, 77 when the requested hardware oracle is unavailable.
Skips are counted with their reasons.
"""
import argparse
import gc
import os
import random
import re
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

VREGS = tuple(range(0, 8)) + tuple(range(12, 20))   # data VGPRs; v8..v11 are the observation
SREGS = tuple(range(4, 20))                          # data SGPRs
MASKS = ("s20", "s21", "vcc_lo")
N_SEED = len(VREGS) + len(SREGS)
N_OBS = 4
INLINE_F = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000, "2.0": 0x40000000,
            "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000, "0.15915494": 0x3E22F983}
INTERESTING = (0, 1, 2, 31, 32, 33, 0x7FFFFFFF, 0x80000000, M32, 0xFFFFFFFE, 0xFFFF, 0x10000, 0x8000,
               0x00FFFFFF, 0x00800000, 0xFF800000, 0x12345678, 0xDEADBEEF, 0x55555555, 0xAAAAAAAA, 0x0C0D0E0F,
               0x03020100, 0x07060504, 0x0B0A0908, 0x01000302)


class Skip(Exception):
    pass


class RefError(Exception):
    pass


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sx(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def ffbh_u(a):
    return M32 if a == 0 else 32 - a.bit_length()


def ffbh_i(a):
    if a in (0, M32):
        return M32
    return 32 - ((~a & M32) if a >> 31 else a).bit_length()


def ffbl(a):
    return M32 if a == 0 else (a & -a).bit_length() - 1


def bfe(a, off, w, signed):
    off, w = off & 31, w & 31
    if w == 0:
        return 0
    w = min(w, 32 - off)
    f = (a >> off) & ((1 << w) - 1)
    return (sx(f, w) if signed else f) & M32


def perm(s0, s1, sel):
    x = (s0 << 32) | s1
    out = 0
    for i in range(4):
        b = (sel >> (8 * i)) & 0xFF
        if b >= 13:
            v = 0xFF
        elif b == 12:
            v = 0
        elif b >= 8:
            v = 0xFF if (x >> (16 * (b - 8) + 15)) & 1 else 0   # 8: bit 15, 9: bit 31, 10: bit 47, 11: bit 63
        else:
            v = (x >> (8 * b)) & 0xFF
        out |= v << (8 * i)
    return out


def sat(v, lo, hi):
    return max(lo, min(hi, v))


# name -> (encoding, nsrc, fn on 32-bit sources). vop1 / vop2: generated as e32 and e64; vop3: VOP3-only.
OPS = {
    "v_mov_b32": ("vop1", 1, lambda a: a),
    "v_not_b32": ("vop1", 1, lambda a: ~a & M32),
    "v_bfrev_b32": ("vop1", 1, lambda a: int(f"{a:032b}"[::-1], 2)),
    "v_ffbh_u32": ("vop1", 1, ffbh_u),
    "v_ffbh_i32": ("vop1", 1, ffbh_i),
    "v_ffbl_b32": ("vop1", 1, ffbl),
    "v_add_nc_u32": ("vop2", 2, lambda a, b: (a + b) & M32),
    "v_sub_nc_u32": ("vop2", 2, lambda a, b: (a - b) & M32),
    "v_subrev_nc_u32": ("vop2", 2, lambda a, b: (b - a) & M32),
    "v_and_b32": ("vop2", 2, lambda a, b: a & b),
    "v_or_b32": ("vop2", 2, lambda a, b: a | b),
    "v_xor_b32": ("vop2", 2, lambda a, b: a ^ b),
    "v_xnor_b32": ("vop2", 2, lambda a, b: ~(a ^ b) & M32),
    "v_lshlrev_b32": ("vop2", 2, lambda a, b: (b << (a & 31)) & M32),
    "v_lshrrev_b32": ("vop2", 2, lambda a, b: b >> (a & 31)),
    "v_ashrrev_i32": ("vop2", 2, lambda a, b: (s32(b) >> (a & 31)) & M32),
    "v_min_i32": ("vop2", 2, lambda a, b: min(s32(a), s32(b)) & M32),
    "v_max_i32": ("vop2", 2, lambda a, b: max(s32(a), s32(b)) & M32),
    "v_min_u32": ("vop2", 2, min),
    "v_max_u32": ("vop2", 2, max),
    "v_mul_i32_i24": ("vop2", 2, lambda a, b: (sx(a, 24) * sx(b, 24)) & M32),
    "v_mul_hi_i32_i24": ("vop2", 2, lambda a, b: ((sx(a, 24) * sx(b, 24)) >> 32) & M32),
    "v_mul_u32_u24": ("vop2", 2, lambda a, b: ((a & 0xFFFFFF) * (b & 0xFFFFFF)) & M32),
    "v_mul_hi_u32_u24": ("vop2", 2, lambda a, b: ((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 32),
    "v_add3_u32": ("vop3", 3, lambda a, b, c: (a + b + c) & M32),
    "v_lshl_add_u32": ("vop3", 3, lambda a, b, c: ((a << (b & 31)) + c) & M32),
    "v_add_lshl_u32": ("vop3", 3, lambda a, b, c: (((a + b) & M32) << (c & 31)) & M32),
    "v_lshl_or_b32": ("vop3", 3, lambda a, b, c: ((a << (b & 31)) & M32) | c),
    "v_and_or_b32": ("vop3", 3, lambda a, b, c: (a & b) | c),
    "v_or3_b32": ("vop3", 3, lambda a, b, c: a | b | c),
    "v_xor3_b32": ("vop3", 3, lambda a, b, c: a ^ b ^ c),
    "v_xad_u32": ("vop3", 3, lambda a, b, c: ((a ^ b) + c) & M32),
    "v_bfe_u32": ("vop3", 3, lambda a, b, c: bfe(a, b, c, False)),
    "v_bfe_i32": ("vop3", 3, lambda a, b, c: bfe(a, b, c, True)),
    "v_bfi_b32": ("vop3", 3, lambda a, b, c: (a & b) | (~a & c & M32)),
    "v_alignbit_b32": ("vop3", 3, lambda a, b, c: (((a << 32) | b) >> (c & 31)) & M32),
    "v_alignbyte_b32": ("vop3", 3, lambda a, b, c: (((a << 32) | b) >> (8 * (c & 3))) & M32),
    "v_perm_b32": ("vop3", 3, perm),
    "v_min3_i32": ("vop3", 3, lambda a, b, c: min(s32(a), s32(b), s32(c)) & M32),
    "v_max3_i32": ("vop3", 3, lambda a, b, c: max(s32(a), s32(b), s32(c)) & M32),
    "v_med3_i32": ("vop3", 3, lambda a, b, c: sorted((s32(a), s32(b), s32(c)))[1] & M32),
    "v_min3_u32": ("vop3", 3, lambda a, b, c: min(a, b, c)),
    "v_max3_u32": ("vop3", 3, lambda a, b, c: max(a, b, c)),
    "v_med3_u32": ("vop3", 3, lambda a, b, c: sorted((a, b, c))[1]),
    "v_mul_lo_u32": ("vop3", 2, lambda a, b: (a * b) & M32),
    "v_mul_hi_u32": ("vop3", 2, lambda a, b: (a * b) >> 32),
    "v_mul_hi_i32": ("vop3", 2, lambda a, b: ((s32(a) * s32(b)) >> 32) & M32),
    "v_mad_u32_u24": ("vop3", 3, lambda a, b, c: ((a & 0xFFFFFF) * (b & 0xFFFFFF) + c) & M32),
    "v_mad_i32_i24": ("vop3", 3, lambda a, b, c: (sx(a, 24) * sx(b, 24) + s32(c)) & M32),
    "v_mad_u32_u16": ("vop3", 3, lambda a, b, c: ((a & 0xFFFF) * (b & 0xFFFF) + c) & M32),
    "v_mad_i32_i16": ("vop3", 3, lambda a, b, c: (sx(a, 16) * sx(b, 16) + s32(c)) & M32),
    "v_bcnt_u32_b32": ("vop3", 2, lambda a, b: (bin(a).count("1") + b) & M32),
    "v_bfm_b32": ("vop3", 2, lambda a, b: (((1 << (a & 31)) - 1) << (b & 31)) & M32),
    "v_cvt_pk_u16_u32": ("vop3", 2, lambda a, b: min(a, 0xFFFF) | (min(b, 0xFFFF) << 16)),
    "v_cvt_pk_i16_i32": ("vop3", 2, lambda a, b: (sat(s32(a), -32768, 32767) & 0xFFFF)
                         | ((sat(s32(b), -32768, 32767) & 0xFFFF) << 16)),
}
OP_SEL_OPS = ("v_mad_u32_u16", "v_mad_i32_i16")
CMPS = {
    "eq_u32": lambda a, b: a == b, "ne_u32": lambda a, b: a != b, "lt_u32": lambda a, b: a < b,
    "le_u32": lambda a, b: a <= b, "gt_u32": lambda a, b: a > b, "ge_u32": lambda a, b: a >= b,
    "lt_i32": lambda a, b: s32(a) < s32(b), "le_i32": lambda a, b: s32(a) <= s32(b),
    "gt_i32": lambda a, b: s32(a) > s32(b), "ge_i32": lambda a, b: s32(a) >= s32(b),
    "eq_i32": lambda a, b: a == b, "ne_i32": lambda a, b: a != b,
    "f_u32": lambda a, b: False, "t_u32": lambda a, b: True, "f_i32": lambda a, b: False, "t_i32": lambda a, b: True,
}
ALL_OPS = sorted(OPS) + ["cmp"]   # cmp: a VOPC e64 into a mask, then v_cndmask_b32_e64 on it

KNOWN = False   # also generate the forms in KNOWN_DIVERGENCES (--known; always on for a hardware-only run)
LITERALS = True
KNOWN_DIVERGENCES = {
    "mask-as-data": "a compare's lane mask (s20, s21, vcc_lo) read as a 32-bit VALU source: the hardware reads the "
                    "wave's bits, the translator 0 or 0xffffffff per lane (#293)",
}


# ---- program generation -------------------------------------------------------------------------------------

def rand_value(rng):
    r = rng.random()
    if r < 0.4:
        return rng.choice(INTERESTING)
    if r < 0.55:
        return rng.randrange(0, 64)
    return rng.getrandbits(32)


class Gen:
    def __init__(self, rng):
        self.r = rng
        self.masks = set()    # mask registers a compare has written (read as data only with KNOWN)

    def vreg(self):
        return f"v{self.r.choice(VREGS)}"

    def const(self, lit_ok, lit16=False):
        r = self.r
        k = r.random()
        if k < 0.2:
            return r.choice(tuple(INLINE_F))
        if k < 0.6 or not lit_ok:
            return str(r.randint(-16, 64))
        v = rand_value(r) & (0xFFFF if lit16 else M32)
        return f"0x{v:x}" if not -16 <= s32(v) <= 64 else ("0x1234" if lit16 else "0x12345678")

    def srcs(self, n, vop3, e32_vop2=False, lit16=0):
        """n sources. e32: src0 any kind (a literal only there), src1 a VGPR. VOP3: at most one literal and at most
        two scalar values, the literal counted among them. The first lit16 sources are 16-bit operands: a literal
        there must fit 16 bits (the assembler rejects a wider one)."""
        r = self.r
        out, bus, lit = [], set(), False
        for i in range(n):
            k = r.random()
            if (e32_vop2 and i == 1) or k < 0.5:
                out.append(self.vreg())
            elif k < 0.75:
                if KNOWN and self.masks and r.random() < 0.25:
                    s = r.choice(sorted(self.masks))
                else:
                    s = f"s{r.choice(SREGS)}"
                if s not in bus and len(bus) >= 2:
                    out.append(self.vreg())
                    continue
                bus.add(s)
                out.append(s)
            else:
                lit_ok = LITERALS and not lit and len(bus) < 2 and (vop3 or i == 0)
                c = self.const(lit_ok, i < lit16)
                if c.startswith("0x"):
                    lit = True
                    bus.add("lit")
                out.append(c)
        return out

    def op(self, ops):
        r = self.r
        name = r.choice(ops)
        d = r.choice(VREGS)
        if name == "cmp":
            c = r.choice(tuple(CMPS))
            m = r.choice(MASKS)
            a, b = self.srcs(2, True)
            # the mask is on the constant bus of the v_cndmask too: its sources are VGPRs or inline constants
            sel = [x if not x.startswith(("s", "0x")) else self.vreg() for x in self.srcs(2, True)]
            self.masks.add(m)
            return [f"v_cmp_{c}_e64 {m}, {a}, {b}", f"v_cndmask_b32_e64 v{d}, {sel[0]}, {sel[1]}, {m}"]
        enc, n, _ = OPS[name]
        if enc == "vop1":
            if r.random() < 0.5:
                return [f"{name}_e32 v{d}, {self.srcs(1, False)[0]}"]
            return [f"{name}_e64 v{d}, {self.srcs(1, True)[0]}"]
        if enc == "vop2":
            if r.random() < 0.5:
                return [f"{name}_e32 v{d}, {', '.join(self.srcs(2, False, True))}"]
            return [f"{name}_e64 v{d}, {', '.join(self.srcs(2, True))}"]
        line = f"{name} v{d}, {', '.join(self.srcs(n, True, lit16=2 if name in OP_SEL_OPS else 0))}"
        if name in OP_SEL_OPS and r.random() < 0.5:
            line += f" op_sel:[{r.randint(0, 1)},{r.randint(0, 1)},0,0]"
        return [line]


def seeds(rng):
    return ([f"v_mov_b32_e32 v{v}, 0x{rand_value(rng):x}" for v in VREGS]
            + [f"s_mov_b32 s{s}, 0x{rand_value(rng):x}" for s in SREGS])


def observe(regs):
    return [f"v_mov_b32_e32 v{8 + i}, v{r}" for i, r in enumerate(regs)]


def generate(rng, n_ops, ops):
    gen = Gen(rng)
    body = []
    for _ in range(n_ops):
        body += gen.op(ops)
    written = []
    for line in reversed(body):
        if line.startswith("v_cmp"):
            continue
        dst = int(line.split()[1].rstrip(",")[1:])
        if dst not in written:
            written.append(dst)
    regs = written[:4]
    regs += [rng.choice(VREGS) for _ in range(4 - len(regs))]
    return seeds(rng) + body + observe(regs)


# ---- the reference interpreter --------------------------------------------------------------------------------

_OP_SEL = re.compile(r"\s+op_sel:\[([01,]+)\]\s*$")


def value(tok, st):
    if re.fullmatch(r"v\d+", tok):
        return st["v"].get(int(tok[1:]), 0)
    if re.fullmatch(r"s\d+|vcc_lo", tok):
        if tok not in st["s"]:
            raise RefError(f"{tok} read before it is written")
        return st["s"][tok]
    if tok in INLINE_F:
        return INLINE_F[tok]
    return int(tok, 0) & M32


def reference(lines):
    """Run the program; returns the four exported words."""
    st = {"v": {}, "s": {}}
    for line in lines:
        op, _, rest = line.partition(" ")
        sel = None
        m = _OP_SEL.search(rest)
        if m:
            sel = [int(x) for x in m.group(1).split(",")]
            rest = rest[:m.start()]
        args = [a.strip() for a in rest.split(",")]
        if op == "s_mov_b32":
            st["s"][args[0]] = value(args[1], st)
            continue
        if op.startswith("v_cmp_"):
            c = op[len("v_cmp_"):-len("_e64")]
            st["s"][args[0]] = int(CMPS[c](value(args[1], st), value(args[2], st)))   # this lane's bit
            continue
        if op == "v_cndmask_b32_e64":
            st["v"][int(args[0][1:])] = value(args[2] if value(args[3], st) & 1 else args[1], st)
            continue
        base = op[:-4] if op.endswith(("_e32", "_e64")) else op
        if base not in OPS:
            raise RefError(f"{op} is not modelled")
        vals = [value(a, st) for a in args[1:]]
        if sel:
            vals = [v >> 16 if i < 2 and sel[i] else v for i, v in enumerate(vals)]
        st["v"][int(args[0][1:])] = OPS[base][2](*vals) & M32
    return [st["v"].get(8 + i, 0) for i in range(4)]


# ---- the translator side -----------------------------------------------------------------------------------

def translated(lines, name="ps_vint"):
    """Translate and execute; returns the four exported words, or raises Skip."""
    import rdna2spirv
    from spirv_mini_interp import run_fragment
    with tempfile.TemporaryDirectory(prefix="vint-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  vint_diff case\n")
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


def hw_kernel(lines, wave=32):
    """A raw kernel for tools/hw_oracle: the program on one lane, then v8..v11 stored at the output pointer. wave 64
    runs it in a wave64 wave with the lane masks as 64-bit pairs."""
    from hw_oracle import asm
    if wave == 64:
        import vk_fuzz_run
        lines = [vk_fuzz_run.wave64_text(ln) for ln in lines]
    return asm.Kernel(list(lines) + ["v_mov_b32 v0, 0", "global_store_b128 v0, v[8:11], s[2:3]", "s_endpgm"],
                      wave=wave, threads=1, words=4, raw=True)


# ---- comparison, shrinking, the run --------------------------------------------------------------------------

def _diff(lines, want, got, label):
    bad = [k for k in range(4) if want[k] != got[k]]
    if not bad:
        return None
    i = bad[0]
    return f"word {i} ({lines[-N_OBS + i].split(', ')[1]}): reference 0x{want[i]:08x}, {label} 0x{got[i]:08x}"


def check(lines, backends=("translator",), hw=None):
    """None on a match, else a description. Raises Skip (every backend skipped) / RefError.
    hw is the hardware result for this program (four words or a str reason), or a callable lines -> that."""
    want = reference(lines)
    problems, skips = [], []
    if "translator" in backends:
        try:
            problems.append(_diff(lines, want, translated(lines), "translated"))
        except Skip as e:
            skips.append(e)
    if "hw" in backends:
        got = hw(lines) if callable(hw) else hw
        if isinstance(got, str):
            skips.append(Skip(f"hardware: {got}"))
        else:
            problems.append(_diff(lines, want, got, "hardware"))
    if skips and len(skips) == len(backends):
        raise skips[0]
    return "; ".join(p for p in problems if p) or None


def shrink(lines, check_fn=check):
    """Observe one register if that still fails, drop instructions (a compare and its v_cndmask together) while the
    program still mismatches, then the seeds of registers nothing reads."""
    def fails(trial):
        try:
            return bool(check_fn(trial))
        except (Skip, RefError):
            return False
    seed, body, obs = lines[:N_SEED], lines[N_SEED:-N_OBS], lines[-N_OBS:]
    for line in obs:
        one = [line.replace(line.split()[1], f"v{8 + i},") for i in range(4)]
        if fails(seed + body + one):
            obs = one
            break
    changed = True
    while changed:
        changed = False
        i = len(body) - 1
        while i >= 0:
            n = 2 if body[i].startswith("v_cndmask") and i and body[i - 1].startswith("v_cmp") else 1
            trial = body[:i + 1 - n] + body[i + 1:]
            if fails(seed + trial + obs):
                body, changed = trial, True
            i -= n
    used = set()
    for line in body + obs:
        used |= set(re.findall(r"\b[vs]\d+\b", line.partition(" ")[2]))
    seed = [line for line in seed if line.split()[1].rstrip(",") in used]
    return seed + body + obs


def run_suite(programs, seed, max_ops, ops, backends=("translator",), oracle=None, wave=32, build=None, work=None,
              vk_constants=False):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "vk": {}, "vk_failures": []}
    failures = []
    rng = random.Random(seed)
    progs = [generate(rng, rng.randrange(1, max_ops + 1), ops) for _ in range(programs)]
    hw = oracle.run([hw_kernel(p, wave) for p in progs]) if "hw" in backends else [None] * programs
    if "vk" in backends:
        import vk_fuzz_run
        counts["vk"], counts["vk_failures"] = vk_fuzz_run.judge(progs, hw, vk_fuzz_run.exact_verdict, wave=wave,
                                                                  build=build, work_parent=work, opaque=not vk_constants)
    backends = tuple(b for b in backends if b != "vk")
    for i, lines in enumerate(progs):
        try:
            problem = check(lines, backends, hw[i])
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
    """Harness canaries: a hand program matches, the reference gives hand-worked values, a bad form is a skip."""
    base = [f"v_mov_b32_e32 v{v}, 0x{0x01010101 * v:x}" for v in VREGS] + [f"s_mov_b32 s{s}, {s}" for s in SREGS]
    body = ["v_add3_u32 v1, v2, s4, 0x10",                           # 0x02020202 + 4 + 0x10
            "v_perm_b32 v3, v2, v3, 0x0c0a0400",                     # bytes: v3.b0, v2.b0, sign(bit 47), 0
            "v_cmp_gt_u32_e64 s20, v1, v2",
            "v_cndmask_b32_e64 v4, 1, 2, s20"]
    lines = base + body + observe([1, 3, 4, 2])
    ok = check(lines) is None
    worked = reference(lines) == [0x02020216, 0x00000203, 2, 0x02020202]
    try:
        translated(base + ["v_add_nc_u32_sdwa_dpp v1, v2, v3"] + observe([1, 2, 3, 4]))
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, hand values {worked}, unsupported form skipped {skipped}")
    return ok and worked and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="32-bit integer VALU differential fuzzing (all encodings and sources).")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=6)
    ap.add_argument("--ops", default=None, help="comma-separated opcodes (--list-ops; cmp = compare + v_cndmask)")
    ap.add_argument("--no-literals", action="store_true", help="no 32-bit literal sources")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--all-failures", action="store_true", help="print every failing program, not just the first")
    ap.add_argument("--known", action="store_true",
                    help="also generate the forms where the translator is known to differ (KNOWN_DIVERGENCES; "
                         "always on with --oracle hw alone)")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated comparison targets: translator (default), hw (natively on an AMD GPU; see "
                         "tools/hw_oracle/). 'hw' alone compares the hardware with the reference only")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--wave", type=int, choices=(32, 64), default=32,
                    help="the wave size of the hardware kernels and of the vk programs (64 needs native 64-wide "
                         "subgroups; lane 0 is judged)")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(HERE), "runtime", "build"),
                    help="with vk: the folder with vk_compute_test")
    ap.add_argument("--vk-constants", action="store_true",
                    help="with vk: keep the seeds as literal v_mov_b32 constants (the driver may fold them)")
    ap.add_argument("--work", default=None, help="with vk: a folder for the temporary files (VK_FUZZ_KEEP=1 keeps them)")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw", "vk") for t in backends):
        ap.error("--oracle takes translator, hw, vk (with hw) or a comma-separated mix")
    if "vk" in backends and "hw" not in backends:
        ap.error("--oracle vk judges the translated programs against the hardware's words: use --oracle hw,vk")
    global KNOWN, LITERALS
    KNOWN = a.known or backends == ("hw",)
    LITERALS = not a.no_literals
    if a.list_ops:
        for name in ALL_OPS:
            if name == "cmp":
                print(f"{'cmp':20} VOPC e64 into s20/s21/vcc_lo + v_cndmask_b32_e64: {', '.join(sorted(CMPS))}")
                continue
            enc, n, _ = OPS[name]
            kind = "VOP3 only" if enc == "vop3" else f"{enc.upper()} e32/e64"
            print(f"{name:20} {kind}, {n} sources{', op_sel' if name in OP_SEL_OPS else ''}")
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.ops is not None:
        tokens = [t.strip() for t in a.ops.split(",")]
        if not any(tokens) or any(not t for t in tokens):
            ap.error("--ops must be a comma-separated list of opcodes")
        ops = sorted(set(tokens))
    else:
        ops = list(ALL_OPS)
    unknown = [o for o in ops if o not in ALL_OPS]
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
        counts, failures = run_suite(a.programs, a.seed, a.max_ops, ops, backends, oracle, a.wave, a.build, a.work,
                                     a.vk_constants)
        vk_bad = counts.pop("vk_failures")
        skips = sum(counts["skip"].values())
        print(f"{a.programs} programs (seed {a.seed}, {'+'.join(backends)}{', known' if KNOWN else ''}): "
              f"{counts['pass']} pass, {counts['mismatch']} mismatch, {skips} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
            print(f"  skip x{n}: {reason}")

        want_vk = "vk" in backends
        if want_vk:
            print(f"translated on the Vulkan GPU (wave{a.wave}) against the hardware: "
                  + ", ".join(f"{k} {n}" for k, n in sorted(counts["vk"].items(), key=lambda kv: kv[0].startswith("skip"))))
            for i, detail, lines in vk_bad[: None if a.all_failures else 3]:
                print(f"VK MISMATCH program {i}: {detail}")
                print("\n".join("\t" + line for line in lines))
        backends = tuple(b for b in backends if b != "vk")

        def check_fn(lines):
            return check(lines, backends, (lambda l: oracle.run([hw_kernel(l, a.wave)])[0]) if oracle else None)
        for i, problem, lines in failures[: None if a.all_failures else 1]:
            small = shrink(lines, check_fn)
            print(f"MISMATCH program {i}: {check_fn(small) or problem}")
            print("\n".join("\t" + line for line in small))
    except (RefError, RuntimeError) as e:
        print(f"harness error: {e}")
        return 2
    finally:
        if oracle:
            oracle.close()
    if failures or vk_bad:
        return 1
    if not counts["pass"]:   # nothing was compared: every program was skipped
        print("no program ran: every one was skipped")
        return 2
    if want_vk:
        import vk_fuzz_run
        if vk_fuzz_run.all_skipped(counts["vk"]):
            print("no Vulkan program ran: every one was skipped")
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
