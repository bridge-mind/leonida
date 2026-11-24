r"""Whole-wave differential fuzzing of integer VALU under random EXEC masks. Standard library only.

Generates seeded straight-line programs of integer vector instructions with EXEC rewritten between them, runs them
through

  * the reference below, a per-lane model written from the RDNA2 ISA guide's pseudocode, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_wave_interp.py (one subgroup = one wave,
    every lane at once),

and compares four exported words of every lane bit for bit. The reference does not share code with the translator,
with tools/cpu_ref.py or with the other fuzzers. The one-lane fuzzers (shader_diff.py, salu_diff.py, sdwa_diff.py)
cannot see what this checks: which lanes an instruction writes, and the lane masks that compares and carries produce
and consume.

What is modelled:
  * EXEC: `s_mov_b32 exec_lo, K` (and `exec_hi` in wave64) between instructions, K from a set of edge masks and
    random words; an inactive lane keeps its VGPRs, and gets a 0 bit in every mask an instruction writes.
    v_cmpx_*_e64 writes EXEC (inactive lanes 0).
  * lane masks: VOPC e64 into an SGPR mask (s20, s21 / s[20:21], s[22:23]) or VCC, VOPC e32 into VCC,
    v_cndmask_b32 e64 on any of them and e32 on VCC; an SGPR mask can also be overwritten with plain data
    (s_mov_b32), which a mask read takes bit L of for lane L.
  * carries: v_add/sub/subrev_co_u32 e64 (carry out to a mask), v_add/sub/subrev_co_ci_u32 e64 (carry in from a
    mask) and e32 (VCC in and out); v_mad_u64_u32 / v_mad_i64_i32 with a mask carry-out (bit 64 of the 65-bit sum;
    for the signed form, the sign of the exact result, as in docs/design.md).
  * 32-bit integer VOP1/VOP2 (e32 and e64) and VOP3-only ops (OPS32), and 16-bit integer ops in their e64 / VOP3
    form with op_sel (OPS16): the result goes to the half op_sel[n] names and the other half is kept.
  * sources: VGPRs, SGPRs s4..s19 (seeded with s_mov_b32), integer inline constants -16..64; at most two scalar
    values per VOP3 instruction (the GFX10 constant bus), a mask operand counted among them.
The prologue sets v0 to the lane number (v_mbcnt), seeds VGPRs with a different value per lane plus two
v_writelane_b32 patches each, the SGPRs, and every mask register from a compare, all with EXEC all on.
The epilogue turns EXEC back on and copies four VGPRs (the last ones written) to v8..v11.

Wave size: --wave 32 (default), 64 or both. A wave64 program is detected by the translator from v_mbcnt_hi and the
64-bit masks; when it decides otherwise, the program is a counted skip.

Known divergences (KNOWN_DIVERGENCES): by default the generator leaves out the forms where the translator is known to
differ from the hardware, so that a run shows new problems only; --known puts them back:
  * vcc-data-mask: VCC written with plain data (`s_mov_b32 vcc_lo, K`, and `vcc_hi` in wave64) and then read as a
    lane mask (v_cndmask, carry-in). The translator selects every lane for any non-zero VCC; wave32 is fixed by #2206,
    wave64 is open (finding in #1902).
  * mask-as-data: a lane mask SGPR or VCC read as a 32-bit VALU source. The translator's view of a mask is per lane;
    #293 is the open change that gives the wave's bits.
A hardware-only run (--oracle hw) always generates them.

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/), one workgroup of one wave,
and every lane's four words are compared with the reference: that validates the reference itself. Instructions are
separated by s_nop 4 there so no VALU -> SGPR -> VALU hazard depends on the assembler. Without the oracle tools the
run exits 77.

Usage:
    python -I tools/wint_diff.py [--programs N] [--seed S] [--max-ops N] [--wave 32|64|both]
    python -I tools/wint_diff.py --ops v_add_co_ci_u32,v_cndmask_b32,exec
    python -I tools/wint_diff.py --known                # also the forms in KNOWN_DIVERGENCES
    python -I tools/wint_diff.py --oracle hw             # hardware vs reference only
    python -I tools/wint_diff.py --oracle translator,hw
    python -I tools/wint_diff.py --list-ops
    python -I tools/wint_diff.py --self-test

Exit 0 when every program matches, 1 on a mismatch (the shortest failing program is printed after removing
instructions that do not matter), 2 on a harness error or when every program was skipped, 77 when the requested hardware oracle is unavailable.
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

M16 = 0xFFFF
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

DREGS = tuple(range(1, 8)) + tuple(range(12, 20))   # data VGPRs; v0 = lane number, v8..v11 = the observation
PAIRS = (2, 4, 6, 12, 14, 16, 18)                     # 64-bit destinations / addends of v_mad_*64
SREGS = tuple(range(4, 20))                           # data SGPRs
OBS = (8, 9, 10, 11)

INTERESTING = (0, 1, 2, 31, 32, 0x7FFFFFFF, 0x80000000, M32, 0xFFFFFFFE, 0xFFFF, 0x10000, 0x8000, 0x00FFFFFF,
               0x12345678, 0xDEADBEEF, 0x55555555, 0xAAAAAAAA)
EXEC_EDGES = (M32, 0, 1, 0x80000000, 0x0000FFFF, 0xFFFF0000, 0x55555555)


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


def bfe_u(a, off, w):
    off, w = off & 31, w & 31
    return (a >> off) & ((1 << min(w, 32 - off)) - 1) if w else 0


# ---- the reference tables ------------------------------------------------------------------------------------
# 32-bit ops: name -> (encoding, nsrc, fn on 32-bit sources). vop1 / vop2 are generated as e32 or e64, vop3 bare.
OPS32 = {
    "v_mov_b32": ("vop1", 1, lambda a: a),
    "v_not_b32": ("vop1", 1, lambda a: ~a & M32),
    "v_ffbh_u32": ("vop1", 1, lambda a: M32 if a == 0 else 32 - a.bit_length()),
    "v_add_nc_u32": ("vop2", 2, lambda a, b: (a + b) & M32),
    "v_sub_nc_u32": ("vop2", 2, lambda a, b: (a - b) & M32),
    "v_subrev_nc_u32": ("vop2", 2, lambda a, b: (b - a) & M32),
    "v_and_b32": ("vop2", 2, lambda a, b: a & b),
    "v_or_b32": ("vop2", 2, lambda a, b: a | b),
    "v_xor_b32": ("vop2", 2, lambda a, b: a ^ b),
    "v_lshlrev_b32": ("vop2", 2, lambda a, b: (b << (a & 31)) & M32),
    "v_lshrrev_b32": ("vop2", 2, lambda a, b: b >> (a & 31)),
    "v_ashrrev_i32": ("vop2", 2, lambda a, b: (s32(b) >> (a & 31)) & M32),
    "v_min_i32": ("vop2", 2, lambda a, b: min(s32(a), s32(b)) & M32),
    "v_max_u32": ("vop2", 2, max),
    "v_mul_u32_u24": ("vop2", 2, lambda a, b: ((a & 0xFFFFFF) * (b & 0xFFFFFF)) & M32),
    "v_add3_u32": ("vop3", 3, lambda a, b, c: (a + b + c) & M32),
    "v_lshl_or_b32": ("vop3", 3, lambda a, b, c: ((a << (b & 31)) & M32) | c),
    "v_bfe_u32": ("vop3", 3, bfe_u),
    "v_alignbit_b32": ("vop3", 3, lambda a, b, c: (((a << 32) | b) >> (c & 31)) & M32),
    "v_med3_i32": ("vop3", 3, lambda a, b, c: sorted((s32(a), s32(b), s32(c)))[1] & M32),
    "v_mul_lo_u32": ("vop3", 2, lambda a, b: (a * b) & M32),
    "v_mul_hi_u32": ("vop3", 2, lambda a, b: (a * b) >> 32),
    "v_bcnt_u32_b32": ("vop3", 2, lambda a, b: (bin(a).count("1") + b) & M32),
}
# 16-bit ops in their e64 / VOP3 form: name -> (spelling suffix, nsrc, fn on 16-bit sources). The result is the low
# 16 bits; op_sel[nsrc] picks the destination half, the other half is kept.
OPS16 = {
    "v_add_nc_u16": ("", 2, lambda a, b: a + b),
    "v_sub_nc_u16": ("", 2, lambda a, b: a - b),
    "v_mul_lo_u16": ("_e64", 2, lambda a, b: a * b),
    "v_min_u16": ("_e64", 2, min),
    "v_max_i16": ("_e64", 2, lambda a, b: max(sx(a, 16), sx(b, 16))),
    "v_lshlrev_b16": ("_e64", 2, lambda a, b: b << (a & 15)),
    "v_ashrrev_i16": ("_e64", 2, lambda a, b: sx(b, 16) >> (a & 15)),
    "v_mad_u16": ("", 3, lambda a, b, c: a * b + c),
    "v_med3_i16": ("", 3, lambda a, b, c: sorted((sx(a, 16), sx(b, 16), sx(c, 16)))[1]),
}
CMPS = {
    "eq_u32": lambda a, b: a == b, "ne_u32": lambda a, b: a != b, "lt_u32": lambda a, b: a < b,
    "ge_u32": lambda a, b: a >= b, "gt_i32": lambda a, b: s32(a) > s32(b), "le_i32": lambda a, b: s32(a) <= s32(b),
    "t_u32": lambda a, b: True, "f_i32": lambda a, b: False,
}
# carry ops: name -> fn(a, b, carry_in) -> (32-bit result, carry out)
CARRY = {
    "v_add_co_u32": lambda a, b, c: ((a + b) & M32, (a + b) >> 32),
    "v_sub_co_u32": lambda a, b, c: ((a - b) & M32, int(b > a)),
    "v_subrev_co_u32": lambda a, b, c: ((b - a) & M32, int(a > b)),
    "v_add_co_ci_u32": lambda a, b, c: ((a + b + c) & M32, (a + b + c) >> 32),
    "v_sub_co_ci_u32": lambda a, b, c: ((a - b - c) & M32, int(b + c > a)),
    "v_subrev_co_ci_u32": lambda a, b, c: ((b - a - c) & M32, int(a + c > b)),
}
MAD64 = ("v_mad_u64_u32", "v_mad_i64_i32")
# --ops names beyond the opcode tables: v_cmp_<cmp>, v_cmpx_<cmp>, v_cndmask_b32, exec (s_mov_b32 exec_lo/hi),
# smask (s_mov_b32 of a constant into an SGPR mask, read later as a lane mask)
ALL_OPS = (sorted(OPS32) + sorted(OPS16) + [f"v_cmp_{c}" for c in CMPS] + [f"v_cmpx_{c}" for c in CMPS]
           + sorted(CARRY) + list(MAD64) + ["v_cndmask_b32", "exec", "smask"])

KNOWN = False   # also generate the forms in KNOWN_DIVERGENCES (--known; always on for a hardware-only run)
KNOWN_DIVERGENCES = {
    "vcc-data-mask": "VCC written with plain data (s_mov_b32 vcc_lo/vcc_hi, K) and read as a lane mask: the "
                     "translator selects every lane for any non-zero VCC (wave32 fixed by #2206, wave64 open)",
    "mask-as-data": "a lane mask SGPR or VCC read as a 32-bit VALU source: the translator reads its per-lane view "
                    "of the mask (#293)",
}


def masks(wave):
    """The mask operands, as written in instructions."""
    return ("s[20:21]", "s[22:23]", "vcc") if wave == 64 else ("s20", "s21", "vcc_lo")


def vcc_name(wave):
    return "vcc" if wave == 64 else "vcc_lo"


# ---- program generation -------------------------------------------------------------------------------------

def rand_value(rng):
    r = rng.random()
    if r < 0.4:
        return rng.choice(INTERESTING)
    if r < 0.55:
        return rng.randrange(0, 64)
    return rng.getrandbits(32)


class Gen:
    def __init__(self, rng, wave, ops):
        self.r, self.wave = rng, wave
        allowed = set(ops)
        cats = {
            "cmp": [o for o in allowed if o.startswith("v_cmp_")],
            "cmpx": [o for o in allowed if o.startswith("v_cmpx_")],
            "cndmask": [o for o in allowed if o == "v_cndmask_b32"],
            "carry": sorted(o for o in allowed if o in CARRY),
            "mad64": sorted(o for o in allowed if o in MAD64),
            "op16": sorted(o for o in allowed if o in OPS16),
            "exec": [o for o in allowed if o == "exec"],
            "smask": [o for o in allowed if o == "smask"],
            "op32": sorted(o for o in allowed if o in OPS32),
        }
        weights = {"cmp": 12, "cmpx": 3, "cndmask": 12, "carry": 12, "mad64": 6, "op16": 10, "exec": 10, "smask": 4, "op32": 35}
        self.cats = [(c, sorted(v), weights[c]) for c, v in sorted(cats.items()) if v]
        self.last_mask = None   # the mask written last: a mask read picks it more often than chance

    def pick_mask(self):
        ms = masks(self.wave)
        return self.last_mask if self.last_mask and self.r.random() < 0.6 else self.r.choice(ms)

    def wrote(self, m):
        self.last_mask = m
        return m
        if not self.cats:
            raise ValueError("no ops selected")

    def vreg(self):
        return f"v{self.r.choice(DREGS + (0,))}"

    def src(self, bus):
        """A source: VGPR, SGPR, mask (as data, --known only) or inline constant, within the constant bus."""
        r = self.r
        k = r.random()
        if k < 0.55:
            return self.vreg()
        if k < 0.75:
            s = f"s{r.choice(SREGS)}"
        elif k < 0.82 and KNOWN:
            m = r.choice(masks(self.wave))   # mask-as-data: one 32-bit half of a mask
            s = {"s[20:21]": ("s20", "s21"), "s[22:23]": ("s22", "s23"), "vcc": ("vcc_lo", "vcc_hi")}.get(m, (m,))
            s = r.choice(s)
        else:
            return str(r.randint(-16, 64))
        if s in bus or len(bus) < 2:
            bus.add(s)
            return s
        return self.vreg()

    def e32_src(self):
        """src0 of an e32 instruction: anything but a mask (no constant bus limit with one scalar)."""
        r = self.r
        k = r.random()
        if k < 0.6:
            return self.vreg()
        if k < 0.8:
            return f"s{r.choice(SREGS)}"
        return str(r.randint(-16, 64))

    def exec_lines(self):
        def word():
            return self.r.choice(EXEC_EDGES + (None,) * 3) or self.r.getrandbits(32)
        out = [f"s_mov_b32 exec_lo, 0x{word():x}"]
        if self.wave == 64:
            out.append(f"s_mov_b32 exec_hi, 0x{word():x}")
        return out

    def op(self):
        r = self.r
        tot = sum(w for _, _, w in self.cats)
        x = r.uniform(0, tot)
        for cat, names, w in self.cats:
            x -= w
            if x <= 0:
                break
        name = r.choice(names)
        d = r.choice(DREGS)
        ms = masks(self.wave)
        vcc = vcc_name(self.wave)
        if KNOWN and r.random() < 0.06:
            # vcc-data-mask: plain data into VCC; a later mask read sees it
            out = [f"s_mov_b32 vcc_lo, 0x{rand_value(r):x}"]
            if self.wave == 64:
                out.append(f"s_mov_b32 vcc_hi, 0x{rand_value(r):x}")
            return out
        if cat == "exec":
            return self.exec_lines()
        if cat == "smask":
            m = r.choice(ms[:2])
            lo = int(re.match(r"s\[?(\d+)", m).group(1))
            return [f"s_mov_b32 s{lo + i}, 0x{rand_value(r):x}" for i in range(2 if self.wave == 64 else 1)]
        if cat == "cmp":
            c = name[len("v_cmp_"):]
            if r.random() < 0.3:
                self.wrote(vcc)
                return [f"v_cmp_{c}_e32 {vcc}, {self.e32_src()}, {self.vreg()}"]
            m = self.wrote(r.choice(ms))
            bus = {m}
            return [f"v_cmp_{c}_e64 {m}, {self.src(bus)}, {self.src(bus)}"]
        if cat == "cmpx":
            bus = set()
            return [f"{name}_e64 {self.src(bus)}, {self.src(bus)}"]
        if cat == "cndmask":
            if r.random() < 0.3:
                return [f"v_cndmask_b32_e32 v{d}, {self.e32_src()}, {self.vreg()}, {vcc}"]
            m = self.pick_mask()
            bus = {m}
            return [f"v_cndmask_b32_e64 v{d}, {self.src(bus)}, {self.src(bus)}, {m}"]
        if cat == "carry":
            ci = "_ci_" in name
            if ci and r.random() < 0.3:
                self.wrote(vcc)
                return [f"{name}_e32 v{d}, {vcc}, {self.e32_src()}, {self.vreg()}, {vcc}"]
            cin = self.pick_mask()
            co = self.wrote(r.choice(ms))
            bus = {co}
            if ci:
                bus.add(cin)
                if len(bus) > 2:
                    bus.discard(cin)
                    cin = co
                a, b = self.src(bus), self.src(bus)
                return [f"{name}_e64 v{d}, {co}, {a}, {b}, {cin}"]
            a, b = self.src(bus), self.src(bus)
            return [f"{name}_e64 v{d}, {co}, {a}, {b}"]
        if cat == "mad64":
            co = self.wrote(r.choice(ms))
            dd, cc = r.choice(PAIRS), r.choice(PAIRS)
            bus = {co}
            return [f"{name} v[{dd}:{dd + 1}], {co}, {self.src(bus)}, {self.src(bus)}, v[{cc}:{cc + 1}]"]
        if cat == "op16":
            suffix, n, _ = OPS16[name]
            osel = [r.random() < 0.3 for _ in range(n + 1)]
            bus = set()
            # op_sel on a constant is not generated (what a constant's high half is is not modelled here)
            srcs = [self.vreg() if osel[i] else self.src(bus) for i in range(n)]
            line = f"{name}{suffix} v{d}, {', '.join(srcs)}"
            if any(osel):
                line += f" op_sel:[{','.join(str(int(x)) for x in osel)}]"
            return [line]
        enc, n, _ = OPS32[name]
        if enc == "vop1":
            if r.random() < 0.5:
                return [f"{name}_e32 v{d}, {self.e32_src()}"]
            return [f"{name}_e64 v{d}, {self.src(set())}"]
        if enc == "vop2" and r.random() < 0.5:
            return [f"{name}_e32 v{d}, {self.e32_src()}, {self.vreg()}"]
        bus = set()
        srcs = [self.src(bus) for _ in range(n)]
        return [f"{name}{'_e64' if enc == 'vop2' else ''} v{d}, {', '.join(srcs)}"]


def prologue(rng, wave):
    lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0"]
    if wave == 64:
        lines.append("v_mbcnt_hi_u32_b32 v0, -1, v0")
    for v in DREGS:
        lines.append(f"v_mul_lo_u32 v{v}, v0, 0x{rng.getrandbits(32) | 1:x}")
        lines.append(f"v_xor_b32_e32 v{v}, 0x{rng.getrandbits(32):x}, v{v}")
        for _ in range(2):
            lines.append(f"s_mov_b32 s30, 0x{rand_value(rng):x}")
            lines.append(f"v_writelane_b32 v{v}, s30, {rng.randrange(wave)}")
    for s in SREGS:
        lines.append(f"s_mov_b32 s{s}, 0x{rand_value(rng):x}")
    for m in masks(wave):
        lines.append(f"v_cmp_lt_u32_e64 {m}, v{rng.choice(DREGS)}, v{rng.choice(DREGS)}")
    return lines


def epilogue(obs, wave):
    out = ["s_mov_b32 exec_lo, -1"] + (["s_mov_b32 exec_hi, -1"] if wave == 64 else [])
    return out + [f"v_mov_b32_e32 v{OBS[i]}, v{r}" for i, r in enumerate(obs)]


def written(body):
    out = []
    for line in body:
        op, _, rest = line.partition(" ")
        if op.startswith(("v_cmp", "s_")):
            continue
        a0 = rest.split(",")[0].strip()
        if a0.startswith("v["):
            lo = int(a0[2:].split(":")[0])
            out += [lo + 1, lo]
        else:
            out.append(int(a0[1:]))
    return out


def pick_obs(body, rng):
    last = []
    for d in reversed(written(body)):
        if d not in last:
            last.append(d)
    last = last[:4]
    while len(last) < 4:
        last.append(rng.choice(DREGS))
    return last


class Program:
    """init + body + epilogue(obs); body is what shrinking removes from."""

    def __init__(self, init, body, obs, wave):
        self.init, self.body, self.obs, self.wave = list(init), list(body), list(obs), wave

    def lines(self):
        return self.init + self.body + epilogue(self.obs, self.wave)

    def with_body(self, body):
        return Program(self.init, body, self.obs, self.wave)


def generate(rng, n_ops, ops, wave):
    gen = Gen(rng, wave, ops)
    init = prologue(rng, wave)
    body = []
    for _ in range(n_ops):
        body += gen.op()
    return Program(init, body, pick_obs(body, rng), wave)


# ---- the reference interpreter --------------------------------------------------------------------------------

_MODS = re.compile(r"\s+op_sel:\[([01,]+)\]\s*$")


class State:
    def __init__(self, wave):
        self.wave = wave
        self.v = {}            # VGPR -> list of per-lane words
        self.s = {}            # SGPR name ("s4", "vcc_lo", ...) -> 32-bit word
        self.exec = (1 << wave) - 1

    def vget(self, r):
        return self.v.setdefault(r, [0] * self.wave)

    def sget(self, name):
        if name not in self.s:
            raise RefError(f"{name} read before it is written")
        return self.s[name]

    def mask(self, m):
        """The lane mask an operand names (64-bit pair, VCC or a single SGPR)."""
        if m.startswith("s["):
            lo = int(m[2:].split(":")[0])
            return self.sget(f"s{lo}") | (self.sget(f"s{lo + 1}") << 32)
        if m == "vcc":
            return self.sget("vcc_lo") | (self.sget("vcc_hi") << 32)
        return self.sget(m)

    def set_mask(self, m, val):
        if m.startswith("s["):
            lo = int(m[2:].split(":")[0])
            self.s[f"s{lo}"], self.s[f"s{lo + 1}"] = val & M32, (val >> 32) & M32
        elif m == "vcc":
            self.s["vcc_lo"], self.s["vcc_hi"] = val & M32, (val >> 32) & M32
        else:
            self.s[m] = val & M32

    def val(self, tok, lane):
        if re.fullmatch(r"v\d+", tok):
            return self.vget(int(tok[1:]))[lane]
        if re.fullmatch(r"s\d+|vcc_lo|vcc_hi", tok):
            return self.sget(tok)
        return int(tok, 0) & M32

    def lanes(self):
        return [lane for lane in range(self.wave) if (self.exec >> lane) & 1]


def step(st, line):
    op, _, rest = line.partition(" ")
    osel = None
    m = _MODS.search(rest)
    if m:
        osel = [int(x) for x in m.group(1).split(",")]
        rest = rest[:m.start()]
    args = [a.strip() for a in rest.split(",")]
    if op == "s_mov_b32":
        v = st.val(args[1], 0)
        if args[0] == "exec_lo":
            st.exec = (st.exec & ~M32) | v if st.wave == 64 else v
        elif args[0] == "exec_hi":
            st.exec = (st.exec & M32) | (v << 32)
        else:
            st.s[args[0]] = v
        return
    if op == "v_mbcnt_lo_u32_b32":
        st.v[0] = [min(lane, 32) for lane in range(st.wave)]
        return
    if op == "v_mbcnt_hi_u32_b32":
        st.v[0] = [st.v[0][lane] + max(0, lane - 32) for lane in range(st.wave)]
        return
    if op == "v_writelane_b32":
        st.vget(int(args[0][1:]))[int(args[2])] = st.sget(args[1])
        return
    base = op[:-4] if op.endswith(("_e32", "_e64")) else op
    if base.startswith(("v_cmp_", "v_cmpx_")):
        cmpx = base.startswith("v_cmpx_")
        c = base[len("v_cmpx_" if cmpx else "v_cmp_"):]
        a, b = (args[0], args[1]) if cmpx else (args[1], args[2])
        res = 0
        for lane in st.lanes():
            if CMPS[c](st.val(a, lane), st.val(b, lane)):
                res |= 1 << lane
        if cmpx:
            st.exec = res
        else:
            st.set_mask(args[0], res)
        return
    if base == "v_cndmask_b32":
        mk = st.mask(args[3])
        d = st.vget(int(args[0][1:]))
        new = list(d)
        for lane in st.lanes():
            new[lane] = st.val(args[2] if (mk >> lane) & 1 else args[1], lane)
        st.v[int(args[0][1:])] = new
        return
    if base in CARRY:
        cin = st.mask(args[4]) if len(args) > 4 else 0
        new = list(st.vget(int(args[0][1:])))
        co = 0
        for lane in st.lanes():
            res, c = CARRY[base](st.val(args[2], lane), st.val(args[3], lane), (cin >> lane) & 1)
            new[lane] = res
            co |= c << lane
        st.v[int(args[0][1:])] = new
        st.set_mask(args[1], co)
        return
    if base in MAD64:
        dd = int(args[0][2:].split(":")[0])
        cc = int(args[4][2:].split(":")[0])
        lo, hi = list(st.vget(dd)), list(st.vget(dd + 1))
        co = 0
        for lane in st.lanes():
            a, b = st.val(args[2], lane), st.val(args[3], lane)
            c = st.vget(cc)[lane] | (st.vget(cc + 1)[lane] << 32)
            if base == "v_mad_u64_u32":
                full = a * b + c
                carry = full >> 64
            else:
                full = s32(a) * s32(b) + (c - (1 << 64) if c >> 63 else c)
                carry = int(full < 0)
            full &= M64
            lo[lane], hi[lane] = full & M32, full >> 32
            co |= carry << lane
        st.v[dd], st.v[dd + 1] = lo, hi
        st.set_mask(args[1], co)
        return
    d = int(args[0][1:])
    new = list(st.vget(d))
    if base in OPS16:
        _, n, fn = OPS16[base]
        sel = osel or [0] * (n + 1)
        for lane in st.lanes():
            ins = [(v >> 16) & M16 if sel[i] else v & M16
                   for i, v in enumerate(st.val(a, lane) for a in args[1:])]
            res = fn(*ins) & M16
            new[lane] = (new[lane] & M16) | (res << 16) if sel[n] else (new[lane] & 0xFFFF0000) | res
        st.v[d] = new
        return
    if base in OPS32:
        fn = OPS32[base][2]
        for lane in st.lanes():
            new[lane] = fn(*[st.val(a, lane) for a in args[1:]]) & M32
        st.v[d] = new
        return
    raise RefError(f"{op} is not modelled")


def reference(lines, wave):
    """Run the program; returns per lane the four words of v8..v11."""
    st = State(wave)
    for line in lines:
        step(st, line)
    return [tuple(st.vget(r)[lane] for r in OBS) for lane in range(wave)]


# ---- the translator side -----------------------------------------------------------------------------------

def translated(lines, wave, name="ps_wint"):
    """Translate and run one subgroup; returns per lane the four words, or raises Skip."""
    import rdna2spirv
    from spirv_wave_interp import run_wave
    with tempfile.TemporaryDirectory(prefix="wint-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  wint_diff case\n")
            for line in lines:
                f.write("\t" + line + "\n")
            f.write("\texp mrt0 v8, v9, v10, v11 done vm\n\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gen = rdna2spirv.translate.last_gen
                gc.collect()
        except rdna2spirv.Unsupported as e:
            raise Skip(f"translator unsupported: {e}") from None
    if bool(getattr(gen, "wave64", False)) != (wave == 64):
        raise Skip(f"translator compiled the wave{wave} program as wave{64 if wave == 32 else 32}")
    try:
        out = run_wave(spv, wave=wave)["outputs"]
    except Exception as e:  # noqa: BLE001 - an interpreter gap is a counted skip, never a pass
        raise Skip(f"wave interp: {type(e).__name__}: {str(e)[:80]}") from None
    res = []
    for lane in range(wave):
        words = (out[lane] or {}).get("%out_mrt0")
        res.append(None if words is None else tuple(w & M32 for w in words))
    return res


# ---- the hardware side --------------------------------------------------------------------------------------

def hw_kernel(prog):
    """A raw kernel for tools/hw_oracle: the program on one wave, then each lane stores v8..v11 at 16 * lane."""
    from hw_oracle import asm
    body = []
    for line in prog.lines():
        body += [line, "s_nop 4"]
    body += ["v_lshlrev_b32 v20, 4, v0", "global_store_b128 v20, v[8:11], s[2:3]", "s_endpgm"]
    return asm.Kernel(body, wave=prog.wave, threads=prog.wave, words=4 * prog.wave, raw=True)


def hw_words(words, wave):
    """The oracle's flat word list -> per lane four words (or a str reason through)."""
    if isinstance(words, str):
        return words
    return [tuple(words[4 * lane:4 * lane + 4]) for lane in range(wave)]


# ---- comparison, shrinking, the run --------------------------------------------------------------------------

def _diff(prog, want, got, label):
    for lane in range(prog.wave):
        g = got[lane]
        if g is None:
            return f"lane {lane}: {label} produced no output (killed)"
        for i in range(4):
            if want[lane][i] != g[i]:
                return (f"lane {lane} word {i} (v{prog.obs[i]}): reference 0x{want[lane][i]:08x}, "
                        f"{label} 0x{g[i]:08x}")
    return None


def check(prog, backends=("translator",), hw=None):
    """None on a match, else a description. Raises Skip when every backend skipped. hw: the hardware result
    (per-lane words or a str reason) or a callable Program -> that."""
    lines = prog.lines()
    want = reference(lines, prog.wave)
    problems, skips = [], []
    if "translator" in backends:
        try:
            problems.append(_diff(prog, want, translated(lines, prog.wave), "translated"))
        except Skip as e:
            skips.append(e)
    if "hw" in backends:
        got = hw(prog) if callable(hw) else hw
        if isinstance(got, str):
            skips.append(Skip(f"hardware: {got}"))
        else:
            problems.append(_diff(prog, want, got, "hardware"))
    if skips and len(skips) == len(backends):
        raise skips[0]
    return "; ".join(p for p in problems if p) or None


def shrink(prog, check_fn=check):
    """Observe one register if that still fails, then drop body instructions while the program still mismatches."""
    def fails(p):
        try:
            return bool(check_fn(p))
        except (Skip, RefError):
            return False
    for r in prog.obs:
        p = Program(prog.init, prog.body, [r] * 4, prog.wave)
        if fails(p):
            prog = p
            break
    changed = True
    while changed:
        changed = False
        for i in range(len(prog.body) - 1, -1, -1):
            p = prog.with_body(prog.body[:i] + prog.body[i + 1:])
            if fails(p):
                prog, changed = p, True
    # the seeds of registers nothing reads any more, then the v_writelane patches that do not matter
    used = set()
    for line in prog.body + epilogue(prog.obs, prog.wave) + [x for x in prog.init if x.startswith("v_cmp")]:
        used |= regs_in(line)
    init, i = [], 0
    while i < len(prog.init):
        line = prog.init[i]
        if line.startswith("s_mov_b32 s30,"):
            pair = prog.init[i:i + 2]
            if regs_in(pair[1]) & used:
                init += pair
            i += 2
            continue
        dst = line.partition(" ")[2].split(",")[0].strip()
        if "mbcnt" in line or line.startswith("v_cmp") or dst in used:
            init.append(line)
        i += 1
    prog = Program(init, prog.body, prog.obs, prog.wave)
    for i in range(len(prog.init) - 2, -1, -1):
        if prog.init[i].startswith("s_mov_b32 s30,"):
            p = Program(prog.init[:i] + prog.init[i + 2:], prog.body, prog.obs, prog.wave)
            if fails(p):
                prog = p
    return prog


def regs_in(line):
    """The VGPR and SGPR names a line mentions (pairs expanded)."""
    out = set()
    for kind, lo, hi in re.findall(r"\b([vs])\[(\d+):(\d+)\]", line):
        out |= {f"{kind}{r}" for r in range(int(lo), int(hi) + 1)}
    return out | set(re.findall(r"\b[vs]\d+\b", line))


def vk_judge(progs, hw, build=None, work=None):
    """The programs on a Vulkan GPU, every lane judged against the hardware's words. A program is one wave of its own
    size (the lines already spell it); returns ({class: n}, [(i, detail, program)])."""
    import vk_fuzz_run
    counts, bad = {}, []
    for wave in sorted({p.wave for p in progs}):
        idx = [i for i, p in enumerate(progs) if p.wave == wave]
        got = vk_fuzz_run.run_programs([progs[i].lines() for i in idx], 4, wave, build, work, opaque=False,
                                       rewrite=False, per_lane=True)
        for i, g in zip(idx, got):
            h = hw[i]
            if isinstance(h, str) or isinstance(g, str):
                why = "skip: " + (("hardware: " + h) if isinstance(h, str) else g)
                counts[why] = counts.get(why, 0) + 1
                continue
            first = next(((lane, k) for lane in range(wave) for k in range(4) if tuple(g[lane])[k] != h[lane][k]), None)
            if first is None:
                counts["match"] = counts.get("match", 0) + 1
                continue
            lane, k = first
            counts["mismatch"] = counts.get("mismatch", 0) + 1
            bad.append((i, f"wave{wave} lane {lane} word {k} (v{progs[i].obs[k]}): hardware 0x{h[lane][k]:08x}, "
                           f"translated on the GPU 0x{g[lane][k]:08x}", progs[i]))
    return counts, bad


def run_suite(programs, seed, max_ops, ops, waves=(32,), backends=("translator",), oracle=None, build=None, work=None):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "vk": {}, "vk_failures": []}
    failures = []
    rng = random.Random(seed)
    progs = [generate(rng, rng.randrange(1, max_ops + 1), ops, waves[i % len(waves)]) for i in range(programs)]
    if "hw" in backends:
        raw = oracle.run([hw_kernel(p) for p in progs])
        hw = [hw_words(w, p.wave) for w, p in zip(raw, progs)]
    else:
        hw = [None] * programs
    if "vk" in backends:
        counts["vk"], counts["vk_failures"] = vk_judge(progs, hw, build, work)
    backends = tuple(b for b in backends if b != "vk")
    for i, prog in enumerate(progs):
        try:
            problem = check(prog, backends, hw[i])
        except Skip as e:
            counts["skip"][str(e)] = counts["skip"].get(str(e), 0) + 1
            continue
        if problem is None:
            counts["pass"] += 1
            continue
        counts["mismatch"] += 1
        failures.append((i, problem, prog))
    return counts, failures


def self_test():
    """Harness canaries: a hand program matches, the reference gives hand-worked values, a bad form is a skip."""
    init = ["v_mbcnt_lo_u32_b32 v0, -1, 0"] + [f"s_mov_b32 s{s}, {s}" for s in SREGS]
    init += [f"v_mov_b32_e32 v{v}, v0" for v in DREGS] + ["v_cmp_lt_u32_e64 s20, v0, 4",
                                                          "v_cmp_lt_u32_e64 s21, v0, 2", "v_cmp_ne_u32_e64 vcc_lo, 0, 0"]
    body = ["s_mov_b32 exec_lo, 0x0000000f",                      # lanes 0..3
            "v_add_co_u32_e64 v1, s20, v0, -2",                    # lanes 2, 3 carry; v1 = lane - 2
            "v_add_co_ci_u32_e64 v2, s21, v0, 0, s20",             # v2 = lane + carry
            "v_cndmask_b32_e64 v3, 7, 9, s20"]
    prog = Program(init, body, [1, 2, 3, 4], 32)
    ok = check(prog) is None
    want = reference(prog.lines(), 32)
    worked = (want[0] == (0xFFFFFFFE, 0, 7, 0) and want[3] == (1, 4, 9, 3) and want[5] == (5, 5, 5, 5))
    try:
        translated(init + ["v_add_nc_u32_sdwa_dpp v1, v2, v3"] + epilogue([1, 2, 3, 4], 32), 32)
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, hand values {worked}, unsupported form skipped {skipped}")
    return ok and worked and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="Whole-wave integer VALU differential fuzzing under random EXEC.")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=8)
    ap.add_argument("--wave", default="32", choices=("32", "64", "both"))
    ap.add_argument("--ops", default=None, help="comma-separated ops (--list-ops); default all")
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
    ap.add_argument("--build", default=os.path.join(os.path.dirname(HERE), "runtime", "build"),
                    help="with vk: the folder with vk_compute_test")
    ap.add_argument("--work", default=None, help="with vk: a folder for the temporary files (VK_FUZZ_KEEP=1 keeps them)")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    backends = tuple(dict.fromkeys(t.strip() for t in a.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw", "vk") for t in backends):
        ap.error("--oracle takes translator, hw, vk (with hw) or a comma-separated mix")
    if "vk" in backends and "hw" not in backends:
        ap.error("--oracle vk judges the translated programs against the hardware's words: use --oracle hw,vk")
    global KNOWN
    KNOWN = a.known or backends == ("hw",)
    if a.list_ops:
        for name in ALL_OPS:
            kind = ("vop3 32-bit" if name in OPS32 and OPS32[name][0] == "vop3" else
                    f"{OPS32[name][0]} 32-bit (e32/e64)" if name in OPS32 else
                    "16-bit e64/VOP3, op_sel" if name in OPS16 else
                    "carry" if name in CARRY else "mad64, mask carry" if name in MAD64 else
                    "compare -> EXEC" if name.startswith("v_cmpx_") else
                    "compare -> mask" if name.startswith("v_cmp_") else
                    "mask select" if name == "v_cndmask_b32" else
                    "s_mov_b32 of data into an SGPR mask" if name == "smask" else "s_mov_b32 exec_lo/hi")
            print(f"{name:20} {kind}")
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.ops is not None:
        tokens = [t.strip() for t in a.ops.split(",")]
        if not any(tokens) or any(not t for t in tokens):
            ap.error("--ops must be a comma-separated list of ops")
        ops = sorted(set(tokens))
    else:
        ops = list(ALL_OPS)
    unknown = [o for o in ops if o not in ALL_OPS]
    if unknown:
        ap.error("unknown ops: " + ", ".join(unknown))
    waves = (32, 64) if a.wave == "both" else (int(a.wave),)
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
        counts, failures = run_suite(a.programs, a.seed, a.max_ops, ops, waves, backends, oracle, a.build, a.work)
        vk_bad = counts.pop("vk_failures")
        skips = sum(counts["skip"].values())
        print(f"{a.programs} programs (seed {a.seed}, wave {a.wave}, {'+'.join(backends)}"
              f"{', known' if KNOWN else ''}): {counts['pass']} pass, {counts['mismatch']} mismatch, "
              f"{skips} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
            print(f"  skip x{n}: {reason}")

        want_vk = "vk" in backends
        if want_vk:
            print("translated on the Vulkan GPU against the hardware, every lane: "
                  + ", ".join(f"{k} {n}" for k, n in sorted(counts["vk"].items(), key=lambda kv: kv[0].startswith("skip"))))
            for i, detail, prog in vk_bad[: None if a.all_failures else 3]:
                print(f"VK MISMATCH program {i}: {detail}")
                print("\n".join("\t" + line for line in prog.lines()))
        backends = tuple(b for b in backends if b != "vk")

        def check_fn(prog):
            return check(prog, backends,
                         (lambda p: hw_words(oracle.run([hw_kernel(p)])[0], p.wave)) if oracle else None)
        for i, problem, prog in failures[: None if a.all_failures else 1]:
            small = shrink(prog, check_fn)
            print(f"MISMATCH program {i} (wave{prog.wave}): {check_fn(small) or problem}")
            print("\n".join("\t" + line for line in small.lines()))
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
