r"""Differential fuzzer for the translator's wave and cross-lane operations (FUZZ-7 in #156 / #374).

Generates seeded random compute programs built from cross-lane RDNA2 instructions (v_readlane, v_writelane,
v_readfirstlane, ballots through v_cmp / v_cmpx, s_bcnt1 / s_ff1 of masks, v_mbcnt, v_cndmask on a mask, masks
read as data, v_permlane16 / v_permlanex16, forward s_cbranch_execz / vccz / vccnz around them, and with --dpp
DPP16 / DPP8 sources) under random EXEC masks: constants per half, v_cmpx, nested s_and_saveexec and its
restore, partial waves (a group size that isn't a multiple of the wave size) and groups of two or more waves.
Each program runs through a Python model of the ISA (the reference, below) and through the translator on the
local GPU: llvm-mc assembles it, tools/rdna2spirv.py translates it (wave32 natively; wave64 with
--wave64-emulate, and natively with --w64-native), runtime/build/vk_compute_test runs it with VKC_OUT_WORDS /
VKC_DUMP, and every word every thread stores is compared. docs/shader_validation.md has the details.

    python -I tools/crosslane_fuzz.py --programs 200 --seed 1 [--wave 32|64|both] [--modes native,emulate]
    python -I tools/crosslane_fuzz.py --ext --wave 64    # more wave64 ops (below); wave32 programs are unchanged
    python -I tools/crosslane_fuzz.py --self-test        # the model against hand-computed values, no GPU
    python -I tools/crosslane_fuzz.py --oracle hw        # model vs the AMD GPU natively (wave32 and wave64); exit 77
                                                         # when tools/hw_oracle's tools are missing. --oracle translator,hw
                                                         # does both. No llvm-mc, Vulkan or vk_compute_test for hw alone.
    python -I tools/crosslane_fuzz.py --dpp-sweep --oracle hw   # one program per DPP case instead of random ones:
                                                         # every DPP16 control x bound_ctrl x fi x EXEC pattern, all
                                                         # row_mask / bank_mask pairs, DPP8 selects (7,206 per wave size)

Exit code: 0 when every program that ran matches (or differs only in a program with a hypothesis op), 1 on a
mismatch or an untranslated program, 2 when the only failures are harness errors: vk_compute_test timed out or
gave no dump for a program. Such a program is listed with the others (and kept with --keep) and the run goes on.
A missing vk_compute_test for the translator oracle is one line and exit 1, before any program is generated.
A wave32 program is skipped where vk_compute_test can't pin 32-lane subgroups ("compute subgroup size: 32 not
available", e.g. Mesa llvmpipe at 8 lanes), and a run where every program was skipped exits 2. Wave64 programs
translated with --wave64-emulate use no subgroup operations, so they are compared at any subgroup size.

Needs LLVM_MC (llvm-mc with AMDGPU, as for the shaders stage), the Vulkan SDK and a built vk_compute_test
(cmake --build runtime/build --target vk_compute_test). No game files.

Reference semantics (AMD "RDNA 2" ISA reference guide; Mesa ACO and LLVM where the guide is terse):
  * v_readlane_b32 sD, vS, lane: sD = vS of lane (lane & 63 in wave64, & 31 in wave32), EXEC ignored.
  * v_writelane_b32 vD, s, lane: vD of that lane = s, EXEC ignored.
  * v_readfirstlane_b32: vS of the lowest lane on in EXEC, lane 0 if EXEC is 0.
  * VOPC v_cmp: bit l = condition for active lanes, 0 for inactive ones (sD / VCC is written as a whole).
    v_cmpx: EXEC = EXEC & condition (inactive lanes stay off).
  * s_and_saveexec_bN sD, s: sD = EXEC; EXEC = s & EXEC.
  * v_mbcnt_lo_u32_b32 vD, s, acc: popcount(s & ((1 << lane) - 1)) + acc (all 32 bits for lane >= 32);
    v_mbcnt_hi_u32_b32: popcount(s & ((1 << (lane - 32)) - 1)) + acc for lane >= 32, acc for lane < 32.
  * v_permlane16_b32 vD, vS, s1, s2: lane l reads lane (l & ~15) | sel[l & 15], sel = nibbles of {s2, s1};
    v_permlanex16_b32 reads lane (l & ~31) | ((l & 16) ^ 16) | sel[l & 15]. Only written in active lanes.
    A source lane that is off in EXEC (op_sel FI = 0) is the DPP "invalid source" case: with BOUND_CTRL = 0 the
    lane keeps its old vD. That was measured on RDNA2 and RDNA3.5 (docs/shader_validation.md, #374), but the
    translator models it only with #580, so programs that reach it are still reported apart.
  * VALU results are written only in active lanes; SALU is wave-uniform.

--ext adds these to wave64 programs (the model of each was checked against an RDNA2 GPU, gfx1036, with --oracle hw):
  * v_add_co / v_sub_co / v_subrev_co_u32 and the _ci forms, carries in SGPR pairs or VCC: the carry-out mask has
    the lane's carry (borrow) for an active lane and 0 for an inactive one.
  * ds_bpermute_b32: lane l reads vData of lane ((vAddr + offset) >> 2) & 31 of its own 32-lane half; a source
    lane that is off in EXEC reads 0. ds_permute_b32: every active lane sends vData to that lane; a lane nobody
    sends to gets 0, and with several senders the highest lane wins. ds_swizzle_b32 (QUAD_PERM, BITMASK_PERM, SWAP,
    REVERSE, BROADCAST) works inside each group of 32 lanes; an EXEC-off source lane reads 0, which the
    translator models only with #2780, so programs that reach it are reported apart like the permlane case.
    On hardware the three need `s_waitcnt lgkmcnt(0)` before the result is used; the generator emits it.
  * v_readlane / v_writelane with an SGPR lane select of 64 or more (S1[5:0]).
  * s_bcnt0/1 and s_ff0/1 of a mask, s_and/or/xor/andn2_b64 of masks, the ten s_*_saveexec_b64 forms and
    s_andn1/andn2_wrexec_b64, s_wqm_b64, s_cmp_eq/lg_u64 and s_bitcmp0/1_b64 of masks, each followed by an
    s_cselect_b32 so its SCC reaches a stored SGPR; s_cselect_b64 between masks; s_mov_b64 exec, mask;
    s_bfm_b64 (D = ((1 << S0[5:0]) - 1) << S1[5:0]) into a mask or EXEC.
  * A loop in LLVM's SI_LOOP shape: per-lane trip counts, a break mask, s_andn2_b64 exec and s_cbranch_execnz
    back to the head, so lanes of both halves leave on different iterations while the rest keep running.
Usage: `python -I tools/crosslane_fuzz.py --help` lists the options.
"""
import argparse
import os
import random
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
DATA_V = (2, 3, 4, 5, 6, 7)            # data VGPRs, stored at the end
DATA_S = (8, 9, 10, 11, 12, 13, 14, 15)  # data SGPRs, stored at the end
OUT_REGS = [("v", r) for r in DATA_V] + [("s", r) for r in DATA_S]
SAVE = (22, 24, 26)                    # saveexec destinations (pairs in wave64)
ENTRY = 30                             # entry EXEC (pair in wave64)
SEL = 28                               # scratch SGPRs for lane selects / permlane selectors (28, 29)
PERM_V = 23                            # --ext: scratch VGPR for ds_permute addresses
LOOP_V = (24, 25)                      # --ext: a loop's per-lane trip limit and counter
LOOP_S = (32, 34)                      # --ext: a loop's break mask and "done this iteration" mask (pairs)
DS_WAIT = "\n\ts_waitcnt lgkmcnt(0)"   # after ds_bpermute / ds_permute / ds_swizzle: the result comes back late
PERMLANE_FETCH_INACTIVE = False
TRANSLATOR = [os.path.join(HERE, "rdna2spirv.py")]   # --translator        # --permlane-fetch-inactive: model FI = 1 (read inactive sources)


class Wave:
    def __init__(self, size, exists):
        self.size = size
        self.exists = exists           # bit mask of lanes that exist (partial last wave)
        self.v = {}                    # vgpr -> [lane values]
        self.s = {}                    # sgpr -> value
        self.exec = exists
        self.vcc = 0

    def vr(self, r):
        return self.v.setdefault(r, [0] * self.size)

    def sr(self, r):
        return self.s.get(r, 0)

    def active(self, lane):
        return self.exec >> lane & 1

    def mask(self, name):
        """A mask operand: 'vcc', 'exec' or an SGPR number (pair in wave64)."""
        if name == "vcc":
            return self.vcc
        if name == "exec":
            return self.exec
        lo = self.sr(name)
        return lo | (self.sr(name + 1) << 32 if self.size == 64 else 0)

    def set_mask(self, name, m):
        m &= (1 << self.size) - 1
        if name == "vcc":
            self.vcc = m
        elif name == "exec":
            self.exec = m
        else:
            self.s[name] = m & M32
            if self.size == 64:
                self.s[name + 1] = m >> 32

    def vwrite(self, r, vals):
        dst = self.vr(r)
        for lane in range(self.size):
            if self.active(lane) and self.exists >> lane & 1:
                dst[lane] = vals[lane] & M32


def popcount(x):
    return bin(x).count("1")


def mreg(wave, name):
    """Assembler name of a mask operand."""
    if name == "vcc":
        return "vcc" if wave == 64 else "vcc_lo"
    if name == "exec":
        return "exec" if wave == 64 else "exec_lo"
    return f"s[{name}:{name + 1}]" if wave == 64 else f"s{name}"


class Op:
    """One instruction (or a short fixed group) with its assembly and its effect on the model."""
    hyp = False                         # depends on a hypothesis about the ISA (reported apart)

    def __init__(self, text, fn, tag, hyp=False):
        self.text, self.fn, self.tag, self.hyp = text, fn, tag, hyp

    def apply(self, waves):
        hyp = False
        for w in waves:
            hyp |= bool(self.fn(w))
        return hyp


DPP16_CONTROLS = ("quad_perm", "row_shl", "row_shr", "row_ror", "row_mirror", "row_half_mirror", "row_share",
                  "row_xmask")
DPP_OPS = {"mov": "v_mov_b32", "not": "v_not_b32", "add": "v_add_nc_u32", "xor": "v_xor_b32"}


def dpp16_select(control, arg=None):
    """(modifier text, f) for one DPP16 control; f(x) gives (source lane in the 16-lane row, whether it is in the
    row) for row lane x. arg: the four quad_perm selects, or the shift / rotate / share / xmask amount."""
    if control == "quad_perm":
        q = list(arg)
        return f"quad_perm:[{','.join(map(str, q))}]", (lambda x: ((x & ~3) | q[x & 3], True))
    if control == "row_shl":
        return f"row_shl:{arg}", (lambda x: (x + arg, x + arg < 16))
    if control == "row_shr":
        return f"row_shr:{arg}", (lambda x: (x - arg, x >= arg))
    if control == "row_ror":
        return f"row_ror:{arg}", (lambda x: ((x - arg) & 15, True))
    if control == "row_mirror":
        return "row_mirror", (lambda x: (15 - x, True))
    if control == "row_half_mirror":
        return "row_half_mirror", (lambda x: ((x & 8) | (7 - (x & 7)), True))
    if control == "row_share":
        return f"row_share:{arg}", (lambda x: (arg, True))
    if control == "row_xmask":
        return f"row_xmask:{arg}", (lambda x: (x ^ arg, True))
    raise ValueError(f"unknown DPP16 control {control!r}")


def dpp_op(d, a, b, kind, control=None, dpp8=None, fi=False, bound=False, row_mask=0xF, bank_mask=0xF):
    """A VOP1/VOP2 op with a DPP16 (control = (name, arg)) or DPP8 (dpp8 = the eight selects) src0, and its
    effect on the model. From the RDNA 2 ISA guide's DPP section: DPP16 picks the source lane inside the 16-lane
    row; a source out of the row, or one that is off in EXEC while fi = 0, is invalid: bound_ctrl 1 uses 0 as
    src0, bound_ctrl 0 leaves the lane unwritten. Lanes in a row or bank that row_mask / bank_mask turn off are
    not written. DPP8 picks a lane in the 8-lane group; an inactive source with fi = 0 reads 0 (as #238
    implements it; reported apart as a hypothesis). Every source is read before vD is written, so d may be a."""
    if dpp8 is not None:
        sel = list(dpp8)
        ctl = f"dpp8:[{','.join(map(str, sel))}]" + (" fi:1" if fi else "")
        pick = lambda l: ((l & ~7) | sel[l & 7], True)
    else:
        ctl, f = dpp16_select(*control)

        def pick(l, f=f):
            s_, ok = f(l & 15)
            return (l & ~15) | (s_ & 15), ok
        ctl += f" row_mask:0x{row_mask:x} bank_mask:0x{bank_mask:x}" + (" bound_ctrl:1" if bound else "") + \
            (" fi:1" if fi else "")
    name = DPP_OPS[kind]
    if kind in ("mov", "not"):
        text = f"{name}_dpp v{d}, v{a} {ctl}"
    else:
        text = f"{name}_dpp v{d}, v{a}, v{b} {ctl}"
    alu = {"mov": lambda x, y: x, "not": lambda x, y: ~x, "add": lambda x, y: x + y,
           "xor": lambda x, y: x ^ y}[kind]

    def fn(w):
        src, other = w.vr(a), w.vr(b)
        vals, hyp = list(w.vr(d)), False
        for lane in range(w.size):
            if not w.active(lane):
                continue
            sl, ok = pick(lane)
            if dpp8 is not None:
                if w.active(sl) or fi:
                    x = src[sl]
                else:
                    x, hyp = 0, True
            else:
                if not (row_mask >> (lane >> 4) & 1 and bank_mask >> ((lane >> 2) & 3) & 1):
                    continue
                if ok and (w.active(sl) or fi):
                    x = src[sl]
                elif bound:
                    x = 0
                else:
                    continue
            vals[lane] = alu(x, other[lane]) & M32
        w.vwrite(d, vals)
        return hyp
    return Op(text, fn, "dpp")


def exec_op(wave, mask):
    """EXEC = mask (a constant), as op_exec_const writes it."""
    if wave == 32:
        text = f"s_mov_b32 exec_lo, 0x{mask & M32:x}"
    else:
        text = f"s_mov_b32 exec_lo, 0x{mask & M32:x}\n\ts_mov_b32 exec_hi, 0x{mask >> 32:x}"

    def fn(w):
        w.exec = mask & ((1 << w.size) - 1)
    return Op(text, fn, "exec")


# --dpp-sweep: EXEC patterns. "all" has no inactive source; "odd" turns off every other lane, so every source a
# lane reads one or three lanes away is off; "banks" turns off the first lane of each 4-lane bank and the whole
# second row, so inactive sources sit next to active ones and one row (row 1) runs nothing.
SWEEP_EXEC = {
    32: (("all", M32), ("odd", 0x55555555), ("banks", 0x0000EEEE)),
    64: (("all", M64), ("odd", 0x5555555555555555), ("banks", 0xEEEEEEEE0000EEEE)),
}
SWEEP_DPP8 = ((0, 1, 2, 3, 4, 5, 6, 7), (7, 6, 5, 4, 3, 2, 1, 0), (1, 0, 3, 2, 5, 4, 7, 6), (4, 5, 6, 7, 0, 1, 2, 3)) + \
    tuple((k,) * 8 for k in range(8)) + tuple(tuple((i + k) & 7 for i in range(8)) for k in range(1, 8))
SWEEP_MASK_CONTROLS = (("row_shr", 1), ("quad_perm", (1, 0, 3, 2)), ("row_share", 5))


def dpp16_sweep_controls():
    """Every DPP16 control the RDNA2 assembler takes: 256 quad_perms, row_shl/shr/ror 1-15, both mirrors,
    row_share and row_xmask 0-15 (335)."""
    out = [("quad_perm", tuple(q >> s & 3 for s in (0, 2, 4, 6))) for q in range(256)]
    out += [(c, n) for c in ("row_shl", "row_shr", "row_ror") for n in range(1, 16)]
    out += [("row_mirror", None), ("row_half_mirror", None)]
    out += [(c, n) for c in ("row_share", "row_xmask") for n in range(16)]
    return out


def dpp_sweep_cases(wave):
    """The --dpp-sweep cases for one wave size, as (family, exec name, exec mask, op kwargs). Three parts:
    every DPP16 control x bound_ctrl x fi x EXEC pattern with both masks on; all 256 row_mask / bank_mask pairs
    for three controls, with and without bound_ctrl, under the full and the odd EXEC; every DPP8 select set
    x fi x EXEC pattern. The op (mov, not, add, xor) and whether vD is also src0 (in place, as a reduction
    writes it) rotate through the cases, so each control meets all four ops and both register layouts."""
    cases = []
    execs = SWEEP_EXEC[wave]
    for control in dpp16_sweep_controls():
        for bound in (False, True):
            for fi in (False, True):
                for ename, emask in execs:
                    cases.append((control[0], ename, emask,
                                  dict(control=control, fi=fi, bound=bound)))
    for control in SWEEP_MASK_CONTROLS:
        for row_mask in range(16):
            for bank_mask in range(16):
                for bound in (False, True):
                    for ename, emask in execs[:2]:
                        cases.append(("masks", ename, emask, dict(control=control, bound=bound, row_mask=row_mask,
                                                                   bank_mask=bank_mask)))
    for sel in SWEEP_DPP8:
        for fi in (False, True):
            for ename, emask in execs:
                cases.append(("dpp8", ename, emask, dict(dpp8=sel, fi=fi)))
    return cases


def dpp_sweep_programs(wave):
    """One program per --dpp-sweep case: the usual prologue (data in v2-v7), EXEC set to the case's pattern, one
    DPP op, and the usual epilogue (EXEC restored, every word stored). One full wave, so no source lane is missing."""
    progs = []
    kinds = tuple(DPP_OPS)
    for i, (family, ename, emask, kw) in enumerate(dpp_sweep_cases(wave)):
        rng = random.Random(wave * 1000003 + i)
        pro, init = prologue(rng, wave)
        kind = kinds[i % 4]
        d, a, b = (2, 2, 4) if (i // 4) % 2 else (2, 3, 4)
        op = dpp_op(d, a, b, kind, **kw)
        p = Program(f"cs_dpp{wave}_{i}", wave, wave, pro, init, [exec_op(wave, emask), op])
        p.label = f"{family}, EXEC {ename}: {op.text}"
        p.family = family
        progs.append(p)
    return progs


class Gen:
    def __init__(self, rng, wave, group, allow_maskdata=True, wave32_hi_lanes=True, dpp=False, ext=False):
        self.rng, self.W, self.L = rng, wave, group
        self.nwaves = (group + wave - 1) // wave
        last = group - (self.nwaves - 1) * wave
        self.min_exist = last                       # lanes every wave has
        self.full = last == wave
        self.allow_maskdata, self.hi_lanes = allow_maskdata, wave32_hi_lanes
        self.dpp = dpp
        self.ext = ext and wave == 64               # --ext: the wave64 extension ops
        self.masks = []                             # mask registers written so far
        self.depth = 0
        self.in_branch, self.branch_id = False, 0

    def r(self, seq):
        return self.rng.choice(seq)

    def lane(self):
        return self.rng.randrange(self.min_exist)

    def vsrc(self):
        return self.r(DATA_V + (1,))

    def mask_name(self, write=False):
        pool = ["vcc", 16, 18, 20]
        if write:
            return self.r(pool)
        return self.r(self.masks) if self.masks else None

    # ---- op builders ------------------------------------------------------------------------------------
    def op_cmp(self):
        d = self.mask_name(write=True)
        a, cond = self.vsrc(), self.r(("lt", "gt", "eq", "ne", "le", "ge"))
        f = {"lt": lambda x, y: x < y, "gt": lambda x, y: x > y, "eq": lambda x, y: x == y,
             "ne": lambda x, y: x != y, "le": lambda x, y: x <= y, "ge": lambda x, y: x >= y}[cond]
        if d == "vcc" and self.rng.random() < 0.5:
            k = self.rng.randrange(65)
            text = f"v_cmp_{cond}_u32_e32 {mreg(self.W, 'vcc')}, {k}, v{a}"

            def fn(w):
                m = 0
                for lane in range(w.size):
                    if w.active(lane) and f(k, w.vr(a)[lane]):
                        m |= 1 << lane
                w.set_mask("vcc", m)
        else:
            b = self.vsrc()
            text = f"v_cmp_{cond}_u32_e64 {mreg(self.W, d)}, v{a}, v{b}"

            def fn(w):
                m = 0
                for lane in range(w.size):
                    if w.active(lane) and f(w.vr(a)[lane], w.vr(b)[lane]):
                        m |= 1 << lane
                w.set_mask(d, m)
        if d not in self.masks:
            self.masks.append(d)
        return Op(text, fn, "cmp")

    def op_cmpx(self):
        a, k = self.vsrc(), self.rng.randrange(65)
        cond = self.r(("gt", "lt", "ne"))
        f = {"gt": lambda x, y: x > y, "lt": lambda x, y: x < y, "ne": lambda x, y: x != y}[cond]
        text = f"v_cmpx_{cond}_u32_e32 {k}, v{a}"

        def fn(w):
            m = 0
            for lane in range(w.size):
                if w.active(lane) and f(k, w.vr(a)[lane]):
                    m |= 1 << lane
            w.exec = m
        return Op(text, fn, "exec")

    def op_exec_const(self):
        ex = (1 << self.min_exist) - 1
        lo = self.rng.getrandbits(32) & ex
        if self.rng.random() < 0.15:
            lo = 0
        if self.W == 32:
            text = f"s_mov_b32 exec_lo, 0x{lo:x}"

            def fn(w):
                w.exec = lo
            return Op(text, fn, "exec")
        hi = self.rng.getrandbits(32) & (ex >> 32)
        if self.rng.random() < 0.15:
            hi = 0
        text = f"s_mov_b32 exec_lo, 0x{lo:x}\n\ts_mov_b32 exec_hi, 0x{hi:x}"

        def fn(w):
            w.exec = lo | hi << 32
        return Op(text, fn, "exec")

    def op_saveexec(self):
        m = self.mask_name()
        if m is None or self.depth >= len(SAVE):
            return None
        d = SAVE[self.depth]
        self.depth += 1
        op = "s_and_saveexec_b64" if self.W == 64 else "s_and_saveexec_b32"
        text = f"{op} {mreg(self.W, d)}, {mreg(self.W, m)}"

        def fn(w):
            old = w.exec
            w.exec = w.mask(m) & old
            w.set_mask(d, old)
        return Op(text, fn, "exec")

    def op_restore(self):
        if self.depth == 0:
            return None
        self.depth -= 1
        d = SAVE[self.depth]
        op = "s_mov_b64 exec," if self.W == 64 else "s_mov_b32 exec_lo,"
        text = f"{op} {mreg(self.W, d)}"

        def fn(w):
            w.exec = w.mask(d) & w.exists
        return Op(text, fn, "exec")

    def op_readlane(self):
        d, v = self.r(DATA_S), self.vsrc()
        lane = self.lane()
        enc = lane
        if self.W == 32 and self.hi_lanes and self.rng.random() < 0.3:
            enc = lane + 32 * self.rng.randrange(1, 8)     # wave32 uses lane select bits [4:0]
        if self.rng.random() < 0.5:
            text = f"v_readlane_b32 s{d}, v{v}, {enc}" if enc <= 64 else \
                f"s_mov_b32 s{SEL}, {enc}\n\tv_readlane_b32 s{d}, v{v}, s{SEL}"
        else:
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_readlane_b32 s{d}, v{v}, s{SEL}"
        tag = "readlane" if enc == lane else "readlane_w32hi"

        def fn(w):
            w.s[d] = w.vr(v)[lane]
            if "s_mov_b32" in text:
                w.s[SEL] = enc
        return Op(text, fn, tag)

    def op_writelane(self):
        d = self.r(DATA_V)
        lane = self.lane() if self.rng.random() < 0.7 else self.rng.randrange(self.W)
        enc = lane
        if self.W == 32 and self.hi_lanes and self.rng.random() < 0.3:
            enc = lane + 32 * self.rng.randrange(1, 8)
        if self.rng.random() < 0.5:
            src = f"s{self.r(DATA_S)}"
        else:
            src = str(self.rng.randrange(-16, 65))
        if self.rng.random() < 0.5 and enc <= 64:
            text = f"v_writelane_b32 v{d}, {src}, {enc}"
        else:
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_writelane_b32 v{d}, {src}, s{SEL}"
        tag = "writelane" if enc == lane else "writelane_w32hi"

        def fn(w):
            val = w.sr(int(src[1:])) if src.startswith("s") else int(src) & M32
            if lane < w.size and w.exists >> lane & 1:
                w.vr(d)[lane] = val
            if "s_mov_b32" in text:
                w.s[SEL] = enc
        return Op(text, fn, tag)

    def op_readfirstlane(self):
        d, v = self.r(DATA_S), self.vsrc()
        text = f"v_readfirstlane_b32 s{d}, v{v}"

        def fn(w):
            e = w.exec
            first = (e & -e).bit_length() - 1 if e else 0
            w.s[d] = w.vr(v)[first]
        return Op(text, fn, "readfirstlane")

    def op_bcnt(self):
        m = self.mask_name() if self.rng.random() < 0.7 else "exec"
        if m is None:
            m = "exec"
        d = self.r(DATA_S)
        kind = self.r(("bcnt1", "ff1", "bcnt0"))
        sfx = "b64" if self.W == 64 else "b32"
        text = f"s_{kind}_i32_{sfx} s{d}, {mreg(self.W, m)}"

        def fn(w):
            x = w.mask(m)
            if kind == "bcnt1":
                w.s[d] = popcount(x)
            elif kind == "bcnt0":
                w.s[d] = self.W - popcount(x)
            else:
                w.s[d] = ((x & -x).bit_length() - 1) & M32 if x else M32
        return Op(text, fn, "bcnt")

    def op_mbcnt(self):
        d, acc = self.r(DATA_V), self.vsrc()
        m = self.mask_name() if self.rng.random() < 0.6 else "exec"
        if m is None:
            m = "exec"
        if self.W == 32:
            text = f"v_mbcnt_lo_u32_b32 v{d}, {mreg(32, m)}, v{acc}"
        elif m in ("exec", "vcc"):
            text = (f"v_mbcnt_lo_u32_b32 v{d}, {m}_lo, v{acc}\n\t"
                    f"v_mbcnt_hi_u32_b32 v{d}, {m}_hi, v{d}")
        else:
            text = f"v_mbcnt_lo_u32_b32 v{d}, s{m}, v{acc}\n\tv_mbcnt_hi_u32_b32 v{d}, s{m + 1}, v{d}"

        def fn(w):
            x = w.mask(m)
            vals = []
            for lane in range(w.size):
                vals.append(popcount(x & ((1 << lane) - 1)) + w.vr(acc)[lane])
            w.vwrite(d, vals)
        return Op(text, fn, "mbcnt")

    def op_mbcnt_data(self):
        """v_mbcnt with an ordinary SGPR (data, not a mask) as its mask operand."""
        d, acc, s = self.r(DATA_V), self.vsrc(), self.r(DATA_S)
        hi = self.W == 64 and self.rng.random() < 0.5
        name = "v_mbcnt_hi_u32_b32" if hi else "v_mbcnt_lo_u32_b32"
        text = f"{name} v{d}, s{s}, v{acc}"

        def fn(w):
            x = w.sr(s)
            vals = []
            for lane in range(w.size):
                if hi:
                    below = 0 if lane < 32 else popcount(x & ((1 << (lane - 32)) - 1))
                else:
                    below = popcount(x & ((1 << min(lane, 32)) - 1))
                vals.append(below + w.vr(acc)[lane])
            w.vwrite(d, vals)
        return Op(text, fn, "mbcnt_data")

    def op_cndmask(self):
        m = self.mask_name()
        if m is None:
            return None
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        text = f"v_cndmask_b32_e64 v{d}, v{a}, v{b}, {mreg(self.W, m)}"

        def fn(w):
            x = w.mask(m)
            w.vwrite(d, [w.vr(b)[l] if x >> l & 1 else w.vr(a)[l] for l in range(w.size)])
        return Op(text, fn, "cndmask")

    def op_maskdata(self):
        m = self.mask_name()
        if m is None or not self.allow_maskdata:
            return None
        d = self.r(DATA_V)
        half = self.rng.randrange(2) if self.W == 64 else 0
        if m == "vcc":
            src = "vcc_hi" if half else "vcc_lo"
        else:
            src = f"s{m + half}"
        text = f"v_mov_b32_e32 v{d}, {src}"

        def fn(w):
            x = w.mask(m) >> (32 * half) & M32
            w.vwrite(d, [x] * w.size)
        return Op(text, fn, "maskdata")

    def op_mask_salu(self):
        if len(self.masks) < 1:
            return None
        a, b = self.r(self.masks), self.r(self.masks)
        d = self.r([16, 18, 20])
        kind = self.r(("and", "or", "xor", "andn2"))
        sfx = "b64" if self.W == 64 else "b32"
        text = f"s_{kind}_{sfx} {mreg(self.W, d)}, {mreg(self.W, a)}, {mreg(self.W, b)}"
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            x, y = w.mask(a), w.mask(b)
            r = {"and": x & y, "or": x | y, "xor": x ^ y, "andn2": x & ~y}[kind]
            w.set_mask(d, r)
        return Op(text, fn, "mask_salu")

    def op_permlane(self):
        if not self.full:
            return None
        d, v = self.r(DATA_V), self.vsrc()
        x16 = self.rng.random() < 0.5
        s1, s2 = self.rng.getrandbits(32), self.rng.getrandbits(32)
        name = "v_permlanex16_b32" if x16 else "v_permlane16_b32"
        if self.rng.random() < 0.5:     # VOP3 takes one literal: s1 as a literal, s2 from an SGPR
            text = f"s_mov_b32 s{SEL + 1}, 0x{s2:x}\n\t{name} v{d}, v{v}, 0x{s1:x}, s{SEL + 1}"
        else:
            text = (f"s_mov_b32 s{SEL}, 0x{s1:x}\n\ts_mov_b32 s{SEL + 1}, 0x{s2:x}\n\t"
                    f"{name} v{d}, v{v}, s{SEL}, s{SEL + 1}")
        sel = s1 | s2 << 32

        def fn(w):
            if text.count("s_mov_b32") == 2:
                w.s[SEL] = s1
            w.s[SEL + 1] = s2
            src = w.vr(v)
            old = list(w.vr(d))
            vals, hyp = list(old), False
            for lane in range(w.size):
                n = sel >> (4 * (lane & 15)) & 15
                sl = (lane & ~31) | ((lane & 16) ^ 16) | n if x16 else (lane & ~15) | n
                if not w.active(lane):
                    continue
                if w.active(sl) or PERMLANE_FETCH_INACTIVE:
                    vals[lane] = src[sl]
                if not w.active(sl):
                    hyp = not PERMLANE_FETCH_INACTIVE   # FI = 0, BOUND_CTRL = 0: keeps vD (measured; translator: #580)
            w.vwrite(d, vals)
            return hyp
        return Op(text, fn, "permlane")

    def op_dpp(self):
        """A VOP1/VOP2 op with a DPP16 or DPP8 src0 (--dpp). Model, from the RDNA 2 ISA guide's DPP section:
        DPP16 picks the source lane inside the 16-lane row (quad_perm, row_shl/shr/ror, row_mirror,
        row_half_mirror, row_share, row_xmask); a source out of the row, or one that is off in EXEC while fi = 0,
        is invalid: bound_ctrl 1 uses 0 as src0, bound_ctrl 0 leaves the lane unwritten. Lanes in a row or bank
        that row_mask / bank_mask turn off are not written. DPP8 picks a lane in the 8-lane group; an inactive
        source with fi = 0 reads 0 (as #238 implements it; reported apart as a hypothesis)."""
        if not self.dpp or not self.full:           # a source lane in the missing part of a wave is undefined
            return None
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        kind = self.r(("mov", "mov", "not", "add", "xor"))
        r = self.rng
        # The draws stay in this order, so a seed gives the same programs as before dpp_op was split out.
        dpp8 = r.random() < 0.25
        fi = r.random() < 0.3
        bound = r.random() < 0.5
        row_mask = 0xF if r.random() < 0.6 else r.randrange(16)
        bank_mask = 0xF if r.random() < 0.6 else r.randrange(16)
        if dpp8:
            return dpp_op(d, a, b, kind, fi=fi, dpp8=[r.randrange(8) for _ in range(8)])
        c = r.choice(DPP16_CONTROLS)
        if c == "quad_perm":
            arg = [r.randrange(4) for _ in range(4)]
        elif c in ("row_shl", "row_shr", "row_ror"):
            arg = r.randrange(1, 16)
        elif c in ("row_share", "row_xmask"):
            arg = r.randrange(16)
        else:
            arg = None
        return dpp_op(d, a, b, kind, control=(c, arg), fi=fi, bound=bound, row_mask=row_mask, bank_mask=bank_mask)

    def op_valu(self):
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        kind = self.r(("add", "xor", "movs", "lshr"))
        if kind == "add":
            text, f = f"v_add_nc_u32_e32 v{d}, v{a}, v{b}", lambda w, l: w.vr(a)[l] + w.vr(b)[l]
        elif kind == "xor":
            text, f = f"v_xor_b32_e32 v{d}, v{a}, v{b}", lambda w, l: w.vr(a)[l] ^ w.vr(b)[l]
        elif kind == "lshr":
            k = self.rng.randrange(1, 17)
            text, f = f"v_lshrrev_b32_e32 v{d}, {k}, v{a}", lambda w, l: w.vr(a)[l] >> k
        else:
            s = self.r(DATA_S)
            text, f = f"v_mov_b32_e32 v{d}, s{s}", lambda w, l: w.sr(s)
        return Op(text, lambda w: w.vwrite(d, [f(w, l) for l in range(w.size)]), "valu")

    def op_salu(self):
        d, a, b = self.r(DATA_S), self.r(DATA_S), self.r(DATA_S)
        if self.rng.random() < 0.5:
            text = f"s_add_u32 s{d}, s{a}, s{b}"

            def fn(w):
                w.s[d] = (w.sr(a) + w.sr(b)) & M32
        else:
            text = f"s_xor_b32 s{d}, s{a}, s{b}"

            def fn(w):
                w.s[d] = w.sr(a) ^ w.sr(b)
        return Op(text, fn, "salu")

    def op_branch(self):
        """A forward branch over a few ops: s_cbranch_execz / vccz / vccnz (taken per wave, as on hardware)."""
        if self.in_branch:
            return None
        kind = self.r(("execz", "execz", "vccz", "vccnz"))
        if kind != "execz" and "vcc" not in self.masks:
            kind = "execz"
        self.in_branch = True
        body = self.program(self.rng.randrange(1, 5))
        self.in_branch = False
        k = self.branch_id
        self.branch_id += 1
        text = "\n\t".join([f"s_cbranch_{kind} @{k}"] + [o.text for o in body] + [f";@END{k}"])

        def fn(w):
            taken = {"execz": w.exec == 0, "vccz": w.vcc == 0, "vccnz": w.vcc != 0}[kind]
            hyp = False
            if not taken:
                for o in body:
                    hyp |= bool(o.fn(w))
            return hyp
        op = Op(text, fn, "branch")
        op.tags = {"branch"} | {t for o in body for t in getattr(o, "tags", {o.tag})}
        return op

    # ---- --ext: more wave64 ops (all return None in a wave32 program) -----------------------------------
    def scc_probe(self):
        """(text, fn) of an s_cselect_b32 right after an op, so the op's SCC reaches a stored SGPR."""
        d, a, b = self.r(DATA_S), self.r(DATA_S), self.r(DATA_S)

        def fn(w, scc):
            w.s[d] = w.sr(a) if scc else w.sr(b)
        return f"\n\ts_cselect_b32 s{d}, s{a}, s{b}", fn

    def ext_mask(self):
        """A mask that has been written, or EXEC."""
        return self.r(self.masks + ["exec"])

    def op_scan64(self):
        """s_bcnt1 / s_bcnt0 / s_ff1 / s_ff0 of a 64-bit mask; the counts also show their SCC (result != 0)."""
        if self.W != 64:
            return None
        m, d = self.ext_mask(), self.r(DATA_S)
        kind = self.r(("bcnt1", "bcnt0", "ff1", "ff0"))
        ptext, pfn = self.scc_probe()
        probe = kind.startswith("bcnt")
        text = f"s_{kind}_i32_b64 s{d}, {mreg(64, m)}" + (ptext if probe else "")

        def fn(w):
            x = w.mask(m)
            if kind == "bcnt1":
                r = popcount(x)
            elif kind == "bcnt0":
                r = 64 - popcount(x)
            else:
                y = x if kind == "ff1" else ~x & M64
                r = ((y & -y).bit_length() - 1) & M32 if y else M32
            w.s[d] = r
            if probe:
                pfn(w, r != 0)
        return Op(text, fn, "scan64")

    def op_mask_scc(self):
        """s_and/or/xor/andn2_b64 of two masks into a mask or VCC, and its SCC (result != 0)."""
        if self.W != 64 or not self.masks:
            return None
        a, b = self.r(self.masks), self.r(self.masks)
        d = self.r([16, 18, 20, "vcc"])
        kind = self.r(("and", "or", "xor", "andn2"))
        ptext, pfn = self.scc_probe()
        text = f"s_{kind}_b64 {mreg(64, d)}, {mreg(64, a)}, {mreg(64, b)}" + ptext
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            x, y = w.mask(a), w.mask(b)
            r = {"and": x & y, "or": x | y, "xor": x ^ y, "andn2": x & ~y}[kind] & M64
            w.set_mask(d, r)
            pfn(w, r != 0)
        return Op(text, fn, "mask_scc")

    def op_exec_logic(self):
        """The saveexec / wrexec family (RDNA2 ISA SOP1): saveexec D = EXEC, EXEC = f(S0, EXEC); wrexec
        EXEC = f(S0, EXEC), D = EXEC; SCC = (EXEC != 0). D is one of the mask registers, not the saveexec stack of
        op_saveexec. The forms that complement a whole mask stay out of a partial wave: they would turn on lanes
        that do not exist."""
        if self.W != 64 or not self.masks:
            return None
        m, d = self.r(self.masks), self.r([16, 18, 20])
        kinds = ["and", "or", "xor", "andn1", "andn2"]
        if self.full:
            kinds += ["orn1", "orn2", "nand", "nor", "xnor"]
        kind = self.r(kinds)
        wr = kind in ("andn1", "andn2") and self.rng.random() < 0.4
        ptext, pfn = self.scc_probe()
        text = f"s_{kind}_{'wrexec' if wr else 'saveexec'}_b64 {mreg(64, d)}, {mreg(64, m)}" + ptext
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            s, e = w.mask(m), w.exec
            r = {"and": s & e, "or": s | e, "xor": s ^ e, "andn1": ~s & e, "andn2": s & ~e, "orn1": ~s | e,
                 "orn2": s | ~e, "nand": ~(s & e), "nor": ~(s | e), "xnor": ~(s ^ e)}[kind] & M64
            w.exec = r
            w.set_mask(d, r if wr else e)
            pfn(w, r != 0)
        return Op(text, fn, "exec")

    def op_exec_mask(self):
        """EXEC = a mask (s_mov_b64), not only the restore of a saved EXEC."""
        if self.W != 64 or not self.masks:
            return None
        m = self.r(self.masks)

        def fn(w):
            w.exec = w.mask(m)
        return Op(f"s_mov_b64 exec, {mreg(64, m)}", fn, "exec")

    def op_cselect64(self):
        """s_cselect_b64 between two masks on the SCC of a scalar compare."""
        if self.W != 64 or not self.masks:
            return None
        a, b = self.ext_mask(), self.r(self.masks)
        d = self.r([16, 18, 20, "vcc"])
        sa, sb = self.r(DATA_S), self.r(DATA_S)
        text = f"s_cmp_lt_u32 s{sa}, s{sb}\n\ts_cselect_b64 {mreg(64, d)}, {mreg(64, a)}, {mreg(64, b)}"
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            w.set_mask(d, w.mask(a) if w.sr(sa) < w.sr(sb) else w.mask(b))
        return Op(text, fn, "cselect64")

    def op_mask_test(self):
        """s_cmp_eq_u64 / s_cmp_lg_u64 of two masks, or s_bitcmp0/1_b64 of one lane's bit, seen through SCC."""
        if self.W != 64 or not self.masks:
            return None
        ptext, pfn = self.scc_probe()
        if self.rng.random() < 0.5:
            a, b = self.ext_mask(), self.ext_mask()
            kind = self.r(("eq", "lg"))
            text = f"s_cmp_{kind}_u64 {mreg(64, a)}, {mreg(64, b)}" + ptext

            def fn(w):
                pfn(w, (w.mask(a) == w.mask(b)) == (kind == "eq"))
        else:
            m, one, lane = self.ext_mask(), self.rng.randrange(2), self.rng.randrange(64)
            text = f"s_bitcmp{one}_b64 {mreg(64, m)}, {lane}" + ptext

            def fn(w):
                pfn(w, (w.mask(m) >> lane & 1) == one)
        return Op(text, fn, "mask_test")

    def op_bfm(self):
        """s_bfm_b64 D, S0, S1: D = ((1 << S0[5:0]) - 1) << S1[5:0], into a mask, VCC or EXEC (full waves only)."""
        if self.W != 64 or not self.full:
            return None
        d = self.r([16, 18, 20, "vcc", "exec"])
        form = self.r(("cc", "sc", "cs", "ss"))
        sa, sb = self.r(DATA_S), self.r(DATA_S)
        ka, kb = self.rng.randrange(65), self.rng.randrange(65)
        text = (f"s_bfm_b64 {mreg(64, d)}, " + (f"s{sa}" if form[0] == "s" else str(ka)) + ", " +
                (f"s{sb}" if form[1] == "s" else str(kb)))
        if d not in self.masks and d != "exec":
            self.masks.append(d)

        def fn(w):
            size = (w.sr(sa) if form[0] == "s" else ka) & 63
            off = (w.sr(sb) if form[1] == "s" else kb) & 63
            w.set_mask(d, (((1 << size) - 1) << off) & M64)
        return Op(text, fn, "exec" if d == "exec" else "bfm")

    def op_wqm(self):
        """s_wqm_b64: every group of four bits with a bit set becomes 1111; SCC = (D != 0). Full waves only."""
        if self.W != 64 or not self.full or not self.masks:
            return None
        m, d = self.ext_mask(), self.r([16, 18, 20, "vcc"])
        ptext, pfn = self.scc_probe()
        text = f"s_wqm_b64 {mreg(64, d)}, {mreg(64, m)}" + ptext
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            x, r = w.mask(m), 0
            for q in range(16):
                if x >> (4 * q) & 15:
                    r |= 15 << (4 * q)
            w.set_mask(d, r)
            pfn(w, r != 0)
        return Op(text, fn, "wqm")

    def op_carry(self):
        """v_add_co / v_sub_co / v_subrev_co (and the _ci forms) with the carry in and out in SGPR pairs or VCC.
        Carry out: bit l for an active lane, 0 for an inactive one; the carry in is read before it is written."""
        if self.W != 64:
            return None
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        kind = self.r(("add_co", "sub_co", "subrev_co", "add_co_ci", "sub_co_ci", "subrev_co_ci"))
        ci = kind.endswith("_ci")
        if ci and not self.masks:
            return None
        mo = self.mask_name(write=True)
        mi = self.r(self.masks) if ci else None
        sfx = "e32" if mo == "vcc" and mi == "vcc" and self.rng.random() < 0.5 else "e64"   # VOP2 only has VCC
        text = f"v_{kind}_u32_{sfx} v{d}, {mreg(64, mo)}, v{a}, v{b}" + (f", {mreg(64, mi)}" if ci else "")
        if mo not in self.masks:
            self.masks.append(mo)

        def fn(w):
            cin = w.mask(mi) if ci else 0
            vals, m = [], 0
            for l in range(w.size):
                x, y, c = w.vr(a)[l], w.vr(b)[l], cin >> l & 1
                if kind.startswith("add"):
                    t = x + y + c
                    co = t >> 32
                else:
                    t = (y - x - c) if kind.startswith("subrev") else (x - y - c)
                    co = int(t < 0)
                vals.append(t & M32)
                if w.active(l) and co:
                    m |= 1 << l
            w.vwrite(d, vals)
            w.set_mask(mo, m)
        return Op(text, fn, "carry")

    def op_bpermute(self):
        """ds_bpermute_b32 vD, vAddr, vData: lane l reads vData of lane ((vAddr + offset) >> 2) & 31 of its own
        32-lane half; a source lane that is off in EXEC (or missing from a partial wave) reads 0."""
        if self.W != 64:
            return None
        d, ad, dat = self.r(DATA_V), self.vsrc(), self.vsrc()
        off = self.r((0, 0, 4, 60, 128, 252, 65532))
        text = f"ds_bpermute_b32 v{d}, v{ad}, v{dat}" + (f" offset:{off}" if off else "") + DS_WAIT

        def fn(w):
            vals = []
            for l in range(w.size):
                sl = (l & 32) | (((w.vr(ad)[l] + off) >> 2) & 31)
                vals.append(w.vr(dat)[sl] if w.active(sl) else 0)
            w.vwrite(d, vals)
        return Op(text, fn, "bpermute")

    def op_permute(self):
        """ds_permute_b32 vD, vAddr, vData: every lane on in EXEC sends vData to lane ((vAddr + offset) >> 2) & 31
        of its own half; a lane nobody sends to gets 0; several senders: the highest lane wins (measured). Half of
        the time the addresses are a permutation (lane * odd + c), else any data VGPR."""
        if self.W != 64:
            return None
        d, dat = self.r(DATA_V), self.vsrc()
        if self.rng.random() < 0.5:
            k, c = self.r((1, 3, 5, 7, 9, 11, 13, 15, 17, 31)), self.rng.randrange(64)
            ad, off = PERM_V, self.r((0, 0, 4, 124))
            pre = f"v_mad_u32_u24 v{ad}, v1, {k}, {c}\n\tv_lshlrev_b32_e32 v{ad}, 2, v{ad}\n\t"
        else:
            k = c = None
            ad, off, pre = self.vsrc(), 0, ""
        text = pre + f"ds_permute_b32 v{d}, v{ad}, v{dat}" + (f" offset:{off}" if off else "") + DS_WAIT

        def fn(w):
            if k is not None:
                w.vwrite(ad, [((l * k + c) << 2) & M32 for l in range(w.size)])
            vals = [0] * w.size
            for l in range(w.size):
                if w.active(l):
                    vals[(l & 32) | (((w.vr(ad)[l] + off) >> 2) & 31)] = w.vr(dat)[l]
            w.vwrite(d, vals)
        return Op(text, fn, "permute")

    def op_swizzle(self):
        """ds_swizzle_b32 vD, vS offset:swizzle(...): QUAD_PERM inside each group of four lanes; the other modes
        are source lane = ((lane & and) | or) ^ xor inside each group of 32. A source lane that is off in EXEC
        reads 0 (measured; the translator does that with #2780, so a program that reaches it is reported apart)."""
        if self.W != 64:
            return None
        d, v = self.r(DATA_V), self.vsrc()
        mode = self.r(("quad", "bits", "bits", "swap", "reverse", "broadcast"))
        and_m, or_m, xor_m = 31, 0, 0
        if mode == "quad":
            q = [self.rng.randrange(4) for _ in range(4)]
            sw = f"swizzle(QUAD_PERM, {', '.join(map(str, q))})"
        elif mode == "bits":
            chars = [self.r("01pi") for _ in range(5)]
            and_m = 0
            for i, ch in enumerate(chars):          # bit 4 first: p = keep, i = invert, 0 / 1 = force
                bit = 1 << (4 - i)
                and_m |= bit if ch in "pi" else 0
                xor_m |= bit if ch == "i" else 0
                or_m |= bit if ch == "1" else 0
            sw = f'swizzle(BITMASK_PERM, "{"".join(chars)}")'
        elif mode == "swap":
            xor_m = self.r((1, 2, 4, 8, 16))
            sw = f"swizzle(SWAP,{xor_m})"
        elif mode == "reverse":
            n = self.r((2, 4, 8, 16, 32))
            xor_m = n - 1
            sw = f"swizzle(REVERSE,{n})"
        else:
            n = self.r((2, 4, 8, 16, 32))
            or_m = self.rng.randrange(n)
            and_m = ~(n - 1) & 31
            sw = f"swizzle(BROADCAST,{n},{or_m})"
        text = f"ds_swizzle_b32 v{d}, v{v} offset:{sw}" + DS_WAIT

        def fn(w):
            vals, hyp = [], False
            for l in range(w.size):
                sl = (l & ~3) | q[l & 3] if mode == "quad" else (l & ~31) | (((l & 31 & and_m) | or_m) ^ xor_m)
                if w.active(sl):
                    vals.append(w.vr(v)[sl])
                else:
                    vals.append(0)
                    hyp |= bool(w.active(l))
            w.vwrite(d, vals)
            return hyp
        return Op(text, fn, "swizzle")

    def op_lane_wrap(self):
        """v_readlane / v_writelane with an SGPR lane select of 64 or more: wave64 uses S1[5:0]."""
        if self.W != 64:
            return None
        lane = self.lane()
        enc = lane + 64 * self.rng.randrange(1, 1 << 20)
        if self.rng.random() < 0.5:
            d, v = self.r(DATA_S), self.vsrc()
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_readlane_b32 s{d}, v{v}, s{SEL}"

            def fn(w):
                w.s[SEL] = enc
                w.s[d] = w.vr(v)[lane]
        else:
            d, s = self.r(DATA_V), self.r(DATA_S)
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_writelane_b32 v{d}, s{s}, s{SEL}"

            def fn(w):
                w.s[SEL] = enc
                if w.exists >> lane & 1:
                    w.vr(d)[lane] = w.sr(s)
        return Op(text, fn, "lane_wrap")

    LOOP_BODY = ("op_readfirstlane", "op_bcnt", "op_mbcnt", "op_valu", "op_valu", "op_readlane", "op_writelane",
                 "op_cmp", "op_cndmask", "op_bpermute", "op_permute", "op_carry", "op_salu", "op_mask_salu",
                 "op_scan64", "op_cselect64", "op_mask_test")

    def op_loop(self):
        """A loop in the shape LLVM's SI_LOOP gives: every lane has its own trip count (1 to 8), a lane that is
        done joins the break mask, s_andn2_b64 takes it out of EXEC and s_cbranch_execnz closes the loop while a
        lane is left; EXEC comes back from the break mask after it. The body is one to four ops that do not write
        EXEC, then the count of the lanes still running added to a data VGPR, so every iteration shows."""
        if self.W != 64 or self.in_branch:
            return None
        src, trips = self.vsrc(), self.r((1, 3, 3, 7))
        self.in_branch = True                       # no branch and no second loop inside
        body, n = [], self.rng.randrange(1, 5)
        while len(body) < n:
            o = getattr(self, self.r(self.LOOP_BODY))()
            if o is not None:
                body.append(o)
        sd, vx = self.r(DATA_S), self.r(DATA_V)
        self.in_branch = False
        k = self.branch_id
        self.branch_id += 1
        lim, it = LOOP_V
        brk, done = LOOP_S
        lines = [f"v_and_b32_e32 v{lim}, {trips}, v{src}", f"v_mov_b32_e32 v{it}, 0",
                 f"s_mov_b64 {mreg(64, brk)}, 0", f";@LOOP{k}"]
        lines += [o.text for o in body]
        lines += [f"s_bcnt1_i32_b64 s{sd}, exec", f"v_add_nc_u32_e32 v{vx}, s{sd}, v{vx}",
                  f"v_cmp_ge_u32_e64 {mreg(64, done)}, v{it}, v{lim}", f"v_add_nc_u32_e32 v{it}, 1, v{it}",
                  f"s_or_b64 {mreg(64, brk)}, {mreg(64, done)}, {mreg(64, brk)}",
                  f"s_andn2_b64 exec, exec, {mreg(64, brk)}", f"s_cbranch_execnz @-{k}",
                  f"s_or_b64 exec, exec, {mreg(64, brk)}"]

        def fn(w):
            hyp = False
            w.vwrite(lim, [x & trips for x in w.vr(src)])
            w.vwrite(it, [0] * w.size)
            broke = 0
            for _ in range(trips + 2):
                for o in body:
                    hyp |= bool(o.fn(w))
                w.s[sd] = popcount(w.exec)
                w.vwrite(vx, [w.sr(sd) + x for x in w.vr(vx)])
                m = 0
                for l in range(w.size):
                    if w.active(l) and w.vr(it)[l] >= w.vr(lim)[l]:
                        m |= 1 << l
                w.vwrite(it, [x + 1 for x in w.vr(it)])
                broke |= m
                w.exec &= ~broke
                if not w.exec:
                    break
            else:
                raise AssertionError("loop model did not end")
            w.set_mask(brk, broke)
            w.set_mask(done, m)
            w.exec |= broke
            return hyp
        op = Op("\n\t".join(lines), fn, "loop")
        op.tags = {"loop"} | {t for o in body for t in getattr(o, "tags", {o.tag})}
        return op

    EXT_WEIGHTS = (("op_scan64", 3), ("op_mask_scc", 3), ("op_exec_logic", 3), ("op_exec_mask", 1),
                   ("op_cselect64", 2), ("op_mask_test", 3), ("op_bfm", 2), ("op_wqm", 2), ("op_carry", 4),
                   ("op_bpermute", 3), ("op_permute", 4), ("op_swizzle", 3), ("op_lane_wrap", 2), ("op_loop", 3))

    WEIGHTS = (("op_branch", 3), ("op_cmp", 5), ("op_cmpx", 2), ("op_exec_const", 2), ("op_saveexec", 2), ("op_restore", 2),
               ("op_readlane", 4), ("op_writelane", 3), ("op_readfirstlane", 3), ("op_bcnt", 3), ("op_mbcnt", 3),
               ("op_mbcnt_data", 1), ("op_cndmask", 2), ("op_maskdata", 1), ("op_mask_salu", 2),
               ("op_permlane", 3), ("op_valu", 3), ("op_salu", 1), ("op_dpp", 6))

    def program(self, n):
        table = [kw for kw in self.WEIGHTS if kw[0] != "op_dpp" or self.dpp]   # seeds stay stable without --dpp
        if self.ext:                                # and without --ext
            table += self.EXT_WEIGHTS
        names = [k for k, _ in table]
        weights = [w for _, w in table]
        ops = []
        while len(ops) < n:
            op = getattr(self, self.rng.choices(names, weights)[0])()
            if op is not None:
                ops.append(op)
        return ops


def prologue(rng, wave):
    """Lane id in v1, data in v2-v7 and s8-s15, entry EXEC saved. Returns (text lines, init fn)."""
    lines = ["v_mbcnt_lo_u32_b32 v1, -1, 0"]
    if wave == 64:
        lines.append("v_mbcnt_hi_u32_b32 v1, -1, v1")
        lines.append(f"s_mov_b64 s[{ENTRY}:{ENTRY + 1}], exec")
    else:
        lines.append(f"s_mov_b32 s{ENTRY}, exec_lo")
    coef = []
    for r in DATA_V:
        a, b = rng.randrange(1, 65), rng.getrandbits(32)
        coef.append((r, a, b))
        lines.append(f"v_mad_u32_u24 v{r}, v0, {a}, 0x{b:x}")
    svals = [(r, rng.getrandbits(32)) for r in DATA_S]
    lines += [f"s_mov_b32 s{r}, 0x{v:x}" for r, v in svals]
    # Every mask register holds a mask on every path, as in compiled code: a branch may skip the v_cmp that
    # writes one, and a register that is undefined on one path has no defined value after the join.
    mov, ex = ("s_mov_b64", "exec") if wave == 64 else ("s_mov_b32", "exec_lo")
    lines += [f"{mov} {mreg(wave, m)}, 0" for m in ("vcc", 16, 18, 20)]
    lines += [f"{mov} {mreg(wave, m)}, {ex}" for m in SAVE]

    def init(waves):
        for wi, w in enumerate(waves):
            tid = [wi * w.size + l for l in range(w.size)]
            w.v[0] = tid
            w.v[1] = list(range(w.size))
            for r, a, b in coef:
                w.v[r] = [((t & 0xFFFFFF) * a + b) & M32 for t in tid]
            for r, v in svals:
                w.s[r] = v
            w.set_mask(ENTRY, w.exists)
            for m in ("vcc", 16, 18, 20):
                w.set_mask(m, 0)
            for m in SAVE:
                w.set_mask(m, w.exists)
    return lines, init


def epilogue(wave, hw=False):
    """hw=True stores with global_store_b32 through the output pointer in s[2:3] (hw_oracle) instead of a buffer."""
    k = len(OUT_REGS)
    lines = [f"s_mov_b64 exec, s[{ENTRY}:{ENTRY + 1}]" if wave == 64 else f"s_mov_b32 exec_lo, s{ENTRY}",
             f"v_mul_u32_u24_e32 v22, {4 * k}, v0"]
    for j, (kind, r) in enumerate(OUT_REGS):
        lines.append(f"v_mov_b32_e32 v20, {kind}{r}")
        lines.append(f"global_store_b32 v22, v20, s[2:3] offset:{4 * j}" if hw else
                     f"buffer_store_dword v20, v22, s[0:3], 0 offen offset:{4 * j}")
    lines.append("s_endpgm")
    return lines


class Program:
    label = None      # --dpp-sweep: what the case tests, printed with a failure
    family = None     # --dpp-sweep: the control family, for the per-family summary

    def __init__(self, name, wave, group, pro, init, ops):
        self.name, self.wave, self.group, self.pro, self.init, self.ops = name, wave, group, pro, init, ops

    def source(self, ops=None):
        ops = self.ops if ops is None else ops
        mattr = "+wavefrontsize64,-wavefrontsize32" if self.wave == 64 else "+wavefrontsize32,-wavefrontsize64"
        head = [f"; {self.name}: crosslane_fuzz, wave{self.wave}, group {self.group}",
                f"; group: {self.group},1,1", f"; mattr: {mattr}"]
        body = self.pro + [o.text for o in ops] + epilogue(self.wave)
        return "\n".join(head + body) + "\n"

    def hw_lines(self, ops=None, guard=True):
        """The program as a native kernel body for tools/hw_oracle: branch placeholders become labels, the stores go
        through the output pointer, and with guard each instruction is followed by s_nop 7. The ISA model has no
        hazards; the real unit needs wait states after some VALU writes of SGPRs, VCC and EXEC (GFX10 and GFX11
        both), and compiled code gets them from the compiler."""
        ops = self.ops if ops is None else ops
        out = []
        for text in self.pro + [o.text for o in ops] + epilogue(self.wave, hw=True):
            for ln in (x.strip() for x in text.split("\n")):
                m = re.fullmatch(r";@(END|LOOP)(\d+)", ln)
                if m:
                    out.append(f".L{self.name}_{'B' if m.group(1) == 'LOOP' else ''}{m.group(2)}:")
                    continue
                ln = re.sub(r"^(s_cbranch_\w+) @(-?)(\d+)$",
                            lambda b: f"{b.group(1)} .L{self.name}_{'B' if b.group(2) else ''}{b.group(3)}", ln)
                out.append(ln)
                if guard and ln != "s_endpgm":
                    out.append("s_nop 7")
        return out

    def expected(self, ops=None):
        """{(thread, word): value}, plus the set of ops whose result depends on a hypothesis."""
        ops = self.ops if ops is None else ops
        n = (self.group + self.wave - 1) // self.wave
        waves = []
        for wi in range(n):
            cnt = min(self.wave, self.group - wi * self.wave)
            waves.append(Wave(self.wave, (1 << cnt) - 1))
        self.init(waves)
        hyp = set()
        for i, o in enumerate(ops):
            if o.apply(waves):
                hyp.add(i)
        out = {}
        for wi, w in enumerate(waves):
            for lane in range(w.size):
                if not w.exists >> lane & 1:
                    continue
                tid = wi * w.size + lane
                for j, (kind, r) in enumerate(OUT_REGS):
                    out[(tid, j)] = w.vr(r)[lane] if kind == "v" else w.sr(r)
        return out, hyp


def make_program(seed, wave, n_ops, **kw):
    rng = random.Random(seed)
    if wave == 64:
        group = rng.choice((64, 64, 128, 40, 100, 192))
    else:
        group = rng.choice((32, 32, 64, 20, 50, 96))
    g = Gen(rng, wave, group, **kw)
    pro, init = prologue(rng, wave)
    ops = g.program(n_ops)
    return Program(f"cs_xl{wave}_{seed}", wave, group, pro, init, ops)


# ---- running on the GPU ---------------------------------------------------------------------------------
def assemble(mc, prog, src, disasm, fssm):
    import compute_fixture
    mattr = "+wavefrontsize64,-wavefrontsize32" if prog.wave == 64 else "+wavefrontsize32,-wavefrontsize64"
    cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", f"-mattr={mattr}", "-show-encoding"]
    if "@" in src:
        src = resolve_branches(cmd, prog, src)
    r = mc_command.run(cmd, input=src, capture_output=True, text=True)
    if r.returncode or "error:" in r.stderr:
        raise RuntimeError(f"{prog.name} did not assemble:\n{r.stderr}\n{src}")
    code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
    with open(os.path.join(disasm, prog.name + ".s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {prog.name}  crosslane_fuzz\n{code}")
    asm_path = os.path.join(fssm, prog.name + ".asm")
    with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    with open(os.path.join(fssm, prog.name + ".fssm"), "wb") as f:
        f.write(compute_fixture.compute_header(asm_path))
    os.remove(asm_path)


def branch_offsets(lines, size_at):
    """lines with every `s_cbranch_* @k` replaced by the dword offset to its `;@ENDk` line below it, and every
    `@-k` (--ext loops) by the negative offset back to its `;@LOOPk` line above it. size_at: line index -> the
    instruction's size in dwords. A branch offset counts from the instruction after the branch."""
    out = list(lines)
    for i, ln in enumerate(lines):
        m = re.search(r"@(-?)(\d+)$", ln.strip())
        if not m:
            continue
        if m.group(1):
            head = next(j for j in range(i - 1, -1, -1) if lines[j].strip() == f";@LOOP{m.group(2)}")
            off = -sum(size_at.get(j, 0) for j in range(head + 1, i + 1))
        else:
            end = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == f";@END{m.group(2)}")
            off = sum(size_at.get(j, 0) for j in range(i + 1, end))
        out[i] = ln[:ln.rindex("@")] + str(off)
    return out


def resolve_branches(cmd, prog, src):
    """Replace each `s_cbranch_* @k` with the dword offset to its `;@ENDk` line (and `@-k` with the offset back
    to `;@LOOPk`): assemble once with offset 0, take each instruction's size from its encoding, and sum the
    sizes in between (branch_offsets)."""
    lines = src.splitlines()
    insns = [i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith(";")]
    first = "\n".join(re.sub(r"@-?\d+$", "0", ln) for ln in lines) + "\n"
    r = mc_command.run(cmd, input=first, capture_output=True, text=True)
    if r.returncode or "error:" in r.stderr:
        raise RuntimeError(f"{prog.name} did not assemble:\n{r.stderr}\n{first}")
    sizes = [len(re.findall(r"0x[0-9a-f]{2}", m)) // 4 for m in re.findall(r"encoding: \[([^\]]*)\]", r.stdout)]
    if len(sizes) != len(insns):
        raise RuntimeError(f"{prog.name}: {len(sizes)} encodings for {len(insns)} instructions")
    return "\n".join(branch_offsets(lines, dict(zip(insns, sizes)))) + "\n"


def translate(work, emulate):
    out = os.path.join(work, "spirv_emu" if emulate else "spirv_nat")
    cmd = [sys.executable, "-I", TRANSLATOR[0], os.path.join(work, "disasm"), "--out", out]
    if emulate:
        cmd.append("--wave64-emulate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out, r.stdout + r.stderr


def gpu_env(prog, emulate, dump, base=None):
    """vk_compute_test's environment (FUZZ-6, tools/lds_fuzz.py): VKC_OUT_WORDS sizes the buffer, VKC_DUMP gets its
    raw bytes. A wave32 program, or a wave64 one translated with --wave64-emulate, is written for 32-wide subgroups,
    so it asks for them with VKC_SUBGROUP_SIZE=32, as lds_fuzz.py and valu_fuzz.py do (FUZZ-13): on AMD, whose
    compute default is 64, a group over 32 would otherwise run as one 64-wide subgroup and mismatch in wave 1.
    A native wave64 program keeps the device default (--w64-native runs it on whatever the GPU has)."""
    env = dict(os.environ if base is None else base, VKC_OUT_WORDS=str(max(prog.group * len(OUT_REGS), 64)),
               VKC_DUMP=dump)
    if prog.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    return env


class HarnessError(RuntimeError):
    """vk_compute_test did not give a result for one program: it hung, crashed or wrote no dump."""


class Unpinned(RuntimeError):
    """vk_compute_test ran a wave32 program but couldn't pin the 32-lane subgroups it needs, so its words came from
    another wave size (Mesa llvmpipe: 8 lanes) and are no evidence either way: the program is skipped. Programs
    translated with --wave64-emulate use no subgroup operations, so they are judged at any size."""


GPU_TIMEOUT = 60    # seconds for one program; a translated program that never ends shows up as this


def run_gpu(exe, spv, prog, work, emulate):
    dump = os.path.join(work, "dump.bin")
    if os.path.exists(dump):
        os.remove(dump)                 # never compare a stale dump
    env = gpu_env(prog, emulate, dump)
    try:
        r = subprocess.run([exe, spv, "1", "dump"], capture_output=True, text=True, env=env, timeout=GPU_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HarnessError(f"vk_compute_test timed out after {GPU_TIMEOUT} s") from None
    import compute_fixture
    lanes = compute_fixture.unpinned_subgroup(r.stdout + r.stderr) if prog.wave == 32 else None
    if lanes:
        raise Unpinned(f"the device can't run it at {lanes} lanes (vk_compute_test: compute subgroup size: {lanes} "
                       "not available)")
    if not os.path.exists(dump):
        detail = " ".join(f"{r.stdout}{r.stderr}".split())
        raise HarnessError(f"vk_compute_test gave no dump (exit {r.returncode}){': ' + detail if detail else ''}")
    with open(dump, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def compare(prog, got, ops=None):
    exp, hyp = prog.expected(ops)
    bad = []
    for (tid, j), v in sorted(exp.items()):
        g = got[tid * len(OUT_REGS) + j]
        if g != v:
            bad.append((tid, j, v, g))
    return bad, hyp


def describe(prog, bad, limit=6):
    lines = []
    for tid, j, v, g in bad[:limit]:
        kind, r = OUT_REGS[j]
        lines.append(f"    thread {tid} (wave {tid // prog.wave}, lane {tid % prog.wave}) {kind}{r}: "
                     f"expected 0x{v:08x}, got 0x{g:08x}")
    if len(bad) > limit:
        lines.append(f"    ... {len(bad) - limit} more")
    return "\n".join(lines)


def minimize(prog, mode, ctx):
    """Greedy: drop ops while the program still mismatches (same mode). Returns the reduced op list."""
    ops = list(prog.ops)
    i = len(ops) - 1
    while i >= 0:
        trial = ops[:i] + ops[i + 1:]
        if still_fails(prog, trial, mode, ctx):
            ops = trial
        i -= 1
    return ops


def still_fails(prog, ops, mode, ctx):
    mc, exe = ctx
    with tempfile.TemporaryDirectory(prefix="xlmin-") as work:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(work, d))
        try:
            assemble(mc, prog, prog.source(ops), os.path.join(work, "disasm"), os.path.join(work, "fssm"))
        except RuntimeError:
            return False
        spdir, _ = translate(work, mode == "emulate")
        spv = os.path.join(spdir, prog.name + ".spv")
        if not os.path.exists(spv):
            return False
        try:
            bad, hyp = compare(prog, run_gpu(exe, spv, prog, work, mode == "emulate"), ops)
        except (HarnessError, Unpinned):
            return False                # a reduction that hangs or crashes is not the mismatch being reduced
        return bool(bad) and not hyp


MODES = ("native", "emulate")


def parse_modes(text):
    """--modes: a comma list of native and emulate, or both. Any other word is an error: the run loop only knows
    these two, and an unknown one used to run every program natively, wave64 included."""
    modes = []
    for word in (w.strip() for w in text.split(",")):
        if not word:
            continue
        for mode in MODES if word == "both" else (word,):
            if mode not in MODES:
                raise argparse.ArgumentTypeError(f"unknown mode {word!r} (use native, emulate or both)")
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("no mode given (use native, emulate or both)")
    return modes


def _at_least_one(text):
    """--programs and --ops: a count below 1 ran nothing (or a program of no ops) and still printed a summary, which
    reads as a clean pass. Refuse it as a usage error instead, as the other fuzzers do."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {value})")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--programs", type=_at_least_one, default=100,
                    help="programs per wave (at least 1; default 100)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ops", type=_at_least_one, default=24, help="ops per program (at least 1; default 24)")
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--modes", type=parse_modes, default="native,emulate",
                    help="native, emulate (wave64 only), or both (the default)")
    ap.add_argument("--w64-native", action="store_true",
                    help="also run wave64 programs without --wave64-emulate (on a 32-wide subgroup that is the "
                         "documented cross-half approximation, so mismatches are expected there)")
    ap.add_argument("--no-maskdata", action="store_true", help="leave out masks read as data (#293)")
    ap.add_argument("--no-w32-hi-lanes", action="store_true", help="wave32 lane selects stay below 32")
    ap.add_argument("--exe", default=os.path.join(ROOT, "runtime", "build", "vk_compute_test" +
                                                  (".exe" if os.name == "nt" else "")))
    ap.add_argument("--keep", help="copy failing programs (.asm) here")
    ap.add_argument("--minimize", action="store_true", help="reduce each failing program")
    ap.add_argument("--permlane-fetch-inactive", action="store_true",
                    help="model an inactive permlane source as read (FI = 1) instead of the hypothesis")
    ap.add_argument("--dpp", action="store_true", help="also generate DPP16 / DPP8 ops")
    ap.add_argument("--ext", action="store_true",
                    help="wave64 programs also get carries, ds_bpermute/ds_permute/ds_swizzle, lane selects of 64 "
                         "and up, b64 mask SALU with SCC probes, the saveexec/wrexec family, s_bfm_b64, s_wqm_b64 "
                         "and SI_LOOP-shaped loops (wave32 programs and seeds without the flag are unchanged)")
    ap.add_argument("--dpp-sweep", action="store_true",
                    help="instead of random programs, one program per DPP case: every DPP16 control x bound_ctrl x "
                         "fi x EXEC pattern, all row_mask / bank_mask pairs, and a set of DPP8 selects "
                         "(--programs, --seed, --ops and --dpp are ignored; --wave applies)")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated: translator (default; needs llvm-mc, the Vulkan SDK and vk_compute_test), hw "
                         "(natively on the AMD GPU, wave32 and native wave64; tools/hw_oracle/), or translator,hw")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--no-hw-guard", action="store_true",
                    help="no s_nop 7 after each instruction in the native kernels (hazards then show up as mismatches)")
    ap.add_argument("--translator", help="rdna2spirv.py to test (default: the one next to this file)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(args.exe):   # a Visual Studio build puts it in a configuration subfolder
        folder, name = os.path.split(args.exe)
        for config in ("RelWithDebInfo", "Release"):
            if os.path.isfile(os.path.join(folder, config, name)):
                args.exe = os.path.join(folder, config, name)
                break
    # Windows' CreateProcess searches for a relative path written with "/" that doesn't start with "./" or "../"
    # (runtime/build/x.exe) instead of taking it from the working directory, though os.path.isfile does, so such an
    # --exe passed the check below and the first GPU run ended in a FileNotFoundError traceback. An absolute path
    # (abspath also turns "/" into "\" there) runs from anywhere.
    args.exe = os.path.abspath(args.exe)
    global PERMLANE_FETCH_INACTIVE
    PERMLANE_FETCH_INACTIVE = args.permlane_fetch_inactive
    if args.translator:
        TRANSLATOR[0] = os.path.abspath(args.translator)
    if args.self_test:
        return self_test()
    backends = tuple(dict.fromkeys(t.strip() for t in args.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    if args.wave == "32" and "native" not in args.modes:
        # emulate runs a wave64 program as two 32-lane halves, so a wave32-only selection has nothing to run: the run
        # used to end after a bare `summary:` with exit 0, the same false pass #1452 fixed for a count below 1.
        # Only when emulate is the sole mode: with the default (native,emulate) native runs every wave32 program.
        ap.error("--modes emulate needs wave64 programs: with --wave 32 nothing runs (use --wave 64 or both)")
    if args.wave == "64" and "emulate" not in args.modes and not args.w64_native:
        # A wave64 program only runs natively when --w64-native asks for the documented cross-half approximation, so
        # this combination was empty in the same silent way. With the default modes emulate runs them.
        ap.error("--wave 64 --modes native runs nothing without --w64-native; pass it to run the documented "
                 "cross-half approximation, or use --modes emulate")
    if "translator" in backends and not os.path.isfile(args.exe):   # before any program is generated or run
        sys.exit(f"vk_compute_test not found at {args.exe} or in its RelWithDebInfo/Release folder; build it first "
                 "(cmake --build <build dir> --target vk_compute_test) or give its path with --exe")
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=args.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    waves = (32, 64) if args.wave == "both" else (int(args.wave),)
    modes = args.modes
    progs = []
    if args.dpp_sweep:
        for wave in waves:
            progs += dpp_sweep_programs(wave)
        print(f"dpp sweep: {len(progs)} cases")
    for i in range(0 if args.dpp_sweep else args.programs):
        wave = waves[i % len(waves)]
        progs.append(make_program(args.seed * 100003 + i, wave, args.ops, allow_maskdata=not args.no_maskdata,
                                  wave32_hi_lanes=not args.no_w32_hi_lanes, dpp=args.dpp, ext=args.ext))
    stats = {}
    fails = []
    try:
        if oracle:
            run_hw(oracle, progs, stats, fails, guard=not args.no_hw_guard)
        mc = None
        if "translator" in backends:
            import mc_command
            mc = mc_command.mc_command()
            run_translator(args, mc, progs, waves, modes, stats, fails)
        tr_status = {}
        for p, mode, what, bad in fails:
            if mode != "hw":
                tr_status.setdefault(p.name, []).append(f"{mode}: {what}")
        for p, mode, what, bad in fails:
            if mode == "hw" and bad and "translator" in backends:
                print(f"{p.name}: hardware and reference differ; translator: " +
                      ("; ".join(tr_status.get(p.name, [])) or "matches the reference"))
        for p, mode, what, bad in fails:
            tags = sorted({t for o in p.ops for t in getattr(o, 'tags', {o.tag})})
            print(f"{p.name} [{mode}] {what}; group {p.group}; ops: {', '.join(tags)}")
            if p.label:
                print(f"  case: {p.label}")
            if bad:
                print(describe(p, bad))
            if args.keep:
                os.makedirs(args.keep, exist_ok=True)
                with open(os.path.join(args.keep, f"{p.name}_{mode}.asm"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(kept_source(p, mode, None, not args.no_hw_guard))
            if args.minimize and bad and what.startswith("mismatch"):
                if mode == "hw":
                    ops = minimize_hw(p, oracle, not args.no_hw_guard)
                elif mc is not None:
                    ops = minimize(p, mode, (mc, args.exe))
                else:
                    continue
                hyp_ops = p.expected(ops)[1]
                print("  minimized:\n\t" + "\n\t".join(o.text + ("   ; hypothesis op" if i in hyp_ops else "")
                                                         for i, o in enumerate(ops)))
                if args.keep:
                    with open(os.path.join(args.keep, f"{p.name}_{mode}_min.asm"), "w", encoding="utf-8", newline="\n") as f:
                        f.write(kept_source(p, mode, ops, not args.no_hw_guard))
    finally:
        if oracle:
            oracle.close()
    print("summary:")
    for key, st in sorted(stats.items()):
        print(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    if args.dpp_sweep:
        print("dpp sweep by family (cases, failing per oracle mode):")
        failing = {}
        for p, mode, what, bad in fails:
            failing.setdefault((p.wave, p.family, mode), set()).add(p.name)
        for wave in waves:
            for family in dict.fromkeys(p.family for p in progs if p.wave == wave):
                n = sum(p.wave == wave and p.family == family for p in progs)
                bad = ", ".join(f"{m} {len(s)}" for (w, f, m), s in sorted(failing.items())
                                if w == wave and f == family) or "none"
                print(f"  wave{wave} {family}: {n} cases; failing: {bad}")
    if any(st["mismatch"] or st["rejected"] for st in stats.values()):
        return 1
    if any(st.get("harness_errors") for st in stats.values()):
        return 2
    if stats and not any(st["match"] or st["hyp_only"] for st in stats.values()):
        print("no program ran: every one was skipped")
        return 2
    return 0


def kept_source(prog, mode, ops, guard):
    """What --keep writes: for a hardware failure the kernel body that actually ran, else the translator input."""
    if mode == "hw":
        return "\n".join(prog.hw_lines(ops, guard)) + "\n"
    return prog.source(ops)


def hw_skip_reason(prog, arch):
    """Why a program cannot be compared on this GPU, or None. Nothing the generator emits differs between GFX10 and
    GFX11 in a way that changes a result (checked by running it; docs/shader_validation.md lists what was looked
    at), so the only skips are assembler rejections, which the runner reports."""
    return None


def run_hw(oracle, progs, stats, fails, guard=True):
    """Run every program natively (wave32 or wave64 as the program is, its real group size, one thread per lane)
    and compare all stored words with the model."""
    from hw_oracle import asm
    kernels, which = [], []
    for p in progs:
        st = stats.setdefault(f"wave{p.wave} hw", {"programs": 0, "match": 0, "mismatch": 0, "hyp_only": 0,
                                                    "hyp_confirmed": 0, "rejected": 0})
        st["programs"] += 1
        why = hw_skip_reason(p, oracle.arch)
        if why:
            st["rejected"] += 1
            fails.append((p, "hw", f"skipped: {why}", None))
            continue
        kernels.append(asm.Kernel(p.hw_lines(guard=guard), p.wave, p.group, p.group * len(OUT_REGS), raw=True))
        which.append(p)
    for p, got in zip(which, oracle.run(kernels)):
        st = stats[f"wave{p.wave} hw"]
        if isinstance(got, str):
            st["rejected"] += 1
            fails.append((p, "hw", got, None))
            continue
        bad, hyp = compare(p, got)
        if not bad:
            st["match"] += 1
            st["hyp_confirmed"] += bool(hyp)
            for kind in sorted({t for i in hyp for t in getattr(p.ops[i], "tags", {p.ops[i].tag})
                                if t in ("permlane", "dpp", "swizzle")}):
                st["hyp_" + kind] = st.get("hyp_" + kind, 0) + 1
        else:
            # On the hardware a failing hypothesis is a falsified claim, and it could hide a model bug elsewhere in
            # the program, so it is a mismatch like any other (hyp_only counts the ones that had a hypothesis op).
            st["mismatch"] += 1
            if hyp:
                st["hyp_only"] += 1
            fails.append((p, "hw", "mismatch" + (" (program has a hypothesis op)" if hyp else ""), bad))


def minimize_hw(prog, oracle, guard=True):
    """Drop ops while the program still differs on the hardware."""
    from hw_oracle import asm

    def fails_with(ops):
        got = oracle.run([asm.Kernel(prog.hw_lines(ops, guard), prog.wave, prog.group,
                                     prog.group * len(OUT_REGS), raw=True)])[0]
        if isinstance(got, str):
            return False
        return bool(compare(prog, got, ops)[0])
    ops = list(prog.ops)
    for i in range(len(ops) - 1, -1, -1):
        trial = ops[:i] + ops[i + 1:]
        if fails_with(trial):
            ops = trial
    return ops


def run_translator(args, mc, progs, waves, modes, stats, fails):
    with tempfile.TemporaryDirectory(prefix="crosslane-") as work:
        disasm, fssm = os.path.join(work, "disasm"), os.path.join(work, "fssm")
        os.mkdir(disasm)
        os.mkdir(fssm)
        for p in progs:
            assemble(mc, p, p.source(), disasm, fssm)
        for mode in modes:
            if mode == "emulate" and 64 not in waves:
                continue
            spdir, log = translate(work, mode == "emulate")
            for p in progs:
                if mode == "emulate" and p.wave != 64:
                    continue
                if mode == "native" and p.wave == 64 and not args.w64_native:
                    continue
                key = f"wave{p.wave} {mode}"
                st = stats.setdefault(key, {"programs": 0, "match": 0, "mismatch": 0, "hyp_only": 0,
                                            "rejected": 0, "harness_errors": 0, "skipped": 0})
                st["programs"] += 1
                spv = os.path.join(spdir, p.name + ".spv")
                if not os.path.exists(spv):
                    st["rejected"] += 1
                    fails.append((p, mode, "not translated", None))
                    continue
                try:
                    bad, hyp = compare(p, run_gpu(args.exe, spv, p, work, mode == "emulate"))
                except HarnessError as e:
                    # One program that hangs or crashes the harness used to end the run with a traceback and
                    # lose every result before it. It is reported like the other failures and the run goes on.
                    st["harness_errors"] += 1
                    fails.append((p, mode, f"harness error: {e}", None))
                    continue
                except Unpinned as e:
                    st["skipped"] += 1
                    print(f"SKIP {p.name} [{mode}]: {e}")
                    continue
                if not bad:
                    st["match"] += 1
                elif hyp:
                    st["hyp_only"] += 1
                    fails.append((p, mode, "mismatch (program has a hypothesis op)", bad))
                else:
                    st["mismatch"] += 1
                    fails.append((p, mode, "mismatch", bad))


def self_test():
    """The model against values worked out by hand (no GPU, no llvm-mc)."""
    w = Wave(64, M64)
    w.v[2] = [3 * l + 1 for l in range(64)]
    g = Gen(random.Random(0), 64, 64)
    # readlane: lane 40 of 3l+1 = 121, independent of EXEC.
    w.exec = 1 << 63
    Op("", lambda x: x.s.__setitem__(8, x.vr(2)[40]), "t").apply([w])
    assert w.s[8] == 121
    # readfirstlane with only lane 63 on: 190; with EXEC 0: lane 0 = 1.
    rf = g.op_readfirstlane()
    w.exec = 1 << 63
    rf.apply([w])
    d = int(rf.text.split()[1].strip(",")[1:])
    v = int(rf.text.split()[2][1:])
    assert w.s[d] == w.vr(v)[63]
    w.exec = 0
    rf.apply([w])
    assert w.s[d] == w.vr(v)[0]
    # mbcnt of EXEC = all ones gives the lane.
    w.exec = M64
    vals = [popcount(M64 & ((1 << l) - 1)) for l in range(64)]
    assert vals[0] == 0 and vals[40] == 40 and vals[63] == 63
    # permlanex16 with selectors 0: lane 5 reads lane 16, lane 20 reads lane 0, lane 40 reads lane 48.
    x16 = [((l & ~31) | ((l & 16) ^ 16)) for l in range(64)]
    assert x16[5] == 16 and x16[20] == 0 and x16[40] == 48
    print("self-test OK")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
