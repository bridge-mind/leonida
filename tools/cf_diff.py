r"""FUZZ-8: differential fuzzer for control flow and EXEC (standard library only).

Generates seeded RDNA2 pixel-shader programs built from the control-flow shapes compilers emit, runs every lane
of a wave through two independent sides and compares each lane's export bit for bit:

  * tools/cf_ref.py: the ISA model (cpu_ref's Machine plus branches, compares, v_cmpx and saveexec), one wave
  * tools/rdna2spirv.py + tests/semantic/spirv_wave_interp.py: the translated SPIR-V, every invocation of one
    subgroup run in lockstep with structured reconvergence (subgroup size = wave size, the translator's contract)

Shapes (each nests, up to --depth):
  if        v_cmp -> s_and_saveexec -> [s_cbranch_execz] -> body -> s_or exec        (LLVM SI_IF / SI_END_CF)
  ifelse    ... s_xor saveexec mask, s_or_saveexec + s_xor exec for the else side     (LLVM SI_ELSE)
  cmpx      s_mov save, exec -> v_cmpx -> [s_cbranch_execz] -> body -> s_mov exec, save
  vccz      v_cmp -> s_cbranch_vccz/vccnz over a body that every active lane runs      (wave-uniform branch)
  sif       s_cmp -> s_cbranch_scc0/1 [-> s_branch over an else]                       (uniform branch)
  sloop     uniform counter loop closed by s_cbranch_scc1                              (natural loop)
  vloop     per-lane trip counts: break mask, s_andn2 exec, s_cbranch_execnz          (LLVM SI_LOOP)
  irr       two-entry cycle on uniform SCC branches                                   (irreducible graph)
  sccmask   s_and/andn2/or/xor of VCC and EXEC -> s_cbranch_scc0/1 (SCC = whole mask != 0)
  csel      the same SCC read by s_cselect_b32 into a uniform SGPR
(vloop also closes some loops with s_cbranch_scc1 on the SCC of its s_andn2 exec.)
Bodies mix integer VALU and SALU ops with "probes": v11 = v11 * 33 + K at each site, so v11 records which sites
each lane ran, in which order and how often. v8/v9 carry data, v10 the uniform SGPRs. Immediates stay in the
inline-constant range so every instruction has a known size, and the listing carries `; encoding:` bytes of
those sizes (zeros: the translator reads only their length), which places the branch targets.

Usage:
    python -I tools/cf_diff.py [--programs N] [--seed S] [--wave 32|64|both] [--dispatch-loop] [--depth D]
    python -I tools/cf_diff.py --self-test
    python -I tools/cf_diff.py --spirv-val ...                              # also spirv-as + spirv-val each module
    python -I tools/cf_diff.py --seed S --program K --show [--minimize]     # print (and shrink) one program

Exit code 0 when every program matches, 1 on a mismatch (case, seed, lane, the expected and actual words and the
listing; with --minimize also a reduced listing), 2 on harness errors or when every program was skipped (nothing was
compared). Skips are counted with reasons.
"""
import argparse
import gc
import os
import random
import shutil
import struct
import subprocess
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

import cf_ref  # noqa: E402
import cpu_ref  # noqa: E402
import rdna2spirv  # noqa: E402
from spirv_wave_interp import UnsupportedConstruct, run_wave  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
DATA = (1, 2, 3, 8, 9)            # VGPRs the bodies compute on (v0 = lane id is kept intact)
SREGS = (10, 11, 12, 13)          # uniform SGPRs the bodies compute on
CCS = ("eq", "ne", "lt", "le", "gt", "ge")
VOP3 = ("v_mul_lo_u32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32")


# ---- program tree --------------------------------------------------------------------------------------------
# A program is a list of nodes. ("op", line) is one instruction; compound nodes hold child lists and are expanded
# by Emitter into instructions, labels and branches. Keeping the tree lets --minimize delete whole structures.

class Gen:
    def __init__(self, rng, wave, max_depth, budget):
        self.r, self.wave, self.max_depth, self.budget = rng, wave, max_depth, budget
        self.site = 0

    def cmp(self):
        """A per-lane condition: (cc, type, constant, vgpr). Constants near 0 and 64 make some conditions
        true or false for every lane, so the execz/vccz skips are taken too."""
        r = self.r
        k = r.choice((0, 1, 2, 3, 4, 5, 7, 8, 12, 16, 31, 32, 63, 64, -1, -16))
        return (r.choice(CCS), r.choice(("u32", "u32", "i32")), k, r.choice((0, 0, 1, 2, 3, 8, 9)))

    def probe(self):
        self.site += 1
        return ("probe", 1 + (self.site * 37) % 64)

    def leaf(self):
        r = self.r
        d, s = r.choice(DATA), r.choice(DATA)
        k = r.randrange(-16, 65)
        sr = r.choice(SREGS)
        choices = [
            ("op", f"v_add_nc_u32 v{d}, {k}, v{s}"),
            ("op", f"v_sub_nc_u32 v{d}, v{s}, v{r.choice(DATA)}"),
            ("op", f"v_xor_b32 v{d}, s{sr}, v{s}"),
            ("op", f"v_and_b32 v{d}, {r.randrange(0, 65)}, v{s}"),
            ("op", f"v_or_b32 v{d}, v{r.choice(DATA)}, v{s}"),
            ("op", f"v_lshlrev_b32 v{d}, {r.randrange(0, 8)}, v{s}"),
            ("op", f"v_lshrrev_b32 v{d}, {r.randrange(0, 8)}, v{s}"),
            ("op", f"v_mul_lo_u32 v{d}, v{s}, {r.randrange(0, 65)}"),
            ("op", f"v_mov_b32 v{d}, s{sr}"),
            ("op", f"v_add_nc_u32 v{d}, s{sr}, v{s}"),
            ("op", f"s_add_u32 s{sr}, s{sr}, {r.randrange(0, 65)}"),
            ("op", f"s_xor_b32 s{sr}, s{r.choice(SREGS)}, {r.randrange(0, 65)}"),
            ("op", f"s_mov_b32 s{sr}, {r.randrange(-16, 65)}"),
            ("cnd", d, s, self.cmp()),
        ]
        return r.choice(choices)

    def block(self, depth):
        out = []
        n = self.r.randrange(1, 4)
        for _ in range(n):
            if self.budget <= 0:
                break
            self.budget -= 1
            if depth < self.max_depth and self.r.random() < 0.55:
                out.append(self.compound(depth + 1))
            else:
                out.append(self.leaf())
            if self.r.random() < 0.6:
                out.append(self.probe())
        return out

    def compound(self, depth):
        r = self.r
        kind = r.choice(("if", "if", "ifelse", "ifelse", "cmpx", "saveexec", "vccz", "sif", "sloop", "vloop",
                         "vloop", "irr", "sccmask", "csel"))
        if kind == "saveexec":
            return ("saveexec", self.cmp(), r.choice(("andn2", "orn2", "xor", "nand", "nor", "xnor")),
                    r.choice((None, "s_cbranch_execz", "s_cbranch_scc0")), [self.probe()] + self.block(depth))
        if kind == "sccmask":
            return ("sccmask", self.cmp(), r.choice(("and", "andn2", "or", "xor")),
                    r.choice(("s_cbranch_scc0", "s_cbranch_scc1")), [self.probe()] + self.block(depth))
        if kind == "csel":
            return ("csel", self.cmp(), r.choice(("and", "andn2", "or", "xor")), r.choice(SREGS))
        if kind == "if":
            return ("if", self.cmp(), r.random() < 0.7, [self.probe()] + self.block(depth))
        if kind == "ifelse":
            return ("ifelse", self.cmp(), [self.probe()] + self.block(depth), [self.probe()] + self.block(depth))
        if kind == "cmpx":
            return ("cmpx", self.cmp(), r.random() < 0.7, [self.probe()] + self.block(depth))
        if kind == "vccz":
            return ("vccz", self.cmp(), r.choice(("s_cbranch_vccz", "s_cbranch_vccnz")),
                    [self.probe()] + self.block(depth))
        if kind == "sif":
            other = [self.probe()] + self.block(depth) if r.random() < 0.5 else None
            return ("sif", (r.choice(CCS), r.choice(SREGS), r.randrange(0, 65)), r.random() < 0.5,
                    [self.probe()] + self.block(depth), other)
        if kind == "sloop":
            return ("sloop", r.randrange(1, 4), [self.probe()] + self.block(depth))
        if kind == "vloop":
            return ("vloop", (r.choice((0, 1, 2, 3)), r.choice((1, 3)), r.random() < 0.3),
                    self.cmp() if r.random() < 0.4 else None, [self.probe()] + self.block(depth))
        return ("irr", (r.choice(SREGS), r.randrange(0, 65)), [self.probe()] + self.block(depth),
                [self.probe()] + self.block(depth))


def generate(rng, wave, max_depth=3, budget=10):
    g = Gen(rng, wave, max_depth, budget)
    body = []
    while g.budget > 0:
        body.extend(g.block(0))
    body.append(g.probe())
    return body


class Emitter:
    """Expands a program tree into tab-less instruction lines with sizes, labels and symbolic branches."""

    def __init__(self, wave):
        self.wave = wave
        self.items = []            # ("ins", text) | ("label", name) | ("br", mnemonic, label)
        self.n_labels = 0
        self.depth = 0

    # register names -------------------------------------------------------------------------------------------
    def exec_(self):
        return "exec" if self.wave == 64 else "exec_lo"

    def vcc(self):
        return "vcc" if self.wave == 64 else "vcc_lo"

    def b(self):
        return "b64" if self.wave == 64 else "b32"

    def sreg(self, base):
        """A mask SGPR for the current nesting depth (an aligned pair in wave64)."""
        n = base + 2 * self.depth
        return f"s[{n}:{n + 1}]" if self.wave == 64 else f"s{n}"

    def label(self):
        self.n_labels += 1
        return f"L{self.n_labels}"

    def ins(self, text):
        self.items.append(("ins", text))

    def br(self, mnemonic, label):
        self.items.append(("br", mnemonic, label))

    def place(self, label):
        self.items.append(("label", label))

    def vcmp(self, c, dst=None):
        cc, ty, k, v = c
        self.ins(f"v_cmp_{cc}_{ty}_e32 {dst or self.vcc()}, {k}, v{v}")

    # program ------------------------------------------------------------------------------------------------------
    def program(self, body):
        if self.wave == 64:
            self.ins("v_mbcnt_lo_u32_b32 v0, -1, 0")
            self.ins("v_mbcnt_hi_u32_b32 v0, -1, v0")
        else:
            self.ins("v_mbcnt_lo_u32_b32 v0, -1, 0")
        self.ins("v_and_b32 v1, 7, v0")
        self.ins("v_lshrrev_b32 v2, 2, v0")
        self.ins("v_mul_lo_u32 v3, v0, 5")
        self.ins("v_and_b32 v3, 15, v3")
        self.ins("v_xor_b32 v8, 9, v0")
        self.ins("v_mov_b32 v9, 0")
        self.ins("v_mov_b32 v11, 1")
        for i, s in enumerate(SREGS):
            self.ins(f"s_mov_b32 s{s}, {(3, 5, 0, 12)[i]}")
        self.nodes(body)
        self.ins(f"v_mov_b32 v10, s{SREGS[0]}")
        for s in SREGS[1:]:
            self.ins("v_lshlrev_b32 v10, 3, v10")
            self.ins(f"v_xor_b32 v10, s{s}, v10")
        self.ins("exp mrt0 v8, v9, v10, v11 done vm")
        self.ins("s_endpgm")

    def nodes(self, nodes):
        for n in nodes:
            self.node(n)

    def node(self, n):
        kind = n[0]
        w = self.b()
        ex = self.exec_()
        if kind == "op":
            self.ins(n[1])
        elif kind == "probe":
            self.ins("v_mul_lo_u32 v11, v11, 33")
            self.ins(f"v_add_nc_u32 v11, {n[1]}, v11")
        elif kind == "cnd":
            _, d, s, c = n
            self.vcmp(c)
            self.ins(f"v_cndmask_b32_e32 v{d}, 7, v{s}, {self.vcc()}")
        elif kind == "if":
            _, c, skip, body = n
            save, end = self.sreg(20), self.label()
            self.vcmp(c)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "ifelse":
            _, c, then, other = n
            save, els, end = self.sreg(20), self.label(), self.label()
            self.vcmp(c)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            self.ins(f"s_xor_{w} {save}, {ex}, {save}")
            self.br("s_cbranch_execz", els)
            self.depth += 1
            self.nodes(then)
            self.depth -= 1
            self.place(els)
            self.ins(f"s_or_saveexec_{w} {save}, {save}")
            self.ins(f"s_xor_{w} {ex}, {ex}, {save}")
            self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(other)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "cmpx":
            _, (cc, ty, k, v), skip, body = n
            save, end = self.sreg(20), self.label()
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.ins(f"v_cmpx_{cc}_{ty}_e32 {k}, v{v}")
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "saveexec":
            _, c, op_name, skip_br, body = n
            save, end = self.sreg(20), self.label()
            self.vcmp(c)
            self.ins(f"s_{op_name}_saveexec_{w} {save}, {self.vcc()}")
            if skip_br is not None:
                self.br(skip_br, end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "vccz":
            _, c, mnemonic, body = n
            end = self.label()
            self.vcmp(c)
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "sif":
            _, (cc, sr, k), on_one, then, other = n
            els, end = self.label(), self.label()
            self.ins(f"s_cmp_{cc}_u32 s{sr}, {k}")
            self.br("s_cbranch_scc0" if on_one else "s_cbranch_scc1", els if other else end)
            self.nodes(then)
            if other:
                self.br("s_branch", end)
                self.place(els)
                self.nodes(other)
            self.place(end)
        elif kind == "sloop":
            _, trips, body = n
            ctr, head = 14 + self.depth, self.label()
            self.ins(f"s_mov_b32 s{ctr}, 0")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 1")
            self.ins(f"s_cmp_lt_u32 s{ctr}, {trips}")
            self.br("s_cbranch_scc1", head)
        elif kind == "sccmask":
            # SCC of a mask op is (whole result != 0): one value for the wave, like a vote.
            _, c, logic, mnemonic, body = n
            tmp, end = self.sreg(50), self.label()
            self.vcmp(c)
            self.ins(f"s_{logic}_{w} {tmp}, {self.vcc()}, {ex}")
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "csel":
            _, c, logic, sr = n
            self.vcmp(c)
            self.ins(f"s_{logic}_{w} {self.sreg(50)}, {self.vcc()}, {ex}")
            self.ins(f"s_cselect_b32 s{sr}, s{sr}, 17")
        elif kind == "vloop":
            _, (src, mask, scc_exit), brk, body = n
            it, lim, brk_mask, head = 4 + self.depth, 30 + self.depth, self.sreg(40), self.label()
            self.ins(f"v_and_b32 v{lim}, {mask}, v{src}")
            self.ins(f"v_add_nc_u32 v{lim}, 1, v{lim}")
            self.ins(f"v_mov_b32 v{it}, 0")
            self.ins(f"s_mov_{w} {brk_mask}, 0")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            if brk is not None:
                # An early exit (SI_IF_BREAK): lanes meeting `brk` join the break mask for this iteration.
                self.vcmp(brk)
                self.ins(f"s_or_{w} {brk_mask}, {self.vcc()}, {brk_mask}")
            self.ins(f"v_add_nc_u32 v{it}, 1, v{it}")
            self.ins(f"v_cmp_ge_u32_e32 {self.vcc()}, v{it}, v{lim}")
            self.ins(f"s_or_{w} {brk_mask}, {self.vcc()}, {brk_mask}")
            self.ins(f"s_andn2_{w} {ex}, {ex}, {brk_mask}")
            # Loop again while any lane is left: EXECNZ, or SCC of the s_andn2 (= new EXEC != 0).
            self.br("s_cbranch_scc1" if scc_exit else "s_cbranch_execnz", head)
            self.ins(f"s_or_{w} {ex}, {ex}, {brk_mask}")
        elif kind == "irr":
            _, (sr, k), part_a, part_b = n
            ctr, a_lbl, b_lbl, end = 14 + self.depth, self.label(), self.label(), self.label()
            self.ins(f"s_mov_b32 s{ctr}, 0")
            self.ins(f"s_cmp_lt_u32 s{sr}, {k}")
            self.br("s_cbranch_scc0", b_lbl)
            self.place(a_lbl)
            self.depth += 1
            self.nodes(part_a)
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 1")
            self.ins(f"s_cmp_lt_u32 s{ctr}, 4")
            self.br("s_cbranch_scc0", end)
            self.place(b_lbl)
            self.nodes(part_b)
            self.depth -= 1
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 2")
            self.ins(f"s_cmp_lt_u32 s{ctr}, 7")
            self.br("s_cbranch_scc1", a_lbl)
            self.place(end)
        else:
            raise ValueError(f"unknown node {kind}")

    def assemble(self):
        """Lines with real branch offsets: [(text, size_in_bytes)]."""
        pc, where, sized = 0, {}, []
        for it in self.items:
            if it[0] == "label":
                where[it[1]] = pc
                continue
            size = 4 if it[0] == "br" else _size(it[1])
            sized.append((it, pc, size))
            pc += size
        out = []
        for it, at, size in sized:
            if it[0] == "br":
                off = (where[it[2]] - (at + 4)) // 4
                out.append((f"{it[1]} {off}", size))
            else:
                out.append((it[1], size))
        return out


def _size(text):
    op = text.split()[0]
    if op == "exp" or op.endswith("_e64") or op in VOP3:
        return 8
    return 4


def listing_text(lines, name):
    out = [f"; {name}  code 0x0..0x{sum(s for _, s in lines):x}"]
    for text, size in lines:
        out.append("\t" + text + " ; encoding: [" + ",".join(["0x00"] * size) + "]")
    return "\n".join(out) + "\n"


def build(body, wave):
    e = Emitter(wave)
    e.program(body)
    return e.assemble()


# ---- one comparison ------------------------------------------------------------------------------------------

def run_case(name, body, wave, structured=True):
    """Returns (status, detail): status in {"match", "mismatch", "skip"}."""
    lines = build(body, wave)
    text = listing_text(lines, name)
    with tempfile.TemporaryDirectory(prefix="cf-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            ref = cf_ref.run(path, wave=wave)
        except cpu_ref.CpuRefError as e:
            return ("skip", f"cf_ref: {e}")
        old = rdna2spirv.STRUCTURED
        rdna2spirv.STRUCTURED = structured
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gen = rdna2spirv.translate.last_gen
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return ("skip", f"translator unsupported: {e}")
        finally:
            rdna2spirv.STRUCTURED = old
        if SPIRV_VAL:
            bad = _spirv_val(spv, tmp)
            if bad:
                return ("mismatch", f"spirv-val: {bad}")
    if bool(getattr(gen, "wave64", False)) != (wave == 64):
        return ("mismatch", f"translator detected wave{64 if gen.wave64 else 32}, the program is wave{wave}")
    try:
        got = run_wave(spv, wave=wave)["outputs"]
    except UnsupportedConstruct as e:
        return ("skip", f"wave interp: {e}")
    for lane in range(wave):
        want = ref["exports"].get(lane)
        have = (got[lane] or {}).get("%out_mrt0") if got[lane] is not None else None
        if want == have:
            continue
        if want is None or have is None:
            return ("mismatch", f"lane {lane}: export {'missing' if have is None else 'unexpected'} in the "
                                f"translation (reference {_words(want)}, translation {_words(have)})")
        bad = [i for i in range(4) if want[i] != have[i]]
        what = {0: "v8 (data)", 1: "v9 (data)", 2: "v10 (uniform SGPRs)", 3: "v11 (path probes)"}
        return ("mismatch", f"lane {lane}: {', '.join(what[i] for i in bad)} differ: reference {_words(want)}, "
                            f"translation {_words(have)}")
    return ("match", "")


SPIRV_VAL = False       # --spirv-val: also assemble and validate every translation (spirv-as/spirv-val on PATH)


def _spirv_val(spv, tmp):
    """First spirv-as/spirv-val error for the module (Vulkan 1.2, as tests/conftest.py), or None."""
    sa, sv = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
    with open(sa, "w", encoding="utf-8", newline="\n") as f:
        f.write(spv)
    for cmd in (["spirv-as", sa, "-o", sv, "--target-env", "vulkan1.2"],
                ["spirv-val", "--target-env", "vulkan1.2", sv]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return (cmd[0] + ": " + (r.stderr or r.stdout).strip().splitlines()[0]) if (r.stderr or r.stdout) \
                else f"{cmd[0]} exited with {r.returncode}"
    return None


def _words(w):
    return "none" if w is None else "(" + ", ".join(f"0x{x:08x}" for x in w) + ")"


# ---- reduction -----------------------------------------------------------------------------------------------

def _variants(body):
    """Smaller programs: each node deleted, and each compound node replaced by one of its child lists."""
    for i, n in enumerate(body):
        yield body[:i] + body[i + 1:]
        kids = [x for x in n[1:] if isinstance(x, list)]
        for k in kids:
            yield body[:i] + k + body[i + 1:]
        for j, x in enumerate(n):
            if isinstance(x, list):
                for v in _variants(x):
                    m = list(n)
                    m[j] = v
                    yield body[:i] + [tuple(m)] + body[i + 1:]


def minimize(body, wave, structured=True):
    """Greedy reduction that keeps a mismatch."""
    changed = True
    while changed:
        changed = False
        for cand in _variants(body):
            status, _ = run_case("ps_cf_min", cand, wave, structured)
            if status == "mismatch":
                body, changed = cand, True
                break
    return body


# ---- suite ---------------------------------------------------------------------------------------------------

def program_for(seed, index, wave, depth):
    rng = random.Random(f"{seed}:{index}:{wave}")
    return generate(rng, wave, max_depth=depth, budget=rng.randrange(4, 13))


def run_suite(programs, seed, waves, structured=True, depth=3, limit=None):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "harness-error": 0}
    failures = []
    total = programs if limit is None else min(programs, limit)
    for i in range(total):
        for wave in waves:
            name = f"ps_cf_{i:04d}_w{wave}"
            body = program_for(seed, i, wave, depth)
            try:
                status, detail = run_case(name, body, wave, structured)
            except Exception as e:  # noqa: BLE001 - harness bugs must be loud
                counts["harness-error"] += 1
                failures.append((i, wave, f"harness error {type(e).__name__}: {e}"))
                continue
            if status == "match":
                counts["pass"] += 1
            elif status == "mismatch":
                counts["mismatch"] += 1
                failures.append((i, wave, detail))
            else:
                counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
    return counts, failures


def self_test():
    """Harness canaries: a program must match, a sabotaged reference must be caught, the interpreter must
    reconverge after a divergent branch."""
    ok = True
    body = [("if", ("gt", "u32", 4, 0), True, [("probe", 5), ("op", "v_add_nc_u32 v9, 1, v9")]), ("probe", 6)]
    for wave in (32, 64):
        status, detail = run_case("ps_canary_ok", body, wave)
        if status != "match":
            print(f"canary FAILED (wave{wave}): hand-written if did not match ({status}: {detail})")
            ok = False
    # The reference must be able to disagree: drop the s_or that restores EXEC and the lanes > 4 never export.
    e = Emitter(32)
    e.program(body)
    e.items = [it for it in e.items if not (it[0] == "ins" and it[1].startswith("s_or_b32 exec_lo"))]
    with tempfile.TemporaryDirectory(prefix="cf-diff-") as tmp:
        p = os.path.join(tmp, "c.s")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(listing_text(e.assemble(), "ps_canary_bad"))
        if len(cf_ref.run(p, wave=32)["exports"]) != 4:     # 4 > lane: lanes 0-3 stay on
            print("canary FAILED: the reference did not drop lanes whose EXEC was never restored")
            ok = False
    # Divergent loop + vote: every lane must leave the loop with its own trip count.
    body = [("vloop", (0, 3, False), None, [("probe", 9), ("vccz", ("eq", "u32", 0, 1), "s_cbranch_vccz",
                                                    [("op", "s_add_u32 s10, s10, 1")])])]
    for wave in (32, 64):
        status, detail = run_case("ps_canary_loop", body, wave)
        if status != "match":
            print(f"canary FAILED (wave{wave}): divergent loop did not match ({status}: {detail})")
            ok = False
    if ok:
        print("canaries: match ok, sabotaged reference detected, divergent loop reconverges")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="FUZZ-8: control-flow/EXEC differential fuzzer.")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--depth", type=int, default=3, help="maximum nesting depth")
    ap.add_argument("--dispatch-loop", action="store_true", help="translate in the dispatch-loop form")
    ap.add_argument("--program", type=int, help="only this program index (with --show / --minimize)")
    ap.add_argument("--show", action="store_true", help="print the listing of --program")
    ap.add_argument("--minimize", action="store_true", help="reduce a mismatching program before printing it")
    ap.add_argument("--spirv-val", action="store_true",
                    help="also run spirv-as and spirv-val on every translation (a failure counts as a mismatch)")
    ap.add_argument("--self-test", action="store_true")
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        ap.error("no arguments given; pass --help for the available options")
    a = ap.parse_args(argv)
    if a.show and a.program is None:
        ap.error("--show requires --program")
    if a.programs < 1:
        ap.error("--programs must be at least 1")
    if a.spirv_val:
        missing = [t for t in ("spirv-as", "spirv-val") if not shutil.which(t)]
        if missing:
            print(f"--spirv-val needs {' and '.join(missing)} on PATH")
            return 2
        global SPIRV_VAL
        SPIRV_VAL = True
    if a.self_test:
        return 0 if self_test() else 1
    waves = (32, 64) if a.wave == "both" else (int(a.wave),)
    structured = not a.dispatch_loop
    if a.program is not None:
        rc = 0
        matched = 0
        for wave in waves:
            body = program_for(a.seed, a.program, wave, a.depth)
            status, detail = run_case(f"ps_cf_{a.program:04d}_w{wave}", body, wave, structured)
            print(f"program {a.program} wave{wave}: {status} {detail}")
            matched += status == "match"
            if status == "mismatch":
                rc = 1
                if a.minimize:
                    body = minimize(body, wave, structured)
                    print("reduced:", run_case("ps_cf_min", body, wave, structured)[1])
            if a.show or a.minimize:
                print(listing_text(build(body, wave), f"ps_cf_{a.program:04d}_w{wave}"))
        if not rc and not matched:   # same order as the suite: a mismatch (1) outranks "nothing was compared" (2)
            print("no program ran: every one was skipped")
            return 2
        return rc
    counts, failures = run_suite(a.programs, a.seed, waves, structured, a.depth)
    skips = sum(counts["skip"].values())
    form = "dispatch loop" if a.dispatch_loop else "structured"
    print(f"{a.programs} programs x wave {'/'.join(map(str, waves))} (seed {a.seed}, {form}): "
          f"{counts['pass']} pass, {counts['mismatch']} mismatch, {skips} skipped, "
          f"{counts['harness-error']} harness errors")
    for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
        print(f"  skip x{n}: {reason}")
    for i, wave, detail in failures[:10]:
        print(f"  program {i} wave{wave}: {detail}")
    if len(failures) > 10:
        print(f"  ... {len(failures) - 10} more")
    if counts["harness-error"]:
        return 2
    if counts["mismatch"]:
        return 1
    if not counts["pass"]:
        print("no program ran: every one was skipped")
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
