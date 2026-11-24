r"""Differential fuzzing of packed and 16-bit integer VALU (v_pk_* integer ops, 16-bit VOP3 integer ops). Standard
library only.

Generates seeded straight-line programs of 16-bit integer vector instructions, one lane, runs them through

  * the reference below, written from the RDNA2 ISA guide's pseudocode for each opcode, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py (lane 0 of a wave32),

and compares four exported words bit for bit (v8..v11). The reference does not share code with the translator, with
tools/cpu_ref.py or with the other fuzzers. tools/valu_fuzz.py --family f16 runs some of the same opcodes on the GPU
with VGPR sources; this one adds the packed add/sub/mul/min/max/mad, the signed add/sub, min3/max3/med3 and mad,
clamp, and SGPR, inline-constant and literal sources in every position the encoding allows.

What is modelled, per instruction:
  * v_pk_* (VOP3P, OPS_PK): each result half is the op on the halves its sources select: op_sel bit i picks the
    half of source i for the low result, op_sel_hi bit i for the high result (default op_sel:[0,..],
    op_sel_hi:[1,..]). clamp saturates the exact result of add/sub/mad to [0, 0xffff] or [-32768, 32767];
    without it the result wraps to 16 bits. Shift amounts are their low 4 bits.
  * 16-bit VOP3 ops (OPS_16): op_sel bits 0..n-1 pick source halves, bit n (the last) the destination half. The
    result is written to that half and the other half of vD is kept. clamp as above (v_add_nc_*16, v_sub_nc_*16,
    v_mad_*16). On GFX10 these are VOP3 only: the VOP2 forms of v_mul_lo_u16, v_min/max_u16/i16 and the 16-bit
    shifts do not exist (the assembler refuses v_mul_lo_u16_e32). LLVM prints their VOP3 form without a suffix
    and the translator tells it from a VOP2 write (which zeroes the high half) by the listing's encoding bytes.
    A program here has no encodings, so these are written with _e64.
  * sources: VGPRs v0..v7 and v12..v19, SGPRs s4..s19, integer inline constants -16..64, float inline constants
    (0.5, 1.0, 2.0, 4.0, their negatives, 1/(2*pi)) and literals, at most one literal and two scalar values per
    instruction (the GFX10 constant bus). A constant is its 32-bit value split into halves like a register: -16 is
    0xfff0 / 0xffff, 1.0 is 0x0000 / 0x3f80 (its f32 bits, not the f16 constant of the f16 ops), a literal of a
    16-bit VOP3 op has 16 bits (the assembler rejects a wider one), so its high half is 0.
Every rule was checked against an RDNA2 GPU (AMD Radeon iGPU, gfx1036, Windows driver 26.10.1) with --oracle hw.

Not generated: neg_lo/neg_hi on the integer packed ops (the translator rejects them, their meaning on integers is
unmeasured), clamp on min/max/mul_lo/shifts, DPP and SDWA forms (tools/sdwa_diff.py), v_mad_u32_u16 and the other
16x16->32 ops (tools/vint_diff.py), v_dot* (#2229), f16 ops (tools/valu_fuzz.py --family f16).

Known divergences (KNOWN_DIVERGENCES): by default the generator leaves out the forms where the translator is known to
differ from the hardware, so that a run shows new problems only; --known puts them back (a hardware-only run always
has them):
  * pk-const-hi: a constant (inline or literal) in a packed add/sub/mul_lo/min/max/mad source whose high half
    op_sel or op_sel_hi reads. The translator raises Unsupported (counted as a skip); #2224 removes that.

With --oracle hw the same programs also run natively on an AMD GPU (tools/hw_oracle/), one lane of a wave32 compute
dispatch, and the four words are compared with the reference: that validates the reference itself. Without the
oracle tools the run exits 77.

Usage:
    python -I tools/pk16_diff.py [--programs N] [--seed S] [--max-ops N] [--ops v_pk_mad_u16,v_med3_i16]
    python -I tools/pk16_diff.py --no-literals          # sources without literals
    python -I tools/pk16_diff.py --known                # also the forms in KNOWN_DIVERGENCES
    python -I tools/pk16_diff.py --oracle hw            # hardware vs reference only
    python -I tools/pk16_diff.py --oracle translator,hw
    python -I tools/pk16_diff.py --list-ops
    python -I tools/pk16_diff.py --self-test

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

M16 = 0xFFFF
M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

VREGS = tuple(range(0, 8)) + tuple(range(12, 20))   # data VGPRs; v8..v11 are the observation
SREGS = tuple(range(4, 20))                          # data SGPRs
N_SEED = len(VREGS) + len(SREGS)
N_OBS = 4
INLINE_F = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000, "2.0": 0x40000000,
            "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000, "0.15915494": 0x3E22F983}
HALVES = (0, 1, 2, 15, 16, 17, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0xFFFE, 0x00FF, 0x0100, 0x5555, 0xAAAA, 0x1234,
          0xFF00, 0x4000, 0xC000, 0x00F0)


class Skip(Exception):
    pass


class RefError(Exception):
    pass


def sx16(x):
    x &= M16
    return x - 0x10000 if x & 0x8000 else x


def satu(v):
    return max(0, min(M16, v))


def sati(v):
    return max(-32768, min(32767, v))


def med3(a, b, c):
    return sorted((a, b, c))[1]


# name -> (nsrc, fn on 16-bit source halves -> exact integer, clamp fn or None). The result keeps its low 16 bits.
OPS_PK = {
    "v_pk_add_u16": (2, lambda a, b: a + b, satu),
    "v_pk_add_i16": (2, lambda a, b: sx16(a) + sx16(b), sati),
    "v_pk_sub_u16": (2, lambda a, b: a - b, satu),
    "v_pk_sub_i16": (2, lambda a, b: sx16(a) - sx16(b), sati),
    "v_pk_mul_lo_u16": (2, lambda a, b: a * b, None),
    "v_pk_min_u16": (2, min, None),
    "v_pk_max_u16": (2, max, None),
    "v_pk_min_i16": (2, lambda a, b: min(sx16(a), sx16(b)), None),
    "v_pk_max_i16": (2, lambda a, b: max(sx16(a), sx16(b)), None),
    "v_pk_lshlrev_b16": (2, lambda a, b: b << (a & 15), None),
    "v_pk_lshrrev_b16": (2, lambda a, b: b >> (a & 15), None),
    "v_pk_ashrrev_i16": (2, lambda a, b: sx16(b) >> (a & 15), None),
    "v_pk_mad_u16": (3, lambda a, b, c: a * b + c, satu),
    "v_pk_mad_i16": (3, lambda a, b, c: sx16(a) * sx16(b) + sx16(c), sati),
}
PK_SHIFTS = ("v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16")
OPS_16 = {
    "v_add_nc_u16": (2, lambda a, b: a + b, satu),
    "v_sub_nc_u16": (2, lambda a, b: a - b, satu),
    "v_add_nc_i16": (2, lambda a, b: sx16(a) + sx16(b), sati),
    "v_sub_nc_i16": (2, lambda a, b: sx16(a) - sx16(b), sati),
    "v_mul_lo_u16": (2, lambda a, b: a * b, None),
    "v_min_u16": (2, min, None),
    "v_max_u16": (2, max, None),
    "v_min_i16": (2, lambda a, b: min(sx16(a), sx16(b)), None),
    "v_max_i16": (2, lambda a, b: max(sx16(a), sx16(b)), None),
    "v_lshlrev_b16": (2, lambda a, b: b << (a & 15), None),
    "v_lshrrev_b16": (2, lambda a, b: b >> (a & 15), None),
    "v_ashrrev_i16": (2, lambda a, b: sx16(b) >> (a & 15), None),
    "v_min3_u16": (3, min, None),
    "v_max3_u16": (3, max, None),
    "v_med3_u16": (3, med3, None),
    "v_min3_i16": (3, lambda a, b, c: min(sx16(a), sx16(b), sx16(c)), None),
    "v_max3_i16": (3, lambda a, b, c: max(sx16(a), sx16(b), sx16(c)), None),
    "v_med3_i16": (3, lambda a, b, c: med3(sx16(a), sx16(b), sx16(c)), None),
    "v_mad_u16": (3, lambda a, b, c: a * b + c, satu),
    "v_mad_i16": (3, lambda a, b, c: sx16(a) * sx16(b) + sx16(c), sati),
}
# GFX10 has only the VOP3 encoding of these (VOP2 names on earlier chips). LLVM prints them without a suffix and the
# translator then tells VOP3 from VOP2 by the listing's encoding bytes; a program here has none, so it says _e64.
E64 = ("v_mul_lo_u16", "v_min_u16", "v_max_u16", "v_min_i16", "v_max_i16", "v_lshlrev_b16", "v_lshrrev_b16",
       "v_ashrrev_i16")
ALL_OPS = sorted(OPS_PK) + sorted(OPS_16)

KNOWN = False   # also generate the forms in KNOWN_DIVERGENCES (--known; always on for a hardware-only run)
LITERALS = True
KNOWN_DIVERGENCES = {
    "pk-const-hi": "a constant in a packed add/sub/mul_lo/min/max/mad source whose high half op_sel or op_sel_hi "
                   "reads: the translator raises Unsupported, #2224 reads the constant's 32-bit value",
}


# ---- program generation -------------------------------------------------------------------------------------

def rand_value(rng):
    r = rng.random()
    if r < 0.5:
        return rng.choice(HALVES) | (rng.choice(HALVES) << 16)
    if r < 0.6:
        return rng.randrange(0, 64)
    return rng.getrandbits(32)


def is_const(tok):
    return not re.fullmatch(r"[vs]\d+", tok)


class Gen:
    def __init__(self, rng):
        self.r = rng

    def vreg(self):
        return f"v{self.r.choice(VREGS)}"

    def const(self, lit_ok, lit16):
        r = self.r
        k = r.random()
        if k < 0.25:
            return r.choice(tuple(INLINE_F))
        if k < 0.6 or not lit_ok:
            return str(r.randint(-16, 64))
        v = rand_value(r) & (M16 if lit16 else M32)
        if v <= 64:
            v = 0x1234 if lit16 else 0x12345678
        return f"0x{v:x}"

    def srcs(self, n, lit16, const_ok):
        """n sources: at most one literal and two scalar values (the literal counted among them). const_ok[i]: a
        constant may go in source i."""
        r = self.r
        out, bus, lit = [], set(), False
        for i in range(n):
            k = r.random()
            if k < 0.5:
                out.append(self.vreg())
            elif k < 0.75 or not const_ok[i]:
                s = f"s{r.choice(SREGS)}"
                if s not in bus and len(bus) >= 2:
                    out.append(self.vreg())
                    continue
                bus.add(s)
                out.append(s)
            else:
                c = self.const(LITERALS and not lit and len(bus) < 2, lit16)
                if c.startswith("0x"):
                    lit = True
                    bus.add("lit")
                out.append(c)
        return out

    def op(self, ops):
        r = self.r
        name = r.choice(ops)
        d = r.choice(VREGS)
        if name in OPS_PK:
            n, _, cl = OPS_PK[name]
            sel = [int(r.random() < 0.3) for _ in range(n)]
            hi = [int(r.random() < 0.7) for _ in range(n)]
            free = KNOWN or name in PK_SHIFTS
            srcs = self.srcs(n, False, [free or not (sel[i] or hi[i]) for i in range(n)])
            line = f"{name} v{d}, {', '.join(srcs)}"
            if any(sel):
                line += f" op_sel:[{','.join(map(str, sel))}]"
            if not all(hi):
                line += f" op_sel_hi:[{','.join(map(str, hi))}]"
            if cl and r.random() < 0.3:
                line += " clamp"
            return [line]
        n, _, cl = OPS_16[name]
        line = f"{name}{'_e64' if name in E64 else ''} v{d}, {', '.join(self.srcs(n, True, [True] * n))}"
        sel = [int(r.random() < 0.3) for _ in range(n + 1)]
        if any(sel):
            line += f" op_sel:[{','.join(map(str, sel))}]"
        if cl and r.random() < 0.3:
            line += " clamp"
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
        dst = int(line.split()[1].rstrip(",")[1:])
        if dst not in written:
            written.append(dst)
    regs = written[:4]
    regs += [rng.choice(VREGS) for _ in range(4 - len(regs))]
    return seeds(rng) + body + observe(regs)


# ---- the reference interpreter --------------------------------------------------------------------------------

_MOD = re.compile(r"\s+(op_sel|op_sel_hi):\[([01,]+)\]|\s+(clamp)\b")


def parse(rest):
    """Operands and modifiers of one instruction."""
    mods = {}
    for m in _MOD.finditer(rest):
        if m.group(3):
            mods["clamp"] = True
        else:
            mods[m.group(1)] = [int(x) for x in m.group(2).split(",")]
    rest = _MOD.sub("", rest)
    return [a.strip() for a in rest.split(",")], mods


def value(tok, st):
    if re.fullmatch(r"v\d+", tok):
        return st["v"].get(int(tok[1:]), 0)
    if re.fullmatch(r"s\d+", tok):
        if tok not in st["s"]:
            raise RefError(f"{tok} read before it is written")
        return st["s"][tok]
    if tok in INLINE_F:
        return INLINE_F[tok]
    return int(tok, 0) & M32


def half(v, hi):
    return (v >> 16) & M16 if hi else v & M16


def reference(lines):
    """Run the program; returns the four exported words."""
    st = {"v": {}, "s": {}}
    for line in lines:
        op, _, rest = line.partition(" ")
        args, mods = parse(rest)
        if op == "s_mov_b32":
            st["s"][args[0]] = value(args[1], st)
            continue
        d = int(args[0][1:])
        vals = [value(a, st) for a in args[1:]]
        if op == "v_mov_b32_e32":
            st["v"][d] = vals[0]
            continue
        if op in OPS_PK:
            n, fn, cl = OPS_PK[op]
            out = 0
            for k, sel in enumerate((mods.get("op_sel", [0] * n), mods.get("op_sel_hi", [1] * n))):
                r = fn(*[half(v, sel[i]) for i, v in enumerate(vals)])
                if mods.get("clamp") and cl:
                    r = cl(r)
                out |= (r & M16) << (16 * k)
            st["v"][d] = out
            continue
        op = op[:-4] if op.endswith("_e64") and op[:-4] in E64 else op
        if op not in OPS_16:
            raise RefError(f"{op} is not modelled")
        n, fn, cl = OPS_16[op]
        sel = mods.get("op_sel", [0] * (n + 1))
        r = fn(*[half(v, sel[i]) for i, v in enumerate(vals)])
        if mods.get("clamp") and cl:
            r = cl(r)
        old = st["v"].get(d, 0)
        st["v"][d] = ((old & M16) | ((r & M16) << 16)) if sel[n] else ((old & 0xFFFF0000) | (r & M16))
    return [st["v"].get(8 + i, 0) for i in range(4)]


# ---- the translator side -----------------------------------------------------------------------------------

def translated(lines, name="ps_pk16"):
    """Translate and execute; returns the four exported words, or raises Skip."""
    import rdna2spirv
    from spirv_mini_interp import run_fragment
    with tempfile.TemporaryDirectory(prefix="pk16-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  pk16_diff case\n")
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
    """Observe one register if that still fails, drop instructions while the program still mismatches, then the
    seeds of registers nothing reads."""
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
        for i in range(len(body) - 1, -1, -1):
            trial = body[:i] + body[i + 1:]
            if fails(seed + trial + obs):
                body, changed = trial, True
    used = set()
    for line in body:
        used |= set(re.findall(r"\b[vs]\d+\b", line))      # a 16-bit write keeps the other half of vD: it reads vD
    for line in obs:
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
    base = [f"v_mov_b32_e32 v{v}, 0x{0x00010001 * v:x}" for v in VREGS] + [f"s_mov_b32 s{s}, {s}" for s in SREGS]
    body = ["v_pk_add_u16 v1, v2, s4 op_sel:[1,0] op_sel_hi:[0,0]",    # lo: 2 + 4, hi: 2 + 4
            "v_pk_mad_i16 v3, v3, -1, v5 op_sel_hi:[1,0,1] clamp",     # -1 in both halves: 5 - 3
            "v_mad_u16 v4, v6, 0x8000, v7 op_sel:[0,0,0,1] clamp",     # 6 * 0x8000 + 7 saturates; high half
            "v_max3_i16 v2, v2, -16, s5 op_sel:[0,0,0,0]"]
    lines = base + body + observe([1, 3, 4, 2])
    ok = check(lines) is None
    worked = reference(lines) == [0x00060006, 0x00020002, 0xFFFF0004, 0x00020005]
    try:
        translated(base + ["v_pk_add_u16_sdwa_dpp v1, v2, v3"] + observe([1, 2, 3, 4]))
        skipped = False
    except Skip:
        skipped = True
    print(f"self-test: match {ok}, hand values {worked}, unsupported form skipped {skipped}")
    return ok and worked and skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="Packed and 16-bit integer VALU differential fuzzing.")
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=6)
    ap.add_argument("--ops", default=None, help="comma-separated opcodes (--list-ops)")
    ap.add_argument("--no-literals", action="store_true", help="no literal sources")
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
            if name in OPS_PK:
                n, _, cl = OPS_PK[name]
                print(f"{name:18} VOP3P, {n} sources, op_sel/op_sel_hi{', clamp' if cl else ''}")
            else:
                n, _, cl = OPS_16[name]
                print(f"{name:18} VOP3, {n} sources, op_sel (sources and destination half){', clamp' if cl else ''}")
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
