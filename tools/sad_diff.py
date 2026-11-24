r"""Differential fuzzer for the RDNA2 sum-of-absolute-differences family (standard library only).

Covers seven opcodes, one family (the "Sum of absolute differences" and `v_qsad` rows of
docs/design.md's contract table):

    v_sad_u8  v_sad_hi_u8  v_sad_u16  v_sad_u32  v_msad_u8     32-bit results
    v_qsad_pk_u16_u8  v_mqsad_pk_u16_u8                    packed four-lane u16 results

Seeded straight-line programs run through two independent oracles, and the four exported mrt0
words are compared bit for bit (everything here is integer, so a mismatch is never a rounding
convention):

  * the reference below, written from the RDNA2 ISA pseudocode as recorded in docs/design.md
    (hardware-verified there: the SAD row on gfx1035 via SHADER-137's int_hw_probe run,
    v_qsad/v_mqsad bit-exact on an RTX 3060 in #623), and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py.

Program shape: seeds in v8..v15 and s4..s7, then the family ops, then `exp mrt0 v8, v9, v10, v11`
so every result is observed. Every op's destination is v8..v11 (quad destinations are pairs inside
that), so later ops can chain on earlier results.

Source kinds, in every operand position the encoding allows:
  VGPR v8..v15, SGPR s4..s7, a hex literal (32 bits), an inline constant (decimal, 0..64 or
  -16..-1), and `null` (reads 0) for the 32-bit sources. The quad ops' S0 and S2 are 64-bit: VGPR
  or SGPR pairs, a hex literal (zero high word) or an inline constant (sign-extended), the rule the
  contract table records for 64-bit constant operands.

Two modes run one after the other:
  * a sweep over the whole source-kind space: every (S0, S1, S2) kind combination of every op,
    125 per plain op and 80 per quad op, 785 programs, with values drawn from byte/half edge words;
  * random programs (--programs), which widen the value coverage over the same kind space.

Not generated, on purpose: `clamp` (the contract rows say the translator rejects it; --self-test
pins that as an expected rejection) and op_sel / neg / abs (the ISA does not define them for these
opcodes). v_mqsad_u32_u8 is left out: open PRs #1926/#2163 are on that op.

Usage:
    python -I tools/sad_diff.py [--programs N] [--seed S] [--max-ops N] [--no-sweep]
    python -I tools/sad_diff.py --ops v_sad_u8,v_msad_u8
    python -I tools/sad_diff.py --list-ops
    python -I tools/sad_diff.py --self-test

Exit code 0 when every program matches, 1 on a mismatch (the first one is printed with its
listing), 2 on a harness error. Skips are counted with reasons and never silent.
Nothing here reads game files.
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
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or HERE
sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import rdna2spirv  # noqa: E402
from spirv_mini_interp import UnsupportedConstruct, run_fragment  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

EXPORT = "exp mrt0 v8, v9, v10, v11 done vm"
EXPORT_REGS = (8, 9, 10, 11)

# Byte/half boundary words: the constants a SAD lane can sit on (0x00/0xff bytes, half carries,
# bit 31 set for the literal-vs-inline 64-bit rule).
EDGE = (0x00000000, 0x00000001, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x01010101, 0xFEFEFEFE,
        0x80808080, 0x7F7F7F7F, 0xFF00FF00, 0x00FF00FF, 0xFFFF0000, 0x0000FFFF, 0x00010001,
        0xFFFEFFFE, 0x55555555, 0xAAAAAAAA, 0x01234567, 0x89ABCDEF, 0x000000FF, 0x0000FF00)
INLINE = tuple(range(0, 65)) + tuple(range(-16, 0))

PLAIN_OPS = ("v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8")
QUAD_OPS = ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8")
OP_FORMS = {
    "v_sad_u8": "v_sad_u8 d, s0, s1, s2",
    "v_sad_hi_u8": "v_sad_hi_u8 d, s0, s1, s2",
    "v_sad_u16": "v_sad_u16 d, s0, s1, s2",
    "v_sad_u32": "v_sad_u32 d, s0, s1, s2",
    "v_msad_u8": "v_msad_u8 d, s0, s1, s2",
    "v_qsad_pk_u16_u8": "v_qsad_pk_u16_u8 d[2], s0[2], s1, s2[2]",
    "v_mqsad_pk_u16_u8": "v_mqsad_pk_u16_u8 d[2], s0[2], s1, s2[2]",
}
PLAIN_KINDS = ("vgpr", "sgpr", "lit", "inline", "null")          # every 32-bit source position
QUAD64_KINDS = ("vgpr", "sgpr", "lit", "inline")                 # the 64-bit S0/S2 positions

# ---- the reference, from the RDNA2 ISA pseudocode ----

def _absdiff(x, y):
    return x - y if x >= y else y - x


def _lanes(word, width):
    return [(word >> off) & ((1 << width) - 1) for off in range(0, 32, width)]


def sad_u8(x, y, masked=False):
    """Sum over the four bytes of |x.b - y.b|; the masked form skips a byte where y's byte is 0."""
    return sum(_absdiff(p, q) for p, q in zip(_lanes(x, 8), _lanes(y, 8)) if not (masked and q == 0))


def sad_u16(x, y):
    return sum(_absdiff(p, q) for p, q in zip(_lanes(x, 16), _lanes(y, 16)))


def quad_sad(s0_lo, s0_hi, s1, acc_lo, acc_hi, masked=False):
    """v_qsad_pk_u16_u8 / v_mqsad_pk_u16_u8 as one 64-bit result: for k = 0..3,
    D.u16[k] = (S2.u16[k] + byte SAD of the 32-bit window S0[8k+31 : 8k] against S1) & 0xFFFF."""
    accs = _lanes(acc_lo, 16) + _lanes(acc_hi, 16)
    out = 0
    for k in range(4):
        if k == 0:
            win = s0_lo & M32
        else:
            win = ((s0_hi << (32 - 8 * k)) | (s0_lo >> (8 * k))) & M32
        total = sad_u8(win, s1 & M32, masked=masked)
        out |= ((accs[k] + total) & 0xFFFF) << (16 * k)
    return out & M64


SINGLE_OPS = {
    "v_sad_u8": lambda x, y, z: (sad_u8(x, y) + z) & M32,
    "v_sad_hi_u8": lambda x, y, z: ((sad_u8(x, y) << 16) + z) & M32,
    "v_sad_u16": lambda x, y, z: (sad_u16(x, y) + z) & M32,
    "v_sad_u32": lambda x, y, z: (_absdiff(x & M32, y & M32) + z) & M32,
    "v_msad_u8": lambda x, y, z: (sad_u8(x, y, masked=True) + z) & M32,
}

_PAIR = re.compile(r"([sv])\[(\d+):(\d+)\]$")
_REG = re.compile(r"([sv])(\d+)$")


class HarnessError(Exception):
    pass


def _single_word(tok, vgpr, sgpr):
    """The 32-bit word a single-source token reads: a register, a hex literal, an inline constant
    (two's complement at 32 bits) or `null`."""
    tok = tok.strip()
    m = _REG.fullmatch(tok)
    if m:
        regs = vgpr if m.group(1) == "v" else sgpr
        return regs.get(int(m.group(2)), 0)
    if tok == "null":
        return 0
    if tok.lower().startswith("0x"):
        return int(tok, 16) & M32
    if tok.lstrip("+-").isdigit():
        return int(tok, 10) & M32
    raise HarnessError(f"unknown single source {tok!r}")


def _pair_words(tok, vgpr, sgpr):
    """The two dwords a 64-bit source token reads: a register pair, a hex literal (zero high word)
    or an inline constant (sign-extended)."""
    tok = tok.strip()
    m = _PAIR.fullmatch(tok)
    if m:
        regs = vgpr if m.group(1) == "v" else sgpr
        lo, hi = int(m.group(2)), int(m.group(3))
        if hi != lo + 1:
            raise HarnessError(f"not an even pair {tok!r}")
        return regs.get(lo, 0), regs.get(hi, 0)
    if tok.lower().startswith("0x"):
        return int(tok, 16) & M32, 0
    if tok.lstrip("+-").isdigit():
        v = int(tok, 10) & M64
        return v & M32, (v >> 32) & M32
    raise HarnessError(f"unknown pair source {tok!r}")


def _dst_index(dst):
    m = _REG.fullmatch(dst.strip())
    if not m or m.group(1) != "v":
        raise HarnessError(f"unknown destination {dst!r}")
    return int(m.group(2))


def step(line, vgpr, sgpr):
    """Execute one generated listing line on the reference's registers."""
    head, *args = line.split(",")
    parts = head.split()
    if len(parts) != 2:
        raise HarnessError(line)
    op, dst = parts
    args = [a.strip() for a in args]
    if op == "v_mov_b32":
        vgpr[_dst_index(dst)] = _single_word(args[0], vgpr, sgpr)
    elif op == "s_mov_b32":
        if not dst.startswith("s"):
            raise HarnessError(line)
        sgpr[int(dst[1:])] = _single_word(args[0], vgpr, sgpr)
    elif op in SINGLE_OPS:
        x, y, z = (_single_word(a, vgpr, sgpr) for a in args)
        vgpr[_dst_index(dst)] = SINGLE_OPS[op](x, y, z)
    elif op in QUAD_OPS:
        m = _PAIR.fullmatch(dst.strip())
        if not m or m.group(1) != "v" or int(m.group(3)) != int(m.group(2)) + 1:
            raise HarnessError(f"quad destination {dst!r}")
        d = int(m.group(2))
        (x_lo, x_hi), y, (c_lo, c_hi) = (_pair_words(args[0], vgpr, sgpr),
                                         _single_word(args[1], vgpr, sgpr),
                                         _pair_words(args[2], vgpr, sgpr))
        out = quad_sad(x_lo, x_hi, y, c_lo, c_hi, masked=(op == "v_mqsad_pk_u16_u8"))
        vgpr[d] = out & M32
        vgpr[d + 1] = (out >> 32) & M32
    else:
        raise HarnessError(line)


def reference(lines):
    """The four exported mrt0 words the ISA pseudocode gives for a program (seeds and ops; the exp
    and s_endpgm lines are ignored)."""
    vgpr, sgpr = {}, {}
    for line in lines:
        first = line.split()[0]
        if first in ("exp", "s_endpgm"):
            continue
        step(line, vgpr, sgpr)
    return [vgpr.get(r, 0) for r in EXPORT_REGS]


# ---- program generation ----

def _seed_word(rng):
    return rng.choice(EDGE) if rng.random() < 0.6 else rng.getrandbits(32)


def _single_source(rng, kind):
    if kind == "vgpr":
        return f"v{rng.randrange(8, 16)}"
    if kind == "sgpr":
        return f"s{rng.randrange(4, 8)}"
    if kind == "lit":
        return f"0x{_seed_word(rng):08x}"
    if kind == "inline":
        return str(rng.choice(INLINE))
    return "null"


def _pair_source(rng, kind):
    if kind == "vgpr":
        a = rng.randrange(8, 15)
        return f"v[{a}:{a + 1}]"
    if kind == "sgpr":
        c = rng.randrange(4, 7)
        return f"s[{c}:{c + 1}]"
    if kind == "lit":
        return f"0x{_seed_word(rng):08x}"
    return str(rng.choice(INLINE))


def program(rng, n_ops, ops=None):
    """One straight-line program: (lines, ops_used). Ops drawn from `ops` (default: the family)."""
    ops = list(ops) if ops else list(PLAIN_OPS) + list(QUAD_OPS)
    lines = []
    used = Counter()
    for r in range(8, 16):
        lines.append(f"v_mov_b32 v{r}, 0x{_seed_word(rng):08x}")
    for r in range(4, 8):
        lines.append(f"s_mov_b32 s{r}, 0x{_seed_word(rng):08x}")
    for _ in range(n_ops):
        op = rng.choice(ops)
        used[op] += 1
        if op in PLAIN_OPS:
            dst = f"v{rng.choice(EXPORT_REGS)}"
            srcs = ", ".join(_single_source(rng, rng.choice(PLAIN_KINDS)) for _ in range(3))
        else:
            dpair = rng.choice((8, 9, 10))
            dst = f"v[{dpair}:{dpair + 1}]"
            srcs = ", ".join((_pair_source(rng, rng.choice(QUAD64_KINDS)),
                              _single_source(rng, rng.choice(PLAIN_KINDS)),
                              _pair_source(rng, rng.choice(QUAD64_KINDS))))
        lines.append(f"{op} {dst}, {srcs}")
    return lines, used


def kind_counts(lines):
    """Per-opcode source-kind instances for the coverage report: ("v_sad_u8", "vgpr") -> n."""
    out = Counter()
    for line in lines:
        first = line.split()[0]
        if first not in OP_FORMS:
            continue
        head, *args = line.split(",")
        dst = head.split()[1].strip()
        out[(first, "vpair" if ":" in dst else "vgpr")] += 1
        if first in PLAIN_OPS:
            for a in args:
                out[(first, _kind_of(a))] += 1
        else:
            out[(first, _pair_kind_of(args[0]))] += 1
            out[(first, _kind_of(args[1]))] += 1
            out[(first, _pair_kind_of(args[2]))] += 1
    return out


def _kind_of(tok):
    tok = tok.strip()
    if tok == "null":
        return "null"
    if tok.lower().startswith("0x"):
        return "lit"
    if tok.lstrip("+-").isdigit():
        return "inline"
    return "vgpr" if tok.startswith("v") else "sgpr"


def _pair_kind_of(tok):
    tok = tok.strip()
    if tok.lower().startswith("0x"):
        return "lit"
    if tok.lstrip("+-").isdigit():
        return "inline"
    return "vpair" if tok.startswith("v") else "spair"


def sweep_ops(ops=None):
    """Every op with every (S0, S1, S2) source-kind combination its encoding allows: 125 per plain
    op, 80 per quad op. Returns a list of (label, kinds, value-rng seed)."""
    ops = list(ops) if ops else list(PLAIN_OPS) + list(QUAD_OPS)
    cases = []
    for op in ops:
        kinds64 = QUAD64_KINDS if op in QUAD_OPS else PLAIN_KINDS
        kinds32 = PLAIN_KINDS
        if op in QUAD_OPS:
            combos = ((k0, k1, k2) for k0 in kinds64 for k1 in kinds32 for k2 in kinds64)
        else:
            combos = ((k0, k1, k2) for k0 in kinds32 for k1 in kinds32 for k2 in kinds32)
        for combo in combos:
            cases.append((op, combo))
    return cases


def sweep_program(op, kinds, rng):
    """One sweep program: seeds, then the op once, with the fixed kind combination."""
    lines = [f"v_mov_b32 v{r}, 0x{_seed_word(rng):08x}" for r in range(8, 16)]
    lines += [f"s_mov_b32 s{r}, 0x{_seed_word(rng):08x}" for r in range(4, 8)]
    if op in PLAIN_OPS:
        srcs = ", ".join(_single_source(rng, k) for k in kinds)
        lines.append(f"{op} v{rng.choice(EXPORT_REGS)}, {srcs}")
    else:
        dpair = rng.choice((8, 9, 10))
        srcs = ", ".join((_pair_source(rng, kinds[0]), _single_source(rng, kinds[1]),
                          _pair_source(rng, kinds[2])))
        lines.append(f"{op} v[{dpair}:{dpair + 1}], {srcs}")
    return lines


# ---- the translator oracle ----

def write_listing(path, lines, name):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}  code 0x0..0x1000\n")
        for line in lines:
            f.write("\t" + line + "\n")
        f.write("\t" + EXPORT + "\n")
        f.write("\ts_endpgm\n")


def translate_listing(lines, name, tmpdir):
    """The four mrt0 words the translator path produces. Raises rdna2spirv.Unsupported (and mini
    interp's UnsupportedConstruct) for anything outside the oracles' model."""
    dis = os.path.join(tmpdir, "disasm")
    os.makedirs(dis, exist_ok=True)
    path = os.path.join(dis, name + ".s")
    write_listing(path, lines, name)
    with open(os.path.join(tmpdir, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spvasm = rdna2spirv.translate(path)
    gc.collect()
    out = run_fragment(spvasm)
    if list(out) != ["%out_mrt0"]:
        raise HarnessError(f"outputs {sorted(out)}, want %out_mrt0")
    return [w & M32 for w in out["%out_mrt0"]]


def check(lines, name="ps_sad_case"):
    """None when both oracles agree, else the first differing word with both values."""
    with tempfile.TemporaryDirectory(prefix="sad-diff-") as tmp:
        want = reference(lines)
        got = translate_listing(lines, name, tmp)
    for i, (w, g) in enumerate(zip(want, got)):
        if w != (g & M32):
            return f"word {i} (v{EXPORT_REGS[i]}): expected 0x{w:08x}, actual 0x{g:08x}"
    return None


def run_one(lines, name, tmpdir):
    """(status, detail) with status in {"match", "mismatch", "skip"}."""
    try:
        want = reference(lines)
    except HarnessError as e:
        return ("skip", f"reference cannot run the program: {e}")
    try:
        got = translate_listing(lines, name, tmpdir)
    except (rdna2spirv.Unsupported, UnsupportedConstruct) as e:
        return ("skip", f"{type(e).__name__}: {e}")
    except (IndexError, KeyError, ValueError) as e:
        return ("skip", f"translator error {type(e).__name__}: {e}")
    for i, (w, g) in enumerate(zip(want, got)):
        if w != (g & M32):
            return ("mismatch", f"word {i} (v{EXPORT_REGS[i]}): expected 0x{w:08x}, "
                                f"actual 0x{g:08x}")
    return ("match", "")


# ---- the suites ----

def run_sweep(seed, ops=None):
    """Every source-kind combination of every op, one program each. Returns (counts, failure)."""
    counts = {"pass": 0, "mismatch": 0, "skip": Counter(), "ops": Counter(), "kinds": Counter()}
    failure = None
    cases = sweep_ops(ops)
    with tempfile.TemporaryDirectory(prefix="sad-diff-sweep-") as tmp:
        for i, (op, kinds) in enumerate(cases):
            rng = random.Random(f"{seed}:{op}:{'/'.join(kinds)}")
            lines = sweep_program(op, kinds, rng)
            counts["ops"][op] += 1
            for key, n in kind_counts(lines).items():
                counts["kinds"][key] += n
            status, detail = run_one(lines, f"ps_sad_sw{i:04d}", tmp)
            if status == "match":
                counts["pass"] += 1
            elif status == "skip":
                counts["skip"][detail] += 1
            else:
                counts["mismatch"] += 1
                if failure is None:
                    failure = (f"sweep {op} {'/'.join(kinds)} (case {i}, seed {seed})", lines, detail)
    return counts, failure


def run_suite(programs, seed, max_ops=6, ops=None):
    """Random seeded programs. Returns (counts, failure)."""
    counts = {"pass": 0, "mismatch": 0, "skip": Counter(), "ops": Counter(), "kinds": Counter()}
    failure = None
    rng = random.Random(f"suite:{seed}")
    with tempfile.TemporaryDirectory(prefix="sad-diff-run-") as tmp:
        for i in range(programs):
            lines, used = program(rng, rng.randint(1, max_ops), ops)
            counts["ops"].update(used)
            for key, n in kind_counts(lines).items():
                counts["kinds"][key] += n
            status, detail = run_one(lines, f"ps_sad_{i:04d}", tmp)
            if status == "match":
                counts["pass"] += 1
            elif status == "skip":
                counts["skip"][detail] += 1
            else:
                counts["mismatch"] += 1
                if failure is None:
                    failure = (f"program {i} (seed {seed})", lines, detail)
    return counts, failure


def self_test():
    """True when the reference's hand-pinned words, the negative controls and the harness canaries
    all hold."""
    ok = True

    def expect(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    # Hand-pinned reference words, each derived from the ISA pseudocode in the docstring.
    expect(SINGLE_OPS["v_sad_u8"](0x01020304, 0xFF00FF00, 5) == 517,
           "v_sad_u8(0x01020304, 0xFF00FF00) + 5 = 254+2+252+4+5 = 517")
    expect(SINGLE_OPS["v_sad_hi_u8"](0x01020304, 0xFF00FF00, 5) == 0x02000005,
           "v_sad_hi_u8: (byte sum << 16) + 5")
    expect(SINGLE_OPS["v_sad_u16"](0x0001FFFF, 0xFFFF0001, 0) == 0x0001FFFC,
           "v_sad_u16(0x0001FFFF, 0xFFFF0001) = 65534+65534")
    expect(SINGLE_OPS["v_sad_u32"](0, 0xFFFFFFFF, 1) == 0,
           "v_sad_u32(0, 0xFFFFFFFF) + 1 wraps to 0")
    expect(SINGLE_OPS["v_msad_u8"](0x01020304, 0xFF000000, 5) == 259,
           "v_msad_u8: only the byte where S1's byte is 0xFF counts: |01-FF|+5")
    expect(quad_sad(0x04030201, 0x08070605, 0x04030201, 0, 0) == 0x000C000800040000,
           "v_qsad_pk_u16_u8: the four sliding windows of 0x0807060504030201 give 0,4,8,12")
    expect(quad_sad(M32, M32, 0, 5, 0, masked=True) == 5,
           "v_mqsad_pk_u16_u8 with S1 = 0: everything masked, D = S2")
    # 64-bit constant rule: a hex literal has a zero high word, an inline constant sign-extends
    # (S1 = 0 makes every window's byte sum the lane value, so the words tell the high word apart).
    lit_lit = reference(["v_qsad_pk_u16_u8 v[8:9], 0xFFFF8000, 0, 0"])
    lit_pair = reference(["v_mov_b32 v8, 0xFFFF8000", "v_qsad_pk_u16_u8 v[8:9], v[8:9], 0, 0"])
    expect(lit_lit == [0x027E027E, 0x00FF01FE, 0, 0],
           "the literal 0xFFFF8000 as a 64-bit S0: lanes 638, 638, 510, 255 (zero high word)")
    expect(lit_lit == lit_pair,
           "that literal reads the same as the pair holding its word and 0")
    inline = reference(["v_qsad_pk_u16_u8 v[8:9], -16, 0, 0"])
    expect(inline == [0x03FC03ED, 0x03FC03FC, 0, 0],
           "the inline constant -16 as a 64-bit S0 sign-extends (lanes 1005, 1020, 1020, 1020)")
    expect(inline[1] != lit_lit[1],
           "the literal and the inline constant differ in the high word (the rule bites)")

    # The two-oracle path agrees on a fixed program, and a sabotaged word is caught.
    lines = ["v_mov_b32 v8, 0x01020304", "v_mov_b32 v9, 0xFF00FF00", "v_mov_b32 v10, 5",
             "v_sad_u8 v8, v8, v9, v10"]
    expect(check(lines) is None, "a hand-pinned program matches on both oracles")
    real = translate_listing
    try:
        globals()["translate_listing"] = lambda l, n, t: [516] + real(l, n, t)[1:]
        expect(check(lines) is not None, "a sabotaged export word is reported")
    finally:
        globals()["translate_listing"] = real

    # Negative controls: forms the contract rows say are rejected, not guessed.
    with tempfile.TemporaryDirectory(prefix="sad-diff-neg-") as tmp:
        for what, bad in (
            ("clamp on v_sad_u8", ["v_mov_b32 v8, 1", "v_sad_u8 v8, v8, v8, v8 clamp"]),
            ("clamp on v_qsad_pk_u16_u8", ["v_qsad_pk_u16_u8 v[8:9], v[9:10], v11, v[12:13] clamp"]),
            ("a quad op to a single VGPR", ["v_qsad_pk_u16_u8 v8, v[9:10], v11, v[12:13]"]),
            ("a quad op with a single-VGPR S0", ["v_qsad_pk_u16_u8 v[8:9], v9, v10, v[11:12]"]),
        ):
            try:
                translate_listing(bad, "ps_sad_neg", tmp)
                expect(False, f"{what}: the translator rejected it")
            except (rdna2spirv.Unsupported, UnsupportedConstruct):
                expect(True, f"{what}: the translator rejected it")

    # Determinism: the same seed gives the same program stream.
    expect(program(random.Random(9), 4) == program(random.Random(9), 4),
           "generation is deterministic for a seed")
    counts, failure = run_suite(6, 3, 3)
    expect(failure is None and counts["pass"] == 6 and not counts["skip"],
           f"a short seeded run matches ({counts['pass']} programs, skips {dict(counts['skip'])})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", type=int, default=300, help="random programs after the sweep")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=6, help="ops per random program")
    ap.add_argument("--ops", default=None, help="comma-separated opcodes to cover (default: all)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the source-kind sweep")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.list_ops:
        for op, form in OP_FORMS.items():
            print(f"{op:22s} {form}")
        return 0
    if args.self_test:
        return 0 if self_test() else 1

    ops = [o.strip() for o in args.ops.split(",")] if args.ops else None
    bad = [o for o in (ops or []) if o not in OP_FORMS]
    if bad:
        print(f"unknown op(s) {bad}; --list-ops shows the family", file=sys.stderr)
        return 2

    total_pass = total_run = total_mismatch = 0
    per_op = Counter()
    failure = None
    try:
        if not args.no_sweep:
            counts, first = run_sweep(args.seed, ops)
            total_pass += counts["pass"]
            total_mismatch += counts["mismatch"]
            total_run += counts["pass"] + counts["mismatch"] + sum(counts["skip"].values())
            per_op.update(counts["ops"])
            print(f"sweep: {counts['pass']}/{counts['pass'] + counts['mismatch'] + sum(counts['skip'].values())} "
                  f"source-kind programs match, {counts['mismatch']} mismatch; "
                  f"skips: {dict(counts['skip'])}")
            failure = failure or first
        counts, first = run_suite(args.programs, args.seed, args.max_ops, ops)
        total_pass += counts["pass"]
        total_mismatch += counts["mismatch"]
        total_run += counts["pass"] + counts["mismatch"] + sum(counts["skip"].values())
        per_op.update(counts["ops"])
        print(f"random: {counts['pass']}/{counts['pass'] + counts['mismatch'] + sum(counts['skip'].values())} "
              f"programs match, {counts['mismatch']} mismatch; skips: {dict(counts['skip'])}")
        failure = failure or first
    except HarnessError as e:
        print(f"harness error: {e}", file=sys.stderr)
        return 2
    print("cases per op: " + ", ".join(f"{op} {per_op[op]}" for op in sorted(per_op)))
    print(f"SAD family: {total_pass}/{total_run} programs match, {total_mismatch} mismatch "
          f"({len(per_op)} of {len(OP_FORMS)} ops)")
    if failure:
        _report(failure)
        return 1
    if not total_pass:   # nothing was compared: every program was skipped
        print("no program ran: every one was skipped")
        return 2
    return 0


def _report(failure):
    name, lines, detail = failure
    print(f"MISMATCH {name}: {detail}")
    print("program:")
    for line in lines:
        print("    " + line)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
