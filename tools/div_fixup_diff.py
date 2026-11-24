r"""Differential fuzzer for v_div_fixup_f16, v_div_fixup_f32 and v_div_fixup_f64 (standard library only).

No other fuzzer generates these three ops: they are on no `--list-ops` list and in no generator.

    v_div_fixup_f16  VOP3  d, s0, s1, s2  [op_sel:[a,b,c,d]] [clamp] [mul:2 | mul:4 | div:2]
    v_div_fixup_f32  VOP3  d, s0, s1, s2  [clamp] [mul:2 | mul:4 | div:2]
    v_div_fixup_f64  VOP3  d[2], s0[2], s1[2], s2[2]  [mul:2 | mul:4 | div:2]
                                                                      s0 quotient, s1 denominator, s2 numerator

Seeded straight-line programs run through two independent oracles, and the four exported mrt0 words are compared:

  * the reference below, written from the docs/design.md rows "v_div_fixup_f32" (measured on gfx1036) and
    "v_div_fixup_f16, v_div_fixup_f64" (the gfx1036 measurements in it). It works on the operand bits and shares no
    code with the translator:
      - abs clears and neg then flips the sign bit of a source, NaNs included;
      - f32 reads an S1 or S2 denormal as a zero of its sign (denorm32 = 0) and keeps an S0 denormal; f16 and f64
        keep every denormal (denorm16_64 = 3);
      - the first rule that holds gives the result, with sign = sign(S1) ^ sign(S2):
          1. NaN(S2) -> S2, unchanged (a signalling NaN is not quieted under IEEE = 0)
          2. NaN(S1) -> S1, unchanged
          3. 0/0 or inf/inf -> the negative quiet NaN (0xfe00, 0xffc00000, 0xfff8000000000000)
          4. a zero numerator or an inf denominator -> a zero with the sign
          5. a zero denominator or an inf numerator -> inf with the sign
          6. exp(S2) - exp(S1) <= -151 on f32 or <= -1076 on f64 (biased exponents) -> a zero with the sign;
             f16 has no such cutoff
          7. NaN(S0) -> inf with the sign
          8. S0's magnitude with the sign.
        The f16/f64 row lists rule 6 with rule 4 and NaN(S0) with rule 5; the order is the same, because rule 6 needs
        exp(S1) >= 1076 and a finite S2, so it never meets rule 5.
      - f32 output modifiers: mul:2, mul:4 or div:2, then clamp to [0, 1], where a NaN gives +0 (DX10_CLAMP = 1).
        f16 clamp: to [0, 1].
      - f16 and f64 output multipliers (mul:2, mul:4, div:2) change nothing: the hardware ignores them on a 16-bit or
        binary64 result while denorm_mode_16_64 = 3, the game's MODE (docs/design.md, measured on gfx1036 with
        tools/f64_omod_probe.py, #6314). An f16 clamp still applies.
      - an f16 result goes to the half op_sel's last bit names, and the other half is kept.
    tests/test_div_fixup_diff.py pins it to the gfx1036 words that tests/semantic records, and
  * tools/rdna2spirv.py, executing the SPIR-V with tests/semantic/spirv_mini_interp.py.

Comparison: bit for bit. Only an f32 NaN that went through an output multiplier matches any NaN (payloads are not
modelled there). A program is not judged (counted with the reason, not a failure) when:
  - f64 rule 6 depends on how a denormal's exponent counts (its field 0, 1, or its normalized exponent): the rows
    say "exp(S2) - exp(S1)" and only normal operands were measured;
  - an output multiplier without clamp meets -0 (the hardware gives +0, which docs/design.md lists as not
    modelled), or an output multiplier meets an f32 denormal or makes one (whether it flushes is the MODE
    question, SHADER-35);
  - clamp meets a positive f32 denormal, an f16 -0 (measured on f32 only, where it gives +0) or an f16 NaN (the
    translator's FClamp of it is undefined in SPIR-V);
  - an op reads part of an f32 NaN whose payload the reference does not model.

Program shape: seeds in v8..v15 and s4..s7, then the ops, then `exp mrt0 v8, v9, v10, v11` so every result is
observed. Every destination is in v8..v11 (an f64 one is v[8:9], v[9:10] or v[10:11]), so later ops read earlier
results, f64 halves as f32 or f16 values included.

Sources: VGPRs, SGPRs (f64: a VGPR pair from v[8:9] to v[14:15], s[4:5] or s[6:7]), a hex literal (not on f64),
an integer inline constant (-16..64) and a float inline constant (0.5, 1.0, 2.0, 4.0, their negatives and 1/(2 pi),
printed 0.15915494, and 0.15915494309189532 for an f64 operand). An f16 operand reads the low half of a register or
literal (the high half with its op_sel bit), a float inline constant as its f16 value and an integer inline constant
as the low half of its word (docs/design.md, "f16 inline constants"). An f64 operand reads an integer inline constant
sign-extended. Every program is encodable on GFX10: at most one literal and at most two constant-bus reads.

Modifiers: neg/abs on the register sources, op_sel on f16 register sources and the destination bit, clamp on f16
and f32, and mul:2 / mul:4 / div:2 on all three. Not generated, on purpose: a literal on f64 (the translator puts it
in bits 31:0; #3791 moves it to bits 63:32), clamp on f64 (the translation needs Float64, which spirv_mini_interp
does not run), op_sel on a constant, and the forms the translator refuses (--self-test pins those as expected
rejections).

Stages:
  1. a sweep: every encodable source-kind combination of every op, twice, with random modifiers;
  2. an edge grid: zeros, denormals, the normal bounds, 1, the largest finite value, inf and quiet and signalling
     NaNs as S1 and S2 under five S0 values; the rule 6 boundaries (one exponent above, at and below the cutoff, also
     against a NaN or inf S0, and on f64 with a denormal S2); abs/neg on every kind of source; the output
     modifiers on every rule's result (f32), f16 op_sel, clamp and output multipliers, f64 output multipliers;
  3. random seeded programs.

Exit code 0 when every judged program matches, 1 on a mismatch (the first one is printed with its listing) or a
translator skip (every generated form is one the translator takes), 2 on a harness error.

Usage:
    python -I tools/div_fixup_diff.py [--programs N] [--seed S] [--max-ops N] [--ops v_div_fixup_f64]
    python -I tools/div_fixup_diff.py --list-ops
    python -I tools/div_fixup_diff.py --self-test
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

M16 = 0xFFFF
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

EXPORT = "exp mrt0 v8, v9, v10, v11 done vm"
EXPORT_REGS = (8, 9, 10, 11)
DSTS = (8, 9, 10, 11)
DSTS64 = (8, 9, 10)
VGPRS = tuple(range(8, 16))
SGPRS = (4, 5, 6, 7)
SGPR_PAIRS = (4, 6)

OPS = ("v_div_fixup_f16", "v_div_fixup_f32", "v_div_fixup_f64")
OP_FORMS = {
    "v_div_fixup_f16": "d, [-][|]s0[|], [-][|]s1[|], [-][|]s2[|] [op_sel:[a,b,c,d]] [clamp] [mul:2|mul:4|div:2]",
    "v_div_fixup_f32": "d, [-][|]s0[|], [-][|]s1[|], [-][|]s2[|] [clamp] [mul:2|mul:4|div:2]",
    "v_div_fixup_f64": "d[2], [-][|]s0[2][|], [-][|]s1[2][|], [-][|]s2[2][|] [mul:2|mul:4|div:2] (no literal)",
}
RULES = ("NaN(S2)", "NaN(S1)", "0/0 or inf/inf", "zero", "inf", "exponent cutoff", "NaN(S0)", "copysign(S0)")

KINDS = ("vgpr", "sgpr", "lit", "int", "float")
KINDS64 = ("vgpr", "sgpr", "int", "float")
MODS = ("", "-", "|", "-|")         # source modifiers on a register: none, neg, abs, neg(abs)
OMODS = ("", "", "", "mul:2", "mul:4", "div:2")
# the grid's output modifiers on f16 and f64, where the hardware ignores the multipliers in the game's MODE (#6314)
F16_F64_OMODS = {16: ("mul:2", "mul:4", "div:2", "clamp mul:2", "clamp mul:4", "clamp div:2"),
                 64: ("mul:2", "mul:4", "div:2")}
FLOAT_INLINE = {  # token -> (f32 word, f16 half)
    "0.5": (0x3F000000, 0x3800), "-0.5": (0xBF000000, 0xB800), "1.0": (0x3F800000, 0x3C00),
    "-1.0": (0xBF800000, 0xBC00), "2.0": (0x40000000, 0x4000), "-2.0": (0xC0000000, 0xC000),
    "4.0": (0x40800000, 0x4400), "-4.0": (0xC0800000, 0xC400), "0.15915494": (0x3E22F983, 0x3118),
}
FLOAT_INLINE64 = {
    "0.5": 0x3FE0000000000000, "-0.5": 0xBFE0000000000000, "1.0": 0x3FF0000000000000,
    "-1.0": 0xBFF0000000000000, "2.0": 0x4000000000000000, "-2.0": 0xC000000000000000,
    "4.0": 0x4010000000000000, "-4.0": 0xC010000000000000, "0.15915494309189532": 0x3FC45F306DC9C882,
}
INT_INLINE = tuple(range(0, 65)) + tuple(range(-16, 0))


class HarnessError(Exception):
    """The generator or the reference met something outside the modelled forms."""


class NotJudged(Exception):
    """The reference does not judge this program (the reason is counted)."""


# ---- the reference ----

class Fmt:
    def __init__(self, name, width, ebits, cutoff, canon):
        self.name = name
        self.width = width
        self.sign = 1 << (width - 1)
        self.mag = self.sign - 1
        self.mbits = width - 1 - ebits
        self.inf = ((1 << ebits) - 1) << self.mbits
        self.min_norm = 1 << self.mbits
        self.cutoff = cutoff
        self.canon = canon


FMT = {
    "v_div_fixup_f16": Fmt("f16", 16, 5, None, 0xFE00),
    "v_div_fixup_f32": Fmt("f32", 32, 8, -151, 0xFFC00000),
    "v_div_fixup_f64": Fmt("f64", 64, 11, -1076, 0xFFF8000000000000),
}


def _exponents(fmt, a):
    """The readings of the biased exponent of a finite nonzero magnitude: its field, and for a denormal also 1 and
    its normalized exponent (rule 6 was measured on normal operands only)."""
    e = a >> fmt.mbits
    if e:
        return {e}
    return {0, 1, a.bit_length() - fmt.mbits}


def _cutoff(fmt, a1, a2):
    hits = {e2 - e1 <= fmt.cutoff for e1 in _exponents(fmt, a1) for e2 in _exponents(fmt, a2)}
    if len(hits) > 1:
        raise NotJudged(f"{fmt.name} exponent cutoff on a denormal: how its exponent counts is not measured")
    return hits.pop()


def fixup(fmt, s0, s1, s2):
    """(rule index, result bits) of div_fixup on the modified source bits, by the order in the module docstring."""
    a0, a1, a2 = s0 & fmt.mag, s1 & fmt.mag, s2 & fmt.mag
    sign = (s1 ^ s2) & fmt.sign
    if a2 > fmt.inf:
        return 0, s2
    if a1 > fmt.inf:
        return 1, s1
    if fmt.name == "f32":
        z1, z2 = a1 < fmt.min_norm, a2 < fmt.min_norm      # denorm32 = 0: S1 and S2 denormals are zeros
    else:
        z1, z2 = a1 == 0, a2 == 0
    i1, i2 = a1 == fmt.inf, a2 == fmt.inf
    if (z1 and z2) or (i1 and i2):
        return 2, fmt.canon
    if z2 or i1:
        return 3, sign
    if z1 or i2:
        return 4, sign | fmt.inf
    if fmt.cutoff is not None and _cutoff(fmt, a1, a2):
        return 5, sign
    if a0 > fmt.inf:
        return 6, sign | fmt.inf
    return 7, sign | a0


def _f32_value(w):
    """The f32 word as an exact (numerator, power of two) of its magnitude, for a normal or zero word."""
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    return (1 << 23) | m, e - 150


def out_f32(w, mods):
    """The f32 output modifiers on result word ``w``: (word, state), state "w32" for a NaN of unmodelled payload."""
    k = next((m for m in mods if m in ("mul:2", "mul:4", "div:2")), None)
    clamp = "clamp" in mods
    a = w & 0x7FFFFFFF
    if a > 0x7F800000:
        if clamp:
            return 0, None                          # DX10_CLAMP: a NaN gives +0
        return (w, None) if k is None else (0x7FC00000, "w32")
    if clamp and w >> 31:
        return 0, None                              # any negative value, -0 included (measured), clamps to +0
    if k is not None and a != 0x7F800000:
        if a == 0:
            if w >> 31:
                raise NotJudged("an output multiplier on -0 (the hardware gives +0, not modelled)")
        elif a < 0x00800000:
            raise NotJudged("an output multiplier on an f32 denormal")
        else:
            e = (a >> 23) + {"mul:2": 1, "mul:4": 2, "div:2": -1}[k]
            if e >= 0xFF:
                a = 0x7F800000
            elif e <= 0:
                raise NotJudged("an output multiplier gives an f32 denormal")
            else:
                a = (e << 23) | (a & 0x7FFFFF)
            w = (w & 0x80000000) | a
    if clamp:
        if 0 < a < 0x00800000:
            raise NotJudged("clamp of an f32 denormal")
        if a >= 0x3F800000:                          # above 1, or +inf
            return 0x3F800000, None
    return w, None


def clamp_f16(h):
    a = h & 0x7FFF
    if a > 0x7C00:
        raise NotJudged("clamp of an f16 NaN (FClamp of a NaN is undefined in SPIR-V)")
    if a == 0 and h & 0x8000:
        raise NotJudged("clamp of -0")
    if h & 0x8000:
        return 0
    return min(h, 0x3C00)


# ---- the program text ----

_REG = re.compile(r"([sv])(\d+)$")
_PAIR = re.compile(r"([sv])\[(\d+):(\d+)\]$")
_SRC = re.compile(r"(-?)(\|?)([sv]\d+|[sv]\[\d+:\d+\])\|?$")
_MOD = re.compile(r"\s+(op_sel:\[[01](?:,[01])*\]|clamp|mul:2|mul:4|div:2)$")


def split_line(line):
    """'op d, a, b mods' -> (op, [operands], [mods])."""
    mods = []
    while True:
        m = _MOD.search(line)
        if not m:
            break
        mods.insert(0, m.group(1))
        line = line[:m.start()]
    op, _, rest = line.partition(" ")
    return op, [t.strip() for t in rest.split(",")], mods


def _opsel(mods):
    m = next((x for x in mods if x.startswith("op_sel:")), None)
    if m is None:
        return [0, 0, 0, 0]
    b = [int(x) for x in m.split("[")[1].rstrip("]").split(",")]
    if len(b) != 4:
        raise HarnessError(f"{m}: want 4 bits")
    return b


class Machine:
    """The reference's register file: words, and per-half states: None (known bits) or "w32" (half of an f32 NaN
    whose payload the reference does not model)."""

    def __init__(self):
        self.v = {}
        self.s = {}
        self.st = {}

    def state(self, r):
        return self.st.setdefault(r, [None, None])

    def reg(self, k, i):
        if k == "v":
            if i not in self.v:
                raise HarnessError(f"v{i} read before it is written")
            return self.v[i], list(self.state(i))
        if i not in self.s:
            raise HarnessError(f"s{i} read before it is written")
        return self.s[i], [None, None]

    def source(self, tok, fmt, hi=0):
        """(modified source bits, unknown) of an operand of width fmt.width; ``unknown`` marks an f32 NaN whose
        payload is not modelled."""
        m = _SRC.match(tok)
        mod = (m.group(1) + m.group(2)) if m else ""
        base = m.group(3) if m else tok
        unknown = False
        pm, rm = _PAIR.match(base), _REG.match(base)
        if fmt.width == 64:
            if pm:
                k, lo, top = pm.group(1), int(pm.group(2)), int(pm.group(3))
                if top != lo + 1:
                    raise HarnessError(f"bad pair {base}")
                (wl, sl), (wh, sh) = self.reg(k, lo), self.reg(k, lo + 1)
                if "w32" in sl + sh:
                    raise NotJudged("an f32 NaN of unmodelled payload read as half of an f64")
                bits = wl | (wh << 32)
            elif base in FLOAT_INLINE64:
                bits = FLOAT_INLINE64[base]
            else:
                bits = self._int(base) & M64
        elif rm:
            w, sts = self.reg(rm.group(1), int(rm.group(2)))
            if fmt.width == 16:
                if sts[hi] == "w32":
                    raise NotJudged("half of an f32 NaN of unmodelled payload read as an f16")
                bits = (w >> 16) & M16 if hi else w & M16
            else:
                if sts == ["w32", "w32"]:
                    unknown = True
                elif "w32" in sts:
                    raise NotJudged("an f32 NaN of unmodelled payload read in part")
                bits = w
        elif base in FLOAT_INLINE:
            if hi:
                raise HarnessError("op_sel on a constant")
            bits = FLOAT_INLINE[base][1 if fmt.width == 16 else 0]
        elif base.startswith("0x"):
            if hi:
                raise HarnessError("op_sel on a constant")
            bits = int(base, 16) & (M16 if fmt.width == 16 else M32)
        else:
            if hi:
                raise HarnessError("op_sel on a constant")
            bits = self._int(base) & (M16 if fmt.width == 16 else M32)
        if "|" in mod:
            bits &= fmt.mag
        if mod.startswith("-"):
            bits ^= fmt.sign
        if unknown:
            bits = (bits & fmt.sign) | 0x7FC00000
        return bits, unknown

    @staticmethod
    def _int(tok):
        try:
            n = int(tok)
        except ValueError:
            raise HarnessError(f"unknown source {tok!r}") from None
        if n not in INT_INLINE:
            raise HarnessError(f"{n} is not an inline constant")
        return n

    def dst(self, tok, width):
        if width == 64:
            m = _PAIR.match(tok)
            if not m or m.group(1) != "v" or int(m.group(3)) != int(m.group(2)) + 1:
                raise HarnessError(f"destination {tok!r}")
            return int(m.group(2))
        m = _REG.match(tok)
        if not m or m.group(1) != "v":
            raise HarnessError(f"destination {tok!r}")
        return int(m.group(2))

    def write_half(self, r, hi, h):
        old = self.v.get(r, 0)
        self.v[r] = (old & M16) | (h << 16) if hi else (old & 0xFFFF0000) | h
        self.state(r)[hi] = None

    def write_word(self, r, w, st=None):
        self.v[r] = w & M32
        self.st[r] = [st, st]


def _count(events, op, fmt, rule, bits):
    if events is None:
        return
    ev = events.setdefault(op, Counter())
    ev[RULES[rule]] += 1
    for b in bits:
        a = b & fmt.mag
        if a > fmt.inf:
            ev["NaN in"] += 1
        elif a == fmt.inf:
            ev["inf in"] += 1
        elif 0 < a < fmt.min_norm:
            ev["denormal in"] += 1


def step(line, mc, events=None):
    """Run one op line on the Machine."""
    op, args, mods = split_line(line)
    fmt = FMT.get(op)
    if fmt is None:
        raise HarnessError(f"op {op!r} is not in the family")
    if len(args) != 4:
        raise HarnessError(f"{op}: want 4 operands")
    sel = _opsel(mods)
    if fmt.width != 16 and any(sel):
        raise HarnessError(f"op_sel on {op}")
    if sum(m in ("mul:2", "mul:4", "div:2") for m in mods) > 1:
        raise HarnessError(f"two output multipliers on {op}")
    if fmt.width == 64 and "clamp" in mods:
        raise HarnessError("clamp on v_div_fixup_f64")
    d = mc.dst(args[0], fmt.width)
    srcs = [mc.source(t, fmt, sel[i]) for i, t in enumerate(args[1:])]
    bits = [b for b, _ in srcs]
    rule, r = fixup(fmt, *bits)
    _count(events, op, fmt, rule, bits)
    # an output multiplier on an f16 or f64 result is ignored under denorm_mode_16_64 = 3 (#6314): nothing to do
    if fmt.width == 16:
        if "clamp" in mods:
            r = clamp_f16(r)
        mc.write_half(d, sel[3], r)
    elif fmt.width == 32:
        unknown = rule in (0, 1) and srcs[2 - rule][1]
        if unknown and "clamp" not in mods:
            mc.write_word(d, 0x7FC00000, "w32")
        else:
            w, st = out_f32(r, mods)
            mc.write_word(d, w, st)
    else:
        mc.write_word(d, r & M32)
        mc.write_word(d + 1, r >> 32)


def reference(lines, events=None):
    """Run the program; returns (four exported words, per-word half states). Raises NotJudged."""
    mc = Machine()
    for line in lines:
        op, args, _ = split_line(line)
        if op == "v_mov_b32":
            mc.write_word(mc.dst(args[0], 32), int(args[1], 16))
        elif op == "s_mov_b32":
            mc.s[int(args[0][1:])] = int(args[1], 16) & M32
        else:
            step(line, mc, events)
    return [mc.v[r] for r in EXPORT_REGS], [list(mc.state(r)) for r in EXPORT_REGS]


def compare(want, states, got):
    """None when every judged bit agrees, else a description of the first difference."""
    for i, (w, sts, g) in enumerate(zip(want, states, got)):
        g &= M32
        if sts == ["w32", "w32"]:
            if not ((g >> 23) & 0xFF == 0xFF and g & 0x7FFFFF):
                return f"word {i} (v{EXPORT_REGS[i]}): expected an f32 NaN, actual 0x{g:08x}"
            continue
        mask = (0 if sts[0] else M16) | (0 if sts[1] else M16 << 16)
        if (w ^ g) & mask:
            return f"word {i} (v{EXPORT_REGS[i]}): expected 0x{w:08x}, actual 0x{g:08x}"
    return None


# ---- the generator ----

F16_EDGES = (0x0000, 0x8000, 0x0001, 0x83FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0x7C00, 0xFC00, 0x7C01, 0xFE00)
F32_EDGES = (0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
             0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00000)
F64_EDGES = (0, 1 << 63, 1, (1 << 63) | ((1 << 52) - 1), 1 << 52, 0x3FF0000000000000, 0xBFF0000000000000,
             0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000001, 0xFFF8000000000000)
EDGES = {16: F16_EDGES, 32: F32_EDGES, 64: F64_EDGES}
# S0 under the edge grid: 1.5, a denormal, -0, -inf and a NaN with a payload
S0_EDGES = {16: (0x3E00, 0x0003, 0x8000, 0xFC00, 0x7D11),
            32: (0x3FC00000, 0x00000003, 0x80000000, 0xFF800000, 0x7FA00011),
            64: (0x3FF8000000000000, 3, 1 << 63, 0xFFF0000000000000, 0x7FF4000000000011)}


def _rand_f16(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(F16_EDGES)
    if r < 0.4:
        return (rng.getrandbits(1) << 15) | rng.randrange(1, 0x400)
    if r < 0.5:
        return rng.getrandbits(16)
    return (rng.getrandbits(1) << 15) | (rng.randrange(1, 31) << 10) | rng.getrandbits(10)


def _rand_f32(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(F32_EDGES)
    if r < 0.4:
        return rng.getrandbits(32)
    if r < 0.5:
        return (rng.getrandbits(1) << 31) | rng.randrange(1, 0x800000)
    e = rng.randrange(1, 255) if rng.random() < 0.6 else rng.choice((1, 2, 3, 100, 127, 150, 151, 152, 153, 254))
    return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)


def _f64_high(rng):
    """The high word of a binary64: sign, an exponent (often an extreme one), 20 mantissa bits."""
    r = rng.random()
    if r < 0.15:
        return rng.choice(F64_EDGES) >> 32
    if r < 0.4:
        e = rng.choice((0, 0, 1, 2, 3, 52, 1022, 1023, 1024, 1076, 1077, 1078, 2045, 2046, 2047))
    else:
        e = rng.randrange(0, 2048)
    return (rng.getrandbits(1) << 31) | (e << 20) | (rng.getrandbits(20) if rng.random() < 0.7 else 0)


def _rand_f64(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(F64_EDGES)
    lo = 0 if rng.random() < 0.4 else rng.getrandbits(32)
    return (_f64_high(rng) << 32) | lo


def _seed_word(rng):
    r = rng.random()
    if r < 0.3:
        return _rand_f16(rng) | (_rand_f16(rng) << 16)
    if r < 0.6:
        return _rand_f32(rng)
    if r < 0.85:
        return _f64_high(rng)
    return 0 if rng.random() < 0.5 else rng.getrandbits(32)


def _source(rng, kind, width):
    """A source token of ``kind`` (no modifier) for an operand of ``width`` bits."""
    if width == 64:
        if kind == "vgpr":
            n = rng.randrange(8, 15)
            return f"v[{n}:{n + 1}]"
        if kind == "sgpr":
            n = rng.choice(SGPR_PAIRS)
            return f"s[{n}:{n + 1}]"
        if kind == "int":
            return str(rng.choice(INT_INLINE))
        if kind == "float":
            return rng.choice(tuple(FLOAT_INLINE64))
        raise HarnessError(f"{kind} source on f64")
    if kind == "vgpr":
        return f"v{rng.choice(VGPRS)}"
    if kind == "sgpr":
        return f"s{rng.choice(SGPRS)}"
    if kind == "lit":
        w = (_rand_f16(rng) | (rng.getrandbits(16) << 16)) if width == 16 else _rand_f32(rng)
        return f"0x{w:08x}"
    if kind == "int":
        return str(rng.choice(INT_INLINE))
    return rng.choice(tuple(FLOAT_INLINE))


def encodable(kinds):
    """GFX10 VOP3: at most one literal, at most two constant-bus reads (SGPRs and the literal)."""
    return sum(k == "lit" for k in kinds) <= 1 and sum(k in ("sgpr", "lit") for k in kinds) <= 2


def _reg_mod(rng, tok, kind):
    if kind not in ("vgpr", "sgpr"):
        return tok
    m = rng.choice(MODS)
    return f"{'-' if m.startswith('-') else ''}{'|' if '|' in m else ''}{tok}{'|' if '|' in m else ''}"


def _out_mods(rng, width):
    """The output modifiers of a random line: clamp on f16 and f32, an output multiplier on all three."""
    extra = ""
    if width in (16, 32) and rng.random() < 0.25:
        extra += " clamp"
    k = rng.choice(OMODS)
    if k:
        extra += " " + k
    return extra


def op_line(rng, op, kinds=None, mods=True):
    """One line of ``op``; ``kinds`` fixes the three source kinds (else random)."""
    width = FMT[op].width
    allowed = KINDS64 if width == 64 else KINDS
    while True:
        ks = list(kinds) if kinds else [rng.choice(allowed) for _ in range(3)]
        if encodable(ks) and all(k in allowed for k in ks):
            break
        if kinds:
            raise HarnessError(f"{op}: {kinds} not encodable")
    toks = [_source(rng, k, width) for k in ks]
    srcs = [_reg_mod(rng, t, k) if mods else t for t, k in zip(toks, ks)]
    if width == 64:
        n = rng.choice(DSTS64)
        return f"{op} v[{n}:{n + 1}], {', '.join(srcs)}{_out_mods(rng, width) if mods else ''}", ks
    extra = ""
    if width == 16:
        sel = [rng.getrandbits(1) if k in ("vgpr", "sgpr") else 0 for k in ks] + [rng.getrandbits(1)]
        if any(sel) or rng.random() < 0.2:
            extra = f" op_sel:[{','.join(map(str, sel))}]"
    if mods:
        extra += _out_mods(rng, width)
    return f"{op} v{rng.choice(DSTS)}, {', '.join(srcs)}{extra}", ks


def seeds(rng):
    lines = [f"v_mov_b32 v{r}, 0x{_seed_word(rng):08x}" for r in VGPRS]
    lines += [f"s_mov_b32 s{r}, 0x{_seed_word(rng):08x}" for r in SGPRS]
    return lines


def program(rng, n_ops, ops=None):
    """A random program: seeds, n_ops family ops, the export. Returns (lines, the ops used)."""
    ops = list(ops or OPS)
    lines = seeds(rng)
    used = []
    for _ in range(n_ops):
        op = rng.choice(ops)
        lines.append(op_line(rng, op)[0])
        used.append(op)
    return lines, used


def sweep_cases(ops=None):
    """(op, kinds) for every encodable source-kind combination of every op."""
    cases = []
    for op in ops or OPS:
        allowed = KINDS64 if FMT[op].width == 64 else KINDS
        for a in allowed:
            for b in allowed:
                for c in allowed:
                    if encodable([a, b, c]):
                        cases.append((op, (a, b, c)))
    return cases


SWEEP_REPEAT = 2


def sweep_program(case, rng):
    op, kinds = case
    return seeds(rng) + [op_line(rng, op, kinds)[0]]


# ---- the edge grid ----

GRID_PER_PROGRAM = 4
_GRID_RNG_SEED = "div_fixup_diff:grid"


def _with_exp(width, sign, e, rng):
    """A value of ``width`` with biased exponent ``e`` (0: a nonzero denormal) and a random mantissa."""
    f = FMT[{16: "v_div_fixup_f16", 32: "v_div_fixup_f32", 64: "v_div_fixup_f64"}[width]]
    m = rng.randrange(1, f.min_norm) if e == 0 else rng.getrandbits(f.mbits)
    return (sign << (width - 1)) | (e << f.mbits) | m


def grid_cases(ops=None):
    """(op, (s0, s1, s2) words, (three source modifiers), op_sel bits, output modifiers) cases."""
    rng = random.Random(_GRID_RNG_SEED)
    cases = []
    for op in ops or OPS:
        f = FMT[op]
        w = f.width
        plain = ("", "", "")
        # 1. every edge pair as S1 and S2, under five S0 values
        for s0 in S0_EDGES[w]:
            for s1 in EDGES[w]:
                for s2 in EDGES[w]:
                    cases.append((op, (s0, s1, s2), plain, (0, 0, 0, 0), ""))
        # 2. the exponent cutoff: one above, at, and one below it, also against a NaN and an inf S0
        if f.cutoff is not None:
            top = (1 << (w - 1 - f.mbits)) - 2                     # the largest finite biased exponent
            for diff in (f.cutoff + 1, f.cutoff, f.cutoff - 1):
                for e1 in (top, top - 1, -diff + 1, -diff + 2, -diff + 40):
                    e2 = e1 + diff
                    if not (1 <= e1 <= top and 1 <= e2 <= top):
                        continue
                    for s0 in (S0_EDGES[w][0], S0_EDGES[w][4], f.inf):
                        sg = rng.getrandbits(2)
                        cases.append((op, (s0, _with_exp(w, sg & 1, e1, rng), _with_exp(w, sg >> 1, e2, rng)),
                                      plain, (0, 0, 0, 0), ""))
            if w == 64:
                # a denormal numerator against large denominators: the readings of its exponent agree below and
                # above the band where they differ, and disagree (not judged) inside it
                for e1 in (1000, 1024, 1025, 1050, 1075, 1076, 1077, 1100, 2046):
                    for m2 in (1, 1 << 20, (1 << 52) - 1):
                        cases.append((op, (S0_EDGES[w][0], _with_exp(w, 0, e1, rng), m2), plain, (0, 0, 0, 0), ""))
        # 3. abs/neg on every kind of source, NaNs included
        for _ in range(240):
            words = (rng.choice(S0_EDGES[w] + EDGES[w]), rng.choice(EDGES[w]), rng.choice(EDGES[w]))
            cases.append((op, words, tuple(rng.choice(MODS) for _ in range(3)), (0, 0, 0, 0), ""))
        # 4. the output modifiers on each rule's result (f32), clamp and op_sel (f16)
        one, neg_one = {16: (0x3C00, 0xBC00), 32: (0x3F800000, 0xBF800000), 64: (F64_EDGES[5], F64_EDGES[6])}[w]
        nan = EDGES[w][10]
        zero = 0
        outcomes = ((one, one), (zero, zero), (one, zero), (zero, one), (one, nan), (neg_one, one), (nan, one))
        if w == 64:
            # the multipliers change nothing on a binary64 result (#6314); clamp is not generated on f64
            for s0 in S0_EDGES[w] + (F64_EDGES[7], F64_EDGES[4]):
                for s1, s2 in outcomes:
                    for om in F16_F64_OMODS[w]:
                        cases.append((op, (s0, s1, s2), plain, (0, 0, 0, 0), om))
            continue
        if w == 32:
            s0s = (0x3F000000, 0x3FC00000, 0xBF000000, 0x7F7FFFFF, 0x00800000, 0x00000001, 0x7FC00001, 0x3E800000)
            omods = ("clamp", "mul:2", "mul:4", "div:2", "clamp mul:2", "clamp mul:4", "clamp div:2")
            for s0 in s0s:
                for s1, s2 in outcomes:
                    for om in omods:
                        cases.append((op, (s0, s1, s2), plain, (0, 0, 0, 0), om))
        else:
            s0s = (0x3800, 0x3E00, 0xB800, 0x7BFF, 0x0001, 0x7D00, 0x3C00)
            for s0 in s0s:
                for s1, s2 in outcomes:
                    for clamp in ("", "clamp"):
                        for _ in range(2):
                            sel = tuple(rng.getrandbits(1) for _ in range(4))
                            cases.append((op, (s0, s1, s2), plain, sel, clamp))
            # the multipliers change nothing on an f16 result (#6314), with or without clamp, in either half
            for i, (s0, (s1, s2)) in enumerate((s0, o) for s0 in s0s for o in outcomes):
                for om in F16_F64_OMODS[w]:
                    cases.append((op, (s0, s1, s2), plain, (0, 0, 0, i & 1), om))
    return cases


def _mod(tok, m):
    return f"{'-' if m.startswith('-') else ''}{'|' if '|' in m else ''}{tok}{'|' if '|' in m else ''}"


def grid_program(chunk, rng):
    """A program for a chunk of grid cases of one op. Case i reads its own registers from v16 + 6i on and writes
    v{8 + i} (f64: v[8 + 2i : 9 + 2i])."""
    lines = [f"v_mov_b32 v{r}, 0x{rng.getrandbits(32):08x}" for r in VGPRS]
    lines += [f"s_mov_b32 s{r}, 0x{rng.getrandbits(32):08x}" for r in SGPRS]
    body = []
    for i, (op, words, mods, sel, om) in enumerate(chunk):
        w = FMT[op].width
        base = 16 + 6 * i
        if w == 64:
            for j, x in enumerate(words):
                lines.append(f"v_mov_b32 v{base + 2 * j}, 0x{x & M32:08x}")
                lines.append(f"v_mov_b32 v{base + 2 * j + 1}, 0x{x >> 32:08x}")
            srcs = [_mod(f"v[{base + 2 * j}:{base + 2 * j + 1}]", m) for j, m in enumerate(mods)]
            body.append(f"{op} v[{8 + 2 * i}:{9 + 2 * i}], {', '.join(srcs)}{' ' + om if om else ''}")
            continue
        for j, x in enumerate(words):
            if w == 16:
                other = rng.getrandbits(16)
                x = (x << 16) | other if sel[j] else x | (other << 16)
            lines.append(f"v_mov_b32 v{base + j}, 0x{x:08x}")
        srcs = [_mod(f"v{base + j}", m) for j, m in enumerate(mods)]
        extra = f" op_sel:[{','.join(map(str, sel))}]" if any(sel) else ""
        body.append(f"{op} v{8 + i}, {', '.join(srcs)}{extra}{' ' + om if om else ''}")
    return lines + body


# ---- the oracles ----

def write_listing(path, lines, name):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}  code 0x0..0x1000\n")
        for line in lines:
            f.write("\t" + line + "\n")
        f.write("\t" + EXPORT + "\n")
        f.write("\ts_endpgm\n")


def translate_listing(lines, name, tmpdir):
    """The four mrt0 words the translator path produces. Raises rdna2spirv.Unsupported (and the mini interp's
    UnsupportedConstruct) for anything outside the oracles' model."""
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


def check(lines, name="ps_dfx_case"):
    """None when both oracles agree, else the first difference."""
    with tempfile.TemporaryDirectory(prefix="dfx-diff-") as tmp:
        want, states = reference(lines)
        got = translate_listing(lines, name, tmp)
    return compare(want, states, got)


def run_one(lines, name, tmpdir, events=None):
    """(status, detail) with status in {"match", "mismatch", "skip", "unjudged"}."""
    try:
        want, states = reference(lines, events)
    except NotJudged as e:
        return ("unjudged", str(e))
    try:
        got = translate_listing(lines, name, tmpdir)
    except (rdna2spirv.Unsupported, UnsupportedConstruct) as e:
        return ("skip", f"{type(e).__name__}: {e}")
    except (IndexError, KeyError, ValueError) as e:
        return ("skip", f"translator error {type(e).__name__}: {e}")
    diff = compare(want, states, got)
    if diff:
        return ("mismatch", diff)
    return ("match", "")


# ---- the suites ----

def _new_counts():
    return {"pass": 0, "mismatch": 0, "skip": Counter(), "unjudged": Counter(), "ops": Counter(), "events": {}}


def _tally(counts, status, detail):
    if status == "match":
        counts["pass"] += 1
    elif status in ("skip", "unjudged"):
        counts[status][detail] += 1
    else:
        counts["mismatch"] += 1


def _run(counts, items, tag):
    failure = None
    with tempfile.TemporaryDirectory(prefix=f"dfx-diff-{tag}-") as tmp:
        for i, (label, lines, used) in enumerate(items):
            counts["ops"].update(used)
            status, detail = run_one(lines, f"ps_dfx_{tag}{i:05d}", tmp, counts["events"])
            _tally(counts, status, detail)
            if status == "mismatch" and failure is None:
                failure = (label, lines, detail)
    return failure


def sweep_items(seed, ops=None):
    """(label, lines, ops) of the sweep, yielded one at a time so a run holds one program's text."""
    for case in sweep_cases(ops):
        op, kinds = case
        for rep in range(SWEEP_REPEAT):
            tag = f"{op}:{'/'.join(kinds)}:{rep}"
            rng = random.Random(f"{seed}:{tag}")
            yield f"sweep {tag} (seed {seed})", sweep_program(case, rng), [op]


def run_sweep(seed, ops=None):
    counts = _new_counts()
    return counts, _run(counts, sweep_items(seed, ops), "sw")


def grid_items(ops=None):
    """(label, lines, ops) of the edge grid: four cases per program, two for f64, yielded one at a time."""
    rng = random.Random("div_fixup_diff:grid-programs")
    cases = grid_cases(ops)
    for op in ops or OPS:
        mine = [c for c in cases if c[0] == op]
        n = 2 if FMT[op].width == 64 else GRID_PER_PROGRAM
        for i in range(0, len(mine), n):
            chunk = mine[i:i + n]
            yield f"edge grid {op} program {i // n}", grid_program(chunk, rng), [op] * len(chunk)


def run_grid(ops=None):
    counts = _new_counts()
    return counts, _run(counts, grid_items(ops), "gr")


def suite_items(programs, seed, max_ops=6, ops=None):
    """(label, lines, ops) of the random programs, yielded one at a time."""
    rng = random.Random(f"suite:{seed}")
    for i in range(programs):
        lines, used = program(rng, rng.randint(1, max_ops), ops)
        yield f"program {i} (seed {seed})", lines, used


def run_suite(programs, seed, max_ops=6, ops=None):
    counts = _new_counts()
    return counts, _run(counts, suite_items(programs, seed, max_ops, ops), "rn")


REJECTED = (
    ("op_sel on v_div_fixup_f32", "v_div_fixup_f32 v8, v9, v10, v11 op_sel:[1,0,0,0]"),
    ("a single-VGPR destination of v_div_fixup_f64", "v_div_fixup_f64 v8, v[10:11], v[12:13], v[14:15]"),
)


def self_test():
    ok = True

    def report(good, what):
        nonlocal ok
        ok &= good
        print(("ok   " if good else "FAIL ") + what)

    # reference spot checks, from the docs/design.md rows
    f16, f32, f64 = (FMT[op] for op in OPS)
    for fmt, words, want, what in (
            (f32, (0x3F800000, 0x7FC00011, 0x7F800005), 0x7F800005, "f32: NaN(S2) beats NaN(S1), unchanged"),
            (f32, (0x3F800000, 0x00000001, 0x807FFFFF), 0xFFC00000, "f32: denormal S1 and S2 read as 0/0"),
            (f32, (0x7FC00000, 0x4C000000, 0x00800000), 0x00000000, "f32: e2 - e1 = -151 gives 0, beats NaN(S0)"),
            (f32, (0x7FC00000, 0x4B800000, 0x00800000), 0x7F800000, "f32: e2 - e1 = -150 keeps NaN(S0) -> inf"),
            (f32, (0x00000003, 0xBF800000, 0x3F800000), 0x80000003, "f32: an S0 denormal is kept"),
            (f16, (0x3C00, 0x7800, 0x0400), 0x3C00, "f16: no exponent cutoff (e2 - e1 = -29)"),
            (f16, (0x7E35, 0xC000, 0x3C00), 0xFC00, "f16: NaN(S0) gives -inf"),
            (f64, (0x3FF0000000000000, 2046 << 52, 971 << 52), 0x3FF0000000000000, "f64: e2 - e1 = -1075 keeps S0"),
            (f64, (0x3FF0000000000000, 2046 << 52, 970 << 52), 0, "f64: e2 - e1 = -1076 gives +0"),
            (f64, (0x3FF0000000000000, 0, 1 << 63), 0xFFF8000000000000, "f64: 0/0 gives 0xfff8000000000000")):
        report(fixup(fmt, *words)[1] == want, what)
    report(out_f32(0x3FC00000, ["clamp", "mul:2"]) == (0x3F800000, None), "f32: mul:2 then clamp of 1.5 is 1.0")
    report(out_f32(0xFFC00000, ["clamp"]) == (0, None), "f32: clamp of a NaN is +0")
    zeros = [f"v_mov_b32 v{r}, 0x0" for r in EXPORT_REGS]
    words = reference(zeros + ["v_mov_b32 v12, 0x3e00", "v_mov_b32 v13, 0x3c00", "v_mov_b32 v14, 0x3c00",
                               "v_div_fixup_f16 v8, v12, v13, v14 clamp div:2"])[0]
    report(words[0] & 0xFFFF == 0x3C00, "f16: div:2 is ignored and clamp still applies (1.5 gives 1.0, not 0.75)")
    words = reference(zeros + ["v_mov_b32 v12, 0x0", "v_mov_b32 v13, 0x3ff80000", "v_mov_b32 v14, 0x0",
                               "v_mov_b32 v15, 0x3ff00000", "v_div_fixup_f64 v[8:9], v[12:13], v[14:15], v[14:15] mul:4"])[0]
    report(words[:2] == [0, 0x3FF80000], "f64: mul:4 is ignored (1.5 stays 1.5)")
    for what, line in REJECTED:
        try:
            with tempfile.TemporaryDirectory(prefix="dfx-diff-st-") as tmp:
                translate_listing(seeds(random.Random(1)) + [line], "ps_dfx_rej", tmp)
            report(False, f"{what}: the translator accepted it")
        except (rdna2spirv.Unsupported, UnsupportedConstruct):
            report(True, f"{what}: the translator rejected it")
    a = program(random.Random("st"), 5)[0]
    b = program(random.Random("st"), 5)[0]
    report(a == b, "generation is deterministic for a seed")
    counts, failure = run_suite(8, 1)
    report(failure is None and not counts["skip"],
           f"a short seeded run matches ({counts['pass']} programs, skips {dict(counts['skip'])})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--programs", type=int, default=300, help="random programs after the sweep and the grid")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=6, help="ops per random program")
    ap.add_argument("--ops", default=None, help="comma-separated opcodes to cover (default: all three)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the source-kind sweep and the edge grid")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.list_ops:
        for op in OPS:
            print(f"{op:16s} {OP_FORMS[op]}")
        return 0
    if args.self_test:
        return 0 if self_test() else 1
    ops = None
    if args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]
        bad = [o for o in ops if o not in OPS]
        if bad:
            print(f"unknown op(s): {', '.join(bad)} (see --list-ops)", file=sys.stderr)
            return 2
    stages = []
    if not args.no_sweep:
        stages += [("sweep", lambda: run_sweep(args.seed, ops)), ("edge grid", lambda: run_grid(ops))]
    if args.programs > 0:
        stages.append(("random", lambda: run_suite(args.programs, args.seed, args.max_ops, ops)))
    total = {"pass": 0, "mismatch": 0, "skip": 0, "unjudged": Counter(), "ops": Counter(), "events": {}}
    first = None
    try:
        for name, fn in stages:
            counts, failure = fn()
            n = counts["pass"] + counts["mismatch"] + sum(counts["skip"].values()) + sum(counts["unjudged"].values())
            print(f"{name}: {counts['pass']}/{n} programs match, {counts['mismatch']} mismatch; "
                  f"skips: {dict(counts['skip'])}; not judged: {sum(counts['unjudged'].values())}")
            total["pass"] += counts["pass"]
            total["mismatch"] += counts["mismatch"]
            total["skip"] += sum(counts["skip"].values())
            total["unjudged"].update(counts["unjudged"])
            total["ops"].update(counts["ops"])
            for op, ev in counts["events"].items():
                total["events"].setdefault(op, Counter()).update(ev)
            if failure and first is None:
                first = failure
    except HarnessError as e:
        print(f"harness error: {e}", file=sys.stderr)
        return 2
    print("op lines per op: " + ", ".join(f"{op} {n}" for op, n in sorted(total["ops"].items())))
    for op, ev in sorted(total["events"].items()):
        print(f"rules taken by {op}: " + ", ".join(f"{r} {ev[r]}" for r in RULES)
              + f"; NaN / inf / denormal inputs {ev['NaN in']}/{ev['inf in']}/{ev['denormal in']}")
    if total["unjudged"]:
        print("not judged: " + ", ".join(f"{why} {n}" for why, n in total["unjudged"].most_common()))
    judged = total["pass"] + total["mismatch"] + total["skip"]
    print(f"div_fixup family: {total['pass']}/{judged} judged programs match, {total['mismatch']} mismatch "
          f"({len(total['ops'])} of {len(OPS)} ops)")
    if first:
        _report(first)
        return 1
    if total["skip"]:
        print(f"SKIPPED {total['skip']} program(s): every generated form should translate (reasons above)")
        return 1
    return 0


def _report(failure):
    label, lines, detail = failure
    print(f"\nFIRST MISMATCH: {label}: {detail}")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    sys.exit(main())
