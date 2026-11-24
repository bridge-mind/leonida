r"""Measure RDNA1 f32 denormal flush in v_mad_f32, v_mac_f32 and v_mad_legacy_f32 (SHADER-340).

docs/design.md ("Denormal flush in v_mad_f32, v_mac_f32, v_mad_legacy_f32") states that
these three GFX10.1 instructions have no f32 denormal support: subnormal operands, rounded
products and results flush to signed zero regardless of MODE, and the flush runs after
output modifiers (`div:2`/`mul:2`/`mul:4`/`clamp`). Both claims were inferred from LLVM's
`SITargetLowering::isFMADLegal` and unmeasured on hardware (GFX10.3+ dropped these opcodes).

Each case tests one of:
  input    subnormal operand (src0, src1, src2 / accumulator)
  product  two normal operands whose rounded product is subnormal
  result   normal operands and product whose sum is subnormal
  omod     output modifier (`div:2`, `mul:2`, `mul:4`, `clamp`) crossing the subnormal boundary
  legacy   v_mad_legacy_f32 `0 * anything -> +0` with subnormal multipliers
  control  normal values, signed zeros, infinities, NaNs

Sides:
    translator  tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py, on the CPU.
    hw          the native instruction on an AMD GPU (tools/hw_oracle: HIP on Windows,
                libdrm on Linux), wave32 and wave64, across five MODE combinations
                including `denorm32=3` (f32 denormals kept).

    python -I tools/mad_denorm_probe.py                 # translator side only
    python -I tools/mad_denorm_probe.py --oracle hw     # + the AMD GPU in this machine

Exit code: 0 when every hardware word equals the translator's, 1 when one differs, 2 on a
harness error, 77 when the hardware oracle is unavailable. Nothing here reads game files.
Usage: `python -I tools/mad_denorm_probe.py --help` lists the options.
"""
import argparse
import os
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or HERE
for p in (TOOLS_DIR, os.path.join(ROOT, "tests", "semantic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SENTINEL = 0xDEAD0000

POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
SUBNORM_MIN = 0x00000001       # 2^-149
NEG_SUBNORM_MIN = 0x80000001   # -2^-149
SUBNORM_HALF = 0x00400000      # 2^-127 (0.5 * FLT_MIN)
NEG_SUBNORM_HALF = 0x80400000  # -2^-127
SUBNORM_MAX = 0x007FFFFF       # (1 - 2^-23) * 2^-126
FLT_MIN = 0x00800000           # 2^-126
NEG_FLT_MIN = 0x80800000       # -2^-126
FLT_MIN_1_5 = 0x00C00000       # 1.5 * 2^-126
NEG_FLT_MIN_1_5 = 0x80C00000   # -1.5 * 2^-126
HALF = 0x3F000000              # 0.5
ONE = 0x3F800000               # 1.0
NEG_ONE = 0xBF800000           # -1.0
TWO = 0x40000000               # 2.0
THREE = 0x40400000             # 3.0
FOUR = 0x40800000              # 4.0
POS_INF = 0x7F800000           # +inf
NEG_INF = 0xFF800000           # -inf
QNAN = 0x7FC00000              # canonical quiet NaN
SNAN = 0x7F800001              # signalling NaN

GAME_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round32": 0, "round16_64": 0}
MODES = (
    ("game", GAME_MODE),
    ("ieee1", dict(GAME_MODE, ieee=1)),
    ("dx10clamp0", dict(GAME_MODE, dx10_clamp=0)),
    ("ieee1_dx10clamp0", dict(GAME_MODE, ieee=1, dx10_clamp=0)),
    ("denorm32_kept", dict(GAME_MODE, denorm32=3)),
)


class Case:
    __slots__ = ("group", "op", "a", "b", "c", "mods", "neg_a", "what")

    def __init__(self, group, op, a, b, c, what, mods="", neg_a=False):
        self.group = group
        self.op = op
        self.a = a
        self.b = b
        self.c = c
        self.mods = mods
        self.neg_a = neg_a
        self.what = what


CASES = (
    # Input subnormal flush across all three operand positions
    Case("input", "v_mad_f32", SUBNORM_MIN, ONE, POS_ZERO, "src0 +subnorm -> +0"),
    Case("input", "v_mad_f32", NEG_SUBNORM_MIN, ONE, POS_ZERO, "src0 -subnorm + +0 -> +0"),
    Case("input", "v_mad_f32", NEG_SUBNORM_MIN, ONE, NEG_ZERO, "src0 -subnorm + -0 -> -0"),
    Case("input", "v_mad_f32", ONE, SUBNORM_MAX, POS_ZERO, "src1 +subnorm -> +0"),
    Case("input", "v_mad_f32", ONE, NEG_SUBNORM_MIN, NEG_ZERO, "src1 -subnorm + -0 -> -0"),
    Case("input", "v_mad_f32", POS_ZERO, POS_ZERO, SUBNORM_HALF, "src2 +subnorm -> +0"),
    Case("input", "v_mad_f32", NEG_ZERO, ONE, NEG_SUBNORM_HALF, "(-0)*1 + (-subnorm) -> -0"),
    Case("input", "v_mad_f32", SUBNORM_MIN, ONE, POS_ZERO, "-src0(+subnorm) + +0 -> +0", neg_a=True),
    Case("input", "v_mad_f32", SUBNORM_MIN, ONE, NEG_ZERO, "-src0(+subnorm) + -0 -> -0", neg_a=True),
    Case("input", "v_mac_f32", SUBNORM_MIN, ONE, SUBNORM_MIN, "mac subnorm src0, acc -> +0"),
    Case("input", "v_mac_f32", NEG_SUBNORM_MIN, ONE, NEG_SUBNORM_MIN, "mac -subnorm src0, acc -> -0"),
    Case("input", "v_mac_f32", POS_ZERO, POS_ZERO, SUBNORM_HALF, "mac subnorm acc -> +0"),
    Case("input", "v_mac_f32", ONE, ONE, SUBNORM_MIN, "mac 1*1 + subnorm -> 1.0"),
    Case("input", "v_mad_legacy_f32", POS_ZERO, POS_ZERO, SUBNORM_HALF, "legacy subnorm src2 -> +0"),
    Case("input", "v_mad_legacy_f32", ONE, ONE, NEG_SUBNORM_HALF, "legacy 1*1 + (-subnorm) -> 1.0"),

    # Normal operands whose rounded product is subnormal
    Case("product", "v_mad_f32", FLT_MIN, HALF, FLT_MIN, "2^-126 * 0.5 + 2^-126 (prod flushed)"),
    Case("product", "v_mac_f32", FLT_MIN, HALF, FLT_MIN, "mac 2^-126 * 0.5 + 2^-126"),
    Case("product", "v_mad_legacy_f32", FLT_MIN, HALF, FLT_MIN, "legacy 2^-126 * 0.5 + 2^-126"),
    Case("product", "v_mad_f32", 0x17800000, 0x1C800000, SUBNORM_MIN, "2^-80 * 2^-70 + 2^-149 -> +0"),
    Case("product", "v_mad_f32", 0x17800000, 0x21800000, 0x00000200, "2^-80 * 2^-59 + 2^-140 -> +0"),
    Case("product", "v_mad_f32", NEG_FLT_MIN, HALF, NEG_ZERO, "-2^-126 * 0.5 + -0 -> -0"),
    Case("product", "v_mac_f32", NEG_FLT_MIN, HALF, NEG_ZERO, "mac -2^-126 * 0.5 + -0 -> -0"),

    # Normal operands and normal product whose sum is subnormal
    Case("result", "v_mad_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "1.5*2^-126 - 2^-126 -> +0"),
    Case("result", "v_mad_f32", NEG_FLT_MIN_1_5, ONE, FLT_MIN, "-1.5*2^-126 + 2^-126 -> -0"),
    Case("result", "v_mac_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "mac 1.5*2^-126 - 2^-126 -> +0"),
    Case("result", "v_mac_f32", NEG_FLT_MIN_1_5, ONE, FLT_MIN, "mac -1.5*2^-126 + 2^-126 -> -0"),
    Case("result", "v_mad_legacy_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "legacy 1.5*2^-126 - 2^-126 -> +0"),
    Case("result", "v_mad_legacy_f32", NEG_FLT_MIN_1_5, ONE, FLT_MIN, "legacy -1.5*2^-126 + 2^-126 -> -0"),

    # Output modifiers (div:2, mul:2, mul:4, clamp) vs flush order
    Case("omod", "v_mad_f32", FLT_MIN, ONE, POS_ZERO, "FLT_MIN div:2 -> +0", mods="div:2"),
    Case("omod", "v_mad_f32", NEG_FLT_MIN, ONE, NEG_ZERO, "-FLT_MIN div:2 -> -0", mods="div:2"),
    Case("omod", "v_mad_f32", NEG_FLT_MIN, ONE, NEG_ZERO, "-FLT_MIN clamp div:2 -> +0", mods="clamp div:2"),
    Case("omod", "v_mad_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "(1.5*2^-126 - 2^-126) mul:2", mods="mul:2"),
    Case("omod", "v_mad_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "(1.5*2^-126 - 2^-126) mul:4", mods="mul:4"),
    Case("omod", "v_mad_f32", SUBNORM_HALF, ONE, POS_ZERO, "subnorm_half mul:2 (input flushed)", mods="mul:2"),
    Case("omod", "v_mad_legacy_f32", FLT_MIN, ONE, POS_ZERO, "legacy FLT_MIN div:2 -> +0", mods="div:2"),
    Case("omod", "v_mad_legacy_f32", FLT_MIN_1_5, ONE, NEG_FLT_MIN, "legacy (1.5*2^-126 - 2^-126) mul:2", mods="mul:2"),

    # Legacy 0 * anything -> +0 after subnormal input flush
    Case("legacy", "v_mad_legacy_f32", SUBNORM_MIN, POS_INF, NEG_ONE, "subnorm * +inf + -1 -> -1"),
    Case("legacy", "v_mad_legacy_f32", NEG_SUBNORM_MIN, POS_INF, ONE, "-subnorm * +inf + 1 -> 1"),
    Case("legacy", "v_mad_legacy_f32", SUBNORM_MIN, QNAN, ONE, "subnorm * qNaN + 1 -> 1"),
    Case("legacy", "v_mad_legacy_f32", NEG_SUBNORM_MIN, SNAN, TWO, "-subnorm * sNaN + 2 -> 2"),
    Case("legacy", "v_mad_legacy_f32", SUBNORM_MIN, NEG_SUBNORM_MIN, NEG_ZERO, "subnorm * -subnorm + -0 -> +0"),

    # Control cases
    Case("control", "v_mad_f32", TWO, THREE, FOUR, "2*3 + 4 = 10"),
    Case("control", "v_mac_f32", TWO, THREE, FOUR, "mac 2*3 + 4 = 10"),
    Case("control", "v_mad_legacy_f32", TWO, THREE, FOUR, "legacy 2*3 + 4 = 10"),
    Case("control", "v_mad_f32", NEG_ZERO, TWO, NEG_ZERO, "(-0)*2 + (-0) -> -0"),
    Case("control", "v_mad_legacy_f32", NEG_ZERO, TWO, NEG_ZERO, "legacy (-0)*2 + (-0) -> +0"),
    Case("control", "v_mad_f32", POS_INF, ONE, NEG_INF, "+inf + (-inf) -> NaN"),
    Case("control", "v_mad_f32", QNAN, ONE, ONE, "qNaN pass-through"),
)


def _is_nan(word):
    return (word & 0x7F800000) == 0x7F800000 and (word & 0x007FFFFF) != 0


def _equal_word(got, want):
    if _is_nan(got) and _is_nan(want):
        return True
    return got == want


def _insn_lines(case, dst="v0"):
    s0 = "-v1" if case.neg_a else "v1"
    mod_s = f" {case.mods}" if case.mods else ""
    if case.op == "v_mac_f32":
        return [
            f"v_mov_b32 v1, 0x{case.a:x}",
            f"v_mov_b32 v2, 0x{case.b:x}",
            f"v_mov_b32 {dst}, 0x{case.c:x}",
            f"{case.op} {dst}, {s0}, v2{mod_s}",
        ]
    return [
        f"v_mov_b32 v1, 0x{case.a:x}",
        f"v_mov_b32 v2, 0x{case.b:x}",
        f"v_mov_b32 v3, 0x{case.c:x}",
        f"v_mov_b32 {dst}, 0x{SENTINEL:x}",
        f"{case.op} {dst}, {s0}, v2, v3{mod_s}",
    ]


# ---- the translator, on the CPU ----

def _translate(work, name, text):
    os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
    src = os.path.join(work, "disasm", name + ".s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(work, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def ps_listing(case):
    # Bit-exact export: spirv_mini_interp keeps raw f32 bit patterns on %out_mrt0.
    lines = _insn_lines(case, dst="v0") + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return "; mad_denorm_probe\n" + "".join(f"\t{ln}\n" for ln in lines)


def run_translator(case_list=CASES):
    out = []
    with tempfile.TemporaryDirectory(prefix="mad_denorm_probe_") as work:
        for i, c in enumerate(case_list):
            spv = _translate(work, f"ps_mad_denorm_{i}", ps_listing(c))
            out.append(spirv_mini_interp.run_fragment(spv)["%out_mrt0"][0])
    return out


# ---- the hardware ----

def hw_lines(case):
    return _insn_lines(case, dst="v8") + [
        "v_mov_b32 v20, 0",
        "global_store_dword v20, v8, s[2:3]",
        "s_endpgm",
    ]


def run_hw(oracle, case_list, wave, mode):
    from hw_oracle import asm
    got = oracle.run([asm.Kernel(hw_lines(c), wave, 1, 1, raw=True, mode=mode) for c in case_list], words=1)
    return [w if isinstance(w, str) else w[0] for w in got]


def check_mode_denorm_effect(oracle):
    """Sanity check: MODE.denorm32=3 actually keeps a subnormal product on v_mul_f32 while MODE.denorm32=0 flushes it."""
    from hw_oracle import asm
    prog = [
        f"v_mov_b32 v1, 0x{FLT_MIN:x}",
        f"v_mov_b32 v2, 0x{HALF:x}",
        "v_mul_f32 v8, v1, v2",
        "v_mov_b32 v20, 0",
        "global_store_dword v20, v8, s[2:3]",
        "s_endpgm",
    ]
    kernels = [
        asm.Kernel(prog, 32, 1, 1, raw=True, mode=GAME_MODE),
        asm.Kernel(prog, 32, 1, 1, raw=True, mode=dict(GAME_MODE, denorm32=3)),
    ]
    got = oracle.run(kernels, words=1)
    return [w if isinstance(w, str) else w[0] for w in got]


def _has_scale_omod(mods):
    return any(m.startswith(("mul:", "div:")) for m in mods.split())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--oracle", default="translator", help="translator (default) or translator,hw / hw")
    ap.add_argument("--hw-arch", help="override the hardware architecture (LEONIDA_HW_ARCH)")
    a = ap.parse_args(argv)
    sides = {s.strip() for s in a.oracle.split(",") if s.strip()} - {"translator"}
    if sides - {"hw"}:
        ap.error(f"unknown --oracle side(s): {', '.join(sorted(sides - {'hw'}))}")
    try:
        want = run_translator(CASES)
        # Under MODE.IEEE=1, AMD hardware ignores VOP3 mul:2/mul:4/div:2 (docs/design.md).
        no_scale_cases = [
            Case(c.group, c.op, c.a, c.b, c.c, c.what, mods=("clamp" if "clamp" in c.mods.split() else ""), neg_a=c.neg_a)
            for c in CASES
        ]
        want_no_scale = run_translator(no_scale_cases)
    except Exception as e:  # noqa: BLE001 - harness error
        print(f"translator side failed: {e}")
        return 2
    groups = ("input", "product", "result", "omod", "legacy", "control")
    print(f"translator: {len(CASES)} cases across {len(groups)} groups")
    if "hw" not in sides:
        for c, w in zip(CASES, want):
            print(f"  [{c.group:7s}] {c.op:16s} {c.what:38s} -> 0x{w:08x}")
        return 0
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    # The probe's three opcodes are GFX10.1 only: on gfx1030 and later the assembler rejects every kernel, which is
    # "this GPU can't run it" (a skip), not a harness error.
    why = asm.valu_skip_reason("v_mad_f32", oracle.arch)
    if why:
        oracle.close()
        print(f"hardware oracle unavailable: {why}")
        return 77
    status = 0
    try:
        print(f"hardware oracle: {oracle.device} ({oracle.arch})")
        sanity = check_mode_denorm_effect(oracle)
        if any(isinstance(x, str) for x in sanity):
            print(f"MODE denorm32 sanity error: {sanity}")
            return 2
        print(f"MODE sanity (v_mul_f32 FLT_MIN * 0.5): denorm32=0 -> 0x{sanity[0]:08x}, denorm32=3 -> 0x{sanity[1]:08x}")
        if sanity != [POS_ZERO, SUBNORM_HALF]:
            print(f"MODE denorm32 sanity failed: expected [0x{POS_ZERO:08x}, 0x{SUBNORM_HALF:08x}], got {[hex(x) for x in sanity]}")
            return 2
        for wave in (32, 64):
            for name, mode in MODES:
                got = run_hw(oracle, CASES, wave, mode)
                errors = [g for g in got if isinstance(g, str)]
                if errors:
                    print(f"wave{wave} {name}: harness error: {errors[0]}")
                    return 2
                expected = [
                    (want_no_scale[i] if (mode["ieee"] == 1 and _has_scale_omod(c.mods)) else want[i])
                    for i, c in enumerate(CASES)
                ]
                diffs = [(c, g, w) for c, g, w in zip(CASES, got, expected) if not _equal_word(g, w)]
                per_group = {
                    gr: (
                        sum(1 for c, g, w in zip(CASES, got, expected) if c.group == gr and _equal_word(g, w)),
                        sum(1 for c in CASES if c.group == gr),
                    )
                    for gr in groups
                }
                summary = ", ".join(f"{gr} {ok}/{tot}" for gr, (ok, tot) in per_group.items())
                note = " [IEEE=1 ignores omod]" if mode["ieee"] == 1 else ""
                print(f"wave{wave} {name}: {len(CASES) - len(diffs)}/{len(CASES)} match ({summary}){note}")
                for c, g, w in diffs:
                    print(f"  DIFF [{c.group}] {c.op} ({c.what}): hw 0x{g:08x}, expected 0x{w:08x}")
                if diffs:
                    status = 1
    finally:
        oracle.close()
    return status


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
