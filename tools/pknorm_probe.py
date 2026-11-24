r"""What do v_cvt_pknorm_i16_f32 / v_cvt_pknorm_u16_f32 give for NaN on RDNA2? A probe, standard library only.

docs/design.md ("v_cvt_pknorm_i16_f32, v_cvt_pknorm_u16_f32" row) says NaN gives 0, as v_cvt_i32_f32 does, and that
this is not measured on hardware (SHADER-339). D[15:0] comes from S0 and D[31:16] from S1, so each case puts a value in
one half and another value in the other; every NaN pattern below appears in both halves.

The translator multiplies in f32 (OpFMul) and then rounds to nearest even, two roundings. NEAR_TIES holds inputs where
that differs by one from rounding the exact x * 32767 (or 65535) once: the f32 product lands exactly on k + 0.5 while
the exact product is just below it (the translator goes up to the even k + 1, single rounding gives k) or just above it
(the translator goes down to the even k, single rounding gives k + 1). The probe reports which rule each side follows.

Sides (each prints the dword per case; the probe compares them):

    translator  tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py, on the CPU. Always runs.
    hw          the native instruction on an AMD GPU (tools/hw_oracle: HIP on Windows, libdrm on Linux), one lane, in
                wave32 and wave64, under each MODE in MODES: the one inferred for the game (IEEE 0, DX10_CLAMP 1, f32
                denormals flushed, f16/f64 kept, nearest) and variants of IEEE, DX10_CLAMP and the f32 denormal mode.

    python -I tools/pknorm_probe.py                 # translator side only
    python -I tools/pknorm_probe.py --oracle hw     # + the AMD GPU in this machine

Exit code: 0 when every hardware word equals the translator's, 1 when one differs, 2 on a harness error, 77 when the
hardware oracle is unavailable. Nothing here reads game files.
Usage: `python -I tools/pknorm_probe.py --help` lists the options.
"""
import argparse
import os
import struct
import sys
import tempfile
import warnings
from fractions import Fraction

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
OPS = ("v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32")


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


NANS = (0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001, 0x7FA00000)
VALUES = tuple(_bits(v) for v in (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 0.75, 0.999, -0.999,
                                   1 / 32767, 1.5 / 32767, 1.5 / 65535, float("inf"), float("-inf"))) + (0x00000001, 0x80000001)
SCALE = {"v_cvt_pknorm_i16_f32": 32767, "v_cvt_pknorm_u16_f32": 65535}
NEAR_TIES = {
    "v_cvt_pknorm_i16_f32": (0x38400180, 0x38E001C0, 0x39300160, 0x3C004101, 0x3C014103, 0x3C024105),
    "v_cvt_pknorm_u16_f32": (0x37C000C0, 0x386000E0, 0x38B000B0, 0x3B008081, 0x3B028083, 0x3B048085),
}
GAME_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round32": 0, "round16_64": 0}
MODES = (
    ("game", GAME_MODE),
    ("ieee1", dict(GAME_MODE, ieee=1)),
    ("dx10clamp0", dict(GAME_MODE, dx10_clamp=0)),
    ("ieee1_dx10clamp0", dict(GAME_MODE, ieee=1, dx10_clamp=0)),
    ("denorm32_kept", dict(GAME_MODE, denorm32=3)),
)


def cases():
    """(op, s0, s1) triples: every NaN in each half next to a finite value, and every value next to a NaN."""
    out = []
    for op in OPS:
        for i, n in enumerate(NANS):
            other = VALUES[i % len(VALUES)]
            out += [(op, n, other), (op, other, n)]
        for i, v in enumerate(VALUES):
            out.append((op, v, NANS[i % len(NANS)]))
            out.append((op, v, VALUES[(i * 7 + 3) % len(VALUES)]))
        out.append((op, NANS[0], NANS[4]))
        for i, t in enumerate(NEAR_TIES[op]):
            out += [(op, t, VALUES[i]), (op, VALUES[i], t)]
    return out


def _f32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _round_even(q):
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2):
        n += 1
    return n


def single_half(op, word):
    """One half under single rounding: NaN gives 0, else clamp, the exact product, round to nearest even once."""
    x = _f32(word)
    if x != x:
        return 0
    lo = -1 if op == "v_cvt_pknorm_i16_f32" else 0
    q = Fraction(min(max(x, lo), 1))
    return _round_even(q * SCALE[op]) & 0xFFFF


def single_word(op, s0, s1):
    return single_half(op, s0) | single_half(op, s1) << 16


def group(case):
    _, s0, s1 = case
    if s0 in NANS or s1 in NANS:
        return "nan"
    if any(w in NEAR_TIES[case[0]] for w in (s0, s1)):
        return "near_tie"
    return "other"


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


def ps_listing(op, s0, s1):
    """A pixel shader that runs one case. D leaves as two 16-bit halves ORed into 0x40000000, so no export channel is a
    NaN or denormal pattern a float round trip could change (as tests/semantic/test_alu_cvt_pknorm_f32.py does)."""
    body = [f"v_mov_b32 v1, 0x{s0:x}", f"v_mov_b32 v2, 0x{s1:x}", "v_mov_b32 v0, 0xdeadbeef", f"{op} v0, v1, v2",
            "v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0", "v_or_b32 v5, 0x40000000, v5",
            "v_or_b32 v6, 0x40000000, v6", "exp mrt0 v5, v6, v5, v5 done vm", "s_endpgm"]
    return "; pknorm_probe\n" + "".join(f"\t{ln}\n" for ln in body)


def run_translator(case_list):
    out = []
    with tempfile.TemporaryDirectory(prefix="pknorm_probe_") as work:
        for i, (op, s0, s1) in enumerate(case_list):
            spv = _translate(work, f"ps_pknorm_{i}", ps_listing(op, s0, s1))
            lo, hi = spirv_mini_interp.run_fragment(spv)["%out_mrt0"][:2]
            out.append((lo & 0xFFFF) | (hi & 0xFFFF) << 16)
    return out


# ---- the hardware ----

def hw_lines(op, s0, s1):
    return [f"v_mov_b32 v1, 0x{s0:x}", f"v_mov_b32 v2, 0x{s1:x}", "v_mov_b32 v8, 0xdeadbeef", f"{op} v8, v1, v2",
            "v_mov_b32 v20, 0", "global_store_b32 v20, v8, s[2:3]", "s_endpgm"]


def run_hw(oracle, case_list, wave, mode):
    from hw_oracle import asm
    got = oracle.run([asm.Kernel(hw_lines(*c), wave, 1, 1, raw=True, mode=mode) for c in case_list], words=1)
    return [w if isinstance(w, str) else w[0] for w in got]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--oracle", default="translator", help="translator (default) or translator,hw / hw")
    ap.add_argument("--hw-arch", help="override the hardware architecture (LEONIDA_HW_ARCH)")
    a = ap.parse_args(argv)
    sides = {s.strip() for s in a.oracle.split(",") if s.strip()} - {"translator"}
    if sides - {"hw"}:
        ap.error(f"unknown --oracle side(s): {', '.join(sorted(sides - {'hw'}))}")
    case_list = cases()
    try:
        want = run_translator(case_list)
    except Exception as e:  # noqa: BLE001 - a harness error, not a finding
        print(f"translator side failed: {e}")
        return 2
    groups = [group(c) for c in case_list]
    counts = {g: groups.count(g) for g in ("nan", "near_tie", "other")}
    print(f"translator: {len(case_list)} cases ({counts['nan']} with a NaN operand, {counts['near_tie']} near a tie, "
          f"{counts['other']} other)")
    single = [single_word(*c) for c in case_list]
    t_single = sum(1 for g, w, s in zip(groups, want, single) if g == "near_tie" and w == s)
    print(f"translator near ties: {t_single} of {counts['near_tie']} equal single rounding of the exact product")
    if "hw" not in sides:
        for (op, s0, s1), w in zip(case_list, want):
            print(f"  {op} {s0:08x} {s1:08x} -> {w:08x}")
        return 0
    from hw_oracle import runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    status = 0
    try:
        print(f"hardware oracle: {oracle.device} ({oracle.arch})")
        for wave in (32, 64):
            for name, mode in MODES:
                got = run_hw(oracle, case_list, wave, mode)
                diffs = [(c, g, w) for c, g, w in zip(case_list, got, want) if g != w]
                errors = [g for g in got if isinstance(g, str)]
                if errors:
                    print(f"wave{wave} {name}: harness error: {errors[0]}")
                    return 2
                same = {k: sum(1 for gr, g, w in zip(groups, got, want) if gr == k and g == w) for k in counts}
                hw_single = sum(1 for gr, g, s in zip(groups, got, single) if gr == "near_tie" and g == s)
                nan_halves = [(g >> (16 * h)) & 0xFFFF for (_, s0, s1), g in zip(case_list, got)
                              for h, s in enumerate((s0, s1)) if s in NANS]
                print(f"wave{wave} {name}: NaN halves giving 0: {nan_halves.count(0)}/{len(nan_halves)}")
                print(f"wave{wave} {name}: equal to the translator: NaN {same['nan']}/{counts['nan']}, "
                      f"near ties {same['near_tie']}/{counts['near_tie']}, other {same['other']}/{counts['other']}; "
                      f"near ties equal to single rounding: {hw_single}/{counts['near_tie']}")
                for (op, s0, s1), g, w in diffs:
                    print(f"  {op} {s0:08x} {s1:08x}: hw {g:08x}, translator {w:08x}")
                if diffs:
                    status = 1
    finally:
        oracle.close()
    return status


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
