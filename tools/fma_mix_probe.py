r"""Does v_fma_mixlo_f16 / v_fma_mixhi_f16 round once or twice to nearest even? A probe, standard library only.

docs/design.md (Known approximations) leaves this open. The translator computes the f32 `Fma` and narrows that to
f16 ("double": two roundings, f32 then f16). The hardware may instead round the exact x * y + z once, straight to f16
("single"). The RX 6800 word behind the toward-zero path (1 * 2 - 2^-30 -> 0x3fff) can't tell them apart in nearest
mode: there both give 0x4000, and toward zero, two truncations equal one.

The two rules differ only when the f32 rounding lands exactly on an f16 tie (or moves the value off one). CASES holds
seven such inputs, each one f16 ulp apart under the two rules, and one control where both agree:

    A  1 * 2^-40 + (1 + 2^-11)             just above a tie, f32 rounds onto it, ties-to-even goes down
    B  1 * -2^-40 + (1 + 3 * 2^-11)        just below a tie with an odd lower neighbour: double goes up
    C  -1 * 2^-40 - (1 + 2^-11)            A negated
    D  2^-30 * 2^-30 + 2^-25               the tie between 0 and the smallest f16 subnormal (needs f16 denormals)
    E  1 * -2^-20 + 65520                  the overflow tie: double gives inf, single 65504
    F  (1 + 2^-12) * (1 + 2^-12) + 0       the tie comes from the product (1 + 2^-11 + 2^-24), not from z
    G  f16 operands: (1 + 2^-7) * (2^-11 - 2^-18) + (1 + 2^-10)    as B, through op_sel_hi:[1,1,1]
    H  1 * 2 + 0.5 = 2.5                   control

Sides (each prints the f16 word per case and the rule its words follow):

    translator  tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py, on the CPU. Always runs.
    hw          the native instruction on an AMD GPU (tools/hw_oracle: HIP on Windows, libdrm on Linux), one wave32
                lane, MODE as inferred for the game (IEEE 0, DX10_CLAMP 1, f32 denormals flushed, f16 kept, nearest).
                This is the measurement the docs lack.
    vulkan      the translated compute shader on this machine's Vulkan GPU through vk_compute_test (--build DIR,
                needs spirv-as). The operands come in user SGPRs (VKC_SGPRS), so the driver can't fold them. It checks
                that the driver runs the translator's sequence as written; it says nothing about RDNA2 itself.

    python -I tools/fma_mix_probe.py                                # translator side only
    python -I tools/fma_mix_probe.py --oracle hw                    # + the AMD GPU in this machine
    python -I tools/fma_mix_probe.py --oracle vulkan --build runtime/build

Exit code: 0 when every side that ran follows the same rule, 1 when they differ or a side follows neither rule (or
its control is wrong), 2 on a harness error, 77 when a requested side is unavailable. Nothing here reads game files.
Usage: `python -I tools/fma_mix_probe.py --help` lists the options.
"""
import argparse
import os
import re
import struct
import subprocess
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
SENTINEL = 0x5A5A5A5A            # the destination's previous value: the half the op doesn't write must keep it
SINGLE, DOUBLE = "single", "double"


class Case:
    __slots__ = ("name", "op", "f16", "words", "what")

    def __init__(self, name, op, f16, words, what):
        self.name, self.op, self.f16, self.words, self.what = name, op, f16, words, what

    @property
    def hi(self):
        return self.op == "v_fma_mixhi_f16"

    @property
    def mods(self):
        return " op_sel_hi:[1,1,1]" if self.f16 else ""


# x, y, z as raw words: f32, or (f16 = True) an f16 in the low half.
CASES = (
    Case("A", "v_fma_mixlo_f16", False, (0x3F800000, 0x2B800000, 0x3F801000), "1 * 2^-40 + (1 + 2^-11)"),
    Case("B", "v_fma_mixhi_f16", False, (0x3F800000, 0xAB800000, 0x3F803000), "1 * -2^-40 + (1 + 3 * 2^-11)"),
    Case("C", "v_fma_mixlo_f16", False, (0xBF800000, 0x2B800000, 0xBF801000), "-1 * 2^-40 - (1 + 2^-11)"),
    Case("D", "v_fma_mixhi_f16", False, (0x30800000, 0x30800000, 0x33000000), "2^-30 * 2^-30 + 2^-25"),
    Case("E", "v_fma_mixlo_f16", False, (0x3F800000, 0xB5800000, 0x477FF000), "1 * -2^-20 + 65520"),
    Case("F", "v_fma_mixhi_f16", False, (0x3F800800, 0x3F800800, 0x00000000), "(1 + 2^-12)^2 + 0"),
    Case("G", "v_fma_mixlo_f16", True, (0x3C08, 0x0FF0, 0x3C01), "f16 (1 + 2^-7) * (2^-11 - 2^-18) + (1 + 2^-10)"),
    Case("H", "v_fma_mixhi_f16", False, (0x3F800000, 0x40000000, 0x3F000000), "control: 1 * 2 + 0.5"),
)


# ---- the two rules, in exact arithmetic (finite normal or zero operands only; CASES has nothing else) ----

def operand(word, f16):
    """The exact value of an f32 word, or of the f16 in its low half."""
    if f16:
        e, m, s = (word >> 10) & 0x1F, word & 0x3FF, -1 if word & 0x8000 else 1
        if e == 0x1F:
            raise ValueError(f"f16 0x{word:04x} is not finite")
        return s * (Fraction(m, 1 << 24) if e == 0 else Fraction(0x400 | m, 1 << 10) * Fraction(2) ** (e - 15))
    e, m, s = (word >> 23) & 0xFF, word & 0x7FFFFF, -1 if word >> 31 else 1
    if e == 0xFF or (e == 0 and m):
        raise ValueError(f"f32 0x{word:08x} is not a finite normal or zero")
    return Fraction(0) if e == 0 else s * Fraction(0x800000 | m, 1 << 23) * Fraction(2) ** (e - 127)


def round_rne(x, bits, emin):
    """x rounded to nearest even with `bits` significand bits and subnormals below 2^emin (no overflow)."""
    if x == 0:
        return x
    a, e = abs(x), emin
    while a >= Fraction(2) ** (e + 1):
        e += 1
    q = Fraction(2) ** (e - bits + 1)            # the ulp at a's binade (or the subnormal ulp)
    n = a / q
    k, rem = n.numerator // n.denominator, n - n.numerator // n.denominator
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and k % 2):
        k += 1
    return (k * q) if x > 0 else -(k * q)


def f16_word(x):
    """f16 bits of x rounded to nearest even; |x| >= 65520 is inf."""
    r = round_rne(x, 11, -14)
    s = 0x8000 if x < 0 else 0
    a = abs(r)
    if a > 65504:
        return s | 0x7C00
    if a < Fraction(1, 1 << 14):
        return s | int(a * (1 << 24))
    e = 0
    while a >= Fraction(2) ** (e + 1):
        e += 1
    while a < Fraction(2) ** e:
        e -= 1
    return s | (e + 15) << 10 | int((a / Fraction(2) ** e - 1) * 1024)


def expected(case):
    """{"single": word, "double": word}: the f16 result under each rule."""
    x, y, z = (operand(w, case.f16) for w in case.words)
    exact = x * y + z
    return {SINGLE: f16_word(exact), DOUBLE: f16_word(round_rne(exact, 24, -126))}


def half(case, word):
    """The f16 a case wrote, or None when the other half of the destination lost the sentinel."""
    keep, mine = (word & 0xFFFF, word >> 16) if case.hi else (word >> 16, word & 0xFFFF)
    other = SENTINEL & 0xFFFF
    return mine if keep == other else None


def verdict(results):
    """results: [(case, word or None)]. Returns (rule or None, one line per case). The rule is the one every
    discriminating case follows, with the control right; None when the words follow neither or a mix."""
    rules, lines = set(), []
    for case, word in results:
        want = expected(case)
        h = None if word is None else half(case, word)
        if word is None:
            got, tag = "no result", "missing"
        elif h is None:
            got, tag = f"0x{word:08x}", "other half not kept"
        else:
            got = f"0x{h:04x}"
            tag = [r for r in (SINGLE, DOUBLE) if want[r] == h]
            tag = ("both (control)" if want[SINGLE] == want[DOUBLE] else tag[0]) if tag else "neither"
        if tag in (SINGLE, DOUBLE):
            rules.add(tag)
        elif tag != "both (control)":
            rules.add(None)
        lines.append(f"  {case.name} {case.op:<16} {got:<12} single 0x{want[SINGLE]:04x}  double 0x{want[DOUBLE]:04x}"
                     f"  -> {tag:<16} {case.what}")
    rule = rules.pop() if len(rules) == 1 else None
    return rule, lines


# ---- the translator, on the CPU ----

def _translate(work, name, text, fssm=None):
    os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
    src = os.path.join(work, "disasm", name + ".s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(work, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    if fssm is not None:
        os.makedirs(os.path.join(work, "fssm"), exist_ok=True)
        with open(os.path.join(work, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def ps_listing(case):
    """A pixel shader that runs one case and exports the destination word as mrt0.x."""
    x, y, z = case.words
    body = [f"v_mov_b32 v1, 0x{x:x}", f"v_mov_b32 v2, 0x{y:x}", f"v_mov_b32 v3, 0x{z:x}",
            f"v_mov_b32 v0, 0x{SENTINEL:x}", f"{case.op} v0, v1, v2, v3{case.mods}", "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm"]
    return "; fma_mix_probe\n" + "".join(f"\t{ln}\n" for ln in body)


def run_translator(cases=CASES):
    out = []
    with tempfile.TemporaryDirectory(prefix="fma_mix_probe_") as work:
        for case in cases:
            spv = _translate(work, f"ps_fma_mix_{case.name}", ps_listing(case))
            out.append(spirv_mini_interp.run_fragment(spv)["%out_mrt0"][0])
    return out


# ---- the translated compute shader on a Vulkan GPU ----
# s[0:3] is the output V#, then three operand words per case in s4.. (VKC_SGPRS), the workgroup id after them.

FIRST_RESULT = 10


def cs_rsrc2(cases):
    user = 4 + 3 * len(cases)
    if user > 32:
        raise ValueError("too many cases for the user SGPRs")
    return 0x8080 | user << 1            # USER_SGPR [5:1], TGID_X_EN, 512 bytes of LDS (as build.py's default)


def cs_listing(cases=CASES):
    """One thread runs every case from its SGPR operands and stores the destination words at offset 0, 4, ..."""
    body = []
    for k, case in enumerate(cases):
        s = 4 + 3 * k
        d = FIRST_RESULT + k
        body += [f"v_mov_b32 v1, s{s}", f"v_mov_b32 v2, s{s + 1}", f"v_mov_b32 v3, s{s + 2}",
                 f"v_mov_b32 v{d}, 0x{SENTINEL:x}", f"{case.op} v{d}, v1, v2, v3{case.mods}"]
    for k in range(len(cases)):
        body += [f"v_mov_b32 v4, {4 * k}", f"buffer_store_dword v{FIRST_RESULT + k}, v4, s[0:3], 0 offen"]
    body.append("s_endpgm")
    return "; fma_mix_probe\n" + "".join(f"\t{ln}\n" for ln in body)


def cs_fssm(cases=CASES):
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, cs_rsrc2(cases), 1, 1, 1)
    return bytes(header)


def sgpr_words(cases=CASES):
    return [w for case in cases for w in case.words]


def translate_cs(work, cases=CASES):
    return _translate(work, "cs_fma_mix_probe", cs_listing(cases), cs_fssm(cases))


def run_vulkan(build, cases=CASES):
    """[word per case] from vk_compute_test, or raise RuntimeError with the reason (Unavailable-like: FileNotFound)."""
    sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
    from compute_fixture import find_test_program
    folder = find_test_program(build, "vk_compute_test")
    if folder is None:
        raise FileNotFoundError(f"vk_compute_test is not built in {build}")
    exe = os.path.join(folder, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    if len(cases) > 8:
        raise ValueError("vk_compute_test prints eight words")
    with tempfile.TemporaryDirectory(prefix="fma_mix_probe_") as work:
        text = translate_cs(work, cases)
        sa, sv = os.path.join(work, "cs.spvasm"), os.path.join(work, "cs.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        try:
            r = subprocess.run(["spirv-as", sa, "-o", sv, "--target-env", "vulkan1.2"], capture_output=True, text=True)
        except FileNotFoundError:
            raise FileNotFoundError("spirv-as is not on PATH (SPIRV-Tools)") from None
        if r.returncode:
            raise RuntimeError("spirv-as failed: " + (r.stderr or r.stdout).strip()[:300])
        env = dict(os.environ, VKC_SGPRS=",".join(f"0x{w:x}" for w in sgpr_words(cases)))
        r = subprocess.run([exe, sv, "1", "dump"], capture_output=True, text=True, env=env, timeout=120)
    msg = r.stdout + r.stderr
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", msg, re.MULTILINE)
    if r.returncode == 77:
        raise FileNotFoundError("vk_compute_test: no usable Vulkan device (exit 77)")
    if r.returncode or not m:
        raise RuntimeError(f"vk_compute_test exited {r.returncode}: {msg.strip()[-300:]}")
    return [int(x) for x in m.group(1).split()][:len(cases)]


# ---- the native instruction on an AMD GPU ----

HW_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round16_64": 0}


def hw_lines(case):
    x, y, z = case.words
    return [f"v_mov_b32 v1, 0x{x:x}", f"v_mov_b32 v2, 0x{y:x}", f"v_mov_b32 v3, 0x{z:x}",
            f"v_mov_b32 v8, 0x{SENTINEL:x}", f"{case.op} v8, v1, v2, v3{case.mods}", "v_mov_b32 v20, 0",
            "global_store_b32 v20, v8, s[2:3]", "s_endpgm"]


def run_hw(oracle, cases=CASES):
    from hw_oracle import asm
    got = oracle.run([asm.Kernel(hw_lines(c), 32, 1, 1, raw=True, mode=HW_MODE) for c in cases], words=1)
    return [None if isinstance(w, str) else w[0] for w in got], [w for w in got if isinstance(w, str)]


# ---- driver ----

def report(side, results):
    rule, lines = verdict(results)
    print(f"{side}: {'follows ' + rule + ' rounding' if rule else 'follows neither rule'}")
    for ln in lines:
        print(ln)
    return rule


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated sides besides the translator, which always runs: hw, vulkan")
    ap.add_argument("--build", help="the runtime build folder holding vk_compute_test (for --oracle vulkan)")
    ap.add_argument("--hw-arch", help="GPU architecture to assemble for (default: detected; LEONIDA_HW_ARCH)")
    a = ap.parse_args(argv)
    sides = {s.strip() for s in a.oracle.split(",") if s.strip()} - {"translator"}
    if sides - {"hw", "vulkan"}:
        ap.error(f"unknown --oracle side(s): {', '.join(sorted(sides - {'hw', 'vulkan'}))}")
    if "vulkan" in sides and not a.build:
        ap.error("--oracle vulkan needs --build DIR")
    for case in CASES:                  # the table itself: seven inputs that tell the rules apart, one that doesn't
        want = expected(case)
        if (want[SINGLE] == want[DOUBLE]) != case.what.startswith("control"):
            print(f"harness error: case {case.name} does not discriminate as described")
            return 2
    rules = []
    try:
        rules.append(report("translator", list(zip(CASES, run_translator()))))
        if "vulkan" in sides:
            try:
                words = run_vulkan(a.build)
            except FileNotFoundError as e:
                print(f"vulkan side unavailable: {e}")
                return 77
            rules.append(report("vulkan (translated, this GPU)", list(zip(CASES, words))))
        if "hw" in sides:
            from hw_oracle import runner
            try:
                oracle = runner.open_oracle(arch=a.hw_arch)
            except runner.Unavailable as e:
                print(f"hardware oracle unavailable: {e}")
                return 77
            try:
                print(f"hardware oracle: {oracle.device}")
                words, rejected = run_hw(oracle)
            finally:
                oracle.close()
            for why in rejected:
                print(f"  {why}")
            rules.append(report(f"hw (native, {oracle.arch})", list(zip(CASES, words))))
    except (RuntimeError, OSError, subprocess.SubprocessError, spirv_mini_interp.UnsupportedConstruct) as e:
        print(f"harness error: {e}")
        return 2
    if None in rules or len(set(rules)) != 1:
        print("sides disagree" if None not in rules else "a side follows neither rule")
        return 1
    print(f"all sides follow {rules[0]} rounding")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
