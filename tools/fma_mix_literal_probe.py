r"""Does a v_fma_mix source with op_sel_hi and op_sel set read the high half of a 32-bit literal? A probe, standard
library only.

A v_fma_mix* source read as f16 (op_sel_hi[i] = 1) takes the half op_sel[i] names. When two sources share one 32-bit
literal, the console compiler (in the FidelityFX DLL's FSPX code) and Mesa's ACO (`optimize_constants()` in
src/amd/compiler/aco_optimizer.cpp) both put two f16 values in it and give one source the high half. llvm-mc prints
such a literal as its low 16 bits, so the translator takes it from the blob (docs/design.md, "Printed encodings").
That the hardware reads the high half that way is the part no one has measured. Two readings:

    halves    op_sel picks the half of the 32-bit literal (what the translator does with the blob)
    low       the literal is 16 bits, its high half reads +0.0 (what llvm-mc's text says)

CASES (values made up for the probe; f16 halves written hi:lo):

    A  src0 = literal.lo, src2 = literal.hi       op_sel:[0,0,1] op_sel_hi:[1,0,1]   the shape seen in FSPX code
    B  src0 = literal.hi, src2 = literal.hi       op_sel:[1,0,1] op_sel_hi:[1,0,1]
    E  src0 = v (f32),    src2 = literal.hi       op_sel:[0,0,1] op_sel_hi:[0,0,1]
    C  src0 = literal.lo, src2 = literal.lo       control: both readings agree
    D  src0 = literal as f32, src2 = v (f32)      control: an f32 literal is the whole 32 bits

src1 is always a VGPR holding 2.0 (f32). Each case computes fma(src0, src1, src2) with values that are exact in f32,
so rounding plays no part.

Sides:
    translator  tools/rdna2spirv.py on a listing whose printed encodings are llvm-mc's (the literal cut to 16 bits)
                with the real bytes next to it, run in tests/semantic/spirv_mini_interp.py on the CPU. Always runs.
                The text and encodings are what llvm-mc 23.1.5 prints for these bytes, except D's text: it prints
                that f32 literal as lit(0x40400000), which the translator does not parse, so D's listing says
                0x40400000.
    hw          the instruction's exact bytes (`.long`, since LLVM's assembler can't encode the literal either) on an
                AMD GPU through tools/hw_oracle (HIP on Windows, libdrm on Linux), one wave32 lane. The measurement.

    python -I tools/fma_mix_literal_probe.py                    # translator side only
    python -I tools/fma_mix_literal_probe.py --oracle hw        # + the AMD GPU in this machine

Exit code: 0 when every side that ran follows the same reading, 1 when they differ or a side follows neither (or a
control is wrong), 2 on a harness error, 77 when the hardware side is unavailable. Nothing here reads game files.
Usage: `python -I tools/fma_mix_literal_probe.py --help` lists the options.
"""
import argparse
import os
import struct
import subprocess
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
SENTINEL = 0x5A5A5A5A
HALVES, LOW = "halves", "low"
OP_FMA_MIX_F32 = 0x20             # VOP3P opcode, as in the console's bytes and LLVM's encoding
LITERAL = 0xFF                    # SRC field value for a literal
V2_VALUE = 2.0
# The game's MODE as the other probes infer it (fma_mix_probe.py): IEEE 0, DX10_CLAMP 1, f32 denormals flushed,
# f16 kept.
HW_MODE = {"ieee": 0, "dx10_clamp": 1, "denorm32": 0, "denorm16_64": 3, "round16_64": 0}


def h(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def f(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Case:
    def __init__(self, name, srcs, op_sel, op_sel_hi, literal, what):
        self.name, self.srcs, self.op_sel, self.op_sel_hi = name, srcs, op_sel, op_sel_hi
        self.literal, self.what = literal, what

    def operand(self, i, reading):
        """Value of source i under a reading; srcs[i] is 'lit' or 'v'."""
        if self.srcs[i] == "v":
            return V2_VALUE
        if not self.op_sel_hi[i]:
            return f(self.literal)                         # f32: the whole literal under either reading
        if self.op_sel[i] == 0:
            return h(self.literal)
        return h(self.literal >> 16) if reading == HALVES else 0.0

    def expected(self, reading):
        return f2u(self.operand(0, reading) * V2_VALUE + self.operand(2, reading))

    def printed_literal(self):
        """What llvm-mc re-encodes: the literal's low 16 bits. llvm-mc 23.1.5 does that for every v_fma_mix literal,
        one that a source reads as f32 (D) too."""
        return self.literal & 0xFFFF

    def words(self, vdst, vsrc1, literal):
        """The instruction's three dwords: VOP3P word 0, word 1, literal."""
        src = [LITERAL if self.srcs[0] == "lit" else 256 + vsrc1, 256 + vsrc1,
               LITERAL if self.srcs[2] == "lit" else 256 + vsrc1]
        w0 = (0xCC000000 | OP_FMA_MIX_F32 << 16 | self.op_sel_hi[2] << 14 | self.op_sel[2] << 13 |
              self.op_sel[1] << 12 | self.op_sel[0] << 11 | vdst)
        w1 = self.op_sel_hi[1] << 28 | self.op_sel_hi[0] << 27 | src[2] << 18 | src[1] << 9 | src[0]
        return w0, w1, literal

    def text(self, vdst, vsrc1):
        f16 = any(self.srcs[i] == "lit" and self.op_sel_hi[i] for i in (0, 2))
        lit = f"0x{self.printed_literal() if f16 else self.literal:x}"      # D: hand-written, see the docstring
        reg = f"v{vsrc1}"
        ops = [lit if self.srcs[0] == "lit" else reg, reg, lit if self.srcs[2] == "lit" else reg]
        mods = [f"{name}:[{','.join(map(str, bits))}]" for name, bits in (("op_sel", self.op_sel),
                                                                          ("op_sel_hi", self.op_sel_hi)) if any(bits)]
        return " ".join([f"v_fma_mix_f32 v{vdst}, {ops[0]}, {ops[1]}, {ops[2]}"] + mods)  # llvm-mc omits [0,0,0]


CASES = (
    Case("A", ("lit", "v", "lit"), (0, 0, 1), (1, 0, 1), 0x40004200, "src0 = lo, src2 = hi (the FSPX shape)"),
    Case("B", ("lit", "v", "lit"), (1, 0, 1), (1, 0, 1), 0x42004500, "src0 = hi, src2 = hi"),
    Case("E", ("v", "v", "lit"), (0, 0, 1), (0, 0, 1), 0xBC004200, "src0 = v (f32), src2 = hi"),
    Case("C", ("lit", "v", "lit"), (0, 0, 0), (1, 0, 1), 0x40004200, "control: src0 = src2 = lo"),
    Case("D", ("lit", "v", "v"), (0, 0, 0), (0, 0, 0), 0x40400000, "control: src0 = the f32 literal"),
)


def is_control(case):
    return case.expected(HALVES) == case.expected(LOW)


def verdict(results):
    """(reading or None, report lines) for [(case, word or None)]."""
    readings, lines = set(), []
    for case, word in results:
        want = {r: case.expected(r) for r in (HALVES, LOW)}
        if word is None:
            got, tag = "no result", "missing"
        else:
            got = f"{f(word):g}"
            tag = [r for r in (HALVES, LOW) if want[r] == word]
            tag = ("both (control)" if is_control(case) else tag[0]) if tag else "neither"
        if tag in (HALVES, LOW):
            readings.add(tag)
        elif tag != "both (control)":
            readings.add(None)
        lines.append(f"  {case.name} {got:<10} halves {f(want[HALVES]):g}  low {f(want[LOW]):g}  -> {tag:<16} "
                     f"{case.what}")
    reading = readings.pop() if len(readings) == 1 else None
    return reading, lines


# ---- the translator, on the CPU ----

VOP1_MOV = 0x7E000200             # v_mov_b32 (VOP1 opcode 1); | vdst << 17 | src0
EXP_MRT0_V0 = "0f 18 00 f8 00 00 00 00"
S_ENDPGM = "00 00 81 bf"
CODE_START = 0x100


def hexs(*dwords):
    return " ".join(f"{b:02x}" for w in dwords for b in struct.pack("<I", w))


def ps_program(case):
    """(listing text, blob): llvm-mc's text and encodings for the case, and the real bytes at CODE_START."""
    mov2 = (f"v_mov_b32_e32 v2, 0x{f2u(V2_VALUE):x}", hexs(VOP1_MOV | 2 << 17 | LITERAL, f2u(V2_VALUE)))
    mov0 = (f"v_mov_b32_e32 v0, 0x{SENTINEL:x}", hexs(VOP1_MOV | 0 << 17 | LITERAL, SENTINEL))
    printed = (case.text(0, 2), hexs(*case.words(0, 2, case.printed_literal())))
    stored = (case.text(0, 2), hexs(*case.words(0, 2, case.literal)))
    tail = [("exp mrt0 v0, v0, v0, v0 done vm", EXP_MRT0_V0), ("s_endpgm", S_ENDPGM)]
    shown, real = [mov2, mov0, printed] + tail, [mov2, mov0, stored] + tail
    size = sum(len(bytes.fromhex(e)) for _, e in shown)
    lines = [f"; fma_mix_literal_{case.name}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (probe), gfx1030, "
             "0 invalid words"]
    for text, enc in shown:
        lines.append(f"\t{text:<48}; encoding: [{','.join('0x' + b for b in enc.split())}]")
    blob = bytes(CODE_START) + b"".join(bytes.fromhex(e) for _, e in real)
    return "\n".join(lines) + "\n", blob


def run_translator(cases=CASES):
    out = []
    with tempfile.TemporaryDirectory(prefix="fma_mix_literal_probe_") as work:
        for case in cases:
            name = f"ps_fma_mix_literal_{case.name}"
            text, blob = ps_program(case)
            os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
            os.makedirs(os.path.join(work, "fssm"), exist_ok=True)
            src = os.path.join(work, "disasm", name + ".s")
            with open(src, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            with open(os.path.join(work, name + ".dxbc"), "wb") as fh:
                fh.write(STUB_DXBC)
            with open(os.path.join(work, "fssm", name + ".fssm"), "wb") as fh:
                fh.write(blob)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
            out.append(spirv_mini_interp.run_fragment(spv)["%out_mrt0"][0])
    return out


# ---- the native instruction on an AMD GPU ----

def hw_lines(case):
    w0, w1, lit = case.words(8, 2, case.literal)
    return [f"v_mov_b32 v2, 0x{f2u(V2_VALUE):x}", f"v_mov_b32 v8, 0x{SENTINEL:x}",
            f".long 0x{w0:08x}, 0x{w1:08x}, 0x{lit:08x}", "v_mov_b32 v20, 0",
            "global_store_b32 v20, v8, s[2:3]", "s_endpgm"]


def run_hw(oracle, cases=CASES):
    from hw_oracle import asm
    got = oracle.run([asm.Kernel(hw_lines(c), 32, 1, 1, raw=True, mode=HW_MODE) for c in cases], words=1)
    return [None if isinstance(w, str) else w[0] for w in got], [w for w in got if isinstance(w, str)]


# ---- driver ----

def report(side, results):
    reading, lines = verdict(results)
    print(f"{side}: {'follows the ' + reading + ' reading' if reading else 'follows neither reading'}")
    for ln in lines:
        print(ln)
    return reading


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--oracle", default="translator", help="comma-separated sides besides the translator: hw")
    ap.add_argument("--hw-arch", help="GPU architecture to assemble for (default: detected; LEONIDA_HW_ARCH)")
    a = ap.parse_args(argv)
    sides = {s.strip() for s in a.oracle.split(",") if s.strip()} - {"translator"}
    if sides - {"hw"}:
        ap.error(f"unknown --oracle side(s): {', '.join(sorted(sides - {'hw'}))}")
    for case in CASES:                  # the table itself: three cases that tell the readings apart, two that don't
        if is_control(case) != case.what.startswith("control"):
            print(f"harness error: case {case.name} does not discriminate as described")
            return 2
    readings = []
    try:
        readings.append(report("translator", list(zip(CASES, run_translator()))))
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
            readings.append(report(f"hw (native, {oracle.arch})", list(zip(CASES, words))))
    except (RuntimeError, OSError, subprocess.SubprocessError, spirv_mini_interp.UnsupportedConstruct,
            rdna2spirv.Unsupported) as e:
        print(f"harness error: {e}")
        return 2
    if None in readings or len(set(readings)) != 1:
        print("sides disagree" if None not in readings else "a side follows neither reading")
        return 1
    print(f"all sides follow the {readings[0]} reading")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
