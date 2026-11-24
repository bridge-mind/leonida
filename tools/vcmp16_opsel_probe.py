r"""TEST-650: measure on an AMD GPU (tools/hw_oracle) whether VOP3 op_sel picks the half of each source on the 16-bit
vector compares v_cmp_{lt,eq,gt}_{f16,u16,i16}_e64.

    python -I tools/vcmp16_opsel_probe.py                # the two hypotheses' answers only (no GPU)
    python -I tools/vcmp16_opsel_probe.py --oracle hw    # native AMD GPU, wave32 and wave64

The docs/design.md row "16-bit vector compare op_sel" says op_sel:[s0,s1] selects bits [15:0] (0) or [31:16] (1)
of each source before neg/abs (f16) or zero/sign extension (u16/i16), from the ISA text and ACO; it was not
measured. llvm-mc rejects op_sel on these compares and drops the bits when disassembling, so no listing can carry
it: the compares are written here as raw words. Their encodings are llvm-mc's (gfx1036) for
`v_cmp_<op>_e64 s10, v1, v2`, with OPSEL set by hand in word 0 [12:11] (src1, src0) and NEG in word 1 [30:29].

Each row loads v1 and v2 with two halves, runs one compare and stores bit 0 of s10 (lane 0's result). The rows'
values are chosen so that honouring op_sel and ignoring it (reading the low halves) give different answers on most
of them. Standard library only, no game files.

Exit codes: 0 every row matches op_sel being ignored (the gfx1036 measurement), 1 a row differs, 2 harness error, 77 no hardware oracle.
Usage: `python -I tools/vcmp16_opsel_probe.py --help` lists the options.
"""
import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hw_oracle"))

# VOP3 opcode byte (word 0 bits [23:16]) of v_cmp_<op>_e64, from llvm-mc 21.1.8 -mcpu=gfx1036.
OPCODES = {("lt", "f16"): 0xC9, ("eq", "f16"): 0xCA, ("gt", "f16"): 0xCC,
           ("lt", "u16"): 0xA9, ("eq", "u16"): 0xAA, ("gt", "u16"): 0xAC,
           ("lt", "i16"): 0x89, ("gt", "i16"): 0x8C}
SDST = 10                 # s10 (s[10:11] in wave64); lane 0 is bit 0 either way
SRC0, SRC1 = 0x101, 0x102  # v1, v2
OPSEL_SRC0, OPSEL_SRC1 = 1 << 11, 1 << 12
NEG_SRC0 = 1 << 29


def words(op, kind, opsel, neg0=False):
    """The two dwords of v_cmp_<op>_<kind>_e64 s10, v1, v2 with op_sel:[s0, s1] and neg on src0."""
    w0 = 0xD4000000 | OPCODES[(op, kind)] << 16 | SDST
    w0 |= (OPSEL_SRC0 if opsel[0] else 0) | (OPSEL_SRC1 if opsel[1] else 0)
    w1 = SRC0 | SRC1 << 9 | (NEG_SRC0 if neg0 else 0)
    return w0, w1


def half_value(bits, kind):
    if kind == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if kind == "i16":
        return bits - 0x10000 if bits & 0x8000 else bits
    return bits


def compare(op, a, b):
    return int({"lt": a < b, "eq": a == b, "gt": a > b}[op])


def answer(row, honour_opsel):
    """Lane 0's result if op_sel is honoured, or if the low halves are read whatever it says."""
    _name, op, kind, (v1, v2), opsel, neg0 = row
    s0 = (v1 >> 16 if honour_opsel and opsel[0] else v1) & 0xFFFF
    s1 = (v2 >> 16 if honour_opsel and opsel[1] else v2) & 0xFFFF
    a, b = half_value(s0, kind), half_value(s1, kind)
    if neg0:
        a = -a
    return compare(op, a, b)


def pack(hi, lo):
    return (hi & 0xFFFF) << 16 | (lo & 0xFFFF)


F1, F2, F3, F4, FH = 0x3C00, 0x4000, 0x4200, 0x4400, 0x3800   # 1.0 2.0 3.0 4.0 0.5


# (name, op, kind, (v1, v2), op_sel (src0, src1), neg on src0)
def rows():
    out = []
    # f16, set A: v1 = (hi 3.0, lo 1.0), v2 = (hi 1.0, lo 2.0); set B: v2 = (hi 2.0, lo 4.0)
    for setname, v1, v2 in (("A", pack(F3, F1), pack(F1, F2)), ("B", pack(F3, F1), pack(F2, F4))):
        for opsel in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for op in ("lt", "eq", "gt"):
                out.append((f"f16{setname}_{op}_opsel{opsel[0]}{opsel[1]}", op, "f16", (v1, v2), opsel, False))
    # neg applies to the selected half: -(3.0) < 2.0 is true; -(1.0) < 2.0 too, so pick v2 low 0.5 to split them
    out.append(("f16_neg_hi_lt", "lt", "f16", (pack(F3, F1), pack(F1, FH)), (1, 0), True))    # -3 < 0.5: 1
    out.append(("f16_neg_hi_gt", "gt", "f16", (pack(FH, F3), pack(F1, F1)), (1, 0), True))    # -0.5 > 1: 0, lo -3 > 1: 0
    # u16: high halves 0x8000 / 0x0003, low halves 5 / 7
    for opsel in ((0, 0), (1, 0), (0, 1), (1, 1)):
        for op in ("lt", "eq", "gt"):
            out.append((f"u16_{op}_opsel{opsel[0]}{opsel[1]}", op, "u16", (pack(0x8000, 5), pack(3, 7)), opsel, False))
    # i16: the same words, so a high 0x8000 is -32768 (sign extension of the selected half)
    for opsel in ((0, 0), (1, 0), (0, 1), (1, 1)):
        for op in ("lt", "gt"):
            out.append((f"i16_{op}_opsel{opsel[0]}{opsel[1]}", op, "i16", (pack(0x8000, 5), pack(3, 7)), opsel, False))
    # i16 set B: low half 9 against 7, so reading src0's high half (-32768) instead flips the result for op_sel 10/11
    for opsel in ((1, 0), (1, 1)):
        for op in ("lt", "gt"):
            out.append((f"i16B_{op}_opsel{opsel[0]}{opsel[1]}", op, "i16", (pack(0x8000, 9), pack(3, 7)), opsel, False))
    return out


# Control: the same OPSEL bits hand-set on v_add_nc_u16_e64 v8, v1, v2 (llvm-mc puts op_sel:[1,0,0] at word 0 bit 11
# and [0,1,0] at bit 12 there, encoding d7030000 + vdst). If the hardware honours them here and not on the compares,
# the bit positions are right and the compares ignore them.
ADD_NC_U16 = 0xD7030000
CONTROL_V1, CONTROL_V2 = pack(0x0300, 0x0005), pack(0x0040, 0x0007)   # hi 0x300 / 0x40, lo 5 / 7


def control_rows():
    return [(f"control_add_nc_u16_opsel{s0}{s1}", (s0, s1)) for s0, s1 in ((0, 0), (1, 0), (0, 1), (1, 1))]


def control_answer(opsel, honour_opsel):
    a = (CONTROL_V1 >> 16 if honour_opsel and opsel[0] else CONTROL_V1) & 0xFFFF
    b = (CONTROL_V2 >> 16 if honour_opsel and opsel[1] else CONTROL_V2) & 0xFFFF
    return (a + b) & 0xFFFF


def control_kernel(start):
    out = []
    for i, (_name, opsel) in enumerate(control_rows()):
        w0 = ADD_NC_U16 | 8 | (OPSEL_SRC0 if opsel[0] else 0) | (OPSEL_SRC1 if opsel[1] else 0)
        w1 = SRC0 | SRC1 << 9
        out += [f"v_mov_b32 v1, 0x{CONTROL_V1:08x}", f"v_mov_b32 v2, 0x{CONTROL_V2:08x}", "v_mov_b32 v8, 0",
                f".long 0x{w0:08x}, 0x{w1:08x}", "v_and_b32 v8, 0xffff, v8",
                f"v_mov_b32 v5, {4 * (start + i)}", "global_store_b32 v5, v8, s[2:3]"]
    return out


def kernel(table):
    out = control_kernel(len(table))
    for i, (_name, op, kind, (v1, v2), opsel, neg0) in enumerate(table):
        w0, w1 = words(op, kind, opsel, neg0)
        out += [f"v_mov_b32 v1, 0x{v1:08x}", f"v_mov_b32 v2, 0x{v2:08x}", "s_mov_b32 s10, 0", "s_mov_b32 s11, 0",
                f".long 0x{w0:08x}, 0x{w1:08x}",
                "v_mov_b32 v8, s10", "v_and_b32 v8, 1, v8",
                f"v_mov_b32 v5, {4 * i}", "global_store_b32 v5, v8, s[2:3]"]
    return out + ["s_waitcnt vmcnt(0)", "s_endpgm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("rule", "hw"), default="rule")
    a = ap.parse_args()
    table = rows()
    if a.oracle == "rule":
        for row in table:
            h, ig = answer(row, True), answer(row, False)
            print(f"{row[0]:22} op_sel honoured {h}  ignored {ig}{'' if h != ig else '  (does not tell them apart)'}")
        return 0
    import asm
    import runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    print(f"device: {oracle.device}")
    got = {}
    controls = control_rows()
    for wave in (32, 64):
        res = oracle.run([asm.Kernel(kernel(table), wave=wave, threads=1, words=len(table) + len(controls),
                                     raw=True, vgprs=16, sgprs=24)])[0]
        if isinstance(res, str):
            print(f"wave{wave}: kernel rejected: {res}")
            return 2
        got[wave] = res
    # The control must show the hand-set bits working on a VOP3 op that takes op_sel; otherwise nothing is measured.
    for k, (name, opsel) in enumerate(controls):
        h, ig = control_answer(opsel, True), control_answer(opsel, False)
        g = got[32][len(table) + k]
        print(f"{name:30} hw 0x{g:04x} honoured 0x{h:04x} ignored 0x{ig:04x}  {'ok' if g == h else 'CONTROL FAILED'}")
        if g != h or got[64][len(table) + k] != h:
            print("the control does not honour the hand-set OPSEL bits: the encoding is wrong, nothing measured")
            return 2
    # The rule checked is the measured one (gfx1036): the compares read the low halves whatever OPSEL says.
    bad = telling = 0
    for i, row in enumerate(table):
        h, ig = answer(row, True), answer(row, False)
        g32, g64 = got[32][i], got[64][i]
        ok = g32 == ig and g64 == ig
        bad += not ok
        telling += h != ig
        extra = "" if g32 == g64 else f" (wave64 {g64})"
        note = "" if h == ig else ("  op_sel ignored" if ok else "  op_sel HONOURED")
        print(f"{row[0]:22} hw {g32}{extra:12} honoured {h} ignored {ig}  {'ok' if ok else 'DIFFERS'}{note}")
    print(f"{len(table) - bad}/{len(table)} rows match op_sel being ignored; {telling} of them tell the two apart")
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
