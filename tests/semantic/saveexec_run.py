"""Shared harness for the test_salu_*_saveexec_b64.py files: run one s_<op>_saveexec_b64 through tools/rdna2spirv.py
(support.translate_lines) on every lane of a wave64 in spirv_mini_interp.run_workgroup and return what the program stored.

RDNA2 ISA, SOP1: D = EXEC; EXEC = f(S0, EXEC); SCC = (EXEC != 0). f is in ISA below, one plain-Python lambda per op
(andn2 complements EXEC, orn2 too, nand/nor/xnor complement the result). S0 comes from VCC (v_cmp_e32), from an SGPR
pair written by v_cmp_e64, from an SGPR pair loaded with 32-bit literals, or is EXEC itself. EXEC is a literal in each half.
Every lane stores whether it is on after the instruction (a lane that EXEC had off can come on: nand, nor, orn2, xnor);
lane 0 stores SCC and the population count and lowest lane of D.
"""
import struct

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL64 = (1 << 64) - 1
SCC_AT, BCNT_AT, FF1_AT = 256, 260, 264
SOURCES = ("vcc", "s[10:11]", "literal", "exec")

ISA = {
    "and": lambda s, e: s & e, "or": lambda s, e: s | e, "xor": lambda s, e: s ^ e,
    "andn2": lambda s, e: s & ~e, "orn2": lambda s, e: s | ~e,
    "nand": lambda s, e: ~(s & e), "nor": lambda s, e: ~(s | e), "xnor": lambda s, e: ~(s ^ e),
}

# (multiplier, addend, and-mask, EXEC): S0 = the lanes l with ((l * multiplier + addend) & and-mask) == 0.
CASES = (
    (1, 0, 8, 0x0000FFFFFFFF0000), (1, 0, 32, 0xFFFFFFFF00000000), (1, 32, 32, 0x0000FFFF0000FFFF),
    (1, 32, 32, 0xFF00000000000000), (3, 1, 5, 0x5555555555555555), (7, 2, 12, 0xF0F0F0F00F0F0F0F),
    (1, 0, 0, 0), (1, 0, 0, ALL64), (1, 0, 64, 0x123456789ABCDEF0), (1, 0, 127, 0),
    (1, 0, 63, 0x8000000000000000), (5, 3, 63, 0x00000000FFFFFFFF), (1, 64, 64, 0), (1, 64, 64, ALL64),
)


def fssm(threads=64):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def s0_mask(mul, add, mask):
    return sum(1 << lane for lane in range(64) if ((lane * mul + add) & mask) == 0)


def model(kind, s0, old_exec):
    """(new EXEC, D, SCC) from the ISA definition."""
    new = ISA[kind](s0, old_exec) & ALL64
    return new, old_exec, int(new != 0)


def program(kind, src, s0, old_exec, mul, add, mask):
    def store(reg, at):
        return [f"v_mov_b32_e32 v20, {reg}", f"v_mov_b32_e32 v21, {at}", "buffer_store_dword v20, v21, s[0:3], 0 offen"]

    setup = {"vcc": ["v_cmp_eq_u32_e32 vcc, 0, v3"], "s[10:11]": ["v_cmp_eq_u32_e64 s[10:11], 0, v3"],
             "literal": [f"s_mov_b32 s12, {s0 & 0xFFFFFFFF:#x}", f"s_mov_b32 s13, {s0 >> 32:#x}"], "exec": []}[src]
    operand = {"vcc": "vcc", "s[10:11]": "s[10:11]", "literal": "s[12:13]", "exec": "exec"}[src]
    return (["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", "v_mov_b32_e32 v2, 0",
             f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3"] + setup +
            [f"s_mov_b32 exec_lo, {old_exec & 0xFFFFFFFF:#x}", f"s_mov_b32 exec_hi, {old_exec >> 32:#x}",
             f"s_{kind}_saveexec_b64 s[6:7], {operand}", "s_cselect_b32 s8, 1, 0",
             "v_mov_b32_e32 v2, 1", "s_mov_b64 exec, -1",
             "s_bcnt1_i32_b64 s9, s[6:7]", "s_ff1_i32_b64 s10, s[6:7]",
             "v_lshlrev_b32_e32 v21, 2, v1", "buffer_store_dword v2, v21, s[0:3], 0 offen",
             "v_cmpx_eq_u32_e32 0, v1"]
            + store("s8", SCC_AT) + store("s9", BCNT_AT) + store("s10", FF1_AT) + ["s_endpgm"])


def run(kind, src, mul, add, mask, old_exec):
    s0 = s0_mask(mul, add, mask)
    spv = support.translate_lines("cs_saveexec_b64", program(kind, src, s0, old_exec, mul, add, mask), fssm=fssm())
    return si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=VSHARP + [0] * 28, memory={})


def check(test, kind):
    """Every case and source for `kind` against the ISA model."""
    for src in SOURCES:
        for mul, add, mask, old_exec in CASES:
            s0 = old_exec if src == "exec" else s0_mask(mul, add, mask)
            with test.subTest(op=kind, src=src, s0=(mul, add, mask), exec=hex(old_exec)):
                new, d, scc = model(kind, s0, old_exec)
                memory = run(kind, src, mul, add, mask, old_exec)
                got = sum(memory[4 * lane] << lane for lane in range(64))
                test.assertEqual(got, new, f"EXEC {got:#018x}, ISA {new:#018x}")
                test.assertEqual(memory[SCC_AT], scc, "SCC")
                test.assertEqual(memory[BCNT_AT], bin(d).count("1"), "population count of D")
                test.assertEqual(memory[FF1_AT], (d & -d).bit_length() - 1 if d else 0xFFFFFFFF, "lowest lane of D")
