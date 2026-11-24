"""The 64-bit EXEC-saving ops with a compare result in VCC or in an SGPR pair, run across every lane of a wave64.

RDNA2 ISA:
  s_and_saveexec_b64    D = EXEC; EXEC = S0 & EXEC;  SCC = (EXEC != 0).
  s_or_saveexec_b64     D = EXEC; EXEC = S0 | EXEC;  SCC = (EXEC != 0).
  s_andn2_saveexec_b64  D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0).
  s_andn2_wrexec_b64    EXEC = S0 & ~EXEC; D = EXEC (the new one); SCC = (EXEC != 0).

Programs are written here (no game data) and translated by tools/rdna2spirv.py; spirv_mini_interp.run_workgroup runs
one workgroup of 64 lanes on a 64-wide subgroup. S0 is written by v_cmp into VCC (_e32) or into s[10:11] (_e64),
EXEC is a constant in each half. Every lane stores whether it is on in the new EXEC, and lane 0 stores SCC and the
population count and lowest lane of D. The expected values come from the ISA lines above, written as integers in
`model`, not from the translator. Constant sources are in test_value_semantics.py (cs_w64_mask_imm); the VCC form of
the andn2 ops is in test_andn2_exec_b64.py.

The GPU test is runtime/test/shaders/cs_w64x_saveexec_b64.asm; its fixture also runs here. The --wave64-emulate
output needs Workgroup memory, which the interpreter rejects, so it is covered by the GPU test only.
"""
import struct
import unittest

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL64 = (1 << 64) - 1
EXPECTED_FIXTURE = [6160, 12304, 5120, 2088, 6168, 8192, 7176, 1837171]   # derivation in cs_w64x_saveexec_b64.asm
SCC_AT, BCNT_AT, FF1_AT = 256, 260, 264                                    # bytes in out[], after the 64 lane words
OPS = ("s_and_saveexec_b64", "s_or_saveexec_b64", "s_andn2_saveexec_b64", "s_andn2_wrexec_b64")
SOURCES = ("vcc", "s[10:11]")

# (multiplier, addend, and-mask, EXEC): S0 = the lanes l with ((l * multiplier + addend) & and-mask) == 0.
CASES = (
    (1, 0, 8, 0x0000FFFFFFFF0000),            # S0 in both halves, EXEC across the middle
    (1, 0, 32, 0xFFFFFFFF00000000),           # S0 = the low half, EXEC = the high half
    (1, 32, 32, 0x0000FFFF0000FFFF),          # S0 = the high half
    (1, 32, 32, 0xFF00000000000000),          # S0 = the high half, EXEC = lanes 56..63: every result is high only
    (3, 1, 5, 0x5555555555555555),
    (7, 2, 12, 0xF0F0F0F00F0F0F0F),
    (1, 0, 0, 0),                             # S0 = every lane, EXEC = 0
    (1, 0, 0, ALL64),                         # S0 = every lane, EXEC = every lane
    (1, 0, 64, 0x123456789ABCDEF0),           # S0 = every lane (bit 6 is never set)
    (1, 0, 127, 0),                           # S0 = lane 0 only, EXEC = 0
    (1, 0, 63, 0x8000000000000000),           # S0 = lane 0, EXEC = lane 63
    (5, 3, 63, 0x00000000FFFFFFFF),           # a single lane in S0, EXEC = the low half
    (1, 64, 64, 0),                           # S0 = no lane (bit 6 of l + 64 is always set), EXEC = 0
)


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def s0_mask(mul, add, mask):
    return sum(1 << lane for lane in range(64) if ((lane * mul + add) & mask) == 0)


def model(op, s0, old_exec):
    """(new EXEC, D, SCC) from the ISA's definition."""
    new_exec = {"s_and_saveexec_b64": s0 & old_exec, "s_or_saveexec_b64": s0 | old_exec,
                "s_andn2_saveexec_b64": s0 & ~old_exec, "s_andn2_wrexec_b64": s0 & ~old_exec}[op] & ALL64
    return new_exec, (new_exec if op == "s_andn2_wrexec_b64" else old_exec), int(new_exec != 0)


def program(op, src, mul, add, mask, old_exec):
    def store(reg, at):
        return [f"v_mov_b32_e32 v20, {reg}", f"v_mov_b32_e32 v21, {at}", "buffer_store_dword v20, v21, s[0:3], 0 offen"]

    cmp = "v_cmp_eq_u32_e32 vcc, 0, v3" if src == "vcc" else f"v_cmp_eq_u32_e64 {src}, 0, v3"
    return (["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", "v_mov_b32_e32 v2, 0",
             f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3", cmp,
             f"s_mov_b32 exec_lo, {old_exec & 0xFFFFFFFF:#x}", f"s_mov_b32 exec_hi, {old_exec >> 32:#x}",
             f"{op} s[6:7], {src}", "s_cselect_b32 s8, 1, 0",
             "v_mov_b32_e32 v2, 1",                      # only the lanes the instruction left on
             "s_mov_b64 exec, -1",
             "s_bcnt1_i32_b64 s9, s[6:7]", "s_ff1_i32_b64 s10, s[6:7]",
             "v_lshlrev_b32_e32 v21, 2, v1", "buffer_store_dword v2, v21, s[0:3], 0 offen",
             "v_cmpx_eq_u32_e32 0, v1"]
            + store("s8", SCC_AT) + store("s9", BCNT_AT) + store("s10", FF1_AT) + ["s_endpgm"])


def run(spv, subgroup=64):
    return si.run_workgroup(
        spv, 64, subgroup, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup},
        push_constants=VSHARP + [0] * 28, memory={})


class SaveexecB64(unittest.TestCase):
    def test_model_examples(self):
        # The ISA lines on the masks of cs_w64x_saveexec_b64.asm, by hand.
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        self.assertEqual(model("s_and_saveexec_b64", lanes(0, 40), lanes(16, 64)), (lanes(16, 40), lanes(16, 64), 1))
        self.assertEqual(model("s_or_saveexec_b64", lanes(0, 12), lanes(40, 48)),
                         (lanes(0, 12) | lanes(40, 48), lanes(40, 48), 1))
        self.assertEqual(model("s_or_saveexec_b64", 0, 0), (0, 0, 0))
        self.assertEqual(model("s_and_saveexec_b64", lanes(0, 32), lanes(32, 64)), (0, lanes(32, 64), 0))
        self.assertEqual(model("s_and_saveexec_b64", lanes(48, 64), lanes(0, 16) | lanes(32, 64)),
                         (lanes(48, 64), lanes(0, 16) | lanes(32, 64), 1))
        self.assertEqual(model("s_andn2_saveexec_b64", lanes(24, 64), lanes(0, 16) | lanes(48, 64)),
                         (lanes(24, 48), lanes(0, 16) | lanes(48, 64), 1))
        self.assertEqual(model("s_andn2_wrexec_b64", lanes(0, 36), lanes(0, 8)), (lanes(8, 36), lanes(8, 36), 1))

    def test_every_lane(self):
        for op in OPS:
            for src in SOURCES:
                for mul, add, mask, old_exec in CASES:
                    with self.subTest(op=op, src=src, s0=(mul, add, mask), exec=hex(old_exec)):
                        new_exec, d, scc = model(op, s0_mask(mul, add, mask), old_exec)
                        spv = support.translate_lines("cs_saveexec_b64", program(op, src, mul, add, mask, old_exec),
                                                      fssm=fssm(64))
                        memory = run(spv)
                        got_exec = sum(memory[4 * lane] << lane for lane in range(64))
                        self.assertEqual(got_exec, new_exec, f"EXEC {got_exec:#018x}, ISA {new_exec:#018x}")
                        self.assertEqual(memory[SCC_AT], scc, "SCC")
                        self.assertEqual(memory[BCNT_AT], bin(d).count("1"), "population count of D")
                        lowest = (d & -d).bit_length() - 1 if d else 0xFFFFFFFF   # s_ff1 of 0 is -1
                        self.assertEqual(memory[FF1_AT], lowest, "lowest lane of D")

    def test_cases_cover_both_halves(self):
        # For each op the table has results in the low half only, in the high half only, in both and in neither.
        for op in OPS:
            seen = set()
            for mul, add, mask, old_exec in CASES:
                new_exec = model(op, s0_mask(mul, add, mask), old_exec)[0]
                seen.add((bool(new_exec & 0xFFFFFFFF), bool(new_exec >> 32)))
            self.assertEqual(seen, {(False, False), (True, False), (False, True), (True, True)}, op)

    def test_gpu_test_fixture(self):
        """cs_w64x_saveexec_b64.asm: the words check.py expects (lane 0 stores them)."""
        spv = support.translate_fixture("cs_w64x_saveexec_b64", fssm=fssm(64))
        memory = run(spv)
        self.assertEqual(sorted(memory), list(range(0, 32, 4)), "lane 0 stores out[0..7] only")
        self.assertEqual([memory[4 * k] for k in range(8)], EXPECTED_FIXTURE)


if __name__ == "__main__":
    unittest.main()
