"""s_andn2_saveexec_b64 and s_andn2_wrexec_b64 on a 64-lane wave, run across every lane of the translated SPIR-V.

RDNA2 ISA:
  s_andn2_saveexec_b64  D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0).
  s_andn2_wrexec_b64    EXEC = S0 & ~EXEC; D = EXEC (the new one); SCC = (EXEC != 0).

Programs are written here (no game data) and translated by tools/rdna2spirv.py; spirv_mini_interp.run_workgroup runs
one workgroup of 64 lanes on a 64-wide subgroup. S0 is a compare result in VCC, EXEC a constant in each half. Every
lane stores whether it is on in the new EXEC, and lane 0 stores SCC and the population count and lowest lane of D,
so the three results are each checked over both halves of the wave. The expected values come from the two lines of
the ISA above, written as integers in `model`, not from the translator.

The GPU test is runtime/test/shaders/cs_w64x_andn2_exec.asm; its fixture also runs here. The --wave64-emulate output
needs Workgroup memory, which the interpreter rejects, so it is covered by the GPU test only (as in
test_ds_permute.py). The _b32 forms are tested with cs_w32_andn2_exec.
"""
import struct
import unittest

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL64 = (1 << 64) - 1
EXPECTED_FIXTURE = [1, 16, 32, 4144, 2052, 3, 8, 1544]   # derivation in cs_w64x_andn2_exec.asm
SCC_AT, BCNT_AT, FF1_AT = 256, 260, 264                   # bytes in out[], after the 64 lane words

# (multiplier, addend, and-mask, EXEC): S0 = the lanes l with ((l * multiplier + addend) & and-mask) == 0.
CASES = (
    (1, 0, 8, 0x0000FFFFFFFF0000),            # S0 in both halves, EXEC across the middle
    (1, 0, 32, 0x0000FFFFFFFF0000),           # S0 = the low half: the result is lanes 0..15 only
    (1, 32, 32, 0x0000FFFFFFFF0000),          # S0 = the high half: the result is lanes 48..63 only
    (3, 1, 5, 0x5555555555555555),
    (7, 2, 12, 0xF0F0F0F00F0F0F0F),
    (1, 0, 0, 0),                             # S0 = every lane, EXEC = 0: every lane on, D = 0 for saveexec
    (1, 0, 0, ALL64),                         # S0 = every lane, EXEC = every lane: nothing left, SCC 0
    (1, 1, 1, 0xFFFFFFFF00000000),            # odd lanes, EXEC = the high half: odd lanes of the low half
    (1, 0, 1, 0x00000000FFFFFFFF),            # even lanes, EXEC = the low half: even lanes of the high half
    (5, 3, 63, 0x8000000000000001),           # a single lane in S0
    (1, 0, 64, 0x123456789ABCDEF0),           # S0 = every lane (bit 6 is never set): EXEC is inverted
    (1, 0, 127, 0xFFFFFFFFFFFFFFFE),          # S0 = lane 0, the only lane off in EXEC
)


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def s0_mask(mul, add, mask):
    return sum(1 << lane for lane in range(64) if ((lane * mul + add) & mask) == 0)


def model(op, s0, old_exec):
    """(new EXEC, D, SCC) from the ISA's definition."""
    new_exec = s0 & ~old_exec & ALL64
    return new_exec, (old_exec if op == "s_andn2_saveexec_b64" else new_exec), int(new_exec != 0)


def program(op, mul, add, mask, old_exec):
    def store(reg, at):
        return [f"v_mov_b32_e32 v20, {reg}", f"v_mov_b32_e32 v21, {at}", "buffer_store_dword v20, v21, s[0:3], 0 offen"]

    return (["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", "v_mov_b32_e32 v2, 0",
             f"v_mad_u32_u24 v3, v1, {mul}, {add}", f"v_and_b32_e32 v3, {mask}, v3", "v_cmp_eq_u32_e32 vcc, 0, v3",
             f"s_mov_b32 exec_lo, {old_exec & 0xFFFFFFFF:#x}", f"s_mov_b32 exec_hi, {old_exec >> 32:#x}",
             f"{op} s[6:7], vcc", "s_cselect_b32 s8, 1, 0",
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


class Andn2ExecB64(unittest.TestCase):
    def test_model_examples(self):
        # The two lines of the ISA on the masks of cs_w64x_andn2_exec.asm, by hand.
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        self.assertEqual(model("s_andn2_saveexec_b64", lanes(40, 64), lanes(16, 48)), (lanes(48, 64), lanes(16, 48), 1))
        self.assertEqual(model("s_andn2_wrexec_b64", 0x00FF00FF00FF00FF, 0xFFFFFFFF0F0F0F0F),
                         (0x00F000F0, 0x00F000F0, 1))
        self.assertEqual(model("s_andn2_wrexec_b64", lanes(0, 8), ALL64), (0, 0, 0))

    def test_every_lane(self):
        for op in ("s_andn2_saveexec_b64", "s_andn2_wrexec_b64"):
            for mul, add, mask, old_exec in CASES:
                with self.subTest(op=op, s0=(mul, add, mask), exec=hex(old_exec)):
                    s0 = s0_mask(mul, add, mask)
                    new_exec, d, scc = model(op, s0, old_exec)
                    spv = support.translate_lines("cs_andn2_b64", program(op, mul, add, mask, old_exec), fssm=fssm(64))
                    memory = run(spv)
                    got_exec = sum(memory[4 * lane] << lane for lane in range(64))
                    self.assertEqual(got_exec, new_exec, f"EXEC {got_exec:#018x}, ISA {new_exec:#018x}")
                    self.assertEqual(memory[SCC_AT], scc, "SCC")
                    self.assertEqual(memory[BCNT_AT], bin(d).count("1"), "population count of D")
                    lowest = (d & -d).bit_length() - 1 if d else 0xFFFFFFFF   # s_ff1 of 0 is -1
                    self.assertEqual(memory[FF1_AT], lowest, "lowest lane of D")

    def test_cases_cover_both_halves(self):
        # The table has results in the low half only, in the high half only, in both and in neither.
        seen = set()
        for mul, add, mask, old_exec in CASES:
            new_exec = model("s_andn2_wrexec_b64", s0_mask(mul, add, mask), old_exec)[0]
            seen.add((bool(new_exec & 0xFFFFFFFF), bool(new_exec >> 32)))
        self.assertEqual(seen, {(False, False), (True, False), (False, True), (True, True)})

    def test_gpu_test_fixture(self):
        """cs_w64x_andn2_exec.asm: the words check.py expects (lane 0 stores them)."""
        spv = support.translate_fixture("cs_w64x_andn2_exec", fssm=fssm(64))
        memory = run(spv)
        self.assertEqual(sorted(memory), list(range(0, 32, 4)), "lane 0 stores out[0..7] only")
        self.assertEqual([memory[4 * k] for k in range(8)], EXPECTED_FIXTURE)


if __name__ == "__main__":
    unittest.main()
