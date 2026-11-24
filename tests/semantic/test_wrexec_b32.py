"""s_andn2_wrexec_b32 and s_andn1_wrexec_b32 on a 32-lane wave, run across every lane of the translated SPIR-V.

RDNA2 ISA (wave32 SOP1):
  s_andn2_wrexec_b32  EXEC = S0 & ~EXEC; D = EXEC (the new one); SCC = (EXEC != 0).
  s_andn1_wrexec_b32  EXEC = ~S0 & EXEC; D = EXEC (the new one); SCC = (EXEC != 0).

Programs are written here (no game data) and translated by tools/rdna2spirv.py; spirv_mini_interp.run_workgroup runs
one workgroup of 32 lanes on a 32-wide subgroup. S0 is a compare result in VCC_LO (via v_cmp_eq_u32_e32), and
exec_lo is initialized to multiple masks (including all-zero results giving SCC=0 and partial masks 0x0000FFFF,
0x55555555, 0x80000001). Only lanes left active in the new EXEC write to the lane output slots; SCC is pre-seeded
to the opposite of the expected value before the op; D is checked via all 32 per-lane bits plus s_bcnt1 and s_ff1
to confirm it holds the new EXEC (not the old EXEC as in saveexec) and that andn2/andn1 are asymmetric.
"""
import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]
ALL32 = 0xFFFFFFFF
SCC_AT, BCNT_AT, FF1_AT, D_BASE = 128, 132, 136, 140  # bytes in out[], after 32 lane-active words (0..124)

# (mul, add, and_mask, old_exec): S0 = lanes l in 0..31 with ((l * mul + add) & and_mask) == 0.
# When mul=0, add=1, and_mask=1: S0 = 0. When and_mask=0: S0 = ALL32.
# When mul=1, add=0, and_mask=16: S0 = 0x0000FFFF (lanes 0..15).
# When mul=1, add=0, and_mask=1: S0 = 0x55555555 (even lanes).
# When mul=1, add=1, and_mask=1: S0 = 0xAAAAAAAA (odd lanes).
# When mul=1, add=0, and_mask=31: S0 = 0x00000001 (lane 0 only).
CASES = (
    (1, 0, 16, 0x0000FFFF),   # S0 = low half, EXEC = low half -> both ops yield 0 (SCC=0)
    (1, 0, 16, 0x55555555),   # S0 = low half, EXEC = even -> andn2=0x0000AAAA, andn1=0x55550000
    (1, 0, 16, 0x80000001),   # S0 = low half, EXEC = endpoints -> andn2=0x0000FFFE, andn1=0x80000000
    (1, 0, 1, 0x0000FFFF),    # S0 = 0x55555555, EXEC = 0x0000FFFF -> andn2=0x55550000, andn1=0x0000AAAA
    (1, 0, 1, 0x55555555),    # S0 = 0x55555555, EXEC = 0x55555555 -> both ops yield 0 (SCC=0)
    (1, 0, 1, 0x80000001),    # S0 = 0x55555555, EXEC = 0x80000001 -> andn2=0x55555554, andn1=0x80000000
    (1, 1, 1, 0x80000001),    # S0 = 0xAAAAAAAA, EXEC = 0x80000001 -> andn2=0x2AAAAAAA, andn1=0x00000001
    (0, 1, 1, 0x0000FFFF),    # S0 = 0, EXEC = 0x0000FFFF -> andn2=0 (SCC=0), andn1=0x0000FFFF (SCC=1)
    (0, 1, 1, 0x55555555),    # S0 = 0, EXEC = 0x55555555 -> andn2=0 (SCC=0), andn1=0x55555555 (SCC=1)
    (1, 0, 0, 0xFFFFFFFF),    # S0 = ALL32, EXEC = ALL32 -> both ops yield 0 (SCC=0)
    (1, 0, 0, 0x00000000),    # S0 = ALL32, EXEC = 0 -> andn2=ALL32 (SCC=1), andn1=0 (SCC=0)
    (1, 0, 0, 0x80000001),    # S0 = ALL32, EXEC = 0x80000001 -> andn2=0x7FFFFFFE, andn1=0 (SCC=0)
    (1, 0, 31, 0x00000001),   # S0 = lane 0, EXEC = lane 0 -> both ops yield 0 (SCC=0)
    (1, 0, 31, 0xFFFF0000),   # S0 = lane 0, EXEC = high half -> andn2=0x00000001, andn1=0xFFFF0000
)


def fssm(threads=32):
    """Compute-shader header: COMPUTE_PGM_RSRC2 0x8088 (4 user SGPRs, s[0:3] = output V#; TGID_X), wave32."""
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def s0_mask(mul, add, mask):
    return sum(1 << lane for lane in range(32) if ((lane * mul + add) & mask) == 0)


def model(op, s0, old_exec):
    """(new EXEC, D, SCC) from the RDNA2 ISA definition."""
    if op == "s_andn2_wrexec_b32":
        new_exec = s0 & (~old_exec & ALL32)
    elif op == "s_andn1_wrexec_b32":
        new_exec = (~s0 & ALL32) & old_exec
    else:
        raise ValueError(f"unknown op: {op}")
    return new_exec, new_exec, int(new_exec != 0)


def program(op, mul, add, mask, old_exec, expected_scc):
    def store(reg, at):
        return [
            f"v_mov_b32_e32 v20, {reg}",
            f"v_mov_b32_e32 v22, {at}",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
        ]

    # Pre-seed SCC to the opposite of expected_scc so an untouched SCC cannot pass.
    seed_scc = "s_cmp_lg_u32 0, 0" if expected_scc else "s_cmp_eq_u32 0, 0"
    return (
        [
            "s_mov_b32 exec_lo, -1",
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_lshlrev_b32_e32 v21, 2, v1",
            "v_mov_b32_e32 v2, 0",
            "s_mov_b32 s6, -1",
            f"v_mad_u32_u24 v3, v1, {mul}, {add}",
            f"v_and_b32_e32 v3, {mask}, v3",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
            f"s_mov_b32 exec_lo, {old_exec:#x}",
            seed_scc,
            f"{op} s6, vcc_lo",
            "s_cselect_b32 s8, 1, 0",
            # Only lanes kept active in the new EXEC write 1 to out[4 * lane].
            "v_mov_b32_e32 v2, 1",
            "buffer_store_dword v2, v21, s[0:3], 0 offen",
            # Restore full EXEC to inspect D (all 32 bits, popcount, ff1) and SCC.
            "s_mov_b32 exec_lo, -1",
            "s_bcnt1_i32_b32 s9, s6",
            "s_ff1_i32_b32 s10, s6",
            "v_cndmask_b32_e64 v7, 0, 1, s6",
            f"v_add_nc_u32 v23, {D_BASE}, v21",
            "buffer_store_dword v7, v23, s[0:3], 0 offen",
            "v_cmpx_eq_u32_e32 0, v1",
        ]
        + store("s8", SCC_AT)
        + store("s9", BCNT_AT)
        + store("s10", FF1_AT)
        + ["s_endpgm"]
    )


def run(spv):
    return si.run_workgroup(
        spv,
        32,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


class WrexecB32Tests(unittest.TestCase):
    def test_model_examples(self):
        self.assertEqual(model("s_andn2_wrexec_b32", 0x0000FFFF, 0x55555555), (0x0000AAAA, 0x0000AAAA, 1))
        self.assertEqual(model("s_andn1_wrexec_b32", 0x0000FFFF, 0x55555555), (0x55550000, 0x55550000, 1))
        self.assertEqual(model("s_andn2_wrexec_b32", 0xFFFFFFFF, 0xFFFFFFFF), (0, 0, 0))
        self.assertEqual(model("s_andn1_wrexec_b32", 0xFFFFFFFF, 0x55555555), (0, 0, 0))

    def test_every_lane_d_and_scc(self):
        for op in ("s_andn2_wrexec_b32", "s_andn1_wrexec_b32"):
            for mul, add, mask, old_exec in CASES:
                with self.subTest(op=op, s0=(mul, add, mask), exec=hex(old_exec)):
                    s0 = s0_mask(mul, add, mask)
                    new_exec, d, scc = model(op, s0, old_exec)
                    spv = support.translate_lines(
                        f"cs_{op}",
                        program(op, mul, add, mask, old_exec, scc),
                        fssm=fssm(32),
                    )
                    self.assertFalse(support.rdna2spirv.translate.last_gen.wave64)
                    memory = run(spv)

                    # Verify which lanes actually wrote to the output buffer (0..124).
                    written_lanes = sorted(addr for addr in memory if addr < 128)
                    expected_written = [4 * lane for lane in range(32) if (new_exec >> lane) & 1]
                    self.assertEqual(written_lanes, expected_written, "active lanes writing to buffer")
                    for addr in written_lanes:
                        self.assertEqual(memory[addr], 1)

                    # Verify SCC, popcount(D), ff1(D), and all 32 bits of D.
                    self.assertEqual(memory[SCC_AT], scc, "SCC")
                    self.assertEqual(memory[BCNT_AT], bin(d).count("1"), "population count of D")
                    lowest = (d & -d).bit_length() - 1 if d else 0xFFFFFFFF
                    self.assertEqual(memory[FF1_AT], lowest, "lowest lane of D")
                    got_d = sum(memory[D_BASE + 4 * lane] << lane for lane in range(32))
                    self.assertEqual(got_d, d, f"D {got_d:#010x}, expected {d:#010x}")

    def test_d_is_new_exec_not_old_exec(self):
        # Cases where old_exec != new_exec so saveexec (D=old_exec) would fail.
        distinct_cases = [
            ("s_andn2_wrexec_b32", 1, 0, 16, 0x55555555),   # old=0x55555555, new=0x0000AAAA
            ("s_andn2_wrexec_b32", 1, 0, 1, 0x80000001),    # old=0x80000001, new=0x55555554
            ("s_andn1_wrexec_b32", 1, 0, 16, 0x55555555),   # old=0x55555555, new=0x55550000
            ("s_andn1_wrexec_b32", 1, 0, 1, 0x0000FFFF),    # old=0x0000FFFF, new=0x0000AAAA
        ]
        for op, mul, add, mask, old_exec in distinct_cases:
            with self.subTest(op=op, exec=hex(old_exec)):
                s0 = s0_mask(mul, add, mask)
                new_exec, d, scc = model(op, s0, old_exec)
                self.assertNotEqual(d, old_exec)
                spv = support.translate_lines(
                    f"cs_{op}_distinct",
                    program(op, mul, add, mask, old_exec, scc),
                    fssm=fssm(32),
                )
                memory = run(spv)
                got_d = sum(memory[D_BASE + 4 * lane] << lane for lane in range(32))
                self.assertEqual(got_d, new_exec)
                self.assertNotEqual(got_d, old_exec)

    def test_asymmetry_andn2_vs_andn1(self):
        # On identical (S0, old_exec), s_andn2_wrexec_b32 (S0 & ~EXEC) and s_andn1_wrexec_b32 (~S0 & EXEC)
        # must produce different masks.
        pairs = [
            (1, 0, 16, 0x55555555),   # S0=0x0000FFFF, EXEC=0x55555555 -> 0x0000AAAA vs 0x55550000
            (1, 0, 1, 0x80000001),    # S0=0x55555555, EXEC=0x80000001 -> 0x55555554 vs 0x80000000
            (0, 1, 1, 0x0000FFFF),    # S0=0,          EXEC=0x0000FFFF -> 0 (SCC=0)  vs 0x0000FFFF (SCC=1)
        ]
        for mul, add, mask, old_exec in pairs:
            with self.subTest(s0=(mul, add, mask), exec=hex(old_exec)):
                s0 = s0_mask(mul, add, mask)
                exp2 = model("s_andn2_wrexec_b32", s0, old_exec)
                exp1 = model("s_andn1_wrexec_b32", s0, old_exec)
                self.assertNotEqual(exp2[0], exp1[0])

                spv2 = support.translate_lines("cs_asym_2", program("s_andn2_wrexec_b32", mul, add, mask, old_exec, exp2[2]), fssm=fssm(32))
                spv1 = support.translate_lines("cs_asym_1", program("s_andn1_wrexec_b32", mul, add, mask, old_exec, exp1[2]), fssm=fssm(32))
                mem2 = run(spv2)
                mem1 = run(spv1)
                d2 = sum(mem2[D_BASE + 4 * lane] << lane for lane in range(32))
                d1 = sum(mem1[D_BASE + 4 * lane] << lane for lane in range(32))
                self.assertEqual((d2, mem2[SCC_AT]), (exp2[1], exp2[2]))
                self.assertEqual((d1, mem1[SCC_AT]), (exp1[1], exp1[2]))
                self.assertNotEqual(d2, d1)

    def test_spirv_text_negative_checks(self):
        for op in ("s_andn2_wrexec_b32", "s_andn1_wrexec_b32"):
            with self.subTest(op=op):
                spv = support.translate_lines(
                    f"cs_{op}_min",
                    [
                        "s_mov_b32 exec_lo, -1",
                        "v_mbcnt_lo_u32_b32 v1, -1, 0",
                        "v_cmp_eq_u32_e32 vcc_lo, 0, v1",
                        f"{op} s6, vcc_lo",
                        "s_endpgm",
                    ],
                    fssm=fssm(32),
                )
                self.assertFalse(support.rdna2spirv.translate.last_gen.wave64)
                self.assertIn("OpNot %uint", spv)
                self.assertIn("OpBitwiseAnd %uint", spv)
                self.assertIn("OpGroupNonUniformAny", spv)
                self.assertNotIn("OpTypePointer Workgroup", spv)
                self.assertNotIn("OpControlBarrier", spv)
                self.assertNotIn("OpGroupNonUniformBallot", spv)
                self.assertNotIn("OpBitwiseOr", spv)
                self.assertNotIn("OpBitwiseXor", spv)


if __name__ == "__main__":
    unittest.main()
