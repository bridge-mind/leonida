"""Semantic interpreter tests for RDNA2 s_andn1_wrexec_b32, s_andn2_wrexec_b32,
s_andn1_wrexec_b64, and s_flbit_i32_b32.

s_andn1_wrexec_b{32,64} (RDNA2 ISA): EXEC = ~S0 & EXEC; D = EXEC; SCC = (EXEC != 0).
s_andn2_wrexec_b{32,64} (RDNA2 ISA): EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0).
Each wrexec case translates a synthetic pixel shader with tools/rdna2spirv.py, runs all 32 or 64
lanes in tests/semantic/spirv_wave_interp.py, and checks per-lane EXEC update, D mask, and SCC.

s_flbit_i32_b32 (RDNA2 ISA): D = leading-zero count from MSB (32 - bit_length(S0), or 0xFFFFFFFF if
S0 == 0); SCC is not written. Each case translates a synthetic pixel shader with tools/rdna2spirv.py,
runs it in tests/semantic/spirv_mini_interp.py, and checks D and preserved SCC.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
ALL64 = 0xFFFFFFFFFFFFFFFF


def ref_wrexec(op: str, s0: int, old_exec: int, wave: int) -> tuple:
    mask = (1 << wave) - 1
    if "andn1" in op:
        new_exec = (~s0 & old_exec) & mask
    else:
        new_exec = (s0 & ~old_exec) & mask
    return new_exec, new_exec, int(new_exec != 0)


def ref_flbit_i32_b32(x: int) -> int:
    x &= MASK32
    return MASK32 if x == 0 else 32 - x.bit_length()


def run_wrexec_wave(op: str, wave: int, s0_mask: int, old_exec: int) -> tuple:
    """Run `op` on a `wave`-lane PS; return (new_exec_mask, d_mask, scc)."""
    w64 = wave == 64
    s0_reg, d_reg = ("s[2:3]", "s[4:5]") if w64 else ("s2", "s4")
    # Build S0 in s0_reg from per-lane bit tests against v4 (lane id).
    if w64:
        s0_setup = [
            f"s_mov_b32 s6, {s0_mask & MASK32:#x}",
            f"s_mov_b32 s7, {(s0_mask >> 32) & MASK32:#x}",
            "v_and_b32_e32 v5, 31, v4",
            "v_lshlrev_b32_e32 v5, v5, 1",
            "v_and_b32_e32 v6, v5, s6",
            "v_and_b32_e32 v7, v5, s7",
            "v_cmp_lt_u32_e32 vcc, 31, v4",
            "v_cndmask_b32_e32 v5, v6, v7, vcc",
            f"v_cmp_ne_u32_e64 {s0_reg}, 0, v5",
        ]
        exec_setup = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 exec_hi, {(old_exec >> 32) & MASK32:#x}",
        ]
    else:
        s0_setup = [
            f"s_mov_b32 s6, {s0_mask & MASK32:#x}",
            "v_lshlrev_b32_e32 v5, v4, 1",
            "v_and_b32_e32 v5, v5, s6",
            f"v_cmp_ne_u32_e64 {s0_reg}, 0, v5",
        ]
        exec_setup = [f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}"]

    lines = (
        ["v_mbcnt_lo_u32_b32 v4, -1, 0"]
        + (["v_mbcnt_hi_u32_b32 v4, -1, v4"] if w64 else [])
        + s0_setup
        + ["v_mov_b32 v1, 0x11"]
        + exec_setup
        + [
            f"{op} {d_reg}, {s0_reg}",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v1, 0x22",
            f"s_mov_b{wave} exec, -1",
            f"v_cndmask_b32_e64 v2, 0, 1, {d_reg}",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
    )
    spv = support.translate_lines(f"ps_{op}", lines)
    res = wi.run_wave(spv, wave=wave)
    outs = res["outputs"]
    assert len(outs) == wave
    new_exec = 0
    d_mask = 0
    scc = outs[0]["%out_mrt0"][0]
    for lane, out in enumerate(outs):
        assert out is not None, f"lane {lane} killed"
        lane_scc, exec_tag, d_bit, _ = out["%out_mrt0"]
        assert lane_scc == scc, f"lane {lane} SCC mismatch"
        assert exec_tag in (0x11, 0x22), f"lane {lane} unexpected exec_tag {exec_tag:#x}"
        assert d_bit in (0, 1), f"lane {lane} unexpected d_bit {d_bit}"
        if exec_tag == 0x22:
            new_exec |= 1 << lane
        d_mask |= d_bit << lane
    return new_exec, d_mask, scc


def run_flbit(setup_lines, src_op: str, scc_in: int) -> tuple:
    lines = [
        *setup_lines,
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"s_flbit_i32_b32 s0, {src_op}",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_salu_flbit32", lines)
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0], out[1]


class WrexecB32Tests(unittest.TestCase):
    CASES_32 = [
        (0x0000FFFF, 0x00FF00FF),   # mixed overlap
        (0xFFFFFFFF, 0xFFFFFFFF),   # andn1 -> 0 (SCC=0), andn2 -> 0 (SCC=0)
        (0x00000000, 0xFFFFFFFF),   # andn1 -> ALL32 (SCC=1), andn2 -> 0 (SCC=0)
        (0xFFFFFFFF, 0x00000000),   # andn1 -> 0 (SCC=0), andn2 -> ALL32 (SCC=1)
        (0xAAAAAAAA, 0x55555555),   # disjoint alternating lanes
        (0x80000001, 0xFFFFFFFF),   # boundary lanes 0 and 31
        (0x00000000, 0x00000000),   # both zero -> 0 (SCC=0)
    ]

    def test_wrexec_b32_and_contrast(self):
        for s0, old_exec in self.CASES_32:
            for op in ("s_andn1_wrexec_b32", "s_andn2_wrexec_b32"):
                with self.subTest(op=op, s0=hex(s0), exec=hex(old_exec)):
                    self.assertEqual(
                        run_wrexec_wave(op, 32, s0, old_exec),
                        ref_wrexec(op, s0, old_exec, 32),
                    )


class WrexecB64Tests(unittest.TestCase):
    CASES_64 = [
        (0xFFFFFFFF00000000, 0x00000000FFFFFFFF, "low-half only"),
        (0x00000000FFFFFFFF, 0xFFFFFFFF00000000, "high-half only"),
        (0xFFFF00000000FFFF, 0x0000FFFFFFFF0000, "straddling bit 31/32"),
        (0x0000FFFF0000FFFF, 0x000001FFFE000000, "straddling 25..40"),
        (0x0000000000000000, ALL64, "all-ones for andn1, all-zero for andn2"),
        (ALL64, 0x0000000000000000, "all-zero for andn1, all-ones for andn2"),
        (ALL64, ALL64, "all-zero for both (SCC=0)"),
        (0x5555555555555555, 0xF0F0F0F00F0F0F0F, "mixed alternating"),
    ]

    def test_wrexec_b64_and_contrast(self):
        for s0, old_exec, desc in self.CASES_64:
            for op in ("s_andn1_wrexec_b64", "s_andn2_wrexec_b64"):
                with self.subTest(op=op, case=desc):
                    self.assertEqual(
                        run_wrexec_wave(op, 64, s0, old_exec),
                        ref_wrexec(op, s0, old_exec, 64),
                    )


class FlbitI32B32Tests(unittest.TestCase):
    BOUNDARIES = [
        (0, 0xFFFFFFFF),
        (0x80000000, 0),
        (0x40000000, 1),
        (1, 31),
        (0x00010000, 15),
        (0xFFFFFFFF, 0),
        (0x7FFFFFFF, 1),
        (0x00000002, 30),
    ]

    def test_reference(self):
        for val, want in self.BOUNDARIES:
            self.assertEqual(ref_flbit_i32_b32(val), want)

    def test_boundaries_and_scc_preserved(self):
        for val, want in self.BOUNDARIES:
            for scc_in in (0, 1):
                with self.subTest(val=hex(val), scc_in=scc_in):
                    self.assertEqual(
                        run_flbit([f"s_mov_b32 s2, {val:#x}"], "s2", scc_in),
                        (want, scc_in),
                    )

    def test_all_single_bits(self):
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                self.assertEqual(
                    run_flbit([f"s_mov_b32 s2, {val:#x}"], "s2", bit & 1),
                    (31 - bit, bit & 1),
                )

    def test_inline_and_literal_operands(self):
        for lit, val in [("0", 0), ("1", 1), ("-1", 0xFFFFFFFF), ("0x40000000", 0x40000000), ("0x00010000", 0x00010000)]:
            with self.subTest(lit=lit):
                self.assertEqual(run_flbit([], lit, 1), (ref_flbit_i32_b32(val), 1))

    def test_seeded_random(self):
        rng = random.Random(0xF1B17)
        for _ in range(30):
            val = rng.getrandbits(32) >> rng.randint(0, 31)
            scc_in = rng.randint(0, 1)
            with self.subTest(val=hex(val)):
                self.assertEqual(
                    run_flbit([f"s_mov_b32 s2, {val:#x}"], "s2", scc_in),
                    (ref_flbit_i32_b32(val), scc_in),
                )


if __name__ == "__main__":
    unittest.main()
