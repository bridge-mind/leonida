"""Semantic interpreter tests for RDNA2 SOPK scalar 16-bit immediate compares.

Covers all 12 s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32} opcodes, verifying:
- 16-bit sign-extension for _i32 (0xFFFF -> -1, 0x8000 -> -32768, 0x7FFF -> +32767)
  versus 16-bit zero-extension for _u32 (0xFFFF -> 65535, 0x8000 -> 32768);
- both SCC = 1 and SCC = 0 outcomes (pre-setting SCC = 1 before false compares
  to verify SCC is cleared);
- in-memory negative controls on the generated SPIR-V text.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF

S0_VALUES = [
    0,
    1,
    0xFFFFFFFF,  # -1
    0x80000000,  # INT32_MIN
    0x7FFFFFFF,  # INT32_MAX
    32768,       # 0x8000
    65535,       # 0xFFFF
]

IMM16_VALUES = [0, 1, 0x7FFF, 0x8000, 0xFFFF]

CONDS = {
    "eq": lambda a, b: a == b,
    "lg": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
}


def s32(x: int) -> int:
    x &= MASK32
    return x - 0x100000000 if (x & 0x80000000) else x


def sext16(imm: int) -> int:
    imm &= 0xFFFF
    return imm - 0x10000 if (imm & 0x8000) else imm


def run_cmpk(op: str, s0_val: int, imm16: int, preset_scc: bool = False) -> int:
    """Translate and execute one s_cmpk_* instruction, returning SCC (0 or 1)."""
    lines = []
    if preset_scc:
        lines.append("s_cmp_eq_u32 1, 1")
    lines.extend([
        f"s_mov_b32 s0, {s0_val:#x}",
        f"{op} s0, {imm16:#x}",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_salu_cmpk", lines)
    return support.mrt0_words(spv)[0]


class SaluCmpkTests(unittest.TestCase):
    def test_all_12_opcodes_signed_and_unsigned(self):
        for cond, cmp_fn in CONDS.items():
            for ty in ("i32", "u32"):
                op = f"s_cmpk_{cond}_{ty}"
                for s0_val in S0_VALUES:
                    for imm16 in IMM16_VALUES:
                        lhs = s32(s0_val) if ty == "i32" else (s0_val & MASK32)
                        rhs = sext16(imm16) if ty == "i32" else (imm16 & 0xFFFF)
                        want = int(cmp_fn(lhs, rhs))
                        with self.subTest(op=op, s0=hex(s0_val), imm16=hex(imm16)):
                            # Pre-set SCC=1 before expected-false compares to verify it clears.
                            got = run_cmpk(op, s0_val, imm16, preset_scc=(want == 0))
                            self.assertEqual(got, want)

    def test_sign_vs_zero_extension_divergence(self):
        # 0xFFFF is -1 in i32 (equal to 0xFFFFFFFF) vs 65535 in u32 (equal to 65535).
        self.assertEqual(run_cmpk("s_cmpk_eq_i32", 0xFFFFFFFF, 0xFFFF, preset_scc=False), 1)
        self.assertEqual(run_cmpk("s_cmpk_eq_u32", 0xFFFFFFFF, 0xFFFF, preset_scc=True), 0)
        self.assertEqual(run_cmpk("s_cmpk_eq_u32", 65535, 0xFFFF, preset_scc=False), 1)
        self.assertEqual(run_cmpk("s_cmpk_eq_i32", 65535, 0xFFFF, preset_scc=True), 0)

        # 0x8000 is -32768 in i32 (< 0) vs +32768 in u32 (> 0).
        self.assertEqual(run_cmpk("s_cmpk_lt_i32", 0, 0x8000, preset_scc=True), 0)
        self.assertEqual(run_cmpk("s_cmpk_gt_i32", 0, 0x8000, preset_scc=False), 1)
        self.assertEqual(run_cmpk("s_cmpk_lt_u32", 0, 0x8000, preset_scc=False), 1)
        self.assertEqual(run_cmpk("s_cmpk_gt_u32", 0, 0x8000, preset_scc=True), 0)

    def test_imm16_edges_decimal_tokens_and_s0_kept(self):
        # S0 one below, at and one above the extended immediate (-32768 for _i32, 65535 for _u32), plus a
        # high-bit S0 that equals sext16(0x8000) only when sign-extended; decimal immediates written signed.
        # S0 must read back unchanged after the compare.
        cases = [
            ("i32", 0xFFFF7FFF, "0x8000"),   # -32769 vs -32768
            ("i32", 0xFFFF8000, "0x8000"),   # -32768 vs -32768
            ("i32", 0xFFFF8001, "0x8000"),   # -32767 vs -32768
            ("i32", 0xFFFF8000, "-32768"),
            ("i32", 0xFFFFFFFE, "-1"),
            ("i32", 0x00007FFE, "32767"),
            ("u32", 0x0000FFFE, "0xffff"),   # 65534 vs 65535
            ("u32", 0x00010000, "0xffff"),   # 65536 vs 65535
            ("u32", 0xFFFF8000, "0x8000"),   # not sign-extended: 0xffff8000 > 0x8000
        ]
        for ty, s0_val, tok in cases:
            imm16 = int(tok, 0) & 0xFFFF
            for cond, cmp_fn in CONDS.items():
                lhs = s32(s0_val) if ty == "i32" else s0_val
                rhs = sext16(imm16) if ty == "i32" else imm16
                want = int(cmp_fn(lhs, rhs))
                with self.subTest(op=f"s_cmpk_{cond}_{ty}", s0=hex(s0_val), imm=tok):
                    spv = support.translate_lines("ps_salu_cmpk_edge", [
                        "s_cmp_eq_u32 1, 1" if want == 0 else "s_cmp_eq_u32 0, 1",
                        f"s_mov_b32 s0, {s0_val:#x}",
                        f"s_cmpk_{cond}_{ty} s0, {tok}",
                        "s_cselect_b32 s1, 1, 0",
                        "v_mov_b32 v0, s1",
                        "v_mov_b32 v1, s0",
                        "exp mrt0 v0, v1, v0, v0 done vm",
                        "s_endpgm",
                    ])
                    self.assertEqual(support.mrt0_words(spv)[:2], [want, s0_val])

    def test_negative_controls_catch_spirv_regressions(self):
        # 1) Signed vs unsigned operator swap: s_cmpk_gt_i32 (0xFFFFFFFF vs -32768 -> True)
        #    becomes OpUGreaterThan (0xFFFFFFFF > 0xFFFF8000 -> True? No, let's pick 0 vs 0xFFFF:
        #    0 > -1 is True in i32, 0 > 0xFFFFFFFF is False in u32).
        spv_signed = support.translate_lines("ps_salu_cmpk_neg1", [
            "s_mov_b32 s0, 0x0",
            "s_cmpk_gt_i32 s0, 0xffff",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpSGreaterThan %bool", spv_signed)
        self.assertEqual(si.run_fragment(spv_signed)["%out_mrt0"][0], 1)
        bad_unsigned = spv_signed.replace("OpSGreaterThan %bool", "OpUGreaterThan %bool", 1)
        self.assertEqual(si.run_fragment(bad_unsigned)["%out_mrt0"][0], 0)

        # 2) Equality inversion: OpIEqual <-> OpINotEqual on s_cmpk_eq_u32 (65535 == 0xFFFF -> True).
        spv_eq = support.translate_lines("ps_salu_cmpk_neg2", [
            "s_mov_b32 s0, 0xffff",
            "s_cmpk_eq_u32 s0, 0xffff",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpIEqual %bool", spv_eq)
        self.assertEqual(si.run_fragment(spv_eq)["%out_mrt0"][0], 1)
        bad_ne = spv_eq.replace("OpIEqual %bool", "OpINotEqual %bool", 1)
        self.assertEqual(si.run_fragment(bad_ne)["%out_mrt0"][0], 0)


if __name__ == "__main__":
    unittest.main()
