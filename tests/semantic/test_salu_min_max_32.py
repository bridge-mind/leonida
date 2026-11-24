"""Semantic interpreter tests for RDNA2 32-bit scalar integer min/max ops.

Covers s_min_i32, s_max_i32, s_min_u32 and s_max_u32:
- Result value and SCC flag across signed and unsigned 32-bit boundaries.
- Tie semantics (SCC == 0 when S0 == S1) including downstream s_cbranch_scc1 control flow.
- Translation-time constant folding and sgpr_imm propagation across chained SALU ops.
- Mixed SGPR, inline constant and 32-bit literal operands, plus destination aliasing.
- Negative controls verifying SPIR-V signed/unsigned and strict-inequality mutations fail.

Each case translates a hand-written pixel shader through tools/rdna2spirv.py and executes it in
tests/semantic/spirv_mini_interp.py. Standard library only; no GPU, no game files.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

OPS = ("s_min_i32", "s_max_i32", "s_min_u32", "s_max_u32")
BOUNDARY_VALUES = (
    0x00000000,
    0x00000001,
    0x00000005,
    0x7FFFFFFE,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
)


def s32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def ref_salu_minmax(op, a, b):
    """Independent RDNA2 ISA model returning (dst_u32, scc)."""
    ua, ub = a & 0xFFFFFFFF, b & 0xFFFFFFFF
    sa, sb = s32(ua), s32(ub)
    if op == "s_min_i32":
        scc = 1 if sa < sb else 0
        return (ua if scc else ub), scc
    if op == "s_max_i32":
        scc = 1 if sa > sb else 0
        return (ua if scc else ub), scc
    if op == "s_min_u32":
        scc = 1 if ua < ub else 0
        return (ua if scc else ub), scc
    if op == "s_max_u32":
        scc = 1 if ua > ub else 0
        return (ua if scc else ub), scc
    raise ValueError(f"unexpected op {op}")


def translate_salu(lines):
    return support.translate_lines("ps_salu_min_max_32", lines)


def run_op(op, src0="s0", src1="s1", push=None):
    lines = [
        f"{op} s4, {src0}, {src1}",
        "s_cselect_b32 s5, 1, 0",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_salu(lines)
    out = support.mrt0_words(spv, push_constants=push)
    return out[0], out[1], spv


class SaluMinMax32Tests(unittest.TestCase):
    def test_reference_sanity(self):
        self.assertEqual(ref_salu_minmax("s_min_i32", 0xFFFFFFFF, 1), (0xFFFFFFFF, 1))
        self.assertEqual(ref_salu_minmax("s_max_i32", 0xFFFFFFFF, 1), (1, 0))
        self.assertEqual(ref_salu_minmax("s_min_u32", 0xFFFFFFFF, 1), (1, 0))
        self.assertEqual(ref_salu_minmax("s_max_u32", 0xFFFFFFFF, 1), (0xFFFFFFFF, 1))
        for op in OPS:
            self.assertEqual(ref_salu_minmax(op, 5, 5), (5, 0))
            self.assertEqual(ref_salu_minmax(op, 0x80000000, 0x80000000), (0x80000000, 0))

    def test_boundary_grid_values_and_scc(self):
        for op in OPS:
            spv = translate_salu([
                f"{op} s4, s0, s1",
                "s_cselect_b32 s5, 1, 0",
                "v_mov_b32 v0, s4",
                "v_mov_b32 v1, s5",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
                "exp mrt0 v0, v1, v2, v3 done vm",
                "s_endpgm",
            ])
            for a in BOUNDARY_VALUES:
                for b in BOUNDARY_VALUES:
                    with self.subTest(op=op, a=hex(a), b=hex(b)):
                        out = support.mrt0_words(spv, push_constants=[a, b])
                        self.assertEqual((out[0], out[1]), ref_salu_minmax(op, a, b))

    def test_cbranch_scc_on_tie_and_inequality(self):
        for op in OPS:
            spv = translate_salu([
                "s_mov_b32 s4, 10 ; encoding: [0x8a,0x03,0x84,0xbe]",
                f"{op} s5, s0, s1 ; encoding: [0x00,0x01,0x05,0x80]",
                "s_cbranch_scc1 1 ; encoding: [0x01,0x00,0x85,0xbf]",
                "s_mov_b32 s4, 20 ; encoding: [0x94,0x03,0x84,0xbe]",
                "v_mov_b32 v0, s5 ; encoding: [0x05,0x02,0x00,0x7e]",
                "v_mov_b32 v1, s4 ; encoding: [0x04,0x02,0x00,0x7e]",
                "v_mov_b32 v2, 0 ; encoding: [0x80,0x00,0x00,0x7e]",
                "v_mov_b32 v3, 0 ; encoding: [0x80,0x00,0x00,0x7e]",
                "exp mrt0 v0, v1, v2, v3 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
                "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
            ])
            pairs = (
                (0, 0),
                (5, 5),
                (0x80000000, 0x80000000),
                (0xFFFFFFFF, 0xFFFFFFFF),
                (0xFFFFFFFF, 1),
                (1, 0xFFFFFFFF),
                (0x7FFFFFFF, 0x80000000),
                (0x80000000, 0x7FFFFFFF),
            )
            for a, b in pairs:
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    want_val, want_scc = ref_salu_minmax(op, a, b)
                    want_branch_val = 10 if want_scc else 20
                    out = support.mrt0_words(spv, push_constants=[a, b])
                    self.assertEqual((out[0], out[1]), (want_val, want_branch_val))

    def test_constant_folding_and_sgpr_imm_propagation(self):
        cases = (
            ("s_min_i32", "-5", "1", -5, 1),
            ("s_min_i32", "5", "5", 5, 5),
            ("s_max_i32", "-5", "-1", -5, -1),
            ("s_max_i32", "-1", "-1", -1, -1),
            ("s_min_i32", "0x80000000", "0x7fffffff", 0x80000000, 0x7FFFFFFF),
            ("s_max_i32", "0x7fffffff", "-1", 0x7FFFFFFF, 0xFFFFFFFF),
            ("s_min_u32", "-1", "5", 0xFFFFFFFF, 5),
            ("s_max_u32", "-1", "5", 0xFFFFFFFF, 5),
            ("s_min_u32", "5", "5", 5, 5),
            ("s_max_u32", "0x80000000", "1", 0x80000000, 1),
        )
        for op, a_tok, b_tok, a_val, b_val in cases:
            with self.subTest(op=op, a=a_tok, b=b_tok):
                lines = [
                    f"s_mov_b32 s0, {a_tok}",
                    f"s_mov_b32 s1, {b_tok}",
                    f"{op} s4, s0, s1",
                    "s_cselect_b32 s5, 1, 0",
                    "s_add_u32 s6, s4, 7",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, s5",
                    "v_mov_b32 v2, s6",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_salu(lines)
                want_val, want_scc = ref_salu_minmax(op, a_val, b_val)
                want_add = (want_val + 7) & 0xFFFFFFFF
                out = support.mrt0_words(spv)
                self.assertEqual((out[0], out[1], out[2]), (want_val, want_scc, want_add))
                if op.endswith("_i32"):
                    self.assertNotIn("SMin", spv)
                    self.assertNotIn("SMax", spv)

        # Propagated sgpr_imm from s_min_u32 / s_max_u32 eliminates runtime SMEM bounds branches.
        for op, a_tok, b_tok in (
            ("s_min_u32", "64", "200"),
            ("s_max_u32", "16", "64"),
        ):
            with self.subTest(op=op, check="smem_const_prop"):
                buf_spv = translate_salu([
                    "s_mov_b32 s1, 0",
                    "s_mov_b32 s2, 256",
                    f"s_mov_b32 s4, {a_tok}",
                    f"s_mov_b32 s5, {b_tok}",
                    f"{op} s6, s4, s5",
                    "s_buffer_load_dword s8, s[0:3], s6",
                    "v_mov_b32 v0, s8",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ])
                self.assertNotIn("OpSelectionMerge", buf_spv)
                self.assertNotIn("OpBranchConditional", buf_spv)
                self.assertNotIn("OpPhi", buf_spv)

        # Multi-step chain through s_min_i32 -> s_max_i32 -> s_add_u32 with constant propagation.
        chain_spv = translate_salu([
            "s_mov_b32 s0, -10",
            "s_mov_b32 s1, 3",
            "s_mov_b32 s2, -4",
            "s_min_i32 s4, s0, s1",   # -10
            "s_max_i32 s5, s4, s2",   # -4
            "s_cselect_b32 s6, 1, 0", # 0 (s4 < s2 -> s2 chosen)
            "s_add_u32 s7, s5, 10",   # 6
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, s7",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertEqual(support.mrt0_words(chain_spv), [0xFFFFFFF6, 0xFFFFFFFC, 0, 6])
        self.assertNotIn("SMin", chain_spv)
        self.assertNotIn("SMax", chain_spv)

    def test_mixed_operands_and_aliasing(self):
        mixed_cases = (
            ("s_min_i32", "s0", "-16", [5], ref_salu_minmax("s_min_i32", 5, 0xFFFFFFF0)),
            ("s_max_i32", "-16", "s0", [0xFFFFFFE0], ref_salu_minmax("s_max_i32", 0xFFFFFFF0, 0xFFFFFFE0)),
            ("s_min_u32", "s0", "0x80000000", [0x7FFFFFFF], ref_salu_minmax("s_min_u32", 0x7FFFFFFF, 0x80000000)),
            ("s_max_u32", "0x80000000", "s0", [0x7FFFFFFF], ref_salu_minmax("s_max_u32", 0x80000000, 0x7FFFFFFF)),
            ("s_min_i32", "0x12345678", "s0", [0x12345678], ref_salu_minmax("s_min_i32", 0x12345678, 0x12345678)),
            ("s_max_u32", "s0", "0", [0], ref_salu_minmax("s_max_u32", 0, 0)),
        )
        for op, src0, src1, push, want in mixed_cases:
            with self.subTest(op=op, src0=src0, src1=src1):
                val, scc, _ = run_op(op, src0=src0, src1=src1, push=push)
                self.assertEqual((val, scc), want)

        # Destination aliasing src0, src1, or both.
        a, b = 0xFFFFFFF0, 0x00000007
        for op in OPS:
            want_ab = ref_salu_minmax(op, a, b)
            want_aa = ref_salu_minmax(op, a, a)
            for pattern, push, want in (
                (f"{op} s0, s0, s1", [a, b], want_ab),
                (f"{op} s1, s0, s1", [a, b], want_ab),
                (f"{op} s0, s0, s0", [a, b], want_aa),
            ):
                dst_reg = pattern.split()[1].rstrip(",")
                spv = translate_salu([
                    pattern,
                    "s_cselect_b32 s4, 1, 0",
                    f"v_mov_b32 v0, {dst_reg}",
                    "v_mov_b32 v1, s4",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ])
                with self.subTest(pattern=pattern):
                    out = support.mrt0_words(spv, push_constants=push)
                    self.assertEqual((out[0], out[1]), want)

    def test_seeded_random_cases(self):
        rng = random.Random(0x534D4D33)
        for op in OPS:
            spv = translate_salu([
                f"{op} s4, s0, s1",
                "s_cselect_b32 s5, 1, 0",
                "v_mov_b32 v0, s4",
                "v_mov_b32 v1, s5",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
                "exp mrt0 v0, v1, v2, v3 done vm",
                "s_endpgm",
            ])
            for i in range(48):
                a = rng.getrandbits(32)
                b = a if i % 8 == 0 else rng.getrandbits(32)
                with self.subTest(op=op, case=i):
                    out = support.mrt0_words(spv, push_constants=[a, b])
                    self.assertEqual((out[0], out[1]), ref_salu_minmax(op, a, b))

    def test_negative_controls(self):
        # 1. Signed s_min_i32 tie comparison mutated to OpSLessThanEqual must corrupt SCC on ties.
        _, _, spv_min_i = run_op("s_min_i32", push=[5, 5])
        self.assertIn("OpSLessThan ", spv_min_i)
        self.assertIn("SMin", spv_min_i)
        bad_tie = spv_min_i.replace("OpSLessThan ", "OpSLessThanEqual ")
        self.assertEqual(support.mrt0_words(spv_min_i, push_constants=[5, 5])[:2], [5, 0])
        self.assertNotEqual(support.mrt0_words(bad_tie, push_constants=[5, 5])[:2], [5, 0])

        # 2. Signed SMin mutated to unsigned UMin must fail on (-5, 1).
        bad_sign = spv_min_i.replace("SMin", "UMin")
        self.assertEqual(support.mrt0_words(spv_min_i, push_constants=[0xFFFFFFFB, 1])[:2], [0xFFFFFFFB, 1])
        self.assertNotEqual(support.mrt0_words(bad_sign, push_constants=[0xFFFFFFFB, 1])[:2], [0xFFFFFFFB, 1])

        # 3. Unsigned s_min_u32 comparison mutated to signed OpSLessThan must corrupt SCC on (0xFFFFFFFF, 1).
        _, _, spv_min_u = run_op("s_min_u32", push=[0xFFFFFFFF, 1])
        self.assertIn("OpULessThan ", spv_min_u)
        bad_ucmp = spv_min_u.replace("OpULessThan ", "OpSLessThan ")
        self.assertEqual(support.mrt0_words(spv_min_u, push_constants=[0xFFFFFFFF, 1])[:2], [1, 0])
        self.assertNotEqual(support.mrt0_words(bad_ucmp, push_constants=[0xFFFFFFFF, 1])[:2], [1, 0])


if __name__ == "__main__":
    unittest.main()
