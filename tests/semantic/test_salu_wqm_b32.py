"""Semantic interpreter tests for RDNA2 s_wqm_b32 (Whole-Quad Mode).

Verifies both the 32-bit result D (each 4-bit quad becomes 0xF if any bit in the
quad is 1, else 0x0) and the SCC flag (1 when D != 0, 0 when D == 0), across
SGPR, inline, and literal sources, plus in-memory SPIR-V negative controls.
"""
import os
import re
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_wqm_b32(x: int) -> int:
    """Plain-Python reference for s_wqm_b32 on 32-bit data."""
    out = 0
    for q in range(8):
        if (x >> (4 * q)) & 0xF:
            out |= 0xF << (4 * q)
    return out & MASK32


def build_lines(body_lines) -> list:
    """Wrap s_wqm_b32 test instructions with SCC capture and mrt0 export."""
    return list(body_lines) + [
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]


def run_wqm(body_lines) -> tuple:
    """Translate and run lines, returning (D, SCC)."""
    spv = support.translate_lines("ps_salu_wqm_b32", build_lines(body_lines))
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


class SaluWqmB32Tests(unittest.TestCase):
    def test_single_bit_all_quads_and_positions(self):
        for q in range(8):
            for b in range(4):
                x = 1 << (4 * q + b)
                want = 0xF << (4 * q)
                self.assertEqual(want, ref_wqm_b32(x))
                with self.subTest(quad=q, bit=b, x=hex(x)):
                    got = run_wqm([
                        f"s_mov_b32 s2, {x:#x}",
                        "s_wqm_b32 s0, s2",
                    ])
                    self.assertEqual(got, (want, 1))

    def test_boundary_and_non_adjacent_quad_patterns(self):
        cases = [
            0x00000000,
            0xFFFFFFFF,
            0x80000000,
            0x00000001,
            0x10402081,
            0x80040201,
            0x050A0C03,
            0xA0000005,
            0x0F00F000,
            0x12345678,
        ]
        for x in cases:
            want = ref_wqm_b32(x)
            with self.subTest(x=hex(x)):
                got = run_wqm([
                    f"s_mov_b32 s2, {x:#x}",
                    "s_wqm_b32 s0, s2",
                ])
                self.assertEqual(got, (want, int(want != 0)))

    def test_zero_clears_preset_scc(self):
        for src in ("s2", "0"):
            with self.subTest(src=src):
                lines = [
                    "s_mov_b32 s4, 7",
                    "s_cmp_eq_u32 s4, s4",  # pre-set SCC = 1
                ]
                if src == "s2":
                    lines.append("s_mov_b32 s2, 0")
                lines.append(f"s_wqm_b32 s0, {src}")
                self.assertEqual(run_wqm(lines), (0, 0))

    def test_inline_and_literal_sources(self):
        cases = [
            ("0", 0),
            ("1", 1),
            ("8", 8),
            ("16", 16),
            ("-1", 0xFFFFFFFF),
            ("0x80000000", 0x80000000),
            ("0x10400201", 0x10400201),
            ("0x00800002", 0x00800002),
        ]
        for asm_imm, val in cases:
            want = ref_wqm_b32(val)
            with self.subTest(imm=asm_imm):
                got = run_wqm([f"s_wqm_b32 s0, {asm_imm}"])
                self.assertEqual(got, (want, int(want != 0)))

    def test_negative_controls_spirv_mutations(self):
        # 1) Mutate quad mask constant 0x11111111 (286331153 -> 286331152, dropping quad 0).
        spv_mask = support.translate_lines(
            "ps_wqm_neg_mask",
            build_lines(["s_mov_b32 s2, 1", "s_wqm_b32 s0, s2"]),
        )
        self.assertIn("OpConstant %uint 286331153", spv_mask)
        bad_mask = spv_mask.replace("OpConstant %uint 286331153", "OpConstant %uint 286331152", 1)
        got_bad_mask = tuple(si.run_fragment(bad_mask)["%out_mrt0"][:2])
        self.assertNotEqual(got_bad_mask, (0x0000000F, 1))

        # 2) Mutate shift-by-3 in wqm_word (%c3 -> %c2) so bit 3 of quad 0 fails to reach bit 0.
        spv_shift = support.translate_lines(
            "ps_wqm_neg_shift",
            build_lines(["s_mov_b32 s2, 8", "s_wqm_b32 s0, s2"]),
        )
        bad_shift, n_sub = re.subn(
            r"(OpShiftRightLogical %uint %\w+ )%c3\b",
            r"\1%c2",
            spv_shift,
            count=1,
        )
        self.assertEqual(n_sub, 1)
        got_bad_shift = tuple(si.run_fragment(bad_shift)["%out_mrt0"][:2])
        self.assertNotEqual(got_bad_shift, (0x0000000F, 1))

        # 3) Mutate SCC nonzero check (OpINotEqual -> OpIEqual) on s_wqm_b32.
        spv_scc = support.translate_lines(
            "ps_wqm_neg_scc",
            build_lines(["s_mov_b32 s2, 1", "s_wqm_b32 s0, s2"]),
        )
        self.assertIn("OpINotEqual %bool", spv_scc)
        bad_scc = spv_scc.replace("OpINotEqual %bool", "OpIEqual %bool", 1)
        got_bad_scc = tuple(si.run_fragment(bad_scc)["%out_mrt0"][:2])
        self.assertNotEqual(got_bad_scc, (0x0000000F, 1))


if __name__ == "__main__":
    unittest.main()
