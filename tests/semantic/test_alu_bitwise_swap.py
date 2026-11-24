"""Translator-to-interpreter semantic tests for v_swap_b32, v_xnor_b32, v_or3_b32, and v_xor3_b32.

RDNA2 ISA:
  v_swap_b32  D0, D1        exchange D0 and D1 per active lane (inactive lanes preserve their values)
  v_xnor_b32  D, S0, S1     D = ~(S0 ^ S1)
  v_or3_b32   D, S0, S1, S2 D = S0 | S1 | S2
  v_xor3_b32  D, S0, S1, S2 D = S0 ^ S1 ^ S2

All tests translate synthetic hand-written RDNA2 disassembly via tools/rdna2spirv.py and execute the emitted
SPIR-V in spirv_mini_interp.py (single-lane) and spirv_wave_interp.py (32-lane wave with partial EXEC mask).
Standard library only; no game data.
"""
import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_wave_interp as wi  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_xnor(a, b):
    return (~(a ^ b)) & MASK32


def ref_or3(a, b, c):
    return (a | b | c) & MASK32


def ref_xor3(a, b, c):
    return (a ^ b ^ c) & MASK32


XNOR_CONSTANTS = [
    (0x00000000, 0x00000000, 0xFFFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    (0x00000000, 0xFFFFFFFF, 0x00000000),
    (0xFFFFFFFF, 0x00000000, 0x00000000),
    (0xAAAAAAAA, 0x55555555, 0x00000000),
    (0xAAAAAAAA, 0xAAAAAAAA, 0xFFFFFFFF),
    (0x12345678, 0x9ABCDEF0, 0x77777777),
    (0x80000000, 0x00000001, 0x7FFFFFFE),
    (0xDEADBEEF, 0xCAFEBABE, 0xEBACFBAE),
]

OR3_CONSTANTS = [
    (0x00000000, 0x00000000, 0x00000000, 0x00000000),
    (0x000000FF, 0x0000FF00, 0xFFFF0000, 0xFFFFFFFF),
    (0x0000000F, 0x0000000F, 0x000000F0, 0x000000FF),
    (0x80000000, 0x00000001, 0x00010000, 0x80010001),
    (0x12340000, 0x00005678, 0x00000000, 0x12345678),
    (0xAAAAAAAA, 0x55555555, 0x00000000, 0xFFFFFFFF),
    (0xDEADBEEF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF),
]

XOR3_CONSTANTS = [
    (0x00000000, 0x00000000, 0x00000000, 0x00000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000),
    (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
    (0xAAAAAAAA, 0x55555555, 0xFFFFFFFF, 0x00000000),
    (0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF, 0x0FF0F00F),
    (0x80000000, 0x00000001, 0x00000001, 0x80000000),
]


def run_ps_lines(name, body_lines, export_regs=("v0", "v0", "v0", "v0")):
    lines = list(body_lines) + [f"exp mrt0 {', '.join(export_regs)} done vm", "s_endpgm"]
    spv = translate_lines(name, lines)
    return mrt0_words(spv), spv


class ReferenceConstantsTests(unittest.TestCase):
    def test_hand_computed_constants(self):
        for a, b, expected in XNOR_CONSTANTS:
            with self.subTest(op="xnor", a=hex(a), b=hex(b)):
                self.assertEqual(ref_xnor(a, b), expected)
        for a, b, c, expected in OR3_CONSTANTS:
            with self.subTest(op="or3", a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_or3(a, b, c), expected)
        for a, b, c, expected in XOR3_CONSTANTS:
            with self.subTest(op="xor3", a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_xor3(a, b, c), expected)


class VSwapB32Tests(unittest.TestCase):
    def test_pair_swap_boundaries(self):
        pairs = [
            (0x00000000, 0xFFFFFFFF),
            (0x80000000, 0x00000001),
            (0x12345678, 0xDEADBEEF),
            (0xCAFEBABE, 0xCAFEBABE),
        ]
        for mnemonic in ("v_swap_b32", "v_swap_b32_e32"):
            for a, b in pairs:
                with self.subTest(mnemonic=mnemonic, a=hex(a), b=hex(b)):
                    words, _ = run_ps_lines(
                        "ps_swap_pair",
                        [
                            f"v_mov_b32 v1, {a:#x}",
                            f"v_mov_b32 v2, {b:#x}",
                            f"{mnemonic} v1, v2",
                        ],
                        export_regs=("v1", "v2", "v1", "v2"),
                    )
                    self.assertEqual(words, [b, a, b, a])

    def test_self_swap_is_identity(self):
        for val in (0x00000000, 0x80000000, 0x12345678, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                words, _ = run_ps_lines(
                    "ps_swap_self",
                    [f"v_mov_b32 v1, {val:#x}", "v_swap_b32 v1, v1"],
                    export_regs=("v1", "v1", "v1", "v1"),
                )
                self.assertEqual(words, [val, val, val, val])

    def test_double_swap_involution_and_cyclic_rotation(self):
        words, _ = run_ps_lines(
            "ps_swap_double",
            [
                "v_mov_b32 v1, 0x11223344",
                "v_mov_b32 v2, 0xAABBCCDD",
                "v_swap_b32 v1, v2",
                "v_swap_b32 v1, v2",
            ],
            export_regs=("v1", "v2", "v1", "v2"),
        )
        self.assertEqual(words, [0x11223344, 0xAABBCCDD, 0x11223344, 0xAABBCCDD])

        rot_words, _ = run_ps_lines(
            "ps_swap_cycle",
            [
                "v_mov_b32 v1, 0x11111111",
                "v_mov_b32 v2, 0x22222222",
                "v_mov_b32 v3, 0x33333333",
                "v_swap_b32 v1, v2",
                "v_swap_b32 v2, v3",
            ],
            export_regs=("v1", "v2", "v3", "v3"),
        )
        self.assertEqual(rot_words, [0x22222222, 0x33333333, 0x11111111, 0x11111111])

    def test_partial_exec_mask_wave32_via_exec_const(self):
        masks = (0xA55A0F0F, 0x00000001, 0x80000000, 0x00000000, 0xFFFFFFFF)
        for mask in masks:
            with self.subTest(mask=hex(mask)):
                spv = translate_lines(
                    "ps_swap_exec_mask",
                    [
                        "v_mbcnt_lo_u32_b32 v0, -1, 0",
                        "v_add_nc_u32 v1, 0x1000, v0",
                        "v_add_nc_u32 v2, 0x2000, v0",
                        "s_mov_b32 s1, exec_lo",
                        f"s_mov_b32 exec_lo, {mask:#x}",
                        "v_swap_b32 v1, v2",
                        "s_mov_b32 exec_lo, s1",
                        "exp mrt0 v1, v2, v0, v0 done vm",
                        "s_endpgm",
                    ],
                )
                outputs = wi.run_wave(spv, wave=32)["outputs"]
                self.assertEqual(len(outputs), 32)
                for lane, out in enumerate(outputs):
                    self.assertIsNotNone(out)
                    v1_got, v2_got, lane_got, _ = out["%out_mrt0"]
                    self.assertEqual(lane_got, lane)
                    a = 0x1000 + lane
                    b = 0x2000 + lane
                    if (mask >> lane) & 1:
                        self.assertEqual((v1_got, v2_got), (b, a), f"lane {lane} active")
                    else:
                        self.assertEqual((v1_got, v2_got), (a, b), f"lane {lane} inactive")

    def test_partial_exec_mask_wave32_via_vcmp(self):
        spv = translate_lines(
            "ps_swap_vcmp_exec",
            [
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                "v_add_nc_u32 v1, 0x3000, v0",
                "v_add_nc_u32 v2, 0x4000, v0",
                "v_and_b32_e32 v3, 1, v0",
                "v_cmp_eq_u32_e32 vcc_lo, 0, v3",
                "s_mov_b32 s1, exec_lo",
                "s_and_b32 exec_lo, exec_lo, vcc_lo",
                "v_swap_b32 v1, v2",
                "s_mov_b32 exec_lo, s1",
                "exp mrt0 v1, v2, v0, v0 done vm",
                "s_endpgm",
            ],
        )
        outputs = wi.run_wave(spv, wave=32)["outputs"]
        for lane, out in enumerate(outputs):
            self.assertIsNotNone(out)
            v1_got, v2_got, lane_got, _ = out["%out_mrt0"]
            self.assertEqual(lane_got, lane)
            a = 0x3000 + lane
            b = 0x4000 + lane
            if lane % 2 == 0:
                self.assertEqual((v1_got, v2_got), (b, a))
            else:
                self.assertEqual((v1_got, v2_got), (a, b))


class VXnorB32Tests(unittest.TestCase):
    def test_boundary_constants(self):
        for a, b, expected in XNOR_CONSTANTS:
            with self.subTest(a=hex(a), b=hex(b)):
                words, _ = run_ps_lines(
                    "ps_xnor_const",
                    [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", "v_xnor_b32 v0, v1, v2"],
                )
                self.assertEqual(words[0], expected)

    def test_algebraic_identities_and_commutativity(self):
        for x in (0x00000000, 0x00000001, 0x12345678, 0x80000000, 0xFFFFFFFF):
            inv_x = (~x) & MASK32
            with self.subTest(x=hex(x)):
                w_self, _ = run_ps_lines(
                    "ps_xnor_id", [f"v_mov_b32 v1, {x:#x}", "v_xnor_b32 v0, v1, v1"]
                )
                self.assertEqual(w_self[0], MASK32)
                w_comp, _ = run_ps_lines(
                    "ps_xnor_comp",
                    [f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {inv_x:#x}", "v_xnor_b32 v0, v1, v2"],
                )
                self.assertEqual(w_comp[0], 0)
                w_zero, _ = run_ps_lines(
                    "ps_xnor_zero", [f"v_mov_b32 v1, {x:#x}", "v_xnor_b32 v0, v1, 0"]
                )
                self.assertEqual(w_zero[0], inv_x)
                w_ones, _ = run_ps_lines(
                    "ps_xnor_ones", [f"v_mov_b32 v1, {x:#x}", "v_xnor_b32 v0, v1, -1"]
                )
                self.assertEqual(w_ones[0], x)

        a, b = 0x13579BDF, 0x2468ACE0
        w_ab, _ = run_ps_lines(
            "ps_xnor_ab", [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", "v_xnor_b32 v0, v1, v2"]
        )
        w_ba, _ = run_ps_lines(
            "ps_xnor_ba", [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", "v_xnor_b32 v0, v2, v1"]
        )
        self.assertEqual(w_ab[0], w_ba[0])
        self.assertEqual(w_ab[0], ref_xnor(a, b))

    def test_operand_forms_sgpr_inline_literal(self):
        cases = [
            (["s_mov_b32 s0, 0x12345678", "v_mov_b32 v1, 0xF0F0F0F0", "v_xnor_b32 v0, s0, v1"], 0x12345678, 0xF0F0F0F0),
            (["v_mov_b32 v1, 0x00FF00FF", "v_xnor_b32 v0, 15, v1"], 15, 0x00FF00FF),
            (["v_mov_b32 v1, 0xCAFEBABE", "v_xnor_b32 v0, 0xDEADBEEF, v1"], 0xDEADBEEF, 0xCAFEBABE),
            (["v_mov_b32 v1, 0xCAFEBABE", "v_xnor_b32_e32 v0, 0xDEADBEEF, v1"], 0xDEADBEEF, 0xCAFEBABE),
        ]
        for lines, a, b in cases:
            with self.subTest(inst=lines[-1]):
                words, _ = run_ps_lines("ps_xnor_forms", lines)
                self.assertEqual(words[0], ref_xnor(a, b))

    def test_seeded_random_pairs(self):
        rng = random.Random(0x584E4F52)
        for _ in range(12):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            with self.subTest(a=hex(a), b=hex(b)):
                words, _ = run_ps_lines(
                    "ps_xnor_rand",
                    [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", "v_xnor_b32 v0, v1, v2"],
                )
                self.assertEqual(words[0], ref_xnor(a, b))


class VOr3B32Tests(unittest.TestCase):
    def test_boundary_constants(self):
        for a, b, c, expected in OR3_CONSTANTS:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                words, _ = run_ps_lines(
                    "ps_or3_const",
                    [
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        "v_or3_b32 v0, v1, v2, v3",
                    ],
                )
                self.assertEqual(words[0], expected)

    def test_permutation_invariance_and_operand_mixes(self):
        a, b, c = 0x0F000001, 0x00F00010, 0x000F0100
        expected = ref_or3(a, b, c)
        for s0, s1, s2 in (("v1", "v2", "v3"), ("v2", "v3", "v1"), ("v3", "v1", "v2")):
            with self.subTest(order=(s0, s1, s2)):
                words, _ = run_ps_lines(
                    "ps_or3_perm",
                    [
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        f"v_or3_b32 v0, {s0}, {s1}, {s2}",
                    ],
                )
                self.assertEqual(words[0], expected)

        mix_words, _ = run_ps_lines(
            "ps_or3_mix",
            [
                "s_mov_b32 s0, 0xF0000000",
                "v_mov_b32 v1, 0x00FF0000",
                "v_or3_b32 v0, s0, v1, 7",
            ],
        )
        self.assertEqual(mix_words[0], ref_or3(0xF0000000, 0x00FF0000, 7))

        lit_words, _ = run_ps_lines(
            "ps_or3_lit",
            [
                "v_mov_b32 v1, 0x0000FF00",
                "v_mov_b32 v2, 0x000000FF",
                "v_or3_b32 v0, 0xCAFE0000, v1, v2",
            ],
        )
        self.assertEqual(lit_words[0], 0xCAFEFFFF)

    def test_seeded_random_triples(self):
        rng = random.Random(0x4F523332)
        for _ in range(12):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            c = rng.getrandbits(32)
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                words, _ = run_ps_lines(
                    "ps_or3_rand",
                    [
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        "v_or3_b32 v0, v1, v2, v3",
                    ],
                )
                self.assertEqual(words[0], ref_or3(a, b, c))


class VXor3B32Tests(unittest.TestCase):
    def test_boundary_constants(self):
        for a, b, c, expected in XOR3_CONSTANTS:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                words, _ = run_ps_lines(
                    "ps_xor3_const",
                    [
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        "v_xor3_b32 v0, v1, v2, v3",
                    ],
                )
                self.assertEqual(words[0], expected)

    def test_cancellation_and_permutation_identities(self):
        x, y = 0x12345678, 0xDEADBEEF
        w_cancel, _ = run_ps_lines(
            "ps_xor3_cancel",
            [f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", "v_xor3_b32 v0, v1, v2, v2"],
        )
        self.assertEqual(w_cancel[0], x)

        w_triple, _ = run_ps_lines(
            "ps_xor3_triple",
            [f"v_mov_b32 v1, {x:#x}", "v_xor3_b32 v0, v1, v1, v1"],
        )
        self.assertEqual(w_triple[0], x)

        w_zero, _ = run_ps_lines(
            "ps_xor3_zero",
            [f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}", "v_xor3_b32 v0, v1, v2, 0"],
        )
        self.assertEqual(w_zero[0], x ^ y)

        c = 0x0F0F0F0F
        expected = ref_xor3(x, y, c)
        for s0, s1, s2 in (("v1", "v2", "v3"), ("v2", "v3", "v1"), ("v3", "v1", "v2")):
            with self.subTest(order=(s0, s1, s2)):
                words, _ = run_ps_lines(
                    "ps_xor3_perm",
                    [
                        f"v_mov_b32 v1, {x:#x}",
                        f"v_mov_b32 v2, {y:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        f"v_xor3_b32 v0, {s0}, {s1}, {s2}",
                    ],
                )
                self.assertEqual(words[0], expected)

    def test_operand_mixes_and_seeded_random_triples(self):
        mix_words, _ = run_ps_lines(
            "ps_xor3_mix",
            [
                "s_mov_b32 s0, 0x12340000",
                "v_mov_b32 v1, 0x00005678",
                "v_xor3_b32 v0, s0, v1, 15",
            ],
        )
        self.assertEqual(mix_words[0], ref_xor3(0x12340000, 0x00005678, 15))

        lit_words, _ = run_ps_lines(
            "ps_xor3_lit",
            [
                "v_mov_b32 v1, 0xFFFF0000",
                "v_mov_b32 v2, 0x0000FFFF",
                "v_xor3_b32 v0, 0xA5A5A5A5, v1, v2",
            ],
        )
        self.assertEqual(lit_words[0], ref_xor3(0xA5A5A5A5, 0xFFFF0000, 0x0000FFFF))

        rng = random.Random(0x584F5233)
        for _ in range(12):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            c = rng.getrandbits(32)
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                words, _ = run_ps_lines(
                    "ps_xor3_rand",
                    [
                        f"v_mov_b32 v1, {a:#x}",
                        f"v_mov_b32 v2, {b:#x}",
                        f"v_mov_b32 v3, {c:#x}",
                        "v_xor3_b32 v0, v1, v2, v3",
                    ],
                )
                self.assertEqual(words[0], ref_xor3(a, b, c))


class SpirvStructureTests(unittest.TestCase):
    def test_swap_emits_op_copy_object(self):
        _, spv = run_ps_lines(
            "ps_struct_swap",
            ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_swap_b32 v1, v2"],
            export_regs=("v1", "v2", "v1", "v2"),
        )
        self.assertGreaterEqual(spv.count("= OpCopyObject %uint"), 2)

    def test_xnor_emits_op_not_after_op_bitwise_xor(self):
        _, spv = run_ps_lines(
            "ps_struct_xnor",
            ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_xnor_b32 v0, v1, v2"],
        )
        match = re.search(r"(%\w+)\s*=\s*OpBitwiseXor %uint.*\n\s*(%\w+)\s*=\s*OpNot %uint \1\b", spv)
        self.assertIsNotNone(match, "expected OpNot %uint directly consuming OpBitwiseXor result")

    def test_or3_and_xor3_emit_chained_bitwise_ops(self):
        _, spv_or = run_ps_lines(
            "ps_struct_or3",
            ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 4", "v_or3_b32 v0, v1, v2, v3"],
        )
        self.assertIsNotNone(
            re.search(r"(%\w+)\s*=\s*OpBitwiseOr %uint.*\n\s*%\w+\s*=\s*OpBitwiseOr %uint \1\b", spv_or)
        )
        _, spv_xor = run_ps_lines(
            "ps_struct_xor3",
            ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 4", "v_xor3_b32 v0, v1, v2, v3"],
        )
        self.assertIsNotNone(
            re.search(r"(%\w+)\s*=\s*OpBitwiseXor %uint.*\n\s*%\w+\s*=\s*OpBitwiseXor %uint \1\b", spv_xor)
        )


if __name__ == "__main__":
    unittest.main()
