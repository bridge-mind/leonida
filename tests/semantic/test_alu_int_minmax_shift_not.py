"""Translator-to-interpreter semantic tests for integer VALU min/max, arithmetic shift, and bitwise NOT:

  v_min_i32      D = min_s32(S0, S1)
  v_max_i32      D = max_s32(S0, S1)
  v_min_u32      D = min_u32(S0, S1)
  v_max_u32      D = max_u32(S0, S1)
  v_ashrrev_i32  D = sext(S1) >> (S0 & 31)   (note: S0 is shift amount, S1 is value)
  v_not_b32      D = ~S0

Every test translates synthetic hand-written RDNA2 disassembly via tools/rdna2spirv.py and executes the emitted
SPIR-V in tests/semantic/spirv_mini_interp.py. Standard library only; no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def s32(x):
    x &= MASK32
    return x - (1 << 32) if (x & 0x80000000) else x


def ref_min_i32(a, b):
    return (a if s32(a) <= s32(b) else b) & MASK32


def ref_max_i32(a, b):
    return (a if s32(a) >= s32(b) else b) & MASK32


def ref_min_u32(a, b):
    return (a if (a & MASK32) <= (b & MASK32) else b) & MASK32


def ref_max_u32(a, b):
    return (a if (a & MASK32) >= (b & MASK32) else b) & MASK32


def ref_ashrrev_i32(shift, val):
    return (s32(val) >> (shift & 31)) & MASK32


def ref_not_b32(val):
    return (~val) & MASK32


BOUNDARIES = (0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)

MINMAX_HAND_CONSTANTS = [
    ("v_min_i32", 0x80000000, 0x7FFFFFFF, 0x80000000),
    ("v_min_i32", 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF),
    ("v_min_i32", 0xFFFFFFFF, 0x00000001, 0xFFFFFFFF),
    ("v_min_i32", 0x80000000, 0xFFFFFFFF, 0x80000000),
    ("v_max_i32", 0x80000000, 0x7FFFFFFF, 0x7FFFFFFF),
    ("v_max_i32", 0xFFFFFFFF, 0x00000000, 0x00000000),
    ("v_max_i32", 0xFFFFFFFF, 0x00000001, 0x00000001),
    ("v_max_i32", 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
    ("v_min_u32", 0x80000000, 0x7FFFFFFF, 0x7FFFFFFF),
    ("v_min_u32", 0xFFFFFFFF, 0x00000000, 0x00000000),
    ("v_min_u32", 0xFFFFFFFF, 0x00000001, 0x00000001),
    ("v_min_u32", 0x80000000, 0xFFFFFFFF, 0x80000000),
    ("v_max_u32", 0x80000000, 0x7FFFFFFF, 0x80000000),
    ("v_max_u32", 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF),
    ("v_max_u32", 0xFFFFFFFF, 0x00000001, 0xFFFFFFFF),
    ("v_max_u32", 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
]

ASHRREV_HAND_CONSTANTS = [
    (0, 0x80000000, 0x80000000),
    (1, 0x80000000, 0xC0000000),
    (31, 0x80000000, 0xFFFFFFFF),
    (31, 0x80000001, 0xFFFFFFFF),
    (4, 0xFFFFFFFF, 0xFFFFFFFF),
    (4, 0x7FFFFFFF, 0x07FFFFFF),
    (31, 0x7FFFFFFF, 0x00000000),
    (32, 0x80000000, 0x80000000),
    (33, 0x80000000, 0xC0000000),
    (0xFFFFFFFF, 0x80000001, 0xFFFFFFFF),
    (0xFFFFFFFF, 0x7FFFFFFF, 0x00000000),
]

NOT_HAND_CONSTANTS = [
    (0x00000000, 0xFFFFFFFF),
    (0xFFFFFFFF, 0x00000000),
    (0x55555555, 0xAAAAAAAA),
    (0xAAAAAAAA, 0x55555555),
    (0x80000000, 0x7FFFFFFF),
    (0x12345678, 0xEDCBA987),
]

REF_MAP = {
    "v_min_i32": ref_min_i32,
    "v_max_i32": ref_max_i32,
    "v_min_u32": ref_min_u32,
    "v_max_u32": ref_max_u32,
}


def run_ps(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    return words[0], spv


class ReferenceConstantsTests(unittest.TestCase):
    def test_hand_computed_constants(self):
        for op, a, b, expected in MINMAX_HAND_CONSTANTS:
            with self.subTest(op=op, a=hex(a), b=hex(b)):
                self.assertEqual(REF_MAP[op](a, b), expected)
        for sh, val, expected in ASHRREV_HAND_CONSTANTS:
            with self.subTest(op="v_ashrrev_i32", sh=hex(sh), val=hex(val)):
                self.assertEqual(ref_ashrrev_i32(sh, val), expected)
        for val, expected in NOT_HAND_CONSTANTS:
            with self.subTest(op="v_not_b32", val=hex(val)):
                self.assertEqual(ref_not_b32(val), expected)


class MinMaxInt32Tests(unittest.TestCase):
    def test_signed_vs_unsigned_boundary_matrix(self):
        for op, ref in REF_MAP.items():
            for a in BOUNDARIES:
                for b in BOUNDARIES:
                    with self.subTest(op=op, a=hex(a), b=hex(b)):
                        got, _ = run_ps(
                            "ps_minmax_mat",
                            [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"{op} v0, v1, v2"],
                        )
                        self.assertEqual(got, ref(a, b))

    def test_commutativity_and_equal_operands(self):
        pairs = [(0x80000000, 0x7FFFFFFF), (0xFFFFFFFF, 1), (0x12345678, 0x87654321), (42, 42)]
        for op, ref in REF_MAP.items():
            for a, b in pairs:
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    g1, _ = run_ps(
                        "ps_mm_1", [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"{op} v0, v1, v2"]
                    )
                    g2, _ = run_ps(
                        "ps_mm_2", [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"{op} v0, v2, v1"]
                    )
                    self.assertEqual(g1, g2)
                    self.assertEqual(g1, ref(a, b))

    def test_operand_forms_sgpr_literal_e32_e64(self):
        for op, ref in REF_MAP.items():
            for suffix in ("", "_e32", "_e64"):
                with self.subTest(op=op + suffix, form="sgpr"):
                    got, _ = run_ps(
                        "ps_mm_sgpr",
                        ["s_mov_b32 s0, 0x80000005", "v_mov_b32 v1, 5", f"{op}{suffix} v0, s0, v1"],
                    )
                    self.assertEqual(got, ref(0x80000005, 5))
                with self.subTest(op=op + suffix, form="literal"):
                    got, _ = run_ps(
                        "ps_mm_lit",
                        ["v_mov_b32 v1, 0x7FFFFFFF", f"{op}{suffix} v0, 0x80000000, v1"],
                    )
                    self.assertEqual(got, ref(0x80000000, 0x7FFFFFFF))

    def test_seeded_random_pairs(self):
        rng = random.Random(0x4D494E4D)
        for _ in range(10):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            for op, ref in REF_MAP.items():
                with self.subTest(op=op, a=hex(a), b=hex(b)):
                    got, _ = run_ps(
                        "ps_mm_rand",
                        [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"{op} v0, v1, v2"],
                    )
                    self.assertEqual(got, ref(a, b))


class VAshrrevI32Tests(unittest.TestCase):
    def test_hand_constants_and_shift_masking(self):
        for sh, val, expected in ASHRREV_HAND_CONSTANTS:
            with self.subTest(sh=hex(sh), val=hex(val)):
                got, _ = run_ps(
                    "ps_ashr_const",
                    [f"v_mov_b32 v1, {sh:#x}", f"v_mov_b32 v2, {val:#x}", "v_ashrrev_i32 v0, v1, v2"],
                )
                self.assertEqual(got, expected)

    def test_operand_forms_sgpr_literal_e32_e64(self):
        for suffix in ("", "_e32", "_e64"):
            with self.subTest(suffix=suffix, form="sgpr"):
                got, _ = run_ps(
                    "ps_ashr_sgpr",
                    ["s_mov_b32 s0, 35", "v_mov_b32 v1, 0x80000000", f"v_ashrrev_i32{suffix} v0, s0, v1"],
                )
                self.assertEqual(got, ref_ashrrev_i32(35, 0x80000000))
            with self.subTest(suffix=suffix, form="literal"):
                got, _ = run_ps(
                    "ps_ashr_lit",
                    ["v_mov_b32 v1, 0xF0000000", f"v_ashrrev_i32{suffix} v0, 0x10000004, v1"],
                )
                self.assertEqual(got, ref_ashrrev_i32(0x10000004, 0xF0000000))

    def test_seeded_random_pairs(self):
        rng = random.Random(0x41534852)
        for _ in range(12):
            sh = rng.getrandbits(32)
            val = rng.getrandbits(32)
            with self.subTest(sh=hex(sh), val=hex(val)):
                got, _ = run_ps(
                    "ps_ashr_rand",
                    [f"v_mov_b32 v1, {sh:#x}", f"v_mov_b32 v2, {val:#x}", "v_ashrrev_i32 v0, v1, v2"],
                )
                self.assertEqual(got, ref_ashrrev_i32(sh, val))


class VNotB32Tests(unittest.TestCase):
    def test_boundary_patterns_and_double_not_involution(self):
        for val, expected in NOT_HAND_CONSTANTS:
            with self.subTest(val=hex(val)):
                got, _ = run_ps("ps_not_const", [f"v_mov_b32 v1, {val:#x}", "v_not_b32 v0, v1"])
                self.assertEqual(got, expected)
                got2, _ = run_ps(
                    "ps_not_inv",
                    [f"v_mov_b32 v1, {val:#x}", "v_not_b32 v2, v1", "v_not_b32 v0, v2"],
                )
                self.assertEqual(got2, val)

    def test_operand_forms_sgpr_literal_e32_e64(self):
        for suffix in ("", "_e32", "_e64"):
            with self.subTest(suffix=suffix, form="sgpr"):
                got, _ = run_ps("ps_not_sgpr", ["s_mov_b32 s0, 0x12345678", f"v_not_b32{suffix} v0, s0"])
                self.assertEqual(got, ref_not_b32(0x12345678))
            with self.subTest(suffix=suffix, form="literal"):
                got, _ = run_ps("ps_not_lit", [f"v_not_b32{suffix} v0, 0xDEADBEEF"])
                self.assertEqual(got, ref_not_b32(0xDEADBEEF))

    def test_seeded_random_values(self):
        rng = random.Random(0x4E4F5432)
        for _ in range(10):
            val = rng.getrandbits(32)
            with self.subTest(val=hex(val)):
                got, _ = run_ps("ps_not_rand", [f"v_mov_b32 v1, {val:#x}", "v_not_b32 v0, v1"])
                self.assertEqual(got, ref_not_b32(val))


class SpirvStructureTests(unittest.TestCase):
    def test_emitted_opcodes(self):
        for op, glsl_name in (
            ("v_min_i32", "SMin"),
            ("v_max_i32", "SMax"),
            ("v_min_u32", "UMin"),
            ("v_max_u32", "UMax"),
        ):
            with self.subTest(op=op):
                _, spv = run_ps("ps_struct_mm", ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", f"{op} v0, v1, v2"])
                self.assertIn(f"= OpExtInst %uint %glsl {glsl_name}", spv)

        _, spv_ashr = run_ps(
            "ps_struct_ashr", ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_ashrrev_i32 v0, v1, v2"]
        )
        self.assertIn("= OpShiftRightArithmetic %uint", spv_ashr)

        _, spv_not = run_ps("ps_struct_not", ["v_mov_b32 v1, 1", "v_not_b32 v0, v1"])
        self.assertIn("= OpNot %uint", spv_not)


if __name__ == "__main__":
    unittest.main()
