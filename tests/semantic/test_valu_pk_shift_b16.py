"""Semantic interpreter tests for RDNA2 packed 16-bit integer shift ops (VOP3P).

Covers v_pk_lshlrev_b16, v_pk_lshrrev_b16 and v_pk_ashrrev_i16:
- 4-bit shift count masking (& 15) and full 0..15 range on both halves independently.
- Half-word isolation (no carry/borrow or shift bleed between low and high 16-bit lanes).
- Unsigned zero-fill vs signed arithmetic sign-extension in v_pk_ashrrev_i16.
- op_sel / op_sel_hi operand half selection, SGPR/inline/literal sources, destination aliasing.
- Rejection of unsupported modifiers (clamp, neg_lo/neg_hi, mul/div, src neg/abs, malformed op_sel).

Each test translates a hand-written pixel shader through tools/rdna2spirv.py and executes it in
tests/semantic/spirv_mini_interp.py. Standard library only; no GPU, no game files.
"""
import os
import random
import re
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

OPS = ("v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16")
WORDS = (
    0x00000000,
    0x00010002,
    0x7FFF8000,
    0x80007FFF,
    0x8001FFFF,
    0x8004FFF0,
    0x00FF0100,
    0xFFFFFFFF,
)


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def ref_pk_shift(op, sh_word, val_word, sel=(0, 0), selhi=(1, 1)):
    """Independent ISA model for v_pk_lshlrev_b16, v_pk_lshrrev_b16, v_pk_ashrrev_i16."""
    out = 0
    for h, sl in enumerate((sel, selhi)):
        amt = (sh_word >> (16 * sl[0])) & 0xF
        val = (val_word >> (16 * sl[1])) & 0xFFFF
        if op == "v_pk_lshlrev_b16":
            r = (val << amt) & 0xFFFF
        elif op == "v_pk_lshrrev_b16":
            r = val >> amt
        elif op == "v_pk_ashrrev_i16":
            r = (s16(val) >> amt) & 0xFFFF
        else:
            raise ValueError(f"unexpected op {op}")
        out |= (r & 0xFFFF) << (16 * h)
    return out


def mod_str(name, bits):
    return f" {name}:[{','.join(map(str, bits))}]"


def translate_shift(lines):
    return support.translate_lines("ps_pk_shift_b16", lines)


def run_shift(line, push=None):
    prog = [
        "v_mov_b32 v1, s0",
        "v_mov_b32 v2, s1",
        "v_mov_b32 v0, 0xdeadbeef",
        line,
        "v_mov_b32 v3, 0",
        "v_mov_b32 v4, 0",
        "v_mov_b32 v5, 0",
        "exp mrt0 v0, v3, v4, v5 done vm",
        "s_endpgm",
    ]
    spv = translate_shift(prog)
    return support.mrt0_words(spv, push_constants=push)[0]


class PkShiftB16Tests(unittest.TestCase):
    def test_reference_sanity(self):
        self.assertEqual(ref_pk_shift("v_pk_lshlrev_b16", 0x00130002, 0x8004FFF0), 0x0020FFC0)
        self.assertEqual(ref_pk_shift("v_pk_lshrrev_b16", 0x00130002, 0x8004FFF0), 0x10003FFC)
        self.assertEqual(ref_pk_shift("v_pk_ashrrev_i16", 0x00130002, 0x8004FFF0), 0xF000FFFC)
        self.assertEqual(
            ref_pk_shift("v_pk_ashrrev_i16", 0x00130002, 0x8004FFF0, sel=(1, 0), selhi=(0, 1)),
            0xE001FFFE,
        )

    def test_shift_amounts_and_4bit_masking(self):
        val = 0x80017FFF
        for amt_lo, amt_hi in ((0, 0), (1, 15), (4, 8), (15, 1), (16, 17), (31, 0xFFF3), (0xFFFF, 0x0020)):
            sh = (amt_lo & 0xFFFF) | ((amt_hi & 0xFFFF) << 16)
            for op in OPS:
                with self.subTest(op=op, amt_lo=amt_lo, amt_hi=amt_hi):
                    got = run_shift(f"{op} v0, v1, v2", push=[sh, val])
                    self.assertEqual(got, ref_pk_shift(op, sh, val))

    def test_signed_vs_unsigned_and_half_isolation(self):
        shifts = (0x00010001, 0x0004000F, 0x000F0000, 0x00170013)
        for op in OPS:
            for sh in shifts:
                for val in WORDS:
                    with self.subTest(op=op, sh=hex(sh), val=hex(val)):
                        got = run_shift(f"{op} v0, v1, v2", push=[sh, val])
                        self.assertEqual(got, ref_pk_shift(op, sh, val))

    def test_op_sel_and_op_sel_hi_permutations(self):
        sh = 0x00070003
        val = 0x81234567
        combos = (
            (None, None, ""),
            ([1, 0], [0, 1], ""),
            ([0, 1], [1, 0], ""),
            ([0, 0], [0, 0], ""),
            ([1, 1], [1, 1], ""),
            ([1, 1], None, ""),
            (None, [0, 0], ""),
            ([1, 0], [0, 1], " neg_lo:[0,0] neg_hi:[0,0]"),
        )
        for op in OPS:
            for sel, selhi, extra in combos:
                line = f"{op} v0, v1, v2"
                if sel is not None:
                    line += mod_str("op_sel", sel)
                if selhi is not None:
                    line += mod_str("op_sel_hi", selhi)
                line += extra
                with self.subTest(op=op, line=line):
                    got = run_shift(line, push=[sh, val])
                    want = ref_pk_shift(op, sh, val, sel or (0, 0), selhi or (1, 1))
                    self.assertEqual(got, want)

    def test_constants_sgpr_and_aliasing(self):
        # Inline positive constants zero-extend; negative inline constants sign-extend to 32-bit dwords.
        cases = (
            ("v_pk_lshlrev_b16 v0, 4, v2", None, [0, 0x12345678], ref_pk_shift("v_pk_lshlrev_b16", 4, 0x12345678)),
            ("v_pk_lshlrev_b16 v0, 4, v2 op_sel_hi:[0,1]", None, [0, 0x12345678],
             ref_pk_shift("v_pk_lshlrev_b16", 4, 0x12345678, selhi=(0, 1))),
            ("v_pk_ashrrev_i16 v0, -1, v2", None, [0, 0x80040008], ref_pk_shift("v_pk_ashrrev_i16", 0xFFFFFFFF, 0x80040008)),
            ("v_pk_lshrrev_b16 v0, -16, v2", None, [0, 0x8004FFF0], ref_pk_shift("v_pk_lshrrev_b16", 0xFFFFFFF0, 0x8004FFF0)),
            ("v_pk_lshlrev_b16 v0, v1, -16", None, [0x00020001, 0], ref_pk_shift("v_pk_lshlrev_b16", 0x00020001, 0xFFFFFFF0)),
            ("v_pk_ashrrev_i16 v0, 0x00040001, v2", None, [0, 0x8000FFF0],
             ref_pk_shift("v_pk_ashrrev_i16", 0x00040001, 0x8000FFF0)),
            ("v_pk_lshrrev_b16 v0, s0, v2", None, [0x00030005, 0x80F01234],
             ref_pk_shift("v_pk_lshrrev_b16", 0x00030005, 0x80F01234)),
            ("v_pk_ashrrev_i16 v0, v1, s1", None, [0x00030005, 0x80F01234],
             ref_pk_shift("v_pk_ashrrev_i16", 0x00030005, 0x80F01234)),
        )
        for line, _, push, want in cases:
            with self.subTest(line=line):
                self.assertEqual(run_shift(line, push=push), want)

        # Destination aliasing src0 or src1 must still read the original operands.
        for op in OPS:
            sh, val = 0x00050002, 0x8123F00F
            prog_alias0 = [
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
                f"{op} v0, v0, v1",
                "v_mov_b32 v2, 0",
                "exp mrt0 v0, v2, v2, v2 done vm",
                "s_endpgm",
            ]
            prog_alias1 = [
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
                f"{op} v1, v0, v1",
                "v_mov_b32 v2, 0",
                "exp mrt0 v1, v2, v2, v2 done vm",
                "s_endpgm",
            ]
            with self.subTest(op=op, alias="dst==src0"):
                self.assertEqual(
                    support.mrt0_words(translate_shift(prog_alias0), push_constants=[sh, val])[0],
                    ref_pk_shift(op, sh, val),
                )
            with self.subTest(op=op, alias="dst==src1"):
                self.assertEqual(
                    support.mrt0_words(translate_shift(prog_alias1), push_constants=[sh, val])[0],
                    ref_pk_shift(op, sh, val),
                )

    def test_seeded_random_cases(self):
        rng = random.Random(0x504B5348)
        for op in OPS:
            for i in range(24):
                sh = rng.getrandbits(32)
                val = rng.getrandbits(32)
                sel = [rng.randint(0, 1), rng.randint(0, 1)]
                selhi = [rng.randint(0, 1), rng.randint(0, 1)]
                line = f"{op} v0, v1, v2" + mod_str("op_sel", sel) + mod_str("op_sel_hi", selhi)
                with self.subTest(op=op, case=i):
                    got = run_shift(line, push=[sh, val])
                    self.assertEqual(got, ref_pk_shift(op, sh, val, tuple(sel), tuple(selhi)))

    def test_rejected_modifiers(self):
        bad_lines = (
            "v_pk_lshlrev_b16 v0, v1, v2 clamp",
            "v_pk_lshrrev_b16 v0, v1, v2 clamp",
            "v_pk_ashrrev_i16 v0, v1, v2 clamp",
            "v_pk_lshlrev_b16 v0, v1, v2 neg_lo:[1,0]",
            "v_pk_lshrrev_b16 v0, v1, v2 neg_lo:[0,1]",
            "v_pk_ashrrev_i16 v0, v1, v2 neg_hi:[1,0]",
            "v_pk_lshlrev_b16 v0, v1, v2 neg_hi:[0,1]",
            "v_pk_lshlrev_b16 v0, v1, v2 mul:2",
            "v_pk_lshrrev_b16 v0, v1, v2 div:2",
            "v_pk_lshlrev_b16 v0, v1, -v2",
            "v_pk_ashrrev_i16 v0, |v1|, v2",
            # a three-entry op_sel is llvm-mc gfx1013 syntax (runtime/test/shaders/ps_ops6.asm) and is accepted
            "v_pk_lshlrev_b16 v0, v1, v2 op_sel:[0,0,0,0]",
            "v_pk_lshrrev_b16 v0, v1, v2 op_sel_hi:[1]",
            "v_pk_ashrrev_i16 v0, v1, v2 op_sel:[0,2]",
            "v_pk_lshlrev_b16 v0, v1, v2 op_sel:bad",
        )
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_shift([
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        line,
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])

    def test_negative_controls(self):
        # 1. Sabotaging signed 16-bit extraction to unsigned must lose sign-extension on negative halves.
        spv_ashr = translate_shift([
            "v_mov_b32 v1, s0",
            "v_mov_b32 v2, s1",
            "v_pk_ashrrev_i16 v0, v1, v2",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v3, v3, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpBitFieldSExtract", spv_ashr)
        sabotaged_ashr = spv_ashr.replace("OpBitFieldSExtract", "OpBitFieldUExtract")
        push = [0x00020002, 0x8004FFF0]
        self.assertEqual(support.mrt0_words(spv_ashr, push_constants=push)[0], 0xE001FFFC)
        self.assertNotEqual(support.mrt0_words(sabotaged_ashr, push_constants=push)[0], 0xE001FFFC)

        # 2. Sabotaging the 4-bit shift count mask to 5 bits must fail when shift count == 16.
        spv_lshl = translate_shift([
            "v_mov_b32 v1, s0",
            "v_mov_b32 v2, s1",
            "v_pk_lshlrev_b16 v0, v1, v2",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v3, v3, v3 done vm",
            "s_endpgm",
        ])
        m = re.search(r"=\s*OpBitFieldUExtract\s+%uint\s+%\S+\s+%\S+\s+(%\S+)\n", spv_lshl)
        self.assertIsNotNone(m)
        const4 = m.group(1)
        m5 = re.search(r"^(%\S+)\s*=\s*OpConstant\s+%uint\s+5\s*$", spv_lshl, re.M)
        if m5:
            const5 = m5.group(1)
            sabotaged_mask = re.sub(
                r"(OpBitFieldUExtract\s+%uint\s+%\S+\s+%\S+\s+)" + re.escape(const4) + r"\b",
                r"\1" + const5,
                spv_lshl,
            )
        else:
            sabotaged_mask = re.sub(
                r"^" + re.escape(const4) + r"\s*=\s*OpConstant\s+%uint\s+4\s*$",
                f"{const4} = OpConstant %uint 5",
                spv_lshl,
                flags=re.M,
            )
        self.assertNotEqual(sabotaged_mask, spv_lshl)
        push16 = [0x00100010, 0x12345678]
        self.assertEqual(support.mrt0_words(spv_lshl, push_constants=push16)[0], 0x12345678)
        self.assertNotEqual(support.mrt0_words(sabotaged_mask, push_constants=push16)[0], 0x12345678)


if __name__ == "__main__":
    unittest.main()
