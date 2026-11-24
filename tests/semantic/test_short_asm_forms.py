"""Hand-written assembly forms that the assembler accepts and canonicalises, but that tests copy unchanged.

tests/semantic/regen_fixtures.py writes a hand-written .asm into a fixture as it is, without the llvm-mc pass that
the GPU runner applies, so the translator sees the source spelling. Two spellings used to end in an IndexError
instead of a result:
  * the short VOP2 carry form `v_add_co_ci_u32 vD, src0, src1` (llvm-mc prints it as
    `v_add_co_ci_u32_e32 vD, vcc_lo, src0, src1, vcc_lo`), also v_sub_co_ci_u32 and v_subrev_co_ci_u32;
  * a modifier list shorter than the operands, `op_sel_hi:[0,0]` or `neg_lo:[1,0]` on a three-source op
    (v_fma_mix_*, v_pk_fma_f16); llvm-mc pads the missing entries with 0, so `[0,0]` is `[0,0,0]`.
Each test checks that the short spelling gives the same words as the long one, and one value from the ISA.
Standard library only; no GPU, no game files.
"""
import unittest

import support

TAIL = ["exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]


def run(lines):
    text = "; short_form\n" + "".join("\t" + ln + "\n" for ln in lines + TAIL)
    return support.mrt0_words(support.translate_listing("ps_short_form", text.encode("utf-8")))


def hexes(words):
    return [f"{w:#010x}" for w in words]


class ShortCarryForms(unittest.TestCase):
    SETUP = ["v_mov_b32 v1, 0xfffffff0", "v_mov_b32 v2, 0x20", "v_cmp_eq_u32 vcc_lo, v1, v1"]      # VCC = 1 (carry in)

    def check(self, op, want):
        short = run(self.SETUP + [f"{op} v10, v1, v2", f"v_cndmask_b32 v11, 0, 1, vcc_lo", "v_mov_b32 v12, 0",
                                  "v_mov_b32 v13, 0"])
        explicit = run(self.SETUP + [f"{op}_e32 v10, vcc_lo, v1, v2, vcc_lo", f"v_cndmask_b32 v11, 0, 1, vcc_lo",
                                     "v_mov_b32 v12, 0", "v_mov_b32 v13, 0"])
        self.assertEqual(hexes(short), hexes(explicit))
        self.assertEqual(short[0], want[0])
        self.assertEqual(short[1], want[1])

    def test_add_co_ci(self):
        # 0xfffffff0 + 0x20 + 1 = 0x1_0000_0011: result 0x11, carry out 1
        self.check("v_add_co_ci_u32", (0x11, 1))

    def test_sub_co_ci(self):
        # 0xfffffff0 - 0x20 - 1 = 0xffffffcf, no borrow
        self.check("v_sub_co_ci_u32", (0xFFFFFFCF, 0))

    def test_subrev_co_ci(self):
        # 0x20 - 0xfffffff0 - 1 = 0x2f with a borrow
        self.check("v_subrev_co_ci_u32", (0x2F, 1))


class ShortModifierLists(unittest.TestCase):
    INPUT = ["v_mov_b32 v1, 0x40003c00", "v_mov_b32 v2, 0xc4004200", "v_mov_b32 v3, 0x3c004000"]    # (1,2) (3,-4) (2,1)

    def same(self, short, full):
        a = run(self.INPUT + [short, "v_mov_b32 v11, 0", "v_mov_b32 v12, 0", "v_mov_b32 v13, 0"])
        b = run(self.INPUT + [full, "v_mov_b32 v11, 0", "v_mov_b32 v12, 0", "v_mov_b32 v13, 0"])
        self.assertEqual(hexes(a), hexes(b))
        return a[0]

    def test_pk_fma_op_sel(self):
        # op_sel:[1,0] is op_sel:[1,0,0]: the low half takes src0's high half (2.0), src1's low (3.0), src2's low (2.0)
        # = 2*3 + 2 = 8.0 (0x4800); the high half takes src0 hi (2.0), src1 hi (-4.0), src2 hi (1.0) = -7.0 (0xc700)
        got = self.same("v_pk_fma_f16 v10, v1, v2, v3 op_sel:[1,0]", "v_pk_fma_f16 v10, v1, v2, v3 op_sel:[1,0,0]")
        self.assertEqual(got, 0xC7004800)

    def test_pk_fma_neg_lo(self):
        # neg_lo:[1,0] is neg_lo:[1,0,0]: lo = -(1.0) * 3.0 + 2.0 = -1.0 (0xbc00); hi = 2.0 * -4.0 + 1.0 = -7.0 (0xc700)
        got = self.same("v_pk_fma_f16 v10, v1, v2, v3 neg_lo:[1,0]", "v_pk_fma_f16 v10, v1, v2, v3 neg_lo:[1,0,0]")
        self.assertEqual(got, 0xC700BC00)

    def test_pk_fma_op_sel_hi(self):
        # op_sel_hi:[0,0] is [0,0,0]: the high half uses the low halves too: 1.0 * 3.0 + 2.0 = 5.0 (0x4500)
        got = self.same("v_pk_fma_f16 v10, v1, v2, v3 op_sel_hi:[0,0]", "v_pk_fma_f16 v10, v1, v2, v3 op_sel_hi:[0,0,0]")
        self.assertEqual(got >> 16, 0x4500)

    def test_fma_mix_op_sel_hi(self):
        # [0,0] is [0,0,0]: all three operands read as f32 (the registers' bit patterns), none as a half
        self.same("v_fma_mix_f32 v10, v1, v2, v3 op_sel_hi:[0,0]", "v_fma_mix_f32 v10, v1, v2, v3 op_sel_hi:[0,0,0]")


if __name__ == "__main__":
    unittest.main()
