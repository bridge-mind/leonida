"""s_add_u32, s_addc_u32, s_sub_u32, s_subb_u32 (the scalar add/subtract with carry) through the translator.

RDNA2 ISA, SOP2:
  s_add_u32   D = S0 + S1 (mod 2^32),             SCC = carry out of bit 31
  s_addc_u32  D = S0 + S1 + SCC (mod 2^32),       SCC = carry out
  s_sub_u32   D = S0 - S1 (mod 2^32),             SCC = 1 if S1 > S0 (borrow)
  s_subb_u32  D = S0 - S1 - SCC (mod 2^32),       SCC = 1 if S1 + SCC > S0 (borrow)
The operands are SGPRs, 32-bit literals or inline constants (-16..64); the carry-in is made with s_cmp_eq_u32.

Each case translates a hand-written pixel shader with tools/rdna2spirv.py (support.translate_lines), runs it
(support.mrt0_words) and compares the destination and SCC with a plain-Python reference and with hand-written words.
test_carry_chain.py already runs the 64-bit add/subtract chains and the vector forms; this file covers the operand
forms and the SCC in/out of the scalar ones. Until this file, the test of this name ran a self-built interpreter and
never called the translator.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

M32 = 0xFFFFFFFF


def ref(op, s0, s1, scc):
    if op == "s_add_u32":
        t = s0 + s1
    elif op == "s_addc_u32":
        t = s0 + s1 + scc
    elif op == "s_sub_u32":
        t = s0 - s1
    else:
        t = s0 - s1 - scc
    return t & M32, int(t >= 1 << 32) if "add" in op else int(t < 0)


def operand(kind, reg, v):
    """(setup lines, operand text) giving S0/S1 the value v."""
    if kind == "sgpr":
        return [f"s_mov_b32 s{reg}, {v:#x}"], f"s{reg}"
    return [], f"{v:#x}"                                  # a literal (or inline constant when it is one)


def run(op, s0, s1, scc, k0="sgpr", k1="sgpr"):
    a, t0 = operand(k0, 10, s0)
    b, t1 = operand(k1, 11, s1)
    lines = a + b + ["s_mov_b32 s12, 0", "s_cmp_eq_u32 s12, %d" % (0 if scc else 1),     # SCC = scc
                     f"{op} s13, {t0}, {t1}", "s_cselect_b32 s14, 1, 0", "v_mov_b32_e32 v0, s13", "v_mov_b32_e32 v1, s14",
                     "exp mrt0 v0, v1, v0, v0 done vm", "s_endpgm"]
    return tuple(support.mrt0_words(support.translate_lines("ps_salu_add_sub_co", lines))[:2])


OPS = ("s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32")
EDGE = (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)


class TestByHand(unittest.TestCase):
    def test_words(self):
        # (op, S0, S1, SCC in) -> (D, SCC out), written out from the formulas above.
        for case, want in (
            (("s_add_u32", 0xFFFFFFFF, 1, 0), (0, 1)), (("s_add_u32", 0x7FFFFFFF, 1, 1), (0x80000000, 0)),
            (("s_addc_u32", 0xFFFFFFFF, 0, 1), (0, 1)), (("s_addc_u32", 0xFFFFFFFF, 0, 0), (0xFFFFFFFF, 0)),
            (("s_addc_u32", 5, 6, 1), (12, 0)), (("s_addc_u32", 0xFFFFFFFF, 0xFFFFFFFF, 1), (0xFFFFFFFF, 1)),
            (("s_sub_u32", 0, 1, 0), (0xFFFFFFFF, 1)), (("s_sub_u32", 5, 5, 1), (0, 0)),
            (("s_subb_u32", 5, 5, 1), (0xFFFFFFFF, 1)), (("s_subb_u32", 5, 5, 0), (0, 0)),
            (("s_subb_u32", 0, 0xFFFFFFFF, 1), (0, 1)), (("s_subb_u32", 0x80000000, 1, 1), (0x7FFFFFFE, 0)),
        ):
            with self.subTest(case=case):
                self.assertEqual(ref(*case), want)
                self.assertEqual(run(*case), want)


class TestForms(unittest.TestCase):
    def test_operand_forms(self):
        for op in OPS:
            for k0, k1 in (("sgpr", "sgpr"), ("literal", "sgpr"), ("sgpr", "literal")):
                for s0, s1, scc in ((0xFFFFFFFF, 1, 1), (0x80000000, 0x80000000, 0), (7, 0x12345678, 1), (3, 3, 0)):
                    with self.subTest(op=op, forms=(k0, k1), s0=hex(s0), s1=hex(s1), scc=scc):
                        self.assertEqual(run(op, s0, s1, scc, k0, k1), ref(op, s0, s1, scc))

    def test_inline_constants(self):
        for op in OPS:
            for s0, s1 in ((64, 1), (0, 16), (63, 64), (1, 0)):
                with self.subTest(op=op, s0=s0, s1=s1):
                    self.assertEqual(run(op, s0, s1, 1, "literal", "literal"), ref(op, s0, s1, 1))

    def test_seeded(self):
        rng = random.Random(0x5CC)
        for _ in range(30):
            op, scc = rng.choice(OPS), rng.getrandbits(1)
            s0, s1 = rng.choice((rng.getrandbits(32),) + EDGE), rng.choice((rng.getrandbits(32),) + EDGE)
            with self.subTest(op=op, s0=hex(s0), s1=hex(s1), scc=scc):
                self.assertEqual(run(op, s0, s1, scc), ref(op, s0, s1, scc))


if __name__ == "__main__":
    unittest.main()
