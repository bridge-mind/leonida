"""v_dot4_i32_i8 and v_dot4c_i32_i8: the dot product of the signed bytes of two dwords, plus an addend.

ISA rule (RDNA2): D = S0.b0*S1.b0 + S0.b1*S1.b1 + S0.b2*S1.b2 + S0.b3*S1.b3 + addend, every byte sign-extended, the
sum kept modulo 2^32. The addend is the third source for v_dot4_i32_i8 (VOP3P) and the old value of the destination
for v_dot4c_i32_i8 (VOP2). The 29 FSPX programs of the package's upscaler DLL (docs/fspx.md) use the pair 27,271
times.

The expected words below were measured on RDNA2 hardware (gfx1036, the integrated GPU of a Ryzen 9 7950X3D, through
tools/hw_oracle): each case ran `v_dot4_i32_i8 v8, v1, v2, v3`, `v_dot4c_i32_i8 v10, v1, v2` with v10 holding the
addend, and the same with the two sources swapped. All three gave the word in the table, which the plain-Python
reference `ref` reproduces. With `clamp` the hardware saturates v_dot4_i32_i8 to a signed 32-bit value
(0x7fffffff and 0x80000000 in the two overflowing cases); the translator refuses that form.

op_sel. The plain instruction has op_sel 0 and op_sel_hi 7. 89 words in 8 of those programs are v_dot4_i32_i8 with
op_sel 2 and op_sel_hi 5, which llvm-mc does not decode (docs/fspx.md). Run on the same hardware as two raw words
(0xCC165008 0x0C0E0501: v8 = v1 . v2 + v3 with those fields), that form gave the words of MEASURED_OP_SEL: the dot
product with the two 16-bit halves of the second source exchanged, as op_sel selects halves for the packed 16-bit
ops. `ref_op_sel` reproduces them; the translator refuses any op_sel on these two instructions until a listing can
carry it.

neg_lo and neg_hi. llvm-mc accepts them on v_dot4_i32_i8 for gfx1036 (not on v_dot4c_i32_i8), and the hardware does
not ignore them: on the same GPU, 5 operand sets with 8 forms each gave the words of `ref_neg`, 9 of which are in
MEASURED_NEG. For S0 and S1, neg_lo flips bit 15 and neg_hi flips bit 31 of the dword; for the addend, neg_hi clears
bit 31 and neg_lo then flips it. That is what the fields do to float operands, applied to integer ones. The
translator refuses a set bit and takes an all-zero field as the plain instruction.

    python -m unittest discover -s tests/semantic -p test_alu_dot4_i8.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

rdna2spirv = support.rdna2spirv

# (S0, S1, addend, result measured on gfx1036)
MEASURED = [
    (0x01020304, 0x05060708, 0x00000000, 0x00000046),   # 4*8 + 3*7 + 2*6 + 1*5 = 70
    (0x01020304, 0x05060708, 0x000003E8, 0x0000042E),   # 70 + 1000
    (0xFF80017F, 0x7F7F80FF, 0x00000000, 0xFFFFBF02),   # 127*-1 + 1*-128 + -128*127 + -1*127 = -16638
    (0xFF80017F, 0x7F7F80FF, 0x00010000, 0x0000BF02),
    (0x80808080, 0x80808080, 0x00000000, 0x00010000),   # 4 * (-128 * -128)
    (0x80808080, 0x7F7F7F7F, 0x00000005, 0xFFFF0205),   # 4 * (-128 * 127) + 5
    (0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF, 0x8000FC03),   # wraps upward: no saturation without clamp
    (0x80808080, 0x7F7F7F7F, 0x80000000, 0x7FFF0200),   # wraps downward
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000003),   # 4 * (-1 * -1) - 1
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
]


# (S0, S1, addend, result measured on gfx1036 with op_sel 2 and op_sel_hi 5)
MEASURED_OP_SEL = [
    (0x04030201, 0x08070605, 0x00000000, 62),           # the plain form gives 70
    (0x04030201, 0x08070605, 0x000003E8, 1062),
    (0x7F01FE80, 0x02FD7F81, 0x00000000, 16382),        # the plain form gives 16253
    (0x11223344, 0x55667788, 0x00010000, 74750),        # the plain form gives 68358
    (0xFFFFFFFF, 0x01020304, 0x00000007, 0xFFFFFFFD),   # every byte of S0 the same: the order cannot show
]


# (neg_lo, neg_hi, S0, S1, addend, result measured on gfx1036), one operand set under every form that was run
MEASURED_NEG = [
    ((0, 0, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0x000000AA),   # plain: 70 + 100
    ((1, 0, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFDAA),   # S0 byte 1 reads as 0x82
    ((0, 0, 0), (1, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFCAA),   # S0 byte 3 reads as 0x84
    ((0, 1, 0), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFFAA),   # S1 byte 1 reads as 0x86
    ((0, 0, 0), (0, 1, 0), 0x04030201, 0x08070605, 0x00000064, 0xFFFFFEAA),   # S1 byte 3 reads as 0x88
    ((0, 0, 1), (0, 0, 0), 0x04030201, 0x08070605, 0x00000064, 0x800000AA),   # the addend's bit 31 flipped
    ((0, 0, 0), (0, 0, 1), 0x04030201, 0x08070605, 0x00000064, 0x000000AA),   # abs of a positive addend
    ((0, 0, 1), (0, 0, 1), 0x04030201, 0x08070605, 0x00000064, 0x800000AA),   # abs, then negated
    ((1, 1, 1), (1, 1, 1), 0x04030201, 0x08070605, 0x00000064, 0x800076AA),
]


def ref(a, b, addend):
    def s8(x):
        return x - 256 if x & 0x80 else x
    total = sum(s8((a >> (8 * i)) & 0xFF) * s8((b >> (8 * i)) & 0xFF) for i in range(4))
    return (total + addend) & 0xFFFFFFFF


def ref_neg(neg_lo, neg_hi, a, b, addend):
    """neg_lo[k] flips bit 15 and neg_hi[k] bit 31 of source k; the addend loses bit 31 under neg_hi[2] and has it
    flipped under neg_lo[2], in that order."""
    a ^= (0x8000 if neg_lo[0] else 0) ^ (0x80000000 if neg_hi[0] else 0)
    b ^= (0x8000 if neg_lo[1] else 0) ^ (0x80000000 if neg_hi[1] else 0)
    if neg_hi[2]:
        addend &= 0x7FFFFFFF
    if neg_lo[2]:
        addend ^= 0x80000000
    return ref(a, b, addend)


def ref_op_sel(a, b, addend):
    """op_sel 2, op_sel_hi 5: the low half of the product reads the high half of S1 and the high half its low half."""
    return ref(a, (b & 0xFFFF) << 16 | b >> 16, addend)


def program(a, b, addend):
    """mrt0 = (v_dot4_i32_i8, v_dot4c_i32_i8, v_dot4c_i32_i8 with the sources swapped, v_dot4_i32_i8 with the
    sources and the addend taken from SGPRs and a literal, as the shaders do)."""
    return [
        f"v_mov_b32_e32 v1, 0x{a:08x}",
        f"v_mov_b32_e32 v2, 0x{b:08x}",
        f"v_mov_b32_e32 v3, 0x{addend:08x}",
        f"s_mov_b32 s4, 0x{addend:08x}",
        "v_dot4_i32_i8 v8, v1, v2, v3",
        f"v_mov_b32_e32 v9, 0x{addend:08x}",
        "v_dot4c_i32_i8 v9, v1, v2",
        f"v_mov_b32_e32 v10, 0x{addend:08x}",
        "v_dot4c_i32_i8 v10, v2, v1",
        f"v_dot4_i32_i8 v11, v1, 0x{b:08x}, s4",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]


class Dot4I8(unittest.TestCase):
    def test_the_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            self.assertEqual(ref(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")

    def test_both_forms_give_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                spv = support.translate_lines("ps_dot4_i8", program(a, b, addend))
                self.assertEqual(support.mrt0_words(spv), [measured] * 4)

    def test_the_accumulating_form_reads_its_destination(self):
        # v9 is never written before v_dot4c reads it: the old value is then the register's initial 0, not garbage
        # from an expression that dropped the addend. 70 is the dot product alone.
        lines = ["v_mov_b32_e32 v1, 0x01020304", "v_mov_b32_e32 v2, 0x05060708", "v_dot4c_i32_i8 v9, v1, v2",
                 "v_dot4c_i32_i8 v9, v1, v2", "exp mrt0 v9, v9, v9, v9 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4c_twice", lines)), [140] * 4)

    def test_the_bytes_are_signed(self):
        # 0xff * 0x01: -1 as a signed byte, 255 as an unsigned one.
        lines = ["v_mov_b32_e32 v1, 0x000000ff", "v_mov_b32_e32 v2, 0x00000001", "v_mov_b32_e32 v3, 0",
                 "v_dot4_i32_i8 v8, v1, v2, v3", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4_signed", lines)), [0xFFFFFFFF] * 4)

    def test_the_op_sel_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED_OP_SEL:
            self.assertEqual(ref_op_sel(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")
        # The table can tell the two readings apart: the plain rule gives another word in four of the five cases.
        self.assertEqual(sum(ref(a, b, c) != m for a, b, c, m in MEASURED_OP_SEL), 4)

    def test_op_sel_matches_hardware_words(self):
        for a, b, addend, measured in MEASURED_OP_SEL:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                    "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_dot4_op_sel", lines)
                self.assertEqual(support.mrt0_words(spv), [measured] * 4)

    def test_the_neg_reference_reproduces_the_hardware_words(self):
        for neg_lo, neg_hi, a, b, addend, measured in MEASURED_NEG:
            self.assertEqual(ref_neg(neg_lo, neg_hi, a, b, addend), measured, f"neg_lo {neg_lo} neg_hi {neg_hi}")
        # The fields are not ignored: seven of the eight forms give another word than the plain instruction (the
        # eighth takes the absolute value of an addend that is positive already).
        plain = MEASURED_NEG[0][5]
        self.assertEqual(sum(m != plain for *_, m in MEASURED_NEG[1:]), 7)

    def test_a_set_neg_bit_is_refused(self):
        for field in ("neg_lo:[1,0,0]", "neg_hi:[0,1,0]", "neg_lo:[0,0,1] neg_hi:[0,0,1]"):
            with self.subTest(field=field):
                lines = ["v_mov_b32_e32 v1, 1", "v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v3, 0",
                         f"v_dot4_i32_i8 v8, v1, v2, v3 {field}", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_dot4_neg", lines)
                self.assertEqual(str(refused.exception), "v_dot4_i32_i8 neg")

    def test_an_all_zero_neg_field_is_the_plain_instruction(self):
        lines = ["v_mov_b32_e32 v1, 0x04030201", "v_mov_b32_e32 v2, 0x08070605", "v_mov_b32_e32 v3, 0x64",
                 "v_dot4_i32_i8 v8, v1, v2, v3 neg_lo:[0,0,0] neg_hi:[0,0,0]",
                 "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4_neg_zero", lines)), [0xAA] * 4)

    def test_op_sel_source0_swap_and_broadcast(self):
        # Swapping S0's halves instead of S1's gives the same result when sources are swapped.
        a, b, addend, measured = MEASURED_OP_SEL[0]
        lines = [
            f"v_mov_b32_e32 v1, 0x{b:08x}",
            f"v_mov_b32_e32 v2, 0x{a:08x}",
            f"v_mov_b32_e32 v3, 0x{addend:08x}",
            "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[0,1,1]",
            # Broadcast low half of both S0 and S1 into both halves: 2 * (b0*s1_b0 + b1*s1_b1)
            "v_dot4_i32_i8 v9, v2, v1, v3 op_sel:[0,0,0] op_sel_hi:[0,0,1]",
            "exp mrt0 v8, v9, v8, v9 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_dot4_op_sel_s0", lines)
        # a=0x04030201 (lo=0x0201: bytes 1,2), b=0x08070605 (lo=0x0605: bytes 5,6) -> 2*(1*5 + 2*6) = 34
        self.assertEqual(support.mrt0_words(spv), [measured, 34, measured, 34])

    def test_clamp_matches_hardware_saturation(self):
        # The two overflowing hardware cases (gfx1036) saturate to 0x7FFFFFFF / 0x80000000; non-overflowing cases stay exact.
        cases = [
            (0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF, 0x7FFFFFFF),   # positive overflow -> INT32_MAX
            (0x80808080, 0x7F7F7F7F, 0x80000000, 0x80000000),   # negative overflow -> INT32_MIN
            (0x01020304, 0x05060708, 0x000003E8, 0x0000042E),   # positive non-overflow
            (0xFF80017F, 0x7F7F80FF, 0x00000000, 0xFFFFBF02),   # negative non-overflow
            (0x7F7F7F7F, 0x7F7F7F7F, 0x80000000, (0x80000000 + 4 * 127 * 127) & 0xFFFFFFFF),  # opposite signs, no ov
        ]
        for a, b, addend, expected in cases:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                lines = [
                    f"v_mov_b32_e32 v1, 0x{a:08x}",
                    f"v_mov_b32_e32 v2, 0x{b:08x}",
                    f"v_mov_b32_e32 v3, 0x{addend:08x}",
                    "v_dot4_i32_i8 v8, v1, v2, v3 clamp",
                    "exp mrt0 v8, v8, v8, v8 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines("ps_dot4_clamp", lines)
                self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_unsupported_modifiers_refused(self):
        # Non-default S2 half selection on v_dot4_i32_i8 and clamp/op_sel on VOP2 v_dot4c_i32_i8 stay refused.
        for name, inst, err in (
            ("ps_dot4_s2_opsel", "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,0,1]", "v_dot4_i32_i8 op_sel"),
            ("ps_dot4c_clamp", "v_dot4c_i32_i8 v8, v1, v2 clamp", "v_dot4c_i32_i8 clamp"),
        ):
            lines = ["v_mov_b32_e32 v1, 1", "v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v3, 0",
                     inst, "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
            with self.assertRaises(rdna2spirv.Unsupported) as refused:
                support.translate_lines(name, lines)
            self.assertEqual(str(refused.exception), err)


if __name__ == "__main__":
    unittest.main()
