"""A v_fma_mix* literal that llvm-mc prints cut to its low half is read from the blob.

A v_fma_mix source read as f16 (op_sel_hi set) takes the half op_sel names, so one 32-bit literal can give src0 its
low half and src2 its high half. llvm-mc prints that literal as its low 16 bits and re-encodes it with a zero high
half, so the printed-encoding check (#2026) refused such an instruction, and LLVM's text alone reads the high half as
+0.0. The translator now takes the literal from the blob when only its high half differs. Hand-built encodings
(registers and values made up for the test), no game data.

    python -I tests/semantic/test_fma_mix_literal.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_listing  # noqa: E402

NAME = "ps_fma_mix_literal"
CODE_START = 0x100
# v_mov_b32 v2, 2.0 (VOP1); v_fma_mix_f32 v1, lit, v2, lit with op_sel:[0,0,1] op_sel_hi:[1,0,1]: src0 = the
# literal's low half as f16, src1 = v2 as f32, src2 = the literal's high half as f16. The stored literal is
# 0x40004200 (low 3.0, high 2.0); llvm-mc prints 0x4200 and encodes 0x00004200. The text and the printed encodings
# here are llvm-mc 23.1.5's for these bytes (a low half that is an inline f16 constant, such as 0x3c00, prints as
# lit(0x...) instead, so it is avoided).
MOV = ("v_mov_b32_e32 v2, 2.0", "f4 02 04 7e")
MIX_TEXT = "v_fma_mix_f32 v1, 0x4200, v2, 0x4200 op_sel:[0,0,1] op_sel_hi:[1,0,1]"
MIX_WORDS = "01 60 20 cc ff 04 fe 0b"
EXP = ("exp mrt0 v1, v1, v1, v1 done vm", "0f 18 00 f8 01 01 01 01")
END = ("s_endpgm", "00 00 81 bf")


def lit(value):
    return struct.pack("<I", value).hex(" ")


def program(literal, words=MIX_WORDS, text=MIX_TEXT):
    return [MOV, (text, f"{words} {lit(literal)}"), EXP, END]


def listing(prog, invalid=False):
    size = sum(len(bytes.fromhex(enc)) for _, enc in prog)
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, "
             f"{1 if invalid else 0} invalid words"]
    for text, enc in prog:
        printed = ",".join(f"0x{b:02x}" for b in bytes.fromhex(enc))
        lines.append(f"\t{text:<40}; encoding: [{printed}]")
    if invalid:
        # translate() takes the re-anchoring path when the listing reports an undecodable word.
        lines += ["", "; --- llvm-mc messages ---", "; <stdin>:1:1: warning: invalid instruction encoding"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def blob(prog):
    return bytes(CODE_START) + b"".join(bytes.fromhex(enc) for _, enc in prog)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


PRINTED = program(0x00004200)
STORED = program(0x40004200)


class FmaMixLiteralTest(unittest.TestCase):
    def run_ps(self, data, fssm):
        return mrt0_words(translate_listing(NAME, data, fssm=fssm))

    def test_high_half_comes_from_the_blob(self):
        # fma(3.0, 2.0, 2.0) = 8.0
        self.assertEqual(self.run_ps(listing(PRINTED), blob(STORED)), [f2u(8.0)] * 4)

    def test_reanchored_listing_takes_it_too(self):
        self.assertEqual(self.run_ps(listing(PRINTED, invalid=True), blob(STORED)), [f2u(8.0)] * 4)

    def test_without_the_blob_the_printed_literal_is_all_there_is(self):
        # LLVM's text alone: the high half reads +0.0, fma(3.0, 2.0, 0.0) = 6.0 (what the listing says).
        self.assertEqual(self.run_ps(listing(PRINTED), None), [f2u(6.0)] * 4)

    def test_printed_low_half_must_match(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.run_ps(listing(PRINTED), blob(program(0x40004201)))
        self.assertIn("v_fma_mix_f32 at 0x104", str(cm.exception))

    def test_any_other_difference_is_still_refused(self):
        # The stored instruction reads v3 as src1 (SRC1 field 0x103), not the printed v2.
        other = program(0x40004200, words="01 60 20 cc ff 06 fe 0b")
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.run_ps(listing(PRINTED), blob(other))
        self.assertIn("dropped", str(cm.exception))

    def test_a_live_source_printed_as_inline_zero_is_refused(self):
        # Stored SRC1 0x000 (s0, a source v_fma_mix reads) against a printed inline 0 (0x080). same_encoding forgives
        # that for an unused source field; the repair must not, since all three v_fma_mix sources are live.
        text = "v_fma_mix_f32 v1, 0x4200, 0, 0x4200 op_sel:[0,0,1] op_sel_hi:[1,0,1]"
        printed = program(0x00004200, words="01 60 20 cc ff 00 fd 0b", text=text)
        stored = program(0x40004200, words="01 60 20 cc ff 00 fc 0b", text=text)
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.run_ps(listing(printed), blob(stored))
        self.assertIn("v_fma_mix_f32 at 0x104", str(cm.exception))

    def test_an_inline_constant_with_the_printed_value_keeps_it(self):
        # src0 = v2 (2.0) as f32, src1 = the inline integer 1 as f32, src2 = the literal's high half (op_sel:[0,0,1]
        # op_sel_hi:[0,0,1]). The stored literal 0x3c000001 prints as 0x1, the same value as src1, so only the
        # SRC field (0xff) tells the literal's user apart. Hand-written text: llvm-mc 23.1.5 prints this literal as
        # lit(0x3c000001). fma(2.0, 1.4e-45, 1.0) = 1.0, also with the denormal flushed; src1 overwritten with the
        # literal would give 2 * 0.0078125 + 1.0.
        text = "v_fma_mix_f32 v1, v2, 1, 0x1 op_sel:[0,0,1] op_sel_hi:[0,0,1]"
        words = "01 60 20 cc 02 03 fd 03"
        printed = program(0x00000001, words=words, text=text)
        stored = program(0x3C000001, words=words, text=text)
        self.assertEqual(self.run_ps(listing(printed), blob(stored)), [f2u(1.0)] * 4)

    def test_llvm23_lit_form_of_an_f32_source(self):
        # llvm-mc 23.1.5's own text and encoding for these bytes: the f32 source's literal 0x40400000 (3.0) prints
        # in full as lit(...), and the printed encoding keeps only its low half (0). LLVM 21 prints the inline 0 and
        # drops the literal word. fma(3.0, 2.0, 0) = 6.0. The pipeline library has 5,733 such operands (#4343).
        text = "v_fma_mix_f32 v1, lit(0x40400000), v2, 0"
        words = "01 00 20 cc ff 04 02 02"
        printed = program(0x00000000, words=words, text=text)
        stored = program(0x40400000, words=words, text=text)
        self.assertEqual(self.run_ps(listing(printed), blob(stored)), [f2u(6.0)] * 4)
        self.assertEqual(self.run_ps(listing(printed, invalid=True), blob(stored)), [f2u(6.0)] * 4)

    def test_llvm23_lit_form_next_to_an_inline_constant(self):
        # The case above, as llvm-mc 23.1.5 prints it: lit(0x3c000001), whose high half 1.0 is src2.
        text = "v_fma_mix_f32 v1, v2, 1, lit(0x3c000001) op_sel:[0,0,1] op_sel_hi:[0,0,1]"
        words = "01 60 20 cc 02 03 fd 03"
        printed = program(0x00000001, words=words, text=text)
        stored = program(0x3C000001, words=words, text=text)
        self.assertEqual(self.run_ps(listing(printed), blob(stored)), [f2u(1.0)] * 4)

    def test_lit_around_a_register_is_refused(self):
        # A register inside lit() is no literal: refused, as before lit() was read (review on #6808).
        program_ = program(0, words="01 00 20 cc ff 04 02 02", text="v_fma_mix_f32 v1, lit(v2), v2, 0")
        with self.assertRaises(rdna2spirv.Unsupported):
            self.run_ps(listing(program_), None)

    def test_llvm23_lit_form_with_another_literal_is_refused(self):
        # The text names 3.0, the blob holds a literal whose low half differs: refused, as for the cut form.
        text = "v_fma_mix_f32 v1, lit(0x40400000), v2, 0"
        words = "01 00 20 cc ff 04 02 02"
        with self.assertRaises(rdna2spirv.Unsupported):
            self.run_ps(listing(program(0x00000000, words=words, text=text)),
                        blob(program(0x40400001, words=words, text=text)))


if __name__ == "__main__":
    unittest.main()
