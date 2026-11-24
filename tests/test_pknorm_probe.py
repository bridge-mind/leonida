"""tools/pknorm_probe.py: its near-tie inputs tell single from double rounding of v_cvt_pknorm_i16/u16_f32, and the
translator side gives 0 for every NaN half. Standard library only; no GPU, no LLVM, no game files.

The near-tie words below are worked out by hand, not taken from the tool's own model. x is the f32 the word encodes;
"exact" is x * 32767 (i16) or x * 65535 (u16) without rounding; "f32" is that product rounded to f32 (ulp 2^-23 on
[1, 2), 2^-15 on [256, 512)), which is what a plain OpFMul gives before RoundEven. RDNA2 rounds once (gfx1036,
SHADER-339) and so does the translator since SHADER-363:

  0x38400180 (i16)  x ~ 4.5778e-5; exact 1.5 - 1.4e-9: once -> 1; f32 rounds it to 1.5, even -> 2
  0x37c000c0 (u16)  x ~ 2.2889e-5; exact 1.5 - 3.5e-10: once -> 1; f32 rounds it to 1.5 -> 2
  0x3c004101 (i16)  x ~ 7.8280e-3; exact 256.5 + 1.5e-5 (under half an f32 ulp, 2^-16): once -> 257; f32 256.5, even -> 256
  0x3b008081 (u16)  x ~ 1.9608e-3; exact 128.5 + 7.6e-6 (under 2^-17): once -> 129; f32 128.5, even -> 128
"""
import os
import struct
import sys
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import pknorm_probe as probe  # noqa: E402

BY_HAND = {   # (op, word): (single, double)
    ("v_cvt_pknorm_i16_f32", 0x38400180): (1, 2),
    ("v_cvt_pknorm_u16_f32", 0x37C000C0): (1, 2),
    ("v_cvt_pknorm_i16_f32", 0x3C004101): (257, 256),
    ("v_cvt_pknorm_u16_f32", 0x3B008081): (129, 128),
}


def f32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


class Inputs(unittest.TestCase):
    def test_hand_derivation(self):
        for (op, word), (single, double) in BY_HAND.items():
            with self.subTest(op=op, word=hex(word)):
                self.assertIn(word, probe.NEAR_TIES[op])
                self.assertEqual(probe.single_half(op, word), single)
                exact = Fraction(f32(word)) * probe.SCALE[op]
                tie = min(single, double) + Fraction(1, 2)
                self.assertNotEqual(exact, tie)
                self.assertLess(abs(exact - tie), Fraction(1, 1 << 16))       # within half an f32 ulp at 256
                self.assertEqual(exact > tie, single > double)               # the side of the tie picks the word

    def test_cases_cover_every_nan_and_near_tie_in_both_halves(self):
        cases = probe.cases()
        for op in probe.OPS:
            for w in probe.NANS + probe.NEAR_TIES[op]:
                with self.subTest(op=op, word=hex(w)):
                    self.assertTrue(any(c[0] == op and c[1] == w for c in cases))
                    self.assertTrue(any(c[0] == op and c[2] == w for c in cases))

    def test_nan_reference_is_zero(self):
        for op in probe.OPS:
            for w in probe.NANS:
                self.assertEqual(probe.single_half(op, w), 0, (op, hex(w)))


class Translator(unittest.TestCase):
    def test_translator_rounds_once_on_the_near_ties_and_zeroes_nan(self):
        # SHADER-363: the translator rounds the exact product once, as RDNA2 does (gfx1036, SHADER-339).
        cases = [(op, word, probe.NANS[0]) for op, word in BY_HAND] + [(op, probe.NANS[3], 0) for op in probe.OPS]
        got = probe.run_translator(cases)
        for (op, word, _), w in zip(cases[:len(BY_HAND)], got):
            self.assertEqual(w, BY_HAND[(op, word)][0], (op, hex(word)))     # low half: single rounding; high: NaN -> 0
        for w in got[len(BY_HAND):]:
            self.assertEqual(w, 0)


if __name__ == "__main__":
    unittest.main()
