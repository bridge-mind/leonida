"""Exact f16 Fract infinity word measured on gfx1036 (docs/design.md)."""
import unittest
from pathlib import Path

from support import mrt0_words, translate_lines


class FractInfinityTests(unittest.TestCase):
    def test_fract_f16_infinity(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "ps_fract_f16_infinity.asm"
        module = translate_lines(fixture.stem, fixture.read_text(encoding="ascii").splitlines())

        for source, clamped in ((0x3E00, 0x3800), (0x7C00, 0), (0xFC00, 0)):
            with self.subTest(source=hex(source)):
                words = mrt0_words(module, push_constants=[source] + [0] * 31)
                self.assertEqual(words, [0x1357FE00, 0x2468FE00, 0x3F800000, 0x12340000 | clamped])
