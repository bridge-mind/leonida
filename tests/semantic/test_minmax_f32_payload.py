"""Binary min/max with quiet NaNs, using the operand priority measured on gfx1030."""
import unittest
from pathlib import Path

from support import mrt0_words, translate_lines


class MinMaxPayloadTests(unittest.TestCase):
    def test_minmax_f32_payload(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "ps_minmax_f32_payload.asm"
        module = translate_lines(fixture.stem, fixture.read_text(encoding="ascii").splitlines())

        a, b, half = 0x7FE00080, 0xFFD10040, 0x3F000000
        for lhs, rhs, expected in (
            (a, b, (a, a, b, b)),
            (a, half, (half, half, half, half)),
            (half, b, (half, half, half, half)),
            (0, 0x80000000, (0x80000000, 0, 0x80000000, 0)),
        ):
            with self.subTest(lhs=hex(lhs), rhs=hex(rhs)):
                words = mrt0_words(module, push_constants=[lhs, rhs] + [0] * 30)
                self.assertEqual(words, [word ^ 0x40C00000 for word in expected])
