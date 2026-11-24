import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import mrt0_words, translate_lines  # noqa: E402


class RelativeVgprBoundsTests(unittest.TestCase):
    def test_moves_and_swap_keep_vgpr_accesses_in_range(self):
        fixture = Path(__file__).with_name("fixtures") / "relative_vgpr_bounds.asm"
        lines = [line for line in fixture.read_text(encoding="ascii").splitlines()
                 if line and not line.startswith(";")]
        spvasm = translate_lines("ps_relative_vgpr_bounds", lines)
        self.assertEqual(mrt0_words(spvasm), [0x41100000, 0x3F800000, 0x3F800000, 0x40800000])


if __name__ == "__main__":
    unittest.main()
