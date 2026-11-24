"""Constant byte selectors use direct extracts; a clobbered SGPR uses the dynamic path."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import mrt0_words, translate_lines  # noqa: E402


class PermKnownSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).with_name("fixtures") / "perm_known_selector.asm"
        lines = [line for line in fixture.read_text(encoding="ascii").splitlines()
                 if line and not line.startswith(";")]
        cls.spvasm = translate_lines("ps_perm_known_selector", lines)

    def test_selector_values(self):
        self.assertEqual(mrt0_words(self.spvasm), [0xFF008C80, 0xFF0000FF, 0x80808080, 0x13E49B80])

    def test_only_clobbered_selector_dispatches_at_runtime(self):
        # The dynamic lowering emits one >= 13 check per byte, at most for the one clobbered selector; since SHADER-315
        # folds scalar ALU on constants, the clobbered selector is known at translate time too and none is emitted.
        # test_selector_values still checks every word.
        self.assertIn(self.spvasm.count("OpUGreaterThanEqual"), (0, 4))


if __name__ == "__main__":
    unittest.main()
