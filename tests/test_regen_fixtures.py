"""tests/semantic/regen_fixtures.py finds its fixtures through a comment line in each source instead of a list.

    python -I tests/test_regen_fixtures.py

On throwaway folders (the repository root included, so a TEMP on another drive works): `regen_fixtures.py NAME`
marks the source and writes the fixture, a hand-written fixture is left alone, --check reports a stale fixture, a
missing one and a generated one whose source is no longer marked, and an unknown name is a usage error. Plus the
repository: every generated fixture has a marked source and the other way round.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import regen_fixtures as rf  # noqa: E402


class RegenFixturesTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida regen ")
        self.addCleanup(td.cleanup)
        self.asm = os.path.join(td.name, "shaders")
        self.fix = os.path.join(td.name, "disasm")
        os.makedirs(self.asm)
        os.makedirs(self.fix)
        for p in (mock.patch.object(rf, "ROOT", td.name), mock.patch.object(rf, "ASM_DIR", self.asm),
                  mock.patch.object(rf, "FIXTURE_DIR", self.fix)):
            p.start()
            self.addCleanup(p.stop)
        self.write(os.path.join(self.asm, "cs_a.asm"), "; what cs_a does\nv_mov_b32 v0, 1\ns_endpgm\n")
        self.write(os.path.join(self.asm, "cs_b.asm"), "s_endpgm\n")
        self.write(os.path.join(self.fix, "hand.s"), "; written by hand, not by the script\n\ts_endpgm\n")

    @staticmethod
    def write(path, text):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rf.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_a_named_source_is_marked_and_its_fixture_written(self):
        self.assertEqual(rf.marked_names(self.asm), [])
        self.assertEqual(self.run_main("cs_a")[0], 0)
        self.assertEqual(rf.marked_names(self.asm), ["cs_a"])
        self.assertEqual(rf.generated_names(self.fix), ["cs_a"])
        with open(os.path.join(self.fix, "cs_a.s"), encoding="utf-8") as f:
            self.assertEqual(f.read(), rf.render("cs_a"))
        self.assertNotIn(rf.ASM_MARK, rf.render("cs_a"))  # comments never reach the fixture
        self.assertEqual(self.run_main("cs_a")[0], 0)      # marking twice adds no second line
        with open(os.path.join(self.asm, "cs_a.asm"), encoding="utf-8") as f:
            self.assertEqual(f.read().count(rf.ASM_MARK), 1)
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_hand_written_fixtures_are_left_alone(self):
        self.run_main("cs_a", "cs_b")
        code, out, _ = self.run_main()  # rewrite everything it owns
        self.assertEqual(code, 0)
        self.assertNotIn("hand.s", out)
        with open(os.path.join(self.fix, "hand.s"), encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("; written by hand"))
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_check_reports_a_stale_fixture(self):
        self.run_main("cs_a")
        self.write(os.path.join(self.asm, "cs_a.asm"), rf.ASM_MARK + "\nv_mov_b32 v0, 2\ns_endpgm\n")
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("cs_a: fixture is stale or missing", out)

    def test_check_reports_a_missing_fixture(self):
        self.run_main("cs_a")
        os.remove(os.path.join(self.fix, "cs_a.s"))
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("cs_a: fixture is stale or missing", out)

    def test_check_reports_a_fixture_whose_source_lost_its_mark(self):
        self.run_main("cs_a")
        self.write(os.path.join(self.asm, "cs_a.asm"), "v_mov_b32 v0, 1\ns_endpgm\n")
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("cs_a: generated fixture whose runtime/test/shaders/cs_a.asm is not marked", out)

    def test_an_unknown_name_is_a_usage_error(self):
        code, _, err = self.run_main("cs_nope")
        self.assertEqual(code, 2)
        self.assertIn("cs_nope.asm", err)


class RepositoryTest(unittest.TestCase):
    def test_marked_sources_and_generated_fixtures_match(self):
        marked, generated = rf.marked_names(), rf.generated_names()
        self.assertGreater(len(marked), 0)
        self.assertEqual(marked, generated)
        self.assertEqual(tuple(marked), rf.SHADERS)


if __name__ == "__main__":
    unittest.main()
