"""make_game.py: the shaders stage refuses a translation index that lost carved shaders.

rdna2spirv.py drops a shader that is refused or whose module fails spirv-val and still exits 0, so a full run can
carve 26,503 shaders and finish "done" with an index of fewer: a port built from it silently lacks the rest
(measured end to end at aa2988cc: 26,503 carved, 26,473 indexed, exit 0). verify_translation_completeness()
compares the counts and fails; an empty merge is already refused inside rdna2spirv.py.

python -I tests/test_make_game_translation_completeness.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def workdir(dxbc_count, index_names):
    """A carve index of dxbc_count entries beside a translation index naming index_names."""
    tmp = tempfile.TemporaryDirectory()
    dxbc = os.path.join(tmp.name, "dxbc")
    spirv = os.path.join(tmp.name, "spirv")
    os.makedirs(dxbc)
    os.makedirs(spirv)
    with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": f"s{i}"} for i in range(dxbc_count)], f)
    with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({name: {} for name in index_names}, f)
    return tmp, dxbc, spirv


class TranslationCompletenessTests(unittest.TestCase):
    def run_check(self, dxbc, spirv):
        """(exit code, stderr) of verify_translation_completeness, with fail() allowed to exit."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                make_game.verify_translation_completeness(dxbc, spirv)
                code = 0
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def test_index_with_every_carved_shader_passes_quietly(self):
        tmp, dxbc, spirv = workdir(3, ["s0", "s1", "s2"])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_index_that_lost_shaders_fails_naming_both_counts(self):
        # The shape measured at aa2988cc: 26,503 carved, 26,473 reached the index.
        tmp, dxbc, spirv = workdir(26503, [f"s{i}" for i in range(26473)])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)
            self.assertIn("26503", err)
            self.assertIn("26473", err)
            self.assertIn("30", err)

    def test_an_index_that_is_not_an_object_fails_rather_than_passing(self):
        tmp = tempfile.TemporaryDirectory()
        with tmp:
            dxbc = os.path.join(tmp.name, "dxbc")
            spirv = os.path.join(tmp.name, "spirv")
            os.makedirs(dxbc)
            os.makedirs(spirv)
            with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([], f)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([], f)          # a list where the merged index should be an object
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)

    def test_a_missing_translation_index_is_left_to_the_paths_that_own_it(self):
        # A translation that did not run, or a merge that failed, already fails the stage before this check
        # (rdna2spirv.py exits nonzero; an empty merge is refused at the merge): a missing or unreadable index
        # here is an intermediate state, not the silent partial translation this guard exists for.
        tmp = tempfile.TemporaryDirectory()
        with tmp:
            dxbc = os.path.join(tmp.name, "dxbc")
            spirv = os.path.join(tmp.name, "spirv")
            os.makedirs(dxbc)
            os.makedirs(spirv)
            with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": "s0"}], f)
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_an_empty_index_is_left_to_its_own_guard(self):
        # #3328 refuses a merge that publishes {} and translates a saved one again; comparing its count against
        # the carve here would race that fix, so a nonempty-but-short index is the partial translation to catch.
        tmp, dxbc, spirv = workdir(2, [])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
