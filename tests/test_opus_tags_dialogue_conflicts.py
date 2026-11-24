"""Conflicting dialogue CSV rows must not choose arbitrary tags or modify Ogg files.

    python -I tests/test_opus_tags_dialogue_conflicts.py
"""
import csv
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_opus_tags as F  # noqa: E402


class DialogueConflictTest(unittest.TestCase):
    def setUp(self):
        self.fixture = F.TestFolder()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        with open(self.fixture.dialogue, encoding="utf-8", newline="") as source:
            self.row = next(csv.DictReader(source))
        self.before = self.snapshot()

    def snapshot(self):
        return {path: path.read_bytes() for path in Path(self.fixture.opus).rglob("*.ogg")}

    def cli(self, rows, *extra):
        F.write_csv(self.fixture.dialogue, F.DIALOGUE, rows)
        return subprocess.run([sys.executable, "-I", str(ROOT / "tools/opus_tags.py"), self.fixture.opus,
                               "--names", self.fixture.names, "--source", "story",
                               "--dialogue", self.fixture.dialogue, *extra],
                              capture_output=True, text=True, timeout=30)

    def assert_refused_without_writes(self, result):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("conflicting dialogue rows", result.stderr)
        self.assertIn(self.fixture.dialogue, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.snapshot(), self.before)

    def test_conflicting_subtitles_are_refused_in_either_order(self):
        other = dict(self.row, wave=self.row["wave"].upper(), text_en="A different invented subtitle.")
        for rows in ([self.row, other], [other, self.row]):
            with self.subTest(reversed=rows[0] is other):
                self.assert_refused_without_writes(self.cli(rows))

    def test_conflicts_in_labels_speakers_and_other_languages_are_refused(self):
        for field in ("label", "speaker", "text_de"):
            with self.subTest(field=field):
                other = dict(self.row, **{field: "different"})
                self.assert_refused_without_writes(self.cli([self.row, other]))

    def test_dry_run_also_reports_conflicting_input(self):
        other = dict(self.row, text_en="A different invented subtitle.")
        self.assert_refused_without_writes(self.cli([self.row, other], "--dry-run"))

    def test_identical_tags_with_case_only_wave_difference_are_allowed(self):
        other = dict(self.row, wave=self.row["wave"].upper(), key="UNUSED_METADATA")
        result = self.cli([self.row, other, self.row])
        self.assertEqual(result.returncode, 0, result.stderr)
        tags = self.fixture.tags("story", "bank_01", "0ABCDEF1.ogg")
        self.assertIn("SUBTITLE=Move. Now.", tags)
        self.assertIn("SPEAKER=ZED", tags)

    def test_markup_equivalent_subtitles_are_allowed(self):
        other = dict(self.row, text_en="Move. Now.")
        result = self.cli([self.row, other])
        self.assertEqual(result.returncode, 0, result.stderr)
        tags = self.fixture.tags("story", "bank_01", "0ABCDEF1.ogg")
        self.assertIn("SUBTITLE=Move. Now.", tags)

    def test_different_waves_can_have_different_dialogue_tags(self):
        other = dict(self.row, wave="door_slam_03", label="INVENTED_LINE_2", speaker="OTHER",
                     text_en="Another invented subtitle.")
        result = self.cli([self.row, other])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SUBTITLE=Move. Now.", self.fixture.tags("story", "bank_01", "0ABCDEF1.ogg"))
        self.assertIn("SUBTITLE=Another invented subtitle.", self.fixture.tags("story", "bank_02", "1FFFFFFF.ogg"))


if __name__ == "__main__":
    unittest.main()
