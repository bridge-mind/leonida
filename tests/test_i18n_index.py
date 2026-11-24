"""docs/i18n/README.md: one row per language, linked to a synced quickstart.

    python -I tests/test_i18n_index.py

Reads the index, docs/README.md, and linked quickstarts; no game data. Synthetic duplicate rows and
sync markers are checked so these checks cannot pass by parsing nothing.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "i18n", "README.md")
DOCS_README = os.path.join(ROOT, "docs", "README.md")

ROW = re.compile(r"^\|\s*(?P<language>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*"
                 r"(?P<commit>[^|]+?)\s*\|\s*(?P<maintainer>[^|]+?)\s*\|$")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE = re.compile(r"`([A-Za-z][A-Za-z0-9-]*)`")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
SYNC_MARKER = re.compile(r"`([0-9a-f]{7,40})`")


def sync_marker_matches(index_commit, quickstart_text):
    """Whether the opening note has exactly the commit recorded in the index."""
    lines = quickstart_text.splitlines()
    heading = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    start = heading + 1 if heading is not None else 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start == len(lines):
        return False

    opening_note = []
    for line in lines[start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            break
        opening_note.append(line[1:].lstrip() if line.startswith(">") else line)

    markers = SYNC_MARKER.findall("\n".join(opening_note))
    return len(markers) == 1 and markers[0] == index_commit


def parse(text):
    """Rows of the language table. Raises ValueError for a row the table format does not parse."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").replace(" ", "")) <= set("-:"):
            continue                       # the header separator
        match = ROW.match(line)
        if not match:
            raise ValueError(f"malformed table row: {line}")
        if match.group("language").strip() == "Language":
            continue                       # the header
        link = LINK.search(match.group("file"))
        code = CODE.search(match.group("language"))
        if not link or not code:
            raise ValueError(f"row without a file link or a language code: {line}")
        rows.append({"code": code.group(1), "file": link.group(1),
                     "commit": match.group("commit").strip().strip("`"),
                     "maintainer": match.group("maintainer").strip().strip("`")})
    return rows


def duplicate_codes(rows):
    seen = set()
    duplicates = []
    for row in rows:
        if row["code"] in seen:
            duplicates.append(row["code"])
        seen.add(row["code"])
    return duplicates


class IndexTest(unittest.TestCase):
    def read_rows(self):
        with open(INDEX, encoding="utf-8") as f:
            return parse(f.read())

    def test_one_row_per_language(self):
        rows = self.read_rows()
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(duplicate_codes(rows), [])

    def test_every_row_points_at_an_existing_quickstart(self):
        base = os.path.dirname(INDEX)
        for row in self.read_rows():
            with self.subTest(language=row["code"]):
                path = os.path.normpath(os.path.join(base, row["file"]))
                self.assertTrue(os.path.isfile(path), path)
                self.assertTrue(row["maintainer"].startswith("@"), row)
                self.assertRegex(row["commit"], COMMIT.pattern, row)

    def test_every_synced_commit_matches_its_quickstart_marker(self):
        base = os.path.realpath(os.path.dirname(INDEX))
        for row in self.read_rows():
            path = os.path.realpath(os.path.join(base, row["file"]))
            with self.subTest(language=row["code"]):
                try:
                    within_i18n = os.path.commonpath((base, path)) == base
                except ValueError:
                    within_i18n = False
                self.assertTrue(within_i18n, f"{path} escapes the i18n directory")
                with open(path, encoding="utf-8") as f:
                    quickstart = f.read()
                self.assertTrue(sync_marker_matches(row["commit"], quickstart),
                                f"{path} sync marker does not match {row['commit']}")

    def test_docs_readme_names_every_language_once(self):
        codes = [row["code"] for row in self.read_rows()]
        with open(DOCS_README, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines()
                     if line.startswith("Community quickstarts (stable onboarding text only")]
        self.assertEqual(len(lines), 1, f"docs/README.md has {len(lines)} 'Other languages' lines")
        line = lines[0]
        for code in codes:
            self.assertEqual(line.count(f"`{code}`"), 1, f"{code} is not named exactly once: {line}")
        listed = set(re.findall(r"`([A-Za-z]{2,3}(?:-[A-Za-z]+)?)`", line))
        self.assertEqual(listed - set(codes), set(),
                         f"docs/README.md names languages without an i18n row: {listed - set(codes)}")

    def test_a_duplicate_language_is_reported(self):
        table = ("| Language | File | Synced commit | Maintainer |\n"
                 "|---|---|---|---|\n"
                 "| 简体中文 (`zh-CN`) | [`zh-CN/quickstart.md`](zh-CN/quickstart.md) | `e7584c5` | `@a` |\n"
                 "| 简体中文 (`zh-CN`) | [`zh-CN/quickstart.md`](zh-CN/quickstart.md) | `0d61c357` | `@a` |\n")
        rows = parse(table)
        self.assertEqual(duplicate_codes(rows), ["zh-CN"])

    def test_a_malformed_row_is_rejected(self):
        table = ("| Language | File | Synced commit | Maintainer |\n"
                 "|---|---|---|---|\n"
                 "| 简体中文 | `zh-CN/quickstart.md` | `e7584c5` | `@a` |\n")
        with self.assertRaises(ValueError):
            parse(table)

    def test_synced_commit_matches_quickstart_marker(self):
        page = "> This guide is synced with commit `01234567`.\n"
        self.assertTrue(sync_marker_matches("01234567", page))

    def test_synced_commit_mismatch_is_detected(self):
        page = "> This guide is synced with commit `01234567`.\n"
        self.assertFalse(sync_marker_matches("89abcdef", page))

    def test_hash_outside_opening_note_does_not_count_as_sync_marker(self):
        page = ("> [!NOTE]\n"
                "> This guide has no sync marker.\n"
                "\n"
                "Example output: `01234567`.\n"
                "\n"
                "## First section\n")
        self.assertFalse(sync_marker_matches("01234567", page))


if __name__ == "__main__":
    unittest.main()
