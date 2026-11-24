"""Each table in docs/design.md, docs/README.md and docs/testing.md names a topic once, and every row has the
same number of cells as its header.

    python -I tests/test_docs_design_rows.py

The translator contract table is where every new rule is recorded (CONTRIBUTING.md, AGENTS.md). When two branches
edit the same row and a merge keeps both sides, the table ends up with two rows for one topic, each holding facts the
other lacks, and a reader who stops at the first one misses the rest. On 2026-11-22 twenty topics appeared two to six
times; the docs index and the testing guide's "Known problems" had the same pattern. This test fails on a repeated
first cell within one table of those pages, where the first cell is the topic (other pages repeat it on purpose, e.g.
one row per benchmark size). No game files or build tools.

The second check is the table's shape: a rule that writes a literal pipe (`|v|` in a list of source modifiers)
has to escape it as `\\|`, or GitHub reads it as a separator, renders the row into columns the header does not
have and drops the tail of the rule — text lost in a page nobody re-reads. On 2026-11-22 eighteen rows of the
contract table had that shape (four to thirteen cells against a three-cell header).
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ("design.md", "README.md", "testing.md")


def unescaped_pipes(line):
    """How many table separators a row really has: a cell that needs a literal pipe writes it ``\\|``."""
    count, i = 0, 0
    while i < len(line):
        if line[i] == "\\":                          # the next character is literal, pipe or not
            i += 2
            continue
        if line[i] == "|":
            count += 1
        i += 1
    return count


def mis_shaped_rows(lines):
    """[(line, cells, header cells)] for rows whose cell count differs from their table's header.

    A rule that lists modifiers (`|v|`, `-|v|`) has to escape those pipes as `\\|`. Left raw, GitHub reads each one
    as a separator and renders the row into columns the header does not have, dropping the tail of the rule, so the
    table silently loses text instead of failing.
    """
    found = []
    header, in_fence = None, False
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if in_fence or not line.startswith("|"):
            header = None
            continue
        cells = unescaped_pipes(line)
        if set(line.strip()) <= set("|-: "):           # the |---| rule carries the header's shape
            header = cells
            continue
        if header is None:
            header = cells
            continue
        if cells != header:
            found.append((n, cells, header))
    return found


def repeated_topics(lines):
    """[(topic, [1-based line numbers])] for first cells that appear more than once in one table."""
    found = []
    table, in_fence = [], False

    def close():
        seen = {}
        for n in table[2:]:                         # skip the header and the |---| rule
            cells = lines[n - 1].split("|")
            if len(cells) > 2 and cells[1].strip():
                seen.setdefault(cells[1].strip(), []).append(n)
        found.extend((k, v) for k, v in seen.items() if len(v) > 1)

    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("|"):
            table.append(n)
            continue
        if table:
            close()
            table = []
    if table:
        close()
    return found


class DesignTableRows(unittest.TestCase):
    def test_checker_finds_a_repeated_topic_within_one_table_only(self):
        lines = ["| Topic | Rule |", "|---|---|", "| EXEC | old |", "| LDS | x |", "| EXEC | new |", "",
                 "| Topic | Rule |", "|---|---|", "| LDS | other table |", "```", "| EXEC | in a code block |",
                 "| EXEC | in a code block |", "```"]
        self.assertEqual(repeated_topics(lines), [("EXEC", [3, 5])])

    def test_checker_finds_a_stray_pipe_but_not_an_escaped_one(self):
        lines = ["| Topic | Rule |", "|---|---|", "| EXEC | writes |v| |", "| LDS | fine |", "",
                 "| Topic | Rule |", "|---|---|", "| EXEC | writes \\|v\\| |", "```", "| x | a | b |", "```"]
        self.assertEqual(mis_shaped_rows(lines), [(3, 5, 3)])

    def test_each_page_has_rows_its_header_can_hold(self):
        for page in PAGES:
            with self.subTest(page=page):
                with open(os.path.join(ROOT, "docs", page), encoding="utf-8") as f:
                    lines = f.read().split("\n")
                wrong = mis_shaped_rows(lines)
                self.assertEqual(wrong, [], f"rows to fix in docs/{page} (line, cells, header cells): {wrong}")

    def test_each_page_names_each_topic_once_per_table(self):
        for page in PAGES:
            with self.subTest(page=page):
                with open(os.path.join(ROOT, "docs", page), encoding="utf-8") as f:
                    lines = f.read().split("\n")
                repeated = repeated_topics(lines)
                self.assertEqual(repeated, [], f"rows to merge into one in docs/{page} (topic, lines): " +
                                 "; ".join(f"{k[:60]!r} at {v}" for k, v in repeated))

    def test_deduplicating_a_row_keeps_its_unique_border_mode_facts(self):
        """Removing a duplicated sentence must not take unique measured facts with it.

        The "Other additions" row once held two copies of the border-mode and GPU-check paragraphs. A cleanup that
        removed the whole block instead of the duplicates lost the BORDER_COLOR_TYPE / SHADER-348 description while
        the tests above stayed green, and left a ``". ."`` artifact where the join had been.
        """
        with open(os.path.join(ROOT, "docs", "design.md"), encoding="utf-8") as f:
            rows = [line for line in f.read().split("\n") if line.startswith("| Other additions |")]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for fact in ("BORDER_COLOR_TYPE", "SHADER-348", "FORCE_UNNORMALIZED",
                     "Modes 6 (clamp border) and 7 (mirror once border)", "remaining approximations",
                     "ps_gather_wrap.asm"):
            with self.subTest(fact=fact):
                self.assertIn(fact, row)
        self.assertNotIn(". .", row)


if __name__ == "__main__":
    unittest.main()
