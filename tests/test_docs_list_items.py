"""No list item in the Markdown docs is repeated a few lines later.

    python -I tests/test_docs_list_items.py

When two branches rewrite the same bullet and a merge keeps both sides, the page shows the bullet twice: the current
text and the one it replaced, often next to each other (docs/d3d12_vulkan.md had its RTV/DSV bullet twice and its
"Hash-table cost" / "Wave64" risks twice each; docs/character_models.md kept the first line of a rewritten bullet).
A reader cannot tell which copy is current. This test flags two list items within WINDOW lines of each other whose
openings match. Reads tracked Markdown only; no game files.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(\S.*)$")
WINDOW = 8      # lines between the two items
OPENING = 60    # characters of the item's text that must match (after the list marker)


def repeated_items(text):
    """[(line a, line b)] for list items within WINDOW lines whose first OPENING characters are equal."""
    items, in_fence = [], False
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = None if in_fence else ITEM.match(line)
        if m and len(m.group(2)) >= OPENING:
            items.append((n, len(m.group(1)), m.group(2)[:OPENING]))
    found = []
    for i, (n, indent, head) in enumerate(items):
        for n2, indent2, head2 in items[i + 1:]:
            if n2 - n > WINDOW:
                break
            if indent2 == indent and head2 == head:
                found.append((n, n2))
    return found


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=ROOT, capture_output=True, check=True).stdout
        files = [os.fsdecode(name) for name in out.split(b"\0") if name]
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedListItems(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "requires Git")
    def test_reads_tracked_markdown_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "core.quotePath", "true"], cwd=tmp, check=True)
            page = "space name.md"
            with open(os.path.join(tmp, page), "w", encoding="utf-8", newline="\n") as f:
                f.write("# Example\n")
            subprocess.run(["git", "add", "--", page], cwd=tmp, check=True)
            with patch(__name__ + ".ROOT", tmp):
                self.test_docs_have_no_repeated_list_items()
                self.assertEqual(markdown_files(), [page])

    def test_checker_flags_a_repeated_item_only(self):
        current = "- **Wave64.** A native wave64 shader needs a 64-wide subgroup in its stage: AMD can require it"
        old = "- **Wave64.** A native wave64 shader needs a 64-wide subgroup in its stage: AMD (old wording)"
        text = "\n".join([current, "  continued", "- **Other.** " + "x" * 70, old, "",
                          "- **19 two-slice 2D arrays** (`depth = 2`, `depth - 1 = 1`) appeared as `depth = 1`",
                          "- **99 three-slice 2D arrays** (`depth = 3`, `depth - 1 = 2`) appeared as `depth = 2`",
                          "```", current, current, "```"])
        self.assertEqual(repeated_items(text), [(1, 4)])

    def test_docs_have_no_repeated_list_items(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                found += [f"{path}:{a} and :{b}" for a, b in repeated_items(f.read())]
        self.assertEqual(found, [], "a list item repeated a few lines later (keep the current copy): " +
                         "; ".join(found))


if __name__ == "__main__":
    unittest.main()
