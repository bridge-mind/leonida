"""AGENTS.md names every tool that needs numpy or Pillow, and CONTRIBUTING.md says how pytest-based tests run.

    python -I tests/test_docs_python_deps.py

Both documents promise that Python tools use only the standard library "unless a file says otherwise" and that
`python -I tests/test_<name>.py` runs a unit test. The exceptions drift: `calibrated_texture.py` and `skinned_glb.py`
took numpy without joining the list, and 24 test files are pytest-based and fail (or run nothing) under `python -I`.
Reads tracked files only; no game files.
"""
import ast
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = {"numpy", "PIL"}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
            found += [os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/") for f in files]
        return found


def imports_third_party(source):
    """True if the module imports numpy or PIL anywhere (top level, in a try block or inside a function)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in THIRD_PARTY for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] in THIRD_PARTY:
            return True
    return False


def listed_tools(agents_text):
    """Basenames of the `*.py` tools in the AGENTS.md convention bullet that names the numpy/Pillow exceptions."""
    bullet = re.search(r"^- Python tools run with `python -I`.*?(?=^- |^\Z)", agents_text, re.S | re.M)
    return set(re.findall(r"`([\w]+\.py)`", bullet.group(0))) if bullet else set()


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


class DocsPythonDeps(unittest.TestCase):
    def test_listing_checker_sees_a_missing_and_a_stale_name(self):
        text = "- Python tools run with `python -I`; the exceptions are `a.py` and `gone.py`.\n- Next bullet `b.py`.\n"
        self.assertEqual(listed_tools(text), {"a.py", "gone.py"})
        self.assertTrue(imports_third_party("try:\n    import numpy as np\nexcept ImportError:\n    np = None\n"))
        self.assertTrue(imports_third_party("def f():\n    from PIL import Image\n"))
        self.assertFalse(imports_third_party("import json\nfrom os import path\n"))

    def test_agents_lists_the_tools_that_need_numpy_or_pillow(self):
        files = tracked_files()
        needing = {os.path.basename(p) for p in files
                   if p.endswith(".py") and (p.startswith(("tools/", "scripts/")) or p == "make_game.py")
                   and imports_third_party(read(p))}
        listed = listed_tools(read("AGENTS.md"))
        self.assertTrue(needing and listed)
        self.assertEqual(sorted(needing - listed), [], "tools that import numpy/Pillow but are not in AGENTS.md")
        self.assertEqual(sorted(listed - needing), [], "tools named in AGENTS.md that no longer import numpy/Pillow")

    def test_contributing_says_pytest_files_do_not_run_under_python_i(self):
        pytest_only = [p for p in tracked_files()
                       if re.fullmatch(r"tests/test_[\w]+\.py", p) and "__main__" not in read(p)]
        if pytest_only:
            self.assertIn("pytest", read("CONTRIBUTING.md"),
                          f"{len(pytest_only)} test files have no __main__ block (e.g. {pytest_only[0]}) but "
                          "CONTRIBUTING.md never mentions pytest")


if __name__ == "__main__":
    unittest.main()
