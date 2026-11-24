"""Every tool that imports a sibling module starts under the project's own `python -I` convention.

`-I` implies `-P`, so the script's directory is not put on `sys.path`; a tool that imports the module next to it
has to add that directory itself (the pattern in `tools/anim_clips.py:40`). Two tools did not and died with
`ModuleNotFoundError` before `main()` — `calibrated_texture.py` and `ttpf_assets.py` — which this test caught.

The tool list is derived from the sources rather than written down, so a new tool with a sibling import is covered
without touching this file. `--help` is used because it reaches the argument parser without game files or a GPU;
the three module-only tools (`bcpack`, `calibrated_texture`, `cf_ref`) have no `__main__` and simply exit 0 on the
import. Tools that need numpy are skipped where it is not installed, as the texture tools are allowed to use it.
"""
import ast
import glob
import importlib.util
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
TOOL_NAMES = {os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(TOOLS, "*.py"))}
NUMERIC = {"amd_swizzle", "calibrated_texture", "swizzle_search", "tex_export", "tex_mode_probe", "tile_probe"}

# One process per tool is a few seconds in total; a hung import is a failure, not an infinite test.
TIMEOUT = 120


def imports(path):
    """The module names `path` imports at top level, relative imports excluded."""
    names = set()
    with open(path, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def sibling_importers():
    found = []
    for path in sorted(glob.glob(os.path.join(TOOLS, "*.py"))):
        own = os.path.splitext(os.path.basename(path))[0]
        if imports(path) & (TOOL_NAMES - {own}):
            found.append(path)
    return found


class ToolStartupTests(unittest.TestCase):
    def test_each_sibling_importing_tool_starts(self):
        paths = sibling_importers()
        self.assertGreater(len(paths), 20, "the sibling-import scan found almost nothing")
        have_numpy = importlib.util.find_spec("numpy") is not None
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name in NUMERIC and not have_numpy:
                continue
            with self.subTest(tool=name):
                proc = subprocess.run([sys.executable, "-I", path, "--help"], capture_output=True, text=True,
                                      timeout=TIMEOUT, errors="replace")
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertNotIn("Traceback", proc.stderr)

    def test_the_two_fixed_tools_are_in_the_scan(self):
        names = {os.path.splitext(os.path.basename(path))[0] for path in sibling_importers()}
        self.assertIn("calibrated_texture", names)
        self.assertIn("ttpf_assets", names)


if __name__ == "__main__":
    unittest.main()
