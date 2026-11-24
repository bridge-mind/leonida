"""Every text-mode open and Path read/write in the Python tools names its encoding.

    python -I tests/test_text_encoding.py

`open(path, "w")` with no `encoding=` writes in the locale code page (cp1252 on most Windows installs, UTF-8 on
Linux), so the same report is byte-different per platform, and a character the code page lacks raises
UnicodeEncodeError. CONTRIBUTING.md asks for `encoding="utf-8"` on text files. The test reads the source with `ast`
(standard library, no game data, no GPU) and fails for any `open` / `os.fdopen` in a text mode, or any
`read_text` / `write_text`, under tools/, scripts/, make_game.py and runtime/test/shaders/ without an `encoding=`
argument. Binary modes are exempt.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_text_write_line_endings import ROOT, python_files  # noqa: E402


def unpinned_encodings(source: str):
    """Line numbers of text-mode file calls in `source` that pass no `encoding=`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", getattr(func, "id", None))
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if name == "open" and isinstance(func, ast.Attribute) and not (
                isinstance(func.value, ast.Name) and func.value.id in ("io", "codecs")):
            continue                      # os.open, webbrowser.open, Image.open, ...
        if name in ("open", "fdopen"):
            mode = "r"
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if isinstance(mode, str) and "b" not in mode:
                found.append(node.lineno)
        elif name in ("read_text", "write_text"):
            found.append(node.lineno)
    return found


class TextEncoding(unittest.TestCase):
    def test_detector_finds_the_unpinned_forms(self):
        src = (
            "open('a')\n"                                          # 1
            "open('a', 'w', newline='\\n')\n"                      # 2
            "open('a', mode='a')\n"                                # 3
            "os.fdopen(3, 'w', newline='\\n')\n"                   # 4
            "p.write_text('x')\n"                                  # 5
            "p.read_text()\n"                                      # 6
            "open('a', encoding='utf-8')\n"                        # pinned
            "open('a', 'w', encoding='utf-8', newline='\\n')\n"    # pinned
            "open('a', 'rb')\n"                                    # binary
            "open('a', 'wb')\n"                                    # binary
            "os.open('a', 0)\n"                                    # not a text open
            "webbrowser.open('u')\n"                               # not a file
            "p.write_text('x', encoding='utf-8')\n"                # pinned
        )
        self.assertEqual(unpinned_encodings(src), [1, 2, 3, 4, 5, 6])

    def test_every_text_file_call_pins_its_encoding(self):
        bad = []
        count = 0
        for path in python_files():
            count += 1
            with open(path, encoding="utf-8") as f:
                for line in unpinned_encodings(f.read()):
                    bad.append(f"{os.path.relpath(path, ROOT)}:{line}")
        self.assertGreater(count, 50, "the scan found almost no Python files; is the layout different?")
        self.assertEqual(bad, [], "text files opened without encoding= (locale code page): " + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
