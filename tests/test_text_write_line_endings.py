"""Every text-mode file write in the Python tools names its line ending (BUILD-468).

`open(path, "w")` with no `newline=` turns each "\\n" into "\\r\\n" on Windows, so the same JSON report, manifest or
assembly listing is CRLF there and LF on Linux and macOS: the same input gives byte-different output per platform
(checksums and diffs differ). The CSV writers were fixed in #1573; this covers the other text writers. This test reads
the source with `ast` (standard library, no game data, no GPU) and fails for any `open`, `io.open`, `Path.open`,
`os.fdopen` or `tempfile.NamedTemporaryFile` (and `TemporaryFile`, `SpooledTemporaryFile`) in a text write mode (`w`,
`a`, `x`, `r+`), or any `Path.write_text`, under tools/, scripts/, make_game.py and runtime/test/shaders/ that has no
`newline=` argument. `codecs.open` is left alone: it never translates line ends and takes no `newline=`. Pass `newline="\\n"` for LF output (or `newline=""` when the text carries its own line ends,
like a csv writer).
"""
import ast
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN = ("tools", "scripts", "make_game.py", os.path.join("runtime", "test", "shaders"))


def python_files():
    for entry in SCAN:
        path = os.path.join(ROOT, entry)
        if os.path.isfile(path):
            yield path
            continue
        for base, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "build") and not d.startswith(".")]
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(base, name)


MODE_CHARS = set("rwaxbt+")
# Where each call takes its mode: open(file, mode), io/gzip/bz2/lzma.open(file, mode), os.fdopen(fd, mode),
# Path.open(mode) and tempfile.*TemporaryFile(mode) (binary unless the mode says otherwise).
MODULE_OPENS = {"io", "gzip", "bz2", "lzma"}
TEMPFILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
# open() calls that never write translated text: codecs.open writes "\n" as it is and takes no newline=; os.open
# takes flags; tarfile, zipfile and webbrowser open archives or URLs.
NOT_TEXT_OPENS = {"codecs", "os", "tarfile", "zipfile", "webbrowser"}


def text_write_mode(mode) -> bool:
    """True for a constant mode string that can write text ("w", "a+", "xt", "r+", ...)."""
    return (isinstance(mode, str) and bool(mode) and set(mode) <= MODE_CHARS and "b" not in mode
            and (any(c in mode for c in "wax") or ("r" in mode and "+" in mode)))


def call_mode(node: ast.Call, index: int):
    mode = None
    if len(node.args) > index and isinstance(node.args[index], ast.Constant):
        mode = node.args[index].value
    for k in node.keywords:
        if k.arg == "mode" and isinstance(k.value, ast.Constant):
            mode = k.value.value
    return mode


def mode_index(func):
    """Position of the mode argument for the file-opening calls this test knows, or None for any other call."""
    name = getattr(func, "attr", getattr(func, "id", None))
    if name in TEMPFILES:
        return 0
    if name == "fdopen":
        return 1
    if name != "open":
        return None
    if isinstance(func, ast.Name):
        return 1   # the builtin open
    owner = func.value.id if isinstance(func.value, ast.Name) else None
    if owner in NOT_TEXT_OPENS:
        return None
    return 1 if owner in MODULE_OPENS else 0   # anything else is taken as Path.open(mode)


def unpinned_text_writes(source: str):
    """Line numbers of text-mode writes in `source` that pass no `newline=`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or any(k.arg == "newline" for k in node.keywords):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name == "write_text":
            found.append(node.lineno)
            continue
        index = mode_index(node.func)
        if index is not None and text_write_mode(call_mode(node, index)):
            found.append(node.lineno)
    return found


class TextWriteLineEndings(unittest.TestCase):
    def test_detector_finds_the_unpinned_forms(self):
        src = (
            "open('a', 'w')\n"                                     # 1
            "open('a', 'w', encoding='utf-8')\n"                   # 2
            "open('a', mode='a')\n"                                # 3
            "os.fdopen(3, 'w', encoding='utf-8')\n"                # 4
            "p.write_text('x', encoding='utf-8')\n"                # 5
            "open('a', 'w', newline='\\n')\n"                      # pinned
            "open('a', 'w', newline='')\n"                         # pinned
            "open('a', 'wb')\n"                                    # binary
            "open('a')\n"                                          # read
            "open('a', 'r', encoding='utf-8')\n"                   # read
            "p.write_text('x', newline='\\n')\n"                   # pinned
            "pathlib.Path('a').open('w', encoding='utf-8')\n"     # 12
            "p.open(mode='a')\n"                                   # 13
            "io.open('out.txt', 'w')\n"                            # 14
            "tempfile.NamedTemporaryFile('w', delete=False)\n"     # 15
            "NamedTemporaryFile(mode='w+', encoding='utf-8')\n"    # 16
            "tempfile.SpooledTemporaryFile(mode='wt')\n"           # 17
            "gzip.open('out.txt.gz', 'wt')\n"                      # 18
            "tempfile.NamedTemporaryFile(mode='w', newline='\\n')\n"  # pinned
            "tempfile.NamedTemporaryFile()\n"                      # binary (the default w+b)
            "tempfile.NamedTemporaryFile('w+b')\n"                 # binary
            "p.open()\n"                                           # read
            "p.open('rb')\n"                                       # read
            "zf.open('words.txt')\n"                               # a member name, not a mode
            "codecs.open('a', 'w', 'utf-8')\n"                     # no newline translation in codecs
            "tarfile.open('a', 'w:gz')\n"                          # an archive mode, binary
            "os.open('a', os.O_WRONLY)\n"                          # flags, not a mode string
        )
        self.assertEqual(unpinned_text_writes(src), [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18])

    def test_detector_finds_unpinned_text_update_modes(self):
        src = (
            "open('a', 'r+')\n"
            "p.open(mode='rt+')\n"
            "os.fdopen(3, 'r+t')\n"
            "open('a', 'r+', newline='\\n')\n"
            "open('a', 'rb+')\n"
            "open('a', 'r')\n"
        )
        self.assertEqual(unpinned_text_writes(src), [1, 2, 3])

    def test_every_text_write_pins_its_newline(self):
        bad = []
        count = 0
        for path in python_files():
            count += 1
            with open(path, encoding="utf-8") as f:
                for line in unpinned_text_writes(f.read()):
                    bad.append(f"{os.path.relpath(path, ROOT)}:{line}")
        self.assertGreater(count, 50, "the scan found almost no Python files; is the layout different?")
        self.assertEqual(bad, [], "text writes without newline= (CRLF on Windows, LF elsewhere): " + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
