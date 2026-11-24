"""CSV reports must have the same bytes on every platform.

`csv.writer` defaults to `lineterminator="\\r\\n"`, so a tool that leaves it at the default writes CRLF on Windows and
LF everywhere else, and the same archive gives different report bytes depending on the machine that exported it.
`tools/awc_manifest.py` and `tools/awc_shorts.py` pass `lineterminator="\\n"` and `tools/bink_index.py` documents why
its stdout writer needs it; this test keeps every other writer on the same convention. No game files are read.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")

# Every csv writer call in tools/, with the two lines that may continue it.
WRITER = re.compile(r"\bcsv\.(?:Dict)?[Ww]riter\(")


def writer_calls():
    """Yield (file, line number, the call's source text) for every csv writer constructed in tools/."""
    for name in sorted(os.listdir(TOOLS)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(TOOLS, name)
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for index, line in enumerate(lines):
            if WRITER.search(line):
                yield name, index + 1, "\n".join(lines[index:index + 3])


class CsvTerminatorTest(unittest.TestCase):
    def test_every_csv_writer_sets_an_lf_terminator(self):
        missing = [f"{name}:{line}" for name, line, source in writer_calls()
                   if 'lineterminator="\\n"' not in source]
        self.assertEqual(missing, [], "csv writers without lineterminator='\\n' (CRLF on Windows)")

    def run_tool(self, tool, source):
        out = os.path.join(self.work, os.path.splitext(tool)[0] + ".csv")
        result = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool), source, out],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.isfile(out), result.stdout)
        with open(out, "rb") as f:
            return f.read()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = self.temp.name
        self.source = os.path.join(self.work, "empty")
        os.makedirs(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_reports_have_no_carriage_return(self):
        # These three scan a folder and write a header even when it holds nothing, so they need no archive.
        for tool in ("page_index.py", "rbf0_index.py", "restofworld_index.py"):
            with self.subTest(tool=tool):
                data = self.run_tool(tool, self.source)
                self.assertNotIn(b"\r", data, f"{tool} wrote a carriage return")
                self.assertTrue(data.endswith(b"\n"), f"{tool} did not end its last row with a newline")
                self.assertTrue(data.count(b"\n") >= 1)


if __name__ == "__main__":
    unittest.main()
