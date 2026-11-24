"""Check the boot-trace entry point against the standalone producer."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "dev.py"
PRODUCER = ROOT / "tools" / "boot_trace.py"


class BootTraceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.startup = "sess\u00e3o sint\u00e9tica.jsonl"
        (self.cwd / self.startup).write_text(
            '{"kind":"trace_open","pid":17}\n{"kind":"exit","code":7}\n',
            encoding="utf-8", newline="\n",
        )
        self.d3d = "render calls.log"
        (self.cwd / self.d3d).write_text(
            "[d3d12-trace] #2 Example call unsupported\n", encoding="utf-8", newline="\n",
        )

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-I", "-B", *args],
            cwd=self.cwd, capture_output=True, timeout=10,
        )

    def compare(self, expected_code, *args):
        direct = self.invoke(PRODUCER, *args)
        result = self.invoke(CLI, "boot-trace", *args)
        self.assertEqual(direct.returncode, expected_code, direct.stderr)
        self.assertEqual(result.returncode, expected_code, result.stderr)
        self.assertEqual(result.stdout, direct.stdout)
        self.assertEqual(result.stderr, direct.stderr)
        return result

    def test_help_and_text(self):
        self.assertIn(b"--d3d12", self.compare(0, "--help").stdout)
        self.assertTrue(self.compare(0, self.startup).stdout.strip())

    def test_combined_json(self):
        result = self.compare(0, self.startup, "--d3d12", self.d3d, "--json", "--limit", "1")
        report = json.loads(result.stdout)
        self.assertEqual(report["startup"]["exit"], 7)
        self.assertEqual(report["d3d12"]["calls"], 1)
        self.assertEqual(result.stderr, b"")

    def test_partial_json(self):
        for startup, d3d, missing in (
            (self.startup, "absent.log", "d3d12"),
            ("absent.jsonl", self.d3d, "startup"),
        ):
            with self.subTest(missing=missing):
                result = self.compare(2, startup, "--d3d12", d3d, "--json")
                report = json.loads(result.stdout)
                self.assertIsNone(report[missing])
                self.assertTrue(any(report.values()))
                self.assertIn(b"absent", result.stderr)

    def test_total_failure(self):
        result = self.compare(2, "absent.jsonl", "--d3d12", "absent.log", "--json")
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"absent.jsonl", result.stderr)
        self.assertIn(b"absent.log", result.stderr)

    def test_missing_producer(self):
        isolated_cli = self.cwd / "empty checkout" / "scripts" / "dev.py"
        isolated_cli.parent.mkdir(parents=True)
        shutil.copyfile(CLI, isolated_cli)
        result = self.invoke(isolated_cli, "boot-trace", self.startup, "--json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"boot_trace.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
