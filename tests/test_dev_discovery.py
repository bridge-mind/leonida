"""Exercise developer tool discovery through its public command line."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ENTRY = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
PRODUCERS = ("tools/boot_trace.py", "scripts/run_guest_scene.py")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.cwd = Path(scratch.name)
        self.checkout = self.cwd / "checkout d\u00e9mo"
        self.entry = self.checkout / "scripts" / "dev.py"
        self.entry.parent.mkdir(parents=True)
        shutil.copyfile(ENTRY, self.entry)

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-I", "-B", str(self.entry), *args],
            cwd=self.cwd, capture_output=True, timeout=10,
        )

    def test_help_lists_discovery(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"discover", result.stdout)

    def test_discovery_leaves_producers_unexecuted(self):
        for relative in PRODUCERS:
            producer = self.checkout / relative
            producer.parent.mkdir(exist_ok=True)
            producer.write_text(
                "from pathlib import Path\nPath(__file__).with_suffix('.ran').touch()\n",
                encoding="utf-8", newline="\n",
            )
        result = self.invoke("discover", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsInstance(json.loads(result.stdout), dict)
        for relative in PRODUCERS:
            self.assertFalse((self.checkout / relative).with_suffix(".ran").exists())

    def test_inventory_reflects_this_checkout(self):
        for tool_id, present_path, missing_id, missing_path in (
            ("boot-trace", PRODUCERS[0], "guest-scene", PRODUCERS[1]),
            ("guest-scene", PRODUCERS[1], "boot-trace", PRODUCERS[0]),
        ):
            with self.subTest(tool=tool_id), ExitStack() as cleanup:
                producer = self.checkout / present_path
                producer.parent.mkdir(exist_ok=True)
                producer.write_text(
                    "import sys\nprint(sys.argv[1:])\n", encoding="utf-8", newline="\n",
                )
                cleanup.callback(producer.unlink)
                # A namesake file in the caller's directory must not count.
                decoy = self.cwd / missing_path
                decoy.parent.mkdir(exist_ok=True)
                decoy.touch()
                result = self.invoke("discover", "--json")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, b"")
                inventory = json.loads(result.stdout)
                self.assertEqual(inventory["schema_version"], 1)
                tools = {entry["id"]: entry for entry in inventory["tools"]}
                self.assertEqual(len(tools), len(inventory["tools"]))
                found, missing = tools[tool_id], tools[missing_id]
                self.assertIs(found["present"], True)
                self.assertEqual(found["path"], present_path)
                self.assertTrue(found["requirements"])
                self.assertIs(missing["present"], False)
                self.assertEqual(missing["path"], missing_path)
                self.assertIsNone(missing["help_argv"])
                self.assertIs(inventory["live_controls"]["available"], False)
                self.assertTrue(inventory["live_controls"]["reason"].strip())
                self.assertEqual(found["help_argv"][0], sys.executable)
                help_result = subprocess.run(found["help_argv"], cwd=self.checkout, capture_output=True,
                                             env={**os.environ, "PATH": ""}, timeout=10)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertEqual(help_result.stdout.strip(), b"['--help']")

    def test_invalid_option_is_an_error(self):
        result = self.invoke("discover", "--no-such-option")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()
