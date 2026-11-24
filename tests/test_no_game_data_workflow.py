"""The extension check in .github/workflows/no-game-data.yml sees every tracked name, quoted or not.

    python -I tests/test_no_game_data_workflow.py

By default `git ls-files` prints a name with a byte outside printable ASCII (an accented letter, a tab) in double
quotes with octal escapes, so a line-anchored `\\.spvasm$` never matches it. scripts/check_no_game_data.py cannot
stand in for that step: it reads magics, and text formats such as `.spvasm` and `.obj` have none. This test takes
the step's `run:` script out of the workflow (not a copy, so it cannot drift) and runs it with bash in a throwaway
git repository of empty files. Needs git and bash; no game files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "no-game-data.yml")
STEP = "- name: No game data tracked"


def step_script():
    with open(WORKFLOW, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == STEP)
    run = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "run: |")
    body = []
    indent = None
    for line in lines[run + 1:]:
        if line.strip():
            width = len(line) - len(line.lstrip())
            if indent is None:
                indent = width
            if width < indent:
                break
        body.append(line[indent:] if indent else line)
    return "\n".join(body).strip() + "\n"


def bash():
    # On Windows, prefer Git's bash: System32\bash.exe is WSL and does not see this file system the same way.
    git = shutil.which("git")
    if os.name == "nt":
        folder = os.path.dirname(git or "")
        for _ in range(3):                         # git.exe sits in Git\cmd, Git\bin or Git\mingw64\bin
            if os.path.isfile(os.path.join(folder, "bin", "bash.exe")):
                return os.path.join(folder, "bin", "bash.exe")
            folder = os.path.dirname(folder)
        return None
    return shutil.which("bash")


@unittest.skipUnless(shutil.which("git") and bash(), "needs git and bash")
class ExtensionCheck(unittest.TestCase):
    def run_step(self, names):
        with tempfile.TemporaryDirectory() as repo:
            git = ["git", "-c", "core.quotePath=true", "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(git + ["init", "-q", repo], check=True)
            for name in names:
                with open(os.path.join(repo, name), "wb"):
                    pass
            subprocess.run(git + ["-C", repo, "add", "--", *names], check=True)
            subprocess.run(git + ["-C", repo, "config", "core.quotePath", "true"], check=True)
            script = os.path.join(repo, "step.sh")
            with open(script, "w", encoding="utf-8", newline="\n") as f:
                f.write(step_script())
            # GitHub runs a `run:` step as `bash --noprofile --norc -eo pipefail {0}`.
            return subprocess.run([bash(), "--noprofile", "--norc", "-eo", "pipefail", "step.sh"], cwd=repo,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_a_clean_tree_passes(self):
        result = self.run_step(["README.md", "notes.txt"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_plain_name_is_caught(self):
        result = self.run_step(["README.md", "shader.spvasm"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("shader.spvasm", result.stdout)

    def test_a_name_git_quotes_is_caught(self):
        names = ["README.md", "ünï.spvasm", "café.obj"]
        result = self.run_step(names)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("ünï.spvasm", result.stdout)
        self.assertIn("café.obj", result.stdout)
        self.assertNotIn("README.md", result.stdout)


if __name__ == "__main__":
    unittest.main()
