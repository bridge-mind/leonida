"""scripts/check_apt_key.py accepts only the pinned key, and the workflows check every apt key they fetch.

    python -I tests/test_check_apt_key.py

GnuPG is stood in for by a small Python program that prints the `--with-colons` listing written in each test's "key
file", so the tests run anywhere without a real key or gpg: the one key, another key, two primary keys, a primary key
with subkeys, a file gpg can't read. The workflows are read as text: every `.key`/`.gpg`/`.asc` URL they fetch must go
to a temporary file that `scripts/check_apt_key.py` checks before anything installs it. Standard library only.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import check_apt_key  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "check_apt_key.py")
WINEHQ = "D43F640145369C51D786DDEA76F1A20FF987672F"
LLVM = "6084F3CF814B57C1CF12EFD515CF4D18AF4F7421"

# A stand-in for `gpg --batch --show-keys --with-colons FILE`: prints FILE, or fails like gpg on a file starting "junk".
FAKE_GPG = """import sys
data = open(sys.argv[-1], encoding="utf-8").read()
if data.startswith("junk"):
    sys.stderr.write("gpg: no valid OpenPGP data found.\\n")
    sys.exit(2)
sys.stdout.write(data)
"""


def listing(*keys):
    """`--with-colons` text for primary keys, each (fingerprint, [subkey fingerprints])."""
    lines = []
    for primary, subkeys in keys:
        lines += [f"pub:-:3072:1:{primary[-16:]}:1544460984:::-:::scESC::::::23::0:",
                  f"fpr:::::::::{primary}:",
                  "uid:-::::1544461093::0000::Example <example@example.org>::::::::::0:"]
        for sub in subkeys:
            lines += [f"sub:-:3072:1:{sub[-16:]}:1544460984::::::e::::::23:", f"fpr:::::::::{sub}:"]
    return "\n".join(lines) + "\n"


class PrimaryFingerprints(unittest.TestCase):
    def test_one_key_with_subkeys(self):
        sub = "9CDD4BB7ED8C37C95FFF34F1DDB25B3BE89767D6"
        self.assertEqual(check_apt_key.primary_fingerprints(listing((WINEHQ, [sub]))), [WINEHQ])

    def test_two_primary_keys(self):
        self.assertEqual(check_apt_key.primary_fingerprints(listing((WINEHQ, []), (LLVM, []))), [WINEHQ, LLVM])

    def test_nothing(self):
        self.assertEqual(check_apt_key.primary_fingerprints(""), [])

    def test_normalize(self):
        self.assertEqual(check_apt_key.normalize("d43f 6401 4536 9c51 d786  ddea 76f1 a20f f987 672f"), WINEHQ)
        for bad in ("", "D43F", WINEHQ + "00", "G" * 40):
            self.assertIsNone(check_apt_key.normalize(bad), bad)


class CheckWithStandInGpg(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        fake = os.path.join(self.dir, "fake_gpg.py")
        with open(fake, "w", encoding="utf-8", newline="\n") as f:
            f.write(FAKE_GPG)
        self.gpg = (sys.executable, "-I", fake)

    def key(self, text):
        path = os.path.join(self.dir, f"key{len(os.listdir(self.dir))}.asc")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def test_the_pinned_key_passes(self):
        self.assertIsNone(check_apt_key.check(self.key(listing((WINEHQ, ["9CDD4BB7ED8C37C95FFF34F1DDB25B3BE89767D6"]))),
                                              WINEHQ, self.gpg))

    def test_another_key_fails(self):
        problem = check_apt_key.check(self.key(listing((LLVM, []))), WINEHQ, self.gpg)
        self.assertIn(f"is key {LLVM}, expected {WINEHQ}", problem)

    def test_a_second_primary_key_fails(self):
        problem = check_apt_key.check(self.key(listing((WINEHQ, []), (LLVM, []))), WINEHQ, self.gpg)
        self.assertIn("holds 2 primary keys", problem)

    def test_no_key_fails(self):
        self.assertIn("holds no public key", check_apt_key.check(self.key(""), WINEHQ, self.gpg))

    def test_unreadable_file_fails(self):
        problem = check_apt_key.check(self.key("junk"), WINEHQ, self.gpg)
        self.assertIn("could not read", problem)
        self.assertIn("no valid OpenPGP data", problem)

    def test_missing_gpg_fails(self):
        problem = check_apt_key.check(self.key(listing((WINEHQ, []))), WINEHQ,
                                      (os.path.join(self.dir, "no-such-gpg"),))
        self.assertIsNotNone(problem)


class CommandLine(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, "-I", SCRIPT, *args], capture_output=True, text=True)

    def test_usage_errors(self):
        self.assertEqual(self.run_script().returncode, 2)
        self.assertEqual(self.run_script(__file__, "not-a-fingerprint").returncode, 2)

    def test_missing_file(self):
        result = self.run_script(os.path.join(ROOT, "no-such.key"), WINEHQ)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such key file", result.stderr)


class Workflows(unittest.TestCase):
    """Every apt key a workflow downloads is checked by check_apt_key.py before it is installed."""

    KEY_URL = re.compile(r"https://\S+?\.(?:key|gpg|asc)\b")

    def test_every_fetched_key_is_checked_first(self):
        folder = os.path.join(ROOT, ".github", "workflows")
        fetched = 0
        for name in sorted(os.listdir(folder)):
            if not name.endswith((".yml", ".yaml")):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                lines = f.read().splitlines()
            for n, line in enumerate(lines):
                if not self.KEY_URL.search(line) or not re.search(r"\b(curl|wget)\b", line):
                    continue
                fetched += 1
                where = f"{name}:{n + 1}"
                with self.subTest(where=where):
                    self.assertNotIn("|", line.split("#")[0], f"{where} pipes a key straight into a command")
                    self.assertNotRegex(line, r"/etc/apt|/usr/share/keyrings", f"{where} downloads into apt's keyrings")
                    following = "\n".join(lines[n + 1:n + 4])
                    self.assertRegex(following, r"scripts/check_apt_key\.py \"?\$\w+\"? [0-9A-F]{40}\b",
                                     f"{where}: the key isn't checked by scripts/check_apt_key.py right after")
        self.assertGreaterEqual(fetched, 2, "expected the WineHQ and apt.llvm.org key downloads")


if __name__ == "__main__":
    unittest.main()
