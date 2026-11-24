"""Check that a downloaded apt signing key is the one CI pins, before apt is told to trust it.

    python3 -I scripts/check_apt_key.py KEYFILE FINGERPRINT

The CI workflows fetch the WineHQ and apt.llvm.org signing keys over HTTPS and hand them to apt. Every other download
in CI is pinned (actions by commit, glslang by SHA-256, the Vulkan headers by commit); this pins the keys by the
fingerprint each project publishes. The key file must hold exactly one primary key, and its fingerprint must be
FINGERPRINT (40 hex digits; spaces and case don't matter). A second primary key appended to the file fails the check
too, since apt would trust it as well. Subkeys are allowed: they are bound to the primary key.

Reads the key with `gpg --show-keys --with-colons` (GnuPG 2.2.8+), which lists a key file without importing it, so no
keyring changes. Exits 0 when the key matches, 1 when it doesn't or gpg can't read it, 2 on a usage error. Standard
library only.
"""
import os
import re
import subprocess
import sys


def normalize(fingerprint):
    """`fingerprint` as 40 upper-case hex digits, or None if it isn't a v4 fingerprint."""
    digits = re.sub(r"\s+", "", fingerprint).upper()
    return digits if re.fullmatch(r"[0-9A-F]{40}", digits) else None


def primary_fingerprints(colons):
    """The fingerprint of each primary key in `gpg --with-colons` output: the `fpr` record right after a `pub` one."""
    out = []
    expect_primary_fpr = False
    for line in colons.splitlines():
        fields = line.split(":")
        if fields[0] == "pub":
            expect_primary_fpr = True
        elif fields[0] == "fpr" and expect_primary_fpr:
            out.append(fields[9].upper() if len(fields) > 9 else "")
            expect_primary_fpr = False
        elif fields[0] in ("sub", "ssb", "sec"):
            expect_primary_fpr = False
    return out


def show_keys(path, gpg=("gpg",)):
    """`gpg --show-keys --with-colons` output for the key file at `path` (it lists the file, importing nothing)."""
    result = subprocess.run([*gpg, "--batch", "--show-keys", "--with-colons", path],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise ValueError(f"gpg could not read {path}: {result.stderr.strip() or 'exit ' + str(result.returncode)}")
    return result.stdout


def check(path, expected, gpg=("gpg",)):
    """None if the key file at `path` holds exactly the one primary key `expected`, else why not. `gpg` is the command
    that runs GnuPG (a list, so a test can stand another program in)."""
    try:
        found = primary_fingerprints(show_keys(path, gpg))
    except (OSError, ValueError) as e:
        return str(e)
    if not found:
        return f"{path} holds no public key"
    if len(found) > 1:
        return f"{path} holds {len(found)} primary keys ({', '.join(found)}); expected only {expected}"
    if found[0] != expected:
        return f"{path} is key {found[0]}, expected {expected}"
    return None


def main(argv):
    if len(argv) != 2:
        print("usage: check_apt_key.py KEYFILE FINGERPRINT", file=sys.stderr)
        return 2
    path, fingerprint = argv
    expected = normalize(fingerprint)
    if not expected:
        print(f"not a 40-digit key fingerprint: {fingerprint!r}", file=sys.stderr)
        return 2
    if not os.path.isfile(path):
        print(f"no such key file: {path}", file=sys.stderr)
        return 1
    problem = check(path, expected)
    if problem:
        print(f"apt key check failed: {problem}", file=sys.stderr)
        return 1
    print(f"{path}: key {expected}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv[1:]))
