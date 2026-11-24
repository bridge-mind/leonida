"""scripts/check_no_game_data.py catches game-data magics under any file name.

    python -I tests/test_check_no_game_data.py

Every input is a few hand-written header bytes in a temporary directory (no game files): each format under a harmless
name, files that must pass (text, PNG, a lone "MZ", magics past the start), and the default mode over a throwaway git
repository. The script runs as itself, through its own command line.
"""
import bz2
import gzip
import io
import lzma
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import tarfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "check_no_game_data.py")


def pe_header(e_lfanew=0x80):
    head = bytearray(e_lfanew + 4)
    head[0:2] = b"MZ"
    head[0x3C:0x40] = e_lfanew.to_bytes(4, "little")
    head[e_lfanew:e_lfanew + 4] = b"PE\0\0"
    return bytes(head)


# file name (deliberately not the format's extension), contents, format the script must report
CAUGHT = [
    ("archive.bin", b"9FPR" + bytes(12), "RPF9 archive"),
    ("pool.dat", b"TTPF" + bytes(12), "TTPF texture pool"),
    ("resource.dat", b"RSC8" + bytes(12), "RAGE resource"),
    ("page.dat", b"0DRP" + bytes(12), "RAGE page directory"),
    ("tree.dat", b"RBF0" + bytes(12), "RBF0 tree"),
    ("shader.txt", b"XBCH" + (3).to_bytes(4, "little") + bytes(8), "XBCH shader container"),
    ("shader2.txt", b"FSSM" + bytes(12), "FSSM shader"),
    ("container.txt", b"DXBC" + bytes(12), "DXBC container"),
    ("module.txt", bytes.fromhex("03022307") + bytes(12), "SPIR-V module"),
    ("module_be.txt", bytes.fromhex("07230203") + bytes(12), "SPIR-V module"),
    ("texture.png.txt", b"DDS " + (124).to_bytes(4, "little") + bytes(8), "DDS texture"),
    ("bank.json", b"ADAT" + bytes(12), "AWC audio bank"),
    ("music.md", b"OggS\0\x02" + bytes(10), "Ogg stream"),
    ("sound.csv", b"RIFF" + bytes(4) + b"WAVEfmt ", "RIFF WAVE audio"),
    ("movie.txt", b"KB2j" + bytes(12), "Bink 2 video"),
    ("model.json", b"glTF" + (2).to_bytes(4, "little") + bytes(8), "glTF binary"),
    ("game.txt", pe_header(), "PE executable"),
    ("far_pe.txt", pe_header(0x400), "PE executable"),
    # e_lfanew around the end of the first bytes read (0x208): before, straddling and after it
    *((f"pe_{n}.txt", pe_header(n), "PE executable") for n in range(0x204, 0x20A)),
    ("pe_1mib.txt", pe_header(1 << 20), "PE executable"),
    ("pe_1mib_plus.txt", pe_header((1 << 20) + 4), "PE executable"),
    ("image.bin", bytes(0x200) + b"msft-xvd" + bytes(8), "XVD image"),
    ("container2.dat", b"PSIN" + bytes(12), "PSIN metadata container"),
    ("heightmap.dat", b"PAMH" + bytes(12), "PAMH heightmap"),
    ("heightmap2.dat", b"MHIU" + bytes(12), "MHIU heightmap"),
    ("presets.dat", b"SPS1" + bytes(12), "SPS1 shader-preset table"),
    ("effects.dat", b"SGD5" + bytes(12), "SGD5 effects database"),
    ("texcontainer.dat", b"RDS5" + bytes(12), "RDS5 texture container"),
    ("bank2.json", b"AWC\x00" + bytes(12), "AWC audio bank"),
    ("script.lol", b"YSC\x00" + bytes(12), "YSC compiled script"),
    ("anim.bin", b"MoVE" + (19).to_bytes(4, "little") + bytes(8), "MoVE animation container"),
    ("bones.dat", b"SKEL_" + b"SKEL_Head\x00" + bytes(8), "SKEL_ bone table"),
    ("script2.lol", b"SCRP" + bytes(12), "SCRP compiled script"),
]

PASSED = [
    ("README.md", b"# Notes\n\nThe RPF9 magic is stored as 9FPR; DDS files start with 'DDS '.\n"),
    ("logo.png", b"\x89PNG\r\n\x1a\n" + bytes(16)),
    ("mz.txt", b"MZ is not enough on its own\n"),
    ("metadata.txt", b"METADATA ADAT later in the line\n"),
    ("bones.md", b"# SKEL_HEAD, MoVE and SCRP name formats; none of them starts this file\n"),
    ("riff.bin", b"RIFF" + bytes(4) + b"AVI LIST"),
    ("empty.txt", b""),
    ("mz_past_end.txt", pe_header(0x80)[:0x40]),  # e_lfanew points past the end of the file
    ("mz_cut_signature.txt", pe_header(0x300)[:-2]),  # only "PE" of the signature is there
    ("mz_not_pe.txt", pe_header(0x80)[:-4] + b"NE\0\0"),
]


def rmtree(path):
    """shutil.rmtree that also removes read-only files (Git's objects on Windows)."""
    def retry(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def zipped(members):
    """A zip archive (bytes) holding `members`, {name: bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def tarred(members):
    """A tar archive (bytes) holding `members`, {name: bytes}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def corrupt_deflate(archive):
    """`archive` (one deflated zip member) with the first byte of its deflate data made an invalid block type."""
    data = bytearray(archive)
    name_len, extra_len = int.from_bytes(data[26:28], "little"), int.from_bytes(data[28:30], "little")
    data[30 + name_len + extra_len] = 0xFF  # BTYPE 11: zlib.error "invalid block type"
    return bytes(data)


DDS = b"DDS " + (124).to_bytes(4, "little") + bytes(8)
FSSM = b"FSSM" + bytes(12)
RPF = b"9FPR" + bytes(12)

# file name, contents, the line the script must print after "<name>: "
CAUGHT_IN_ARCHIVES = [
    ("notes.zip.txt", zipped({"readme.txt": b"text\n", "tex/a.png": DDS}), "DDS texture in zip archive (tex/a.png)"),
    ("data.bin", gzip.compress(FSSM), "FSSM shader in gzip stream"),
    ("tool.bin", bz2.compress(pe_header()), "PE executable in bzip2 stream"),
    ("pack.bin", lzma.compress(RPF), "RPF9 archive in xz stream"),
    ("far.bin", gzip.compress(pe_header(0x400)), "PE executable in gzip stream"),
    ("nested.bin", gzip.compress(zipped({"inner/x.bin": RPF})), "RPF9 archive in zip archive (inner/x.bin) in gzip stream"),
    ("a.bin", b"7z\xbc\xaf\x27\x1c" + bytes(26), "7-Zip archive (contents can't be checked)"),
    ("b.bin", b"\x28\xb5\x2f\xfd" + bytes(12), "Zstandard stream (contents can't be checked)"),
    ("c.bin", b"Rar!\x1a\x07\x01\x00" + bytes(8), "RAR archive (contents can't be checked)"),
    ("broken.bin", b"PK\x03\x04" + bytes(40), "zip archive that can't be opened (contents can't be checked)"),
    ("broken.gz", b"\x1f\x8b\x09\x00" + bytes(12), "gzip stream that can't be opened (contents can't be checked)"),
    # broken.gz names compression method 9, which isn't deflate; deep.bin is a DDS under four gzip layers
    ("corrupt.gz", gzip.compress(DDS)[:12] + b"\xff" * 12,
     "gzip stream that can't be opened (contents can't be checked)"),
    ("deep.bin", gzip.compress(gzip.compress(gzip.compress(gzip.compress(DDS)))),
     "gzip stream nested more than 3 deep (contents can't be checked) in gzip stream in gzip stream in gzip stream"),
    # Review of #4040: tar archives, zips with bytes in front, a corrupt deflate stream, a newline in a member name.
    ("x.tar", tarred({"x.dds": DDS}), "DDS texture in tar archive (x.dds)"),
    ("x.tar.gz", gzip.compress(tarred({"a.txt": b"text\n", "y.bin": FSSM})),
     "FSSM shader in tar archive (y.bin) in gzip stream"),
    ("x.tar.xz", lzma.compress(tarred({"z.bin": RPF})), "RPF9 archive in tar archive (z.bin) in xz stream"),
    ("prefixed.txt", b"JUNK" + zipped({"a.png": DDS}), "DDS texture in zip archive (a.png)"),
    ("sfx.exe.txt", b"#!/bin/sh\nexit 0\n" + zipped({"b.bin": RPF}), "RPF9 archive in zip archive (b.bin)"),
    ("nested_prefixed.zip", zipped({"inner.bin": bytes(8) + zipped({"x.bin": RPF})}),
     "RPF9 archive in zip archive (x.bin) in zip archive (inner.bin)"),
    ("bad_deflate.zip", corrupt_deflate(zipped({"a.bin": DDS * 64})),
     "zip archive that can't be opened (contents can't be checked)"),
    ("newline.zip", zipped({"a\nb.png": DDS}), "DDS texture in zip archive (a?b.png)"),
]

PASSED_ARCHIVES = [
    ("docs.zip", zipped({"a.txt": b"plain text\n", "dir/b.md": b"# notes\n", "empty.txt": b""})),
    ("docs.tar", tarred({"a.txt": b"plain text\n"})),
    ("docs.tar.bz2", bz2.compress(tarred({"a.txt": b"plain text\n"}))),
    ("prefixed_clean.bin", b"JUNK" + zipped({"a.txt": b"text\n"})),
    ("notes.gz", gzip.compress(b"The RPF9 magic is stored as 9FPR.\n")),
    ("notes.bz2", bz2.compress(b"plain text\n")),
    ("notes.xz", lzma.compress(b"plain text\n")),
    ("nested_clean.gz", gzip.compress(zipped({"a.txt": b"text\n"}))),
]


def run(args, cwd=None):
    return subprocess.run([sys.executable, "-I", SCRIPT, *args], cwd=cwd, capture_output=True, text=True)


class CheckNoGameData(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(rmtree, self.dir)

    def write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_each_format_is_caught_under_another_name(self):
        for name, data, fmt in CAUGHT:
            with self.subTest(name=name):
                result = run([self.write(name, data)])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{name}: {fmt}", result.stdout)

    def test_game_data_inside_archives_is_caught(self):
        for name, data, line in CAUGHT_IN_ARCHIVES:
            with self.subTest(name=name):
                result = run([self.write(name, data)])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{name}: {line}", result.stdout)

    def test_decompression_is_bounded(self):
        # A gzip of 65 MiB of zeros (about 64 KiB on disk) holding a zip: past the 64 MiB budget, reported, not read
        # to the end; and a zip with more members than the budget lists.
        bomb = io.BytesIO()
        with gzip.GzipFile(fileobj=bomb, mode="wb") as g:
            g.write(b"PK\x03\x04")
            chunk = bytes(1 << 20)
            for _ in range(65):
                g.write(chunk)
        many = io.BytesIO()
        with zipfile.ZipFile(many, "w", zipfile.ZIP_STORED) as z:
            for i in range(10001):
                z.writestr(f"{i}.txt", b"")
        for name, data, kind in [("bomb.gz", bomb.getvalue(), "gzip stream"), ("many.zip", many.getvalue(), "zip archive")]:
            with self.subTest(name=name):
                result = run([self.write(name, data)])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{name}: {kind} too large to check", result.stdout)

    def test_archives_of_ordinary_files_pass(self):
        paths = [self.write(name, data) for name, data in PASSED_ARCHIVES]
        result = run(paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_ordinary_files_pass(self):
        paths = [self.write(name, data) for name, data in PASSED]
        result = run(paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_reports_every_match_among_clean_files(self):
        clean = self.write("README.md", b"# fine\n")
        bad1 = self.write("a.bin", b"DDS " + bytes(12))
        bad2 = self.write("b.bin", b"OggS" + bytes(12))
        result = run([clean, bad1, bad2])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(result.stdout.splitlines()), 2, result.stdout)
        self.assertIn("2 file(s)", result.stderr)

    def test_directories_and_missing_paths_are_skipped(self):
        os.mkdir(os.path.join(self.dir, "sub"))
        result = run([os.path.join(self.dir, "sub"), os.path.join(self.dir, "missing.bin")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("git"), "git not installed")
    def test_default_mode_reads_tracked_files_only(self):
        git = ["git", "-c", "init.defaultBranch=main"]
        subprocess.run([*git, "init", "-q", self.dir], check=True)
        self.write("notes.txt", b"plain text\n")
        self.write("untracked.bin", b"DDS " + bytes(12))
        subprocess.run(["git", "add", "notes.txt"], cwd=self.dir, check=True)
        result = run([], cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        self.write("renamed.txt", b"FSSM" + bytes(12))
        subprocess.run(["git", "add", "renamed.txt"], cwd=self.dir, check=True)
        result = run([], cwd=self.dir)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("renamed.txt: FSSM shader", result.stdout)
        self.assertNotIn("untracked.bin", result.stdout)


if __name__ == "__main__":
    unittest.main()
