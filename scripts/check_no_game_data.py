"""Fail when a tracked file starts like game data, whatever its name.

    python3 -I scripts/check_no_game_data.py            # every file `git ls-files` lists
    python3 -I scripts/check_no_game_data.py PATH...    # only these files

The extension check in .github/workflows/no-game-data.yml misses a renamed file: a DDS saved as .bin, an executable
as .txt. This reads the first bytes of each file and matches the magic numbers of the package's formats and the
files the build extracts from it: RPF9 archives, TTPF texture pools, RAGE resources, XBCH/FSSM/DXBC shader
containers, SPIR-V, DDS, AWC audio, PSIN/PAMH/MHIU/SPS1/SGD5/RDS5 containers, YSC/SCRP compiled scripts, the
animation MoVE container and SKEL_ bone table, Ogg, RIFF WAVE, Bink, glTF binaries, PE executables and XVD
images. The magics are
the ones docs/ measures (docs/gamedata.md and the shader and audio notes). Text formats (disassembly, .obj, .spvasm)
have no magic and stay with the extension check.

A compressed file hides those magics from both checks, so the script looks inside what the standard library opens:
each member of a zip (found by its central directory too, so with bytes in front of it) or a tar, and the stream of
a gzip, bzip2 or xz file, nested up to three levels, is matched the same way. What's inside is read into memory
under a budget per checked file (64 MiB expanded, 10,000 members), so a zip bomb can't make the check run long.
7-Zip, Zstandard and RAR files, which it can't open, archives that fail to open and archives over the budget are
reported as archives that can't be checked.

Prints one line per match, `<path>: <format>` (`<format> in <archive> (<member>)` for a match inside an archive), and
exits 1 if there is any. Standard library only, no game files.
"""
import bz2
import gzip
import io
import lzma
import os
import subprocess
import sys
import tarfile
import zipfile
import zlib

HEAD = 0x208  # enough for the XVD magic at 0x200

# (format, offset, magic). Matched at the start of the file only, so the same bytes inside text don't count.
MAGICS = (
    ("RPF9 archive", 0, b"9FPR"),
    ("RPF9 archive", 0, b"RPF9"),
    ("TTPF texture pool", 0, b"TTPF"),
    ("RAGE resource", 0, b"RSC7"),
    ("RAGE resource", 0, b"RSC8"),
    ("RAGE resource", 0, b"RSCC"),
    ("RAGE page directory", 0, b"0DRP"),
    ("RBF0 tree", 0, b"RBF0"),
    ("XBCH shader container", 0, b"XBCH"),
    ("FSSM shader", 0, b"FSSM"),
    ("DXBC container", 0, b"DXBC"),
    ("SPIR-V module", 0, b"\x03\x02\x23\x07"),
    ("SPIR-V module", 0, b"\x07\x23\x02\x03"),
    ("DDS texture", 0, b"DDS "),
    ("AWC audio bank", 0, b"ADAT"),
    ("AWC audio bank", 0, b"TADA"),
    ("AWC audio bank", 0, b"AWC\x00"),
    ("PSIN metadata container", 0, b"PSIN"),
    ("PAMH heightmap", 0, b"PAMH"),
    ("MHIU heightmap", 0, b"MHIU"),
    ("SPS1 shader-preset table", 0, b"SPS1"),
    ("SGD5 effects database", 0, b"SGD5"),
    ("RDS5 texture container", 0, b"RDS5"),
    ("YSC compiled script", 0, b"YSC\x00"),
    ("MoVE animation container", 0, b"MoVE"),
    ("SKEL_ bone table", 0, b"SKEL_"),
    ("SCRP compiled script", 0, b"SCRP"),
    ("Ogg stream", 0, b"OggS"),
    ("Bink 2 video", 0, b"KB2"),
    ("Bink video", 0, b"BIK"),
    ("glTF binary", 0, b"glTF"),
    ("XVD image", 0x200, b"msft-xvd"),
)


def classify(head, pe_signature=None):
    """The format a file starts like, or None. `head` is its first bytes; `pe_signature` the 4 bytes at e_lfanew."""
    for name, offset, magic in MAGICS:
        if head[offset:offset + len(magic)] == magic:
            return name
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "RIFF WAVE audio"
    if head.startswith(b"MZ") and pe_signature == b"PE\0\0":
        return "PE executable"
    return None


def read_head_from(f):
    """(first bytes, the 4 bytes at e_lfanew or None) of the open binary file `f`. The PE signature is read where
    e_lfanew points, even past the first bytes or straddling their end; a file too short for it gives fewer than 4
    bytes and no match."""
    head = f.read(HEAD)
    if not head.startswith(b"MZ") or len(head) < 0x40:
        return head, None
    offset = int.from_bytes(head[0x3C:0x40], "little")
    if offset + 4 <= len(head):
        return head, head[offset:offset + 4]
    f.seek(offset)
    return head, f.read(4)


def read_head(path):
    """(first bytes, the 4 bytes at e_lfanew or None). The PE signature is read where e_lfanew points, even past
    the first bytes or straddling their end; a file too short for it gives fewer than 4 bytes and no match."""
    with open(path, "rb") as f:
        return read_head_from(f)


ZIP = b"PK\x03\x04"
# Single compressed streams the standard library opens: (name, magic, opener of a binary file object). GzipFile
# gets its mode explicitly: it reads the file object's own `mode`, which for a GzipFile inside it isn't a string.
STREAMS = (
    ("gzip stream", b"\x1f\x8b", lambda f: gzip.GzipFile(fileobj=f, mode="rb")),
    ("bzip2 stream", b"BZh", lambda f: bz2.BZ2File(f)),
    ("xz stream", b"\xfd7zXZ\x00", lambda f: lzma.LZMAFile(f)),
)
# Archives the standard library can't open, so what they hold can't be checked.
OPAQUE = (
    ("7-Zip archive", b"7z\xbc\xaf\x27\x1c"),
    ("Zstandard stream", b"\x28\xb5\x2f\xfd"),
    ("RAR archive", b"Rar!\x1a\x07"),
)
MAX_DEPTH = 3  # an archive inside an archive inside an archive; deeper ones are reported as unchecked
# What looking into one file may cost: the bytes decompressed or copied out of its archives in all, and the archive
# members listed. Everything inside an archive is read into memory under this budget before it is looked at, so a
# zip bomb, or a zip inside a gzip (whose central directory sits at the end of the expanded stream), can't make the
# check run long or grow; past either limit the file is reported as too large to check, not passed.
MAX_EXPANDED = 64 << 20
MAX_MEMBERS = 10000
CHUNK = 1 << 20
TAR_MAGIC = (257, b"ustar")  # POSIX and GNU tar headers
# What a damaged archive raises while it is opened or read.
ARCHIVE_ERRORS = (OSError, EOFError, ValueError, zipfile.BadZipFile, tarfile.TarError, lzma.LZMAError, zlib.error,
                  NotImplementedError, RuntimeError)


class TooLarge(Exception):
    """Looking into an archive would take more than its file's budget."""


class Budget:
    """The bytes and members left for one checked file."""

    def __init__(self):
        self.bytes = MAX_EXPANDED
        self.members = MAX_MEMBERS

    def spool(self, f):
        """The rest of `f` read into memory, counted against the byte budget."""
        out = io.BytesIO()
        while True:
            chunk = f.read(min(CHUNK, self.bytes + 1))
            if not chunk:
                break
            self.bytes -= len(chunk)
            if self.bytes < 0:
                raise TooLarge
            out.write(chunk)
        out.seek(0)
        return out

    def member(self):
        self.members -= 1
        if self.members < 0:
            raise TooLarge


def printable(name):
    """A member name on one output line: control characters (a newline in a name, say) shown as '?'."""
    return "".join(c if c.isprintable() else "?" for c in name)


def archive_kind(f, head):
    """The kind of archive the seekable file `f` (first bytes `head`) is, or None."""
    stream = next((s for s in STREAMS if head.startswith(s[1])), None)
    if stream:
        return stream[0]
    if head[TAR_MAGIC[0]:TAR_MAGIC[0] + len(TAR_MAGIC[1])] == TAR_MAGIC[1]:
        return "tar archive"
    # is_zipfile finds the end-of-central-directory record, so a zip with bytes in front of it (a self-extracting
    # archive, a spanned zip starting PK\x07\x08) is found as zipfile itself would open it.
    if head.startswith(ZIP) or zipfile.is_zipfile(f):
        return "zip archive"
    return None


def classify_open(f, first, depth=0, budget=None):
    """The format of the seekable file `f`, whose read_head_from() is `first`: a game-data format, the game-data
    format of the first match inside it if it's an archive, an archive that can't be checked, or None. Members and
    streams inside are read into memory under `budget` (one per checked file) before they are looked at."""
    budget = budget or Budget()
    head = first[0]
    name = classify(*first)
    if name:
        return name
    for kind, magic in OPAQUE:
        if head.startswith(magic):
            return f"{kind} (contents can't be checked)"
    kind = archive_kind(f, head)
    if not kind:
        return None
    if depth >= MAX_DEPTH:
        return f"{kind} nested more than {MAX_DEPTH} deep (contents can't be checked)"

    def look_into(member):
        inner = budget.spool(member)
        return classify_open(inner, read_head_from(inner), depth + 1, budget)

    try:
        f.seek(0)
        if kind == "zip archive":
            with zipfile.ZipFile(f) as archive:
                for info in archive.infolist():
                    budget.member()
                    if info.is_dir():
                        continue
                    with archive.open(info) as member:
                        inner = look_into(member)
                    if inner:
                        return f"{inner} in {kind} ({printable(info.filename)})"
            return None
        if kind == "tar archive":
            with tarfile.open(fileobj=f, mode="r:") as archive:
                for info in archive:
                    budget.member()
                    if not info.isfile():
                        continue
                    with archive.extractfile(info) as member:
                        inner = look_into(member)
                    if inner:
                        return f"{inner} in {kind} ({printable(info.name)})"
            return None
        opener = next(s[2] for s in STREAMS if s[0] == kind)
        with opener(f) as member:
            inner = look_into(member)
        return f"{inner} in {kind}" if inner else None
    except TooLarge:
        return (f"{kind} too large to check (over {MAX_EXPANDED >> 20} MiB expanded or {MAX_MEMBERS} members; "
                "contents can't be checked)")
    except ARCHIVE_ERRORS:
        return f"{kind} that can't be opened (contents can't be checked)"


def classify_file(path):
    """The format the file at `path` starts like (looking inside archives), or None."""
    with open(path, "rb") as f:
        return classify_open(f, read_head_from(f))


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [os.path.join(root, p) for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def check(paths):
    """[(path, format)] for every regular file in `paths` that starts like game data."""
    found = []
    for path in paths:
        if os.path.islink(path) or not os.path.isfile(path):
            continue  # submodules, symlinks, files deleted in the working tree
        name = classify_file(path)
        if name:
            found.append((path, name))
    return found


def display(path):
    try:
        path = os.path.relpath(path)
    except ValueError:  # Windows: another drive than the working directory
        pass
    return path.replace("\n", "\\n").replace("\r", "\\r")  # one line per file, even for a name with a newline


def main(argv):
    paths = argv or tracked_files(os.getcwd())
    found = check(paths)
    for path, name in found:
        print(f"{display(path)}: {name}")
    if found:
        print(f"{len(found)} file(s) start like game data. Remove them; see CONTRIBUTING.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv[1:]))
