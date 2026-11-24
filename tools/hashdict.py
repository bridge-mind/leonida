"""Build a joaat (Jenkins one-at-a-time, lowercased) hash -> string dictionary from every string we can find.

Sources: printable runs in carved blobs (selected archives fully; PSO/RBF blobs everywhere), shader names, and
any extra text/binary files given (e.g. the exe once it arrives: run with --extra path\\to\\exe).
A carved stream that inflates to a whole number of 64 KiB blocks continues in the next one (docs/page_directory.md),
so a blob is harvested together with its continuation streams: a name across the boundary stays whole, and a PSO or
RBF blob in an archive that is not harvested fully is read past its first stream.
When multiple names share a hash, the dictionary keeps the lexicographically first spelling.
The output cannot overwrite an input index, extra file, or harvested blob (including hard-link aliases).
An index.json saved with a UTF-8 BOM (Windows PowerShell 5.1, old Notepad) is read, not refused.

Usage:
    python -I hashdict.py <carve dir> --out hashes.json [--archives a,b] [--extra f1 f2 ...] [--min 3]
      <carve dir> holds one folder per archive (<work>/carve), or is one archive's folder (<work>/carve/common).
    python -I hashdict.py --lookup <hashes.json> 0x28cca8df ...
"""
import argparse
import json
import os
import re
import sys

# A whole printable run, however long (one linear scan: a single character class, no nested quantifiers).
STR = re.compile(rb"[A-Za-z_][A-Za-z0-9_./:\\-]{2,}")
# For --min below 3 the run matcher must also accept one- and two-character identifiers, so the same
# identifier syntax (first byte [A-Za-z_], the rest of the run [A-Za-z0-9_./:\\-]) is used with no minimum tail.
SHORT_STR = re.compile(rb"[A-Za-z_][A-Za-z0-9_./:\\-]*")
# A name (a whole run or one piece of it) longer than this is not added: it is a blob of text, not a name.
# The old pattern cut every run at 201 characters, so no entry was ever longer than this.
MAX_NAME_LEN = 201
# The pieces of a path or dotted name, found one at a time (a run can be megabytes: no list of all its parts).
PART = re.compile(r"[^./:\\]+")
# Where re.split would yield an empty piece; only matters for --min 0 or less, which keeps "" as a name.
EMPTY_PART = re.compile(r"[./:\\]{2}|[./:\\]$")
# Carved stream heads (hex, as index.json has them) of the PSO and RBF blobs harvested in every archive.
META_HEADS = ("5053494e", "52424630")                # PSIN, RBF0
# Heads that start a stream of their own, never the continuation of the stream before (an XML declaration with or
# without a UTF-8 BOM, besides the blob heads above).
NEW_STREAM_HEADS = META_HEADS + ("3c3f786d6c", "efbbbf3c3f786d6c")
# A stream whose inflated size is a non-zero multiple of this continues in the next one (docs/page_directory.md).
CONTINUATION_BLOCK = 65536


def joaat(s: str) -> int:
    h = 0
    for c in s.lower().encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def harvest(data: bytes, out: set, minlen: int):
    run = STR if minlen >= 3 else SHORT_STR
    for m in run.finditer(data):
        s = m.group(0).decode("latin-1")
        if len(s) >= minlen:
            if len(s) <= MAX_NAME_LEN:
                out.add(s)
            # also the pieces of paths and dotted names, which are hashed on their own
            for pm in PART.finditer(s):
                if minlen <= pm.end() - pm.start() <= MAX_NAME_LEN:
                    out.add(pm.group(0))
            if minlen <= 0 and EMPTY_PART.search(s):
                out.add("")


def _raise_scan_error(error):
    raise error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carve", nargs="?")
    ap.add_argument("--out", default="hashes.json")
    ap.add_argument("--archives", default="common,data,script,pack_story000_story000_script,textures")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--lookup", nargs="+", metavar="VALUE", help="dictionary path followed by one or more hashes")
    a = ap.parse_args()
    if a.lookup:
        if len(a.lookup) < 2:
            ap.error("--lookup requires a dictionary and at least one hash")
        values = []
        for h in a.lookup[1:]:
            try:
                values.append((h, int(h, 16)))
            except ValueError:
                ap.error(f"--lookup: {h!r} is not a hexadecimal hash")
        try:
            with open(a.lookup[0], encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"ERROR: cannot read the dictionary {a.lookup[0]}: {getattr(e, 'strerror', None) or e}",
                  file=sys.stderr)
            return 2
        if not isinstance(d, dict):
            print(f"ERROR: the dictionary {a.lookup[0]} is not a JSON object of hash -> name", file=sys.stderr)
            return 2
        for h, v in values:
            print(h, d.get(f"{v:08x}", "?"))
        return 0
    if a.carve is None:
        ap.error("a carve directory is required to build a dictionary")
    if not os.path.isdir(a.carve):
        what = "is not a folder" if os.path.exists(a.carve) else "does not exist"
        print(f"ERROR: the carve folder {a.carve} {what}", file=sys.stderr)
        return 2
    for extra in a.extra:
        if not os.path.exists(extra):
            print(f"ERROR: --extra input does not exist: {extra}", file=sys.stderr)
            return 2
    # A folder with its own index.json is one archive's carve (<work>\carve\common, the folder the gamedata_*
    # tools take): harvest it as that archive. Otherwise each subfolder with an index.json is one archive.
    if os.path.isfile(os.path.join(a.carve, "index.json")):
        archives = [(os.path.basename(os.path.normpath(os.path.abspath(a.carve))), a.carve)]
    else:
        archives = [(d, os.path.join(a.carve, d)) for d in sorted(os.listdir(a.carve))]
    # The dictionary is written after the harvest: an --out naming one of the inputs would replace it.
    # samefile also catches a case-only difference on Windows and macOS's default volume.
    if os.path.exists(a.out):
        inputs = [x for x in a.extra if os.path.isfile(x)]
        inputs += [os.path.join(d, "index.json") for _, d in archives]
        for x in inputs:
            try:
                same = os.path.isfile(x) and os.path.samefile(a.out, x)
            except OSError:
                same = False
            if same:
                print(f"ERROR: --out {a.out} is one of the inputs ({x}); choose another output file",
                      file=sys.stderr)
                return 2
    # A report can also alias a harvested <offset>.bin, including through a hard link.
    # Snapshot its file identity once; compare against the input's already-open handle
    # rather than statting every carve path (there may be millions of entries).
    try:
        output_stat = os.stat(a.out)
    except OSError:
        output_stat = None
    strings = set()
    full = set(a.archives.split(","))
    missing = []                    # carved blobs an index.json lists that can't be read: skipped, reported
    for arch, arch_dir in archives:
        p = os.path.join(arch_dir, "index.json")
        if not os.path.isfile(p):
            continue
        try:
            # utf-8-sig: PowerShell 5.1 and old Notepad write JSON with a UTF-8 BOM, as the other readers here accept.
            with open(p, encoding="utf-8-sig") as f:
                rows = json.load(f)["files"]
            if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
                raise ValueError('"files" is not a list of entries')
        except (OSError, ValueError, KeyError, TypeError) as e:
            reason = getattr(e, "strerror", None) or (f"no {e} key" if isinstance(e, KeyError) else e)
            print(f"ERROR: cannot read {p}: {reason}", file=sys.stderr)
            return 2
        chain = []          # the streams of the blob being read: its first stream and those that continue it
        for row_idx, r in enumerate(rows):
            if r.get("type") == "bink2":
                harvest(b"".join(chain), strings, a.min)
                chain = []
                continue
            head = r.get("head")
            off = r.get("off")
            err = None
            if not isinstance(head, str):
                err = f"row {row_idx} has a non-string head"
            elif not isinstance(off, int) or isinstance(off, bool) or off < 0:
                err = f"row {row_idx} has an invalid offset"
            if err is not None:
                print(f"ERROR: cannot read {p}: {err}", file=sys.stderr)
                return 2
            # A row continues the blob before it when that stream filled whole 64 KiB blocks and this one does
            # not start a stream of its own; it is read with that blob, whatever its own head says.
            last = len(chain[-1]) if chain else 0
            if not (last and last % CONTINUATION_BLOCK == 0 and not head.startswith(NEW_STREAM_HEADS)):
                harvest(b"".join(chain), strings, a.min)
                chain = []
                if not (arch in full or head.startswith(META_HEADS)):
                    continue
            blob = os.path.join(arch_dir, f"{off:012x}.bin")
            try:
                with open(blob, "rb") as f:
                    if output_stat is not None and os.path.samestat(os.fstat(f.fileno()), output_stat):
                        print(f"ERROR: --out {a.out} is one of the inputs ({blob}); choose another output file",
                              file=sys.stderr)
                        return 2
                    data = f.read()
            except OSError:
                missing.append(blob)
                harvest(b"".join(chain), strings, a.min)
                chain = []
                continue
            chain.append(data)
        harvest(b"".join(chain), strings, a.min)
        print(f"{arch}: {len(strings)} strings so far", flush=True)
    if missing:
        print(f"WARNING: {len(missing)} carved blobs listed in an index.json could not be read and were skipped "
              f"(first: {missing[0]}). Re-run the carve to get their strings.", file=sys.stderr)
    for x in a.extra:
        if os.path.isdir(x):
            try:
                for _, _, files in os.walk(x, onerror=_raise_scan_error):
                    for fn in files:
                        strings.add(os.path.splitext(fn)[0])
            except OSError as e:
                print(f"ERROR: cannot scan --extra directory {e.filename or x}: {e.strerror or e}", file=sys.stderr)
                return 2
        elif os.path.isfile(x):
            with open(x, "rb") as f:
                harvest(f.read(), strings, a.min)
    d = {}
    for s in sorted(strings):
        d.setdefault(f"{joaat(s):08x}", s)
    # Other tools create the output parent too: a long harvest must not be lost to a missing folder.
    try:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f)
    except OSError as e:
        print(f"ERROR: cannot write {a.out}: {e}", file=sys.stderr)
        return 2
    print(f"{len(strings)} strings -> {len(d)} hashes -> {a.out}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
