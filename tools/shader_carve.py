"""Carve shader containers out of an RPF9 archive without reading its (encrypted) table of contents.

Each shader in shaders.rpf is stored as:

    <name> 00  <u8 kind>  <u32 stored>  <u32 id>  <u32 stored>  <u32 size>  <raw deflate, `stored` bytes>

and inflates to a DXBC container (`size` bytes) holding DXIL plus an Xbox PRIV part.

Usage:
    python -I shader_carve.py <archive.rpf> <out_dir>

Writes <out_dir>/<name>.dxbc and <out_dir>/index.json (name, offset, stage, shader model, parts).
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib

# `python -I` leaves the script's directory off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replace_retry import replace as replace_retry  # noqa: E402

ENTRY = re.compile(rb"([\x21-\x7e]{3,200})\x00(.)(....)(....)\3(....)", re.DOTALL)
STAGES = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs", 6: "lib", 13: "ms", 14: "as"}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def parts(dxbc: bytes) -> dict:
    if len(dxbc) < 32:
        raise ValueError("truncated DXBC header")
    count = struct.unpack_from("<I", dxbc, 28)[0]
    table_end = 32 + 4 * count
    if table_end > len(dxbc):
        raise ValueError("truncated DXBC part table")
    out = {}
    for entry in range(32, table_end, 4):
        off = struct.unpack_from("<I", dxbc, entry)[0]
        if not table_end <= off <= len(dxbc) - 8:
            raise ValueError("DXBC part header outside the part data")
        fourcc, size = struct.unpack_from("<4sI", dxbc, off)
        if off + 8 + size > len(dxbc):
            raise ValueError("truncated DXBC part payload")
        out[fourcc.decode("latin-1")] = (off + 8, size)
    return out


def shader_model(dxbc: bytes, p: dict):
    if "DXIL" not in p:
        return None, None
    off, size = p["DXIL"]
    if size < 4:
        raise ValueError("truncated DXIL version and stage")
    ver, kind = struct.unpack_from("<HH", dxbc, off)
    return STAGES.get(kind, str(kind)), f"{ver >> 4}.{ver & 0xF}"


def file_name(name: str, seen: dict, used: set) -> str:
    """A file name for `name` that no earlier shader has, as Windows compares them: without case, and with the
    device names (CON, NUL, COM1...) that open a device instead of a file. Same name twice means distinct
    variants; the second gets ~1, the third ~2."""
    base = re.sub(r'[<>:"/\\|?*]', "_", name)
    if base.split(".")[0].upper() in RESERVED:
        base = "_" + base
    key = base.lower()
    n = seen.get(key, 0)
    while True:
        fname = base + (f"~{n}" if n else "") + ".dxbc"
        if fname.lower() not in used:
            break
        n += 1
    seen[key] = n + 1
    used.add(fname.lower())
    return fname


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Carve DXBC shader containers out of an RPF9 archive.")
    ap.add_argument("src", help="the shaders.rpf archive")
    ap.add_argument("out", help="folder for the .dxbc files and index.json")
    a = ap.parse_args(argv)
    src, out = a.src, a.out
    if not os.path.isfile(src):
        ap.error(f"{src} is not a file")
    if os.path.getsize(src) == 0:
        ap.error(f"{src} is empty")
    if os.path.isfile(out):
        ap.error(f"{out} is a file, not a folder")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:  # a regular file where a parent folder is expected
        ap.error(f"cannot create {out}: {e.strerror or e}")
    index, seen, used, bad = [], {}, set(), 0
    with open(src, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        pos = 0
        while (mt := ENTRY.search(m, pos)) is not None:
            # A rejected header is rescanned past its name (a real entry can start inside it); a carved
            # shader resumes after its stored bytes, so deflate output never becomes a name or a match.
            pos = mt.end(1) + 1
            name = mt.group(1).decode("latin-1")
            stored = struct.unpack("<I", mt.group(3))[0]
            size = struct.unpack("<I", mt.group(5))[0]
            if not (16 < stored < 64 << 20 and 32 < size < 256 << 20):
                continue
            start = mt.end()
            decompressor = zlib.decompressobj(-15)
            try:
                data = decompressor.decompress(m[start:start + stored], size + 1)
            except zlib.error:
                bad += 1
                continue
            if not decompressor.eof or data[:4] != b"DXBC" or len(data) != size:
                bad += 1
                continue
            try:
                p = parts(data)
                stage, sm = shader_model(data, p)
            except ValueError:
                bad += 1
                continue
            pos = start + stored
            fname = file_name(name, seen, used)
            with open(os.path.join(out, fname), "wb") as f:
                f.write(data)
            index.append({"name": name, "file": fname, "offset": mt.start(), "kind": mt.group(2)[0],
                          "stage": stage, "sm": sm, "size": size, "parts": {k: v[1] for k, v in p.items()}})
    if not index:
        print(f"ERROR: no shaders carved from {src} (rejected candidates: {bad}).", file=sys.stderr)
        return 1
    # make_game.py takes index.json as proof the carve finished: write it last, under another name, then rename.
    with tempfile.TemporaryDirectory(dir=out, prefix=".shader-index-") as staging:
        tmp = os.path.join(staging, "index.json")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(index, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        replace_retry(tmp, os.path.join(out, "index.json"))
    from collections import Counter
    print(f"{len(index)} shaders carved, {bad} candidates rejected")
    print("stages:", dict(Counter(e["stage"] for e in index)))
    print("shader models:", dict(Counter(e["sm"] for e in index)))
    print("parts:", dict(Counter(k for e in index for k in e["parts"])))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
