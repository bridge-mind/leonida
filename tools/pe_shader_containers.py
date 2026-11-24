"""Inventory the DXBC shader containers embedded in a PE image, outside `shaders.rpf`.

The package ships shader programs in two places: the game's own containers inside `shaders.rpf`, which
`tools/shader_carve.py` recovers, and containers embedded in DLLs. This tool scans any binary for the latter,
validates each container's chunk table, reads the `ILDN` module name, and reports what each container carries:

* `has_xbch`: the `DXIL` chunk's `XDAT` tag 0x0d starts with `XBCH`. This is the console program:
  `tools/fssm_unpack.py` decompresses it to `FSSM` and the game's RDNA2 pipeline (disasm + `rdna2spirv.py`)
  translates it, exactly like a shader from `shaders.rpf`. `docs/embedded_shaders.md` has the measurement.
* `has_fspx`: the `ACCC` chunk starts with `FSPX`. That is the FidelityFX AOT compute program, a different
  format (`tools/fspx_unpack.py`, `docs/fspx.md`).
* `has_bitcode`: the `DXIL` chunk contains an LLVM bitcode magic (`BC C0 DE`). Some containers carry both the
  bitcode and the console `XBCH` record; the console runtime uses the latter.

Usage:
    python -I pe_shader_containers.py <image> [<image> ...] [--json FILE] [--extract DIR] [--verbose]

`--json` writes the per-container records (offset, size, sha256, chunk list, name, stage, flags). `--extract` writes
each container as `<module>.dxbc` plus an `index.json` whose entries the translator reads (with `stage` from
`PSV0`, defaulting to `"cs"` when absent), so the next step can be `fssm_unpack.py <DIR>`: nothing is extracted unless asked for, and extracted shaders are game
data that must stay in the user's work folder, never in the repository. Nothing here prints shader bytes.

Standard library only; reads files, never runs them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

DXBC = b"DXBC"
XDAT = b"XDAT"
XBCH = b"XBCH"
FSPX = b"FSPX"
BITCODE = b"BC\xc0\xde"
MAX_CHUNKS = 64  # a container with more chunks is not one this parser accepts
NAME_LIMIT = 4096  # longer than any ILDN path seen; stops a malformed length from reading the image
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class ContainerError(ValueError):
    """The bytes at this offset do not form a DXBC container this parser accepts."""


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def parse_container(data: bytes, offset: int):
    """Return (total_size, chunks) for the container at `offset`, or raise ContainerError.

    The package's containers differ from DXC's in one place only: the chunk table is a u32 offset array at
    +32 (count at +28), and each chunk starts with its fourcc and a u32 size. `tools/fspx_unpack.parts()`
    reads the same shape. The checks below reject the four-byte `DXBC` immediates and error strings that
    occur in ordinary code.
    """
    if data[offset : offset + 4] != DXBC:
        raise ContainerError("no DXBC magic")
    if offset + 32 > len(data):
        raise ContainerError("truncated header")
    total, count = u32(data, offset + 24), u32(data, offset + 28)
    if not 32 < total <= len(data) - offset:
        raise ContainerError(f"total size {total} does not fit the image")
    if not 0 < count <= MAX_CHUNKS:
        raise ContainerError(f"implausible chunk count {count}")
    table_end = 32 + 4 * count
    if table_end > total:
        raise ContainerError("chunk table past the container")
    chunks = []
    for k in range(count):
        start = u32(data, offset + 32 + 4 * k)
        if not table_end <= start <= total - 8:
            raise ContainerError(f"chunk {k} offset {start} outside the container")
        fourcc = data[offset + start : offset + start + 4]
        size = u32(data, offset + start + 4)
        if start + 8 + size > total:
            raise ContainerError(f"chunk {k} payload past the container")
        if start % 4:
            raise ContainerError(f"chunk {k} offset {start} not 4-byte aligned")
        if not all(0x20 <= b < 0x7F for b in fourcc):
            raise ContainerError(f"chunk {k} fourcc is not printable")
        chunks.append((fourcc, start, size))
    starts = [start for _, start, _ in chunks]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ContainerError("chunk table is not sorted and unique")
    if not chunks or max(start + 8 + size for _, start, size in chunks) != total:
        raise ContainerError("the last chunk does not end at the container size")
    return total, chunks


def chunk_payload(data: bytes, offset: int, chunk) -> bytes:
    _, start, size = chunk
    return data[offset + start + 8 : offset + start + 8 + size]


def module_name(ildn: bytes) -> str | None:
    """The module name in an `ILDN` part: `00 00 <u16 length> <PDB path>`; basename without `.pdb`."""
    if len(ildn) < 4:
        return None
    length = struct.unpack_from("<H", ildn, 2)[0]
    path = ildn[4 : 4 + min(length, NAME_LIMIT)].split(b"\0", 1)[0]
    base = path.replace(b"\\", b"/").rsplit(b"/", 1)[-1].decode("latin-1")
    return base[:-4] if base.lower().endswith(".pdb") else base or None


PSV0_STAGES = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs", 13: "ms", 14: "as"}


def psv0_stage(psv0: bytes) -> str | None:
    """Return the shader stage from a `PSV0` chunk payload, or None if short/unknown."""
    if len(psv0) < 29:
        return None
    info_size = u32(psv0, 0)
    if info_size < 25 or 4 + 24 >= len(psv0):
        return None
    return PSV0_STAGES.get(psv0[4 + 24])


def xdat_tags(dxil: bytes) -> list[dict]:
    """The XDAT entries of a `DXIL` part, as {tag, size, magic} with the first four payload bytes as ASCII."""
    start = dxil.find(XDAT)
    if start < 0 or start + 12 > len(dxil):
        return []
    count, _header = struct.unpack_from("<II", dxil, start + 4)
    if count > 256:
        return []
    out = []
    for k in range(count):
        at = start + 12 + 12 * k
        if at + 12 > len(dxil):
            break
        tag, offset, size = struct.unpack_from("<III", dxil, at)
        if offset + size > len(dxil):
            continue
        magic = dxil[offset : offset + 4]
        out.append(
            {
                "tag": tag,
                "size": size,
                "magic": magic.decode("latin-1")
                if all(0x20 <= b < 0x7F for b in magic)
                else "",
            }
        )
    return out


def inspect(data: bytes, image: str) -> tuple[list[dict], list[int]]:
    """Every valid container in `data`, and the offsets of `DXBC` hits that failed validation."""
    found, rejected = [], []
    i = 0
    while True:
        i = data.find(DXBC, i)
        if i < 0:
            break
        try:
            total, chunks = parse_container(data, i)
        except ContainerError:
            rejected.append(i)
            i += 4
            continue
        parts = {fourcc: (start, size) for fourcc, start, size in chunks}
        dialect = b""
        if b"DXIL" in parts:
            dialect = chunk_payload(data, i, (b"DXIL",) + parts[b"DXIL"])
        elif b"ACCC" in parts:
            dialect = chunk_payload(data, i, (b"ACCC",) + parts[b"ACCC"])
        tags = xdat_tags(dialect) if b"DXIL" in parts else []
        stage = (
            psv0_stage(chunk_payload(data, i, (b"PSV0",) + parts[b"PSV0"]))
            if b"PSV0" in parts
            else None
        )
        record = {
            "image": image,
            "offset": i,
            "size": total,
            "sha256": hashlib.sha256(data[i : i + total]).hexdigest(),
            "name": module_name(chunk_payload(data, i, (b"ILDN",) + parts[b"ILDN"]))
            if b"ILDN" in parts
            else None,
            "stage": stage,
            "chunks": [
                {"fourcc": fourcc.decode("latin-1"), "size": size}
                for fourcc, _, size in chunks
            ],
            "xdat_tags": tags,
            "has_xbch": any(t["tag"] == 0x0D and t["magic"] == "XBCH" for t in tags),
            "has_fspx": (b"ACCC" in parts and dialect[:4] == FSPX),
            "has_bitcode": BITCODE in dialect,
        }
        found.append(record)
        i += total
    return found, rejected


def safe_stem(name: str | None, index: int) -> str:
    stem = SAFE_NAME.sub("_", name or "").strip("._")
    return stem or f"container_{index:02d}"


def extract(data: bytes, records: list[dict], directory: str) -> list[dict]:
    """Write each container as `<module>.dxbc`; return the translator's index entries."""
    os.makedirs(directory, exist_ok=True)
    index = []
    for k, record in enumerate(records):
        stem = safe_stem(record["name"], k)
        file = f"{stem}.dxbc"
        target = os.path.join(directory, file)
        n = k
        while True:
            try:
                with open(target, "xb") as f:
                    f.write(data[record["offset"] : record["offset"] + record["size"]])
                break
            except FileExistsError:
                file = f"{stem}_{n:02d}.dxbc"
                target = os.path.join(directory, file)
                n += 1
        index.append({"name": stem, "file": file, "stage": record.get("stage") or "cs"})
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory the DXBC shader containers embedded in PE images."
    )
    ap.add_argument("images", nargs="+", help="PE image(s) to scan")
    ap.add_argument("--json", help="write the per-container records here")
    ap.add_argument(
        "--extract",
        metavar="DIR",
        help="write each container as <module>.dxbc plus index.json (game data: keep it out of the repo)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="print one line per container"
    )
    a = ap.parse_args(argv)

    if a.extract and os.path.exists(a.extract) and not os.path.isdir(a.extract):
        ap.error(f"--extract {a.extract} is not a folder")
    if a.extract:
        index_path = os.path.join(a.extract, "index.json")
        for image in a.images:
            if same_report_path(index_path, image):
                ap.error(f"--extract index {index_path} would overwrite the input image {image}")
    if a.json:
        if os.path.isdir(a.json):
            ap.error(f"--json {a.json} is a folder, not a file")
        for image in a.images:
            if same_report_path(a.json, image):
                ap.error(f"--json {a.json} would overwrite the input image {image}")
        if a.extract and same_report_path(a.json, index_path):
            ap.error(f"--json {a.json} would overwrite the extract index")

    all_records, status, index = [], 0, []
    for image in a.images:
        try:
            with open(image, "rb") as f:
                data = f.read()
        except OSError as e:
            print(f"ERROR: cannot read {image}: {e.strerror or e}", file=sys.stderr)
            status = 2
            continue
        records, rejected = inspect(data, image)
        all_records.extend(records)
        print(
            f"{image}: {len(records)} container(s), {len(rejected)} rejected DXBC hit(s)"
        )
        if a.verbose:
            for r in records:
                name = r["name"] or "?"
                flags = "".join(
                    c
                    for c, on in (
                        ("X", r["has_xbch"]),
                        ("F", r["has_fspx"]),
                        ("B", r["has_bitcode"]),
                    )
                    if on
                )
                chunks = ",".join(c["fourcc"] for c in r["chunks"])
                print(
                    f"  {r['offset']:#x} {r['size']:>7} {flags:<3} {name:<28} {chunks}"
                )
        if a.extract:
            try:
                index.extend(extract(data, records, a.extract))
            except OSError as e:
                print(f"ERROR: cannot write {a.extract}: {e}", file=sys.stderr)
                return 2

    if a.extract:
        try:
            with open(index_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f, indent=0)
        except OSError as e:
            print(f"ERROR: cannot write {index_path}: {e}", file=sys.stderr)
            return 2
        print(f"  extracted {len(index)} file(s) to {a.extract}")
        print("  these are game data: do not commit them", file=sys.stderr)

    if a.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(all_records, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e}", file=sys.stderr)
            return 2
        print(f"wrote {a.json}")
    return status


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
