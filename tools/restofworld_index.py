"""Index the stream signatures in the rest-of-world and script RPF9 archives.

The RPF9 table of contents is encrypted, so this walks the data area the same way
`rpf9_carve.py` does: find every zlib stream that inflates cleanly to its end, plus
validated Bink 2 streams, and classify what starts each stream. It reports the
measured signatures from `docs/restofworld.md` and writes one row per stream. It
does not decode resources, write payload bytes, or touch the table of contents.

Signatures are only accepted at a stream start, except `0DRP`, which is accepted at
+0 or +0x10 because the PSIN LRSC payload and a 0x7a metadata resource use both
placements. A four-byte magic inside an inflated page or XML text is not a stream
signature; that distinction matters because `ADAT` occurs inside `METADATA`, and
`AWC` occurs inside printable names.

Usage:
    python -I restofworld_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES]
    python -I restofworld_index.py <carved-dir> <out.csv|out.json>
"""
import argparse
import array
from collections import Counter
import csv
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zero_span import gap_split  # noqa: E402

# zlib stream start, or Bink 2 video ('KB2' + revision letter). Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
PSIN_TAGS = (b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS")
COLUMNS = ["offset", "stored", "size", "signature", "version", "count", "detail"]
MAX_INFLATE = 512 << 20
MAX_TEX_SCAN = 16 << 20
_UINT64_CODE = "Q" if array.array("Q").itemsize == 8 else None

# DXGI format codes observed in RAGE9 0x980 texture descriptors (kept in step with tools/tex_headers.py).
DXGI_FORMATS = {
    2: "R32G32B32A32_FLOAT", 3: "R32G32B32A32_UINT", 4: "R32G32B32A32_SINT",
    6: "R32G32B32_FLOAT", 7: "R32G32B32_UINT", 8: "R32G32B32_SINT",
    10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM", 12: "R16G16B16A16_UINT",
    13: "R16G16B16A16_SNORM", 14: "R16G16B16A16_SINT",
    16: "R32G32_FLOAT", 17: "R32G32_UINT", 18: "R32G32_SINT",
    24: "R10G10B10A2_UNORM", 25: "R10G10B10A2_UINT", 26: "R11G11B10_FLOAT",
    28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB", 30: "R8G8B8A8_UINT",
    31: "R8G8B8A8_SNORM", 32: "R8G8B8A8_SINT",
    34: "R16G16_FLOAT", 35: "R16G16_UNORM", 36: "R16G16_UINT",
    37: "R16G16_SNORM", 38: "R16G16_SINT",
    41: "R32_FLOAT", 42: "R32_UINT", 43: "R32_SINT",
    49: "R8G8_UNORM", 50: "R8G8_UINT", 51: "R8G8_SNORM", 52: "R8G8_SINT",
    54: "R16_FLOAT", 56: "R16_UNORM", 57: "R16_UINT", 58: "R16_SNORM", 59: "R16_SINT",
    61: "R8_UNORM", 62: "R8_UINT", 63: "R8_SNORM", 64: "R8_SINT",
    71: "BC1_UNORM", 72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM", 75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM", 78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM", 81: "BC4_SNORM",
    83: "BC5_UNORM", 84: "BC5_SNORM",
    87: "B8G8R8A8_UNORM", 91: "B8G8R8A8_UNORM_SRGB",
    95: "BC6H_UF16", 96: "BC6H_SF16",
    98: "BC7_UNORM", 99: "BC7_UNORM_SRGB",
}


def find_rage_textures(data):
    """Find validated RAGE9 0x980 texture descriptors (ASSET-148; layout from tools/tex_headers.py).

    Returns a list of (width, height, depth, dxgi_name, flags_lo, mips) tuples. Requires dtype == 0x980,
    256-byte-aligned 0x60000000..0x6FFFFFFF GPU pointer, known DXGI format, and sane dimensions.
    """
    n_qwords = len(data) // 8
    if n_qwords < 8:
        return []
    mv = memoryview(data)[:n_qwords * 8]
    if _UINT64_CODE is not None:
        q = array.array(_UINT64_CODE)
        q.frombytes(mv)
        if sys.byteorder != "little":
            q.byteswap()
    else:
        q = struct.unpack(f"<{n_qwords}Q", mv)
    out = []
    for k in range(2, n_qwords - 5):
        ptr = q[k]
        if not (0x60000000 <= ptr <= 0x6FFFFFFF) or (ptr & 0xFF) != 0:
            continue
        if q[k - 1] != 0x980:
            continue
        d = q[k + 4].to_bytes(8, "little")
        fmt = d[7]
        if fmt not in DXGI_FORMATS:
            continue
        w = int.from_bytes(d[0:2], "little")
        h = int.from_bytes(d[2:4], "little")
        depth = int.from_bytes(d[4:6], "little")
        if not (1 <= w <= 16384 and 1 <= h <= 16384 and 1 <= depth <= 2048):
            continue
        fword = q[k + 5]
        mips = (fword >> 16) & 0xFF
        if min(w, h) < 4 and not 1 <= mips <= max(w, h, depth).bit_length():
            continue
        out.append((w, h, depth, DXGI_FORMATS[fmt], fword & 0xFFFF, mips))
    return out


def is_pg_tex_dictionary(data, count):
    """True when a zero-prefix stream carries a standalone pgDictionary<grcTexture> header matching count."""
    if count <= 0 or len(data) < 0x40:
        return False
    u8, u10 = struct.unpack_from("<QQ", data, 8)
    if (u8 >> 28) != 5 or u10 != 0:
        return False
    p20, c28, _cap2a, p30, c38, _cap3a = struct.unpack_from("<QHHxxxxQHH", data, 0x20)
    return (p20 >> 28) == 5 and (p30 >> 28) == 5 and c28 == c38 == count


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if len(data) >= at + 8 else None


def _be32(data, at):
    return int.from_bytes(data[at:at + 4], "big") if len(data) >= at + 4 else 0


def _printable(data):
    if not data:
        return False
    sample = data[:512]
    good = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
    return len(sample) >= 64 and good / len(sample) > 0.9


def psin_sections(data):
    """Return a PSIN section chain as [(tag, offset, size), ...].

    The header's big-endian first-section offset at +4 is authoritative when it points at a
    known tag. A record is a four-byte tag and a big-endian u32 total size that includes the
    eight-byte record header. The chain stops at CHKS or at the first unknown/out-of-range
    record. Only the six measured tags are accepted.
    """
    def parse(at):
        chain = []
        pos = at
        while pos + 8 <= len(data):
            tag = data[pos:pos + 4]
            if tag not in PSIN_TAGS:
                break
            size = _be32(data, pos + 4)
            if size < 8 or pos + size > len(data):
                break
            chain.append((tag.decode("ascii"), pos, size))
            if tag == b"CHKS":
                break
            pos += size
        return chain

    first = _be32(data, 4)
    if 8 <= first < len(data) and data[first:first + 4] in PSIN_TAGS:
        chain = parse(first)
        if chain:
            return chain
    best = []
    start = 0
    while True:
        at = data.find(b"LSCH", start)
        if at < 0:
            break
        chain = parse(at)
        if len(chain) > len(best):
            best = chain
        start = at + 1
    return best


def _is_xml(data):
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.lstrip()[:5] == b"<?xml"


def classify(data, offset, stored):
    """Return one row for an inflated stream or a Bink 2 header."""
    row = {"offset": offset, "stored": stored, "size": len(data), "signature": "raw",
           "version": None, "count": None, "detail": ""}
    if data[:4] == b"PSIN":
        chain = psin_sections(data)
        complete = bool(chain) and chain[-1][0] == "CHKS" and chain[-1][1] + chain[-1][2] == len(data)
        row.update(signature="psin", count=len(chain),
                   detail=f"tags={','.join(t for t, _, _ in chain)};"
                          f"field8={_u32(data, 8)};complete={int(complete)}")
        return row
    if data[:4] == b"RBF0":
        found = re.search(rb"[\x20-\x7e]{4,120}", data[8:160])
        schema = found.group().decode("ascii", "replace") if found else "?"
        row.update(signature="rbf0", detail=f"schema={schema}")
        return row
    for at in (0x10, 0):
        if data[at:at + 4] == b"0DRP":
            count = _u64(data, at + 0x10) if at == 0x10 else _u32(data, at + 0x0c)
            row.update(signature="0drp", version=_u32(data, at + 4), count=count,
                       detail=f"0DRP@{at:#x}")
            return row
    if data[:4] in (b"RSC8", b"RSC7"):
        row.update(signature=data[:4].decode("ascii").lower(), version=_u32(data, 4),
                   detail=f"declared={_u32(data, 8)};flags={_u32(data, 12)}")
        return row
    if data[:4] == b"9FPR":
        row.update(signature="rpf9-nested",
                   detail=f"data_off={_u32(data, 4)};toc={_u32(data, 8)}")
        return row
    if data[:4] == b"AWC\x00":
        row.update(signature="awc", version=_u32(data, 4))
        return row
    if data[:4] == b"ADAT":
        row.update(signature="adat")
        return row
    if data[:4] == b"FSSM":
        row.update(signature="fssm", version=_u32(data, 4))
        return row
    if data[:4] == b"SCRP":
        row.update(signature="scrp", version=_u32(data, 4))
        return row
    if data[:4] == b"YSC\x00":
        row.update(signature="ysc")
        return row
    if data[:4] == b"RSCC":
        row.update(signature="rscc", version=_u32(data, 4))
        return row
    if data[:3] == b"KB2" and len(data) >= 0x20:
        row.update(signature="bink2", version=chr(data[3]), count=_u32(data, 8),
                   detail=f"{_u32(data, 0x14)}x{_u32(data, 0x18)}", size=stored)
        return row
    if data[:4] == b"\x42\x40\x00\x00":
        row.update(signature="respage", detail="0x4042")
        return row
    if data[:4] in (b"DDS ", b"DXT1", b"DXT3", b"DXT5", b"ATI2", b"BC5\x00", b"BC7\x00"):
        row.update(signature=data[:4].decode("ascii", "replace"))
        return row
    if _is_xml(data):
        row.update(signature="xml", detail="XML text")
        return row
    if len(data) >= 16 and _u32(data, 4) == 8 and _u32(data, 12) == 0 and 0 < (_u32(data, 8) or 0) <= 4096:
        row.update(signature="typehash", version=8, count=_u32(data, 8),
                   detail=f"type={_u32(data, 0):#010x}")
        return row
    if len(data) >= 8 and _u32(data, 0) in (2, 3) and data[4:8] == b"\x01\x01\x00\x00":
        row.update(signature="clip", version=_u32(data, 0))
        return row
    if data[:8] == b"\x00" * 8:
        pointer = _u64(data, 8)
        detail = f"u64@8={pointer:#x}" if pointer is not None else "u64@8=?"
        texs = find_rage_textures(data) if pointer not in (None, 0) else []
        if texs:
            is_dict = is_pg_tex_dictionary(data, len(texs))
            fmts = Counter(t[3] for t in texs).most_common(4)
            fmt_str = ",".join(f"{name}:{n}" for name, n in fmts)
            detail += f";tex980={len(texs)};dict={int(is_dict)};fmts={fmt_str}"
            row.update(signature="zero-prefix", count=len(texs), detail=detail)
        else:
            row.update(signature="zero-prefix", detail=detail)
        return row
    if _printable(data):
        row.update(signature="text", detail="printable text")
        return row
    row.update(signature="raw:" + data[:4].hex())
    return row



def scan_rpf(path, start=None, limit=None, max_inflate=MAX_INFLATE):
    """Walk an RPF9 data area. Returns (rows, summary). No TOC is read or decrypted."""
    rows = []
    zero = unread = skipped = 0
    t0 = time.time()
    if os.path.getsize(path) == 0:
        raise ValueError("empty file")
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            size = len(m)
            first = struct.unpack_from("<I", m, 4)[0] if size >= 8 else 0
            pos = start if start is not None else first
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit else size
            span_start = pos
            last = pos
            while pos < end:
                found = ZHDR.search(m, pos, end)
                if not found:
                    break
                s = found.start()
                if m[s:s + 3] == b"KB2":
                    blen = (_u32(m, s + 4) or 0) + 8
                    w, h = _u32(m, s + 0x14), _u32(m, s + 0x18)
                    if not (64 <= blen <= end - s and w and h and 16 <= w <= 8192 and 16 <= h <= 8192):
                        pos = s + 1
                        continue
                    z, u = gap_split(m, last, s)
                    zero += z
                    unread += u
                    rows.append(classify(m[s:s + 256], s, blen))
                    last = pos = s + blen
                    continue
                d = zlib.decompressobj()
                p, total, head, body = s, 0, b"", bytearray()
                too_big = False
                keep_body = True
                try:
                    while not d.eof and p < end:   # inside the --limit window, as in story_index (#712)
                        block = m[p:min(p + (1 << 20), end)]
                        if not block:
                            break
                        p += len(block)
                        while block and not d.eof:
                            chunk = d.decompress(block, 1 << 20)
                            block = d.unconsumed_tail
                            if chunk:
                                total += len(chunk)
                                if total > max_inflate:
                                    too_big = True
                                    break
                                if len(head) < 256:
                                    head += chunk[:256 - len(head)]
                                if keep_body:
                                    if len(head) >= 16:
                                        is_psin = head[:4] == b"PSIN"
                                        is_cpu = head[:8] == b"\x00" * 8 and head[:16] != b"\x00" * 16
                                        if not is_psin and not (is_cpu and total <= MAX_TEX_SCAN):
                                            keep_body = False
                                            body.clear()
                                    if keep_body:
                                        body += chunk
                            if not block:
                                break
                        if too_big:
                            break
                except zlib.error:
                    pos = s + 1
                    continue
                if too_big or not d.eof or total < 64:
                    skipped += 1
                    pos = s + 1
                    continue
                used = p - s - len(d.unused_data)
                z, u = gap_split(m, last, s)
                zero += z
                unread += u
                row = classify(bytes(body) if keep_body and body else head, s, used)
                row["size"] = total
                rows.append(row)
                last = pos = s + used
            z, u = gap_split(m, last, end)
            zero += z
            unread += u
    summary = summarise(rows, end - span_start, zero, unread)
    summary.update(mode="rpf9", archive=path, elapsed_seconds=round(time.time() - t0, 3),
                   skipped_candidates=skipped)
    return rows, summary


def _carve_index(path):
    """rpf9_carve.py's index.json in `path`: ({file name: stored bytes}, summary), or ({}, {}) without one."""
    try:
        with open(os.path.join(path, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        stored = {f"{int(e['off']):012x}.bin": int(e["stored"]) for e in index.get("files", [])}
        summary = index.get("summary", {})
        return stored, summary if isinstance(summary, dict) else {}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}, {}


def scan_dir(path, max_inflate=MAX_INFLATE):
    """Index already-carved files. A leading hexadecimal file name supplies the offset.

    With rpf9_carve.py's index.json in the folder, each file's stored (compressed) size and the carve's span,
    zero padding and unread bytes come from it, so compressed_bytes and readable_fraction mean what they mean in
    archive mode. Without it, a file's stored size can only be its length on disk (the inflated size)."""
    rows = []
    skipped = 0
    t0 = time.time()
    carve_stored, carve_summary = _carve_index(path)
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full) or not (name.endswith(".bin") or name.endswith(".bk2")):
            continue
        if name.endswith(".bin") and os.path.getsize(full) > max_inflate:
            skipped += 1
            continue
        found = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(found.group(1), 16) if found else len(rows)
        with open(full, "rb") as f:
            data = f.read()
        stored = carve_stored.get(name, len(data))
        if len(data) >= 3 and data[:3] == b"KB2":
            row = classify(data, offset, stored)
        else:
            row = classify(data, offset, stored)
            row["size"] = len(data)
        rows.append(row)
    if carve_stored and "span" in carve_summary:
        summary = summarise(rows, int(carve_summary["span"]), int(carve_summary.get("zero_padding", 0)),
                            int(carve_summary.get("unread_bytes", 0)))
    else:
        summary = summarise(rows, sum(r["stored"] for r in rows), 0, 0)
    summary.update(mode="directory", directory=path, elapsed_seconds=round(time.time() - t0, 3),
                   skipped_candidates=skipped)
    return rows, summary


def summarise(rows, span, zero, unread):
    by_sig = {}
    for row in rows:
        entry = by_sig.setdefault(row["signature"], {"files": 0, "stored": 0, "inflated": 0})
        entry["files"] += 1
        entry["stored"] += row["stored"]
        entry["inflated"] += row["size"]
    stored = sum(row["stored"] for row in rows)
    tex_rows = [row for row in rows if row["signature"] == "zero-prefix" and ";tex980=" in row["detail"]]
    return {
        "files": len(rows),
        "compressed_bytes": stored,
        "inflated_bytes": sum(row["size"] for row in rows),
        "bink2_files": sum(1 for row in rows if row["signature"] == "bink2"),
        "bink2_bytes": sum(row["stored"] for row in rows if row["signature"] == "bink2"),
        "texture_streams": len(tex_rows),
        "texture_dictionaries": sum(1 for row in tex_rows if ";dict=1" in row["detail"]),
        "texture_headers": sum(row["count"] or 0 for row in tex_rows),
        "zero_padding": zero,
        "unread_bytes": unread,
        "span": span,
        "readable_fraction": round(stored / span, 6) if span else 0.0,
        "signatures": by_sig,
    }


def write_output(path, rows, summary):
    """Write CSV or JSON from the file extension."""
    if os.path.splitext(path)[1].lower() == ".json":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows}, f, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([f"0x{row['offset']:x}" if i == 0 else ("" if row[c] is None else row[c])
                             for i, c in enumerate(COLUMNS)])


def offset(text):
    """--start: 0 or more, decimal or 0x hex."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def byte_count(text):
    """--limit/--max-inflate: at least 1; 0 used to mean no limit."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _samefile(source, out):
    """True when `out` names the same existing file as `source`.

    os.path.samefile needs both paths on disk, so a not-yet-written output is
    reported as "no alias" instead of forcing a scan or a write first. Any other
    comparison failure propagates for the caller to report.
    """
    try:
        return os.path.samefile(source, out)
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved .bin/.bk2 files")
    ap.add_argument("out", help="output path ending in .csv or .json")
    ap.add_argument("--start", type=offset, help="first byte to scan (default: RPF data start)")
    ap.add_argument("--limit", type=byte_count, help="number of bytes to scan")
    ap.add_argument("--max-inflate", type=byte_count, default=MAX_INFLATE,
                    help="skip a stream that inflates above this many bytes (default: 512 MiB)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if os.path.isdir(args.source) and (args.start is not None or args.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    if os.path.isdir(args.out):
        ap.error(f"{args.out} is a folder, not a .csv or .json file")
    folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(folder):
        ap.error(f"{folder} is not a folder, so {args.out} cannot be written")
    try:
        overwrites_source = _samefile(args.source, args.out)
        if not overwrites_source and os.path.isdir(args.source) and os.path.isfile(args.out):
            # Protect scanned payloads and their carve manifest, including hard-link aliases.
            for name in sorted(os.listdir(args.source)):
                if name == "index.json" or name.endswith(".bin") or name.endswith(".bk2"):
                    full = os.path.join(args.source, name)
                    if os.path.isfile(full) and _samefile(full, args.out):
                        overwrites_source = True
                        break
    except OSError as e:
        ap.error(f"cannot compare source and output paths: {e}")
    if overwrites_source:
        ap.error("output must not overwrite the source archive")
    try:
        if os.path.isdir(args.source):
            rows, summary = scan_dir(args.source, args.max_inflate)
        else:
            rows, summary = scan_rpf(args.source, args.start, args.limit, args.max_inflate)
    except ValueError as e:
        sys.exit(f"{args.source}: {e}")
    except OSError as e:
        sys.exit(f"{args.source}: {e.strerror or e}")
    try:
        write_output(args.out, rows, summary)
    except OSError as e:
        print(f"ERROR: cannot write {args.out}: {e.strerror or e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps({k: summary[k] for k in
                          ("files", "compressed_bytes", "inflated_bytes", "zero_padding",
                           "unread_bytes", "span", "readable_fraction", "bink2_files",
                           "skipped_candidates", "elapsed_seconds")}, indent=1))
        for name, counts in sorted(summary["signatures"].items(), key=lambda kv: -kv[1]["files"]):
            print(f"  {name:24} {counts['files']:6} files {counts['stored']:12} stored "
                  f"{counts['inflated']:12} inflated")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
