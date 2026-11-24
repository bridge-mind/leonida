"""Carve the zlib-compressed files out of an RPF9 archive without its (encrypted) table of contents.

The data area is mostly zlib streams laid end to end, separated by zero padding. This walks the archive, inflates
every stream that decodes cleanly to the end, and writes each one as <offset>.bin. It also records which byte
ranges it couldn't read, so coverage is measurable.

index.json holds the summary (totals), `files` (one row per carved stream) and `unread`: one row per gap between
streams (or after the last one) that is not all zero bytes, with its offset, size and the Shannon entropy in
bits/byte of its first ENTROPY_SAMPLE bytes (`sampled`). An all-zero gap is padding and only counts toward
`zero_padding`. The bytes are counted, never decoded.

Usage:
    python -I rpf9_carve.py <archive.rpf> <out_dir> [--start N] [--limit BYTES] [--until-adat] [--index-only]

--until-adat ends the scan at the first raw AWC bank (a little-endian `ADAT` whose header parses as one,
rsc8_audio.parse_adat_header_bytes), so the zlib-compressed audio config at the start of audio.rpf is carved without
touching the banks and without a build-specific --limit (docs/audio_config.md). The bytes `ADAT` alone are not
enough: they also occur inside `METADATA` and in compressed data. index.json's summary records the boundary.
"""
import argparse
import collections
import json
import math
import mmap
import os
import re
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsc8_audio import parse_adat_header_bytes  # noqa: E402
from restofworld_model_join import ModelSystemPages  # noqa: E402

# zlib stream start, or a Bink 2 video ('KB2' + revision letter, then u32 size-8).
# RPF9 magic as it sits on disk: the tag "RPF9" stored little-endian (docs/rpf.md, docs/bvh.md).
MAGIC = b"9FPR"
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
ENTROPY_SAMPLE = 1 << 20   # bytes of an unread range used for its entropy
GAP_CHUNK = 1 << 20        # a gap is checked this many bytes at a time: one can be gigabytes long


def all_zero(buf, a, b, chunk=GAP_CHUNK):
    """Whether buf[a:b] is all zero bytes, reading at most `chunk` bytes at a time.

    Slicing a whole gap of a multi-gigabyte archive at once ran out of memory (a 20 GB archive in this package).
    """
    for p in range(a, b, chunk):
        piece = buf[p:min(p + chunk, b)]
        if piece.count(0) != len(piece):
            return False
    return True


def entropy(sample):
    """Shannon entropy of a byte string in bits per byte (0.0 for an empty one)."""
    if not sample:
        return 0.0
    n = len(sample)
    return -sum(c / n * math.log2(c / n) for c in collections.Counter(sample).values())


def below_normal():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        # GetCurrentProcess returns a pointer-sized pseudo-handle.
        k.GetCurrentProcess.argtypes = ()
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.SetPriorityClass.restype = wintypes.BOOL
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


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
    """--limit: at least 1; 0 used to mean no limit and a negative value silently scanned less."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


ADAT_HEADER_READ = 1 << 20   # enough for every bank header the parser accepts in practice; it rejects longer ones


def first_adat_bank(m, start, end):
    """Offset of the first raw AWC bank in m[start:end] whose header parses (not just the four bytes), or None."""
    size = len(m)
    pos = start
    while pos < end:
        at = m.find(b"ADAT", pos, end)
        if at < 0:
            return None
        if parse_adat_header_bytes(m[at:min(size, at + ADAT_HEADER_READ)], at, size) is not None:
            return at
        pos = at + 1
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--start", type=offset)
    ap.add_argument("--limit", type=byte_count)
    ap.add_argument("--until-adat", action="store_true",
                    help="end the scan at the first raw AWC (ADAT) bank, e.g. the audio config at the start of "
                         "audio.rpf")
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(argv)
    below_normal()
    if os.path.isdir(a.archive):
        sys.exit(f"{a.archive}: is a directory, not an RPF9 archive")
    try:
        size = os.path.getsize(a.archive)
    except OSError as e:
        sys.exit(f"{a.archive}: {e.strerror or e}")
    if size == 0:
        sys.exit(f"{a.archive}: empty file")
    if size < 8:
        sys.exit(f"{a.archive}: too short for an RPF9 header ({size} bytes)")
    try:
        with open(a.archive, "rb") as fh:
            hdr = fh.read(8)
    except OSError as e:
        sys.exit(f"{a.archive}: {e.strerror or e}")
    if hdr[:4] != MAGIC:
        sys.exit(f"{a.archive}: not an RPF9 archive (expected the magic {MAGIC.decode()} at offset 0)")
    if a.start is not None and not 0 <= a.start < size:
        sys.exit(f"{a.archive}: --start {a.start} is outside the file (0..{size - 1})")
    if a.start is None:
        base = struct.unpack_from("<I", hdr, 4)[0]
        if base > size:   # base == size is an empty data area, which is fine
            sys.exit(f"{a.archive}: header data offset {base} is past the end of the file ({size} bytes)")
    else:
        base = a.start
    if os.path.exists(a.out) and not os.path.isdir(a.out):
        sys.exit(f"{a.out}: output path is a file, not a directory")
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"{a.out}: cannot create output directory: {e.strerror or e}")
    with open(a.archive, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        pos = base
        end = min(size, pos + a.limit) if a.limit else size
        boundary = None
        if a.until_adat:
            boundary = first_adat_bank(m, pos, end)
            if boundary is None:
                print(f"--until-adat: no AWC bank between {pos:#x} and {end:#x}; scanning all of it", flush=True)
            else:
                end = boundary
                print(f"--until-adat: first AWC bank at {boundary:#x}; scanning {pos:#x}..{boundary:#x}", flush=True)
        rows, last, unread, zero = [], pos, 0, 0
        unread_ranges = []
        system_pages = ModelSystemPages()

        def account(a, b):
            """Count the gap m[a:b] as zero padding, or as an unread range that is also listed."""
            nonlocal zero, unread
            if all_zero(m, a, b):
                zero += b - a
                return
            unread += b - a
            sample = m[a:min(b, a + ENTROPY_SAMPLE)]
            unread_ranges.append({"off": a, "size": b - a, "entropy": round(entropy(sample), 3),
                                  "sampled": len(sample)})

        t0 = time.time()
        while pos < end:
            mt = ZHDR.search(m, pos, end)
            if not mt:
                break
            s = mt.start()
            if m[s:s + 3] == b"KB2":
                if s + 0x1C > end:   # header cut off by the end of the file or of --limit's window: not a video
                    pos = s + 1
                    continue
                blen = struct.unpack_from("<I", m, s + 4)[0] + 8
                w, h = struct.unpack_from("<II", m, s + 0x14)
                if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                    pos = s + 1
                    continue
                account(last, s)
                rows.append({"off": s, "stored": blen, "size": blen, "type": "bink2", "w": w, "h": h,
                             "frames": struct.unpack_from("<I", m, s + 8)[0], "pre": "", "head": m[s:s + 16].hex()})
                if not a.index_only:
                    out_file = os.path.join(a.out, f"{s:012x}.bk2")
                    try:
                        with open(out_file, "wb") as f:
                            f.write(m[s:s + blen])
                    except OSError as e:
                        sys.exit(f"{out_file}: cannot write: {e.strerror or e}")
                last = pos = s + blen
                system_pages.reset()
                continue
            d = zlib.decompressobj()
            try:
                chunks, p = [], s
                # Inflate only inside the scan window (all of the file without --limit): a stream that runs past
                # its end is left out, so readable + zero + unread add up to the span, as in story_index (#712).
                while not d.eof and p < end:
                    chunk = m[p:min(p + (1 << 20), end)]
                    if not chunk:
                        break
                    chunks.append(d.decompress(chunk))
                    p += len(chunk)   # the slice can be short at EOF; advancing by 1 MiB overran it
                    if d.unused_data or d.eof:
                        break
                out = b"".join(chunks) + d.flush()
            except zlib.error:
                pos = s + 1
                continue
            adjacent = s == ((last + 15) & ~15) and not any(m[last:s])
            if not d.eof or len(out) < system_pages.minimum_size(adjacent):
                pos = s + 1
                continue
            used = p - s - len(d.unused_data)
            system_pages.accept(out, adjacent=adjacent,
                                root_framed=(s & 0xFFF) == 0x10 and s >= 16 and m[s - 16:s] == b"\0" * 16)
            account(last, s)
            rows.append({"off": s, "stored": used, "size": len(out), "pre": m[max(0, s - 16):s].hex(),
                         "head": out[:16].hex()})
            if not a.index_only:
                out_file = os.path.join(a.out, f"{s:012x}.bin")
                try:
                    with open(out_file, "wb") as f:
                        f.write(out)
                except OSError as e:
                    sys.exit(f"{out_file}: cannot write: {e.strerror or e}")
            last = pos = s + used
            if len(rows) % 2000 == 0:
                done = (pos - (a.start or 0)) / max(1, end - (a.start or 0))
                print(f"[{done:6.1%}] {len(rows)} files, {time.time() - t0:.0f}s", flush=True)
        account(last, end)
    total = sum(r["stored"] for r in rows)
    bink = sum(r["stored"] for r in rows if r.get("type") == "bink2")
    span = end - base
    summary = {"archive": a.archive, "files": len(rows), "compressed_bytes": total,
               "inflated_bytes": sum(r["size"] for r in rows), "bink2_bytes": bink,
               "bink2_files": sum(1 for r in rows if r.get("type") == "bink2"),
               "zero_padding": zero, "unread_bytes": unread, "unread_ranges": len(unread_ranges),
               "span": span}
    if a.until_adat:
        summary["first_adat_bank"] = boundary   # None: no bank in the window, the whole window was scanned
    index_path = os.path.join(a.out, "index.json")
    try:
        with open(index_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows, "unread": unread_ranges}, f)
    except OSError as e:
        sys.exit(f"{index_path}: cannot write: {e.strerror or e}")
    print(json.dumps(summary, indent=1))
    span = max(1, span)   # an empty data area (or --start at the end) has nothing to divide by
    print(f"readable: {total / span:.1%}, zero padding {zero / span:.1%}, unread {unread / span:.1%}")
    if unread_ranges:
        print(f"unread: {len(unread_ranges)} range(s), largest {max(r['size'] for r in unread_ranges)} bytes "
              f"(listed in index.json)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
