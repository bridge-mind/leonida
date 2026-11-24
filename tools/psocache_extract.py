"""Carve the PSO/shader cache out of an RPF9 archive, reproducibly and without the encrypted TOC.

The cache starts with the magic ``PS\\0\\x09``. In the measured package it is one resource stored as a chain of
zlib streams in an archive's data area: 4,205 adjacent streams, 4,204 of 64 KiB and a 27,436-byte last one,
31,437 bytes of zero padding between them, 275,540,780 inflated bytes in total. It is the last resource in its
archive, but that is not assumed here.

How the search stays bounded and deterministic:

1. The archive header gives the data start (`u32` at +4). From there the tool walks the stream table in archive
   order, the same walk `rpf9_carve.py` uses. A stream starts at a zlib header (`78 01/5e/9c/da`); anything else
   between streams is zero padding or unread bytes. Every stream is inflated in 1 MiB pieces so the walk learns
   where it ends; only the first 4 inflated bytes are kept for the test. The output is discarded otherwise.
2. The first stream whose inflated bytes start with ``PS\\0\\x09`` wins and the scan stops. If none is found
   before ``--budget`` compressed bytes (default 8 GiB) or ``--max-streams`` streams (default 200,000), or at the
   end of the archive, the tool exits 2 without writing anything.
3. The header of that stream declares the total: ``0x80 + graphics*0x70 + compute*0x18 + meta_index*8 +
   shader_index*8 + meta_size + code_size``. The tool appends following streams after at most 1 MiB of zero
   padding until exactly that many inflated bytes exist. A stream that would overshoot, a missing stream, or a
   missing header is reported with the byte counts, not guessed.
4. The file is written to ``--out``, which must be inside the system temporary directory, and then checked with
   ``psocache_inspect.inspect``. The exit code is 0 only when that model holds.

Usage:
    python -I psocache_extract.py <archive.rpf|folder> <out> [--budget BYTES] [--max-streams N] [--quiet]

Exit code: 0 extracted and the model holds, 1 extracted but the model disagrees, 2 not found, unreadable input,
a source budget exhausted, or an output path outside the temporary directory. The tool prints counts, offsets
and sizes only, never an archive name or a path from the package.
"""

import argparse
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psocache_inspect  # noqa: E402

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
MAGIC = psocache_inspect.MAGIC
HEADER_SIZE = psocache_inspect.HEADER_SIZE
GRAPHICS_STRIDE = psocache_inspect.GRAPHICS_STRIDE
COMPUTE_STRIDE = psocache_inspect.COMPUTE_STRIDE
INDEX_STRIDE = psocache_inspect.INDEX_STRIDE
COUNT_OFFS = psocache_inspect.COUNT_OFFS
SIZE_OFFS = psocache_inspect.SIZE_OFFS
MAX_GAP = 1 << 20  # zero padding allowed between two streams of one resource
DEFAULT_BUDGET = 1 << 33  # compressed bytes examined before the search gives up
DEFAULT_MAX_STREAMS = 200000


class NotFound(Exception):
    """The cache was not found inside the search budget."""


def inflate(m, start, end, keep_all=False):
    """Inflate one zlib stream at ``start``.

    Returns ``(used, head)`` where ``used`` is the first byte after the stream (the decompressor may have read
    past its end) and ``head`` is the first 4 inflated bytes, or ``(None, None)`` when it does not decode.
    With ``keep_all`` the inflated bytes are returned in place of the 4-byte head; memory then grows with the
    stream, which is why the search path does not use it.
    """
    d = zlib.decompressobj()
    head = bytearray()
    out = bytearray() if keep_all else None
    p = start
    try:
        while p < end and not d.eof:
            piece = m[p : min(p + (1 << 20), end)]
            chunk = d.decompress(piece)
            p += len(piece)
            if keep_all:
                out += chunk
            elif len(head) < len(MAGIC):
                head += chunk[: len(MAGIC) - len(head)]
    except zlib.error:
        return None, None
    if not d.eof:
        return None, None
    used = p - len(d.unused_data)
    return used, bytes(out if keep_all else head)


def declared_total(header):
    """The total the cache header declares, or None when the header is too short."""
    if len(header) < HEADER_SIZE:
        return None
    counts = tuple(struct.unpack_from("<I", header, off)[0] for off in COUNT_OFFS)
    sizes = tuple(struct.unpack_from("<I", header, off)[0] for off in SIZE_OFFS)
    graphics, compute, meta_index, shader_index = counts
    meta_size, code_size = sizes
    return (
        HEADER_SIZE
        + graphics * GRAPHICS_STRIDE
        + compute * COMPUTE_STRIDE
        + meta_index * INDEX_STRIDE
        + shader_index * INDEX_STRIDE
        + meta_size
        + code_size
    )


def find_match(m, size, start, budget, max_streams):
    """Walk the stream table from ``start`` and return the first stream whose output starts with the magic."""
    pos = start
    spent = 0
    streams = 0
    while pos < size and streams < max_streams and spent < budget:
        found = ZHDR.search(m, pos, size)
        if not found:
            break
        s = found.start()
        used, head = inflate(m, s, size)
        if used is None:
            pos = s + 1  # a false zlib header inside another stream
            continue
        streams += 1
        spent += used - s
        if head == MAGIC:
            return {"offset": s, "streams": streams, "spent": spent}
        pos = used
    return {"offset": None, "streams": streams, "spent": spent}


def assemble(m, size, first, total):
    """Append following contiguous streams until ``total`` inflated bytes exist, and return them."""
    used, first_bytes = inflate(m, first, size, keep_all=True)
    if used is None or len(first_bytes) < HEADER_SIZE:
        raise NotFound("the candidate stream does not inflate to a full header")
    data = bytearray(first_bytes)
    if len(data) > total:
        raise ValueError(
            "first stream is %d bytes, more than the declared %d" % (len(data), total)
        )
    pos = used
    while len(data) < total:
        gap = 0
        while pos < size and m[pos] == 0 and gap < MAX_GAP:
            pos += 1
            gap += 1
        if gap >= MAX_GAP or not ZHDR.match(m, pos):
            raise ValueError(
                "only %d of %d declared bytes are in contiguous streams"
                % (len(data), total)
            )
        used, chunk = inflate(m, pos, size, keep_all=True)
        if used is None:
            raise ValueError("stream after %d bytes does not inflate" % len(data))
        if len(data) + len(chunk) > total:
            raise ValueError(
                "stream after %d bytes adds %d, past the declared %d"
                % (len(data), len(chunk), total)
            )
        data += chunk
        pos = used
    return bytes(data)


def inside_tempdir(path):
    root = os.path.realpath(tempfile.gettempdir())
    target = os.path.realpath(os.path.abspath(path))
    return target.startswith(root + os.sep)


def archives(source):
    if os.path.isdir(source):
        return [
            os.path.join(source, name)
            for name in sorted(os.listdir(source))
            if os.path.isfile(os.path.join(source, name))
        ]
    return [source]


def extract(
    source, out, budget=DEFAULT_BUDGET, max_streams=DEFAULT_MAX_STREAMS, quiet=False
):
    """Find, carve and validate the cache. Returns ``(archive_index, report, search)`` or raises NotFound."""
    if not inside_tempdir(out):
        raise ValueError("--out must be a file inside the temporary directory")
    paths = archives(source)
    left = budget
    spent = 0
    streams = 0
    for index, path in enumerate(paths, 1):
        if left <= 0:
            break
        try:
            fh = open(path, "rb")
        except OSError as e:
            raise ValueError(
                "archive #%d cannot be opened: %s" % (index, e.strerror or "I/O error")
            )
        with fh:
            size = os.fstat(fh.fileno()).st_size
            if size < 8:
                continue
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
                if bytes(m[:4]) not in (b"RPF9", b"9FPR"):
                    continue
                data_start = struct.unpack_from("<I", m, 4)[0]
                if not 8 <= data_start <= size:
                    continue
                search = find_match(m, size, data_start, left, max_streams - streams)
                streams += search["streams"]
                spent += search["spent"]
                left -= search["spent"]
                if search["offset"] is None:
                    if not quiet:
                        print(
                            "archive #%d: size %d, %d streams, no cache in %d compressed bytes"
                            % (index, size, search["streams"], search["spent"])
                        )
                    continue
                used, header = inflate(m, search["offset"], size, keep_all=True)
                total = declared_total(header)
                if total is None:
                    raise ValueError(
                        "archive #%d at stream offset 0x%x: candidate header is %d bytes, shorter than 0x%x"
                        % (index, search["offset"], len(header), HEADER_SIZE)
                    )
                search["candidates"] = 1
                search["total"] = total
                search["header_bytes"] = len(header)
                try:
                    data = assemble(m, size, search["offset"], total)
                except ValueError as e:
                    raise ValueError(
                        "archive #%d at stream offset 0x%x: %s"
                        % (index, search["offset"], e)
                    )
                with open(out, "wb") as f:
                    f.write(data)
                report = psocache_inspect.inspect(out)
                return (
                    index,
                    report,
                    {"size": size, **search, "archive_stream_offset": search["offset"]},
                )
    raise NotFound(
        "no cache found: %d archives, %d streams, %d compressed bytes examined"
        % (len(paths), streams, spent)
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Carve the PS\\0\\x09 PSO/shader cache out of an RPF9 archive, using its stream table."
    )
    ap.add_argument("source", help="RPF9 archive, or a folder of archives")
    ap.add_argument("out", help="output cache file, inside the temporary directory")
    ap.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="compressed bytes to examine before giving up (default %(default)s)",
    )
    ap.add_argument(
        "--max-streams",
        type=int,
        default=DEFAULT_MAX_STREAMS,
        help="streams to inflate before giving up (default %(default)s)",
    )
    ap.add_argument(
        "--quiet", action="store_true", help="print nothing, only set the exit code"
    )
    a = ap.parse_args(argv)
    if a.budget < 1:
        ap.error("--budget must be at least 1")
    if a.max_streams < 1:
        ap.error("--max-streams must be at least 1")
    try:
        index, report, search = extract(
            a.source, a.out, a.budget, a.max_streams, a.quiet
        )
    except (OSError, ValueError, NotFound) as e:
        print("%s" % e, file=sys.stderr)
        return 2
    if not a.quiet:
        print(
            "archive #%d: size %d, %d streams scanned, %d compressed bytes"
            % (index, search["size"], search["streams"], search["spent"])
        )
        print(
            "match: archive stream offset 0x%x, declared %d bytes"
            % (search["offset"], search["total"])
        )
        print(
            "wrote %d bytes; graphics=%d compute=%d meta_index=%d shader_index=%d"
            % (
                report["size"],
                report["counts"]["graphics"],
                report["counts"]["compute"],
                report["counts"]["meta_index"],
                report["counts"]["shader_index"],
            )
        )
        print("verdict: %s" % report["verdict"])
        for problem in report["problems"]:
            print("problem: %s" % problem)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
