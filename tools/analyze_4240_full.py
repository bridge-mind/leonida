#!/usr/bin/env python3
"""Analyze `42 40 00 00` vertex blocks in RPF9 archives (ASSET-57).

These blocks use the same 0x60-byte slot-word header as `63 30 00 00` mesh blocks,
but with different vertex formats. This tool finds and measures their strides.
Usage: `python -I tools/analyze_4240_full.py --help` lists the options.
"""

import argparse
import json
import mmap
import os
import re
import struct
import sys
import zlib

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

# Constants from meshlet_page.py
BLOCK_MAGIC = 0x3063       # 63 30 00 00
OTHER_BLOCK_MAGIC = 0x4042 # 42 40 00 00
HEADER_SIZE = 0x60
SLOT_COUNT = 23
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")

def _u32(data, at):
    if at + 4 > len(data):
        raise ValueError(f"truncated at +0x{at:x}")
    return struct.unpack_from("<I", data, at)[0]

def is_mesh_page(data):
    """Check whether data starts with either vertex block header or a meshlet group."""
    if len(data) < 16:
        return False
    word = struct.unpack_from("<I", data, 0)[0]
    if word in (BLOCK_MAGIC, OTHER_BLOCK_MAGIC):
        return True
    if not word & 0x80000000:
        return False
    try:
        head = word
        meshlets = head & 0x7FFFFFFF
        triangles = _u32(data, 4)
        refs = _u32(data, 8)
        rec_at = 12
        tri_at = rec_at + 16 * meshlets
        ref_at = tri_at + 4 * triangles
        end = ref_at + 2 * refs
        if end > len(data):
            return False
        vbase = tbase = 0
        for m in range(meshlets):
            w0, w1, _, _ = struct.unpack_from("<4I", data, rec_at + 16 * m)
            vcount = (w0 & 0x7F) + 1
            tcount = ((w0 >> 7) & 0x7F) + 1
            if (w1 & 0xFFFF) != (vbase & 0xFFFF) or (w1 >> 16) != (tbase & 0xFFFF):
                return False
            if tbase + tcount > triangles or vbase + vcount > refs:
                return False
            vbase += vcount
            tbase += tcount
        return vbase == refs and tbase == triangles
    except Exception:
        return False

def iter_archive_streams(path):
    """Yield (stream_offset, decompressed_data) for every zlib stream that is a mesh page."""
    with open(path, "rb") as fh:
        if not os.fstat(fh.fileno()).st_size:
            return
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = struct.unpack_from("<I", m, 4)[0] if size >= 8 else size
            if pos > size:
                raise ValueError(f"{path}: header data offset {pos} is past the end of the file ({size} bytes)")
            while pos < size:
                match = ZHDR.search(m, pos)
                if not match:
                    break
                s = match.start()
                try:
                    d = zlib.decompressobj()
                    chunks = []
                    p = s
                    while not d.eof and p < size:
                        block = m[p:p + (1 << 20)]
                        chunks.append(d.decompress(block))
                        p += len(block)
                        if d.unused_data or d.eof:
                            break
                    out = b"".join(chunks) + d.flush()
                except zlib.error:
                    pos = s + 1
                    continue
                if not d.eof or len(out) < 64:
                    pos = s + 1
                    continue
                if is_mesh_page(out):
                    yield f"0x{s:x}", out
                pos = p - len(d.unused_data)
        finally:
            m.close()

def parse_4240_block(data, at, stream_end):
    """Parse a `42 40 00 00` vertex block. Returns (record, end_offset)."""
    if at + HEADER_SIZE > len(data):
        raise ValueError(f"block header truncated at +0x{at:x}")
    words = struct.unpack_from("<24I", data, at)
    if words[0] != OTHER_BLOCK_MAGIC:
        raise ValueError(f"no 4240 block magic at +0x{at:x} (got 0x{words[0]:08x})")
    
    # Parse slot words: low byte = tag, high 24 bits = offset/2
    slots = {i: (w & 0xFF, (w >> 8) * 2) for i, w in enumerate(words[:SLOT_COUNT]) if w}
    
    record = {
        "kind": "4240_block",
        "offset": at,
        "flags": words[23],
        "slots": [[i, tag, off] for i, (tag, off) in sorted(slots.items())],
        "slot_count": len(slots),
        "slot_tags": {i: tag for i, (tag, _) in slots.items()},
        "slot_offsets": {i: off for i, (_, off) in slots.items()},
    }
    
    # Compute strides from slot offsets
    if slots:
        ordered = sorted(slots.items(), key=lambda x: x[1][1])
        for j in range(len(ordered) - 1):
            idx1, (tag1, off1) = ordered[j]
            idx2, (tag2, off2) = ordered[j + 1]
            stride = off2 - off1
            if stride > 0:
                record.setdefault("strides", []).append({
                    "from_slot": idx1,
                    "from_tag": tag1,
                    "from_offset": off1,
                    "to_slot": idx2,
                    "to_tag": tag2,
                    "to_offset": off2,
                    "stride": stride,
                })
        
        # Also compute stride from slot 0 (assumed position) to each subsequent slot
        if 0 in slots and slots[0][1] == HEADER_SIZE:
            base_off = HEADER_SIZE
            for idx, (tag, off) in slots.items():
                if idx != 0 and off > base_off:
                    stride = off - base_off
                    record.setdefault("strides_from_slot0", []).append({
                        "slot": idx,
                        "tag": tag,
                        "offset": off,
                        "stride": stride,
                    })
    
    return record, at + HEADER_SIZE

def process_archive(archive_path, limit=None):
    """Process an RPF9 archive and return stats on 4240 blocks."""
    stats = {
        "archive": archive_path,
        "streams": 0,
        "blocks_4240": 0,
        "blocks_6330": 0,
        "groups": 0,
        "block_formats": {},
        "strides": {},
        "strides_from_slot0": {},
        "slot_tag_counts": {},
        "slot_offsets": {},
        "errors": 0,
    }
    
    for stream_name, stream_data in iter_archive_streams(archive_path):
        if limit and stats["streams"] >= limit:
            break
        stats["streams"] += 1
        
        # Walk stream
        pos = 0
        while pos < len(stream_data):
            pos = (pos + 15) & ~15
            if pos >= len(stream_data):
                break
            while pos + 4 <= len(stream_data) and _u32(stream_data, pos) == 0:
                pos += 4
            if pos >= len(stream_data):
                break
            pos &= ~15
            if pos + 4 > len(stream_data):
                break
            word = _u32(stream_data, pos)
            
            if word == OTHER_BLOCK_MAGIC:
                try:
                    record, pos = parse_4240_block(stream_data, pos, len(stream_data))
                    stats["blocks_4240"] += 1
                    
                    # Record slot tags
                    for slot_idx, tag in record["slot_tags"].items():
                        stats["slot_tag_counts"][(slot_idx, tag)] = stats["slot_tag_counts"].get((slot_idx, tag), 0) + 1
                    
                    # Record strides between consecutive slots
                    if "strides" in record:
                        for s in record["strides"]:
                            key = (s["from_slot"], s["from_tag"], s["to_slot"], s["to_tag"], s["stride"])
                            stats["strides"][key] = stats["strides"].get(key, 0) + 1
                    
                    # Record strides from slot 0
                    if "strides_from_slot0" in record:
                        for s in record["strides_from_slot0"]:
                            key = (s["slot"], s["tag"], s["stride"])
                            stats["strides_from_slot0"][key] = stats["strides_from_slot0"].get(key, 0) + 1
                    
                    # Format signature
                    fmt_sig = tuple(sorted(record["slot_tags"].items()))
                    stats["block_formats"][fmt_sig] = stats["block_formats"].get(fmt_sig, 0) + 1
                    
                except ValueError as e:
                    # Advance past the rejected block: staying on the same word parsed it again for ever.
                    stats["errors"] += 1
                    stats.setdefault("first_error", str(e))
                    pos += 4
            elif word == BLOCK_MAGIC:
                stats["blocks_6330"] += 1
                pos += HEADER_SIZE
            elif word & 0x80000000:
                head = _u32(stream_data, pos)
                meshlets = head & 0x7FFFFFFF
                triangles = _u32(stream_data, pos + 4)
                refs = _u32(stream_data, pos + 8)
                rec_at = pos + 12
                tri_at = rec_at + 16 * meshlets
                ref_at = tri_at + 4 * triangles
                end = ref_at + 2 * refs
                if end <= len(stream_data):
                    pos = end
                else:
                    pos += 4
                stats["groups"] += 1
            else:
                pos += 4
    
    return stats

def positive_int(text):
    """--limit: 0 is falsy and scanned everything, and a negative number scanned one stream."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze 42 40 00 00 vertex blocks in RPF9 archives")
    parser.add_argument("inputs", nargs="+", help="RPF9 archive paths")
    parser.add_argument("--json", help="Write results to JSON file")
    parser.add_argument("--limit", type=positive_int, help="Limit streams per archive")
    args = parser.parse_args()
    # Checked before the scan: the report is written after every archive has been read, so a --json naming one of
    # them replaced the archive with the report, and a folder only failed once the whole scan was done.
    if args.json:
        if os.path.isdir(args.json):
            print(f"ERROR: --json {args.json} is a folder, not a file", file=sys.stderr)
            return 2
        for path in args.inputs:
            if same_report_path(args.json, path):
                print(f"ERROR: --json {args.json} would overwrite the input archive {path}", file=sys.stderr)
                return 2

    all_stats, status = [], 0
    for path in args.inputs:
        if not os.path.exists(path):
            # A missing input is a usage error (2); a folder that exists but is not an archive is a read error (1).
            print(f"ERROR: {path}: does not exist", file=sys.stderr)
            status = max(status, 2)
            continue
        if not os.path.isfile(path):
            # A folder passed for an archive used to reach open() and end the whole run with PermissionError.
            print(f"ERROR: {path} is not a file", file=sys.stderr)
            status = max(status, 1)
            continue
        print(f"Processing {path}...")
        try:
            stats = process_archive(path, limit=args.limit)
        except ValueError as e:
            # A truncated archive or a bad data offset otherwise looks like an empty successful scan.
            print(f"ERROR: {e}", file=sys.stderr)
            status = 1
            continue
        except OSError as e:
            print(f"ERROR: cannot read {path}: {e.strerror or e}", file=sys.stderr)
            status = max(status, 1)
            continue
        all_stats.append(stats)
        
        print(f"  Streams: {stats['streams']}")
        print(f"  4240 blocks: {stats['blocks_4240']}")
        print(f"  6330 blocks: {stats['blocks_6330']}")
        print(f"  Groups: {stats['groups']}")
        print(f"  Unique formats: {len(stats['block_formats'])}")
        print(f"  Slot tag counts: {sorted(stats['slot_tag_counts'].items())}")
        print(f"  Strides (consecutive): {sorted(stats['strides'].items())}")
        print(f"  Strides from slot 0: {sorted(stats['strides_from_slot0'].items())}")
        
        # Show top formats
        for fmt, count in sorted(stats['block_formats'].items(), key=lambda x: -x[1])[:10]:
            tags = [f"slot{i}:tag{tag}" for i, tag in fmt]
            print(f"  Format ({count}x): {', '.join(tags)}")
    
    if args.json:
        output = []
        for s in all_stats:
            s_copy = dict(s)
            s_copy["block_formats"] = {str(k): v for k, v in s["block_formats"].items()}
            s_copy["strides"] = {str(k): v for k, v in s["strides"].items()}
            s_copy["strides_from_slot0"] = {str(k): v for k, v in s["strides_from_slot0"].items()}
            s_copy["slot_tag_counts"] = {str(k): v for k, v in s["slot_tag_counts"].items()}
            output.append(s_copy)
        parent = os.path.dirname(os.path.abspath(args.json))
        try:
            if parent:      # every archive has been read by now; a missing folder used to lose the run
                os.makedirs(parent, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(output, f, indent=2)
        except OSError as e:
            print(f"ERROR: cannot write {args.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return status


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
