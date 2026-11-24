"""Read the PAL stage descriptors that head an FSSM blob.

An FSSM header carries a table of fixed-size hardware-stage records at offset 0x100.  Record k starts
at 0x100 + 0x88 * k and its first qword is

    0x000000C0_0002_AAHH

    HH  byte 0  PAL hardware stage
    AA  byte 1  API stage

Measured over all 26,503 blobs of the 1.0.2.1 corpus (the "stage" column is the carve index's, which
the record's API id matches in every case):

    placement    0x100, stride 0x88; the table ends at the first zero or non-tag qword
    count        26,308 blobs have one record, 195 have two (all GS: the first is GS, the second its
                 VS copy at 0x188)
    API id       0 vs 2,236, 1 ps 12,101, 2 gs 195, 3 cs 2,689, 6 ms 9,282
    hardware id  3 gs and 4 vs for the 195 GS pairs, 4 vs 2,236, 5 ps 12,101, 6 cs 2,689, 7 the
                 9,282 mesh shaders, 2 es in the first XBCH payload of the 195 two-payload VS
                 containers (their second payload is the 4 vs one)
    sentinel     860 blobs (1,055 records: the 195 GS pairs included) fill unused fields with paired
                 qwords 0xDEADBEEFDEADBEEF; a lone 32-bit 0xDEADBEEF never occurs there.  The other
                 25,643 records store zeros.  Those 860 are the blobs whose 0x120/0x12C/0x130
                 section-directory words are 0xDEADBEEF.

The hardware ids 0 and 1 and the API ids 4 and 5 are not in this corpus.  The hardware-stage names
(ls, hs, es, gs, vs, ps, cs, ngg) come from an outside project's description of the id order; only the
numeric correlation with the carve-index stage is measured here.  The rest of a record is not decoded.

Usage:
    python -I pal_stages.py <fssm file, dxbc file, or folder of either> [--json] [--limit N]

A DXBC container is read through `fssm_unpack.unpack_all`, so every tag-0x0D payload is reported (195
containers have two).  `--limit` counts files, not payloads.
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fssm_header  # noqa: E402
import fssm_unpack  # noqa: E402

MAGIC = b"FSSM"
TABLE_OFFSET = 0x100
RECORD_SIZE = 0x88
CODE_OFFSET = 0xC0
TAG_MASK = 0xFFFFFFFFFFFF0000
TAG = 0x000000C000020000
SENTINEL32 = 0xDEADBEEF
SENTINEL64 = 0xDEADBEEFDEADBEEF
MAX_RECORDS = 16

# Low byte of the tag; 0 and 1 are not in the corpus.
HARDWARE_STAGES = {0: "ls", 1: "hs", 2: "es", 3: "gs", 4: "vs", 5: "ps", 6: "cs", 7: "ngg"}
# Second byte of the tag; 4 and 5 are not in the corpus.
API_STAGES = {0: "vs", 1: "ps", 2: "gs", 3: "cs", 6: "ms"}


def _u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def header_end(blob):
    """End of the header region: the code offset when it is set and in range, else the blob length."""
    if len(blob) >= CODE_OFFSET + 8:
        code_off = _u64(blob, CODE_OFFSET)
        if 0 < code_off <= len(blob):
            return code_off
    return len(blob)


def sentinels(record):
    """(aligned 64-bit 0xDEADBEEFDEADBEEF pairs, lone 32-bit 0xDEADBEEF words) in one record."""
    pairs = [offset for offset in range(0, len(record) - 7, 8) if _u64(record, offset) == SENTINEL64]
    covered = {offset + half for offset in pairs for half in (0, 4)}
    words = [offset for offset in range(0, len(record) - 3, 4) if _u32(record, offset) == SENTINEL32]
    return len(pairs), sum(1 for offset in words if offset not in covered)


def stage_descriptors(blob):
    """Return the stage records of an FSSM blob, in table order, as dicts.

    Keys: index, offset, stride, tag, api_id, api_stage, hardware_id, hardware_stage, sentinel_words
    (paired 0xDEADBEEFDEADBEEF qwords) and lone_sentinel_words (32-bit 0xDEADBEEF not part of a
    pair).  An unknown id maps to None.  An empty list means the blob lacks the FSSM magic or does not
    start with a tag.
    """
    if len(blob) < TABLE_OFFSET + 8 or blob[:4] != MAGIC:
        return []
    end = min(header_end(blob), len(blob))
    out = []
    for index in range(MAX_RECORDS):
        offset = TABLE_OFFSET + RECORD_SIZE * index
        if offset + 8 > end:
            break
        tag = _u64(blob, offset)
        if tag == 0 or (tag & TAG_MASK) != TAG:
            break
        api_id, hardware_id = (tag >> 8) & 0xFF, tag & 0xFF
        paired, lone = sentinels(blob[offset:min(offset + RECORD_SIZE, end)])
        out.append({
            "index": index,
            "offset": offset,
            "stride": RECORD_SIZE,
            "tag": tag,
            "api_id": api_id,
            "api_stage": API_STAGES.get(api_id),
            "hardware_id": hardware_id,
            "hardware_stage": HARDWARE_STAGES.get(hardware_id),
            "sentinel_words": paired,
            "lone_sentinel_words": lone,
        })
    return out


def api_stage(blob):
    """API stage name of the blob's first record ("vs", "ps", "gs", "cs", "ms"), or None."""
    records = stage_descriptors(blob)
    return records[0]["api_stage"] if records else None


def hardware_stage(blob):
    """PAL hardware stage name of the blob's first record, or None."""
    records = stage_descriptors(blob)
    return records[0]["hardware_stage"] if records else None


def files_of(path):
    """Files to read: every .fssm, plus a .dxbc whose stem has no .fssm beside it."""
    if os.path.isfile(path):
        return [path]
    with os.scandir(path) as entries:
        names = sorted(entry.path for entry in entries if entry.is_file())
    fssm = [name for name in names if name.endswith(".fssm")]
    taken = {os.path.splitext(os.path.basename(name))[0] for name in fssm}
    dxbc = [name for name in names
            if name.endswith(".dxbc") and os.path.splitext(os.path.basename(name))[0] not in taken]
    return sorted(fssm + dxbc)


def blob_rows(path, limit=None):
    """One row per FSSM blob in a file or folder: a bare FSSM, or every tag-0x0D payload of a container."""
    rows = []
    for full in files_of(path)[:limit]:
        name = os.path.basename(full)
        try:
            with open(full, "rb") as f:
                data = f.read()
            if data[:4] == MAGIC:
                rows.append({"name": name, "payload": "fssm", "blob": data})
                continue
            blobs = fssm_unpack.unpack_all(data)
        except (OSError, ValueError, struct.error) as e:
            rows.append({"name": name, "payload": None, "error": str(e)})
            continue
        if not blobs:
            rows.append({"name": name, "payload": None, "error": "no tag-0x0D payload"})
        for index, blob in enumerate(blobs):
            rows.append({"name": name, "payload": f"xdat{index}", "blob": blob})
    return rows


def summarize(row, stage=None):
    """Decode one blob_rows() row for the text and JSON output."""
    out = {"name": row["name"], "payload": row["payload"], "stage": stage}
    if "blob" not in row:
        out["error"] = row["error"]
        return out
    records = stage_descriptors(row["blob"])
    out["descriptors"] = records
    out["api_stage"] = records[0]["api_stage"] if records else None
    out["hardware_stage"] = records[0]["hardware_stage"] if records else None
    return out


def tally(rows):
    decoded = [row for row in rows if "descriptors" in row]
    records = [record for row in decoded for record in row["descriptors"]]
    known = [row for row in decoded if row["stage"] and row["api_stage"]]
    sentinel_rows = sum(1 for row in decoded if any(r["sentinel_words"] for r in row["descriptors"]))
    return {
        "blobs": len(decoded),
        "errors": sum(1 for row in rows if "error" in row),
        "descriptor_counts": dict(sorted(Counter(len(row["descriptors"]) for row in decoded).items())),
        "api_ids": dict(sorted(Counter(r["api_id"] for r in records).items())),
        "hardware_ids": dict(sorted(Counter(r["hardware_id"] for r in records).items())),
        "sentinel_blobs": sentinel_rows,
        "lone_sentinel_words": sum(r["lone_sentinel_words"] for r in records),
        "stage_known": len(known),
        "stage_matches": sum(1 for row in known if row["stage"] == row["api_stage"]),
    }


def positive_count(text):
    """--limit: a count of files, at least 1. 0 used to select no file at all and report an empty run as
    a clean one; a negative one sliced from the end of the folder listing."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read the PAL stage descriptors in FSSM headers.")
    ap.add_argument("path", help="an .fssm file, a DXBC container, or a folder of either")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of lines")
    ap.add_argument("--limit", type=positive_count, help="stop after this many files (at least 1)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        return 2
    try:
        rows = blob_rows(args.path, args.limit)
    except OSError as e:
        print(f"ERROR: cannot list {args.path}: {e.strerror or e}", file=sys.stderr)
        return 2
    stages = fssm_header.stage_index(args.path)
    decoded = []
    for row in rows:
        stem = os.path.splitext(row["name"])[0]
        decoded.append(summarize(row, stages.get(stem)))
    summary = tally(decoded)

    if args.json:
        json.dump({"blobs": decoded, "tally": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    for row in decoded:
        label = f"[{row['payload']}]" if row["payload"] else ""
        if "error" in row:
            print(f"{row['name']} {label}: ERROR {row['error']}")
            continue
        print(f"{row['name']} {label}: stage={row['stage'] or '?'} "
              f"descriptors={len(row['descriptors'])} stride=0x{RECORD_SIZE:x}")
        for record in row["descriptors"]:
            lone = f" lone_sentinel={record['lone_sentinel_words']}" if record["lone_sentinel_words"] else ""
            print(f"  +0x{record['offset']:03x} tag=0x{record['tag']:016x} "
                  f"api={record['api_id']} ({record['api_stage'] or '?'}) "
                  f"hw={record['hardware_id']} ({record['hardware_stage'] or '?'}) "
                  f"sentinel={record['sentinel_words']}{lone}")
    total = summary
    print(f"{total['blobs']} blobs, errors {total['errors']}, "
          f"descriptors per blob {total['descriptor_counts'] or '{}'}")
    print("api ids: " + ", ".join(f"{k}:{v}" for k, v in total["api_ids"].items()))
    print("hardware ids: " + ", ".join(f"{k}:{v}" for k, v in total["hardware_ids"].items()))
    print(f"sentinel blobs {total['sentinel_blobs']}, lone 32-bit sentinel words "
          f"{total['lone_sentinel_words']}")
    if total["stage_known"]:
        print(f"API stage matches the index: {total['stage_matches']}/{total['stage_known']}")
    else:
        print("(no index.json found: no stage to compare the API id with)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
