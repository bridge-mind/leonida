"""Read the pixel-shader input table from an FSSM blob.

FSSM is the precompiled RDNA2 program that the Xbox shader container carries.  For pixel shaders a
table of 8-byte input records starts at offset 0x138.  A record is present while its first byte is 6:

    byte 0      record kind (6 for an input)
    byte 1      semantic id
    byte 2      component mask (bit 0 = x, bit 3 = w)
    byte 3      running index (same semantic id, next index)
    bytes 4-5   unused here
    byte 6      parameter slot
    byte 7      flags, the read method: bit 0 (0x01) read through v_interp_mov_f32 (flat),
                bit 1 (0x02) through v_interp_p1/p2 or the f16 forms (interpolated), bit 3
                (0x08) a Float16-typed parameter (#1256), bit 4 (0x10) a 16-bit interpolant

The record's position in the table is the `v_interp attr<N>` number the code uses, not byte 6 - 1.

`directory_ok` checks the section-directory identity that most blobs follow:

    u32 at 0x120 (offset) + u32 at 0x12C (size) == u32 at 0x130 (next offset)

All facts above are from the format description supplied with this tool.  The optional stage lookup
reads the carve index (`<fssm dir>/../index.json`) only to report how many *pixel* shaders have a
table; the decoder itself needs no index.

On the local 26,503-blob corpus the directory identity held for 25,643 blobs (the 860 failures all
carry 0xDEADBEEF at 0x130), and 12,101 blobs were pixel shaders, 10,307 of them with a non-empty
table (55,791 records, 11,199 of them 16-bit).  The table count equals the number of distinct
`v_interp attr` numbers in 11,316/12,101 shaders (93.5%); 300 more declare exactly one unread input,
475 declare two or more, and the 10 shaders whose table starts with a non-6 record are the only ones
that undercount.  A 200-PS spot check agreed on 187/200.  Flag bit 4 matched the 16-bit interpolant
opcodes in 12,067/12,101 (99.7%).  `rdna2spirv.py` consumes it in the PS input path (pixel_interpolants:
every table position declared as a fragment input, 16-bit-flagged positions
rounded f32->f16->f32 on 32-bit reads).

Read method (SHADER-181).  Bits 0-1 say how the compiled code reads each input, not the D3D
interpolation mode (that is PSV0's, decoded by `tools/ps_interp_modes.py`, #1256).  On the same
corpus exactly one of the two is set: bit 1 in 51,191 records, bit 0 in 4,600, neither or both in
none.  They agree with the code in all 12,101 PS: every bit-0 position the code reads is read only
by `v_interp_mov_f32` (4,295 read positions), every bit-1 position only by `v_interp_p1/p2` or the
f16 forms (49,594), and no position is read both ways.  The FSSM stores no SPI_PS_INPUT_ENA/ADDR
word.  Bit 3 (0x08) is set in 11,619 records.

The header flags word at offset 0xD4 records the compiled wave size: bit 5 (0x20) is 1 for wave32
and 0 for wave64 (`wave_size()`).  On the local 26,503-blob corpus every blob starts with `FSSM`
and is at least 0xD8 bytes long: 2,662 are wave32 (1,535 CS, 1,127 PS) and 23,841 are wave64
(1,154 CS, 195 GS, 9,282 MS, 10,974 PS, 2,236 VS), with zero contradictions against GFX10
code-proven wave32/wave64 instructions across all five stages.

The post-`code_end` tail carries a binding table.  After 16 bytes of `0xBF9F0000` filler and an
8-byte header, 20-byte records run until a `0xFFFFFFFF` sentinel; then `u32 0`, `u16 0`, `u16`
name length and the name.  `binding_table()` returns those records.  On the local 26,503-blob
corpus every blob parses (5,640,704 records), and the tail name is a substring of the container
name in all 26,503 cases, exactly equal in 691.

Usage:
    python -I fssm_header.py <fssm dir or file> [--json] [--limit N] [--bindings]

Prints one line per blob and a corpus tally:
    N files, directory_ok X, PS with table Y, entries Z
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

MAGIC = b"FSSM"
TABLE_OFFSET = 0x138
RECORD_SIZE = 8
RECORD_KIND = 6
FLAG_FP16 = 0x10
READ_MASK = 0x03
READ_MOV = 0x01      # v_interp_mov_f32: flat
READ_INTERP = 0x02   # v_interp_p1/p2 and the f16 forms: interpolated with the v0/v1 pair
READ_NAMES = {READ_MOV: "mov", READ_INTERP: "interp"}
DIRECTORY_OFFSET = 0x120
DIRECTORY_SIZE = 0x12C
DIRECTORY_NEXT = 0x130
CODE_OFFSET = 0xC0
CODE_SIZE = 0xC8
WAVE_FLAGS_OFFSET = 0xD4
FLAG_WAVE32 = 0x20
CODE_END = 0xE8
CODE_END_UNSET = 0xDEADBEEFDEADBEEF
TAIL_FILLER = 0xBF9F0000
TAIL_PAD = 16
TAIL_HEADER = 8
BINDING_SIZE = 20
BINDING_SENTINEL = 0xFFFFFFFF
MAX_BINDINGS = 100_000


def _u32(blob: bytes, offset: int):
    return struct.unpack_from("<I", blob, offset)[0]


def _u64(blob: bytes, offset: int):
    return struct.unpack_from("<Q", blob, offset)[0]


def pixel_inputs(blob):
    """Return the pixel-input records as dicts, in table order.

    Keys: semantic, mask, index, slot, flags.  An empty list means the blob has no table (the byte
    at 0x138 is not 6) or the blob is too short to hold one.
    """
    out = []
    offset = TABLE_OFFSET
    while offset + RECORD_SIZE <= len(blob) and blob[offset] == RECORD_KIND:
        record = blob[offset:offset + RECORD_SIZE]
        out.append({
            "semantic": record[1],
            "mask": record[2],
            "index": record[3],
            "slot": record[6],
            "flags": record[7],
        })
        offset += RECORD_SIZE
    return out


def interpolants(blob):
    """Return (count, fp16_mask) where bit i of fp16_mask is set when table position i is 16-bit."""
    inputs = pixel_inputs(blob)
    mask = 0
    for position, entry in enumerate(inputs):
        if entry["flags"] & FLAG_FP16:
            mask |= 1 << position
    return len(inputs), mask


def read_methods(blob):
    """Return the read method (flags bits 0-1) of each pixel input, in table order.

    READ_MOV (1): read through v_interp_mov_f32, a flat input.  READ_INTERP (2): read through
    v_interp_p1/p2 or the f16 forms, interpolated with the v0/v1 pair.  0 and 3 (neither or both
    bits) never occur in the corpus.
    """
    return [entry["flags"] & READ_MASK for entry in pixel_inputs(blob)]


def read_name(method):
    return READ_NAMES.get(method, f"read{method}")


def directory_ok(blob):
    """True when offset + size == next offset in the section directory (0x120/0x12C/0x130)."""
    if len(blob) < DIRECTORY_NEXT + 4:
        return False
    return _u32(blob, DIRECTORY_OFFSET) + _u32(blob, DIRECTORY_SIZE) == _u32(blob, DIRECTORY_NEXT)


def wave_size(blob):
    """Return the compiled wave size (32 or 64) from u32[0xD4] bit 5, or None.

    Bit 5 (0x20) is set for wave32 and clear for wave64.  Returns None when the blob is
    shorter than 0xD8 bytes or does not begin with the FSSM magic (for example, synthetic
    zero-prefixed compute headers).
    """
    if len(blob) < WAVE_FLAGS_OFFSET + 4 or blob[:4] != MAGIC:
        return None
    return 32 if (_u32(blob, WAVE_FLAGS_OFFSET) & FLAG_WAVE32) else 64


def binding_table(blob):
    """Return the post-code binding table, or None when the tail does not match the layout.

    The table starts at `code_end`, or at `code_off + code_size` when `code_end` is the
    0xDEADBEEFDEADBEEF sentinel:

        +0x00  4 x 0xBF9F0000 (s_code_end filler)
        +0x10  u32 header (4, 3, or 0 on the corpus), u32 zero
        +0x18  20-byte records until a u32 0xFFFFFFFF
        at the sentinel:
        +0x04  u32 0
        +0x08  u16 0, u16 name length
        +0x0C  name bytes

    Keys: code_end, header, records (5-word tuples), name, trailing.  `trailing` counts the
    bytes after the name, which are not parsed here.  None means the tail is shorter than the
    layout, lacks the filler or sentinel, or has a malformed length or prefix.
    """
    if len(blob) < CODE_END + 8:
        return None
    code_end = _u64(blob, CODE_END)
    if code_end == CODE_END_UNSET:
        code_end = _u64(blob, CODE_OFFSET) + _u64(blob, CODE_SIZE)
    start = code_end + TAIL_PAD
    if code_end > len(blob) or start + TAIL_HEADER > len(blob):
        return None
    if blob[code_end:start] != struct.pack("<I", TAIL_FILLER) * 4:
        return None
    header = _u32(blob, start)
    offset = start + TAIL_HEADER
    records = []
    while offset + BINDING_SIZE <= len(blob) and _u32(blob, offset) != BINDING_SENTINEL:
        if len(records) >= MAX_BINDINGS:
            return None
        records.append(struct.unpack_from("<5I", blob, offset))
        offset += BINDING_SIZE
    if offset + 12 > len(blob) or _u32(blob, offset) != BINDING_SENTINEL:
        return None
    zero, pad, length = struct.unpack_from("<IHH", blob, offset + 4)
    if zero != 0 or pad != 0 or not 0 < length <= 512 or offset + 12 + length > len(blob):
        return None
    raw_name = blob[offset + 12:offset + 12 + length]
    if not all(0x20 <= byte < 0x7F for byte in raw_name):
        return None
    name = raw_name.decode("latin-1")
    return {
        "code_end": code_end,
        "header": header,
        "records": records,
        "name": name,
        "trailing": len(blob) - (offset + 12 + length),
    }


def stage_index(path):
    """Map shader stems to stages from the carve index, or {} when it is not next to the corpus.

    The index is the one `make_game.py` writes next to the carved `.dxbc` files.  Both the `fssm`
    directory (parent holds index.json) and a directory that contains index.json are accepted.
    Prefer the local index when it has stage rows.
    """
    base = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
    for candidate in (os.path.join(base, "index.json"), os.path.join(base, "..", "index.json")):
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as f:
                    rows = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(rows, list):
                continue
            stages = {}
            ambiguous = set()
            for row in rows:
                if not (
                    isinstance(row, dict)
                    and isinstance(file := row.get("file"), str)
                    and (stem := os.path.splitext(os.path.basename(file.replace("\\", "/")))[0])
                    and isinstance(stage := row.get("stage"), str)
                    and stage
                ):
                    continue
                if stem in ambiguous:
                    continue
                prev = stages.get(stem)
                if prev is not None and prev != stage:
                    del stages[stem]
                    ambiguous.add(stem)
                else:
                    stages[stem] = stage
            if stages:
                return stages
    return {}


def files_of(path):
    if os.path.isfile(path):
        return [path]
    return [
        full
        for name in sorted(os.listdir(path))
        if name.endswith(".fssm") and os.path.isfile(full := os.path.join(path, name))
    ]


def summarize(path, stages, bindings=False):
    """Decode one blob and return the row used by the text and JSON output."""
    with open(path, "rb") as f:
        blob = f.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    count, fp16 = interpolants(blob)
    row = {
        "name": os.path.basename(path),
        "stage": stages.get(stem),
        "directory_ok": directory_ok(blob),
        "wave_size": wave_size(blob),
        "entries": count,
        "fp16_mask": fp16,
        "inputs": pixel_inputs(blob),
    }
    if bindings:
        table = binding_table(blob)
        row["bindings"] = None if table is None else {
            "header": table["header"],
            "records": len(table["records"]),
            "name": table["name"],
            "trailing": table["trailing"],
        }
    return row


def tally(rows, stages_known):
    if stages_known:
        ps = [row for row in rows if row["stage"] == "ps"]
        counts = Counter(row["entries"] for row in ps)
        scope = "ps"
    else:
        # Without a stage index, a non-empty table is the only PS signal the blob carries.
        ps = [row for row in rows if row["entries"]]
        # Keep the histogram in the same scope as ps_with_table/entries: non-PS blobs have no table
        # and must not show up as a 0:N bucket in the reported corpus counts.
        counts = Counter(row["entries"] for row in ps)
        scope = "all"
    methods = Counter(read_name(entry["flags"] & READ_MASK) for row in ps for entry in row["inputs"])
    waves = Counter(f"wave{ws}" for row in rows if (ws := row.get("wave_size")) is not None)
    return {
        "files": len(rows),
        "directory_ok": sum(row["directory_ok"] for row in rows),
        "ps_with_table": sum(1 for row in ps if row["entries"]),
        "entries": sum(row["entries"] for row in ps),
        "entry_counts": dict(sorted(counts.items())),
        "entry_counts_scope": scope,
        "read_methods": dict(sorted(methods.items())),
        "wave_sizes": dict(sorted(waves.items())),
        "stage_source": "index.json" if stages_known else "table-only",
    }


def non_negative(text):
    """--limit: a count of blobs. A negative one would slice from the end (paths[:-1] drops the last blob)."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read the FSSM pixel-shader input table.")
    ap.add_argument("path", help="an .fssm file or a directory of them")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of lines")
    ap.add_argument("--limit", type=non_negative, help="stop after this many blobs")
    ap.add_argument("--bindings", action="store_true", help="also parse the post-code binding table")
    args = ap.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        return 2
    try:
        paths = files_of(args.path)
    except OSError as e:
        print(f"ERROR: cannot list {args.path}: {e.strerror or e}", file=sys.stderr)
        return 2
    if args.limit is not None:
        paths = paths[:args.limit]
    stages = stage_index(args.path)
    try:
        rows = [summarize(path, stages, args.bindings) for path in paths]
    except OSError as e:
        print(f"ERROR: cannot read {e.filename or args.path}: {e.strerror or e}", file=sys.stderr)
        return 2
    stages_known = any(row.get("stage") for row in rows) if rows else bool(stages)
    summary = tally(rows, stages_known)

    if args.json:
        json.dump({"files": rows, "tally": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    for row in rows:
        stage = row["stage"] or "?"
        wave = row["wave_size"] if row["wave_size"] is not None else "?"
        print(f"{row['name']}: stage={stage} directory_ok={row['directory_ok']} "
              f"entries={row['entries']} fp16=0x{row['fp16_mask']:x} wave={wave}")
        if args.bindings:
            table = row["bindings"]
            if table is None:
                print("  bindings unparsed")
            else:
                print(f"  bindings header={table['header']} records={table['records']} "
                      f"name_len={len(table['name'])} trailing={table['trailing']}")
    total = summary
    print(f"{total['files']} files, directory_ok {total['directory_ok']}, "
          f"PS with table {total['ps_with_table']}, entries {total['entries']}")
    if total["wave_sizes"]:
        print("wave sizes: " + ", ".join(f"{k}:{v}" for k, v in total["wave_sizes"].items()))
    if total["read_methods"]:
        print("read methods: " + ", ".join(f"{k}:{v}" for k, v in total["read_methods"].items()))
    label = "PS entry counts" if total["entry_counts_scope"] == "ps" else "entry counts"
    print(label + ": " + ", ".join(f"{k}:{v}" for k, v in total["entry_counts"].items()))
    if total["stage_source"] == "table-only":
        why = "no inspected blob in index.json" if stages else "no index.json found"
        print(f"({why}: counted every blob with a non-empty table as a PS)")
    if args.bindings:
        parsed = [row for row in rows if row["bindings"]]
        records = sum(row["bindings"]["records"] for row in parsed)
        exact = sum(1 for row in parsed
                    if os.path.splitext(row["name"])[0].casefold() == row["bindings"]["name"].casefold())
        print(f"binding tables {len(parsed)}, records {records}, tail name equals file stem {exact}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
