"""Read mission-state tables: 32-byte state records and the 64-byte pool elements they point at.

`docs/mission_logic.md` describes the tables that carry mission structure in the PSIN streams of the script
archives. This tool is the reader for them. It takes inflated PSIN files (a directory of carved `.bin`
streams works too) and prints, per file, the number of records, the distinct state keys, the distinct
archetype and link hashes, and every entity it could resolve to a world transform.

Record, 32 bytes:

    +0   u32 key          joaat of the state name, lower-cased first
    +4   u32 0
    +8   u32 colB         an ascending band-like column; 1 on every record measured here
    +12  u32 0
    +16  u32 (u16 kind at the low half, u16 offset at the high half)
    +20  u32 band         an ascending band counter, not always zero (0, 0, 1, 2, 2, ...)
    +24  u32 (u16 n, u16 n)   the entity count, stored twice
    +28  u32 0

The offset is a byte offset. It is stored in the **high half** of the dword at +16, which is the big end of
the little-endian word; the low half is `kind`. The two u16s are not byte-swapped: the measured bytes
`03 00 80 07` resolve kind 3 and offset 0x780, not 0x8007. A record whose offset is outside the file is
still a record; it simply resolves to no entity.

On PSIN input the two dwords at +16 and +20 are the halves of the container's u64 pointer: the low 20
bits are a 1-based block number (the tool reads its low 16 bits as `kind`) and the bits above are the
byte offset, split between `band` (the high dword) and `poolOffset` (the high half of the low dword).
That is what the block-indexed resolution below reads; on a bare `.bin` there is no block table to
index and only the first two resolutions apply.

Pool element, 64 bytes, 16 dwords:

    +0x00  u32 archetype/state hash
    +0x04  u32 0
    +0x08  u32 0
    +0x0C  u32 0
    +0x10  f32 pos.x, pos.y, pos.z     (world coordinates in RAGE space, Z up)
    +0x1C  f32 pos.x                   the first position component, duplicated
    +0x20  f32 rot.x, rot.y, rot.z
    +0x2C  f32 rot.x                   the first rotation component, duplicated
    +0x30  u32 hash                    repeats the archetype hash on most elements
    +0x34  u32 linkHash                joaat of the lower-cased linked name (a character, for example)
    +0x38  u32 0                       a second hash on some elements, otherwise zero
    +0x3C  u32 0                       zero on every element measured here

The duplicated first component is the cheap structural test that separates a real element from a random
64-byte window. This tool requires it for both the position and the rotation, and finite values in a
plausible world range. The last two dwords are not required to be zero: a measured minority of elements
carries a second link hash in the third-last dword. Requiring `+0x38 == +0x3C == 0` would silently drop
those entities, so the reader keeps them and reports the extra hash as `extra`.

Resolution is three-step, because the measured streams prove three encodings:

1. **File-absolute** (the common case): the record's offset is the byte address of the element.
2. **Linked pool** (the family documented as the "next even-tagged pool chunk" rule): inside a PSIN
   container, a block of records is followed by the block that holds its elements. The record offset is
   then a 16-byte-unit offset into that pool block: `element = pool_block_base + (offset >> 4)`. The
   measured family uses offsets such as 0x0, 0xC00, 0x1800, ... while its elements sit 0xC0 bytes apart.
3. **Block-indexed** (the container's own pointer form): `kind` is a 1-based block number, so
   `block = blocks[kind - 1]` and the byte offset is `(band << 12) | (offset >> 4)`. This names a pool
   even when the block is not the one after the record block, and it keeps the offset bits that do not
   fit in `offset` alone. The records of one chunk are slices of one array, so their slot stride is
   `(start[i + 1] - start[i]) / count[i]` when every slice agrees; a slot wider than 64 bytes then steps
   by that stride rather than by 64.

All three addresses are scored when they exist: the one that validates more of the record's `n` slots
wins, and a tie keeps the earlier reading (file-absolute, then linked pool, then block-indexed). A slot
that fails the duplicated-component test is counted as skipped, never reported as an entity. The scoring
matters on the pool-relative families, where an offset such as 0x3000 can also name an unrelated element
elsewhere in the file.

Detection is structural rather than schema-driven so a bare `.bin` works as well as a PSIN container. A
record chunk is a run of at least two consecutive 32-byte records with the same `kind`; a lone record is
kept when it has a hash-sized key, a small `colB`, and an offset that already names an element. For PSIN
input the block table is used to find the linked pool block (skip sub-64-byte blocks and other record
blocks, then take the first block that starts with a valid element). Bytes outside a record chunk and a
resolved entity are counted as skipped.

Usage:
    python -I mission_tables.py <file-or-dir> [--json] [--limit N]

Text output prints one line per file and one line per entity; `--json` writes `{files, summary}` with the
same numbers and the entity list (hashes as eight lowercase hex digits).
"""
import argparse
import collections
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import psin_index  # noqa: E402
except ImportError:  # a bare checkout without the sibling readers: scan raw bytes only
    psin_index = None
try:
    import psin_data  # noqa: E402
except ImportError:
    psin_data = None

RECORD_SIZE = 32
ELEMENT_SIZE = 64
MAX_ENTITIES = 4096        # a sane upper bound for the duplicated u16 count
MAX_WORLD = 1e7            # positions and rotations outside this are not world transforms
RECORD_KEY_MIN = 0x10000   # below this a u32 is a small field, not a joaat state key
POOL_SHIFT = 4             # linked-pool offsets are in 16-byte units


def joaat(text):
    """joaat32 of a lower-cased name (Jenkins one-at-a-time, seed 0)."""
    h = 0
    for byte in text.lower().encode("latin-1", "replace"):
        h = (h + byte) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def decode_record(blob, at):
    """Decode the 32-byte record at `at`, or None when the bytes are not a record.

    The only zero fields are +4, +12 and +28; `band` is deliberately not required to be zero (it is an
    ascending band counter, not a reserved word).
    """
    if at < 0 or at + RECORD_SIZE > len(blob):
        return None
    key, zero_a, col_b, zero_b, kind_offset, band, duplicated, zero_c = struct.unpack_from("<8I", blob, at)
    kind = kind_offset & 0xFFFF
    offset = kind_offset >> 16
    count_a, count_b = duplicated >> 16, duplicated & 0xFFFF
    if key < RECORD_KEY_MIN or zero_a or zero_b or zero_c:
        return None
    if count_a != count_b or count_a > MAX_ENTITIES:
        return None
    return {
        "offset": at,
        "key": key,
        "colB": col_b,
        "kind": kind,
        "poolOffset": offset,
        "band": band,
        "count": count_a,
    }


def decode_element(blob, at):
    """Decode the 64-byte pool element at `at`, or None when the layout does not hold."""
    if at < 0 or at + ELEMENT_SIZE > len(blob):
        return None
    words = struct.unpack_from("<16I", blob, at)
    if words[0] == 0 or words[1] or words[2] or words[3]:
        return None
    if words[4] != words[7] or words[8] != words[11]:
        return None
    values = struct.unpack_from("<8f", blob, at + 16)
    if not all(math.isfinite(v) and abs(v) <= MAX_WORLD for v in values):
        return None
    return {
        "offset": at,
        "archetype": words[0],
        "pos": list(values[0:3]),
        "rot": list(values[4:7]),
        "hash": words[12],
        "link": words[13],
        "extra": words[14],
    }


def pool_byte_offset(record):
    """The byte offset a PSIN block pointer carries: `band` holds its high bits, `offset` its low bits."""
    return (record["band"] << 12) | (record["poolOffset"] >> POOL_SHIFT)


def block_at(blocks, record):
    """(base, size) of the block a PSIN pointer names, or (None, None) when there is no such block."""
    kind = record["kind"]
    if not blocks or not 1 <= kind <= len(blocks):
        return None, None
    _type, base, size = blocks[kind - 1]
    return base, size


def chunk_stride(records, blocks):
    """The slot stride shared by a chunk's slices of one block, or None when the bytes do not prove one.

    Consecutive records of one chunk are adjacent slices of one array, so the gap between two slice
    starts is `count * stride`. The stride is returned only when every gap agrees, it is at least one
    element wide, and the last slice fits in the block. A chunk whose records do not tile one block
    (a single slice, or slices with gaps) returns None and the reader steps by ELEMENT_SIZE.
    """
    kinds = {record["kind"] for record in records}
    if len(kinds) != 1 or not blocks:
        return None
    base, size = block_at(blocks, records[0])
    if base is None:
        return None
    starts = [base + pool_byte_offset(record) for record in records]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        return None
    strides = set()
    for index in range(len(records) - 1):
        count = records[index]["count"]
        gap = starts[index + 1] - starts[index]
        if not count or gap % count:
            return None
        strides.add(gap // count)
    if len(strides) != 1:
        return None
    stride = strides.pop()
    if stride < ELEMENT_SIZE or stride % 4:
        return None
    if starts[-1] + records[-1]["count"] * stride > base + size:
        return None
    return stride


def scan_elements(blob, end):
    """Every valid pool element that starts in [0, end), keyed by its byte address."""
    found = {}
    for at in range(0, max(0, end - ELEMENT_SIZE + 1), 4):
        element = decode_element(blob, at)
        if element is not None:
            found[at] = element
    return found


def find_chunks(blob, end, elements):
    """Maximal record runs. Returns [{offset, records}] in file order.

    A run is two or more consecutive 32-byte records of the same kind. A lone record survives only when
    its file-absolute offset already names a valid element, which keeps a synthetic one-record table
    readable without turning every aligned window in a schema into a table.
    """
    covered = bytearray(end)
    chunks = []
    at = 0
    while at + RECORD_SIZE <= end:
        if covered[at]:
            at += 4
            continue
        first = decode_record(blob, at)
        if first is None:
            at += 4
            continue
        run = [(at, first)]
        nxt = at + RECORD_SIZE
        while nxt + RECORD_SIZE <= end:
            following = decode_record(blob, nxt)
            if following is None or following["kind"] != first["kind"]:
                break
            run.append((nxt, following))
            nxt += RECORD_SIZE
        if len(run) >= 2 or (first["key"] >= 0x1000000 and first["colB"] <= 4
                             and first["poolOffset"] in elements):
            chunks.append({"offset": at, "records": run})
            for record_at, _ in run:
                covered[record_at:record_at + RECORD_SIZE] = b"\x01" * RECORD_SIZE
            at = nxt
        else:
            # A rejected lone window is not a real record, so resume scanning at the next 4-byte slot.
            at += 4
    return chunks


def record_blocks(blob, blocks):
    """The indices of blocks that are a whole table of same-kind records."""
    out = []
    for index, (_kind, base, size) in enumerate(blocks):
        if size < RECORD_SIZE * 2 or size % RECORD_SIZE:
            continue
        first = decode_record(blob, base)
        if first is None:
            continue
        rows = [decode_record(blob, base + at) for at in range(0, size, RECORD_SIZE)]
        if all(row is not None and row["kind"] == first["kind"] for row in rows):
            out.append(index)
    return out


def linked_pool(blob, blocks, block_indices, chunk_at, elements):
    """The base of the pool block linked to the record block that holds `chunk_at`, or None.

    The element block follows the record block. Sub-64-byte metadata blocks are stepped over; another
    record block is skipped (its own pool follows it). The first other block must start with a valid
    element, otherwise this chunk belongs to a pool family this reader does not decode.
    """
    if not blocks:
        return None
    owner = None
    for index, (_kind, base, size) in enumerate(blocks):
        if base <= chunk_at < base + size:
            owner = index
            break
    if owner is None or owner not in block_indices:
        return None
    for index in range(owner + 1, len(blocks)):
        _kind, base, size = blocks[index]
        if size < ELEMENT_SIZE:
            continue
        if index in block_indices:
            continue
        return base if base in elements else None
    return None


def resolve(record, elements, pool_base, blocks=None, stride=None):
    """Resolve one record to [(address, element)].

    All three addresses are tried when they exist: the file-absolute address, the linked-pool address,
    and the block-indexed address (`blocks[kind - 1]` plus the split pointer offset). The one that
    validates more of the record's `n` slots wins; a tie keeps the earlier reading. This matters on the
    pool-relative families, where an offset such as 0x3000 can also land on an unrelated element in the
    file and would otherwise shadow the correct pool slot.

    Returns (entities, mode, skipped_slots). mode is "absolute", "pool", "block", "empty" or
    "unresolved".
    """
    count = record["count"]
    if count == 0:
        return [], "empty", 0
    default_step = stride or ELEMENT_SIZE
    candidates = [(record["poolOffset"], "absolute", ELEMENT_SIZE)]
    if pool_base is not None:
        address = pool_base + (record["poolOffset"] >> POOL_SHIFT)
        candidates.append((address, "pool", ELEMENT_SIZE))
    base, size = block_at(blocks, record)
    if base is not None:
        offset = pool_byte_offset(record)
        if offset + count * default_step <= size:
            candidates.append((base + offset, "block", default_step))
    best = None
    for base_at, mode, slot_step in candidates:
        entities = []
        skipped = 0
        for slot in range(count):
            address = base_at + slot * slot_step
            element = elements.get(address)
            if element is None:
                skipped += 1
            else:
                entities.append((address, element))
        if best is None or len(entities) > len(best[0]):
            best = (entities, mode, skipped)
    entities, mode, skipped = best
    if not entities:
        return [], "unresolved", skipped
    return entities, mode, skipped


def parse_blob(blob, name="<memory>"):
    """Parse one inflated stream (PSIN or bare bytes) into the tool's per-file record."""
    first_section = len(blob)
    blocks = []
    if psin_index is not None and blob[:4] == b"PSIN":
        parsed = psin_index.parse(blob)
        if parsed is not None:
            first_section = min(max(parsed["first_section"], 0), len(blob))
            if psin_data is not None:
                try:
                    blocks, _fill = psin_data.block_table(blob, first_section)
                except (ValueError, struct.error):
                    blocks = []
    end = first_section
    elements = scan_elements(blob, end)
    chunks = find_chunks(blob, end, elements)
    table_blocks = record_blocks(blob, blocks) if blocks else []
    if table_blocks:
        known = {at for chunk in chunks for at, _record in chunk["records"]}
        for index in table_blocks:
            _kind, base, size = blocks[index]
            addresses = list(range(base, base + size, RECORD_SIZE))
            if any(address in known for address in addresses):
                continue
            rows = [(address, decode_record(blob, address)) for address in addresses]
            if all(row is not None for _address, row in rows):
                chunks.append({"offset": base, "records": rows})
                known.update(addresses)
        chunks.sort(key=lambda chunk: chunk["offset"])

    records = []
    entity_by_address = {}
    states = collections.defaultdict(lambda: {"key": 0, "records": 0, "slots": 0, "entities": 0})
    skipped_slots = 0
    unresolved = 0
    for chunk in chunks:
        pool_base = linked_pool(blob, blocks, table_blocks, chunk["offset"], elements)
        stride = chunk_stride([record for _at, record in chunk["records"]], blocks)
        for _at, record in chunk["records"]:
            records.append(record)
            state = states[record["key"]]
            state["key"] = record["key"]
            state["records"] += 1
            found, _mode, skipped = resolve(record, elements, pool_base, blocks, stride)
            skipped_slots += skipped
            if not found and record["count"]:
                unresolved += 1
            for address, element in found:
                state["slots"] += 1
                entity_by_address.setdefault(address, element)
    for state in states.values():
        state["entities"] = state["slots"]

    entities = [entity_by_address[address] for address in sorted(entity_by_address)]
    covered = bytearray(end)
    for chunk in chunks:
        for at, _record in chunk["records"]:
            covered[at:at + RECORD_SIZE] = b"\x01" * RECORD_SIZE
    for address in entity_by_address:
        covered[address:address + ELEMENT_SIZE] = b"\x01" * ELEMENT_SIZE
    skipped_bytes = end - sum(covered)
    trailing = end
    while trailing > 0 and not covered[trailing - 1]:
        trailing -= 1
    trailing_bytes = end - trailing

    return {
        "file": name,
        "size": len(blob),
        "scan_bytes": end,
        "records": len(records),
        "chunks": len(chunks),
        "record_blocks": len(table_blocks),
        "state_keys": len(states),
        "entities": len(entities),
        "entity_slots": sum(state["slots"] for state in states.values()),
        "unresolved_records": unresolved,
        "skipped_slots": skipped_slots,
        "skipped_bytes": skipped_bytes,
        "trailing_bytes": trailing_bytes,
        "distinct_archetypes": len({entity["archetype"] for entity in entities}),
        "distinct_links": len({entity["link"] for entity in entities}),
        "states": [
            {
                "key": state["key"],
                "records": state["records"],
                "entities": state["entities"],
            }
            for _key, state in sorted(states.items())
        ],
        "entity_list": [
            {
                "archetype": entity["archetype"],
                "link": entity["link"],
                "hash": entity["hash"],
                "extra": entity["extra"],
                "pos": entity["pos"],
                "rot": entity["rot"],
                "offset": entity["offset"],
            }
            for entity in entities
        ],
    }


def parse_file(path):
    with open(path, "rb") as handle:
        blob = handle.read()
    return parse_blob(blob, os.path.basename(path))


def files_in(src):
    """(path, name) pairs, sorted by name for a directory; one pair for a file."""
    if os.path.isfile(src):
        yield src, os.path.basename(src)
        return
    for name in sorted(os.listdir(src)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(src, name)
        if os.path.isfile(path):
            yield path, name


def collect(src, limit=None):
    files = []
    for path, name in files_in(src):
        record = parse_file(path)
        record["file"] = name
        files.append(record)
        if limit and len(files) >= limit:
            break
    return files


def summarize(files):
    """Totals over files; the distinct hash counts are over the union, not a sum of per-file counts."""
    archetypes = set()
    links = set()
    states = set()
    for record in files:
        states.update(state["key"] for state in record["states"])
        archetypes.update(entity["archetype"] for entity in record["entity_list"])
        links.update(entity["link"] for entity in record["entity_list"])
    return {
        "files": len(files),
        "records": sum(record["records"] for record in files),
        "chunks": sum(record["chunks"] for record in files),
        "state_keys": len(states),
        "entities": sum(record["entities"] for record in files),
        "entity_slots": sum(record["entity_slots"] for record in files),
        "distinct_archetypes": len(archetypes),
        "distinct_links": len(links),
        "unresolved_records": sum(record["unresolved_records"] for record in files),
        "skipped_slots": sum(record["skipped_slots"] for record in files),
        "skipped_bytes": sum(record["skipped_bytes"] for record in files),
    }


def _hex(value):
    return f"{value:08x}"


def _json_record(record):
    """The same record with hashes formatted as eight hex digits."""
    out = dict(record)
    out["states"] = [dict(state, key=_hex(state["key"])) for state in record["states"]]
    out["entity_list"] = [
        {
            "archetype": _hex(entity["archetype"]),
            "link": _hex(entity["link"]),
            "hash": _hex(entity["hash"]),
            "extra": _hex(entity["extra"]),
            "pos": [round(value, 6) for value in entity["pos"]],
            "rot": [round(value, 6) for value in entity["rot"]],
            "offset": entity["offset"],
        }
        for entity in record["entity_list"]
    ]
    return out


def count(text):
    """--limit: a positive file count."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read mission-state tables from inflated PSIN streams.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write the files and summary as JSON")
    ap.add_argument("--limit", type=count, help="stop after this many files")
    args = ap.parse_args(argv)
    if not os.path.exists(args.src):
        print(f"ERROR: {args.src}: no such file or folder", file=sys.stderr)
        return 2
    if os.path.isfile(args.src):
        # A lone argument is the inflated PSIN file; the bare carved streams are what a folder holds.
        try:
            with open(args.src, "rb") as handle:
                magic = handle.read(4)
        except OSError as error:
            print(f"ERROR: {args.src}: {error}", file=sys.stderr)
            return 2
        if magic != b"PSIN":
            print(f"ERROR: {args.src}: not a PSIN container", file=sys.stderr)
            return 2
    try:
        files = collect(args.src, args.limit)
    except OSError as error:
        print(f"ERROR: {args.src}: {error}", file=sys.stderr)
        return 2
    summary = summarize(files)
    if args.json:
        json.dump({"files": [_json_record(record) for record in files], "summary": summary},
                  sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for record in files:
        print(f"{record['file']}  size={record['size']}  records={record['records']}  "
              f"chunks={record['chunks']}  states={record['state_keys']}  "
              f"entities={record['entities']}  archetypes={record['distinct_archetypes']}  "
              f"links={record['distinct_links']}  unresolved={record['unresolved_records']}  "
              f"skipped_slots={record['skipped_slots']}  skipped_bytes={record['skipped_bytes']}")
        for entity in record["entity_list"]:
            pos = ", ".join(f"{value:.3f}" for value in entity["pos"])
            rot = ", ".join(f"{value:.3f}" for value in entity["rot"])
            print(f"  entity {_hex(entity['archetype'])}  {_hex(entity['link'])}  "
                  f"pos=({pos})  rot=({rot})  +0x{entity['offset']:x}")
    print(f"{summary['files']} files, {summary['records']} records in {summary['chunks']} chunks; "
          f"{summary['state_keys']} distinct state keys; {summary['entities']} entities "
          f"({summary['entity_slots']} slots), {summary['distinct_archetypes']} archetypes, "
          f"{summary['distinct_links']} links; unresolved {summary['unresolved_records']}, "
          f"skipped slots {summary['skipped_slots']}, skipped bytes {summary['skipped_bytes']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
