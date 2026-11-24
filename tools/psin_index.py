"""Index the PSIN metadata container used by the game's data and script archives.

PSIN is a small metadata container. It appears at the start of carved zlib streams in `data.rpf` and
`script.rpf`; it is not a script program and it is not a model or texture resource. Every measured
file uses the same header:

    +0   50 53 49 4e          "PSIN"
    +4   u32 big-endian       absolute offset of the first section (LSCH)
    +8   u32 little-endian    version (22 on every measured file)
    +12  u32 little-endian    byte count of the hash table that follows the header

The section table starts at the offset in `+4` and runs to the end of the file. Every section is:

    +0   4-byte ASCII tag
    +4   u32 big-endian       total size, including this 8-byte header

Sections are contiguous: the next section starts at `offset + size`, and the last one ends at EOF.
The measured tags are `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC` and `CHKS`. `LGID` is an empty section
(8 bytes). `LRSC` starts with a `0DRP` page directory header. `CHKS` carries the file size and a
checksum.

`LSNS` is a block table followed by the blocks (measured over every `LSNS` section in the package; docs/psin.md):

    +0   u32 little-endian    block count n (0 for the empty 12-byte section)
    +4   n x (u32 kind, u32 byte count)
         then the n blocks back to back, each a whole number of 16-byte records

The byte counts fill the payload exactly. The kind's high byte is 1 or 2 (records read as 4- or 8-byte words);
its low byte varies. The records themselves are uniformly random bytes that match no tested name hash; this tool
reads the layout, not the contents.

`LSCH` is a type schema and `LENM` its enum table (measured over all 18,640 PSIN streams in the archives of package
1.0.2.1; see docs/psin.md). Both store all headers first and then all elements, and neither stores the
header count: it is the one count k for which the headers' element counts fill the payload exactly.

    LSCH struct header, 32 bytes        LSCH member, 16 bytes        LENM enum header, 24 bytes
    +0  u32 name hash                    +0  u32 name hash            +0  u32 name hash
    +4  u32 layout signature             +4  u32 byte offset          +4  u32 layout signature
    +8  u8  flags                        +8  u8  type code            +8  8 fill bytes (0xee)
    +9  u8  log2 alignment               +9  u8  sub-code             +16 u32 value count
    +10 u16 0, +12 u32 0                 +10 u16 element index        +20 u32 0
    +16 8 fill bytes (0xaa)              +12 u32 reference hash      LENM value, 16 bytes
    +24 u32 byte size                                                  +0  i64 value
    +28 u16 0, +30 u16 member count                                    +8  u64 name hash (high half 0)

Names are case-sensitive joaat. A member whose name hash is 0x100 is the element descriptor of an array member
(type 0x5c), whose element index points at it; a struct member (type 5) references a struct name hash, and the
enum members (0x60, 0x62, 0x65, ...) reference an enum name hash in `LENM`.

On the package measured here the `script.rpf` carve contains 671 streams that start with `PSIN`:
670 are complete and one is the first 64 KiB of a larger record (its first-section offset is beyond
the carved block). A complete record in `data.rpf` is validated by the project's existing
`ArchiveReader` path, not by this tool. The tool deliberately indexes containers; it does not assign
meaning to the CHKS payload or to the bytes between the header and the first section.

Usage:
    python -I psin_index.py <file-or-dir> [--json] [--schema] [--limit N]

Text output prints one line per PSIN file and a summary; `--json` writes `{files, summary}`. `--schema` adds the
decoded LSCH structs and LENM enums (hashes as hex) to each JSON record, or prints them in text mode.
"""
import argparse
import json
import os
import struct
import sys

MAGIC = b"PSIN"
HEADER = 16
SECTION_HEADER = 8
STRUCT_HEADER = 32
MEMBER = 16
ENUM_HEADER = 24
ENUM_VALUE = 16
ARRAY_ELEMENT = 0x100   # member name hash of an array's element descriptor
ARRAY_TYPE = 0x5C


def _u32_be(blob, off):
    return struct.unpack_from(">I", blob, off)[0]


def _u32_le(blob, off):
    return struct.unpack_from("<I", blob, off)[0]


def _header_count(payload, header, element, count_at, count_fmt):
    """The header count k for which k headers and their elements fill the payload exactly.

    Neither LSCH nor LENM stores this count. Returns None when no k fits.
    """
    if not payload:
        return 0
    elements = 0
    for k in range(1, len(payload) // header + 1):
        elements += struct.unpack_from(count_fmt, payload, header * (k - 1) + count_at)[0]
        if header * k + element * elements == len(payload):
            return k
    return None


def decode_lsch(payload):
    """Decode an LSCH payload into a list of structs; raises ValueError when no struct count fits."""
    k = _header_count(payload, STRUCT_HEADER, MEMBER, 30, "<H")
    if k is None:
        raise ValueError(f"LSCH: no struct count fills {len(payload)} bytes")
    structs = []
    at = STRUCT_HEADER * k
    for i in range(k):
        name, sig, flags, align, _z10, _z12, size, _z28, count = struct.unpack_from(
            "<IIBBHI8xIHH", payload, STRUCT_HEADER * i)
        members = []
        for _ in range(count):
            mname, off, mtype, sub, index, ref = struct.unpack_from("<IIBBHI", payload, at)
            at += MEMBER
            members.append({"hash": mname, "offset": off, "type": mtype, "sub": sub,
                            "index": index, "ref": ref})
        structs.append({"hash": name, "signature": sig, "flags": flags, "align_log2": align,
                        "size": size, "members": members})
    return structs


def decode_lenm(payload):
    """Decode an LENM payload into a list of enums; raises ValueError when no enum count fits."""
    k = _header_count(payload, ENUM_HEADER, ENUM_VALUE, 16, "<I")
    if k is None:
        raise ValueError(f"LENM: no enum count fills {len(payload)} bytes")
    enums = []
    at = ENUM_HEADER * k
    for i in range(k):
        name, sig = struct.unpack_from("<II", payload, ENUM_HEADER * i)
        count = _u32_le(payload, ENUM_HEADER * i + 16)
        values = []
        for _ in range(count):
            value, vname = struct.unpack_from("<qQ", payload, at)
            at += ENUM_VALUE
            values.append({"value": value, "hash": vname})
        enums.append({"hash": name, "signature": sig, "values": values})
    return enums


LSNS_RECORD = 16


def decode_lsns(payload):
    """The LSNS block table: [(kind, byte count)]. Raises ValueError unless the blocks fill the payload exactly."""
    if len(payload) < 4:
        raise ValueError("LSNS shorter than its block count")
    n = _u32_le(payload, 0)
    table_end = 4 + 8 * n
    if table_end > len(payload):
        raise ValueError(f"LSNS block table ({n} blocks) runs past the section")
    blocks = [struct.unpack_from("<II", payload, 4 + 8 * i) for i in range(n)]
    for kind, size in blocks:
        if size % LSNS_RECORD:
            raise ValueError(f"LSNS block of kind 0x{kind:x} has {size} bytes, not a multiple of {LSNS_RECORD}")
    if table_end + sum(size for _kind, size in blocks) != len(payload):
        raise ValueError("LSNS blocks do not fill the section")
    return blocks


def check_schema(structs):
    """Checks that hold on every measured LSCH. Returns the first problem, or ''."""
    for s in structs:
        for i, m in enumerate(s["members"]):
            if m["hash"] != ARRAY_ELEMENT and m["offset"] >= s["size"]:
                return f"struct {s['hash']:08x} member {i} at offset {m['offset']} is past size {s['size']}"
            if m["type"] == ARRAY_TYPE:
                j = m["index"]
                if j >= len(s["members"]) or s["members"][j]["hash"] != ARRAY_ELEMENT:
                    return f"struct {s['hash']:08x} array member {i} points at {j}, not an element descriptor"
    return ""


def _hexed(structs, enums):
    """The decoded schema with hashes as 8-digit hex, for JSON and text output."""
    out_s = [dict(s, hash=f"{s['hash']:08x}", signature=f"{s['signature']:08x}",
                  members=[dict(m, hash=f"{m['hash']:08x}", ref=f"{m['ref']:08x}") for m in s["members"]])
             for s in structs]
    out_e = [dict(e, hash=f"{e['hash']:08x}", signature=f"{e['signature']:08x}",
                  values=[{"value": v["value"], "hash": f"{v['hash']:08x}"} for v in e["values"]])
             for e in enums]
    return {"structs": out_s, "enums": out_e}


def parse(blob, schema=False):
    """Parse one PSIN container. Returns None when the blob is not PSIN.

    With `schema`, the record also carries the decoded LSCH structs and LENM enums under "schema".
    """
    if len(blob) < HEADER or blob[:4] != MAGIC:
        return None
    first = _u32_be(blob, 4)
    version = _u32_le(blob, 8)
    table_bytes = _u32_le(blob, 12)
    record = {
        "magic": "PSIN",
        "size": len(blob),
        "first_section": first,
        "version": version,
        "table_bytes": table_bytes,
        "sections": [],
        "rsc_version": None,
        "checksum_size": None,
        "checksum": "",
        "structs": None,
        "members": None,
        "enums": None,
        "enum_values": None,
        "schema_error": "",
        "lsns_blocks": None,
        "lsns_records": None,
        "lsns_error": "",
        "complete": False,
        "error": "",
    }
    structs = enums = None
    lsch_seen = lenm_seen = False
    lsch_error = lenm_error = ""
    if first < HEADER or first > len(blob):
        record["error"] = f"first section offset {first} is outside the file"
        return _with_schema(record, schema, structs, enums)
    if len(blob) - first < SECTION_HEADER:
        record["error"] = f"missing or truncated first section header at {first}"
        return _with_schema(record, schema, structs, enums)
    offset = first
    while offset + SECTION_HEADER <= len(blob):
        tag = blob[offset:offset + 4]
        size = _u32_be(blob, offset + 4)
        if any(c < 0x20 or c > 0x7E for c in tag):
            record["error"] = f"non-text section tag at {offset}"
            break
        if size < SECTION_HEADER or offset + size > len(blob):
            record["error"] = f"section {tag.decode('latin-1')} at {offset} has size {size}"
            break
        payload_off = offset + SECTION_HEADER
        section = {
            "tag": tag.decode("latin-1"),
            "offset": offset,
            "size": size,
            "payload": size - SECTION_HEADER,
        }
        if tag == b"LRSC" and size - SECTION_HEADER >= 8 and blob[payload_off:payload_off + 4] == b"0DRP":
            section["rsc_magic"] = "0DRP"
            section["rsc_version"] = _u32_le(blob, payload_off + 4)
            if record["rsc_version"] is None:
                record["rsc_version"] = section["rsc_version"]
        elif tag == b"CHKS" and size - SECTION_HEADER >= 8:
            section["checksum_size"] = _u32_be(blob, payload_off)
            section["checksum"] = blob[payload_off + 4:payload_off + 8].hex()
            if record["checksum_size"] is None:
                record["checksum_size"] = section["checksum_size"]
                record["checksum"] = section["checksum"]
        elif tag == b"LSNS":
            try:
                blocks = decode_lsns(blob[payload_off:offset + size])
            except ValueError as e:
                section["lsns_error"] = record["lsns_error"] = str(e)
            else:
                section["blocks"] = [{"kind": f"0x{k:x}", "bytes": b} for k, b in blocks]
                record["lsns_blocks"] = len(blocks)
                record["lsns_records"] = sum(b for _k, b in blocks) // LSNS_RECORD
        elif tag == b"LSCH" and not lsch_seen:
            lsch_seen = True
            try:
                structs = decode_lsch(blob[payload_off:offset + size])
            except ValueError as e:
                lsch_error = str(e)
            else:
                record["structs"] = len(structs)
                record["members"] = sum(len(s["members"]) for s in structs)
                lsch_error = check_schema(structs)
        elif tag == b"LENM" and not lenm_seen:
            lenm_seen = True
            try:
                enums = decode_lenm(blob[payload_off:offset + size])
            except ValueError as e:
                lenm_error = str(e)
            else:
                record["enums"] = len(enums)
                record["enum_values"] = sum(len(e["values"]) for e in enums)
        record["sections"].append(section)
        offset += size
    record["schema_error"] = "; ".join(e for e in (lsch_error, lenm_error) if e)
    if not record["error"] and offset != len(blob):
        record["error"] = f"section walk ended at {offset} of {len(blob)}"
    record["complete"] = not record["error"] and offset == len(blob)
    return _with_schema(record, schema, structs, enums, lsch_error, lenm_error)


def _with_schema(record, schema, structs, enums, lsch_error="", lenm_error=""):
    """With `schema`, give every record the same "schema" shape, whether or not the sections were reached.

    "lsch" and "lenm" say whether each section was decoded; "reason" says why one was not (a structural error
    that stopped the walk first, a missing section, or the decode error), and is "" when both were.
    """
    if not schema:
        return record
    entry = _hexed(structs or [], enums or [])
    entry["lsch"] = structs is not None
    entry["lenm"] = enums is not None
    reasons = []
    for name, decoded, section_error in (
        ("LSCH", entry["lsch"], lsch_error),
        ("LENM", entry["lenm"], lenm_error),
    ):
        if decoded:
            continue
        if any(s["tag"] == name for s in record["sections"]):
            reasons.append(section_error or f"{name} not decoded")
        elif record["error"]:
            reasons.append(f"no whole {name} section: {record['error']}")
        else:
            reasons.append(f"no {name} section")
    entry["reason"] = "; ".join(dict.fromkeys(reasons))
    record["schema"] = entry
    return record


def input_paths(src):
    """Paths read from a single source file or a directory of carved PSIN streams."""
    if os.path.isfile(src):
        yield src
        return
    for name in sorted(os.listdir(src)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(src, name)
        if os.path.isfile(path):
            yield path


def files_in(src):
    for path in input_paths(src):
        with open(path, "rb") as f:
            yield os.path.basename(path), f.read()


def collect(src, limit=None, schema=False):
    records = []
    for name, blob in files_in(src):
        record = parse(blob, schema)
        if record is None:
            continue
        record["file"] = name
        records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def summarize(records):
    tags = {}
    versions = {}
    for r in records:
        versions[r["version"]] = versions.get(r["version"], 0) + 1
        for s in r["sections"]:
            tags[s["tag"]] = tags.get(s["tag"], 0) + 1
    return {
        "files": len(records),
        "complete": sum(1 for r in records if r["complete"]),
        "incomplete": sum(1 for r in records if not r["complete"]),
        "versions": {str(k): v for k, v in sorted(versions.items())},
        "section_tags": dict(sorted(tags.items())),
        "section_bytes": sum(s["size"] for r in records for s in r["sections"]),
        "rsc_122": sum(1 for r in records if r["rsc_version"] == 122),
        "checksummed": sum(1 for r in records if r["checksum_size"] is not None),
        "lsch_decoded": sum(1 for r in records if r["structs"] is not None),
        "lenm_decoded": sum(1 for r in records if r["enums"] is not None),
        "schema_errors": sum(1 for r in records if r["schema_error"]),
        "structs": sum(r["structs"] or 0 for r in records),
        "members": sum(r["members"] or 0 for r in records),
        "enums": sum(r["enums"] or 0 for r in records),
        "enum_values": sum(r["enum_values"] or 0 for r in records),
        "lsns_decoded": sum(1 for r in records if r["lsns_blocks"] is not None),
        "lsns_empty": sum(1 for r in records if r["lsns_blocks"] == 0),
        "lsns_errors": sum(1 for r in records if r["lsns_error"]),
        "lsns_records": sum(r["lsns_records"] or 0 for r in records),
        "lsns_kinds": dict(sorted(_lsns_kinds(records).items())),
    }


def _lsns_kinds(records):
    kinds = {}
    for r in records:
        for s in r["sections"]:
            for b in s.get("blocks", ()):
                kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return kinds


def count(text):
    """--limit: 0 or a negative value used to mean no limit (0 is falsy, a negative count is never reached)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    ap = argparse.ArgumentParser(description="Index PSIN metadata containers.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write records and summary as JSON")
    ap.add_argument("--schema", action="store_true", help="include the decoded LSCH structs and LENM enums")
    ap.add_argument("--limit", type=count, help="stop after this many PSIN files")
    a = ap.parse_args()

    if not os.path.exists(a.src):   # files_in took any non-file for a folder: a typo ended in a traceback
        print(f"ERROR: {a.src}: no such file or folder", file=sys.stderr)
        return 2
    try:
        records = collect(a.src, a.limit, a.schema)
    except OSError as e:
        print(f"ERROR: {a.src}: {e}", file=sys.stderr)
        return 2
    summary = summarize(records)
    if a.json:
        json.dump({"files": records, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for r in records:
        tags = " ".join(f"{s['tag']}:{s['size']}" for s in r["sections"]) or "-"
        state = "ok" if r["complete"] else f"INCOMPLETE ({r['error']})"
        extra = ""
        if r["rsc_version"] is not None:
            extra += f"  rsc={r['rsc_version']}"
        if r["checksum_size"] is not None:
            extra += f"  chks={r['checksum_size']}"
        if r["structs"] is not None:
            extra += f"  structs={r['structs']} members={r['members']}"
        if r["enums"] is not None:
            extra += f"  enums={r['enums']} values={r['enum_values']}"
        if r["schema_error"]:
            extra += f"  SCHEMA ({r['schema_error']})"
        print(f"{r['file']}  size={r['size']}  v={r['version']}  {state}  {tags}{extra}")
        if a.schema:
            if r["schema"]["reason"]:
                print(f"  schema: {r['schema']['reason']}")
            for s in r["schema"]["structs"]:
                print(f"  struct {s['hash']} sig={s['signature']} size={s['size']} "
                      f"align=2^{s['align_log2']} flags={s['flags']:#x} members={len(s['members'])}")
                for m in s["members"]:
                    print(f"    {m['hash']} +{m['offset']:#x} type={m['type']:#04x} sub={m['sub']} "
                          f"index={m['index']} ref={m['ref']}")
            for e in r["schema"]["enums"]:
                print(f"  enum {e['hash']} sig={e['signature']} values={len(e['values'])}")
                for v in e["values"]:
                    print(f"    {v['hash']} = {v['value']}")
    print(f"{summary['files']} PSIN files, {summary['complete']} complete, "
          f"{summary['incomplete']} incomplete; tags {summary['section_tags']}; "
          f"rsc122 {summary['rsc_122']}, checksummed {summary['checksummed']}; "
          f"LSCH decoded {summary['lsch_decoded']} ({summary['structs']} structs, {summary['members']} members), "
          f"LENM decoded {summary['lenm_decoded']} ({summary['enums']} enums, {summary['enum_values']} values), "
          f"schema errors {summary['schema_errors']}; "
          f"LSNS decoded {summary['lsns_decoded']} ({summary['lsns_empty']} empty, {summary['lsns_records']} records), "
          f"LSNS errors {summary['lsns_errors']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
