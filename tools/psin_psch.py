"""Decode the PSO-style PMAP/PSCH/PSIG/STNS PSIN containers (docs/psin.md).

57 complete PSIN streams (54 in data.rpf, 2 in packs.rpf, 1 in anim.rpf) use the
PSO-style `PMAP`/`PSCH`/`PSIG`/`STNS` (plus optional `STFN`/`STDN`) section chain
instead of `LSCH`/`LENM`/`LSNS`/`LRSC`:

    +0x00.."PSIN", +0x04 u32be first_section, +0x08.."pppppppp", data blocks at +0x10
    PMAP:  u32be root_block_id, u16be flags, u16be 0, u16be block_count, "pppppp",
           then `block_count` x 16B big-endian rows {u32 name_hash, u32 offset, u32 0, u32 length}
    PSCH:  u32be entry_count, `entry_count` x (u32 name_hash, u32 section_offset),
           followed by struct entries (kind=0, 12B hdr + 12B members) and enum entries (kind=1, 4B hdr + 16B values)
    PSIG:  `entry_count` x u32be layout signatures
    STNS:  u32le block_count, `block_count` x (u32le kind, u32le bytes), 16B-aligned blocks (+ 0/4 zero pad)

Usage:
    python -I tools/psin_psch.py <file_or_carve_dir_or_rpf> [--carve-dir DIR] [--json FILE]

Prints and writes aggregate counts, schema validation checks and type histograms
only; no game payloads or decoded values leave local runs. Standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing sibling helpers.
TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import gamedata_index as gi  # noqa: E402
import gamedata_maps as gm  # noqa: E402
import psin_index  # noqa: E402
from report_paths import same_report_path  # noqa: E402

ARRAY_ELEMENT = psin_index.ARRAY_ELEMENT
PSCH_TYPE_STRUCT = 0x0C
PSCH_TYPE_ARRAY = 0x0D
PSCH_TYPE_ENUM = 0x0E
PSCH_TYPE_MAP = 0x10
PMAP_FILL = b"pppppp"
HEADER_FILL = b"pppppppp"


def is_psch_container(blob: bytes) -> bool:
    """True when `blob` is a PSIN container carrying PMAP or PSCH sections."""
    rec = psin_index.parse(blob)
    if rec is None or not rec["complete"]:
        return False
    tags = {s["tag"] for s in rec["sections"]}
    return "PMAP" in tags or "PSCH" in tags


def decode_pmap(payload: bytes, blob: bytes, first_section: int) -> dict:
    """Decode the PMAP block table (big-endian)."""
    if len(payload) < 16:
        raise ValueError("PMAP payload shorter than 16-byte header")
    if len(blob) < 16 or blob[8:16] != HEADER_FILL:
        raise ValueError("PMAP container lacks 8-byte 'pppppppp' header fill at +0x08")
    root_id, flags, z0, block_count = struct.unpack_from(">IHHH", payload, 0)
    if payload[10:16] != PMAP_FILL:
        raise ValueError("PMAP header lacks 6-byte 'pppppp' fill at +10")
    if len(payload) != 16 + block_count * 16:
        raise ValueError(f"PMAP size {len(payload)} != 16 + {block_count} * 16")
    if block_count == 0 or not (1 <= root_id <= block_count):
        raise ValueError(f"PMAP root_block_id {root_id} outside 1..{block_count}")

    blocks = []
    nonzero_row_reserved = 0
    for idx in range(block_count):
        name_hash, offset, reserved, length = struct.unpack_from(">4I", payload, 16 + idx * 16)
        if reserved != 0:
            nonzero_row_reserved += 1
        if offset < 16 or offset + length > first_section:
            raise ValueError(f"PMAP block {idx + 1} ({offset:#x}+{length}) outside data area 0x10..{first_section:#x}")
        blocks.append({
            "id": idx + 1,
            "hash": name_hash,
            "offset": offset,
            "reserved": reserved,
            "length": length,
        })

    if blocks[0]["offset"] != 16:
        raise ValueError(f"first PMAP block starts at {blocks[0]['offset']:#x}, expected 0x10")
    exact_end = (blocks[-1]["offset"] + blocks[-1]["length"]) == first_section
    if not exact_end and ((blocks[-1]["offset"] + blocks[-1]["length"] + 15) & ~15) != first_section:
        raise ValueError("last PMAP block does not end at first_section")

    return {
        "root_block_id": root_id,
        "flags": flags,
        "reserved_zero": z0 == 0 and nonzero_row_reserved == 0,
        "blocks": blocks,
        "exact_end": exact_end,
    }


def decode_psch(section_bytes: bytes) -> dict:
    """Decode a complete PSCH section (8-byte PSCH header included)."""
    if len(section_bytes) < 12 or section_bytes[:4] != b"PSCH":
        raise ValueError("not a PSCH section")
    entry_count = struct.unpack_from(">I", section_bytes, 8)[0]
    dir_end = 12 + entry_count * 8
    if dir_end > len(section_bytes):
        raise ValueError(f"PSCH directory ({entry_count} entries) runs past section end")

    directory = [struct.unpack_from(">2I", section_bytes, 12 + i * 8) for i in range(entry_count)]
    offsets = [d[1] for d in directory] + [len(section_bytes)]
    if entry_count and offsets[0] != dir_end:
        raise ValueError(f"first PSCH entry at {offsets[0]} != directory end {dir_end}")
    if any(offsets[i] >= offsets[i + 1] or offsets[i + 1] > len(section_bytes) for i in range(entry_count)):
        raise ValueError("PSCH directory offsets are not strictly ascending inside the section")

    structs = []
    enums = []
    for i, (name_hash, _off) in enumerate(directory):
        eb = section_bytes[offsets[i]:offsets[i + 1]]
        if len(eb) < 4:
            raise ValueError(f"PSCH entry {i} shorter than 4 bytes")
        kind, flags, count = struct.unpack_from(">BBH", eb, 0)
        if kind == 0:
            if len(eb) != 12 + count * 12:
                raise ValueError(f"PSCH struct {name_hash:08x} size {len(eb)} != 12 + {count} * 12")
            byte_size, reserved = struct.unpack_from(">2I", eb, 4)
            members = []
            for j in range(count):
                m_hash, m_type, m_sub, m_off, m_ref = struct.unpack_from(">IBBHI", eb, 12 + j * 12)
                members.append({
                    "hash": m_hash,
                    "type": m_type,
                    "sub": m_sub,
                    "offset": m_off,
                    "ref": m_ref,
                })
            structs.append({
                "hash": name_hash,
                "flags": flags,
                "size": byte_size,
                "reserved": reserved,
                "members": members,
            })
        elif kind == 1:
            if len(eb) != 4 + count * 16:
                raise ValueError(f"PSCH enum {name_hash:08x} size {len(eb)} != 4 + {count} * 16")
            values = []
            for j in range(count):
                v_hash, v_zero, v_val = struct.unpack_from(">IIq", eb, 4 + j * 16)
                values.append({"hash": v_hash, "reserved": v_zero, "value": v_val})
            enums.append({"hash": name_hash, "flags": flags, "values": values})
        else:
            raise ValueError(f"unknown PSCH entry kind {kind} for {name_hash:08x}")

    return {"entry_count": entry_count, "directory": directory, "structs": structs, "enums": enums}


def decode_psig(payload: bytes, entry_count: int) -> list[int]:
    """Decode PSIG big-endian per-entry layout signatures."""
    if len(payload) != 4 * entry_count:
        raise ValueError(f"PSIG payload {len(payload)} != 4 * {entry_count}")
    return list(struct.unpack_from(f">{entry_count}I", payload, 0)) if entry_count else []


def decode_stns(payload: bytes) -> dict:
    """Decode STNS little-endian 16-byte record block table."""
    if len(payload) < 4:
        raise ValueError("STNS shorter than 4-byte block count")
    count = struct.unpack_from("<I", payload, 0)[0]
    table_end = 4 + 8 * count
    if table_end > len(payload):
        raise ValueError(f"STNS table ({count} blocks) runs past section end")
    blocks = []
    data_bytes = 0
    for i in range(count):
        kind, size = struct.unpack_from("<2I", payload, 4 + i * 8)
        if size % 16 != 0:
            raise ValueError(f"STNS block {i} (kind {kind:#x}) size {size} is not a multiple of 16")
        blocks.append({"kind": kind, "bytes": size, "records": size // 16})
        data_bytes += size
    expected_len = ((table_end + data_bytes) + 15) & ~15
    tail_len = len(payload) - (table_end + data_bytes)
    if len(payload) != expected_len or any(b != 0 for b in payload[table_end + data_bytes:]):
        raise ValueError(f"STNS length/padding invalid (len={len(payload)}, expected={expected_len})")
    return {
        "blocks": blocks,
        "records": data_bytes // 16,
        "tail_pad": tail_len,
        "raw_blocks": payload[table_end:table_end + data_bytes],
    }


def check_psch_schema(pmap: dict, psch: dict) -> list[str]:
    """Validate PMAP block sizes/root and PSCH member bounds and local descriptors."""
    errors = []
    struct_by_hash = {s["hash"]: s for s in psch["structs"]}
    enum_hashes = {e["hash"] for e in psch["enums"]}

    root_block = pmap["blocks"][pmap["root_block_id"] - 1]
    if root_block["hash"] not in struct_by_hash:
        errors.append(f"PMAP root block hash {root_block['hash']:08x} missing from PSCH")

    for blk in pmap["blocks"]:
        st = struct_by_hash.get(blk["hash"])
        if st is not None and st["size"] > 0 and blk["length"] % st["size"] != 0:
            errors.append(f"PMAP block {blk['id']} length {blk['length']} not divisible by struct size {st['size']}")

    for s in psch["structs"]:
        members = s["members"]
        for idx, m in enumerate(members):
            if m["hash"] != ARRAY_ELEMENT and m["offset"] >= s["size"]:
                errors.append(f"struct {s['hash']:08x} member {idx} offset {m['offset']} >= size {s['size']}")
            if m["type"] == PSCH_TYPE_ARRAY:
                elem_idx = m["ref"] & 0xFFFF
                if elem_idx >= len(members) or members[elem_idx]["hash"] != ARRAY_ELEMENT:
                    errors.append(f"struct {s['hash']:08x} array member {idx} bad descriptor {elem_idx}")
            elif m["type"] == PSCH_TYPE_ENUM and m["ref"] not in enum_hashes:
                errors.append(f"struct {s['hash']:08x} enum member {idx} missing enum {m['ref']:08x}")
            elif m["type"] == PSCH_TYPE_MAP:
                k_idx = m["ref"] & 0xFFFF
                v_idx = m["ref"] >> 16
                if k_idx >= len(members) or members[k_idx]["hash"] != ARRAY_ELEMENT:
                    errors.append(f"struct {s['hash']:08x} map member {idx} bad key descriptor {k_idx}")
                if v_idx >= len(members) or members[v_idx]["hash"] != ARRAY_ELEMENT:
                    errors.append(f"struct {s['hash']:08x} map member {idx} bad value descriptor {v_idx}")

    return errors


def parse_psch_container(blob: bytes) -> dict:
    """Parse and cross-check one PMAP/PSCH/PSIG/STNS PSIN container."""
    rec = psin_index.parse(blob)
    if rec is None:
        raise ValueError("not a PSIN container")
    if not rec["complete"]:
        raise ValueError(f"incomplete PSIN container: {rec['error']}")

    secs = {s["tag"]: s for s in rec["sections"]}
    for required in ("PMAP", "PSCH", "PSIG", "STNS"):
        if required not in secs:
            raise ValueError(f"missing {required} section")

    def section_payload(tag: str) -> bytes:
        s = secs[tag]
        return blob[s["offset"] + 8:s["offset"] + s["size"]]

    pmap = decode_pmap(section_payload("PMAP"), blob, rec["first_section"])
    psch_sec = secs["PSCH"]
    psch = decode_psch(blob[psch_sec["offset"]:psch_sec["offset"] + psch_sec["size"]])
    psig = decode_psig(section_payload("PSIG"), psch["entry_count"])
    stns = decode_stns(section_payload("STNS"))

    errors = check_psch_schema(pmap, psch)

    stdn_match = None
    if "STDN" in secs:
        stdn_pay = section_payload("STDN")
        stdn_match = stns["raw_blocks"].startswith(stdn_pay)
        if not stdn_match:
            errors.append("STDN payload does not match STNS block data prefix")

    stfn_bytes = len(section_payload("STFN")) if "STFN" in secs else None
    struct_map = {s["hash"]: s for s in psch["structs"]}
    anon_kinds = Counter()
    struct_blocks = 0
    for blk in pmap["blocks"]:
        if blk["hash"] in struct_map:
            struct_blocks += 1
        else:
            anon_kinds[f"{blk['hash']:#x}"] += 1

    mtypes = Counter()
    for s in psch["structs"]:
        for m in s["members"]:
            mtypes[f"{m['type']:#04x}"] += 1

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "size": len(blob),
        "first_section": rec["first_section"],
        "section_tags": [s["tag"] for s in rec["sections"]],
        "pmap_blocks": len(pmap["blocks"]),
        "pmap_struct_blocks": struct_blocks,
        "pmap_anon_kinds": dict(sorted(anon_kinds.items())),
        "pmap_exact_end": pmap["exact_end"],
        "psch_entries": psch["entry_count"],
        "structs": len(psch["structs"]),
        "members": sum(len(s["members"]) for s in psch["structs"]),
        "enums": len(psch["enums"]),
        "enum_values": sum(len(e["values"]) for e in psch["enums"]),
        "member_type_counts": dict(sorted(mtypes.items())),
        "psig_count": len(psig),
        "stns_blocks": len(stns["blocks"]),
        "stns_records": stns["records"],
        "stns_tail_pad": stns["tail_pad"],
        "stns_kinds": [f"{b['kind']:#x}" for b in stns["blocks"]],
        "stdn_match": stdn_match,
        "stfn_bytes": stfn_bytes,
        "_struct_hashes": {s["hash"] for s in psch["structs"]},
        "_enum_hashes": {e["hash"] for e in psch["enums"]},
        "_member_hashes": {m["hash"] for s in psch["structs"] for m in s["members"]},
        "_enum_value_hashes": {v["hash"] for e in psch["enums"] for v in e["values"]},
        "_signatures": set(psig),
        "_raw_structs": psch["structs"],
    }


def scan_target(target: str, carve_dir: str | None = None) -> dict:
    """Collect and summarize all PMAP/PSCH containers in a file, carve folder or .rpf archive."""
    if os.path.isfile(target):
        with open(target, "rb") as f:
            head = f.read(16)
        if head[:4] == b"9FPR":
            source = gi.ArchiveSource(target, carve_dir)
            try:
                rows = source.rows
                blobs = []
                for r in rows:
                    if not r.get("head", "").startswith("5053494e"):
                        continue
                    data = source.read(r)
                    if is_psch_container(data):
                        blobs.append((f"{r['off']:#x}", data))
            finally:
                source.close()
        else:
            with open(target, "rb") as f:
                data = f.read()
            if not is_psch_container(data):
                raise ValueError(f"{target}: not a PMAP/PSCH PSIN container")
            blobs = [(os.path.basename(target), data)]
    elif os.path.isdir(target):
        if os.path.isfile(os.path.join(target, "index.json")):
            source = gi.DirectorySource(target)
            blobs = []
            for r in source.rows:
                if not r.get("head", "").startswith("5053494e"):
                    continue
                data = source.read(r)
                if is_psch_container(data):
                    blobs.append((f"{r['off']:#x}", data))
        else:
            blobs = []
            for name in sorted(os.listdir(target)):
                if not name.lower().endswith(".bin"):
                    continue
                full = os.path.join(target, name)
                if not os.path.isfile(full):
                    continue
                with open(full, "rb") as f:
                    data = f.read()
                if is_psch_container(data):
                    blobs.append((name, data))
    else:
        raise FileNotFoundError(target)

    if not blobs:
        raise ValueError(f"{target}: no PMAP/PSCH PSIN containers found")

    parsed = []
    all_structs: set[int] = set()
    all_enums: set[int] = set()
    all_members: set[int] = set()
    all_evals: set[int] = set()
    all_sigs: set[int] = set()
    mtype_totals: Counter[str] = Counter()
    anon_totals: Counter[str] = Counter()
    chain_totals: Counter[str] = Counter()
    stns_kind_totals: Counter[str] = Counter()

    for label, blob in blobs:
        info = parse_psch_container(blob)
        info["label"] = label
        parsed.append(info)
        all_structs.update(info["_struct_hashes"])
        all_enums.update(info["_enum_hashes"])
        all_members.update(info["_member_hashes"])
        all_evals.update(info["_enum_value_hashes"])
        all_sigs.update(info["_signatures"])
        mtype_totals.update(info["member_type_counts"])
        anon_totals.update(info["pmap_anon_kinds"])
        chain_totals["+".join(info["section_tags"])] += 1
        for k in info["stns_kinds"]:
            stns_kind_totals[k] += 1

    # Measure struct reference resolution locally vs across containers in the scan.
    struct_refs_local = struct_refs_cross = struct_refs_unresolved = 0
    for info in parsed:
        local_s = info["_struct_hashes"]
        for s in info["_raw_structs"]:
            for m in s["members"]:
                if m["type"] == PSCH_TYPE_STRUCT and m["ref"] != 0:
                    if m["ref"] in local_s:
                        struct_refs_local += 1
                    elif m["ref"] in all_structs:
                        struct_refs_cross += 1
                    else:
                        struct_refs_unresolved += 1

    valid_count = sum(1 for p in parsed if p["valid"])
    return {
        "valid": valid_count == len(parsed),
        "containers": len(parsed),
        "valid_containers": valid_count,
        "section_chains": dict(sorted(chain_totals.items())),
        "pmap_blocks": sum(p["pmap_blocks"] for p in parsed),
        "pmap_struct_blocks": sum(p["pmap_struct_blocks"] for p in parsed),
        "pmap_anon_kinds": dict(sorted(anon_totals.items())),
        "pmap_exact_end": sum(1 for p in parsed if p["pmap_exact_end"]),
        "structs": sum(p["structs"] for p in parsed),
        "distinct_structs": len(all_structs),
        "members": sum(p["members"] for p in parsed),
        "distinct_members": len(all_members),
        "enums": sum(p["enums"] for p in parsed),
        "distinct_enums": len(all_enums),
        "enum_values": sum(p["enum_values"] for p in parsed),
        "distinct_enum_values": len(all_evals),
        "distinct_signatures": len(all_sigs),
        "member_type_counts": dict(sorted(mtype_totals.items())),
        "struct_refs": {
            "local": struct_refs_local,
            "cross_file": struct_refs_cross,
            "unresolved": struct_refs_unresolved,
        },
        "stns_blocks": sum(p["stns_blocks"] for p in parsed),
        "stns_records": sum(p["stns_records"] for p in parsed),
        "stns_kinds": dict(sorted(stns_kind_totals.items())),
        "stdn_containers": sum(1 for p in parsed if p["stdn_match"] is not None),
        "stdn_matches": sum(1 for p in parsed if p["stdn_match"] is True),
        "stfn_containers": sum(1 for p in parsed if p["stfn_bytes"] is not None),
    }


def check_report_path(out_path: str, target: str, carve_dir: str | None = None,
                      rows: list[dict] | None = None) -> None:
    """Refuse a --json destination that is a directory or collides with any input file."""
    if os.path.isdir(out_path):
        raise ValueError(f"--json is a directory: {out_path}")
    if carve_dir:
        idx = os.path.join(carve_dir, "index.json")
        if same_report_path(out_path, carve_dir) or same_report_path(out_path, idx):
            raise ValueError(f"--json would overwrite the input {carve_dir}")
    if os.path.isdir(target):
        if os.path.isfile(os.path.join(target, "index.json")):
            gm.check_report_path(out_path, target, rows=rows)
        elif same_report_path(out_path, target):
            raise ValueError(f"--json would overwrite the input {target}")
        else:
            for name in os.listdir(target):
                if name.lower().endswith(".bin") and same_report_path(out_path, os.path.join(target, name)):
                    raise ValueError(f"--json would overwrite the input {name}")
    elif same_report_path(out_path, target):
        raise ValueError(f"--json would overwrite the input {target}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="PSIN file, carve folder, or .rpf archive")
    ap.add_argument("--carve-dir", help="existing rpf9_carve.py index.json folder for an .rpf archive")
    ap.add_argument("--json", help="write aggregate decoded counts to this JSON file")
    a = ap.parse_args(argv)

    if not os.path.exists(a.target):
        print(f"ERROR: cannot read {a.target}: no such file or folder", file=sys.stderr)
        return 2
    try:
        if a.json:
            rows = (
                gm._read_index_rows(a.target)
                if os.path.isdir(a.target) and os.path.isfile(os.path.join(a.target, "index.json"))
                else None
            )
            check_report_path(a.json, a.target, a.carve_dir, rows=rows)
        summary = scan_target(a.target, carve_dir=a.carve_dir)
    except (OSError, ValueError, KeyError, TypeError, struct.error, RuntimeError) as e:
        print("error:", e, file=sys.stderr)
        return 2 if isinstance(e, ValueError) and "--json" in str(e) else 1

    print(
        f'PSCH containers: {summary["valid_containers"]}/{summary["containers"]} valid '
        f'(pmap blocks={summary["pmap_blocks"]}: {summary["pmap_struct_blocks"]} struct, '
        f'exact_end={summary["pmap_exact_end"]}/{summary["containers"]})'
    )
    print(
        f'  schema: {summary["structs"]} structs ({summary["distinct_structs"]} distinct), '
        f'{summary["members"]} members ({summary["distinct_members"]} distinct), '
        f'{summary["enums"]} enums ({summary["distinct_enums"]} distinct), '
        f'{summary["enum_values"]} values ({summary["distinct_enum_values"]} distinct), '
        f'{summary["distinct_signatures"]} signatures'
    )
    print(
        f'  refs: local={summary["struct_refs"]["local"]}, '
        f'cross={summary["struct_refs"]["cross_file"]}, '
        f'unresolved={summary["struct_refs"]["unresolved"]}'
    )
    print(
        f'  STNS: {summary["stns_blocks"]} blocks, {summary["stns_records"]} records '
        f'(STDN match={summary["stdn_matches"]}/{summary["stdn_containers"]}, '
        f'STFN={summary["stfn_containers"]})'
    )

    if a.json:
        try:
            gm._make_parent(a.json)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=1)
        except OSError as e:
            print(f"error: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 1
        print("wrote", a.json)

    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
