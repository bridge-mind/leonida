"""Distinguish and parse standalone `0DRP`/`0x7a` resources vs `PSIN` `LRSC` records.

`0DRP` + version `122` (`0x7a`) occurs in two distinct placements across RPF9 archives:

1. `standalone` (`0x7a` metadata or map-placement resource):
   Starts an inflated RAGE system stream with an 8-byte zero prefix, a `0x50000000`-based
   page-map pointer at `+0x08`, and the 80-byte `0DRP` header at `+0x10..+0x5f` (`0x60` bytes
   total). Table slots at `+0x28` (structs, 32 B each) and `+0x38` (instance blocks, 16 B each)
   hold `0x50000000`-based virtual pointers (`+0x30` enums and `+0x40` hold `0` or a
   `0x50000000`-based pointer), and `+0x48` stores four `u16` values
   `(structs, enums, blocks, unknown_4e)`. When a table or virtual block crosses the first
   stream's end, contiguous zlib streams continue the resource.

2. `psin_lrsc` (`PSIN` `LRSC` root/count summary record):
   Sits at `+0x00` of an 80-byte `LRSC` section payload (`"LRSC"` + big-endian `88` = 88 bytes
   total) near the end of a `PSIN` container, followed by a 20-byte `CHKS` section. It reuses
   the same 80-byte `0DRP` header frame without the 16-byte page prefix, but fills all four
   table-pointer slots (`+0x18..+0x37`) and `+0x40..+0x47` with `0xbb` (`0xbbbbbbbbbbbbbbbb`)
   because structs, enums and instance blocks live in `PSIN`'s own `LSCH`, `LENM` and `+0x08`
   block table. Its `+0x08..+0x0f` qword is zero (`+0x0c` is never a page count), `u32@+0x10`
   is the 1-based root block index in `1..blocks`, and the four `u16`s at `+0x38` are
   `(structs, enums, blocks, 0)`.

This module parses and validates both placements without misreading one as the other, and scans
RPF9 data areas, carved `.bin` directories, or single carved files. Output contains structural
counts and offsets only; no game payloads, names, or asset hashes are written.

Usage:
    python -I tools/odrp_7a.py <source> <out.json|out.csv> [--start N] [--limit BYTES]
                               [--max-inflate BYTES] [--quiet]
"""
import argparse
import csv
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

SIGNATURE = b"0DRP"
VERSION = 0x7A
PSIN_MAGIC = b"PSIN"
LRSC_TAG = b"LRSC"
CHKS_TAG = b"CHKS"

STANDALONE_HEADER_SIZE = 0x60
ODRP_FRAME_SIZE = 0x50
LRSC_PAYLOAD_SIZE = 0x50
LRSC_SECTION_SIZE = 8 + LRSC_PAYLOAD_SIZE
CHKS_SECTION_SIZE = 20

VIRTUAL_BASE = 0x50000000
PHYSICAL_BASE = 0x60000000
SEGMENT_SPAN = 0x10000000

LRSC_PTR_FILL_32 = b"\xbb" * 32
LRSC_TAIL_FILL_8 = b"\xbb" * 8
BLOCK_ROW_FILL_8 = b"\xaa" * 8
BLOCK_TABLE_END = 0x70707070

ARRAY_ELEMENT_HASH = 0x100
ARRAY_MEMBER_TYPE = 0x5C
ENTITY_STRUCT = 0xCE501483
MAPDATA_STRUCT = 0xD3593FA6

MAX_INFLATE = 64 << 20
MAX_CONTINUATIONS = 32
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")

COLUMNS = [
    "offset",
    "stored",
    "size",
    "streams",
    "placement",
    "framing",
    "version",
    "complete",
    "resource_kind",
    "structs",
    "enums",
    "blocks",
    "root_objects",
    "root_block",
    "entities",
    "schema_checked",
    "error",
]


def _u16(data, at):
    return struct.unpack_from("<H", data, at)[0] if 0 <= at and at + 2 <= len(data) else None


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if 0 <= at and at + 4 <= len(data) else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if 0 <= at and at + 8 <= len(data) else None


def _be32(data, at):
    return struct.unpack_from(">I", data, at)[0] if 0 <= at and at + 4 <= len(data) else None


def virtual_offset(pointer):
    """Resolve a zero-extended `0x50000000`-based virtual pointer to its logical resource offset."""
    if pointer is None:
        return None
    if (pointer >> 32) == 0 and VIRTUAL_BASE <= pointer < VIRTUAL_BASE + SEGMENT_SPAN:
        return pointer - VIRTUAL_BASE
    return None


def physical_offset(pointer):
    """Resolve a zero-extended `0x60000000`-based physical/graphics pointer to its segment offset."""
    if pointer is None:
        return None
    if (pointer >> 32) == 0 and PHYSICAL_BASE <= pointer < PHYSICAL_BASE + SEGMENT_SPAN:
        return pointer - PHYSICAL_BASE
    return None


def _optional_virtual_offset(pointer):
    """Allow either `0` (absent optional table) or a `0x50000000`-based virtual pointer."""
    if pointer == 0:
        return 0
    return virtual_offset(pointer)


def parse_standalone_header(data):
    """Validate the fixed `0x60`-byte header of a standalone `0DRP`/`0x7a` resource.

    Returns a header dict when `data` begins with the 16-byte RAGE system-page prefix
    (`8` zero bytes + `0x50000000` pointer) followed by `"0DRP"`, version `122`, and valid
    `0x50000000` table pointers; otherwise returns `None`.
    """
    if len(data) < STANDALONE_HEADER_SIZE:
        return None
    if data[0x00:0x08] != b"\x00" * 8:
        return None
    page_map = virtual_offset(_u64(data, 0x08))
    if page_map is None:
        return None
    if data[0x10:0x14] != SIGNATURE or _u32(data, 0x14) != VERSION:
        return None
    if data[0x18:0x20] != b"\x00" * 8:
        return None
    root_objects = _u32(data, 0x20)
    if _u32(data, 0x24) != 0:
        return None
    q28 = virtual_offset(_u64(data, 0x28))
    q30 = _optional_virtual_offset(_u64(data, 0x30))
    q38 = virtual_offset(_u64(data, 0x38))
    q40 = _optional_virtual_offset(_u64(data, 0x40))
    q50 = _optional_virtual_offset(_u64(data, 0x50))
    if None in (q28, q30, q38, q40, q50):
        return None
    if data[0x58:0x60] != b"\x00" * 8:
        return None
    structs, enums, blocks, unknown_4e = struct.unpack_from("<4H", data, 0x48)
    if not (1 <= structs <= 4096 and enums <= 4096 and 1 <= blocks <= 4096):
        return None
    if enums > 0 and _u64(data, 0x30) == 0:
        return None
    return {
        "page_map": page_map,
        "version": VERSION,
        "root_objects": root_objects,
        "structs": structs,
        "enums": enums,
        "blocks": blocks,
        "unknown_4e": unknown_4e,
        "tables": {
            "q28": q28,
            "q30": None if _u64(data, 0x30) == 0 else q30,
            "q38": q38,
            "q40": None if _u64(data, 0x40) == 0 else q40,
            "q50": None if _u64(data, 0x50) == 0 else q50,
        },
    }


def parse_standalone(data):
    """Parse and validate a standalone `0DRP`/`0x7a` resource (single or joined system bytes).

    Returns `None` when `data` does not start with a valid standalone `0DRP`/`0x7a` header.
    When the header validates, checks struct/member/enum/instance tables:
    - If a table or virtual block extends past `len(data)`, returns `complete=False` and
      `needs_continuation=True` so an archive/directory scanner can append continuation streams.
    - If an in-bounds pointer, struct size, or member offset violates the `0x7a` schema rules,
      returns `complete=False` and `needs_continuation=False` with a descriptive `error`.
    """
    hdr = parse_standalone_header(data)
    if hdr is None:
        return None

    size = len(data)
    nst = hdr["structs"]
    nen = hdr["enums"]
    nbl = hdr["blocks"]
    q28 = hdr["tables"]["q28"]
    q30 = hdr["tables"]["q30"]
    q38 = hdr["tables"]["q38"]

    base_row = {
        "placement": "standalone",
        "framing": "resource",
        "version": VERSION,
        "complete": False,
        "needs_continuation": False,
        "resource_kind": "meta",
        "structs": nst,
        "enums": nen,
        "blocks": nbl,
        "root_objects": hdr["root_objects"],
        "root_block": None,
        "unknown_4e": hdr["unknown_4e"],
        "virtual_blocks": 0,
        "physical_blocks": 0,
        "entities": 0,
        "schema_checked": False,
        "error": "",
    }

    if q28 + 32 * nst > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "struct table extends past stream end"
        return base_row
    if nen > 0 and q30 is not None and q30 + 24 * nen > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "enum table extends past stream end"
        return base_row
    if q38 + 16 * nbl > size:
        base_row["needs_continuation"] = True
        base_row["error"] = "instance table extends past stream end"
        return base_row

    struct_sizes = {}
    for i in range(nst):
        off = q28 + 32 * i
        shash, _sig, _flags, _zero0c, mptr_raw, ssize, _zero1c, mcount = struct.unpack_from(
            "<IIIIQIHH", data, off
        )
        if ssize > (1 << 20) or mcount > 256:
            base_row["error"] = f"struct {i} exceeds size or member cap"
            return base_row
        if mcount > 0:
            moff = virtual_offset(mptr_raw)
            if moff is None:
                base_row["error"] = f"struct {i} has non-virtual member pointer"
                return base_row
            if moff + 16 * mcount > size:
                base_row["needs_continuation"] = True
                base_row["error"] = f"struct {i} member table extends past stream end"
                return base_row
            for j in range(mcount):
                mhash, field_off, mtype, _sub, elem_idx, _ref = struct.unpack_from(
                    "<IIBBHI", data, moff + 16 * j
                )
                if mhash != ARRAY_ELEMENT_HASH and field_off >= ssize:
                    base_row["error"] = f"struct {i} member {j} offset {field_off} >= size {ssize}"
                    return base_row
                if mtype == ARRAY_MEMBER_TYPE:
                    if elem_idx >= mcount:
                        base_row["error"] = f"struct {i} array member {j} index {elem_idx} out of range"
                        return base_row
                    desc_hash = _u32(data, moff + 16 * elem_idx)
                    if desc_hash != ARRAY_ELEMENT_HASH:
                        base_row["error"] = f"struct {i} array member {j} target is not 0x100"
                        return base_row
        struct_sizes.setdefault(shash, ssize)

    virtual_blocks = 0
    physical_blocks = 0
    resolved_struct_block = False
    needs_more_blocks = False
    block_list = []

    for i in range(nbl):
        off = q38 + 16 * i
        bhash, bsize, bptr = struct.unpack_from("<IIQ", data, off)
        if bsize > (1 << 24):
            base_row["error"] = f"block {i} exceeds 16 MiB cap"
            return base_row
        voff = virtual_offset(bptr)
        if voff is not None:
            virtual_blocks += 1
            if voff + bsize > size:
                needs_more_blocks = True
            else:
                if bhash in struct_sizes:
                    resolved_struct_block = True
            block_list.append((bhash, bsize, voff))
            continue
        poff = physical_offset(bptr)
        if poff is not None:
            physical_blocks += 1
            block_list.append((bhash, bsize, None))
            continue
        base_row["error"] = f"block {i} has invalid segment pointer"
        return base_row

    base_row["virtual_blocks"] = virtual_blocks
    base_row["physical_blocks"] = physical_blocks

    if needs_more_blocks:
        base_row["needs_continuation"] = True
        base_row["error"] = "virtual block payload extends past stream end"
        return base_row

    if not resolved_struct_block:
        base_row["error"] = "no virtual block matches a declared struct"
        return base_row

    entity_size = struct_sizes.get(ENTITY_STRUCT, 0)
    map_size = struct_sizes.get(MAPDATA_STRUCT, 0)
    if entity_size > 0 and map_size > 0:
        base_row["resource_kind"] = "map"
        base_row["entities"] = sum(
            bsize // entity_size
            for bhash, bsize, voff in block_list
            if bhash == ENTITY_STRUCT and voff is not None and voff + bsize <= size and bsize % entity_size == 0
        )

    base_row["complete"] = True
    base_row["schema_checked"] = True
    return base_row


def parse_lrsc_payload(payload):
    """Validate an 80-byte `PSIN` `LRSC` `0DRP` payload.

    Returns a dict on an exact 80-byte `LRSC` payload with `"0DRP"` + `122`, zero `+0x08..+0x0f`,
    `1 <= root_block <= blocks`, `0xbb` fill at `+0x18..+0x37` and `+0x40..+0x47`, `u16@+0x3e == 0`,
    and zero `+0x48..+0x4f`; otherwise returns `None`.
    """
    if len(payload) != LRSC_PAYLOAD_SIZE:
        return None
    if payload[0x00:0x04] != SIGNATURE or _u32(payload, 0x04) != VERSION:
        return None
    if payload[0x08:0x10] != b"\x00" * 8:
        return None
    root_block, root_hi = struct.unpack_from("<II", payload, 0x10)
    if root_hi != 0:
        return None
    if payload[0x18:0x38] != LRSC_PTR_FILL_32:
        return None
    structs, enums, blocks, reserved = struct.unpack_from("<4H", payload, 0x38)
    if reserved != 0 or blocks == 0 or not (1 <= root_block <= blocks):
        return None
    if payload[0x40:0x48] != LRSC_TAIL_FILL_8:
        return None
    if payload[0x48:0x50] != b"\x00" * 8:
        return None
    return {
        "version": VERSION,
        "root_block": root_block,
        "structs": structs,
        "enums": enums,
        "blocks": blocks,
    }


def _header_count(payload, header_size, elem_size, count_at, count_fmt):
    if not payload:
        return 0
    elements = 0
    for k in range(1, len(payload) // header_size + 1):
        elements += struct.unpack_from(count_fmt, payload, header_size * (k - 1) + count_at)[0]
        if header_size * k + elem_size * elements == len(payload):
            return k
    return None


def _psin_block_count(blob, first_section):
    """Count rows in a `PSIN` `+0x08` instance block table, or return `None` if malformed."""
    if first_section < 16 or first_section > len(blob):
        return None
    at = 8
    count = 0
    sizes = []
    while at + 8 <= first_section:
        kind = _u32(blob, at)
        if kind == BLOCK_TABLE_END:
            break
        if at + 16 > first_section or blob[at + 8:at + 16] != BLOCK_ROW_FILL_8:
            return None
        sizes.append(_u32(blob, at + 4))
        count += 1
        at += 16
    else:
        return None
    end = (at + 15) & ~15
    if end > first_section or blob[at:end] != b"\x70" * (end - at):
        return None
    pos = end
    for bsize in sizes:
        if pos + bsize > first_section:
            return None
        pos += (bsize + 15) & ~15
    if pos != first_section:
        return None
    return count


def _parse_psin_container(blob):
    """Walk a `PSIN` container to locate and cross-check its `LRSC` `0DRP` record.

    Returns:
      - a record dict when `LRSC` contains `"0DRP"` (complete or truncated), OR when a `PSIN`
        stream is truncated before reaching `LRSC` (`needs_continuation=True`, `has_odrp=False`),
      - `None` when the `PSIN` container is complete and has no `LRSC` section (e.g. `PMAP`/`PSCH`).
    """
    first = _be32(blob, 4)
    if first is None or first < 16:
        return None
    if first + 8 > len(blob):
        return {
            "has_odrp": False,
            "needs_continuation": True,
            "error": "first section offset extends past stream end",
        }
    pos = first
    lsch_count = None
    lenm_count = None
    lrsc_off = None
    lrsc_size = None
    walk_error = ""
    needs_cont = False

    while pos + 8 <= len(blob):
        tag = blob[pos:pos + 4]
        if any(c < 0x20 or c > 0x7E for c in tag):
            walk_error = f"non-ASCII section tag at {pos}"
            break
        ssize = _be32(blob, pos + 4)
        if ssize is None or ssize < 8:
            walk_error = f"invalid section size at {pos}"
            break
        if pos + ssize > len(blob):
            needs_cont = True
            walk_error = f"section {tag.decode('ascii')} extends past stream end"
            if tag == LRSC_TAG:
                lrsc_off = pos
                lrsc_size = ssize
            break
        pay = blob[pos + 8:pos + ssize]
        if tag == b"LSCH" and lsch_count is None:
            lsch_count = _header_count(pay, 32, 16, 30, "<H")
        elif tag == b"LENM" and lenm_count is None:
            lenm_count = _header_count(pay, 24, 16, 16, "<I")
        elif tag == LRSC_TAG and lrsc_off is None:
            lrsc_off = pos
            lrsc_size = ssize
        pos += ssize

    if not walk_error and pos != len(blob):
        if len(blob) - pos < 8:
            needs_cont = True
        walk_error = f"section walk ended at {pos} of {len(blob)}"

    if lrsc_off is None:
        if needs_cont:
            return {"has_odrp": False, "needs_continuation": True, "error": walk_error}
        return None

    avail_pay = blob[lrsc_off + 8:min(len(blob), lrsc_off + lrsc_size)]
    if len(avail_pay) >= 4 and avail_pay[:4] != SIGNATURE:
        return None
    if len(avail_pay) < LRSC_PAYLOAD_SIZE or lrsc_off + lrsc_size > len(blob):
        return {
            "has_odrp": True,
            "placement": "psin_lrsc",
            "framing": "psin",
            "version": _u32(avail_pay, 4) if len(avail_pay) >= 8 else VERSION,
            "complete": False,
            "needs_continuation": True,
            "resource_kind": None,
            "structs": None,
            "enums": None,
            "blocks": None,
            "root_objects": None,
            "root_block": _u32(avail_pay, 0x10) if len(avail_pay) >= 0x14 else None,
            "entities": None,
            "schema_checked": False,
            "error": walk_error or "truncated LRSC payload",
        }

    if lrsc_size != LRSC_SECTION_SIZE:
        return {
            "has_odrp": True,
            "placement": "psin_lrsc",
            "framing": "psin",
            "version": _u32(avail_pay, 4),
            "complete": False,
            "needs_continuation": False,
            "resource_kind": None,
            "structs": None,
            "enums": None,
            "blocks": None,
            "root_objects": None,
            "root_block": None,
            "entities": None,
            "schema_checked": False,
            "error": f"LRSC section size is {lrsc_size}, expected {LRSC_SECTION_SIZE}",
        }

    parsed = parse_lrsc_payload(avail_pay)
    if parsed is None:
        return {
            "has_odrp": True,
            "placement": "psin_lrsc",
            "framing": "psin",
            "version": _u32(avail_pay, 4),
            "complete": False,
            "needs_continuation": False,
            "resource_kind": None,
            "structs": None,
            "enums": None,
            "blocks": None,
            "root_objects": None,
            "root_block": None,
            "entities": None,
            "schema_checked": False,
            "error": "LRSC payload failed 0DRP/0x7a validation",
        }

    block_count = _psin_block_count(blob, first)
    mismatches = []
    if lsch_count is not None and parsed["structs"] != lsch_count:
        mismatches.append(f"structs {parsed['structs']} != LSCH {lsch_count}")
    if lenm_count is not None and parsed["enums"] != lenm_count:
        mismatches.append(f"enums {parsed['enums']} != LENM {lenm_count}")
    if block_count is not None and parsed["blocks"] != block_count:
        mismatches.append(f"blocks {parsed['blocks']} != block_table {block_count}")

    err = "; ".join(x for x in (walk_error, ", ".join(mismatches)) if x)
    checked = (
        not err
        and lsch_count is not None
        and lenm_count is not None
        and block_count is not None
    )
    return {
        "has_odrp": True,
        "placement": "psin_lrsc",
        "framing": "psin",
        "version": VERSION,
        "complete": not err,
        "needs_continuation": needs_cont,
        "resource_kind": None,
        "structs": parsed["structs"],
        "enums": parsed["enums"],
        "blocks": parsed["blocks"],
        "root_objects": None,
        "root_block": parsed["root_block"],
        "entities": None,
        "schema_checked": checked,
        "error": err,
    }


def parse_psin_lrsc(blob):
    """Parse a `PSIN` `LRSC` `0DRP`/`0x7a` record from any of its four framings:

    - `"psin"`: a `PSIN` container starting with `"PSIN"` at `+0x00`;
    - `"lrsc_section"`: a bare 88-byte `"LRSC"` section (`"LRSC"` + big-endian `88` + payload);
    - `"lrsc_payload"`: a bare 80-byte `LRSC` `0DRP` payload starting with `"0DRP"` at `+0x00`;
    - `"psin_tail"`: an unjoined `PSIN` continuation stream ending with an 88-byte `"LRSC"`
      section + 20-byte `"CHKS"` section at EOF.

    Returns a record dict when `blob` matches one of those `LRSC` framings, or `None` otherwise.
    """
    if len(blob) >= 16 and blob[:4] == PSIN_MAGIC:
        res = _parse_psin_container(blob)
        if res is None or not res.get("has_odrp"):
            return None
        out = dict(res)
        out.pop("has_odrp", None)
        return out

    if len(blob) >= 8 and blob[:4] == LRSC_TAG:
        declared = _be32(blob, 4)
        if declared != LRSC_SECTION_SIZE or len(blob) != LRSC_SECTION_SIZE:
            return {
                "placement": "psin_lrsc",
                "framing": "lrsc_section",
                "version": _u32(blob, 12) if len(blob) >= 16 else None,
                "complete": False,
                "needs_continuation": len(blob) < LRSC_SECTION_SIZE,
                "resource_kind": None,
                "structs": None,
                "enums": None,
                "blocks": None,
                "root_objects": None,
                "root_block": None,
                "entities": None,
                "schema_checked": False,
                "error": f"LRSC section length {len(blob)} (declared {declared}) != {LRSC_SECTION_SIZE}",
            }
        parsed = parse_lrsc_payload(blob[8:])
        if parsed is None:
            return {
                "placement": "psin_lrsc",
                "framing": "lrsc_section",
                "version": _u32(blob, 12),
                "complete": False,
                "needs_continuation": False,
                "resource_kind": None,
                "structs": None,
                "enums": None,
                "blocks": None,
                "root_objects": None,
                "root_block": None,
                "entities": None,
                "schema_checked": False,
                "error": "LRSC payload failed 0DRP/0x7a validation",
            }
        return {
            "placement": "psin_lrsc",
            "framing": "lrsc_section",
            "version": VERSION,
            "complete": True,
            "needs_continuation": False,
            "resource_kind": None,
            "structs": parsed["structs"],
            "enums": parsed["enums"],
            "blocks": parsed["blocks"],
            "root_objects": None,
            "root_block": parsed["root_block"],
            "entities": None,
            "schema_checked": False,
            "error": "",
        }

    if len(blob) >= 4 and blob[:4] == SIGNATURE:
        parsed = parse_lrsc_payload(blob)
        if parsed is not None:
            return {
                "placement": "psin_lrsc",
                "framing": "lrsc_payload",
                "version": VERSION,
                "complete": True,
                "needs_continuation": False,
                "resource_kind": None,
                "structs": parsed["structs"],
                "enums": parsed["enums"],
                "blocks": parsed["blocks"],
                "root_objects": None,
                "root_block": parsed["root_block"],
                "entities": None,
                "schema_checked": False,
                "error": "",
            }
        # Reject stripped standalone headers (`0x50000000` pointers at +0x18/+0x28) or malformed
        # `0DRP@0x0` buffers explicitly rather than guessing a placement.
        return None

    tail_len = LRSC_SECTION_SIZE + CHKS_SECTION_SIZE
    if len(blob) >= tail_len and parse_standalone_header(blob) is None:
        lrsc_at = len(blob) - tail_len
        chks_at = len(blob) - CHKS_SECTION_SIZE
        if (
            blob[lrsc_at:lrsc_at + 4] == LRSC_TAG
            and _be32(blob, lrsc_at + 4) == LRSC_SECTION_SIZE
            and blob[chks_at:chks_at + 4] == CHKS_TAG
            and _be32(blob, chks_at + 4) == CHKS_SECTION_SIZE
        ):
            parsed = parse_lrsc_payload(blob[lrsc_at + 8:chks_at])
            if parsed is not None:
                return {
                    "placement": "psin_lrsc",
                    "framing": "psin_tail",
                    "version": VERSION,
                    "complete": True,
                    "needs_continuation": False,
                    "resource_kind": None,
                    "structs": parsed["structs"],
                    "enums": parsed["enums"],
                    "blocks": parsed["blocks"],
                    "root_objects": None,
                    "root_block": parsed["root_block"],
                    "entities": None,
                    "schema_checked": False,
                    "error": "",
                }
    return None


def classify_blob(blob):
    """Classify and parse a single buffer as `standalone` or `psin_lrsc`, or `None` if neither."""
    standalone = parse_standalone(blob)
    if standalone is not None:
        return standalone
    return parse_psin_lrsc(blob)


def _starts_new_container(data):
    return (
        (len(data) >= 16 and data[:4] == PSIN_MAGIC)
        or parse_standalone_header(data) is not None
    )


def _finalize_row(record, offset, stored, size, streams):
    out = dict(record)
    out.pop("needs_continuation", None)
    out.pop("unknown_4e", None)
    out.pop("virtual_blocks", None)
    out.pop("physical_blocks", None)
    out["offset"] = offset
    out["stored"] = stored
    out["size"] = size
    out["streams"] = streams
    return out


class _StreamConsumer:
    """Bounded-memory stream consumer that joins contiguous `standalone` and `PSIN` continuations."""

    def __init__(self, max_inflate=MAX_INFLATE):
        self.max_inflate = max_inflate
        self.rows = []
        self.pending = None

    def _flush_pending(self):
        if self.pending is None:
            return
        item = self.pending
        self.pending = None
        if item["kind"] == "standalone":
            self.rows.append(
                _finalize_row(item["rec"], item["offset"], item["stored"], len(item["data"]), item["streams"])
            )
        elif item["rec"] is not None and item["rec"].get("has_odrp"):
            rec = dict(item["rec"])
            rec.pop("has_odrp", None)
            self.rows.append(
                _finalize_row(rec, item["offset"], item["stored"], len(item["data"]), item["streams"])
            )

    def feed(self, offset, stored, data, inflated_size=None):
        actual_size = len(data) if inflated_size is None else inflated_size
        next_off = ((offset + stored + 15) & ~15) if stored is not None else None

        if self.pending is not None:
            item = self.pending
            can_join = (
                item["next_off"] is not None
                and item["next_off"] == offset
                and item["streams"] <= MAX_CONTINUATIONS
                and not _starts_new_container(data)
                and actual_size == len(data)
                and len(item["data"]) + len(data) <= self.max_inflate
            )
            if can_join:
                item["data"].extend(data)
                if item["stored"] is not None and stored is not None:
                    item["stored"] += stored
                item["next_off"] = next_off
                item["streams"] += 1
                joined = bytes(item["data"])
                if item["kind"] == "standalone":
                    rec = parse_standalone(joined)
                else:
                    rec = _parse_psin_container(joined)
                item["rec"] = rec
                if rec is None or not rec.get("needs_continuation"):
                    self._flush_pending()
                return
            self._flush_pending()

        if parse_standalone_header(data) is not None:
            rec = parse_standalone(data)
            if rec["needs_continuation"] and next_off is not None:
                self.pending = {
                    "kind": "standalone",
                    "offset": offset,
                    "stored": stored,
                    "next_off": next_off,
                    "data": bytearray(data),
                    "streams": 1,
                    "rec": rec,
                }
            else:
                self.rows.append(_finalize_row(rec, offset, stored, actual_size, 1))
            return

        if len(data) >= 16 and data[:4] == PSIN_MAGIC:
            res = _parse_psin_container(data)
            if res is not None and res["needs_continuation"] and next_off is not None:
                self.pending = {
                    "kind": "psin",
                    "offset": offset,
                    "stored": stored,
                    "next_off": next_off,
                    "data": bytearray(data),
                    "streams": 1,
                    "rec": res,
                }
            elif res is not None and res.get("has_odrp"):
                rec = dict(res)
                rec.pop("has_odrp", None)
                self.rows.append(_finalize_row(rec, offset, stored, actual_size, 1))
            return

        rec = parse_psin_lrsc(data)
        if rec is not None:
            self.rows.append(_finalize_row(rec, offset, stored, actual_size, 1))

    def finish(self):
        self._flush_pending()
        return self.rows


def _gap_bytes(gap):
    return (len(gap), 0) if gap.count(0) == len(gap) else (0, len(gap))


def scan_rpf(path, start=None, limit=None, max_inflate=MAX_INFLATE):
    """Walk an RPF9 archive data area inside `[start, start + limit)`. No TOC is touched."""
    t0 = time.time()
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("empty file")

    consumer = _StreamConsumer(max_inflate=max_inflate)
    tail_keep = LRSC_SECTION_SIZE + CHKS_SECTION_SIZE
    stream_bytes = 0
    inflated_bytes = 0
    stream_count = 0
    zero = unread = skipped = 0

    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            size = len(m)
            first = _u32(m, 4) if size >= 8 else 0
            pos = start if start is not None else first
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit is not None else size
            span = end - pos
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
                    consumer._flush_pending()
                    z, u = _gap_bytes(m[last:s])
                    zero += z
                    unread += u
                    stream_count += 1
                    stream_bytes += blen
                    inflated_bytes += blen
                    last = pos = s + blen
                    continue

                if consumer.pending is not None and consumer.pending["next_off"] != s:
                    consumer._flush_pending()
                keep_full = consumer.pending is not None

                d = zlib.decompressobj()
                p = s
                total = 0
                head = bytearray()
                tail = bytearray()
                chunks = []
                decided = False
                too_big = False
                try:
                    while not d.eof and p < end:
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
                                if keep_full:
                                    chunks.append(chunk)
                                elif not decided:
                                    chunks.append(chunk)
                                    if len(head) < STANDALONE_HEADER_SIZE:
                                        head.extend(chunk[:STANDALONE_HEADER_SIZE - len(head)])
                                    if len(head) >= STANDALONE_HEADER_SIZE:
                                        decided = True
                                        if head[:4] == PSIN_MAGIC or parse_standalone_header(head) is not None:
                                            keep_full = True
                                        else:
                                            joined_so_far = b"".join(chunks)
                                            tail = bytearray(joined_so_far[-tail_keep:])
                                            chunks.clear()
                                else:
                                    if len(chunk) >= tail_keep:
                                        tail = bytearray(chunk[-tail_keep:])
                                    else:
                                        tail.extend(chunk)
                                        if len(tail) > tail_keep:
                                            del tail[:-tail_keep]
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
                z, u = _gap_bytes(m[last:s])
                zero += z
                unread += u
                if keep_full or not decided:
                    data = b"".join(chunks)
                    consumer.feed(s, used, data, inflated_size=total)
                else:
                    # Non-container stream: head (0x60 B) + tail (<= 108 B) suffices for psin_tail check.
                    probe = bytes(head) + bytes(tail) if total > STANDALONE_HEADER_SIZE + len(tail) else bytes(head)
                    if total <= STANDALONE_HEADER_SIZE + tail_keep and len(probe) != total:
                        # Overlap between head and tail on short streams (64..196 B): reconstruct exact bytes.
                        overlap = STANDALONE_HEADER_SIZE + len(tail) - total
                        probe = bytes(head) + bytes(tail[overlap:])
                    consumer.feed(s, used, probe, inflated_size=total)
                stream_count += 1
                stream_bytes += used
                inflated_bytes += total
                last = pos = s + used

            z, u = _gap_bytes(m[last:end])
            zero += z
            unread += u

    rows = consumer.finish()
    summary = summarise(
        rows,
        mode="rpf9",
        source=path,
        streams=stream_count,
        compressed_bytes=stream_bytes,
        inflated_bytes=inflated_bytes,
        span=span,
        zero_padding=zero,
        unread_bytes=unread,
        skipped_candidates=skipped,
        elapsed=round(time.time() - t0, 3),
    )
    return rows, summary


def _carve_stored_map(folder):
    manifest = os.path.join(folder, "index.json")
    try:
        with open(manifest, encoding="utf-8") as fh:
            files = json.load(fh).get("files", [])
        return {int(item["off"]): int(item["stored"]) for item in files}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def _is_carved_file(path):
    return path.lower().endswith(".bin") and os.path.isfile(path)


def scan_dir(path, max_inflate=MAX_INFLATE):
    """Scan a directory of carved `.bin` files in bounded memory."""
    t0 = time.time()
    by_off = _carve_stored_map(path)
    candidates = []
    skipped = 0
    for idx, name in enumerate(sorted(os.listdir(path))):
        full = os.path.join(path, name)
        if not _is_carved_file(full):
            continue
        fsize = os.path.getsize(full)
        if fsize > max_inflate:
            skipped += 1
            continue
        m = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1), 16) if m else idx
        candidates.append((not bool(m), offset, full, fsize))

    candidates.sort(key=lambda item: (item[0], item[1]))
    consumer = _StreamConsumer(max_inflate=max_inflate)
    inflated = 0
    stored_sum = 0
    for _no_hex, offset, full, _fsize in candidates:
        with open(full, "rb") as fh:
            data = fh.read()
        stored = by_off.get(offset)
        inflated += len(data)
        stored_sum += stored if stored is not None else len(data)
        consumer.feed(offset, stored, data)

    rows = consumer.finish()
    summary = summarise(
        rows,
        mode="directory",
        source=path,
        streams=len(candidates),
        compressed_bytes=stored_sum,
        inflated_bytes=inflated,
        span=stored_sum,
        zero_padding=0,
        unread_bytes=0,
        skipped_candidates=skipped,
        elapsed=round(time.time() - t0, 3),
    )
    return rows, summary


def scan_file(path, max_inflate=MAX_INFLATE):
    """Parse a single carved/extracted binary file."""
    t0 = time.time()
    fsize = os.path.getsize(path)
    if fsize > max_inflate:
        return [], summarise(
            [],
            mode="file",
            source=path,
            streams=0,
            compressed_bytes=0,
            inflated_bytes=0,
            span=0,
            zero_padding=0,
            unread_bytes=0,
            skipped_candidates=1,
            elapsed=round(time.time() - t0, 3),
        )
    with open(path, "rb") as fh:
        data = fh.read()
    rec = classify_blob(data)
    rows = [_finalize_row(rec, 0, len(data), len(data), 1)] if rec is not None else []
    summary = summarise(
        rows,
        mode="file",
        source=path,
        streams=1,
        compressed_bytes=len(data),
        inflated_bytes=len(data),
        span=len(data),
        zero_padding=0,
        unread_bytes=0,
        skipped_candidates=0,
        elapsed=round(time.time() - t0, 3),
    )
    return rows, summary


def summarise(
    rows,
    *,
    mode,
    source,
    streams,
    compressed_bytes,
    inflated_bytes,
    span,
    zero_padding,
    unread_bytes,
    skipped_candidates,
    elapsed,
):
    standalone = [r for r in rows if r["placement"] == "standalone"]
    psin_lrsc = [r for r in rows if r["placement"] == "psin_lrsc"]
    framings = {}
    for r in rows:
        framings[r["framing"]] = framings.get(r["framing"], 0) + 1
    return {
        "mode": mode,
        "source": source,
        "streams": streams,
        "compressed_bytes": compressed_bytes,
        "inflated_bytes": inflated_bytes,
        "span": span,
        "zero_padding": zero_padding,
        "unread_bytes": unread_bytes,
        "readable_fraction": round(compressed_bytes / span, 6) if span else 0.0,
        "skipped_candidates": skipped_candidates,
        "records": len(rows),
        "complete": sum(1 for r in rows if r["complete"]),
        "incomplete": sum(1 for r in rows if not r["complete"]),
        "standalone": len(standalone),
        "standalone_complete": sum(1 for r in standalone if r["complete"]),
        "standalone_meta": sum(1 for r in standalone if r["resource_kind"] == "meta"),
        "standalone_map": sum(1 for r in standalone if r["resource_kind"] == "map"),
        "standalone_joined": sum(1 for r in standalone if r["streams"] > 1),
        "standalone_entities": sum(r["entities"] or 0 for r in standalone),
        "psin_lrsc": len(psin_lrsc),
        "psin_lrsc_complete": sum(1 for r in psin_lrsc if r["complete"]),
        "psin_lrsc_joined": sum(1 for r in psin_lrsc if r["streams"] > 1),
        "psin_lrsc_schema_checked": sum(1 for r in psin_lrsc if r["schema_checked"]),
        "framings": dict(sorted(framings.items())),
        "elapsed_seconds": elapsed,
    }


def write_output(path, rows, summary):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary, "records": rows}, fh, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for col in COLUMNS:
                val = row[col]
                if col == "offset":
                    out[col] = f"0x{val:x}"
                elif isinstance(val, bool):
                    out[col] = int(val)
                elif val is None:
                    out[col] = ""
                else:
                    out[col] = val
            writer.writerow(out)


def _offset_arg(text):
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _positive_bytes_arg(text):
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _same_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _is_rpf_file(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 4:
        return path.lower().endswith(".rpf")
    with open(path, "rb") as fh:
        return fh.read(4) == b"9FPR"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help=".rpf archive, carved directory, or single binary stream")
    ap.add_argument("out", help="output path ending in .json or .csv")
    ap.add_argument("--start", type=_offset_arg, help="first archive offset to scan (default: RPF data start)")
    ap.add_argument("--limit", type=_positive_bytes_arg, help="number of archive bytes to scan")
    ap.add_argument(
        "--max-inflate",
        type=_positive_bytes_arg,
        default=MAX_INFLATE,
        help="skip or cap resources inflating above this many bytes (default: 64 MiB)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if os.path.isdir(args.out):
        ap.error(f"{args.out} is a folder, not a .json or .csv file")
    ext = os.path.splitext(args.out)[1].lower()
    if ext not in (".json", ".csv"):
        ap.error("output must end in .json or .csv")
    parent = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(parent):
        ap.error(f"{parent} is not a folder, so {args.out} cannot be written")
    if _same_path(args.source, args.out):
        ap.error("output must not overwrite the source")

    if os.path.isdir(args.source):
        if args.start is not None or args.limit is not None:
            ap.error("--start and --limit apply to an .rpf archive, not a directory")
        manifest = os.path.join(args.source, "index.json")
        if os.path.isfile(manifest) and _same_path(manifest, args.out):
            ap.error(f"output would overwrite the carve manifest: {args.out}")
        if os.path.isfile(args.out):
            for name in os.listdir(args.source):
                cand = os.path.join(args.source, name)
                if _is_carved_file(cand) and _same_path(cand, args.out):
                    ap.error(f"output would overwrite an input file: {args.out}")
    elif os.path.isfile(args.source):
        is_rpf = _is_rpf_file(args.source)
        if not is_rpf and (args.start is not None or args.limit is not None):
            ap.error("--start and --limit apply to an .rpf archive, not a single stream file")
    else:
        sys.exit(f"{args.source}: no such file or folder")

    try:
        if os.path.isdir(args.source):
            rows, summary = scan_dir(args.source, args.max_inflate)
        elif _is_rpf_file(args.source):
            rows, summary = scan_rpf(args.source, args.start, args.limit, args.max_inflate)
        else:
            rows, summary = scan_file(args.source, args.max_inflate)
        write_output(args.out, rows, summary)
    except ValueError as exc:
        sys.exit(f"{args.source}: {exc}")
    except OSError as exc:
        sys.exit(f"{args.source}: {exc.strerror or exc}")

    if not args.quiet:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
