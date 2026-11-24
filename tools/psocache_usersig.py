"""Describe the root-signature / user-data side of the graphics PSO records in a PS\\0\\x09 cache.

Each graphics PSO record (0x70 bytes) names a stage record at +8 and an input-layout record at +0xC. The stage
record names the per-stage chunk records that carry the user data. A chunk record (0x41/0x42/0x43 list) is
decoded here as:

- 0x41 entry: kind, offset (w2), key (w3), count (w4), and w5 split into `a` (bits 16..31), `b` (bits 8..15)
  and element size (bits 0..7). For 0x300/0x302/0x303 the tool reads `b` as the register space, `a` as the base
  register and w4 as the count. For 0x401 it reads `a` as the number of 32-bit constants. The 0x2 kind is a
  sampler entry with no slot.
- 0x42 entry: the slot word splits as 0x00SS_IIRR. SS is read as the descriptor-table / root-parameter index,
  II as the D3D12_DESCRIPTOR_RANGE_TYPE order (0 SRV, 1 UAV, 2 CBV, 3 sampler) and RR as a width (1, or 2 for
  the extra slots). w4 is reported as found.
- 0x43 entry: a 16-byte value, not printed.

The 32-dword root-argument budget is computed as `constants + 2 * tables` (the layer's default mapping of two
dwords per descriptor table) and alternatively as `constants + 1 * tables` (the one-dword-per-table model
measured on the console shaders, see docs/d3d12_vulkan.md 3.5). A record fits when the total is at most 32.

The names and readings above are inferences from measured shapes, not a decoded driver format; the tool prints
the raw bytes it used next to each field so the reading can be checked. It never prints a hash or a path.

Usage:
    python -I psocache_usersig.py <cache> [--entry N] [--limit N] [--json] [--quiet]

--entry N prints one graphics PSO (0-based). --limit N samples the first N graphics PSOs (default: all).
Exit code: 0 on success, 2 when the cache cannot be read or the inspector reports a disagreement.
"""

import argparse
import collections
import json
import mmap
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psocache_inspect as pi  # noqa: E402

KIND_NAMES = {
    0x300: "SRV",
    0x302: "CBV",
    0x303: "UAV",
    0x2: "Sampler",
    0x401: "constants",
    0x100: "unknown-0x100",
}
DESCRIPTOR_KINDS = (0x300, 0x302, 0x303)
RANGE_NAMES = {0: "SRV", 1: "UAV", 2: "CBV", 3: "Sampler"}
STAGE_REFS = (0x4, 0x2C, 0x78, 0x90, 0x150, 0x1A4)
CONTRACT_DWORDS = 32


def describe_chunk(w):
    """One decoded chunk as a JSON-safe dict."""
    if w[0] == 0x41:
        kind = w[1]
        item = {
            "chunk": "0x41",
            "tag": "0x%03x" % kind,
            "name": KIND_NAMES.get(kind, "unknown"),
            "offset": w[2],
            "key": w[3],
            "count": w[4],
            "a": w[5] >> 16,
            "b": (w[5] >> 8) & 0xFF,
            "size": w[5] & 0xFF,
        }
        if kind in DESCRIPTOR_KINDS:
            item["space"] = item["b"]
            item["base_register"] = item["a"]
        elif kind == 0x401:
            item["constant_count"] = item["a"]
        return item
    if w[0] == 0x42:
        return {
            "chunk": "0x42",
            "tag": "0x42",
            "slot": "0x%06x" % w[2],
            "SS": (w[2] >> 16) & 0xFF,
            "II": (w[2] >> 8) & 0xFF,
            "RR": w[2] & 0xFF,
            "key": w[3],
            "w4": w[4],
        }
    return {"tag": "0x%x" % w[0]}


def describe_record(index, chunks):
    """Decode one chunk record and compute its 32-dword cost under both table models."""
    entries = []
    descriptors = []
    slots = []
    hashes = 0
    for w in chunks:
        if w[0] == 0x41:
            entries.append(describe_chunk(w))
            if w[1] in DESCRIPTOR_KINDS:
                descriptors.append((w, len(entries) - 1))
        elif w[0] == 0x42:
            slots.append(w)
        elif w[0] == 0x43:
            hashes += 1
    pairs = []
    for i, (w, entry_index) in enumerate(descriptors):
        slot = describe_chunk(slots[i]) if i < len(slots) else None
        pairs.append({"entry": entries[entry_index], "slot": slot})
    extras = [describe_chunk(w) for w in slots[len(descriptors) :]]
    tables = sorted({(w[2] >> 16) & 0xFF for w in slots})
    constants = sum(w[5] >> 16 for w in chunks if w[0] == 0x41 and w[1] == 0x401)
    cost_two = constants + 2 * len(tables)
    cost_one = constants + len(tables)
    kinds = collections.Counter(e["name"] for e in entries)
    return {
        "record": index,
        "chunks": len(chunks),
        "hashes": hashes,
        "entries": entries,
        "pairs": pairs,
        "extras": extras,
        "tables": tables,
        "constants": constants,
        "cost_two_per_table": cost_two,
        "cost_one_per_table": cost_one,
        "fits_two_per_table": cost_two <= CONTRACT_DWORDS,
        "fits_one_per_table": cost_one <= CONTRACT_DWORDS,
        "kinds": dict(kinds),
    }


def stage_record_refs(m, meta_off, meta_index, chunk_records, stage):
    """(stage offset, chunk record index) for every stage field that names a chunk record, deduplicated."""
    if not 0 <= stage < len(meta_index):
        return []
    a, b = meta_index[stage]
    out = []
    seen = set()
    for p in STAGE_REFS:
        if p + 4 <= b:
            value = pi.u32(m, meta_off + a + p)
            if chunk_records.get(value) is not None and value not in seen:
                seen.add(value)
                out.append((p, value))
    return out


def describe_pso(m, region, meta_off, meta_index, chunk_records, index):
    """Decode one graphics PSO and the chunk records its stage record names."""
    offset = region + index * pi.GRAPHICS_STRIDE
    stage = pi.u32(m, offset + 8)
    layout = pi.u32(m, offset + 0xC)
    refs = stage_record_refs(m, meta_off, meta_index, chunk_records, stage)
    records = [describe_record(record, chunk_records[record]) for _, record in refs]
    fits_two = all(r["fits_two_per_table"] for r in records)
    fits_one = all(r["fits_one_per_table"] for r in records)
    return {
        "pso": index,
        "offset": offset,
        "stage": stage,
        "input_layout": layout,
        "records": [
            {
                "field": "0x%x" % p,
                "index": record,
                "size": meta_index[record][1],
                "decode": decode,
            }
            for (p, record), decode in zip(refs, records)
        ],
        "fits_two_per_table": fits_two,
        "fits_one_per_table": fits_one,
    }


def summarise(m, report, limit=None):
    """Aggregate the sampled graphics PSOs. Returns a JSON-safe dict."""
    counts = report["counts"]
    graphics = counts["graphics"]
    sampled = graphics if limit is None else min(graphics, limit)
    meta_off = report["regions"]["meta_blob"]["offset"]
    meta_region = report["regions"]["meta_index"]
    graphics_region = report["regions"]["graphics"]
    meta_index = [
        struct.unpack_from("<II", m, meta_region["offset"] + i * pi.INDEX_STRIDE)
        for i in range(counts["meta_index"])
    ]
    chunk_records = {
        i: pi.entry_chunks(m, meta_off, a, b) for i, (a, b) in enumerate(meta_index)
    }
    chunk_records = {i: c for i, c in chunk_records.items() if c}

    kind_counts = collections.Counter()
    range_counts = collections.Counter()
    space_counts = collections.Counter()
    constants_hist = collections.Counter()
    extras = 0
    extras_records = 0
    fit_two = fit_one = 0
    used_records = set()
    failing = []
    for index in range(sampled):
        pso = describe_pso(
            m, graphics_region["offset"], meta_off, meta_index, chunk_records, index
        )
        fit_two += pso["fits_two_per_table"]
        fit_one += pso["fits_one_per_table"]
        for rec in pso["records"]:
            used_records.add(rec["index"])
            decode = rec["decode"]
            for name, n in decode["kinds"].items():
                kind_counts[name] += n
            for pair in decode["pairs"]:
                if pair["slot"] is not None:
                    range_counts[pair["slot"]["II"]] += 1
            for table in decode["tables"]:
                space_counts[table] += 1
            constants_hist[decode["constants"]] += 1
            if decode["extras"]:
                extras_records += 1
                extras += len(decode["extras"])
            if not decode["fits_two_per_table"] and len(failing) < 10:
                failing.append(
                    {
                        "pso": index,
                        "record": rec["index"],
                        "constants": decode["constants"],
                        "tables": decode["tables"],
                        "cost_two_per_table": decode["cost_two_per_table"],
                        "cost_one_per_table": decode["cost_one_per_table"],
                    }
                )
    return {
        "graphics": graphics,
        "sampled": sampled,
        "distinct_chunk_records": len(used_records),
        "fit_two_per_table": fit_two,
        "fit_one_per_table": fit_one,
        "kind_counts": dict(sorted(kind_counts.items())),
        "range_counts": {
            RANGE_NAMES.get(k, "0x%x" % k): v for k, v in sorted(range_counts.items())
        },
        "table_index_counts": {str(k): v for k, v in sorted(space_counts.items())},
        "constants_hist": {str(k): v for k, v in sorted(constants_hist.items())},
        "extras": extras,
        "records_with_extras": extras_records,
        "failing": failing,
    }


def print_summary(s, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p("graphics PSOs sampled: %d of %d" % (s["sampled"], s["graphics"]))
    p("distinct chunk records resolved: %d" % s["distinct_chunk_records"])
    p(
        "fit 32 dwords: %d of %d with 2 dwords per table, %d with 1 dword per table"
        % (s["fit_two_per_table"], s["sampled"], s["fit_one_per_table"])
    )
    p(
        "0x41 kinds (per PSO-record): %s"
        % " ".join("%s=%d" % kv for kv in s["kind_counts"].items())
    )
    p(
        "0x42 range types: %s"
        % " ".join("%s=%d" % kv for kv in s["range_counts"].items())
    )
    p(
        "0x42 table indices (SS): %s"
        % " ".join("%s=%d" % kv for kv in s["table_index_counts"].items())
    )
    p(
        "constant dwords per record: %s"
        % " ".join("%s=%d" % kv for kv in s["constants_hist"].items())
    )
    p("extra slots: %d in %d records" % (s["extras"], s["records_with_extras"]))
    if s["failing"]:
        p(
            "records that do not fit 32 dwords with 2 per table (first %d):"
            % len(s["failing"])
        )
        for f in s["failing"]:
            p(
                "  pso %d record %d: constants=%d tables=%s cost=%d (1/table %d)"
                % (
                    f["pso"],
                    f["record"],
                    f["constants"],
                    f["tables"],
                    f["cost_two_per_table"],
                    f["cost_one_per_table"],
                )
            )
    else:
        p("every sampled record fits 32 dwords with 2 per table")


def print_pso(pso, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(
        "graphics PSO %d at file offset 0x%x: stage entry %d, input layout entry %d"
        % (pso["pso"], pso["offset"], pso["stage"], pso["input_layout"])
    )
    for rec in pso["records"]:
        d = rec["decode"]
        p(
            "  chunk record %d (stage field %s, %d chunks): %d constants, tables %s, cost %d/%d dwords"
            % (
                rec["index"],
                rec["field"],
                d["chunks"],
                d["constants"],
                d["tables"],
                d["cost_two_per_table"],
                d["cost_one_per_table"],
            )
        )
        for entry in d["entries"]:
            if entry.get("chunk") == "0x41":
                extra = ""
                if "constant_count" in entry:
                    extra = " constant_count(a)=%d" % entry["constant_count"]
                elif "space" in entry:
                    extra = " space(b)=%d base(a)=%d" % (
                        entry["space"],
                        entry["base_register"],
                    )
                p(
                    "    0x41 kind=%s(%s) offset=0x%x key=0x%x count=0x%x%s size=%d"
                    % (
                        entry["tag"],
                        entry["name"],
                        entry["offset"],
                        entry["key"],
                        entry["count"],
                        extra,
                        entry["size"],
                    )
                )
        for pair in d["pairs"]:
            slot = pair["slot"]
            entry = pair["entry"]
            if slot is None:
                p("    slot: none")
            else:
                p(
                    "    slot table(SS)=%d range(II)=%d(%s) width(RR)=%d key=0x%x w4=0x%x"
                    % (
                        slot["SS"],
                        slot["II"],
                        RANGE_NAMES.get(slot["II"], "?"),
                        slot["RR"],
                        slot["key"],
                        slot["w4"],
                    )
                )
        for slot in d["extras"]:
            p(
                "    extra slot table(SS)=%d range(II)=%d width(RR)=%d key=0x%x w4=0x%x"
                % (slot["SS"], slot["II"], slot["RR"], slot["key"], slot["w4"])
            )
        p(
            "    fits 32 dwords: %s with 2/table, %s with 1/table"
            % (d["fits_two_per_table"], d["fits_one_per_table"])
        )
        if not d["fits_two_per_table"]:
            p(
                "    does not fit: %d constant dwords + %d tables = %d, over by %d"
                % (
                    d["constants"],
                    len(d["tables"]),
                    d["cost_two_per_table"],
                    d["cost_two_per_table"] - CONTRACT_DWORDS,
                )
            )
    p(
        "  PSO fits: %s with 2/table, %s with 1/table"
        % (pso["fits_two_per_table"], pso["fits_one_per_table"])
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Describe graphics PSO user data and its 32-dword root-argument fit."
    )
    ap.add_argument("cache", help="cache file (read-only)")
    ap.add_argument(
        "--entry", type=int, default=None, metavar="N", help="print graphics PSO N"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="sample the first N graphics PSOs (default: all)",
    )
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument(
        "--quiet", action="store_true", help="print nothing, only set the exit code"
    )
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    try:
        report = pi.inspect(a.cache)
    except (OSError, ValueError) as e:
        print("%s: %s" % (a.cache, e), file=sys.stderr)
        return 2
    if report["problems"]:
        print("cache disagrees with the layout model:", file=sys.stderr)
        for problem in report["problems"]:
            print("  %s" % problem, file=sys.stderr)
        return 2
    with open(a.cache, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if a.entry is not None:
                meta_off = report["regions"]["meta_blob"]["offset"]
                meta_region = report["regions"]["meta_index"]
                meta_index = [
                    struct.unpack_from(
                        "<II", m, meta_region["offset"] + i * pi.INDEX_STRIDE
                    )
                    for i in range(report["counts"]["meta_index"])
                ]
                chunk_records = {
                    i: pi.entry_chunks(m, meta_off, off, size)
                    for i, (off, size) in enumerate(meta_index)
                }
                chunk_records = {i: c for i, c in chunk_records.items() if c}
                if not 0 <= a.entry < report["counts"]["graphics"]:
                    print(
                        "--entry %d is outside the %d graphics records"
                        % (a.entry, report["counts"]["graphics"]),
                        file=sys.stderr,
                    )
                    return 2
                pso = describe_pso(
                    m,
                    report["regions"]["graphics"]["offset"],
                    meta_off,
                    meta_index,
                    chunk_records,
                    a.entry,
                )
                result = {"summary": None, "pso": pso}
            else:
                result = {"summary": summarise(m, report, a.limit), "pso": None}
    if not a.quiet:
        if a.json:
            print(json.dumps(result, indent=1))
        else:
            if result["summary"] is not None:
                print_summary(result["summary"])
            if result["pso"] is not None:
                print_pso(result["pso"])
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
