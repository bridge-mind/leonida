"""Read the rig dictionaries in anim.rpf and join the RBF0 rig-set manager's entries to them (ASSET-20).

    python -I tools/anim_rigs.py <package>\\anim.rpf <work>\\anim-rigs.json [--rigs <work>\\rigs.json]

A rig dictionary has the clip dictionary's root layout (docs/animation.md, "Clip dictionaries are RAGE
resources") with `RIG` + a version byte at +0x70 instead of the pack array. Its map is keyed by `joaat32` of a rig
path. Each value points at +0x20 to a `RIG` block, which holds tables of 8-byte `(hash, word)` items. `--rigs`
takes the JSON `anim_index.py --rigs` writes and resolves every rig-set entry (source = rig dictionary name,
rig = key). The output holds counts and one row per rig dictionary, no item values.
Usage: `python -I tools/anim_rigs.py --help` lists the options.
"""
import argparse
import collections
import json
import mmap
import os
import struct
import sys
import zlib

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anim_clips  # noqa: E402
import anim_index  # noqa: E402

SYSTEM = anim_clips.SYSTEM
MAGIC = b"RIG"


def is_rig_dictionary(data):
    return len(data) >= 0x78 and data[0x70:0x73] == MAGIC


def block_tables(s, at):
    """Tables of a `RIG` block at page offset `at`: `RIG` + u8 version, u32 length of what follows, then tables
    `{u32 kind hash; u32 size = 0xC + 8 * count; u32 count; u32 word}` + count × `{u32 hash; u32 word}`, with other
    bytes between them. Returns (version, length, [(kind, count)])."""
    if s.d[at:at + 3] != MAGIC:
        raise anim_clips.DecodeError(f"no RIG block at 0x{at:x}")
    if at + 8 > len(s.d):
        raise anim_clips.DecodeError(f"truncated RIG block at 0x{at:x}")
    version, length = s.d[at + 3], s.u32(at + 4)
    end = min(len(s.d), at + 8 + length)
    o, tables = at + 8, []
    while o + 16 <= end:
        kind, size, count, _word = struct.unpack_from("<4I", s.d, o)
        if 0 < count < 4096 and size == 0xC + 8 * count and o + 16 + 8 * count <= end:
            tables.append((kind, count))
            o += 16 + 8 * count
        else:
            o += 4
    return version, length, tables


def parse_rig_dictionary(s):
    """Root: +0x38 name, +0x48 map (key = joaat32(rig path), high half 0), +0x68 joaat32(name), +0x70 `RIG`."""
    d = {"name": s.string(0x38), "hash": s.u32(0x68), "version": s.d[0x73], "entries": []}
    for key, value in s.atmap(0x48):
        if value is None:
            continue
        block = s.ptr(value + 0x20)
        e = {"key": key & 0xFFFFFFFF, "key_high": key >> 32, "block": block}
        if block is not None:
            e["block_version"], e["block_length"], e["tables"] = block_tables(s, block)
        d["entries"].append(e)
    return d


def _read(path, segments):
    with open(path, "rb") as fh:
        data = b""
        for off, stored, raw in segments:
            fh.seek(off)
            data += fh.read(stored) if raw else zlib.decompressobj().decompress(fh.read(stored))
    return data


def scan(path, start=None, limit=None):
    rows, walk = anim_index.scan_rpf(path, start, limit)
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        groups, _unresolved, _gaps = anim_clips.group(m, rows)
    stats, dicts = collections.Counter(), []
    for _i, parts in groups:
        segments = [anim_clips.segment(rows, k) for k in parts]
        with open(path, "rb") as fh:
            fh.seek(segments[0][0])
            head = fh.read(segments[0][1])
        first = head if segments[0][2] else zlib.decompressobj().decompress(head, 0x80)
        if not is_rig_dictionary(first):
            continue
        s = anim_clips.System(_read(path, segments))
        try:
            d = parse_rig_dictionary(s)
        except (anim_clips.DecodeError, struct.error) as e:
            stats["rig_dictionaries_unparsed"] += 1
            dicts.append({"offset": segments[0][0], "error": str(e)[:120]})
            continue
        d["offset"], d["streams"], d["system_bytes"] = segments[0][0], len(segments), len(s.d)
        dicts.append(d)
        stats["rig_dictionaries"] += 1
        stats["rig_dictionary_hash_matches_name"] += d["name"] is not None and anim_index.joaat32(d["name"]) == d["hash"]
        stats[f"rig_dictionary_version:{d['version']}"] += 1
        for e in d["entries"]:
            stats["entries"] += 1
            stats["entries_key_high_zero"] += e["key_high"] == 0
            if e["block"] is None:
                stats["entries_without_block"] += 1
                continue
            stats["entries_with_block"] += 1
            stats[f"block_version:{e['block_version']}"] += 1
            stats["tables"] += len(e["tables"])
            for kind, count in e["tables"]:
                stats[f"table_kind:0x{kind:08x}"] += 1
                stats[f"table_items:0x{kind:08x}"] += count
    return dicts, {"streams": walk["files"], "dictionary_roots": len(groups), "counts": dict(sorted(stats.items()))}


def join_rbf0(dicts, rigs):
    """Resolve the rig-set manager's entries: source = rig dictionary name, joaat32(rig) = a key there."""
    keys: collections.defaultdict[str, set[int]] = collections.defaultdict(set)
    for dictionary in dicts:
        if name := dictionary.get("name"):
            keys[name].update(entry["key"] for entry in dictionary.get("entries", ()))
    entries = [e for rs in rigs["rbf0"]["rigsets"] for e in rs["entries"]]
    c = collections.Counter()
    for e in entries:
        h = anim_index.joaat32(e["rig"].lower())
        if h in keys.get(e["source"], ()):
            c["resolved"] += 1
        elif any(h in k for k in keys.values()):
            c["in_another_dictionary"] += 1
        elif e["source"] not in keys:
            c["no_dictionary_named_source"] += 1
        else:
            c["key_missing"] += 1
    distinct = {(e["source"], e["rig"].lower()) for e in entries}
    used = {(src, anim_index.joaat32(r)) for src, r in distinct}
    c["distinct_rigs"] = len(distinct)
    c["keys"] = sum(len(k) for k in keys.values())
    c["keys_named_by_an_entry"] = sum(1 for name, k in keys.items() for h in k if (name, h) in used)
    return dict(sorted(c.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="anim.rpf")
    ap.add_argument("out", help="output .json (counts and one row per rig dictionary)")
    ap.add_argument("--rigs", help="the JSON anim_index.py --rigs wrote, to resolve the RBF0 entries")
    ap.add_argument("--start", type=anim_index.offset, help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=anim_index.byte_count, help="number of bytes to scan")
    a = ap.parse_args()
    if os.path.isdir(a.out):
        ap.error(f"{a.out} is a folder, not a file")
    if same_report_path(a.out, a.archive):
        ap.error(f"{a.out} would overwrite the input archive")
    if a.rigs and same_report_path(a.out, a.rigs):
        ap.error(f"{a.out} would overwrite the --rigs input")
    try:
        dicts, summary = scan(a.archive, a.start, a.limit)
    except ValueError as e:
        sys.exit(f"{a.archive}: {e}")
    except OSError as e:
        sys.exit(f"{e.filename or a.archive}: {e.strerror or e}")
    if a.rigs:
        try:
            with open(a.rigs, encoding="utf-8-sig") as f:
                summary["rbf0_join"] = join_rbf0(dicts, json.load(f))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"ERROR: {a.rigs}: invalid rig report ({getattr(e, 'strerror', None) or e})", file=sys.stderr)
            return 2
    rows = [{k: v for k, v in d.items() if k != "entries"} | {"entries": len(d.get("entries", ()))} for d in dicts]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "rig_dictionaries": rows}, f, indent=1)
    except OSError as e:
        print(f"ERROR: cannot write {a.out}: {e.strerror or e}", file=sys.stderr)
        return 2
    c = summary["counts"]
    print(json.dumps({k: c.get(k, 0) for k in ("rig_dictionaries", "entries", "entries_with_block", "tables")}))
    if "rbf0_join" in summary:
        print(json.dumps(summary["rbf0_join"]))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
