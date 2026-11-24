"""Decode the non-XML text and binary payloads of common.rpf and join to shaders.rpf.

Targets the 3 plain-text payloads (0x640000 VFX_SETTINGS_FILE, 0x64e000 and
0x64f1c0 .fx shader preload lists) and the 2 fixed/grid binary tables
(0xad000 801x1296 B table and 0x23c000 3-region spatial grid / record table):

    python -I tools/gamedata_blobs.py <carve_dir_or_rpf> [--shaders PATH/TO/shaders.rpf] [--json FILE]

Only payload types, offsets, byte sizes, structural counts and SGD5 match counts
are printed or written; shader/setting names and raw bytes stay local. Standard
library only.
Usage: `python -I tools/gamedata_blobs.py --help` lists the options.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import posixpath
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing sibling helpers.
TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import gamedata_index as gi  # noqa: E402
import gamedata_maps as gm  # noqa: E402
from report_paths import same_report_path  # noqa: E402
import sgd5_effects as se  # noqa: E402

COMMENT_PREFIXES = ("#", "//", ";")
CELL_SENTINEL = 0xC0000000
CELL_OFFSET_MASK = 0x7FFFFF


def _is_fx_preload_lines(lines: list[str]) -> bool:
    data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith(COMMENT_PREFIXES)]
    if not data_lines:
        return False
    return all(len(l.split()) == 1 and l.lower().endswith(".fx") for l in data_lines)


def parse_fx_preload_list(text: str) -> dict:
    """Parse a plain-text .fx shader preload list (one filename or path per line)."""
    raw_lines = text.splitlines()
    entries: list[str] = []
    comment_lines = blank_lines = root_entries = subfolder_entries = 0
    for line in raw_lines:
        s = line.strip()
        if not s:
            blank_lines += 1
        elif s.startswith(COMMENT_PREFIXES):
            comment_lines += 1
        else:
            toks = s.split()
            if len(toks) != 1 or not toks[0].lower().endswith(".fx"):
                raise ValueError(f"invalid .fx preload entry: {len(toks)} tokens")
            norm = toks[0].replace("\\", "/")
            if "/" in norm:
                subfolder_entries += 1
            else:
                root_entries += 1
            entries.append(os.path.splitext(posixpath.basename(norm))[0].lower())
    return {
        "kind": "fx_preload_list",
        "lines": len(raw_lines),
        "entries": len(entries),
        "distinct_entries": len(set(entries)),
        "root_entries": root_entries,
        "subfolder_entries": subfolder_entries,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "_stems": entries,
    }


def parse_vfx_settings_dat(text: str) -> dict:
    """Parse the plain-text VFX_SETTINGS_FILE (<dotted.key> <values...>) table."""
    raw_lines = text.splitlines()
    comment_lines = blank_lines = data_lines = scalar_pairs = equation_lines = tuple_lines = 0
    token_hist: Counter[int] = Counter()
    keys: set[str] = set()
    prefixes: list[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            blank_lines += 1
            continue
        if s.startswith(COMMENT_PREFIXES):
            comment_lines += 1
            continue
        data_lines += 1
        toks = s.split()
        token_hist[len(toks)] += 1
        key = toks[0].lower()
        keys.add(key)
        prefixes.append(key.split(".", 1)[0])
        if "=" in s:
            equation_lines += 1
        elif len(toks) == 2:
            scalar_pairs += 1
        else:
            tuple_lines += 1
    return {
        "kind": "vfx_settings_dat",
        "lines": len(raw_lines),
        "data_lines": data_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "scalar_pairs": scalar_pairs,
        "equation_lines": equation_lines,
        "tuple_lines": tuple_lines,
        "distinct_keys": len(keys),
        "distinct_prefixes": len(set(prefixes)),
        "token_counts": {str(k): token_hist[k] for k in sorted(token_hist)},
        "_prefixes": sorted(set(prefixes)),
        "_prefix_counts": dict(Counter(prefixes)),
    }


def parse_fixed_table_1296(blob: bytes) -> dict:
    """Decode the u32 record_count + N x 1,296-byte fixed record table (0xad000)."""
    if len(blob) < 4:
        raise ValueError("fixed table payload is shorter than 4-byte count")
    count = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + count * 1296
    if count == 0 or len(blob) != expected:
        raise ValueError(f"fixed table size {len(blob)} != 4 + {count} * 1296 ({expected})")
    duplicated_pairs = 0
    distinct_pairs: set[tuple[int, int]] = set()
    for i in range(count):
        base = 4 + i * 1296
        w0, w1, w2, w3 = struct.unpack_from("<4I", blob, base)
        if (w0, w1) == (w2, w3):
            duplicated_pairs += 1
        distinct_pairs.add((w0, w1))
    return {
        "kind": "fixed_table_1296",
        "record_count": count,
        "record_bytes": 1296,
        "record_header_bytes": 16,
        "record_body_bytes": 1280,
        "duplicated_header_pairs": duplicated_pairs,
        "distinct_header_pairs": len(distinct_pairs),
    }


def parse_spatial_grid_blob(blob: bytes) -> dict:
    """Decode the 3-region spatial grid / record / pointer-list blob (0x23c000)."""
    if len(blob) < 16 or len(blob) % 4 != 0:
        raise ValueError("spatial grid blob length must be at least 16 and 4-byte aligned")
    words = struct.unpack_from(f"<{len(blob) // 4}I", blob, 0)
    total_words = len(words)

    # Region 3 at the tail consists strictly of 4-byte-aligned u32 offsets into Region 2.
    k = total_words
    while k > 0 and 4 <= words[k - 1] < len(blob) and words[k - 1] % 4 == 0:
        k -= 1
    if k == total_words or k == 0:
        raise ValueError("missing trailing Region 3 offset table")
    region3_start = k * 4
    region3 = words[k:]

    # Region 2 starts at the lowest target offset referenced by Region 3.
    region2_start = min(region3)
    if region2_start == 0 or region2_start >= region3_start or region2_start % 4 != 0:
        raise ValueError(f"invalid Region 2 start {region2_start:#x} vs Region 3 {region3_start:#x}")
    if any(w < region2_start or w >= region3_start or w % 4 != 0 for w in region3):
        raise ValueError("Region 3 offset leaves Region 2 bounds")

    grid = words[:region2_start // 4]
    empty_cells = sum(1 for w in grid if w == CELL_SENTINEL)
    populated = [w for w in grid if w != CELL_SENTINEL]
    cell_offs = [w & CELL_OFFSET_MASK for w in populated]
    if not populated or any(o < region3_start or o >= len(blob) or o % 4 != 0 for o in cell_offs):
        raise ValueError("Region 1 cell offset leaves Region 3 bounds")

    r2_targets = sorted(set(region3))
    if r2_targets[0] != region2_start:
        raise ValueError("Region 2 does not start at first referenced record")
    r2_bounds = r2_targets + [region3_start]
    r2_word_hist = Counter((r2_bounds[i + 1] - r2_bounds[i]) // 4 for i in range(len(r2_targets)))

    r3_targets = sorted(set(cell_offs))
    if r3_targets[0] != region3_start:
        raise ValueError(f"Region 3 start {region3_start:#x} does not match lowest cell offset {r3_targets[0]:#x}")
    r3_bounds = r3_targets + [len(blob)]
    r3_span_lens = [(r3_bounds[i + 1] - r3_bounds[i]) // 4 for i in range(len(r3_targets))]

    return {
        "kind": "spatial_grid_table",
        "grid_cells": len(grid),
        "empty_cells": empty_cells,
        "populated_cells": len(populated),
        "distinct_cell_offsets": len(r3_targets),
        "cells_in_region3": len(populated),
        "region2_offset": region2_start,
        "region2_bytes": region3_start - region2_start,
        "region2_records": len(r2_targets),
        "region2_record_words_hist": {str(sz): r2_word_hist[sz] for sz in sorted(r2_word_hist)},
        "region3_offset": region3_start,
        "region3_bytes": len(blob) - region3_start,
        "region3_end_inclusive": len(blob) - 1,
        "region3_words": len(region3),
        "region3_words_in_region2": len(region3),
        "region3_spans": len(r3_targets),
        "region3_span_words_min": min(r3_span_lens),
        "region3_span_words_max": max(r3_span_lens),
    }


def _read_group_bytes(source, rows: list[dict], start: int, chunks: int) -> bytes:
    return b"".join(source.read(rows[start + idx]) for idx in range(chunks))


def _analyze_from_source(source) -> list[dict]:
    grouped = gi.build_rows(source.rows, source)
    off_to_idx = {r["off"]: idx for idx, r in enumerate(source.rows)}
    out = []
    for row in grouped:
        if row["type"] not in ("text", "binary"):
            continue
        blob = _read_group_bytes(source, source.rows, off_to_idx[row["offset"]], row["chunks"])
        item = {
            "offset": row["offset"],
            "offset_hex": row["offset_hex"],
            "stored": row["stored"],
            "size": row["size"],
            "chunks": row["chunks"],
            "archive_type": row["type"],
        }
        if row["type"] == "text":
            text = blob.decode("utf-8", "replace")
            if _is_fx_preload_lines(text.splitlines()):
                item.update(parse_fx_preload_list(text))
            else:
                item.update(parse_vfx_settings_dat(text))
        else:
            count0 = struct.unpack_from("<I", blob, 0)[0] if len(blob) >= 4 else 0
            if count0 > 0 and len(blob) == 4 + count0 * 1296:
                item.update(parse_fixed_table_1296(blob))
            else:
                item.update(parse_spatial_grid_blob(blob))
        out.append(item)
    return out


def check_carve_index(directory: str) -> list[dict]:
    """Name the index and the bad row of a folder that is not an rpf9_carve.py carve, before the scan.

    DirectorySource reads `index['files']` and each row's `off` and `stored` straight through, so a hand-edited
    or half-written index fails as `error: 'files'` and names nothing.
    """
    index_path = os.path.join(directory, "index.json")
    if not os.path.isfile(index_path):
        raise ValueError(f"{directory} is not an rpf9_carve.py folder (no index.json)")
    try:
        with open(index_path, encoding="utf-8-sig") as f:
            index = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read {index_path}: {e.strerror or e}") from e
    except ValueError as e:
        raise ValueError(f"{index_path} is not valid JSON: {e}") from e
    rows = index.get("files") if isinstance(index, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{index_path} has no 'files' list")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "off" not in row or "stored" not in row:
            raise ValueError(f"{index_path}: row {i} needs an 'off' and a 'stored'")
    return sorted(rows, key=lambda r: r["off"])


def read_payloads(target: str) -> list[dict]:
    """Decode all text and binary payloads from an rpf9_carve.py dir or .rpf archive."""
    if os.path.isdir(target):
        check_carve_index(target)
        source = gi.DirectorySource(target)
        return _analyze_from_source(source)
    if not os.path.isfile(target):
        raise ValueError(f"{target}: no such file or folder")
    source = gi.ArchiveSource(target)
    try:
        return _analyze_from_source(source)
    finally:
        source.close()


def _is_subsequence(subseq: list[str], target_seq: list[str]) -> bool:
    it = iter(target_seq)
    return all(item in it for item in subseq)


def cross_reference_sgd5(payloads: list[dict], shaders_rpf: str) -> dict:
    """Compare .fx preload lists and vfx prefixes against shaders.rpf SGD5 DB0/DB1."""
    with open(shaders_rpf, "rb") as stream:
        db0_list = [m["effect"].lower() for m in se.read_database(stream, se.DATABASES[0], list_effects=True)["matches"]]
        db1_list = [m["effect"].lower() for m in se.read_database(stream, se.DATABASES[1], list_effects=True)["matches"]]

    db0_set = set(db0_list)
    db1_set = set(db1_list)
    preload_checks = []
    vfx_prefix_matches = []

    for p in payloads:
        if p["kind"] == "fx_preload_list":
            stems = p["_stems"]
            in_db0 = [s for s in stems if s in db0_set]
            in_db1 = [s for s in stems if s in db1_set]
            preload_checks.append({
                "offset_hex": p["offset_hex"],
                "entries": p["entries"],
                "in_db0": len(in_db0),
                "in_db1": len(in_db1),
                "unshipped": p["entries"] - len(in_db0) - len(in_db1),
                "exact_order_db0": in_db0 == db0_list and _is_subsequence(db0_list, stems),
                "exact_order_db1": in_db1 == db1_list and _is_subsequence(db1_list, stems),
            })
        elif p["kind"] == "vfx_settings_dat":
            prefixes = set(p["_prefixes"])
            pfx_counts = p["_prefix_counts"]
            vfx_prefix_matches.append({
                "offset_hex": p["offset_hex"],
                "data_lines": p["data_lines"],
                "distinct_prefixes": p["distinct_prefixes"],
                "prefixes_in_db0": len(prefixes & db0_set),
                "prefixes_in_db1": len(prefixes & db1_set),
                "data_lines_in_db0": sum(c for k, c in pfx_counts.items() if k in db0_set),
                "data_lines_in_db1": sum(c for k, c in pfx_counts.items() if k in db1_set),
            })

    return {
        "sgd5_db0_effects": len(db0_list),
        "sgd5_db1_effects": len(db1_list),
        "preload_lists": preload_checks,
        "vfx_settings": vfx_prefix_matches,
    }


def check_report_path(out_path: str, target: str, shaders_rpf: str | None = None,
                      rows: list[dict] | None = None) -> None:
    """Refuse a --json destination that is a directory or aliases any input file."""
    if os.path.isdir(out_path):
        raise ValueError(f"--json is a directory: {out_path}")
    if shaders_rpf and same_report_path(out_path, shaders_rpf):
        raise ValueError(f"--json would overwrite the input {shaders_rpf}")
    if os.path.isdir(target):
        gm.check_report_path(out_path, target, rows=rows)
    elif same_report_path(out_path, target):
        raise ValueError(f"--json would overwrite the input {target}")


def _public(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="rpf9_carve.py directory or common.rpf archive")
    ap.add_argument("--shaders", help="shaders.rpf archive to verify preload lists against SGD5 DB0/DB1")
    ap.add_argument("--json", help="write aggregate decoded counts to this JSON")
    a = ap.parse_args(argv)

    if not os.path.exists(a.target):
        print(f"ERROR: cannot read {a.target}: no such file or folder", file=sys.stderr)
        return 2
    try:
        if a.json:
            check_report_path(a.json, a.target, a.shaders)
        if os.path.isdir(a.target):
            rows = check_carve_index(a.target)
            if a.json:
                check_report_path(a.json, a.target, a.shaders, rows=rows)
        payloads = read_payloads(a.target)
        xref = cross_reference_sgd5(payloads, a.shaders) if a.shaders else None
    except (OSError, ValueError, KeyError, TypeError, struct.error, RuntimeError) as e:
        print("error:", e, file=sys.stderr)
        return 1

    for p in payloads:
        k = p["kind"]
        if k == "fx_preload_list":
            print(f'text {p["offset_hex"]}: fx_preload_list, {p["entries"]} entries ({p["root_entries"]} root)')
        elif k == "vfx_settings_dat":
            print(f'text {p["offset_hex"]}: vfx_settings_dat, {p["lines"]} lines '
                  f'({p["data_lines"]} data, {p["scalar_pairs"]} pairs, {p["distinct_prefixes"]} prefixes)')
        elif k == "fixed_table_1296":
            print(f'binary {p["offset_hex"]}: fixed_table_1296, {p["record_count"]} x {p["record_bytes"]} B '
                  f'({p["duplicated_header_pairs"]}/{p["record_count"]} paired headers)')
        else:
            print(f'binary {p["offset_hex"]}: spatial_grid_table, {p["grid_cells"]} cells '
                  f'({p["empty_cells"]} empty), {p["region2_records"]} records, {p["region3_words"]} ptrs')
    if not payloads:
        print("no text or binary payloads")
    if xref is not None:
        for row in xref["preload_lists"]:
            print(f'SGD5 xref {row["offset_hex"]}: in_db0={row["in_db0"]} (order={row["exact_order_db0"]}), '
                  f'in_db1={row["in_db1"]} (order={row["exact_order_db1"]}), unshipped={row["unshipped"]}')
        for row in xref["vfx_settings"]:
            print(
                f'VFX xref {row["offset_hex"]}: {row["prefixes_in_db0"]}/{row["distinct_prefixes"]} prefixes '
                f'({row["data_lines_in_db0"]}/{row["data_lines"]} data lines) in DB0'
            )

    if a.json:
        out: dict = {"payloads": [_public(p) for p in payloads]}
        if xref is not None:
            out["sgd5_xref"] = xref
        try:
            gm._make_parent(a.json)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1)
        except OSError as e:
            print(f"error: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 1
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
