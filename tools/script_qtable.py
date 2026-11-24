"""Decode the 0x71717171-tagged ('qqqq') object-tree container in script.rpf.

`rpf9_carve.py` extracts 1,012 PSIN containers from `script.rpf` plus one
42-stream binary container at offset `0x53000` (2,702,384 bytes inflated). It
begins with a 40-byte header holding `u32 record_count` at `+0x14`, followed by
`record_count` ascending `u64` byte offsets and `record_count` 16-byte-aligned
records. Internal 8-byte pointers carry `0x71717171` (`'qqqq'`) in the high
32 bits and `offsets[target] + 8` in the low 32 bits.

Usage:
    python -I tools/script_qtable.py <carve_dir_or_rpf_or_bin> [--root-hash HEX] [--type-hash HEX] [--json FILE]

Default output and `--json` reports emit aggregate structure counts, size and
stride histograms, invariant checks and optional local query counts only; no
payload bytes or string tables are exported. Standard library only.
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
from report_paths import same_report_path  # noqa: E402

HEADER_BYTES = 40
Q_TAG = 0x71717171
ARRAY_MARKER = 0x14
ARRAY_HEADER_BYTES = 48
ARRAY_FLAG_POINTERS = 0x10000
MAX_RECORDS = 1_000_000


class UsageError(ValueError):
    """CLI argument or output-path error reported with exit code 2."""


def _align16(value: int) -> int:
    return (value + 15) & ~15


def is_qtable_prefix(blob: bytes) -> bool:
    """True when `blob` begins with a valid 40-byte qtable header and offset[0]."""
    if len(blob) < HEADER_BYTES + 8:
        return False
    words = struct.unpack_from("<10I", blob, 0)
    record_count = words[5]
    if not (1 <= record_count <= MAX_RECORDS):
        return False
    if any(words[i] != 0 for i in range(10) if i != 5):
        return False
    first_off = struct.unpack_from("<Q", blob, HEADER_BYTES)[0]
    table_end = HEADER_BYTES + record_count * 8
    return first_off in (table_end, _align16(table_end)) and first_off % 16 == 0


def _parse_hex_u32(text: str) -> int:
    try:
        value = int(text, 16)
    except ValueError as e:
        raise UsageError(f"invalid hex u32: {text}") from e
    if not (0 <= value <= 0xFFFFFFFF):
        raise UsageError(f"hash out of u32 range: {text}")
    return value


def _classify_record(idx: int, rec: bytes, ptrs: list[tuple[int, int, int | None]]) -> dict:
    w2, w3 = struct.unpack_from("<2I", rec, 8)
    if w2 == ARRAY_MARKER and w3 == 0:
        if len(rec) < ARRAY_HEADER_BYTES:
            return {
                "index": idx,
                "kind": "array",
                "bytes": len(rec),
                "count": 0,
                "capacity": 0,
                "stride": 0,
                "array_flags": 0,
                "element_type_hash": 0,
                "header_complete": False,
                "count_eq_capacity": False,
                "size_exact": False,
                "extra_words_zero": False,
                "zero_element_hash": True,
            }
        cnt, cap, stride, aflags, w8, w9, elem_hash, w11 = struct.unpack_from("<8I", rec, 16)
        expected_len = _align16(ARRAY_HEADER_BYTES + cnt * stride)
        return {
            "index": idx,
            "kind": "array",
            "bytes": len(rec),
            "count": cnt,
            "capacity": cap,
            "stride": stride,
            "array_flags": aflags,
            "element_type_hash": elem_hash,
            "header_complete": True,
            "count_eq_capacity": cnt == cap,
            "size_exact": expected_len == len(rec),
            "extra_words_zero": (w8 == 0 and w9 == 0 and w11 == 0),
            "zero_element_hash": elem_hash == 0,
        }

    is_wrapper = (
        len(rec) == 32
        and w3 == 0
        and len(ptrs) == 1
        and ptrs[0][0] == 16
        and ptrs[0][2] == idx + 1
        and struct.unpack_from("<Q", rec, 24)[0] == 0
    )
    return {
        "index": idx,
        "kind": "struct",
        "bytes": len(rec),
        "struct_type_hash": w2,
        "w3_zero": w3 == 0,
        "is_wrapper": is_wrapper,
    }


def _collect_subtree(root_idx: int, adj: list[list[int]], records: list[dict],
                     offsets: list[int], end_off: int, limit_hi: int) -> dict:
    """Walk one tree from `root_idx` to measure node count, depth, bytes and type composition."""
    stack = [(root_idx, 0)]
    seen: set[int] = set()
    arrays = structs = max_depth = total_bytes = cross_edges = 0
    expected_next = root_idx
    dfs_preorder = True
    stride_counts: Counter[int] = Counter()
    struct_sizes: Counter[int] = Counter()

    while stack:
        node, depth = stack.pop()
        if node in seen:
            dfs_preorder = False
            continue
        seen.add(node)
        if node != expected_next:
            dfs_preorder = False
        expected_next += 1
        if depth > max_depth:
            max_depth = depth
        rec_end = offsets[node + 1] if node + 1 < len(offsets) else end_off
        total_bytes += rec_end - offsets[node]
        info = records[node]
        if info["kind"] == "array":
            arrays += 1
            stride_counts[info["stride"]] += 1
        else:
            structs += 1
            struct_sizes[info["bytes"]] += 1
        for child in reversed(adj[node]):
            if root_idx <= child < limit_hi:
                stack.append((child, depth + 1))
            else:
                cross_edges += 1
                dfs_preorder = False

    if expected_next != limit_hi:
        dfs_preorder = False

    return {
        "nodes": len(seen),
        "max_depth": max_depth,
        "bytes": total_bytes,
        "arrays": arrays,
        "structs": structs,
        "cross_tree_edges": cross_edges,
        "dfs_preorder": dfs_preorder,
        "array_strides": {str(k): stride_counts[k] for k in sorted(stride_counts)},
        "struct_sizes": {str(k): struct_sizes[k] for k in sorted(struct_sizes)},
    }


def parse_qtable(blob: bytes, *, root_hash: int | None = None, type_hash: int | None = None) -> dict:
    """Validate and summarize a 0x71717171-tagged object-tree payload."""
    if not is_qtable_prefix(blob):
        raise ValueError("payload does not begin with a valid 40-byte qtable header")
    if len(blob) % 16 != 0:
        raise ValueError(f"payload length {len(blob)} is not 16-byte aligned")

    n = struct.unpack_from("<I", blob, 0x14)[0]
    table_end = HEADER_BYTES + n * 8
    if len(blob) < table_end:
        raise ValueError(
            f"payload length {len(blob)} is shorter than {n}-record offset table ({table_end} bytes)"
        )
    offsets = list(struct.unpack_from(f"<{n}Q", blob, HEADER_BYTES))

    for i, off in enumerate(offsets):
        if off % 16 != 0:
            raise ValueError(f"record {i} offset {off:#x} is not 16-byte aligned")
        if off < table_end or off >= len(blob):
            raise ValueError(f"record {i} offset {off:#x} is out of bounds for {len(blob):#x} bytes")
        if i > 0 and off <= offsets[i - 1]:
            raise ValueError(f"record {i} offset {off:#x} <= previous {offsets[i - 1]:#x}")

    bounds = offsets + [len(blob)]
    if any((bounds[i + 1] - bounds[i]) < 16 for i in range(n)):
        raise ValueError("record shorter than 16 bytes")

    body_to_index = {offsets[i] + 8: i for i in range(n)}
    in_degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    records: list[dict] = []

    root_indices: list[int] = []
    root_hashes_by_idx: dict[int, int] = {}
    root_flags_hist: Counter[int] = Counter()
    nonroot_flags_nonzero = 0

    total_pointers = resolved_pointers = unresolved_pointers = forward_edges = 0
    size_hist: Counter[int] = Counter()
    array_stride_hist: Counter[int] = Counter()
    array_flags_hist: Counter[int] = Counter()
    struct_size_hist: Counter[int] = Counter()
    element_hashes: set[int] = set()
    struct_hashes: set[int] = set()
    wrapper_count = struct_w3_nonzero = 0
    array_header_complete = array_count_eq_capacity = 0
    array_exact = array_extra_zero = array_zero_elem_hash = 0
    array_ptr_stride8 = array_ptr_count = 0

    for i in range(n):
        rec = blob[bounds[i]:bounds[i + 1]]
        size_hist[len(rec)] += 1
        rkey, pflags = struct.unpack_from("<2I", rec, 0)
        if rkey != 0:
            root_indices.append(i)
            root_hashes_by_idx[i] = rkey
            root_flags_hist[pflags] += 1
        elif pflags != 0:
            nonroot_flags_nonzero += 1

        rec_ptrs: list[tuple[int, int, int | None]] = []
        for rel in range(8, len(rec), 8):
            word = struct.unpack_from("<Q", rec, rel)[0]
            if (word >> 32) == Q_TAG:
                total_pointers += 1
                target_off = word & 0xFFFFFFFF
                tgt = body_to_index.get(target_off)
                rec_ptrs.append((rel, target_off, tgt))
                if tgt is None:
                    unresolved_pointers += 1
                else:
                    resolved_pointers += 1
                    in_degree[tgt] += 1
                    adj[i].append(tgt)
                    if tgt > i:
                        forward_edges += 1

        info = _classify_record(i, rec, rec_ptrs)
        info["root_key_hash"] = rkey
        info["prefix_flags"] = pflags
        records.append(info)

        if info["kind"] == "array":
            array_stride_hist[info["stride"]] += 1
            array_flags_hist[info["array_flags"]] += 1
            element_hashes.add(info["element_type_hash"])
            if info["header_complete"]:
                array_header_complete += 1
            if info["count_eq_capacity"]:
                array_count_eq_capacity += 1
            if info["size_exact"]:
                array_exact += 1
            if info["extra_words_zero"]:
                array_extra_zero += 1
            if info["zero_element_hash"]:
                array_zero_elem_hash += 1
            if info["array_flags"] == ARRAY_FLAG_POINTERS:
                array_ptr_count += 1
                if info["stride"] == 8:
                    array_ptr_stride8 += 1
        else:
            struct_size_hist[info["bytes"]] += 1
            struct_hashes.add(info["struct_type_hash"])
            if not info["w3_zero"]:
                struct_w3_nonzero += 1
            if info["is_wrapper"]:
                wrapper_count += 1

    root_set = set(root_indices)
    in_deg_zero = sum(1 for d in in_degree if d == 0)
    nonroot_in_deg_one = all(in_degree[i] == 1 for i in range(n) if i not in root_set)
    root_bijection_holds = (
        all((i in root_set) == (in_degree[i] == 0) for i in range(n))
        and nonroot_in_deg_one
    )

    tree_sizes: list[int] = []
    tree_depths: list[int] = []
    cross_tree_edges = 0
    tree_owner: list[int] = [-1] * n
    all_dfs_preorder = bool(root_indices) and root_indices[0] == 0
    for pos, r_idx in enumerate(root_indices):
        hi = root_indices[pos + 1] if pos + 1 < len(root_indices) else n
        for member in range(r_idx, hi):
            tree_owner[member] = r_idx
        stats = _collect_subtree(r_idx, adj, records, offsets, len(blob), hi)
        tree_sizes.append(stats["nodes"])
        tree_depths.append(stats["max_depth"])
        cross_tree_edges += stats["cross_tree_edges"]
        if not stats["dfs_preorder"]:
            all_dfs_preorder = False

    trees_partition_records = sum(tree_sizes) == n and cross_tree_edges == 0
    contiguous_preorder = (
        trees_partition_records
        and forward_edges == resolved_pointers
        and all_dfs_preorder
    )

    array_count = sum(1 for r in records if r["kind"] == "array")
    struct_count = n - array_count

    valid = (
        unresolved_pointers == 0
        and root_bijection_holds
        and contiguous_preorder
        and array_header_complete == array_count
        and array_count_eq_capacity == array_count
        and array_exact == array_count
    )

    result: dict = {
        "valid": valid,
        "payload_bytes": len(blob),
        "header_bytes": HEADER_BYTES,
        "offset_table_bytes": n * 8,
        "record_pool_offset": offsets[0],
        "record_pool_bytes": len(blob) - offsets[0],
        "record_count": n,
        "root_count": len(root_indices),
        "distinct_root_hashes": len(set(root_hashes_by_idx.values())),
        "in_deg_zero_count": in_deg_zero,
        "nonroot_in_deg_one": nonroot_in_deg_one,
        "root_bijection_holds": root_bijection_holds,
        "total_pointers": total_pointers,
        "resolved_pointers": resolved_pointers,
        "unresolved_pointers": unresolved_pointers,
        "forward_edges": forward_edges,
        "cross_tree_edges": cross_tree_edges,
        "trees_partition_records": trees_partition_records,
        "contiguous_preorder": contiguous_preorder,
        "nonroot_prefix_flags_nonzero": nonroot_flags_nonzero,
        "root_prefix_flags": {f"{k:#x}": root_flags_hist[k] for k in sorted(root_flags_hist)},
        "array_records": array_count,
        "array_header_complete": array_header_complete,
        "array_count_eq_capacity": array_count_eq_capacity,
        "array_size_exact": array_exact,
        "array_extra_words_zero": array_extra_zero,
        "array_zero_element_hashes": array_zero_elem_hash,
        "array_value_records": array_flags_hist.get(0, 0),
        "array_pointer_records": array_ptr_count,
        "array_pointer_stride8": array_ptr_stride8,
        "array_flags": {f"{k:#x}": array_flags_hist[k] for k in sorted(array_flags_hist)},
        "array_strides": {str(k): array_stride_hist[k] for k in sorted(array_stride_hist)},
        "distinct_element_type_hashes": len(element_hashes),
        "struct_records": struct_count,
        "struct_wrappers": wrapper_count,
        "struct_w3_nonzero": struct_w3_nonzero,
        "distinct_struct_type_hashes": len(struct_hashes),
        "struct_sizes": {str(k): struct_size_hist[k] for k in sorted(struct_size_hist)},
        "record_sizes": {str(k): size_hist[k] for k in sorted(size_hist)},
        "tree_nodes_min": min(tree_sizes) if tree_sizes else 0,
        "tree_nodes_median": sorted(tree_sizes)[len(tree_sizes) // 2] if tree_sizes else 0,
        "tree_nodes_max": max(tree_sizes) if tree_sizes else 0,
        "tree_depth_min": min(tree_depths) if tree_depths else 0,
        "tree_depth_max": max(tree_depths) if tree_depths else 0,
    }

    if root_hash is not None:
        matches = []
        for pos, r_idx in enumerate(root_indices):
            if root_hashes_by_idx[r_idx] == root_hash:
                hi = root_indices[pos + 1] if pos + 1 < len(root_indices) else n
                sub = _collect_subtree(r_idx, adj, records, offsets, len(blob), hi)
                sub["root_index"] = r_idx
                sub["root_offset_hex"] = f"{offsets[r_idx]:#x}"
                sub["root_type_hash"] = f"{records[r_idx].get('struct_type_hash', ARRAY_MARKER):#010x}"
                matches.append(sub)
        result["root_query"] = {
            "hash": f"{root_hash:#010x}",
            "matches": len(matches),
            "trees": matches,
        }

    if type_hash is not None:
        matching_arrays = [r for r in records if r["kind"] == "array" and r["element_type_hash"] == type_hash]
        matching_structs = [r for r in records if r["kind"] == "struct" and r["struct_type_hash"] == type_hash]
        owner_roots = {tree_owner[r["index"]] for r in matching_arrays + matching_structs if tree_owner[r["index"]] >= 0}
        result["type_query"] = {
            "hash": f"{type_hash:#010x}",
            "array_matches": len(matching_arrays),
            "struct_matches": len(matching_structs),
            "distinct_trees": len(owner_roots),
            "array_strides": dict(sorted(Counter(str(r["stride"]) for r in matching_arrays).items())),
            "struct_sizes": dict(sorted(Counter(str(r["bytes"]) for r in matching_structs).items())),
        }

    return result


def _is_complete_qtable(blob: bytes) -> bool:
    """True when `blob` already holds a complete, structurally valid qtable payload."""
    if not is_qtable_prefix(blob) or len(blob) % 16 != 0:
        return False
    n = struct.unpack_from("<I", blob, 0x14)[0]
    table_end = HEADER_BYTES + n * 8
    if len(blob) < table_end + 16:
        return False
    last_off = struct.unpack_from("<Q", blob, HEADER_BYTES + (n - 1) * 8)[0]
    if last_off < table_end or last_off + 16 > len(blob):
        return False
    try:
        return bool(parse_qtable(blob)["valid"])
    except ValueError:
        return False


def _join_qtable_from_source(source) -> tuple[bytes, dict]:
    rows = source.rows
    n = len(rows)
    for i in range(n):
        data = source.read(rows[i])
        if not is_qtable_prefix(data):
            continue
        parts = [data]
        stored = rows[i].get("stored", len(data))
        j = i + 1
        prev_len = len(data)
        while j < n and prev_len > 0 and prev_len % 65536 == 0:
            if _is_complete_qtable(b"".join(parts)):
                break
            nxt = source.read(rows[j])
            nk = gi.stream_kind(nxt, rows[j])
            if nk in ("decl", "bink2", "rbf0", "psin", "other") or gi.standalone_xml(nxt):
                break
            parts.append(nxt)
            stored += rows[j].get("stored", len(nxt))
            prev_len = len(nxt)
            j += 1
        blob = b"".join(parts)
        meta = {
            "source_type": "directory" if isinstance(source, (gi.DirectorySource, _BinFolderSource)) else "archive",
            "offset": rows[i]["off"],
            "offset_hex": f"{rows[i]['off']:#x}",
            "chunks": j - i,
            "stored": stored,
            "inflated": len(blob),
        }
        return blob, meta
    raise ValueError("no 0x71717171 object-tree container found")


class _BinFolderSource:
    """Minimal sorted *.bin reader when a test/carve directory has no index.json."""

    def __init__(self, directory: str):
        names = sorted(n for n in os.listdir(directory) if n.lower().endswith(".bin"))
        self.directory = directory
        self.rows = []
        for idx, name in enumerate(names):
            full = os.path.join(directory, name)
            sz = os.path.getsize(full)
            try:
                off = int(os.path.splitext(name)[0], 16)
            except ValueError:
                off = idx * 0x10000
            self.rows.append({"off": off, "stored": sz, "size": sz, "_name": name})

    def read(self, row: dict) -> bytes:
        with open(os.path.join(self.directory, row["_name"]), "rb") as f:
            return f.read()


def load_qtable_payload(target: str) -> tuple[bytes, dict]:
    """Load the object-tree container from a carve directory, .rpf archive or raw .bin."""
    if os.path.isdir(target):
        index_path = os.path.join(target, "index.json")
        source = gi.DirectorySource(target) if os.path.isfile(index_path) else _BinFolderSource(target)
        return _join_qtable_from_source(source)

    if not os.path.isfile(target):
        raise FileNotFoundError(target)

    with open(target, "rb") as f:
        head = f.read(64)

    if head[:4] == b"9FPR":
        source = gi.ArchiveSource(target)
        try:
            return _join_qtable_from_source(source)
        finally:
            source.close()

    with open(target, "rb") as f:
        blob = f.read()
    if not is_qtable_prefix(blob):
        raise ValueError(f"{target}: not an RPF9 archive or 0x71717171 object-tree payload")
    return blob, {
        "source_type": "file",
        "offset": 0,
        "offset_hex": "0x0",
        "chunks": 1,
        "stored": len(blob),
        "inflated": len(blob),
    }


def check_report_path(out_path: str, target: str, rows: list[dict] | None = None) -> None:
    """Refuse a --json destination that is a directory or collides with any input."""
    if os.path.isdir(out_path):
        raise UsageError(f"--json is a directory: {out_path}")
    if os.path.isdir(target):
        if os.path.isfile(os.path.join(target, "index.json")):
            try:
                gm.check_report_path(out_path, target, rows=rows)
            except ValueError as e:
                raise UsageError(str(e)) from e
        elif same_report_path(out_path, target):
            raise UsageError(f"--json would overwrite the input {target}")
        else:
            for name in os.listdir(target):
                if name.lower().endswith(".bin") and same_report_path(out_path, os.path.join(target, name)):
                    raise UsageError(f"--json would overwrite the input {name}")
    elif same_report_path(out_path, target):
        raise UsageError(f"--json would overwrite the input {target}")


def _print_summary(report: dict) -> None:
    loc = report["location"]
    print(
        f'qtable {loc["offset_hex"]}: {loc["chunks"]} streams, '
        f'{loc["stored"]:,} stored, {report["payload_bytes"]:,} inflated '
        f'(valid={report["valid"]})'
    )
    print(
        f'  records: {report["record_count"]:,} '
        f'(arrays={report["array_records"]:,} exact={report["array_size_exact"]:,}, '
        f'structs={report["struct_records"]:,} wrappers={report["struct_wrappers"]:,})'
    )
    print(
        f'  trees: {report["root_count"]:,} roots ({report["distinct_root_hashes"]:,} distinct hashes), '
        f'bijection={report["root_bijection_holds"]}, '
        f'preorder={report["contiguous_preorder"]}, '
        f'nodes min/med/max={report["tree_nodes_min"]}/{report["tree_nodes_median"]}/{report["tree_nodes_max"]}, '
        f'depth {report["tree_depth_min"]}..{report["tree_depth_max"]}'
    )
    print(
        f'  pointers: {report["resolved_pointers"]:,}/{report["total_pointers"]:,} resolved '
        f'(forward={report["forward_edges"]:,}, unresolved={report["unresolved_pointers"]})'
    )
    print(f'  array strides: {report["array_strides"]} ({report["distinct_element_type_hashes"]} element hashes)')
    print(f'  struct sizes:  {report["struct_sizes"]} ({report["distinct_struct_type_hashes"]} struct hashes)')
    if "root_query" in report:
        rq = report["root_query"]
        print(f'  --root-hash {rq["hash"]}: {rq["matches"]} match(es) {rq["trees"]}')
    if "type_query" in report:
        tq = report["type_query"]
        print(
            f'  --type-hash {tq["hash"]}: arrays={tq["array_matches"]}, '
            f'structs={tq["struct_matches"]}, distinct_trees={tq["distinct_trees"]}'
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="carved script directory, script.rpf archive, or joined .bin file")
    ap.add_argument("--root-hash", help="inspect subtrees for this u32 root hash (hex)")
    ap.add_argument("--type-hash", help="summarize array/struct records matching this u32 type hash (hex)")
    ap.add_argument("--json", help="write aggregate statistics JSON to this file")
    a = ap.parse_args(argv)

    try:
        root_h = _parse_hex_u32(a.root_hash) if a.root_hash is not None else None
        type_h = _parse_hex_u32(a.type_hash) if a.type_hash is not None else None
        if a.json:
            rows = gm._read_index_rows(a.target) if os.path.isdir(a.target) and os.path.isfile(os.path.join(a.target, "index.json")) else None
            check_report_path(a.json, a.target, rows=rows)
        blob, meta = load_qtable_payload(a.target)
        report = parse_qtable(blob, root_hash=root_h, type_hash=type_h)
    except UsageError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"ERROR: {a.target}: does not exist", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"ERROR: cannot read {a.target}: {e.strerror or e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, struct.error, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = {"location": meta, **report}
    _print_summary(out)

    if a.json:
        try:
            gm._make_parent(a.json)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
        print("wrote", a.json)

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
