"""Decode the MoVE (version 19) state-machine and motion-tree containers in anim.rpf (ASSET-86).

    python -I tools/anim_move.py <package>\\anim.rpf <work>\\anim-move.json

Each MoVE resource has a fixed 32-byte header (`MoVE`, u32 version = 19, three zero u32s, u32 network hash at
+0x14, u32 body_size at +0x18, one zero u32 at +0x1C), followed at +0x20 by a variable preamble:
  - u32 N_req + N_req × u32 request hashes
  - u32 N_flag + N_flag × u32 flag hashes
  - u32 preamble_mode (0, 1 or 2)
  - u32 N_desc + N_desc × 5-byte `{u32 hash; u8 kind}` descriptors
  - 10-byte tail `{u16 sub_flag; u32 ref_hash_0; u32 ref_hash_1}` (zero when preamble_mode is 0 or 1)

`body_size` at +0x18 excludes the 46 fixed/count/tail preamble bytes outside it and the `5 * N_desc` descriptor
table, so every complete MoVE v19 resource satisfies:

    total_bytes == 46 + 5 * N_desc + body_size

One resource in anim.rpf (at 0x105a7d000, 71,099 bytes) exceeds RPF9's 64 KiB stream ceiling and spans two
consecutive 16-byte-aligned zlib streams (65,536 + 5,563 bytes). After the 10-byte preamble tail sits the root
state container (`Type 1` = StateMachine, `Type 3` = InlinedStateMachine, `Type 22` = MotionTree), whose child
states, transitions and length-prefixed parameter blocks recurse through signed 32-bit self-relative offsets.
Usage: `python -I tools/anim_move.py --help` lists the options.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anim_index  # noqa: E402
from report_paths import same_report_path  # noqa: E402

MAGIC = b"MoVE"
VERSION = 19
TYPE_STATE_MACHINE = 1
TYPE_INLINED_STATE_MACHINE = 3
TYPE_MOTION_TREE = 22
CONTAINER_TYPES = {
    TYPE_STATE_MACHINE: "state_machine",
    TYPE_INLINED_STATE_MACHINE: "inlined_state_machine",
    TYPE_MOTION_TREE: "motion_tree",
}
KNOWN_SCHEMA_HASHES = {
    anim_index.joaat32("statemachine"): "statemachine",
    anim_index.joaat32("motiontree"): "motiontree",
}
MAX_STANDALONE_FILE_BYTES = 128 * 1024 * 1024


class DecodeError(ValueError):
    pass


def is_move_container(data):
    return len(data) >= 32 and data[:4] == MAGIC


def expected_total_size(data):
    """Return `46 + 5 * N_desc + body_size` once the preamble up to `N_desc` is available, or None if truncated."""
    if not is_move_container(data) or len(data) < 48:
        return None
    body_size = struct.unpack_from("<I", data, 0x18)[0]
    pos = 0x20
    n_req = struct.unpack_from("<I", data, pos)[0]
    pos += 4 + 4 * n_req
    if pos + 4 > len(data):
        return None
    n_flag = struct.unpack_from("<I", data, pos)[0]
    pos += 4 + 4 * n_flag
    if pos + 8 > len(data):
        return None
    n_desc = struct.unpack_from("<I", data, pos + 4)[0]
    return 46 + 5 * n_desc + body_size


def _parse_transitions(data, start, end, count):
    """Parse `count` outgoing state transitions in `data[start:end]`. Each has `u32 size` at +0x10 (`size >= 0x4C`)."""
    if start < 0 or end < start or end > len(data):
        raise DecodeError(f"transition span 0x{start:x}..0x{end:x} outside resource")
    out, p = [], start
    for _ in range(count):
        if p + 0x4C > end:
            raise DecodeError(f"truncated transition header at 0x{p:x}")
        tr_size = struct.unpack_from("<I", data, p + 0x10)[0]
        if tr_size < 0x4C or p + tr_size > end:
            raise DecodeError(f"invalid transition size 0x{tr_size:x} at 0x{p:x}")
        target_rel = struct.unpack_from("<i", data, p + 0x48)[0]
        cond_kind = struct.unpack_from("<I", data, p + 0x4C)[0] if tr_size >= 0x50 else 0
        out.append({
            "offset": p,
            "size": tr_size,
            "target": p + 0x48 + target_rel,
            "condition_kind": cond_kind,
            "condition_bytes": max(0, tr_size - 0x50),
        })
        p += tr_size
    if p != end:
        raise DecodeError(f"transition block at 0x{start:x} ended at 0x{p:x}, expected 0x{end:x}")
    return out


def _parse_default_transitions(data, start, end, count):
    """Parse `count` default/any-state transitions in `data[start:end]`: `{u16 size; u16 cond_count; i32 target_rel}`."""
    if start < 0 or end < start or end > len(data):
        raise DecodeError(f"default-transition span 0x{start:x}..0x{end:x} outside resource")
    out, p = [], start
    for _ in range(count):
        if p + 16 > end:
            raise DecodeError(f"truncated default transition at 0x{p:x}")
        size, cond_count, target_rel = struct.unpack_from("<HHi", data, p)
        if size < 16 or p + size > end:
            raise DecodeError(f"invalid default-transition size {size} at 0x{p:x}")
        out.append({
            "offset": p,
            "size": size,
            "conditions": cond_count,
            "target": p + 4 + target_rel,
        })
        p += size
    if p != end:
        raise DecodeError(f"default-transition block at 0x{start:x} ended at 0x{p:x}, expected 0x{end:x}")
    return out


def _parse_param_blocks(data, start, end_limit, count):
    """Parse `count` length-prefixed MotionTree parameter blocks starting at `start`.

    Every block satisfies:
        block_size == 12 + 16 * (b1 + b2 + b3 + b4) + 8 * b5 + sum(sub_size for _ in range(b6))
    with `b7 == 0`. Returns `(blocks, end_offset)`.
    """
    if start < 0 or start > end_limit:
        raise DecodeError(f"parameter block start 0x{start:x} outside container")
    blocks, p = [], start
    for _ in range(count):
        if p + 12 > end_limit:
            raise DecodeError(f"truncated parameter block header at 0x{p:x}")
        bsz = struct.unpack_from("<I", data, p)[0]
        if bsz < 12 or p + bsz > end_limit:
            raise DecodeError(f"invalid parameter block size {bsz} at 0x{p:x}")
        b0, b1, b2, b3, b4, b5, b6, b7 = data[p + 4:p + 12]
        if b7 != 0:
            raise DecodeError(f"non-zero parameter block reserved byte b7={b7} at 0x{p:x}")
        n16 = b1 + b2 + b3 + b4
        q = p + 12 + 16 * n16 + 8 * b5
        if q > p + bsz:
            raise DecodeError(f"parameter block table overflow at 0x{p:x}")
        sub_sizes = []
        for _ in range(b6):
            if q + 6 > p + bsz:
                raise DecodeError(f"truncated curve sub-record at 0x{q:x}")
            sub_sz = struct.unpack_from("<H", data, q + 4)[0]
            if sub_sz < 6 or q + sub_sz > p + bsz:
                raise DecodeError(f"invalid curve sub-record size {sub_sz} at 0x{q:x}")
            sub_sizes.append(sub_sz)
            q += sub_sz
        if q != p + bsz:
            raise DecodeError(f"parameter block size mismatch at 0x{p:x}: walked {q - p}, header {bsz}")
        blocks.append({
            "offset": p,
            "size": bsz,
            "index": b0,
            "bindings_16b": n16,
            "binding_counts": (b1, b2, b3, b4),
            "outputs_8b": b5,
            "curves": b6,
            "curve_bytes": sum(sub_sizes),
        })
        p += bsz
    return blocks, p


def _parse_container(data, off, end_limit, stats):
    """Recursively decode a `Type 1` (`StateMachine`), `Type 3` (`InlinedStateMachine`), or `Type 22` (`MotionTree`)."""
    if off + 0x24 > end_limit or end_limit > len(data):
        raise DecodeError(f"truncated state container at 0x{off:x}")
    (ntype, nidx, name_hash, init_rel, flags_0c,
     b10, b11, n_children, n_transitions) = struct.unpack_from("<HHIiI4B", data, off)
    if ntype not in CONTAINER_TYPES:
        raise DecodeError(f"unknown state container type {ntype} at 0x{off:x}")
    kind = CONTAINER_TYPES[ntype]
    stats["containers"] += 1
    stats[f"container:{kind}"] += 1
    if name_hash in KNOWN_SCHEMA_HASHES:
        stats[f"schema_name:{KNOWN_SCHEMA_HASHES[name_hash]}"] += 1
    init_target = off + 8 + init_rel
    if not (off < init_target < end_limit):
        raise DecodeError(f"initial/root target 0x{init_target:x} outside container 0x{off:x}..0x{end_limit:x}")

    if ntype in (TYPE_STATE_MACHINE, TYPE_INLINED_STATE_MACHINE):
        min_hdr = 0x58 if ntype == TYPE_STATE_MACHINE else 0x5C
        if off + min_hdr > end_limit:
            raise DecodeError(f"truncated {kind} header at 0x{off:x}")
        tr_rel = struct.unpack_from("<i", data, off + 0x24)[0]
        n_def_tr, def_tr_rel = struct.unpack_from("<Ii", data, off + 0x50)
        def_tr_start = off + 0x54 + def_tr_rel
        inputs = []
        children_end_limit = end_limit
        if ntype == TYPE_STATE_MACHINE:
            child_table_off = off + 0x58
        else:
            n_inputs = struct.unpack_from("<I", data, off + 0x58)[0]
            child_table_off = off + 0x5C + 4 * n_inputs
            if child_table_off > end_limit:
                raise DecodeError(f"truncated inlined_state_machine input table at 0x{off:x}")
            for i in range(n_inputs):
                ipos = off + 0x5C + 4 * i
                inp_target = ipos + struct.unpack_from("<i", data, ipos)[0]
                if not (0x20 <= inp_target < len(data)):
                    raise DecodeError(f"inlined_state_machine input target 0x{inp_target:x} out of bounds")
                inputs.append(inp_target)
            stats["inlined_state_machine_inputs"] += n_inputs
            fwd_inputs = [t for t in inputs if off < t <= end_limit]
            if fwd_inputs:
                children_end_limit = min(fwd_inputs)
        child_table_end = child_table_off + 8 * n_children
        tr_start = off + 0x24 + tr_rel if n_transitions > 0 else child_table_end
        if n_transitions > 0 and tr_start != child_table_end:
            raise DecodeError(
                f"{kind} at 0x{off:x}: transition start 0x{tr_start:x} != child table end 0x{child_table_end:x}"
            )
        child_entries = []
        for i in range(n_children):
            epos = child_table_off + 8 * i
            ch_hash, ch_rel = struct.unpack_from("<Ii", data, epos)
            ch_target = epos + 4 + ch_rel
            if not (child_table_end <= ch_target < children_end_limit):
                raise DecodeError(f"child state target 0x{ch_target:x} outside 0x{child_table_end:x}..0x{children_end_limit:x}")
            child_entries.append((ch_hash, ch_target))
        if n_children and init_target not in [t for _, t in child_entries]:
            raise DecodeError(f"{kind} at 0x{off:x}: initial state 0x{init_target:x} not in child directory")
        first_child_off = child_entries[0][1] if child_entries else children_end_limit
        if n_def_tr == 0 and def_tr_rel == 0:
            def_tr_start = first_child_off
        transitions = _parse_transitions(data, tr_start, def_tr_start, n_transitions) if n_transitions else []
        def_transitions = _parse_default_transitions(data, def_tr_start, first_child_off, n_def_tr) if n_def_tr else []
        stats["transitions"] += len(transitions)
        stats["default_transitions"] += len(def_transitions)
        stats["child_states"] += n_children
        stats[f"{kind}_children"] += n_children
        children = []
        for i, (ch_hash, ch_off) in enumerate(child_entries):
            ch_end = child_entries[i + 1][1] if i + 1 < n_children else children_end_limit
            child = _parse_container(data, ch_off, ch_end, stats)
            if child["name_hash"] != ch_hash:
                raise DecodeError(
                    f"child state at 0x{ch_off:x} hash 0x{child['name_hash']:08x} != directory hash 0x{ch_hash:08x}"
                )
            children.append(child)
        node = {
            "offset": off,
            "size": children_end_limit - off,
            "type": ntype,
            "kind": kind,
            "index": nidx,
            "name_hash": name_hash,
            "flags": flags_0c,
            "initial_state": init_target,
            "transitions": transitions,
            "default_transitions": def_transitions,
            "children": children,
        }
        if ntype == TYPE_INLINED_STATE_MACHINE:
            node["inputs"] = inputs
        return node

    # Type 22: MotionTree
    if off + 0x30 > end_limit:
        raise DecodeError(f"truncated motion_tree header at 0x{off:x}")
    tr_rel, blk_rel, n_param_blocks = struct.unpack_from("<iiI", data, off + 0x24)
    tr_start = off + 0x24 + tr_rel
    blk_start = off + 0x28 + blk_rel
    if tr_start != off + 0x30:
        raise DecodeError(f"motion_tree at 0x{off:x}: unexpected transition offset 0x{tr_start:x}")
    transitions = _parse_transitions(data, tr_start, blk_start, n_transitions)
    param_blocks, nodes_start = _parse_param_blocks(data, blk_start, end_limit, n_param_blocks)
    if not (nodes_start <= init_target <= end_limit - 8):
        raise DecodeError(
            f"motion_tree at 0x{off:x}: root node 0x{init_target:x} outside node arena 0x{nodes_start:x}..0x{end_limit:x}"
        )
    first_node_type, first_node_idx = struct.unpack_from("<HH", data, nodes_start)
    if first_node_idx != 0:
        raise DecodeError(f"motion_tree at 0x{off:x}: first node index is {first_node_idx}, expected 0")
    root_node_type, root_node_idx, root_node_hash = struct.unpack_from("<HHI", data, init_target)
    stats["transitions"] += len(transitions)
    stats["param_blocks"] += len(param_blocks)
    stats["param_bindings_16b"] += sum(b["bindings_16b"] for b in param_blocks)
    stats["param_outputs_8b"] += sum(b["outputs_8b"] for b in param_blocks)
    stats["param_curves"] += sum(b["curves"] for b in param_blocks)
    stats["motion_tree_children"] += n_children
    stats[f"root_node_type:{root_node_type}"] += 1

    embedded = []
    if root_node_type in (TYPE_STATE_MACHINE, TYPE_INLINED_STATE_MACHINE):
        embedded.append(_parse_container(data, init_target, end_limit, stats))

    return {
        "offset": off,
        "size": end_limit - off,
        "type": ntype,
        "kind": kind,
        "index": nidx,
        "name_hash": name_hash,
        "flags": flags_0c,
        "declared_children": n_children,
        "nodes_start": nodes_start,
        "node_arena_bytes": end_limit - nodes_start,
        "first_node_type": first_node_type,
        "root_node": {
            "offset": init_target,
            "type": root_node_type,
            "index": root_node_idx,
            "name_hash": root_node_hash,
        },
        "transitions": transitions,
        "param_blocks": param_blocks,
        "embedded_state_machines": embedded,
    }


def parse_move(data):
    """Decode one complete MoVE v19 resource buffer and verify its header, preamble, and state graph."""
    if not is_move_container(data):
        raise DecodeError("no MoVE magic at +0x00")
    version, z0, z1, z2, network_hash, body_size, z3 = struct.unpack_from("<7I", data, 4)
    if version != VERSION:
        raise DecodeError(f"unsupported MoVE version {version} (expected {VERSION})")
    if (z0, z1, z2, z3) != (0, 0, 0, 0):
        raise DecodeError("non-zero reserved header words in MoVE header")
    pos = 0x20
    if pos + 4 > len(data):
        raise DecodeError("truncated MoVE preamble at request count")
    n_req = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if pos + 4 * n_req + 4 > len(data):
        raise DecodeError("truncated MoVE request table")
    requests = list(struct.unpack_from(f"<{n_req}I", data, pos)) if n_req else []
    pos += 4 * n_req
    n_flag = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if pos + 4 * n_flag + 8 > len(data):
        raise DecodeError("truncated MoVE flag table")
    flags = list(struct.unpack_from(f"<{n_flag}I", data, pos)) if n_flag else []
    pos += 4 * n_flag
    preamble_mode, n_desc = struct.unpack_from("<2I", data, pos)
    pos += 8
    if preamble_mode not in (0, 1, 2):
        raise DecodeError(f"unexpected MoVE preamble_mode {preamble_mode}")
    if pos + 5 * n_desc + 10 > len(data):
        raise DecodeError("truncated MoVE 5-byte descriptor table or 10-byte preamble tail")
    descriptors = []
    for _ in range(n_desc):
        h, kind = struct.unpack_from("<IB", data, pos)
        descriptors.append({"hash": h, "kind": kind})
        pos += 5
    sub_flag, ref_hash_0, ref_hash_1 = struct.unpack_from("<HII", data, pos)
    pos += 10
    node_start = pos
    expected_size = 46 + 5 * n_desc + body_size
    if len(data) != expected_size:
        raise DecodeError(f"MoVE size mismatch: len={len(data)}, expected 46 + 5*{n_desc} + {body_size} = {expected_size}")
    if preamble_mode in (0, 1) and (sub_flag, ref_hash_0, ref_hash_1) != (0, 0, 0):
        raise DecodeError("non-zero 10-byte preamble tail when preamble_mode is 0 or 1")
    stats = collections.Counter()
    root = _parse_container(data, node_start, len(data), stats)
    return {
        "version": version,
        "network_hash": network_hash,
        "body_size": body_size,
        "bytes": len(data),
        "node_start": node_start,
        "graph_bytes": len(data) - node_start,
        "preamble": {
            "mode": preamble_mode,
            "requests": requests,
            "flags": flags,
            "descriptors_5b": descriptors,
            "tail": {"sub_flag": sub_flag, "ref_hash_0": ref_hash_0, "ref_hash_1": ref_hash_1},
        },
        "root": root,
        "counts": dict(sorted(stats.items())),
    }


def _inflate_at(fh, offset, stored):
    fh.seek(offset)
    return zlib.decompressobj().decompress(fh.read(stored))


def _collect_move_streams_rpf(path, start=None, limit=None):
    rows, walk = anim_index.scan_rpf(path, start, limit)
    items = []
    with open(path, "rb") as fh:
        i = 0
        while i < len(rows):
            row = rows[i]
            if row["signature"] != "move":
                i += 1
                continue
            blob = bytearray(_inflate_at(fh, row["offset"], row["stored"]))
            stored_total = row["stored"]
            stream_count = 1
            want = expected_total_size(blob)
            prev_end = row["offset"] + row["stored"]
            prev_size = row["size"]
            j = i + 1
            while (want is not None and len(blob) < want and j < len(rows)
                   and prev_size == anim_index.MAX_CHUNK
                   and rows[j]["offset"] <= prev_end + 16
                   and rows[j]["signature"] not in anim_index.STOP_SIGNATURES):
                blob += _inflate_at(fh, rows[j]["offset"], rows[j]["stored"])
                stored_total += rows[j]["stored"]
                prev_end = rows[j]["offset"] + rows[j]["stored"]
                prev_size = rows[j]["size"]
                stream_count += 1
                j += 1
            items.append((row["offset"], stream_count, stored_total, bytes(blob)))
            i = j
    return items, walk["files"]


def _iter_dir_stream_paths(path):
    """Yield (name, full_path) for every candidate stream file in a carved-streams directory."""
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if name.lower().endswith((".json", ".csv")) or os.path.islink(full) or not os.path.isfile(full):
            continue
        yield name, full


def _collect_move_streams_dir(path):
    entries = []
    for name, full in _iter_dir_stream_paths(path):
        size = os.path.getsize(full)
        if size == 0 or size > MAX_STANDALONE_FILE_BYTES:
            continue
        m = re.search(r"0x([0-9a-fA-F]+)|([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1) or m.group(2), 16) if m else len(entries)
        with open(full, "rb") as fh:
            data = fh.read()
        entries.append((offset, name, data))
    entries.sort(key=lambda e: (e[0], e[1]))
    items = []
    i = 0
    while i < len(entries):
        offset, _name, data = entries[i]
        if not is_move_container(data):
            i += 1
            continue
        blob = bytearray(data)
        stream_count = 1
        want = expected_total_size(blob)
        j = i + 1
        while (want is not None and len(blob) < want and j < len(entries)
               and len(entries[j - 1][2]) == anim_index.MAX_CHUNK
               and not is_move_container(entries[j][2])):
            blob += entries[j][2]
            stream_count += 1
            j += 1
        items.append((offset, stream_count, len(blob), bytes(blob)))
        i = j
    return items, len(entries)


def scan(path, start=None, limit=None):
    """Scan an RPF9 archive or a directory of carved streams and decode every MoVE v19 container."""
    if os.path.isdir(path):
        if start is not None or limit is not None:
            raise ValueError("--start and --limit apply only to an RPF archive")
        raw_items, total_streams = _collect_move_streams_dir(path)
    else:
        raw_items, total_streams = _collect_move_streams_rpf(path, start, limit)

    stats = collections.Counter()
    resources = []
    total_stored = 0
    total_inflated = 0
    total_move_streams = 0
    for offset, stream_count, stored_bytes, blob in raw_items:
        total_stored += stored_bytes
        total_inflated += len(blob)
        total_move_streams += stream_count
        try:
            parsed = parse_move(blob)
        except (DecodeError, struct.error) as e:
            stats["move_resources_unparsed"] += 1
            resources.append({
                "offset": offset,
                "streams": stream_count,
                "stored_bytes": stored_bytes,
                "bytes": len(blob),
                "error": str(e)[:160],
            })
            continue
        parsed["offset"] = offset
        parsed["streams"] = stream_count
        parsed["stored_bytes"] = stored_bytes
        resources.append(parsed)

        stats["move_resources"] += 1
        stats["move_streams"] += stream_count
        if stream_count > 1:
            stats["multi_stream_resources"] += 1
        stats[f"version:{parsed['version']}"] += 1
        p = parsed["preamble"]
        stats[f"preamble_mode:{p['mode']}"] += 1
        if p["requests"]:
            stats["resources_with_requests"] += 1
            stats["requests"] += len(p["requests"])
        if p["flags"]:
            stats["resources_with_flags"] += 1
            stats["flags"] += len(p["flags"])
        if p["descriptors_5b"]:
            stats["resources_with_descriptors_5b"] += 1
            stats["descriptors_5b"] += len(p["descriptors_5b"])
            for d in p["descriptors_5b"]:
                stats[f"descriptor_5b_kind:{d['kind']}"] += 1
        root = parsed["root"]
        stats[f"root_container:{root['kind']}"] += 1
        for k, v in parsed["counts"].items():
            stats[k] += v

    summary = {
        "streams": total_streams,
        "move_resources": len(raw_items),
        "move_streams": total_move_streams,
        "stored_bytes": total_stored,
        "inflated_bytes": total_inflated,
        "counts": dict(sorted(stats.items())),
    }
    return resources, summary


def _summary_row(r):
    """Build a game-data-free summary row for JSON output (counts and structural metadata only, no raw hash arrays)."""
    if "error" in r:
        return r
    p = r["preamble"]
    root = r["root"]
    c = r["counts"]
    return {
        "offset": r["offset"],
        "streams": r["streams"],
        "stored_bytes": r["stored_bytes"],
        "bytes": r["bytes"],
        "version": r["version"],
        "body_size": r["body_size"],
        "node_start": r["node_start"],
        "graph_bytes": r["graph_bytes"],
        "preamble_mode": p["mode"],
        "requests": len(p["requests"]),
        "flags": len(p["flags"]),
        "descriptors_5b": len(p["descriptors_5b"]),
        "root_type": root["type"],
        "root_kind": root["kind"],
        "containers": c.get("containers", 0),
        "state_machines": c.get("container:state_machine", 0),
        "inlined_state_machines": c.get("container:inlined_state_machine", 0),
        "motion_trees": c.get("container:motion_tree", 0),
        "transitions": c.get("transitions", 0),
        "default_transitions": c.get("default_transitions", 0),
        "param_blocks": c.get("param_blocks", 0),
    }


def _write_json_atomic(path, doc):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".anim_move_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="anim.rpf or a directory of carved MoVE streams")
    ap.add_argument("out", help="output .json (counts and one row per MoVE resource)")
    ap.add_argument("--start", type=anim_index.offset, help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=anim_index.byte_count, help="number of bytes to scan")
    a = ap.parse_args(argv)
    if not os.path.exists(a.archive):
        sys.exit(f"{a.archive}: No such file or directory")
    if os.path.isdir(a.archive) and (a.start is not None or a.limit is not None):
        ap.error("--start and --limit cannot be used when the input is a directory")
    if os.path.isdir(a.out):
        sys.exit(f"{a.out}: cannot write: Is a directory")
    inputs = [full for _name, full in _iter_dir_stream_paths(a.archive)] if os.path.isdir(a.archive) else [a.archive]
    if any(same_report_path(path, a.out) for path in inputs):
        sys.exit(f"{a.out}: refuses to overwrite the input archive")
    try:
        resources, summary = scan(a.archive, a.start, a.limit)
    except ValueError as e:
        sys.exit(f"{a.archive}: {e}")
    except OSError as e:
        sys.exit(f"{e.filename or a.archive}: {e.strerror or e}")
    rows = [_summary_row(r) for r in resources]
    try:
        _write_json_atomic(a.out, {"summary": summary, "move_resources": rows})
    except OSError as e:
        sys.exit(f"{a.out}: cannot write: {e.strerror or e}")
    c = summary["counts"]
    print(json.dumps({
        k: c.get(k, 0)
        for k in (
            "move_resources",
            "move_streams",
            "multi_stream_resources",
            "containers",
            "container:state_machine",
            "container:inlined_state_machine",
            "container:motion_tree",
            "transitions",
            "default_transitions",
            "param_blocks",
        )
    }))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
