"""Decode the SPS1 shader-preset table in common.rpf and cross-check with shaders.rpf.

Format measured on the Xbox Series 1.0.2.1 common.rpf (offset 0x650000, 43,791 B):

    0x00  "SPS1"              4-byte magic
    0x04  u32 entry_count     number of records (858 = 0x0000035A; low byte 0x5A
                              is ASCII 'Z', which earlier notes misread as "SPS1Z")
    0x08  entry_count records of:
            u8   preset_len   byte length of NUL-terminated ASCII preset path
            char preset[preset_len]
            u8   effect_len   byte length of NUL-terminated target SGD5 effect name
            char effect[effect_len]
            u8   render_bucket  default RAGE render bucket (0..7)

    python -I tools/gamedata_sps1.py <carve_dir_or_rpf_or_bin> [--shaders PATH/TO/shaders.rpf] [--preset-hash HEX] [--json FILE]

Only header fields, aggregate counts, bucket distributions and hashed queries are
printed or written; preset and effect names stay on the local machine. Standard
library only.
Usage: `python -I tools/gamedata_sps1.py --help` lists the options.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import posixpath
import re
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing sibling helpers.
TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import gamedata_index as gi  # noqa: E402
import gamedata_maps as gm  # noqa: E402
from hashdict import joaat  # noqa: E402
from report_paths import same_report_path  # noqa: E402
import sgd5_effects as se  # noqa: E402

SPS1_MAGIC = b"SPS1"
HEADER = 8
BOUNDARY_MAGICS = (b"MHIU", b"PAMH", b"SPS1", b"RBF0", b"PSIN")
XML_PREFIXES = (b"<?xml", b"\xef\xbb\xbf<")


def _read_c_string(blob: bytes, pos: int, label: str) -> tuple[str, int]:
    """Read one u8-length-prefixed, NUL-terminated printable ASCII string."""
    if pos >= len(blob):
        raise ValueError(f"truncated {label} length at {pos:#x}")
    length = blob[pos]
    pos += 1
    if length < 2:
        raise ValueError(f"invalid {label} string length at {pos - 1:#x}")
    if pos + length > len(blob):
        raise ValueError(f"truncated {label} string at {pos - 1:#x}")
    raw = blob[pos:pos + length]
    if raw[-1] != 0 or 0 in raw[:-1]:
        raise ValueError(f"{label} string at {pos - 1:#x} must end with a single NUL")
    body = raw[:-1]
    if any(b < 0x20 or b > 0x7E for b in body):
        raise ValueError(f"non-printable byte in {label} string at {pos - 1:#x}")
    return body.decode("ascii"), pos + length


def _preset_stem(preset_path: str) -> str:
    """Basename of a .sps preset path without its .sps suffix."""
    base = posixpath.basename(preset_path.replace("\\", "/"))
    if base.lower().endswith(".sps"):
        return base[:-4]
    return base


def parse_sps1(blob: bytes) -> dict:
    """Decode an SPS1 binary blob into aggregate statistics plus private _entries."""
    if len(blob) < HEADER or blob[:4] != SPS1_MAGIC:
        raise ValueError("not an SPS1 payload")
    entry_count = struct.unpack_from("<I", blob, 4)[0]
    pos = HEADER
    entries = []
    for index in range(entry_count):
        preset, pos = _read_c_string(blob, pos, f"preset[{index}]")
        effect, pos = _read_c_string(blob, pos, f"effect[{index}]")
        if pos >= len(blob):
            raise ValueError(f"truncated render_bucket[{index}] at {pos:#x}")
        bucket = blob[pos]
        pos += 1
        entries.append((preset, effect, bucket))
    if pos != len(blob):
        raise ValueError(f"SPS1 trailing bytes: consumed {pos} of {len(blob)} bytes")

    buckets_counter: Counter[int] = Counter()
    buckets_by_effect: dict[str, set[int]] = defaultdict(set)
    distinct_presets: set[str] = set()
    distinct_stems: set[str] = set()
    root_stems: set[str] = set()
    subfolder_stems: list[str] = []
    preset_lengths: list[int] = []
    effect_lengths: list[int] = []
    sps_ext = root_presets = subfolder_presets = identity = 0

    for preset, effect, bucket in entries:
        low_preset = preset.lower()
        low_effect = effect.lower()
        stem = _preset_stem(preset).lower()
        distinct_presets.add(low_preset)
        distinct_stems.add(stem)
        preset_lengths.append(len(preset))
        effect_lengths.append(len(effect))
        buckets_counter[bucket] += 1
        buckets_by_effect[low_effect].add(bucket)
        if low_preset.endswith(".sps"):
            sps_ext += 1
        if "/" in preset or "\\" in preset:
            subfolder_presets += 1
            subfolder_stems.append(stem)
        else:
            root_presets += 1
            root_stems.add(stem)
        if stem == low_effect:
            identity += 1

    multi_combos = Counter(
        ",".join(str(b) for b in sorted(bset))
        for bset in buckets_by_effect.values()
        if len(bset) > 1
    )
    low_byte = entry_count & 0xFF
    return {
        "type": "SPS1",
        "header": {
            "magic": "SPS1",
            "entry_count": entry_count,
            "low_count_byte_ascii": chr(low_byte) if 0x20 <= low_byte <= 0x7E else None,
        },
        "table": {
            "records": len(entries),
            "data_bytes": len(blob) - HEADER,
            "distinct_presets": len(distinct_presets),
            "distinct_preset_stems": len(distinct_stems),
            "distinct_effects": len(buckets_by_effect),
            "sps_extension_count": sps_ext,
            "root_presets": root_presets,
            "subfolder_presets": subfolder_presets,
            "subfolder_stem_collisions": sum(1 for s in subfolder_stems if s in root_stems),
            "identity_presets": identity,
            "aliased_presets": len(entries) - identity,
            "preset_length_min": min(preset_lengths) if preset_lengths else 0,
            "preset_length_max": max(preset_lengths) if preset_lengths else 0,
            "effect_length_min": min(effect_lengths) if effect_lengths else 0,
            "effect_length_max": max(effect_lengths) if effect_lengths else 0,
            "render_buckets": {str(k): buckets_counter[k] for k in sorted(buckets_counter)},
            "multi_bucket_effects": sum(1 for bset in buckets_by_effect.values() if len(bset) > 1),
            "multi_bucket_combinations": {
                k: multi_combos[k] for k in sorted(multi_combos, key=lambda s: (-multi_combos[s], s))
            },
        },
        "_entries": entries,
    }


def _load_from_source(rows: list[dict], read_row) -> list[dict]:
    """Extract and parse all SPS1 payloads from sorted index rows."""
    out = []
    i = 0
    while i < len(rows):
        row = rows[i]
        head_hex = row.get("head", "")
        if len(head_hex) >= 8 and not head_hex.lower().startswith("53505331"):
            i += 1
            continue
        data = read_row(row)
        if data[:4] != SPS1_MAGIC:
            i += 1
            continue
        parts = [data]
        j = i
        blob = data
        while True:
            try:
                parsed = parse_sps1(blob)
                break
            except ValueError as err:
                if (
                    "truncated" not in str(err)
                    or not (len(parts[-1]) > 0 and len(parts[-1]) % 65536 == 0)
                    or j + 1 >= len(rows)
                ):
                    raise
                nxt = read_row(rows[j + 1])
                if nxt[:4] in BOUNDARY_MAGICS or nxt.startswith(XML_PREFIXES):
                    raise
                parts.append(nxt)
                j += 1
                blob = b"".join(parts)
        parsed.update({
            "offset": row["off"],
            "offset_hex": f'{row["off"]:x}',
            "stored": sum(rows[x].get("stored", 0) for x in range(i, j + 1)),
            "size": len(blob),
            "chunks": j - i + 1,
        })
        out.append(parsed)
        i = j + 1
    return out


def read_payloads(target: str) -> list[dict]:
    """Load SPS1 payloads from a carve directory, a raw .bin file, or an .rpf archive."""
    if os.path.isdir(target):
        rows = gm._read_index_rows(target)
        return _load_from_source(rows, lambda r: gm._read_row(target, r))
    if not os.path.isfile(target):
        raise FileNotFoundError(target)
    with open(target, "rb") as f:
        head = f.read(4)
    if head == SPS1_MAGIC:
        with open(target, "rb") as f:
            blob = f.read()
        parsed = parse_sps1(blob)
        parsed.update({
            "offset": 0,
            "offset_hex": "0",
            "stored": len(blob),
            "size": len(blob),
            "chunks": 1,
        })
        return [parsed]
    if head != b"9FPR":
        raise ValueError(f"{target} is neither an SPS1 payload nor an RPF9 archive")
    source = gi.ArchiveSource(target)
    try:
        return _load_from_source(source.rows, source.read)
    finally:
        source.close()


def cross_reference_sgd5(payloads: list[dict], shaders_rpf: str) -> dict:
    """Resolve SPS1 target effect names and preset stems against shaders.rpf SGD5 databases."""
    with open(shaders_rpf, "rb") as stream:
        db0_res = se.read_database(stream, se.DATABASES[0], list_effects=True)
        db1_res = se.read_database(stream, se.DATABASES[1], list_effects=True)

    db0_names = {m["effect"].lower() for m in db0_res["matches"]}
    db1_names = {m["effect"].lower() for m in db1_res["matches"]}
    all_sgd5 = db0_names | db1_names

    resolved_db0 = resolved_db1 = unresolved = 0
    matched_effects_db0: set[str] = set()
    matched_effects_db1: set[str] = set()
    distinct_stems: set[str] = set()
    stems_in_sgd5: set[str] = set()

    for payload in payloads:
        for preset, effect, _ in payload["_entries"]:
            low_effect = effect.lower()
            stem = _preset_stem(preset).lower()
            distinct_stems.add(stem)
            if stem in all_sgd5:
                stems_in_sgd5.add(stem)
            if low_effect in db0_names:
                resolved_db0 += 1
                matched_effects_db0.add(low_effect)
            elif low_effect in db1_names:
                resolved_db1 += 1
                matched_effects_db1.add(low_effect)
            else:
                unresolved += 1

    return {
        "sgd5_db0_effects": len(db0_names),
        "sgd5_db1_effects": len(db1_names),
        "sgd5_total_effects": len(all_sgd5),
        "records_resolved_db0": resolved_db0,
        "records_resolved_db1": resolved_db1,
        "records_unresolved": unresolved,
        "distinct_effects_resolved_db0": len(matched_effects_db0),
        "distinct_effects_resolved_db1": len(matched_effects_db1),
        "preset_stems_in_sgd5": len(stems_in_sgd5),
        "sps1_only_preset_stems": len(distinct_stems - stems_in_sgd5),
    }


def lookup_preset_hash(payloads: list[dict], target_hash: int) -> list[dict]:
    """Match a joaat hash against preset stems, preset paths, or effect names (hashes only returned)."""
    matches = []
    for payload in payloads:
        for idx, (preset, effect, bucket) in enumerate(payload["_entries"]):
            stem = _preset_stem(preset)
            stem_h = joaat(stem)
            path_h = joaat(preset)
            eff_h = joaat(effect)
            if target_hash in (stem_h, path_h, eff_h):
                matches.append({
                    "record_index": idx,
                    "preset_stem_hash": f"{stem_h:08X}",
                    "preset_path_hash": f"{path_h:08X}",
                    "effect_hash": f"{eff_h:08X}",
                    "render_bucket": bucket,
                    "aliased": stem.lower() != effect.lower(),
                })
    return matches


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
    ap.add_argument("target", help="rpf9_carve.py directory, common.rpf archive, or SPS1 .bin file")
    ap.add_argument("--shaders", help="shaders.rpf archive to cross-reference target effects against SGD5")
    ap.add_argument("--preset-hash", metavar="HEX",
                    help="match a joaat32 hash against preset stem, path, or effect (prints hashes only)")
    ap.add_argument("--json", help="write the aggregate summary and lookup hashes to this JSON")
    a = ap.parse_args(argv)

    target_hash = None
    if a.preset_hash is not None:
        if not re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,8}", a.preset_hash):
            ap.error("--preset-hash expects up to eight hexadecimal digits")
        target_hash = int(a.preset_hash, 16)

    # A missing input path is a usage error (exit 2), not a failed run, as in BUILD-1046.
    for path in (a.target, a.shaders):
        if path is not None and not os.path.exists(path):
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            return 2

    try:
        if a.json:
            check_report_path(a.json, a.target, a.shaders)
        if os.path.isdir(a.target):
            rows = gm._read_index_rows(a.target)
            if a.json:
                check_report_path(a.json, a.target, a.shaders, rows=rows)
        payloads = read_payloads(a.target)
        xref = cross_reference_sgd5(payloads, a.shaders) if a.shaders else None
        matches = lookup_preset_hash(payloads, target_hash) if target_hash is not None else None
    except (OSError, ValueError, KeyError, TypeError, struct.error, RuntimeError) as e:
        print("error:", e, file=sys.stderr)
        return 1

    for p in payloads:
        t = p["table"]
        print(
            f'SPS1 {p["offset_hex"]}: {p["size"]:,} B, {t["records"]} presets -> '
            f'{t["distinct_effects"]} effects ({t["identity_presets"]} identity, {t["aliased_presets"]} aliased), '
            f'buckets {t["render_buckets"]}'
        )
    if not payloads:
        print("no SPS1 payloads")
    if xref is not None:
        print(
            f'SGD5 xref: {xref["records_resolved_db0"]}/{sum(p["table"]["records"] for p in payloads)} '
            f'records in DB0 ({xref["distinct_effects_resolved_db0"]} distinct effects), '
            f'{xref["preset_stems_in_sgd5"]} stems in SGD5, {xref["sps1_only_preset_stems"]} SPS1-only'
        )
    if matches is not None:
        print(f"hash matches: {len(matches)}")
        for m in matches:
            print(
                f'  [{m["record_index"]}] stem={m["preset_stem_hash"]} '
                f'effect={m["effect_hash"]} bucket={m["render_bucket"]} aliased={m["aliased"]}'
            )

    if a.json:
        out: dict = {"payloads": [_public(p) for p in payloads]}
        if xref is not None:
            out["sgd5_xref"] = xref
        if matches is not None:
            out["hash_matches"] = matches
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
