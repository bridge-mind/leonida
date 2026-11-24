"""Decode RSC8 (version 8001) facial/lip-sync animation clips embedded in AWC (ADAT) speech banks.

`docs/story.md` and #716 flagged `RSC8` (version 8001) as the remaining undecoded mission-side
container in the story/minigame archives (`story000.rpf`, `shared_minigame000.rpf`,
`story000_restofworld.rpf`). Full-corpus decoding shows that `RSC8` v8001 is not a standalone
audio codec or audio container: every validated `RSC8` v8001 resource sits inside a standard
little-endian `ADAT` (AWC) bank as chunk tag `0x5c`, attached to a spoken dialogue stream whose
audio lives in sibling tag `0x55` (48 kHz mono Opus, codec `0x0b`, format tag `0x76`).

Inside tag `0x5c`, the payload is an unencrypted RAGE `crClipDictionary` + `crClipAnimation` +
`crAnimationDictionary` + `crAnimation` facial/lip-sync animation resource:
  - +0x00  4s   "RSC8"
  - +0x04  u32  version (8001 = 0x00001F41, or 0x01001F41 with version flag bit 24 set)
  - +0x08  u32  body_size (bytes after +0x0C; exact tag 0x5c length == body_size + 12)
  - +0x0C  u32  flags (1)
  - +0x10  ...  virtual segment (`0x50xxxxxx` pointers resolve to `0x10 + (ptr - 0x50000000)`)

This tool is stdlib-only, runs under `python -I`, decrypts nothing, and never writes game
strings or raw resource payloads to stdout, JSON, or CSV. Only structural offsets, sizes,
counts, hashes, string storage/lengths, frame counts, durations and track types are reported.

Usage:
    python -I tools/rsc8_audio.py <archive.rpf | bank.awc | clip.rsc8 | dir> [out.json|out.csv]
        [--start N] [--limit BYTES] [--max-banks N] [--no-deep] [--curves] [--summary-only] [--quiet]

`--curves` also decodes every clip's keyframes (the blocks are the sequence layout of
tools/anim_clips.py) and counts the outcome; it reports counts only, never a value.

Every scan also reads the `0xbd` cue records of each stream; the summary's `marker_regions` block
matches the records whose value is joaat("start") or joaat("end") into region pairs, name by name
(pair_marker_regions(), docs/rsc8_audio.md).
"""
import argparse
import csv
import json
import math
import os
import statistics
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_clips  # noqa: E402
from file_mode import published_mode  # noqa: E402

RSC8_MAGIC = b"RSC8"
RSC8_VERSION = 8001
RSC8_VERSION_FLAGGED = 0x01001F41
RSC8_VERSIONS = (RSC8_VERSION, RSC8_VERSION_FLAGGED)
RSC8_FLAGS = 1
RSC8_VIRTUAL_BASE = 0x10
RSC8_VPTR_BASE = 0x50000000
RSC8_VPTR_LIMIT = 0x60000000
TRACK_SCHEMA_HASH = 0x364D3BDB  # joaat("facialtranslation"): the track type of every directory entry

ADAT_MAGIC = b"ADAT"
TAG_PEAK = 0x36
TAG_STREAMED_MUSIC = 0x48
TAG_DATA = 0x55
TAG_GRID20 = 0x5A  # undocumented 20-byte-entry table; decode_grid20_table()
TAG_RSC8_CLIP = 0x5C
TAG_FORMAT = 0x76
TAG_SEEKTABLE = 0xA3
TAG_LOUDNESS = 0xB2
TAG_MARKER = 0xBD
TAG_INDEXRUN = 0xD9  # undocumented grouped index runs into the 0x5a entries; decode_index_runs()
# The most common cue-name hash of a `0xbd` record, not a magic: it is the first word of 88,082 of the
# 420,168 `0xbd` tags in `audio.rpf` (decode_marker_records()). The old name is kept for callers.
MARKER_NAME_COMMON = 0xE5C74063
MARKER_MAGIC = MARKER_NAME_COMMON
MARKER_RECORD_BYTES = 16
MAX_MARKER_BYTES = 16 * 1024 * 1024
# A region cue stores a second name hash in its `value` word (pair_marker_regions()).
MARKER_VALUE_START = 0x84DC271F  # joaat("start")
MARKER_VALUE_END = 0xB0C485D7  # joaat("end")
_REGION_VALUES = {MARKER_VALUE_START: "start", MARKER_VALUE_END: "end"}
MAX_REGION_EXCEPTIONS_LISTED = 64

ATTR_FLOAT = 0xFFFF0001
ATTR_INT = 0xFFFF0002
ATTR_BOOL = 0xFFFF0003
ATTR_STRING = 0xFFFF0004
ATTR_GUID = 0xFFFF000E
PROP_KEY_FPS = 0x10BF2F94
PROP_KEY_LEAD_IN = 0xAAA6AF06

MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_RSC8_BYTES = 32 * 1024 * 1024
MAX_AWC_FILE_BYTES = 512 * 1024 * 1024
MAX_BANK_STREAMS = 200_000
MAX_ANIM_BLOCKS = 256
MAX_TRACK_ENTRIES = 4096
MAX_PROP_ENTRIES = 256
SCAN_CHUNK_BYTES = 32 * 1024 * 1024
PROGRESS_BYTES = 2 * 1024 * 1024 * 1024

CSV_COLUMNS = [
    "bank_offset",
    "tag_offset",
    "stream_id",
    "tag_size",
    "body_size",
    "version",
    "flags",
    "stream_hash",
    "dict_name_storage",
    "dict_name_len",
    "anim_block_count",
    "anim_size",
    "frame_count",
    "clip_duration",
    "audio_duration",
    "audio_codec",
    "audio_channels",
    "audio_sample_rate",
    "track_count",
    "prop_count",
    "valid",
]


def _u16(buf, off):
    return struct.unpack_from("<H", buf, off)[0]


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def _u64(buf, off):
    return struct.unpack_from("<Q", buf, off)[0]


def _f32(buf, off):
    return struct.unpack_from("<f", buf, off)[0]


def resolve_vptr(virt, voff):
    """Resolve one 64-bit virtual pointer at `voff` inside `virt`.

    Returns:
      - `0` when the stored pointer is null.
      - An integer offset `0 <= target < len(virt)` for a valid `0x50xxxxxx` pointer.
      - `None` if `voff` is out of bounds or the pointer points outside `virt`.
    """
    vlen = len(virt)
    if voff < 0 or voff + 8 > vlen:
        return None
    q = _u64(virt, voff)
    if q == 0:
        return 0
    if RSC8_VPTR_BASE <= q < RSC8_VPTR_LIMIT:
        target = q - RSC8_VPTR_BASE
        if 0 <= target < vlen:
            return target
    return None


def _cstr_info(virt, off, max_scan=1024):
    """Measure a NUL-terminated printable ASCII string without returning its text."""
    vlen = len(virt)
    if off is None or off < 0 or off >= vlen:
        return False, 0
    end = min(vlen, off + max_scan)
    seg = bytes(virt[off:end])
    nul = seg.find(b"\x00")
    if nul < 0:
        return False, 0
    printable = all(32 <= b < 127 for b in seg[:nul])
    return printable, nul


def decode_sso_string(virt, voff):
    """Decode a 16-byte RAGE SSO `atString` at `virt[voff : voff + 16]`.

    Layout:
      - Inline mode: `byte[15] & 0x80 != 0`; `length = byte[15] & 0x7F` (<= 14),
        ASCII text in `raw[0:length]`, NUL at `raw[length]`.
      - Heap mode: `u64 ptr` (`0x50xxxxxx`), `u16 length`, `u16 capacity`, `u32 flags (= 1)`,
        NUL-terminated ASCII string at `virt[ptr : ptr + length + 1]`.

    Never returns raw string content.
    """
    vlen = len(virt)
    if voff < 0 or voff + 16 > vlen:
        return {
            "storage": "invalid",
            "length": 0,
            "capacity": None,
            "flags": None,
            "ptr": None,
            "ok": False,
        }
    raw = bytes(virt[voff : voff + 16])
    last = raw[15]
    if last & 0x80:
        length = last & 0x7F
        ok = (
            length <= 14
            and raw[length] == 0
            and all(32 <= b < 127 for b in raw[:length])
        )
        return {
            "storage": "inline",
            "length": length,
            "capacity": 15,
            "flags": last,
            "ptr": None,
            "ok": ok,
        }
    ptr = resolve_vptr(virt, voff)
    length, capacity, flags = struct.unpack_from("<HHI", raw, 8)
    if ptr is None or ptr == 0:
        return {
            "storage": "invalid",
            "length": length,
            "capacity": capacity,
            "flags": flags,
            "ptr": ptr,
            "ok": False,
        }
    if capacity < length + 1 or ptr + length >= vlen:
        return {
            "storage": "heap",
            "length": length,
            "capacity": capacity,
            "flags": flags,
            "ptr": ptr,
            "ok": False,
        }
    text_ok = virt[ptr + length] == 0 and all(32 <= b < 127 for b in virt[ptr : ptr + length])
    return {
        "storage": "heap",
        "length": length,
        "capacity": capacity,
        "flags": flags,
        "ptr": ptr,
        "ok": text_ok and flags == 1,
    }


def decode_properties(virt, props_off, props_size):
    """Decode the `crProperties` table at `virt[props_off : props_off + props_size]`."""
    vlen = len(virt)
    if (
        props_off is None
        or props_off <= 0
        or props_size < 0x20
        or props_off + props_size > vlen
    ):
        return {
            "in_bounds": False,
            "clip_ptr": None,
            "count": 0,
            "capacity": 0,
            "attributes": [],
            "fps": None,
            "lead_in": None,
            "attrs_tiled_to_end": False,
        }

    clip_ptr = resolve_vptr(virt, props_off + 0x10)
    prop_count = _u32(virt, props_off + 0x18)
    prop_cap = _u32(virt, props_off + 0x1C)
    result = {
        "in_bounds": True,
        "clip_ptr": clip_ptr,
        "count": prop_count,
        "capacity": prop_cap,
        "attributes": [],
        "fps": None,
        "lead_in": None,
        "attrs_tiled_to_end": False,
    }
    if prop_count > MAX_PROP_ENTRIES:
        result["in_bounds"] = False
        return result

    # Locate typed attribute records `(u32 key_hash, u32 0xFFFFxxxx type_id, ...)` in the tail
    # after the `0x48`-byte property descriptors (descriptors start after the 0x70-byte map header).
    tail_start = 0x70 + prop_count * 0x48
    if tail_start > props_size:
        result["in_bounds"] = False
        return result
    # Advance to the first aligned typed attribute (0xFFFFxxxx marker at +4).
    pos = tail_start
    while pos + 8 <= props_size and (_u32(virt, props_off + pos + 4) >> 16) != 0xFFFF:
        pos += 4

    attrs = []
    while pos + 8 <= props_size and len(attrs) < MAX_PROP_ENTRIES:
        key_hash = _u32(virt, props_off + pos)
        attr_type = _u32(virt, props_off + pos + 4)
        if (attr_type >> 16) != 0xFFFF:
            break
        if attr_type == ATTR_STRING:
            if pos + 0x40 > props_size:
                break
            str_len = _u64(virt, props_off + pos + 0x18)
            attrs.append({"key_hash": key_hash, "type": attr_type, "string_len": str_len})
            pos += 0x40
        elif attr_type == ATTR_GUID:
            if pos + 0x20 > props_size:
                break
            attrs.append({"key_hash": key_hash, "type": attr_type, "guid_bytes": 16})
            pos += 0x20
        elif attr_type in (ATTR_FLOAT, ATTR_INT, ATTR_BOOL):
            if pos + 0x0C > props_size:
                break
            if attr_type == ATTR_FLOAT:
                val = _f32(virt, props_off + pos + 8)
                entry = {"key_hash": key_hash, "type": attr_type, "value": round(val, 6) if math.isfinite(val) else None}
                if key_hash == PROP_KEY_LEAD_IN and math.isfinite(val):
                    result["lead_in"] = round(val, 6)
            else:
                val = _u32(virt, props_off + pos + 8)
                entry = {"key_hash": key_hash, "type": attr_type, "value": val}
                if key_hash == PROP_KEY_FPS and attr_type == ATTR_INT:
                    result["fps"] = val
            attrs.append(entry)
            pos += 0x0C
        else:
            break

    result["attributes"] = attrs
    result["attrs_tiled_to_end"] = pos == props_size
    return result


def _has_embedded_audio_container(data):
    """Return True only if `data` contains a validated AWC, Ogg Opus, or RIFF WAVE header."""
    raw = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    mv_raw = memoryview(raw)
    rlen = len(raw)
    pos = 0
    while True:
        idx = raw.find(ADAT_MAGIC, pos)
        if idx < 0:
            break
        if parse_adat_header_bytes(mv_raw[idx:], 0, rlen - idx) is not None:
            return True
        pos = idx + 4
    if b"OggS\x00" in raw and b"OpusHead" in raw:
        return True
    pos = 0
    while True:
        idx = raw.find(b"RIFF", pos)
        if idx < 0:
            break
        if idx + 12 <= rlen and raw[idx + 8 : idx + 12] == b"WAVE":
            return True
        pos = idx + 4
    return False


def decode_facial_curves(virt, anim, anim_desc, stats=None):
    """Decode the keyframes of one clip: `({track index: [value tuple per frame]}, frames, tracks)`.

    `virt` is the resource from `RSC8_VIRTUAL_BASE` on; `anim` and `anim_desc` come from `decode_rsc8`.
    A packed block is one sequence of tools/anim_clips.py (0x20-byte header, version 2, 11 channel
    types) and the track directory is its 12-byte `{id hash, track-type hash, value format}` list, so
    the block and track decoders are that tool's. `tracks` is the directory as stored. Raises
    `anim_clips.DecodeError` when a block does not decode. A contiguous view is decoded in place,
    without copying the resource.
    """
    if isinstance(virt, memoryview):
        virt = virt.cast("B") if virt.c_contiguous else bytes(virt)
    system = anim_clips.System(virt)
    at, count = anim_desc["track_ptr"] or 0, anim_desc["track_count"] or 0
    if count and at + 12 * count > len(system.d):
        raise anim_clips.DecodeError("track directory runs past the resource")
    tracks = [struct.unpack_from("<III", system.d, at + 12 * i) for i in range(count)]
    clip = {"sequences": [b["voff"] for b in anim["blocks"]], "tracks": tracks, "frames": anim["frame_count"]}
    try:
        values, frames = anim_clips.decode_animation(system, clip, stats)
    except (struct.error, IndexError, OverflowError, ZeroDivisionError) as e:
        kind = type(e).__name__ if type(e).__module__ == "builtins" else f"{type(e).__module__}.{type(e).__name__}"
        raise anim_clips.DecodeError(f"malformed block: {kind}") from e
    return values, frames, tracks


def _extents_overlap(anim, anim_desc):
    """True when two packed blocks, or a block and the track directory, share a byte."""
    spans = [(b["voff"], b["voff"] + b["anim_size"]) for b in anim["blocks"]]
    count = anim_desc["track_count"] or 0
    if count:
        spans.append((anim_desc["track_ptr"], anim_desc["track_ptr"] + 12 * count))
    spans.sort()
    return any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans)))


def facial_curve_summary(virt, anim, anim_desc):
    """Counts from decoding one clip's keyframes; no value leaves this function.

    `decoded` is False with the `error` text when a block does not decode. Otherwise: `channels` by
    channel type, the type-10 chunk boundaries and how many close, `components` as
    `"<value format>:<components per frame>"` per track, and whether the frame count and the header's
    quantised-bit sum agree with the decoded data.
    """
    stats = anim_clips.new_stats()
    try:
        values, frames, tracks = decode_facial_curves(virt, anim, anim_desc, stats)
    except anim_clips.DecodeError as e:
        return {"decoded": False, "error": str(e)[:80]}
    components = {}
    for index, rows in values.items():
        key = f"{tracks[index][2]}:{len(rows[0]) if rows else 0}"
        components[key] = components.get(key, 0) + 1
    blocks = len(anim["blocks"])
    return {
        "decoded": True,
        "error": None,
        "frames": frames,
        "frames_match": frames == anim["frame_count"] and stats["frames_sum_matches"] == 1,
        "quantize_bits_match": stats["quantize_bits_match"] == blocks,
        "tracks_with_values": len(values),
        "channels": {k[len("channel:"):]: v for k, v in sorted(stats.items()) if k.startswith("channel:")},
        "delta2_boundaries": stats["delta2_boundaries"],
        "delta2_boundaries_closed": stats["delta2_boundaries_closed"],
        "components": components,
    }


def decode_rsc8(data, *, stream_id=None, deep=True, curves=False):
    """Decode one `RSC8` (version 8001) facial/lip-sync animation resource.

    Accepts `bytes`, `bytearray`, or `memoryview`. Returns a structured dict of decoded fields
    and boolean integrity checks. Never raises on malformed/truncated input (returns `valid=False`).
    """
    mv = memoryview(data)
    total_size = len(mv)
    checks = {
        "min_size_ok": total_size >= 0xE0,
        "magic_ok": total_size >= 4 and bytes(mv[:4]) == RSC8_MAGIC,
        "version_ok": False,
        "size_matches_body_plus_12": False,
        "flags_ok": False,
        "aligned_16": (total_size % 16) == 0,
        "dict_name_ok": False,
        "anim_dict_entry_ok": False,
        "props_in_bounds": False,
        "clip_in_bounds": False,
        "clip_name_ok": False,
        "clip_hash_matches_root": False,
        "clip_duration_matches_frames": False,
        "anim_desc_in_bounds": False,
        "anim_blocks_in_bounds": False,
        "anim_payload_plus_32_eq_size": False,
        "anim_chunk_off_in_bounds": False,
        "anim_format_v2": False,
        "anim_track_tag_11": False,
        "anim_max_block_size_matches_desc": False,
        "tracks_in_bounds": False,
        "tracks_sorted": False,
        "tracks_schema_ok": False,
        "no_embedded_audio_container": True,
    }
    if stream_id is not None:
        checks["stream_id_matches"] = False

    out = {
        "total_size": total_size,
        "magic": bytes(mv[:4]).decode("ascii", "replace") if total_size >= 4 else "",
        "version": None,
        "version_base": None,
        "version_flags": None,
        "body_size": None,
        "flags": None,
        "virtual_size": max(0, total_size - RSC8_VIRTUAL_BASE),
        "stream_hash": None,
        "dict_flags": None,
        "dict_name": None,
        "clip_map_buckets": None,
        "clip_map_count": None,
        "hash_map_buckets": None,
        "hash_map_count": None,
        "props_ptr": None,
        "props_size": None,
        "anim": None,
        "clip": None,
        "anim_desc": None,
        "properties": None,
        "tracks": None,
        "curves": None,
        "checks": checks,
        "valid": False,
    }

    if total_size < 0xE0 or not checks["magic_ok"]:
        return out

    checks["no_embedded_audio_container"] = not _has_embedded_audio_container(data if isinstance(data, (bytes, bytearray)) else mv)

    version, body_size, flags = struct.unpack_from("<III", mv, 4)
    out["version"] = version
    out["version_base"] = version & 0xFFFF
    out["version_flags"] = version >> 16
    out["body_size"] = body_size
    out["flags"] = flags
    checks["version_ok"] = version in RSC8_VERSIONS
    checks["size_matches_body_plus_12"] = body_size + 12 == total_size
    checks["flags_ok"] = flags == RSC8_FLAGS

    virt = mv[RSC8_VIRTUAL_BASE:]
    vlen = len(virt)

    # Root crClipDictionary at virt+0x00 .. virt+0xAF
    dict_name = decode_sso_string(virt, 0x38)
    out["dict_name"] = dict_name
    checks["dict_name_ok"] = dict_name["ok"]

    out["clip_map_buckets"] = _u32(virt, 0x50)
    out["clip_map_count"] = _u32(virt, 0x58)
    stream_hash = _u32(virt, 0x68)
    dict_flags = _u32(virt, 0x6C)
    out["stream_hash"] = stream_hash
    out["dict_flags"] = dict_flags
    if stream_id is not None:
        checks["stream_id_matches"] = (stream_hash & 0x1FFFFFFF) == stream_id

    out["hash_map_buckets"] = _u32(virt, 0x88)
    out["hash_map_count"] = _u32(virt, 0x90)

    props_ptr = resolve_vptr(virt, 0xA0)
    props_size = _u32(virt, 0xA8)
    out["props_ptr"] = props_ptr
    out["props_size"] = props_size

    # anim_dict_array at virt+0x70 (16-byte entries: u64 anim_dict_ptr, u32 anim_dict_hash, u32 flags)
    arr70_ptr = resolve_vptr(virt, 0x70)
    arr70_cnt, arr70_cap = struct.unpack_from("<HH", virt, 0x78)
    if arr70_ptr and 1 <= arr70_cnt <= arr70_cap and arr70_ptr + arr70_cnt * 0x10 <= vlen:
        anim_dict_ptr = resolve_vptr(virt, arr70_ptr)
        checks["anim_dict_entry_ok"] = bool(anim_dict_ptr)

    # crProperties at props_ptr
    if deep:
        props = decode_properties(virt, props_ptr, props_size)
    else:
        props_ok = bool(props_ptr and props_ptr > 0 and props_size >= 0x20 and props_ptr + props_size <= vlen)
        p_cnt = _u32(virt, props_ptr + 0x18) if props_ok else 0
        p_cap = _u32(virt, props_ptr + 0x1C) if props_ok else 0
        if props_ok and (p_cnt > MAX_PROP_ENTRIES or 0x70 + p_cnt * 0x48 > props_size):
            props_ok = False
        props = {
            "in_bounds": props_ok,
            "clip_ptr": resolve_vptr(virt, props_ptr + 0x10) if props_ok else None,
            "count": p_cnt,
            "capacity": p_cap,
            "attributes": [],
            "fps": None,
            "lead_in": None,
            "attrs_tiled_to_end": False,
        }
    out["properties"] = props
    checks["props_in_bounds"] = props["in_bounds"]

    clip_ptr = props["clip_ptr"]
    if clip_ptr and clip_ptr > 0 and clip_ptr + 0xA0 <= vlen:
        checks["clip_in_bounds"] = True
        clip_name = decode_sso_string(virt, clip_ptr + 0x28)
        clip_hash = _u32(virt, clip_ptr + 0x58)
        anim_desc_ptr = resolve_vptr(virt, clip_ptr + 0x68)
        duration = _f32(virt, clip_ptr + 0x94)
        rate = _f32(virt, clip_ptr + 0x98)
        out["clip"] = {
            "voff": clip_ptr,
            "name": clip_name,
            "hash": clip_hash,
            "anim_desc_ptr": anim_desc_ptr,
            "duration": round(duration, 6) if math.isfinite(duration) else None,
            "rate": round(rate, 6) if math.isfinite(rate) else None,
        }
        checks["clip_name_ok"] = clip_name["ok"]
        checks["clip_hash_matches_root"] = clip_hash == stream_hash

        # crAnimation descriptor at anim_desc_ptr
        if anim_desc_ptr and anim_desc_ptr > 0 and anim_desc_ptr + 0x70 <= vlen:
            checks["anim_desc_in_bounds"] = True
            anim_dict_ref = resolve_vptr(virt, anim_desc_ptr + 0x18)
            anim_name_ptr = resolve_vptr(virt, anim_desc_ptr + 0x30)
            anim_name_ok, anim_name_len = _cstr_info(virt, anim_name_ptr) if anim_name_ptr else (False, 0)
            max_blk_desc = _u32(virt, anim_desc_ptr + 0x48)
            blk_arr_ptr = resolve_vptr(virt, anim_desc_ptr + 0x50)
            blk_cnt, blk_cap = struct.unpack_from("<HH", virt, anim_desc_ptr + 0x58)
            tr_ptr = resolve_vptr(virt, anim_desc_ptr + 0x60)
            tr_cnt, tr_cap = struct.unpack_from("<HH", virt, anim_desc_ptr + 0x68)

            out["anim_desc"] = {
                "voff": anim_desc_ptr,
                "anim_dict_ptr": anim_dict_ref,
                "anim_name_len": anim_name_len,
                "anim_name_ok": anim_name_ok,
                "max_block_size": max_blk_desc,
                "block_count": blk_cnt,
                "block_capacity": blk_cap,
                "track_ptr": tr_ptr,
                "track_count": tr_cnt,
                "track_capacity": tr_cap,
            }

            # Walk all packed crAnimation blocks referenced by anim_desc+0x50 (`atArray<crAnimBlock*>`).
            # Multi-block clips split long facial animations across adjacent blocks sharing the boundary
            # frame, so total_frames = sum(block.frame_count) - (block_count - 1).
            if (
                blk_arr_ptr
                and 1 <= blk_cnt <= blk_cap
                and blk_cnt <= MAX_ANIM_BLOCKS
                and blk_arr_ptr + blk_cnt * 8 <= vlen
            ):
                blocks = []
                seen_bo = set()
                all_in_bounds = True
                all_payload_ok = True
                all_chunk_ok = True
                all_v2 = True
                all_tag11 = True
                for bi in range(blk_cnt):
                    bo = resolve_vptr(virt, blk_arr_ptr + bi * 8)
                    if bo is None or bo < 0xB0 or bo + 32 > vlen or bo in seen_bo:
                        all_in_bounds = False
                        break
                    seen_bo.add(bo)
                    (
                        anim_sig,
                        payload_sz,
                        zero_b8,
                        chunk_off,
                        anim_sz,
                        fmt_ver,
                        b_frames,
                        extra_flags,
                        sub_count,
                        track_tag,
                    ) = struct.unpack_from("<IIIIIHHIHH", virt, bo)
                    if anim_sz < 32 or bo + anim_sz > vlen:
                        all_in_bounds = False
                    if payload_sz + 32 != anim_sz:
                        all_payload_ok = False
                    if chunk_off > payload_sz:
                        all_chunk_ok = False
                    if fmt_ver != 2:
                        all_v2 = False
                    if track_tag != 11:
                        all_tag11 = False
                    blocks.append({
                        "voff": bo,
                        "signature": anim_sig,
                        "payload_size": payload_sz,
                        "reserved": zero_b8,
                        "chunk_data_offset": chunk_off,
                        "anim_size": anim_sz,
                        "format_version": fmt_ver,
                        "frame_count": b_frames,
                        "extra_flags": extra_flags,
                        "sub_count": sub_count,
                        "track_format_tag": track_tag,
                    })

                if len(blocks) == blk_cnt:
                    total_anim_sz = sum(b["anim_size"] for b in blocks)
                    max_blk_sz = max(b["anim_size"] for b in blocks)
                    total_frames = sum(b["frame_count"] for b in blocks) - (len(blocks) - 1)
                    out["anim"] = {
                        "block_count": len(blocks),
                        "anim_size": total_anim_sz,
                        "max_block_size": max_blk_sz,
                        "frame_count": total_frames,
                        "blocks": blocks,
                    }
                    checks["anim_blocks_in_bounds"] = all_in_bounds
                    checks["anim_payload_plus_32_eq_size"] = all_payload_ok
                    checks["anim_chunk_off_in_bounds"] = all_chunk_ok
                    checks["anim_format_v2"] = all_v2
                    checks["anim_track_tag_11"] = all_tag11
                    checks["anim_max_block_size_matches_desc"] = max_blk_sz == max_blk_desc
                    if deep:
                        raw_fps = props.get("fps")
                        fps = 60 if raw_fps is None else raw_fps
                        expected_dur = (total_frames - 1) / float(fps) if total_frames > 0 and fps > 0 else 0.0
                        checks["clip_duration_matches_frames"] = (
                            math.isfinite(duration)
                            and fps > 0
                            and all(b["frame_count"] > 0 for b in blocks)
                            and total_frames > 0
                            and abs(duration - expected_dur) < 1e-4
                        )

            # Sorted track directory: count * 12 bytes
            if tr_cnt == 0:
                checks["tracks_in_bounds"] = tr_ptr == 0 or (tr_ptr is not None and tr_ptr <= vlen)
                if deep:
                    checks["tracks_sorted"] = True
                    checks["tracks_schema_ok"] = True
                out["tracks"] = {"count": 0, "type_counts": {}}
            elif (
                tr_ptr
                and tr_ptr > 0
                and tr_cnt <= tr_cap
                and tr_cnt <= MAX_TRACK_ENTRIES
                and tr_ptr + tr_cnt * 12 <= vlen
            ):
                checks["tracks_in_bounds"] = True
                if deep:
                    prev_h = -1
                    is_sorted = True
                    schema_ok = True
                    type_counts = {}
                    for ti in range(tr_cnt):
                        th, schema_h, ttype = struct.unpack_from("<III", virt, tr_ptr + ti * 12)
                        if th < prev_h:
                            is_sorted = False
                        prev_h = th
                        if schema_h != TRACK_SCHEMA_HASH:
                            schema_ok = False
                        type_counts[ttype] = type_counts.get(ttype, 0) + 1
                    checks["tracks_sorted"] = is_sorted
                    checks["tracks_schema_ok"] = schema_ok
                    out["tracks"] = {"count": tr_cnt, "type_counts": type_counts}
                else:
                    out["tracks"] = {"count": tr_cnt, "type_counts": {}}

    core_checks = [
        "min_size_ok",
        "magic_ok",
        "version_ok",
        "size_matches_body_plus_12",
        "flags_ok",
        "aligned_16",
        "dict_name_ok",
        "anim_dict_entry_ok",
        "props_in_bounds",
        "clip_in_bounds",
        "clip_name_ok",
        "clip_hash_matches_root",
        "anim_desc_in_bounds",
        "anim_blocks_in_bounds",
        "anim_payload_plus_32_eq_size",
        "anim_chunk_off_in_bounds",
        "anim_format_v2",
        "anim_track_tag_11",
        "anim_max_block_size_matches_desc",
        "tracks_in_bounds",
        "no_embedded_audio_container",
    ]
    if deep:
        core_checks.extend(("clip_duration_matches_frames", "tracks_sorted", "tracks_schema_ok"))
    else:
        checks.pop("clip_duration_matches_frames", None)
        checks.pop("tracks_sorted", None)
        checks.pop("tracks_schema_ok", None)
    out["valid"] = all(checks[k] for k in core_checks)
    if curves:
        # Keyframes are decoded only from a clip whose structure checks out, so a block whose sizes disagree
        # or that runs into another block or the track directory is counted as an error, never decoded.
        if not out["valid"] or not out["anim"] or not out["anim_desc"]:
            out["curves"] = {"decoded": False, "error": "structural checks failed"}
        elif _extents_overlap(out["anim"], out["anim_desc"]):
            out["curves"] = {"decoded": False, "error": "block or track extents overlap"}
        else:
            out["curves"] = facial_curve_summary(virt, out["anim"], out["anim_desc"])
    return out


def parse_adat_header_bytes(hb, bank_offset, file_size):
    """Parse one little-endian AWC (`ADAT`) bank header from bytes `hb`.

    Returns a bank descriptor dict or `None` if malformed or out of bounds. A zero-stream bank
    (`count == 0`) is valid when the flag-selected tables end exactly at the declared header size;
    the package's `audio.rpf` carries two such 20-byte banks that the encrypted TOC lists as
    `.awc` entries.
    """
    if bank_offset < 0 or len(hb) < 16 or hb[:4] != ADAT_MAGIC:
        return None
    flags, count, hdr_sz = struct.unpack_from("<III", hb, 4)
    if (
        count > MAX_BANK_STREAMS
        or hdr_sz < 16
        or hdr_sz > MAX_HEADER_BYTES
        or hdr_sz > len(hb)
        or bank_offset + hdr_sz > file_size
    ):
        return None
    pos = 16
    if flags & 0x00400000:
        pos += 4
    if flags & 0x00010000:
        pos += count * 2
    if flags & 0x00200000:
        pos += (count + 1) * 4
    if count == 0:
        # No stream ids and no tag rows follow, so the tables must fill the header exactly.
        if pos != hdr_sz:
            return None
    elif pos + count * 4 > hdr_sz:
        return None

    streams = []
    total_tags = 0
    for _ in range(count):
        w = _u32(hb, pos)
        pos += 4
        tc = w >> 29
        sid = w & 0x1FFFFFFF
        total_tags += tc
        streams.append((tc, sid))
    if pos + total_tags * 8 > hdr_sz:
        return None

    max_end = hdr_sz
    entries = []
    for tc, sid in streams:
        tags = []
        for _ in range(tc):
            q = _u64(hb, pos)
            pos += 8
            tt = (q >> 56) & 0xFF
            tsz = (q >> 28) & 0x0FFFFFFF
            toff = q & 0x0FFFFFFF
            if bank_offset + toff + tsz > file_size:
                return None
            if toff + tsz > max_end:
                max_end = toff + tsz
            tags.append((tt, toff, tsz))
        entries.append((sid, tags))

    return {
        "offset": bank_offset,
        "flags": flags,
        "count": count,
        "header_size": hdr_sz,
        "end": bank_offset + max_end,
        "entries": entries,
    }


def parse_adat_at_file(fh, bank_offset, file_size):
    """Read and parse one `ADAT` bank header at `bank_offset` in open binary file `fh`.

    Zero-stream banks are validated by :func:`parse_adat_header_bytes`.
    """
    if bank_offset < 0 or bank_offset + 16 > file_size:
        return None
    fh.seek(bank_offset)
    hdr16 = fh.read(16)
    if len(hdr16) < 16 or hdr16[:4] != ADAT_MAGIC:
        return None
    _, count, hdr_sz = struct.unpack_from("<III", hdr16, 4)
    if (
        count > MAX_BANK_STREAMS
        or hdr_sz < 16
        or hdr_sz > MAX_HEADER_BYTES
        or bank_offset + hdr_sz > file_size
    ):
        return None
    fh.seek(bank_offset)
    hb = fh.read(hdr_sz)
    if len(hb) < hdr_sz:
        return None
    return parse_adat_header_bytes(hb, bank_offset, file_size)


GRID20_MAGIC = 12345678
MAX_GRID20_BYTES = 64 * 1024 * 1024
MAX_INDEXRUN_GROUPS = 1 << 20


def decode_grid20_table(blob):
    """Structurally decode the undocumented ``0x5a`` companion table.

    Measured on every one of the 1,185 ``audio.rpf`` streams that carry the tag (ASSET-256, framing
    from @kotorvr's finding in #3232): a u32 ``12345678`` marker, ``N`` 20-byte entries, then an f32
    trailer, so ``size == 8 + 20 * N``::

        +0  u32  position (a sample position at the stream's rate: strictly increasing, below the
                 stream's 0x76 sample count)
        +4  f32  value (3.66..146.8; (position[k+1] - position[k]) * value[k] / rate is within 1% of 1
                 on 365,803 of the 386,048 entry pairs, so it reads as a pitch in Hz)
        +8  u16  p (a multiple of 512 on every measured entry), +10 u16  z (0..16 measured)
        +12 u16  index_lo, +14 u16  index_hi
        +16 u16  s, +18 u16  t
    trailer: f32 (2/3 on 1,182 of the 1,185 streams), returned as ``tail_f32``

    Nothing is guessed about ``p``, ``z``, ``s``, ``t`` or the trailer; the return value carries the
    structural facts and the invariant checks only.
    """
    out = {
        "size_ok": False,
        "magic_ok": None,
        "entries": None,
        "position_first": None,
        "position_last": None,
        "position_increasing": None,
        "tail_f32": None,
        "p_multiple_of_32": None,
    }
    raw = bytes(blob)
    if len(raw) < 8 or len(raw) > MAX_GRID20_BYTES or (len(raw) - 8) % 20:
        return out
    out["size_ok"] = True
    magic, = struct.unpack_from("<I", raw, 0)
    out["magic_ok"] = magic == GRID20_MAGIC
    n = (len(raw) - 8) // 20
    out["entries"] = n
    out["tail_f32"], = struct.unpack_from("<f", raw, len(raw) - 4)
    prev = None
    p_ok = True
    increasing = True
    for i in range(n):
        at = 4 + 20 * i
        position, = struct.unpack_from("<I", raw, at)
        p, = struct.unpack_from("<H", raw, at + 8)
        if p % 32:
            p_ok = False
        if prev is None:
            out["position_first"] = position
        elif position <= prev:
            increasing = False
        prev = position
    out["position_last"] = prev
    out["position_increasing"] = increasing
    out["p_multiple_of_32"] = p_ok
    return out


def decode_index_runs(blob, grid_entries=None):
    """Structurally decode the undocumented ``0xd9`` companion table.

    Measured on all 804 ``audio.rpf`` streams that carry the tag: a u32 group count ``K``, then
    ``K`` groups of ``{u32 2; u32 len; u32 hash; len u32s}``. The values ascend strictly inside a
    group, exactly one distinct ``hash`` appears per stream, and every value is smaller than the
    sibling ``0x5a`` table's entry count, so they index ``0x5a`` entries.
    """
    out = {
        "size_ok": False,
        "group_count_matches": None,
        "marker_ok": None,
        "hash_single": None,
        "hashes": None,
        "groups": None,
        "values": None,
        "indices_ascending": None,
        "index_max": None,
        "indices_within_grid": None,
    }
    raw = bytes(blob)
    if len(raw) < 4 or len(raw) % 4 or len(raw) > MAX_GRID20_BYTES:
        return out
    out["size_ok"] = True
    vals = struct.unpack_from(f"<{len(raw) // 4}I", raw, 0)
    declared, at = vals[0], 1
    groups = 0
    hashes = set()
    values = 0
    index_max = None
    ascending = True
    marker_ok = True
    while at < len(vals):
        if at + 3 > len(vals) or groups >= MAX_INDEXRUN_GROUPS:
            out["size_ok"] = False
            return out
        marker, length, h = vals[at], vals[at + 1], vals[at + 2]
        if marker != 2:
            marker_ok = False
        if length > len(vals) - at - 3:
            out["size_ok"] = False
            return out
        hashes.add(h)
        run = vals[at + 3 : at + 3 + length]
        if any(run[i] >= run[i + 1] for i in range(len(run) - 1)):
            ascending = False
        if run:
            run_max = max(run)
            index_max = run_max if index_max is None else max(index_max, run_max)
        values += length
        at += 3 + length
        groups += 1
    out["group_count_matches"] = groups == declared
    out["marker_ok"] = marker_ok
    out["hash_single"] = len(hashes) == 1
    out["hashes"] = sorted(hashes)
    out["groups"] = groups
    out["values"] = values
    out["indices_ascending"] = ascending
    out["index_max"] = index_max
    if grid_entries is None:
        out["indices_within_grid"] = None
    else:
        out["indices_within_grid"] = True if index_max is None else index_max < grid_entries
    return out


def decode_marker_records(blob, sample_count=None):
    """Structurally decode a ``0xbd`` (``markers``) tag: a list of 16-byte cue records.

    Measured on all 420,168 ``0xbd`` tags of ``audio.rpf`` (781,350 records): the size is always a
    multiple of 16, there is no header, and every record is::

        +0  u32  name    joaat of the cue name (167 distinct values)
        +4  u32  value   an f32 for most names, a second name hash for region cues
        +8  u32  offset  sample position in the stream; never decreases from one record to the next
        +12 u32  extra   0..6 measured

    ``name`` and ``value`` are returned as stored. With ``sample_count`` (the stream's ``0x76`` count),
    ``offsets_within_samples`` says whether every offset is at most that count.
    """
    out = {
        "size_ok": False,
        "records": None,
        "names": None,
        "first_name": None,
        "offsets_ascending": None,
        "offset_max": None,
        "offsets_within_samples": None,
        "extra_max": None,
    }
    raw = bytes(blob)
    if not raw or len(raw) % MARKER_RECORD_BYTES or len(raw) > MAX_MARKER_BYTES:
        return out
    out["size_ok"] = True
    n = len(raw) // MARKER_RECORD_BYTES
    names = {}
    ascending = True
    prev = None
    offset_max = 0
    extra_max = 0
    for i in range(n):
        name, _value, offset, extra = struct.unpack_from("<IIII", raw, MARKER_RECORD_BYTES * i)
        names[name] = names.get(name, 0) + 1
        if prev is not None and offset < prev:
            ascending = False
        prev = offset
        offset_max = max(offset_max, offset)
        extra_max = max(extra_max, extra)
    out["records"] = n
    out["names"] = names
    out["first_name"] = _u32(raw, 0)
    out["offsets_ascending"] = ascending
    out["offset_max"] = offset_max
    out["extra_max"] = extra_max
    if sample_count is not None:
        out["offsets_within_samples"] = offset_max <= sample_count
    return out


def marker_records(blob):
    """The ``(name, value, offset, extra)`` u32 words of every whole 16-byte record of a ``0xbd`` tag."""
    raw = bytes(blob)
    return list(struct.iter_unpack("<IIII", raw[: len(raw) - len(raw) % MARKER_RECORD_BYTES]))


def pair_marker_regions(records):
    """Match the region cues of one ``0xbd`` tag into start/end pairs, record by record (ASSET-320).

    `records` is `marker_records()` output. A region record is one whose ``value`` word is
    `MARKER_VALUE_START` or `MARKER_VALUE_END`, and a name with at least one of them is a region name of
    this tag. Each region name's records are walked in stored order, and a pair is a ``start`` whose
    next record of the same name is an ``end`` at the same or a later offset. Every other record of a
    region name is one exception:

    - ``start_without_end``: the next record of the name is not an ``end``, or there is none;
    - ``end_without_start``: the record of the name before it is not a ``start``, or there is none;
    - ``end_before_start``: an ``end`` right after a ``start`` but at a smaller offset (the pair is refused
      and the ``start`` is counted here, not as ``start_without_end``);
    - ``other_value``: the name also has a record whose value is neither word; it opens and closes nothing.

    So per name ``starts == pairs + start_without_end + end_before_start`` and
    ``ends == pairs + end_without_start + end_before_start``. Two regions of one name therefore never
    nest or overlap by construction; what would be a nested region shows up as a ``start_without_end``
    followed by a ``start`` and an ``end_without_start`` preceded by an ``end``.

    Returns ``{"regions": [...], "exceptions": [...], "names": {name: counts}}`` with a region as
    ``{"name", "start", "end", "start_index", "end_index", "extra_differs"}`` (offsets in samples,
    indices into `records`) and an exception as ``{"kind", "name", "value", "index", "offset",
    "before", "before_offset", "after", "after_offset"}``, where ``before`` and ``after`` say what the
    neighbouring records of the same name are (``"start"``, ``"end"``, ``"other"``, or ``None`` when
    the record is the name's first or last of the tag) and the two offsets are theirs.
    """
    out = {"regions": [], "exceptions": [], "names": {}}
    if not any(value in _REGION_VALUES for _name, value, _offset, _extra in records):
        return out
    by_name = {}
    for index, (name, value, offset, extra) in enumerate(records):
        by_name.setdefault(name, []).append((index, value, offset, extra))
    for name, rows in by_name.items():
        kinds = [_REGION_VALUES.get(value, "other") for _index, value, _offset, _extra in rows]
        if "start" not in kinds and "end" not in kinds:
            continue
        counts = {"starts": 0, "ends": 0, "other_value": 0, "pairs": 0,
                  "start_without_end": 0, "end_without_start": 0, "end_before_start": 0}

        def refuse(kind, k):
            index, value, offset, _extra = rows[k]
            counts[kind] += 1
            out["exceptions"].append({
                "kind": kind, "name": name, "value": value, "index": index, "offset": offset,
                "before": kinds[k - 1] if k else None,
                "before_offset": rows[k - 1][2] if k else None,
                "after": kinds[k + 1] if k + 1 < len(kinds) else None,
                "after_offset": rows[k + 1][2] if k + 1 < len(kinds) else None,
            })

        opened = None  # position in `rows` of a start that no record has followed yet
        for k, (index, _value, offset, extra) in enumerate(rows):
            kind = kinds[k]
            if kind != "end" and opened is not None:
                refuse("start_without_end", opened)
                opened = None
            if kind == "start":
                counts["starts"] += 1
                opened = k
            elif kind == "end":
                counts["ends"] += 1
                if opened is None:
                    refuse("end_without_start", k)
                elif offset < rows[opened][2]:
                    refuse("end_before_start", k)
                else:
                    start_index, _v, start, start_extra = rows[opened]
                    counts["pairs"] += 1
                    out["regions"].append({
                        "name": name, "start": start, "end": offset, "start_index": start_index,
                        "end_index": index, "extra_differs": extra != start_extra,
                    })
                opened = None
            else:
                refuse("other_value", k)
        if opened is not None:
            refuse("start_without_end", opened)
        out["names"][name] = counts
    out["regions"].sort(key=lambda r: r["start_index"])
    out["exceptions"].sort(key=lambda e: e["index"])
    return out


def marker_region_overlaps(regions):
    """How the paired regions of different names in one tag lie against each other.

    Returns ``{(name, other name): [nested, crossing]}`` with the smaller hash first, counting pairs of
    regions that share at least one sample (``[start, end)``, so a zero-length region and two regions
    that only touch share none). ``nested`` is one region inside the other (equal bounds included),
    ``crossing`` is each one starting or ending inside the other. `regions` come from one
    `pair_marker_regions()` call on a tag whose offsets never decrease, where regions of one name
    cannot overlap, so at most one region per name is open at a time.
    """
    out = {}
    active = []
    for region in sorted(regions, key=lambda r: (r["start"], r["start_index"])):
        if region["end"] <= region["start"]:
            continue
        active = [a for a in active if a["end"] > region["start"]]
        for a in active:
            if a["name"] == region["name"]:
                continue
            counts = out.setdefault((min(a["name"], region["name"]), max(a["name"], region["name"])), [0, 0])
            nested = region["end"] <= a["end"] or region["start"] == a["start"]
            counts[0 if nested else 1] += 1
        active.append(region)
    return out


def _new_region_tally():
    """Running totals of pair_marker_regions() over the tags of a scan; marker_region_report() finishes them."""
    return {
        "tags": 0,  # tags with at least one start/end record
        "tags_all_paired": 0,  # ... and no exception
        "tags_with_sample_count": 0,  # ... on a stream that has a 0x76 sample count
        "pairs_per_tag": [],
        "pairs_extra_differs": 0,
        "overlaps": {},  # marker_region_overlaps(), added up
        "names": {},  # name -> the counts of pair_marker_regions(), "tags" and the "lengths" of its pairs
        "exceptions_total": 0,
        "exceptions": [],  # the first MAX_REGION_EXCEPTIONS_LISTED in scan order
    }


def _tally_marker_regions(tally, records, *, bank_offset=0, stream_id=None, sample_count=None):
    """Add one valid ``0xbd`` tag's region pairing to a `_new_region_tally()`."""
    paired = pair_marker_regions(records)
    if not paired["names"]:
        return
    tally["tags"] += 1
    tally["tags_all_paired"] += not paired["exceptions"]
    tally["tags_with_sample_count"] += sample_count is not None
    tally["pairs_per_tag"].append(len(paired["regions"]))
    for name, counts in paired["names"].items():
        into = tally["names"].setdefault(name, {"tags": 0, "lengths": []})
        into["tags"] += 1
        for key, count in counts.items():
            into[key] = into.get(key, 0) + count
    for region in paired["regions"]:
        tally["names"][region["name"]]["lengths"].append(region["end"] - region["start"])
        tally["pairs_extra_differs"] += region["extra_differs"]
    _add_region_overlaps(tally["overlaps"], marker_region_overlaps(paired["regions"]))
    tally["exceptions_total"] += len(paired["exceptions"])
    for exc in paired["exceptions"]:
        if len(tally["exceptions"]) >= MAX_REGION_EXCEPTIONS_LISTED:
            break
        tally["exceptions"].append({
            "kind": exc["kind"],
            "name": f"0x{exc['name']:08x}",
            "value": f"0x{exc['value']:08x}",
            "before": exc["before"],
            "before_offset": exc["before_offset"],
            "after": exc["after"],
            "after_offset": exc["after_offset"],
            "record": exc["index"],
            "tag_records": len(records),
            "offset": exc["offset"],
            "tag_last_offset": records[-1][2],
            "stream_samples": sample_count,
            "bank_offset": bank_offset,
            "stream_id": None if stream_id is None else f"0x{stream_id:08x}",
        })


def _add_region_overlaps(dst, src):
    for names, (nested, crossing) in src.items():
        into = dst.setdefault(names, [0, 0])
        into[0] += nested
        into[1] += crossing


def _merge_region_tally(dst, src):
    for key, value in src.items():
        if key == "overlaps":
            _add_region_overlaps(dst[key], value)
        elif key == "names":
            for name, counts in value.items():
                into = dst["names"].setdefault(name, {"tags": 0, "lengths": []})
                for field, count in counts.items():
                    if field == "lengths":
                        into["lengths"].extend(count)
                    else:
                        into[field] = into.get(field, 0) + count
        elif key == "exceptions":
            dst[key].extend(value[: MAX_REGION_EXCEPTIONS_LISTED - len(dst[key])])
        elif isinstance(value, list):
            dst[key].extend(value)
        else:
            dst[key] += value


def _min_median_max(values):
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def marker_region_report(tally):
    """The `marker_regions` block of a scan summary, from a `_new_region_tally()`: counts, sample
    offsets and 32-bit hashes only.

    ``names`` maps each region name (hex) to its ``starts``, ``ends``, ``pairs``, the four exception
    counts of pair_marker_regions() and the ``length_samples`` (min/median/max) of its pairs.
    ``overlaps_between_names`` counts pairs of paired regions of different names in one tag that share a
    sample (marker_region_overlaps()), in all and per pair of names. ``exceptions`` lists the first
    `MAX_REGION_EXCEPTIONS_LISTED` in scan order with where each one sits: its record index among the
    tag's ``tag_records``, its offset, the tag's last offset, the stream's ``0x76`` sample count when it
    has one, and what and where the neighbouring records of the same name are (``None``: it is the
    name's first or last record of the tag). ``exceptions_total`` counts them all.
    """
    names = {}
    for name, counts in sorted(tally["names"].items(), key=lambda kv: (-kv[1]["starts"] - kv[1]["ends"], kv[0])):
        row = {key: value for key, value in counts.items() if key != "lengths"}
        row["length_samples"] = _min_median_max(counts["lengths"])
        names[f"0x{name:08x}"] = row
    return {
        "tags": tally["tags"],
        "tags_all_paired": tally["tags_all_paired"],
        "tags_with_sample_count": tally["tags_with_sample_count"],
        "starts": sum(c["starts"] for c in tally["names"].values()),
        "ends": sum(c["ends"] for c in tally["names"].values()),
        "pairs": sum(c["pairs"] for c in tally["names"].values()),
        "pairs_per_tag": _min_median_max(tally["pairs_per_tag"]),
        "pairs_extra_differs": tally["pairs_extra_differs"],
        "overlaps_between_names": {
            "nested": sum(n for n, _c in tally["overlaps"].values()),
            "crossing": sum(c for _n, c in tally["overlaps"].values()),
            "by_names": {
                f"0x{a:08x}+0x{b:08x}": {"nested": n, "crossing": c}
                for (a, b), (n, c) in sorted(tally["overlaps"].items())
            },
        },
        "names": names,
        "exceptions_total": tally["exceptions_total"],
        "exceptions": list(tally["exceptions"]),
    }


def _decode_companion_tags(read_slice, bank_offset, tag_map):
    """Decode companion speech tags (`0x76`, `0x55`, `0x36`, `0x5a`, `0xa3`, `0xb2`, `0xbd`, `0xd9`) on a stream."""
    comp = {
        "has_audio_data": TAG_DATA in tag_map,
        "audio_data_bytes": tag_map[TAG_DATA][0][1] if TAG_DATA in tag_map else 0,
        "audio_samples": None,
        "audio_sample_rate": None,
        "audio_codec": None,
        "audio_channels": None,
        "audio_duration": None,
        "peak_entries": tag_map[TAG_PEAK][0][1] // 2 if TAG_PEAK in tag_map else 0,
        "seek_entries": tag_map[TAG_SEEKTABLE][0][1] // 4 if TAG_SEEKTABLE in tag_map else 0,
        "loudness_entries": tag_map[TAG_LOUDNESS][0][1] // 8 if TAG_LOUDNESS in tag_map else 0,
        "markers": None,
    }
    if TAG_FORMAT in tag_map:
        fo, fs = tag_map[TAG_FORMAT][0]
        if fs >= 30:
            fb = read_slice(bank_offset + fo, min(fs, 32))
            if len(fb) >= 30:
                samples = _u32(fb, 0)
                sr = _u16(fb, 8)
                codec = fb[28]
                ch = fb[29]
                comp["audio_samples"] = samples
                comp["audio_sample_rate"] = sr
                comp["audio_codec"] = codec
                comp["audio_channels"] = ch
                if sr > 0:
                    comp["audio_duration"] = round(samples / float(sr), 6)
    if TAG_MARKER in tag_map:
        mo, ms = tag_map[TAG_MARKER][0]
        if 0 < ms <= MAX_MARKER_BYTES:
            comp["markers"] = decode_marker_records(read_slice(bank_offset + mo, ms), comp["audio_samples"])
    grid = None
    if TAG_GRID20 in tag_map:
        go, gs = tag_map[TAG_GRID20][0]
        if 0 < gs <= MAX_GRID20_BYTES:
            grid = decode_grid20_table(read_slice(bank_offset + go, gs))
            comp["grid20"] = grid
    if TAG_INDEXRUN in tag_map:
        io, isz = tag_map[TAG_INDEXRUN][0]
        if 0 < isz <= MAX_GRID20_BYTES:
            comp["indexrun"] = decode_index_runs(
                read_slice(bank_offset + io, isz),
                grid["entries"] if grid and grid["size_ok"] else None,
            )
    return comp


def _make_record(bank_offset, tag_offset, stream_id, decoded, comp):
    anim = decoded.get("anim") or {}
    clip = decoded.get("clip") or {}
    props = decoded.get("properties") or {}
    tracks = decoded.get("tracks") or {}
    dname = decoded.get("dict_name") or {}
    return {
        "bank_offset": bank_offset,
        "tag_offset": tag_offset,
        "stream_id": stream_id,
        "tag_size": decoded["total_size"],
        "body_size": decoded["body_size"],
        "version": decoded["version"],
        "flags": decoded["flags"],
        "stream_hash": decoded["stream_hash"],
        "dict_name_storage": dname.get("storage"),
        "dict_name_len": dname.get("length"),
        "anim_block_count": anim.get("block_count"),
        "anim_size": anim.get("anim_size"),
        "frame_count": anim.get("frame_count"),
        "clip_duration": clip.get("duration"),
        "has_audio_data": comp["has_audio_data"] if comp else None,
        "audio_duration": comp["audio_duration"] if comp else None,
        "audio_codec": comp["audio_codec"] if comp else None,
        "audio_channels": comp["audio_channels"] if comp else None,
        "audio_sample_rate": comp["audio_sample_rate"] if comp else None,
        "track_count": tracks.get("count"),
        "track_types": tracks.get("type_counts") or {},
        "prop_count": props.get("count"),
        "curves": decoded.get("curves"),
        "valid": decoded["valid"],
        "checks": decoded["checks"],
    }


def _new_grid_tally():
    return {
        "grid20_streams": 0,
        "grid20_entries": 0,
        "grid20_invalid": 0,
        # Streams with a 0x76 sample count whose last position is not below it (0 over audio.rpf, ASSET-256).
        "grid20_position_past_samples": 0,
        "indexrun_streams": 0,
        "indexrun_groups": 0,
        "indexrun_invalid": 0,
        "indexrun_index_overruns": 0,
        "marker_streams": 0,
        "marker_records": 0,
        "marker_invalid": 0,
        "marker_first_names": {},
        # Region cues matched into start/end pairs (ASSET-320). A running `_new_region_tally()` while banks are
        # read and in decode_awc_bank()'s result; a scan's summary carries marker_region_report() of it.
        "marker_regions": _new_region_tally(),
    }


def _merge_grid_tally(dst, src):
    """Add one bank's companion-table tally to a running summary (counts add, per-name maps merge)."""
    for key, start in _new_grid_tally().items():
        if key == "marker_regions":
            _merge_region_tally(dst[key], src[key])
        elif isinstance(start, dict):
            for name, count in src[key].items():
                dst[key][name] = dst[key].get(name, 0) + count
        else:
            dst[key] += src[key]


def _tally_undocumented_tables(read_slice, bank_base, tmap, tally, stream_id=None):
    """Decode the ``0x5a``/``0xd9`` tables and the ``0xbd`` cue list on one stream into a running tally.

    A stream counts as invalid when one of the layout invariants measured over the whole ``audio.rpf``
    population fails, so the tally doubles as a package-wide validator. The region cues of a valid
    ``0xbd`` tag are matched into start/end pairs (`marker_regions`); `stream_id` only labels the
    exceptions listed there.
    """
    grid = None
    if TAG_GRID20 in tmap:
        go, gs = tmap[TAG_GRID20][0]
        if 0 < gs <= MAX_GRID20_BYTES:
            grid = decode_grid20_table(read_slice(bank_base + go, gs))
        tally["grid20_streams"] += 1
        if grid is None or not (
            grid["size_ok"]
            and grid["magic_ok"]
            and grid["position_increasing"]
            and grid["p_multiple_of_32"]
        ):
            tally["grid20_invalid"] += 1
        elif grid["entries"]:
            tally["grid20_entries"] += grid["entries"]
            if TAG_FORMAT in tmap and grid["position_last"] is not None:
                fo, fs = tmap[TAG_FORMAT][0]
                fb = read_slice(bank_base + fo, 4) if fs >= 4 else b""
                if len(fb) == 4 and grid["position_last"] >= _u32(fb, 0):
                    tally["grid20_position_past_samples"] += 1
    if TAG_INDEXRUN in tmap:
        io, isz = tmap[TAG_INDEXRUN][0]
        runs = None
        if 0 < isz <= MAX_GRID20_BYTES:
            runs = decode_index_runs(
                read_slice(bank_base + io, isz),
                grid["entries"] if grid and grid["size_ok"] else None,
            )
        tally["indexrun_streams"] += 1
        if runs is None or not (
            runs["size_ok"]
            and runs["group_count_matches"]
            and runs["marker_ok"]
            and runs["hash_single"]
            and runs["indices_ascending"]
        ):
            tally["indexrun_invalid"] += 1
        else:
            tally["indexrun_groups"] += runs["groups"]
            if runs["indices_within_grid"] is False:
                tally["indexrun_index_overruns"] += 1
    if TAG_MARKER in tmap:
        mo, ms = tmap[TAG_MARKER][0]
        marks = blob = None
        if 0 < ms <= MAX_MARKER_BYTES:
            blob = read_slice(bank_base + mo, ms)
            marks = decode_marker_records(blob)
        tally["marker_streams"] += 1
        if marks is None or not (marks["size_ok"] and marks["offsets_ascending"]):
            tally["marker_invalid"] += 1
        else:
            tally["marker_records"] += marks["records"]
            key = f"0x{marks['first_name']:08x}"
            firsts = tally["marker_first_names"]
            firsts[key] = firsts.get(key, 0) + 1
            records = marker_records(blob)
            if any(value in _REGION_VALUES for _name, value, _offset, _extra in records):
                samples = None
                if TAG_FORMAT in tmap:
                    fo, fs = tmap[TAG_FORMAT][0]
                    fb = read_slice(bank_base + fo, 4) if fs >= 4 else b""
                    samples = _u32(fb, 0) if len(fb) == 4 else None
                _tally_marker_regions(tally["marker_regions"], records, bank_offset=bank_base,
                                      stream_id=stream_id, sample_count=samples)


def decode_awc_bank(data, bank_offset=0, *, deep=True, curves=False):
    """Decode all `0x5c` (`RSC8` v8001) entries in one in-memory AWC (`ADAT`) bank."""
    bank = parse_adat_header_bytes(data, 0, len(data))
    if bank is None:
        raise ValueError("not a valid little-endian ADAT (AWC) bank")

    def read_slice(off, n):
        return data[off : off + n]

    records = []
    tag_counts = {}
    non_rsc8_0x5c = 0
    tally = _new_grid_tally()
    for sid, tags in bank["entries"]:
        tmap = {}
        for tt, toff, tsz in tags:
            key = f"0x{tt:02x}"
            tag_counts[key] = tag_counts.get(key, 0) + 1
            tmap.setdefault(tt, []).append((toff, tsz))
        _tally_undocumented_tables(read_slice, 0, tmap, tally, sid)
        for tt, toff, tsz in tags:
            if tt == TAG_RSC8_CLIP:
                if tsz < 4 or tsz > MAX_RSC8_BYTES:
                    non_rsc8_0x5c += 1
                    continue
                blob = data[toff : toff + tsz]
                if len(blob) < 4 or blob[:4] != RSC8_MAGIC:
                    non_rsc8_0x5c += 1
                    continue
                dec = decode_rsc8(blob, stream_id=sid, deep=deep, curves=curves)
                comp = _decode_companion_tags(read_slice, 0, tmap)
                records.append(_make_record(bank_offset, bank_offset + toff, sid, dec, comp))

    return {
        "bank_offset": bank_offset,
        "bank_size": bank["end"] - bank["offset"],
        "stream_count": bank["count"],
        "tag_counts": tag_counts,
        "non_rsc8_0x5c": non_rsc8_0x5c,
        "records": records,
        **tally,
    }


def _new_summary(mode, source):
    return {
        "mode": mode,
        "source": source,
        "adat_banks": 0,
        "adat_streams": 0,
        "adat_bytes": 0,
        "adat_banks_past_window": 0,
        "adat_bytes_past_window": 0,
        "window_starts_in_bank": None,
        "adat_tag_counts": {},
        "invalid_files": 0,
        "rsc8_decoded": 0,
        "rsc8_in_adat": 0,
        "rsc8_standalone": 0,
        "rsc8_valid": 0,
        "rsc8_invalid": 0,
        "failed_checks": {},
        "rsc8_non_rsc8_0x5c": 0,
        "rsc8_outside_adat": 0,
        "rsc8_versions": {},
        "rsc8_other_versions": {},
        "rsc8_total_bytes": 0,
        "rsc8_body_bytes": 0,
        "stream_id_matches": 0,
        "stream_id_mismatches": 0,
        "dict_name_storage": {},
        "anim_block_counts": {},
        "track_counts": {},
        "track_types": {},
        "prop_counts": {},
        "audio_codecs": {},
        "same_stream_audio_yes": 0,
        "same_stream_audio_no": 0,
        "min_frames": None,
        "max_frames": None,
        "min_tag_size": None,
        "max_tag_size": None,
        "elapsed_seconds": 0.0,
        **_new_grid_tally(),
        "curve_clips": 0,
        "curve_clips_decoded": 0,
        "curve_errors": {},
        "curve_frames_mismatch": 0,
        "curve_quantize_bits_mismatch": 0,
        "curve_channels": {},
        "curve_components": {},
        "curve_delta2_boundaries": 0,
        "curve_delta2_boundaries_closed": 0,
    }


def _finish_summary(summary, t0):
    """Close a scan's summary: the running region tally becomes its report, and the clock stops."""
    summary["marker_regions"] = marker_region_report(summary["marker_regions"])
    summary["elapsed_seconds"] = round(time.time() - t0, 3)


def _accumulate_record(summary, rec, *, in_adat=True, has_audio=None):
    summary["rsc8_decoded"] += 1
    if in_adat:
        summary["rsc8_in_adat"] += 1
    else:
        summary["rsc8_standalone"] += 1
    if rec["valid"]:
        summary["rsc8_valid"] += 1
    else:
        summary["rsc8_invalid"] += 1
        for ck_name, ck_ok in (rec.get("checks") or {}).items():
            if not ck_ok and ck_name != "stream_id_matches":
                summary["failed_checks"][ck_name] = summary["failed_checks"].get(ck_name, 0) + 1
    ver = rec.get("version")
    if ver is not None:
        vk = f"0x{ver:08x}"
        summary["rsc8_versions"][vk] = summary["rsc8_versions"].get(vk, 0) + 1
    sid_chk = (rec.get("checks") or {}).get("stream_id_matches")
    if sid_chk is True:
        summary["stream_id_matches"] += 1
    elif sid_chk is False:
        summary["stream_id_mismatches"] += 1
    tsz = rec["tag_size"] or 0
    bsz = rec["body_size"] or 0
    summary["rsc8_total_bytes"] += tsz
    summary["rsc8_body_bytes"] += bsz
    summary["min_tag_size"] = tsz if summary["min_tag_size"] is None else min(summary["min_tag_size"], tsz)
    summary["max_tag_size"] = tsz if summary["max_tag_size"] is None else max(summary["max_tag_size"], tsz)

    st = rec.get("dict_name_storage") or "unknown"
    summary["dict_name_storage"][st] = summary["dict_name_storage"].get(st, 0) + 1

    bc = rec.get("anim_block_count")
    if bc is not None:
        bk = str(bc)
        summary["anim_block_counts"][bk] = summary["anim_block_counts"].get(bk, 0) + 1

    fr = rec.get("frame_count")
    if fr is not None:
        summary["min_frames"] = fr if summary["min_frames"] is None else min(summary["min_frames"], fr)
        summary["max_frames"] = fr if summary["max_frames"] is None else max(summary["max_frames"], fr)

    tc = rec.get("track_count")
    if tc is not None:
        k = str(tc)
        summary["track_counts"][k] = summary["track_counts"].get(k, 0) + 1
    for tt_k, tt_v in (rec.get("track_types") or {}).items():
        ks = str(tt_k)
        summary["track_types"][ks] = summary["track_types"].get(ks, 0) + tt_v

    pc = rec.get("prop_count")
    if pc is not None:
        k = str(pc)
        summary["prop_counts"][k] = summary["prop_counts"].get(k, 0) + 1

    if has_audio is True:
        summary["same_stream_audio_yes"] += 1
    elif has_audio is False:
        summary["same_stream_audio_no"] += 1

    if rec.get("audio_codec") is not None:
        ck = f"codec={rec['audio_codec']},ch={rec['audio_channels']},sr={rec['audio_sample_rate']}"
        summary["audio_codecs"][ck] = summary["audio_codecs"].get(ck, 0) + 1

    cv = rec.get("curves")
    if cv is not None:
        summary["curve_clips"] += 1
        if not cv["decoded"]:
            summary["curve_errors"][cv["error"]] = summary["curve_errors"].get(cv["error"], 0) + 1
        else:
            summary["curve_clips_decoded"] += 1
            summary["curve_frames_mismatch"] += not cv["frames_match"]
            summary["curve_quantize_bits_mismatch"] += not cv["quantize_bits_match"]
            summary["curve_delta2_boundaries"] += cv["delta2_boundaries"]
            summary["curve_delta2_boundaries_closed"] += cv["delta2_boundaries_closed"]
            for key in ("channels", "components"):
                into = summary["curve_" + key]
                for name, n in cv[key].items():
                    into[name] = into.get(name, 0) + n


def _bank_covering(fh, start, file_size):
    """The `ADAT` bank that starts before `start` and runs past it, or `None`.

    Looks back from `start` (at most `MAX_AWC_FILE_BYTES`) for the nearest header that parses as a bank; banks
    do not overlap, so only that one can cover `start`.
    """
    floor = max(0, start - MAX_AWC_FILE_BYTES)
    hi = start
    while hi > floor:
        lo = max(floor, hi - SCAN_CHUNK_BYTES)
        fh.seek(lo)
        buf = fh.read(hi - lo + len(ADAT_MAGIC) - 1)   # a magic that starts just before hi; none starts at hi
        idx = buf.rfind(ADAT_MAGIC)
        while idx >= 0:
            bank = parse_adat_at_file(fh, lo + idx, file_size)
            if bank is not None:
                return bank if bank["end"] > start else None
            idx = buf.rfind(ADAT_MAGIC, 0, idx + len(ADAT_MAGIC) - 1)
        hi = lo
    return None


def scan_rpf(path, *, start=0, limit=None, max_banks=None, deep=True, collect_records=True, progress=None,
             curves=False):
    """Scan an RPF9 archive sequentially for `ADAT` banks and `RSC8` v8001 resources.

    With a window (`start`/`limit`), `adat_bytes` counts the window's bytes that belong to a bank, so it never
    exceeds `span` and adjacent windows add up to the whole archive. A bank that starts inside the window is
    decoded whole and counted in `adat_banks`; its bytes past the window are `adat_bytes_past_window`. A bank that
    starts before the window and covers `start` (its offset is `window_starts_in_bank`) is not decoded or counted
    as a bank, and the `RSC8` headers inside it are not reported as outside `ADAT`.
    """
    t0 = time.time()
    sz = os.path.getsize(path)
    if start < 0 or start > sz:
        raise ValueError("start offset is outside the archive")
    scan_end = min(sz, start + limit) if limit is not None else sz

    summary = _new_summary("rpf9", path)
    summary["span"] = scan_end - start
    records = []
    adat_ranges = []

    with open(path, "rb") as f, open(path, "rb") as f2:
        def read_slice(off, n):
            f2.seek(off)
            return f2.read(n)

        if start > 0:
            earlier = _bank_covering(f2, start, sz)
            if earlier is not None:
                summary["window_starts_in_bank"] = earlier["offset"]
                summary["adat_bytes"] += min(earlier["end"], scan_end) - start
                adat_ranges.append((earlier["offset"], earlier["end"]))

        pos = start
        next_prog = start + PROGRESS_BYTES
        while pos < scan_end:
            f.seek(pos)
            buf = f.read(min(SCAN_CHUNK_BYTES + 16, scan_end - pos))
            if not buf:
                break
            chunk_lim = min(len(buf), SCAN_CHUNK_BYTES)

            # 1. Discover valid ADAT banks in [pos, pos + chunk_lim)
            cur = 0
            while cur < chunk_lim:
                if max_banks is not None and summary["adat_banks"] >= max_banks:
                    break
                idx = buf.find(ADAT_MAGIC, cur, chunk_lim + 4)
                if idx < 0 or idx >= chunk_lim:
                    break
                aoff = pos + idx
                if adat_ranges and aoff < adat_ranges[-1][1]:
                    cur = min(chunk_lim, adat_ranges[-1][1] - pos)
                    continue
                bank = parse_adat_at_file(f2, aoff, sz)
                if bank is None:
                    cur = idx + 4
                    continue
                summary["adat_banks"] += 1
                summary["adat_streams"] += bank["count"]
                summary["adat_bytes"] += min(bank["end"], scan_end) - bank["offset"]
                if bank["end"] > scan_end:
                    summary["adat_banks_past_window"] += 1
                    summary["adat_bytes_past_window"] += bank["end"] - scan_end
                adat_ranges.append((bank["offset"], bank["end"]))

                bank_size = bank["end"] - bank["offset"]
                has_5c = any(tt == TAG_RSC8_CLIP for _, tags in bank["entries"] for tt, _, _ in tags)
                bank_bytes = None
                if has_5c:
                    if bank["end"] <= pos + len(buf):
                        bank_bytes = memoryview(buf)[idx : bank["end"] - pos]
                    elif bank_size <= 64 * 1024 * 1024:
                        raw_bank = read_slice(aoff, bank_size)
                        if len(raw_bank) >= bank_size:
                            bank_bytes = memoryview(raw_bank)

                def bank_read_slice(off, n):
                    if bank_bytes is not None:
                        rel = off - aoff
                        return bytes(bank_bytes[rel : rel + n])
                    return read_slice(off, n)

                for sid, tags in bank["entries"]:
                    tmap = {}
                    for tt, toff, tsz in tags:
                        tk = f"0x{tt:02x}"
                        summary["adat_tag_counts"][tk] = summary["adat_tag_counts"].get(tk, 0) + 1
                        tmap.setdefault(tt, []).append((toff, tsz))
                    _tally_undocumented_tables(bank_read_slice, aoff, tmap, summary, sid)
                    if not has_5c:
                        continue
                    for tt, toff, tsz in tags:
                        if tt == TAG_RSC8_CLIP:
                            if tsz < 4 or tsz > MAX_RSC8_BYTES:
                                summary["rsc8_non_rsc8_0x5c"] += 1
                                continue
                            head4 = bank_read_slice(aoff + toff, 4)
                            if head4 != RSC8_MAGIC:
                                summary["rsc8_non_rsc8_0x5c"] += 1
                                continue
                            blob = bank_read_slice(aoff + toff, tsz)
                            dec = decode_rsc8(blob, stream_id=sid, deep=deep, curves=curves)
                            comp = _decode_companion_tags(bank_read_slice, aoff, tmap)
                            rec = _make_record(aoff, aoff + toff, sid, dec, comp)
                            _accumulate_record(summary, rec, in_adat=True, has_audio=comp["has_audio_data"])
                            if collect_records:
                                records.append(rec)

                cur = (bank["end"] - pos) if (bank["end"] - pos < chunk_lim) else chunk_lim

            # 2. Check for any raw RSC8 headers outside discovered ADAT banks
            if max_banks is None:
                cur = 0
                while cur < chunk_lim:
                    idx = buf.find(RSC8_MAGIC, cur, chunk_lim + 4)
                    if idx < 0 or idx >= chunk_lim:
                        break
                    if idx + 8 <= len(buf):
                        ver = _u32(buf, idx + 4)
                        if ver in RSC8_VERSIONS:
                            roff = pos + idx
                            lo, hi = 0, len(adat_ranges) - 1
                            inside = False
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                rs, re = adat_ranges[mid]
                                if roff < rs:
                                    hi = mid - 1
                                elif roff >= re:
                                    lo = mid + 1
                                else:
                                    inside = True
                                    break
                            if not inside:
                                summary["rsc8_outside_adat"] += 1
                        elif ver < 100_000:
                            vk = str(ver)
                            summary["rsc8_other_versions"][vk] = summary["rsc8_other_versions"].get(vk, 0) + 1
                    cur = idx + 4

            pos += SCAN_CHUNK_BYTES
            if max_banks is not None and summary["adat_banks"] >= max_banks:
                break
            if progress and pos >= next_prog:
                progress(pos, scan_end, summary)
                next_prog += PROGRESS_BYTES

    _finish_summary(summary, t0)
    return records, summary


def scan_dir(path, *, max_banks=None, deep=True, collect_records=True, curves=False):
    """Scan a directory of `.awc` banks and/or standalone `.rsc8` files."""
    t0 = time.time()
    summary = _new_summary("directory", path)
    records = []
    files = []
    for root, _, names in os.walk(path):
        for name in sorted(names):
            low = name.lower()
            if low.endswith((".awc", ".rsc8")):
                fp = os.path.join(root, name)
                if os.path.islink(fp) or not os.path.isfile(fp):
                    continue
                files.append(fp)
    files.sort()

    for fp in files:
        if max_banks is not None and summary["adat_banks"] >= max_banks:
            break
        try:
            fsize = os.path.getsize(fp)
            with open(fp, "rb") as f:
                head4 = f.read(4)
            if head4 == ADAT_MAGIC:
                if fsize > MAX_AWC_FILE_BYTES:
                    raise ValueError(f"AWC bank exceeds {MAX_AWC_FILE_BYTES} bytes")
                with open(fp, "rb") as f:
                    raw = f.read()
                bank_res = decode_awc_bank(raw, 0, deep=deep, curves=curves)
                summary["adat_banks"] += 1
                summary["adat_streams"] += bank_res["stream_count"]
                summary["adat_bytes"] += bank_res["bank_size"]
                summary["rsc8_non_rsc8_0x5c"] += bank_res["non_rsc8_0x5c"]
                for tk, tv in bank_res["tag_counts"].items():
                    summary["adat_tag_counts"][tk] = summary["adat_tag_counts"].get(tk, 0) + tv
                _merge_grid_tally(summary, bank_res)
                for rec in bank_res["records"]:
                    _accumulate_record(summary, rec, in_adat=True, has_audio=rec.get("has_audio_data"))
                    if collect_records:
                        records.append(rec)
            elif head4 == RSC8_MAGIC:
                if fsize > MAX_RSC8_BYTES:
                    raise ValueError(f"RSC8 file exceeds {MAX_RSC8_BYTES} bytes")
                with open(fp, "rb") as f:
                    raw = f.read()
                dec = decode_rsc8(raw, deep=deep, curves=curves)
                rec = _make_record(0, 0, None, dec, None)
                _accumulate_record(summary, rec, in_adat=False, has_audio=None)
                if collect_records:
                    records.append(rec)
            else:
                raise ValueError("not a valid little-endian ADAT (AWC) bank or RSC8 resource")
        except (OSError, ValueError) as e:
            summary["invalid_files"] += 1
            print(f"{fp}: {e}", file=sys.stderr)

    _finish_summary(summary, t0)
    return records, summary


def scan_single_file(path, *, start=0, limit=None, max_banks=None, deep=True, collect_records=True, progress=None,
                     curves=False):
    """Inspect a single `.rsc8`, `.awc`, or `.rpf` file."""
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4)
    if head == RSC8_MAGIC and start == 0 and limit is None:
        if fsize > MAX_RSC8_BYTES:
            raise ValueError(f"RSC8 file exceeds {MAX_RSC8_BYTES} bytes")
        t0 = time.time()
        with open(path, "rb") as f:
            raw = f.read()
        dec = decode_rsc8(raw, deep=deep, curves=curves)
        rec = _make_record(0, 0, None, dec, None)
        summary = _new_summary("rsc8", path)
        _accumulate_record(summary, rec, in_adat=False, has_audio=None)
        _finish_summary(summary, t0)
        return ([rec] if collect_records else []), summary
    if head == ADAT_MAGIC and start == 0 and limit is None and fsize <= MAX_AWC_FILE_BYTES:
        t0 = time.time()
        with open(path, "rb") as f:
            raw = f.read()
        bank_res = decode_awc_bank(raw, 0, deep=deep, curves=curves)
        summary = _new_summary("awc", path)
        summary["adat_banks"] = 1
        summary["adat_streams"] = bank_res["stream_count"]
        _merge_grid_tally(summary, bank_res)
        summary["adat_bytes"] = bank_res["bank_size"]
        summary["adat_tag_counts"] = bank_res["tag_counts"]
        summary["rsc8_non_rsc8_0x5c"] = bank_res["non_rsc8_0x5c"]
        for rec in bank_res["records"]:
            _accumulate_record(summary, rec, in_adat=True, has_audio=rec.get("has_audio_data"))
        _finish_summary(summary, t0)
        return (bank_res["records"] if collect_records else []), summary
    return scan_rpf(
        path,
        start=start,
        limit=limit,
        max_banks=max_banks,
        deep=deep,
        collect_records=collect_records,
        progress=progress,
        curves=curves,
    )


def _atomic_write_text(path, write_fn):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=parent or ".",
            prefix=".rsc8_audio.",
            suffix=".part",
            delete=False,
        ) as f:
            tmp = f.name
            write_fn(f)
        os.chmod(tmp, published_mode(path))   # NamedTemporaryFile creates 0600
        os.replace(tmp, path)
    except BaseException:
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def write_report(path, records, summary):
    """Atomically write JSON or CSV report (no game strings or raw payloads)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        def _write_csv(f):
            w = csv.writer(f, lineterminator="\n")
            w.writerow(CSV_COLUMNS)
            for r in records:
                row = []
                for col in CSV_COLUMNS:
                    v = r.get(col)
                    if col in ("bank_offset", "tag_offset") and v is not None:
                        row.append(f"0x{v:x}")
                    elif col in ("stream_id", "stream_hash") and v is not None:
                        row.append(f"0x{v:08x}")
                    elif v is None:
                        row.append("")
                    else:
                        row.append(v)
                w.writerow(row)
        _atomic_write_text(path, _write_csv)
    else:
        def _write_json(f):
            json.dump({"summary": summary, "records": records}, f, indent=1)
        _atomic_write_text(path, _write_json)


def _parse_nonneg_int(text):
    try:
        val = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if val < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return val


def _parse_pos_int(text):
    try:
        val = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if val < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive, .awc bank, .rsc8 file, or directory of .awc/.rsc8 files")
    ap.add_argument("out", nargs="?", help="optional output report path (.json or .csv)")
    ap.add_argument("--start", type=_parse_nonneg_int, default=None, help="start byte offset (single archive only)")
    ap.add_argument("--limit", type=_parse_pos_int, default=None, help="maximum bytes to scan (single archive only)")
    ap.add_argument("--max-banks", type=_parse_pos_int, default=None, help="stop after N ADAT banks")
    ap.add_argument("--no-deep", action="store_true", help="skip per-track and property-attribute walks")
    ap.add_argument("--curves", action="store_true",
                    help="also decode every clip's keyframes and count the outcome (slower; counts only)")
    ap.add_argument("--summary-only", action="store_true", help="do not store per-clip records in memory/JSON")
    ap.add_argument("--quiet", action="store_true", help="suppress progress on stderr; stdout contains the JSON summary")
    args = ap.parse_args(argv)

    if not os.path.exists(args.source):
        print(f"{args.source}: path not found", file=sys.stderr)
        return 2
    if args.out and os.path.isfile(args.source):
        try:
            overwrites_source = os.path.samefile(args.source, args.out)
        except FileNotFoundError:
            overwrites_source = False
        except OSError as e:
            ap.error(f"cannot compare source and output paths: {e}")
        if overwrites_source:
            ap.error("output must not overwrite the source file")
    collect = (not args.summary_only) and (args.out is not None)
    if os.path.isdir(args.source):
        if args.start is not None or args.limit is not None:
            ap.error("--start and --limit apply to a single archive, not a directory")
        try:
            records, summary = scan_dir(
                args.source,
                max_banks=args.max_banks,
                deep=not args.no_deep,
                collect_records=collect,
                curves=args.curves,
            )
        except (OSError, ValueError) as e:
            print(f"{args.source}: {e}", file=sys.stderr)
            return 2
    else:
        def progress(pos, end, s):
            if not args.quiet:
                print(
                    f"[{min(pos, end) / max(1, end):6.1%}] banks={s['adat_banks']} "
                    f"rsc8={s['rsc8_in_adat']} valid={s['rsc8_valid']} outside={s['rsc8_outside_adat']}",
                    file=sys.stderr, flush=True,
                )
        try:
            records, summary = scan_single_file(
                args.source,
                start=args.start or 0,
                limit=args.limit,
                max_banks=args.max_banks,
                deep=not args.no_deep,
                collect_records=collect,
                progress=progress,
                curves=args.curves,
            )
        except (OSError, ValueError) as e:
            print(f"{args.source}: {e}", file=sys.stderr)
            return 2

    if args.out:
        try:
            write_report(args.out, records, summary)
        except OSError as e:
            print(f"{args.out}: {e}", file=sys.stderr)
            return 2

    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
