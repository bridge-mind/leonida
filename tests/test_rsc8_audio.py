"""Synthetic unit tests for tools/rsc8_audio.py (ASSET-56).

    python -I tests/test_rsc8_audio.py

Every byte is synthesized by the test. No game file, game audio, or game asset name is used.
"""
import contextlib
import csv
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rsc8_audio.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rsc8_audio  # noqa: E402


def _align16(n):
    return (n + 15) & ~15


def build_synthetic_rsc8(
    *,
    version=8001,
    stream_hash=0x20112233,
    frames=142,
    block_frames=None,
    tracks_before_anim=False,
    fps=60,
    track_count=90,
    inline_dict_name=False,
    extra_bool_prop=False,
    anim_payload_len=64,
):
    """Build a synthetically valid RSC8 v8001 facial/lip-sync clip matching the measured layout."""
    if block_frames is None:
        block_frames = [frames]
    blk_cnt = len(block_frames)
    block_payloads = [anim_payload_len + bi * 16 for bi in range(blk_cnt)]
    block_sizes = [32 + pl for pl in block_payloads]
    tracks_bytes = track_count * 12

    # Place tracks before anim blocks if requested (matches the 7 reordered real-corpus clips),
    # otherwise place anim blocks starting at virt+0xB0 followed by the sorted track directory.
    block_voffs = []
    cur = 0xB0
    if tracks_before_anim:
        tracks_voff = cur
        cur = _align16(cur + tracks_bytes)
        for bsz in block_sizes:
            block_voffs.append(cur)
            cur = _align16(cur + bsz)
    else:
        for bsz in block_sizes:
            block_voffs.append(cur)
            cur = _align16(cur + bsz)
        tracks_voff = cur
        cur = _align16(cur + tracks_bytes)

    props_voff = cur

    prop_count = 5 if extra_bool_prop else 4
    props_tail_off = 0x70 + prop_count * 0x48
    # Typed attributes: string (0x40), guid (0x20), optional bool (0x0C), fps int (0x0C), lead_in float (0x0C)
    attrs_size = 0x40 + 0x20 + (0x0C if extra_bool_prop else 0) + 0x0C + 0x0C
    props_size = props_tail_off + attrs_size

    clip_voff = _align16(props_voff + props_size)
    anim_dict_voff = clip_voff + 0xA0
    anim_desc_voff = anim_dict_voff + 0x70
    hash_map_voff = anim_desc_voff + 0x70
    clip_map_voff = hash_map_voff + 0x60
    tail_voff = clip_map_voff + 0x60

    # Tail sub-allocations
    dict_str_voff = tail_voff + 0x00          # 0x20 bytes (heap dict name)
    anim_dict_str_voff = tail_voff + 0x20     # 0x20 bytes
    clip_str_voff = tail_voff + 0x40          # 0x30 bytes
    anim_name_voff = tail_voff + 0x70         # 0x20 bytes
    block_map_voff = tail_voff + 0x90         # 0x20 bytes
    arr70_voff = tail_voff + 0xB0             # 0x10 bytes (anim_dict_ptr, hash, 1)
    blk_arr_voff = arr70_voff + 0x10          # blk_cnt * 8 bytes (atArray<crAnimBlock*>)
    vlen = _align16(blk_arr_voff + blk_cnt * 8)

    virt = bytearray(vlen)

    def put_vptr(off, target):
        struct.pack_into("<Q", virt, off, rsc8_audio.RSC8_VPTR_BASE + target)

    # ---- Root crClipDictionary ----
    put_vptr(0x08, block_map_voff)
    if inline_dict_name:
        name_bytes = b"synth_dict_001"  # 14 bytes
        virt[0x38 : 0x38 + len(name_bytes)] = name_bytes
        virt[0x38 + len(name_bytes)] = 0
        virt[0x38 + 15] = 0x80 | len(name_bytes)
    else:
        name_bytes = b"synthetic_bank_001"  # 18 bytes
        virt[dict_str_voff : dict_str_voff + len(name_bytes)] = name_bytes
        virt[dict_str_voff + len(name_bytes)] = 0
        put_vptr(0x38, dict_str_voff)
        struct.pack_into("<HHI", virt, 0x40, len(name_bytes), len(name_bytes) + 1, 1)

    put_vptr(0x48, clip_map_voff)
    struct.pack_into("<I", virt, 0x50, 11)
    struct.pack_into("<II", virt, 0x58, 1, 0x01000000)
    struct.pack_into("<II", virt, 0x68, stream_hash, 2)
    put_vptr(0x70, arr70_voff)
    struct.pack_into("<HH", virt, 0x78, 1, 1)
    put_vptr(0x80, hash_map_voff)
    struct.pack_into("<I", virt, 0x88, 11)
    struct.pack_into("<II", virt, 0x90, 1, 0x01000000)
    put_vptr(0xA0, props_voff)
    struct.pack_into("<I", virt, 0xA8, props_size)

    # ---- Embedded crAnimation packed blocks ----
    for bi, (bo, pl, bsz, bfr) in enumerate(zip(block_voffs, block_payloads, block_sizes, block_frames)):
        struct.pack_into(
            "<IIIIIHHIHH",
            virt,
            bo,
            0x11223344 + bi,   # signature
            pl,                # payload_size == anim_size - 32
            0,                 # reserved
            pl // 2,           # chunk_data_offset
            bsz,               # anim_size
            2,                 # format_version
            bfr,               # frame_count
            4,                 # extra_flags
            14,                # sub_count
            11,                # track_format_tag
        )
        # Deterministic synthetic keyframe bytes (avoiding audio magic sequences)
        for i in range(pl):
            virt[bo + 32 + i] = ((i + bi) * 37 + 13) & 0xFF

    # ---- Sorted Track Directory ----
    for ti in range(track_count):
        th = 0x01000000 + ti * 0x01010101
        ttype = 3 if ti < 10 else 4
        struct.pack_into("<III", virt, tracks_voff + ti * 12, th, rsc8_audio.TRACK_SCHEMA_HASH, ttype)

    # ---- crProperties ----
    put_vptr(props_voff + 0x10, clip_voff)
    struct.pack_into("<II", virt, props_voff + 0x18, prop_count, prop_count)
    # Descriptor owner pointers (minimal synthetic descriptors)
    for pi in range(prop_count):
        put_vptr(props_voff + 0x88 + pi * 0x48, props_voff)
    # Typed attributes at props_voff + props_tail_off
    ap = props_voff + props_tail_off
    struct.pack_into("<II", virt, ap, 0x11110001, rsc8_audio.ATTR_STRING)
    struct.pack_into("<Q", virt, ap + 0x18, 8)
    ap += 0x40
    struct.pack_into("<II", virt, ap, 0x11110002, rsc8_audio.ATTR_GUID)
    ap += 0x20
    if extra_bool_prop:
        struct.pack_into("<III", virt, ap, 0x11110003, rsc8_audio.ATTR_BOOL, 1)
        ap += 0x0C
    struct.pack_into("<III", virt, ap, rsc8_audio.PROP_KEY_FPS, rsc8_audio.ATTR_INT, fps)
    ap += 0x0C
    struct.pack_into("<IIf", virt, ap, rsc8_audio.PROP_KEY_LEAD_IN, rsc8_audio.ATTR_FLOAT, 1.0 / 6.0)
    ap += 0x0C
    assert ap == props_voff + props_size

    # ---- crClipAnimation ----
    struct.pack_into("<I", virt, clip_voff + 0x10, 1)
    put_vptr(clip_voff + 0x18, 0)
    clip_name = b"synthetic_clip_name_01"
    virt[clip_str_voff : clip_str_voff + len(clip_name)] = clip_name
    virt[clip_str_voff + len(clip_name)] = 0
    put_vptr(clip_voff + 0x28, clip_str_voff)
    struct.pack_into("<HHI", virt, clip_voff + 0x30, len(clip_name), len(clip_name) + 1, 1)
    put_vptr(clip_voff + 0x48, props_voff)
    struct.pack_into("<II", virt, clip_voff + 0x58, stream_hash, stream_hash)
    put_vptr(clip_voff + 0x68, anim_desc_voff)
    struct.pack_into("<I", virt, clip_voff + 0x70, stream_hash)
    total_frames = sum(block_frames) - (blk_cnt - 1)
    duration = (total_frames - 1) / float(fps) if total_frames > 0 else 0.0
    struct.pack_into("<ff", virt, clip_voff + 0x94, duration, 1.0)

    # ---- crAnimationDictionary ----
    ad_name = b"synthetic_anim_dict"
    virt[anim_dict_str_voff : anim_dict_str_voff + len(ad_name)] = ad_name
    virt[anim_dict_str_voff + len(ad_name)] = 0
    put_vptr(anim_dict_voff + 0x38, anim_dict_str_voff)
    struct.pack_into("<HHI", virt, anim_dict_voff + 0x40, len(ad_name), len(ad_name) + 1, 1)
    struct.pack_into("<I", virt, anim_dict_voff + 0x68, 0x12345678)

    # ---- crAnimation descriptor ----
    struct.pack_into("<I", virt, anim_desc_voff + 0x10, 2)
    put_vptr(anim_desc_voff + 0x18, anim_dict_voff)
    aname = b"synthetic_anim"
    virt[anim_name_voff : anim_name_voff + len(aname)] = aname
    virt[anim_name_voff + len(aname)] = 0
    put_vptr(anim_desc_voff + 0x30, anim_name_voff)
    struct.pack_into("<I", virt, anim_desc_voff + 0x48, max(block_sizes))
    put_vptr(anim_desc_voff + 0x50, blk_arr_voff)
    struct.pack_into("<HH", virt, anim_desc_voff + 0x58, blk_cnt, blk_cnt)
    if track_count > 0:
        put_vptr(anim_desc_voff + 0x60, tracks_voff)
    struct.pack_into("<HH", virt, anim_desc_voff + 0x68, track_count, track_count)

    # ---- block_map node, arr70, and block pointer array ----
    struct.pack_into("<I", virt, block_map_voff + 8, 1)
    put_vptr(arr70_voff, anim_dict_voff)
    struct.pack_into("<II", virt, arr70_voff + 8, 0x12345678, 1)
    for bi, bo in enumerate(block_voffs):
        put_vptr(blk_arr_voff + bi * 8, bo)

    # ---- Outer 16-byte RSC8 wrapper ----
    total_size = 16 + vlen
    body_size = total_size - 12
    hdr = struct.pack("<4sIII", b"RSC8", version, body_size, 1)
    return hdr + bytes(virt)


def build_synthetic_awc(stream_specs):
    """Build a synthetic little-endian ADAT (AWC) speech bank.

    Each item in `stream_specs` is a dict:
      `{"id": int, "rsc8": bytes, "samples": int, "rate": int, "codec": int, "channels": int}`

    A spec may also carry `"extra_tags": {0x5A: bytes, 0xD9: bytes, ...}`; each payload is
    appended after the fixed companions and gets its own tag row.
    """
    n = len(stream_specs)
    extra_per_stream = [spec.get("extra_tags") or {} for spec in stream_specs]
    tag_total = sum(7 + len(extra) for extra in extra_per_stream)
    # 7 fixed tags per stream (0x36, 0x55, 0x5c, 0x76, 0xa3, 0xb2, 0xbd) plus the extras
    hdr_sz = 16 + 4 * n + 8 * tag_total
    data_buf = bytearray()
    tag_tables = []

    for spec in stream_specs:
        # 0x55 Opus audio placeholder (2048 bytes)
        o55 = hdr_sz + len(data_buf)
        s55 = 2048
        pkt = bytearray(s55)
        struct.pack_into("<HB", pkt, 0, 160, 0xF8)
        data_buf.extend(pkt)

        # 0x5c RSC8 clip
        o5c = hdr_sz + len(data_buf)
        blob = spec["rsc8"]
        data_buf.extend(blob)

        # 0x36 peak (12 bytes)
        o36 = hdr_sz + len(data_buf)
        data_buf.extend(struct.pack("<6H", 0x5000, 0x5000, 0x4000, 0x3000, 0x2000, 0x1000))

        # 0xa3 seek table (12 bytes)
        oa3 = hdr_sz + len(data_buf)
        data_buf.extend(struct.pack("<3I", 0, 11520, 23040))

        # 0xbd marker (16 bytes)
        obd = hdr_sz + len(data_buf)
        data_buf.extend(struct.pack("<IfII", rsc8_audio.MARKER_MAGIC, 2.0, 0, 2))

        # 0x76 format (32 bytes)
        o76 = hdr_sz + len(data_buf)
        fmt = bytearray(32)
        struct.pack_into("<I", fmt, 0, spec.get("samples", 89600))
        struct.pack_into("<H", fmt, 8, spec.get("rate", 48000))
        fmt[28] = spec.get("codec", 0x0B)
        fmt[29] = spec.get("channels", 1)
        data_buf.extend(fmt)

        # 0xb2 loudness (16 bytes = 2 pairs)
        ob2 = hdr_sz + len(data_buf)
        data_buf.extend(struct.pack("<IfIf", 0, -37.5, 4800, -38.0))

        extra_rows = []
        for tt, payload in sorted((spec.get("extra_tags") or {}).items()):
            oxx = hdr_sz + len(data_buf)
            data_buf.extend(payload)
            extra_rows.append((tt, len(payload), oxx))

        tag_tables.append([
            (0x36, 12, o36),
            (0x55, s55, o55),
            (0x5C, len(blob), o5c),
            (0x76, 32, o76),
            (0xA3, 12, oa3),
            (0xB2, 16, ob2),
            (0xBD, 16, obd),
        ] + extra_rows)

    hdr = bytearray()
    hdr.extend(struct.pack("<4sIII", b"ADAT", 0xFF020001, n, hdr_sz))
    for spec, tags in zip(stream_specs, tag_tables):
        hdr.extend(struct.pack("<I", (len(tags) << 29) | (spec["id"] & 0x1FFFFFFF)))
    for tags in tag_tables:
        for tt, tsz, toff in tags:
            hdr.extend(struct.pack("<Q", (tt << 56) | (tsz << 28) | toff))
    assert len(hdr) == hdr_sz
    return bytes(hdr) + bytes(data_buf)


class DecodeRsc8Test(unittest.TestCase):
    def test_valid_heap_sso_clip(self):
        blob = build_synthetic_rsc8(stream_hash=0x20112233, frames=142, inline_dict_name=False)
        res = rsc8_audio.decode_rsc8(blob, stream_id=0x00112233)
        self.assertTrue(res["valid"], res["checks"])
        self.assertEqual(res["version"], 8001)
        self.assertEqual(res["flags"], 1)
        self.assertEqual(res["body_size"] + 12, len(blob))
        self.assertEqual(res["dict_name"]["storage"], "heap")
        self.assertEqual(res["dict_name"]["length"], 18)
        self.assertEqual(res["anim"]["frame_count"], 142)
        self.assertAlmostEqual(res["clip"]["duration"], 141 / 60.0, places=5)
        self.assertEqual(res["properties"]["count"], 4)
        self.assertEqual(res["properties"]["fps"], 60)
        self.assertAlmostEqual(res["properties"]["lead_in"], 1.0 / 6.0, places=5)
        self.assertTrue(res["properties"]["attrs_tiled_to_end"])
        self.assertEqual(res["tracks"]["count"], 90)
        self.assertEqual(res["tracks"]["type_counts"], {3: 10, 4: 80})
        # No raw synthetic string leaked anywhere in the JSON output
        dumped = json.dumps(res)
        self.assertNotIn("synthetic_bank_001", dumped)
        self.assertNotIn("synthetic_clip_name_01", dumped)
        self.assertNotIn("synthetic_anim", dumped)

    def test_valid_inline_sso_and_5_props(self):
        blob = build_synthetic_rsc8(
            stream_hash=0x60445566,
            frames=50,
            inline_dict_name=True,
            extra_bool_prop=True,
        )
        res = rsc8_audio.decode_rsc8(blob, stream_id=0x00445566)
        self.assertTrue(res["valid"], res["checks"])
        self.assertEqual(res["dict_name"]["storage"], "inline")
        self.assertEqual(res["dict_name"]["length"], 14)
        self.assertEqual(res["properties"]["count"], 5)
        self.assertAlmostEqual(res["clip"]["duration"], 49 / 60.0, places=5)
        self.assertNotIn("synth_dict_001", json.dumps(res))

    def test_stream_id_mismatch_detected(self):
        blob = build_synthetic_rsc8(stream_hash=0x20112233)
        res = rsc8_audio.decode_rsc8(blob, stream_id=0x11111111)
        # Structurally valid RSC8 whose internal clip hash differs from the enclosing AWC stream id
        self.assertTrue(res["valid"], res["checks"])
        self.assertFalse(res["checks"]["stream_id_matches"])

    def test_reject_truncated_and_bad_magic(self):
        self.assertFalse(rsc8_audio.decode_rsc8(b"")["valid"])
        self.assertFalse(rsc8_audio.decode_rsc8(b"RSC7" + bytes(0x100))["valid"])
        blob = build_synthetic_rsc8()
        self.assertFalse(rsc8_audio.decode_rsc8(blob[:0x100])["valid"])

    def test_reject_bad_body_size_and_version(self):
        blob = bytearray(build_synthetic_rsc8())
        struct.pack_into("<I", blob, 8, len(blob) - 16)  # off by 4
        res = rsc8_audio.decode_rsc8(bytes(blob))
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["size_matches_body_plus_12"])

        blob2 = bytearray(build_synthetic_rsc8())
        struct.pack_into("<I", blob2, 4, 8002)
        res2 = rsc8_audio.decode_rsc8(bytes(blob2))
        self.assertFalse(res2["valid"])
        self.assertFalse(res2["checks"]["version_ok"])

    def test_reject_unsorted_tracks_and_null_track_ptr_with_nonzero_count(self):
        blob = bytearray(build_synthetic_rsc8(track_count=12))
        dec = rsc8_audio.decode_rsc8(bytes(blob))
        tr_voff = dec["anim_desc"]["track_ptr"]
        # Swap hashes of track 0 and track 1 -> unsorted
        h0 = struct.unpack_from("<I", blob, 0x10 + tr_voff)[0]
        h1 = struct.unpack_from("<I", blob, 0x10 + tr_voff + 12)[0]
        struct.pack_into("<I", blob, 0x10 + tr_voff, h1)
        struct.pack_into("<I", blob, 0x10 + tr_voff + 12, h0)
        res = rsc8_audio.decode_rsc8(bytes(blob))
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["tracks_sorted"])

        # Null track pointer with non-zero count must fail tracks_in_bounds
        blob2 = bytearray(build_synthetic_rsc8(track_count=12))
        desc_voff = dec["anim_desc"]["voff"]
        struct.pack_into("<Q", blob2, 0x10 + desc_voff + 0x60, 0)
        res2 = rsc8_audio.decode_rsc8(bytes(blob2))
        self.assertFalse(res2["valid"])
        self.assertFalse(res2["checks"]["tracks_in_bounds"])

    def test_untrusted_huge_counts_do_not_oom(self):
        blob = bytearray(build_synthetic_rsc8(track_count=12))
        dec = rsc8_audio.decode_rsc8(bytes(blob))
        desc_voff = dec["anim_desc"]["voff"]
        props_voff = dec["props_ptr"]
        # Craft huge track count and huge property count
        struct.pack_into("<HH", blob, 0x10 + desc_voff + 0x68, 0xFFFF, 0xFFFF)
        struct.pack_into("<II", blob, 0x10 + props_voff + 0x18, 0xFFFFFFFF, 0xFFFFFFFF)
        res = rsc8_audio.decode_rsc8(bytes(blob))
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["tracks_in_bounds"])
        self.assertFalse(res["checks"]["props_in_bounds"])

    def test_valid_multi_block_reordered_and_flagged_version(self):
        # Multi-block clip (2 blocks sharing boundary frame: 64 + 37 - 1 = 100 frames -> 99/60 = 1.65s)
        multi = build_synthetic_rsc8(block_frames=[64, 37], anim_payload_len=64)
        res_m = rsc8_audio.decode_rsc8(multi, stream_id=0x00112233)
        self.assertTrue(res_m["valid"], res_m["checks"])
        self.assertEqual(res_m["anim"]["block_count"], 2)
        self.assertEqual(res_m["anim"]["frame_count"], 100)
        self.assertEqual(res_m["anim"]["max_block_size"], res_m["anim_desc"]["max_block_size"])
        self.assertAlmostEqual(res_m["clip"]["duration"], 99 / 60.0, places=5)

        # Reordered sub-allocation: track directory placed at virt+0xB0, anim block after it
        reord = build_synthetic_rsc8(frames=120, tracks_before_anim=True)
        res_r = rsc8_audio.decode_rsc8(reord, stream_id=0x00112233)
        self.assertTrue(res_r["valid"], res_r["checks"])
        self.assertEqual(res_r["anim_desc"]["track_ptr"], 0xB0)
        self.assertGreater(res_r["anim"]["blocks"][0]["voff"], 0xB0)

        # Flagged version 0x01001F41 (bit 24 set, base version 8001)
        flagged = build_synthetic_rsc8(version=0x01001F41, frames=79)
        res_f = rsc8_audio.decode_rsc8(flagged, stream_id=0x00112233)
        self.assertTrue(res_f["valid"], res_f["checks"])
        self.assertEqual(res_f["version"], 0x01001F41)
        self.assertEqual(res_f["version_base"], 8001)
        self.assertEqual(res_f["version_flags"], 0x0100)

        # Non-deep mode omits skipped track and property-dependent duration checks rather than reporting False/True
        res_nd = rsc8_audio.decode_rsc8(multi, deep=False)
        self.assertTrue(res_nd["valid"], res_nd["checks"])
        self.assertNotIn("clip_duration_matches_frames", res_nd["checks"])
        self.assertNotIn("tracks_sorted", res_nd["checks"])
        self.assertNotIn("tracks_schema_ok", res_nd["checks"])

        # Non-60 fps clip (fps=30, 61 frames -> 2.0s) validates in both deep and shallow (--no-deep) modes
        clip_30fps = build_synthetic_rsc8(frames=61, fps=30)
        res_30_deep = rsc8_audio.decode_rsc8(clip_30fps, deep=True)
        res_30_shallow = rsc8_audio.decode_rsc8(clip_30fps, deep=False)
        self.assertTrue(res_30_deep["valid"], res_30_deep["checks"])
        self.assertAlmostEqual(res_30_deep["clip"]["duration"], 2.0, places=5)
        self.assertTrue(res_30_shallow["valid"], res_30_shallow["checks"])
        self.assertNotIn("clip_duration_matches_frames", res_30_shallow["checks"])

    def test_reject_truncated_props_and_zero_fps(self):
        blob = bytearray(build_synthetic_rsc8(frames=60))
        dec = rsc8_audio.decode_rsc8(bytes(blob))
        # Shrink props_size so descriptors cannot fit
        struct.pack_into("<I", blob, 0x10 + 0xA8, 0x40)
        res = rsc8_audio.decode_rsc8(bytes(blob))
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["props_in_bounds"])

        # Explicit FPS=0 property must fail duration match rather than silently defaulting to 60
        blob2 = bytearray(build_synthetic_rsc8(frames=60))
        fps_marker = struct.pack("<II", rsc8_audio.PROP_KEY_FPS, rsc8_audio.ATTR_INT)
        pos = blob2.find(fps_marker)
        self.assertGreater(pos, 0)
        struct.pack_into("<I", blob2, pos + 8, 0)
        res2 = rsc8_audio.decode_rsc8(bytes(blob2))
        self.assertFalse(res2["valid"])
        self.assertFalse(res2["checks"]["clip_duration_matches_frames"])

    def test_embedded_audio_container_flagged(self):
        # 0x10 (wrapper) + 0xB0 (block voff) + 0x20 (block header) = 0xE0 (animation payload)
        blob = bytearray(build_synthetic_rsc8(anim_payload_len=64))
        blob[0xE0:0xEC] = b"RIFF\x24\x00\x00\x00WAVE"
        res = rsc8_audio.decode_rsc8(bytes(blob))
        self.assertFalse(res["valid"])
        self.assertEqual([k for k, v in res["checks"].items() if not v], ["no_embedded_audio_container"])

        blob_ogg = bytearray(build_synthetic_rsc8(anim_payload_len=64))
        blob_ogg[0xE0:0xED] = b"OggS\x00OpusHead"
        res_ogg = rsc8_audio.decode_rsc8(bytes(blob_ogg))
        self.assertFalse(res_ogg["valid"])
        self.assertEqual([k for k, v in res_ogg["checks"].items() if not v], ["no_embedded_audio_container"])

        # Quadratic memory-copy DoS regression: many bogus "ADAT" occurrences must complete linearly
        dos_blob = bytearray(build_synthetic_rsc8(anim_payload_len=40_000))
        dos_blob[0xE0 : 0xE0 + 40_000] = b"ADAT" * 10_000
        res_dos = rsc8_audio.decode_rsc8(bytes(dos_blob))
        self.assertTrue(res_dos["valid"], res_dos["checks"])

    def test_reject_duplicate_block_ptr_and_nan_duration(self):
        multi = bytearray(build_synthetic_rsc8(block_frames=[64, 37], anim_payload_len=64))
        dec = rsc8_audio.decode_rsc8(bytes(multi))
        desc_voff = dec["anim_desc"]["voff"]
        blk_arr_voff = struct.unpack_from("<Q", multi, 0x10 + desc_voff + 0x50)[0] - rsc8_audio.RSC8_VPTR_BASE
        b0_ptr = struct.unpack_from("<Q", multi, 0x10 + blk_arr_voff)[0]
        # Duplicate block 0 pointer into slot 1
        struct.pack_into("<Q", multi, 0x10 + blk_arr_voff + 8, b0_ptr)
        res_dup = rsc8_audio.decode_rsc8(bytes(multi))
        self.assertFalse(res_dup["valid"])
        self.assertFalse(res_dup["checks"]["anim_blocks_in_bounds"])

        # NaN clip duration must fail cleanly with None in JSON-safe output
        nan_blob = bytearray(build_synthetic_rsc8(frames=60))
        dec2 = rsc8_audio.decode_rsc8(bytes(nan_blob))
        clip_voff = dec2["clip"]["voff"]
        struct.pack_into("<f", nan_blob, 0x10 + clip_voff + 0x94, float("nan"))
        res_nan = rsc8_audio.decode_rsc8(bytes(nan_blob))
        self.assertFalse(res_nan["valid"])
        self.assertIsNone(res_nan["clip"]["duration"])
        self.assertFalse(res_nan["checks"]["clip_duration_matches_frames"])


class AwcAndArchiveScanTest(unittest.TestCase):
    def test_decode_awc_bank(self):
        c0 = build_synthetic_rsc8(stream_hash=0x20112233, frames=142, inline_dict_name=False)
        c1 = build_synthetic_rsc8(stream_hash=0x60445566, frames=50, inline_dict_name=True)
        bank = build_synthetic_awc([
            {"id": 0x00112233, "rsc8": c0, "samples": 89600, "rate": 48000, "codec": 0x0B, "channels": 1},
            {"id": 0x00445566, "rsc8": c1, "samples": 19392, "rate": 48000, "codec": 0x0B, "channels": 1},
        ])
        res = rsc8_audio.decode_awc_bank(bank)
        self.assertEqual(res["stream_count"], 2)
        self.assertEqual(len(res["records"]), 2)
        self.assertEqual(res["non_rsc8_0x5c"], 0)
        self.assertTrue(all(r["valid"] for r in res["records"]))
        self.assertEqual(res["records"][0]["audio_codec"], 0x0B)
        self.assertAlmostEqual(res["records"][0]["audio_duration"], 89600 / 48000.0, places=5)

    def test_progress_does_not_interrupt_the_json_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "empty.rpf")
            with open(archive, "wb") as f:
                f.write(bytes(64))
            for quiet in (False, True):
                with self.subTest(quiet=quiet), mock.patch.object(rsc8_audio, "PROGRESS_BYTES", 64), \
                        contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(rsc8_audio.main([archive] + (["--quiet"] if quiet else [])), 0)
                    self.assertEqual(json.loads(out.getvalue())["span"], 64)
                    if quiet:
                        self.assertEqual(err.getvalue(), "")
                    else:
                        self.assertIn("100.0%", err.getvalue())

    def test_scan_rpf_with_and_without_stray_rsc8(self):
        c0 = build_synthetic_rsc8(stream_hash=0x20112233, frames=142)
        c1 = build_synthetic_rsc8(stream_hash=0x80778899, frames=74, inline_dict_name=True)
        bank = build_synthetic_awc([
            {"id": 0x00112233, "rsc8": c0, "samples": 89600},
            {"id": 0x00778899, "rsc8": c1, "samples": 28800},
        ])
        stray = build_synthetic_rsc8(stream_hash=0x11112222, frames=61)

        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            hdr = b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0)
            with open(rpf, "wb") as f:
                f.write(hdr)
                f.write(bytes(0x40))
                f.write(bank)
                f.write(bytes(0x80))
                f.write(stray)

            records, summary = rsc8_audio.scan_rpf(rpf)
            self.assertEqual(summary["adat_banks"], 1)
            self.assertEqual(summary["adat_streams"], 2)
            self.assertEqual(summary["rsc8_in_adat"], 2)
            self.assertEqual(summary["rsc8_valid"], 2)
            self.assertEqual(summary["rsc8_invalid"], 0)
            self.assertEqual(summary["rsc8_outside_adat"], 1)
            self.assertEqual(summary["same_stream_audio_yes"], 2)
            self.assertEqual(len(records), 2)

            # Directory scan + CLI JSON/CSV output
            awc_path = os.path.join(tmp, "speech.awc")
            rsc_path = os.path.join(tmp, "standalone.rsc8")
            with open(awc_path, "wb") as f:
                f.write(bank)
            with open(rsc_path, "wb") as f:
                f.write(stray)

            json_out = os.path.join(tmp, "report.json")
            csv_out = os.path.join(tmp, "report.csv")
            cp = subprocess.run([sys.executable, "-I", TOOL, tmp, json_out, "--quiet"], capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with open(json_out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["summary"]["adat_banks"], 1)
            self.assertEqual(rep["summary"]["invalid_files"], 0)
            self.assertEqual(rep["summary"]["rsc8_in_adat"], 2)
            self.assertEqual(rep["summary"]["rsc8_standalone"], 1)
            self.assertEqual(rep["summary"]["rsc8_valid"], 3)

            # A truncated/malformed .awc in the directory is counted and named on stderr without aborting the scan
            bad_awc = os.path.join(tmp, "truncated.awc")
            with open(bad_awc, "wb") as f:
                f.write(bank[:20])
            cp_bad = subprocess.run([sys.executable, "-I", TOOL, tmp, json_out, "--quiet"], capture_output=True, text=True)
            self.assertEqual(cp_bad.returncode, 0, cp_bad.stderr)
            self.assertIn("truncated.awc", cp_bad.stderr)
            with open(json_out, encoding="utf-8") as f:
                rep_bad = json.load(f)
            self.assertEqual(rep_bad["summary"]["adat_banks"], 1)
            self.assertEqual(rep_bad["summary"]["invalid_files"], 1)
            self.assertEqual(rep_bad["summary"]["rsc8_valid"], 3)

            cp2 = subprocess.run([sys.executable, "-I", TOOL, rpf, csv_out, "--quiet"], capture_output=True, text=True)
            self.assertEqual(cp2.returncode, 0, cp2.stderr)
            with open(csv_out, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["version"], "8001")
            self.assertEqual(rows[0]["valid"], "True")

    def test_cli_rejects_output_aliasing_source(self):
        clip = build_synthetic_rsc8(stream_hash=0x20112233, frames=90)
        original = build_synthetic_awc([{"id": 0x00112233, "rsc8": clip, "samples": 48000}])
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "bank.awc")
            with open(source, "wb") as f:
                f.write(original)
            alias = os.path.join(tmp, ".", "bank.awc")
            result = subprocess.run(
                [sys.executable, "-I", TOOL, source, alias, "--quiet"],
                capture_output=True, text=True,
            )
            with open(source, "rb") as f:
                self.assertEqual(f.read(), original)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("output must not overwrite the source file", result.stderr)

    def test_window_edges_inside_a_bank(self):
        """--start/--limit inside a bank: adat_bytes stays within the span, and RSC8 headers in a bank that
        started before the window are not reported as outside ADAT."""
        c0 = build_synthetic_rsc8(stream_hash=0x20112233, frames=142)
        c1 = build_synthetic_rsc8(stream_hash=0x80778899, frames=74, inline_dict_name=True)
        bank = build_synthetic_awc([
            {"id": 0x00112233, "rsc8": c0, "samples": 89600},
            {"id": 0x00778899, "rsc8": c1, "samples": 28800},
        ])
        stray = build_synthetic_rsc8(stream_hash=0x11112222, frames=61)
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                boff = f.tell()
                f.write(bank)
                f.write(bytes(0x80))
                soff = f.tell()
                f.write(stray)
            _, full = rsc8_audio.scan_rpf(rpf)
            bank_size = full["adat_bytes"]
            self.assertGreater(bank_size, 4 * 256)

            # The window ends halfway through the bank: decoded whole, counted up to the window's end.
            _, w = rsc8_audio.scan_rpf(rpf, start=boff, limit=bank_size // 2)
            self.assertEqual((w["adat_banks"], w["rsc8_in_adat"]), (1, 2))
            self.assertEqual(w["adat_bytes"], w["span"])
            self.assertEqual((w["adat_banks_past_window"], w["adat_bytes_past_window"]),
                             (1, bank_size - bank_size // 2))
            # The rest of the bank, in the next window, completes the archive's ADAT bytes.
            _, rest = rsc8_audio.scan_rpf(rpf, start=boff + bank_size // 2)
            self.assertEqual(w["adat_bytes"] + rest["adat_bytes"], bank_size)
            self.assertEqual(rest["adat_banks"], 0)

            # The window starts inside the bank (looked back over several scan chunks): only the stray clip
            # after the bank is outside ADAT.
            old_chunk = rsc8_audio.SCAN_CHUNK_BYTES
            try:
                rsc8_audio.SCAN_CHUNK_BYTES = 256
                for st in (boff + 16, boff + bank_size - 64):
                    _, w = rsc8_audio.scan_rpf(rpf, start=st)
                    self.assertEqual((w["adat_banks"], w["rsc8_outside_adat"]), (0, 1), st)
                    self.assertEqual(w["window_starts_in_bank"], boff)
                    self.assertEqual(w["adat_bytes"], boff + bank_size - st)
                # Past the bank's end, no bank covers the start.
                _, w = rsc8_audio.scan_rpf(rpf, start=soff - 0x40)
            finally:
                rsc8_audio.SCAN_CHUNK_BYTES = old_chunk
            self.assertEqual((w["rsc8_outside_adat"], w["window_starts_in_bank"]), (1, None))
            self.assertEqual((full["adat_banks_past_window"], full["window_starts_in_bank"]), (0, None))

    def test_cli_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.rpf")
            r = subprocess.run([sys.executable, "-I", TOOL, missing], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("path not found", r.stderr)

            r2 = subprocess.run([sys.executable, "-I", TOOL, tmp, "--limit", "100"], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 2)

            r3 = subprocess.run([sys.executable, "-I", TOOL, tmp, "--start", "-1"], capture_output=True, text=True)
            self.assertEqual(r3.returncode, 2)

            r4 = subprocess.run([sys.executable, "-I", TOOL, tmp, "--limit", "0"], capture_output=True, text=True)
            self.assertEqual(r4.returncode, 2)

            tiny = os.path.join(tmp, "tiny.rpf")
            with open(tiny, "wb") as f:
                f.write(b"9FPR" + bytes(28))
            r5 = subprocess.run([sys.executable, "-I", TOOL, tiny, "--start", "1024"], capture_output=True, text=True)
            self.assertEqual(r5.returncode, 2)

            r6 = subprocess.run([sys.executable, "-I", TOOL, tiny, "--max-banks", "0"], capture_output=True, text=True)
            self.assertEqual(r6.returncode, 2)

    def test_chunk_boundary_straddling_and_summary_only(self):
        c0 = build_synthetic_rsc8(stream_hash=0x20112233, frames=90)
        bank = build_synthetic_awc([{"id": 0x00112233, "rsc8": c0, "samples": 48000}])
        stray = build_synthetic_rsc8(stream_hash=0x11112222, frames=45)

        with tempfile.TemporaryDirectory() as tmp:
            # Place ADAT header straddling a 256-byte chunk boundary (at offset 254)
            rpf = os.path.join(tmp, "straddle.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + bytes(250))
                f.write(bank)
                pad = (256 - (f.tell() % 256) - 2) % 256
                f.write(bytes(pad))
                f.write(stray)

            old_chunk = rsc8_audio.SCAN_CHUNK_BYTES
            try:
                rsc8_audio.SCAN_CHUNK_BYTES = 256
                records, summary = rsc8_audio.scan_rpf(rpf)
            finally:
                rsc8_audio.SCAN_CHUNK_BYTES = old_chunk

            self.assertEqual(summary["adat_banks"], 1)
            self.assertEqual(summary["rsc8_in_adat"], 1)
            self.assertEqual(summary["rsc8_valid"], 1)
            self.assertEqual(summary["rsc8_outside_adat"], 1)
            self.assertEqual(len(records), 1)

            # Verify --summary-only writes empty records for dir, standalone .awc, and standalone .rsc8
            awc_path = os.path.join(tmp, "only.awc")
            rsc_path = os.path.join(tmp, "only.rsc8")
            with open(awc_path, "wb") as f:
                f.write(bank)
            with open(rsc_path, "wb") as f:
                f.write(stray)

            for idx, target in enumerate((tmp, awc_path, rsc_path)):
                out_json = os.path.join(tmp, f"sum_{idx}.json")
                cp = subprocess.run(
                    [sys.executable, "-I", TOOL, target, out_json, "--summary-only", "--quiet"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(cp.returncode, 0, cp.stderr)
                with open(out_json, encoding="utf-8") as f:
                    rep = json.load(f)
                self.assertEqual(rep["records"], [])

# The packed blocks are anim_clips sequences, so the synthetic ones come from that tool's test encoder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_anim_clips import encode_sequence  # noqa: E402


def build_synthetic_curve_clip(sequences, frames, formats, pad=()):
    """A synthetic RSC8 clip whose blocks are real encoded sequences.

    `sequences` are `encode_sequence` outputs (each grown by its `pad` bytes so the builder's block sizes
    fit), `frames` their frame counts and `formats` the value format of each track-directory entry.
    """
    pad = list(pad) or [0] * len(sequences)
    blocks = []
    for seq, extra in zip(sequences, pad):
        seq = bytearray(seq) + bytes(extra)
        struct.pack_into("<I", seq, 0x04, len(seq) - 0x20)
        struct.pack_into("<I", seq, 0x10, len(seq))
        blocks.append(bytes(seq))
    clip = bytearray(build_synthetic_rsc8(
        block_frames=list(frames), track_count=len(formats), anim_payload_len=len(blocks[0]) - 0x20,
    ))
    dec = rsc8_audio.decode_rsc8(bytes(clip))
    base = rsc8_audio.RSC8_VIRTUAL_BASE
    for seq, block in zip(blocks, dec["anim"]["blocks"]):
        assert block["anim_size"] == len(seq), (block["anim_size"], len(seq))
        clip[base + block["voff"]:base + block["voff"] + len(seq)] = seq
    at = base + dec["anim_desc"]["track_ptr"]
    for i, fmt in enumerate(formats):
        struct.pack_into("<III", clip, at + 12 * i, 0x01000000 + i, rsc8_audio.TRACK_SCHEMA_HASH, fmt)
    return bytes(clip)


class FacialCurvesTest(unittest.TestCase):
    NF = 70                                              # more than 64 frames: two type-10 chunks, one boundary
    RAMP = [3 * f + (f * f) // 7 for f in range(NF)]     # track 2, component 0 (type 10)
    STEPS = [(5 * f) % 16 for f in range(NF)]            # track 1 (type 4, 4 bits)
    FORMATS = (4, 4, 3)                                  # two one-value tracks and one two-value track

    def clip(self):
        seq = encode_sequence(self.NF, [
            (2, 0, 0, 0.25),
            (4, 1, 0, (4, 0.5, -1.0, self.STEPS)),
            (10, 2, 0, (0.125, 2.0, self.RAMP)),
            (2, 2, 1, -0.75),
        ])
        return build_synthetic_curve_clip([seq], [self.NF], self.FORMATS)

    def test_track_type_hash_is_facialtranslation(self):
        from anim_index import joaat32
        self.assertEqual(joaat32("facialtranslation"), rsc8_audio.TRACK_SCHEMA_HASH)
        self.assertIn(rsc8_audio.TRACK_SCHEMA_HASH, rsc8_audio.anim_clips.TRACK_TYPES)

    def test_block_decodes_to_the_encoded_values(self):
        clip = self.clip()
        dec = rsc8_audio.decode_rsc8(clip, curves=True)
        self.assertTrue(dec["valid"], dec["checks"])
        self.assertEqual(dec["curves"], {
            "decoded": True,
            "error": None,
            "frames": self.NF,
            "frames_match": True,
            "quantize_bits_match": True,
            "tracks_with_values": 3,
            "channels": {"02_static_float": 2, "04_quantize_float": 1, "10_delta2_float": 1},
            "delta2_boundaries": 1,
            "delta2_boundaries_closed": 1,
            "components": {"4:1": 2, "3:2": 1},
        })
        values, frames, tracks = rsc8_audio.decode_facial_curves(
            memoryview(clip)[rsc8_audio.RSC8_VIRTUAL_BASE:], dec["anim"], dec["anim_desc"])
        self.assertEqual(frames, self.NF)
        self.assertEqual([t[2] for t in tracks], list(self.FORMATS))
        self.assertEqual(values[0], [(0.25,)] * self.NF)
        self.assertEqual(values[1], [(x * 0.5 - 1.0,) for x in self.STEPS])
        self.assertEqual(values[2], [(x * 0.125 + 2.0, -0.75) for x in self.RAMP])

    def test_curves_are_opt_in(self):
        clip = self.clip()
        self.assertIsNone(rsc8_audio.decode_rsc8(clip)["curves"])
        bank = build_synthetic_awc([{"id": 0x00112233, "rsc8": clip, "samples": 56000}])
        res = rsc8_audio.decode_awc_bank(bank)
        self.assertIsNone(res["records"][0]["curves"])

    def test_two_blocks_share_their_boundary_frame(self):
        def seq(lo):
            ints = [(lo + f) % 16 for f in range(33)]
            return encode_sequence(33, [(4, 0, 0, (4, 1.0, 0.0, ints)), (2, 1, 0, 0.5)])

        clip = build_synthetic_curve_clip([seq(0), seq(32)], [33, 33], (4, 4), pad=(0, 16))
        dec = rsc8_audio.decode_rsc8(clip, curves=True)
        self.assertTrue(dec["valid"], dec["checks"])
        self.assertEqual(dec["anim"]["frame_count"], 65)
        self.assertTrue(dec["curves"]["decoded"])
        self.assertEqual(dec["curves"]["frames"], 65)
        self.assertTrue(dec["curves"]["frames_match"])
        values, _frames, _tracks = rsc8_audio.decode_facial_curves(
            memoryview(clip)[rsc8_audio.RSC8_VIRTUAL_BASE:], dec["anim"], dec["anim_desc"])
        self.assertEqual(values[0], [(float(f % 16),) for f in range(65)])

    def test_a_clip_that_fails_its_structure_checks_is_not_decoded(self):
        def seq(lo):
            ints = [(lo + f) % 16 for f in range(33)]
            return encode_sequence(33, [(4, 0, 0, (4, 1.0, 0.0, ints)), (2, 1, 0, 0.5)])

        good = build_synthetic_curve_clip([seq(0), seq(32)], [33, 33], (4, 4), pad=(0, 16))
        dec = rsc8_audio.decode_rsc8(good)
        base = rsc8_audio.RSC8_VIRTUAL_BASE
        first, second = dec["anim"]["blocks"][0]["voff"], dec["anim"]["blocks"][1]["voff"]

        # The first block's size word cut to its header: the size checks fail, so nothing is decoded.
        short = bytearray(good)
        struct.pack_into("<I", short, base + first + 0x10, 32)
        bad = rsc8_audio.decode_rsc8(bytes(short), curves=True)
        self.assertFalse(bad["valid"])
        self.assertEqual(bad["curves"], {"decoded": False, "error": "structural checks failed"})

        # The first block grown four bytes into the second one's header, with its payload size and the
        # descriptor's maximum kept consistent: every structure check passes, the overlap is caught.
        grown = bytearray(good)
        size = second - first + 4
        struct.pack_into("<I", grown, base + first + 0x04, size - 32)
        struct.pack_into("<I", grown, base + first + 0x10, size)
        desc = base + dec["anim_desc"]["voff"]
        struct.pack_into("<I", grown, desc + 0x48, max(size, dec["anim"]["blocks"][1]["anim_size"]))
        overlap = rsc8_audio.decode_rsc8(bytes(grown), curves=True)
        self.assertTrue(overlap["valid"], overlap["checks"])
        self.assertEqual(overlap["curves"], {"decoded": False, "error": "block or track extents overlap"})

        bank = build_synthetic_awc([
            {"id": 0x00112233, "rsc8": good, "samples": 56000},
            {"id": 0x00112233, "rsc8": bytes(short), "samples": 56000},
            {"id": 0x00112233, "rsc8": bytes(grown), "samples": 56000},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            awc = os.path.join(tmp, "bank.awc")
            with open(awc, "wb") as f:
                f.write(bank)
            out_json = os.path.join(tmp, "report.json")
            cp = subprocess.run([sys.executable, "-I", TOOL, awc, out_json, "--curves", "--quiet"],
                                capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with open(out_json, encoding="utf-8") as f:
                summary = json.load(f)["summary"]
        self.assertEqual(summary["curve_clips"], 3)
        self.assertEqual(summary["curve_clips_decoded"], 1)
        self.assertEqual(summary["curve_errors"],
                         {"structural checks failed": 1, "block or track extents overlap": 1})

    def test_curves_decode_the_resource_in_place(self):
        import tracemalloc

        clip = self.clip()
        dec = rsc8_audio.decode_rsc8(clip)
        padded = clip + bytes(4 << 20)                  # the curve data is unchanged; only the buffer grows
        view = memoryview(padded)[rsc8_audio.RSC8_VIRTUAL_BASE:]
        want = rsc8_audio.decode_facial_curves(memoryview(clip)[rsc8_audio.RSC8_VIRTUAL_BASE:],
                                               dec["anim"], dec["anim_desc"])
        tracemalloc.start()
        try:
            got = rsc8_audio.decode_facial_curves(view, dec["anim"], dec["anim_desc"])
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(got, want)
        self.assertLess(peak, 1 << 20)                  # a copy of the 4 MiB resource would exceed this
        # A non-contiguous view is still accepted (it is copied).
        stride = memoryview(bytes(b for c in clip for b in (c, 0)))[::2]
        self.assertEqual(rsc8_audio.decode_facial_curves(stride[rsc8_audio.RSC8_VIRTUAL_BASE:],
                                                         dec["anim"], dec["anim_desc"]), want)

    def test_block_that_does_not_decode_is_counted_not_raised(self):
        clip = bytearray(self.clip())
        dec = rsc8_audio.decode_rsc8(bytes(clip))
        block = rsc8_audio.RSC8_VIRTUAL_BASE + dec["anim"]["blocks"][0]["voff"]
        frame_offset = struct.unpack_from("<I", clip, block + 0x0C)[0]
        struct.pack_into("<I", clip, block + 0x0C, frame_offset - 4)     # definitions no longer end at the frames
        bad = rsc8_audio.decode_rsc8(bytes(clip), curves=True)
        self.assertTrue(bad["valid"], bad["checks"])                     # the structural checks still pass
        self.assertFalse(bad["curves"]["decoded"])
        self.assertTrue(bad["curves"]["error"])

        bank = build_synthetic_awc([
            {"id": 0x00112233, "rsc8": self.clip(), "samples": 56000},
            {"id": 0x00112233, "rsc8": bytes(clip), "samples": 56000},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                f.write(bank)
                f.write(bytes(0x40))
            _records, plain = rsc8_audio.scan_rpf(rpf)
            self.assertEqual(plain["curve_clips"], 0)
            records, summary = rsc8_audio.scan_rpf(rpf, curves=True)
            self.assertEqual(summary["curve_clips"], 2)
            self.assertEqual(summary["curve_clips_decoded"], 1)
            self.assertEqual(sum(summary["curve_errors"].values()), 1)
            self.assertEqual(summary["curve_frames_mismatch"], 0)
            self.assertEqual(summary["curve_quantize_bits_mismatch"], 0)
            self.assertEqual(summary["curve_channels"],
                             {"02_static_float": 2, "04_quantize_float": 1, "10_delta2_float": 1})
            self.assertEqual(summary["curve_components"], {"4:1": 2, "3:2": 1})
            self.assertEqual(summary["curve_delta2_boundaries"], 1)
            self.assertEqual(summary["curve_delta2_boundaries_closed"], 1)
            self.assertEqual([r["curves"]["decoded"] for r in records], [True, False])

            # The CLI flag reaches bank, directory and archive scans, and the report holds counts, not values.
            awc = os.path.join(tmp, "bank.awc")
            with open(awc, "wb") as f:
                f.write(bank)
            for source in (rpf, awc, tmp):
                out_json = os.path.join(tmp, "report.json")
                if os.path.exists(out_json):
                    os.remove(out_json)
                cp = subprocess.run(
                    [sys.executable, "-I", TOOL, source, out_json, "--curves", "--quiet"],
                    capture_output=True, text=True,
                )
                self.assertEqual(cp.returncode, 0, cp.stderr)
                with open(out_json, encoding="utf-8") as f:
                    rep = json.load(f)
                self.assertEqual(rep["summary"]["curve_clips"], 2, source)
                self.assertEqual(rep["summary"]["curve_clips_decoded"], 1, source)
                for rec in rep["records"]:
                    self.assertFalse([k for k, v in rec["curves"].items() if isinstance(v, list)], source)


def build_synthetic_empty_bank(flags=0xFF210001, hdr_sz=20):
    """Build a synthetic zero-stream ADAT bank whose flag-selected tables fill the header exactly.

    With `count == 0` the `0x10000` table contributes no bytes and `0x200000` adds `(count + 1) * 4`
    bytes, so the header is the 16-byte prefix plus 4 bytes when that bit is set (the measured
    20-byte shape), or exactly 16 bytes with no table flags.
    """
    if flags & 0x00400000:
        raise ValueError("synthetic shape does not model the 0x400000 table")
    expected = 16 + (4 if flags & 0x00200000 else 0)
    if hdr_sz != expected:
        raise ValueError(f"tables end at {expected}, not {hdr_sz}")
    return b"ADAT" + struct.pack("<III", flags, 0, hdr_sz) + bytes(hdr_sz - 16)


class ZeroStreamBankTest(unittest.TestCase):
    def test_parses_when_tables_fill_the_header(self):
        bank20 = build_synthetic_empty_bank()
        parsed = rsc8_audio.parse_adat_header_bytes(bank20, 0, len(bank20))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["count"], 0)
        self.assertEqual(parsed["header_size"], 20)
        self.assertEqual(parsed["end"], 20)
        self.assertEqual(parsed["entries"], [])

        bank16 = build_synthetic_empty_bank(flags=0, hdr_sz=16)
        parsed16 = rsc8_audio.parse_adat_header_bytes(bank16, 0, len(bank16))
        self.assertIsNotNone(parsed16)
        self.assertEqual(parsed16["header_size"], 16)
        self.assertEqual(parsed16["end"], 16)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.awc")
            with open(path, "wb") as f:
                f.write(bank20)
            with open(path, "rb") as f:
                from_file = rsc8_audio.parse_adat_at_file(f, 0, os.path.getsize(path))
            self.assertIsNotNone(from_file)
            self.assertEqual(from_file["count"], 0)

    def test_rejects_header_slop_and_truncation(self):
        # Declared size 24 but the flag-selected tables end at 20: unaccounted bytes.
        slop = b"ADAT" + struct.pack("<III", 0xFF210001, 0, 24) + bytes(8)
        self.assertIsNone(rsc8_audio.parse_adat_header_bytes(slop, 0, len(slop)))

        # Header says 20 but the buffer/file is shorter.
        short = build_synthetic_empty_bank()[:18]
        self.assertIsNone(rsc8_audio.parse_adat_header_bytes(short, 0, len(short)))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.awc")
            with open(path, "wb") as f:
                f.write(short)
            with open(path, "rb") as f:
                self.assertIsNone(rsc8_audio.parse_adat_at_file(f, 0, os.path.getsize(path)))

        # A zero count cannot rescue an out-of-bounds header.
        oob = build_synthetic_empty_bank()
        self.assertIsNone(rsc8_audio.parse_adat_header_bytes(oob, 8, len(oob)))

    def test_decode_and_scan_count_empty_banks(self):
        empty = build_synthetic_empty_bank()
        decoded = rsc8_audio.decode_awc_bank(empty)
        self.assertEqual(decoded["stream_count"], 0)
        self.assertEqual(decoded["records"], [])
        self.assertEqual(decoded["bank_size"], 20)

        clip = build_synthetic_rsc8(stream_hash=0x20112233, frames=90)
        speech = build_synthetic_awc([{"id": 0x00112233, "rsc8": clip, "samples": 48000}])
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                f.write(speech)
                f.write(bytes(0x40))
                f.write(empty)
                f.write(bytes(0x40))

            records, summary = rsc8_audio.scan_rpf(rpf)
            self.assertEqual(summary["adat_banks"], 2)
            self.assertEqual(summary["adat_streams"], 1)
            self.assertEqual(summary["rsc8_in_adat"], 1)
            self.assertEqual(summary["rsc8_valid"], 1)
            self.assertEqual(summary["rsc8_outside_adat"], 0)
            self.assertEqual(len(records), 1)

            # A directory scan sees the empty bank as a valid bank with no records, not an
            # invalid file.
            awc_path = os.path.join(tmp, "empty.awc")
            with open(awc_path, "wb") as f:
                f.write(empty)
            out_json = os.path.join(tmp, "report.json")
            cp = subprocess.run(
                [sys.executable, "-I", TOOL, tmp, out_json, "--quiet"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)

def build_minimal_tag_bank(sid, tags):
    """Build a one-stream ADAT bank whose stream carries exactly `tags` ({tag: payload}, max 7)."""
    assert 1 <= len(tags) <= 7
    hdr_sz = 16 + 4 + 8 * len(tags)
    rows = []
    parts = []
    off = hdr_sz
    for tt, payload in sorted(tags.items()):
        rows.append((tt, len(payload), off))
        parts.append(payload)
        off += len(payload)
    hdr = bytearray(struct.pack("<4sIII", b"ADAT", 0xFF020001, 1, hdr_sz))
    hdr += struct.pack("<I", (len(tags) << 29) | (sid & 0x1FFFFFFF))
    for tt, tsz, toff in rows:
        hdr += struct.pack("<Q", (tt << 56) | (tsz << 28) | toff)
    assert len(hdr) == hdr_sz
    return bytes(hdr) + b"".join(parts)


def build_synthetic_grid20(entries, trailer=2 / 3):
    """u32 magic 12345678 + 20-byte entries {u32 position, f32 value, 6 x u16} + f32 trailer (ASSET-256)."""
    out = struct.pack("<I", rsc8_audio.GRID20_MAGIC)
    for position, value, p, z, lo, hi, s, t in entries:
        out += struct.pack("<IfHHHHHH", position, value, p, z, lo, hi, s, t)
    return out + struct.pack("<f", trailer)


def build_synthetic_indexruns(groups, declared=None, hash_=0x4C633D07):
    """u32 group count + groups of {u32 2; u32 len; u32 hash; len ascending u32s}."""
    out = struct.pack("<I", declared if declared is not None else len(groups))
    for marker, run in groups:
        out += struct.pack("<III", marker, len(run), hash_) + b"".join(struct.pack("<I", v) for v in run)
    return out


class UndocumentedTagsTest(unittest.TestCase):
    GRID = build_synthetic_grid20([
        (20, 14.21, 0, 0, 1, 2, 0x39CC, 0x4C15),
        (2894, 14.12, 32, 0, 1, 2, 0x4193, 0x4C03),
        (5181, 14.03, 32, 1, 2, 3, 0x49B5, 0x4B15),
    ])
    RUNS = build_synthetic_indexruns([(2, [1, 5, 9]), (2, [20, 21])])

    def test_grid20_decodes_measured_layout(self):
        g = rsc8_audio.decode_grid20_table(self.GRID)
        self.assertTrue(g["size_ok"])
        self.assertTrue(g["magic_ok"])
        self.assertEqual(g["entries"], 3)
        self.assertEqual(g["position_first"], 20)
        self.assertEqual(g["position_last"], 5181)
        self.assertTrue(g["position_increasing"])
        self.assertAlmostEqual(g["tail_f32"], 2 / 3, places=6)
        self.assertTrue(g["p_multiple_of_32"])
        empty = rsc8_audio.decode_grid20_table(build_synthetic_grid20([]))
        self.assertTrue(empty["size_ok"] and empty["magic_ok"])
        self.assertEqual(empty["entries"], 0)
        self.assertIsNone(empty["position_last"])

    def test_grid20_flags_violations(self):
        bad_magic = struct.pack("<II", 87654321, 0) + self.GRID[8:]
        self.assertFalse(rsc8_audio.decode_grid20_table(bad_magic)["magic_ok"])
        self.assertFalse(rsc8_audio.decode_grid20_table(self.GRID[:-1])["size_ok"])
        self.assertFalse(rsc8_audio.decode_grid20_table(self.GRID[:6])["size_ok"])
        flat = build_synthetic_grid20([
            (500, 1.0, 0, 0, 0, 1, 0, 0),
            (400, 2.0, 32, 0, 0, 1, 0, 0),
        ])
        self.assertFalse(rsc8_audio.decode_grid20_table(flat)["position_increasing"])
        # The trailer is a float after the entries, not a position: its bits never enter the position checks.
        tail = build_synthetic_grid20([
            (500, 1.0, 0, 0, 0, 1, 0, 0),
            (900, 2.0, 32, 0, 0, 1, 0, 0),
        ], trailer=0.5)
        g = rsc8_audio.decode_grid20_table(tail)
        self.assertTrue(g["position_increasing"])
        self.assertEqual(g["position_last"], 900)
        self.assertEqual(g["tail_f32"], 0.5)
        skewed = build_synthetic_grid20([(10, 1.0, 33, 0, 0, 1, 0, 0)])
        self.assertFalse(rsc8_audio.decode_grid20_table(skewed)["p_multiple_of_32"])

    def test_indexrun_group_parse(self):
        r = rsc8_audio.decode_index_runs(self.RUNS, grid_entries=22)
        self.assertTrue(r["size_ok"])
        self.assertTrue(r["group_count_matches"])
        self.assertTrue(r["marker_ok"])
        self.assertTrue(r["hash_single"])
        self.assertEqual(r["hashes"], [0x4C633D07])
        self.assertEqual(r["groups"], 2)
        self.assertEqual(r["values"], 5)
        self.assertEqual(r["index_max"], 21)
        self.assertTrue(r["indices_ascending"])
        self.assertTrue(r["indices_within_grid"])
        self.assertFalse(rsc8_audio.decode_index_runs(self.RUNS, grid_entries=10)["indices_within_grid"])
        self.assertIsNone(rsc8_audio.decode_index_runs(self.RUNS)["indices_within_grid"])

        # Non-ascending runs must still check the true maximum, not only run[-1].
        non_asc = build_synthetic_indexruns([(2, [100, 0])])
        r_na = rsc8_audio.decode_index_runs(non_asc, grid_entries=10)
        self.assertFalse(r_na["indices_ascending"])
        self.assertEqual(r_na["index_max"], 100)
        self.assertFalse(r_na["indices_within_grid"])

        # Empty runs contain no indices and must not report an overrun against grid_entries=0.
        empty = struct.pack("<I", 0)
        r_empty = rsc8_audio.decode_index_runs(empty, grid_entries=0)
        self.assertEqual((r_empty["groups"], r_empty["values"]), (0, 0))
        self.assertIsNone(r_empty["index_max"])
        self.assertTrue(r_empty["indices_within_grid"])

    def test_indexrun_flags_violations(self):
        declared_bad = build_synthetic_indexruns([(2, [1, 2])], declared=3)
        self.assertFalse(rsc8_audio.decode_index_runs(declared_bad)["group_count_matches"])
        marker_bad = build_synthetic_indexruns([(3, [1, 2])])
        self.assertFalse(rsc8_audio.decode_index_runs(marker_bad)["marker_ok"])
        descending = build_synthetic_indexruns([(2, [5, 4])])
        self.assertFalse(rsc8_audio.decode_index_runs(descending)["indices_ascending"])
        multi_hash = self.RUNS[:12] + struct.pack("<I", 0x11223344) + self.RUNS[16:]
        self.assertFalse(rsc8_audio.decode_index_runs(multi_hash)["hash_single"])
        # A run length that overruns the payload must not read past the buffer.
        overrun = struct.pack("<IIIII", 1, 2, 0x4C633D07, 9, 5)
        self.assertFalse(rsc8_audio.decode_index_runs(overrun)["size_ok"])
        self.assertFalse(rsc8_audio.decode_index_runs(self.RUNS[:-2])["size_ok"])

    def test_bank_tally_and_scan_summary(self):
        grid25 = build_synthetic_grid20([
            (1000 + 7 * i, float(i), 32 * i, i % 17, i, i + 1, 0x39CC + i, 0x4C15 - i)
            for i in range(25)
        ])
        bank = build_minimal_tag_bank(0x00112233, {
            0x55: b"\xF8\x00" + bytes(62),
            0x76: struct.pack("<I", 96000) + bytes(4) + struct.pack("<H", 48000) + bytes(18) + b"\x0B\x01",
            0x5A: grid25,
            0xD9: self.RUNS,
        })
        res = rsc8_audio.decode_awc_bank(bank)
        self.assertEqual(res["stream_count"], 1)
        self.assertEqual(res["grid20_streams"], 1)
        self.assertEqual(res["grid20_entries"], 25)
        self.assertEqual(res["grid20_invalid"], 0)
        self.assertEqual(res["grid20_position_past_samples"], 0)
        self.assertEqual(res["indexrun_streams"], 1)
        self.assertEqual(res["indexrun_groups"], 2)
        self.assertEqual(res["indexrun_invalid"], 0)
        self.assertEqual(res["indexrun_index_overruns"], 0)

        # A stream whose 0xd9 run points past its own 0x5a entries is counted as an overrun.
        bad = build_minimal_tag_bank(0x00445566, {
            0x55: b"\xF8\x00" + bytes(62),
            0x5A: self.GRID,
            0xD9: build_synthetic_indexruns([(2, [1, 2, 90])]),
        })
        res_bad = rsc8_audio.decode_awc_bank(bad)
        self.assertEqual(res_bad["indexrun_index_overruns"], 1)
        self.assertEqual(res_bad["grid20_invalid"], 0)

        # Positions that reach the 0x76 sample count are counted (the package has none).
        long_grid = build_synthetic_grid20([
            (50000, 1.0, 0, 0, 0, 1, 0, 0),
            (96000, 2.0, 32, 0, 0, 1, 0, 0),
        ])
        past = build_minimal_tag_bank(0x00778899, {
            0x55: b"\xF8\x00" + bytes(62),
            0x76: struct.pack("<I", 96000) + bytes(4) + struct.pack("<H", 48000) + bytes(18) + b"\x0B\x01",
            0x5A: long_grid,
        })
        self.assertEqual(rsc8_audio.decode_awc_bank(past)["grid20_position_past_samples"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                f.write(bank)
                f.write(bytes(0x40))
            _records, summary = rsc8_audio.scan_rpf(rpf)
            self.assertEqual(summary["grid20_streams"], 1)
            self.assertEqual(summary["grid20_entries"], 25)
            self.assertEqual(summary["grid20_invalid"], 0)
            self.assertEqual(summary["indexrun_groups"], 2)

            awc_path = os.path.join(tmp, "grid.awc")
            with open(awc_path, "wb") as f:
                f.write(bank)
            out_json = os.path.join(tmp, "report.json")
            cp = subprocess.run(
                [sys.executable, "-I", TOOL, awc_path, out_json, "--quiet"],
                capture_output=True, text=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with open(out_json, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["summary"]["grid20_entries"], 25)
            self.assertEqual(rep["summary"]["indexrun_groups"], 2)


def _joaat(name):
    """Jenkins one-at-a-time, the hash the cue names and the tag ids are stored as."""
    h = 0
    for c in name.encode("ascii"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def build_synthetic_markers(records):
    """16-byte cue records: u32 name hash, f32 value, u32 sample offset, u32 extra. No header."""
    return b"".join(struct.pack("<IfII", name, value, offset, extra) for name, value, offset, extra in records)


class MarkerRecordsTest(unittest.TestCase):
    INTRO, BEAT, OUTRO = _joaat("intro"), _joaat("beat"), _joaat("outro")
    MARKS = build_synthetic_markers([
        (INTRO, 0.0, 0, 0),
        (BEAT, 1.0, 24000, 0),
        (BEAT, 2.0, 48000, 0),
        (OUTRO, 0.0, 90000, 3),
    ])

    def test_tag_ids_are_the_low_byte_of_the_chunk_name_hash(self):
        for name, tag in (
            ("markers", rsc8_audio.TAG_MARKER),
            ("data", rsc8_audio.TAG_DATA),
            ("peak", rsc8_audio.TAG_PEAK),
            ("seektable", rsc8_audio.TAG_SEEKTABLE),
            ("streamformat", rsc8_audio.TAG_STREAMED_MUSIC),
        ):
            self.assertEqual(_joaat(name) & 0xFF, tag, name)

    def test_decodes_every_record_not_a_header(self):
        m = rsc8_audio.decode_marker_records(self.MARKS, 96000)
        self.assertTrue(m["size_ok"])
        self.assertEqual(m["records"], 4)
        self.assertEqual(m["names"], {self.INTRO: 1, self.BEAT: 2, self.OUTRO: 1})
        self.assertEqual(m["first_name"], self.INTRO)
        self.assertNotEqual(m["first_name"], rsc8_audio.MARKER_NAME_COMMON)
        self.assertTrue(m["offsets_ascending"])
        self.assertEqual(m["offset_max"], 90000)
        self.assertEqual(m["extra_max"], 3)
        self.assertTrue(m["offsets_within_samples"])
        # Two cues may share one sample offset; that is still in order.
        tied = build_synthetic_markers([(self.BEAT, 1.0, 10, 0), (self.OUTRO, 0.0, 10, 0)])
        self.assertTrue(rsc8_audio.decode_marker_records(tied)["offsets_ascending"])
        self.assertIsNone(rsc8_audio.decode_marker_records(tied)["offsets_within_samples"])

    def test_flags_violations(self):
        self.assertFalse(rsc8_audio.decode_marker_records(b"")["size_ok"])
        self.assertFalse(rsc8_audio.decode_marker_records(self.MARKS[:-1])["size_ok"])
        self.assertFalse(rsc8_audio.decode_marker_records(self.MARKS[:8])["size_ok"])
        swapped = build_synthetic_markers([(self.BEAT, 1.0, 500, 0), (self.INTRO, 0.0, 0, 0)])
        self.assertFalse(rsc8_audio.decode_marker_records(swapped)["offsets_ascending"])
        self.assertFalse(rsc8_audio.decode_marker_records(self.MARKS, 89999)["offsets_within_samples"])

    def test_bank_tally_and_scan_summary(self):
        fmt = struct.pack("<I", 96000) + bytes(4) + struct.pack("<H", 48000) + bytes(18) + b"\x0B\x01"
        music = build_minimal_tag_bank(0x00112233, {0x55: b"\xF8\x00" + bytes(62), 0x76: fmt, 0xBD: self.MARKS})
        speech = build_minimal_tag_bank(0x00445566, {
            0x55: b"\xF8\x00" + bytes(62),
            0xBD: build_synthetic_markers([(rsc8_audio.MARKER_NAME_COMMON, 2.0, 0, 2)]),
        })
        broken = build_minimal_tag_bank(0x00778899, {0x55: b"\xF8\x00" + bytes(62), 0xBD: self.MARKS[:-4]})

        res = rsc8_audio.decode_awc_bank(music)
        self.assertEqual(res["marker_streams"], 1)
        self.assertEqual(res["marker_records"], 4)
        self.assertEqual(res["marker_invalid"], 0)
        self.assertEqual(res["marker_first_names"], {f"0x{self.INTRO:08x}": 1})
        res_bad = rsc8_audio.decode_awc_bank(broken)
        self.assertEqual(res_bad["marker_streams"], 1)
        self.assertEqual(res_bad["marker_invalid"], 1)
        self.assertEqual(res_bad["marker_records"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                for bank in (music, speech, music, broken):
                    f.write(bank)
                    f.write(bytes(0x40))
            _records, summary = rsc8_audio.scan_rpf(rpf)
            self.assertEqual(summary["marker_streams"], 4)
            self.assertEqual(summary["marker_records"], 9)
            self.assertEqual(summary["marker_invalid"], 1)
            self.assertEqual(summary["marker_first_names"], {
                f"0x{self.INTRO:08x}": 2,
                f"0x{rsc8_audio.MARKER_NAME_COMMON:08x}": 1,
            })

            # Directory and single-bank scans merge the per-name map the same way.
            for i, bank in enumerate((music, speech, music)):
                with open(os.path.join(tmp, f"bank{i}.awc"), "wb") as f:
                    f.write(bank)
            os.remove(rpf)
            out_json = os.path.join(tmp, "report.json")
            cp = subprocess.run(
                [sys.executable, "-I", TOOL, tmp, out_json, "--quiet"],
                capture_output=True, text=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with open(out_json, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["summary"]["marker_records"], 9)
            self.assertEqual(rep["summary"]["marker_first_names"][f"0x{self.INTRO:08x}"], 2)
            _records, single = rsc8_audio.scan_single_file(os.path.join(tmp, "bank1.awc"))
            self.assertEqual(single["marker_first_names"], {f"0x{rsc8_audio.MARKER_NAME_COMMON:08x}": 1})

    def test_speech_stream_companion_decodes_its_marker(self):
        clip = build_synthetic_rsc8()
        bank = build_synthetic_awc([{"id": 0x00112233, "rsc8": clip, "samples": 48000}])
        comp = {}
        real = rsc8_audio._make_record

        def spy(bank_offset, tag_offset, stream_id, decoded, companion):
            comp.update(companion or {})
            return real(bank_offset, tag_offset, stream_id, decoded, companion)

        rsc8_audio._make_record = spy
        try:
            rsc8_audio.decode_awc_bank(bank)
        finally:
            rsc8_audio._make_record = real
        self.assertEqual(comp["markers"]["records"], 1)
        self.assertEqual(comp["markers"]["first_name"], rsc8_audio.MARKER_NAME_COMMON)
        self.assertTrue(comp["markers"]["offsets_within_samples"])


def build_synthetic_region_markers(records):
    """16-byte cue records with the value as a u32 word, as a region cue stores it: name, value, offset, extra."""
    return b"".join(struct.pack("<IIII", *record) for record in records)


class MarkerRegionsTest(unittest.TestCase):
    """Region cues matched into start/end pairs record by record (ASSET-320). The names are made up."""
    S, E = _joaat("start"), _joaat("end")
    VERSE, FILL, BEAT = 0x11111111, 0x22222222, _joaat("beat")
    FMT = struct.pack("<I", 96000) + bytes(4) + struct.pack("<H", 48000) + bytes(18) + b"\x0B\x01"

    def pair(self, records):
        return rsc8_audio.pair_marker_regions(rsc8_audio.marker_records(build_synthetic_region_markers(records)))

    def kinds(self, paired):
        return [(e["kind"], e["index"], e["before"], e["after"]) for e in paired["exceptions"]]

    def check_sums(self, paired):
        for counts in paired["names"].values():
            self.assertEqual(counts["starts"],
                             counts["pairs"] + counts["start_without_end"] + counts["end_before_start"])
            self.assertEqual(counts["ends"], counts["pairs"] + counts["end_without_start"] + counts["end_before_start"])

    def test_value_words_are_the_hashes_of_start_and_end(self):
        self.assertEqual(rsc8_audio.MARKER_VALUE_START, self.S)
        self.assertEqual(rsc8_audio.MARKER_VALUE_END, self.E)

    def test_marker_records_are_the_whole_records(self):
        blob = build_synthetic_region_markers([(self.VERSE, self.S, 100, 0), (self.VERSE, self.E, 200, 1)])
        self.assertEqual(rsc8_audio.marker_records(blob), [(self.VERSE, self.S, 100, 0), (self.VERSE, self.E, 200, 1)])
        self.assertEqual(len(rsc8_audio.marker_records(blob[:-1])), 1)
        self.assertEqual(rsc8_audio.marker_records(b""), [])

    def test_well_formed_pairs(self):
        p = self.pair([
            (self.BEAT, struct.unpack("<I", struct.pack("<f", 4.0))[0], 0, 0),   # not a region cue: ignored
            (self.VERSE, self.S, 100, 0),
            (self.VERSE, self.E, 200, 0),
            (self.VERSE, self.S, 200, 0),   # starts where the last one ended
            (self.BEAT, struct.unpack("<I", struct.pack("<f", 8.0))[0], 200, 0),
            (self.VERSE, self.E, 200, 0),   # a zero-length region is still a pair
            (self.VERSE, self.S, 300, 2),
            (self.VERSE, self.E, 900, 3),
        ])
        self.assertEqual(p["exceptions"], [])
        self.assertEqual(list(p["names"]), [self.VERSE])
        self.assertEqual(p["names"][self.VERSE], {
            "starts": 3, "ends": 3, "other_value": 0, "pairs": 3,
            "start_without_end": 0, "end_without_start": 0, "end_before_start": 0,
        })
        self.assertEqual([(r["start"], r["end"], r["start_index"], r["end_index"]) for r in p["regions"]],
                         [(100, 200, 1, 2), (200, 200, 3, 5), (300, 900, 6, 7)])
        self.assertEqual([r["extra_differs"] for r in p["regions"]], [False, False, True])
        self.check_sums(p)
        # A tag without a start/end value has no region name at all.
        self.assertEqual(self.pair([(self.BEAT, 0, 0, 0), (self.VERSE, 0x33333333, 5, 0)]),
                         {"regions": [], "exceptions": [], "names": {}})

    def test_unmatched_start(self):
        p = self.pair([(self.VERSE, self.S, 100, 0), (self.VERSE, self.E, 200, 0), (self.VERSE, self.S, 300, 0)])
        self.assertEqual(len(p["regions"]), 1)
        self.assertEqual(self.kinds(p), [("start_without_end", 2, "end", None)])   # open at the end of the tag
        self.assertEqual(p["exceptions"][0], {
            "kind": "start_without_end", "name": self.VERSE, "value": self.S, "index": 2, "offset": 300,
            "before": "end", "before_offset": 200, "after": None, "after_offset": None,
        })
        self.check_sums(p)

    def test_unmatched_end(self):
        p = self.pair([(self.VERSE, self.E, 50, 0), (self.VERSE, self.S, 100, 0), (self.VERSE, self.E, 200, 0),
                       (self.VERSE, self.E, 250, 0)])
        self.assertEqual([(r["start"], r["end"]) for r in p["regions"]], [(100, 200)])
        self.assertEqual(self.kinds(p), [("end_without_start", 0, None, "start"),   # open at the start of the tag
                                         ("end_without_start", 3, "end", None)])    # a second end
        self.check_sums(p)

    def test_end_before_its_start(self):
        # In record order: the end comes first, so neither record has a partner.
        p = self.pair([(self.VERSE, self.E, 100, 0), (self.VERSE, self.S, 200, 0)])
        self.assertEqual(p["regions"], [])
        self.assertEqual(self.kinds(p), [("end_without_start", 0, None, "start"),
                                         ("start_without_end", 1, "end", None)])
        self.check_sums(p)
        # In offsets: the end follows its start but lies before it. The pair is refused, once.
        p = self.pair([(self.VERSE, self.S, 500, 0), (self.VERSE, self.E, 400, 0), (self.VERSE, self.S, 600, 0),
                       (self.VERSE, self.E, 600, 0)])
        self.assertEqual([(r["start"], r["end"]) for r in p["regions"]], [(600, 600)])
        self.assertEqual(self.kinds(p), [("end_before_start", 1, "start", "start")])
        self.assertEqual(p["names"][self.VERSE]["end_before_start"], 1)
        self.assertEqual(p["names"][self.VERSE]["start_without_end"], 0)
        self.check_sums(p)

    def test_nesting_in_one_name_is_not_paired_inside_out(self):
        p = self.pair([(self.VERSE, self.S, 100, 0), (self.VERSE, self.S, 200, 0), (self.VERSE, self.E, 300, 0),
                       (self.VERSE, self.E, 400, 0)])
        self.assertEqual([(r["start"], r["end"]) for r in p["regions"]], [(200, 300)])
        self.assertEqual(self.kinds(p), [("start_without_end", 0, None, "start"),
                                         ("end_without_start", 3, "end", None)])
        self.check_sums(p)

    def test_two_names_interleaved(self):
        p = self.pair([
            (self.VERSE, self.S, 100, 0),
            (self.FILL, self.S, 150, 0),
            (self.VERSE, self.E, 200, 0),    # VERSE 100..200 and FILL 150..250 cross
            (self.FILL, self.E, 250, 0),
            (self.VERSE, self.S, 300, 0),
            (self.FILL, self.S, 300, 0),
            (self.FILL, self.E, 350, 0),     # FILL 300..350 inside VERSE 300..400, same start
            (self.VERSE, self.E, 400, 0),
            (self.FILL, self.S, 400, 0),     # FILL 400..450 only touches VERSE 300..400
            (self.FILL, self.E, 450, 0),
            (self.VERSE, self.S, 500, 0),
            (self.FILL, self.S, 520, 0),
            (self.FILL, self.E, 520, 0),     # a zero-length region shares no sample
            (self.FILL, self.S, 530, 0),
            (self.FILL, self.E, 600, 0),     # FILL 530..600 inside VERSE 500..600, same end
            (self.VERSE, self.E, 600, 0),
        ])
        self.assertEqual(p["exceptions"], [])
        self.assertEqual(p["names"][self.VERSE]["pairs"], 3)
        self.assertEqual(p["names"][self.FILL]["pairs"], 5)
        self.assertEqual([r["start_index"] for r in p["regions"]], [0, 1, 4, 5, 8, 10, 11, 13])
        # {(smaller hash, larger hash): [nested, crossing]}
        self.assertEqual(rsc8_audio.marker_region_overlaps(p["regions"]), {(self.VERSE, self.FILL): [2, 1]})
        # Each name's regions alone overlap nothing.
        for name in (self.VERSE, self.FILL):
            own = [r for r in p["regions"] if r["name"] == name]
            self.assertEqual(rsc8_audio.marker_region_overlaps(own), {})
        self.check_sums(p)

    def test_a_third_value_opens_and_closes_nothing(self):
        other = 0x44444444
        p = self.pair([(self.VERSE, self.S, 100, 0), (self.VERSE, other, 150, 0), (self.VERSE, self.E, 200, 0),
                       (self.VERSE, self.S, 300, 0), (self.VERSE, self.E, 400, 0)])
        self.assertEqual([(r["start"], r["end"]) for r in p["regions"]], [(300, 400)])
        self.assertEqual(self.kinds(p), [("start_without_end", 0, None, "other"),
                                         ("other_value", 1, "start", "end"),
                                         ("end_without_start", 2, "other", "start")])
        self.assertEqual(p["exceptions"][1]["value"], other)
        self.assertEqual(p["names"][self.VERSE]["other_value"], 1)
        self.check_sums(p)

    def region_bank(self, sid, records, fmt=True):
        tags = {0x55: b"\xF8\x00" + bytes(62), 0xBD: build_synthetic_region_markers(records)}
        if fmt:
            tags[0x76] = self.FMT
        return build_minimal_tag_bank(sid, tags)

    def test_bank_tally_and_scan_summary(self):
        clean = self.region_bank(0x00112233, [
            (self.VERSE, self.S, 100, 0), (self.FILL, self.S, 150, 0), (self.VERSE, self.E, 200, 0),
            (self.FILL, self.E, 450, 0), (self.VERSE, self.S, 1000, 0), (self.VERSE, self.E, 1700, 1),
        ])
        ragged = self.region_bank(0x00445566, [
            (self.BEAT, 0, 0, 0), (self.FILL, self.E, 10, 0), (self.FILL, self.S, 20, 0), (self.FILL, self.E, 60, 0),
            (self.FILL, self.S, 90, 0),
        ], fmt=False)
        plain = self.region_bank(0x00778899, [(self.BEAT, 0, 0, 0), (self.BEAT, 0, 100, 0)])
        # Offsets that decrease make the tag invalid; its records are not paired.
        broken = self.region_bank(0x00AABBCC, [(self.VERSE, self.S, 500, 0), (self.VERSE, self.E, 400, 0)])

        res = rsc8_audio.decode_awc_bank(clean)
        rep = rsc8_audio.marker_region_report(res["marker_regions"])
        self.assertEqual((rep["tags"], rep["tags_all_paired"], rep["tags_with_sample_count"]), (1, 1, 1))
        self.assertEqual((rep["starts"], rep["ends"], rep["pairs"]), (3, 3, 3))
        self.assertEqual(rep["pairs_per_tag"], {"min": 3, "median": 3, "max": 3})
        self.assertEqual(rep["pairs_extra_differs"], 1)
        both = f"0x{self.VERSE:08x}+0x{self.FILL:08x}"
        self.assertEqual(rep["overlaps_between_names"], {
            "nested": 0, "crossing": 1, "by_names": {both: {"nested": 0, "crossing": 1}},
        })
        self.assertEqual(list(rep["names"]), [f"0x{self.VERSE:08x}", f"0x{self.FILL:08x}"])   # most records first
        self.assertEqual(rep["names"][f"0x{self.VERSE:08x}"], {
            "tags": 1, "starts": 2, "ends": 2, "other_value": 0, "pairs": 2,
            "start_without_end": 0, "end_without_start": 0, "end_before_start": 0,
            "length_samples": {"min": 100, "median": 400, "max": 700},
        })
        self.assertEqual((rep["exceptions_total"], rep["exceptions"]), (0, []))
        self.assertEqual(rsc8_audio.decode_awc_bank(plain)["marker_regions"]["tags"], 0)
        res_broken = rsc8_audio.decode_awc_bank(broken)
        self.assertEqual((res_broken["marker_invalid"], res_broken["marker_regions"]["tags"]), (1, 0))

        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "synth.rpf")
            offsets = []
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + struct.pack("<III", 0x100, 0, 0) + bytes(0xF0))
                for bank in (clean, ragged, plain, broken, clean):
                    offsets.append(f.tell())
                    f.write(bank)
                    f.write(bytes(0x40))
            _records, summary = rsc8_audio.scan_rpf(rpf)
            self.assertEqual((summary["marker_streams"], summary["marker_invalid"]), (5, 1))
            rep = summary["marker_regions"]
            self.assertEqual((rep["tags"], rep["tags_all_paired"], rep["tags_with_sample_count"]), (3, 2, 2))
            self.assertEqual((rep["starts"], rep["ends"], rep["pairs"]), (8, 8, 7))
            self.assertEqual(rep["pairs_per_tag"], {"min": 1, "median": 3, "max": 3})
            self.assertEqual(rep["overlaps_between_names"], {
                "nested": 0, "crossing": 2, "by_names": {both: {"nested": 0, "crossing": 2}},
            })
            self.assertEqual(rep["names"][f"0x{self.FILL:08x}"], {
                "tags": 3, "starts": 4, "ends": 4, "other_value": 0, "pairs": 3,
                "start_without_end": 1, "end_without_start": 1, "end_before_start": 0,
                "length_samples": {"min": 40, "median": 300, "max": 300},
            })
            self.assertEqual(rep["exceptions_total"], 2)
            self.assertEqual(rep["exceptions"], [
                {"kind": "end_without_start", "name": f"0x{self.FILL:08x}", "value": f"0x{self.E:08x}",
                 "before": None, "before_offset": None, "after": "start", "after_offset": 20,
                 "record": 1, "tag_records": 5, "offset": 10,
                 "tag_last_offset": 90, "stream_samples": None, "bank_offset": offsets[1],
                 "stream_id": "0x00445566"},
                {"kind": "start_without_end", "name": f"0x{self.FILL:08x}", "value": f"0x{self.S:08x}",
                 "before": "end", "before_offset": 60, "after": None, "after_offset": None,
                 "record": 4, "tag_records": 5, "offset": 90,
                 "tag_last_offset": 90, "stream_samples": None, "bank_offset": offsets[1],
                 "stream_id": "0x00445566"},
            ])

            # The list of exceptions is capped; the counts are not.
            old_cap = rsc8_audio.MAX_REGION_EXCEPTIONS_LISTED
            try:
                rsc8_audio.MAX_REGION_EXCEPTIONS_LISTED = 1
                _records, capped = rsc8_audio.scan_rpf(rpf)
            finally:
                rsc8_audio.MAX_REGION_EXCEPTIONS_LISTED = old_cap
            self.assertEqual(capped["marker_regions"]["exceptions_total"], 2)
            self.assertEqual(capped["marker_regions"]["exceptions"], rep["exceptions"][:1])

            # Directory and single-bank scans report the same block; the CLI writes it as JSON.
            os.remove(rpf)
            for i, bank in enumerate((clean, ragged, plain, broken, clean)):
                with open(os.path.join(tmp, f"bank{i}.awc"), "wb") as f:
                    f.write(bank)
            out_json = os.path.join(tmp, "report.json")
            cp = subprocess.run(
                [sys.executable, "-I", TOOL, tmp, out_json, "--summary-only", "--quiet"],
                capture_output=True, text=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)
            with open(out_json, encoding="utf-8") as f:
                from_dir = json.load(f)["summary"]["marker_regions"]
            self.assertEqual(json.loads(cp.stdout)["marker_regions"], from_dir)
            for entry in from_dir["exceptions"]:
                self.assertEqual(entry.pop("bank_offset"), 0)
            for entry in rep["exceptions"]:
                entry.pop("bank_offset")
            self.assertEqual(from_dir, rep)
            _records, single = rsc8_audio.scan_single_file(os.path.join(tmp, "bank1.awc"))
            self.assertEqual((single["marker_regions"]["pairs"], single["marker_regions"]["exceptions_total"]), (1, 2))
            self.assertEqual(single["marker_regions"]["names"][f"0x{self.FILL:08x}"]["length_samples"],
                             {"min": 40, "median": 40, "max": 40})

    def test_a_scan_without_region_cues_reports_an_empty_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            rsc_path = os.path.join(tmp, "only.rsc8")
            with open(rsc_path, "wb") as f:
                f.write(build_synthetic_rsc8(stream_hash=0x11112222, frames=45))
            _records, summary = rsc8_audio.scan_single_file(rsc_path)
        self.assertEqual(summary["marker_regions"], {
            "tags": 0, "tags_all_paired": 0, "tags_with_sample_count": 0, "starts": 0, "ends": 0, "pairs": 0,
            "pairs_per_tag": None, "pairs_extra_differs": 0,
            "overlaps_between_names": {"nested": 0, "crossing": 0, "by_names": {}},
            "names": {}, "exceptions_total": 0, "exceptions": [],
        })


if __name__ == "__main__":
    unittest.main()
