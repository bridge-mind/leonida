"""Synthetic unit tests for `tools/ped_catalogues.py`.

All test inputs are synthesized in memory or inside a temporary directory: no game files,
no real asset names, and no retail hashes are used.

    python -I tests/test_ped_catalogues.py
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import odrp_7a  # noqa: E402
import ped_catalogues as pc  # noqa: E402


def _encode_ptr(block_1based: int, byte_offset: int) -> int:
    return (byte_offset << 20) | (block_1based & 0xFFFFF)


def _align16(n: int) -> int:
    return (n + 15) & ~15


def build_synthetic_0drp(
    blocks: list[tuple[int, bytes]],
    *,
    root_block: int = 1,
    truncate_by: int = 0,
) -> bytes:
    """Build a valid standalone `0DRP` v122 resource buffer from a 1-based list of `(key, payload)`."""
    # Layout:
    #   +0x00..+0x5f: 0x60-byte 0DRP/0x7a header
    #   +0x60..     : minimal struct table (1 struct of 32 bytes matching root block type)
    #   aligned     : block directory (nb * 16 bytes)
    #   aligned     : block payloads
    root_type = blocks[root_block - 1][0] if 1 <= root_block <= len(blocks) else 0x11111111
    q28 = 0x60
    struct_table = struct.pack("<IIIIQIHH", root_type, 0, 0, 0, 0, max(16, len(blocks[root_block - 1][1])), 0, 0)
    q38 = _align16(q28 + len(struct_table))
    nb = len(blocks)
    payload_start = _align16(q38 + nb * 16)

    out = bytearray(payload_start)
    # 16-byte system prefix: 8 zero bytes + virtual page-map pointer at +0x08
    struct.pack_into("<Q", out, 0x08, odrp_7a.VIRTUAL_BASE + 0x10)
    # 80-byte 0DRP v122 frame at +0x10..+0x5f
    out[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", out, 0x14, 122)
    struct.pack_into("<I", out, 0x20, root_block)
    struct.pack_into("<Q", out, 0x28, odrp_7a.VIRTUAL_BASE + q28)
    struct.pack_into("<Q", out, 0x30, 0)
    struct.pack_into("<Q", out, 0x38, odrp_7a.VIRTUAL_BASE + q38)
    struct.pack_into("<Q", out, 0x40, 0)
    struct.pack_into("<4H", out, 0x48, 1, 0, nb, 0)
    struct.pack_into("<Q", out, 0x50, 0)
    out[q28:q28 + len(struct_table)] = struct_table

    for i, (key, payload) in enumerate(blocks):
        off = len(out)
        struct.pack_into("<IIQ", out, q38 + i * 16, key, len(payload), odrp_7a.VIRTUAL_BASE + off)
        out.extend(payload)
        pad = _align16(len(out)) - len(out)
        if pad:
            out.extend(b"\x00" * pad)

    if truncate_by > 0:
        return bytes(out[:-truncate_by])
    return bytes(out)


def build_synthetic_psin(
    blocks: list[tuple[int, bytes]],
    *,
    root_block: int = 1,
) -> bytes:
    """Build a valid standalone `PSIN` container from a 1-based list of `(key, payload)`."""
    nb = len(blocks)
    table_bytes = 8 + nb * 16 + 8  # header(8) + nb*16 + 8 bytes of 0x70 padding
    assert table_bytes % 16 == 0
    payloads = bytearray()
    for _key, payload in blocks:
        payloads.extend(payload)
        pad = _align16(len(payloads)) - len(payloads)
        if pad:
            payloads.extend(b"\x00" * pad)
    first_section = table_bytes + len(payloads)

    out = bytearray()
    out.extend(b"PSIN")
    out.extend(struct.pack(">I", first_section))
    for key, payload in blocks:
        out.extend(struct.pack("<II", key, len(payload)))
        out.extend(b"\xaa" * 8)
    out.extend(b"\x70" * 8)
    out.extend(payloads)

    # Minimal valid LSCH (1 struct header of 32 bytes, 0 members), LENM (empty), LRSC (80 B 0DRP), CHKS (12 B payload).
    root_type = blocks[root_block - 1][0] if 1 <= root_block <= nb else 0x11111111
    lsch_pay = struct.pack("<IIIIQIHH", root_type, 0, 0, 0, 0, 192, 0, 0)
    out.extend(b"LSCH" + struct.pack(">I", 8 + len(lsch_pay)) + lsch_pay)
    out.extend(b"LENM" + struct.pack(">I", 8))

    lrsc_pay = bytearray(80)
    lrsc_pay[0:4] = b"0DRP"
    struct.pack_into("<I", lrsc_pay, 4, 122)
    struct.pack_into("<I", lrsc_pay, 0x10, root_block)
    lrsc_pay[0x18:0x38] = b"\xbb" * 32
    struct.pack_into("<4H", lrsc_pay, 0x38, 1, 0, nb, 0)
    lrsc_pay[0x40:0x48] = b"\xbb" * 8
    out.extend(b"LRSC" + struct.pack(">I", 8 + len(lrsc_pay)) + lrsc_pay)
    out.extend(b"CHKS" + struct.pack(">I", 20) + b"\x00" * 12)
    return bytes(out)


def _pack_root(bucket0_count: int, bucket0_ptr: int = 0, bucket1_count: int = 0, bucket1_ptr: int = 0) -> bytes:
    buf = bytearray(pc.CATALOG_ROOT_SIZE)
    struct.pack_into("<IIQ", buf, 0x08, bucket0_count, bucket0_count, bucket0_ptr)
    struct.pack_into("<IIQ", buf, 0x38, bucket1_count, bucket1_count, bucket1_ptr)
    return bytes(buf)


def _pack_preset(
    *,
    preset_id: int,
    parent_id: int = 0,
    state_count: int = 0,
    state_ptr: int = 0,
    category: int = 0,
    tag_ptr: int = 0,
    tag_count: int = 0,
) -> bytes:
    buf = bytearray(pc.PRESET_STRIDE)
    struct.pack_into("<IIIIQI", buf, 0x00, preset_id, parent_id, state_count, state_count, state_ptr, category)
    struct.pack_into("<QQHH", buf, 0x28, 1 if tag_count else 0, tag_ptr, tag_count, tag_count)
    return bytes(buf)


def _pack_state(
    *,
    state_id: int = 1,
    flags: int = 0,
    drawable_hash: int = 0,
    albedo_hash: int = 0,
    normal_hash: int = 0,
    extra_hash: int = 0,
    ext_ptr: int = 0,
    ext_count: int = 0,
    tag_ptr: int = 0,
    tag_count: int = 0,
) -> bytes:
    buf = bytearray(pc.STATE_STRIDE)
    struct.pack_into("<II", buf, 0x00, state_id, flags)
    struct.pack_into("<QQQQ", buf, 0x08, drawable_hash, albedo_hash, normal_hash, extra_hash)
    struct.pack_into("<QQHH", buf, 0x28, 1 if ext_count else 0, ext_ptr, ext_count, ext_count)
    struct.pack_into("<QQHH", buf, 0xA0, 1 if tag_count else 0, tag_ptr, tag_count, tag_count)
    return bytes(buf)


def build_rich_catalogue_blocks(
    *,
    tag_block_type: int = 0x101,
    ext_block_type: int = 0x103,
) -> tuple[list[tuple[int, bytes]], dict[str, int]]:
    """Build a synthetic `0x8dcb4cf5` catalogue exercising tags, 2-hop inheritance, states, and `cull_drawable`.

    Uses non-default synthetic block IDs (`0x101` for tags, `0x103` for `0xb70a22c1` ext maps)
    to verify the parser does not hardcode `0x100` / `0x101`.
    """
    names = {
        "direct_head": "synth_head_direct_00",
        "inherited_head": "synth_head_inherited_01",
        "state_tag_item": "synth_hat_statetag_02",
        "h0_only_body": "synth_body_full_03",
        "cull_target": "synth_body_cull_03",
    }
    hashes = {f"{k}_j32": pc.joaat32(v) for k, v in names.items()}
    hashes.update({f"{k}_m64": pc.murmur64(v) for k, v in names.items()})

    # Block numbering (1-based):
    #   1: 0x8dcb4cf5 root
    #   2: 0xf795e8ba presets (5 presets)
    #   3: 0x37e52277 states (4 states)
    #   4: tag_block_type (24 B slots)
    #   5: 0x4a u32 tag values
    #   6: ext_block_type (16 B slots for 0xb70a22c1)
    #   7: 0x3d780b74 cull_drawable targets (16 B each)
    tag_vals = bytearray()
    tag_slots = bytearray()

    def add_tag_map(entries: list[tuple[int, list[int]]]) -> tuple[int, int]:
        slot_off = len(tag_slots)
        for tk, vals in entries:
            val_off = len(tag_vals)
            for v in vals:
                tag_vals.extend(struct.pack("<I", v))
            val_ptr = _encode_ptr(5, val_off) if vals else 0
            tag_slots.extend(struct.pack("<IxxxxIIQ", tk, len(vals), len(vals), val_ptr))
        return _encode_ptr(4, slot_off), len(entries)

    # Tag maps
    t_direct_ptr, t_direct_cnt = add_tag_map([(0x4BFD27DD, [hashes["direct_head_j32"]])])
    t_grand_ptr, t_grand_cnt = add_tag_map([(0x1C114E61, [hashes["inherited_head_j32"]]), (0x12345678, [0xCAFEBABE])])
    t_child_override_ptr, t_child_override_cnt = add_tag_map([(0x12345678, [0xDEADBEEF])])
    t_state_ptr, t_state_cnt = add_tag_map([(0x4BFD27DD, [hashes["state_tag_item_j32"]])])

    # 0x3d780b74 cull target record (block 7, offset 0)
    cull_block = struct.pack("<QQ", 0, hashes["cull_target_m64"])

    # 0xb70a22c1 ext map (block 6): decoy key + 0x46e1a048 ("cull_drawable") -> block 7+0
    ext_block = bytearray()
    ext_block.extend(struct.pack("<IxxxxQ", 0x1A271DE4, 0))
    ext_block.extend(struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, _encode_ptr(7, 0)))

    # States (block 3): 4 states
    #   state 0: direct preset head with albedo + normal
    #   state 1: grandchild preset inheriting drawable tag from grandparent, holding face textures
    #   state 2: state-level drawable tag + textures
    #   state 3: no drawable tags; h0 == h0_only_body + cull_drawable -> cull_target
    s0 = _pack_state(
        state_id=0x101,
        drawable_hash=hashes["direct_head_m64"],
        albedo_hash=0xAAA0000000000001,
        normal_hash=0xBBB0000000000001,
        extra_hash=0xCCC0000000000001,
    )
    s1 = _pack_state(
        state_id=0x102,
        drawable_hash=0,
        albedo_hash=0xAAA0000000000002,
        normal_hash=0xBBB0000000000002,
    )
    s2 = _pack_state(
        state_id=0x103,
        drawable_hash=0,
        albedo_hash=0xAAA0000000000003,
        normal_hash=0,
        tag_ptr=t_state_ptr,
        tag_count=t_state_cnt,
    )
    s3 = _pack_state(
        state_id=0x104,
        flags=7,
        drawable_hash=hashes["h0_only_body_m64"],
        albedo_hash=0xAAA0000000000004,
        normal_hash=0xBBB0000000000004,
        ext_ptr=_encode_ptr(6, 0),
        ext_count=2,
    )
    states_block = s0 + s1 + s2 + s3

    # Presets (block 2): 5 presets
    #   0x1000: direct head (state 0)
    #   0x2000: grandparent (has drawable tag 0x1c114e61, 0 states)
    #   0x2001: middle parent (parent_id=0x2000, no drawable tag, 0 states)
    #   0x2002: grandchild (parent_id=0x2001, overrides non-drawable tag, state 1)
    #   0x3000: garment with state 2 (state tag) and state 3 (h0 + cull_drawable)
    p0 = _pack_preset(
        preset_id=0x1000,
        parent_id=0,
        state_count=1,
        state_ptr=_encode_ptr(3, 0),
        category=pc.HEAD_CATEGORY,
        tag_ptr=t_direct_ptr,
        tag_count=t_direct_cnt,
    )
    p1 = _pack_preset(
        preset_id=0x2000,
        parent_id=0,
        state_count=0,
        category=pc.HEAD_CATEGORY,
        tag_ptr=t_grand_ptr,
        tag_count=t_grand_cnt,
    )
    p2 = _pack_preset(
        preset_id=0x2001,
        parent_id=0x2000,
        state_count=0,
        category=pc.HEAD_CATEGORY,
    )
    p3 = _pack_preset(
        preset_id=0x2002,
        parent_id=0x2001,
        state_count=1,
        state_ptr=_encode_ptr(3, pc.STATE_STRIDE),
        category=pc.HEAD_CATEGORY,
        tag_ptr=t_child_override_ptr,
        tag_count=t_child_override_cnt,
    )
    p4 = _pack_preset(
        preset_id=0x3000,
        parent_id=0,
        state_count=2,
        state_ptr=_encode_ptr(3, 2 * pc.STATE_STRIDE),
        category=0x3817131D,
    )
    presets_block = p0 + p1 + p2 + p3 + p4

    root_block = _pack_root(bucket0_count=5, bucket0_ptr=_encode_ptr(2, 0))
    blocks = [
        (pc.CATALOG_ROOT_HASH, root_block),
        (pc.PRESET_STRUCT_HASH, presets_block),
        (pc.STATE_STRUCT_HASH, states_block),
        (tag_block_type, bytes(tag_slots)),
        (pc.TAG_VALUE_BLOCK_TYPE, bytes(tag_vals)),
        (ext_block_type, bytes(ext_block)),
        (pc.CULL_DRAWABLE_STRUCT_HASH, cull_block),
    ]
    return blocks, hashes


def build_synthetic_rpf(streams: list[bytes]) -> bytes:
    """Build a minimal synthetic RPF9 (`9FPR`) archive with 16-byte-aligned zlib streams."""
    data_start = 0x100
    out = bytearray(data_start)
    out[0:4] = b"9FPR"
    struct.pack_into("<I", out, 4, data_start)
    for raw in streams:
        z = zlib.compress(raw, 6)
        out.extend(z)
        pad = _align16(len(out)) - len(out)
        if pad:
            out.extend(b"\x00" * pad)
    return bytes(out)


class PedCataloguesTests(unittest.TestCase):
    def test_cull_drawable_key_matches_joaat(self):
        self.assertEqual(pc.joaat32("cull_drawable"), pc.CULL_DRAWABLE_KEY)
        self.assertEqual(pc.CULL_DRAWABLE_KEY, 0x46E1A048)

    def test_0drp_and_psin_parse_identically(self):
        blocks, hashes = build_rich_catalogue_blocks(tag_block_type=0x101, ext_block_type=0x103)
        raw_0drp = build_synthetic_0drp(blocks)
        raw_psin = build_synthetic_psin(blocks)

        cat_0drp = pc.parse_catalogue_strict(raw_0drp, source="synth.0drp")
        cat_psin = pc.parse_catalogue_strict(raw_psin, source="synth.pso")

        self.assertEqual(cat_0drp.framing, "0drp")
        self.assertEqual(cat_psin.framing, "psin")
        for cat in (cat_0drp, cat_psin):
            self.assertEqual(cat.root_type, pc.CATALOG_ROOT_HASH)
            self.assertEqual(cat.warnings, ())
            self.assertEqual(cat.stats["presets"], 5)
            self.assertEqual(cat.stats["root_bucket_presets"], 5)
            self.assertEqual(cat.stats["presets_with_parent"], 2)
            self.assertEqual(cat.stats["max_parent_depth"], 2)
            self.assertEqual(cat.stats["parent_cycles"], 0)
            self.assertEqual(cat.stats["states"], 4)
            self.assertEqual(cat.stats["states_with_cull_drawable"], 1)
            self.assertEqual(cat.stats["cull_nonzero_hashes"], 1)
            self.assertEqual(cat.stats["cull_records_in_blocks"], 1)
            # Non-default synthetic block IDs were tracked without hardcoding 0x100 / 0x101
            self.assertIn("tag_map_block_0x101", cat.stats)
            self.assertIn("ext_map_block_0x103", cat.stats)

            # Verify cull_drawable state detail
            p4 = cat.presets_by_id[0x3000]
            self.assertEqual(len(p4.states), 2)
            s3 = p4.states[1]
            self.assertEqual(s3.drawable_hash, hashes["h0_only_body_m64"])
            self.assertEqual(s3.albedo_hash, 0xAAA0000000000004)
            self.assertEqual(s3.normal_hash, 0xBBB0000000000004)
            self.assertEqual(s3.cull_drawable, hashes["cull_target_m64"])
            self.assertIsNotNone(s3.cull_ref)
            self.assertEqual(s3.cull_ref.target_type, pc.CULL_DRAWABLE_STRUCT_HASH)
            self.assertEqual(s3.ext_map_keys, (0x1A271DE4, pc.CULL_DRAWABLE_KEY))

            # Verify cull map helper
            cmap = pc.resolve_cull_map(cat)
            self.assertEqual(cmap, {hashes["h0_only_body_m64"]: [hashes["cull_target_m64"]]})

    def test_preset_inheritance_and_resolution(self):
        blocks, hashes = build_rich_catalogue_blocks()
        cat = pc.parse_catalogue_strict(build_synthetic_0drp(blocks), source="inh.0drp")

        grandchild = cat.presets_by_id[0x2002]
        anc_tags, chain, hit_cycle = pc.resolve_ancestor_tags(cat, grandchild)
        self.assertFalse(hit_cycle)
        self.assertEqual(chain, (0x2001, 0x2000))
        self.assertEqual(anc_tags, {0x1C114E61: (hashes["inherited_head_j32"],)})

        # Effective tags merge ancestor keys while preserving child overrides
        eff, eff_chain, eff_cyc = pc.resolve_effective_tags(cat, grandchild)
        self.assertFalse(eff_cyc)
        self.assertEqual(eff_chain, (0x2001, 0x2000))
        self.assertEqual(eff[0x1C114E61], (hashes["inherited_head_j32"],))
        self.assertEqual(eff[0x12345678], (0xDEADBEEF,))

        # Full binding resolution without name filter
        all_bindings = pc.resolve_preset_bindings(cat)
        by_preset = {(b.preset_id, b.state_index): b for b in all_bindings}
        self.assertEqual(by_preset[(0x1000, 0)].match_kind, "drawable_tag")
        self.assertEqual(by_preset[(0x2002, 0)].match_kind, "inherited_drawable_tag")
        self.assertEqual(by_preset[(0x2002, 0)].parent_chain, (0x2001, 0x2000))
        self.assertEqual(by_preset[(0x2002, 0)].albedo_hash, 0xAAA0000000000002)
        self.assertEqual(by_preset[(0x3000, 0)].match_kind, "state_drawable_tag")
        self.assertEqual(by_preset[(0x3000, 1)].match_kind, "state_drawable_hash")
        self.assertEqual(by_preset[(0x3000, 1)].cull_drawable_hash, hashes["cull_target_m64"])

        # Filtered by drawable name (including cull target name annotation)
        named = pc.resolve_preset_bindings(
            cat,
            drawable_names=["synth_head_inherited_01", "synth_body_full_03", "synth_body_cull_03"],
        )
        self.assertEqual(len(named), 2)
        b_inh = next(b for b in named if b.preset_id == 0x2002)
        self.assertEqual(b_inh.matched_name, "synth_head_inherited_01")
        b_body = next(b for b in named if b.preset_id == 0x3000)
        self.assertEqual(b_body.matched_name, "synth_body_full_03")
        self.assertEqual(b_body.cull_drawable_name, "synth_body_cull_03")

        # Filtered specifically by the cull target name alone
        cull_hit = pc.resolve_preset_bindings(cat, drawable_names=["synth_body_cull_03"])
        self.assertEqual(len(cull_hit), 1)
        self.assertEqual(cull_hit[0].match_kind, "cull_drawable_hash")
        self.assertEqual(cull_hit[0].matched_name, "synth_body_cull_03")

    def test_parent_id_cycle_guard_and_unresolved_parent(self):
        # Build presets with:
        #   0x10 -> self-cycle (parent_id = 0x10)
        #   0x20 -> 0x21 -> 0x20 (2-cycle)
        #   0x30 -> 0x9999 (unresolved parent)
        presets_block = (
            _pack_preset(preset_id=0x10, parent_id=0x10)
            + _pack_preset(preset_id=0x20, parent_id=0x21)
            + _pack_preset(preset_id=0x21, parent_id=0x20)
            + _pack_preset(preset_id=0x30, parent_id=0x9999)
        )
        blocks = [
            (pc.CATALOG_ROOT_HASH, _pack_root(bucket0_count=4, bucket0_ptr=_encode_ptr(2, 0))),
            (pc.PRESET_STRUCT_HASH, presets_block),
        ]
        cat = pc.parse_catalogue_strict(build_synthetic_0drp(blocks), source="cycles.0drp")
        self.assertEqual(cat.stats["parent_cycles"], 3)
        self.assertEqual(cat.stats["unresolved_parents"], 1)
        self.assertTrue(any("cycle" in w for w in cat.warnings))

        # Cycle guards on both resolve helpers terminate cleanly
        tags_self, chain_self, cyc_self = pc.resolve_ancestor_tags(cat, cat.presets_by_id[0x10])
        self.assertTrue(cyc_self)
        self.assertEqual(tags_self, {})
        self.assertEqual(chain_self, ())

        tags_two, chain_two, cyc_two = pc.resolve_ancestor_tags(cat, cat.presets_by_id[0x20])
        self.assertTrue(cyc_two)
        self.assertEqual(chain_two, (0x21,))

        eff_two, eff_chain, eff_cyc = pc.resolve_effective_tags(cat, cat.presets_by_id[0x20])
        self.assertTrue(eff_cyc)
        self.assertEqual(eff_two, {})
        self.assertEqual(eff_chain, (0x21,))

        tags_unres, chain_unres, cyc_unres = pc.resolve_ancestor_tags(cat, cat.presets_by_id[0x30])
        self.assertFalse(cyc_unres)
        self.assertEqual(tags_unres, {})
        self.assertEqual(chain_unres, ())

    def test_malformed_and_truncated_inputs(self):
        # Empty / garbage
        self.assertIsNone(pc.parse_catalogue(b""))
        self.assertIsNone(pc.parse_catalogue(b"not a catalogue"))
        with self.assertRaises(ValueError):
            pc.parse_catalogue_strict(b"not a catalogue")

        # Truncated 0DRP
        blocks, _ = build_rich_catalogue_blocks()
        full_0drp = build_synthetic_0drp(blocks)
        truncated = full_0drp[:-32]
        self.assertIsNone(pc.parse_catalogue(truncated))
        with self.assertRaisesRegex(ValueError, "truncated"):
            pc.parse_catalogue_strict(truncated)

        # Wrong root struct type (non-catalogue 0DRP and PSIN)
        non_cat = [(0xD3593FA6, b"\x00" * 64)]
        self.assertIsNone(pc.parse_catalogue(build_synthetic_0drp(non_cat)))
        self.assertIsNone(pc.parse_catalogue(build_synthetic_psin(non_cat)))
        with self.assertRaisesRegex(ValueError, "expected 0x8dcb4cf5"):
            pc.parse_catalogue_strict(build_synthetic_0drp(non_cat))

        # Malformed internal pointers inside a valid 0x8dcb4cf5 container produce warnings, never crash
        bad_ext = struct.pack("<IxxxxQ", pc.CULL_DRAWABLE_KEY, _encode_ptr(5, 0))  # points to wrong block type 0x4a
        bad_state = _pack_state(
            state_id=1,
            ext_ptr=_encode_ptr(4, 0),
            ext_count=1,
            tag_ptr=_encode_ptr(99, 0),  # missing block
            tag_count=2,
        )
        bad_preset = _pack_preset(
            preset_id=0x55,
            state_count=3,  # claims 3 states, block only has 1 -> truncation warning
            state_ptr=_encode_ptr(3, 0),
        )
        bad_blocks = [
            (pc.CATALOG_ROOT_HASH, _pack_root(bucket0_count=1, bucket0_ptr=_encode_ptr(2, 0))),
            (pc.PRESET_STRUCT_HASH, bad_preset),
            (pc.STATE_STRUCT_HASH, bad_state),
            (0x101, bad_ext),
            (pc.TAG_VALUE_BLOCK_TYPE, b"\x00" * 16),
        ]
        cat_bad = pc.parse_catalogue_strict(build_synthetic_0drp(bad_blocks), source="bad_ptrs.0drp")
        self.assertEqual(len(cat_bad.presets), 1)
        self.assertEqual(len(cat_bad.presets[0].states), 1)
        self.assertIsNone(cat_bad.presets[0].states[0].cull_drawable)
        self.assertGreaterEqual(len(cat_bad.warnings), 3)

    def test_rpf_and_carved_dir_continuation_joining(self):
        blocks, _ = build_rich_catalogue_blocks()
        raw_0drp = build_synthetic_0drp(blocks)
        raw_psin = build_synthetic_psin(blocks)
        non_cat = build_synthetic_0drp([(0xD3593FA6, b"\x00" * 64)])
        noise = b"unrelated stream payload" * 10

        # Split both catalogues into 2 streams to exercise multi-stream continuation joining!
        split_0drp = len(raw_0drp) // 2
        split_psin = len(raw_psin) // 2
        rpf_bytes = build_synthetic_rpf([
            noise,
            raw_0drp[:split_0drp],
            raw_0drp[split_0drp:],
            non_cat,
            raw_psin[:split_psin],
            raw_psin[split_psin:],
        ])

        with tempfile.TemporaryDirectory() as tmp:
            rpf_path = os.path.join(tmp, "synth.rpf")
            with open(rpf_path, "wb") as fh:
                fh.write(rpf_bytes)

            rep = pc.census(rpf_path)
            self.assertEqual(rep["catalogues"], 2)
            self.assertEqual(rep["by_framing"], {"0drp": 1, "psin": 1})
            self.assertEqual(rep["presets"], 10)
            self.assertEqual(rep["states"], 8)
            self.assertEqual(rep["states_with_cull_drawable"], 2)
            self.assertEqual(rep["catalogues_with_cull_drawable"], 2)
            self.assertEqual(rep["warnings"], 0)

            # Also test carved directory with index.json joining a 2-stream 0DRP
            carve_dir = os.path.join(tmp, "carved")
            os.makedirs(carve_dir)
            part0 = raw_0drp[:split_0drp]
            part1 = raw_0drp[split_0drp:]
            with open(os.path.join(carve_dir, "000000000100.bin"), "wb") as fh:
                fh.write(part0)
            with open(os.path.join(carve_dir, "000000000300.bin"), "wb") as fh:
                fh.write(part1)
            with open(os.path.join(carve_dir, "index.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "files": [
                            {"offset": 0x100, "stored": 0x1F5, "file": "000000000100.bin"},
                            {"offset": 0x300, "stored": 0x200, "file": "000000000300.bin"},
                        ]
                    },
                    fh,
                )
            dir_rep = pc.census(carve_dir)
            self.assertEqual(dir_rep["catalogues"], 1)
            self.assertEqual(dir_rep["by_framing"], {"0drp": 1})
            self.assertEqual(dir_rep["presets"], 5)

    def test_large_single_stream_and_adjacent_complete_0drp_in_rpf(self):
        # 1. A >1 MiB single zlib stream followed by trailing bytes (exercises multi-chunk decompress + eof/unused_data).
        blocks_big, _ = build_rich_catalogue_blocks()
        blocks_big.append((0x99999999, b"\x00" * (1_200_000)))
        big_0drp = build_synthetic_0drp(blocks_big)
        self.assertGreater(len(big_0drp), 1 << 20)

        # 2. A complete 64 KiB-multiple 0DRP immediately followed by another complete 0DRP
        #    (must not swallow the second catalogue as a continuation).
        blocks_a, _ = build_rich_catalogue_blocks()
        base_a = build_synthetic_0drp(blocks_a)
        pad_len = 65536 - (len(base_a) % 65536)
        exact_64k_0drp = base_a + (b"\x00" * pad_len)
        self.assertEqual(len(exact_64k_0drp) % 65536, 0)
        second_0drp = build_synthetic_0drp(blocks_a)

        rpf_bytes = build_synthetic_rpf([big_0drp, exact_64k_0drp, second_0drp])
        with tempfile.TemporaryDirectory() as tmp:
            rpf_path = os.path.join(tmp, "edge.rpf")
            with open(rpf_path, "wb") as fh:
                fh.write(rpf_bytes)
            rep = pc.census(rpf_path)
            self.assertEqual(rep["catalogues"], 3)
            self.assertEqual(rep["by_framing"], {"0drp": 3})
            self.assertEqual(rep["presets"], 15)
            self.assertEqual(rep["warnings"], 0)

    def test_cli_census_inspect_resolve_and_guards(self):
        blocks, hashes = build_rich_catalogue_blocks()
        raw_0drp = build_synthetic_0drp(blocks)
        script = os.path.join(TOOLS, "ped_catalogues.py")
        py = [sys.executable, "-I", script]

        with tempfile.TemporaryDirectory() as tmp:
            cat_file = os.path.join(tmp, "sample.0drp")
            with open(cat_file, "wb") as fh:
                fh.write(raw_0drp)

            # inspect --json --out
            out_ins = os.path.join(tmp, "inspect.json")
            r1 = subprocess.run(py + ["inspect", cat_file, "--json", "--out", out_ins], capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            doc_ins = json.loads(r1.stdout)
            self.assertEqual(doc_ins["framing"], "0drp")
            self.assertEqual(doc_ins["stats"]["presets"], 5)
            with open(out_ins, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), doc_ins)

            # census --json
            r2 = subprocess.run(py + ["census", cat_file, "--json"], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            doc_cen = json.loads(r2.stdout)
            self.assertEqual(doc_cen["catalogues"], 1)
            self.assertEqual(doc_cen["states_with_cull_drawable"], 1)

            # resolve --name --cull-only --json
            r3 = subprocess.run(
                py + ["resolve", cat_file, "--name", "synth_body_full_03", "--cull-only", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r3.returncode, 0, r3.stderr)
            doc_res = json.loads(r3.stdout)
            self.assertEqual(doc_res["matches"], 1)
            self.assertEqual(doc_res["bindings"][0]["cull_drawable_hash"], f"{hashes['cull_target_m64']:016x}")

            # resolve --hash (32-bit joaat of inherited head)
            r4 = subprocess.run(
                py + ["resolve", cat_file, "--hash", f"0x{hashes['inherited_head_j32']:08x}", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r4.returncode, 0, r4.stderr)
            doc_res4 = json.loads(r4.stdout)
            self.assertEqual(doc_res4["matches"], 1)
            self.assertEqual(doc_res4["bindings"][0]["match_kind"], "inherited_drawable_tag")
            self.assertEqual(doc_res4["bindings"][0]["parent_chain"], ["0x00002001", "0x00002000"])

            # Refuses overwriting input file
            r5 = subprocess.run(py + ["census", cat_file, "--out", cat_file], capture_output=True, text=True)
            self.assertEqual(r5.returncode, 2)
            self.assertIn("overwrite", r5.stderr)

            # Private-use character in path under -X utf8=0 must not crash stdout
            weird_dir = os.path.join(tmp, "dir-\ue9c3")
            os.makedirs(weird_dir)
            weird_file = os.path.join(weird_dir, "cat-\ue9c3.0drp")
            with open(weird_file, "wb") as fh:
                fh.write(raw_0drp)
            r6 = subprocess.run(
                [sys.executable, "-I", "-X", "utf8=0", script, "census", weird_file],
                capture_output=True,
            )
            self.assertEqual(r6.returncode, 0, r6.stderr.decode("utf-8", "replace"))
            self.assertNotIn("UnicodeEncodeError", r6.stderr.decode("utf-8", "replace"))

    def test_cli_out_path_guards_directory_inputs(self):
        blocks, _ = build_rich_catalogue_blocks()
        raw_0drp = build_synthetic_0drp(blocks)
        script = os.path.join(TOOLS, "ped_catalogues.py")
        py = [sys.executable, "-I", script]

        with tempfile.TemporaryDirectory() as tmp:
            cat_dir = os.path.join(tmp, "cats")
            sub_dir = os.path.join(cat_dir, "sub")
            sibling_dir = os.path.join(tmp, "cats_out")
            os.makedirs(sub_dir)
            os.makedirs(sibling_dir)

            cat_file = os.path.join(cat_dir, "sample.0drp")
            with open(cat_file, "wb") as fh:
                fh.write(raw_0drp)

            # census <dir> --out <dir>/<catalogue> -> exit 2, catalogue untouched
            r1 = subprocess.run(py + ["census", cat_dir, "--out", cat_file], capture_output=True, text=True)
            self.assertEqual(r1.returncode, 2)
            self.assertIn("inside input folder", r1.stderr)
            with open(cat_file, "rb") as fh:
                self.assertEqual(fh.read(), raw_0drp)

            # resolve <dir> --out <dir>/<catalogue> -> exit 2, catalogue untouched
            r2 = subprocess.run(py + ["resolve", cat_dir, "--out", cat_file], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 2)
            self.assertIn("inside input folder", r2.stderr)
            with open(cat_file, "rb") as fh:
                self.assertEqual(fh.read(), raw_0drp)

            # census <dir> --out <dir>/sub/report.json (existing and non-existing nested dir) -> exit 2
            nested_out = os.path.join(sub_dir, "report.json")
            r3 = subprocess.run(py + ["census", cat_dir, "--out", nested_out], capture_output=True, text=True)
            self.assertEqual(r3.returncode, 2)
            self.assertIn("inside input folder", r3.stderr)
            self.assertFalse(os.path.exists(nested_out))

            missing_nested = os.path.join(cat_dir, "missing_sub", "report.json")
            r3b = subprocess.run(py + ["census", cat_dir, "--out", missing_nested], capture_output=True, text=True)
            self.assertEqual(r3b.returncode, 2)
            self.assertIn("inside input folder", r3b.stderr)

            # Relative path with '..' and trailing slash still resolves inside <dir> -> exit 2
            trav_out = os.path.join(sibling_dir, "..", "cats", "sub", "report.json")
            r3c = subprocess.run(
                py + ["resolve", cat_dir + os.sep, "--out", trav_out],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r3c.returncode, 2)
            self.assertIn("inside input folder", r3c.stderr)
            self.assertFalse(os.path.exists(nested_out))

            # Windows case and separator differences still match <dir> -> exit 2
            if os.name == "nt":
                mixed_out = os.path.join(cat_dir, "SUB", "report.json").replace("\\", "/")
                r3d = subprocess.run(py + ["census", cat_dir, "--out", mixed_out], capture_output=True, text=True)
                self.assertEqual(r3d.returncode, 2)
                self.assertIn("inside input folder", r3d.stderr)
                self.assertFalse(os.path.exists(nested_out))
                # Different drive letters must not raise ValueError
                self.assertFalse(pc._is_under_dir(r"Z:\elsewhere\report.json", r"C:\cats"))

            # Symlink outside <dir> pointing at a file inside <dir> -> exit 2, target untouched
            link_out = os.path.join(sibling_dir, "link.0drp")
            try:
                os.symlink(cat_file, link_out)
            except (OSError, NotImplementedError):
                link_out = None
            if link_out is not None:
                r_sym = subprocess.run(py + ["census", cat_dir, "--out", link_out], capture_output=True, text=True)
                self.assertEqual(r_sym.returncode, 2)
                self.assertIn("inside input folder", r_sym.stderr)
                with open(cat_file, "rb") as fh:
                    self.assertEqual(fh.read(), raw_0drp)

            # inspect <file> --out <file> -> exit 2, file untouched
            r4 = subprocess.run(py + ["inspect", cat_file, "--out", cat_file], capture_output=True, text=True)
            self.assertEqual(r4.returncode, 2)
            self.assertIn("overwrite", r4.stderr)
            with open(cat_file, "rb") as fh:
                self.assertEqual(fh.read(), raw_0drp)

            # resolve <dir> --names-file <names> --out <names> -> exit 2, names untouched
            names_file = os.path.join(sibling_dir, "names.txt")
            with open(names_file, "w", encoding="utf-8") as fh:
                fh.write("synth_body_full_03\n")
            r5 = subprocess.run(
                py + ["resolve", cat_dir, "--names-file", names_file, "--out", names_file],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r5.returncode, 2)
            self.assertIn("overwrite", r5.stderr)
            with open(names_file, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "synth_body_full_03\n")

            # census <dir> --out <sibling_dir>/report.json (shares string prefix "<dir>_out") -> exit 0
            sibling_out = os.path.join(sibling_dir, "report.json")
            r6 = subprocess.run(py + ["census", cat_dir, "--out", sibling_out], capture_output=True, text=True)
            self.assertEqual(r6.returncode, 0, r6.stderr)
            with open(sibling_out, encoding="utf-8") as fh:
                rep = json.load(fh)
            self.assertEqual(rep["catalogues"], 1)
            self.assertEqual(rep["presets"], 5)


if __name__ == "__main__":
    unittest.main()
