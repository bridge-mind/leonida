"""Synthetic unit and CLI tests for tools/odrp_7a.py.

    python -I tests/test_odrp_7a.py

Every byte in this file is constructed synthetically. No game file, carved payload,
asset name, path string, or asset hash is used.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "odrp_7a.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import odrp_7a  # noqa: E402


def _align16(n):
    return (n + 15) & ~15


def build_standalone_resource(
    *,
    root_objects=4,
    enums=1,
    as_map=False,
    entity_count=3,
    include_physical_block=False,
):
    """Build a complete synthetic standalone `0DRP`/`0x7a` resource."""
    # Struct 0: custom root struct (or CMapData when as_map=True), 32 bytes, 3 members:
    #   member 0: scalar field at offset 0
    #   member 1: 0x100 array-element descriptor (field_off may exceed struct size)
    #   member 2: 0x5c array member at offset 16 targeting descriptor index 1
    root_hash = odrp_7a.MAPDATA_STRUCT if as_map else 0x11223344
    structs_spec = [
        (
            root_hash,
            32,
            [
                (0xAABBCC01, 0, 0x04, 0, 0, 0),
                (odrp_7a.ARRAY_ELEMENT_HASH, 64, 0x04, 0, 0, 0),
                (0xAABBCC02, 16, odrp_7a.ARRAY_MEMBER_TYPE, 0, 1, 0),
            ],
        )
    ]
    if as_map:
        structs_spec.append(
            (
                odrp_7a.ENTITY_STRUCT,
                48,
                [
                    (0xAABBCC10, 0, 0x04, 0, 0, 0),
                    (0xAABBCC11, 16, 0x04, 0, 0, 0),
                ],
            )
        )

    nst = len(structs_spec)
    blocks_spec = [(root_hash, 32, True)]
    if as_map:
        blocks_spec.append((odrp_7a.ENTITY_STRUCT, 48 * entity_count, True))
    if include_physical_block:
        blocks_spec.append((0x778899AA, 128, False))
    nbl = len(blocks_spec)

    pos = odrp_7a.STANDALONE_HEADER_SIZE  # 0x60
    block_offsets = []
    for _bhash, bsize, is_virtual in blocks_spec:
        if is_virtual:
            pos = _align16(pos)
            block_offsets.append(pos)
            pos += bsize
        else:
            block_offsets.append(None)

    pos = _align16(pos)
    q38 = pos
    pos += 16 * nbl

    member_offsets = []
    for _shash, _ssize, members in structs_spec:
        pos = _align16(pos)
        member_offsets.append(pos)
        pos += 16 * len(members)

    pos = _align16(pos)
    q28 = pos
    pos += 32 * nst

    q30 = 0
    if enums > 0:
        pos = _align16(pos)
        q30 = pos
        pos += 24 * enums

    total_size = _align16(pos)
    buf = bytearray(total_size)

    # +0x00..+0x0f: 8 zero bytes + virtual self/page-map pointer
    struct.pack_into("<Q", buf, 0x08, odrp_7a.VIRTUAL_BASE + 0x10)
    # +0x10..+0x5f: 80-byte 0DRP/0x7a header
    buf[0x10:0x14] = odrp_7a.SIGNATURE
    struct.pack_into("<I", buf, 0x14, odrp_7a.VERSION)
    struct.pack_into("<II", buf, 0x20, root_objects, 0)
    struct.pack_into("<Q", buf, 0x28, odrp_7a.VIRTUAL_BASE + q28)
    struct.pack_into("<Q", buf, 0x30, (odrp_7a.VIRTUAL_BASE + q30) if enums > 0 else 0)
    struct.pack_into("<Q", buf, 0x38, odrp_7a.VIRTUAL_BASE + q38)
    struct.pack_into("<Q", buf, 0x40, 0)
    struct.pack_into("<4H", buf, 0x48, nst, enums, nbl, 0)
    struct.pack_into("<Q", buf, 0x50, 0)

    for i, ((shash, ssize, members), moff) in enumerate(zip(structs_spec, member_offsets)):
        struct.pack_into(
            "<IIIIQIHH",
            buf,
            q28 + 32 * i,
            shash,
            0x00010001,
            0,
            0,
            odrp_7a.VIRTUAL_BASE + moff,
            ssize,
            0,
            len(members),
        )
        for j, mtuple in enumerate(members):
            struct.pack_into("<IIBBHI", buf, moff + 16 * j, *mtuple)

    for i, ((bhash, bsize, is_virtual), boff) in enumerate(zip(blocks_spec, block_offsets)):
        ptr = (odrp_7a.VIRTUAL_BASE + boff) if is_virtual else (odrp_7a.PHYSICAL_BASE + 0x2000)
        struct.pack_into("<IIQ", buf, q38 + 16 * i, bhash, bsize, ptr)

    return bytes(buf)


def build_lrsc_payload(root_block=2, structs=2, enums=1, blocks=3):
    """Build an exact 80-byte `PSIN` `LRSC` `0DRP`/`0x7a` payload."""
    pay = bytearray(odrp_7a.LRSC_PAYLOAD_SIZE)
    pay[0x00:0x04] = odrp_7a.SIGNATURE
    struct.pack_into("<I", pay, 0x04, odrp_7a.VERSION)
    # +0x08..+0x0f stays 8 zero bytes (+0x0c is 0)
    struct.pack_into("<II", pay, 0x10, root_block, 0)
    pay[0x18:0x38] = odrp_7a.LRSC_PTR_FILL_32
    struct.pack_into("<4H", pay, 0x38, structs, enums, blocks, 0)
    pay[0x40:0x48] = odrp_7a.LRSC_TAIL_FILL_8
    # +0x48..+0x4f stays 8 zero bytes
    return bytes(pay)


def build_psin_container(
    *,
    structs=2,
    enums=1,
    block_sizes=(32, 48, 16),
    root_block=2,
    include_lrsc=True,
    lrsc_override=None,
    lsns_payload_size=32,
):
    """Build a synthetic `PSIN` container with `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC`, `CHKS`."""
    out = bytearray(b"PSIN\x00\x00\x00\x00")
    for bsize in block_sizes:
        out += struct.pack("<II", 261, bsize) + odrp_7a.BLOCK_ROW_FILL_8
    term_len = _align16(len(out) + 4) - len(out)
    out += b"\x70" * term_len
    for idx, bsize in enumerate(block_sizes):
        out += bytes([(idx + 1) & 0xFF]) * bsize
        pad = _align16(len(out)) - len(out)
        out += b"\x00" * pad

    first_section = len(out)
    out[4:8] = first_section.to_bytes(4, "big")

    def section(tag, payload):
        return tag + (8 + len(payload)).to_bytes(4, "big") + payload

    # LSCH: `structs` 32-byte headers (each declaring 1 member at +30) + `structs` 16-byte members
    lsch_headers = b"".join(
        struct.pack("<IIIIQIHH", 0x2000 + i, 1, 0, 0, 0, 16, 0, 1) for i in range(structs)
    )
    lsch_members = b"".join(struct.pack("<IIBBHI", 0x3000 + i, 0, 4, 0, 0, 0) for i in range(structs))
    out += section(b"LSCH", lsch_headers + lsch_members)

    # LENM: `enums` 24-byte headers (each declaring 2 values at +16) + `2 * enums` 16-byte items
    lenm_headers = b"".join(struct.pack("<IIIIII", 0x4000 + i, 1, 0, 0, 2, 0) for i in range(enums))
    lenm_items = bytes(16 * 2 * enums)
    out += section(b"LENM", lenm_headers + lenm_items)

    out += section(b"LSNS", bytes(lsns_payload_size))
    out += section(b"LGID", b"")
    if include_lrsc:
        lrsc_pay = (
            build_lrsc_payload(
                root_block=root_block,
                structs=structs,
                enums=enums,
                blocks=len(block_sizes),
            )
            if lrsc_override is None
            else lrsc_override
        )
        out += section(odrp_7a.LRSC_TAG, lrsc_pay)
    out += section(odrp_7a.CHKS_TAG, bytes(12))
    return bytes(out)


def build_bink2(size=64):
    buf = bytearray(size)
    buf[:4] = b"KB2n"
    struct.pack_into("<I", buf, 4, size - 8)
    struct.pack_into("<I", buf, 8, 1)
    struct.pack_into("<I", buf, 0x14, 64)
    struct.pack_into("<I", buf, 0x18, 64)
    return bytes(buf)


def build_rpf9(entries):
    """Build a synthetic RPF9 archive from `(kind, payload)` entries.

    `kind` is `"zlib"`, `"bink"`, or `"raw_gap"`. Zlib and Bink streams are padded to 16-byte
    alignment with zero bytes.
    """
    header = bytearray(0x100)
    header[:4] = b"9FPR"
    struct.pack_into("<I", header, 4, len(header))
    body = bytearray()
    offsets = []
    stored_sizes = []
    for kind, payload in entries:
        offsets.append(len(header) + len(body))
        if kind == "zlib":
            comp = zlib.compress(payload)
            stored_sizes.append(len(comp))
            body += comp
            body += bytes((-len(body)) % 16)
        elif kind == "bink":
            stored_sizes.append(len(payload))
            body += payload
            body += bytes((-len(body)) % 16)
        elif kind == "raw_gap":
            stored_sizes.append(len(payload))
            body += payload
            body += bytes((-len(body)) % 16)
        else:
            raise ValueError(kind)
    return bytes(header + body), offsets, stored_sizes


class PlacementDistinctionTest(unittest.TestCase):
    def test_standalone_meta_resource(self):
        blob = build_standalone_resource(root_objects=9, enums=2, as_map=False)
        rec = odrp_7a.classify_blob(blob)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["placement"], "standalone")
        self.assertEqual(rec["framing"], "resource")
        self.assertEqual(rec["version"], 122)
        self.assertTrue(rec["complete"])
        self.assertFalse(rec["needs_continuation"])
        self.assertTrue(rec["schema_checked"])
        self.assertEqual(rec["resource_kind"], "meta")
        self.assertEqual((rec["structs"], rec["enums"], rec["blocks"]), (1, 2, 1))
        self.assertEqual(rec["root_objects"], 9)
        self.assertIsNone(rec["root_block"])
        self.assertEqual(rec["entities"], 0)
        self.assertEqual(rec["error"], "")

    def test_standalone_map_resource_and_physical_block(self):
        blob = build_standalone_resource(
            root_objects=1,
            enums=0,
            as_map=True,
            entity_count=5,
            include_physical_block=True,
        )
        rec = odrp_7a.classify_blob(blob)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["placement"], "standalone")
        self.assertTrue(rec["complete"])
        self.assertEqual(rec["resource_kind"], "map")
        self.assertEqual((rec["structs"], rec["enums"], rec["blocks"]), (2, 0, 3))
        self.assertEqual(rec["virtual_blocks"], 2)
        self.assertEqual(rec["physical_blocks"], 1)
        self.assertEqual(rec["entities"], 5)

    def test_psin_container_lrsc_cross_checks_lsch_lenm_and_blocks(self):
        psin = build_psin_container(structs=3, enums=2, block_sizes=(16, 32, 48, 16), root_block=4)
        rec = odrp_7a.classify_blob(psin)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["placement"], "psin_lrsc")
        self.assertEqual(rec["framing"], "psin")
        self.assertEqual(rec["version"], 122)
        self.assertTrue(rec["complete"])
        self.assertTrue(rec["schema_checked"])
        self.assertIsNone(rec["resource_kind"])
        self.assertEqual((rec["structs"], rec["enums"], rec["blocks"]), (3, 2, 4))
        self.assertIsNone(rec["root_objects"])
        self.assertEqual(rec["root_block"], 4)
        self.assertIsNone(rec["entities"])
        self.assertEqual(rec["error"], "")

    def test_lrsc_section_payload_and_psin_tail_framings(self):
        pay = build_lrsc_payload(root_block=3, structs=5, enums=1, blocks=3)
        sec = odrp_7a.LRSC_TAG + (88).to_bytes(4, "big") + pay
        tail_stream = bytes(200) + sec + odrp_7a.CHKS_TAG + (20).to_bytes(4, "big") + bytes(12)

        r_pay = odrp_7a.classify_blob(pay)
        self.assertEqual((r_pay["placement"], r_pay["framing"], r_pay["complete"]), ("psin_lrsc", "lrsc_payload", True))
        self.assertEqual((r_pay["root_block"], r_pay["structs"], r_pay["enums"], r_pay["blocks"]), (3, 5, 1, 3))
        self.assertFalse(r_pay["schema_checked"])

        r_sec = odrp_7a.classify_blob(sec)
        self.assertEqual((r_sec["placement"], r_sec["framing"], r_sec["complete"]), ("psin_lrsc", "lrsc_section", True))
        self.assertEqual((r_sec["root_block"], r_sec["blocks"]), (3, 3))

        r_tail = odrp_7a.classify_blob(tail_stream)
        self.assertEqual((r_tail["placement"], r_tail["framing"], r_tail["complete"]), ("psin_lrsc", "psin_tail", True))
        self.assertEqual((r_tail["root_block"], r_tail["structs"], r_tail["blocks"]), (3, 5, 3))

    def test_two_placements_are_never_confused(self):
        standalone = build_standalone_resource(root_objects=7, enums=1)
        self.assertIsNone(odrp_7a.parse_psin_lrsc(standalone))

        # Stripping the 16-byte system prefix leaves 0DRP@+0x00 with 0x50000000 pointers rather
        # than 0xbb pointer-slot fill: must NOT be misclassified as an LRSC record.
        stripped = standalone[0x10:0x60]
        self.assertEqual(stripped[:4], b"0DRP")
        self.assertIsNone(odrp_7a.parse_standalone(stripped))
        self.assertIsNone(odrp_7a.parse_psin_lrsc(stripped))
        self.assertIsNone(odrp_7a.classify_blob(stripped))

        # Old synthetic shape with a count at +0x0c and zero pointer slots is neither a standalone
        # resource nor a valid PSIN LRSC record (+0x08..+0x0f is 0 and +0x18..+0x37 is 0xbb in LRSC).
        legacy_zero_at_0c = bytearray(0x50)
        legacy_zero_at_0c[:4] = b"0DRP"
        struct.pack_into("<I", legacy_zero_at_0c, 4, 122)
        struct.pack_into("<I", legacy_zero_at_0c, 0x0C, 7)
        self.assertIsNone(odrp_7a.classify_blob(bytes(legacy_zero_at_0c)))

        # LRSC root_block must satisfy 1 <= root_block <= blocks, and u16@+0x3e must be 0.
        bad_zero_root = bytearray(build_lrsc_payload(root_block=1, blocks=3))
        struct.pack_into("<I", bad_zero_root, 0x10, 0)
        self.assertIsNone(odrp_7a.parse_lrsc_payload(bytes(bad_zero_root)))

        bad_high_root = bytearray(build_lrsc_payload(root_block=1, blocks=3))
        struct.pack_into("<I", bad_high_root, 0x10, 4)
        self.assertIsNone(odrp_7a.parse_lrsc_payload(bytes(bad_high_root)))

        bad_reserved = bytearray(build_lrsc_payload(root_block=2, blocks=3))
        struct.pack_into("<H", bad_reserved, 0x3E, 1)
        self.assertIsNone(odrp_7a.parse_lrsc_payload(bytes(bad_reserved)))

    def test_standalone_schema_validation_catches_malformed_tables(self):
        valid = bytearray(build_standalone_resource(root_objects=2, enums=1))
        hdr = odrp_7a.parse_standalone_header(valid)
        q28 = hdr["tables"]["q28"]
        q38 = hdr["tables"]["q38"]
        moff = odrp_7a.virtual_offset(struct.unpack_from("<Q", valid, q28 + 16)[0])

        # 1. Scalar member offset >= struct size -> hard schema error (not continuation)
        bad_member = bytearray(valid)
        struct.pack_into("<I", bad_member, moff + 4, 32)  # ssize is 32
        res = odrp_7a.parse_standalone(bytes(bad_member))
        self.assertFalse(res["complete"])
        self.assertFalse(res["needs_continuation"])
        self.assertIn("offset 32 >= size 32", res["error"])

        # 2. Array member (0x5c) targeting non-0x100 member -> hard schema error
        bad_array = bytearray(valid)
        struct.pack_into("<H", bad_array, moff + 16 * 2 + 10, 0)  # member 0 is not 0x100
        res = odrp_7a.parse_standalone(bytes(bad_array))
        self.assertFalse(res["complete"])
        self.assertFalse(res["needs_continuation"])
        self.assertIn("0x100", res["error"])

        # 3. Only physical blocks (no virtual struct block) -> hard schema error
        phys_only = bytearray(valid)
        struct.pack_into("<Q", phys_only, q38 + 8, odrp_7a.PHYSICAL_BASE + 0x100)
        res = odrp_7a.parse_standalone(bytes(phys_only))
        self.assertFalse(res["complete"])
        self.assertFalse(res["needs_continuation"])
        self.assertIn("no virtual block", res["error"])

        # 4. Truncated before q28 struct table -> needs_continuation=True
        truncated = bytes(valid[:q28])
        res = odrp_7a.parse_standalone(truncated)
        self.assertFalse(res["complete"])
        self.assertTrue(res["needs_continuation"])

    def test_psin_mismatch_and_non_lrsc_psin(self):
        mismatch_pay = build_lrsc_payload(root_block=1, structs=9, enums=1, blocks=2)
        bad_psin = build_psin_container(
            structs=2,
            enums=1,
            block_sizes=(16, 16),
            lrsc_override=mismatch_pay,
        )
        res = odrp_7a.classify_blob(bad_psin)
        self.assertIsNotNone(res)
        self.assertEqual(res["placement"], "psin_lrsc")
        self.assertFalse(res["complete"])
        self.assertFalse(res["schema_checked"])
        self.assertIn("structs 9 != LSCH 2", res["error"])

        # A PSIN container without LRSC (e.g. LSCH+LENM+LSNS+LGID+CHKS or PMAP/PSCH) emits None.
        no_lrsc = build_psin_container(include_lrsc=False)
        self.assertIsNone(odrp_7a.classify_blob(no_lrsc))


class ArchiveAndCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

        self.meta_single = build_standalone_resource(root_objects=3, enums=1, as_map=False)
        self.map_full = build_standalone_resource(root_objects=1, enums=0, as_map=True, entity_count=4)
        # Split `map_full` at 0x80 so q38/q28 sit in the second zlib stream.
        self.map_part1 = self.map_full[:0x80]
        self.map_part2 = self.map_full[0x80:]
        self.assertTrue(odrp_7a.parse_standalone(self.map_part1)["needs_continuation"])

        self.psin_single = build_psin_container(structs=2, enums=1, block_sizes=(16, 32), root_block=2)
        self.psin_multi = build_psin_container(
            structs=3,
            enums=1,
            block_sizes=(32, 32, 32),
            root_block=3,
            lsns_payload_size=240,
        )
        # Split `psin_multi` into 3 contiguous 128-byte-ish chunks (>= 64 B each for zlib scanner).
        c1 = 128
        c2 = len(self.psin_multi) - 96
        self.psin_p1 = self.psin_multi[:c1]
        self.psin_p2 = self.psin_multi[c1:c2]
        self.psin_p3 = self.psin_multi[c2:]

        # Unjoined psin_tail stream preceded by a non-zero raw gap (emulating an uncompressed LSNS block).
        tail_pay = build_lrsc_payload(root_block=1, structs=4, enums=0, blocks=2)
        self.psin_tail_stream = (
            bytes([0x5A]) * 96
            + odrp_7a.LRSC_TAG
            + (88).to_bytes(4, "big")
            + tail_pay
            + odrp_7a.CHKS_TAG
            + (20).to_bytes(4, "big")
            + bytes(12)
        )
        self.no_lrsc_psin = build_psin_container(include_lrsc=False)
        self.zero_page = bytes(128)
        self.bink = build_bink2(64)

        entries = [
            ("zlib", self.meta_single),       # 0: standalone meta (1 stream)
            ("zlib", self.map_part1),         # 1: standalone map part 1
            ("zlib", self.map_part2),         # 2: standalone map part 2 (joined -> 2 streams)
            ("bink", self.bink),              # 3: KB2n skipped cleanly
            ("zlib", self.psin_single),       # 4: single-stream PSIN LRSC
            ("zlib", self.psin_p1),           # 5: 3-stream PSIN part 1
            ("zlib", self.psin_p2),           # 6: 3-stream PSIN part 2
            ("zlib", self.psin_p3),           # 7: 3-stream PSIN part 3 (joined -> 3 streams)
            ("raw_gap", b"\xaa" * 64),        # 8: unread gap before unjoined PSIN tail
            ("zlib", self.psin_tail_stream),  # 9: psin_tail (1 stream)
            ("zlib", self.no_lrsc_psin),      # 10: PSIN without LRSC (ignored)
            ("zlib", self.zero_page),         # 11: plain system page (ignored)
        ]
        rpf_bytes, self.offsets, self.stored_sizes = build_rpf9(entries)
        self.rpf = os.path.join(self.root, "synthetic.rpf")
        with open(self.rpf, "wb") as fh:
            fh.write(rpf_bytes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_rpf_joins_split_containers_and_distinguishes_all_records(self):
        rows, summary = odrp_7a.scan_rpf(self.rpf)
        self.assertEqual(summary["streams"], 11)
        expected_unread = ((-self.stored_sizes[7]) % 16) + 64
        self.assertEqual(summary["unread_bytes"], expected_unread)
        self.assertEqual(summary["records"], 5)
        self.assertEqual(summary["complete"], 5)
        self.assertEqual(summary["incomplete"], 0)
        self.assertEqual(summary["standalone"], 2)
        self.assertEqual(summary["standalone_complete"], 2)
        self.assertEqual(summary["standalone_meta"], 1)
        self.assertEqual(summary["standalone_map"], 1)
        self.assertEqual(summary["standalone_joined"], 1)
        self.assertEqual(summary["standalone_entities"], 4)
        self.assertEqual(summary["psin_lrsc"], 3)
        self.assertEqual(summary["psin_lrsc_complete"], 3)
        self.assertEqual(summary["psin_lrsc_joined"], 1)
        self.assertEqual(summary["psin_lrsc_schema_checked"], 2)
        self.assertEqual(summary["framings"], {"psin": 2, "psin_tail": 1, "resource": 2})

        self.assertEqual(
            [r["offset"] for r in rows],
            [self.offsets[0], self.offsets[1], self.offsets[4], self.offsets[5], self.offsets[9]],
        )
        self.assertEqual([r["streams"] for r in rows], [1, 2, 1, 3, 1])
        self.assertEqual(
            [r["placement"] for r in rows],
            ["standalone", "standalone", "psin_lrsc", "psin_lrsc", "psin_lrsc"],
        )
        self.assertEqual([r["framing"] for r in rows], ["resource", "resource", "psin", "psin", "psin_tail"])
        self.assertEqual(rows[1]["size"], len(self.map_full))
        self.assertEqual(rows[1]["stored"], self.stored_sizes[1] + self.stored_sizes[2])
        self.assertEqual(rows[3]["size"], len(self.psin_multi))
        self.assertEqual(rows[3]["root_block"], 3)
        self.assertEqual(rows[4]["root_block"], 1)

    def test_scan_limit_excludes_stream_extending_past_window(self):
        # Cut 1 byte before the end of stream 0's compressed body: stream 0 must be left out.
        limit = self.stored_sizes[0] - 1
        rows, summary = odrp_7a.scan_rpf(self.rpf, limit=limit)
        self.assertEqual(rows, [])
        self.assertEqual(summary["streams"], 0)
        self.assertEqual(summary["span"], limit)

    def test_scan_dir_joins_with_manifest_and_respects_max_inflate(self):
        folder = os.path.join(self.root, "carved")
        os.makedirs(folder)
        manifest_files = []

        # Write map_part1 and map_part2 with contiguous 16-byte-aligned archive offsets in index.json,
        # plus psin_single and an oversized .bin that exceeds max_inflate=512.
        off1 = 0x1000
        stored1 = 0x93
        off2 = _align16(off1 + stored1)
        stored2 = 0xA4
        off3 = 0x3000
        stored3 = 0x80
        off4 = 0x4000

        items = [
            (off1, stored1, self.map_part1),
            (off2, stored2, self.map_part2),
            (off3, stored3, self.psin_single),
        ]
        for off, st, data in items:
            with open(os.path.join(folder, f"{off:012x}.bin"), "wb") as fh:
                fh.write(data)
            manifest_files.append({"off": off, "stored": st, "size": len(data)})

        with open(os.path.join(folder, f"{off4:012x}.bin"), "wb") as fh:
            fh.write(bytes(600))
        with open(os.path.join(folder, f"{0x5000:012x}.bk2"), "wb") as fh:
            fh.write(self.bink)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"files": manifest_files}, fh)

        rows, summary = odrp_7a.scan_dir(folder, max_inflate=512)
        self.assertEqual(summary["mode"], "directory")
        self.assertEqual(summary["streams"], 3)
        self.assertEqual(summary["skipped_candidates"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["placement"], rows[0]["streams"], rows[0]["entities"]), ("standalone", 2, 4))
        self.assertEqual((rows[1]["placement"], rows[1]["framing"], rows[1]["complete"]), ("psin_lrsc", "psin", True))

    def test_single_file_cli_and_lf_line_endings(self):
        single_meta = os.path.join(self.root, "meta.bin")
        with open(single_meta, "wb") as fh:
            fh.write(self.meta_single)

        out_json = os.path.join(self.root, "single.json")
        proc = subprocess.run(
            [sys.executable, "-I", TOOL, single_meta, out_json, "--quiet"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out_json, "rb") as fh:
            raw_json = fh.read()
        self.assertNotIn(b"\r", raw_json)
        doc = json.loads(raw_json.decode("utf-8"))
        self.assertEqual(doc["summary"]["mode"], "file")
        self.assertEqual(doc["summary"]["standalone"], 1)
        self.assertEqual(doc["records"][0]["root_objects"], 3)

        out_csv = os.path.join(self.root, "rpf.csv")
        proc = subprocess.run(
            [sys.executable, "-I", TOOL, self.rpf, out_csv, "--quiet"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out_csv, "rb") as fh:
            raw_csv = fh.read()
        self.assertNotIn(b"\r", raw_csv)
        table = list(csv.reader(raw_csv.decode("utf-8").splitlines()))
        self.assertEqual(table[0], odrp_7a.COLUMNS)
        self.assertEqual(len(table), 6)

    def test_reports_can_be_replaced_inside_the_carve_folder(self):
        folder = os.path.join(self.root, "carved")
        os.mkdir(folder)
        source = os.path.join(folder, "00000001.bin")
        with open(source, "wb") as f:
            f.write(self.meta_single)
        for name in ("report.json", "report.csv"):
            with self.subTest(name=name):
                out = os.path.join(folder, name)
                with open(out, "wb") as f:
                    f.write(b"old report")
                result = subprocess.run([sys.executable, "-I", TOOL, folder, out, "--quiet"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                with open(out, "rb") as f:
                    self.assertNotEqual(f.read(), b"old report")
        with open(source, "rb") as f:
            self.assertEqual(f.read(), self.meta_single)

    def test_report_alias_to_a_binary_payload_is_rejected(self):
        folder = os.path.join(self.root, "carved")
        os.mkdir(folder)
        source = os.path.join(folder, "00000001.bin")
        alias = os.path.join(folder, "report.json")
        with open(source, "wb") as f:
            f.write(self.meta_single)
        try:
            os.link(source, alias)
        except (OSError, NotImplementedError) as e:
            self.skipTest(str(e))
        result = subprocess.run([sys.executable, "-I", TOOL, folder, alias, "--quiet"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("would overwrite", result.stderr)
        with open(source, "rb") as f:
            self.assertEqual(f.read(), self.meta_single)

    def test_cli_validation_and_error_handling(self):
        # Bad numeric flags exit 2 without creating output.
        for flag, value, message in (
            ("--start", "-1", "must be 0 or more"),
            ("--limit", "0", "must be at least 1"),
            ("--limit", "-5", "must be at least 1"),
            ("--max-inflate", "0", "must be at least 1"),
        ):
            out = os.path.join(self.root, "bad_num.json")
            proc = subprocess.run(
                [sys.executable, "-I", TOOL, self.rpf, out, "--quiet", flag, value],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn(message, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(out))

        # Missing source exits 1 without traceback.
        missing_out = os.path.join(self.root, "missing.json")
        proc = subprocess.run(
            [sys.executable, "-I", TOOL, os.path.join(self.root, "absent.rpf"), missing_out, "--quiet"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(os.path.exists(missing_out))

        # Unwritable/bad output paths and source aliasing exit 2.
        for bad_out in (
            self.root,
            os.path.join(self.root, "missing_parent", "out.json"),
            os.path.join(self.root, "out.txt"),
        ):
            proc = subprocess.run(
                [sys.executable, "-I", TOOL, self.rpf, bad_out, "--quiet"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

        json_src = os.path.join(self.root, "same.json")
        with open(json_src, "wb") as fh:
            fh.write(self.meta_single)
        alias_out = os.path.join(self.root, ".", "same.json")
        proc = subprocess.run(
            [sys.executable, "-I", TOOL, json_src, alias_out, "--quiet"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("output must not overwrite the source", proc.stderr)

        # --start / --limit rejected on directory and non-RPF single file.
        for non_rpf in (self.root, json_src):
            proc = subprocess.run(
                [sys.executable, "-I", TOOL, non_rpf, os.path.join(self.root, "x.json"), "--start", "0"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)

    def test_scan_rpf_closes_mmap_on_success_and_out_of_range_start(self):
        created = []
        real_mmap = odrp_7a.mmap.mmap

        def track_mmap(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            created.append(m)
            return m

        with open(self.rpf, "rb") as src:
            raw_bytes = src.read()
        rpf_ok = os.path.join(self.root, "ok.rpf")
        rpf_err = os.path.join(self.root, "err.rpf")
        for path in (rpf_ok, rpf_err):
            with open(path, "wb") as dst:
                dst.write(raw_bytes)

        try:
            with unittest.mock.patch.object(odrp_7a.mmap, "mmap", side_effect=track_mmap):
                odrp_7a.scan_rpf(rpf_ok)
                self.assertEqual(len(created), 1)
                self.assertTrue(created[-1].closed)
                os.remove(rpf_ok)
                self.assertFalse(os.path.exists(rpf_ok))

                with self.assertRaisesRegex(ValueError, "start offset is outside the archive"):
                    odrp_7a.scan_rpf(rpf_err, start=len(raw_bytes) + 1)
                self.assertEqual(len(created), 2)
                self.assertTrue(created[-1].closed)
                os.remove(rpf_err)
                self.assertFalse(os.path.exists(rpf_err))
        finally:
            for m in created:
                if not m.closed:
                    m.close()


if __name__ == "__main__":
    unittest.main()
