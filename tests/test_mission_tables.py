"""Synthetic tests for tools/mission_tables.py (no game data: every byte below is built here)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "mission_tables.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mission_tables as M  # noqa: E402


def record(key, kind, offset, count, col_b=1, band=0):
    """One 32-byte record. The offset sits in the high half of the dword at +16."""
    word = (kind & 0xFFFF) | ((offset & 0xFFFF) << 16)
    return struct.pack("<8I", key, 0, col_b, 0, word, band, (count << 16) | count, 0)


def element(archetype, pos, rot, link, hash_value=None, extra=0):
    """One 64-byte pool element: hash, 3 zeros, pos(x,y,z,x), rot(x,y,z,x), hash, link, extra, 0."""
    head = struct.pack("<4I", archetype, 0, 0, 0)
    pos_bytes = struct.pack("<4f", pos[0], pos[1], pos[2], pos[0])
    rot_bytes = struct.pack("<4f", rot[0], rot[1], rot[2], rot[0])
    tail = struct.pack("<4I", archetype if hash_value is None else hash_value, link, extra, 0)
    return head + pos_bytes + rot_bytes + tail


def blob_with(records, elements):
    """records: [(address, bytes)]; elements: [(address, bytes)]. Returns a zero-padded bytearray."""
    end = max([address + len(data) for address, data in records + elements] + [0])
    blob = bytearray(end)
    for address, data in records + elements:
        blob[address:address + len(data)] = data
    return blob


def slot112(key, pos, rot):
    """One 112-byte pair slot: u32 key, 12 zero bytes, pos (x,y,z,x), rot (x,y,z,x), 64 tail bytes."""
    return (struct.pack("<I", key) + bytes(12)
            + struct.pack("<4f", pos[0], pos[1], pos[2], pos[0])
            + struct.pack("<4f", rot[0], rot[1], rot[2], rot[0])
            + bytes(64))


def psin_with_blocks(blocks):
    """A minimal PSIN container: block table + blocks + one empty LGID section, no LSCH/LRSC.

    blocks: [(kind, bytes)] in block-table order. Row 0 is the header's version/table_bytes dword pair,
    exactly as in a measured container, so a block number of 1 is the first row.
    Returns (blob, [base per block]).
    """
    rows = len(blocks)
    pad_at = 8 + 16 * rows
    pad_end = (pad_at + 15) & ~15
    bases, at = [], pad_end
    for _kind, data in blocks:
        bases.append(at)
        at += (len(data) + 15) & ~15
    first = at
    blob = bytearray(first + 8)
    blob[0:4] = b"PSIN"
    struct.pack_into(">I", blob, 4, first)
    for index, (kind, data) in enumerate(blocks):
        row = 8 + 16 * index
        struct.pack_into("<II", blob, row, kind, len(data))
        blob[row + 8:row + 16] = b"\xaa" * 8
    struct.pack_into("<I", blob, pad_at, 0x70707070)
    blob[pad_at + 4:pad_end] = b"\x70" * (pad_end - pad_at - 4)
    for (_kind, data), base in zip(blocks, bases):
        blob[base:base + len(data)] = data
    blob[first:first + 4] = b"LGID"
    struct.pack_into(">I", blob, first + 4, 8)
    return bytes(blob), bases


def good_element(archetype=0x11223344, pos=(1.5, -2.25, 3.75), rot=(0.25, 0.5, -0.75),
                 link=0x55667788, extra=0):
    return element(archetype, pos, rot, link, extra=extra)


class TestRecord(unittest.TestCase):
    def test_big_endian_offset_half(self):
        # The measured bytes of a real record: 03 00 80 07. kind is 3 and the offset is 0x780.
        raw = struct.pack("<8I", 0x3B643C5E, 0, 1, 0, 0x07800003, 0, 0x00010001, 0)
        self.assertEqual(raw[16:20], bytes((0x03, 0x00, 0x80, 0x07)))
        decoded = M.decode_record(raw, 0)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["key"], 0x3B643C5E)
        self.assertEqual(decoded["kind"], 3)
        self.assertEqual(decoded["poolOffset"], 0x780)   # not 0x8007: the u16 is still little-endian
        self.assertEqual(decoded["count"], 1)
        self.assertEqual(decoded["band"], 0)
        self.assertEqual(decoded["colB"], 1)

    def test_duplicated_count_and_zero_fields_are_required(self):
        self.assertIsNone(M.decode_record(bytes(16), 0))   # shorter than a record
        bad_count = bytearray(record(0x12345678, 1, 0x100, 1))
        struct.pack_into("<I", bad_count, 24, 0x00010002)
        self.assertIsNone(M.decode_record(bytes(bad_count), 0))
        bad_zero = bytearray(record(0x12345678, 1, 0x100, 1))
        struct.pack_into("<I", bad_zero, 28, 1)
        self.assertIsNone(M.decode_record(bytes(bad_zero), 0))

    def test_joaat_folds_case(self):
        self.assertEqual(M.joaat("AlphaBeta"), M.joaat("alphabeta"))
        self.assertEqual(M.joaat("ALPHABETA"), M.joaat("alphabeta"))
        # An independent transcription of Jenkins one-at-a-time, seeded at 0.
        def reference(text):
            value = 0
            for byte in text.lower().encode("latin-1"):
                value = (value + byte) & 0xFFFFFFFF
                value = (value + (value << 10)) & 0xFFFFFFFF
                value ^= value >> 6
            value = (value + (value << 3)) & 0xFFFFFFFF
            value ^= value >> 11
            return (value + (value << 15)) & 0xFFFFFFFF
        for name in ("", "StateOne", "RoadJob", "a_b-c.d"):
            self.assertEqual(M.joaat(name), reference(name))


class TestElement(unittest.TestCase):
    def test_round_trip(self):
        state_one = 0x11111111
        state_two = 0x22222222
        blob = blob_with(
            [(0x100, record(state_one, 7, 0x1C0, 1)),
             (0x120, record(state_two, 7, 0x200, 1))],
            [(0x1C0, good_element(archetype=0xAABBCCDD, link=0x01020304)),
             (0x200, good_element(archetype=0x99887766, link=0x05060708,
                                  pos=(-10.0, 20.0, 30.0), rot=(0.0, 1.0, 2.0)))],
        )
        result = M.parse_blob(bytes(blob), "round-trip.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["state_keys"], 2)
        self.assertEqual(result["entities"], 2)
        self.assertEqual(result["unresolved_records"], 0)
        self.assertEqual(result["skipped_slots"], 0)
        by_archetype = {entity["archetype"]: entity for entity in result["entity_list"]}
        self.assertEqual(set(by_archetype), {0xAABBCCDD, 0x99887766})
        first = by_archetype[0xAABBCCDD]
        self.assertEqual(first["link"], 0x01020304)
        self.assertEqual([round(value, 4) for value in first["pos"]], [1.5, -2.25, 3.75])
        self.assertEqual([round(value, 4) for value in first["rot"]], [0.25, 0.5, -0.75])
        second = by_archetype[0x99887766]
        self.assertEqual([round(value, 4) for value in second["pos"]], [-10.0, 20.0, 30.0])

    def test_resolves_later_slots_when_first_slot_is_missing(self) -> None:
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 2)),
             (0x120, record(0x22222222, 7, 0x200, 1))],
            [(0x1C0, good_element(archetype=0xAABBCCDD, pos=(10.0, 20.0, 30.0))),
             (0x200, good_element(archetype=0x99887766))],
        )

        result = M.parse_blob(bytes(blob), "missing-first-slot.bin")

        self.assertEqual((result["records"], result["entities"]), (2, 2))
        self.assertEqual(result["skipped_slots"], 1)
        self.assertEqual(result["unresolved_records"], 0)
        first = result["entity_list"][0]
        self.assertEqual((first["offset"], first["archetype"], first["pos"]),
                         (0x1C0, 0xAABBCCDD, [10.0, 20.0, 30.0]))

    def test_mismatched_duplicated_position_is_not_an_entity(self):
        broken = bytearray(good_element(archetype=0xDEADBEEF, link=0x00000001))
        struct.pack_into("<I", broken, 0x1C, 0x7FC00000)  # pos.x copy no longer equals pos.x
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x1C0, 1))],
            [(0x180, bytes(broken)),
             (0x1C0, good_element(archetype=0xCAFEBABE, link=0x00000002))],
        )
        result = M.parse_blob(bytes(blob), "mismatch.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["skipped_slots"], 1)
        self.assertEqual(result["unresolved_records"], 1)
        self.assertEqual([entity["archetype"] for entity in result["entity_list"]], [0xCAFEBABE])

    def test_mismatched_duplicated_rotation_is_not_an_entity(self):
        broken = bytearray(good_element(archetype=0xDEADBEEF))
        struct.pack_into("<I", broken, 0x2C, 0x40000000)  # rot.x copy no longer equals rot.x
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, bytes(broken))],
        )
        result = M.parse_blob(bytes(blob), "mismatch-rot.bin")
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_slots"], 2)

    def test_extra_tail_hash_does_not_hide_an_entity(self):
        # Some elements carry a second hash in the third-last dword; it must not invalidate the element.
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element(archetype=0x1234ABCD, link=0x0BADF00D, extra=0xFEEDFACE))],
        )
        result = M.parse_blob(bytes(blob), "extra.bin")
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["entity_list"][0]["extra"], 0xFEEDFACE)

    def test_multi_slot_record_with_invalid_first_slot_resolves_later_slots(self):
        broken = bytearray(good_element(archetype=0xDEADBEEF))
        struct.pack_into("<I", broken, 0x1C, 0x7FC00000)
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 2)),
             (0x120, record(0x22222222, 7, 0x200, 1))],
            [(0x180, bytes(broken)),
             (0x1C0, good_element(archetype=0xCAFEBABE, link=0x00000002)),
             (0x200, good_element(archetype=0x13572468, link=0x00000003))],
        )
        result = M.parse_blob(bytes(blob), "partial-slots.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 2)
        self.assertEqual(result["skipped_slots"], 1)
        self.assertEqual(result["unresolved_records"], 0)
        self.assertEqual([e["archetype"] for e in result["entity_list"]], [0xCAFEBABE, 0x13572468])

    def test_rejected_single_record_window_does_not_skip_overlapping_table(self):
        # A 16-byte prefix at 0x100 combined with the first 16 bytes of a real colB=0 record at 0x110
        # forms a 32-byte window that decodes as a single record (key=0x999999, colB=99, count=0) and
        # is rejected by the lone-record filter. Rejecting 0x100 must advance by 4 bytes, not 32, so
        # the real two-record table at 0x110..0x150 is still found.
        false_prefix = struct.pack("<4I", 0x00999999, 0, 99, 0)
        blob = blob_with(
            [(0x100, false_prefix),
             (0x110, record(0x11111111, 7, 0x180, 1, col_b=0)),
             (0x130, record(0x22222222, 7, 0x1C0, 1, col_b=0))],
            [(0x180, good_element(archetype=0xAAAA1111)),
             (0x1C0, good_element(archetype=0xBBBB2222))],
        )
        result = M.parse_blob(bytes(blob), "overlap.bin")
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 2)


class TestBlockIndexed(unittest.TestCase):
    """The third resolution: kind is a 1-based block number and the u64 at +16 is a PSIN pointer."""

    def test_stride_comes_from_the_chunk_slices(self):
        blocks = [(0x11, 0x100, 0x1000), (0x22, 0x1100, 0x1000)]
        records = [
            {"kind": 2, "band": 0, "poolOffset": 0x0000, "count": 2},
            {"kind": 2, "band": 0, "poolOffset": 0x0E00, "count": 2},
        ]
        # 0x0E00 >> 4 = 0xE0 = 2 * 112, so the two slices tile the block at a 112-byte stride.
        self.assertEqual(M.chunk_stride(records, blocks), 112)
        self.assertIsNone(M.chunk_stride(records, []))
        broken = [dict(records[0], count=5), dict(records[1])]   # 0xC0 / 5 is not an integer
        self.assertIsNone(M.chunk_stride(broken, blocks))

    def test_band_carries_the_offset_high_bits(self):
        blocks = [(0x11, 0x0000, 0x40), (0x22, 0x0040, 0x40), (0x33, 0x0080, 0x2000)]
        blob = bytearray(0x2100)
        base = blocks[2][1]
        blob[base + 0x1070:base + 0x1070 + 112] = slot112(0xAAAA0001, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
        blob[base + 0x10E0:base + 0x10E0 + 112] = slot112(0xAAAA0002, (4.0, 5.0, 6.0), (0.4, 0.5, 0.6))
        elements = M.scan_elements(bytes(blob), len(blob))
        record = {"kind": 3, "band": 1, "poolOffset": 0x0700, "count": 2}   # 0x1070, not 0x0070
        found, mode, skipped = M.resolve(record, elements, None, blocks, 112)
        self.assertEqual(mode, "block")
        self.assertEqual(len(found), 2)
        self.assertEqual(skipped, 0)
        self.assertEqual([address for address, _element in found], [base + 0x1070, base + 0x10E0])

    def test_block_indexed_slots_resume_after_the_first_slot(self) -> None:
        blocks = [(0x11, 0x100, 0x1000)]
        record = {"kind": 1, "band": 0, "poolOffset": 0, "count": 2}
        address = 0x100 + 112
        entity = {"pos": [1.0, 2.0, 3.0], "archetype": 0x1000}

        found, mode, skipped = M.resolve(record, {address: entity}, None, blocks, 112)

        self.assertEqual((found, mode, skipped), ([(address, entity)], "block", 1))

    def test_psin_container_resolves_a_112_byte_slot_array(self):
        slots = bytearray(0x1240)
        for index, (position, rotation) in enumerate((
                ((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
                ((4.0, 5.0, 6.0), (0.4, 0.5, 0.6)),
                ((7.0, 8.0, 9.0), (0.7, 0.8, 0.9)),
                ((-1.0, -2.0, -3.0), (-0.1, -0.2, -0.3)))):
            at = 0x1070 + index * 112
            slots[at:at + 112] = slot112(0x1000 + index, position, rotation)
        block_records = (record(0x11111111, 3, 0x0700, 2, band=1)     # byte offset 0x1070
                         + record(0x22222222, 3, 0x1500, 2, band=1))    # byte offset 0x1150
        blob, bases = psin_with_blocks([(0x11111111, bytes(0x20)),
                                        (0x22222222, block_records),
                                        (0x33333333, bytes(slots))])
        result = M.parse_blob(blob, "block-indexed.bin")
        self.assertEqual(result["record_blocks"], 1)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["unresolved_records"], 0)
        self.assertEqual(result["skipped_slots"], 0)
        self.assertEqual(result["entities"], 4)
        self.assertEqual([entity["offset"] for entity in result["entity_list"]],
                         [bases[2] + 0x1070, bases[2] + 0x10E0,
                          bases[2] + 0x1150, bases[2] + 0x11C0])


class TestBrokenInput(unittest.TestCase):
    def test_truncated_element_is_skipped_without_a_crash(self):
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x1C0, 1)),
             (0x120, record(0x22222222, 7, 0x1C0, 1))],
            [(0x1C0, good_element())],
        )
        truncated = bytes(blob[:0x1C0 + 40])   # the element loses its last 24 bytes
        result = M.parse_blob(truncated, "truncated.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_slots"], 2)
        self.assertEqual(result["trailing_bytes"], len(truncated) - 0x140)
        self.assertEqual(result["skipped_bytes"], len(truncated) - 2 * M.RECORD_SIZE)

    def test_shorter_than_one_record_is_all_skipped(self):
        result = M.parse_blob(b"\x01" * 20, "tiny.bin")
        self.assertEqual(result["records"], 0)
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_bytes"], 20)
        self.assertEqual(result["trailing_bytes"], 20)

    def test_psin_small_pre_section_does_not_scan_section_chain(self):
        # A valid PSIN container whose first section starts at offset 16 (< RECORD_SIZE) must clamp
        # scan_bytes to 16 rather than falling back to len(blob) and scanning inside the section chain.
        section_body = (record(0x11111111, 7, 0x60, 1)
                        + record(0x22222222, 7, 0x60, 1)
                        + bytes(16)
                        + good_element(archetype=0xCAFEBABE))
        blob = bytearray(16 + 8 + len(section_body))
        blob[0:4] = b"PSIN"
        struct.pack_into(">I", blob, 4, 16)
        blob[16:20] = b"LGID"
        struct.pack_into(">I", blob, 20, 8 + len(section_body))
        blob[24:] = section_body
        result = M.parse_blob(bytes(blob), "small-pre.psin")
        self.assertEqual(result["scan_bytes"], 16)
        self.assertEqual(result["records"], 0)
        self.assertEqual(result["entities"], 0)

    def test_offset_outside_the_file_is_a_record_but_not_an_entity(self):
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0xFFF0, 1))],
            [(0x180, good_element(archetype=0xABCDEF01, link=0x00000003))],
        )
        result = M.parse_blob(bytes(blob), "outside.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["state_keys"], 2)
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["unresolved_records"], 1)
        self.assertEqual(result["entity_list"][0]["archetype"], 0xABCDEF01)
        self.assertIn(0x22222222, {state["key"] for state in result["states"]})

    def test_rejected_lone_record_does_not_skip_adjacent_table(self):
        # 24-byte prefix + valid 2-record chunk at +0x18: bytes 0x00..0x1F decode as a rejected lone
        # false positive, so advancing by 32 bytes instead of 4 used to jump past +0x18 and drop the table.
        prefix = struct.pack("<6I", 0x12345678, 0, 0, 0, 0x99990001, 0)
        rec0 = struct.pack("<8I", 0x00010001, 0, 1, 0, (0x0000 << 16) | 3, 0, (1 << 16) | 1, 0)
        rec1 = struct.pack("<8I", 0x11223344, 0, 1, 0, (0x0080 << 16) | 3, 0, (1 << 16) | 1, 0)
        result = M.parse_blob(prefix + rec0 + rec1, "after_false_positive.bin")
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["records"], 2)


class TestCli(unittest.TestCase):
    def test_directory_is_sorted_and_json_is_machine_readable(self):
        one = bytes(blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element())],
        ))
        with tempfile.TemporaryDirectory() as folder:
            for name in ("b.bin", "a.bin"):
                with open(os.path.join(folder, name), "wb") as handle:
                    handle.write(one)
            with open(os.path.join(folder, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("not a stream")
            result = M.collect(folder)
            self.assertEqual([record["file"] for record in result], ["a.bin", "b.bin"])
            completed = subprocess.run(
                [sys.executable, "-I", TOOL, folder, "--json"],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual([record["file"] for record in payload["files"]], ["a.bin", "b.bin"])
            self.assertEqual(payload["summary"]["files"], 2)
            self.assertEqual(payload["summary"]["records"], 4)
            self.assertEqual(payload["summary"]["entities"], 2)
            self.assertRegex(payload["files"][0]["entity_list"][0]["archetype"], r"^[0-9a-f]{8}$")

    def test_limit_stops_after_the_requested_file_count(self):
        one = bytes(blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element())],
        ))
        with tempfile.TemporaryDirectory() as folder:
            for name in ("a.bin", "b.bin", "c.bin"):
                with open(os.path.join(folder, name), "wb") as handle:
                    handle.write(one)
            completed = subprocess.run(
                [sys.executable, "-I", TOOL, folder, "--limit", "2"],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(completed.stdout.count(" records="), 2)
            self.assertIn("2 files,", completed.stdout)

    def test_a_single_file_without_the_magic_is_refused(self):
        # A lone argument is a PSIN file; a folder is where the bare carved .bin streams live. Naming a file
        # that is neither used to end in "1 files, 0 records" and exit 0, so a wrong path looked like a run.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "not-a-container.bin")
            with open(path, "wb") as handle:
                handle.write(b"\x01" * 64)
            completed = subprocess.run([sys.executable, "-I", TOOL, path], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("not a PSIN container", completed.stderr)
            self.assertEqual(completed.stdout, "")

    def test_a_folder_still_scans_a_stream_without_the_magic(self):
        # Negative control: the same bytes in a folder are a carved stream and are still counted.
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "bare.bin"), "wb") as handle:
                handle.write(b"\x01" * 64)
            completed = subprocess.run([sys.executable, "-I", TOOL, folder], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("1 files, 0 records", completed.stdout)
            self.assertIn("skipped bytes 64", completed.stdout)


if __name__ == "__main__":
    unittest.main()
