"""Synthetic tests for tools/audio_config.py (no game data)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_config.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import audio_config as A  # noqa: E402
import hashdict  # noqa: E402


def make_record(config_id, duration_ms, hash_a=0x11111111, hash_b=0x22222222, tail=b"\x00" * 22):
    """One 46-byte record: duration, then the marker block at +4."""
    b = bytearray(46)
    struct.pack_into("<I", b, 0, duration_ms)
    b[4:6] = b"\x0c\x01"
    struct.pack_into("<H", b, 6, config_id)
    b[8:12] = b"\x02\x00\x0c\x00"
    struct.pack_into("<I", b, 12, hash_a)
    struct.pack_into("<I", b, 16, hash_b)
    b[20:24] = b"\x00\x00\x02\x23"
    b[24:24 + len(tail)] = tail
    return bytes(b)


def write_parts(tmp, parts):
    """Write `parts` as 00.bin, 01.bin, ... so name order is part order."""
    for i, data in enumerate(parts):
        with open(os.path.join(tmp, f"{i:02d}.bin"), "wb") as f:
            f.write(data)


def located(rows):
    return [(row["file"], row["file_offset"]) for row in rows]


def without_location(rows):
    return [{k: v for k, v in row.items() if k not in ("file", "file_offset")} for row in rows]


BLOCKS = (1, 3, 24, 25, 46, 64, A.JOIN_BLOCK_BYTES)


class TestHash(unittest.TestCase):
    def test_joaat_vectors(self):
        self.assertEqual(A.joaat("test"), 0x3F75CCC1)
        self.assertEqual(A.joaat("Radio_Test"), 0xE663FCAE)
        self.assertEqual(A.joaat(""), 0)

    def test_normalized_name(self):
        self.assertEqual(A.normalized_name("Hello,  World!"), "hello_world")
        self.assertEqual(A.normalized_name("A.B-C"), "a_b_c")
        self.assertEqual(A.normalized_name("Already_ok"), "already_ok")

    def test_title_stream_id(self):
        title = "Hello,  World!"
        self.assertEqual(A.title_stream_id(title), A.joaat("hello_world") & A.STREAM_MASK)
        self.assertEqual(A.title_stream_id("Test Song"),
                         A.joaat(A.normalized_name("Test Song")) & A.STREAM_MASK)


class TestScan(unittest.TestCase):
    def test_single_record(self):
        blob = b"\x00\x00\x00\x00" + make_record(0xB422, 264_091, 0x49FC8153, 0xDAA8AC2F)
        got = A.scan(blob)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], 0xB422)
        self.assertEqual(got[0]["duration_ms"], 264_091)
        self.assertEqual(got[0]["duration_s"], 264.091)
        self.assertEqual(got[0]["hash_a"], 0x49FC8153)
        self.assertEqual(got[0]["hash_b"], 0xDAA8AC2F)
        self.assertEqual(got[0]["stream_id"], 0xDAA8AC2F & A.STREAM_MASK)
        self.assertEqual(got[0]["offset"], 8)

    def test_two_adjacent_records(self):
        blob = b"\x00" * 4 + make_record(0x1000, 100_000) + make_record(0x1001, 200_000)
        got = A.scan(blob)
        self.assertEqual([r["id"] for r in got], [0x1000, 0x1001])
        self.assertEqual([r["offset"] for r in got], [8, 54])

    def test_rejects_bad_marker_and_duration(self):
        wrong_marker = bytearray(make_record(0x1000, 100_000))
        wrong_marker[8] = 0
        low = make_record(0x1001, 1_000)
        high = make_record(0x1002, 3_000_000)
        blob = b"\x00" * 4 + bytes(wrong_marker) + low + high
        self.assertEqual(A.scan(blob), [])

    def test_truncated_record(self):
        # A partial record (marker present, third marker cut off) must not be reported.
        blob = b"\x00" * 4 + make_record(0x1000, 100_000)[:22]
        self.assertEqual(A.scan(blob), [])

    def test_join_across_streams(self):
        # A zlib block boundary can split a record. Split before the third marker ends.
        record = make_record(0x3000, 150_000)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(b"\x00" * 4 + record[:22])
            with open(os.path.join(tmp, "00000002.bin"), "wb") as f:
                f.write(record[22:])
            self.assertEqual(len(A.collect(tmp)), 0)
            self.assertEqual(len(A.collect(tmp, join=True)), 1)

    def test_non_joined_scan_processes_each_blob_before_reading_the_next(self):
        def parts(_src):
            yield "first.bin", make_record(0x1000, 100_000)
            self.assertEqual(scan.call_count, 1)
            yield "second.bin", make_record(0x1001, 200_000)

        with patch.object(A, "files_in", parts), patch.object(A, "scan", wraps=A.scan) as scan:
            rows = A.collect("unused")
        self.assertEqual([(row["file"], row["id"]) for row in rows],
                         [("first.bin", 0x1000), ("second.bin", 0x1001)])

    def test_join_marker_at_stream_boundary_uses_the_file_containing_it(self):
        record = make_record(0x3000, 150_000)
        name = "\U0001f3b5.bin"
        with tempfile.TemporaryDirectory() as tmp:
            for filename, data in (("a.bin", record[:4]), ("b.bin", b""), (name, record[4:])):
                with open(os.path.join(tmp, filename), "wb") as f:
                    f.write(data)
            rows = A.collect(tmp, join=True)
        self.assertEqual([(row["offset"], row["file"], row["file_offset"]) for row in rows],
                         [(4, name, 0)])


class TestJoin(unittest.TestCase):
    def assert_every_split(self, stream, markers, blocks=BLOCKS):
        """Split `stream` at every byte, alone and between empty parts; check both coordinates."""
        want = A.scan(stream)
        self.assertEqual([row["offset"] for row in want], markers)
        for split in range(len(stream) + 1):
            a, b = stream[:split], stream[split:]
            for parts, first, second in (((a, b), 0, 1), ((b"", a, b"", b, b""), 1, 3)):
                expected = [(f"{first:02d}.bin", m) if m < split else (f"{second:02d}.bin", m - split)
                            for m in markers]
                with tempfile.TemporaryDirectory() as tmp:
                    write_parts(tmp, parts)
                    for block in blocks:
                        with self.subTest(split=split, parts=len(parts), block=block), \
                                patch.object(A, "JOIN_BLOCK_BYTES", block):
                            rows = A.collect(tmp, join=True)
                            self.assertEqual(without_location(rows), want)
                            self.assertEqual(located(rows), expected)

    def test_every_split_of_junk_adjacent_records_and_gap(self):
        stream = (b"\x0c\x01junk" + make_record(0x10, 100_000) + make_record(0x11, 200_000)
                  + b"\x00" * 7 + make_record(0x12, 300_000))
        self.assert_every_split(stream, [10, 56, 109])

    def test_lookalikes_and_skip_window_at_every_split(self):
        real = make_record(0x30, 120_000)
        bad2, bad3 = bytearray(real), bytearray(real)
        bad2[9] = 0xFF
        bad3[22] = 0xFF
        near = (b"\x00" * 4 + bytes(bad2) + bytes(bad3) + make_record(0x31, 1_000)
                + b"\x0c\x02" + real)
        self.assert_every_split(near, [len(near) - 42], (1, A.JOIN_BLOCK_BYTES))
        # A valid record whose duration starts in the previous record's tail lies inside the
        # 46-byte step, so it is skipped in every layout, as in scan().
        hidden = make_record(0x41, 150_000)
        self.assertEqual(len(A.scan(hidden)), 1)
        skip = b"\x00" * 4 + make_record(0x40, 150_000, tail=b"\x00" * 18 + hidden[:4]) + hidden[4:]
        self.assert_every_split(skip, [8], (1, A.JOIN_BLOCK_BYTES))

    def test_marker_with_twenty_following_bytes_and_tiny_parts(self):
        record = make_record(0x20, 150_000)
        complete = b"\x00" * 4 + record[:24]       # marker at 8, then exactly 20 bytes
        self.assertEqual(A.scan(complete[:-1]), [])
        self.assertEqual(len(A.scan(complete)), 1)
        self.assertEqual(len(list(A.scan_stream([complete]))), 1)
        cases = (((complete,), [("00.bin", 8)]),
                 ((complete, b"", b"", b"\x00"), [("00.bin", 8)]),
                 ((record[:1], record[1:2], record[2:3], record[3:]), [("03.bin", 1)]),
                 ((record[:5], record[5:]), [("00.bin", 4)]),
                 ((b"", record[:2], record[2:]), [("02.bin", 2)]))
        for parts, expected in cases:
            with self.subTest(sizes=[len(p) for p in parts]), tempfile.TemporaryDirectory() as tmp:
                write_parts(tmp, parts)
                self.assertEqual(located(A.collect(tmp, join=True)), expected)

    def test_non_joined_offsets_are_local_and_join_handles_no_parts(self):
        record = make_record(0x50, 100_000)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(A.collect(tmp, join=True), [])
            write_parts(tmp, (b"\x00" * 10, b"\x00" * 3 + record))
            with open(os.path.join(tmp, "zz.txt"), "wb") as f:
                f.write(b"\x00" * 4 + record)
            for src, join, offset in ((tmp, False, 7), (tmp, True, 17),
                                      (os.path.join(tmp, "01.bin"), False, 7)):
                with self.subTest(src=os.path.basename(src), join=join):
                    rows = A.collect(src, join=join)
                    self.assertEqual([(r["file"], r["offset"], r["file_offset"]) for r in rows],
                                     [("01.bin", offset, 7)])

    def test_join_memory_is_bounded_by_the_read_block(self):
        record = make_record(0x70, 100_000)
        stream = (b"\x00" * ((1 << 19) - 23) + record) * 8    # 4 MiB, records straddle parts
        with tempfile.TemporaryDirectory() as tmp:
            write_parts(tmp, [stream[i:i + (1 << 19)] for i in range(0, len(stream), 1 << 19)])
            tracemalloc.start()
            try:
                with patch.object(A, "JOIN_BLOCK_BYTES", 1 << 16):
                    rows = A.collect(tmp, join=True)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        self.assertEqual(without_location(rows), A.scan(stream))
        self.assertEqual(len(rows), 8)
        self.assertLess(peak, 1 << 20)


class TestCli(unittest.TestCase):
    def test_missing_or_unreadable_inputs_are_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = os.path.join(tmp, "00000001.bin")
            with open(blob, "wb") as f:
                f.write(b"\x00" * 4 + make_record(0x2000, 123_456))
            missing = os.path.join(tmp, "missing")
            cases = (([missing], missing, "does not exist", 2),
                     ([tmp, "--names", os.path.join(tmp, "no_pool.txt")], "no_pool.txt", "name pool", 1),
                     ([tmp, "--names", tmp], tmp, "name pool", 1))          # a folder as the pool
            for args, path, message, expected in cases:
                with self.subTest(message=message, args=[os.path.basename(x) for x in args]):
                    p = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
                    self.assertEqual(p.returncode, expected)
                    self.assertIn(path, p.stderr)
                    self.assertIn(message, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
                    self.assertEqual(p.stdout, "")

    def test_unreadable_stream_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(b"\x00" * 4 + make_record(0x2000, 123_456))
            denied = PermissionError(13, "Permission denied", os.path.join(tmp, "00000001.bin"))
            with patch.object(sys, "argv", ["audio_config.py", tmp]), \
                    patch.object(A, "collect", side_effect=denied), self.assertRaises(SystemExit) as ctx:
                A.main()
            self.assertIn("00000001.bin: cannot read: Permission denied", str(ctx.exception.code))

    def test_json_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "Test Song"
            hb = A.joaat(A.normalized_name(name))
            blob = b"\x00" * 4 + make_record(0x2000, 123_456, hash_b=hb)
            with open(os.path.join(tmp, "00000001.bin"), "wb") as f:
                f.write(blob)
            pool = os.path.join(tmp, "names.txt")
            with open(pool, "wb") as f:
                f.write(name.encode() + b"\x00Other\x00")
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--names", pool],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(p.stdout)
            self.assertEqual(doc["summary"]["records"], 1)
            self.assertEqual(doc["summary"]["distinct_ids"], 1)
            self.assertEqual(doc["summary"]["named"], 1)
            self.assertEqual(doc["records"][0]["name"], name)
            self.assertEqual(doc["records"][0]["stream_id"], hb & A.STREAM_MASK)

    def test_name_pool_is_read_as_latin_1_like_hashdict(self):
        # A pool string is a name's bytes; hashdict.py and awc_names.py read them as latin-1 and the joaat rule is
        # the lowercased latin-1 byte hash (docs/audio_config.md), so one name has to hash alike in every tool.
        # "CAFÉ" is one byte 0xC9 for É, not its UTF-8 pair, and no record can carry a hash this tool would not see.
        with tempfile.TemporaryDirectory() as tmp:
            pool = os.path.join(tmp, "names.txt")
            with open(pool, "wb") as f:
                f.write(b"CAF\xc9\x00Other\x00")
            names = A.load_names(pool)
            self.assertEqual(names.get(hashdict.joaat(A.normalized_name("café"))), "CAFÉ")

    def test_invalid_ms_range_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = os.path.join(tmp, "00000001.bin")
            with open(blob, "wb") as f:
                f.write(b"\x00" * 4 + make_record(0x2000, 123_456))
            for extra in (["--min-ms", "0"], ["--min-ms", "-5"], ["--min-ms", "50000", "--max-ms", "10000"]):
                with self.subTest(extra=extra):
                    p = subprocess.run([sys.executable, "-I", TOOL, blob, *extra],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 2)
                    self.assertIn("require 1 <= min-ms <= max-ms", p.stderr)
                    self.assertEqual(p.stdout, "")

    def test_join_rejected_on_single_file_and_allowed_on_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = os.path.join(tmp, "00000001.bin")
            with open(blob, "wb") as f:
                f.write(b"\x00" * 4 + make_record(0x2000, 123_456))
            p_file = subprocess.run([sys.executable, "-I", TOOL, blob, "--join"],
                                    capture_output=True, text=True)
            self.assertEqual(p_file.returncode, 2)
            self.assertIn("--join applies to a directory, not a single file", p_file.stderr)
            self.assertEqual(p_file.stdout, "")

            p_dir = subprocess.run([sys.executable, "-I", TOOL, tmp, "--join", "--json"],
                                   capture_output=True, text=True)
            self.assertEqual(p_dir.returncode, 0, p_dir.stderr)
            self.assertEqual(json.loads(p_dir.stdout)["summary"]["records"], 1)

    def test_join_text_uses_local_offsets_and_json_keeps_global_ones(self):
        first, second = make_record(0x64, 100_000), make_record(0x65, 200_000)
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (("first.bin", b"\x00" * 128 + first[:4]),
                               ("middle.bin", first[4:] + b"\x00" * 40 + second)):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(data)
            text = subprocess.run([sys.executable, "-I", TOOL, tmp, "--join"],
                                  capture_output=True, text=True)
            doc = subprocess.run([sys.executable, "-I", TOOL, tmp, "--join", "--json"],
                                 capture_output=True, text=True)
        self.assertEqual(text.returncode, 0, text.stderr)
        self.assertEqual(text.stdout.splitlines(), [
            "middle.bin:     0  id=0x0064  dur=  100.000s  stream=0x02222222  0x22222222",
            "middle.bin:    86  id=0x0065  dur=  200.000s  stream=0x02222222  0x22222222",
            "2 records, 2 distinct ids, ids 0x0064-0x0065, 300.0 s total"])
        self.assertEqual(doc.returncode, 0, doc.stderr)
        rows = json.loads(doc.stdout)["records"]
        self.assertEqual([(r["file"], r["offset"], r["file_offset"]) for r in rows],
                         [("middle.bin", 132, 0), ("middle.bin", 218, 86)])


if __name__ == "__main__":
    unittest.main()
