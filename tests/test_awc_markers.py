"""Tests for tools/awc_markers.py, built from synthetic AWC data only (no game files).

    python -I tests/test_awc_markers.py
"""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "awc_markers.py")
sys.path.insert(0, os.path.dirname(TOOL))
import awc_markers  # noqa: E402

TRACK, TICK, SPAN = 0x11111111, 0x22222222, 0x33333333      # made-up type hashes
NAME_A, NAME_B = 0x84000001, 0xB0000002                         # made-up value hashes (not sensible floats)


def event(typ, value, time, flags=0):
    """One 16-byte event; a float value is stored as f32, an int as the raw u32."""
    bits = value if isinstance(value, int) else struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<IIII", typ, bits, time, flags)


def build_bank(entries):
    """entries: [(stream id, [(tag type, blob), ...])]; the blobs follow the header in order."""
    ntags = sum(len(tags) for _, tags in entries)
    header = 16 + len(entries) * 4 + ntags * 8
    body, placed = bytearray(), []
    for sid, tags in entries:
        recs = []
        for typ, blob in tags:
            recs.append((typ, header + len(body), len(blob)))
            body += blob
        placed.append((sid, recs))
    out = bytearray(b"ADAT" + struct.pack("<III", 0, len(entries), header))
    out += b"".join(struct.pack("<I", sid | (len(recs) << 29)) for sid, recs in placed)
    out += b"".join(struct.pack("<Q", (typ << 56) | (size << 28) | at) for _, recs in placed for typ, at, size in recs)
    return bytes(out + body)


def music_bank(channel_id, samples, toc, marker_chunk):
    """A one-channel streamed-music bank: stream 0 holds 0x48 + 0x55, the channel's own stream holds 0xBD
    (or no tag at all when marker_chunk is None)."""
    desc = bytearray(28)
    struct.pack_into("<III", desc, 0, 1, 4096, 1)             # blocks, chunk size, channels
    struct.pack_into("<II", desc, 12, channel_id, samples)    # interleaved sample count
    struct.pack_into("<H", desc, 22, 48000)
    desc[24] = 0x0B
    data = bytearray(4096)
    struct.pack_into("<I", data, 4, 1)                        # one subblock for the channel
    struct.pack_into("<H", data, 2048, 1)                     # first packet: 1 byte, the TOC
    data[2050] = toc
    markers = [] if marker_chunk is None else [(0xBD, marker_chunk)]
    return build_bank([(0, [(0x48, bytes(desc)), (0x55, bytes(data))]), (channel_id, markers)])


def speech_bank(sid, samples, marker_chunk, payload=b"\0" * 64):
    fmt = bytearray(32)
    struct.pack_into("<I", fmt, 0, samples)
    struct.pack_into("<H", fmt, 8, 48000)
    fmt[28], fmt[29] = 0x0B, 1
    return build_bank([(sid, [(0x55, payload), (0x76, bytes(fmt)), (0xBD, marker_chunk)])])


class MusicCueSheetTest(unittest.TestCase):
    def test_failed_cli_rerun_clears_previous_regions(self):
        from rsc8_audio import MARKER_VALUE_START, MARKER_VALUE_END

        cues = event(1, MARKER_VALUE_START, 10) + event(1, MARKER_VALUE_END, 30)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "music.awc"
            out = Path(tmp) / "out"
            sheet = out / "music" / "cue_regions.json"
            args = [sys.executable, "-I", Path(TOOL).with_name("awc_opus.py"), src, out, "--cue-sheet", "--jobs", "1"]
            for failure in ("parse", "write"):
                with self.subTest(failure=failure):
                    src.write_bytes(music_bank(7, 200, 0xFC, cues))
                    subprocess.run(args, check=True, capture_output=True)
                    self.assertEqual(json.loads(sheet.read_bytes())["regions"][0]["file"], "00000007.ogg")
                    if failure == "parse":
                        src.write_bytes(b"invalid bank")
                    else:
                        src.write_bytes(music_bank(8, 200, 0xFC, cues))
                        (sheet.parent / "00000008.ogg").mkdir()
                    result = subprocess.run(args, capture_output=True)
                    self.assertEqual(result.returncode, 0 if failure == "parse" else 1)
                    self.assertEqual(json.loads(sheet.read_bytes()), {"sample_rate": 48000, "regions": []})

    def test_exported_frame_coordinates_and_ambiguous_inputs(self):
        import awc_opus
        from rsc8_audio import MARKER_VALUE_START as start, MARKER_VALUE_END as end, parse_adat_header_bytes

        cues = (event(1, start, 10) + event(2, start, 20) + event(1, end, 30)
                + event(2, end, 40) + event(3, start, 60) + event(3, end, 500))
        bank = music_bank(7, 200, 0xFC, cues)  # 100 frames of stereo
        header = parse_adat_header_bytes(bank, 0, len(bank))
        entries = [(sid, [(kind, bank[off:off + size]) for kind, off, size in tags])
                   for sid, tags in header["entries"]]
        desc, data = [blob for _, blob in entries[0][1]]
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "music.awc")
            out = os.path.join(tmp, "out")
            sheet = os.path.join(out, "music", "cue_regions.json")

            def export(payload, flag=True):
                with open(src, "wb") as f:
                    f.write(payload)
                args = [src, out, "--jobs", "1"] + (["--cue-sheet"] if flag else [])
                self.assertEqual(awc_opus.main(args), 0)
                if flag:
                    with open(sheet, encoding="utf-8") as f:
                        return json.load(f)

            export(bank, False)
            self.assertFalse(os.path.exists(sheet))
            self.assertEqual(export(bank), {"sample_rate": 48000, "regions": [
                {"file": "00000007.ogg", "name_hash": "00000001", "start_sample": 10, "end_sample": 30},
                {"file": "00000007.ogg", "name_hash": "00000002", "start_sample": 20, "end_sample": 40}]})
            self.assertTrue(os.path.isfile(os.path.join(out, "music", "00000007.ogg")))
            for ambiguous in (entries + [entries[0]], entries + [entries[1]],
                              [(0, [(0x48, desc), (0x55, bytes(len(data)))]), entries[1]],
                              [entries[0], (9, entries[1][1])]):
                self.assertEqual(export(build_bank(ambiguous))["regions"], [])


class EventLayoutTest(unittest.TestCase):
    def test_sixteen_byte_events_and_a_dropped_tail(self):
        chunk = event(TRACK, 1234.0, 0) + event(TICK, 8.0, 48000, 5) + b"\x01\x02\x03"
        self.assertEqual(awc_markers.events(chunk),
                         [(TRACK, 0x449A4000, 0, 0), (TICK, 0x41000000, 48000, 5)])
        self.assertEqual(awc_markers.events(b""), [])


class AgreesWithTheBankDecoderTest(unittest.TestCase):
    def test_same_records_as_rsc8_audio(self):
        import rsc8_audio

        chunk = event(0x11111111, 3.0, 0) + event(0x22222222, 7.0, 480, 2) + event(0x11111111, 4.0, 96000)
        mine = awc_markers.events(chunk)
        theirs = rsc8_audio.decode_marker_records(chunk, sample_count=96000)
        self.assertEqual(theirs["records"], len(mine))
        self.assertEqual(theirs["names"], {0x11111111: 2, 0x22222222: 1})
        self.assertEqual(theirs["first_name"], mine[0][0])
        self.assertEqual(theirs["offset_max"], max(e[2] for e in mine))
        self.assertEqual(theirs["extra_max"], max(e[3] for e in mine))
        self.assertTrue(theirs["offsets_ascending"] and theirs["offsets_within_samples"])


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def archive(self):
        # Stereo packets: 192,000 interleaved samples are 96,000 frames, so the marker at 95,000 is inside.
        music = music_bank(0x1234, 192000, 0xFC, event(TRACK, 1001.0, 0) + event(SPAN, NAME_A, 4800)
                           + event(TICK, 8.0, 9600) + event(TICK, 4.0, 95000) + event(SPAN, NAME_B, 95000))
        # Mono packets: the same count is 192,000 frames. No marker chunk here.
        plain_music = music_bank(0x2345, 192000, 0xF8, None)
        # A speech bank whose 0x55 payload holds a complete small bank: it must not be counted as a third bank.
        inner = speech_bank(0x99, 10, event(TRACK, 1.0, 0))
        speech = speech_bank(0x77, 48000, event(TICK, 3.0, 50000, 3) + event(TRACK, 2.0, 100) + b"\xAA" * 5,
                             payload=inner)
        stray = b"ADAT" + b"\xff" * 12                         # the magic alone is not a bank
        return b"\x78\x9c" + b"\0" * 30 + stray + music + b"\0" * 100 + plain_music + b"\0" * 7 + speech + b"\0" * 9

    def test_unlistable_child_preserves_the_previous_report(self):
        self.write("visible.awc", self.archive())
        report = self.write("markers.json", b"previous report\n")
        blocked = os.path.join(self.tmp.name, "blocked")
        os.mkdir(blocked)
        scandir = os.scandir

        def scan_directory(path):
            if path == blocked:
                raise PermissionError(13, "permission denied", blocked)
            return scandir(path)

        with mock.patch.object(os, "scandir", scan_directory), contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(awc_markers.main([self.tmp.name, "--out", report]), 1)
        self.assertIn(blocked, err.getvalue())
        self.assertEqual(out.getvalue(), "")
        with open(report, "rb") as f:
            self.assertEqual(f.read(), b"previous report\n")

    def test_archive_census(self):
        report = awc_markers.scan(self.write("fake.rpf", self.archive()))
        self.assertEqual((report["banks"], report["streams"]), (3, 5))
        m, o = report["streamed_music"], report["other"]
        self.assertEqual((m["banks"], m["banks_with_chunk"], m["chunks"], m["events"]), (2, 1, 1, 5))
        self.assertEqual((m["chunks_not_multiple_of_16"], m["chunks_time_decreases"]), (0, 0))
        self.assertEqual((m["chunks_with_length"], m["chunks_past_end"]), (1, 0))
        self.assertEqual(m["flags"], {"0x00000000": 5})
        self.assertEqual(m["events_per_chunk"], {"min": 5, "median": 5, "max": 5})
        self.assertEqual(list(m["types"]), ["0x22222222", "0x33333333", "0x11111111"])   # most events first
        track = m["types"]["0x11111111"]
        self.assertEqual((track["events"], track["chunks"], track["chunks_with_one"], track["at_time_0"]),
                         (1, 1, 1, 1))
        self.assertEqual((track["value_min"], track["value_max"], track["values_all_integers"]),
                         (1001.0, 1001.0, True))
        tick = m["types"]["0x22222222"]
        self.assertEqual((tick["events"], tick["max_per_chunk"], tick["chunks_with_one"]), (2, 2, 0))
        self.assertEqual((tick["value_min"], tick["value_max"]), (4.0, 8.0))
        span = m["types"]["0x33333333"]
        self.assertEqual(span["value_bits"], ["0x84000001", "0xB0000002"])
        self.assertFalse(span["values_all_integers"])
        # The speech chunk: 37 bytes, times 50,000 then 100, and 50,000 is past its 48,000 frames.
        self.assertEqual((o["banks"], o["banks_with_chunk"], o["chunks"], o["events"]), (1, 1, 1, 2))
        self.assertEqual((o["chunks_not_multiple_of_16"], o["chunks_time_decreases"]), (1, 1))
        self.assertEqual((o["chunks_with_length"], o["chunks_past_end"]), (1, 1))
        self.assertEqual(o["flags"], {"0x00000000": 1, "0x00000003": 1})

    def test_mono_packets_keep_the_interleaved_count(self):
        # The marker at 150,000 is past 96,000 (the stereo reading) but inside 192,000 mono frames.
        bank = music_bank(0x1234, 192000, 0xF8, event(TRACK, 1.0, 150000))
        m = awc_markers.scan(self.write("mono.awc", bank))["streamed_music"]
        self.assertEqual((m["chunks_with_length"], m["chunks_past_end"]), (1, 0))
        bank = music_bank(0x1234, 192000, 0xFC, event(TRACK, 1.0, 150000))
        m = awc_markers.scan(self.write("stereo.awc", bank))["streamed_music"]
        self.assertEqual((m["chunks_with_length"], m["chunks_past_end"]), (1, 1))

    def test_a_bank_magic_across_the_search_window(self):
        bank = speech_bank(0x55, 48000, event(TICK, 1.0, 0))
        report = awc_markers.scan(self.write("edge.rpf", b"\0" * (awc_markers.WINDOW - 2) + bank))
        self.assertEqual((report["banks"], report["other"]["events"]), (1, 1))

    def test_many_small_banks_are_read_about_once(self):
        class Counting(io.BytesIO):
            total = 0

            def read(self, n=-1):
                buf = super().read(n)
                self.total += len(buf)
                return buf

        bank = speech_bank(3, 64, event(TICK, 1.0, 2))
        totals = {}
        for count in (128, 512):
            fh = Counting(bank * count)
            self.assertEqual(sum(1 for _ in awc_markers.iter_banks(fh, len(bank) * count)), count)
            totals[count] = fh.total
        # Four times the banks: about four times the bytes. Reading a fresh window per bank gave 16 times.
        self.assertLess(totals[512], 5 * totals[128])
        self.assertLess(totals[512], 8 * len(bank) * 512)

    def test_directory_and_report_file(self):
        folder = os.path.join(self.tmp.name, "banks", "nested")
        os.makedirs(folder)
        with open(os.path.join(folder, "a.awc"), "wb") as f:
            f.write(music_bank(0x1234, 96000, 0xF8, event(TRACK, 7.0, 0)))
        with open(os.path.join(folder, "notes.txt"), "wb") as f:
            f.write(speech_bank(0x55, 48000, event(TICK, 1.0, 0)))          # not *.awc: ignored
        out = os.path.join(self.tmp.name, "report.json")
        with contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(awc_markers.main([os.path.join(self.tmp.name, "banks"), "--out", out]), 0)
        self.assertIn("1 banks, 2 streams", text.getvalue())
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        report = json.loads(raw)
        self.assertEqual((report["banks"], report["streamed_music"]["events"], report["other"]["chunks"]), (1, 1, 0))
        self.assertIsNone(report["other"]["events_per_chunk"])

    def test_runs_under_python_dash_i(self):
        path = self.write("one.awc", speech_bank(0x55, 48000, event(TICK, 1.0, 0)))
        proc = subprocess.run([sys.executable, "-I", TOOL, path], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("other: 1 banks, 1 with a 0xBD chunk; 1 chunks, 1 events, 1 types", proc.stdout)
        missing = subprocess.run([sys.executable, "-I", TOOL, path + ".missing"], capture_output=True, text=True,
                                 timeout=120)
        self.assertEqual(missing.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
