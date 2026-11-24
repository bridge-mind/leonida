"""Regression tests for tools/awc_opus.py, built entirely from synthetic AWC data (no game files).

    python -I tests/test_awc_opus.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
import wave
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_awc_opus():
    path = os.path.join(ROOT, "tools", "awc_opus.py")
    spec = importlib.util.spec_from_file_location("awc_opus_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


awc = load_awc_opus()


def read_wav(path):
    """Read the RIFF chunks so the audio can follow either PCM or float metadata."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AssertionError("not a RIFF/WAVE file")
    riff_size, = struct.unpack_from("<I", data, 4)
    chunks = {}
    offset = 12
    while offset < len(data):
        name, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        if offset + size > len(data):
            raise AssertionError(f"truncated WAV chunk {name!r}")
        chunks[name] = data[offset:offset + size]
        offset += size + (size & 1)
    fmt, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    return {
        "data": data,
        "riff_size": riff_size,
        "fmt_size": len(chunks[b"fmt "]),
        "chunks": chunks,
        "format": fmt,
        "channels": channels,
        "rate": rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits": bits,
        "data_size": len(chunks[b"data"]),
        "pcm": chunks[b"data"],
    }


def parse_ogg_pages(data):
    """All pages as dicts; also verifies that every stored CRC matches the page contents."""
    pages = []
    p = 0
    while p < len(data):
        if data[p:p + 4] != b"OggS":
            raise AssertionError("bad Ogg capture pattern at %d" % p)
        flags = data[p + 5]
        granule, = struct.unpack_from("<q", data, p + 6)
        serial, seq, crc = struct.unpack_from("<III", data, p + 14)
        nseg = data[p + 26]
        lacing = data[p + 27:p + 27 + nseg]
        body_size = sum(lacing)
        body_start = p + 27 + nseg
        body = data[body_start:body_start + body_size]
        page = bytearray(data[p:body_start + body_size])
        struct.pack_into("<I", page, 22, 0)
        if awc.ogg_crc(bytes(page)) != crc:
            raise AssertionError("bad Ogg CRC on page %d" % seq)
        pages.append({"flags": flags, "granule": granule, "serial": serial, "seq": seq,
                      "lacing": lacing, "body": body})
        p = body_start + body_size
    return pages


def tag(typ, offset, size):
    return (typ << 56) | (size << 28) | offset


def format_block(samples, rate, codec, channels):
    block = bytearray(32)
    struct.pack_into("<I", block, 0, samples)
    struct.pack_into("<H", block, 8, rate)
    block[28] = codec
    block[29] = channels
    return bytes(block)


def build_bank(streams):
    """A minimal little-endian AWC: ADAT header, stream table, two tags per stream, then fmt/data.

    streams: [(id, format bytes, data bytes)]. The format bytes are 32 bytes:
    u32 samples, u16 rate at +8, codec at +28, channels at +29.
    """
    n = len(streams)
    header_size = 16 + n * 4 + n * 2 * 8
    body = bytearray()
    entries = []
    for sid, fmt, data in streams:
        fmt_at = header_size + len(body)
        body += fmt
        data_at = header_size + len(body)
        body += data
        entries.append((sid, fmt_at, len(fmt), data_at, len(data)))
    out = bytearray(b"ADAT" + struct.pack("<III", 0, n, header_size))
    out += b"".join(struct.pack("<I", (sid & 0x1FFFFFFF) | (2 << 29))
                    for sid, _, _, _, _ in entries)
    for _, fmt_at, fmt_size, data_at, data_size in entries:
        out += struct.pack("<Q", tag(0x76, fmt_at, fmt_size))
        out += struct.pack("<Q", tag(0x55, data_at, data_size))
    if len(out) != header_size:
        raise AssertionError("stream table/tag layout does not match header_size")
    out += body
    return bytes(out)


def build_music_bank(block_tocs, samples, rate=48000, last_size=None):
    """One TOC per channel group in each block, with one 2048-byte subblock per group."""
    channels = len(block_tocs[0])
    chunk = 2048 * (1 + channels)
    desc = bytearray(12 + 16 * channels)
    struct.pack_into("<III", desc, 0, len(block_tocs), chunk, channels)
    for i in range(channels):
        base = 12 + 16 * i
        struct.pack_into("<II", desc, base, 0x100 + i, samples)
        struct.pack_into("<H", desc, base + 10, rate)
        desc[base + 12] = 0x0B
    data = bytearray(len(block_tocs) * chunk)
    for block, tocs in enumerate(block_tocs):
        at = block * chunk
        for i, toc in enumerate(tocs):
            struct.pack_into("<I", data, at + 24 * i + 4, 1)
            packet = at + 2048 * (1 + i)
            struct.pack_into("<HB", data, packet, 1, toc)
    if last_size is not None:
        del data[(len(block_tocs) - 1) * chunk + last_size:]
    header = 16 + 4 + 2 * 8
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header))
    out += struct.pack("<I", 0x20 | (2 << 29))
    out += struct.pack("<Q", tag(0x48, header, len(desc)))
    out += struct.pack("<Q", tag(0x55, header + len(desc), len(data)))
    return bytes(out + desc + data)


def build_repeating_music_bank(chunks, group_samples=None):
    """Each chunk lists (skip samples, packets) for each channel group."""
    channels = len(chunks[0])
    chunk_size = 2048 * (channels + 1)
    bank = bytearray(build_music_bank([[0xF8] * channels for _ in chunks], 2880))
    _, music = awc.parse_awc(bank)
    (descriptor_offset, _), (data_offset, _) = music[0]
    if group_samples is not None:
        for group, samples in enumerate(group_samples):
            struct.pack_into("<I", bank, descriptor_offset + 12 + 16 * group + 4, samples)
    for block, groups in enumerate(chunks):
        at = data_offset + block * chunk_size
        for group, (skip, packets) in enumerate(groups):
            struct.pack_into("<I", bank, at + group * 24 + 8, skip)
            start = at + (group + 1) * 2048
            payload = b"".join(struct.pack("<H", len(packet)) + packet for packet in packets)
            if len(payload) > 2048:
                raise AssertionError("synthetic packets do not fit their subblock")
            bank[start:start + 2048] = payload.ljust(2048, b"\x00")
    return bytes(bank)


def build_music_pcm_bank(blocks=1, count=1, pcm=None):
    """One streamed-music descriptor (one mono Opus group) plus one PCM16 stream.

    The music data tag holds exactly one real chunk: a 2048-byte header subblock and one
    2048-byte data subblock with a single CELT 20 ms mono packet. The PCM payload follows the
    music data directly. `blocks` and `count` are what the descriptor and the chunk record
    declare, so a test can declare more than the data tag holds.
    """
    chunk = 4096
    desc = bytearray(12 + 16)
    struct.pack_into("<III", desc, 0, blocks, chunk, 1)
    struct.pack_into("<II", desc, 12, 0x100, 960)
    struct.pack_into("<H", desc, 22, 48000)
    desc[24] = 0x0B
    music_data = bytearray(chunk)
    struct.pack_into("<I", music_data, 4, count)
    struct.pack_into("<HB", music_data, 2048, 1, 0x78)
    if pcm is None:
        pcm = struct.pack("<4h", 11, 22, 33, 44)
    fmt = format_block(len(pcm) // 2, 48000, 0, 1)
    header_size = 16 + 2 * 4 + 4 * 8
    desc_at = header_size
    mdata_at = desc_at + len(desc)
    pcm_at = mdata_at + len(music_data)
    fmt_at = pcm_at + len(pcm)
    out = bytearray(b"ADAT" + struct.pack("<III", 0, 2, header_size))
    out += struct.pack("<I", 0x20 | (2 << 29))         # music stream: tags 0x48 + 0x55
    out += struct.pack("<I", 0x21 | (2 << 29))         # PCM stream: tags 0x76 + 0x55
    out += struct.pack("<Q", tag(0x48, desc_at, len(desc)))
    out += struct.pack("<Q", tag(0x55, mdata_at, len(music_data)))
    out += struct.pack("<Q", tag(0x76, fmt_at, len(fmt)))
    out += struct.pack("<Q", tag(0x55, pcm_at, len(pcm)))
    if len(out) != header_size:
        raise AssertionError("stream table/tag layout does not match header_size")
    return bytes(out + desc + music_data + pcm + fmt)


class WriteWavTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_pcm16_fields(self):
        pcm = bytes(i & 0xFF for i in range(400))
        path = self.path("pcm16.wav")
        awc.write_wav(path, pcm, 22050, 2)
        w = read_wav(path)
        self.assertEqual(w["riff_size"], 36 + len(pcm))
        self.assertEqual(w["fmt_size"], 16)
        self.assertEqual(w["format"], 1)
        self.assertEqual(w["channels"], 2)
        self.assertEqual(w["rate"], 22050)
        self.assertEqual(w["byte_rate"], 22050 * 2 * 2)
        self.assertEqual(w["block_align"], 2 * 2)
        self.assertEqual(w["bits"], 16)
        self.assertEqual(w["data_size"], len(pcm))
        self.assertEqual(w["pcm"], pcm)
        self.assertEqual(len(w["data"]), 44 + len(pcm))

    def test_float32_fields(self):
        pcm = struct.pack("<8f", -2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.0)
        path = self.path("float32.wav")
        awc.write_wav(path, pcm, 44100, 2, float32=True)
        w = read_wav(path)
        self.assertEqual(list(w["chunks"]), [b"fmt ", b"fact", b"data"])
        self.assertEqual(w["chunks"][b"fmt "][16:], b"\x00\x00")
        self.assertEqual(struct.unpack("<I", w["chunks"][b"fact"]), (4,))
        self.assertEqual(w["riff_size"], len(w["data"]) - 8)
        self.assertEqual(w["format"], 3)
        self.assertEqual(w["channels"], 2)
        self.assertEqual(w["rate"], 44100)
        self.assertEqual(w["byte_rate"], 44100 * 2 * 4)
        self.assertEqual(w["block_align"], 2 * 4)
        self.assertEqual(w["bits"], 32)
        self.assertEqual(w["data_size"], len(pcm))
        self.assertEqual(w["pcm"], pcm)
        self.assertEqual(len(w["data"]), 58 + len(pcm))

    def test_sliced_memoryview_matches_bytes_for_pcm_and_float(self):
        pcm = bytes(range(64))
        view = memoryview(b"prefix" + pcm + b"suffix")[6:-6]
        for float32 in (False, True):
            for channels in (1, 2):
                with self.subTest(float32=float32, channels=channels):
                    first, second = self.path("bytes.wav"), self.path("view.wav")
                    awc.write_wav(first, pcm, 48000, channels, float32)
                    awc.write_wav(second, view, 48000, channels, float32)
                    self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_typed_pcm_views_use_byte_lengths_in_every_wav_chunk(self):
        pcm = bytes(range(64))
        views = [memoryview(pcm).cast(fmt) for fmt in ("h", "I", "f")]
        views.append(memoryview(pcm).cast("B", shape=(8, 8)))
        for view in views:
            for float32 in (False, True):
                for channels in (1, 2):
                    with self.subTest(format=view.format, ndim=view.ndim, float32=float32, channels=channels):
                        first, second = self.path("bytes.wav"), self.path("typed.wav")
                        awc.write_wav(first, pcm, 48000, channels, float32)
                        awc.write_wav(second, view, 48000, channels, float32)
                        self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())
                        wav = read_wav(second)
                        self.assertEqual(wav["riff_size"], len(wav["data"]) - 8)
                        self.assertEqual(wav["data_size"], view.nbytes)
                        if float32:
                            self.assertEqual(struct.unpack("<I", wav["chunks"][b"fact"])[0],
                                             view.nbytes // (channels * 4))
                        else:
                            with wave.open(second, "rb") as reader:
                                self.assertEqual(reader.getnframes(), view.nbytes // (channels * 2))
                                self.assertEqual(reader.readframes(reader.getnframes()), pcm)

    def test_noncontiguous_pcm_view_is_refused_before_staging(self):
        destination = self.path("existing.wav")
        Path(destination).write_bytes(b"previous complete file")
        with self.assertRaises(TypeError):
            awc.write_wav(destination, memoryview(bytes(64))[::2], 48000, 1)
        self.assertEqual(Path(destination).read_bytes(), b"previous complete file")
        self.assertEqual(os.listdir(self.tmp.name), ["existing.wav"])

    def test_payload_write_failure_preserves_destination_and_cleans_staging(self):
        destination = self.path("existing.wav")
        Path(destination).write_bytes(b"previous complete file")
        make_temp = awc.tempfile.NamedTemporaryFile

        class FailingPayload:
            def __init__(self, *args, **kwargs):
                self.file = make_temp(*args, **kwargs)
                self.writes = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self.file.__exit__(*exc)

            @property
            def name(self):
                return self.file.name

            def write(self, data):
                self.writes += 1
                if self.writes == 2:
                    self.file.write(data[:3])
                    raise OSError("payload write failed")
                return self.file.write(data)

        with mock.patch.object(awc.tempfile, "NamedTemporaryFile", FailingPayload):
            with self.assertRaisesRegex(OSError, "payload write failed"):
                awc.write_wav(destination, bytes(64), 48000, 1)
        self.assertEqual(Path(destination).read_bytes(), b"previous complete file")
        self.assertEqual(os.listdir(self.tmp.name), ["existing.wav"])


class OggCrcTest(unittest.TestCase):
    # A fixed 32-byte page: 27-byte header (checksum field zero), one segment of four bytes, body b"test".
    PAGE = bytes.fromhex("4f67675300020000000000000000785634120000000000000000010474657374")

    def test_known_page_crc(self):
        self.assertEqual(len(self.PAGE), 32)
        self.assertEqual(self.PAGE[:4], b"OggS")
        self.assertEqual(awc.ogg_crc(self.PAGE), 0xF7708715)

    def test_payload_changes_crc(self):
        self.assertNotEqual(awc.ogg_crc(self.PAGE[:-1] + b"u"), 0xF7708715)

    def test_page_segment_limit(self):
        # 254 full segments plus a terminating partial one; a multiple of 255 needs an extra zero segment.
        page = awc.ogg_page(1, 0, 0, [b"x" * (254 * 255 + 254)], 2)
        self.assertEqual(len(page), 27 + 255 + 254 * 255 + 254)
        with self.assertRaisesRegex(ValueError, "255 Ogg segments"):
            awc.ogg_page(1, 0, 0, [b"x" * (255 * 255)], 2)


class OpusSamplesTest(unittest.TestCase):
    def test_toc_patterns(self):
        cases = [
            ("SILK 20 ms, code 0", bytes([0x08]), 960),         # config 1: 960-sample frame, one frame
            ("SILK 40 ms, code 0", bytes([0x10]), 1920),        # config 2: 1920-sample frame, one frame
            ("SILK 40 ms, code 1", bytes([0x11]), 3840),        # two 1920-sample frames
            ("CELT 10 ms, code 0", bytes([0x90]), 480),         # config 18: 480-sample frame, one frame
            ("CELT 20 ms, code 0", bytes([0x98]), 960),         # config 19: 960-sample frame, one frame
            ("CELT 10 ms, code 2", bytes([0x92]), 960),         # two 480-sample frames
            ("CELT 20 ms, code 3", bytes([0x9B, 0x03]), 2880),  # packet[1] & 0x3F == 3 frames of 960
        ]
        for name, packet, expected in cases:
            with self.subTest(name):
                self.assertEqual(awc.opus_samples(packet), expected)


class ExportBankTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def make_bank(self):
        pcm16 = bytes(i & 0xFF for i in range(100 * 2 * 2))
        floats = struct.pack("<50f", *[((i % 9) - 4) / 4.0 for i in range(50)])
        opus = bytearray(2 * 2048)
        packet_a = bytes([0x02])        # SILK 10 ms, code 2: 2 * 480 = 960 samples
        packet_b = bytes([0x0B, 0x02])  # SILK 20 ms, code 3: 2 * 960 = 1920 samples
        struct.pack_into("<H", opus, 0, len(packet_a))
        opus[2:2 + len(packet_a)] = packet_a
        struct.pack_into("<H", opus, 2048, len(packet_b))
        opus[2050:2050 + len(packet_b)] = packet_b
        bank = build_bank([
            (1, format_block(100, 22050, 0x00, 2), pcm16),
            (2, format_block(50, 48000, 0x02, 1), floats),
            (3, format_block(1920, 48000, 0x0B, 1), bytes(opus)),
        ])
        path = os.path.join(self.tmp.name, "synthetic.awc")
        with open(path, "wb") as f:
            f.write(bank)
        return path, pcm16, floats

    def test_recursive_cli_preserves_bank_paths(self):
        src = Path(self.tmp.name) / "banks"
        samples = {"north/shared": 11, "south/shared": 22, "flat": 33}
        fmt = format_block(1, 48000, 0, 1)
        for stem, sample in samples.items():
            bank = src / (stem + ".awc")
            bank.parent.mkdir(parents=True, exist_ok=True)
            bank.write_bytes(build_bank([(1, fmt, struct.pack("<h", sample))]))

        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                out = Path(self.out) / str(jobs)
                previous = out / "shared" / "00000001.wav"
                previous.parent.mkdir(parents=True)
                previous.write_bytes(b"previous export")
                subprocess.run([sys.executable, "-I", awc.__file__, src, out, "--jobs", str(jobs)],
                               check=True, capture_output=True)
                for stem, sample in samples.items():
                    self.assertEqual(read_wav(out / stem / "00000001.wav")["pcm"], struct.pack("<h", sample))
                self.assertEqual(previous.read_bytes(), b"previous export")

    def test_cli_fails_when_bank_enumeration_is_incomplete(self):
        src = Path(self.tmp.name) / "banks"
        src.mkdir()
        blocked = src / "blocked"

        def walk(root, onerror=None):
            yield root, ["blocked"], ["valid.awc"]
            if onerror:
                onerror(PermissionError(13, "Permission denied", blocked))

        totals = {"opus": 0, "pcm": 0, "other": 0, "music": 0}
        with mock.patch.object(awc.os, "walk", walk), mock.patch.object(awc, "export_bank", return_value=totals) as export, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(awc.main([str(src), self.out, "--jobs", "1"]), 1)
        self.assertIn("blocked", err.getvalue())
        self.assertIn("Permission denied", err.getvalue())
        export.assert_not_called()

    def test_colliding_bank_output_directories_are_refused_before_export(self):
        src = Path(self.tmp.name) / "banks"
        src.mkdir()
        listing = [(str(src), [], ["fixture.awc", "fixture.AWC"])]
        with mock.patch.object(awc.os, "walk", return_value=listing), \
                mock.patch.object(awc, "export_bank") as export, \
                contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as exc:
            awc.main([str(src), self.out, "--jobs", "1"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("fixture.awc", err.getvalue())
        self.assertIn("fixture.AWC", err.getvalue())
        self.assertIn("would both export to", err.getvalue())
        export.assert_not_called()

    def test_cli_bank_limits(self):
        src = Path(self.tmp.name) / "banks"
        src.mkdir()
        fmt = format_block(1, 48000, 0, 1)
        for stem in ("second", "first"):
            (src / (stem + ".awc")).write_bytes(build_bank([(1, fmt, struct.pack("<h", 42))]))

        for limit, expected in ((0, []), (1, ["first"]), (9, ["first", "second"])):
            with self.subTest(limit=limit):
                out = Path(self.out) / str(limit)
                result = subprocess.run([sys.executable, "-I", awc.__file__, src, out, "--limit", str(limit),
                                         "--jobs", "1"], check=True, capture_output=True, text=True)
                self.assertIn(f"{len(expected)} banks:", result.stdout)
                self.assertEqual(sorted(wav.parent.name for wav in out.glob("*/00000001.wav")), expected)

        result = subprocess.run([sys.executable, "-I", awc.__file__, src, self.out, "--limit", "-1"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--limit must be non-negative", result.stderr)
        self.assertFalse((Path(self.out) / "first").exists())

    def test_cli_rejects_bad_src_out_and_jobs(self):
        bank, _, _ = self.make_bank()
        out_file = os.path.join(self.tmp.name, "out_file")
        with open(out_file, "wb") as f:
            f.write(b"x")
        cases = (
            ([os.path.join(self.tmp.name, "missing.awc"), self.out], "does not exist"),
            ([bank, out_file], "is a file, not a folder"),
            ([bank, self.out, "--jobs", "0"], "--jobs must be at least 1"),
            ([bank, self.out, "--jobs", "-2"], "--jobs must be at least 1"),
            ([bank, self.out, "--pre-skip", "-1"], "--pre-skip must be between 0 and 65535"),
            ([bank, self.out, "--pre-skip", "65536"], "--pre-skip must be between 0 and 65535"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    awc.main(args)
                self.assertEqual(cm.exception.code, 2)
                self.assertIn(expected, err.getvalue())

    def test_single_bank_cli_keeps_flat_output(self):
        bank, pcm, _ = self.make_bank()
        subprocess.run([sys.executable, "-I", awc.__file__, bank, self.out, "--jobs", "1"],
                       check=True, capture_output=True)
        self.assertEqual(read_wav(Path(self.out) / "synthetic" / "00000001.wav")["pcm"], pcm)

    def test_pcm16_and_float_and_opus(self):
        bank, pcm16, floats = self.make_bank()
        counts = awc.export_bank(bank, self.out)
        self.assertEqual(counts, {"opus": 1, "pcm": 2, "other": 0, "music": 0})
        od = os.path.join(self.out, "synthetic")

        with self.subTest("PCM16 stream -> WAV"):
            w = read_wav(os.path.join(od, "00000001.wav"))
            self.assertEqual(w["format"], 1)
            self.assertEqual(w["bits"], 16)
            self.assertEqual(w["rate"], 22050)
            self.assertEqual(w["channels"], 2)
            self.assertEqual(w["data_size"], 100 * 2 * 2)      # declared samples * channels * 2 bytes
            self.assertEqual(w["pcm"], pcm16)

        with self.subTest("float32 stream -> float WAV"):
            w = read_wav(os.path.join(od, "00000002.wav"))
            self.assertEqual(struct.unpack("<I", w["chunks"][b"fact"]), (50,))
            self.assertEqual(w["format"], 3)
            self.assertEqual(w["bits"], 32)
            self.assertEqual(w["rate"], 48000)
            self.assertEqual(w["channels"], 1)
            self.assertEqual(w["data_size"], 50 * 1 * 4)       # declared samples * channels * 4 bytes
            self.assertEqual(w["pcm"], floats)

        with self.subTest("Opus stream -> Ogg Opus"):
            with open(os.path.join(od, "00000003.ogg"), "rb") as f:
                data = f.read()
            pages = parse_ogg_pages(data)
            self.assertGreaterEqual(len(pages), 3)
            head = pages[0]
            self.assertEqual(head["seq"], 0)
            self.assertEqual(head["flags"] & 0x02, 0x02)       # beginning of stream
            self.assertEqual(head["body"][:8], b"OpusHead")
            version, channels, preskip, in_rate, gain, mapping = struct.unpack_from("<BBHIhB", head["body"], 8)
            self.assertEqual((version, channels, preskip, in_rate, gain, mapping), (1, 1, 312, 48000, 0, 0))
            tags = pages[1]
            self.assertEqual(tags["seq"], 1)
            self.assertEqual(tags["body"][:8], b"OpusTags")
            last = pages[-1]
            self.assertEqual(last["flags"] & 0x04, 0x04)       # end of stream
            self.assertEqual(last["granule"], 312 + 1920)      # pre-skip + declared real samples
            self.assertGreater(last["granule"], 0)
            self.assertEqual(last["serial"], 3)

    def test_new_outputs_in_one_folder_probe_the_file_mode_once(self):
        fmt = format_block(1, 48000, 0, 1)
        path = os.path.join(self.tmp.name, "many.awc")
        with open(path, "wb") as f:
            f.write(build_bank([(sid, fmt, struct.pack("<h", sid)) for sid in range(1, 6)]))
        real_open = os.open
        probes = []

        def counting_open(file, *args, **kwargs):
            if str(file).endswith(".probe"):
                probes.append(file)
            return real_open(file, *args, **kwargs)
        with mock.patch.object(os, "open", counting_open):
            counts = awc.export_bank(path, self.out)
        self.assertEqual(counts["pcm"], 5)
        self.assertEqual(len(os.listdir(os.path.join(self.out, "many"))), 5)
        self.assertEqual(len(probes), 1)

    def test_pcm_data_size_must_match_the_declared_samples(self):
        fmt = format_block(100, 22050, 0x00, 2)
        path = os.path.join(self.tmp.name, "odd.awc")
        with open(path, "wb") as f:
            f.write(build_bank([(1, fmt, bytes(401)), (2, fmt, bytes(399)), (3, fmt, bytes(400))]))
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 2, "music": 0})
        self.assertEqual(os.listdir(os.path.join(self.out, "odd")), ["00000003.wav"])

    def test_pcm_export_avoids_payload_sized_temporary_copies(self):
        if tracemalloc.is_tracing():
            self.skipTest("this allocation check needs its own tracing interval")
        payload_size = 2 * 1024 * 1024
        for codec in (0, 2):
            for channels in (1, 2):
                with self.subTest(codec=codec, channels=channels):
                    samples = payload_size // (channels * (4 if codec == 2 else 2))
                    bank = build_bank([(1, format_block(samples, 48000, codec, channels), bytes(payload_size))])
                    path = os.path.join(self.tmp.name, f"pcm_{codec}_{channels}.awc")
                    Path(path).write_bytes(bank)
                    del bank
                    tracemalloc.start()
                    try:
                        counts = awc.export_bank(path, self.out)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    self.assertEqual(counts["pcm"], 1)
                    # Reading the bank needs one payload-sized allocation. A second full copy, as a slice
                    # or a combined RIFF buffer, would exceed this generous allowance for headers and I/O.
                    self.assertLess(peak, payload_size * 2, f"peak {peak:,} for {payload_size:,} PCM bytes")

    def test_stereo_plain_stream_length_is_per_channel(self):
        # A stereo format tag stores interleaved samples: 3840 counts = 1920 frames per channel.
        # Three 960-frame packets carry 2880 frames per channel, so the declared 1920 trims
        # 960 frames of payload; the point is the halving, not a realistic stream length.
        packet = bytes([0x9C])          # CELT 20 ms, code 0, stereo bit: 960 frames per channel
        opus = bytearray(3 * 2048)
        for block in range(3):
            struct.pack_into("<H", opus, block * 2048, len(packet))
            opus[block * 2048 + 2:block * 2048 + 2 + len(packet)] = packet
        bank = build_bank([(7, format_block(3840, 48000, 0x0B, 2), bytes(opus))])
        path = os.path.join(self.tmp.name, "stereo.awc")
        with open(path, "wb") as f:
            f.write(bank)

        awc.export_bank(path, self.out)
        with open(os.path.join(self.out, "stereo", "00000007.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(pages[0]["body"][9], 2)
        self.assertEqual(struct.unpack_from("<H", pages[0]["body"], 10)[0], 312)
        self.assertEqual(pages[-1]["granule"], 312 + 1920)

    def test_declared_samples_are_48khz_frames_not_scaled_by_rate(self):
        # Four 960-frame packets are 3840 frames; the tag declares 2880 real frames and a 32 kHz
        # source rate. Scaling by 48000/32000 would make the target 4320, above the payload, so
        # the old code kept all 3840 packet frames (granule 3840, 3528 after the pre-skip).
        packet = bytes([0x08])          # SILK 20 ms, code 0: 960 frames
        opus = bytearray(4 * 2048)
        for block in range(4):
            struct.pack_into("<H", opus, block * 2048, len(packet))
            opus[block * 2048 + 2:block * 2048 + 2 + len(packet)] = packet
        bank = build_bank([(9, format_block(2880, 32000, 0x0B, 1), bytes(opus))])
        path = os.path.join(self.tmp.name, "rate32k.awc")
        with open(path, "wb") as f:
            f.write(bank)

        awc.export_bank(path, self.out)
        with open(os.path.join(self.out, "rate32k", "00000009.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(pages[-1]["granule"], 312 + 2880)

    def test_inconsistent_format_frame_count_falls_back(self):
        # +0x0C is authoritative only when it agrees with the interleaved +0 count:
        # 1920 * 2 != 9999, so the block is malformed and +0 / channels (4999) is used
        # instead of silently truncating the real audio.
        packet = bytes([0x9C])          # CELT 20 ms, code 0, stereo bit: 960 frames per channel
        opus = bytearray(6 * 2048)
        for block in range(6):
            struct.pack_into("<H", opus, block * 2048, len(packet))
            opus[block * 2048 + 2:block * 2048 + 2 + len(packet)] = packet
        fmt = bytearray(format_block(9999, 48000, 0x0B, 2))
        struct.pack_into("<I", fmt, 12, 1920)
        bank = build_bank([(11, bytes(fmt), bytes(opus))])
        path = os.path.join(self.tmp.name, "frames.awc")
        with open(path, "wb") as f:
            f.write(bank)

        awc.export_bank(path, self.out)
        with open(os.path.join(self.out, "frames", "0000000B.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(pages[-1]["granule"], 312 + 4999)

    def test_consistent_format_frame_count_is_used(self):
        # 3840 interleaved stereo samples = 1920 per channel, and +0x0C agrees.
        packet = bytes([0x9C])          # CELT 20 ms, code 0, stereo bit: 960 frames per channel
        opus = bytearray(3 * 2048)
        for block in range(3):
            struct.pack_into("<H", opus, block * 2048, len(packet))
            opus[block * 2048 + 2:block * 2048 + 2 + len(packet)] = packet
        fmt = bytearray(format_block(3840, 48000, 0x0B, 2))
        struct.pack_into("<I", fmt, 12, 1920)
        bank = build_bank([(12, bytes(fmt), bytes(opus))])
        path = os.path.join(self.tmp.name, "frames_ok.awc")
        with open(path, "wb") as f:
            f.write(bank)

        awc.export_bank(path, self.out)
        with open(os.path.join(self.out, "frames_ok", "0000000C.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(pages[-1]["granule"], 312 + 1920)

    def test_eos_page_is_emitted_at_the_trim_point(self):
        # 27 packets of 160 bytes / 960 frames: the 26th packet crosses the target
        # (312 + 24000 = 24312) and would otherwise flush a full page first. The EOS
        # page must carry the reaching packet and drop the 27th padding packet, not
        # emit a later page whose granule is smaller than the previous one.
        packet = bytes([0x08]) + bytes(159)     # SILK 20 ms, 960 frames
        packets = [packet] * 27
        path = os.path.join(self.tmp.name, "trim.ogg")
        awc.write_ogg_opus(path, packets, 1, 21, 24000)
        with open(path, "rb") as f:
            pages = parse_ogg_pages(f.read())
        audio = pages[2:]
        granules = [p["granule"] for p in audio]
        self.assertEqual(granules, sorted(granules))
        self.assertEqual(audio[-1]["granule"], 312 + 24000)
        self.assertEqual(audio[-1]["flags"] & 0x04, 0x04)
        self.assertEqual(sum(1 for p in audio for value in p["lacing"] if value < 255), 26)

    def test_pre_skip_larger_than_the_stream_is_clamped(self):
        # Six 960-frame packets are 5760 samples; a pre-skip of 65535 would put the
        # EOS granule below the pre-skip and make the stream undecodable.
        packet = bytes([0x08])
        packets = [packet] * 6
        path = os.path.join(self.tmp.name, "preskip.ogg")
        awc.write_ogg_opus(path, packets, 1, 23, 4800, pre_skip=65535)
        with open(path, "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(struct.unpack_from("<H", pages[0]["body"], 10)[0], 5760)
        self.assertGreaterEqual(pages[-1]["granule"], 5760)
        with self.assertRaises(ValueError):
            awc.write_ogg_opus(path, packets, 1, 24, 4800, pre_skip=65536)

    def test_empty_packet_list_writes_an_eos_page(self):
        # No audio packets is still a valid empty stream: the EOS flag must be present.
        path = os.path.join(self.tmp.name, "empty.ogg")
        self.assertEqual(awc.write_ogg_opus(path, [], 1, 25, 0), 0)
        with open(path, "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[-1]["flags"] & 0x04, 0x04)
        self.assertEqual(pages[-1]["granule"], 0)

    def test_cli_pre_skip_override(self):
        bank, _, _ = self.make_bank()
        subprocess.run([sys.executable, "-I", awc.__file__, bank, self.out, "--jobs", "1", "--pre-skip", "0"],
                       check=True, capture_output=True)
        with open(os.path.join(self.out, "synthetic", "00000003.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        self.assertEqual(struct.unpack_from("<H", pages[0]["body"], 10)[0], 0)
        self.assertEqual(pages[-1]["granule"], 1920)


class DuplicateIdTest(unittest.TestCase):
    """Two streams in one bank can share a 29-bit id; the second used to overwrite the first's file."""

    def test_streams_with_the_same_id_both_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dup.awc")
            fmt = format_block(2, 48000, 0x00, 1)
            with open(path, "wb") as f:
                f.write(build_bank([(0x123, fmt, struct.pack("<hh", 1, 2)),
                                    (0x123, fmt, struct.pack("<hh", 3, 4))]))
            counts = awc.export_bank(path, os.path.join(tmp, "out"))
            od = os.path.join(tmp, "out", "dup")
            self.assertEqual(counts["pcm"], 2)
            self.assertEqual(sorted(os.listdir(od)), ["00000123.wav", "00000123~2.wav"])
            self.assertNotEqual(read_wav(os.path.join(od, "00000123.wav")),
                                read_wav(os.path.join(od, "00000123~2.wav")))


class StreamedMusicTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def export(self, block_tocs, samples):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank(block_tocs, samples))
        counts = awc.export_bank(path, self.out)
        return counts, os.path.join(self.out, "streamed")

    def test_stereo_packets_export_one_stereo_ogg(self):
        counts, od = self.export([[0xFC], [0xFC]], 1920)      # 2 x 960 interleaved samples
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 1})
        with open(os.path.join(od, "00000100.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        version, channels, preskip, in_rate, gain, mapping = struct.unpack_from("<BBHIhB", pages[0]["body"], 8)
        self.assertEqual((version, channels, preskip, in_rate, gain, mapping), (1, 2, 312, 48000, 0, 0))
        self.assertEqual(pages[-1]["granule"], 312 + 960)

    def test_short_final_block_keeps_its_music(self):
        # With two groups, a chunk normally occupies 6144 bytes. The last group in the second
        # chunk has no subblocks, so the bank only stores the 4096 bytes that chunk needs.
        bank = bytearray(build_music_bank([[0x78, 0x78], [0x78, 0x78]], 1920, last_size=4096))
        _, music = awc.parse_awc(bank)
        _, (offset, _) = music[0]
        struct.pack_into("<I", bank, offset + 6144 + 24 + 4, 0)
        channels = awc.music_channels(bank, *music[0])
        self.assertEqual([len(channel[4]) for channel in channels], [2, 1])
        path = Path(self.tmp.name) / "short-final.awc"
        path.write_bytes(bank)
        self.assertEqual(awc.export_bank(str(path), self.out)["music"], 2)
        for sid, granule in (("00000100", 1920), ("00000101", 960)):
            output = Path(self.out, "short-final", sid + ".ogg")
            self.assertEqual(parse_ogg_pages(output.read_bytes())[-1]["granule"], granule)

    def test_removes_only_verified_packet_repeats_from_each_group(self):
        a, b, c = b"\xf8a", b"\xf0b", b"\xf8c"   # 960, 480, 960 samples
        x, y, z = b"\xfcx", b"\xfcy", b"\xfcz"   # stereo, 960 each
        bank = build_repeating_music_bank([
            [(0, [a, b]), (0, [x, y])],
            [(1440, [a, b, c]), (960, [y, z])],
        ], group_samples=[2400, 2 * 2880])
        _, music = awc.parse_awc(bank)
        channels = awc.music_channels(bank, *music[0])
        self.assertEqual([group[4] for group in channels], [[a, b, c], [x, y, z]])
        metadata = awc.music_channels(bank, *music[0], collect_packets=False)
        self.assertEqual([(group[4], group[5]) for group in metadata], [([], 1), ([], 2)])

        path = Path(self.tmp.name) / "repeated.awc"
        path.write_bytes(bank)
        self.assertEqual(awc.export_bank(str(path), self.out, pre_skip=0)["music"], 2)
        for sid, expected, frames in (("00000100", a + b + c, 2400), ("00000101", x + y + z, 2880)):
            pages = parse_ogg_pages(Path(self.out, "repeated", sid + ".ogg").read_bytes())
            self.assertEqual(b"".join(page["body"] for page in pages[2:]), expected)
            self.assertEqual(pages[-1]["granule"], frames)

    def test_rejects_inconsistent_repeats_in_both_parse_modes(self):
        a, b = b"\xf8a", b"\xf8b"
        cases = {
            "repeat in first chunk": [[(960, [a])]],
            "partial packet": [[(0, [a])], [(959, [a, b])]],
            "different tail": [[(0, [a, b])], [(960, [a])]],
            "beyond the current payload": [[(0, [a])], [(1920, [a])]],
            "beyond the previous payload": [[(0, [a])], [(1920, [a, b])]],
            "empty chunk claims a repeat": [[(0, [a])], [(960, [])]],
            "repeat from the other group": [[(0, [a]), (0, [b])], [(960, [b]), (0, [b])]],
        }
        for name, chunks in cases.items():
            with self.subTest(name=name):
                bank = build_repeating_music_bank(chunks)
                _, music = awc.parse_awc(bank)
                for collect in (True, False):
                    self.assertEqual(awc.music_channels(bank, *music[0], collect_packets=collect), [])

    def test_two_mono_groups_export_two_mono_oggs(self):
        counts, od = self.export([[0x78, 0x78], [0x78, 0x78]], 960)
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 2})
        for sid in ("00000100", "00000101"):
            with self.subTest(sid=sid):
                with open(os.path.join(od, sid + ".ogg"), "rb") as f:
                    pages = parse_ogg_pages(f.read())
                channels = pages[0]["body"][9]
                self.assertEqual(channels, 1)
                self.assertEqual(pages[-1]["granule"], 312 + 960)

    def test_two_stereo_groups_export_two_stereo_oggs(self):
        counts, od = self.export([[0xFC, 0xFC], [0xFC, 0xFC]], 1920)
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 2})
        for sid in ("00000100", "00000101"):
            with self.subTest(sid=sid):
                with open(os.path.join(od, sid + ".ogg"), "rb") as f:
                    pages = parse_ogg_pages(f.read())
                self.assertEqual(pages[0]["body"][9], 2)
                self.assertEqual(pages[-1]["granule"], 312 + 960)

    def test_channel_count_includes_later_blocks(self):
        for tocs in ((0xF8, 0xFC), (0xFC, 0xF8)):
            with self.subTest(tocs=tocs):
                blocks = [[toc] for toc in tocs]
                declared_frames = 1600
                _, od = self.export(blocks, 2 * declared_frames)
                with open(os.path.join(od, "00000100.ogg"), "rb") as f:
                    pages = parse_ogg_pages(f.read())
                self.assertEqual(pages[0]["body"][9], 2)
                self.assertEqual(pages[-1]["granule"], 312 + declared_frames)
                self.assertEqual(pages[-1]["body"], bytes(tocs))

                bank = build_music_bank(blocks, 2 * declared_frames)
                _, music = awc.parse_awc(bank)
                group, = awc.music_channels(bank, *music[0], collect_packets=False)
                self.assertEqual((group[1], group[4], group[5]), (declared_frames, [], 2))

    def test_music_channels_halves_interleaved_samples(self):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank([[0xFC]], 1920))
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        info = awc.music_channels(d, music[0][0], music[0][1])
        self.assertEqual((info[0][1], info[0][5]), (960, 2))

    def test_music_channels_metadata_only_still_reports_packet_channels(self):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_bank([[0xFC]], 1920))
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        info = awc.music_channels(d, music[0][0], music[0][1], collect_packets=False)
        self.assertEqual((info[0][1], info[0][4], info[0][5]), (960, [], 2))


class MalformedMusicGeometryTest(unittest.TestCase):
    """A music descriptor that declares more blocks or subblocks than its data tag holds is
    rejected, instead of reading past the tag into a sibling stream or past the end of the bank."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def write_bank(self, **kw):
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(build_music_pcm_bank(**kw))
        return path

    def test_blocks_beyond_data_tag_keep_sibling_streams(self):
        # Before the fix: struct.error out of music_channels, so main() skipped the whole
        # bank and the valid PCM16 stream was never exported.
        path = self.write_bank(blocks=3)
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        self.assertEqual(awc.music_channels(d, *music[0]), [])
        self.assertEqual(awc.music_channels(d, *music[0], collect_packets=False), [])
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 0, "music": 0})
        od = os.path.join(self.out, "streamed")
        self.assertTrue(os.path.isfile(os.path.join(od, "00000021.wav")))
        self.assertFalse(os.path.exists(os.path.join(od, "00000100.ogg")))

    def test_short_descriptor_at_bank_end_keeps_sibling_streams(self):
        bank = bytearray(build_music_pcm_bank())
        struct.pack_into("<Q", bank, 24, tag(0x48, len(bank), 4))
        bank += bytes(4)  # the descriptor tag fits, but its fixed header does not
        path = Path(self.tmp.name) / "short-descriptor.awc"
        path.write_bytes(bank)
        counts = awc.export_bank(str(path), self.out)
        self.assertEqual((counts["pcm"], counts["music"]), (1, 0))
        self.assertTrue(Path(self.out, "short-descriptor", "00000021.wav").is_file())

    def test_overread_into_sibling_payload_exports_no_music(self):
        # The bytes right after the one real chunk are the PCM stream's payload, crafted to
        # look like a second chunk. Before the fix its 'packet' was exported as music audio.
        pcm = bytearray(4096)
        struct.pack_into("<hh", pcm, 0, 11, 22)
        struct.pack_into("<I", pcm, 4, 1)
        struct.pack_into("<HB", pcm, 2048, 1, 0x78)
        path = self.write_bank(blocks=2, pcm=bytes(pcm))
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 0, "music": 0})
        od = os.path.join(self.out, "streamed")
        self.assertFalse(os.path.exists(os.path.join(od, "00000100.ogg")))
        self.assertEqual(read_wav(os.path.join(od, "00000021.wav"))["pcm"], bytes(pcm))

    def test_subblock_run_beyond_chunk_rejected(self):
        # count=2 needs a 2048-byte header plus 2 data subblocks, but the data tag holds
        # only 4096 bytes. Before the fix the second subblock was read from the PCM payload.
        path = self.write_bank(count=2)
        with open(path, "rb") as f:
            d = f.read()
        _, music = awc.parse_awc(d)
        self.assertEqual(awc.music_channels(d, *music[0]), [])
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 0, "music": 0})

    def test_subblocks_cannot_cross_into_the_next_chunk(self):
        bank = bytearray(build_music_bank([[0x78], [0x78]], 1920))
        _, music = awc.parse_awc(bank)
        _, (offset, _) = music[0]
        struct.pack_into("<I", bank, offset + 4, 2)  # first chunk only has room for one subblock
        self.assertEqual(awc.music_channels(bank, *music[0]), [])
        self.assertEqual(awc.music_channels(bank, *music[0], collect_packets=False), [])

    def test_short_last_chunk_cannot_read_a_partial_channel_table(self):
        bank = build_music_bank([[0x78], [0x78]], 1920, last_size=8)
        _, music = awc.parse_awc(bank)
        self.assertEqual(awc.music_channels(bank, *music[0]), [])

    def test_cli_exports_the_sibling_stream_without_skipping_the_bank(self):
        path = self.write_bank(blocks=3)
        result = subprocess.run([sys.executable, "-I", awc.__file__, path, self.out, "--jobs", "1"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("1 WAV", result.stdout)
        self.assertNotIn("skip", result.stdout)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "streamed", "00000021.wav")))

    def test_valid_combined_bank_still_exports_music(self):
        path = self.write_bank()
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 0, "music": 1})
        od = os.path.join(self.out, "streamed")
        with open(os.path.join(od, "00000100.ogg"), "rb") as f:
            pages = parse_ogg_pages(f.read())
        # One 20 ms packet is 960 decoded samples; the declared 960 leaves no room for the
        # 312-sample pre-skip, so the final granule is the packet total.
        self.assertEqual(pages[-1]["granule"], 960)
        self.assertEqual(pages[-1]["body"], bytes([0x78]))
        self.assertTrue(os.path.isfile(os.path.join(od, "00000021.wav")))


class ShortLastMusicBlockTest(unittest.TestCase):
    """Game banks declare a large `chunk` stride and end the data tag inside the last block: blocks * chunk is
    larger than the tag (1 block of a 524,288-byte chunk in a 264,192-byte tag on a 1.0.2.1 copy). Such a
    descriptor is valid and its channels must still export; only blocks that start past the tag are refused."""

    @staticmethod
    def bank(blocks, stride, tail_cut=0):
        """A mono music bank: each block is a 2048-byte record header plus one 2048-byte packet subblock,
        laid out `stride` bytes apart; the data tag ends right after the last block's subblock."""
        desc = bytearray(12 + 16)
        struct.pack_into("<III", desc, 0, blocks, stride, 1)
        struct.pack_into("<II", desc, 12, 0x100, 960 * blocks)
        struct.pack_into("<H", desc, 22, 48000)
        desc[24] = 0x0B
        data = bytearray((blocks - 1) * stride + 4096 - tail_cut)
        for b in range(blocks):
            at = b * stride
            struct.pack_into("<I", data, at + 4, 1)
            if at + 2048 + 3 <= len(data):
                struct.pack_into("<HB", data, at + 2048, 1, 0x78)
        header = 16 + 4 + 2 * 8
        out = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header))
        out += struct.pack("<I", 0x20 | (2 << 29))
        out += struct.pack("<Q", tag(0x48, header, len(desc)))
        out += struct.pack("<Q", tag(0x55, header + len(desc), len(data)))
        return bytes(out + desc + data)

    def channels(self, d):
        _, music = awc.parse_awc(d)
        return awc.music_channels(d, *music[0])

    def test_single_short_block_exports(self):
        ch = self.channels(self.bank(1, 524288))
        self.assertEqual([(c[0], len(c[4])) for c in ch], [(0x100, 1)])

    def test_last_block_shorter_than_the_stride_exports_every_block(self):
        ch = self.channels(self.bank(3, 16384))
        self.assertEqual([(c[0], len(c[4])) for c in ch], [(0x100, 3)])

    def test_a_last_block_cut_inside_its_subblock_is_still_refused(self):
        # The data tag ends 1,024 bytes into the last block's packet subblock: the subblock check refuses it.
        self.assertEqual(self.channels(self.bank(2, 16384, tail_cut=1024)), [])

    def test_a_block_starting_at_or_past_the_tag_end_is_refused(self):
        d = bytearray(self.bank(2, 16384))
        _, music = awc.parse_awc(bytes(d))
        (m_at, _), _ = music[0]
        struct.pack_into("<I", d, m_at, 3)           # declare a third block starting past the tag
        self.assertEqual(self.channels(bytes(d)), [])


class AwcIndexTableTest(unittest.TestCase):
    """The optional index table stores the tag count before each stream, as AwcRecovery.cs checks."""

    @staticmethod
    def bank(indices, tag_counts=(2,)):
        n = len(indices)
        header_size = 16 + n * 2 + n * 4 + sum(tag_counts) * 8
        body = bytearray()
        entries = []
        for count in tag_counts:
            fmt_at = header_size + len(body)
            body += format_block(2, 48000, 0, 1)
            data_at = header_size + len(body)
            body += b"\x01\x00\x02\x00"
            entries.append((fmt_at, data_at, count))
        out = bytearray(b"ADAT" + struct.pack("<III", 0x10000, n, header_size))
        for index in indices:
            out += struct.pack("<H", index)
        for i, (_fmt, _data, count) in enumerate(entries):
            out += struct.pack("<I", (i + 1) | (count << 29))
        for fmt_at, data_at, _count in entries:
            out += struct.pack("<Q", tag(0x76, fmt_at, 32))
            out += struct.pack("<Q", tag(0x55, data_at, 4))
        if len(out) != header_size:
            raise AssertionError("index-table fixture layout does not match header_size")
        return bytes(out + body)

    def test_valid_index_table_is_accepted(self):
        streams, music = awc.parse_awc(self.bank([0]))
        self.assertEqual([s["id"] for s in streams], [1])
        self.assertEqual((streams[0]["samples"], streams[0]["rate"]), (2, 48000))
        self.assertEqual(music, [])

    def test_bogus_index_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "index"):
            awc.parse_awc(self.bank([0x1234]))

    def test_index_tracks_cumulative_tag_count(self):
        with self.assertRaisesRegex(ValueError, "index"):
            awc.parse_awc(self.bank([0, 1], (2, 2)))
        streams, _music = awc.parse_awc(self.bank([0, 2], (2, 2)))
        self.assertEqual([s["id"] for s in streams], [1, 2])


class MusicChannelFormatTest(unittest.TestCase):
    """AwcRecovery.cs:380 rejects music channels outside 1000..192000 Hz or not coded as Opus."""

    RATE_AT = 56 + 22      # one-channel descriptor starts at 56; rate is at +22
    CODEC_AT = 56 + 24

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def bank(self, *, rate=None, codec=None):
        d = bytearray(build_music_pcm_bank())
        if rate is not None:
            struct.pack_into("<H", d, self.RATE_AT, rate)
        if codec is not None:
            d[self.CODEC_AT] = codec
        return bytes(d)

    def channels(self, d):
        _, music = awc.parse_awc(d)
        return awc.music_channels(d, *music[0], collect_packets=False)

    def test_zero_rate_is_rejected_and_sibling_pcm_is_kept(self):
        d = self.bank(rate=0)
        self.assertEqual(self.channels(d), [])
        path = os.path.join(self.tmp.name, "streamed.awc")
        with open(path, "wb") as f:
            f.write(d)
        counts = awc.export_bank(path, self.out)
        self.assertEqual(counts, {"opus": 0, "pcm": 1, "other": 0, "music": 0})
        self.assertTrue(os.path.isfile(os.path.join(self.out, "streamed", "00000021.wav")))

    def test_rate_below_reference_bound_is_rejected(self):
        self.assertEqual(self.channels(self.bank(rate=999)), [])

    def test_reference_rate_bounds_are_accepted(self):
        # The field is u16, so 65535 is the highest rate a bank can actually declare.
        for rate in (1000, 65535):
            with self.subTest(rate=rate):
                self.assertEqual(len(self.channels(self.bank(rate=rate))), 1)

    def test_non_opus_codec_is_rejected(self):
        self.assertEqual(self.channels(self.bank(codec=0)), [])


class MalformedBankTest(unittest.TestCase):
    def test_truncated_pcm_bank_creates_no_outputs(self):
        pcm = b"\x01\x00" * 4
        fmt = format_block(4, 48000, 0, 1)
        bank = build_bank([(1, fmt, pcm), (2, fmt, pcm)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.awc")
            with open(path, "wb") as f:
                f.write(bank[:-2])
            out = os.path.join(tmp, "out")
            with self.assertRaisesRegex(ValueError, "0x55.*outside"):
                awc.export_bank(path, out)
            self.assertFalse(os.path.exists(out))

    def test_bad_magic_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.awc")
            with open(path, "wb") as f:
                f.write(b"XATD" + b"\x00" * 60)
            with self.assertRaises(ValueError):
                awc.export_bank(path, os.path.join(tmp, "out"))

    def test_main_skips_bad_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.awc")
            with open(path, "wb") as f:
                f.write(b"XATD" + b"\x00" * 60)
            out = io.StringIO()
            # --jobs 1: the module is loaded under a private name, which worker processes cannot import
            argv = ["awc_opus.py", path, os.path.join(tmp, "out"), "--jobs", "1"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                self.assertEqual(awc.main(), 0)
            self.assertIn("skip bad.awc", out.getvalue())


class WriteFileReadOnlyTest(unittest.TestCase):
    def test_failed_replace_over_a_read_only_file_leaves_no_staging_file(self):
        real_remove = os.remove

        def windows_remove(p):
            if not os.access(p, os.W_OK):
                raise PermissionError(13, "read-only file", p)
            real_remove(p)

        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "a.ogg")
            with open(dest, "wb") as f:
                f.write(b"old")
            os.chmod(dest, 0o444)
            try:
                with mock.patch.object(awc.os, "replace", side_effect=OSError("replace failed")), \
                        mock.patch.object(awc.os, "remove", side_effect=windows_remove):
                    with self.assertRaises(OSError):
                        awc.write_file(dest, b"new")
                self.assertEqual(sorted(os.listdir(d)), ["a.ogg"])
                with open(dest, "rb") as f:
                    self.assertEqual(f.read(), b"old")
            finally:
                os.chmod(dest, 0o644)


if __name__ == "__main__":
    unittest.main(verbosity=2)
