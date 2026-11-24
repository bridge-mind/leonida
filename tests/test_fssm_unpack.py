"""Unit tests for tools/fssm_unpack.py covering XBCH carving, FSSM headers, alignment, and CRC bounds.

    python -I tests/test_fssm_unpack.py

Zero game data: all tests use deterministic, synthetic byte sequences constructed in-memory.
"""
import errno
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fssm_unpack  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "fssm_unpack.py")
S_ENDPGM = b"\x00\x00\x81\xbf"


def make_fssm(version=0x0101, instructions=None, pad_to=0x200):
    blob = bytearray(pad_to)
    blob[0:4] = b"FSSM"
    struct.pack_into("<H", blob, 4, version)
    if instructions:
        for off, code in instructions:
            blob[off:off + len(code)] = code
    return bytes(blob)


def make_xbch(fssm_data, flags_crc=0, stored_override=None, unpacked_override=None, corrupt_zlib=False):
    compressed = zlib.compress(fssm_data)
    if corrupt_zlib:
        corrupted = bytearray(compressed)
        corrupted[-4:] = b"\x00\x00\x00\x00"
        compressed = bytes(corrupted)
    stored = len(compressed) if stored_override is None else stored_override
    unpacked = len(fssm_data) if unpacked_override is None else unpacked_override
    header = struct.pack("<4sIIIII", b"XBCH", flags_crc, stored, 0, unpacked, 0)
    return header + compressed


def make_xdat(entries, pre_padding=b""):
    count = len(entries)
    hdr_size = 8
    table_len = 12 + 12 * count
    dxil_base_offset = len(pre_padding) + table_len

    table_entries = []
    payloads = bytearray()
    for tag in sorted(entries.keys()):
        payload = entries[tag]
        off = dxil_base_offset + len(payloads)
        table_entries.append(struct.pack("<III", tag, off, len(payload)))
        payloads.extend(payload)

    xdat_header = b"XDAT" + struct.pack("<II", count, hdr_size) + b"".join(table_entries)
    return pre_padding + xdat_header + bytes(payloads)


def make_xdat_list(entries, pre_padding=b""):
    """Like make_xdat, but entries is an ordered list so duplicate tags can be represented."""
    count = len(entries)
    table_len = 12 + 12 * count
    dxil_base_offset = len(pre_padding) + table_len
    table_entries = []
    payloads = bytearray()
    for tag, payload in entries:
        off = dxil_base_offset + len(payloads)
        table_entries.append(struct.pack("<III", tag, off, len(payload)))
        payloads.extend(payload)
    xdat_header = b"XDAT" + struct.pack("<II", count, 8) + b"".join(table_entries)
    return pre_padding + xdat_header + bytes(payloads)


def make_dxbc(chunks, digest=bytes(16), version=1):
    count = len(chunks)
    head_len = 32 + 4 * count
    offsets = []
    body = bytearray()
    for fourcc, payload in chunks:
        offsets.append(head_len + len(body))
        body.extend(fourcc)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
    total_size = head_len + len(body)
    header = b"DXBC" + digest + struct.pack("<III", version, total_size, count)
    offset_table = b"".join(struct.pack("<I", off) for off in offsets)
    return header + offset_table + bytes(body)


class TestPartsParsing(unittest.TestCase):
    def test_empty_chunks(self):
        container = make_dxbc([])
        self.assertEqual(fssm_unpack.parts(container), {})

    def test_single_chunk(self):
        container = make_dxbc([(b"DXIL", b"sample-dxil-data")])
        parts = fssm_unpack.parts(container)
        self.assertEqual(parts, {b"DXIL": b"sample-dxil-data"})

    def test_multiple_chunks(self):
        chunks = [
            (b"DXIL", b"dxil-content"),
            (b"PRIV", b"priv-content"),
            (b"ISG1", b"isg1-content"),
        ]
        container = make_dxbc(chunks)
        parts = fssm_unpack.parts(container)
        self.assertEqual(parts, {
            b"DXIL": b"dxil-content",
            b"PRIV": b"priv-content",
            b"ISG1": b"isg1-content",
        })

    def test_chunk_with_empty_payload(self):
        container = make_dxbc([(b"DXIL", b"")])
        self.assertEqual(fssm_unpack.parts(container), {b"DXIL": b""})


class TestDirectoryBounds(unittest.TestCase):
    def test_dxbc_bounds(self):
        blob = make_dxbc([(b"TEST", b"abc")])
        table = 32
        part = table + 4
        for label, data in [
            ("header", blob[:31]),
            ("offset table", blob[:part - 1]),
            ("part header", blob[:part + 7]),
            ("part payload", blob[:-1]),
        ]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "DXBC"):
                fssm_unpack.parts(data)

        for offset in (table, len(blob) + 1):
            data = bytearray(blob)
            struct.pack_into("<I", data, table, offset)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "DXBC"):
                fssm_unpack.parts(bytes(data))

    def test_xdat_bounds(self):
        prefix = b"prefix"
        blob = make_xdat({7: b"abc"}, pre_padding=prefix)
        x = len(prefix)
        for label, data in [
            ("header", blob[:x + 11]),
            ("directory", blob[:x + 23]),
            ("payload", blob[:-1]),
        ]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "XDAT"):
                fssm_unpack.xdat_entries_all(data)

        for offset, size in [(x + 12, 1), (len(blob) + 1, 0), (len(blob), 1)]:
            data = bytearray(blob)
            struct.pack_into("<II", data, x + 16, offset, size)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "XDAT"):
                fssm_unpack.xdat_entries_all(bytes(data))

    def test_empty_payload_at_end_of_dxil(self):
        self.assertEqual(fssm_unpack.xdat_entries_all(make_xdat({7: b""})), [(7, b"")])


class TestXdatEntries(unittest.TestCase):
    def test_missing_xdat_magic(self):
        dxil = b"arbitrary-dxil-bytes-without-magic"
        self.assertEqual(fssm_unpack.xdat_entries(dxil), {})

    def test_xdat_with_zero_count(self):
        dxil = b"prefix" + b"XDAT" + struct.pack("<II", 0, 8)
        self.assertEqual(fssm_unpack.xdat_entries(dxil), {})

    def test_xdat_single_entry(self):
        entries = {0x0D: b"xbch-blob-data"}
        dxil = make_xdat(entries)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)

    def test_xdat_multiple_entries_and_relative_offsets(self):
        entries = {
            0x07: b"dxil-program-header",
            0x0D: b"xbch-fssm-container",
            0x12: b"\xab" * 16,
        }
        pre_padding = b"\x00" * 32
        dxil = make_xdat(entries, pre_padding=pre_padding)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)

    def test_xdat_zero_size_entry(self):
        entries = {0x01: b"", 0x02: b"non-empty"}
        dxil = make_xdat(entries)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)


class TestDuplicateXdatPayloads(unittest.TestCase):
    def test_duplicate_tag_keeps_order_and_selects_first_or_last(self):
        first = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        last = make_fssm(version=0x0102, instructions=[(0x100, S_ENDPGM)])
        xdat = make_xdat_list([
            (0x12, b"\x11" * 16),
            (0x0D, make_xbch(first)),
            (0x0D, make_xbch(last)),
        ])
        entries = fssm_unpack.xdat_entries_all(xdat)
        self.assertEqual([tag for tag, _ in entries], [0x12, 0x0D, 0x0D])
        self.assertEqual(entries[1][1], make_xbch(first))
        self.assertEqual(entries[2][1], make_xbch(last))
        self.assertEqual(fssm_unpack.xdat_entries(xdat)[0x0D], make_xbch(last))
        dxbc = make_dxbc([(b"DXIL", xdat)])
        data, _ = fssm_unpack.unpack(dxbc)
        self.assertEqual(data, last)
        data, _ = fssm_unpack.unpack(dxbc, payload="first")
        self.assertEqual(data, first)
        with self.assertRaises(ValueError):
            fssm_unpack.unpack(dxbc, payload="middle")

    def test_cli_payload_first(self):
        first = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        last = make_fssm(version=0x0102, instructions=[(0x100, S_ENDPGM)])
        xdat = make_xdat_list([(0x0D, make_xbch(first)), (0x0D, make_xbch(last))])
        dxbc = make_dxbc([(b"DXIL", xdat)])
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "dup.dxbc"), "wb") as f:
                f.write(dxbc)
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--payload", "first"],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(tmp, "fssm", "dup.fssm"), "rb") as f:
                self.assertEqual(f.read(), first)


class TestXbchContainerCarvingAndUnpack(unittest.TestCase):
    def test_unpack_valid_container(self):
        fssm_payload = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        xbch_blob = make_xbch(fssm_payload)
        dxil = make_xdat({0x0D: xbch_blob, 0x12: b"\x11" * 16})
        dxbc = make_dxbc([(b"DXIL", dxil), (b"PRIV", b"priv-data")])

        data, ents = fssm_unpack.unpack(dxbc)
        self.assertEqual(data, fssm_payload)
        self.assertIn(0x0D, ents)
        self.assertIn(0x12, ents)

    def test_unpack_missing_dxil_chunk(self):
        dxbc = make_dxbc([(b"PRIV", b"priv-data-only")])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(ents, {})

    def test_unpack_missing_xdat_magic(self):
        dxbc = make_dxbc([(b"DXIL", b"no-xdat-here")])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(ents, {})

    def test_unpack_missing_xbch_tag(self):
        dxil = make_xdat({0x07: b"other-tag-content"})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(list(ents.keys()), [0x07])

    def test_unpack_invalid_xbch_magic(self):
        bad_blob = b"NOBC" + bytes(20) + zlib.compress(b"FSSM\x00\x00")
        dxil = make_xdat({0x0D: bad_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertIn(0x0D, ents)

    def test_unpack_truncated_xbch_header(self):
        bad_blob = b"XBCH" + bytes(8)
        dxil = make_xdat({0x0D: bad_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(struct.error):
            fssm_unpack.unpack(dxbc)

    def test_unpack_size_mismatch_raises_value_error(self):
        fssm_payload = make_fssm()
        xbch_blob = make_xbch(fssm_payload, unpacked_override=len(fssm_payload) + 10)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(ValueError) as ctx:
            fssm_unpack.unpack(dxbc)
        self.assertIn("size mismatch", str(ctx.exception))

    def test_unpack_rejects_declared_size_over_fixed_limit_before_inflating(self):
        fssm_payload = make_fssm(pad_to=64)
        xbch_blob = make_xbch(fssm_payload, unpacked_override=65)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        real_decompressobj = zlib.decompressobj

        with mock.patch.object(fssm_unpack, "MAX_FSSM_BYTES", 64, create=True), \
                mock.patch.object(fssm_unpack.zlib, "decompressobj", wraps=real_decompressobj) as decompressobj:
            with self.assertRaisesRegex(ValueError, "exceeds limit"):
                fssm_unpack.unpack(dxbc)
            decompressobj.assert_not_called()

    def test_unpack_caps_expansion_at_declared_size_plus_one(self):
        fssm_payload = make_fssm(pad_to=4096)
        xbch_blob = make_xbch(fssm_payload, unpacked_override=32)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        real_decompressobj = zlib.decompressobj
        observed = []

        class RecordingDecompressor:
            def __init__(self, *args, **kwargs):
                self.inner = real_decompressobj(*args, **kwargs)

            def decompress(self, data, max_length=0):
                output = self.inner.decompress(data, max_length)
                observed.append((max_length, len(output)))
                return output

            @property
            def eof(self):
                return self.inner.eof

        with mock.patch.object(fssm_unpack, "MAX_FSSM_BYTES", 64, create=True), \
                mock.patch.object(fssm_unpack.zlib, "decompressobj", side_effect=RecordingDecompressor):
            with self.assertRaisesRegex(ValueError, "exceeds declared size"):
                fssm_unpack.unpack(dxbc)

        self.assertEqual(observed, [(33, 33)])

    def test_unpack_rejects_truncated_zlib_stream(self):
        fssm_payload = make_fssm()
        compressed = zlib.compress(fssm_payload)[:-4]
        header = struct.pack("<4sIIIII", b"XBCH", 0, len(compressed), 0, len(fssm_payload), 0)
        xbch_blob = header + compressed
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])

        with self.assertRaisesRegex(ValueError, "truncated XBCH zlib stream"):
            fssm_unpack.unpack(dxbc)

    def test_unpack_rejects_trailing_data_after_zlib_stream(self):
        fssm_payload = make_fssm()
        compressed = zlib.compress(fssm_payload) + b"trailing"
        header = struct.pack("<4sIIIII", b"XBCH", 0, len(compressed), 0, len(fssm_payload), 0)
        xbch_blob = header + compressed
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])

        with self.assertRaisesRegex(ValueError, "trailing data after XBCH zlib stream"):
            fssm_unpack.unpack(dxbc)

    def test_unpack_rejects_mismatched_stored_size(self):
        fssm_payload = make_fssm()
        xbch_blob = make_xbch(fssm_payload, stored_override=1)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])

        with self.assertRaisesRegex(ValueError, "stored size"):
            fssm_unpack.unpack(dxbc)

    def test_unpack_corrupt_zlib_raises_error(self):
        fssm_payload = make_fssm()
        xbch_blob = make_xbch(fssm_payload, corrupt_zlib=True)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(zlib.error):
            fssm_unpack.unpack(dxbc)


class TestFssmChunkHeadersAndMetadata(unittest.TestCase):
    def test_fssm_magic_and_version(self):
        blob_v257 = make_fssm(version=0x0101)
        self.assertEqual(blob_v257[:4].decode("latin-1"), "FSSM")
        self.assertEqual(struct.unpack_from("<H", blob_v257, 4)[0], 0x0101)

        blob_v512 = make_fssm(version=0x0200)
        self.assertEqual(blob_v512[:4].decode("latin-1"), "FSSM")
        self.assertEqual(struct.unpack_from("<H", blob_v512, 4)[0], 0x0200)


class TestPayloadAlignment(unittest.TestCase):
    def test_sendpgm_aligned_offsets_detected(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x20:0x24] = S_ENDPGM
        blob[0x80:0x84] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [0x20, 0x80])

    def test_sendpgm_unaligned_offsets_rejected(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x21:0x25] = S_ENDPGM
        blob[0x43:0x47] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [])

    def test_sendpgm_mixed_aligned_and_unaligned(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x11:0x15] = S_ENDPGM
        blob[0x20:0x24] = S_ENDPGM
        blob[0x33:0x37] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [0x20])

    def test_sendpgm_short_fssm_bounds(self):
        for size in (0, 1, 2, 3):
            blob = b"FSS"[:size]
            ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
            self.assertEqual(ends, [])


class TestCrcAndChecksumBounds(unittest.TestCase):
    def test_dxbc_container_digest_bounds(self):
        fssm_data = make_fssm()
        xbch = make_xbch(fssm_data)
        dxil = make_xdat({0x0D: xbch})

        digests = [
            bytes(16),
            b"\xff" * 16,
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        ]
        for digest in digests:
            dxbc = make_dxbc([(b"DXIL", dxil)], digest=digest)
            data, ents = fssm_unpack.unpack(dxbc)
            self.assertEqual(data, fssm_data)

    def test_xdat_hash_tag_0x12_preservation(self):
        hash_16 = b"\xca\xfe\xba\xbe\xde\xad\xbe\xef\x01\x02\x03\x04\x05\x06\x07\x08"
        dxil = make_xdat({0x0D: make_xbch(make_fssm()), 0x12: hash_16})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        _, ents = fssm_unpack.unpack(dxbc)
        self.assertEqual(ents[0x12], hash_16)

    def test_xbch_flags_crc_bounds(self):
        fssm_data = make_fssm()
        test_crcs = [0x00000000, 0xFFFFFFFF, 0x80000000, 0x12345678]
        for crc_val in test_crcs:
            xbch = make_xbch(fssm_data, flags_crc=crc_val)
            dxil = make_xdat({0x0D: xbch})
            dxbc = make_dxbc([(b"DXIL", dxil)])
            data, _ = fssm_unpack.unpack(dxbc)
            self.assertEqual(data, fssm_data)

    def test_zlib_adler32_integrity_rejection(self):
        fssm_data = make_fssm()
        raw_compressed = zlib.compress(fssm_data)
        corrupted_adler = raw_compressed[:-4] + bytes([raw_compressed[-4] ^ 0xFF]) + raw_compressed[-3:]
        header = struct.pack("<4sIIIII", b"XBCH", 0, len(corrupted_adler), 0, len(fssm_data), 0)
        xbch = header + corrupted_adler
        dxil = make_xdat({0x0D: xbch})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(zlib.error):
            fssm_unpack.unpack(dxbc)


class TestMainBadOutput(unittest.TestCase):
    def test_file_at_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as src:
            out = os.path.join(src, "fssm")
            with open(out, "wb") as f:
                f.write(b"keep me")

            res = subprocess.run([sys.executable, "-I", TOOL, src], capture_output=True, text=True, errors="replace")

            self.assertEqual(res.returncode, 2)
            self.assertEqual(res.stdout, "")
            self.assertEqual(len(res.stderr.splitlines()), 1, res.stderr)
            self.assertIn(f"cannot create output directory {out}", res.stderr)
            self.assertNotIn("Traceback", res.stderr)
            self.assertEqual(os.listdir(src), ["fssm"])
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"keep me")

    def test_output_directory_creation_errors_name_the_path(self):
        with tempfile.TemporaryDirectory() as src:
            out = os.path.join(src, "fssm")
            for error in (PermissionError(errno.EACCES, "Permission denied"),
                          OSError(errno.ENOSPC, "No space left on device")):
                with self.subTest(error=error), mock.patch.object(fssm_unpack.os, "makedirs",
                                                                 side_effect=error) as mkdir, \
                        mock.patch.object(sys, "stderr", io.StringIO()) as stderr:
                    self.assertEqual(fssm_unpack.main([src]), 2)
                    mkdir.assert_called_once_with(out, exist_ok=True)
                    self.assertEqual(stderr.getvalue().splitlines(),
                                     [f"ERROR: cannot create output directory {out}: {error.strerror}"])
                    self.assertEqual(os.listdir(src), [])


class TestMainBadSource(unittest.TestCase):
    """A source that is missing, a file, or unlistable is one line on stderr, exit 2, and nothing is created."""

    def run_tool(self, src):
        return subprocess.run([sys.executable, "-I", TOOL, src], capture_output=True, text=True, errors="replace")

    def test_missing_source_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "carve", "shaders")      # mistyped: neither folder exists
            res = self.run_tool(src)
            self.assertEqual(res.returncode, 2)
            self.assertIn(src, res.stderr)
            self.assertIn("does not exist", res.stderr)
            self.assertNotIn("Traceback", res.stderr)
            self.assertEqual(res.stdout, "")
            self.assertEqual(os.listdir(tmp), [])            # on main: carve/shaders/fssm was made

    def test_file_as_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shader_a.dxbc")
            with open(src, "wb") as f:
                f.write(b"DXBC")
            res = self.run_tool(src)
            self.assertEqual(res.returncode, 2)
            self.assertIn("is not a folder", res.stderr)
            self.assertNotIn("Traceback", res.stderr)
            self.assertEqual(os.listdir(tmp), ["shader_a.dxbc"])

    def test_unlistable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with mock.patch.object(fssm_unpack.os, "scandir", side_effect=PermissionError(13, "Permission denied")), \
                    mock.patch.object(sys, "stderr", out):
                self.assertEqual(fssm_unpack.main([tmp]), 2)
            self.assertIn("cannot list", out.getvalue())
            self.assertIn("Permission denied", out.getvalue())
            self.assertFalse(os.path.exists(os.path.join(tmp, "fssm")))


    def test_unreadable_shader_leaves_completion_index_unchanged(self):
        with tempfile.TemporaryDirectory() as src:
            shader = os.path.join(src, "input.dxbc")
            index = os.path.join(src, "fssm_index.json")
            with open(shader, "wb") as f:
                f.write(b"synthetic")
            with open(index, "wb") as f:
                f.write(b"previous index")
            error = PermissionError(errno.EACCES, "input is locked", shader)
            with mock.patch.object(fssm_unpack, "open", create=True, side_effect=error), \
                    mock.patch.object(sys, "stderr", io.StringIO()) as stderr:
                self.assertEqual(fssm_unpack.main([src]), 2)
            self.assertEqual(stderr.getvalue().splitlines(), [f"ERROR: cannot read {shader}: input is locked"])
            with open(index, "rb") as f:
                self.assertEqual(f.read(), b"previous index")


class TestMainDirectoryProcessing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_carve_index_selects_current_containers(self):
        shader = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(make_fssm())}))])
        for name in ("current", "obsolete"):
            with open(os.path.join(self.src, name + ".dxbc"), "wb") as f:
                f.write(shader)
        index = os.path.join(self.src, "index.json")
        with open(index, "w", encoding="utf-8") as f:
            json.dump([{"file": "current.dxbc"}], f)
        result = subprocess.run([sys.executable, "-I", TOOL, self.src, "--carve-index", index],
                                capture_output=True, text=True, errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(os.listdir(os.path.join(self.src, "fssm")), ["current.fssm"])
        self.assertTrue(os.path.isfile(os.path.join(self.src, "obsolete.dxbc")))

    def test_main_rejects_invalid_fssm_magic(self):
        bad_payload = b"JUNK\x01\x01" + bytes(58)
        bad_dxbc = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(bad_payload)}))])
        bad_path = os.path.join(self.src, "bad.dxbc")
        with open(bad_path, "wb") as f:
            f.write(bad_dxbc)

        cmd = [sys.executable, "-I", TOOL, self.src]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        self.assertEqual(res.returncode, 1, res.stdout)
        self.assertIn("'ValueError': 1", res.stdout)
        self.assertIn("no shader in", res.stderr)
        fssm_dir = os.path.join(self.src, "fssm")
        self.assertEqual(os.listdir(fssm_dir) if os.path.isdir(fssm_dir) else [], [])
        self.assertFalse(os.path.exists(os.path.join(self.src, "fssm_index.json")))

        good_dxbc = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(make_fssm())}))])
        with open(os.path.join(self.src, "good.dxbc"), "wb") as f:
            f.write(good_dxbc)

        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("'ValueError': 1", res.stdout)
        self.assertEqual(os.listdir(fssm_dir), ["good.fssm"])
        with open(os.path.join(self.src, "fssm_index.json"), encoding="utf-8") as f:
            self.assertEqual([row["name"] for row in json.load(f)], ["good"])

    def test_main_cli_processing_and_index_generation(self):
        fssm_1 = make_fssm(version=0x0101, instructions=[(0x40, S_ENDPGM)])
        fssm_2 = make_fssm(version=0x0102, instructions=[(0x20, S_ENDPGM), (0x60, S_ENDPGM)])

        dxbc_1 = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1), 0x12: b"\xaa" * 16}))])
        dxbc_2 = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_2)}))])
        dxbc_no_xbch = make_dxbc([(b"DXIL", make_xdat({0x07: b"dxil-hdr"}))])
        dxbc_mismatch = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1, unpacked_override=9999)}))])
        dxbc_bad_zlib = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1, corrupt_zlib=True)}))])
        dxbc_short_fssm = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(b"FSSM")}))])

        with open(os.path.join(self.src, "shader_a.dxbc"), "wb") as f:
            f.write(dxbc_1)
        with open(os.path.join(self.src, "shader_b.dxbc"), "wb") as f:
            f.write(dxbc_2)
        with open(os.path.join(self.src, "no_xbch.dxbc"), "wb") as f:
            f.write(dxbc_no_xbch)
        with open(os.path.join(self.src, "mismatch.dxbc"), "wb") as f:
            f.write(dxbc_mismatch)
        with open(os.path.join(self.src, "bad_zlib.dxbc"), "wb") as f:
            f.write(dxbc_bad_zlib)
        with open(os.path.join(self.src, "short_fssm.dxbc"), "wb") as f:
            f.write(dxbc_short_fssm)
        with open(os.path.join(self.src, "ignored.txt"), "w") as f:
            f.write("not a dxbc container")
        os.mkdir(os.path.join(self.src, "cache.dxbc"))

        res = subprocess.run([sys.executable, "-I", TOOL, self.src], capture_output=True, text=True, errors="replace")
        self.assertEqual(res.returncode, 0, res.stderr)

        self.assertIn("2 FSSM blobs", res.stdout)
        self.assertIn("no XBCH", res.stdout)
        self.assertIn("ValueError", res.stdout)
        self.assertIn("error", res.stdout)
        self.assertIn("magic: {'FSSM': 2}", res.stdout)

        fssm_dir = os.path.join(self.src, "fssm")
        self.assertTrue(os.path.isdir(fssm_dir))
        out_files = sorted(os.listdir(fssm_dir))
        self.assertEqual(out_files, ["shader_a.fssm", "shader_b.fssm"])

        with open(os.path.join(fssm_dir, "shader_a.fssm"), "rb") as f:
            self.assertEqual(f.read(), fssm_1)
        with open(os.path.join(fssm_dir, "shader_b.fssm"), "rb") as f:
            self.assertEqual(f.read(), fssm_2)

        index_path = os.path.join(self.src, "fssm_index.json")
        self.assertTrue(os.path.isfile(index_path))
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        self.assertEqual(len(index), 2)
        entry_a = next(r for r in index if r["name"] == "shader_a")
        self.assertEqual(entry_a["size"], len(fssm_1))
        self.assertEqual(entry_a["magic"], "FSSM")
        self.assertEqual(entry_a["version"], 0x0101)
        self.assertEqual(entry_a["endpgm"], [0x40])
        self.assertEqual(entry_a["tags"], ["0x12", "0xd"])

        entry_b = next(r for r in index if r["name"] == "shader_b")
        self.assertEqual(entry_b["size"], len(fssm_2))
        self.assertEqual(entry_b["magic"], "FSSM")
        self.assertEqual(entry_b["version"], 0x0102)
        self.assertEqual(entry_b["endpgm"], [0x20, 0x60])
        self.assertEqual(entry_b["tags"], ["0xd"])

    def test_index_publication_never_uses_fixed_tmp_sentinel(self):
        """The old fixed fssm_index.json.tmp staging path could truncate an unrelated user file (or a
        hardlink to one). Index publication must use private staging and leave any pre-existing
        .tmp sentinel untouched."""
        fssm_1 = make_fssm(version=0x0101, instructions=[(0x40, S_ENDPGM)])
        with open(os.path.join(self.src, "shader_a.dxbc"), "wb") as f:
            f.write(make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1)}))]))

        sentinel = os.path.join(self.src, "fssm_index.json.tmp")
        with open(sentinel, "w") as f:
            f.write("keep-me")

        res = subprocess.run([sys.executable, "-I", TOOL, self.src], capture_output=True, text=True, errors="replace")
        self.assertEqual(res.returncode, 0, res.stderr)

        with open(sentinel, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep-me")
        with open(os.path.join(self.src, "fssm_index.json"), encoding="utf-8") as f:
            self.assertEqual([row["name"] for row in json.load(f)], ["shader_a"])
        self.assertEqual(sorted(os.listdir(self.src)),
                         ["fssm", "fssm_index.json", "fssm_index.json.tmp", "shader_a.dxbc"])


if __name__ == "__main__":
    unittest.main()
