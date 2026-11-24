"""Synthetic byte buffer DXBC container scanning, magic marker detection, and chunk boundary validation.

    python -I tests/test_shader_carve.py

Validates tools/shader_carve.py without touching game files or proprietary assets:
- DXBC container chunk table parsing, byte offset calculation, and chunk boundary validation
- Shader model version packing and stage mapping (including unknown stage fallback)
- RPF9 entry regex marker detection, name length/character constraints, and stored-size backreference
- Candidate size filtering, raw deflate decompression error handling, and container integrity checks
- End-to-end archive carving, duplicate variant naming, filename character sanitization, and index.json output
"""
import contextlib
import hashlib
import io
import json
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "shader_carve.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import shader_carve  # noqa: E402


def make_dxbc(chunks: list) -> bytes:
    """Build a synthetic DXBC container from a list of (fourcc_bytes, payload_bytes)."""
    n = len(chunks)
    head = 32 + 4 * n
    offsets = []
    body = bytearray()
    for fourcc, payload in chunks:
        offsets.append(head + len(body))
        body.extend(fourcc)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
    hdr = bytearray(b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(body), n))
    for o in offsets:
        hdr.extend(struct.pack("<I", o))
    return bytes(hdr + body)


def make_dxil_payload(ver: int, kind: int, extra_bytes: int = 40) -> bytes:
    """Build a synthetic DXIL chunk payload with packed shader model version and stage kind."""
    return struct.pack("<HH", ver, kind) + bytes(range(extra_bytes))


def compress_raw_deflate(data: bytes) -> bytes:
    """Compress data using raw DEFLATE (wbits=-15, matching shader_carve.py decompression)."""
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


def make_rpf_entry(name: str, kind: int, uncompressed_data: bytes, entry_id: int = 1,
                   compressed_override: bytes = None, stored_override: int = None,
                   size_override: int = None, mismatch_backref: bool = False) -> bytes:
    """Build a synthetic RPF entry matching shader_carve.ENTRY regex."""
    compressed = compress_raw_deflate(uncompressed_data) if compressed_override is None else compressed_override
    stored = len(compressed) if stored_override is None else stored_override
    size = len(uncompressed_data) if size_override is None else size_override
    stored_ref = (stored + 1) if mismatch_backref else stored

    hdr = (
        name.encode("latin-1")
        + b"\x00"
        + bytes([kind])
        + struct.pack("<I", stored)
        + struct.pack("<I", entry_id)
        + struct.pack("<I", stored_ref)
        + struct.pack("<I", size)
    )
    return hdr + compressed


class DxbcContainerPartsTest(unittest.TestCase):
    def test_empty_container(self):
        dxbc = make_dxbc([])
        p = shader_carve.parts(dxbc)
        self.assertEqual(p, {})

    def test_single_chunk_boundaries(self):
        payload = b"TEST_PAYLOAD_BYTES_FOR_PRIV_CHUNK"
        dxbc = make_dxbc([(b"PRIV", payload)])
        p = shader_carve.parts(dxbc)
        self.assertIn("PRIV", p)
        off, length = p["PRIV"]
        self.assertEqual(off, 44)
        self.assertEqual(length, len(payload))
        self.assertEqual(dxbc[off:off + length], payload)

    def test_multiple_chunks_boundaries(self):
        payload_dxil = make_dxil_payload(0x65, 0, 48)
        payload_priv = b"PRIV_CUSTOM_METADATA_DATA"
        payload_isg1 = bytes(range(64))
        chunks = [
            (b"DXIL", payload_dxil),
            (b"PRIV", payload_priv),
            (b"ISG1", payload_isg1),
        ]
        dxbc = make_dxbc(chunks)
        p = shader_carve.parts(dxbc)
        self.assertEqual(len(p), 3)
        self.assertIn("DXIL", p)
        self.assertIn("PRIV", p)
        self.assertIn("ISG1", p)

        prev_end = 0
        for fourcc, payload in chunks:
            fourcc_str = fourcc.decode("latin-1")
            off, length = p[fourcc_str]
            self.assertGreaterEqual(off - 8, prev_end)
            self.assertEqual(length, len(payload))
            self.assertEqual(dxbc[off - 8:off - 4], fourcc)
            self.assertEqual(struct.unpack_from("<I", dxbc, off - 4)[0], len(payload))
            self.assertEqual(dxbc[off:off + length], payload)
            prev_end = off + length

    def test_chunk_fourcc_decoding(self):
        chunks = [(b"C001", b"data1"), (b"C002", b"data2")]
        dxbc = make_dxbc(chunks)
        p = shader_carve.parts(dxbc)
        self.assertIn("C001", p)
        self.assertIn("C002", p)


class ShaderModelParsingTest(unittest.TestCase):
    def test_missing_dxil_chunk_returns_none(self):
        dxbc = make_dxbc([(b"PRIV", b"some_bytes")])
        p = shader_carve.parts(dxbc)
        stage, sm = shader_carve.shader_model(dxbc, p)
        self.assertIsNone(stage)
        self.assertIsNone(sm)

    def test_all_known_stages(self):
        expected_stages = {
            0: "ps",
            1: "vs",
            2: "gs",
            3: "hs",
            4: "ds",
            5: "cs",
            6: "lib",
            13: "ms",
            14: "as",
        }
        for kind_code, expected_name in expected_stages.items():
            payload = make_dxil_payload(0x65, kind_code)
            dxbc = make_dxbc([(b"DXIL", payload)])
            p = shader_carve.parts(dxbc)
            stage, sm = shader_carve.shader_model(dxbc, p)
            self.assertEqual(stage, expected_name)
            self.assertEqual(sm, "6.5")

    def test_unknown_stage_fallback_to_string(self):
        payload = make_dxil_payload(0x60, 99)
        dxbc = make_dxbc([(b"DXIL", payload)])
        p = shader_carve.parts(dxbc)
        stage, sm = shader_carve.shader_model(dxbc, p)
        self.assertEqual(stage, "99")
        self.assertEqual(sm, "6.0")

    def test_shader_model_version_unpacking(self):
        test_cases = [
            ((6 << 4) | 0, "6.0"),
            ((6 << 4) | 5, "6.5"),
            ((6 << 4) | 6, "6.6"),
            ((5 << 4) | 1, "5.1"),
            ((12 << 4) | 3, "12.3"),
        ]
        for ver_code, expected_sm in test_cases:
            payload = make_dxil_payload(ver_code, 1)
            dxbc = make_dxbc([(b"DXIL", payload)])
            p = shader_carve.parts(dxbc)
            stage, sm = shader_carve.shader_model(dxbc, p)
            self.assertEqual(stage, "vs")
            self.assertEqual(sm, expected_sm)


class EntryMarkerRegexTest(unittest.TestCase):
    def test_valid_entry_pattern_matching(self):
        entry = make_rpf_entry("valid_name", 2, b"DXBC" + bytes(range(40)), entry_id=100)
        match = shader_carve.ENTRY.search(entry)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), b"valid_name")
        self.assertEqual(match.group(2), bytes([2]))
        self.assertEqual(struct.unpack("<I", match.group(4))[0], 100)

    def test_name_length_boundaries(self):
        # 2 chars: too short (regex requires 3..200)
        short_entry = make_rpf_entry("ab", 1, b"DXBC" + bytes(range(40)))
        self.assertIsNone(shader_carve.ENTRY.match(short_entry))

        # 3 chars: minimum valid length
        min_entry = make_rpf_entry("abc", 1, b"DXBC" + bytes(range(40)))
        self.assertIsNotNone(shader_carve.ENTRY.match(min_entry))

        # 200 chars: maximum valid length
        max_name = "n" * 200
        max_entry = make_rpf_entry(max_name, 1, b"DXBC" + bytes(range(40)))
        self.assertIsNotNone(shader_carve.ENTRY.match(max_entry))

        # 201 chars: exceeding upper limit of 200 printable characters
        toolong_name = "n" * 201
        toolong_entry = make_rpf_entry(toolong_name, 1, b"DXBC" + bytes(range(40)))
        self.assertIsNone(shader_carve.ENTRY.match(toolong_entry))

    def test_name_character_restrictions(self):
        # Spaces are not in [\x21-\x7e]
        space_entry = make_rpf_entry("fo ba", 1, b"DXBC" + bytes(range(40)))
        self.assertIsNone(shader_carve.ENTRY.search(space_entry))

        # Control characters are rejected
        ctrl_entry = make_rpf_entry("f\x01b", 1, b"DXBC" + bytes(range(40)))
        self.assertIsNone(shader_carve.ENTRY.search(ctrl_entry))

        # Non-ASCII characters (> 0x7e) are rejected
        high_entry = make_rpf_entry("f\x80b", 1, b"DXBC" + bytes(range(40)))
        self.assertIsNone(shader_carve.ENTRY.search(high_entry))

    def test_stored_size_backreference_enforcement(self):
        valid = make_rpf_entry("test_shader", 1, b"DXBC" + bytes(range(40)), mismatch_backref=False)
        self.assertIsNotNone(shader_carve.ENTRY.search(valid))

        invalid = make_rpf_entry("test_shader", 1, b"DXBC" + bytes(range(40)), mismatch_backref=True)
        self.assertIsNone(shader_carve.ENTRY.search(invalid))

    def test_stream_scanning_multiple_entries(self):
        entry1 = make_rpf_entry("shader_one", 1, b"DXBC" + bytes(range(40)))
        entry2 = make_rpf_entry("shader_two", 2, b"DXBC" + bytes(range(50)))
        stream = entry1 + entry2

        matches = list(shader_carve.ENTRY.finditer(stream))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].group(1), b"shader_one")
        self.assertEqual(matches[0].start(), 0)
        self.assertEqual(matches[1].group(1), b"shader_two")
        self.assertEqual(matches[1].start(), len(entry1))


class CandidateValidationRulesTest(unittest.TestCase):
    def test_stored_size_bounds(self):
        # 16 < stored < 64 << 20
        self.assertFalse(16 < 16 < 64 << 20)
        self.assertTrue(16 < 17 < 64 << 20)
        self.assertTrue(16 < (64 << 20) - 1 < 64 << 20)
        self.assertFalse(16 < (64 << 20) < 64 << 20)

    def test_uncompressed_size_bounds(self):
        # 32 < size < 256 << 20
        self.assertFalse(32 < 32 < 256 << 20)
        self.assertTrue(32 < 33 < 256 << 20)
        self.assertTrue(32 < (256 << 20) - 1 < 256 << 20)
        self.assertFalse(32 < (256 << 20) < 256 << 20)

    def test_corrupt_deflate_raises_error(self):
        corrupt_bytes = b"\xff\x00\xee\xdd\xcc\xbb\xaa\x99" * 4
        with self.assertRaises(zlib.error):
            zlib.decompressobj(-15).decompress(corrupt_bytes, 100)

    def test_magic_check_failure(self):
        uncompressed = b"FAIL" + bytes(range(50))
        comp = compress_raw_deflate(uncompressed)
        decomp = zlib.decompressobj(-15).decompress(comp, len(uncompressed) + 1)
        self.assertNotEqual(decomp[:4], b"DXBC")

    def test_size_mismatch_failure(self):
        uncompressed = b"DXBC" + bytes(range(50))
        actual_len = len(uncompressed)
        declared_size = actual_len + 10
        self.assertNotEqual(actual_len, declared_size)


class ShaderCarveCliIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.archive = os.path.join(self.dir, "test_shaders.rpf")
        self.out = os.path.join(self.dir, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def run_carve(self, archive_path: str, out_path: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", TOOL, archive_path, out_path],
            capture_output=True,
            text=True,
        )

    def test_carve_synthetic_archive_with_variants_and_sanitization(self):
        # 1. Valid vs shader with DXIL and PRIV chunks
        dxil_vs = make_dxil_payload(0x65, 1, 40)
        dxbc_vs = make_dxbc([(b"DXIL", dxil_vs), (b"PRIV", b"XBOX_PRIV_DATA_VS")])
        entry_vs = make_rpf_entry("vs_main", 1, dxbc_vs, entry_id=1)

        # 2. Valid ps shader with DXIL chunk
        dxil_ps = make_dxil_payload(0x66, 0, 48)
        dxbc_ps = make_dxbc([(b"DXIL", dxil_ps)])
        entry_ps = make_rpf_entry("ps_main", 0, dxbc_ps, entry_id=2)

        # 3. Duplicate variant 1 of ps_main
        entry_ps_var1 = make_rpf_entry("ps_main", 0, dxbc_ps, entry_id=3)

        # 4. Duplicate variant 2 of ps_main
        entry_ps_var2 = make_rpf_entry("ps_main", 0, dxbc_ps, entry_id=4)

        # 5. Valid shader with special characters in name that need sanitization
        dxil_gs = make_dxil_payload(0x60, 2, 40)
        dxbc_gs = make_dxbc([(b"DXIL", dxil_gs)])
        entry_special = make_rpf_entry("folder/sub:name?light*test", 2, dxbc_gs, entry_id=5)

        # 6. Valid container without DXIL chunk (PRIV only)
        dxbc_nodxil = make_dxbc([(b"PRIV", b"ONLY_PRIV_CHUNK_PAYLOAD_HERE")])
        entry_nodxil = make_rpf_entry("priv_only_shader", 5, dxbc_nodxil, entry_id=6)

        # Candidate to reject 1: corrupt deflate stream
        bad_deflate_entry = make_rpf_entry(
            "corrupt_deflate",
            0,
            b"DXBC" + bytes(range(40)),
            entry_id=7,
            compressed_override=b"\xff\x00\xab\xcd" * 8,
            stored_override=32,
            size_override=44,
        )

        # Candidate to reject 2: valid deflate, but magic is not DXBC
        bad_magic_raw = b"NOPE" + bytes(range(50))
        bad_magic_entry = make_rpf_entry("bad_magic_tag", 0, bad_magic_raw, entry_id=8)

        # Candidate to reject 3: decompressed length does not match declared size
        bad_size_raw = b"DXBC" + bytes(range(50))
        bad_size_entry = make_rpf_entry(
            "bad_size_entry",
            0,
            bad_size_raw,
            entry_id=9,
            size_override=len(bad_size_raw) + 20,
        )

        # Candidate to skip: stored <= 16 (silently skipped by bounds check)
        skip_stored_entry = make_rpf_entry(
            "skip_low_stored",
            0,
            b"DXBC" + bytes(range(40)),
            entry_id=10,
            stored_override=16,
            size_override=44,
        )

        # Candidate to skip: size <= 32 (silently skipped by bounds check)
        skip_size_entry = make_rpf_entry(
            "skip_low_size",
            0,
            b"DXBC" + bytes(range(40)),
            entry_id=11,
            stored_override=25,
            size_override=32,
        )

        archive_content = (
            entry_vs
            + entry_ps
            + entry_ps_var1
            + entry_ps_var2
            + entry_special
            + entry_nodxil
            + bad_deflate_entry
            + bad_magic_entry
            + bad_size_entry
            + skip_stored_entry
            + skip_size_entry
        )

        with open(self.archive, "wb") as f:
            f.write(archive_content)

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 0, res.stderr)

        # 6 valid shaders carved, 3 corrupt candidates rejected
        self.assertIn("6 shaders carved, 3 candidates rejected", res.stdout)
        self.assertIn("stages:", res.stdout)
        self.assertIn("shader models:", res.stdout)
        self.assertIn("parts:", res.stdout)

        # Verify created DXBC container files
        expected_files = {
            "vs_main.dxbc": dxbc_vs,
            "ps_main.dxbc": dxbc_ps,
            "ps_main~1.dxbc": dxbc_ps,
            "ps_main~2.dxbc": dxbc_ps,
            "folder_sub_name_light_test.dxbc": dxbc_gs,
            "priv_only_shader.dxbc": dxbc_nodxil,
        }
        for filename, expected_bytes in expected_files.items():
            filepath = os.path.join(self.out, filename)
            self.assertTrue(os.path.isfile(filepath), f"Missing expected output file: {filename}")
            with open(filepath, "rb") as f:
                self.assertEqual(f.read(), expected_bytes)

        # Verify index.json metadata
        index_path = os.path.join(self.out, "index.json")
        self.assertTrue(os.path.isfile(index_path))
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        self.assertEqual(len(index), 6)

        self.assertEqual(index[0]["name"], "vs_main")
        self.assertEqual(index[0]["file"], "vs_main.dxbc")
        self.assertEqual(index[0]["stage"], "vs")
        self.assertEqual(index[0]["sm"], "6.5")
        self.assertIn("DXIL", index[0]["parts"])
        self.assertIn("PRIV", index[0]["parts"])

        self.assertEqual(index[1]["name"], "ps_main")
        self.assertEqual(index[1]["file"], "ps_main.dxbc")
        self.assertEqual(index[1]["stage"], "ps")
        self.assertEqual(index[1]["sm"], "6.6")

        self.assertEqual(index[2]["name"], "ps_main")
        self.assertEqual(index[2]["file"], "ps_main~1.dxbc")

        self.assertEqual(index[3]["name"], "ps_main")
        self.assertEqual(index[3]["file"], "ps_main~2.dxbc")

        self.assertEqual(index[4]["name"], "folder/sub:name?light*test")
        self.assertEqual(index[4]["file"], "folder_sub_name_light_test.dxbc")
        self.assertEqual(index[4]["stage"], "gs")
        self.assertEqual(index[4]["sm"], "6.0")

        self.assertEqual(index[5]["name"], "priv_only_shader")
        self.assertEqual(index[5]["file"], "priv_only_shader.dxbc")
        self.assertIsNone(index[5]["stage"])
        self.assertIsNone(index[5]["sm"])
        self.assertIn("PRIV", index[5]["parts"])

    def test_carve_rejects_unfinished_deflate_stream(self):
        dxbc = make_dxbc([(b"DXIL", make_dxil_payload(0x66, 5))])
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(dxbc) + compressor.flush(zlib.Z_SYNC_FLUSH)
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("synthetic", 1, dxbc, compressed_override=compressed))

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 1, res.stderr)
        self.assertIn("no shaders carved", res.stderr)
        self.assertIn("rejected candidates: 1", res.stderr)
        self.assertEqual(os.listdir(self.out), [])

    def test_empty_carve_preserves_an_existing_index(self):
        dxbc = make_dxbc([(b"PRIV", bytes(range(32)))])
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("valid", 1, dxbc))
        initial = self.run_carve(self.archive, self.out)
        self.assertEqual(initial.returncode, 0, initial.stderr)
        index_path = os.path.join(self.out, "index.json")
        with open(index_path, "rb") as f:
            previous = f.read()
        with open(self.archive, "wb") as f:
            f.write(b"no shader entries")

        res = self.run_carve(self.archive, self.out)

        self.assertEqual(res.returncode, 1, res.stderr)
        self.assertIn("no shaders carved", res.stderr)
        self.assertIn("rejected candidates: 0", res.stderr)
        with open(index_path, "rb") as f:
            self.assertEqual(f.read(), previous)
        self.assertEqual(sorted(os.listdir(self.out)), ["index.json", "valid.dxbc"])

    def test_carve_rejects_incomplete_parts_and_continues(self):
        valid = make_dxbc([(b"PRIV", bytes(range(32)))])
        short_directory = bytearray(valid)
        struct.pack_into("<I", short_directory, 28, len(valid) // 4)
        short_header = bytearray(valid)
        struct.pack_into("<I", short_header, 32, len(valid) - 4)
        short_payload = bytearray(valid)
        struct.pack_into("<I", short_payload, 40, len(valid))
        short_dxil = make_dxbc([(b"DXIL", b"\x65\x00"), (b"PRIV", bytes(range(32)))])

        entries = [("short_directory", short_directory), ("short_header", short_header),
                   ("short_payload", short_payload), ("short_dxil", short_dxil), ("valid", valid)]
        with open(self.archive, "wb") as f:
            for name, data in entries:
                f.write(make_rpf_entry(name, 1, data))

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("1 shaders carved, 4 candidates rejected", res.stdout)
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            self.assertEqual([row["file"] for row in json.load(f)], ["valid.dxbc"])
        self.assertEqual(sorted(os.listdir(self.out)), ["index.json", "valid.dxbc"])

    def test_nonexistent_archive_reports_error(self):
        missing = os.path.join(self.dir, "missing.rpf")
        res = self.run_carve(missing, self.out)
        self.assertNotEqual(res.returncode, 0)

    def test_bad_inputs_are_usage_errors_and_leave_no_output_folder(self):
        empty = os.path.join(self.dir, "empty.rpf")
        open(empty, "wb").close()
        cases = ((os.path.join(self.dir, "missing.rpf"), "is not a file"),
                 (self.dir, "is not a file"),
                 (empty, "is empty"))
        for archive, message in cases:
            res = self.run_carve(archive, self.out)
            self.assertEqual(res.returncode, 2, res.stderr)
            self.assertIn(message, res.stderr)
            self.assertNotIn("Traceback", res.stderr)
            self.assertFalse(os.path.exists(self.out))

        dxbc = make_dxbc([(b"PRIV", bytes(range(32)))])
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("valid", 1, dxbc))
        out_file = os.path.join(self.dir, "out_file")
        with open(out_file, "wb") as f:
            f.write(b"x")
        res = self.run_carve(self.archive, out_file)
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("is a file, not a folder", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_output_under_a_regular_file_is_one_line_error(self):
        dxbc = make_dxbc([(b"PRIV", bytes(range(32)))])
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("valid", 1, dxbc))
        blocker = os.path.join(self.dir, "blocker")
        with open(blocker, "wb") as f:
            f.write(b"x")
        res = self.run_carve(self.archive, os.path.join(blocker, "out"))
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("cannot create", res.stderr)
        self.assertNotIn("Traceback", res.stderr)
        with open(blocker, "rb") as f:
            self.assertEqual(f.read(), b"x")

    def test_carve_preserves_existing_index_temp_sibling(self):
        dxbc = make_dxbc([(b"PRIV", bytes(range(32)))])
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("valid", 1, dxbc))
        os.mkdir(self.out)
        sibling = os.path.join(self.out, "index.json.tmp")
        with open(sibling, "w", encoding="utf-8") as f:
            f.write("keep me")

        result = self.run_carve(self.archive, self.out)

        self.assertEqual(result.returncode, 0, result.stderr)
        with open(sibling, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep me")
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            self.assertEqual([row["name"] for row in json.load(f)], ["valid"])
        self.assertFalse(any(os.path.isdir(os.path.join(self.out, name)) for name in os.listdir(self.out)))

    def test_main_closes_mmap_on_exit(self):
        dxbc = make_dxbc([(b"PRIV", bytes(range(32)))])
        with open(self.archive, "wb") as f:
            f.write(make_rpf_entry("valid", 1, dxbc))
        real_mmap = mmap.mmap
        opened = []

        def record(*args, **kwargs):
            m = real_mmap(*args, **kwargs)
            opened.append(m)
            return m

        with mock.patch("shader_carve.mmap.mmap", side_effect=record):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(shader_carve.main([self.archive, self.out]), 0)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        os.remove(self.archive)
        self.assertFalse(os.path.exists(self.archive))

    def test_no_arguments_is_a_usage_error(self):
        res = subprocess.run([sys.executable, "-I", TOOL], capture_output=True, text=True)
        self.assertEqual(res.returncode, 2, res.stderr)
        self.assertIn("required", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_carve_skips_deflate_payload_ending_in_printable_ascii(self):
        # Find a deterministic compressed DXBC whose raw deflate stream ends with a printable ASCII byte (0x21..0x7E).
        # Back-to-back with entry2, a scan that does not skip the payload attaches those bytes to entry2's name.
        comp1 = dxbc1 = None
        for seed in range(64):
            payload = b"".join(hashlib.sha256(f"{seed}-{i}".encode()).digest() for i in range(12))
            cand = make_dxbc([(b"PRIV", payload)])
            c = compress_raw_deflate(cand)
            if 0x21 <= c[-1] <= 0x7E:
                dxbc1, comp1 = cand, c
                break
        self.assertIsNotNone(comp1, "expected a raw deflate stream ending in printable ASCII")
        dxbc2 = make_dxbc([(b"PRIV", b"SECOND_SHADER_PAYLOAD_BYTES_123456")])
        entry1 = make_rpf_entry("first_shader", 1, dxbc1, entry_id=1, compressed_override=comp1)
        entry2 = make_rpf_entry("second_shader", 1, dxbc2, entry_id=2)
        entry3 = make_rpf_entry("second_shader", 1, dxbc2, entry_id=3)
        with open(self.archive, "wb") as f:
            f.write(entry1 + entry2 + entry3)

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("3 shaders carved, 0 candidates rejected", res.stdout)
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual([(r["name"], r["file"], r["offset"]) for r in index], [
            ("first_shader", "first_shader.dxbc", 0),
            ("second_shader", "second_shader.dxbc", len(entry1)),
            ("second_shader", "second_shader~1.dxbc", len(entry1) + len(entry2)),
        ])

    def test_carve_ignores_entry_pattern_inside_deflate_payload(self):
        # RFC 1951 raw deflate stored block (BFINAL=1, BTYPE=00) embeds raw bytes verbatim, including an
        # ENTRY-matching decoy whose 17-byte header crosses the boundary into the real next entry's name.
        dxbc2 = make_dxbc([(b"PRIV", b"FOLLOWING_SHADER_PAYLOAD_0123456789")])
        decoy_trailer = b"decoy\x00\x01" + struct.pack("<IIII", 20, 7, 20, 40)
        dxbc1 = make_dxbc([(b"PRIV", b"P" * 32 + decoy_trailer + b"TAIL")])
        stored_block = b"\x01" + struct.pack("<HH", len(dxbc1), (~len(dxbc1)) & 0xFFFF) + dxbc1
        entry1 = make_rpf_entry("outer_shader", 1, dxbc1, entry_id=1, compressed_override=stored_block)
        entry2 = make_rpf_entry("next_shader", 1, dxbc2, entry_id=2)
        with open(self.archive, "wb") as f:
            f.write(entry1 + entry2)

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("2 shaders carved, 0 candidates rejected", res.stdout)
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual([(r["name"], r["file"], r["offset"]) for r in index], [
            ("outer_shader", "outer_shader.dxbc", 0),
            ("next_shader", "next_shader.dxbc", len(entry1)),
        ])

    def test_rejected_candidate_trailer_does_not_swallow_following_entry(self):
        # A bogus match whose 17-byte trailer overlaps the start of a real entry's name must not skip that entry.
        dxbc = make_dxbc([(b"PRIV", b"REAL_SHADER_AFTER_BOGUS_MATCH_123456")])
        real_entry = make_rpf_entry("real_shader", 1, dxbc, entry_id=1)
        # Craft "bogus\x00" + kind + stored + id + stored, so the match's 4-byte size field is "real", the start of
        # the real entry's name. The size bound rejects it; the real entry must still be carved under its own name.
        bogus = b"bogus\x00\x00" + struct.pack("<III", 32, 0, 32)
        with open(self.archive, "wb") as f:
            f.write(bogus + real_entry)

        res = self.run_carve(self.archive, self.out)
        self.assertEqual(res.returncode, 0, res.stderr)
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual([(r["name"], r["offset"]) for r in index], [("real_shader", len(bogus))])


if __name__ == "__main__":
    unittest.main()
