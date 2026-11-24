"""Tests for tools/world_census.py, built from synthetic 0x7a resources (no game data).

    python -I tests/test_world_census.py
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WC = load("world_census_under_test", os.path.join(ROOT, "tools", "world_census.py"))
TWI = load("world_index_test_builder", os.path.join(ROOT, "tests", "test_world_index.py"))


def make_archive(path):
    """A small RPF9: normal map, zlib-split map, raw-split map, meta and a duplicate."""
    out = bytearray(0x40)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, 0x40)

    def stream(blob):
        pos = (len(out) + 15) & ~15
        out.extend(b"\0" * (pos - len(out)))
        out.extend(zlib.compress(blob))

    def raw(blob):
        pos = (len(out) + 15) & ~15
        out.extend(b"\0" * (pos - len(out)))
        out.extend(blob)

    normal = TWI.make_resource(True)
    stream(normal)
    stream(TWI.make_resource(False))
    zsplit = TWI.make_resource(True)
    stream(zsplit[:0x150])
    stream(zsplit[0x150:])
    split = TWI.make_resource(True)
    stream(split[:0x150])
    raw(split[0x150:])
    stream(normal)
    with open(path, "wb") as f:
        f.write(out)


class WorldCensusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_census(self, shard_size=1 << 30, jobs=1, out_name="out"):
        arc = os.path.join(self.dir, "sample.rpf")
        make_archive(arc)
        out = os.path.join(self.dir, out_name)
        code = WC.main([arc, "--out", out, "--shard-size", str(shard_size), "--jobs", str(jobs)])
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            return code, json.load(f)

    def test_whole_archive_counts_and_coverage(self):
        code, s = self.run_census()
        self.assertEqual(code, 0)
        self.assertEqual(s["maps"], 4)
        self.assertEqual(s["entities"], 4)
        self.assertTrue(s["complete"])
        self.assertTrue(s["census_exact"])
        self.assertEqual(s["coverage"]["accounted"], s["coverage"]["span"])
        self.assertGreater(s["coverage"]["raw_system"], 0)
        self.assertEqual(s["coverage"]["unread"], 0)

    def test_duplicates_are_reported(self):
        _, s = self.run_census()
        self.assertGreaterEqual(s["duplicates"]["content_extra_maps"], 1)
        self.assertGreaterEqual(s["duplicates"]["content_extra_entities"], 1)
        self.assertGreaterEqual(s["duplicates"]["name_groups"], 1)
        self.assertEqual(s["duplicate_offsets"], 0)

    def test_tiny_shards_do_not_double_count(self):
        code, s = self.run_census(shard_size=400, jobs=1)
        self.assertEqual(code, 0)
        self.assertEqual(s["maps"], 4)
        self.assertEqual(s["entities"], 4)
        self.assertEqual(s["duplicate_offsets"], 0)
        self.assertEqual(s["coverage"]["accounted"], s["coverage"]["span"])

    def test_tiny_shards_give_the_same_stream_totals(self):
        """Each shard starts scanning WARMUP bytes early to find its first stream boundary. Streams that start in
        that window belong to the previous shard, so the stream and byte totals must not depend on the shard size."""
        _, whole = self.run_census(out_name="whole")
        _, tiny = self.run_census(shard_size=400, out_name="tiny")
        self.assertGreater(tiny["shards_expected"], 1)
        for key, value in whole["totals"].items():
            with self.subTest(total=key):
                self.assertEqual(tiny["totals"].get(key), value)
        self.assertGreater(whole["totals"]["streams"], 0)

    def test_version_4_warmup_shard_requires_a_rescan(self):
        archive = Path(self.dir) / "sample.rpf"
        out = Path(self.dir) / "out"
        start, shard_size = 0x40, 0x200
        boundary = start + shard_size
        map_data, meta_data = TWI.make_resource(True), TWI.make_resource(False)
        first, second = zlib.compress(map_data), zlib.compress(meta_data)
        data = bytearray(boundary)
        struct.pack_into("<4sI", data, 0, b"9FPR", start)
        data[start:start + len(first)] = first
        data.extend(second)
        archive.write_bytes(data)

        def run(*flags):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = WC.main([str(archive), "--out", str(out), "--shard-size", str(shard_size),
                                "--jobs", "1", *flags])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            return code, stdout.getvalue(), summary

        code, _, fresh = run()
        self.assertEqual(code, 0)
        expected = {"streams": 2, "stored": len(first) + len(second),
                    "inflated": len(map_data) + len(meta_data), "rejected": 0}
        self.assertEqual({k: fresh["totals"][k] for k in expected}, expected)
        self.assertTrue(fresh["census_exact"])
        self.assertEqual(fresh["shards_complete"], 2)
        maps = (out / "maps.json").read_bytes()

        compatible_path, legacy_path = [Path(WC._shard_path(out, s)) for s in fresh["shards"]]
        compatible = compatible_path.read_bytes()
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        # Format 4 counted the first shard's stream again during the second shard's warm-up.
        legacy["format"] = 4
        legacy["intervals"].insert(0, [start, start + len(first)])
        legacy["stats"]["streams"] += 1
        legacy["stats"]["stored"] += len(first)
        legacy["stats"]["inflated"] += len(map_data)
        legacy["stats"]["adopted"] = 1
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        stale = legacy_path.read_bytes()

        code, _, partial = run("--merge-only")
        self.assertEqual(code, 1)
        self.assertEqual(partial["missing_ranges"], [[boundary, len(data)]])
        self.assertEqual(partial["shards_complete"], 1)
        self.assertFalse(partial["census_exact"])
        self.assertEqual(partial["totals"]["streams"], 1)
        self.assertEqual(legacy_path.read_bytes(), stale)

        code, stdout, rebuilt = run("--resume")
        self.assertEqual(code, 0)
        self.assertIn("2 shards, 1 to scan", stdout)
        self.assertEqual(rebuilt["totals"], fresh["totals"])
        self.assertEqual(rebuilt["missing_ranges"], [])
        self.assertTrue(rebuilt["census_exact"])
        self.assertEqual((out / "maps.json").read_bytes(), maps)
        self.assertEqual(compatible_path.read_bytes(), compatible)
        self.assertEqual(json.loads(legacy_path.read_bytes())["format"], WC.FORMAT)

        code, stdout, resumed = run("--resume")
        self.assertEqual(code, 0)
        self.assertIn("2 shards, 0 to scan", stdout)
        code, _, merged = run("--merge-only")
        self.assertEqual(code, 0)
        for summary in (resumed, merged):
            self.assertEqual(summary["totals"], fresh["totals"])
            self.assertTrue(summary["census_exact"])

    def test_required_len_detects_split_page(self):
        blob = TWI.make_resource(True)
        self.assertEqual(WC.required_len(blob), len(blob))
        self.assertGreater(WC.required_len(blob[:0x150]), 0x150)


class CensusArgumentTest(unittest.TestCase):
    """--shard-size 0 planned shards forever (memory grew without bound); --jobs 0 crashed in ProcessPoolExecutor."""

    def test_expected_shards_rejects_a_size_below_one(self):
        self.assertEqual(WC._expected_shards(0, 10, 4),
                         [{"index": 0, "start": 0, "end": 4}, {"index": 1, "start": 4, "end": 8},
                          {"index": 2, "start": 8, "end": 10}])

        class Counted(int):
            """Counts pos + shard_size, so a missing guard fails here instead of looping forever."""
            adds = 0

            def __radd__(self, other):
                Counted.adds += 1
                if Counted.adds > 1000:
                    raise RuntimeError("shard planning never finished")
                return other + int(self)

        for size in (0, -5):
            Counted.adds = 0
            with self.subTest(size=size), self.assertRaises(ValueError):
                WC._expected_shards(0, 10, Counted(size))

    def test_cli_rejects_shard_size_and_jobs_below_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "sample.rpf")
            make_archive(arc)
            out = os.path.join(tmp, "out")
            tool = os.path.join(ROOT, "tools", "world_census.py")
            for flag, value in (("--shard-size", "0"), ("--shard-size", "-1"), ("--jobs", "0")):
                with self.subTest(flag=flag, value=value):
                    # A subprocess with a timeout, so the old endless loop fails the test instead of hanging it.
                    p = subprocess.run([sys.executable, "-I", tool, arc, "--out", out, flag, value],
                                       capture_output=True, text=True, timeout=10)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn("must be at least 1", p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_jobs_above_the_platform_limit_are_rejected(self):
        """--jobs 100 crashed on Windows: ProcessPoolExecutor allows at most 61 workers there."""
        saved = WC.MAX_JOBS
        try:
            WC.MAX_JOBS = 61
            self.assertEqual(WC.job_count("61"), 61)
            for value in ("62", "100"):
                with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError) as ctx:
                    WC.job_count(value)
                self.assertIn("at most 61", str(ctx.exception))
            WC.MAX_JOBS = None
            self.assertEqual(WC.job_count("100"), 100)
        finally:
            WC.MAX_JOBS = saved
        self.assertEqual(WC.MAX_JOBS, 61 if os.name == "nt" else None)

    @unittest.skipUnless(os.name == "nt", "the 61-worker limit is Windows-only")
    def test_cli_rejects_jobs_above_61_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "sample.rpf")
            make_archive(arc)
            tool = os.path.join(ROOT, "tools", "world_census.py")
            p = subprocess.run([sys.executable, "-I", tool, arc, "--out", os.path.join(tmp, "out"), "--jobs", "100"],
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("must be at most 61", p.stderr)
            self.assertNotIn("Traceback", p.stderr)


def aligned(*parts):
    """Concatenate parts, each starting on a 16-byte boundary (the walker only accepts aligned streams)."""
    out = bytearray()
    for part in parts:
        out.extend(b"\0" * (((len(out) + 15) & ~15) - len(out)))
        out.extend(part)
    return bytes(out)


def with_block(blob, name, size, pointer):
    """Add a third block-table entry (the builder leaves 0x220.. free) to a map resource."""
    b = bytearray(blob)
    struct.pack_into("<H", b, 0x4C, 3)
    struct.pack_into("<IIQ", b, 0x220, name, size, pointer)
    return bytes(b)


GARBAGE_ZLIB = b"\x78\x9c" + b"\xff" * 30  # a valid zlib header followed by an invalid deflate block type


class InflateTest(unittest.TestCase):
    def test_non_meta_stream_is_measured_but_not_kept(self):
        payload = b"plain payload " * 8
        comp = zlib.compress(payload)
        m = comp + b"\xaa" * 10
        data, used, eof, is_meta, inflated, truncated = WC.inflate(m, 0, len(m))
        self.assertEqual((data, used, eof, is_meta, inflated, truncated), (b"", len(comp), True, False, len(payload), False))
        data, used, eof, is_meta, _, _ = WC.inflate(m, 0, len(m), keep=True)
        self.assertEqual((data, used, eof, is_meta), (payload, len(comp), True, False))

    def test_meta_stream_is_kept_up_to_the_cap(self):
        blob = TWI.make_resource(True)
        comp = zlib.compress(blob)
        data, used, eof, is_meta, inflated, truncated = WC.inflate(comp, 0, len(comp))
        self.assertEqual((data, used, eof, is_meta, inflated, truncated), (blob, len(comp), True, True, len(blob), False))
        data, _, eof, is_meta, inflated, truncated = WC.inflate(comp, 0, len(comp), max_inflate=0x100)
        self.assertEqual(data, blob[:0x100])
        self.assertTrue(eof and is_meta and truncated)
        self.assertEqual(inflated, len(blob))

    def test_stream_at_an_offset(self):
        blob = TWI.make_resource(False)
        m = aligned(b"\x01" * 0x30, zlib.compress(blob))
        data, used, eof, is_meta, _, _ = WC.inflate(m, 0x30, len(m))
        self.assertTrue(eof and is_meta)
        self.assertEqual(data, blob)
        self.assertEqual(used, len(m) - 0x30)

    def test_stream_that_does_not_end_is_not_accepted(self):
        comp = zlib.compress(TWI.make_resource(True))
        data, used, eof, is_meta, _, truncated = WC.inflate(comp[:len(comp) // 2], 0, len(comp) // 2)
        self.assertEqual((data, used, eof, is_meta, truncated), (b"", 0, False, False, False))

    def test_invalid_deflate_data_raises_zlib_error(self):
        with self.assertRaises(zlib.error):
            WC.inflate(GARBAGE_ZLIB, 0, len(GARBAGE_ZLIB))


class RequiredLenTest(unittest.TestCase):
    def setUp(self):
        self.blob = TWI.make_resource(True)

    def patched(self, fmt, offset, value):
        b = bytearray(self.blob)
        struct.pack_into(fmt, b, offset, value)
        return bytes(b)

    def test_rejects_what_is_not_a_0x7a_header(self):
        self.assertEqual(WC.required_len(b""), 0)
        self.assertEqual(WC.required_len(self.blob[:0x4F]), 0)
        self.assertEqual(WC.required_len(b"\0" * 0x700), 0)
        self.assertEqual(WC.required_len(self.patched("<B", 0x14, 0x79)), 0)

    def test_rejects_bad_counts(self):
        self.assertEqual(WC.required_len(self.patched("<H", 0x48, 0)), 0)       # no structs
        self.assertEqual(WC.required_len(self.patched("<H", 0x48, 4097)), 0)    # too many structs
        self.assertEqual(WC.required_len(self.patched("<H", 0x4C, 0)), 0)       # no blocks
        self.assertEqual(WC.required_len(self.patched("<H", 0x4A, 4097)), 0)    # too many enums

    def test_rejects_pointers_outside_the_system_segment(self):
        self.assertEqual(WC.required_len(self.patched("<Q", 0x28, 0x60000100)), 0)   # struct table
        self.assertEqual(WC.required_len(self.patched("<Q", 0x38, 0x60000200)), 0)   # block table
        self.assertEqual(WC.required_len(self.patched("<Q", 0x110, 0x60000300)), 0)  # a member table

    def test_rejects_an_oversized_block(self):
        self.assertEqual(WC.required_len(self.patched("<I", 0x204, (1 << 24) + 1)), 0)

    def test_block_past_the_end_needs_more(self):
        longer = self.patched("<I", 0x204, 0x400)  # entity block 0x400..0x800 > 0x700
        self.assertEqual(WC.required_len(longer), 0x800)
        self.assertTrue(WC.needs_more(longer))
        self.assertFalse(WC.needs_more(self.blob))

    def test_graphics_block_does_not_count(self):
        gfx = self.patched("<Q", 0x208, 0x60010000)
        self.assertEqual(WC.required_len(gfx), len(gfx))

    def test_split_header_reports_the_member_tables(self):
        # 0x150 bytes hold the struct table but not the block table: the highest need is the
        # CMapData member table (0x3A0 + 7 * 16).
        self.assertEqual(WC.required_len(self.blob[:0x150]), 0x410)
        self.assertTrue(WC.needs_more(self.blob[:0x150]))


class AssembleTest(unittest.TestCase):
    def test_stream_that_does_not_end_is_none(self):
        comp = zlib.compress(TWI.make_resource(True))
        self.assertIsNone(WC.assemble(comp[:20], 0, 20))

    def test_complete_resource(self):
        blob = TWI.make_resource(True)
        m = zlib.compress(blob)
        res = WC.assemble(m, 0, len(m))
        self.assertTrue(res["is_meta"] and res["complete"])
        self.assertFalse(res["joined"])
        self.assertEqual((res["data"], res["streams"], res["pages"]), (blob, 1, 1))
        self.assertEqual(res["spans"], [(0, len(m))])
        self.assertEqual(res["end"], len(m))

    def test_non_meta_stream_is_returned_without_joining(self):
        m = aligned(zlib.compress(b"x" * 64), zlib.compress(b"y" * 64))
        res = WC.assemble(m, 0, len(m))
        self.assertFalse(res["is_meta"])
        self.assertEqual(res["streams"], 1)
        self.assertNotIn("complete", res)

    def test_zlib_continuation_is_joined(self):
        blob = TWI.make_resource(True)
        first, second = zlib.compress(blob[:0x150]), zlib.compress(blob[0x150:])
        m = aligned(first, second)
        res = WC.assemble(m, 0, len(m))
        self.assertTrue(res["complete"] and res["joined"])
        self.assertEqual(res["data"], blob)
        self.assertEqual((res["streams"], res["pages"]), (2, 2))
        self.assertEqual(res["spans"], [(0, len(first)), ((len(first) + 15) & ~15, len(second))])
        self.assertEqual(res["used"], len(first) + len(second))

    def test_second_0x7a_resource_is_not_a_continuation(self):
        blob = TWI.make_resource(True)
        m = aligned(zlib.compress(blob[:0x150]), zlib.compress(blob))
        res = WC.assemble(m, 0, len(m))
        self.assertFalse(res["complete"])
        self.assertFalse(res["joined"])
        self.assertEqual(res["streams"], 1)

    def test_zero_filled_raw_continuation_is_not_joined(self):
        blob = TWI.make_resource(True)
        m = aligned(zlib.compress(blob[:0x150]), b"\0" * 0x800)
        res = WC.assemble(m, 0, len(m))
        self.assertFalse(res["complete"])
        self.assertEqual((res["raw_system"], res["pages"]), (0, 1))

    def test_raw_continuation_past_the_archive_end_is_not_joined(self):
        blob = TWI.make_resource(True)
        m = aligned(zlib.compress(blob[:0x150]), blob[0x150:0x200])  # the rest of the page is missing
        res = WC.assemble(m, 0, len(m))
        self.assertFalse(res["complete"])
        self.assertEqual(res["raw_spans"], [])

    def test_raw_continuation_is_joined(self):
        blob = TWI.make_resource(True)
        first = zlib.compress(blob[:0x150])
        m = aligned(first, blob[0x150:])
        res = WC.assemble(m, 0, len(m))
        self.assertTrue(res["complete"] and res["joined"])
        # Two raw pages: first up to the member tables (0x410), then up to the end of the CMapData block (0x620).
        n0 = (len(first) + 15) & ~15
        self.assertEqual(res["data"], blob[:0x620])
        self.assertEqual(res["raw_spans"], [(n0, 0x410 - 0x150), (n0 + 0x410 - 0x150, 0x620 - 0x410)])
        self.assertEqual((res["raw_system"], res["pages"], res["streams"]), (0x620 - 0x150, 3, 1))


class ResourceInspectionTest(unittest.TestCase):
    def res_for(self, data):
        return {"offset": 0x1230, "used": 77, "raw_system": 0, "inflated": len(data), "streams": 1, "end": 0x1300}

    def inspect(self, data):
        row = WC.WI.parse_resource(data)
        self.assertIsNotNone(row)
        return WC.inspect_map(data, row, self.res_for(data))

    def test_gfx_extent_and_map_root(self):
        self.assertEqual(WC.gfx_extent(TWI.make_resource(True)), (0, 0))
        gfx = with_block(TWI.make_resource(True), 0x12345678, 0x80, 0x60000040)
        self.assertEqual(WC.gfx_extent(gfx), (1, 0xC0))
        self.assertTrue(WC.has_map_root(TWI.make_resource(True)))
        self.assertFalse(WC.has_map_root(TWI.make_resource(False)))
        b = bytearray(TWI.make_resource(True))
        struct.pack_into("<Q", b, 0x28, 0x60000100)
        self.assertFalse(WC.has_map_root(bytes(b)))
        self.assertEqual(WC.gfx_extent(b"\0" * 0x50), (0, 0))

    def test_clean_map_is_exact(self):
        blob = TWI.make_resource(True)
        out = self.inspect(blob)
        self.assertTrue(out["exact"])
        self.assertEqual(out["entity_blocks_counted"], 1)
        self.assertEqual((out["entity_blocks_graphics"], out["entity_blocks_unresolved"], out["duplicate_blocks"]),
                         (0, 0, 0))
        self.assertEqual(out["sha1"], hashlib.sha1(blob).hexdigest())
        self.assertEqual((out["offset"], out["size"], out["stored"], out["system_end"]), (0x1230, len(blob), 77, 0x1300))
        self.assertEqual(out["entities"], 1)

    def test_duplicate_block_is_not_exact(self):
        out = self.inspect(with_block(TWI.make_resource(True), WC.WI.ENTITY, 0x90, TWI.ptr(0x400)))
        self.assertEqual((out["duplicate_blocks"], out["entity_blocks_counted"]), (1, 2))
        self.assertFalse(out["exact"])

    def test_entity_block_in_the_graphics_segment_is_not_exact(self):
        out = self.inspect(with_block(TWI.make_resource(True), WC.WI.ENTITY, 0x90, 0x60000000))
        self.assertEqual((out["entity_blocks_graphics"], out["entity_blocks_counted"]), (1, 1))
        self.assertFalse(out["exact"])

    def test_entity_block_with_a_partial_entity_is_unresolved(self):
        out = self.inspect(with_block(TWI.make_resource(True), WC.WI.ENTITY, 0x50, TWI.ptr(0x600)))
        self.assertEqual(out["entity_blocks_unresolved"], 1)
        self.assertFalse(out["exact"])

    def test_graphics_data_block_is_measured_and_still_exact(self):
        out = self.inspect(with_block(TWI.make_resource(True), 0x12345678, 0x80, 0x60000040))
        self.assertEqual((out["gfx_blocks"], out["gfx_need"]), (1, 0xC0))
        self.assertTrue(out["exact"])


class ScanShardTest(unittest.TestCase):
    def test_counts_rejected_non_meta_and_incomplete_streams(self):
        blob = TWI.make_resource(True)
        header = bytearray(0x40)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, 0x40)
        data = aligned(bytes(header), zlib.compress(blob), zlib.compress(b"plain payload " * 8),
                       GARBAGE_ZLIB, zlib.compress(TWI.make_resource(False)), zlib.compress(blob[:0x150]))
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "s.rpf")
            with open(arc, "wb") as f:
                f.write(data)
            r = WC.scan_shard((arc, 0, 0x40, len(data), tmp, 0x40))
            self.assertEqual((r["maps"], r["entities"]), (1, 1))
            self.assertEqual(r["rejected"], 1)
            self.assertEqual(r["incomplete"], 1)
            self.assertEqual(r["streams"], 4)
            self.assertEqual((r["meta"], r["meta_other"], r["map_meta"]), (1, 1, 0))
            want = {"index": 0, "start": 0x40, "end": len(data)}
            with open(arc, "rb") as f:
                _, _, source_id = WC._source_info(f)
            doc = WC._read_shard(tmp, want, source_id)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["map_first"], 0x40)
            self.assertEqual(len(doc["intervals"]), 4)
            self.assertEqual(sorted(os.listdir(tmp)), ["s.rpf", os.path.basename(WC._shard_path(tmp, want))])
            s = WC.summarize(arc, tmp, 0x40, len(data), 1 << 30, 0x40)
            self.assertTrue(s["complete"])
            self.assertFalse(s["census_exact"])  # the incomplete resource keeps the census from being exact
            self.assertEqual(s["totals"]["incomplete"], 1)


class ClassifyGapsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arc = os.path.join(self.tmp.name, "gaps.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        with open(self.arc, "wb") as f:
            f.write(data)

    def test_every_byte_is_accounted_for(self):
        data = bytearray(b"\x33" * 0x140)
        data[0x80:0xC0] = b"\0" * 0x40     # zero padding
        data[0xC0:0xD0] = b"\x11" * 0x10   # unread
        data[0xD0:0xF0] = b"\x22" * 0x20   # a raw graphics page
        data[0xF0:0x100] = b"\x44" * 0x10  # a raw system continuation
        self.write(bytes(data))
        r = WC.classify_gaps(self.arc, [[0x40, 0x80], [0x100, 0x140]], [[0xD0, 0x20]], [[0xF0, 0x100]], 0x40, 0x140)
        self.assertEqual(r["covered"], 0x80)
        self.assertEqual(r["zero_padding"], 0x40)
        self.assertEqual(r["unread"], 0x10)
        self.assertEqual(r["unread_sample"], [[0xC0, 0x10]])
        self.assertEqual(r["raw_graphics"], 0x20)
        self.assertEqual(r["raw_system"], 0x10)
        self.assertEqual(r["gaps"], 1)
        self.assertEqual((r["span"], r["accounted"]), (0x100, 0x100))
        self.assertEqual((r["coverage"], r["accounted_ratio"]), (0.5, 1.0))
        self.assertEqual((r["gap_0drp"], r["raw_maps"], r["missed_zlib_meta"]), (0, 0, 0))

    def test_intervals_are_clipped_and_merged(self):
        self.write(b"\x33" * 0x100)
        r = WC.classify_gaps(self.arc, [[0, 0x60], [0x50, 0x70], [0x90, 0x1000], [0x10, 0x20]], [], [], 0x40, 0xA0)
        self.assertEqual(r["covered"], 0x30 + 0x10)
        self.assertEqual((r["gaps"], r["unread"]), (1, 0x20))
        self.assertEqual(r["accounted"], r["span"])

    def test_empty_range(self):
        self.write(b"\x33" * 0x40)
        r = WC.classify_gaps(self.arc, [], [], [], 0x20, 0x20)
        self.assertEqual((r["span"], r["coverage"], r["accounted_ratio"], r["gaps"]), (0, 0.0, 0.0, 0))

    def test_raw_0x7a_resource_in_a_gap(self):
        blob = bytearray(TWI.make_resource(True))
        struct.pack_into("<Q", blob, 8, TWI.ptr(0))  # a stored resource header carries a system pointer here
        data = aligned(b"\x33" * 0x40, bytes(blob))
        self.write(data)
        r = WC.classify_gaps(self.arc, [[0, 0x40]], [], [], 0x40, len(data))
        self.assertEqual((r["gap_0drp"], r["raw_0drp"], r["raw_maps"], r["raw_meta"]), (1, 1, 1, 0))
        self.assertEqual(r["raw_0drp_offsets"], [0x50])
        self.assertEqual(r["gap_0drp_other"], 0)

    def test_signature_without_a_resource_header(self):
        data = b"\x33" * 0x40 + b"\x11" * 0x10 + b"0DRPz" + b"\x11" * 0x1B + b"0DRP\x79" + b"\x11" * 0x10
        self.write(data)
        r = WC.classify_gaps(self.arc, [], [], [], 0, len(data))
        self.assertEqual((r["gap_0drp"], r["gap_0drp_other"], r["raw_0drp"]), (1, 1, 0))

    def test_unaligned_zlib_0x7a_stream_in_a_gap(self):
        comp = zlib.compress(TWI.make_resource(True))
        data = b"\x33" * 0x43 + comp + b"\0" * 0x0D
        self.write(data)
        r = WC.classify_gaps(self.arc, [[0, 0x40]], [], [], 0x40, len(data))
        self.assertEqual(r["missed_zlib_meta"], 1)
        self.assertEqual(r["missed_zlib_offsets"], [0x43])

    def test_audit_gap_across_chunk_boundaries(self):
        # 0DRP ending right on a GAP_CHUNK boundary crashed with IndexError on piece[i + 4],
        # and 0DRPz / 2-byte zlib headers straddling a chunk boundary were silently skipped.
        buf = bytearray(b"\x11" * (WC.GAP_CHUNK + 16))
        buf[WC.GAP_CHUNK - 4:WC.GAP_CHUNK + 1] = b"0DRPz"
        self.write(bytes(buf))
        r = WC.classify_gaps(self.arc, [], [], [], 0, len(buf))
        self.assertEqual((r["gap_0drp"], r["gap_0drp_other"]), (1, 1))

        # Split 0DRPz (starts at chunk - 2) and split unaligned zlib header (starts at chunk - 1).
        comp = zlib.compress(TWI.make_resource(True))
        chunk = 64
        small = bytearray(b"\x11" * (chunk * 3 + len(comp) + 16))
        small[chunk - 2:chunk + 3] = b"0DRPz"
        z_off = chunk * 2 - 1  # unaligned (odd) and straddles the 2nd chunk boundary
        small[z_off:z_off + len(comp)] = comp
        self.write(bytes(small))
        with mock.patch.object(WC, "GAP_CHUNK", chunk):
            r2 = WC.classify_gaps(self.arc, [], [], [], 0, len(small))
        self.assertEqual(r2["gap_0drp"], 1)
        self.assertEqual(r2["missed_zlib_meta"], 1)
        self.assertEqual(r2["missed_zlib_offsets"], [z_off])

    def test_gap_before_raw_segment_is_audited(self):
        # A raw 0x7a map and an unaligned zlib 0x7a stream sitting before raw_graphics/raw_system
        # segments in the same gap must still be audited so census_exact cannot falsely report True.
        blob = bytearray(TWI.make_resource(True))
        struct.pack_into("<Q", blob, 8, TWI.ptr(0))
        comp = zlib.compress(TWI.make_resource(True))
        # Layout: [0..0x40) covered, [0x40..0x40+len(blob)) raw map, raw_graphics at raw_off,
        # then unaligned zlib at z_off before raw_system at sys_off.
        raw_off = (0x40 + len(blob) + 0x1F) & ~0x0F
        z_off = raw_off + 0x43
        sys_off = (z_off + len(comp) + 0x1F) & ~0x0F
        total = sys_off + 0x40
        data = bytearray(b"\x33" * total)
        data[0x40:0x40 + len(blob)] = blob
        data[z_off:z_off + len(comp)] = comp
        self.write(bytes(data))
        r = WC.classify_gaps(self.arc, [[0, 0x40]], [[raw_off, 0x20]], [[sys_off, total]], 0x40, total)
        self.assertEqual((r["gap_0drp"], r["raw_0drp"], r["raw_maps"]), (1, 1, 1))
        self.assertEqual(r["missed_zlib_meta"], 1)
        self.assertEqual(r["missed_zlib_offsets"], [z_off])
        self.assertEqual(r["accounted"], r["span"])

    def test_straddling_raw_segment_is_not_skipped(self):
        # A raw_graphics segment that straddles scan_start (or overlaps an earlier raw segment in the
        # same gap) must account for [pos, s1), whereas a raw_pages candidate whose start was covered
        # by a compressed stream [0x200..0x240) must not spill its uncompressed tail into [0x240..0x300).
        self.write(b"\x22" * 0x300)
        r = WC.classify_gaps(
            self.arc,
            [[0x200, 0x240]],
            [[0x080, 0x100], [0x0A0, 0x040], [0x200, 0x0C0]],
            [],
            0x100,
            0x300,
        )
        self.assertEqual(r["covered"], 0x40)
        self.assertEqual(r["raw_graphics"], 0x80)
        self.assertEqual(r["unread"], 0x140)
        self.assertEqual(r["accounted"], r["span"])

    def test_duplicates_handles_rows_without_map_name(self):
        rows = [
            {"sha1": "aa" * 20, "entities": 2},
            {"sha1": "aa" * 20, "entities": 2},
        ]
        dup = WC._duplicates(rows)
        self.assertEqual(dup["content_groups"], 1)
        self.assertEqual(dup["content_extra_maps"], 1)
        self.assertEqual(dup["content_extra_entities"], 2)
        self.assertEqual(dup["content_group_sample"], [[None, 2]])
        self.assertEqual(dup["name_groups"], 0)


class ShardFileTest(unittest.TestCase):
    def test_read_and_load_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            want = WC._expected_shards(0x40, 0x1000, 0x800)
            self.assertEqual(len(want), 2)
            source_id = "one archive"
            good = {**want[0], "format": WC.FORMAT, "complete": True, "source_id": source_id,
                    "stats": {}, "rows": [], "intervals": [], "raw_pages": [], "raw_system": []}
            path0, path1 = WC._shard_path(tmp, want[0]), WC._shard_path(tmp, want[1])
            self.assertEqual(os.path.basename(path0), "shard_000_64_2112.json")
            self.assertIsNone(WC._read_shard(tmp, want[0], source_id))
            for doc, valid in ((good, True), ({**good, "format": WC.FORMAT - 1}, False),
                               ({**good, "start": 0}, False), ({**good, "end": 0}, False),
                               ({**good, "index": 1}, False), ({**good, "complete": False}, False),
                               ({k: v for k, v in good.items() if k != "complete"}, False),
                               ({k: v for k, v in good.items() if k != "source_id"}, False),
                               ({**good, "index": False}, False), ({**good, "rows": None}, False), ([], False)):
                with self.subTest(doc=doc):
                    with open(path0, "w", encoding="utf-8") as f:
                        json.dump(doc, f)
                    self.assertEqual(WC._read_shard(tmp, want[0], source_id), doc if valid else None)
            with open(path0, "w", encoding="utf-8") as f:
                f.write("{")
            self.assertIsNone(WC._read_shard(tmp, want[0], source_id))
            with open(path1, "w", encoding="utf-8") as f:
                json.dump({**good, **want[1]}, f)
            self.assertEqual(list(WC._load_shards(tmp, want, source_id)), [1])


class AggregateTest(unittest.TestCase):
    def test_duplicates(self):
        rows = [{"sha1": "a", "entities": 3, "map_name": 1}, {"sha1": "a", "entities": 3, "map_name": 1},
                {"sha1": "b", "entities": 2, "map_name": 1}, {"sha1": "c", "entities": 5, "map_name": None},
                {"sha1": "d", "entities": 7}]
        self.assertEqual(WC._duplicates(rows), {
            "content_groups": 1, "content_extra_maps": 1, "content_extra_entities": 3,
            "content_group_sample": [[1, 2]], "name_groups": 1, "name_extra_maps": 2})
        self.assertEqual(WC._duplicates([])["content_groups"], 0)

    def test_bands(self):
        far = 200 + WC.BAND_GAP + 1000
        rows = [{"offset": far, "stored": 10, "entities": 4}, {"offset": 200, "stored": 50, "entities": 2},
                {"offset": 0, "stored": 100, "entities": 1}]
        self.assertEqual(WC._bands(rows), [
            {"first": 0, "last_end": 250, "maps": 2, "entities": 3},
            {"first": far, "last_end": far + 10, "maps": 1, "entities": 4}])
        self.assertEqual(WC._bands([]), [])


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.arc = os.path.join(self.dir, "sample.rpf")
        make_archive(self.arc)
        self.out = os.path.join(self.dir, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = WC.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def summary(self):
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_not_an_rpf9_archive(self):
        bad = os.path.join(self.dir, "bad.rpf")
        for content in (b"NOPE" + b"\0" * 60, b"9F"):
            with self.subTest(content=content[:4]):
                with open(bad, "wb") as f:
                    f.write(content)
                code, _, err = self.run_main(bad, "--out", self.out)
                self.assertEqual(code, 2)
                self.assertIn("not an RPF9 archive", err)
        self.assertFalse(os.path.exists(self.out))

    def test_magic_without_the_data_start_is_not_an_rpf9_archive(self):
        # The data start is the u32 at +4: a file with the magic but under 8 bytes used to end in a
        # struct.error traceback with exit 1, the code for an incomplete census.
        bad = os.path.join(self.dir, "short.rpf")
        for content in (b"9FPR", b"9FPR\x40", b"9FPR\x40\x00\x00"):
            with self.subTest(size=len(content)):
                with open(bad, "wb") as f:
                    f.write(content)
                code, out, err = self.run_main(bad, "--out", self.out)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertEqual(err.strip(), "not an RPF9 archive")
        self.assertFalse(os.path.exists(self.out))

    def test_missing_or_folder_archive_is_one_line(self):
        for archive in (os.path.join(self.dir, "no_such.rpf"), self.dir):
            with self.subTest(archive=os.path.basename(archive)):
                code, out, err = self.run_main(archive, "--out", self.out)
                self.assertEqual(code, 2)
                self.assertTrue(err.startswith(archive + ": "), err)
                self.assertEqual(len(err.strip().splitlines()), 1)
                self.assertEqual(out, "")
        self.assertFalse(os.path.exists(self.out))

    def test_same_report_path_identity(self):
        a = os.path.join(self.dir, "test.rpf")
        self.assertTrue(WC.same_report_path(a, a))
        b = os.path.join(self.dir, "other.rpf")
        self.assertFalse(WC.same_report_path(a, b))

    def test_out_matching_archive_or_reports_is_rejected(self):
        code, _, err = self.run_main(self.arc, "--out", self.arc)
        self.assertEqual(code, 2)
        self.assertIn("error: --out must not overwrite the input archive", err)

        # Archive named summary.json would be overwritten if output directory is its parent
        arc_summary = os.path.join(self.dir, "summary.json")
        shutil.copyfile(self.arc, arc_summary)
        code, _, err = self.run_main(arc_summary, "--out", self.dir)
        self.assertEqual(code, 2)
        self.assertIn("error: --out reports would overwrite input archive", err)

    def test_out_as_existing_file_is_rejected(self):
        file_out = os.path.join(self.dir, "existing_file.txt")
        with open(file_out, "w") as f:
            f.write("data")
        code, _, err = self.run_main(self.arc, "--out", file_out)
        self.assertEqual(code, 2)
        self.assertIn("is a file, not a directory", err)

    def test_bad_scan_range(self):
        size = os.path.getsize(self.arc)
        for args in (("--start", "16"), ("--start", str(size)), ("--start", str(size + 100)), ("--limit", "-1"),
                     ("--limit", "0")):     # 0 bytes is an empty range, not "no limit" (it scanned everything)
            with self.subTest(args=args):
                code, _, err = self.run_main(self.arc, "--out", self.out, "--jobs", "1", *args)
                self.assertEqual(code, 2)
                self.assertIn("bad scan range", err)
        self.assertFalse(os.path.exists(self.out))

    def test_merge_only_without_shards_is_incomplete(self):
        size = os.path.getsize(self.arc)
        code, _, _ = self.run_main(self.arc, "--out", self.out, "--merge-only")
        self.assertEqual(code, 1)
        s = self.summary()
        self.assertFalse(s["complete"])
        self.assertFalse(s["census_exact"])
        self.assertEqual((s["shards_complete"], s["shards_expected"], s["maps"]), (0, 1, 0))
        self.assertEqual(s["missing_ranges"], [[0x40, size]])

    def test_resume_skips_finished_shards_and_rescans_bad_ones(self):
        code, out, _ = self.run_main(self.arc, "--out", self.out, "--jobs", "1")
        self.assertEqual(code, 0)
        self.assertIn("1 shards, 1 to scan", out)
        with open(os.path.join(self.out, "maps.json"), encoding="utf-8") as f:
            maps = json.load(f)
        self.assertEqual(len(maps["maps"]), 4)
        self.assertTrue(maps["summary"]["census_exact"])

        code, out, _ = self.run_main(self.arc, "--out", self.out, "--jobs", "1", "--resume")
        self.assertEqual(code, 0)
        self.assertIn("1 shards, 0 to scan", out)
        self.assertEqual(self.summary()["maps"], 4)

        size = os.path.getsize(self.arc)
        path = WC._shard_path(self.out, WC._expected_shards(0x40, size, 2 << 30)[0])
        with open(path, "w", encoding="utf-8") as f:
            f.write("{")
        code, out, _ = self.run_main(self.arc, "--out", self.out, "--jobs", "1", "--resume")
        self.assertEqual(code, 0)
        self.assertIn("1 shards, 1 to scan", out)
        self.assertEqual(self.summary()["maps"], 4)

    def test_merge_only_after_a_scan(self):
        self.assertEqual(self.run_main(self.arc, "--out", self.out, "--jobs", "1", "--shard-size", "400")[0], 0)
        code, out, _ = self.run_main(self.arc, "--out", self.out, "--merge-only", "--shard-size", "400")
        self.assertEqual(code, 0)
        self.assertNotIn("to scan", out)
        s = self.summary()
        self.assertEqual((s["maps"], s["entities"]), (4, 4))
        self.assertTrue(s["census_exact"])
        self.assertEqual(s["totals"]["meta_other"], 1)

    def check_report_failure_and_retry(self, out, failure):
        paths = [Path(out) / name for name in ("maps.json", "summary.json")]
        before = [p.read_bytes() if p.exists() else None for p in paths]
        args = (self.arc, "--out", out, "--jobs", "1", "--resume")
        with failure:
            code, _, err = self.run_main(*args)
        self.assertEqual(code, 1, err)
        for path, old_bytes in zip(paths, before):
            self.assertEqual(path.read_bytes() if path.exists() else None, old_bytes, path.name)

        code, stdout, err = self.run_main(*args)
        self.assertEqual(code, 0, err)
        self.assertIn("1 shards, 0 to scan", stdout)
        maps, summary = (json.loads(path.read_bytes()) for path in paths)
        self.assertEqual(len(maps["maps"]), 4)
        self.assertEqual(maps["summary"], {k: summary[k] for k in maps["summary"]})

    def test_failed_report_replace_restores_previous_files(self):
        names = ("maps.json", "summary.json")
        replace = os.replace
        for failed in names:
            for previous in (names, names[:1], names[1:], ()):
                with self.subTest(failed=failed, previous=previous), tempfile.TemporaryDirectory() as out:
                    for name in previous:
                        (Path(out) / name).write_text(json.dumps({"previous": name}), encoding="utf-8")
                    injected = False

                    def fail_once(src, dst):
                        nonlocal injected
                        if not injected and Path(dst) == Path(out) / failed:
                            injected = True
                            raise PermissionError("report replacement failed")
                        return replace(src, dst)

                    self.check_report_failure_and_retry(out, mock.patch.object(os, "replace", fail_once))
                    self.assertTrue(injected)

    def test_failed_report_restore_keeps_recovery_copy(self):
        out = Path(self.out)
        out.mkdir()
        maps, summary = out / "maps.json", out / "summary.json"
        old_summary = b'{"previous": "summary"}'
        old_maps = b'{"previous": "maps"}'
        summary.write_bytes(old_summary)
        maps.write_bytes(old_maps)
        replace = os.replace

        def fail_publish_and_restore(src, dst):
            if Path(dst) == maps:
                raise PermissionError("publish denied")
            if Path(dst) == summary and Path(src).read_bytes() == old_summary:
                raise PermissionError("restore denied")
            return replace(src, dst)

        with mock.patch.object(os, "replace", fail_publish_and_restore):
            code, _, err = self.run_main(self.arc, "--out", self.out, "--jobs", "1")
        self.assertEqual(code, 1)
        self.assertIn("restore denied", err)
        self.assertEqual(maps.read_bytes(), old_maps)
        backups = [p for p in out.rglob("summary.json") if p != summary and p.read_bytes() == old_summary]
        self.assertEqual(len(backups), 1)
        self.assertIn(str(backups[0].parent), err)

    @unittest.skipUnless(os.name == "nt", "Windows readers can deny replacement")
    def test_report_held_open_preserves_both_files(self):
        for held in ("maps.json", "summary.json"):
            with self.subTest(held=held), tempfile.TemporaryDirectory() as out:
                self.assertEqual(self.run_main(self.arc, "--out", out, "--jobs", "1", "--limit", "64")[0], 0)
                self.check_report_failure_and_retry(out, open(Path(out) / held, "rb"))

    def test_limit_narrows_the_scan(self):
        code, _, _ = self.run_main(self.arc, "--out", self.out, "--jobs", "1", "--limit", "64")
        self.assertEqual(code, 0)
        s = self.summary()
        self.assertEqual((s["scan_start"], s["scan_end"]), (0x40, 0x80))


class CensusSourceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.arc = os.path.join(tmp.name, "sample.rpf")
        self.out = os.path.join(tmp.name, "out")
        make_archive(self.arc)

    def scan(self, archive, *flags, code=0, to_scan=None):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = WC.main([archive, "--out", self.out, "--shard-size", "850", "--jobs", "1", *flags])
        self.assertEqual(result, code, stdout.getvalue())
        if to_scan is not None:
            self.assertIn(f"4 shards, {to_scan} to scan", stdout.getvalue())
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(self.out, "maps.json"), encoding="utf-8") as f:
            maps = json.load(f)
        self.assertEqual(maps["summary"], {k: summary[k] for k in maps["summary"]})
        self.assertEqual(len(maps["maps"]), summary["maps"])
        self.assertEqual(sum(r["entities"] for r in maps["maps"]), summary["entities"])
        return summary, maps["maps"]

    def write_empty_archive(self, path):
        size = os.path.getsize(self.arc)
        with open(self.arc, "rb") as f:
            header = f.read(0x40)
        with open(path, "wb") as f:
            f.write(header)
            f.truncate(size)

    def test_other_archive_with_same_size_and_header(self):
        original, _ = self.scan(self.arc)
        self.assertEqual(original["maps"], 4)
        other = os.path.join(os.path.dirname(self.arc), "other.rpf")
        self.write_empty_archive(other)

        missing, _ = self.scan(other, "--merge-only", code=1)
        self.assertEqual((missing["shards_complete"], missing["maps"]), (0, 0))
        self.assertFalse(missing["census_exact"])
        rebuilt, _ = self.scan(other, "--resume", to_scan=4)
        self.assertEqual(rebuilt["maps"], 0)
        self.assertTrue(rebuilt["census_exact"])
        self.scan(other, "--resume", to_scan=0)

    def test_files_without_file_ids_are_distinguished_by_name(self):
        other = os.path.join(os.path.dirname(self.arc), "copy.rpf")
        make_archive(other)
        stamp = mock.Mock(st_size=os.path.getsize(other), st_dev=0, st_ino=0, st_mtime_ns=1, st_ctime_ns=1)
        with open(self.arc, "rb") as first, open(other, "rb") as second:
            with mock.patch.object(os, "fstat", return_value=stamp):
                self.assertNotEqual(WC._source_info(first)[2], WC._source_info(second)[2])
                self.assertEqual(WC._source_info(first)[2], WC._source_info(first)[2])

    def test_edit_at_same_path_with_restored_mtime(self):
        self.scan(self.arc)
        before = os.stat(self.arc)
        self.write_empty_archive(self.arc)
        os.utime(self.arc, ns=(before.st_atime_ns, before.st_mtime_ns))
        summary, _ = self.scan(self.arc, "--merge-only", code=1)
        self.assertEqual((summary["shards_complete"], summary["maps"]), (0, 0))
        self.assertFalse(summary["census_exact"])

    def test_source_replaced_after_gap_audit(self):
        other = os.path.join(os.path.dirname(self.arc), "replacement.rpf")
        self.write_empty_archive(other)
        audit = WC.classify_gaps

        def swap_after_audit(*args):
            coverage = audit(*args)
            os.replace(other, self.arc)
            return coverage

        with mock.patch.object(WC, "classify_gaps", side_effect=swap_after_audit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = WC.main([self.arc, "--out", self.out, "--jobs", "1"])
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.out, "summary.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "maps.json")))

    def test_mixed_and_legacy_shards_preserve_valid_coverage(self):
        original, rows = self.scan(self.arc)
        foreign, legacy = original["shards"][:2]
        for shard in (foreign, legacy):
            path = WC._shard_path(self.out, shard)
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            if shard is foreign:
                doc["source_id"] = "different archive"
            else:
                doc.pop("source_id", None)
                doc["format"] = WC.FORMAT - 1
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)

        partial, present_rows = self.scan(self.arc, "--merge-only", code=1)
        self.assertFalse(partial["census_exact"])
        self.assertEqual(partial["shards_complete"], 2)
        self.assertEqual(partial["missing_ranges"], [[s["start"], s["end"]] for s in (foreign, legacy)])
        self.assertTrue(present_rows)
        self.assertEqual(present_rows, [r for r in rows if r["offset"] >= legacy["end"]])
        rebuilt, new_rows = self.scan(self.arc, "--resume", to_scan=2)
        self.assertTrue(rebuilt["census_exact"])
        self.assertEqual(new_rows, rows)


def setUpModule():
    # The tools' main() prints its summary JSON; keep it out of the test log (tests that check it redirect again).
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    unittest.addModuleCleanup(quiet.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
