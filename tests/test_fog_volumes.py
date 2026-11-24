"""Synthetic unit tests for tools/fog_volumes.py (ASSET-308). No game files."""
import concurrent.futures
import json
import os
import struct
import sys
import tempfile
import unittest
import unittest.mock
import zlib
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(os.path.dirname(TESTS_DIR), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import fog_volumes as fv  # noqa: E402
import pool_workers  # noqa: E402

SYS = 0x50000000


def build_synthetic_fog_0x7a(volumes: list[dict]) -> bytes:
    """Build a minimal valid 0x7a resource with CScatteringTransmittance__FogVolumeListData and FogVolume."""
    vsz = 512
    # Layout:
    #   0x000..0x050: 0x7a header
    #   0x050..0x090: 2 struct descriptors (32 bytes each)
    #   0x090..0x0b0: 2 block descriptors (16 bytes each)
    #   0x0b0..0x1a0: 15 member descriptors for FogVolume (16 bytes each = 240 bytes)
    #   0x1a0..0x1b0: 1 FogVolumeListData instance (16 bytes)
    #   0x1b0..     : len(volumes) * 512 FogVolume instances
    st_off = 0x50
    it_off = 0x90
    mem_off = 0xB0
    list_off = 0x1A0
    vol_off = 0x1B0
    total = vol_off + len(volumes) * vsz
    buf = bytearray(total)

    # Header
    buf[0x10:0x14] = b"0DRP"
    buf[0x14] = 0x7A
    struct.pack_into("<I", buf, 0x20, 1)                # root_objects
    struct.pack_into("<Q", buf, 0x28, SYS | st_off)     # struct table ptr
    struct.pack_into("<Q", buf, 0x38, SYS | it_off)     # block table ptr
    struct.pack_into("<H", buf, 0x48, 2)                # nst
    struct.pack_into("<H", buf, 0x4A, 0)                # nen
    struct.pack_into("<H", buf, 0x4C, 2)                # nbl

    # Member specs matching the measured 0x7a FogVolume layout
    member_specs = [
        ("IsBox", 0x000, 0x01),
        ("Position", 0x010, 0x33),
        ("PositionOffset", 0x020, 0x33),
        ("Scale", 0x040, 0x33),
        ("Rotation", 0x050, 0x33),
        ("Shear", 0x060, 0x33),
        ("RotationMtx", 0x0B0, 0x35),
        ("TransformMtx", 0x0E0, 0x35),
        ("BlendMode", 0x1B8, 0x15),
        ("IntExtMode", 0x1BC, 0x15),
        ("Type", 0x1C0, 0x15),
        ("SortKey", 0x1C8, 0x15),
        ("ProfileIndex", 0x1CC, 0x15),
        ("SecondaryProfileIndex", 0x1D4, 0x15),
        ("ProfileInterpolation", 0x1D8, 0x21),
    ]
    assert len(member_specs) == 15

    # Struct 0: FogVolumeListData
    struct.pack_into("<I", buf, st_off + 0x00, fv.STRUCT_LIST)
    struct.pack_into("<Q", buf, st_off + 0x10, 0)
    struct.pack_into("<I", buf, st_off + 0x18, 16)
    struct.pack_into("<H", buf, st_off + 0x1E, 0)

    # Struct 1: FogVolume
    struct.pack_into("<I", buf, st_off + 32 + 0x00, fv.STRUCT_VOLUME)
    struct.pack_into("<Q", buf, st_off + 32 + 0x10, SYS | mem_off)
    struct.pack_into("<I", buf, st_off + 32 + 0x18, vsz)
    struct.pack_into("<H", buf, st_off + 32 + 0x1E, len(member_specs))

    # Block 0: FogVolumeListData; Block 1: FogVolume array
    struct.pack_into("<IIQ", buf, it_off, fv.STRUCT_LIST, 16, SYS | list_off)
    struct.pack_into("<IIQ", buf, it_off + 16, fv.STRUCT_VOLUME, len(volumes) * vsz, SYS | vol_off)

    for i, (name, foff, fty) in enumerate(member_specs):
        o = mem_off + 16 * i
        struct.pack_into("<IIBBHI", buf, o, fv.joaat_cs(name), foff, fty, 0, 0, 0)

    ident12 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for k, v in enumerate(volumes):
        base = vol_off + k * vsz
        buf[base + 0x000] = v.get("IsBox", 1)
        struct.pack_into("<3f", buf, base + 0x010, *v.get("Position", (0.0, 0.0, 0.0)))
        struct.pack_into("<3f", buf, base + 0x020, *v.get("PositionOffset", (0.0, 0.0, 0.0)))
        struct.pack_into("<3f", buf, base + 0x040, *v.get("Scale", (1.0, 2.0, 3.0)))
        struct.pack_into("<3f", buf, base + 0x050, *v.get("Rotation", (0.0, 0.0, 0.0)))
        struct.pack_into("<3f", buf, base + 0x060, *v.get("Shear", (0.0, 0.0, 0.0)))
        struct.pack_into("<12f", buf, base + 0x0B0, *v.get("RotationMtx", ident12))
        struct.pack_into("<12f", buf, base + 0x0E0, *v.get("TransformMtx", ident12))
        struct.pack_into("<I", buf, base + 0x1B8, v.get("BlendMode", 1))
        struct.pack_into("<I", buf, base + 0x1BC, v.get("IntExtMode", 1))
        struct.pack_into("<I", buf, base + 0x1C0, v.get("Type", 0))
        struct.pack_into("<I", buf, base + 0x1C8, v.get("SortKey", 0))
        struct.pack_into("<I", buf, base + 0x1CC, v.get("ProfileIndex", 12))
        struct.pack_into("<I", buf, base + 0x1D4, v.get("SecondaryProfileIndex", 0))
        struct.pack_into("<f", buf, base + 0x1D8, v.get("ProfileInterpolation", 0.0))

    return bytes(buf)


class FogVolumesTests(unittest.TestCase):
    def test_parse_and_summarize_carved_file(self):
        raw = build_synthetic_fog_0x7a([
            {"IsBox": 1, "Position": (0.0, 0.0, 0.0), "PositionOffset": (1.5, -2.0, 0.5),
             "Scale": (2.0, 4.0, 1.5), "Shear": (0.0, 0.25, 0.0), "IntExtMode": 1, "ProfileIndex": 12},
            {"IsBox": 0, "Position": (0.0, 0.0, 0.0), "PositionOffset": (0.0, 0.0, 0.0),
             "Scale": (0.5, 1.0, 0.5), "Rotation": (0.0, 0.0, 1.57), "IntExtMode": 0, "ProfileIndex": 10},
        ])
        vols = fv.parse_fog_volumes(raw)
        self.assertEqual(len(vols), 2)
        self.assertEqual(vols[0]["IsBox"], 1)
        self.assertEqual(vols[1]["IsBox"], 0)

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "sample.bin")
            with open(fp, "wb") as f:
                f.write(raw)
            doc = fv.summarize(fp, jobs=1)
            self.assertEqual(doc["resources"], 1)
            self.assertEqual(doc["volumes"], 2)
            self.assertEqual(doc["is_box"], {"0": 1, "1": 1})
            self.assertEqual(doc["int_ext_mode"], {"0": 1, "1": 1})
            self.assertEqual(doc["profile_index"], {"10": 1, "12": 1})
            self.assertEqual(doc["position_at_origin"], 2)
            self.assertEqual(doc["position_offset_nonzero"], 1)
            self.assertEqual(doc["rotation_nonzero"], 1)
            self.assertEqual(doc["shear_nonzero"], 1)
            self.assertEqual(doc["rotation_mtx_identity"], 2)
            self.assertEqual(doc["transform_mtx_identity"], 2)

    def test_summarize_synthetic_rpf9(self):
        raw = build_synthetic_fog_0x7a([
            {"IsBox": 1, "PositionOffset": (-1.0, 3.0, 2.0), "ProfileIndex": 12},
        ])
        comp = zlib.compress(raw)
        with tempfile.TemporaryDirectory() as td:
            rpf = os.path.join(td, "test.rpf")
            out_json = os.path.join(td, "sub", "out.json")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + b"\x00" * 12 + comp + b"\x00" * 64)
            rc = fv.main([rpf, "--out", out_json, "--jobs", "1"])
            self.assertEqual(rc, 0)
            with open(out_json, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["resources"], 1)
            self.assertEqual(doc["volumes"], 1)
            self.assertEqual(doc["distinct_serialized_volumes"], 1)
            self.assertEqual(doc["offset_range"], ["0x10", "0x10"])
            self.assertEqual(doc["position_offset_nonzero"], 1)

    def test_large_resource_exceeding_256kib_and_dedup(self):
        # 520 * 512 = 266,240 bytes (> 256 KiB): exercises uncapped 0x7a inflation and distinct record counting.
        base_vols = [
            {"IsBox": 1, "PositionOffset": (float(i), -1.0, 0.5), "ProfileIndex": 12}
            for i in range(260)
        ]
        raw = build_synthetic_fog_0x7a(base_vols + base_vols)
        self.assertGreater(len(raw), 262144)
        comp = zlib.compress(raw)
        with tempfile.TemporaryDirectory() as td:
            rpf = os.path.join(td, "large.rpf")
            with open(rpf, "wb") as f:
                f.write(b"9FPR" + b"\x00" * 12 + comp + b"\x00" * 64)
            doc = fv.summarize(rpf, jobs=1)
            self.assertEqual(doc["resources"], 1)
            self.assertEqual(doc["volumes"], 520)
            self.assertEqual(doc["distinct_serialized_volumes"], 260)
            self.assertEqual(doc["resources_by_volume_count"], {"520": 1})
            self.assertEqual(doc["scanned_0x7a_complete"], 1)
            self.assertEqual(doc["scanned_0x7a_incomplete"], 0)

    def test_corrupt_and_truncated_compressed_stream(self):
        # Build a valid 0x7a FogVolume payload large enough that its compressed body has
        # room for a corrupt byte past the 24-byte header probe, plus a truncated copy.
        raw_valid = build_synthetic_fog_0x7a([
            {"IsBox": 1, "PositionOffset": (float(i), 2.0, -0.5), "ProfileIndex": 12}
            for i in range(16)
        ])
        comp_valid = zlib.compress(raw_valid)
        self.assertGreater(len(comp_valid), 64)

        corrupt = bytearray(comp_valid)
        corrupt[-10] ^= 0xFF
        truncated = comp_valid[:-12]

        with tempfile.TemporaryDirectory() as td:
            # 1. Corrupt stream followed by a valid stream: must not raise zlib.error and must still find the valid one.
            rpf_corrupt = os.path.join(td, "corrupt_then_valid.rpf")
            with open(rpf_corrupt, "wb") as f:
                f.write(b"9FPR" + b"\x00" * 12 + bytes(corrupt) + b"\x00" * 32 + comp_valid + b"\x00" * 32)
            doc = fv.summarize(rpf_corrupt, jobs=1)
            self.assertEqual(doc["resources"], 1)
            self.assertEqual(doc["volumes"], 16)
            self.assertGreaterEqual(doc["scanned_0x7a_incomplete"], 1)
            self.assertEqual(doc["scanned_0x7a_complete"], 1)

            # 2. Truncated stream at EOF (missing last 12 bytes): must return 0 resources without crashing.
            rpf_trunc = os.path.join(td, "truncated.rpf")
            with open(rpf_trunc, "wb") as f:
                f.write(b"9FPR" + b"\x00" * 12 + truncated)
            doc_t = fv.summarize(rpf_trunc, jobs=1)
            self.assertEqual(doc_t["resources"], 0)
            self.assertEqual(doc_t["volumes"], 0)
            self.assertEqual(doc_t["scanned_0x7a_incomplete"], 1)
            self.assertEqual(doc_t["scanned_0x7a_complete"], 0)

    def test_cli_validation_guards(self):
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "in.bin")
            with open(fp, "wb") as f:
                f.write(b"\x00" * 64)
            # Refuses overwriting input file
            with self.assertRaises(SystemExit) as cm:
                fv.main([fp, "--out", fp])
            self.assertEqual(cm.exception.code, 2)
            # Refuses directory as --out
            with self.assertRaises(SystemExit) as cm:
                fv.main([fp, "--out", td])
            self.assertEqual(cm.exception.code, 2)
            # Refuses missing input
            with self.assertRaises(SystemExit) as cm:
                fv.main([os.path.join(td, "missing.rpf")])
            self.assertEqual(cm.exception.code, 2)

    def test_jobs_below_one_is_a_usage_error(self):
        # A --jobs of 0 used to run with max(1, 0) = one worker and exit 0, so a caller that meant to ask for
        # an impossible worker count got a silent sequential run instead of an error.
        with tempfile.TemporaryDirectory() as td:
            for bad in ("0", "-3"):
                with self.assertRaises(SystemExit) as cm:
                    fv.main([td, "--jobs", bad])
                self.assertEqual(cm.exception.code, 2)
    @unittest.skipIf(pool_workers.MAX_PROCESS_WORKERS is None, "this platform has no process-pool worker cap")
    def test_jobs_above_the_pool_cap_ask_for_the_cap(self):
        # ProcessPoolExecutor takes at most 61 workers on Windows (WaitForMultipleObjects' handle limit), so a
        # --jobs of 1000 used to reach it unchanged and the run died with ValueError before the first chunk.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "big.rpf")
            with open(path, "wb") as f:
                f.truncate(16 * 1024 * 1024)  # four chunks of the 4 MiB floor, so the pool path is taken
            asked = []

            class RecordingPool:
                def __init__(self, max_workers=None, **_kwargs):
                    asked.append(max_workers)

                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    return False

                def map(self, fn, items):
                    return [fn(item) for item in items]

            with unittest.mock.patch.object(concurrent.futures, "ProcessPoolExecutor", RecordingPool):
                result = fv._scan_rpf_offsets(path, jobs=1000)
        self.assertEqual(asked, [pool_workers.MAX_PROCESS_WORKERS])
        self.assertEqual(result, ([], 0, 0))

    def test_one_chunk_input_is_scanned_without_a_pool(self):
        # A file that fits in one chunk is scanned in the calling process, whatever --jobs says.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "small.rpf")
            with open(path, "wb") as f:
                f.write(b"\x00" * 1024)

            def no_pool(*_args, **_kwargs):
                raise AssertionError("a one-chunk input must not start a pool")

            with unittest.mock.patch.object(concurrent.futures, "ProcessPoolExecutor", no_pool):
                self.assertEqual(fv._scan_rpf_offsets(path, jobs=1000), ([], 0, 0))


    def test_pool_is_capped_at_the_platform_limit(self):
        # A --jobs above the worker-process limit used to reach ProcessPoolExecutor unchanged; on Windows the
        # constructor then raises ValueError(max_workers must be <= 61). The other pool tools lower it with
        # pool_workers.process_workers. 61 is forced here so the check does not depend on the host OS.
        seen = []

        class RecordingPool:
            def __init__(self, max_workers=None):
                seen.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, iterable):
                return [([], 0, 0) for _ in iterable]

        with tempfile.TemporaryDirectory() as td:
            rpf = os.path.join(td, "big.rpf")
            with open(rpf, "wb") as f:
                f.write(b"\x00" * (9 * 1024 * 1024))     # three 4 MiB chunks, so the pool branch is taken
            with mock.patch.object(pool_workers, "MAX_PROCESS_WORKERS", 61), \
                    mock.patch.object(fv.concurrent.futures, "ProcessPoolExecutor", RecordingPool):
                offs, complete, incomplete = fv._scan_rpf_offsets(rpf, jobs=200)
        self.assertEqual((offs, complete, incomplete), ([], 0, 0))
        self.assertEqual(seen, [61])


if __name__ == "__main__":
    unittest.main()
