"""Unit tests for tools/tex_mode_probe.py covering texture tiling mode detection,
descriptor format classification, mipmap chain alignment, and candidate scoring.

    python -I tests/test_tex_mode_probe.py
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    np = None
    HAVE_NUMPY = False

try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy required for tex_mode_probe tests")

if HAVE_NUMPY:
    import tex_mode_probe
else:
    tex_mode_probe = None

    def setUpModule():
        raise unittest.SkipTest("numpy required for tex_mode_probe tests")

# Synthetic 4K and 64K equations providing exact bijections within tiles:
# 4K, bpe=8: elems=512, bw=32 (5 bits x), bh=16 (4 bits y), 3 bits intra-element
SYNTHETIC_EQ_4K_BC1 = [
    (0, 0), (0, 0), (0, 0),
    (1, 0), (0, 1),
    (2, 0), (0, 2),
    (4, 0), (0, 4),
    (8, 0), (0, 8),
    (16, 0),
]

# 64K, bpe=8: elems=8192, bw=128 (7 bits x), bh=64 (6 bits y), 3 bits intra-element
SYNTHETIC_EQ_64K_BC1 = [
    (0, 0), (0, 0), (0, 0),
    (1, 0), (0, 1),
    (2, 0), (0, 2),
    (4, 0), (0, 4),
    (8, 0), (0, 8),
    (16, 0), (0, 16),
    (32, 0), (0, 32),
    (64, 0),
]

# 4K, bpe=16: elems=256, bw=16 (4 bits x), bh=16 (4 bits y), 4 bits intra-element
SYNTHETIC_EQ_4K_BC2_3 = [
    (0, 0), (0, 0), (0, 0), (0, 0),
    (1, 0), (0, 1),
    (2, 0), (0, 2),
    (4, 0), (0, 4),
    (8, 0), (0, 8),
]

# 64K, bpe=16: elems=4096, bw=64 (6 bits x), bh=64 (6 bits y), 4 bits intra-element
SYNTHETIC_EQ_64K_BC2_3 = [
    (0, 0), (0, 0), (0, 0), (0, 0),
    (1, 0), (0, 1),
    (2, 0), (0, 2),
    (4, 0), (0, 4),
    (8, 0), (0, 8),
    (16, 0), (0, 16),
    (32, 0), (0, 32),
]


def make_synthetic_bc1_blocks(wb: int = 64, hb: int = 64) -> np.ndarray:
    """Create a 2D smooth pattern of BC1 blocks with varying color endpoints."""
    bpe = 8
    y_coord, x_coord = np.mgrid[0:hb, 0:wb]
    img_r = (np.sin(x_coord / 5.0) * 0.5 + 0.5) * 255.0
    img_g = (np.cos(y_coord / 5.0) * 0.5 + 0.5) * 255.0
    img_b = ((x_coord + y_coord) % 32) * (255.0 / 32.0)

    raw_blocks = np.zeros((hb, wb, bpe), dtype=np.uint8)
    for y in range(hb):
        for x in range(wb):
            r = int(img_r[y, x] / 255.0 * 31)
            g = int(img_g[y, x] / 255.0 * 63)
            b = int(img_b[y, x] / 255.0 * 31)
            c = (r << 11) | (g << 5) | b
            c_lo, c_hi = c & 0xFF, (c >> 8) & 0xFF
            raw_blocks[y, x] = [c_lo, c_hi, c_lo, c_hi, 0, 0, 0, 0]
    return raw_blocks


def make_swizzled_gpu_buffer(
    raw_blocks: np.ndarray,
    offsets: np.ndarray,
    bpe: int,
    total_size: int,
    start_offset: int,
) -> bytes:
    """Pack 2D blocks into a swizzled GPU buffer at start_offset with pseudo-noise background."""
    hb, wb, _ = raw_blocks.shape
    noise = (np.arange(total_size, dtype=np.uint8) * 37 + 11) % 251
    buf = bytearray(noise.tobytes())
    for y in range(hb):
        for x in range(wb):
            idx = int(offsets[y, x])
            pos = start_offset + idx * bpe
            buf[pos : pos + bpe] = bytes(raw_blocks[y, x])
    return bytes(buf)


class TestDescriptorFormatClassification(unittest.TestCase):
    """Test texture descriptor format classification and CLI format validation."""

    def test_bc_bpe_valid_formats(self):
        self.assertEqual(tex_mode_probe.bc_bpe("BC1"), 8)
        self.assertEqual(tex_mode_probe.bc_bpe("BC2"), 16)
        self.assertEqual(tex_mode_probe.bc_bpe("BC3"), 16)

    def test_bc_bpe_dxgi_format_suffixes(self):
        cases = [
            ("BC1_UNORM", 8),
            ("BC1_UNORM_SRGB", 8),
            ("BC1_TYPELESS", 8),
            ("BC2_UNORM", 16),
            ("BC2_UNORM_SRGB", 16),
            ("BC2_TYPELESS", 16),
            ("BC3_UNORM", 16),
            ("BC3_UNORM_SRGB", 16),
            ("BC3_TYPELESS", 16),
        ]
        for fmt_name, expected_bpe in cases:
            with self.subTest(fmt_name=fmt_name):
                self.assertEqual(tex_mode_probe.bc_bpe(fmt_name), expected_bpe)

    def test_bc_bpe_unsupported_bc_formats(self):
        unsupported = [
            "BC4", "BC4_UNORM", "BC4_SNORM",
            "BC5", "BC5_UNORM", "BC5_SNORM",
            "BC6H", "BC6H_UF16", "BC6H_SF16",
            "BC7", "BC7_UNORM", "BC7_UNORM_SRGB",
        ]
        for fmt_name in unsupported:
            with self.subTest(fmt_name=fmt_name):
                self.assertIsNone(tex_mode_probe.bc_bpe(fmt_name))

    def test_bc_bpe_uncompressed_and_unknown_formats(self):
        invalids = [
            "R8_UNORM",
            "R8G8B8A8_UNORM",
            "R16G16B16A16_FLOAT",
            "UNKNOWN",
            "",
            "bc1",
            "bc1_unorm",
            "BC10_UNORM",
        ]
        for fmt_name in invalids:
            with self.subTest(fmt_name=fmt_name):
                self.assertIsNone(tex_mode_probe.bc_bpe(fmt_name))

    def test_probe_refuses_formats_without_a_color_scorer(self):
        for fmt in ["BC4", "BC5", "BC6H", "BC7"]:
            with self.subTest(fmt=fmt):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    index = os.path.join(tmp_dir, "index.json")
                    with open(index, "w", encoding="utf-8") as f:
                        f.write("[]")
                    with mock.patch.object(tex_mode_probe, "load", return_value=({}, {})), \
                         mock.patch.object(sys, "argv", ["tex_mode_probe.py", index, tmp_dir, "--fmt", fmt]), \
                         mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                        with self.assertRaises(SystemExit) as exc:
                            tex_mode_probe.main()
                        self.assertEqual(exc.exception.code, 2)
                        self.assertIn(fmt, mock_stderr.getvalue())
                        self.assertIn("no color scorer", mock_stderr.getvalue())

    def test_format_prefix_does_not_select_an_unsupported_index_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = os.path.join(tmp_dir, "index.json")
            rows = [{"fmt_name": "BC10_UNORM", "w": 256, "h": 256, "archive": "synthetic", "gpu": 16}]
            with open(index, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            with mock.patch.object(tex_mode_probe, "load", return_value=({}, {})), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index, tmp_dir, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()
            self.assertEqual(mock_stdout.getvalue(), "winners: []\nprobed 0 of 0 textures; 0 skipped\n")

    def test_descriptor_index_filtering(self):
        descriptors = [
            {"fmt_name": "BC1_UNORM", "w": 256, "h": 256, "archive": "a", "gpu": 1, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC1_UNORM", "w": 512, "h": 512, "archive": "a", "gpu": 2, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC1_UNORM", "w": 255, "h": 256, "archive": "a", "gpu": 3, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC1_UNORM", "w": 256, "h": 254, "archive": "a", "gpu": 4, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC2_UNORM", "w": 256, "h": 256, "archive": "a", "gpu": 5, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC4_UNORM", "w": 256, "h": 256, "archive": "a", "gpu": 6, "flags": 0, "data_off": 0, "gpu_size": 4096},
            {"fmt_name": "BC10_UNORM", "w": 256, "h": 256, "archive": "a", "gpu": 7, "flags": 0, "data_off": 0, "gpu_size": 4096},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = os.path.join(tmp_dir, "index.json")
            with open(idx_file, "w", encoding="utf-8") as f:
                json.dump(descriptors, f)

            with open(idx_file, encoding="utf-8") as f:
                all_texs = json.load(f)

            filtered_bc1 = [
                t for t in all_texs
                if t["fmt_name"].startswith("BC1")
                and tex_mode_probe.bc_bpe(t["fmt_name"]) is not None
                and t["w"] >= 256 and t["h"] >= 256
            ]
            self.assertEqual(len(filtered_bc1), 2)
            self.assertEqual(filtered_bc1[0]["gpu"], 1)
            self.assertEqual(filtered_bc1[1]["gpu"], 2)

            filtered_bc2 = [
                t for t in all_texs
                if t["fmt_name"].startswith("BC2")
                and tex_mode_probe.bc_bpe(t["fmt_name"]) is not None
                and t["w"] >= 256 and t["h"] >= 256
            ]
            self.assertEqual(len(filtered_bc2), 1)
            self.assertEqual(filtered_bc2[0]["gpu"], 5)

            filtered_bc4 = [
                t for t in all_texs
                if t["fmt_name"].startswith("BC4")
                and tex_mode_probe.bc_bpe(t["fmt_name"]) is not None
                and t["w"] >= 256 and t["h"] >= 256
            ]
            self.assertEqual(len(filtered_bc4), 0)

    def test_descriptor_flags_mip_extraction(self):
        cases = [
            (0x00000000, 0),
            (0x00010001, 1),
            (0x00040003, 4),
            (0x000B0000, 11),
            (0x00FF0000, 255),
            (0x12050000, 5),
        ]
        for flags, expected_mips in cases:
            with self.subTest(flags=hex(flags)):
                extracted = (flags >> 16) & 0xFF
                self.assertEqual(extracted, expected_mips)


class TestArgumentValidation(unittest.TestCase):
    """Test CLI argument parsing and validator functions."""

    def test_positive_int_valid_inputs(self):
        self.assertEqual(tex_mode_probe.positive_int("1"), 1)
        self.assertEqual(tex_mode_probe.positive_int("12"), 12)
        self.assertEqual(tex_mode_probe.positive_int("500"), 500)

    def test_positive_int_rejects_zero_and_negative(self):
        for bad in ["0", "-1", "-10"]:
            with self.subTest(value=bad):
                with self.assertRaises(argparse.ArgumentTypeError) as exc:
                    tex_mode_probe.positive_int(bad)
                self.assertIn("must be at least 1", str(exc.exception))

    def test_positive_int_rejects_non_integers(self):
        for bad in ["abc", "", "1.5", "None"]:
            with self.subTest(value=bad):
                with self.assertRaises(argparse.ArgumentTypeError) as exc:
                    tex_mode_probe.positive_int(bad)
                self.assertIn("not a whole number", str(exc.exception))

    def test_probe_refuses_invalid_indexes_before_loading_tables(self):
        for contents, diagnostic in ((None, "cannot read texture index"), ("[", "cannot read texture index"),
                                     ("{}", "texture index must contain a list")):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                index = os.path.join(directory, "index.json")
                if contents is not None:
                    with open(index, "w", encoding="utf-8", newline="\n") as f:
                        f.write(contents)
                stderr = io.StringIO()
                with mock.patch.object(tex_mode_probe, "load") as load, \
                        mock.patch.object(sys, "argv", ["tex_mode_probe.py", index, directory]), \
                        mock.patch.object(sys, "stderr", stderr), self.assertRaises(SystemExit) as exc:
                    tex_mode_probe.main()
                load.assert_not_called()
                self.assertEqual(exc.exception.code, 2)
                self.assertIn(diagnostic, stderr.getvalue())
                self.assertIn(index, stderr.getvalue())

    def test_probe_refuses_a_count_below_one_cli(self):
        for val in ["0", "-1", "x"]:
            with self.subTest(value=val):
                with tempfile.TemporaryDirectory() as td:
                    index = os.path.join(td, "index.json")
                    with open(index, "w", encoding="utf-8") as f:
                        f.write("[]")
                    with mock.patch.object(tex_mode_probe, "load", side_effect=AssertionError("should not load addrlib")), \
                         mock.patch.object(sys, "argv", ["tex_mode_probe.py", index, td, "--n", val]), \
                         mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                        with self.assertRaises(SystemExit) as exc:
                            tex_mode_probe.main()
                        self.assertEqual(exc.exception.code, 2)
                        self.assertIn("--n", mock_stderr.getvalue())


class TestGpuBlobAssembly(unittest.TestCase):
    """Test GPU buffer loading with single file and chained multi-part files."""

    def test_gpu_blob_single_file_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "archive_a")
            os.makedirs(arch_dir, exist_ok=True)
            gpu_id = 0x1000
            content = b"SingleGpuBlobData12345"
            with open(os.path.join(arch_dir, f"{gpu_id:012x}.bin"), "wb") as f:
                f.write(content)

            t = {"archive": "archive_a", "gpu": gpu_id, "gpu_size": len(content)}
            blob = tex_mode_probe.gpu_blob(td, t)
            self.assertEqual(blob, content)

    def test_gpu_blob_multi_file_gpu_chain(self):
        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "archive_b")
            os.makedirs(arch_dir, exist_ok=True)
            part1 = b"PartOne_"
            part2 = b"PartTwo_"
            part3 = b"PartThree"
            off1, off2, off3 = 0x2000, 0x3000, 0x4000
            with open(os.path.join(arch_dir, f"{off1:012x}.bin"), "wb") as f:
                f.write(part1)
            with open(os.path.join(arch_dir, f"{off2:012x}.bin"), "wb") as f:
                f.write(part2)
            with open(os.path.join(arch_dir, f"{off3:012x}.bin"), "wb") as f:
                f.write(part3)

            t = {
                "archive": "archive_b",
                "gpu": off1,
                "gpu_size": len(part1) + len(part2) + len(part3),
                "gpu_chain": [
                    [off1, len(part1)],
                    [off2, len(part2)],
                    [off3, len(part3)],
                ],
            }
            blob = tex_mode_probe.gpu_blob(td, t)
            self.assertEqual(blob, part1 + part2 + part3)


class TestMipBytesAndAlignment(unittest.TestCase):
    """Test mip byte size calculation and alignment rounding."""

    def test_block_dimension_quantization(self):
        cases = [
            (1, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 3, 1, 1),
            (4, 4, 1, 1),
            (5, 4, 2, 1),
            (4, 5, 1, 2),
            (5, 5, 2, 2),
            (8, 8, 2, 2),
            (9, 8, 3, 2),
            (256, 256, 64, 64),
            (257, 257, 65, 65),
        ]
        for w, h, exp_bw, exp_bh in cases:
            with self.subTest(w=w, h=h):
                bw = max(1, (w + 3) // 4)
                bh = max(1, (h + 3) // 4)
                self.assertEqual(bw, exp_bw)
                self.assertEqual(bh, exp_bh)

    def test_mip_bytes_alignment_rounding(self):
        # 4x4, bpe=8 -> 1 block = 8 bytes
        self.assertEqual(tex_mode_probe.mip_bytes(4, 4, 8, 1), 8)
        self.assertEqual(tex_mode_probe.mip_bytes(4, 4, 8, 256), 256)
        self.assertEqual(tex_mode_probe.mip_bytes(4, 4, 8, 4096), 4096)
        self.assertEqual(tex_mode_probe.mip_bytes(4, 4, 8, 65536), 65536)

        # 256x256, bpe=8 -> 64*64*8 = 32768 bytes
        self.assertEqual(tex_mode_probe.mip_bytes(256, 256, 8, 1), 32768)
        self.assertEqual(tex_mode_probe.mip_bytes(256, 256, 8, 4096), 32768)
        self.assertEqual(tex_mode_probe.mip_bytes(256, 256, 8, 65536), 65536)

        # 256x256, bpe=16 -> 64*64*16 = 65536 bytes
        self.assertEqual(tex_mode_probe.mip_bytes(256, 256, 16, 4096), 65536)
        self.assertEqual(tex_mode_probe.mip_bytes(256, 256, 16, 65536), 65536)

    def test_mip_bytes_non_power_of_two(self):
        # 37x19, bpe=8: bw=10, bh=5, 50 blocks * 8 = 400 bytes
        self.assertEqual(tex_mode_probe.mip_bytes(37, 19, 8, 1), 400)
        self.assertEqual(tex_mode_probe.mip_bytes(37, 19, 8, 256), 512)
        self.assertEqual(tex_mode_probe.mip_bytes(37, 19, 8, 4096), 4096)

    def test_mip_chain_layout_and_placement_offsets(self):
        w, h, bpe, align = 256, 256, 8, 4096
        top = tex_mode_probe.mip_bytes(w, h, bpe, align)
        self.assertEqual(top, 32768)

        # Single mip: chain == top
        chain_1 = sum(tex_mode_probe.mip_bytes(max(1, w >> i), max(1, h >> i), bpe, align) for i in range(1))
        self.assertEqual(chain_1, top)
        start_first_1 = 0x100
        start_last_1 = 0x100 + chain_1 - top
        self.assertEqual(start_first_1, start_last_1)

        # 4 mips:
        # mip 0: 256x256 -> 32768
        # mip 1: 128x128 -> 8192
        # mip 2: 64x64   -> 4096
        # mip 3: 32x32   -> 4096
        # chain = 32768 + 8192 + 4096 + 4096 = 49152
        chain_4 = sum(tex_mode_probe.mip_bytes(max(1, w >> i), max(1, h >> i), bpe, align) for i in range(4))
        self.assertEqual(chain_4, 49152)
        start_first_4 = 0
        start_last_4 = 0 + chain_4 - top
        self.assertEqual(start_last_4, 16384)
        self.assertGreater(start_last_4, start_first_4)


@unittest.skipUnless(HAVE_NUMPY, "numpy required for tex_mode_probe tiling tests")
class TestOffsetsForTilingModes(unittest.TestCase):
    """Test macro/micro-tile geometry and swizzled offset generation."""

    def test_tile_geometry_4k_and_64k(self):
        cases = [
            # bb, bpe, expected_elems, expected_bw, expected_bh
            (12, 8, 512, 32, 16),
            (12, 16, 256, 16, 16),
            (16, 8, 8192, 128, 64),
            (16, 16, 4096, 64, 64),
        ]
        for bb, bpe, exp_elems, exp_bw, exp_bh in cases:
            with self.subTest(bb=bb, bpe=bpe):
                elems = (1 << bb) // bpe
                lb = elems.bit_length() - 1
                bw = 1 << ((lb + 1) // 2)
                bh = elems // bw
                self.assertEqual(elems, exp_elems)
                self.assertEqual(bw, exp_bw)
                self.assertEqual(bh, exp_bh)
                self.assertEqual(bw * bh, exp_elems)

    def test_offsets_single_tile_bijection_4k_bc1(self):
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC1, 12, 8, 32, 16)
        self.assertEqual(o.shape, (16, 32))
        self.assertEqual(o.min(), 0)
        self.assertEqual(o.max(), 511)
        self.assertEqual(len(np.unique(o)), 512)

    def test_offsets_single_tile_bijection_64k_bc1(self):
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_64K_BC1, 16, 8, 128, 64)
        self.assertEqual(o.shape, (64, 128))
        self.assertEqual(o.min(), 0)
        self.assertEqual(o.max(), 8191)
        self.assertEqual(len(np.unique(o)), 8192)

    def test_offsets_single_tile_bijection_4k_bc2_3(self):
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC2_3, 12, 16, 16, 16)
        self.assertEqual(o.shape, (16, 16))
        self.assertEqual(o.min(), 0)
        self.assertEqual(o.max(), 255)
        self.assertEqual(len(np.unique(o)), 256)

    def test_offsets_single_tile_bijection_64k_bc2_3(self):
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_64K_BC2_3, 16, 16, 64, 64)
        self.assertEqual(o.shape, (64, 64))
        self.assertEqual(o.min(), 0)
        self.assertEqual(o.max(), 4095)
        self.assertEqual(len(np.unique(o)), 4096)

    def test_offsets_multi_tile_grid(self):
        # 64x64 blocks = 2x4 tiles of 32x16
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC1, 12, 8, 64, 64)
        self.assertEqual(o.shape, (64, 64))
        self.assertEqual(o.min(), 0)
        self.assertEqual(o.max(), 4095)
        self.assertEqual(len(np.unique(o)), 4096)

        # Check macro-tile partition spacing:
        tile_0_0 = o[0:16, 0:32]
        tile_1_0 = o[0:16, 32:64]
        tile_0_1 = o[16:32, 0:32]
        self.assertEqual(set(tile_0_0.flat) & set(tile_1_0.flat), set())
        self.assertEqual(set(tile_0_0.flat) & set(tile_0_1.flat), set())
        self.assertEqual(tile_1_0.min(), 512)
        self.assertEqual(tile_0_1.min(), 1024)

    def test_offsets_non_tile_multiple(self):
        # wb=35, hb=19 with tile 32x16 -> 2x2 macro tiles
        o = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC1, 12, 8, 35, 19)
        self.assertEqual(o.shape, (19, 35))
        self.assertEqual(len(np.unique(o)), 19 * 35)
        self.assertLess(o.max(), 4 * 512)

    def test_offsets_zero_equation(self):
        eq_zero = [(0, 0)] * 12
        o = tex_mode_probe.offsets_for(eq_zero, 12, 8, 64, 32)
        self.assertEqual(o.shape, (32, 64))
        # Within tile (0, 0), all values evaluate to 0
        self.assertTrue(np.all(o[0:16, 0:32] == 0))
        # Within tile (1, 0), macro offset is 1 * 512
        self.assertTrue(np.all(o[0:16, 32:64] == 512))
        # Within tile (0, 1), macro offset is (1 * 2 + 0) * 512 = 1024
        self.assertTrue(np.all(o[16:32, 0:32] == 1024))


class TestCandidatePatternFiltering(unittest.TestCase):
    """Test candidate extraction and pattern filtering from addrlib tables."""

    def test_candidate_filtering_and_deduplication(self):
        tab = {
            "GFX10_SW_64K_S_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
            "GFX10_SW_4K_D_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
            "GFX10_SW_64K_R_X_1xaa_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
            "GFX10_SW_64K_Z_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
            "GFX10_SW_256K_S_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
            "GFX10_SW_4K_S_DUPE_PATINFO": [(0, 0, 0, 0, 0, "1 pipes 8 bpe")],
        }

        def mock_equation(nib, e, bb):
            if "DUPE" in e[5] or bb == 12:
                return [(0, 0)] * bb
            return [(1, 0)] * bb

        cands = {}
        for tname, ents in tab.items():
            for size, bb in (("_64K_", 16), ("_4K_", 12)):
                if size in tname and ("_S" in tname or "_D" in tname or "_R_X_1xaa" in tname):
                    for e in ents:
                        if " 8 bpe" in e[5]:
                            eq = mock_equation(None, e, bb)
                            cands.setdefault(
                                (bb, tuple(eq)),
                                f"{tname.replace('GFX10_SW_', '').replace('_PATINFO', '')} {e[5][:22]}",
                            )

        # 64K_Z is excluded (_Z not in _S, _D, _R_X_1xaa)
        # 256K_S is excluded (neither _64K_ nor _4K_)
        # Both 4K entries generate [(0, 0)]*12 so they deduplicate to a single entry in cands
        keys = list(cands.keys())
        self.assertEqual(len(keys), 2)
        bbs = {k[0] for k in keys}
        self.assertEqual(bbs, {12, 16})

    def test_candidate_bpe_matching(self):
        tab = {
            "GFX10_SW_4K_S_PATINFO": [
                (0, 0, 0, 0, 0, "1 pipes 8 bpe"),
                (0, 0, 0, 0, 0, "1 pipes 16 bpe"),
                (0, 0, 0, 0, 0, "1 pipes 4 bpe"),
            ]
        }
        bpe_8_matches = [e for e in tab["GFX10_SW_4K_S_PATINFO"] if f" {tex_mode_probe.bc_bpe('BC1')} bpe" in e[5]]
        bpe_16_matches = [e for e in tab["GFX10_SW_4K_S_PATINFO"] if f" {tex_mode_probe.bc_bpe('BC3')} bpe" in e[5]]
        self.assertEqual(len(bpe_8_matches), 1)
        self.assertEqual(len(bpe_16_matches), 1)
        self.assertIn("8 bpe", bpe_8_matches[0][5])
        self.assertIn("16 bpe", bpe_16_matches[0][5])


@unittest.skipUnless(HAVE_NUMPY, "numpy required for tex_mode_probe end-to-end probing")
class TestTextureModeProbing(unittest.TestCase):
    """Test texture mode probing logic and winner ranking with synthetic chunks."""

    def setUp(self):
        self.wb, self.hb = 64, 64
        self.w, self.h = 256, 256
        self.bpe = 8
        self.mips = 2
        self.flags = (self.mips << 16) | 1

        self.raw_blocks = make_synthetic_bc1_blocks(self.wb, self.hb)
        self.o_correct = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC1, 12, self.bpe, self.wb, self.hb)

        self.top = tex_mode_probe.mip_bytes(self.w, self.h, self.bpe, 4096)
        self.chain = sum(
            tex_mode_probe.mip_bytes(max(1, self.w >> i), max(1, self.h >> i), self.bpe, 4096)
            for i in range(self.mips)
        )

        self.nib = {
            "NIBBLE01": [[0] * 8, [0] * 8],
            "NIBBLE2": [[0] * 4, [0] * 4],
            "NIBBLE3": [[0] * 4, [0] * 4],
            "NIBBLE4": [[0] * 4, [0] * 4],
        }
        self.entry_correct = (0, 0, 0, 0, 0, "synthetic 8 bpe")
        self.entry_incorrect = (1, 1, 1, 1, 1, "synthetic 8 bpe")
        self.tab = {
            "GFX10_SW_4K_S_PATINFO": [self.entry_correct],
            "GFX10_SW_4K_D_PATINFO": [self.entry_incorrect],
        }

    def _mock_equation(self, nib, entry, bb):
        if entry == self.entry_correct:
            return SYNTHETIC_EQ_4K_BC1
        # Alternate permutation with swapped coordinates:
        return [
            (0, 0), (0, 0), (0, 0),
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 4), (4, 0),
            (0, 8), (8, 0),
            (0, 16),
        ][:bb]

    def test_probe_winner_selection_top_mip_first(self):
        # Pack data at 'first' placement (start = 0)
        gpu_bytes = make_swizzled_gpu_buffer(
            self.raw_blocks, self.o_correct, self.bpe, self.chain, start_offset=0
        )

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch0")
            os.makedirs(arch_dir, exist_ok=True)
            gpu_file = os.path.join(arch_dir, f"{0x1000:012x}.bin")
            with open(gpu_file, "wb") as f:
                f.write(gpu_bytes)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([{
                    "archive": "arch0",
                    "gpu": 0x1000,
                    "gpu_size": len(gpu_bytes),
                    "fmt_name": "BC1_UNORM",
                    "w": self.w,
                    "h": self.h,
                    "flags": self.flags,
                    "data_off": 0,
                }], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertIn("best", output)
            self.assertIn("4K_S", output)
            self.assertIn("[first]", output)
            self.assertIn("winners: [(('4K_S', 'first'), 1)]", output)
            self.assertIn("probed 1 of 1 textures; 0 skipped", output)

    def test_probe_winner_selection_top_mip_last(self):
        # Pack data at 'last' placement (start = chain - top = 8192)
        start_last = self.chain - self.top
        gpu_bytes = make_swizzled_gpu_buffer(
            self.raw_blocks, self.o_correct, self.bpe, self.chain, start_offset=start_last
        )

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch0")
            os.makedirs(arch_dir, exist_ok=True)
            gpu_file = os.path.join(arch_dir, f"{0x2000:012x}.bin")
            with open(gpu_file, "wb") as f:
                f.write(gpu_bytes)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([{
                    "archive": "arch0",
                    "gpu": 0x2000,
                    "gpu_size": len(gpu_bytes),
                    "fmt_name": "BC1_UNORM",
                    "w": self.w,
                    "h": self.h,
                    "flags": self.flags,
                    "data_off": 0,
                }], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertIn("best", output)
            self.assertIn("4K_S", output)
            self.assertIn("[last]", output)
            self.assertIn("winners: [(('4K_S', 'last'), 1)]", output)
            self.assertIn("probed 1 of 1 textures; 0 skipped", output)

    def test_probe_top_mip_after_non_aligned_tile_grid(self) -> None:
        # The 132x136 lower mip is 33x34 BC1 blocks, requiring 2x3 tiles of 32x16 blocks.
        # The 264x272 top mip is 66x68 blocks, requiring 3x5 tiles.
        width, height = 264, 272
        top_blocks_w, top_blocks_h = 66, 68
        block = np.array([0, 248, 0, 248, 0, 0, 0, 0], dtype=np.uint8)
        top = np.tile(block, (top_blocks_h, top_blocks_w, 1))
        offsets = tex_mode_probe.offsets_for(SYNTHETIC_EQ_4K_BC1, 12, 8, top_blocks_w, top_blocks_h)
        gpu = make_swizzled_gpu_buffer(top, offsets, 8, (6 + 15) * 4096, start_offset=6 * 4096)

        with tempfile.TemporaryDirectory() as directory:
            archive = os.path.join(directory, "synthetic")
            os.mkdir(archive)
            with open(os.path.join(archive, f"{0x1000:012x}.bin"), "wb") as f:
                f.write(gpu)
            index = os.path.join(directory, "index.json")
            with open(index, "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"archive": "synthetic", "gpu": 0x1000, "gpu_size": len(gpu),
                            "fmt_name": "BC1_UNORM", "w": width, "h": height,
                            "flags": self.flags, "data_off": 0}], f)
            table = {"GFX10_SW_4K_S_PATINFO": [self.entry_correct]}
            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, table)), \
                 mock.patch.object(tex_mode_probe, "equation", return_value=SYNTHETIC_EQ_4K_BC1), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index, directory]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                tex_mode_probe.main()

        self.assertRegex(stdout.getvalue(), r"best 0\.0000 4K_S .*? \[last\]")
        self.assertIn("winners: [(('4K_S', 'last'), 1)]", stdout.getvalue())

    def test_probe_skips_truncated_gpu_segment(self):
        # Provide a GPU segment file that is too short for any candidate
        truncated_bytes = b"\x00" * 128

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch0")
            os.makedirs(arch_dir, exist_ok=True)
            gpu_file = os.path.join(arch_dir, f"{0x3000:012x}.bin")
            with open(gpu_file, "wb") as f:
                f.write(truncated_bytes)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([{
                    "archive": "arch0",
                    "gpu": 0x3000,
                    "gpu_size": len(truncated_bytes),
                    "fmt_name": "BC1_UNORM",
                    "w": self.w,
                    "h": self.h,
                    "flags": self.flags,
                    "data_off": 0,
                }], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertIn("skipped: no candidate fits in the GPU blob", output)
            self.assertIn("winners: []", output)
            self.assertIn("probed 0 of 1 textures; 1 skipped", output)

    def test_probe_multi_part_gpu_chain(self):
        # Split GPU buffer into 2 chained parts:
        start_last = self.chain - self.top
        gpu_bytes = make_swizzled_gpu_buffer(
            self.raw_blocks, self.o_correct, self.bpe, self.chain, start_offset=start_last
        )
        half = len(gpu_bytes) // 2
        part1 = gpu_bytes[:half]
        part2 = gpu_bytes[half:]
        off1, off2 = 0x6001, 0x6002

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch_chain")
            os.makedirs(arch_dir, exist_ok=True)
            with open(os.path.join(arch_dir, f"{off1:012x}.bin"), "wb") as f:
                f.write(part1)
            with open(os.path.join(arch_dir, f"{off2:012x}.bin"), "wb") as f:
                f.write(part2)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([{
                    "archive": "arch_chain",
                    "gpu": off1,
                    "gpu_size": len(gpu_bytes),
                    "gpu_chain": [[off1, len(part1)], [off2, len(part2)]],
                    "fmt_name": "BC1_UNORM",
                    "w": self.w,
                    "h": self.h,
                    "flags": self.flags,
                    "data_off": 0,
                }], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertIn("best", output)
            self.assertIn("4K_S", output)
            self.assertIn("[last]", output)
            self.assertIn("winners: [(('4K_S', 'last'), 1)]", output)
            self.assertIn("probed 1 of 1 textures; 0 skipped", output)

    def test_probe_linear_score_reporting(self):
        # Verify linear score reporting when segment is full vs truncated
        gpu_bytes = make_swizzled_gpu_buffer(
            self.raw_blocks, self.o_correct, self.bpe, self.chain, start_offset=0
        )

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch0")
            os.makedirs(arch_dir, exist_ok=True)
            gpu_file = os.path.join(arch_dir, f"{0x4000:012x}.bin")
            with open(gpu_file, "wb") as f:
                f.write(gpu_bytes)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([{
                    "archive": "arch0",
                    "gpu": 0x4000,
                    "gpu_size": len(gpu_bytes),
                    "fmt_name": "BC1_UNORM",
                    "w": self.w,
                    "h": self.h,
                    "flags": self.flags,
                    "data_off": 0,
                }], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertIn("linear", output)
            self.assertNotIn("linear -1.0000", output)

    def test_probe_empty_index_outputs_empty_winners(self):
        with tempfile.TemporaryDirectory() as td:
            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), "winners: []\nprobed 0 of 0 textures; 0 skipped")

    def test_probe_max_texture_limit_arg(self):
        gpu_bytes = make_swizzled_gpu_buffer(
            self.raw_blocks, self.o_correct, self.bpe, self.chain, start_offset=0
        )

        with tempfile.TemporaryDirectory() as td:
            arch_dir = os.path.join(td, "arch0")
            os.makedirs(arch_dir, exist_ok=True)
            for gpu_id in (0x5001, 0x5002, 0x5003):
                with open(os.path.join(arch_dir, f"{gpu_id:012x}.bin"), "wb") as f:
                    f.write(gpu_bytes)

            index_file = os.path.join(td, "textures_index.json")
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump([
                    {"archive": "arch0", "gpu": 0x5001, "gpu_size": len(gpu_bytes), "fmt_name": "BC1_UNORM", "w": self.w, "h": self.h, "flags": self.flags, "data_off": 0},
                    {"archive": "arch0", "gpu": 0x5002, "gpu_size": len(gpu_bytes), "fmt_name": "BC1_UNORM", "w": self.w, "h": self.h, "flags": self.flags, "data_off": 0},
                    {"archive": "arch0", "gpu": 0x5003, "gpu_size": len(gpu_bytes), "fmt_name": "BC1_UNORM", "w": self.w, "h": self.h, "flags": self.flags, "data_off": 0},
                ], f)

            with mock.patch.object(tex_mode_probe, "load", return_value=(self.nib, self.tab)), \
                 mock.patch.object(tex_mode_probe, "equation", side_effect=self._mock_equation), \
                 mock.patch.object(sys, "argv", ["tex_mode_probe.py", index_file, td, "--fmt", "BC1", "--n", "1"]), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                tex_mode_probe.main()

            output = mock_stdout.getvalue()
            lines = [line for line in output.strip().splitlines() if "256x256" in line]
            self.assertEqual(len(lines), 1)
            self.assertIn("probed 1 of 1 textures; 0 skipped", output)


# Maintain compatibility with pytest parameterised runners:
if pytest is not None:
    @pytest.mark.parametrize("fmt", ["BC4", "BC5", "BC6H", "BC7"])
    def test_probe_refuses_formats_without_a_color_scorer_pytest(fmt, monkeypatch, capsys, tmp_path):
        index = tmp_path / "index.json"
        index.write_text("[]", encoding="utf-8")
        monkeypatch.setattr(tex_mode_probe, "load", lambda: ({}, {}))
        monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path), "--fmt", fmt])
        with pytest.raises(SystemExit) as exc:
            tex_mode_probe.main()
        assert exc.value.code == 2
        assert fmt in capsys.readouterr().err

    @pytest.mark.parametrize("value", ["0", "-1", "x"])
    def test_probe_refuses_a_count_below_one_pytest(value, monkeypatch, capsys, tmp_path):
        index = tmp_path / "index.json"
        index.write_text("[]", encoding="utf-8")
        monkeypatch.setattr(tex_mode_probe, "load", lambda: pytest.fail("the probe must stop before loading addrlib"))
        monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path), "--n", value])
        with pytest.raises(SystemExit) as exc:
            tex_mode_probe.main()
        assert exc.value.code == 2
        assert "--n" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "data_size, candidate_count, parts, mips",
        [(32760, 0, 1, 2), (32768, 1, 1, 2), (40960, 2, 1, 2), (32768, 1, 2, 2),
         (40960, 2, 3, 2), (32768, 1, 1, 1)],
    )
    def test_probe_reports_available_candidates_pytest(
        data_size: int, candidate_count: int, parts: int, mips: int,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
    ) -> None:
        eq = [(0, 0)] * 3 + [(1 << i, 0) for i in range(5)] + [(0, 1 << i) for i in range(4)]
        entry = (0, 0, 0, 0, 0, "synthetic 8 bpe")
        monkeypatch.setattr(tex_mode_probe, "load", lambda: ({}, {"GFX10_SW_4K_S_PATINFO": [entry]}))
        monkeypatch.setattr(tex_mode_probe, "equation", lambda nib, e, bb: eq)
        carve = tmp_path / "synthetic"
        carve.mkdir()
        cuts = [data_size * i // parts for i in range(parts + 1)]
        chain = [[16 + 0x100000 * i, cuts[i + 1] - cuts[i]] for i in range(parts)]
        for off, size in chain:
            (carve / f"{off:012x}.bin").write_bytes(bytes(size))
        row = {
            "fmt_name": "BC1_UNORM",
            "w": 256,
            "h": 256,
            "archive": "synthetic",
            "gpu": 16,
            "flags": mips << 16,
            "data_off": 0,
            "gpu_size": data_size,
        }
        if parts > 1:
            row["gpu_chain"] = chain
        index = tmp_path / "index.json"
        index.write_text(json.dumps([row]), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path)])

        tex_mode_probe.main()

        out = capsys.readouterr().out
        if candidate_count == 0:
            assert out == (
                "256x256 BC1_UNORM gpu=0x10  skipped: no candidate fits in the GPU blob\n"
                "winners: []\nprobed 0 of 1 textures; 1 skipped\n"
            )
        else:
            second = "n/a" if candidate_count == 1 else "0.0000 4K_S synthetic 8 bpe [last]"
            assert out == (
                f"256x256 BC1_UNORM mips={mips} gpu={data_size}  best 0.0000 4K_S synthetic 8 bpe [first]"
                f"  2nd {second}  linear 0.0000\n"
                "winners: [(('4K_S', 'first'), 1)]\n"
                "probed 1 of 1 textures; 0 skipped\n"
            )


if __name__ == "__main__":
    unittest.main()
