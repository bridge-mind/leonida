"""Unit tests for tools/tex_headers.py: DDS header synthesis, mipmap dimension calculations,
GFX10 format mapping, and texture descriptor parsing from synthetic blobs.

    python -I tests/test_tex_headers.py
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import tex_headers


class TestGFX10FormatMapping(unittest.TestCase):
    def test_known_bc_format_mappings(self):
        cases = [
            (169, 71, "BC1_UNORM"),
            (170, 72, "BC1_UNORM_SRGB"),
            (171, 74, "BC2_UNORM"),
            (172, 75, "BC2_UNORM_SRGB"),
            (173, 77, "BC3_UNORM"),
            (174, 78, "BC3_UNORM_SRGB"),
            (175, 80, "BC4_UNORM"),
            (176, 81, "BC4_SNORM"),
            (177, 83, "BC5_UNORM"),
            (178, 84, "BC5_SNORM"),
            (179, 95, "BC6H_UF16"),
            (180, 96, "BC6H_SF16"),
            (181, 98, "BC7_UNORM"),
            (182, 99, "BC7_UNORM_SRGB"),
        ]
        for gfx10_code, expected_dxgi, expected_name in cases:
            with self.subTest(gfx10=gfx10_code, name=expected_name):
                dxgi_id, name = tex_headers.map_gfx10_format(gfx10_code)
                self.assertEqual(dxgi_id, expected_dxgi)
                self.assertEqual(name, expected_name)

    def test_known_uncompressed_mappings(self):
        cases = [
            (1, 61, "R8_UNORM"),
            (7, 56, "R16_UNORM"),
            (13, 54, "R16_FLOAT"),
            (14, 49, "R8G8_UNORM"),
            (22, 41, "R32_FLOAT"),
            (29, 34, "R16G16_FLOAT"),
            (36, 26, "R11G11B10_FLOAT"),
            (50, 24, "R10G10B10A2_UNORM"),
            (56, 28, "R8G8B8A8_UNORM"),
            (130, 29, "R8G8B8A8_UNORM_SRGB"),
            (64, 16, "R32G32_FLOAT"),
            (71, 10, "R16G16B16A16_FLOAT"),
            (77, 2, "R32G32B32A32_FLOAT"),
        ]
        for gfx10_code, expected_dxgi, expected_name in cases:
            with self.subTest(gfx10=gfx10_code, name=expected_name):
                dxgi_id, name = tex_headers.map_gfx10_format(gfx10_code)
                self.assertEqual(dxgi_id, expected_dxgi)
                self.assertEqual(name, expected_name)

    def test_packed_formats_follow_top_bit_first_names(self):
        # GFX10 names list widths from the top bit: 2_10_10_10 is R10G10B10A2, 10_11_11 is R11G11B10.
        with open(os.path.join(TOOLS_DIR, "gfx10_formats.json"), encoding="utf-8") as f:
            names = json.load(f)
        for gfx10, dxgi, name in ((36, 26, "10_11_11_FLOAT"), (50, 24, "2_10_10_10_UNORM"),
                                  (54, 25, "2_10_10_10_UINT")):
            with self.subTest(gfx10=gfx10):
                self.assertEqual(names[str(gfx10)], "GFX10_FORMAT_" + name)
                self.assertEqual(tex_headers.map_gfx10_format(gfx10)[0], dxgi)
                self.assertEqual(tex_headers.dxgi_to_gfx10(dxgi), gfx10)
        for gfx10 in (43, 44, 48):   # 11_11_10_FLOAT, 10_10_10_2_*: no DXGI format has these layouts
            with self.subTest(gfx10=gfx10), self.assertRaises(ValueError):
                tex_headers.map_gfx10_format(gfx10)

    def test_unsupported_gfx10_raises(self):
        for invalid in [0, 9999, -1, 400]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tex_headers.map_gfx10_format(invalid)

    def test_dxgi_to_gfx10_roundtrip(self):
        major_dxgi = [71, 72, 74, 77, 78, 80, 83, 95, 96, 98, 99, 28, 29, 56, 61]
        for dxgi_id in major_dxgi:
            with self.subTest(dxgi=dxgi_id):
                gfx10 = tex_headers.dxgi_to_gfx10(dxgi_id)
                mapped_dxgi, _ = tex_headers.map_gfx10_format(gfx10)
                self.assertEqual(mapped_dxgi, dxgi_id)

    def test_unsupported_dxgi_raises(self):
        with self.assertRaises(ValueError):
            tex_headers.dxgi_to_gfx10(99999)

    def test_dxgi_dict_integrity(self):
        for dxgi_id, name in tex_headers.DXGI.items():
            self.assertIsInstance(dxgi_id, int)
            self.assertIsInstance(name, str)
            self.assertTrue(len(name) > 0)


class TestMipmapCalculations(unittest.TestCase):
    def test_calc_mipmap_count(self):
        cases = [
            (1, 1, 1, 1),
            (2, 2, 1, 2),
            (4, 4, 1, 3),
            (8, 8, 1, 4),
            (256, 256, 1, 9),
            (1024, 1024, 1, 11),
            (1920, 1080, 1, 11),
            (4096, 4096, 1, 13),
            (512, 64, 1, 10),
            (64, 64, 16, 7),
        ]
        for w, h, d, expected in cases:
            with self.subTest(w=w, h=h, d=d):
                self.assertEqual(tex_headers.calc_mipmap_count(w, h, d), expected)

    def test_calc_mipmap_count_zero_or_negative(self):
        self.assertEqual(tex_headers.calc_mipmap_count(0, 100), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(100, 0), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(-10, 100), 0)
        self.assertEqual(tex_headers.calc_mipmap_count(100, 100, 0), 0)

    def test_calc_mipmap_dimensions_full_chain(self):
        levels = tex_headers.calc_mipmap_dimensions(16, 16, 1, mips=0)
        expected = [
            (16, 16, 1),
            (8, 8, 1),
            (4, 4, 1),
            (2, 2, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_non_square(self):
        levels = tex_headers.calc_mipmap_dimensions(32, 8, 1, mips=0)
        expected = [
            (32, 8, 1),
            (16, 4, 1),
            (8, 2, 1),
            (4, 1, 1),
            (2, 1, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_limited_count(self):
        levels = tex_headers.calc_mipmap_dimensions(1024, 1024, 1, mips=3)
        expected = [
            (1024, 1024, 1),
            (512, 512, 1),
            (256, 256, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_3d_volume(self):
        levels = tex_headers.calc_mipmap_dimensions(16, 16, 8, mips=0)
        expected = [
            (16, 16, 8),
            (8, 8, 4),
            (4, 4, 2),
            (2, 2, 1),
            (1, 1, 1),
        ]
        self.assertEqual(levels, expected)

    def test_calc_mipmap_dimensions_invalid(self):
        self.assertEqual(tex_headers.calc_mipmap_dimensions(0, 10), [])
        self.assertEqual(tex_headers.calc_mipmap_dimensions(10, -5), [])

    def test_calc_mip_level_bytes_bc1(self):
        # BC1 is 8 bytes per 4x4 block
        # 4x4 -> 1 block = 8 bytes
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71), 8)
        # 1024x1024 -> 256x256 blocks = 65536 blocks * 8 bytes = 524288 bytes
        self.assertEqual(tex_headers.calc_mip_level_bytes(1024, 1024, 1, fmt=71), 524288)
        # Sub-block mip levels (2x2, 1x1) must still pad to a 4x4 block
        self.assertEqual(tex_headers.calc_mip_level_bytes(2, 2, 1, fmt=71), 8)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1, 1, 1, fmt=71), 8)

    def test_calc_mip_level_bytes_bc7(self):
        # BC7 is 16 bytes per 4x4 block
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=98), 16)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1024, 1024, 1, fmt=98), 1048576)
        self.assertEqual(tex_headers.calc_mip_level_bytes(1, 1, 1, fmt=98), 16)

    def test_calc_mip_level_bytes_uncompressed(self):
        # R8_UNORM = 1 byte per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=61), 4096)
        # R8G8B8A8_UNORM = 4 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=28), 16384)
        # R16G16B16A16_FLOAT = 8 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=10), 32768)
        # R32G32B32A32_FLOAT = 16 bytes per pixel
        self.assertEqual(tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=2), 65536)

    def test_calc_mip_level_bytes_aligned(self):
        # 8 bytes aligned to 256 -> 256
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71, align=256), 256)
        # 8 bytes aligned to 4096 -> 4096
        self.assertEqual(tex_headers.calc_mip_level_bytes(4, 4, 1, fmt=71, align=4096), 4096)

    def test_calc_mip_level_bytes_unsupported_format(self):
        with self.assertRaises(ValueError):
            tex_headers.calc_mip_level_bytes(64, 64, 1, fmt=9999)


class TestCompactMipLayout(unittest.TestCase):
    def test_descriptor_mode_is_flag_plus_tile_mode(self):
        cases = {
            0: (0, 65536), 1: (0, 65536),
            2: (1, 4096), 3: (1, 4096),
            4: (2, 4096), 5: (2, 4096),
            6: (3, 4096), 7: (3, 4096),
            32: (0, 65536), 34: (1, 4096), 36: (2, 4096), 38: (3, 4096),
            18: (9, 4096), 20: (10, 4096), 22: (11, 4096), 24: (12, 4096),
        }
        for mode, (tile_mode, tile_bytes) in cases.items():
            with self.subTest(mode=mode):
                self.assertEqual(tex_headers.gfx10_tile_mode(mode), tile_mode)
                self.assertEqual(tex_headers.gfx10_tile_bytes(mode), tile_bytes)

    def test_invalid_mode_rejected(self):
        for mode in (-1, -32, "4", None):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    tex_headers.gfx10_tile_mode(mode)

    def test_known_4k_chain_sizes(self):
        lay = tex_headers.compact_layout_for_dxgi(256, 256, 71, 7)
        self.assertEqual(lay["size"], 45056)
        self.assertEqual(lay["first_tail"], 2)
        self.assertEqual(lay["block_w"], 32)
        self.assertEqual(lay["block_h"], 16)
        self.assertEqual([m["region"] for m in lay["mips"]],
                         [32768, 8192, 4096, 4096, 4096, 4096, 4096])
        self.assertEqual([m["offset"] for m in lay["mips"][:2]], [12288, 4096])
        self.assertEqual((lay["mips"][2]["tile_x"], lay["mips"][2]["tile_y"]), (16, 0))
        self.assertEqual((lay["mips"][6]["tile_x"], lay["mips"][6]["tile_y"]), (8, 4))

    def test_known_1024_chain_matches_allocation(self):
        lay = tex_headers.compact_layout_for_dxgi(1024, 1024, 71, 9)
        self.assertEqual(lay["size"], 700416)
        self.assertEqual(lay["first_tail"], 4)
        self.assertEqual([m["region"] for m in lay["mips"][:4]],
                         [524288, 131072, 32768, 8192])
        self.assertEqual([m["offset"] for m in lay["mips"][:4]],
                         [176128, 45056, 12288, 4096])
        self.assertTrue(tex_headers.compact_span_matches(lay, 700416))

    def test_known_64k_tile_chain(self):
        lay = tex_headers.compact_layout_for_dxgi(1024, 1024, 98, 9,
                                                  tile=tex_headers.COMPACT_TILE_64K)
        self.assertEqual(lay["tile"], 65536)
        self.assertEqual(lay["first_tail"], 3)
        self.assertEqual(lay["size"], 1441792)
        self.assertEqual([m["region"] for m in lay["mips"][:3]],
                         [1048576, 262144, 65536])

    def test_non_square_chain(self):
        lay = tex_headers.compact_layout_for_dxgi(512, 256, 71, 7)
        self.assertEqual(lay["size"], 90112)
        self.assertEqual(lay["first_tail"], 3)
        self.assertEqual([m["region"] for m in lay["mips"][:3]],
                         [65536, 16384, 4096])
        self.assertEqual(lay["mips"][3]["pixels_w"], 64)
        self.assertEqual(lay["mips"][3]["pixels_h"], 32)

    def test_single_mip_is_padded_to_one_tile(self):
        lay = tex_headers.compact_layout_for_dxgi(64, 64, 71, 1)
        self.assertEqual(lay["size"], 4096)
        self.assertEqual(lay["first_tail"], 1)
        self.assertEqual(lay["mips"][0]["offset"], 0)
        self.assertEqual(lay["mips"][0]["region"], 4096)
        self.assertFalse(lay["mips"][0]["in_tail"])

    def test_sub_tile_mips_share_the_tail(self):
        lay = tex_headers.compact_layout_for_dxgi(8, 4, 71, 2)
        self.assertEqual(lay["size"], 4096)
        self.assertEqual(lay["first_tail"], 0)
        self.assertTrue(all(m["in_tail"] for m in lay["mips"]))
        self.assertEqual([(m["tile_x"], m["tile_y"]) for m in lay["mips"]],
                         [(16, 0), (8, 8)])

    def test_uncompressed_chain_is_sized_by_pixels(self):
        lay = tex_headers.compact_layout_for_dxgi(128, 128, 28, 8)
        self.assertEqual(lay["element_bytes"], 4)
        self.assertEqual(lay["size"], 90112)
        self.assertEqual(lay["first_tail"], 3)
        self.assertEqual([m["region"] for m in lay["mips"][:3]],
                         [65536, 16384, 4096])
        self.assertEqual(lay["mips"][0]["blocks_w"], 128)
        self.assertEqual(lay["mips"][0]["blocks_h"], 128)

    def test_size_is_standalone_regions_plus_one_tail_tile(self):
        for args in [(256, 256, 71, 7), (1024, 1024, 98, 9),
                     (128, 256, 98, 6), (512, 256, 28, 9)]:
            with self.subTest(args=args):
                lay = tex_headers.compact_layout_for_dxgi(*args)
                standalone = sum(m["region"] for m in lay["mips"] if not m["in_tail"])
                tail = lay["tile"] if lay["first_tail"] < len(lay["mips"]) else 0
                self.assertEqual(lay["size"], standalone + tail)
                self.assertTrue(tex_headers.compact_span_matches(lay, lay["size"]))

    def test_inconsistent_span_rejected(self):
        lay = tex_headers.compact_layout_for_dxgi(256, 256, 71, 7)
        self.assertTrue(tex_headers.compact_span_matches(lay, 45056))
        self.assertFalse(tex_headers.compact_span_matches(lay, 45057))
        self.assertFalse(tex_headers.compact_span_matches(lay, 40960))
        self.assertFalse(tex_headers.compact_span_matches(lay, None))

    def test_invalid_mip_counts_rejected(self):
        for width, height, mips in [(256, 256, 0), (256, 256, -1),
                                    (64, 64, 20), (1, 1, 2)]:
            with self.subTest(width=width, height=height, mips=mips):
                with self.assertRaises(ValueError):
                    tex_headers.compact_surface_layout(width, height, mips, 8, 4)

    def test_invalid_sizes_rejected(self):
        bad = [
            dict(width=0, height=256, mips=1, element_bytes=8, block=4),
            dict(width=256, height=-1, mips=1, element_bytes=8, block=4),
            dict(width=256, height=256, mips=7, element_bytes=3, block=4),
            dict(width=256, height=256, mips=7, element_bytes=12, block=4),
            dict(width=256, height=256, mips=7, element_bytes=8, block=2),
            dict(width=256, height=256, mips=7, element_bytes=8, block=4, tile=8192),
        ]
        for case in bad:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    tex_headers.compact_surface_layout(**case)

    def test_unknown_dxgi_format_rejected(self):
        with self.assertRaises(ValueError):
            tex_headers.compact_layout_for_dxgi(256, 256, 9999, 7)

    def test_parse_etpf_record(self):
        a = (512 - 1) | ((256 - 1) << 14) | (7 << 28)
        b = 1 | (71 << 3) | (0 << 11) | (1 << 21) | (4 << 23)
        c = 0x1234 | (0x103 << 21)
        d = 0x04030201
        desc = tex_headers.parse_etpf_texture_record(struct.pack("<IIII", a, b, c, d))
        self.assertEqual(desc["width"], 512)
        self.assertEqual(desc["height"], 256)
        self.assertEqual(desc["packed_mips"], 7)
        self.assertEqual(desc["non_tail"], 1)
        self.assertEqual(desc["mips"], 8)
        self.assertEqual(desc["format"], 71)
        self.assertEqual(desc["depth"], 1)
        self.assertEqual(desc["dimension"], 1)
        self.assertEqual(desc["mode"], 4)
        self.assertEqual(desc["meta"], 0x103)
        self.assertEqual(desc["extra"], 0x04030201)
        self.assertEqual(desc["base"], 0x1234 * 4096)
        self.assertEqual(tex_headers.compact_profile_key(desc, 8),
                         (4, 512, 256, 8, 1, 8))
        with self.assertRaises(ValueError):
            tex_headers.parse_etpf_texture_record(b"\x00" * 15)

    def test_parse_etpf_maximum_fields(self):
        # All packed fields at their maximum: 16 Ki mips worth of split bits,
        # 7-bit format (0..127), 11-bit depth (1..2048), dimension/mode/meta/base.
        # The parser reports the bit fields; layout validation is what rejects
        # an impossible mip count.
        record = struct.pack("<IIII", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        desc = tex_headers.parse_etpf_texture_record(record)
        self.assertEqual((desc["width"], desc["height"]), (16384, 16384))
        self.assertEqual((desc["packed_mips"], desc["non_tail"], desc["mips"]), (15, 7, 22))
        self.assertEqual((desc["format"], desc["depth"], desc["dimension"]), (127, 2048, 3))
        self.assertEqual(desc["mode"], 511)
        self.assertEqual(desc["meta"], 0x7FF)
        self.assertEqual(desc["base"], 0x1FFFFF * 4096)
        with self.assertRaises(ValueError):
            tex_headers.compact_surface_layout(desc["width"], desc["height"],
                                               desc["mips"], 16, 4)

    def test_parse_etpf_bit10_format_depth_boundary(self):
        # Bit 10 of word b is depth bit 0 (depth - 1), not format bit 7.
        cases = [
            (28, 2, 1),   # old 8-bit format view would have read 28 | 128 = 156, depth=1
            (71, 6, 2),   # cubemap: depth - 1 = 5 (odd -> bit 10 set)
            (54, 8, 3),   # 3D volume: depth - 1 = 7 (odd -> bit 10 set)
        ]
        for fmt, depth, dim in cases:
            with self.subTest(fmt=fmt, depth=depth, dim=dim):
                a = (64 - 1) | ((64 - 1) << 14) | (1 << 28)
                b = 0 | (fmt << 3) | ((depth - 1) << 10) | (dim << 21) | (1 << 23)
                desc = tex_headers.parse_etpf_texture_record(struct.pack("<IIII", a, b, 1, 0))
                self.assertEqual(desc["format"], fmt)
                self.assertEqual(desc["depth"], depth)
                self.assertEqual(desc["dimension"], dim)

    def test_decode_etpf_chunk_pages_and_padding(self):
        # Empty metadata decodes to empty tuple.
        self.assertEqual(tex_headers.decode_etpf_chunk_pages(0, 0), ())

        # Single-chunk code in low 5 bits of meta (code 11 -> 16 pages).
        self.assertEqual(tex_headers.decode_etpf_chunk_pages(11, 0), (16,))

        # Multi-chunk codes spanning the 11-bit meta / extra boundary:
        # codes (6, 21, 15, 16) -> pages (6, 18, 64, 128).
        val20 = 6 | (21 << 5) | (15 << 10) | (16 << 15)
        meta = val20 & 0x7FF
        extra_low = val20 >> 11
        # Upper 13 bits of extra carry (align_group=2, pad_pages=9); reserved bits 9..18 = 0.
        extra = extra_low | (2 << 19) | (9 << 21)
        self.assertEqual(tex_headers.decode_etpf_chunk_pages(meta, extra), (6, 18, 64, 128))
        self.assertEqual(tex_headers.etpf_streaming_padding(extra), (2, 9))
        self.assertEqual(tex_headers.etpf_streaming_padding(0), (0, 0))

        # Negative controls: unknown 5-bit code (9, 17, 31 not in table).
        for bad_code in (9, 17, 25, 31):
            with self.subTest(bad_code=bad_code):
                self.assertIsNone(tex_headers.decode_etpf_chunk_pages(bad_code, 0))

        # Negative controls: non-zero code after zero terminator.
        self.assertIsNone(tex_headers.decode_etpf_chunk_pages(1 << 5, 0))
        self.assertIsNone(tex_headers.decode_etpf_chunk_pages(1 | (2 << 10), 0))

        # Negative control: non-zero reserved bits (extra >> 9) & 0x3FF.
        for bit in (9, 13, 18):
            with self.subTest(reserved_bit=bit):
                self.assertIsNone(tex_headers.decode_etpf_chunk_pages(1, 1 << bit))

        # Negative controls: out-of-range or non-integer inputs.
        for bad_meta, bad_extra in [(-1, 0), (0x800, 0), (0, -1), (0, 1 << 32), ("1", 0), (0, None)]:
            with self.subTest(meta=bad_meta, extra=bad_extra):
                self.assertIsNone(tex_headers.decode_etpf_chunk_pages(bad_meta, bad_extra))

    def test_array_surface_layout(self):
        lay = tex_headers.array_layout_for_dxgi(256, 256, 3, 83, 6, tile=tex_headers.COMPACT_TILE_64K)
        self.assertTrue(lay["mip_major"])
        self.assertEqual(lay["depth"], 3)
        self.assertEqual(lay["tile"], 65536)
        self.assertEqual(lay["first_tail"], 1)
        self.assertEqual(lay["slice_size"], 131072)
        self.assertEqual(lay["size"], 3 * lay["slice_size"])
        # Packed tail sits at offset 0 with slice_stride == tile; standalone mip 0 follows depth * tile.
        self.assertEqual(lay["mips"][1]["offset"], 0)
        self.assertEqual(lay["mips"][1]["slice_stride"], 65536)
        self.assertEqual(lay["mips"][0]["offset"], 3 * 65536)
        self.assertEqual(lay["mips"][0]["slice_stride"], lay["mips"][0]["region"])

        # Negative controls: depth <= 0 or unknown format.
        for bad_depth in (0, -1, "3"):
            with self.subTest(depth=bad_depth):
                with self.assertRaises(ValueError):
                    tex_headers.array_surface_layout(256, 256, bad_depth, 6, 16, 4)
        with self.assertRaises(ValueError):
            tex_headers.array_layout_for_dxgi(256, 256, 2, 9999, 6)

    def test_cubemap_surface_layout(self):
        # non_tail=0 (mode 18): single 6-face snapshot at offset 0.
        lay0 = tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=0)
        self.assertEqual(lay0["data_offset"], 0)
        self.assertEqual(lay0["face_size"], 12288)
        self.assertEqual(lay0["full_bytes"], 73728)
        self.assertEqual(lay0["padded_full_bytes"], 73728)
        self.assertEqual(lay0["size"], 73728)
        self.assertEqual(lay0["snapshots"], 1)
        self.assertEqual(lay0["depth"], 6)
        # Explicit matching 1-chunk list is accepted for non_tail=0.
        lay0_pages = tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=0, chunk_pages=(18,))
        self.assertEqual(lay0_pages["size"], 73728)

        # Streaming non_tail=1 below 36-page 64 KiB padding threshold: 18 pages stays 18 pages.
        lay1_small = tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=1, chunk_pages=(6, 18))
        self.assertEqual(lay1_small["full_bytes"], 18 * 4096)
        self.assertEqual(lay1_small["padded_full_bytes"], 18 * 4096)
        self.assertEqual(lay1_small["data_offset"], 6 * 4096)
        self.assertEqual(lay1_small["size"], 24 * 4096)
        self.assertEqual(lay1_small["snapshots"], 2)

        # Streaming non_tail=2 at exact 36-page threshold (128x128 BC7 -> 6 * 24576 = 36 * 4096)
        # rounds final chunk up to 64 KiB boundary (48 pages = 196608 bytes).
        lay2_thresh = tex_headers.cubemap_layout_for_dxgi(128, 128, 98, 6, non_tail=2, chunk_pages=(6, 12, 48))
        self.assertEqual(lay2_thresh["full_bytes"], 36 * 4096)
        self.assertEqual(lay2_thresh["padded_full_bytes"], 48 * 4096)
        self.assertEqual(lay2_thresh["data_offset"], 18 * 4096)
        self.assertEqual(lay2_thresh["size"], 66 * 4096)
        self.assertEqual(lay2_thresh["snapshots"], 3)

        # Streaming non_tail=3 above threshold (256x256 BC1 -> 6 * 45056 = 66 * 4096 -> 80 * 4096).
        lay3 = tex_headers.cubemap_layout_for_dxgi(256, 256, 71, 7, non_tail=3, chunk_pages=(6, 12, 24, 80))
        self.assertEqual(lay3["full_bytes"], 66 * 4096)
        self.assertEqual(lay3["padded_full_bytes"], 80 * 4096)
        self.assertEqual(lay3["data_offset"], 42 * 4096)
        self.assertEqual(lay3["size"], 122 * 4096)

        # Negative controls: bad non_tail, wrong len(chunk_pages), final chunk size mismatch, bad format.
        for bad_nt in (-1, 4, "1"):
            with self.subTest(non_tail=bad_nt):
                with self.assertRaises(ValueError):
                    tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=bad_nt)
        with self.assertRaises(ValueError):
            tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=1, chunk_pages=(18,))
        with self.assertRaises(ValueError):
            tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=0, chunk_pages=(6, 18))
        with self.assertRaises(ValueError):
            tex_headers.cubemap_layout_for_dxgi(128, 128, 71, 6, non_tail=1, chunk_pages=(6, 16))
        with self.assertRaises(ValueError):
            tex_headers.cubemap_layout_for_dxgi(128, 128, 98, 6, non_tail=2, chunk_pages=(6, 12, 36))
        with self.assertRaises(ValueError):
            tex_headers.cubemap_layout_for_dxgi(128, 128, 9999, 6, non_tail=0)

    def test_volume_surface_layout(self):
        v1 = tex_headers.volume_layout_for_dxgi(256, 32, 8, 54)
        self.assertEqual(v1["size"], 262144)
        self.assertEqual((v1["block_d"], v1["padded_d"]), (16, 16))

        v2 = tex_headers.volume_layout_for_dxgi(32, 32, 32, 10)
        self.assertEqual(v2["size"], 262144)
        self.assertEqual((v2["block_d"], v2["padded_d"]), (8, 32))

        v3 = tex_headers.volume_layout_for_dxgi(32, 64, 32, 10)
        self.assertEqual(v3["size"], 524288)
        self.assertEqual((v3["block_d"], v3["padded_d"]), (8, 32))

        # Negative controls: mips != 1, BC format, invalid dims, invalid element size, unknown format.
        for bad_mips in (0, 2):
            with self.subTest(mips=bad_mips):
                with self.assertRaises(ValueError):
                    tex_headers.volume_layout_for_dxgi(32, 32, 32, 10, mips=bad_mips)
        with self.assertRaises(ValueError):
            tex_headers.volume_layout_for_dxgi(32, 32, 8, 71)
        with self.assertRaises(ValueError):
            tex_headers.volume_layout_for_dxgi(32, 32, 8, 9999)
        for bad_args in [(0, 32, 8, 1, 2), (32, -1, 8, 1, 2), (32, 32, 0, 1, 2), (32, 32, 8, 1, 3)]:
            with self.subTest(bad_args=bad_args):
                with self.assertRaises(ValueError):
                    tex_headers.volume_surface_layout(*bad_args)


class TestDDSHeaderSynthesis(unittest.TestCase):
    def test_dds_header_2d_basic(self):
        hdr = tex_headers.dds_header(512, 256, depth=1, mips=1, fmt=28, is_cube=False, linear_size=512 * 256 * 4)
        self.assertEqual(len(hdr), 148)
        self.assertEqual(hdr[:4], b"DDS ")

        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["magic"], b"DDS ")
        self.assertEqual(info["width"], 512)
        self.assertEqual(info["height"], 256)
        self.assertEqual(info["depth"], 0)
        self.assertEqual(info["mips"], 1)
        self.assertEqual(info["fourcc"], b"DX10")
        self.assertEqual(info["dxgi_format"], 28)
        self.assertEqual(info["resource_dimension"], tex_headers.D3D10_RESOURCE_DIMENSION_TEXTURE2D)
        self.assertEqual(info["misc_flags"], 0)
        self.assertEqual(info["array_size"], 1)

    def test_dds_header_with_mips(self):
        hdr = tex_headers.dds_header(1024, 1024, depth=1, mips=11, fmt=71)
        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["width"], 1024)
        self.assertEqual(info["height"], 1024)
        self.assertEqual(info["mips"], 11)
        self.assertTrue(info["flags"] & tex_headers.DDSD_MIPMAPCOUNT)
        self.assertTrue(info["caps"] & tex_headers.DDSCAPS_MIPMAP)
        self.assertTrue(info["caps"] & tex_headers.DDSCAPS_COMPLEX)
        self.assertEqual(info["dxgi_format"], 71)

    def test_dds_header_3d_volume(self):
        hdr = tex_headers.dds_header(64, 64, depth=16, mips=1, fmt=28)
        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["depth"], 16)
        self.assertTrue(info["flags"] & tex_headers.DDSD_DEPTH)
        self.assertTrue(info["caps2"] & tex_headers.DDSCAPS2_VOLUME)
        self.assertEqual(info["resource_dimension"], tex_headers.D3D10_RESOURCE_DIMENSION_TEXTURE3D)

    def test_dds_header_cubemap(self):
        hdr = tex_headers.dds_header(256, 256, depth=1, mips=1, fmt=28, is_cube=True)
        info = tex_headers.parse_dds_header(hdr)
        self.assertTrue(info["caps2"] & tex_headers.DDSCAPS2_CUBEMAP_ALLFACES)
        self.assertEqual(info["misc_flags"], tex_headers.D3D10_RESOURCE_MISC_TEXTURECUBE)
        self.assertEqual(info["array_size"], 1)

    def test_dds_cubemap_array_size_matches_payload(self):
        # DDS_HEADER_DXT10.arraySize counts cubes, each containing six faces:
        # https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
        # Check the serialized field directly, independently of our header parser.
        for cubes in (1, 2, 6):
            with self.subTest(cubes=cubes):
                # One RGBA texel per face, with distinct values for every face.
                payload = b"".join(bytes((face, 0, 0, 255)) for face in range(cubes * 6))
                dds = tex_headers.dds_bytes(1, 1, fmt=28, payload=payload,
                                            is_cube=True, array_size=cubes)
                fmt, dimension, misc, array_size, _ = struct.unpack_from("<IIIII", dds, 128)
                self.assertEqual((fmt, dimension, misc), (28, 3, 4))
                self.assertEqual(array_size, cubes)
                self.assertEqual(len(dds) - 148, array_size * 6 * 4)
                self.assertEqual(dds[148:], payload)

    def test_dds_bytes_describes_the_top_subresource(self) -> None:
        cases = (
            # 6x5 RGBA8 top level (120 bytes) and 3x2 next level (24 bytes).
            (6, 5, 28, bytes(range(144)), 0x8, 24),
            # 9x5 BC1 top level (six blocks) and 4x2 next level (one block).
            (9, 5, 71, bytes(range(56)), 0x80000, 48),
        )
        for width, height, fmt, payload, expected_flags, expected_size in cases:
            with self.subTest(dxgi_format=fmt):
                dds = tex_headers.dds_bytes(width, height, fmt, payload, mips=2)
                self.assertEqual(dds[148:], payload)
                _, flags, got_height, got_width, pitch_or_size, _, mips = struct.unpack_from("<7I", dds, 4)
                self.assertEqual((got_width, got_height, mips), (width, height, 2))
                self.assertEqual((flags & (0x8 | 0x80000), pitch_or_size), (expected_flags, expected_size))

    def test_dds_header_2d_array(self):
        hdr = tex_headers.dds_header(256, 256, depth=1, mips=1, fmt=83, is_cube=False, array_size=5)
        info = tex_headers.parse_dds_header(hdr)
        self.assertEqual(info["resource_dimension"], tex_headers.D3D10_RESOURCE_DIMENSION_TEXTURE2D)
        self.assertEqual(info["misc_flags"], 0)
        self.assertEqual(info["array_size"], 5)
        self.assertTrue(info["caps"] & tex_headers.DDSCAPS_COMPLEX)
        self.assertEqual(info["caps2"], 0)
        blob = tex_headers.dds_bytes(16, 16, fmt=71, payload=b"\x00" * 128, array_size=4)
        self.assertEqual(tex_headers.parse_dds_header(blob)["array_size"], 4)

    def test_dds_bytes_payload(self):
        payload = b"\xAA\xBB\xCC\xDD" * 32  # Sixteen BC1 blocks at eight bytes each.
        data = tex_headers.dds_bytes(16, 16, fmt=71, payload=payload)
        self.assertEqual(len(data), 148 + len(payload))
        self.assertEqual(data[148:], payload)
        info = tex_headers.parse_dds_header(data)
        self.assertEqual(info["width"], 16)
        self.assertEqual(info["height"], 16)
        self.assertEqual(info["pitch_linear"], len(payload))

    def test_parse_corrupt_headers(self):
        # Truncated header (< 128 bytes)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(b"DDS " + b"\x00" * 50)

        # Bad magic
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(b"NOTD" + b"\x00" * 140)

        # Bad header size (not 124)
        bad_size = bytearray(b"DDS " + b"\x00" * 140)
        struct.pack_into("<I", bad_size, 4, 100)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_size))

        # Bad pixel format size (not 32)
        bad_pf = bytearray(b"DDS " + b"\x00" * 140)
        struct.pack_into("<I", bad_pf, 4, 124)
        struct.pack_into("<I", bad_pf, 76, 20)
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_pf))

        # Truncated DX10 extension (< 148 bytes when fourcc == b"DX10")
        bad_dx10 = bytearray(b"DDS " + b"\x00" * 130)
        struct.pack_into("<I", bad_dx10, 4, 124)
        struct.pack_into("<II4s", bad_dx10, 76, 32, 4, b"DX10")
        with self.assertRaises(ValueError):
            tex_headers.parse_dds_header(bytes(bad_dx10))


class TestTextureDescriptorParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carve = self.tmp.name
        self.arch = "arch_001"
        self.arch_dir = os.path.join(self.carve, self.arch)
        os.makedirs(self.arch_dir, exist_ok=True)

    def _make_synthetic_texture_blobs(self, w=512, h=256, depth=1, fmt=71, data_off=0x1000, mips=9, *, dtype: int = 0x980):
        cpu_off = 0x1000
        gpu_off = 0x2000

        # Construct CPU blob:
        # Texture descriptor layout:
        #   q[k-2] = packed u64
        #   q[k-1] = 0x980 (descriptor type)
        #   q[k]   = 0x60000000 + data_off
        #   q[k+4] = u16 width, u16 height, u16 depth, u8 b6, u8 fmt
        #   q[k+5] = flags / mips ((mips << 16) | flags)
        qwords = [0] * 32
        # Starts with zero qword, but head not 32 zeros
        qwords[0] = 0
        qwords[1] = 0x123456789ABCDEF0

        k = 10
        qwords[k - 2] = 0xCAFEBABE00000001
        qwords[k - 1] = dtype
        qwords[k] = 0x60000000 + data_off
        d_bytes = struct.pack("<HHHBB", w, h, depth, 0x42, fmt)
        qwords[k + 4] = struct.unpack("<Q", d_bytes)[0]
        qwords[k + 5] = (mips << 16) | 0x0001

        cpu_bytes = struct.pack(f"<{len(qwords)}Q", *qwords)
        gpu_bytes = b"\x00" * (data_off + 4096)

        cpu_head = cpu_bytes[:16].hex()
        gpu_head = gpu_bytes[:16].hex()

        with open(os.path.join(self.arch_dir, f"{cpu_off:012x}.bin"), "wb") as f:
            f.write(cpu_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu_off:012x}.bin"), "wb") as f:
            f.write(gpu_bytes)

        index = {
            "files": [
                {"off": cpu_off, "head": cpu_head, "size": len(cpu_bytes)},
                {"off": gpu_off, "head": gpu_head, "size": len(gpu_bytes)},
            ]
        }
        with open(os.path.join(self.arch_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        return cpu_off, gpu_off, k

    def test_scan_archive_success(self):
        cpu_off, gpu_off, k = self._make_synthetic_texture_blobs(w=1024, h=512, depth=1, fmt=71, data_off=0x2000, mips=11)
        results = tex_headers.scan_archive(self.carve, self.arch)

        self.assertEqual(len(results), 1)
        t = results[0]
        self.assertEqual(t["archive"], self.arch)
        self.assertEqual(t["cpu"], cpu_off)
        self.assertEqual(t["gpu"], gpu_off)
        self.assertEqual(t["w"], 1024)
        self.assertEqual(t["h"], 512)
        self.assertEqual(t["depth"], 1)
        self.assertEqual(t["fmt"], 71)
        self.assertEqual(t["fmt_name"], "BC1_UNORM")
        self.assertEqual(t["data_off"], 0x2000)
        self.assertEqual(t["mips"], 11)
        self.assertEqual(t["b6"], 0x42)
        self.assertEqual(t["dtype"], 0x980)
        self.assertEqual(t["packed"], 0xCAFEBABE00000001)
        self.assertEqual(t["q_index"], k)

    def test_scan_archive_utf8_bom_index(self):
        cpu_off, gpu_off, k = self._make_synthetic_texture_blobs(w=64, h=64, depth=1, fmt=71, data_off=0x1000, mips=4)
        index_path = os.path.join(self.arch_dir, "index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            raw = f.read()
        with open(index_path, "w", encoding="utf-8-sig") as f:
            f.write(raw)

        results = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(results), 1)
        t = results[0]
        self.assertEqual(t["cpu"], cpu_off)
        self.assertEqual(t["gpu"], gpu_off)
        self.assertEqual(t["w"], 64)
        self.assertEqual(t["h"], 64)
        self.assertEqual(t["fmt"], 71)
        self.assertEqual(t["q_index"], k)

    def test_scan_archive_unaligned_pointer_rejected(self):
        # Texture data pointer must be 256-byte aligned (int(q[k]) & 0xFF == 0)
        self._make_synthetic_texture_blobs(data_off=0x1001)
        results = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(results), 0)

    def test_cli_indexes_a_small_texture_with_its_gpu_location(self) -> None:
        cpu_off, gpu_off, index = self._make_synthetic_texture_blobs(w=2, h=3, fmt=28, mips=2)
        output = os.path.join(self.carve, "textures.json")
        with mock.patch.object(sys, "argv", ["tex_headers.py", self.carve, "--out", output]), \
                contextlib.redirect_stdout(io.StringIO()):
            tex_headers.main()
        with open(output, encoding="utf-8") as handle:
            entries = json.load(handle)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual((entry["w"], entry["h"], entry["depth"], entry["mips"]), (2, 3, 1, 2))
        self.assertEqual(entry["fmt_name"], "R8G8B8A8_UNORM")
        self.assertEqual((entry["cpu"], entry["gpu"], entry["data_off"], entry["q_index"]),
                         (cpu_off, gpu_off, 0x1000, index))

    def test_narrow_texture_requires_a_known_type_and_valid_mips(self) -> None:
        for dtype, mips in ((0x900, 1), (0x980, 0), (0x980, 3)):
            with self.subTest(dtype=dtype, mips=mips):
                self._make_synthetic_texture_blobs(w=1, h=2, fmt=28, mips=mips, dtype=dtype)
                self.assertEqual(tex_headers.scan_archive(self.carve, self.arch), [])

    def test_scan_archive_invalid_dimensions_rejected(self):
        # Zero width
        self._make_synthetic_texture_blobs(w=0, h=256)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

        # Height > 16384
        self._make_synthetic_texture_blobs(w=256, h=20000)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

        # Depth > 2048
        self._make_synthetic_texture_blobs(w=256, h=256, depth=3000)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

    def test_scan_archive_unknown_format_rejected(self):
        # Format not in DXGI
        self._make_synthetic_texture_blobs(fmt=254)
        self.assertEqual(len(tex_headers.scan_archive(self.carve, self.arch)), 0)

    def test_scan_archive_gpu_chain_and_skips(self):
        cpu_off = 0x1000
        gpu1_off = 0x2000
        gpu2_off = 0x3000
        bink_off = 0x4000

        qwords = [0] * 32
        qwords[0] = 0
        qwords[1] = 0x123456789ABCDEF0
        k = 10
        qwords[k - 2] = 1
        qwords[k - 1] = 0x980
        qwords[k] = 0x60000000 + 0x1000
        qwords[k + 4] = struct.unpack("<Q", struct.pack("<HHHBB", 256, 256, 1, 0, 71))[0]
        qwords[k + 5] = (9 << 16) | 1

        cpu_bytes = struct.pack(f"<{len(qwords)}Q", *qwords)
        gpu1_bytes = b"\x11" * 4096
        gpu2_bytes = b"\x22" * 4096
        bink_bytes = b"\x33" * 4096

        with open(os.path.join(self.arch_dir, f"{cpu_off:012x}.bin"), "wb") as f:
            f.write(cpu_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu1_off:012x}.bin"), "wb") as f:
            f.write(gpu1_bytes)
        with open(os.path.join(self.arch_dir, f"{gpu2_off:012x}.bin"), "wb") as f:
            f.write(gpu2_bytes)
        with open(os.path.join(self.arch_dir, f"{bink_off:012x}.bin"), "wb") as f:
            f.write(bink_bytes)

        index = {
            "files": [
                {"off": cpu_off, "head": cpu_bytes[:16].hex(), "size": len(cpu_bytes)},
                {"off": gpu1_off, "head": gpu1_bytes[:16].hex(), "size": len(gpu1_bytes)},
                {"off": gpu2_off, "head": gpu2_bytes[:16].hex(), "size": len(gpu2_bytes)},
                {"off": bink_off, "head": "4240000000000000", "size": len(bink_bytes), "type": "bink2"},
            ]
        }
        with open(os.path.join(self.arch_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        results = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(results), 1)
        # GPU chain should contain gpu1 and gpu2, stopping before bink
        chain = results[0]["gpu_chain"]
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0], [gpu1_off, len(gpu1_bytes)])
        self.assertEqual(chain[1], [gpu2_off, len(gpu2_bytes)])

    def test_main_cli(self):
        self._make_synthetic_texture_blobs(w=512, h=512, fmt=71)
        out_json = os.path.join(self.tmp.name, "out_textures.json")
        argv = ["tex_headers.py", self.carve, "--out", out_json]
        out_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out_buf):
            tex_headers.main()

        self.assertTrue(os.path.isfile(out_json))
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["w"], 512)
        self.assertEqual(data[0]["h"], 512)
        self.assertEqual(data[0]["fmt_name"], "BC1_UNORM")

    def test_stdlib_qword_scanner_edge_cases_and_parity(self):
        # Empty / short (<8 bytes) buffers yield no hits.
        q_empty, hits_empty = tex_headers._find_gpu_ptr_qwords_stdlib(b"\x00\x01\x02")
        self.assertEqual(len(q_empty), 0)
        self.assertEqual(hits_empty, [])

        # Matches only [0x60000000, 0x6FFFFFFF]; ignores 0x5FFFFFFF, 0x70000000, and high-32-bit 0x160000000.
        vals = [
            0x5FFFFFFF,
            0x60000000,
            0x60001000,
            0x6FFFFFFF,
            0x70000000,
            0x0000000160000000,
            0xFFFFFFFF60000000,
        ]
        # Include trailing unaligned bytes (<8) to verify truncation matches numpy path.
        blob = struct.pack(f"<{len(vals)}Q", *vals) + b"\xAA\xBB\xCC"
        q_std, hits_std = tex_headers._find_gpu_ptr_qwords_stdlib(blob)
        self.assertEqual(len(q_std), len(vals))
        self.assertEqual(hits_std, [1, 2, 3])

        # Verify scan_archive works when numpy is forced unavailable.
        self._make_synthetic_texture_blobs(w=256, h=128, fmt=77, data_off=0x3000, mips=8)
        with mock.patch.object(tex_headers, "_find_gpu_ptr_qwords", tex_headers._find_gpu_ptr_qwords_stdlib):
            res = tex_headers.scan_archive(self.carve, self.arch)
        self.assertEqual(len(res), 1)
        self.assertEqual((res[0]["w"], res[0]["h"], res[0]["fmt_name"]), (256, 128, "BC3_UNORM"))

        # Verify exact parity with the active _find_gpu_ptr_qwords path (numpy if installed, stdlib otherwise).
        q_active, hits_active = tex_headers._find_gpu_ptr_qwords(blob)
        self.assertEqual([int(v) for v in q_active], [int(v) for v in q_std])
        self.assertEqual([int(k) for k in hits_active], hits_std)


class CliInputTest(unittest.TestCase):
    """A missing carve folder or an unknown --archives name ended in a FileNotFoundError traceback."""

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(TOOLS_DIR, "tex_headers.py"), *args],
                              capture_output=True, text=True, errors="replace", timeout=60)

    def test_out_in_a_missing_folder_is_created(self):
        # Before: main() opened --out directly after the scan, so this ended in a FileNotFoundError traceback.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "carve", "a"))
            with open(os.path.join(tmp, "carve", "a", "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            out = os.path.join(tmp, "no", "such", "textures.json")
            p = self.run_tool(os.path.join(tmp, "carve"), "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])

    def test_bad_carve_folders_are_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "carve", "a"))
            with open(os.path.join(tmp, "carve", "a", "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            os.makedirs(os.path.join(tmp, "empty"))
            out = os.path.join(tmp, "t.json")
            cases = (((os.path.join(tmp, "missing"),), "is not a folder"),
                     ((os.path.join(tmp, "empty"),), "no carved archive"),
                     ((os.path.join(tmp, "carve"), "--archives", "a,nope"), "no index.json for nope"))
            for args, message in cases:
                with self.subTest(args=args):
                    p = self.run_tool(*args, "--out", out)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn(message, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))
            p = self.run_tool(os.path.join(tmp, "carve"), "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("0 textures", p.stdout)

    def test_malformed_index_is_one_error_line(self):
        # An index.json of the wrong shape ended in a KeyError/TypeError traceback after the carve had finished;
        # gamedata_index.py already reports this as one ERROR line with exit 2.
        cases = {"no files key": {}, "files not a list": {"files": "oops"}, "files null": {"files": None},
                 "row not an object": {"files": [1, 2]}, "row without head": {"files": [{"off": 0}, {"off": 16}]}}
        for label, index in cases.items():
            with self.subTest(index=label), tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "carve", "a"))
                with open(os.path.join(tmp, "carve", "a", "index.json"), "w", encoding="utf-8") as f:
                    json.dump(index, f)
                out = os.path.join(tmp, "t.json")
                p = self.run_tool(os.path.join(tmp, "carve"), "--out", out)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertIn("ERROR: cannot read", p.stderr)
                self.assertIn("index.json", p.stderr)
                self.assertFalse(os.path.exists(out))

    def test_out_overwriting_input_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "carve", "a"))
            manifest = os.path.join(tmp, "carve", "a", "index.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            p = self.run_tool(os.path.join(tmp, "carve"), "--out", manifest)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("output would overwrite an input manifest", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            with open(manifest, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"files": []})

    def test_output_parent_creation_failure_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            archive = os.path.join(carve, "a")
            os.makedirs(archive)
            with open(os.path.join(archive, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"files": []}, f)
            blocker = os.path.join(tmp, "not-a-directory")
            with open(blocker, "w", encoding="utf-8", newline="\n") as f:
                f.write("existing file")
            result = self.run_tool(carve, "--out", os.path.join(blocker, "out.json"))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("ERROR: cannot write", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_out_that_is_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "carve", "a"))
            with open(os.path.join(tmp, "carve", "a", "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": []}, f)
            p = self.run_tool(os.path.join(tmp, "carve"), "--out", tmp)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("is a directory", p.stderr)
            self.assertNotIn("Traceback", p.stderr)


class TestTextureDictionaryParsing(unittest.TestCase):
    def _create_synthetic_page(self, name0=b"waterfog2", name1=b"waterfog", fmt=95, w=128, h=128, mips=1, b6=7):
        buf = bytearray(400)
        names_chunk = name0 + b"\x00" + name1 + b"\x00"
        buf[0x140:0x140 + len(names_chunk)] = names_chunk
        struct.pack_into("<QQ", buf, 0x170, 0x50000040, 0x50000090)
        struct.pack_into("<II", buf, 0x180, 0x9b9b31df, 0xf1389e16)

        # desc 0
        struct.pack_into("<Q", buf, 0x40 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x40 + 0x18, 0x60000000)
        struct.pack_into("<HHHBB", buf, 0x40 + 0x38, w, h, 1, b6, fmt)
        struct.pack_into("<Q", buf, 0x40 + 0x40, (mips << 16) | 0x1)

        # desc 1
        struct.pack_into("<Q", buf, 0x90 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x90 + 0x18, 0x60004000)
        struct.pack_into("<HHHBB", buf, 0x90 + 0x38, w, h, 1, b6, fmt)
        struct.pack_into("<Q", buf, 0x90 + 0x40, (mips << 16) | 0x1)
        return bytes(buf)

    def test_parse_valid_texture_dict_page(self):
        data = self._create_synthetic_page(mips=1, b6=7)
        res = tex_headers.parse_texture_dict_page(data)
        self.assertEqual(len(res), 2)

        t0 = res[0]
        self.assertEqual(t0["name"], "waterfog2")
        self.assertEqual(t0["hash"], 0x9b9b31df)
        self.assertEqual(t0["width"], 128)
        self.assertEqual(t0["height"], 128)
        self.assertEqual(t0["b6"], 7)
        self.assertEqual(t0["mips"], 1)
        self.assertEqual(t0["format"], 95)
        self.assertEqual(t0["format_name"], "BC6H_UF16")
        self.assertEqual(t0["data_offset"], 0)
        self.assertEqual(t0["dtype"], 0x980)

        t1 = res[1]
        self.assertEqual(t1["name"], "waterfog")
        self.assertEqual(t1["hash"], 0xf1389e16)
        self.assertEqual(t1["width"], 128)
        self.assertEqual(t1["height"], 128)
        self.assertEqual(t1["b6"], 7)
        self.assertEqual(t1["mips"], 1)
        self.assertEqual(t1["format"], 95)
        self.assertEqual(t1["format_name"], "BC6H_UF16")
        self.assertEqual(t1["data_offset"], 0x4000)

    def test_mips_and_b6_offsets_differ(self):
        # b6 at +0x3e is 42, mips at +0x42 (flags >> 16) is 5
        data = self._create_synthetic_page(mips=5, b6=42)
        res = tex_headers.parse_texture_dict_page(data)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["b6"], 42)
        self.assertEqual(res[0]["mips"], 5)
        self.assertEqual(res[1]["b6"], 42)
        self.assertEqual(res[1]["mips"], 5)

    def test_truncated_buffer_raises_value_error(self):
        with self.assertRaises(ValueError):
            tex_headers.parse_texture_dict_page(b"\x00" * 399)

    def test_all_zeros_returns_empty_list(self):
        # Null pointers return no texture records
        res = tex_headers.parse_texture_dict_page(b"\x00" * 400)
        self.assertEqual(res, [])

    def test_invalid_pointer_tag_skipped(self):
        buf = bytearray(self._create_synthetic_page())
        # Pointer without 0x5... tag (e.g. 0x10000040)
        struct.pack_into("<Q", buf, 0x170, 0x10000040)
        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "waterfog")

    def test_reversed_names_order_matched_by_joaat_hash(self):
        # Even if name1 is placed first in the string table, joaat matching maps each name to its hash
        buf = bytearray(self._create_synthetic_page(name0=b"waterfog", name1=b"waterfog2"))
        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], "waterfog2")
        self.assertEqual(res[0]["hash"], 0x9b9b31df)
        self.assertEqual(res[1]["name"], "waterfog")
        self.assertEqual(res[1]["hash"], 0xf1389e16)

    def test_missing_first_name_does_not_shift_second(self):
        # Only waterfog is in the names table, waterfog2 is omitted
        buf = bytearray(400)
        buf[0x140:0x140 + len(b"waterfog\x00")] = b"waterfog\x00"
        struct.pack_into("<QQ", buf, 0x170, 0x50000040, 0x50000090)
        struct.pack_into("<II", buf, 0x180, 0x9b9b31df, 0xf1389e16)
        struct.pack_into("<Q", buf, 0x40 + 0x10, 0x980)
        struct.pack_into("<HHHBB", buf, 0x40 + 0x38, 128, 128, 1, 0, 95)
        struct.pack_into("<Q", buf, 0x40 + 0x40, 1 << 16)
        struct.pack_into("<Q", buf, 0x90 + 0x10, 0x980)
        struct.pack_into("<HHHBB", buf, 0x90 + 0x38, 128, 128, 1, 0, 95)
        struct.pack_into("<Q", buf, 0x90 + 0x40, 1 << 16)

        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 2)
        # First texture has hash 0x9b9b31df (waterfog2), falls back to hex name
        self.assertEqual(res[0]["name"], f"tex_{0x9b9b31df:08x}")
        # Second texture has hash 0xf1389e16, gets correctly matched to "waterfog"
        self.assertEqual(res[1]["name"], "waterfog")

    def test_missing_names_fallback_to_hash(self):
        buf = bytearray(self._create_synthetic_page())
        buf[0x140:0x170] = b"\x00" * 0x30
        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], f"tex_{0x9b9b31df:08x}")
        self.assertEqual(res[1]["name"], f"tex_{0xf1389e16:08x}")

    def test_out_of_bounds_pointer_skipped(self):
        buf = bytearray(self._create_synthetic_page())
        struct.pack_into("<Q", buf, 0x178, 0x50000300)
        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "waterfog2")

    def test_tag_validation_rejects_synthetic_0x150000000_offset_zero(self):
        buf = bytearray(self._create_synthetic_page())
        # 0x150000000 has high nibble 5 if masked with & 0xF, but high tag is 0x15 != 5
        struct.pack_into("<Q", buf, 0x170, 0x150000000)
        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "waterfog")

    def test_shifted_table_mixed_dimensions_bc3(self):
        # 400-byte stream with longer names in 0x140..0x178 shifting table by 8 bytes:
        # pointers at +0x178 and +0x180, hashes at +0x188 and +0x18C.
        # mixed dimensions: 64x64 and 512x512 with format 77 (BC3_UNORM).
        buf = bytearray(400)
        names = b"water_deep_caustic\x00water_foam\x00"
        buf[0x140:0x140 + len(names)] = names

        h0 = tex_headers.joaat("water_deep_caustic")
        h1 = tex_headers.joaat("water_foam")
        struct.pack_into("<QQ", buf, 0x178, 0x50000040, 0x50000090)
        struct.pack_into("<II", buf, 0x188, h0, h1)

        # desc 0: 64x64 format 77 (BC3_UNORM)
        struct.pack_into("<Q", buf, 0x40 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x40 + 0x18, 0x60000000)
        struct.pack_into("<HHHBB", buf, 0x40 + 0x38, 64, 64, 1, 0, 77)
        struct.pack_into("<Q", buf, 0x40 + 0x40, (1 << 16) | 0x1)

        # desc 1: 512x512 format 77 (BC3_UNORM)
        struct.pack_into("<Q", buf, 0x90 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x90 + 0x18, 0x60001000)
        struct.pack_into("<HHHBB", buf, 0x90 + 0x38, 512, 512, 1, 0, 77)
        struct.pack_into("<Q", buf, 0x90 + 0x40, (1 << 16) | 0x1)

        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], "water_deep_caustic")
        self.assertEqual(res[0]["hash"], h0)
        self.assertEqual(res[0]["width"], 64)
        self.assertEqual(res[0]["height"], 64)
        self.assertEqual(res[0]["format"], 77)
        self.assertEqual(res[0]["format_name"], "BC3_UNORM")

        self.assertEqual(res[1]["name"], "water_foam")
        self.assertEqual(res[1]["hash"], h1)
        self.assertEqual(res[1]["width"], 512)
        self.assertEqual(res[1]["height"], 512)
        self.assertEqual(res[1]["format"], 77)
        self.assertEqual(res[1]["format_name"], "BC3_UNORM")

    def test_header_pointers_used_when_present(self):
        # Header with pointers explicitly defined in pgDictionary header at +0x20 and +0x30
        buf = bytearray(400)
        names = b"water_deep_caustic\x00water_foam\x00"
        buf[0x140:0x140 + len(names)] = names

        h0 = tex_headers.joaat("water_deep_caustic")
        h1 = tex_headers.joaat("water_foam")
        struct.pack_into("<Q", buf, 0x20, 0x50000188)  # hashes at 0x188
        struct.pack_into("<Q", buf, 0x30, 0x50000178)  # pointers at 0x178
        struct.pack_into("<H", buf, 0x38, 2)           # count = 2

        struct.pack_into("<QQ", buf, 0x178, 0x50000040, 0x50000090)
        struct.pack_into("<II", buf, 0x188, h0, h1)

        struct.pack_into("<Q", buf, 0x40 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x40 + 0x18, 0x60000000)
        struct.pack_into("<HHHBB", buf, 0x40 + 0x38, 64, 64, 1, 0, 77)
        struct.pack_into("<Q", buf, 0x40 + 0x40, (1 << 16) | 0x1)

        struct.pack_into("<Q", buf, 0x90 + 0x10, 0x980)
        struct.pack_into("<Q", buf, 0x90 + 0x18, 0x60001000)
        struct.pack_into("<HHHBB", buf, 0x90 + 0x38, 512, 512, 1, 0, 77)
        struct.pack_into("<Q", buf, 0x90 + 0x40, (1 << 16) | 0x1)

        res = tex_headers.parse_texture_dict_page(bytes(buf))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["width"], 64)
        self.assertEqual(res[1]["width"], 512)


if __name__ == "__main__":
    unittest.main(verbosity=2)
