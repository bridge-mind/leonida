"""Tests for tools/dds_png.py, built from synthetic DDS buffers (no game data).

    python -I tests/test_dds_png.py
"""
import contextlib
import importlib.util
import io
import os
import struct
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()

DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40


def dds_buffer(width, height, payload=b"", *, flags=0, fourcc=b"\x00\x00\x00\x00",
               bit_count=0, masks=(0, 0, 0, 0), mips=0, dxgi=None, pitch=None, alpha_mode=0):
    """A 128-byte DDS header, a DX10 header when dxgi is set, then the payload."""
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    if pitch is not None:
        struct.pack_into("<I", header, 8, 0x8)  # DDSD_PITCH
        struct.pack_into("<I", header, 20, pitch)
    struct.pack_into("<I", header, 28, mips)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, flags)
    header[84:88] = fourcc
    struct.pack_into("<I", header, 88, bit_count)
    struct.pack_into("<IIII", header, 92, *masks)
    raw = bytes(header)
    if dxgi is not None:
        raw += struct.pack("<IIIII", dxgi, 3, 0, 1, alpha_mode)
    return raw + payload


def bc1_block(c0, c1, rows):
    """One BC1 block; rows holds the four per-row index bytes."""
    return struct.pack("<HHI", c0, c1, int.from_bytes(bytes(rows), "little"))


def bc3_block(alpha0, alpha1, alpha_indices, c0=0xFFFF, c1=0x0000, color_bits=0):
    """One BC3 block: explicit alpha endpoints plus 48 index bits, then the BC1 part."""
    alpha = struct.pack("<BB", alpha0, alpha1) + alpha_indices.to_bytes(6, "little")
    return alpha + struct.pack("<HHI", c0, c1, color_bits)


def bc4_block(v0, v1, indices, base=0):
    """One BC4 block starting at struct offset base (BC5 stacks two of them)."""
    return struct.pack("<BB", v0, v1) + indices.to_bytes(6, "little")


# BC7 vectors are hand-assembled from the Microsoft BC7 format mode reference and
# the Khronos BPTC specification. Expected values are worked out by hand from the
# spec formulas (endpoint MSB replication, weights, (a*(64-w)+b*w+32)>>6); the
# 0-to-255 ramps below are those interpolations for each weight table.
RAMP2 = (0, 84, 171, 255)
RAMP3 = (0, 36, 72, 108, 147, 183, 219, 255)
RAMP4 = (0, 16, 36, 52, 68, 84, 104, 120, 135, 151, 171, 187, 203, 219, 239, 255)


def bc7_block(*fields):
    """Pack (value, bit_width) fields LSB-first into one 128-bit BC7 block."""
    packed = 0
    pos = 0
    for value, width in fields:
        assert 0 <= value < (1 << width), (value, width)
        packed |= value << pos
        pos += width
    assert pos == 128, pos
    return packed.to_bytes(16, "little")


def bc7_mode(mode):
    """The mode field: `mode` zero bits, then a one bit."""
    return (1 << mode, mode + 1)


def bc7_values(values, width):
    return [(v, width) for v in values]


def bc7_indices(indices, width, anchors):
    """Per-pixel index fields; anchor pixels drop their implicit zero MSB."""
    return [(v, width - 1 if p in anchors else width) for p, v in enumerate(indices)]


def gray(v, a=255):
    return (v, v, v, a)


# Mode 0, partition 0 (subsets 0011/0011/0221/2222, anchors 0, 3, 15), 4-bit RGB
# endpoints plus one p-bit each. Subset 0 is a black-to-white ramp; subsets 1 and
# 2 are flat (8, 4, 2) with p-bit 1 -> 5-bit (17, 9, 5) and p-bit 0 -> (16, 8, 4).
BC7_MODE0 = bc7_block(
    bc7_mode(0), (0, 4),
    *bc7_values([0, 15, 8, 8, 8, 8], 4),
    *bc7_values([0, 15, 4, 4, 4, 4], 4),
    *bc7_values([0, 15, 2, 2, 2, 2], 4),
    *bc7_values([0, 1, 1, 1, 0, 0], 1),
    *bc7_indices([3, 7, 5, 3, 1, 4, 5, 5, 6, 5, 5, 5, 5, 5, 5, 3], 3, (0, 3, 15)),
)
BC7_MODE0_PIXELS = [
    gray(108), gray(255), (140, 74, 41, 255), (140, 74, 41, 255),
    gray(36), gray(147), (140, 74, 41, 255), (140, 74, 41, 255),
    gray(219), (132, 66, 33, 255), (132, 66, 33, 255), (140, 74, 41, 255),
    (132, 66, 33, 255), (132, 66, 33, 255), (132, 66, 33, 255), (132, 66, 33, 255),
]

# Mode 1, partition 17 (subset 1 = pixels 1, 2, 3, 7; anchors 0, 2), 6-bit RGB and
# one shared p-bit per subset. Subset 0 is flat (32, 16, 8) + p 1 -> 7-bit
# (65, 33, 17) -> (131, 66, 34). Subset 1 runs red -> green with p 0: 63 -> 126 -> 253.
BC7_MODE1 = bc7_block(
    bc7_mode(1), (17, 6),
    *bc7_values([32, 32, 63, 0], 6),
    *bc7_values([16, 16, 0, 63], 6),
    *bc7_values([8, 8, 0, 0], 6),
    *bc7_values([1, 0], 1),
    *bc7_indices([2, 0, 3, 4, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6], 3, (0, 2)),
)
BC7_MODE1_PIXELS = [
    (131, 66, 34, 255), (253, 0, 0, 255), (146, 107, 0, 255), (107, 146, 0, 255),
    (131, 66, 34, 255), (131, 66, 34, 255), (131, 66, 34, 255), (0, 253, 0, 255),
] + [(131, 66, 34, 255)] * 8

# Mode 2, partition 11 (columns 0, 0, 1, 2; anchors 0, 6, 15), 5-bit RGB, no
# p-bits. Subset 0 is a gray ramp, subset 1 flat (16, 8, 4) -> (132, 66, 33),
# subset 2 runs red -> blue.
BC7_MODE2 = bc7_block(
    bc7_mode(2), (11, 6),
    *bc7_values([0, 31, 16, 16, 31, 0], 5),
    *bc7_values([0, 31, 8, 8, 0, 0], 5),
    *bc7_values([0, 31, 4, 4, 0, 31], 5),
    *bc7_indices([1, 2, 3, 0, 3, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 1], 2, (0, 6, 15)),
)
BC7_MODE2_PIXELS = [
    gray(84), gray(171), (132, 66, 33, 255), (255, 0, 0, 255),
    gray(255), gray(0), (132, 66, 33, 255), (171, 0, 84, 255),
    gray(0), gray(0), (132, 66, 33, 255), (84, 0, 171, 255),
    gray(0), gray(0), (132, 66, 33, 255), (171, 0, 84, 255),
]

# Mode 3, partition 18 (subset 1 = pixels 8, 12, 13, 14; anchors 0, 8), 7-bit RGB
# plus one p-bit per endpoint: subset 1 endpoints (64, 32, 16) become
# (128, 64, 32) with p 0 and (129, 65, 33) with p 1.
BC7_MODE3 = bc7_block(
    bc7_mode(3), (18, 6),
    *bc7_values([0, 127, 64, 64], 7),
    *bc7_values([0, 127, 32, 32], 7),
    *bc7_values([0, 127, 16, 16], 7),
    *bc7_values([0, 1, 0, 1], 1),
    *bc7_indices([1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 2, 3], 2, (0, 8)),
)
BC7_MODE3_PIXELS = [
    gray(84), gray(171), gray(255), gray(0),
    gray(0), gray(0), gray(0), gray(0),
    (128, 64, 32, 255), gray(0), gray(0), gray(0),
    (129, 65, 33, 255), (128, 64, 32, 255), (129, 65, 33, 255), gray(255),
]


def bc7_mode4(rotation, selector):
    """Mode 4: gray 5-bit colour ramp, 6-bit alpha 255 -> 0, 2-bit then 3-bit indices."""
    return bc7_block(
        bc7_mode(4), (rotation, 2), (selector, 1),
        *bc7_values([0, 31], 5), *bc7_values([0, 31], 5), *bc7_values([0, 31], 5),
        *bc7_values([63, 0], 6),
        *bc7_indices([p % 4 for p in range(16)], 2, (0,)),
        *bc7_indices([p % 8 for p in range(16)], 3, (0,)),
    )


# Mode 5, rotation 2 (swap G and A): 7-bit colour (127, 0..127, 64) -> (255, ramp,
# 129), 8-bit alpha 10 -> 250, colour indices by column, alpha indices by row.
BC7_MODE5 = bc7_block(
    bc7_mode(5), (2, 2),
    *bc7_values([127, 127], 7), *bc7_values([0, 127], 7), *bc7_values([64, 64], 7),
    *bc7_values([10, 250], 8),
    *bc7_indices([p % 4 for p in range(16)], 2, (0,)),
    *bc7_indices([p // 4 for p in range(16)], 2, (0,)),
)
BC7_MODE5_PIXELS = [
    (255, (10, 89, 171, 250)[p // 4], 129, RAMP2[p % 4]) for p in range(16)
]

# Mode 6: 7-bit RGBA plus p-bits 0 / 1, so endpoints are R 0 / 255, G 254 / 1,
# B 128 / 129, A 254 / 255. Pixel p uses 4-bit index p.
BC7_MODE6 = bc7_block(
    bc7_mode(6),
    *bc7_values([0, 127], 7), *bc7_values([127, 0], 7),
    *bc7_values([64, 64], 7), *bc7_values([127, 127], 7),
    *bc7_values([0, 1], 1),
    *bc7_indices(list(range(16)), 4, (0,)),
)

# Mode 7, partition 0 (columns 0, 0, 1, 1; anchors 0, 15), 5-bit RGBA plus p-bits.
# Subset 0 ramps every channel 0 -> 255; subset 1 is flat (16, 8, 4, 2) + p 1 ->
# 6-bit (33, 17, 9, 5) -> (134, 69, 36, 20), so the p-bit reaches alpha too.
BC7_MODE7 = bc7_block(
    bc7_mode(7), (0, 6),
    *bc7_values([0, 31, 16, 16], 5), *bc7_values([0, 31, 8, 8], 5),
    *bc7_values([0, 31, 4, 4], 5), *bc7_values([0, 31, 2, 2], 5),
    *bc7_values([0, 1, 1, 1], 1),
    *bc7_indices([1, 2, 3, 3, 3, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 0], 2, (0, 15)),
)
BC7_MODE7_PIXELS = [
    gray(84, 84), gray(171, 171), (134, 69, 36, 20), (134, 69, 36, 20),
    gray(255, 255), gray(0, 0), (134, 69, 36, 20), (134, 69, 36, 20),
    gray(0, 0), gray(0, 0), (134, 69, 36, 20), (134, 69, 36, 20),
    gray(0, 0), gray(0, 0), (134, 69, 36, 20), (134, 69, 36, 20),
]


def bc7_dds(width, height, payload, dxgi=98):
    return dds_buffer(width, height, payload, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=dxgi)


def pixel(rgba, width, x, y):
    start = (y * width + x) * 4
    return tuple(rgba[start:start + 4])


def pixels(rgba):
    return [tuple(rgba[i:i + 4]) for i in range(0, len(rgba), 4)]


def png_chunks(png):
    """Split a PNG into (type, payload, stored_crc) tuples without trusting it."""
    assert png[:8] == dds.PNG_SIGNATURE, "missing PNG signature"
    pos = 8
    chunks = []
    while pos < len(png):
        length = struct.unpack_from(">I", png, pos)[0]
        ctype = png[pos + 4:pos + 8]
        payload = png[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", png, pos + 8 + length)[0]
        chunks.append((ctype, payload, crc))
        pos += 12 + length
    return chunks


class HeaderTests(unittest.TestCase):
    def test_legacy_dxt1_header(self):
        info = dds.parse_dds_header(dds_buffer(8, 4, flags=DDPF_FOURCC, fourcc=b"DXT1"))
        self.assertEqual(info["width"], 8)
        self.assertEqual(info["height"], 4)
        self.assertEqual(info["format"], "BC1")
        self.assertIsNone(info["dxgi_format"])
        self.assertEqual(info["mip_count"], 1)
        self.assertEqual(info["data_offset"], 128)

    def test_mip_count_defaults_to_one_when_zero(self):
        self.assertEqual(
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DXT5", mips=0))["mip_count"],
            1,
        )

    def test_mip_count_is_reported(self):
        self.assertEqual(
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DXT5", mips=3))["mip_count"],
            3,
        )

    def test_dx10_header_moves_the_payload(self):
        info = dds.parse_dds_header(dds_buffer(16, 16, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=71))
        self.assertEqual(info["format"], "BC1")
        self.assertEqual(info["dxgi_format"], 71)
        self.assertEqual(info["data_offset"], 148)

    def test_fourcc_aliases(self):
        cases = {
            b"DXT1": "BC1", b"DXT3": "BC2", b"DXT5": "BC3",
            b"ATI1": "BC4", b"BC4U": "BC4", b"BC4S": "BC4_SNORM",
            b"ATI2": "BC5", b"BC5U": "BC5", b"BC5S": "BC5_SNORM",
        }
        for fourcc, expected in cases.items():
            with self.subTest(fourcc=fourcc):
                info = dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=fourcc))
                self.assertEqual(info["format"], expected)

    def test_canonical_uncompressed_masks(self):
        cases = {
            (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000): "BGRA8",
            (0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): "BGRX8",
            (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000): "RGBA8",
            (0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000): "RGBX8",
        }
        for masks, expected in cases.items():
            with self.subTest(masks=masks):
                info = dds.parse_dds_header(
                    dds_buffer(
                        2, 2, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32, masks=masks,
                    )
                )
                self.assertEqual(info["format"], expected)

    def test_alpha_flag_removes_the_alpha_mask(self):
        info = dds.parse_dds_header(
            dds_buffer(
                2, 2, flags=DDPF_RGB, bit_count=32,
                masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
            )
        )
        self.assertEqual(info["format"], "BGRX8")

    def test_custom_bitmask_falls_back_to_rgba8(self):
        info = dds.parse_dds_header(
            dds_buffer(
                1, 1, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
                masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000),
            )
        )
        self.assertEqual(info["format"], "RGBA8")

    def test_rejects_short_buffer(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(b"DDS " + b"\x00" * 60)

    def test_rejects_bad_magic(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(b"NOPE" + b"\x00" * 124)
        self.assertIn("magic", str(ctx.exception))

    def test_rejects_zero_dimensions(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(dds_buffer(0, 4, flags=DDPF_FOURCC, fourcc=b"DXT1"))

    def test_rejects_unknown_fourcc(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"XXXX"))
        self.assertIn("FourCC", str(ctx.exception))

    def test_rejects_non_24_or_32_bit_rgb(self):
        for bit_count in (8, 16):
            with self.subTest(bit_count=bit_count):
                with self.assertRaises(dds.DdsError) as ctx:
                    dds.parse_dds_header(
                        dds_buffer(4, 4, flags=DDPF_RGB, bit_count=bit_count,
                                   masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0))
                    )
                self.assertIn("bit count", str(ctx.exception))

    def test_rejects_zero_rgb_masks(self):
        with self.assertRaises(dds.DdsError):
            dds.parse_dds_header(
                dds_buffer(4, 4, flags=DDPF_RGB, bit_count=32, masks=(0, 0, 0, 0))
            )

    def test_rejects_truncated_dx10(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(
                dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DX10")
            )
        self.assertIn("DX10", str(ctx.exception))

    def test_rejects_unknown_dxgi_format(self):
        # 94-96 are BC6H typeless/UF16/SF16, which this decoder deliberately does not claim.
        for dxgi in (94, 95, 96):
            with self.subTest(dxgi=dxgi):
                with self.assertRaises(dds.DdsError) as ctx:
                    dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=dxgi))
                self.assertIn(str(dxgi), str(ctx.exception))

    def test_bc7_dxgi_aliases(self):
        for dxgi in (97, 98, 99):
            with self.subTest(dxgi=dxgi):
                info = dds.parse_dds_header(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=dxgi))
                self.assertEqual(info["format"], "BC7")
                self.assertEqual(info["dxgi_format"], dxgi)
                self.assertEqual(info["data_offset"], 148)

    def test_rejects_non_contiguous_mask(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.parse_dds_header(
                dds_buffer(2, 2, flags=DDPF_RGB, bit_count=32,
                           masks=(0x00FF00FF, 0x0000FF00, 0x000000FF, 0))
            )
        self.assertIn("Non-contiguous", str(ctx.exception))


class UncompressedTests(unittest.TestCase):
    def test_dx10_typeless_byte_layouts(self):
        payload = bytes((0x10, 0x20, 0x30, 0x40))
        for dxgi, fmt, expected in (
            (27, "RGBA8", (0x10, 0x20, 0x30, 0x40)),
            (90, "BGRA8", (0x30, 0x20, 0x10, 0x40)),
            (92, "BGRX8", (0x30, 0x20, 0x10, 0xFF)),
        ):
            with self.subTest(dxgi=dxgi):
                data = dds_buffer(1, 1, payload, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=dxgi)
                info = dds.parse_dds_header(data)
                self.assertEqual(info["format"], fmt)
                self.assertEqual(info["dxgi_format"], dxgi)
                self.assertEqual(dds.decode_dds(data), (1, 1, bytes(expected)))

    def test_padded_rows_are_removed_before_channel_conversion(self):
        first_row = bytes((1, 2, 3, 4))
        second_row = bytes((5, 6, 7, 8))
        payload = first_row + b"\x99" * 4 + second_row
        legacy = dict(flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
                      masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000))
        dx10 = dict(flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=87)
        for header, expected in ((legacy, [(1, 2, 3, 4), (5, 6, 7, 8)]),
                                 (dx10, [(3, 2, 1, 4), (7, 6, 5, 8)])):
            with self.subTest(header=header):
                width, height, rgba = dds.decode_dds(dds_buffer(1, 2, payload, pitch=8, **header))
                self.assertEqual((width, height), (1, 2))
                self.assertEqual(pixels(rgba), expected)

    def test_pitch_with_custom_byte_masks(self):
        payload = bytes((1, 2, 3, 4)) + b"\xcc" * 4 + bytes((5, 6, 7, 8))
        data = dds_buffer(
            1, 2, payload, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32, pitch=8,
            masks=(0x0000FF00, 0x00FF0000, 0xFF000000, 0x000000FF),
        )
        self.assertEqual(dds.decode_dds(data), (1, 2, bytes((2, 3, 4, 1, 6, 7, 8, 5))))

    def test_pitch_with_r10g10b10a2_masks(self):
        first = (1023 << 20) | (512 << 10) | 255 | (3 << 30)
        second = (512 << 10) | 1023 | (1 << 30)
        payload = struct.pack("<I", first) + b"\xcc" * 4 + struct.pack("<I", second)
        data = dds_buffer(
            1, 2, payload, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32, pitch=8,
            masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000),
        )
        self.assertEqual(dds.decode_dds(data), (1, 2, bytes((255, 128, 64, 255, 0, 128, 255, 85))))

    def test_dx10_r10g10b10a2_unorm_with_padded_rows(self):
        first = 1023 | (512 << 10) | (17 << 20) | (2 << 30)
        second = 256 | (1023 << 20) | (1 << 30)
        payload = struct.pack("<I", first) + b"\xcc" * 4 + struct.pack("<I", second)
        data = dds_buffer(1, 2, payload, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=24, pitch=8)
        self.assertEqual(dds.decode_dds(data), (1, 2, bytes((255, 128, 4, 170, 64, 0, 255, 85))))

    def test_pitch_errors_are_reported(self):
        header = dict(flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=28)
        with self.assertRaisesRegex(dds.DdsError, "pitch"):
            dds.decode_dds(dds_buffer(1, 2, b"\x00" * 8, pitch=2, **header))
        with self.assertRaisesRegex(dds.DdsError, "Truncated"):
            dds.decode_dds(dds_buffer(1, 2, b"\x00" * 11, pitch=8, **header))

    def test_zero_and_unflagged_pitch_use_tightly_packed_rows(self):
        payload = bytes(range(8))
        header = dict(flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=28)
        self.assertEqual(dds.decode_dds(dds_buffer(1, 2, payload, pitch=0, **header)), (1, 2, payload))
        unflagged = bytearray(dds_buffer(1, 2, payload, **header))
        struct.pack_into("<I", unflagged, 20, 8)
        self.assertEqual(dds.decode_dds(unflagged), (1, 2, payload))

    def test_rgba8_is_copied(self):
        payload = bytes((1, 2, 3, 4, 5, 6, 7, 8))
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(1, 2, 3, 4), (5, 6, 7, 8)])

    def test_bgra8_swaps_red_and_blue(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x10, 0x20, 0x30, 0x40)), flags=DDPF_RGB | DDPF_ALPHAPIXELS,
            bit_count=32, masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(0x30, 0x20, 0x10, 0x40)])

    def test_bgrx8_forces_opaque_alpha(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x10, 0x20, 0x30, 0x00)), flags=DDPF_RGB, bit_count=32,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
        ))
        self.assertEqual(pixels(rgba), [(0x30, 0x20, 0x10, 0xFF)])

    def test_rgbx8_masks(self):
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, bytes((0x01, 0x02, 0x03, 0x00)), flags=DDPF_RGB, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000),
        ))
        self.assertEqual(pixels(rgba), [(0x01, 0x02, 0x03, 0xFF)])

    def test_r10g10b10a2_uses_the_general_bitmask_path(self):
        px = (1023 << 20) | (512 << 10) | 255 | (3 << 30)
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, struct.pack("<I", px), flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000),
        ))
        # Each channel is scaled by value * 255 / max with round-half-up.
        self.assertEqual(pixels(rgba), [(255, 128, 64, 255)])

    def test_truncated_payload(self):
        with self.assertRaises(dds.DdsError) as ctx:
            dds.decode_dds(dds_buffer(
                2, 1, b"\x00" * 7, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
                masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
            ))
        self.assertIn("Truncated", str(ctx.exception))

    def test_bytearray_input_is_accepted(self):
        buf = dds_buffer(
            1, 1, bytes((9, 8, 7, 6)), flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=32,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000),
        )
        _, _, rgba = dds.decode_dds(bytearray(buf))
        self.assertEqual(pixels(rgba), [(9, 8, 7, 6)])


class Rgb24Tests(unittest.TestCase):
    """Legacy 24-bit RGB layouts (D3DFMT_R8G8B8 stores bytes B, G, R; no alpha)."""

    def test_bgr8_header_and_opaque_decode(self):
        payload = bytes((0x10, 0x20, 0x30, 0x40, 0x50, 0x60))
        data = dds_buffer(
            2, 1, payload, flags=DDPF_RGB, bit_count=24,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000),
        )
        info = dds.parse_dds_header(data)
        self.assertEqual(info["format"], "BGR8")
        self.assertIsNone(info["dxgi_format"])
        self.assertEqual(info["data_offset"], 128)
        self.assertEqual(dds.decode_dds(data),
                         (2, 1, bytes((0x30, 0x20, 0x10, 255, 0x60, 0x50, 0x40, 255))))

    def test_rgb8_byte_order(self):
        payload = bytes((0x10, 0x20, 0x30, 0x40, 0x50, 0x60))
        data = dds_buffer(
            2, 1, payload, flags=DDPF_RGB, bit_count=24,
            masks=(0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000),
        )
        self.assertEqual(dds.parse_dds_header(data)["format"], "RGB8")
        self.assertEqual(dds.decode_dds(data),
                         (2, 1, bytes((0x10, 0x20, 0x30, 255, 0x40, 0x50, 0x60, 255))))

    def test_padded_rows_are_removed(self):
        # width 2 -> 6 data bytes per row, padded to the 8-byte pitch some writers emit.
        payload = bytes((1, 2, 3, 4, 5, 6)) + b"\x99" * 2 + bytes((7, 8, 9, 10, 11, 12)) + b"\x99" * 2
        data = dds_buffer(
            2, 2, payload, flags=DDPF_RGB, bit_count=24, pitch=8,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000),
        )
        self.assertEqual(pixels(dds.decode_dds(data)[2]),
                         [(3, 2, 1, 255), (6, 5, 4, 255), (9, 8, 7, 255), (12, 11, 10, 255)])

    def test_alpha_flag_with_zero_mask_still_decodes(self):
        # Some writers set DDPF_ALPHAPIXELS on 24-bit headers with a zero alpha mask.
        payload = bytes((1, 2, 3))
        data = dds_buffer(
            1, 1, payload, flags=DDPF_RGB | DDPF_ALPHAPIXELS, bit_count=24,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000),
        )
        self.assertEqual(dds.decode_dds(data), (1, 1, bytes((3, 2, 1, 255))))

    def test_pitch_and_truncation_errors_are_reported(self):
        header = dict(flags=DDPF_RGB, bit_count=24,
                      masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000))
        with self.assertRaisesRegex(dds.DdsError, "pitch"):
            dds.decode_dds(dds_buffer(2, 2, b"\x00" * 6, pitch=5, **header))
        with self.assertRaisesRegex(dds.DdsError, "Truncated"):
            dds.decode_dds(dds_buffer(2, 2, b"\x00" * 9, pitch=6, **header))

    def test_unsupported_24_bit_mask_layouts_are_rejected(self):
        for masks in ((0x000000FF, 0x00000000, 0x00000000, 0x00000000),
                      (0x000000FF, 0x000000FF, 0x000000FF, 0x00000000),
                      (0x00000FFF, 0x000FF000, 0x0FF00000, 0x00000000)):
            with self.subTest(masks=masks):
                data = dds_buffer(1, 1, flags=DDPF_RGB, bit_count=24, masks=masks)
                with self.assertRaisesRegex(dds.DdsError, "24-bit RGB bitmask layout"):
                    dds.decode_dds(data)

    def test_png_preview_round_trip(self):
        payload = bytes((0x10, 0x20, 0x30, 0x40, 0x50, 0x60))
        data = dds_buffer(
            2, 1, payload, flags=DDPF_RGB, bit_count=24,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000),
        )
        png = dds.dds_to_png_bytes(data)
        self.assertEqual(png[:8], dds.PNG_SIGNATURE)
        self.assertEqual(struct.unpack(">II", png[16:24]), (2, 1))


class AlphaModeTests(unittest.TestCase):
    def test_premultiplied_rgba_and_bgra_decode_to_straight_alpha(self):
        for dxgi, rgba in ((28, (64, 32, 16, 128)), (87, (16, 32, 64, 128))):
            with self.subTest(dxgi=dxgi):
                data = dds_buffer(1, 1, bytes(rgba), flags=DDPF_FOURCC, fourcc=b"DX10",
                                  dxgi=dxgi, alpha_mode=2)
                self.assertEqual(dds.decode_dds(data), (1, 1, bytes((128, 64, 32, 128))))

    def test_premultiplied_alpha_rounding_clamping_and_endpoints(self):
        stored = bytes((1, 1, 1, 2, 9, 3, 1, 2, 4, 5, 6, 0, 7, 8, 9, 255))
        expected = bytes((128, 128, 128, 2, 255, 255, 128, 2, 0, 0, 0, 0, 7, 8, 9, 255))
        data = dds_buffer(4, 1, stored, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=28, alpha_mode=2)
        self.assertEqual(dds.decode_dds(data), (4, 1, expected))

    def test_premultiplied_srgb_is_unmultiplied_in_linear_light(self):
        # Decode sRGB, divide linear RGB by 128/255, encode sRGB, then round to 8 bits.
        # Direct division of the encoded bytes would produce (255, 197, 141), which is wrong.
        for dxgi, rgba in ((29, (137, 99, 71, 128)), (91, (71, 99, 137, 128))):
            with self.subTest(dxgi=dxgi):
                data = dds_buffer(1, 1, bytes(rgba), flags=DDPF_FOURCC, fourcc=b"DX10",
                                  dxgi=dxgi, alpha_mode=2)
                self.assertEqual(dds.decode_dds(data), (1, 1, bytes((187, 137, 99, 128))))

    def test_premultiplied_bc3_is_converted_after_block_decoding(self):
        block = bc3_block(128, 0, 0, c0=0x4000, c1=0)
        data = dds_buffer(4, 4, block, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=77, alpha_mode=2)
        self.assertEqual(dds.decode_dds(data), (4, 4, bytes((131, 0, 0, 128)) * 16))

    def test_legacy_premultiplied_fourcc_png_pixels(self):
        # The two FourCCs in each pair share their block layout and differ in alpha interpretation.
        bc2 = bytes([0x88] * 8) + struct.pack("<HHI", 0x4000, 0, 0)
        bc3 = bc3_block(128, 0, 0, c0=0x4000, c1=0)
        for fourcc, block, red, alpha in ((b"DXT2", bc2, 124, 136), (b"DXT3", bc2, 66, 136),
                                         (b"DXT4", bc3, 131, 128), (b"DXT5", bc3, 66, 128)):
            with self.subTest(fourcc=fourcc):
                data = dds_buffer(1, 1, block, flags=DDPF_FOURCC, fourcc=fourcc)
                chunks = png_chunks(dds.dds_to_png_bytes(data))
                raw = zlib.decompress(b"".join(payload for kind, payload, _ in chunks if kind == b"IDAT"))
                self.assertEqual(raw, bytes((0, red, 0, 0, alpha)))

    def test_alpha_mode_uses_only_the_low_three_bits(self):
        data = dds_buffer(1, 1, bytes((64, 32, 16, 128)), flags=DDPF_FOURCC, fourcc=b"DX10",
                          dxgi=28, alpha_mode=0x80000002)
        self.assertEqual(dds.decode_dds(data)[2], bytes((128, 64, 32, 128)))

    def test_other_alpha_modes_keep_their_existing_behavior(self):
        for mode in (0, 1, 3, 4):
            with self.subTest(mode=mode):
                data = dds_buffer(1, 1, bytes((64, 32, 16, 128)), flags=DDPF_FOURCC,
                                  fourcc=b"DX10", dxgi=28, alpha_mode=mode)
                expected_alpha = 255 if mode == 3 else 128
                self.assertEqual(dds.decode_dds(data)[2], bytes((64, 32, 16, expected_alpha)))

    def test_png_stores_straight_color_and_original_alpha(self):
        data = dds_buffer(1, 1, bytes((64, 32, 16, 128)), flags=DDPF_FOURCC, fourcc=b"DX10",
                          dxgi=28, alpha_mode=2)
        chunks = png_chunks(dds.dds_to_png_bytes(data))
        raw = zlib.decompress(b"".join(payload for kind, payload, _ in chunks if kind == b"IDAT"))
        self.assertEqual(raw, bytes((0, 128, 64, 32, 128)))  # filter byte, then one RGBA pixel


class Bc1Tests(unittest.TestCase):
    def test_four_color_mode_row(self):
        # White and black endpoints in c0 > c1 mode, indices 0,1,2,3 across the row.
        payload = bc1_block(0xFFFF, 0x0000, [0b11100100, 0, 0, 0])
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual(pixels(rgba), [
            (255, 255, 255, 255),
            (0, 0, 0, 255),
            (170, 170, 170, 255),
            (85, 85, 85, 255),
        ])

    def test_three_color_mode_row(self):
        # c0 <= c1 switches to the three-color palette with a transparent slot.
        payload = bc1_block(0x0000, 0xFFFF, [0b11100100, 0, 0, 0])
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual(pixels(rgba), [
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            (127, 127, 127, 255),
            (0, 0, 0, 0),
        ])

    def test_second_block_lands_at_column_four(self):
        payload = (
            bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0])
            + bc1_block(0xF800, 0x001F, [0, 0, 0, 0])
        )
        width, height, rgba = dds.decode_dds(dds_buffer(
            8, 4, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        self.assertEqual((width, height), (8, 4))
        self.assertEqual(pixel(rgba, 8, 0, 0), (255, 255, 255, 255))
        self.assertEqual(pixel(rgba, 8, 3, 3), (255, 255, 255, 255))
        self.assertEqual(pixel(rgba, 8, 4, 0), (255, 0, 0, 255))
        self.assertEqual(pixel(rgba, 8, 7, 3), (255, 0, 0, 255))


class Bc2Tests(unittest.TestCase):
    def test_alpha_nibbles_scale_to_bytes(self):
        alpha_q = 0x3210  # nibbles 0, 1, 2, 3 from left pixel to right
        payload = struct.pack("<QHHI", alpha_q, 0xFFFF, 0x0000, 0)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT3",
        ))
        self.assertEqual([p[3] for p in pixels(rgba)], [0, 17, 34, 51])
        self.assertEqual([p[:3] for p in pixels(rgba)], [(255, 255, 255)] * 4)


class Bc3Tests(unittest.TestCase):
    def test_explicit_alpha_interpolation(self):
        # 255 > 0 selects the eight-value alpha ramp.
        indices = 0 | (1 << 3) | (2 << 6) | (7 << 9)
        payload = bc3_block(255, 0, indices)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"DXT5",
        ))
        self.assertEqual([p[3] for p in pixels(rgba)], [255, 0, 219, 36])
        self.assertEqual([p[:3] for p in pixels(rgba)], [(255, 255, 255)] * 4)


class Bc4Tests(unittest.TestCase):
    def test_unsigned_gradient(self):
        indices = 0 | (1 << 3) | (2 << 6) | (7 << 9)
        payload = bc4_block(200, 100, indices)
        _, _, rgba = dds.decode_dds(dds_buffer(
            4, 1, payload, flags=DDPF_FOURCC, fourcc=b"ATI1",
        ))
        self.assertEqual(pixels(rgba), [
            (200, 200, 200, 255),
            (100, 100, 100, 255),
            (186, 186, 186, 255),
            (114, 114, 114, 255),
        ])

    def test_snorm_negative_endpoint_maps_to_zero(self):
        payload = bc4_block(129, 64, 0 | (1 << 3))
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_FOURCC, fourcc=b"BC4S",
        ))
        # 129 as int8 is -127, which maps to 0; 64 maps to round(191.75) = 192.
        self.assertEqual(pixels(rgba), [(0, 0, 0, 255), (192, 192, 192, 255)])


class Bc5Tests(unittest.TestCase):
    def test_two_channel_z_reconstruction(self):
        red = bc4_block(128, 0, 0)
        green = bc4_block(128, 0, 0)
        payload = red + green
        _, _, rgba = dds.decode_dds(dds_buffer(
            1, 1, payload, flags=DDPF_FOURCC, fourcc=b"ATI2",
        ))
        # r = g = 128 is almost exactly normal (0, 0), so z rounds to 255.
        self.assertEqual(pixels(rgba), [(128, 128, 255, 255)])

    def test_z_lut_matches_the_direct_formula(self):
        lut = dds._get_bc5_z_lut()
        self.assertEqual(len(lut), 65536)
        for r in (0, 1, 64, 127, 128, 200, 254, 255):
            for g in (0, 1, 64, 127, 128, 200, 254, 255):
                with self.subTest(r=r, g=g):
                    self.assertEqual(lut[(r << 8) | g], dds._bc5_z_direct(r, g))

    def test_large_image_uses_the_cached_lut(self):
        # 64x64 is 4096 pixels, where the decoder switches to the precomputed table.
        red = bc4_block(200, 100, 0 | (1 << 3) | (2 << 6) | (3 << 9))
        green = bc4_block(100, 200, 0 | (1 << 3) | (2 << 6) | (3 << 9))
        payload = (red + green) * (16 * 16)
        width, height, rgba = dds.decode_dds(dds_buffer(
            64, 64, payload, flags=DDPF_FOURCC, fourcc=b"BC5U",
        ))
        self.assertEqual((width, height), (64, 64))
        self.assertIsNotNone(dds._BC5_Z_LUT)
        lut = dds._get_bc5_z_lut()
        # Red uses the 200 > 100 eight-value ramp, green the 100 <= 200 six-value one.
        for x, expected in enumerate([(200, 100), (100, 200), (186, 120), (171, 140)]):
            r, g = expected
            self.assertEqual(pixel(rgba, 64, x, 0), (r, g, lut[(r << 8) | g], 255))

    def test_snorm_channels(self):
        payload = bc4_block(127, 0, 0 | (1 << 3)) * 2
        _, _, rgba = dds.decode_dds(dds_buffer(
            2, 1, payload, flags=DDPF_FOURCC, fourcc=b"BC5S",
        ))
        self.assertEqual([p[:2] for p in pixels(rgba)], [(255, 255), (128, 128)])
        self.assertEqual([p[3] for p in pixels(rgba)], [255, 255])


class Bc7Tests(unittest.TestCase):
    def decode(self, payload, width=4, height=4, dxgi=98):
        w, h, rgba = dds.decode_dds(bc7_dds(width, height, payload, dxgi))
        self.assertEqual((w, h), (width, height))
        self.assertEqual(len(rgba), width * height * 4)
        return pixels(rgba)

    def test_partitioned_modes(self):
        cases = {
            0: (BC7_MODE0, BC7_MODE0_PIXELS),
            1: (BC7_MODE1, BC7_MODE1_PIXELS),
            2: (BC7_MODE2, BC7_MODE2_PIXELS),
            3: (BC7_MODE3, BC7_MODE3_PIXELS),
            7: (BC7_MODE7, BC7_MODE7_PIXELS),
        }
        for mode, (block, expected) in cases.items():
            with self.subTest(mode=mode):
                self.assertEqual(self.decode(block), expected)

    def test_mode4_index_selector_and_rotation(self):
        # Selector 0: colour takes the 2-bit indices, alpha (255 -> 0) the 3-bit ones.
        self.assertEqual(
            self.decode(bc7_mode4(rotation=0, selector=0)),
            [gray(RAMP2[p % 4], RAMP3[7 - p % 8]) for p in range(16)],
        )
        # Selector 1 swaps the index sets; rotation 1 then swaps R and A.
        self.assertEqual(
            self.decode(bc7_mode4(rotation=1, selector=1)),
            [(RAMP2[3 - p % 4], RAMP3[p % 8], RAMP3[p % 8], RAMP3[p % 8]) for p in range(16)],
        )

    def test_mode5_separate_alpha_indices_and_rotation(self):
        self.assertEqual(self.decode(BC7_MODE5), BC7_MODE5_PIXELS)

    def test_mode6_four_bit_indices_and_pbits(self):
        decoded = self.decode(BC7_MODE6)
        self.assertEqual([p[0] for p in decoded], list(RAMP4))
        self.assertEqual(decoded[0], (0, 254, 128, 254))
        self.assertEqual(decoded[8], (135, 120, 129, 255))
        self.assertEqual(decoded[15], (255, 1, 129, 255))

    def test_reserved_mode_is_transparent_black(self):
        for block in (bytes(16), b"\x00" + b"\xff" * 15):
            with self.subTest(block=block.hex()):
                self.assertEqual(self.decode(block), [(0, 0, 0, 0)] * 16)

    def test_srgb_and_typeless_keep_stored_bytes(self):
        payload = BC7_MODE6 + BC7_MODE7
        unorm = self.decode(payload, 8, 4, dxgi=98)
        # Mode 6 pixel 8 is at row 2, column 0 of the 8-wide image.
        self.assertEqual(unorm[16], (135, 120, 129, 255))
        for dxgi in (97, 99):
            with self.subTest(dxgi=dxgi):
                self.assertEqual(self.decode(payload, 8, 4, dxgi=dxgi), unorm)

    def test_blocks_are_placed_and_clipped(self):
        blocks = [BC7_MODE6, BC7_MODE7, BC7_MODE2, BC7_MODE6]
        expected_blocks = [self.decode(block) for block in blocks]
        for width, height in ((5, 7), (8, 8)):
            with self.subTest(size=(width, height)):
                decoded = self.decode(b"".join(blocks), width, height)
                for y in range(height):
                    for x in range(width):
                        block = expected_blocks[(y // 4) * 2 + x // 4]
                        self.assertEqual(decoded[y * width + x], block[(y % 4) * 4 + x % 4], (x, y))
        # A 3x5 image is a 1x2 grid of blocks; 1x1 keeps only the first pixel.
        decoded = self.decode(BC7_MODE0 + BC7_MODE6, 3, 5)
        self.assertEqual(decoded[:3], BC7_MODE0_PIXELS[:3])
        self.assertEqual(decoded[3:6], BC7_MODE0_PIXELS[4:7])
        self.assertEqual(decoded[12:15], [(0, 254, 128, 254), (16, 238, 128, 254), (36, 218, 128, 254)])
        self.assertEqual(self.decode(BC7_MODE6, 1, 1), [(0, 254, 128, 254)])

    def test_truncated_payload(self):
        for width, height, size in ((4, 4, 15), (5, 7, 63)):
            with self.subTest(size=(width, height)):
                with self.assertRaisesRegex(dds.DdsError, "Truncated BC7"):
                    dds.decode_dds(bc7_dds(width, height, b"\x00" * size))


class TruncationTests(unittest.TestCase):
    def test_truncated_block_payloads(self):
        cases = {
            b"DXT1": 8, b"DXT3": 16, b"DXT5": 16, b"ATI1": 8, b"ATI2": 16,
        }
        for fourcc, block in cases.items():
            with self.subTest(fourcc=fourcc):
                with self.assertRaises(dds.DdsError) as ctx:
                    dds.decode_dds(dds_buffer(
                        4, 4, b"\x00" * (block - 1), flags=DDPF_FOURCC, fourcc=fourcc,
                    ))
                self.assertIn("Truncated", str(ctx.exception))


class DownscaleTests(unittest.TestCase):
    def rgba_grid(self, width, height):
        buf = bytearray(width * height * 4)
        for i in range(width * height):
            buf[i * 4] = i
            buf[i * 4 + 3] = 255
        return bytes(buf)

    def test_box_filter_4x4(self):
        w, h, out = dds.downscale_rgba(4, 4, self.rgba_grid(4, 4), 2)
        self.assertEqual((w, h), (2, 2))
        # Each output pixel averages a 2x2 block with round-half-up.
        self.assertEqual([p[0] for p in pixels(out)], [3, 5, 11, 13])
        self.assertEqual([p[3] for p in pixels(out)], [255, 255, 255, 255])

    def test_single_column_averages_pairs(self):
        buf = bytearray()
        for value in (0, 10, 20, 30):
            buf += bytes((value, value, value, 255))
        w, h, out = dds.downscale_rgba(1, 4, bytes(buf), 2)
        self.assertEqual((w, h), (1, 2))
        self.assertEqual([p[0] for p in pixels(out)], [5, 25])

    def test_odd_extents_are_averaged_without_cropping(self):
        line = b"".join(bytes((v, v, v, 255)) for v in (0, 0, 100, 200, 200))
        for width, height in ((5, 1), (1, 5)):
            with self.subTest(width=width, height=height):
                w, h, out = dds.downscale_rgba(width, height, line, 2)
                self.assertEqual((w, h), (max(1, width // 2), max(1, height // 2)))
                self.assertEqual(pixels(out), [(20, 20, 20, 255), (180, 180, 180, 255)])

        rgba = bytes((0, 0, 0, 255)) * 8 + bytes((90, 180, 0, 255))
        self.assertEqual(dds.downscale_rgba(3, 3, rgba, 1), (1, 1, bytes((10, 20, 0, 255))))

    def test_repeated_halving(self):
        w, h, out = dds.downscale_rgba(8, 8, self.rgba_grid(8, 8), 2)
        self.assertEqual((w, h), (2, 2))
        self.assertEqual(len(out), 2 * 2 * 4)

    def test_no_change_needed(self):
        buf = self.rgba_grid(4, 4)
        self.assertEqual(dds.downscale_rgba(4, 4, buf, 4), (4, 4, buf))
        self.assertEqual(dds.downscale_rgba(4, 4, buf, 0), (4, 4, buf))

    def test_invalid_buffer_length(self):
        with self.assertRaises(dds.DdsError):
            dds.downscale_rgba(4, 4, b"\x00" * 8, 2)

    def test_invalid_dimensions(self):
        with self.assertRaises(dds.DdsError):
            dds.downscale_rgba(0, 4, b"", 2)

    def test_negative_max_size_rejected(self):
        buf = self.rgba_grid(4, 4)
        with self.assertRaisesRegex(dds.DdsError, "Invalid max_size: -1"):
            dds.downscale_rgba(4, 4, buf, -1)
        self.assertEqual(dds.downscale_rgba(4, 4, buf, 0), (4, 4, buf))


class PngTests(unittest.TestCase):
    def test_png_structure_and_crcs(self):
        rgba = bytes((1, 2, 3, 4, 5, 6, 7, 8))
        png = dds.encode_png(2, 1, rgba)
        chunks = png_chunks(png)
        self.assertEqual([c[0] for c in chunks], [b"IHDR", b"IDAT", b"IEND"])
        for ctype, payload, crc in chunks:
            expected = zlib.crc32(payload, zlib.crc32(ctype)) & 0xFFFFFFFF
            self.assertEqual(crc, expected, ctype)
        self.assertEqual(chunks[0][1], struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0))
        self.assertEqual(chunks[2][1], b"")
        # One filter byte (0 = None) per row, then the RGBA row itself.
        self.assertEqual(zlib.decompress(chunks[1][1]), b"\x00" + rgba)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            dds.encode_png(0, 1, b"")
        with self.assertRaises(ValueError):
            dds.encode_png(2, 1, b"\x00" * 7)

    def test_packed_r11g11b10_preview_respects_row_pitch_and_channel_bits(self) -> None:
        def packed(red: int, green: int, blue: int) -> int:
            return red | (green << 11) | (blue << 22)

        bright = packed(15 << 6, 14 << 6, 13 << 5)  # 1.0, 0.5, 0.25
        fractions = packed((14 << 6) | 32, (13 << 6) | 32, (14 << 5) | 8)  # 0.75, 0.375, 0.625
        edge = packed(0, 1, 31 << 5)  # zero, subnormal, infinity
        nan_hdr = packed((31 << 6) | 1, 16 << 6, 31)  # NaN, 2.0, subnormal
        payload = struct.pack("<II", bright, fractions) + b"\xee" * 4 + struct.pack("<II", edge, nan_hdr)
        data = dds_buffer(2, 2, payload, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=26, pitch=12)

        chunks = png_chunks(dds.dds_to_png_bytes(data))
        self.assertEqual(struct.unpack(">II", chunks[0][1][:8]), (2, 2))
        self.assertEqual(zlib.decompress(chunks[1][1]), bytes((
            0, 255, 128, 64, 255, 191, 96, 159, 255,
            0, 0, 0, 255, 255, 0, 255, 0, 255,
        )))

    def test_dds_to_png_roundtrip(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        png = dds.dds_to_png_bytes(dds_buffer(
            8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ))
        chunks = png_chunks(png)
        self.assertEqual(struct.unpack(">II", chunks[0][1][:8]), (8, 8))
        raw = zlib.decompress(chunks[1][1])
        stride = 8 * 4 + 1
        self.assertEqual(len(raw), stride * 8)
        for row in range(8):
            self.assertEqual(raw[row * stride], 0)
            row_bytes = raw[row * stride + 1:(row + 1) * stride]
            self.assertEqual(row_bytes, b"\xff" * (8 * 4))

    def test_downscaled_dds_to_png(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 16
        png = dds.dds_to_png_bytes(dds_buffer(
            16, 16, payload, flags=DDPF_FOURCC, fourcc=b"DXT1",
        ), max_size=4)
        self.assertEqual(struct.unpack(">II", png_chunks(png)[0][1][:8]), (4, 4))

    def test_downscaled_png_weights_color_by_alpha(self) -> None:
        red = bytes((255, 0, 0, 255))
        translucent_green = bytes((0, 255, 0, 128))
        transparent_blue = bytes((0, 0, 255, 0))
        cases = (
            (2, red + transparent_blue, (255, 0, 0, 128)),
            (3, red + translucent_green + transparent_blue, (170, 85, 0, 128)),
        )
        for width, rgba, expected in cases:
            with self.subTest(width=width):
                data = dds_buffer(width, 1, rgba, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=28)
                chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=1))
                self.assertEqual(struct.unpack(">II", chunks[0][1][:8]), (1, 1))
                self.assertEqual(zlib.decompress(chunks[1][1]), bytes((0, *expected)))

    def test_png_downscale_filters_tagged_srgb_in_linear_light(self):
        rgba = bytes((0, 32, 0, 255, 255, 160, 10, 255))
        # BC1 indices 1 and 0 select black and white in the first two texels.
        bc1 = bc1_block(0xFFFF, 0, [1, 0, 0, 0])
        for dxgi, payload, expected in (
            (29, rgba, (188, 119, 5, 255)),
            (28, rgba, (128, 96, 5, 255)),
            (72, bc1, (188, 188, 188, 255)),
            (71, bc1, (128, 128, 128, 255)),
        ):
            with self.subTest(dxgi=dxgi):
                data = dds_buffer(2, 1, payload, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=dxgi)
                chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=1))
                self.assertEqual(zlib.decompress(chunks[1][1]), bytes((0, *expected)))

        data = dds_buffer(2, 1, rgba, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=29)
        for max_size in (0, 2):
            chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=max_size))
            self.assertEqual(zlib.decompress(chunks[1][1]), b"\0" + rgba)

    def test_srgb_downscale_weights_linear_color_by_alpha(self):
        # Opaque red, half-covered green and invisible blue. Alpha is linear.
        rgba = bytes((255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0))
        data = dds_buffer(3, 1, rgba, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=29)
        chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=1))
        self.assertEqual(zlib.decompress(chunks[1][1]), bytes((0, 213, 156, 0, 128)))

    def test_srgb_linear_segment_rounds_half_bytes_up(self):
        # All channels lie on the linear segment. Their means are 5.5, 5.5 and 3.5.
        for alpha in (128, 255):
            left = bytes((2, 5, 6, alpha))
            right = bytes((9, 6, 1, alpha))
            for rgba in (left + right, right + left):
                with self.subTest(rgba=rgba):
                    data = dds_buffer(2, 1, rgba, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=29)
                    chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=1))
                    self.assertEqual(zlib.decompress(chunks[1][1]), bytes((0, 6, 6, 4, alpha)))

    def test_srgb_downscale_across_two_levels(self):
        # An opaque white first row and three black rows give 25% linear light.
        for width in (1, 4):
            with self.subTest(width=width):
                rgba = b"\xff" * (4 * width) + bytes((0, 0, 0, 255)) * (3 * width)
                data = dds_buffer(width, 4, rgba, flags=DDPF_FOURCC, fourcc=b"DX10", dxgi=29)
                chunks = png_chunks(dds.dds_to_png_bytes(data, max_size=1))
                self.assertEqual(zlib.decompress(chunks[1][1]), bytes((0, 137, 137, 137, 255)))

    def test_dds_to_png_rejects_negative_max_size(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        data = dds_buffer(8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1")
        with self.assertRaisesRegex(dds.DdsError, "Invalid max_size: -1"):
            dds.dds_to_png_bytes(data, max_size=-1)
        png = dds.dds_to_png_bytes(data, max_size=0)
        self.assertEqual(struct.unpack(">II", png_chunks(png)[0][1][:8]), (8, 8))


class CliTests(unittest.TestCase):
    def test_cli_writes_png(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "nested", "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1"))
            self.assertEqual(dds.main([src, dst]), 0)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(8), dds.PNG_SIGNATURE)

    def test_cli_rejects_output_aliasing_input(self):
        original = dds_buffer(4, 4, bc1_block(0xFFFF, 0, [0, 0, 0, 0]),
                              flags=DDPF_FOURCC, fourcc=b"DXT1")
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            alias = os.path.join(tmp, ".", "in.dds")
            with open(src, "wb") as f:
                f.write(original)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = dds.main([src, alias])
            with open(src, "rb") as f:
                self.assertEqual(f.read(), original)
            self.assertEqual(result, 1)
            self.assertIn("output must not overwrite the input DDS", stderr.getvalue())

    def test_cli_publishes_only_after_a_complete_write(self):
        original = dds_buffer(4, 4, bc1_block(0xFFFF, 0, [0, 0, 0, 0]),
                              flags=DDPF_FOURCC, fourcc=b"DXT1")
        previous = dds.encode_png(1, 1, b"\x10\x20\x30\xff")
        replacement = dds.dds_to_png_bytes(original)
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "in.dds")
                dst = os.path.join(tmp, "out.png")
                with open(src, "wb") as f:
                    f.write(original)
                if existing:
                    with open(dst, "wb") as f:
                        f.write(previous)

                @contextlib.contextmanager
                def fail_partial_write(path, mode):
                    with open(path, mode) as f:
                        def write(data):
                            f.write(data[:4])
                            raise OSError("synthetic disk-full write")

                        yield mock.Mock(write=write) if mode == "wb" else f

                err = io.StringIO()
                with mock.patch.object(dds, "open", side_effect=fail_partial_write, create=True), \
                        contextlib.redirect_stderr(err):
                    self.assertEqual(dds.main([src, dst]), 1)
                self.assertIn("synthetic disk-full write", err.getvalue())
                self.assertCountEqual(os.listdir(tmp), ["in.dds", "out.png"] if existing else ["in.dds"])
                if existing:
                    with open(dst, "rb") as f:
                        self.assertEqual(f.read(), previous)
                self.assertEqual(dds.main([src, dst]), 0)
                with open(dst, "rb") as f:
                    self.assertEqual(f.read(), replacement)

    def test_cli_reports_unsupported_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(4, 4, flags=DDPF_FOURCC, fourcc=b"XXXX"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(dds.main([src, dst]), 1)
            self.assertIn("FourCC", err.getvalue())
            self.assertFalse(os.path.exists(dst))

    def test_cli_rejects_negative_max_size(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                dds.main([src, dst, "--max-size", "-1"])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--max-size must be 0 or more", err.getvalue())
            self.assertFalse(os.path.exists(dst))

    def test_cli_allows_zero_max_size(self):
        payload = bc1_block(0xFFFF, 0x0000, [0, 0, 0, 0]) * 4
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(8, 8, payload, flags=DDPF_FOURCC, fourcc=b"DXT1"))
            self.assertEqual(dds.main([src, dst, "--max-size", "0"]), 0)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(8), dds.PNG_SIGNATURE)


def halves(*codes):
    return struct.pack(f"<{len(codes)}H", *codes)


HALF_FLOAT = dict(flags=DDPF_FOURCC, fourcc=b"DX10")


class HalfFloatTests(unittest.TestCase):
    """DXGI 54 (R16_FLOAT) and 10 (R16G16B16A16_FLOAT): clamp to [0, 1], round(v * 255) half up."""

    def test_header_names_the_half_float_formats(self):
        for dxgi, name in ((54, "R16_FLOAT"), (10, "RGBA16_FLOAT")):
            with self.subTest(dxgi=dxgi):
                info = dds.parse_dds_header(dds_buffer(1, 1, dxgi=dxgi, **HALF_FLOAT))
                self.assertEqual((info["format"], info["dxgi_format"]), (name, dxgi))

    def test_r16_float_fills_red_and_skips_row_padding(self):
        # 1.0, 0.5 | 2^-8, 2.0 with four padding bytes after each 4-byte row.
        payload = halves(0x3C00, 0x3800) + b"\xcc" * 4 + halves(0x1C00, 0x4000)
        width, height, rgba = dds.decode_dds(dds_buffer(2, 2, payload, pitch=8, dxgi=54, **HALF_FLOAT))
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(pixels(rgba), [(255, 0, 0, 255), (128, 0, 0, 255), (1, 0, 0, 255), (255, 0, 0, 255)])

    def test_rgba16_float_keeps_alpha_and_skips_row_padding(self):
        payload = (halves(0x3C00, 0x3800, 0x0000, 0x3C00) + b"\xcc" * 4
                   + halves(0x3555, 0x1C00, 0x3BFF, 0x3800))
        width, height, rgba = dds.decode_dds(dds_buffer(1, 2, payload, pitch=12, dxgi=10, **HALF_FLOAT))
        self.assertEqual((width, height), (1, 2))
        # 0x3555 = 0.333252 -> 85; 0x3BFF = 0.999512 -> 255.
        self.assertEqual(pixels(rgba), [(255, 128, 0, 255), (85, 1, 255, 128)])

    def test_tightly_packed_rows_without_a_pitch(self):
        payload = halves(0x3800, 0x3C00, 0x0000, 0x3400)  # 0.5, 1, 0, 0.25
        for pitch in (None, 0):
            with self.subTest(pitch=pitch):
                self.assertEqual(dds.decode_dds(dds_buffer(2, 2, payload, pitch=pitch, dxgi=54, **HALF_FLOAT)),
                                 (2, 2, bytes((128, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 64, 0, 0, 255))))

    def test_nonfinite_negative_and_subnormal_values(self):
        cases = (
            (0x7C00, 255),  # +inf
            (0xFC00, 0),  # -inf
            (0x7E00, 0),  # quiet NaN
            (0x7C01, 0),  # signalling NaN
            (0xFFFF, 0),  # negative NaN
            (0x8000, 0),  # -0
            (0xBC00, 0),  # -1
            (0x7BFF, 255),  # 65504, largest finite
            (0x0001, 0),  # smallest subnormal
            (0x0000, 0),
        )
        payload = halves(*(code for code, _ in cases))
        _, _, r16 = dds.decode_dds(dds_buffer(len(cases), 1, payload, dxgi=54, **HALF_FLOAT))
        self.assertEqual(list(r16[0::4]), [v for _, v in cases])
        self.assertEqual(set(r16[3::4]), {255})
        # Same codes in every RGBA16_FLOAT channel, alpha included.
        rgba_payload = b"".join(halves(code, code, code, code) for code, _ in cases)
        _, _, rgba = dds.decode_dds(dds_buffer(len(cases), 1, rgba_payload, dxgi=10, **HALF_FLOAT))
        self.assertEqual(pixels(rgba), [(v, v, v, v) for _, v in cases])

    def test_every_half_code_maps_deterministically(self):
        payload = struct.pack("<65536H", *range(65536))
        data = dds_buffer(256, 256, payload, dxgi=54, **HALF_FLOAT)
        _, _, rgba = dds.decode_dds(data)
        red = rgba[0::4]
        self.assertEqual(dds.decode_dds(data)[2], rgba)
        positive = red[:0x7C00]  # +0 up to 65504, in increasing order
        self.assertTrue(all(a <= b for a, b in zip(positive, positive[1:])))
        self.assertEqual((red[0x3C00], red[0x7C00]), (255, 255))
        self.assertFalse(any(red[0x7C01:0x8000]), "NaN must be 0")
        self.assertFalse(any(red[0x8000:]), "the sign bit must give 0")

    def test_pitch_and_truncation_errors(self):
        with self.assertRaisesRegex(dds.DdsError, "RGBA16_FLOAT row pitch"):
            dds.decode_dds(dds_buffer(1, 2, bytes(16), pitch=4, dxgi=10, **HALF_FLOAT))
        with self.assertRaisesRegex(dds.DdsError, "Truncated R16_FLOAT"):
            dds.decode_dds(dds_buffer(2, 2, bytes(11), pitch=8, dxgi=54, **HALF_FLOAT))
        # The last row needs only its own bytes, not a full pitch.
        self.assertEqual(dds.decode_dds(dds_buffer(2, 2, bytes(12), pitch=8, dxgi=54, **HALF_FLOAT))[0:2], (2, 2))

    def test_cli_writes_the_png(self):
        payload = halves(0x3C00, 0x0000, 0x3800, 0x3C00, 0x7E00, 0x7C00, 0x3400, 0x3800)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.dds")
            dst = os.path.join(tmp, "out.png")
            with open(src, "wb") as f:
                f.write(dds_buffer(2, 1, payload, pitch=16, dxgi=10, **HALF_FLOAT))
            self.assertEqual(dds.main([src, dst]), 0)
            with open(dst, "rb") as f:
                chunks = png_chunks(f.read())
        ihdr = next(p for t, p, _ in chunks if t == b"IHDR")
        self.assertEqual(struct.unpack(">II", ihdr[:8]), (2, 1))
        raw = zlib.decompress(b"".join(p for t, p, _ in chunks if t == b"IDAT"))
        self.assertEqual(raw[1:], bytes((255, 0, 128, 255, 0, 255, 64, 128)))


if __name__ == "__main__":
    unittest.main()
