"""Tests for DX10 alpha modes using synthetic DDS buffers (no game data).

    python3 -I tests/test_dds_png_dx10.py
"""
import importlib.util
import os
import struct
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()


class Dx10AlphaTests(unittest.TestCase):
    def test_rgba8_alpha_mode_in_png(self):
        header = bytearray(148)
        header[:4] = b"DDS "
        struct.pack_into("<IIIII", header, 4, 124, 0x100F, 1, 1, 4)
        struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
        struct.pack_into("<I", header, 108, 0x1000)  # DDSCAPS_TEXTURE
        struct.pack_into("<IIII", header, 128, 28, 3, 0, 1)  # RGBA8, 2D, one slice
        pixel = b"\xff\x00\x00\x00"

        for mode, alpha in ((3, 255), (1, 0)):  # Opaque and straight alpha
            with self.subTest(mode=mode):
                struct.pack_into("<I", header, 144, mode)
                png = dds.dds_to_png_bytes(bytes(header) + pixel)
                # The first chunk after IHDR contains the compressed row.
                pos = 8 + 12 + 13
                size, kind = struct.unpack_from(">I4s", png, pos)
                self.assertEqual(kind, b"IDAT")
                row = zlib.decompress(png[pos + 8:pos + 8 + size])
                self.assertEqual(row, bytes((0, 255, 0, 0, alpha)))


def _dx10(width, height, dxgi, payload, pitch=0):
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<IIIII", header, 4, 124, 0x100F | (8 if pitch else 0), height, width, pitch)
    struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
    struct.pack_into("<IIII", header, 128, dxgi, 3, 0, 1)
    return bytes(header) + payload


class SingleChannelTests(unittest.TestCase):
    def test_r32_float_linear_values_with_padding(self):
        payload = (struct.pack("<4f", -1, 0, 0.5, 0.25) + b"skip"
                   + struct.pack("<4f", 2, float("inf"), float("-inf"), float("nan")))
        data = _dx10(4, 2, 41, payload, pitch=20)
        self.assertEqual(dds.parse_dds_header(data)["format"], "R32_FLOAT")
        width, height, rgba = dds.decode_dds(data)
        self.assertEqual((width, height), (4, 2))
        self.assertEqual(rgba, b"".join(bytes((r, 0, 0, 255)) for r in (0, 0, 128, 64, 255, 255, 0, 0)))

    def test_r16_unorm_red_opaque(self):
        px = struct.pack("<HHHH", 0, 65535, 32768, 257)
        _, _, rgba = dds.decode_dds(_dx10(2, 2, 56, px))
        self.assertEqual(rgba, bytes([0, 0, 0, 255, 255, 0, 0, 255, 128, 0, 0, 255, 1, 0, 0, 255]))

    def test_a8_unorm_alpha_only_with_pitch(self):
        payload = bytes([1, 2, 3, 99, 4, 5, 6, 99])  # 3x2, pitch 4
        _, _, rgba = dds.decode_dds(_dx10(3, 2, 65, payload, pitch=4))
        self.assertEqual(rgba[3::4], bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(set(rgba[0::4]) | set(rgba[1::4]) | set(rgba[2::4]), {0})

    def test_truncated_rejected(self):
        for dxgi, n in ((56, 7), (65, 3), (41, 15)):
            with self.assertRaises(dds.DdsError):
                dds.decode_dds(_dx10(2, 2, dxgi, bytes(n)))


if __name__ == "__main__":
    unittest.main()
