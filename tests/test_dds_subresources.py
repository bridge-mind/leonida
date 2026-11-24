"""Preview synthetic DDS mip levels, array layers, cube faces and volume slices.

    python -I tests/test_dds_subresources.py
"""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ttpf_mips as pool_fixtures
from test_dds_png import dds, dds_buffer

FACES = ("px", "nx", "py", "ny", "pz", "nz")
ttpf = pool_fixtures.ttpf


def dx10(width, height, payload, *, mips=1, dimension=3, layers=1, cube=False,
         depth=1, fmt=28, pitch=None):
    data = bytearray(dds_buffer(width, height, payload, flags=4, fourcc=b"DX10",
                                dxgi=fmt, mips=mips, pitch=pitch))
    flags = 0x1007 | (0x8 if pitch is not None else 0) | (0x20000 if mips > 1 else 0)
    struct.pack_into("<I", data, 8, flags | (0x800000 if dimension == 4 else 0))
    caps = 0x1000 | (0x400008 if mips > 1 else 0)
    struct.pack_into("<I", data, 108, caps | (0x8 if cube or layers > 1 or dimension == 4 else 0))
    struct.pack_into("<III", data, 132, dimension, 4 if cube else 0, layers)
    if cube:
        struct.pack_into("<I", data, 112, 0xFE00)  # all six faces
    if dimension == 4:
        struct.pack_into("<I", data, 24, depth)
        struct.pack_into("<I", data, 112, 0x200000)  # DDSCAPS2_VOLUME
    return bytes(data)


def alpha_plane(values):
    return b"".join(bytes((0, 0, 0, v)) for v in values)


class SubresourceTests(unittest.TestCase):
    def test_legacy_partial_cube_follows_declared_faces_and_bc_block_floor(self):
        red, green, blue = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        # A 5x5 DXT1 face has four blocks, and each 2x2 and 1x1 mip takes one.
        blocks = [red, green, blue, green, blue, red, green, blue, red, red, red, blue]
        data = bytearray(dds_buffer(5, 5, b"".join(map(pool_fixtures.solid_bc1, blocks)),
                                    flags=4, fourcc=b"DXT1", mips=3))
        struct.pack_into("<I", data, 8, 0xA1007)  # linear size, mip count
        struct.pack_into("<I", data, 20, 32)
        struct.pack_into("<I", data, 108, 0x401008)
        struct.pack_into("<I", data, 112, 0x200 | 0x800 | 0x8000)  # -X, -Z
        self.assertEqual(dds.parse_dds_header(data)["faces"], ("nx", "nz"))
        first_row = bytes((*red, 255)) * 4 + bytes((*green, 255))
        last_row = bytes((*blue, 255)) * 4 + bytes((*green, 255))
        self.assertEqual(dds.decode_dds(data), (5, 5, first_row * 4 + last_row))
        self.assertEqual(dds.decode_dds(data, face="nz", mip=1), (2, 2, bytes((*red, 255)) * 4))
        self.assertEqual(dds.decode_dds(data, face="nz", mip=2), (1, 1, bytes((*blue, 255))))
        with self.assertRaises(dds.DdsError):
            dds.decode_dds(data, face="px")

    def test_project_writer_array_and_cube(self):
        for cube, count, selection in ((False, 3, {"layer": 2}), (True, 6, {"face": "nz"})):
            with self.subTest(cube=cube):
                # Each image has a 2x2 top level followed immediately by one 1x1 child.
                payload = b"".join(bytes([v]) * 16 + bytes([v + 10]) * 4 for v in range(count))
                data = ttpf.dds_bytes(2, 2, 28, 2, 8, payload, array_size=count, is_cube=cube)
                self.assertEqual(dds.decode_dds(data, mip=1, **selection),
                                 (1, 1, bytes([count + 9]) * 4))

    def test_dx10_cube_array_order_and_public_png_apis(self):
        # Two cubes, six faces each. The tail byte is distinct for each face in both cubes.
        payload = b"".join(bytes([v]) * 16 + bytes([v + 32]) * 4 for v in range(12))
        data = dx10(2, 2, payload, mips=2, layers=2, cube=True)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp, "cube.dds")
            src.write_bytes(data)
            for layer in (0, 1):
                for index, face in enumerate(FACES):
                    with self.subTest(layer=layer, face=face):
                        select = dict(layer=layer, face=face, mip=1)
                        pixel = bytes([32 + 6 * layer + index]) * 4
                        expected = dds.encode_png(1, 1, pixel)
                        self.assertEqual(dds.dds_to_png_bytes(data, **select), expected)
                        self.assertEqual(dds.dds_file_to_png_bytes(src, **select), expected)

    def test_dx10_1d_array_mips(self):
        # Five, two and one A8 texel per mip, packed for each layer in turn.
        data = dx10(5, 1, bytes(range(16)), mips=3, dimension=2, layers=2, fmt=65)
        info = dds.parse_dds_header(data)
        self.assertEqual((info["dimension"], info["depth"], info["array_size"]), ("1d", 1, 2))
        self.assertEqual(dds.decode_dds(data, layer=1, mip=1), (2, 1, alpha_plane((13, 14))))
        self.assertEqual(dds.decode_dds(data, layer=1, mip=2), (1, 1, alpha_plane((15,))))

    def test_volume_depth_shrinks_at_each_mip(self):
        # 4x2x5, then 2x1x2, then 1x1x1. Each texel's alpha is its file byte index.
        data = dx10(4, 2, bytes(range(45)), mips=3, dimension=4, depth=5, fmt=65)
        info = dds.parse_dds_header(data)
        self.assertEqual((info["dimension"], info["depth"], info["faces"]), ("3d", 5, ()))
        self.assertEqual(dds.decode_dds(data, z=4), (4, 2, alpha_plane(range(32, 40))))
        self.assertEqual(dds.decode_dds(data, mip=1, z=1), (2, 1, alpha_plane((42, 43))))
        self.assertEqual(dds.decode_dds(data, mip=2), (1, 1, alpha_plane((44,))))
        with self.assertRaises(dds.DdsError):
            dds.decode_dds(data, mip=1, z=2)

    def test_top_pitch_applies_per_layer_with_packed_lower_mips(self):
        # Two 1x2 A8 layers with four-byte rows. Each layer's 1x1 tail takes one byte.
        payload = bytes((10, 99, 99, 99, 20, 99, 99, 99, 30, 40, 99, 99, 99, 50, 99, 99, 99, 60))
        data = dx10(1, 2, payload, mips=2, layers=2, fmt=65, pitch=4)
        self.assertEqual(dds.decode_dds(data, layer=1, mip=1), (1, 1, alpha_plane((60,))))
        # A preview only needs the visible byte of its final scanline.
        self.assertEqual(dds.decode_dds(data[:-4], layer=1), (1, 2, alpha_plane((40, 50))))

    def test_selected_srgb_mip_is_filtered_in_linear_light(self):
        # Four top pixels then black and white in layer 0's child mip.
        data = dx10(4, 1, bytes(16) + bytes((0, 0, 0, 255, 255, 255, 255, 255)), mips=2, fmt=29)
        self.assertEqual(dds.dds_to_png_bytes(data, 1, mip=1),
                         dds.encode_png(1, 1, bytes((188, 188, 188, 255))))

    def test_only_selected_plane_needs_to_be_present(self):
        # The default preview remains valid when only the base plane was downloaded.
        pixel = bytes((11, 22, 33, 255))
        data = dx10(2, 2, pixel * 4, mips=2, layers=3)
        self.assertEqual(dds.dds_to_png_bytes(data), dds.encode_png(2, 2, pixel * 4))
        for select in ({"mip": 1}, {"layer": 1}):
            with self.subTest(select=select), self.assertRaises(dds.DdsError):
                dds.decode_dds(data, **select)

    def test_file_preview_reads_only_header_and_selected_layer(self):
        class CountReads(io.BytesIO):
            bytes_read = 0

            def read(self, size=-1):
                chunk = super().read(size)
                self.bytes_read += len(chunk)
                return chunk

        pixel = bytes((34, 67, 129, 255))
        source = CountReads(dx10(1, 1, bytes(4 * 1023) + pixel, layers=1024))
        with patch("builtins.open", return_value=source):
            png = dds.dds_file_to_png_bytes("array.dds", layer=1023)
        self.assertEqual(png, dds.encode_png(1, 1, pixel))
        self.assertLessEqual(source.bytes_read, 148 + len(pixel))

    def test_selection_requires_existing_subresource(self):
        data = dx10(2, 2, bytes(20), mips=2)
        for select in ({"mip": -1}, {"mip": 2}, {"layer": -1}, {"layer": 1},
                       {"z": -1}, {"z": 1}, {"face": "px"}, {"face": "bad"}):
            with self.subTest(select=select), self.assertRaises(dds.DdsError):
                dds.decode_dds(data, **select)

    def test_invalid_dx10_resource_shapes(self):
        for header in (dx10(2, 2, b"", layers=0), dx10(2, 2, b"", dimension=2),
                       dx10(2, 2, b"", dimension=4, depth=2, layers=2)):
            with self.subTest(header=header[128:148]), self.assertRaises(dds.DdsError):
                dds.parse_dds_header(header)

    def test_exported_float16_volume_to_png(self):
        # Synthetic RDS5 pool -> normal export path (3D detile) -> preview its last slice.
        w, h, depth = 8, 8, 8
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, last_pixel, expected in (
                (54, (0.5,), bytes((128, 0, 0, 255))),
                (10, (-1.0, 0.25, 2.0, 0.5), bytes((0, 64, 255, 128))),
            ):
                with self.subTest(fmt=fmt):
                    sample = struct.pack("<" + "e" * len(last_pixel), *last_pixel)
                    linear = bytes((depth - 1) * w * h * len(sample)) + sample * (w * h)
                    layout = ttpf.volume_surface_layout(w, h, depth, 1, len(sample))
                    tiled = pool_fixtures.retile_volume(layout, linear)
                    rec = pool_fixtures.pack_etpf_record(w, h, 1, 0, fmt, depth, 3, 2, 4096)
                    pool_path = Path(tmp, f"vol{fmt}.ttpf")
                    dds_path = pool_path.with_suffix(".dds")
                    pool_fixtures.write_rds5_pool(pool_path, [rec], [tiled])
                    with contextlib.closing(ttpf.Pool(pool_path)) as pool, contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(ttpf.export(pool, 0, str(dds_path)), 0)
                    self.assertEqual(dds.dds_file_to_png_bytes(dds_path, z=7),
                                     dds.encode_png(w, h, expected * (w * h)))
                    if fmt == 54:
                        dst = Path(tmp, "slice.png")
                        self.assertEqual(dds.main([str(dds_path), str(dst), "--z", "7"]), 0)
                        self.assertEqual(dst.read_bytes(), dds.encode_png(w, h, expected * (w * h)))

    def test_cli_header_only_info_and_selected_pixels(self):
        data = dx10(2, 2, b"", mips=2, layers=2, cube=True)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "cube.dds"), Path(tmp, "out.png")
            src.write_bytes(data)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(dds.main([str(src), "--info"]), 0)
            info = json.loads(output.getvalue())
            self.assertEqual((info["width"], info["height"], info["mip_count"]), (2, 2, 2))
            self.assertEqual((info["dimension"], info["array_size"], info["faces"]), ("2d", 2, list(FACES)))
            self.assertEqual(list(Path(tmp).iterdir()), [src])
            last_pixel = bytes((12, 34, 56, 78))
            src.write_bytes(data + bytes(236) + last_pixel)
            self.assertEqual(dds.main([str(src), str(dst), "--mip", "1", "--layer", "1",
                                       "--face", "nz"]), 0)
            self.assertEqual(dst.read_bytes(), dds.encode_png(1, 1, last_pixel))


if __name__ == "__main__":
    unittest.main()
