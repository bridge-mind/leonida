"""Synthetic unit and integration tests for the world-build pipeline (no game data).

    python -I tests/test_world_build.py -v

Covers:
  - tools/dds_png.py     (DDS header parsing, BGRA8/RGBA8/BGRX8/BC1/BC2/BC3/BC4/BC5, downscale, PNG encoder)
  - tools/glb_writer.py  (OBJ parser, bounds, binary glTF 2.0 writer, texture dedup, 32-bit indices, summary)
  - tools/world_build.py (manifest discovery, TTPF indexing, texture resolution, spatial/grid/no-tex/limit builds,
                          --base-color-param selection with GLB pixel readback)
  - viewer/serve.py      (safe_world_path traversal hardening)

Every texture, mesh, manifest, and archive layout is generated synthetically in memory or a temporary directory.
"""
import functools
import http.client
import importlib.util
import json
import math
import os
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
import zlib
from http.server import HTTPServer, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
VIEWER = os.path.join(ROOT, "viewer")

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import dds_png      # noqa: E402
import glb_writer   # noqa: E402
import world_build  # noqa: E402


def _load_viewer_serve():
    path = os.path.join(VIEWER, "serve.py")
    spec = importlib.util.spec_from_file_location("viewer_serve_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load viewer/serve.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["viewer_serve_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


viewer_serve = _load_viewer_serve()


# ---------------------------------------------------------------------------
# Independent synthetic DDS builders (no game files, pure spec bit layouts)
# ---------------------------------------------------------------------------

def make_bgra8_dds(width: int, height: int, rgba_pixels: bytes) -> bytes:
    """Build a legacy 32-bit BGRA8 DDS matching tools/rpf9dump/Program.cs Util.BgraDds."""
    assert len(rgba_pixels) == width * height * 4
    bgra = bytearray(len(rgba_pixels))
    for i in range(0, len(rgba_pixels), 4):
        r, g, b, a = rgba_pixels[i:i + 4]
        bgra[i:i + 4] = bytes((b, g, r, a))
    hdr = bytearray(128)
    struct.pack_into("<I", hdr, 0, 0x20534444)   # "DDS "
    struct.pack_into("<I", hdr, 4, 124)          # dwSize
    struct.pack_into("<I", hdr, 8, 0x100F)       # CAPS | HEIGHT | WIDTH | PITCH | PIXELFORMAT
    struct.pack_into("<I", hdr, 12, height)
    struct.pack_into("<I", hdr, 16, width)
    struct.pack_into("<I", hdr, 20, width * 4)   # dwPitchOrLinearSize
    struct.pack_into("<I", hdr, 76, 32)          # ddspf.dwSize
    struct.pack_into("<I", hdr, 80, 0x41)        # DDPF_RGB | DDPF_ALPHAPIXELS
    struct.pack_into("<I", hdr, 88, 32)          # dwRGBBitCount
    struct.pack_into("<I", hdr, 92, 0x00FF0000)  # R mask
    struct.pack_into("<I", hdr, 96, 0x0000FF00)  # G mask
    struct.pack_into("<I", hdr, 100, 0x000000FF) # B mask
    struct.pack_into("<I", hdr, 104, 0xFF000000) # A mask
    struct.pack_into("<I", hdr, 108, 0x1000)     # DDSCAPS_TEXTURE
    return bytes(hdr) + bytes(bgra)


def make_dx10_dds(width: int, height: int, dxgi_format: int, payload: bytes, *, compressed: bool = False) -> bytes:
    """Build a DX10 DDS container (128-byte legacy header + 20-byte DX10 extension + payload)."""
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    flags = 0x1007 | (0x80000 if compressed else 0x8)
    pitch = len(payload) if compressed else width * 4
    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, height, width, pitch, 0, 1)
    struct.pack_into("<II4s", hdr, 76, 32, 0x4, b"DX10")
    struct.pack_into("<I", hdr, 108, 0x1000)
    dx10 = struct.pack("<IIIII", dxgi_format, 3, 0, 1, 0)
    return bytes(hdr) + dx10 + payload


def make_fourcc_dds(width: int, height: int, fourcc: bytes, payload: bytes) -> bytes:
    """Build a legacy FourCC compressed DDS (DXT1/DXT3/DXT5/ATI1/ATI2)."""
    assert len(fourcc) == 4
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<IIIIIII", hdr, 4, 124, 0x1007 | 0x80000, height, width, len(payload), 0, 1)
    struct.pack_into("<II4s", hdr, 76, 32, 0x4, fourcc)
    struct.pack_into("<I", hdr, 108, 0x1000)
    return bytes(hdr) + payload


def pack_bc1_block(c0: int, c1: int, codes) -> bytes:
    """Pack one 8-byte BC1/DXT1 block from two RGB565 endpoints and 16 2-bit codes."""
    assert len(codes) == 16
    bits = 0
    for i, c in enumerate(codes):
        bits |= (c & 3) << (2 * i)
    return struct.pack("<HHI", c0, c1, bits)


def pack_bc2_block(alphas4, c0: int, c1: int, color_codes) -> bytes:
    """Pack one 16-byte BC2/DXT3 block (16 4-bit alphas + 8-byte color block)."""
    assert len(alphas4) == 16
    alpha_bits = 0
    for i, a4 in enumerate(alphas4):
        alpha_bits |= (a4 & 0xF) << (4 * i)
    return struct.pack("<Q", alpha_bits) + pack_bc1_block(c0, c1, color_codes)


def pack_bc4_block(a0: int, a1: int, codes) -> bytes:
    """Pack one 8-byte BC4 / BC3-alpha interpolated block (2 endpoints + 16 3-bit codes)."""
    assert len(codes) == 16
    bits = 0
    for i, c in enumerate(codes):
        bits |= (c & 7) << (3 * i)
    return bytes((a0 & 0xFF, a1 & 0xFF)) + bits.to_bytes(6, "little")


def pack_bc3_block(a0: int, a1: int, alpha_codes, c0: int, c1: int, color_codes) -> bytes:
    """Pack one 16-byte BC3/DXT5 block (8-byte interpolated alpha + 8-byte color block)."""
    return pack_bc4_block(a0, a1, alpha_codes) + pack_bc1_block(c0, c1, color_codes)


def pack_bc5_block(r0: int, r1: int, r_codes, g0: int, g1: int, g_codes) -> bytes:
    """Pack one 16-byte BC5 two-channel block (R BC4 block + G BC4 block)."""
    return pack_bc4_block(r0, r1, r_codes) + pack_bc4_block(g0, g1, g_codes)


# ---------------------------------------------------------------------------
# Independent PNG validator (RFC 2083, full CRC32 + zlib + scanline unfilter)
# ---------------------------------------------------------------------------

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def parse_and_validate_png(data: bytes) -> tuple[int, int, bytes]:
    """Independently validate a PNG byte stream and return (width, height, rgba_bytes).

    Verifies:
      - 8-byte PNG signature
      - exact chunk framing & zlib.crc32 on EVERY chunk
      - IHDR first, 8-bit RGBA (color_type=6), deflate/standard/non-interlaced
      - zlib decompression of concatenated IDAT chunks and exact scanline unfiltering
      - IEND last with length 0 and no trailing bytes
    """
    sig = b"\x89PNG\r\n\x1a\n"
    if not isinstance(data, (bytes, bytearray)):
        raise AssertionError(f"PNG must be bytes, got {type(data).__name__}")
    if data[:8] != sig:
        raise AssertionError(f"bad PNG signature: {data[:8]!r}")

    pos = 8
    chunks = []
    while pos < len(data):
        if pos + 12 > len(data):
            raise AssertionError(f"truncated PNG chunk header at offset {pos}")
        length, ctype = struct.unpack_from(">I4s", data, pos)
        pos += 8
        if pos + length + 4 > len(data):
            raise AssertionError(f"truncated PNG chunk {ctype!r} payload at offset {pos}")
        cdata = data[pos:pos + length]
        pos += length
        stored_crc, = struct.unpack_from(">I", data, pos)
        pos += 4
        calc_crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        if stored_crc != calc_crc:
            raise AssertionError(
                f"CRC mismatch on PNG chunk {ctype!r}: stored=0x{stored_crc:08X}, calc=0x{calc_crc:08X}"
            )
        chunks.append((ctype, cdata))
        if ctype == b"IEND":
            break

    if pos != len(data):
        raise AssertionError(f"trailing bytes after IEND: {len(data) - pos}")
    if len(chunks) < 3:
        raise AssertionError(f"too few PNG chunks: {[c[0] for c in chunks]}")
    if chunks[0][0] != b"IHDR":
        raise AssertionError(f"first PNG chunk must be IHDR, got {chunks[0][0]!r}")
    if chunks[-1][0] != b"IEND" or len(chunks[-1][1]) != 0:
        raise AssertionError(f"last PNG chunk must be empty IEND, got {chunks[-1][0]!r} len={len(chunks[-1][1])}")

    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise AssertionError(f"IHDR length must be 13, got {len(ihdr)}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (bit_depth, color_type, compression, filter_method, interlace) != (8, 6, 0, 0, 0):
        raise AssertionError(
            f"unexpected IHDR format: bit_depth={bit_depth}, color_type={color_type}, "
            f"comp={compression}, filter={filter_method}, interlace={interlace}"
        )

    idat_blob = b"".join(cdata for ctype, cdata in chunks if ctype == b"IDAT")
    if not idat_blob:
        raise AssertionError("PNG has no IDAT chunks")
    raw = zlib.decompress(idat_blob)
    bpp = 4
    stride = width * bpp
    expected_len = height * (1 + stride)
    if len(raw) != expected_len:
        raise AssertionError(f"decompressed IDAT length {len(raw)} != expected {expected_len}")

    out = bytearray(height * stride)
    prev = bytes(stride)
    for y in range(height):
        row_start = y * (1 + stride)
        ftype = raw[row_start]
        filt = raw[row_start + 1:row_start + 1 + stride]
        recon = bytearray(stride)
        if ftype == 0:
            recon[:] = filt
        elif ftype == 1:
            for x in range(stride):
                a = recon[x - bpp] if x >= bpp else 0
                recon[x] = (filt[x] + a) & 0xFF
        elif ftype == 2:
            for x in range(stride):
                recon[x] = (filt[x] + prev[x]) & 0xFF
        elif ftype == 3:
            for x in range(stride):
                a = recon[x - bpp] if x >= bpp else 0
                b = prev[x]
                recon[x] = (filt[x] + ((a + b) >> 1)) & 0xFF
        elif ftype == 4:
            for x in range(stride):
                a = recon[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                recon[x] = (filt[x] + _paeth(a, b, c)) & 0xFF
        else:
            raise AssertionError(f"unsupported PNG scanline filter type {ftype} at row {y}")
        out[y * stride:(y + 1) * stride] = recon
        prev = bytes(recon)

    return width, height, bytes(out)


# ---------------------------------------------------------------------------
# Independent binary glTF 2.0 (.glb) validator
# ---------------------------------------------------------------------------

def parse_and_validate_glb(data: bytes) -> tuple[dict, bytes]:
    """Independently validate a binary glTF 2.0 (.glb) container and return (gltf_json, bin_chunk)."""
    if len(data) < 20:
        raise AssertionError(f"GLB too short: {len(data)} bytes")
    magic, version, total_len = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise AssertionError(f"bad GLB magic: {magic!r}")
    if version != 2:
        raise AssertionError(f"bad GLB version: {version}")
    if total_len != len(data):
        raise AssertionError(f"GLB length mismatch: header={total_len}, actual={len(data)}")
    if total_len % 4 != 0:
        raise AssertionError(f"GLB total length {total_len} not 4-byte aligned")

    json_len, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != b"JSON":
        raise AssertionError(f"chunk 0 must be JSON, got {json_type!r}")
    if json_len % 4 != 0:
        raise AssertionError(f"JSON chunk length {json_len} not 4-byte aligned")
    json_end = 20 + json_len
    if json_end > total_len:
        raise AssertionError("JSON chunk extends past EOF")
    gltf = json.loads(data[20:json_end].decode("utf-8"))

    bin_data = b""
    if json_end < total_len:
        if json_end + 8 > total_len:
            raise AssertionError("truncated BIN chunk header")
        bin_len, bin_type = struct.unpack_from("<I4s", data, json_end)
        if bin_type != b"BIN\x00":
            raise AssertionError(f"chunk 1 must be BIN\\x00, got {bin_type!r}")
        if bin_len % 4 != 0:
            raise AssertionError(f"BIN chunk length {bin_len} not 4-byte aligned")
        bin_end = json_end + 8 + bin_len
        if bin_end != total_len:
            raise AssertionError(f"BIN chunk end {bin_end} != total_len {total_len}")
        bin_data = data[json_end + 8:bin_end]

    if gltf.get("asset", {}).get("version") != "2.0":
        raise AssertionError(f"unexpected glTF asset version: {gltf.get('asset')}")

    for idx, bv in enumerate(gltf.get("bufferViews", [])):
        off = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        if off % 4 != 0:
            raise AssertionError(f"bufferView[{idx}] byteOffset={off} is not 4-byte aligned")
        if off + length > len(bin_data):
            raise AssertionError(f"bufferView[{idx}] overflows BIN chunk ({off}+{length} > {len(bin_data)})")

    return gltf, bin_data


def read_accessor_floats(gltf: dict, bin_data: bytes, acc_idx: int) -> list[tuple[float, ...]]:
    """Extract float32 vectors from a glTF accessor."""
    acc = gltf["accessors"][acc_idx]
    assert acc["componentType"] == 5126, f"expected FLOAT(5126), got {acc['componentType']}"
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    bv = gltf["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or (comps * 4)
    fmt = f"<{comps}f"
    return [struct.unpack_from(fmt, bin_data, base + i * stride) for i in range(acc["count"])]


def read_accessor_indices(gltf: dict, bin_data: bytes, acc_idx: int) -> list[int]:
    """Extract scalar integer indices from a glTF accessor."""
    acc = gltf["accessors"][acc_idx]
    fmt_map = {5121: "<B", 5123: "<H", 5125: "<I"}
    assert acc["componentType"] in fmt_map, f"unexpected index componentType {acc['componentType']}"
    fmt = fmt_map[acc["componentType"]]
    size = struct.calcsize(fmt)
    bv = gltf["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or size
    return [struct.unpack_from(fmt, bin_data, base + i * stride)[0] for i in range(acc["count"])]


# ---------------------------------------------------------------------------
# 1. tools/dds_png.py tests
# ---------------------------------------------------------------------------

class DdsPngTest(unittest.TestCase):
    def test_legacy_bgra8_exact_rgba(self):
        expected = bytes([
            10, 20, 30, 255,
            250, 128, 5, 180,
            0, 255, 127, 64,
            77, 88, 99, 0,
        ])
        dds = make_bgra8_dds(2, 2, expected)
        hdr = dds_png.parse_dds_header(dds)
        self.assertEqual(hdr["width"], 2)
        self.assertEqual(hdr["height"], 2)
        self.assertEqual(hdr["format"], "BGRA8")
        self.assertEqual(hdr["data_offset"], 128)

        w, h, rgba = dds_png.decode_dds(dds)
        self.assertEqual((w, h), (2, 2))
        self.assertEqual(rgba, expected)

    def test_dx10_rgba8_and_bgrx8_alpha_255(self):
        rgba_in = bytes([
            12, 34, 56, 78,
            90, 111, 122, 255,
            255, 0, 128, 1,
            0, 255, 64, 0,
        ])
        dds_rgba = make_dx10_dds(2, 2, 28, rgba_in, compressed=False)
        hdr_rgba = dds_png.parse_dds_header(dds_rgba)
        self.assertEqual((hdr_rgba["format"], hdr_rgba["dxgi_format"], hdr_rgba["data_offset"]), ("RGBA8", 28, 148))
        w, h, rgba = dds_png.decode_dds(dds_rgba)
        self.assertEqual((w, h), (2, 2))
        self.assertEqual(rgba, rgba_in)

        # BGRX8 (dxgi_format=88): X byte is non-255 (0, 17, 128, 254); decoded alpha MUST be 255.
        bgrx_in = bytes([
            10, 20, 30, 0,
            40, 50, 60, 17,
            70, 80, 90, 128,
            100, 110, 120, 254,
        ])
        expected_bgrx = bytes([
            30, 20, 10, 255,
            60, 50, 40, 255,
            90, 80, 70, 255,
            120, 110, 100, 255,
        ])
        dds_bgrx = make_dx10_dds(2, 2, 88, bgrx_in, compressed=False)
        w2, h2, rgba2 = dds_png.decode_dds(dds_bgrx)
        self.assertEqual((w2, h2), (2, 2))
        self.assertEqual(rgba2, expected_bgrx)

    def test_bc1_opaque_4color(self):
        # c0 = pure red (255,0,0), c1 = pure blue (0,0,255); 255 % 3 == 0 so interpolation is exact.
        c0, c1 = 0xF800, 0x001F
        self.assertGreater(c0, c1)
        codes = [0, 1, 2, 3, 3, 2, 1, 0, 1, 3, 0, 2, 2, 0, 3, 1]
        palette = [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (170, 0, 85, 255),
            (85, 0, 170, 255),
        ]
        dds = make_fourcc_dds(4, 4, b"DXT1", pack_bc1_block(c0, c1, codes))
        w, h, rgba = dds_png.decode_dds(dds)
        self.assertEqual((w, h), (4, 4))
        expected = b"".join(bytes(palette[c]) for c in codes)
        self.assertEqual(rgba, expected)

    def test_bc1_transparent_3color(self):
        # c0 = (0,0,0), c1 = 0x8410 -> (132, 130, 132), c0 <= c1 -> exact mid (66, 65, 66), code 3 -> (0,0,0,0).
        c0, c1 = 0x0000, 0x8410
        self.assertLessEqual(c0, c1)
        codes = [0, 1, 2, 3, 3, 3, 2, 1, 0, 2, 3, 1, 1, 0, 3, 2]
        palette = [
            (0, 0, 0, 255),
            (132, 130, 132, 255),
            (66, 65, 66, 255),
            (0, 0, 0, 0),
        ]
        dds = make_fourcc_dds(4, 4, b"DXT1", pack_bc1_block(c0, c1, codes))
        w, h, rgba = dds_png.decode_dds(dds)
        self.assertEqual((w, h), (4, 4))
        expected = b"".join(bytes(palette[c]) for c in codes)
        self.assertEqual(rgba, expected)

        # Also check c0 == c1 still enters 3-color + transparent mode (code 3 -> (0,0,0,0)).
        dds_eq = make_dx10_dds(4, 4, 71, pack_bc1_block(0xF800, 0xF800, [0, 3] * 8), compressed=True)
        w_eq, h_eq, rgba_eq = dds_png.decode_dds(dds_eq)
        self.assertEqual((w_eq, h_eq), (4, 4))
        self.assertEqual(rgba_eq[0:4], bytes((255, 0, 0, 255)))
        self.assertEqual(rgba_eq[4:8], bytes((0, 0, 0, 0)))

    def test_bc1_odd_3x3_size(self):
        # Sentinel code 3 fills column 3 and row 3; decoded 3x3 image must clip them cleanly.
        c0, c1 = 0xF800, 0x001F
        block_codes = [
            0, 1, 2, 3,
            1, 2, 0, 3,
            2, 0, 1, 3,
            3, 3, 3, 3,
        ]
        palette = [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (170, 0, 85, 255),
            (85, 0, 170, 255),
        ]
        dds = make_fourcc_dds(3, 3, b"DXT1", pack_bc1_block(c0, c1, block_codes))
        w, h, rgba = dds_png.decode_dds(dds)
        self.assertEqual((w, h), (3, 3))
        self.assertEqual(len(rgba), 3 * 3 * 4)
        visible_codes = [0, 1, 2, 1, 2, 0, 2, 0, 1]
        expected = b"".join(bytes(palette[c]) for c in visible_codes)
        self.assertEqual(rgba, expected)

    def test_bc2_explicit_4bit_alpha(self):
        alphas4 = [0x0, 0x8, 0xF, 0x1, 0x2, 0x4, 0x7, 0xA, 0xC, 0xE, 0x0, 0x8, 0xF, 0x5, 0x9, 0xB]
        color_codes = [0, 1, 2, 3] * 4
        c0, c1 = 0xF800, 0x001F
        rgb_pal = [(255, 0, 0), (0, 0, 255), (170, 0, 85), (85, 0, 170)]
        dds = make_fourcc_dds(4, 4, b"DXT3", pack_bc2_block(alphas4, c0, c1, color_codes))
        w, h, rgba = dds_png.decode_dds(dds)
        self.assertEqual((w, h), (4, 4))
        expected = b"".join(bytes((*rgb_pal[cc], a4 * 17)) for a4, cc in zip(alphas4, color_codes))
        self.assertEqual(rgba, expected)
        # Explicitly pin 0x0 -> 0, 0x8 -> 136, 0xF -> 255
        self.assertEqual((rgba[3], rgba[7], rgba[11]), (0, 136, 255))

    def test_bc3_interpolated_alpha_both_modes(self):
        c0, c1 = 0x07E0, 0x0000  # pure green at code 0
        color_codes = [0] * 16

        with self.subTest("a0 > a1 (8-value palette)"):
            # 210 / 7 = 30 exact step -> [210, 0, 180, 150, 120, 90, 60, 30]
            a_codes = [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
            a_pal = [210, 0, 180, 150, 120, 90, 60, 30]
            dds = make_fourcc_dds(4, 4, b"DXT5", pack_bc3_block(210, 0, a_codes, c0, c1, color_codes))
            w, h, rgba = dds_png.decode_dds(dds)
            self.assertEqual((w, h), (4, 4))
            expected = b"".join(bytes((0, 255, 0, a_pal[c])) for c in a_codes)
            self.assertEqual(rgba, expected)

        with self.subTest("a0 <= a1 (6-value palette + literal 0 and 255)"):
            # (200 - 50) / 5 = 30 exact step -> [50, 200, 80, 110, 140, 170, 0, 255]
            a_codes = [0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 0, 1, 2, 3, 4, 5]
            a_pal = [50, 200, 80, 110, 140, 170, 0, 255]
            dds = make_fourcc_dds(4, 4, b"DXT5", pack_bc3_block(50, 200, a_codes, c0, c1, color_codes))
            w, h, rgba = dds_png.decode_dds(dds)
            self.assertEqual((w, h), (4, 4))
            expected = b"".join(bytes((0, 255, 0, a_pal[c])) for c in a_codes)
            self.assertEqual(rgba, expected)
            self.assertEqual(rgba[6 * 4 + 3], 0)
            self.assertEqual(rgba[7 * 4 + 3], 255)

    def test_bc4_single_channel_grayscale(self):
        codes = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        pal = [210, 0, 180, 150, 120, 90, 60, 30]
        block = pack_bc4_block(210, 0, codes)
        expected = b"".join(bytes((pal[c], pal[c], pal[c], 255)) for c in codes)

        for label, dds in [
            ("DX10 BC4_UNORM", make_dx10_dds(4, 4, 80, block, compressed=True)),
            ("Legacy ATI1", make_fourcc_dds(4, 4, b"ATI1", block)),
        ]:
            with self.subTest(label):
                w, h, rgba = dds_png.decode_dds(dds)
                self.assertEqual((w, h), (4, 4))
                self.assertEqual(rgba, expected)

    def test_bc5_normal_map_z_reconstruction(self):
        # Pixels 0..14: r=128, g=128 -> flat tangent normal, reconstructed z >= 254.
        # Pixel 15:     r=255, g=128 -> steep normal (nx=1, ny~0), reconstructed z < 200.
        r_codes = [0] * 15 + [1]
        g_codes = [0] * 16
        block = pack_bc5_block(128, 255, r_codes, 128, 128, g_codes)

        for label, dds in [
            ("DX10 BC5_UNORM", make_dx10_dds(4, 4, 83, block, compressed=True)),
            ("Legacy ATI2", make_fourcc_dds(4, 4, b"ATI2", block)),
        ]:
            with self.subTest(label):
                w, h, rgba = dds_png.decode_dds(dds)
                self.assertEqual((w, h), (4, 4))
                for i in range(15):
                    r, g, b, a = rgba[i * 4:i * 4 + 4]
                    self.assertEqual((r, g, a), (128, 128, 255))
                    self.assertGreaterEqual(b, 254)
                r15, g15, b15, a15 = rgba[15 * 4:16 * 4]
                self.assertEqual((r15, g15, a15), (255, 128, 255))
                self.assertLess(b15, 200)

    def test_downscale_rgba_4x4_to_2x2_exact(self):
        # 4 quadrants whose 2x2 channel sums are exact multiples of 4.
        grid = [
            (0, 40, 80, 200), (4, 44, 84, 204), (100, 0, 200, 252), (100, 0, 200, 252),
            (8, 48, 88, 208), (12, 52, 92, 212), (200, 40, 100, 252), (200, 40, 100, 252),
            (10, 20, 30, 40), (30, 40, 50, 60), (255, 255, 255, 255), (255, 255, 255, 255),
            (50, 60, 70, 80), (70, 80, 90, 100), (255, 255, 255, 255), (255, 255, 255, 255),
        ]
        rgba4 = b"".join(bytes(p) for p in grid)
        w, h, out = dds_png.downscale_rgba(4, 4, rgba4, 2)
        self.assertEqual((w, h), (2, 2))
        expected = bytes([
            6, 46, 86, 206,
            150, 20, 150, 252,
            47, 57, 67, 70,         # colours weighted by alpha (dds_png, 62804d0d): sum(c*a) / sum(a)
            255, 255, 255, 255,
        ])
        self.assertEqual(out, expected)

        # No-op when max_size == 0
        self.assertEqual(dds_png.downscale_rgba(4, 4, rgba4, 0), (4, 4, rgba4))

    def test_encode_png_and_dds_to_png_roundtrip_with_independent_validator(self):
        pixels = bytes([
            255, 0, 0, 255,
            0, 255, 0, 128,
            0, 0, 255, 64,
            12, 34, 56, 0,
            200, 150, 100, 255,
            10, 20, 30, 240,
        ])
        png = dds_png.encode_png(3, 2, pixels)
        self.assertEqual(parse_and_validate_png(png), (3, 2, pixels))

        # dds_to_png_bytes + dds_file_to_png_bytes + max_size downscale
        rgba4 = bytes(i % 256 for i in range(4 * 4 * 4))
        dds = make_bgra8_dds(4, 4, rgba4)
        png_full = dds_png.dds_to_png_bytes(dds, max_size=0)
        self.assertEqual(parse_and_validate_png(png_full), (4, 4, rgba4))

        png_small = dds_png.dds_to_png_bytes(dds, max_size=2)
        ws, hs, _ = parse_and_validate_png(png_small)
        self.assertEqual((ws, hs), (2, 2))

        with tempfile.TemporaryDirectory() as tmp:
            dds_path = os.path.join(tmp, "sample.dds")
            with open(dds_path, "wb") as f:
                f.write(dds)
            png_from_file = dds_png.dds_file_to_png_bytes(dds_path, max_size=0)
            self.assertEqual(parse_and_validate_png(png_from_file), (4, 4, rgba4))

    def test_invalid_dds_raises_dds_error(self):
        with self.subTest("invalid magic"):
            with self.assertRaises(dds_png.DdsError):
                dds_png.decode_dds(b"NOPE" + b"\x00" * 124)

        with self.subTest("truncated header"):
            with self.assertRaises(dds_png.DdsError):
                dds_png.decode_dds(b"DDS " + b"\x00" * 40)

        with self.subTest("truncated uncompressed payload"):
            good = make_bgra8_dds(4, 4, bytes(4 * 4 * 4))
            with self.assertRaises(dds_png.DdsError):
                dds_png.decode_dds(good[:-10])

        with self.subTest("truncated BC1 payload"):
            good_bc1 = make_fourcc_dds(4, 4, b"DXT1", pack_bc1_block(0xF800, 0x001F, [0] * 16))
            with self.assertRaises(dds_png.DdsError):
                dds_png.decode_dds(good_bc1[:-3])

        with self.subTest("unsupported DXGI format"):
            bad_dxgi = make_dx10_dds(4, 4, 9999, bytes(64), compressed=False)
            with self.assertRaises(dds_png.DdsError):
                dds_png.decode_dds(bad_dxgi)

    def test_bc4_and_bc5_snorm_signed_decoding(self):
        # r0 = -127 (0x81), r1 = +127 (0x7F) -> s0 <= s1 mode:
        # codes [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 255, 51, 102, 153, 204, 0, 255]
        codes = [0, 1, 2, 3, 4, 5, 6, 7] * 2
        expected_vals = [0, 255, 51, 102, 153, 204, 0, 255] * 2
        block4 = pack_bc4_block(0x81, 0x7F, codes)
        expected_bc4 = b"".join(bytes((v, v, v, 255)) for v in expected_vals)

        for label, dds in [
            ("DX10 BC4_SNORM (81)", make_dx10_dds(4, 4, 81, block4, compressed=True)),
            ("Legacy BC4S", make_fourcc_dds(4, 4, b"BC4S", block4)),
        ]:
            with self.subTest(label):
                hdr = dds_png.parse_dds_header(dds)
                self.assertEqual(hdr["format"], "BC4_SNORM")
                w, h, rgba = dds_png.decode_dds(dds)
                self.assertEqual((w, h), (4, 4))
                self.assertEqual(rgba, expected_bc4)

        # -128 (0x80) clamps to -127 (-1.0 -> 0); s0=0, s1=-128 -> s0 > s1 mode
        block4_clamp = pack_bc4_block(0, 0x80, [0, 1] + [0] * 14)
        _, _, rgba_clamp = dds_png.decode_dds(make_dx10_dds(4, 4, 81, block4_clamp, compressed=True))
        self.assertEqual(rgba_clamp[0], 128)  # 0.0 -> 128
        self.assertEqual(rgba_clamp[4], 0)    # -128 clamped to -127 -> 0

        # BC5_SNORM: signed (0, 0) -> flat normal (128, 128, 255, 255); (+127, 0) -> steep normal (255, 128, <200, 255)
        r_codes = [0] * 15 + [1]
        g_codes = [0] * 16
        block5 = pack_bc5_block(0, 0x7F, r_codes, 0, 0, g_codes)
        for label, dds in [
            ("DX10 BC5_SNORM (84)", make_dx10_dds(4, 4, 84, block5, compressed=True)),
            ("Legacy BC5S", make_fourcc_dds(4, 4, b"BC5S", block5)),
        ]:
            with self.subTest(label):
                hdr = dds_png.parse_dds_header(dds)
                self.assertEqual(hdr["format"], "BC5_SNORM")
                w, h, rgba = dds_png.decode_dds(dds)
                self.assertEqual((w, h), (4, 4))
                r0, g0, b0, a0 = rgba[0:4]
                self.assertEqual((r0, g0, a0), (128, 128, 255))
                self.assertGreaterEqual(b0, 254)
                r15, g15, b15, a15 = rgba[15 * 4:16 * 4]
                self.assertEqual((r15, g15, a15), (255, 128, 255))
                self.assertLess(b15, 200)

    def test_legacy_rgb_without_alphapixels_forces_opaque_alpha(self):
        # DDPF_RGB (0x40) WITHOUT DDPF_ALPHAPIXELS (0x1), even with stray non-zero dwABitMask,
        # must ignore alpha bits and force alpha = 255.
        raw = bytearray(make_bgra8_dds(2, 1, bytes([10, 20, 30, 0, 40, 50, 60, 12])))
        struct.pack_into("<I", raw, 80, 0x40)  # DDPF_RGB only (no DDPF_ALPHAPIXELS)
        hdr = dds_png.parse_dds_header(bytes(raw))
        self.assertEqual(hdr["format"], "BGRX8")
        w, h, rgba = dds_png.decode_dds(bytes(raw))
        self.assertEqual((w, h), (2, 1))
        self.assertEqual(rgba, bytes([10, 20, 30, 255, 40, 50, 60, 255]))

    def test_bc2_and_bc3_always_use_4color_rgb_even_when_c0_le_c1(self):
        # c0 = pure blue (0,0,255), c1 = pure red (255,0,0), so c0 < c1.
        # Unlike BC1, BC2/BC3 MUST still interpolate 4-color RGB (code 3 -> (170, 0, 85), NOT transparent black).
        c0, c1 = 0x001F, 0xF800
        self.assertLess(c0, c1)
        color_codes = [0, 1, 2, 3] * 4
        rgb_pal = [(0, 0, 255), (255, 0, 0), (85, 0, 170), (170, 0, 85)]

        bc2 = make_fourcc_dds(4, 4, b"DXT3", pack_bc2_block([0xF] * 16, c0, c1, color_codes))
        _, _, rgba2 = dds_png.decode_dds(bc2)
        self.assertEqual(rgba2, b"".join(bytes((*rgb_pal[c], 255)) for c in color_codes))

        bc3 = make_fourcc_dds(4, 4, b"DXT5", pack_bc3_block(255, 255, [0] * 16, c0, c1, color_codes))
        _, _, rgba3 = dds_png.decode_dds(bc3)
        self.assertEqual(rgba3, b"".join(bytes((*rgb_pal[c], 255)) for c in color_codes))

    def test_downscale_rgba_odd_and_1px_dimensions(self):
        for (w, h, max_s, exp_w, exp_h) in [
            (3, 3, 2, 1, 1),
            (1, 5, 2, 1, 2),
            (5, 1, 2, 2, 1),
            (1, 1, 1, 1, 1),
            (4, 4, 1, 1, 1),
        ]:
            with self.subTest(size=(w, h), max_size=max_s):
                buf = bytes([80, 120, 160, 255] * (w * h))
                nw, nh, out = dds_png.downscale_rgba(w, h, buf, max_s)
                self.assertEqual((nw, nh), (exp_w, exp_h))
                self.assertEqual(len(out), exp_w * exp_h * 4)
                self.assertEqual(out, bytes([80, 120, 160, 255] * (exp_w * exp_h)))


# ---------------------------------------------------------------------------
# 2. tools/glb_writer.py tests
# ---------------------------------------------------------------------------

class GlbWriterTest(unittest.TestCase):
    def test_negative_obj_scale_preserves_front_faces_and_normals(self):
        vertices = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\n"
        for face in ("f 1 2 3", "f 1//1 2//1 3//1"):
            with self.subTest(face=face):
                prim, = glb_writer.parse_obj(vertices + face, is_path=False, up_axis="y", scale=-2)
                corners = [prim["positions"][index] for index in prim["indices"]]
                self.assertEqual(corners, [(0, 0, 0), (0, -2, 0), (-2, 0, 0)])
                self.assertEqual(prim["normals"], [(0, 0, -1)] * 3)

    def test_parse_obj_quads_negative_indices_normals_uv_flip_zup_and_groups(self):
        # Group 0: quad in XY plane (Z-up) using positive indices, no vn -> auto unit normals.
        # Group 1: single triangle containing (1, 2, 3) using negative indices.
        obj = "\n".join([
            "g mesh_0",
            "v 0 0 0",
            "v 2 0 0",
            "v 2 2 0",
            "v 0 2 0",
            "vt 0.0 0.2",
            "vt 1.0 0.2",
            "vt 1.0 0.8",
            "vt 0.0 0.8",
            "f 1/1 2/2 3/3 4/4",
            "g mesh_1",
            "v 1 2 3",
            "v 3 2 3",
            "v 1 4 3",
            "vt 0.25 0.25",
            "vt 0.75 0.25",
            "vt 0.25 0.75",
            "f -3/-3 -2/-2 -1/-1",
            "",
        ])
        prims = glb_writer.parse_obj(obj, is_path=False, up_axis="z", offset=(0, 0, 0), scale=1.0)
        self.assertEqual(len(prims), 2)
        self.assertEqual(prims[0]["group_index"], 0)
        self.assertEqual(prims[1]["group_index"], 1)

        # Quad fan-triangulated -> 2 tris = 6 indices
        self.assertEqual(len(prims[0]["indices"]), 6)

        # Auto-computed unit normals for missing vn
        self.assertEqual(len(prims[0]["normals"]), len(prims[0]["positions"]))
        for nx, ny, nz in prims[0]["normals"]:
            self.assertAlmostEqual(math.sqrt(nx * nx + ny * ny + nz * nz), 1.0, places=5)

        # UV V-flip: vt (u, v) -> (u, 1 - v)
        uv0_rounded = {(round(u, 4), round(v, 4)) for u, v in prims[0]["uvs"]}
        self.assertEqual(uv0_rounded, {(0.0, 0.8), (1.0, 0.8), (1.0, 0.2), (0.0, 0.2)})

        # Z-up -> Y-up transform: (1, 2, 3) -> (1, 3, -2)
        pos1_rounded = {(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in prims[1]["positions"]}
        self.assertEqual(pos1_rounded, {(1.0, 3.0, -2.0), (3.0, 3.0, -2.0), (1.0, 3.0, -4.0)})

        # Negative indices resolved -> 1 tri = 3 indices
        self.assertEqual(len(prims[1]["indices"]), 3)

        # compute_bounds matches exact transformed extents across both groups
        bounds = glb_writer.compute_bounds(prims)
        self.assertIsNotNone(bounds)
        for got, exp in zip(bounds, [0.0, 0.0, -4.0, 3.0, 3.0, 0.0]):
            self.assertAlmostEqual(got, exp, places=5)

        # Also verify is_path=True with offset and scale
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "mesh.obj")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("v 1 2 3\nv 2 2 3\nv 1 3 3\nf 1 2 3\n")
            p_file = glb_writer.parse_obj(obj_path, is_path=True, up_axis="z", offset=(10.0, 20.0, 30.0), scale=2.0)
            self.assertEqual(len(p_file), 1)
            pts = {(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in p_file[0]["positions"]}
            # (1,2,3) Z-up -> (1,3,-2) * 2 + (10,20,30) = (12, 26, 26)
            self.assertIn((12.0, 26.0, 26.0), pts)
            b = glb_writer.compute_bounds(p_file)
            self.assertAlmostEqual(b[3] - b[0], 2.0, places=5)

    def test_textured_glb_dedup_alignment_bounds_and_png_roundtrip(self):
        obj = "\n".join([
            "g mesh_0",
            "usemtl mat_a",
            "v 0 0 0",
            "v 2 0 0",
            "v 0 0 4",
            "vt 0 0",
            "vt 1 0",
            "vt 0 1",
            "f 1/1 2/2 3/3",
            "g mesh_1",
            "usemtl mat_b",
            "v 5 0 0",
            "v 7 0 0",
            "v 5 0 4",
            "vt 0 0",
            "vt 1 0",
            "vt 0 1",
            "f 4/4 5/5 6/6",
            "",
        ])
        prims = glb_writer.parse_obj(obj, is_path=False, up_axis="z")
        self.assertEqual([p["material"] for p in prims], ["mat_a", "mat_b"])

        png_bytes = dds_png.encode_png(2, 2, bytes([
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 0, 255,
        ]))
        materials = {
            "mat_a": {"name": "mat_a", "png_bytes": png_bytes},
            "mat_b": {"name": "mat_b", "png_bytes": png_bytes},
        }

        glb_data = glb_writer.build_glb(prims, materials=materials)
        gltf, bin_data = parse_and_validate_glb(glb_data)

        # Deduplication: two primitives/materials sharing identical png_bytes -> 1 image & 1 texture
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["textures"]), 1)
        self.assertEqual(len(gltf["samplers"]), 1)
        self.assertEqual(gltf["images"][0]["mimeType"], "image/png")

        # Image bufferView MUST NOT have "target" and MUST be 4-byte aligned
        img_bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        self.assertNotIn("target", img_bv)
        self.assertEqual(img_bv.get("byteOffset", 0) % 4, 0)

        # Exact round-trip of embedded PNG bytes from BIN chunk
        off = img_bv.get("byteOffset", 0)
        embedded_png = bin_data[off:off + img_bv["byteLength"]]
        self.assertEqual(embedded_png, png_bytes)
        self.assertEqual(parse_and_validate_png(embedded_png)[0:2], (2, 2))

        # Both materials point to the single deduplicated texture index 0
        self.assertEqual(len(gltf["materials"]), 2)
        for mat in gltf["materials"]:
            self.assertEqual(mat["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0)

        # Check UNSIGNED_SHORT (5123) indices and exact POSITION min/max on every primitive
        mesh_prims = [p for m in gltf["meshes"] for p in m["primitives"]]
        self.assertEqual(len(mesh_prims), 2)
        for mp in mesh_prims:
            idx_acc = gltf["accessors"][mp["indices"]]
            self.assertEqual(idx_acc["componentType"], 5123)
            pos_acc = gltf["accessors"][mp["attributes"]["POSITION"]]
            pts = read_accessor_floats(gltf, bin_data, mp["attributes"]["POSITION"])
            exp_min = [min(p[k] for p in pts) for k in range(3)]
            exp_max = [max(p[k] for p in pts) for k in range(3)]
            for k in range(3):
                self.assertAlmostEqual(pos_acc["min"][k], exp_min[k], places=5)
                self.assertAlmostEqual(pos_acc["max"][k], exp_max[k], places=5)

        # write_glb + read_glb_summary roundtrip
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.glb")
            stats = glb_writer.write_glb(out_path, prims, materials=materials)
            self.assertEqual(stats["primitives"], 2)
            self.assertEqual(stats["textures"], 1)
            self.assertEqual(stats["triangles"], 2)
            with open(out_path, "rb") as f:
                disk_data = f.read()
            self.assertEqual(disk_data, glb_data)
            summary = glb_writer.read_glb_summary(disk_data)
            self.assertEqual(summary["version"], 2)
            self.assertEqual(summary["total_bytes"], len(disk_data))
            self.assertEqual(len(summary["json"]["images"]), 1)

    def test_untextured_glb_has_no_images_and_valid_clay_material(self):
        prims = glb_writer.parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", is_path=False)
        glb_data = glb_writer.build_glb(prims, materials=None)
        gltf, _ = parse_and_validate_glb(glb_data)
        self.assertNotIn("images", gltf)
        self.assertNotIn("textures", gltf)
        self.assertNotIn("samplers", gltf)
        self.assertEqual(len(gltf["materials"]), 1)
        pbr = gltf["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(len(pbr["baseColorFactor"]), 4)
        self.assertNotIn("baseColorTexture", pbr)

    def test_large_primitive_65536_vertices_uses_unsigned_int_indices(self):
        # 65536 unique vertex positions referenced by 21846 triangles -> 65536 vertices in one primitive.
        lines = [f"v {i & 255} {(i >> 8) & 255} 0" for i in range(65536)]
        lines.append("vt 0 0")
        lines.append("vn 0 0 1")
        for k in range(0, 65535, 3):
            lines.append(f"f {k + 1}/1/1 {k + 2}/1/1 {k + 3}/1/1")
        lines.append("f 65534/1/1 65535/1/1 65536/1/1")
        obj = "\n".join(lines) + "\n"

        prims = glb_writer.parse_obj(obj, is_path=False, up_axis="z")
        self.assertEqual(len(prims), 1)
        glb_data = glb_writer.build_glb(prims)
        gltf, bin_data = parse_and_validate_glb(glb_data)
        mp = gltf["meshes"][0]["primitives"][0]
        pos_acc = gltf["accessors"][mp["attributes"]["POSITION"]]
        idx_acc = gltf["accessors"][mp["indices"]]
        self.assertEqual(pos_acc["count"], 65536)
        self.assertEqual(idx_acc["componentType"], 5125)  # UNSIGNED_INT
        indices = read_accessor_indices(gltf, bin_data, mp["indices"])
        self.assertEqual(max(indices), 65535)

    def test_empty_primitives_and_corrupt_glb_raise_glb_error(self):
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([])

        for label, bad in [
            ("empty", b""),
            ("short", b"glTF\x02\x00\x00\x00"),
            ("bad magic", b"NOPE" + struct.pack("<II", 2, 20) + struct.pack("<I4s", 0, b"JSON")),
            ("bad version", b"glTF" + struct.pack("<II", 1, 20) + struct.pack("<I4s", 0, b"JSON")),
            ("bad length", b"glTF" + struct.pack("<II", 2, 9999) + struct.pack("<I4s", 0, b"JSON")),
            ("bad chunk0", b"glTF" + struct.pack("<II", 2, 20) + struct.pack("<I4s", 0, b"BIN\x00")),
        ]:
            with self.subTest(label):
                with self.assertRaises(glb_writer.GlbError):
                    glb_writer.read_glb_summary(bad)

        # read_glb_summary rejects bool byteLength and bufferView exceeding buffers[0].byteLength
        for label, payload_doc in [
            ("bool buffer byteLength", {"asset": {"version": "2.0"}, "buffers": [{"byteLength": True}]}),
            (
                "bufferView exceeds declared buffer",
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"byteLength": 2}],
                    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}],
                },
            ),
        ]:
            with self.subTest(label):
                jb = json.dumps(payload_doc).encode("utf-8")
                jb += b" " * ((-len(jb)) % 4)
                bb = b"\x00\x00\x00\x00"
                total = 12 + 8 + len(jb) + 8 + len(bb)
                raw = (
                    struct.pack("<4sII", b"glTF", 2, total)
                    + struct.pack("<II", len(jb), 0x4E4F534A)
                    + jb
                    + struct.pack("<II", len(bb), 0x004E4942)
                    + bb
                )
                with self.assertRaises(glb_writer.GlbError):
                    glb_writer.read_glb_summary(raw)

    def test_obj_object_then_groups_preserves_group_indices_and_comment_stripping(self):
        obj = "\n".join([
            "o building # top object",
            "g mesh_0 # first group",
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 3",
            "g #comment_only_group",
            "v 2 0 0",
            "v 3 0 0",
            "v 2 1 0",
            "f 4 5 6",
            "",
        ])
        prims = glb_writer.parse_obj(obj, is_path=False, up_axis="z")
        self.assertEqual([p["group_index"] for p in prims], [0, 1])
        self.assertEqual(prims[0]["name"], "mesh_0")
        self.assertEqual(prims[1]["name"], "")

    def test_textured_material_without_uvs_omits_base_color_texture_and_unused_png(self):
        # glTF 2.0 §3.8.2: a primitive without TEXCOORD_0 MUST NOT bind a material with baseColorTexture.
        prims = glb_writer.parse_obj("usemtl wall\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", is_path=False)
        self.assertEqual(prims[0]["uvs"], [])
        png_bytes = dds_png.encode_png(2, 2, bytes([255, 0, 0, 255] * 4))
        glb_data = glb_writer.build_glb(prims, materials={"wall": {"name": "wall", "png_bytes": png_bytes}})
        gltf, _ = parse_and_validate_glb(glb_data)
        self.assertNotIn("images", gltf)
        self.assertNotIn("textures", gltf)
        self.assertNotIn("samplers", gltf)
        self.assertNotIn("baseColorTexture", gltf["materials"][0]["pbrMetallicRoughness"])

        # Non-empty mismatched UVs and out-of-range base_color_factor must raise GlbError.
        bad_uv_prim = dict(prims[0], uvs=[(0.0, 0.0)])
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([bad_uv_prim])
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb(prims, materials={"wall": {"base_color_factor": [1.5, 0.0, 0.0, 1.0]}})

    def test_65535_vertex_boundary_uses_unsigned_short_without_restart_index(self):
        # 65535 vertices -> max vertex index is 65534 (< 0xFFFF restart index), so UNSIGNED_SHORT (5123) is valid.
        positions = [(float(i & 255), float((i >> 8) & 255), 0.0) for i in range(65535)]
        normals = [(0.0, 1.0, 0.0)] * 65535
        indices = list(range(65535))  # 21845 triangles, divisible by 3
        glb_data = glb_writer.build_glb([{"positions": positions, "normals": normals, "uvs": [], "indices": indices}])
        gltf, bin_data = parse_and_validate_glb(glb_data)
        mp = gltf["meshes"][0]["primitives"][0]
        self.assertEqual(gltf["accessors"][mp["attributes"]["POSITION"]]["count"], 65535)
        self.assertEqual(gltf["accessors"][mp["indices"]]["componentType"], 5123)
        decoded_idx = read_accessor_indices(gltf, bin_data, mp["indices"])
        self.assertEqual(max(decoded_idx), 65534)


# ---------------------------------------------------------------------------
# 3. tools/world_build.py tests
# ---------------------------------------------------------------------------

def _cube_obj(cx: float, cy: float, cz: float, half: float = 1.0, groups: int = 1) -> str:
    """Synthetic Z-up cube(s) OBJ with UVs and normals."""
    lines = ["vn 0 0 1", "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    v_base = 1
    for g in range(groups):
        gx = cx + g * (half * 2.5)
        x0, x1 = gx - half, gx + half
        y0, y1 = cy - half, cy + half
        z0, z1 = cz - half, cz + half
        lines.append(f"g mesh_{g}")
        verts = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        for vx, vy, vz in verts:
            lines.append(f"v {vx} {vy} {vz}")
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (1, 2, 6, 5), (3, 0, 4, 7),
        ]
        for a, b, c, d in faces:
            lines.append(
                f"f {v_base + a}/1/1 {v_base + b}/2/1 {v_base + c}/3/1 {v_base + d}/4/1"
            )
        v_base += 8
    return "\n".join(lines) + "\n"


def _write_synthetic_assets(assets_dir: str, *, origin_overlap: bool = False) -> dict:
    """Create a 100% synthetic <assets_dir> tree matching rpf9dump's layout."""
    models_dir = os.path.join(assets_dir, "arch_a", "models")
    os.makedirs(models_dir, exist_ok=True)
    ttpf_dir = os.path.join(assets_dir, "ttpf", "pool_a")
    os.makedirs(ttpf_dir, exist_ok=True)

    # 1) Embedded BGRA8 DDS for model 1
    wall_rgba = bytes([220, 40, 40, 255] * 16)
    wall_dds = make_bgra8_dds(4, 4, wall_rgba)
    wall_tex_dir = os.path.join(models_dir, "model_one_tex")
    os.makedirs(wall_tex_dir, exist_ok=True)
    wall_dds_path = os.path.join(wall_tex_dir, "000_wall.dds")
    with open(wall_dds_path, "wb") as f:
        f.write(wall_dds)

    # 2) Valid BC1 DDS in TTPF pool for model 2 material 0
    brick_bc1 = make_dx10_dds(4, 4, 71, pack_bc1_block(0xF800, 0x001F, [0, 1, 2, 3] * 4), compressed=True)
    brick_dds_path = os.path.join(ttpf_dir, "000000_brick_diff.dds")
    with open(brick_dds_path, "wb") as f:
        f.write(brick_bc1)

    # 3) Corrupt DDS in TTPF pool for model 2 material 1 (must trigger graceful fallback)
    corrupt_dds_path = os.path.join(ttpf_dir, "000001_corrupt.dds")
    with open(corrupt_dds_path, "wb") as f:
        f.write(b"NOT_A_VALID_DDS_PAYLOAD")

    ttpf_manifest = {
        "src": "synthetic_pool_a.ttpf",
        "Version": 1,
        "Count": 2,
        "Metadata": {},
        "Textures": [
            {
                "Index": 0,
                "Name": "brick_diff",
                "Id": 1,
                "AssetKey": "AABBCCDDEEFF0011",
                "Width": 4,
                "Height": 4,
                "DxgiFormat": 71,
                "Mips": 1,
                "SourceMips": 1,
                "BaseMipLevel": 0,
                "Limitation": None,
                "File": "000000_brick_diff.dds",
            },
            {
                "Index": 1,
                "Name": "corrupt_tex",
                "Id": 2,
                "AssetKey": "AABBCCDDEEFF0022",
                "Width": 4,
                "Height": 4,
                "DxgiFormat": 71,
                "Mips": 1,
                "SourceMips": 1,
                "BaseMipLevel": 0,
                "Limitation": None,
                "File": "000001_corrupt.dds",
            },
        ],
    }
    with open(os.path.join(ttpf_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(ttpf_manifest, f, indent=2)

    if origin_overlap:
        # Both models sit at origin with negative Z (-1..+1) -> overlapping in spatial, min_y < 0 until lifted.
        m1_center = (0.0, 0.0, 0.0)
        m2_center = (0.0, 0.0, 0.0)
        m1_groups, m2_groups = 1, 1
    else:
        # Distant world coords -> 2 distinct spatial chunks (x=10 vs x=350, >256m apart).
        m1_center = (10.0, 10.0, 1.0)
        m2_center = (350.0, 10.0, 1.0)
        m1_groups, m2_groups = 1, 2

    with open(os.path.join(models_dir, "model_one.obj"), "w", encoding="utf-8") as f:
        f.write(_cube_obj(*m1_center, half=1.0, groups=m1_groups))

    with open(os.path.join(models_dir, "model_two.obj"), "w", encoding="utf-8") as f:
        f.write(_cube_obj(*m2_center, half=1.0, groups=m2_groups))

    m2_shaders = [0] if m2_groups == 1 else [0, 1]
    m2_materials = [
        {
            "ShaderIndex": 0,
            "Shader": "11111111",
            "Preset": "22222222",
            "RenderBucket": 0,
            "Textures": [
                {
                    "Param": "2B5170FD",
                    "TextureIndex": -1,
                    "Name": "55667788",
                    "Id64": "AABBCCDDEEFF0011",
                    "Tag": "00000000",
                }
            ],
        },
    ]
    if m2_groups == 2:
        m2_materials.append({
            "ShaderIndex": 1,
            "Shader": "33333333",
            "Preset": "44444444",
            "RenderBucket": 0,
            "Textures": [
                {
                    "Param": "2B5170FD",
                    "TextureIndex": -1,
                    "Name": "99AABBCC",
                    "Id64": "AABBCCDDEEFF0022",
                    "Tag": "00000000",
                }
            ],
        })

    models_manifest = {
        "src": "synthetic_arch_a.rpf",
        "Count": 2,
        "Items": [
            {
                "Offset": 4096,
                "File": "model_one.obj",
                "Name": "model_one",
                "Meshes": m1_groups,
                "Verts": 8 * m1_groups,
                "Tris": 12 * m1_groups,
                "IsFragment": False,
                "HasBoneBindings": False,
                "Min": [m1_center[0] - 1.0, m1_center[1] - 1.0, m1_center[2] - 1.0],
                "Max": [m1_center[0] + 1.0, m1_center[1] + 1.0, m1_center[2] + 1.0],
                "MeshShaders": [0],
                "Embedded": [
                    {
                        "Index": 0,
                        "Name": "wall",
                        "Width": 4,
                        "Height": 4,
                        "Format": "BGRA8",
                        "Hash": "11223344",
                        "Key": None,
                    }
                ],
                "Materials": [
                    {
                        "ShaderIndex": 0,
                        "Shader": "00000001",
                        "Preset": "00000002",
                        "RenderBucket": 0,
                        "Textures": [
                            {
                                "Param": "2B5170FD",
                                "TextureIndex": 0,
                                "Name": "11223344",
                                "Id64": "0000000000000000",
                                "Tag": "00000000",
                            }
                        ],
                    }
                ],
            },
            {
                "Offset": 8192,
                "File": "model_two.obj",
                "Name": "model_two",
                "Meshes": m2_groups,
                "Verts": 8 * m2_groups,
                "Tris": 12 * m2_groups,
                "IsFragment": False,
                "HasBoneBindings": False,
                "Min": [m2_center[0] - 1.0, m2_center[1] - 1.0, m2_center[2] - 1.0],
                "Max": [m2_center[0] + 1.0 + (2.5 if m2_groups == 2 else 0.0), m2_center[1] + 1.0, m2_center[2] + 1.0],
                "MeshShaders": m2_shaders,
                "Embedded": [],
                "Materials": m2_materials,
            },
        ],
    }
    models_manifest_path = os.path.join(models_dir, "manifest.json")
    with open(models_manifest_path, "w", encoding="utf-8") as f:
        json.dump(models_manifest, f, indent=2)

    return {
        "models_dir": models_dir,
        "models_manifest_path": models_manifest_path,
        "models_manifest": models_manifest,
        "wall_dds_path": wall_dds_path,
        "brick_dds_path": brick_dds_path,
        "corrupt_dds_path": corrupt_dds_path,
    }


def _chunk_disk_path(world_dir: str, chunk_url: str) -> str:
    rel = chunk_url[len("world/"):] if chunk_url.startswith("world/") else chunk_url.lstrip("/")
    return os.path.join(world_dir, *rel.split("/"))


NORMAL_RGBA = (128, 128, 255, 255)
ALBEDO_RGBA = (240, 40, 20, 255)
PLAIN_RGBA = (20, 70, 140, 255)
ALBEDO_PARAM = "C8E8C282"
NORMAL_PARAM = "C6F46E79"
TTPF_ALBEDO_KEY = "AABBCCDDEEFF0011"


def _uniform_bgra_dds(rgba, size=4):
    return make_bgra8_dds(size, size, bytes(rgba) * (size * size))


def _half_float_dds(dxgi, texel_codes, size=4, pad=4):
    """A DX10 R16_FLOAT (54) or R16G16B16A16_FLOAT (10) DDS of one repeated texel, rows padded by pad bytes."""
    texel = struct.pack(f"<{len(texel_codes)}H", *texel_codes)
    row = texel * size
    data = bytearray(make_dx10_dds(size, size, dxgi, (row + bytes(pad)) * size))
    struct.pack_into("<I", data, 20, len(row) + pad)  # DDSD_PITCH is already set
    return bytes(data)


def _param_ref(param, texture_index, id64="0000000000000000"):
    ref = {"TextureIndex": texture_index, "Name": "00000000", "Id64": id64, "Tag": "00000000"}
    if param is not None:
        ref["Param"] = param
    return ref


PARAM_CASES = {
    "normal-first": [_param_ref(NORMAL_PARAM, 0), _param_ref(ALBEDO_PARAM, 1)],
    "albedo-first": [_param_ref(ALBEDO_PARAM, 1), _param_ref(NORMAL_PARAM, 0)],
    "no-albedo": [_param_ref(NORMAL_PARAM, 0)],
    "unannotated": [_param_ref(None, 2)],
}


def _write_param_fixture(assets_dir, refs, *, materials=True, ttpf=False):
    """One textured triangle whose shader 0 references embedded normal/albedo/plain DDS files."""
    models_dir = os.path.join(assets_dir, "models")
    tex_dir = os.path.join(models_dir, "tri_tex")
    os.makedirs(tex_dir)
    names = ("normal", "albedo", "plain")
    for idx, (name, rgba) in enumerate(zip(names, (NORMAL_RGBA, ALBEDO_RGBA, PLAIN_RGBA))):
        with open(os.path.join(tex_dir, f"{idx:03d}_{name}.dds"), "wb") as f:
            f.write(_uniform_bgra_dds(rgba))
    with open(os.path.join(models_dir, "tri.obj"), "w", encoding="utf-8") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    if ttpf:
        ttpf_dir = os.path.join(assets_dir, "ttpf")
        os.makedirs(ttpf_dir)
        with open(os.path.join(ttpf_dir, "albedo.dds"), "wb") as f:
            f.write(_uniform_bgra_dds(ALBEDO_RGBA))
        with open(os.path.join(ttpf_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"Textures": [{"AssetKey": TTPF_ALBEDO_KEY, "File": "albedo.dds"}]}, f)
    item = {
        "File": "tri.obj",
        "Name": "tri",
        "MeshShaders": [0],
        "Embedded": [
            {"Index": idx, "Name": name, "Hash": "00000000", "Key": None}
            for idx, name in enumerate(names)
        ],
        "Materials": [{"ShaderIndex": 0, "Textures": refs}] if materials else [],
    }
    with open(os.path.join(models_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"Items": [item]}, f)


def _glb_first_base_color_pixel(glb_path):
    """Read the first base-color texel with the suite's independent GLB and PNG readers."""
    with open(glb_path, "rb") as f:
        gltf, bin_data = parse_and_validate_glb(f.read())
    for mat in gltf.get("materials", []):
        tex_ref = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex_ref is None:
            continue
        image = gltf["images"][gltf["textures"][tex_ref["index"]]["source"]]
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        _, _, rgba = parse_and_validate_png(bin_data[start:start + view["byteLength"]])
        return gltf, tuple(rgba[:4])
    return gltf, None


class BaseColorParamTest(unittest.TestCase):
    """--base-color-param selects base colour by shader parameter hash; checked by GLB pixel readback."""

    TOOL = os.path.join(TOOLS, "world_build.py")

    def _run_cli(self, assets_dir, out_dir, *extra_args):
        cmd = [sys.executable, "-I", self.TOOL, assets_dir, "--out", out_dir, *extra_args]
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    def _read_world(self, out_dir):
        with open(os.path.join(out_dir, "world.json"), encoding="utf-8") as f:
            world = json.load(f)
        self.assertEqual(len(world["chunks"]), 1)
        gltf, pixel = _glb_first_base_color_pixel(_chunk_disk_path(out_dir, world["chunks"][0]["url"]))
        return world["stats"], gltf, pixel

    def _export(self, refs, *args, materials=True, ttpf=False):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world")
            _write_param_fixture(assets_dir, refs, materials=materials, ttpf=ttpf)
            proc = self._run_cli(assets_dir, out_dir, *args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return self._read_world(out_dir)

    def _assert_textured(self, expected_rgba, refs, *args, **fixture):
        stats, _, pixel = self._export(refs, *args, **fixture)
        self.assertEqual(stats["textures_embedded"], 1)
        self.assertEqual(pixel, expected_rgba)

    def _assert_clay(self, refs, *args, **fixture):
        stats, gltf, pixel = self._export(refs, *args, **fixture)
        self.assertIsNone(pixel)
        self.assertEqual(stats["textures_embedded"], 0)
        self.assertEqual(gltf.get("images", []), [])
        _, clay_gltf, _ = self._export(refs, "--no-textures", **fixture)
        self.assertEqual(gltf.get("materials"), clay_gltf.get("materials"))

    def test_default_behaviour_unchanged(self):
        expected = {
            "normal-first": NORMAL_RGBA,
            "albedo-first": ALBEDO_RGBA,
            "no-albedo": NORMAL_RGBA,
            "unannotated": PLAIN_RGBA,
        }
        for case, rgba in expected.items():
            with self.subTest(case=case):
                self._assert_textured(rgba, PARAM_CASES[case])
        with self.subTest(case="no-materials"):
            # Legacy fallback maps the first sorted DDS to shader 0.
            self._assert_textured(NORMAL_RGBA, [], materials=False)

    def test_half_float_textures_are_embedded(self):
        # 0.5, 0.25, 1, 1 -> (128, 64, 255, 255); 0.75 -> 191 in red only.
        cases = (("normal-first", "000_normal.dds", _half_float_dds(10, (0x3800, 0x3400, 0x3C00, 0x3C00)),
                  (128, 64, 255, 255)),
                 ("unannotated", "002_plain.dds", _half_float_dds(54, (0x3A00,)), (191, 0, 0, 255)))
        for case, name, data, expected in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                assets_dir = os.path.join(tmp, "assets")
                out_dir = os.path.join(tmp, "world")
                _write_param_fixture(assets_dir, PARAM_CASES[case])
                with open(os.path.join(assets_dir, "models", "tri_tex", name), "wb") as f:
                    f.write(data)
                proc = self._run_cli(assets_dir, out_dir)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                stats, _, pixel = self._read_world(out_dir)
                self.assertEqual(stats["textures_embedded"], 1)
                self.assertEqual(pixel, expected)

    def test_selected_param_picks_albedo_regardless_of_order(self):
        for case in ("normal-first", "albedo-first"):
            for value in (ALBEDO_PARAM, "0xc8e8c282"):
                with self.subTest(case=case, value=value):
                    self._assert_textured(ALBEDO_RGBA, PARAM_CASES[case], "--base-color-param", value)

    def test_selected_param_without_resolvable_match_is_clay(self):
        cases = (
            ("no-albedo", PARAM_CASES["no-albedo"], True),
            ("unannotated", PARAM_CASES["unannotated"], True),
            ("unresolved-match", [_param_ref(ALBEDO_PARAM, 7), _param_ref(NORMAL_PARAM, 0)], True),
            ("no-materials", [], False),
        )
        for case, refs, materials in cases:
            with self.subTest(case=case):
                self._assert_clay(refs, "--base-color-param", ALBEDO_PARAM, materials=materials)

    def test_first_resolvable_match_wins(self):
        refs = [_param_ref(ALBEDO_PARAM, 7), _param_ref(NORMAL_PARAM, 0), _param_ref(ALBEDO_PARAM, 1)]
        self._assert_textured(ALBEDO_RGBA, refs, "--base-color-param", ALBEDO_PARAM)

    def test_selected_param_resolves_external_ttpf(self):
        refs = [_param_ref(NORMAL_PARAM, 0), _param_ref(ALBEDO_PARAM, -1, TTPF_ALBEDO_KEY)]
        self._assert_textured(NORMAL_RGBA, refs, ttpf=True)
        self._assert_textured(ALBEDO_RGBA, refs, "--base-color-param", ALBEDO_PARAM, ttpf=True)

    def test_no_textures_with_param_is_clay(self):
        stats, gltf, pixel = self._export(
            PARAM_CASES["normal-first"], "--no-textures", "--base-color-param", ALBEDO_PARAM
        )
        self.assertIsNone(pixel)
        self.assertEqual(stats["textures_embedded"], 0)
        self.assertEqual(gltf.get("images", []), [])

    def test_api_matches_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world")
            _write_param_fixture(assets_dir, PARAM_CASES["normal-first"])
            world_build.build_world(assets_dir, out_dir, base_color_param="0xC8E8C282")
            stats, _, pixel = self._read_world(out_dir)
        self.assertEqual(stats["textures_embedded"], 1)
        self.assertEqual(pixel, ALBEDO_RGBA)

    def test_invalid_param_rejected_before_output(self):
        for value in ("C8E8C28", "C8E8C2822", "0xZZZZZZZZ", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                assets_dir = os.path.join(tmp, "assets")
                out_dir = os.path.join(tmp, "world")
                _write_param_fixture(assets_dir, PARAM_CASES["normal-first"])
                proc = self._run_cli(assets_dir, out_dir, "--base-color-param", value)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("--base-color-param", proc.stderr)
                with self.assertRaises(ValueError):
                    world_build.build_world(assets_dir, out_dir, base_color_param=value)
                self.assertFalse(os.path.exists(out_dir))


class WorldBuildTest(unittest.TestCase):
    TOOL = os.path.join(TOOLS, "world_build.py")

    def _run_cli(self, assets_dir: str, out_dir: str, *extra_args: str) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-I", self.TOOL, assets_dir, "--out", out_dir, *extra_args]
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    @staticmethod
    def _output_snapshot(directory):
        snapshot = {}
        for root, dirs, files in os.walk(directory):
            for name in dirs:
                snapshot[os.path.relpath(os.path.join(root, name), directory)] = None
            for name in files:
                path = os.path.join(root, name)
                with open(path, "rb") as f:
                    snapshot[os.path.relpath(path, directory)] = f.read()
        return snapshot

    def test_compute_spawn_keeps_large_chunks_within_viewer_load_radius(self):
        load_radius = 1400.0  # viewer/index.html LOAD_R
        floor = [{"bounds": [x, -2.0, -7000.0, x + 4000.0, 0.0, -3000.0]} for x in (1000.0, 5000.0, 9000.0)]
        tower = [{"bounds": [-500.0, 0.0, -500.0, -490.0, 3000.0, -490.0]}]
        for chunks, b in ((floor, floor[1]["bounds"]), (tower, tower[0]["bounds"])):
            with self.subTest(bounds=b):
                spawn = world_build._compute_spawn(chunks)
                self.assertEqual(spawn[0], 0.5 * (b[0] + b[3]))
                nearest = [min(max(p, lo), hi) for p, lo, hi in zip(spawn, b[:3], b[3:])]
                self.assertLess(math.dist(spawn, nearest), load_radius)
        small = [{"bounds": [10.0, 0.0, 20.0, 110.0, 5.0, 120.0]}]
        self.assertEqual(world_build._compute_spawn(small), [60.0, 30.0, 180.0])
        self.assertEqual(world_build._compute_spawn([]), [0.0, 20.0, 40.0])

    def test_unusable_input_preserves_output_and_reports_failure(self):
        cases = ("unrelated_directory", "corrupt_manifest", "invalid_schema", "empty_manifest",
                 "missing_objs", "empty_geometry", "invalid_file_fields")
        for existing_output in (False, True):
            for case in cases:
                with self.subTest(existing_output=existing_output, case=case), tempfile.TemporaryDirectory() as tmp:
                    assets = os.path.join(tmp, "assets")
                    output = os.path.join(tmp, "world")
                    info = _write_synthetic_assets(assets, origin_overlap=False)
                    if existing_output:
                        world_build.build_world(assets, output, textures=False)
                    before = self._output_snapshot(output)
                    source = assets
                    if case == "unrelated_directory":
                        source = os.path.join(tmp, "wrong_input")
                        os.mkdir(source)
                    elif case in ("corrupt_manifest", "invalid_schema", "empty_manifest", "invalid_file_fields"):
                        content = {
                            "corrupt_manifest": "{broken",
                            "invalid_schema": '{"Items": null}',
                            "empty_manifest": '{"Items": []}',
                            "invalid_file_fields": '{"Items": [{"File": ""}, {"File": 123}]}',
                        }[case]
                        with open(info["models_manifest_path"], "w", encoding="utf-8") as f:
                            f.write(content)
                    else:
                        for root, _, files in os.walk(assets):
                            for name in files:
                                if name.endswith(".obj"):
                                    path = os.path.join(root, name)
                                    if case == "missing_objs":
                                        os.remove(path)
                                    else:
                                        with open(path, "w", encoding="utf-8") as f:
                                            f.write("# Synthetic OBJ without faces\nv 0 0 0\n")
                    diagnostic = "No readable model manifests" if case in (
                        "unrelated_directory", "corrupt_manifest", "invalid_schema") else "No usable models"
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        world_build.build_world(source, output, textures=False)
                    self.assertEqual(self._output_snapshot(output), before)
                    result = self._run_cli(source, output, "--no-textures")
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("ERROR:", result.stderr)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertIn(source, result.stderr)
                    self.assertNotIn("world built:", result.stdout)
                    self.assertEqual(self._output_snapshot(output), before)
                    if not existing_output:
                        self.assertFalse(os.path.exists(output))

    def test_malformed_optional_material_arrays_match_missing_fields(self):
        cases = (
            ("MeshShaders", 7),
            ("MeshShaders", "0"),
            ("MeshShaders", {"0": 0}),
            ("Embedded", True),
            ("Materials", 7),
            ("Materials", {"0": {}}),
            ("Textures", 1.5),
        )
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            path = _write_synthetic_assets(assets)["models_manifest_path"]
            with open(path, encoding="utf-8") as f:
                original = f.read()

            def build(manifest):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                result = world_build.build_world(assets, output)
                self.assertEqual(result["stats"]["models"], 2)
                return self._output_snapshot(output)

            for field, malformed in cases:
                with self.subTest(field=field, malformed=malformed):
                    manifest = json.loads(original)
                    record = manifest["Items"][0]
                    if field == "Textures":
                        record = record["Materials"][0]
                    del record[field]
                    expected = build(manifest)
                    record[field] = malformed
                    self.assertEqual(build(manifest), expected)

    def test_limit_zero_explicitly_replaces_world_with_empty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            _write_synthetic_assets(assets, origin_overlap=False)
            for cli in (False, True):
                world_build.build_world(assets, output, textures=False)
                empty_source = os.path.join(tmp, "empty_input")
                os.makedirs(empty_source, exist_ok=True)
                if cli:
                    result = self._run_cli(empty_source, output, "--no-textures", "--limit", "0")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    world_build.build_world(empty_source, output, textures=False, limit=0)
                with open(os.path.join(output, "world.json"), encoding="utf-8") as f:
                    document = json.load(f)
                self.assertEqual(document["stats"]["models"], 0)
                self.assertEqual(document["chunks"], [])
                self.assertEqual(os.listdir(os.path.join(output, "chunks")), [])

    def test_limit_does_not_fall_back_to_unselected_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            info = _write_synthetic_assets(assets, origin_overlap=False)
            world_build.build_world(assets, output, textures=False)
            before = self._output_snapshot(output)
            with open(info["models_manifest_path"], encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["Items"].insert(0, {"File": "missing.obj"})
            with open(info["models_manifest_path"], "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(ValueError, "1 selected manifest items"):
                world_build.build_world(assets, output, textures=False, limit=1)
            self.assertEqual(self._output_snapshot(output), before)

    def test_mixed_input_still_builds_usable_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            info = _write_synthetic_assets(assets, origin_overlap=False)
            with open(info["models_manifest_path"], encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["Items"].append({"File": "missing.obj"})
            with open(info["models_manifest_path"], "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            result = world_build.build_world(assets, output, textures=False, jobs=4)
            self.assertEqual(result["stats"]["models"], 2)
            self.assertEqual(len(result["chunks"]), 2)

    def test_discover_index_and_resolve_textures(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            info = _write_synthetic_assets(assets_dir)

            manifests = world_build.discover_model_manifests(assets_dir)
            self.assertEqual(len(manifests), 1)
            self.assertEqual(
                os.path.normcase(os.path.abspath(manifests[0])),
                os.path.normcase(os.path.abspath(info["models_manifest_path"])),
            )

            ttpf_index = world_build.index_ttpf_textures(assets_dir)
            self.assertIn("AABBCCDDEEFF0011", ttpf_index)
            self.assertEqual(
                os.path.normcase(os.path.abspath(ttpf_index["AABBCCDDEEFF0011"])),
                os.path.normcase(os.path.abspath(info["brick_dds_path"])),
            )

            item1, item2 = info["models_manifest"]["Items"]
            res1 = world_build.resolve_model_textures(info["models_dir"], item1, ttpf_index)
            self.assertEqual(
                os.path.normcase(os.path.abspath(res1[0])),
                os.path.normcase(os.path.abspath(info["wall_dds_path"])),
            )

            res2 = world_build.resolve_model_textures(info["models_dir"], item2, ttpf_index)
            self.assertEqual(
                os.path.normcase(os.path.abspath(res2[0])),
                os.path.normcase(os.path.abspath(info["brick_dds_path"])),
            )
            self.assertEqual(
                os.path.normcase(os.path.abspath(res2[1])),
                os.path.normcase(os.path.abspath(info["corrupt_dds_path"])),
            )

    def test_spatial_build_two_chunks_and_corrupt_texture_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world")
            _write_synthetic_assets(assets_dir, origin_overlap=False)

            world_doc = world_build.build_world(assets_dir, out_dir)
            stats = world_doc["stats"]
            self.assertEqual(stats["models"], 2)
            self.assertEqual(stats["chunks"], 2)
            self.assertEqual(stats["textures_embedded"], 2)
            self.assertGreaterEqual(stats["textures_skipped"], 1)

            world_json_path = os.path.join(out_dir, "world.json")
            self.assertTrue(os.path.isfile(world_json_path))
            with open(world_json_path, encoding="utf-8") as f:
                world = json.load(f)

            self.assertIn("spawn", world)
            self.assertEqual(len(world["spawn"]), 3)
            self.assertTrue(all(isinstance(v, (int, float)) and math.isfinite(v) for v in world["spawn"]))

            chunks = world["chunks"]
            self.assertEqual(len(chunks), 2)
            total_images = 0
            for c in chunks:
                self.assertIn("id", c)
                self.assertRegex(c["url"], r"^world/chunks/c_.*\.glb$")
                bounds = c["bounds"]
                self.assertEqual(len(bounds), 6)
                for k in range(3):
                    self.assertLessEqual(bounds[k], bounds[k + 3])

                glb_path = _chunk_disk_path(out_dir, c["url"])
                self.assertTrue(os.path.isfile(glb_path), f"missing chunk file {glb_path}")
                with open(glb_path, "rb") as f:
                    glb_bytes = f.read()
                summary = glb_writer.read_glb_summary(glb_bytes)
                self.assertEqual(summary["version"], 2)
                gltf, bin_data = parse_and_validate_glb(glb_bytes)
                for img in gltf.get("images", []):
                    bv = gltf["bufferViews"][img["bufferView"]]
                    off = bv.get("byteOffset", 0)
                    parse_and_validate_png(bin_data[off:off + bv["byteLength"]])
                total_images += len(gltf.get("images", []))

            # Model 1 (BGRA8 wall) + Model 2 mat 0 (BC1 brick) embedded; corrupt mat 1 skipped gracefully.
            self.assertEqual(total_images, 2)

    def test_grid_layout_non_overlapping_and_min_y_non_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world_grid")
            _write_synthetic_assets(assets_dir, origin_overlap=True)

            p = self._run_cli(assets_dir, out_dir, "--grid-layout")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

            with open(os.path.join(out_dir, "world.json"), encoding="utf-8") as f:
                world = json.load(f)
            self.assertEqual(world["layout"], "grid")
            self.assertGreaterEqual(len(world["chunks"]), 1)

            prim_boxes = []
            for c in world["chunks"]:
                self.assertGreaterEqual(c["bounds"][1], -1e-4)
                glb_path = _chunk_disk_path(out_dir, c["url"])
                with open(glb_path, "rb") as f:
                    glb_bytes = f.read()
                glb_writer.read_glb_summary(glb_bytes)
                gltf, bin_data = parse_and_validate_glb(glb_bytes)
                for mesh in gltf.get("meshes", []):
                    for mp in mesh.get("primitives", []):
                        pts = read_accessor_floats(gltf, bin_data, mp["attributes"]["POSITION"])
                        mn = [min(pt[k] for pt in pts) for k in range(3)]
                        mx = [max(pt[k] for pt in pts) for k in range(3)]
                        self.assertGreaterEqual(mn[1], -1e-4, f"primitive min_y < 0: {mn}")
                        prim_boxes.append((mn, mx))

            self.assertEqual(len(prim_boxes), 2)
            eps = 1e-3
            for i in range(len(prim_boxes)):
                for j in range(i + 1, len(prim_boxes)):
                    a_min, a_max = prim_boxes[i]
                    b_min, b_max = prim_boxes[j]
                    separated = any(
                        a_max[k] <= b_min[k] + eps or b_max[k] <= a_min[k] + eps
                        for k in range(3)
                    )
                    self.assertTrue(
                        separated,
                        f"grid-layout models overlap: box{i}=({a_min}, {a_max}) vs box{j}=({b_min}, {b_max})",
                    )

    def test_no_textures_flag_produces_zero_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world_notex")
            _write_synthetic_assets(assets_dir, origin_overlap=False)

            p = self._run_cli(assets_dir, out_dir, "--no-textures")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

            with open(os.path.join(out_dir, "world.json"), encoding="utf-8") as f:
                world = json.load(f)
            self.assertEqual(len(world["chunks"]), 2)
            for c in world["chunks"]:
                glb_path = _chunk_disk_path(out_dir, c["url"])
                with open(glb_path, "rb") as f:
                    glb_bytes = f.read()
                gltf, _ = parse_and_validate_glb(glb_bytes)
                self.assertEqual(len(gltf.get("images", [])), 0)
                self.assertEqual(len(gltf.get("textures", [])), 0)

    def test_limit_flag_caps_processed_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world_limit")
            _write_synthetic_assets(assets_dir, origin_overlap=False)

            p = self._run_cli(assets_dir, out_dir, "--limit", "1")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

            with open(os.path.join(out_dir, "world.json"), encoding="utf-8") as f:
                world = json.load(f)
            self.assertEqual(world["stats"]["models"], 1)
            self.assertEqual(len(world["chunks"]), 1)
            glb_path = _chunk_disk_path(out_dir, world["chunks"][0]["url"])
            with open(glb_path, "rb") as f:
                glb_writer.read_glb_summary(f.read())

    def test_textures_manifest_excluded_and_zero_sentinel_keys_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            info = _write_synthetic_assets(assets_dir)

            # Add an rpf9dump textures/manifest.json that must NOT be picked up as a model manifest.
            tex_dir = os.path.join(assets_dir, "arch_a", "textures")
            os.makedirs(tex_dir, exist_ok=True)
            with open(os.path.join(tex_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"Items": [{"Dir": "txd_0", "Textures": []}]}, f)

            # Add a TTPF entry with all-zero sentinel AssetKey that must NOT be indexed.
            zero_dds = os.path.join(assets_dir, "ttpf", "pool_a", "zero.dds")
            with open(zero_dds, "wb") as f:
                f.write(make_bgra8_dds(2, 2, bytes([255, 255, 255, 255] * 4)))
            ttpf_manifest_path = os.path.join(assets_dir, "ttpf", "pool_a", "manifest.json")
            with open(ttpf_manifest_path, encoding="utf-8") as f:
                doc = json.load(f)
            doc["Textures"].append({"AssetKey": "0000000000000000", "File": "zero.dds"})
            with open(ttpf_manifest_path, "w", encoding="utf-8") as f:
                json.dump(doc, f)

            manifests = world_build.discover_model_manifests(assets_dir)
            self.assertEqual(len(manifests), 1)
            self.assertEqual(
                os.path.normcase(os.path.abspath(manifests[0])),
                os.path.normcase(os.path.abspath(info["models_manifest_path"])),
            )

            ttpf_index = world_build.index_ttpf_textures(assets_dir)
            self.assertNotIn("0000000000000000", ttpf_index)

    def test_stale_chunks_cleaned_on_rebuild_parallel_jobs_and_missing_assets_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = os.path.join(tmp, "assets")
            out_dir = os.path.join(tmp, "world")
            _write_synthetic_assets(assets_dir, origin_overlap=False)

            # Build 2 chunks with --jobs 4 (parallel workers + shared cache).
            res_full = world_build.build_world(assets_dir, out_dir, jobs=4)
            self.assertEqual(len(res_full["chunks"]), 2)
            chunks_dir = os.path.join(out_dir, "chunks")
            before_files = sorted(os.listdir(chunks_dir))
            self.assertEqual(len(before_files), 2)

            # Include a partial generated chunk and unrelated files in the same directory.
            obsolete = ["c_-3_12.glb", "c_0_-7.glb.tmp"]
            unrelated = ["my-model.glb", "my-model.glb.tmp", "c_01_0.glb", "c_-0_0.glb"]
            for name in obsolete + unrelated:
                with open(os.path.join(chunks_dir, name), "wb") as f:
                    f.write(b"fixture")
            res_limited = world_build.build_world(assets_dir, out_dir, limit=1, jobs=4)
            self.assertEqual(len(res_limited["chunks"]), 1)
            after_files = sorted(os.listdir(chunks_dir))
            expected = [f"{res_limited['chunks'][0]['id']}.glb"] + unrelated
            self.assertEqual(after_files, sorted(expected))

            # Non-existent assets_dir raises FileNotFoundError in API and returns exit 1 in CLI.
            missing = os.path.join(tmp, "does_not_exist")
            with self.assertRaises(FileNotFoundError):
                world_build.build_world(missing, out_dir)
            p = self._run_cli(missing, out_dir)
            self.assertNotEqual(p.returncode, 0)


class WorldBuildStagingTest(unittest.TestCase):
    """A rebuild that fails or is interrupted must leave the previous world.json and its chunks as they were."""

    _snapshot = staticmethod(WorldBuildTest._output_snapshot)

    @staticmethod
    def _failing_write_glb(exc: BaseException, fail_on_call: int = 2):
        """Wrap the real write_glb so that call number `fail_on_call` raises `exc` instead of writing."""
        real = glb_writer.write_glb
        lock = threading.Lock()
        calls = [0]

        def write_glb(path, *args, **kwargs):
            with lock:
                calls[0] += 1
                n = calls[0]
            if n == fail_on_call:
                raise exc
            return real(path, *args, **kwargs)

        return write_glb

    def _previous_world(self, tmp: str) -> tuple[str, str, dict]:
        # The previous world is untextured; the failing rebuild embeds textures, so every chunk it
        # would publish differs from the one on disk and an early overwrite shows up in the snapshot.
        assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
        _write_synthetic_assets(assets, origin_overlap=False)
        world_build.build_world(assets, output, textures=False)
        before = self._snapshot(output)
        self.assertEqual(sum(1 for k in before if k.startswith("chunks" + os.sep)), 2)
        return assets, output, before

    def test_successful_build_publishes_every_chunk_and_world_json(self):
        for jobs in (1, 4):
            with self.subTest(jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
                _write_synthetic_assets(assets, origin_overlap=False)
                result = world_build.build_world(assets, output, jobs=jobs)
                with open(os.path.join(output, "world.json"), encoding="utf-8") as f:
                    document = json.load(f)
                self.assertEqual(document, result)
                self.assertEqual(len(document["chunks"]), 2)
                for chunk in document["chunks"]:
                    with open(_chunk_disk_path(output, chunk["url"]), "rb") as f:
                        parse_and_validate_glb(f.read())
                self.assertEqual(sorted(os.listdir(output)), ["chunks", "world.json"])
                self.assertEqual(sorted(os.listdir(os.path.join(output, "chunks"))),
                                 sorted(f"{c['id']}.glb" for c in document["chunks"]))

    def test_failed_chunk_write_leaves_existing_world_untouched(self):
        failures = (glb_writer.GlbError("synthetic broken asset"), OSError("synthetic write error"),
                    KeyboardInterrupt())
        for exc in failures:
            for jobs in (1, 4):
                with self.subTest(exc=type(exc).__name__, jobs=jobs), tempfile.TemporaryDirectory() as tmp:
                    assets, output, before = self._previous_world(tmp)
                    with unittest.mock.patch.object(glb_writer, "write_glb", self._failing_write_glb(exc)):
                        with self.assertRaises(type(exc)):
                            world_build.build_world(assets, output, jobs=jobs)
                    self.assertEqual(self._snapshot(output), before)

    def test_failed_first_build_leaves_no_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            _write_synthetic_assets(assets, origin_overlap=False)
            with unittest.mock.patch.object(glb_writer, "write_glb", self._failing_write_glb(OSError("disk full"))):
                with self.assertRaises(OSError):
                    world_build.build_world(assets, output)
            self.assertFalse(os.path.exists(output))

    def test_corrupt_staged_chunk_is_rejected_before_publishing(self):
        real = glb_writer.write_glb

        def truncating_write_glb(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            with open(path, "r+b") as f:
                f.truncate(16)
            return info

        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            with unittest.mock.patch.object(glb_writer, "write_glb", truncating_write_glb):
                with self.assertRaises(ValueError):
                    world_build.build_world(assets, output)
            self.assertEqual(self._snapshot(output), before)

    def test_world_json_write_failure_restores_previous_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            world_path = os.path.join(output, "world.json")
            replace = glb_writer._replace

            def fail_manifest_replace(source, destination):
                if destination == world_path:
                    raise OSError("synthetic world.json rename error")
                replace(source, destination)

            with unittest.mock.patch.object(glb_writer, "_replace", fail_manifest_replace):
                with self.assertRaisesRegex(OSError, "world.json rename"):
                    world_build.build_world(assets, output)
            self.assertEqual(self._snapshot(output), before)

    def _kill_during_publication(self, assets, output, point):
        child = """
import os
import sys
tools, assets, output, point = sys.argv[1:]
sys.path.insert(0, tools)
import glb_writer
import world_build

replace = glb_writer._replace
chunks = os.path.join(output, 'chunks')
def kill_after_rename(src, dst):
    adds_chunk = os.path.dirname(dst) == chunks and not os.path.lexists(dst)
    replace(src, dst)
    if ((point == 'backup' and os.path.basename(os.path.dirname(dst)) == 'previous')
            or (point == 'new' and adds_chunk)
            or (point == 'manifest' and dst == os.path.join(output, 'world.json'))):
        os._exit(77)

glb_writer._replace = kill_after_rename
world_build.build_world(assets, output)
"""
        run = subprocess.run([sys.executable, "-I", "-c", child, TOOLS, assets, output, point],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 77, run.stderr)

    def _rerun_with_missing_assets(self, tmp, output):
        # Recovery has to happen even if the new export cannot start.
        run = subprocess.run([sys.executable, "-I", WorldBuildTest.TOOL,
                              os.path.join(tmp, "missing"), "--out", output],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn("assets path does not exist:", run.stderr)

    def test_killed_after_backing_up_live_chunk_restores_previous_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            self._kill_during_publication(assets, output, "backup")
            self.assertEqual(len(os.listdir(os.path.join(output, "chunks"))), 1)
            self._rerun_with_missing_assets(tmp, output)
            self.assertEqual(self._snapshot(output), before)

    @unittest.skipUnless(os.name == "nt", "Windows refuses to replace a read-only destination")
    def test_killed_after_installing_readonly_chunk_restores_previous_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            chunk_dir = os.path.join(output, "chunks")
            chunks = [os.path.join(chunk_dir, name) for name in os.listdir(chunk_dir)]
            for chunk in chunks:
                os.chmod(chunk, stat.S_IREAD)

            self._kill_during_publication(assets, output, "new")
            self.assertNotEqual(self._snapshot(output), before)
            self._rerun_with_missing_assets(tmp, output)

            self.assertEqual(self._snapshot(output), before)
            for chunk in chunks:
                self.assertFalse(os.stat(chunk).st_mode & stat.S_IWRITE)

    def test_killed_after_installing_new_chunk_removes_it_on_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, _ = self._previous_world(tmp)
            world_build.build_world(assets, output, limit=0)
            before = self._snapshot(output)
            self._kill_during_publication(assets, output, "new")
            self.assertEqual(len(os.listdir(os.path.join(output, "chunks"))), 1)
            self._rerun_with_missing_assets(tmp, output)
            self.assertEqual(self._snapshot(output), before)

    def test_killed_after_manifest_commit_keeps_the_new_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            reference = os.path.join(tmp, "reference")
            world_build.build_world(assets, reference)
            expected = self._snapshot(reference)
            self.assertNotEqual(expected, before)
            self._kill_during_publication(assets, output, "manifest")
            self._rerun_with_missing_assets(tmp, output)
            self.assertEqual(self._snapshot(output), expected)

    def test_recovery_that_cannot_restore_a_chunk_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output, before = self._previous_world(tmp)
            self._kill_during_publication(assets, output, "backup")
            interrupted = self._snapshot(output)
            with unittest.mock.patch.object(glb_writer, "_replace", side_effect=PermissionError("file in use")):
                with self.assertRaisesRegex(OSError, "could not recover the previous world"):
                    world_build.build_world(assets, output)
            self.assertEqual(self._snapshot(output), interrupted)
            self._rerun_with_missing_assets(tmp, output)
            self.assertEqual(self._snapshot(output), before)

    def test_staging_left_by_a_killed_build_is_removed_on_the_next_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = os.path.join(tmp, "assets"), os.path.join(tmp, "world")
            _write_synthetic_assets(assets, origin_overlap=False)
            leftover = os.path.join(output, world_build.CHUNK_STAGING_PREFIX + "killed")
            os.makedirs(leftover)
            with open(os.path.join(leftover, "c_0_0.glb"), "wb") as f:
                f.write(b"partial")
            world_build.build_world(assets, output)
            self.assertEqual(sorted(os.listdir(output)), ["chunks", "world.json"])

    def test_stale_cleanup_preserves_staging_with_unrestored_previous_chunks(self):
        prefix = world_build.CHUNK_STAGING_PREFIX
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, prefix + "empty")
            os.makedirs(empty)
            incomplete = os.path.join(tmp, prefix + "incomplete")
            os.makedirs(os.path.join(incomplete, "previous"))  # an empty previous/ holds nothing to recover
            with open(os.path.join(incomplete, "c_1_0.glb"), "wb") as f:
                f.write(b"partial")
            recovery = os.path.join(tmp, prefix + "recovery")
            os.makedirs(os.path.join(recovery, "previous"))
            with open(os.path.join(recovery, "previous", "c_0_0.glb"), "wb") as f:
                f.write(b"previous chunk")
            unrelated = os.path.join(tmp, "notes")
            os.makedirs(unrelated)

            world_build._remove_stale_staging(tmp)

            self.assertEqual(sorted(os.listdir(tmp)), sorted([prefix + "recovery", "notes"]))
            with open(os.path.join(recovery, "previous", "c_0_0.glb"), "rb") as f:
                self.assertEqual(f.read(), b"previous chunk")


# ---------------------------------------------------------------------------
# 4. viewer/serve.py tests
# ---------------------------------------------------------------------------

class ViewerServeTest(unittest.TestCase):
    def test_canceled_chunk_download_does_not_report_a_server_error(self):
        errors = []

        class Server(HTTPServer):
            def handle_error(self, request, client_address):
                errors.append(sys.exc_info()[1])

        class QuietHandler(viewer_serve.Handler):
            def log_message(self, format, *args):
                pass

        with tempfile.TemporaryDirectory() as world_dir:
            # Large enough that the client can close before the server finishes sending it.
            with open(os.path.join(world_dir, "chunk.glb"), "wb") as f:
                f.truncate(16 * 1024 * 1024)
            QuietHandler.world = world_dir
            with Server(("127.0.0.1", 0), functools.partial(QuietHandler, directory=VIEWER)) as srv:
                srv.timeout = 5
                thread = threading.Thread(target=srv.handle_request, daemon=True)
                thread.start()
                try:
                    with socket.create_connection(srv.server_address, timeout=5) as conn:
                        # Force a TCP reset when the partially consumed response is closed.
                        linger = struct.pack("HH" if os.name == "nt" else "ii", 1, 0)
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
                        conn.sendall(b"GET /world/chunk.glb HTTP/1.1\r\nHost: localhost\r\n\r\n")
                        with conn.makefile("rb") as response:
                            self.assertTrue(response.readline().startswith(b"HTTP/1.0 200"))
                finally:
                    thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_safe_world_path_allows_valid_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            world_dir = os.path.join(tmp, "world")
            chunks_dir = os.path.join(world_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            expected_json = os.path.realpath(os.path.join(world_dir, "world.json"))
            expected_glb = os.path.realpath(os.path.join(chunks_dir, "c_0_0.glb"))

            got_json = viewer_serve.safe_world_path(world_dir, "/world/world.json")
            self.assertIsNotNone(got_json)
            self.assertEqual(os.path.normcase(os.path.realpath(got_json)), os.path.normcase(expected_json))

            got_glb = viewer_serve.safe_world_path(world_dir, "/world/chunks/c_0_0.glb")
            self.assertIsNotNone(got_glb)
            self.assertEqual(os.path.normcase(os.path.realpath(got_glb)), os.path.normcase(expected_glb))

            got_query = viewer_serve.safe_world_path(world_dir, "/world/world.json?cache=1")
            self.assertIsNotNone(got_query)
            self.assertEqual(os.path.normcase(os.path.realpath(got_query)), os.path.normcase(expected_json))

    def test_safe_world_path_blocks_traversal_and_drive_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            world_dir = os.path.join(tmp, "world")
            os.makedirs(os.path.join(world_dir, "chunks"), exist_ok=True)
            attacks = [
                "/world/../secret.txt",
                "/world/%2e%2e/secret.txt",
                "/world/..\\secret.txt",
                "/world/C:/Windows/win.ini",
                "/world/chunks/../../outside.txt",
                "/world/%2e%2e%5csecret.txt",
                "/world/chunks/..%2f..%2foutside.txt",
                "/world/",
                "/world/chunks/",
                "/world/chunks",
                "/world/a\x00b",
                "/world/NUL",
                "/world/chunks/CON",
                "/world/COM1.txt",
                "/world/LPT9",
                "/world/world.json:ads",
                "/notworld/world.json",
            ]
            for req_path in attacks:
                with self.subTest(req_path=req_path):
                    self.assertIsNone(viewer_serve.safe_world_path(world_dir, req_path))

    def test_handler_http_serves_world_and_forbids_escapes_even_with_denied_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            world_dir = os.path.join(tmp, "world")
            chunks_dir = os.path.join(world_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            with open(os.path.join(world_dir, "world.json"), "w", encoding="utf-8") as f:
                json.dump({"spawn": [0, 10, 20], "chunks": []}, f)
            with open(os.path.join(chunks_dir, "c_0_0.glb"), "wb") as f:
                f.write(b"glTF\x02\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00JSON")
            # Decoy file literally named __denied__ MUST NOT be leaked on blocked requests.
            with open(os.path.join(world_dir, "__denied__"), "w", encoding="utf-8") as f:
                f.write("DECOY_DENIED_PAYLOAD")
            with open(os.path.join(tmp, "secret.txt"), "w", encoding="utf-8") as f:
                f.write("OUTSIDE_SECRET")

            # Optional symlink escape test (skipped silently if OS denies symlink creation).
            escape_link = os.path.join(world_dir, "escape.txt")
            has_symlink = False
            try:
                os.symlink(os.path.join(tmp, "secret.txt"), escape_link)
                has_symlink = True
            except OSError:
                pass

            class QuietHandler(viewer_serve.Handler):
                def log_message(self, format, *args):
                    pass

            QuietHandler.world = os.path.abspath(world_dir)
            srv = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietHandler, directory=VIEWER),
            )
            port = srv.server_address[1]
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    conn.request("GET", "/world/world.json")
                    r = conn.getresponse()
                    body = r.read()
                    self.assertEqual(r.status, 200)
                    self.assertEqual(json.loads(body.decode("utf-8"))["spawn"], [0, 10, 20])

                    conn.request("GET", "/world/chunks/c_0_0.glb")
                    r = conn.getresponse()
                    body = r.read()
                    self.assertEqual(r.status, 200)
                    self.assertEqual(r.getheader("Content-Type"), "model/gltf-binary")
                    self.assertTrue(body.startswith(b"glTF"))

                    blocked = [
                        "/world/../secret.txt",
                        "/world/%2e%2e/secret.txt",
                        "/world/NUL",
                        "/world/chunks/",
                        "/world/",
                    ]
                    if has_symlink:
                        blocked.append("/world/escape.txt")
                    for bad_path in blocked:
                        with self.subTest(bad_path=bad_path):
                            conn.request("GET", bad_path)
                            r = conn.getresponse()
                            body = r.read()
                            self.assertEqual(r.status, 403)
                            self.assertNotIn(b"DECOY_DENIED_PAYLOAD", body)
                            self.assertNotIn(b"OUTSIDE_SECRET", body)
                finally:
                    conn.close()
            finally:
                srv.shutdown()
                srv.server_close()

    def test_host_allowed_accepts_loopback_names_only(self):
        for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "LOCALHOST:8765", "localhost.:8765",
                     "[::1]", "[::1]:8765", None):
            with self.subTest(host=host):
                self.assertTrue(viewer_serve.host_allowed(host))
        for host in ("evil.example", "evil.example:8765", "127.0.0.1.evil.example", "localhost.evil.example:8765",
                     "192.168.1.5:8765", "0.0.0.0:8765", "[::2]:8765", ""):
            with self.subTest(host=host):
                self.assertFalse(viewer_serve.host_allowed(host))

    def test_handler_refuses_a_foreign_host_header(self):
        # DNS rebinding: a web page points its own domain at 127.0.0.1, so the request arrives here with that domain.
        with tempfile.TemporaryDirectory() as tmp:
            world_dir = os.path.join(tmp, "world")
            os.makedirs(world_dir)
            with open(os.path.join(world_dir, "world.json"), "w", encoding="utf-8") as f:
                f.write('{"spawn": "WORLD_PAYLOAD"}')

            class QuietHandler(viewer_serve.Handler):
                def log_message(self, format, *args):
                    pass

            QuietHandler.world = os.path.abspath(world_dir)
            srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=VIEWER))
            port = srv.server_address[1]
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                for host, status in ((f"rebind.example:{port}", 403), ("evil.example", 403),
                                     (f"localhost:{port}", 200), (f"127.0.0.1:{port}", 200)):
                    for path in ("/world/world.json", "/index.html"):
                        with self.subTest(host=host, path=path):
                            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                            try:
                                conn.request("GET", path, headers={"Host": host})
                                r = conn.getresponse()
                                body = r.read()
                            finally:
                                conn.close()
                            self.assertEqual(r.status, status)
                            if status == 403:
                                self.assertNotIn(b"WORLD_PAYLOAD", body)
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
