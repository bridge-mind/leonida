"""Pure standard-library DDS to PNG decoder and 2x box-filter downscaler.

Supports legacy and DX10 DDS headers across uncompressed 32-bit formats
(BGRA8, RGBA8, BGRX8, RGBX8, R10G10B10A2_UNORM, and arbitrary 32-bit RGB/RGBA bitmasks), the
legacy 24-bit RGB layouts (BGR8/RGB8, e.g. the D3DFMT_R8G8B8 byte order),
and block-compressed formats (BC1, BC2, BC3, BC4, BC5, BC7).

BC7 covers all eight modes. Decoding DXGI 97/98/99 (typeless, UNORM, UNORM_SRGB)
preserves the stored RGBA8 values. The reserved mode (no mode bit in the first
byte) decodes as transparent black. BC6H is not supported.
DXT2, DXT4 and premultiplied DX10 images are converted to straight alpha for PNG.
For tagged sRGB images, unpremultiplication and downscaling use linear light. The PNG stores sRGB bytes.

DX10 R16_FLOAT (54), R32_FLOAT (41) and R16G16B16A16_FLOAT (10) are read as linear values and
written like the UNORM formats, with no tone mapping or sRGB transfer: clamp to
[0, 1], then round(v * 255) half up. +inf saturates to 255; -inf, negative values
and NaN map to 0. The single-channel formats fill red, with green and blue 0 and alpha 255;
R16G16B16A16_FLOAT keeps its own alpha.
R11G11B10_FLOAT previews clamp RGB to [0, 1], map NaN to zero and use opaque alpha.

Usage:
    python -I tools/dds_png.py <in.dds> --info
    python -I tools/dds_png.py <in.dds> <out.png> [--mip N] [--layer N] [--face px] [--z N] [--max-size N]
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from functools import cache
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from typing import BinaryIO, NamedTuple, TypedDict
import zlib

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import bcpack_tables_bptc as _bptc  # noqa: E402

DDS_MAGIC = b"DDS "
DDS_HEADER_SIZE = 128
DDS_DX10_HEADER_SIZE = 148
DDSD_PITCH = 0x8
DDSD_DEPTH = 0x800000
DDSCAPS2_CUBEMAP = 0x200
DDSCAPS2_VOLUME = 0x200000
_CUBE_FACES = ("px", "nx", "py", "ny", "pz", "nz")
DDS_ALPHA_MODE_MASK = 0x7
DDS_ALPHA_MODE_PREMULTIPLIED = 2
DDS_ALPHA_MODE_OPAQUE = 3

DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_FOURCC_MAP: dict[bytes, str] = {
    b"DXT1": "BC1",
    b"DXT2": "BC2",
    b"DXT3": "BC2",
    b"DXT4": "BC3",
    b"DXT5": "BC3",
    b"ATI1": "BC4",
    b"BC4U": "BC4",
    b"BC4S": "BC4_SNORM",
    b"ATI2": "BC5",
    b"BC5U": "BC5",
    b"BC5S": "BC5_SNORM",
}

_DXGI_MAP: dict[int, str] = {
    # Typeless byte formats use the same channel layout as their UNORM variants.
    27: "RGBA8",
    28: "RGBA8",
    29: "RGBA8",
    87: "BGRA8",
    90: "BGRA8",
    91: "BGRA8",
    88: "BGRX8",
    92: "BGRX8",
    93: "BGRX8",
    26: "R11G11B10_FLOAT",
    41: "R32_FLOAT",
    56: "R16_UNORM",
    65: "A8_UNORM",
    70: "BC1",
    71: "BC1",
    72: "BC1",
    73: "BC2",
    74: "BC2",
    75: "BC2",
    76: "BC3",
    77: "BC3",
    78: "BC3",
    79: "BC4",
    80: "BC4",
    81: "BC4_SNORM",
    82: "BC5",
    83: "BC5",
    84: "BC5_SNORM",
    97: "BC7",
    98: "BC7",
    99: "BC7",
    10: "RGBA16_FLOAT",
    54: "R16_FLOAT",
}

# Bytes per texel and channels per texel of the half-float formats.
_HALF_FLOAT_LAYOUT = {"RGBA16_FLOAT": (8, 4), "R16_FLOAT": (2, 1)}
_HALF_TO_U8: bytes | None = None

_SRGB_DXGI_FORMATS = frozenset((29, 72, 75, 78, 91, 93, 99))


def _srgb_byte_to_linear(value: float) -> float:
    # Use the intensity of sRGB byte 1 as the unit so half-byte ties on the linear segment are exact.
    if value <= 0.04045 * 255:
        return value
    return ((value / 255.0 + 0.055) / 1.055) ** 2.4 * (255 * 12.92)


_SRGB_TO_LINEAR = tuple(_srgb_byte_to_linear(v) for v in range(256))
# Crossing the linear intensity of encoded value n + 0.5 rounds the output up to byte n + 1.
_SRGB_HALF_STEPS = tuple(_srgb_byte_to_linear(v + 0.5) for v in range(255))

_CANONICAL_MASKS: dict[tuple[int, int, int, int], str] = {
    (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000): "BGRA8",
    (0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): "BGRX8",
    (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000): "RGBA8",
    (0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000): "RGBX8",
}

# Legacy 24-bit RGB layouts: channel letters in stored byte order (D3DFMT_R8G8B8
# stores B, G, R; alpha is absent, so decoded pixels are opaque).
_RGB24_LAYOUTS: dict[tuple[int, int, int, int], str] = {
    (0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): "BGR8",
    (0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000): "RGB8",
}

_R5_TO_8 = tuple((i << 3) | (i >> 2) for i in range(32))
_G6_TO_8 = tuple((i << 2) | (i >> 4) for i in range(64))

_UNPACK_BC1 = struct.Struct("<HHI").unpack_from
_UNPACK_BC23 = struct.Struct("<QHHI").unpack_from
_UNPACK_BC4 = struct.Struct("<Q").unpack_from
_UNPACK_BC5 = struct.Struct("<QQ").unpack_from

_BC5_Z_LUT: bytes | None = None

# BC7 mode layouts (Microsoft BC7 format mode reference, Khronos BPTC):
# (subsets, partition bits, rotation bits, index-selector bits, colour bits,
#  alpha bits, p-bit per endpoint, p-bit per subset, index bits, second index bits).
_BC7_MODES = (
    (3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    (1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    (1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
)
_BC7_WEIGHTS = {
    2: (0, 21, 43, 64),
    3: (0, 9, 18, 27, 37, 46, 55, 64),
    4: (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64),
}
# Endpoint precision (bits plus p-bit, 5..8) -> 8-bit value by MSB replication.
_BC7_UNQUANTIZE = {
    prec: bytes((v << (8 - prec)) | (v >> (2 * prec - 8)) for v in range(1 << prec))
    for prec in range(5, 9)
}
# Output byte for interpolated R, G, B, A under rotation 0-3 (A swaps with R, G or B).
_BC7_ROTATIONS = ((0, 1, 2, 3), (3, 1, 2, 0), (0, 3, 2, 1), (0, 1, 3, 2))
_BC7_RESERVED_BLOCK = bytes(64)
_BC7_BLOCK_CACHE_LIMIT = 4096


def _bc7_partition_layouts() -> tuple:
    """Per subset count, each partition's (subset of pixel, pixel is anchor) tuples."""

    def layout(subset_of, anchors):
        return tuple(subset_of), tuple(int(p in anchors) for p in range(16))

    two = tuple(
        layout(((mask >> p) & 1 for p in range(16)), (0, anchor))
        for mask, anchor in zip(_bptc.BC7_PART2_PACKED, _bptc.BC7_ANCHOR2)
    )
    three = tuple(
        layout(((mask >> (2 * p)) & 3 for p in range(16)), (0, anchor_a, anchor_b))
        for mask, anchor_a, anchor_b in zip(
            _bptc.BC7_PART3_PACKED, _bptc.BC7_ANCHOR3A, _bptc.BC7_ANCHOR3B
        )
    )
    return (None, (layout((0,) * 16, (0,)),), two, three)


_BC7_LAYOUTS = _bc7_partition_layouts()


class DdsError(ValueError):
    """Raised when DDS header or payload data is invalid or unsupported."""


def _validate_mask(mask: int) -> tuple[int, int]:
    """Return (shift, max_value) for a contiguous 32-bit bitmask, or (0, 0) if zero."""
    if mask == 0:
        return 0, 0
    shift = (mask & -mask).bit_length() - 1
    max_val = mask >> shift
    if (max_val & (max_val + 1)) != 0:
        raise DdsError(f"Non-contiguous channel bitmask: 0x{mask:08X}")
    return shift, max_val


class _DdsHeader(TypedDict):
    width: int
    height: int
    depth: int
    dimension: str
    array_size: int
    faces: tuple[str, ...]
    format: str
    dxgi_format: int | None
    mip_count: int
    pitch: int
    data_offset: int


def _parse_header_internal(
    data: bytes | bytearray | memoryview,
) -> tuple[_DdsHeader, tuple[int, int, int, int] | None, int]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise DdsError(f"Expected bytes-like object, got {type(data).__name__}")
    if len(data) < DDS_HEADER_SIZE:
        raise DdsError(f"DDS header too short: {len(data)} bytes (need at least {DDS_HEADER_SIZE})")
    if bytes(data[:4]) != DDS_MAGIC:
        raise DdsError(f"Invalid DDS magic: {bytes(data[:4])!r}")

    flags, height, width, pitch, raw_depth, raw_mips = struct.unpack_from("<6I", data, 8)
    if width <= 0 or height <= 0:
        raise DdsError(f"Invalid DDS dimensions: {width}x{height}")

    mip_count = raw_mips if raw_mips > 0 else 1

    pf_flags, fourcc, rgb_bit_count, r_mask, g_mask, b_mask, a_mask = struct.unpack_from(
        "<I4sIIIII", data, 80
    )

    dxgi_format: int | None = None
    alpha_mode = 0
    data_offset = DDS_HEADER_SIZE
    masks: tuple[int, int, int, int] | None = None

    if pf_flags & DDPF_FOURCC:
        if fourcc == b"DX10":
            if len(data) < DDS_DX10_HEADER_SIZE:
                raise DdsError(
                    f"Truncated DX10 DDS header: {len(data)} bytes (need at least {DDS_DX10_HEADER_SIZE})"
                )
            dxgi_format = struct.unpack_from("<I", data, 128)[0]
            if dxgi_format == 24:  # DXGI_FORMAT_R10G10B10A2_UNORM, R in the low bits.
                fmt = "RGBA8"
                masks = (0x000003FF, 0x000FFC00, 0x3FF00000, 0xC0000000)
            else:
                fmt = _DXGI_MAP.get(dxgi_format)
            if fmt is None:
                raise DdsError(f"Unsupported DXGI format: {dxgi_format}")
            data_offset = DDS_DX10_HEADER_SIZE
            alpha_mode = struct.unpack_from("<I", data, 144)[0] & DDS_ALPHA_MODE_MASK
        else:
            fmt = _FOURCC_MAP.get(fourcc)
            if fourcc in (b"DXT2", b"DXT4"):
                alpha_mode = DDS_ALPHA_MODE_PREMULTIPLIED
            if fmt is None:
                raise DdsError(f"Unsupported DDS FourCC: {fourcc!r}")
    elif pf_flags & DDPF_RGB:
        if rgb_bit_count not in (24, 32):
            raise DdsError(
                f"Unsupported uncompressed RGB bit count: {rgb_bit_count} (only 24- and 32-bit supported)"
            )
        if not (pf_flags & DDPF_ALPHAPIXELS):
            a_mask = 0
        if (r_mask | g_mask | b_mask) == 0:
            raise DdsError("Invalid uncompressed DDS: all RGB bitmasks are zero")
        for m in (r_mask, g_mask, b_mask, a_mask):
            _validate_mask(m)
        masks = (r_mask, g_mask, b_mask, a_mask)
        if rgb_bit_count == 24:
            fmt = _RGB24_LAYOUTS.get(masks)
            if fmt is None:
                raise DdsError(
                    "Unsupported 24-bit RGB bitmask layout: "
                    f"0x{r_mask:08X}/0x{g_mask:08X}/0x{b_mask:08X}"
                )
        else:
            fmt = _CANONICAL_MASKS.get(masks)
            if fmt is None:
                fmt = "RGBA8" if a_mask != 0 else "RGBX8"
    else:
        raise DdsError(f"Unsupported DDS pixel format flags: 0x{pf_flags:08X}")

    caps2 = struct.unpack_from("<I", data, 112)[0]
    if dxgi_format is not None:
        dim, misc, array_size = struct.unpack_from("<III", data, 132)
        if dim not in (2, 3, 4) or not array_size:
            raise DdsError(f"Invalid DX10 texture dimension {dim} or array size {array_size}")
        dimension = {2: "1d", 3: "2d", 4: "3d"}[dim]
        faces = _CUBE_FACES if misc & 0x4 else ()
    else:
        dimension = "3d" if flags & DDSD_DEPTH or caps2 & DDSCAPS2_VOLUME else "2d"
        array_size = 1
        faces = ()
        if caps2 & DDSCAPS2_CUBEMAP:
            faces = tuple(face for index, face in enumerate(_CUBE_FACES)
                          if caps2 & (0x400 << index))
            if not faces:
                raise DdsError("Legacy DDS cubemap has no faces")

    if dimension == "1d" and height != 1:
        raise DdsError("A 1D DDS must have height 1")
    if faces and (dimension != "2d" or width != height):
        raise DdsError("DDS cubemap faces must be square 2D textures")
    depth = 1
    if dimension == "3d":
        if array_size != 1 or not flags & DDSD_DEPTH or not raw_depth:
            raise DdsError("A DDS volume must specify depth with a single array element")
        depth = raw_depth
    if mip_count > max(width, height, depth).bit_length():
        raise DdsError(f"Too many DDS mip levels ({mip_count}) for {width}x{height}x{depth}")

    info: _DdsHeader = {
        "width": width,
        "height": height,
        "depth": depth,
        "dimension": dimension,
        "array_size": array_size,
        "faces": faces,
        "format": fmt,
        "dxgi_format": dxgi_format,
        "mip_count": mip_count,
        "data_offset": data_offset,
        "pitch": pitch if flags & DDSD_PITCH else 0,
    }
    return info, masks, alpha_mode


def parse_dds_header(data: bytes) -> _DdsHeader:
    """Return format and shape metadata from a legacy or DX10 DDS header."""
    info, _, _ = _parse_header_internal(data)
    return info


class _Plane(NamedTuple):
    offset: int
    width: int
    height: int
    row_pitch: int
    byte_length: int


_TEXEL_BYTES = {
    "RGBA8": 4, "BGRA8": 4, "BGRX8": 4, "RGBX8": 4, "R11G11B10_FLOAT": 4,
    "BGR8": 3, "RGB8": 3, "R16_UNORM": 2, "A8_UNORM": 1,
    "R32_FLOAT": 4,
    **{fmt: layout[0] for fmt, layout in _HALF_FLOAT_LAYOUT.items()},
}


def _locate_plane(info: _DdsHeader, *, mip: int = 0, layer: int = 0,
                  face: str | None = None, z: int = 0) -> _Plane:
    """Locate one stored 2D plane without walking individual layers or depth slices."""
    for name, value, count in (("mip", mip, info["mip_count"]), ("layer", layer, info["array_size"])):
        if type(value) is not int or not 0 <= value < count:
            raise DdsError(f"DDS {name} must be in 0..{count - 1}, got {value!r}")
    mip_depth = max(1, info["depth"] >> mip)
    if type(z) is not int or not 0 <= z < mip_depth:
        raise DdsError(f"DDS z must be in 0..{mip_depth - 1} at mip {mip}, got {z!r}")

    faces = info["faces"]
    if face is not None and face not in faces:
        raise DdsError(f"DDS does not contain face {face!r}")
    image_index = layer
    if faces:
        image_index = layer * len(faces) + (faces.index(face) if face is not None else 0)

    fmt = info["format"]
    if fmt.startswith("BC"):
        block = 4
        unit_bytes = 8 if fmt in ("BC1", "BC4", "BC4_SNORM") else 16
    else:
        block = 1
        unit_bytes = _TEXEL_BYTES[fmt]

    # Arrays and cubemaps store one complete mip chain per image. A volume
    # instead stores all depth slices of each mip together. A header has at
    # most 32 mip levels, irrespective of its declared array size or depth.
    chain_bytes = 0
    for level in range(info["mip_count"]):
        width = max(1, info["width"] >> level)
        height = max(1, info["height"] >> level)
        rows = (height + block - 1) // block
        row_bytes = (width + block - 1) // block * unit_bytes
        row_pitch = row_bytes
        if level == 0 and block == 1 and info["pitch"]:
            row_pitch = info["pitch"]
            if row_pitch < row_bytes:
                raise DdsError(f"Invalid {fmt} row pitch: {row_pitch} bytes (need at least {row_bytes})")
        slice_bytes = rows * row_pitch
        if level == mip:
            # The next plane starts after full rows including padding. Decoding
            # this plane only needs the visible bytes of its last row.
            plane = _Plane(chain_bytes + z * slice_bytes, width, height, row_pitch,
                           slice_bytes - row_pitch + row_bytes)
        chain_bytes += slice_bytes * max(1, info["depth"] >> level)
    return plane._replace(offset=info["data_offset"] + image_index * chain_bytes + plane.offset)


def _decode_uncompressed_32(
    payload: memoryview,
    width: int,
    height: int,
    fmt: str,
    masks: tuple[int, int, int, int] | None,
    pitch: int,
) -> bytes:
    pixel_count = width * height
    row_bytes = width * 4
    pitch = pitch or row_bytes
    if pitch < row_bytes:
        raise DdsError(f"Invalid {fmt} row pitch: {pitch} bytes (need at least {row_bytes})")
    expected = (height - 1) * pitch + row_bytes
    if len(payload) < expected:
        raise DdsError(f"Truncated {fmt} payload: {len(payload)} bytes (need {expected})")

    if pitch == row_bytes:
        src = bytes(payload[:expected])
    else:
        src = b"".join(payload[start : start + row_bytes] for start in range(0, height * pitch, pitch))

    if masks is None or masks in _CANONICAL_MASKS:
        if fmt == "RGBA8":
            return src
        if fmt == "RGBX8":
            out = bytearray(src)
            out[3::4] = b"\xff" * pixel_count
            return bytes(out)
        if fmt == "BGRA8":
            out = bytearray(src)
            out[0::4], out[2::4] = out[2::4], out[0::4]
            return bytes(out)
        if fmt == "BGRX8":
            out = bytearray(src)
            out[0::4], out[2::4] = out[2::4], out[0::4]
            out[3::4] = b"\xff" * pixel_count
            return bytes(out)

    assert masks is not None
    r_shift, r_max = _validate_mask(masks[0])
    g_shift, g_max = _validate_mask(masks[1])
    b_shift, b_max = _validate_mask(masks[2])
    a_shift, a_max = _validate_mask(masks[3])

    out = bytearray(len(src))
    # Fast path when all non-zero masks are byte-aligned 8-bit lanes.
    if (
        (r_max in (0, 255) and (r_shift & 7) == 0)
        and (g_max in (0, 255) and (g_shift & 7) == 0)
        and (b_max in (0, 255) and (b_shift & 7) == 0)
        and (a_max in (0, 255) and (a_shift & 7) == 0)
    ):
        if r_max:
            out[0::4] = src[r_shift >> 3 :: 4]
        if g_max:
            out[1::4] = src[g_shift >> 3 :: 4]
        if b_max:
            out[2::4] = src[b_shift >> 3 :: 4]
        if a_max:
            out[3::4] = src[a_shift >> 3 :: 4]
        else:
            out[3::4] = b"\xff" * pixel_count
        return bytes(out)

    # General bitmask fallback (e.g. R10G10B10A2).
    r_half = r_max >> 1
    g_half = g_max >> 1
    b_half = b_max >> 1
    a_half = a_max >> 1
    dst = 0
    for (px,) in struct.iter_unpack("<I", src):
        out[dst] = (((px >> r_shift) & r_max) * 255 + r_half) // r_max if r_max else 0
        out[dst + 1] = (((px >> g_shift) & g_max) * 255 + g_half) // g_max if g_max else 0
        out[dst + 2] = (((px >> b_shift) & b_max) * 255 + b_half) // b_max if b_max else 0
        out[dst + 3] = (((px >> a_shift) & a_max) * 255 + a_half) // a_max if a_max else 255
        dst += 4
    return bytes(out)


def _half_to_u8_lut() -> bytes:
    """Byte for every IEEE 754 binary16 code: clamp to [0, 1], round half up; NaN -> 0, +inf -> 255."""
    global _HALF_TO_U8
    if _HALF_TO_U8 is None:
        lut = bytearray(65536)
        for code, (v,) in enumerate(struct.iter_unpack("<e", struct.pack("<65536H", *range(65536)))):
            if v != v or v <= 0.0:  # NaN, -0, negatives and -inf
                continue
            lut[code] = 255 if v >= 1.0 else int(v * 255.0 + 0.5)  # v * 255 is exact in a double
        _HALF_TO_U8 = bytes(lut)
    return _HALF_TO_U8


def _decode_half_float(payload: memoryview, width: int, height: int, fmt: str, pitch: int) -> bytes:
    texel_bytes, channels = _HALF_FLOAT_LAYOUT[fmt]
    row_bytes = width * texel_bytes
    pitch = pitch or row_bytes
    if pitch < row_bytes:
        raise DdsError(f"Invalid {fmt} row pitch: {pitch} bytes (need at least {row_bytes})")
    expected = (height - 1) * pitch + row_bytes
    if len(payload) < expected:
        raise DdsError(f"Truncated {fmt} payload: {len(payload)} bytes (need {expected})")

    lut = _half_to_u8_lut()
    row_count = width * channels
    unpack_row = struct.Struct(f"<{row_count}H").unpack_from
    out = bytearray(width * height * 4)
    if channels == 1:
        out[3::4] = b"\xff" * (width * height)
    for y in range(height):
        values = bytes(map(lut.__getitem__, unpack_row(payload, y * pitch)))
        start = y * width * 4
        if channels == 4:
            out[start : start + row_bytes // 2] = values
        else:
            out[start : start + width * 4 : 4] = values
    return bytes(out)


def _decode_uncompressed_24(
    payload: memoryview,
    width: int,
    height: int,
    fmt: str,
    masks: tuple[int, int, int, int],
    pitch: int,
) -> bytes:
    """Decode a legacy 24-bit RGB payload (one byte per channel, no alpha) to RGBA8."""
    pixel_count = width * height
    row_bytes = width * 3
    pitch = pitch or row_bytes
    if pitch < row_bytes:
        raise DdsError(f"Invalid {fmt} row pitch: {pitch} bytes (need at least {row_bytes})")
    expected = (height - 1) * pitch + row_bytes
    if len(payload) < expected:
        raise DdsError(f"Truncated {fmt} payload: {len(payload)} bytes (need {expected})")

    if pitch == row_bytes:
        src = bytes(payload[:expected])
    else:
        src = b"".join(payload[start : start + row_bytes] for start in range(0, height * pitch, pitch))

    # The header validation pinned each mask to a byte-aligned 8-bit lane.
    r_byte = _validate_mask(masks[0])[0] >> 3
    g_byte = _validate_mask(masks[1])[0] >> 3
    b_byte = _validate_mask(masks[2])[0] >> 3
    out = bytearray(pixel_count * 4)
    out[0::4] = src[r_byte::3]
    out[1::4] = src[g_byte::3]
    out[2::4] = src[b_byte::3]
    out[3::4] = b"\xff" * pixel_count
    return bytes(out)


def _bc_interp_lut(v0: int, v1: int) -> tuple[int, int, int, int, int, int, int, int]:
    """Build the 8-entry BC3-alpha / BC4 / BC5 unsigned interpolation table."""
    if v0 > v1:
        return (
            v0,
            v1,
            (6 * v0 + v1 + 3) // 7,
            (5 * v0 + 2 * v1 + 3) // 7,
            (4 * v0 + 3 * v1 + 3) // 7,
            (3 * v0 + 4 * v1 + 3) // 7,
            (2 * v0 + 5 * v1 + 3) // 7,
            (v0 + 6 * v1 + 3) // 7,
        )
    return (
        v0,
        v1,
        (4 * v0 + v1 + 2) // 5,
        (3 * v0 + 2 * v1 + 2) // 5,
        (2 * v0 + 3 * v1 + 2) // 5,
        (v0 + 4 * v1 + 2) // 5,
        0,
        255,
    )


def _bc_interp_lut_snorm(b0: int, b1: int) -> tuple[int, int, int, int, int, int, int, int]:
    """Build the 8-entry BC4_SNORM / BC5_SNORM table mapped from [-1, 1] to [0, 255]."""
    s0 = b0 - 256 if b0 >= 128 else b0
    s1 = b1 - 256 if b1 >= 128 else b1
    if s0 == -128:
        s0 = -127
    if s1 == -128:
        s1 = -127

    def to_u8(v: float) -> int:
        # Map [-127.0, 127.0] -> [0, 255]
        return int(round((v / 127.0 * 0.5 + 0.5) * 255.0))

    if s0 > s1:
        return (
            to_u8(s0),
            to_u8(s1),
            to_u8((6 * s0 + s1) / 7.0),
            to_u8((5 * s0 + 2 * s1) / 7.0),
            to_u8((4 * s0 + 3 * s1) / 7.0),
            to_u8((3 * s0 + 4 * s1) / 7.0),
            to_u8((2 * s0 + 5 * s1) / 7.0),
            to_u8((s0 + 6 * s1) / 7.0),
        )
    return (
        to_u8(s0),
        to_u8(s1),
        to_u8((4 * s0 + s1) / 5.0),
        to_u8((3 * s0 + 2 * s1) / 5.0),
        to_u8((2 * s0 + 3 * s1) / 5.0),
        to_u8((s0 + 4 * s1) / 5.0),
        0,
        255,
    )


def _bc5_z_direct(r: int, g: int) -> int:
    nx = r / 127.5 - 1.0
    ny = g / 127.5 - 1.0
    return int(round((math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny)) * 0.5 + 0.5) * 255))


def _get_bc5_z_lut() -> bytes:
    global _BC5_Z_LUT
    if _BC5_Z_LUT is None:
        norm = [i / 127.5 - 1.0 for i in range(256)]
        norm_sq = [x * x for x in norm]
        lut = bytearray(65536)
        idx = 0
        for r in range(256):
            rem_r = 1.0 - norm_sq[r]
            for g in range(256):
                rem = rem_r - norm_sq[g]
                z = math.sqrt(rem) if rem > 0.0 else 0.0
                lut[idx] = int(round((z * 0.5 + 0.5) * 255))
                idx += 1
        _BC5_Z_LUT = bytes(lut)
    return _BC5_Z_LUT


def _decode_bc1(data: bytes | bytearray | memoryview, offset: int, width: int, height: int) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 8
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC1 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    r5_tab = _R5_TO_8
    g6_tab = _G6_TO_8
    unpack = _UNPACK_BC1
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            c0, c1, bits = unpack(data, pos)
            pos += 8

            r0 = r5_tab[c0 >> 11]
            g0 = g6_tab[(c0 >> 5) & 0x3F]
            b0 = r5_tab[c0 & 0x1F]
            r1 = r5_tab[c1 >> 11]
            g1 = g6_tab[(c1 >> 5) & 0x3F]
            b1 = r5_tab[c1 & 0x1F]

            if c0 > c1:
                pal = (
                    bytes((r0, g0, b0, 255)),
                    bytes((r1, g1, b1, 255)),
                    bytes(((2 * r0 + r1 + 1) // 3, (2 * g0 + g1 + 1) // 3, (2 * b0 + b1 + 1) // 3, 255)),
                    bytes(((r0 + 2 * r1 + 1) // 3, (g0 + 2 * g1 + 1) // 3, (b0 + 2 * b1 + 1) // 3, 255)),
                )
            else:
                pal = (
                    bytes((r0, g0, b0, 255)),
                    bytes((r1, g1, b1, 255)),
                    bytes(((r0 + r1) >> 1, (g0 + g1) >> 1, (b0 + b1) >> 1, 255)),
                    b"\x00\x00\x00\x00",
                )

            px0 = bx * 4
            max_x = width - px0
            dst = row_base + px0 * 4
            if max_x >= 4:
                for ty in range(max_y):
                    rb = (bits >> (ty * 8)) & 0xFF
                    out[dst : dst + 16] = (
                        pal[rb & 3] + pal[(rb >> 2) & 3] + pal[(rb >> 4) & 3] + pal[rb >> 6]
                    )
                    dst += row_stride
            else:
                for ty in range(max_y):
                    rb = (bits >> (ty * 8)) & 0xFF
                    d = dst
                    for tx in range(max_x):
                        out[d : d + 4] = pal[(rb >> (tx * 2)) & 3]
                        d += 4
                    dst += row_stride

    return bytes(out)


def _decode_bc2(data: bytes | bytearray | memoryview, offset: int, width: int, height: int) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 16
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC2 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    r5_tab = _R5_TO_8
    g6_tab = _G6_TO_8
    unpack = _UNPACK_BC23
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            alpha_q, c0, c1, bits = unpack(data, pos)
            pos += 16

            r0 = r5_tab[c0 >> 11]
            g0 = g6_tab[(c0 >> 5) & 0x3F]
            b0 = r5_tab[c0 & 0x1F]
            r1 = r5_tab[c1 >> 11]
            g1 = g6_tab[(c1 >> 5) & 0x3F]
            b1 = r5_tab[c1 & 0x1F]
            rgb = (
                (r0, g0, b0),
                (r1, g1, b1),
                ((2 * r0 + r1 + 1) // 3, (2 * g0 + g1 + 1) // 3, (2 * b0 + b1 + 1) // 3),
                ((r0 + 2 * r1 + 1) // 3, (g0 + 2 * g1 + 1) // 3, (b0 + 2 * b1 + 1) // 3),
            )

            px0 = bx * 4
            max_x = width - px0
            dst = row_base + px0 * 4
            if max_x >= 4:
                for ty in range(max_y):
                    ar = (alpha_q >> (ty * 16)) & 0xFFFF
                    rb = (bits >> (ty * 8)) & 0xFF
                    p0 = rgb[rb & 3]
                    p1 = rgb[(rb >> 2) & 3]
                    p2 = rgb[(rb >> 4) & 3]
                    p3 = rgb[rb >> 6]
                    out[dst : dst + 16] = bytes((
                        p0[0], p0[1], p0[2], (ar & 0xF) * 17,
                        p1[0], p1[1], p1[2], ((ar >> 4) & 0xF) * 17,
                        p2[0], p2[1], p2[2], ((ar >> 8) & 0xF) * 17,
                        p3[0], p3[1], p3[2], (ar >> 12) * 17,
                    ))
                    dst += row_stride
            else:
                for ty in range(max_y):
                    ar = (alpha_q >> (ty * 16)) & 0xFFFF
                    rb = (bits >> (ty * 8)) & 0xFF
                    d = dst
                    for tx in range(max_x):
                        p = rgb[(rb >> (tx * 2)) & 3]
                        out[d] = p[0]
                        out[d + 1] = p[1]
                        out[d + 2] = p[2]
                        out[d + 3] = ((ar >> (tx * 4)) & 0xF) * 17
                        d += 4
                    dst += row_stride

    return bytes(out)


def _decode_bc3(data: bytes | bytearray | memoryview, offset: int, width: int, height: int) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 16
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC3 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    r5_tab = _R5_TO_8
    g6_tab = _G6_TO_8
    unpack = _UNPACK_BC23
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            alpha_q, c0, c1, bits = unpack(data, pos)
            pos += 16

            a_lut = _bc_interp_lut(alpha_q & 0xFF, (alpha_q >> 8) & 0xFF)
            a_idx = alpha_q >> 16

            r0 = r5_tab[c0 >> 11]
            g0 = g6_tab[(c0 >> 5) & 0x3F]
            b0 = r5_tab[c0 & 0x1F]
            r1 = r5_tab[c1 >> 11]
            g1 = g6_tab[(c1 >> 5) & 0x3F]
            b1 = r5_tab[c1 & 0x1F]
            rgb = (
                (r0, g0, b0),
                (r1, g1, b1),
                ((2 * r0 + r1 + 1) // 3, (2 * g0 + g1 + 1) // 3, (2 * b0 + b1 + 1) // 3),
                ((r0 + 2 * r1 + 1) // 3, (g0 + 2 * g1 + 1) // 3, (b0 + 2 * b1 + 1) // 3),
            )

            px0 = bx * 4
            max_x = width - px0
            dst = row_base + px0 * 4
            if max_x >= 4:
                for ty in range(max_y):
                    ar = (a_idx >> (ty * 12)) & 0xFFF
                    rb = (bits >> (ty * 8)) & 0xFF
                    p0 = rgb[rb & 3]
                    p1 = rgb[(rb >> 2) & 3]
                    p2 = rgb[(rb >> 4) & 3]
                    p3 = rgb[rb >> 6]
                    out[dst : dst + 16] = bytes((
                        p0[0], p0[1], p0[2], a_lut[ar & 7],
                        p1[0], p1[1], p1[2], a_lut[(ar >> 3) & 7],
                        p2[0], p2[1], p2[2], a_lut[(ar >> 6) & 7],
                        p3[0], p3[1], p3[2], a_lut[ar >> 9],
                    ))
                    dst += row_stride
            else:
                for ty in range(max_y):
                    ar = (a_idx >> (ty * 12)) & 0xFFF
                    rb = (bits >> (ty * 8)) & 0xFF
                    d = dst
                    for tx in range(max_x):
                        p = rgb[(rb >> (tx * 2)) & 3]
                        out[d] = p[0]
                        out[d + 1] = p[1]
                        out[d + 2] = p[2]
                        out[d + 3] = a_lut[(ar >> (tx * 3)) & 7]
                        d += 4
                    dst += row_stride

    return bytes(out)


def _decode_bc4(
    data: bytes | bytearray | memoryview,
    offset: int,
    width: int,
    height: int,
    *,
    signed: bool = False,
) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 8
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC4 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    unpack = _UNPACK_BC4
    lut_fn = _bc_interp_lut_snorm if signed else _bc_interp_lut
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            (rq,) = unpack(data, pos)
            pos += 8

            r_lut = lut_fn(rq & 0xFF, (rq >> 8) & 0xFF)
            px = tuple(bytes((v, v, v, 255)) for v in r_lut)
            r_idx = rq >> 16

            px0 = bx * 4
            max_x = width - px0
            dst = row_base + px0 * 4
            if max_x >= 4:
                for ty in range(max_y):
                    rr = (r_idx >> (ty * 12)) & 0xFFF
                    out[dst : dst + 16] = (
                        px[rr & 7] + px[(rr >> 3) & 7] + px[(rr >> 6) & 7] + px[rr >> 9]
                    )
                    dst += row_stride
            else:
                for ty in range(max_y):
                    rr = (r_idx >> (ty * 12)) & 0xFFF
                    d = dst
                    for tx in range(max_x):
                        out[d : d + 4] = px[(rr >> (tx * 3)) & 7]
                        d += 4
                    dst += row_stride

    return bytes(out)


def _decode_bc5(
    data: bytes | bytearray | memoryview,
    offset: int,
    width: int,
    height: int,
    *,
    signed: bool = False,
) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 16
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC5 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    unpack = _UNPACK_BC5
    lut_fn = _bc_interp_lut_snorm if signed else _bc_interp_lut
    use_lut = width * height >= 4096 or _BC5_Z_LUT is not None
    z_lut = _get_bc5_z_lut() if use_lut else None
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            rq, gq = unpack(data, pos)
            pos += 16

            r_lut = lut_fn(rq & 0xFF, (rq >> 8) & 0xFF)
            g_lut = lut_fn(gq & 0xFF, (gq >> 8) & 0xFF)
            r_idx = rq >> 16
            g_idx = gq >> 16

            px0 = bx * 4
            max_x = width - px0
            dst = row_base + px0 * 4
            if z_lut is not None and max_x >= 4:
                for ty in range(max_y):
                    rr = (r_idx >> (ty * 12)) & 0xFFF
                    gr = (g_idx >> (ty * 12)) & 0xFFF
                    r0 = r_lut[rr & 7]
                    g0 = g_lut[gr & 7]
                    r1 = r_lut[(rr >> 3) & 7]
                    g1 = g_lut[(gr >> 3) & 7]
                    r2 = r_lut[(rr >> 6) & 7]
                    g2 = g_lut[(gr >> 6) & 7]
                    r3 = r_lut[rr >> 9]
                    g3 = g_lut[gr >> 9]
                    out[dst : dst + 16] = bytes((
                        r0, g0, z_lut[(r0 << 8) | g0], 255,
                        r1, g1, z_lut[(r1 << 8) | g1], 255,
                        r2, g2, z_lut[(r2 << 8) | g2], 255,
                        r3, g3, z_lut[(r3 << 8) | g3], 255,
                    ))
                    dst += row_stride
            else:
                for ty in range(max_y):
                    rr = (r_idx >> (ty * 12)) & 0xFFF
                    gr = (g_idx >> (ty * 12)) & 0xFFF
                    d = dst
                    for tx in range(max_x):
                        r = r_lut[(rr >> (tx * 3)) & 7]
                        g = g_lut[(gr >> (tx * 3)) & 7]
                        z = z_lut[(r << 8) | g] if z_lut is not None else _bc5_z_direct(r, g)
                        out[d] = r
                        out[d + 1] = g
                        out[d + 2] = z
                        out[d + 3] = 255
                        d += 4
                    dst += row_stride

    return bytes(out)


@cache
def _unsigned_float_to_u8(mantissa_bits: int) -> bytes:
    """Map a five-bit-exponent unsigned float channel to preview bytes."""
    table = bytearray(1 << (5 + mantissa_bits))
    fraction_mask = (1 << mantissa_bits) - 1
    for code in range(len(table)):
        exponent = code >> mantissa_bits
        fraction = code & fraction_mask
        if exponent == 31:
            table[code] = 0 if fraction else 255
            continue
        significand = fraction + ((1 << mantissa_bits) if exponent else 0)
        value = math.ldexp(significand, max(1, exponent) - 15 - mantissa_bits)
        table[code] = int(min(1.0, value) * 255 + 0.5)
    return bytes(table)


def _decode_r11g11b10(payload: memoryview, width: int, height: int, pitch: int) -> bytes:
    row_bytes = width * 4
    pitch = pitch or row_bytes
    if pitch < row_bytes:
        raise DdsError(f"Invalid R11G11B10_FLOAT row pitch: {pitch} bytes (need at least {row_bytes})")
    expected = (height - 1) * pitch + row_bytes
    if len(payload) < expected:
        raise DdsError(f"Truncated R11G11B10_FLOAT payload: {len(payload)} bytes (need {expected})")

    rg_table = _unsigned_float_to_u8(6)
    b_table = _unsigned_float_to_u8(5)
    rgba = bytearray(width * height * 4)
    rgba[3::4] = b"\xff" * (width * height)
    dst = 0
    for start in range(0, height * pitch, pitch):
        for (pixel,) in struct.iter_unpack("<I", payload[start:start + row_bytes]):
            rgba[dst] = rg_table[pixel & 0x7FF]
            rgba[dst + 1] = rg_table[(pixel >> 11) & 0x7FF]
            rgba[dst + 2] = b_table[pixel >> 22]
            dst += 4
    return bytes(rgba)


def _decode_single_channel(payload: memoryview, width: int, height: int, pitch: int, fmt: str) -> bytes:
    """Red-only R16_UNORM/R32_FLOAT and alpha-only A8_UNORM."""
    bpp = _TEXEL_BYTES[fmt]
    row_bytes = width * bpp
    pitch = pitch or row_bytes
    if pitch < row_bytes:
        raise DdsError(f"Invalid {fmt} row pitch: {pitch} bytes (need at least {row_bytes})")
    expected = (height - 1) * pitch + row_bytes
    if len(payload) < expected:
        raise DdsError(f"Truncated {fmt} payload: {len(payload)} bytes (need {expected})")

    rgba = bytearray(width * height * 4)
    if fmt != "A8_UNORM":
        rgba[3::4] = b"\xff" * (width * height)
    dst = 0
    for start in range(0, height * pitch, pitch):
        row = payload[start:start + row_bytes]
        if fmt == "R16_UNORM":
            for (v,) in struct.iter_unpack("<H", row):
                rgba[dst] = (v * 255 + 32767) // 65535
                dst += 4
        elif fmt == "R32_FLOAT":
            for (v,) in struct.iter_unpack("<f", row):
                if v >= 1:
                    rgba[dst] = 255
                elif v > 0:  # Negative values and NaN leave red at zero.
                    rgba[dst] = int(v * 255 + 0.5)
                dst += 4
        else:
            rgba[dst + 3:dst + 4 * width:4] = row
            dst += 4 * width
    return bytes(rgba)


def _bc7_read_indices(bits: int, index_bits: int, is_anchor: tuple[int, ...]) -> tuple[list[int], int]:
    """Read 16 LSB-first pixel indices; anchor pixels omit their implicit zero MSB."""
    indices = []
    for anchor in is_anchor:
        width = index_bits - anchor
        indices.append(bits & ((1 << width) - 1))
        bits >>= width
    return indices, bits


def _decode_bc7_block(block: int) -> bytes:
    """Decode one little-endian 128-bit BC7 block to 64 bytes of row-major RGBA8."""
    if not block & 0xFF:
        return _BC7_RESERVED_BLOCK
    mode = (block & -block).bit_length() - 1
    (subsets, partition_bits, rotation_bits, selector_bits, color_bits, alpha_bits,
     endpoint_pbits, shared_pbits, index_bits, index2_bits) = _BC7_MODES[mode]

    # Fields are LSB-first: mode, partition, rotation, index selector, endpoints,
    # p-bits, then the index set(s).
    bits = block >> (mode + 1)
    partition = bits & ((1 << partition_bits) - 1)
    bits >>= partition_bits
    rotation = bits & ((1 << rotation_bits) - 1)
    bits >>= rotation_bits
    selector = bits & ((1 << selector_bits) - 1)
    bits >>= selector_bits

    # All R endpoints, then all G, B and A, each ordered subset 0 end 0, subset 0 end 1, ...
    endpoint_count = 2 * subsets
    channel_bits = (color_bits,) * 3 + ((alpha_bits,) if alpha_bits else ())
    raw_channels = []
    for width in channel_bits:
        mask = (1 << width) - 1
        values = []
        for _ in range(endpoint_count):
            values.append(bits & mask)
            bits >>= width
        raw_channels.append(values)

    if endpoint_pbits:
        pbits = [(bits >> e) & 1 for e in range(endpoint_count)]
        bits >>= endpoint_count
    elif shared_pbits:
        pbits = [(bits >> (e >> 1)) & 1 for e in range(endpoint_count)]
        bits >>= subsets
    else:
        pbits = None

    endpoints = []
    for width, values in zip(channel_bits, raw_channels):
        if pbits is not None:
            values = [(v << 1) | p for v, p in zip(values, pbits)]
            width += 1
        unquantize = _BC7_UNQUANTIZE[width]
        endpoints.append([unquantize[v] for v in values])

    subset_of, is_anchor = _BC7_LAYOUTS[subsets][partition]
    color_indices, bits = _bc7_read_indices(bits, index_bits, is_anchor)
    color_weights = alpha_weights = _BC7_WEIGHTS[index_bits]
    if index2_bits:
        # Modes 4 and 5: the selector swaps which index set drives colour and alpha.
        alpha_indices, _ = _bc7_read_indices(bits, index2_bits, is_anchor)
        alpha_weights = _BC7_WEIGHTS[index2_bits]
        if selector:
            color_indices, alpha_indices = alpha_indices, color_indices
            color_weights, alpha_weights = alpha_weights, color_weights
    else:
        if subsets > 1:
            # Address the per-subset palettes concatenated below.
            stride = len(color_weights)
            color_indices = [s * stride + i for s, i in zip(subset_of, color_indices)]
        alpha_indices = color_indices

    # Alpha stays 255 in modes 0-3; rotation is applied after interpolation.
    out = bytearray(b"\xff" * 64)
    order = _BC7_ROTATIONS[rotation]
    for channel, ends in enumerate(endpoints):
        if channel == 3:
            weights, indices = alpha_weights, alpha_indices
        else:
            weights, indices = color_weights, color_indices
        palette = []
        for e0, e1 in zip(ends[0::2], ends[1::2]):
            palette += [(e0 * (64 - w) + e1 * w + 32) >> 6 for w in weights]
        out[order[channel]::4] = bytes(map(palette.__getitem__, indices))
    return bytes(out)


def _decode_bc7(data: bytes | bytearray | memoryview, offset: int, width: int, height: int) -> bytes:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    expected = blocks_x * blocks_y * 16
    if len(data) - offset < expected:
        raise DdsError(f"Truncated BC7 payload: {len(data) - offset} bytes (need {expected})")

    out = bytearray(width * height * 4)
    row_stride = width * 4
    cache: dict[bytes, bytes] = {}
    from_bytes = int.from_bytes
    pos = offset

    for by in range(blocks_y):
        py0 = by * 4
        max_y = height - py0
        if max_y > 4:
            max_y = 4
        row_base = py0 * row_stride

        for bx in range(blocks_x):
            key = bytes(data[pos : pos + 16])
            pos += 16
            block_rgba = cache.get(key)
            if block_rgba is None:
                block_rgba = _decode_bc7_block(from_bytes(key, "little"))
                if len(cache) < _BC7_BLOCK_CACHE_LIMIT:
                    cache[key] = block_rgba

            px0 = bx * 4
            span = min(width - px0, 4) * 4
            dst = row_base + px0 * 4
            for ty in range(max_y):
                src = ty * 16
                out[dst : dst + span] = block_rgba[src : src + span]
                dst += row_stride

    return bytes(out)


def _linear_to_srgb_byte(linear: float) -> int:
    return bisect_right(_SRGB_HALF_STEPS, linear)


def _unpremultiply_rgba(rgba: bytes, srgb: bool) -> bytes:
    """Straight RGBA8, rounding color channels to the nearest byte and keeping alpha unchanged."""
    out = bytearray(rgba)
    for p in range(0, len(out), 4):
        alpha = rgba[p + 3]
        if alpha == 0:
            out[p:p + 3] = b"\x00\x00\x00"
        elif alpha < 255:
            for c in range(p, p + 3):
                if srgb:
                    out[c] = _linear_to_srgb_byte(_SRGB_TO_LINEAR[rgba[c]] * 255.0 / alpha)
                else:
                    out[c] = min(255, (rgba[c] * 255 + alpha // 2) // alpha)
    return bytes(out)


def _decode_plane(data: bytes | bytearray | memoryview, plane: _Plane, info: _DdsHeader,
                  masks: tuple[int, int, int, int] | None, alpha_mode: int) -> tuple[int, int, bytes, bool]:
    """Decode data starting at the selected plane, in its source color space."""
    payload = memoryview(data)
    srgb = info["dxgi_format"] in _SRGB_DXGI_FORMATS
    width, height, pitch = plane.width, plane.height, plane.row_pitch
    fmt = info["format"]

    if fmt in ("RGBA8", "BGRA8", "BGRX8", "RGBX8"):
        rgba = _decode_uncompressed_32(payload, width, height, fmt, masks, pitch)
    elif fmt in _HALF_FLOAT_LAYOUT:
        rgba = _decode_half_float(payload, width, height, fmt, pitch)
    elif fmt in _RGB24_LAYOUTS.values():
        assert masks is not None
        rgba = _decode_uncompressed_24(payload, width, height, fmt, masks, pitch)
    elif fmt == "BC1":
        rgba = _decode_bc1(payload, 0, width, height)
    elif fmt == "BC2":
        rgba = _decode_bc2(payload, 0, width, height)
    elif fmt == "BC3":
        rgba = _decode_bc3(payload, 0, width, height)
    elif fmt == "BC4":
        rgba = _decode_bc4(payload, 0, width, height, signed=False)
    elif fmt == "BC4_SNORM":
        rgba = _decode_bc4(payload, 0, width, height, signed=True)
    elif fmt == "BC5":
        rgba = _decode_bc5(payload, 0, width, height, signed=False)
    elif fmt == "BC5_SNORM":
        rgba = _decode_bc5(payload, 0, width, height, signed=True)
    elif fmt == "R11G11B10_FLOAT":
        rgba = _decode_r11g11b10(payload, width, height, pitch)
    elif fmt in ("R16_UNORM", "R32_FLOAT", "A8_UNORM"):
        rgba = _decode_single_channel(payload, width, height, pitch, fmt)
    elif fmt == "BC7":
        rgba = _decode_bc7(payload, 0, width, height)
    else:
        raise DdsError(f"Unsupported DDS format: {fmt}")

    if alpha_mode == DDS_ALPHA_MODE_PREMULTIPLIED:
        rgba = _unpremultiply_rgba(rgba, srgb)
    elif alpha_mode == DDS_ALPHA_MODE_OPAQUE:
        opaque = bytearray(rgba)
        opaque[3::4] = b"\xff" * (width * height)
        rgba = bytes(opaque)

    return width, height, rgba, srgb


def _decode_dds(data: bytes, *, mip: int = 0, layer: int = 0, face: str | None = None,
                z: int = 0) -> tuple[int, int, bytes, bool]:
    info, masks, alpha_mode = _parse_header_internal(data)
    plane = _locate_plane(info, mip=mip, layer=layer, face=face, z=z)
    payload = memoryview(data)[plane.offset:plane.offset + plane.byte_length]
    return _decode_plane(payload, plane, info, masks, alpha_mode)


def decode_dds(data: bytes, *, mip: int = 0, layer: int = 0, face: str | None = None,
               z: int = 0) -> tuple[int, int, bytes]:
    """Decode one DDS image plane into top-down row-major RGBA8 with straight alpha."""
    width, height, rgba, _ = _decode_dds(data, mip=mip, layer=layer, face=face, z=z)
    return width, height, rgba


def _box_spans(source: int, target: int) -> list:
    """Source pixel indices and integer overlap weights for each output pixel."""
    spans = []
    for i in range(target):
        # In these units a source pixel is target units wide, and an output pixel is source units wide.
        left, right = i * source, (i + 1) * source
        span = []
        for j in range(left // target, (right - 1) // target + 1):
            weight = min(right, (j + 1) * target) - max(left, j * target)
            span.append((j, weight))
        spans.append(span)
    return spans


def _resample_box(width: int, height: int, rgba: bytes, nw: int, nh: int, srgb: bool = False) -> bytes:
    x_spans = _box_spans(width, nw)
    y_spans = _box_spans(height, nh)
    area = width * height
    out = bytearray(nw * nh * 4)
    for y, rows in enumerate(y_spans):
        for x, cols in enumerate(x_spans):
            samples = []
            for sy, wy in rows:
                for sx, wx in cols:
                    offset = 4 * (sy * width + sx)
                    samples.append((offset, rgba[offset + 3] * wx * wy))
            coverage = sum(weight for _, weight in samples)
            dst = 4 * (y * nw + x)
            out[dst + 3] = (coverage + area // 2) // area
            if coverage:
                for channel in range(3):
                    if srgb:
                        light = sum(_SRGB_TO_LINEAR[rgba[offset + channel]] * weight for offset, weight in samples)
                        out[dst + channel] = _linear_to_srgb_byte(light / coverage)
                    else:
                        total = sum(rgba[offset + channel] * weight for offset, weight in samples)
                        out[dst + channel] = (total + coverage // 2) // coverage
    return bytes(out)


def _halve_rgba(width: int, height: int, rgba: bytes, srgb: bool = False) -> tuple[int, int, bytes]:
    nw = max(1, width // 2)
    nh = max(1, height // 2)
    if (width > 1 and width % 2) or (height > 1 and height % 2) or rgba[3::4].count(255) != width * height:
        return nw, nh, _resample_box(width, height, rgba, nw, nh, srgb)
    row_bytes = width * 4
    out_row_bytes = nw * 4
    out = bytearray(nw * nh * 4)
    out[3::4] = b"\xff" * (nw * nh)
    # A column has the same contiguous pixel pairs as a single row.
    if width == 1:
        if height == 1:
            return nw, nh, rgba
        row_bytes, out_row_bytes = height * 4, nh * 4
    linear = _SRGB_TO_LINEAR
    for oy, s0 in enumerate(range(0, len(rgba), 2 * row_bytes)):
        r0 = rgba[s0 : s0 + row_bytes]
        r1 = rgba[s0 + row_bytes : s0 + 2 * row_bytes] or r0
        dst = oy * out_row_bytes
        dst_end = dst + out_row_bytes
        for c in range(3):
            samples = zip(r0[c::8], r0[c + 4 :: 8], r1[c::8], r1[c + 4 :: 8])
            if srgb:
                values = (_linear_to_srgb_byte((linear[a] + linear[b] + linear[d] + linear[e]) * 0.25)
                          for a, b, d, e in samples)
            else:
                values = ((a + b + d + e + 2) >> 2 for a, b, d, e in samples)
            out[dst + c : dst_end : 4] = bytes(values)

    return nw, nh, bytes(out)


def downscale_rgba(
    width: int, height: int, rgba: bytes, max_size: int, *, srgb: bool = False,
) -> tuple[int, int, bytes]:
    """Halve straight-alpha RGBA8 with an alpha-weighted box filter. Tagged sRGB uses linear light."""
    if width <= 0 or height <= 0:
        raise DdsError(f"Invalid image dimensions: {width}x{height}")
    expected = width * height * 4
    if len(rgba) != expected:
        raise DdsError(f"Invalid RGBA buffer length: {len(rgba)} (expected {expected})")

    if max_size < 0:
        raise DdsError(f"Invalid max_size: {max_size}")
    if max_size == 0 or max(width, height) <= max_size:
        return width, height, bytes(rgba) if not isinstance(rgba, bytes) else rgba

    w, h = width, height
    cur = bytes(rgba) if not isinstance(rgba, bytes) else rgba
    while max(w, h) > max_size:
        w, h, cur = _halve_rgba(w, h, cur, srgb)
    return w, h, cur


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def encode_png(width: int, height: int, rgba: bytes) -> bytes:
    """Encode top-down row-major RGBA8 bytes as a spec-compliant PNG."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {width}x{height}")
    stride = width * 4
    expected = stride * height
    if len(rgba) != expected:
        raise ValueError(f"Invalid RGBA buffer length: {len(rgba)} (expected {expected})")

    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))

    raw = bytearray(height * (stride + 1))
    mv = memoryview(rgba)
    dst = 1
    src = 0
    dst_step = stride + 1
    for _ in range(height):
        raw[dst : dst + stride] = mv[src : src + stride]
        dst += dst_step
        src += stride

    idat = _png_chunk(b"IDAT", zlib.compress(raw))
    iend = _png_chunk(b"IEND", b"")
    return PNG_SIGNATURE + ihdr + idat + iend


def _preview_png(image: tuple[int, int, bytes, bool], max_size: int) -> bytes:
    w, h, rgba, srgb = image
    if max_size > 0:
        w, h, rgba = downscale_rgba(w, h, rgba, max_size, srgb=srgb)
    return encode_png(w, h, rgba)


def _read_header(source: BinaryIO) -> bytes:
    header = source.read(DDS_HEADER_SIZE)
    if header[84:88] == b"DX10":
        header += source.read(DDS_DX10_HEADER_SIZE - DDS_HEADER_SIZE)
    return header


def dds_to_png_bytes(
    data: bytes, max_size: int = 0, *, mip: int = 0, layer: int = 0, face: str | None = None, z: int = 0,
) -> bytes:
    """Encode one DDS plane as PNG, optionally downscaling the selected mip."""
    if max_size < 0:
        raise DdsError(f"Invalid max_size: {max_size}")
    return _preview_png(_decode_dds(data, mip=mip, layer=layer, face=face, z=z), max_size)


def read_half_texel(path: str, *, x: int, y: int, mip: int = 0, layer: int = 0,
                    z: int = 0) -> dict[str, str]:
    """Return exact stored channel values as text, including signed zero and nonfinite values."""
    with open(path, "rb") as source:
        info = parse_dds_header(_read_header(source))
        fmt = info["format"]
        if fmt not in _HALF_FLOAT_LAYOUT:
            raise DdsError(f"Expected a half-float DDS, got {fmt}")
        stride, channels = _HALF_FLOAT_LAYOUT[fmt]
        plane = _locate_plane(info, mip=mip, layer=layer, z=z)
        if (type(x) is not int or not 0 <= x < plane.width or
                type(y) is not int or not 0 <= y < plane.height):
            raise DdsError("Texel outside the selected mip")
        if plane.offset + plane.byte_length > source.seek(0, os.SEEK_END):
            raise DdsError("Truncated DDS plane")
        source.seek(plane.offset + y * plane.row_pitch + x * stride)
        raw = source.read(stride)
        if len(raw) != stride:
            raise DdsError("Truncated DDS texel")
    return dict(zip("RGBA", map(str, struct.unpack(f"<{channels}e", raw))))


def dds_file_to_png_bytes(
    path: str, max_size: int = 0, *, mip: int = 0, layer: int = 0, face: str | None = None, z: int = 0,
) -> bytes:
    """Seek directly to a mip, array layer, cube face or volume slice and encode it as PNG."""
    if max_size < 0:
        raise DdsError(f"Invalid max_size: {max_size}")
    with open(path, "rb") as f:
        if not f.seekable():
            return dds_to_png_bytes(f.read(), max_size, mip=mip, layer=layer, face=face, z=z)
        info, masks, alpha_mode = _parse_header_internal(_read_header(f))
        plane = _locate_plane(info, mip=mip, layer=layer, face=face, z=z)
        # Check before seek/read: an invalid header may describe an offset larger than off_t.
        if plane.offset + plane.byte_length > f.seek(0, os.SEEK_END):
            raise DdsError(f"Truncated {info['format']} payload for the selected plane")
        f.seek(plane.offset)
        payload = f.read(plane.byte_length)
    return _preview_png(_decode_plane(payload, plane, info, masks, alpha_mode), max_size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Decode a DDS texture to PNG (pure Python stdlib).")
    ap.add_argument("input", help="Input .dds file path")
    ap.add_argument("output", nargs="?", help="Output .png file path")
    ap.add_argument("--info", action="store_true", help="Print DDS header metadata as JSON")
    ap.add_argument("--mip", type=int, help="Mip level (0 is full resolution)")
    ap.add_argument("--layer", type=int, help="Zero-based array element or cube index")
    ap.add_argument("--face", choices=_CUBE_FACES, help="Cubemap face (default: first face present)")
    ap.add_argument("--z", type=int, help="Zero-based volume slice within the selected mip")
    ap.add_argument(
        "--max-size",
        type=int,
        help="Downscale the selected plane to a maximum side length of N (0 = disabled)",
    )
    args = ap.parse_args(argv)
    if args.max_size is not None and args.max_size < 0:
        ap.error("--max-size must be 0 or more")
    export_options = (args.output, args.mip, args.layer, args.face, args.z, args.max_size)
    if args.info and any(value is not None for value in export_options):
        ap.error("--info takes only an input DDS")
    if not args.info and args.output is None:
        ap.error("an output PNG is required unless --info is used")

    try:
        if args.info:
            with open(args.input, "rb") as f:
                print(json.dumps(parse_dds_header(_read_header(f)), indent=2))
            return 0
        try:
            overwrites_input = os.path.samefile(args.input, args.output)
        except FileNotFoundError:
            overwrites_input = False
        if overwrites_input:
            raise DdsError("output must not overwrite the input DDS")
        png = dds_file_to_png_bytes(
            args.input, args.max_size or 0, mip=args.mip or 0, layer=args.layer or 0,
            face=args.face, z=args.z or 0,
        )
        # Resolve an existing symlink as the old direct write did, and stage on the same filesystem.
        output = os.path.realpath(args.output)
        parent = os.path.dirname(output)
        os.makedirs(parent, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dds-preview-", dir=parent) as staging:
            pending = os.path.join(staging, "preview.png")
            with open(pending, "wb") as f:
                f.write(png)
            if os.path.isfile(output):
                shutil.copymode(output, pending)
            os.replace(pending, output)
    except (DdsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
