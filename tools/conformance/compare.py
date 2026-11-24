#!/usr/bin/env python3
"""Verify rendered PNGs against declared pixel checks, and diff two renderers.

Every expectation comes from a manifest.json written by a renderer. This tool
never reads C++ or scene definitions: the images and the manifest are the whole
contract. PNG support is deliberately small and complete for that contract:
8-bit RGB or RGBA, non-interlaced, with all five filter types (0-4), because any
encoder that writes those formats may use any of them.
RGB must be opaque; tRNS transparency is rejected. Use RGBA for explicit alpha.

Exit codes, identical in every mode:
    0  everything the mode checked passed
    1  the mode ran and found a mismatch (pixels, thresholds or selftest)
    2  the mode could not run (bad arguments, unreadable or invalid input)
"""

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import zlib
from dataclasses import dataclass

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_ERROR = 2

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOUR_TYPE_RGB = 2
PNG_COLOUR_TYPE_RGBA = 6
PNG_BIT_DEPTH_8 = 8
PNG_COMPRESSION_DEFLATE = 0
PNG_FILTER_ADAPTIVE = 0
PNG_INTERLACE_NONE = 0
OPAQUE_ALPHA = 255
MAX_CHANNEL_VALUE = 255

# Diff arithmetic always works on RGBA so an RGB reference and an RGBA candidate
# are comparable; RGB images supply an implicit alpha of 255.
RGBA_CHANNEL_COUNT = 4

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVERAGE = 3
FILTER_PAETH = 4

# Dark grey rather than black so that a black pixel in a near-black render does
# not vanish into the "unchanged" background of a heatmap.
HEATMAP_UNCHANGED = (48, 48, 48)

# Coordinates beyond this are certainly a manifest bug, not a big render.
MAX_SANE_COORDINATE = 1_000_000

# Heatmaps are written as "<scene>.diff.png", so a scene name must be usable as
# a file name. These are Windows' forbidden characters; the tool runs on Windows.
UNSAFE_FILE_NAME_CHARACTERS = '<>:"/\\|?*'


class ToolError(Exception):
    """A problem the operator can act on: bad input, or a comparison that cannot run."""


class SelftestMismatch(Exception):
    """A selftest expectation did not hold; the message names the expectation."""


# ---------------------------------------------------------------------------
# PNG reading and writing
# ---------------------------------------------------------------------------


@dataclass
class Raster:
    """Decoded pixels plus the metadata a caller needs to interpret them."""

    width: int
    height: int
    colour_type: int
    pixels: bytearray
    source_path: str

    def channels(self):
        """Return 3 or 4, matching the stored PNG colour type."""
        return channels_for_colour_type(self.colour_type, self.source_path)

    def colour_name(self):
        """Return the PNG colour type as a short display name."""
        return "RGB" if self.channels() == 3 else "RGBA"

    def stored_pixel(self, x, y):
        """Return the channels actually stored at (x, y), three or four values."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise ToolError(
                f"pixel ({x},{y}) is outside the {self.width}x{self.height} image "
                f"'{self.source_path}'; check the requested coordinate or the render size"
            )
        channels = self.channels()
        offset = (y * self.width + x) * channels
        return tuple(self.pixels[offset:offset + channels])

    def pixel_rgba(self, x, y):
        """Return (r, g, b, a); RGB images are opaque, because they store no alpha."""
        stored = self.stored_pixel(x, y)
        if self.colour_type == PNG_COLOUR_TYPE_RGB:
            return stored + (OPAQUE_ALPHA,)
        return stored


def channels_for_colour_type(colour_type, source_path):
    """Return 3 channels for RGB and 4 for RGBA; reject every other PNG colour type."""
    if colour_type == PNG_COLOUR_TYPE_RGB:
        return 3
    if colour_type == PNG_COLOUR_TYPE_RGBA:
        return 4
    raise ToolError(
        f"'{source_path}': PNG colour type {colour_type} is not supported; "
        f"only 2 (RGB) and 6 (RGBA) are. Re-export without a palette or grayscale mode"
    )


def read_file_bytes(path, description):
    """Read a file whole, naming the path and the OS error when it cannot."""
    try:
        with open(path, "rb") as file_handle:
            return file_handle.read()
    except OSError as error:
        raise ToolError(
            f"cannot read {description} '{path}': {error.strerror or error}. "
            f"Check that the file exists and is readable"
        )


def read_text_file(path, description):
    """Read UTF-8 text, accepting an optional BOM, naming the bad byte on failure."""
    raw_bytes = read_file_bytes(path, description)
    try:
        # utf-8-sig accepts a BOM, which PowerShell-written manifests may carry.
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ToolError(
            f"'{path}' is not valid UTF-8 text: byte {error.start} is invalid. "
            f"Check that the file was not saved in a legacy code page"
        )


def write_file_bytes(path, data, description):
    """Write bytes into an existing directory; never creates one."""
    try:
        with open(path, "wb") as file_handle:
            file_handle.write(data)
    except OSError as error:
        raise ToolError(
            f"cannot write {description} '{path}': {error.strerror or error}. "
            f"Check that the directory exists and is writable"
        )


def split_png_chunks(file_bytes, source_path):
    """Split a PNG into (type, data) chunks, verifying every length and CRC.

    IEND ends the chunk list; bytes after it are rejected rather than ignored so a
    concatenated or padded file is reported instead of silently half-read.
    """
    if not file_bytes.startswith(PNG_SIGNATURE):
        raise ToolError(
            f"'{source_path}' is not a PNG: the 8-byte signature is missing. "
            f"Check that the renderer wrote a PNG and not a raw or JPEG dump"
        )

    chunks = []
    offset = len(PNG_SIGNATURE)
    while True:
        header_end = offset + 8
        if header_end > len(file_bytes):
            raise ToolError(
                f"'{source_path}': truncated chunk header at byte offset {offset}; "
                f"the file ends before the chunk length and type. Check for a partial write"
            )
        chunk_length = int.from_bytes(file_bytes[offset:offset + 4], "big")
        chunk_type = file_bytes[offset + 4:header_end]
        data_end = header_end + chunk_length
        crc_end = data_end + 4
        if crc_end > len(file_bytes):
            raise ToolError(
                f"'{source_path}': truncated {chunk_type.decode('ascii', 'replace')} chunk "
                f"at byte offset {offset}: declared length {chunk_length} runs past the end "
                f"of the file. Check for a partial write"
            )
        chunk_data = file_bytes[header_end:data_end]
        stored_crc = int.from_bytes(file_bytes[data_end:crc_end], "big")
        computed_crc = png_chunk_crc(chunk_type, chunk_data)
        if stored_crc != computed_crc:
            raise ToolError(
                f"'{source_path}': CRC mismatch in {chunk_type.decode('ascii', 'replace')} "
                f"chunk at byte offset {offset}: stored {stored_crc:08x}, computed "
                f"{computed_crc:08x}. The file is corrupt; re-render it"
            )
        chunks.append((chunk_type, chunk_data))
        offset = crc_end
        if chunk_type == b"IEND":
            break

    if offset != len(file_bytes):
        raise ToolError(
            f"'{source_path}': {len(file_bytes) - offset} byte(s) follow IEND at offset "
            f"{offset}. Check that two files were not concatenated"
        )
    return chunks


def parse_image_header(chunks, source_path):
    """Validate IHDR and return (width, height, colour_type)."""
    first_type, first_data = chunks[0]
    if first_type != b"IHDR":
        raise ToolError(
            f"'{source_path}': first chunk is "
            f"{first_type.decode('ascii', 'replace')}, not IHDR; the file is not a valid PNG"
        )
    if len(first_data) != 13:
        raise ToolError(
            f"'{source_path}': IHDR is {len(first_data)} bytes, expected 13. "
            f"The file is corrupt; re-render it"
        )

    width = int.from_bytes(first_data[0:4], "big")
    height = int.from_bytes(first_data[4:8], "big")
    bit_depth = first_data[8]
    colour_type = first_data[9]
    compression_method = first_data[10]
    filter_method = first_data[11]
    interlace_method = first_data[12]

    if width == 0 or height == 0:
        raise ToolError(
            f"'{source_path}': IHDR declares {width}x{height}; a PNG must have at "
            f"least one pixel in each direction"
        )
    if bit_depth != PNG_BIT_DEPTH_8:
        raise ToolError(
            f"'{source_path}': bit depth {bit_depth} is not supported; only 8 is. "
            f"Re-export from the renderer without 16-bit or packed formats"
        )
    if colour_type not in (PNG_COLOUR_TYPE_RGB, PNG_COLOUR_TYPE_RGBA):
        raise ToolError(
            f"'{source_path}': colour type {colour_type} is not supported; only 2 (RGB) "
            f"and 6 (RGBA) are. Re-export without palette or grayscale mode"
        )
    if compression_method != PNG_COMPRESSION_DEFLATE:
        raise ToolError(
            f"'{source_path}': compression method {compression_method} is not deflate(0); "
            f"the file is not a PNG this tool can read"
        )
    if filter_method != PNG_FILTER_ADAPTIVE:
        raise ToolError(
            f"'{source_path}': filter method {filter_method} is not adaptive(0); "
            f"the file is not a PNG this tool can read"
        )
    if interlace_method != PNG_INTERLACE_NONE:
        raise ToolError(
            f"'{source_path}': interlace method {interlace_method}; only non-interlaced(0) "
            f"is supported. Re-export as non-interlaced"
        )
    return width, height, colour_type


def png_chunk_crc(chunk_type, chunk_data):
    """CRC-32 over a PNG chunk's type and data, as the format defines it."""
    return zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF


def read_png(path):
    """Read a supported PNG from disk. Raises ToolError naming the file on any defect."""
    file_bytes = read_file_bytes(path, "PNG image")
    chunks = split_png_chunks(file_bytes, path)
    width, height, colour_type = parse_image_header(chunks, path)
    if any(chunk_type == b"tRNS" for chunk_type, _ in chunks):
        if colour_type == PNG_COLOUR_TYPE_RGBA:
            raise ToolError(f"'{path}': tRNS is forbidden in an RGBA PNG; re-export a valid RGBA image")
        raise ToolError(
            f"'{path}': PNG tRNS transparency is not supported; re-export as RGBA "
            f"with an explicit alpha channel. Ignoring tRNS would compare transparent pixels as opaque"
        )

    compressed_data = b"".join(
        chunk_data for chunk_type, chunk_data in chunks if chunk_type == b"IDAT"
    )
    if not compressed_data:
        raise ToolError(
            f"'{path}': no IDAT chunk, so there are no pixels. The file is incomplete; "
            f"re-render it"
        )
    try:
        decompressed_data = zlib.decompress(compressed_data)
    except zlib.error as error:
        raise ToolError(
            f"'{path}': IDAT does not decompress as zlib: {error}. "
            f"Check that the file is complete and not a partial write"
        )

    channels = channels_for_colour_type(colour_type, path)
    pixels = unfilter_image(decompressed_data, width, height, channels, path)
    return Raster(width, height, colour_type, pixels, path)


def unfilter_image(filtered_bytes, width, height, channels, source_path):
    """Turn filtered scanlines into packed pixels, one filter type per row.

    PNG filters each row against the row above, so decoding must proceed top to
    bottom, feeding each reconstructed row into the next row's predictor.
    """
    row_byte_count = width * channels
    row_stride = row_byte_count + 1  # filter-type byte plus the row
    expected_length = row_stride * height
    if len(filtered_bytes) != expected_length:
        raise ToolError(
            f"'{source_path}': decompressed pixel data is {len(filtered_bytes)} bytes, "
            f"expected {expected_length} for {width}x{height} with {channels} channels "
            f"(rows include one filter-type byte). Check the render size and bit depth"
        )

    pixels = bytearray(row_byte_count * height)
    # The specification defines the row above the first row as zeroes, so the
    # first row's Up, Average and Paeth predictors have a defined input.
    previous_row = bytes(row_byte_count)
    for row_index in range(height):
        row_start = row_index * row_stride
        filter_type = filtered_bytes[row_start]
        filtered_row = filtered_bytes[row_start + 1:row_start + row_stride]
        reconstructed_row = unfilter_row(
            filter_type, filtered_row, previous_row, channels, row_index, source_path
        )
        row_offset = row_index * row_byte_count
        pixels[row_offset:row_offset + row_byte_count] = reconstructed_row
        previous_row = reconstructed_row
    return pixels


def unfilter_row(filter_type, filtered_row, previous_row, bytes_per_pixel,
                 row_index, source_path):
    """Dispatch one scanline to its filter decoder; reject undefined filter types."""
    if filter_type == FILTER_NONE:
        return unfilter_none(filtered_row)
    if filter_type == FILTER_SUB:
        return unfilter_sub(filtered_row, bytes_per_pixel)
    if filter_type == FILTER_UP:
        return unfilter_up(filtered_row, previous_row)
    if filter_type == FILTER_AVERAGE:
        return unfilter_average(filtered_row, previous_row, bytes_per_pixel)
    if filter_type == FILTER_PAETH:
        return unfilter_paeth(filtered_row, previous_row, bytes_per_pixel)
    raise ToolError(
        f"'{source_path}': row {row_index} uses PNG filter type {filter_type}; "
        f"only 0-4 are defined. The file is corrupt; re-render it"
    )


def unfilter_none(filtered_row):
    """Return the scanline unchanged (filter 0)."""
    return bytes(filtered_row)


def unfilter_sub(filtered_row, bytes_per_pixel):
    """Add the byte one pixel to the left in the same row (filter 1)."""
    reconstructed = bytearray(len(filtered_row))
    for index in range(len(filtered_row)):
        left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        # PNG filter arithmetic is modulo 256; masking keeps the byte in range.
        reconstructed[index] = (filtered_row[index] + left) & 0xFF
    return bytes(reconstructed)


def unfilter_up(filtered_row, previous_row):
    """Add the byte directly above in the previous row (filter 2)."""
    reconstructed = bytearray(len(filtered_row))
    for index in range(len(filtered_row)):
        reconstructed[index] = (filtered_row[index] + previous_row[index]) & 0xFF
    return bytes(reconstructed)


def unfilter_average(filtered_row, previous_row, bytes_per_pixel):
    """Add the floor of the left and above average (filter 3)."""
    reconstructed = bytearray(len(filtered_row))
    for index in range(len(filtered_row)):
        left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        above = previous_row[index]
        reconstructed[index] = (filtered_row[index] + ((left + above) // 2)) & 0xFF
    return bytes(reconstructed)


def unfilter_paeth(filtered_row, previous_row, bytes_per_pixel):
    """Add the Paeth predictor of left, above and upper-left (filter 4)."""
    reconstructed = bytearray(len(filtered_row))
    for index in range(len(filtered_row)):
        left = reconstructed[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        above = previous_row[index]
        upper_left = previous_row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        predictor = paeth_predictor(left, above, upper_left)
        reconstructed[index] = (filtered_row[index] + predictor) & 0xFF
    return bytes(reconstructed)


def paeth_predictor(left, above, upper_left):
    """Return PNG's Paeth predictor for one byte.

    The predictor is whichever neighbour is closest to left + above - upper_left.
    Ties go left, then above, then upper-left; that order is part of the PNG
    specification and swapping it silently corrupts smooth gradients.
    """
    estimate = left + above - upper_left
    distance_left = abs(estimate - left)
    distance_above = abs(estimate - above)
    distance_upper_left = abs(estimate - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def make_chunk(chunk_type, chunk_data):
    """Wrap chunk data as length, type, data and CRC, as PNG requires."""
    length_prefix = len(chunk_data).to_bytes(4, "big")
    crc = png_chunk_crc(chunk_type, chunk_data)
    return length_prefix + chunk_type + chunk_data + crc.to_bytes(4, "big")


def write_png(path, raster):
    """Write an 8-bit RGB or RGBA PNG, filter type 0 on every row."""
    channels = channels_for_colour_type(raster.colour_type, path)
    row_byte_count = raster.width * channels
    if len(raster.pixels) != row_byte_count * raster.height:
        raise ToolError(
            f"cannot write '{path}': raster holds {len(raster.pixels)} bytes, expected "
            f"{row_byte_count * raster.height} for {raster.width}x{raster.height} "
            f"{raster.colour_name()}"
        )

    filtered_rows = bytearray()
    for row_index in range(raster.height):
        filtered_rows.append(FILTER_NONE)
        row_offset = row_index * row_byte_count
        filtered_rows.extend(raster.pixels[row_offset:row_offset + row_byte_count])

    header = (
        raster.width.to_bytes(4, "big")
        + raster.height.to_bytes(4, "big")
        + bytes((PNG_BIT_DEPTH_8, raster.colour_type, PNG_COMPRESSION_DEFLATE,
                 PNG_FILTER_ADAPTIVE, PNG_INTERLACE_NONE))
    )
    file_bytes = (
        PNG_SIGNATURE
        + make_chunk(b"IHDR", header)
        + make_chunk(b"IDAT", zlib.compress(bytes(filtered_rows), 9))
        + make_chunk(b"IEND", b"")
    )
    write_file_bytes(path, file_bytes, "PNG image")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestCheck:
    x: int
    y: int
    expected_red: int
    expected_green: int
    expected_blue: int
    expected_alpha: int
    reason: str


@dataclass(frozen=True)
class ManifestScene:
    name: str
    png_name: str
    checks: tuple


@dataclass(frozen=True)
class Manifest:
    manifest_path: str
    renderer: str
    adapter: str
    scenes: tuple


def require_key(mapping, key, context):
    """Return a required mapping key, or fail naming the context and key."""
    if key not in mapping:
        raise ToolError(f'{context}: required key "{key}" is missing')
    return mapping[key]


def require_string(mapping, key, context):
    """Return a required string value and reject other JSON types."""
    value = require_key(mapping, key, context)
    if not isinstance(value, str):
        raise ToolError(f'{context}: "{key}" must be a string, got {value!r}')
    return value


def require_list(mapping, key, context):
    """Return a required JSON list and reject other JSON types."""
    value = require_key(mapping, key, context)
    if not isinstance(value, list):
        raise ToolError(f'{context}: "{key}" must be a list, got {value!r}')
    return value


def require_int(mapping, key, minimum, maximum, context):
    """Return a required integer in range, rejecting booleans and floats."""
    value = require_key(mapping, key, context)
    # bool is an int subclass; true would otherwise pass as coordinate 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f'{context}: "{key}" must be an integer, got {value!r}')
    if value < minimum or value > maximum:
        raise ToolError(
            f'{context}: "{key}" must be between {minimum} and {maximum}, got {value}'
        )
    return value


def parse_manifest_check(raw_check, check_index, scene_context):
    """Validate one declared pixel into a ManifestCheck."""
    context = f"{scene_context}, check {check_index + 1}"
    if not isinstance(raw_check, dict):
        raise ToolError(f"{context}: must be an object, got {raw_check!r}")
    reason = raw_check.get("why", "")
    if not isinstance(reason, str):
        raise ToolError(f'{context}: "why" must be a string when present, got {reason!r}')
    return ManifestCheck(
        x=require_int(raw_check, "x", 0, MAX_SANE_COORDINATE, context),
        y=require_int(raw_check, "y", 0, MAX_SANE_COORDINATE, context),
        expected_red=require_int(raw_check, "r", 0, MAX_CHANNEL_VALUE, context),
        expected_green=require_int(raw_check, "g", 0, MAX_CHANNEL_VALUE, context),
        expected_blue=require_int(raw_check, "b", 0, MAX_CHANNEL_VALUE, context),
        expected_alpha=require_int(raw_check, "a", 0, MAX_CHANNEL_VALUE, context),
        reason=reason,
    )


def require_safe_file_name(value, key, context, reason):
    """Reject a manifest string that would escape its directory or break a path."""
    has_control_character = any(ord(character) < 32 for character in value)
    has_unsafe_character = any(
        character in UNSAFE_FILE_NAME_CHARACTERS for character in value
    )
    if (os.path.basename(value) != value or value in (".", "..")
            or has_control_character or has_unsafe_character):
        raise ToolError(
            f'{context}: "{key}" must be a plain file name; got {value!r}. {reason}'
        )
    return value


def parse_manifest_scene(raw_scene, scene_index, manifest_path):
    """Validate one scene entry, including a non-empty check list."""
    context = f"manifest '{manifest_path}': scene {scene_index + 1}"
    if not isinstance(raw_scene, dict):
        raise ToolError(f"{context}: must be an object, got {raw_scene!r}")
    name = require_string(raw_scene, "name", context)
    if not name.strip():
        raise ToolError(f"{context}: \"name\" is empty; it identifies the scene and the diff row")
    name = require_safe_file_name(
        name, "name", context, "It is also used as the <scene>.diff.png file name"
    )
    context = f"manifest '{manifest_path}': scene {scene_index + 1} ('{name}')"

    png_name = require_string(raw_scene, "png", context)
    png_name = require_safe_file_name(
        png_name, "png", context, "It must name an image beside the manifest"
    )
    raw_checks = require_list(raw_scene, "checks", context)
    if not raw_checks:
        raise ToolError(
            f"{context}: \"checks\" is empty; a scene with no pixel checks proves "
            f"nothing. Add at least one check or remove the scene"
        )
    checks = tuple(
        parse_manifest_check(raw_check, check_index, context)
        for check_index, raw_check in enumerate(raw_checks)
    )
    return ManifestScene(name=name, png_name=png_name, checks=checks)


def parse_manifest_scenes(raw_scenes, manifest_path):
    """Validate all scenes and reject duplicate scene or PNG names."""
    scenes = []
    scene_number_by_name = {}
    png_number_by_name = {}
    for scene_index, raw_scene in enumerate(raw_scenes):
        scene = parse_manifest_scene(raw_scene, scene_index, manifest_path)
        if scene.name in scene_number_by_name:
            raise ToolError(
                f"manifest '{manifest_path}': scene {scene_index + 1} repeats name "
                f"'{scene.name}' from scene {scene_number_by_name[scene.name] + 1}; "
                f"diff matches scenes by name, so duplicates are ambiguous"
            )
        if scene.png_name in png_number_by_name:
            raise ToolError(
                f"manifest '{manifest_path}': scene {scene_index + 1} repeats png "
                f"'{scene.png_name}' from scene {png_number_by_name[scene.png_name] + 1}; "
                f"diff would write the same .diff.png twice"
            )
        scene_number_by_name[scene.name] = scene_index
        png_number_by_name[scene.png_name] = scene_index
        scenes.append(scene)
    return tuple(scenes)


def load_manifest(path):
    """Load and validate a renderer manifest. Unknown extra keys are ignored."""
    manifest_path = os.path.abspath(path)
    text = read_text_file(manifest_path, "scene manifest")

    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ToolError(f"manifest '{manifest_path}': duplicate key {key!r}")
            obj[key] = value
        return obj

    try:
        document = json.loads(text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as error:
        raise ToolError(
            f"manifest '{manifest_path}' is not valid JSON: {error.msg} at line "
            f"{error.lineno} column {error.colno}. Check for a missing comma or "
            f"trailing text"
        )
    if not isinstance(document, dict):
        raise ToolError(
            f"manifest '{manifest_path}': top level must be an object, got "
            f"{type(document).__name__}"
        )

    context = f"manifest '{manifest_path}'"
    renderer = require_string(document, "renderer", context)
    adapter = require_string(document, "adapter", context)
    raw_scenes = require_list(document, "scenes", context)
    if not raw_scenes:
        raise ToolError(
            f"manifest '{manifest_path}': \"scenes\" is empty; there is nothing to "
            f"check. Point --manifest at the renderer's manifest.json"
        )
    return Manifest(
        manifest_path=manifest_path,
        renderer=renderer,
        adapter=adapter,
        scenes=parse_manifest_scenes(raw_scenes, manifest_path),
    )


def resolve_image_directory(manifest, override):
    """Return the absolute image directory, checking that an explicit --dir exists."""
    if override:
        if not os.path.isdir(override):
            raise ToolError(
                f"image directory '{override}' does not exist or is not a directory. "
                f"Check the --dir argument"
            )
        return os.path.abspath(override)
    return os.path.dirname(manifest.manifest_path)


def scene_image_path(image_directory, png_name):
    """Join an image directory with a manifest's PNG file name."""
    return os.path.join(image_directory, png_name)


# ---------------------------------------------------------------------------
# expect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckOutcome:
    scene_name: str
    check: ManifestCheck
    actual: tuple
    failure_note: str
    # Decided from the comparison itself. Deriving it from failure_note made the verdict
    # depend on whether a "why" string happened to be present.
    passed_check: bool

    @property
    def passed(self):
        """True when the pixel matched; failure_note is display text only."""
        return self.passed_check


@dataclass(frozen=True)
class SceneOutcome:
    scene: ManifestScene
    checks: tuple
    image_problem: str = ""


def evaluate_check(scene_name, image, check):
    """Compare one declared pixel; bounds and absent alpha are explained, not raised."""
    if check.x >= image.width or check.y >= image.height:
        note = (
            f"coordinate is outside the {image.width}x{image.height} image "
            f"'{image.source_path}'; check the manifest coordinate and the render size"
        )
        return CheckOutcome(scene_name, check, (), note, False)

    stored = image.stored_pixel(check.x, check.y)
    declared = (check.expected_red, check.expected_green, check.expected_blue,
                check.expected_alpha)
    if image.pixel_rgba(check.x, check.y) == declared:
        return CheckOutcome(scene_name, check, stored, "", True)

    if image.colour_type == PNG_COLOUR_TYPE_RGB and check.expected_alpha != OPAQUE_ALPHA:
        note = (
            f"image stores no alpha channel, so declared alpha {check.expected_alpha} "
            f"cannot match; {check.reason or 'fix the check or render RGBA'}"
        )
    else:
        note = check.reason or "pixel does not match the declared value"
    return CheckOutcome(scene_name, check, stored, note, False)


def verify_scene(scene, image_directory):
    """Read one scene's image and evaluate every check; image errors stay scene-local."""
    image_path = scene_image_path(image_directory, scene.png_name)
    try:
        image = read_png(image_path)
    except ToolError as error:
        return SceneOutcome(scene, (), str(error))
    checks = tuple(
        evaluate_check(scene.name, image, check) for check in scene.checks
    )
    return SceneOutcome(scene, checks)


def format_channel_values(values):
    """Render channel values as comma-separated decimals for the report."""
    return ",".join(str(value) for value in values)


def plural(count, singular):
    """Return the singular or plural noun for a count, for readable summaries."""
    return singular if count == 1 else singular + "s"


def format_check_line(outcome, scene_column_width):
    """Format one PASS or FAIL row with expected, actual and failure reason."""
    status = "PASS" if outcome.passed else "FAIL"
    coordinate = f"({outcome.check.x},{outcome.check.y})"
    expected = format_channel_values((
        outcome.check.expected_red,
        outcome.check.expected_green,
        outcome.check.expected_blue,
        outcome.check.expected_alpha,
    ))
    got = format_channel_values(outcome.actual) if outcome.actual else "not read"
    line = (
        f"{status}  {outcome.scene_name:<{scene_column_width}}  {coordinate:<12}  "
        f"expected {expected:<17}  got {got}"
    )
    if not outcome.passed:
        line += f"  ({outcome.failure_note})"
    return line


def print_expect_report(scene_outcomes):
    """Print every check row and the summary; return True when all passed."""
    scene_column_width = max(len(outcome.scene.name) for outcome in scene_outcomes)
    passed_count = 0
    failed_count = 0
    unreadable_count = 0
    for scene_outcome in scene_outcomes:
        if scene_outcome.image_problem:
            unreadable_count += 1
            failed_count += len(scene_outcome.scene.checks)
            print(
                f"FAIL  {scene_outcome.scene.name:<{scene_column_width}}  "
                f"image problem: {scene_outcome.image_problem}"
            )
            continue
        for check_outcome in scene_outcome.checks:
            print(format_check_line(check_outcome, scene_column_width))
            if check_outcome.passed:
                passed_count += 1
            else:
                failed_count += 1

    check_count = passed_count + failed_count
    summary = (
        f"{len(scene_outcomes)} {plural(len(scene_outcomes), 'scene')}, "
        f"{check_count} {plural(check_count, 'check')}: "
        f"{passed_count} passed, {failed_count} failed"
    )
    if unreadable_count:
        summary += (
            f", {unreadable_count} {plural(unreadable_count, 'image')} "
            f"could not be read"
        )
    print()
    print(summary)
    return failed_count == 0


def command_expect(arguments):
    """Run the expect mode and return its exit code."""
    manifest = load_manifest(arguments.manifest)
    image_directory = resolve_image_directory(manifest, arguments.dir)
    scene_outcomes = [
        verify_scene(scene, image_directory) for scene in manifest.scenes
    ]
    return EXIT_PASSED if print_expect_report(scene_outcomes) else EXIT_FAILED


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffStats:
    differing_pixels: int
    total_pixels: int
    differing_percent: float
    mean_absolute_delta: float
    max_channel_delta: int
    first_difference: tuple
    # The RGBA values at first_difference, so a failure line can show what actually
    # differed instead of only how much.
    first_reference_pixel: tuple
    first_candidate_pixel: tuple


def compare_rasters(reference, candidate):
    """Compare two same-size images using RGBA; RGB images count as opaque."""
    if (reference.width, reference.height) != (candidate.width, candidate.height):
        raise ToolError(
            f"cannot compare '{reference.source_path}' "
            f"({reference.width}x{reference.height}) with '{candidate.source_path}' "
            f"({candidate.width}x{candidate.height}); re-render at the same size"
        )

    total_pixels = reference.width * reference.height
    # Every RGBA sample compared, so this is 4x the pixel count, not the pixel count.
    channel_sample_count = total_pixels * RGBA_CHANNEL_COUNT
    differing_pixels = 0
    total_absolute_delta = 0
    max_channel_delta = 0
    first_difference = None
    first_reference_pixel = ()
    first_candidate_pixel = ()

    for y in range(reference.height):
        for x in range(reference.width):
            reference_pixel = reference.pixel_rgba(x, y)
            candidate_pixel = candidate.pixel_rgba(x, y)
            pixel_delta = 0
            for channel_index in range(RGBA_CHANNEL_COUNT):
                channel_delta = abs(reference_pixel[channel_index]
                                    - candidate_pixel[channel_index])
                total_absolute_delta += channel_delta
                pixel_delta = max(pixel_delta, channel_delta)
            if pixel_delta == 0:
                continue
            differing_pixels += 1
            max_channel_delta = max(max_channel_delta, pixel_delta)
            if first_difference is None:
                first_difference = (x, y)
                first_reference_pixel = reference_pixel
                first_candidate_pixel = candidate_pixel

    return DiffStats(
        differing_pixels=differing_pixels,
        total_pixels=total_pixels,
        differing_percent=100.0 * differing_pixels / total_pixels,
        # Mean over every channel sample of every pixel. Read it together with
        # differing_pixels and max_channel_delta: on its own, one pixel off by 255 and 255
        # pixels off by 1 give the same mean, so the mean alone cannot rank a failure.
        mean_absolute_delta=total_absolute_delta / channel_sample_count,
        max_channel_delta=max_channel_delta,
        first_difference=first_difference,
        first_reference_pixel=first_reference_pixel,
        first_candidate_pixel=first_candidate_pixel,
    )


def heatmap_colour(channel_delta):
    """Map delta 1 to pure blue and delta 255 to pure red, on a linear ramp."""
    if channel_delta <= 0:
        return HEATMAP_UNCHANGED
    red = round(MAX_CHANNEL_VALUE * (channel_delta - 1) / (MAX_CHANNEL_VALUE - 1))
    return (red, 0, MAX_CHANNEL_VALUE - red)


def build_heatmap(reference, candidate):
    """Build an RGB image: unchanged grey, differences ramped blue to red."""
    if (reference.width, reference.height) != (candidate.width, candidate.height):
        raise ToolError(
            f"cannot build a heatmap from {reference.width}x{reference.height} and "
            f"{candidate.width}x{candidate.height} images; sizes must match"
        )
    pixels = bytearray(reference.width * reference.height * 3)
    for y in range(reference.height):
        for x in range(reference.width):
            reference_pixel = reference.pixel_rgba(x, y)
            candidate_pixel = candidate.pixel_rgba(x, y)
            pixel_delta = max(
                abs(reference_pixel[channel] - candidate_pixel[channel])
                for channel in range(RGBA_CHANNEL_COUNT)
            )
            offset = (y * reference.width + x) * 3
            pixels[offset:offset + 3] = bytes(heatmap_colour(pixel_delta))
    return Raster(
        width=reference.width,
        height=reference.height,
        colour_type=PNG_COLOUR_TYPE_RGB,
        pixels=pixels,
        source_path=os.path.join(
            os.path.dirname(candidate.source_path),
            "difference heatmap for " + os.path.basename(candidate.source_path),
        ),
    )


def format_rgba(pixel):
    """Format an RGBA tuple for a failure line; empty reads as a dash."""
    if not pixel:
        return "-"
    return ",".join(str(channel) for channel in pixel)


def threshold_violation(stats, max_differing_percent, max_mean_delta):
    """Return why the scene fails the thresholds, or "" when it passes."""
    if stats.differing_percent > max_differing_percent:
        return (
            f"differing pixels {stats.differing_percent:.3f}% exceed "
            f"--max-differing-percent {max_differing_percent:.3f}%"
        )
    if stats.mean_absolute_delta > max_mean_delta:
        return (
            f"mean absolute channel delta {stats.mean_absolute_delta:.4f} exceeds "
            f"--max-mean-delta {max_mean_delta:.4f}"
        )
    return ""


def whole_frame_difference_note(scene_name, stats):
    """Explain a whole-frame difference, with the values that identify the cause.

    A whole-frame difference is not automatically the tint: an unblended draw of the
    correct colour also differs everywhere, and so does a scene that was never drawn.
    The pixel values are what tell those apart, so they are printed.
    """
    if stats.total_pixels == 0 or stats.differing_pixels != stats.total_pixels:
        return ""
    reference_text = format_rgba(stats.first_reference_pixel)
    candidate_text = format_rgba(stats.first_candidate_pixel)
    return (
        f"scene '{scene_name}': every pixel differs - at the first difference the "
        f"reference is {reference_text} and the candidate is {candidate_text}. "
        f"Suspects, in order: the constant buffer tint, the clear colour, blend state, "
        f"or a draw that never happened"
    )


def format_diff_line(scene_name, stats, scene_column_width, violation):
    """Format one scene's diff metrics as a single table row."""
    status = "FAIL" if violation else "PASS"
    if stats.first_difference:
        first = f"({stats.first_difference[0]},{stats.first_difference[1]})"
    else:
        first = "-"
    line = (
        f"{status}  {scene_name:<{scene_column_width}}  "
        f"differing {stats.differing_pixels}/{stats.total_pixels} "
        f"({stats.differing_percent:.3f}%)  mean {stats.mean_absolute_delta:.4f}  "
        f"max {stats.max_channel_delta}  first {first}"
    )
    if violation:
        line += f"  ({violation})"
    return line


def match_manifest_scenes(reference, candidate):
    """Require both manifests to declare exactly the same scene names."""
    reference_names = {scene.name for scene in reference.scenes}
    candidate_names = {scene.name for scene in candidate.scenes}
    missing_names = sorted(reference_names - candidate_names)
    extra_names = sorted(candidate_names - reference_names)
    if missing_names:
        raise ToolError(
            f"candidate manifest '{candidate.manifest_path}' is missing scene(s) the "
            f"reference declares: {', '.join(missing_names)}. Re-render them"
        )
    if extra_names:
        raise ToolError(
            f"candidate manifest '{candidate.manifest_path}' has scenes the reference "
            f"does not declare: {', '.join(extra_names)}. Remove them or update the reference"
        )


def command_diff(arguments):
    """Run the diff mode: metrics, heatmaps, thresholds and exit code."""
    reference = load_manifest(arguments.reference)
    candidate = load_manifest(arguments.candidate)
    reference_directory = resolve_image_directory(reference, arguments.reference_dir)
    candidate_directory = resolve_image_directory(candidate, arguments.candidate_dir)
    match_manifest_scenes(reference, candidate)

    print(f"reference: {reference.manifest_path}")
    print(f"  renderer '{reference.renderer}', adapter '{reference.adapter}'")
    print(f"candidate: {candidate.manifest_path}")
    print(f"  renderer '{candidate.renderer}', adapter '{candidate.adapter}'")
    print()
    # Flush before a later read can fail onto stderr, so the error is not
    # printed above the header it belongs to when output is redirected.
    sys.stdout.flush()

    # Each manifest names its own files, so look up the candidate scene itself
    # rather than assuming both renderers wrote the same PNG file name.
    candidate_scenes_by_name = {scene.name: scene for scene in candidate.scenes}
    scene_column_width = max(len(scene.name) for scene in reference.scenes)
    failing_scenes = []
    for scene in reference.scenes:
        candidate_scene = candidate_scenes_by_name[scene.name]
        reference_image = read_png(scene_image_path(reference_directory, scene.png_name))
        candidate_image = read_png(
            scene_image_path(candidate_directory, candidate_scene.png_name)
        )
        stats = compare_rasters(reference_image, candidate_image)
        heatmap = build_heatmap(reference_image, candidate_image)
        heatmap_path = os.path.join(candidate_directory, f"{scene.name}.diff.png")
        write_png(heatmap_path, heatmap)

        violation = threshold_violation(
            stats, arguments.max_differing_percent, arguments.max_mean_delta
        )
        print(format_diff_line(scene.name, stats, scene_column_width, violation))
        whole_frame_note = whole_frame_difference_note(scene.name, stats)
        if whole_frame_note:
            print(f"      {whole_frame_note}")
        if violation:
            failing_scenes.append(scene.name)

    print()
    scene_count = len(reference.scenes)
    print(
        f"{scene_count} {plural(scene_count, 'scene')} compared, "
        f"{len(failing_scenes)} exceed the thresholds. Heatmaps written beside the "
        f"candidate images as <scene>.diff.png"
    )
    return EXIT_FAILED if failing_scenes else EXIT_PASSED


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def pixel_coordinate_argument(text):
    """Parse an argparse X,Y pixel argument into an (x, y) tuple."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not x,y; use for example --pixel 12,34"
        )
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{text}' has a non-integer part; use for example --pixel 12,34"
        )
    return (x, y)


def format_inspection_line(label, x, y, channel_values):
    """Format one labelled pixel row for inspect output."""
    coordinate = f"({x},{y})"
    return (
        f"  {label:<12} {coordinate:<11}= "
        f"{format_channel_values(channel_values)}"
    )


def command_inspect(arguments):
    """Run the inspect mode and return its exit code.

    Top-left and bottom-right are always printed so a vertical flip is visible:
    the corners are labelled and their coordinates are explicit.
    """
    image = read_png(arguments.image)
    print(arguments.image)
    print(
        f"  {image.width} x {image.height}, {image.colour_name()}, 8 bits per channel, "
        f"non-interlaced"
    )
    sys.stdout.flush()
    bottom_right_x = image.width - 1
    bottom_right_y = image.height - 1
    print(format_inspection_line("top-left", 0, 0, image.stored_pixel(0, 0)))
    print(
        format_inspection_line(
            "bottom-right", bottom_right_x, bottom_right_y,
            image.stored_pixel(bottom_right_x, bottom_right_y),
        )
    )
    if arguments.pixel:
        for x, y in arguments.pixel:
            print(format_inspection_line("pixel", x, y, image.stored_pixel(x, y)))
    else:
        print("  no --pixel given; also showing top-right, bottom-left and centre")
        for label, x, y in (
            ("top-right", bottom_right_x, 0),
            ("bottom-left", 0, bottom_right_y),
            ("centre", image.width // 2, image.height // 2),
        ):
            print(format_inspection_line(label, x, y, image.stored_pixel(x, y)))
    return EXIT_PASSED


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def require_equal(description, actual, expected):
    """Fail the selftest unless actual equals expected."""
    if actual != expected:
        raise SelftestMismatch(
            f"{description}: expected {expected!r}, got {actual!r}"
        )


def require_true(description, condition):
    """Fail the selftest unless the condition holds."""
    if not condition:
        raise SelftestMismatch(description)


def assert_tool_error(action, expected_fragment, description):
    """Require the action to raise ToolError containing an expected fragment."""
    try:
        action()
    except ToolError as error:
        require_true(
            f"{description}: message should contain {expected_fragment!r}, got '{error}'",
            expected_fragment in str(error),
        )
        return
    raise SelftestMismatch(f"{description}: expected a ToolError, but the call succeeded")


def assert_argument_error(action, description):
    """Require the action to raise argparse.ArgumentTypeError."""
    try:
        action()
    except argparse.ArgumentTypeError:
        return
    raise SelftestMismatch(
        f"{description}: expected an ArgumentTypeError, but the call succeeded"
    )


def make_solid_raster(width, height, colour_type, colour, source_path):
    """Build an in-memory raster filled with one colour."""
    return Raster(
        width=width,
        height=height,
        colour_type=colour_type,
        pixels=bytearray(bytes(colour) * (width * height)),
        source_path=source_path,
    )


def write_selftest_manifest(path, renderer, adapter, scenes):
    """Write a manifest in the renderer's shape for selftest fixtures."""
    document = {"renderer": renderer, "adapter": adapter, "scenes": scenes}
    write_file_bytes(
        path, json.dumps(document, indent=2).encode("utf-8"), "selftest manifest"
    )


def run_and_capture(command, arguments):
    """Run a command with stdout captured; return its exit code and text."""
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        exit_code = command(arguments)
    return exit_code, output_buffer.getvalue()


def make_selftest_directory(path):
    """Create one selftest fixture directory, reporting failure as a mismatch."""
    try:
        os.makedirs(path)
    except OSError as error:
        raise SelftestMismatch(
            f"cannot create selftest fixture directory '{path}': {error}"
        )


def step_png_roundtrip(directory):
    """Assert RGB and RGBA round-trip, rectangle pixels and write determinism."""
    solid = make_solid_raster(8, 6, PNG_COLOUR_TYPE_RGB, (10, 20, 30), "selftest solid")
    solid_path = os.path.join(directory, "solid.png")
    write_png(solid_path, solid)
    decoded = read_png(solid_path)
    require_equal("solid PNG width", decoded.width, 8)
    require_equal("solid PNG height", decoded.height, 6)
    require_equal("solid PNG colour type", decoded.colour_type, PNG_COLOUR_TYPE_RGB)
    require_equal("solid PNG pixels", bytes(decoded.pixels), bytes(solid.pixels))
    require_equal("solid PNG pixel (0,0)", decoded.stored_pixel(0, 0), (10, 20, 30))
    print("ok    PNG round-trip: RGB pixels and alpha-free pixel reads")

    rectangle = make_solid_raster(
        8, 6, PNG_COLOUR_TYPE_RGBA, (10, 20, 30, 255), "selftest rectangle"
    )
    for y in range(1, 5):
        for x in range(2, 6):
            offset = (y * 8 + x) * 4
            rectangle.pixels[offset:offset + 4] = bytes((200, 30, 60, 200))
    rectangle_path = os.path.join(directory, "rectangle.png")
    write_png(rectangle_path, rectangle)
    decoded_rectangle = read_png(rectangle_path)
    require_equal(
        "rectangle PNG pixel inside the rectangle",
        decoded_rectangle.stored_pixel(2, 1),
        (200, 30, 60, 200),
    )
    require_equal(
        "rectangle PNG pixel at its bottom-right corner",
        decoded_rectangle.stored_pixel(5, 4),
        (200, 30, 60, 200),
    )
    require_equal(
        "rectangle PNG pixel left of the rectangle",
        decoded_rectangle.pixel_rgba(1, 1),
        (10, 20, 30, 255),
    )
    require_equal(
        "rectangle PNG pixel right of the rectangle",
        decoded_rectangle.stored_pixel(6, 1),
        (10, 20, 30, 255),
    )
    print("ok    PNG round-trip: RGBA rectangle at (2,1)-(5,4), alpha 200 preserved")

    second_path = os.path.join(directory, "solid-again.png")
    write_png(second_path, solid)
    require_equal(
        "two writes of the same raster are byte-identical",
        read_file_bytes(second_path, "selftest PNG"),
        read_file_bytes(solid_path, "selftest PNG"),
    )
    print("ok    PNG writer is deterministic: two writes of one raster match byte for byte")


def step_png_error_paths(directory):
    """Assert unwritable directory, corrupt CRC and truncation are reported."""
    solid = make_solid_raster(8, 6, PNG_COLOUR_TYPE_RGB, (10, 20, 30), "selftest solid")
    solid_path = os.path.join(directory, "error-fixture.png")
    write_png(solid_path, solid)

    missing_directory_path = os.path.join(directory, "missing-subdirectory", "image.png")
    assert_tool_error(
        lambda: write_png(missing_directory_path, solid),
        "cannot write",
        "writing into a missing directory",
    )
    print("ok    failure path: unwritable output directory names the path")

    crc_path = os.path.join(directory, "bad-crc.png")
    write_png(crc_path, solid)
    corrupt_bytes = bytearray(read_file_bytes(crc_path, "selftest PNG"))
    corrupt_bytes[45] ^= 0xFF  # inside IDAT data: offset 33 starts IDAT, 8 bytes of header
    write_file_bytes(crc_path, bytes(corrupt_bytes), "selftest PNG")
    assert_tool_error(
        lambda: read_png(crc_path),
        "CRC mismatch",
        "reading a PNG with a corrupted IDAT",
    )
    print("ok    failure path: corrupt IDAT reports the CRC and byte offset")

    truncated_path = os.path.join(directory, "truncated.png")
    truncated_bytes = read_file_bytes(solid_path, "selftest PNG")[:60]
    write_file_bytes(truncated_path, truncated_bytes, "selftest PNG")
    assert_tool_error(
        lambda: read_png(truncated_path),
        "truncated",
        "reading a truncated PNG",
    )
    print("ok    failure path: truncated file reports the byte offset")


def step_filter_decoders():
    """Assert filters 0-4 decode hand-computed streams and bad streams fail."""
    # 3x2 RGB fixture: row 0 is (10,20,30) (200,30,60) (105,25,45);
    # row 1 is (40,50,60) (70,80,90) (100,110,120). The five filtered streams
    # were computed by hand from the PNG filter definitions, independently of
    # the decoder, so a mirrored encoder bug cannot hide here.
    expected_pixels = bytes((
        10, 20, 30, 200, 30, 60, 105, 25, 45,
        40, 50, 60, 70, 80, 90, 100, 110, 120,
    ))
    filtered_streams = {
        FILTER_NONE: bytes((
            FILTER_NONE, 10, 20, 30, 200, 30, 60, 105, 25, 45,
            FILTER_NONE, 40, 50, 60, 70, 80, 90, 100, 110, 120,
        )),
        FILTER_SUB: bytes((
            FILTER_SUB, 10, 20, 30, 190, 10, 30, 161, 251, 241,
            FILTER_SUB, 40, 50, 60, 30, 30, 30, 30, 30, 30,
        )),
        FILTER_UP: bytes((
            FILTER_UP, 10, 20, 30, 200, 30, 60, 105, 25, 45,
            FILTER_UP, 30, 30, 30, 126, 50, 30, 251, 85, 75,
        )),
        FILTER_AVERAGE: bytes((
            FILTER_AVERAGE, 10, 20, 30, 195, 20, 45, 5, 10, 15,
            FILTER_AVERAGE, 35, 40, 45, 206, 40, 30, 13, 58, 53,
        )),
        FILTER_PAETH: bytes((
            FILTER_PAETH, 10, 20, 30, 190, 10, 30, 161, 251, 241,
            FILTER_PAETH, 30, 30, 30, 126, 30, 30, 30, 30, 30,
        )),
    }
    for filter_type in sorted(filtered_streams):
        decoded = unfilter_image(filtered_streams[filter_type], 3, 2, 3,
                                 "selftest filter fixture")
        require_equal(
            f"filter {filter_type} reconstructs the fixture",
            bytes(decoded),
            expected_pixels,
        )
    print("ok    PNG filters 0-4 decode: hand-computed streams for a 3x2 RGB fixture")

    truncated_stream = filtered_streams[FILTER_NONE][:-1]
    assert_tool_error(
        lambda: unfilter_image(truncated_stream, 3, 2, 3, "selftest filter fixture"),
        "decompressed pixel data",
        "filter data one byte short",
    )
    unknown_filter_stream = bytearray(filtered_streams[FILTER_NONE])
    unknown_filter_stream[0] = 5
    assert_tool_error(
        lambda: unfilter_image(bytes(unknown_filter_stream), 3, 2, 3, "selftest fixture"),
        "filter type 5",
        "undefined filter type",
    )
    print("ok    failure path: short filter data and undefined filter type are rejected")


def step_orientation_and_channel_order(directory):
    """Assert row 0 is the top, channels stay R,G,B, and inspect labels both corners."""
    top_colour = (200, 30, 60)
    bottom_colour = (10, 20, 30)
    image = make_solid_raster(
        4, 4, PNG_COLOUR_TYPE_RGB, bottom_colour, "selftest orientation"
    )
    for y in range(2):
        for x in range(4):
            offset = (y * 4 + x) * 3
            image.pixels[offset:offset + 3] = bytes(top_colour)

    image_path = os.path.join(directory, "orientation.png")
    write_png(image_path, image)
    decoded = read_png(image_path)
    require_equal("row 0 keeps the top colour", decoded.stored_pixel(0, 0), top_colour)
    require_equal("last row keeps the bottom colour", decoded.stored_pixel(3, 3),
                  bottom_colour)
    require_equal("channels stay in R,G,B order", decoded.stored_pixel(0, 0),
                  (200, 30, 60))

    exit_code, output = run_and_capture(
        command_inspect, argparse.Namespace(image=image_path, pixel=None)
    )
    require_equal("inspect exit code", exit_code, EXIT_PASSED)
    require_true("inspect labels top-left with its value",
                 "top-left" in output and "(0,0)" in output and "200,30,60" in output)
    require_true("inspect labels bottom-right with its value",
                 "bottom-right" in output and "(3,3)" in output and "10,20,30" in output)
    print("ok    orientation: top and bottom halves survive; inspect labels both corners")


def step_expect_matching(directory):
    """Assert expect passes checks that match the fixture image."""
    image_path = os.path.join(directory, "clear.png")
    write_png(
        image_path,
        make_solid_raster(8, 6, PNG_COLOUR_TYPE_RGB, (10, 20, 30), "selftest clear"),
    )
    manifest_path = os.path.join(directory, "manifest-good.json")
    write_selftest_manifest(manifest_path, "selftest", "fixture", [{
        "name": "clear",
        "png": "clear.png",
        "checks": [
            {"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "top-left is the clear colour"},
            {"x": 7, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "top-right is the clear colour"},
            {"x": 0, "y": 5, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "bottom-left is the clear colour"},
            {"x": 7, "y": 5, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "bottom-right is the clear colour"},
            {"x": 4, "y": 3, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "centre-ish pixel is the clear colour"},
        ],
    }])
    exit_code, output = run_and_capture(
        command_expect, argparse.Namespace(manifest=manifest_path, dir=None)
    )
    require_equal("expect exit code for matching pixels", exit_code, EXIT_PASSED)
    require_true("expect reports all five checks", output.count("PASS  clear") == 5)
    require_true("expect output has no FAIL row", "FAIL" not in output)
    require_equal("expect evaluates checks in any count", len(
        load_manifest(manifest_path).scenes[0].checks
    ), 5)
    print("ok    expect: a five-check manifest exits 0 and prints five PASS rows")

    bom_path = os.path.join(directory, "manifest-bom.json")
    bom_bytes = b"\xef\xbb\xbf" + read_file_bytes(manifest_path, "selftest manifest")
    write_file_bytes(bom_path, bom_bytes, "selftest manifest")
    require_equal("manifest with a UTF-8 BOM still loads",
                  len(load_manifest(bom_path).scenes), 1)
    print("ok    manifest reader accepts a UTF-8 BOM, as PowerShell writes")


def step_expect_failures(directory):
    """Assert a wrong pixel and an out-of-range coordinate both fail."""
    image_path = os.path.join(directory, "clear.png")
    write_png(
        image_path,
        make_solid_raster(8, 6, PNG_COLOUR_TYPE_RGB, (10, 20, 30), "selftest clear"),
    )
    bad_manifest_path = os.path.join(directory, "manifest-bad.json")
    write_selftest_manifest(bad_manifest_path, "selftest", "fixture", [{
        "name": "clear",
        "png": "clear.png",
        "checks": [
            {"x": 0, "y": 0, "r": 1, "g": 2, "b": 3, "a": 255,
             "why": "deliberate wrong expectation"},
            {"x": 8, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255,
             "why": "coordinate one past the right edge"},
        ],
    }])
    exit_code, output = run_and_capture(
        command_expect, argparse.Namespace(manifest=bad_manifest_path, dir=None)
    )
    require_equal("expect exit code for a wrong pixel", exit_code, EXIT_FAILED)
    require_true("wrong pixel failure quotes its why", "deliberate wrong expectation" in output)
    require_true("out-of-range pixel names the image size", "outside the 8x6 image" in output)
    print("ok    expect: wrong pixel and out-of-range coordinate fail with explanations")

    manifest = load_manifest(bad_manifest_path)
    scene_outcome = verify_scene(manifest.scenes[0], directory)
    require_equal("declared checks are all evaluated", len(scene_outcome.checks), 2)
    require_true("first check fails", not scene_outcome.checks[0].passed)
    require_equal("failed check records the stored pixel", scene_outcome.checks[0].actual,
                  (10, 20, 30))
    require_true("second check fails on bounds", not scene_outcome.checks[1].passed)


def step_expect_missing_and_invalid(directory):
    """Assert a missing image fails the scene and an empty check list is rejected."""
    missing_manifest_path = os.path.join(directory, "manifest-missing.json")
    write_selftest_manifest(missing_manifest_path, "selftest", "fixture", [{
        "name": "clear",
        "png": "absent.png",
        "checks": [{"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255}],
    }])
    exit_code, output = run_and_capture(
        command_expect, argparse.Namespace(manifest=missing_manifest_path, dir=None)
    )
    require_equal("expect exit code for a missing image", exit_code, EXIT_FAILED)
    require_true("missing image message names the file", "absent.png" in output)
    require_true(
        "missing image message says what to check",
        "Check that the file exists" in output,
    )
    print("ok    expect: a missing image fails once for the scene and names the path")

    empty_checks_manifest_path = os.path.join(directory, "manifest-empty.json")
    write_selftest_manifest(empty_checks_manifest_path, "selftest", "fixture", [{
        "name": "clear", "png": "clear.png", "checks": [],
    }])
    assert_tool_error(
        lambda: load_manifest(empty_checks_manifest_path),
        "proves nothing",
        "manifest scene with no checks",
    )
    print("ok    failure path: a scene with no checks is rejected, not silently passed")

    duplicate_path = os.path.join(directory, "manifest-duplicate.json")
    for key, document in (
        ("renderer", b'{"renderer":"first","renderer":"second"}'),
        ("r", b'{"scenes":[{"checks":[{"r":255,"r":0}]}]}'),
    ):
        write_file_bytes(duplicate_path, document, "duplicate key fixture")
        assert_tool_error(lambda: load_manifest(duplicate_path),
                          f"manifest '{duplicate_path}': duplicate key {key!r}",
                          f"duplicate manifest key {key}")
    print("ok    manifest: duplicate top-level and nested keys cannot hide declarations")


def step_diff_metrics():
    """Assert diff metrics and heatmap colours for one known two-pixel change."""
    reference = make_solid_raster(4, 3, PNG_COLOUR_TYPE_RGB, (0, 20, 30), "selftest reference")
    candidate = Raster(
        width=4, height=3, colour_type=PNG_COLOUR_TYPE_RGB,
        pixels=bytearray(reference.pixels), source_path="selftest candidate",
    )
    candidate.pixels[(1 * 4 + 1) * 3:(1 * 4 + 1) * 3 + 3] = bytes((255, 20, 30))
    candidate.pixels[(2 * 4 + 2) * 3:(2 * 4 + 2) * 3 + 3] = bytes((0, 20, 31))

    stats = compare_rasters(reference, candidate)
    require_equal("differing pixel count", stats.differing_pixels, 2)
    require_equal("total pixel count", stats.total_pixels, 12)
    require_equal("first differing coordinate", stats.first_difference, (1, 1))
    require_equal("max channel delta", stats.max_channel_delta, 255)
    require_equal("differing percent", round(stats.differing_percent, 6),
                  round(100.0 * 2 / 12, 6))
    require_equal("mean absolute channel delta", round(stats.mean_absolute_delta, 6),
                  round(256 / 48, 6))
    require_equal("identical rasters report no difference",
                  compare_rasters(reference, reference).differing_pixels, 0)

    heatmap = build_heatmap(reference, candidate)
    require_equal("heatmap unchanged pixel is dark grey", heatmap.stored_pixel(0, 0),
                  HEATMAP_UNCHANGED)
    require_equal("heatmap delta 255 pixel is red", heatmap.stored_pixel(1, 1), (255, 0, 0))
    require_equal("heatmap delta 1 pixel is blue", heatmap.stored_pixel(2, 2), (0, 0, 255))
    print("ok    diff metrics: 2 known pixels give exact count, percent, mean, max and first")
    print("ok    heatmap: grey unchanged, red at delta 255, blue at delta 1")


def step_diff_edge_cases():
    """Assert scan order, alpha-only differences and size-mismatch refusal."""
    reference = make_solid_raster(
        4, 3, PNG_COLOUR_TYPE_RGB, (0, 20, 30), "selftest reference"
    )
    row_order_candidate = Raster(
        width=4, height=3, colour_type=PNG_COLOUR_TYPE_RGB,
        pixels=bytearray(reference.pixels), source_path="selftest row order",
    )
    row_order_candidate.pixels[3 * 3:3 * 3 + 3] = bytes((1, 20, 30))
    row_order_candidate.pixels[(1 * 4) * 3:(1 * 4) * 3 + 3] = bytes((2, 20, 30))
    require_equal(
        "first difference follows row-major order, not column order",
        compare_rasters(reference, row_order_candidate).first_difference,
        (3, 0),
    )

    alpha_pixels = bytearray()
    for offset in range(0, len(reference.pixels), 3):
        alpha_pixels.extend(reference.pixels[offset:offset + 3])
        alpha_pixels.append(OPAQUE_ALPHA)
    alpha_candidate = Raster(
        width=4, height=3, colour_type=PNG_COLOUR_TYPE_RGBA,
        pixels=alpha_pixels, source_path="selftest alpha",
    )
    alpha_candidate.pixels[3] = OPAQUE_ALPHA - 1
    alpha_stats = compare_rasters(reference, alpha_candidate)
    require_equal("alpha-only difference is counted", alpha_stats.differing_pixels, 1)
    require_equal("alpha-only difference has delta 1", alpha_stats.max_channel_delta, 1)
    require_equal("alpha-only difference coordinate", alpha_stats.first_difference, (0, 0))

    wrong_tint = make_solid_raster(
        4, 3, PNG_COLOUR_TYPE_RGB, (200, 30, 60), "selftest wrong tint"
    )
    wrong_tint_stats = compare_rasters(reference, wrong_tint)
    whole_frame_note = whole_frame_difference_note("fullscreen", wrong_tint_stats)
    require_true("whole-frame hint names the scene", "'fullscreen'" in whole_frame_note)
    require_true("whole-frame hint points at the tint", "tint" in whole_frame_note)
    require_equal("partial difference gets no whole-frame hint",
                  whole_frame_difference_note("clear", alpha_stats), "")
    print("ok    diff edge cases: row-major first difference, alpha-only delta, tint hint")

    larger = make_solid_raster(5, 3, PNG_COLOUR_TYPE_RGB, (0, 20, 30), "selftest wider")
    assert_tool_error(
        lambda: compare_rasters(reference, larger),
        "re-render at the same size",
        "comparing images of different sizes",
    )
    print("ok    failure path: mismatched image sizes refuse to compare and say so")


def step_diff_command(directory):
    """Assert diff exit codes and the heatmap file it writes."""
    reference_directory = os.path.join(directory, "reference-images")
    candidate_directory = os.path.join(directory, "candidate-images")
    make_selftest_directory(reference_directory)
    make_selftest_directory(candidate_directory)

    reference_image = make_solid_raster(
        4, 3, PNG_COLOUR_TYPE_RGB, (0, 20, 30), "selftest reference image"
    )
    candidate_image = Raster(
        width=4, height=3, colour_type=PNG_COLOUR_TYPE_RGB,
        pixels=bytearray(reference_image.pixels), source_path="selftest candidate image",
    )
    candidate_image.pixels[(0 * 4 + 3) * 3:(0 * 4 + 3) * 3 + 3] = bytes((255, 20, 30))
    write_png(os.path.join(reference_directory, "clear.png"), reference_image)
    # Different file names prove diff reads each manifest's own png field.
    write_png(os.path.join(candidate_directory, "shim-clear.png"), candidate_image)

    reference_manifest_path = os.path.join(reference_directory, "manifest.json")
    candidate_manifest_path = os.path.join(candidate_directory, "manifest.json")
    common = {
        "name": "clear",
        "checks": [{"x": 0, "y": 0, "r": 10, "g": 20, "b": 30, "a": 255}],
    }
    write_selftest_manifest(
        reference_manifest_path, "native", "fixture",
        [dict(common, png="clear.png")],
    )
    write_selftest_manifest(
        candidate_manifest_path, "shim", "fixture",
        [dict(common, png="shim-clear.png")],
    )

    arguments = argparse.Namespace(
        reference=reference_manifest_path,
        candidate=candidate_manifest_path,
        reference_dir=None,
        candidate_dir=None,
        max_differing_percent=0.0,
        max_mean_delta=0.0,
    )
    exit_code, output = run_and_capture(command_diff, arguments)
    require_equal("diff exit code for a changed pixel", exit_code, EXIT_FAILED)
    require_true("diff row names the scene", "FAIL  clear" in output)
    diff_path = os.path.join(candidate_directory, "clear.diff.png")
    require_true("diff wrote the heatmap beside the candidate", os.path.isfile(diff_path))
    require_equal("written heatmap marks the changed pixel red",
                  read_png(diff_path).stored_pixel(3, 0), (255, 0, 0))
    print("ok    diff: changed scene exits 1 and writes clear.diff.png beside the candidate")

    arguments.max_differing_percent = 100.0
    arguments.max_mean_delta = 1000.0
    exit_code, output = run_and_capture(command_diff, arguments)
    require_equal("diff exit code inside generous thresholds", exit_code, EXIT_PASSED)
    require_true("relaxed thresholds still report the pixel count",
                  "differing 1/12" in output)
    print("ok    diff: generous thresholds pass the same scene without hiding its delta")


def step_argument_parsing():
    """Assert threshold and pixel arguments accept good text and reject the rest."""
    require_equal("zero threshold", non_negative_float("0"), 0.0)
    require_equal("fractional threshold", non_negative_float("1.5"), 1.5)
    require_equal("pixel coordinate", pixel_coordinate_argument("12,34"), (12, 34))
    for bad_threshold in ("-1", "nan", "inf", "abc"):
        assert_argument_error(
            lambda text=bad_threshold: non_negative_float(text),
            f"threshold {bad_threshold!r}",
        )
    for bad_pixel in ("12", "1,2,3", "a,b"):
        assert_argument_error(
            lambda text=bad_pixel: pixel_coordinate_argument(text),
            f"pixel {bad_pixel!r}",
        )
    print("ok    arguments: thresholds reject negative, NaN, inf and text; pixels need x,y")


def run_selftest_steps(directory):
    """Run every selftest step in order; each step prints its own ok line."""
    step_png_roundtrip(directory)
    step_png_error_paths(directory)
    step_filter_decoders()
    step_argument_parsing()
    step_orientation_and_channel_order(directory)
    step_expect_matching(directory)
    step_expect_failures(directory)
    step_expect_missing_and_invalid(directory)
    step_diff_metrics()
    step_diff_edge_cases()
    step_diff_command(directory)


def command_selftest(arguments):
    """Run the selftest in a temporary directory and return its exit code."""
    try:
        working_directory = tempfile.mkdtemp(prefix="compare-selftest-")
    except OSError as error:
        raise ToolError(
            f"selftest cannot create a temporary directory: {error}. "
            f"Check that %TEMP% is writable"
        )
    print(f"selftest: fixtures in {working_directory}")
    try:
        run_selftest_steps(working_directory)
    except SelftestMismatch as mismatch:
        print(f"FAIL  {mismatch}")
        print(f"fixtures kept for inspection: {working_directory}")
        return EXIT_FAILED
    except ToolError as error:
        # A broken fixture is a failed selftest, not a usage error: the tool ran.
        print(f"FAIL  selftest could not complete: {error}")
        print(f"fixtures kept for inspection: {working_directory}")
        return EXIT_FAILED
    shutil.rmtree(working_directory, ignore_errors=True)
    print("selftest passed")
    return EXIT_PASSED


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def non_negative_float(text):
    """argparse type for a finite threshold that may be zero but never negative."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number")
    # NaN would compare false against every threshold and silently pass all scenes.
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"'{text}' is not a finite number")
    if value < 0:
        raise argparse.ArgumentTypeError(f"'{text}' is negative; use 0 or more")
    return value


def add_expect_command(subparsers):
    """Register the expect subcommand and its arguments."""
    expect_parser = subparsers.add_parser(
        "expect",
        help="verify every pixel declared in a manifest",
        description=(
            "Read each scene's PNG, compare every declared check, and print one PASS "
            "or FAIL line per check with expected and actual values. A missing or "
            "unreadable image fails its whole scene with the path and what to check."
        ),
    )
    expect_parser.add_argument("--manifest", required=True,
                               help="manifest.json written by the renderer")
    expect_parser.add_argument(
        "--dir", default=None,
        help="directory holding the PNGs (default: the manifest's directory)",
    )
    expect_parser.set_defaults(handler=command_expect)


def add_diff_command(subparsers):
    """Register the diff subcommand and its arguments."""
    diff_parser = subparsers.add_parser(
        "diff",
        help="compare two renderers' images scene by scene",
        description=(
            "For every scene the reference manifest declares, compare its PNG against "
            "the candidate's. Prints differing pixel count, percent, mean absolute "
            "channel delta, maximum channel delta and the first differing coordinate. "
            "Writes <scene>.diff.png beside the candidate image: unchanged pixels are "
            "dark grey, differing pixels ramp from blue (delta 1) to red (delta 255)."
        ),
    )
    diff_parser.add_argument("--reference", required=True,
                             help="manifest.json for the reference renderer")
    diff_parser.add_argument("--candidate", required=True,
                             help="manifest.json for the renderer under test")
    diff_parser.add_argument(
        "--reference-dir", default=None,
        help="reference image directory (default: that manifest's directory)",
    )
    diff_parser.add_argument(
        "--candidate-dir", default=None,
        help="candidate image directory (default: that manifest's directory)",
    )
    diff_parser.add_argument(
        "--max-differing-percent", type=non_negative_float, default=0.0,
        help="fail a scene above this percent of differing pixels (default 0.0)",
    )
    diff_parser.add_argument(
        "--max-mean-delta", type=non_negative_float, default=0.0,
        help="fail a scene above this mean absolute channel delta (default 0.0)",
    )
    diff_parser.set_defaults(handler=command_diff)


def add_inspect_command(subparsers):
    """Register the inspect subcommand and its arguments."""
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="print an image's size, colour type and pixel values",
        description=(
            "Decode one PNG and print its dimensions, colour type and the requested "
            "pixels. Without --pixel it prints the four corners and the centre."
        ),
    )
    inspect_parser.add_argument("--image", required=True, help="PNG file to read")
    inspect_parser.add_argument(
        "--pixel", type=pixel_coordinate_argument, action="append", default=None,
        metavar="X,Y", help="pixel to print, for example --pixel 12,34 (repeatable)",
    )
    inspect_parser.set_defaults(handler=command_inspect)


def add_selftest_command(subparsers):
    """Register the selftest subcommand."""
    selftest_parser = subparsers.add_parser(
        "selftest",
        help="prove the reader, writer, expect, diff and heatmap on built-in fixtures",
        description=(
            "Write fixture PNGs into a temporary directory, then assert known pixel "
            "reads, all five PNG filters, expectation matching, diff metrics, heatmap "
            "colours and the failure paths. Leaves nothing behind; exits 1 on a mismatch."
        ),
    )
    selftest_parser.set_defaults(handler=command_selftest)


def build_argument_parser():
    """Build the full CLI: shared help text plus four subcommands."""
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description=(
            "Check rendered PNGs against the pixel checks a manifest declares, or "
            "measure how far two renderers' images differ. The tool reads only "
            "manifest.json and PNG files; it never parses C++."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python compare.py expect --manifest D:\\conform\\native\\manifest.json\n"
            "  python compare.py diff --reference native\\manifest.json "
            "--candidate shim\\manifest.json\n"
            "  python compare.py inspect --image shim\\clear.png --pixel 0,0\n"
            "  python compare.py selftest\n"
            "\n"
            "exit codes: 0 passed, 1 mismatch found, 2 could not run (bad input).\n"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="mode", required=True, metavar="{expect,diff,inspect,selftest}"
    )
    add_expect_command(subparsers)
    add_diff_command(subparsers)
    add_inspect_command(subparsers)
    add_selftest_command(subparsers)
    return parser


def main(argv=None):
    """Parse arguments, run one mode and translate failures into exit codes.

    0 passed, 1 the run completed and found a mismatch, 2 the tool could not run.
    An exception the tool did not anticipate is a 2, never a 1: a calling script must be
    able to tell "the pixels differ" from "this program crashed".
    """
    arguments = build_argument_parser().parse_args(argv)
    try:
        return arguments.handler(arguments)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - deliberate: report and exit 2, never mask as a mismatch
        traceback.print_exc()
        print("error: unexpected failure; this is a bug in compare.py, not a pixel mismatch",
              file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
