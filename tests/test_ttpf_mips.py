"""Tests for tools/ttpf_mips.py: tile modes, mip-tail layout, BC decode, metric, and a
synthetic RDS5/ETPF pool round trip. No game data is used.

    python -I tests/test_ttpf_mips.py
"""
import importlib.util
import io
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
import zlib
from contextlib import ExitStack, closing, redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "ttpf_mips.py")
spec = importlib.util.spec_from_file_location("ttpf_mips", TOOL)
ttpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttpf)


def rgb565(r, g, b):
    return ((r * 31 // 255) << 11) | ((g * 63 // 255) << 5) | (b * 31 // 255)


def solid_bc1(color):
    c = rgb565(*color)
    return struct.pack("<HHI", c, c, 0)


def solid_bc4(v):
    return bytes([v, v, 0, 0, 0, 0, 0, 0])


def solid_bc5(r, g):
    return solid_bc4(r) + solid_bc4(g)


def pack_bc7_block(mode, *, part=0, rot=0, isel=0, r=(), g=(), b=(), a=(),
                   pbits=(), idx1=None, idx2=None):
    """Assemble one 16-byte BC7 block for any valid mode (0..7)."""
    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = ttpf._BC7_MODES[mode]
    blk = 1 << mode
    pos = mode + 1
    if pb:
        blk |= (part & ((1 << pb) - 1)) << pos
        pos += pb
    if rb:
        blk |= (rot & ((1 << rb) - 1)) << pos
        pos += rb
    if isb:
        blk |= (isel & 1) << pos
        pos += isb
    n_ep = 2 * ns
    for vals, bits in ((r, cb), (g, cb), (b, cb)):
        assert len(vals) == n_ep
        mask = (1 << bits) - 1
        for v in vals:
            blk |= (v & mask) << pos
            pos += bits
    if ab:
        assert len(a) == n_ep
        mask = (1 << ab) - 1
        for v in a:
            blk |= (v & mask) << pos
            pos += ab
    if epb:
        assert len(pbits) == n_ep
        for p in pbits:
            blk |= (p & 1) << pos
            pos += 1
    elif spb:
        assert len(pbits) == ns
        for p in pbits:
            blk |= (p & 1) << pos
            pos += 1
    tab = ttpf._bptc.load_bc7()
    if ns == 1:
        sm = None
        anchors = (0,)
    elif ns == 2:
        sm = tab["partition2"][part]
        anchors = (0, tab["anchor2"][part])
    else:
        sm = tab["partition3"][part]
        anchors = (0, tab["anchor3a"][part], tab["anchor3b"][part])
    idx1 = [0] * 16 if idx1 is None else idx1
    for p in range(16):
        sub = 0 if sm is None else sm[p]
        nb = (ib - 1) if p == anchors[sub] else ib
        blk |= (idx1[p] & ((1 << nb) - 1)) << pos
        pos += nb
    if ib2:
        idx2 = [0] * 16 if idx2 is None else idx2
        for p in range(16):
            nb = (ib2 - 1) if p == 0 else ib2
            blk |= (idx2[p] & ((1 << nb) - 1)) << pos
            pos += nb
    assert pos == 128, f"mode {mode} ended at bit {pos}"
    return blk.to_bytes(16, "little")


def solid_bc7(r, g, b):
    """Mode 6 solid BC7 block for an 8-bit RGB colour (exact to +-1 LSB)."""
    return pack_bc7_block(
        6,
        r=(r >> 1, r >> 1),
        g=(g >> 1, g >> 1),
        b=(b >> 1, b >> 1),
        a=(127, 127),
        pbits=(r & 1, r & 1),
    )


def pack_bc6h_block(code, *, part=0, fields=None, indices=None):
    """Assemble one 16-byte BC6H block for any of the 14 valid mode codes."""
    _, _, _, reg = ttpf.bcpack._BC6H_INFO[code]
    plan, hdr = ttpf.bcpack._bc6h_plan(code)
    blk = code
    fields = fields or [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for c, q, bit_idx, pos in plan:
        bit = ((part >> bit_idx) & 1) if c < 0 else ((fields[c][q] >> bit_idx) & 1)
        if bit:
            blk |= 1 << pos
    pos = hdr
    indices = [0] * 16 if indices is None else indices
    if reg == 1:
        for p in range(16):
            nb = 3 if p == 0 else 4
            blk |= (indices[p] & ((1 << nb) - 1)) << pos
            pos += nb
    else:
        anchor1 = ttpf._bptc.load_bc7()["anchor2"][part]
        for p in range(16):
            nb = 2 if (p == 0 or p == anchor1) else 3
            blk |= (indices[p] & ((1 << nb) - 1)) << pos
            pos += nb
    assert pos == 128, f"BC6H mode {code:#x} ended at bit {pos}"
    return blk.to_bytes(16, "little")


def solid_bc6h(r, g, b, *, signed=False):
    """Mode 0x0F (16-bit base, zero delta) solid BC6H block for float16 RGB."""
    fields = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for c, v in enumerate((r, g, b)):
        neg = signed and v < 0
        hv = struct.unpack("<H", struct.pack("<e", abs(v) if signed else max(0.0, v)))[0]
        hv = min(hv, 0x7BFF)
        if signed:
            mag = min(0x7FFF, (hv * 32 + 30) // 31)
            fields[c][0] = (-mag if neg else mag) & 0xFFFF
        else:
            fields[c][0] = min(0xFFFF, (hv * 64 + 30) // 31)
    return pack_bc6h_block(0x0F, fields=fields)


def retile_to(layout, mip, linear, out, base_offset=0):
    pat = (ttpf.PATTERNS_4K if layout["tile"] == 4096 else ttpf.PATTERNS_64K)[
        {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[layout["element_bytes"]]]
    eb = layout["element_bytes"]
    bw, bh = layout["block_w"], layout["block_h"]
    mip_off = base_offset + mip["offset"]
    for y in range(mip["blocks_h"]):
        for x in range(mip["blocks_w"]):
            off = ((y // bh) * (mip["padded_w"] // bw) + x // bw) * layout["tile"]
            for bit, s in enumerate(pat):
                if s > 0:
                    off |= (((x + mip["tile_x"]) >> (s - 1)) & 1) << bit
                elif s < 0:
                    off |= (((y + mip["tile_y"]) >> (-s - 1)) & 1) << bit
            at = (y * mip["blocks_w"] + x) * eb
            out[mip_off + off:mip_off + off + eb] = linear[at:at + eb]


def retile_volume(layout, linear):
    eb = layout["element_bytes"]
    pat = ttpf.VOLUME_PATTERNS_4K[{1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]]
    bw, bh, bd = layout["block_w"], layout["block_h"], layout["block_d"]
    mip = layout["mips"][0]
    w, h, d = mip["blocks_w"], mip["blocks_h"], mip["blocks_d"]
    tiles_x = mip["padded_w"] // bw
    out = bytearray(layout["size"])
    for z in range(d):
        z_base = (z // bd) * layout["z_tile_stride"]
        z_in_tile = 0
        for bit, s in enumerate(pat):
            if s >= 101:
                z_in_tile |= ((z >> (s - 101)) & 1) << bit
        for y in range(h):
            y_tile = (y // bh) * tiles_x
            y_in_tile = z_in_tile
            for bit, s in enumerate(pat):
                if s < 0:
                    y_in_tile |= ((y >> (-s - 1)) & 1) << bit
            for x in range(w):
                off = z_base + (y_tile + x // bw) * layout["tile"] + y_in_tile
                for bit, s in enumerate(pat):
                    if 0 < s < 100:
                        off |= ((x >> (s - 1)) & 1) << bit
                at = ((z * h + y) * w + x) * eb
                out[off:off + eb] = linear[at:at + eb]
    return bytes(out)


def solid_chain(layout, fmt=71, color_fn=None):
    """Build block-linear bytes for every mip; each block is one solid BC colour."""
    eb = layout["element_bytes"]
    out = {}
    is_hdr = fmt in (95, 96)
    # Smooth gradient at the top level; every lower mip is the 2x2 block mean.
    top = layout["mips"][0]
    colours = {}
    for y in range(top["blocks_h"]):
        for x in range(top["blocks_w"]):
            if color_fn is not None:
                colours[(0, x, y)] = color_fn(x, y, top["blocks_w"], top["blocks_h"])
            elif fmt == 95:
                colours[(0, x, y)] = (x * 16.0 / max(1, top["blocks_w"] - 1),
                                      y * 8.0 / max(1, top["blocks_h"] - 1), 1.5)
            elif fmt == 96:
                colours[(0, x, y)] = (-4.0 + x * 8.0 / max(1, top["blocks_w"] - 1),
                                      -2.0 + y * 4.0 / max(1, top["blocks_h"] - 1), 0.5)
            else:
                colours[(0, x, y)] = (min(255, x * 255 // max(1, top["blocks_w"] - 1)),
                                      min(255, y * 255 // max(1, top["blocks_h"] - 1)), 128)
    for m in range(1, len(layout["mips"])):
        pm = layout["mips"][m - 1]
        cm = layout["mips"][m]
        for y in range(cm["blocks_h"]):
            for x in range(cm["blocks_w"]):
                acc = [0.0, 0.0, 0.0] if is_hdr else [0, 0, 0]
                n = 0
                for dy in range(2):
                    for dx in range(2):
                        px, py = min(pm["blocks_w"] - 1, 2 * x + dx), min(pm["blocks_h"] - 1, 2 * y + dy)
                        q = colours[(m - 1, px, py)]
                        for k in range(3):
                            acc[k] += q[k]
                        n += 1
                colours[(m, x, y)] = tuple((a / n) if is_hdr else (a // n) for a in acc)
    for mip in layout["mips"]:
        buf = bytearray(mip["blocks_w"] * mip["blocks_h"] * eb)
        for y in range(mip["blocks_h"]):
            for x in range(mip["blocks_w"]):
                at = (y * mip["blocks_w"] + x) * eb
                c = colours[(mip["index"], x, y)]
                if fmt == 80:
                    blk = solid_bc4(c[0])
                elif fmt == 83:
                    blk = solid_bc5(c[0], c[1])
                elif fmt in (95, 96):
                    blk = solid_bc6h(c[0], c[1], c[2], signed=(fmt == 96))
                elif fmt in (98, 99):
                    blk = solid_bc7(c[0], c[1], c[2])
                else:
                    blk = solid_bc1(c)[:eb].ljust(eb, b"\0")
                buf[at:at + eb] = blk
        out[mip["index"]] = bytes(buf)
    return out


def write_rds5_pool(path, records, chunks, area=None):
    """Write a synthetic RDS5/ETPF pool with one or more zlib-compressed chunks.

    `area` is the size of the record region in bytes. The default, len(records) * 24, fits
    every 8-byte asset key after the 16-byte descriptors. A smaller area (at least
    len(records) * 16) lets the key array continue into the 32-byte ID region, as most
    retail pools do."""
    count = len(chunks) + 2
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    area = len(records) * 24 if area is None else area
    metadata = 16 + area + len(records) * 32
    padded = ttpf.align(metadata, 4096)

    source = etpf + padded
    virt = padded
    entries = []
    packed_chunks = []
    for raw in chunks:
        assert len(raw) % 4096 == 0 and len(raw) > 0
        packed_payload = zlib.compress(raw, 9)
        pad = (-len(packed_payload)) % 256
        entries.append((source // 256) | (8 << 26) | ((virt // 4096) << 30) | (pad << 52))
        packed_chunks.append((source, packed_payload))
        source += len(packed_payload) + pad
        virt += len(raw)

    vsize = virt
    entries.append((source // 256) | ((vsize // 4096) << 30))

    header = bytearray(24)
    header[0:4] = b"RDS5"
    struct.pack_into("<I", header, 4, count)
    struct.pack_into("<Q", header, 8, vsize | (22 << 56))
    struct.pack_into("<I", header, 16, (count + 31) // 32)
    struct.pack_into("<I", header, 20, 0)

    blob = bytearray(source)
    blob[0:24] = header
    for i, e in enumerate(entries):
        struct.pack_into("<Q", blob, 24 + i * 8, e)
    blob[etpf:etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, len(records), area)
    for i, rec in enumerate(records):
        blob[etpf + 16 + i * 16:etpf + 16 + (i + 1) * 16] = rec
        struct.pack_into("<Q", blob, etpf + 16 + len(records) * 16 + i * 8, 0x1234567890ABCDEF + i)
    for src, payload in packed_chunks:
        blob[src:src + len(payload)] = payload
    with open(path, "wb") as f:
        f.write(blob)
    return padded


def pack_etpf_record(width, height, packed_mips, non_tail, fmt, depth, dimension, mode,
                     base, meta=0, extra=0):
    a = (width - 1) | ((height - 1) << 14) | ((packed_mips & 0xF) << 28)
    b = non_tail | (fmt << 3) | ((depth - 1) << 10) | (dimension << 21) | (mode << 23)
    c = (base // 4096) | (meta << 21)
    return struct.pack("<IIII", a, b, c, extra)


def build_pool(path, width=256, height=256, mips=7, non_tail=1, fmt=71, mode=4,
               meta=0x103, extra=0, packed=6):
    layout = ttpf.surface_layout(width, height, fmt, mips, ttpf.tile_bytes(mode))
    linear = solid_chain(layout, fmt)
    payload = bytearray(layout["size"])
    for mip in layout["mips"]:
        retile_to(layout, mip, linear[mip["index"]], payload)
    packed_payload = zlib.compress(bytes(payload), 9)
    pad = (-len(packed_payload)) % 256
    count = 3
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    metadata = 16 + 24 + 32
    padded = ttpf.align(metadata, 4096)
    source = etpf + padded
    chunk_end = source + len(packed_payload) + pad
    virt = padded
    vsize = virt + layout["size"]
    assert vsize % 4096 == 0
    entry = (source // 256) | (8 << 26) | ((virt // 4096) << 30) | (pad << 52)
    sentinel = (chunk_end // 256) | ((vsize // 4096) << 30)
    header = bytearray(24)
    header[0:4] = b"RDS5"
    struct.pack_into("<I", header, 4, count)
    struct.pack_into("<Q", header, 8, vsize | (22 << 56))
    struct.pack_into("<I", header, 16, (count + 31) // 32)
    struct.pack_into("<I", header, 20, 0)
    a = (width - 1) | ((height - 1) << 14) | ((packed & 0xF) << 28)
    b = non_tail | (fmt << 3) | (1 << 21) | (mode << 23)
    c = (virt // 4096) | (meta << 21)
    record = struct.pack("<IIII", a, b, c, extra) + struct.pack("<Q", 0x1234567890ABCDEF)
    blob = bytearray(chunk_end)
    blob[0:24] = header
    struct.pack_into("<QQ", blob, 24, entry, sentinel)
    blob[etpf:etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, 1, 24)
    blob[etpf + 16:etpf + 16 + 24] = record
    blob[source:source + len(packed_payload)] = packed_payload
    with open(path, "wb") as f:
        f.write(blob)
    return layout


class TileModeTest(unittest.TestCase):
    def test_mode_is_flag_plus_tile_mode(self):
        self.assertEqual([ttpf.tile_mode(m) for m in (0, 1, 2, 4, 5, 6, 7, 32, 34, 36, 38)],
                         [0, 0, 1, 2, 2, 3, 3, 0, 1, 2, 3])
        self.assertEqual([ttpf.tile_bytes(m) for m in (0, 1, 2, 4, 32, 34)],
                         [65536, 65536, 4096, 4096, 65536, 4096])


class LayoutTest(unittest.TestCase):
    def test_known_bc1_chain_sizes(self):
        lay = ttpf.surface_layout(256, 256, 71, 7, 4096)
        self.assertEqual(lay["first_tail"], 2)
        self.assertEqual(lay["size"], 45056)
        self.assertEqual([m["region"] for m in lay["mips"]],
                         [32768, 8192, 4096, 4096, 4096, 4096, 4096])
        self.assertEqual([m["offset"] for m in lay["mips"][:2]], [12288, 4096])
        # 16x16 pixels = 4x4 blocks at in-tile origin (16, 0)
        self.assertEqual((lay["mips"][2]["blocks_w"], lay["mips"][2]["blocks_h"]), (16, 16))
        self.assertEqual((lay["mips"][2]["tile_x"], lay["mips"][2]["tile_y"]), (16, 0))

    def test_validated_1024_chain(self):
        lay = ttpf.surface_layout(1024, 1024, 71, 9, 4096)
        self.assertEqual(lay["first_tail"], 4)
        self.assertEqual(lay["size"], 700416)
        self.assertEqual([m["region"] for m in lay["mips"][:4]],
                         [524288, 131072, 32768, 8192])
        self.assertEqual(lay["mips"][4]["tile_x"], 16)
        self.assertEqual(lay["mips"][4]["tile_y"], 0)

    def test_64k_tile_mode(self):
        lay = ttpf.surface_layout(1024, 1024, 98, 9, 65536)
        self.assertEqual(lay["size"], 1441792)
        self.assertEqual(lay["first_tail"], 3)
        self.assertEqual(lay["tile"], 65536)

    def test_bad_mip_count(self):
        with self.assertRaises(ttpf.LayoutError):
            ttpf.surface_layout(64, 64, 71, 20, 4096)


class BcDecodeTest(unittest.TestCase):
    def test_bc1_solid_block(self):
        block = solid_bc1((255, 0, 0))
        px = ttpf.decode_pixels(71, block, 1, 1)
        self.assertEqual(len(px), 16)
        self.assertGreater(px[0][0], 250)
        self.assertLess(px[0][1], 8)

    def test_bc1_four_colour_mode(self):
        # c0=white > c1=black: index 0 -> white, index 3 -> (c0 + 2*c1)/3
        white = ttpf.decode_pixels(71, struct.pack("<HHI", 0xFFFF, 0x0000, 0), 1, 1)
        self.assertGreater(white[0][0], 250)
        dark = ttpf.decode_pixels(71, struct.pack("<HHI", 0xFFFF, 0x0000, 0xFFFFFFFF), 1, 1)
        self.assertLess(dark[0][0], 100)

    def test_bc3_colour_ignores_endpoint_order_but_bc1_does_not(self):
        # Row selectors 0, 1, 2, 3 (0xE4); BC3 = alpha half + the same colour half.
        for c0, c1, bc1_row, bc3_row in (
                (0x0000, 0xFFFF, [0.0, 255.0, 127.5, 0.0], [0.0, 255.0, 85.0, 170.0]),
                (0xFFFF, 0xFFFF, [255.0, 255.0, 255.0, 0.0], [255.0] * 4)):
            colour = struct.pack("<HHI", c0, c1, 0xE4E4E4E4)
            with self.subTest(c0=c0, c1=c1):
                bc1 = ttpf.decode_pixels(71, colour, 1, 1)
                bc3 = ttpf.decode_pixels(77, solid_bc4(255) + colour, 1, 1)
                self.assertEqual([p[0] for p in bc1[:4]], bc1_row)
                self.assertEqual([p[0] for p in bc3[:4]], bc3_row)

    def test_metric_rewards_the_true_downsample(self):
        parent = [(0.0, 0.0, 0.0), (255.0, 0.0, 0.0), (0.0, 255.0, 0.0), (0.0, 0.0, 255.0)]
        child = [(63.75, 63.75, 63.75)]
        self.assertLess(ttpf.mean_abs_error(ttpf.downsample2(parent, 2, 2), child), 1.0)
        wrong = [(0.0, 255.0, 0.0)]
        self.assertGreater(ttpf.mean_abs_error(ttpf.downsample2(parent, 2, 2), wrong), 50)

    def test_bc4_and_bc5_interpolation_modes(self):
        # BC4 8-value mode (e0=210 > e1=70): index 0 -> 210, index 1 -> 70, index 2 -> (6*210+70)//7 = 190
        bits8 = 0 | (1 << 3) | (2 << 6)
        blk8 = bytes([210, 70]) + bits8.to_bytes(6, "little")
        px8 = ttpf.decode_pixels(80, blk8, 1, 1)
        self.assertEqual(px8[0], (210.0, 210.0, 210.0))
        self.assertEqual(px8[1], (70.0, 70.0, 70.0))
        self.assertEqual(px8[2], (190.0, 190.0, 190.0))

        # BC4 6-value mode (e0=50 <= e1=150): index 6 -> 0, index 7 -> 255
        bits6 = 0 | (1 << 3) | (6 << 6) | (7 << 9)
        blk6 = bytes([50, 150]) + bits6.to_bytes(6, "little")
        px6 = ttpf.decode_pixels(80, blk6, 1, 1)
        self.assertEqual(px6[0], (50.0, 50.0, 50.0))
        self.assertEqual(px6[1], (150.0, 150.0, 150.0))
        self.assertEqual(px6[2], (0.0, 0.0, 0.0))
        self.assertEqual(px6[3], (255.0, 255.0, 255.0))

        # BC5 (fmt=83): channel R from first BC4 block, channel G from second, B=0.0
        px5 = ttpf.decode_pixels(83, blk8 + blk6, 1, 1)
        self.assertEqual(px5[0], (210.0, 50.0, 0.0))
        self.assertEqual(px5[1], (70.0, 150.0, 0.0))
        self.assertEqual(px5[2], (190.0, 0.0, 0.0))
        self.assertEqual(px5[3], (210.0, 255.0, 0.0))

    def test_block_means_sub_block_clamping(self):
        # 4x4 block where only the top-left 2x2 texels are active; remaining 12 are zero-padded.
        pixels = []
        for y in range(4):
            for x in range(4):
                pixels.append((200.0, 100.0, 50.0) if (x < 2 and y < 2) else (0.0, 0.0, 0.0))
        clamped = ttpf.block_means(pixels, 1, 1, pixels_w=2, pixels_h=2)
        self.assertEqual(clamped, [(200.0, 100.0, 50.0)])
        unclamped = ttpf.block_means(pixels, 1, 1)
        self.assertEqual(unclamped, [(50.0, 25.0, 12.5)])

    def test_bc7_all_eight_modes_and_reserved(self):
        tab = ttpf._bptc.load_bc7()
        # Reserved mode (b0 == 0): decodes to all-zero black.
        self.assertEqual(ttpf.decode_pixels(98, b"\x00" * 16, 1, 1), [(0.0, 0.0, 0.0)] * 16)
        self.assertEqual(ttpf.decode_pixels(99, b"\x00" * 16, 1, 1), [(0.0, 0.0, 0.0)] * 16)

        # Mode 0: 3 subsets, 4-bit endpoints + per-endpoint p-bit -> 5-bit expanded via MSB rep.
        # Subset 0: (15,0,0) p=1 -> (255,8,8); Subset 1: (0,15,0) p=0 -> (0,247,0); Subset 2: (0,0,15) p=1 -> (8,8,255).
        blk0 = pack_bc7_block(
            0, part=0,
            r=(15, 15, 0, 0, 0, 0), g=(0, 0, 15, 15, 0, 0), b=(0, 0, 0, 0, 15, 15),
            pbits=(1, 1, 0, 0, 1, 1),
        )
        px0 = ttpf.decode_pixels(98, blk0, 1, 1)
        exp0 = [(255.0, 8.0, 8.0), (0.0, 247.0, 0.0), (8.0, 8.0, 255.0)]
        for p, sub in enumerate(tab["partition3"][0]):
            self.assertEqual(px0[p], exp0[sub], f"mode 0 pixel {p}")

        # Mode 1: 2 subsets, 6-bit endpoints + shared per-subset p-bit -> 7-bit expanded.
        # Subset 0: r=0..63, p=1 -> 2..255; Subset 1: g=32..32, p=0 -> 129..129.
        # Index 3 on 3-bit table has weight 27 -> ((37*2 + 27*255 + 32) >> 6) = 109.
        idx1 = [3 if p == 1 else 0 for p in range(16)]
        blk1 = pack_bc7_block(
            1, part=0,
            r=(0, 63, 0, 0), g=(0, 0, 32, 32), b=(0, 0, 0, 0),
            pbits=(1, 0), idx1=idx1,
        )
        px1 = ttpf.decode_pixels(98, blk1, 1, 1)
        self.assertEqual(px1[0], (2.0, 2.0, 2.0))
        self.assertEqual(px1[1], (109.0, 2.0, 2.0))
        self.assertEqual(px1[2], (0.0, 129.0, 0.0))  # partition2[0][2] == 1

        # Mode 2: 3 subsets, 5-bit endpoints (no p-bits), 2-bit indices.
        blk2 = pack_bc7_block(
            2, part=1,
            r=(31, 0, 16, 16, 0, 0), g=(0, 0, 0, 0, 31, 31), b=(0, 31, 0, 0, 0, 0),
            idx1=[3 if p == 1 else 0 for p in range(16)],
        )
        px2 = ttpf.decode_pixels(98, blk2, 1, 1)
        self.assertEqual(px2[0], (255.0, 0.0, 0.0))
        # pixel 1 is subset 0 in partition3[1]; idx=3 -> weight 64 -> (0, 0, 255)
        self.assertEqual(tab["partition3"][1][1], 0)
        self.assertEqual(px2[1], (0.0, 0.0, 255.0))

        # Mode 3: 2 subsets, 7-bit + per-endpoint p-bit (exact 8-bit), 2-bit indices.
        # Subset 0: R=0..255, idx=1 (w=21) -> (21*255 + 32)>>6 = 84.
        blk3 = pack_bc7_block(
            3, part=0,
            r=(0, 127, 50, 50), g=(0, 0, 100, 100), b=(0, 0, 0, 0),
            pbits=(0, 1, 0, 0), idx1=[1 if p == 1 else 0 for p in range(16)],
        )
        px3 = ttpf.decode_pixels(98, blk3, 1, 1)
        self.assertEqual(px3[0], (0.0, 0.0, 0.0))
        self.assertEqual(px3[1], (84.0, 0.0, 0.0))
        self.assertEqual(px3[2], (100.0, 200.0, 0.0))

        # Mode 4: 1 subset, 5-bit color, 6-bit alpha, rotation + index selection (isel=0 and isel=1).
        # Color endpoints: R=0..255, G=0..0, B=0..0. Alpha endpoints: A=0..255.
        # idx1[1]=1 (2-bit, w=21 -> 84), idx2[1]=4 (3-bit, w=37 -> 147).
        for rot, isel, expected in (
            (0, 0, (84.0, 0.0, 0.0)),    # color uses idx1 (84), no swap
            (0, 1, (147.0, 0.0, 0.0)),   # color uses idx2 (147), no swap
            (1, 0, (147.0, 0.0, 0.0)),   # rot=1 swaps R and A -> R gets alpha(idx2=147)
            (2, 0, (84.0, 147.0, 0.0)),  # rot=2 swaps G and A -> G gets alpha(147)
            (3, 1, (147.0, 0.0, 84.0)),  # isel=1: color=idx2(147), alpha=idx1(84), rot=3 puts A into B
        ):
            blk4 = pack_bc7_block(
                4, rot=rot, isel=isel,
                r=(0, 31), g=(0, 0), b=(0, 0), a=(0, 63),
                idx1=[1 if p == 1 else 0 for p in range(16)],
                idx2=[4 if p == 1 else 0 for p in range(16)],
            )
            px4 = ttpf.decode_pixels(98, blk4, 1, 1)
            self.assertEqual(px4[1], expected, f"mode 4 rot={rot} isel={isel}")

        # Mode 5: 1 subset, 7-bit color, 8-bit alpha, rotation, 2-bit color & alpha indices.
        blk5 = pack_bc7_block(
            5, rot=2,
            r=(64, 64), g=(0, 0), b=(16, 16), a=(0, 200),
            idx1=[0] * 16, idx2=[3 if p == 5 else 0 for p in range(16)],
        )
        px5 = ttpf.decode_pixels(98, blk5, 1, 1)
        self.assertEqual(px5[0], (129.0, 0.0, 32.0))
        self.assertEqual(px5[5], (129.0, 200.0, 32.0))  # rot=2 swaps A into G

        # Mode 6: 1 subset, 7-bit RGBA + per-endpoint p-bit (exact 8-bit), 4-bit indices.
        # idx=5 has weight 21 -> ((43*20 + 21*220 + 32) >> 6) = 86.
        blk6 = pack_bc7_block(
            6, r=(10, 110), g=(0, 0), b=(127, 127), a=(127, 127),
            pbits=(0, 0), idx1=[5 if p == 3 else 0 for p in range(16)],
        )
        px6 = ttpf.decode_pixels(99, blk6, 1, 1)
        self.assertEqual(px6[0], (20.0, 0.0, 254.0))
        self.assertEqual(px6[3], (86.0, 0.0, 254.0))

        # Mode 7: 2 subsets, 5-bit RGBA + per-endpoint p-bit -> 6-bit expanded, 2-bit indices.
        blk7 = pack_bc7_block(
            7, part=0,
            r=(31, 31, 0, 0), g=(0, 0, 31, 31), b=(0, 0, 0, 0), a=(31, 31, 31, 31),
            pbits=(1, 1, 1, 1),
        )
        px7 = ttpf.decode_pixels(98, blk7, 1, 1)
        self.assertEqual(px7[0], (255.0, 4.0, 4.0))
        self.assertEqual(px7[2], (4.0, 255.0, 4.0))

    def test_bc6h_modes_uf16_sf16_and_reserved(self):
        lut = ttpf._f16_lut()
        # Reserved modes (0x13, 0x17, 0x1B, 0x1F) decode to zero.
        for bad_mode in (0x13, 0x17, 0x1B, 0x1F):
            blk_bad = bytes([bad_mode]) + b"\x00" * 15
            self.assertEqual(ttpf.decode_pixels(95, blk_bad, 1, 1), [(0.0, 0.0, 0.0)] * 16)
            self.assertEqual(ttpf.decode_pixels(96, blk_bad, 1, 1), [(0.0, 0.0, 0.0)] * 16)

        # Mode 0x03 (1-region untransformed 10-bit): UF16 max endpoint 1023 -> 0xFFFF -> 0x7BF7 -> ~65024.0
        blk03 = pack_bc6h_block(0x03, fields=[[0, 1023, 0, 0], [512, 512, 0, 0], [0, 0, 0, 0]],
                                indices=[0, 15] + [0] * 14)
        px03_u = ttpf.decode_pixels(95, blk03, 1, 1)
        self.assertEqual(px03_u[0][0], 0.0)
        self.assertAlmostEqual(px03_u[1][0], lut[(0xFFFF * 31) >> 6], places=5)
        self.assertGreater(px03_u[0][1], 1.0)
        # Same bit pattern as SF16: 1023 -> -1 (negative!), 512 -> -512 (max negative magnitude -> -65504.0).
        px03_s = ttpf.decode_pixels(96, blk03, 1, 1)
        self.assertEqual(px03_s[0][0], 0.0)
        self.assertLess(px03_s[1][0], 0.0)
        self.assertEqual(px03_s[0][1], -65504.0)

        # Mode 0x0B (1-region transformed 12-bit base + 8-bit delta, reversed rw[11:10]).
        # e0 = 0x900 (2304, uses both bits 11 and 8), delta = -16 -> e1 = 2288.
        blk0b = pack_bc6h_block(0x0B, fields=[[0x900, (-16) & 0xFF, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                indices=[0, 15] + [0] * 14)
        px0b = ttpf.decode_pixels(95, blk0b, 1, 1)
        exp_e0 = ttpf._bc6h_finish(ttpf._bc6h_unquantize(0x900, 12, False), False, lut)
        exp_e1 = ttpf._bc6h_finish(ttpf._bc6h_unquantize(0x8F0, 12, False), False, lut)
        self.assertEqual(px0b[0][0], exp_e0)
        self.assertEqual(px0b[1][0], exp_e1)

        # Mode 0x0F (1-region transformed 16-bit base + 4-bit delta, reversed rw[15:10]).
        # Test exact round-trip of large positive UF16 and negative SF16 via solid_bc6h,
        # plus -32768 clamp in SF16 -> finite -65504.0 (never -inf/NaN).
        px_hi = ttpf.decode_pixels(95, solid_bc6h(128.0, 0.5, 0.0, signed=False), 1, 1)
        self.assertAlmostEqual(px_hi[0][0], 128.0, places=2)
        self.assertAlmostEqual(px_hi[0][1], 0.5, places=3)
        px_neg = ttpf.decode_pixels(96, solid_bc6h(-12.5, 3.25, -0.25, signed=True), 1, 1)
        self.assertAlmostEqual(px_neg[0][0], -12.5, places=2)
        self.assertAlmostEqual(px_neg[0][1], 3.25, places=2)
        self.assertAlmostEqual(px_neg[0][2], -0.25, places=3)
        blk_min16 = pack_bc6h_block(0x0F, fields=[[0x8000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        px_min16 = ttpf.decode_pixels(96, blk_min16, 1, 1)
        self.assertEqual(px_min16[0][0], -65504.0)

        # Mode 0x02 (2-region transformed 11-bit base, deltas (5,4,4)) with partition 1.
        # Subset 0: R=1000, Subset 1: R=1000+10=1010..1000-6=994.
        blk02 = pack_bc6h_block(
            0x02, part=1,
            fields=[[1000, 0, 10, (-6) & 0x1F], [0, 0, 0, 0], [0, 0, 0, 0]],
            indices=[0] * 16,
        )
        px02 = ttpf.decode_pixels(95, blk02, 1, 1)
        sm1 = ttpf._bptc.load_bc7()["partition2"][1]
        exp_s0 = ttpf._bc6h_finish(ttpf._bc6h_unquantize(1000, 11, False), False, lut)
        exp_s1 = ttpf._bc6h_finish(ttpf._bc6h_unquantize(1010, 11, False), False, lut)
        for p, sub in enumerate(sm1):
            self.assertEqual(px02[p][0], exp_s0 if sub == 0 else exp_s1, f"mode 0x02 pixel {p}")

        # Mode 0x1E (2-region untransformed 6-bit) in SF16: negative subset 1 endpoint.
        blk1e = pack_bc6h_block(
            0x1E, part=0,
            fields=[[16, 16, (-16) & 0x3F, (-16) & 0x3F], [0, 0, 0, 0], [0, 0, 0, 0]],
            indices=[0] * 16,
        )
        px1e = ttpf.decode_pixels(96, blk1e, 1, 1)
        self.assertGreater(px1e[0][0], 0.0)
        self.assertEqual(px1e[2][0], -px1e[0][0])

    def test_hdr_tonemap_metric(self):
        same = [(0.0, 1.0, 500.0), (1000.0, 0.25, 65504.0)]
        self.assertEqual(ttpf.mean_abs_error(same, same, hdr=True), 0.0)
        # Bounded even on extreme HDR highlights.
        hi_err = ttpf.mean_abs_error([(0.0, 0.0, 0.0)], [(65504.0, 65504.0, 65504.0)], hdr=True)
        self.assertGreater(hi_err, 250.0)
        self.assertLess(hi_err, 255.0)
        # Signed sensitivity for SF16: -1.0 -> -127.5, +1.0 -> +127.5 -> diff = 255.0.
        sign_err = ttpf.mean_abs_error([(-1.0, -1.0, -1.0)], [(1.0, 1.0, 1.0)], hdr=True)
        self.assertAlmostEqual(sign_err, 255.0, places=5)


class Bc2AndBc3CheckTest(unittest.TestCase):
    def run_check(self, fmt, lower_colour):
        import io
        from contextlib import redirect_stdout
        # Black then white, selector 3: the high-resolution image is uniform gray 170.
        top_colour = struct.pack("<HHI", 0, 0xFFFF, 0xFFFFFFFF)
        layout = ttpf.surface_layout(256, 256, fmt, 7, 4096)
        payload = bytearray(layout["size"])
        for mip in layout["mips"]:
            colour = top_colour if mip["index"] == 0 else lower_colour
            alpha = b"\xff" * 8 if fmt in (74, 75) else solid_bc4(255)
            block = alpha + colour
            retile_to(layout, mip, block * (mip["blocks_w"] * mip["blocks_h"]), payload)
        record = pack_etpf_record(256, 256, 6, 1, fmt, 1, 1, 4, 4096, 0x103)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rgb-mips.ttpf")
            write_rds5_pool(path, [record], [payload])
            pool = ttpf.Pool(path)
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    code = ttpf.check(pool, 0)
            finally:
                pool.close()
        return code, json.loads(buf.getvalue())

    def test_bc2_decodes_all_four_rgb_selectors(self) -> None:
        # Explicit alpha bytes precede the color block. Black comes before
        # white, but BC2 still uses both interpolated colors.
        block = struct.pack("<QHHI", 0xFEDCBA9876543210, 0, 0xFFFF, 0xE4E4E4E4)
        expected = [(value, value, value) for value in (0.0, 255.0, 85.0, 170.0)] * 4
        for fmt in (74, 75):
            with self.subTest(fmt=fmt):
                self.assertEqual(ttpf.decode_pixels(fmt, block, 1, 1), expected)

    def test_gray_chain_passes_whatever_the_endpoint_order(self):
        # Reversed endpoints with selector 2 describe the same gray as the top level.
        gray = struct.pack("<HHI", 0xFFFF, 0, 0xAAAAAAAA)
        for fmt in (74, 75, 77, 78):
            with self.subTest(fmt=fmt):
                code, result = self.run_check(fmt, gray)
                self.assertEqual(code, 0, result)
                self.assertEqual(result["chain_error"], [0.0] * 6)

    def test_black_child_under_gray_top_fails(self):
        black = struct.pack("<HHI", 0, 0, 0)
        for fmt in (74, 75, 77, 78):
            with self.subTest(fmt=fmt):
                code, result = self.run_check(fmt, black)
                self.assertEqual(code, 1, result)
                self.assertEqual(result["chain_error"], [170.0, 0.0, 0.0, 0.0, 0.0, 0.0])


BC1_BLACK = bytes(8)
BC1_WHITE = struct.pack("<HHI", 0xFFFF, 0xFFFF, 0)
# Four-colour mode (white > black): selector 1 = black, selector 0 = white.
BC1_LEFT_BLACK = struct.pack("<HHI", 0xFFFF, 0, 0x05050505)  # columns 0-1 black, 2-3 white
BC1_LEFT_WHITE = struct.pack("<HHI", 0xFFFF, 0, 0x50505050)  # columns 0-1 white, 2-3 black


def bc1_black(x, y):
    return BC1_BLACK


def bc1_column(cx, block=BC1_WHITE):
    return lambda x, y: block if x == cx else BC1_BLACK


def bc1_single(cx, cy):
    return lambda x, y: BC1_WHITE if (x, y) == (cx, cy) else BC1_BLACK


def bc1_blocks(mip, block_fn):
    return b"".join(block_fn(x, y) for y in range(mip["blocks_h"]) for x in range(mip["blocks_w"]))


def odd_mip(index, pw, ph):
    return dict(index=index, pixels_w=pw, pixels_h=ph, blocks_w=(pw + 3) // 4, blocks_h=(ph + 3) // 4)


def pair_error(parent, parent_fn, child, child_fn):
    ppx = ttpf.decode_pixels(71, bc1_blocks(parent, parent_fn), parent["blocks_w"], parent["blocks_h"])
    cpx = ttpf.decode_pixels(71, bc1_blocks(child, child_fn), child["blocks_w"], child["blocks_h"])
    means = ttpf.block_means(cpx, child["blocks_w"], child["blocks_h"],
                             child["pixels_w"], child["pixels_h"])
    return ttpf.mean_abs_error(ttpf.predict_child_means(ppx, parent, child),
                               ttpf.visible_block_means(means, child))


class PredictChildMeansTest(unittest.TestCase):
    def test_odd_block_counts_match_box_filter_oracles(self):
        cases = (
            ((12, 8), bc1_black, bc1_single(1, 0), 127.5),
            ((8, 12), bc1_black, bc1_single(0, 1), 127.5),
            ((12, 12), bc1_black, bc1_single(1, 1), 63.75),
            ((12, 8), bc1_column(2), bc1_single(1, 0), 0.0),
            ((12, 8), bc1_column(2), bc1_black, 127.5),
            # 5x3 -> 3x2 blocks: a flat zip of 2x1 predictions would misalign rows.
            ((20, 12), bc1_column(4), bc1_column(2), 0.0),
            ((20, 12), bc1_column(4), bc1_black, 85.0),
            # Child texels 6-7 are padding: only the visible half of the edge block counts.
            ((12, 8), bc1_black, bc1_column(1, BC1_LEFT_BLACK), 0.0),
            ((12, 8), bc1_black, bc1_column(1, BC1_LEFT_WHITE), 127.5),
            # Parent texels 10-11 are padding and must not leak into child texel 4.
            ((10, 8), bc1_column(2, BC1_LEFT_BLACK), bc1_black, 0.0),
            ((10, 8), bc1_column(2, BC1_LEFT_WHITE), bc1_black, 127.5),
        )
        for (w, h), parent_fn, child_fn, expected in cases:
            with self.subTest(size=(w, h), expected=expected):
                self.assertAlmostEqual(pair_error(odd_mip(0, w, h), parent_fn,
                                                  odd_mip(1, w // 2, h // 2), child_fn),
                                       expected, places=9)

    def test_matches_block_downsample_on_power_of_two_chains(self):
        rng = random.Random(1)
        for w, h in ((16, 8), (8, 8), (4, 4), (2, 2), (1, 1)):
            with self.subTest(size=(w, h)):
                parent = odd_mip(0, w, h)
                child = odd_mip(1, max(1, w // 2), max(1, h // 2))
                bw, bh = parent["blocks_w"], parent["blocks_h"]
                data = bytes(rng.randrange(256) for _ in range(8 * bw * bh))
                px = ttpf.decode_pixels(71, data, bw, bh)
                old = ttpf.downsample2(ttpf.block_means(px, bw, bh, w, h), bw, bh)
                new = ttpf.predict_child_means(px, parent, child)
                self.assertEqual(len(new), len(old))
                for a, b in zip(new, old):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=9)

    def test_matches_nested_list_box_filter_on_odd_sizes(self):
        rng = random.Random(2)
        for w, h in ((28, 20), (22, 14), (6, 1), (1, 6)):
            with self.subTest(size=(w, h)):
                parent = odd_mip(0, w, h)
                child = odd_mip(1, max(1, w // 2), max(1, h // 2))
                bw, bh = parent["blocks_w"], parent["blocks_h"]
                px = ttpf.decode_pixels(71, bytes(rng.randrange(256) for _ in range(8 * bw * bh)), bw, bh)
                grid = [[px[y * bw * 4 + x] for x in range(w)] for y in range(h)]
                cw, ch = child["pixels_w"], child["pixels_h"]
                texels = [[tuple(sum(grid[min(h - 1, 2 * y + dy)][min(w - 1, 2 * x + dx)][k]
                                     for dy in (0, 1) for dx in (0, 1)) / 4 for k in range(3))
                           for x in range(cw)] for y in range(ch)]
                expected = []
                for by in range(0, ch, 4):
                    for bx in range(0, cw, 4):
                        cell = [t for row in texels[by:by + 4] for t in row[bx:bx + 4]]
                        expected.append(tuple(sum(t[k] for t in cell) / len(cell) for k in range(3)))
                got = ttpf.predict_child_means(px, parent, child)
                self.assertEqual(len(got), child["blocks_w"] * child["blocks_h"])
                for a, b in zip(got, expected):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=9)

    def test_mismatched_geometry_is_rejected(self):
        parent = odd_mip(0, 12, 8)
        px = ttpf.decode_pixels(71, BC1_BLACK * 6, 3, 2)
        for child in (odd_mip(1, 5, 4), odd_mip(1, 6, 3), dict(odd_mip(1, 6, 4), blocks_w=1)):
            with self.subTest(child=child), self.assertRaises(ttpf.MipGeometryError):
                ttpf.predict_child_means(px, parent, child)
        with self.assertRaises(ttpf.MipGeometryError):
            ttpf.predict_child_means(px[:-1], parent, odd_mip(1, 6, 4))
        with self.assertRaises(ttpf.MipGeometryError):
            ttpf.visible_block_means([(0.0, 0.0, 0.0)], odd_mip(1, 6, 4))
        with self.assertRaises(ValueError):
            ttpf.mean_abs_error([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)] * 2)


def linear_layout(sizes, first_tail=0, depth=1):
    """Fake layout storing each mip block-linear, mip-major across `depth` slices."""
    mips, off = [], 0
    for i, (w, h) in enumerate(sizes):
        m = odd_mip(i, w, h)
        m.update(offset=off, region=8 * m["blocks_w"] * m["blocks_h"], tile_x=0, tile_y=0)
        m["slice_stride"] = m["region"]
        mips.append(m)
        off += m["region"] * depth
    return dict(mips=mips, size=off, slice_size=off // depth, face_size=off, tile=4096,
                first_tail=first_tail, element_bytes=8)


def run_shape_check(shape, sizes, chunks, first_tail=0, depth=1):
    layout = linear_layout(sizes, first_tail, depth)
    if shape == "cubemap":
        layout.update(data_offset=0, full_bytes=6 * layout["size"], snapshots=1)
    w, h = sizes[0]
    d = dict(base=0, format=71, width=w, height=h, depth=depth, mips=len(sizes), non_tail=0)
    a = dict(layout=layout, descriptor=d, available=True, full_chain=True)
    if shape != "2d":
        a["shape"] = shape
    tiled = b"".join(chunks) + bytes(1 << 17)
    buf = io.StringIO()
    with mock.patch.object(ttpf, "analyze", lambda pool, index: a), \
            mock.patch.object(ttpf, "_read_range_or_block", lambda *args, **kw: tiled), \
            mock.patch.object(ttpf, "detile", lambda lay, mip, data: bytes(
                data[:8 * mip["blocks_w"] * mip["blocks_h"]])), \
            redirect_stdout(buf):
        code = ttpf.check(None, 0)
    return code, json.loads(buf.getvalue())


class OddBlockShapeCheckTest(unittest.TestCase):
    """check() on 12x8 -> 6x4 BC1 (3x2 parent blocks) for each 2D-family shape. The container
    and tiling are faked (block-linear mips); OddBlockPoolTest covers the real path."""
    SIZES = ((12, 8), (6, 4))

    def chunk(self, level, block_fn):
        return bc1_blocks(odd_mip(level, *self.SIZES[level]), block_fn)

    def test_2d_edge_column_and_tail_origin(self):
        parent = self.chunk(0, bc1_column(2))
        code, result = run_shape_check("2d", self.SIZES, [parent, self.chunk(1, bc1_column(1))],
                                       first_tail=1)
        self.assertEqual(code, 0, result)
        self.assertEqual(result["chain_error"], [0.0])
        # The fake tile-origin read sees parent blocks 0-1 (black) where the white edge belongs.
        self.assertEqual(result["tail_origin_control"], 127.5)
        code, result = run_shape_check("2d", self.SIZES, [parent, self.chunk(1, bc1_black)],
                                       first_tail=1)
        self.assertEqual(code, 1, result)
        self.assertEqual(result["chain_error"], [127.5])

    def test_array_reports_the_slice_with_a_dropped_edge(self):
        parents = [self.chunk(0, bc1_column(2))] * 3
        children = [self.chunk(1, bc1_column(1)), self.chunk(1, bc1_black),
                    self.chunk(1, bc1_column(1))]
        code, result = run_shape_check("2d_array", self.SIZES, parents + children, depth=3)
        self.assertEqual(code, 1, result)
        self.assertEqual(result["per_slice_max_error"], [0.0, 127.5, 0.0])
        self.assertEqual(result["max_error"], 127.5)

    def test_cube_reports_the_face_with_a_dropped_edge_and_tail_origin(self):
        chunks = []
        for face in range(6):
            chunks += [self.chunk(0, bc1_column(2)),
                       self.chunk(1, bc1_black if face == 4 else bc1_column(1))]
        code, result = run_shape_check("cubemap", self.SIZES, chunks, first_tail=1)
        self.assertEqual(code, 1, result)
        self.assertEqual(result["per_face_max_error"], [0.0, 0.0, 0.0, 0.0, 127.5, 0.0])
        self.assertEqual(result["tail_origin_control"], 127.5)

    def test_mismatched_mip_geometry_is_blocked(self):
        sizes = ((12, 8), (5, 4))
        chunks = [bc1_blocks(odd_mip(0, 12, 8), bc1_black), bc1_blocks(odd_mip(1, 5, 4), bc1_black)]
        code, result = run_shape_check("2d", sizes, chunks)
        self.assertEqual(code, 2, result)
        self.assertIn("geometry", result["blocked"])


def odd_bc1_pool(path, width, height, parent_fn, child_fn):
    """Write a two-mip BC1 pool through the real layout and tiling."""
    layout = ttpf.surface_layout(width, height, 71, 2, 4096)
    payload = bytearray(layout["size"])
    for mip, block_fn in zip(layout["mips"], (parent_fn, child_fn)):
        retile_to(layout, mip, bc1_blocks(mip, block_fn), payload)
    non_tail = layout["first_tail"]
    record = pack_etpf_record(width, height, 2 - non_tail, non_tail, 71, 1, 1, 2, 4096)
    write_rds5_pool(path, [record], [payload])


class OddBlockPoolTest(unittest.TestCase):
    """Before the fix the changed 12x8, 8x12 and 12x12 children passed with error 0."""
    CASES = (
        # (width, height, parent, child, rc, chain_error)
        (12, 8, bc1_black, bc1_black, 0, [0.0]),
        (12, 8, bc1_black, bc1_single(1, 0), 1, [127.5]),
        (8, 12, bc1_black, bc1_black, 0, [0.0]),
        (8, 12, bc1_black, bc1_single(0, 1), 1, [127.5]),
        (12, 12, bc1_black, bc1_black, 0, [0.0]),
        (12, 12, bc1_black, bc1_single(1, 1), 1, [63.75]),
        (12, 8, bc1_column(2), bc1_column(1), 0, [0.0]),
        (16, 8, bc1_black, bc1_black, 0, [0.0]),
        (16, 8, bc1_black, bc1_single(1, 0), 1, [127.5]),
        (256, 256, bc1_black, bc1_black, 0, [0.0]),
        (256, 256, bc1_black, bc1_single(31, 31), 0, [0.25]),
    )

    def run_check(self, *case):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "odd.ttpf")
            odd_bc1_pool(path, *case)
            pool = ttpf.Pool(path)
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    code = ttpf.check(pool, 0)
            finally:
                pool.close()
        return code, json.loads(buf.getvalue())

    def test_odd_block_counts_cover_every_child_block(self):
        for w, h, parent_fn, child_fn, rc, errs in self.CASES:
            with self.subTest(size=(w, h), rc=rc, chain_error=errs):
                code, result = self.run_check(w, h, parent_fn, child_fn)
                self.assertEqual(code, rc, result)
                self.assertEqual(result["chain_error"], errs)

    def test_cli_rejects_a_dropped_edge_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "odd.ttpf")
            odd_bc1_pool(path, 12, 8, bc1_black, bc1_single(1, 0))
            proc = subprocess.run([sys.executable, "-I", TOOL, "check", path, "0"],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["max_error"], 127.5)
        self.assertTrue(result["full_chain"])


class VisibleExportTest(unittest.TestCase):
    """Before the fix 17x13 BC1 exported its 2x2-block child as 3x2 storage blocks (208 bytes)."""
    RED = (255, 0, 0)

    def test_export_packs_visible_blocks_only(self):
        def child(x, y):
            return solid_bc1(self.RED) if y == 1 else bc1_black(x, y)
        for w, h in ((17, 13), (17, 16), (16, 13), (16, 16)):
            with self.subTest(size=(w, h)), tempfile.TemporaryDirectory() as tmp:
                path, out = os.path.join(tmp, "edge.ttpf"), os.path.join(tmp, "edge.dds")
                odd_bc1_pool(path, w, h, bc1_black, child)
                pool = ttpf.Pool(path)
                try:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(ttpf.export(pool, 0, out), 0)
                finally:
                    pool.close()
                with open(out, "rb") as f:
                    blob = f.read()
                pw, ph, cw, ch = (w + 3) // 4, (h + 3) // 4, (w // 2 + 3) // 4, (h // 2 + 3) // 4
                self.assertEqual(len(blob) - 148, 8 * (pw * ph + cw * ch))
                self.assertEqual(struct.unpack_from("<I", blob, 20)[0], 8 * pw * ph)
                at = 148 + 8 * (pw * ph + cw)
                px = ttpf.decode_pixels(71, blob[at:at + 8], 1, 1)
                self.assertEqual(tuple(px[0][:3]), self.RED)


class BptcCheckTest(unittest.TestCase):
    def _run_pool_check(self, fmt, *, corrupt_mip1=False):
        layout = ttpf.surface_layout(256, 256, fmt, 7, 4096)
        linear = solid_chain(layout, fmt=fmt)
        if corrupt_mip1:
            m1 = layout["mips"][1]
            bad = solid_bc6h(0.0, 0.0, 0.0, signed=(fmt == 96)) if fmt in (95, 96) else solid_bc7(0, 0, 0)
            linear[1] = bad * (m1["blocks_w"] * m1["blocks_h"])
        payload = bytearray(layout["size"])
        for mip in layout["mips"]:
            retile_to(layout, mip, linear[mip["index"]], payload)
        record = pack_etpf_record(256, 256, 6, 1, fmt, 1, 1, 4, 4096, 0x103)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"bptc_{fmt}.ttpf")
            write_rds5_pool(path, [record], [payload])
            pool = ttpf.Pool(path)
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    code = ttpf.check(pool, 0)
            finally:
                pool.close()
        return code, json.loads(buf.getvalue())

    def test_bc7_unorm_and_srgb_chain_check(self):
        for fmt in (98, 99):
            with self.subTest(fmt=fmt):
                code, res = self._run_pool_check(fmt)
                self.assertEqual(code, 0, res)
                self.assertLess(res["max_error"], 2.0)
                self.assertGreater(res["tail_origin_control"], 5 * max(res["max_error"], 1.0))

                bad_code, bad_res = self._run_pool_check(fmt, corrupt_mip1=True)
                self.assertEqual(bad_code, 1, bad_res)
                self.assertGreater(bad_res["chain_error"][0], 50.0)

    def test_bc6h_uf16_and_sf16_chain_check(self):
        for fmt in (95, 96):
            with self.subTest(fmt=fmt):
                code, res = self._run_pool_check(fmt)
                self.assertEqual(code, 0, res)
                self.assertLess(res["max_error"], 2.0)
                self.assertGreater(res["tail_origin_control"], 5 * max(res["max_error"], 1.0))

                bad_code, bad_res = self._run_pool_check(fmt, corrupt_mip1=True)
                self.assertEqual(bad_code, 1, bad_res)
                self.assertGreater(bad_res["chain_error"][0], 50.0)


class SyntheticPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "synthetic.ttpf")
        self.layout = build_pool(self.path)
        self.pool = ttpf.Pool(self.path)
        self.addCleanup(self.pool.close)

    def test_descriptor_and_layout(self):
        d = self.pool.descriptor(0)
        self.assertEqual((d["width"], d["height"], d["format"], d["mips"], d["non_tail"]),
                         (256, 256, 71, 7, 1))
        self.assertEqual(d["mode"], 4)
        a = ttpf.analyze(self.pool, 0)
        self.assertTrue(a["span_matches"])
        self.assertTrue(a["full_chain"])
        self.assertEqual(a["layout"]["first_tail"], 2)

    def test_detile_round_trip(self):
        a = ttpf.analyze(self.pool, 0)
        layout = a["layout"]
        tiled = self.pool.read_range(a["descriptor"]["base"], layout["size"])
        linear = solid_chain(layout)
        for mip in layout["mips"]:
            data = ttpf.detile(layout, mip, tiled[mip["offset"]:mip["offset"] + mip["region"]])
            self.assertEqual(data, linear[mip["index"]])

    def test_chain_metric(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ttpf.check(self.pool, 0)
        result = json.loads(buf.getvalue())
        self.assertEqual(code, 0, result)
        self.assertLess(result["max_error"], 4.0)
        # a wrong in-tile origin for the first tail mip must be much worse
        self.assertGreater(result["tail_origin_control"], 5 * result["max_error"])

    def test_export_full_chain(self):
        out = os.path.join(self.tmp.name, "chain.dds")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.export(self.pool, 0, out), 0)
        with open(out, "rb") as f:
            data = f.read()
        self.assertEqual(data[:4], b"DDS ")
        mips = struct.unpack_from("<I", data, 28)[0]
        self.assertEqual(mips, 7)
        expected = 148 + sum(m["blocks_w"] * m["blocks_h"] * 8 for m in self.layout["mips"])
        self.assertEqual(len(data), expected)
        # payload is exactly the concatenated block-linear chain
        linear = solid_chain(self.layout)
        self.assertEqual(data[148:], b"".join(linear[m["index"]] for m in self.layout["mips"]))

    def test_export_requested_mip_prefix_with_unavailable_lower_mips(self) -> None:
        layout = self.layout
        linear = solid_chain(layout)
        payload = bytearray(layout["size"])
        for mip in layout["mips"]:
            retile_to(layout, mip, linear[mip["index"]], payload)
        top_start = layout["mips"][0]["offset"]
        path = os.path.join(self.tmp.name, "partially-readable.ttpf")
        record = pack_etpf_record(256, 256, 6, 1, 71, 1, 1, 4, 4096, meta=0x103)
        write_rds5_pool(path, [record], [payload[:top_start], payload[top_start:]])
        # Give the lower mip chunk an unsupported codec. The second chunk is still valid zlib.
        with open(path, "r+b") as f:
            f.seek(24)
            entry, = struct.unpack("<Q", f.read(8))
            f.seek(24)
            f.write(struct.pack("<Q", (entry & ~(15 << 26)) | (15 << 26)))

        out = os.path.join(self.tmp.name, "top.dds")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.main(["export", path, "0", out, "--mips", "1"]), 0)
        with open(out, "rb") as f:
            data = f.read()
        self.assertEqual(struct.unpack_from("<I", data, 28), (1,))
        self.assertTrue(data[148:] == linear[0], "DDS payload differs from the source top mip")

        full = os.path.join(self.tmp.name, "full.dds")
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(ttpf.main(["export", path, "0", full]), 2)
        self.assertIn("not every requested mip uses a supported codec", err.getvalue())
        self.assertFalse(os.path.exists(full))

    def test_export_write_error_names_the_output_without_a_traceback(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        out = os.path.join(self.tmp.name, "destination")
        os.mkdir(out)
        error = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(error):
            self.assertEqual(ttpf.export(self.pool, 0, out, force=True), 1)
        self.assertIn(out, error.getvalue())
        self.assertIn("cannot write", error.getvalue())

    def test_scan_reports_the_whitelist_gate(self):
        # The synthetic tuple is validated, so it is not gated: the partial export is
        # the descriptor non_tail limit, not the 319-byte profile table.
        out = os.path.join(self.tmp.name, "scan.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([self.path], out)
        with open(out, encoding="utf-8") as f:
            result = json.load(f)["total"]
        self.assertEqual(result["entries"], 1)
        self.assertEqual(result["whitelist_gated"], 0)
        self.assertEqual(result["reader_partial"], 1)
        self.assertEqual(result["compact_partial_rows"], 1)
        self.assertEqual(result["compact_missing_mips"], 6)
        self.assertEqual(result["compact_decodable_mips"], 6)
        self.assertEqual(result["compact_full_rows"], 1)
        self.assertEqual(result["gated_accounted"], 0)
        self.assertEqual(result["gated_tuple_distinct"], 0)
        self.assertEqual((result["after_full"], result["after_partial"],
                          result["after_skip"]), (1, 0, 0))

    def test_scan_finds_pools_below_any_folder_and_rejects_paths_without_one(self):
        """A pool folder other than the game root (data/ttdb itself, one pack) is searched recursively; a
        path with no pool is an error, not an all-zero scan with exit code 0."""
        import io
        from contextlib import redirect_stderr, redirect_stdout
        ttdb = os.path.join(self.tmp.name, "game [v1]", "data", "ttdb", "props")   # [..] is not a glob here
        os.makedirs(ttdb)
        build_pool(os.path.join(ttdb, "a.ttpf"))
        out = os.path.join(self.tmp.name, "scan.json")
        for folder in (os.path.join(self.tmp.name, "game [v1]"), os.path.dirname(ttdb), ttdb):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ttpf.main(["scan", folder, "--json", out]), 0, folder)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["total"]["pools"], 1, folder)
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        for bad, why in ((os.path.join(self.tmp.name, "missing"), "no such file or folder"),
                         (empty, "no .ttpf pool in it")):
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(ttpf.main(["scan", self.path, bad]), 2, bad)
            self.assertIn(why, err.getvalue())
            self.assertIn(bad, err.getvalue())

    def test_scan_skips_directories_named_like_pools(self):
        root = os.path.join(self.tmp.name, "game")
        standard = os.path.join(root, "data", "ttdb")
        os.makedirs(os.path.join(standard, "backup.ttpf"))
        os.makedirs(os.path.join(root, "pack", "patch", "data", "ttdb", "backup.ttpf"))

        fallback = os.path.join(root, "extra", "backup.ttpf")
        os.makedirs(fallback)
        fallback_pool = os.path.join(fallback, "actual.ttpf")
        build_pool(fallback_pool)
        self.assertEqual(ttpf.find_pools(root), [fallback_pool])

        standard_pool = os.path.join(standard, "actual.ttpf")
        build_pool(standard_pool)
        self.assertEqual(ttpf.find_pools(root), [standard_pool])
        out = os.path.join(self.tmp.name, "scan.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.main(["scan", root, "--json", out]), 0)
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["total"]["pools"], 1)

    def test_scan_counts_an_unlisted_tuple_as_gated(self):
        path = os.path.join(self.tmp.name, "unlisted.ttpf")
        build_pool(path, meta=0x599)
        out = os.path.join(self.tmp.name, "scan2.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], out)
        with open(out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["whitelist_gated"], 1)
        self.assertEqual(total["gated_layout_ok"], 1)
        self.assertEqual(total["gated_full_chain"], 1)
        self.assertEqual(total["gated_full_mips"], 7)
        self.assertEqual(total["gated_partial"], 0)
        self.assertEqual(total["gated_no_contiguous"], 0)
        self.assertEqual(total["gated_no_layout"], 0)
        self.assertEqual(total["gated_unknown_format"], 0)
        self.assertEqual(total["gated_other_layout_error"], 0)
        self.assertEqual(total["gated_span_mismatch"], 0)
        self.assertEqual(total["gated_unreadable_range"], 0)
        self.assertEqual(total["gated_accounted"], total["whitelist_gated"])
        self.assertEqual(total["gated_tuple_distinct"], 1)
        self.assertEqual((total["after_full"], total["after_partial"],
                          total["after_skip"]), (1, 0, 0))

    def test_scan_reports_the_rejected_tuple(self):
        path = os.path.join(self.tmp.name, "tuple.ttpf")
        build_pool(path, meta=0x599)
        out = os.path.join(self.tmp.name, "scan3.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], out, include_tuples=True)
        with open(out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["gated_tuples"],
                         [{"tuple": [4, 256, 256, 7, 1, 8], "count": 1}])

    def test_scan_buckets_an_unknown_element_size(self):
        path = os.path.join(self.tmp.name, "unknown-format.ttpf")
        build_pool(path, meta=0x599)
        # Rewrite the 7-bit format field to DXGI 120 (unmapped in ELEMENT_BYTES).
        # The scan rejects the record before reading texture data.
        etpf = ttpf.align(24 + 2 * 8, 256)
        at = etpf + 16 + 4
        with open(path, "r+b") as f:
            f.seek(at)
            b = struct.unpack("<I", f.read(4))[0]
            f.seek(at)
            f.write(struct.pack("<I", (b & ~(0x7F << 3)) | (120 << 3)))
        out = os.path.join(self.tmp.name, "scan4.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], out)
        with open(out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["unsupported_format"], 1)
        self.assertEqual(total["format11"], 0)
        self.assertEqual(total["whitelist_gated"], 1)
        self.assertEqual(total["gated_layout_ok"], 0)
        self.assertEqual(total["gated_no_layout"], 1)
        self.assertEqual(total["gated_unknown_format"], 1)
        self.assertEqual(total["gated_other_layout_error"], 0)
        self.assertEqual(total["gated_accounted"], 1)
        self.assertEqual(total["gated_full_chain"], 0)
        self.assertEqual(total["after_full"], 0)
        self.assertEqual(total["after_skip"], 1)

    def test_legacy_format_156_parses_as_format_28_depth_2(self):
        # Raw 156 << 3 in the old 8-bit format view sets bit 10 (depth bit 0), so
        # the fixed 7-bit format / 11-bit depth parser reads format=28, depth=2 (2D array).
        path = os.path.join(self.tmp.name, "legacy156.ttpf")
        build_pool(path, meta=0x599)
        etpf = ttpf.align(24 + 2 * 8, 256)
        at = etpf + 16 + 4
        with open(path, "r+b") as f:
            f.seek(at)
            b = struct.unpack("<I", f.read(4))[0]
            f.seek(at)
            f.write(struct.pack("<I", (b & ~(0xFF << 3)) | (156 << 3)))
        pool = ttpf.Pool(path)
        self.addCleanup(pool.close)
        d = pool.descriptor(0)
        self.assertEqual((d["format"], d["depth"], d["dimension"]), (28, 2, 1))
        a = ttpf.analyze(pool, 0)
        self.assertEqual(a["shape"], "2d_array")


def build_bc1_bcpack_pool(path, width=64, height=64, mips=5, non_tail=1, mode=4,
                          meta=0x103, extra=0, packed=4, ref_dists=None):
    """Build a synthetic RDS5 pool whose chunk is encoded with BCPack codec 1 (BC1)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_bcpack as tb
    layout = ttpf.surface_layout(width, height, 71, mips, ttpf.tile_bytes(mode))
    linear = solid_chain(layout, 71)
    payload = bytearray(layout["size"])
    for mip in layout["mips"]:
        retile_to(layout, mip, linear[mip["index"]], payload)
    packed_payload = tb.encode_bc1_blocks(bytes(payload), ref_dists=ref_dists)
    pad = (-len(packed_payload)) % 256
    count = 3
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    metadata = 16 + 24 + 32
    padded = ttpf.align(metadata, 4096)
    source = etpf + padded
    chunk_end = source + len(packed_payload) + pad
    virt = padded
    vsize = virt + layout["size"]
    entry = (source // 256) | (1 << 26) | ((virt // 4096) << 30) | (pad << 52)
    sentinel = (chunk_end // 256) | ((vsize // 4096) << 30)
    header = bytearray(24)
    header[0:4] = b"RDS5"
    struct.pack_into("<I", header, 4, count)
    struct.pack_into("<Q", header, 8, vsize | (22 << 56))
    struct.pack_into("<I", header, 16, (count + 31) // 32)
    struct.pack_into("<I", header, 20, 0)
    a = (width - 1) | ((height - 1) << 14) | ((packed & 0xF) << 28)
    b = non_tail | (71 << 3) | (1 << 21) | (mode << 23)
    c = (virt // 4096) | (meta << 21)
    record = struct.pack("<IIII", a, b, c, extra) + struct.pack("<Q", 0xCAFEBABE12345678)
    blob = bytearray(chunk_end)
    blob[0:24] = header
    struct.pack_into("<QQ", blob, 24, entry, sentinel)
    blob[etpf:etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, 1, 24)
    blob[etpf + 16:etpf + 16 + 24] = record
    blob[source:source + len(packed_payload)] = packed_payload
    with open(path, "wb") as f:
        f.write(blob)
    return layout, linear, bytes(payload)


class AssetKeyStraddleTest(unittest.TestCase):
    """The asset-key array follows the 16-byte records and, in most retail pools, is longer
    than the record area has room for, so it continues into the 32-byte ID region."""

    RECORDS = 3

    def write(self, tmp, name, area=None):
        path = os.path.join(tmp, name)
        records = [pack_etpf_record(4, 4, 1, 0, 71, 1, 1, 4, 4096)] * self.RECORDS
        write_rds5_pool(path, records, [bytes(4096)], area=area)
        return path

    def keys(self, path):
        with closing(ttpf.Pool(path)) as pool:
            return [pool.descriptor(i)["asset_key_at"] for i in range(self.RECORDS)]

    def test_keys_past_the_record_area_come_from_the_id_region(self) -> None:
        # Room for exactly one key after the descriptors: the other two sit in the ID region.
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "straddle.ttpf", area=self.RECORDS * 16 + 8)
            self.assertEqual(self.keys(path), [0x1234567890ABCDEF + i for i in range(self.RECORDS)])

    def test_keys_inside_the_record_area_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "plain.ttpf")
            self.assertEqual(self.keys(path), [0x1234567890ABCDEF + i for i in range(self.RECORDS)])

    def test_a_key_split_by_the_records_ids_boundary_is_refused(self) -> None:
        # A record area that cuts the first key in half, as TtpfReader refuses it.
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "split.ttpf", area=self.RECORDS * 16 + 4)
            with closing(ttpf.Pool(path)) as pool:
                with self.assertRaises(ValueError):
                    pool.descriptor(0)


class PoolCacheTest(unittest.TestCase):
    def test_zlib_chunk_rejects_a_decoded_size_larger_than_its_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "pool.ttpf")
            record = pack_etpf_record(4, 4, 1, 0, 71, 1, 1, 4, 4096)
            write_rds5_pool(path, [record], [bytes(8192)])
            with open(path, "r+b") as stream:
                stream.seek(8)
                stream.write(struct.pack("<Q", 8192 | (22 << 56)))
                stream.seek(32)
                sentinel = struct.unpack("<Q", stream.read(8))[0]
                stream.seek(32)
                stream.write(struct.pack("<Q", (sentinel & 0x3FFFFFF) | (2 << 30)))

            with closing(ttpf.Pool(path)) as pool:
                cache = {}
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        pool.read_range(4096, 16, cache)
                    self.assertEqual(cache, {})

    def test_shared_cache_keeps_matching_chunk_offsets_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            record = pack_etpf_record(4, 4, 1, 0, 71, 1, 1, 4, 4096)
            pools = []
            payloads = (bytes((0x11,)) * 4096, bytes((0x22,)) * 4096)
            for i, payload in enumerate(payloads):
                path = os.path.join(temp, f"pool_{i}.ttpf")
                write_rds5_pool(path, [record], [payload])
                pools.append(stack.enter_context(closing(ttpf.Pool(path))))
            self.assertEqual(pools[0].chunks, pools[1].chunks)
            cache = {}
            for pool, payload in zip(pools, payloads):
                self.assertEqual(pool.read_range(4096, 4096, cache), payload)
            self.assertEqual(pools[0].read_range(4096, 4096, cache), payloads[0])


class SyntheticBcpackPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_bc4_and_bc5_decode_pixels(self):
        bc4_blk = bytes([200, 100]) + (0).to_bytes(6, "little")
        px4 = ttpf.decode_pixels(80, bc4_blk, 1, 1)
        self.assertEqual(len(px4), 16)
        self.assertEqual(px4[0], (200.0, 200.0, 200.0))
        bc5_blk = bc4_blk + bytes([50, 25]) + (0).to_bytes(6, "little")
        px5 = ttpf.decode_pixels(83, bc5_blk, 1, 1)
        self.assertEqual(len(px5), 16)
        self.assertEqual(px5[0], (200.0, 50.0, 0.0))

    def test_bc1_bcpack_pool_check_export_and_scan(self):
        import io
        from contextlib import redirect_stdout
        path = os.path.join(self.tmp.name, "bc1_bcpack.ttpf")
        layout, linear, expected_tiled = build_bc1_bcpack_pool(path)
        pool = ttpf.Pool(path)
        self.addCleanup(pool.close)

        a = ttpf.analyze(pool, 0)
        self.assertTrue(a["full_chain"])
        self.assertEqual(a["mip_codecs"][0], [1])
        with self.assertRaises(ttpf.BcpackRequired):
            pool.read_range(a["descriptor"]["base"], layout["size"], bcpack_mode="off")
        self.assertEqual(pool.read_range(a["descriptor"]["base"], layout["size"]), expected_tiled)

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ttpf.check(pool, 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(code, 0, res)
        self.assertLess(res["max_error"], 4.0)

        out_dds = os.path.join(self.tmp.name, "bc1_bcpack.dds")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.export(pool, 0, out_dds), 0)
        with open(out_dds, "rb") as f:
            dds = f.read()
        self.assertEqual(dds[148:], b"".join(linear[m["index"]] for m in layout["mips"]))

        scan_out = os.path.join(self.tmp.name, "scan_bcpack.json")
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], scan_out)
        with open(scan_out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual((total["after_full"], total["after_skip"]), (0, 1))
        self.assertEqual((total["after_bcpack_full"], total["after_bcpack_skip"],
                          total["bcpack_unlocked_full"]), (1, 0, 1))

    def test_provisional_history_strict_refusal_and_opt_in(self):
        import io
        from contextlib import redirect_stdout
        path = os.path.join(self.tmp.name, "bc1_prov.ttpf")
        layout = ttpf.surface_layout(64, 64, 71, 5, ttpf.tile_bytes(4))
        nblocks = layout["size"] // 8
        # Block 0 references provisional entry 6 (256 + 0 - 250 = 6).
        build_bc1_bcpack_pool(path, ref_dists=[250] + [0] * (nblocks - 1))
        pool = ttpf.Pool(path)
        self.addCleanup(pool.close)

        buf = io.StringIO()
        with redirect_stdout(buf):
            code_strict = ttpf.check(pool, 0)
        self.assertEqual(code_strict, 2)
        self.assertIn("provisional BCPack history required", json.loads(buf.getvalue())["blocked"])

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            code_prov = ttpf.check(pool, 0, allow_provisional=True)
        self.assertEqual(code_prov, 0, buf2.getvalue())

    def test_bc7_bcpack_pool_export_and_scan(self):
        import io
        from contextlib import redirect_stdout
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import test_bcpack as tb

        # 64x64 DXGI 98 (BC7_UNORM, 16-byte blocks), 1 mip, 4 KiB tiles, codec 7.
        layout = ttpf.surface_layout(64, 64, 98, 1, ttpf.tile_bytes(4))
        mip = layout["mips"][0]
        nblocks = mip["blocks_w"] * mip["blocks_h"]
        # Solid mode-6 BC7 blocks with distinct per-block endpoints.
        linear = bytearray()
        for i in range(nblocks):
            r0 = (i * 5) & 0x7F
            g0 = (i * 3 + 17) & 0x7F
            b0 = (i * 7 + 31) & 0x7F
            acc = (1 << 6) | (r0 << 7) | (r0 << 14) | (g0 << 21) | (g0 << 28) | (b0 << 35) | (b0 << 42)
            acc |= (0x7F << 49) | (0x7F << 56) | (1 << 63) | (1 << 64)
            linear += acc.to_bytes(16, "little")
        payload = bytearray(layout["size"])
        retile_to(layout, mip, bytes(linear), payload)
        # Fill any unused tail padding blocks in the tile with mode-6 blocks too.
        mode6_pad = (1 << 6).to_bytes(16, "little")
        for off in range(0, len(payload), 16):
            if payload[off:off + 16] == b"\x00" * 16:
                payload[off:off + 16] = mode6_pad
        packed_payload = tb.encode_bc7_mode6_blocks(bytes(payload))

        path = os.path.join(self.tmp.name, "bc7_bcpack.ttpf")
        pad = (-len(packed_payload)) % 256
        count = 3
        table_end = 24 + (count - 1) * 8
        etpf = ttpf.align(table_end, 256)
        padded = ttpf.align(16 + 24 + 32, 4096)
        source = etpf + padded
        chunk_end = source + len(packed_payload) + pad
        virt = padded
        vsize = virt + layout["size"]
        entry = (source // 256) | (7 << 26) | ((virt // 4096) << 30) | (pad << 52)
        sentinel = (chunk_end // 256) | ((vsize // 4096) << 30)
        header = bytearray(24)
        header[0:4] = b"RDS5"
        struct.pack_into("<I", header, 4, count)
        struct.pack_into("<Q", header, 8, vsize | (22 << 56))
        struct.pack_into("<I", header, 16, (count + 31) // 32)
        a = (64 - 1) | ((64 - 1) << 14) | (0 << 28)
        b = 1 | (98 << 3) | (1 << 21) | (4 << 23)
        c = (virt // 4096) | (0x103 << 21)
        record = struct.pack("<IIII", a, b, c, 0) + struct.pack("<Q", 0xCAFEBABE87654321)
        blob = bytearray(chunk_end)
        blob[0:24] = header
        struct.pack_into("<QQ", blob, 24, entry, sentinel)
        blob[etpf:etpf + 4] = b"ETPF"
        struct.pack_into("<III", blob, etpf + 4, 634, 1, 24)
        blob[etpf + 16:etpf + 16 + 24] = record
        blob[source:source + len(packed_payload)] = packed_payload
        with open(path, "wb") as f:
            f.write(blob)

        pool = ttpf.Pool(path)
        self.addCleanup(pool.close)
        a_info = ttpf.analyze(pool, 0)
        self.assertTrue(a_info["full_chain"])
        self.assertEqual(a_info["mip_codecs"][0], [7])

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(ttpf.check(pool, 0), 0)
        chk = json.loads(buf.getvalue())
        self.assertEqual(chk["format"], 98)
        self.assertEqual(chk["chain_error"], [])

        out_dds = os.path.join(self.tmp.name, "bc7_bcpack.dds")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.export(pool, 0, out_dds), 0)
        with open(out_dds, "rb") as f:
            dds = f.read()
        self.assertEqual(dds[:4], b"DDS ")
        self.assertEqual(struct.unpack_from("<I", dds, 128)[0], 98)
        self.assertEqual(dds[148:], bytes(linear))

        scan_out = os.path.join(self.tmp.name, "scan_bc7.json")
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], scan_out)
        with open(scan_out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual((total["after_bcpack_full"], total["after_bcpack_skip"],
                          total["bcpack_unlocked_full"]), (1, 0, 1))

    def test_bc7_bcpack_external_history_cli(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import test_bcpack as tb

        layout = ttpf.surface_layout(64, 64, 98, 1, ttpf.tile_bytes(4))
        mip = layout["mips"][0]
        nblocks = mip["blocks_w"] * mip["blocks_h"]
        linear = bytearray()
        for i in range(nblocks):
            r0 = (i * 5 + 11) & 0x7F
            g0 = (i * 3 + 23) & 0x7F
            b0 = (i * 7 + 41) & 0x7F
            acc = (1 << 6) | (r0 << 7) | (r0 << 14) | (g0 << 21) | (g0 << 28) | (b0 << 35) | (b0 << 42)
            acc |= (0x7F << 49) | (0x7F << 56) | (1 << 63) | (1 << 64)
            linear += acc.to_bytes(16, "little")
        payload = bytearray(layout["size"])
        retile_to(layout, mip, bytes(linear), payload)
        mode6_pad = (1 << 6).to_bytes(16, "little")
        for off in range(0, len(payload), 16):
            if payload[off:off + 16] == b"\x00" * 16:
                payload[off:off + 16] = mode6_pad

        hist_k = 19
        ep_row = [0x45, 0x22 << 1, 0x33 << 1, 0x7F << 1, 0x45, 0x22 << 1, 0x33 << 1, 0x7F << 1] * 3
        packed_payload = tb.encode_bc7_mode6_blocks(bytes(payload), first_ep_history=(hist_k, ep_row))

        path = os.path.join(self.tmp.name, "bc7_hist.ttpf")
        pad = (-len(packed_payload)) % 256
        count = 3
        etpf = ttpf.align(24 + (count - 1) * 8, 256)
        padded = ttpf.align(16 + 24 + 32, 4096)
        source = etpf + padded
        chunk_end = source + len(packed_payload) + pad
        virt = padded
        vsize = virt + layout["size"]
        entry = (source // 256) | (7 << 26) | ((virt // 4096) << 30) | (pad << 52)
        sentinel = (chunk_end // 256) | ((vsize // 4096) << 30)
        header = bytearray(24)
        header[0:4] = b"RDS5"
        struct.pack_into("<I", header, 4, count)
        struct.pack_into("<Q", header, 8, vsize | (22 << 56))
        struct.pack_into("<I", header, 16, (count + 31) // 32)
        a = (64 - 1) | ((64 - 1) << 14) | (0 << 28)
        b = 1 | (98 << 3) | (1 << 21) | (4 << 23)
        c = (virt // 4096) | (0x103 << 21)
        record = struct.pack("<IIII", a, b, c, 0) + struct.pack("<Q", 0xCAFEBABE11223344)
        blob = bytearray(chunk_end)
        blob[0:24] = header
        struct.pack_into("<QQ", blob, 24, entry, sentinel)
        blob[etpf:etpf + 4] = b"ETPF"
        struct.pack_into("<III", blob, etpf + 4, 634, 1, 24)
        blob[etpf + 16:etpf + 16 + 24] = record
        blob[source:source + len(packed_payload)] = packed_payload
        with open(path, "wb") as f:
            f.write(blob)

        hist_json = os.path.join(self.tmp.name, "bc7_history.json")
        with open(hist_json, "w", encoding="utf-8") as f:
            json.dump({
                "bc7": {
                    "historyEndpoints": {str(hist_k): ep_row},
                    "historyEndpointBits": {str(hist_k): [8] * 24},
                    "historyEndpointConfirmedBits": {str(hist_k): [8] * 24},
                }
            }, f)

        out_dds = os.path.join(self.tmp.name, "bc7_hist.dds")
        # Without --history: cleanly refuses with exit 2 and --history hint.
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc_no_hist = ttpf.main(["export", path, "0", out_dds])
        self.assertEqual(rc_no_hist, 2)
        self.assertIn("--history", err.getvalue())
        self.assertFalse(os.path.exists(out_dds))

        # With --history: strict export succeeds and matches exact linear blocks.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc_ok = ttpf.main(["export", path, "0", out_dds, "--history", hist_json])
        self.assertEqual(rc_ok, 0)
        with open(out_dds, "rb") as f:
            dds = f.read()
        self.assertEqual(dds[148:], bytes(linear))

        # Missing --history path: clean exit 2, no traceback.
        err_missing = io.StringIO()
        missing = os.path.join(self.tmp.name, "nope.json")
        with redirect_stdout(io.StringIO()), redirect_stderr(err_missing):
            rc_bad = ttpf.main(["export", path, "0", out_dds, "--history", missing])
        self.assertEqual(rc_bad, 2)
        self.assertNotIn("Traceback", err_missing.getvalue())


class BadPoolTest(unittest.TestCase):
    """A truncated or unsupported pool, or an index past the pool, is a one-line error, not a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.good = os.path.join(self.tmp.name, "good.ttpf")
        build_pool(self.good)
        self.bad = os.path.join(self.tmp.name, "bad.ttpf")
        with open(self.good, "rb") as f:
            data = f.read()
        with open(self.bad, "wb") as f:
            f.write(data[:300])         # RDS5 header intact, chunk data cut off

    def run_main(self, *argv):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = ttpf.main(list(argv))
        return code, err.getvalue()

    def test_failed_open_releases_the_file(self):
        with self.assertRaises(ValueError):
            ttpf.Pool(self.bad)
        os.remove(self.bad)             # on Windows this failed while the pool was left open and mapped

    def test_layout_check_export_name_the_bad_pool(self):
        out = os.path.join(self.tmp.name, "x.dds")
        for argv in (["layout", self.bad, "0"], ["check", self.bad, "0"], ["export", self.bad, "0", out]):
            code, err = self.run_main(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(self.bad, err)
            self.assertNotIn("Traceback", err)
        self.assertFalse(os.path.exists(out))

    def test_scan_names_the_bad_pool(self):
        code, err = self.run_main("scan", self.good, self.bad)
        self.assertEqual(code, 2)
        self.assertIn(self.bad, err)

    def test_scan_json_over_an_input_pool_is_refused(self):
        with open(self.good, "rb") as f:
            before = f.read()
        alias = os.path.join(self.tmp.name, ".", "good.ttpf")
        code, err = self.run_main("scan", self.good, "--json", alias)
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the input", err)
        with open(self.good, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_scan_json_over_a_pool_found_in_a_game_folder_is_refused(self):
        game = os.path.join(self.tmp.name, "game")
        os.makedirs(os.path.join(game, "data", "ttdb"))
        pool = os.path.join(game, "data", "ttdb", "found.ttpf")
        build_pool(pool)
        with open(pool, "rb") as f:
            before = f.read()
        code, err = self.run_main("scan", game, "--json", pool)
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the input", err)
        with open(pool, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_scan_json_that_is_a_directory_is_refused(self):
        code, err = self.run_main("scan", self.good, "--json", self.tmp.name)
        self.assertEqual(code, 2)
        self.assertIn("is a directory", err)

    def test_scan_json_in_a_missing_folder_is_created(self):
        out = os.path.join(self.tmp.name, "reports", "ttpf", "scan.json")
        code, err = self.run_main("scan", self.good, "--json", out)
        self.assertEqual(code, 0, err)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["total"]["entries"], 1)

    def test_scan_json_that_cannot_be_written_keeps_the_printed_report(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        blocker = os.path.join(self.tmp.name, "blocker")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("a file, not a folder")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ttpf.main(["scan", self.good, "--json", os.path.join(blocker, "scan.json")])
        self.assertEqual(code, 2)
        self.assertIn("cannot write", err.getvalue())
        self.assertEqual(json.loads(out.getvalue())["total"]["entries"], 1)

    def test_index_past_the_pool(self):
        for index in ("1", "-1"):
            code, err = self.run_main("layout", self.good, index)
            self.assertEqual(code, 2)
            self.assertIn("out of range", err)

    def test_export_force_protects_input_files(self):
        metadata = os.path.join(self.tmp.name, "metadata.json")
        with open(metadata, "wb") as f:
            f.write(b"{}")
        cases = [
            (self.good, [], "input pool"),
            (metadata, ["--calibration", metadata], "--calibration profile"),
            (metadata, ["--history", metadata], "--history file"),
        ]
        for source, options, message in cases:
            with self.subTest(options=options):
                with open(source, "rb") as f:
                    before = f.read()
                code, err = self.run_main("export", self.good, "0", source, "--force", *options)
                self.assertEqual(code, 2)
                self.assertIn(message, err)
                with open(source, "rb") as f:
                    self.assertEqual(f.read(), before)


class SyntheticArrayPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _build_array_pool(self, path, width=256, height=256, depth=3, fmt=83, mips=6,
                          mode=1, truncate_pages=0):
        eb = ttpf.element_bytes(fmt)
        blk = 4 if fmt in ttpf.BC_FORMATS else 1
        layout = ttpf.array_surface_layout(width, height, depth, mips, eb, blk,
                                           ttpf.tile_bytes(mode))
        payload = bytearray(layout["size"])
        linear_slices = []
        # Distinct smooth gradient per slice so slice-major control strongly disagrees.
        for s in range(depth):
            def color_fn(x, y, bw, bh, sl=s):
                base_r = (sl * 90 + 20) % 256
                base_g = (220 - sl * 80) % 256
                r = min(255, max(0, (base_r + x * 120 // max(1, bw - 1)) % 256))
                g = min(255, max(0, (base_g + y * 120 // max(1, bh - 1)) % 256))
                return (r, g, 128)
            lin = solid_chain(layout, fmt=fmt, color_fn=color_fn)
            linear_slices.append(lin)
            for mip in layout["mips"]:
                retile_to(layout, mip, lin[mip["index"]], payload,
                          base_offset=s * mip["slice_stride"])
        chunk = bytes(payload[:-truncate_pages * 4096] if truncate_pages else payload)
        base = ttpf.align(16 + 24 + 32, 4096)
        rec = pack_etpf_record(width, height, mips, 0, fmt, depth, 1, mode, base)
        write_rds5_pool(path, [rec], [chunk])
        return layout, linear_slices

    def test_2d_array_round_trip_check_export_and_scan(self):
        import io
        from contextlib import redirect_stdout
        path = os.path.join(self.tmp.name, "array_bc5.ttpf")
        layout, linear_slices = self._build_array_pool(path)
        pool = ttpf.Pool(path)
        self.addCleanup(pool.close)

        a = ttpf.analyze(pool, 0)
        self.assertEqual(a["shape"], "2d_array")
        self.assertTrue(a["span_matches"])
        self.assertTrue(a["full_chain"])
        self.assertEqual(a["layout"]["depth"], 3)

        desc = ttpf.describe(pool, 0)
        self.assertEqual(desc["shape"], "2d_array")
        self.assertEqual(desc["depth"], 3)
        self.assertEqual(desc["mips_detail"][0]["slice_stride"], layout["mips"][0]["region"])

        ok_meta, pages = ttpf.validate_chunk_metadata(pool, a["descriptor"], a["allocation_span"])
        self.assertTrue(ok_meta)
        self.assertEqual(pages, ())

        # Byte-exact mip-major detile round-trip across all slices and mips.
        tiled = pool.read_range(a["descriptor"]["base"], layout["size"])
        for s in range(3):
            for mip in layout["mips"]:
                off = mip["offset"] + s * mip["slice_stride"]
                got = ttpf.detile(layout, mip, tiled[off:off + mip["region"]])
                self.assertEqual(got, linear_slices[s][mip["index"]])

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ttpf.check(pool, 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(rc, 0, res)
        self.assertLessEqual(res["max_error"], 6.0)
        self.assertGreater(res["slice_major_control_max_error"], 5 * res["max_error"])
        self.assertGreater(res["slice_major_control_max_error"], 10.0)

        out_dds = os.path.join(self.tmp.name, "array_bc5.dds")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.export(pool, 0, out_dds), 0)
        with open(out_dds, "rb") as f:
            dds = f.read()
        self.assertEqual(dds[:4], b"DDS ")
        dxgi_fmt, dim, misc, array_size, _ = struct.unpack_from("<IIIII", dds, 128)
        self.assertEqual((dxgi_fmt, dim, misc, array_size), (83, 3, 0, 3))
        expected_payload = b"".join(
            linear_slices[s][m["index"]]
            for s in range(3)
            for m in layout["mips"]
        )
        self.assertEqual(dds[148:], expected_payload)

        scan_out = os.path.join(self.tmp.name, "scan_array.json")
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], scan_out)
        with open(scan_out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual((total["arrays"], total["arrays_2d"], total["arrays_2d_full"]), (1, 1, 1))
        self.assertEqual((total["after_full"], total["after_skip"]), (1, 0))

    def test_2d_array_negative_controls(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # Unsupported array mode (mode=4 instead of mode=1) rejected.
        bad_mode = os.path.join(self.tmp.name, "array_bad_mode.ttpf")
        self._build_array_pool(bad_mode, mode=4)
        pool_m = ttpf.Pool(bad_mode)
        self.addCleanup(pool_m.close)
        am = ttpf.analyze(pool_m, 0)
        self.assertIn("unsupported 2D array mode", am["blocked"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.check(pool_m, 0), 2)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(ttpf.export(pool_m, 0, os.path.join(self.tmp.name, "bad1.dds")), 2)

        # Truncated span rejected by analyze, export, and scan_pools.
        trunc = os.path.join(self.tmp.name, "array_trunc.ttpf")
        self._build_array_pool(trunc, truncate_pages=1)
        pool_t = ttpf.Pool(trunc)
        self.addCleanup(pool_t.close)
        at = ttpf.analyze(pool_t, 0)
        self.assertFalse(at["span_matches"])
        self.assertNotIn("available", at)
        self.assertIn("allocation span", at["blocked"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(ttpf.export(pool_t, 0, os.path.join(self.tmp.name, "bad2.dds")), 2)
        scan_out = os.path.join(self.tmp.name, "scan_trunc.json")
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([trunc], scan_out)
        with open(scan_out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["multi_slice_layout_mismatch"], 1)
        self.assertEqual((total["after_full"], total["after_skip"]), (0, 1))


class SyntheticCubemapPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _build_cubemap_faces(self, layout, fmt=71):
        tiled_full = bytearray(layout["full_bytes"])
        linear_faces = []
        for face in range(6):
            def color_fn(x, y, bw, bh, f=face):
                r = min(255, (f * 40 + x * 180 // max(1, bw - 1)) % 256)
                g = min(255, ((5 - f) * 40 + y * 180 // max(1, bh - 1)) % 256)
                b = 32 + f * 36
                return (r, g, b)
            lin = solid_chain(layout, fmt=fmt, color_fn=color_fn)
            linear_faces.append(lin)
            for mip in layout["mips"]:
                retile_to(layout, mip, lin[mip["index"]], tiled_full,
                          base_offset=face * layout["face_size"])
        return bytes(tiled_full), linear_faces

    def _build_cubemap_pool(self, path, non_tail=0, mode=None, depth=6,
                            meta=0, extra=0, chunks_override=None):
        width, height, fmt, mips = 128, 128, 71, 6
        if mode is None:
            mode = 18 + 2 * non_tail
        if non_tail == 0:
            chunk_pages = None
        else:
            chunk_pages = (6, 18)
        layout = ttpf.cubemap_surface_layout(width, height, mips, 8, 4, non_tail, chunk_pages)
        tiled_full, linear_faces = self._build_cubemap_faces(layout, fmt=fmt)
        if chunks_override is not None:
            chunks = chunks_override
        elif non_tail == 0:
            chunks = [tiled_full]
        else:
            # Chunk 0: 6-page dummy low-res snapshot; Chunk 1: 18-page full-res 6-face cubemap.
            chunks = [b"\x5A" * (6 * 4096), tiled_full]
            if meta == 0 and extra == 0:
                # 5-bit codes: slot 0 = 6 (6 pages), slot 1 = 21 (18 pages) -> 6 | (21 << 5) = 0x2A6
                meta = 6 | (21 << 5)
        base = ttpf.align(16 + 24 + 32, 4096)
        packed_mips = mips - non_tail
        rec = pack_etpf_record(width, height, packed_mips, non_tail, fmt, depth, 2, mode,
                               base, meta=meta, extra=extra)
        write_rds5_pool(path, [rec], chunks)
        return layout, linear_faces

    def test_cubemap_mode18_and_streaming_mode20(self):
        import io
        from contextlib import redirect_stdout

        for non_tail, mode, expected_offset, expected_pages in [
            (0, 18, 0, ()),
            (1, 20, 6 * 4096, (6, 18)),
        ]:
            with self.subTest(non_tail=non_tail, mode=mode):
                path = os.path.join(self.tmp.name, f"cube_mode{mode}.ttpf")
                layout, linear_faces = self._build_cubemap_pool(path, non_tail=non_tail, mode=mode)
                pool = ttpf.Pool(path)
                self.addCleanup(pool.close)

                a = ttpf.analyze(pool, 0)
                self.assertEqual(a["shape"], "cubemap")
                self.assertTrue(a["span_matches"])
                self.assertTrue(a["full_chain"])
                self.assertEqual(a["layout"]["data_offset"], expected_offset)

                ok_meta, pages = ttpf.validate_chunk_metadata(pool, a["descriptor"], a["allocation_span"])
                self.assertTrue(ok_meta)
                self.assertEqual(pages, expected_pages)

                desc = ttpf.describe(pool, 0)
                self.assertEqual(desc["shape"], "cubemap")
                self.assertEqual(desc["data_offset"], expected_offset)
                self.assertEqual(desc["snapshots"], non_tail + 1)

                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = ttpf.check(pool, 0)
                res = json.loads(buf.getvalue())
                self.assertEqual(rc, 0, res)
                self.assertEqual(len(res["per_face_max_error"]), 6)
                self.assertLessEqual(res["max_error"], 6.0)
                self.assertGreater(res["tail_origin_control"], 5 * res["max_error"])

                out_dds = os.path.join(self.tmp.name, f"cube_mode{mode}.dds")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(ttpf.export(pool, 0, out_dds), 0)
                with open(out_dds, "rb") as f:
                    dds = f.read()
                self.assertEqual(dds[:4], b"DDS ")
                caps2 = struct.unpack_from("<I", dds, 112)[0]
                self.assertEqual(caps2 & 0xFE00, 0xFE00)
                dxgi_fmt, dim, misc, array_size, _ = struct.unpack_from("<IIIII", dds, 128)
                self.assertEqual((dxgi_fmt, dim, misc, array_size), (71, 3, 4, 1))
                expected_payload = b"".join(
                    linear_faces[face][m["index"]]
                    for face in range(6)
                    for m in layout["mips"]
                )
                self.assertEqual(dds[148:], expected_payload)

                scan_out = os.path.join(self.tmp.name, f"scan_cube{mode}.json")
                with redirect_stdout(io.StringIO()):
                    ttpf.scan_pools([path], scan_out)
                with open(scan_out, encoding="utf-8") as f:
                    total = json.load(f)["total"]
                self.assertEqual((total["cubemaps"], total["cubemaps_full"]), (1, 1))
                self.assertEqual((total["chunk_metadata_ok"], total["chunk_metadata_fail"]), (1, 0))
                self.assertEqual((total["after_full"], total["after_skip"]), (1, 0))

    def test_cubemap_negative_controls(self):
        import io
        from contextlib import redirect_stdout

        # Wrong chunk metadata (code 4 = 4 pages instead of 6 for chunk 0).
        bad_meta = os.path.join(self.tmp.name, "cube_bad_meta.ttpf")
        self._build_cubemap_pool(bad_meta, non_tail=1, meta=(4 | (21 << 5)))
        pool_bm = ttpf.Pool(bad_meta)
        self.addCleanup(pool_bm.close)
        abm = ttpf.analyze(pool_bm, 0)
        self.assertIn("chunk metadata mismatch", abm["blocked"])
        ok_meta, _ = ttpf.validate_chunk_metadata(pool_bm, abm["descriptor"], abm["allocation_span"])
        self.assertFalse(ok_meta)

        # depth != 6 rejected.
        bad_depth = os.path.join(self.tmp.name, "cube_bad_depth.ttpf")
        self._build_cubemap_pool(bad_depth, non_tail=0, depth=4)
        pool_bd = ttpf.Pool(bad_depth)
        self.addCleanup(pool_bd.close)
        abd = ttpf.analyze(pool_bd, 0)
        self.assertIn("unsupported cubemap depth", abd["blocked"])

        # Invalid mode (mode=22 with non_tail=0) rejected.
        bad_mode = os.path.join(self.tmp.name, "cube_bad_mode.ttpf")
        self._build_cubemap_pool(bad_mode, non_tail=0, mode=22)
        pool_bmod = ttpf.Pool(bad_mode)
        self.addCleanup(pool_bmod.close)
        abmod = ttpf.analyze(pool_bmod, 0)
        self.assertIn("unsupported cubemap mode", abmod["blocked"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.check(pool_bmod, 0), 2)

    def test_cubemap_first_tail_zero_and_single_mip_check(self):
        import io
        from contextlib import redirect_stdout

        base = ttpf.align(16 + 24 + 32, 4096)
        # 32x32 BC1 4-mip and 1-mip cubemaps: every mip fits in one 4 KiB tail tile (first_tail == 0).
        for mips, name in ((4, "cube_tail0.ttpf"), (1, "cube_1mip.ttpf")):
            w = h = 32
            layout = ttpf.cubemap_surface_layout(w, h, mips, 8, 4, 0)
            self.assertEqual(layout["first_tail"], 0 if mips > 1 else 1)
            buf = bytearray(layout["full_bytes"])
            for face in range(6):
                color = (face * 35, 60 + face * 25, 20 + face * 30)
                for m in layout["mips"]:
                    lin = solid_bc1(color) * (m["blocks_w"] * m["blocks_h"])
                    retile_to(layout, m, lin, buf, base_offset=face * layout["face_size"])
            path = os.path.join(self.tmp.name, name)
            rec = pack_etpf_record(w, h, mips, 0, 71, 6, 2, 18, base, meta=0, extra=0)
            write_rds5_pool(path, [rec], [bytes(buf)])
            pool = ttpf.Pool(path)
            self.addCleanup(pool.close)
            out = io.StringIO()
            with redirect_stdout(out):
                rc = ttpf.check(pool, 0)
            self.assertEqual(rc, 0)
            res = json.loads(out.getvalue())
            self.assertEqual(res["first_tail"], 0 if mips > 1 else 1)
            self.assertIsNone(res["tail_origin_control"])
            if mips == 1:
                self.assertIsNone(res["max_error"])
            else:
                self.assertLessEqual(res["max_error"], 1.0)

        # Mid-chunk base mismatch rejected by validate_chunk_metadata.
        misaligned = os.path.join(self.tmp.name, "cube_mid_chunk.ttpf")
        rec = pack_etpf_record(32, 32, 1, 0, 71, 6, 2, 18, base + 4096, meta=6, extra=0)
        write_rds5_pool(misaligned, [rec], [bytes(6 * 4096)])
        pool_m = ttpf.Pool(misaligned)
        self.addCleanup(pool_m.close)
        ok_meta, _ = ttpf.validate_chunk_metadata(pool_m, pool_m.descriptor(0), 6 * 4096)
        self.assertFalse(ok_meta)

        # non_tail == 3 alignment-padding validation (align_group 1/2/3 positive + negative).
        # Group 1: chunk0 (4 pages) + pad (12) == 16; chunks = [4, 16, 64].
        # Group 2: chunk0 (1) + chunk1 (3) + pad (12) == 16.
        # Group 3: chunk0 (1) + chunk1 (3) + pad (4) == 8.
        cases = [
            ("g1_ok", (4 | (11 << 5) | (15 << 10)), (1 << 19) | (12 << 21), [4, 16, 64], True),
            ("g1_bad", (4 | (11 << 5) | (15 << 10)), (1 << 19) | (11 << 21), [4, 16, 64], False),
            ("g2_ok", (1 | (3 << 5) | (11 << 10)), (2 << 19) | (12 << 21), [1, 3, 16], True),
            ("g3_ok", (1 | (3 << 5) | (8 << 10)), (3 << 19) | (4 << 21), [1, 3, 8], True),
            ("g0_bad", (1 | (3 << 5) | (8 << 10)), (0 << 19) | (4 << 21), [1, 3, 8], False),
        ]
        for tag, val20, hi_extra, page_counts, expected in cases:
            meta = val20 & 0x7FF
            extra = ((val20 >> 11) & 0x1FF) | hi_extra
            p_path = os.path.join(self.tmp.name, f"align_{tag}.ttpf")
            chunks = [bytes(pg * 4096) for pg in page_counts]
            span = sum(len(c) for c in chunks)
            r = pack_etpf_record(256, 256, 6, 3, 71, 1, 1, 4, base, meta=meta, extra=extra)
            write_rds5_pool(p_path, [r], chunks)
            pl = ttpf.Pool(p_path)
            self.addCleanup(pl.close)
            ok_al, _ = ttpf.validate_chunk_metadata(pl, pl.descriptor(0), span)
            self.assertEqual(ok_al, expected, tag)


class SyntheticVolumePoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _build_volume_pool(self, path, width, height, depth, fmt, mips=1, mode=2):
        eb = ttpf.element_bytes(fmt)
        if fmt in ttpf.VOLUME_FLOAT_FORMATS and mips == 1:
            layout = ttpf.volume_surface_layout(width, height, depth, mips, eb)
            channels = ttpf.VOLUME_FLOAT_FORMATS[fmt]
            vals = []
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        if channels == 1:
                            vals.append(10.0 + 0.05 * z + 0.1 * x + 0.1 * y)
                        else:
                            vals.extend((
                                10.0 + 0.02 * z + 0.05 * x,
                                5.0 + 0.02 * z + 0.05 * y,
                                8.0 + 0.01 * z,
                                1.0,
                            ))
            linear = struct.pack(f"<{len(vals)}e", *vals)
            tiled = retile_volume(layout, linear)
        else:
            layout = None
            linear = b""
            tiled = b"\x00" * 4096
        base = ttpf.align(16 + 24 + 32, 4096)
        rec = pack_etpf_record(width, height, mips, 0, fmt, depth, 3, mode, base)
        write_rds5_pool(path, [rec], [tiled])
        return layout, linear, tiled

    def test_volume_fmt54_and_fmt10_round_trip_check_export_and_scan(self):
        import io
        from contextlib import redirect_stdout

        cases = [
            # fmt=54 (R16_FLOAT): depth=8 pads to block_d=16 -> padded_d=16
            (16, 8, 8, 54, 16, 16),
            # fmt=10 (R16G16B16A16_FLOAT): block_d=8 -> padded_d=8
            (8, 8, 8, 10, 8, 8),
        ]
        for w, h, d, fmt, exp_bd, exp_pd in cases:
            with self.subTest(dims=(w, h, d), fmt=fmt):
                path = os.path.join(self.tmp.name, f"vol_fmt{fmt}.ttpf")
                layout, linear, tiled = self._build_volume_pool(path, w, h, d, fmt)
                pool = ttpf.Pool(path)
                self.addCleanup(pool.close)

                a = ttpf.analyze(pool, 0)
                self.assertEqual(a["shape"], "volume")
                self.assertTrue(a["span_matches"])
                self.assertTrue(a["full_chain"])
                self.assertEqual((a["layout"]["block_d"], a["layout"]["padded_d"]), (exp_bd, exp_pd))

                desc = ttpf.describe(pool, 0)
                self.assertEqual(desc["shape"], "volume")
                self.assertEqual((desc["block_d"], desc["padded_d"]), (exp_bd, exp_pd))

                # Byte-exact 3D detile round-trip.
                self.assertEqual(ttpf.detile_volume(layout, tiled), linear)

                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = ttpf.check(pool, 0)
                res = json.loads(buf.getvalue())
                self.assertEqual(rc, 0, res)
                self.assertLess(res["volume_3d_z_diff"], 0.5 * res["volume_2d_control_z_diff"])

                out_dds = os.path.join(self.tmp.name, f"vol_fmt{fmt}.dds")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(ttpf.export(pool, 0, out_dds), 0)
                with open(out_dds, "rb") as f:
                    dds = f.read()
                self.assertEqual(dds[:4], b"DDS ")
                dds_depth = struct.unpack_from("<I", dds, 24)[0]
                caps2 = struct.unpack_from("<I", dds, 112)[0]
                self.assertEqual(dds_depth, d)
                self.assertEqual(caps2 & 0x200000, 0x200000)
                dxgi_fmt, dim, misc, array_size, _ = struct.unpack_from("<IIIII", dds, 128)
                self.assertEqual((dxgi_fmt, dim, misc, array_size), (fmt, 4, 0, 1))
                self.assertEqual(dds[148:], linear)

                scan_out = os.path.join(self.tmp.name, f"scan_vol{fmt}.json")
                with redirect_stdout(io.StringIO()):
                    ttpf.scan_pools([path], scan_out)
                with open(scan_out, encoding="utf-8") as f:
                    total = json.load(f)["total"]
                self.assertEqual((total["volumes"], total["volumes_full"]), (1, 1))
                self.assertEqual((total["after_full"], total["after_skip"]), (1, 0))

    def test_volume_negative_controls(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # mips=2 volume rejected.
        bad_mips = os.path.join(self.tmp.name, "vol_mips2.ttpf")
        self._build_volume_pool(bad_mips, 16, 8, 8, 54, mips=2)
        pool_m = ttpf.Pool(bad_mips)
        self.addCleanup(pool_m.close)
        am = ttpf.analyze(pool_m, 0)
        self.assertIn("unsupported 3D volume", am["blocked"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.check(pool_m, 0), 2)

        # BC volume (fmt=71) rejected.
        bad_bc = os.path.join(self.tmp.name, "vol_bc1.ttpf")
        self._build_volume_pool(bad_bc, 16, 16, 8, 71, mips=1)
        pool_bc = ttpf.Pool(bad_bc)
        self.addCleanup(pool_bc.close)
        abc = ttpf.analyze(pool_bc, 0)
        self.assertIn("unsupported 3D volume", abc["blocked"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(ttpf.export(pool_bc, 0, os.path.join(self.tmp.name, "bad_vol.dds")), 2)


class Uncompressed2DTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def _encode_pixel(fmt, r, g, b, a=255):
        if fmt in (61, 65):
            return bytes((r,))
        if fmt in (28, 29):
            return bytes((r, g, b, a))
        if fmt in (87, 91):
            return bytes((b, g, r, a))
        if fmt == 49:
            return bytes((r, g))
        if fmt == 56:
            return struct.pack("<H", r * 257)
        if fmt == 54:
            return struct.pack("<e", r / 64.0)
        if fmt == 11:
            return struct.pack("<HHHH", r * 257, g * 257, b * 257, a * 257)
        if fmt == 10:
            return struct.pack("<eeee", r / 64.0, g / 64.0, b / 64.0, 1.0)
        if fmt == 2:
            return struct.pack("<ffff", r / 255.0, g / 255.0, b / 255.0, 1.0)
        raise AssertionError(f"unsupported test format {fmt}")

    def _build_uncompressed_pool(self, path, width, height, fmt, mips, mode=2,
                                 corrupt_mip=None):
        tile = ttpf.tile_bytes(mode)
        layout = ttpf.surface_layout(width, height, fmt, mips, tile)
        eb = layout["element_bytes"]
        # Build top mip gradient, then exact 2x2 box-downsample each child mip.
        grids = {}
        top = layout["mips"][0]
        grids[0] = [
            (min(255, x * 255 // max(1, top["pixels_w"] - 1)),
             min(255, y * 255 // max(1, top["pixels_h"] - 1)),
             (x * 7 + y * 13) & 0xFF)
            for y in range(top["pixels_h"])
            for x in range(top["pixels_w"])
        ]
        for k in range(1, mips):
            pm, cm = layout["mips"][k - 1], layout["mips"][k]
            pw, ph = pm["pixels_w"], pm["pixels_h"]
            cw, ch = cm["pixels_w"], cm["pixels_h"]
            parent = grids[k - 1]
            cur = []
            for y in range(ch):
                y0, y1 = min(ph - 1, 2 * y), min(ph - 1, 2 * y + 1)
                for x in range(cw):
                    x0, x1 = min(pw - 1, 2 * x), min(pw - 1, 2 * x + 1)
                    p00 = parent[y0 * pw + x0]
                    p10 = parent[y0 * pw + x1]
                    p01 = parent[y1 * pw + x0]
                    p11 = parent[y1 * pw + x1]
                    cur.append(tuple((p00[c] + p10[c] + p01[c] + p11[c] + 2) // 4 for c in range(3)))
            grids[k] = cur

        tiled = bytearray(layout["size"])
        expected_payload = bytearray()
        for k, mip in enumerate(layout["mips"]):
            bw, bh = mip["blocks_w"], mip["blocks_h"]
            pw, ph = mip["pixels_w"], mip["pixels_h"]
            lin = bytearray(bw * bh * eb)
            vis = bytearray()
            g = grids[k]
            for y in range(bh):
                for x in range(bw):
                    if x < pw and y < ph:
                        r, gr, b = g[y * pw + x]
                        if corrupt_mip == k:
                            r, gr, b = 255 - r, 255 - gr, 255 - b
                        px = self._encode_pixel(fmt, r, gr, b)
                        vis += px
                    else:
                        # Non-zero sentinel in padding cells to verify cropping and masking.
                        px = self._encode_pixel(fmt, 255, 0, 255)
                    at = (y * bw + x) * eb
                    lin[at:at + eb] = px
            retile_to(layout, mip, lin, tiled)
            expected_payload += vis

        base = ttpf.align(16 + 24 + 32, 4096)
        non_tail = min(7, layout["first_tail"])
        rec = pack_etpf_record(width, height, mips - non_tail, non_tail, fmt, 1, 1, mode, base)
        write_rds5_pool(path, [rec], [bytes(tiled)])
        return layout, bytes(expected_payload)

    def test_rdna2_sw_64k_d_4bpp_pattern_and_round_trip(self):
        # Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO (RDNA2/Xbox Series): 4 BPP swaps
        # bits 6..7 vs Navi 1x to (3, -3).
        self.assertEqual(
            ttpf.PATTERNS_64K[2],
            (0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7),
        )
        # Pixel (x=4, y=0) sets X2 (bit 6 -> +64 bytes); pixel (x=0, y=4) sets Y2 (bit 7 -> +128 bytes).
        layout = ttpf.surface_layout(128, 128, 28, 1, tile=65536)
        mip = layout["mips"][0]
        tiled = bytearray(65536)
        tiled[64:68] = b"\x11\x22\x33\x44"
        tiled[128:132] = b"\x55\x66\x77\x88"
        det = ttpf.detile(layout, mip, bytes(tiled))
        self.assertEqual(det[4 * 4:4 * 4 + 4], b"\x11\x22\x33\x44")
        self.assertEqual(det[(4 * 128) * 4:(4 * 128) * 4 + 4], b"\x55\x66\x77\x88")

    def test_decode_pixels_all_validated_uncompressed_formats(self):
        self.assertEqual(ttpf.ELEMENT_BYTES[11], 8)
        for fmt in sorted(ttpf.UNCOMPRESSED_2D_FORMATS):
            with self.subTest(fmt=fmt):
                raw = self._encode_pixel(fmt, 64, 128, 192, 255)
                px = ttpf.decode_pixels(fmt, raw, 1, 1)
                self.assertEqual(len(px), 1)
                r, g, b = px[0]
                if fmt in (61, 65, 56):
                    self.assertAlmostEqual(r, 64.0, places=3)
                    self.assertAlmostEqual(g, 64.0, places=3)
                    self.assertAlmostEqual(b, 64.0, places=3)
                elif fmt == 49:
                    self.assertEqual((r, g, b), (64.0, 128.0, 0.0))
                elif fmt == 54:
                    self.assertAlmostEqual(r, 1.0, places=3)
                    self.assertAlmostEqual(g, 1.0, places=3)
                    self.assertAlmostEqual(b, 1.0, places=3)
                elif fmt == 10:
                    self.assertAlmostEqual(r, 1.0, places=3)
                    self.assertAlmostEqual(g, 2.0, places=3)
                    self.assertAlmostEqual(b, 3.0, places=3)
                else:  # 11, 28, 29, 87, 91
                    self.assertAlmostEqual(r, 64.0, places=3)
                    self.assertAlmostEqual(g, 128.0, places=3)
                    self.assertAlmostEqual(b, 192.0, places=3)
        # Non-finite float16 values clamp cleanly in _hdr_tonemap.
        self.assertEqual(ttpf._hdr_tonemap(float("nan")), 0.0)
        self.assertEqual(ttpf._hdr_tonemap(float("inf")), 255.0)
        self.assertEqual(ttpf._hdr_tonemap(float("-inf")), -255.0)

    def test_uncompressed_2d_check_export_and_scan(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        cases = [
            # (name, w, h, fmt, mips, mode, expect_tail_control)
            ("dxgi11_1mip_4k", 8, 4, 11, 1, 2, False),
            ("dxgi11_multi_4k", 32, 32, 11, 4, 2, True),
            ("dxgi28_multi_64k", 256, 256, 28, 5, 0, True),
            ("dxgi28_odd_4k", 17, 13, 28, 4, 2, True),
            ("dxgi49_multi_4k", 64, 64, 49, 4, 2, True),
            ("dxgi56_multi_4k", 64, 64, 56, 4, 2, True),
            ("dxgi54_multi_64k", 128, 128, 54, 3, 0, False),
            ("dxgi61_multi_64k", 256, 256, 61, 4, 0, True),
            ("dxgi65_multi_4k", 64, 64, 65, 5, 2, True),
            ("dxgi87_multi_4k", 64, 64, 87, 5, 2, True),
            ("dxgi10_multi_4k", 32, 32, 10, 4, 2, True),
        ]
        for name, w, h, fmt, mips, mode, expect_ctrl in cases:
            with self.subTest(case=name):
                path = os.path.join(self.tmp.name, f"{name}.ttpf")
                _, expected_payload = self._build_uncompressed_pool(path, w, h, fmt, mips, mode)
                pool = ttpf.Pool(path)
                self.addCleanup(pool.close)

                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = ttpf.check(pool, 0)
                res = json.loads(buf.getvalue())
                self.assertEqual(rc, 0, res)
                if mips > 1:
                    self.assertLessEqual(res["max_error"], 1.0, res)
                if expect_ctrl and res["tail_origin_control"] is not None:
                    self.assertGreater(res["tail_origin_control"], 6.0, res)

                out_dds = os.path.join(self.tmp.name, f"{name}.dds")
                with redirect_stdout(io.StringIO()):
                    erc = ttpf.export(pool, 0, out_dds)
                self.assertEqual(erc, 0)
                with open(out_dds, "rb") as f:
                    dds = f.read()
                self.assertEqual(dds[:4], b"DDS ")
                flags, dh, dw, pitch = struct.unpack_from("<IIII", dds, 8)
                self.assertEqual(flags & 0x8, 0x8)  # DDSD_PITCH for uncompressed
                self.assertEqual((dw, dh, pitch), (w, h, w * ttpf.ELEMENT_BYTES[fmt]))
                dxgi_fmt, dim, misc, arr, _ = struct.unpack_from("<IIIII", dds, 128)
                self.assertEqual((dxgi_fmt, dim, misc, arr), (fmt, 3, 0, 1))
                self.assertEqual(dds[148:], expected_payload)

                if fmt == 11:
                    scan_out = os.path.join(self.tmp.name, f"scan_{name}.json")
                    with redirect_stdout(io.StringIO()):
                        ttpf.scan_pools([path], scan_out)
                    with open(scan_out, encoding="utf-8") as f:
                        total = json.load(f)["total"]
                    self.assertEqual((total["unsupported_format"], total["format11"]), (0, 0))
                    self.assertEqual((total["after_full"], total["after_skip"]), (1, 0))

        # Negative control 1: corrupted child mip fails check (rc == 1).
        bad = os.path.join(self.tmp.name, "dxgi28_bad.ttpf")
        self._build_uncompressed_pool(bad, 256, 256, 28, 4, mode=0, corrupt_mip=2)
        pool_bad = ttpf.Pool(bad)
        self.addCleanup(pool_bad.close)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ttpf.check(pool_bad, 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(rc, 1, res)
        self.assertGreater(res["max_error"], 50.0)

        # Negative control 2: unvalidated uncompressed format (DXGI 2, R32G32B32A32_FLOAT)
        # is still blocked in check (rc == 2) and refused by export (rc == 2).
        unval = os.path.join(self.tmp.name, "dxgi2_unval.ttpf")
        self._build_uncompressed_pool(unval, 16, 16, 2, 2, mode=2)
        pool_u = ttpf.Pool(unval)
        self.addCleanup(pool_u.close)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.check(pool_u, 0), 2)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(ttpf.export(pool_u, 0, os.path.join(self.tmp.name, "dxgi2.dds")), 2)


class VolumePatternTest(unittest.TestCase):
    """3D SW_4KB_S: the intra-tile bit order must be AMD's, and the tile extents must follow from it."""

    # Mesa src/amd/addrlib/src/gfx10/gfx10SwizzlePattern.h, GFX10_SW_4K_S3_PATINFO (1 pipe, 1/2/4/8/16 bpe):
    # nibble01 + nibble2 entries; GFX10_SW_4K_S3_RBPLUS_PATINFO gives the same rows.
    MESA_SW_4K_S3 = {
        0: (1, 2, 101, -1, 102, -2, 3, 103, -3, 4, 104, -4),
        1: (0, 1, 101, -1, 102, -2, 2, 103, -3, 3, 104, -4),
        2: (0, 0, 1, -1, 101, -2, 2, 102, -3, 3, 103, -4),
        3: (0, 0, 0, 1, 101, -1, 2, 102, -2, 3, 103, -3),
        4: (0, 0, 0, 0, 101, -1, 1, 102, -2, 2, 103, -3),
    }

    def test_patterns_are_amds_sw_4k_s3(self):
        self.assertEqual(ttpf.VOLUME_PATTERNS_4K, self.MESA_SW_4K_S3)

    def test_each_row_uses_every_x_y_z_bit_once_from_bit_0(self):
        for log, pat in ttpf.VOLUME_PATTERNS_4K.items():
            with self.subTest(element_bytes=1 << log):
                self.assertEqual(pat[:log], (0,) * log)        # the element's own byte-address bits
                for axis, keep in (("x", lambda s: 0 < s < 100), ("y", lambda s: s < 0), ("z", lambda s: s >= 101)):
                    bits = sorted(abs(s) - (100 if s >= 101 else 0) for s in pat if keep(s))
                    self.assertEqual(bits, list(range(1, len(bits) + 1)), axis)

    def test_tile_extents_follow_from_the_bit_order(self):
        tex = sys.modules["tex_headers"]   # imported by ttpf_mips.py
        for log, pat in ttpf.VOLUME_PATTERNS_4K.items():
            with self.subTest(element_bytes=1 << log):
                want = (1 << sum(0 < s < 100 for s in pat), 1 << sum(s < 0 for s in pat), 1 << sum(s >= 101 for s in pat))
                self.assertEqual(tex._VOLUME_4K_TILE[log], want)
                lay = tex.volume_surface_layout(64, 64, 32, 1, 1 << log)
                self.assertEqual((lay["block_w"], lay["block_h"], lay["block_d"]), want)
                self.assertEqual(lay["z_tile_stride"] % 4096, 0)


class UnmappedSentinelSpanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_span_skips_unmapped_sentinels_and_rejects_unmapped_base(self):
        # A valid 256x256 BC1 texture followed by unmapped ETPF records (base=0 and base=0x1FFFFF*4096),
        # and then a second valid texture, must resolve both mapped spans cleanly and return None for
        # the unmapped records (including in scan_pools, which previously passed a negative span).
        layout = ttpf.surface_layout(256, 256, 71, 7, ttpf.tile_bytes(4))
        size = layout["size"]
        base0 = 4096
        base1 = base0 + size
        rec0 = pack_etpf_record(256, 256, 5, 2, 71, 1, 1, 4, base0)
        rec_zero = pack_etpf_record(256, 256, 5, 2, 71, 1, 1, 4, 0)
        rec_max = pack_etpf_record(256, 256, 5, 2, 71, 1, 1, 4, 0x1FFFFF * 4096)
        rec1 = pack_etpf_record(256, 256, 5, 2, 71, 1, 1, 4, base1)

        for label, recs, chunks, mapped_indices in (
            ("zero_between", [rec0, rec_zero, rec1], [bytes(size), bytes(size)], (0, 2)),
            ("max_between", [rec0, rec_max, rec1], [bytes(size), bytes(size)], (0, 2)),
            ("trailing_zero_and_max", [rec0, rec_zero, rec_max], [bytes(size)], (0,)),
        ):
            with self.subTest(case=label):
                path = os.path.join(self.tmp.name, f"{label}.ttpf")
                write_rds5_pool(path, recs, chunks)
                pool = ttpf.Pool(path)
                self.addCleanup(pool.close)
                for idx in mapped_indices:
                    d = pool.descriptor(idx)
                    self.assertEqual(pool.span(idx, d["base"]), size)
                    a = ttpf.analyze(pool, idx)
                    self.assertTrue(a["span_matches"], a)
                    self.assertNotIn("blocked", a)
                for idx in range(len(recs)):
                    if idx not in mapped_indices:
                        d = pool.descriptor(idx)
                        self.assertIsNone(pool.span(idx, d["base"]))
                        a = ttpf.analyze(pool, idx)
                        self.assertEqual(a["blocked"], "no mapped allocation")
                scan_out = os.path.join(self.tmp.name, f"{label}.json")
                with redirect_stdout(io.StringIO()):
                    ttpf.scan_pools([path], scan_out)
                with open(scan_out, encoding="utf-8") as f:
                    total = json.load(f)["total"]
                self.assertEqual(total["chunk_metadata_fail"], 0)
                self.assertEqual(total["chunk_metadata_ok"], len(mapped_indices))


class DetileFastPathTest(unittest.TestCase):
    def test_coalesce_x_runs_merges_contiguous_blocks(self):
        self.assertEqual(ttpf._coalesce_x_runs([], 16), [])
        self.assertEqual(ttpf._coalesce_x_runs([0, 16, 32, 128, 144, 64], 16),
                         [(0, 48), (128, 32), (64, 16)])

    def test_detile_matches_reference_formula_across_all_element_sizes_and_tiles(self):
        def ref_detile(layout, mip, tiled):
            pat = (ttpf.PATTERNS_4K if layout["tile"] == 4096 else ttpf.PATTERNS_64K)[
                {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[layout["element_bytes"]]]
            bw, bh = layout["block_w"], layout["block_h"]
            eb = layout["element_bytes"]
            out = bytearray(mip["blocks_w"] * mip["blocks_h"] * eb)
            for y in range(mip["blocks_h"]):
                for x in range(mip["blocks_w"]):
                    off = ((y // bh) * (mip["padded_w"] // bw) + x // bw) * layout["tile"]
                    for bit, s in enumerate(pat):
                        if s > 0:
                            off |= (((x + mip["tile_x"]) >> (s - 1)) & 1) << bit
                        elif s < 0:
                            off |= (((y + mip["tile_y"]) >> (-s - 1)) & 1) << bit
                    at = (y * mip["blocks_w"] + x) * eb
                    out[at:at + eb] = tiled[off:off + eb]
            return bytes(out)

        for fmt, eb in ((61, 1), (49, 2), (28, 4), (71, 8), (98, 16)):
            for tile in (4096, 65536):
                layout = ttpf.surface_layout(60, 44, fmt, 4, tile=tile)
                for idx, mip in enumerate(layout["mips"]):
                    nbytes = max(mip["padded_w"] * mip["padded_h"] * eb, tile)
                    tiled = bytes((i * 131 + 17) & 255 for i in range(nbytes))
                    with self.subTest(fmt=fmt, tile=tile, level=idx):
                        self.assertEqual(ttpf.detile(layout, mip, tiled), ref_detile(layout, mip, tiled))

        # Truncated input falls back to the exact slice behavior without raising ValueError.
        layout_t = ttpf.surface_layout(64, 64, 98, 1, tile=4096)
        mip_t = layout_t["mips"][0]
        short_tiled = bytes(range(200))
        self.assertEqual(ttpf.detile(layout_t, mip_t, short_tiled), ref_detile(layout_t, mip_t, short_tiled))


    @staticmethod
    def _ref_detile(layout, mip, tiled):
        pat = (ttpf.PATTERNS_4K if layout["tile"] == 4096 else ttpf.PATTERNS_64K)[
            {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[layout["element_bytes"]]]
        bw, bh, eb = layout["block_w"], layout["block_h"], layout["element_bytes"]
        out = bytearray(mip["blocks_w"] * mip["blocks_h"] * eb)
        for y in range(mip["blocks_h"]):
            for x in range(mip["blocks_w"]):
                off = ((y // bh) * (mip["padded_w"] // bw) + x // bw) * layout["tile"]
                for bit, s in enumerate(pat):
                    if s > 0:
                        off |= (((x + mip["tile_x"]) >> (s - 1)) & 1) << bit
                    elif s < 0:
                        off |= (((y + mip["tile_y"]) >> (-s - 1)) & 1) << bit
                at = (y * mip["blocks_w"] + x) * eb
                out[at:at + eb] = tiled[off:off + eb]
        return bytes(out)

    def test_narrow_shapes_stream_and_match_reference(self):
        for w, h in ((16384, 1), (1, 16384), (1, 1), (3, 1), (1, 3)):
            for tile in (4096, 65536):
                layout = ttpf.surface_layout(w, h, 28, 1, tile=tile)
                mip = layout["mips"][0]
                nbytes = max(mip["padded_w"] * mip["padded_h"] * 4, tile)
                full = bytes((i * 131 + 17) & 255 for i in range(nbytes))
                for tiled in (full, full[:nbytes // 3 + 5]):  # complete and truncated input
                    with self.subTest(w=w, h=h, tile=tile, size=len(tiled)):
                        self.assertEqual(ttpf.detile(layout, mip, tiled),
                                         self._ref_detile(layout, mip, tiled))

    def test_volume_slice_nonzero_z_truncated_matches_reference(self):
        layout = ttpf.volume_surface_layout(16, 8, 8, 1, 2)
        w, h, d = (layout["mips"][0][k] for k in ("blocks_w", "blocks_h", "blocks_d"))
        linear = bytes((i * 7 + 3) & 255 for i in range(w * h * d * 2))
        tiled = retile_volume(layout, linear)
        for z in (0, 3, 7):
            self.assertEqual(ttpf.detile_volume_slice(layout, tiled, z),
                             linear[z * w * h * 2:(z + 1) * w * h * 2])
        # Truncated allocation: the exact slice-copy fallback must match a per-element reference.
        eb = layout["element_bytes"]
        pat = ttpf.VOLUME_PATTERNS_4K[{1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]]
        bw, bh, bd = layout["block_w"], layout["block_h"], layout["block_d"]
        tiles_x = layout["mips"][0]["padded_w"] // bw
        cut = tiled[:len(tiled) // 2 + 3]
        for z in (1, 5):
            ref = bytearray(w * h * eb)
            for y in range(h):
                for x in range(w):
                    off = (z // bd) * layout["z_tile_stride"] + ((y // bh) * tiles_x + x // bw) * layout["tile"]
                    for bit, s in enumerate(pat):
                        if s >= 101:
                            off |= ((z >> (s - 101)) & 1) << bit
                        elif s < 0:
                            off |= ((y >> (-s - 1)) & 1) << bit
                        elif 0 < s < 100:
                            off |= ((x >> (s - 1)) & 1) << bit
                    at = (y * w + x) * eb
                    ref[at:at + eb] = cut[off:off + eb]
            self.assertEqual(ttpf.detile_volume_slice(layout, cut, z), bytes(ref))
        self.assertEqual(ttpf.detile_volume_slice(layout, tiled, 0), linear[:w * h * 2])


# --- The BC1/BC4 decode and block_means as they were before the table-driven rewrite -------
# Kept verbatim as the reference the fast paths must reproduce float for float.

def _ref_rgb565(c):
    return ((c >> 11) & 31) * 255.0 / 31, ((c >> 5) & 63) * 255.0 / 63, (c & 31) * 255.0 / 31


def _ref_bc1_block(data, at, four_color=False):
    c0, c1, idx = struct.unpack_from("<HHI", data, at)
    p0, p1 = _ref_rgb565(c0), _ref_rgb565(c1)
    if four_color or c0 > c1:
        pal = [p0, p1, tuple((2 * a + b) / 3 for a, b in zip(p0, p1)),
               tuple((a + 2 * b) / 3 for a, b in zip(p0, p1))]
    else:
        pal = [p0, p1, tuple((a + b) / 2 for a, b in zip(p0, p1)), (0.0, 0.0, 0.0)]
    return [pal[(idx >> (2 * (y * 4 + x))) & 3] for y in range(4) for x in range(4)]


def _ref_bc4_block(data, at):
    e0, e1 = data[at], data[at + 1]
    bits = int.from_bytes(data[at + 2:at + 8], "little")
    if e0 > e1:
        pal = [e0, e1] + [((7 - i) * e0 + i * e1) // 7 for i in range(1, 7)]
    else:
        pal = [e0, e1] + [((5 - i) * e0 + i * e1) // 5 for i in range(1, 5)] + [0, 255]
    vals = [pal[(bits >> (3 * (y * 4 + x))) & 7] for y in range(4) for x in range(4)]
    return [(float(v),) * 3 for v in vals]


def _ref_decode_pixels(fmt, data, blocks_w, blocks_h):
    px = []
    for by in range(blocks_h):
        rows = [[None] * (blocks_w * 4) for _ in range(4)]
        for bx in range(blocks_w):
            at = (by * blocks_w + bx) * 16
            if fmt in (71, 72):
                blk = _ref_bc1_block(data, (by * blocks_w + bx) * 8)
            elif fmt in (74, 75, 77, 78):
                blk = _ref_bc1_block(data, at + 8, four_color=True)
            elif fmt in (80,):
                blk = _ref_bc4_block(data, (by * blocks_w + bx) * 8)
            elif fmt in (83,):
                a = _ref_bc4_block(data, at)
                b = _ref_bc4_block(data, at + 8)
                blk = [(p[0], q[0], 0.0) for p, q in zip(a, b)]
            else:
                return None
            for y in range(4):
                for x in range(4):
                    rows[y][bx * 4 + x] = blk[y * 4 + x]
        px.extend(rows)
    return [p for row in px for p in row]


def _ref_block_means(pixels, blocks_w, blocks_h, pixels_w=None, pixels_h=None, block=4):
    if block == 1:
        return list(pixels)
    pw = blocks_w * block if pixels_w is None else pixels_w
    ph = blocks_h * block if pixels_h is None else pixels_h
    out = []
    for by in range(blocks_h):
        for bx in range(blocks_w):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for y in range(block):
                if by * block + y >= ph:
                    continue
                for x in range(block):
                    if bx * block + x >= pw:
                        continue
                    q = pixels[(by * block + y) * (blocks_w * block) + bx * block + x]
                    for k in range(3):
                        acc[k] += q[k]
                    n += 1
            out.append((acc[0] / n, acc[1] / n, acc[2] / n) if n else (0.0, 0.0, 0.0))
    return out


def _float_bits(pixels):
    """Every channel's IEEE bits (so -0.0 counts); each channel must be a float. All NaNs map
    to one: which NaN comes out of `NaN + NaN` follows the operand order the interpreter's
    float add happens to use (it differs between a cold and a specialized `+`), in the scalar
    loop as much as here, and no consumer of the means reads it (`_hdr_tonemap` maps NaN to 0)."""
    flat = [c for p in pixels for c in p]
    assert all(type(c) is float for c in flat), "a channel is not a float"
    return struct.pack(f"<{len(flat)}d", *[float("nan") if c != c else c for c in flat])


class BcFastDecodeParityTest(unittest.TestCase):
    """decode_bc1_block / decode_bc4_block / decode_pixels / block_means against the scalar
    code they replaced, plus hand-derived values that need no reference."""

    def assert_same_pixels(self, got, want):
        self.assertEqual(len(got), len(want))
        self.assertEqual(_float_bits(got), _float_bits(want))

    def test_bc1_hand_derived_palette_and_pixel_order(self):
        red, blue = 0xF800, 0x001F
        # c0 > c1: four colors; index = row number, so row y is all pal[y].
        blk = struct.pack("<HHI", red, blue, 0xFFAA5500)
        pal = [(255.0, 0.0, 0.0), (0.0, 0.0, 255.0), (170.0, 0.0, 85.0), (85.0, 0.0, 170.0)]
        px = ttpf.decode_pixels(71, blk, 1, 1)
        self.assertEqual(px, [pal[y] for y in range(4) for _ in range(4)])
        self.assertEqual(ttpf.decode_bc1_block(blk, 0), px)
        # index = column number: every row is pal[0..3]
        blk = struct.pack("<HHI", red, blue, 0xE4E4E4E4)
        self.assertEqual(ttpf.decode_pixels(72, blk, 1, 1), pal * 4)
        # c0 <= c1: three colors and black; index 3 is (0, 0, 0), index 2 the midpoint.
        blk = struct.pack("<HHI", blue, red, 0xE4E4E4E4)
        pal3 = [(0.0, 0.0, 255.0), (255.0, 0.0, 0.0), (127.5, 0.0, 127.5), (0.0, 0.0, 0.0)]
        self.assertEqual(ttpf.decode_pixels(71, blk, 1, 1), pal3 * 4)
        # BC3 reads the same color block four-color even when c0 <= c1 (alpha bytes first).
        self.assertEqual(ttpf.decode_pixels(77, b"\xAA" * 8 + blk, 1, 1),
                         [(0.0, 0.0, 255.0), (255.0, 0.0, 0.0), (85.0, 0.0, 170.0),
                          (170.0, 0.0, 85.0)] * 4)
        # RGB565 channel levels: the 5/6/5-bit value scaled to 255 in float, not rounded.
        self.assertEqual(ttpf._rgb565(0x7BEF), (15 * 255.0 / 31, 31 * 255.0 / 63, 15 * 255.0 / 31))
        self.assertEqual(ttpf._rgb565(0xFFFF), (255.0, 255.0, 255.0))
        self.assertEqual(ttpf._rgb565(0), (0.0, 0.0, 0.0))

    def test_bc4_hand_derived_pixel_order(self):
        # 8-value mode, pixel (y, x) takes index (4 * y + x) % 8; palette is e0, e1, then
        # (6e0 + e1)//7 ... (e0 + 6e1)//7.
        e0, e1 = 200, 11
        bits = sum(((4 * y + x) % 8) << (3 * (4 * y + x)) for y in range(4) for x in range(4))
        blk = bytes([e0, e1]) + bits.to_bytes(6, "little")
        pal = [200, 11, 173, 146, 119, 92, 65, 38]
        want = [(float(pal[(4 * y + x) % 8]),) * 3 for y in range(4) for x in range(4)]
        self.assertEqual(ttpf.decode_pixels(80, blk, 1, 1), want)
        self.assertEqual(ttpf.decode_bc4_block(blk, 0), want)
        # 6-value mode: the last two indices are the constants 0 and 255.
        blk = bytes([20, 220]) + bits.to_bytes(6, "little")
        pal = [20, 220, 60, 100, 140, 180, 0, 255]
        self.assertEqual(ttpf.decode_bc4_block(blk, 0),
                         [(float(pal[(4 * y + x) % 8]),) * 3 for y in range(4) for x in range(4)])

    def test_decode_pixels_matches_reference_on_random_blocks(self):
        rng = random.Random(0xBC14)
        for fmt, size in ((71, 8), (72, 8), (74, 16), (75, 16), (77, 16), (78, 16),
                          (80, 8), (83, 16)):
            for bw, bh in ((1, 1), (7, 5), (16, 3)):
                data = rng.randbytes(bw * bh * size)
                self.assert_same_pixels(ttpf.decode_pixels(fmt, data, bw, bh),
                                        _ref_decode_pixels(fmt, data, bw, bh))

    def test_decode_pixels_matches_reference_on_edge_endpoints(self):
        # Both endpoint orders, equal endpoints, the extremes and the 6-value/8-value BC4 split.
        ends = [0, 1, 0x001F, 0x07E0, 0xF800, 0x7BEF, 0xFFFF, 0xFFFE]
        rng = random.Random(7)
        bc1 = b"".join(struct.pack("<HHI", a, b, rng.getrandbits(32)) for a in ends for b in ends)
        n = len(ends) ** 2
        self.assert_same_pixels(ttpf.decode_pixels(71, bc1, 8, n // 8),
                                _ref_decode_pixels(71, bc1, 8, n // 8))
        vals = [0, 1, 7, 127, 128, 254, 255]
        bc4 = b"".join(bytes([a, b]) + rng.randbytes(6) for a in vals for b in vals)
        self.assert_same_pixels(ttpf.decode_pixels(80, bc4, 7, 7), _ref_decode_pixels(80, bc4, 7, 7))
        for at in range(0, len(bc1) - 7, 8):
            for four in (False, True):
                self.assert_same_pixels(ttpf.decode_bc1_block(bc1, at, four),
                                        _ref_bc1_block(bc1, at, four))
        for at in range(0, len(bc4), 8):
            self.assert_same_pixels(ttpf.decode_bc4_block(bc4, at), _ref_bc4_block(bc4, at))

    def test_palette_cache_reset_does_not_change_pixels(self):
        rng = random.Random(99)
        bc1, bc4 = rng.randbytes(20 * 20 * 8), rng.randbytes(20 * 20 * 8)
        with mock.patch.object(ttpf, "_ROW_MAX_CACHE", 3):
            self.assert_same_pixels(ttpf.decode_pixels(71, bc1, 20, 20),
                                    _ref_decode_pixels(71, bc1, 20, 20))
            self.assert_same_pixels(ttpf.decode_pixels(80, bc4, 20, 20),
                                    _ref_decode_pixels(80, bc4, 20, 20))

    def test_decode_pixels_accepts_any_buffer_and_keeps_its_failures(self):
        rng = random.Random(3)
        data = rng.randbytes(6 * 8)
        want = _ref_decode_pixels(71, data, 3, 2)
        for buf in (bytearray(data), memoryview(data)):
            self.assert_same_pixels(ttpf.decode_pixels(71, buf, 3, 2), want)
        # Truncated input fails the same way as before (and in the reference).
        for fmt, size in ((71, 8), (74, 16), (80, 8)):
            short = bytes(size * 4 - (9 if fmt == 80 else 1))
            with self.assertRaises(IndexError if fmt == 80 else struct.error):
                _ref_decode_pixels(fmt, short, 2, 2)
            with self.assertRaises(IndexError if fmt == 80 else struct.error):
                ttpf.decode_pixels(fmt, short, 2, 2)
        # A BC format this decoder has no block code for is None, or [] on an empty grid.
        self.assertIsNone(ttpf.decode_pixels(70, bytes(16), 1, 1))
        self.assertEqual(ttpf.decode_pixels(70, b"", 0, 4), [])
        self.assertEqual(ttpf.decode_pixels(71, b"", 3, 0), [])
        self.assertEqual(ttpf.decode_pixels(80, b"", 0, 0), [])

    def test_bc4_levels_are_shared_and_peak_memory_does_not_grow(self):
        rng = random.Random(11)
        bw = bh = 96
        for fmt in (71, 80):
            data = rng.randbytes(bw * bh * 8)
            peaks = []
            for fn in (ttpf.decode_pixels, _ref_decode_pixels):
                tracemalloc.start()
                try:
                    keep = fn(fmt, data, bw, bh)
                    peaks.append(tracemalloc.get_traced_memory()[1])
                finally:
                    tracemalloc.stop()
                del keep
            self.assertLessEqual(peaks[0], peaks[1], f"fmt {fmt} peak {peaks}")
        px = ttpf.decode_pixels(80, data, bw, bh)
        self.assertLessEqual(len({id(p) for p in px}), 256)

    def test_block_means_matches_reference_for_every_clip(self):
        rng = random.Random(0xB10C)
        for block in (2, 3, 4, 5, 8):
            for bw, bh in ((1, 1), (2, 3), (5, 2), (7, 7)):
                w, h = bw * block, bh * block
                px = [(rng.uniform(-300, 300), rng.random(), float(rng.randrange(256)))
                      for _ in range(w * h)]
                clips = [(None, None), (w, h), (w - 1, h - 1), (w - block, h), (w, h - block + 1),
                         (1, 1), (0, h), (w, 0), (0, 0), (-3, 2), (w + 9, h + 9),
                         (max(1, w - block - 1), max(1, h - block - 1)), (w - 1, None)]
                for pw, ph in clips:
                    got = ttpf.block_means(px, bw, bh, pw, ph, block)
                    want = _ref_block_means(px, bw, bh, pw, ph, block)
                    self.assertEqual(_float_bits(got), _float_bits(want),
                                     f"block {block} grid {bw}x{bh} clip {pw}x{ph}")
                    self.assertEqual(len(got), bw * bh)

    def test_block_means_keeps_summation_order_and_special_values(self):
        # Sequential float adds from 0.0: a naive left-to-right sum, not a compensated one.
        inf, nan = float("inf"), float("nan")
        values = [1e16, 1.0, -1e16, 1.0, 0.1, 0.2, 0.3, -0.0, -0.0, inf, -inf, nan,
                  1e308, 1e308, -0.0, 5e-324]
        px = [(v, -v, v * 0.5) for v in values] * 2
        for bw, bh in ((1, 2), (2, 1)):
            for pw, ph in ((None, None), (3, 5), (4, 8)):
                self.assertEqual(_float_bits(ttpf.block_means(px, bw, bh, pw, ph)),
                                 _float_bits(_ref_block_means(px, bw, bh, pw, ph)))
        # Values whose sum depends on the order (1e16 + 1.0 == 1e16 in float).
        order = [(1e16, 0.0, 0.0), (1.0, 0.0, 0.0)] * 8
        self.assertEqual(ttpf.block_means(order, 1, 1), _ref_block_means(order, 1, 1))
        # Each +1.0 is lost against 1e16 (the spacing there is 2); a compensated sum keeps them.
        self.assertEqual(ttpf.block_means(order, 1, 1)[0][0], 8e16 / 16)
        # A block of negative zeros stays +0.0 because the sum starts from 0.0.
        zero = ttpf.block_means([(-0.0, -0.0, -0.0)] * 16, 1, 1)
        self.assertEqual([math.copysign(1.0, c) for c in zero[0]], [1.0, 1.0, 1.0])

    def test_block_means_integer_pixels_lists_and_block_one(self):
        px = [[x + 3 * y, 2 * x, y] for y in range(8) for x in range(8)]
        self.assertEqual(_float_bits(ttpf.block_means(px, 2, 2)),
                         _float_bits(_ref_block_means(px, 2, 2)))
        self.assertEqual(ttpf.block_means(px, 8, 8, block=1), px)
        self.assertEqual(ttpf.block_means(tuple(px), 2, 2), _ref_block_means(px, 2, 2))
        self.assertEqual(ttpf.block_means([], 0, 0), [])
        self.assertEqual(ttpf.block_means([], 3, 0), [])
        self.assertEqual(ttpf.block_means([], 0, 3), [])
        self.assertEqual(ttpf.block_means(px, 2, 2, block=0), [(0.0, 0.0, 0.0)] * 4)

    def test_block_means_reads_only_the_visible_texels(self):
        # 5x3 visible texels in a 2x1 block grid: the list needs only the texels the clipped
        # blocks use (rows are 8 apart), so a list ending at the last visible texel works.
        full = [(float(i), float(2 * i), 1.0) for i in range(8 * 4)]
        need = 2 * 8 + 5  # last visible texel is (row 2, column 4)
        got = ttpf.block_means(full[:need], 2, 1, 5, 3)
        self.assertEqual(got, _ref_block_means(full, 2, 1, 5, 3))
        with self.assertRaises(IndexError):
            ttpf.block_means(full[:need - 1], 2, 1, 5, 3)
        with self.assertRaises(IndexError):
            _ref_block_means(full[:need - 1], 2, 1, 5, 3)
        with self.assertRaises(IndexError):
            ttpf.block_means([(0.0, 0.0, 0.0)] * 15, 1, 1)

    def test_decode_then_means_on_a_large_surface_matches_reference(self):
        rng = random.Random(2024)
        bw, bh = 33, 17
        for fmt in (71, 80):
            data = rng.randbytes(bw * bh * 8)
            got = ttpf.decode_pixels(fmt, data, bw, bh)
            want = _ref_decode_pixels(fmt, data, bw, bh)
            self.assert_same_pixels(got, want)
            for pw, ph in ((None, None), (bw * 4 - 3, bh * 4 - 1)):
                self.assertEqual(_float_bits(ttpf.block_means(got, bw, bh, pw, ph)),
                                 _float_bits(_ref_block_means(want, bw, bh, pw, ph)))


if __name__ == "__main__":
    unittest.main()
