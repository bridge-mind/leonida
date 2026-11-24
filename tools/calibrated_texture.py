"""Numerical reconstruction with an explicitly supplied local calibration profile."""

from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zlib

# `python -I` implies `-P`, so the script's directory is not on sys.path; the sibling modules are next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import bcpack
import bcpack_tables
import ttpf_mips as ttpf


@lru_cache(maxsize=1)
def native():
    name = {
        "win32": "bcpack_calibration.dll",
        "darwin": "libbcpack_calibration.dylib",
    }.get(sys.platform, "libbcpack_calibration.so")
    path = Path(__file__).with_name("bcpack_calibration") / "target/release" / name
    lib = ctypes.CDLL(str(path))
    lib.unpack.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.unpack.restype = ctypes.c_ssize_t
    lib.rebuild.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_void_p] * 5
    lib.rebuild.restype = None
    return lib


def array(values, shape, dtype, lo, hi):
    data = np.asarray(values)
    if (
        data.shape != shape
        or data.dtype.kind not in "iu"
        or np.any(data < lo)
        or np.any(data > hi)
    ):
        raise ValueError(f"Invalid calibration field (expected integer array {shape})")
    return np.ascontiguousarray(data, dtype=dtype)


def unzig(a):
    return (a >> 1) ^ -(a & 1)


def distance(symbols, lane):
    spread = np.array(
        [sum(((v >> k) & 1) << (2 * k) for k in range(4)) for v in range(16)]
    )
    return spread[symbols[:, 0, lane]] + 2 * spread[symbols[:, 1, lane]]


def accumulate(distances, residuals):
    roots = np.arange(len(distances)) - distances
    roots[distances == 0] = -512
    while (mask := roots >= 0).any():
        parents = roots[mask]
        residuals[mask] += residuals[parents]
        roots[mask] = roots[parents]
    return roots


def bc1_blocks(s, params):
    seeds, selectors = params
    ep = unzig(s[:, 2:8].astype("int32"))
    roots = accumulate(unzig(s[:, 0].astype("int32")) % 256, ep)
    external = roots != -512
    base = seeds[-roots[external]]
    if np.any(base < 0):
        raise ValueError("Missing BC1 endpoint history")
    ep[external] += base
    ep %= (32, 64, 32, 32, 64, 32)
    rgb = (ep[:, ::3] << 11) | (ep[:, 1::3] << 5) | ep[:, 2::3]
    indices = np.bitwise_or.reduce(selectors[np.arange(4), s[:, 8:12]], axis=1)
    words = (
        rgb[:, 0].astype("uint64")
        | (rgb[:, 1].astype("uint64") << 16)
        | (indices.astype("uint64") << 32)
    )
    return words.astype("<u8").tobytes()


def bc7_blocks(s, config, seed):
    count = len(s) // 2
    out = np.zeros(count, dtype="V16")
    ok = np.zeros(count, dtype="uint8")
    labels, anchors = config
    native().rebuild(
        s.ctypes.data,
        count,
        labels.ctypes.data,
        seed.ctypes.data,
        anchors.ctypes.data,
        out.ctypes.data,
        ok.ctypes.data,
    )
    return out, ok.astype(bool)


class LocalDecoder:
    """One calibrated pool reader. Its chunk cache belongs to this instance."""

    formats = frozenset((71, 72, 77, 78, 80, 83, 95, 96, 98, 99))

    def __init__(self, pool, calibration):
        self.pool = pool
        self.cal = calibration
        if not isinstance(calibration, dict) or calibration.get("version") != 1:
            raise ValueError("Unsupported calibration version")
        self.edges = np.array([c[0] for c in pool.chunks] + [pool.virtual_size])
        self.read_chunk = lru_cache(maxsize=32)(self._read_chunk)
        self.seeds = {}
        self.config = None
        self.bc1 = None

    def seed(self, name):
        if name not in self.seeds:
            data = array(self.cal["bc7"]["seeds"][name], (256, 56, 2), "uint8", 0, 255)
            if np.any(data[:, :, 1] > 8):
                raise ValueError("Invalid history precision")
            self.seeds[name] = data
        return self.seeds[name]

    def symbols(self, c):
        _, size, offset, packed, codec, _ = self.pool.chunks[c]
        raw = self.pool.data[offset : offset + packed]
        widths = bcpack.LANE_WIDTHS[codec]
        sy = np.empty((size // 8, len(widths)), "uint8")
        used = native().unpack(
            raw,
            packed,
            bytes(w - 1 for w in widths),
            len(widths),
            len(sy),
            sy.ctypes.data,
        )
        if used != packed:
            lanes, consumed = bcpack.decode_lanes(raw, codec, size)
            if consumed != packed:
                raise ValueError(f"Entropy decode failed for chunk {c}")
            sy = np.ascontiguousarray(np.asarray(lanes, dtype="uint8").T)
        # Historical feeder interpretations may release a short group of symbols
        # in a different order. The profile identifies the exact compressed input.
        if codec == 1 and self.cal.get("compatibility"):
            cycle = self.cal["compatibility"].get(hashlib.sha256(raw).hexdigest())
            if cycle:
                if any(type(v) != int or not 0 <= v < sy.size for v in cycle) or len(
                    set(cycle)
                ) != len(cycle):
                    raise ValueError("Invalid symbol rotation")
                sy.flat[cycle] = np.roll(sy.ravel()[cycle], -1)
        return sy

    def _read_chunk(self, c, bs, profile):
        _, size, start, length, codec, _ = self.pool.chunks[c]
        if size % bs or (codec not in (0, 8) and bcpack.BLOCK_BYTES.get(codec) != bs):
            raise ValueError("Block size and codec disagree")
        if codec in (0, 8):
            raw = self.pool.data[start : start + length]
            if codec:
                raw = zlib.decompress(raw)
            if len(raw) != size:
                raise ValueError("Invalid raw chunk length")
            return np.frombuffer(raw, f"V{bs}"), None
        sy = self.symbols(c)
        if codec == 7:
            if self.config is None:
                cfg = self.cal["bc7"]
                self.config = (
                    array(cfg["labels"], (1024, 4), "int16", -1, 255),
                    array(cfg["anchors"], (128,), "uint16", 0, 65535),
                )
            names = ["early"] if profile == "early" else ["base", "secondary"]
            if profile == "refined":
                names.append("refined")
            blocks, valid = bc7_blocks(sy, self.config, self.seed(names[0]))
            for name in names[1:]:
                if valid.all():
                    break
                alternative, known = bc7_blocks(sy, self.config, self.seed(name))
                blocks[~valid] = alternative[~valid]
                valid |= known
            return blocks, valid
        if codec == 1:
            if self.bc1 is None:
                cfg = self.cal["bc1"]
                self.bc1 = (
                    array(cfg["endpoints"], (256, 6), "int32", -1, 63),
                    array(cfg["selectors"], (4, 256), "uint32", 0, 0xFFFFFFFF),
                )
            raw = bc1_blocks(sy, self.bc1)
        elif codec == 6:
            raw = bcpack._bc6h_blocks(
                [l.tobytes() for l in sy.T], size // bs, set(), history=self.cal["bc6h"]
            )
        elif codec == 4:
            table = {int(k): v for k, v in self.cal["bc4"][profile].items()}
            raw = scalar_bc4(sy, table)
        elif codec == 3:
            raw = local_bc3(sy, self.cal["bc3"])
        elif codec == 5:
            raw = local_bc5(sy)
        return np.frombuffer(raw, f"V{bs}"), None

    def surface(self, at, bs, profile):
        chunks = np.searchsorted(self.edges, at, side="right") - 1
        if np.any(chunks < 0) or np.any(chunks >= len(self.pool.chunks)):
            raise ValueError("Texture exceeds virtual pool")
        result = np.empty(len(at), f"V{bs}")
        for c in np.unique(chunks):
            mask = chunks == c
            ix = (at[mask] - self.edges[c]) // bs
            blocks, ok = self.read_chunk(int(c), bs, profile)
            chosen = blocks[ix]
            if ok is not None and not ok[ix].all():
                missing = ~ok[ix]
                if profile != "refined":
                    raise ValueError("Missing calibrated blocks")
                sy = self.symbols(c)
                key = hashlib.sha256(
                    sy.tobytes() + ix[missing].astype("<i8").tobytes()
                ).hexdigest()
                seed = self.seed("refined").copy()
                for age, field, value in self.cal["bc7"]["adjustments"][key]:
                    if not (0 <= age < 256 and 0 <= field < 24 and 0 <= value <= 255):
                        raise ValueError("Invalid fitted endpoint")
                    seed[age, field] = value, 8
                other, valid = bc7_blocks(sy, self.config, seed)
                if not valid[ix[missing]].all():
                    raise ValueError("Incomplete calibrated endpoints")
                chosen[missing] = other[ix[missing]]
            result[mask] = chosen
        return result.tobytes()

    def dds(self, index, mips=None):
        d = self.pool.descriptor(index)
        width, height, fmt = d["width"], d["height"], d["format"]
        if fmt not in self.formats or d["dimension"] != 1:
            raise ValueError(
                "Calibrated export requires a supported BCn 2D texture or array"
            )
        levels = d["mips"]
        count = levels if mips is None else mips
        if not 1 <= count <= levels:
            raise ValueError("Mip count outside descriptor range")
        bs = 8 if fmt in (71, 72, 80) else 16
        tile = ttpf.tile_bytes(d["mode"])
        faces = d["depth"]
        linear = ((width + 3) // 4) * ((height + 3) // 4) * bs
        expected = 148 + faces * sum(
            ((max(1, width >> i) + 3) // 4) * ((max(1, height >> i) + 3) // 4) * bs
            for i in range(count)
        )
        if expected > ttpf.MAX_OUTPUT:
            raise ValueError("DDS exceeds output limit")
        grids = [
            ((max(1, width >> i) + 3) // 4, (max(1, height >> i) + 3) // 4)
            for i in range(levels)
        ]
        if faces > 1 and d["mode"] == 1:
            # Mode-1 arrays group source slices by mip. DDS groups mips by slice.
            layout = ttpf.array_surface_layout(width, height, faces, levels, bs, 4, tile)
            bw, bh = layout["block_w"], layout["block_h"]
            offsets = {
                mip["index"]: (mip["offset"], mip["tile_x"], mip["tile_y"], mip["slice_stride"])
                for mip in layout["mips"]
            }
        else:
            # ASSET-568: the compact mip chain is computed in tools/tex_headers.py. Re-deriving
            # first_tail here got NPOT chains and single-mip sub-tiles wrong; consume the layout.
            bw, bh, grids, offsets = compact_dds_layout(width, height, fmt, levels, tile)
        profile = (
            self.cal.get("entries", {})
            .get(Path(self.pool.path).name, {})
            .get(str(index), "base")
        )
        span = self.pool.span(index, d["base"])
        if not span or d["base"] < self.edges[0]:
            raise ValueError("Texture has no mapped allocation")
        payload = bytearray()
        for face in range(faces):
            for i, (w, h) in enumerate(grids[:count]):
                offset, x, y, slice_stride = offsets[i]
                at = (
                    addresses(tile, bs, bw, bh, w, h, x, y)
                    + d["base"]
                    + face * slice_stride
                    + offset
                )
                if at.max() + bs > d["base"] + span:
                    raise ValueError("Texture crosses its allocation boundary")
                payload.extend(self.surface(at, bs, profile))
        return ttpf.dds_bytes(
            width, height, fmt, count, linear, bytes(payload), array_size=faces
        )


def compact_dds_layout(width, height, fmt, levels, tile):
    """Block dimensions and per-mip (offset, x, y, slice_stride) from the measured compact layout.

    ASSET-568: the packed-tail boundary and standalone subresource offsets belong to
    ``tools/tex_headers.py``; callers that re-derived them used the wrong grid on NPOT
    chains and treated a single mip as a packed tail.
    """
    layout = ttpf.surface_layout(width, height, fmt, levels, tile)
    first_tail = layout["first_tail"]
    stride = tile if first_tail < levels else 0
    offsets, dims = {}, []
    for mip in layout["mips"]:
        i = mip["index"]
        dims.append((mip["blocks_w"], mip["blocks_h"]))
        if i < first_tail:
            offsets[i] = (mip["offset"], 0, 0, stride)
        else:
            offsets[i] = (0, mip["tile_x"], mip["tile_y"], stride)
    return layout["block_w"], layout["block_h"], dims, offsets


@lru_cache(maxsize=64)
def addresses(tile, bs, bw, bh, w, h, x0, y0):
    x = np.arange(x0, x0 + w, dtype="int64")[None, :]
    y = np.arange(y0, y0 + h, dtype="int64")[:, None]
    off = ((y // bh) * ((w + bw - 1) // bw) + x // bw) * tile
    patterns = ttpf.PATTERNS_64K if tile == 65536 else ttpf.PATTERNS_4K
    for bit, axis in enumerate(patterns[bs.bit_length() - 1]):
        if axis:
            off |= (((x if axis > 0 else y) >> (abs(axis) - 1)) & 1) << bit
    return off.ravel()


def scalar_bc4(symbols, table):
    v = symbols.astype("int32")
    d = unzig(v[:, 0]) % 256
    anc = np.arange(len(d)) - d
    anc[d == 0] = -512
    delta = unzig(v[:, 2:4])
    while (mask := anc >= 0).any():
        parent = anc[mask]
        delta[mask] += delta[parent]
        anc[mask] = anc[parent]
    for slot in np.unique(anc):
        delta[anc == slot] += (128, 128) if slot == -512 else table[int(slot) + 256]
    pairs = np.asarray(bcpack_tables.BC4_PAIR, dtype="uint64")
    ranks = (pairs[:, 0] | (pairs[:, 1] << 3))[v[:, 4:]]
    selectors = np.bitwise_or.reduce(
        ranks << (6 * np.arange(8, dtype="uint64")), axis=1
    )
    ep = (delta % 256).astype("uint64")
    return ((selectors << 16) | (ep[:, 1] << 8) | ep[:, 0]).astype("<u8").tobytes()


def local_bc5(symbols):
    s = symbols.reshape(-1, 2, 12).astype("int32")
    distance_back = unzig(distance(s, 0)) % 256
    ancestors = np.arange(len(s)) - distance_back
    ancestors[distance_back == 0] = -512
    endpoints = unzig(s[:, :, 2:4])
    while (mask := ancestors >= 0).any():
        parent = ancestors[mask]
        endpoints[mask] += endpoints[parent]
        ancestors[mask] = ancestors[parent]
    table = bcpack_tables.BC5_HISTORY_PROVISIONAL | bcpack_tables.BC5_HISTORY_CONFIRMED
    for slot in np.unique(ancestors):
        seed = (128,) * 4 if slot == -512 else table[int(slot) + 256]
        endpoints[ancestors == slot] += np.asarray(seed).reshape(2, 2)
    ep = (endpoints % 256).astype("uint64")
    pairs = np.asarray(bcpack_tables.BC5_PAIR, dtype="uint64")
    ranks = s[:, :, 4:].reshape(-1, 2, 2, 4).transpose(0, 2, 3, 1).reshape(-1, 2, 8)
    values = (pairs[:, 0] | (pairs[:, 1] << 3))[ranks]
    selectors = np.bitwise_or.reduce(
        values << (6 * np.arange(8, dtype="uint64")), axis=2
    )
    return (
        ((selectors << 16) | (ep[:, :, 1] << 8) | ep[:, :, 0]).astype("<u8").tobytes()
    )


def local_bc3(symbols, profile):
    CAP = np.array([32, 64, 32, 256] * 2)
    zig = unzig
    s = symbols.reshape(-1, 2, 12).astype("int32")
    age = zig(distance(s, 0)) % 256
    ancestors = np.arange(len(s)) - age
    ancestors[age == 0] = -512
    ep = zig(s[:, :, [7, 8, 9, 2]]).reshape(-1, 8)
    while (mask := ancestors >= 0).any():
        parents = ancestors[mask]
        ep[mask] += ep[parents]
        ancestors[mask] = ancestors[parents]
    for slot in np.unique(ancestors):
        ep[ancestors == slot] += profile["endpoints"][str(slot)]
    ep %= CAP
    colors = np.asarray(profile["color"])[s[:, :, 10:12].reshape(-1, 4)]
    quadrants = (colors[:, [0, 2, 1, 3], None] >> (2 * np.arange(4))) & 3
    ci = quadrants.reshape(-1, 2, 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 16)
    alpha = np.asarray(profile["alpha"])[s[:, :, 3:7].reshape(-1, 8)]
    pairs = (alpha[:, :, None] >> (3 * np.arange(2))) & 7
    ai = pairs.reshape(-1, 16)[
        :, [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    ]
    rgb = (ep[:, ::4] << 11) | (ep[:, 1::4] << 5) | ep[:, 2::4]
    cw = ci.astype("uint64") @ (1 << (2 * np.arange(16, dtype="uint64")))
    aw = ai.astype("uint64") @ (1 << (3 * np.arange(16, dtype="uint64")))
    low = (aw << 16) | ep[:, 3].astype("uint64") | (ep[:, 7].astype("uint64") << 8)
    high = (cw << 32) | rgb[:, 0].astype("uint64") | (rgb[:, 1].astype("uint64") << 16)
    return np.column_stack((low, high)).astype("<u8").tobytes()


def export_file(pool, index, target, mips, force, profile):
    target = Path(target)
    with open(profile, encoding="utf-8") as source:
        decoder = LocalDecoder(pool, json.load(source))
    try:
        data = decoder.dds(index, mips)
    finally:
        decoder.read_chunk.cache_clear()
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.close()
            if force:
                os.replace(tmp, target)
            else:
                os.link(tmp, target)
        finally:
            f.close()
            tmp.unlink(missing_ok=True)
    return 0
