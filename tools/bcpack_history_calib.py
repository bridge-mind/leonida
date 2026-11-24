"""Local pure-Python BCPack initial-history calibration and validation tool.

Derives, merges, and validates local `--history` JSON tables for `tools/bcpack.py`
and `tools/ttpf_mips.py` strictly from the player's own TTPF pools and/or
supplied reference DDS/block pairs. No game data is embedded in or written by
this repository.

Two complementary calibration paths are supported (and may be combined):
1. Exact reference-pair inversion (`--ref-dds POOL:INDEX=REF.dds` or
   `--ref-pairs PAIRS.json`): algebraically inverts built-in history references
   against exact BCn reference blocks, producing **confirmed** entries
   (`class 3` with per-field `ConfirmedBits` for BC6H, `ConfirmedBits` for BC7,
   `confirmed` for BC1-BC5).
2. Pool self-calibration (from TTPF `.ttpf` files or directories): combines
   exact algebraic bitstream constraints (BC6H/BC7 anchor MSB constraints and
   BC7 Mode-1 shared p-bit parity), intra-block smoothness index voting, and
   sibling raw/zlib mip endpoint fitting on mixed-codec textures to produce
   **provisional** tables usable with `--history <path> --allow-provisional`.
"""
import argparse
import json
import os
import struct
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import bcpack  # noqa: E402
import bcpack_tables as _tables  # noqa: E402
import bcpack_tables_bptc as _bptc  # noqa: E402
import ttpf_mips as _ttpf  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass


class CliError(Exception):
    """User-facing CLI error reported cleanly without a traceback."""


ALL_CODECS = (1, 3, 4, 5, 6, 7)
DEFAULT_CODECS = (6, 7)
CODEC_KEYS = {1: "bc1", 3: "bc3", 4: "bc4", 5: "bc5", 6: "bc6h", 7: "bc7"}
CODEC_ALIASES = {
    "1": 1, "bc1": 1,
    "3": 3, "bc3": 3,
    "4": 4, "bc4": 4,
    "5": 5, "bc5": 5,
    "6": 6, "bc6": 6, "bc6h": 6,
    "7": 7, "bc7": 7,
}
CODEC_DXGI = {
    1: frozenset((70, 71, 72)),
    3: frozenset((76, 77, 78)),
    4: frozenset((79, 80, 81)),
    5: frozenset((82, 83, 84)),
    6: frozenset((94, 95, 96)),
    7: frozenset((97, 98, 99)),
}
BC_MODS = {
    1: (32, 64, 32, 32, 64, 32),
    3: (256, 256, 32, 64, 32, 32, 64, 32),
    4: (256, 256),
    5: (256, 256, 256, 256),
}
_BC6H_CODE_TO_MODE = {code: mode for mode, code in enumerate(bcpack._BC6H_CODE) if mode > 0}
_FOURCC_DXGI = {
    b"DXT1": 71,
    b"DXT5": 77,
    b"ATI1": 80,
    b"BC4U": 80,
    b"BC4S": 81,
    b"ATI2": 83,
    b"BC5U": 83,
    b"BC5S": 84,
}


def parse_codecs(spec):
    """Parse a comma-separated codec list or `'all'` into a deduplicated tuple of IDs."""
    if not spec or not spec.strip():
        raise CliError("--codec must not be empty")
    tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise CliError("--codec must not be empty")
    out = []
    for tok in tokens:
        if tok == "all":
            for cid in ALL_CODECS:
                if cid not in out:
                    out.append(cid)
            continue
        cid = CODEC_ALIASES.get(tok)
        if cid is None:
            valid = ", ".join(sorted(CODEC_KEYS.values()) + ["all"])
            raise CliError(f"unknown codec {tok!r} (expected one of: {valid})")
        if cid not in out:
            out.append(cid)
    return tuple(out)


# ---------------------------------------------------------------------------
# Fast prefix rANS lane decoder
# ---------------------------------------------------------------------------

def decode_lanes_prefix(raw, codec, out_size, max_sym):
    """Decode up to `max_sym` symbols per lane from a BCPack chunk.

    Because direct built-in history references (`d > i`) can only occur in
    blocks `0..255`, stopping the forward rANS scheduler after a short prefix
    avoids decoding tens of thousands of trailing blocks per chunk.
    """
    nsym = out_size // 8
    if max_sym <= 0 or max_sym >= nsym:
        return bcpack.decode_lanes(raw, codec, out_size)[0]
    widths = bcpack.LANE_WIDTHS[codec]
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = bcpack.RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [False] * nl
    full = [False] * nl
    held = [False] * nl
    vz = [0] * nl
    inbuf = [0] * nl
    sym = [0] * nl
    in_rem = [0] * nl
    p0 = [0] * nl
    p1 = [0] * nl
    model = [None] * nl
    slotfull = [False] * nl
    slotval = [0] * nl
    stall = [False] * nl
    out = [[] for _ in range(nl)]
    pos = 0
    nraw = len(raw)
    data = memoryview(raw)
    emitted = 0
    while emitted < max_sym:
        left = 8
        for i in range(nl):
            if done[i] or (held[i] and slotfull[i]):
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                if pos >= nraw:
                    raise bcpack.BcpackError("BCPack lane stream ended before prefix symbols finished")
                inbuf[i] = data[pos]
                pos += 1
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False
        for i in range(nl):
            if done[i] or stall[i]:
                continue
            c = ctrl[i]
            if c == HDR0:
                if not full[i]:
                    continue
                h = inbuf[i]
                q[i] = h & 15
                flush[i] = (h >> 4) & 1
                h3[i] = (h >> 5) & 1
                w[i] = widths[i] - (h >> 6)
                full[i] = False
                ctrl[i] = HDR1
                vz[i] = 0
            elif c == HDR1:
                if not full[i]:
                    continue
                full[i] = False
                model[i] = bcpack._get_model(w[i], q[i])
                if h3[i]:
                    in_rem[i] = inbuf[i]
                    ctrl[i] = HDR2
                else:
                    in_rem[i] = inbuf[i] + 1
                    ctrl[i] = PROC
            elif c == HDR2:
                if not full[i]:
                    continue
                in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257
                ctrl[i] = PROC
                full[i] = False
            else:
                st = p0[i]
                if st < L:
                    if in_rem[i] > 0:
                        if full[i]:
                            st = (st << 8) | inbuf[i]
                            in_rem[i] -= 1
                            full[i] = False
                            if in_rem[i] == 0 and flush[i]:
                                while st < L and st != 0 and vz[i] < 8:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        while st < L and st != 0 and vz[i] < 8:
                            st <<= 8
                            vz[i] += 1
                    else:
                        p1[i] = st
                        if out_rem[i] > 0:
                            ctrl[i] = HDR0
                        else:
                            done[i] = True
                        continue
                nxt = st > L and out_rem[i] > 0
                if nxt:
                    sym[i] = model[i].lookup(st & 0xFFFF)
                    out_rem[i] -= 1
                held[i] = nxt
                p1[i] = st
                if in_rem[i] == 0:
                    if flush[i]:
                        if st == L:
                            if out_rem[i] > 0:
                                ctrl[i] = HDR0
                                p0[i] = 0
                                p1[i] = 0
                            else:
                                done[i] = True
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
        ready = True
        anyfull = False
        for i in range(nl):
            if slotfull[i]:
                anyfull = True
            elif not (done[i] and not held[i]):
                ready = False
                break
        if ready and anyfull:
            for i in range(nl):
                if slotfull[i]:
                    out[i].append(slotval[i])
                    slotfull[i] = False
            emitted += 1
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False
    return out


# ---------------------------------------------------------------------------
# RDNA2 tile coordinate helpers and DDS reader
# ---------------------------------------------------------------------------

def _tile_pattern(layout):
    eb = layout["element_bytes"]
    pats = _ttpf.PATTERNS_4K if layout["tile"] == 4096 else _ttpf.PATTERNS_64K
    return pats[{1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]]


def tiled_offset(layout, mip, bx, by):
    """Return the byte offset of block `(bx, by)` inside `mip`'s tiled region."""
    pat = _tile_pattern(layout)
    bw, bh = layout["block_w"], layout["block_h"]
    off = ((by // bh) * (mip["padded_w"] // bw) + bx // bw) * layout["tile"]
    tx, ty = bx + mip["tile_x"], by + mip["tile_y"]
    for bit, s in enumerate(pat):
        if s > 0:
            off |= ((tx >> (s - 1)) & 1) << bit
        elif s < 0:
            off |= ((ty >> (-s - 1)) & 1) << bit
    return off


def make_inverse_tile_lookup(layout, mip):
    """Return a fast function mapping a mip-relative byte offset to `(bx, by)` or `None`."""
    pat = _tile_pattern(layout)
    tile = layout["tile"]
    bw, bh = layout["block_w"], layout["block_h"]
    tiles_x = mip["padded_w"] // bw
    tx0, ty0 = mip["tile_x"], mip["tile_y"]
    w, h = mip["blocks_w"], mip["blocks_h"]
    reg = mip["region"]
    x_bits = tuple((bit, 1 << (s - 1)) for bit, s in enumerate(pat) if s > 0)
    y_bits = tuple((bit, 1 << (-s - 1)) for bit, s in enumerate(pat) if s < 0)

    def lookup(rel_off):
        if rel_off < 0 or rel_off >= reg:
            return None
        tile_idx = rel_off // tile
        in_tile = rel_off % tile
        lx = 0
        for bit, val in x_bits:
            if (in_tile >> bit) & 1:
                lx |= val
        ly = 0
        for bit, val in y_bits:
            if (in_tile >> bit) & 1:
                ly |= val
        bx = (tile_idx % tiles_x) * bw + lx - tx0
        by = (tile_idx // tiles_x) * bh + ly - ty0
        if 0 <= bx < w and 0 <= by < h:
            return bx, by
        return None

    return lookup


def retile_mip(linear, layout, mip):
    """Place a linear DDS mip payload into its RDNA2 tiled byte buffer."""
    out = bytearray(mip["region"])
    _retile_mip_into(linear, layout, mip, out)
    return bytes(out)


def _retile_mip_into(linear: bytes, layout: dict, mip: dict, out: bytearray,
                     known: bytearray | None = None) -> None:
    """Write one mip's blocks into its region, which may also hold other tail mips."""
    eb = layout["element_bytes"]
    bw, bh = mip["blocks_w"], mip["blocks_h"]
    expected = bw * bh * eb
    if len(linear) < expected:
        raise ValueError(f"DDS mip payload truncated ({len(linear)} < {expected} bytes)")
    mv = memoryview(linear)
    row_stride = bw * eb
    for by in range(bh):
        row_at = by * row_stride
        for bx in range(bw):
            off = tiled_offset(layout, mip, bx, by)
            src = row_at + bx * eb
            out[off:off + eb] = mv[src:src + eb]
            if known is not None:
                known[off:off + eb] = b"\x01" * eb


def read_dds(path):
    """Read a DX9 or DX10 2D BCn DDS file into `(dxgi, width, height, mips_list)`."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError(f"{path}: not a valid DDS file")
    hdr_size = struct.unpack_from("<I", data, 4)[0]
    if hdr_size != 124:
        raise ValueError(f"{path}: unexpected DDS header size {hdr_size}")
    height, width = struct.unpack_from("<II", data, 12)
    mip_count = struct.unpack_from("<I", data, 28)[0] or 1
    pf_flags = struct.unpack_from("<I", data, 80)[0]
    fourcc = data[84:88]
    off = 128
    if (pf_flags & 0x4) and fourcc == b"DX10":
        if len(data) < 148:
            raise ValueError(f"{path}: truncated DX10 DDS header")
        dxgi, dim, _misc, arr = struct.unpack_from("<IIII", data, 128)
        if dim != 3 or arr != 1:
            raise ValueError(f"{path}: only single-slice 2D DDS references are supported")
        off = 148
    elif pf_flags & 0x4:
        dxgi = _FOURCC_DXGI.get(fourcc)
        if dxgi is None:
            raise ValueError(f"{path}: unsupported DDS FourCC {fourcc!r}")
    else:
        raise ValueError(f"{path}: uncompressed DX9 DDS is not a BCn reference")
    eb = _ttpf.ELEMENT_BYTES.get(dxgi)
    if eb is None or dxgi not in _ttpf.BC_FORMATS:
        raise ValueError(f"{path}: unsupported DXGI format {dxgi}")
    mips = []
    mw, mh = width, height
    for m in range(mip_count):
        bw = max(1, (mw + 3) // 4)
        bh = max(1, (mh + 3) // 4)
        sz = bw * bh * eb
        if off + sz > len(data):
            raise ValueError(f"{path}: truncated payload at mip {m}")
        mips.append(data[off:off + sz])
        off += sz
        mw = max(1, mw >> 1)
        mh = max(1, mh >> 1)
    return dxgi, width, height, mips


# ---------------------------------------------------------------------------
# Pure-Python BC1 / BC3 / BC4 / BC5 / BC6H / BC7 block decoders (for sibling mips)
# ---------------------------------------------------------------------------

def _rgb565_channels(c):
    r = ((c >> 11) & 31) * 255.0 / 31.0
    g = ((c >> 5) & 63) * 255.0 / 63.0
    b = (c & 31) * 255.0 / 31.0
    return r, g, b


def decode_bc1_block_rgba(data, at):
    c0, c1, idx = struct.unpack_from("<HHI", data, at)
    r0, g0, b0 = _rgb565_channels(c0)
    r1, g1, b1 = _rgb565_channels(c1)
    if c0 > c1:
        pal = (
            (r0, g0, b0, 255.0),
            (r1, g1, b1, 255.0),
            ((2.0 * r0 + r1) / 3.0, (2.0 * g0 + g1) / 3.0, (2.0 * b0 + b1) / 3.0, 255.0),
            ((r0 + 2.0 * r1) / 3.0, (g0 + 2.0 * g1) / 3.0, (b0 + 2.0 * b1) / 3.0, 255.0),
        )
    else:
        pal = (
            (r0, g0, b0, 255.0),
            (r1, g1, b1, 255.0),
            (0.5 * (r0 + r1), 0.5 * (g0 + g1), 0.5 * (b0 + b1), 255.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    return [pal[(idx >> (2 * p)) & 3] for p in range(16)]


def decode_bc4_block_u8(data, at):
    a0, a1 = data[at], data[at + 1]
    bits = int.from_bytes(data[at + 2:at + 8], "little")
    if a0 > a1:
        pal = (
            float(a0), float(a1),
            (6.0 * a0 + 1.0 * a1) / 7.0, (5.0 * a0 + 2.0 * a1) / 7.0,
            (4.0 * a0 + 3.0 * a1) / 7.0, (3.0 * a0 + 4.0 * a1) / 7.0,
            (2.0 * a0 + 5.0 * a1) / 7.0, (1.0 * a0 + 6.0 * a1) / 7.0,
        )
    else:
        pal = (
            float(a0), float(a1),
            (4.0 * a0 + 1.0 * a1) / 5.0, (3.0 * a0 + 2.0 * a1) / 5.0,
            (2.0 * a0 + 3.0 * a1) / 5.0, (1.0 * a0 + 4.0 * a1) / 5.0,
            0.0, 255.0,
        )
    return [pal[(bits >> (3 * p)) & 7] for p in range(16)]


def decode_bc3_block_rgba(data, at):
    al = decode_bc4_block_u8(data, at)
    c0, c1, idx = struct.unpack_from("<HHI", data, at + 8)
    r0, g0, b0 = _rgb565_channels(c0)
    r1, g1, b1 = _rgb565_channels(c1)
    pal = (
        (r0, g0, b0),
        (r1, g1, b1),
        ((2.0 * r0 + r1) / 3.0, (2.0 * g0 + g1) / 3.0, (2.0 * b0 + b1) / 3.0),
        ((r0 + 2.0 * r1) / 3.0, (g0 + 2.0 * g1) / 3.0, (b0 + 2.0 * b1) / 3.0),
    )
    out = [None] * 16
    for p in range(16):
        r, g, b = pal[(idx >> (2 * p)) & 3]
        out[p] = (r, g, b, al[p])
    return out


def decode_bc5_block_rg(data, at):
    rs = decode_bc4_block_u8(data, at)
    gs = decode_bc4_block_u8(data, at + 8)
    return [(rs[p], gs[p]) for p in range(16)]


def _sign_extend(val, bits):
    sign = 1 << (bits - 1)
    return (val ^ sign) - sign


def _bc6h_unquantize(val, bits, signed=False):
    if not signed:
        if bits >= 15:
            return val
        if val == 0:
            return 0
        if val == (1 << bits) - 1:
            return 0xFFFF
        return ((val << 16) + 0x8000) >> bits
    neg = val < 0
    v = -val if neg else val
    if bits >= 16:
        out = v
    elif v == 0:
        out = 0
    elif v >= (1 << (bits - 1)) - 1:
        out = 0x7FFF
    else:
        out = ((v << 15) + 0x4000) >> (bits - 1)
    return -out if neg else out


def unpack_bc6h_block(data, at):
    """Unpack a 16-byte BC6H block into `(mode, part, f_wxyz, ix_block)` or `None`."""
    acc = int.from_bytes(data[at:at + 16], "little")
    m2 = acc & 3
    code = m2 if m2 < 2 else (acc & 0x1F)
    mode = _BC6H_CODE_TO_MODE.get(code)
    if mode is None:
        return None
    _trans, _base, _deltas, reg = bcpack._BC6H_INFO[code]
    plan, header = bcpack._bc6h_plan(code)
    f = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    part = 0
    for c, q, bit_idx, pos in plan:
        if (acc >> pos) & 1:
            if c < 0:
                part |= 1 << bit_idx
            else:
                f[c][q] |= 1 << bit_idx
    for c in range(3):
        f[c][0] = bcpack._bc6h_stream_w(mode, f[c][0])
    ibits = 3 if reg == 2 else 4
    second = _bptc.BC7_ANCHOR2[part] if reg == 2 else -1
    pos = header
    ix = [0] * 16
    for p in range(16):
        nb = (ibits - 1) if (p == 0 or p == second) else ibits
        ix[p] = (acc >> pos) & ((1 << nb) - 1)
        pos += nb
    return mode, part, f, ix


def decode_bc6h_block_unq(data, at, signed=False):
    """Decode one 16-byte BC6H block into 16 `(r, g, b)` tuples in 16-bit unquantized space."""
    parsed = unpack_bc6h_block(data, at)
    if parsed is None:
        return [(0.0, 0.0, 0.0)] * 16
    mode, part, f, ix = parsed
    code = bcpack._BC6H_CODE[mode]
    trans, base, deltas, reg = bcpack._BC6H_INFO[code]
    mw = (1 << base) - 1
    ep = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]
    for c in range(3):
        w, x, y, z = f[c]
        w = bcpack._bc6h_stream_w(mode, w)
        dc = deltas[c]
        if signed:
            w = _sign_extend(w, base)
        if trans:
            e0 = w
            e1 = (w + _sign_extend(x, dc)) & mw
            if signed:
                e1 = _sign_extend(e1, base)
            ep[c][0] = float(_bc6h_unquantize(e0, base, signed))
            ep[c][1] = float(_bc6h_unquantize(e1, base, signed))
            if reg == 2:
                e2 = (w + _sign_extend(y, dc)) & mw
                e3 = (w + _sign_extend(z, dc)) & mw
                if signed:
                    e2 = _sign_extend(e2, base)
                    e3 = _sign_extend(e3, base)
                ep[c][2] = float(_bc6h_unquantize(e2, base, signed))
                ep[c][3] = float(_bc6h_unquantize(e3, base, signed))
        else:
            if signed:
                x = _sign_extend(x, dc)
                y = _sign_extend(y, dc)
                z = _sign_extend(z, dc)
            ep[c][0] = float(_bc6h_unquantize(w, base, signed))
            ep[c][1] = float(_bc6h_unquantize(x, dc, signed))
            if reg == 2:
                ep[c][2] = float(_bc6h_unquantize(y, dc, signed))
                ep[c][3] = float(_bc6h_unquantize(z, dc, signed))
    wt = _ttpf._BPTC_WEIGHTS[3 if reg == 2 else 4]
    sm = _bptc.load_bc7()["partition2"][part] if reg == 2 else None
    out = [None] * 16
    for p in range(16):
        s2 = 0 if sm is None else (sm[p] * 2)
        w1 = wt[ix[p]]
        w0 = 64 - w1
        out[p] = (
            (w0 * ep[0][s2] + w1 * ep[0][s2 + 1]) / 64.0,
            (w0 * ep[1][s2] + w1 * ep[1][s2 + 1]) / 64.0,
            (w0 * ep[2][s2] + w1 * ep[2][s2 + 1]) / 64.0,
        )
    return out


def unpack_bc7_block(data, at):
    """Unpack one 16-byte BC7 block into `(mode, part, rot, isel, val_8bit, pri, sec)` or `None`.

    `val_8bit[sub][e][c]` is left-aligned 8-bit in display RGBA order (matching
    `val` inside `bcpack._bc7_rebuild` before channel rotation is applied).
    """
    b0 = data[at]
    if b0 == 0:
        return None
    mode = (b0 & -b0).bit_length() - 1
    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = _ttpf._BC7_MODES[mode]
    blk = int.from_bytes(data[at:at + 16], "little")
    pos = mode + 1
    part = (blk >> pos) & ((1 << pb) - 1) if pb else 0
    pos += pb
    rot = (blk >> pos) & ((1 << rb) - 1) if rb else 0
    pos += rb
    isel = (blk >> pos) & 1 if isb else 0
    pos += isb
    n_ep = 2 * ns
    cb_mask = (1 << cb) - 1
    raw_c = [[0] * n_ep for _ in range(4)]
    for ch in range(3):
        row = raw_c[ch]
        for k in range(n_ep):
            row[k] = (blk >> pos) & cb_mask
            pos += cb
    if ab:
        ab_mask = (1 << ab) - 1
        row = raw_c[3]
        for k in range(n_ep):
            row[k] = (blk >> pos) & ab_mask
            pos += ab
    pbits = None
    if epb:
        pbits = [(blk >> (pos + k)) & 1 for k in range(n_ep)]
        pos += n_ep
    elif spb:
        sp = [(blk >> (pos + s)) & 1 for s in range(ns)]
        pos += ns
        pbits = [sp[k >> 1] for k in range(n_ep)]

    # Build left-aligned 8-bit values in pre-rotation stream order first.
    val = [[[0, 0, 0, 0], [0, 0, 0, 0]] for _ in range(ns)]
    for sub in range(ns):
        for e in range(2):
            k = sub * 2 + e
            for ch in range(3):
                if pbits is not None:
                    v = ((raw_c[ch][k] << 1) | pbits[k]) << (8 - cb - 1)
                else:
                    v = raw_c[ch][k] << (8 - cb)
                val[sub][e][ch] = v
            if ab:
                if pbits is not None and epb:
                    v = ((raw_c[3][k] << 1) | pbits[k]) << (8 - ab - 1)
                else:
                    v = raw_c[3][k] << (8 - ab)
                val[sub][e][3] = v
            if rot:
                # Undo channel rotation so val[sub][e][0..3] matches display RGBA
                # (same channel order stored in BCPack's BC7 endpoint history).
                val[sub][e][rot - 1], val[sub][e][3] = val[sub][e][3], val[sub][e][rot - 1]

    tab = _bptc.load_bc7()
    if ns == 1:
        sm = None
        anchors = (0,)
    elif ns == 2:
        sm = tab["partition2"][part]
        anchors = (0, tab["anchor2"][part])
    else:
        sm = tab["partition3"][part]
        anchors = (0, tab["anchor3a"][part], tab["anchor3b"][part])

    ib_mask = (1 << ib) - 1
    ib_amask = (1 << (ib - 1)) - 1
    idx1 = [0] * 16
    for p in range(16):
        sub = 0 if sm is None else sm[p]
        if p == anchors[sub]:
            idx1[p] = (blk >> pos) & ib_amask
            pos += ib - 1
        else:
            idx1[p] = (blk >> pos) & ib_mask
            pos += ib

    idx2 = None
    if ib2:
        ib2_mask = (1 << ib2) - 1
        ib2_amask = (1 << (ib2 - 1)) - 1
        idx2 = [0] * 16
        for p in range(16):
            if p == 0:
                idx2[p] = (blk >> pos) & ib2_amask
                pos += ib2 - 1
            else:
                idx2[p] = (blk >> pos) & ib2_mask
                pos += ib2

    # Map D3D11 (idx1, idx2) to BCPack's (s1=rows1, s2=rows2): in Mode 4,
    # D3D11 writes the 2-bit set (`s2`) first and the 3-bit set (`s1`) second.
    s1, s2 = (idx2, idx1) if mode == 4 else (idx1, idx2)
    return mode, part, rot, isel, val, s1, s2


def decode_bc7_block_rgba(data, at):
    """Decode one 16-byte BC7 block into 16 `(r, g, b, a)` floats in `0..255`."""
    b0 = data[at]
    if b0 == 0:
        return [(0.0, 0.0, 0.0, 0.0)] * 16
    mode = (b0 & -b0).bit_length() - 1
    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = _ttpf._BC7_MODES[mode]
    blk = int.from_bytes(data[at:at + 16], "little")
    pos = mode + 1
    part = (blk >> pos) & ((1 << pb) - 1) if pb else 0
    pos += pb
    rot = (blk >> pos) & ((1 << rb) - 1) if rb else 0
    pos += rb
    isel = (blk >> pos) & 1 if isb else 0
    pos += isb
    n_ep = 2 * ns
    cb_mask = (1 << cb) - 1
    raw_c = [[0] * n_ep for _ in range(4)]
    for ch in range(3):
        row = raw_c[ch]
        for k in range(n_ep):
            row[k] = (blk >> pos) & cb_mask
            pos += cb
    if ab:
        ab_mask = (1 << ab) - 1
        row = raw_c[3]
        for k in range(n_ep):
            row[k] = (blk >> pos) & ab_mask
            pos += ab
    c_bits = cb
    a_bits = ab
    if epb:
        pbits = [(blk >> (pos + k)) & 1 for k in range(n_ep)]
        pos += n_ep
        c_bits += 1
        if ab:
            a_bits += 1
    elif spb:
        sp = [(blk >> (pos + s)) & 1 for s in range(ns)]
        pos += ns
        pbits = [sp[k >> 1] for k in range(n_ep)]
        c_bits += 1
    else:
        pbits = None
    c_sh = 8 - c_bits
    c_rep = 2 * c_bits - 8
    ep = [[255] * n_ep for _ in range(4)]
    for ch in range(3):
        src = raw_c[ch]
        dst = ep[ch]
        for k in range(n_ep):
            v = (src[k] << 1) | pbits[k] if pbits is not None else src[k]
            dst[k] = (v << c_sh) | (v >> c_rep)
    if ab:
        a_sh = 8 - a_bits
        a_rep = 2 * a_bits - 8
        src = raw_c[3]
        dst = ep[3]
        for k in range(n_ep):
            v = (src[k] << 1) | pbits[k] if (pbits is not None and epb) else src[k]
            dst[k] = (v << a_sh) | (v >> a_rep)
    tab = _bptc.load_bc7()
    if ns == 1:
        sm = None
        anchors = (0,)
    elif ns == 2:
        sm = tab["partition2"][part]
        anchors = (0, tab["anchor2"][part])
    else:
        sm = tab["partition3"][part]
        anchors = (0, tab["anchor3a"][part], tab["anchor3b"][part])
    ib_mask = (1 << ib) - 1
    ib_amask = (1 << (ib - 1)) - 1
    idx1 = [0] * 16
    for p in range(16):
        sub = 0 if sm is None else sm[p]
        if p == anchors[sub]:
            idx1[p] = (blk >> pos) & ib_amask
            pos += ib - 1
        else:
            idx1[p] = (blk >> pos) & ib_mask
            pos += ib
    if ib2:
        ib2_mask = (1 << ib2) - 1
        ib2_amask = (1 << (ib2 - 1)) - 1
        idx2 = [0] * 16
        for p in range(16):
            if p == 0:
                idx2[p] = (blk >> pos) & ib2_amask
                pos += ib2 - 1
            else:
                idx2[p] = (blk >> pos) & ib2_mask
                pos += ib2
        if isel:
            c_idx, c_wt = idx2, _ttpf._BPTC_WEIGHTS[ib2]
            a_idx, a_wt = idx1, _ttpf._BPTC_WEIGHTS[ib]
        else:
            c_idx, c_wt = idx1, _ttpf._BPTC_WEIGHTS[ib]
            a_idx, a_wt = idx2, _ttpf._BPTC_WEIGHTS[ib2]
    else:
        c_idx = a_idx = idx1
        c_wt = a_wt = _ttpf._BPTC_WEIGHTS[ib]
    er, eg, eb_, ea = ep
    out = [None] * 16
    for p in range(16):
        s2 = 0 if sm is None else sm[p] * 2
        wc = c_wt[c_idx[p]]
        iwc = 64 - wc
        r = (iwc * er[s2] + wc * er[s2 + 1] + 32) >> 6
        g = (iwc * eg[s2] + wc * eg[s2 + 1] + 32) >> 6
        b = (iwc * eb_[s2] + wc * eb_[s2 + 1] + 32) >> 6
        if ab:
            wa = a_wt[a_idx[p]]
            a = ((64 - wa) * ea[s2] + wa * ea[s2 + 1] + 32) >> 6
        else:
            a = 255
        if rot == 1:
            r, a = a, r
        elif rot == 2:
            g, a = a, g
        elif rot == 3:
            b, a = a, b
        out[p] = (float(r), float(g), float(b), float(a))
    return out


# ---------------------------------------------------------------------------
# Bit-precision constraint helpers
# ---------------------------------------------------------------------------

def _merge_top_bits(cur_bits, cur_val, new_bits, new_val, total_bits):
    """Merge two left-aligned top-bit constraints, returning `(bits, val, conflict)`."""
    if cur_bits == 0:
        return new_bits, new_val, False
    overlap = min(cur_bits, new_bits)
    shift = total_bits - overlap
    if (cur_val >> shift) != (new_val >> shift):
        return cur_bits, cur_val, True
    if new_bits > cur_bits:
        return new_bits, new_val, False
    return cur_bits, cur_val, False


def _lead_bits_4(mask):
    """Return `(left_aligned_val, exact_bits)` shared by all set bits in a 16-bit mask."""
    if mask == 0:
        return 0, 0
    surv = [h for h in range(16) if (mask >> h) & 1]
    diff = 0
    for h in surv[1:]:
        diff |= h ^ surv[0]
    bits = 4 - diff.bit_length()
    val = surv[0] & (0xF << (4 - bits)) if bits else 0
    return val, bits


def _eval_bc7_ops(ops, h):
    v = h
    for bits, dz in ops:
        sh = 8 - bits
        m = (1 << bits) - 1
        v = (((v >> sh) + dz) & m) << sh
    return v


# ---------------------------------------------------------------------------
# Per-codec calibrators
# ---------------------------------------------------------------------------

class BcSimpleCalibrator:
    """Calibrator for BC1 (1), BC3 (3), BC4 (4), and BC5 (5) endpoint history tables."""

    def __init__(self, codec):
        self.codec = codec
        self.mods = BC_MODS[codec]
        self.ncols = len(self.mods)
        self.confirmed = {}
        self.conflicts = 0
        self.ref_blocks = 0
        self.pool_chunks = 0
        self.pool_textures = 0
        # obs[(entry, pair_idx)] -> list of (dz0, dz1, q_targets_4, q_weights_4, weight)
        self.obs = {}

    def _lane_step(self, lanes, i):
        if self.codec == 1:
            v = lanes[0][i]
            d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
            dz = tuple(bcpack._unzig(lanes[2 + c][i]) for c in range(6))
            return d, dz
        if self.codec == 3:
            a, b = 2 * i, 2 * i + 1
            d = bcpack._unzig(bcpack._inter8(lanes[0][a], lanes[0][b])) & 255
            dz = (
                bcpack._unzig(lanes[2][a]), bcpack._unzig(lanes[2][b]),
                bcpack._unzig(lanes[7][a]), bcpack._unzig(lanes[8][a]), bcpack._unzig(lanes[9][a]),
                bcpack._unzig(lanes[7][b]), bcpack._unzig(lanes[8][b]), bcpack._unzig(lanes[9][b]),
            )
            return d, dz
        if self.codec == 4:
            v = lanes[0][i]
            d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
            dz = (bcpack._unzig(lanes[2][i]), bcpack._unzig(lanes[3][i]))
            return d, dz
        # codec 5
        a, b = 2 * i, 2 * i + 1
        d = bcpack._unzig(bcpack._inter8(lanes[0][a], lanes[0][b])) & 255
        dz = (
            bcpack._unzig(lanes[2][a]), bcpack._unzig(lanes[2][b]),
            bcpack._unzig(lanes[3][a]), bcpack._unzig(lanes[3][b]),
        )
        return d, dz

    def _extract_ref_endpoints(self, ref_blocks, i):
        eb = bcpack.BLOCK_BYTES[self.codec]
        at = i * eb
        if self.codec == 1:
            c0, c1 = struct.unpack_from("<HH", ref_blocks, at)
            return ((c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31, (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31)
        if self.codec == 3:
            a0, a1 = ref_blocks[at], ref_blocks[at + 1]
            c0, c1 = struct.unpack_from("<HH", ref_blocks, at + 8)
            return (a0, a1, (c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31, (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31)
        if self.codec == 4:
            return (ref_blocks[at], ref_blocks[at + 1])
        return (ref_blocks[at], ref_blocks[at + 1], ref_blocks[at + 8], ref_blocks[at + 9])

    def add_reference_chunk(self, lanes, ref_blocks, nblocks, *, valid_blocks=None):
        """Algebraically invert direct built-in references from exact BCn blocks."""
        mods = self.mods
        for i in range(min(nblocks, 256)):
            if valid_blocks is not None and not valid_blocks[i]:
                continue
            d, dz = self._lane_step(lanes, i)
            if d == 0 or d <= i:
                continue
            entry = 256 + i - d
            ep = self._extract_ref_endpoints(ref_blocks, i)
            solved = tuple((ep[k] - dz[k]) % mods[k] for k in range(self.ncols))
            prev = self.confirmed.get(entry)
            if prev is None:
                self.confirmed[entry] = solved
            elif prev != solved:
                self.conflicts += 1
            self.ref_blocks += 1

    def _quadrant_weights(self, lanes, i):
        """Return per-channel 4-quadrant average interpolation weight in `[0, 1]`."""
        if self.codec == 1:
            quad = _tables.BC1_QUADRANT
            qvals = (quad[lanes[8][i]], quad[lanes[9][i]], quad[lanes[10][i]], quad[lanes[11][i]])
            w_lut = (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0)
            qw = tuple(
                sum(w_lut[(q >> (2 * p)) & 3] for p in range(4)) / 4.0
                for q in qvals
            )
            return (qw, qw, qw)
        if self.codec == 3:
            a, b = 2 * i, 2 * i + 1
            pairs = _tables.BC3_ALPHA_PAIR
            a_lut = (0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0)
            aq = [0.0, 0.0, 0.0, 0.0]
            for r in range(4):
                left = pairs[lanes[3 + r][a]]
                right = pairs[lanes[3 + r][b]]
                qi_l = (r // 2) * 2
                aq[qi_l] += a_lut[left[0]] + a_lut[left[1]]
                aq[qi_l + 1] += a_lut[right[0]] + a_lut[right[1]]
            aq_t = tuple(v / 4.0 for v in aq)
            quads = _tables.BC3_COLOUR_QUADRANT
            qvals = (quads[lanes[10][a]], quads[lanes[10][b]], quads[lanes[11][a]], quads[lanes[11][b]])
            w_lut = (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0)
            cw = tuple(
                sum(w_lut[(q >> (2 * p)) & 3] for p in range(4)) / 4.0
                for q in qvals
            )
            return (aq_t, cw, cw, cw)
        if self.codec == 4:
            pairs = _tables.BC4_PAIR
            a_lut = (0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0)
            aq = [0.0, 0.0, 0.0, 0.0]
            for r in range(4):
                left = pairs[lanes[4 + 2 * r][i]]
                right = pairs[lanes[4 + 2 * r + 1][i]]
                qi_l = (r // 2) * 2
                aq[qi_l] += a_lut[left[0]] + a_lut[left[1]]
                aq[qi_l + 1] += a_lut[right[0]] + a_lut[right[1]]
            return (tuple(v / 4.0 for v in aq),)
        # codec 5
        a, b = 2 * i, 2 * i + 1
        pairs = _tables.BC5_PAIR
        a_lut = (0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0)
        out = []
        for ch in range(2):
            aq = [0.0, 0.0, 0.0, 0.0]
            for r in range(4):
                left = pairs[lanes[4 + 4 * ch + r][a]]
                right = pairs[lanes[4 + 4 * ch + r][b]]
                qi_l = (r // 2) * 2
                aq[qi_l] += a_lut[left[0]] + a_lut[left[1]]
                aq[qi_l + 1] += a_lut[right[0]] + a_lut[right[1]]
            out.append(tuple(v / 4.0 for v in aq))
        return tuple(out)

    def add_pool_chunk(self, lanes, nblocks, inv_lookup, chunk_rel_start, get_quads):
        """Accumulate sibling-mip observations for built-in history chains."""
        self.pool_chunks += 1
        eb = bcpack.BLOCK_BYTES[self.codec]
        mods = self.mods
        roots = [-1] * nblocks
        deltas = [None] * nblocks
        for i in range(nblocks):
            d, dz = self._lane_step(lanes, i)
            if d == 0:
                continue
            if d > i:
                root = 256 + i - d
                acc = tuple(dz[k] % mods[k] for k in range(self.ncols))
            else:
                root = roots[i - d]
                if root < 0:
                    continue
                prev = deltas[i - d]
                acc = tuple((prev[k] + dz[k]) % mods[k] for k in range(self.ncols))
            roots[i] = root
            deltas[i] = acc
            if root in self.confirmed:
                continue
            xy = inv_lookup(chunk_rel_start + i * eb)
            if xy is None:
                continue
            qpts = get_quads(xy[0], xy[1])
            qw = self._quadrant_weights(lanes, i)
            if self.codec == 1:
                spread = max(max(p[c] for p in qpts) - min(p[c] for p in qpts) for c in range(3))
                wt = 4.0 if spread <= 8.0 else (2.0 if spread <= 24.0 else 0.75)
                scales = (31.0 / 255.0, 63.0 / 255.0, 31.0 / 255.0)
                for c in range(3):
                    lst = self.obs.setdefault((root, c), [])
                    if len(lst) < 64:
                        tgts = tuple(p[c] * scales[c] for p in qpts)
                        lst.append((acc[c], acc[c + 3], tgts, qw[c], wt))
            elif self.codec == 3:
                spread = max(max(p[c] for p in qpts) - min(p[c] for p in qpts) for c in range(4))
                wt = 4.0 if spread <= 8.0 else (2.0 if spread <= 24.0 else 0.75)
                # pair 0 = alpha (p[3]), pairs 1..3 = RGB565 (p[0..2])
                lst_a = self.obs.setdefault((root, 0), [])
                if len(lst_a) < 64:
                    lst_a.append((acc[0], acc[1], tuple(p[3] for p in qpts), qw[0], wt))
                scales = (31.0 / 255.0, 63.0 / 255.0, 31.0 / 255.0)
                for c in range(3):
                    lst = self.obs.setdefault((root, 1 + c), [])
                    if len(lst) < 64:
                        tgts = tuple(p[c] * scales[c] for p in qpts)
                        lst.append((acc[2 + c], acc[5 + c], tgts, qw[1 + c], wt))
            elif self.codec == 4:
                spread = max(qpts) - min(qpts)
                wt = 4.0 if spread <= 8.0 else (2.0 if spread <= 24.0 else 0.75)
                lst = self.obs.setdefault((root, 0), [])
                if len(lst) < 64:
                    lst.append((acc[0], acc[1], qpts, qw[0], wt))
            else:
                spread = max(max(p[c] for p in qpts) - min(p[c] for p in qpts) for c in range(2))
                wt = 4.0 if spread <= 8.0 else (2.0 if spread <= 24.0 else 0.75)
                for c in range(2):
                    lst = self.obs.setdefault((root, c), [])
                    if len(lst) < 64:
                        lst.append((acc[2 * c], acc[2 * c + 1], tuple(p[c] for p in qpts), qw[c], wt))

    @staticmethod
    def _fit_pair(obs_list, mod):
        cap = float(mod) * 0.25
        if mod <= 64:
            best_err = 1e30
            best = (0, 0)
            for h0 in range(mod):
                for h1 in range(mod):
                    err = 0.0
                    for d0, d1, tgts, qw, wt in obs_list:
                        e0 = (h0 + d0) % mod
                        e1 = (h1 + d1) % mod
                        de = e1 - e0
                        for q in range(4):
                            d = abs((e0 + qw[q] * de) - tgts[q])
                            err += wt * (d if d < cap else cap)
                    if err < best_err:
                        best_err = err
                        best = (h0, h1)
            return best
        # Coarse-to-fine search for 8-bit channels (0..255)
        cands = []
        for h0 in range(0, mod, 16):
            for h1 in range(0, mod, 16):
                err = 0.0
                for d0, d1, tgts, qw, wt in obs_list:
                    e0 = (h0 + 8 + d0) & 255
                    e1 = (h1 + 8 + d1) & 255
                    de = e1 - e0
                    for q in range(4):
                        d = abs((e0 + qw[q] * de) - tgts[q])
                        err += wt * (d if d < cap else cap)
                cands.append((err, h0, h1))
        cands.sort()
        top = cands[:4]
        for step in (8, 4, 2, 1):
            nxt = {}
            mid = step // 2
            for _, b0, b1 in top:
                for o0 in (0, step):
                    for o1 in (0, step):
                        h0, h1 = (b0 + o0) & 255, (b1 + o1) & 255
                        key = (h0, h1)
                        if key in nxt:
                            continue
                        err = 0.0
                        for d0, d1, tgts, qw, wt in obs_list:
                            e0 = (h0 + mid + d0) & 255
                            e1 = (h1 + mid + d1) & 255
                            de = e1 - e0
                            for q in range(4):
                                d = abs((e0 + qw[q] * de) - tgts[q])
                                err += wt * (d if d < cap else cap)
                        nxt[key] = err
            top = sorted((e, a, b) for (a, b), e in nxt.items())[:4]
        return top[0][1], top[0][2]

    def build_document(self, replace=False):
        provisional = {}
        entries = sorted({k for (k, _) in self.obs} - set(self.confirmed))
        for k in entries:
            if self.codec == 1:
                if not all((k, c) in self.obs for c in range(3)):
                    continue
                r0, r1 = self._fit_pair(self.obs[(k, 0)], 32)
                g0, g1 = self._fit_pair(self.obs[(k, 1)], 64)
                b0, b1 = self._fit_pair(self.obs[(k, 2)], 32)
                provisional[k] = [r0, g0, b0, r1, g1, b1]
            elif self.codec == 3:
                if not all((k, c) in self.obs for c in range(4)):
                    continue
                a0, a1 = self._fit_pair(self.obs[(k, 0)], 256)
                r0, r1 = self._fit_pair(self.obs[(k, 1)], 32)
                g0, g1 = self._fit_pair(self.obs[(k, 2)], 64)
                b0, b1 = self._fit_pair(self.obs[(k, 3)], 32)
                provisional[k] = [a0, a1, r0, g0, b0, r1, g1, b1]
            elif self.codec == 4:
                a0, a1 = self._fit_pair(self.obs[(k, 0)], 256)
                provisional[k] = [a0, a1]
            else:
                if not all((k, c) in self.obs for c in range(2)):
                    continue
                r0, r1 = self._fit_pair(self.obs[(k, 0)], 256)
                g0, g1 = self._fit_pair(self.obs[(k, 1)], 256)
                provisional[k] = [r0, r1, g0, g1]

        doc = {
            "codec": CODEC_KEYS[self.codec],
            "confirmed": {str(k): list(self.confirmed[k]) for k in sorted(self.confirmed)},
            "provisional": {str(k): list(provisional[k]) for k in sorted(provisional)},
        }
        if replace:
            doc["replace"] = True
        stats = {
            "confirmed_entries": len(self.confirmed),
            "provisional_entries": len(provisional),
            "reference_blocks": self.ref_blocks,
            "reference_conflicts": self.conflicts,
            "pool_chunks": self.pool_chunks,
            "pool_textures": self.pool_textures,
        }
        return doc, stats


class Bc6hCalibrator:
    """Calibrator for BC6H (`codec 6`) endpoint and index initial history tables."""

    def __init__(self):
        # Exact reference-pair constraints:
        # ref_ep[k][c][q] = [bits, val] (left-aligned to (16, 10, 6, 6))
        self.ref_ep = [[[[0, 0] for _ in range(4)] for _ in range(3)] for _ in range(256)]
        self.ref_ix = [[[0, 0] for _ in range(16)] for _ in range(256)]
        self.ref_ep_seen = [False] * 256
        self.ref_ix_seen = [False] * 256
        self.conflicts = 0
        self.ref_blocks = 0
        self.pool_chunks = 0
        self.pool_textures = 0

        # Algebraic constraints & pool observations:
        self.valid_ix = [[0xFFFF] * 16 for _ in range(256)]
        self.ix_obs = [[0] * 16 for _ in range(256)]
        self.ix_rel_votes = {(k, p): [0] * 16 for k in range(256) for p in range(16)}
        # Endpoint observations:
        # w_obs[(k, c)] -> list of (ops_w, tgt_unq, wt) where ops_w is ((base, dz0), ...)
        self.w_obs = {}
        # d_obs[(k, c, q)] -> list of (dc, dz) for transformed-mode centering (q in 1, 2, 3)
        self.d_obs = {}
        self.ep_seen = [False] * 256

    def add_reference_chunk(self, lanes, ref_blocks, nblocks, *, valid_blocks=None):
        """Algebraically invert direct built-in BC6H history references from exact blocks."""
        l0, l1, l15 = lanes[0], lanes[1], lanes[15]
        deint0, deint1 = bcpack._DEINT
        slot_widths = (16, 10, 6, 6)
        for i in range(min(nblocks, 256)):
            if valid_blocks is not None and not valid_blocks[i]:
                continue
            a, b = 2 * i, 2 * i + 1
            m_sym = l15[a]
            if m_sym > 13:
                continue
            parsed = unpack_bc6h_block(ref_blocks, i * 16)
            if parsed is None:
                continue
            mode, part, f, ix = parsed
            if mode != m_sym + 1:
                self.conflicts += 1
                continue
            code = bcpack._BC6H_CODE[mode]
            _trans, base, deltas, reg = bcpack._BC6H_INFO[code]
            di = bcpack._inter8(l1[a], l1[b])
            de = (di + bcpack._unzig(bcpack._inter8(l0[a], l0[b]))) & 255
            if de > i:
                ke = 256 + i - de
                self.ref_ep_seen[ke] = True
                mw = (1 << base) - 1
                w_sh = 16 - base
                for c in range(3):
                    dc = deltas[c]
                    mD = (1 << dc) - 1
                    sh = 6 - dc if dc < 6 else 0
                    dz0 = bcpack._unzig(bcpack._inter(lanes[2 + c][a], lanes[2 + c][b], 8))
                    dz1 = bcpack._unzig(bcpack._inter(lanes[5 + c][a], lanes[5 + c][b], 5))
                    dz2 = bcpack._unzig(lanes[8 + c][a])
                    dz3 = bcpack._unzig(lanes[8 + c][b])
                    w_val = ((f[c][0] - dz0) & mw) << w_sh
                    x_val = ((f[c][1] - dz1) & mD) << (10 - dc)
                    updates = [(0, base, w_val), (1, dc, x_val)]
                    if reg == 2:
                        y_val = ((f[c][2] - dz2) & mD) << sh
                        z_val = ((f[c][3] - dz3) & mD) << sh
                        updates.append((2, dc, y_val))
                        updates.append((3, dc, z_val))
                    for q, nb, val in updates:
                        cb, cv = self.ref_ep[ke][c][q]
                        nb2, nv2, conf = _merge_top_bits(cb, cv, nb, val, slot_widths[q])
                        if conf:
                            self.conflicts += 1
                        else:
                            self.ref_ep[ke][c][q][0] = nb2
                            self.ref_ep[ke][c][q][1] = nv2
                self.ref_blocks += 1
            if di > i:
                ki = 256 + i - di
                self.ref_ix_seen[ki] = True
                ibits = 3 if reg == 2 else 4
                imask = (1 << ibits) - 1
                ishift = 4 - ibits
                for yy in range(4):
                    sa = lanes[11 + yy][a]
                    sb = lanes[11 + yy][b]
                    row = yy * 4
                    if yy < 3:
                        ds = (deint1[sa], deint0[sa], deint1[sb], deint0[sb])
                    else:
                        ds = (deint0[sa], deint1[sa], deint0[sb], deint1[sb])
                    for off, dv in enumerate(ds):
                        p = row + off
                        v = ((ix[p] - bcpack._unzig(dv)) & imask) << ishift
                        cb, cv = self.ref_ix[ki][p]
                        nb2, nv2, conf = _merge_top_bits(cb, cv, ibits, v, 4)
                        if conf:
                            self.conflicts += 1
                        else:
                            self.ref_ix[ki][p][0] = nb2
                            self.ref_ix[ki][p][1] = nv2

    def add_pool_chunk(self, lanes, nblocks, inv_lookup=None, chunk_rel_start=0, get_quads=None):
        """Accumulate algebraic constraints and optional sibling-mip HDR endpoint targets."""
        self.pool_chunks += 1
        l0, l1, l15 = lanes[0], lanes[1], lanes[15]
        deint0, deint1 = bcpack._DEINT
        a2 = _bptc.BC7_ANCHOR2
        p2 = _bptc.load_bc7()["partition2"]
        eroot = [-1] * nblocks
        wchain = [None] * nblocks
        iroot = [-1] * nblocks
        imap = [None] * nblocks

        for i in range(nblocks):
            a, b = 2 * i, 2 * i + 1
            m_sym = l15[a]
            if m_sym > 13:
                continue
            mode = m_sym + 1
            code = bcpack._BC6H_CODE[mode]
            trans, base, deltas, reg = bcpack._BC6H_INFO[code]
            part = l15[b] if l15[b] < 31 else 31
            di = bcpack._inter8(l1[a], l1[b])
            de = (di + bcpack._unzig(bcpack._inter8(l0[a], l0[b]))) & 255
            ibits = 3 if reg == 2 else 4
            imask = (1 << ibits) - 1
            ishift = 4 - ibits

            # Index anchor constraints & intra-block smoothness
            if di > 0:
                if di > i:
                    ki = 256 + i - di
                    pmap = None
                else:
                    ki = iroot[i - di]
                    pmap = imap[i - di] if ki >= 0 else None
                if ki >= 0:
                    iroot[i] = ki
                    anchors = (0, a2[part]) if reg == 2 else (0,)
                    cur = {}
                    for p_idx in anchors:
                        if pmap is not None and p_idx not in pmap:
                            continue
                        yy, off = divmod(p_idx, 4)
                        s = lanes[11 + yy][a] if off < 2 else lanes[11 + yy][b]
                        dv = bcpack._unzig(deint1[s] if ((off % 2 == 0) if yy < 3 else (off % 2 == 1)) else deint0[s])
                        if pmap is None:
                            mp = tuple((((h >> ishift) + dv) & imask) << ishift for h in range(16))
                        else:
                            mp = tuple((((pmap[p_idx][h] >> ishift) + dv) & imask) << ishift for h in range(16))
                        cur[p_idx] = mp
                        mask = 0
                        for h in range(16):
                            if mp[h] < 8:
                                mask |= 1 << h
                        if self.valid_ix[ki][p_idx] & mask:
                            self.valid_ix[ki][p_idx] &= mask
                        self.ix_obs[ki][p_idx] += 1
                    if pmap is None:
                        sm_i = p2[part] if reg == 2 else None
                        s0 = lanes[11][a]
                        dv0 = bcpack._unzig(deint1[s0])
                        for p_idx in range(1, 16):
                            self.ix_obs[ki][p_idx] += 1
                            if sm_i is None or sm_i[p_idx] == sm_i[0]:
                                yy, off = divmod(p_idx, 4)
                                sp = lanes[11 + yy][a] if off < 2 else lanes[11 + yy][b]
                                dvp = bcpack._unzig(deint1[sp] if ((off % 2 == 0) if yy < 3 else (off % 2 == 1)) else deint0[sp])
                                diff_top = ((dv0 - dvp) & imask) << ishift
                                for lo in range(1 << ishift):
                                    self.ix_rel_votes[(ki, p_idx)][(diff_top | lo) & 15] += 1
                    imap[i] = cur

            # Endpoint chains and observations
            if de == 0:
                continue
            if de > i:
                ke = 256 + i - de
                pchain = None
            else:
                ke = eroot[i - de]
                pchain = wchain[i - de] if ke >= 0 else None
            if ke < 0:
                continue
            eroot[i] = ke
            self.ep_seen[ke] = True
            mw = (1 << base) - 1
            cur_w = [None, None, None]
            for c in range(3):
                dc = deltas[c]
                dz0 = bcpack._unzig(bcpack._inter(lanes[2 + c][a], lanes[2 + c][b], 8)) & mw
                if pchain is None:
                    cur_w[c] = ((base, dz0),)
                    if trans:
                        dz1 = bcpack._unzig(bcpack._inter(lanes[5 + c][a], lanes[5 + c][b], 5))
                        lst1 = self.d_obs.setdefault((ke, c, 1), [])
                        if len(lst1) < 96:
                            lst1.append((dc, dz1))
                        if reg == 2:
                            dz2 = bcpack._unzig(lanes[8 + c][a])
                            dz3 = bcpack._unzig(lanes[8 + c][b])
                            lst2 = self.d_obs.setdefault((ke, c, 2), [])
                            if len(lst2) < 96:
                                lst2.append((dc, dz2))
                            lst3 = self.d_obs.setdefault((ke, c, 3), [])
                            if len(lst3) < 96:
                                lst3.append((dc, dz3))
                elif pchain[c] is not None:
                    pops = pchain[c]
                    if pops[-1][0] == base:
                        cur_w[c] = pops[:-1] + ((base, (pops[-1][1] + dz0) & mw),)
                    elif len(pops) < 3:
                        cur_w[c] = pops + ((base, dz0),)
            wchain[i] = cur_w

            if inv_lookup is None or get_quads is None:
                continue
            xy = inv_lookup(chunk_rel_start + i * 16)
            if xy is None:
                continue
            qpts = get_quads(xy[0], xy[1])
            spread = max(max(p[c] for p in qpts) - min(p[c] for p in qpts) for c in range(3))
            wt = 4.0 if spread <= 512.0 else (2.0 if spread <= 2048.0 else 0.75)
            mean_rgb = tuple(sum(p[c] for p in qpts) / 4.0 for c in range(3))
            for c in range(3):
                if cur_w[c] is not None:
                    lst = self.w_obs.setdefault((ke, c), [])
                    if len(lst) < 64:
                        lst.append((cur_w[c], mode, max(0.0, min(65535.0, mean_rgb[c])), wt))

    @staticmethod
    def _eval_w_ops(ops, mode, r0):
        v = r0
        last_base = 16
        for base, dz in ops:
            sh = 16 - base
            m = (1 << base) - 1
            v = (((v >> sh) + dz) & m) << sh
            last_base = base
        d3d_w = bcpack._bc6h_stream_w(mode, v >> (16 - last_base))
        return _bc6h_unquantize(d3d_w, last_base, signed=False)

    @classmethod
    def _fit_w(cls, obs_list):
        cap = 8192.0
        cands = []
        for h in range(0, 65536, 1024):
            err = 0.0
            for ops, mode, tgt, wt in obs_list:
                d = abs(cls._eval_w_ops(ops, mode, h | 512) - tgt)
                err += wt * (d if d < cap else cap)
            cands.append((err, h))
        cands.sort()
        top = cands[:4]
        for step in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            nxt = {}
            mid = step // 2
            for _, b in top:
                for off in (0, step):
                    h = (b + off) & 65535
                    if h in nxt:
                        continue
                    err = 0.0
                    for ops, mode, tgt, wt in obs_list:
                        d = abs(cls._eval_w_ops(ops, mode, h | mid) - tgt)
                        err += wt * (d if d < cap else cap)
                    nxt[h] = err
            top = sorted((e, h) for h, e in nxt.items())[:4]
        return top[0][1]

    @staticmethod
    def _fit_signed_delta(obs_list, total_bits):
        if not obs_list:
            return 1 << (total_bits - 1)
        limit = 1 << total_bits
        best_err = 1e30
        best_r = 0
        for r in range(limit):
            err = 0.0
            for dc, dz in obs_list:
                sh = total_bits - dc if dc < total_bits else 0
                mD = (1 << dc) - 1
                v = ((r >> sh) + dz) & mD
                err += abs(_sign_extend(v, dc)) / float(1 << (dc - 1))
            if err < best_err:
                best_err = err
                best_r = r
        return best_r

    def build_document(self):
        eh = [[[0, 0, 0, 0] for _ in range(3)] for _ in range(256)]
        ih = [[0] * 16 for _ in range(256)]
        ep_cb = [[[self.ref_ep[k][c][q][0] for q in range(4)] for c in range(3)] for k in range(256)]
        ix_cb = [[self.ref_ix[k][p][0] for p in range(16)] for k in range(256)]
        ec = [0] * 256
        ic = [0] * 256

        for k in range(256):
            # Endpoints
            if self.ref_ep_seen[k]:
                defaults = (0x8000, 0x200, 0x20, 0x20)
                for c in range(3):
                    for q in range(4):
                        rb, rv = self.ref_ep[k][c][q]
                        eh[k][c][q] = rv if rb > 0 else defaults[q]
                ec[k] = 3
            elif self.ep_seen[k]:
                has_w = 0
                has_d = 0
                for c in range(3):
                    wl = self.w_obs.get((k, c))
                    if wl:
                        w_val = self._fit_w(wl)
                        has_w += 1
                    else:
                        w_val = 0x8000
                    x_obs = self.d_obs.get((k, c, 1))
                    y_obs = self.d_obs.get((k, c, 2))
                    z_obs = self.d_obs.get((k, c, 3))
                    if x_obs or y_obs or z_obs:
                        has_d += 1
                    x_val = self._fit_signed_delta(x_obs, 10)
                    y_val = self._fit_signed_delta(y_obs, 6)
                    z_val = self._fit_signed_delta(z_obs, 6)
                    eh[k][c] = [w_val, x_val, y_val, z_val]
                if has_w == 3:
                    ec[k] = 1
                elif has_w > 0 or has_d > 0:
                    ec[k] = 2

            # Indices
            if self.ref_ix_seen[k] and all(self.ref_ix[k][p][0] > 0 for p in range(16)):
                ih[k] = [self.ref_ix[k][p][1] for p in range(16)]
                ic[k] = 3
            elif any(self.ix_obs[k]):
                v0_lead, b0 = _lead_bits_4(self.valid_ix[k][0])
                surv0 = [h for h in range(16) if (self.valid_ix[k][0] >> h) & 1]
                v0 = v0_lead if b0 == 4 else (surv0[len(surv0) // 2] if surv0 else 0)
                row = [0] * 16
                total_votes = 0
                for p in range(16):
                    v, b = _lead_bits_4(self.valid_ix[k][p])
                    if b == 4:
                        row[p] = v
                        continue
                    mask = self.valid_ix[k][p]
                    votes = self.ix_rel_votes[(k, p)]
                    sv = sum(votes)
                    total_votes += sv
                    if p == 0:
                        best_h = v0
                    elif sv > 0:
                        best_h = max(
                            (h for h in range(16) if (mask >> h) & 1),
                            key=lambda h: votes[(h - v0) & 15],
                            default=v0,
                        )
                    else:
                        surv = [h for h in range(16) if (mask >> h) & 1]
                        best_h = surv[len(surv) // 2] if surv else 0
                    row[p] = best_h
                ih[k] = row
                ic[k] = 1 if (b0 >= 2 and total_votes > 0) else 2

        doc = {
            "endpointHistory": eh,
            "indexHistory": ih,
            "endpointClass": ec,
            "indexClass": ic,
            "endpointConfirmedBits": ep_cb,
            "indexConfirmedBits": ix_cb,
        }
        stats = {
            "endpoint_confirmed": sum(1 for c in ec if c == 3),
            "endpoint_provisional": sum(1 for c in ec if c == 1),
            "endpoint_last_resort": sum(1 for c in ec if c == 2),
            "index_confirmed": sum(1 for c in ic if c == 3),
            "index_provisional": sum(1 for c in ic if c == 1),
            "index_last_resort": sum(1 for c in ic if c == 2),
            "reference_blocks": self.ref_blocks,
            "reference_conflicts": self.conflicts,
            "pool_chunks": self.pool_chunks,
            "pool_textures": self.pool_textures,
        }
        return doc, stats


class Bc7Calibrator:
    """Calibrator for BC7 (`codec 7`) endpoint and index initial history tables."""

    def __init__(self):
        self.tab = _bptc.load_bc7()
        # Exact reference-pair constraints
        self.ref_ep = [[[0, 0] for _ in range(24)] for _ in range(256)]
        self.ref_ix = [[[0, 0] for _ in range(32)] for _ in range(256)]
        self.conflicts = 0
        self.ref_blocks = 0
        self.pool_chunks = 0
        self.pool_textures = 0

        # Algebraic & pool self-calibration accumulators
        self.valid_ix = [[0xFFFF] * 32 for _ in range(256)]
        self.ix_obs = [[0] * 32 for _ in range(256)]
        self.ix_rel_votes = {(k, p): [0] * 16 for k in range(256) for p in range(32)}
        self.pbit_parity = {}
        self.ep_obs = {}

    def add_reference_chunk(self, lanes, ref_blocks, nblocks, *, valid_blocks=None):
        """Algebraically invert direct built-in BC7 history references from exact blocks."""
        cfg = self.tab["config"]
        rank_of = self.tab["rank_of"]
        l0, l1, l15 = lanes[0], lanes[1], lanes[15]
        rows1 = (lanes[11], lanes[12], lanes[13], lanes[14])
        rows2 = (lanes[6], lanes[7], lanes[8], lanes[10])
        deint0, deint1 = bcpack._DEINT

        for i in range(min(nblocks, 256)):
            if valid_blocks is not None and not valid_blocks[i]:
                continue
            aa, bb = 2 * i, 2 * i + 1
            s15a, s15b = l15[aa], l15[bb]
            rank = rank_of[s15a * 16 + s15b] if s15a < 32 and s15b < 16 else -1
            if rank < 0 or cfg[rank][0] > 7:
                continue
            parsed = unpack_bc7_block(ref_blocks, i * 16)
            if parsed is None:
                continue
            mode, part, rot, isel, val, idx1, idx2 = parsed
            if (mode, part, rot, isel) != cfg[rank]:
                self.conflicts += 1
                continue
            di = bcpack._inter8(l1[aa], l1[bb])
            de = (di + bcpack._unzig(bcpack._inter8(l0[aa], l0[bb]))) & 255
            ns = bcpack._B7_NS[mode]
            cb = bcpack._B7_CB[mode]
            ab = bcpack._B7_AB[mode]
            pk = bcpack._B7_PK[mode]
            b1 = bcpack._B7_IB[mode]

            if de > i:
                ke = 256 + i - de
                w = [cb, cb, cb, ab]
                if rot:
                    w[rot - 1], w[3] = ab, cb
                for sub in range(ns):
                    lrow = bcpack._B7_LANE[sub]
                    bsub = sub * 8
                    for c in range(4):
                        li = lrow[c]
                        wc = w[c]
                        if not wc or li < 0:
                            continue
                        bits = wc + (1 if (pk and c == 0) else 0)
                        sh = 8 - bits
                        mask_b = (1 << bits) - 1
                        dz0 = bcpack._unzig(lanes[li][aa])
                        dz1 = bcpack._unzig(lanes[li][bb])
                        for e_idx, dz in ((0, dz0), (1, dz1)):
                            slot = bsub + e_idx * 4 + c
                            v = (((val[sub][e_idx][c] >> sh) - dz) & mask_b) << sh
                            cb0, cv0 = self.ref_ep[ke][slot]
                            nb2, nv2, conf = _merge_top_bits(cb0, cv0, bits, v, 8)
                            if conf:
                                self.conflicts += 1
                            else:
                                self.ref_ep[ke][slot][0] = nb2
                                self.ref_ep[ke][slot][1] = nv2
                self.ref_blocks += 1

            if di > i:
                ki = 256 + i - di
                # `idx1` / `idx2` from `unpack_bc7_block` are already mapped to
                # BCPack's `(s1=rows1, s2=rows2)` order; `rows2` is always 2 bits.
                pri_bits = bcpack._B7_IB[mode]
                pri_sh = 4 - pri_bits
                pri_m = (1 << pri_bits) - 1
                for yy in range(4):
                    sa = rows1[yy][aa]
                    sb = rows1[yy][bb]
                    row = yy * 4
                    if yy < 3:
                        ds = (deint1[sa], deint0[sa], deint1[sb], deint0[sb])
                    else:
                        ds = (deint0[sa], deint1[sa], deint0[sb], deint1[sb])
                    for off, dv in enumerate(ds):
                        p = row + off
                        v = ((idx1[p] - bcpack._unzig(dv)) & pri_m) << pri_sh
                        cb0, cv0 = self.ref_ix[ki][p]
                        nb2, nv2, conf = _merge_top_bits(cb0, cv0, pri_bits, v, 4)
                        if conf:
                            self.conflicts += 1
                        else:
                            self.ref_ix[ki][p][0] = nb2
                            self.ref_ix[ki][p][1] = nv2
                if mode in (4, 5) and idx2 is not None:
                    sec_bits = 2
                    sec_sh = 2
                    sec_m = 3
                    for yy in range(4):
                        sa = rows2[yy][aa]
                        sb = rows2[yy][bb]
                        row = yy * 4
                        if yy < 3:
                            ds = (deint1[sa], deint0[sa], deint1[sb], deint0[sb])
                        else:
                            ds = (deint0[sa], deint1[sa], deint0[sb], deint1[sb])
                        for off, dv in enumerate(ds):
                            p = row + off
                            v = ((idx2[p] - bcpack._unzig(dv)) & sec_m) << sec_sh
                            cb0, cv0 = self.ref_ix[ki][16 + p]
                            nb2, nv2, conf = _merge_top_bits(cb0, cv0, sec_bits, v, 4)
                            if conf:
                                self.conflicts += 1
                            else:
                                self.ref_ix[ki][16 + p][0] = nb2
                                self.ref_ix[ki][16 + p][1] = nv2

    def add_pool_chunk(self, lanes, nblocks, inv_lookup=None, chunk_rel_start=0, get_quads=None):
        """Accumulate BC7 anchor/p-bit constraints and optional sibling-mip RGBA endpoint fits."""
        self.pool_chunks += 1
        cfg = self.tab["config"]
        rank_of = self.tab["rank_of"]
        inferred = self.tab["inferred"]
        a2, a3a, a3b = self.tab["anchor2"], self.tab["anchor3a"], self.tab["anchor3b"]
        p2, p3 = self.tab["partition2"], self.tab["partition3"]
        deint0, deint1 = bcpack._DEINT
        l0, l1, l15 = lanes[0], lanes[1], lanes[15]
        rows1 = (lanes[11], lanes[12], lanes[13], lanes[14])
        rows2 = (lanes[6], lanes[7], lanes[8], lanes[10])

        eroot = [-1] * nblocks
        echain = [None] * nblocks
        iroot = [-1] * nblocks
        imap = [None] * nblocks

        for i in range(nblocks):
            aa, bb = 2 * i, 2 * i + 1
            s15a, s15b = l15[aa], l15[bb]
            rank = rank_of[s15a * 16 + s15b] if s15a < 32 and s15b < 16 else -1
            if rank < 0 or rank in inferred:
                continue
            mode, part, rot, _isel = cfg[rank]
            if mode > 7:
                continue
            di = bcpack._inter8(l1[aa], l1[bb])
            de = (di + bcpack._unzig(bcpack._inter8(l0[aa], l0[bb]))) & 255
            ns = bcpack._B7_NS[mode]
            b1 = bcpack._B7_IB[mode]
            b1_sh = 4 - b1
            b1_m = (1 << b1) - 1

            # Index anchor constraints & intra-block smoothness
            if di > 0:
                if di > i:
                    ki = 256 + i - di
                    pmap = None
                else:
                    ki = iroot[i - di]
                    pmap = imap[i - di] if ki >= 0 else None
                if ki >= 0:
                    iroot[i] = ki
                    anchors = (0,) if ns == 1 else ((0, a2[part]) if ns == 2 else (0, a3a[part], a3b[part]))
                    pri_anc = (0,) if mode in (4, 5) else anchors
                    cur = {}
                    for p_idx in pri_anc:
                        if pmap is not None and p_idx not in pmap:
                            continue
                        yy, off = divmod(p_idx, 4)
                        s = rows1[yy][aa] if off < 2 else rows1[yy][bb]
                        dv = bcpack._unzig(deint1[s] if ((off % 2 == 0) if yy < 3 else (off % 2 == 1)) else deint0[s])
                        if pmap is None:
                            mp = tuple((((h >> b1_sh) + dv) & b1_m) << b1_sh for h in range(16))
                        else:
                            mp = tuple((((pmap[p_idx][h] >> b1_sh) + dv) & b1_m) << b1_sh for h in range(16))
                        cur[p_idx] = mp
                        mask = 0
                        for h in range(16):
                            if mp[h] < 8:
                                mask |= 1 << h
                        if self.valid_ix[ki][p_idx] & mask:
                            self.valid_ix[ki][p_idx] &= mask
                        self.ix_obs[ki][p_idx] += 1
                    if pmap is None:
                        sm_i = None if ns == 1 else (p2[part] if ns == 2 else p3[part])
                        s0 = rows1[0][aa]
                        dv0 = bcpack._unzig(deint1[s0])
                        for p_idx in range(1, 16):
                            self.ix_obs[ki][p_idx] += 1
                            if sm_i is None or sm_i[p_idx] == sm_i[0]:
                                yy, off = divmod(p_idx, 4)
                                sp = rows1[yy][aa] if off < 2 else rows1[yy][bb]
                                dvp = bcpack._unzig(deint1[sp] if ((off % 2 == 0) if yy < 3 else (off % 2 == 1)) else deint0[sp])
                                diff_top = ((dv0 - dvp) & b1_m) << b1_sh
                                for lo in range(1 << b1_sh):
                                    self.ix_rel_votes[(ki, p_idx)][(diff_top | lo) & 15] += 1
                    if mode in (4, 5):
                        b2 = 2
                        b2_sh = 2
                        b2_m = 3
                        s2a = rows2[0][aa]
                        dv2 = bcpack._unzig(deint1[s2a])
                        if pmap is None:
                            m2 = tuple((((h >> b2_sh) + dv2) & b2_m) << b2_sh for h in range(16))
                        elif 16 in pmap:
                            m2 = tuple((((pmap[16][h] >> b2_sh) + dv2) & b2_m) << b2_sh for h in range(16))
                        else:
                            m2 = None
                        if m2 is not None:
                            cur[16] = m2
                            mask = 0
                            for h in range(16):
                                if m2[h] < 8:
                                    mask |= 1 << h
                            if self.valid_ix[ki][16] & mask:
                                self.valid_ix[ki][16] &= mask
                            self.ix_obs[ki][16] += 1
                        if pmap is None:
                            for p_idx in range(1, 16):
                                self.ix_obs[ki][16 + p_idx] += 1
                                yy, off = divmod(p_idx, 4)
                                sp = rows2[yy][aa] if off < 2 else rows2[yy][bb]
                                dvp = bcpack._unzig(deint1[sp] if ((off % 2 == 0) if yy < 3 else (off % 2 == 1)) else deint0[sp])
                                diff_top = ((dv2 - dvp) & b2_m) << b2_sh
                                for lo in range(1 << b2_sh):
                                    self.ix_rel_votes[(ki, 16 + p_idx)][(diff_top | lo) & 15] += 1
                    imap[i] = cur

            # Endpoint chains & observations
            if de == 0:
                continue
            if de > i:
                ke = 256 + i - de
                pchain = None
            else:
                ke = eroot[i - de]
                pchain = echain[i - de] if ke >= 0 else None
            if ke < 0:
                continue
            eroot[i] = ke
            cb = bcpack._B7_CB[mode]
            ab = bcpack._B7_AB[mode]
            pk = bcpack._B7_PK[mode]
            w = [cb, cb, cb, ab]
            if rot:
                w[rot - 1], w[3] = ab, cb
            cur_e = [None] * 24
            for sub in range(ns):
                lrow = bcpack._B7_LANE[sub]
                bsub = sub * 8
                for c in range(4):
                    li = lrow[c]
                    wc = w[c]
                    if not wc or li < 0:
                        continue
                    bits = wc + (1 if (pk and c == 0) else 0)
                    mask_b = (1 << bits) - 1
                    dz0 = bcpack._unzig(lanes[li][aa]) & mask_b
                    dz1 = bcpack._unzig(lanes[li][bb]) & mask_b
                    for e_idx, dz in ((0, dz0), (1, dz1)):
                        slot = bsub + e_idx * 4 + c
                        if pchain is None:
                            cur_e[slot] = (slot, ((bits, dz),))
                        else:
                            prev = pchain[slot]
                            if prev is not None:
                                ss, pops = prev
                                if pops[-1][0] == bits:
                                    nops = pops[:-1] + ((bits, (pops[-1][1] + dz) & mask_b),)
                                elif len(pops) < 3:
                                    nops = pops + ((bits, dz),)
                                else:
                                    nops = None
                                if nops is not None:
                                    cur_e[slot] = (ss, nops)
                if pk == 2 and pchain is None:
                    par = (bcpack._unzig(lanes[lrow[0]][aa]) ^ bcpack._unzig(lanes[lrow[0]][bb])) & 1
                    rec = self.pbit_parity.setdefault((ke, sub), [0, 0])
                    rec[par] += 1
            for sub in range(ns, 3):
                cur_e[sub * 8:sub * 8 + 8] = cur_e[0:8]
            echain[i] = cur_e

            if inv_lookup is None or get_quads is None:
                continue
            xy = inv_lookup(chunk_rel_start + i * 16)
            if xy is None:
                continue
            qpts = get_quads(xy[0], xy[1])
            spread = max(max(p[c] for p in qpts) - min(p[c] for p in qpts) for c in range(4))
            weight = 4.0 if spread <= 6.0 else (2.0 if spread <= 18.0 else 0.5)
            sm = None if ns == 1 else (p2[part] if ns == 2 else p3[part])
            for sub in range(ns):
                if sm is None:
                    sub_rgba = tuple(sum(p[c] for p in qpts) / 4.0 for c in range(4))
                else:
                    acc_c = [0.0, 0.0, 0.0, 0.0]
                    sc = 0
                    for px_i in range(16):
                        if sm[px_i] == sub:
                            qi = ((px_i // 4) // 2) * 2 + ((px_i % 4) // 2)
                            qp = qpts[qi]
                            for cc in range(4):
                                acc_c[cc] += qp[cc]
                            sc += 1
                    if not sc:
                        continue
                    sub_rgba = tuple(v / sc for v in acc_c)
                for c in range(4):
                    s0 = cur_e[sub * 8 + c]
                    s1 = cur_e[sub * 8 + 4 + c]
                    if s0 is None or s1 is None:
                        continue
                    ss0, ops0 = s0
                    ss1, ops1 = s1
                    if ss0 // 8 != ss1 // 8 or (ss0 % 8) != c or (ss1 % 8) != (4 + c):
                        continue
                    orig_sub = ss0 // 8
                    lst = self.ep_obs.setdefault((ke, orig_sub, c), [])
                    if len(lst) < 60:
                        lst.append((ops0, ops1, sub_rgba[c], weight))

    @staticmethod
    def _fit_ep_pair(obs_list):
        cands = []
        for h0 in range(0, 256, 16):
            for h1 in range(0, 256, 16):
                err = 0.0
                for ops0, ops1, tgt, wt in obs_list:
                    e0 = _eval_bc7_ops(ops0, h0 | 8)
                    e1 = _eval_bc7_ops(ops1, h1 | 8)
                    d = abs(0.5 * (e0 + e1) - tgt)
                    err += wt * ((d if d < 40.0 else 40.0) + 0.01 * abs(e0 - e1))
                cands.append((err, h0, h1))
        cands.sort()
        top = cands[:4]
        for step in (8, 4, 2, 1):
            nxt = {}
            mid = step // 2
            for _, b0, b1 in top:
                for d0 in (0, step):
                    for d1 in (0, step):
                        h0, h1 = (b0 + d0) & 255, (b1 + d1) & 255
                        key = (h0, h1)
                        if key in nxt:
                            continue
                        err = 0.0
                        for ops0, ops1, tgt, wt in obs_list:
                            e0 = _eval_bc7_ops(ops0, h0 | mid)
                            e1 = _eval_bc7_ops(ops1, h1 | mid)
                            d = abs(0.5 * (e0 + e1) - tgt)
                            err += wt * ((d if d < 40.0 else 40.0) + 0.01 * abs(e0 - e1))
                        nxt[key] = err
            top = sorted((e, a, b) for (a, b), e in nxt.items())[:4]
        return top[0][1], top[0][2]

    def build_document(self):
        hist_ep = [[0] * 24 for _ in range(256)]
        hist_ep_b = [[0] * 24 for _ in range(256)]
        hist_ep_cb = [[0] * 24 for _ in range(256)]
        hist_ix = [[0] * 32 for _ in range(256)]
        hist_ix_b = [[0] * 32 for _ in range(256)]
        hist_ix_cb = [[0] * 32 for _ in range(256)]
        last_resort = []

        ep_conf_slots = 0
        ep_prov_slots = 0
        ix_conf_slots = 0
        ix_prov_slots = 0

        for k in range(256):
            # Indices: exact reference beats algebraic/smoothness
            eff_masks = list(self.valid_ix[k])
            for p in range(32):
                rb, rv = self.ref_ix[k][p]
                if rb > 0:
                    sh = 4 - rb
                    rmask = 0
                    for h in range(16):
                        if (h >> sh) == (rv >> sh):
                            rmask |= 1 << h
                    if eff_masks[p] & rmask:
                        eff_masks[p] &= rmask
                    else:
                        eff_masks[p] = rmask
                    hist_ix[k][p] = rv
                    hist_ix_b[k][p] = rb
                    hist_ix_cb[k][p] = rb
                    ix_conf_slots += 1
                    continue
                v, b = _lead_bits_4(eff_masks[p])
                if b > 0:
                    hist_ix[k][p] = v
                    hist_ix_b[k][p] = b
                    hist_ix_cb[k][p] = b
            surv0 = [h for h in range(16) if (eff_masks[0] >> h) & 1]
            v0 = hist_ix[k][0] if hist_ix_b[k][0] == 4 else (surv0[len(surv0) // 2] if surv0 else 0)
            surv16 = [h for h in range(16) if (eff_masks[16] >> h) & 1]
            v16 = hist_ix[k][16] if hist_ix_b[k][16] >= 2 else (surv16[len(surv16) // 2] if surv16 else 0)
            if any(self.ix_obs[k][:16]):
                for p in range(16):
                    if hist_ix_b[k][p] < 4:
                        mask = eff_masks[p]
                        votes = self.ix_rel_votes[(k, p)]
                        if p == 0:
                            best_h = v0
                        elif sum(votes) > 0:
                            best_h = max(
                                (h for h in range(16) if (mask >> h) & 1),
                                key=lambda h: votes[(h - v0) & 15],
                                default=v0,
                            )
                        else:
                            surv = [h for h in range(16) if (mask >> h) & 1]
                            best_h = surv[len(surv) // 2] if surv else 0
                        hist_ix[k][p] = best_h
                        hist_ix_b[k][p] = 4
                        if self.ref_ix[k][p][0] == 0:
                            ix_prov_slots += 1
            if any(self.ix_obs[k][16:]):
                for p in range(16, 32):
                    if hist_ix_b[k][p] < 2:
                        mask = eff_masks[p]
                        votes = self.ix_rel_votes[(k, p)]
                        if p == 16:
                            best_h = v16
                        elif sum(votes) > 0:
                            best_h = max(
                                (h for h in range(16) if (mask >> h) & 1),
                                key=lambda h: votes[(h - v16) & 15],
                                default=v16,
                            )
                        else:
                            surv = [h for h in range(16) if (mask >> h) & 1]
                            best_h = surv[len(surv) // 2] if surv else 0
                        hist_ix[k][p] = best_h
                        hist_ix_b[k][p] = 2
                        if self.ref_ix[k][p][0] == 0:
                            ix_prov_slots += 1

            # Endpoints: exact reference takes precedence per slot; otherwise sibling-mip fit
            for sub in range(3):
                for c in range(4):
                    s0 = sub * 8 + c
                    s1 = sub * 8 + 4 + c
                    rb0, rv0 = self.ref_ep[k][s0]
                    rb1, rv1 = self.ref_ep[k][s1]
                    if rb0 > 0:
                        hist_ep[k][s0] = rv0
                        hist_ep_b[k][s0] = rb0
                        hist_ep_cb[k][s0] = rb0
                        ep_conf_slots += 1
                    if rb1 > 0:
                        hist_ep[k][s1] = rv1
                        hist_ep_b[k][s1] = rb1
                        hist_ep_cb[k][s1] = rb1
                        ep_conf_slots += 1
                    if rb0 > 0 and rb1 > 0:
                        continue
                    obs_list = self.ep_obs.get((k, sub, c))
                    if not obs_list:
                        continue
                    best0, best1 = self._fit_ep_pair(obs_list)
                    if c == 0 and (k, sub) in self.pbit_parity:
                        c0, c1 = self.pbit_parity[(k, sub)]
                        if (c0 > 0) != (c1 > 0):
                            req = 0 if c0 > 0 else 1
                            if (((best0 >> 1) ^ (best1 >> 1)) & 1) != req:
                                best1 ^= 2
                    depth = 8 if len(obs_list) >= 4 else 5
                    if rb0 == 0:
                        hist_ep[k][s0] = best0
                        hist_ep_b[k][s0] = depth
                        ep_prov_slots += 1
                        if len(obs_list) < 3:
                            last_resort.append([k, sub, 0, c])
                    if rb1 == 0:
                        hist_ep[k][s1] = best1
                        hist_ep_b[k][s1] = depth
                        ep_prov_slots += 1
                        if len(obs_list) < 3:
                            last_resort.append([k, sub, 1, c])

        last_resort.sort()
        doc = {
            "historyEndpoints": hist_ep,
            "historyEndpointBits": hist_ep_b,
            "historyEndpointConfirmedBits": hist_ep_cb,
            "historyIndices": hist_ix,
            "historyIndexBits": hist_ix_b,
            "historyIndexConfirmedBits": hist_ix_cb,
            "historyEndpointLastResort": last_resort,
        }
        stats = {
            "endpoint_confirmed_slots": ep_conf_slots,
            "endpoint_provisional_slots": ep_prov_slots,
            "endpoint_last_resort_slots": len(last_resort),
            "index_confirmed_slots": ix_conf_slots,
            "index_provisional_slots": ix_prov_slots,
            "reference_blocks": self.ref_blocks,
            "reference_conflicts": self.conflicts,
            "pool_chunks": self.pool_chunks,
            "pool_textures": self.pool_textures,
        }
        return doc, stats


# ---------------------------------------------------------------------------
# Reference-pair loaders (`--ref-dds` and `--ref-pairs`)
# ---------------------------------------------------------------------------

def parse_ref_dds_spec(spec):
    """Parse `POOL:INDEX=REF.dds`, preserving Windows drive letters."""
    if "=" not in spec:
        raise CliError(f"invalid --ref-dds {spec!r}: expected POOL:INDEX=REF.dds")
    left, dds_path = spec.split("=", 1)
    if ":" not in left:
        raise CliError(f"invalid --ref-dds {spec!r}: expected POOL:INDEX=REF.dds")
    pool_path, idx_s = left.rsplit(":", 1)
    if not pool_path or not dds_path:
        raise CliError(f"invalid --ref-dds {spec!r}: expected POOL:INDEX=REF.dds")
    try:
        index = int(idx_s)
    except ValueError:
        raise CliError(f"invalid texture index in --ref-dds {spec!r}")
    if index < 0:
        raise CliError(f"texture index must be non-negative in --ref-dds {spec!r}")
    return pool_path, index, dds_path


def apply_ref_dds(pool_path, index, dds_path, calibrators):
    """Retile a reference DDS against a TTPF texture entry and invert all matching chunks."""
    if not os.path.isfile(pool_path):
        raise CliError(f"reference pool not found: {pool_path}")
    if not os.path.isfile(dds_path):
        raise CliError(f"reference DDS not found: {dds_path}")
    dxgi, width, height, mips = read_dds(dds_path)
    pool = _ttpf.Pool(pool_path)
    try:
        if not 0 <= index < pool.textures:
            raise CliError(f"texture index {index} out of range for {pool_path} ({pool.textures} entries)")
        d = pool.descriptor(index)
        a = _ttpf.analyze(pool, index, allow_bcpack=True)
        if "layout" not in a or not a.get("available"):
            raise CliError(f"{pool_path}:{index}: unsupported or unavailable layout for reference calibration")
        if d["width"] != width or d["height"] != height:
            raise CliError(
                f"{pool_path}:{index}: DDS size {width}x{height} disagrees with pool {d['width']}x{d['height']}"
            )
        lay = a["layout"]
        seen_chunks = set()
        # Build virtual-offset map of retiled DDS mip bytes.
        tiled_ranges: dict[int, bytearray] = {}
        known_ranges: dict[int, bytearray] = {}
        for m, mip in enumerate(lay["mips"]):
            if m >= len(mips):
                break
            start = d["base"] + mip["offset"]
            if start not in tiled_ranges:
                tiled_ranges[start] = bytearray(mip["region"])
                known_ranges[start] = bytearray(mip["region"])
            _retile_mip_into(mips[m], lay, mip, tiled_ranges[start], known_ranges[start])
        # Match every BCPack chunk touching those mips.
        for m, mip in enumerate(lay["mips"]):
            if m >= len(mips):
                break
            for virt, csize, src, packed, codec, _pad in pool.chunks_for(d["base"] + mip["offset"], mip["region"]):
                if codec not in calibrators:
                    continue
                if dxgi not in CODEC_DXGI[codec]:
                    raise CliError(f"{dds_path}: DXGI format {dxgi} does not match codec {CODEC_KEYS[codec]}")
                key = (src, packed, codec)
                if key in seen_chunks:
                    continue
                # Assemble the chunk's reference tiled bytes from `tiled_ranges`.
                ref_chunk = bytearray(csize)
                known_chunk = bytearray(csize)
                for mstart, mtiled in tiled_ranges.items():
                    mend = mstart + len(mtiled)
                    lo = max(virt, mstart)
                    hi = min(virt + csize, mend)
                    if lo < hi:
                        ref_chunk[lo - virt:hi - virt] = mtiled[lo - mstart:hi - mstart]
                        known_chunk[lo - virt:hi - virt] = known_ranges[mstart][lo - mstart:hi - mstart]
                eb = bcpack.BLOCK_BYTES[codec]
                nblocks = csize // eb
                max_b = min(nblocks, 256)
                # DDS files contain visible blocks only. Tiled padding and
                # bytes belonging to absent mips are not reference evidence.
                valid_blocks = [all(known_chunk[i * eb:(i + 1) * eb]) for i in range(max_b)]
                if not any(valid_blocks):
                    continue
                raw = pool.data[src:src + packed]
                sym_per_blk = 1 if codec in (1, 4) else 2
                lanes = decode_lanes_prefix(raw, codec, csize, max_b * sym_per_blk)
                calibrators[codec].add_reference_chunk(lanes, bytes(ref_chunk), max_b,
                                                      valid_blocks=valid_blocks)
                seen_chunks.add(key)
    finally:
        pool.close()


def apply_ref_pairs_file(pairs_path, calibrators):
    """Apply a JSON file of raw `(codec, chunk, reference)` block pairs or DDS pairs."""
    if not os.path.isfile(pairs_path):
        raise CliError(f"reference pairs file not found: {pairs_path}")
    try:
        with open(pairs_path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"{pairs_path}: {exc}")
    items = data.get("pairs") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise CliError(f"{pairs_path}: expected a JSON list or {{\"pairs\": [...]}}")
    base_dir = os.path.dirname(os.path.abspath(pairs_path))
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise CliError(f"{pairs_path}: pair {idx} must be a JSON object")
        if "pool" in item and "dds" in item:
            p_path = item["pool"]
            d_path = item["dds"]
            if not os.path.isabs(p_path):
                p_path = os.path.join(base_dir, p_path)
            if not os.path.isabs(d_path):
                d_path = os.path.join(base_dir, d_path)
            apply_ref_dds(p_path, int(item["index"]), d_path, calibrators)
            continue
        cspec = str(item.get("codec", "")).strip().lower()
        codec = CODEC_ALIASES.get(cspec)
        if codec is None:
            raise CliError(f"{pairs_path}: pair {idx} has invalid codec {item.get('codec')!r}")
        if codec not in calibrators:
            continue
        if "chunk_hex" in item:
            raw = bytes.fromhex(item["chunk_hex"])
        elif "chunk_file" in item:
            cp = item["chunk_file"]
            if not os.path.isabs(cp):
                cp = os.path.join(base_dir, cp)
            with open(cp, "rb") as fh:
                raw = fh.read()
        else:
            raise CliError(f"{pairs_path}: pair {idx} missing chunk_hex or chunk_file")
        if "ref_hex" in item:
            ref_bytes = bytes.fromhex(item["ref_hex"])
        elif "ref_file" in item:
            rp = item["ref_file"]
            if not os.path.isabs(rp):
                rp = os.path.join(base_dir, rp)
            with open(rp, "rb") as fh:
                ref_bytes = fh.read()
        else:
            raise CliError(f"{pairs_path}: pair {idx} missing ref_hex or ref_file")
        eb = bcpack.BLOCK_BYTES[codec]
        out_size = int(item.get("out_size", len(ref_bytes)))
        if out_size <= 0 or out_size % eb != 0 or len(ref_bytes) < out_size:
            raise CliError(f"{pairs_path}: pair {idx} out_size/ref length mismatch")
        nblocks = out_size // eb
        max_b = min(nblocks, 256)
        sym_per_blk = 1 if codec in (1, 4) else 2
        lanes = decode_lanes_prefix(raw, codec, out_size, max_b * sym_per_blk)
        calibrators[codec].add_reference_chunk(lanes, ref_bytes, max_b)


# ---------------------------------------------------------------------------
# Pool self-calibration runner
# ---------------------------------------------------------------------------

def _make_sibling_quads_getter(codec, lay, mip_z, ztiled, kind, signed=False):
    zblk_cache = {}
    pw, ph = mip_z["pixels_w"], mip_z["pixels_h"]

    def _decode_block(bx_z, by_z):
        key = (bx_z, by_z)
        blk = zblk_cache.get(key)
        if blk is None:
            off = tiled_offset(lay, mip_z, bx_z, by_z)
            if codec == 1:
                blk = decode_bc1_block_rgba(ztiled, off)
            elif codec == 3:
                blk = decode_bc3_block_rgba(ztiled, off)
            elif codec == 4:
                blk = decode_bc4_block_u8(ztiled, off)
            elif codec == 5:
                blk = decode_bc5_block_rg(ztiled, off)
            elif codec == 6:
                blk = decode_bc6h_block_unq(ztiled, off, signed=signed)
            else:
                blk = decode_bc7_block_rgba(ztiled, off)
            zblk_cache[key] = blk
        return blk

    def _get_pixel(px, py):
        px = min(pw - 1, max(0, px))
        py = min(ph - 1, max(0, py))
        blk = _decode_block(px >> 2, py >> 2)
        return blk[((py & 3) << 2) | (px & 3)]

    def get_quads(bx, by):
        if kind == "child":
            return tuple(_get_pixel(2 * bx + dx, 2 * by + dy) for dy in (0, 1) for dx in (0, 1))
        out = []
        for qy in (0, 1):
            for qx in (0, 1):
                x0, y0 = 8 * bx + 4 * qx, 8 * by + 4 * qy
                x1, y1 = min(pw, x0 + 4), min(ph, y0 + 4)
                if codec == 4:
                    acc = 0.0
                    cnt = 0
                    for py in range(y0, y1):
                        for px in range(x0, x1):
                            acc += _get_pixel(px, py)
                            cnt += 1
                    out.append(acc / cnt if cnt else 0.0)
                else:
                    nch = 2 if codec == 5 else (3 if codec == 6 else 4)
                    acc = [0.0] * nch
                    cnt = 0
                    for py in range(y0, y1):
                        for px in range(x0, x1):
                            pv = _get_pixel(px, py)
                            for c in range(nch):
                                acc[c] += pv[c]
                            cnt += 1
                    out.append(tuple(v / cnt for v in acc) if cnt else (0.0,) * nch)
        return tuple(out)

    return get_quads


def calibrate_from_pools(pools, calibrators, *, limit=0, max_blocks=256):
    """Scan TTPF pools read-only to populate self-calibrated history tables."""
    target_codecs = set(calibrators.keys())
    dxgi_to_codec = {}
    for cid in target_codecs:
        for dfmt in CODEC_DXGI[cid]:
            dxgi_to_codec[dfmt] = cid

    # When `limit > 0`, divide quota round-robin across pools so every pool
    # contributes textures rather than exhausting `--limit` on the first pool.
    n_pools = len(pools)
    per_pool_cap = {}
    if limit > 0 and n_pools > 0:
        base_q = max(1, (limit + n_pools - 1) // n_pools)
        for cid in target_codecs:
            per_pool_cap[cid] = base_q

    tex_counts = {cid: 0 for cid in target_codecs}
    alg_counts = {cid: 0 for cid in target_codecs}
    alg_limit = max(limit * 4, 256) if limit > 0 else 0

    for pool_path in pools:
        if limit > 0 and all(tex_counts[c] >= limit for c in target_codecs):
            break
        pool = _ttpf.Pool(pool_path)
        try:
            cache = {}
            pool_tex = {cid: 0 for cid in target_codecs}
            seen_alg = set()
            for ti in range(pool.textures):
                if limit > 0 and all(
                    tex_counts[c] >= limit or pool_tex[c] >= per_pool_cap[c]
                    for c in target_codecs
                ):
                    break
                d = pool.descriptor(ti)
                if d["base"] == 0 or d["dimension"] != 1 or d["depth"] != 1:
                    continue
                cid = dxgi_to_codec.get(d["format"])
                if cid is None:
                    continue
                need_mixed = (limit == 0) or (
                    tex_counts[cid] < limit and pool_tex[cid] < per_pool_cap[cid]
                )
                need_alg = cid in (6, 7) and (alg_limit == 0 or alg_counts[cid] < alg_limit)
                if not need_mixed and not need_alg:
                    continue
                a = _ttpf.analyze(pool, ti, allow_bcpack=True)
                if "layout" not in a or not a.get("available"):
                    continue
                mc = a["mip_codecs"]
                lay = a["layout"]
                nmips = len(lay["mips"])
                # Find mixed (BCPack mip, adjacent raw/zlib mip) pairs
                pairs = []
                if need_mixed:
                    for m in range(nmips):
                        if cid not in mc[m]:
                            continue
                        if m + 1 < nmips and set(mc[m + 1]) <= {0, 8}:
                            pairs.append((m, m + 1, "child"))
                        elif m > 0 and set(mc[m - 1]) <= {0, 8} and lay["mips"][m - 1]["region"] <= 1048576:
                            pairs.append((m, m - 1, "parent"))
                cal = calibrators[cid]
                eb = bcpack.BLOCK_BYTES[cid]
                sym_per_blk = 1 if cid in (1, 4) else 2
                signed = d["format"] in (81, 84, 96)
                used_mixed = False

                for m, mz, kind in pairs:
                    mip_bc = lay["mips"][m]
                    mip_z = lay["mips"][mz]
                    mip_virt = d["base"] + mip_bc["offset"]
                    chunks = pool.chunks_for(mip_virt, mip_bc["region"])
                    bc_chunks = [c for c in chunks if c[4] == cid]
                    if not bc_chunks:
                        continue
                    ztiled = pool.read_range(d["base"] + mip_z["offset"], mip_z["region"], cache, bcpack_mode="off")
                    inv_lookup = make_inverse_tile_lookup(lay, mip_bc)
                    get_quads = _make_sibling_quads_getter(cid, lay, mip_z, ztiled, kind, signed=signed)
                    for virt, csize, src, packed, codec, _pad in bc_chunks:
                        nb = min(csize // eb, max_blocks)
                        if nb <= 0:
                            continue
                        raw = pool.data[src:src + packed]
                        lanes = decode_lanes_prefix(raw, cid, csize, nb * sym_per_blk)
                        cal.add_pool_chunk(lanes, nb, inv_lookup, virt - mip_virt, get_quads)
                        seen_alg.add((src, packed, cid))
                        used_mixed = True

                if used_mixed:
                    tex_counts[cid] += 1
                    pool_tex[cid] += 1
                    cal.pool_textures += 1
                elif need_alg:
                    # Pure-BCPack BC6H/BC7 chunks still yield exact anchor & p-bit constraints
                    for m in range(nmips):
                        if cid not in mc[m]:
                            continue
                        mip_bc = lay["mips"][m]
                        for virt, csize, src, packed, codec, _pad in pool.chunks_for(
                            d["base"] + mip_bc["offset"], mip_bc["region"]
                        ):
                            if codec != cid or (src, packed, cid) in seen_alg:
                                continue
                            nb = min(csize // eb, max_blocks)
                            if nb <= 0:
                                continue
                            raw = pool.data[src:src + packed]
                            lanes = decode_lanes_prefix(raw, cid, csize, nb * sym_per_blk)
                            cal.add_pool_chunk(lanes, nb)
                            seen_alg.add((src, packed, cid))
                            alg_counts[cid] += 1
                            break
                        break
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Schema validation & pool verification (`--verify` / `--check`)
# ---------------------------------------------------------------------------

def validate_history_schema(history_path, codecs):
    """Validate history JSON schema for `codecs` and return aggregate coverage counts."""
    if not os.path.exists(history_path):
        raise CliError(f"history path not found: {history_path}")
    summary = {}
    for cid in codecs:
        ckey = CODEC_KEYS[cid]
        if cid == 6:
            eh, ih, ec, ic = _bptc.load_bc6h(history_path)
            for i in range(256):
                if not 0 <= ec[i] <= 3:
                    raise ValueError(f"BC6H endpointClass[{i}]={ec[i]} outside 0..3")
                if not 0 <= ic[i] <= 3:
                    raise ValueError(f"BC6H indexClass[{i}]={ic[i]} outside 0..3")
                for c in range(3):
                    w, x, y, z = eh[i][c]
                    if not (0 <= w <= 65535 and 0 <= x <= 1023 and 0 <= y <= 63 and 0 <= z <= 63):
                        raise ValueError(f"BC6H endpointHistory[{i}][{c}] out of bounds")
                if any(not 0 <= p <= 15 for p in ih[i]):
                    raise ValueError(f"BC6H indexHistory[{i}] pixel out of 0..15")
            summary[ckey] = {
                "endpoint_confirmed": sum(1 for v in ec if v == 3),
                "endpoint_provisional": sum(1 for v in ec if v == 1),
                "endpoint_last_resort": sum(1 for v in ec if v == 2),
                "endpoint_unknown": sum(1 for v in ec if v == 0),
                "index_confirmed": sum(1 for v in ic if v == 3),
                "index_provisional": sum(1 for v in ic if v == 1),
                "index_last_resort": sum(1 for v in ic if v == 2),
                "index_unknown": sum(1 for v in ic if v == 0),
            }
        elif cid == 7:
            tab = _bptc.load_bc7(history_path)
            ep = tab["history_endpoints"]
            ep_b = tab["history_endpoint_bits"]
            ep_cb = tab["history_endpoint_confirmed_bits"]
            ix = tab["history_indices"]
            ix_b = tab["history_index_bits"]
            ix_cb = tab["history_index_confirmed_bits"]
            for i in range(256):
                for j in range(24):
                    if not 0 <= ep[i][j] <= 255:
                        raise ValueError(f"BC7 historyEndpoints[{i}][{j}] out of 0..255")
                    if not 0 <= ep_cb[i][j] <= ep_b[i][j] <= 8:
                        raise ValueError(f"BC7 endpoint bit depth invalid at [{i}][{j}]")
                for j in range(32):
                    if not 0 <= ix[i][j] <= 15:
                        raise ValueError(f"BC7 historyIndices[{i}][{j}] out of 0..15")
                    if not 0 <= ix_cb[i][j] <= ix_b[i][j] <= 4:
                        raise ValueError(f"BC7 index bit depth invalid at [{i}][{j}]")
            summary[ckey] = {
                "endpoint_confirmed_slots": sum(1 for row in ep_cb for v in row if v > 0),
                "endpoint_known_slots": sum(1 for row in ep_b for v in row if v > 0),
                "endpoint_last_resort_slots": len(tab["last_resort"]),
                "index_confirmed_slots": sum(1 for row in ix_cb for v in row if v > 0),
                "index_known_slots": sum(1 for row in ix_b for v in row if v > 0),
            }
        else:
            loaded = _bptc.load_bc_history(cid, history_path)
            if loaded is None:
                summary[ckey] = {"present": False}
            else:
                conf, prov, rep = loaded
                summary[ckey] = {
                    "present": True,
                    "confirmed_entries": len(conf),
                    "provisional_entries": len(prov),
                    "replace": rep,
                }
    return summary


def verify_against_pools(history_path, codecs, pools, *, limit=20):
    """Test sampled BCPack chunks from `pools` with `history_path` (aggregate stats only)."""
    target = set(codecs)
    cap = limit if limit > 0 else 20
    results = {
        CODEC_KEYS[c]: {"tested": 0, "strict_ok": 0, "provisional_ok": 0, "refused": 0, "errors": 0}
        for c in codecs
    }
    tested = {c: 0 for c in codecs}
    seen = set()
    for pool_path in pools:
        if all(tested[c] >= cap for c in codecs):
            break
        pool = _ttpf.Pool(pool_path)
        try:
            for virt, csize, src, packed, codec, _pad in pool.chunks:
                if all(tested[c] >= cap for c in codecs):
                    break
                if codec not in target or tested[codec] >= cap:
                    continue
                key = (pool_path, src, packed, codec)
                if key in seen:
                    continue
                seen.add(key)
                raw = pool.data[src:src + packed]
                rec = results[CODEC_KEYS[codec]]
                rec["tested"] += 1
                tested[codec] += 1
                try:
                    bcpack.decode_with_info(raw, codec, csize, allow_provisional=False, history=history_path)
                    rec["strict_ok"] += 1
                    rec["provisional_ok"] += 1
                    continue
                except bcpack.BcpackProvisionalError:
                    pass
                except bcpack.BcpackError:
                    rec["errors"] += 1
                    continue
                try:
                    bcpack.decode_with_info(raw, codec, csize, allow_provisional=True, history=history_path)
                    rec["provisional_ok"] += 1
                except bcpack.BcpackProvisionalError:
                    rec["refused"] += 1
                except bcpack.BcpackError:
                    rec["errors"] += 1
        finally:
            pool.close()
    return results


# ---------------------------------------------------------------------------
# Output writer & path guards
# ---------------------------------------------------------------------------

def _resolve_pools(paths):
    pools = []
    for p in paths:
        found = _ttpf.find_pools(p)
        if not found:
            why = "no such file or folder" if not os.path.exists(p) else "no .ttpf pool in it"
            raise CliError(f"{p}: {why}")
        pools.extend(found)
    # Deterministic sorted deduplicated order
    seen = set()
    unique = []
    for p in sorted(pools, key=lambda s: os.path.normcase(os.path.abspath(s))):
        norm = os.path.normcase(os.path.abspath(p))
        if norm not in seen:
            seen.add(norm)
            unique.append(p)
    return unique


def _write_json_deterministic(path, obj):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_outputs(out_path, docs, pools):
    """Write either a combined `.json` file or per-codec `<codec>.json` files into a directory."""
    is_json_file = out_path.lower().endswith(".json")
    if is_json_file:
        if os.path.isdir(out_path):
            raise CliError(f"--out {out_path} is an existing directory")
        for p in pools:
            if _ttpf.same_output_path(out_path, p):
                raise CliError(f"--out would overwrite input pool {p}")
        _write_json_deterministic(out_path, docs)
        return [out_path]
    if os.path.isfile(out_path):
        raise CliError(f"--out {out_path} is an existing file (use a directory or a .json path)")
    os.makedirs(out_path, exist_ok=True)
    written = []
    for ckey in sorted(docs.keys()):
        target = os.path.join(out_path, f"{ckey}.json")
        for p in pools:
            if _ttpf.same_output_path(target, p):
                raise CliError(f"--out would overwrite input pool {p}")
        _write_json_deterministic(target, docs[ckey])
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="bcpack_history_calib.py",
        description=(
            "Locally calibrate and validate BCPack initial history JSON tables "
            "for tools/bcpack.py and tools/ttpf_mips.py --history."
        ),
    )
    ap.add_argument(
        "inputs",
        nargs="*",
        help="read-only .ttpf pool files or directories (e.g. game root or ttdb folder)",
    )
    ap.add_argument(
        "--out",
        metavar="PATH",
        help="output directory (writes bc6h.json, bc7.json, ...) or combined .json file",
    )
    ap.add_argument(
        "--codec",
        default="bc6h,bc7",
        help="comma-separated codecs (bc1,bc3,bc4,bc5,bc6h,bc7) or 'all' (default: bc6h,bc7)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max mixed textures per codec to scan from pools (0 = scan all; also verify sample cap)",
    )
    ap.add_argument(
        "--max-blocks",
        type=int,
        default=256,
        help="prefix blocks decoded per BCPack chunk during self-calibration (default: 256)",
    )
    ap.add_argument(
        "--ref-dds",
        action="append",
        default=[],
        metavar="POOL:INDEX=REF.dds",
        help="exact reference DDS paired with a pool texture (produces confirmed history entries)",
    )
    ap.add_argument(
        "--ref-pairs",
        action="append",
        default=[],
        metavar="PAIRS.json",
        help="JSON file of exact chunk/reference block pairs (produces confirmed history entries)",
    )
    ap.add_argument(
        "--verify",
        nargs="?",
        const="__OUT__",
        default=None,
        metavar="PATH",
        help="validate history JSON schema (and test sampled chunks if pools are given)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="alias for --verify on the written --out path",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="set replace=true on generated BC1-BC5 JSON files instead of overlaying defaults",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable aggregate JSON summary to stdout",
    )
    return ap


def run(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.limit < 0:
        raise CliError("--limit must be non-negative")
    if args.max_blocks <= 0:
        raise CliError("--max-blocks must be positive")
    codecs = parse_codecs(args.codec)
    pools = _resolve_pools(args.inputs) if args.inputs else []

    do_verify = args.check or (args.verify is not None)
    verify_target = None
    if args.verify == "__OUT__" or (args.check and args.verify is None):
        verify_target = args.out
    elif args.verify is not None:
        verify_target = args.verify

    if not args.out and not do_verify:
        raise CliError("specify --out <path> to calibrate or --verify <path> to validate")
    if do_verify and not verify_target:
        raise CliError("--verify / --check requires a history path or --out <path>")

    report = {"codecs": [CODEC_KEYS[c] for c in codecs]}

    if args.out:
        if not pools and not args.ref_dds and not args.ref_pairs:
            raise CliError("calibration requires TTPF inputs, --ref-dds, or --ref-pairs")
        calibrators = {}
        for cid in codecs:
            if cid == 6:
                calibrators[cid] = Bc6hCalibrator()
            elif cid == 7:
                calibrators[cid] = Bc7Calibrator()
            else:
                calibrators[cid] = BcSimpleCalibrator(cid)

        for spec in args.ref_dds:
            pool_path, index, dds_path = parse_ref_dds_spec(spec)
            apply_ref_dds(pool_path, index, dds_path, calibrators)
        for pairs_path in args.ref_pairs:
            apply_ref_pairs_file(pairs_path, calibrators)
        if pools:
            calibrate_from_pools(
                pools, calibrators, limit=args.limit, max_blocks=args.max_blocks
            )

        docs = {}
        cal_stats = {}
        for cid in codecs:
            ckey = CODEC_KEYS[cid]
            cal = calibrators[cid]
            if cid in (1, 3, 4, 5):
                doc, st = cal.build_document(replace=args.replace)
            else:
                doc, st = cal.build_document()
            docs[ckey] = doc
            cal_stats[ckey] = st

        written = write_outputs(args.out, docs, pools)
        report["out"] = args.out
        report["files_written"] = len(written)
        report["pools_scanned"] = len(pools)
        report["calibration"] = cal_stats

    if do_verify:
        schema = validate_history_schema(verify_target, codecs)
        verify_info = {"path": verify_target, "schema_ok": True, "tables": schema}
        if pools:
            verify_info["pool_check"] = verify_against_pools(
                verify_target, codecs, pools, limit=(args.limit or 20)
            )
        report["verify"] = verify_info

    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        if "calibration" in report:
            print(f"Calibrated {', '.join(report['codecs'])} -> {report['out']} ({report['files_written']} file(s))")
            for ckey, st in report["calibration"].items():
                parts = ", ".join(f"{k}={v}" for k, v in st.items())
                print(f"  {ckey}: {parts}")
        if "verify" in report:
            vi = report["verify"]
            print(f"Verified history schema at {vi['path']}: OK")
            for ckey, st in vi["tables"].items():
                parts = ", ".join(f"{k}={v}" for k, v in st.items())
                print(f"  {ckey} table: {parts}")
            if "pool_check" in vi:
                for ckey, st in vi["pool_check"].items():
                    print(
                        f"  {ckey} pool check: tested={st['tested']} strict={st['strict_ok']} "
                        f"provisional={st['provisional_ok']} refused={st['refused']} errors={st['errors']}"
                    )
    return 0


def main(argv=None):
    try:
        return run(argv)
    except (CliError, ValueError, OSError, json.JSONDecodeError, bcpack.BcpackError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
