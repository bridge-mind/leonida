"""Clean-room BCPack bitstream decoder (Python standard library only).

Implements the Xbox Series DirectStorage BCPack entropy layer (interleaved
12/16-lane 16-bit rANS with 8-byte/cycle prefetch scheduling and exponential
frequency tables) and block rebuilders for all six active modes: BC1, BC3,
BC4, BC5, BC6H and BC7 (codecs 1, 3, 4, 5, 6 and 7).

Codec identifiers follow `DSTORAGE_BCPACK_MODE` (matching `tools/ttpf_mips.py`
and `docs/textures.md`):
    1 = BC1, 3 = BC3, 4 = BC4, 5 = BC5, 6 = BC6H, 7 = BC7.

Strict decoding (`decode(..., allow_provisional=False)`, the default) refuses
any chunk that depends on an unconfirmed built-in history entry or inferred BC7
signature. Pass `allow_provisional=True` (or use `decode_with_info`) to allow
provisional entries and inspect which ones were used.

See `docs/textures.md` for the bitstream specification and provenance.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcpack_tables as _tables  # noqa: E402
import bcpack_tables_bptc as _bptc  # noqa: E402

# rANS and lane-array constants
RANS_L = 1 << 24
MAX_WIDTH = 10
PREFETCH_BUDGET = 8
MAX_VIRTUAL_ZEROS = 8

CODEC_NAMES = {
    1: "BC1",
    3: "BC3",
    4: "BC4",
    5: "BC5",
    6: "BC6H",
    7: "BC7",
}

# Bytes represented by one lane symbol-tuple (BC3/BC5/BC6H/BC7 use 2 elements per 16-byte block).
ELEMENT_BYTES = 8

BLOCK_BYTES = {
    1: 8,
    3: 16,
    4: 8,
    5: 16,
    6: 16,
    7: 16,
}

# Default per-lane symbol widths (bits) for each DSTORAGE_BCPACK_MODE.
LANE_WIDTHS = {
    1: (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8),
    3: (4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8),
    4: (8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    5: (4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    6: (4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5),
    7: (4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5),
}

ALL_CODECS = frozenset(CODEC_NAMES)
SUPPORTED_REBUILD_CODECS = frozenset({1, 3, 4, 5, 6, 7})


class BcpackError(ValueError):
    """Raised when a BCPack stream fails structural or rANS validation."""


class BcpackProvisionalError(BcpackError):
    """Raised when strict decoding encounters an unconfirmed or unknown history entry."""


class BcpackUnsupportedCodec(NotImplementedError):
    """Raised when a BCPack codec has no block rebuilder yet."""


class _Model:
    """16-bit rANS banded frequency model for one (symbol_width, q) pair."""

    __slots__ = ("w", "freq", "base")

    def __init__(self, w, freq):
        self.w = w
        self.freq = freq
        base = []
        acc = 0
        for p, f in enumerate(freq):
            base.append(acc)
            acc += f * (1 if p == 0 else 1 << (p - 1))
        if acc != 65536:
            raise BcpackError(f"probability table ({w}) does not sum to 65536 (got {acc})")
        self.base = tuple(base)

    def lookup(self, cf):
        p = self.w
        base = self.base
        while p > 0 and cf < base[p]:
            p -= 1
        return 0 if p == 0 else (1 << (p - 1)) + (cf - base[p]) // self.freq[p]

    def advance(self, s, x):
        p = s.bit_length()
        f = self.freq[p]
        c = self.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if c == 0:
            f, c = f - 1, 1
        return f * (x >> 16) + (x & 0xFFFF) - c


_MODEL_CACHE = {}


def _get_model(w, q):
    key = (w, q)
    m = _MODEL_CACHE.get(key)
    if m is None:
        freq = _tables.PROB_TABLES.get(key)
        if freq is None:
            return None
        m = _Model(w, freq)
        _MODEL_CACHE[key] = m
    return m


def _decode_lanes_raw(raw, widths, nsym):
    """Decode the interleaved rANS lane array into one symbol list per lane.

    Returns `(lanes, bytes_used)`. Raises `BcpackError` if any fragment fails
    to land at the exact rANS lower bound `RANS_L`, runs out of input, or
    produces an out-of-range symbol.
    """
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [nsym <= 0] * nl
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

    def fail(msg, lane):
        raise BcpackError(
            f"lane {lane} (width {w[lane]}, q {q[lane]}) at input byte {pos}: {msg}"
        )

    while True:
        if all(done[i] and not slotfull[i] for i in range(nl)):
            return out, pos

        # Phase 1: state update, prefetch arbitration (at most 8 bytes/cycle),
        # and symbol release for non-stalled lanes.
        left = PREFETCH_BUDGET
        for i in range(nl):
            if done[i]:
                stall[i] = True
                continue
            if held[i] and slotfull[i]:
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
                    fail("input ran out", i)
                inbuf[i] = raw[pos]
                pos += 1
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False

        # Phase 2: header parsing, byte merge, virtual zero fill on flush, and symbol lookup.
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
                wi = w[i]
                if not 1 <= wi <= MAX_WIDTH:
                    fail("symbol width out of range", i)
                m = _get_model(wi, q[i])
                if m is None:
                    fail("probability table for this width and q is not known", i)
                model[i] = m
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
                                while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        if st == 0 or vz[i] >= MAX_VIRTUAL_ZEROS:
                            fail("fragment did not end at its home state", i)
                        while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
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
                    s = model[i].lookup(st & 0xFFFF)
                    sym[i] = s
                    out_rem[i] -= 1
                    if s >> w[i]:
                        fail("symbol outside its width", i)
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
                        elif st > L and not nxt:
                            fail("fragment did not end at its home state", i)
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
                elif st == L or (out_rem[i] == 0 and not nxt and st > L):
                    fail("fragment did not end at its home state", i)

        # Phase 3: emit one synchronized element when every active lane has filled its slot.
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
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False


_NATIVE = []


def _native_unpack():
    """`(unpack_model, tables)` of the optional Rust lane decoder, or None.

    `tools/bcpack_calibration` (built with Cargo, see its README.md) exports
    `unpack_model`, a port of `_decode_lanes_raw` that is fed this module's
    probability tables, so it accepts the same streams and gives the same
    symbols. Set `LEONIDA_BCPACK_NATIVE=0` to use Python only.
    """
    if not _NATIVE:
        found = None
        if os.environ.get("LEONIDA_BCPACK_NATIVE", "1") != "0":
            import ctypes
            name = {"win32": "bcpack_calibration.dll",
                    "darwin": "libbcpack_calibration.dylib"}.get(sys.platform, "libbcpack_calibration.so")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "bcpack_calibration", "target", "release", name)
            if os.path.isfile(path):
                try:
                    lib = ctypes.CDLL(path)
                    fn = lib.unpack_model
                    blocks = lib.decode_blocks
                except (OSError, AttributeError):
                    fn = None
                if fn is not None:
                    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
                    fn.restype = ctypes.c_ssize_t
                    side = MAX_WIDTH + 1
                    flat = [-1] * (side * 16 * side)
                    for (w, q), freq in _tables.PROB_TABLES.items():
                        if 0 <= w <= MAX_WIDTH and 0 <= q < 16 and len(freq) == w + 1:
                            flat[(w * 16 + q) * side:(w * 16 + q) * side + w + 1] = freq
                    blocks.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_char_p,
                                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t]
                    blocks.restype = ctypes.c_ssize_t
                    found = fn, (ctypes.c_int32 * len(flat))(*flat), blocks
        _NATIVE.append(found)
    return _NATIVE[0]


def _decode_lanes_native(raw, widths, nsym):
    """`(lanes, bytes_used)` from the Rust port of `_decode_lanes_raw`, or None.

    None means the library is not built or the stream is refused (or is one
    the port leaves to Python); `decode_lanes` then runs the Python model,
    which raises its usual error. `tests/test_bcpack_native.py` checks that the
    two agree.
    """
    native = _native_unpack()
    if native is None or nsym <= 0:
        return None
    import ctypes
    fn, tables, _ = native
    nl = len(widths)
    data = bytes(raw)
    out = (ctypes.c_uint16 * (nsym * nl))()
    used = fn(data, len(data), bytes(widths), nl, nsym, tables, len(tables), out)
    if used < 0:
        return None
    flat = memoryview(out).cast("B").cast("H")
    return [flat[i * nsym:(i + 1) * nsym].tolist() for i in range(nl)], used


_NATIVE_BLOCKS = {
    # codec: (lookup table bytes, confirmed defaults, provisional defaults, values per entry)
    1: (lambda: bytes(_tables.BC1_QUADRANT), "BC1_HISTORY_CONFIRMED", "BC1_HISTORY_PROVISIONAL", 6),
    3: (lambda: bytes(v for pair in _tables.BC3_ALPHA_PAIR for v in pair) + bytes(_tables.BC3_COLOUR_QUADRANT),
        "BC3_HISTORY_CONFIRMED", "BC3_HISTORY_PROVISIONAL", 8),
    4: (lambda: bytes(v for pair in _tables.BC4_PAIR for v in pair), "BC4_HISTORY_CONFIRMED",
        "BC4_HISTORY_PROVISIONAL", 2),
    5: (lambda: bytes(v for pair in _tables.BC5_PAIR for v in pair), "BC5_HISTORY_CONFIRMED",
        "BC5_HISTORY_PROVISIONAL", 4),
}
_NATIVE_BUILTIN = {}


def _decode_blocks_native(raw, codec, out_size, history):
    """BC1/BC3/BC4/BC5 bytes from the Rust lanes + strict rebuild, or None to decode in Python.

    Rust only takes the strict path: a stream it refuses, a non-zero lane 1 or
    a history entry that is not confirmed returns None, and Python decodes the
    chunk itself, so its errors and provisional entries are unchanged.
    """
    native = _native_unpack()
    spec = _NATIVE_BLOCKS.get(codec)
    if native is None or spec is None or out_size <= 0 or out_size % BLOCK_BYTES[codec]:
        return None
    import ctypes
    _, tables, blocks = native
    lut, conf_name, prov_name, k = spec
    cached = _NATIVE_BUILTIN.get(codec) if history is None else None
    if cached is None:
        try:
            confirmed, _ = _resolve_bc_history(
                codec, getattr(_tables, conf_name), getattr(_tables, prov_name), history)
        except (ValueError, TypeError, OSError):
            return None  # Python reads the same history after the lanes, so its error order is unchanged
        flat = [-1] * (256 * k)
        for entry, values in confirmed.items():
            if 0 <= entry < 256 and len(values) == k:
                flat[entry * k:entry * k + k] = values
        cached = lut(), (ctypes.c_int32 * len(flat))(*flat)
        if history is None:
            _NATIVE_BUILTIN[codec] = cached
    table, builtin = cached
    widths = LANE_WIDTHS[codec]
    data = bytes(raw)
    out = ctypes.create_string_buffer(out_size)
    used = blocks(data, len(data), codec, bytes(widths), len(widths), out_size // ELEMENT_BYTES,
                  tables, len(tables), table, len(table), builtin, len(builtin), out, out_size)
    return out.raw if used >= 0 else None


def decode_lanes(raw, codec, out_size):
    """Entropy layer: decode `raw` into per-lane symbol lists and bytes consumed.

    Works for all six active codecs (`1, 3, 4, 5, 6, 7`). Trailing non-zero
    bytes after the last lane finishes are rejected. Uses the Rust decoder when
    it is built (`_native_unpack`) and Python for the streams it refuses.
    """
    if codec not in LANE_WIDTHS:
        raise BcpackError(f"unsupported BCPack codec id: {codec}")
    if out_size < 0 or out_size % BLOCK_BYTES[codec] != 0:
        raise BcpackError(
            f"decoded size {out_size} is not a whole number of {CODEC_NAMES[codec]} blocks"
        )
    nsym = out_size // ELEMENT_BYTES
    fast = _decode_lanes_native(raw, LANE_WIDTHS[codec], nsym)
    if fast is not None:
        return fast
    data = memoryview(raw)
    lanes, used = _decode_lanes_raw(data, LANE_WIDTHS[codec], nsym)
    if any(data[used:]):
        raise BcpackError("non-zero data left over after the last lane finished")
    return lanes, used


def _unzig(v):
    return -((v + 1) >> 1) if (v & 1) else (v >> 1)


def _inter(a, b, n):
    d = 0
    for k in range(n):
        d |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1))
    return d


def _inter8(a, b):
    return _inter(a, b, 4)


# 4-bit de-interleave lookup tables for 8-bit index-pair symbols (even / odd bits).
_DEINT = (
    tuple(
        ((s >> 0) & 1) | (((s >> 2) & 1) << 1) | (((s >> 4) & 1) << 2) | (((s >> 6) & 1) << 3)
        for s in range(256)
    ),
    tuple(
        ((s >> 1) & 1) | (((s >> 3) & 1) << 1) | (((s >> 5) & 1) << 2) | (((s >> 7) & 1) << 3)
        for s in range(256)
    ),
)


def _loose_limit(nblocks):
    """Maximum undetermined blocks permitted in provisional mode (1 in 1,024, min 2)."""
    return max(2, nblocks // 1024)


def _unknown_blocks(e_roots, i_roots, unknown_e, unknown_i):
    """Blocks whose endpoint or index history root is an unknown entry. A block can depend on one of each, and
    it is one undetermined block, not two, against `_loose_limit`."""
    return sum(1 for e, i in zip(e_roots, i_roots) if e in unknown_e or i in unknown_i)


def _resolve_bc_history(codec, default_confirmed, default_provisional, history):
    """Merge or replace BC1/BC3/BC4/BC5 history tables when external `history` is provided."""
    loaded = _bptc.load_bc_history(codec, history)
    if loaded is None:
        return default_confirmed, default_provisional
    ext_conf, ext_prov, replace = loaded
    if replace:
        confirmed = dict(ext_conf)
        loose = {k: v for k, v in ext_prov.items() if k not in confirmed}
        return confirmed, loose
    confirmed = dict(default_confirmed)
    confirmed.update(ext_conf)
    loose = {k: v for k, v in default_provisional.items() if k not in confirmed}
    for k, v in ext_prov.items():
        if k not in confirmed:
            loose[k] = v
    return confirmed, loose


def _bc1_blocks(lanes, nblocks, provisional, *, history=None):
    if any(lanes[1]):
        raise BcpackError("BC1 lane 1 is non-zero (unexpected)")
    quad = _tables.BC1_QUADRANT
    confirmed, loose = _resolve_bc_history(
        1, _tables.BC1_HISTORY_CONFIRMED, _tables.BC1_HISTORY_PROVISIONAL, history
    )
    out = bytearray(nblocks * 8)
    hist = [None] * nblocks
    mods = (32, 64, 32, 32, 64, 32)
    l0 = lanes[0]
    cols = lanes[2:8]
    q0, q1, q2, q3 = lanes[8:12]
    for i in range(nblocks):
        v = l0[i]
        d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
        if d == 0:
            base = (0, 0, 0, 0, 0, 0)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None:
                if provisional is not None and entry in loose:
                    base = loose[entry]
                    provisional.add(f"BC1 history {entry}")
                else:
                    state = "unconfirmed" if entry in loose else "unknown"
                    raise BcpackProvisionalError(
                        f"BC1 block {i} references {state} built-in history entry {entry}"
                    )
        e0 = (base[0] + _unzig(cols[0][i])) % mods[0]
        e1 = (base[1] + _unzig(cols[1][i])) % mods[1]
        e2 = (base[2] + _unzig(cols[2][i])) % mods[2]
        e3 = (base[3] + _unzig(cols[3][i])) % mods[3]
        e4 = (base[4] + _unzig(cols[4][i])) % mods[4]
        e5 = (base[5] + _unzig(cols[5][i])) % mods[5]
        hist[i] = (e0, e1, e2, e3, e4, e5)
        c0 = (e0 << 11) | (e1 << 5) | e2
        c1 = (e3 << 11) | (e4 << 5) | e5
        a, b, c, dd = quad[q0[i]], quad[q1[i]], quad[q2[i]], quad[q3[i]]
        # Quadrant byte: pixels TL, TR, BL, BR in bits 0-1, 2-3, 4-5, 6-7.
        struct.pack_into(
            "<HHBBBB",
            out,
            i * 8,
            c0,
            c1,
            (a & 15) | ((b & 15) << 4),
            (a >> 4) | ((b >> 4) << 4),
            (c & 15) | ((dd & 15) << 4),
            (c >> 4) | ((dd >> 4) << 4),
        )
    return bytes(out)


_BC3_MODS = (256, 256, 32, 64, 32, 32, 64, 32)
_BC3_NONE = (128, 128, 0, 0, 0, 0, 0, 0)


def _bc3_blocks(lanes, nblocks, provisional, *, history=None):
    if any(lanes[1]):
        raise BcpackError("BC3 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC3_ALPHA_PAIR
    quads = _tables.BC3_COLOUR_QUADRANT
    confirmed, loose = _resolve_bc_history(
        3, _tables.BC3_HISTORY_CONFIRMED, _tables.BC3_HISTORY_PROVISIONAL, history
    )
    out = bytearray(nblocks * 16)
    hist = [None] * nblocks
    lost = [False] * nblocks
    l0, alpha = lanes[0], lanes[2]
    rows = lanes[3:7]
    lr, lg, lb = lanes[7], lanes[8], lanes[9]
    top, bottom = lanes[10], lanes[11]
    unknown = set()
    mods = _BC3_MODS
    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        d = _unzig(_inter8(l0[a], l0[b])) & 255
        if d == 0:
            base = _BC3_NONE
        elif i - d >= 0:
            base = hist[i - d]
            lost[i] = lost[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None and entry in loose:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC3 block {i} references unconfirmed built-in history entry {entry}"
                    )
                base = loose[entry]
                provisional.add(f"BC3 history {entry}")
            if base is None:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC3 block {i} references unknown built-in history entry {entry}"
                    )
                base = _BC3_NONE
                lost[i] = True
                unknown.add(entry)
        dz = (alpha[a], alpha[b], lr[a], lg[a], lb[a], lr[b], lg[b], lb[b])
        e = tuple((base[k] + _unzig(dz[k])) % mods[k] for k in range(8))
        hist[i] = e
        bits = 0
        for r in range(4):
            left, right = pairs[rows[r][a]], pairs[rows[r][b]]
            bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
        at = i * 16
        out[at] = e[0]
        out[at + 1] = e[1]
        out[at + 2:at + 8] = bits.to_bytes(6, "little")
        c0 = (e[2] << 11) | (e[3] << 5) | e[4]
        c1 = (e[5] << 11) | (e[6] << 5) | e[7]
        qa, qb = quads[top[a]], quads[top[b]]
        qc, qd = quads[bottom[a]], quads[bottom[b]]
        struct.pack_into(
            "<HHBBBB",
            out,
            at + 8,
            c0,
            c1,
            (qa & 15) | ((qb & 15) << 4),
            (qa >> 4) | ((qb >> 4) << 4),
            (qc & 15) | ((qd & 15) << 4),
            (qc >> 4) | ((qd >> 4) << 4),
        )
    if unknown:
        total = sum(lost)
        if total > _loose_limit(nblocks):
            raise BcpackProvisionalError(
                f"BC3 chunk has {total}/{nblocks} blocks from unknown built-in history entries"
            )
        provisional.add(f"BC3 unknown history {sorted(unknown)} ({total} blocks)")
    return bytes(out)


def _bc4_blocks(lanes, nblocks, provisional, *, history=None):
    if any(lanes[1]):
        raise BcpackError("BC4 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC4_PAIR
    confirmed, loose = _resolve_bc_history(
        4, _tables.BC4_HISTORY_CONFIRMED, _tables.BC4_HISTORY_PROVISIONAL, history
    )
    out = bytearray(nblocks * 8)
    hist = [None] * nblocks
    l0, l2, l3 = lanes[0], lanes[2], lanes[3]
    rows = lanes[4:12]
    for i in range(nblocks):
        v = l0[i]
        d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
        if d == 0:
            base = (128, 128)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None:
                if provisional is not None and entry in loose:
                    base = loose[entry]
                    provisional.add(f"BC4 history {entry}")
                else:
                    state = "unconfirmed" if entry in loose else "unknown"
                    raise BcpackProvisionalError(
                        f"BC4 block {i} references {state} built-in history entry {entry}"
                    )
        a0 = (base[0] + _unzig(l2[i])) & 255
        a1 = (base[1] + _unzig(l3[i])) & 255
        hist[i] = (a0, a1)
        bits = 0
        for r in range(4):
            left, right = pairs[rows[2 * r][i]], pairs[rows[2 * r + 1][i]]
            bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
        at = i * 8
        out[at] = a0
        out[at + 1] = a1
        out[at + 2:at + 8] = bits.to_bytes(6, "little")
    return bytes(out)


def _bc5_blocks(lanes, nblocks, provisional, *, history=None):
    if any(lanes[1]):
        raise BcpackError("BC5 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC5_PAIR
    confirmed, loose = _resolve_bc_history(
        5, _tables.BC5_HISTORY_CONFIRMED, _tables.BC5_HISTORY_PROVISIONAL, history
    )
    out = bytearray(nblocks * 16)
    hist = [None] * nblocks
    lost = [False] * nblocks
    l0, red, green = lanes[0], lanes[2], lanes[3]
    rows = lanes[4:12]
    unknown = set()
    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        d = _unzig(_inter8(l0[a], l0[b])) & 255
        if d == 0:
            base = (128, 128, 128, 128)
        elif i - d >= 0:
            base = hist[i - d]
            lost[i] = lost[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None and entry in loose:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC5 block {i} references unconfirmed built-in history entry {entry}"
                    )
                base = loose[entry]
                provisional.add(f"BC5 history {entry}")
            if base is None:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC5 block {i} references unknown built-in history entry {entry}"
                    )
                base = (128, 128, 128, 128)
                lost[i] = True
                unknown.add(entry)
        e = (
            (base[0] + _unzig(red[a])) & 255,
            (base[1] + _unzig(red[b])) & 255,
            (base[2] + _unzig(green[a])) & 255,
            (base[3] + _unzig(green[b])) & 255,
        )
        hist[i] = e
        at = i * 16
        for ch in range(2):
            bits = 0
            for r in range(4):
                left, right = pairs[rows[4 * ch + r][a]], pairs[rows[4 * ch + r][b]]
                bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
            out[at + 8 * ch] = e[2 * ch]
            out[at + 8 * ch + 1] = e[2 * ch + 1]
            out[at + 8 * ch + 2:at + 8 * ch + 8] = bits.to_bytes(6, "little")
    if unknown:
        total = sum(lost)
        if total > _loose_limit(nblocks):
            raise BcpackProvisionalError(
                f"BC5 chunk has {total}/{nblocks} blocks from unknown built-in history entries"
            )
        provisional.add(f"BC5 unknown history {sorted(unknown)} ({total} blocks)")
    return bytes(out)


# BC6H ------------------------------------------------------------------------

# Mode index (1..14, where mode = lane15[2*i] + 1) -> D3D11/Khronos mode bits.
_BC6H_CODE = (
    0, 0x00, 0x01, 0x02, 0x06, 0x0A, 0x0E, 0x12,
    0x16, 0x1A, 0x1E, 0x03, 0x07, 0x0B, 0x0F,
)

# Mode code -> (transformed, base_bits, (dr, dg, db), regions).
_BC6H_INFO = {
    0x00: (1, 10, (5, 5, 5), 2),
    0x01: (1, 7, (6, 6, 6), 2),
    0x02: (1, 11, (5, 4, 4), 2),
    0x06: (1, 11, (4, 5, 4), 2),
    0x0A: (1, 11, (4, 4, 5), 2),
    0x0E: (1, 9, (5, 5, 5), 2),
    0x12: (1, 8, (6, 5, 5), 2),
    0x16: (1, 8, (5, 6, 5), 2),
    0x1A: (1, 8, (5, 5, 6), 2),
    0x1E: (0, 6, (6, 6, 6), 2),
    0x03: (0, 10, (10, 10, 10), 1),
    0x07: (1, 11, (9, 9, 9), 1),
    0x0B: (1, 12, (8, 8, 8), 1),
    0x0F: (1, 16, (4, 4, 4), 1),
}

# Standard D3D11/Khronos BC6H header bit layouts (LSB-first after the 2- or 5-bit mode).
_BC6H_LAYOUT = {
    0x00: "gy4 by4 bz4 rw10 gw10 bw10 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x01: "gy5 gz4 gz5 rw7 bz0 bz1 by4 gw7 by5 bz2 gy4 bw7 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d",
    0x02: "rw10 gw10 bw10 rx5 rw:10 gy4_ gx4 gw:10 bz0 gz4_ bx4 bw:10 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x06: "rw10 gw10 bw10 rx4 rw:10 gz4 gy4_ gx5 gw:10 gz4_ bx4 bw:10 bz1 by4_ ry4 bz0 bz2 rz4 gy4 bz3 d",
    0x0A: "rw10 gw10 bw10 rx4 rw:10 by4 gy4_ gx4 gw:10 bz0 gz4_ bx5 bw:10 by4_ ry4 bz1 bz2 rz4 bz4 bz3 d",
    0x0E: "rw9 by4 gw9 gy4 bw9 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x12: "rw8 gz4 by4 gw8 bz2 gy4 bw8 bz3 bz4 rx6 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry6 rz6 d",
    0x16: "rw8 bz0 by4 gw8 gy5 gy4 bw8 gz5 bz4 rx5 gz4 gy4_ gx6 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x1A: "rw8 bz1 by4 gw8 by5 gy4 bw8 bz5 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx6 by4_ ry5 bz2 rz5 bz3 d",
    0x1E: "rw6 gz4 bz0 bz1 by4 gw6 gy5 by5 bz2 gy4 bw6 gz5 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d",
    0x03: "rw10 gw10 bw10 rx10 gx10 bx10",
    0x07: "rw10 gw10 bw10 rx9 rw:10 gx9 gw:10 bx9 bw:10",
    0x0B: "rw10 gw10 bw10 rx8 rw:11-10 gx8 gw:11-10 bx8 bw:11-10",
    0x0F: "rw10 gw10 bw10 rx4 rw:15-10 gx4 gw:15-10 bx4 bw:15-10",
}

_BC6H_FIELD_MAP = {
    "rw": (0, 0), "gw": (1, 0), "bw": (2, 0),
    "rx": (0, 1), "gx": (1, 1), "bx": (2, 1),
    "ry": (0, 2), "gy": (1, 2), "by": (2, 2),
    "rz": (0, 3), "gz": (1, 3), "bz": (2, 3),
}

_BC6H_NONE_E = ((0x8000, 0x200, 0x20, 0x20),) * 3
_BC6H_NONE_I = (8,) * 16
_BC6H_PLANS = {}


def _bc6h_plan(code):
    """Cached `(plan, header_bits)` for a BC6H mode code.

    Each plan item is `(channel, endpoint_slot, bit_index, stream_pos)` with
    `channel == -1` for the 5-bit partition field `d`.
    """
    cached = _BC6H_PLANS.get(code)
    if cached is not None:
        return cached
    pos = 2 if code < 2 else 5
    plan = []
    for token in _BC6H_LAYOUT[code].split():
        if token == "d":
            for b in range(5):
                plan.append((-1, -1, b, pos + b))
            pos += 5
        elif ":" in token:
            name, span = token.split(":")
            c, q = _BC6H_FIELD_MAP[name]
            hi, lo = (int(x) for x in span.split("-")) if "-" in span else (int(span), int(span))
            for b in range(hi, lo - 1, -1):
                plan.append((c, q, b, pos))
                pos += 1
        elif token.endswith("_"):
            name, n = token[:2], int(token[2:-1])
            c, q = _BC6H_FIELD_MAP[name]
            for b in range(n):
                plan.append((c, q, b, pos + b))
            pos += n
        else:
            name, n = token[:2], int(token[2:])
            c, q = _BC6H_FIELD_MAP[name]
            if name[1] in "wx" or name in ("ry", "rz"):
                for b in range(n):
                    plan.append((c, q, b, pos + b))
                pos += n
            else:
                plan.append((c, q, n, pos))
                pos += 1
    res = (tuple(plan), pos)
    _BC6H_PLANS[code] = res
    return res


def _bc6h_stream_w(mode, w):
    """Map BCPack's base endpoint `w` into the D3D11/Khronos BC6H bit order."""
    if mode == 13:
        return (w & 0x3FF) | (((w >> 10) & 1) << 11) | (((w >> 11) & 1) << 10)
    if mode == 14:
        out = w & 0x3FF
        for b in range(10, 16):
            out |= ((w >> b) & 1) << (25 - b)
        return out
    return w


def _bc6h_blocks(lanes, nblocks, provisional, *, history=None):
    """Rebuild `nblocks` 16-byte BC6H blocks from the 16 BCPack lanes.

    With `history=None`, no built-in BC6H initial history is loaded, so strict
    mode (`provisional is None`) accepts only self-contained chunks whose delta
    chains stay within the chunk (`d == 0` or `d <= i`). Pass a locally
    generated `history` JSON file/directory to supply built-in history entries.
    """
    if len(lanes[0]) != 2 * nblocks:
        raise BcpackError("BC6H lane length does not match the block count")
    eh_table, ih_table, eclass, iclass, ep_cb, ix_cb = _bptc.load_bc6h(history, with_confirmed_bits=True)
    anchor2 = _bptc.BC7_ANCHOR2
    out = bytearray(nblocks * 16)
    st_e = [None] * nblocks
    st_i = [None] * nblocks
    eroot = [-1] * nblocks
    iroot = [-1] * nblocks
    ends = {}
    idxs = {}
    weak_e = set()
    weak_i = set()
    violations = 0
    l0, l1, l15 = lanes[0], lanes[1], lanes[15]
    deint0, deint1 = _DEINT

    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        m_sym = l15[a]
        if m_sym > 13:
            raise BcpackError(f"BC6H block {i} mode symbol {m_sym} outside 0..13")
        mode = m_sym + 1
        code = _BC6H_CODE[mode]
        trans, base, deltas, reg = _BC6H_INFO[code]
        part = l15[b] if l15[b] < 31 else 31

        di = _inter8(l1[a], l1[b])
        de = (di + _unzig(_inter8(l0[a], l0[b]))) & 255
        if de == 0:
            eref = _BC6H_NONE_E
        elif i >= de:
            eref = st_e[i - de]
            eroot[i] = eroot[i - de]
        else:
            entry = 256 + i - de
            eref = eh_table[entry]
            eroot[i] = entry
            # Only direct initial-history reads consume the calibrated bits.
            # Later blocks read the reconstructed state, whose truncation and
            # zero padding are determined by this block's mode and deltas.
            for c in range(3):
                dc = deltas[c]
                need = (base, dc, dc, dc) if reg == 2 else (base, dc, 0, 0)
                if any(ep_cb[entry][c][q] < n for q, n in enumerate(need)):
                    weak_e.add(entry)
        if eroot[i] >= 0:
            r = eroot[i]
            ends[r] = ends.get(r, 0) + 1

        if di == 0:
            iref = _BC6H_NONE_I
        elif i >= di:
            iref = st_i[i - di]
            iroot[i] = iroot[i - di]
        else:
            entry = 256 + i - di
            iref = ih_table[entry]
            iroot[i] = entry
            if any(b < (3 if reg == 2 else 4) for b in ix_cb[entry]):
                weak_i.add(entry)
        if iroot[i] >= 0:
            r = iroot[i]
            idxs[r] = idxs.get(r, 0) + 1

        mw = (1 << base) - 1
        w_shift = 16 - base
        f_block = [None, None, None]
        st_block = [None, None, None]
        for c in range(3):
            dc = deltas[c]
            mD = (1 << dc) - 1
            sh = 6 - dc if dc < 6 else 0
            dz0 = _unzig(_inter(lanes[2 + c][a], lanes[2 + c][b], 8))
            dz1 = _unzig(_inter(lanes[5 + c][a], lanes[5 + c][b], 5))
            dz2 = _unzig(lanes[8 + c][a])
            dz3 = _unzig(lanes[8 + c][b])
            r0, r1, r2, r3 = eref[c]
            w = ((r0 >> w_shift) + dz0) & mw
            x = ((r1 >> (10 - dc)) + dz1) & mD
            y = ((r2 >> sh) + dz2) & mD
            z = ((r3 >> sh) + dz3) & mD
            x10 = x << (10 - dc)
            if reg == 2:
                st_block[c] = (w << w_shift, x10, y << sh, z << sh)
                f_block[c] = (_bc6h_stream_w(mode, w), x, y, z)
            else:
                v = (w & mD) ^ ((1 << (dc - 1)) if trans == 1 else 0)
                low_y = (v >> (dc - 6)) if dc >= 6 else (v << sh)
                st_block[c] = (w << w_shift, x10, low_y, x10 >> 4)
                f_block[c] = (_bc6h_stream_w(mode, w), x, 0, 0)
        st_e[i] = (st_block[0], st_block[1], st_block[2])

        ibits = 3 if reg == 2 else 4
        imask = (1 << ibits) - 1
        ishift = 4 - ibits
        ix_block = [0] * 16
        si_block = [0] * 16
        for yy in range(4):
            sa = lanes[11 + yy][a]
            sb = lanes[11 + yy][b]
            row = yy * 4
            if yy < 3:
                d0, d1 = deint1[sa], deint0[sa]
                d2, d3 = deint1[sb], deint0[sb]
            else:
                d0, d1 = deint0[sa], deint1[sa]
                d2, d3 = deint0[sb], deint1[sb]
            for off, dv in ((0, d0), (1, d1), (2, d2), (3, d3)):
                p = row + off
                ix = ((iref[p] >> ishift) + _unzig(dv)) & imask
                ix_block[p] = ix
                si_block[p] = ix << ishift
        st_i[i] = tuple(si_block)

        second = anchor2[part] if reg == 2 else -1
        if ix_block[0] & (4 if reg == 2 else 8):
            violations += 1
        if reg == 2 and (ix_block[second] & 4):
            violations += 1

        plan, header = _bc6h_plan(code)
        acc = code
        for c, q, bit_idx, pos in plan:
            if c < 0:
                if (part >> bit_idx) & 1:
                    acc |= 1 << pos
            elif (f_block[c][q] >> bit_idx) & 1:
                acc |= 1 << pos
        at = header
        for p in range(16):
            nb = (ibits - 1) if (p == 0 or p == second) else ibits
            acc |= (ix_block[p] & ((1 << nb) - 1)) << at
            at += nb
        out[16 * i:16 * i + 16] = acc.to_bytes(16, "little")

    if provisional is None:
        unconf_e = [k for k in ends if eclass[k] != 3 or k in weak_e]
        unconf_i = [k for k in idxs if iclass[k] != 3 or k in weak_i]
        if unconf_e or unconf_i:
            k = unconf_e[0] if unconf_e else unconf_i[0]
            kind = "endpoint" if unconf_e else "index"
            cls = eclass[k] if unconf_e else iclass[k]
            state = "unknown" if cls == 0 else "unconfirmed"
            hint = " (pass --history <path>)" if history is None else ""
            raise BcpackProvisionalError(
                f"BC6H chunk references {state} built-in {kind} history entry {k}{hint}"
            )
        if violations:
            raise BcpackError(
                f"BC6H chunk has {violations} anchor pixel(s) with an inconsistent index"
            )
        return bytes(out)

    lost = _unknown_blocks(eroot, iroot, {k for k in ends if eclass[k] == 0}, {k for k in idxs if iclass[k] == 0})
    missing = sorted(
        {f"endpoint {k}" for k in ends if eclass[k] == 0}
        | {f"index {k}" for k in idxs if iclass[k] == 0}
    )
    if lost > _loose_limit(nblocks):
        hint = " (pass --history <path>)" if history is None else ""
        raise BcpackProvisionalError(
            f"BC6H chunk has {lost}/{nblocks} blocks from unknown built-in history entries "
            f"({', '.join(missing[:6])}{', ...' if len(missing) > 6 else ''}){hint}"
        )
    if violations > _loose_limit(nblocks):
        raise BcpackProvisionalError(
            f"BC6H chunk has {violations} anchor pixels with an inconsistent index"
        )
    if missing:
        provisional.add(f"BC6H unknown history {', '.join(missing)} ({lost} blocks)")
    if violations:
        provisional.add(f"BC6H anchor index inconsistent in {violations} pixels")
    for kind, entries, cls in (("endpoint", weak_e, eclass), ("index", weak_i, iclass)):
        for k in sorted(entries):
            if cls[k] == 3:
                provisional.add(f"BC6H {kind} history {k} (unconfirmed bits)")
    for name, entries, cls in (("endpoint", ends, eclass), ("index", idxs, iclass)):
        fitted = sorted(k for k in entries if cls[k] == 1)
        rough = sorted(k for k in entries if cls[k] == 2)
        if fitted:
            provisional.add(f"BC6H {name} history {', '.join(map(str, fitted))}")
        if rough:
            provisional.add(f"BC6H {name} history {', '.join(map(str, rough))} (last-resort)")
    return bytes(out)


# BC7 -------------------------------------------------------------------------

_B7_NS = (3, 2, 3, 2, 1, 1, 1, 2)
_B7_CB = (4, 6, 5, 7, 5, 7, 7, 5)
_B7_AB = (0, 0, 0, 0, 6, 8, 7, 5)
_B7_PK = (1, 2, 0, 1, 0, 0, 1, 1)      # 0=none, 1=per-endpoint p-bit, 2=shared per subset
_B7_IB = (3, 3, 2, 2, 3, 2, 4, 2)      # index bits of primary index set (lanes 11..14)
_B7_PB = (4, 6, 6, 6, 0, 0, 0, 6)      # partition field bits
_B7_LANE = ((2, 3, 4, 5), (6, 7, 8, 9), (10, 5, 9, -1))
_B7_ANCHOR = 8
_B7_PBIT = 16
_B7_SIGNATURE = 32
_B7_MIN_DEPTH = 4


def _bc7_field_bits(mode, rot):
    """Leading bits read from a stored endpoint byte for channels (R, G, B, A)."""
    w = [_B7_CB[mode], _B7_CB[mode], _B7_CB[mode], _B7_AB[mode]]
    if rot:
        w[rot - 1], w[3] = _B7_AB[mode], _B7_CB[mode]
    pk = _B7_PK[mode]
    return [w[c] + (1 if (pk and c == 0) else 0) if w[c] else 0 for c in range(4)]


def _bc7_check(lanes, nblocks, table, provisional, *, has_history=False):
    """Validate signatures and built-in history coverage for a BC7 chunk.

    In strict mode (`provisional is None`), raises `BcpackError` for unknown
    signatures and `BcpackProvisionalError` for inferred signatures or any
    unconfirmed/unknown built-in history entry. In provisional mode, populates
    `provisional` with descriptive labels and returns a list of `(kind, entry)`
    pairs whose depth is below `_B7_MIN_DEPTH` (or missing for indices).
    """
    cfg = table["config"]
    rank_of = table["rank_of"]
    l0, l1, l15 = lanes[0], lanes[1], lanes[15]
    ranks = set()
    for a, b in set(zip(l15[0::2], l15[1::2])):
        rank = rank_of[a * 16 + b] if a < 32 and b < 16 else -1
        if rank < 0 or cfg[rank][0] > 7 or cfg[rank][1] == 255:
            raise BcpackError(f"BC7 signature ({a}, {b}) is outside the signature table")
        ranks.add(rank)
    for rank in sorted(ranks & table["inferred"]):
        if provisional is None:
            raise BcpackProvisionalError(
                f"BC7 signature rank {rank} is inferred-only (unconfirmed)"
            )
        provisional.add(f"BC7 signature {rank}")

    need_e = {}
    need_i = {}
    for i in range(min(nblocks, 256)):
        rank = rank_of[l15[2 * i] * 16 + l15[2 * i + 1]]
        mode, _, rot, _ = cfg[rank]
        di = _inter8(l1[2 * i], l1[2 * i + 1])
        de = (di + _unzig(_inter8(l0[2 * i], l0[2 * i + 1]))) & 255
        if de > i:
            bits = _bc7_field_bits(mode, rot)
            row = need_e.setdefault(256 + i - de, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            for sub in range(_B7_NS[mode]):
                for c in range(4):
                    if bits[c] > row[sub][c]:
                        row[sub][c] = bits[c]
        if di > i:
            row = need_i.setdefault(256 + i - di, [0, 0])
            if _B7_IB[mode] > row[0]:
                row[0] = _B7_IB[mode]
            if mode in (4, 5):
                row[1] = 2

    unknown = []
    last_resort = table["last_resort"]
    checks = (
        ("endpoint", need_e, table["history_endpoint_bits"], table["history_endpoint_confirmed_bits"]),
        ("index", need_i, table["history_index_bits"], table["history_index_confirmed_bits"]),
    )
    for kinds, need, bits_t, conf_t in checks:
        for k, want in sorted(need.items()):
            if kinds == "endpoint":
                slots = [
                    (sub * 8 + e * 4 + c, want[sub][c])
                    for sub in range(3) for e in range(2) for c in range(4)
                    if want[sub][c]
                ]
            else:
                slots = [
                    (st * 16 + p, want[st])
                    for st in range(2) for p in range(16)
                    if want[st]
                ]
            weak = False
            missing = False
            shallow = 8
            bk = bits_t[k]
            ck = conf_t[k]
            for j, b in slots:
                if bk[j] < b:
                    missing = True
                    if bk[j] < shallow:
                        shallow = bk[j]
                if ck[j] < b:
                    weak = True
            if missing:
                if provisional is None:
                    hint = " (pass --history <path>)" if not has_history else ""
                    raise BcpackProvisionalError(
                        f"BC7 chunk references unknown built-in {kinds} history entry {k}{hint}"
                    )
                if kinds == "endpoint" and shallow >= _B7_MIN_DEPTH:
                    provisional.add(f"BC7 endpoint history {k} (top {shallow} bits known)")
                else:
                    unknown.append((kinds, k))
            elif weak:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC7 chunk references unconfirmed built-in {kinds} history entry {k}"
                    )
                provisional.add(f"BC7 {kinds} history {k}")
            if kinds == "endpoint" and provisional is not None:
                if any((k, j) in last_resort for j, _ in slots):
                    provisional.add(f"BC7 endpoint history {k} (last-resort)")
    return unknown


def _bc7_rebuild(lanes, nblocks, table):
    """Rebuild `nblocks` 16-byte BC7 blocks and per-block consistency status flags."""
    cfg = table["config"]
    rank_of = table["rank_of"]
    p2 = table["partition2"]
    p3 = table["partition3"]
    a2 = table["anchor2"]
    a3a = table["anchor3a"]
    a3b = table["anchor3b"]
    pre_e = table["history_endpoints"]
    pre_i = table["history_indices"]
    out = bytearray(16 * nblocks)
    status = [0] * nblocks
    eps = [None] * nblocks
    ixs = [None] * nblocks
    zero_e = (0,) * 24
    zero_i = (0,) * 32
    rows1 = (lanes[11], lanes[12], lanes[13], lanes[14])
    rows2 = (lanes[6], lanes[7], lanes[8], lanes[10])
    l0, l1, l15 = lanes[0], lanes[1], lanes[15]
    deint0, deint1 = _DEINT

    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        s15a, s15b = l15[a], l15[b]
        rank = rank_of[s15a * 16 + s15b] if s15a < 32 and s15b < 16 else -1
        if rank < 0 or cfg[rank][0] > 7:
            status[i] = _B7_SIGNATURE
            eps[i] = zero_e
            ixs[i] = zero_i
            continue
        mode, part, rot, isel = cfg[rank]
        st = 0
        di = _inter8(l1[a], l1[b])
        de = (di + _unzig(_inter8(l0[a], l0[b]))) & 255
        rep = zero_e if de == 0 else (eps[i - de] if i >= de else pre_e[256 + i - de])
        rix = zero_i if di == 0 else (ixs[i - di] if i >= di else pre_i[256 + i - di])

        ns = _B7_NS[mode]
        cb = _B7_CB[mode]
        ab = _B7_AB[mode]
        pk = _B7_PK[mode]
        w = [cb, cb, cb, ab]
        if rot:
            w[rot - 1], w[3] = ab, cb

        val = [[[0, 0, 0, 0], [0, 0, 0, 0]] for _ in range(3)]
        sep = [0] * 24
        pb = [[0, 0], [0, 0], [0, 0]]
        for sub in range(ns):
            lane_row = _B7_LANE[sub]
            base_sub = sub * 8
            for c in range(4):
                lane_idx = lane_row[c]
                wc = w[c]
                if not wc or lane_idx < 0:
                    continue
                bits = wc + (1 if (pk and c == 0) else 0)
                shift = 8 - bits
                mask = (1 << bits) - 1
                ln = lanes[lane_idx]
                v0 = ((rep[base_sub + c] >> shift) + _unzig(ln[a])) & mask
                v1 = ((rep[base_sub + 4 + c] >> shift) + _unzig(ln[b])) & mask
                val[sub][0][c] = v0
                val[sub][1][c] = v1
                sep[base_sub + c] = v0 << shift
                sep[base_sub + 4 + c] = v1 << shift
            if pk == 1:
                pb[sub][0] = val[sub][0][0] & 1
                pb[sub][1] = val[sub][1][0] & 1
            elif pk == 2:
                p0 = val[sub][0][0] & 1
                pb[sub][0] = p0
                pb[sub][1] = p0
                if (val[sub][1][0] & 1) != p0:
                    st |= _B7_PBIT
        for sub in range(ns, 3):
            sep[sub * 8:sub * 8 + 8] = sep[0:8]
        eps[i] = sep

        b1 = _B7_IB[mode]
        b1_shift = 4 - b1
        b1_mask = (1 << b1) - 1
        two = mode in (4, 5)
        s1 = [0] * 16
        s2 = [0] * 16 if two else None
        six = [0] * 32
        for y in range(4):
            r1 = rows1[y]
            sa, sb = r1[a], r1[b]
            row = y * 4
            if y < 3:
                d0, d1, d2, d3 = deint1[sa], deint0[sa], deint1[sb], deint0[sb]
            else:
                d0, d1, d2, d3 = deint0[sa], deint1[sa], deint0[sb], deint1[sb]
            for off, dv in ((0, d0), (1, d1), (2, d2), (3, d3)):
                p = row + off
                iv = ((rix[p] >> b1_shift) + _unzig(dv)) & b1_mask
                s1[p] = iv
                six[p] = iv << b1_shift
            if two:
                r2 = rows2[y]
                s2a, s2b = r2[a], r2[b]
                if y < 3:
                    e0, e1, e2, e3 = deint1[s2a], deint0[s2a], deint1[s2b], deint0[s2b]
                else:
                    e0, e1, e2, e3 = deint0[s2a], deint1[s2a], deint0[s2b], deint1[s2b]
                for off, dv in ((0, e0), (1, e1), (2, e2), (3, e3)):
                    p = row + off
                    iv2 = ((rix[16 + p] >> 2) + _unzig(dv)) & 3
                    s2[p] = iv2
                    six[16 + p] = iv2 << 2
        ixs[i] = six

        if mode == 0 and part == 0 and not any(sep) and not any(s1):
            status[i] = st
            continue

        sm = p2[part] if ns == 2 else (p3[part] if ns == 3 else None)
        anchors = (0,) if ns == 1 else ((0, a2[part]) if ns == 2 else (0, a3a[part], a3b[part]))
        if mode == 4:
            pri, ib, sec, ib2 = s2, 2, s1, 3
        elif mode == 5:
            pri, ib, sec, ib2 = s1, 2, s2, 2
        else:
            pri, ib, sec, ib2 = s1, b1, None, 0

        acc = 1 << mode
        pos = mode + 1
        pbits = _B7_PB[mode]
        if pbits:
            acc |= (part & ((1 << pbits) - 1)) << pos
            pos += pbits
        if mode in (4, 5):
            acc |= (rot & 3) << pos
            pos += 2
        if mode == 4:
            acc |= (isel & 1) << pos
            pos += 1
        cb_mask = (1 << cb) - 1
        for c in range(3):
            tc = 3 if (rot and c == rot - 1) else c
            drop = bool(pk and tc == 0)
            for sub in range(ns):
                for e in range(2):
                    v = val[sub][e][tc]
                    acc |= ((v >> 1 if drop else v) & cb_mask) << pos
                    pos += cb
        if ab:
            ab_mask = (1 << ab) - 1
            tc = rot - 1 if rot else 3
            drop = bool(pk and tc == 0)
            for sub in range(ns):
                for e in range(2):
                    v = val[sub][e][tc]
                    acc |= ((v >> 1 if drop else v) & ab_mask) << pos
                    pos += ab
        if pk == 1:
            for sub in range(ns):
                acc |= (pb[sub][0] & 1) << pos
                pos += 1
                acc |= (pb[sub][1] & 1) << pos
                pos += 1
        elif pk == 2:
            for sub in range(ns):
                acc |= (pb[sub][0] & 1) << pos
                pos += 1

        ib_mask = (1 << ib) - 1
        ib_amask = (1 << (ib - 1)) - 1
        for p in range(16):
            anchor = p == anchors[sm[p] if sm is not None else 0]
            pv = pri[p]
            if anchor:
                if pv >> (ib - 1):
                    st |= _B7_ANCHOR
                acc |= (pv & ib_amask) << pos
                pos += ib - 1
            else:
                acc |= (pv & ib_mask) << pos
                pos += ib
        if sec is not None:
            ib2_mask = (1 << ib2) - 1
            ib2_amask = (1 << (ib2 - 1)) - 1
            for p in range(16):
                sv = sec[p]
                if p == 0:
                    if sv >> (ib2 - 1):
                        st |= _B7_ANCHOR
                    acc |= (sv & ib2_amask) << pos
                    pos += ib2 - 1
                else:
                    acc |= (sv & ib2_mask) << pos
                    pos += ib2
        if pos != 128:
            st |= _B7_SIGNATURE
        out[16 * i:16 * i + 16] = acc.to_bytes(16, "little")
        status[i] = st
    return bytes(out), status


def _bc7_blocks(lanes, nblocks, provisional, *, history=None):
    """Rebuild `nblocks` 16-byte BC7 blocks from the 16 BCPack lanes."""
    if len(lanes[0]) != 2 * nblocks:
        raise BcpackError("BC7 lane length does not match the block count")
    table = _bptc.load_bc7(history)
    unknown = _bc7_check(lanes, nblocks, table, provisional, has_history=(history is not None))
    if unknown:
        l0, l1 = lanes[0], lanes[1]
        er = [0] * nblocks
        ir = [0] * nblocks
        for i in range(nblocks):
            di = _inter8(l1[2 * i], l1[2 * i + 1])
            de = (di + _unzig(_inter8(l0[2 * i], l0[2 * i + 1]))) & 255
            er[i] = -1 if de == 0 else (256 + i - de if de > i else er[i - de])
            ir[i] = -1 if di == 0 else (256 + i - di if di > i else ir[i - di])
        counts = [(kinds, k, (er if kinds == "endpoint" else ir).count(k)) for kinds, k in unknown]
        total = _unknown_blocks(er, ir, {k for kinds, k in unknown if kinds == "endpoint"},
                                {k for kinds, k in unknown if kinds == "index"})
        if total > _loose_limit(nblocks):
            names = ", ".join(f"{kinds} {k}" for kinds, k, _ in counts[:6])
            if len(counts) > 6:
                names += ", ..."
            hint = " (pass --history <path>)" if history is None else ""
            raise BcpackProvisionalError(
                f"BC7 chunk has {total}/{nblocks} blocks from unknown built-in history entries ({names}){hint}"
            )
        for kinds, k, count in counts:
            provisional.add(f"BC7 unknown {kinds} history {k} ({count} blocks)")
    data, status = _bc7_rebuild(lanes, nblocks, table)
    bad = [i for i, st in enumerate(status) if st]
    if bad:
        first_st = status[bad[0]]
        if not (first_st & (_B7_ANCHOR | _B7_PBIT)):
            raise BcpackError(
                f"BC7 chunk failed signature check ({len(bad)}/{nblocks} blocks; first: block {bad[0]})"
            )
        if provisional is None or not provisional:
            raise BcpackError(
                f"BC7 chunk failed anchor/p-bit consistency check "
                f"({len(bad)}/{nblocks} blocks; first: block {bad[0]})"
            )
        if len(bad) > _loose_limit(nblocks):
            raise BcpackProvisionalError(
                f"BC7 chunk failed anchor/p-bit consistency check in {len(bad)}/{nblocks} blocks"
            )
        provisional.add(f"BC7 consistency check failed in {len(bad)} blocks")
    return data


_REBUILDERS = {
    1: _bc1_blocks,
    3: _bc3_blocks,
    4: _bc4_blocks,
    5: _bc5_blocks,
    6: _bc6h_blocks,
    7: _bc7_blocks,
}


def decode_with_info(raw, codec, out_size, *, allow_provisional=False, history=None):
    """Decode one BCPack chunk, returning `(decoded_bytes, provisional_labels)`.

    When `allow_provisional` is False (strict mode), any unconfirmed or unknown
    built-in history reference raises `BcpackProvisionalError` and the returned
    set is always empty. `history` may point to a locally generated JSON file,
    directory, or dict supplying initial history tables for BC6H/BC7 (and
    optional overrides for BC1-BC5).
    """
    rebuilder = _REBUILDERS.get(codec)
    if rebuilder is None:
        name = CODEC_NAMES.get(codec, str(codec))
        raise BcpackUnsupportedCodec(f"BCPack block rebuilder for codec {codec} ({name}) is not implemented yet")
    fast = _decode_blocks_native(raw, codec, out_size, history)
    if fast is not None:
        return fast, set()
    lanes, _ = decode_lanes(raw, codec, out_size)
    nblocks = out_size // BLOCK_BYTES[codec]
    prov = set() if allow_provisional else None
    data = rebuilder(lanes, nblocks, prov, history=history)
    return data, (prov if prov is not None else set())


def decode(raw, codec, out_size, *, allow_provisional=False, history=None):
    """Decode one BCPack chunk into `out_size` bytes of standard BCn blocks."""
    data, _ = decode_with_info(
        raw, codec, out_size, allow_provisional=allow_provisional, history=history
    )
    return data
