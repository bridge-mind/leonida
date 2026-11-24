"""Public BPTC (BC6H and BC7) specification tables and runtime history loader.

Everything embedded in this module is either standard public D3D11 / Khronos
BPTC specification data (2-subset and 3-subset partition masks and anchor pixel
indices) or the BC7 lane-15 bitstream signature permutation (`BC7_CONFIG`).

BCPack's 256-entry initial endpoint/index history buffers for BC6H and BC7 are
**not** embedded in this repository because they are unpublished codec
constants that can only be recovered locally from the player's own texture
pools. Callers can decode self-contained BC6H/BC7 chunks (delta chains that
stay within the chunk) with `history=None`, or pass a locally generated JSON
file or directory (`--history <path>`) at runtime.
"""
import json
import os

# Standard D3D11 / Khronos BPTC 2-subset (64 x 16 pixels, 1 bit/pixel) and
# 3-subset (64 x 16 pixels, 2 bits/pixel) partition masks.
BC7_PART2_PACKED = (
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
)

BC7_PART3_PACKED = (
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050,
    0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
    0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250, 0xa5945040, 0x0a425054,
    0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414,
    0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
    0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50, 0x500aa550, 0xaaaa4444,
    0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580,
    0xaa141414, 0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
    0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
)

# Second and third subset anchor pixel indices (subset 0 anchor is always 0).
# BC6H two-region partitions use the first 32 entries of BC7_ANCHOR2.
BC7_ANCHOR2 = (
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
)

BC7_ANCHOR3A = (
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
)

BC7_ANCHOR3B = (
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
)

# BC7 lane-15 signature rank -> (mode, partition, rotation, index_selection).
# Lane-15 symbol pairs (a, b) on the 32 x 16 grid are ranked by
# `((a + 1) * (b + 1), a)` ascending. Modes 7, 6, 5, 4, 3 and 1 follow exact
# descending formulas; the 22 mode-2 partitions placed before mode 1 and the
# remaining 42 placed after mode 0 are both descending subsets of 0..63.
_BC7_MODE2_HEAD = (
    63, 59, 58, 57, 54, 53, 52, 47, 43, 42, 40,
    38, 37, 35, 34, 32, 23, 21, 18, 17, 16, 11,
)


def _build_bc7_config():
    head_set = set(_BC7_MODE2_HEAD)
    mode2_tail = tuple(p for p in range(63, -1, -1) if p not in head_set)
    return (
        tuple((7, 63 - r, 0, 0) for r in range(64))
        + ((6, 0, 0, 0),)
        + tuple((5, 0, 3 - r, 0) for r in range(4))
        + tuple((4, 0, (7 - r) >> 1, (7 - r) & 1) for r in range(8))
        + tuple((3, 63 - r, 0, 0) for r in range(64))
        + tuple((2, p, 0, 0) for p in _BC7_MODE2_HEAD)
        + tuple((1, 63 - r, 0, 0) for r in range(64))
        + tuple((0, p, 0, 0) for p in (*range(15, 7, -1), *range(5, -1, -1)))
        + tuple((2, p, 0, 0) for p in mode2_tail)
        + ((0, 7, 0, 0), (0, 6, 0, 0), (255, 255, 0, 0))
    )


BC7_CONFIG = _build_bc7_config()

# Signature ranks whose mode-2 partition assignment is inferred from anchor
# positions and order only; strict mode refuses chunks that reference them.
BC7_CONFIG_INFERRED = frozenset((149, 151, 152, 153, 246, 248, 251, 252, 253, 254))

_EMPTY_BC6H_EH = tuple(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)) for _ in range(256))
_EMPTY_BC6H_IH = tuple((0,) * 16 for _ in range(256))
_EMPTY_BC6H_CLS = (0,) * 256

_EMPTY_BC7_EP = tuple((0,) * 24 for _ in range(256))
_EMPTY_BC7_IX = tuple((0,) * 32 for _ in range(256))

_BC7_STATIC_CACHE = None
_FILE_CACHE = {}


def _bc7_static():
    """Return unpacked public D3D11/Khronos BC7 partition/anchor/signature tables."""
    global _BC7_STATIC_CACHE
    if _BC7_STATIC_CACHE is None:
        part2 = tuple(
            tuple((mask >> p) & 1 for p in range(16)) for mask in BC7_PART2_PACKED
        )
        part3 = tuple(
            tuple((mask >> (2 * p)) & 3 for p in range(16)) for mask in BC7_PART3_PACKED
        )
        cells = sorted(
            ((a, b) for a in range(32) for b in range(16)),
            key=lambda ab: ((ab[0] + 1) * (ab[1] + 1), ab[0]),
        )
        rank_of = [-1] * 512
        for rank, (a, b) in enumerate(cells):
            if rank < len(BC7_CONFIG):
                rank_of[a * 16 + b] = rank
        _BC7_STATIC_CACHE = {
            "partition2": part2,
            "partition3": part3,
            "anchor2": BC7_ANCHOR2,
            "anchor3a": BC7_ANCHOR3A,
            "anchor3b": BC7_ANCHOR3B,
            "config": BC7_CONFIG,
            "inferred": BC7_CONFIG_INFERRED,
            "rank_of": tuple(rank_of),
        }
    return _BC7_STATIC_CACHE


def _read_history_section(history, codec_key, marker_key):
    """Resolve `history` (None, dict, JSON file, or directory) for one codec."""
    if history is None:
        return None
    if isinstance(history, dict):
        if codec_key in history:
            return history[codec_key]
        return history if marker_key in history else None
    path = os.fspath(history)
    if os.path.isdir(path):
        candidate = os.path.join(path, f"{codec_key}.json")
        if not os.path.isfile(candidate):
            return None
        path = candidate
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    cache_key = (st.st_mtime_ns, st.st_size)
    cached = _FILE_CACHE.get(abspath)
    if cached is not None and cached[0] == cache_key:
        data = cached[1]
    else:
        with open(abspath, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        _FILE_CACHE[abspath] = (cache_key, data)
    if codec_key in data:
        return data[codec_key]
    return data if marker_key in data else None


def _iter_entries(obj, count=256):
    """Yield `(index, value)` pairs from a 256-list or sparse `{index: value}` dict."""
    if obj is None:
        return
    if isinstance(obj, list):
        if len(obj) != count:
            raise ValueError(f"expected {count} entries, got {len(obj)}")
        for i, v in enumerate(obj):
            yield i, v
    elif isinstance(obj, dict):
        for k, v in obj.items():
            i = int(k)
            if not 0 <= i < count:
                raise ValueError(f"history entry index {i} outside 0..{count - 1}")
            yield i, v
    else:
        raise ValueError("history table must be a 256-element list or index-keyed dict")


def _flatten_bc7_entry(val, expected_len):
    """Flatten nested `[3][2][4]` / `[2][16]` or flat `24`/`32` BC7 entry to a tuple of ints."""
    out = []

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
        else:
            out.append(int(x))

    _walk(val)
    if len(out) != expected_len:
        raise ValueError(f"expected {expected_len} integers in BC7 history entry, got {len(out)}")
    return tuple(out)


def load_bc6h(history=None, *, with_confirmed_bits=False):
    """Return `(endpoint_history, index_history, endpoint_class, index_class)`.

    With `history=None`, returns empty/unknown tables (class `0` for all 256
    entries) so self-contained BC6H chunks still decode strictly without any
    external file. When `history` is a local JSON file, directory containing
    `bc6h.json`, or dict, populates the BC6H initial history tables from it.
    With `with_confirmed_bits=True`, append endpoint `[256][3][4]` and index
    `[256][16]` leading-bit confirmation depths. Legacy documents without the
    optional depth tables retain full precision for class-3 entries.
    """
    sec = _read_history_section(history, "bc6h", "endpointHistory")
    if sec is None:
        result = (_EMPTY_BC6H_EH, _EMPTY_BC6H_IH, _EMPTY_BC6H_CLS, _EMPTY_BC6H_CLS)
        return result + (_EMPTY_BC6H_EH, _EMPTY_BC6H_IH) if with_confirmed_bits else result
    eh = list(_EMPTY_BC6H_EH)
    ih = list(_EMPTY_BC6H_IH)
    ec = list(_EMPTY_BC6H_CLS)
    ic = list(_EMPTY_BC6H_CLS)
    for i, channels in _iter_entries(sec.get("endpointHistory")):
        if len(channels) != 3 or any(len(ch) != 4 for ch in channels):
            raise ValueError(f"BC6H endpointHistory[{i}] must be 3x4 integers")
        eh[i] = tuple(tuple(int(v) for v in ch) for ch in channels)
    for i, pixels in _iter_entries(sec.get("indexHistory")):
        if len(pixels) != 16:
            raise ValueError(f"BC6H indexHistory[{i}] must have 16 integers")
        ih[i] = tuple(int(v) for v in pixels)
    for i, cls in _iter_entries(sec.get("endpointClass")):
        ec[i] = int(cls)
    for i, cls in _iter_entries(sec.get("indexClass")):
        ic[i] = int(cls)
    ep_cb = [tuple((16, 10, 6, 6) for _ in range(3)) if c == 3 else _EMPTY_BC6H_EH[0] for c in ec]
    ix_cb = [(4,) * 16 if c == 3 else _EMPTY_BC6H_IH[0] for c in ic]
    for key, target, empty, widths in (
        ("endpointConfirmedBits", ep_cb, _EMPTY_BC6H_EH, (16, 10, 6, 6)),
        ("indexConfirmedBits", ix_cb, _EMPTY_BC6H_IH, (4,) * 16),
    ):
        if key not in sec:
            continue
        # In an explicit sparse depth table, omitted rows confirm no bits.
        target[:] = empty
        for i, row in _iter_entries(sec[key]):
            if key == "endpointConfirmedBits":
                if len(row) != 3 or any(len(ch) != 4 for ch in row):
                    raise ValueError(f"BC6H {key}[{i}] must be 3x4 integers")
                value = tuple(tuple(int(v) for v in ch) for ch in row)
                valid = all(0 <= v <= widths[q] for ch in value for q, v in enumerate(ch))
            else:
                if len(row) != 16:
                    raise ValueError(f"BC6H {key}[{i}] must have 16 integers")
                value = tuple(int(v) for v in row)
                valid = all(0 <= v <= 4 for v in value)
            if not valid:
                raise ValueError(f"BC6H {key}[{i}] bit depth out of bounds")
            target[i] = value
    result = (tuple(eh), tuple(ih), tuple(ec), tuple(ic))
    return result + (tuple(ep_cb), tuple(ix_cb)) if with_confirmed_bits else result


def load_bc7(history=None):
    """Return the BC7 partition, anchor, signature, and history tables.

    With `history=None`, built-in history slots have depth `0` (unknown) so
    self-contained BC7 chunks decode strictly without external data while any
    reference to built-in history cleanly refuses. When `history` is a local
    JSON file, directory containing `bc7.json`, or dict, loads the BC7 initial
    history tables from it.
    """
    static = _bc7_static()
    sec = _read_history_section(history, "bc7", "historyEndpoints")
    if sec is None:
        return {
            **static,
            "history_endpoints": _EMPTY_BC7_EP,
            "history_endpoint_bits": _EMPTY_BC7_EP,
            "history_endpoint_confirmed_bits": _EMPTY_BC7_EP,
            "history_indices": _EMPTY_BC7_IX,
            "history_index_bits": _EMPTY_BC7_IX,
            "history_index_confirmed_bits": _EMPTY_BC7_IX,
            "last_resort": frozenset(),
        }
    ep = list(_EMPTY_BC7_EP)
    ep_b = list(_EMPTY_BC7_EP)
    ep_cb = list(_EMPTY_BC7_EP)
    ix = list(_EMPTY_BC7_IX)
    ix_b = list(_EMPTY_BC7_IX)
    ix_cb = list(_EMPTY_BC7_IX)
    for i, v in _iter_entries(sec.get("historyEndpoints")):
        ep[i] = _flatten_bc7_entry(v, 24)
    for i, v in _iter_entries(sec.get("historyEndpointBits")):
        ep_b[i] = _flatten_bc7_entry(v, 24)
    for i, v in _iter_entries(sec.get("historyEndpointConfirmedBits")):
        ep_cb[i] = _flatten_bc7_entry(v, 24)
    for i, v in _iter_entries(sec.get("historyIndices")):
        ix[i] = _flatten_bc7_entry(v, 32)
    for i, v in _iter_entries(sec.get("historyIndexBits")):
        ix_b[i] = _flatten_bc7_entry(v, 32)
    for i, v in _iter_entries(sec.get("historyIndexConfirmedBits")):
        ix_cb[i] = _flatten_bc7_entry(v, 32)
    # Mid-range fill for endpoints known to 0 < d < 8 leading bits.
    # Readers taking <= d bits shift the low bits away unchanged.
    for i in range(256):
        depth = ep_b[i]
        if any(0 < d < 8 for d in depth):
            row = ep[i]
            ep[i] = tuple(
                ((v >> (8 - d) << (8 - d)) | (1 << (7 - d))) if 0 < d < 8 else v
                for v, d in zip(row, depth)
            )
    last_resort = set()
    for item in sec.get("historyEndpointLastResort") or ():
        if len(item) == 4:
            k, sub, e, c = (int(x) for x in item)
            last_resort.add((k, sub * 8 + e * 4 + c))
        elif len(item) == 2:
            last_resort.add((int(item[0]), int(item[1])))
        else:
            raise ValueError("historyEndpointLastResort entries must have 2 or 4 integers")
    return {
        **static,
        "history_endpoints": tuple(ep),
        "history_endpoint_bits": tuple(ep_b),
        "history_endpoint_confirmed_bits": tuple(ep_cb),
        "history_indices": tuple(ix),
        "history_index_bits": tuple(ix_b),
        "history_index_confirmed_bits": tuple(ix_cb),
        "last_resort": frozenset(last_resort),
    }


_BC_HISTORY_SPEC = {
    1: ("bc1", (32, 64, 32, 32, 64, 32)),
    3: ("bc3", (256, 256, 32, 64, 32, 32, 64, 32)),
    4: ("bc4", (256, 256)),
    5: ("bc5", (256, 256, 256, 256)),
}


def load_bc_history(codec, history=None):
    """Load optional BC1/BC3/BC4/BC5 initial history overrides from `history`.

    Returns `None` when `history` is `None` or does not contain a section for
    `codec`. Otherwise returns `(confirmed, provisional, replace)` where
    `confirmed` and `provisional` map `0..255` entry indices to validated
    endpoint tuples and `replace` is a bool (default False: overlay onto the
    built-in `bcpack_tables.py` defaults).
    """
    if history is None:
        return None
    spec = _BC_HISTORY_SPEC.get(codec)
    if spec is None:
        raise ValueError(f"unsupported BC history codec: {codec}")
    codec_key, mods = spec
    if isinstance(history, dict):
        if codec_key in history:
            sec = history[codec_key]
        elif history.get("codec") == codec_key:
            sec = history
        else:
            return None
    else:
        path = os.fspath(history)
        if os.path.isdir(path):
            candidate = os.path.join(path, f"{codec_key}.json")
            if not os.path.isfile(candidate):
                return None
            path = candidate
        abspath = os.path.abspath(path)
        st = os.stat(abspath)
        cache_key = (st.st_mtime_ns, st.st_size)
        cached = _FILE_CACHE.get(abspath)
        if cached is not None and cached[0] == cache_key:
            data = cached[1]
        else:
            with open(abspath, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a JSON object")
            _FILE_CACHE[abspath] = (cache_key, data)
        if codec_key in data:
            sec = data[codec_key]
        elif data.get("codec") == codec_key:
            sec = data
        else:
            return None
    if not isinstance(sec, dict):
        raise ValueError(f"{codec_key} history section must be a JSON object")
    if "codec" in sec and sec["codec"] != codec_key:
        raise ValueError(f"expected codec {codec_key!r}, got {sec['codec']!r}")
    ncols = len(mods)

    def _parse_map(obj, label):
        out = {}
        for i, vals in _iter_entries(obj):
            if not isinstance(vals, (list, tuple)) or len(vals) != ncols:
                raise ValueError(f"{codec_key} {label}[{i}] must have {ncols} integers")
            row = []
            for v, mod in zip(vals, mods):
                iv = int(v)
                if not 0 <= iv < mod:
                    raise ValueError(f"{codec_key} {label}[{i}] value {iv} outside 0..{mod - 1}")
                row.append(iv)
            out[i] = tuple(row)
        return out

    confirmed = _parse_map(sec.get("confirmed"), "confirmed")
    provisional = _parse_map(sec.get("provisional"), "provisional")
    replace = bool(sec.get("replace", False))
    return confirmed, provisional, replace
