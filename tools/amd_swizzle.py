"""AMD GFX10/RDNA2 texture detiling from addrlib's published swizzle patterns.

Reads gfx10SwizzlePattern.h (Mesa's addrlib, MIT licence) and builds, for a swizzle mode and pipe/packer
configuration, the equation that maps an element's (x, y) to its byte offset. 64 KB macro blocks are laid out
row-major; inside a block every address bit is an XOR of x and y bits (including bits above the block, which is
how the XOR modes spread neighbouring blocks across pipes).

As a tool:
    python -I amd_swizzle.py <blob> --bpe 8 --width 2048 --height 2048 [--offset N] [--modes 64K] [--out best.png]
tries every pattern table whose name has the --modes part (64K by default; 4K, 256, S_X, ...) and configuration
for that element size, each with its own block size, scores each by image smoothness, and renders the best. As a
module, `detile(data, pattern, bpe, wb, hb, block_bytes)` returns linear blocks.
"""
import argparse
import os
import re
import sys

import numpy as np

HEADER = os.environ.get("AMD_SWIZZLE_HEADER", r"<ref>\mesa-addrlib\gfx10SwizzlePattern.h")
BPE_INDEX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
# Address bits of one block for each pattern table size, as tex_mode_probe.py uses them (12 for 4 KB, 16 for 64 KB).
BLOCK_BITS = {"256": 8, "4K": 12, "64K": 16}
# Unsigned/void dtype whose item size is the element size, so detile gathers whole elements in one take.
_ELEMENT_DTYPE = {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8", 16: "V16"}


def block_bits(table_name):
    """Address bits of one block of a pattern table (`GFX10_SW_4K_S_PATINFO` -> 12), or None for another size."""
    parts = table_name.split("_")
    return BLOCK_BITS.get(parts[2]) if len(parts) > 3 else None


def table_matches(table_name, modes):
    """Whether --modes picks this table: whole name parts, so 4K does not also pick the 64K tables."""
    return f"_{modes.strip('_')}_" in f"{table_name}_"


def _symbols():
    env = {}
    for i in range(16):
        env[f"X{i}"] = 1 << i
        env[f"Y{i}"] = 1 << (16 + i)
        env[f"Z{i}"] = 1 << (32 + i)
        env[f"S{i}"] = 1 << (48 + i)
    return env


def _term(text, env):
    """Value of one pattern cell: symbols and integers joined by ^, the only syntax addrlib's tables use."""
    value = 0
    for part in text.split("^"):
        part = part.strip()
        if part in env:
            value ^= env[part]
        elif re.fullmatch(r"\d+", part):
            value ^= int(part)
        else:
            raise ValueError(f"unexpected swizzle pattern term: {text!r}")
    return value


def load(path=HEADER):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    env = _symbols()
    nibbles = {}
    for name in ("NIBBLE01", "NIBBLE2", "NIBBLE3", "NIBBLE4"):
        m = re.search(r"GFX10_SW_PATTERN_" + name + r"\[\]\[\d+\]\s*=\s*\{(.*?)\n\};", text, re.S)
        rows = re.findall(r"\{([^{}]*)\}", m.group(1))
        nibbles[name] = [[_term(t, env) for t in r.split(",") if t.strip()] for r in rows]
    tables = {}
    for m in re.finditer(r"const ADDR_SW_PATINFO (GFX10_SW_\w+_PATINFO)\[\]\s*=\s*\{(.*?)\n\};", text, re.S):
        entries = []
        for row in m.group(2).splitlines():
            mm = re.match(r"\s*\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*\}\s*,\s*//\s*(.*)", row)
            if mm:
                entries.append(tuple(int(v) for v in mm.groups()[:5]) + (mm.group(6).strip(),))
        tables[m.group(1)] = entries
    return nibbles, tables


def equation(nibbles, entry, block_bits):
    """Per address bit, (x mask, y mask) for the given pattern entry, for the first `block_bits` bits."""
    _, n01, n2, n3, n4, _ = entry
    bits = list(nibbles["NIBBLE01"][n01]) + list(nibbles["NIBBLE2"][n2]) + list(nibbles["NIBBLE3"][n3]) + \
        list(nibbles["NIBBLE4"][n4])
    return [(v & 0xFFFF, (v >> 16) & 0xFFFF) for v in bits[:block_bits]]


def _parity(a):
    a = a ^ (a >> 8)
    a = a ^ (a >> 4)
    a = a ^ (a >> 2)
    a = a ^ (a >> 1)
    return a & 1


def _axis_offsets(eq, n, mask_of, shift):
    """In-block address contribution of one axis: each address bit is an XOR of x bits and y bits, so the offset
    of (x, y) is the XOR of a function of x alone and a function of y alone. Evaluating those two 1-D tables
    (length wb and hb) replaces the per-bit parity over the whole wb x hb grid. `shift` drops the low address
    bits (a right shift distributes over XOR) so detile gets element indices without a pass over the grid."""
    coord = np.arange(n, dtype=np.int64)
    acc = np.zeros(n, dtype=np.int64)
    for bit, masks in enumerate(eq):
        m = masks[mask_of]
        if m:
            acc |= _parity(coord & m) << bit
    return acc >> shift if shift else acc


def _grid(eq, wb, hb, bw, bh, block_bytes, shift):
    """(hb, wb) int64 array of ((XOR of the address bits) + block * block_bytes) >> shift, built in place."""
    blocks_x = (wb + bw - 1) // bw
    col = _axis_offsets(eq, wb, 0, shift)
    row = _axis_offsets(eq, hb, 1, shift)
    unit = block_bytes >> shift  # block_bytes is a multiple of 1 << shift, so the shift splits over the sum
    out = np.bitwise_xor(row[:, None], col[None, :])
    out += (np.arange(hb, dtype=np.int64) // bh * (blocks_x * unit))[:, None]
    out += (np.arange(wb, dtype=np.int64) // bw * unit)[None, :]
    return out


def _block_shape(bpe, block_bytes):
    """Macro-block size in elements, from the element count.

    Deriving it from the x/y bits that appear alone in an address bit only works for the S modes; in the XOR modes
    the high bits are almost always XORed, so the block came out too small (72/75 RDNA2 R_X entries).
    tex_export.detile_offsets uses this same formula for 4 KB blocks and it matches every 64 KB entry."""
    elems = block_bytes // bpe
    lb = elems.bit_length() - 1
    bw = 1 << ((lb + 1) // 2)
    return bw, elems // bw


def offsets(eq, bpe, wb, hb, block_bytes=65536):
    """Byte offset of every element (row-major y, x) for a wb x hb element grid."""
    xb = max((mx.bit_length() for mx, _ in eq if mx), default=0)
    yb = max((my.bit_length() for _, my in eq if my), default=0)
    bw, bh = _block_shape(bpe, block_bytes)
    return _grid(eq, wb, hb, bw, bh, block_bytes, 0), (bw, bh, xb, yb)


def feature(linear, bpe):
    """Per-element RGB for the smoothness score: raw RGB for uncompressed bpe, colour endpoint for BC."""
    if bpe == 4:
        return linear.reshape(-1, 4)[:, :3].astype(np.float32)
    if bpe == 2:
        p = linear.reshape(-1, 2).astype(np.float32)
        return np.stack([p[:, 0], p[:, 1], (p[:, 0] + p[:, 1]) * 0.5], axis=-1)
    if bpe == 1:
        return np.repeat(linear.reshape(-1, 1).astype(np.float32), 3, axis=1)
    from tile_probe import block_feature
    return block_feature(linear, bpe)


def detile(data, eq, bpe, wb, hb, block_bytes=65536):
    raw = np.frombuffer(data[: len(data) // bpe * bpe], np.uint8).reshape(-1, bpe)
    bw, bh = _block_shape(bpe, block_bytes)
    shift = bpe.bit_length() - 1
    if bpe & (bpe - 1) == 0 and block_bytes % bpe == 0:
        idx = _grid(eq, wb, hb, bw, bh, block_bytes, shift)
    else:
        idx = _grid(eq, wb, hb, bw, bh, block_bytes, 0) // bpe
    if idx.size and idx.max() >= len(raw):
        return None
    flat = idx.reshape(-1)
    if bpe in _ELEMENT_DTYPE:
        # One gather of whole elements instead of bpe separate byte rows.
        # Reading `data` as one wide element needs its length to be a multiple of bpe, which the slice above ensures.
        wide = raw.reshape(-1).view(_ELEMENT_DTYPE[bpe])
        return wide.take(flat).view(np.uint8).reshape(-1, bpe)
    return raw[flat]


def validate_out_path(out, blob, same_report_path):
    """Return an error string if `out` cannot be used as an --out image path, or None."""
    if not out or not out.strip():
        return "--out path must not be empty"
    if "\x00" in out:
        return "--out path contains a NUL byte"
    if out.endswith(os.sep) or (os.altsep and out.endswith(os.altsep)) or os.path.isdir(out):
        return f"--out path is a directory: {out}"
    cur = os.path.dirname(os.path.abspath(out))
    while cur:
        if os.path.isdir(cur):
            break
        if os.path.lexists(cur):
            return f"--out parent is not a directory: {cur}"
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    if same_report_path(out, blob) or (os.path.exists(out) and os.path.exists(blob) and os.path.samefile(out, blob)):
        return f"--out must not overwrite the input {blob}"
    return None


def main(argv=None):
    sys.path.insert(0, os.path.dirname(__file__))
    from tile_probe import same_report_path, score
    from swizzle_search import render  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--bpe", type=int, choices=(1, 2, 4, 8, 16), default=8)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--modes", default="_64K_")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.out is not None:
        err = validate_out_path(a.out, a.blob, same_report_path)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
    try:
        with open(a.blob, "rb") as f:
            data = f.read()[a.offset:]
    except OSError as e:
        print(f"ERROR: {a.blob}: {e.strerror or e}", file=sys.stderr)
        return 2
    nibbles, tables = load()
    # BC formats cover 4x4 pixels per element; uncompressed 1/2/4 bpe formats have one pixel per element.
    block = 4 if a.bpe in (8, 16) else 1
    wb = (a.width + block - 1) // block
    hb = (a.height + block - 1) // block
    res = []
    for tname, entries in tables.items():
        bits = block_bits(tname)
        if bits is None or not table_matches(tname, a.modes):
            continue
        for e in entries:
            if f" {a.bpe} bpe" not in e[5]:
                continue
            eq = equation(nibbles, e, bits)
            lin = detile(data, eq, a.bpe, wb, hb, 1 << bits)
            if lin is None:
                continue
            img = feature(lin, a.bpe).reshape(hb, wb, 3)
            res.append((score(img), tname, e[5], e, bits))
    res.sort(key=lambda r: r[0])
    seen = set()
    for s, t, c, _, _ in res:
        if (t, c) in seen:
            continue
        seen.add((t, c))
        print(f"{s:8.2f}  {t:40} {c}")
        if len(seen) >= 10:
            break
    if a.out and res:
        import io
        import struct
        from PIL import Image
        bits = res[0][4]
        lin = detile(data, equation(nibbles, res[0][3], bits), a.bpe, wb, hb, 1 << bits)
        if a.bpe == 1:
            preview = Image.fromarray(lin.reshape(hb, wb)).convert("RGB")
        elif a.bpe == 2:
            rgb = np.zeros((hb, wb, 3), dtype=np.uint8)
            rgb[:, :, :2] = lin.reshape(hb, wb, 2)
            preview = Image.fromarray(rgb)
        else:
            hdr = bytearray(128)
            hdr[0:4] = b"DDS "
            struct.pack_into("<IIIIIII", hdr, 4, 124, 0x1007, a.height, a.width, lin.size, 0, 1)
            struct.pack_into("<I", hdr, 108, 0x1000)
            if a.bpe == 8:
                struct.pack_into("<II4s", hdr, 76, 32, 4, b"DXT1")
                body = bytes(hdr)
            elif a.bpe == 4:
                # Uncompressed RGBA8, not BC7: the 4 bpe path is a plain pixel grid.
                struct.pack_into("<IIIIIIII", hdr, 76, 32, 0x41, 0, 32,
                                 0xFF, 0xFF00, 0xFF0000, 0xFF000000)
                body = bytes(hdr)
            else:
                struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
                body = bytes(hdr) + struct.pack("<IIIII", 98, 3, 0, 1, 0)
            preview = Image.open(io.BytesIO(body + lin.tobytes())).convert("RGB")
        try:
            out_dir = os.path.dirname(os.path.abspath(a.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            preview.save(a.out)
        except (OSError, ValueError) as e:
            msg = getattr(e, "strerror", None) or str(e)
            print(f"ERROR: {a.out}: {msg}", file=sys.stderr)
            return 2
        print("best ->", a.out)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
