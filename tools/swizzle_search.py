"""Search for the in-tile swizzle of a GPU texture by brute force.

Within a tile of 2^B blocks, each address bit maps to one x or one y bit. Every ordering with the right
x/y split is tried (bit k goes to the next free x or y bit), tiles are laid out row-major, and the ordering
whose block-endpoint image is smoothest wins.

Usage:
    python -I swizzle_search.py <blob> --bpe 8 --tile 65536 [--widths 1024,2048] [--tiles 48] [--out preview.png]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tile_probe import block_feature, score, same_report_path  # noqa: E402


def positions(seq, nblocks, tw, th, wblocks):
    idx = np.arange(nblocks, dtype=np.int64)
    per = tw * th
    tile, inner = idx // per, idx % per
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    xi = yi = 0
    for k, c in enumerate(seq):
        bit = (inner >> k) & 1
        if c == "x":
            x |= bit << xi
            xi += 1
        else:
            y |= bit << yi
            yi += 1
    tiles_x = wblocks // tw
    return ((tile // tiles_x) * th + y) * wblocks + (tile % tiles_x) * tw + x


def search(data, bpe, tile_bytes, widths, ntiles):
    raw = np.frombuffer(data[: len(data) // bpe * bpe], np.uint8).reshape(-1, bpe)
    feat = block_feature(raw, bpe)
    bits = int(np.log2(tile_bytes // bpe))
    results = []
    for xbits in sorted({bits // 2, (bits + 1) // 2}):
        tw, th = 1 << xbits, 1 << (bits - xbits)
        for wpx in widths:
            wb = wpx // 4
            if wb % tw:
                continue
            tiles_x = wb // tw
            blocks_per_row = tiles_x * tw * th
            rows = min(max(1, ntiles // tiles_x), len(raw) // blocks_per_row)
            if not rows:
                continue
            n = blocks_per_row * rows
            for xs in itertools.combinations(range(bits), xbits):
                seq = ["x" if k in xs else "y" for k in range(bits)]
                pos = positions(seq, n, tw, th, wb)
                img = np.zeros((n, 3), np.float32)
                img[pos] = feat[:n]
                results.append((score(img.reshape(-1, wb, 3)), wpx, "".join(seq)))
    results.sort()
    return results


def render(data, bpe, tile_bytes, wpx, seq, out):
    import io
    import struct
    from PIL import Image
    raw = np.frombuffer(data[: len(data) // bpe * bpe], np.uint8).reshape(-1, bpe)
    bits = len(seq)
    xbits = seq.count("x")
    tw, th = 1 << xbits, 1 << (bits - xbits)
    wb = wpx // 4
    hb = min(len(raw) // wb // th * th, wb // th * th or th)
    n = wb * hb
    pos = positions(list(seq), n, tw, th, wb)
    lin = np.zeros((n, bpe), np.uint8)
    lin[pos] = raw[:n]
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<IIIIIII", hdr, 4, 124, 0x1007, hb * 4, wb * 4, n * bpe, 0, 1)
    struct.pack_into("<I", hdr, 108, 0x1000)
    if bpe == 8:
        struct.pack_into("<II4s", hdr, 76, 32, 4, b"DXT1")
        body = bytes(hdr)
    else:
        struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
        body = bytes(hdr) + struct.pack("<IIIII", 98, 3, 0, 1, 0)
    Image.open(io.BytesIO(body + lin.tobytes())).convert("RGB").save(out)


def validate_out_path(path):
    """Return an error string if `path` cannot be used as a --out image file, or None."""
    if not path or not path.strip():
        return "--out path must not be empty"
    if "\x00" in path:
        return "--out path contains a NUL byte"
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)) or os.path.isdir(path):
        return f"--out path is a directory: {path}"
    cur = os.path.dirname(os.path.abspath(path))
    while cur:
        if os.path.isdir(cur):
            break
        if os.path.lexists(cur):
            return f"--out parent is not a directory: {cur}"
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--bpe", type=int, default=8)
    ap.add_argument("--tile", type=int, default=65536)
    ap.add_argument("--widths", default="1024,2048")
    ap.add_argument("--tiles", type=int, default=48, help="target number of tiles to score (default: 48)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.out is not None:
        out_err = validate_out_path(a.out)
        if out_err:
            print(f"ERROR: {out_err}", file=sys.stderr)
            return 2
        if same_report_path(a.out, a.blob):
            ap.error("--out would overwrite the input blob")
    if a.bpe < 1:
        ap.error("--bpe must be at least 1")
    if a.bpe not in (8, 16):
        ap.error("--bpe must be 8 or 16")
    if a.tile < a.bpe or a.tile % a.bpe:
        ap.error("--tile must be a multiple of --bpe and at least --bpe")
    blocks = a.tile // a.bpe
    if blocks & (blocks - 1):
        ap.error("--tile / --bpe must be a power of two")
    try:
        widths = [int(w) for w in a.widths.split(",")]
    except ValueError:
        ap.error("--widths must be a comma-separated list of integers")
    if not widths or any(w < 4 or w % 4 for w in widths):
        ap.error("--widths must be positive multiples of 4")
    if a.tiles < 1:
        ap.error("--tiles must be at least 1")
    try:
        with open(a.blob, "rb") as f:
            data = f.read()
    except OSError as e:  # a missing or unreadable input: one ERROR line, as the other tools do
        print(f"ERROR: cannot read {a.blob}: {e.strerror or e}", file=sys.stderr)
        return 2
    res = search(data, a.bpe, a.tile, widths, a.tiles)
    for s, w, seq in res[:8]:
        print(f"score {s:7.2f}  width {w:5}  bits(low->high) {seq}")
    if a.out is not None:
        if not res:
            print("ERROR: no layout fits the input, tile size and requested widths", file=sys.stderr)
            return 1
        parent = os.path.dirname(os.path.abspath(a.out))
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            render(data, a.bpe, a.tile, res[0][1], res[0][2], a.out)
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or str(e)
            print(f"ERROR: cannot write {a.out}: {detail}", file=sys.stderr)
            return 2
        print("preview ->", a.out)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
