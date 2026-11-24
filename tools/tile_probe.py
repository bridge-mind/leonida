"""Find the memory layout of a GPU texture blob empirically.

For each candidate (block size 8 or 16 bytes, linear or tiled with a Z-order inside 4 KB / 64 KB tiles, and a range
of widths), it reorders the compressed blocks, builds a small image from each block's colour endpoint, and scores
how smooth that image is. Real textures are spatially coherent, so the right layout scores best.

Usage:
    python -I tile_probe.py <blob> [--out preview.png] [--offset N]
"""
import argparse
import os
import sys

import numpy as np

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402


def morton_decode(i, bits):
    """Split a Z-order index into (x, y), x taking the even bits."""
    x = np.zeros_like(i)
    y = np.zeros_like(i)
    for b in range(bits):
        x |= ((i >> (2 * b)) & 1) << b
        y |= ((i >> (2 * b + 1)) & 1) << b
    return x, y


def tile_dims(tile_bytes, bpe):
    n = tile_bytes // bpe              # blocks per tile
    bits = int(np.log2(n))
    tw = 1 << ((bits + 1) // 2)        # wider than tall when bits is odd
    return tw, n // tw, bits


def block_feature(blocks, bpe):
    """One RGB value per compressed block, from its first colour endpoint."""
    if bpe == 8:
        c = blocks[:, 0].astype(np.uint32) | (blocks[:, 1].astype(np.uint32) << 8)       # BC1 color0 (565)
    elif bpe == 16:
        c = blocks[:, 8].astype(np.uint32) | (blocks[:, 9].astype(np.uint32) << 8)       # BC3 color0 (565)
    else:
        raise ValueError(f"unsupported block size bpe={bpe} (expected 8 or 16)")
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    if bpe == 16:
        # BC5/BC7 have no 565 endpoint there; mix in the first byte (BC5 red0 / BC7 mode+endpoint bits).
        r = (r + blocks[:, 0]) // 2
    return np.stack([r, g, b], -1).astype(np.float32)


def layout(nblocks, wblocks, mode, bpe):
    """Linear position (y * wblocks + x) for each stored block, or None if the layout doesn't fit."""
    idx = np.arange(nblocks, dtype=np.int64)
    if mode == "linear":
        x, y = idx % wblocks, idx // wblocks
    else:
        tb = 4096 if mode == "4k" else 65536
        tw, th, bits = tile_dims(tb, bpe)
        if wblocks % tw:
            return None
        per = tw * th
        if nblocks % (wblocks * th):
            return None
        tile, inner = idx // per, idx % per
        # x takes the even bits, y the odd ones; with an odd bit count x gets the extra top bit.
        ix, iy = morton_decode(inner, (bits + 1) // 2)
        tiles_x = wblocks // tw
        x = (tile % tiles_x) * tw + ix
        y = (tile // tiles_x) * th + iy
    return y * wblocks + x


def score(img: np.ndarray) -> float:
    """Sum the mean color differences along axes with neighboring blocks."""
    return float(sum(np.abs(np.diff(img, axis=axis)).mean()
                     for axis in (1, 0) if img.shape[axis] > 1))


def probe(data: bytes, offset=0):
    best = []
    for bpe in (8, 16):
        raw = np.frombuffer(data[offset:offset + (len(data) - offset) // bpe * bpe], np.uint8).reshape(-1, bpe)
        feat = block_feature(raw, bpe)
        for wpx in (256, 512, 1024, 2048, 4096):
            wb = wpx // 4
            hb0 = min(len(raw) // wb, wb)         # top mip assumed square or wider
            for mode in ("linear", "4k", "64k"):
                hb = hb0
                if mode != "linear":
                    th = tile_dims(4096 if mode == "4k" else 65536, bpe)[1]
                    hb = hb0 // th * th
                if hb < 16:
                    continue
                n = wb * hb
                pos = layout(n, wb, mode, bpe)
                if pos is None:
                    continue
                img = np.zeros((hb * wb, 3), np.float32)
                img[pos] = feat[:n]
                s = score(img.reshape(hb, wb, 3))
                best.append((s, bpe, wpx, mode))
    best.sort()
    return best


def render(data, offset, bpe, wpx, mode, out):
    """Full BC decode of the reordered top mip through Pillow (BC1 for 8-byte blocks, BC7 for 16-byte)."""
    import io
    import struct
    try:
        from PIL import Image
    except ModuleNotFoundError:
        # only the preview needs Pillow; -I does not see a --user install of it
        raise SystemExit("--out needs Pillow: the scan runs without it, but the preview cannot be written "
                         "(run without -I, or install Pillow where this interpreter can see it)")
    if bpe not in (8, 16):
        raise ValueError(f"unsupported block size bpe={bpe} (expected 8 or 16)")
    wb = wpx // 4
    raw = np.frombuffer(data[offset:offset + (len(data) - offset) // bpe * bpe], np.uint8).reshape(-1, bpe)
    hb = min(len(raw) // wb, wb)
    if mode != "linear":
        th = tile_dims(4096 if mode == "4k" else 65536, bpe)[1]
        hb = hb // th * th
    n = wb * hb
    pos = layout(n, wb, mode, bpe)
    if n == 0 or pos is None:
        raise ValueError(f"layout does not fit: wpx={wpx}, mode={mode!r}, bpe={bpe}")
    lin = np.zeros((n, bpe), np.uint8)
    lin[pos] = raw[:n]
    w, h = wb * 4, hb * 4
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<IIIIIII", hdr, 4, 124, 0x1007, h, w, n * bpe, 0, 1)
    struct.pack_into("<I", hdr, 108, 0x1000)
    if bpe == 8:
        struct.pack_into("<II4s", hdr, 76, 32, 4, b"DXT1")
        body = bytes(hdr)
    else:
        struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
        body = bytes(hdr) + struct.pack("<IIIII", 98, 3, 0, 1, 0)  # BC7_UNORM, TEXTURE2D
    im = Image.open(io.BytesIO(body + lin.tobytes())).convert("RGB")
    im.save(out)
    return im.size


def check_preview_path(out, blob):
    """Refuse a preview that would destroy the sample, and create its folder.

    Before: `--out <blob>` opened the blob for the probe and then wrote the preview over it, truncating the
    sample the run had just read; a missing folder raised a PIL traceback after the whole probe had run.
    """
    if os.path.isdir(out):
        raise SystemExit(f"{out} is a directory, not an image file")
    if same_report_path(out, blob):
        raise SystemExit(f"--out would overwrite the input {blob}")
    out_dir = os.path.dirname(os.path.abspath(out))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:  # a regular file where the folder should be: one line, not a traceback
        raise SystemExit(f"--out {out}: cannot create {out_dir}: {e.strerror or e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.path.exists(a.blob):
        print(f"ERROR: {a.blob}: does not exist", file=sys.stderr)
        return 2
    if a.out:
        check_preview_path(a.out, a.blob)
    try:
        with open(a.blob, "rb") as f:
            data = f.read()
    except OSError as e:
        raise SystemExit(f"{a.blob}: {e.strerror or e}")
    if not 0 <= a.offset < len(data):  # a negative offset used to probe the file's tail, one past the end nothing
        raise SystemExit(f"--offset {a.offset} is outside {a.blob} ({len(data)} bytes)")
    res = probe(data, a.offset)
    if not res:
        # Every candidate needs at least 16 rows of blocks at one of the widths; a short sample used to print
        # nothing, write no preview and exit 0, which reads as a successful run with no result.
        print(f"ERROR: no layout fits {a.blob} at offset {a.offset}: the sample is too short for every candidate "
              "width (use a larger blob, or an --offset nearer its start)", file=sys.stderr)
        return 1
    for s, bpe, wpx, mode in res[:6]:
        print(f"score {s:8.2f}  {bpe}-byte blocks  width {wpx:5}  {mode}")
    if a.out and res:
        _, bpe, wpx, mode = res[0]
        print("preview", render(data, a.offset, bpe, wpx, mode, a.out), "->", a.out)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
