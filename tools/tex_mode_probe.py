"""For textures with known size/format/mips, find which AMD swizzle mode and top-mip placement decodes cleanly.

Candidates: every addrlib 4K/64K pattern for the element size, crossed with two placements of the top mip
(at the texture's data offset, or at the end of its mip chain, since GFX10 stores mips smallest first).
Scored with average RGB colours for BC1, BC2 and BC3 blocks (smoothness relative to the image's own spread).

Usage:
    python -I tex_mode_probe.py <textures_index.json> <carve dir> [--fmt BC1] [--n 12]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from amd_swizzle import _parity, equation, load  # noqa: E402
from bc_feature import block_average, smoothness  # noqa: E402

BPE = {"BC1": 8, "BC2": 16, "BC3": 16}


def bc_bpe(fmt_name):
    return BPE.get(fmt_name.split("_", 1)[0])


def _tile_shape(tile_bytes: int, bpe: int) -> tuple[int, int]:
    elements = tile_bytes // bpe
    width = 1 << (elements.bit_length() // 2)
    return width, elements // width


def mip_bytes(w: int, h: int, bpe: int, align: int) -> int:
    """Size of a mip in 4K/64K tiles, or its aligned linear size for other alignments."""
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    if align in (4096, 65536):
        tw, th = _tile_shape(align, bpe)
        return ((bw + tw - 1) // tw) * ((bh + th - 1) // th) * align
    n = bw * bh * bpe
    return (n + align - 1) // align * align


def offsets_for(eq, bb, bpe, wb, hb):
    bw, bh = _tile_shape(1 << bb, bpe)
    y, x = np.mgrid[0:hb, 0:wb].astype(np.int64)
    o = np.zeros_like(x)
    for bit, (mx, my) in enumerate(eq):
        if mx or my:
            o |= (_parity(x & mx) ^ _parity(y & my)) << bit
    nbx = (wb + bw - 1) // bw
    return (o + ((y // bh) * nbx + x // bw) * (1 << bb)) // bpe


def gpu_blob(carve, t):
    """The texture's whole GPU blob: every carved part of `gpu_chain` in order (data_off counts from the first),
    as tex_export.py reads it; a single `gpu` file when the index has no chain."""
    parts = []
    for off, _ in t.get("gpu_chain") or [[t["gpu"], t["gpu_size"]]]:
        with open(os.path.join(carve, t["archive"], f"{off:012x}.bin"), "rb") as f:
            parts.append(f.read())
    return b"".join(parts)


def positive_int(text):
    """--n: 0 probed nothing and printed an empty ranking as a result; a negative N dropped the last N textures."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("carve")
    ap.add_argument("--fmt", default="BC1", help="BC1, BC2, or BC3, optionally with a DXGI suffix")
    ap.add_argument("--n", type=positive_int, default=12, help="probe the first N matching textures (N >= 1)")
    a = ap.parse_args()
    if bc_bpe(a.fmt) is None:
        ap.error(f"no color scorer for {a.fmt!r}; expected BC1, BC2, or BC3")
    try:
        with open(a.index, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as error:
        ap.error(f"cannot read texture index {a.index}: {error}")
    if not isinstance(index, list):
        ap.error(f"{a.index}: texture index must contain a list")
    nib, tab = load()
    texs = [t for t in index if t["fmt_name"].startswith(a.fmt)
            and bc_bpe(t["fmt_name"]) is not None and t["w"] >= 256 and t["h"] >= 256]
    texs = texs[: a.n]
    cands = {}
    for tname, ents in tab.items():
        for size, bb in (("_64K_", 16), ("_4K_", 12)):
            if size in tname and ("_S" in tname or "_D" in tname or "_R_X_1xaa" in tname):
                for e in ents:
                    if f" {bc_bpe(a.fmt)} bpe" in e[5]:
                        eq = equation(nib, e, bb)
                        cands.setdefault((bb, tuple(eq)), f"{tname.replace('GFX10_SW_', '').replace('_PATINFO', '')} {e[5][:22]}")
    wins = Counter()
    skipped = 0
    for t in texs:
        bpe = bc_bpe(t["fmt_name"])
        G = gpu_blob(a.carve, t)
        wb, hb = (t["w"] + 3) // 4, (t["h"] + 3) // 4
        mips = (t["flags"] >> 16) & 0xFF
        res = []
        for (bb, eq), label in cands.items():
            align = 1 << bb
            top = mip_bytes(t["w"], t["h"], bpe, align)
            chain = sum(mip_bytes(max(1, t["w"] >> i), max(1, t["h"] >> i), bpe, align) for i in range(max(1, mips)))
            placements = [("first", t["data_off"])]
            if chain != top:
                placements.append(("last", t["data_off"] + chain - top))
            for place, start in placements:
                o = offsets_for(list(eq), bb, bpe, wb, hb)
                seg = G[start:start + (int(o.max()) + 1) * bpe]
                if len(seg) < (int(o.max()) + 1) * bpe:
                    continue
                raw = np.frombuffer(seg, np.uint8).reshape(-1, bpe)
                img = block_average(raw[o.reshape(-1)], bpe).reshape(hb, wb, 3)
                res.append((smoothness(img), label, place))
        res.sort()
        if res:
            lin_raw = np.frombuffer(G[t["data_off"]:t["data_off"] + wb * hb * bpe], np.uint8)
            lin = smoothness(block_average(lin_raw.reshape(-1, bpe), bpe).reshape(hb, wb, 3)) if len(lin_raw) == wb * hb * bpe else -1
            second = f"{res[1][0]:.4f} {res[1][1]} [{res[1][2]}]" if len(res) > 1 else "n/a"
            print(f"{t['w']}x{t['h']} {t['fmt_name']} mips={mips} gpu={t['gpu_size']}  best {res[0][0]:.4f} {res[0][1]} [{res[0][2]}]"
                  f"  2nd {second}  linear {lin:.4f}")
            wins[(res[0][1].split(' ')[0], res[0][2])] += 1
        else:
            skipped += 1
            print(f"{t['w']}x{t['h']} {t['fmt_name']} gpu={t['gpu']:#x}  skipped: no candidate fits in the GPU blob")
    print("winners:", wins.most_common())
    print(f"probed {len(texs) - skipped} of {len(texs)} textures; {skipped} skipped")


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
