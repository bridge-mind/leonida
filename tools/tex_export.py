"""Export textures found by tex_headers.py as standard DDS files (top mip, slice 0), detiled from AMD SW_4K_S.

The compact TTPF mip-tail rule lives in tex_headers.py and is used by tools/ttpf_mips.py.
This carve exporter cannot apply it: the carved descriptor has no tile-mode field or native
allocation span, so the first/last heuristic below stays the only verified placement here.

For each texture:
  - element size from tex_headers.FMT_INFO (BC: 8/16 bytes per 4x4 block; uncompressed: bytes per pixel); only
    1/2/4/8/16-byte elements have a SW_4K_S equation, so 12-byte R32G32B32 stays unsupported
  - the top mip is either at the data offset or at the end of the mip chain (GFX10 stores mips smallest first);
    both are detiled and the smoother one wins (scored on the decoded image where Pillow can decode the format,
    otherwise on raw element bytes)
  - writes <out>/<archive>/<cpu blob>_<q index>_<WxH>_<format>.dds and records the choice in export_log.json,
    with "heuristic": true whenever two distinct placements fitted (the pick is a score, not a verified layout)
  - with --png, also writes the same path with .png for every texture tools/dds_png.py can decode (BC1-BC5, BC7
    and 8-bit RGBA/BGRA, R10G10B10A2_UNORM or R11G11B10_FLOAT, with no sRGB transfer; BC5 blue is a reconstructed normal Z;
    R16_FLOAT and R16G16B16A16_FLOAT clamped to [0, 1], NaN as 0).
    The rest (including BC6H and R32_FLOAT) keep their
    DDS only and are counted as "png unsupported". A PNG that fails to decode or write is counted as
    "png error:<Type>" and its DDS stays; neither changes the exit code

Usage:
    python -I tex_export.py <textures_index.json> <carve dir> <out dir> [--limit N] [--png]
"""
import argparse
import io
import json
import os
import struct
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from amd_swizzle import _parity, equation, load  # noqa: E402
from tex_headers import FMT_INFO  # noqa: E402

# DXGI format -> (bytes per element, block dimension), from the indexer's table; eq4k has SW_4K_S equations
# for 1/2/4/8/16-byte elements only, so 12-byte R32G32B32 (6, 7, 8) is left out and reported unsupported.
FMT = {fmt: info for fmt, info in FMT_INFO.items() if info[0] in (1, 2, 4, 8, 16)}
PILLOW_OK = {71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 98, 99, 28, 29, 87, 91}
# Pillow can't open sRGB DXGI codes; the bytes are identical to the UNORM twin, so decode with that for scoring.
DECODE_AS = {72: 71, 75: 74, 78: 77, 99: 98, 29: 28, 91: 87}

_EQ = {}
_PATTERNS = []  # (nibbles, tables) from addrlib, read on first use so the module imports without the header


def eq4k(bpe):
    if bpe not in _EQ:
        if not _PATTERNS:
            _PATTERNS.append(load())
        nib, tab = _PATTERNS[0]
        e = [e for e in tab["GFX10_SW_4K_S_PATINFO"] if f" {bpe} bpe" in e[5]][0]
        _EQ[bpe] = equation(nib, e, 12)
    return _EQ[bpe]


def detile_offsets(bpe, we, he):
    eq = eq4k(bpe)
    elems = 4096 // bpe
    lb = elems.bit_length() - 1
    bw = 1 << ((lb + 1) // 2)
    bh = elems // bw
    y, x = np.mgrid[0:he, 0:we].astype(np.int64)
    o = np.zeros_like(x)
    for bit, (mx, my) in enumerate(eq):
        if mx or my:
            o |= (_parity(x & mx) ^ _parity(y & my)) << bit
    nbx = (we + bw - 1) // bw
    return ((o + ((y // bh) * nbx + x // bw) * 4096) // bpe).reshape(-1), bw, bh


def level_bytes(we, he, bpe, bw, bh):
    return ((we + bw - 1) // bw) * ((he + bh - 1) // bh) * 4096


def dds_bytes(w, h, fmt, payload):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    bpe, blk = FMT[fmt]
    if blk == 1:  # DDSD_PITCH: bytes in one row of the top mip
        flags, pitch = 0x1007 | 0x8, w * bpe
    else:  # DDSD_LINEARSIZE: bytes in the whole top mip
        flags, pitch = 0x1007 | 0x80000, len(payload)
    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, h, w, pitch, 0, 1)
    struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
    struct.pack_into("<I", hdr, 108, 0x1000)
    return bytes(hdr) + struct.pack("<IIIII", fmt, 3, 0, 1, 0) + payload


def score_image(w, h, fmt, payload, bpe):
    a = None
    if fmt == 56:  # R16_UNORM
        a = np.frombuffer(payload, "<u2").reshape(h, w).astype(np.float32)
    elif fmt in PILLOW_OK:
        try:
            from PIL import Image

            im = Image.open(io.BytesIO(dds_bytes(w, h, DECODE_AS.get(fmt, fmt), payload))).convert("RGB")
            a = np.asarray(im.reduce(4) if min(w, h) >= 64 else im, np.float32)
        except (ImportError, NotImplementedError, OSError):
            # Some Pillow installations cannot decode these formats. Score the raw elements below.
            a = None
    if a is None:
        block = FMT[fmt][1]
        we, he = (w + block - 1) // block, (h + block - 1) // block
        a = np.frombuffer(payload, np.uint8).reshape(he, we, bpe)[:, :, :4].astype(np.float32)
    d = sum(np.abs(np.diff(a, axis=axis)).mean() for axis in (0, 1) if a.shape[axis] > 1)
    return float(d / (a.std() + 1e-6))


def export_one(t, carve, out, blob_cache, png=False):
    fmt = t["fmt"]
    if fmt not in FMT:
        return "unsupported format"
    bpe, blk = FMT[fmt]
    w, h = t["w"], t["h"]
    we, he = (w + blk - 1) // blk, (h + blk - 1) // blk
    chain_offs = tuple(off for off, _ in (t.get("gpu_chain") or ((t["gpu"], t["gpu_size"]),)))
    key = (t["archive"], chain_offs)
    if key not in blob_cache:
        blob_cache.clear()
        parts = []
        for off in chain_offs:
            with open(os.path.join(carve, t["archive"], f"{off:012x}.bin"), "rb") as f:
                parts.append(f.read())
        blob_cache[key] = b"".join(parts)
    G = blob_cache[key]
    offs, bw, bh = detile_offsets(bpe, we, he)
    need = (int(offs.max()) + 1) * bpe
    top = level_bytes(we, he, bpe, bw, bh)
    chain = sum(level_bytes(max(1, (max(1, w >> i) + blk - 1) // blk), max(1, (max(1, h >> i) + blk - 1) // blk), bpe, bw, bh)
                for i in range(max(1, t["mips"])))
    cands = []
    for place, start in (("first", t["data_off"]), ("last", t["data_off"] + chain - top)):
        if start < 0 or start + need > len(G):
            continue
        raw = np.frombuffer(G[start:start + need], np.uint8).reshape(-1, bpe)
        payload = raw[offs].tobytes()
        try:
            s = score_image(w, h, fmt, payload, bpe)
        except Exception:  # noqa: BLE001
            s = 9e9
        cands.append((s, place, payload))
    if not cands:
        return "data outside GPU blob"
    cands.sort(key=lambda c: c[0])
    s, place, payload = cands[0]
    name = f"{t['cpu']:012x}_{t['q_index']}_{w}x{h}_{t['fmt_name']}.dds"
    d = os.path.join(out, t["archive"])
    os.makedirs(d, exist_ok=True)
    dds = dds_bytes(w, h, fmt, payload)
    with open(os.path.join(d, name), "wb") as f:
        f.write(dds)
    # two placements that fit at different offsets leave the pick to the score, not to a known layout
    t["export"] = {"file": name, "place": place, "score": round(s, 4), "heuristic": len(cands) > 1 and chain > top}
    if png:
        import dds_png  # lazy: only --png pays for it
        png_path = os.path.join(d, name[:-4] + ".png")
        try:
            data = dds_png.dds_to_png_bytes(dds)
            with open(png_path, "wb") as f:
                f.write(data)
            t["export"]["png"] = "written"
        except dds_png.DdsError:  # a format it cannot decode; the payload is built here, so never a truncated one
            t["export"]["png"] = "unsupported"
        except Exception as e:  # noqa: BLE001  the DDS is already on disk and logged: a bad PNG must not undo that
            t["export"]["png"] = "error:" + type(e).__name__
            try:
                os.remove(png_path)  # no truncated PNG left behind
            except OSError:
                pass
    return "ok:" + place


def positive_int(text):
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
    ap.add_argument("out")
    ap.add_argument("--limit", type=positive_int, help="export only the first N textures (N >= 1)")
    ap.add_argument("--png", action="store_true", help="also write a PNG beside each DDS dds_png.py can decode")
    a = ap.parse_args()
    try:
        with open(a.index, encoding="utf-8-sig") as f:  # Notepad and PowerShell ">" save a BOM; json would refuse it
            texs = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {a.index}: {e}", file=sys.stderr)
        return 2
    if not isinstance(texs, list):
        print(f"ERROR: cannot read {a.index}: expected a texture list, got {type(texs).__name__}", file=sys.stderr)
        return 2
    try:
        texs.sort(key=lambda t: (t["archive"], t["gpu"]))
    except KeyError as e:
        print(f"ERROR: cannot read {a.index}: row has no {e.args[0]!r}", file=sys.stderr)
        return 2
    except TypeError as e:
        print(f"ERROR: cannot read {a.index}: {e}", file=sys.stderr)
        return 2
    if a.limit is not None:         # 0 used to read as "no limit", and a negative N dropped the last N
        texs = texs[: a.limit]
    if os.path.exists(a.out) and not os.path.isdir(a.out):
        print(f"ERROR: {a.out} is a file, not a folder", file=sys.stderr)
        return 2
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:  # a report path under an existing file, or a dangling symlink
        print(f"ERROR: cannot create {a.out}: {e.strerror or e}", file=sys.stderr)
        return 2
    res, cache, pngs = Counter(), {}, Counter()
    io_failures = 0
    first_io_error = None
    for i, t in enumerate(texs, 1):
        try:
            r = export_one(t, a.carve, a.out, cache, a.png)
        except Exception as e:  # noqa: BLE001
            r = "error:" + type(e).__name__
            if isinstance(e, OSError):
                io_failures += 1
                if first_io_error is None:
                    first_io_error = str(e)
        res[r] += 1
        if r.startswith("ok:") and "png" in t["export"]:  # an "export" read from the index may be stale
            pngs[t["export"]["png"]] += 1
        if i % 2000 == 0:
            print(f"[{i}/{len(texs)}] {dict(res)}", flush=True)
    report = os.path.join(a.out, "export_log.json")
    try:
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            json.dump(texs, f)
    except OSError as e:
        print(f"ERROR: cannot write {report}: {e.strerror or e}", file=sys.stderr)
        return 1
    print(f"{len(texs)} textures: {dict(res)} -> {a.out}")
    if a.png:
        print(f"png: {dict(pngs)}")
    if io_failures:
        print(f"ERROR: {io_failures} texture(s) could not be read or written, first: {first_io_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
