"""Measure the vertex skin streams (joint indices and weights) of recovered drawable models (ASSET-12).

Rigged models cannot be exported with their skeleton until the joint-index and skin-weight layout is
known. This tool walks every drawable resource that carves out of an RPF9 archive without its (encrypted)
table of contents — drawable dictionaries, fragments and standalone drawables — reads each geometry's FVF
semantic table, classifies the skin streams and validates the decoded bytes against the model's own bone
count. docs/character_skeleton.md carries the measured tables; no package data is stored or emitted.

Resource shape (measured; the same walk rpf9-explorer's RageModelDecoder uses, pinned in third_party/):

    system page      0x50000000-based pointers; a resource starts at a 4096-aligned offset + 16 that is
                     preceded by 16 zero bytes and a zlib header (rpf9-explorer's candidate scan)
    page directory   +0x08 -> {+0x08 systemPages, +0x09 graphicsPages}; one zlib stream per page, system
                     pages first, then graphics pages (0x60000000-based pointers)
    drawable         LOD pointer fields +80..+104 -> {+0x08 model count u16} -> model pointer array
    model (grmModel) +0x08 geometry pointer array, +0x10 geometry count u16, +0x14 local bone count u16
    geometry         +0x08 -> vertex buffer header (vb); vb+0x40 -> FVF table; vb+0x18 -> graphics buffer
    FVF table        per semantic i: stream offset u32 at +4*i, element size u8 at +0xD0+i, format u8 at
                     +0x104+i; u32 flags at +0x138, vertex count = flags >> 10, bit 0 set

The vertex buffer starts with one native buffer descriptor word per used semantic (stream offset in bits
7..28, a size code in bits 3..6); the tool cross-checks every skin slot's stream offset against its
descriptor word (the size code equals the element bytes only for the 4- and 8-byte slots).

Skin semantics verified from the bytes (not from format names):

    semantic 16/17   weights: four u8 UNORM per slot; a vertex's bytes over all present weight slots sum
                     to 255 (a small tail rounds: 247-254, and below 247 in one packed 4-influence
                     population)
    semantic 20/21   bone indices into the model's own bone list. In the packed layout each u32 holds a
                     12-bit index in bits 0..11 (bits 9..11 are never set; bits 12..31 are a separate
                     unidentified field). A zero-weight component may carry any index, so indices are
                     range-checked only where the weight is non-zero.

Usage:
    python -I tools/skin_streams.py <archive.rpf> [...] [--sample N] [--json out.json]
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

PAGE = 4096
MAX_SYS = 32 * 1024 * 1024
MAX_GFX = 224 * 1024 * 1024
MAX_PAGE_COUNT = 32
WEIGHT_SLOTS = (16, 17)
INDEX_SLOTS = (20, 21)
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")

# FVF format byte -> (name, element bytes). Anchored on entries whose decoding is verified against
# package bytes: positions (12/6 float3, 8/16 quantized), UVs (4/38 snorm16 pairs), weights (4/28, rows
# sum to 255), bone indices (4/30, values below the bone count), rigid 16-bit indices (2/57, the format
# breadbbbb/rpf9-explorer#10 reads). Unverified format bytes fall back to the element size alone.
FORMATS = {
    2: ("R32G32B32A32_FLOAT", 16),
    3: ("R32G32B32A32_UINT", 16),
    6: ("R32G32B32_FLOAT", 12),
    12: ("R16G16B16A16_UINT", 8),
    16: ("R32G32_FLOAT", 8),
    28: ("R8G8B8A8_UNORM", 4),
    30: ("R8G8B8A8_UINT", 4),
    38: ("R16G16_SNORM", 4),
    41: ("R32_FLOAT", 4),
    57: ("R16_UINT", 2),
    62: ("R8_UINT", 1),
}


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def sys_ptr(p):
    return 0x50000000 <= p < 0x50000000 + MAX_SYS


class Skip(Exception):
    """The candidate is not a drawable resource; the scanner moves on."""


def check(condition):
    if not condition:
        raise Skip()


def s_off(b, at, size):
    """Resolve a system pointer field to a system-page offset, bounds-checked."""
    off = u64(b, at) - 0x50000000
    if not (0 <= off <= len(b) - size):
        raise Skip()
    return off


def finite_bounds(b, at):
    mn = struct.unpack_from("<3f", b, at)
    mx = struct.unpack_from("<3f", b, at + 16)
    for v in mn + mx:
        if v != v or v in (float("inf"), float("-inf")) or abs(v) >= 1e7:
            return False
    return mn[0] <= mx[0] and mn[1] <= mx[1] and mn[2] <= mx[2]
def resource_kind(b):
    """Drawable / drawable dictionary / fragment signature on the system page prefix, or None."""
    if len(b) < 192 or u64(b, 0) != 0:
        return None
    if sys_ptr(u64(b, 8)) and sys_ptr(u64(b, 32)) and sys_ptr(u64(b, 48)) and 0 < u16(b, 56) <= 4096 \
            and u16(b, 56) == u16(b, 58) == u16(b, 40) == u16(b, 42):
        # DrawableDictionaryCheckBytes also reads the first child when its pointer array is inside
        # the prefix; beyond it the decoder accepts the header without that read.
        values = u64(b, 48) - 0x50000000
        if values > len(b) - 8 or sys_ptr(u64(b, values)):
            return "dictionary"
    if all(sys_ptr(u64(b, at)) for at in (8, 24, 168)) \
            and all(p == 0 or sys_ptr(p) for p in (u64(b, 16), u64(b, 32), u64(b, 40))) \
            and 0 < u16(b, 48) <= 4096 and u16(b, 50) <= 4096 \
            and u16(b, 52) <= u16(b, 54) <= 4096:
        return "fragment"
    if (sys_ptr(u64(b, 8)) or u64(b, 8) == 0) and sys_ptr(u64(b, 16)) and sys_ptr(u64(b, 80)) \
            and (sys_ptr(u64(b, 168)) or u64(b, 168) == 0):
        radius = struct.unpack_from("<f", b, 44)[0]
        if not (0 < radius < 1e6) or not finite_bounds(b, 48):
            return None
        return "drawable"
    return None


def drawable_offsets(sys, kind):
    """Yield the system-page offset of every drawable header inside the resource."""
    if kind == "dictionary":
        count = u16(sys, 56)
        ptrs = s_off(sys, 48, count * 8)
        for i in range(count):
            child = s_off(sys, ptrs + i * 8, 192)
            check(u64(sys, child) == 0 and u64(sys, child + 8) == 0)
            yield child
        return
    if kind == "fragment":
        yield s_off(sys, 24, 192)
        return
    yield 0


class Mesh:
    __slots__ = ("vertices", "bones", "slots", "gfx_at", "gfx_len")

    def __init__(self, vertices, bones, slots, gfx_at, gfx_len):
        self.vertices = vertices
        self.bones = bones
        self.slots = slots          # semantic -> (size, fmt, offset)
        self.gfx_at = gfx_at        # graphics-page offset of the vertex buffer
        self.gfx_len = gfx_len      # vertex buffer length in bytes


def walk_resource(sys, kind):
    """Walk drawable -> LOD -> model -> geometry -> FVF; return meshes carrying skin slots."""
    meshes = []
    for drawable in drawable_offsets(sys, kind):
        seen_lods = set()
        for lod_field in range(80, 105, 8):
            if u64(sys, drawable + lod_field) == 0:
                continue
            lod = s_off(sys, drawable + lod_field, 16)
            if lod in seen_lods:
                continue
            seen_lods.add(lod)
            models = u16(sys, lod + 8)
            check(0 < models <= 256 and models == u16(sys, lod + 10))
            ptrs = s_off(sys, lod, models * 8)
            for mi in range(models):
                model = s_off(sys, ptrs + mi * 8, 56)
                count = u16(sys, model + 16)
                bones = u16(sys, model + 20)
                check(0 < count <= 256 and count == u16(sys, model + 18) == u16(sys, model + 54)
                      and bones <= 4096)
                geometries = s_off(sys, model + 8, count * 8)
                for gi in range(count):
                    geo = s_off(sys, geometries + gi * 8, 160)
                    vb = s_off(sys, geo + 8, 80)
                    fvf = s_off(sys, vb + 64, 0x13C)
                    flags = u32(sys, fvf + 0x138)
                    vertices = flags >> 10
                    check(flags & 1 and 0 < vertices <= 1_000_000)
                    psize, pfmt = sys[fvf + 0xD0], sys[fvf + 0x104]
                    check((psize, pfmt) in ((12, 6), (8, 16)))   # position semantic 0
                    slots = {}
                    for i in range(0x34):
                        size, fmt = sys[fvf + 0xD0 + i], sys[fvf + 0x104 + i]
                        offset = u32(sys, fvf + 4 * i)
                        if size:   # a size-0 slot is a placeholder with no stream
                            if fmt in FORMATS:
                                check(FORMATS[fmt][1] == size)
                            slots[i] = (size, fmt, offset)
                    gfx_len = u32(sys, vb + 8) * u16(sys, vb + 12)
                    check(0 < gfx_len <= MAX_GFX and u16(sys, vb + 12) <= 32)
                    p_off = u32(sys, fvf)
                    check(p_off >= 96 and p_off <= gfx_len and psize * vertices <= gfx_len - p_off)
                    for size, _fmt, offset in slots.values():
                        check(offset <= gfx_len and size * vertices <= gfx_len - offset)
                    gfx_at = u64(sys, vb + 24) - 0x60000000
                    check(0 <= gfx_at <= MAX_GFX - gfx_len)
                    if any(i in slots for i in WEIGHT_SLOTS + INDEX_SLOTS):
                        meshes.append(Mesh(vertices, bones, slots, gfx_at, gfx_len))
                    check(len(meshes) <= 4096)
    return meshes


def slot_signature(mesh):
    parts = []
    for i in WEIGHT_SLOTS + INDEX_SLOTS:
        if i in mesh.slots:
            size, fmt, _ = mesh.slots[i]
            parts.append(f"{i}:{size}b/{FORMATS.get(fmt, (f'fmt{fmt}', 0))[0]}")
    return ",".join(parts) or "none"


def classify(mesh):
    """Layout name for a mesh's skin slots, or None when they don't form a known layout."""
    weights = [i for i in WEIGHT_SLOTS if i in mesh.slots]
    indices = [i for i in INDEX_SLOTS if i in mesh.slots]
    if not indices:
        return None
    isz, ifmt, _ = mesh.slots[indices[0]]
    for i in indices[1:]:
        if mesh.slots[i][:2] != (isz, ifmt):
            return None
    for i in weights:
        if mesh.slots[i][:2] != (4, 28):
            return None
    if not weights:
        if (isz, ifmt) == (1, 62):
            return "rigid-8bit"
        if (isz, ifmt) == (2, 57):
            return "rigid-16bit"
        return None
    influences = 4 * len(weights)
    if (isz, ifmt) == (4, 30):
        return f"weighted-{influences}inf-8bit"
    if (isz, ifmt) == (16, 3):
        return f"weighted-{influences}inf-packed"
    if (isz, ifmt) == (8, 12):
        return f"weighted-{influences}inf-16bit"
    return None


def descriptor_offsets(data, limit):
    """Stream offsets the native buffer descriptor words at the vertex buffer head encode.

    One word per used semantic, so the scan runs to `limit` — the slot-0 stream offset, the first
    byte past the head. A word packs the stream offset in bits 7..28 and a size code in bits 3..6;
    the code equals the element bytes for the 4- and 8-byte slots, but reads 1 for the 16-byte
    packed index slot and 0 for the 1-byte rigid slot, so only the offset is checked.
    """
    found = set()
    for w in range(0, min(limit, len(data)) - 3, 4):
        word = u32(data, w)
        if word:
            found.add((word >> 7) & 0x1FFFFFC)
    return found


def validate(mesh, gfx, layout):
    """Check the decoded skin bytes against the model's own bone count; returns a tally dict."""
    if mesh.gfx_at < 0 or mesh.gfx_at + mesh.gfx_len > len(gfx):
        return {"buffer-short": 1, "vertices": mesh.vertices}
    data = gfx[mesh.gfx_at:mesh.gfx_at + mesh.gfx_len]
    n = mesh.vertices
    t = {"vertices": n, "meshes": 1, "weighted-components": 0, "components-out-of-range": 0,
         "weight-sum-255": 0, "weight-sum-247-254": 0, "weight-sum-other": 0,
         "descriptor-mismatch": 0, "packed-bits-9-11-set": 0, "packed-bit-8-set": 0}
    desc = descriptor_offsets(data, mesh.slots[0][2] if 0 in mesh.slots else 96)
    weight_rows, index_rows = [], []
    for i in WEIGHT_SLOTS:
        if i in mesh.slots:
            size, _fmt, offset = mesh.slots[i]
            weight_rows.append(data[offset:offset + size * n])
            if offset not in desc:
                t["descriptor-mismatch"] += 1
    for i in INDEX_SLOTS:
        if i in mesh.slots:
            size, _fmt, offset = mesh.slots[i]
            index_rows.append(data[offset:offset + size * n])
            if offset not in desc:
                t["descriptor-mismatch"] += 1
    if weight_rows:
        low = high = None
        for v in range(n):
            total = sum(sum(row[4 * v:4 * v + 4]) for row in weight_rows)
            low = total if low is None else min(low, total)
            high = total if high is None else max(high, total)
            if total == 255:
                t["weight-sum-255"] += 1
            elif 247 <= total <= 254:
                t["weight-sum-247-254"] += 1
            else:
                t["weight-sum-other"] += 1
        t["weight-sum-min"] = low
        t["weight-sum-max"] = high
    if layout.endswith("packed"):
        for row in index_rows:
            for word in struct.unpack_from(f"<{4 * n}I", row, 0):
                t["weighted-components"] += 1
                if (word >> 9) & 7:
                    t["packed-bits-9-11-set"] += 1
                if (word >> 8) & 1:
                    t["packed-bit-8-set"] += 1
                if (word & 0xFFF) >= mesh.bones:
                    t["components-out-of-range"] += 1
        # The index is only meaningful where the matching weight is non-zero; the packed census keeps
        # the raw in-range share, and the weight-masked count is reported for the 4-influence form.
    elif index_rows:
        for k, row in enumerate(index_rows):
            weights = weight_rows[k] if k < len(weight_rows) else (weight_rows[0] if weight_rows else None)
            if len(row) == n:  # rigid-8bit
                t["weighted-components"] += n
                if max(row) >= mesh.bones:
                    t["components-out-of-range"] += sum(1 for b in row if b >= mesh.bones)
            elif len(row) == 2 * n:  # rigid-16bit
                for idx in struct.unpack_from(f"<{n}H", row, 0):
                    t["weighted-components"] += 1
                    if idx >= mesh.bones:
                        t["components-out-of-range"] += 1
            else:  # four 8-bit or 16-bit indices per vertex, masked by the matching weights
                wide = len(row) == 8 * n
                for v in range(n):
                    base = 4 * v
                    joints = struct.unpack_from("<4H", row, 2 * base) if wide else row[base:base + 4]
                    for c, joint in enumerate(joints):
                        if weights is None or weights[base + c]:
                            t["weighted-components"] += 1
                            if joint >= mesh.bones:
                                t["components-out-of-range"] += 1
    return t


def inflate_at(m, at, size):
    """Inflate the zlib stream at `at`; returns (data, consumed) or (None, 0)."""
    d = zlib.decompressobj()
    out, p = [], at
    try:
        while not d.eof and p < size:
            chunk = m[p:min(p + (1 << 20), size)]
            if not chunk:
                break
            out.append(d.decompress(chunk))
            p += len(chunk)
            if d.unused_data or d.eof:
                break
        data = b"".join(out) + d.flush()
    except zlib.error:
        return None, 0
    if not d.eof:
        return None, 0
    return data, p - at - len(d.unused_data)

def next_stream(m, fp, size):
    """The next zlib stream at or after `fp` (padding between streams is zero): (data, start, used)."""
    mt = ZHDR.search(m, fp, size)
    if not mt:
        return None
    at = mt.start()
    data, used = inflate_at(m, at, size)
    if data is None:
        return None
    return data, at, used


def scan_archive(path, sample, log):
    """Scan one archive; returns census and validation tallies per skin layout."""
    stats = {"candidates": 0, "resources": 0, "skipped": 0, "meshes": 0, "vertices": 0,
             "unclassified": 0, "layouts": {}, "signatures": {}, "validated": {}}
    size = os.path.getsize(path)
    if size == 0:
        return stats    # mmap refuses an empty file; an empty archive has no resources to count
    t0 = time.time()
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        pos = 0
        while pos < size:
            page = (pos + PAGE - 1) & ~(PAGE - 1)
            at = page + 16
            if at + 2 > size:
                break
            pos = page + PAGE
            if m[at - 16:at] != b"\0" * 16 or not ZHDR.match(m, at):
                continue
            data, used = inflate_at(m, at, size)
            if data is None:
                continue
            kind = resource_kind(data) if len(data) >= 192 else None
            if kind is None:
                continue
            stats["candidates"] += 1
            try:
                directory = s_off(data, 8, 16)
                system_pages, graphics_pages = data[directory + 8], data[directory + 9]
                check(0 < system_pages <= MAX_PAGE_COUNT and 0 < graphics_pages <= MAX_PAGE_COUNT)
                following, fp = [], at + used
                for _ in range(system_pages - 1 + graphics_pages):
                    nxt = next_stream(m, fp, size)
                    if nxt is None:
                        break
                    more, start, used2 = nxt
                    following.append(more)
                    fp = start + used2
                resource_sys = b"".join([data] + following[:system_pages - 1])
                resource_gfx = b"".join(following[system_pages - 1:])
                meshes = walk_resource(resource_sys, kind)
            except (Skip, struct.error):
                stats["skipped"] += 1
                continue
            stats["resources"] += 1
            for mesh in meshes:
                layout = classify(mesh) or "unclassified"
                row = stats["layouts"].setdefault(layout, {"meshes": 0, "vertices": 0})
                row["meshes"] += 1
                row["vertices"] += mesh.vertices
                stats["meshes"] += 1
                stats["vertices"] += mesh.vertices
                sig = slot_signature(mesh)
                stats["signatures"][sig] = stats["signatures"].get(sig, 0) + 1
                if layout == "unclassified":
                    stats["unclassified"] += 1
                    continue
                acc = stats["validated"].setdefault(layout, {"_meshes": 0})
                if acc["_meshes"] < sample and 0 <= mesh.gfx_at and mesh.gfx_at + mesh.gfx_len <= len(resource_gfx):
                    acc["_meshes"] += 1
                    for key, value in validate(mesh, resource_gfx, layout).items():
                        if key.endswith(("-min", "-max")):
                            acc[key] = min(acc.get(key, value), value) if key.endswith("-min") \
                                else max(acc.get(key, value), value)
                        else:
                            acc[key] = acc.get(key, 0) + value
            pos = max(pos, fp)   # the candidate scan resumes after the pages this resource consumed
            if stats["resources"] % 500 == 0:
                log(f"  {os.path.basename(path)} @{page:#x}: {stats['resources']} resources, "
                    f"{stats['meshes']} meshes, {time.time() - t0:.0f}s")
    return stats


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="RPF9 archives to scan")
    ap.add_argument("--sample", type=int, default=64,
                    help="meshes validated per layout per archive (the census covers every mesh)")
    ap.add_argument("--json", help="write the full tallies to this file")
    a = ap.parse_args(argv)
    if a.sample < 1:
        print("error: --sample must be at least 1", file=sys.stderr)
        return 2
    for path in a.inputs:
        if not os.path.isfile(path):
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2
    if a.json:
        if os.path.isdir(a.json):
            print(f"error: --json {a.json} is a directory", file=sys.stderr)
            return 2
        for path in a.inputs:
            if same_output_path(a.json, path):
                print(f"error: --json would overwrite the input {path}", file=sys.stderr)
                return 2
    report, t0 = {}, time.time()
    for path in a.inputs:
        try:
            stats = scan_archive(path, a.sample, lambda text: print(text, flush=True))
        except OSError as error:
            print(f"error: cannot read {path}: {error.strerror or error}", file=sys.stderr)
            return 2
        report[path] = stats
        print(f"{os.path.basename(path)}: {stats['resources']} resources, {stats['meshes']} skin meshes, "
              f"{stats['vertices']} vertices ({time.time() - t0:.0f}s)")
        for layout in sorted(stats["layouts"]):
            row = stats["layouts"][layout]
            print(f"  {layout:26s} {row['meshes']:8d} meshes {row['vertices']:12d} vertices")
    if a.json:
        try:
            parent = os.path.dirname(os.path.abspath(a.json))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(report, fh, indent=1)
        except OSError as exc:
            print(f"error: cannot write {a.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
