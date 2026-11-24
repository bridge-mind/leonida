"""Scan RPF9 archives or carved 0x7a files for CScatteringTransmittance__FogVolumeListData resources.

Reports only counts, offsets, schema hashes, and aggregate numeric/flag distributions (no game strings or names).
See docs/gamedata.md, "Fog volumes" (ASSET-308).

Usage:
    python -I fog_volumes.py <archive.rpf|carved file|folder> [--out FILE] [--jobs N]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import pool_workers  # noqa: E402
import world_census as WC  # noqa: E402
import world_index as WI  # noqa: E402
from report_paths import same_report_path  # noqa: E402

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


def joaat_cs(s: str) -> int:
    """Case-sensitive joaat32 used by 0x7a schema struct and member tables."""
    h = 0
    for c in s.encode("latin-1"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


STRUCT_LIST = joaat_cs("CScatteringTransmittance__FogVolumeListData")  # 0x03073767
STRUCT_VOLUME = joaat_cs("CScatteringTransmittance__FogVolume")        # 0x81907918

FIELDS = {
    name: joaat_cs(name)
    for name in (
        "IsBox",
        "Position",
        "PositionOffset",
        "Scale",
        "Rotation",
        "Shear",
        "RotationMtx",
        "TransformMtx",
        "BlendMode",
        "IntExtMode",
        "Type",
        "SortKey",
        "ProfileIndex",
        "SecondaryProfileIndex",
        "ProfileInterpolation",
    )
}

_IDENT3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _ptr(val: int, size: int) -> int | None:
    if val >> 28 != 5:
        return None
    off = val & 0x0FFFFFFF
    return off if off < size else None


def parse_schema(b: bytes) -> tuple[dict[int, tuple[int, dict]], list[tuple[int, int, int | None]]]:
    """Return ({struct_hash: (struct_size, members)}, [(block_struct_hash, block_size, offset_or_None)])."""
    n = len(b)
    if n < 0x50 or b[0x10:0x14] != b"0DRP" or b[0x14] != 0x7A:
        return {}, []
    nst, nbl = WI._u16(b, 0x48), WI._u16(b, 0x4C)
    st = _ptr(WI._u64(b, 0x28), n)
    it = _ptr(WI._u64(b, 0x38), n)
    if not (1 <= nst <= 4096 and 1 <= nbl <= 4096) or st is None or it is None:
        return {}, []
    if st + 32 * nst > n or it + 16 * nbl > n:
        return {}, []
    structs: dict[int, tuple[int, dict]] = {}
    for i in range(nst):
        o = st + 32 * i
        sh = WI._u32(b, o)
        sz = WI._u32(b, o + 0x18)
        cnt = WI._u16(b, o + 0x1E)
        tbl = _ptr(WI._u64(b, o + 0x10), n)
        members = WI._members(b, tbl, cnt, n) or {}
        structs.setdefault(sh, (sz, members))
    blocks = []
    for i in range(nbl):
        o = it + 16 * i
        blocks.append((WI._u32(b, o), WI._u32(b, o + 4), _ptr(WI._u64(b, o + 8), n)))
    return structs, blocks


def _read_field(b: bytes, base: int, member):
    if member is None:
        return None
    off, ty = member[0], member[1]
    p = base + off
    if ty == 0x01:  # bool / u8
        return b[p] if p < len(b) else None
    if ty == 0x35:  # Mat34V / 3x vec4f (48 bytes)
        if p + 48 > len(b):
            return None
        m = struct.unpack_from("<12f", b, p)
        return (m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10])
    if ty == 0x62:  # enum u32
        return struct.unpack_from("<I", b, p)[0] if p + 4 <= len(b) else None
    return WI._value(b, base, member)


def parse_fog_volumes(b: bytes, *, include_digest: bool = False) -> list[dict]:
    """Decode all CScatteringTransmittance__FogVolume instances in a 0x7a resource."""
    structs, blocks = parse_schema(b)
    if STRUCT_VOLUME not in structs:
        return []
    vsz, vmems = structs[STRUCT_VOLUME]
    if not vsz:
        return []
    out: list[dict] = []
    for bsh, bsz, boff in blocks:
        if bsh != STRUCT_VOLUME or boff is None or bsz % vsz or boff + bsz > len(b):
            continue
        for k in range(bsz // vsz):
            base = boff + k * vsz
            row = {name: _read_field(b, base, vmems.get(h)) for name, h in FIELDS.items()}
            cleaned = {k_: v for k_, v in row.items() if v is not None}
            if include_digest:
                cleaned["_sha256"] = hashlib.sha256(b[base:base + vsz]).digest()
            out.append(cleaned)
    return out


def _is_identity3(m) -> bool:
    return m is not None and len(m) == 9 and all(abs(a - c) < 1e-4 for a, c in zip(m, _IDENT3))


def _scan_rpf_chunk(args: tuple[str, int, int]) -> tuple[list[int], int, int]:
    """Scan [start, end) of an RPF9 archive for complete 0x7a FogVolume resources without a 256 KiB cap.

    Returns (matching_offsets, complete_0x7a_count, incomplete_0x7a_count).
    """
    path, start, end = args
    offs: list[int] = []
    complete_0x7a = 0
    incomplete_0x7a = 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        pos = start
        while pos < end:
            mt = ZHDR.search(m, pos, end)
            if not mt:
                break
            off = mt.start()
            if ((m[off] << 8) | m[off + 1]) % 31 != 0:
                pos = off + 1
                continue
            d = zlib.decompressobj()
            try:
                head = d.decompress(m[off:min(size, off + 256)], 24)
            except zlib.error:
                pos = off + 1
                continue
            if len(head) < 21 or head[0x10:0x14] != b"0DRP" or head[0x14] != 0x7A:
                pos = off + 2
                continue
            try:
                res = WC.assemble(m, off, size)
            except zlib.error:
                incomplete_0x7a += 1
                pos = off + 2
                continue
            if not res:
                incomplete_0x7a += 1
                pos = off + 2
                continue
            pos = max(off + 2, res.get("end", off + 2))
            if not res.get("complete"):
                incomplete_0x7a += 1
                continue
            complete_0x7a += 1
            data = res["data"]
            nst = struct.unpack_from("<H", data, 0x48)[0]
            st = _ptr(struct.unpack_from("<Q", data, 0x28)[0], len(data))
            if not (1 <= nst <= 4096) or st is None or st + 32 * nst > len(data):
                continue
            for i in range(nst):
                if struct.unpack_from("<I", data, st + 32 * i)[0] == STRUCT_VOLUME:
                    offs.append(off)
                    break
    return offs, complete_0x7a, incomplete_0x7a


def _scan_rpf_offsets(path: str, jobs: int = 1) -> tuple[list[int], int, int]:
    size = os.path.getsize(path)
    if size < 16:
        return [], 0, 0
    workers = pool_workers.process_workers(max(1, jobs))
    chunk_sz = max(4 * 1024 * 1024, min(64 * 1024 * 1024, (size + workers - 1) // workers))
    ranges = [(path, s, min(size, s + chunk_sz)) for s in range(0, size, chunk_sz)]
    offs: list[int] = []
    complete_0x7a = 0
    incomplete_0x7a = 0
    if len(ranges) == 1 or workers == 1:
        for r in ranges:
            part_offs, c7a, i7a = _scan_rpf_chunk(r)
            offs.extend(part_offs)
            complete_0x7a += c7a
            incomplete_0x7a += i7a
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for part_offs, c7a, i7a in ex.map(_scan_rpf_chunk, ranges):
                offs.extend(part_offs)
                complete_0x7a += c7a
                incomplete_0x7a += i7a
    offs.sort()
    return offs, complete_0x7a, incomplete_0x7a


def iter_fog_resources(path: str, jobs: int = 1, stats: dict | None = None):
    """Yield (source_id, raw_0x7a_bytes) for every complete 0x7a fog-volume resource in `path`."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as f:
                data = f.read(WC.CAP + 1)
            if len(data) >= 0x50 and data[0x10:0x14] == b"0DRP" and data[0x14] == 0x7A:
                if STRUCT_VOLUME in parse_schema(data)[0]:
                    yield name, data
        return

    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"9FPR":
        with open(path, "rb") as f:
            data = f.read(WC.CAP + 1)
        if STRUCT_VOLUME in parse_schema(data)[0]:
            yield os.path.basename(path), data
        return

    size = os.path.getsize(path)
    offs, complete_0x7a, incomplete_0x7a = _scan_rpf_offsets(path, jobs=jobs)
    if stats is not None:
        stats["scanned_0x7a_complete"] = complete_0x7a
        stats["scanned_0x7a_incomplete"] = incomplete_0x7a
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for off in offs:
            try:
                res = WC.assemble(m, off, size)
            except zlib.error:
                continue
            if res and res.get("complete"):
                yield off, res["data"]


def summarize(path: str, jobs: int = 1) -> dict:
    """Aggregate counts and numeric distributions (no game strings or names)."""
    resources = 0
    volumes = 0
    distinct_digests: set[bytes] = set()
    per_res: list[int] = []
    by_vol_count: collections.Counter[str] = collections.Counter()
    offsets: list[int] = []
    scan_stats: dict[str, int] = {}
    is_box: collections.Counter[str] = collections.Counter()
    blend_mode: collections.Counter[str] = collections.Counter()
    int_ext: collections.Counter[str] = collections.Counter()
    vol_type: collections.Counter[str] = collections.Counter()
    prof_idx: collections.Counter[str] = collections.Counter()
    sec_prof_idx: collections.Counter[str] = collections.Counter()
    prof_interp_nz = 0
    at_origin = 0
    pos_offset_nz = 0
    rot_nz = 0
    shear_nz = 0
    rot_mtx_id = 0
    tr_mtx_id = 0
    pos_axes: list[list[float]] = [[], [], []]
    po_axes: list[list[float]] = [[], [], []]
    sc_axes: list[list[float]] = [[], [], []]

    for src, data in iter_fog_resources(path, jobs=jobs, stats=scan_stats):
        vols = parse_fog_volumes(data, include_digest=True)
        resources += 1
        if isinstance(src, int):
            offsets.append(src)
        per_res.append(len(vols))
        by_vol_count[str(len(vols))] += 1
        for v in vols:
            volumes += 1
            distinct_digests.add(v["_sha256"])
            if "IsBox" in v:
                is_box[str(v["IsBox"])] += 1
            if "BlendMode" in v:
                blend_mode[str(v["BlendMode"])] += 1
            if "IntExtMode" in v:
                int_ext[str(v["IntExtMode"])] += 1
            if "Type" in v:
                vol_type[str(v["Type"])] += 1
            if "ProfileIndex" in v:
                prof_idx[str(v["ProfileIndex"])] += 1
            if "SecondaryProfileIndex" in v:
                sec_prof_idx[str(v["SecondaryProfileIndex"])] += 1
            if v.get("ProfileInterpolation") and abs(v["ProfileInterpolation"]) > 1e-6:
                prof_interp_nz += 1
            pos = v.get("Position")
            if pos:
                for i in range(3):
                    pos_axes[i].append(pos[i])
                if pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0:
                    at_origin += 1
            po = v.get("PositionOffset")
            if po:
                for i in range(3):
                    po_axes[i].append(po[i])
                if any(abs(c) > 1e-6 for c in po):
                    pos_offset_nz += 1
            sc = v.get("Scale")
            if sc:
                for i in range(3):
                    sc_axes[i].append(sc[i])
            rot = v.get("Rotation")
            if rot and any(abs(c) > 1e-6 for c in rot):
                rot_nz += 1
            sh = v.get("Shear")
            if sh and any(abs(c) > 1e-6 for c in sh):
                shear_nz += 1
            if _is_identity3(v.get("RotationMtx")):
                rot_mtx_id += 1
            if _is_identity3(v.get("TransformMtx")):
                tr_mtx_id += 1

    def _rng(axes):
        if not axes[0]:
            return None
        return {
            "x": [round(min(axes[0]), 4), round(max(axes[0]), 4)],
            "y": [round(min(axes[1]), 4), round(max(axes[1]), 4)],
            "z": [round(min(axes[2]), 4), round(max(axes[2]), 4)],
        }

    out = {
        "resources": resources,
        "volumes": volumes,
        "distinct_serialized_volumes": len(distinct_digests),
        "volumes_per_resource": {
            "min": min(per_res) if per_res else 0,
            "max": max(per_res) if per_res else 0,
            "median": sorted(per_res)[len(per_res) // 2] if per_res else 0,
        },
        "resources_by_volume_count": dict(sorted(by_vol_count.items(), key=lambda kv: int(kv[0]))),
        "offset_range": [hex(min(offsets)), hex(max(offsets))] if offsets else None,
        "is_box": dict(sorted(is_box.items())),
        "blend_mode": dict(sorted(blend_mode.items())),
        "int_ext_mode": dict(sorted(int_ext.items())),
        "type": dict(sorted(vol_type.items())),
        "profile_index": dict(sorted(prof_idx.items(), key=lambda kv: int(kv[0]))),
        "secondary_profile_index": dict(sorted(sec_prof_idx.items(), key=lambda kv: int(kv[0]))),
        "profile_interpolation_nonzero": prof_interp_nz,
        "position_at_origin": at_origin,
        "position_offset_nonzero": pos_offset_nz,
        "rotation_nonzero": rot_nz,
        "shear_nonzero": shear_nz,
        "rotation_mtx_identity": rot_mtx_id,
        "transform_mtx_identity": tr_mtx_id,
        "position_range": _rng(pos_axes),
        "position_offset_range": _rng(po_axes),
        "scale_range": _rng(sc_axes),
    }
    out.update(scan_stats)
    return out


def _write_json_atomic(path: str, doc: dict) -> None:
    out_abs = os.path.abspath(path)
    parent = os.path.dirname(out_abs)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fog-volumes-", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_abs)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="RPF9 archive, carved 0x7a file, or folder of carved files")
    ap.add_argument("--out", help="optional JSON summary output path")
    ap.add_argument("--jobs", type=WC.job_count, default=min(16, os.cpu_count() or 4),
                    help="worker processes for RPF9 scanning (at least 1)")
    a = ap.parse_args(argv)
    if not os.path.exists(a.path):
        ap.error(f"no such file or folder: {a.path}")
    if a.out:
        if os.path.isdir(a.out):
            ap.error(f"--out is a directory: {a.out}")
        if os.path.isfile(a.path) and same_report_path(a.path, a.out):
            ap.error(f"--out would overwrite input: {a.out}")
    try:
        doc = summarize(a.path, jobs=a.jobs)
        if a.out:
            _write_json_atomic(a.out, doc)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
