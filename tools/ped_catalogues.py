"""Parse and resolve resource-packed `0DRP` v122 and standalone `PSIN` ped appearance catalogues (`0x8dcb4cf5`).

Ped appearance catalogues exist in two container framings that share the exact same schema and
internal `(block = ptr & 0xfffff, offset = ptr >> 20)` pointer encoding:

1. Standalone `PSIN` containers (e.g. player/wardrobe catalogues):
   Standard `PSIN` block table + `LSCH`/`LENM`/`LRSC` sections (`tools/psin_data.py`), with root
   struct `0x8dcb4cf5`.
2. Resource-packed `0DRP` v122 containers (compiled per-archetype ped catalogues inside RPF9):
   Standard `0x60`-byte `0DRP`/`0x7a` resource header (`tools/odrp_7a.py`): 16-byte RAGE system
   prefix (`8` zero bytes + `0x50000000` page-map pointer), `b"0DRP"` + version `122` at `+0x10`,
   1-based root block index at `+0x20`, struct table at `+0x28`, enum table at `+0x30`, block
   directory at `+0x38` (`(u32 key, u32 size, u64 virtual_ptr)`), and `(ns, ne, nb, u4e)` `u16`
   counts at `+0x48..+0x4e`.

Schema chain parsed by this module:
- `0x8dcb4cf5` (192 B root): two `0x60bcb6f3` preset buckets at `+0x00`/`+0x30` (each carrying a
  `0x5c` array of `0xf795e8ba` presets at `+0x08`) and two `0x4c70a132` secondary buckets at
  `+0x60`/`+0x90`.
- `0xf795e8ba` (88 B preset): `id` (`+0x00`), `parent_id` (`+0x04`, cycle-guarded inheritance),
  `0x5c` state array (`+0x08` count, `+0x10` ptr -> `0x37e52277`), `category` (`+0x18`), and
  `0x71` tag map `0x12f85293` (`+0x30` ptr, `+0x38` count -> 24 B slots in synthetic block
  `0x100..0x1ff` -> `u32` values in block `0x4a`).
- `0x37e52277` (184 B drawable/texture state): `state_id` (`+0x00`), `flags` (`+0x04`), embedded
  56 B `0x147fdab5` at `+0x08` (`drawable_hash` `h0` at `+0x08`, `albedo_hash` at `+0x10`,
  `normal_hash` at `+0x18`, `extra_hash` at `+0x20`, `0x71` extension map `0xb70a22c1` at
  `+0x28` with 16 B `(u32 key, pad4, u64 target_ptr)` slots), and state tag map `0x12f85293` at
  `+0xa0` (`+0xa8` ptr, `+0xb0` count).
- `0x147fdab5` -> `0xb70a22c1` -> key `0x46e1a048` (`joaat("cull_drawable")`) -> `0x3d780b74`
  (16 B record: 8 B zero header at `+0x00`, `u64` `murmur64` cull drawable hash `0x7deba6f8` at
  `+0x08`).

Synthetic map-slot block IDs (`0x100..0x1ff`) depend on per-file schema emission order and are
never hardcoded to `0x100` or `0x101`. Standard library only; no game data is bundled.

Usage:
    python -I tools/ped_catalogues.py census  <file|dir|archive.rpf> [--json] [--out FILE]
                                              [--start N] [--limit BYTES] [--max-inflate BYTES]
    python -I tools/ped_catalogues.py inspect <file> [--json] [--out FILE] [--limit N]
    python -I tools/ped_catalogues.py resolve <file|dir|archive.rpf> [--name NAME ...]
                                              [--names-file FILE] [--hash HEX ...] [--cull-only]
                                              [--require-textures] [--json] [--out FILE]
                                              [--start N] [--limit BYTES] [--max-inflate BYTES]
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import json
import mmap
import os
import re
import struct
import sys
import zlib
from typing import Iterable, Iterator

# -I keeps the script directory off sys.path, so add it explicitly before importing sibling modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import odrp_7a  # noqa: E402
import psin_data  # noqa: E402
from report_paths import same_report_path, path_key  # noqa: E402
import world_names  # noqa: E402

# Schema & struct hashes (measured over retail packs.rpf 0DRP + PSIN catalogues).
CATALOG_ROOT_HASH = 0x8DCB4CF5
CATALOG_ROOT_SIZE = 192
PRESET_BUCKET_HASH = 0x60BCB6F3
SECONDARY_BUCKET_HASH = 0x4C70A132
PRESET_STRUCT_HASH = 0xF795E8BA
PRESET_STRIDE = 88
STATE_STRUCT_HASH = 0x37E52277
STATE_STRIDE = 184
STATE_INNER_HASH = 0x147FDAB5
STATE_INNER_OFFSET = 0x08
EXT_MAP_FIELD_HASH = 0xB70A22C1
EXT_MAP_OFFSET = 0x28  # 0x37e52277 + 0x08 (0x147fdab5) + 0x20 (0xb70a22c1)
EXT_MAP_SLOT_SIZE = 16
TAG_MAP_FIELD_HASH = 0x12F85293
TAG_MAP_SLOT_SIZE = 24
TAG_VALUE_BLOCK_TYPE = 0x4A

# Extension map keys and target structs.
CULL_DRAWABLE_KEY = 0x46E1A048  # joaat("cull_drawable")
CULL_DRAWABLE_STRUCT_HASH = 0x3D780B74
CULL_DRAWABLE_STRIDE = 16
CULL_DRAWABLE_HASH_FIELD = 0x7DEBA6F8

# Well-known tag keys and preset categories.
DRAWABLE_TAG_KEYS = (0x4BFD27DD, 0x1C114E61)
HEAD_CATEGORY = 0x378AD10C
BODY_SKIN_CATEGORIES = frozenset((0x3817131D, 0x44925BFD, 0x9D283797))

# Synthetic anonymous map-slot struct IDs vary per file with schema emission order.
SYNTHETIC_BLOCK_MIN = 0x100
SYNTHETIC_BLOCK_MAX = 0x1FF

# Bounds and safety guards against malformed or cyclic inputs.
MAX_TAG_VALUES = 256
MAX_PARENT_DEPTH = 16
MAX_INFLATE = 64 << 20
MAX_CONTINUATIONS = 2048
RPF9_MAGIC = b"9FPR"
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")


def decode_ptr(ptr: int) -> tuple[int, int]:
    """Decode a PSIN/0DRP internal pointer into `(block_number_1based, byte_offset)`."""
    return ptr & 0xFFFFF, ptr >> 20


def joaat32(text: str, *, case_sensitive: bool = False) -> int:
    """32-bit Jenkins one-at-a-time hash matching RAGE name/tag hashes."""
    return world_names.joaat(text, case_sensitive=case_sensitive)


def murmur64(text: str) -> int:
    """64-bit asset hash: low 64-bit word of MurmurHash3 x64_128 (seed 0) over UTF-8 bytes."""
    return world_names.hash64(text)


@dataclass(frozen=True)
class CullDrawableRef:
    """One resolved `0x147fdab5` -> `0xb70a22c1` (`0x46e1a048`) -> `0x3d780b74` link."""

    key: int
    block: int
    offset: int
    target_type: int
    drawable_hash: int
    header_qword: int = 0

    def to_dict(self) -> dict:
        return {
            "key": f"0x{self.key:08x}",
            "block": self.block,
            "offset": self.offset,
            "target_type": f"0x{self.target_type:08x}",
            "drawable_hash": f"{self.drawable_hash:016x}",
            "header_qword": f"0x{self.header_qword:016x}",
        }


@dataclass(frozen=True)
class StateRecord:
    """One 184-byte `0x37e52277` drawable/texture state record."""

    index: int
    block: int
    offset: int
    state_id: int
    flags: int
    drawable_hash: int
    albedo_hash: int
    normal_hash: int
    extra_hash: int
    cull_drawable: int | None
    cull_ref: CullDrawableRef | None
    ext_map_keys: tuple[int, ...]
    tags: dict[int, tuple[int, ...]]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "block": self.block,
            "offset": self.offset,
            "state_id": f"0x{self.state_id:08x}",
            "flags": self.flags,
            "drawable_hash": f"{self.drawable_hash:016x}",
            "albedo_hash": f"{self.albedo_hash:016x}",
            "normal_hash": f"{self.normal_hash:016x}",
            "extra_hash": f"{self.extra_hash:016x}",
            "cull_drawable": f"{self.cull_drawable:016x}" if self.cull_drawable is not None else None,
            "cull_ref": self.cull_ref.to_dict() if self.cull_ref is not None else None,
            "ext_map_keys": [f"0x{k:08x}" for k in self.ext_map_keys],
            "tags": {f"0x{k:08x}": [f"0x{v:08x}" for v in vs] for k, vs in sorted(self.tags.items())},
        }


@dataclass(frozen=True)
class PresetRecord:
    """One 88-byte `0xf795e8ba` preset record."""

    block: int
    index: int
    preset_id: int
    parent_id: int
    category: int
    tags: dict[int, tuple[int, ...]]
    states: tuple[StateRecord, ...]

    def to_dict(self) -> dict:
        return {
            "block": self.block,
            "index": self.index,
            "preset_id": f"0x{self.preset_id:08x}",
            "parent_id": f"0x{self.parent_id:08x}",
            "category": f"0x{self.category:08x}",
            "tags": {f"0x{k:08x}": [f"0x{v:08x}" for v in vs] for k, vs in sorted(self.tags.items())},
            "states": [s.to_dict() for s in self.states],
        }


@dataclass(frozen=True)
class CatalogueContainer:
    """Parsed `0x8dcb4cf5` ped appearance catalogue from a `0DRP` v122 or `PSIN` container."""

    source: str
    framing: str
    root_block: int
    root_type: int
    block_count: int
    presets: tuple[PresetRecord, ...]
    presets_by_id: dict[int, PresetRecord]
    warnings: tuple[str, ...]
    stats: dict[str, int]

    def summary_dict(self) -> dict:
        return {
            "source": self.source,
            "framing": self.framing,
            "root_block": self.root_block,
            "root_type": f"0x{self.root_type:08x}",
            "block_count": self.block_count,
            "stats": dict(self.stats),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ResolvedBinding:
    """One resolved drawable/texture/cull_drawable binding from a preset state."""

    source: str
    framing: str
    preset_id: int
    parent_chain: tuple[int, ...]
    category: int
    state_index: int
    match_kind: str
    matched_hash: int
    matched_hash_bits: int
    matched_name: str | None
    drawable_hash: int
    albedo_hash: int
    normal_hash: int
    extra_hash: int
    cull_drawable_hash: int | None
    cull_drawable_name: str | None = None

    def to_dict(self) -> dict:
        fmt = f"0x{self.matched_hash:08x}" if self.matched_hash_bits == 32 else f"{self.matched_hash:016x}"
        out = {
            "source": self.source,
            "framing": self.framing,
            "preset_id": f"0x{self.preset_id:08x}",
            "parent_chain": [f"0x{p:08x}" for p in self.parent_chain],
            "category": f"0x{self.category:08x}",
            "state_index": self.state_index,
            "match_kind": self.match_kind,
            "matched_hash": fmt,
            "drawable_hash": f"{self.drawable_hash:016x}",
            "albedo_hash": f"{self.albedo_hash:016x}",
            "normal_hash": f"{self.normal_hash:016x}",
            "extra_hash": f"{self.extra_hash:016x}",
            "cull_drawable_hash": (
                f"{self.cull_drawable_hash:016x}" if self.cull_drawable_hash is not None else None
            ),
        }
        if self.matched_name is not None:
            out["matched_name"] = self.matched_name
        if self.cull_drawable_name is not None:
            out["cull_drawable_name"] = self.cull_drawable_name
        return out


# ---------------------------------------------------------------- container parsing


def _parse_0drp_blocks_state(
    data: bytes | bytearray | memoryview,
) -> tuple[str, int, int, dict[int, tuple[int, memoryview]], str]:
    """Parse a resource-packed `0DRP` v122 block directory.

    Returns `(status, root_block, root_type, blocks, error)` where `status` is one of:
    - `"ok"`: complete block directory parsed and all virtual blocks in bounds
    - `"truncated"`: valid `0DRP` v122 header, but block table or a virtual block extends past `len(data)`
    - `"invalid"`: not a valid `0DRP` v122 resource or malformed table/pointers
    """
    mv = memoryview(data)
    hdr = odrp_7a.parse_standalone_header(mv)
    if hdr is None:
        return "invalid", 0, 0, {}, "not a valid standalone 0DRP/0x7a header"
    bp = hdr["tables"]["q38"]
    nb = hdr["blocks"]
    rblk = hdr["root_objects"]
    if bp + nb * 16 > len(mv):
        return "truncated", rblk, 0, {}, "0DRP block directory extends past buffer end"

    blocks: dict[int, tuple[int, memoryview]] = {}
    for i in range(nb):
        key, size, bptr = struct.unpack_from("<IIQ", mv, bp + i * 16)
        if size > (1 << 24):
            return "invalid", rblk, 0, {}, f"0DRP block {i + 1} exceeds 16 MiB cap"
        voff = odrp_7a.virtual_offset(bptr)
        if voff is not None:
            if voff + size > len(mv):
                return "truncated", rblk, 0, {}, f"0DRP block {i + 1} extends past buffer end"
            blocks[i + 1] = (key, mv[voff:voff + size])
            continue
        if odrp_7a.physical_offset(bptr) is not None:
            continue
        return "invalid", rblk, 0, {}, f"0DRP block {i + 1} has invalid segment pointer 0x{bptr:x}"

    if not (1 <= rblk <= nb) or rblk not in blocks:
        return "invalid", rblk, 0, {}, f"0DRP root block index {rblk} is missing or out of range (1..{nb})"
    return "ok", rblk, blocks[rblk][0], blocks, ""


def parse_0drp_blocks(
    data: bytes | bytearray | memoryview,
) -> tuple[int, int, dict[int, tuple[int, memoryview]]] | None:
    """Parse a complete resource-packed `0DRP` v122 buffer into `(root_block, root_type, blocks)`."""
    status, rblk, rtype, blocks, _err = _parse_0drp_blocks_state(data)
    if status != "ok":
        return None
    return rblk, rtype, blocks


def _parse_psin_blocks_state(
    data: bytes | bytearray | memoryview,
) -> tuple[str, int, int, dict[int, tuple[int, memoryview]], str]:
    """Parse a standalone `PSIN` container's blocks into `(status, root_block, root_type, blocks, error)`."""
    raw = bytes(data)
    if len(raw) < 16 or raw[:4] != odrp_7a.PSIN_MAGIC:
        return "invalid", 0, 0, {}, "not a PSIN container"
    probe = odrp_7a._parse_psin_container(raw)
    if probe is not None and probe.get("needs_continuation"):
        return "truncated", 0, 0, {}, probe.get("error") or "PSIN container truncated"
    try:
        container = psin_data.Container(raw)
    except (ValueError, struct.error, IndexError) as exc:
        return "invalid", 0, 0, {}, str(exc)
    mv = memoryview(raw)
    blocks = {i + 1: (kind, mv[off:off + size]) for i, (kind, off, size) in enumerate(container.blocks)}
    rblk = container.root
    if not (1 <= rblk <= len(container.blocks)) or rblk not in blocks:
        return "invalid", rblk, 0, {}, f"PSIN root block index {rblk} out of range"
    return "ok", rblk, blocks[rblk][0], blocks, ""


def parse_psin_blocks(
    data: bytes | bytearray | memoryview,
) -> tuple[int, int, dict[int, tuple[int, memoryview]]] | None:
    """Parse a complete standalone `PSIN` container into `(root_block, root_type, blocks)`."""
    status, rblk, rtype, blocks, _err = _parse_psin_blocks_state(data)
    if status != "ok":
        return None
    return rblk, rtype, blocks


# ---------------------------------------------------------------- record parsing


def read_tag_map(
    blocks: dict[int, tuple[int, memoryview]],
    map_ptr: int,
    count: int,
    *,
    warnings: list[str] | None = None,
    observed_block_types: collections.Counter | None = None,
) -> dict[int, tuple[int, ...]]:
    """Read a `0x71` sorted tag map (`0x12f85293`, 24-byte slots -> `0x4a` `u32` arrays).

    Synthetic block IDs (`0x100..0x1ff`) depend on per-file schema emission order and are not
    hardcoded to `0x100`.
    """
    tags: dict[int, tuple[int, ...]] = {}
    if not count or not map_ptr:
        return tags
    tb, to = decode_ptr(map_ptr)
    if tb not in blocks:
        if warnings is not None:
            warnings.append(f"tag map points to missing block {tb}")
        return tags
    bk, tdata = blocks[tb]
    if not (SYNTHETIC_BLOCK_MIN <= bk <= SYNTHETIC_BLOCK_MAX):
        if warnings is not None:
            warnings.append(f"tag map points to non-synthetic block type 0x{bk:x}")
        return tags
    if observed_block_types is not None:
        observed_block_types[bk] += 1
    if to < 0 or to + count * TAG_MAP_SLOT_SIZE > len(tdata):
        if warnings is not None:
            warnings.append(f"tag map truncated at block {tb}+0x{to:x} (count={count})")
    max_slots = max(0, (len(tdata) - to) // TAG_MAP_SLOT_SIZE) if to >= 0 else 0
    for j in range(min(count, max_slots)):
        slot_off = to + j * TAG_MAP_SLOT_SIZE
        tk, tc, _tcap, tpp = struct.unpack_from("<IxxxxIIQ", tdata, slot_off)
        if not tc:
            tags[tk] = ()
            continue
        if tc > MAX_TAG_VALUES:
            if warnings is not None:
                warnings.append(f"tag key 0x{tk:08x} value count {tc} exceeds cap {MAX_TAG_VALUES}")
            continue
        ub, uo = decode_ptr(tpp)
        if ub not in blocks or blocks[ub][0] != TAG_VALUE_BLOCK_TYPE:
            if warnings is not None:
                warnings.append(f"tag key 0x{tk:08x} value block {ub} missing or not 0x4a")
            continue
        udata = blocks[ub][1]
        if uo < 0 or uo + tc * 4 > len(udata):
            if warnings is not None:
                warnings.append(f"tag key 0x{tk:08x} value array truncated in block {ub}+0x{uo:x}")
        avail = max(0, (len(udata) - uo) // 4) if uo >= 0 else 0
        vals = tuple(struct.unpack_from("<I", udata, uo + 4 * q)[0] for q in range(min(tc, avail)))
        tags[tk] = vals
    return tags


def read_cull_drawable(
    blocks: dict[int, tuple[int, memoryview]],
    ext_ptr: int,
    ext_count: int,
    *,
    warnings: list[str] | None = None,
    observed_block_types: collections.Counter | None = None,
) -> tuple[CullDrawableRef | None, tuple[int, ...]]:
    """Resolve `0x147fdab5` -> `0xb70a22c1` (`0x46e1a048` `"cull_drawable"`) -> `0x3d780b74`.

    Returns `(cull_ref, all_ext_map_keys)`. Synthetic block IDs (`0x100..0x1ff`) are matched
    dynamically rather than hardcoding `0x101`.
    """
    if not ext_count or not ext_ptr:
        return None, ()
    mb, mo = decode_ptr(ext_ptr)
    if mb not in blocks:
        if warnings is not None:
            warnings.append(f"0xb70a22c1 map points to missing block {mb}")
        return None, ()
    mbk, mdata = blocks[mb]
    if not (SYNTHETIC_BLOCK_MIN <= mbk <= SYNTHETIC_BLOCK_MAX):
        if warnings is not None:
            warnings.append(f"0xb70a22c1 map points to non-synthetic block type 0x{mbk:x}")
        return None, ()
    if observed_block_types is not None:
        observed_block_types[mbk] += 1
    if mo < 0 or mo + ext_count * EXT_MAP_SLOT_SIZE > len(mdata):
        if warnings is not None:
            warnings.append(f"0xb70a22c1 map truncated at block {mb}+0x{mo:x} (count={ext_count})")
    max_slots = max(0, (len(mdata) - mo) // EXT_MAP_SLOT_SIZE) if mo >= 0 else 0
    keys: list[int] = []
    cull_ref: CullDrawableRef | None = None
    for j in range(min(ext_count, max_slots)):
        mk, mp = struct.unpack_from("<IxxxxQ", mdata, mo + j * EXT_MAP_SLOT_SIZE)
        keys.append(mk)
        if mk != CULL_DRAWABLE_KEY or cull_ref is not None:
            continue
        tb, to = decode_ptr(mp)
        if tb not in blocks:
            if warnings is not None:
                warnings.append(f"cull_drawable points to missing block {tb}")
            continue
        tbk, tdata = blocks[tb]
        if tbk != CULL_DRAWABLE_STRUCT_HASH:
            if warnings is not None:
                warnings.append(f"cull_drawable target block {tb} has type 0x{tbk:08x}, expected 0x3d780b74")
            continue
        if to < 0 or to + CULL_DRAWABLE_STRIDE > len(tdata):
            if warnings is not None:
                warnings.append(f"cull_drawable record truncated at block {tb}+0x{to:x}")
            continue
        header_qword, cull_hash = struct.unpack_from("<QQ", tdata, to)
        cull_ref = CullDrawableRef(
            key=mk,
            block=tb,
            offset=to,
            target_type=tbk,
            drawable_hash=cull_hash,
            header_qword=header_qword,
        )
    return cull_ref, tuple(keys)


def parse_catalogue_blocks(
    blocks: dict[int, tuple[int, memoryview]],
    root_block: int,
    *,
    framing: str,
    source: str = "",
) -> CatalogueContainer:
    """Parse a `0x8dcb4cf5` ped appearance catalogue from a validated 1-based block dictionary."""
    if root_block not in blocks:
        raise ValueError(f"root block {root_block} not present in block table")
    root_type, root_data = blocks[root_block]
    if root_type != CATALOG_ROOT_HASH:
        raise ValueError(f"root type is 0x{root_type:08x}, expected 0x{CATALOG_ROOT_HASH:08x}")

    warnings: list[str] = []
    tag_block_types: collections.Counter[int] = collections.Counter()
    ext_block_types: collections.Counter[int] = collections.Counter()

    # Measure preset count referenced by the root struct's two 0x60bcb6f3 buckets.
    root_bucket_presets = 0
    if len(root_data) >= 96:
        for slot in range(2):
            cnt, _cap, _ptr = struct.unpack_from("<IIQ", root_data, slot * 48 + 8)
            root_bucket_presets += cnt
    else:
        warnings.append(f"root 0x8dcb4cf5 block size {len(root_data)} < 96")

    presets: list[PresetRecord] = []
    presets_by_id: dict[int, PresetRecord] = {}
    duplicate_preset_ids = 0
    states_total = 0
    states_with_drawable = 0
    states_with_albedo = 0
    states_with_normal = 0
    states_with_extra = 0
    states_with_tags = 0
    states_with_cull_drawable = 0
    cull_nonzero_hashes = 0

    for bnum in sorted(blocks):
        bk, data = blocks[bnum]
        if bk != PRESET_STRUCT_HASH:
            continue
        if len(data) % PRESET_STRIDE != 0:
            warnings.append(f"0xf795e8ba block {bnum} size {len(data)} is not a multiple of {PRESET_STRIDE}")
        for idx in range(len(data) // PRESET_STRIDE):
            base = idx * PRESET_STRIDE
            pid, parent_id, c1, _cap1, p1, cat = struct.unpack_from("<IIIIQI", data, base)
            p2 = struct.unpack_from("<Q", data, base + 48)[0]
            c2 = struct.unpack_from("<H", data, base + 56)[0]
            tags = read_tag_map(blocks, p2, c2, warnings=warnings, observed_block_types=tag_block_types)

            states: list[StateRecord] = []
            if c1:
                rb, ro = decode_ptr(p1)
                if rb not in blocks:
                    warnings.append(f"preset 0x{pid:08x} state array points to missing block {rb}")
                elif blocks[rb][0] != STATE_STRUCT_HASH:
                    warnings.append(
                        f"preset 0x{pid:08x} state array points to block type 0x{blocks[rb][0]:08x}, expected 0x37e52277"
                    )
                else:
                    rdata = blocks[rb][1]
                    if ro < 0 or ro + c1 * STATE_STRIDE > len(rdata):
                        warnings.append(f"preset 0x{pid:08x} state array truncated at block {rb}+0x{ro:x} (count={c1})")
                    avail = max(0, (len(rdata) - ro) // STATE_STRIDE) if ro >= 0 else 0
                    for si in range(min(c1, avail)):
                        sbase = ro + si * STATE_STRIDE
                        state_id, sflags = struct.unpack_from("<II", rdata, sbase)
                        h0, alb, nrm, h3 = struct.unpack_from("<QQQQ", rdata, sbase + STATE_INNER_OFFSET)
                        _ext_flag, ext_ptr, ext_cnt, _ext_cap = struct.unpack_from("<QQHH", rdata, sbase + EXT_MAP_OFFSET)
                        cull_ref, ext_keys = read_cull_drawable(
                            blocks,
                            ext_ptr,
                            ext_cnt,
                            warnings=warnings,
                            observed_block_types=ext_block_types,
                        )
                        sp2 = struct.unpack_from("<Q", rdata, sbase + 168)[0]
                        sc2 = struct.unpack_from("<H", rdata, sbase + 176)[0]
                        stags = (
                            read_tag_map(blocks, sp2, sc2, warnings=warnings, observed_block_types=tag_block_types)
                            if sc2
                            else {}
                        )
                        cull_hash = cull_ref.drawable_hash if cull_ref is not None else None
                        states.append(
                            StateRecord(
                                index=si,
                                block=rb,
                                offset=sbase,
                                state_id=state_id,
                                flags=sflags,
                                drawable_hash=h0,
                                albedo_hash=alb,
                                normal_hash=nrm,
                                extra_hash=h3,
                                cull_drawable=cull_hash,
                                cull_ref=cull_ref,
                                ext_map_keys=ext_keys,
                                tags=stags,
                            )
                        )
                        states_total += 1
                        if h0:
                            states_with_drawable += 1
                        if alb:
                            states_with_albedo += 1
                        if nrm:
                            states_with_normal += 1
                        if h3:
                            states_with_extra += 1
                        if stags:
                            states_with_tags += 1
                        if cull_ref is not None:
                            states_with_cull_drawable += 1
                            if cull_ref.drawable_hash:
                                cull_nonzero_hashes += 1

            rec = PresetRecord(
                block=bnum,
                index=idx,
                preset_id=pid,
                parent_id=parent_id,
                category=cat,
                tags=tags,
                states=tuple(states),
            )
            presets.append(rec)
            if pid in presets_by_id:
                duplicate_preset_ids += 1
            else:
                presets_by_id[pid] = rec

    # Measure parent_id inheritance graph properties across the catalogue.
    presets_with_parent = 0
    max_parent_depth = 0
    parent_cycles = 0
    unresolved_parents = 0
    for rec in presets:
        if not rec.parent_id:
            continue
        presets_with_parent += 1
        cur = rec.parent_id
        depth = 0
        seen = {rec.preset_id}
        while cur:
            if cur in seen:
                parent_cycles += 1
                warnings.append(f"parent_id cycle detected starting at preset 0x{rec.preset_id:08x}")
                break
            parent = presets_by_id.get(cur)
            if parent is None:
                unresolved_parents += 1
                break
            seen.add(cur)
            depth += 1
            if depth >= MAX_PARENT_DEPTH:
                warnings.append(f"parent_id chain depth reached cap ({MAX_PARENT_DEPTH}) at preset 0x{rec.preset_id:08x}")
                break
            cur = parent.parent_id
        if depth > max_parent_depth:
            max_parent_depth = depth

    state_records_in_blocks = sum(
        len(data) // STATE_STRIDE for bk, data in blocks.values() if bk == STATE_STRUCT_HASH
    )
    cull_records_in_blocks = sum(
        len(data) // CULL_DRAWABLE_STRIDE for bk, data in blocks.values() if bk == CULL_DRAWABLE_STRUCT_HASH
    )

    stats: dict[str, int] = {
        "presets": len(presets),
        "root_bucket_presets": root_bucket_presets,
        "duplicate_preset_ids": duplicate_preset_ids,
        "presets_with_parent": presets_with_parent,
        "max_parent_depth": max_parent_depth,
        "parent_cycles": parent_cycles,
        "unresolved_parents": unresolved_parents,
        "states": states_total,
        "state_records_in_blocks": state_records_in_blocks,
        "states_with_drawable": states_with_drawable,
        "states_with_albedo": states_with_albedo,
        "states_with_normal": states_with_normal,
        "states_with_extra": states_with_extra,
        "states_with_tags": states_with_tags,
        "states_with_cull_drawable": states_with_cull_drawable,
        "cull_nonzero_hashes": cull_nonzero_hashes,
        "cull_records_in_blocks": cull_records_in_blocks,
    }
    for bk, cnt in sorted(tag_block_types.items()):
        stats[f"tag_map_block_0x{bk:x}"] = cnt
    for bk, cnt in sorted(ext_block_types.items()):
        stats[f"ext_map_block_0x{bk:x}"] = cnt

    return CatalogueContainer(
        source=source,
        framing=framing,
        root_block=root_block,
        root_type=root_type,
        block_count=len(blocks),
        presets=tuple(presets),
        presets_by_id=presets_by_id,
        warnings=tuple(warnings),
        stats=stats,
    )


def parse_catalogue_strict(
    data: bytes | bytearray | memoryview,
    *,
    source: str = "",
) -> CatalogueContainer:
    """Parse one `0DRP` v122 or `PSIN` `0x8dcb4cf5` catalogue; raise `ValueError` on any format error."""
    mv = memoryview(data)
    if len(mv) >= 16 and mv[:4].tobytes() == odrp_7a.PSIN_MAGIC:
        status, rblk, _rtype, blocks, err = _parse_psin_blocks_state(mv)
        if status != "ok":
            raise ValueError(f"PSIN parse failed ({status}): {err}")
        return parse_catalogue_blocks(blocks, rblk, framing="psin", source=source)

    if len(mv) >= 20 and mv[16:20].tobytes() == odrp_7a.SIGNATURE:
        status, rblk, _rtype, blocks, err = _parse_0drp_blocks_state(mv)
        if status != "ok":
            raise ValueError(f"0DRP parse failed ({status}): {err}")
        return parse_catalogue_blocks(blocks, rblk, framing="0drp", source=source)

    raise ValueError("input is neither a PSIN container nor a standalone 0DRP/0x7a resource")


def parse_catalogue(
    data: bytes | bytearray | memoryview,
    *,
    source: str = "",
) -> CatalogueContainer | None:
    """Parse a `0x8dcb4cf5` catalogue from a `0DRP` v122 or `PSIN` buffer, or return `None` if not one."""
    try:
        return parse_catalogue_strict(data, source=source)
    except ValueError:
        return None


# ---------------------------------------------------------------- inheritance & resolution


def resolve_ancestor_tags(
    catalogue: CatalogueContainer,
    preset: PresetRecord,
    tag_keys: tuple[int, ...] = DRAWABLE_TAG_KEYS,
    *,
    max_depth: int = MAX_PARENT_DEPTH,
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...], bool]:
    """Walk `parent_id` ancestors with cycle protection until an ancestor defines one of `tag_keys`.

    Returns `(matched_tags_subset, parent_chain_traversed, hit_cycle)`.
    """
    if not preset.parent_id:
        return {}, (), False
    visited = {preset.preset_id}
    chain: list[int] = []
    cur = preset.parent_id
    while cur and len(chain) < max_depth:
        if cur in visited:
            return {}, tuple(chain), True
        parent = catalogue.presets_by_id.get(cur)
        if parent is None:
            break
        visited.add(cur)
        chain.append(cur)
        found = {k: parent.tags[k] for k in tag_keys if k in parent.tags and parent.tags[k]}
        if found:
            return found, tuple(chain), False
        cur = parent.parent_id
    return {}, tuple(chain), False


def resolve_effective_tags(
    catalogue: CatalogueContainer,
    preset: PresetRecord,
    *,
    max_depth: int = MAX_PARENT_DEPTH,
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...], bool]:
    """Merge tags along the `parent_id` chain (child keys override ancestor keys) with cycle guard."""
    visited = {preset.preset_id}
    ancestors: list[PresetRecord] = [preset]
    chain: list[int] = []
    hit_cycle = False
    cur = preset.parent_id
    while cur and len(chain) < max_depth:
        if cur in visited:
            hit_cycle = True
            break
        parent = catalogue.presets_by_id.get(cur)
        if parent is None:
            break
        visited.add(cur)
        chain.append(cur)
        ancestors.append(parent)
        cur = parent.parent_id
    merged: dict[int, tuple[int, ...]] = {}
    for rec in reversed(ancestors):
        merged.update(rec.tags)
    return merged, tuple(chain), hit_cycle


def resolve_cull_map(catalogue: CatalogueContainer) -> dict[int, list[int]]:
    """Map non-zero state `drawable_hash` (`h0`) -> sorted unique `cull_drawable` hashes."""
    mapping: dict[int, set[int]] = collections.defaultdict(set)
    for preset in catalogue.presets:
        for state in preset.states:
            if state.drawable_hash and state.cull_drawable:
                mapping[state.drawable_hash].add(state.cull_drawable)
    return {h0: sorted(culls) for h0, culls in sorted(mapping.items())}


def resolve_preset_bindings(
    catalogue: CatalogueContainer,
    *,
    drawable_names: Iterable[str] | None = None,
    filter_hashes_32: set[int] | None = None,
    filter_hashes_64: set[int] | None = None,
    require_textures: bool = False,
    cull_only: bool = False,
    max_depth: int = MAX_PARENT_DEPTH,
) -> list[ResolvedBinding]:
    """Resolve drawable, texture, and `cull_drawable` bindings across all presets in `catalogue`.

    Resolution precedence per state mirrors runtime catalogue lookup:
    1. Direct preset drawable tag (`0x4bfd27dd` / `0x1c114e61`, `joaat32`).
    2. Direct state drawable tag (`0x4bfd27dd` / `0x1c114e61`, `joaat32`).
    3. Inherited ancestor drawable tag via `parent_id` chain (`joaat32`, cycle-guarded).
    4. Direct state drawable `h0` (`murmur64`).
    5. Direct `cull_drawable` target (`murmur64`, when `h0 == 0` or matched by a `cull_drawable` query).
    """
    j32_names: dict[int, list[str]] = collections.defaultdict(list)
    m64_names: dict[int, list[str]] = collections.defaultdict(list)
    want_32: set[int] = set(filter_hashes_32 or ())
    want_64: set[int] = set(filter_hashes_64 or ())
    if drawable_names is not None:
        for raw in drawable_names:
            name = raw.strip()
            if not name:
                continue
            h32 = joaat32(name)
            h64 = murmur64(name)
            j32_names[h32].append(name)
            m64_names[h64].append(name)
            want_32.add(h32)
            want_64.add(h64)
    filtering = bool(want_32 or want_64 or drawable_names is not None)

    out: list[ResolvedBinding] = []
    for preset in catalogue.presets:
        preset_dtags = [v for tk in DRAWABLE_TAG_KEYS for v in preset.tags.get(tk, ()) if v]
        inherited_tags: dict[int, tuple[int, ...]] | None = None
        inherited_chain: tuple[int, ...] = ()

        for state in preset.states:
            if require_textures and not state.albedo_hash and not state.normal_hash:
                continue
            if cull_only and not state.cull_drawable:
                continue

            state_dtags = [v for tk in DRAWABLE_TAG_KEYS for v in state.tags.get(tk, ()) if v]
            cull_name = m64_names[state.cull_drawable][0] if state.cull_drawable in m64_names else None

            candidates: list[tuple[str, int, int, tuple[int, ...]]] = []
            for v in preset_dtags:
                candidates.append(("drawable_tag", v, 32, ()))
            for v in state_dtags:
                if v not in preset_dtags:
                    candidates.append(("state_drawable_tag", v, 32, ()))
            if not candidates and preset.parent_id:
                if inherited_tags is None:
                    inherited_tags, inherited_chain, _cyc = resolve_ancestor_tags(
                        catalogue, preset, DRAWABLE_TAG_KEYS, max_depth=max_depth
                    )
                for tk in DRAWABLE_TAG_KEYS:
                    for v in inherited_tags.get(tk, ()):
                        if v:
                            candidates.append(("inherited_drawable_tag", v, 32, inherited_chain))
            if not candidates and state.drawable_hash:
                candidates.append(("state_drawable_hash", state.drawable_hash, 64, ()))
            if not candidates and state.cull_drawable:
                candidates.append(("cull_drawable_hash", state.cull_drawable, 64, ()))

            matched_any = False
            for kind, mhash, bits, pchain in candidates:
                if filtering:
                    if bits == 32 and mhash not in want_32:
                        continue
                    if bits == 64 and mhash not in want_64:
                        continue
                names = j32_names.get(mhash) if bits == 32 else m64_names.get(mhash)
                if names:
                    for nm in names:
                        out.append(
                            ResolvedBinding(
                                source=catalogue.source,
                                framing=catalogue.framing,
                                preset_id=preset.preset_id,
                                parent_chain=pchain,
                                category=preset.category,
                                state_index=state.index,
                                match_kind=kind,
                                matched_hash=mhash,
                                matched_hash_bits=bits,
                                matched_name=nm,
                                drawable_hash=state.drawable_hash,
                                albedo_hash=state.albedo_hash,
                                normal_hash=state.normal_hash,
                                extra_hash=state.extra_hash,
                                cull_drawable_hash=state.cull_drawable,
                                cull_drawable_name=cull_name,
                            )
                        )
                        matched_any = True
                else:
                    out.append(
                        ResolvedBinding(
                            source=catalogue.source,
                            framing=catalogue.framing,
                            preset_id=preset.preset_id,
                            parent_chain=pchain,
                            category=preset.category,
                            state_index=state.index,
                            match_kind=kind,
                            matched_hash=mhash,
                            matched_hash_bits=bits,
                            matched_name=None,
                            drawable_hash=state.drawable_hash,
                            albedo_hash=state.albedo_hash,
                            normal_hash=state.normal_hash,
                            extra_hash=state.extra_hash,
                            cull_drawable_hash=state.cull_drawable,
                            cull_drawable_name=cull_name,
                        )
                    )
                    matched_any = True

            # When filtering by name/hash, also match direct h0 or cull_drawable even if preset had a tag.
            if filtering and not matched_any:
                if state.drawable_hash and state.drawable_hash in want_64:
                    names = m64_names.get(state.drawable_hash) or [None]
                    for nm in names:
                        out.append(
                            ResolvedBinding(
                                source=catalogue.source,
                                framing=catalogue.framing,
                                preset_id=preset.preset_id,
                                parent_chain=(),
                                category=preset.category,
                                state_index=state.index,
                                match_kind="state_drawable_hash",
                                matched_hash=state.drawable_hash,
                                matched_hash_bits=64,
                                matched_name=nm,
                                drawable_hash=state.drawable_hash,
                                albedo_hash=state.albedo_hash,
                                normal_hash=state.normal_hash,
                                extra_hash=state.extra_hash,
                                cull_drawable_hash=state.cull_drawable,
                                cull_drawable_name=cull_name,
                            )
                        )
                elif state.cull_drawable and state.cull_drawable in want_64:
                    names = m64_names.get(state.cull_drawable) or [None]
                    for nm in names:
                        out.append(
                            ResolvedBinding(
                                source=catalogue.source,
                                framing=catalogue.framing,
                                preset_id=preset.preset_id,
                                parent_chain=(),
                                category=preset.category,
                                state_index=state.index,
                                match_kind="cull_drawable_hash",
                                matched_hash=state.cull_drawable,
                                matched_hash_bits=64,
                                matched_name=nm,
                                drawable_hash=state.drawable_hash,
                                albedo_hash=state.albedo_hash,
                                normal_hash=state.normal_hash,
                                extra_hash=state.extra_hash,
                                cull_drawable_hash=state.cull_drawable,
                                cull_drawable_name=cull_name,
                            )
                        )
    return out


# ---------------------------------------------------------------- source scanners


def _is_rpf_file(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 4:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) == RPF9_MAGIC


def _inflate_at(buf: mmap.mmap | bytes, pos: int, end: int, max_inflate: int) -> tuple[bytes, int] | None:
    """Inflate one complete zlib stream at `pos` inside `[pos, end)`. Returns `(data, stored_len)` or `None`."""
    d = zlib.decompressobj()
    p = pos
    total = 0
    chunks: list[bytes] = []
    try:
        while not d.eof and p < end:
            block = buf[p:min(p + (1 << 20), end)]
            if not block:
                break
            p += len(block)
            while block and not d.eof:
                chunk = d.decompress(block, 1 << 20)
                block = d.unconsumed_tail
                if chunk:
                    total += len(chunk)
                    if total > max_inflate:
                        return None
                    chunks.append(chunk)
                if not block:
                    break
        if not d.eof:
            tail = d.flush()
            if tail:
                total += len(tail)
                if total > max_inflate:
                    return None
                chunks.append(tail)
    except zlib.error:
        return None
    if not d.eof or total == 0:
        return None
    stored = (p - pos) - len(d.unused_data)
    if stored <= 0:
        return None
    return b"".join(chunks), stored


def _peek_zlib_head(buf: mmap.mmap | bytes, pos: int, end: int) -> bytes | None:
    """Quickly inflate up to `0x60` bytes of a candidate zlib stream to test for `PSIN` or `0DRP` v122."""
    try:
        d = zlib.decompressobj()
        return d.decompress(buf[pos:min(pos + 512, end)], odrp_7a.STANDALONE_HEADER_SIZE)
    except zlib.error:
        return None


def _try_parse_candidate(
    blob: bytes | bytearray,
    *,
    source: str,
) -> tuple[str, CatalogueContainer | None]:
    """Return `("catalogue", cat)`, `("truncated", None)`, or `("skip", None)`."""
    if len(blob) >= 16 and blob[:4] == odrp_7a.PSIN_MAGIC:
        status, rblk, rtype, blocks, _err = _parse_psin_blocks_state(blob)
        if status == "truncated":
            return "truncated", None
        if status == "ok" and rtype == CATALOG_ROOT_HASH:
            return "catalogue", parse_catalogue_blocks(blocks, rblk, framing="psin", source=source)
        return "skip", None

    if len(blob) >= odrp_7a.STANDALONE_HEADER_SIZE and blob[16:20] == odrp_7a.SIGNATURE:
        status, rblk, rtype, blocks, _err = _parse_0drp_blocks_state(blob)
        if status == "truncated":
            return "truncated", None
        if status == "ok" and rtype == CATALOG_ROOT_HASH:
            return "catalogue", parse_catalogue_blocks(blocks, rblk, framing="0drp", source=source)
        return "skip", None

    return "skip", None


def iter_catalogues_in_file(path: str, *, max_inflate: int = MAX_INFLATE) -> Iterator[CatalogueContainer]:
    """Yield a `0x8dcb4cf5` catalogue if `path` contains one (raw `0DRP`/`PSIN` or single zlib stream)."""
    if os.path.getsize(path) > max_inflate:
        return
    with open(path, "rb") as fh:
        data = fh.read()
    cat = parse_catalogue(data, source=path)
    if cat is not None:
        yield cat
        return
    # Also accept a single zlib-compressed 0DRP/PSIN stream.
    if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
        inflated = _inflate_at(data, 0, len(data), max_inflate)
        if inflated is not None:
            cat = parse_catalogue(inflated[0], source=path)
            if cat is not None:
                yield cat


def iter_catalogues_in_dir(path: str, *, max_inflate: int = MAX_INFLATE) -> Iterator[CatalogueContainer]:
    """Yield all `0x8dcb4cf5` catalogues in a carved directory (joining contiguous streams when `index.json` exists)."""
    manifest = os.path.join(path, "index.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("files"), list):
            rows = sorted(
                (r for r in doc["files"] if isinstance(r, dict) and isinstance(r.get("offset"), int)),
                key=lambda r: r["offset"],
            )
            i = 0
            while i < len(rows):
                row = rows[i]
                fname = row.get("file") or f"{row['offset']:012x}.bin"
                fpath = os.path.join(path, fname)
                if not os.path.isfile(fpath) or os.path.getsize(fpath) > max_inflate:
                    i += 1
                    continue
                with open(fpath, "rb") as fh:
                    head = fh.read(odrp_7a.STANDALONE_HEADER_SIZE)
                    if not (
                        (len(head) >= 16 and head[:4] == odrp_7a.PSIN_MAGIC)
                        or odrp_7a.parse_standalone_header(head) is not None
                    ):
                        i += 1
                        continue
                    buf = bytearray(head + fh.read())
                src = f"{path}:{fname}"
                status, cat = _try_parse_candidate(buf, source=src)
                j = i
                streams = 1
                while status == "truncated" and j + 1 < len(rows) and streams <= MAX_CONTINUATIONS:
                    stored = rows[j].get("stored")
                    if not isinstance(stored, int):
                        break
                    next_off = (rows[j]["offset"] + stored + 15) & ~15
                    if rows[j + 1]["offset"] != next_off:
                        break
                    nname = rows[j + 1].get("file") or f"{rows[j + 1]['offset']:012x}.bin"
                    npath = os.path.join(path, nname)
                    if not os.path.isfile(npath):
                        break
                    with open(npath, "rb") as fh:
                        chunk = fh.read()
                    if len(buf) + len(chunk) > max_inflate:
                        break
                    if (len(chunk) >= 16 and chunk[:4] == odrp_7a.PSIN_MAGIC) or (
                        odrp_7a.parse_standalone_header(chunk) is not None
                    ):
                        break
                    buf.extend(chunk)
                    j += 1
                    streams += 1
                    status, cat = _try_parse_candidate(buf, source=src)
                if status == "catalogue" and cat is not None:
                    yield cat
                i = j + 1
            return

    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            fpath = os.path.join(root, name)
            yield from iter_catalogues_in_file(fpath, max_inflate=max_inflate)


def iter_catalogues_in_rpf(
    path: str,
    *,
    start: int | None = None,
    limit: int | None = None,
    max_inflate: int = MAX_INFLATE,
) -> Iterator[CatalogueContainer]:
    """Walk an RPF9 archive data area and yield every `0x8dcb4cf5` `0DRP` and `PSIN` catalogue.

    Uses a 96-byte zlib header peek to skip non-`PSIN`/non-`0DRP` streams without inflating them,
    and joins contiguous continuation streams automatically when a catalogue spans multiple zlib blocks.
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("empty file")
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            size = len(mm)
            first = struct.unpack_from("<I", mm, 4)[0] if size >= 8 else 0
            pos = start if start is not None else first
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit is not None else size
            while pos < end:
                found = ZHDR.search(mm, pos, end)
                if not found:
                    break
                s = found.start()
                if mm[s:s + 3] == b"KB2":
                    blen = (struct.unpack_from("<I", mm, s + 4)[0] + 8) if s + 8 <= end else 0
                    w = struct.unpack_from("<I", mm, s + 0x14)[0] if s + 0x1C <= end else 0
                    h = struct.unpack_from("<I", mm, s + 0x18)[0] if s + 0x1C <= end else 0
                    if 64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192:
                        pos = s + blen
                        continue
                    pos = s + 1
                    continue

                head = _peek_zlib_head(mm, s, end)
                if head is None or not (
                    (len(head) >= 16 and head[:4] == odrp_7a.PSIN_MAGIC)
                    or odrp_7a.parse_standalone_header(head) is not None
                ):
                    pos = s + 1
                    continue

                inflated = _inflate_at(mm, s, end, max_inflate)
                if inflated is None:
                    pos = s + 1
                    continue
                data, stored = inflated
                buf = bytearray(data)
                src = f"{path}@0x{s:x}"
                status, cat = _try_parse_candidate(buf, source=src)
                cur_end = s + stored
                streams = 1
                while status == "truncated" and streams <= MAX_CONTINUATIONS:
                    next_off = (cur_end + 15) & ~15
                    if next_off + 2 > end:
                        break
                    if not (mm[next_off] == 0x78 and mm[next_off + 1] in (0x01, 0x5E, 0x9C, 0xDA)):
                        break
                    nxt = _inflate_at(mm, next_off, end, max_inflate - len(buf))
                    if nxt is None:
                        break
                    ndata, nstored = nxt
                    if (len(ndata) >= 16 and ndata[:4] == odrp_7a.PSIN_MAGIC) or (
                        odrp_7a.parse_standalone_header(ndata) is not None
                    ):
                        break
                    buf.extend(ndata)
                    cur_end = next_off + nstored
                    streams += 1
                    status, cat = _try_parse_candidate(buf, source=src)
                if status == "catalogue" and cat is not None:
                    yield cat
                pos = cur_end


def iter_catalogues(
    source: str,
    *,
    start: int | None = None,
    limit: int | None = None,
    max_inflate: int = MAX_INFLATE,
) -> Iterator[CatalogueContainer]:
    """Dispatch `source` (file, carved directory, or `.rpf` archive) to the matching scanner."""
    if os.path.isdir(source):
        yield from iter_catalogues_in_dir(source, max_inflate=max_inflate)
    elif _is_rpf_file(source):
        yield from iter_catalogues_in_rpf(source, start=start, limit=limit, max_inflate=max_inflate)
    elif os.path.isfile(source):
        yield from iter_catalogues_in_file(source, max_inflate=max_inflate)
    else:
        raise FileNotFoundError(f"{source}: no such file or folder")


def census(
    source: str,
    *,
    start: int | None = None,
    limit: int | None = None,
    max_inflate: int = MAX_INFLATE,
) -> dict:
    """Collect aggregate structural counts across all `0x8dcb4cf5` catalogues in `source`."""
    totals: collections.Counter[str] = collections.Counter()
    by_framing: collections.Counter[str] = collections.Counter()
    tag_map_blocks: collections.Counter[str] = collections.Counter()
    ext_map_blocks: collections.Counter[str] = collections.Counter()
    max_parent_depth = 0
    catalogues_with_cull = 0
    catalogues_with_warnings = 0
    warning_count = 0

    for cat in iter_catalogues(source, start=start, limit=limit, max_inflate=max_inflate):
        totals["catalogues"] += 1
        by_framing[cat.framing] += 1
        if cat.stats.get("states_with_cull_drawable", 0) > 0:
            catalogues_with_cull += 1
        if cat.warnings:
            catalogues_with_warnings += 1
            warning_count += len(cat.warnings)
        if cat.stats.get("max_parent_depth", 0) > max_parent_depth:
            max_parent_depth = cat.stats["max_parent_depth"]
        for k, v in cat.stats.items():
            if k == "max_parent_depth":
                continue
            if k.startswith("tag_map_block_"):
                tag_map_blocks[k.removeprefix("tag_map_block_")] += 1
            elif k.startswith("ext_map_block_"):
                ext_map_blocks[k.removeprefix("ext_map_block_")] += 1
            else:
                totals[k] += v

    return {
        "source": source,
        "catalogues": totals["catalogues"],
        "by_framing": dict(sorted(by_framing.items())),
        "presets": totals["presets"],
        "root_bucket_presets": totals["root_bucket_presets"],
        "duplicate_preset_ids": totals["duplicate_preset_ids"],
        "presets_with_parent": totals["presets_with_parent"],
        "max_parent_depth": max_parent_depth,
        "parent_cycles": totals["parent_cycles"],
        "unresolved_parents": totals["unresolved_parents"],
        "states": totals["states"],
        "state_records_in_blocks": totals["state_records_in_blocks"],
        "states_with_drawable": totals["states_with_drawable"],
        "states_with_albedo": totals["states_with_albedo"],
        "states_with_normal": totals["states_with_normal"],
        "states_with_extra": totals["states_with_extra"],
        "states_with_tags": totals["states_with_tags"],
        "catalogues_with_cull_drawable": catalogues_with_cull,
        "states_with_cull_drawable": totals["states_with_cull_drawable"],
        "cull_nonzero_hashes": totals["cull_nonzero_hashes"],
        "cull_records_in_blocks": totals["cull_records_in_blocks"],
        "tag_map_block_types": dict(sorted(tag_map_blocks.items())),
        "ext_map_block_types": dict(sorted(ext_map_blocks.items())),
        "catalogues_with_warnings": catalogues_with_warnings,
        "warnings": warning_count,
    }


# ---------------------------------------------------------------- CLI


def _nonneg_int(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer >= 0") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _pos_int(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _parse_hex_hash(text: str) -> tuple[int, int]:
    """Parse a 32-bit or 64-bit hex hash string into `(bits, value)`."""
    s = text.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if not s or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"invalid hex hash: {text!r}")
    val = int(s, 16)
    if len(s) <= 8:
        return 32, val
    if len(s) <= 16:
        return 64, val
    raise ValueError(f"hash wider than 64 bits: {text!r}")


def _is_under_dir(path: str, folder: str) -> bool:
    """Return True when `path` resolves to `folder` or any path beneath it."""
    folder_key = path_key(folder)
    target_key = path_key(path)
    try:
        return os.path.commonpath([folder_key, target_key]) == folder_key
    except ValueError:
        return False


def _check_out_path(out_path: str, inputs: list[str]) -> str | None:
    if os.path.isdir(out_path):
        return f"{out_path} is a folder, not a file"
    for inp in inputs:
        if not inp:
            continue
        if same_report_path(inp, out_path):
            return f"output must not overwrite input {inp}"
        if os.path.isdir(inp) and _is_under_dir(out_path, inp):
            return f"output must not be inside input folder {inp}"
    parent = os.path.dirname(os.path.abspath(out_path))
    if not os.path.isdir(parent):
        return f"{parent} is not a folder"
    return None


def _write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")


def _cmd_census(args: argparse.Namespace) -> int:
    if args.out:
        err = _check_out_path(args.out, [args.source])
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
    if not os.path.exists(args.source):
        print(f"ERROR: {args.source}: no such file or folder", file=sys.stderr)
        return 2
    if (args.start is not None or args.limit is not None) and not _is_rpf_file(args.source):
        print("ERROR: --start and --limit apply to an .rpf archive only", file=sys.stderr)
        return 2
    try:
        report = census(args.source, start=args.start, limit=args.limit, max_inflate=args.max_inflate)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {args.source}: {exc}", file=sys.stderr)
        return 2
    if args.out:
        _write_json(args.out, report)
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    print(f"source:                       {report['source']}")
    print(f"catalogues (0x8dcb4cf5):      {report['catalogues']:,}  {report['by_framing']}")
    print(f"presets (0xf795e8ba):         {report['presets']:,}  (root buckets: {report['root_bucket_presets']:,})")
    print(
        f"presets with parent_id:       {report['presets_with_parent']:,}  "
        f"(max depth: {report['max_parent_depth']}, cycles: {report['parent_cycles']}, "
        f"unresolved: {report['unresolved_parents']})"
    )
    print(
        f"states (0x37e52277):          {report['states']:,}  "
        f"(blocks: {report['state_records_in_blocks']:,}, "
        f"h0: {report['states_with_drawable']:,}, "
        f"albedo: {report['states_with_albedo']:,}, "
        f"normal: {report['states_with_normal']:,})"
    )
    print(
        f"cull_drawable (0x3d780b74):   {report['states_with_cull_drawable']:,} links in "
        f"{report['catalogues_with_cull_drawable']:,} catalogues "
        f"(non-zero: {report['cull_nonzero_hashes']:,}, block records: {report['cull_records_in_blocks']:,})"
    )
    print(f"tag map block types:          {report['tag_map_block_types']}")
    print(f"ext map block types:          {report['ext_map_block_types']}")
    if report["warnings"]:
        print(f"warnings:                     {report['warnings']:,} across {report['catalogues_with_warnings']:,} catalogues")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    if args.out:
        err = _check_out_path(args.out, [args.file])
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
    if not os.path.isfile(args.file):
        print(f"ERROR: {args.file}: no such file", file=sys.stderr)
        return 2
    try:
        with open(args.file, "rb") as fh:
            data = fh.read()
        if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
            inflated = _inflate_at(data, 0, len(data), MAX_INFLATE)
            if inflated is not None:
                data = inflated[0]
        cat = parse_catalogue_strict(data, source=args.file)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {args.file}: {exc}", file=sys.stderr)
        return 2

    payload = cat.summary_dict()
    if args.limit > 0:
        payload["presets"] = [p.to_dict() for p in cat.presets[:args.limit]]
    if args.out:
        _write_json(args.out, payload)
    if args.json:
        json.dump(payload, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    print(f"{cat.source}  [{cat.framing}] root=block {cat.root_block} (0x{cat.root_type:08x}) blocks={cat.block_count}")
    for k, v in sorted(cat.stats.items()):
        print(f"  {k:32} {v:,}")
    if cat.warnings:
        print(f"  warnings ({len(cat.warnings)}):")
        for w in cat.warnings[:10]:
            print(f"    - {w}")
    if args.limit > 0:
        shown = min(args.limit, len(cat.presets))
        print(f"  first {shown} preset(s):")
        for p in cat.presets[:shown]:
            culls = sum(1 for s in p.states if s.cull_drawable is not None)
            print(
                f"    preset 0x{p.preset_id:08x} parent=0x{p.parent_id:08x} "
                f"cat=0x{p.category:08x} tags={len(p.tags)} states={len(p.states)} culls={culls}"
            )
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    inputs = [args.source]
    if args.names_file:
        inputs.append(args.names_file)
    if args.out:
        err = _check_out_path(args.out, inputs)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
    if not os.path.exists(args.source):
        print(f"ERROR: {args.source}: no such file or folder", file=sys.stderr)
        return 2
    if (args.start is not None or args.limit is not None) and not _is_rpf_file(args.source):
        print("ERROR: --start and --limit apply to an .rpf archive only", file=sys.stderr)
        return 2

    names: list[str] = list(args.name or ())
    if args.names_file:
        try:
            with open(args.names_file, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        names.append(s)
        except OSError as exc:
            print(f"ERROR: {args.names_file}: {exc}", file=sys.stderr)
            return 2

    h32: set[int] = set()
    h64: set[int] = set()
    for raw in args.hash or ():
        try:
            bits, val = _parse_hex_hash(raw)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if bits == 32:
            h32.add(val)
        else:
            h64.add(val)

    bindings: list[ResolvedBinding] = []
    catalogues_scanned = 0
    try:
        for cat in iter_catalogues(args.source, start=args.start, limit=args.limit, max_inflate=args.max_inflate):
            catalogues_scanned += 1
            bindings.extend(
                resolve_preset_bindings(
                    cat,
                    drawable_names=names if names else None,
                    filter_hashes_32=h32 or None,
                    filter_hashes_64=h64 or None,
                    require_textures=args.require_textures,
                    cull_only=args.cull_only,
                )
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {args.source}: {exc}", file=sys.stderr)
        return 2

    payload = {
        "source": args.source,
        "catalogues_scanned": catalogues_scanned,
        "matches": len(bindings),
        "bindings": [b.to_dict() for b in bindings],
    }
    if args.out:
        _write_json(args.out, payload)
    if args.json:
        json.dump(payload, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    print(f"scanned {catalogues_scanned:,} catalogue(s) -> {len(bindings):,} resolved binding(s)")
    for b in bindings[:50]:
        label = b.matched_name or (f"0x{b.matched_hash:08x}" if b.matched_hash_bits == 32 else f"{b.matched_hash:016x}")
        cull = f" cull={b.cull_drawable_hash:016x}" if b.cull_drawable_hash is not None else ""
        chain = f" via={[f'0x{p:08x}' for p in b.parent_chain]}" if b.parent_chain else ""
        print(
            f"  {label} [{b.match_kind}{chain}] preset=0x{b.preset_id:08x} "
            f"h0={b.drawable_hash:016x} alb={b.albedo_hash:016x} nrm={b.normal_hash:016x}{cull}"
        )
    if len(bindings) > 50:
        print(f"  ... ({len(bindings) - 50:,} more; use --json or --out FILE for full output)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Parse and resolve 0DRP v122 and PSIN ped appearance catalogues (0x8dcb4cf5)."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_cen = sub.add_parser("census", help="count 0x8dcb4cf5 catalogues, presets, states, and cull_drawable links")
    p_cen.add_argument("source", help=".pso/.0drp file, carved directory, or .rpf archive")
    p_cen.add_argument("--json", action="store_true", help="print counts as JSON")
    p_cen.add_argument("--out", help="write JSON report to FILE")
    p_cen.add_argument("--start", type=_nonneg_int, help="first RPF archive offset to scan")
    p_cen.add_argument("--limit", type=_pos_int, help="maximum RPF archive bytes to scan")
    p_cen.add_argument("--max-inflate", type=_pos_int, default=MAX_INFLATE, help="inflate cap in bytes (default 64 MiB)")
    p_cen.set_defaults(func=_cmd_census)

    p_ins = sub.add_parser("inspect", help="parse one standalone .pso or .0drp file strictly")
    p_ins.add_argument("file", help="path to one .pso or .0drp file")
    p_ins.add_argument("--json", action="store_true", help="print inspection result as JSON")
    p_ins.add_argument("--out", help="write JSON result to FILE")
    p_ins.add_argument("--limit", type=_nonneg_int, default=10, help="number of presets to include (default 10, 0 for summary)")
    p_ins.set_defaults(func=_cmd_inspect)

    p_res = sub.add_parser("resolve", help="resolve preset inheritance, textures, and cull_drawable swaps")
    p_res.add_argument("source", help=".pso/.0drp file, carved directory, or .rpf archive")
    p_res.add_argument("--name", action="append", help="drawable name to filter by (repeatable)")
    p_res.add_argument("--names-file", help="newline-delimited drawable name list")
    p_res.add_argument("--hash", action="append", help="32-bit joaat or 64-bit murmur64 hex hash to filter by (repeatable)")
    p_res.add_argument("--cull-only", action="store_true", help="keep only states carrying a cull_drawable target")
    p_res.add_argument("--require-textures", action="store_true", help="keep only states with non-zero albedo or normal")
    p_res.add_argument("--json", action="store_true", help="print resolved bindings as JSON")
    p_res.add_argument("--out", help="write JSON result to FILE")
    p_res.add_argument("--start", type=_nonneg_int, help="first RPF archive offset to scan")
    p_res.add_argument("--limit", type=_pos_int, help="maximum RPF archive bytes to scan")
    p_res.add_argument("--max-inflate", type=_pos_int, default=MAX_INFLATE, help="inflate cap in bytes (default 64 MiB)")
    p_res.set_defaults(func=_cmd_resolve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
