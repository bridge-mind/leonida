"""ASSET-70 draw-ownership tool: system-segment pointer layers, synthetically tested.

ASSET-70 tooling; used by the draw-ownership measurement campaign.

The pointer schema follows `tools/meshlet_page.py`'s `drawable_geometries()` and
`draw_binding()` with unchanged semantics (system pointers are 0x50000000-based,
graphics pointers are 0x60000000-based):

    drawable+80  -> LOD record (16B)
    LOD+8        -> used/capacity u16 pair; used == capacity, 1..256
    LOD+0        -> 8B model-pointer table (8 * used bytes)
    model+16     -> used/capacity u16 pair (geometry count)
    model+54     -> u16; must match the model+16 number
    model+8      -> 8B geometry-pointer entries table
    geometry     -> 160B record: +8 -> vertex-buffer record (80B: +8 count u32 /
                    stride u16, +24 -> GRAPHICS pointer), vb+64 -> vertex-layout
                    (320B), geometry+64 -> meshlet descriptor (16B: +0 -> 64B
                    index-buffer record; +8 meshlets u16 / triangles u16 / binding u32)

Layers:
    walk_chain / walk_drawable  walk the regular drawable -> LOD -> model -> geometry
                                chain; a schema violation raises a WalkError subclass
                                carrying the concrete reason (used != capacity, a
                                model+54 contradiction, a pointer outside the segment).
    reverse_pointers            read a u64 at every 8-byte-aligned position; every value
                                that is 0x50000000-based and points into the segment
                                becomes an edge, and the edges build a reverse index
                                (target -> source positions).
    classify_parent             classify the source records that point at one
                                geometry-entries table by the shape of the record
                                ("model-compatible" / "unknown parent"), with a 32-byte
                                hex dump of each source.
    geometry_shape_check        the system-side half of the geometry record (the part of
                                draw_binding that needs no graphics segment).

Tested on synthetic bytes only (tests/test_draw_ownership.py); it reads no real game data.
"""

import struct

PTR_BASE = 0x50000000
GRAPHICS_BASE = 0x60000000


class WalkError(Exception):
    """A schema violation in the drawable chain or the geometry shape; the message gives the concrete reason."""


class OutOfSegmentError(WalkError):
    """A pointer field, pointer target or record does not fit in the system segment."""


class CountError(WalkError):
    """A used/capacity u16 pair violates the schema (used != capacity, or used not in 1..256)."""


class ModelCountMismatchError(WalkError):
    """The model+16 used/capacity pair and the model+54 u16 contradict each other."""


def _u16(system, at):
    if not 0 <= at <= len(system) - 2:
        raise OutOfSegmentError(f"u16 field outside the segment: +0x{at:x}")
    return struct.unpack_from("<H", system, at)[0]


def _u32(system, at):
    if not 0 <= at <= len(system) - 4:
        raise OutOfSegmentError(f"u32 field outside the segment: +0x{at:x}")
    return struct.unpack_from("<I", system, at)[0]


def _u64(system, at):
    if not 0 <= at <= len(system) - 8:
        raise OutOfSegmentError(f"u64/pointer field outside the segment: +0x{at:x}")
    return struct.unpack_from("<Q", system, at)[0]


def _system_pointer(system, at, size, what):
    """A copy of meshlet_page._resource_pointer: a 0x50000000-based system pointer.

    The pointer field must sit in the segment, and its target needs at least `size`
    bytes of room left to the end of the segment.
    """
    raw = _u64(system, at)
    if raw < PTR_BASE:
        raise OutOfSegmentError(
            f"{what} pointer field +0x{at:x}: 0x{raw:016x} is not a 0x50000000-based system pointer")
    target = raw - PTR_BASE
    if not 0 <= target <= len(system) - size:
        raise OutOfSegmentError(
            f"{what} pointer +0x{at:x} targets +0x{target:x}, but {size} bytes do not fit "
            f"in the {len(system)}-byte system segment")
    return target


def _count(system, at, what):
    """A copy of the inner count() of meshlet_page.drawable_geometries(): a used/capacity
    u16 pair; used == capacity, 1..256."""
    if not 0 <= at <= len(system) - 4:
        raise OutOfSegmentError(f"{what} used/capacity field outside the segment: +0x{at:x}")
    used, capacity = struct.unpack_from("<HH", system, at)
    if used != capacity:
        raise CountError(f"{what} count field +0x{at:x}: used={used} != capacity={capacity}")
    if not 1 <= used <= 256:
        raise CountError(f"{what} count field +0x{at:x}: used={used} is not in the 1..256 range")
    return used


def walk_chain(system, drawable_off):
    """Walk the regular drawable -> LOD -> model -> geometry chain.

    Returns the whole chain: {"drawable": ..., "lod": ...,
    "models": [{"model": ..., "entries": ..., "geometries": [...]}, ...]}.
    A schema violation raises a WalkError subclass with the concrete reason.
    """
    if not 0 <= drawable_off <= len(system) - 176:
        raise OutOfSegmentError(
            f"drawable +0x{drawable_off:x} falls outside the {len(system)}-byte system segment (176 B needed)")
    lod = _system_pointer(system, drawable_off + 80, 16, "LOD")
    models_n = _count(system, lod + 8, f"LOD +0x{lod:x}")
    table = _system_pointer(system, lod, 8 * models_n, "model-pointer table")
    models = []
    for i in range(models_n):
        model = _system_pointer(system, table + 8 * i, 56, "model record")
        geo_n = _count(system, model + 16, f"model +0x{model:x}")
        at54 = _u16(system, model + 54)
        if at54 != geo_n:
            raise ModelCountMismatchError(
                f"model +0x{model:x}: the geometry count in the +16 used/capacity pair is {geo_n}, "
                f"but the +54 u16 is {at54}")
        entries = _system_pointer(system, model + 8, 8 * geo_n, "geometry-entries table")
        geometries = [_system_pointer(system, entries + 8 * j, 160, "geometry record")
                      for j in range(geo_n)]
        models.append({"model": model, "entries": entries, "geometries": geometries})
    return {"drawable": drawable_off, "lod": lod, "models": models}


def walk_drawable(system, drawable_off):
    """Enumerate the geometry offsets of the drawable's primary LOD (a generator).

    Exactly the schema of meshlet_page.drawable_geometries(); on a violation it raises
    a WalkError subclass (used != capacity, a model+54 contradiction, a pointer outside
    the segment).
    """
    for model in walk_chain(system, drawable_off)["models"]:
        yield from model["geometries"]


def reverse_pointers(system):
    """Reverse pointer analysis: read a u64 at every 8-byte-aligned position.

    Every value that is 0x50000000-based and points into the segment
    (0x50000000 <= value < 0x50000000 + len(system)) becomes an edge (source position ->
    target offset). Graphics pointers (0x60000000-based) and zeroes give no edge.
    Returns the reverse index: {target offset: [source positions, in ascending order]}.
    """
    index = {}
    low, high = PTR_BASE, PTR_BASE + len(system)
    unpack = struct.Struct("<Q").unpack_from
    for pos in range(0, len(system) - 7, 8):
        value = unpack(system, pos)[0]
        if low <= value < high:
            index.setdefault(value - PTR_BASE, []).append(pos)
    return index


def _hexdump(system, at, nbytes=32):
    """A hex dump of `nbytes` bytes from `at`, 16-byte rows with the address."""
    rows = []
    for row in range(0, nbytes, 16):
        chunk = system[at + row:at + row + 16]
        rows.append(f"+0x{at + row:06x}  {chunk.hex(' ')}")
    return "\n".join(rows)


def classify_parent(system, entries_table_off, index=None):
    """Classify what points at a geometry-entries table.

    From the reverse index (target -> sources) this finds the source positions that
    point at the table, and classifies each by the shape of the source record:
      - "model-compatible": the source corresponds to a model+8 field — the record at
        source-8 has a schema-shaped +16 used/capacity pair (used == capacity, 1..256)
        and its +54 u16 gives the same number;
      - "unknown parent": every other source-record shape.
    Each candidate gets a 32-byte hex dump from its surroundings.

    A reverse_pointers() index can be passed as `index`; when it is missing the tool
    builds one itself. Returns a list of {source, target, classification, ...} records,
    in ascending order of source position.
    """
    if index is None:
        index = reverse_pointers(system)
    out = []
    for src in index.get(entries_table_off, []):
        rec = {"source": src, "target": entries_table_off}
        model = src - 8
        if model >= 0 and model + 56 <= len(system):
            used, capacity = struct.unpack_from("<HH", system, model + 16)
            at54 = struct.unpack_from("<H", system, model + 54)[0]
            if 1 <= used <= 256 and used == capacity and at54 == used:
                rec["classification"] = "model-compatible"
                rec["model"] = model
                rec["geometry_count"] = used
                rec["model_referenced"] = bool(index.get(model))
            else:
                rec["classification"] = "unknown parent"
                rec["model_candidate"] = model
        else:
            rec["classification"] = "unknown parent"
        start = max(0, src - 8)
        rec["context_offset"] = start
        rec["context_dump"] = _hexdump(system, start, 32)
        out.append(rec)
    return out


def geometry_shape_check(system, geometry_off):
    """Check the system-side half of a geometry record (without the graphics segment).

    Repeats the system-side checks of draw_binding(): vertex-buffer (80B) extent and
    GRAPHICS-pointer base, vertex-layout (320B) format constants (+0=128, +0xD0=8,
    +0x104=16, count=flags>>10 from the +0x138), meshlet descriptor (16B:
    meshlets/triangles/refs non-zero, the binding's top bit zero) and index buffer
    (64B, stride 4, at least 12 bytes). It only reports the +24 graphics pointers by
    their base (whether they are 0x60000000-based); the 0x4042 / group header magics
    cannot be checked without the graphics segment. A schema violation raises WalkError.
    """
    if not 0 <= geometry_off <= len(system) - 160:
        raise OutOfSegmentError(
            f"geometry +0x{geometry_off:x} falls outside the {len(system)}-byte segment (160 B needed)")
    vb = _system_pointer(system, geometry_off + 8, 80, "vertex-buffer record")
    layout = _system_pointer(system, vb + 64, 320, "vertex-layout")
    descriptor = _system_pointer(system, geometry_off + 64, 16, "meshlet descriptor")
    ib = _system_pointer(system, descriptor, 64, "index-buffer record")
    for field_at, what in ((vb + 24, "vertex-buffer"), (ib + 24, "index-buffer")):
        if struct.unpack_from("<Q", system, field_at)[0] < GRAPHICS_BASE:
            raise WalkError(
                f"{what} +0x{field_at - 24:x}: the +24 pointer is not a 0x60000000-based graphics pointer")
    vcount, vstride = struct.unpack_from("<IH", system, vb + 8)
    if not vcount or not 1 <= vstride <= 32:
        raise WalkError(f"vertex-buffer +0x{vb:x}: invalid extent "
                        f"(count={vcount}, stride={vstride})")
    flags = _u32(system, layout + 0x138)
    vertices = flags >> 10
    if (not flags & 1 or not vertices or _u32(system, layout) != 128
            or system[layout + 0xD0] != 8 or system[layout + 0x104] != 16
            or vcount * vstride < 128 + vertices * 8):
        raise WalkError(f"vertex-layout +0x{layout:x}: unsupported format or position extent "
                        f"(flags=0x{flags:x})")
    meshlets, triangles, binding = struct.unpack_from("<HHI", system, descriptor + 8)
    refs = (binding >> 8) & 0x7FFFFF
    icount, istride = struct.unpack_from("<IH", system, ib + 8)
    if istride != 4 or icount * 4 < 12 or binding >> 31 or not (meshlets and triangles and refs):
        raise WalkError(f"meshlet descriptor +0x{descriptor:x}: invalid or unsupported binding "
                        f"(meshlets={meshlets}, triangles={triangles}, "
                        f"refs={refs}, index-buffer stride={istride})")
    return {"geometry": geometry_off, "vertex_buffer": vb, "vertex_layout": layout,
            "descriptor": descriptor, "index_buffer": ib, "vertices": vertices,
            "vertex_count": vcount, "vertex_stride": vstride,
            "meshlets": meshlets, "triangles": triangles, "refs": refs}
