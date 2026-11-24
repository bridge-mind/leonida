"""Look up a stored effect/technique pass in the Xbox 1.0.2.1 shaders.rpf.

python -I tools/sgd5_effects.py PATH/TO/shaders.rpf
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --list-effects
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --effect NAME
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --effect NAME --technique NAME
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --shader-hash HEX --technique NAME

The default JSON contains only aggregate counts. A named query also returns
the database, group, ordinal and archive offsets of the selected programs.
Group 0 contains vertex and mesh programs, 1 pixel, 2 geometry and 5 compute.
Groups 3 and 4 are empty in this build. All returned offsets are absolute.
Use carve_offset to join to shader_carve's index.json by offset. Repeated
effect or technique names return every match with its offset and row index.
--shader-hash accepts the Materials[].Shader hex value from an rpf9dump model manifest.
This reads plain metadata in place. Runtime selection of a technique is unknown.
Usage: `python -I tools/sgd5_effects.py --help` lists the options.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import sys
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashdict import joaat
from shader_carve import ENTRY

# SGD5 signature, first native group count, end of the padded effect region.
DATABASES = ((0x9000, 0x10E39, 0x10287000), (0x10287000, 0x10289C81, 0x10579000))
IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{0,254}\x00")
POOL = re.compile(rb"[\x20-\x7e\x00]+")
STRING = re.compile(rb"[^\x00]+\x00")


def u32(data, pos=0):
    if pos + 4 > len(data):
        raise ValueError("truncated SGD5 count")
    return struct.unpack_from("<I", data, pos)[0]


def program_groups(stream, limit):
    groups = []
    for _ in range(6):
        count = u32(stream.read(4))
        offsets = []
        for _ in range(count):
            start = stream.tell()
            length = stream.read(1)
            if not length or length[0] < 2:
                raise ValueError("invalid program label length")
            label = stream.read(length[0])
            if len(label) != length[0] or label[-1:] != b"\x00":
                raise ValueError("truncated program label")
            stream.seek(1, 1)  # tag
            size = u32(stream.read(4))
            if size < 12 or stream.tell() + size + 156 > limit:
                raise ValueError("program crosses the database boundary")
            stream.seek(size + 152, 1)
            trailer = u32(stream.read(4))
            if stream.tell() + trailer > limit:
                raise ValueError("program trailer crosses the database boundary")
            stream.seek(trailer, 1)
            offsets.append(start)
        groups.append(offsets)
    return groups


def program_identity(stream, group, ordinal, start):
    stream.seek(start)
    length = stream.read(1)[0]
    # shader_carve.ENTRY can include a printable native length byte (and up to
    # 200 printable bytes in all). Match the same wire header to recover its key.
    terminator = start + length
    window_start = max(0, terminator - 200)
    stream.seek(window_start)
    window = stream.read(terminator + 18 - window_start)
    pos = terminator - window_start
    while pos and 0x21 <= window[pos - 1] <= 0x7e:
        pos -= 1
    match = ENTRY.fullmatch(window, pos)
    if match is None:
        raise ValueError(f"no shader_carve entry matching program at {start:#x}")
    return {"group": group, "ordinal": ordinal, "record_offset": start, "carve_offset": window_start + pos}


def program_lists(data, pos, groups):
    lists = []
    for group in groups:
        if pos + 4 > len(data):
            return None
        count = u32(data, pos)
        pos += 4
        after = pos + 4 * count
        if after > len(data):
            return None
        ordinals = []
        for (ordinal,) in struct.iter_unpack("<I", memoryview(data)[pos:after]):
            if ordinal >= len(group):
                return None
            ordinals.append(ordinal)
        lists.append(ordinals)
        pos = after
    return lists, pos


def advance(data, pos, size):
    if pos + size > len(data):
        raise ValueError("effect field exceeds the database boundary")
    return pos + size


def pool_after_passes(data, pos):
    # Counted words, a marker and two length-prefixed byte blocks.
    pos = advance(data, pos + 4, 4 * u32(data, pos))
    if u32(data, pos) != 0x80000000:
        raise ValueError("invalid post-pass marker")
    pos += 4
    for _ in range(2):
        pos = advance(data, pos + 4, u32(data, pos))
    # In 1.0.2.1 the observed entry counts fit in the low byte. Bits 8..23
    # are zero, and the high byte can be nonzero. Payload meanings are unknown.
    packed = u32(data, pos)
    if packed & 0x00ffff00:
        raise ValueError("unsupported packed post-pass count")
    pos += 4
    for _ in range(packed & 0xff):
        tag = u32(data, pos)
        pos = advance(data, pos + 4, 4 if tag == 0x80000000 else tag)
    return pos


class Effect(NamedTuple):
    start: int
    name: str
    lists: list
    technique_rows: int
    first_pass: int
    pass_end: int
    pool: int
    end: int


def effect_records(data, groups, count):
    start = 0
    for _ in range(count):
        if start >= len(data):
            raise ValueError("missing effect record")
        name_end = advance(data, start + 1, data[start])
        if not IDENTIFIER.fullmatch(data, start + 1, name_end):
            raise ValueError("invalid counted effect name")
        name = data[start + 1:name_end - 1].decode("ascii")
        arrays = program_lists(data, name_end + 4, groups)  # word after the name is opaque
        if arrays is None:
            raise ValueError("invalid effect program lists")
        lists, pos = arrays
        pos = advance(data, pos, 80)
        for width in (16, 24, 68, 36):
            pos = advance(data, pos + 4, width * u32(data, pos))
        rows = pos + 4
        pass_count_at = advance(data, rows, 4 * u32(data, pos))
        first = pass_count_at + 4
        pass_end = advance(data, first, 24 * u32(data, pass_count_at))
        pool = pool_after_passes(data, pass_end)
        end = advance(data, pool + 4, u32(data, pool))
        if (end == pool + 4 or data[pool + 4] != 0 or data[end - 1] != 0
                or not POOL.fullmatch(data, pool + 4, end)):
            raise ValueError("invalid counted string pool")
        yield Effect(start, name, lists, rows, first, pass_end, pool, end)
        start = end
    if not 4 <= len(data) - start <= 4096 or u32(data, start) != 1 or any(data[start + 4:]):
        raise ValueError("declared effects do not reach the database terminator and zero padding")


def techniques(data, effect):
    """Yield the stored name and passes, including zero-program passes."""
    names = {m.start() - effect.pool - 4: m.group()[:-1].decode("ascii")
             for m in STRING.finditer(data, effect.pool + 4, effect.end)}
    cursor = effect.first_pass
    rows = memoryview(data)[effect.technique_rows:cursor - 4]
    for name_bits, _, length in struct.iter_unpack("<HBB", rows):
        name = names.get(name_bits & 0x7fff)
        if name is None or cursor + 24 * length > effect.pass_end:
            raise ValueError("invalid technique name offset or pass count")
        passes = []
        for _ in range(length):
            programs = []
            for group, selector in enumerate(data[cursor:cursor + 6]):
                if selector > len(effect.lists[group]):
                    raise ValueError("pass selector is outside the effect's program list")
                if selector:
                    programs.append((group, effect.lists[group][selector - 1]))
            passes.append(programs)
            cursor += 24
        yield name, passes
    if cursor != effect.pass_end:
        raise ValueError("technique pass counts do not sum to the table count")


def read_database(stream, layout, effect=None, technique=None, *, list_effects=False, shader_hash=None):
    database, first_group, end = layout
    stream.seek(database)
    if stream.read(4) != b"SGD5":
        raise ValueError(f"expected Xbox 1.0.2.1 SGD5 database at {database:#x}")
    stream.seek(first_group)
    groups = program_groups(stream, end)
    count = u32(stream.read(4))
    base = stream.tell()
    if base >= end:
        raise ValueError("missing effect region")
    data = stream.read(end - base)
    if len(data) != end - base:
        raise ValueError("truncated effect region")
    result = {"database": database, "groups": list(map(len, groups)), "effects": count,
              "named_effects": 0, "techniques": 0, "passes": 0, "matches": []}
    for record in effect_records(data, groups, count):
        name = record.name
        key = joaat(name)
        selected = name == effect or key == shader_hash
        if list_effects:
            result["matches"].append({"effect": name, "shader_hash": f"{key:08X}", "effect_offset": base + record.start})
        listing = []
        for index, (stored_name, passes) in enumerate(techniques(data, record)):
            result["named_effects"] += index == 0
            result["techniques"] += 1
            result["passes"] += len(passes)
            if not selected:
                continue
            if technique is None:
                listing.append({"technique": stored_name, "technique_index": index, "pass_count": len(passes)})
            elif stored_name == technique:
                resolved = [[program_identity(stream, g, ordinal, groups[g][ordinal]) for g, ordinal in row]
                            for row in passes]
                result["matches"].append({"effect": name, "effect_offset": base + record.start,
                                          "technique_index": index, "passes": resolved})
        if selected and technique is None:
            result["matches"].append({"effect": name, "effect_offset": base + record.start, "techniques": listing})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--list-effects", action="store_true", help="list effect names and archive offsets locally")
    query.add_argument("--effect", help="exact effect name, or list its techniques when --technique is omitted")
    query.add_argument("--shader-hash", metavar="HEX", help="hexadecimal Materials[].Shader value from rpf9dump")
    parser.add_argument("--technique", help="exact technique name from that effect")
    args = parser.parse_args(argv)
    shader_hash = None
    if args.shader_hash is not None:
        if not re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,8}", args.shader_hash):
            parser.error("--shader-hash expects up to eight hexadecimal digits")
        shader_hash = int(args.shader_hash, 16)
    if args.technique is not None and args.effect is None and shader_hash is None:
        parser.error("--technique requires --effect or --shader-hash")
    try:
        with args.archive.open("rb") as stream:
            results = [read_database(stream, layout, args.effect, args.technique,
                                     list_effects=args.list_effects, shader_hash=shader_hash)
                       for layout in DATABASES]
        if (args.effect is not None or shader_hash is not None) and not any(row["matches"] for row in results):
            parser.error("no matching effect or technique; use --list-effects, then --effect NAME")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
