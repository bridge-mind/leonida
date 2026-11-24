"""Match the bone names in anim.rpf's bone-table streams against the skeletons of packs.rpf (ASSET-231).

docs/animation.md ("Bone tables and motion matchers") lists four small anim.rpf streams that carry full `SKEL_*` /
`PH_*` bone names: a standalone 896-byte name list and three motion-matcher resources. Those pages carry no model
name, so which skeleton they belong to was not measured. A skeleton does carry its names: each bone record holds
`joaat32(lowercased name)` at +0x1C and a name pointer at +0x30 (docs/character_skeleton.md, read by
`anim_glb.read_skeleton`). This tool reads the names out of the four streams (`anim_index.parse_skel_names`), reads
every skeleton that `anim_glb` can read in packs.rpf (drawables, drawable-dictionary children and fragments, the
resource walk of `skin_streams`), and reports per stream and for their union:

- how many names have their `joaat32` among the joints of some skeleton, and whether the joint's own name string
  is the same spelling;
- how many distinct skeletons (by their ordered joint-hash list) and how many skeleton copies hold every name;
- the joint counts of the skeletons that hold every name of the union.

Only counts and joint counts are printed; no bone list or game bytes are written.

Usage:
    python -I tools/anim_skel_match.py <anim.rpf> <packs.rpf> [--stream OFFSET ...] [--json out.json]

The default streams are the four offsets in docs/animation.md (1.0.2.1 package). Exit 0 when the report was
written, 2 on a read error. Nothing here decrypts anything.
Usage: `python -I tools/anim_skel_match.py --help` lists the options.
"""
import argparse
import collections
import json
import mmap
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anim_index  # noqa: E402
from report_paths import same_report_path  # noqa: E402

DEFAULT_STREAMS = (0x101A32160, 0x101C379A0, 0x101C41D50, 0x101EA66C0)


def match(names_by_stream, skeletons):
    """names_by_stream: {label: [bone name]}; skeletons: [{"hashes": [u32], "names": [str or None]}].

    Returns the report described in the module docstring (a JSON-ready dict)."""
    union = sorted(set().union(*names_by_stream.values())) if names_by_stream else []
    hashes = {n: anim_index.joaat32(n.lower()) for n in union}
    distinct = collections.Counter(tuple(s["hashes"]) for s in skeletons)
    spellings = collections.defaultdict(set)
    for s in skeletons:
        for h, name in zip(s["hashes"], s["names"]):
            if name:
                spellings[h].add(name)
    every_hash = set(spellings) | {h for s in skeletons for h in s["hashes"]}
    joint_sets = [(set(k), len(k), copies) for k, copies in distinct.items()]

    def holding(names):
        want = {hashes[n] for n in names}
        rows = [(size, copies) for joints, size, copies in joint_sets if want <= joints]
        return {"distinct": len(rows), "copies": sum(c for _, c in rows), "joint_counts": sorted({s for s, _ in rows})}

    def summary(names):
        found = [n for n in names if hashes[n] in every_hash]
        return {"names": len(names), "in_a_skeleton": len(found),
                "same_spelling": sum(1 for n in found if n in spellings[hashes[n]]),
                "joint_unnamed": sum(1 for n in found if not spellings[hashes[n]]),
                "held_by_skeletons": holding(names)}

    return {"skeletons": len(skeletons), "distinct_skeletons": len(distinct),
            "streams": {label: summary(sorted(set(names))) for label, names in names_by_stream.items()},
            "union": summary(union),
            "not_in_any_skeleton": sum(1 for n in union if hashes[n] not in every_hash)}


def stream_names(anim_path, offsets):
    """{"0x...": [bone name]} for each stream offset of anim.rpf."""
    import skin_streams
    out = {}
    with open(anim_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for off in offsets:
            data, _used = skin_streams.inflate_at(m, off, len(m))
            if data is None:
                raise ValueError(f"no zlib stream at 0x{off:x} in {anim_path}")
            out[f"0x{off:x}"] = anim_index.parse_skel_names(data, off)["names"]
    return out


def read_skeletons(packs_path, progress=None):
    """Every skeleton anim_glb reads in the archive: [{"hashes": [...], "names": [...]}]."""
    import anim_glb
    import skin_streams as ss
    found = []
    size = os.path.getsize(packs_path)
    with open(packs_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        pos = 0
        while pos < size:
            page = (pos + ss.PAGE - 1) & ~(ss.PAGE - 1)
            at, pos = page + 16, page + ss.PAGE
            if at + 2 > size:
                break
            if m[at - 16:at] != b"\0" * 16 or not ss.ZHDR.match(m, at):
                continue
            data, _used = ss.inflate_at(m, at, size)
            if data is None or len(data) < 192:
                continue
            kind = ss.resource_kind(data)
            if kind is None:
                continue
            try:
                system, _graphics = anim_glb.load_resource(m, at)
                drawables = list(ss.drawable_offsets(system, kind))
            except (anim_glb.ExportError, ss.Skip, IndexError, ValueError):
                continue
            for drawable in drawables:
                try:
                    joints, _worst = anim_glb.read_skeleton(system, drawable)
                except (anim_glb.ExportError, IndexError, ValueError):
                    continue
                found.append({"hashes": [j["hash"] for j in joints], "names": [j["name"] for j in joints]})
            if progress and found and len(found) % 2000 == 0:
                progress(f"  {len(found)} skeletons by 0x{page:x}")
    return found


def print_report(report):
    print(f"{report['skeletons']} skeletons, {report['distinct_skeletons']} distinct")
    rows = list(report["streams"].items()) + [("union", report["union"])]
    for label, r in rows:
        h = r["held_by_skeletons"]
        print(f"{label}: {r['in_a_skeleton']} of {r['names']} names are joints ({r['same_spelling']} same spelling, "
              f"{r['joint_unnamed']} unnamed); all held by {h['distinct']} distinct skeletons "
              f"({h['copies']} copies), joint counts {h['joint_counts'][:6]}")


def parse_offset(text):
    try:
        return int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an offset: {text!r}") from None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("anim", help="anim.rpf")
    ap.add_argument("packs", help="packs.rpf")
    ap.add_argument("--stream", type=parse_offset, action="append", help="bone-table stream offset in anim.rpf")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    if a.json:
        for source in (a.anim, a.packs):
            if same_report_path(a.json, source):
                print(f"anim_skel_match: --json must not overwrite input {source}", file=sys.stderr)
                return 2
    try:
        names = stream_names(a.anim, a.stream or DEFAULT_STREAMS)
        skeletons = read_skeletons(a.packs, lambda t: print(t, file=sys.stderr, flush=True))
    except (OSError, ValueError) as e:
        print(f"anim_skel_match: {e}", file=sys.stderr)
        return 2
    report = match(names, skeletons)
    print_report(report)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
