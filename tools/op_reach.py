"""How many shaders of a disassembled corpus use each instruction, all encodings of it counted as one.

    python -I tools/op_reach.py <work>/shaders/dxbc/disasm [OP ...] [--index <work>/shaders/dxbc/index.json]
                                [--fssm <work>/shaders/dxbc/fssm] [--json OUT]

Reads every `.s` listing in the folder (shader_disasm.py output: an instruction line starts with a tab). An
instruction's base name drops the encoding suffix LLVM prints (`_e32`, `_e64`, `_sdwa`, `_dpp`, the same ones
rdna2spirv's `base_op()` strips), so `v_cndmask_b32_e32` and `v_cndmask_b32_e64` count as `v_cndmask_b32`, and a
shader using both counts once. tools/shader_profile.py counts the printed names instead.

An OP with `*` or `?` is a pattern over the printed names instead (`fnmatch`, case-sensitive): `'*_dpp'` counts
the shaders with any DPP instruction, `'v_*_f16_sdwa'` the 16-bit SDWA forms.

Only the program counts: lines after a listing's last `s_endpgm` are data words LLVM decoded anyway (some
listings run past the program; on 1.0.2.1 every `s_cmov_b32` in a listing is there), so they are left out and
their total is reported separately as `after_last_endpgm`. A listing with no `s_endpgm` counts whole.

With OPs, prints those only, an OP no listing uses with 0 (that is the answer to "does the game reach this
rule"). Without, prints every base name, most-used first. `--index` (the carve index) adds the count per stage;
`--fssm` (the folder of sidecar `.fssm` headers) the count per compiled wave size, from the header flag
`fssm_header.wave_size()` reads (`u32[0xD4]` bit 5), and with both, per stage and wave size.

Standard library only. Prints and writes counts and instruction names only, nothing from the listings.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fssm_header  # noqa: E402

INSTRUCTION = re.compile(r"^\t([a-z][a-z0-9_]*)", re.M)
ENCODING = re.compile(r"_(e32|e64|sdwa|dpp)$")
HEADER_BYTES = 0x100            # wave_size() reads u32[0xD4]


def base_name(op):
    """The instruction without the encoding suffix LLVM prints."""
    return ENCODING.sub("", op)


def is_pattern(op):
    return any(c in op for c in "*?[")


def listing_ops(text):
    """(printed name -> instructions up to the last s_endpgm, instructions after it) for one listing."""
    ops = INSTRUCTION.findall(text)
    end = max((i for i, op in enumerate(ops) if op == "s_endpgm"), default=len(ops) - 1)
    return Counter(ops[:end + 1]), len(ops) - end - 1


def stages_of(index_path):
    """Listing stem -> stage from the carve index."""
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, list) or not all(isinstance(e, dict) for e in index):
        raise ValueError("index.json is not a list of shader entries")
    return {os.path.splitext(e["file"])[0]: e.get("stage") for e in index if isinstance(e.get("file"), str)}


def waves_of(fssm_dir, stems):
    """Listing stem -> 32, 64, or None (no sidecar, or not an FSSM header)."""
    waves = {}
    for stem in stems:
        try:
            with open(os.path.join(fssm_dir, stem + ".fssm"), "rb") as f:
                waves[stem] = fssm_header.wave_size(f.read(HEADER_BYTES))
        except FileNotFoundError:
            waves[stem] = None
    return waves


def group_keys(stage, wave, has_waves):
    """(stage key, wave key, stage-and-wave key) a listing counts under; None where that split is not asked for."""
    w = (f"w{wave}" if wave else "w?") if has_waves else None
    return stage, w, (f"{stage}/{w}" if stage and w else None)


def census(disasm_dir, stages=None, fssm_dir=None):
    """Every .s file's printed-name counts, with its stage (index) and wave size (FSSM header) when given."""
    names = sorted(n for n in os.listdir(disasm_dir) if n.endswith(".s"))
    stems = [n[:-2] for n in names]
    waves = waves_of(fssm_dir, stems) if fssm_dir is not None else None
    listings = []
    after = 0
    by_name = defaultdict(list)     # printed name -> [(listing, count)]
    for i, (name, stem) in enumerate(zip(names, stems)):
        with open(os.path.join(disasm_dir, name), encoding="utf-8", errors="replace") as f:
            ops, tail = listing_ops(f.read())
        after += tail
        stage = (stages.get(stem) or "unindexed") if stages is not None else None
        wave = waves[stem] if waves is not None else None
        listings.append((stage, wave))
        for op, count in ops.items():
            by_name[op].append((i, count))
    groups = [Counter(), Counter(), Counter()]
    for stage, wave in listings:
        for g, key in zip(groups, group_keys(stage, wave, waves is not None)):
            if key:
                g[key] += 1
    shaders, instructions = Counter(), Counter()
    for op, uses in by_name.items():
        base = base_name(op)
        instructions[base] += sum(c for _, c in uses)
    for base in instructions:
        shaders[base] = len(_users(by_name, [op for op in by_name if base_name(op) == base]))
    return {"listings": len(names), "after_last_endpgm": after, "has_stages": stages is not None,
            "has_waves": waves is not None, "listing_groups": listings, "by_name": by_name,
            "listings_by_stage": dict(sorted(groups[0].items())), "listings_by_wave": dict(sorted(groups[1].items())),
            "listings_by_stage_wave": dict(sorted(groups[2].items())),
            "shaders": shaders, "instructions": instructions}


def _users(by_name, names):
    users = set()
    for op in names:
        users.update(i for i, _ in by_name[op])
    return users


def report(result, ops=None):
    """Rows for the selected OPs (base names or patterns), or every base name: shaders, instructions, and the
    shader count per stage / wave size / both when the census has them."""
    by_name = result["by_name"]
    if ops:
        chosen = [op if is_pattern(op) else base_name(op) for op in ops]
    else:
        chosen = sorted(result["shaders"], key=lambda o: (-result["shaders"][o], o))
    rows = []
    for op in chosen:
        if is_pattern(op):
            names = [n for n in by_name if fnmatch.fnmatchcase(n, op)]
        else:
            names = [n for n in by_name if base_name(n) == op]
        users = _users(by_name, names)
        row = {"op": op, "shaders": len(users), "instructions": sum(c for n in names for _, c in by_name[n])}
        groups = [Counter(), Counter(), Counter()]
        for i in users:
            stage, wave = result["listing_groups"][i]
            for g, key in zip(groups, group_keys(stage, wave, result["has_waves"])):
                if key:
                    g[key] += 1
        row["by_stage"] = dict(sorted(groups[0].items()))
        if result["has_waves"]:
            row["by_wave"] = dict(sorted(groups[1].items()))
            if result["has_stages"]:
                row["by_stage_wave"] = dict(sorted(groups[2].items()))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="folder of .s listings (<work>/shaders/dxbc/disasm)")
    ap.add_argument("ops", nargs="*", help="base instruction names or name patterns to report (default: all names)")
    ap.add_argument("--index", help="carve index.json: also count per stage")
    ap.add_argument("--fssm", help="folder of the sidecar .fssm headers: also count per wave size")
    ap.add_argument("--json", help="also write the rows as JSON")
    a = ap.parse_args(argv)
    for path in (a.disasm, a.fssm):
        if path is not None and not os.path.isdir(path):
            print(f"ERROR: {path} is not a folder", file=sys.stderr)
            return 2
    try:
        stages = stages_of(a.index) if a.index else None
        result = census(a.disasm, stages, a.fssm)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not result["listings"]:
        print(f"ERROR: no .s listing in {a.disasm}", file=sys.stderr)
        return 1
    rows = report(result, a.ops)
    groups = {k: result[k] for k in ("listings_by_stage", "listings_by_wave", "listings_by_stage_wave") if result[k]}
    print(f"{result['listings']} listings" + "".join(f" {v}" for v in groups.values()) +
          f", {result['after_last_endpgm']} decoded instructions after the last s_endpgm left out")
    for row in rows:
        split = row.get("by_stage_wave") or row.get("by_stage") or row.get("by_wave")
        detail = f"  {split}" if (stages is not None or a.fssm) and split else ""
        print(f"{row['op']:32s} {row['shaders']:7d} shaders {row['instructions']:9d} instructions{detail}")
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"listings": result["listings"], **groups,
                           "after_last_endpgm": result["after_last_endpgm"], "rows": rows}, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
