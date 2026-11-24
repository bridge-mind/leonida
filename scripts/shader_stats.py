"""How efficiently the driver compiles the translated shaders, against the game's own RDNA code.

    python3 -I scripts/shader_stats.py <stats.csv> <disasm dir> [--details]

<stats.csv> comes from vk_pipeline_check <spirv dir> --stats <stats.csv> (the driver's statistics for each translated
shader); <disasm dir> holds the original <name>.s listings the translator read. For every shader in both it compares
the driver's instruction count with the original one (code growth) and its VGPRs with the highest VGPR the original
code uses, and it flags scratch memory and spills: a register file that didn't fit in registers and went to memory,
which is the expensive case (relative VGPR indexing, v_movrels/v_movreld, forces it). It prints a summary per stage and,
with --details, one line per shader.
Growth is unavailable when the original listing has no parsed instructions; its driver counters are still shown.

Metrics are read only where the driver emits these exact RADV/ACO statistic names with a numeric value: Instructions,
VGPRs, SGPRs, Scratch size, Spilled VGPRs, Spilled SGPRs, Subgroups per SIMD. No equivalence is claimed for other
drivers' statistics, even similarly named ones. A metric that isn't reported is shown as - (n/a in the summary), never
as 0; a summary clause over fewer shaders than the stage has says so with (n of N), and the original VGPRs it compares
with are those of the same shaders. With only one of the two spill statistics reported, spills shows it when positive,
with >= to mark a lower bound, and - when it is 0. When a driver omits a comparison metric, --details also lists any additional driver counters under that
shader with their original names and values. A note lists those names in summary mode.
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import shader_ir  # noqa: E402

# (row key, RADV/ACO statistic name) of the statistics read as numbers; spills combine SPILLS.
METRICS = [("aco", "Instructions"), ("vgprs", "VGPRs"), ("sgprs", "SGPRs"), ("scratch", "Scratch size"),
           ("waves", "Subgroups per SIMD")]
SPILLS = ("Spilled VGPRs", "Spilled SGPRs")
KNOWN = {stat for _, stat in METRICS} | set(SPILLS)
# --details columns after shader and stage: (row key, header, width, format spec).
COLUMNS = [("orig", "orig", 5, ""), ("aco", "aco", 5, ""), ("growth", "growth", 6, ".2f"), ("v_orig", "v_orig", 6, ""),
           ("vgprs", "VGPRs", 5, ""), ("sgprs", "SGPRs", 5, ""), ("scratch", "scratch", 7, ""),
           ("spill_detail", "spills", 6, ""), ("waves", "waves", 5, "")]


def original(path: str) -> tuple[int, int]:
    """(instruction count, VGPRs used) of an original RDNA listing."""
    insns = shader_ir.parse_file(path)["insns"]
    top = 0
    for insn in insns:
        for operand in insn["args"]:
            kind = operand.get("k")
            if kind == "vgpr":
                top = max(top, operand["r"] + operand.get("n", 1))
            elif kind == "vlist":
                top = max(top, max(operand["r"]) + 1)
    return len(insns), top


REQUIRED = ("name", "stage", "statistic", "value")


def load_stats(path):
    """Read counters keyed by (shader, stage) from vk_pipeline_check --stats."""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, strict=True)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ValueError(f"line 1: missing required header column(s): {', '.join(missing)}")
        duplicates = [c for c in REQUIRED if header.count(c) > 1]
        if duplicates:
            raise ValueError(f"line 1: duplicated required header column(s): {', '.join(duplicates)}")
        for row in reader:
            line = reader.line_num
            if None in row:
                raise ValueError(f"line {line}: row has more fields than the header")
            absent = [c for c in REQUIRED if row[c] is None]
            if absent:
                raise ValueError(f"line {line}: row missing required column(s): {', '.join(absent)}")
            out[row["name"], row["stage"]][row["statistic"]] = row["value"]
    return out


def number(raw):
    """The exact int, or else the finite float, a raw value holds; None if it is missing or not such a number."""
    if raw is None:
        return None
    for parse in (int, float):
        try:
            value = parse(raw)
            if not math.isfinite(value):
                return None
            return int(value) if isinstance(value, float) and value.is_integer() else value
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def shader_row(name, stage, st, src):
    n_orig, v_orig = original(src)
    r = {"name": name, "stage": stage, "orig": n_orig, "v_orig": v_orig}
    for key, stat in METRICS:
        r[key] = number(st.get(stat))
    r["growth"] = r["aco"] / n_orig if n_orig and r["aco"] is not None else None

    spill_counts = [number(st.get(stat)) for stat in SPILLS]
    known_spills = [v for v in spill_counts if v is not None]
    complete_spills = len(known_spills) == len(SPILLS)
    total = sum(known_spills)
    r["spills"] = total if complete_spills or total > 0 else None
    r["spill_detail"] = f">={total}" if total > 0 and not complete_spills else r["spills"]

    complete = complete_spills and all(r[key] is not None for key, _ in METRICS)
    r["raw"] = {} if complete else {stat: value for stat, value in st.items() if stat not in KNOWN}
    return r


def cell(value, spec=""):
    """A details cell: the value, or - when it isn't reported."""
    return "-" if value is None else format(value, spec)


def reported(rs, key):
    return [r for r in rs if r[key] is not None]


def coverage(n, total):
    return f" ({n} of {total})" if n < total else ""


def count_clause(rs, key, label):
    """How many of the shaders reporting key have it above 0."""
    known = reported(rs, key)
    if not known:
        return f"{label} n/a"
    return f"{label} in {sum(r[key] > 0 for r in known)}{coverage(len(known), len(rs))}"


def median_clause(rs, key, label):
    known = reported(rs, key)
    if not known:
        return f"{label} n/a"
    return f"{label} median {statistics.median(r[key] for r in known):g}{coverage(len(known), len(rs))}"


USAGE = "usage: shader_stats.py <stats.csv> <disasm dir> [--details]"


def main():
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    # A misspelt option (--detail) used to be dropped, and the summary printed as if it had been asked for.
    unknown = [a for a in argv if a.startswith("--") and a != "--details"]
    if unknown:
        print(f"unknown option {unknown[0]}\n{USAGE}", file=sys.stderr)
        return 2
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    path = args[0]
    try:
        stats = load_stats(path)
    except (OSError, ValueError, csv.Error) as e:
        sys.exit(f"{path}: {e}")
    disasm = args[1]
    rows = []
    for (name, stage), st in sorted(stats.items()):
        src = os.path.join(disasm, name + ".s")
        if os.path.isfile(src):
            try:
                rows.append(shader_row(name, stage, st, src))
            except OSError as e:
                sys.exit(f"{src}: {e.strerror or e}")
    if not rows:
        sys.exit("no shader is in both the statistics and the disasm folder")
    if "--details" in sys.argv:
        print(f"{'shader':26} {'stage':5} " + " ".join(f"{head:>{width}}" for _, head, width, _ in COLUMNS))
        for r in rows:
            print(f"{r['name']:26} {r['stage']:5} "
                  + " ".join(f"{cell(r[key], spec):>{width}}" for key, _, width, spec in COLUMNS))
            for stat, raw in sorted(r["raw"].items()):
                print(f"    {stat} = {raw}")
        print()
    else:
        unread = sorted({stat for r in rows for stat in r["raw"]})
        if unread:
            print("additional driver statistics (see --details): " + "; ".join(unread), file=sys.stderr)
    for stage in sorted({r["stage"] for r in rows}):
        rs = [r for r in rows if r["stage"] == stage]
        g = sorted(r["growth"] for r in reported(rs, "growth"))
        growth = "code growth n/a"
        if g:
            growth = (f"code growth median {statistics.median(g):.2f}x "
                      f"(p90 {g[int(0.9 * (len(g) - 1))]:.2f}x, max {g[-1]:.2f}x){coverage(len(g), len(rs))}")
        v = reported(rs, "vgprs")
        vgprs = "VGPRs n/a"
        if v:
            vgprs = (f"VGPRs median {statistics.median(r['vgprs'] for r in v):g} "
                     f"vs original {statistics.median(r['v_orig'] for r in v):g}{coverage(len(v), len(rs))}")
        print(f"{stage}: {len(rs)} shaders, {growth}; {vgprs}; "
              f"{count_clause(rs, 'scratch', 'scratch')}, {count_clause(rs, 'spills', 'spills')}; "
              f"{median_clause(rs, 'waves', 'waves/SIMD')}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
