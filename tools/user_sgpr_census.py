r"""Census of user-SGPR reads per shader stage, from the translator's index.json (stdlib only).

Answers, from data the pipeline already produces, where the console puts root arguments and how wide a
descriptor-table pointer is (docs/d3d12_vulkan.md §3.5, the "least verifiable part"):

  * per stage: how many shaders read each user-SGPR kind (const32, ptr32_lo, ptr64_lo/hi, v#/t#/s#);
  * per stage: histogram of the LOWEST user SGPR read as a pointer (ptr32_lo / ptr64_lo), i.e. where the
    first pointer-sized root argument sits;
  * whether pointers are 32-bit (one SGPR, high half implicit zero) or 64-bit pairs;
  * the distance between consecutive pointer SGPRs in one shader, and what the SGPR right after a 32-bit pointer
    holds. These separate the two packing models: one dword per table puts pointers at distance 1 with no high
    halves; two dwords per table put a ptr64_hi right after every table pointer (distance 2 between tables);
  * SGPRs read below a stage base given with --base (reserved / system block candidates).

With --pairs it also compares the VS and PS of one technique (D3D-318, root-parameter visibility): shaders whose
names differ only in the "VS_"/"PS_" stage token, after an optional one-character prefix and without the trailing
8-digit hash and "_Wrapped". If the console omitted the parameters a stage can't see, a VS and a PS would hold
different parameters at the same SGPR wherever one of them has a parameter the other lacks below it. With one block
in root order they hold the same ones. So it counts, within those pairs, the SGPRs both stages read and whether the
kind agrees (pointer or constant), separately for the reads above an SGPR that only one stage reads, against all
VS x PS combinations as a control. Names are used for pairing only and are never printed.

What it does NOT measure: logical root-parameter indices. It sees physical SGPRs only; mapping them to root
parameters needs a root signature, which the shaders do not carry. Pairing by technique name approximates the
pipelines the executable creates; it does not see them.

It reads only spirv/index.json (user_sgpr_reads and stage from tools/shader_meta.py): no game bytes.

Usage:
    python -I tools/user_sgpr_census.py <spirv/index.json> [--json OUT] [--pairs]
    python -I tools/user_sgpr_census.py <spirv/index.json> --expect ps=8 vs=8 cs=4 ms=16
        (exit 1 if, for a listed stage, the most common lowest-pointer SGPR is not the expected one;
         ties are broken in favor of the smaller SGPR)

An index that cannot be read, is not JSON or is not the shape tools/shader_meta.py writes, and a --json path that
cannot be written, each give one ERROR line and exit 2.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

POINTER_KINDS = ("ptr32_lo", "ptr64_lo")


def census(index, base=None):
    """index: {name: {"stage": ..., "user_sgpr_reads": [{"sgpr": n, "kind": k}, ...]}} -> summary dict.
    base: optional {stage: first root-argument SGPR}; reads below it are reported per (sgpr, kind)."""
    base = base or {}
    below = defaultdict(Counter)
    shaders = Counter()
    kinds = defaultdict(Counter)            # stage -> kind -> shaders reading it
    lowest_pointer = defaultdict(Counter)   # stage -> sgpr -> shaders
    lowest_pointer_kind = defaultdict(Counter)
    pointer_gap = defaultdict(Counter)      # stage -> distance between consecutive pointer SGPRs -> count
    after_ptr32 = defaultdict(Counter)      # stage -> what the SGPR after a ptr32_lo is read as
    for entry in index.values():
        stage = entry.get("stage", "?")
        reads = entry.get("user_sgpr_reads") or []
        shaders[stage] += 1
        for k in {r.get("kind") for r in reads}:
            kinds[stage][k] += 1
        ptrs = sorted((r["sgpr"], r["kind"]) for r in reads if r.get("kind") in POINTER_KINDS and "sgpr" in r)
        at = defaultdict(set)
        for r in reads:
            if "sgpr" in r:
                at[r["sgpr"]].add(r.get("kind"))
                if stage in base and r["sgpr"] < base[stage]:
                    below[stage][f"s{r['sgpr']}:{r.get('kind')}"] += 1
        regs = [p for p, _ in ptrs]
        for x, y in zip(regs, regs[1:]):
            pointer_gap[stage][y - x] += 1
        for p, k in ptrs:
            if k == "ptr32_lo":
                nxt = at.get(p + 1, set())
                after_ptr32[stage]["ptr64_hi" if "ptr64_hi" in nxt else "pointer" if nxt & set(POINTER_KINDS)
                                   else "const32" if "const32" in nxt else "not read" if not nxt else "other"] += 1
        if ptrs:
            sgpr, kind = ptrs[0]
            lowest_pointer[stage][sgpr] += 1
            lowest_pointer_kind[stage][kind] += 1
    out = {}
    for stage in sorted(shaders):
        lp = lowest_pointer[stage]
        out[stage] = {
            "shaders": shaders[stage],
            "shaders_reading_kind": dict(sorted(kinds[stage].items())),
            "lowest_pointer_sgpr": {f"s{k}": v for k, v in sorted(lp.items())},
            "lowest_pointer_kind": dict(sorted(lowest_pointer_kind[stage].items())),
            "most_common_lowest_pointer_sgpr": max(lp.items(), key=lambda kv: (kv[1], -kv[0]))[0] if lp else None,
            "pointer_gap": dict(sorted(pointer_gap[stage].items())),
            "after_ptr32": dict(sorted(after_ptr32[stage].items())),
        }
        if stage in base:
            out[stage]["below_base"] = dict(sorted(below[stage].items()))
    return out


def load_index(path):
    """The index as {name: entry}. ValueError with a one-line reason when the file cannot be read, is not JSON or
    is not the shape census() reads; the wording follows tools/root_sig_contract.py, which reads the same file.

    utf-8-sig: a BOM left by a Windows editor is not a JSON error.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            index = json.load(f)
    except (OSError, ValueError, RecursionError) as e:   # ValueError: not JSON, or not UTF-8
        reason = getattr(e, "strerror", None) or e
        raise ValueError(f"cannot read {path}: {reason}") from None
    if not isinstance(index, dict):
        raise ValueError(f"{path}: expected a JSON object of shader -> metadata entries")
    for name, entry in index.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {name}: each shader entry must be a JSON object")
        if not isinstance(entry.get("stage", ""), str):
            raise ValueError(f"{path}: {name}: 'stage' must be a string")
        reads = entry.get("user_sgpr_reads") or []
        if not isinstance(reads, list):
            raise ValueError(f"{path}: {name}: 'user_sgpr_reads' must be a JSON array")
        for read in reads:
            if not isinstance(read, dict) or "sgpr" not in read or "kind" not in read:
                raise ValueError(f"{path}: {name}: each read needs 'sgpr' and 'kind' fields")
            sgpr = read["sgpr"]
            if isinstance(sgpr, bool) or not isinstance(sgpr, int) or sgpr < 0:
                raise ValueError(f"{path}: {name}: each read 'sgpr' must be a nonnegative integer")
            if not isinstance(read["kind"], str):
                raise ValueError(f"{path}: {name}: each read kind must be a string")
    return index


# "<optional one-character prefix>VS_<technique>_<8 hex digits>_Wrapped" and the same with PS_: hash and suffix optional.
TECHNIQUE_RE = re.compile(r"^.?(VS|PS)_(.+?)(?:_[0-9a-f]{8})?(?:_Wrapped)?$")


def technique(name, stage):
    """The technique a VS or PS name belongs to, or None when the name doesn't carry the stage's token."""
    m = TECHNIQUE_RE.match(name)
    return m.group(2) if m and m.group(1).lower() == stage else None


def read_map(entry):
    """{sgpr: kind} for one shader; an SGPR read as more than one kind keeps the sorted first."""
    out = {}
    for r in entry.get("user_sgpr_reads") or []:
        if "sgpr" in r and r.get("kind"):
            out[r["sgpr"]] = min(out.get(r["sgpr"], r["kind"]), r["kind"])
    return out


def pair_census(index):
    """VS/PS of one technique (see the module docstring): agreement of the SGPRs both read, and the control."""
    groups = defaultdict(lambda: {"vs": [], "ps": []})
    stage_reads = {"vs": Counter(), "ps": Counter()}       # (sgpr, kind) -> shaders, for the control
    for name, entry in index.items():
        stage = entry.get("stage")
        if stage not in stage_reads:
            continue
        m = read_map(entry)
        for sk in m.items():
            stage_reads[stage][sk] += 1
        t = technique(name, stage)
        if t is not None:
            groups[t][stage].append(m)
    paired = {t: g for t, g in groups.items() if g["vs"] and g["ps"]}
    c = Counter()
    for g in paired.values():
        for v in g["vs"]:
            for p in g["ps"]:
                c["combinations"] += 1
                shared = set(v) & set(p)
                one_sided = set(v) ^ set(p)
                c["identical_read_maps"] += v == p
                if shared:
                    c["combinations_sharing_an_sgpr"] += 1
                if one_sided and shared and min(one_sided) < max(shared):
                    c["combinations_with_a_one_sided_sgpr_below_a_shared_one"] += 1
                for s in shared:
                    same = v[s] == p[s]
                    c["shared_same_kind" if same else "shared_different_kind"] += 1
                    if any(k < s for k in one_sided):
                        c["above_one_sided_same_kind" if same else "above_one_sided_different_kind"] += 1
    control = Counter()
    for (s, kv), nv in stage_reads["vs"].items():
        for (s2, kp), np_ in stage_reads["ps"].items():
            if s == s2:
                control["same_kind" if kv == kp else "different_kind"] += nv * np_
    keys = ("combinations", "combinations_sharing_an_sgpr", "identical_read_maps", "shared_same_kind",
            "shared_different_kind", "combinations_with_a_one_sided_sgpr_below_a_shared_one",
            "above_one_sided_same_kind", "above_one_sided_different_kind")
    return {
        "techniques_with_vs_and_ps": len(paired),
        "vs_in_pairs": sum(len(g["vs"]) for g in paired.values()),
        "ps_in_pairs": sum(len(g["ps"]) for g in paired.values()),
        **{k: c[k] for k in keys},
        "control_all_vs_ps": {"combinations": sum(1 for e in index.values() if e.get("stage") == "vs")
                              * sum(1 for e in index.values() if e.get("stage") == "ps"),
                              "shared_same_kind": control["same_kind"],
                              "shared_different_kind": control["different_kind"]},
    }


def stage_sgpr(text):
    stage, _, value = text.partition("=")
    message = f"expected STAGE=SGPR with a nonnegative SGPR: {text!r}"
    try:
        sgpr = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not stage or sgpr < 0:
        raise argparse.ArgumentTypeError(message)
    return stage, sgpr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("index", help="spirv/index.json written by tools/rdna2spirv.py")
    ap.add_argument("--json", help="also write the summary here")
    ap.add_argument("--expect", nargs="*", type=stage_sgpr, default=[], metavar="STAGE=SGPR",
                    help="fail if the most common lowest-pointer SGPR of STAGE is not SGPR")
    ap.add_argument("--base", nargs="*", type=stage_sgpr, default=[], metavar="STAGE=SGPR",
                    help="report the reads below this first root-argument SGPR per stage")
    ap.add_argument("--pairs", action="store_true",
                    help="compare the VS and PS of one technique (root-parameter visibility, D3D-318)")
    a = ap.parse_args(argv)
    if a.json and same_report_path(a.index, a.json):
        ap.error("--json must not overwrite the input index")
    try:
        index = load_index(a.index)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    summary = census(index, dict(a.base))
    for stage, s in summary.items():
        top = s["most_common_lowest_pointer_sgpr"]
        print(f"{stage}: {s['shaders']} shaders; lowest pointer SGPR "
              f"{('s%d' % top) if top is not None else '-'} {s['lowest_pointer_sgpr']}; kinds {s['lowest_pointer_kind']}; "
              f"pointer gaps {s['pointer_gap']}; after a ptr32 {s['after_ptr32']}"
              + (f"; below base {s['below_base']}" if "below_base" in s else ""))
    report = summary
    if a.pairs:
        pc = pair_census(index)
        ctl = pc["control_all_vs_ps"]
        def share(same, diff):
            return f"{diff / (same + diff):.1%}" if same + diff else "-"
        print(f"VS/PS pairs: {pc['techniques_with_vs_and_ps']} techniques with both stages ({pc['vs_in_pairs']} VS, "
              f"{pc['ps_in_pairs']} PS), {pc['combinations']} combinations, {pc['combinations_sharing_an_sgpr']} "
              f"sharing an SGPR, {pc['identical_read_maps']} with identical read maps")
        print(f"  SGPRs both read: same kind {pc['shared_same_kind']}, different kind {pc['shared_different_kind']} "
              f"({share(pc['shared_same_kind'], pc['shared_different_kind'])} different)")
        print(f"  above an SGPR only one stage reads ({pc['combinations_with_a_one_sided_sgpr_below_a_shared_one']} "
              f"combinations): same kind {pc['above_one_sided_same_kind']}, different kind "
              f"{pc['above_one_sided_different_kind']}")
        print(f"  control, all {ctl['combinations']} VS x PS combinations: same kind {ctl['shared_same_kind']}, "
              f"different kind {ctl['shared_different_kind']} "
              f"({share(ctl['shared_same_kind'], ctl['shared_different_kind'])} different)")
        report = {**summary, "vs_ps_pairs": pc}
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=1, sort_keys=True)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    bad = []
    for stage, sgpr in a.expect:
        got = summary.get(stage, {}).get("most_common_lowest_pointer_sgpr")
        if got != sgpr:
            bad.append(f"{stage}: expected s{sgpr}, got {'s%d' % got if got is not None else 'none'}")
    for b in bad:
        print("MISMATCH", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
