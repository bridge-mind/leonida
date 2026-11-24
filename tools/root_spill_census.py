r"""Census of root-signature spill evidence in the translated corpus (stdlib only).

Answers D3D-315's measuring half (docs/d3d12_vulkan.md section 3.5, "Root signature larger than 32 dwords"): does any
shader in the corpus read a root-signature spill pointer, and how close does each stage come to the 32 user SGPRs?

On the console's register-based ABI the shader compiler decides the spill (`__XB_GetRootSignatureSpillPointer`), so
a shader whose root signature spilled must read a spill pointer from its user data. For compute that pointer was
measured at s[0:1], below the stage's first root-argument SGPR (section 7). This tool counts, per stage:

  * spill-pointer candidates: user-SGPR reads used as an address (ptr32_lo, ptr64_lo, ptr64_hi) below the stage
    base (the census bases ps=8 vs=8 gs=8 cs=4 ms=16; --base overrides the stages it names and keeps the others).
    A shader with none of them has no pointer a spill could sit behind in that block;
  * the highest user SGPR each shader reads, and what that read is (a constant or an address);
  * where index.json records it (user_sgprs, the RSRC2 USER_SGPR count; compute today): the largest count and the
    root-argument dwords that leaves above the base;
  * with --carve <dxbc/index.json>: how many shader containers carry an RTS0 part (a root signature) at all.

What it does NOT measure: the root signatures themselves. The game builds them at run time; without RTS0 parts in
the containers they can only be dumped from the executable. A spill hidden behind a pointer at or above the base
(a root parameter the compiler happens to address like a table) is indistinguishable here from a table.

It reads only index files (user_sgpr_reads, user_sgprs and stage from tools/shader_meta.py; the parts table from
tools/shader_carve.py): no game bytes, and it prints counts only.

Usage:
    python -I tools/root_spill_census.py <spirv/index.json> [--carve <dxbc/index.json>] [--json OUT]
    python -I tools/root_spill_census.py <spirv/index.json> --expect-no-spill
        (exit 1 if any stage has a spill-pointer candidate)
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

POINTER_KINDS = ("ptr32_lo", "ptr64_lo", "ptr64_hi")
DEFAULT_BASE = {"ps": 8, "vs": 8, "gs": 8, "cs": 4, "ms": 16}
USER_SGPRS = 32


def census(index, base=None, carve=None):
    """index: spirv/index.json as a dict; base: {stage: first root-argument SGPR}; carve: dxbc/index.json (a list of
    entries with a "parts" table) or None. Returns a summary dict of counts."""
    base = DEFAULT_BASE if base is None else base
    shaders = Counter()
    spill = defaultdict(Counter)        # stage -> "sN:kind" -> shaders
    spill_shaders = Counter()           # stage -> shaders with at least one candidate
    highest = defaultdict(Counter)      # stage -> highest SGPR read -> shaders
    highest_kind = defaultdict(Counter)  # stage -> kind of that read -> shaders
    user_sgprs = defaultdict(Counter)   # stage -> RSRC2 USER_SGPR count -> shaders (where recorded)
    for entry in index.values():
        if not isinstance(entry, dict):
            continue
        stage = entry.get("stage", "?")
        shaders[stage] += 1
        reads = [r for r in entry.get("user_sgpr_reads") or [] if isinstance(r.get("sgpr"), int)]
        b = base.get(stage)
        found = set()
        if b is not None:
            for r in reads:
                if r["sgpr"] < b and r.get("kind") in POINTER_KINDS:
                    found.add(f"s{r['sgpr']}:{r['kind']}")
        for key in found:
            spill[stage][key] += 1
        if found:
            spill_shaders[stage] += 1
        if reads:
            top = max(r["sgpr"] for r in reads)
            highest[stage][top] += 1
            kinds = {r.get("kind") for r in reads if r["sgpr"] == top}
            highest_kind[stage]["pointer" if kinds & set(POINTER_KINDS) else "/".join(sorted(map(str, kinds)))] += 1
        count = entry.get("user_sgprs")
        if isinstance(count, int):
            user_sgprs[stage][count] += 1
    out = {"stages": {}}
    for stage in sorted(shaders):
        s = {
            "shaders": shaders[stage],
            "base": base.get(stage),
            "spill_pointer_candidates": dict(sorted(spill[stage].items())),
            "shaders_with_spill_pointer_candidate": spill_shaders[stage],
            "highest_sgpr_read": {f"s{k}": v for k, v in sorted(highest[stage].items())},
            "highest_read_kind": dict(sorted(highest_kind[stage].items())),
            "max_highest_sgpr_read": max(highest[stage]) if highest[stage] else None,
        }
        if user_sgprs[stage]:
            top = max(user_sgprs[stage])
            s["user_sgprs"] = {str(k): v for k, v in sorted(user_sgprs[stage].items())}
            s["max_user_sgprs"] = top
            s["shaders_using_all_user_sgprs"] = user_sgprs[stage][USER_SGPRS]
            if base.get(stage) is not None:
                s["max_root_dwords"] = max(top - base[stage], 0)
                s["root_dwords_available"] = USER_SGPRS - base[stage]
        out["stages"][stage] = s
    if carve is not None:
        entries = carve.values() if isinstance(carve, dict) else carve
        containers = [e for e in entries if isinstance(e, dict)]
        out["containers"] = len(containers)
        out["containers_with_rts0"] = sum(1 for e in containers if "RTS0" in (e.get("parts") or {}))
    return out


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
    ap.add_argument("--carve", help="dxbc/index.json written by tools/shader_carve.py (counts RTS0 parts)")
    ap.add_argument("--base", nargs="*", type=stage_sgpr, default=[], metavar="STAGE=SGPR",
                    help="first root-argument SGPR for the stages named; the others keep the defaults "
                         "ps=8 vs=8 gs=8 cs=4 ms=16")
    ap.add_argument("--json", help="also write the summary here")
    ap.add_argument("--expect-no-spill", action="store_true",
                    help="exit 1 if any shader reads a spill-pointer candidate")
    a = ap.parse_args(argv)
    if a.json:
        if os.path.isdir(a.json):
            ap.error(f"--json {a.json} is a folder, not a file")
        for other in (a.index, a.carve):
            if other and same_report_path(other, a.json):
                ap.error("--json must not overwrite an input index")
    if not os.path.exists(a.index):
        print(f"ERROR: cannot read {a.index}: no such file", file=sys.stderr)
        return 2
    if a.carve and not os.path.exists(a.carve):
        print(f"ERROR: cannot read {a.carve}: no such file", file=sys.stderr)
        return 2
    with open(a.index, encoding="utf-8-sig") as f:  # an index saved with a UTF-8 BOM is still JSON
        index = json.load(f)
    if not isinstance(index, dict):
        ap.error(f"{a.index} is not a translator index (expected a JSON object)")
    carve = None
    if a.carve:
        with open(a.carve, encoding="utf-8-sig") as f:  # same for the carve index
            carve = json.load(f)
    # --base overrides the stages it names and keeps the defaults for the rest, so a partial override never drops
    # a stage from the check.
    summary = census(index, {**DEFAULT_BASE, **dict(a.base)}, carve)
    if "containers" in summary:
        print(f"containers: {summary['containers']}, with an RTS0 part: {summary['containers_with_rts0']}")
    for stage, s in summary["stages"].items():
        line = (f"{stage}: {s['shaders']} shaders; base s{s['base']}; " if s["base"] is not None
                else f"{stage}: {s['shaders']} shaders; no base; ")
        line += (f"spill-pointer candidates {s['shaders_with_spill_pointer_candidate']} "
                 f"{s['spill_pointer_candidates']}; highest read {s['highest_sgpr_read']} "
                 f"as {s['highest_read_kind']}")
        if "max_user_sgprs" in s:
            line += f"; user_sgprs up to {s['max_user_sgprs']}"
            if "max_root_dwords" in s:
                line += f" ({s['max_root_dwords']} of {s['root_dwords_available']} root dwords)"
        print(line)
    if a.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            print(f"error: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    spilled = {st: s["shaders_with_spill_pointer_candidate"] for st, s in summary["stages"].items()
               if s["shaders_with_spill_pointer_candidate"]}
    if a.expect_no_spill and spilled:
        print(f"SPILL POINTER CANDIDATES {spilled}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
