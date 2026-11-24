r"""Census of the pointers whose high dword a shader forces to zero (stdlib only, no SPIR-V written).

docs/d3d12_vulkan.md section 3.1 and the "Descriptor-table high-dword truncation" item say that shaders truncate
pointers to 32 bits; this counts them. It runs `tools/rdna2spirv.py`'s `translate()` on every listing only as far as
the `tools/shader_meta.py` call (the same instructions, CFG, user-SGPR limit and system SGPRs a full run hands it),
stops there, and re-runs `shader_meta.analyse_instructions` with every pointer dereference it sees recorded. Nothing
is emitted or assembled, so a corpus pass needs no disk beyond the optional --json.

A pointer dereference is a `s_load_dword*` / `s_store_dword*` (SMEM base) or a `global_*` with an SGPR base pair:
the sites where `shader_meta._record_ptr_pair` looks at a base pair. For each site it records:

  * the access: `table` (s_load_dwordx4/x8/x16, a V#/T#/S# fetched from a descriptor table, which is what
    shader_meta calls a table load), `scalar` (s_load_dword/x2, s_store_dword*: constants or a pointer read through a
    pointer), `global` (global_* load/store/atomic with an SGPR base);
  * the low half: a user SGPR as launched (`user`), a user SGPR plus an offset or index (`user_off`), or anything else;
  * the high half: a user SGPR (a 64-bit pointer), the constant 0 (truncated), another constant, or unknown.

A *pointer* below is one user SGPR that roots at least one dereference (the low half of the base pair). It is
*truncated* when one of its sites has the high half 0: exactly shader_meta's `truncates_high` / `ptr32_lo` rule, so
the per-shader flag is checked against `analyse_instructions` and every mismatch is counted. A shader truncates
"all" its pointers when it has at least one truncated pointer and no pointer with a site whose high half is
something else.

What it cannot see (the classifier's limits, printed with the table):

  * a pointer that is not rooted in a user SGPR (loaded from memory, built from s_getpc_b64 or computed) and then
    truncated: counted separately as `lo not a user SGPR`, never as a truncated pointer;
  * a high half that is 0 on some paths only (shader_meta keeps a tag only when every path agrees): `hi unknown`;
  * base addresses inside a V# (buffer_*/tbuffer_*) or T#, global_* addresses held in VGPRs (saddr `off`), and
    flat_*/scratch_* addresses (shader_meta does not track them): the shaders using each are counted so the size
    of that gap is visible;
  * which root parameter feeds the SGPR: no game shader carries its root signature (section 3.5), so the finest
    role is the physical user SGPR, reported per stage.

Usage:
    python -I tools/ptr_truncation_census.py <disasm dir> [--jobs N] [--limit N] [--json OUT] [--index-out OUT]

<disasm dir> is `<work>/shaders/disasm` (the carve's index.json one level up gives each stage). --json writes the
summary; --index-out writes {number: {"stage", "user_sgpr_reads"}} with shaders numbered in listing order, not
named, which `tools/user_sgpr_census.py` reads. Neither prints or writes a shader name, listing or byte. --limit
stops after that many listings and must be at least 1; a smaller value is a usage error.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_workers  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_meta  # noqa: E402
from report_paths import same_report_path  # noqa: E402

STAGES = ("vs", "ps", "cs", "gs", "hs", "ds", "ms", "as")
TABLE_LOADS = ("s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16")


class _Reached(Exception):
    """Raised from the patched shader_meta.collect: translate() got as far as the metadata call."""

    def __init__(self, blocks, succ, user_sgpr_limit, system_sgprs):
        super().__init__("metadata reached")
        self.args_ = (blocks, succ, user_sgpr_limit, system_sgprs)


def _capture(_path, _name, blocks, succ, user_sgpr_limit=32, system_sgprs=()):
    raise _Reached(blocks, succ, user_sgpr_limit, system_sgprs)


def access_of(op):
    op = shader_meta.base_op(op)
    if op.startswith("global_"):
        return "global"
    return "table" if op in TABLE_LOADS else "scalar"


def high_of(hi_t):
    if shader_meta._user_of(hi_t) is not None:
        return "user"
    if hi_t is not None and hi_t[0] == "const":
        return "zero" if hi_t[1] == 0 else "const"
    return "unknown"


def pointer_sites(blocks, succ, user_sgpr_limit=32, system_sgprs=()):
    """Run analyse_instructions with every base pair it looks at recorded.

    Returns (result of analyse_instructions, sites); a site is (sgpr or None, low tag kind, high, access),
    deduplicated by instruction address (node splitting clones blocks)."""
    current = {}
    seen = {}
    step, record = shader_meta._step_state, shader_meta._record_ptr_pair

    def step_state(sym, scale, ptr_stride, vscale, ins, events=None):
        current["ins"] = ins
        return step(sym, scale, ptr_stride, vscale, ins, events)

    def record_ptr_pair(events, sym, slots, is_desc_width=False):
        lo_t, hi_t = shader_meta._slot_tag(sym, slots[0]), shader_meta._slot_tag(sym, slots[1])
        ins = current["ins"]
        key = (ins.get("pc", id(ins)), ins["op"])
        lo = shader_meta._user_of(lo_t)
        site = (lo, lo_t[0] if lo is not None else "other", high_of(hi_t), access_of(ins["op"]))
        seen.setdefault(key, set()).add(site)
        return record(events, sym, slots, is_desc_width)

    shader_meta._step_state, shader_meta._record_ptr_pair = step_state, record_ptr_pair
    try:
        result = shader_meta.analyse_instructions(blocks, succ, user_sgpr_limit=user_sgpr_limit,
                                                  system_sgprs=system_sgprs)
    finally:
        shader_meta._step_state, shader_meta._record_ptr_pair = step, record
    sites = sorted({s for group in seen.values() for s in group}, key=repr)
    return result, sites


def global_without_sgpr_base(ins):
    """A global_* instruction whose address is a VGPR pair (saddr `off`): shader_meta never looks at it."""
    op = shader_meta.base_op(ins["op"])
    if not op.startswith("global_"):
        return False
    a = ins["args"]
    k = 3 if op.startswith("global_atomic_") and "glc" in ins.get("mods", ()) else 2
    return not (len(a) > k and shader_meta._sgpr_slots(a[k], 2) is not None)


def scan_one(path):
    """Per-shader record (no name): stage, whether translate() reached the metadata call, and the sites."""
    real = shader_meta.collect
    shader_meta.collect = _capture
    rdna2spirv._shader_meta = shader_meta
    reached = None
    try:
        rdna2spirv.translate(path)
        error = "translate returned without reaching shader_meta.collect"
    except _Reached as r:
        reached, error = r.args_, None
    except rdna2spirv.Unsupported as e:
        error = f"Unsupported: {e}"
    except (IndexError, KeyError, ValueError) as e:
        error = f"translator error {type(e).__name__}"
    finally:
        shader_meta.collect = real
    name = os.path.splitext(os.path.basename(path))[0]
    stage = rdna2spirv.stage_of(name, path)
    out = {"stage": stage, "error": error}
    if reached is None:
        return out
    blocks, succ, limit, system = reached
    stock = shader_meta.analyse_instructions(blocks, succ, user_sgpr_limit=limit, system_sgprs=system)
    result, sites = pointer_sites(blocks, succ, user_sgpr_limit=limit, system_sgprs=system)
    ops = {shader_meta.base_op(i["op"]) for b in blocks for i in b}
    out.update({
        "sites": [list(s) for s in sites],
        "truncates_high": stock["truncates_high"],
        "instrumented_truncates_high": result["truncates_high"],
        "user_sgpr_reads": stock["user_sgpr_reads"],
        "flat": any(o.startswith("flat_") for o in ops),
        "scratch": any(o.startswith("scratch_") for o in ops),
        "vsharp_memory": any(o.startswith(("buffer_", "tbuffer_", "s_buffer_")) for o in ops),
        "getpc": "s_getpc_b64" in ops,
        "global_vgpr_address": sum(1 for b in blocks for i in b if global_without_sgpr_base(i)),
    })
    return out


def scan_many(paths):
    return [scan_one(p) for p in paths]


def summarise(records):
    """records: scan_one() results -> summary dict (counts only)."""
    shaders, reached, errors = Counter(), Counter(), Counter()
    trunc_shaders, mode, per_shader = Counter(), defaultdict(Counter), defaultdict(Counter)
    trunc_access, trunc_low, trunc_sgpr = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    untrunc_high, both_ways, kind_shown = defaultdict(Counter), Counter(), defaultdict(Counter)
    blind = defaultdict(Counter)
    mismatch = 0
    for r in records:
        st = r["stage"]
        shaders[st] += 1
        if r.get("error"):
            errors[r["error"]] += 1
            continue
        reached[st] += 1
        trunc, other = defaultdict(set), defaultdict(set)
        untracked = set()
        for sgpr, low, high, access in r["sites"]:
            if sgpr is None:
                untracked.add(f"lo not a user SGPR, hi {high} ({access})")
                continue
            (trunc if high == "zero" else other)[sgpr].add((low, high, access))
        for what in untracked:
            blind[st]["shaders with " + what] += 1
        flag = bool(trunc)
        if flag != r["truncates_high"] or flag != r["instrumented_truncates_high"]:
            mismatch += 1
        if r.get("flat"):
            blind[st]["shaders using flat_*"] += 1
        if r.get("scratch"):
            blind[st]["shaders using scratch_*"] += 1
        if r.get("global_vgpr_address"):
            blind[st]["shaders with a VGPR-addressed global_*"] += 1
            blind[st]["VGPR-addressed global_* instructions"] += r["global_vgpr_address"]
        if r.get("getpc"):
            blind[st]["shaders using s_getpc_b64"] += 1
        if r.get("vsharp_memory"):
            blind[st]["shaders using a V# (buffer_*, tbuffer_*, s_buffer_*)"] += 1
        pointers = set(trunc) | set(other)
        if not pointers:
            mode[st]["no pointer"] += 1
            continue
        if not trunc:
            mode[st]["none truncated"] += 1
        elif other:
            mode[st]["some truncated"] += 1
        else:
            mode[st]["all truncated"] += 1
        if not trunc:
            for sites in other.values():
                for _, high, access in sites:
                    untrunc_high[st][f"{high} ({access})"] += 1
            continue
        trunc_shaders[st] += 1
        per_shader[st][len(trunc)] += 1
        kinds = {row["sgpr"]: row["kind"] for row in r["user_sgpr_reads"]}
        for sgpr, sites in trunc.items():
            trunc_sgpr[st][f"s{sgpr}"] += 1
            kind_shown[st][kinds.get(sgpr, "not read")] += 1
            for a in sorted({s[2] for s in sites}):
                trunc_access[st][a] += 1
            trunc_low[st]["+".join(sorted({s[0] for s in sites}))] += 1
            if sgpr in other:
                both_ways[st] += 1
        for sgpr, sites in other.items():
            for _, high, access in sites:
                untrunc_high[st][f"{high} ({access})"] += 1
    stages = [s for s in STAGES if shaders[s]] + sorted(s for s in shaders if s not in STAGES)

    def per(d):
        return {s: dict(sorted(d[s].items())) for s in stages if d[s]}

    return {
        "shaders": {s: shaders[s] for s in stages},
        "total": sum(shaders.values()),
        "reached_metadata": {s: reached[s] for s in stages},
        "not_reached": dict(errors.most_common()),
        "truncating_shaders": {s: trunc_shaders[s] for s in stages},
        "pointer_mode": per(mode),
        "truncated_pointers_per_shader": {s: {str(k): v for k, v in sorted(per_shader[s].items())}
                                          for s in stages if per_shader[s]},
        "truncated_pointer_access": per(trunc_access),
        "truncated_pointer_low_half": per(trunc_low),
        "truncated_pointer_sgpr": {s: dict(sorted(trunc_sgpr[s].items(), key=lambda kv: int(kv[0][1:])))
                                   for s in stages if trunc_sgpr[s]},
        "truncated_pointer_kind_in_index": per(kind_shown),
        "truncated_also_used_untruncated": {s: both_ways[s] for s in stages if both_ways[s]},
        "untruncated_site_high_half": per(untrunc_high),
        "not_seen": per(blind),
        "flag_mismatches": mismatch,
    }


def print_summary(s, out=None):
    out = out or sys.stdout
    w = lambda text="": print(text, file=out)  # noqa: E731
    w(f"{s['total']} shaders: " + ", ".join(f"{k} {v}" for k, v in s["shaders"].items()))
    w("reached shader_meta: " + ", ".join(f"{k} {v}" for k, v in s["reached_metadata"].items())
      + (f"; not reached {s['not_reached']}" if s["not_reached"] else ""))
    total = sum(s["truncating_shaders"].values())
    w(f"truncate a pointer's high dword: {total}: " + ", ".join(f"{k} {v}" for k, v in s["truncating_shaders"].items()))
    for title, key in (("pointers (all/some/none truncated)", "pointer_mode"),
                       ("truncated pointers per shader", "truncated_pointers_per_shader"),
                       ("truncated pointer, by access", "truncated_pointer_access"),
                       ("truncated pointer, low half", "truncated_pointer_low_half"),
                       ("truncated pointer, user SGPR", "truncated_pointer_sgpr"),
                       ("truncated pointer, kind in user_sgpr_reads", "truncated_pointer_kind_in_index"),
                       ("high half of untruncated sites", "untruncated_site_high_half"),
                       ("not seen by the classifier", "not_seen")):
        w(f"{title}:")
        for st, row in s[key].items():
            w(f"  {st}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
    if s["truncated_also_used_untruncated"]:
        w(f"truncated pointer also dereferenced untruncated: {s['truncated_also_used_untruncated']}")
    w(f"flag mismatches against shader_meta.analyse_instructions: {s['flag_mismatches']}")


def listings(src):
    with os.scandir(src) as entries:
        return sorted(e.path for e in entries if e.name.endswith(".s") and e.is_file())


def positive_int(text: str) -> int:
    """--limit -1 dropped the last listing and --limit 0 scanned none, both with exit 0."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="folder of .s listings (<work>/shaders/disasm)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=positive_int, help="scan at most this many listings (at least 1)")
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument("--index-out", help="write {number: {stage, user_sgpr_reads}} for tools/user_sgpr_census.py")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1 (a smaller value would silently scan fewer listings)")
    if not os.path.isdir(a.src):
        print(f"ERROR: {a.src} is not a folder", file=sys.stderr)
        return 2
    if a.json and a.index_out and same_report_path(a.json, a.index_out):
        ap.error(f"--json and --index-out name one file: {a.json}; the second report would replace the first")
    files = listings(a.src)[: a.limit] if a.limit is not None else listings(a.src)
    jobs = pool_workers.process_workers(max(1, a.jobs))
    if jobs == 1 or len(files) < 64:
        records = scan_many(files)
    else:
        size = max(1, len(files) // (jobs * 8))
        chunks = [files[i:i + size] for i in range(0, len(files), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            records = [r for part in pool.map(scan_many, chunks) for r in part]
    summary = summarise(records)
    print_summary(summary)
    try:
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=1, sort_keys=True)
        if a.index_out:
            index = {str(n): {"stage": r["stage"], "user_sgpr_reads": r.get("user_sgpr_reads", [])}
                     for n, r in enumerate(records) if not r.get("error")}
            with open(a.index_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f, indent=1, sort_keys=True)
    except OSError as e:
        print(f"ERROR: cannot write: {e.strerror or e}", file=sys.stderr)
        return 2
    return 1 if summary["flag_mismatches"] else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
