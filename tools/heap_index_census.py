r"""Where does a directly heap-indexing shader get the descriptor-heap base? (D3D-329, stdlib only)

docs/d3d12_vulkan.md section 3.4 leaves open where the console driver passes the bound CBV/SRV/UAV and sampler heap
base (and stride) to a shader that indexes `ResourceDescriptorHeap` / `SamplerDescriptorHeap`: user SGPRs, which the
layer fills from the root signature, or system SGPRs, which the translator would have to expose as a new runtime
constant. This tool measures it over a disassembly folder.

For every shader it reads the DXBC `SFI0` heap-indexing bits (tools/shader_meta.read_dxbc_meta) and follows a forward
dataflow over the reachable basic blocks, tagging every SGPR with where its value came from:

  * ("in", n)       the value SGPR n held at entry (user data below the stage's user count, system above it; the
                    count is COMPUTE_PGM_RSRC2's USER_SGPR for CS, and for the graphics stages the translator's 32,
                    with the NGG s3 as system in MS/GS: no graphics count is measured yet);
  * ("const", v)    a 32-bit constant;
  * ("ld", pc, i)   dword i of an s_load / s_buffer_load at code offset pc;
  * ("off", t, d)   tag t plus an offset, d = True when the offset is not a constant.

Each descriptor-width scalar load (s_load_dwordx4/x8/x16 from an SGPR pair) is then classified by its base: a
64-bit pointer from an input pair, a 32-bit pointer from one input SGPR with the high word a constant 0, a pointer
that was itself loaded from memory, or something else. Its offset is static (immediate or a known constant) or
dynamic, and the result's role (T#, S#, V#) comes from the instruction that consumes it. The census groups the
loads per stage and SFI0 class, so the heap base shows up as the input SGPR the loads of heap-indexing shaders
start from, and as the difference to the shaders that use only descriptor tables.

It reads only the listings, the DXBC containers next to them (`<disasm>/../<name>.dxbc`), the carve index
(`<disasm>/../index.json`, for the stage) and, for compute shaders, COMPUTE_PGM_RSRC2 at FSSM header word 0x138
(`<disasm>/../fssm/<name>.fssm`, as tools/rdna2spirv.py does). It prints counts only, no shader bytes.

Usage:
    python -I tools/heap_index_census.py <work>/shaders/dxbc/disasm [--json OUT] [--limit N]

A missing folder, an unreadable carve index and an unwritable --json each give one ERROR line and exit 2.
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402
from report_paths import same_report_path  # noqa: E402

USER_SGPRS = 32          # graphics stages: user data s0..s31 (tools/rdna2spirv.py USER_SGPRS)
SYSTEM_SGPRS = {"ms": (3,), "gs": (3,)}   # NGG wave info in s3 (tools/rdna2spirv.py sys_sgprs)
DESC_LOADS = {"s_load_dwordx4": 4, "s_load_dwordx8": 8, "s_load_dwordx16": 16}
BRANCHES = {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
            "s_cbranch_execz", "s_cbranch_execnz"}
ADDS = {"s_add_u32", "s_add_i32", "s_sub_u32", "s_sub_i32", "s_addc_u32", "s_subb_u32"}


class Skip(Exception):
    """A listing the census cannot follow; the reason is counted, not fatal."""


# ---- inputs -------------------------------------------------------------------------------------------------

def stage_index(disasm_dir):
    path = os.path.join(os.path.dirname(os.path.abspath(disasm_dir)), "index.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {e["file"][:-5]: e["stage"] for e in rows}


def sfi0_class(disasm_dir, name):
    """'res+smp', 'res', 'smp', 'none', or None when the container is missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(disasm_dir)), name + ".dxbc")
    try:
        with open(path, "rb") as f:
            meta = shader_meta.read_dxbc_meta(f.read())
    except OSError:
        return None
    if "sfi0_resource_heap_directly_indexed" not in meta:
        return None
    res = meta["sfi0_resource_heap_directly_indexed"]
    smp = meta["sfi0_sampler_heap_directly_indexed"]
    return "res+smp" if res and smp else "res" if res else "smp" if smp else "none"


def cs_user_count(disasm_dir, name):
    """USER_SGPR [5:1] of COMPUTE_PGM_RSRC2 (FSSM header word 0x138), or None."""
    path = os.path.join(os.path.dirname(os.path.abspath(disasm_dir)), "fssm", name + ".fssm")
    try:
        with open(path, "rb") as f:
            hdr = f.read(0x13C)
    except OSError:
        return None
    if len(hdr) < 0x13C:
        return None
    return (struct.unpack_from("<I", hdr, 0x138)[0] >> 1) & 31


# ---- control flow -------------------------------------------------------------------------------------------

def blocks_of(insns):
    """Basic blocks and successor lists, split at branches as tools/rdna2spirv.py does."""
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        if imm is None:
            raise Skip("branch without offset")
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    if any(shader_meta.base_op(i["op"]) in BRANCHES and not i.get("enc") for i in insns):
        raise Skip("branch without encoding")
    leaders = {insns[0]["pc"]}
    for ins in insns:
        if shader_meta.base_op(ins["op"]) in BRANCHES:
            leaders.add(target(ins))
            leaders.add(next_pc(ins))
    pcs = {i["pc"] for i in insns}
    if any(p not in pcs for p in leaders if p != next_pc(insns[-1])):
        raise Skip("branch target outside code")
    starts = sorted(p for p in leaders if p in pcs)
    index = {p: n for n, p in enumerate(starts)}
    if any(target(i) not in index for i in insns if shader_meta.base_op(i["op"]) in BRANCHES):
        raise Skip("branch target outside code")
    blocks = [[] for _ in starts]
    cur = -1
    for ins in insns:
        cur = index.get(ins["pc"], cur)
        blocks[cur].append(ins)
    succ = []
    for n, blk in enumerate(blocks):
        last = next((i for i in blk if shader_meta.base_op(i["op"]) == "s_endpgm" or
                     shader_meta.base_op(i["op"]) in BRANCHES), None)
        fall = [n + 1] if n + 1 < len(blocks) else []
        if last is None:
            succ.append(fall)
        elif shader_meta.base_op(last["op"]) == "s_endpgm":
            succ.append([])
        elif shader_meta.base_op(last["op"]) == "s_branch":
            succ.append([index[target(last)]])
        else:
            succ.append(list(dict.fromkeys([index[target(last)]] + fall)))
    return blocks, succ


# ---- dataflow -----------------------------------------------------------------------------------------------

def root(tag):
    """(underlying tag, dynamic offset?) with the ("off", ...) wrappers removed."""
    dyn = False
    while tag is not None and tag[0] == "off":
        dyn = dyn or tag[2]
        tag = tag[1]
    return tag, dyn


def _src(sym, a):
    """Tag of a one-dword source operand, or None."""
    c = shader_meta._imm_u32(a)
    if c is not None:
        return ("const", c)
    if a["k"] == "sgpr" and a.get("n", 1) == 1:
        return sym.get(a["r"])
    if a["k"] == "special" and a.get("t") == "null":
        return ("const", 0)
    return None


def _pointerish(tag):
    t, _ = root(tag)
    return t is not None and t[0] in ("in", "ld")


def step(sym, ins, on_load=None, on_use=None):
    """Advance the SGPR tags over one instruction; report descriptor loads and their consumers."""
    op = shader_meta.base_op(ins["op"])
    a = ins["args"]
    if not a:
        return
    if on_use is not None:
        if op.startswith("image_") and len(a) >= 3 and a[2]["k"] == "sgpr":
            on_use(sym.get(a[2]["r"]), "T#")
            if op.startswith(shader_meta.SAMPLER_IMAGE_OPS) and len(a) >= 4 and a[3]["k"] == "sgpr":
                on_use(sym.get(a[3]["r"]), "S#")
        elif op.startswith(("buffer_", "tbuffer_")) and len(a) >= 3 and a[2]["k"] == "sgpr":
            on_use(sym.get(a[2]["r"]), "V#")
        elif op.startswith("s_buffer_") and len(a) >= 2 and a[1]["k"] == "sgpr":
            on_use(sym.get(a[1]["r"]), "V#")
    if on_load is not None and op in DESC_LOADS and len(a) >= 2 and a[1]["k"] == "sgpr" and a[1].get("n") == 2:
        b = a[1]["r"]
        off = _src(sym, a[2]) if len(a) >= 3 else ("const", 0)
        on_load(ins["pc"], op, sym.get(b), sym.get(b + 1), off)

    _, writes = shader_meta._split_operands(ins)
    if not writes:
        return
    new = {}
    if op.startswith(("s_load_dword", "s_buffer_load_dword")) and a[0]["k"] == "sgpr":
        for i in range(a[0].get("n", 1)):
            new[a[0]["r"] + i] = ("ld", ins["pc"], i)
    elif op in ("s_mov_b32", "s_movk_i32") and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        if op == "s_movk_i32":
            v = a[1]["v"] & 0xFFFF
            new[a[0]["r"]] = ("const", (v - 0x10000 if v & 0x8000 else v) & 0xFFFFFFFF)
        else:
            t = _src(sym, a[1])
            if t is not None:
                new[a[0]["r"]] = t
    elif op == "s_mov_b64" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n") == 2:
        d = a[0]["r"]
        if a[1]["k"] == "imm":
            v = a[1]["v"] & 0xFFFFFFFFFFFFFFFF
            new[d], new[d + 1] = ("const", v & 0xFFFFFFFF), ("const", v >> 32)
        elif a[1]["k"] == "sgpr" and a[1].get("n") == 2:
            for i in (0, 1):
                if a[1]["r"] + i in sym:
                    new[d + i] = sym[a[1]["r"] + i]
    elif (op in ADDS or op in shader_meta.LSHL_ADD_SHIFT) and len(a) >= 3 and a[0]["k"] == "sgpr":
        tx, ty = _src(sym, a[1]), _src(sym, a[2])
        if op in shader_meta.LSHL_ADD_SHIFT:
            tx, ty = ty, tx            # s_lshlK_add_u32 d, idx, base: the base is the second source
        cx = tx[1] if tx is not None and tx[0] == "const" else None
        cy = ty[1] if ty is not None and ty[0] == "const" else None
        if cx is not None and cy is not None and op not in ("s_addc_u32", "s_subb_u32"):
            if op in shader_meta.LSHL_ADD_SHIFT:
                new[a[0]["r"]] = ("const", ((cy << shader_meta.LSHL_ADD_SHIFT[op]) + cx) & 0xFFFFFFFF)
            else:
                new[a[0]["r"]] = ("const", shader_meta._fold_scalar_alu(op, cx, cy))
        elif cx == 0 and cy == 0:
            new[a[0]["r"]] = ("const", 0)          # high word of a pointer below 4 GB plus a carry of 0
        elif _pointerish(tx) and not _pointerish(ty):
            new[a[0]["r"]] = ("off", tx, cy is None)
        elif _pointerish(ty) and not _pointerish(tx) and not op.startswith(("s_sub", "s_subb")):
            new[a[0]["r"]] = ("off", ty, cx is None)
    elif op == "s_addk_i32" and a[0]["k"] == "sgpr":
        old = sym.get(a[0]["r"])
        if old is not None and old[0] == "const":
            v = a[1]["v"] & 0xFFFF
            new[a[0]["r"]] = ("const", (old[1] + (v - 0x10000 if v & 0x8000 else v)) & 0xFFFFFFFF)
        elif _pointerish(old):
            new[a[0]["r"]] = ("off", old, False)
    elif op in ("s_and_b32", "s_or_b32", "s_lshl_b32", "s_lshr_b32", "s_mul_i32", "s_xor_b32") and len(a) >= 3:
        tx, ty = _src(sym, a[1]), _src(sym, a[2])
        if tx is not None and ty is not None and tx[0] == ty[0] == "const":
            new[a[0]["r"]] = ("const", shader_meta._fold_scalar_alu(op, tx[1], ty[1]))
    for r in writes:
        if r in new:
            sym[r] = new[r]
        else:
            sym.pop(r, None)


def analyse(insns, stage, user_count):
    """Descriptor loads of one shader: [{"op", "base", "ptr", "offset", "roles"}], in code order."""
    blocks, succ = blocks_of(insns)
    system = set(SYSTEM_SGPRS.get(stage, ()))
    entry = [None] * len(blocks)
    entry[0] = {r: ("in", r) for r in range(106)}
    work = [0]
    while work:
        b = work.pop()
        sym = dict(entry[b])
        for ins in blocks[b]:
            if shader_meta.base_op(ins["op"]) == "s_endpgm":
                break
            step(sym, ins)
        for s in succ[b]:
            merged = sym if entry[s] is None else {k: v for k, v in entry[s].items() if sym.get(k) == v}
            if merged != entry[s]:
                entry[s] = dict(merged)
                work.append(s)

    loads = {}

    def on_load(pc, op, lo, hi, off):
        rlo, dyn_lo = root(lo)
        rhi, _ = root(hi)
        if rlo is not None and rlo[0] == "in":
            n = rlo[1]
            where = "system" if (n in system or n >= user_count) else "user"
            if rhi == ("in", n + 1):
                base, ptr = f"s[{n}:{n + 1}]", "ptr64"
            elif hi == ("const", 0):
                base, ptr = f"s{n}", "ptr32"
            else:
                base, ptr = f"s{n}", "ptr64_lo_only"
            base = f"{where} {base}"
        elif rlo is not None and rlo[0] == "ld":
            base, ptr = "loaded pointer", "ptr32" if hi == ("const", 0) else "ptr64"
        elif rlo is not None and rlo[0] == "const":
            base, ptr = "constant", "const"
        else:
            base, ptr = "unknown", "unknown"
        if off is not None and off[0] == "const" and not dyn_lo:
            offset = ("static", off[1])
        else:
            offset = ("dynamic", None)
        loads[pc] = {"op": op, "base": base, "ptr": ptr, "offset": offset, "roles": set()}

    def on_use(tag, role):
        if tag is not None and tag[0] == "ld" and tag[2] == 0 and tag[1] in loads:
            loads[tag[1]]["roles"].add(role)

    for b in range(len(blocks)):
        if entry[b] is None:
            continue
        sym = dict(entry[b])
        for ins in blocks[b]:
            if shader_meta.base_op(ins["op"]) == "s_endpgm":
                break
            step(sym, ins, on_load, on_use)
    return [loads[pc] for pc in sorted(loads)]


# ---- census -------------------------------------------------------------------------------------------------

def role_of(load):
    if len(load["roles"]) == 1:
        return next(iter(load["roles"]))
    if not load["roles"]:
        return "unused" if load["op"] != "s_load_dwordx16" else "x16"
    return "mixed"


def census(disasm_dir, limit=None, progress=None, stages=None):
    if stages is None:
        stages = stage_index(disasm_dir)
    names = sorted(n[:-2] for n in os.listdir(disasm_dir) if n.endswith(".s"))
    if limit:
        names = names[:limit]
    skipped = Counter()
    shaders = Counter()                                  # (stage, sfi0) -> shaders
    with_loads = Counter()                               # (stage, sfi0) -> shaders with a descriptor load
    base_shaders = defaultdict(Counter)                  # (stage, sfi0) -> base -> shaders loading from it
    load_rows = defaultdict(Counter)                     # (stage, sfi0) -> (base, ptr, role, offset kind) -> loads
    static_align = defaultdict(Counter)                  # (stage, sfi0, role) -> "aligned32"/"aligned16"/"other"
    static_max = defaultdict(int)                        # (stage, sfi0, role) -> largest static offset
    for k, name in enumerate(names):
        if progress and k % 2000 == 0:
            progress(k, len(names))
        stage = stages.get(name)
        sfi0 = sfi0_class(disasm_dir, name)
        if stage is None or sfi0 is None:
            skipped["no stage or SFI0"] += 1
            continue
        info = shader_ir.parse_file(os.path.join(disasm_dir, name + ".s"))
        if not info["insns"]:
            skipped["no instructions"] += 1
            continue
        if info["invalid"]:
            skipped["embedded data words"] += 1
            continue
        user = USER_SGPRS
        if stage == "cs":
            user = cs_user_count(disasm_dir, name)
            if user is None:
                skipped["no FSSM header"] += 1
                continue
        try:
            loads = analyse(info["insns"], stage, user)
        except Skip as e:
            skipped[str(e)] += 1
            continue
        key = (stage, sfi0)
        shaders[key] += 1
        if loads:
            with_loads[key] += 1
        for base in {ld["base"] for ld in loads}:
            base_shaders[key][base] += 1
        for ld in loads:
            role = role_of(ld)
            load_rows[key][(ld["base"], ld["ptr"], role, ld["offset"][0])] += 1
            if ld["offset"][0] == "static":
                v = ld["offset"][1]
                static_align[(stage, sfi0, role)]["aligned32" if v % 32 == 0 else
                                                  "aligned16" if v % 16 == 0 else "other"] += 1
                static_max[(stage, sfi0, role)] = max(static_max[(stage, sfi0, role)], v)
    out = {"shaders_read": len(names), "skipped": dict(skipped), "groups": {}}
    for key in sorted(shaders):
        stage, sfi0 = key
        g = {
            "shaders": shaders[key],
            "shaders_with_descriptor_loads": with_loads[key],
            "shaders_per_base": dict(base_shaders[key].most_common()),
            "loads": [{"base": b, "ptr": p, "role": r, "offset": o, "count": c}
                      for (b, p, r, o), c in load_rows[key].most_common()],
            "static_offsets": {role: {**dict(static_align[(s, f, role)]), "max": static_max[(s, f, role)]}
                               for (s, f, role) in sorted(static_align) if (s, f) == key},
        }
        out["groups"][f"{stage} sfi0={sfi0}"] = g
    return out


def print_report(out, top=6):
    print(f"shaders read: {out['shaders_read']}; skipped: {out['skipped'] or 'none'}")
    for name, g in out["groups"].items():
        print(f"\n{name}: {g['shaders']} shaders, {g['shaders_with_descriptor_loads']} with descriptor loads")
        for base, n in list(g["shaders_per_base"].items())[:top]:
            print(f"  base {base:<22} {n:>6} shaders")
        for row in g["loads"][:top]:
            print(f"  load {row['base']:<22} {row['ptr']:<14} {row['role']:<6} {row['offset']:<8} {row['count']:>7}")
        for role, st in g["static_offsets"].items():
            print(f"  static {role:<6} " + ", ".join(f"{k} {v}" for k, v in st.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="<work>/shaders/dxbc/disasm")
    ap.add_argument("--json", help="write the full census here")
    ap.add_argument("--limit", type=shader_ir.positive_int, help="only the first N listings (sorted by name)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.disasm):
        print(f"ERROR: {args.disasm} is not a folder", file=sys.stderr)
        return 2
    if args.json:
        # The census is written last, after every listing has been read, so a report replacing one of
        # them or the carve index would silently destroy what the run just used.
        index_path = os.path.join(os.path.dirname(os.path.abspath(args.disasm)), "index.json")
        if same_report_path(args.json, index_path):
            print(f"ERROR: --json would overwrite the carve index: {args.json}", file=sys.stderr)
            return 2
        disasm = os.path.realpath(args.disasm)
        target = os.path.realpath(os.path.abspath(args.json))
        try:
            inside = os.path.commonpath([disasm, target]) == disasm
        except ValueError:  # paths on different Windows drives
            inside = False
        if inside:
            print(f"ERROR: --json must not write inside the disasm folder it reads: {args.json}", file=sys.stderr)
            return 2
    try:
        stages = stage_index(args.disasm)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"ERROR: cannot read the carve index next to {args.disasm}: {e}", file=sys.stderr)
        return 2
    if args.json:
        # Besides the listings and the carve index, the census reads each shader's DXBC container and
        # FSSM header next to the disasm folder. A report landing on one of those (or a hard link to it)
        # would destroy an input the run just used, so refuse those names too.
        parent = os.path.dirname(os.path.abspath(args.disasm))
        for name in stages:
            for path in (os.path.join(parent, name + ".dxbc"),
                         os.path.join(parent, "fssm", name + ".fssm")):
                if same_report_path(args.json, path):
                    print(f"ERROR: --json would overwrite an input the census reads: {path}", file=sys.stderr)
                    return 2
    out = census(args.disasm, args.limit, stages=stages,
                 progress=lambda k, n: print(f"  {k}/{n}", file=sys.stderr, flush=True))
    print_report(out)
    if args.json:
        try:
            d = os.path.dirname(os.path.abspath(args.json))
            os.makedirs(d, exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {args.json}: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
