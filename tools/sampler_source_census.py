r"""Where does the S# of every sampling instruction come from? (SHADER-735, stdlib only)

docs/d3d12_vulkan.md section 3.5 left open whether the console driver hands a shader its static samplers through
a fixed table of its own: a shader that expected one would read the S# from an address the layer never fills, and
the layer would have to emulate that table. This tool measures it over a disassembly folder.

For every `image_sample*`, `image_gather4*` and `image_get_lod` it follows the four SGPRs of the sampler operand
back to where their values were made. The forward dataflow is the one of tools/heap_index_census.py (entry SGPRs,
constants, dwords of a scalar load, offsets) with what the sampler operands of the corpus needed on top: constants
built with `s_brev_b32`, `s_getpc_b64` as a code-relative address, SGPRs spilled to a VGPR and read back
(`v_mov_b32` / `v_readfirstlane_b32`, `v_writelane_b32` / `v_readlane_b32`), and joins that keep the sources of
every path instead of giving the value up. Each instruction gets one class:

  * user_sgprs     every dword is a user-data SGPR at entry: the S# is root data itself;
  * user_pointer   every dword was loaded from memory (s_load_dword*, s_buffer_load_dword*) at an address that
                   starts from user data: a pointer in a user SGPR, or a pointer or V# that was itself loaded
                   through one. A descriptor table and a directly indexed sampler heap both look like this
                   (D3D-329: the heap base is a user SGPR too), so the class is also counted by offset (static
                   or run-time) and by the shader's DXBC `SFI0` sampler-heap bit;
  * literal        every dword is a constant built from immediates, the `literal_samplers` of
                   tools/shader_meta.py; the ones that take `s_brev_b32` or `s_not_b32` are counted apart;
  * other_base     some dword, on some path, is held in or loaded through something that is not user data: a
                   constant address, a system SGPR (above the stage's user count, or the NGG s3 in MS/GS) or a
                   code-relative address. This is what a driver sampler table would look like, so it wins over
                   every other class, and the form is counted;
  * undetermined   anything else, with the reason: a value the dataflow does not model, sources of different
                   classes on different paths, a pointer whose high word is not followed, an S# that comes back
                   from a VGPR a vector load wrote, or from a spill slot that one path into the instruction does
                   not save to (the reason then names the class of what the other paths saved).

Most of the undetermined ones in the corpus have one cause: the compiler builds or saves the S# inside a region it
jumps over with `s_cbranch_execz` when no lane is active, and samples after the region, so the path that skips it
brings whatever the registers held. The census therefore analyses a shader with such an instruction a second time
without the taken edge of those branches (the region always runs, which is what the code does when a lane is
active) and reports, next to the classes above, what the undetermined instructions become. The first analysis
stays the one the classes and --expect-none use: the second trusts that nothing samples on the skipping path.

The user count is COMPUTE_PGM_RSRC2's USER_SGPR for CS and the translator's 32 for the graphics stages, as in
tools/heap_index_census.py: no graphics count is measured yet, so "user" there means "an entry SGPR below s32".
A listing with data words inside its code range is re-anchored to the FSSM blob and cut to the reachable code the
way tools/rdna2spirv.py does it.

What it does NOT show:
  * `user pointer + constant` and `constant address + user index` are the same arithmetic. Instead of deciding,
    the census reports the largest constant added on the way to the address of a user_pointer load and the
    largest static offset of such a load.
  * It sees the shaders: not what the driver writes at the addresses they read, and not which literal S# is a
    root signature's static sampler.
  * Lane masks are not modelled. A spill is a masked write (`v_mov_b32` writes the active lanes only); a restore
    is taken to read what the last save on its path wrote.

It reads the listings, the carve index (`<disasm>/../index.json`, for the stage), the DXBC containers
(`<disasm>/../<name>.dxbc`, for SFI0) and the FSSM blobs (`<disasm>/../fssm/<name>.fssm`). It prints counts only,
no shader names or bytes.

Usage:
    python -I tools/sampler_source_census.py <work>/shaders/dxbc/disasm [--json OUT] [--jobs N] [--limit N]
    python -I tools/sampler_source_census.py <work>/shaders/dxbc/disasm --expect-none other_base
        (exit 1 if an instruction falls in a listed class)

A missing folder, an unreadable carve index and an unwritable --json each give one ERROR line and exit 2.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heap_index_census as hic  # noqa: E402
import pool_workers  # noqa: E402
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402

CLASSES = ("user_sgprs", "user_pointer", "literal", "other_base", "undetermined")
ENDS = ("s_endpgm", "s_setpc_b64")           # no fall-through and no successor the listing names
SCALAR_LOADS = ("s_load_dword", "s_buffer_load_dword")
# Constants the compiler builds with one more instruction than heap_index_census.step folds.
UNARY = {"s_brev_b32": lambda v: int(f"{v:032b}"[::-1], 2), "s_not_b32": lambda v: ~v & 0xFFFFFFFF}
PHI_MAX = 8                                  # sources kept at a join before the value is given up
VGPR_READ_ONLY = ("exp", "buffer_store", "tbuffer_store", "global_store", "flat_store", "scratch_store",
                  "image_store", "ds_write", "v_cmp")   # a[0] is a VGPR these read; every other one writes it
VGPR_LOADS = ("buffer_load", "tbuffer_load", "global_load", "flat_load", "scratch_load", "ds_read")
GAP, VLOAD, TOP = ("gap",), ("vload",), ("top",)   # spill-slot contents that are not a saved SGPR (see dataflow)
NOT_USER = {"constant": "constant address", "system": "system SGPR", "code": "code-relative address"}


# ---- listing -> basic blocks --------------------------------------------------------------------------------

def anchored(path, info):
    """The instructions of a listing with data words in its code range, at their true offsets in the FSSM blob and
    without the unreachable ones (tools/rdna2spirv.py translate() does the same before it builds the blocks)."""
    import rdna2spirv              # only these listings need the translator's front end
    try:
        blob = rdna2spirv.native_blob(path, info["name"])
    except OSError:
        raise hic.Skip("embedded data words, no FSSM blob") from None
    pos = info["code_start"]
    for ins in info["insns"]:
        if not ins["enc"]:
            raise hic.Skip("embedded data words, instruction without encoding")
        enc = bytes.fromhex(ins["enc"])
        at, skipped = pos, 0
        while not rdna2spirv.same_encoding(blob[at:at + len(enc)], enc, ins["op"], len(ins["args"])):
            at += 4
            skipped += 4
            if skipped > 4096 or at >= len(blob):
                at = pos           # a data word LLVM printed normalized: it sits where the decoder stood
                ins["normalized"] = True
                break
        ins["pc"] = at
        pos = at + len(enc)
    try:
        insns = rdna2spirv.reachable_code(rdna2spirv.fold_far_branches(info["insns"]))
    except rdna2spirv.Unsupported as e:
        raise hic.Skip(f"embedded data words, {e}") from None
    if any(i.get("normalized") for i in insns):
        raise hic.Skip("embedded data words, reachable instruction with normalized encoding")
    return insns


def blocks_of(insns):
    """Basic blocks and successor lists. Unlike heap_index_census.blocks_of it takes the explicit `target` of a
    folded far branch and code that is not contiguous (data words between two reachable runs)."""
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        if "target" in ins:
            return ins["target"]
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        if imm is None:
            raise hic.Skip("branch without offset")
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    def is_branch(ins):
        return shader_meta.base_op(ins["op"]) in hic.BRANCHES

    if any(is_branch(i) and not i.get("enc") for i in insns):
        raise hic.Skip("branch without encoding")
    pcs = {i["pc"] for i in insns}
    leaders = {insns[0]["pc"]}
    for prev, ins in zip(insns, insns[1:]):
        if next_pc(prev) != ins["pc"]:
            leaders.add(ins["pc"])
    for ins in insns:
        if is_branch(ins):
            leaders.add(target(ins))
            leaders.add(next_pc(ins))
    starts = sorted(p for p in leaders if p in pcs)
    index = {p: n for n, p in enumerate(starts)}
    if any(target(i) not in index for i in insns if is_branch(i)):
        raise hic.Skip("branch target outside code")
    blocks = [[] for _ in starts]
    cur = -1
    for ins in insns:
        cur = index.get(ins["pc"], cur)
        blocks[cur].append(ins)
    succ = []
    for n, blk in enumerate(blocks):
        last = next((i for i in blk if is_branch(i) or shader_meta.base_op(i["op"]) in ENDS), None)
        fall = [n + 1] if n + 1 < len(blocks) and next_pc(blk[-1]) == starts[n + 1] else []
        if last is None:
            succ.append(fall)
        elif not is_branch(last):
            succ.append([])
        elif shader_meta.base_op(last["op"]) == "s_branch":
            succ.append([index[target(last)]])
        else:
            succ.append(list(dict.fromkeys([index[target(last)]] + fall)))
    return blocks, succ


# ---- dataflow -----------------------------------------------------------------------------------------------
#
# The state maps an SGPR number to a tag, as in heap_index_census.step, and adds:
#   ("code", pc, word)        the pair s_getpc_b64 writes: an address next to the code;
#   ("const", v, op)          a constant folded here from op (UNARY), which that step leaves unknown;
#   ("off", t, False, k)      the constant k of a static offset, so the largest one can be reported;
#   ("phi", {tags})           a join whose paths bring different constants, loads or entry SGPRs;
#   key ("v", n)              the tag of an SGPR spilled whole to VGPR n (v_mov_b32 vN, sM);
#   key ("lane", n, lane)     the tag of an SGPR spilled to one lane of VGPR n (v_writelane_b32).
# A spill slot holds, besides a saved tag: nothing an SGPR was saved to (no key; GAP inside a phi), VLOAD when a
# vector memory load wrote the VGPR, TOP when a join gave it up.

def _signed(v):
    return v - (1 << 32) if v & 0x80000000 else v


def spill_lanes(insns):
    """{vgpr: {lanes}} of every v_writelane_b32 with an immediate lane: the spill slots a VGPR write clears."""
    out = defaultdict(set)
    for ins in insns:
        a = ins["args"]
        if shader_meta.base_op(ins["op"]) == "v_writelane_b32" and len(a) >= 3 and a[0]["k"] == "vgpr" \
                and a[2]["k"] == "imm":
            out[a[0]["r"]].add(a[2]["v"])
    return out


def step(sym, ins, lanes=None):
    """heap_index_census.step plus the tags above."""
    op = shader_meta.base_op(ins["op"])
    a = ins["args"]
    lanes = lanes or {}
    srcs = unary = None
    if (op in hic.ADDS or op in shader_meta.LSHL_ADD_SHIFT) and len(a) >= 3 and a[0]["k"] == "sgpr":
        srcs = (hic._src(sym, a[1]), hic._src(sym, a[2]))
    elif op == "s_addk_i32" and len(a) >= 2 and a[0]["k"] == "sgpr":
        v = a[1]["v"] & 0xFFFF
        srcs = (sym.get(a[0]["r"]), ("const", (v - 0x10000 if v & 0x8000 else v) & 0xFFFFFFFF))
    elif op in UNARY and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        unary = hic._src(sym, a[1])
    elif op == "v_readfirstlane_b32" and len(a) >= 2 and a[1]["k"] == "vgpr":
        unary = sym.get(("v", a[1]["r"]))
    elif op == "v_readlane_b32" and len(a) >= 3 and a[1]["k"] == "vgpr" and a[2]["k"] == "imm":
        unary = sym.get(("lane", a[1]["r"], a[2]["v"]))
    hic.step(sym, ins)
    if not a:
        return
    if a[0]["k"] == "vgpr":
        if op.startswith(VGPR_READ_ONLY):
            return
        written = [x for x in (a if op == "v_swap_b32" else a[:1]) if x["k"] == "vgpr"]
        lane = a[2]["v"] if op == "v_writelane_b32" and len(a) >= 3 and a[2]["k"] == "imm" else None
        moved = None
        if op in ("v_mov_b32", "v_writelane_b32") and len(a) >= 2 and a[0].get("n", 1) == 1:
            moved = sym.get(("v", a[1]["r"])) if a[1]["k"] == "vgpr" else hic._src(sym, a[1])
        for x in written:
            for r in range(x["r"], x["r"] + x.get("n", 1)):
                sym.pop(("v", r), None)
                if op.startswith(VGPR_LOADS):
                    sym[("v", r)] = VLOAD
                for one in (lane,) if lane is not None else lanes.get(r, ()):   # v_writelane leaves the others
                    sym.pop(("lane", r, one), None)
        if moved is not None and (op == "v_mov_b32" or lane is not None):
            sym[("v", a[0]["r"]) if op == "v_mov_b32" else ("lane", a[0]["r"], lane)] = moved
    elif a[0]["k"] != "sgpr":
        return
    elif op in UNARY and unary is not None and unary[0] == "const":
        sym[a[0]["r"]] = ("const", UNARY[op](unary[1]), op)
    elif op.startswith(("v_readfirstlane", "v_readlane")) and unary not in (None, TOP) and a[0].get("n", 1) == 1:
        sym[a[0]["r"]] = unary
    elif op == "s_getpc_b64" and a[0].get("n") == 2:
        sym[a[0]["r"]], sym[a[0]["r"] + 1] = ("code", ins["pc"], 0), ("code", ins["pc"], 1)
    elif srcs is not None:
        d = a[0]["r"]
        consts = [t[1] for t in srcs if t is not None and t[0] == "const"]
        code = [t for t in srcs if hic.root(t)[0] is not None and hic.root(t)[0][0] == "code"]
        if d not in sym and len(code) == 1 and not (srcs[1] is code[0] and op.startswith(("s_sub", "s_subb"))):
            sym[d] = ("off", code[0], not consts)
        new = sym.get(d)
        if consts and new is not None and new[0] == "off" and not new[2]:
            shifted = srcs[0] is not None and srcs[0][0] == "const"     # s_lshlK_add_u32: the index is shifted
            sym[d] = new[:3] + (consts[0] << shader_meta.LSHL_ADD_SHIFT.get(op, 0) if shifted else consts[0],)


def join(x, y):
    """The tag of a value that is x on one path and y on another, or None when that is not worth keeping."""
    alts = (x[1] if x[0] == "phi" else frozenset((x,))) | (y[1] if y[0] == "phi" else frozenset((y,)))
    if len(alts) > PHI_MAX or any(t[0] not in ("const", "ld", "in", "gap", "vload") for t in alts):
        return None
    return ("phi", alts)


def merge(old, new):
    """The state at a join: `old` is what the block was entered with so far, `new` what one more path brings.
    An SGPR whose value is given up stays unknown. A spill slot one path did not write is GAP there, so the slot
    keeps what the other paths saved and the gap; given up, it is TOP."""
    out = {}
    for k, v in old.items():
        w = new.get(k, GAP if isinstance(k, tuple) else None)
        if w is not None:
            w = v if w == v else join(v, w)
            if w is not None or isinstance(k, tuple):
                out[k] = w or TOP
    for k, w in new.items():
        if isinstance(k, tuple) and k not in old:
            out[k] = join(GAP, w) or TOP
    return out


def analyse(insns, stage, user_count, follow_skips=True):
    """{code offset: (class, detail)} of the reachable sampling instructions, in code order, and the shader's
    statistics. detail: the offset kind for user_pointer ("static" / "dynamic"), the form for other_base, the
    reason for undetermined, the instruction a literal needed folded (or None).
    follow_skips=False leaves out the taken edge of every s_cbranch_execz that also falls through: the branch
    skips a region when no lane is active, so without it the region always runs (see the module docstring)."""
    blocks, succ = blocks_of(insns)
    if not follow_skips:
        succ = [s[1:] if len(s) == 2 and shader_meta.base_op(blk[-1]["op"]) == "s_cbranch_execz" else s
                for blk, s in zip(blocks, succ)]
    lanes = spill_lanes(insns)
    system = set(hic.SYSTEM_SGPRS.get(stage, ()))
    entry = [None] * len(blocks)
    entry[0] = {r: ("in", r) for r in range(106)}
    work = [0]
    while work:
        b = work.pop()
        sym = dict(entry[b])
        for ins in blocks[b]:
            if shader_meta.base_op(ins["op"]) in ENDS:
                break
            step(sym, ins, lanes)
        for s in succ[b]:
            merged = sym if entry[s] is None else merge(entry[s], sym)
            if merged != entry[s]:
                entry[s] = dict(merged)
                work.append(s)

    stats = {"max_pointer_constant": 0, "max_static_offset": 0, "unreachable": 0}
    loads, uses = {}, {}                     # pc -> load record; pc -> sampler dword tags
    for b, blk in enumerate(blocks):
        sym = dict(entry[b]) if entry[b] is not None else None
        for ins in blk:
            op = shader_meta.base_op(ins["op"])
            a = ins["args"]
            if sym is None:                  # a block nothing reaches, or the rest of one after s_endpgm
                stats["unreachable"] += op.startswith(shader_meta.SAMPLER_IMAGE_OPS) and len(a) >= 4
                continue
            if op in ENDS:
                sym = None
                continue
            if op.startswith(SCALAR_LOADS) and len(a) >= 2:
                slots = shader_meta._sgpr_slots(a[1], 2 if op.startswith("s_load_dword") else 4)
                loads[ins["pc"]] = {
                    "base": None if slots is None else [shader_meta._slot_tag(sym, s) for s in slots],
                    "offset": hic._src(sym, a[2]) if len(a) >= 3 else ("const", 0),
                }
            elif op.startswith(shader_meta.SAMPLER_IMAGE_OPS) and len(a) >= 4:
                slots = shader_meta._sgpr_slots(a[3], 4, exact=False)
                uses[ins["pc"]] = None if slots is None else [shader_meta._slot_tag(sym, s) for s in slots]
            step(sym, ins, lanes)

    # What load_source found for a load, by code offset. A descriptor's dwords all come from one load, and a chain
    # of loads asks for the first one from every dword of every later one, so each load is worked out once. Only
    # an answer that ran into no cycle is kept: that one does not depend on where the question started.
    found_load, cycles = {}, [0]

    def source(tag, seen=()):
        """(where the address or value `tag` starts: "user" | "system" | "constant" | "code" | None,
        whether an offset on the way is run-time, the largest constant offset on the way)."""
        if tag is not None and tag[0] == "phi":
            got = [source(t, seen) for t in tag[1]]
            where = {g[0] for g in got}
            return (where.pop() if len(where) == 1 else None), any(g[1] for g in got), max(g[2] for g in got)
        dyn, k = False, 0
        while tag is not None and tag[0] == "off":
            dyn = dyn or tag[2]
            k = max(k, abs(_signed(tag[3] & 0xFFFFFFFF)) if len(tag) > 3 else 0)
            tag = tag[1]
        if tag is None:
            return None, dyn, k
        if tag[0] == "in":
            return ("system" if tag[1] in system or tag[1] >= user_count else "user"), dyn, k
        if tag[0] == "const":
            return "constant", dyn, k
        if tag[0] == "code":
            return "code", dyn, k
        if tag[0] != "ld" or tag[1] not in loads:
            return None, dyn, k
        if tag[1] in seen:                             # a load whose address depends on the load itself
            cycles[0] += 1
            return None, dyn, k
        # ("ld", pc, i): a pointer or V# that was itself loaded
        where, _, k2, _ = found_load.get(tag[1]) or load_source(tag[1], seen)
        return where, dyn, max(k, k2)

    def load_source(pc, seen=()):
        """source() of the address the scalar load at pc reads, with the load's own offset counted as well
        (run-time or not, and the constants in it), and the reason when the start is unknown. `seen` holds the
        loads this one is being asked for."""
        if pc in found_load:
            return found_load[pc]
        before = cycles[0]
        found = one_load_source(loads[pc], seen + (pc,))
        if cycles[0] == before:
            found_load[pc] = found
        return found

    def one_load_source(ld, seen):
        base = ld["base"]
        if base is None:
            return None, False, 0, "load base not an SGPR range"
        if len(base) == 4:                             # s_buffer_load: the V# decides
            got = {source(t, seen) for t in base}
            where, dyn, k = next(iter(got)) if len(got) == 1 else (None, False, 0)
            reason = "V# from different sources"
        else:
            where, dyn, k = source(base[0], seen)
            reason = "load base not followed"
            if where == "user" and base[1] != ("const", 0) and source(base[1], seen)[0] != "user":
                where, reason = None, "pointer high word not followed"
        off = ld["offset"]
        static = off is not None and off[0] == "const" and not dyn
        return where, not static, k if static or off is None else max(k, source(off, seen)[2]), reason

    by_load = {}

    def one_load(pc):
        """The class of a dword the load at pc brought."""
        if pc not in by_load:
            ld = loads.get(pc)
            where, dyn, k, reason = load_source(pc) if ld else (None, False, 0, "load not seen")
            if where is None:
                by_load[pc] = "undetermined", reason
            elif where != "user":
                by_load[pc] = "other_base", NOT_USER[where]
            else:
                stats["max_pointer_constant"] = max(stats["max_pointer_constant"], k)
                if not dyn:
                    stats["max_static_offset"] = max(stats["max_static_offset"], ld["offset"][1])
                by_load[pc] = "user_pointer", "dynamic" if dyn else "static"
        return by_load[pc]

    def one_dword(tag):
        if tag[0] == "const":
            return "literal", f"with {tag[2]}" if len(tag) > 2 else None
        if tag[0] == "in":
            return ("user_sgprs", None) if source(tag)[0] == "user" else ("other_base", "system SGPR")
        return one_load(tag[1])

    def classify(tags):
        """The class of a sampler operand from the tags of its four dwords: the one class every source of every
        dword has, other_base as soon as one source is, undetermined otherwise."""
        if tags is None:
            return "undetermined", "sampler operand not an SGPR range"
        known = [t for t in tags if t is not None and t[0] in ("phi", "const", "in", "ld", "vload")]
        alts = [t[1] if t[0] == "phi" else frozenset((t,)) for t in known]
        found = {one_dword(t) for a in alts for t in a if t not in (GAP, VLOAD)}
        classes = {c for c, _ in found}
        if "other_base" in classes:                        # one dword, on one path, from a fixed table is enough
            return sorted(f for f in found if f[0] == "other_base")[0]
        if len(known) < len(tags):
            return "undetermined", "a dword the dataflow does not follow"
        if "undetermined" in classes:
            return sorted(f for f in found if f[0] == "undetermined")[0]
        if any(a <= {GAP, VLOAD} for a in alts):
            return "undetermined", "restored from a VGPR a vector load wrote"
        if len(classes) > 1:
            return "undetermined", "sources of different classes: " + "+".join(sorted(classes))
        cls = classes.pop()
        slot = {t for a in alts for t in a} & {GAP, VLOAD}
        if slot:
            how = "fills with a vector load" if VLOAD in slot else "does not save to"
            return "undetermined", f"spill slot a path {how} ({cls} where saved)"
        details = sorted({d for _, d in found if d})
        return cls, ("dynamic" if "dynamic" in details else "/".join(details)) if details else None

    return {pc: classify(tags) for pc, tags in uses.items()}, stats


# ---- census -------------------------------------------------------------------------------------------------

def read_shader(job):
    """One listing -> {"stage", "heap", "skip"}, plus {"uses": [(class, detail)], "stats", "straight"} when it
    was analysed. "straight" counts, per class, the instructions that are undetermined with every edge followed
    and get that class when the s_cbranch_execz skips are left out."""
    disasm_dir, name, stage = job
    sfi0 = hic.sfi0_class(disasm_dir, name)
    row = {"stage": stage, "heap": None if sfi0 is None else "smp" in sfi0, "skip": None}
    if stage is None:
        row["skip"] = "not in the carve index"
        return row
    path = os.path.join(disasm_dir, name + ".s")
    info = shader_ir.parse_file(path)
    if not info["insns"]:
        row["skip"] = "no instructions"
        return row
    user = hic.USER_SGPRS
    if stage == "cs":
        user = hic.cs_user_count(disasm_dir, name)
        if user is None:
            row["skip"] = "no FSSM header"
            return row
    try:
        insns = anchored(path, info) if info["invalid"] else info["insns"]
        uses, row["stats"] = analyse(insns, stage, user)
        row["uses"], row["straight"] = list(uses.values()), {}
        if any(c == "undetermined" for c, _ in row["uses"]):
            straight = analyse(insns, stage, user, follow_skips=False)[0]
            row["straight"] = dict(Counter(straight.get(pc, ("not reached",))[0] for pc, (c, _) in uses.items()
                                           if c == "undetermined"))
    except hic.Skip as e:
        row["skip"] = str(e)
        row["sampling"] = sum(1 for i in info["insns"] if len(i["args"]) >= 4 and
                              shader_meta.base_op(i["op"]).startswith(shader_meta.SAMPLER_IMAGE_OPS))
    return row


def summarise(rows):
    """The census from read_shader rows: per stage, instructions and shaders per class and per detail."""
    def counts():
        return {"instructions": 0, "shaders": 0}

    stages = defaultdict(lambda: {
        "shaders": 0, "shaders_sampling": 0, "shaders_without_sfi0": 0, "instructions": 0,
        "unreachable_instructions": 0, "undetermined_without_skips": Counter(),
        "classes": {c: {**counts(), "in_sampler_heap_shaders": 0} for c in CLASSES},
        "detail": defaultdict(lambda: defaultdict(counts))})
    skipped, skipped_sampling = Counter(), Counter()
    limits = {"max_pointer_constant": 0, "max_static_offset": 0}
    n = 0
    for row in rows:
        n += 1
        if row["skip"]:
            skipped[row["skip"]] += 1
            skipped_sampling[row["skip"]] += row.get("sampling", 0)
            continue
        st = stages[row["stage"]]
        st["shaders"] += 1
        st["shaders_sampling"] += bool(row["uses"])
        st["shaders_without_sfi0"] += row["heap"] is None
        st["instructions"] += len(row["uses"])
        st["unreachable_instructions"] += row["stats"]["unreachable"]
        st["undetermined_without_skips"].update(row["straight"])
        for (cls, detail), k in Counter(row["uses"]).items():
            st["classes"][cls]["instructions"] += k
            st["classes"][cls]["in_sampler_heap_shaders"] += k * bool(row["heap"])
            if detail:
                st["detail"][cls][detail]["instructions"] += k
                st["detail"][cls][detail]["shaders"] += 1
        for cls in {c for c, _ in row["uses"]}:
            st["classes"][cls]["shaders"] += 1
        if row["uses"]:
            for k in limits:
                limits[k] = max(limits[k], row["stats"][k])
    total = {c: sum(st["classes"][c]["instructions"] for st in stages.values()) for c in CLASSES}
    return {
        "shaders_read": n,
        "skipped": dict(sorted(skipped.items())),
        "skipped_sampling_instructions": {k: v for k, v in sorted(skipped_sampling.items()) if v},
        "stages": {s: {**st, "detail": {c: dict(sorted(d.items())) for c, d in sorted(st["detail"].items())},
                       "undetermined_without_skips": dict(sorted(st["undetermined_without_skips"].items()))}
                   for s, st in sorted(stages.items())},
        "total": {"instructions": sum(total.values()), "classes": total, **limits},
    }


def census(disasm_dir, limit=None, jobs=1, stages=None):
    if stages is None:
        stages = hic.stage_index(disasm_dir)
    names = sorted(n[:-2] for n in os.listdir(disasm_dir) if n.endswith(".s"))
    if limit:
        names = names[:limit]
    work = [(disasm_dir, name, stages.get(name)) for name in names]
    if jobs > 1:
        with ProcessPoolExecutor(pool_workers.process_workers(jobs)) as pool:
            return summarise(pool.map(read_shader, work, chunksize=64))
    return summarise(map(read_shader, work))


def print_report(out):
    print(f"shaders read: {out['shaders_read']}; skipped: {out['skipped'] or 'none'}"
          + (f" (sampling instructions in them: {out['skipped_sampling_instructions']})"
             if out["skipped_sampling_instructions"] else ""))
    print(f"{'stage':<6}{'shaders':>8}{'sampling':>9}{'instr.':>8}" + "".join(f"{c:>14}" for c in CLASSES))
    for stage, st in out["stages"].items():
        print(f"{stage:<6}{st['shaders']:>8}{st['shaders_sampling']:>9}{st['instructions']:>8}"
              + "".join(f"{st['classes'][c]['instructions']:>14}" for c in CLASSES))
    t = out["total"]
    print(f"{'all':<6}{sum(s['shaders'] for s in out['stages'].values()):>8}"
          f"{sum(s['shaders_sampling'] for s in out['stages'].values()):>9}{t['instructions']:>8}"
          + "".join(f"{t['classes'][c]:>14}" for c in CLASSES))
    print("instructions in shaders whose SFI0 sets the sampler-heap bit:")
    for stage, st in out["stages"].items():
        if st["instructions"]:
            print(f"{stage:<6}{'':>25}" + "".join(f"{st['classes'][c]['in_sampler_heap_shaders']:>14}" for c in CLASSES)
                  + (f"  ({st['shaders_without_sfi0']} shaders without SFI0)" if st["shaders_without_sfi0"] else ""))
    for stage, st in out["stages"].items():
        for cls, detail in st["detail"].items():
            print(f"  {stage} {cls}: "
                  + "; ".join(f"{k}: {v['instructions']} in {v['shaders']} shaders" for k, v in detail.items()))
        if st["undetermined_without_skips"]:
            print(f"  {stage} undetermined, with the s_cbranch_execz skips left out: "
                  + ", ".join(f"{k} {v}" for k, v in st["undetermined_without_skips"].items()))
        if st["unreachable_instructions"]:
            print(f"  {stage}: {st['unreachable_instructions']} sampling instructions in unreachable code, not counted")
    print(f"user_pointer: largest static S# load offset {t['max_static_offset']:#x}, largest constant added to the "
          f"pointer {t['max_pointer_constant']:#x}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="<work>/shaders/dxbc/disasm")
    ap.add_argument("--json", help="write the full census here")
    ap.add_argument("--jobs", type=shader_ir.positive_int, default=1, help="worker processes (default 1)")
    ap.add_argument("--limit", type=shader_ir.positive_int, help="only the first N listings (sorted by name)")
    ap.add_argument("--expect-none", nargs="*", choices=CLASSES, default=[], metavar="CLASS",
                    help="fail if an instruction falls in one of these classes: " + ", ".join(CLASSES))
    args = ap.parse_args(argv)
    if not os.path.isdir(args.disasm):
        print(f"ERROR: {args.disasm} is not a folder", file=sys.stderr)
        return 2
    try:
        stages = hic.stage_index(args.disasm)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"ERROR: cannot read the carve index next to {args.disasm}: {e}", file=sys.stderr)
        return 2
    out = census(args.disasm, args.limit, args.jobs, stages)
    print_report(out)
    if args.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {args.json}: {e}", file=sys.stderr)
            return 2
    bad = [f"{c}: {out['total']['classes'][c]} instructions" for c in args.expect_none if out["total"]["classes"][c]]
    for b in bad:
        print("MISMATCH", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
