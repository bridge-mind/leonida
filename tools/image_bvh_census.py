r"""Census of the image_bvh_intersect_ray / image_bvh64_intersect_ray call sites (counts only, standard library).

    python -I tools/image_bvh_census.py <work>/shaders/dxbc/disasm [--out image_bvh_census.json]

docs/design.md's Ray tracing row describes one traversal step of AMD's BVH format, and the translator emulates it;
this tool counts how the game's shaders call it. For every listing that holds an `image_bvh*` instruction it reports:

* the sites per form (`image_bvh_intersect_ray`, the 32-bit node id, against `image_bvh64_intersect_ray`);
* the ray-tracing resource descriptor (the four SGPRs of the call's third operand) decoded into the fields of PAL's
  `sq_bvh_rsrc_t`: box_grow (bits 55..62), box_sort_en (63), size (64..105), triangle_return_mode (120),
  llc_noalloc (121..122), big_page (123) and type (124..127). Word 0 holds base bits [31:0] and is reported only as
  a literal or not; base itself is not decoded. A word counts as known when every path to the call sets its SGPR
  from a literal (`s_mov_b32`, `s_movk_i32`, `s_brev_b32`, an inline `s_mov_b64`, or copies of those);
* how each of the four result words is read.

How a read is found: the listing is loaded as `tools/rdna2spirv.py`'s `translate()` loads it (encodings re-anchored
when the listing holds data words, far branches folded, reachable code only) and split into basic blocks at
branches. The descriptor words come from a pass over the blocks from the entry. The result words come from one
forward walk per call, which starts at the call and asks, for the lanes that ran it, which definitions each VGPR can
hold. The call's k-th destination register holds word k. A plain `v_mov_b32` copies a register's definitions, and
any other write replaces them with a definition that is not a word. Partial writes (an SDWA byte or word insert with
UNUSED_PRESERVE, `v_writelane_b32`, a d16 load) keep the old definitions as well. Joins take the union. The walk
does not follow an edge back to the call or to code before it. The next pass of a loop around the call is a new
call, and a site whose words are still live on such an edge is counted under limits. An edge back to code that lies
inside the walk (an inner loop) is followed, and it drops every lane proof below.

EXEC decides how much of a register a write replaces. The walk names each EXEC value. The root is the call's lanes.
Under it come a narrowing made by one instruction (`v_cmpx_*`, `s_and_saveexec_*`, `s_and_*` or `s_andn2_*` of EXEC)
and the else-half of a saved mask (`s_andn2_* exec, saved, exec`, `s_andn2_wrexec_*`). A mask saved with
`s_mov_b32 sX, exec` or a saveexec is restored by `s_mov_b32 exec, sX`. Any other EXEC write starts a value the walk
cannot place. A write under the root replaces the register. A write under a narrower value adds its definition for
those lanes and keeps the old one for every other lane. A read does not see a definition in only two cases: every lane
of the read ran an overwrite since (the read runs inside the same narrowing as the write), or the definition was made
only in lanes the read cannot run (the read is under the else-half of a mask whose then-half made it). A join whose two
paths have different EXEC values starts a new value, which lies within their nearest common one. Both proofs are
about one run of the instructions involved. Where the walk cannot give one (an EXEC write it cannot place, two values
with no complement relation, an inner back edge), both definitions stay visible, so the lane rule can only move a
read from "some paths" to "every path" by proving it.

An instruction that reads a register holding a word is one read of that word. The read is on "every path" when
nothing but words can reach the register (and, for the ratio, the reciprocal operand holds only the reciprocal), and
on "some paths" otherwise: a lane merge the rule above cannot resolve, a register that one path overwrites, or a path
the program never takes. Each read gets exactly one category:

* ratio: the word divided by word 1 of the same call, as the hardware's numerator/denominator pairs are meant to be
  used: word 1 into `v_rcp_f32`, or a word times that reciprocal in `v_mul_f32` / `v_mul_legacy_f32`, with no
  abs/neg modifier on either operand;
* high_half: word 1 read as its sign-extended high half, `sext(w1[31:16])`, the SDWA `src0_sel:WORD_1` operand of a
  `v_mul_i32_i24`. This is the claim's "sign only" form. The category records that shape and does not prove that only
  the sign survives: `high_half_product_readers` lists what reads the product (a signed compare against 0 or -1
  depends on the sign and on the high half being 0);
* raw_integer: read as integer bits by a VALU op, without the ratio: bitwise, shift, integer arithmetic, an integer
  compare, a conversion from integer, or an SDWA field that is not the high-half product;
* raw_float: read as a float operand (the opcode's source type is f16/f32/f64), without the ratio;
* raw_other: any other read: a load/store/atomic address or data, a lane move (`v_readlane_b32`,
  `v_readfirstlane_b32`, DPP, permlane), another `image_bvh*` operand;
* select (unclassified): the word is a data operand of `v_cndmask_b32`. The selected value is not followed.

Each category is counted as [sites with a read of it on every path, sites with reads of it on some paths only].
`site_partition` puts each site into one bucket: its every-path categories, then any category it reads on some paths
only. The buckets add up to the site total, and "none" means no every-path read. The words are followed through
registers only. A store counts as raw_other, and a value reloaded after it does not count as a word. Every count is
of static candidates: a read on a path the program never takes counts the same as one on a path it always takes.

Standard library only. Nothing from the game is printed or written: counts, mnemonics and descriptor field values.
Usage: `python -I tools/image_bvh_census.py --help` lists the options.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import rdna2spirv  # noqa: E402  (base_op, BRANCHES, fold_far_branches, reachable_code, native_blob, stage_of)
from shader_ir import parse_file  # noqa: E402
from wave64_graphics_scan import scanned_paths  # noqa: E402

FORMS = ("image_bvh_intersect_ray", "image_bvh64_intersect_ray")
CATEGORIES = ("ratio", "high_half", "raw_integer", "raw_float", "raw_other", "select")
OTHER = ("o",)                          # a definition that is not a result word (or the shader's entry)
OTHER_ENTRIES = frozenset({(OTHER, frozenset())})
R1 = ("r1",)                            # v_rcp_f32 of word 1
H = ("h",)                              # v_mul_i32_i24 of sext(word 1 [31:16])
FIELDS = (                              # (name, first bit, width) in the 128-bit descriptor, PAL sq_bvh_rsrc_t
    ("box_grow", 55, 8), ("box_sort_en", 63, 1), ("size", 64, 42), ("triangle_return_mode", 120, 1),
    ("llc_noalloc", 121, 2), ("big_page", 123, 1), ("type", 124, 4))
RATIO_MULS = ("v_mul_f32", "v_mul_legacy_f32")
MEMORY = ("buffer_", "tbuffer_", "global_", "flat_", "scratch_", "ds_", "image_", "s_buffer_", "s_load")
LANE_OPS = ("v_readlane_b32", "v_readfirstlane_b32", "v_writelane_b32", "v_permlane16_b32", "v_permlanex16_b32")
ACC_DST = ("v_fmac_f32", "v_mac_f32", "v_mac_legacy_f32", "v_fmac_f16", "v_dot4c_i32_i8", "v_dot2c_i32_i16",
           "v_dot8c_i32_i4", "v_pk_fmac_f16", "v_fmac_legacy_f32")
NO_VGPR_DST = ("exp", "buffer_store", "tbuffer_store", "image_store", "v_cmp", "ds_write", "global_store",
               "scratch_store", "flat_store")
LDS_RETURNING = ("ds_read", "ds_swizzle", "ds_permute", "ds_bpermute", "ds_append", "ds_consume", "ds_wrxchg")
SALU_NO_DST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_store", "s_buffer_store", "s_waitcnt", "s_nop",
               "s_endpgm", "s_setpc", "s_barrier", "s_sendmsg", "s_dcache", "s_icache", "s_set", "s_sleep",
               "s_ttrace", "s_inst_prefetch", "s_clause", "s_setprio", "s_trap", "s_wakeup", "s_round_mode",
               "s_denorm_mode")
FLOAT_TYPES = ("f16", "f32", "f64")
LIMITS = ("sites the entry does not reach", "sites with a word in a v_cndmask_b32 (not followed)",
          "sites with a category read on some paths only", "sites with a word live on a back edge to the call "
          "(not followed)", "sites with no classified read on every path", "sites whose words are never read",
          "shaders with s_setpc_b64 or s_swappc_b64 left", "shaders with relative VGPR addressing")


class Skip(Exception):
    """A listing the census cannot follow; the reason is counted, not fatal."""


# ---- loading --------------------------------------------------------------------------------------------------

def load_program(path):
    """The reachable instructions of a listing, as rdna2spirv.translate() takes them (no SPIR-V is built)."""
    info = parse_file(path)
    insns = info["insns"]
    if not insns:
        raise Skip("no instructions")
    if any(not i.get("enc") for i in insns):
        raise Skip("instruction without encoding bytes")      # pcs (and so branch targets) are unknown
    if info["invalid"]:
        try:
            blob = rdna2spirv.native_blob(path, info["name"])
        except OSError:
            raise Skip("data words without the native blob to re-anchor them") from None
        pos = info["code_start"]
        for ins in insns:
            enc = bytes.fromhex(ins["enc"])
            at, skipped = pos, 0
            while not rdna2spirv.same_encoding(blob[at:at + len(enc)], enc, ins["op"], len(ins["args"])):
                at += 4
                skipped += 4
                if skipped > 4096 or at >= len(blob):
                    at = None
                    break
            if at is None:
                if pos + len(enc) > len(blob):
                    raise Skip("undecodable words (cannot re-anchor)")
                at = pos
                ins["normalized"] = True
            ins["pc"] = at
            pos = at + len(enc)
        insns = rdna2spirv.fold_far_branches(insns)
        try:
            insns = rdna2spirv.reachable_code(insns)
        except rdna2spirv.Unsupported:
            raise Skip("branch into data") from None
        if any(i.get("normalized") for i in insns):
            raise Skip("reachable instruction with normalized encoding")
    else:
        insns = rdna2spirv.fold_far_branches(insns)
    return info["name"], insns


def blocks_of(insns):
    """Basic blocks and successors: leaders at branch targets and after branches, s_endpgm and s_setpc_b64."""
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        if "target" in ins:
            return ins["target"]
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        if imm is None or not ins.get("enc"):
            raise Skip("branch without offset or encoding")
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    pcs = {i["pc"] for i in insns}
    leaders = {insns[0]["pc"]}
    for ins in insns:
        op = rdna2spirv.base_op(ins["op"])
        if op in rdna2spirv.BRANCHES:
            if target(ins) not in pcs:
                raise Skip("branch target outside code")
            leaders.update((target(ins), next_pc(ins)))
        elif op in ("s_endpgm", "s_setpc_b64"):
            leaders.add(next_pc(ins))
    starts = sorted(p for p in leaders if p in pcs)
    index = {p: n for n, p in enumerate(starts)}
    blocks = [[] for _ in starts]
    cur = 0
    for ins in insns:
        cur = index.get(ins["pc"], cur)
        blocks[cur].append(ins)
    succ = []
    for n, blk in enumerate(blocks):
        last = blk[-1]
        op = rdna2spirv.base_op(last["op"])
        fall = [n + 1] if n + 1 < len(blocks) else []
        if op in ("s_endpgm", "s_setpc_b64"):
            succ.append([])
        elif op == "s_branch":
            succ.append([index[target(last)]])
        elif op in rdna2spirv.BRANCHES:
            succ.append(list(dict.fromkeys([index[target(last)]] + fall)))
        else:
            succ.append(fall)
    return blocks, succ


# ---- operands -------------------------------------------------------------------------------------------------

def vregs(a):
    if a["k"] == "vgpr":
        return list(range(a["r"], a["r"] + a.get("n", 1)))
    if a["k"] == "vlist":
        return list(a["r"])
    return []


def vgpr_access(ins):
    """(read operands as [(argument index, registers)], written registers, partial write?) by the conventions of
    rdna2spirv.reg_reads_writes, plus atomics and LDS ops that return nothing and the partial writes."""
    op, a, mods = ins["op"], ins["args"], ins.get("mods", [])
    if not a:
        return [], [], False
    bop = rdna2spirv.base_op(op)
    no_dst = op.startswith(NO_VGPR_DST) or op.startswith("v_cmpx")
    if bop.startswith(("global_atomic_", "flat_atomic_", "buffer_atomic_", "image_atomic_")) and "glc" not in mods:
        no_dst = True
    if bop.startswith("ds_") and not (bop.startswith(LDS_RETURNING) or "_rtn_" in bop):
        no_dst = True
    if op.startswith("s_"):
        return [], [], False
    sources = [n for n in range(len(a)) if no_dst or n > 0]
    reads = [(n, vregs(a[n])) for n in sources if vregs(a[n])]
    partial = rdna2spirv.sdwa_reads_dst(op, mods) or bop == "v_writelane_b32" or \
        ("_d16" in bop and bop.startswith(MEMORY))
    if (partial or bop in ACC_DST) and not no_dst and vregs(a[0]):
        reads.append((0, vregs(a[0])))
    writes = [] if no_dst else vregs(a[0])
    if bop == "v_swap_b32" and len(a) > 1:
        writes = vregs(a[0]) + vregs(a[1])
    return reads, writes, partial


def source_args(ins):
    """Indexes of the source operands (everything after the destination, or every operand of a store/compare)."""
    op, a = ins["op"], ins["args"]
    if op.startswith("v_cmpx"):
        return list(range(len(a)))
    if op.startswith(NO_VGPR_DST) and not op.startswith("v_cmp"):
        return list(range(len(a)))
    return list(range(1, len(a)))              # a v_cmp writes VCC or an SGPR pair first


def source_is_float(bop):
    """Whether the opcode reads its sources as floats: the last type in v_cvt_*, else any type in the name."""
    parts = bop.split("_")
    if bop.startswith("v_cvt_"):
        return parts[-1] in FLOAT_TYPES
    return any(p in FLOAT_TYPES for p in parts[1:])


def plain(a):
    return not a.get("neg") and not a.get("abs") and not a.get("sext")


def is_copy(ins):
    """A v_mov_b32 / v_swap_b32 that moves the bits unchanged: no modifier, no SDWA select, no DPP."""
    bop = rdna2spirv.base_op(ins["op"])
    a = ins["args"]
    if bop not in ("v_mov_b32", "v_swap_b32") or ins["op"].endswith(("_sdwa", "_dpp")) or len(a) < 2:
        return False
    return not ins.get("mods") and plain(a[1]) and bool(vregs(a[1]))


def high_half_operand(ins, n):
    """Operand n of a v_mul_i32_i24 is sext(src[31:16]) (SDWA src<n-1>_sel:WORD_1 with sext)."""
    if rdna2spirv.base_op(ins["op"]) != "v_mul_i32_i24" or not ins["op"].endswith("_sdwa") or n not in (1, 2):
        return False
    return ins["args"][n].get("sext") and f"src{n - 1}_sel:WORD_1" in ins.get("mods", [])


def kind_of(ins, n):
    """raw_* category of a read of operand n that is neither ratio nor high_half."""
    op = ins["op"]
    bop = rdna2spirv.base_op(op)
    if bop.startswith(MEMORY) or bop.startswith("image_bvh") or bop in LANE_OPS or op.endswith("_dpp") or \
            any(m.startswith(("quad_perm:", "row_", "dpp8:")) for m in ins.get("mods", [])) or \
            not bop.startswith("v_"):
        return "raw_other"
    return "raw_float" if source_is_float(bop) else "raw_integer"


# ---- SGPR constants (descriptor words) ----------------------------------------------------------------------

def sregs(a):
    return list(range(a["r"], a["r"] + a.get("n", 1))) if a["k"] == "sgpr" else []


def imm32(a):
    if a["k"] == "imm" and not a.get("neg") and not a.get("abs"):
        return a["v"] & 0xFFFFFFFF
    if a["k"] == "fimm" and not a.get("neg") and not a.get("abs"):
        import struct
        return struct.unpack("<I", struct.pack("<f", a["v"]))[0]
    return None


def sgpr_step(sg, ins):
    """Update the SGPR literal map for one instruction (written SGPRs lose their value unless set from a literal)."""
    op, a = ins["op"], ins["args"]
    bop = rdna2spirv.base_op(op)
    written = []
    if a and not op.startswith(SALU_NO_DST):
        written += sregs(a[0])              # SALU destinations, v_cmp_*_e64 / v_readlane_b32 SGPR destinations
    if len(a) > 1 and ("_co_" in bop or bop.startswith("v_div_scale")):
        written += sregs(a[1])              # the carry / scale SGPR of a VOP3b op
    value = {}
    if bop in ("s_mov_b32", "s_movk_i32", "s_brev_b32") and len(a) == 2 and len(sregs(a[0])) == 1:
        v = imm32(a[1])
        if v is not None:
            if bop == "s_movk_i32":
                v = (v & 0xFFFF) - (0x10000 if v & 0x8000 else 0) & 0xFFFFFFFF
            elif bop == "s_brev_b32":
                v = int(f"{v:032b}"[::-1], 2)
            value[a[0]["r"]] = v
        elif bop == "s_mov_b32" and len(sregs(a[1])) == 1 and a[1]["r"] in sg and plain(a[1]):
            value[a[0]["r"]] = sg[a[1]["r"]]
    elif bop == "s_mov_b64" and len(a) == 2 and len(sregs(a[0])) == 2:
        r = a[0]["r"]
        if a[1]["k"] == "imm" and -16 <= a[1]["v"] <= 64:              # inline constant, sign-extended
            v = a[1]["v"] & 0xFFFFFFFFFFFFFFFF
            value = {r: v & 0xFFFFFFFF, r + 1: v >> 32}
        elif len(sregs(a[1])) == 2:
            value = {r + i: sg[a[1]["r"] + i] for i in (0, 1) if a[1]["r"] + i in sg}
    for r in written:
        sg.pop(r, None)
    sg.update(value)


def decode_descriptor(words):
    """{field: value or None} for the descriptor words [w0, w1, w2, w3] (None = not a literal)."""
    out = {}
    for name, lo, width in FIELDS:
        need = range(lo // 32, (lo + width - 1) // 32 + 1)
        if any(words[i] is None for i in need):
            out[name] = None
            continue
        bits = sum(words[i] << (32 * (i - need[0])) for i in need)
        out[name] = (bits >> (lo - 32 * need[0])) & ((1 << width) - 1)
    return out


# ---- the dataflow -----------------------------------------------------------------------------------------------

def join_sgprs(old, inc, at=None):
    """The SGPR literals both maps agree on; None when old is unchanged."""
    if old is None:
        return dict(inc)
    out = {r: v for r, v in old.items() if inc.get(r) == v}
    return out if len(out) != len(old) else None


# ---- EXEC: which lanes a write covers ---------------------------------------------------------------------------
#
# A walk starts at one call, so its lanes are the call's (ROOT). An EXEC value is a node: ROOT, a narrowing of a node
# at one instruction (("n", pc, parent): a subset of the parent), the else-half of a saved mask (("e", pc, parent,
# then): the parent's lanes outside `then`), the value after a join of two paths with different EXEC (("j", block,
# parent): within their nearest common node, or its own root without one), or an EXEC value the walk cannot place
# (("u", pc), its own root). A node stands for one run of its instruction or join: the walk follows no back edge to
# the call, and an inner back edge drops every proof (see strip()).

ROOT = ("root",)


def parent(node):
    return node[2] if node[0] in ("n", "e", "j") else None


def within(node, outer):
    """node's lanes are a subset of outer's: outer is node or one of its ancestors."""
    while node is not None:
        if node == outer:
            return True
        node = parent(node)
    return False


def disjoint(a, b):
    """No lane in both: one of them sits under the else-half of a mask whose then-half holds the other."""
    for x, y in ((a, b), (b, a)):
        node = x
        while node is not None:
            if node[0] == "e" and within(y, node[3]):
                return True
            node = parent(node)
    return False


def common(a, b):
    """The nearest node both lie within, or None."""
    up = set()
    while a is not None:
        up.add(a)
        a = parent(a)
    while b is not None and b not in up:
        b = parent(b)
    return b


def visible(guard, node):
    """Whether a definition with this guard can be what a lane under EXEC `node` reads."""
    for kind, at in guard:
        if kind == "in" and disjoint(at, node):        # defined only in lanes this read does not run
            return False
        if kind == "out" and within(node, at):          # every lane of this read was overwritten at `at`
            return False
    return True


def live(guard):
    """A guard some lane can satisfy (no "in" a set and "out" a superset, no "in" two disjoint sets)."""
    ins = [at for kind, at in guard if kind == "in"]
    outs = [at for kind, at in guard if kind == "out"]
    return not any(within(i, o) for i in ins for o in outs) and \
        not any(disjoint(i, j) for n, i in enumerate(ins) for j in ins[n + 1:])


def entries_union(a, b):
    """Union of two definition sets, dropping an entry whose guard has a subset with the same label."""
    if b <= a:
        return a
    if a <= b:
        return b
    by_label = defaultdict(set)
    for lab, guard in a | b:
        by_label[lab].add(guard)
    out = []
    for lab, guards in by_label.items():
        if len(guards) == 1:
            out.append((lab, next(iter(guards))))
            continue
        out += [(lab, g) for g in guards if not any(h < g for h in guards)]
    return frozenset(out)


def labels(entries, node):
    """The labels a read under `node` can see."""
    return {lab for lab, guard in entries if visible(guard, node)}


def mask_writes(ins):
    """SGPRs an instruction writes (destinations, VOP3b carry/scale outputs)."""
    a, op = ins["args"], ins["op"]
    out = []
    if a and not op.startswith(SALU_NO_DST):
        out += sregs(a[0])
    if len(a) > 1 and ("_co_" in rdna2spirv.base_op(op) or rdna2spirv.base_op(op).startswith("v_div_scale")):
        out += sregs(a[1])
    return out


def is_exec(a):
    return a["k"] == "special" and a.get("t", "").startswith("exec")


def exec_step(st, ins):
    """Apply one instruction to the EXEC node and the saved-mask SGPRs (st.masks: SGPR -> node)."""
    op, a, pc = ins["op"], ins["args"], ins["pc"]
    bop = rdna2spirv.base_op(op)
    cur = st.exec
    saved = lambda x: st.masks.get(x["r"]) if x["k"] == "sgpr" else None  # noqa: E731
    new_masks = {}
    if bop.startswith("v_cmpx"):
        new = ("n", pc, cur)
    elif "saveexec" in bop and a:
        new_masks = {r: cur for r in sregs(a[0])}
        if bop.startswith(("s_and_saveexec", "s_andn1_saveexec")):
            new = ("n", pc, cur)
        elif bop.startswith("s_or_saveexec") and len(a) > 1 and imm32(a[1]) == 0xFFFFFFFF:
            new = ROOT                                  # every lane, so every lane of the call
        else:
            new = ("u", pc)
    elif bop.startswith("s_andn2_wrexec") and len(a) > 1 and saved(a[1]):
        new = ("e", pc, saved(a[1]), cur)               # EXEC = saved & ~EXEC: the else-half
        new_masks = {r: new for r in sregs(a[0])}
    elif not rdna2spirv.writes_exec(ins):
        new = None
    elif bop in ("s_mov_b32", "s_mov_b64") and len(a) == 2:
        new = saved(a[1]) or (ROOT if imm32(a[1]) == 0xFFFFFFFF else ("u", pc))
    elif bop in ("s_and_b32", "s_and_b64") and len(a) == 3:
        if is_exec(a[1]) or is_exec(a[2]):
            new = ("n", pc, cur)
        elif saved(a[1]) or saved(a[2]):
            new = ("n", pc, saved(a[1]) or saved(a[2]))
        else:
            new = ("u", pc)
    elif bop in ("s_andn2_b32", "s_andn2_b64") and len(a) == 3:
        if is_exec(a[1]):
            new = ("n", pc, cur)                        # EXEC & ~x
        elif is_exec(a[2]) and saved(a[1]):
            new = ("e", pc, saved(a[1]), cur)
        else:
            new = ("u", pc)
    else:
        new = ("u", pc)
    for r in mask_writes(ins):
        st.masks.pop(r, None)
    if bop in ("s_mov_b32", "s_mov_b64") and len(a) == 2 and is_exec(a[1]) and not is_exec(a[0]):
        new_masks = {r: cur for r in sregs(a[0])}
    st.masks.update(new_masks)
    if new is not None:
        st.exec = new


class SiteState:
    """One walk's state: VGPR -> definitions [(label, guard)], saved-mask SGPRs, EXEC node."""
    __slots__ = ("vg", "masks", "exec")

    def __init__(self, vg=None, masks=None, ex=ROOT):
        self.vg, self.masks, self.exec = vg or {}, masks or {}, ex

    def copy(self):
        return SiteState(dict(self.vg), dict(self.masks), self.exec)

    def idle(self):
        """No register can hold a word, its reciprocal or its high-half product any more."""
        return all(lab == OTHER for s in self.vg.values() for lab, _ in s)

    def read(self, r):
        return labels(self.vg.get(r, OTHER_ENTRIES), self.exec)

    def write(self, r, new, partial=False):
        """new: definitions [(label, guard)] the active lanes now hold."""
        node = self.exec
        old = self.vg.get(r, OTHER_ENTRIES)
        if node == ROOT and not partial:
            out = frozenset(new)
        else:
            mine = frozenset((lab, guard | {("in", node)}) for lab, guard in new)
            rest = old if partial else frozenset((lab, guard | {("out", node)}) for lab, guard in old)
            out = entries_union(frozenset(e for e in mine if live(e[1])), frozenset(e for e in rest if live(e[1])))
        if out == OTHER_ENTRIES:
            self.vg.pop(r, None)
        else:
            self.vg[r] = out


def join_site(old, inc, at=None):
    """Join of two walk states at the block starting at pc `at`: definitions united, masks kept where equal, and
    two different EXEC values replaced by a new one, ("j", at, nearest common value), that lies within both
    paths' common value."""
    if old is None:
        return inc.copy()
    vg, changed = dict(old.vg), False
    for r in set(old.vg) | set(inc.vg):
        a, b = old.vg.get(r, OTHER_ENTRIES), inc.vg.get(r, OTHER_ENTRIES)
        u = entries_union(a, b)
        if u != a:
            vg[r], changed = u, True
    masks = {r: v for r, v in old.masks.items() if inc.masks.get(r) == v}
    changed = changed or len(masks) != len(old.masks)
    ex = old.exec
    if inc.exec != old.exec:
        ex = ("j", at, common(old.exec, inc.exec))
        changed = changed or ex != old.exec
    return SiteState(vg, masks, ex) if changed else None


def strip(st):
    """Taken an inner back edge: the next pass of an instruction narrows to other lanes, so drop every guard."""
    out = st.copy()
    out.masks = {}                          # a mask saved in this pass must not place a read in the next one
    out.vg = {r: frozenset((lab, frozenset()) for lab, _ in s) for r, s in st.vg.items()}
    out.vg = {r: s for r, s in out.vg.items() if s != OTHER_ENTRIES}
    return out


def flow(blocks, succ, start, transfer, join, initial, copy=dict, cut=None, on_back=None, idle=None):
    """Forward dataflow from (block, instruction index) `start` to a fixpoint. transfer(state, ins) updates a state
    in place; join(old, new) returns the joined state or None when old already covers new. An edge to a block that
    starts at or before pc `cut` is not followed, and the states that reach one are returned. Any other edge to a
    block at or before the end of the one it leaves (an inner back edge) passes its state through on_back first.
    A state for which idle(state) holds is not sent on (nothing it could reach would record anything)."""
    b0, i0 = start
    state_in = [None] * len(blocks)
    work, at_cut = set(), []

    def send(b, state):
        if idle is not None and idle(state):
            return
        for s in succ[b]:
            if cut is not None and blocks[s][0]["pc"] <= cut:
                at_cut.append(state)
                continue
            out = on_back(state) if on_back and blocks[s][0]["pc"] <= blocks[b][-1]["pc"] else state
            merged = join(state_in[s], out, blocks[s][0]["pc"])
            if merged is not None:
                state_in[s] = merged
                work.add(s)
    if i0 == 0 and cut is None:
        state_in[b0] = join(None, initial, blocks[b0][0]["pc"])
        work.add(b0)
    else:                                   # the rest of the start block; a later visit runs all of it
        state = copy(initial)
        for ins in blocks[b0][i0:]:
            transfer(state, ins)
        send(b0, state)
    while work:
        b = min(work)                       # program order converges in fewer passes than LIFO
        work.discard(b)
        state = copy(state_in[b])
        for ins in blocks[b]:
            transfer(state, ins)
        send(b, state)
    return at_cut


def ratio_partner(st, other):
    """"all" when the other multiplicand holds the reciprocal of word 1 on every path, "some" when only on some,
    "none" otherwise."""
    regs = vregs(other)
    s = st.read(regs[0]) if len(regs) == 1 else {OTHER}
    if R1 not in s:
        return "none"
    return "all" if s == {R1} else "some"


def derived(s, label):
    """The labels of a value computed from word 1: `label` when word 1 can reach, plus OTHER when anything else
    can."""
    out = {label} if ("w", 1) in s else set()
    if s - {("w", 1)}:
        out.add(OTHER)
    return out


def product_reader(ins, n):
    """How a high-half product is read: the mnemonic and what the other source is (0, -1, a register...)."""
    others = [m for m in source_args(ins) if m != n]
    if len(others) != 1:
        what = f"{len(others)} other operands"
    else:
        const = imm32(ins["args"][others[0]])
        what = {0: "0", 0xFFFFFFFF: "-1"}.get(const, "a register" if const is None else "a constant")
    return f"{rdna2spirv.base_op(ins['op'])} with {what}"


def classify(st, ins, n, words):
    """(category, reach) of a read of `words` (labels one register can hold) through operand n."""
    bop = rdna2spirv.base_op(ins["op"])
    a = ins["args"]
    only_w1 = all(w[1] == 1 for w in words)
    if bop.startswith("v_cndmask_b32") and n in (1, 2):
        return "select", "all"
    if bop == "v_rcp_f32" and n == 1 and only_w1 and plain(a[1]) and not ins.get("mods"):
        return "ratio", "all"
    if bop in RATIO_MULS and n in (1, 2) and len(a) == 3 and plain(a[1]) and plain(a[2]) and not ins.get("mods"):
        partner = ratio_partner(st, a[3 - n])
        if partner != "none":
            return "ratio", partner
    if high_half_operand(ins, n) and only_w1:
        return "high_half", "all"
    return kind_of(ins, n), "all"


def site_transfer(call_pc, events, product_reads):
    """The transfer function of one call's walk: record the reads of its words, then apply the writes."""
    def transfer(st, ins):
        a = ins["args"]
        bop = rdna2spirv.base_op(ins["op"])
        pc = ins["pc"]
        reads, writes, partial = vgpr_access(ins)
        copy = is_copy(ins)
        ev, pr = [], []
        for n, regs in reads if not copy else ():
            for r in regs:
                s = st.read(r)
                if H in s:
                    pr.append((product_reader(ins, n), "all" if s == {H} else "some"))
                words = sorted(x for x in s if x[0] == "w")
                if not words:
                    continue
                cat, reach = classify(st, ins, n, words)
                if len(words) != len(s):
                    reach = "some"
                ev += [(w[1], cat, reach, bop) for w in words]
        events[pc] = ev
        product_reads[pc] = pr
        if pc == call_pc:
            new = {r: {("w", k)} for k, r in enumerate(writes[:4])}
        elif copy:
            new = {r: st.read(a[1]["r"]) for r in writes[:1]}
            if bop == "v_swap_b32":
                new[a[1]["r"]] = st.read(a[0]["r"])
        elif bop == "v_rcp_f32" and writes and len(a) > 1 and len(vregs(a[1])) == 1:
            new = {writes[0]: derived(st.read(a[1]["r"]), R1)}
        elif writes and any(high_half_operand(ins, n) for n in (1, 2)):
            n = 1 if high_half_operand(ins, 1) else 2
            new = {writes[0]: derived(st.read(a[n]["r"]), H)}
        else:
            new = {r: {OTHER} for r in writes}
        for r, labs in new.items():
            st.write(r, [(lab, frozenset()) for lab in labs], partial)
        exec_step(st, ins)
    return transfer


def descriptor_words(blocks, succ, calls):
    """{call pc: [w0..w3 or None]}: the SGPR literals in force at each call, from the entry."""
    found = {}

    def transfer(sg, ins):
        if ins["pc"] in calls:
            a = ins["args"]
            r = a[2]["r"] if len(a) > 2 and a[2]["k"] == "sgpr" else None
            old = found.get(ins["pc"])
            now = [sg.get(r + i) if r is not None else None for i in range(4)]
            found[ins["pc"]] = now if old is None else [x if x == y else None for x, y in zip(old, now)]
        sgpr_step(sg, ins)
    flow(blocks, succ, (0, 0), transfer, join_sgprs, {})
    return found


def analyse(insns):
    """Per-site records of one program: form, descriptor words, and the reads of its words with their pcs.

    Returns [{"form", "pc", "reached", "desc": [w0..w3 or None],
    "reads": [(pc, word, category, reach, mnemonic, past a later call)], "product_reads": [(pc, reader, reach)]}] in
    program order; reach is "all" or "some" (paths), and the last field says whether the read lies after another
    call in program order. The pcs are for local inspection; summarise() drops them."""
    blocks, succ = blocks_of(insns)
    where = {i["pc"]: (b, n) for b, blk in enumerate(blocks) for n, i in enumerate(blk)}
    calls = [i for i in insns if rdna2spirv.base_op(i["op"]) in FORMS]
    desc = descriptor_words(blocks, succ, {c["pc"] for c in calls})
    sites = []
    for c in calls:
        events, product_reads = {}, {}
        at_cut = flow(blocks, succ, where[c["pc"]], site_transfer(c["pc"], events, product_reads), join_site,
                      SiteState(), copy=SiteState.copy, cut=c["pc"], on_back=strip, idle=SiteState.idle)
        sites.append({"form": rdna2spirv.base_op(c["op"]), "pc": c["pc"], "reached": c["pc"] in desc,
                      "loop_carried": any(lab[0] == "w" for st in at_cut for s in st.vg.values() for lab, _ in s),
                      "desc": desc.get(c["pc"], [None] * 4),
                      "reads": [(pc,) + e + (any(c["pc"] < x["pc"] < pc for x in calls),)
                                for pc in sorted(events) for e in events[pc]],
                      "product_reads": [(pc,) + e for pc in sorted(product_reads) for e in product_reads[pc]]})
    return sites


# ---- corpus -----------------------------------------------------------------------------------------------------

def scan_listing(path):
    """{"stage", "sites"} for a listing with an image_bvh* instruction, None without one; raises Skip."""
    with open(path, encoding="utf-8", errors="replace") as f:
        if "image_bvh" not in f.read():
            return None
    name, insns = load_program(path)
    if not any(rdna2spirv.base_op(i["op"]) in FORMS for i in insns):
        return None
    return {"stage": rdna2spirv.stage_of(name, path), "sites": analyse(insns),
            "setpc": any(rdna2spirv.base_op(i["op"]) in ("s_setpc_b64", "s_swappc_b64") for i in insns),
            "movrel": any(rdna2spirv.base_op(i["op"]).startswith(("v_movrel", "s_set_gpr_idx")) for i in insns)}


def scan_directory(disasm):
    """(listings scanned, per-listing records for those with sites, Counter of skip reasons)."""
    paths = scanned_paths(disasm)
    records, skipped = [], Counter()
    for path in paths:
        try:
            r = scan_listing(path)
        except Skip as e:
            skipped[str(e)] += 1
            continue
        if r is not None:
            records.append(r)
    return len(paths), records, skipped


def hexs(v):
    return str(v) if v < 10 else hex(v)


def site_view(s):
    """(categories read on every path, categories read only on some paths) of one site record."""
    every = {e[2] for e in s["reads"] if e[3] == "all"}
    return every, {e[2] for e in s["reads"]} - every


def split_count(counter, items, every_items):
    """Add 1 to counter[x][0] for x read on every path, to counter[x][1] for x read on some paths only."""
    for x in every_items:
        counter[x][0] += 1
    for x in set(items) - set(every_items):
        counter[x][1] += 1


def pairs():
    return defaultdict(lambda: [0, 0])


def ordered(d):
    def weight(v):
        return -sum(v) if isinstance(v, list) else -v
    return {k: v for k, v in sorted(d.items(), key=lambda kv: (weight(kv[1]), kv[0]))}


def summarise(listings, records, skipped):
    """Counts only, no names: the JSON report."""
    forms, stages, word0 = Counter(), Counter(), Counter()
    limits = Counter({k: 0 for k in LIMITS})
    fields = {name: Counter() for name, _, _ in FIELDS}
    cat_sites, cat_shaders, cat_reads, some_reads = pairs(), Counter(), Counter(), Counter()
    per_word = {str(k): pairs() for k in range(4)}
    partition, partition_shaders = Counter(), defaultdict(set)
    mnemonics = {c: pairs() for c in CATEGORIES}
    products = pairs()
    position = pairs()
    total = 0
    for n, r in enumerate(records):
        stages[r["stage"]] += 1
        in_shader = set()
        for s in r["sites"]:
            total += 1
            forms[s["form"]] += 1
            if not s["reached"]:
                limits["sites the entry does not reach"] += 1
            w = s["desc"]
            word0["literal" if w[0] is not None else "not a literal"] += 1
            for name, v in decode_descriptor(w).items():
                fields[name]["not a literal" if v is None else hexs(v)] += 1
            reads = s["reads"]
            every, some = site_view(s)
            split_count(cat_sites, every | some, every)
            in_shader |= every | some
            for e in reads:
                cat_reads[e[2]] += 1
                some_reads[e[2]] += e[3] == "some"
            for word in range(4):
                mine = [e for e in reads if e[1] == word]
                split_count(per_word[str(word)], {e[2] for e in mine}, {e[2] for e in mine if e[3] == "all"})
            for c in CATEGORIES:
                mine = [e for e in reads if e[2] == c]
                split_count(mnemonics[c], {e[4] for e in mine}, {e[4] for e in mine if e[3] == "all"})
            partial = [e for e in reads if e[3] == "some"]
            for c in {e[2] for e in partial}:
                position[c][0 if any(not e[5] for e in partial if e[2] == c) else 1] += 1
            split_count(products, {p[1] for p in s["product_reads"]},
                        {p[1] for p in s["product_reads"] if p[2] == "all"})
            key = "+".join(c for c in CATEGORIES if c in every) or "none"
            if some:
                key += " (some paths only: " + "+".join(c for c in CATEGORIES if c in some) + ")"
            partition[key] += 1
            partition_shaders[key].add(n)
            if "select" in every | some:
                limits["sites with a word in a v_cndmask_b32 (not followed)"] += 1
            if some:
                limits["sites with a category read on some paths only"] += 1
            if s["loop_carried"]:
                limits["sites with a word live on a back edge to the call (not followed)"] += 1
            if reads and not every - {"select"}:
                limits["sites with no classified read on every path"] += 1
            if not reads:
                limits["sites whose words are never read"] += 1
        for c in in_shader:
            cat_shaders[c] += 1
    limits["shaders with s_setpc_b64 or s_swappc_b64 left"] = sum(1 for r in records if r["setpc"])
    limits["shaders with relative VGPR addressing"] = sum(1 for r in records if r["movrel"])
    return {
        "listings": listings,
        "not_analysed": dict(sorted(skipped.items())),
        "shaders": dict(sorted(stages.items())),
        "sites": {"total": total, **{f: forms[f] for f in FORMS}},
        "descriptor": {"word0": dict(sorted(word0.items())),
                       "fields": {k: dict(sorted(v.items())) for k, v in fields.items()}},
        "word_reads": {"pairs": "[sites read on every path, sites read on some paths only]",
                       "sites": {c: cat_sites[c] for c in CATEGORIES},
                       "shaders": {c: cat_shaders[c] for c in CATEGORIES},
                       "reads": {c: cat_reads[c] for c in CATEGORIES},
                       "reads_some_paths": {c: some_reads[c] for c in CATEGORIES},
                       "per_word_sites": {k: {c: v[c] for c in CATEGORIES if c in v} for k, v in per_word.items()}},
        "site_partition": {k: [v, len(partition_shaders[k])] for k, v in ordered(partition).items()},
        "mnemonic_sites": {c: ordered(m) for c, m in mnemonics.items() if m},
        "high_half_product_readers": ordered(products),
        "some_paths_position": {"pairs": "[sites with a some-paths read before the next call in program order, "
                                          "sites whose some-paths reads all lie after a later call]",
                                **{c: position[c] for c in CATEGORIES if c in position}},
        "limits": dict(sorted(limits.items())),
    }


def print_summary(s, out=None):
    out = out or sys.stdout
    w = lambda text="": print(text, file=out)  # noqa: E731
    st = s["sites"]
    w(f"{st['total']} image_bvh sites in {sum(s['shaders'].values())} shaders ("
      + ", ".join(f"{k} {v}" for k, v in s["shaders"].items()) + f"); {s['listings']} listings scanned"
      + (f"; not analysed {s['not_analysed']}" if s["not_analysed"] else ""))
    w("  " + ", ".join(f"{f} {st[f]}" for f in FORMS))
    d = s["descriptor"]
    w("descriptor word 0: " + ", ".join(f"{k} {v}" for k, v in d["word0"].items()))
    for name, row in d["fields"].items():
        w(f"  {name}: " + ", ".join(f"{k} at {v}" for k, v in row.items()))
    wr = s["word_reads"]
    w("word reads: sites read on every path / on some paths only, shaders, reads")
    for c in CATEGORIES:
        a, b = wr["sites"][c]
        w(f"  {c:<12} {a:5d} / {b:5d}  {wr['shaders'][c]:3d} {wr['reads'][c]:6d}"
          + ("  (unclassified)" if c == "select" else ""))
    for k, row in wr["per_word_sites"].items():
        w(f"  w{k}: " + ", ".join(f"{c} {v[0]}/{v[1]}" for c, v in row.items()))
    w("site partition (sites, shaders):")
    for k, (v, n) in s["site_partition"].items():
        w(f"  {k}: {v}, {n}")
    for c, m in s["mnemonic_sites"].items():
        w(f"{c} mnemonics (sites every/some): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in m.items()))
    if s["high_half_product_readers"]:
        w("high-half product read by (sites every/some): "
          + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in s["high_half_product_readers"].items()))
    w("some-paths reads (sites with one before the next call / sites with all after a later call): "
      + ", ".join(f"{c} {v[0]}/{v[1]}" for c, v in s["some_paths_position"].items() if c != "pairs"))
    w("limits: " + "; ".join(f"{k} {v}" for k, v in s["limits"].items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count image_bvh*_intersect_ray sites, their descriptor fields and how "
                                             "their result words are read (counts only, no names)")
    ap.add_argument("src", help="<work>/shaders/dxbc/disasm from the shaders stage")
    ap.add_argument("--out", help="write the same counts to a JSON file")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.src):
        ap.error(f"not a directory: {args.src}")
    if args.out:
        if os.path.isdir(args.out):
            ap.error(f"{args.out} is a directory, not a JSON file")
        if not os.path.isdir(os.path.dirname(os.path.abspath(args.out))):
            ap.error(f"output directory does not exist: {os.path.dirname(os.path.abspath(args.out))}")
        if os.path.exists(args.out) and any(os.path.samefile(args.out, p) for p in scanned_paths(args.src)):
            ap.error("output must not overwrite a listing it scans")
    summary = summarise(*scan_directory(args.src))
    print_summary(summary)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {args.out}: {e.strerror or e}", file=sys.stderr)
            return 2
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
