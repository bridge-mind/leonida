"""Which scalar registers hold a lane mask?

On hardware a lane mask is one register for the whole wave, a bit per lane. The translator runs one invocation
per lane and keeps only that lane's bit in the register (all ones or zero), which is all that and/or/not, selects
and EXEC need. A few instructions look at the mask as a whole instead: s_bcnt1 (how many lanes), s_ff1 (the first
lane), v_mbcnt (how many lanes below this one). For those the translator has to collect the other lanes' bits.
The same instructions also run on plain 64-bit data, such as a light mask read from a buffer, and nothing in the
instruction says which of the two it has, so this module works it out from what wrote the register.

    masks, writes = analyse(blocks, succ)   # blocks: lists of instructions; succ: successor block numbers per block
    masks[id(ins)]                      # dict mapping each register holding a lane mask on every path to ins
                                        # (SGPR number, "vcc", ("lane", vgpr, lane) parked in a VGPR lane, or
                                        # ("v", vgpr) passing through a VGPR) to its kind (MASK, MASK_LO, MASK_HI)
    id(ins) in writes                   # ins computes a mask into a VGPR (a plain copy does not count)

A register counts as a mask only if every path to the instruction wrote a mask to it. Anything else is treated
as data, which is what the translator did for all of them before.

Parked masks: a shader that runs out of SGPRs moves some into single lanes of a spare VGPR (v_writelane_b32) and
fetches them back later (v_readlane_b32). When the SGPR is a saved EXEC, the value that comes back has to be this
lane's bit again, not the bit of the lane it was parked in.

Masks through a VGPR: the compiler does some scalar work in the vector unit, such as an OR of three SGPRs
(v_mov_b32 / v_or3_b32, then v_readfirstlane_b32 to get the scalar back). On hardware every lane computes the same
whole mask and the first lane's copy is read. Here each lane computes its own bit, so it has to read its own
copy, and it has to compute it whether or not it is active. A plain copy of a mask (v_mov_b32) is different:
it moves an already-computed value, so it is an ordinary VALU write and inactive lanes keep the old VGPR
(SHADER-21 part 2).
"""

MASK = "mask"           # full lane mask (64-bit pair, vcc, or wave32 single register)
MASK_LO = "mask_lo"     # low 32-bit half of a 64-bit lane mask (lanes 0-31)
MASK_HI = "mask_hi"     # high 32-bit half of a 64-bit lane mask (lanes 32-63)
MASK_KINDS = (MASK, MASK_LO, MASK_HI)
EITHER = "either"       # a constant 0 or -1: no lanes or all lanes, and the same bits when read as data
# Absent from the state: data.

COPIES = ("s_mov_b", "s_not_b", "s_wqm_b")
BITWISE = ("s_and_b", "s_or_b", "s_xor_b", "s_andn2_b", "s_orn2_b", "s_nand_b", "s_nor_b", "s_xnor_b",
           "s_cselect_b")
NO_DESTINATION = (                      # their first operand (if any) is a source, not a written register
    "s_cmp", "s_bitcmp", "v_cmpx", "s_set", "s_rfe",
    "s_store_", "s_buffer_store_", "s_dcache_", "s_icache_",
    "s_sendmsg", "s_branch", "s_cbranch", "s_waitcnt", "s_nop", "s_sleep", "s_barrier", "s_endpgm",
    "exp",
    "buffer_store", "tbuffer_store", "global_store", "flat_store", "image_store", "ds_write",
)
DS_HAS_DST = ("ds_read", "ds_swizzle", "ds_permute", "ds_bpermute",
              "ds_append", "ds_consume", "ds_ordered_count", "ds_nop")
VGPR_REL_DST = ("v_movreld_b32", "v_movrelsd_b32", "v_movrelsd_2_b32", "v_swaprel_b32")
SGPR_REL_DST = ("s_movreld_b32", "s_movreld_b64", "s_movrelsd_2_b32")
VECTOR_BITWISE = ("v_mov_b32", "v_or3_b32", "v_or_b32", "v_and_b32", "v_and_or_b32",
                  "v_xor_b32", "v_xor3_b32", "v_xnor_b32", "v_bfi_b32", "v_not_b32")


def base_op(op):
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            return op[:-len(suffix)]
    return op


def kind(state, a):
    if a["k"] == "special":
        t = a.get("t", "")
        if t == "exec_lo":
            return MASK_LO
        if t == "exec_hi":
            return MASK_HI
        if t == "exec":
            return MASK
        if t == "vcc":
            return state.get("vcc")
        if t in ("vcc_lo", "vcc_hi"):
            vk = state.get("vcc")
            if vk in MASK_KINDS:
                return MASK_LO if t == "vcc_lo" else MASK_HI
            return vk if t == "vcc_lo" else None
        return None
    if a["k"] == "sgpr":
        n = a.get("n", 1)
        ks = [state.get(a["r"] + i) for i in range(n)]
        if None in ks:
            return None
        if n == 2 and any(k in MASK_KINDS for k in ks):
            return MASK
        return ks[0] if len(set(ks)) == 1 else None
    if a["k"] == "vgpr" and a.get("n", 1) == 1:
        return state.get(("v", a["r"]))
    if a["k"] == "imm" and a["v"] in (0, -1):
        return EITHER
    return None


def put(state, a, k):
    if a["k"] == "sgpr":
        n = a.get("n", 1)
        if k == MASK and n == 2:
            state[a["r"]] = MASK_LO
            state[a["r"] + 1] = MASK_HI
            return
        regs = [a["r"] + i for i in range(n)]
    elif a["k"] == "slist":
        regs = [r if isinstance(r, int) else "vcc" for r in a["r"]]
    elif a["k"] == "special" and a.get("t") == "vcc":
        regs = ["vcc"]
    elif a["k"] == "special" and a.get("t") in ("vcc_lo", "vcc_hi"):
        # A mask / 0 / -1 half-write updates that half of %vcc in wave64 (put_mask), keeping VCC a lane mask;
        # a plain-data half-write (k is None) ends VCC's life as one lane mask.
        if k is not None and state.get("vcc") in MASK_KINDS:
            state["vcc"] = MASK
        elif a.get("t") == "vcc_lo" and k is not None:
            state["vcc"] = k
        else:
            state.pop("vcc", None)
        return
    else:
        return
    for r in regs:
        if k is None:
            state.pop(r, None)
        else:
            state[r] = k


def _maskable_b32(x):
    return x["k"] in ("sgpr", "imm") or (
        x["k"] == "special" and (x.get("t", "").startswith(("exec", "vcc")) or x.get("t") == "null")
    )


def lane_slot(vgpr, lane):
    """State key for a mask parked in one lane of a VGPR, or None unless both operands are plain."""
    if vgpr["k"] != "vgpr" or lane["k"] != "imm":
        return None
    return ("lane", vgpr["r"], lane["v"] & 63)


def step(state, ins, wave64=True):
    """Update `state` (register -> MASK | MASK_LO | MASK_HI | EITHER) for one instruction. True if it writes a mask to a VGPR.

    `wave64` False: the program runs in 32-lane waves, where a 64-bit s_bfm_b64 result can't be a lane mask."""
    op, a = base_op(ins["op"]), ins["args"]
    if not a or op.startswith(NO_DESTINATION):
        return
    mods = ins.get("mods", ())
    if op.startswith(("buffer_load", "tbuffer_load")) and "lds" in mods:
        return
    if op.startswith(("buffer_atomic", "global_atomic", "flat_atomic", "image_atomic",
                      "s_atomic", "s_buffer_atomic")) and "glc" not in mods:
        return
    if op.startswith("ds_") and not op.startswith(DS_HAS_DST) and "_rtn" not in op:
        return
    if op == "v_readfirstlane_b32" and len(a) == 2:                             # sD = vS, the same in every lane
        put(state, a[0], kind(state, a[1]))
        return
    if op == "v_writelane_b32" and len(a) == 3 and lane_slot(a[0], a[2]):       # vD[lane] = sS
        state.pop(("v", a[0]["r"]), None)
        k = kind(state, a[1])
        if k is None:
            state.pop(lane_slot(a[0], a[2]), None)
        else:
            state[lane_slot(a[0], a[2])] = k
        return
    if op == "v_readlane_b32" and len(a) == 3:                                  # sD = vS[lane]
        put(state, a[0], state.get(lane_slot(a[1], a[2])))
        return
    if op.startswith("v_cmp"):
        # Handle v_cmp before VGPR clobber (defensive for 2-operand forms with implicit vcc).
        put(state, {"k": "special", "t": "vcc"} if len(a) == 2 else a[0], MASK)
        return
    if op == "v_swap_b32":
        written = set()
        for x in a[:2]:
            if x["k"] == "vgpr":
                written.update(range(x["r"], x["r"] + x.get("n", 1)))
        for key in [k for k in state if isinstance(k, tuple) and k[1] in written]:
            del state[key]
        return
    if op in VGPR_REL_DST:
        bases = [x["r"] for x in (a[:2] if op == "v_swaprel_b32" else a[:1]) if x["k"] == "vgpr"]
        if bases:
            lo = min(bases)
            for key in [k for k in state if isinstance(k, tuple) and k[1] >= lo]:
                del state[key]
        return
    if op in SGPR_REL_DST:
        if a[0]["k"] == "sgpr":
            lo = a[0]["r"]
            for key in [k for k in state if isinstance(k, int) and k >= lo]:
                del state[key]
        return
    if a[0]["k"] == "vgpr":                             # any other write to a VGPR ends what was parked in it
        # A mask only when every source is one (or a constant 0 / -1) on every path: no EITHER for VGPRs, whose
        # plain writes skip inactive lanes. The sources are looked at first: the destination can be one of them.
        # SDWA/DPP or modified forms alter sub-dwords or read other lanes, so they are never plain mask ops.
        plain_valu = (op in VECTOR_BITWISE and a[0].get("n", 1) == 1
                      and not ins["op"].endswith(("_sdwa", "_dpp", "_dpp8"))
                      and not mods
                      and not any(x.get("neg") or x.get("abs") or x.get("sext") for x in a[1:]))
        kinds = [kind(state, x) for x in a[1:]] if plain_valu else [None]
        written = range(a[0]["r"], a[0]["r"] + a[0].get("n", 1))
        for key in [k for k in state if isinstance(k, tuple) and k[1] in written]:
            del state[key]
        if None not in kinds and any(k in MASK_KINDS for k in kinds):
            ms = {k for k in kinds if k in MASK_KINDS}
            state[("v", a[0]["r"])] = ms.pop() if len(ms) == 1 else MASK
            # A plain copy (v_mov_b32) is an ordinary VALU write: inactive lanes keep the old
            # VGPR (SHADER-21 part 2). Only per-lane mask computations take the all-lanes store,
            # because whole-wave readers (wave_mask, v_readfirstlane) read every lane's bit.
            return op != "v_mov_b32"
    if "_saveexec_b" in op or "_wrexec_b" in op:
        put(state, a[0], MASK_LO if op.endswith("_b32") else MASK)
    elif op == "s_bfm_b64":
        # In wave32 code it is 64-bit data, e.g. the first two words of an S# built in place (the pipeline library's
        # wave32 compute programs: s_bfm_b64 s[88:89], 2, 54); rdna2spirv computes the value there.
        put(state, a[0], MASK if wave64 else None)
    elif op.startswith(COPIES) and len(a) > 1:
        k = kind(state, a[1])
        dst_t = a[0].get("t") if a[0]["k"] == "special" else None
        if k in (None, EITHER) and op.endswith("b64") and dst_t in ("vcc", "vcc_lo") and a[1]["k"] == "imm":
            k = MASK
        elif op == "s_not_b32" and dst_t in ("vcc", "vcc_lo") and _maskable_b32(a[1]):
            k = MASK
        elif op == "s_not_b32" and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS and _maskable_b32(a[1]):
            return
        put(state, a[0], k)
    elif op.startswith(BITWISE):
        kinds = [kind(state, x) for x in a[1:]]
        ms = {k for k in kinds if k in MASK_KINDS}
        dst_t = a[0].get("t") if a[0]["k"] == "special" else None
        if None not in kinds:
            if op.endswith("b32") and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS:
                return
            fallback = MASK if op.endswith("b64") and dst_t in ("vcc", "vcc_lo") else EITHER
            put(state, a[0], (ms.pop() if len(ms) == 1 else MASK) if ms else fallback)
        elif op.endswith("b64") and (ms or dst_t in ("vcc", "vcc_lo")) and \
                all(k is not None or x["k"] == "imm" for k, x in zip(kinds, a[1:])):
            put(state, a[0], (ms.pop() if len(ms) == 1 else MASK) if ms else MASK)
        elif op.endswith("b32") and dst_t in ("vcc", "vcc_lo") and all(_maskable_b32(x) for x in a[1:]):
            put(state, a[0], MASK)
        elif op.endswith("b32") and dst_t == "vcc_hi" and state.get("vcc") in MASK_KINDS and \
                all(_maskable_b32(x) for x in a[1:]):
            return
        else:
            put(state, a[0], None)
    else:
        put(state, a[0], None)
        if ((op.startswith("v_") and "_co_" in op)
                or op in ("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64")) and len(a) > 1:
            put(state, a[1], MASK)                      # carry/sign/scale lane mask of VOP3b ops


def parked(blocks, masks):
    """The (vgpr, lane) slots some v_readlane_b32 fetches a mask from, given analyse()'s result."""
    found = set()
    for blk in blocks:
        for ins in blk:
            a = ins["args"]
            if base_op(ins["op"]) == "v_readlane_b32" and len(a) == 3:
                slot = lane_slot(a[1], a[2])
                if slot in masks.get(id(ins), ()):
                    found.add(slot[1:])
    return sorted(found)


def analyse(blocks, succ, *, include_either=False, wave64=True):
    entry = [None] * len(blocks)                        # None: not reached yet
    if blocks:
        entry[0] = {}
    work = [0] if blocks else []
    while work:
        n = work.pop()
        state = dict(entry[n])
        for ins in blocks[n]:
            step(state, ins, wave64)
        for s in succ[n]:
            if entry[s] is None:
                merged = dict(state)
            else:                                       # a mask on both paths, or a mask on one and 0 / -1 on the other
                merged = {}
                for r in entry[s].keys() & state.keys():
                    a_k, b_k = entry[s][r], state[r]
                    if a_k == b_k:
                        merged[r] = a_k
                    elif a_k == EITHER:
                        merged[r] = b_k
                    elif b_k == EITHER:
                        merged[r] = a_k
                    else:
                        merged[r] = MASK
            if merged != entry[s]:
                entry[s] = merged
                work.append(s)
    out, writes, either = {}, set(), {}
    for n, blk in enumerate(blocks):
        state = dict(entry[n] or {})
        for ins in blk:
            out[id(ins)] = {r: k for r, k in state.items() if k in MASK_KINDS}
            if include_either:
                either[id(ins)] = {r for r, k in state.items() if k == EITHER}
            if step(state, ins, wave64):
                writes.add(id(ins))
    return (out, writes, either) if include_either else (out, writes)
