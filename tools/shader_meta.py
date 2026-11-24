"""Static build-time metadata extracted from RDNA2 disassembly, DXBC containers and FSSM blobs.

Produces the per-shader fields described in docs/d3d12_vulkan.md (sections 3.5 and 3.6) for spirv/index.json so a
D3D12-to-Vulkan runtime can map root signatures to user SGPRs, match PSO shaders by FSSM content hash, bind vertex
inputs by ISG1 semantic name/index, pre-register literal S# samplers, and check descriptor-indexing modes without
parsing SPIR-V or guessing.

Standard library only; no game data.
"""
import hashlib
import os
import struct

# DXBC SFI0 (Shader Feature Info) 64-bit bitmask flags (Microsoft DXC DxilConstants.h).
SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING = 1 << 25   # 0x02000000
SFI0_SAMPLER_DESCRIPTOR_HEAP_INDEXING = 1 << 26    # 0x04000000

# DXBC ISG1 component types (D3D_REGISTER_COMPONENT_TYPE).
COMPONENT_TYPES = {1: "uint", 2: "sint", 3: "float"}

# Instructions whose operands are all sources (no SGPR destination).
NO_SGPR_DST = (
    "exp", "s_cmp_", "s_cmpk_", "s_bitcmp", "s_set", "s_rfe", "s_sendmsg", "s_branch", "s_cbranch",
    "s_endpgm", "s_waitcnt", "s_nop", "s_sleep", "s_barrier", "s_clause", "s_delay_alu", "s_setprio",
    "s_inst_prefetch", "s_dcache_", "s_icache_", "s_store_", "s_buffer_store_", "buffer_store",
    "buffer_atomic", "global_store", "global_atomic", "flat_store", "flat_atomic", "tbuffer_store",
    "image_store", "image_atomic", "ds_write",
)

# Scalar instructions that read AND write a[0].
SGPR_RMW = (
    "s_addk_i32", "s_mulk_i32", "s_cmovk_i32",
    "s_cmov_b32", "s_cmov_b64",
    "s_bitset0_b32", "s_bitset1_b32", "s_bitset0_b64", "s_bitset1_b64",
)

# Image instructions whose fourth operand a[3] is a 4-dword sampler descriptor (S#).
SAMPLER_IMAGE_OPS = ("image_sample", "image_gather4", "image_get_lod")

# Scalar shift-add instructions: s_lshlK_add_u32 sD, sIdx, sBase -> sD = (sIdx << K) + sBase.
LSHL_ADD_SHIFT = {
    "s_lshl1_add_u32": 1,
    "s_lshl2_add_u32": 2,
    "s_lshl3_add_u32": 3,
    "s_lshl4_add_u32": 4,
}

# Classification precedence when the same user SGPR participates in multiple roles.
KIND_PRIORITY = {
    "const32": 0,
    "ptr32_lo": 1,
    "ptr64_lo": 2,
    "ptr64_hi": 2,
    "vsharp": 3,
    "tsharp": 3,
    "ssharp": 3,
}


def base_op(op):
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            return op[:-len(suffix)]
    return op


def _imm_u32(a):
    """32-bit unsigned constant value of an imm/fimm operand, or None."""
    if a["k"] == "imm":
        v = a["v"] & 0xFFFFFFFF
    elif a["k"] == "fimm":
        try:
            v = struct.unpack("<I", struct.pack("<f", a["v"]))[0]
        except OverflowError:   # a double no float32 holds is not a constant this can name
            return None
    else:
        return None
    if a.get("abs"):
        v &= 0x7FFFFFFF
    if a.get("neg"):
        v ^= 0x80000000
    return v


def _sgpr_regs(a):
    if a["k"] == "sgpr":
        return list(range(a["r"], a["r"] + a.get("n", 1)))
    if a["k"] == "slist":
        return [r for r in a["r"] if isinstance(r, int)]
    return []


def _sgpr_slots(a, n, exact=True):
    """Return `n` scalar operand slots (SGPR int indices or special names like 'vcc_lo'), or None."""
    if a["k"] == "sgpr":
        count = a.get("n", 1)
        if count == n if exact else count >= n:
            return list(range(a["r"], a["r"] + n))
    elif a["k"] == "slist":
        items = a["r"]
        if len(items) == n if exact else len(items) >= n:
            return list(items[:n])
    return None


def _slot_tag(sym, slot):
    return sym.get(slot) if isinstance(slot, int) else None


def _record_ptr_pair(events, sym, slots, is_desc_width=False):
    lo_t, hi_t = _slot_tag(sym, slots[0]), _slot_tag(sym, slots[1])
    u_lo, u_hi = _user_of(lo_t), _user_of(hi_t)
    if u_lo is not None and u_hi is not None:
        events["ptr64"].add((u_lo, u_hi))
        if is_desc_width:
            events["table_loads"] = True
    elif u_lo is not None and hi_t == ("const", 0):
        events["ptr32_lo"].add(u_lo)
        events["truncates_high"] = True
        if is_desc_width:
            events["table_loads"] = True
    elif u_lo is not None:
        events["ptr64_lo_only"].add(u_lo)
        if is_desc_width:
            events["table_loads"] = True


def _user_of(tag):
    if tag is not None and tag[0] in ("user", "user_off"):
        return tag[1]
    return None


def _const_of(sym, a):
    v = _imm_u32(a)
    if v is not None:
        return v
    if a["k"] == "sgpr" and a.get("n", 1) == 1:
        t = sym.get(a["r"])
        if t is not None and t[0] == "const":
            return t[1]
    if a["k"] == "special" and a.get("t") == "null":
        return 0
    return None


def _split_operands(ins):
    """Return (read_sgprs, write_sgprs) for one parsed instruction."""
    op = base_op(ins["op"])
    a = ins["args"]
    if not a:
        return [], []
    if op.startswith(NO_SGPR_DST):
        reads = [r for x in a for r in _sgpr_regs(x)]
        return reads, []
    if op.startswith("v_cmpx_"):
        # e32 has 2 source operands; e64 has dst in a[0] and 2 sources in a[1:3].
        if len(a) <= 2:
            return [r for x in a for r in _sgpr_regs(x)], []
        return [r for x in a[1:] for r in _sgpr_regs(x)], _sgpr_regs(a[0])
    if op.startswith("v_cmp_"):
        if len(a) <= 2:
            return [r for x in a for r in _sgpr_regs(x)], []
        return [r for x in a[1:] for r in _sgpr_regs(x)], _sgpr_regs(a[0])
    if op in SGPR_RMW:
        reads = [r for x in a for r in _sgpr_regs(x)]
        return reads, _sgpr_regs(a[0])
    if (op.startswith("v_") and "_co_" in op) or op in (
        "v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64",
    ):
        # vdst in a[0], scalar carry/scale-out dst in a[1], sources in a[2:].
        writes = _sgpr_regs(a[0]) + (_sgpr_regs(a[1]) if len(a) > 1 else [])
        reads = [r for x in a[2:] for r in _sgpr_regs(x)]
        return reads, writes
    reads = [r for x in a[1:] for r in _sgpr_regs(x)]
    writes = _sgpr_regs(a[0])
    return reads, writes


def _fold_scalar_alu(op, x, y):
    if op in ("s_add_u32", "s_add_i32"):
        return (x + y) & 0xFFFFFFFF
    if op in ("s_sub_u32", "s_sub_i32"):
        return (x - y) & 0xFFFFFFFF
    if op == "s_and_b32":
        return x & y
    if op == "s_or_b32":
        return x | y
    if op == "s_xor_b32":
        return x ^ y
    if op == "s_andn2_b32":
        return x & (~y & 0xFFFFFFFF)
    if op == "s_orn2_b32":
        return (x | (~y & 0xFFFFFFFF)) & 0xFFFFFFFF
    if op == "s_nand_b32":
        return ~(x & y) & 0xFFFFFFFF
    if op == "s_nor_b32":
        return ~(x | y) & 0xFFFFFFFF
    if op == "s_xnor_b32":
        return ~(x ^ y) & 0xFFFFFFFF
    if op == "s_lshl_b32":
        return (x << (y & 31)) & 0xFFFFFFFF
    if op == "s_lshr_b32":
        return (x & 0xFFFFFFFF) >> (y & 31)
    if op == "s_mul_i32":
        return (x * y) & 0xFFFFFFFF
    if op == "s_ashr_i32":
        signed = x - 0x100000000 if x & 0x80000000 else x
        return (signed >> (y & 31)) & 0xFFFFFFFF
    if op == "s_bfm_b32":
        # A mask of (x & 31) ones, shifted left by (y & 31).
        return (((1 << (x & 31)) - 1) << (y & 31)) & 0xFFFFFFFF
    if op == "s_bfe_u32":
        # y packs the offset in bits [4:0] and the width in bits [22:16]; the field stops at bit 31.
        offset = y & 31
        width = min((y >> 16) & 0x7F, 32 - offset)
        return ((x & 0xFFFFFFFF) >> offset) & ((1 << width) - 1)
    if op == "s_min_u32":
        return min(x & 0xFFFFFFFF, y & 0xFFFFFFFF)
    if op == "s_max_u32":
        return max(x & 0xFFFFFFFF, y & 0xFFFFFFFF)
    return None


def _valid_stride(n):
    return isinstance(n, int) and 4 <= n <= 4096 and n % 4 == 0


def _step_state(sym, scale, ptr_stride, vscale, ins, events=None):
    """Advance symbolic SGPR origin/constants and descriptor-stride state by one instruction."""
    op = base_op(ins["op"])
    a = ins["args"]
    if not a:
        return

    # Record pointer, descriptor, stride and literal-sampler observations BEFORE writes clobber operands.
    if events is not None:
        if op.startswith(("s_load_dword", "s_store_dword")) and len(a) >= 2:
            slots = _sgpr_slots(a[1], 2)
            if slots is not None:
                is_desc_width = op in ("s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16")
                _record_ptr_pair(events, sym, slots, is_desc_width=is_desc_width)
                if is_desc_width:
                    if isinstance(slots[0], int):
                        events["strides"].update(ptr_stride.get(slots[0], ()))
                    if len(a) >= 3 and a[2]["k"] == "sgpr" and a[2].get("n", 1) == 1:
                        events["strides"].update(scale.get(a[2]["r"], ()))
        elif op.startswith(("s_buffer_load_dword", "s_buffer_store_dword")) and len(a) >= 2:
            slots = _sgpr_slots(a[1], 4)
            if slots is not None:
                tags = [_slot_tag(sym, s) for s in slots]
                if all(t is not None and t[0] == "user" for t in tags):
                    for t in tags:
                        events["vsharp"].add(t[1])
                if op in ("s_buffer_load_dwordx4", "s_buffer_load_dwordx8", "s_buffer_load_dwordx16"):
                    if len(a) >= 3 and a[2]["k"] == "sgpr" and a[2].get("n", 1) == 1:
                        events["strides"].update(scale.get(a[2]["r"], ()))
        elif op.startswith(("global_load", "global_store")) and len(a) >= 3:
            slots = _sgpr_slots(a[2], 2)
            if slots is not None:
                _record_ptr_pair(events, sym, slots, is_desc_width=False)
        elif op.startswith("global_atomic_"):
            saddr_idx = 3 if "glc" in ins.get("mods", ()) else 2
            if len(a) > saddr_idx:
                slots = _sgpr_slots(a[saddr_idx], 2)
                if slots is not None:
                    _record_ptr_pair(events, sym, slots, is_desc_width=False)
        elif op.startswith(("buffer_load", "buffer_store", "buffer_atomic", "tbuffer_")) and len(a) >= 3:
            slots = _sgpr_slots(a[2], 4)
            if slots is not None:
                tags = [_slot_tag(sym, s) for s in slots]
                if all(t is not None and t[0] == "user" for t in tags):
                    for t in tags:
                        events["vsharp"].add(t[1])
        elif op.startswith("image_") and len(a) >= 3:
            slots = _sgpr_slots(a[2], 8, exact=False)
            if slots is not None:
                tags = [_slot_tag(sym, s) for s in slots]
                if all(t is not None and t[0] == "user" for t in tags):
                    for t in tags:
                        events["tsharp"].add(t[1])
            if op.startswith(SAMPLER_IMAGE_OPS) and len(a) >= 4:
                s_slots = _sgpr_slots(a[3], 4, exact=False)
                if s_slots is not None:
                    tags = [_slot_tag(sym, s) for s in s_slots]
                    if all(t is not None and t[0] == "const" for t in tags):
                        events["literal_samplers"].add(tuple(t[1] for t in tags))
                    elif all(t is not None and t[0] == "user" for t in tags):
                        for t in tags:
                            events["ssharp"].add(t[1])

    # Track VGPR index scaling (e.g. v_lshlrev_b32 -> v_readfirstlane_b32 -> s_load_dwordx8).
    if a[0]["k"] == "vgpr" and not op.startswith(NO_SGPR_DST) and not op.startswith("v_cmp"):
        vd = a[0]["r"]
        vn = a[0].get("n", 1)
        new_vs = frozenset()
        if vn == 1:
            if op == "v_lshlrev_b32" and len(a) >= 3:
                sh = _imm_u32(a[1])
                if sh is not None and 2 <= sh <= 12:
                    new_vs = frozenset((1 << sh,))
            elif op in ("v_mul_lo_u32", "v_mul_u32_u24", "v_mul_i32_i24") and len(a) >= 3:
                c1, c2 = _imm_u32(a[1]), _imm_u32(a[2])
                c = c2 if c1 is None else c1
                if _valid_stride(c):
                    new_vs = frozenset((c,))
            elif op == "v_mov_b32" and len(a) >= 2 and a[1]["k"] == "vgpr" and a[1].get("n", 1) == 1:
                new_vs = vscale.get(a[1]["r"], frozenset())
            elif op == "v_add_nc_u32" and len(a) >= 3:
                if a[1]["k"] == "vgpr" and _imm_u32(a[2]) is not None:
                    new_vs = vscale.get(a[1]["r"], frozenset())
                elif a[2]["k"] == "vgpr" and _imm_u32(a[1]) is not None:
                    new_vs = vscale.get(a[2]["r"], frozenset())
        for r in range(vd, vd + vn):
            if r == vd and new_vs:
                vscale[r] = new_vs
            else:
                vscale.pop(r, None)

    # Update SGPR symbolic state for writes.
    _, writes = _split_operands(ins)
    if not writes:
        return

    new_sym = {}
    new_scale = {}
    new_pstride = {}

    if op == "s_mov_b32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        d = a[0]["r"]
        c = _const_of(sym, a[1])
        if c is not None:
            new_sym[d] = ("const", c)
        elif a[1]["k"] == "sgpr" and a[1].get("n", 1) == 1:
            s = a[1]["r"]
            if s in sym:
                new_sym[d] = sym[s]
            if s in scale:
                new_scale[d] = scale[s]
            if s in ptr_stride:
                new_pstride[d] = ptr_stride[s]
    elif op == "s_brev_b32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        c = _const_of(sym, a[1])
        if c is not None:
            new_sym[a[0]["r"]] = ("const", int(f"{c & 0xFFFFFFFF:032b}"[::-1], 2))
    elif op == "s_movk_i32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1 and a[1]["k"] == "imm":
        v = a[1]["v"] & 0xFFFF
        if v & 0x8000:
            v -= 0x10000
        new_sym[a[0]["r"]] = ("const", v & 0xFFFFFFFF)
    elif op == "s_mov_b64" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n") == 2:
        d = a[0]["r"]
        if a[1]["k"] == "imm":
            v = a[1]["v"] & 0xFFFFFFFFFFFFFFFF
            new_sym[d] = ("const", v & 0xFFFFFFFF)
            new_sym[d + 1] = ("const", (v >> 32) & 0xFFFFFFFF)
        elif a[1]["k"] == "sgpr" and a[1].get("n") == 2:
            s = a[1]["r"]
            for i in (0, 1):
                if s + i in sym:
                    new_sym[d + i] = sym[s + i]
                if s + i in ptr_stride:
                    new_pstride[d + i] = ptr_stride[s + i]
    elif op == "s_not_b32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        # A dword of a literal S# built as the complement of a known constant (for example 0xFFFFFFFF from 0).
        c = _const_of(sym, a[1])
        if c is not None:
            new_sym[a[0]["r"]] = ("const", ~c & 0xFFFFFFFF)
    elif op == "s_abs_i32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        c = _const_of(sym, a[1])
        if c is not None:
            signed = c - 0x100000000 if c & 0x80000000 else c
            new_sym[a[0]["r"]] = ("const", abs(signed) & 0xFFFFFFFF)  # abs(0x80000000) stays 0x80000000
    elif op == "v_readfirstlane_b32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        if a[1]["k"] == "vgpr" and a[1].get("n", 1) == 1 and a[1]["r"] in vscale:
            new_scale[a[0]["r"]] = vscale[a[1]["r"]]
    elif op in LSHL_ADD_SHIFT and len(a) >= 3 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1:
        d = a[0]["r"]
        stride = 1 << LSHL_ADD_SHIFT[op]
        base_tag = sym.get(a[2]["r"]) if a[2]["k"] == "sgpr" and a[2].get("n", 1) == 1 else None
        u = _user_of(base_tag)
        if u is not None:
            new_sym[d] = ("user_off", u)
            if _valid_stride(stride):
                new_pstride[d] = ptr_stride.get(a[2]["r"], frozenset()) | frozenset((stride,))
        else:
            cx, cy = _const_of(sym, a[1]), _const_of(sym, a[2])
            if cx is not None and cy is not None:
                new_sym[d] = ("const", ((cx << LSHL_ADD_SHIFT[op]) + cy) & 0xFFFFFFFF)
    elif op in ("s_add_u32", "s_add_i32", "s_sub_u32", "s_sub_i32") and len(a) >= 3 and a[0]["k"] == "sgpr":
        d = a[0]["r"]
        cx, cy = _const_of(sym, a[1]), _const_of(sym, a[2])
        if cx is not None and cy is not None:
            new_sym[d] = ("const", _fold_scalar_alu(op, cx, cy))
        else:
            tx = sym.get(a[1]["r"]) if a[1]["k"] == "sgpr" and a[1].get("n", 1) == 1 else None
            ty = sym.get(a[2]["r"]) if a[2]["k"] == "sgpr" and a[2].get("n", 1) == 1 else None
            ux, uy = _user_of(tx), _user_of(ty)
            sx = scale.get(a[1]["r"], frozenset()) if a[1]["k"] == "sgpr" else frozenset()
            sy = scale.get(a[2]["r"], frozenset()) if a[2]["k"] == "sgpr" else frozenset()
            psx = ptr_stride.get(a[1]["r"], frozenset()) if a[1]["k"] == "sgpr" else frozenset()
            psy = ptr_stride.get(a[2]["r"], frozenset()) if a[2]["k"] == "sgpr" else frozenset()
            if ux is not None and uy is None:
                new_sym[d] = ("user_off", ux)
                if psx or sy:
                    new_pstride[d] = psx | sy
            elif op.startswith("s_add") and uy is not None and ux is None:
                new_sym[d] = ("user_off", uy)
                if psy or sx:
                    new_pstride[d] = psy | sx
            elif sx and cy is not None:
                new_scale[d] = sx
            elif op.startswith("s_add") and sy and cx is not None:
                new_scale[d] = sy
    elif op in ("s_addc_u32", "s_subb_u32") and len(a) >= 3 and a[0]["k"] == "sgpr":
        d = a[0]["r"]
        tx = sym.get(a[1]["r"]) if a[1]["k"] == "sgpr" and a[1].get("n", 1) == 1 else None
        ty = sym.get(a[2]["r"]) if a[2]["k"] == "sgpr" and a[2].get("n", 1) == 1 else None
        ux, uy = _user_of(tx), _user_of(ty)
        if ux is not None:
            new_sym[d] = ("user_off", ux)
        elif op == "s_addc_u32" and uy is not None:
            new_sym[d] = ("user_off", uy)
        elif _const_of(sym, a[1]) == 0 and _const_of(sym, a[2]) == 0:
            # High-dword carry-add on a zeroed upper word (descriptor table truncated below 4 GB).
            new_sym[d] = ("const", 0)
    elif op in ("s_lshl_b32", "s_lshr_b32", "s_mul_i32", "s_and_b32", "s_or_b32", "s_xor_b32",
                "s_andn2_b32", "s_orn2_b32", "s_nand_b32", "s_nor_b32", "s_xnor_b32", "s_ashr_i32",
                "s_bfm_b32", "s_bfe_u32", "s_min_u32", "s_max_u32") and len(a) >= 3 and a[0]["k"] == "sgpr":
        d = a[0]["r"]
        cx, cy = _const_of(sym, a[1]), _const_of(sym, a[2])
        if cx is not None and cy is not None:
            new_sym[d] = ("const", _fold_scalar_alu(op, cx, cy))
        elif op == "s_lshl_b32" and cy is not None and 2 <= cy <= 12:
            new_scale[d] = frozenset((1 << cy,))
        elif op == "s_mul_i32":
            c = cy if cy is not None else cx
            if _valid_stride(c):
                new_scale[d] = frozenset((c,))
    elif op == "s_addk_i32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1 and a[1]["k"] == "imm":
        d = a[0]["r"]
        v = a[1]["v"] & 0xFFFF
        if v & 0x8000:
            v -= 0x10000
        old = sym.get(d)
        if old is not None and old[0] == "const":
            new_sym[d] = ("const", (old[1] + v) & 0xFFFFFFFF)
        elif _user_of(old) is not None:
            new_sym[d] = ("user_off", _user_of(old))
            if d in ptr_stride:
                new_pstride[d] = ptr_stride[d]
        elif d in scale:
            new_scale[d] = scale[d]
    elif op == "s_mulk_i32" and len(a) >= 2 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1 and a[1]["k"] == "imm":
        d = a[0]["r"]
        v = a[1]["v"] & 0xFFFF
        if v & 0x8000:
            v -= 0x10000
        old = sym.get(d)
        if old is not None and old[0] == "const":
            new_sym[d] = ("const", (old[1] * v) & 0xFFFFFFFF)
        elif _valid_stride(v):
            new_scale[d] = frozenset((v,))

    for r in writes:
        if r in new_sym:
            sym[r] = new_sym[r]
        else:
            sym.pop(r, None)
        if r in new_scale:
            scale[r] = new_scale[r]
        else:
            scale.pop(r, None)
        if r in new_pstride:
            ptr_stride[r] = new_pstride[r]
        else:
            ptr_stride.pop(r, None)


def analyse_instructions(blocks, succ, user_sgpr_limit=32, system_sgprs=()):
    """Analyse reachable RDNA2 basic blocks for user-SGPR reads/kinds, descriptor strides and literal samplers.

    Returns a dict with:
      user_sgpr_reads:   [{"sgpr": r, "kind": "const32"|"ptr64_lo"|"ptr64_hi"|"ptr32_lo"|"vsharp"|"tsharp"|"ssharp", ...}]
      descriptor_strides: sorted list of byte strides implied by descriptor address arithmetic
      literal_samplers:  sorted list of 4-dword literal S# values built from immediates
      table_loads:       bool, whether the shader loads 4/8/16-dword descriptors from a user-SGPR pointer
      truncates_high:    bool, whether any user-SGPR pointer has its high 32-bit word forced to 0
    """
    nblk = len(blocks)
    if nblk == 0:
        return {"user_sgpr_reads": [], "descriptor_strides": [], "literal_samplers": [],
                "table_loads": False, "truncates_high": False}
    sys_set = frozenset(system_sgprs)
    user_range = {r for r in range(user_sgpr_limit) if r not in sys_set}

    # Fixed-point forward dataflow from block 0:
    #   entry_written[b]: SGPRs definitely written on every path to block b (must-write).
    #   entry_sym[b]:     SGPR -> symbolic origin/const that agrees on every path to block b.
    #   entry_scale[b], entry_pstride[b], entry_vscale[b]: union of observed descriptor byte strides.
    entry_written = [None] * nblk
    entry_sym = [None] * nblk
    entry_scale = [None] * nblk
    entry_pstride = [None] * nblk
    entry_vscale = [None] * nblk

    entry_written[0] = frozenset()
    entry_sym[0] = {r: ("user", r) for r in user_range}
    entry_scale[0] = {}
    entry_pstride[0] = {}
    entry_vscale[0] = {}
    work = [0]

    def merge_sets(dst, src):
        out = dict(dst)
        for k, v in src.items():
            out[k] = out[k] | v if k in out else v
        return out

    while work:
        b = work.pop()
        written = set(entry_written[b])
        sym = dict(entry_sym[b])
        scale = dict(entry_scale[b])
        pstride = dict(entry_pstride[b])
        vscale = dict(entry_vscale[b])
        for ins in blocks[b]:
            if base_op(ins["op"]) == "s_endpgm":
                break
            _, writes = _split_operands(ins)
            _step_state(sym, scale, pstride, vscale, ins, events=None)
            written.update(writes)
        out_w = frozenset(written)
        for s in succ[b]:
            if entry_written[s] is None:
                mw = out_w
                ms = dict(sym)
                msc = dict(scale)
                mps = dict(pstride)
                mvs = dict(vscale)
            else:
                mw = entry_written[s] & out_w
                ms = {k: v for k, v in entry_sym[s].items() if sym.get(k) == v}
                msc = merge_sets(entry_scale[s], scale)
                mps = merge_sets(entry_pstride[s], pstride)
                mvs = merge_sets(entry_vscale[s], vscale)
            if (mw != entry_written[s] or ms != entry_sym[s] or msc != entry_scale[s] or
                    mps != entry_pstride[s] or mvs != entry_vscale[s]):
                entry_written[s] = mw
                entry_sym[s] = ms
                entry_scale[s] = msc
                entry_pstride[s] = mps
                entry_vscale[s] = mvs
                work.append(s)

    # Final pass over all reachable blocks to collect read-before-write user SGPRs and events.
    rbw = set()
    events = {
        "ptr64": set(),
        "ptr64_lo_only": set(),
        "ptr32_lo": set(),
        "vsharp": set(),
        "tsharp": set(),
        "ssharp": set(),
        "strides": set(),
        "literal_samplers": set(),
        "table_loads": False,
        "truncates_high": False,
    }
    for b in range(nblk):
        if entry_written[b] is None:
            continue
        written = set(entry_written[b])
        sym = dict(entry_sym[b])
        scale = dict(entry_scale[b])
        pstride = dict(entry_pstride[b])
        vscale = dict(entry_vscale[b])
        for ins in blocks[b]:
            if base_op(ins["op"]) == "s_endpgm":
                break
            reads, writes = _split_operands(ins)
            for r in reads:
                if r in user_range and r not in written:
                    rbw.add(r)
            _step_state(sym, scale, pstride, vscale, ins, events=events)
            written.update(writes)

    # Assign the strongest observed kind to each read-before-write user SGPR.
    kinds = {r: "const32" for r in rbw}
    pairs = {}

    def promote(r, kind):
        if r in kinds and KIND_PRIORITY[kind] > KIND_PRIORITY[kinds[r]]:
            kinds[r] = kind

    for lo, hi in sorted(events["ptr64"]):
        if lo in kinds and hi in kinds:
            promote(lo, "ptr64_lo")
            promote(hi, "ptr64_hi")
            pairs[lo] = [lo, hi]
            pairs[hi] = [lo, hi]
    for lo in sorted(events["ptr64_lo_only"]):
        promote(lo, "ptr64_lo")
    for lo in sorted(events["ptr32_lo"]):
        promote(lo, "ptr32_lo")
    for kname in ("vsharp", "tsharp", "ssharp"):
        for r in sorted(events[kname]):
            promote(r, kname)

    user_reads = []
    for r in sorted(rbw):
        row = {"sgpr": r, "kind": kinds[r]}
        if r in pairs and kinds[r] in ("ptr64_lo", "ptr64_hi"):
            row["pair"] = pairs[r]
        user_reads.append(row)

    return {
        "user_sgpr_reads": user_reads,
        "descriptor_strides": sorted(events["strides"]),
        "literal_samplers": [list(s) for s in sorted(events["literal_samplers"])],
        "table_loads": events["table_loads"],
        "truncates_high": events["truncates_high"],
    }


def read_dxbc_meta(data):
    """Parse ISG1 full input signature and SFI0 descriptor-heap indexing flags from a DXBC container."""
    if not data or len(data) < 32 or data[:4] != b"DXBC":
        return {}
    cnt = struct.unpack_from("<I", data, 28)[0]
    if 32 + 4 * cnt > len(data):
        return {}
    parts = {}
    for k in range(cnt):
        off = struct.unpack_from("<I", data, 32 + 4 * k)[0]
        if off + 8 > len(data):
            continue
        fourcc = data[off:off + 4]
        plen = struct.unpack_from("<I", data, off + 4)[0]
        if off + 8 + plen <= len(data):
            parts[fourcc] = data[off + 8:off + 8 + plen]

    out = {}
    if b"ISG1" in parts:
        p = parts[b"ISG1"]
        if len(p) >= 8:
            ecnt, eo = struct.unpack_from("<II", p, 0)
            strings = eo + 32 * ecnt
            if 8 <= eo and strings <= len(p):
                elems = []
                for i in range(ecnt):
                    base = eo + 32 * i
                    _, noff, sidx, sysv, ctype, reg = struct.unpack_from("<6I", p, base)
                    mask, rw_mask = p[base + 24], p[base + 25]
                    if not strings <= noff < len(p):
                        break
                    end = p.find(b"\x00", noff)
                    if end <= noff:
                        break
                    semantic = p[noff:end].decode("ascii", "replace")
                    elems.append({
                        "semantic": semantic,
                        "semantic_index": sidx,
                        "register": reg,
                        "mask": mask,
                        "rw_mask": rw_mask,
                        "system_value": sysv,
                        "component_type": COMPONENT_TYPES.get(ctype, str(ctype)),
                    })
                if elems and len(elems) == ecnt:
                    out["input_signature"] = elems

    if b"SFI0" in parts and len(parts[b"SFI0"]) >= 8:
        flags = struct.unpack_from("<Q", parts[b"SFI0"], 0)[0]
        out["sfi0_resource_heap_directly_indexed"] = bool(flags & SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING)
        out["sfi0_sampler_heap_directly_indexed"] = bool(flags & SFI0_SAMPLER_DESCRIPTOR_HEAP_INDEXING)
    return out


def fssm_sha256_file(disasm_path, name):
    """SHA-256 hex digest of <disasm_dir>/../fssm/<name>.fssm, or None if absent."""
    parent = os.path.dirname(os.path.dirname(os.path.realpath(disasm_path)))
    fssm = os.path.join(parent, "fssm", name + ".fssm")
    if not os.path.isfile(fssm):
        return None
    with open(fssm, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fspx_code_sha256_file(disasm_path, name):
    """SHA-256 hex digest of <disasm_dir>/../fspx/<name>.code, the kind-6 program tools/fspx_unpack.py writes for an
    FSPX shader without an FSSM blob, or None. The run time computes the same key from the bytes a compute PSO gets
    (docs/fspx.md, "Run-time lookup")."""
    parent = os.path.dirname(os.path.dirname(os.path.realpath(disasm_path)))
    if os.path.isfile(os.path.join(parent, "fssm", name + ".fssm")):
        return None
    code = os.path.join(parent, "fspx", name + ".code")
    if not os.path.isfile(code):
        return None
    with open(code, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fspx_state_sha256_file(disasm_path, name):
    """SHA-256 hex digest of <disasm_dir>/../fspx/<name>.state, the program's kind-0 entry (dispatch state), or None.
    Two FSPX programs can share their kind-6 bytes and differ in kind 0; the run time tells them apart by this key
    when the bytecode carries kind 0 (docs/fspx.md, "Build and run-time lookup")."""
    parent = os.path.dirname(os.path.dirname(os.path.realpath(disasm_path)))
    state = os.path.join(parent, "fspx", name + ".state")
    if not os.path.isfile(state):
        return None
    with open(state, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def collect(disasm_path, name, blocks, succ, user_sgpr_limit=32, system_sgprs=()):
    """Collect all build-time runtime metadata for one translated shader."""
    meta = {}
    digest = fssm_sha256_file(disasm_path, name)
    if digest is not None:
        meta["fssm_sha256"] = digest
    else:
        code_digest = fspx_code_sha256_file(disasm_path, name)
        if code_digest is not None:
            meta["fspx_code_sha256"] = code_digest
            state_digest = fspx_state_sha256_file(disasm_path, name)
            if state_digest is not None:
                meta["fspx_state_sha256"] = state_digest

    dxbc_info = {}
    parent = os.path.dirname(os.path.dirname(os.path.realpath(disasm_path)))
    dxbc_path = os.path.join(parent, name + ".dxbc")
    if os.path.isfile(dxbc_path):
        with open(dxbc_path, "rb") as f:
            dxbc_info = read_dxbc_meta(f.read())
    if "input_signature" in dxbc_info:
        meta["input_signature"] = dxbc_info["input_signature"]

    ins_info = analyse_instructions(blocks, succ, user_sgpr_limit=user_sgpr_limit, system_sgprs=system_sgprs)
    if ins_info["user_sgpr_reads"]:
        meta["user_sgpr_reads"] = ins_info["user_sgpr_reads"]
    if ins_info["descriptor_strides"]:
        meta["descriptor_strides"] = ins_info["descriptor_strides"]
    if ins_info["literal_samplers"]:
        meta["literal_samplers"] = ins_info["literal_samplers"]

    has_sfi0 = "sfi0_resource_heap_directly_indexed" in dxbc_info
    if has_sfi0 or ins_info["table_loads"] or ins_info["truncates_high"]:
        desc_access = {
            "table_loads": ins_info["table_loads"],
            "truncates_table_ptr_high": ins_info["truncates_high"],
        }
        if has_sfi0:
            desc_access["resource_heap_directly_indexed"] = dxbc_info["sfi0_resource_heap_directly_indexed"]
            desc_access["sampler_heap_directly_indexed"] = dxbc_info["sfi0_sampler_heap_directly_indexed"]
        meta["descriptor_access"] = desc_access

    return meta
