"""Pure-Python RDNA2 (GFX10.1 / GFX10.3) binary ISA decoder (no LLVM dependency).

Decodes raw RDNA2 machine code directly into either:
  - `shader_ir`-compatible instruction dictionaries (`decode_bytes`), or
  - `shader_ir`-compatible `.s` disassembly text with `; encoding: [...]` (`disasm_bytes`).

Opcode tables in `tools/rdna2_isa_tables.py` are derived from AMD's MIT-licensed
machine-readable RDNA2 ISA specification (`amdgpu_isa_rdna2.xml`), plus the seven
gfx1013 legacy/waveid instructions documented in `docs/isa_coverage.md`.

Standard library only; no game data, no external tools.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections.abc import Iterator
from io import StringIO

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2_isa_tables as T  # noqa: E402

with open(os.path.join(HERE, "gfx10_formats.json"), encoding="utf-8") as _f:
    _GFX10_FORMATS = {int(k): v for k, v in json.load(_f).items()}

# Inline float constants (codes 240..248) matching LLVM AMDGPUInstPrinter.
_INLINE_FLOAT = {
    240: 0.5,
    241: -0.5,
    242: 1.0,
    243: -1.0,
    244: 2.0,
    245: -2.0,
    246: 4.0,
    247: -4.0,
    248: 0.15915494,  # 1 / (2 * pi)
}

_INLINE_FLOAT_STR = {
    240: "0.5",
    241: "-0.5",
    242: "1.0",
    243: "-1.0",
    244: "2.0",
    245: "-2.0",
    246: "4.0",
    247: "-4.0",
    248: "0.15915494",
}

# VOP2 instructions that always carry a trailing 32-bit literal constant.
_VOP2_MANDATORY_LITERAL = frozenset({
    0x20,  # v_madmk_f32 (gfx1013)
    0x21,  # v_madak_f32 (gfx1013)
    0x2C,  # v_fmamk_f32
    0x2D,  # v_fmaak_f32
    0x37,  # v_fmamk_f16
    0x38,  # v_fmaak_f16
})

# SOPK instructions that always carry a trailing 32-bit literal constant.
_SOPK_MANDATORY_LITERAL = frozenset({
    0x15,  # s_setreg_imm32_b32
})

_MIMG_DIMS = {
    0: "SQ_RSRC_IMG_1D",
    1: "SQ_RSRC_IMG_2D",
    2: "SQ_RSRC_IMG_3D",
    3: "SQ_RSRC_IMG_CUBE",
    4: "SQ_RSRC_IMG_1D_ARRAY",
    5: "SQ_RSRC_IMG_2D_ARRAY",
    6: "SQ_RSRC_IMG_2D_MSAA",
    7: "SQ_RSRC_IMG_2D_MSAA_ARRAY",
}

_MIMG_DIM_COORDS = {0: 1, 1: 2, 2: 3, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4}
_MIMG_GRAD_COORDS = {0: 1, 1: 2, 2: 3, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2}

_SDWA_SELS = {
    0: "BYTE_0",
    1: "BYTE_1",
    2: "BYTE_2",
    3: "BYTE_3",
    4: "WORD_0",
    5: "WORD_1",
    6: "DWORD",
}
_SDWA_UNUSED = {0: "UNUSED_PAD", 1: "UNUSED_SEXT", 2: "UNUSED_PRESERVE"}
_OMOD_MODS = {1: "mul:2", 2: "mul:4", 3: "div:2"}

_HWREG_NAMES = {
    1: "HW_REG_MODE",
    2: "HW_REG_STATUS",
    3: "HW_REG_TRAPSTS",
    4: "HW_REG_HW_ID1",
    5: "HW_REG_GPR_ALLOC",
    6: "HW_REG_LDS_ALLOC",
    7: "HW_REG_IB_STS",
    15: "HW_REG_SH_MEM_BASES",
    16: "HW_REG_TBA_LO",
    17: "HW_REG_TBA_HI",
    18: "HW_REG_TMA_LO",
    19: "HW_REG_TMA_HI",
    20: "HW_REG_FLAT_SCR_LO",
    21: "HW_REG_FLAT_SCR_HI",
    22: "HW_REG_XNACK_MASK",
    23: "HW_REG_HW_ID2",
    24: "HW_REG_POPS_PACKER",
    26: "HW_REG_SHADER_CYCLES",
}

_SENDMSG_NAMES = {
    1: "MSG_INTERRUPT",
    2: "MSG_GS",
    3: "MSG_GS_DONE",
    4: "MSG_SAVEWAVE",
    5: "MSG_STALL_WAVE_GEN",
    6: "MSG_HALT_WAVES",
    7: "MSG_ORDERED_PS_DONE",
    8: "MSG_EARLY_PRIM_DEALLOC",
    9: "MSG_GS_ALLOC_REQ",
    10: "MSG_GET_DOORBELL",
    11: "MSG_GET_DDID",
    15: "MSG_SYSMSG",
}

_GS_OPS = {0: "GS_OP_NOP", 1: "GS_OP_CUT", 2: "GS_OP_EMIT", 3: "GS_OP_EMIT_CUT"}

# SOPP branch instructions (16-bit signed word offset relative to next PC).
_SOPP_BRANCHES = frozenset({
    "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
    "s_cbranch_execz", "s_cbranch_execnz", "s_cbranch_cdbgsys", "s_cbranch_cdbguser",
    "s_cbranch_cdbgsys_or_user", "s_cbranch_cdbgsys_and_user",
})

# SOPP instructions that take no operand at all.
_SOPP_NO_OPERAND = frozenset({
    "s_endpgm", "s_barrier", "s_icache_inv", "s_ttracedata",
    "s_wakeup", "s_endpgm_saved", "s_code_end", "s_endpgm_ordered_ps_done",
})


def _s16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _s12(v: int) -> int:
    v &= 0xFFF
    return v - 0x1000 if v & 0x800 else v


def _s21(v: int) -> int:
    v &= 0x1FFFFF
    return v - 0x200000 if v & 0x100000 else v


def _sreg(code: int, n: int) -> dict:
    """Decode a scalar destination/source register code (0..127) with width `n` dwords."""
    if code <= 105:
        if n <= 1:
            return {"k": "sgpr", "r": code, "n": 1, "neg": False, "abs": False}
        if code + n <= 106:
            return {"k": "sgpr", "r": code, "n": n, "neg": False, "abs": False}
        # LLVM VCCTuple spanning SGPRs into vcc_lo (106) / vcc_hi (107).
        if code + n <= 108:
            items = []
            for r in range(code, code + n):
                if r <= 105:
                    items.append(r)
                elif r == 106:
                    items.append("vcc_lo")
                else:
                    items.append("vcc_hi")
            return {"k": "slist", "r": items, "n": n}
        return {"k": "sgpr", "r": code, "n": n, "neg": False, "abs": False}
    if code == 106:
        return {"k": "special", "t": "vcc" if n == 2 else "vcc_lo", "neg": False, "abs": False}
    if code == 107:
        return {"k": "special", "t": "vcc_hi", "neg": False, "abs": False}
    if 108 <= code <= 123:
        lo = code - 108
        t = f"ttmp{lo}" if n <= 1 else f"ttmp[{lo}:{lo + n - 1}]"
        return {"k": "ttmp", "t": t}
    if code == 124:
        return {"k": "special", "t": "m0", "neg": False, "abs": False}
    if code == 125:
        return {"k": "special", "t": "null", "neg": False, "abs": False}
    if code == 126:
        return {"k": "special", "t": "exec" if n == 2 else "exec_lo", "neg": False, "abs": False}
    if code == 127:
        return {"k": "special", "t": "exec_hi", "neg": False, "abs": False}
    return {"k": "raw", "t": f"sreg_{code}"}


def _vreg(reg: int, n: int = 1) -> dict:
    return {"k": "vgpr", "r": reg & 0xFF, "n": max(1, n), "neg": False, "abs": False}


def _src(code: int, n: int, lit: int | None, *, neg: bool = False, absm: bool = False) -> dict:
    """Decode a 9-bit RDNA2 source operand (0..511)."""
    if code <= 127:
        op = _sreg(code, n)
    elif 128 <= code <= 192:
        op = {"k": "imm", "v": code - 128}
    elif 193 <= code <= 208:
        op = {"k": "imm", "v": -(code - 192)}
    elif code in _INLINE_FLOAT:
        op = {"k": "fimm", "v": _INLINE_FLOAT[code], "abs": False}
    elif code == 251:
        op = {"k": "special", "t": "src_vccz", "neg": False, "abs": False}
    elif code == 252:
        op = {"k": "special", "t": "src_execz", "neg": False, "abs": False}
    elif code == 253:
        op = {"k": "special", "t": "src_scc", "neg": False, "abs": False}
    elif code == 254:
        op = {"k": "special", "t": "src_lds_direct", "neg": False, "abs": False}
    elif code == 255:
        op = {"k": "imm", "v": 0 if lit is None else lit}
    elif 256 <= code <= 511:
        op = _vreg(code - 256, n)
    else:
        op = {"k": "raw", "t": f"src_{code}"}

    if absm:
        op["abs"] = True
    if neg:
        if op["k"] in ("imm", "fimm"):
            # Preserve neg modifier on constants (same semantics as shader_ir.operand("neg(...)")).
            op["neg"] = True
        else:
            op["neg"] = True
    return op


def _vcc_operand(wave32: bool) -> dict:
    return {"k": "special", "t": "vcc_lo" if wave32 else "vcc", "neg": False, "abs": False}


def _width(nregs: int, wave32: bool) -> int:
    if nregs == -1:
        return 1 if wave32 else 2
    return nregs


def _format_hwreg(simm16: int) -> str:
    reg_id = simm16 & 0x3F
    offset = (simm16 >> 6) & 0x1F
    size = ((simm16 >> 11) & 0x1F) + 1
    rname = _HWREG_NAMES.get(reg_id, str(reg_id))
    if offset == 0 and size == 32:
        return f"hwreg({rname})"
    return f"hwreg({rname}, {offset}, {size})"


def _format_waitcnt(simm16: int) -> list[dict]:
    # GFX10 s_waitcnt layout: vmcnt = bits[3:0] | (bits[15:14] << 4), expcnt = bits[6:4], lgkmcnt = bits[13:8].
    vmcnt = (simm16 & 0xF) | (((simm16 >> 14) & 0x3) << 4)
    expcnt = (simm16 >> 4) & 0x7
    lgkmcnt = (simm16 >> 8) & 0x3F
    parts = []
    if vmcnt != 63:
        parts.append({"k": "special", "t": f"vmcnt({vmcnt})"})
    if expcnt != 7:
        parts.append({"k": "special", "t": f"expcnt({expcnt})"})
    if lgkmcnt != 63:
        parts.append({"k": "special", "t": f"lgkmcnt({lgkmcnt})"})
    if not parts:
        parts.append({"k": "imm", "v": simm16})
    return parts


def _format_sendmsg(simm16: int) -> str:
    msg_id = simm16 & 0xF
    op = (simm16 >> 4) & 0x7
    stream = (simm16 >> 8) & 0x3
    mname = _SENDMSG_NAMES.get(msg_id)
    if mname is None:
        return f"sendmsg({simm16})"
    if msg_id in (2, 3):
        op_name = _GS_OPS.get(op, str(op))
        if op == 0:
            return f"sendmsg({mname}|{op_name})"
        return f"sendmsg({mname}|{op_name}|{stream})"
    if op:
        return f"sendmsg({mname}|{op})"
    return f"sendmsg({mname})"


def _format_dpp16_ctrl(ctrl: int) -> str:
    if ctrl <= 0xFF:
        q = [(ctrl >> (2 * i)) & 3 for i in range(4)]
        return f"quad_perm:[{q[0]},{q[1]},{q[2]},{q[3]}]"
    if 0x101 <= ctrl <= 0x10F:
        return f"row_shl:{ctrl - 0x100}"
    if 0x111 <= ctrl <= 0x11F:
        return f"row_shr:{ctrl - 0x110}"
    if 0x121 <= ctrl <= 0x12F:
        return f"row_ror:{ctrl - 0x120}"
    if ctrl == 0x140:
        return "row_mirror"
    if ctrl == 0x141:
        return "row_half_mirror"
    if 0x150 <= ctrl <= 0x15F:
        return f"row_share:{ctrl - 0x150}"
    if 0x160 <= ctrl <= 0x16F:
        return f"row_xmask:{ctrl - 0x160}"
    return f"dpp_ctrl:0x{ctrl:x}"


def _format_ds_swizzle(offset16: int) -> str:
    if (offset16 & 0x8000) == 0:
        # Bitmask mode: bits [4:0] = and_mask, [9:5] = or_mask, [14:10] = xor_mask.
        # Match LLVM's canonical aliases (SWAP, REVERSE, BROADCAST) before general BITMASK_PERM.
        and_m = offset16 & 0x1F
        or_m = (offset16 >> 5) & 0x1F
        xor_m = (offset16 >> 10) & 0x1F
        if and_m == 0x1F and or_m == 0:
            if xor_m in (1, 2, 4, 8, 16):
                return f"offset:swizzle(SWAP,{xor_m})"
            if xor_m in (3, 7, 15, 31):
                return f"offset:swizzle(REVERSE,{xor_m + 1})"
        if xor_m == 0 and (or_m & and_m) == 0 and (0x1F ^ and_m) in (1, 3, 7, 15, 31):
            grp = (0x1F ^ and_m) + 1
            return f"offset:swizzle(BROADCAST,{grp},{or_m})"
        chars = []
        for bit in range(4, -1, -1):
            a = (and_m >> bit) & 1
            o = (or_m >> bit) & 1
            x = (xor_m >> bit) & 1
            if a == 0 and o == 0 and x == 0:
                chars.append("0")
            elif a == 0 and o == 1 and x == 0:
                chars.append("1")
            elif a == 1 and o == 0 and x == 0:
                chars.append("p")
            elif a == 1 and o == 0 and x == 1:
                chars.append("i")
            else:
                return f"offset:{offset16}"
        return f'offset:swizzle(BITMASK_PERM,"{"".join(chars)}")'
    if (offset16 & 0xFF00) == 0x8000:
        p = [(offset16 >> (2 * i)) & 3 for i in range(4)]
        return f"offset:swizzle(QUAD_PERM,{p[0]},{p[1]},{p[2]},{p[3]})"
    return f"offset:{offset16}"


def _mimg_addr_count(name: str, dim: int, a16: bool) -> int:
    """Number of address VGPRs for an MIMG instruction."""
    if name in ("image_bvh_intersect_ray", "image_bvh64_intersect_ray"):
        return 11 if name == "image_bvh_intersect_ray" else 12
    if name == "image_get_resinfo":
        return 1
    if name == "image_get_lod":
        return _MIMG_DIM_COORDS.get(dim, 2)
    base = _MIMG_DIM_COORDS.get(dim, 2)
    if name.startswith("image_atomic_"):
        return (base + 1) // 2 if a16 else base
    # Separate 32-bit address components (offset, bias, z-compare, derivatives) from a16-packable coords/lod/clamp.
    has_o = name.endswith("_o") or "_o_" in name
    base_name = name[:-2] if name.endswith("_o") else name
    has_cl = base_name.endswith("_cl")
    if has_cl:
        base_name = base_name[:-3]
    fixed32 = 1 if has_o else 0
    if "_b" in base_name and not base_name.startswith("image_bvh"):
        fixed32 += 1
    if "_c" in base_name and not base_name.startswith(("image_atomic", "image_get")):
        fixed32 += 1
    if "_d" in base_name and not base_name.startswith(("image_atomic", "image_get")):
        fixed32 += 2 * _MIMG_GRAD_COORDS.get(dim, 2)
    packable = base
    if "_l" in base_name and "_lz" not in base_name and not base_name.startswith(
            ("image_load", "image_store", "image_atomic", "image_get")):
        packable += 1
    if has_cl:
        packable += 1
    if base_name in ("image_load_mip", "image_store_mip"):
        packable += 1
    if a16:
        # On GFX10, a16 packs coordinates (and lod/clamp after coords) two 16-bit values per VGPR; clamp/lod
        # after 2D coords starts a new VGPR when coords are even (e.g., 2D + cl -> 1 + 1 = 2 VGPRs).
        packed = (base + 1) // 2 + (packable - base)
        return fixed32 + packed
    return fixed32 + packable


def _mimg_data_count(name: str, dmask: int, d16: bool, tfe: bool) -> int:
    if name in ("image_bvh_intersect_ray", "image_bvh64_intersect_ray"):
        return 4
    if name.startswith("image_atomic_"):
        # The GFX10 atomic dmask selects the total VDATA width, including
        # both compare and swap values for cmpswap (masks 0x3 and 0xf).
        if dmask in (1, 3, 15):
            return dmask.bit_count()
        n = 2 if "cmpswap" in name else 1
        if name.endswith("_x2"):
            n *= 2
        return n
    if name.startswith("image_gather4"):
        n = 4
    else:
        n = bin(dmask & 0xF).count("1") or 1
    if d16:
        n = (n + 1) // 2
    return n + (1 if tfe else 0)


def _decode_one(code: bytes, pc: int, wave32: bool, mcpu: str) -> tuple[dict | None, int]:
    """Decode one RDNA2 instruction at byte offset `pc`. Returns (ins_dict_or_None, byte_length)."""
    if pc + 4 > len(code):
        return None, len(code) - pc
    w0 = struct.unpack_from("<I", code, pc)[0]

    def need(nbytes: int) -> bool:
        return pc + nbytes <= len(code)

    def finish(op: str, args: list, mods: list, nbytes: int) -> tuple[dict, int]:
        return {
            "op": op,
            "args": args,
            "mods": mods,
            "enc": code[pc:pc + nbytes].hex(),
            "pc": pc,
        }, nbytes

    # ---- 1. Bit 31 == 0: VOP1 / VOPC / VOP2 (plus DPP16 / DPP8 / SDWA / literal) ----
    if (w0 >> 31) == 0:
        hi7 = w0 >> 25
        src0 = w0 & 0x1FF
        is_dpp16 = src0 == 250
        is_dpp8 = src0 in (233, 234)
        is_sdwa = src0 == 249

        if hi7 == 0x3F:  # VOP1
            opc = (w0 >> 9) & 0xFF
            vdst = (w0 >> 17) & 0xFF
            entry = T.VOP1.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            mods = []
            if is_dpp16 or is_dpp8 or is_sdwa:
                if not need(8):
                    return None, 4
                w1 = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
                src0_reg = w1 & 0xFF
                if is_dpp16:
                    op_name = name + "_dpp"
                    ctrl = (w1 >> 8) & 0x1FF
                    fi = (w1 >> 18) & 1
                    bc = (w1 >> 19) & 1
                    bank_m = (w1 >> 24) & 0xF
                    row_m = (w1 >> 28) & 0xF
                    neg0 = bool((w1 >> 20) & 1)
                    abs0 = bool((w1 >> 21) & 1)
                    s0 = _vreg(src0_reg, 1)
                    s0["neg"], s0["abs"] = neg0, abs0
                    mods.append(_format_dpp16_ctrl(ctrl))
                    mods.append(f"row_mask:0x{row_m:x}")
                    mods.append(f"bank_mask:0x{bank_m:x}")
                    if bc:
                        mods.append("bound_ctrl:1")
                    if fi:
                        mods.append("fi:1")
                elif is_dpp8:
                    op_name = name + "_dpp"
                    lanes = [(w1 >> (8 + 3 * i)) & 7 for i in range(8)]
                    s0 = _vreg(src0_reg, 1)
                    mods.append(f"dpp8:[{','.join(str(x) for x in lanes)}]")
                    if src0 == 234:
                        mods.append("fi:1")
                else:  # SDWA
                    op_name = name + "_sdwa"
                    dst_sel = (w1 >> 8) & 7
                    dst_unused = (w1 >> 11) & 3
                    clamp = (w1 >> 13) & 1
                    omod = (w1 >> 14) & 3
                    src0_sel = (w1 >> 16) & 7
                    src0_sext = bool((w1 >> 19) & 1)
                    src0_neg = bool((w1 >> 20) & 1)
                    src0_abs = bool((w1 >> 21) & 1)
                    s0_is_sgpr = bool((w1 >> 23) & 1)
                    s0 = _src(src0_reg, 1, None) if s0_is_sgpr else _vreg(src0_reg, 1)
                    if src0_sext:
                        s0["sext"] = True
                    s0["neg"], s0["abs"] = src0_neg, src0_abs
                    if clamp:
                        mods.append("clamp")
                    if omod in _OMOD_MODS:
                        mods.append(_OMOD_MODS[omod])
                    mods.append(f"dst_sel:{_SDWA_SELS.get(dst_sel, 'DWORD')}")
                    mods.append(f"dst_unused:{_SDWA_UNUSED.get(dst_unused, 'UNUSED_PAD')}")
                    mods.append(f"src0_sel:{_SDWA_SELS.get(src0_sel, 'DWORD')}")
                args = []
                for fn, nr, _ in spec:
                    w = _width(nr, wave32)
                    if fn == "VDST":
                        args.append(_vreg(vdst, w))
                    elif fn == "SRC0":
                        args.append(s0)
                return finish(op_name, args, mods, nbytes)
            lit = None
            nbytes = 4
            if src0 == 255:
                if not need(8):
                    return None, 4
                lit = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
            op_name = name if not spec or name in ("v_nop", "v_clrexcp", "v_pipeflush", "v_readfirstlane_b32") else name + "_e32"
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "VDST":
                    args.append(_sreg(vdst & 0x7F, 1) if name == "v_readfirstlane_b32" else _vreg(vdst, w))
                elif fn == "SRC0":
                    args.append(_src(src0, w, lit))
            return finish(op_name, args, mods, nbytes)

        if hi7 == 0x3E:  # VOPC
            opc = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            entry = T.VOPC.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            mods = []
            if is_sdwa:
                if not need(8):
                    return None, 4
                w1 = struct.unpack_from("<I", code, pc + 4)[0]
                src0_reg = w1 & 0xFF
                sdst = (w1 >> 8) & 0x7F
                sd = bool((w1 >> 15) & 1)
                src0_sel = (w1 >> 16) & 7
                src0_sext = bool((w1 >> 19) & 1)
                src0_neg = bool((w1 >> 20) & 1)
                src0_abs = bool((w1 >> 21) & 1)
                s0_is_sgpr = bool((w1 >> 23) & 1)
                src1_sel = (w1 >> 24) & 7
                src1_sext = bool((w1 >> 27) & 1)
                src1_neg = bool((w1 >> 28) & 1)
                src1_abs = bool((w1 >> 29) & 1)
                s1_is_sgpr = bool((w1 >> 31) & 1)
                s0 = _src(src0_reg, 1, None) if s0_is_sgpr else _vreg(src0_reg, 1)
                s1 = _src(vsrc1, 1, None) if s1_is_sgpr else _vreg(vsrc1, 1)
                if src0_sext:
                    s0["sext"] = True
                if src1_sext:
                    s1["sext"] = True
                s0["neg"], s0["abs"] = src0_neg, src0_abs
                s1["neg"], s1["abs"] = src1_neg, src1_abs
                mods.append(f"src0_sel:{_SDWA_SELS.get(src0_sel, 'DWORD')}")
                mods.append(f"src1_sel:{_SDWA_SELS.get(src1_sel, 'DWORD')}")
                dst_op = _sreg(sdst, 1 if wave32 else 2) if sd else _vcc_operand(wave32)
                args = [dst_op, s0, s1] if not name.startswith("v_cmpx_") else [s0, s1]
                return finish(name + "_sdwa", args, mods, 8)
            lit = None
            nbytes = 4
            if src0 == 255:
                if not need(8):
                    return None, 4
                lit = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "VCC":
                    if not name.startswith("v_cmpx_"):
                        args.append(_vcc_operand(wave32))
                elif fn == "SRC0":
                    args.append(_src(src0, w, lit))
                elif fn == "VSRC1":
                    args.append(_vreg(vsrc1, w))
            return finish(name + "_e32", args, mods, nbytes)

        # VOP2
        opc = (w0 >> 25) & 0x3F
        vdst = (w0 >> 17) & 0xFF
        vsrc1 = (w0 >> 9) & 0xFF
        entry = T.VOP2.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        mods = []
        if is_dpp16 or is_dpp8 or is_sdwa:
            if not need(8):
                return None, 4
            w1 = struct.unpack_from("<I", code, pc + 4)[0]
            src0_reg = w1 & 0xFF
            if is_dpp16:
                op_name = name + "_dpp"
                ctrl = (w1 >> 8) & 0x1FF
                fi = (w1 >> 18) & 1
                bc = (w1 >> 19) & 1
                neg0 = bool((w1 >> 20) & 1)
                abs0 = bool((w1 >> 21) & 1)
                neg1 = bool((w1 >> 22) & 1)
                abs1 = bool((w1 >> 23) & 1)
                bank_m = (w1 >> 24) & 0xF
                row_m = (w1 >> 28) & 0xF
                s0 = _vreg(src0_reg, 1)
                s0["neg"], s0["abs"] = neg0, abs0
                s1 = _vreg(vsrc1, 1)
                s1["neg"], s1["abs"] = neg1, abs1
                mods.append(_format_dpp16_ctrl(ctrl))
                mods.append(f"row_mask:0x{row_m:x}")
                mods.append(f"bank_mask:0x{bank_m:x}")
                if bc:
                    mods.append("bound_ctrl:1")
                if fi:
                    mods.append("fi:1")
            elif is_dpp8:
                op_name = name + "_dpp"
                lanes = [(w1 >> (8 + 3 * i)) & 7 for i in range(8)]
                s0 = _vreg(src0_reg, 1)
                s1 = _vreg(vsrc1, 1)
                mods.append(f"dpp8:[{','.join(str(x) for x in lanes)}]")
                if src0 == 234:
                    mods.append("fi:1")
            else:  # SDWA
                op_name = name + "_sdwa"
                dst_sel = (w1 >> 8) & 7
                dst_unused = (w1 >> 11) & 3
                clamp = (w1 >> 13) & 1
                omod = (w1 >> 14) & 3
                src0_sel = (w1 >> 16) & 7
                src0_sext = bool((w1 >> 19) & 1)
                src0_neg = bool((w1 >> 20) & 1)
                src0_abs = bool((w1 >> 21) & 1)
                s0_is_sgpr = bool((w1 >> 23) & 1)
                src1_sel = (w1 >> 24) & 7
                src1_sext = bool((w1 >> 27) & 1)
                src1_neg = bool((w1 >> 28) & 1)
                src1_abs = bool((w1 >> 29) & 1)
                s1_is_sgpr = bool((w1 >> 31) & 1)
                s0 = _src(src0_reg, 1, None) if s0_is_sgpr else _vreg(src0_reg, 1)
                s1 = _src(vsrc1, 1, None) if s1_is_sgpr else _vreg(vsrc1, 1)
                if src0_sext:
                    s0["sext"] = True
                if src1_sext:
                    s1["sext"] = True
                s0["neg"], s0["abs"] = src0_neg, src0_abs
                s1["neg"], s1["abs"] = src1_neg, src1_abs
                if clamp:
                    mods.append("clamp")
                if omod in _OMOD_MODS:
                    mods.append(_OMOD_MODS[omod])
                mods.append(f"dst_sel:{_SDWA_SELS.get(dst_sel, 'DWORD')}")
                mods.append(f"dst_unused:{_SDWA_UNUSED.get(dst_unused, 'UNUSED_PAD')}")
                mods.append(f"src0_sel:{_SDWA_SELS.get(src0_sel, 'DWORD')}")
                mods.append(f"src1_sel:{_SDWA_SELS.get(src1_sel, 'DWORD')}")
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "VDST":
                    args.append(_vreg(vdst, w))
                elif fn == "VCC":
                    args.append(_vcc_operand(wave32))
                elif fn == "SRC0":
                    args.append(s0)
                elif fn == "VSRC1":
                    args.append(s1)
            return finish(op_name, args, mods, 8)

        has_lit = (src0 == 255) or (opc in _VOP2_MANDATORY_LITERAL)
        lit = None
        nbytes = 4
        if has_lit:
            if not need(8):
                return None, 4
            lit = struct.unpack_from("<I", code, pc + 4)[0]
            nbytes = 8
        op_name = name if opc in _VOP2_MANDATORY_LITERAL else name + "_e32"
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDST":
                args.append(_vreg(vdst, w))
            elif fn == "VCC":
                args.append(_vcc_operand(wave32))
            elif fn == "SRC0":
                args.append(_src(src0, w, lit))
            elif fn == "VSRC1":
                args.append(_vreg(vsrc1, w))
            elif fn in ("SIMM32", "LITERAL"):
                args.append({"k": "imm", "v": 0 if lit is None else lit})
        return finish(op_name, args, mods, nbytes)

    # ---- 2. Bits [31:30] == 0b10: SOP1 / SOPC / SOPP / SOPK / SOP2 ----
    if (w0 >> 30) == 0b10:
        hi9 = w0 >> 23
        if hi9 == 0x17D:  # SOP1
            opc = (w0 >> 8) & 0xFF
            sdst = (w0 >> 16) & 0x7F
            ssrc0 = w0 & 0xFF
            entry = T.SOP1.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            lit = None
            nbytes = 4
            if ssrc0 == 255:
                if not need(8):
                    return None, 4
                lit = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "SDST":
                    args.append(_sreg(sdst, w))
                elif fn == "SSRC0":
                    args.append(_src(ssrc0, w, lit))
            return finish(name, args, [], nbytes)

        if hi9 == 0x17E:  # SOPC
            opc = (w0 >> 16) & 0x7F
            ssrc0 = w0 & 0xFF
            ssrc1 = (w0 >> 8) & 0xFF
            entry = T.SOPC.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            lit = None
            nbytes = 4
            if ssrc0 == 255 or ssrc1 == 255:
                if not need(8):
                    return None, 4
                lit = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "SSRC0":
                    args.append(_src(ssrc0, w, lit))
                elif fn == "SSRC1":
                    args.append(_src(ssrc1, w, lit))
            return finish(name, args, [], nbytes)

        if hi9 == 0x17F:  # SOPP
            opc = (w0 >> 16) & 0x7F
            simm16 = w0 & 0xFFFF
            entry = T.SOPP.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            if name in _SOPP_NO_OPERAND:
                return finish(name, [], [], 4)
            if name == "s_waitcnt":
                return finish(name, _format_waitcnt(simm16), [], 4)
            if name == "s_waitcnt_vscnt":
                # s_waitcnt_vscnt is SOPK, not SOPP; kept defensive.
                return finish(name, [{"k": "imm", "v": simm16}], [], 4)
            if name == "s_sendmsg" or name == "s_sendmsghalt":
                return finish(name, [{"k": "special", "t": _format_sendmsg(simm16)}], [], 4)
            if name in _SOPP_BRANCHES:
                return finish(name, [{"k": "imm", "v": _s16(simm16)}], [], 4)
            if name == "s_clause":
                return finish(name, [{"k": "imm", "v": simm16}], [], 4)
            return finish(name, [{"k": "imm", "v": simm16}], [], 4)

        if (w0 >> 28) == 0xB:  # SOPK
            opc = (w0 >> 23) & 0x1F
            sdst = (w0 >> 16) & 0x7F
            simm16 = w0 & 0xFFFF
            entry = T.SOPK.get(opc)
            if entry is None:
                return None, 4
            name, spec = entry
            lit = None
            nbytes = 4
            if opc in _SOPK_MANDATORY_LITERAL:
                if not need(8):
                    return None, 4
                lit = struct.unpack_from("<I", code, pc + 4)[0]
                nbytes = 8
            if name in ("s_setreg_b32", "s_getreg_b32"):
                hw = {"k": "special", "t": _format_hwreg(simm16)}
                reg = _sreg(sdst, 1)
                args = [hw, reg] if name == "s_setreg_b32" else [reg, hw]
                return finish(name, args, [], 4)
            if name == "s_setreg_imm32_b32":
                return finish(name, [{"k": "special", "t": _format_hwreg(simm16)},
                                     {"k": "imm", "v": 0 if lit is None else lit}], [], nbytes)
            if name in ("s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt"):
                return finish(name, [_sreg(sdst, 1), {"k": "imm", "v": simm16}], [], 4)
            if name == "s_call_b64":
                return finish(name, [_sreg(sdst, 2), {"k": "imm", "v": _s16(simm16)}], [], 4)
            args = []
            for fn, nr, _ in spec:
                w = _width(nr, wave32)
                if fn == "SDST":
                    args.append(_sreg(sdst, w))
                elif fn == "SIMM16":
                    val = _s16(simm16) if name in ("s_movk_i32", "s_cmovk_i32", "s_cmpk_eq_i32",
                                                   "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32",
                                                   "s_cmpk_lt_i32", "s_cmpk_le_i32", "s_addk_i32",
                                                   "s_mulk_i32") else simm16
                    args.append({"k": "imm", "v": val})
            return finish(name, args, [], nbytes)

        # SOP2
        opc = (w0 >> 23) & 0x7F
        sdst = (w0 >> 16) & 0x7F
        ssrc1 = (w0 >> 8) & 0xFF
        ssrc0 = w0 & 0xFF
        entry = T.SOP2.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        lit = None
        nbytes = 4
        if ssrc0 == 255 or ssrc1 == 255:
            if not need(8):
                return None, 4
            lit = struct.unpack_from("<I", code, pc + 4)[0]
            nbytes = 8
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "SDST":
                args.append(_sreg(sdst, w))
            elif fn == "SSRC0":
                args.append(_src(ssrc0, w, lit))
            elif fn == "SSRC1":
                args.append(_src(ssrc1, w, lit))
        return finish(name, args, [], nbytes)

    # ---- 3. Bits [31:26] == 0b11xxxx ----
    hi6 = w0 >> 26

    if hi6 == 0x32:  # VINTRP (32-bit!)
        opc = (w0 >> 16) & 3
        vdst = (w0 >> 18) & 0xFF
        attr = (w0 >> 10) & 0x3F
        chan = "xyzw"[(w0 >> 8) & 3]
        vsrc = w0 & 0xFF
        entry = T.VINTRP.get(opc)
        if entry is None:
            return None, 4
        name, _ = entry
        if name == "v_interp_mov_f32":
            p = {0: "p10", 1: "p20", 2: "p0"}.get(vsrc & 3, f"p{vsrc}")
            args = [_vreg(vdst, 1), {"k": "interp", "t": p}, {"k": "attr", "a": attr, "c": (w0 >> 8) & 3}]
        else:
            args = [_vreg(vdst, 1), _vreg(vsrc, 1), {"k": "attr", "a": attr, "c": (w0 >> 8) & 3}]
        return finish(name, args, [], 4)

    # All remaining encodings are at least 64 bits.
    if not need(8):
        return None, 4
    w1 = struct.unpack_from("<I", code, pc + 4)[0]

    if hi6 == 0x33:  # VOP3P (64-bit, +literal)
        opc = (w0 >> 16) & 0x7F
        vdst = w0 & 0xFF
        neg_hi = (w0 >> 8) & 7
        op_sel = (w0 >> 11) & 7
        op_sel_hi_2 = (w0 >> 14) & 1
        clamp = (w0 >> 15) & 1
        src0 = w1 & 0x1FF
        src1 = (w1 >> 9) & 0x1FF
        src2 = (w1 >> 18) & 0x1FF
        op_sel_hi_01 = (w1 >> 27) & 3
        op_sel_hi = op_sel_hi_01 | (op_sel_hi_2 << 2)
        neg_lo = (w1 >> 29) & 7
        entry = T.VOP3P.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        lit = None
        nbytes = 8
        nsrc = sum(1 for fn, _, _ in spec if fn in ("SRC0", "SRC1", "SRC2"))
        srcs_codes = [src0, src1, src2][:nsrc]
        if 255 in srcs_codes:
            if not need(12):
                return None, 4
            lit = struct.unpack_from("<I", code, pc + 8)[0]
            nbytes = 12
        is_mix = name.startswith("v_fma_mix")
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDST":
                args.append(_vreg(vdst, w))
            elif fn in ("SRC0", "SRC1", "SRC2"):
                idx = int(fn[-1])
                args.append(_src(srcs_codes[idx], w, lit,
                                 neg=is_mix and bool(neg_lo & (1 << idx)),
                                 absm=is_mix and bool(neg_hi & (1 << idx))))
        mods = []
        sel_bits = [(op_sel >> i) & 1 for i in range(nsrc)]
        selhi_bits = [(op_sel_hi >> i) & 1 for i in range(nsrc)]
        neglo_bits = [(neg_lo >> i) & 1 for i in range(nsrc)]
        neghi_bits = [(neg_hi >> i) & 1 for i in range(nsrc)]
        # v_fma_mix* uses op_sel/op_sel_hi with default [0,...]; standard packed ops default op_sel_hi to [1,...].
        if any(sel_bits):
            mods.append(f"op_sel:[{','.join(str(b) for b in sel_bits)}]")
        default_hi = [0] * nsrc if is_mix else [1] * nsrc
        if selhi_bits != default_hi:
            mods.append(f"op_sel_hi:[{','.join(str(b) for b in selhi_bits)}]")
        if not is_mix and any(neglo_bits):
            mods.append(f"neg_lo:[{','.join(str(b) for b in neglo_bits)}]")
        if not is_mix and any(neghi_bits):
            mods.append(f"neg_hi:[{','.join(str(b) for b in neghi_bits)}]")
        if clamp:
            mods.append("clamp")
        return finish(name, args, mods, nbytes)

    if hi6 == 0x35:  # VOP3 / VOP3B (64-bit, +literal)
        opc = (w0 >> 16) & 0x3FF
        vdst = w0 & 0xFF
        clamp = (w0 >> 15) & 1
        src0 = w1 & 0x1FF
        src1 = (w1 >> 9) & 0x1FF
        src2 = (w1 >> 18) & 0x1FF
        omod = (w1 >> 27) & 3
        neg = (w1 >> 29) & 7
        is_vop3b = opc in T.VOP3B
        entry = T.VOP3B.get(opc) if is_vop3b else T.VOP3.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        if opc == 0x140 and mcpu.startswith("gfx101"):
            name = "v_mad_legacy_f32"
        sdst = (w0 >> 8) & 0x7F
        abs_bits = 0 if is_vop3b else ((w0 >> 8) & 7)
        op_sel = 0 if is_vop3b else ((w0 >> 11) & 0xF)
        nsrc = sum(1 for fn, _, _ in spec if fn in ("SRC0", "SRC1", "SRC2"))
        srcs_codes = [src0, src1, src2][:nsrc]
        lit = None
        nbytes = 8
        if 255 in srcs_codes:
            if not need(12):
                return None, 4
            lit = struct.unpack_from("<I", code, pc + 8)[0]
            nbytes = 12
        # Add _e64 suffix for opcodes that also have a 32-bit VOPC/VOP2/VOP1 form (0..511), matching llvm-mc.
        has_e32 = (opc < 0x100 and opc in T.VOPC) or (0x100 <= opc < 0x140 and (opc - 0x100) in T.VOP2) or \
                  (0x180 <= opc < 0x200 and (opc - 0x180) in T.VOP1)
        op_name = (name + "_e64") if (has_e32 and not name.startswith(("v_mad_", "v_fma_"))) else name
        if opc == 0x100 + 0x06 or opc == 0x100 + 0x1F:
            op_name = name + "_e64"
        args = []
        mods = []
        is_interp_vop3 = opc in (0x200, 0x201, 0x202, 0x342, 0x343, 0x35A)
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDST":
                if opc < 0x100:  # VOPC promoted to VOP3: VDST field actually holds SDST (7-bit)
                    if not name.startswith("v_cmpx_"):
                        args.append(_sreg(vdst & 0x7F, 1 if wave32 else 2))
                elif name in ("v_readlane_b32", "v_readfirstlane_b32"):
                    args.append(_sreg(vdst & 0x7F, 1))
                else:
                    args.append(_vreg(vdst, w))
            elif fn == "SDST":
                args.append(_sreg(sdst, 1 if wave32 else 2))
            elif fn == "SRC0":
                if is_interp_vop3:
                    args.append({"k": "attr", "a": src0 & 0x3F, "c": (src0 >> 6) & 3})
                    if opc in (0x342, 0x343, 0x35A) and ((src0 >> 8) & 1):
                        mods.append("high")
                else:
                    args.append(_src(src0, w, lit, neg=bool(neg & 1), absm=bool(abs_bits & 1)))
            elif fn == "SRC1":
                if opc == 0x202:  # v_interp_mov_f32_e64: param selector p10/p20/p0
                    p = {0: "p10", 1: "p20", 2: "p0"}.get(src1 & 3, f"p{src1}")
                    args.append({"k": "interp", "t": p})
                else:
                    args.append(_src(src1, w, lit, neg=bool(neg & 2), absm=bool(abs_bits & 2)))
            elif fn == "SRC2":
                args.append(_src(src2, w, lit, neg=bool(neg & 4), absm=bool(abs_bits & 4)))
        if op_sel:
            # op_sel prints [s0,s1,...,dst] (nsrc + 1 bits, or nsrc for VOPC).
            nbits = nsrc if opc < 0x100 else (nsrc + 1)
            bits = [(op_sel >> i) & 1 for i in range(nsrc)]
            if nbits > nsrc:
                bits.append((op_sel >> 3) & 1)
            mods.append(f"op_sel:[{','.join(str(b) for b in bits)}]")
        if clamp:
            mods.append("clamp")
        if omod in _OMOD_MODS:
            mods.append(_OMOD_MODS[omod])
        return finish(op_name, args, mods, nbytes)

    if hi6 == 0x36:  # DS (64-bit)
        opc = (w0 >> 18) & 0xFF
        offset0 = w0 & 0xFF
        offset1 = (w0 >> 8) & 0xFF
        gds = (w0 >> 17) & 1
        addr = w1 & 0xFF
        data0 = (w1 >> 8) & 0xFF
        data1 = (w1 >> 16) & 0xFF
        vdst = (w1 >> 24) & 0xFF
        entry = T.DS.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDST":
                args.append(_vreg(vdst, w))
            elif fn == "ADDR":
                args.append(_vreg(addr, w))
            elif fn == "DATA0":
                args.append(_vreg(data0, w))
            elif fn == "DATA1":
                args.append(_vreg(data1, w))
        mods = []
        is_2addr = ("2_" in name or "2st64_" in name)
        if name == "ds_swizzle_b32":
            mods.append(_format_ds_swizzle(offset0 | (offset1 << 8)))
        elif is_2addr:
            if offset0:
                mods.append(f"offset0:{offset0}")
            if offset1:
                mods.append(f"offset1:{offset1}")
        else:
            off16 = offset0 | (offset1 << 8)
            if off16:
                mods.append(f"offset:{off16}")
        if gds:
            mods.append("gds")
        return finish(name, args, mods, 8)

    if hi6 == 0x37:  # FLAT / SCRATCH / GLOBAL (64-bit)
        seg = (w0 >> 14) & 3
        opc = (w0 >> 18) & 0x7F
        off12 = w0 & 0xFFF
        dlc = (w0 >> 12) & 1
        lds = (w0 >> 13) & 1
        glc = (w0 >> 16) & 1
        slc = (w0 >> 17) & 1
        addr = w1 & 0xFF
        data = (w1 >> 8) & 0xFF
        saddr = (w1 >> 16) & 0x7F
        vdst = (w1 >> 24) & 0xFF
        tbl = {0: T.FLAT, 1: T.SCRATCH, 2: T.GLOBAL}.get(seg)
        if tbl is None or opc not in tbl:
            return None, 4
        name, spec = tbl[opc]
        is_atomic = "_atomic_" in name
        has_saddr = seg in (1, 2)
        saddr_off = saddr in (0x7D, 0x7F)  # GFX10 null (125) or legacy 127 => off (64-bit vaddr)
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDST":
                if is_atomic and not glc:
                    continue
                args.append(_vreg(vdst, w))
            elif fn == "ADDR":
                aw = 2 if (seg == 0 or (seg == 2 and saddr_off)) else 1
                args.append(_vreg(addr, aw))
            elif fn == "DATA":
                args.append(_vreg(data, w))
            elif fn == "SADDR" and has_saddr:
                if saddr_off:
                    args.append({"k": "special", "t": "off", "neg": False, "abs": False})
                else:
                    args.append(_sreg(saddr, 2 if seg == 2 else 1))
        mods = []
        off_s = _s12(off12) if seg in (1, 2) else off12
        if off_s:
            mods.append(f"offset:{off_s}")
        if glc:
            mods.append("glc")
        if slc:
            mods.append("slc")
        if dlc:
            mods.append("dlc")
        if lds:
            mods.append("lds")
        return finish(name, args, mods, 8)

    if hi6 == 0x38:  # MUBUF (64-bit)
        opc = (w0 >> 18) & 0xFF
        off12 = w0 & 0xFFF
        offen = (w0 >> 12) & 1
        idxen = (w0 >> 13) & 1
        glc = (w0 >> 14) & 1
        dlc = (w0 >> 15) & 1
        lds = (w0 >> 16) & 1
        vaddr = w1 & 0xFF
        vdata = (w1 >> 8) & 0xFF
        srsrc = ((w1 >> 16) & 0x1F) * 4
        slc = (w1 >> 22) & 1
        tfe = (w1 >> 23) & 1
        soff = (w1 >> 24) & 0xFF
        entry = T.MUBUF.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDATA":
                args.append(_vreg(vdata, w))
            elif fn == "VADDR":
                if not offen and not idxen:
                    args.append({"k": "special", "t": "off", "neg": False, "abs": False})
                else:
                    args.append(_vreg(vaddr, 2 if (offen and idxen) else 1))
            elif fn == "SRSRC":
                args.append(_sreg(srsrc, 4))
            elif fn == "SOFFSET":
                args.append(_src(soff, 1, None))
        mods = []
        if idxen:
            mods.append("idxen")
        if offen:
            mods.append("offen")
        if off12:
            mods.append(f"offset:{off12}")
        if glc:
            mods.append("glc")
        if slc:
            mods.append("slc")
        if dlc:
            mods.append("dlc")
        if lds:
            mods.append("lds")
        if tfe:
            mods.append("tfe")
        return finish(name, args, mods, 8)

    if hi6 == 0x3A:  # MTBUF (64-bit)
        opc = ((w0 >> 16) & 0x7) | (((w1 >> 21) & 1) << 3)
        off12 = w0 & 0xFFF
        offen = (w0 >> 12) & 1
        idxen = (w0 >> 13) & 1
        glc = (w0 >> 14) & 1
        dlc = (w0 >> 15) & 1
        fmt_code = (w0 >> 19) & 0x7F
        vaddr = w1 & 0xFF
        vdata = (w1 >> 8) & 0xFF
        srsrc = ((w1 >> 16) & 0x1F) * 4
        slc = (w1 >> 22) & 1
        tfe = (w1 >> 23) & 1
        soff = (w1 >> 24) & 0xFF
        entry = T.MTBUF.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        args = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "VDATA":
                args.append(_vreg(vdata, w))
            elif fn == "VADDR":
                if not offen and not idxen:
                    args.append({"k": "special", "t": "off", "neg": False, "abs": False})
                else:
                    args.append(_vreg(vaddr, 2 if (offen and idxen) else 1))
            elif fn == "SRSRC":
                args.append(_sreg(srsrc, 4))
            elif fn == "SOFFSET":
                args.append(_src(soff, 1, None))
        mods = []
        fmt_name = _GFX10_FORMATS.get(fmt_code)
        if fmt_name and fmt_code != 1:  # BUF_FMT_8_UNORM (code 1) is the GFX10 default
            short = fmt_name.replace("GFX10_FORMAT_", "BUF_FMT_")
            mods.append(f"format:[{short}]")
        elif fmt_code != 1:
            mods.append(f"format:{fmt_code}")
        if idxen:
            mods.append("idxen")
        if offen:
            mods.append("offen")
        if off12:
            mods.append(f"offset:{off12}")
        if glc:
            mods.append("glc")
        if slc:
            mods.append("slc")
        if dlc:
            mods.append("dlc")
        if tfe:
            mods.append("tfe")
        return finish(name, args, mods, 8)

    if hi6 == 0x3C:  # MIMG (64-bit + up to 3 NSA dwords)
        nsa_dwords = (w0 >> 1) & 3
        dim = (w0 >> 3) & 7
        dlc = (w0 >> 7) & 1
        dmask = (w0 >> 8) & 0xF
        unorm = (w0 >> 12) & 1
        glc = (w0 >> 13) & 1
        r128 = (w0 >> 15) & 1
        tfe = (w0 >> 16) & 1
        lwe = (w0 >> 17) & 1
        opc = ((w0 >> 18) & 0x7F) | ((w0 & 1) << 7)
        vaddr0 = w1 & 0xFF
        vdata = (w1 >> 8) & 0xFF
        srsrc = ((w1 >> 16) & 0x1F) * 4
        ssamp = ((w1 >> 21) & 0x1F) * 4
        slc = (w0 >> 25) & 1
        a16 = (w1 >> 30) & 1
        d16 = (w1 >> 31) & 1
        entry = T.MIMG.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        nbytes = 8 + 4 * nsa_dwords
        if not need(nbytes):
            return None, 4
        n_addr = _mimg_addr_count(name, dim, bool(a16))
        if nsa_dwords:
            # First address VGPR is vaddr0; remaining bytes sit in the trailing NSA dwords.
            extra_bytes = code[pc + 8:pc + nbytes]
            vregs = [vaddr0] + list(extra_bytes[:max(0, n_addr - 1)])
            vaddr_op = {"k": "vlist", "r": vregs}
        else:
            vaddr_op = _vreg(vaddr0, n_addr)
        n_data = _mimg_data_count(name, dmask, bool(d16), bool(tfe))
        rsrc_w = 4 if (r128 or name.startswith("image_bvh")) else 8
        args = []
        for fn, nr, _ in spec:
            if fn == "VDATA":
                args.append(_vreg(vdata, n_data))
            elif fn == "VADDR":
                args.append(vaddr_op)
            elif fn == "SRSRC":
                args.append(_sreg(srsrc, rsrc_w))
            elif fn == "SSAMP":
                args.append(_sreg(ssamp, 4))
        mods = []
        if dmask and not name.startswith("image_bvh"):
            mods.append(f"dmask:0x{dmask:x}")
        if not name.startswith("image_bvh"):
            mods.append(f"dim:{_MIMG_DIMS.get(dim, 'SQ_RSRC_IMG_2D')}")
            if unorm:
                mods.append("unorm")
        if glc:
            mods.append("glc")
        if slc:
            mods.append("slc")
        if r128 and not name.startswith("image_bvh"):
            mods.append("r128")
        if tfe:
            mods.append("tfe")
        if lwe:
            mods.append("lwe")
        if a16:
            mods.append("a16")
        if d16:
            mods.append("d16")
        if dlc:
            mods.append("dlc")
        return finish(name, args, mods, nbytes)

    if hi6 == 0x3D:  # SMEM (64-bit)
        sbase = (w0 & 0x3F) * 2
        sdata = (w0 >> 6) & 0x7F
        dlc = (w0 >> 14) & 1
        glc = (w0 >> 16) & 1
        opc = (w0 >> 18) & 0xFF
        off21 = w1 & 0x1FFFFF
        soff = (w1 >> 25) & 0x7F
        entry = T.SMEM.get(opc)
        if entry is None:
            return None, 4
        name, spec = entry
        is_buf = "buffer" in name
        args = []
        mods = []
        for fn, nr, _ in spec:
            w = _width(nr, wave32)
            if fn == "SDATA":
                args.append(_sreg(sdata, w))
            elif fn == "SBASE":
                args.append(_sreg(sbase, w))
            elif fn in ("OFFSET", "SOFFSET"):
                pass
        # SMEM offset operands: soff != 125 (null) => register offset (+ optional offset:IMM); else immediate (or null if 0).
        if any(fn in ("OFFSET", "SOFFSET") for fn, _, _ in spec):
            imm_val = off21 if is_buf else _s21(off21)
            if soff == 125:  # null => immediate offset (or null when 0)
                if imm_val == 0:
                    args.append({"k": "special", "t": "null", "neg": False, "abs": False})
                else:
                    args.append({"k": "imm", "v": imm_val})
            else:
                args.append(_sreg(soff, 1))
                if imm_val > 0:
                    mods.append(f"offset:{imm_val:#x}")
                elif imm_val < 0:
                    mods.append(f"offset:-0x{-imm_val:x}")
        if glc:
            mods.append("glc")
        if dlc:
            mods.append("dlc")
        return finish(name, args, mods, 8)

    if hi6 == 0x3E:  # EXP (64-bit)
        en = w0 & 0xF
        tgt = (w0 >> 4) & 0x3F
        compr = (w0 >> 10) & 1
        done = (w0 >> 11) & 1
        vm = (w0 >> 12) & 1
        vsrcs = [(w1 >> (8 * i)) & 0xFF for i in range(4)]
        if tgt <= 7:
            tname = f"mrt{tgt}"
        elif tgt == 8:
            tname = "mrtz"
        elif tgt == 9:
            tname = "null"
        elif 12 <= tgt <= 16:
            tname = f"pos{tgt - 12}"
        elif tgt == 20:
            tname = "prim"
        elif 32 <= tgt <= 63:
            tname = f"param{tgt - 32}"
        else:
            tname = f"tgt{tgt}"
        args = [{"k": "target", "t": tname}]
        if compr:
            # Compressed export: en bit 0/1 enables vsrc0 (printed twice), bit 2/3 enables vsrc1 (printed twice).
            for pair in (0, 1):
                enabled = bool((en >> (2 * pair)) & 3)
                op = _vreg(vsrcs[pair], 1) if enabled else {"k": "special", "t": "off", "neg": False, "abs": False}
                args.append(op)
                args.append(dict(op))
        else:
            for i in range(4):
                if (en >> i) & 1:
                    args.append(_vreg(vsrcs[i], 1))
                else:
                    args.append({"k": "special", "t": "off", "neg": False, "abs": False})
        mods = []
        if done:
            mods.append("done")
        if compr:
            mods.append("compr")
        if vm:
            mods.append("vm")
        return finish("exp", args, mods, 8)

    return None, 4


def _format_operand(a: dict, hex_imm: bool = False) -> str:
    k = a["k"]
    if k in ("vgpr", "sgpr"):
        p = "v" if k == "vgpr" else "s"
        r, n = a["r"], a.get("n", 1)
        s = f"{p}{r}" if n == 1 else f"{p}[{r}:{r + n - 1}]"
    elif k == "vlist":
        s = "[" + ",".join(f"v{r}" for r in a["r"]) + "]"
    elif k == "slist":
        s = "[" + ",".join(f"s{r}" if isinstance(r, int) else r for r in a["r"]) + "]"
    elif k == "attr":
        s = f"attr{a['a']}.{'xyzw'[a['c']]}"
    elif k in ("special", "target", "interp", "ttmp", "raw"):
        s = a["t"]
    elif k == "imm":
        v = a["v"]
        s = f"0x{v:x}" if hex_imm or v >= 100 else str(v)
    elif k == "fimm":
        v = a["v"]
        s = next((txt for code, txt in _INLINE_FLOAT_STR.items() if _INLINE_FLOAT[code] == v), repr(v))
    else:
        s = str(a)

    if a.get("sext"):
        s = f"sext({s})"
    if a.get("abs"):
        s = f"|{s}|"
    if a.get("neg"):
        if k in ("imm", "fimm"):
            s = f"neg({s})"
        else:
            s = f"-{s}"
    return s


def format_insn(ins: dict) -> str:
    """Format one decoded instruction dict as an LLVM-compatible `.s` line."""
    op = ins["op"]
    args = ins["args"]
    mods = ins["mods"]
    hex_imm = op == "s_ttracedata_imm"
    if op == "exp" and args:
        head = f"{op} {_format_operand(args[0], hex_imm)} " + ", ".join(
            _format_operand(x, hex_imm) for x in args[1:])
    elif args:
        head = f"{op} " + ", ".join(_format_operand(x, hex_imm) for x in args)
    else:
        head = op
    if mods:
        head += " " + " ".join(mods)
    enc_bytes = bytes.fromhex(ins["enc"])
    enc_str = ",".join(f"0x{b:02x}" for b in enc_bytes)
    return f"\t{head:<40s}; encoding: [{enc_str}]"


def _iter_insns(code: bytes, wave32: bool, code_start: int, mcpu: str) -> Iterator[dict | None]:
    """Yield each decoded instruction, or None for each invalid word counted."""
    off = 0
    while off < len(code):
        ins, nbytes = _decode_one(code, off, wave32, mcpu)
        if ins is None:
            off += max(4, nbytes)
        else:
            ins["pc"] = code_start + off
            off += nbytes
        yield ins


def decode_bytes(code: bytes, *, mcpu: str = "gfx1013", wave32: bool = True, code_start: int = 0, name: str = "") -> dict:
    """Decode raw RDNA2 machine code into the exact dict layout `shader_ir.parse_file()` returns."""
    insns = []
    invalid = 0
    for ins in _iter_insns(code, wave32, code_start, mcpu):
        if ins is None:
            invalid += 1
        else:
            insns.append(ins)
    return {"name": name, "code_start": code_start, "insns": insns, "invalid": invalid}


def disasm_bytes(code: bytes, *, mcpu: str = "gfx1013", wave32: bool = True, code_start: int = 0, name: str = "") -> tuple[str, int]:
    """Decode raw RDNA2 machine code into `(llvm_compatible_text, invalid_word_count)`.

    The listing contains only instructions. `name` is accepted for caller compatibility.
    """
    invalid = 0
    with StringIO() as listing:
        for ins in _iter_insns(code, wave32, code_start, mcpu):
            if ins is None:
                invalid += 1
            else:
                listing.write(f"{format_insn(ins)}\n")
        return listing.getvalue(), invalid
