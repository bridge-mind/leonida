"""Scan the shader corpus for features the D3D12 -> Vulkan layer may need (docs/d3d12_vulkan.md §8.2 #10, #14, #16).

    python -I tools/feature_scan.py <work>/shaders [--out feature_scan.json]

Reads what the shaders stage of make_game.py leaves in <work>/shaders, and prints counts only:

- dxbc/index.json and the .dxbc files: the DXIL program kind of every shader (hull, domain and amplification
  stages would show up here), the SFI0 feature flags (D3D_SHADER_REQUIRES_* from the Windows SDK's d3d12shader.h,
  among them WAVE_MMA and SHADING_RATE), and the output-signature (OSG1) stream index of every element: stream
  output to a stream other than 0 needs a geometry shader writing that stream. The PSV0 resource bindings: structured
  UAVs with a counter (append/consume buffers) and typed buffer SRVs/UAVs (#16).
- dxbc/disasm/*.s (shader_disasm.py): the native RDNA2 instructions that mean wave MMA (v_mfma*, v_wmma*,
  v_smfmac*) or GDS / ordered-count traffic (the hardware path NGG stream output uses on RDNA2). Typed buffer
  accesses (#16): the MTBUF instruction formats, and for MUBUF where V# word 3 (the format) comes from, each checked
  against the translator's typed-buffer table (rdna2spirv.Gen.FORMATS).
- spirv/index.json (rdna2spirv.py): uses_code_base per stage. A graphics PSO can only need two code bases
  (§5.2) if two of its stages use one; this gives the per-stage counts. Which stages are paired in one PSO is in
  the executable, not in shaders.rpf.

InnerCoverage metadata is cross-tabulated by stage: SFI0 set/clear/unknown versus ISG1
present/absent/unknown. Unknown includes missing or truncated parts; unreadable containers
remain in the existing unreadable count. DXIL permits InnerCoverage outside the signature,
so ISG1 absence is not evidence of non-use. These counts do not identify an Xbox input VGPR.

Standard library only. Nothing from the game is printed or written: names, code and data stay in the work folder.
Usage: `python -I tools/feature_scan.py --help` lists the options.
"""
import argparse
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shader_carve import parts as dxbc_part_ranges  # noqa: E402

# d3d12shader.h (Windows SDK 10.0.26100.0), D3D_SHADER_REQUIRES_*: the SFI0 part is this 64-bit mask.
SFI0_FLAGS = {
    0x1: "DOUBLES", 0x2: "EARLY_DEPTH_STENCIL", 0x4: "UAVS_AT_EVERY_STAGE", 0x8: "64_UAVS",
    0x10: "MINIMUM_PRECISION", 0x20: "11_1_DOUBLE_EXTENSIONS", 0x40: "11_1_SHADER_EXTENSIONS",
    0x80: "LEVEL_9_COMPARISON_FILTERING", 0x100: "TILED_RESOURCES", 0x200: "STENCIL_REF", 0x400: "INNER_COVERAGE",
    0x800: "TYPED_UAV_LOAD_ADDITIONAL_FORMATS", 0x1000: "ROVS",
    0x2000: "VIEWPORT_AND_RT_ARRAY_INDEX_FROM_ANY_SHADER_FEEDING_RASTERIZER", 0x4000: "WAVE_OPS",
    0x8000: "INT64_OPS", 0x10000: "VIEW_ID", 0x20000: "BARYCENTRICS", 0x40000: "NATIVE_16BIT_OPS",
    0x80000: "SHADING_RATE", 0x100000: "RAYTRACING_TIER_1_1", 0x200000: "SAMPLER_FEEDBACK",
    0x400000: "ATOMIC_INT64_ON_TYPED_RESOURCE", 0x800000: "ATOMIC_INT64_ON_GROUP_SHARED",
    0x1000000: "DERIVATIVES_IN_MESH_AND_AMPLIFICATION_SHADERS", 0x2000000: "RESOURCE_DESCRIPTOR_HEAP_INDEXING",
    0x4000000: "SAMPLER_DESCRIPTOR_HEAP_INDEXING", 0x8000000: "WAVE_MMA",
    0x10000000: "ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE", 0x20000000: "ADVANCED_TEXTURE_OPS",
    0x40000000: "WRITEABLE_MSAA_TEXTURES", 0x80000000: "SAMPLE_CMP_GRADIENT_OR_BIAS",
    0x100000000: "EXTENDED_COMMAND_INFO",
}
# The ones §8.2 #10 asks about; always printed, also when zero.
WATCHED = ("WAVE_MMA", "SHADING_RATE", "VIEW_ID", "ROVS", "STENCIL_REF", "INNER_COVERAGE", "SAMPLER_FEEDBACK",
           "RAYTRACING_TIER_1_1")
ALL_STAGES = ("vs", "ps", "gs", "hs", "ds", "cs", "ms", "as", "lib")

MMA = re.compile(r"^\s*(v_mfma\w*|v_wmma\w*|v_smfmac\w*)\b")
# Ordered count, GWS and the GS-register adds are the GDS-side instructions NGG stream output relies on. ds_append /
# ds_consume are included as the other counter-style DS ops (LDS unless they carry `gds`, which GDS below catches).
ORDERED = re.compile(r"^\s*(ds_ordered_count|ds_gws_\w+|ds_append|ds_consume|ds_add_gs_reg\w*|ds_sub_gs_reg\w*)\b")
GDS = re.compile(r"^\s*ds_\w+\b[^;]*\bgds\b")
REG_SINGLE = re.compile(r"^([sv])(\d+)$")
REG_RANGE = re.compile(r"^([sv])\[(\d+):(\d+)\]$")
CONST_MOV_OPS = frozenset(("s_mov_b32", "s_mov_b32_e32", "v_mov_b32", "v_mov_b32_e32", "v_mov_b32_e64"))


def parts(d: bytes) -> dict:
    """DXBC part fourcc -> payload."""
    if not d.startswith(b"DXBC"):
        raise ValueError("missing DXBC magic")
    return {tag: d[off:off + size] for tag, (off, size) in dxbc_part_ranges(d).items()}


def signature_offsets(sig: bytes) -> range:
    """Offsets of the 32-byte elements in a complete ISG1 or OSG1 table."""
    if len(sig) < 8:
        raise ValueError("truncated signature header")
    count, first = struct.unpack_from("<II", sig, 0)
    end = first + 32 * count
    if first < 8 or end > len(sig):
        raise ValueError("invalid signature table")
    return range(first, end, 32)


def streams(sig: bytes) -> set:
    """Stream indices from a complete signature, or an empty set for an unusable table."""
    try:
        offsets = signature_offsets(sig)
    except ValueError:
        return set()
    return {struct.unpack_from("<I", sig, offset)[0] for offset in offsets}


# DXC include/dxc/DxilContainer/DxilPipelineStateValidation.h (PSVResourceType) and DxilConstants.h
# (DXIL::ResourceKind), by value.
PSV_RES_TYPES = ("Invalid", "Sampler", "CBV", "SRVTyped", "SRVRaw", "SRVStructured", "UAVTyped", "UAVRaw",
                 "UAVStructured", "UAVStructuredWithCounter")
PSV_RES_KINDS = ("Invalid", "Texture1D", "Texture2D", "Texture2DMS", "Texture3D", "TextureCube", "Texture1DArray",
                 "Texture2DArray", "Texture2DMSArray", "TextureCubeArray", "TypedBuffer", "RawBuffer",
                 "StructuredBuffer", "CBuffer", "Sampler", "TBuffer", "RTAccelerationStructure", "FeedbackTexture2D",
                 "FeedbackTexture2DArray")


def psv0_bindings(psv: bytes) -> list:
    """(resource type, resource kind) of every PSV0 resource binding record.

    The part is: u32 runtime-info size, the runtime info, u32 resource count and, when that is not 0, u32 record
    size and the records. A record is PSVResourceBindInfo0 (type, space, lower and upper bound: 16 bytes) or
    PSVResourceBindInfo1 (plus resource kind and flags: 24 bytes); a 16-byte record has kind "?". Raises ValueError
    when the table does not fit in the part."""
    if len(psv) < 4:
        raise ValueError("PSV0 shorter than its runtime-info size")
    pos = 4 + struct.unpack_from("<I", psv, 0)[0]
    if pos + 4 > len(psv):
        raise ValueError("PSV0 truncated before the resource count")
    count = struct.unpack_from("<I", psv, pos)[0]
    pos += 4
    if not count:
        return []
    if pos + 4 > len(psv):
        raise ValueError("PSV0 truncated before the record size")
    size = struct.unpack_from("<I", psv, pos)[0]
    pos += 4
    if size < 16 or pos + size * count > len(psv):
        raise ValueError("PSV0 resource records do not fit")
    out = []
    for i in range(count):
        at = pos + i * size
        t = struct.unpack_from("<I", psv, at)[0]
        k = struct.unpack_from("<I", psv, at + 16)[0] if size >= 20 else None
        out.append((PSV_RES_TYPES[t] if t < len(PSV_RES_TYPES) else f"type{t}",
                    "?" if k is None else PSV_RES_KINDS[k] if k < len(PSV_RES_KINDS) else f"kind{k}"))
    return out


# §8.2 #16: append/consume buffers are the structured UAVs with a counter; typed buffers are typed SRVs/UAVs of
# kind TypedBuffer (a typed SRV/UAV of a texture kind is an image, not a buffer).
WATCHED_BINDINGS = ("uav_with_counter", "typed_buffer_srv", "typed_buffer_uav")


def watched_binding(res_type: str, kind: str):
    if res_type == "UAVStructuredWithCounter":
        return "uav_with_counter"
    if kind == "TypedBuffer" and res_type in ("SRVTyped", "UAVTyped"):
        return "typed_buffer_srv" if res_type == "SRVTyped" else "typed_buffer_uav"
    return None


def stage_name(row: dict) -> str:
    stage = row.get("stage")
    return stage if isinstance(stage, str) and stage else "?"


def inner_coverage_metadata(p: dict) -> tuple[str, str]:
    """Independent SFI0/ISG1 observations, not an Xbox VGPR mapping.

    DXC DxilConstants.h defines ShaderFeatureInfo_InnerCoverage as 0x400;
    DxilContainer.h defines the signature system value as 70. DXIL.rst marks
    InnerCoverage NotInSig: an absent signature entry does not disprove use.
    Missing/truncated metadata must not become evidence of absence.
    """
    sfi = p.get("SFI0", b"")
    feature = "unknown"
    if len(sfi) >= 8:
        feature = "set" if struct.unpack_from("<Q", sfi)[0] & 0x400 else "clear"
    sig = p.get("ISG1", b"")
    signature = "unknown"
    if len(sig) >= 8:
        count, first = struct.unpack_from("<II", sig)
        if 8 <= first <= len(sig) and count <= (len(sig) - first) // 32:
            signature = "present" if any(struct.unpack_from("<I", sig, first + i * 32 + 12)[0] == 70
                                         for i in range(count)) else "absent"
    return feature, signature


def scan_dxbc(dxbc_dir: str, index: list) -> dict:
    stages = Counter()
    flags = defaultdict(Counter)  # flag -> stage -> shaders
    inner = defaultdict(Counter)  # stage -> independent metadata observations, counts only
    gs_streams = Counter()
    non_zero_stream = Counter()  # stage -> shaders with an output element on a stream other than 0
    records, shaders_with = Counter(), Counter()   # PSV0: (type, kind) -> records; watched class -> shaders
    psv_bad = 0
    missing = 0
    for e in index:
        if not isinstance(e, dict):
            missing += 1
            continue
        stage = stage_name(e)
        stages[stage] += 1
        filename = e.get("file")
        if not isinstance(filename, str) or not filename:
            missing += 1
            continue
        path = os.path.join(dxbc_dir, filename)
        try:
            with open(path, "rb") as f:
                p = parts(f.read())
        except (OSError, ValueError):
            missing += 1
            continue
        feature, signature = inner_coverage_metadata(p)
        inner[stage][f"sfi0_{feature}__isg1_{signature}"] += 1
        if len(p.get("SFI0", b"")) >= 8:
            mask = struct.unpack_from("<Q", p["SFI0"], 0)[0]
            for bit, name in SFI0_FLAGS.items():
                if mask & bit:
                    flags[name][stage] += 1
        try:
            binds = psv0_bindings(p["PSV0"]) if "PSV0" in p else None
        except ValueError:
            binds = None
        if binds is None:
            psv_bad += 1
        else:
            records.update(binds)
            for cls in {watched_binding(t, k) for t, k in binds} - {None}:
                shaders_with[cls] += 1
        out = streams(p.get("OSG1", b""))
        if stage == "gs":
            gs_streams[len(out)] += 1
        if out - {0}:
            non_zero_stream[stage] += 1
    by_type = Counter()
    for (t, _), n in records.items():
        by_type[t] += n
    return {"stages": {s: stages.get(s, 0) for s in ALL_STAGES} | {k: v for k, v in stages.items() if k not in ALL_STAGES},
            "sfi0": {n: dict(flags[n]) for n in sorted(flags)},
            "inner_coverage_metadata": {s: dict(sorted(v.items())) for s, v in sorted(inner.items())},
            "gs_output_stream_count": dict(gs_streams), "output_on_stream_other_than_0": dict(non_zero_stream),
            "psv0": {"records": sum(records.values()),
                     "by_type": dict(by_type),
                     "shaders": {c: shaders_with.get(c, 0) for c in WATCHED_BINDINGS},
                     "records_of": {c: sum(n for (t, k), n in records.items() if watched_binding(t, k) == c)
                                    for c in WATCHED_BINDINGS},
                     "unreadable": psv_bad},
            "unreadable": missing}


def getpc_for_data(ops: list) -> bool:
    """s_getpc_b64 not followed within three instructions by s_setpc_b64: the address is used to read data
    embedded in the code (a code base), not for a far branch (getpc; add/sub; addc/subb; setpc)."""
    return any(op == "s_getpc_b64" and "s_setpc_b64" not in ops[i + 1:i + 4] for i, op in enumerate(ops))


def _parse_imm(tok: str):
    """Parse a decimal or 0x-hex integer immediate, or return None."""
    try:
        return int(tok, 0) & 0xFFFFFFFF
    except ValueError:
        return None


def _written_regs(tok: str) -> list:
    """Return individual sN/vN register names covered by a destination operand token."""
    m = REG_SINGLE.match(tok)
    if m:
        return [tok]
    m = REG_RANGE.match(tok)
    if m:
        prefix, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if lo <= hi <= lo + 32:
            return [f"{prefix}{i}" for i in range(lo, hi + 1)]
    return []


def _has_secondary_sgpr_destination(op: str) -> bool:
    return op.startswith("v_") and ("_co_" in op or op.startswith("v_div_scale"))


def _update_const_regs(const_reg: dict, op: str, operands: list) -> None:
    """Track single-register immediate loads and invalidate any destination register written."""
    if not operands:
        return
    dst = operands[0].split()[0] if operands[0].split() else ""
    if op in CONST_MOV_OPS and len(operands) >= 2 and REG_SINGLE.match(dst):
        src_tok = operands[1].split()[0] if operands[1].split() else ""
        imm = _parse_imm(src_tok)
        if imm is not None:
            const_reg[dst] = imm
            return
    destination_count = 2 if _has_secondary_sgpr_destination(op) else 1
    for operand in operands[:destination_count]:
        tokens = operand.split()
        if tokens:
            for register in _written_regs(tokens[0]):
                const_reg.pop(register, None)


def scan_disasm(disasm_dir: str, stage_of: dict) -> dict:
    mma, ordered, gds, code_data = Counter(), Counter(), Counter(), Counter()
    mma_ops, ordered_ops = Counter(), Counter()
    vperm_shaders = Counter()
    vperm_insns = vperm_imm = vperm_creg = vperm_unres = 0
    vperm_sext_shaders = vperm_sext_insns = 0
    vperm_classes = {"0_7": 0, "8_11": 0, "12": 0, "13_plus": 0}
    files = 0
    for fn in sorted(os.listdir(disasm_dir)):
        if not fn.endswith(".s"):
            continue
        files += 1
        stage = stage_of.get(fn[:-2], "?")
        seen_mma = seen_ord = seen_gds = seen_vperm = seen_vperm_sext = False
        ops = []
        const_reg = {}
        with open(os.path.join(disasm_dir, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                code = line.split(";", 1)[0].strip()
                head = code.split(None, 1)
                if head and not head[0].startswith("."):
                    op = head[0]
                    ops.append(op)
                    operands = [x.strip() for x in head[1].split(",")] if len(head) > 1 else []
                    if op in ("v_perm_b32", "v_perm_b32_e64") and len(operands) >= 4:
                        seen_vperm = True
                        vperm_insns += 1
                        sel_tok = operands[3].split()[0] if operands[3].split() else ""
                        sel_val = _parse_imm(sel_tok)
                        if sel_val is not None:
                            vperm_imm += 1
                        elif sel_tok in const_reg:
                            sel_val = const_reg[sel_tok]
                            vperm_creg += 1
                        else:
                            vperm_unres += 1
                        if sel_val is not None:
                            has_sext = False
                            for shift in (0, 8, 16, 24):
                                b = (sel_val >> shift) & 0xFF
                                if b <= 7:
                                    vperm_classes["0_7"] += 1
                                elif b <= 11:
                                    vperm_classes["8_11"] += 1
                                    has_sext = True
                                elif b == 12:
                                    vperm_classes["12"] += 1
                                else:
                                    vperm_classes["13_plus"] += 1
                            if has_sext:
                                vperm_sext_insns += 1
                                seen_vperm_sext = True
                    _update_const_regs(const_reg, op, operands)
                m = MMA.match(line)
                if m:
                    mma_ops[m.group(1)] += 1
                    seen_mma = True
                m = ORDERED.match(line)
                if m:
                    ordered_ops[m.group(1)] += 1
                    seen_ord = True
                if GDS.match(line):
                    seen_gds = True
        mma[stage] += seen_mma
        ordered[stage] += seen_ord
        gds[stage] += seen_gds
        vperm_shaders[stage] += seen_vperm
        vperm_sext_shaders += int(seen_vperm_sext)
        code_data[stage] += getpc_for_data(ops)
    nz = lambda c: {k: v for k, v in c.items() if v}
    return {"files": files, "wave_mma_shaders": nz(mma), "wave_mma_ops": dict(mma_ops),
            "ordered_or_gws_shaders": nz(ordered), "ordered_or_gws_ops": dict(ordered_ops), "gds_shaders": nz(gds),
            "getpc_for_data_shaders": nz(code_data),
            "v_perm_b32": {
                "shaders": nz(vperm_shaders),
                "insns": vperm_insns,
                "immediate": vperm_imm,
                "const_reg": vperm_creg,
                "unresolved": vperm_unres,
                "sign_extend_8_11_shaders": vperm_sext_shaders,
                "sign_extend_8_11_insns": vperm_sext_insns,
                "selector_classes": vperm_classes,
            }}


# Scalar ops whose first operand is a source, not an SGPR they write.
NO_SDST = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_waitcnt", "s_endpgm", "s_store", "s_buffer_store",
           "s_scratch_store", "s_setpc", "s_sendmsg", "s_nop", "s_barrier", "s_dcache", "s_icache", "s_gl1_inv",
           "s_setprio", "s_sleep", "s_ttrace", "s_set_gpr_idx", "s_setreg", "s_inst_prefetch", "s_round_mode",
           "s_denorm_mode", "s_clause", "s_trap", "s_wakeup", "s_code_end", "s_atc_probe", "s_setkill", "s_incperflevel",
           "s_decperflevel", "s_version")
BRANCH_OPS = frozenset(("s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
                        "s_cbranch_execz", "s_cbranch_execnz"))
TYPED_MUBUF = re.compile(r"^buffer_(load|store)_format_(d16_)?(x|xy|xyz|xyzw)$")
TYPED_MTBUF = re.compile(r"^tbuffer_(load|store)_format_(d16_)?(x|xy|xyz|xyzw)$")


def translator_formats():
    """(codes the translator models, GFX10 code -> format name), read from rdna2spirv itself so the scan follows the
    table as it changes. Modelled: the typed-buffer table and INVALID (0), an unbound descriptor, which reads 0 on the
    hardware and in the translator."""
    import rdna2spirv
    return frozenset(rdna2spirv.Gen.FORMATS) | {0}, {c: n.replace("GFX10_FORMAT_", "") for c, n in
                                                rdna2spirv.GFX10_FORMATS.items()}


def sgpr_writes(ins: dict) -> range:
    """The SGPRs an instruction writes (as far as a V# word can be written: scalar destinations, v_readlane and
    v_readfirstlane, the SGPR mask of a VOP3 compare and the carry of a VOP3b add)."""
    op, args = ins["op"], ins["args"]
    dst = None
    if op.startswith("s_") and not op.startswith(NO_SDST):
        dst = args[0] if args else None
    elif op.startswith(("v_readlane", "v_readfirstlane")) or (op.startswith("v_cmp") and not op.startswith("v_cmpx")
                                                               and op.endswith("_e64")):
        dst = args[0] if args else None
    elif len(args) > 1 and _has_secondary_sgpr_destination(op):
        dst = args[1]
    if dst and dst.get("k") == "sgpr":
        return range(dst["r"], dst["r"] + dst.get("n", 1))
    return range(0)


def branch_target(ins: dict):
    """Byte pc a branch goes to (simm16 dwords after the branch), or None without an encoding."""
    imm = next((x for x in ins["args"] if x.get("k") == "imm"), None)
    if imm is None or not ins.get("enc"):
        return None
    v = imm["v"] & 0xFFFF
    return ins["pc"] + len(ins["enc"]) // 2 + 4 * (v - 0x10000 if v & 0x8000 else v)


def typed_buffer_ops(insns: list, table: frozenset, names: dict) -> Counter:
    """Count the typed buffer accesses of one listing (§8.2 #16).

    MTBUF (tbuffer_*) carries its format in the instruction; LLVM leaves `format:` out for GFX10's default, 8_UNORM.
    MUBUF (buffer_*_format_*) takes it from V# word 3 bits 18:12, and the result says where that word comes from in the
    basic block of the access (blocks start at branch targets and after branches, as rdna2spirv.translate splits
    them): an s_mov_b32 literal (the only form the translator folds, so the format is known when translating), a
    scalar load (a descriptor from memory), or any other instruction (computed). A word 3 not written in the block is
    "earlier_" + the class of the last write before it in listing order, or "shader_input" when nothing before it
    writes the register (a user or system SGPR: the root signature's descriptor).

    Keys: ("mtbuf", load|store, format), ("mubuf", load|store, class) and, for a literal word 3 (in the block or
    earlier), ("mubuf_literal", load|store, format). A format the translator's table lacks gets a "!" in front."""
    leaders = {insns[0]["pc"]} if insns else set()
    for ins in insns:
        if ins["op"] in BRANCH_OPS:
            t = branch_target(ins)
            if t is not None:
                leaders.add(t)
            if ins.get("enc"):
                leaders.add(ins["pc"] + len(ins["enc"]) // 2)
    out = Counter()
    block, last = {}, {}                # SGPR -> (class, literal or None): written in this block / last write so far
    for ins in insns:
        if ins["pc"] in leaders:
            block = {}
        op = ins["op"]
        m = TYPED_MTBUF.match(op) or TYPED_MUBUF.match(op)
        if m:
            kind = m.group(1)
            if op.startswith("t"):
                fm = next((x for x in ins["mods"] if x.startswith("format:")), "format:[BUF_FMT_8_UNORM]")
                name = fm.split("BUF_FMT_")[-1].rstrip("]")
                code = next((c for c, n in names.items() if n == name), None)
                out[("mtbuf", kind, ("" if code in table else "!") + name)] += 1
            elif len(ins["args"]) > 2 and ins["args"][2].get("k") == "sgpr":
                w3 = ins["args"][2]["r"] + 3
                if w3 in block:
                    cls, lit = block[w3]
                elif w3 in last:
                    cls, lit = "earlier_" + last[w3][0], last[w3][1]
                else:
                    cls, lit = "shader_input", None
                out[("mubuf", kind, cls)] += 1
                if lit is not None:
                    code = (lit >> 12) & 0x7F
                    out[("mubuf_literal", kind, ("" if code in table else "!") + names.get(code, f"code{code}"))] += 1
        regs = sgpr_writes(ins)
        if regs:
            if op == "s_mov_b32" and len(ins["args"]) > 1 and ins["args"][1].get("k") == "imm":
                state = ("literal", ins["args"][1]["v"] & 0xFFFFFFFF)
            elif op.startswith(("s_load", "s_buffer_load")):
                state = ("memory", None)
            else:
                state = ("computed", None)
            for r in regs:
                block[r] = last[r] = state
    return out


def scan_typed_buffers(disasm_dir: str, stage_of: dict) -> dict:
    """§8.2 #16 over every listing: typed buffer access counts (typed_buffer_ops) summed, and shaders per stage."""
    import shader_ir
    table, names = translator_formats()
    total, shaders = Counter(), defaultdict(Counter)
    for fn in sorted(os.listdir(disasm_dir)):
        if not fn.endswith(".s"):
            continue
        path = os.path.join(disasm_dir, fn)
        with open(path, encoding="utf-8", errors="replace") as f:
            if "_format_" not in f.read():      # no typed buffer op: skip the slower full parse
                continue
        stage = stage_of.get(fn[:-2], "?")
        c = typed_buffer_ops(shader_ir.parse_file(path)["insns"], table, names)
        total.update(c)
        for what in {k[0] for k in c}:
            shaders[what][stage] += 1
        if any(k[0] == "mubuf" and k[2] not in ("literal", "earlier_literal") for k in c):
            shaders["mubuf_runtime_format"][stage] += 1
    group = lambda what: {f"{k[1]} {k[2].lstrip('!')}": n for k, n in sorted(total.items()) if k[0] == what}
    return {"mtbuf": group("mtbuf"), "mubuf_word3": group("mubuf"), "mubuf_literal_formats": group("mubuf_literal"),
            "not_in_translator": {f"{k[0]} {k[1]} {k[2][1:]}": n for k, n in sorted(total.items())
                                  if k[0] != "mubuf" and k[2].startswith("!")},
            "shaders": {w: dict(s) for w, s in sorted(shaders.items())}}


def scan_spirv(spirv_index: str) -> dict:
    with open(spirv_index, encoding="utf-8-sig") as f:
        idx = json.load(f)
    # An object maps shader names to rows; a list is the other accepted shape. A scalar is a
    # broken index, not something to iterate -- report no translated shaders for it.
    if isinstance(idx, dict):
        entries = idx.values()
    elif isinstance(idx, list):
        entries = idx
    else:
        entries = ()
    total, code_base = Counter(), Counter()
    for m in entries:
        if not isinstance(m, dict):
            continue
        stage = stage_name(m)
        total[stage] += 1
        code_base[stage] += bool(m.get("uses_code_base"))
    return {"translated": dict(total), "uses_code_base": {k: v for k, v in code_base.items() if v}}


def same_path(first: str, second: str) -> bool:
    """True when two paths name the same file, including through a link or a case difference on Windows."""
    a = os.path.normcase(os.path.abspath(first))
    b = os.path.normcase(os.path.abspath(second))
    if a == b:
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def inside_dir(path: str, folder: str) -> bool:
    """True when the path itself or its symlink target is in the folder."""
    for resolve in (os.path.abspath, os.path.realpath):
        root = os.path.normcase(resolve(folder))
        target = os.path.normcase(resolve(path))
        if target == root or target.startswith(os.path.join(root, "")):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shaders", help="<work>/shaders from make_game.py")
    ap.add_argument("--out", help="also write the counts as JSON")
    a = ap.parse_args()

    dxbc_dir = os.path.join(a.shaders, "dxbc")
    index_path = os.path.join(dxbc_dir, "index.json")
    if not os.path.isdir(a.shaders):
        what = "is not a folder" if os.path.exists(a.shaders) else "does not exist"
        print(f"ERROR: {a.shaders} {what}. Give the <work>/shaders folder make_game.py writes.", file=sys.stderr)
        return 2
    if a.out:  # a report is not an input: one written over the corpus index leaves the next run unable to scan
        spirv_index = os.path.join(a.shaders, "spirv", "index.json")
        if inside_dir(a.out, dxbc_dir):
            print(f"ERROR: --out {a.out} is inside the scanned corpus {dxbc_dir}. Writing the report there would "
                  "replace index.json or a .dxbc the scan reads. Give a report path outside it.", file=sys.stderr)
            return 2
        for input_index in (index_path, spirv_index):
            if same_path(a.out, input_index):
                print(f"ERROR: --out {a.out} is {input_index}, which the scan reads. Give another report path.",
                      file=sys.stderr)
                return 2
    if not os.path.isfile(index_path):
        print(f"ERROR: {index_path} is missing. Run the shaders stage of make_game.py first.", file=sys.stderr)
        return 2
    try:
        with open(index_path, encoding="utf-8-sig") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {index_path}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
        return 2
    if not isinstance(index, list):
        print(f"ERROR: {index_path} is not a shader list. Run the shaders stage of make_game.py again.",
              file=sys.stderr)
        return 2
    result = {"shaders": len(index), "dxbc": scan_dxbc(dxbc_dir, index)}
    # Original names can be shared by variants, so indexed filenames take precedence.
    # Non-object rows are already counted as unreadable by scan_dxbc; skip them here too.
    rows = [e for e in index if isinstance(e, dict)]
    stage_of = {e["name"]: stage_name(e) for e in rows if isinstance(e.get("name"), str)}
    stage_of |= {os.path.splitext(e["file"])[0]: stage_name(e)
                 for e in rows if isinstance(e.get("file"), str)}
    disasm = os.path.join(dxbc_dir, "disasm")
    if os.path.isdir(disasm):
        result["disasm"] = scan_disasm(disasm, stage_of)
        result["typed_buffers"] = scan_typed_buffers(disasm, stage_of)
    spirv = os.path.join(a.shaders, "spirv", "index.json")
    if os.path.isfile(spirv):
        try:
            result["spirv"] = scan_spirv(spirv)
        except (OSError, ValueError) as e:
            print(f"ERROR: cannot read {spirv}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
            return 2

    d = result["dxbc"]
    print(f"{result['shaders']} shaders")
    print("DXIL stages:", ", ".join(f"{s} {n}" for s, n in d["stages"].items()))
    for name in WATCHED:
        print(f"SFI0 {name}: {sum(d['sfi0'].get(name, {}).values())} {d['sfi0'].get(name, {})}")
    print("SFI0, every flag set anywhere:", {n: sum(v.values()) for n, v in d["sfi0"].items()})
    print("InnerCoverage metadata (ISG1 absence does not disprove use; no Xbox VGPR mapping):",
          d["inner_coverage_metadata"])
    print("GS output-signature stream counts:", d["gs_output_stream_count"],
          "| shaders with an output on a stream other than 0:", d["output_on_stream_other_than_0"] or 0)
    pv = d["psv0"]
    print(f"PSV0 bindings: {pv['records']} records | shaders with "
          + ", ".join(f"{c} {pv['shaders'][c]} ({pv['records_of'][c]} records)" for c in WATCHED_BINDINGS)
          + (f" | {pv['unreadable']} unreadable PSV0" if pv["unreadable"] else ""))
    if "disasm" in result:
        s = result["disasm"]
        print(f"disasm: {s['files']} files | wave MMA shaders {s['wave_mma_shaders'] or 0} | ordered-count/GWS shaders "
              f"{s['ordered_or_gws_shaders'] or 0} {s['ordered_or_gws_ops'] or ''} | GDS shaders {s['gds_shaders'] or 0}")
        print("s_getpc_b64 used for code-embedded data (not a far branch), by stage:", s["getpc_for_data_shaders"] or 0)
        vp = s["v_perm_b32"]
        print(f"v_perm_b32: {sum(vp['shaders'].values())} shaders {vp['shaders'] or 0} | {vp['insns']} insns "
              f"(imm {vp['immediate']}, const_reg {vp['const_reg']}, unresolved {vp['unresolved']}) | "
              f"selectors 8-11: {vp['sign_extend_8_11_shaders']} shaders / {vp['sign_extend_8_11_insns']} insns | "
              f"bytes {vp['selector_classes']}")
        tb = result["typed_buffers"]
        print("typed buffers, MTBUF formats (instruction):", tb["mtbuf"] or 0)
        print("typed buffers, MUBUF V# word 3 source:", tb["mubuf_word3"] or 0, "| literal formats:",
              tb["mubuf_literal_formats"] or 0)
        print("typed buffers, formats the translator's table lacks:", tb["not_in_translator"] or 0,
              "| shaders:", tb["shaders"] or 0)
    else:
        print("disasm: not there yet (run the shaders stage)")
    if "spirv" in result:
        s = result["spirv"]
        print("uses_code_base by stage:", s["uses_code_base"], "of", s["translated"])
    else:
        print("spirv/index.json: not there yet (run the shaders stage)")
    if d["unreadable"]:
        print(f"{d['unreadable']} .dxbc files could not be read")
    if a.out:
        folder = os.path.dirname(a.out)  # --out into a folder that does not exist yet, as awc_manifest.py does
        try:
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.out}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
