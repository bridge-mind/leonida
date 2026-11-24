"""Wrap a scalar-ALU test program in a kernel prologue/epilogue and HSA metadata. Pure text, no hardware.

A program is the list of lines tools/salu_diff.py generates: s_mov_b32 seeds, scalar ops, and an observation tail
that leaves four words in v8..v11. The kernel loads its output pointer from the kernarg segment, runs the program
on one lane (the launcher starts a single thread, so exec = lane 0) and stores v[8:11] as four dwords.

Registers: s[0:1] kernarg pointer, s[2:3] output pointer; the programs use s4..s19 and s30, v8..v11.
"""
import re

# SALU opcodes salu_diff models that GFX11 no longer has. Spellings that were only renamed (s_andn2_b32 ->
# s_and_not1_b32, s_ff1_i32_b32 -> s_ctz_i32_b32, s_flbit_i32_b32 -> s_clz_i32_u32, ...) are accepted by LLVM as
# aliases with the same semantics, so they are not listed here.
GFX11_REMOVED = {
    "s_ff0_i32_b32": "removed in GFX11 (no find-first-zero; the ISA has only s_ctz = ff1)",
    "s_ff0_i32_b64": "removed in GFX11 (no find-first-zero; the ISA has only s_ctz = ff1)",
}
# GFX12 also drops the SOPK compares: AMD clang (rocm-sdk-core 10.0.0) assembles all twelve for gfx1150 and rejects
# every one for gfx1201 ("instruction not supported on this GPU").
GFX12_REMOVED = dict(GFX11_REMOVED, **{
    f"s_cmpk_{cond}_{ty}": "removed in GFX12 (no SOPK compares; s_cmp with a literal is the replacement)"
    for cond in ("eq", "lg", "gt", "ge", "lt", "le") for ty in ("i32", "u32")
})

_LITERAL_FREE = ("s_movk_i32", "s_addk_i32", "s_mulk_i32")


def family(arch):
    """'gfx10', 'gfx11' or 'gfx12' for gfx10xx, gfx11xx and gfx12xx. GFX12 uses GFX11's spellings for everything
    the kernels here emit; it differs only in which SALU opcodes exist (GFX12_REMOVED)."""
    m = re.fullmatch(r"gfx(\d+?)[0-9a-f]{2}", arch)
    if not m or m.group(1) not in ("10", "11", "12"):
        raise ValueError(f"unsupported GPU architecture {arch!r} (expected gfx10xx, gfx11xx or gfx12xx)")
    return "gfx" + m.group(1)


def _is_inline(tok):
    try:
        return -16 <= int(tok, 0) <= 64
    except ValueError:
        return True   # registers and symbols are not literals


def literals(line):
    """The distinct 32-bit literal values an instruction needs (SOPK immediates are not literals)."""
    op, _, rest = line.partition(" ")
    if not op.startswith("s_") or op.rstrip("_") in _LITERAL_FREE or re.match(r"s_cmpk_", op):
        return set()
    toks = [t.strip() for t in rest.split(",")] if rest else []
    # -1 and 0xffffffff are the same 32-bit value, so they share one literal slot
    return {int(t, 0) & 0xFFFFFFFF for t in toks if not t.startswith("s") and not _is_inline(t)}


def skip_reason(lines, arch):
    """Why a program cannot be run on this architecture, or None. Hardware allows one literal per instruction."""
    removed = {"gfx10": {}, "gfx11": GFX11_REMOVED, "gfx12": GFX12_REMOVED}[family(arch)]
    for line in lines:
        op = line.partition(" ")[0]
        if op in removed:
            return f"{op}: {removed[op]}"
        if len(literals(line)) > 1:
            return "two different literals in one instruction (not encodable)"
    return None


class ModeUnavailable(Exception):
    """A requested float MODE field cannot be encoded on this architecture."""


class Kernel:
    """One program to run. lines: the instructions. raw=False wraps them (output pointer load, v[8:11] store);
    raw=True means the lines already end with their own stores and s_endpgm and only get the output pointer in
    s[2:3]. threads: workgroup size; words: dwords read back; wave: 32 or 64. mode: the float MODE the kernel
    starts with (raw kernels only), a dict of MODE_FIELDS names to values; None keeps the assembler's defaults.
    On gfx1170+ the assembler rejects the ieee and dx10_clamp directives, so requesting those fields raises
    ModeUnavailable while the denormal and rounding fields still work.
    lds: bytes of LDS the kernel allocates (.amdhsa_group_segment_fixed_size and the metadata's
    .group_segment_fixed_size; without it the hardware gives the group no LDS); vgprs, sgprs: registers the kernel uses
    (default 32 and 40)."""

    def __init__(self, lines, wave=32, threads=1, words=4, raw=False, mode=None, lds=0, vgprs=32, sgprs=40):
        self.lines, self.wave, self.threads, self.words, self.raw = list(lines), wave, threads, words, raw
        self.lds, self.vgprs, self.sgprs = lds, vgprs, sgprs
        if not 0 <= lds <= 65536 or lds % 4:
            raise ValueError(f"LDS size {lds} is not a multiple of 4 in 0..65536")
        if not 1 <= vgprs <= 256:
            raise ValueError(f"vgprs {vgprs} is not in 1..256")
        if not 1 <= sgprs <= 104:
            raise ValueError(f"sgprs {sgprs} is not in 1..104")
        self.mode = dict(mode or {})
        unknown = set(self.mode) - set(MODE_FIELDS)
        if unknown:
            raise ValueError(f"unknown MODE fields {sorted(unknown)} (known: {sorted(MODE_FIELDS)})")


# Float MODE the kernel descriptor starts a wave with, as .amdhsa_* directives. The assembler's defaults (checked on
# gfx1150 by reading the register back, see docs/shader_validation.md): IEEE 1, DX10_CLAMP 1, f32 denormals
# flushed (denorm_mode_32 = 0), f16/f64 denormals kept (3), all rounding to nearest even (0).
MODE_FIELDS = {
    "ieee": ".amdhsa_ieee_mode",
    "dx10_clamp": ".amdhsa_dx10_clamp",
    "denorm32": ".amdhsa_float_denorm_mode_32",
    "denorm16_64": ".amdhsa_float_denorm_mode_16_64",
    "round32": ".amdhsa_float_round_mode_32",
    "round16_64": ".amdhsa_float_round_mode_16_64",
}
# AMD clang (rocm-sdk-core 10.0.0) rejects these two directives on gfx1170 and later; the hardware keeps its
# fixed IEEE/DX10_CLAMP behaviour there, so a run that asks for either field is a skip, not an assembler failure.
_MODE_FIELDS_GONE_ON_GFX1170 = ("ieee", "dx10_clamp")


# MODE fields a kernel descriptor can no longer set: GFX11.7 and GFX12 have fixed IEEE / DX10_CLAMP behaviour, and AMD
# clang (rocm-sdk-core 10.0.0) rejects .amdhsa_ieee_mode and .amdhsa_dx10_clamp for gfx1201 with "directive
# unsupported on gfx1170+" (FUZZ-43, found by @Dom-in in #1548). The four denormal and rounding fields still assemble.
_MODE_FIXED = ("ieee", "dx10_clamp")


def is_gfx1170_or_later(arch):
    """True for GFX11.7 and GFX12 (gfx1170 and later), which RDNA2 and RDNA3 code does not assemble for as is."""
    fam = family(arch)
    return fam == "gfx12" or (fam == "gfx11" and int(arch[5], 16) >= 7)


def rdna23_skip_reason(arch):
    """Why a fuzzer or test written for RDNA2 and RDNA3 cannot run on this GPU, or None. AMD clang (rocm-sdk-core 10.0.0)
    rejects s_barrier and some 16-bit and memory operands for gfx1201 (TEST-873, found by @PeterGrieffindor in #3918),
    which fails every program of a chunk; on gfx1170+ the caller reports a skip instead."""
    if not is_gfx1170_or_later(arch):
        return None
    return f"{arch} (GFX11.7 or GFX12) is not RDNA2 or RDNA3: the kernels here do not assemble for it"


def mode_skip_reason(mode, arch):
    """Why a kernel's starting MODE cannot be set on this architecture, or None. A kernel that asks for one of
    _MODE_FIXED on gfx1170+ (GFX11.7 and GFX12) is reported as not runnable there, rather than reaching the assembler,
    which rejects the directive and, when every kernel of a chunk sets it, the whole chunk."""
    if not is_gfx1170_or_later(arch):
        return None
    fixed = sorted(k for k in (mode or {}) if k in _MODE_FIXED)
    if not fixed:
        return None
    return (f"MODE {', '.join(fixed)} cannot be set on {arch}: {', '.join(MODE_FIELDS[k] for k in fixed)} "
            "is unsupported on gfx1170+ (the behaviour is fixed there)")


# The global memory instructions GFX11 renamed, under their GFX10 names. A raw kernel body is written with the
# GFX11 ones (crosslane_fuzz stores with global_store_b32); a GFX10 assembler only takes the old spelling.
_GFX10_NAMES = {f"global_{kind}_b{bits}": f"global_{kind}_dword{suffix}"
                for kind in ("load", "store") for bits, suffix in ((32, ""), (64, "x2"), (96, "x3"), (128, "x4"))}


def gfx10_spelling(line):
    """A raw kernel line with a renamed memory mnemonic put back to its GFX10 name (the same instruction)."""
    op, sep, rest = line.partition(" ")
    return _GFX10_NAMES.get(op, op) + sep + rest


_REG = re.compile(r"(?<![\w.])([vs])(?:(\d+)|\[\d+:(\d+)\])(?![\w.])")


def regs_used(lines):
    """(VGPRs, SGPRs) the lines name: one past the highest v<N> / v[a:b] and s<N> / s[a:b], 0 when there are none."""
    top = {"v": 0, "s": 0}
    for line in lines:
        for kind, single, last in _REG.findall(line.split(";", 1)[0]):
            top[kind] = max(top[kind], int(last or single) + 1)
    return top["v"], top["s"]


def register_counts(lines, vgprs, sgprs):
    """The .amdhsa_next_free_vgpr / _sgpr a kernel gets: what was asked for, raised to cover every register the lines
    name (at most 256 VGPRs and 104 SGPRs; a program past that is the assembler's to reject). A descriptor that
    allocates fewer leaves the program registers that read back 0: measured on gfx1036, v16 and up under the wrapped
    kernels' old fixed `.amdhsa_next_free_vgpr 16`."""
    v, s = regs_used(lines)
    return min(max(vgprs, v), 256), min(max(sgprs, s), 104)


def raw_kernel_text(name, lines, arch, wave=32, mode=None, lds=0, vgprs=32, sgprs=40):
    """A kernel whose body does its own stores. s[0:1] is the kernarg pointer, s[2:3] gets the output pointer."""
    fam = family(arch)
    vgprs, sgprs = register_counts(lines, vgprs, sgprs)
    if int(arch[3:], 16) >= 0x1170:
        unsupported = sorted(set(mode or {}) & set(_MODE_FIELDS_GONE_ON_GFX1170))
        if unsupported:
            fields = ", ".join(unsupported)
            raise ModeUnavailable(
                f"{fields} cannot be set on {arch}: AMD clang rejects .amdhsa_ieee_mode and .amdhsa_dx10_clamp "
                "on gfx1170+, and the hardware keeps its fixed behaviour there; only denormal and rounding "
                "mode fields are available")
    modes = "".join(f"\n  {MODE_FIELDS[k]} {int(v)}" for k, v in sorted((mode or {}).items()))
    load = "s_load_b64" if fam == "gfx11" else "s_load_dwordx2"
    if fam == "gfx10":
        lines = [gfx10_spelling(ln) for ln in lines]
    body = "\n".join("  " + ln for ln in lines)
    return f""".globl {name}
.p2align 8
.type {name},@function
{name}:
  {load} s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
{body}
.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_next_free_sgpr {sgprs}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_wavefront_size32 {1 if wave == 32 else 0}
  .amdhsa_system_vgpr_workitem_id 0{modes}
.end_amdhsa_kernel
.text
"""


def kernel_text(name, lines, arch, wave=32, vgprs=16, sgprs=32, lds=0):
    """Assembly for one kernel. The descriptor takes the wave size, the register counts (raised to the registers
    the lines name, register_counts) and the LDS size (the group segment: without it a Linux launcher has nothing
    to program COMPUTE_PGM_RSRC2.LDS_SIZE from). It used to be fixed at wave32, 16 VGPRs and 32 SGPRs: on gfx1036
    v16 and up read back 0 and a Kernel(wave=64) ran as wave32 (the lane ids of a full EXEC had 32 lanes)."""
    fam = family(arch)
    vgprs, sgprs = register_counts(lines, vgprs, sgprs)
    load = "s_load_dwordx2" if fam == "gfx10" else "s_load_b64"
    store = "global_store_dwordx4" if fam == "gfx10" else "global_store_b128"
    body = "\n".join("  " + line for line in lines)
    return f""".globl {name}
.p2align 8
.type {name},@function
{name}:
  {load} s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32 v0, 0
{body}
  {store} v0, v[8:11], s[2:3]
  s_nop 0
  s_endpgm
.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr {vgprs}
  .amdhsa_next_free_sgpr {sgprs}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_wavefront_size32 {1 if wave == 32 else 0}
  .amdhsa_system_vgpr_workitem_id 0
.end_amdhsa_kernel
.text
"""


def metadata(names, wave=32, lds=None, vgprs=None, sgprs=None):
    """The HSA metadata block. Without it the HIP runtime fails the load with hipErrorNoBinaryForGpu (209).
    lds / vgprs / sgprs: per kernel (lists as long as names); default 0 bytes of LDS, 32 VGPRs and 40 SGPRs."""
    out = [".amdgpu_metadata", "---", "amdhsa.version: [1, 2]", "amdhsa.kernels:"]
    for i, n in enumerate(names):
        out += [f"  - .name: {n}", f"    .symbol: {n}.kd", "    .args:",
                "      - {.offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global}",
                f"    .group_segment_fixed_size: {lds[i] if lds else 0}", "    .private_segment_fixed_size: 0",
                "    .kernarg_segment_align: 8", "    .kernarg_segment_size: 8", "    .max_flat_workgroup_size: 1024",
                f"    .sgpr_count: {sgprs[i] if sgprs else 40}", f"    .vgpr_count: {vgprs[i] if vgprs else 32}", f"    .wavefront_size: {wave}"]
    out.append(".end_amdgpu_metadata")
    return "\n".join(out) + "\n"


def module_text(programs, arch, prefix="k", wave=32):
    """(assembly text, kernel names, first source line of each kernel) for a list of programs. A program is a list
    of lines (wrapped as a scalar test) or a Kernel."""
    parts = [f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"', ".text"]
    names, starts, line_no, lds, vgprs, sgprs = [], [], 2, [], [], []
    for i, prog in enumerate(programs):
        name = f"{prefix}{i}"
        lds.append(prog.lds if isinstance(prog, Kernel) else 0)
        if isinstance(prog, Kernel):
            vg, sg = register_counts(prog.lines, prog.vgprs, prog.sgprs)
            text = (raw_kernel_text(name, prog.lines, arch, prog.wave, prog.mode, prog.lds, vg, sg) if prog.raw
                    else kernel_text(name, prog.lines, arch, prog.wave, vg, sg, prog.lds))
        else:
            vg, sg = register_counts(prog, 16, 32)
            text = kernel_text(name, prog, arch, wave, vg, sg)
        vgprs.append(vg)
        sgprs.append(sg)
        names.append(name)
        starts.append(line_no + 1)
        parts.append(text.rstrip("\n"))
        line_no += text.count("\n") + 0
    parts.append(metadata(names, wave, lds, vgprs, sgprs).rstrip("\n"))
    return "\n".join(parts) + "\n", names, starts


# ---- f32 vector ALU: what GFX11 does not have, or has under another name (tools/f32_fuzz.py --oracle hw) ----
# Checked by assembling every f32_fuzz opcode with llvm-mc for gfx1013 (the console), gfx1010, gfx1030, gfx1100,
# gfx1150 and gfx1200. The four mad opcodes exist on GFX10.1 only (gfx1010, gfx1011, gfx1013): gfx1030 (RDNA2
# desktop) and every GFX11 part reject them. They are not "renamed": v_fma_f32, v_fmac_f32, v_fmaak_f32 and
# v_fma_dx9_zero_f32 round once where the mad opcodes round the product first, so substituting one would compare a
# different instruction.
_MAD_GONE = "exists on GFX10.1 only (gfx1010/1011/1013), not on gfx1030 or GFX11; {} is fused, a different instruction"
VALU_REMOVED = {
    "v_mad_f32": _MAD_GONE.format("v_fma_f32"),
    "v_mac_f32": _MAD_GONE.format("v_fmac_f32"),
    "v_madak_f32": _MAD_GONE.format("v_fmaak_f32"),
    "v_madmk_f32": _MAD_GONE.format("v_fmamk_f32"),
    "v_mad_legacy_f32": _MAD_GONE.format("v_fma_dx9_zero_f32"),
}
# Same instruction, new mnemonic: the assembler takes the GFX10 spelling for GFX11 and encodes the GFX11 name.
GFX11_VALU_RENAMED = {
    "v_mul_legacy_f32": "v_mul_dx9_zero_f32",        # DX9 rule: 0 * anything = 0 (the oracle itself checks it)
    "v_cvt_pkrtz_f16_f32": "v_cvt_pk_rtz_f16_f32",
}


# ---- LDS (ds_*): GFX11 renamed the loads, stores and two atomics (tools/lds_fuzz.py --oracle hw) ----
# Checked by assembling every ds_* mnemonic lds_fuzz emits with AMD clang for gfx1030 and gfx1150: the assembler takes
# the GFX10 spelling of all of them on gfx1150 except ds_cmpst_rtn_b32 ("instruction not supported on this GPU"), and
# takes the GFX11 spelling on gfx1030 for none. The new spelling is used on GFX11/GFX12 so the program text is what
# the target names, not what an alias happens to accept. ds_cmpstore_rtn_b32 also takes its two data operands in the
# other order (ds_line), and GFX12 spells the barrier and the LDS wait differently (sync_lines).
DS_RENAMED_GFX11 = {
    "ds_wrxchg_rtn_b32": "ds_storexchg_rtn_b32",
    "ds_cmpst_rtn_b32": "ds_cmpstore_rtn_b32",
}
_DS_STEMS_GFX11 = (("ds_read2st64_", "ds_load_2addr_stride64_"), ("ds_read2_", "ds_load_2addr_"),
                   ("ds_write2st64_", "ds_store_2addr_stride64_"), ("ds_write2_", "ds_store_2addr_"),
                   ("ds_read_", "ds_load_"), ("ds_write_", "ds_store_"))


def ds_mnemonic(op, arch):
    """The spelling of a ds_* mnemonic for this architecture (GFX10 spellings in, as lds_fuzz writes them)."""
    if family(arch) == "gfx10" or not op.startswith("ds_"):
        return op
    if op in DS_RENAMED_GFX11:
        return DS_RENAMED_GFX11[op]
    for old, new in _DS_STEMS_GFX11:
        if op.startswith(old):
            return new + op[len(old):]
    return op


def ds_line(line, arch):
    """A whole instruction line (GFX10 spelling and operand order in) respelled for the architecture. On GFX11 and
    GFX12 ds_cmpstore_rtn_b32 takes the value to store first and the compare value second, the reverse of GFX10's
    ds_cmpst_rtn_b32 vdst, vaddr, cmp, store (measured on gfx1150: tools/lds_fuzz.py --oracle hw, and it is what LLVM's
    cmpxchg selection does for GFX11; measured on gfx1201, an RX 9070 XT: with M = 7 at the address, data v1 = 9 and
    v2 = 7 the dword becomes 9 and with v1 = 7, v2 = 9 it stays 7, for the rtn and the non-returning form, while
    gfx1036's ds_cmpst does the opposite); llvm-mc encodes both spellings of the same text identically, so only the
    operand order tells them apart."""
    op, sep, rest = line.partition(" ")
    if op == "ds_cmpst_rtn_b32" and family(arch) in ("gfx11", "gfx12"):
        m = re.fullmatch(r"(v\d+),\s*(v\d+),\s*(v\d+),\s*(v\d+)(.*)", rest)
        if not m:
            raise ValueError(f"cannot swap the operands of {line!r}")
        dst, addr, cmp_, store, tail = m.groups()
        rest = f"{dst}, {addr}, {store}, {cmp_}{tail}"
    elif op == "ds_cmpst_b32" and family(arch) in ("gfx11", "gfx12"):
        # The non-returning form is renamed as well, and its two data operands swap the same way: GFX10's
        # ds_cmpst_b32 vaddr, cmp, store is GFX11's ds_cmpstore_b32 vaddr, store, cmp (measured on gfx1100: with
        # M = 7, vaddr = 0 and data v1 = 9, v2 = 7 the dword becomes 9, so the store operand is v1; the same on
        # gfx1201). The rename is done here rather than in DS_RENAMED_GFX11 so that it never happens without the
        # swap, which would silently compare and store the wrong way round.
        m = re.fullmatch(r"(v\d+),\s*(v\d+),\s*(v\d+)(.*)", rest)
        if not m:
            raise ValueError(f"cannot swap the operands of {line!r}")
        addr, cmp_, store, tail = m.groups()
        return f"ds_cmpstore_b32 {addr}, {store}, {cmp_}{tail}"
    return ds_mnemonic(op, arch) + sep + rest


def sync_lines(line, arch):
    """A wait or barrier line (GFX10/GFX11 spelling in) as the instructions that do it on this architecture. GFX12 has
    no s_barrier (AMD LLVM 23: "instruction not supported on this GPU (gfx1201)"); a workgroup barrier there is
    s_barrier_signal -1 then s_barrier_wait -1. It also split lgkmcnt into dscnt (LDS) and kmcnt (scalar memory), so
    a wait for the LDS is s_wait_dscnt 0 rather than the legacy s_waitcnt encoding the assembler still takes."""
    if family(arch) == "gfx12":
        text = line.strip()
        if text == "s_barrier":
            return ["s_barrier_signal -1", "s_barrier_wait -1"]
        if text == "s_waitcnt lgkmcnt(0)":
            return ["s_wait_dscnt 0"]
    return [line]


def valu_skip_reason(op, arch):
    """Why an f32 VALU opcode cannot be run on this architecture, or None."""
    if op in VALU_REMOVED and arch not in ("gfx1010", "gfx1011", "gfx1012", "gfx1013"):
        return f"{op}: {VALU_REMOVED[op]}"
    return None


# ---- true16: LLVM 23 makes GFX11's 16-bit operands half registers (v8.l) and rejects a bare v8 there ----
# The one-source f16 ops the fuzzers and sweeps run here: the two conversions and f16_trans_sweep's seven.
_F16_UNARY = "|".join(f"v_{op}_f16" for op in ("exp", "log", "rcp", "rsq", "sqrt", "sin", "cos"))
_F16_DST = re.compile(rf"^([ \t]*(?:v_cvt_f16_f32|{_F16_UNARY})(?:_e32|_e64)?[ \t]+v\d+)(?=[ \t]*,)", re.M)
_F16_SRC = re.compile(rf"^([ \t]*(?:v_cvt_f32_f16|{_F16_UNARY})(?:_e32|_e64)?[ \t]+v\d+(?:\.[lh])?[ \t]*,"
                      r"[-| \ta-z(]*\bv\d+)\b(?!\.[lh])", re.M)
# These binary instructions use 16-bit VGPRs for all three operands. Packed operations and the VOP3 instructions the
# fuzzers emit with op_sel keep their existing operand spelling.
_TRUE16_BINARY = re.compile(
    r"^([ \t]*v_(?:(?:add|sub|mul|max|min)_f16|mul_lo_u16|lshlrev_b16|lshrrev_b16|ashrrev_i16|"
    r"(?:max|min)_[ui]16)(?:_e32|_e64)?[ \t]+)([^/;\r\n]+)", re.M)
_BARE_VGPR = re.compile(r"\bv\d+\b(?!\.[lh])")


def true16_spelling(text):
    """The 16-bit operands of those ops written as the low half, which is what a bare register means there on GFX10
    and to older assemblers. Line numbers don't change, so error lines still map to kernels."""
    def binary(match):
        prefix, operands = match.groups()
        if re.search(r"\bop_sel\s*:", operands):
            return match[0]
        return prefix + _BARE_VGPR.sub(r"\g<0>.l", operands)

    text = _F16_SRC.sub(r"\1.l", _F16_DST.sub(r"\1.l", text))
    return _TRUE16_BINARY.sub(binary, text)


def needs_true16(assembles):
    """Probe fixed input so a failed generated instruction never changes the chosen spelling."""
    bare = "v_cvt_f16_f32 v8, v1\n"
    return not assembles(bare) and assembles(true16_spelling(bare))


# ---- op_sel on GFX11+: the two-source 16-bit integer VOP3 ops want it on the registers too ----
# This is separate from true16_spelling: AMD LLVM 23 (rocm-sdk-core 10.0.0) still takes a bare `v_cvt_f16_f32 v8, v1`
# on gfx1100, gfx1150 and gfx1201, so the runner's true16 probe says no, yet it rejects
# `v_max_u16_e64 v14, s12, v5 op_sel:[1,0,0]` on all three ("operands are not valid for this GPU or mode") and takes
# `v_max_u16_e64 v14.l, s12, v5.l op_sel:[1,0,0]`, which encodes to the same op_sel bits and operands as gfx1036's
# bare form. v_add_nc/v_sub_nc_*16 and the three-source 16-bit ops still take the bare form there.
_OPSEL16_OPS = ("v_ashrrev_i16", "v_lshlrev_b16", "v_lshrrev_b16", "v_max_i16", "v_max_u16", "v_min_i16",
                "v_min_u16", "v_mul_lo_u16")
_OPSEL16 = re.compile(rf"^([ \t]*(?:{'|'.join(_OPSEL16_OPS)})(?:_e64)?[ \t]+)([^\n]*?)([ \t]+op_sel:\[([01,]+)\])"
                      r"([^\n]*)$", re.M)


def _opsel16_halves(m):
    """One _OPSEL16 line with each bare VGPR operand named by the half its op_sel bit picks: bit i for source i,
    the last bit for the destination. The op_sel list itself is kept, since SGPR and constant sources still
    take their half from it."""
    head, operands, opsel, bits, tail = m.groups()
    ops = [o.strip() for o in operands.split(",")]
    bits = [b == "1" for b in bits.split(",")]
    if len(bits) != len(ops):     # not the dst + sources shape this respells; leave it to the assembler to judge
        return m.group(0)
    for i, o in enumerate(ops):
        if re.fullmatch(r"v\d+", o):
            ops[i] = o + (".h" if bits[i - 1] else ".l")    # i - 1: the destination (operand 0) is the last bit
    return head + ", ".join(ops) + opsel + tail


def opsel16_spelling(text):
    """The VGPRs of the _OPSEL16_OPS lines that have op_sel, written as the half it selects, for GFX11 and later.
    Lines without op_sel and every other op are left as they are. Line numbers don't change."""
    return _OPSEL16.sub(_opsel16_halves, text)


def failing_kernels(stderr, starts):
    """Indices of the kernels an assembler error message points into (error lines are 'file:LINE:COL: error')."""
    bad = set()
    for m in re.finditer(r":(\d+):\d+: error", stderr):
        ln = int(m.group(1))
        k = max((i for i, s in enumerate(starts) if s <= ln), default=None)
        if k is not None:
            bad.add(k)
    return bad
