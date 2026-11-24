r"""Measure RDNA2 v_cvt_pknorm_i16_f16 and v_cvt_pknorm_u16_f16 on native hardware. Standard library only.

Runs an exhaustive 65,536-input binary16 sweep across plain, VOP3 `clamp`, `op_sel:[0,0,1]`, and
`op_sel:[0,0,1] clamp` plus targeted edge cases (all 8 `op_sel:[s0,s1,d]` permutations, prior `vdst`
overwrite checks, double-rounding traps, `abs`/`neg`, inline f16/int constants, 16-bit literal, SGPR source,
destination aliasing, and raw `.long` VOP3 `omod=0..3` encodings) across wave32 and wave64 on GFX10 (`gfx1036`).

Usage:
    python -I tools/cvt_pknorm_f16_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/cvt_pknorm_f16_probe.py --self-test

Exit 0 when all hardware results match the exact Fraction reference across requested ops/waves with zero wave32/wave64
diffs, zero clamp diffs, and zero raw-omod diffs; 1 on a mismatch or parity failure; 2 on a harness/CLI error;
77 when the hardware oracle is unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OPS: tuple[str, ...] = ("v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16")
_OPCODES: dict[str, int] = {"v_cvt_pknorm_i16_f16": 0x12, "v_cvt_pknorm_u16_f16": 0x13}
SWEEP_MODES: tuple[tuple[str, bool, bool], ...] = (
    ("", False, False),
    ("clamp", True, False),
    ("op_sel:[0,0,1]", False, True),
    ("op_sel:[0,0,1] clamp", True, True),
)
CLAMP_MODES: tuple[bool, ...] = (False, True)
SWEEP_WORDS_PER_HALF = 16384


@dataclass(frozen=True)
class Case:
    name: str
    src0: int
    src1: int
    src0_asm: str | None = None
    src1_asm: str | None = None
    src0_abs: bool = False
    src0_neg: bool = False
    src1_abs: bool = False
    src1_neg: bool = False
    sel: tuple[int, int] = (0, 0)
    dhi: int = 0
    emit_opsel: bool = False
    init_dst: int = 0xDEADBEEF
    alias: str | None = None  # "src0" | "src1" | None
    sgpr0: bool = False
    raw_omod: int | None = None


CASES: tuple[Case, ...] = (
    # All 8 op_sel:[s0,s1,d] combinations on distinct lo/hi halves (hi0=1.0, lo0=0.25; hi1=-1.0, lo1=0.5+2^-11)
    Case("opsel_000", 0x3C003400, 0xBC003801, sel=(0, 0), dhi=0, emit_opsel=True),
    Case("opsel_001", 0x3C003400, 0xBC003801, sel=(0, 0), dhi=1, emit_opsel=True),
    Case("opsel_010", 0x3C003400, 0xBC003801, sel=(0, 1), dhi=0, emit_opsel=True),
    Case("opsel_011", 0x3C003400, 0xBC003801, sel=(0, 1), dhi=1, emit_opsel=True),
    Case("opsel_100", 0x3C003400, 0xBC003801, sel=(1, 0), dhi=0, emit_opsel=True),
    Case("opsel_101", 0x3C003400, 0xBC003801, sel=(1, 0), dhi=1, emit_opsel=True),
    Case("opsel_110", 0x3C003400, 0xBC003801, sel=(1, 1), dhi=0, emit_opsel=True),
    Case("opsel_111", 0x3C003400, 0xBC003801, sel=(1, 1), dhi=1, emit_opsel=True),
    # Destination overwrite checks: op_sel[2]=1 zeroes D[15:0] regardless of prior vdst bits
    Case("dst_zero_12345678", 0x3C003400, 0xBC003801, dhi=1, emit_opsel=True, init_dst=0x12345678),
    Case("dst_zero_ffffffff", 0x3C003400, 0xBC003801, dhi=1, emit_opsel=True, init_dst=0xFFFFFFFF),
    # Double-rounding traps and inf/NaN/denormal/zero boundaries
    Case("trap_pos_neg_0p5_ulp", 0x7E003801, 0x3C00B801),
    Case("trap_below_0p5_ulp", 0x000137FF, 0x8001B7FF),
    Case("inf_pos_neg", 0x38007C00, 0x3800FC00),
    Case("nan_q_and_s", 0x3C007E00, 0xBC007C01),
    Case("signed_zeros_and_subnorm", 0x00018000, 0x80010000),
    # Source abs/neg modifiers with op_sel
    Case("abs_neg_opsel_110", 0x3C003400, 0xBC003801, src0_abs=True, src1_neg=True, sel=(1, 1), dhi=0, emit_opsel=True),
    Case("negabs_abs_opsel_010", 0x3C003400, 0xBC003801, src0_abs=True, src0_neg=True, src1_abs=True, sel=(0, 1), dhi=0, emit_opsel=True),
    Case("abs_neg_opsel_111", 0x3C003400, 0xBC003801, src0_abs=True, src1_neg=True, sel=(1, 1), dhi=1, emit_opsel=True),
    # Inline f16 / integer constants (op_sel=1 on inline f16 reads high half 0x0000)
    Case("inline_f16_pair", 0x00003800, 0x0000B800, src0_asm="0.5", src1_asm="-0.5"),
    Case("inline_f16_sel100", 0x00003800, 0xBC003801, src0_asm="0.5", sel=(1, 0), dhi=0, emit_opsel=True),
    Case("inline_1_0_sel100", 0x00003C00, 0xBC003801, src0_asm="1.0", sel=(1, 0), dhi=0, emit_opsel=True),
    Case("inline_neg_f16_src", 0x00003800, 0xBC003801, src0_asm="0.5", src0_neg=True),
    Case("inline_int_1_neg1", 0x00000001, 0xFFFFFFFF, src0_asm="1", src1_asm="-1"),
    # 16-bit literal & SGPR source
    Case("lit16_3800_sel000", 0x00003800, 0xBC003801, src0_asm="0x3800"),
    Case("lit16_3800_sel100", 0x00003800, 0xBC003801, src0_asm="0x3800", sel=(1, 0), dhi=0, emit_opsel=True),
    Case("sgpr0_opsel_100", 0x3C003400, 0xBC003801, sel=(1, 0), dhi=0, emit_opsel=True, sgpr0=True),
    Case("sgpr0_opsel_101", 0x3C003400, 0xBC003801, sel=(1, 0), dhi=1, emit_opsel=True, sgpr0=True),
    # Destination aliasing under op_sel[2]=1
    Case("dst_alias_src0_dhi", 0x3C003400, 0xBC003801, dhi=1, emit_opsel=True, alias="src0"),
    Case("dst_alias_src1_dhi", 0x3C003400, 0xBC003801, dhi=1, emit_opsel=True, alias="src1"),
    # Raw VOP3 omod bits [60:59] (llvm-mc rejects textual mul:*/div:* on this op; silicon ignores them)
    Case("raw_omod_0", 0x3C003400, 0xBC003801, raw_omod=0),
    Case("raw_omod_1", 0x3C003400, 0xBC003801, raw_omod=1),
    Case("raw_omod_2", 0x3C003400, 0xBC003801, raw_omod=2),
    Case("raw_omod_3", 0x3C003400, 0xBC003801, raw_omod=3),
)


def ref_norm16_f16(bits16: int, snorm: bool) -> int:
    """Exact single-rounded normalized f16 -> snorm16/unorm16 word (NaN -> 0, +-inf saturate)."""
    x = struct.unpack("<e", struct.pack("<H", bits16 & 0xFFFF))[0]
    if math.isnan(x):
        return 0
    lo = -1.0 if snorm else 0.0
    if x <= lo:
        cl = Fraction(-1 if snorm else 0)
    elif x >= 1.0:
        cl = Fraction(1)
    else:
        cl = Fraction(x)
    scale = 32767 if snorm else 65535
    return round(cl * scale) & 0xFFFF


def _build_ref_table(snorm: bool) -> list[int]:
    return [ref_norm16_f16(b, snorm) for b in range(65536)]


_REF_TABLES: dict[bool, list[int]] = {True: _build_ref_table(True), False: _build_ref_table(False)}


def expected_pknorm_f16(
    op: str,
    src0: int,
    src1: int,
    *,
    sel: tuple[int, int] = (0, 0),
    dhi: bool | int = False,
    clamp: bool = False,
    src0_abs: bool = False,
    src0_neg: bool = False,
    src1_abs: bool = False,
    src1_neg: bool = False,
) -> int:
    """Reference 32-bit packed result for v_cvt_pknorm_{i,u}16_f16; clamp is a no-op, dhi shifts << 16."""
    if op not in OPS:
        raise ValueError(f"unsupported op: {op}")
    del clamp  # measured hardware no-op
    snorm = op == "v_cvt_pknorm_i16_f16"
    tab = _REF_TABLES[snorm]
    halves = []
    for raw, hi, is_abs, is_neg in (
        (src0, sel[0], src0_abs, src0_neg),
        (src1, sel[1], src1_abs, src1_neg),
    ):
        b = (raw >> (16 * hi)) & 0xFFFF
        if is_abs:
            b &= 0x7FFF
        if is_neg:
            b ^= 0x8000
        halves.append(tab[b])
    packed = halves[0] | (halves[1] << 16)
    if dhi:
        return (packed << 16) & 0xFFFFFFFF
    return packed


def sweep_kernel_lines(op: str, mod: str, half: int, tag: int) -> list[str]:
    """Raw GFX10 kernel sweeping 16,384 consecutive f16 pairs (32,768 values) using unique loop label."""
    if half not in (0, 1):
        raise ValueError(f"half must be 0 or 1, got {half}")
    lbl = f".L_pknorm_sweep_{tag}"
    base_val = half * 32768
    inst = f"{op} v8, v1, v2" + (f" {mod}" if mod else "")
    lines = [
        "s_mov_b32 s4, 0",
        f"{lbl}:",
        "v_lshlrev_b32 v3, 6, s4",
        "v_add_nc_u32 v3, v3, v0",
        "v_lshlrev_b32 v1, 1, v3",
    ]
    if base_val:
        lines.append(f"v_add_nc_u32 v1, {base_val:#x}, v1")
    lines += [
        "v_add_nc_u32 v2, 1, v1",
        "v_or_b32 v1, 0x3c000000, v1",
        "v_or_b32 v2, 0xbc000000, v2",
        "v_mov_b32_e32 v8, 0xdeadbeef",
        inst,
        "v_lshlrev_b32 v4, 2, v3",
        "global_store_dword v4, v8, s[2:3]",
        "s_add_u32 s4, s4, 1",
        "s_cmp_lt_u32 s4, 256",
        f"s_cbranch_scc1 {lbl}",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    return lines


def _format_src(reg_or_asm: str, is_abs: bool, is_neg: bool) -> str:
    s = f"|{reg_or_asm}|" if is_abs else reg_or_asm
    return f"-{s}" if is_neg else s


def case_instruction(op: str, c: Case, clamp: bool) -> list[str]:
    """Emit raw GFX10 setup + VOP3 instruction lines storing result in v8."""
    lines: list[str] = []
    if c.raw_omod is not None:
        lines.append(f"v_mov_b32_e32 v1, {c.src0:#010x}")
        lines.append(f"v_mov_b32_e32 v2, {c.src1:#010x}")
        lines.append(f"v_mov_b32_e32 v8, {c.init_dst:#010x}")
        dw0 = 0xD7000008 | (_OPCODES[op] << 16) | (0x8000 if clamp else 0)
        dw1 = 0x00020501 | ((c.raw_omod & 3) << 27)
        lines.append(f".long {dw0:#010x}, {dw1:#010x}")
        return lines

    if c.alias == "src0":
        lines.append(f"v_mov_b32_e32 v8, {c.src0:#010x}")
        lines.append(f"v_mov_b32_e32 v2, {c.src1:#010x}")
        s0_base, s1_base = "v8", "v2"
    elif c.alias == "src1":
        lines.append(f"v_mov_b32_e32 v1, {c.src0:#010x}")
        lines.append(f"v_mov_b32_e32 v8, {c.src1:#010x}")
        s0_base, s1_base = "v1", "v8"
    else:
        if c.sgpr0:
            lines.append(f"s_mov_b32 s4, {c.src0:#010x}")
            s0_base = "s4"
        elif c.src0_asm is not None:
            s0_base = c.src0_asm
        else:
            lines.append(f"v_mov_b32_e32 v1, {c.src0:#010x}")
            s0_base = "v1"
        if c.src1_asm is not None:
            s1_base = c.src1_asm
        else:
            lines.append(f"v_mov_b32_e32 v2, {c.src1:#010x}")
            s1_base = "v2"
        lines.append(f"v_mov_b32_e32 v8, {c.init_dst:#010x}")

    s0_op = _format_src(s0_base, c.src0_abs, c.src0_neg)
    s1_op = _format_src(s1_base, c.src1_abs, c.src1_neg)
    mods: list[str] = []
    if c.emit_opsel or c.sel != (0, 0) or c.dhi:
        mods.append(f"op_sel:[{c.sel[0]},{c.sel[1]},{int(bool(c.dhi))}]")
    if clamp:
        mods.append("clamp")
    suffix = (" " + " ".join(mods)) if mods else ""
    lines.append(f"{op} v8, {s0_op}, {s1_op}{suffix}")
    return lines


def cases_kernel_lines(op: str, clamp: bool) -> list[str]:
    """Raw GFX10 kernel running all targeted CASES sequentially on lane 0."""
    lines: list[str] = []
    for idx, c in enumerate(CASES):
        lines.extend(case_instruction(op, c, clamp))
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v8, s[2:3]")
    lines.append("s_waitcnt vmcnt(0)")
    lines.append("s_endpgm")
    return lines


def build_jobs(ops: list[str], waves: list[int], *, exhaustive: bool = True):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels, meta = [], []
    tag = 0
    for wave in waves:
        for op in ops:
            if exhaustive:
                for mod, clamp, dhi in SWEEP_MODES:
                    for half in (0, 1):
                        kernels.append(
                            asm.Kernel(
                                sweep_kernel_lines(op, mod, half, tag),
                                wave=wave,
                                threads=64,
                                words=SWEEP_WORDS_PER_HALF,
                                raw=True,
                            )
                        )
                        meta.append(("sweep", wave, op, clamp, dhi, half))
                        tag += 1
            for clamp in CLAMP_MODES:
                kernels.append(
                    asm.Kernel(
                        cases_kernel_lines(op, clamp),
                        wave=wave,
                        threads=1,
                        words=len(CASES),
                        raw=True,
                    )
                )
                meta.append(("cases", wave, op, clamp, False, 0))
    return kernels, meta


def evaluate_results(ops: list[str], waves: list[int], meta, raw_results, *, exhaustive: bool = True) -> dict:
    """Compare hardware outputs against exact Fraction reference and check wave/clamp/omod parity."""
    sweep_halves: dict[tuple[int, str, bool, bool, int], list[int]] = {}
    case_words: dict[tuple[int, str, bool], list[int]] = {}
    harness_errors: list[dict] = []

    for entry, res in zip(meta, raw_results):
        kind, wave, op, clamp, dhi, half = entry
        want_words = SWEEP_WORDS_PER_HALF if kind == "sweep" else len(CASES)
        if isinstance(res, str):
            harness_errors.append({"kind": kind, "wave": wave, "op": op, "clamp": clamp, "dhi": dhi, "half": half, "error": res})
            continue
        if len(res) != want_words:
            harness_errors.append({
                "kind": kind,
                "wave": wave,
                "op": op,
                "clamp": clamp,
                "dhi": dhi,
                "half": half,
                "error": f"expected {want_words} words, got {len(res)}",
            })
            continue
        words = [w & 0xFFFFFFFF for w in res]
        if kind == "sweep":
            sweep_halves[(wave, op, clamp, dhi, half)] = words
        else:
            case_words[(wave, op, clamp)] = words

    if harness_errors:
        return {
            "exhaustive": exhaustive,
            "sweep_words": 0,
            "sweep_halves": 0,
            "case_words": 0,
            "total_words": 0,
            "mismatches": 0,
            "wave_diffs": 0,
            "clamp_diffs": 0,
            "raw_omod_diffs": 0,
            "harness_errors": harness_errors,
            "first_mismatches": [],
            "first_wave_diffs": [],
            "first_clamp_diffs": [],
            "first_raw_omod_diffs": [],
        }

    mismatches = 0
    first_mismatches: list[dict] = []
    sweep_by_key: dict[tuple[int, str, bool, bool], list[int]] = {}
    sweep_word_count = 0

    if exhaustive:
        for wave in waves:
            for op in ops:
                tab = _REF_TABLES[op == "v_cvt_pknorm_i16_f16"]
                for _, clamp, dhi in SWEEP_MODES:
                    words = sweep_halves[(wave, op, clamp, dhi, 0)] + sweep_halves[(wave, op, clamp, dhi, 1)]
                    sweep_by_key[(wave, op, clamp, dhi)] = words
                    sweep_word_count += len(words)
                    for idx, got in enumerate(words):
                        lo = tab[2 * idx]
                        hi = tab[2 * idx + 1]
                        want = ((lo << 16) if dhi else (lo | (hi << 16))) & 0xFFFFFFFF
                        if got != want:
                            mismatches += 1
                            if len(first_mismatches) < 16:
                                first_mismatches.append({
                                    "kind": "sweep",
                                    "wave": wave,
                                    "op": op,
                                    "clamp": clamp,
                                    "dhi": dhi,
                                    "case": f"f16({2 * idx:#06x},{2 * idx + 1:#06x})",
                                    "got": f"0x{got:08x}",
                                    "want": f"0x{want:08x}",
                                })

    case_word_count = 0
    for wave in waves:
        for op in ops:
            got_list = case_words[(wave, op, False)] + case_words[(wave, op, True)]
            case_word_count += len(got_list)
            for clamp in CLAMP_MODES:
                for c, got in zip(CASES, case_words[(wave, op, clamp)]):
                    want = expected_pknorm_f16(
                        op,
                        c.src0,
                        c.src1,
                        sel=c.sel,
                        dhi=bool(c.dhi),
                        clamp=clamp,
                        src0_abs=c.src0_abs,
                        src0_neg=c.src0_neg,
                        src1_abs=c.src1_abs,
                        src1_neg=c.src1_neg,
                    )
                    if got != want:
                        mismatches += 1
                        if len(first_mismatches) < 16:
                            first_mismatches.append({
                                "kind": "case",
                                "wave": wave,
                                "op": op,
                                "clamp": clamp,
                                "dhi": bool(c.dhi),
                                "case": c.name,
                                "got": f"0x{got:08x}",
                                "want": f"0x{want:08x}",
                            })

    wave_diffs = 0
    first_wave_diffs: list[dict] = []
    if len(waves) > 1:
        w0 = waves[0]
        for w in waves[1:]:
            for op in ops:
                if exhaustive:
                    for _, clamp, dhi in SWEEP_MODES:
                        for idx, (a_w, b_w) in enumerate(
                            zip(sweep_by_key[(w0, op, clamp, dhi)], sweep_by_key[(w, op, clamp, dhi)])
                        ):
                            if a_w != b_w:
                                wave_diffs += 1
                                if len(first_wave_diffs) < 16:
                                    first_wave_diffs.append({
                                        "kind": "sweep",
                                        "op": op,
                                        "clamp": clamp,
                                        "dhi": dhi,
                                        "case": f"f16({2 * idx:#06x},{2 * idx + 1:#06x})",
                                        f"w{w0}": f"0x{a_w:08x}",
                                        f"w{w}": f"0x{b_w:08x}",
                                    })
                for clamp in CLAMP_MODES:
                    for c, a_w, b_w in zip(CASES, case_words[(w0, op, clamp)], case_words[(w, op, clamp)]):
                        if a_w != b_w:
                            wave_diffs += 1
                            if len(first_wave_diffs) < 16:
                                first_wave_diffs.append({
                                    "kind": "case",
                                    "op": op,
                                    "clamp": clamp,
                                    "dhi": bool(c.dhi),
                                    "case": c.name,
                                    f"w{w0}": f"0x{a_w:08x}",
                                    f"w{w}": f"0x{b_w:08x}",
                                })

    clamp_diffs = 0
    first_clamp_diffs: list[dict] = []
    for wave in waves:
        for op in ops:
            if exhaustive:
                for dhi in (False, True):
                    for idx, (pw, cw) in enumerate(
                        zip(sweep_by_key[(wave, op, False, dhi)], sweep_by_key[(wave, op, True, dhi)])
                    ):
                        if pw != cw:
                            clamp_diffs += 1
                            if len(first_clamp_diffs) < 16:
                                first_clamp_diffs.append({
                                    "kind": "sweep",
                                    "wave": wave,
                                    "op": op,
                                    "dhi": dhi,
                                    "case": f"f16({2 * idx:#06x},{2 * idx + 1:#06x})",
                                    "plain": f"0x{pw:08x}",
                                    "clamp": f"0x{cw:08x}",
                                })
            for c, pw, cw in zip(CASES, case_words[(wave, op, False)], case_words[(wave, op, True)]):
                if pw != cw:
                    clamp_diffs += 1
                    if len(first_clamp_diffs) < 16:
                        first_clamp_diffs.append({
                            "kind": "case",
                            "wave": wave,
                            "op": op,
                            "dhi": bool(c.dhi),
                            "case": c.name,
                            "plain": f"0x{pw:08x}",
                            "clamp": f"0x{cw:08x}",
                        })

    raw_omod_diffs = 0
    first_raw_omod_diffs: list[dict] = []
    raw_indices = [idx for idx, c in enumerate(CASES) if c.raw_omod is not None]
    if raw_indices:
        base_idx = next(idx for idx in raw_indices if CASES[idx].raw_omod == 0)
        for wave in waves:
            for op in ops:
                for clamp in CLAMP_MODES:
                    words = case_words[(wave, op, clamp)]
                    base_w = words[base_idx]
                    for idx in raw_indices:
                        if idx == base_idx:
                            continue
                        w = words[idx]
                        if w != base_w:
                            raw_omod_diffs += 1
                            if len(first_raw_omod_diffs) < 16:
                                first_raw_omod_diffs.append({
                                    "wave": wave,
                                    "op": op,
                                    "clamp": clamp,
                                    "case": CASES[idx].name,
                                    "omod0": f"0x{base_w:08x}",
                                    "got": f"0x{w:08x}",
                                })

    return {
        "exhaustive": exhaustive,
        "sweep_words": sweep_word_count,
        "sweep_halves": sweep_word_count * 2,
        "case_words": case_word_count,
        "total_words": sweep_word_count + case_word_count,
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "clamp_diffs": clamp_diffs,
        "raw_omod_diffs": raw_omod_diffs,
        "harness_errors": [],
        "first_mismatches": first_mismatches,
        "first_wave_diffs": first_wave_diffs,
        "first_clamp_diffs": first_clamp_diffs,
        "first_raw_omod_diffs": first_raw_omod_diffs,
    }


def write_report(path: str, doc: dict) -> bool:
    """Write JSON report with LF line endings, creating parent folders."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e}", file=sys.stderr)
        return False
    return True


def self_test() -> int:
    """Verify the reference model, kernel assembly, and synthetic evaluator without GPU or clang."""
    # Exact single-rounding double-rounding traps (0x3801 = 0.5 + 2^-11)
    assert ref_norm16_f16(0x3801, True) == 16399
    assert ref_norm16_f16(0xB801, True) == (-16399) & 0xFFFF
    assert ref_norm16_f16(0x3801, False) == 32799
    assert ref_norm16_f16(0x37FF, True) == 16376
    assert ref_norm16_f16(0x37FF, False) == 32752
    # NaNs, +-inf, +-0, subnormals
    for nan in (0x7E00, 0xFE00, 0x7C01, 0xFC01):
        assert ref_norm16_f16(nan, True) == 0
        assert ref_norm16_f16(nan, False) == 0
    assert ref_norm16_f16(0x7C00, True) == 32767
    assert ref_norm16_f16(0xFC00, True) == (-32767) & 0xFFFF
    assert ref_norm16_f16(0x7C00, False) == 65535
    assert ref_norm16_f16(0xFC00, False) == 0
    assert ref_norm16_f16(0x8000, True) == 0
    assert ref_norm16_f16(0x0001, True) == 0
    # clamp is no-op; dhi shifts left by 16 zeroing D[15:0]
    assert expected_pknorm_f16("v_cvt_pknorm_i16_f16", 0x3C003400, 0xBC003801, clamp=False) == 0x400F2000
    assert expected_pknorm_f16("v_cvt_pknorm_i16_f16", 0x3C003400, 0xBC003801, clamp=True) == 0x400F2000
    assert expected_pknorm_f16("v_cvt_pknorm_i16_f16", 0x3C003400, 0xBC003801, dhi=True) == 0x20000000
    assert expected_pknorm_f16("v_cvt_pknorm_i16_f16", 0x3C003400, 0xBC003801, sel=(1, 0), dhi=True, clamp=True) == 0x7FFF0000
    assert expected_pknorm_f16("v_cvt_pknorm_u16_f16", 0x3C003400, 0xBC003801, sel=(1, 1), dhi=True) == 0xFFFF0000
    # Kernel assembly sanity
    sk = sweep_kernel_lines("v_cvt_pknorm_i16_f16", "op_sel:[0,0,1] clamp", 1, 7)
    assert sk[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
    assert "v_cvt_pknorm_i16_f16 v8, v1, v2 op_sel:[0,0,1] clamp" in sk
    assert ".L_pknorm_sweep_7:" in sk
    ck = cases_kernel_lines("v_cvt_pknorm_u16_f16", True)
    assert "v_cvt_pknorm_u16_f16 v8, v1, v2 op_sel:[1,0,1] clamp" in ck
    assert "v_cvt_pknorm_u16_f16 v8, 0.5, -0.5 clamp" in ck
    assert ".long 0xd7138008, 0x18020501" in ck
    # Synthetic evaluator pass check
    meta = [
        ("cases", 32, "v_cvt_pknorm_i16_f16", False, False, 0),
        ("cases", 32, "v_cvt_pknorm_i16_f16", True, False, 0),
    ]
    fake_words = [
        expected_pknorm_f16(
            "v_cvt_pknorm_i16_f16", c.src0, c.src1,
            sel=c.sel, dhi=bool(c.dhi), clamp=False,
            src0_abs=c.src0_abs, src0_neg=c.src0_neg, src1_abs=c.src1_abs, src1_neg=c.src1_neg,
        )
        for c in CASES
    ]
    ev = evaluate_results(["v_cvt_pknorm_i16_f16"], [32], meta, [fake_words, fake_words], exhaustive=False)
    assert ev["mismatches"] == 0 and ev["clamp_diffs"] == 0 and ev["raw_omod_diffs"] == 0
    print(f"self-test ok ({len(CASES)} cases, 65,536 f16 ref entries)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--no-exhaustive", action="store_true", help="run targeted CASES only without 65,536-input sweep")
    ap.add_argument("--json", help="write measurement summary JSON here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    wave_tokens = [w.strip() for w in a.waves.split(",") if w.strip()]
    bad_ops = [o for o in ops if o not in OPS]
    bad_waves = [w for w in wave_tokens if w not in ("32", "64")]
    if bad_ops or bad_waves or not ops or not wave_tokens:
        ap.error("unknown or empty: " + ", ".join(bad_ops + bad_waves or ["--ops/--waves"]))
    waves = [int(w) for w in wave_tokens]
    if a.json and os.path.isdir(a.json):
        ap.error(f"--json {a.json} is a folder, not a file")
    exhaustive = not a.no_exhaustive

    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10")
            return 77
        print(f"hardware oracle: {oracle.device} ({oracle.arch})")
        kernels, meta = build_jobs(ops, waves, exhaustive=exhaustive)
        raw_results = oracle.run(kernels)
        summary = evaluate_results(ops, waves, meta, raw_results, exhaustive=exhaustive)
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()

    report = {
        "device": str(oracle.device),
        "arch": oracle.arch,
        "waves": waves,
        "ops": ops,
        "sweep_modes": [m[0] or "plain" for m in SWEEP_MODES] if exhaustive else [],
        "cases": len(CASES),
        **summary,
    }
    if a.json and not write_report(a.json, report):
        return 2

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: {err['kind']} wave{err['wave']} {err['op']} clamp={err['clamp']}: {err['error']}")
        return 2

    print(
        f"sweep_halves={summary['sweep_halves']} ({summary['sweep_words']} words), "
        f"case_words={summary['case_words']} across {len(waves)} wave(s) x {len(ops)} op(s): "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}, "
        f"clamp_diffs={summary['clamp_diffs']}, raw_omod_diffs={summary['raw_omod_diffs']}"
    )
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH {m['kind']} wave{m['wave']} {m['op']} clamp={m['clamp']} {m['case']}: got {m['got']}, want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['kind']} {d['op']} clamp={d['clamp']} {d['case']}: {d}")
    for d in summary["first_clamp_diffs"]:
        print(f"  CLAMP DIFF {d['kind']} wave{d['wave']} {d['op']} {d['case']}: plain={d['plain']} clamp={d['clamp']}")
    for d in summary["first_raw_omod_diffs"]:
        print(f"  RAW OMOD DIFF wave{d['wave']} {d['op']} clamp={d['clamp']} {d['case']}: omod0={d['omod0']} got={d['got']}")

    if (
        summary["mismatches"]
        or summary["wave_diffs"]
        or summary["clamp_diffs"]
        or summary["raw_omod_diffs"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
