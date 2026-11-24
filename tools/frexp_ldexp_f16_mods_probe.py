r"""Measure RDNA2 VOP3 output modifiers on v_frexp_mant_f16, v_frexp_exp_i16_f16, and v_ldexp_f16. Standard library only.

Probes VOP3 ``clamp``, ``mul:2``, ``mul:4``, ``div:2``, and ``clamp`` combined with ``mul:*``/``div:*`` across
curated binary16 edge inputs (``+-0``, ``+-min_sub``, ``+-max_sub``, ``+-min_norm``, ``+-0.25``, ``+-0.5``,
``+-(1-2^-11)``, ``+-1.0``, ``+-(1+2^-10)``, ``+-2.0``, ``+-max_finite``, ``+-inf``, ``+-sNaN``, ``+-qNaN``),
source ``abs``/``neg`` modifiers, dirty upper-16-bit source halves, and destination high-half preservation across
wave32 and wave64 on GFX10 (``gfx1036``) under:
- Console MODE (``0x1c0``: ``IEEE=0``, ``DX10_CLAMP=1``, ``denorm32=0``, ``denorm16_64=3``);
- Flush control MODE (``0x100``: ``denorm16_64=0``);
- IEEE control MODE (``0x3c0``: ``IEEE=1``, ``denorm16_64=3``).

Usage:
    python -I tools/frexp_ldexp_f16_mods_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/frexp_ldexp_f16_mods_probe.py --self-test

Exit 0 when all hardware results match the expected RDNA2 model with zero wave32/wave64 diffs and zero console-MODE
omod diffs; 1 on a mismatch or parity failure; 2 on a harness/CLI error; 77 when the hardware oracle is unavailable or
the GPU is not GFX10. Nothing here reads game files.
"""
from dataclasses import dataclass
import json
import math
import os
import struct
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OPS: tuple[str, ...] = ("v_frexp_mant_f16", "v_frexp_exp_i16_f16", "v_ldexp_f16")
MODS: tuple[str, ...] = ("", "clamp", "mul:2", "mul:4", "div:2", "clamp mul:2", "clamp mul:4", "clamp div:2")

EDGE_F16: tuple[int, ...] = (
    0x0000, 0x8000,  # +-0.0
    0x0001, 0x8001,  # +-min_sub
    0x03FF, 0x83FF,  # +-max_sub
    0x0400, 0x8400,  # +-min_norm
    0x3400, 0xB400,  # +-0.25
    0x3800, 0xB800,  # +-0.5
    0x3BFF, 0xBBFF,  # +-(1 - 2^-11)
    0x3C00, 0xBC00,  # +-1.0
    0x3C01, 0xBC01,  # +-(1 + 2^-10)
    0x4000, 0xC000,  # +-2.0
    0x7BFF, 0xFBFF,  # +-max_finite
    0x7C00, 0xFC00,  # +-inf
    0x7C01, 0xFC01,  # +-sNaN
    0x7E00, 0xFE00,  # +-qNaN
    0x7E55, 0xFE55,  # +-qNaN with payload
)
LDEXP_EXPS: tuple[int, ...] = (-25, -1, 0, 1, 16)

MODE_CONSOLE = 0x1C0  # IEEE=0, DX10_CLAMP=1, denorm32=0, denorm16_64=3
MODE_FLUSH16 = 0x100  # IEEE=0, DX10_CLAMP=1, denorm32=0, denorm16_64=0
MODE_IEEE = 0x3C0     # IEEE=1, DX10_CLAMP=1, denorm32=0, denorm16_64=3
DEFAULT_INIT_DST = 0xDEADBEEF


@dataclass(frozen=True)
class SrcModCase:
    name: str
    op: str
    h: int
    n: int
    mod: str
    src0_abs: bool = False
    src0_neg: bool = False
    init_dst: int = DEFAULT_INIT_DST


SRC_MOD_CASES: tuple[SrcModCase, ...] = (
    SrcModCase("mant_neg_clamp", "v_frexp_mant_f16", 0xBC00, 0, "clamp", src0_neg=True),
    SrcModCase("mant_abs_clamp_mul2", "v_frexp_mant_f16", 0xC000, 0, "clamp mul:2", src0_abs=True),
    SrcModCase("mant_negabs_div2", "v_frexp_mant_f16", 0x3C00, 0, "div:2", src0_abs=True, src0_neg=True),
    SrcModCase("mant_neg_qnan_clamp", "v_frexp_mant_f16", 0xFE55, 0, "clamp", src0_neg=True),
    SrcModCase("exp_neg_clamp", "v_frexp_exp_i16_f16", 0xC800, 0, "clamp", src0_neg=True),
    SrcModCase("exp_abs_mul2", "v_frexp_exp_i16_f16", 0xC800, 0, "mul:2", src0_abs=True),
    SrcModCase("exp_negabs_clamp_div2", "v_frexp_exp_i16_f16", 0x4800, 0, "clamp div:2", src0_abs=True, src0_neg=True),
    SrcModCase("ldexp_neg_clamp", "v_ldexp_f16", 0xBC00, 1, "clamp", src0_neg=True),
    SrcModCase("ldexp_abs_clamp_mul2", "v_ldexp_f16", 0xBC00, 1, "clamp mul:2", src0_abs=True),
    SrcModCase("ldexp_negabs_div2", "v_ldexp_f16", 0x3C00, 1, "div:2", src0_abs=True, src0_neg=True),
    SrcModCase("ldexp_neg_inf_clamp", "v_ldexp_f16", 0xFC00, 0, "clamp", src0_neg=True),
)


@dataclass(frozen=True)
class ControlCase:
    name: str
    op: str
    h: int
    n: int
    mod: str
    want_lo: int
    init_dst: int = DEFAULT_INIT_DST


FLUSH_CONTROL_CASES: tuple[ControlCase, ...] = (
    ControlCase("mant_plain", "v_frexp_mant_f16", 0x3C00, 0, "", 0x3800),
    ControlCase("mant_mul2", "v_frexp_mant_f16", 0x3C00, 0, "mul:2", 0x3C00),
    ControlCase("mant_mul4", "v_frexp_mant_f16", 0x3C00, 0, "mul:4", 0x4000),
    ControlCase("mant_div2", "v_frexp_mant_f16", 0x3C00, 0, "div:2", 0x3400),
    ControlCase("mant_clamp_mul4", "v_frexp_mant_f16", 0x3C00, 0, "clamp mul:4", 0x3C00),
    ControlCase("ldexp_plain", "v_ldexp_f16", 0x3800, 1, "", 0x3C00),
    ControlCase("ldexp_mul2", "v_ldexp_f16", 0x3800, 1, "mul:2", 0x4000),
    ControlCase("ldexp_mul4", "v_ldexp_f16", 0x3800, 1, "mul:4", 0x4400),
    ControlCase("ldexp_div2", "v_ldexp_f16", 0x3800, 1, "div:2", 0x3800),
    ControlCase("ldexp_clamp_mul4", "v_ldexp_f16", 0x3800, 1, "clamp mul:4", 0x3C00),
    # 16-bit integer result ignores clamp and omod even when denorm_mode_16_64=0.
    *(ControlCase(f"exp_{m or 'plain'}", "v_frexp_exp_i16_f16", 0x4800, 0, m, 0x0004) for m in MODS),
)


def _h2f(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def _f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _is_finite_nonzero(h: int) -> bool:
    return ((h >> 10) & 0x1F) != 31 and (h & 0x7FFF) != 0


def ref_clamp_f16(h: int) -> int:
    """DX10_CLAMP [0.0, 1.0] saturation on a binary16 word: negative (including -0.0) and NaN -> +0.0."""
    h &= 0xFFFF
    if (h & 0x8000) or (((h >> 10) & 0x1F) == 31 and (h & 0x3FF) != 0):
        return 0x0000
    return 0x3C00 if h > 0x3C00 else h


def expected_word(
    op: str,
    h: int,
    n: int = 0,
    *,
    mod: str = "",
    init_dst: int = DEFAULT_INIT_DST,
    src0_abs: bool = False,
    src0_neg: bool = False,
) -> int:
    """Expected 32-bit vdst value under console MODE (0x1c0: IEEE=0, DX10_CLAMP=1, denorm16_64=3)."""
    if op not in OPS:
        raise ValueError(f"unsupported op: {op}")
    b = h & 0xFFFF
    if src0_abs:
        b &= 0x7FFF
    if src0_neg:
        b ^= 0x8000

    if op == "v_frexp_mant_f16":
        lo = _f2h(math.frexp(_h2f(b))[0]) if _is_finite_nonzero(b) else b
    elif op == "v_frexp_exp_i16_f16":
        lo = (math.frexp(_h2f(b))[1] & 0xFFFF) if _is_finite_nonzero(b) else 0
    else:
        if _is_finite_nonzero(b):
            n16 = n & 0xFFFF
            if n16 & 0x8000:
                n16 -= 0x10000
            try:
                lo = _f2h(math.ldexp(_h2f(b), max(-64, min(64, n16))))
            except OverflowError:
                lo = (b & 0x8000) | 0x7C00
        else:
            lo = b

    if "clamp" in mod.split() and op != "v_frexp_exp_i16_f16":
        lo = ref_clamp_f16(lo)
    return (init_dst & 0xFFFF0000) | (lo & 0xFFFF)


def _format_src(reg: str, is_abs: bool, is_neg: bool) -> str:
    s = f"|{reg}|" if is_abs else reg
    return f"-{s}" if is_neg else s


def _emit_inst(op: str, mod: str, *, src0_abs: bool = False, src0_neg: bool = False) -> str:
    s0 = _format_src("v1", src0_abs, src0_neg)
    suffix = f" {mod}" if mod else ""
    if op == "v_ldexp_f16":
        return f"v_ldexp_f16 v8, {s0}, v2{suffix}"
    return f"{op}_e64 v8, {s0}{suffix}"


def mode_grid_kernel_lines(op: str, mod: str, mode_val: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Emit raw GFX10 kernel sweeping EDGE_F16 (or EDGE_F16 x LDEXP_EXPS) under ``mode_val``."""
    pairs = [(h, n) for h in EDGE_F16 for n in LDEXP_EXPS] if op == "v_ldexp_f16" else [(h, 0) for h in EDGE_F16]
    lines = [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 10), {mode_val:#x}"]
    for idx, (h, n) in enumerate(pairs):
        lines.append(f"v_mov_b32_e32 v1, {0xABCD0000 | (h & 0xFFFF):#010x}")
        if op == "v_ldexp_f16":
            lines.append(f"v_mov_b32_e32 v2, {0x12340000 | (n & 0xFFFF):#010x}")
        lines.append(f"v_mov_b32_e32 v8, {DEFAULT_INIT_DST:#010x}")
        lines.append(_emit_inst(op, mod))
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v8, s[2:3]")
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines, pairs


def src_mod_kernel_lines(cases: tuple[SrcModCase, ...], mode_val: int) -> list[str]:
    """Emit raw GFX10 kernel running ``cases`` sequentially on lane 0 under ``mode_val``."""
    lines = [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 10), {mode_val:#x}"]
    for idx, c in enumerate(cases):
        lines.append(f"v_mov_b32_e32 v1, {0xABCD0000 | (c.h & 0xFFFF):#010x}")
        if c.op == "v_ldexp_f16":
            lines.append(f"v_mov_b32_e32 v2, {0x12340000 | (c.n & 0xFFFF):#010x}")
        lines.append(f"v_mov_b32_e32 v8, {c.init_dst:#010x}")
        lines.append(_emit_inst(c.op, c.mod, src0_abs=c.src0_abs, src0_neg=c.src0_neg))
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v8, s[2:3]")
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines


def control_kernel_lines(cases: tuple[ControlCase, ...], mode_val: int) -> list[str]:
    """Emit raw GFX10 kernel running ``cases`` sequentially under ``mode_val``."""
    lines = [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 10), {mode_val:#x}"]
    for idx, c in enumerate(cases):
        lines.append(f"v_mov_b32_e32 v1, {0xABCD0000 | (c.h & 0xFFFF):#010x}")
        if c.op == "v_ldexp_f16":
            lines.append(f"v_mov_b32_e32 v2, {0x12340000 | (c.n & 0xFFFF):#010x}")
        lines.append(f"v_mov_b32_e32 v8, {c.init_dst:#010x}")
        lines.append(_emit_inst(c.op, c.mod))
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v8, s[2:3]")
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines


def ieee_control_cases(ops: list[str]) -> list[ControlCase]:
    """Selected control cases verified under MODE_IEEE (0x3c0)."""
    out = []
    for op in ops:
        if op == "v_frexp_mant_f16":
            h, n = 0xBC00, 0
        elif op == "v_frexp_exp_i16_f16":
            h, n = 0x4800, 0
        else:
            h, n = 0xBC00, 1
        for mod in MODS:
            want = expected_word(op, h, n, mod=mod) & 0xFFFF
            out.append(ControlCase(f"ieee_{op}_{mod or 'plain'}", op, h, n, mod, want))
    return out


def build_jobs(ops: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels, meta = [], []
    src_cases = tuple(c for c in SRC_MOD_CASES if c.op in ops)
    flush_cases = tuple(c for c in FLUSH_CONTROL_CASES if c.op in ops)
    ieee_cases = tuple(ieee_control_cases(ops))
    for wave in waves:
        for op in ops:
            for mod in MODS:
                lines, pairs = mode_grid_kernel_lines(op, mod, MODE_CONSOLE)
                kernels.append(asm.Kernel(lines, wave=wave, threads=1, words=len(pairs), raw=True))
                meta.append(("grid", wave, op, mod, pairs))
        if src_cases:
            kernels.append(asm.Kernel(src_mod_kernel_lines(src_cases, MODE_CONSOLE), wave=wave, threads=1,
                                      words=len(src_cases), raw=True))
            meta.append(("src_mods", wave, "all", "src_mods", src_cases))
        if flush_cases:
            kernels.append(asm.Kernel(control_kernel_lines(flush_cases, MODE_FLUSH16), wave=wave, threads=1,
                                      words=len(flush_cases), raw=True))
            meta.append(("flush_ctrl", wave, "all", "flush_ctrl", flush_cases))
        if ieee_cases:
            kernels.append(asm.Kernel(control_kernel_lines(ieee_cases, MODE_IEEE), wave=wave, threads=1,
                                      words=len(ieee_cases), raw=True))
            meta.append(("ieee_ctrl", wave, "all", "ieee_ctrl", ieee_cases))
    return kernels, meta


def evaluate_results(ops: list[str], waves: list[int], meta, raw_results) -> dict:
    """Compare hardware outputs against expected RDNA2 semantics and check wave and omod parity."""
    grid_words: dict[tuple[int, str, str], list[int]] = {}
    src_words: dict[int, list[int]] = {}
    flush_words: dict[int, list[int]] = {}
    ieee_words: dict[int, list[int]] = {}
    harness_errors: list[dict] = []

    for (kind, wave, op, mod, items), res in zip(meta, raw_results):
        want_words = len(items)
        if isinstance(res, str):
            harness_errors.append({"kind": kind, "wave": wave, "op": op, "mod": mod, "error": res})
            continue
        if len(res) != want_words:
            harness_errors.append({
                "kind": kind,
                "wave": wave,
                "op": op,
                "mod": mod,
                "error": f"expected {want_words} words, got {len(res)}",
            })
            continue
        words = [w & 0xFFFFFFFF for w in res]
        if kind == "grid":
            grid_words[(wave, op, mod)] = words
        elif kind == "src_mods":
            src_words[wave] = words
        elif kind == "flush_ctrl":
            flush_words[wave] = words
        elif kind == "ieee_ctrl":
            ieee_words[wave] = words

    if harness_errors:
        return {
            "total_words": 0,
            "grid_words": 0,
            "src_mod_words": 0,
            "control_words": 0,
            "mismatches": 0,
            "wave_diffs": 0,
            "omod_diffs": 0,
            "harness_errors": harness_errors,
            "first_mismatches": [],
            "first_wave_diffs": [],
            "first_omod_diffs": [],
        }

    mismatches = 0
    first_mismatches: list[dict] = []
    grid_word_count = 0
    src_word_count = 0
    control_word_count = 0

    def record_mismatch(entry: dict):
        nonlocal mismatches
        mismatches += 1
        if len(first_mismatches) < 16:
            first_mismatches.append(entry)

    src_cases = tuple(c for c in SRC_MOD_CASES if c.op in ops)
    flush_cases = tuple(c for c in FLUSH_CONTROL_CASES if c.op in ops)
    ieee_cases = tuple(ieee_control_cases(ops))

    for wave in waves:
        for op in ops:
            pairs = [(h, n) for h in EDGE_F16 for n in LDEXP_EXPS] if op == "v_ldexp_f16" else [(h, 0) for h in EDGE_F16]
            for mod in MODS:
                words = grid_words[(wave, op, mod)]
                grid_word_count += len(words)
                for (h, n), got in zip(pairs, words):
                    want = expected_word(op, h, n, mod=mod)
                    if got != want:
                        record_mismatch({
                            "kind": "grid", "wave": wave, "op": op, "mod": mod,
                            "case": f"h={h:#06x},n={n}", "got": f"0x{got:08x}", "want": f"0x{want:08x}",
                        })
        if src_cases:
            words = src_words[wave]
            src_word_count += len(words)
            for c, got in zip(src_cases, words):
                want = expected_word(c.op, c.h, c.n, mod=c.mod, init_dst=c.init_dst,
                                     src0_abs=c.src0_abs, src0_neg=c.src0_neg)
                if got != want:
                    record_mismatch({
                        "kind": "src_mods", "wave": wave, "op": c.op, "mod": c.mod,
                        "case": c.name, "got": f"0x{got:08x}", "want": f"0x{want:08x}",
                    })
        if flush_cases:
            words = flush_words[wave]
            control_word_count += len(words)
            for c, got in zip(flush_cases, words):
                want = (c.init_dst & 0xFFFF0000) | (c.want_lo & 0xFFFF)
                if got != want:
                    record_mismatch({
                        "kind": "flush_ctrl", "wave": wave, "op": c.op, "mod": c.mod,
                        "case": c.name, "got": f"0x{got:08x}", "want": f"0x{want:08x}",
                    })
        if ieee_cases:
            words = ieee_words[wave]
            control_word_count += len(words)
            for c, got in zip(ieee_cases, words):
                want = (c.init_dst & 0xFFFF0000) | (c.want_lo & 0xFFFF)
                if got != want:
                    record_mismatch({
                        "kind": "ieee_ctrl", "wave": wave, "op": c.op, "mod": c.mod,
                        "case": c.name, "got": f"0x{got:08x}", "want": f"0x{want:08x}",
                    })

    wave_diffs = 0
    first_wave_diffs: list[dict] = []
    if len(waves) > 1:
        w0 = waves[0]
        for w in waves[1:]:
            for op in ops:
                for mod in MODS:
                    for idx, (a_w, b_w) in enumerate(zip(grid_words[(w0, op, mod)], grid_words[(w, op, mod)])):
                        if a_w != b_w:
                            wave_diffs += 1
                            if len(first_wave_diffs) < 16:
                                first_wave_diffs.append({
                                    "kind": "grid", "op": op, "mod": mod, "index": idx,
                                    f"w{w0}": f"0x{a_w:08x}", f"w{w}": f"0x{b_w:08x}",
                                })
            for kind, table in (("src_mods", src_words), ("flush_ctrl", flush_words), ("ieee_ctrl", ieee_words)):
                if w0 in table and w in table:
                    for idx, (a_w, b_w) in enumerate(zip(table[w0], table[w])):
                        if a_w != b_w:
                            wave_diffs += 1
                            if len(first_wave_diffs) < 16:
                                first_wave_diffs.append({
                                    "kind": kind, "index": idx,
                                    f"w{w0}": f"0x{a_w:08x}", f"w{w}": f"0x{b_w:08x}",
                                })

    omod_diffs = 0
    first_omod_diffs: list[dict] = []
    for wave in waves:
        for op in ops:
            for base_mod, omods in (("", ("mul:2", "mul:4", "div:2")),
                                    ("clamp", ("clamp mul:2", "clamp mul:4", "clamp div:2"))):
                base = grid_words[(wave, op, base_mod)]
                for mod in omods:
                    for idx, (bw, mw) in enumerate(zip(base, grid_words[(wave, op, mod)])):
                        if bw != mw:
                            omod_diffs += 1
                            if len(first_omod_diffs) < 16:
                                first_omod_diffs.append({
                                    "wave": wave, "op": op, "base": base_mod or "plain", "mod": mod,
                                    "index": idx, "want": f"0x{bw:08x}", "got": f"0x{mw:08x}",
                                })
            if op == "v_frexp_exp_i16_f16":
                base = grid_words[(wave, op, "")]
                for idx, (bw, cw) in enumerate(zip(base, grid_words[(wave, op, "clamp")])):
                    if bw != cw:
                        omod_diffs += 1
                        if len(first_omod_diffs) < 16:
                            first_omod_diffs.append({
                                "wave": wave, "op": op, "base": "plain", "mod": "clamp",
                                "index": idx, "want": f"0x{bw:08x}", "got": f"0x{cw:08x}",
                            })

    return {
        "total_words": grid_word_count + src_word_count + control_word_count,
        "grid_words": grid_word_count,
        "src_mod_words": src_word_count,
        "control_words": control_word_count,
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "omod_diffs": omod_diffs,
        "harness_errors": [],
        "first_mismatches": first_mismatches,
        "first_wave_diffs": first_wave_diffs,
        "first_omod_diffs": first_omod_diffs,
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
    """Verify reference model, assembly emission, and evaluator without GPU or clang."""
    assert ref_clamp_f16(0x8000) == 0x0000
    assert ref_clamp_f16(0xFC00) == 0x0000
    assert ref_clamp_f16(0x7E00) == 0x0000
    assert ref_clamp_f16(0x7C01) == 0x0000
    assert ref_clamp_f16(0x7C00) == 0x3C00
    assert ref_clamp_f16(0x3800) == 0x3800

    # Console MODE expectations: omod is a no-op on all 3 ops; clamp applies only to frexp_mant and ldexp.
    assert expected_word("v_frexp_mant_f16", 0x3C00, mod="") == 0xDEAD3800
    assert expected_word("v_frexp_mant_f16", 0x3C00, mod="mul:2") == 0xDEAD3800
    assert expected_word("v_frexp_mant_f16", 0xBC00, mod="clamp") == 0xDEAD0000
    assert expected_word("v_frexp_mant_f16", 0x7C00, mod="clamp mul:4") == 0xDEAD3C00
    assert expected_word("v_frexp_exp_i16_f16", 0x4800, mod="") == 0xDEAD0004
    assert expected_word("v_frexp_exp_i16_f16", 0x4800, mod="clamp mul:2") == 0xDEAD0004
    assert expected_word("v_ldexp_f16", 0x3800, 1, mod="") == 0xDEAD3C00
    assert expected_word("v_ldexp_f16", 0x3800, 1, mod="mul:2") == 0xDEAD3C00
    assert expected_word("v_ldexp_f16", 0x3C00, 1, mod="clamp div:2") == 0xDEAD3C00
    assert expected_word("v_frexp_mant_f16", 0xBC00, mod="clamp", src0_neg=True) == 0xDEAD3800

    gl, pairs = mode_grid_kernel_lines("v_ldexp_f16", "clamp mul:2", MODE_CONSOLE)
    assert len(pairs) == len(EDGE_F16) * len(LDEXP_EXPS)
    assert "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 10), 0x1c0" in gl
    assert "v_ldexp_f16 v8, v1, v2 clamp mul:2" in gl
    assert gl[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]

    sl = src_mod_kernel_lines(SRC_MOD_CASES, MODE_CONSOLE)
    assert "v_frexp_mant_f16_e64 v8, -|v1| div:2" in sl
    assert "v_ldexp_f16 v8, |v1|, v2 clamp mul:2" in sl

    # Synthetic evaluator pass check across all 3 ops and wave 32
    ops = list(OPS)
    meta, raw = [], []
    for op in ops:
        op_pairs = [(h, n) for h in EDGE_F16 for n in LDEXP_EXPS] if op == "v_ldexp_f16" else [(h, 0) for h in EDGE_F16]
        for mod in MODS:
            meta.append(("grid", 32, op, mod, op_pairs))
            raw.append([expected_word(op, h, n, mod=mod) for h, n in op_pairs])
    meta.append(("src_mods", 32, "all", "src_mods", SRC_MOD_CASES))
    raw.append([
        expected_word(c.op, c.h, c.n, mod=c.mod, init_dst=c.init_dst, src0_abs=c.src0_abs, src0_neg=c.src0_neg)
        for c in SRC_MOD_CASES
    ])
    meta.append(("flush_ctrl", 32, "all", "flush_ctrl", FLUSH_CONTROL_CASES))
    raw.append([(c.init_dst & 0xFFFF0000) | (c.want_lo & 0xFFFF) for c in FLUSH_CONTROL_CASES])
    ieee_cases = tuple(ieee_control_cases(ops))
    meta.append(("ieee_ctrl", 32, "all", "ieee_ctrl", ieee_cases))
    raw.append([(c.init_dst & 0xFFFF0000) | (c.want_lo & 0xFFFF) for c in ieee_cases])

    ev = evaluate_results(ops, [32], meta, raw)
    assert ev["mismatches"] == 0 and ev["wave_diffs"] == 0 and ev["omod_diffs"] == 0, ev
    print(f"self-test ok ({ev['total_words']} synthetic words checked)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--waves", default="32,64")
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
        kernels, meta = build_jobs(ops, waves)
        raw_results = oracle.run(kernels)
        summary = evaluate_results(ops, waves, meta, raw_results)
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
        "mods": list(MODS),
        **summary,
    }
    if a.json and not write_report(a.json, report):
        return 2

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: {err['kind']} wave{err['wave']} {err['op']} {err['mod']}: {err['error']}")
        return 2

    print(
        f"total_words={summary['total_words']} (grid={summary['grid_words']}, "
        f"src_mods={summary['src_mod_words']}, controls={summary['control_words']}) "
        f"across {len(waves)} wave(s) x {len(ops)} op(s): "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}, omod_diffs={summary['omod_diffs']}"
    )
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH {m['kind']} wave{m['wave']} {m['op']} {m['mod']!r} {m['case']}: got {m['got']}, want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['kind']} {d}")
    for d in summary["first_omod_diffs"]:
        print(f"  OMOD DIFF wave{d['wave']} {d['op']} {d['base']} vs {d['mod']} idx={d['index']}: got {d['got']}, want {d['want']}")

    if summary["mismatches"] or summary["wave_diffs"] or summary["omod_diffs"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
