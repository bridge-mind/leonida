r"""Measure RDNA2 VOP3/SDWA output modifiers on v_cvt_norm_{i16,u16}_f16 and v_cvt_pknorm_{i16,u16}_f32.

Standard library only. Uses `tools/hw_oracle` to run native RDNA2 ISA kernels on AMD GFX10 hardware:
- `v_cvt_norm_i16_f16`, `v_cvt_norm_u16_f16`: exhaustive 65,536-input f16 sweeps across VOP3 `_e64`
  (`plain`, `clamp`, `mul:2`, `mul:4`, `div:2`, `clamp mul:2`, `clamp div:2`) in wave32 and wave64,
  plus `HW_REG_MODE` (`RTZ`, `IEEE=1`), `_e64` high-half preservation / `abs` / `neg`, and `_sdwa` `clamp`
  (`WORD_0`/`WORD_1`, `UNUSED_PRESERVE`/`UNUSED_SEXT`/`UNUSED_PAD`).
- `v_cvt_pknorm_i16_f32`, `v_cvt_pknorm_u16_f32`: 40 edge, NaN, inf, subnormal, out-of-range, and single-rounding
  near-tie pairs across `plain` vs VOP3 `clamp`, 6 `HW_REG_MODE` states, source modifiers, and wave32/wave64.

Usage:
    python -I tools/cvt_norm_mods_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/cvt_norm_mods_probe.py --self-test

Exit codes:
    0   all hardware outputs match the reference model across requested waves (with wave32 == wave64)
    1   at least one word mismatch or wave32/wave64 difference
    2   bad CLI arguments, unwritable --json path, or harness error
    77  hardware oracle unavailable or GPU is not GFX10
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

M16 = 0xFFFF
M32 = 0xFFFFFFFF

NORM_F16_OPS = ("v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16")
PKNORM_F32_OPS = ("v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32")
OPS: tuple[str, ...] = NORM_F16_OPS + PKNORM_F32_OPS

VOP3_NORM_MODS: tuple[str, ...] = (
    "",
    "clamp",
    "mul:2",
    "mul:4",
    "div:2",
    "clamp mul:2",
    "clamp div:2",
)

NORM_MODE_VARIANTS: tuple[tuple[str, int], ...] = (
    ("rtz16", 0x000000CC),  # round16_64=3 (RTZ), denorm16_64=3
    ("ieee1", 0x000002C0),  # IEEE=1, denorm16_64=3
)

PKNORM_MODES: tuple[tuple[str, int], ...] = (
    ("game", 0x000001C0),        # IEEE=0, DX10_CLAMP=1, denorm32=0, denorm16_64=3, RNE
    ("ieee0_dx0", 0x000000C0),   # IEEE=0, DX10_CLAMP=0, denorm32=0, denorm16_64=3, RNE
    ("ieee1_dx0", 0x000002C0),   # IEEE=1, DX10_CLAMP=0, denorm32=0, denorm16_64=3, RNE
    ("ieee1_dx1", 0x000003C0),   # IEEE=1, DX10_CLAMP=1, denorm32=0, denorm16_64=3, RNE
    ("rtz", 0x000001CF),         # FP_ROUND32=RTZ, FP_ROUND16_64=RTZ, DX10_CLAMP=1
    ("den32_keep", 0x000001F0),  # denorm32=3, denorm16_64=3, DX10_CLAMP=1, RNE
)

# 40 f32 bit patterns: +-0, subnormals, near-ties from pknorm_probe.py, scale steps, +-0.25/0.5/1/2, huge, +-inf, NaNs.
PKNORM_F32_VALUES: tuple[int, ...] = (
    0x00000000, 0x80000000,  # +0, -0
    0x00000001, 0x80000001,  # min subnormal
    0x007FFFFF, 0x807FFFFF,  # max subnormal
    0x37800000, 0xB7800000,  # +-2^-16
    0x37800080, 0xB7800080,  # just above 0.5 / 32767
    0x37000040, 0x37000080,  # near 0.5 / 65535
    0x38400180, 0x38E001C0,  # pknorm_probe.py near-ties (i16)
    0x39300160, 0x3C004101,
    0x3C014103, 0x3C024105,
    0x37C000C0, 0x386000E0,  # pknorm_probe.py near-ties (u16)
    0x38B000B0, 0x3B008081,
    0x3B028083, 0x3B048085,
    0x3E800000, 0xBE800000,  # +-0.25
    0x3F000000, 0xBF000000,  # +-0.5 (exact halfway ties)
    0x3F000100, 0xBF000100,  # near +-0.5
    0x3F7FFFFF, 0xBF7FFFFF,  # just below +-1.0
    0x3F800000, 0xBF800000,  # +-1.0
    0x3FC00000, 0xBFC00000,  # +-1.5
    0x40000000, 0xC0000000,  # +-2.0
    0x60000000, 0xE0000000,  # huge finite
    0x7F800000, 0xFF800000,  # +inf, -inf
    0x7FC00000, 0xFFC00000,  # +qNaN, -qNaN
    0x7F800001, 0xFF800001,  # +sNaN, -sNaN
    0x7FC01234, 0xFFC01234,  # payload qNaN
)

PKNORM_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (PKNORM_F32_VALUES[i], PKNORM_F32_VALUES[(i * 7 + 3) % len(PKNORM_F32_VALUES)])
    for i in range(40)
)

# SDWA cases for v_cvt_norm_{i16,u16}_f16. Input register v1 = 0x3800B800 (hi=+0.5, lo=-0.5), v0 seeded with 0xCAFE1234.
SDWA_CASES: tuple[tuple[str, int, int, str, str, str], ...] = (
    # (label, seed_v0, src_v1, mod_prefix, src_expr, tail)
    ("w1_preserve", 0xCAFE1234, 0x3800B800, "", "v1", "dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1"),
    ("w0_preserve", 0xCAFE1234, 0x3800B800, "", "v1", "dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0"),
    ("w0_sext", 0xCAFE1234, 0x3800B800, "", "v1", "dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0"),
    ("w0_pad", 0xCAFE1234, 0x3800B800, "", "v1", "dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0"),
    ("w1_abs", 0xCAFE1234, 0xB8003800, "", "|v1|", "dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1"),
    ("w0_neg", 0xCAFE1234, 0x38003800, "", "-v1", "dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0"),
)


def _f16_to_frac(h: int) -> Fraction | str:
    h &= M16
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 31:
        return "nan" if mant else ("-inf" if sign else "+inf")
    if exp == 0:
        val = Fraction(mant, 1 << 24)
    elif exp <= 25:
        val = Fraction(1024 + mant, 1 << (25 - exp))
    else:
        val = Fraction((1024 + mant) << (exp - 25), 1)
    return -val if sign else val


def expected_norm_f16(h: int, snorm: bool) -> int:
    """Exact rational result for v_cvt_norm_i16_f16 (snorm=True) or v_cvt_norm_u16_f16 (snorm=False)."""
    v = _f16_to_frac(h)
    if v == "nan":
        return 0
    if v == "-inf":
        return (-32767) & M16 if snorm else 0
    if v == "+inf":
        return 32767 if snorm else 65535
    lo = Fraction(-1, 1) if snorm else Fraction(0, 1)
    if v < lo:
        v = lo
    elif v > 1:
        v = Fraction(1, 1)
    return round(v * (32767 if snorm else 65535)) & M16


def _f32_to_frac(u: int) -> Fraction | str:
    u &= M32
    sign = (u >> 31) & 1
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    if exp == 255:
        return "nan" if mant else ("-inf" if sign else "+inf")
    if exp == 0:
        val = Fraction(mant, 1 << 149)
    elif exp <= 150:
        val = Fraction((1 << 23) + mant, 1 << (150 - exp))
    else:
        val = Fraction(((1 << 23) + mant) << (exp - 150), 1)
    return -val if sign else val


def expected_pknorm_f32_lane(u: int, snorm: bool, *, flush_denorm: bool = True) -> int:
    """Exact single-rounded 16-bit result for one lane of v_cvt_pknorm_{i16,u16}_f32."""
    u &= M32
    if flush_denorm and ((u >> 23) & 0xFF) == 0:
        return 0
    v = _f32_to_frac(u)
    if v == "nan":
        return 0
    if v == "-inf":
        return (-32767) & M16 if snorm else 0
    if v == "+inf":
        return 32767 if snorm else 65535
    lo = Fraction(-1, 1) if snorm else Fraction(0, 1)
    if v < lo:
        v = lo
    elif v > 1:
        v = Fraction(1, 1)
    return round(v * (32767 if snorm else 65535)) & M16


def expected_pknorm_f32(op: str, s0: int, s1: int, *, flush_denorm: bool = True) -> int:
    """Expected packed 32-bit dword for v_cvt_pknorm_i16_f32 or v_cvt_pknorm_u16_f32."""
    if op not in PKNORM_F32_OPS:
        raise ValueError(f"unknown pknorm op: {op}")
    snorm = op == "v_cvt_pknorm_i16_f32"
    lo = expected_pknorm_f32_lane(s0, snorm, flush_denorm=flush_denorm)
    hi = expected_pknorm_f32_lane(s1, snorm, flush_denorm=flush_denorm)
    return lo | (hi << 16)


def _apply_f16_src_mod(h: int, src_expr: str) -> int:
    h &= M16
    if "|" in src_expr:
        h &= 0x7FFF
    if src_expr.startswith("-"):
        h ^= 0x8000
    return h


def expected_sdwa_words(op: str) -> list[int]:
    """Expected 32-bit words for SDWA_CASES on op (plain or clamp)."""
    snorm = op == "v_cvt_norm_i16_f16"
    out: list[int] = []
    for _, seed_v0, src_v1, _, src_expr, tail in SDWA_CASES:
        src_hi = "src0_sel:WORD_1" in tail
        h = (src_v1 >> 16) & M16 if src_hi else (src_v1 & M16)
        h = _apply_f16_src_mod(h, src_expr)
        val16 = expected_norm_f16(h, snorm)
        if "dst_sel:WORD_1" in tail:
            word = (val16 << 16) | (seed_v0 & M16)
        elif "UNUSED_PRESERVE" in tail:
            word = (seed_v0 & 0xFFFF0000) | val16
        elif "UNUSED_SEXT" in tail:
            word = (0xFFFF0000 | val16) if (val16 & 0x8000) else val16
        else:
            word = val16
        out.append(word & M32)
    return out


def expected_sweep_words(op: str) -> list[int]:
    """All 32,768 packed dwords (65,536 f16 inputs) for v_cvt_norm_{i16,u16}_f16."""
    snorm = op == "v_cvt_norm_i16_f16"
    return [
        expected_norm_f16(2 * wi, snorm) | (expected_norm_f16(2 * wi + 1, snorm) << 16)
        for wi in range(32768)
    ]


def sweep_kernel_lines(op: str, mod_str: str, base_word: int, label: str, mode_val: int | None = None) -> list[str]:
    """GFX10 kernel evaluating 32,768 sequential f16 inputs as 16,384 packed dwords."""
    mod_suf = (" " + mod_str) if mod_str else ""
    lines: list[str] = []
    if mode_val is not None:
        lines.append(f"s_setreg_imm32_b32 hwreg(HW_REG_MODE), {mode_val:#010x}")
    lines += [
        "v_mov_b32_e32 v2, 0",
        "v_mov_b32_e32 v3, 0",
        "v_lshlrev_b32_e32 v10, 4, v0",
        f"v_add_nc_u32_e32 v11, {base_word}, v10",
        "v_lshlrev_b32_e32 v12, 1, v11",
        "v_lshlrev_b32_e32 v13, 2, v10",
        "s_mov_b32 s4, 16",
        f"{label}:",
        f"{op}_e64 v2, v12{mod_suf}",
        "v_add_nc_u32_e32 v14, 1, v12",
        f"{op}_e64 v3, v14{mod_suf}",
        "v_lshl_or_b32 v4, v3, 16, v2",
        "global_store_dword v13, v4, s[2:3]",
        "v_add_nc_u32_e32 v12, 2, v12",
        "v_add_nc_u32_e32 v13, 4, v13",
        "s_sub_u32 s4, s4, 1",
        "s_cmp_lg_u32 s4, 0",
        f"s_cbranch_scc1 {label}",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    return lines


def sdwa_kernel_lines(op: str, clamp: bool) -> list[str]:
    """GFX10 kernel testing SDWA plain vs clamp, dst_unused forms, and VOP3 preservation."""
    c_mod = " clamp" if clamp else ""
    lines: list[str] = []
    for idx, (_, seed_v0, src_v1, _, src_expr, tail) in enumerate(SDWA_CASES):
        lines += [
            f"v_mov_b32_e32 v0, {seed_v0:#010x}",
            f"v_mov_b32_e32 v1, {src_v1:#010x}",
            f"{op}_sdwa v0, {src_expr}{c_mod} {tail}",
            f"v_mov_b32_e32 v12, {idx * 4}",
            "global_store_dword v12, v0, s[2:3]",
        ]
    # Also verify VOP3 _e64 preserves the high half of v0 under clamp and omod.
    vop3_mods = ("clamp", "mul:2", "mul:4", "div:2") if clamp else ("",)
    base_idx = len(SDWA_CASES)
    for k, mod in enumerate(vop3_mods):
        suf = (" " + mod) if mod else ""
        lines += [
            "v_mov_b32_e32 v0, 0xCAFE1234",
            "v_mov_b32_e32 v1, 0xB8003800",  # lo=+0.5, hi=-0.5
            f"{op}_e64 v0, -|v1|{suf}",
            f"v_mov_b32_e32 v12, {(base_idx + k) * 4}",
            "global_store_dword v12, v0, s[2:3]",
        ]
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines


def expected_sdwa_and_pres_words(op: str, clamp: bool) -> list[int]:
    words = expected_sdwa_words(op)
    snorm = op == "v_cvt_norm_i16_f16"
    # -|0x3800| = -0.5 -> 0xB800
    val16 = expected_norm_f16(0xB800, snorm)
    pres_word = 0xCAFE0000 | val16
    count = 4 if clamp else 1
    return words + [pres_word] * count


def pknorm_kernel_lines(op: str, clamp: bool, mode_val: int) -> list[str]:
    """GFX10 kernel running PKNORM_PAIRS + source modifier checks under one HW_REG_MODE."""
    suf = " clamp" if clamp else ""
    lines: list[str] = [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE), {mode_val:#010x}"]
    for idx, (a, b) in enumerate(PKNORM_PAIRS):
        lines += [
            f"v_mov_b32_e32 v1, {a:#010x}",
            f"v_mov_b32_e32 v2, {b:#010x}",
            f"{op} v0, v1, v2{suf}",
            f"v_mov_b32_e32 v12, {idx * 4}",
            "global_store_dword v12, v0, s[2:3]",
        ]
    # Also check VOP3 source modifiers (-v1, |v2|, -|v1|) with clamp.
    mod_cases = (
        (0x3F000000, 0xBF000000, "-v1, |v2|"),
        (0xBF000000, 0x3F000000, "|v1|, -v2"),
        (0x3F800000, 0x3F000000, "-|v1|, -|v2|"),
    )
    base_idx = len(PKNORM_PAIRS)
    for k, (a, b, expr) in enumerate(mod_cases):
        lines += [
            f"v_mov_b32_e32 v1, {a:#010x}",
            f"v_mov_b32_e32 v2, {b:#010x}",
            f"{op} v0, {expr}{suf}",
            f"v_mov_b32_e32 v12, {(base_idx + k) * 4}",
            "global_store_dword v12, v0, s[2:3]",
        ]
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines


def expected_pknorm_words(op: str) -> list[int]:
    words = [expected_pknorm_f32(op, a, b) for a, b in PKNORM_PAIRS]
    # (-0.5, |-0.5|), (|-0.5|, -0.5), (-|1.0|, -|0.5|)
    mod_inputs = (
        (0xBF000000, 0x3F000000),
        (0x3F000000, 0xBF000000),
        (0xBF800000, 0xBF000000),
    )
    words.extend(expected_pknorm_f32(op, a, b) for a, b in mod_inputs)
    return words


def build_jobs(ops: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels = []
    meta = []
    label_id = 0

    for op in ops:
        if op in NORM_F16_OPS:
            exp_full = expected_sweep_words(op)
            for wave in waves:
                for mod_str in VOP3_NORM_MODS:
                    mod_key = mod_str or "plain"
                    for part, base_word in enumerate((0, 16384)):
                        lbl = f".L_cvt_loop_{label_id}"
                        label_id += 1
                        lines = sweep_kernel_lines(op, mod_str, base_word, lbl)
                        kernels.append(asm.Kernel(lines, wave=wave, threads=1024, words=16384, raw=True))
                        exp_slice = exp_full[base_word : base_word + 16384]
                        meta.append(("sweep", op, wave, f"{mod_key}_p{part}", bool(mod_str), exp_slice))
                for clamp in (False, True):
                    lines = sdwa_kernel_lines(op, clamp)
                    exp = expected_sdwa_and_pres_words(op, clamp)
                    kernels.append(asm.Kernel(lines, wave=wave, threads=1, words=len(exp), raw=True))
                    meta.append(("sdwa", op, wave, "clamp" if clamp else "plain", clamp, exp))
            # MODE tests (RTZ and IEEE=1) on the first requested wave.
            mode_wave = waves[0]
            for mode_name, mode_val in NORM_MODE_VARIANTS:
                for part, base_word in enumerate((0, 16384)):
                    lbl = f".L_cvt_loop_{label_id}"
                    label_id += 1
                    lines = sweep_kernel_lines(op, "", base_word, lbl, mode_val=mode_val)
                    kernels.append(asm.Kernel(lines, wave=mode_wave, threads=1024, words=16384, raw=True))
                    exp_slice = exp_full[base_word : base_word + 16384]
                    meta.append(("mode", op, mode_wave, f"{mode_name}_p{part}", False, exp_slice))
        elif op in PKNORM_F32_OPS:
            exp_pk = expected_pknorm_words(op)
            for wave in waves:
                for mode_name, mode_val in PKNORM_MODES:
                    for clamp in (False, True):
                        lines = pknorm_kernel_lines(op, clamp, mode_val)
                        kernels.append(asm.Kernel(lines, wave=wave, threads=1, words=len(exp_pk), raw=True))
                        meta.append(("pknorm", op, wave, mode_name, clamp, exp_pk))
        else:
            raise ValueError(f"unknown op: {op}")

    return kernels, meta


def evaluate_results(waves: list[int], meta, raw_results) -> dict:
    """Compare hardware outputs against expected words and check wave32 == wave64."""
    by_key: dict[tuple[str, str, int, str, bool], list[int]] = {}
    harness_errors: list[dict] = []
    total_words = 0

    for (kind, op, wave, variant, clamp, expected), res in zip(meta, raw_results):
        total_words += len(expected)
        if isinstance(res, str):
            harness_errors.append({"kind": kind, "op": op, "wave": wave, "variant": variant, "error": res})
            continue
        if len(res) != len(expected):
            harness_errors.append({
                "kind": kind,
                "op": op,
                "wave": wave,
                "variant": variant,
                "error": f"expected {len(expected)} words, got {len(res)}",
            })
            continue
        by_key[(kind, op, wave, variant, clamp)] = [w & M32 for w in res]

    if harness_errors:
        return {
            "total_words": total_words,
            "mismatches": 0,
            "wave_diffs": 0,
            "harness_errors": harness_errors,
            "mismatch_details": [],
            "wave_diff_details": [],
        }

    mismatches = 0
    mismatch_details: list[dict] = []
    for (kind, op, wave, variant, clamp, expected), res in zip(meta, raw_results):
        words = [w & M32 for w in res]
        for idx, (got_w, exp_w) in enumerate(zip(words, expected)):
            if got_w != exp_w:
                mismatches += 1
                if len(mismatch_details) < 32:
                    mismatch_details.append({
                        "kind": kind,
                        "op": op,
                        "wave": wave,
                        "variant": variant,
                        "clamp": clamp,
                        "word_idx": idx,
                        "expected": f"0x{exp_w:08x}",
                        "actual": f"0x{got_w:08x}",
                    })

    wave_diffs = 0
    wave_diff_details: list[dict] = []
    if 32 in waves and 64 in waves:
        seen_pairs: set[tuple[str, str, str, bool]] = set()
        for kind, op, wave, variant, clamp, _ in meta:
            if kind == "mode":
                continue
            pkey = (kind, op, variant, clamp)
            if pkey in seen_pairs:
                continue
            seen_pairs.add(pkey)
            k32 = (kind, op, 32, variant, clamp)
            k64 = (kind, op, 64, variant, clamp)
            if k32 not in by_key or k64 not in by_key:
                continue
            for idx, (w32, w64) in enumerate(zip(by_key[k32], by_key[k64])):
                if w32 != w64:
                    wave_diffs += 1
                    if len(wave_diff_details) < 32:
                        wave_diff_details.append({
                            "kind": kind,
                            "op": op,
                            "variant": variant,
                            "clamp": clamp,
                            "word_idx": idx,
                            "wave32": f"0x{w32:08x}",
                            "wave64": f"0x{w64:08x}",
                        })

    return {
        "total_words": total_words,
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "harness_errors": [],
        "mismatch_details": mismatch_details,
        "wave_diff_details": wave_diff_details,
    }


def self_test() -> int:
    """Verify reference model, kernel construction, and evaluation logic without a GPU."""
    # 1. Exact v_cvt_norm_{i16,u16}_f16 edges & single-rounding ties near +-0.5.
    assert expected_norm_f16(0x0000, True) == 0
    assert expected_norm_f16(0x8000, True) == 0
    assert expected_norm_f16(0x3C00, True) == 0x7FFF
    assert expected_norm_f16(0xBC00, True) == 0x8001
    assert expected_norm_f16(0xFC00, True) == 0x8001
    assert expected_norm_f16(0x7E00, True) == 0
    assert expected_norm_f16(0x7C01, False) == 0
    assert expected_norm_f16(0x3800, True) == 16384       # 0.5 * 32767 = 16383.5 -> 16384 (even)
    assert expected_norm_f16(0x3801, True) == 16399       # single rounding near +0.5 (double-rounding gives 16400)
    assert expected_norm_f16(0x37FF, True) == 16376       # single rounding near +0.5 (double-rounding gives 16375)
    assert expected_norm_f16(0x3800, False) == 32768      # 0.5 * 65535 = 32767.5 -> 32768 (even)
    assert expected_norm_f16(0x3801, False) == 32799
    assert expected_norm_f16(0x37FF, False) == 32752

    # 2. Exact v_cvt_pknorm_{i16,u16}_f32 near-ties & saturation.
    assert expected_pknorm_f32("v_cvt_pknorm_i16_f32", 0x38400180, 0x00000000) == 0x00000001
    assert expected_pknorm_f32("v_cvt_pknorm_i16_f32", 0x3C004101, 0xBF800000) == 0x80010101
    assert expected_pknorm_f32("v_cvt_pknorm_u16_f32", 0x37C000C0, 0x40000000) == 0xFFFF0001
    assert expected_pknorm_f32("v_cvt_pknorm_u16_f32", 0x7FC00000, 0xFF800000) == 0x00000000

    # 3. Exhaustive sweep builders produce 32,768 packed dwords, valid labels, and proper endpgm.
    for op in NORM_F16_OPS:
        assert len(expected_sweep_words(op)) == 32768
        l0 = sweep_kernel_lines(op, "clamp mul:2", 0, ".L_test_0")
        l1 = sdwa_kernel_lines(op, True)
        assert l0[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
        assert l1[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
        assert len(expected_sdwa_and_pres_words(op, True)) == len(SDWA_CASES) + 4
    for op in PKNORM_F32_OPS:
        lp = pknorm_kernel_lines(op, True, 0x000001C0)
        assert lp[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
        assert len(expected_pknorm_words(op)) == len(PKNORM_PAIRS) + 3

    # 4. Synthetic evaluation checks pass, mismatch, and wave diff detection.
    meta = [
        ("sdwa", "v_cvt_norm_i16_f16", 32, "clamp", True, expected_sdwa_and_pres_words("v_cvt_norm_i16_f16", True)),
        ("sdwa", "v_cvt_norm_i16_f16", 64, "clamp", True, expected_sdwa_and_pres_words("v_cvt_norm_i16_f16", True)),
        ("pknorm", "v_cvt_pknorm_u16_f32", 32, "game", True, expected_pknorm_words("v_cvt_pknorm_u16_f32")),
        ("pknorm", "v_cvt_pknorm_u16_f32", 64, "game", True, expected_pknorm_words("v_cvt_pknorm_u16_f32")),
    ]
    raw_ok = [list(m[5]) for m in meta]
    rep_ok = evaluate_results([32, 64], meta, raw_ok)
    assert rep_ok["mismatches"] == 0 and rep_ok["wave_diffs"] == 0 and not rep_ok["harness_errors"], rep_ok

    raw_bad = [list(w) for w in raw_ok]
    raw_bad[0][0] ^= 1
    rep_bad = evaluate_results([32, 64], meta, raw_bad)
    assert rep_bad["mismatches"] == 1 and rep_bad["wave_diffs"] == 1, rep_bad

    print(
        f"self-test ok ({len(OPS)} ops, {len(VOP3_NORM_MODS)} VOP3 norm mods, "
        f"{len(SDWA_CASES)} SDWA cases, {len(PKNORM_PAIRS)} pknorm pairs)"
    )
    return 0


def _parse_waves(spec: str) -> list[int] | None:
    parts = [p.strip() for p in spec.split(",")]
    if not parts or any(not p for p in parts):
        return None
    try:
        waves = [int(p) for p in parts]
    except ValueError:
        return None
    if any(w not in (32, 64) for w in waves):
        return None
    return list(dict.fromkeys(waves))


def _parse_ops(spec: str) -> list[str] | None:
    parts = [p.strip() for p in spec.split(",")]
    if not parts or any(not p for p in parts):
        return None
    if any(p not in OPS for p in parts):
        return None
    return list(dict.fromkeys(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="comma-separated wave sizes (default: 32,64)")
    ap.add_argument("--ops", default=",".join(OPS), help="comma-separated conversion opcodes to probe")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    ap.add_argument("--json", dest="json_out", default=None, help="write JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="verify reference model and kernel generator without GPU")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    waves = _parse_waves(args.waves)
    if waves is None:
        print(f"ERROR: --waves must list 32 and/or 64, got {args.waves!r}", file=sys.stderr)
        return 2

    ops = _parse_ops(args.ops)
    if ops is None:
        print(f"ERROR: unknown or malformed --ops {args.ops!r} (valid: {', '.join(OPS)})", file=sys.stderr)
        return 2

    if args.json_out is not None:
        out_check = os.path.abspath(args.json_out)
        if os.path.isdir(out_check):
            print(f"ERROR: --json path is a directory: {out_check}", file=sys.stderr)
            return 2

    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle(arch=args.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    try:
        try:
            fam = asm.family(oracle.arch)
        except ValueError:
            fam = None
        if fam != "gfx10":
            print(f"skipping: requires GFX10 RDNA2 hardware, found {oracle.arch}")
            return 77

        kernels, meta = build_jobs(ops, waves)
        raw_results = oracle.run(kernels, words=16384)
        summary = evaluate_results(waves, meta, raw_results)
        device = oracle.device
        arch = oracle.arch
    finally:
        oracle.close()

    total_words = summary["total_words"]
    print(f"cvt_norm_mods_probe on {device} ({arch}):")
    if summary["harness_errors"]:
        for err in summary["harness_errors"][:8]:
            print(f"  harness error ({err['kind']} wave{err['wave']} {err['op']} {err['variant']}): {err['error']}")
        return 2

    print(
        f"  {total_words} word evaluations across {len(waves)} wave(s) x {len(ops)} op(s): "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}"
    )
    for d in summary["mismatch_details"]:
        print(
            f"  MISMATCH {d['kind']} wave{d['wave']} {d['op']} {d['variant']} clamp={d['clamp']} "
            f"w[{d['word_idx']}]: expected={d['expected']} actual={d['actual']}"
        )
    for d in summary["wave_diff_details"]:
        print(
            f"  WAVE DIFF {d['kind']} {d['op']} {d['variant']} clamp={d['clamp']} "
            f"w[{d['word_idx']}]: wave32={d['wave32']} wave64={d['wave64']}"
        )

    if args.json_out is not None:
        out_path = os.path.abspath(args.json_out)
        if os.path.isdir(out_path):
            print(f"ERROR: cannot write {out_path}: path is a directory", file=sys.stderr)
            return 2
        payload = {
            "device": device,
            "arch": arch,
            "waves": waves,
            "ops": ops,
            **summary,
        }
        try:
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {out_path}: {e}", file=sys.stderr)
            return 2

    return 1 if (summary["mismatches"] or summary["wave_diffs"]) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
