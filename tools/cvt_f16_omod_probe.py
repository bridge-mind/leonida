r"""Measure RDNA2 v_cvt_f16_f32 VOP3 output modifiers (mul:2, mul:4, div:2, clamp) across MODE settings. Standard library only.

docs/design.md records the RX 6800 (RNE) order for v_cvt_f16_f32 output modifiers — convert to f16, apply omod when
IEEE=0 and f16 denormals are flushed, then apply clamp — and marks finite overflow under MODE toward-zero (FP_ROUND=3)
as unmeasured. Running this probe natively on RDNA2 (gfx1036) via tools/hw_oracle/ across wave32 and wave64 confirms
RTZ overflow saturation (+-65504 for mul:2/mul:4, +-65520 div:2 -> +-32752) and catches an edge-case bug: when div:2
underflows a negative smallest normal (exponent 1, -2^-14 .. -(2 - 2^-10) * 2^-14), hardware produces -0.0 (0x8000)
rather than +0.0 (0x0000), while pre-omod zero/denormals (exponent 0, including -0.0 and -2^-15) give +0.0.

Usage:
    python -I tools/cvt_f16_omod_probe.py [--waves 32,64] [--modes ...] [--json out.json]
    python -I tools/cvt_f16_omod_probe.py --self-test

Exit 0 when all hardware results match the model across requested modes (with wave32 and wave64 agreeing bit-for-bit),
1 on a model mismatch or wave32/wave64 difference, 2 on a harness/CLI error, and 77 when the hardware oracle is
unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

HI_SEED = 0xAAAA
LO_SEED = 0x5555
SEED_WORD = (HI_SEED << 16) | LO_SEED


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


INPUTS: tuple[tuple[str, int], ...] = (
    ("+0.0", 0x00000000),
    ("-0.0", 0x80000000),
    ("+2^-24", _f32_bits(2.0 ** -24)),
    ("-2^-24", _f32_bits(-(2.0 ** -24))),
    ("+2^-15", _f32_bits(2.0 ** -15)),
    ("-2^-15", _f32_bits(-(2.0 ** -15))),
    ("+2^-14", _f32_bits(2.0 ** -14)),
    ("-2^-14", _f32_bits(-(2.0 ** -14))),
    ("+1.5*2^-14", _f32_bits(1.5 * (2.0 ** -14))),
    ("-1.5*2^-14", _f32_bits(-1.5 * (2.0 ** -14))),
    ("-max_exp1", 0xB8FFE000),             # -(2 - 2^-10) * 2^-14 -> f16 0x87FF in both RNE and RTZ
    ("-rne_to_exp2", 0xB8FFFE00),          # f16 0x8800 (-2^-13) in RNE, 0x87FF in RTZ
    ("-2^-13", _f32_bits(-(2.0 ** -13))),
    ("0.375", _f32_bits(0.375)),
    ("0.75", _f32_bits(0.75)),
    ("-0.5", _f32_bits(-0.5)),
    ("1.5", _f32_bits(1.5)),
    ("2.0", _f32_bits(2.0)),
    ("+16384.0", _f32_bits(16384.0)),
    ("-16384.0", _f32_bits(-16384.0)),
    ("+32752.0", _f32_bits(32752.0)),
    ("-32752.0", _f32_bits(-32752.0)),
    ("+32768.0", _f32_bits(32768.0)),
    ("-32768.0", _f32_bits(-32768.0)),
    ("+65504.0", _f32_bits(65504.0)),
    ("-65504.0", _f32_bits(-65504.0)),
    ("+65520.0", _f32_bits(65520.0)),
    ("-65520.0", _f32_bits(-65520.0)),
    ("+FLT_MAX", 0x7F7FFFFF),
    ("-FLT_MAX", 0xFF7FFFFF),
    ("+inf", 0x7F800000),
    ("-inf", 0xFF800000),
    ("+qNaN", 0x7FC01234),
    ("-qNaN", 0xFFC01234),
    ("+sNaN", 0x7F801234),
    ("-sNaN", 0xFF801234),
)

# In LLVM AMDGPU syntax for v_cvt_f16_f32_e64, clamp precedes mul:2/mul:4/div:2.
MODS: tuple[str, ...] = (
    "",
    "mul:2",
    "mul:4",
    "div:2",
    "clamp",
    "clamp mul:2",
    "clamp mul:4",
    "clamp div:2",
)

# name -> (kernel-descriptor MODE dict, runtime s_setreg RTZ flag, model parameters)
MODES: dict[str, dict] = {
    "rne_d0_i0": {
        "descriptor": {"round16_64": 0, "denorm16_64": 0, "dx10_clamp": 1, "ieee": 0},
        "setreg_rtz": False,
        "rtz": False,
        "denorm_keep": False,
        "ieee": False,
        "dx10_clamp": True,
    },
    "rtz_d0_i0": {
        "descriptor": {"round16_64": 3, "denorm16_64": 0, "dx10_clamp": 1, "ieee": 0},
        "setreg_rtz": False,
        "rtz": True,
        "denorm_keep": False,
        "ieee": False,
        "dx10_clamp": True,
    },
    "rtz_d3_i0": {
        "descriptor": {"round16_64": 3, "denorm16_64": 3, "dx10_clamp": 1, "ieee": 0},
        "setreg_rtz": False,
        "rtz": True,
        "denorm_keep": True,
        "ieee": False,
        "dx10_clamp": True,
    },
    "rtz_d0_i1": {
        "descriptor": {"round16_64": 3, "denorm16_64": 0, "dx10_clamp": 1, "ieee": 1},
        "setreg_rtz": False,
        "rtz": True,
        "denorm_keep": False,
        "ieee": True,
        "dx10_clamp": True,
    },
    "dyn_rtz_d0_i0": {
        "descriptor": {"round16_64": 0, "denorm16_64": 0, "dx10_clamp": 1, "ieee": 0},
        "setreg_rtz": True,
        "rtz": True,
        "denorm_keep": False,
        "ieee": False,
        "dx10_clamp": True,
    },
}


def f32_to_f16_bits(f32_bits: int, *, rtz: bool, denorm_keep: bool) -> int:
    """Exact binary32 -> binary16 conversion with f32 denormals flushed to signed zero."""
    sg16 = ((f32_bits >> 31) & 1) << 15
    exp = (f32_bits >> 23) & 0xFF
    frac = f32_bits & 0x7FFFFF

    if exp == 0xFF:
        if frac:
            return sg16 | 0x7E00 | ((frac >> 13) & 0x3FF)
        return sg16 | 0x7C00
    if exp == 0:
        return sg16

    e = exp - 127
    if e > 15:
        return sg16 | (0x7BFF if rtz else 0x7C00)

    m = 0x800000 | frac
    if e >= -14:
        q = m >> 13
        rem = m & 0x1FFF
        if not rtz and (rem > 0x1000 or (rem == 0x1000 and (q & 1))):
            q += 1
        if q == 0x800:
            q = 0x400
            e += 1
        if e > 15:
            return sg16 | (0x7BFF if rtz else 0x7C00)
        return sg16 | ((e + 15) << 10) | (q & 0x3FF)

    if not denorm_keep:
        return sg16

    sh = -1 - e
    if sh > 25:
        return sg16
    q = m >> sh
    rem = m & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    if not rtz and (rem > half or (rem == half and (q & 1))):
        q += 1
    return sg16 | q


def _clamp_f16(h: int, *, dx10_clamp: bool) -> int:
    e = (h >> 10) & 31
    if e == 31 and (h & 0x3FF) != 0:
        return 0x0000 if dx10_clamp else h
    if h & 0x8000:
        return 0x0000
    if h > 0x3C00:
        return 0x3C00
    return h


def expected_cvt_f16_omod(
    f32_bits: int,
    mod: str,
    *,
    rtz: bool,
    denorm_keep: bool,
    ieee: bool,
    dx10_clamp: bool = True,
    hi_seed: int = HI_SEED,
) -> int:
    """Expected 32-bit VGPR word after `v_cvt_f16_f32_e64 vD, src [clamp] [mul:2|mul:4|div:2]`."""
    tokens = mod.split()
    omod = next((tok for tok in tokens if tok in ("mul:2", "mul:4", "div:2")), None)
    has_clamp = "clamp" in tokens

    h = f32_to_f16_bits(f32_bits, rtz=rtz, denorm_keep=denorm_keep)

    if omod is not None and not denorm_keep and not ieee:
        e = (h >> 10) & 31
        sign = h & 0x8000
        if e == 31:
            pass
        elif e == 0:
            h = 0x0000
        elif omod == "div:2":
            if e == 1:
                h = sign
            else:
                h = (h - (1 << 10)) & 0xFFFF
        else:
            shift = 1 if omod == "mul:2" else 2
            if e >= 31 - shift:
                h = sign | (0x7BFF if rtz else 0x7C00)
            else:
                h = (h + (shift << 10)) & 0xFFFF

    if has_clamp:
        h = _clamp_f16(h, dx10_clamp=dx10_clamp)

    return ((hi_seed & 0xFFFF) << 16) | (h & 0xFFFF)


def kernel_lines(mode_name: str, mod: str) -> list[str]:
    """Raw GFX10 kernel lines running every input in INPUTS sequentially on lane 0."""
    cfg = MODES[mode_name]
    lines: list[str] = []
    if cfg["setreg_rtz"]:
        lines.append("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3")
        lines.append("s_nop 2")
    suffix = f" {mod}" if mod else ""
    for idx, (_, bits) in enumerate(INPUTS):
        lines.append(f"v_mov_b32_e32 v1, {bits:#010x}")
        lines.append(f"v_mov_b32_e32 v2, {SEED_WORD:#010x}")
        lines.append(f"v_cvt_f16_f32_e64 v2, v1{suffix}")
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v2, s[2:3]")
    lines.append("s_waitcnt vmcnt(0)")
    lines.append("s_endpgm")
    return lines


def build_jobs(modes: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels, meta = [], []
    for wave in waves:
        for mode_name in modes:
            mdict = dict(MODES[mode_name]["descriptor"])
            for mod in MODS:
                lines = kernel_lines(mode_name, mod)
                kernels.append(
                    asm.Kernel(lines, wave=wave, threads=1, words=len(INPUTS), raw=True, mode=mdict)
                )
                meta.append((wave, mode_name, mod))
    return kernels, meta


def evaluate_results(modes: list[str], waves: list[int], meta, raw_results) -> dict:
    """Compare hardware outputs against expected_cvt_f16_omod and check wave32 vs wave64 parity."""
    by_key: dict[tuple[int, str, str], list[int]] = {}
    harness_errors: list[dict] = []
    for (wave, mode_name, mod), res in zip(meta, raw_results):
        if isinstance(res, str):
            harness_errors.append({"wave": wave, "mode": mode_name, "mod": mod, "error": res})
            continue
        if len(res) != len(INPUTS):
            harness_errors.append({
                "wave": wave,
                "mode": mode_name,
                "mod": mod,
                "error": f"expected {len(INPUTS)} words, got {len(res)}",
            })
            continue
        by_key[(wave, mode_name, mod)] = [w & 0xFFFFFFFF for w in res]

    if harness_errors:
        return {
            "total_cases": len(meta) * len(INPUTS),
            "lo_mismatches": 0,
            "hi_mismatches": 0,
            "wave_diffs": 0,
            "dyn_rtz_diffs": 0,
            "harness_errors": harness_errors,
            "first_mismatches": [],
            "first_wave_diffs": [],
        }

    lo_mismatches = 0
    hi_mismatches = 0
    first_mismatches: list[dict] = []
    for wave in waves:
        for mode_name in modes:
            cfg = MODES[mode_name]
            for mod in MODS:
                got_words = by_key[(wave, mode_name, mod)]
                for (iname, bits), got in zip(INPUTS, got_words):
                    want = expected_cvt_f16_omod(
                        bits,
                        mod,
                        rtz=cfg["rtz"],
                        denorm_keep=cfg["denorm_keep"],
                        ieee=cfg["ieee"],
                        dx10_clamp=cfg["dx10_clamp"],
                    )
                    if (got >> 16) != HI_SEED:
                        hi_mismatches += 1
                    if (got & 0xFFFF) != (want & 0xFFFF):
                        lo_mismatches += 1
                    if got != want and len(first_mismatches) < 16:
                        first_mismatches.append({
                            "wave": wave,
                            "mode": mode_name,
                            "mod": mod,
                            "input": iname,
                            "f32": f"0x{bits:08x}",
                            "got": f"0x{got:08x}",
                            "want": f"0x{want:08x}",
                        })

    wave_diffs = 0
    first_wave_diffs: list[dict] = []
    if len(waves) > 1:
        w0 = waves[0]
        for w in waves[1:]:
            for mode_name in modes:
                for mod in MODS:
                    for (iname, _), a_w, b_w in zip(INPUTS, by_key[(w0, mode_name, mod)], by_key[(w, mode_name, mod)]):
                        if a_w != b_w:
                            wave_diffs += 1
                            if len(first_wave_diffs) < 16:
                                first_wave_diffs.append({
                                    "mode": mode_name,
                                    "mod": mod,
                                    "input": iname,
                                    f"w{w0}": f"0x{a_w:08x}",
                                    f"w{w}": f"0x{b_w:08x}",
                                })

    dyn_rtz_diffs = 0
    if "rtz_d0_i0" in modes and "dyn_rtz_d0_i0" in modes:
        for wave in waves:
            for mod in MODS:
                for a_w, b_w in zip(by_key[(wave, "rtz_d0_i0", mod)], by_key[(wave, "dyn_rtz_d0_i0", mod)]):
                    if a_w != b_w:
                        dyn_rtz_diffs += 1

    return {
        "total_cases": len(waves) * len(modes) * len(MODS) * len(INPUTS),
        "lo_mismatches": lo_mismatches,
        "hi_mismatches": hi_mismatches,
        "wave_diffs": wave_diffs,
        "dyn_rtz_diffs": dyn_rtz_diffs,
        "harness_errors": [],
        "first_mismatches": first_mismatches,
        "first_wave_diffs": first_wave_diffs,
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
    """Verify the reference model and kernel generator against hand-checked values without GPU or clang."""
    # RNE vs RTZ conversion and omod overflow
    assert expected_cvt_f16_omod(_f32_bits(32768.0), "mul:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA7C00
    assert expected_cvt_f16_omod(_f32_bits(32768.0), "mul:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA7BFF
    assert expected_cvt_f16_omod(_f32_bits(-32768.0), "mul:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAAFBFF
    assert expected_cvt_f16_omod(_f32_bits(16384.0), "mul:4", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA7BFF
    assert expected_cvt_f16_omod(_f32_bits(65520.0), "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA7C00
    assert expected_cvt_f16_omod(_f32_bits(65520.0), "div:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA77FF
    assert expected_cvt_f16_omod(_f32_bits(-65520.0), "div:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAAF7FF
    assert expected_cvt_f16_omod(0x7F800000, "mul:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA7C00
    # div:2 underflow from exponent 1 preserves sign; exponent 0 flushes to +0
    assert expected_cvt_f16_omod(_f32_bits(-(2.0 ** -14)), "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA8000
    assert expected_cvt_f16_omod(_f32_bits(-1.5 * (2.0 ** -14)), "div:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA8000
    assert expected_cvt_f16_omod(0xB8FFE000, "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA8000
    assert expected_cvt_f16_omod(0xB8FFFE00, "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA8400
    assert expected_cvt_f16_omod(0xB8FFFE00, "div:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA8000
    assert expected_cvt_f16_omod(0x80000000, "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA0000
    assert expected_cvt_f16_omod(_f32_bits(-(2.0 ** -15)), "div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA0000
    assert expected_cvt_f16_omod(_f32_bits(-(2.0 ** -14)), "clamp div:2", rtz=False, denorm_keep=False, ieee=False) == 0xAAAA0000
    # denorm_keep=True or ieee=True ignores omod while keeping clamp
    assert expected_cvt_f16_omod(_f32_bits(0.75), "mul:2", rtz=True, denorm_keep=True, ieee=False) == 0xAAAA3A00
    assert expected_cvt_f16_omod(_f32_bits(0.75), "clamp mul:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA3C00
    assert expected_cvt_f16_omod(_f32_bits(0.75), "clamp mul:2", rtz=True, denorm_keep=False, ieee=True) == 0xAAAA3A00
    assert expected_cvt_f16_omod(0x7FC01234, "clamp mul:2", rtz=True, denorm_keep=False, ieee=False) == 0xAAAA0000
    # Kernel assembly sanity
    kl = kernel_lines("dyn_rtz_d0_i0", "clamp mul:2")
    assert kl[0] == "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
    assert kl[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
    assert any("v_cvt_f16_f32_e64 v2, v1 clamp mul:2" == ln for ln in kl)
    print(f"self-test ok ({len(INPUTS)} inputs, {len(MODES) * len(MODS) * len(INPUTS)} cases/wave)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--json", help="write measurement summary JSON here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    wave_tokens = [w.strip() for w in a.waves.split(",") if w.strip()]
    bad_modes = [m for m in modes if m not in MODES]
    bad_waves = [w for w in wave_tokens if w not in ("32", "64")]
    if bad_modes or bad_waves or not modes or not wave_tokens:
        ap.error("unknown or empty: " + ", ".join(bad_modes + bad_waves or ["--modes/--waves"]))
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
        kernels, meta = build_jobs(modes, waves)
        raw_results = oracle.run(kernels)
        summary = evaluate_results(modes, waves, meta, raw_results)
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()

    report = {
        "device": str(oracle.device),
        "arch": oracle.arch,
        "waves": waves,
        "modes": modes,
        "mods": list(MODS),
        "inputs": len(INPUTS),
        **summary,
    }
    if a.json and not write_report(a.json, report):
        return 2

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: wave{err['wave']} {err['mode']} {err['mod']!r}: {err['error']}")
        return 2

    print(
        f"{summary['total_cases']} evaluations across {len(waves)} wave(s) x {len(modes)} mode(s) x "
        f"{len(MODS)} mod(s): lo_mismatches={summary['lo_mismatches']}, hi_mismatches={summary['hi_mismatches']}, "
        f"wave_diffs={summary['wave_diffs']}, dyn_rtz_diffs={summary['dyn_rtz_diffs']}"
    )
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH wave{m['wave']} {m['mode']} {m['mod']!r} {m['input']} ({m['f32']}): got {m['got']}, want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['mode']} {d['mod']!r} {d['input']}: {d}")

    if summary["lo_mismatches"] or summary["hi_mismatches"] or summary["wave_diffs"] or summary["dyn_rtz_diffs"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
