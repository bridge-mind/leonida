r"""Measure the VOP3 output modifiers of the binary64 conversion, frexp and div_fixup VALU ops across MODE settings. Standard library only.

docs/design.md records for the f32 VALU ops that `mul:2`/`mul:4`/`div:2` are applied only when MODE.IEEE = 0 and the
output denormals of the destination precision are flushed, and `clamp` in every mode. Running this probe natively on
RDNA2 (gfx1036) through tools/hw_oracle extends that gate to the remaining binary64 and 16-bit results:
`v_cvt_f64_i32`, `v_cvt_f64_u32`, `v_frexp_mant_f64`, `v_div_fixup_f64` and `v_div_fixup_f16` ignore the scaling
while `denorm_mode_16_64 = 3` — which is the mode the translator assumes — and apply it only with IEEE = 0 and
`denorm_mode_16_64 = 0`, while `clamp` still applies in every mode. `v_cvt_f32_f64` writes an f32 result and follows
the f32 gate: it scales whenever `denorm32 = 0`, including under `denorm_mode_16_64 = 3`, and stops scaling when
`denorm32 = 3`. The gate is therefore per result precision, not per instruction family. wave32 and wave64 agree on
every word written; for the 16-bit form only the low half of the destination is written, as the ISA defines.

Every expectation is derived from the plain (no-modifier) result measured in the same mode on the same device, so the
probe compares the effect of the modifier and not the semantics of the operation.

Usage:
    python -I tools/f64_omod_probe.py [--waves 32,64] [--modes ...] [--json out.json]
    python -I tools/f64_omod_probe.py --self-test

Exit 0 when every measured result matches the model (with wave32 and wave64 agreeing bit-for-bit), 1 on a model
mismatch or a wave32/wave64 difference, 2 on a harness or CLI error, and 77 when the hardware oracle is unavailable or
the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODES = {
    # The mode the translator emits and the game runs with: scaling ignored, clamp applied.
    "game_i0_d3": {"round16_64": 0, "denorm16_64": 3, "denorm32": 0, "dx10_clamp": 1, "ieee": 0},
    # 16/64-bit denormals flushed, IEEE off: the scaling applies to the f16 and binary64 results.
    "i0_d0": {"round16_64": 0, "denorm16_64": 0, "denorm32": 0, "dx10_clamp": 1, "ieee": 0},
    # IEEE on: the scaling is ignored in both denormal settings.
    "i1_d3": {"round16_64": 0, "denorm16_64": 3, "denorm32": 0, "dx10_clamp": 1, "ieee": 1},
    "i1_d0": {"round16_64": 0, "denorm16_64": 0, "denorm32": 0, "dx10_clamp": 1, "ieee": 1},
    # f32 denormals kept while 16/64-bit ones are flushed: the discriminator for the f32 result of v_cvt_f32_f64.
    "i0_d3_f32keep": {"round16_64": 0, "denorm16_64": 3, "denorm32": 3, "dx10_clamp": 1, "ieee": 0},
}

MODS = ("", "mul:2", "mul:4", "div:2", "clamp")
SCALE = {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}

INT_SOURCES = [("0", 0x00000000), ("1", 0x00000001), ("-1", 0xFFFFFFFF), ("2^24+1", 0x01000001),
               ("2^31-1", 0x7FFFFFFF), ("-2^31", 0x80000000)]
F64_TO_F32 = [("1.0", 1.0), ("2.0", 2.0), ("1.5", 1.5), ("3.25", 3.25), ("-1.0", -1.0),
              ("2^-1074 min sub", 2.0 ** -1074), ("2^-1050 sub", 2.0 ** -1050), ("2^-130", 2.0 ** -130)]
FREXP_INPUTS = [("3.0", 3.0), ("1.0", 1.0), ("0.5", 0.5), ("-1.75", -1.75), ("8.0", 8.0),
                ("2^-1074 min sub", 2.0 ** -1074), ("1.5*2^-1050 sub", 1.5 * 2.0 ** -1050)]


def _f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _f16_bits(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return (0x8000 if math.copysign(1.0, x) < 0 else 0) | 0x7C00


def load(pair_values, first=1):
    lines = []
    for i, value in enumerate(pair_values):
        bits = _f64_bits(value)
        lines.append(f"v_mov_b32_e32 v{first + 2 * i}, {bits & 0xFFFFFFFF:#010x}")
        lines.append(f"v_mov_b32_e32 v{first + 2 * i + 1}, {bits >> 32:#010x}")
    return lines


# (op, e64 form, result kind, [(case name, setup lines)])
OPS = [
    ("v_cvt_f64_i32", "v_cvt_f64_i32_e64 v[8:9], v1", "f64",
     [(name, [f"v_mov_b32_e32 v1, {bits:#010x}"]) for name, bits in INT_SOURCES]),
    ("v_cvt_f64_u32", "v_cvt_f64_u32_e64 v[8:9], v1", "f64",
     [(name, [f"v_mov_b32_e32 v1, {bits:#010x}"]) for name, bits in INT_SOURCES]),
    ("v_cvt_f32_f64", "v_cvt_f32_f64_e64 v8, v[1:2]", "f32",
     [(name, load([value])) for name, value in F64_TO_F32]),
    ("v_frexp_mant_f64", "v_frexp_mant_f64_e64 v[8:9], v[1:2]", "f64",
     [(name, load([value])) for name, value in FREXP_INPUTS]),
    ("v_div_fixup_f64", "v_div_fixup_f64_e64 v[8:9], v[1:2], v[3:4], v[5:6]", "f64",
     [(name, load(values)) for name, values in
      [("q=2 d=1.5 n=3", (2.0, 1.5, 3.0)), ("q=-2 d=1.5 n=3", (-2.0, 1.5, 3.0)),
       ("q=2 d=-1.5 n=3", (2.0, -1.5, 3.0)), ("q=0.75 d=2 n=1.5", (0.75, 2.0, 1.5))]]),
    ("v_div_fixup_f16", "v_div_fixup_f16_e64 v8, v1, v2, v3", "f16",
     [(name, [f"v_mov_b32_e32 v1, {h:#010x}", f"v_mov_b32_e32 v2, {d:#010x}", f"v_mov_b32_e32 v3, {n:#010x}"])
      for name, h, d, n in [("q=2h d=1.5h n=3h", _f16_bits(2.0), _f16_bits(1.5), _f16_bits(3.0)),
                            ("q=-2h d=1.5h n=3h", _f16_bits(-2.0), _f16_bits(1.5), _f16_bits(3.0)),
                            ("q=0.75h d=2h n=1.5h", _f16_bits(0.75), _f16_bits(2.0), _f16_bits(1.5)),
                            ("q=1h d=1h n=1h", _f16_bits(1.0), _f16_bits(1.0), _f16_bits(1.0))]]),
]


def word_count(kind: str) -> int:
    return 2 if kind == "f64" else 1


def build_kernels(asm, modes, waves):
    kernels, meta = [], []
    for wave in waves:
        for mode_name in modes:
            for op, form, kind, cases in OPS:
                lines, words = [], 0
                for _, setup in cases:
                    for mod in MODS:
                        lines.extend(setup)
                        lines.append(form + (f" {mod}" if mod else ""))
                        lines.append(f"v_mov_b32_e32 v7, {words * 4}")
                        lines.append("global_store_dword v7, v8, s[2:3]")
                        words += 1
                        if kind == "f64":
                            lines.append("v_lshrrev_b64 v[10:11], 32, v[8:9]")
                            lines.append(f"v_mov_b32_e32 v7, {words * 4}")
                            lines.append("global_store_dword v7, v10, s[2:3]")
                            words += 1
                lines.append("s_waitcnt vmcnt(0)")
                lines.append("s_endpgm")
                kernels.append(asm.Kernel(lines, wave=wave, threads=1, words=words, raw=True,
                                          mode=dict(MODES[mode_name])))
                meta.append((wave, mode_name, op, kind, [name for name, _ in cases]))
    return kernels, meta


def scalar_value(lo, hi, kind):
    if kind == "f64":
        return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)))[0]
    if kind == "f32":
        return struct.unpack("<f", struct.pack("<I", lo & 0xFFFFFFFF))[0]
    return struct.unpack("<e", struct.pack("<H", lo & 0xFFFF))[0]


def scalar_bits(value, kind):
    if kind == "f64":
        bits = _f64_bits(value)
        return (bits & 0xFFFFFFFF, (bits >> 32) & 0xFFFFFFFF)
    if kind == "f32":
        return (struct.unpack("<I", struct.pack("<f", value))[0], 0)
    return (_f16_bits(value), 0)


def scaling_applies(kind, descriptor):
    """The scaling omod applies only with IEEE off and the result precision's denormals flushed."""
    if descriptor["ieee"]:
        return False
    if kind == "f32":
        return descriptor["denorm32"] == 0
    return descriptor["denorm16_64"] == 0


def expected_words(plain, mod, kind, descriptor):
    """Model of what the hardware does to the plain result of the same mode, given the modifier."""
    lo, hi = plain[0], (plain[1] if kind == "f64" else 0)
    if not mod:
        return (lo, hi)
    value = scalar_value(lo, hi, kind)
    if not math.isfinite(value):
        return (lo, hi)  # inf and NaN pass every modifier unchanged on this hardware
    if mod in SCALE:
        if not scaling_applies(kind, descriptor):
            return (lo, hi)
        return scalar_bits(value * SCALE[mod], kind)
    if mod == "clamp":
        return scalar_bits(min(max(value, 0.0), 1.0), kind)
    raise ValueError(f"unknown modifier {mod!r}")


def evaluate(modes, waves, meta, raw_results):
    measured = {}
    for (wave, mode_name, op, kind, cases), result in zip(meta, raw_results):
        words = [w & 0xFFFFFFFF for w in (result["words"] if isinstance(result, dict) else result)]
        # A 16-bit destination writes only bits [15:0]; the rest of the register is untouched residue.
        mask = 0xFFFF if kind == "f16" else 0xFFFFFFFF
        step = word_count(kind)
        per_mod = {mod: [] for mod in MODS}
        index = 0
        for _ in cases:
            for mod in MODS:
                per_mod[mod].extend(w & mask for w in words[index:index + step])
                index += step
        measured[(wave, mode_name, op)] = (cases, per_mod)

    summary = {"total_cases": 0, "scaling_gated_on": 0, "scaling_gated_off": 0,
               "scaling_changed_value": 0, "clamp_changed_value": 0,
               "mismatches": 0, "wave_diffs": 0, "first_mismatches": [], "first_wave_diffs": [],
               "harness_errors": []}
    for mode_name in modes:
        for op, _form, kind, cases in OPS:
            step = word_count(kind)
            for ci, case_name in enumerate(cases):
                summary["total_cases"] += 1
                plain = tuple(measured[(waves[0], mode_name, op)][1][""][ci * step:ci * step + step])
                for mod in MODS[1:]:
                    got = tuple(measured[(waves[0], mode_name, op)][1][mod][ci * step:ci * step + step])
                    want = tuple(expected_words(plain, mod, kind, MODES[mode_name]))[:step]
                    if mod in SCALE:
                        summary["scaling_gated_on" if scaling_applies(kind, MODES[mode_name]) else "scaling_gated_off"] += 1
                        if got != plain:
                            summary["scaling_changed_value"] += 1
                    if mod == "clamp" and got != plain:
                        summary["clamp_changed_value"] += 1
                    if got != want:
                        summary["mismatches"] += 1
                        if len(summary["first_mismatches"]) < 10:
                            summary["first_mismatches"].append({
                                "mode": mode_name, "op": op, "case": case_name, "mod": mod,
                                "plain": [f"{w:08x}" for w in plain], "got": [f"{w:08x}" for w in got],
                                "want": [f"{w:08x}" for w in want]})
            for wave in waves[1:]:
                for mod in MODS:
                    base = measured[(waves[0], mode_name, op)][1][mod]
                    other = measured[(wave, mode_name, op)][1][mod]
                    if base != other:
                        summary["wave_diffs"] += 1
                        if len(summary["first_wave_diffs"]) < 5:
                            summary["first_wave_diffs"].append({"mode": mode_name, "op": op, "mod": mod,
                                                               "wave": wave})
    return summary


def self_test():
    game = MODES["game_i0_d3"]
    flush = MODES["i0_d0"]
    keep32 = MODES["i0_d3_f32keep"]
    ieee = MODES["i1_d0"]

    def pair(x):
        return scalar_bits(x, "f64")

    # A binary64 result ignores the scaling in the game mode and scales with denorm_mode_16_64 = 0.
    plain = pair(2147483647.0)
    assert expected_words(plain, "mul:2", "f64", game) == plain
    assert expected_words(plain, "div:2", "f64", game) == plain
    assert expected_words(plain, "mul:2", "f64", flush) == pair(4294967294.0)
    assert expected_words(plain, "mul:4", "f64", ieee) == plain
    # A negative mantissa is clamped to +0 in every mode.
    mant = pair(-0.875)
    assert expected_words(mant, "clamp", "f64", game) == pair(0.0)
    assert expected_words(mant, "clamp", "f64", ieee) == pair(0.0)
    assert expected_words(pair(0.75), "clamp", "f64", game) == pair(0.75)
    # inf and NaN pass through unchanged.
    assert expected_words(pair(float("inf")), "mul:2", "f64", flush) == pair(float("inf"))
    # The f32 result of v_cvt_f32_f64 follows the f32 gate, not the 16/64-bit one.
    f32_plain = scalar_bits(0.75, "f32")
    assert expected_words(f32_plain, "mul:2", "f32", game) == scalar_bits(1.5, "f32")
    assert expected_words(f32_plain, "mul:2", "f32", keep32) == f32_plain
    assert expected_words(f32_plain, "mul:2", "f32", ieee) == f32_plain
    # The f16 result of v_div_fixup_f16 follows the 16/64-bit gate and saturates under clamp.
    f16_plain = scalar_bits(2.0, "f16")
    assert expected_words(f16_plain, "mul:2", "f16", game) == f16_plain
    assert expected_words(f16_plain, "mul:2", "f16", flush) == scalar_bits(4.0, "f16")
    assert expected_words(f16_plain, "clamp", "f16", game) == scalar_bits(1.0, "f16")
    assert expected_words(f16_plain, "mul:4", "f16", game) == f16_plain
    cases = sum(len(ops[3]) for ops in OPS)
    print(f"self-test ok ({len(OPS)} ops, {cases} cases, {len(MODES)} modes x {len(MODS)} modifier forms)")
    return 0


def write_report(path, report):
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        return True
    except OSError as exc:
        print(f"cannot write {path}: {exc}")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--json", help="write the measurement report here")
    ap.add_argument("--hw-arch")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    wave_tokens = [w.strip() for w in args.waves.split(",") if w.strip()]
    bad = [m for m in modes if m not in MODES] + [w for w in wave_tokens if w not in ("32", "64")]
    if bad or not modes or not wave_tokens:
        ap.error("unknown or empty --modes/--waves entry: " + ", ".join(bad or ["--modes/--waves"]))
    waves = [int(w) for w in wave_tokens]
    if args.json and os.path.isdir(args.json):
        ap.error(f"--json {args.json} is a folder, not a file")

    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle(arch=args.hw_arch)
    except runner.Unavailable as exc:
        print(f"hardware oracle unavailable: {exc}")
        return 77
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10")
            return 77
        print(f"hardware oracle: {oracle.device} ({oracle.arch})")
        kernels, meta = build_kernels(asm, modes, waves)
        summary = evaluate(modes, waves, meta, oracle.run(kernels))
    except (RuntimeError, KeyError, TypeError) as exc:
        print(f"harness error: {exc}")
        return 2
    finally:
        oracle.close()

    if args.json and not write_report(args.json, {
            "device": str(oracle.device), "arch": oracle.arch, "modes": modes, "waves": waves,
            "mods": list(MODS), "ops": [op for op, *_ in OPS], **summary}):
        return 2

    print(f"{summary['total_cases']} cases across {len(modes)} mode(s) x {len(MODS)} modifier forms "
          f"x {len(waves)} wave(s): scaling_gated_on={summary['scaling_gated_on']}, "
          f"scaling_gated_off={summary['scaling_gated_off']}, "
          f"scaling_changed_value={summary['scaling_changed_value']}, "
          f"clamp_changed_value={summary['clamp_changed_value']}, "
          f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}")
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH {m['mode']} {m['op']} {m['case']} {m['mod']}: plain {m['plain']} "
              f"got {m['got']} want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['mode']} {d['op']} {d['mod']!r} wave{d['wave']}")
    return 1 if (summary["mismatches"] or summary["wave_diffs"]) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
