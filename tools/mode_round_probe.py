r"""Measure RDNA2 (GFX10) f16/f64 results under MODE.FP_ROUND[3:2] nearest-even vs toward-zero (SHADER-132).

Runs two native sweeps through tools/hw_oracle in wave32 and wave64:
1. Targeted f16/f64 rounding cases (v_cvt_f16_f32, v_add_f16, v_sub_f16, v_mul_f16, v_fma_f16, v_pk_add_f16,
   v_pk_fma_f16, v_fma_mixlo_f16, v_fma_mixhi_f16, v_add_f64 including exact-sum and finite-overflow saturation)
   under s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0 (RNE) vs 3 (RTZ).
2. Exhaustive 65,536-input RNE vs RTZ comparison of all seven f16 transcendentals (v_rcp_f16, v_rsq_f16,
   v_sqrt_f16, v_sin_f16, v_cos_f16, v_exp_f16, v_log_f16), confirming that none of them depend on MODE.FP_ROUND.

Usage:
    python -I tools/mode_round_probe.py [--waves 32,64] [--json out.json]
    python -I tools/mode_round_probe.py --self-test        # pure-Python sanity checks, no GPU

Exit 0 when every case matches (or --self-test passed), 1 on a mismatch or --json write failure, 2 on a harness error or invalid CLI arguments, 77 when
tools/hw_oracle is unavailable or the GPU is not GFX10. Standard library only; nothing here reads game files.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"
RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"

# (name, setup_lines, op_line, is_64bit, want_rne, want_rtz)
CASES = (
    ("cvt_f16_f32_pos", ("v_mov_b32 v1, 0x3f801800",), "v_cvt_f16_f32_e32 v3, v1", False, 0x3C01, 0x3C00),
    ("cvt_f16_f32_neg", ("v_mov_b32 v1, 0xbf801800",), "v_cvt_f16_f32_e32 v3, v1", False, 0xBC01, 0xBC00),
    ("cvt_f16_f32_ovf", ("v_mov_b32 v1, 0x4788b800",), "v_cvt_f16_f32_e32 v3, v1", False, 0x7C00, 0x7BFF),
    ("cvt_f16_f32_novf", ("v_mov_b32 v1, 0xc7800000",), "v_cvt_f16_f32_e32 v3, v1", False, 0xFC00, 0xFBFF),
    ("cvt_f16_f32_sub", ("v_mov_b32 v1, 0x33c00000",), "v_cvt_f16_f32_e32 v3, v1", False, 0x0002, 0x0001),
    ("add_f16_exact", ("v_mov_b32 v1, 0x4000", "v_mov_b32 v2, 0x8001"),
     "v_add_f16_e32 v3, v1, v2", False, 0x4000, 0x3FFF),
    ("add_f16_ovf", ("v_mov_b32 v1, 0x7bff", "v_mov_b32 v2, 0x4c00"),
     "v_add_f16_e32 v3, v1, v2", False, 0x7C00, 0x7BFF),
    ("sub_f16_neg", ("v_mov_b32 v1, 0xbc00", "v_mov_b32 v2, 0x1001"),
     "v_sub_f16_e32 v3, v1, v2", False, 0xBC01, 0xBC00),
    ("mul_f16_halfway", ("v_mov_b32 v1, 0x3e01", "v_mov_b32 v2, 0x3e01"),
     "v_mul_f16_e32 v3, v1, v2", False, 0x4082, 0x4081),
    ("fma_f16_exact", ("v_mov_b32 v1, 0x4000", "v_mov_b32 v2, 0x3c00", "v_mov_b32 v5, 0x8001"),
     "v_fma_f16 v3, v1, v2, v5", False, 0x4000, 0x3FFF),
    ("pk_add_f16", ("v_mov_b32 v1, 0xbc003c00", "v_mov_b32 v2, 0x92001200"),
     "v_pk_add_f16 v3, v1, v2", False, 0xBC013C01, 0xBC003C00),
    ("pk_fma_f16", ("v_mov_b32 v1, 0xc0004000", "v_mov_b32 v2, 0x3c003c00", "v_mov_b32 v5, 0x00018001"),
     "v_pk_fma_f16 v3, v1, v2, v5", False, 0xC0004000, 0xBFFF3FFF),
    ("fma_mixlo_f16", ("v_mov_b32 v1, 0x3f800000", "v_mov_b32 v2, 0x40000000", "v_mov_b32 v5, 0xb0800000"),
     "v_fma_mixlo_f16 v3, v1, v2, v5", False, 0x4000, 0x3FFF),
    ("fma_mixhi_f16", ("v_mov_b32 v1, 0x3f800000", "v_mov_b32 v2, 0xc0000000",
                       "v_mov_b32 v5, 0x30800000", "v_mov_b32 v3, 0x1234"),
     "v_fma_mixhi_f16 v3, v1, v2, v5", False, 0xC0001234, 0xBFFF1234),
    ("fma_mixlo_ovf", ("v_mov_b32 v1, 0x5f800000", "v_mov_b32 v2, 0x5f800000", "v_mov_b32 v5, 0"),
     "v_fma_mixlo_f16 v3, v1, v2, v5", False, 0x7C00, 0x7BFF),
    ("add_f64", ("v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff00000", "v_mov_b32 v6, 0", "v_mov_b32 v7, 0xbc300000"),
     "v_add_f64 v[3:4], v[0:1], v[6:7]", True, (0x00000000, 0x3FF00000), (0xFFFFFFFF, 0x3FEFFFFF)),
    ("add_f64_ovf", ("v_mov_b32 v0, -1", "v_mov_b32 v1, 0x7fefffff"),
     "v_add_f64 v[3:4], v[0:1], v[0:1]", True, (0x00000000, 0x7FF00000), (0xFFFFFFFF, 0x7FEFFFFF)),
)

TRANSCENDENTALS = (
    ("v_rcp_f16_e32", 0x3037, 0x4798),
    ("v_rsq_f16_e32", 0x3037, 0x4183),
    ("v_sqrt_f16_e32", 0x4200, 0x3EEE),
    ("v_sin_f16_e32", 0x3037, 0x39E4),
    ("v_cos_f16_e32", 0x3037, 0x396A),
    ("v_exp_f16_e32", 0x3037, 0x3C62),
    ("v_log_f16_e32", 0x4A37, 0x4345),
)


def _case_kernel(asm_mod, setup, op, is_64, wave, mode_set):
    lines = list(setup) + [mode_set, "s_nop 0", op, "v_mov_b32 v10, 0", "global_store_b32 v10, v3, s[2:3]"]
    if is_64:
        lines += ["v_mov_b32 v11, 4", "global_store_b32 v11, v4, s[2:3]"]
    lines += ["s_waitcnt_vscnt null, 0x0", "s_endpgm"]
    return asm_mod.Kernel(lines, wave, 1, 2, raw=True, vgprs=16, sgprs=16)


def probe_cases(oracle, asm_mod, wave):
    ks = []
    for _, setup, op, is_64, _, _ in CASES:
        ks.append(_case_kernel(asm_mod, setup, op, is_64, wave, RNE))
        ks.append(_case_kernel(asm_mod, setup, op, is_64, wave, RTZ))
    res = oracle.run(ks, words=2)
    ok = 0
    mismatches = []
    for i, (name, _, _, is_64, want_rne, want_rtz) in enumerate(CASES):
        r_rne, r_rtz = res[2 * i], res[2 * i + 1]
        if isinstance(r_rne, str) or isinstance(r_rtz, str):
            raise RuntimeError(f"{name} wave{wave}: rne={r_rne} rtz={r_rtz}")
        if is_64:
            got_rne, got_rtz = (r_rne[0], r_rne[1]), (r_rtz[0], r_rtz[1])
        else:
            mask = 0xFFFFFFFF if want_rne > 0xFFFF else 0xFFFF
            got_rne, got_rtz = r_rne[0] & mask, r_rtz[0] & mask
        if got_rne == want_rne and got_rtz == want_rtz:
            ok += 1
        else:
            mismatches.append({"case": name, "got_rne": got_rne, "want_rne": want_rne,
                               "got_rtz": got_rtz, "want_rtz": want_rtz})
    return {"total": len(CASES), "ok": ok, "mismatches": mismatches}


def probe_transcendentals(oracle, asm_mod, wave):
    """Exhaustive 65,536-input RNE vs RTZ diff + spot-check of measured nearest-even words."""
    report = {}
    all_ok = True
    for op, sample_in, sample_want in TRANSCENDENTALS:
        ks = []
        for base in range(0, 65536, 1024):
            lines = [
                f"v_add_nc_u32 v1, 0x{base:x}, v0",
                RNE, "s_nop 0", f"{op} v2, v1",
                RTZ, "s_nop 0", f"{op} v3, v1",
                "v_xor_b32 v4, v2, v3",
                "v_and_b32 v4, 0xffff, v4",
                "v_lshlrev_b32 v5, 2, v0",
                "global_store_b32 v5, v4, s[2:3]",
                "s_waitcnt_vscnt null, 0x0",
                "s_endpgm",
            ]
            ks.append(asm_mod.Kernel(lines, wave, 1024, 1024, raw=True, vgprs=8, sgprs=16))
        res = oracle.run(ks, words=1024)
        diffs = 0
        for chunk in res:
            if isinstance(chunk, str) or len(chunk) != 1024:
                raise RuntimeError(f"{op} wave{wave}: {chunk}")
            diffs += sum(1 for w in chunk if w != 0)
        spot_k = _case_kernel(asm_mod, (f"v_mov_b32 v1, {sample_in:#x}",), f"{op} v3, v1", False, wave, RTZ)
        spot_got = oracle.run([spot_k], words=2)[0][0] & 0xFFFF
        ok = (diffs == 0) and (spot_got == sample_want)
        if not ok:
            all_ok = False
        report[op] = {"inputs": 65536, "rne_rtz_diffs": diffs, "spot_got": f"{spot_got:#06x}",
                      "spot_want": f"{sample_want:#06x}", "ok": ok}
    return {"all_ok": all_ok, "ops": report}


def self_test():
    # Every targeted case must distinguish RNE from RTZ
    for name, _, _, _, want_rne, want_rtz in CASES:
        assert want_rne != want_rtz, f"{name} does not distinguish RNE from RTZ"
    assert len(TRANSCENDENTALS) == 7
    print("self-test OK")


def parse_waves(raw):
    """Parse --waves comma-separated wave sizes; return (waves, error_message)."""
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None, "--waves must not be empty"
    waves = []
    for tok in tokens:
        try:
            w = int(tok)
        except ValueError:
            return None, f"invalid wave size in --waves: {tok!r}"
        if w not in (32, 64):
            return None, f"unsupported wave size in --waves: {w} (expected 32 or 64)"
        waves.append(w)
    return waves, None


def validate_json_path(path):
    """Return an error string if `path` cannot be used as a --json output file, or None."""
    if not path or not path.strip():
        return "--json path must not be empty"
    if "\x00" in path:
        return "--json path contains a NUL byte"
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)) or os.path.isdir(path):
        return f"--json path is a directory: {path}"
    cur = os.path.dirname(os.path.abspath(path))
    while cur:
        if os.path.isdir(cur):
            break
        if os.path.lexists(cur):
            return f"--json parent is not a directory: {cur}"
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def write_json_report(path, data):
    """Create missing parent directories and write `data` as LF-terminated JSON, or return an error string."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            return f"cannot create directory for --json {path}: {exc.strerror or exc}"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        return f"cannot write --json {path}: {exc.strerror or exc}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="wave sizes to run (default: 32,64)")
    ap.add_argument("--json", dest="json_out", help="write full JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="check the case table without a GPU")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    waves, wave_err = parse_waves(args.waves)
    if wave_err:
        print(f"error: {wave_err}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        json_err = validate_json_path(args.json_out)
        if json_err:
            print(f"error: {json_err}", file=sys.stderr)
            return 2
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return 77
    if asm.family(oracle.arch) != "gfx10":
        print(f"SKIP: {oracle.arch} is not GFX10 (RDNA1/RDNA2)", file=sys.stderr)
        return 77
    out = {"arch": oracle.arch, "waves": {}}
    mismatches = 0
    for wave in waves:
        c = probe_cases(oracle, asm, wave)
        t = probe_transcendentals(oracle, asm, wave)
        out["waves"][str(wave)] = {"cases": c, "transcendentals": t}
        if c["ok"] != c["total"] or not t["all_ok"]:
            mismatches += 1
        t_ok = sum(info["ok"] for info in t["ops"].values())
        print(f"wave{wave} ({oracle.arch}): {c['ok']}/{c['total']} f16/f64 RNE/RTZ cases exact; "
              f"{t_ok}/{len(t['ops'])} f16 transcendentals 0/65536 RNE/RTZ diffs")
        for op, info in t["ops"].items():
            if not info["ok"]:
                print(f"  {op}: {info['rne_rtz_diffs']}/{info['inputs']} RNE/RTZ diffs, "
                      f"RTZ spot word {info['spot_got']} (want {info['spot_want']})")
    if args.json_out is not None:
        write_err = write_json_report(args.json_out, out)
        if write_err:
            print(f"error: {write_err}", file=sys.stderr)
            return 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
