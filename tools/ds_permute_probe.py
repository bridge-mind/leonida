r"""Measure RDNA2 (GFX10) `ds_permute_b32` natively via `tools/hw_oracle` (SHADER-133).

Covers collisions (2-way, 4-way, 32-way, and with the highest lane masked off in EXEC), unaddressed active
destination lanes, EXEC off on source and destination, `offset:` addition and low-bit carry (`((vAddr + offset)
>> 2) mod 32`), wave64 independent 32-lane halves, and partial waves (20 threads in wave32, 40 in wave64).

Usage:
    python -I tools/ds_permute_probe.py [--waves 32,64] [--json out.json]
    python -I tools/ds_permute_probe.py --self-test        # pure-Python model checks, no GPU

Exit 0 when every case matches the RDNA2 model (or --self-test passed), 1 on a mismatch or report write error,
2 on a harness or CLI error, 77 when tools/hw_oracle is unavailable or the GPU is not GFX10. Standard library only;
nothing here reads game files.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SENTINEL = 0xCAFE0000
PAYLOAD_BASE = 0x1000


def model_ds_permute(wave, addrs, exec_mask, offset=0, threads=None):
    """RDNA2 hardware model for ds_permute_b32 (wave32 or wave64).

    Wave64 runs as two independent 32-lane halves. Within each half:
    - active source lane i sends (PAYLOAD_BASE + lane) to target ((addrs[lane] + offset) >> 2) % 32
    - on a collision, the highest-numbered active lane in the half wins
    - unaddressed active destination lanes receive 0
    - inactive destination lanes keep SENTINEL | lane
    - lanes >= threads do not execute (initial EXEC bit is 0)
    """
    if threads is None:
        threads = wave
    live_mask = ((1 << threads) - 1) & exec_mask
    out = [SENTINEL | l for l in range(wave)]
    for base in range(0, wave, 32):
        tmp = [0] * 32
        for i in range(32):
            lane = base + i
            if (live_mask >> lane) & 1:
                dst = ((addrs[lane] + offset) >> 2) & 31
                tmp[dst] = PAYLOAD_BASE + lane
        for i in range(32):
            lane = base + i
            if (live_mask >> lane) & 1:
                out[lane] = tmp[i]
    return out


def _make_kernel(asm_mod, body_lines, wave, threads=None):
    if threads is None:
        threads = wave
    restore = "s_mov_b32 exec_lo, 0xffffffff" if wave == 32 else "s_mov_b64 exec, -1"
    lines = [
        f"v_or_b32 v10, 0x{SENTINEL:x}, v0",
        f"v_add_nc_u32 v2, 0x{PAYLOAD_BASE:x}, v0",
    ] + body_lines + [
        restore,
        "v_lshlrev_b32 v11, 2, v0",
        "global_store_b32 v11, v10, s[2:3]",
        "s_waitcnt_vscnt null, 0x0",
        "s_endpgm",
    ]
    return asm_mod.Kernel(lines, wave, threads, wave, raw=True, vgprs=16, sgprs=16)


def build_cases(wave):
    """List of (name, body_lines, addrs, exec_mask, offset, threads) for wave32 or wave64."""
    full_mask = (1 << wave) - 1
    mask_no_lo8 = 0xFFFFFF00 if wave == 32 else ((0xFFFFFF00 << 32) | 0xFFFFFF00)
    mask_no_top = 0x7FFFFFFF if wave == 32 else ((0x7FFFFFFF << 32) | 0x7FFFFFFF)
    exec_no_lo8 = (["s_mov_b32 exec_lo, 0xffffff00"] if wave == 32
                   else ["s_mov_b32 exec_lo, 0xffffff00", "s_mov_b32 exec_hi, 0xffffff00"])
    exec_no_top = (["s_mov_b32 exec_lo, 0x7fffffff"] if wave == 32
                   else ["s_mov_b32 exec_lo, 0x7fffffff", "s_mov_b32 exec_hi, 0x7fffffff"])
    part_threads = 20 if wave == 32 else 40
    return [
        ("rot_plus1_offset4_lowbits",
         ["v_lshlrev_b32 v3, 2, v0", "v_and_b32 v4, 3, v0", "v_or_b32 v3, v3, v4",
          "ds_permute_b32 v10, v3, v2 offset:4", "s_waitcnt lgkmcnt(0)"],
         [(l << 2) | (l & 3) for l in range(wave)], full_mask, 4, wave),
        ("offset_lowbit_carry",
         ["v_mov_b32 v3, 3", "ds_permute_b32 v10, v3, v2 offset:1", "s_waitcnt lgkmcnt(0)"],
         [3] * wave, full_mask, 1, wave),
        ("unaddressed_odd_and_2way_collision",
         ["v_and_b32 v3, 0xfffffffe, v0", "v_lshlrev_b32 v3, 2, v3",
          "ds_permute_b32 v10, v3, v2", "s_waitcnt lgkmcnt(0)"],
         [(l & ~1) << 2 for l in range(wave)], full_mask, 0, wave),
        ("collision_4way",
         ["v_and_b32 v3, 0xfffffffc, v0", "v_lshlrev_b32 v3, 2, v3",
          "ds_permute_b32 v10, v3, v2", "s_waitcnt lgkmcnt(0)"],
         [(l & ~3) << 2 for l in range(wave)], full_mask, 0, wave),
        ("collision_all_to_0",
         ["v_mov_b32 v3, 0", "ds_permute_b32 v10, v3, v2", "s_waitcnt lgkmcnt(0)"],
         [0] * wave, full_mask, 0, wave),
        ("collision_all_to_0_top_off",
         exec_no_top + ["v_mov_b32 v3, 0", "ds_permute_b32 v10, v3, v2", "s_waitcnt lgkmcnt(0)"],
         [0] * wave, mask_no_top, 0, wave),
        ("exec_off_src_and_dst",
         exec_no_lo8 + ["v_lshlrev_b32 v3, 2, v0", "ds_permute_b32 v10, v3, v2 offset:4", "s_waitcnt lgkmcnt(0)"],
         [l << 2 for l in range(wave)], mask_no_lo8, 4, wave),
        ("addr_plus32_wrap_halves",
         ["v_add_nc_u32 v3, 32, v0", "v_lshlrev_b32 v3, 2, v3",
          "ds_permute_b32 v10, v3, v2", "s_waitcnt lgkmcnt(0)"],
         [(l + 32) << 2 for l in range(wave)], full_mask, 0, wave),
        ("partial_wave",
         ["v_lshlrev_b32 v3, 2, v0", "ds_permute_b32 v10, v3, v2 offset:4", "s_waitcnt lgkmcnt(0)"],
         [l << 2 for l in range(wave)], full_mask, 4, part_threads),
    ]


def probe_wave(oracle, asm_mod, wave):
    cases = build_cases(wave)
    kernels = [_make_kernel(asm_mod, body, wave, threads) for _, body, _, _, _, threads in cases]
    results = oracle.run(kernels, words=wave)
    report = {}
    ok = 0
    for (name, _, addrs, exec_mask, offset, threads), got in zip(cases, results):
        if isinstance(got, str) or len(got) < threads:
            raise RuntimeError(f"{name} wave{wave}: {got}")
        want = model_ds_permute(wave, addrs, exec_mask, offset, threads)
        # Only check running lanes 0..threads-1 (lanes >= threads did not execute global_store_b32).
        diffs = [{"lane": i, "hw": f"{got[i]:#x}", "want": f"{want[i]:#x}"}
                 for i in range(threads) if got[i] != want[i]]
        if not diffs:
            ok += 1
        report[name] = {"lanes_checked": threads, "mismatches": len(diffs), "first_diffs": diffs[:4]}
    return {"cases_total": len(cases), "cases_ok": ok, "cases": report}


def self_test():
    # Wave32: all-to-0 collision -> highest lane 31 wins; lane 1..31 unaddressed -> 0
    w32 = model_ds_permute(32, [0] * 32, 0xFFFFFFFF)
    assert w32[0] == PAYLOAD_BASE + 31 and all(w == 0 for w in w32[1:])
    # Wave64: two independent halves, +32 wrap keeps each lane in its own half
    w64 = model_ds_permute(64, [(l + 32) << 2 for l in range(64)], (1 << 64) - 1)
    assert w64 == [PAYLOAD_BASE + l for l in range(64)]
    # EXEC off on destination keeps SENTINEL|lane, off source leaves target 0
    w32_e = model_ds_permute(32, [l << 2 for l in range(32)], 0xFFFFFF00, offset=4)
    assert w32_e[0] == (SENTINEL | 0) and w32_e[8] == 0 and w32_e[9] == PAYLOAD_BASE + 8
    # Low-bit carry: (3 + 1) >> 2 == 1
    w32_c = model_ds_permute(32, [3] * 32, 0xFFFFFFFF, offset=1)
    assert w32_c[1] == PAYLOAD_BASE + 31 and w32_c[0] == 0
    # Partial wave64 (40 threads): low half wraps, high half lane 32 gets 0
    w64_p = model_ds_permute(64, [l << 2 for l in range(64)], (1 << 64) - 1, offset=4, threads=40)
    assert w64_p[0] == PAYLOAD_BASE + 31 and w64_p[32] == 0 and w64_p[33] == PAYLOAD_BASE + 32
    print("self-test OK")


def parse_waves(text):
    """Parse --waves comma-separated list into [32|64, ...], or raise ValueError."""
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--waves must not be empty")
    waves = []
    for tok in tokens:
        try:
            w = int(tok)
        except ValueError:
            raise ValueError(f"invalid wave size {tok!r}; expected 32 or 64") from None
        if w not in (32, 64):
            raise ValueError(f"unsupported wave size {w}; expected 32 or 64")
        waves.append(w)
    return waves


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
        except OSError as e:
            return f"cannot create directory for --json {path}: {e.strerror or e}"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        return f"cannot write --json {path}: {e.strerror or e}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="wave sizes to run (default: 32,64)")
    ap.add_argument("--json", dest="json_out", help="write full JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="check the Python model without a GPU")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        waves = parse_waves(args.waves)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        err = validate_json_path(args.json_out)
        if err:
            print(f"error: {err}", file=sys.stderr)
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
        res = probe_wave(oracle, asm, wave)
        out["waves"][str(wave)] = res
        if res["cases_ok"] != res["cases_total"]:
            mismatches += 1
        print(f"wave{wave} ({oracle.arch}): {res['cases_ok']}/{res['cases_total']} ds_permute_b32 cases exact")
    if args.json_out is not None:
        err = write_json_report(args.json_out, out)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
