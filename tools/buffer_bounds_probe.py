r"""SHADER-341: native GFX10 MUBUF `OOB_SELECT` bounds probe.

Runs hand-written one-lane raw kernels through `tools/hw_oracle` and compares a
4-byte `buffer_load_dword` against the two published bounds columns:

  * Mesa's GFX10 column (`.ac_descriptors.c`), which checks `offset >= stride`
    in structured-with-offset mode and `offset >= NUM_RECORDS` in raw mode;
  * the RDNA2 measurement in docs/buffer_bounds.md (Mesa's GFX11 column), which
    checks the *end* of the dword and includes `soffset` in raw mode.

No game data is involved: every V#, buffer and pattern is built by this file.

Usage:
    python -I tools/buffer_bounds_probe.py --self-test
    python -I tools/buffer_bounds_probe.py [--waves 32,64] [--json report.json]

Exit codes:
    0   every case matches the measured RDNA2/GFX11 column
    1   at least one case disagrees (or a wave32/wave64 result differs)
    2   bad arguments (including a --json path this tool cannot write) or harness error
    77  hardware oracle unavailable or not a GFX10 device
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Raw output V# in s[0:3]; the probed data starts 0x100 bytes into the buffer.
OUT_V_W3 = 0x31016FAC
V_W3_BASE = 0x01000FAC
DATA_OFF = 0x100
DATA_DWORDS = 64
SEED_BASE = 0xA0000000

# name, mode, stride, num_records, index, offset, soffset, idxen, offen.
# The first four cases are the ones where the two columns disagree.
CASES = [
    ("mode0_stride16_off13", 0, 16, 18, 0, 13, 0, False, True),
    ("mode0_stride16_off12", 0, 16, 18, 0, 12, 0, False, True),
    ("mode0_index_at_nr", 0, 16, 2, 2, 0, 0, True, False),
    ("mode1_index_at_nr", 1, 16, 2, 2, 0, 0, True, False),
    ("mode1_index_off48", 1, 16, 18, 1, 48, 0, True, True),
    ("mode2_nr0", 2, 16, 0, 0, 0, 0, False, False),
    ("mode2_nr18", 2, 16, 18, 0, 0, 0, False, False),
    ("mode3_nr18_off16", 3, 0, 18, 0, 16, 0, False, True),
    ("mode3_nr18_off13", 3, 0, 18, 0, 13, 0, False, True),
    ("mode3_nr3_off0", 3, 0, 3, 0, 0, 0, False, True),
    ("mode3_soff8_off8_nr18", 3, 0, 18, 0, 8, 8, False, True),
    ("mode3_stride16_index4", 3, 16, 18, 4, 0, 0, True, False),
]


def oob_gfx10(case):
    """Out-of-range according to Mesa's GFX10 column (docs/buffer_bounds.md)."""
    _name, mode, stride, nr, idx, off, _soff, _idxen, _offen = case
    if mode == 0:
        return idx >= nr or off >= stride
    if mode == 1:
        return idx >= nr
    if mode == 2:
        return nr == 0
    if mode == 3:
        return off >= nr
    raise ValueError(f"bad OOB_SELECT mode {mode}")


def oob_gfx11(case):
    """Out-of-range according to the measured RDNA2 / Mesa GFX11 column."""
    _name, mode, stride, nr, idx, off, soff, _idxen, _offen = case
    if mode == 0:
        return idx >= nr or off + 4 > stride
    if mode == 1:
        return idx >= nr
    if mode == 2:
        return nr == 0
    if mode == 3:
        return off + soff + 4 > nr
    raise ValueError(f"bad OOB_SELECT mode {mode}")


def kernel_lines(case, wave):
    """One self-contained raw kernel: seed the data, load once, store the word."""
    name, mode, stride, nr, idx, off, soff, idxen, offen = case
    lbl, done = f".Lseed_{name}", f".Ldone_{name}"
    if wave == 32:
        save = ["s_mov_b32 s28, exec_lo", "v_cmpx_eq_u32_e32 0, v0"]
        restore = ["s_mov_b32 exec_lo, s28"]
    else:
        save = ["s_mov_b64 s[28:29], exec", "v_cmpx_eq_u32_e32 0, v0"]
        restore = ["s_mov_b64 exec, s[28:29]"]
    return [
        "s_mov_b32 s0, s2",
        "s_and_b32 s1, s3, 0xffff",
        "s_mov_b32 s2, 0xffffffff",
        f"s_mov_b32 s3, {OUT_V_W3:#x}",
        f"s_add_u32 s16, s0, {DATA_OFF:#x}",
        "s_addc_u32 s17, s1, 0",
        "s_and_b32 s17, s17, 0xffff",
        *save,
        "s_mov_b32 s20, 0",
        f"{lbl}:",
        f"s_cmp_ge_u32 s20, {DATA_DWORDS * 4:#x}",
        f"s_cbranch_scc1 {done}",
        f"s_add_u32 s21, s20, {SEED_BASE:#x}",
        "v_mov_b32 v30, s21",
        f"s_add_u32 s22, s20, {DATA_OFF:#x}",
        "buffer_store_dword v30, off, s[0:3], s22",
        "s_add_u32 s20, s20, 4",
        f"s_branch {lbl}",
        f"{done}:",
        "s_waitcnt vmcnt(0)",
        *restore,
        "s_mov_b32 s4, s16",
        f"s_or_b32 s5, s17, {stride << 16:#x}",
        f"s_mov_b32 s6, {nr:#x}",
        f"s_mov_b32 s7, {V_W3_BASE | (22 << 12) | (mode << 28):#x}",
        f"s_mov_b32 s10, {soff:#x}",
        f"v_mov_b32 v1, {idx:#x}",
        f"v_mov_b32 v2, {off:#x}",
        f"buffer_load_dword v8, {_addr(case)}, s[4:7], {_soff(case)}{_mods(case)}",
        "s_waitcnt vmcnt(0)",
        "buffer_store_dword v8, off, s[0:3], 0",
        "s_endpgm",
    ]


def _addr(case):
    _n, _m, _s, _nr, _i, _o, _so, idxen, offen = case
    if idxen and offen:
        return "v[1:2]"
    if idxen:
        return "v1"
    if offen:
        return "v2"
    return "off"


def _soff(case):
    return "s10" if case[6] else "0"


def _mods(case):
    idxen, offen = case[7], case[8]
    if idxen and offen:
        return " idxen offen"
    if idxen:
        return " idxen"
    if offen:
        return " offen"
    return ""


def self_test():
    assert len({c[0] for c in CASES}) == len(CASES), "duplicate case names"
    for c in CASES:
        assert 0 <= c[1] <= 3 and c[2] >= 0 and c[3] >= 0, c
        oob_gfx10(c)
        oob_gfx11(c)
        for wave in (32, 64):
            lines = kernel_lines(c, wave)
            assert any("buffer_load_dword v8," in ln for ln in lines), c
            assert lines[-1] == "s_endpgm", c
    # The four witness cases must separate the two columns.
    witnesses = [c for c in CASES if oob_gfx10(c) != oob_gfx11(c)]
    assert len(witnesses) >= 4, len(witnesses)
    assert oob_gfx10(CASES[10]) is False and oob_gfx11(CASES[10]) is True  # soffset in raw mode
    print("self-test OK")


def probe(oracle, asm, waves):
    out = {"arch": oracle.arch, "device": str(oracle.device), "waves": {}}
    for wave in waves:
        kernels = [asm.Kernel(kernel_lines(c, wave), wave=wave, threads=1,
                              words=4, raw=True) for c in CASES]
        results = oracle.run(kernels, words=4)
        rows = []
        for case, result in zip(CASES, results):
            word = result[0] if isinstance(result, list) else result
            oob = word == 0
            rows.append({"name": case[0], "got": word & 0xFFFFFFFF, "oob": oob,
                         "gfx10_oob": oob_gfx10(case), "gfx11_oob": oob_gfx11(case)})
        out["waves"][str(wave)] = rows
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="wave sizes to run (default: 32,64)")
    ap.add_argument("--json", dest="json_out", help="write the full report to this path")
    ap.add_argument("--self-test", action="store_true", help="check the models without a GPU")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    waves = [int(w) for w in args.waves.split(",") if w.strip()]
    if not waves or any(w not in (32, 64) for w in waves):
        print("--waves must be a non-empty list of 32 and/or 64", file=sys.stderr)
        return 2
    if args.json_out:
        # The report is written after the sweep, but a path that cannot be written must fail before the GPU work.
        if os.path.isdir(args.json_out):
            print(f"ERROR: --json {args.json_out} is a directory", file=sys.stderr)
            return 2
        folder = os.path.dirname(os.path.abspath(args.json_out))
        if not os.path.isdir(folder):
            print(f"ERROR: --json {args.json_out}: its folder {folder} does not exist", file=sys.stderr)
            return 2
    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return 77
    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"SKIP: {oracle.arch} is not GFX10 (RDNA1/RDNA2)", file=sys.stderr)
            return 77
        report = probe(oracle, asm, waves)
    finally:
        oracle.close()
    gfx10_diff = gfx11_diff = wave_diff = 0
    for wave, rows in report["waves"].items():
        for row in rows:
            gfx10_diff += row["oob"] != row["gfx10_oob"]
            gfx11_diff += row["oob"] != row["gfx11_oob"]
            print(f"wave{wave} {row['name']:24s} got={'OOB' if row['oob'] else 'data'} "
                  f"gfx10={'OOB' if row['gfx10_oob'] else 'data'} "
                  f"gfx11={'OOB' if row['gfx11_oob'] else 'data'} 0x{row['got']:08x}")
    if "32" in report["waves"] and "64" in report["waves"]:
        a = {r["name"]: r["oob"] for r in report["waves"]["32"]}
        b = {r["name"]: r["oob"] for r in report["waves"]["64"]}
        wave_diff = sum(a[k] != b.get(k) for k in a)
    print(f"{oracle.arch}: {sum(len(v) for v in report['waves'].values())} cases, "
          f"{gfx11_diff} disagree with the measured RDNA2/GFX11 column, "
          f"{gfx10_diff} disagree with Mesa's GFX10 column, "
          f"{wave_diff} wave32/wave64 differences")
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=2)
        except OSError as exc:
            print(f"ERROR: cannot write {args.json_out}: {getattr(exc, 'strerror', None) or exc}",
                  file=sys.stderr)
            return 2
    return 1 if gfx11_diff or wave_diff else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
