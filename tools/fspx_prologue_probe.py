r"""TEST-849: RDNA2 hardware probe for the 36-byte FSPX shader prologue.

FSPX compute payloads (docs/fspx.md) start with a 36-byte (9 x u32) prologue of three 12-byte VOP3 words with opcode
0x180 (bits 31..16 = 0xD580) plus literals. llvm-mc rejects them, so docs/fspx.md inferred that they are no-ops.

This tool tests synthetic 36-byte prologues of that form directly on an AMD GPU through tools/hw_oracle (wave32 and
wave64) and checks that VGPRs, SGPRs, M0, VCC, EXEC, SCC, multi-lane state under a partial EXEC mask, and PC
advancement (+36 bytes) are unaffected, with negative controls that prove the harness catches state changes.

Usage:
    python -I tools/fspx_prologue_probe.py                 # run on AMD GPU (wave32 + wave64)
    python -I tools/fspx_prologue_probe.py --oracle hw     # same
    python -I tools/fspx_prologue_probe.py --self-test     # offline check (no GPU needed)

Exit codes:
    0   all prologue checks passed and all negative controls detected their clobbers
    1   a prologue modified state, failed to assemble/run, or a negative control went undetected
    2   bad CLI arguments
    77  hardware oracle unavailable on this machine
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hw_oracle import asm, runner  # noqa: E402

WORD0_FIXED_MASK = 0xFFFF0000
WORD0_FIXED_VAL = 0xD5800000
WORD1_FIXED_MASK = 0xFFFE0100
WORD1_FIXED_VAL = 0x03FE0100
TAIL_WORDS = (0xD5800000, 0x03FE0100, 0x00000AC0)

VGPR_SENTINELS = (
    0xA5A5A5A5,  # v8
    0x5A5A5A5A,  # v9
    0x13572468,  # v10
    0xDEADBEEF,  # v11
    0xCAFEBABE,  # v12
    0x01020304,  # v13
    0xFEDCBA98,  # v14
    0x778899AA,  # v15 (toggled by 1 after the prologue to prove execution continues at +36)
)
V0_SENTINEL = 0xF00D0000
V1_SENTINEL = 0x11223344
SGPR_SENTINELS = tuple(0x40010000 + r for r in range(4, 16))  # s4..s15
M0_SENTINEL = 0x76543210
VCC_LO_SENTINEL = 0xA5A55A5A
VCC_HI_SENTINEL = 0x3C3CC3C3
EXEC_LO_SENTINEL = 0x0F0F0F0F  # bit 0 is set so lane 0 remains active
EXEC_HI_SENTINEL = 0xF0F0F0F1
ML_TAG = 0xA5A5A500
ML_EXEC_LO = 0x55555555
ML_EXEC_HI = 0xAAAAAAAA


def build_prologue(w0_lo, s1_0, s0_0, lit0, w3_lo, s1_1, s0_1, lit1):
    """Build a synthetic 9-dword FSPX prologue matching the VOP3 opcode-0x180 structural masks."""
    return (
        WORD0_FIXED_VAL | (w0_lo & 0xFFFF),
        WORD1_FIXED_VAL | ((s1_0 & 0xFF) << 9) | (s0_0 & 0xFF),
        lit0 & 0xFFFFFFFF,
        WORD0_FIXED_VAL | (w3_lo & 0xFFFF),
        WORD1_FIXED_VAL | ((s1_1 & 0xFF) << 9) | (s0_1 & 0xFF),
        lit1 & 0xFFFFFFFF,
        *TAIL_WORDS,
    )


def prologue_asm_lines(words):
    """Render 9 u32 prologue words as raw assembler `.long` directives."""
    if len(words) != 9:
        raise ValueError(f"expected 9 prologue dwords, got {len(words)}")
    return [f".long 0x{w & 0xFFFFFFFF:08x}" for w in words]


# Synthetic-only patterns covering all zero, all high, targeted tested VGPRs/SGPRs, and alternating bit patterns.
PROLOGUES = {
    "zero": build_prologue(0x0000, 0x00, 0x00, 0x00000000, 0x0000, 0x00, 0x00, 0x00000000),
    "all_ones": build_prologue(0xFFFF, 0xFF, 0xFF, 0xFFFFFFFF, 0xFFFF, 0xFF, 0xFF, 0xFFFFFFFF),
    "target_vgpr_ctrl": build_prologue(0x8008, 0x88, 0x09, 0x12345678, 0x810A, 0x6A, 0x7C, 0x9ABCDEF0),
    "target_v1_v12": build_prologue(0x0001, 0x0C, 0x0B, 0xCAFEBABE, 0x000C, 0x01, 0x0F, 0x01020304),
    "alt_bits": build_prologue(0xA55A, 0xAA, 0x55, 0x55AA55AA, 0x5AA5, 0x55, 0xAA, 0xA5A55A5A),
}


def make_vgpr_kernel(mid_lines, wave):
    """Check v0, v1, v8..v15 preservation and post-prologue execution at +36 bytes."""
    lines = [
        f"v_mov_b32 v0, 0x{V0_SENTINEL:08x}",
        f"v_mov_b32 v1, 0x{V1_SENTINEL:08x}",
        *(f"v_mov_b32 v{8 + i}, 0x{val:08x}" for i, val in enumerate(VGPR_SENTINELS)),
        *mid_lines,
        "v_xor_b32 v15, v15, 1",
        "v_mov_b32 v2, v0",
        "v_mov_b32 v0, 0",
        "global_store_b128 v0, v[8:11], s[2:3]",
        "v_mov_b32 v8, v12",
        "v_mov_b32 v9, v13",
        "v_mov_b32 v10, v14",
        "v_mov_b32 v11, v15",
        "global_store_b128 v0, v[8:11], s[2:3] offset:16",
        "v_mov_b32 v8, v2",
        "v_mov_b32 v9, v1",
        "global_store_b32 v0, v8, s[2:3] offset:32",
        "global_store_b32 v0, v9, s[2:3] offset:36",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    expected = [*VGPR_SENTINELS[:-1], VGPR_SENTINELS[-1] ^ 1, V0_SENTINEL, V1_SENTINEL]
    return asm.Kernel(lines, wave=wave, threads=1, words=10, raw=True), expected


def make_sgpr_scc_kernel(mid_lines, wave, scc_init):
    """Check s4..s15, s[0:1] equality, and SCC (0 or 1) preservation across mid_lines."""
    cmp_line = "s_cmp_eq_u32 1, 1" if scc_init else "s_cmp_eq_u32 0, 1"
    lines = [
        *(f"s_mov_b32 s{4 + i}, 0x{val:08x}" for i, val in enumerate(SGPR_SENTINELS)),
        "s_mov_b64 s[18:19], s[0:1]",
        cmp_line,
        *mid_lines,
        "s_cselect_b32 s16, 1, 0",
        "s_xor_b32 s20, s0, s18",
        "s_xor_b32 s21, s1, s19",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v8, s4", "v_mov_b32 v9, s5", "v_mov_b32 v10, s6", "v_mov_b32 v11, s7",
        "global_store_b128 v0, v[8:11], s[2:3]",
        "v_mov_b32 v8, s8", "v_mov_b32 v9, s9", "v_mov_b32 v10, s10", "v_mov_b32 v11, s11",
        "global_store_b128 v0, v[8:11], s[2:3] offset:16",
        "v_mov_b32 v8, s12", "v_mov_b32 v9, s13", "v_mov_b32 v10, s14", "v_mov_b32 v11, s15",
        "global_store_b128 v0, v[8:11], s[2:3] offset:32",
        "v_mov_b32 v8, s16", "v_mov_b32 v9, s20", "v_mov_b32 v10, s21", "v_mov_b32 v11, 0",
        "global_store_b128 v0, v[8:11], s[2:3] offset:48",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    expected = [*SGPR_SENTINELS, 1 if scc_init else 0, 0, 0]
    return asm.Kernel(lines, wave=wave, threads=1, words=15, raw=True), expected


def make_ctrl_kernel(mid_lines, wave):
    """Check M0, VCC (lo/hi), and EXEC (lo/hi) preservation across mid_lines."""
    lines = [
        f"s_mov_b32 m0, 0x{M0_SENTINEL:08x}",
        f"s_mov_b32 vcc_lo, 0x{VCC_LO_SENTINEL:08x}",
        f"s_mov_b32 vcc_hi, 0x{VCC_HI_SENTINEL:08x}",
        f"s_mov_b32 exec_lo, 0x{EXEC_LO_SENTINEL:08x}",
        f"s_mov_b32 exec_hi, 0x{EXEC_HI_SENTINEL:08x}",
        *mid_lines,
        "s_mov_b32 s4, m0",
        "s_mov_b32 s5, vcc_lo",
        "s_mov_b32 s6, vcc_hi",
        "s_mov_b32 s7, exec_lo",
        "s_mov_b32 s8, exec_hi",
        "s_mov_b32 exec_lo, -1",
        "s_mov_b32 exec_hi, -1",
        "v_mov_b32 v0, 0",
        "v_mov_b32 v8, s4", "v_mov_b32 v9, s5", "v_mov_b32 v10, s6", "v_mov_b32 v11, s7",
        "global_store_b128 v0, v[8:11], s[2:3]",
        "v_mov_b32 v8, s8",
        "global_store_b32 v0, v8, s[2:3] offset:16",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    expected = [M0_SENTINEL, VCC_LO_SENTINEL, VCC_HI_SENTINEL, EXEC_LO_SENTINEL, EXEC_HI_SENTINEL]
    return asm.Kernel(lines, wave=wave, threads=1, words=5, raw=True), expected


def make_multilane_kernel(mid_lines, wave):
    """Check all 64 lanes under a split active/inactive EXEC mask."""
    lines = [
        "v_mov_b32 v1, v0",
        f"v_xor_b32 v8, v0, 0x{ML_TAG:08x}",
        f"s_mov_b32 exec_lo, 0x{ML_EXEC_LO:08x}",
        f"s_mov_b32 exec_hi, 0x{ML_EXEC_HI:08x}",
        *mid_lines,
        "s_mov_b32 exec_lo, -1",
        "s_mov_b32 exec_hi, -1",
        "v_xor_b32 v9, v8, v1",
        "v_lshlrev_b32 v2, 2, v1",
        "global_store_b32 v2, v9, s[2:3]",
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]
    return asm.Kernel(lines, wave=wave, threads=64, words=64, raw=True), [ML_TAG] * 64


def _negative_control_specs():
    """Return (name, builder, mid_lines, check_diff_fn) tuples for known clobbers."""
    return (
        ("neg_vgpr", make_vgpr_kernel, ("v_xor_b32 v8, v8, 1",), lambda got, exp: got[0] != exp[0]),
        ("neg_sgpr_scc", lambda mid, w: make_sgpr_scc_kernel(mid, w, 0), ("s_xor_b32 s8, s8, 1",),
         lambda got, exp: got[4] != exp[4] and got[12] != exp[12]),
        ("neg_ctrl", make_ctrl_kernel,
         ("s_mov_b32 m0, 0x11111111", "s_xor_b32 vcc_lo, vcc_lo, 1", "s_xor_b32 exec_lo, exec_lo, 2"),
         lambda got, exp: got[0] != exp[0] and got[1] != exp[1] and got[3] != exp[3]),
        ("neg_multilane", make_multilane_kernel, ("v_xor_b32 v8, v8, 1",),
         lambda got, exp: sum(a != b for a, b in zip(got, exp)) == 32),
    )


def build_probe_plan(waves=(32, 64)):
    """Return (positive_specs, negative_specs) with (tag, kernel, expected_or_checker)."""
    positives = []
    for name, words in PROLOGUES.items():
        mid = prologue_asm_lines(words)
        for wave in waves:
            k_vgpr, exp_vgpr = make_vgpr_kernel(mid, wave)
            positives.append((f"vgpr/{name}/w{wave}", k_vgpr, exp_vgpr))
            for scc in (0, 1):
                k_scc, exp_scc = make_sgpr_scc_kernel(mid, wave, scc)
                positives.append((f"sgpr_scc{scc}/{name}/w{wave}", k_scc, exp_scc))
            k_ctrl, exp_ctrl = make_ctrl_kernel(mid, wave)
            positives.append((f"ctrl/{name}/w{wave}", k_ctrl, exp_ctrl))
            k_ml, exp_ml = make_multilane_kernel(mid, wave)
            positives.append((f"multilane/{name}/w{wave}", k_ml, exp_ml))

    negatives = []
    for name, builder, mid, check_diff in _negative_control_specs():
        for wave in waves:
            k, exp = builder(mid, wave)
            negatives.append((f"{name}/w{wave}", k, exp, check_diff))
    return positives, negatives


def self_test():
    """Offline verification of prologue layout invariants, .long emission, and kernel generation."""
    for name, words in PROLOGUES.items():
        assert len(words) == 9, name
        assert (words[0] & WORD0_FIXED_MASK) == WORD0_FIXED_VAL, name
        assert (words[1] & WORD1_FIXED_MASK) == WORD1_FIXED_VAL, name
        assert (words[3] & WORD0_FIXED_MASK) == WORD0_FIXED_VAL, name
        assert (words[4] & WORD1_FIXED_MASK) == WORD1_FIXED_VAL, name
        assert tuple(words[6:]) == TAIL_WORDS, name
        lines = prologue_asm_lines(words)
        assert len(lines) == 9 and all(ln.startswith(".long 0x") for ln in lines), name

    positives, negatives = build_probe_plan()
    assert len(positives) == len(PROLOGUES) * 5 * 2
    assert len(negatives) == 8
    text32, names32, _ = asm.module_text([k for _, k, _ in positives if k.wave == 32], "gfx1030", wave=32)
    text64, names64, _ = asm.module_text([k for _, k, _ in positives if k.wave == 64], "gfx1030", wave=64)
    assert len(names32) == len(positives) // 2 and len(names64) == len(positives) // 2
    assert ".long 0xd580" in text32 and ".long 0x00000ac0" in text64
    print(f"self-test: {len(positives)} prologue cases and {len(negatives)} negative controls built cleanly")
    return 0


def run_hw(waves=(32, 64), arch=None):
    """Run all positive prologue checks and negative controls on the AMD GPU."""
    try:
        oracle = runner.open_oracle(arch=arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}", file=sys.stderr)
        return 77

    positives, negatives = build_probe_plan(waves=waves)
    all_kernels = [k for _, k, _ in positives] + [k for _, k, _, _ in negatives]
    try:
        raw_results = oracle.run(all_kernels, words=64)
        device = oracle.device
    finally:
        oracle.close()

    pos_results = raw_results[:len(positives)]
    neg_results = raw_results[len(positives):]

    failures = []
    for (tag, k, exp), got in zip(positives, pos_results):
        if isinstance(got, str):
            failures.append(f"{tag}: {got}")
        elif got[:k.words] != exp:
            failures.append(f"{tag}: mismatch (first diff at word "
                            f"{next(i for i, (a, b) in enumerate(zip(got[:k.words], exp)) if a != b)})")

    for (tag, k, exp, check_diff), got in zip(negatives, neg_results):
        if isinstance(got, str):
            failures.append(f"{tag}: {got}")
        elif not check_diff(got[:k.words], exp):
            failures.append(f"{tag}: negative control failed to detect injected change")

    print(device)
    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        print(f"FAIL: {len(failures)} failure(s) across {len(positives)} prologue checks and "
              f"{len(negatives)} negative controls")
        return 1

    print(f"PASS: {len(positives)}/{len(positives)} prologue checks preserved all state "
          f"(VGPRs, SGPRs, s[0:1], M0, VCC, EXEC, SCC=0/1, 64-lane masked EXEC, +36 PC); "
          f"{len(negatives)}/{len(negatives)} negative controls detected clobbers")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("hw",), default="hw")
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    waves = (32, 64) if args.wave == "both" else (int(args.wave),)
    return run_hw(waves=waves, arch=args.hw_arch)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
