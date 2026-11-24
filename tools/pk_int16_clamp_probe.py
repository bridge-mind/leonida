r"""Measure RDNA2 VOP3P clamp behavior on v_pk_mul_lo_u16 and v_pk_{max,min}_{u16,i16}. Standard library only.

`v_pk_add_{u16,i16}`, `v_pk_sub_{u16,i16}`, and `v_pk_mad_{u16,i16}` saturate per 16-bit half when `clamp` is set.
For `v_pk_mul_lo_u16`, `v_pk_max_u16`, `v_pk_min_u16`, `v_pk_max_i16`, and `v_pk_min_i16`, running this probe natively
on RDNA2 (`gfx1036`) across wave32 and wave64 verifies whether VOP3P `clamp` is a hardware no-op (notably whether
overflowing `v_pk_mul_lo_u16 clamp` wraps modulo 2^16 rather than saturating to 0xFFFF), alongside `v_pk_add_u16` as a
positive saturation control.

Usage:
    python -I tools/pk_int16_clamp_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/pk_int16_clamp_probe.py --self-test

Exit 0 when all hardware results match the model across requested ops/waves (with wave32 and wave64 agreeing bit-for-bit,
and plain vs clamp matching bit-for-bit on all five target ops while differing on overflowing v_pk_add_u16), 1 on a
model mismatch or wave32/wave64 difference or control failure, 2 on a harness/CLI error, and 77 when the hardware
oracle is unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
from dataclasses import dataclass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TARGET_OPS: tuple[str, ...] = (
    "v_pk_mul_lo_u16",
    "v_pk_max_u16",
    "v_pk_min_u16",
    "v_pk_max_i16",
    "v_pk_min_i16",
)
CONTROL_OPS: tuple[str, ...] = ("v_pk_add_u16",)
OPS: tuple[str, ...] = TARGET_OPS + CONTROL_OPS
CLAMP_MODES: tuple[bool, ...] = (False, True)


@dataclass(frozen=True)
class Case:
    name: str
    src0: int
    src1: int
    src0_asm: str | None = None
    src1_asm: str | None = None
    sel: tuple[int, int] = (0, 0)
    selhi: tuple[int, int] = (1, 1)
    alias: str | None = None  # "src0" | "src1" | None
    sgpr0: bool = False


CASES: tuple[Case, ...] = (
    Case("zero_zero", 0x00000000, 0x00000000),
    Case("u16_max_x_max", 0xFFFFFFFF, 0xFFFFFFFF),
    Case("u16_max_pair_2", 0xFFFFFFFF, 0x00020002),
    Case("u16_hi_overflow_lo_exact", 0xFFFF0003, 0x00030005),
    Case("i16_min_edges", 0x80008001, 0x7FFF8000),
    Case("i16_signed_boundaries", 0x80007FFF, 0xFFFF0001),
    Case("i16_neg_vs_pos", 0xFFFE8005, 0x00027FFE),
    Case("mixed_halves", 0x12345678, 0x9ABCDEF0),
    Case("pow2_overflow_wrap", 0x02000100, 0x01000200),
    Case("carry_boundary", 0x01010100, 0x01000101),
    Case("u16_diff_overflow", 0xFFFF00FF, 0xFFFF0101),
    Case("half_max_minus1", 0x7FFE8002, 0x7FFF8001),
    Case("sel_swap_both", 0x01000200, 0x03000100, sel=(1, 1), selhi=(0, 0)),
    Case("sel_cross_01_10", 0x80007FFF, 0x7FFE8001, sel=(0, 1), selhi=(1, 0)),
    Case("sel_broadcast_lo", 0x02000300, 0x04000500, sel=(0, 0), selhi=(0, 0)),
    Case("sel_broadcast_hi", 0xFFFF0002, 0x0003FFFF, sel=(1, 1), selhi=(1, 1)),
    Case("sel_mixed_sign", 0xFF800080, 0x0090FF70, sel=(1, 0), selhi=(0, 1)),
    Case("inline_neg1_vgpr", 0xFFFFFFFF, 0x02000002, src0_asm="-1", sel=(0, 0), selhi=(1, 1)),
    Case("inline_neg16_vgpr", 0xFFFFFFF0, 0x00108000, src0_asm="-16", sel=(1, 0), selhi=(0, 1)),
    Case("inline_5_vgpr", 0x00000005, 0x10004000, src0_asm="5", sel=(0, 0), selhi=(1, 1)),
    Case("inline_64_vgpr", 0x00000040, 0x08000400, src0_asm="64", sel=(0, 0), selhi=(1, 1)),
    Case("vgpr_literal_overflow", 0x02000100, 0x02000400, src1_asm="0x02000400", sel=(0, 0), selhi=(1, 1)),
    Case("literal_vgpr_mixed", 0x80017FFF, 0xFFFE0003, src0_asm="0x80017fff", sel=(1, 0), selhi=(0, 1)),
    Case("sgpr_vgpr_mixed", 0x0300FFFE, 0x02000100, sel=(0, 1), selhi=(1, 0), sgpr0=True),
    Case("dst_alias_src0", 0x02007FFF, 0x01008000, alias="src0"),
    Case("dst_alias_src1", 0xFFFF0003, 0x0002FFFF, alias="src1"),
)


def _s16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if (x & 0x8000) else x


def expected_pk_int16(
    op: str,
    src0: int,
    src1: int,
    *,
    sel: tuple[int, int] = (0, 0),
    selhi: tuple[int, int] = (1, 1),
    clamp: bool = False,
) -> int:
    """Reference 32-bit packed result for VOP3P int16 op; clamp saturates add only."""
    if op not in OPS:
        raise ValueError(f"unsupported op: {op}")
    kind = op[len("v_pk_"):].rsplit("_", 1)[0]
    signed = op.endswith("i16")
    out = 0
    for h, sl in enumerate((sel, selhi)):
        a = (src0 >> (16 * sl[0])) & 0xFFFF
        b = (src1 >> (16 * sl[1])) & 0xFFFF
        if signed:
            a, b = _s16(a), _s16(b)
        if kind == "add":
            r = a + b
        elif kind == "mul_lo":
            r = a * b
        elif kind == "max":
            r = max(a, b)
        elif kind == "min":
            r = min(a, b)
        else:
            raise ValueError(f"unsupported kind: {kind}")
        if clamp and kind == "add":
            r = max(-32768, min(32767, r)) if signed else max(0, min(0xFFFF, r))
        out |= (r & 0xFFFF) << (16 * h)
    return out


def case_instruction(op: str, c: Case, clamp: bool) -> list[str]:
    """Emit raw GFX10 setup + VOP3P instruction lines storing result in v2."""
    lines: list[str] = []
    if c.alias == "src0":
        lines.append(f"v_mov_b32_e32 v2, {c.src0:#010x}")
        lines.append(f"v_mov_b32_e32 v1, {c.src1:#010x}")
        s0_op, s1_op = "v2", "v1"
    elif c.alias == "src1":
        lines.append(f"v_mov_b32_e32 v0, {c.src0:#010x}")
        lines.append(f"v_mov_b32_e32 v2, {c.src1:#010x}")
        s0_op, s1_op = "v0", "v2"
    else:
        if c.sgpr0:
            lines.append(f"s_mov_b32 s4, {c.src0:#010x}")
            s0_op = "s4"
        elif c.src0_asm is not None:
            s0_op = c.src0_asm
        else:
            lines.append(f"v_mov_b32_e32 v0, {c.src0:#010x}")
            s0_op = "v0"
        if c.src1_asm is not None:
            s1_op = c.src1_asm
        else:
            lines.append(f"v_mov_b32_e32 v1, {c.src1:#010x}")
            s1_op = "v1"
        lines.append("v_mov_b32_e32 v2, 0xaaaaaaaa")

    mods: list[str] = []
    if c.sel != (0, 0):
        mods.append(f"op_sel:[{c.sel[0]},{c.sel[1]}]")
    if c.selhi != (1, 1):
        mods.append(f"op_sel_hi:[{c.selhi[0]},{c.selhi[1]}]")
    if clamp:
        mods.append("clamp")
    suffix = (" " + " ".join(mods)) if mods else ""
    lines.append(f"{op} v2, {s0_op}, {s1_op}{suffix}")
    return lines


def kernel_lines(op: str, clamp: bool) -> list[str]:
    """Raw GFX10 kernel running every case in CASES sequentially on lane 0."""
    lines: list[str] = []
    for idx, c in enumerate(CASES):
        lines.extend(case_instruction(op, c, clamp))
        lines.append(f"v_mov_b32_e32 v3, {idx * 4}")
        lines.append("global_store_dword v3, v2, s[2:3]")
    lines.append("s_waitcnt vmcnt(0)")
    lines.append("s_endpgm")
    return lines


def build_jobs(ops: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels, meta = [], []
    for wave in waves:
        for op in ops:
            for clamp in CLAMP_MODES:
                kernels.append(asm.Kernel(kernel_lines(op, clamp), wave=wave, threads=1, words=len(CASES), raw=True))
                meta.append((wave, op, clamp))
    return kernels, meta


def evaluate_results(ops: list[str], waves: list[int], meta, raw_results) -> dict:
    """Compare hardware outputs against expected_pk_int16, plain vs clamp parity, and wave32 vs wave64 parity."""
    by_key: dict[tuple[int, str, bool], list[int]] = {}
    harness_errors: list[dict] = []
    for (wave, op, clamp), res in zip(meta, raw_results):
        if isinstance(res, str):
            harness_errors.append({"wave": wave, "op": op, "clamp": clamp, "error": res})
            continue
        if len(res) != len(CASES):
            harness_errors.append({
                "wave": wave,
                "op": op,
                "clamp": clamp,
                "error": f"expected {len(CASES)} words, got {len(res)}",
            })
            continue
        by_key[(wave, op, clamp)] = [w & 0xFFFFFFFF for w in res]

    if harness_errors:
        return {
            "total_cases": len(meta) * len(CASES),
            "mismatches": 0,
            "wave_diffs": 0,
            "target_clamp_diffs": 0,
            "control_clamp_diffs": 0,
            "harness_errors": harness_errors,
            "first_mismatches": [],
            "first_wave_diffs": [],
            "first_target_clamp_diffs": [],
        }

    mismatches = 0
    first_mismatches: list[dict] = []
    for wave in waves:
        for op in ops:
            for clamp in CLAMP_MODES:
                got_words = by_key[(wave, op, clamp)]
                for c, got in zip(CASES, got_words):
                    want = expected_pk_int16(op, c.src0, c.src1, sel=c.sel, selhi=c.selhi, clamp=clamp)
                    if got != want:
                        mismatches += 1
                        if len(first_mismatches) < 16:
                            first_mismatches.append({
                                "wave": wave,
                                "op": op,
                                "clamp": clamp,
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
                for clamp in CLAMP_MODES:
                    for c, a_w, b_w in zip(CASES, by_key[(w0, op, clamp)], by_key[(w, op, clamp)]):
                        if a_w != b_w:
                            wave_diffs += 1
                            if len(first_wave_diffs) < 16:
                                first_wave_diffs.append({
                                    "op": op,
                                    "clamp": clamp,
                                    "case": c.name,
                                    f"w{w0}": f"0x{a_w:08x}",
                                    f"w{w}": f"0x{b_w:08x}",
                                })

    target_clamp_diffs = 0
    first_target_clamp_diffs: list[dict] = []
    control_clamp_diffs = 0
    for wave in waves:
        for op in ops:
            plain_words = by_key[(wave, op, False)]
            clamp_words = by_key[(wave, op, True)]
            for c, pw, cw in zip(CASES, plain_words, clamp_words):
                if op in TARGET_OPS and pw != cw:
                    target_clamp_diffs += 1
                    if len(first_target_clamp_diffs) < 16:
                        first_target_clamp_diffs.append({
                            "wave": wave,
                            "op": op,
                            "case": c.name,
                            "plain": f"0x{pw:08x}",
                            "clamp": f"0x{cw:08x}",
                        })
                elif op in CONTROL_OPS and pw != cw:
                    control_clamp_diffs += 1

    return {
        "total_cases": len(waves) * len(ops) * len(CLAMP_MODES) * len(CASES),
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "target_clamp_diffs": target_clamp_diffs,
        "control_clamp_diffs": control_clamp_diffs,
        "harness_errors": [],
        "first_mismatches": first_mismatches,
        "first_wave_diffs": first_wave_diffs,
        "first_target_clamp_diffs": first_target_clamp_diffs,
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
    """Verify the reference model and kernel generator without GPU or clang."""
    # v_pk_mul_lo_u16 wraps modulo 2^16 under clamp rather than saturating to 0xFFFF
    assert expected_pk_int16("v_pk_mul_lo_u16", 0x02000100, 0x01000200, clamp=False) == 0x00000000
    assert expected_pk_int16("v_pk_mul_lo_u16", 0x02000100, 0x01000200, clamp=True) == 0x00000000
    assert expected_pk_int16("v_pk_mul_lo_u16", 0xFFFFFFFF, 0xFFFFFFFF, clamp=True) == 0x00010001
    assert expected_pk_int16("v_pk_mul_lo_u16", 0xFFFFFFFF, 0x00020002, clamp=True) == 0xFFFEFFFE
    # v_pk_add_u16 control saturates under clamp
    assert expected_pk_int16("v_pk_add_u16", 0xFFFFFFFF, 0x00020002, clamp=False) == 0x00010001
    assert expected_pk_int16("v_pk_add_u16", 0xFFFFFFFF, 0x00020002, clamp=True) == 0xFFFFFFFF
    # v_pk_{max,min}_{u16,i16} ignore clamp and preserve signed vs unsigned comparison
    assert expected_pk_int16("v_pk_max_u16", 0x80007FFF, 0x7FFF8000, clamp=True) == 0x80008000
    assert expected_pk_int16("v_pk_min_u16", 0x80007FFF, 0x7FFF8000, clamp=True) == 0x7FFF7FFF
    assert expected_pk_int16("v_pk_max_i16", 0x80007FFF, 0x7FFF8000, clamp=True) == 0x7FFF7FFF
    assert expected_pk_int16("v_pk_min_i16", 0x80007FFF, 0x7FFF8000, clamp=True) == 0x80008000
    # op_sel and op_sel_hi selection on constants and registers
    assert expected_pk_int16(
        "v_pk_mul_lo_u16", 0x80017FFF, 0xFFFE0003, sel=(1, 0), selhi=(0, 1), clamp=True
    ) == 0x00028003
    # Kernel assembly sanity
    kl = kernel_lines("v_pk_mul_lo_u16", True)
    assert kl[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
    assert "v_pk_mul_lo_u16 v2, v0, v1 clamp" in kl
    assert "v_pk_mul_lo_u16 v2, v0, 0x02000400 clamp" in kl
    assert "v_pk_mul_lo_u16 v2, -16, v1 op_sel:[1,0] op_sel_hi:[0,1] clamp" in kl
    print(f"self-test ok ({len(CASES)} cases, {len(OPS) * len(CLAMP_MODES) * len(CASES)} checks/wave)")
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
        "clamp_modes": ["plain", "clamp"],
        "cases": len(CASES),
        **summary,
    }
    if a.json and not write_report(a.json, report):
        return 2

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: wave{err['wave']} {err['op']} clamp={err['clamp']}: {err['error']}")
        return 2

    print(
        f"{summary['total_cases']} evaluations across {len(waves)} wave(s) x {len(ops)} op(s) x 2 clamp modes: "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}, "
        f"target_clamp_diffs={summary['target_clamp_diffs']}, control_clamp_diffs={summary['control_clamp_diffs']}"
    )
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH wave{m['wave']} {m['op']} clamp={m['clamp']} {m['case']}: got {m['got']}, want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['op']} clamp={d['clamp']} {d['case']}: {d}")
    for d in summary["first_target_clamp_diffs"]:
        print(f"  CLAMP DIFF wave{d['wave']} {d['op']} {d['case']}: plain={d['plain']} clamp={d['clamp']}")

    control_missing = "v_pk_add_u16" in ops and summary["control_clamp_diffs"] == 0
    if summary["mismatches"] or summary["wave_diffs"] or summary["target_clamp_diffs"] or control_missing:
        if control_missing:
            print("  CONTROL ERROR: v_pk_add_u16 clamp did not differ from plain on overflowing inputs")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
