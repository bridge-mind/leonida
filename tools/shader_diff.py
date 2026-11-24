r"""Differential oracle for the RDNA2 translator (standard library only).

Runs seeded straight-line ALU programs through two independent oracles and
compares their output words:
  * tools/cpu_ref.py      - the ISA model, executed per lane (lanes=1, lane=0)
  * rdna2spirv + tests/semantic/spirv_mini_interp.py - the translated SPIR-V

This is the runner designed in docs/shader_validation.md ("Harness design"):
it owns the program table, the seed and the comparison; cpu_ref stays a pure
interpreter and is never modified here.

Usage:
    python -I tools/shader_diff.py [--programs N] [--seed S] [--max-ops N]
    python -I tools/shader_diff.py --program-index I --seed S --max-ops N
    python -I tools/shader_diff.py --self-test     # harness canaries (must fail loudly if broken)
    python -I tools/shader_diff.py --list-ops

Exit code is 0 when every program matches, 1 on the first mismatch (with the
case name, seed, lane, register, expected/actual bits and the RDNA2
instruction that last wrote the value), 2 on harness errors or when every
program was skipped (nothing was compared). Skips are counted with reasons and never silent.
"""
import argparse
import gc
import os
import random
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import cpu_ref  # noqa: E402
import rdna2spirv  # noqa: E402
from spirv_mini_interp import UnsupportedConstruct, run_fragment  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Seeded scratch VGPRs (written first, read by generated ops).
SCRATCH = (8, 9, 10, 11, 12, 13, 14, 15)
EXPORT_REGS = (8, 9, 10, 11)

# Small integers: exact in f32 for add/sub/mul/min/max, so any mismatch is a
# real semantic divergence, not a rounding convention.
_SMALL_INTS = (0, 1, 2, 3, 5, 7, 15, 16, 31, 64, 127, 255)
_EXACT_FLOATS = (0.5, 1.5, 2.5, -1.5, 4.0)


def _u32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _is_nan_bits(word):
    exp = (word >> 23) & 0xFF
    return exp == 0xFF and (word & 0x7FFFFF) != 0


def _emit_perm(rng, dst, srcs):
    """Half the cases pin the selector so bytes 8-13 (sign bits, 0x00, 0xff) are not left to chance."""
    if rng.random() < 0.5:
        sel = 0
        for i in range(4):
            sel |= rng.randrange(0, 16) << (8 * i)
        return f"v_perm_b32 v{dst}, v{srcs[0]}, v{srcs[1]}, 0x{sel:08x}"
    return f"v_perm_b32 v{dst}, v{srcs[0]}, v{srcs[1]}, v{srcs[2]}"


def _op_templates():
    """(name, emitter) table owned by this runner. Emitters take
    (rng, dst, srcs) and return one assembly line (without leading tab)."""
    return [
        ("v_mov_b32", lambda r, d, s: f"v_mov_b32 v{d}, {r.choice(s)}"),
        ("v_add_f32", lambda r, d, s: f"v_add_f32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_sub_f32", lambda r, d, s: f"v_sub_f32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_f32", lambda r, d, s: f"v_mul_f32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_fma_f32", lambda r, d, s: f"v_fma_f32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_and_b32", lambda r, d, s: f"v_and_b32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_or_b32", lambda r, d, s: f"v_or_b32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_xor_b32", lambda r, d, s: f"v_xor_b32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_lshlrev_b32", lambda r, d, s: f"v_lshlrev_b32 v{d}, {r.randrange(0, 8)}, v{s[0]}"),
        ("v_lshrrev_b32", lambda r, d, s: f"v_lshrrev_b32 v{d}, {r.randrange(0, 8)}, v{s[0]}"),
        ("v_ashrrev_i32", lambda r, d, s: f"v_ashrrev_i32 v{d}, {r.randrange(0, 8)}, v{s[0]}"),
        ("v_add_nc_u32", lambda r, d, s: f"v_add_nc_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_sub_nc_u32", lambda r, d, s: f"v_sub_nc_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_subrev_nc_u32", lambda r, d, s: f"v_subrev_nc_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_u32_u24", lambda r, d, s: f"v_mul_u32_u24 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_i32_i24", lambda r, d, s: f"v_mul_i32_i24 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_lo_u32", lambda r, d, s: f"v_mul_lo_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_hi_u32", lambda r, d, s: f"v_mul_hi_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mul_hi_i32", lambda r, d, s: f"v_mul_hi_i32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_mad_u32_u24", lambda r, d, s: f"v_mad_u32_u24 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_mad_i32_i24", lambda r, d, s: f"v_mad_i32_i24 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_min_u32", lambda r, d, s: f"v_min_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_max_u32", lambda r, d, s: f"v_max_u32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_min_i32", lambda r, d, s: f"v_min_i32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_max_i32", lambda r, d, s: f"v_max_i32 v{d}, v{s[0]}, v{s[1]}"),
        ("v_bfe_u32", lambda r, d, s: f"v_bfe_u32 v{d}, v{s[0]}, {r.randrange(0, 40)}, {r.randrange(0, 40)}"),
        ("v_bfe_i32", lambda r, d, s: f"v_bfe_i32 v{d}, v{s[0]}, {r.randrange(0, 40)}, {r.randrange(0, 40)}"),
        ("v_bfi_b32", lambda r, d, s: f"v_bfi_b32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_alignbit_b32", lambda r, d, s: f"v_alignbit_b32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_alignbyte_b32", lambda r, d, s: f"v_alignbyte_b32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_perm_b32", _emit_perm),
        ("v_med3_i32", lambda r, d, s: f"v_med3_i32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("v_med3_u32", lambda r, d, s: f"v_med3_u32 v{d}, v{s[0]}, v{s[1]}, v{s[2]}"),
        ("s_add_u32", lambda r, d, s: f"s_add_u32 s{d}, s{s[0]}, s{s[1]}"),
    ]


# NOTE: s_add_u32 is deliberately NOT generated: the translator lowers scalar
# adds to OpIAddCarry, which tests/semantic/spirv_mini_interp.py does not
# model yet, so every such program would skip with a reason instead of
# proving anything. Scalar reads are still covered: seeds are moved through
# s_regs and observed via the v_mov_b32/sgpr copy into an exported VGPR. Extending the
# mini-interp (not this runner) re-enables scalar ALU coverage.
_VEC_OPS = ("v_mov_b32", "v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32",
            "v_and_b32", "v_or_b32", "v_xor_b32", "v_lshlrev_b32",
            "v_lshrrev_b32", "v_ashrrev_i32",
            "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_mul_u32_u24",
            "v_mul_i32_i24", "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_hi_i32",
            "v_mad_u32_u24", "v_mad_i32_i24", "v_min_u32", "v_max_u32",
            "v_min_i32", "v_max_i32",
            "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_alignbit_b32",
            "v_alignbyte_b32", "v_perm_b32", "v_med3_i32", "v_med3_u32")
_FLOAT_OPS = ("v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32")


def _seed_value(rng):
    if rng.random() < 0.7:
        return rng.choice(_SMALL_INTS)
    return _u32_bits(rng.choice(_EXACT_FLOATS))


def generate_program(rng, n_ops):
    """One straight-line program. Returns (body_lines, export_writers).

    body_lines are tab-less assembly lines (seeds, ops). export_writers maps
    each exported VGPR to the (op_name, line) that last wrote it, for the
    first-mismatch report.
    """
    lines = []
    writers = {}
    for v in SCRATCH:
        lines.append(f"v_mov_b32 v{v}, {_seed_value(rng)}")
        writers[v] = ("v_mov_b32/seed", len(lines))
    s_regs = (4, 5, 6, 7)
    for s in s_regs:
        lines.append(f"s_mov_b32 s{s}, {_seed_value(rng) % 256}")
    vec_ops = [t for t in _op_templates() if t[0] in _VEC_OPS and t[0] != "v_mov_b32"]
    # A float op may return any NaN payload (Vulkan does not promise one) and the
    # comparison accepts both-NaN. If that value then fed an integer op, the two
    # payloads would produce different integer results, so integer ops never read
    # the destination of a float op. Every untainted register is bit-identical on
    # both oracles; a later integer write clears the taint.
    tainted = set()
    for _ in range(n_ops):
        name, emit = rng.choice(vec_ops)
        dst = rng.choice(SCRATCH)
        clean = [v for v in SCRATCH if v not in tainted]
        if name not in _FLOAT_OPS and not clean:
            # All float results may carry oracle-specific NaN payloads. Restore one
            # known bit pattern before choosing integer operands, even when dst is read.
            lines.append(f"v_mov_b32 v{dst}, 0")
            writers[dst] = ("v_mov_b32/seed", len(lines))
            tainted.discard(dst)
            clean = [dst]
        pool = SCRATCH if name in _FLOAT_OPS else clean
        srcs = [rng.choice(pool) for _ in range(4)]
        lines.append(emit(rng, dst, srcs))
        writers[dst] = (name, len(lines))
        if name in _FLOAT_OPS:
            tainted.add(dst)
        else:
            tainted.discard(dst)
    # Reuse the oldest export: an untouched seed if possible, else an ALU result.
    dst = min(EXPORT_REGS, key=lambda reg: writers[reg][1])
    lines.append(f"v_mov_b32 v{dst}, s{rng.choice(s_regs)}")
    writers[dst] = ("v_mov_b32/sgpr-copy", len(lines))
    return lines, writers


def _write_listing(path, lines, with_export):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("; shader_diff case  code 0x0..0x1000\n")
        for line in lines:
            f.write("\t" + line + "\n")
        if with_export:
            f.write("\texp mrt0 v8, v9, v10, v11 done vm\n")
        f.write("\ts_endpgm\n")


def compare_pair(cpu_words, spv_words, *, writers=None):
    """Compare two 4-word exports and return the first mismatch or None.

    Results of floating-point instructions can have different NaN payloads.
    All other words must match bit for bit.
    """
    if len(cpu_words) != 4 or len(spv_words) != 4:
        return f"shape mismatch: cpu has {len(cpu_words)} words, spv has {len(spv_words)}"
    writers = writers or {}
    for i in range(4):
        c = cpu_words[i] & 0xFFFFFFFF
        s = spv_words[i] & 0xFFFFFFFF
        if c == s:
            continue
        writer = writers.get(EXPORT_REGS[i])
        mismatch = f"lane word {i}: expected 0x{c:08x} ({c}), actual 0x{s:08x} ({s})"
        if writer and cpu_ref.base_op(writer[0]) in _FLOAT_OPS:
            c_nan, s_nan = _is_nan_bits(c), _is_nan_bits(s)
            if c_nan and s_nan:
                continue
            if c_nan or s_nan:
                mismatch = f"lane word {i}: only one side is NaN (cpu=0x{c:08x} spv=0x{s:08x})"
        if writer:
            mismatch += f" (last writer {writer[0]} at body line {writer[1]} for v{EXPORT_REGS[i]})"
        return mismatch
    return None


def run_one(name, lines, writers):
    """Translate + execute both oracles for one program.

    Returns (status, detail) with status in {"match", "mismatch", "skip"}.
    """
    with tempfile.TemporaryDirectory(prefix="shader-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        cpu_path = os.path.join(tmp, "cpu.s")
        _write_listing(cpu_path, lines, with_export=False)
        try:
            cpu_res = cpu_ref.run(cpu_path, {"lanes": 1})
        except cpu_ref.CpuRefError as e:
            return ("skip", f"cpu_ref unsupported: {e}")
        spv_path = os.path.join(dis, name + ".s")
        _write_listing(spv_path, lines, with_export=True)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spvasm = rdna2spirv.translate(spv_path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return ("skip", f"translator unsupported: {e}")
        try:
            out = run_fragment(spvasm)
        except UnsupportedConstruct as e:
            return ("skip", f"mini_interp unsupported: {type(e).__name__}: {e}")
        if list(out) != ["%out_mrt0"]:
            return ("skip", f"mini_interp gave outputs {sorted(out)}, want [%out_mrt0]")
        regs = cpu_res["registers"]["vgpr"]
        cpu_words = [regs.get(r, [0])[0] if isinstance(regs.get(r), list) else regs.get(r, 0)
                     for r in EXPORT_REGS]
        mismatch = compare_pair(cpu_words, list(out["%out_mrt0"]), writers=writers)
        if mismatch is None:
            return ("match", "")
        return ("mismatch", mismatch)


def run_suite(programs, seed, max_ops, limit=None, *, start=0):
    """Generate the seeded sequence, executing cases from index start. Return (counts, first_failure)."""
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "harness-error": 0}
    first_failure = None
    rng = random.Random(seed)
    total = programs if limit is None else min(programs, limit)
    for i in range(total):
        lines, writers = generate_program(rng, rng.randrange(1, max_ops + 1))
        if i < start:  # advance the original RNG stream, but skip both oracles
            continue
        name = f"ps_diff_{i:04d}"
        try:
            status, detail = run_one(name, lines, writers)
        except Exception as e:  # noqa: BLE001 - harness bugs must be loud
            counts["harness-error"] += 1
            if first_failure is None:
                first_failure = (f"{name}: harness error {type(e).__name__}: {e}\n"
                                 f"Replay: --seed {seed} --max-ops {max_ops} --program-index {i}")
            continue
        if status == "match":
            counts["pass"] += 1
        elif status == "mismatch":
            counts["mismatch"] += 1
            if first_failure is None:
                first_failure = (f"{name} (seed {seed} program {i}): {detail}\n"
                                 f"Replay: --seed {seed} --max-ops {max_ops} --program-index {i}\n" +
                                 "\n".join("\t" + line for line in lines))
        else:
            counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
    return counts, first_failure


def self_test():
    """Harness canaries. Returns True only if the harness can both pass and fail."""
    # 1. A hand-written program must match (the pipeline works).
    lines = ["v_mov_b32 v8, 4", "v_mov_b32 v9, 8", "v_add_f32 v8, v8, v9",
             "v_mov_b32 v10, 1", "v_mov_b32 v11, 2"]
    writers = {8: ("v_add_f32", 3), 9: ("v_mov_b32/seed", 2),
               10: ("v_mov_b32/seed", 4), 11: ("v_mov_b32/seed", 5)}
    status, detail = run_one("ps_canary_ok", lines, writers)
    if status != "match":
        print(f"canary FAILED: hand-written program did not match ({status}: {detail})")
        return False
    # 2. An intentionally wrong expectation must be caught (the harness can fail).
    bad = compare_pair([1, 2, 3, 4], [1, 2, 3, 5])
    if bad is None or "lane word 3" not in bad:
        print("canary FAILED: corrupted expectation was not flagged")
        return False
    # 3. An unsupported op must be a counted skip, not a silent pass.
    status, detail = run_one("ps_canary_skip", ["v_interp_p1_f32 v8, v9, v10"], {})
    if status != "skip":
        print(f"canary FAILED: unsupported op was not skipped (got {status})")
        return False
    print("canaries: match ok, mismatch detected, skip counted")
    return True


def _at_least_one(value):
    """An integer option that must be at least 1.

    0 and negative values used to run nothing (--programs, --limit) or raise
    `empty range in randrange(1, 1)` (--max-ops) instead of being refused.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {n})")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Differential oracle: cpu_ref vs translated SPIR-V.")
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--programs", type=_at_least_one, help="number of programs to run (default: 200)")
    selection.add_argument("--program-index", type=int, metavar="I",
                           help="replay only zero-based program I using the original --seed and --max-ops")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=_at_least_one, default=8,
                    help="maximum ops per generated program (default: 8)")
    ap.add_argument("--limit", type=_at_least_one, default=None,
                    help="run only the first N programs (N >= 1)")
    ap.add_argument("--list-ops", action="store_true", help="print the op table and exit")
    ap.add_argument("--self-test", action="store_true", help="run harness canaries and exit")
    a = ap.parse_args(argv)
    if a.list_ops:
        for name, _ in _op_templates():
            print(name)
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    if a.program_index is None:
        counts, first_failure = run_suite(a.programs or 200, a.seed, a.max_ops, a.limit)
    else:
        if a.program_index < 0:
            ap.error("--program-index must be nonnegative")
        if a.limit is not None:
            ap.error("--program-index cannot be combined with --limit")
        print(f"replaying program {a.program_index} (seed {a.seed}, max ops {a.max_ops})")
        counts, first_failure = run_suite(a.program_index + 1, a.seed, a.max_ops, start=a.program_index)
    total_skips = sum(counts["skip"].values())
    total_run = counts["pass"] + counts["mismatch"] + total_skips + counts["harness-error"]
    print(f"{total_run} programs (seed {a.seed}): "
          f"{counts['pass']} pass, {counts['mismatch']} mismatch, "
          f"{total_skips} skipped, {counts['harness-error']} harness errors")
    for reason, n in sorted(counts["skip"].items()):
        print(f"  skip x{n}: {reason}")
    if first_failure is not None:
        print("FIRST FAILURE:\n" + first_failure)
    if counts["harness-error"]:
        return 2
    if counts["mismatch"]:
        return 1
    if not counts["pass"]:
        print("no program ran: every one was skipped")
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
