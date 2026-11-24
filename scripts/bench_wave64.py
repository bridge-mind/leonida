"""Time a wave64 compute shader (runtime/test/shaders/cs_bench_mix.asm) native and emulated, on the AMD and NVIDIA GPU.

    python -I scripts/bench_wave64.py [--build DIR] [--llvm-mc PATH] [--groups N] [--reps N] [--procs N] [--shader NAME]

Three configurations: AMD with its native 64-wide subgroup (no flag), AMD with --wave64-emulate, NVIDIA with
--wave64-emulate (the GPU is chosen with VK_LOADER_DRIVERS_SELECT='*amd*,*radeon*' / '*nv*'). Each runs `procs` processes of
vk_compute_test with VKC_BENCH=`reps` timed dispatches of `groups` workgroups; the table gives the median over all
dispatches, the spread of the per-process medians (other programs share the GPU, so look at it), the number of
translated segments (static: blocks + continuation segments; every continuation segment is one more workgroup barrier
on the path through it) and whether the output bytes of the emulated runs equal the native AMD run, which is the reference (they must).
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER_DIR = os.path.join(ROOT, "runtime", "test", "shaders")
BUILD_PY = os.path.join(SHADER_DIR, "build.py")
sys.path.insert(0, SHADER_DIR)
from check import EXE, pick_build_dir  # noqa: E402

COMPUTE_TEST = "vk_compute_test" + EXE
CONFIGS = [("AMD native", "*amd*,*radeon*", False), ("AMD --wave64-emulate", "*amd*,*radeon*", True),
           ("NVIDIA --wave64-emulate", "*nv*", True)]


def build(work, name, emulate, llvm_mc):
    out = os.path.join(work, "emu" if emulate else "native")
    env = dict(os.environ, LLVM_MC=llvm_mc)
    env.pop("LEONIDA_W64_EMULATE", None)
    if emulate:
        env["LEONIDA_W64_EMULATE"] = "1"
    r = subprocess.run([sys.executable, "-I", BUILD_PY, out, name], env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    return os.path.join(out, "spirv", name + ".spv")


def segments(spv: str) -> tuple[int, int] | None:
    """Count scheduler cases and continuations, including inner switches in a partitioned dispatch."""
    with open(spv[:-4] + ".spvasm", encoding="utf-8") as source:
        text = source.read()
    tables = re.findall(r"^[ \t]*OpSwitch %pcv %(?:sw_default|swm\d+) ([^\n]+)", text, re.MULTILINE)
    if not tables:
        return None
    cases = sorted(int(case) for table in tables for case in re.findall(r"(\d+) %blk\d+", table))
    blocks = next((i for i, c in enumerate(cases) if c != i), len(cases))   # blocks 0..n-1, then "done" = n, then segments
    return len(cases), len(cases) - blocks


def run(tools, spv, groups, reps, driver, emulate):
    env = dict(os.environ, VK_LOADER_DRIVERS_SELECT=driver, VKC_BENCH=str(reps))
    args = [os.path.join(tools, COMPUTE_TEST), spv, str(groups), "dump"]
    if not emulate:
        args.append("--require-wave64")
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True)
    except OSError as e:
        return None, f"{args[0]}: {e}", None
    out = r.stdout + r.stderr
    m = re.search(r"bench ms: median=([\d.]+) min=([\d.]+) max=([\d.]+)", out)
    h = re.search(r"bench out fnv1a: (\w+)", out)
    dev = re.search(r"device: (.*) \(subgroup", out)
    if r.returncode or not m or not h or "0 mismatches" not in out:
        why = f"exit {r.returncode}" if r.returncode else "no 'bench ms' / 'bench out fnv1a' line (rebuild vk_compute_test)"
        return None, why + ": " + (out.strip().splitlines()[-1:] or ["no output"])[0], None
    return (float(m.group(1)), float(m.group(2))), h.group(1), dev.group(1) if dev else "?"


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"),
                    help="CMake build directory or a configuration directory such as RelWithDebInfo")
    ap.add_argument("--llvm-mc", default=os.environ.get("LLVM_MC", "llvm-mc"))
    # Vulkan guarantees support for at least 65,535 groups along X in one dispatch.
    ap.add_argument("--groups", type=positive_int, default=65535)
    ap.add_argument("--reps", type=positive_int, default=30)
    ap.add_argument("--procs", type=positive_int, default=5)
    ap.add_argument("--shader", default="cs_bench_mix")
    a = ap.parse_args()
    tools = pick_build_dir(a.build, (COMPUTE_TEST,))
    with tempfile.TemporaryDirectory(prefix="bench-w64-") as work:
        spv = {False: build(work, a.shader, False, a.llvm_mc), True: build(work, a.shader, True, a.llvm_mc)}
        seg = segments(spv[True])
        print(f"{a.shader}: {a.groups} workgroups of the shader's group size, {a.procs} x {a.reps} timed dispatches")
        if seg:
            print(f"emulated: {seg[0]} segments, {seg[1]} of them continuation segments (one extra round each)")
        hashes, rows = {}, []
        for name, driver, emulate in CONFIGS:
            med, mins, seen, dev = [], [], set(), None
            for _ in range(a.procs):
                t, h, dev = run(tools, spv[emulate], a.groups, a.reps, driver, emulate)
                if t is None:
                    med = None
                    print(f"{name}: failed or unsupported: {h}")
                    break
                med.append(t[0])
                mins.append(t[1])
                seen.add(h)
            if med:
                # Only a configuration whose every process ran counts: the hashes of the processes before a
                # failed one used to be kept, so the failure passed as a configuration that had run.
                hashes[name] = seen
                rows.append((name, dev, statistics.median(med), min(mins), min(med), max(med)))
        for name, dev, m, lo, a_, b_ in rows:
            print(f"{name:26} {dev:34} median {m:8.3f} ms   best {lo:8.3f}   process medians {a_:.3f}..{b_:.3f}")
        ref_name = CONFIGS[0][0]
        ref = hashes.get(ref_name)
        if not ref or len(ref) != 1:
            print(f"no reference output from {ref_name}: cannot compare")
            return 1
        bad = [n for n, _, _ in CONFIGS[1:] if n in hashes and hashes[n] != ref]
        missing = [n for n, _, _ in CONFIGS[1:] if n not in hashes]
        print(f"reference output hash: {next(iter(ref))} ({ref_name})")
        if bad:
            print(f"OUTPUT DIFFERS from the reference: {', '.join(bad)}")
        if missing:
            print(f"incomplete comparison: no complete result from {', '.join(missing)}")
        if bad or missing:
            return 1
        print("emulated output equals the reference")
        return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
