"""Draw a stratified sample of translated pixel shaders on one GPU at a time and compare two GPUs.

The worker is `runtime/test/vk_ps_sweep` (build the `vk_ps_sweep` target). It draws one pixel shader against a
generated vertex shader, with the runtime's bindless set 0 holding only its fallbacks and a zeroed heap, so the
inputs are the same on every device. Run the same sample on each GPU, then compare:

    python -I tools/ps_sweep.py sample  --spirv <translated spirv dir> --out <dir> [--n 480] [--seed 20261122]
    python -I tools/ps_sweep.py run     --spirv <spirv dir> --out <dir> --tag a --exe <vk_ps_sweep> --glslang <glslangValidator>
    VK_LOADER_DRIVERS_DISABLE=*nv* python -I tools/ps_sweep.py run ... --tag b      # another GPU (any loader filter)
    python -I tools/ps_sweep.py compare --out <dir> --tags a b

For native and fallback translations drawn from the same sample, compare exact RGBA32F target bytes:

    python -I tools/ps_sweep.py compare --exact --out <dir> --tags native fallback --variant z

The report (exact_<variant>.json) gives one row per sampled draw, with target classes in attachment order for
compatible pairs. An exact draw is observable if any component differs from the worker's -1 clear value.
Pass --occlusion to run on a device supporting precise occlusion queries. Both compare modes then include
samples_passed in tag order for each draw (null for an unavailable measurement).
Pass --depth on both runs for an additional D32_SFLOAT attachment cleared to 1.0. This enables precise
queries and depth writes with an ALWAYS comparison. Both reports compare depth words exactly, count differing
pixels, and report non-clear pixels and passing samples separately. Shaders without an explicit depth output
write rasterized depth. Use --two-clear on both tags to repeat each draw in a fresh worker with the same
inputs and pipeline settings, clearing depth to 1 then 0. The report counts identical depth words, pixels
that follow both clears, and other changes. For deterministic draws, identical words across distinct clears
show an attachment write. Following both clears is consistent with untouched depth. Changing coverage,
side effects or undefined shader inputs limit this inference. Color and query repeatability are checked.
A clear-only color match cannot distinguish a discarded fragment from an explicit write of the clear value.

A `--spirv` folder translated with `--bary-fallback`, with `python -I tools/bary_fallback_gs.py <folder>` run on it
(it writes `<name>.bary_gs.spv`), draws each shader that reads raw barycentrics through that pass-through geometry
shader (`VKS_GS`), so it runs on a device without `fragmentShaderBarycentric`; the sample itself comes from the
normal translation (`sample` keeps its own strata), and the folder only has to contain the sampled shaders.

`--variant z` (default) leaves every texture lookup on the zero fallback; `--variant t` replaces the fallback images
with a UNORM gradient. The sint and uint variants require a matching 2D integer image at binding 0,
element zero, and constant in-bounds texel fetches at mip zero; other modules report unsupported. `--lo/--hi` split the sample over parallel workers
and `--only 3,7` re-runs single shaders (into `<tag>_<variant>_re`, which `compare` prefers), for example the ones that
timed out while another process was starting.

Shaders are numbered by their place in `sample.json`; the summary lines and everything under `<dir>` except
`sample.json` (which names the sampled shaders, so keep it out of reports) are counts and numbers only.
Usage: `python -I tools/ps_sweep.py --help` lists the options.
"""
from collections import Counter
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import time

W = H = 32
NOT_TRANSLATED = -1   # result code of a sampled shader that is not in the --spirv index
TOLERANCE = 1e-3


def load_ps(spirv):
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    return {k: v for k, v in index.items() if v.get("stage") == "ps"}


def stratum(meta):
    """(needs barycentrics, descriptor class, control-flow style): what decides whether and how a shader runs."""
    bary = "fragmentShaderBarycentric" in meta.get("features", [])
    b = meta.get("bindings", [])
    klass = "none" if not b else ("low" if max(b) <= 3 else "high")
    return (bary, klass, meta.get("control_flow"))


def pick_sample(ps, n, seed):
    """At least min(10, size) shaders per stratum and the rest in proportion; `n` in all, or more when the floors alone add up to more."""
    groups = {}
    for name in sorted(ps):
        groups.setdefault(stratum(ps[name]), []).append(name)
    total = sum(len(g) for g in groups.values())
    alloc = {s: min(len(g), max(10, round(n * len(g) / total))) for s, g in groups.items()}
    while sum(alloc.values()) > n:
        big = max(sorted(alloc, key=str), key=lambda s: alloc[s])
        if alloc[big] <= min(10, len(groups[big])):
            break
        alloc[big] -= 1
    rng = random.Random(seed)
    picked = []
    for s in sorted(groups, key=str):
        picked += rng.sample(groups[s], alloc[s])
    return picked, {s: (len(groups[s]), alloc[s]) for s in groups}


def _raw_glsl(loc):
    """The generated vec4 of the vertex shader's output at location `loc` (the value a pixel shader reads there)."""
    return (f"vec4(fract(0.13 * {loc + 1}.0 + p.x * 0.7), fract(0.29 * {loc + 1}.0 + p.y * 0.6),"
            f" 0.1 + 0.1 * {loc % 7}.0, 1.0)")


def vs_glsl(locations, flats, halves=None):
    """A vertex shader for one pixel-shader interface: a viewport-covering triangle from the vertex index and a vec4
    at each input location (varying across the triangle, constant per vertex for the flat ones).

    `halves` is the `f16_halves` of a pixel shader translated with --f16-interp-fallback, {param: [low, high]}. The
    producer of that interface exports the two f16 halves of every param's raw word, decoded to f32, at those two
    locations (UnpackHalf2x16, docs/design.md "16-bit interpolation without PerVertexKHR"). Here the param is the
    same vec4 a default translation reads raw (and per vertex) at its own location, so both translations get the same
    values: the halves are decoded from its bits at each vertex."""
    halves = {int(n): tuple(pair) for n, pair in (halves or {}).items()}
    half_locs = {loc for pair in halves.values() for loc in pair}
    lines = ["#version 450"]
    for loc in locations:
        lines.append(f"layout(location={loc}) {'flat ' if loc in flats else ''}out vec4 o{loc};")
    lines.append("void main() {")
    lines.append("  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));")
    lines.append("  gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);")
    for loc in locations:
        if loc not in half_locs:
            lines.append(f"  o{loc} = {_raw_glsl(loc)};")
    for n, pair in sorted(halves.items()):
        lines.append(f"  {{ vec4 r = {_raw_glsl(n)}; uvec4 w = floatBitsToUint(r);")
        for half, loc in enumerate(pair):
            if loc in locations:
                comps = ", ".join(f"unpackHalf2x16(w.{c})[{half}]" for c in "xyzw")
                lines.append(f"    o{loc} = vec4({comps});")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def gs_for(spirv, name):
    """The pass-through geometry shader of a pixel shader translated with --bary-fallback
    (`python -I tools/bary_fallback_gs.py <spirv dir>` writes `<name>.bary_gs.spv` beside it), or None."""
    path = os.path.join(spirv, name + ".bary_gs.spv")
    return path if os.path.isfile(path) else None


JOB_VARS = ("VKS_GS", "VKS_TEX", "VKS_NEED_MAINTENANCE8")


def job_env(base, timeout, variant, meta, gs):
    """The environment of one draw: the parent's, minus every per-job variable it may have inherited, plus this job's."""
    env = {k: v for k, v in base.items() if k not in JOB_VARS}
    env.update(VKS_TIMEOUT_S=str(timeout), VKS_HEAP_KB="1024")
    if variant == "t":
        env["VKS_TEX"] = "1"
    if variant in ("sint", "uint"):
        env["VKS_TEX"] = variant
    if gs:
        env["VKS_GS"] = gs
    if meta.get("needs_maintenance8"):
        env["VKS_NEED_MAINTENANCE8"] = "1"
    return env


def make_vs(glslang, meta, cache_dir):
    locs = sorted(meta.get("inputs", []))
    flats = set(meta.get("ps_flat_interpolants", [])) & set(locs)
    halves = meta.get("f16_halves") or {}
    key = ("vs_" + ("_".join(map(str, locs)) if locs else "none") + ("_f" + "_".join(map(str, sorted(flats))) if flats else "")
           + "".join(f"_h{n}_{lo}_{hi}" for n, (lo, hi) in sorted(halves.items(), key=lambda kv: int(kv[0]))))
    spv = os.path.join(cache_dir, key + ".spv")
    if not os.path.exists(spv):
        src = spv[:-4] + ".vert"
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(vs_glsl(locs, flats, halves))
        r = subprocess.run([glslang, "-V", "-S", "vert", src, "-o", spv], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("glslangValidator failed: " + r.stdout[-300:])
    return spv


def cmd_sample(a):
    ps = load_ps(a.spirv)
    picked, per = pick_sample(ps, a.n, a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": k, "stratum": list(stratum(ps[k]))} for k in picked], f)
    print(f"population {len(ps)} strata {len(per)} sampled {len(picked)}")
    for s in sorted(per, key=str):
        print(f"  stratum {s}: population {per[s][0]} sampled {per[s][1]}")


def run_dir(a, tag):
    return os.path.join(a.out, f"{tag}_{a.variant}{a.suffix}")


def cmd_run(a):
    ps = load_ps(a.spirv)
    with open(os.path.join(a.out, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    out_dir = run_dir(a, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(a.out, "vs")
    os.makedirs(cache, exist_ok=True)
    log = os.path.join(out_dir, f"results_{a.lo}.jsonl")
    done = set()
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    previous = json.loads(line)
                    if a.occlusion and previous["code"] == 0 and "samples_passed" not in previous:
                        raise SystemExit("--occlusion needs a fresh tag or suffix for previously unmeasured draws")
                    if previous["code"] != NOT_TRANSLATED and bool(previous.get("depth")) != a.depth:
                        raise SystemExit("different depth mode needs a fresh tag or suffix")
                    if previous["code"] != NOT_TRANSLATED and bool(previous.get("two_clear")) != a.two_clear:
                        raise SystemExit("different two-clear mode needs a fresh tag or suffix")
                    done.add(previous["i"])
    hi = a.hi or len(sample)
    for i, s in enumerate(sample):
        if i < a.lo or i >= hi or i in done or (a.only and i not in a.only):
            continue
        meta = ps.get(s["name"])
        if meta is None:   # the translation in --spirv refused it (e.g. --bary-fallback and a computed I/J pair)
            with open(log, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"i": i, "code": NOT_TRANSLATED, "reason": "not translated", "targets": 0, "sec": 0}) + "\n")
            continue
        raw = os.path.join(out_dir, f"{i}.raw")
        if os.path.exists(raw):
            os.unlink(raw)
        targets = max(meta.get("outputs") or [0]) + 1
        gs = gs_for(a.spirv, s["name"]) if meta.get("bary_fallback_input") else None
        if meta.get("bary_fallback_input") and not gs:   # a missing companion is not a pipeline failure
            with open(log, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"i": i, "code": NOT_TRANSLATED, "reason": "no bary_gs companion", "targets": 0, "sec": 0}) + "\n")
            continue
        env = job_env(os.environ, a.timeout, a.variant, meta, gs)
        cmd = [a.exe, make_vs(a.glslang, meta, cache), os.path.join(a.spirv, s["name"] + ".spv"), raw,
               str(W), str(H), str(targets)]
        if a.depth:
            cmd.append("--depth")
            if os.path.exists(raw + ".depth"):
                os.unlink(raw + ".depth")
        if a.occlusion:
            cmd.append("--occlusion")
        start = time.time()
        counts = []
        for zero in ([False, True] if a.two_clear else [False]):
            draw = cmd.copy()
            if zero:
                draw[3] = raw + ".zero"
                draw[draw.index("--depth")] = "--depth-zero"
                for path in (draw[3], draw[3] + ".depth"):
                    if os.path.exists(path):
                        os.unlink(path)
            try:
                r = subprocess.run(draw, env=env, capture_output=True, text=True, timeout=a.timeout + 15,
                                   stdin=subprocess.DEVNULL)
                code, out, err = r.returncode, r.stdout, r.stderr
            except subprocess.TimeoutExpired:
                code, out, err = 3, "", ""
            if code:
                break
            if a.occlusion or a.depth:
                values = [line.removeprefix("samples_passed ") for line in out.splitlines()
                          if line.startswith("samples_passed ")]
                if len(values) != 1 or not values[0].isascii() or not values[0].isdecimal():
                    raise SystemExit("worker succeeded without a valid samples_passed measurement")
                counts.append(int(values[0]))
        reason = ""   # a coarse reason: a missing feature or the failing Vulkan call, never a shader name
        if code == 77:
            reason = out.strip().replace("unsupported: ", "")[:80]
        elif code not in (0, 3):
            for line in err.splitlines():
                if line.startswith(("FAILED", "DEVICE_LOST", "PIPELINE")):
                    reason = line.split(" (line")[0][:60]
                    break
        rec = {"i": i, "code": code, "reason": reason, "targets": targets, "sec": round(time.time() - start, 2)}
        if a.depth:
            rec["depth"] = True
        if a.two_clear:
            rec["two_clear"] = True
        if counts and code == 0:
            rec["samples_passed"] = counts[0]
            if a.two_clear:
                rec["zero_samples_passed"] = counts[1]
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
    print("done", a.tag, a.variant, a.lo)


def written_stats(vals):
    """(has NaN, every written value is zero, nothing written). The harness clears every target to -1."""
    written = [x for x in vals if x != -1.0]
    return (any(math.isnan(x) for x in vals), bool(written) and all(x == 0.0 for x in written), not written)


def classify_pair(rec_a, rec_b, raw_a, raw_b, tol=TOLERANCE):
    """The class of one shader from its two result records and output files."""
    code_a, code_b = rec_a["code"], rec_b["code"]
    if NOT_TRANSLATED in (code_a, code_b):
        return "not translated", ""
    if 3 in (code_a, code_b):
        return "timeout", ""
    if code_a == 0 and code_b == 0:
        targets_a, targets_b = rec_a.get("targets"), rec_b.get("targets")
        if any(type(n) is not int or not 1 <= n <= 8 for n in (targets_a, targets_b)):
            return "invalid readback", "invalid target count"
        if targets_a != targets_b:
            return "invalid readback", "target counts differ"
        size = W * H * 16 * targets_a  # RGBA32F, packed target by target
        outputs = []
        for tag, path in (("a", raw_a), ("b", raw_b)):
            try:
                with open(path, "rb") as f:
                    data = f.read(size + 1)  # detect trailing bytes without reading an unbounded file
            except FileNotFoundError:
                return "invalid readback", f"{tag}: missing output"
            if len(data) != size:
                return "invalid readback", f"{tag}: wrong output size"
            outputs.append(struct.unpack(f"<{size // 4}f", data))
        fa, fb = outputs
        nan_a, zero_a, none_a = written_stats(fa)
        nan_b, zero_b, none_b = written_stats(fb)
        if (nan_a, zero_a, none_a) != (nan_b, zero_b, none_b):
            return "NaN or zero on one GPU only", "nan" if nan_a != nan_b else ("zero" if zero_a != zero_b else "unwritten")
        worst, bad = 0.0, 0
        for x, y in zip(fa, fb):
            if x == y or (math.isnan(x) and math.isnan(y)):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                # A NaN against a number, or an infinity against anything else, always differs; the tolerance
                # arithmetic would pass inf against 1.0 and NaN against NaN in another component.
                bad += 1
                worst = math.inf
                continue
            d = abs(x - y)
            if d > tol + tol * max(abs(x), abs(y)):
                bad += 1
                worst = max(worst, d)
        if bad:
            tier = "gross (>0.1)" if worst > 0.1 else ("moderate (>1e-2)" if worst > 1e-2 else "small (>1e-3)")
            return "differs beyond tolerance", tier
        return "OK on both", ""
    if code_a != 0 and code_b != 0:
        return "fails on both", f"{code_a}/{code_b}"
    bad_rec, which = (rec_a, "a") if code_a != 0 else (rec_b, "b")
    kind = "missing device feature" if bad_rec["code"] == 77 else f"exit {bad_rec['code']}"
    return "fails on one GPU", f"{which}: {kind}"


def load_results(a, tag):
    d = run_dir(a, tag)
    recs = {}
    for folder in (d, d + "_re"):   # re-runs override the first pass
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("results_") and name.endswith(".jsonl"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            r["dir"] = folder
                            recs[r["i"]] = r
    return recs


def exact_pair(i, left, right):
    """Compare successful draws with complete, equally shaped readbacks. Ignore stale files of failed draws."""
    target_bytes = W * H * 16
    clear = struct.pack("<f", -1.0) * (W * H * 4)
    payloads = []
    for rec in (left, right):
        if rec is None or rec["code"] != 0:
            return "missing", []
    if left["targets"] != right["targets"]:
        return "incompatible", []
    count = left["targets"]
    if type(count) is not int or type(right["targets"]) is not int or not 1 <= count <= 8:
        return "incompatible", []
    size = count * target_bytes
    for rec in (left, right):
        try:
            with open(os.path.join(rec["dir"], f"{i}.raw"), "rb") as f:
                data = f.read(size + 1)
        except FileNotFoundError:
            return "missing", []
        if len(data) != size:
            return "incompatible", []
        payloads.append(data)

    targets = []
    for offset in range(0, size, target_bytes):
        a, b = (data[offset:offset + target_bytes] for data in payloads)
        if a != b:
            targets.append("mismatch")
        elif a == clear:
            targets.append("clear-only exact")
        else:
            targets.append("non-clear exact")
    for outcome in ("mismatch", "non-clear exact"):
        if outcome in targets:
            return outcome, targets
    return "clear-only exact", targets


def draw_measurements(left, right):
    counts = [rec.get("samples_passed") if rec and rec["code"] == 0 else None for rec in (left, right)]
    return {"samples_passed": counts} if any(count is not None for count in counts) else {}


def depth_measurements(i, left, right):
    """Compare attachment words independently of the precise occlusion count."""
    pair = (left, right)
    if not any(rec and rec.get("depth") for rec in pair):
        return {}
    pixels = W * H
    clear = struct.pack("<f", 1.0)
    planes, changed = [], []
    controls, zero_planes, zero_nonclear = [], [], []
    for rec in pair:
        if not rec or not rec.get("depth") or rec["code"] != 0:
            return {"depth_class": "missing"}
        try:
            with open(os.path.join(rec["dir"], f"{i}.raw.depth"), "rb") as f:
                data = f.read(pixels * 4 + 1)
        except FileNotFoundError:
            return {"depth_class": "missing"}
        if len(data) != pixels * 4:
            return {"depth_class": "incompatible"}
        count = sum(data[n:n + 4] != clear for n in range(0, len(data), 4))
        passed = rec.get("samples_passed")
        if type(passed) is not int or not count <= passed <= pixels:
            return {"depth_class": "incompatible"}
        planes.append(data)
        changed.append(count)
        if rec.get("two_clear"):
            raw = os.path.join(rec["dir"], f"{i}.raw")
            try:
                with open(raw + ".zero.depth", "rb") as f:
                    zero = f.read(pixels * 4 + 1)
                color_size = pixels * 16 * rec["targets"]
                with open(raw, "rb") as first, open(raw + ".zero", "rb") as second:
                    color = first.read(color_size + 1)
                    same_color = len(color) == color_size and color == second.read(color_size + 1)
            except FileNotFoundError:
                return {"depth_class": "missing"}
            if len(zero) != len(data) or not same_color or rec.get("zero_samples_passed") != passed:
                return {"depth_class": "incompatible"}
            fixed = follows = 0
            for n in range(0, len(data), 4):
                first_word, second_word = data[n:n + 4], zero[n:n + 4]
                fixed += first_word == second_word
                follows += first_word == clear and second_word == bytes(4)
            controls.append({"identical": fixed, "follow_clear": follows, "other": pixels - fixed - follows})
            zero_planes.append(zero)
            zero_nonclear.append(sum(zero[n:n + 4] != bytes(4) for n in range(0, len(zero), 4)))
    differences = sum(planes[0][n:n + 4] != planes[1][n:n + 4] for n in range(0, pixels * 4, 4))
    result = {"depth_different_pixels": differences, "depth_nonclear_pixels": changed}
    if controls:
        if len(controls) != 2:
            return {"depth_class": "incompatible"}
        zero_diff = sum(zero_planes[0][n:n + 4] != zero_planes[1][n:n + 4] for n in range(0, pixels * 4, 4))
        result.update(depth_clears=[1.0, 0.0], depth_control=controls, zero_depth_different_pixels=zero_diff,
                      zero_depth_nonclear_pixels=zero_nonclear)
        differences += zero_diff
    result["depth_class"] = "mismatch" if differences else "exact"
    return result


def compare_exact(a, sample, ra, rb):
    draws, attachments = Counter(), Counter()
    rows = []
    for i in range(len(sample)):
        left, right = ra.get(i), rb.get(i)
        outcome, targets = exact_pair(i, left, right)
        depth = depth_measurements(i, left, right)
        if depth.get("depth_class") in ("missing", "incompatible", "mismatch"):
            outcome = depth["depth_class"]
        elif outcome == "clear-only exact" and (any(depth.get("depth_nonclear_pixels", [])) or
                                                any(depth.get("zero_depth_nonclear_pixels", []))):
            outcome = "non-clear exact"
        draws[outcome] += 1
        attachments.update(targets)
        rows.append({"i": i, "class": outcome, "targets": targets, **depth, **draw_measurements(left, right),
                     "codes": [r["code"] if r else None for r in (left, right)]})
    with open(os.path.join(a.out, f"exact_{a.variant}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"tags": a.tags, "draws": draws, "targets": attachments, "rows": rows}, f)
    for label, counts in (("draws", draws), ("targets", attachments)):
        print(label, "total", sum(counts.values()))
        for outcome, n in sorted(counts.items()):
            print(f"  {outcome}: {n}")


def cmd_compare(a):
    with open(os.path.join(a.out, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    ta, tb = a.tags
    ra, rb = load_results(a, ta), load_results(a, tb)
    if a.exact:
        return compare_exact(a, sample, ra, rb)
    counts, sub, rows = {}, {}, []
    for i, s in enumerate(sample):
        if i not in ra or i not in rb:
            raise SystemExit(f"shader {i} has no result on both tags")
        c, sc = classify_pair(ra[i], rb[i], os.path.join(ra[i]["dir"], f"{i}.raw"), os.path.join(rb[i]["dir"], f"{i}.raw"))
        depth = depth_measurements(i, ra[i], rb[i])
        if c == "OK on both" and depth.get("depth_class") in ("missing", "incompatible", "mismatch"):
            c, sc = "depth " + depth["depth_class"], ""
        counts[c] = counts.get(c, 0) + 1
        sub[(c, sc)] = sub.get((c, sc), 0) + 1
        rows.append({"i": i, "class": c, "sub": sc, "stratum": s["stratum"], **depth, **draw_measurements(ra[i], rb[i]),
                     ta: ra[i]["code"], tb: rb[i]["code"],
                     ta + "_reason": ra[i]["reason"], tb + "_reason": rb[i]["reason"]})
    with open(os.path.join(a.out, f"classes_{a.variant}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f)
    print("variant", a.variant, "total", len(sample))
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    for (c, sc), n in sorted(sub.items()):
        if sc:
            print(f"    {c} / {sc}: {n}")


WARNING = ("warning (run): a few sampled shaders hang NVIDIA GPUs with the zeroed heap (VK_ERROR_DEVICE_LOST, a\n"
           "driver reset, TDR). --timeout only bounds the wait on the host side. On a GPU that drives your display\n"
           "the desktop freezes until each reset, so run on a GPU with no display attached, or expect a freeze of a\n"
           "few seconds per hang. --lo/--hi/--only run part of the sample.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=WARNING)
    ap.add_argument("cmd", choices=["sample", "run", "compare"])
    ap.add_argument("--spirv", help="the translated SPIR-V folder (index.json and the .spv files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--tag", help="a label for the GPU this run uses")
    ap.add_argument("--tags", nargs=2, default=["a", "b"])
    ap.add_argument("--two-clear", action="store_true", help="repeat depth draw at clears 1 and 0 (run)")
    ap.add_argument("--depth", action="store_true", help="capture D32 depth and precise samples passed (run)")
    ap.add_argument("--occlusion", action="store_true", help="measure precise samples passed per draw (run)")
    ap.add_argument("--exact", action="store_true", help="compare target bytes and count clear-only pairs separately")
    ap.add_argument("--variant", choices=["z", "t", "sint", "uint"], default="z")
    ap.add_argument("--exe", help="vk_ps_sweep")
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("--timeout", type=int, default=20, help="seconds the GPU may take for one draw")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=0)
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--suffix", default="", help="results folder suffix (a re-run uses _re)")
    a = ap.parse_args(argv)
    a.depth |= a.two_clear
    if a.depth and a.cmd != "run":
        ap.error("--depth needs run")
    if a.occlusion and a.cmd != "run":
        ap.error("--occlusion needs run")
    if a.exact and (a.cmd != "compare" or a.tags[0] == a.tags[1]):
        ap.error("--exact needs compare with two different tags")
    if a.cmd == "run" and not (a.tag and a.exe and a.spirv):
        ap.error("run needs --tag, --exe and --spirv")
    if a.cmd == "run" and a.only and not a.suffix:
        a.suffix = "_re"
    if a.cmd == "sample" and not a.spirv:
        ap.error("sample needs --spirv")
    {"sample": cmd_sample, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
