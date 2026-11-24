"""GPU A/B of control flow: render branchy pixel shaders translated both ways, as one loop around a switch
(--dispatch-loop) and as real loops and conditionals (the default), and compare the two images byte for byte.

    python -I tools/ab_control_flow.py <repo dir> <work>/shaders <out dir> [--count 150] [--loops only] [--runs 3]

Needs a finished shaders stage (the disassembly and index.json in <work>/shaders), spirv-as and spirv-val on PATH,
and runtime/build/vk_texture_test (or --harness). Shaders are picked from the corpus: pixel stage, at least two
branches, only the bindings and features the harness provides, and nothing that writes memory or images.

What the comparison needs to mean anything, each learned from a run that looked like a translator bug:
  - Every pixel input has to be defined. The vertex shader made here writes all 32 locations; with the game's
    pass-through one, which writes location 0 only, a shader that uses another input as a buffer index reads
    from wherever the driver's leftover value points, differently in each form.
  - Without an MRT0 export, compare only the mask of MRT0 pixels differing from the clear value.
  - Shaders with loops go to a software driver first (--harness with a Linux build under Mesa llvmpipe). One
    that never ends is killed after --timeout there; on a real GPU it resets the driver. Feed the ones that
    finish back with --names.
The heap is zeros. Equal MRT0 readbacks can result from different execution paths. By default the harness binds only
MRT0. The rows in <out dir>/outcomes.tsv distinguish clear-only matches and list unobserved colour targets.
Pass --all-attachments to use vk_ps_sweep and compare each declared RGBA32F target separately.
Usage: `python -I tools/ab_control_flow.py --help` lists the options.
"""
import argparse
import collections
import os
import re
import struct
import subprocess
import sys

# Left out: anything that writes memory or images, LDS, and the rare instructions the harness has no setup for.
# Loads are fine: with empty user data they read zeros from the start of the harness's heap, the same for both.
NEEDS = re.compile(r"^\t(image_store|image_atomic|buffer_(?:store|atomic)|tbuffer_|global_store|global_atomic|ds_|s_swappc|"
                   r"s_getpc|s_sendmsg|s_sleep)", re.M)
BRANCH = re.compile(r"^\ts_c?branch", re.M)
BACK = re.compile(r"^\ts_c?branch\S* (\d+)", re.M)
CLEAR = struct.pack("<4f", -1, -1, -1, -1)      # what vk_texture_test clears the image to
READBACK_BYTES = 64 * 64 * len(CLEAR)  # vk_texture_test writes a 64x64 RGBA32F image
LOCATIONS = 32      # every pixel-shader input location gets a defined value
SUPPORTED_BINDINGS = {0, 1, 2}
HOSTED_FEATURES = {"bufferDeviceAddress", "runtimeDescriptorArray", "shaderInt64", "shaderSampledImageArrayDynamicIndexing"}


def is_backward_branch(disassembly):
    """Return whether the disassembly contains a branch to the backward half of the address space."""
    return any(int(target) >= 0x8000 for target in BACK.findall(disassembly))


def valid_metadata(metadata):
    """Whether one index entry is an object whose bindings and features, when present, are lists of scalars."""
    return isinstance(metadata, dict) and all(
        isinstance(metadata.get(key, []), list)
        and all(isinstance(v, (int, str)) for v in metadata.get(key, []))
        for key in ("bindings", "features"))


def supported_pixel_shaders(index):
    """Return sorted pixel-shader names whose bindings and features are available in the harness."""
    return sorted(name for name, metadata in index.items()
                  if metadata.get("stage") == "ps"
                  and set(metadata.get("bindings", [])) <= SUPPORTED_BINDINGS
                  and set(metadata.get("features", [])) <= HOSTED_FEATURES)


def select_candidates(index, disasm_dir, spirv_dir, count=150, loops="none", names_file=None):
    """Select shaders from synthetic or real metadata/disassembly inputs.

    An explicit names file intentionally bypasses automatic metadata, disassembly, SPIR-V and loop filtering,
    matching the CLI's documented "run exactly these shaders" behavior.
    """
    if loops not in ("none", "only"):
        raise ValueError("loops must be 'none' or 'only'")
    if count < 1:       # the check below runs after the first append, so 0 or less still picked one shader
        raise ValueError("count must be at least 1")
    if names_file:
        with open(names_file, encoding="utf-8") as f:
            return [name for name in f.read().split("\n") if name]

    picked = []
    for name in supported_pixel_shaders(index):
        asm_path = os.path.join(disasm_dir, name + ".s")
        spv_path = os.path.join(spirv_dir, name + ".spv")
        if not os.path.isfile(spv_path) or not os.path.isfile(asm_path):
            continue
        with open(asm_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if NEEDS.search(text) or len(BRANCH.findall(text)) < 2:
            continue
        if is_backward_branch(text) != (loops == "only"):
            continue
        picked.append(name)
        if len(picked) >= count:
            break
    return picked


def classify_result(translation_valid, dispatch_runs, structured_runs):
    """Classify one shader's A/B result from each form's repeated harness outcomes.

    A run is bytes on success, None when the harness failed, or the string "timeout". Every run of each form
    must agree before the two forms are compared.
    """
    if not translation_valid:
        return "a translation is invalid"

    dispatch = dispatch_runs[0] if dispatch_runs else None
    structured = structured_runs[0] if structured_runs else None
    if (any(result != dispatch for result in dispatch_runs[1:])
            or any(result != structured for result in structured_runs[1:])):
        return "unstable from run to run"

    if dispatch is None or structured is None:
        return "dispatch form failed to run" if dispatch is None else "structured form failed to run"
    if dispatch == "timeout" or structured == "timeout":
        if dispatch == "timeout" and structured == "timeout":
            return "neither terminates"
        return "ONLY ONE FORM TERMINATES"
    return "identical" if dispatch == structured else "DIFFERENT"


def write_outcomes(out_dir, outcomes):
    """Write (shader name, outcome) pairs to the harness's tab-separated report."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "outcomes.tsv")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{name}\t{outcome}\n" for name, outcome in outcomes))
    return path


def test_vertex_shader(out_dir, runner=None):
    """Generate a full-screen triangle vertex shader and assemble/validate it.

    runner is injectable so tests can exercise source generation without requiring SPIR-V Tools.
    """
    if runner is None:
        runner = subprocess.run
    nl = "\n"
    asm = f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %pos %vid {" ".join(f"%o{i}" for i in range(LOCATIONS))}
OpDecorate %pos BuiltIn Position
OpDecorate %vid BuiltIn VertexIndex
{nl.join(f"OpDecorate %o{i} Location {i}" for i in range(LOCATIONS))}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v4 = OpTypeVector %float 4
%po = OpTypePointer Output %v4
%pi = OpTypePointer Input %int
%pos = OpVariable %po Output
%vid = OpVariable %pi Input
{nl.join(f"%o{i} = OpVariable %po Output" for i in range(LOCATIONS))}
%f0 = OpConstant %float 0
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%fm1 = OpConstant %float -1
%i1 = OpConstant %int 1
%i2 = OpConstant %int 2
%zero = OpConstantComposite %v4 %f0 %f0 %f0 %f0
%main = OpFunction %void None %fn
%l = OpLabel
%i = OpLoad %int %vid
%a = OpShiftLeftLogical %int %i %i1
%b = OpBitwiseAnd %int %a %i2
%c = OpBitwiseAnd %int %i %i2
%u = OpConvertSToF %float %b
%v = OpConvertSToF %float %c
%x = OpFMul %float %u %f2
%x2 = OpFAdd %float %x %fm1
%y = OpFMul %float %v %f2
%y2 = OpFAdd %float %y %fm1
%p = OpCompositeConstruct %v4 %x2 %y2 %f0 %f1
OpStore %pos %p
%t = OpCompositeConstruct %v4 %u %v %f0 %f1
OpStore %o0 %t
{nl.join(f"OpStore %o{i} %zero" for i in range(1, LOCATIONS))}
OpReturn
OpFunctionEnd
"""
    src, dst = os.path.join(out_dir, "test_vs.spvasm"), os.path.join(out_dir, "test_vs.spv")
    os.makedirs(out_dir, exist_ok=True)
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    runner(["spirv-as", "--target-env", "vulkan1.2", src, "-o", dst], check=True)
    runner(["spirv-val", "--target-env", "vulkan1.2", dst], check=True)
    return dst


def positive_int(text):
    """--count, --runs and --timeout: a whole number of at least 1. --runs 0 rendered nothing and reported every
    shader as "dispatch form failed to run"; --count 0 still ran one shader."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def render_targets(forms, name, harness, vs, out, runs, timeout, observed, declared, all_attachments):
    """Collect repeated readbacks by attachment, retaining only the mask for undeclared MRT0."""
    count = observed[-1] + 1
    size = count * READBACK_BYTES
    images = {target: [[], []] for target in observed}
    for form, (tag, directory) in enumerate(forms):
        spirv = os.path.join(directory, name + ".spv")
        raw = os.path.join(out, f"{tag}.raw")
        args = ["64", "64", str(count)] if all_attachments else ["8", "92", "c00000", "14500000", "0"]
        for _ in range(runs):
            if os.path.isfile(raw):
                os.remove(raw)
            data = None
            failure = None
            try:
                h = subprocess.run([harness, vs, spirv, raw, *args], capture_output=True, text=True,
                                   timeout=timeout)
                if all_attachments and h.returncode == 3:
                    failure = "timeout"
                if h.returncode == 0 and os.path.isfile(raw):
                    with open(raw, "rb") as readback:
                        packed = readback.read(size + 1)
                    if len(packed) == size:
                        data = packed
            except subprocess.TimeoutExpired:
                failure = "timeout"
            for target in observed:
                image = failure
                if data is not None:
                    image = data[target * READBACK_BYTES:(target + 1) * READBACK_BYTES]
                    if target not in declared:
                        image = bytes(image[i:i + 16] != CLEAR for i in range(0, len(image), 16))
                images[target][form].append(image)
            if failure == "timeout":
                break
    return images


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("shaders")
    ap.add_argument("out")
    ap.add_argument("--count", type=positive_int, default=150)
    ap.add_argument("--all-attachments", action="store_true", help="compare each colour export using vk_ps_sweep")
    ap.add_argument("--harness", help="override the executable for the selected harness")
    ap.add_argument("--loops", choices=("none", "only"), default="none",
                    help="'only': shaders with backward branches. Use a software driver for those: one that never "
                         "terminates is killed after --timeout seconds, where a real GPU would reset its driver")
    ap.add_argument("--timeout", type=positive_int, default=60)
    ap.add_argument("--names", help="run exactly the shaders listed in this file (one per line), no selection")
    ap.add_argument("--runs", type=positive_int, default=1,
                    help="render each form this many times; all must agree")
    a = ap.parse_args(argv)
    disasm = os.path.join(a.shaders, "dxbc", "disasm")
    old = os.path.join(a.shaders, "spirv")
    # Everything the run reads is checked before anything is created: a mistyped shaders folder or names file
    # used to leave an empty output tree behind and end in a traceback.
    import json
    index_path = os.path.join(old, "index.json")
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        ap.error(f"cannot read {index_path}: {getattr(e, 'strerror', None) or e}")
    if a.names and not os.path.isfile(a.names):
        ap.error(f"--names {a.names} is not a file")
    if not isinstance(index, dict) or any(not valid_metadata(item) for item in index.values()):
        ap.error(f"{index_path}: expected a JSON object mapping shader names to metadata "
                 "(an object whose bindings and features, when present, are lists of numbers or strings)")
    names = supported_pixel_shaders(index)
    print(f"{len(names)} pixel shaders use only the bindings and features the harness provides")
    try:
        picked = select_candidates(index, disasm, old, count=a.count, loops=a.loops, names_file=a.names)
        branches = []
        for name in picked:
            with open(os.path.join(disasm, name + ".s"), encoding="utf-8", errors="replace") as f:
                branches.append(len(BRANCH.findall(f.read())))
        branches.sort()
    except (OSError, UnicodeError) as e:
        ap.error(f"cannot read selected shader inputs: {e}")
    if not picked:
        print("ERROR: no shader candidates for the control-flow comparison", file=sys.stderr)
        return 1
    new = os.path.join(a.out, "new")
    out_existed = os.path.isdir(a.out)
    new_existed = os.path.isdir(new)
    try:
        if out_existed:
            os.listdir(a.out)  # an unreadable output folder is reported here, before anything is written
        os.makedirs(new, exist_ok=True)
    except OSError as e:
        ap.error(f"cannot create {new}: {e.strerror or e}")
    # The texture test's harness: it binds a heap, a descriptor table, a fallback texture and a sampler.
    exe = "vk_ps_sweep" if a.all_attachments else "vk_texture_test"
    if os.name == "nt":
        exe += ".exe"
    harness = a.harness or os.path.join(a.repo, "runtime", "build", exe)
    try:
        vs = test_vertex_shader(a.out)
    except (OSError, subprocess.CalledProcessError) as e:
        # Remove only what this run created: its own files and the folders it made, and only while they are
        # empty, so output another process wrote beside them survives.
        for path in (os.path.join(a.out, "test_vs.spvasm"), os.path.join(a.out, "test_vs.spv")):
            try:
                os.remove(path)
            except OSError:
                pass
        for path, existed in ((new, new_existed), (a.out, out_existed)):
            if not existed:
                try:
                    os.rmdir(path)
                except OSError:
                    pass
        ap.error(f"cannot build the test vertex shader (needs spirv-as and spirv-val on PATH): {e}")

    print(f"{len(picked)} branchy pixel shaders picked ({'with loops' if a.loops == 'only' else 'loop-free'}); "
          f"branches per shader: median {branches[len(branches) // 2]}, max {branches[-1]}")

    # Both forms from the current translator, so the control-flow arrangement is the only difference.
    loopdir = os.path.join(a.out, "dispatch")
    os.makedirs(loopdir, exist_ok=True)
    res = collections.Counter()
    differ = []
    outcomes = []

    def note(name, what):
        res[what] += 1
        outcomes.append((name, what))

    tool = os.path.join(a.repo, "tools", "rdna2spirv.py")
    for name in picked:
        src = os.path.join(disasm, name + ".s")
        r1 = subprocess.run([sys.executable, "-I", tool, src, "--out", new], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-I", tool, src, "--out", loopdir, "--dispatch-loop"],
                            capture_output=True, text=True)
        if "1 pass spirv-val" not in r1.stdout or "1 pass spirv-val" not in r2.stdout:
            note(name, "a translation is invalid")
            continue
        targets = []
        for directory in (loopdir, new):
            with open(os.path.join(directory, name + ".spvasm"), encoding="utf-8") as assembly:
                targets.append(set(re.findall(r"%out_mrt(\d+)\b", assembly.read())))
        if targets[0] != targets[1]:
            note(name, "DIFFERENT colour target declarations")
            differ.append(name)
            continue
        declared = sorted(map(int, targets[0]))
        observed = sorted({0, *declared}) if a.all_attachments else [0]
        if a.all_attachments:
            cached = index.get(name, {}).get("outputs")
            if cached is not None and (not isinstance(cached, list) or
                                       any(type(t) is not int for t in cached) or sorted(cached) != declared):
                note(name, "cached outputs disagree with translated colour targets")
                continue
            if observed[-1] > 7:
                note(name, "colour target outside MRT0..7")
                continue
        images = render_targets((("old", loopdir), ("new", new)), name, harness, vs, a.out, a.runs,
                                a.timeout, observed, declared, a.all_attachments)
        labels = []
        for target, (old_runs, new_runs) in images.items():
            outcome = classify_result(True, old_runs, new_runs)
            if outcome in ("DIFFERENT", "ONLY ONE FORM TERMINATES") and name not in differ:
                differ.append(name)
            if outcome == "identical":
                image = new_runs[0]
                if target in declared:
                    detail = "clear-only" if image == CLEAR * (READBACK_BYTES // len(CLEAR)) else "non-clear"
                    outcome = f"identical MRT{target}: {detail}"
                else:
                    detail = "nonempty" if any(image) else "empty"
                    outcome = f"identical MRT0 clear-difference mask: {detail}"
                if target == 0:
                    step = len(CLEAR) if target in declared else 1
                    res["identical and not all one colour"] += len(
                        set(image[i:i + step] for i in range(0, len(image), step))) > 1
            elif a.all_attachments:
                outcome = f"MRT{target}: {outcome}"
            labels.append(outcome)
        outcome = "; ".join(labels)
        unbound = sorted(set(declared) - set(observed))
        if unbound:
            outcome += "; unobserved " + ", ".join(f"MRT{n}" for n in unbound)
        note(name, outcome)
    print(dict(res))
    for name in differ[:8]:
        print("   differs:", name[:70])
    write_outcomes(a.out, outcomes)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
