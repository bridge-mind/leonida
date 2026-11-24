#!/usr/bin/env python3
"""Vertex shaders that draw what rdna2spirv.py --mesh-compute mesh shaders wrote.

On a device without VK_EXT_mesh_shader, `rdna2spirv.py --mesh-compute` turns each mesh shader into a compute shader
with the same workgroup size. Each workgroup writes its vertex and primitive counts, its vertices and its
primitives into a slot of an output buffer. The layout is the shader's "mesh_compute" entry in index.json:
rdna2spirv.mesh_compute_layout. The buffer's device address is in push-constant dwords 36..37, for both the
compute shader and this vertex shader.

The draw that follows is non-indexed: vkCmdDraw(groups * max_primitives * 3), where groups is the number of
workgroups dispatched. Vertex v belongs to workgroup v / (3 * max_primitives), primitive (v / 3) % max_primitives,
corner v % 3. A primitive past the workgroup's count, or one marked culled, gets the same position outside the
clip volume at all three corners, so it is clipped away. Otherwise the corner's vertex index (clamped to
max_vertices - 1) selects the vertex whose position, clip and cull distances and params this shader outputs. The
params go out at the locations the mesh shader used, so the pixel shader is unchanged. With rdna2spirv.py
--f16-interp-fallback the layout also has "f16_halves" ({param: [low location, high location]}), and each of those
params' f16 halves goes out decoded to f32 at those locations too, as a translated VS exports them.

    python -I tools/mesh_compute_vs.py <spirv dir>    # <name>.mesh_vs.spv per --mesh-compute MS in index.json

Standard library only; build_for_index assembles and validates with spirv-as / spirv-val from PATH.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SUFFIX = ".mesh_vs"   # <ms name>.mesh_vs.spvasm / .spv next to the mesh shader
USER_SGPRS = 32       # push constants: s0..s31, heap base, code base, then the output address (rdna2spirv.py)


F16_HALF_TOP = 29


def pair_f16_halves(layout, ps_f16_halves, ps_raw_inputs=(), top=F16_HALF_TOP):
    """Choose non-colliding `f16_halves` ({str(param): [lo, hi]}) for a (--mesh-compute, PS) pair (SHADER-378).

    Keeps the PS's own `[lo, hi]` locations whenever every requested param is exported by `layout["params"]` and no
    half location collides with any producer param or exceeds `top`. Otherwise packs all requested pairs downward
    from `top` into locations disjoint from `producer_params | ps_raw_inputs`.
    """
    producer_params = {int(k) for k in layout.get("params", {})}
    ps_pairs = {int(k): [int(v[0]), int(v[1])] for k, v in ps_f16_halves.items()}
    if not ps_pairs:
        return {}
    missing = sorted(p for p in ps_pairs if p not in producer_params)
    if missing:
        raise ValueError(f"mesh_compute_vs: producer does not export f16_halves param(s) {missing}")
    if all(0 <= lo <= top and 0 <= hi <= top and lo != hi and lo not in producer_params and hi not in producer_params
           for lo, hi in ps_pairs.values()):
        used_halves = [loc for pair in ps_pairs.values() for loc in pair]
        if len(set(used_halves)) == len(used_halves):
            return {str(k): list(v) for k, v in sorted(ps_pairs.items())}
    reserved = producer_params | {int(x) for x in ps_raw_inputs}
    free = [loc for loc in range(top, -1, -1) if loc not in reserved]
    need = 2 * len(ps_pairs)
    if len(free) < need:
        raise ValueError(f"mesh_compute_vs: need {need} decoded-half locations, only {len(free)} free in 0..{top}")
    return {str(p): [free[2 * k + 1], free[2 * k]] for k, p in enumerate(sorted(ps_pairs))}


def vs_spvasm(layout, f16_halves=None):
    """SPIR-V assembly of the vertex shader for one --mesh-compute layout (index.json "mesh_compute").

    Optional `f16_halves` ({param: [low_loc, high_loc]}) overrides `layout["f16_halves"]` for per-pipeline pairing
    (SHADER-378), so a producer with many params only exports the decoded halves its paired PS actually reads.
    """
    lay = dict(layout)
    if f16_halves is not None:
        lay["f16_halves"] = f16_halves
    params = sorted((int(k), int(v)) for k, v in lay["params"].items())
    maxv, maxp = int(lay["max_vertices"]), int(lay["max_primitives"])
    clip, cull = int(lay["clip_count"]), int(lay["cull_count"])
    consts = {}

    def c(v):
        consts.setdefault(int(v), f"%c{int(v)}")
        return consts[int(v)]

    ulongs = {}

    def cl(v):
        ulongs.setdefault(int(v), f"%cl{int(v)}")
        return ulongs[int(v)]

    body = []
    n = [0]

    def tmp(p="t"):
        n[0] += 1
        return f"%{p}{n[0]}"

    def load_word(base64, off):
        """uint at slot byte offset `off` (constant) from the ulong address `base64`."""
        a, p, v = tmp("a"), tmp("p"), tmp("w")
        body.extend([f"{a} = OpIAdd %ulong {base64} {cl(off)}" if off else f"{a} = OpCopyObject %ulong {base64}",
                     f"{p} = OpConvertUToPtr %_ptr_PSB_uint {a}", f"{v} = OpLoad %uint {p} Aligned 4"])
        return v

    def load_floats(base64, off, count):
        out = []
        for i in range(count):
            w = load_word(base64, off + 4 * i)
            f = tmp("f")
            body.append(f"{f} = OpBitcast %float {w}")
            out.append(f)
        return out

    c(0), c(1), c(3), c(5), c(6)
    if clip:
        c(clip)
    if cull:
        c(cull)
    body += [
        "%vid = OpLoad %int %vertex_index", "%vidu = OpBitcast %uint %vid",
        f"%per = OpCopyObject %uint {c(3 * maxp)}",
        "%wg = OpUDiv %uint %vidu %per", "%rem = OpUMod %uint %vidu %per",
        f"%prim = OpUDiv %uint %rem {c(3)}", f"%corner = OpUMod %uint %rem {c(3)}",
        f"%alo_p = OpAccessChain %_ptr_PC_uint %pc {c(5)}", f"%ahi_p = OpAccessChain %_ptr_PC_uint %pc {c(6)}",
        "%alo = OpLoad %uint %alo_p", "%ahi = OpLoad %uint %ahi_p",
        "%alo64 = OpUConvert %ulong %alo", "%ahi64 = OpUConvert %ulong %ahi",
        f"%ahis = OpShiftLeftLogical %ulong %ahi64 {cl(32)}", "%addr = OpBitwiseOr %ulong %ahis %alo64",
        "%wg64 = OpUConvert %ulong %wg", f"%soff = OpIMul %ulong %wg64 {cl(lay['slot_bytes'])}",
        "%slot = OpIAdd %ulong %addr %soff"]
    pcount = load_word("%slot", 4)
    body += [f"%inrange = OpULessThan %bool %prim {pcount}",
             f"%pofs = OpIMul %uint %prim {c(lay['primitive_stride'])}", "%pofs64 = OpUConvert %ulong %pofs",
             f"%pbase0 = OpIAdd %ulong %slot {cl(lay['primitives'])}", "%pbase = OpIAdd %ulong %pbase0 %pofs64",
             f"%cofs = OpIMul %uint %corner {c(4)}", "%cofs64 = OpUConvert %ulong %cofs",
             "%iaddr = OpIAdd %ulong %pbase %cofs64", "%iptr = OpConvertUToPtr %_ptr_PSB_uint %iaddr",
             "%idx0 = OpLoad %uint %iptr Aligned 4",
             f"%idx = OpExtInst %uint %glsl UMin %idx0 {c(maxv - 1)}"]
    culled_w = load_word("%pbase", 12)
    body += [f"%culled = OpINotEqual %bool {culled_w} {c(0)}", "%notculled = OpLogicalNot %bool %culled",
             "%valid = OpLogicalAnd %bool %inrange %notculled",
             f"%vofs = OpIMul %uint %idx {c(lay['vertex_stride'])}", "%vofs64 = OpUConvert %ulong %vofs",
             f"%vbase0 = OpIAdd %ulong %slot {cl(lay['vertices'])}", "%vbase = OpIAdd %ulong %vbase0 %vofs64"]
    pos = load_floats("%vbase", 0, 4)
    body += [f"%pos = OpCompositeConstruct %v4float {' '.join(pos)}",
             "%pos_out = OpSelect %v4float %valid %pos %outside", "OpStore %out_pos %pos_out"]
    if clip:
        vals = load_floats("%vbase", lay["clip_offset"], clip)
        body += [f"%clipv = OpCompositeConstruct %arr_clip {' '.join(vals)}", "OpStore %out_clip %clipv"]
    if cull:
        vals = load_floats("%vbase", lay["cull_offset"], cull)
        body += [f"%cullv = OpCompositeConstruct %arr_cull {' '.join(vals)}", "OpStore %out_cull %cullv"]
    for loc, off in params:
        vals = load_floats("%vbase", off, 4)
        body += [f"%par{loc} = OpCompositeConstruct %v4float {' '.join(vals)}", f"OpStore %out{loc} %par{loc}"]
    halves = sorted((int(k), [int(x) for x in v]) for k, v in lay.get("f16_halves", {}).items())
    offsets = dict(params)
    used_locs = set(offsets)
    for loc, hlocs in halves:
        if loc not in offsets:
            raise ValueError(f"mesh_compute_vs: f16_halves param {loc} is not in layout params")
        if len(hlocs) != 2 or hlocs[0] == hlocs[1] or hlocs[0] < 0 or hlocs[1] < 0:
            raise ValueError(f"mesh_compute_vs: invalid f16_halves locations {hlocs} for param {loc}")
        for hloc in hlocs:
            if hloc in used_locs:
                raise ValueError(f"mesh_compute_vs: f16_halves location {hloc} collides with another output")
            used_locs.add(hloc)
    for loc, hlocs in halves:      # --f16-interp-fallback: each channel's two f16 halves decoded to f32
        parts = ([], [])
        for i in range(4):
            w, h2 = load_word("%vbase", offsets[loc] + 4 * i), tmp("h")
            body.append(f"{h2} = OpExtInst %v2float %glsl UnpackHalf2x16 {w}")
            for half in (0, 1):
                e = tmp("e")
                body.append(f"{e} = OpCompositeExtract %float {h2} {half}")
                parts[half].append(e)
        for half, hloc in enumerate(hlocs):
            body += [f"%hv{hloc} = OpCompositeConstruct %v4float {' '.join(parts[half])}",
                     f"OpStore %out{hloc} %hv{hloc}"]
    hout = [hloc for _, hlocs in halves for hloc in hlocs]
    body += ["OpReturn", "OpFunctionEnd"]

    outs = ["%out_pos"] + (["%out_clip"] if clip else []) + (["%out_cull"] if cull else []) + \
        [f"%out{loc}" for loc, _ in params] + [f"%out{h}" for h in hout]
    caps = ["Shader", "Int64", "PhysicalStorageBufferAddresses"] + (["ClipDistance"] if clip else []) + \
        (["CullDistance"] if cull else [])
    head = [f"OpCapability {x}" for x in caps] + [
        'OpExtension "SPV_KHR_physical_storage_buffer"',
        '%glsl = OpExtInstImport "GLSL.std.450"',
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450",
        f'OpEntryPoint Vertex %main "main" %vertex_index %pc {" ".join(outs)}',
        "OpDecorate %vertex_index BuiltIn VertexIndex",
        "OpDecorate %out_pos BuiltIn Position"]
    if clip:
        head.append("OpDecorate %out_clip BuiltIn ClipDistance")
    if cull:
        head.append("OpDecorate %out_cull BuiltIn CullDistance")
    head += [f"OpDecorate %out{loc} Location {loc}" for loc in [p for p, _ in params] + hout]
    head += ["OpDecorate %arr_ud ArrayStride 4", "OpDecorate %PC Block", "OpMemberDecorate %PC 0 Offset 0"]
    head += [f"OpMemberDecorate %PC {k} Offset {4 * USER_SGPRS + 4 * (k - 1)}" for k in range(1, 7)]
    head += ["OpDecorate %rt_uint ArrayStride 4"]
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool",
             "%uint = OpTypeInt 32 0", "%int = OpTypeInt 32 1", "%ulong = OpTypeInt 64 0",
             "%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4", "%v2float = OpTypeVector %float 2"]
    types += [f"{name} = OpConstant %uint {v}" for v, name in sorted(consts.items())]
    types += [f"{name} = OpConstant %ulong {v}" for v, name in sorted(ulongs.items())]
    types += [f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
              "%PC = OpTypeStruct %arr_ud %uint %uint %uint %uint %uint %uint",
              "%_ptr_PC = OpTypePointer PushConstant %PC", "%_ptr_PC_uint = OpTypePointer PushConstant %uint",
              "%rt_uint = OpTypeRuntimeArray %uint",
              "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint",
              "%_ptr_In_int = OpTypePointer Input %int", "%_ptr_Out_v4 = OpTypePointer Output %v4float",
              "%f1 = OpConstant %float 1", "%f2 = OpConstant %float 2",
              "%outside = OpConstantComposite %v4float %f2 %f2 %f2 %f1"]
    if clip:
        types += ["%arr_clip = OpTypeArray %float " + c(clip), "%_ptr_Out_clip = OpTypePointer Output %arr_clip"]
    if cull:
        types += ["%arr_cull = OpTypeArray %float " + c(cull), "%_ptr_Out_cull = OpTypePointer Output %arr_cull"]
    gvars = ["%vertex_index = OpVariable %_ptr_In_int Input", "%pc = OpVariable %_ptr_PC PushConstant",
             "%out_pos = OpVariable %_ptr_Out_v4 Output"]
    if clip:
        gvars.append("%out_clip = OpVariable %_ptr_Out_clip Output")
    if cull:
        gvars.append("%out_cull = OpVariable %_ptr_Out_cull Output")
    gvars += [f"%out{loc} = OpVariable %_ptr_Out_v4 Output" for loc in [p for p, _ in params] + hout]
    fn = ["%main = OpFunction %void None %fn", "%entry = OpLabel"]
    return "\n".join(head + types + gvars + fn + body) + "\n"


def write_for_index(spirv_dir):
    """Write <name>.mesh_vs.spvasm for every entry of <spirv_dir>/index.json with a "mesh_compute" layout."""
    index_path = os.path.join(spirv_dir, "index.json")
    if not os.path.isfile(index_path):
        return []
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (ValueError, RecursionError) as e:
        raise RuntimeError(f"{index_path}: invalid JSON ({e})") from None
    if not isinstance(index, dict):
        raise RuntimeError(f"{index_path}: expected a shader index object")
    written = []
    for name, meta in sorted(index.items()):
        lay = meta.get("mesh_compute") if isinstance(meta, dict) else None
        if not lay:
            continue
        try:
            asm = vs_spvasm(lay)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as e:
            raise RuntimeError(f"{index_path}: invalid mesh_compute layout for {name!r}: {e}") from None
        path = os.path.join(spirv_dir, name + SUFFIX + ".spvasm")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(asm)
        written.append(path)
    return written


def tool_path(name):
    """The SPIR-V tool's full path, or RuntimeError naming it: subprocess's own FileNotFoundError says only
    "the system cannot find the file specified", which does not say which tool is missing."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH (it ships with SPIRV-Tools, e.g. VulkanSDK/Bin)")
    return path


def build_for_index(spirv_dir):
    """write_for_index, then assemble to <name>.mesh_vs.spv and validate (Vulkan 1.2, as for the translator).
    Returns the .spv paths; raises RuntimeError with the tool output on failure."""
    steps = [("spirv-as", tool_path("spirv-as")), ("spirv-val", tool_path("spirv-val"))]
    built = []
    for asm in write_for_index(spirv_dir):
        spv = asm[: -len(".spvasm")] + ".spv"
        with tempfile.TemporaryDirectory(prefix=".mesh-vs-", dir=os.path.dirname(spv)) as staging:
            staged_spv = os.path.join(staging, os.path.basename(spv))
            for name, tool in steps:
                args = [asm, "-o", staged_spv] if name == "spirv-as" else [staged_spv]
                cmd = [tool, "--target-env", "vulkan1.2"] + args
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"{name} failed for {asm}:\n{r.stdout}{r.stderr}")
            os.replace(staged_spv, spv)
        built.append(spv)
    return built


def main(argv):
    if argv[1:] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    if not os.path.isdir(argv[1]):
        print(f"ERROR: {argv[1]}: not a directory", file=sys.stderr)
        return 1
    try:
        built = build_for_index(argv[1])
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ERROR: {e.filename or argv[1]}: {e.strerror or e}", file=sys.stderr)
        return 1
    for path in built:
        print(path)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))
