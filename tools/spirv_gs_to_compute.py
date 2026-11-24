#!/usr/bin/env python3
"""Lower a translated geometry shader to a compute "capture" pass, for GPUs without a geometry stage (SHADER-159).

Apple GPUs (Metal, MoltenVK) have no geometry shaders. A legacy GS can still run there as a compute pass that
writes the vertices it emits into a buffer, followed by an ordinary draw of that buffer. This rewrites a module made
by tools/rdna2spirv.py (stage `Geometry`) into a GLCompute module with one invocation per input primitive:

- the per-vertex inputs (`BuiltIn Position`, `Location N`: Input arrays of vec4) are read from the input buffer
  (set 1, binding 0), `{ uint primitive_count; uint pad[3]; vec4 data[]; }`: vertex v of primitive p has its
  inputs at `data[(p * VERTS_IN + v) * NUM_INPUTS + slot]`;
- the vec4 outputs become private variables, and every `OpEmitVertex` copies them to the record buffer
  (set 1, binding 1), `{ vec4 rec[]; }`: vertex n of primitive p at `rec[(p * MAX_VERTS + n) * NUM_OUTPUTS + slot]`;
- the meta buffer (set 1, binding 2), `{ uint meta[]; }`, gets `meta[p * (MAX_VERTS + 1)]` = the vertices
  primitive p emitted and `meta[p * (MAX_VERTS + 1) + 1 + n]` = 1 when `OpEndPrimitive` ends a strip after vertex n
  (0 otherwise).

Slots are ordered Position first, then by location; `--layout` writes them with the sizes. `PrimitiveId` reads give
p, `InvocationId` reads 0 (only `Invocations 1` is supported). The workgroup is 64x1x1; invocations past
`primitive_count` return at once. Push constants and set 0 are untouched. The draw afterwards turns each strip into
triangles (or lines, points) in emission order.

Usage: spirv_gs_to_compute.py in.spv out.spv [--layout out.json]
Needs spirv-dis and spirv-as (SPIRV-Tools) on PATH, or SPIRV_DIS / SPIRV_AS.
"""
import json
import os
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

LOCAL_SIZE = 64
IN_VERTS = {"InputPoints": 1, "InputLines": 2, "InputLinesAdjacency": 4, "Triangles": 3,
            "InputTrianglesAdjacency": 6}
OUT_TOPOLOGY = {"OutputPoints": "points", "OutputLineStrip": "line_strip", "OutputTriangleStrip": "triangle_strip"}
KEEP_BUILTINS = ("SubgroupLocalInvocationId", "SubgroupSize", "NumSubgroups", "SubgroupId")
FLOAT_CONTROL_MODES = {"DenormPreserve", "DenormFlushToZero", "SignedZeroInfNanPreserve",
                       "RoundingModeRTE", "RoundingModeRTZ"}


class Unsupported(Exception):
    pass


def parse(text):
    """defs: id -> (opcode, operands) for '%id = OpX ...' lines; deco: id -> its OpDecorate operand strings."""
    defs, deco = {}, {}
    for ln in text.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)(.*)$", ln)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split())
        m = re.match(r"\s*OpDecorate (%\w+) (.*)$", ln)
        if m:
            deco.setdefault(m.group(1), []).append(m.group(2))
    return defs, deco


def find_type(defs, op, operands):
    return next((r for r, d in defs.items() if d == (op, operands)), None)


def layout_of(text):
    """The shader's inputs, outputs and sizes; raises Unsupported for what the rewrite can't handle."""
    defs, deco = parse(text)
    ep = re.search(r"^\s*OpEntryPoint Geometry (%\w+) ", text, re.M)
    if not ep:
        raise Unsupported("not a geometry shader")
    main = ep.group(1)
    modes = re.findall(rf"^\s*OpExecutionMode {re.escape(main)} (.*)$", text, re.M)
    modes = [m.split() for m in modes]
    verts_in = next((IN_VERTS[m[0]] for m in modes if m[0] in IN_VERTS), None)
    topology = next((OUT_TOPOLOGY[m[0]] for m in modes if m[0] in OUT_TOPOLOGY), None)
    max_verts = next((int(m[1]) for m in modes if m[0] == "OutputVertices"), None)
    invocations = next((int(m[1]) for m in modes if m[0] == "Invocations"), 1)
    if None in (verts_in, topology, max_verts):
        raise Unsupported(f"execution modes {modes}")
    if invocations != 1:
        raise Unsupported(f"Invocations {invocations}")
    f32 = find_type(defs, "OpTypeFloat", ["32"])
    v4 = find_type(defs, "OpTypeVector", [f32, "4"])

    def slot(v):
        ds = deco.get(v, [])
        if "BuiltIn Position" in ds:
            return (-1, "position")
        loc = next((int(d.split()[1]) for d in ds if d.startswith("Location ")), None)
        if loc is None:
            raise Unsupported(f"{v}: per-vertex variable without Position or Location ({ds})")
        return (loc, f"location {loc}")

    inputs, outputs, builtins = [], [], {}
    for v, (op, ops) in defs.items():
        if op != "OpVariable" or ops[1:2] not in (["Input"], ["Output"]):
            continue
        bi = next((d.split()[1] for d in deco.get(v, []) if d.startswith("BuiltIn ")), None)
        if bi in KEEP_BUILTINS or bi in ("PrimitiveId", "InvocationId"):
            builtins[v] = bi
            continue
        pointee = defs[ops[0]][1][1]
        if ops[1] == "Input":
            t = defs.get(pointee, (None, []))
            if t[0] != "OpTypeArray" or t[1][0] != v4:
                raise Unsupported(f"{v}: input isn't an array of vec4")
            inputs.append(v)
        elif pointee == v4:
            outputs.append(v)
        else:
            raise Unsupported(f"{v}: output isn't a vec4")
    inputs.sort(key=slot)
    outputs.sort(key=slot)
    return {"main": main, "verts_in": verts_in, "max_verts": max_verts, "topology": topology,
            "inputs": [slot(v)[1] for v in inputs], "outputs": [slot(v)[1] for v in outputs],
            "input_vars": inputs, "output_vars": outputs, "builtins": builtins}


def rewrite(text: str, *, version: tuple[int, int] = (1, 5)) -> tuple[str, dict[str, int | str | list[str]]]:
    """Return (compute module text, layout dict for --layout)."""
    lay = layout_of(text)
    defs, _ = parse(text)
    main, ins, outs, builtins = lay["main"], lay["input_vars"], lay["output_vars"], lay["builtins"]
    uint = find_type(defs, "OpTypeInt", ["32", "0"])
    f32 = find_type(defs, "OpTypeFloat", ["32"])
    v4 = find_type(defs, "OpTypeVector", [f32, "4"])
    if not (uint and v4):
        raise Unsupported("module has no uint or vec4 type")
    v3u = find_type(defs, "OpTypeVector", [uint, "3"]) or "%gsc_v3u"
    boolt = find_type(defs, "OpTypeBool", []) or "%gsc_bool"
    float_ptrs = {r for r, (op, ops) in defs.items() if op == "OpTypePointer" and ops[1] == f32}
    nin, nout, mv, vin = len(ins), len(outs), lay["max_verts"], lay["verts_in"]

    def k(v):
        return f"%gsc_k{v}"

    # Before SPIR-V 1.3, storage buffers use Uniform with BufferBlock in core SPIR-V.
    buffer_storage, buffer_block = ("StorageBuffer", "Block") if version >= (1, 3) else ("Uniform", "BufferBlock")
    decos = ["OpDecorate %gsc_gid BuiltIn GlobalInvocationId",
             "OpDecorate %gsc_rt_v4 ArrayStride 16", "OpDecorate %gsc_rt_u ArrayStride 4",
             "OpDecorate %gsc_pad ArrayStride 4",
             f"OpDecorate %gsc_in_t {buffer_block}", "OpMemberDecorate %gsc_in_t 0 Offset 0",
             "OpMemberDecorate %gsc_in_t 1 Offset 4", "OpMemberDecorate %gsc_in_t 2 Offset 16",
             f"OpDecorate %gsc_rec_t {buffer_block}", "OpMemberDecorate %gsc_rec_t 0 Offset 0",
             f"OpDecorate %gsc_meta_t {buffer_block}", "OpMemberDecorate %gsc_meta_t 0 Offset 0",
             "OpDecorate %gsc_in DescriptorSet 1", "OpDecorate %gsc_in Binding 0", "OpDecorate %gsc_in NonWritable",
             "OpDecorate %gsc_rec DescriptorSet 1", "OpDecorate %gsc_rec Binding 1",
             "OpDecorate %gsc_meta DescriptorSet 1", "OpDecorate %gsc_meta Binding 2"]
    types = ([f"{v3u} = OpTypeVector {uint} 3"] if v3u == "%gsc_v3u" else []) + \
            ([f"{boolt} = OpTypeBool"] if boolt == "%gsc_bool" else [])
    types += [f"{k(v)} = OpConstant {uint} {v}" for v in sorted({0, 1, 2, 3, vin, nin, nout, mv, mv + 1}
                                                              | set(range(max(nin, nout))))]
    types += [f"%gsc_ptr_in_v3u = OpTypePointer Input {v3u}",
              f"%gsc_rt_v4 = OpTypeRuntimeArray {v4}", f"%gsc_rt_u = OpTypeRuntimeArray {uint}",
              f"%gsc_pad = OpTypeArray {uint} {k(3)}",
              f"%gsc_in_t = OpTypeStruct {uint} %gsc_pad %gsc_rt_v4",
              "%gsc_rec_t = OpTypeStruct %gsc_rt_v4", "%gsc_meta_t = OpTypeStruct %gsc_rt_u",
              f"%gsc_ptr_in_t = OpTypePointer {buffer_storage} %gsc_in_t",
              f"%gsc_ptr_rec_t = OpTypePointer {buffer_storage} %gsc_rec_t",
              f"%gsc_ptr_meta_t = OpTypePointer {buffer_storage} %gsc_meta_t",
              f"%gsc_ptr_sb_v4 = OpTypePointer {buffer_storage} {v4}",
              f"%gsc_ptr_sb_u = OpTypePointer {buffer_storage} {uint}",
              f"%gsc_ptr_sb_f = OpTypePointer {buffer_storage} {f32}",
              f"%gsc_ptr_pv_v4 = OpTypePointer Private {v4}", f"%gsc_ptr_pv_f = OpTypePointer Private {f32}",
              f"%gsc_ptr_pv_u = OpTypePointer Private {uint}",
              "%gsc_gid = OpVariable %gsc_ptr_in_v3u Input",
              f"%gsc_in = OpVariable %gsc_ptr_in_t {buffer_storage}",
              f"%gsc_rec = OpVariable %gsc_ptr_rec_t {buffer_storage}",
              f"%gsc_meta = OpVariable %gsc_ptr_meta_t {buffer_storage}",
              f"%gsc_n = OpVariable %gsc_ptr_pv_u Private {k(0)}"]
    out_vars = {v: f"%gsc_out{i}" for i, v in enumerate(outs)}
    types += [f"{nv} = OpVariable %gsc_ptr_pv_v4 Private" for nv in out_vars.values()]
    in_slot = {v: i for i, v in enumerate(ins)}
    gone = set(ins) | set(outs) | {v for v, b in builtins.items() if b in ("PrimitiveId", "InvocationId")}
    capture_interface = ["%gsc_gid"]
    # Before SPIR-V 1.4, the interface only lists Input and Output variables.
    if version >= (1, 4):
        capture_interface += ["%gsc_in", "%gsc_rec", "%gsc_meta", "%gsc_n"] + list(out_vars.values())

    count = 0

    def tmp():
        nonlocal count
        count += 1
        return f"%gsc_t{count}"

    out = []
    entry_label = None
    cur_block = None
    emit_split = set()
    block_tail = {}
    main_start = None
    state = "header"        # header -> main_entry -> main_vars -> main -> other
    for ln in text.splitlines():
        s = ln.strip()
        if s == "OpCapability Geometry":
            continue
        m = re.match(r"OpEntryPoint Geometry (%\w+) (\"[^\"]*\")(.*)$", s)
        if m:
            iface = [v for v in m.group(3).split() if v not in gone] + capture_interface
            out += [f"OpEntryPoint GLCompute {m.group(1)} {m.group(2)} {' '.join(iface)}",
                    f"OpExecutionMode {m.group(1)} LocalSize {LOCAL_SIZE} 1 1"]
            continue
        # Floating-point controls apply to compute too.
        m = re.match(rf"OpExecutionMode {re.escape(main)} (\w+)\b", s)
        if m and m.group(1) not in FLOAT_CONTROL_MODES:
            continue
        m = re.match(r"Op(?:Decorate|MemberDecorate|Name) (%\w+)\b", s)
        if m and m.group(1) in gone:
            continue
        if re.match(r"OpGroupDecorate\s", s):
            instruction, sep, comment = s.partition(";")
            opcode, group, *targets = instruction.split()
            live_targets = [target for target in targets if target not in gone]
            if live_targets == targets:
                out.append(ln)
            elif live_targets:
                rewritten = " ".join((opcode, group, *live_targets))
                out.append(rewritten + f" {sep}{comment}" if sep else rewritten)
            continue
        m = re.match(r"(%\w+) = (Op\w+)", s)
        if decos and m and m.group(2).startswith("OpType"):
            out += decos            # decorations end where the first type begins
            decos = []
        if m and m.group(2) == "OpVariable" and m.group(1) in gone:
            continue
        if m and m.group(2) == "OpFunction":
            if types:
                out += types
                types = []
            if m.group(1) == main:
                state = "main_entry"
                main_start = len(out)
            out.append(ln)
            continue
        if s == "OpFunctionEnd":
            if state in ("main", "main_vars") and main_start is not None and block_tail:
                # Any block split by the primitive-count check or by OpEmitVertex now terminates in its final
                # merge label; update every OpPhi predecessor (including loop-header back-edges) to that label.
                for idx in range(main_start, len(out)):
                    m_phi = re.match(r"^(\s*%\w+\s*=\s*OpPhi\s+%\w+)\s+(.*)$", out[idx])
                    if m_phi:
                        ops = m_phi.group(2).split()
                        ops[1::2] = [block_tail.get(lbl, lbl) for lbl in ops[1::2]]
                        out[idx] = f"{m_phi.group(1)} {' '.join(ops)}"
            state = "other"
            cur_block = None
        if state == "main_entry" and re.match(r"%\w+ = OpLabel$", s):
            state = "main_vars"
            entry_label = s.split()[0]
            cur_block = entry_label
            out.append(ln)
            continue
        # OpLine and OpNoLine can be intermixed with the entry block's local variables.
        if state == "main_vars" and not re.match(r"(?:%\w+ = OpVariable |OpLine |OpNoLine$)", s):
            # after the entry block's variables: p = GlobalInvocationId.x; return when p >= primitive_count
            state = "main"
            out += [f"%gsc_gidv = OpLoad {v3u} %gsc_gid", f"%gsc_prim = OpCompositeExtract {uint} %gsc_gidv 0",
                    f"%gsc_cntp = OpAccessChain %gsc_ptr_sb_u %gsc_in {k(0)}", f"%gsc_cnt = OpLoad {uint} %gsc_cntp",
                    f"%gsc_past = OpUGreaterThanEqual {boolt} %gsc_prim %gsc_cnt",
                    "OpSelectionMerge %gsc_body None", "OpBranchConditional %gsc_past %gsc_ret %gsc_body",
                    "%gsc_ret = OpLabel", "OpReturn", "%gsc_body = OpLabel",
                    f"%gsc_inbase = OpIMul {uint} %gsc_prim {k(vin)}",
                    f"%gsc_recbase = OpIMul {uint} %gsc_prim {k(mv)}",
                    f"%gsc_metabase = OpIMul {uint} %gsc_prim {k(mv + 1)}"]
            block_tail[entry_label] = "%gsc_body"
        if state not in ("main", "main_vars"):
            out.append(ln)
            continue
        # ---- the body of main ----
        m_lbl = re.match(r"(%\w+) = OpLabel$", s)
        if m_lbl:
            cur_block = m_lbl.group(1)
            out.append(ln)
            continue
        m = re.match(r"(%\w+) = Op(?:InBounds)?AccessChain (%\w+) (%\w+) (.*)$", s)
        if m and m.group(3) in in_slot:
            idx = m.group(4).split()
            vtx = idx[0]
            if defs.get(vtx, (None, [None]))[1][:1] != [uint]:
                vtx = tmp()
                out.append(f"{vtx} = OpBitcast {uint} {idx[0]}")
            a, b, c = tmp(), tmp(), tmp()
            ptr = "%gsc_ptr_sb_f" if m.group(2) in float_ptrs else "%gsc_ptr_sb_v4"
            out += [f"{a} = OpIAdd {uint} %gsc_inbase {vtx}", f"{b} = OpIMul {uint} {a} {k(nin)}",
                    f"{c} = OpIAdd {uint} {b} {k(in_slot[m.group(3)])}",
                    " ".join([f"{m.group(1)} = OpAccessChain {ptr} %gsc_in {k(2)} {c}"] + idx[1:])]
            continue
        if m and m.group(3) in out_vars:
            ptr = "%gsc_ptr_pv_f" if m.group(2) in float_ptrs else "%gsc_ptr_pv_v4"
            out.append(f"{m.group(1)} = OpAccessChain {ptr} {out_vars[m.group(3)]} {m.group(4)}")
            continue
        m = re.match(r"(%\w+) = OpLoad (%\w+) (%\w+)(?:\s+.*)?$", s)
        if m and builtins.get(m.group(3)) in ("PrimitiveId", "InvocationId"):
            val = "%gsc_prim" if builtins[m.group(3)] == "PrimitiveId" else k(0)
            out.append(f"{m.group(1)} = OpBitcast {m.group(2)} {val}" if m.group(2) != uint
                       else f"{m.group(1)} = OpCopyObject {uint} {val}")
            continue
        for v in list(in_slot) + list(out_vars) + sorted(gone - set(in_slot) - set(out_vars)):
            if re.search(rf"{re.escape(v)}\b", s):
                m2 = re.match(rf"OpStore {re.escape(v)} (%\w+)(?:\s+.*)?$", s)
                if v in out_vars and m2:
                    s = f"OpStore {out_vars[v]} {m2.group(1)}"
                    break
                raise Unsupported(f"unhandled use of {v}: {s}")
        if s == "OpEmitVertex":
            # Guard stores and counter increment on n < OutputVertices so over-emitting drops extra vertices
            # instead of overflowing into primitive p+1's record/meta slots; %gsc_n stays <= mv for OpEndPrimitive
            # and OpReturn as well (SHADER-617).
            n, ok, lbl_do, lbl_end = tmp(), tmp(), tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n",
                    f"{ok} = OpULessThan {boolt} {n} {k(mv)}",
                    f"OpSelectionMerge {lbl_end} None",
                    f"OpBranchConditional {ok} {lbl_do} {lbl_end}",
                    f"{lbl_do} = OpLabel"]
            if out_vars:
                a = tmp()
                out.append(f"{a} = OpIAdd {uint} %gsc_recbase {n}")
                for i, nv in enumerate(out_vars.values()):
                    b, c, val, p = tmp(), tmp(), tmp(), tmp()
                    out += [f"{b} = OpIMul {uint} {a} {k(nout)}", f"{c} = OpIAdd {uint} {b} {k(i)}",
                            f"{val} = OpLoad {v4} {nv}", f"{p} = OpAccessChain %gsc_ptr_sb_v4 %gsc_rec {k(0)} {c}",
                            f"OpStore {p} {val}"]
            d, e, q, n1 = tmp(), tmp(), tmp(), tmp()
            out += [f"{d} = OpIAdd {uint} %gsc_metabase {n}", f"{e} = OpIAdd {uint} {d} {k(1)}",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} {e}", f"OpStore {q} {k(0)}",
                    f"{n1} = OpIAdd {uint} {n} {k(1)}", f"OpStore %gsc_n {n1}",
                    f"OpBranch {lbl_end}",
                    f"{lbl_end} = OpLabel"]
            if cur_block is not None:
                block_tail[cur_block] = lbl_end
                emit_split.add(cur_block)
            continue
        if s == "OpEndPrimitive":
            # flag the last emitted vertex: meta[base + 1 + (n - 1)]; with none emitted this writes the count slot,
            # which the final store overwrites
            n, d, q = tmp(), tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n", f"{d} = OpIAdd {uint} %gsc_metabase {n}",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} {d}", f"OpStore {q} {k(1)}"]
            continue
        if s == "OpReturn":
            n, q = tmp(), tmp()
            out += [f"{n} = OpLoad {uint} %gsc_n",
                    f"{q} = OpAccessChain %gsc_ptr_sb_u %gsc_meta {k(0)} %gsc_metabase", f"OpStore {q} {n}"]
        if s.startswith("OpLoopMerge ") and cur_block in emit_split:
            # Splitting a loop header with OpSelectionMerge moves OpLoopMerge into the merge block while
            # back-edges still target the original header label, violating SPIR-V structured CFG rules.
            raise Unsupported("OpEmitVertex in a loop header before OpLoopMerge")
        if s in ("OpKill", "OpTerminateInvocation", "OpReturnValue"):
            raise Unsupported(s)
        out.append(s)
    public = {key: lay[key] for key in ("verts_in", "max_verts", "topology", "inputs", "outputs")}
    public["local_size"] = LOCAL_SIZE
    return "\n".join(out) + "\n", public


USAGE = "\n".join(__doc__.strip().splitlines()[-2:])


def main(argv):
    args = argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    lay_path = None
    if "--layout" in args:
        i = args.index("--layout")
        lay_path = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    paths = [("input", args[0]), ("SPIR-V output", args[1])]
    if lay_path:
        paths.append(("--layout output", lay_path))
    try:
        resolved = [(name, Path(path).resolve()) for name, path in paths]
        for (left_name, left), (right_name, right) in combinations(resolved, 2):
            if left == right or (left.exists() and right.exists() and left.samefile(right)):
                print(f"{left_name} and {right_name} refer to the same file: {left}", file=sys.stderr)
                return 2
    except (OSError, RuntimeError, ValueError) as e:
        print(f"cannot check input/output paths: {e}", file=sys.stderr)
        return 2
    dis = os.environ.get("SPIRV_DIS", "spirv-dis")
    asm = os.environ.get("SPIRV_AS", "spirv-as")
    text = subprocess.run([dis, "--raw-id", args[0]], check=True, capture_output=True, text=True).stdout
    ver = re.search(r"; Version: (\d+)\.(\d+)", text)
    version = (int(ver[1]), int(ver[2])) if ver else (1, 5)
    try:
        new, lay = rewrite(text, version=version)
    except Unsupported as e:
        print(f"{args[0]}: not supported: {e}", file=sys.stderr)
        return 1
    target = f"spv{version[0]}.{version[1]}" if ver else "vulkan1.2"
    subprocess.run([asm, "--target-env", target, "-", "-o", args[1]], input=new, check=True, text=True)
    if lay_path:
        with open(lay_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(lay, f, indent=1)
    print(json.dumps(lay))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))
