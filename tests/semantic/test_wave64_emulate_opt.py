"""Unit and 64/128-lane workgroup execution tests for tools/wave64_emulate_opt.py (--wave64-emulate).

Verifies both exact optimizations introduced for Issue #884:
  1. Backward CFG SCC liveness: dead `scc_from_mask` / `scc_any_half` votes on lane-mask ops are elided
     under --wave64-emulate, while live SCC writes (direct, cross-block, loop-carried, carry-in) stay exact.
  2. Safe lane-local `s_cbranch_execz` / `s_cbranch_execnz` region folding + continuation chaining: single-entry,
     single-exit regions with no cross-lane ops, no barriers, no memory/LDS writes, no EXEC-widening, and no
     live-out scalars fold inline behind `if (%active)` (and chain single-predecessor continuations), keeping
     all 64 lanes of each wave in the same scheduler round.

Includes a strict workgroup SPIR-V interpreter with real OpControlBarrier uniformity checks, shared Workgroup
memory (%x_buf and %lds), 32-bit Workgroup atomics, OpSwitch, loops, and PhysicalStorageBuffer loads/stores so
the translated --wave64-emulate SPIR-V is executed bit-for-bit across 64-thread (1-wave) and 128-thread
(2-wave) workgroups and compared against an independent RDNA2 ISA reference.

Run from the repository root:  python -I tests/semantic/test_wave64_emulate_opt.py
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

rdna2spirv = support.rdna2spirv
wave64_emulate_opt = rdna2spirv.wave64_emulate_opt

MASK32 = 0xFFFFFFFF
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # raw buffer covering all offsets


# ---- Strict SPIR-V workgroup + barrier interpreter (for --wave64-emulate modules) ----
_SPV_LINE = re.compile(r"^(?:(%\w+)\s*=\s*)?(\w+)(?:\s+(.*))?$")
_SPV_IGNORED = frozenset({
    "OpCapability", "OpExtension", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint",
    "OpExecutionMode", "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction",
    "OpSelectionMerge", "OpLoopMerge", "OpFunctionEnd",
})


def _strip_comments(text):
    out = []
    for raw in text.splitlines():
        s = raw.split(";", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _s32(v):
    v &= MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def _parse_module(text):
    types = {}
    ptr_types = {}
    consts = {}
    id_type = {}
    globals_ = []
    funcs = {}
    cur_fn = None
    cur_params = []
    cur_body = []

    def zero_of(tid):
        k = types[tid]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return 0.0
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(zero_of(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [zero_of(k[1]) for _ in range(k[2])]
        if k[0] == "struct":
            return [zero_of(m) for m in k[1]]
        raise NotImplementedError(f"zero_of {k}")

    for line in _strip_comments(text):
        m = _SPV_LINE.match(line)
        if not m:
            raise AssertionError(f"cannot parse SPIR-V line: {line!r}")
        res, op, rest = m.groups()
        a = rest.split() if rest else []
        if cur_fn is not None:
            if op == "OpFunctionParameter":
                cur_params.append(res)
            elif op == "OpFunctionEnd":
                funcs[cur_fn] = (cur_params, cur_body)
                cur_fn = None
            else:
                cur_body.append((res, op, a, line))
            continue
        if op == "OpFunction":
            cur_fn = res
            cur_params = []
            cur_body = []
            continue
        if op in _SPV_IGNORED:
            continue
        if op == "OpTypeInt":
            types[res] = ("int", int(a[0]), int(a[1]))
        elif op == "OpTypeFloat":
            types[res] = ("float", int(a[0]))
        elif op == "OpTypeBool":
            types[res] = ("bool",)
        elif op == "OpTypeVector":
            types[res] = ("vec", a[0], int(a[1]))
        elif op == "OpTypeArray":
            types[res] = ("arr", a[0], int(consts[a[1]]))
        elif op == "OpTypeRuntimeArray":
            types[res] = ("rarr", a[0])
        elif op == "OpTypeStruct":
            types[res] = ("struct", a)
        elif op == "OpTypePointer":
            ptr_types[res] = (a[0], a[1])
        elif op == "OpConstant":
            k = types[a[0]]
            if k[0] == "int":
                consts[res] = int(a[1]) & ((1 << k[1]) - 1)
                id_type[res] = a[0]
            elif k[0] == "float":
                consts[res] = float(a[1])
                id_type[res] = a[0]
            else:
                raise NotImplementedError(f"OpConstant {k}")
        elif op == "OpConstantTrue":
            consts[res] = True
        elif op == "OpConstantFalse":
            consts[res] = False
        elif op == "OpConstantNull":
            consts[res] = zero_of(a[0])
        elif op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            assert a[1] == cls, f"storage class mismatch: {line}"
            globals_.append((res, cls, pointee, a[2] if len(a) > 2 else None))
        else:
            raise NotImplementedError(f"header op {op}: {line}")

    assert "%main" in funcs
    # Expand single-block helper OpFunctionCalls inline into %main's instruction stream for lockstep interpretation.
    body = []
    call_seq = 0
    for res, op, a, line in funcs["%main"][1]:
        if op != "OpFunctionCall":
            body.append((res, op, a, line))
            continue
        call_seq += 1
        ret_t, fn_name = a[0], a[1]
        call_args = a[2:]
        fn_params, fn_insns = funcs[fn_name]
        assert len(call_args) == len(fn_params)
        id_map = dict(zip(fn_params, call_args))
        for fres, fop, fa, fline in fn_insns:
            if fop == "OpLabel":
                continue
            mapped_a = [id_map.get(x, x) for x in fa]
            if fop == "OpReturnValue":
                body.append((res, "OpCopyObject", [ret_t, mapped_a[0]], f"{res} = OpCopyObject {ret_t} {mapped_a[0]}"))
                break
            new_res = f"%fc{call_seq}_{fres[1:]}" if fres else None
            if fres:
                id_map[fres] = new_res
            body.append((new_res, fop, mapped_a, fline))

    labels = {res: idx for idx, (res, op, _, _) in enumerate(body) if op == "OpLabel" and res}
    return {
        "types": types, "ptr_types": ptr_types, "consts": consts, "id_type": id_type,
        "globals": globals_, "body": body, "labels": labels, "main_pc": 0,
        "zero_of": zero_of,
    }


def _check_spirv_structure(text):
    """Static structural sanity checks on translated SPIR-V text (unique labels, valid targets, no subgroup ops)."""
    lines = _strip_comments(text)
    in_fn = False
    seen_labels = set()
    defined_ids = set()
    referenced_labels = []
    prev_term = True
    for line in lines:
        m = _SPV_LINE.match(line)
        assert m, f"unparseable SPIR-V line: {line!r}"
        res, op, rest = m.groups()
        a = rest.split() if rest else []
        if op == "OpFunction":
            in_fn = True
            prev_term = True
            continue
        if op == "OpFunctionParameter":
            assert in_fn and prev_term, f"OpFunctionParameter outside function header: {line}"
            continue
        if op == "OpFunctionEnd":
            assert prev_term, "function ended without block terminator"
            in_fn = False
            continue
        if not in_fn:
            if res:
                defined_ids.add(res)
            continue
        if op == "OpLabel":
            assert prev_term, f"OpLabel {res} reached without preceding terminator"
            assert res not in seen_labels, f"duplicate OpLabel {res}"
            seen_labels.add(res)
            prev_term = False
            continue
        assert not prev_term, f"instruction after terminator before OpLabel: {line}"
        if op == "OpBranch":
            referenced_labels.append(a[0])
            prev_term = True
        elif op == "OpBranchConditional":
            referenced_labels.extend((a[1], a[2]))
            prev_term = True
        elif op == "OpSwitch":
            referenced_labels.append(a[1])
            for i in range(3, len(a), 2):
                referenced_labels.append(a[i])
            prev_term = True
        elif op in ("OpReturn", "OpReturnValue", "OpKill", "OpUnreachable"):
            prev_term = True
        elif op == "OpSelectionMerge":
            referenced_labels.append(a[0])
        elif op == "OpLoopMerge":
            referenced_labels.extend((a[0], a[1]))
    for lbl in referenced_labels:
        assert lbl in seen_labels, f"branch/merge references undefined label {lbl}"


def run_emulated_workgroup(spvasm, local_size, push_constants=None, memory=None, max_steps=2_000_000, reverse_order=False):
    """Execute a --wave64-emulate compute shader across all `lx * ly * lz` invocations with real shared
    Workgroup memory and strict uniform OpControlBarrier checking. Returns `(memory, barrier_count)`."""
    _check_spirv_structure(spvasm)
    mod = _parse_module(spvasm)
    types = mod["types"]
    ptr_types = mod["ptr_types"]
    consts = mod["consts"]
    body = mod["body"]
    labels = mod["labels"]
    zero_of = mod["zero_of"]

    push_constants = list(push_constants or [])
    memory = {} if memory is None else memory
    lx, ly, lz = local_size
    coords = [(x, y, z) for z in range(lz) for y in range(ly) for x in range(lx)]
    n_inv = len(coords)

    # Shared Workgroup variables across all invocations in the workgroup.
    wg_boxes = {}
    for res, cls, pointee, init_id in mod["globals"]:
        if cls == "Workgroup":
            wg_boxes[res] = [zero_of(pointee)]

    def walk_struct_offset(tid, idxs):
        off = 0
        cur = tid
        for idx in idxs:
            k = types[cur]
            if k[0] == "struct":
                for m_t in k[1][:idx]:
                    off += _type_bytes(m_t)
                cur = k[1][idx]
            elif k[0] == "arr":
                off += idx * _type_bytes(k[1])
                cur = k[1]
            else:
                raise NotImplementedError(f"walk {k}")
        return off, cur

    def _type_bytes(tid):
        k = types[tid]
        if k[0] == "int":
            return k[1] // 8
        if k[0] == "float":
            return k[1] // 8
        if k[0] == "vec":
            return k[2] * _type_bytes(k[1])
        if k[0] == "arr":
            return k[2] * _type_bytes(k[1])
        if k[0] == "struct":
            return sum(_type_bytes(m) for m in k[1])
        raise NotImplementedError(f"size {k}")

    def make_inv(idx, coord):
        vals = dict(consts)
        id_t = dict(mod["id_type"])
        ptrs = {}
        pc_t = {}
        for res, cls, pointee, init_id in mod["globals"]:
            if cls == "PushConstant":
                ptrs[res] = ("pc", 0)
                pc_t[res] = pointee
            elif cls == "Workgroup":
                ptrs[res] = (wg_boxes[res], 0)
            elif cls == "Input":
                if res == "%local_id":
                    box = [coord]
                elif res == "%wg_id":
                    box = [(0, 0, 0)]
                elif res == "%local_index":
                    box = [idx]
                else:
                    box = [zero_of(pointee)]
                ptrs[res] = (box, 0)
            else:
                raise NotImplementedError(f"global storage class {cls}")
        return {
            "pc": mod["main_pc"], "vals": vals, "id_t": id_t, "ptrs": ptrs, "pc_t": pc_t,
            "cur_lbl": None, "prev_lbl": None, "done": False, "barrier": None,
        }

    def step_until_barrier(inv, step_budget):
        vals, id_t, ptrs, pc_t = inv["vals"], inv["id_t"], inv["ptrs"], inv["pc_t"]
        pc = inv["pc"]
        steps = 0
        while pc < len(body):
            steps += 1
            if steps > step_budget:
                raise AssertionError(f"exceeded step budget {step_budget} (infinite loop?)")
            res, op, a, line = body[pc]
            pc += 1
            if res and a:
                id_t[res] = a[0]
            if op == "OpLabel":
                inv["prev_lbl"], inv["cur_lbl"] = inv["cur_lbl"], res
                continue
            if op in _SPV_IGNORED:
                continue
            if op == "OpControlBarrier":
                inv["pc"] = pc
                inv["barrier"] = pc - 1
                return steps
            if op == "OpReturn":
                inv["pc"] = pc
                inv["done"] = True
                return steps
            if op == "OpBranch":
                pc = labels[a[0]]
                continue
            if op == "OpBranchConditional":
                cond = vals[a[0]]
                assert isinstance(cond, bool), f"non-bool branch cond: {line}"
                pc = labels[a[1] if cond else a[2]]
                continue
            if op == "OpSwitch":
                sel = vals[a[0]]
                tgt = a[1]
                for k in range(2, len(a), 2):
                    if int(a[k]) == sel:
                        tgt = a[k + 1]
                        break
                pc = labels[tgt]
                continue
            if op == "OpPhi":
                matched = [a[k] for k in range(1, len(a) - 1, 2) if a[k + 1] == inv["prev_lbl"]]
                assert len(matched) == 1, f"OpPhi prev_lbl={inv['prev_lbl']!r} not in {line}"
                vals[res] = vals[matched[0]]
                continue
            if op == "OpVariable":
                cls, pointee = ptr_types[a[0]]
                assert cls == "Function"
                if len(a) > 2:
                    iv = vals[a[2]]
                    box = [list(iv) if isinstance(iv, list) else iv]
                else:
                    box = [zero_of(pointee)]
                ptrs[res] = (box, 0)
            elif op == "OpAccessChain":
                p = ptrs[a[1]]
                if p[0] == "pc":
                    off, cur = walk_struct_offset(pc_t[a[1]], [vals[i] for i in a[2:]])
                    ptrs[res] = ("pc", p[1] + off)
                    pc_t[res] = cur
                elif p[0] == "psb":
                    elem = ptr_types[a[0]][1]
                    stride = _type_bytes(types[elem][1] if types[elem][0] == "rarr" else elem)
                    ptrs[res] = ("psb", p[1] + stride * vals[a[2]])
                else:
                    cur_p = p
                    for idx_id in a[2:]:
                        idx = vals[idx_id]
                        container = cur_p[0][cur_p[1]]
                        assert 0 <= idx < len(container), f"OOB index {idx} in {line}"
                        cur_p = (container, idx)
                    ptrs[res] = cur_p
            elif op == "OpLoad":
                p = ptrs[a[1]]
                if p[0] == "pc":
                    w = p[1] // 4
                    vals[res] = push_constants[w] if 0 <= w < len(push_constants) else 0
                elif p[0] == "psb":
                    assert p[1] % 4 == 0, f"unaligned PSB load {p[1]:#x}"
                    vals[res] = memory.get(p[1], 0)
                else:
                    v = p[0][p[1]]
                    vals[res] = list(v) if isinstance(v, list) else v
            elif op == "OpStore":
                p = ptrs[a[0]]
                v = vals[a[1]]
                if p[0] == "psb":
                    assert p[1] % 4 == 0, f"unaligned PSB store {p[1]:#x}"
                    memory[p[1]] = v & MASK32
                else:
                    p[0][p[1]] = v
            elif op in ("OpAtomicOr", "OpAtomicIAdd", "OpAtomicAnd", "OpAtomicXor", "OpAtomicExchange"):
                p = ptrs[a[1]]
                old = memory.get(p[1], 0) if p[0] == "psb" else p[0][p[1]]
                v = vals[a[4]] & MASK32
                if op == "OpAtomicOr":
                    new = old | v
                elif op == "OpAtomicIAdd":
                    new = (old + v) & MASK32
                elif op == "OpAtomicAnd":
                    new = old & v
                elif op == "OpAtomicXor":
                    new = old ^ v
                else:
                    new = v
                if p[0] == "psb":
                    memory[p[1]] = new & MASK32
                else:
                    p[0][p[1]] = new & MASK32
                vals[res] = old
            elif op == "OpSelect":
                vals[res] = vals[a[2]] if vals[a[1]] else vals[a[3]]
            elif op == "OpCopyObject":
                vals[res] = vals[a[1]]
            elif op == "OpCompositeExtract":
                v = vals[a[1]]
                for i in a[2:]:
                    v = v[int(i)]
                vals[res] = v
            elif op == "OpCompositeConstruct":
                vals[res] = tuple(vals[x] for x in a[1:])
            elif op == "OpConvertUToPtr":
                ptrs[res] = ("psb", vals[a[1]])
                vals[res] = ptrs[res]
            elif op == "OpUConvert":
                bits = types[a[0]][1]
                vals[res] = vals[a[1]] & ((1 << bits) - 1)
            elif op == "OpBitcast":
                k = types[a[0]]
                v = vals[a[1]]
                if k[0] == "int":
                    vals[res] = v & ((1 << k[1]) - 1) if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0]
                elif k[0] == "float":
                    vals[res] = struct.unpack("<f", struct.pack("<I", v & MASK32))[0] if isinstance(v, int) else v
                else:
                    raise NotImplementedError(f"OpBitcast {line}")
            elif op == "OpBitwiseAnd":
                vals[res] = (vals[a[1]] & vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpBitwiseOr":
                vals[res] = (vals[a[1]] | vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpBitwiseXor":
                vals[res] = (vals[a[1]] ^ vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpNot":
                vals[res] = (~vals[a[1]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpShiftLeftLogical":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] << vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpShiftRightLogical":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] & ((1 << bits) - 1)) >> vals[a[2]]
            elif op == "OpIAdd":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] + vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpISub":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] - vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpIMul":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] * vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpUMod":
                vals[res] = vals[a[1]] % vals[a[2]]
            elif op == "OpUDiv":
                vals[res] = vals[a[1]] // vals[a[2]]
            elif op == "OpIAddCarry":
                s = (vals[a[1]] & MASK32) + (vals[a[2]] & MASK32)
                vals[res] = (s & MASK32, 1 if s > MASK32 else 0)
            elif op == "OpISubBorrow":
                x, y = vals[a[1]] & MASK32, vals[a[2]] & MASK32
                vals[res] = ((x - y) & MASK32, 1 if x < y else 0)
            elif op == "OpIEqual":
                vals[res] = vals[a[1]] == vals[a[2]]
            elif op == "OpINotEqual":
                vals[res] = vals[a[1]] != vals[a[2]]
            elif op == "OpULessThan":
                vals[res] = vals[a[1]] < vals[a[2]]
            elif op == "OpULessThanEqual":
                vals[res] = vals[a[1]] <= vals[a[2]]
            elif op == "OpUGreaterThan":
                vals[res] = vals[a[1]] > vals[a[2]]
            elif op == "OpUGreaterThanEqual":
                vals[res] = vals[a[1]] >= vals[a[2]]
            elif op == "OpSLessThan":
                vals[res] = _s32(vals[a[1]]) < _s32(vals[a[2]])
            elif op == "OpSGreaterThan":
                vals[res] = _s32(vals[a[1]]) > _s32(vals[a[2]])
            elif op == "OpLogicalNot":
                vals[res] = not vals[a[1]]
            elif op == "OpLogicalAnd":
                vals[res] = vals[a[1]] and vals[a[2]]
            elif op == "OpLogicalOr":
                vals[res] = vals[a[1]] or vals[a[2]]
            elif op == "OpBitCount":
                vals[res] = (vals[a[1]] & MASK32).bit_count()
            elif op == "OpBitFieldUExtract":
                base, off, cnt = vals[a[1]], vals[a[2]], vals[a[3]]
                vals[res] = (base >> off) & ((1 << cnt) - 1) if cnt else 0
            elif op == "OpExtInst":
                ext_op = a[2]
                if ext_op == "UMin":
                    vals[res] = min(vals[a[3]], vals[a[4]])
                elif ext_op == "UMax":
                    vals[res] = max(vals[a[3]], vals[a[4]])
                elif ext_op == "FindILsb":
                    v = vals[a[3]] & MASK32
                    vals[res] = (v & -v).bit_length() - 1 if v else MASK32
                elif ext_op == "FindUMsb":
                    v = vals[a[3]] & MASK32
                    vals[res] = v.bit_length() - 1 if v else MASK32
                else:
                    raise NotImplementedError(f"OpExtInst {ext_op}: {line}")
            else:
                raise NotImplementedError(f"unsupported op {op}: {line}")
        inv["pc"] = pc
        inv["done"] = True
        return steps

    invs = [make_inv(i, c) for i, c in enumerate(coords)]
    step_order = list(reversed(invs)) if reverse_order else invs
    total_steps = 0
    barriers = 0
    while True:
        for inv in step_order:
            if not inv["done"]:
                inv["barrier"] = None
                total_steps += step_until_barrier(inv, max_steps - total_steps)
        if all(inv["done"] for inv in invs):
            break
        # Strict uniform barrier check: every invocation must stop at the exact same OpControlBarrier.
        if any(inv["done"] for inv in invs):
            raise AssertionError("non-uniform OpControlBarrier: some invocations returned while others waited")
        sites = {inv["barrier"] for inv in invs}
        if len(sites) != 1 or None in sites:
            raise AssertionError(f"non-uniform OpControlBarrier sites: {sites}")
        barriers += 1
    return memory, barriers


def _translate_encoded(name, insns, local_size=(64, 1, 1), emulate=True, rsrc2=0x8088):
    """Translate a list of `(text, nbytes)` instructions with synthetic encoding placeholders."""
    lines = [f"; {name}  synthetic", "; mattr: +wavefrontsize64,-wavefrontsize32"]
    for txt, nb in insns:
        enc = ",".join(["0x00"] * nb)
        lines.append(f"\t{txt:52s} ; encoding: [{enc}]")
    prev_emu = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(os.path.join(td, "disasm"))
            os.mkdir(os.path.join(td, "fssm"))
            s_path = os.path.join(td, "disasm", name + ".s")
            f_path = os.path.join(td, "fssm", name + ".fssm")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            lx, ly, lz = local_size
            with open(f_path, "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", rsrc2, lx, ly, lz))
            spv = rdna2spirv.translate(s_path)
            gen = rdna2spirv.translate.last_gen
            return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _switch_case_count(gen):
    return sum(1 for b in gen.blocks if b is not None) + len(gen.x_segs)


def _screenprobes_insns(n_steps, with_loads=False):
    insns = [("v_mov_b32_e32 v4, v0", 4)]
    for _ in range(n_steps):
        insns.append(("v_cmp_lt_u32_e64 s[10:11], 16, v4", 8))
        insns.append(("s_and_saveexec_b64 s[12:13], s[10:11]", 4))
        insns.append(("s_cbranch_execz 4" if with_loads else "s_cbranch_execz 2", 4))
        if with_loads:
            insns.append(("v_lshlrev_b32_e32 v8, 2, v0", 4))
            insns.append(("buffer_load_dword v5, v8, s[0:3], 0 offen", 8))
            insns.append(("v_add_nc_u32_e32 v4, v5, v4", 4))
        else:
            insns.append(("v_add_nc_u32_e32 v4, 1, v4", 4))
        insns.append(("v_xor_b32_e32 v4, 7, v4", 4))
        insns.append(("s_andn2_b64 exec, s[12:13], exec", 4))
        insns.append(("s_cbranch_execz 2", 4))
        insns.append(("v_add_nc_u32_e32 v4, 3, v4", 4))
        insns.append(("v_xor_b32_e32 v4, 5, v4", 4))
        insns.append(("s_or_b64 exec, exec, s[12:13]", 4))
    insns.append(("v_lshlrev_b32_e32 v9, 2, v0", 4))
    insns.append(("v_add_nc_u32_e32 v9, 1024, v9", 4))
    insns.append(("buffer_store_dword v4, v9, s[0:3], 0 offen", 8))
    insns.append(("s_endpgm", 4))
    return insns


class DeadSccLivenessTests(unittest.TestCase):
    def test_dead_scc_mask_ops_elide_votes_while_live_scc_is_exact(self):
        # First s_and_saveexec_b64 has DEAD SCC (immediately overwritten by second s_and_saveexec_b64);
        # second s_and_saveexec_b64 has LIVE SCC (read by s_cselect_b32), and only lane 50 is active!
        insns = [
            ("v_cmp_lt_u32_e64 s[6:7], 20, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),      # DEAD SCC -> no vote!
            ("s_or_b64 exec, exec, s[8:9]", 4),            # DEAD SCC -> no vote!
            ("v_cmp_eq_u32_e64 s[10:11], 50, v0", 8),
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),  # LIVE SCC -> must vote across all 64 lanes!
            ("s_cselect_b32 s14, 111, 222", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_scc_live_dead", insns, (64, 1, 1), emulate=True)
        # Only ONE mask-op SCC vote remains (1 OpAtomicOr, not 3).
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        self.assertEqual(len(gen.x_segs), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 111, f"lane {lane} missed live wave64 SCC from lane 50")

    def test_cross_block_live_scc_preserved(self):
        # SCC written in block 0 by s_and_saveexec_b64, read in block 1 by s_cbranch_scc0.
        insns = [
            ("s_mov_b32 s14, 0", 4),
            ("v_cmp_eq_u32_e64 s[6:7], 45, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),      # LIVE SCC across branch!
            ("s_cbranch_scc0 1", 4),                       # not taken because lane 45 is active
            ("s_mov_b32 s14, 77", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_scc_xblock", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 77, f"lane {lane}: expected 77 from cross-block live SCC")

    def test_src_scc_keeps_mask_vote_live(self):
        for threads in (64, 128):
            for consumer in ("scalar", "vector"):
                with self.subTest(threads=threads, consumer=consumer):
                    insns = [
                        ("v_and_b32_e32 v3, 63, v0", 4),
                        ("v_cmp_eq_u32_e64 s[6:7], 45, v3", 8),
                        ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
                    ]
                    if consumer == "scalar":
                        insns.append(("s_mov_b32 s14, src_scc", 4))
                    insns.append(("s_mov_b64 exec, -1", 4))
                    insns.append(("v_mov_b32_e32 v5, s14", 4) if consumer == "scalar"
                                 else ("v_mov_b32_e32 v5, src_scc", 4))
                    insns.extend([
                        ("v_lshlrev_b32_e32 v6, 2, v0", 4),
                        ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
                        ("s_endpgm", 4),
                    ])
                    size = (threads, 1, 1)
                    spirv, _ = _translate_encoded("cs_scc_inline_source", insns, size)
                    mem, _ = run_emulated_workgroup(spirv, size, push_constants=VSHARP + [0] * 28)
                    # Lane 45 is active on every wave, so the scalar SCC reads as one in every lane.
                    actual = [mem[lane * 4] for lane in range(threads)]
                    self.assertEqual(actual, [1] * threads,
                                     f"{actual.count(1)}/{threads} lanes observed the live SCC")

    def test_non_emulated_unchanged_keeps_wqm_scc_vote(self):
        # Verify WAVE64_EMULATE=False keeps dead-SCC votes (required by test_translate.py WQM_MASK).
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("v_cmp_gt_u32_e32 vcc, 16, v0", 4),
            ("s_wqm_b64 s[6:7], vcc", 4),
            ("s_bcnt1_i32_b64 s8, s[6:7]", 4),
            ("s_endpgm", 4),
        ]
        spv_native, _ = _translate_encoded("cs_wqm_native_keep", insns, (64, 1, 1), emulate=False)
        self.assertIn("OpGroupNonUniformAny", spv_native)
        spv_emu, _ = _translate_encoded("cs_wqm_emu_dead_scc", insns, (64, 1, 1), emulate=True)
        self.assertNotIn("OpGroupNonUniform", spv_emu)
        # Only s_bcnt1's wave_mask uses OpAtomicOr (2 words); s_wqm's dead SCC vote is elided.
        self.assertEqual(spv_emu.count("OpAtomicOr"), 2)


class LocalExecFoldTests(unittest.TestCase):
    def test_screenprobes_chain_64_and_128_threads_bit_exact(self):
        # 16 sequential ScreenProbes-shaped if/else steps with buffer loads + final store.
        insns = _screenprobes_insns(16, with_loads=True)
        init_mem = {i * 4: ((i * 17 + 3) & 0xFF) for i in range(128)}
        for wg in ((64, 1, 1), (128, 1, 1)):
            with self.subTest(wg=wg):
                spv, gen = _translate_encoded(f"cs_sp_{wg[0]}", insns, wg, emulate=True)
                # All 16*4=64 blocks collapse into 1 block + 1 memory-order split before the final store!
                self.assertEqual(_switch_case_count(gen), 2)
                self.assertEqual(spv.count("OpAtomicOr"), 0)
                self.assertNotIn("OpGroupNonUniform", spv)
                mem, _ = run_emulated_workgroup(spv, wg, push_constants=VSHARP + [0] * 28, memory=dict(init_mem))
                for lane in range(wg[0]):
                    v = lane
                    ld = init_mem[lane * 4]
                    for _ in range(16):
                        if 16 < v:
                            v = ((v + ld) & MASK32) ^ 7
                        else:
                            v = ((v + 3) & MASK32) ^ 5
                    self.assertEqual(mem[1024 + lane * 4], v, f"wg={wg} lane={lane}")

    def test_nested_exec_and_inner_saveexec_fold_exact(self):
        # Outer s_cbranch_execz contains a nested s_and_saveexec_b64 + s_cbranch_execz + s_or_b64 restore.
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_cmp_lt_u32_e64 s[6:7], 15, v0", 8),        # lanes 16..63
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 7", 4),                      # outer skip -> restore
            ("v_add_nc_u32_e32 v4, 100, v4", 4),
            ("v_cmp_lt_u32_e64 s[10:11], 40, v0", 8),      # inner: lanes 41..63
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),
            ("s_cbranch_execz 1", 4),                      # inner skip -> inner restore
            ("v_add_nc_u32_e32 v4, 1000, v4", 4),
            ("s_or_b64 exec, exec, s[12:13]", 4),          # inner restore (from zero_pair s[12:13])
            ("v_add_nc_u32_e32 v4, 10, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),            # outer restore
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_nested_fold", insns, (64, 1, 1), emulate=True)
        self.assertEqual(_switch_case_count(gen), 1)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = lane
            if lane > 15:
                want += 100
                if lane > 40:
                    want += 1000
                want += 10
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_negative_live_out_scalar_not_folded(self):
        # Like cs_w64x_branch: s9 is written inside s_cbranch_execz and read after join -> MUST NOT fold!
        insns = [
            ("s_mov_b32 s9, 0", 4),
            ("s_mov_b32 exec_lo, 0", 4),
            ("s_mov_b32 exec_hi, 0xff", 4),                # lanes 32..39 active; lane 0 inactive
            ("s_cbranch_execz 1", 4),                      # must vote across all 64 lanes!
            ("s_mov_b32 s9, 5", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s9", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_neg_live_scalar", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 5, f"lane {lane}: stale scalar s9 after s_cbranch_execz")

    def test_negative_store_and_crosslane_and_exec_widen_not_folded(self):
        cases = [
            ("multi_store", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 4", 4),
                ("buffer_store_dword v0, v0, s[0:3], 0 offen", 8),
                ("buffer_store_dword v0, v0, s[0:3], 0 offen offset:4", 8),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
            ("load_and_store", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 4", 4),
                ("buffer_load_dword v4, v0, s[0:3], 0 offen", 8),
                ("buffer_store_dword v4, v0, s[0:3], 0 offen", 8),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
            ("crosslane", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 1", 4),
                ("v_readlane_b32 s10, v0, 40", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
            ("exec_widen", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 2", 4),
                ("s_mov_b64 exec, -1", 4),                 # widens EXEC! must not fold
                ("v_add_nc_u32_e32 v4, 1, v4", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
        ]
        for label, insns in cases:
            with self.subTest(case=label):
                spv, _ = _translate_encoded(f"cs_neg_{label}", insns, (64, 1, 1), emulate=True)
                self.assertGreaterEqual(spv.count("OpAtomicOr"), 1, f"{label} should NOT fold s_cbranch_execz")

    def test_mask_flbit_keeps_wave_exchange_out_of_lane_local_execz_fold(self):
        # Both waves have bit 3 set in VCC_LO. Only lane 32 of each wave writes its leading-zero count.
        for lx in (64, 128):
            for op, ext_inst in (("s_flbit_i32_b32", "FindUMsb"), ("s_flbit_i32", "FindSMsb")):
                for source in ("vcc_lo", "s14"):
                    with self.subTest(threads=lx, op=op, source=source):
                        insns = [
                            ("s_mov_b32 s14, 8", 4),
                            ("v_and_b32_e32 v1, 63, v0", 4),
                            ("v_cmp_eq_u32_e32 vcc, 3, v1", 4),
                            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
                            ("s_mov_b32 exec_lo, 0", 4),
                            ("s_mov_b32 exec_hi, 1", 4),
                            ("s_cbranch_execz 4", 4),
                            (f"{op} s9, {source}", 4),
                            ("v_mov_b32_e32 v4, s9", 4),
                            ("buffer_store_dword v4, v6, s[0:3], 0 offen", 8),
                            ("s_mov_b64 exec, -1", 4),
                            ("s_endpgm", 4),
                        ]
                        spv, _ = _translate_encoded("cs_flbit_in_execz", insns, (lx, 1, 1), emulate=True)
                        self.assertIn(ext_inst, spv)
                        if source == "vcc_lo":
                            self.assertIn("OpAtomicOr", spv)
                        else:
                            self.assertNotIn("OpAtomicOr", spv)
                        if op == "s_flbit_i32_b32":
                            mem, _ = run_emulated_workgroup(spv, (lx, 1, 1), push_constants=VSHARP + [0] * 28)
                            self.assertEqual(mem, {lane * 4: 28 for lane in range(32, lx, 64)})

    def test_write_before_folded_read_orders_across_halves(self):
        # Every lane stores `100 + lane` at `lane * 4`; then ONLY lanes >= 32 enter a folded s_cbranch_execz
        # and read partner lane `(lane ^ 32) * 4`. The uniform x_split before `if (%active)` must order the write!
        insns = [
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 100, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("v_mov_b32_e32 v9, 0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),        # lanes 32..63 only
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 4", 4),
            ("v_xor_b32_e32 v8, 32, v0", 4),
            ("v_lshlrev_b32_e32 v8, 2, v8", 4),
            ("buffer_load_dword v9, v8, s[0:3], 0 offen", 8),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_add_nc_u32_e32 v10, 512, v7", 4),
            ("buffer_store_dword v9, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_raw_before_fold", insns, (64, 1, 1), emulate=True)
        # Folded (0 OpAtomicOr), with 2 uniform memory-order splits: write->read and read->write.
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertEqual(len(gen.x_segs), 2)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = (100 + (lane ^ 32)) if lane >= 32 else 0
            self.assertEqual(mem[512 + lane * 4], want, f"lane {lane}")

    def test_multiwave_128_folded_exec_stays_synchronized_for_later_readlane_and_barrier(self):
        # 128 threads (2 waves): every thread runs a folded s_cbranch_execz, followed by v_readlane(lane 40)
        # and an s_barrier + LDS read from the other wave. Proves zero round desynchronization across waves!
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_and_b32_e32 v2, 1, v0", 4),
            ("v_cmp_eq_u32_e64 s[6:7], 1, v2", 8),         # odd lanes only
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 1", 4),                      # folded!
            ("v_add_nc_u32_e32 v4, 1000, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_readlane_b32 s10, v4, 41", 4),             # cross-half readlane in each wave (odd lane -> +1000)
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("ds_write_b32 v5, v4", 8),
            ("s_barrier", 4),
            ("v_xor_b32_e32 v6, 64, v0", 4),               # read partner thread in the OTHER wave!
            ("v_lshlrev_b32_e32 v6, 2, v6", 4),
            ("ds_read_b32 v7, v6", 8),
            ("v_add_nc_u32_e32 v8, s10, v7", 4),
            ("buffer_store_dword v8, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_mw128_sync", insns, (128, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        mem, barriers = run_emulated_workgroup(spv, (128, 1, 1), push_constants=VSHARP + [0] * 28)
        self.assertGreaterEqual(barriers, 3)
        for tid in range(128):
            wave_base = (tid // 64) * 64
            s10_want = wave_base + 41 + 1000
            partner = tid ^ 64
            v7_want = partner + (1000 if (partner & 1) else 0)
            self.assertEqual(mem[tid * 4], s10_want + v7_want, f"tid {tid}")

    def test_folded_region_with_internal_loop_exact(self):
        # Multi-block folded s_cbranch_execz containing an internal scalar-counted loop (`sub_tree` loop path).
        # Lanes >= 32 add 5 four times (+20); lanes < 32 skip the entire loop. s15 is dead at the join.
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),        # lanes 32..63
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 5", 4),                      # skip 5 dwords (5 x 4B insns) to restore
            ("s_mov_b32 s15, 4", 4),
            ("v_add_nc_u32_e32 v4, 5, v4", 4),             # loop header (pc +4)
            ("s_sub_u32 s15, s15, 1", 4),
            ("s_cmp_gt_u32 s15, 0", 4),
            ("s_cbranch_scc1 -4", 4),                      # back 4 dwords to v_add_nc_u32
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_fold_loop", insns, (64, 1, 1), emulate=True)
        self.assertEqual(_switch_case_count(gen), 1)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertGreaterEqual(spv.count("OpLoopMerge"), 2)  # outer scheduler loop + inner folded loop
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = lane + (20 if lane >= 32 else 0)
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_folded_region_scalar_branch_to_join_skips_the_rest(self):
        # An s_cbranch_scc inside a folded s_cbranch_execz region that jumps straight to the join: the region's
        # sub-CFG must keep that exit edge, or the block after the branch runs on the exit path too. Lanes whose
        # (5 * lane) & 15 > 12 enter the region; s11 decides the scalar branch. The add must run only when the
        # branch falls through.
        for cmp, s11, taken in (("s_cmp_ge_u32 s11, 33", 5, True), ("s_cmp_ge_u32 s11, 33", 40, False),
                                ("s_cmp_lt_u32 s11, 33", 5, True), ("s_cmp_lt_u32 s11, 33", 40, False)):
            branch = "s_cbranch_scc0 2" if cmp.startswith("s_cmp_ge") else "s_cbranch_scc1 2"
            insns = [
                ("v_mul_lo_u32 v3, v0, 5", 8),
                ("v_and_b32_e32 v3, 15, v3", 4),
                ("v_mov_b32_e32 v4, 1", 4),
                (f"s_mov_b32 s11, {s11}", 4),
                ("v_cmp_lt_u32_e32 vcc, 12, v3", 4),
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 4", 4),                  # to the join (4 dwords)
                (cmp, 4),
                (branch, 4),                               # to the join (2 dwords)
                ("v_add_nc_u32_e32 v4, 21, v4", 4),
                ("v_add_nc_u32_e32 v4, 100, v4", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("v_lshlrev_b32_e32 v9, 2, v0", 4),
                ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            for threads in (64, 128):
                with self.subTest(cmp=cmp, s11=s11, threads=threads):
                    spv, gen = _translate_encoded("cs_fold_scc_exit", insns, (threads, 1, 1), emulate=True)
                    self.assertEqual(_switch_case_count(gen), 1)      # the region is folded inline
                    mem, _ = run_emulated_workgroup(spv, (threads, 1, 1), push_constants=VSHARP + [0] * 28)
                    for lane in range(threads):
                        inside = ((lane % 64) * 5) & 15 > 12
                        want = 1 + (121 if inside and not taken else 0)
                        self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_cs_w64x_saveexec_b64_fixture_emulated_exact(self):
        # Run the repo's real cs_w64x_saveexec_b64.s fixture under --wave64-emulate on the 64-lane barrier interpreter.
        fixture = os.path.join(HERE, "fixtures", "disasm", "cs_w64x_saveexec_b64.s")
        if not os.path.isfile(fixture):
            self.skipTest("cs_w64x_saveexec_b64.s fixture absent")
        prev_emu = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            with tempfile.TemporaryDirectory() as td:
                os.mkdir(os.path.join(td, "disasm"))
                os.mkdir(os.path.join(td, "fssm"))
                s_path = os.path.join(td, "disasm", "cs_w64x_saveexec_b64.s")
                f_path = os.path.join(td, "fssm", "cs_w64x_saveexec_b64.fssm")
                with open(fixture, encoding="utf-8") as src, open(s_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                with open(f_path, "wb") as f:
                    f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))
                spv = rdna2spirv.translate(s_path)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev_emu
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        got = [mem.get(i * 4, 0) for i in range(8)]
        self.assertEqual(got, [6160, 12304, 5120, 2088, 6168, 8192, 7176, 1837171])

    def test_write_before_folded_scalar_load_orders_cleanly(self):
        # #1587 interaction: buffer_store before a folded s_cbranch_execz containing s_load_dword must emit
        # a uniform x_split BEFORE `if (%active)`, never inside the divergent branch.
        insns = [
            ("s_mov_b32 s8, 0", 4),
            ("s_mov_b32 s9, 0", 4),
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 200, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("v_mov_b32_e32 v9, 0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),
            ("s_and_saveexec_b64 s[10:11], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("s_load_dword s14, s[8:9], 0x0", 8),          # s14 dead at join; reads mem written by lane 0!
            ("v_mov_b32_e32 v9, s14", 4),
            ("s_or_b64 exec, exec, s[10:11]", 4),
            ("v_add_nc_u32_e32 v10, 512, v7", 4),
            ("buffer_store_dword v9, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_raw_sload_fold", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertEqual(len(gen.x_segs), 2)               # write->s_load split before fold + s_load->store after
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = 200 if lane >= 32 else 0
            self.assertEqual(mem[512 + lane * 4], want, f"lane {lane}")

    def test_nested_reused_saveexec_and_xor_if_else_folds_cleanly(self):
        # Outer `if (lane >= 16)` containing TWO sequential inner `if/else` diamonds that reuse the SAME
        # s[10:11] saved-EXEC pair and use LLVM's `s_xor_b64` + `s_or_saveexec_b64` if/else lowering.
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 16, v0", 8),        # outer: lane >= 16
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 17", 4),                     # outer skip (17 dwords)
            # Inner 1: if (lane >= 32) v4 += 100 else v4 += 10 using s[10:11] + s_xor_b64 + s_or_saveexec_b64
            ("v_cmp_le_u32_e64 vcc, 32, v0", 8),
            ("s_and_saveexec_b64 s[10:11], vcc", 4),
            ("s_xor_b64 s[10:11], exec, s[10:11]", 4),     # else mask in s[10:11]
            ("s_cbranch_execz 1", 4),
            ("v_add_nc_u32_e32 v4, 100, v4", 4),
            ("s_or_saveexec_b64 s[12:13], s[10:11]", 4),   # swap to else, save then-exec
            ("s_andn2_b64 exec, exec, s[12:13]", 4),       # keep else lanes only
            ("s_cbranch_execz 1", 4),
            ("v_add_nc_u32_e32 v4, 10, v4", 4),
            ("s_or_b64 exec, exec, s[12:13]", 4),          # restore outer exec
            # Inner 2: REUSE s[10:11] for a second inner guard: if (lane >= 48) v4 += 1000
            ("v_cmp_le_u32_e64 vcc, 48, v0", 8),
            ("s_and_saveexec_b64 s[10:11], vcc", 4),
            ("s_cbranch_execz 1", 4),
            ("v_add_nc_u32_e32 v4, 1000, v4", 4),
            ("s_or_b64 exec, exec, s[10:11]", 4),
            # Outer restore
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_reused_nested_fold", insns, (64, 1, 1), emulate=True)
        self.assertEqual(_switch_case_count(gen), 1)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = lane
            if lane >= 16:
                want += 100 if lane >= 32 else 10
                if lane >= 48:
                    want += 1000
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_negative_live_in_pair_and_unmasked_vcc_scc_not_folded(self):
        # 1) s[10:11] initialized to -1 BEFORE the region and ORed with exec inside -> live-in at run -> MUST NOT fold!
        insns_live_in = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("s_mov_b64 s[10:11], -1", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("s_or_b64 s[10:11], s[10:11], exec", 4),      # preserves entry -1, NOT guaranteed 0!
            ("s_or_b64 exec, exec, s[10:11]", 4),          # widens EXEC to all-ones!
            ("v_add_nc_u32_e32 v4, 5, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv1, _ = _translate_encoded("cs_neg_livein_zero", insns_live_in, (64, 1, 1), emulate=True)
        self.assertGreaterEqual(spv1.count("OpAtomicOr"), 1)
        mem1, _ = run_emulated_workgroup(spv1, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem1[lane * 4], lane + 5, f"lane {lane}: EXEC widen must run for all lanes")

        # 2) `s_and_b64 s[10:11], vcc, s[12:13]` where vcc is not tracked by lane_masks, with internal live SCC:
        # Gen.is_mask_op treats vcc as a mask op and emits scc_from_mask -> MUST NOT fold or crash!
        insns_vcc_scc = [
            ("s_mov_b32 s12, 1", 4),
            ("s_mov_b32 s13, 0", 4),
            ("s_mov_b64 vcc, s[12:13]", 4),                # vcc written as plain data, NOT in lane_masks
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("s_and_b64 s[10:11], vcc, s[12:13]", 4),      # Gen.is_mask_op(vcc) -> scc_from_mask!
            ("s_cselect_b32 s14, 7, 3", 4),                # reads SCC inside region (s14 dead at join)
            ("v_mov_b32_e32 v4, s14", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv2, _ = _translate_encoded("cs_neg_vcc_scc", insns_vcc_scc, (64, 1, 1), emulate=True)
        vote_calls = spv2.count("OpFunctionCall %uint %x_pre_vote_fn") or spv2.count("OpAtomicOr")
        self.assertGreaterEqual(vote_calls, 2)  # branch vote + scc_from_mask vote

    def test_s_cmovk_i32_keeps_scc_live_and_blocks_execz_fold(self):
        # 1) s_cmovk_i32 implicitly reads SCC: s_and_saveexec_b64 must keep its cross-half SCC vote.
        insns_vote = [
            ("s_mov_b32 s14, 222", 4),
            ("v_cmp_eq_u32_e64 s[10:11], 50, v0", 8),
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),
            ("s_cmovk_i32 s14, 111", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v5, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv1, _ = _translate_encoded("cs_cmovk_scc_live", insns_vote, (64, 1, 1), emulate=True)
        self.assertGreaterEqual(spv1.count("OpAtomicOr"), 1)
        mem1, _ = run_emulated_workgroup(spv1, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem1[lane * 4], 111, f"lane {lane}")

        # 2) Region writing SCC when s_cmovk_i32 reads SCC after join MUST NOT fold (64 & 128 threads).
        insns_fold = [
            ("s_mov_b32 s14, 222", 4),
            ("s_cmp_eq_u32 s14, 222", 4),
            ("v_cmp_lt_u32_e64 s[6:7], 40, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 2", 4),
            ("s_cmp_eq_u32 s14, 0", 4),
            ("v_add_nc_u32_e32 v4, 1, v0", 4),
            ("s_mov_b64 exec, -1", 4),
            ("s_cmovk_i32 s14, 111", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v5, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        for lx in (64, 128):
            spv2, _ = _translate_encoded(f"cs_cmovk_no_fold_{lx}", insns_fold, (lx, 1, 1), emulate=True)
            self.assertGreaterEqual(spv2.count("OpAtomicOr"), 1)
            mem2, _ = run_emulated_workgroup(spv2, (lx, 1, 1), push_constants=VSHARP + [0] * 28)
            for lane in range(lx):
                self.assertEqual(mem2[lane * 4], 222, f"lx={lx} lane={lane}")

        # 3) Every opcode in tools/rdna2spirv.py that emits `OpLoad %uint %scc` must satisfy reads_scc().
        src_path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "rdna2spirv.py")
        with open(src_path, encoding="utf-8") as f:
            lines = f.readlines()
        scc_ops = set()
        for i, line in enumerate(lines):
            if "OpLoad %uint %scc" not in line:
                continue
            for j in range(i, max(-1, i - 25), -1):
                m_ops = re.findall(r'"(s_[a-z0-9_]+)"', lines[j])
                if m_ops and ("elif op" in lines[j] or "if op" in lines[j]):
                    scc_ops.update(m_ops)
                    break
        self.assertGreaterEqual(len(scc_ops), 8)
        for op in sorted(scc_ops):
            self.assertTrue(
                wave64_emulate_opt.reads_scc({"op": op, "args": []}),
                f"{op} loads %scc in rdna2spirv.py but reads_scc() returned False",
            )

    def test_m0_implicit_read_in_v_movrel_blocks_execz_fold(self):
        # Region writes m0 and post-join v_movrels_b32 reads m0 implicitly -> MUST NOT fold.
        insns = [
            ("v_mov_b32_e32 v4, 10", 4),
            ("v_mov_b32_e32 v5, 77", 4),
            ("s_mov_b32 m0, 0", 4),
            ("v_cmp_lt_u32_e64 s[6:7], 40, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 1", 4),
            ("s_mov_b32 m0, 1", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_movrels_b32_e32 v6, v4", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v6, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_m0_vmovrel_no_fold", insns, (64, 1, 1), emulate=True)
        self.assertGreaterEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 77, f"lane {lane}: wave ran region so m0=1 -> v5=77 everywhere")

    def test_single_store_acyclic_fold_orders_war_and_raw(self):
        # Acyclic single-store s_cbranch_execz regions fold inline (0 OpAtomicOr), while uniform pre/post
        # memory ordering protects both WAR (load before folded store) and RAW (folded store before load).
        insns = [
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 50, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("v_xor_b32_e32 v8, 32, v0", 4),
            ("v_lshlrev_b32_e32 v8, 2, v8", 4),
            ("buffer_load_dword v9, v8, s[0:3], 0 offen", 8),  # reads partner half before overwrite
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),
            ("s_and_saveexec_b64 s[10:11], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("v_add_nc_u32_e32 v11, 500, v9", 4),
            ("buffer_store_dword v11, v7, s[0:3], 0 offen", 8),  # single store -> folded with WAR split before!
            ("s_or_b64 exec, exec, s[10:11]", 4),
            ("buffer_load_dword v12, v8, s[0:3], 0 offen", 8),   # reads partner half after folded store -> RAW split!
            ("v_add_nc_u32_e32 v13, 512, v7", 4),
            ("buffer_store_dword v12, v13, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        for size in ((64, 1, 1), (128, 1, 1)):
            with self.subTest(size=size):
                spv, _ = _translate_encoded("cs_single_store_fold", insns, size, emulate=True)
                self.assertEqual(spv.count("OpAtomicOr"), 0)
                for rev in (False, True):
                    mem, _ = run_emulated_workgroup(spv, size, push_constants=VSHARP + [0] * 28, reverse_order=rev)
                    for tid in range(size[0]):
                        partner = tid ^ 32
                        # lanes < 32 kept 50 + tid; lanes >= 32 overwrote their slot with 500 + (50 + (tid ^ 32))
                        if partner >= 32:
                            want = 500 + 50 + (partner ^ 32)
                        else:
                            want = 50 + partner
                        self.assertEqual(mem[512 + tid * 4], want, f"tid {tid} rev={rev}")


class StructuredEmulationTests(unittest.TestCase):
    """Structured --wave64-emulate (`inline_all` for <=64 threads, `structured_hybrid` for >64 threads)."""

    def test_single_wave_inline_all_zero_switch_and_loop_carried_war(self):
        # 64-thread and partial 40-thread shaders with live-out scalars, loop-carried store->load->store WAR/RAW,
        # and v_readfirstlane emit 0 OpSwitch cases and 0 x_spills while staying bit-exact.
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("s_mov_b32 s16, 10", 4),
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            # Un-foldable live-out scalar s_cbranch_execz
            ("v_cmp_le_u32_e64 s[6:7], 20, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 2", 4),
            ("s_add_u32 s16, s16, 5", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            # Loop with store -> cross-half load -> next iteration store (tests back-edge dirty_in!)
            ("s_mov_b32 s20, 3", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("v_xor_b32_e32 v12, 32, v0", 4),
            ("v_lshlrev_b32_e32 v12, 2, v12", 4),
            ("buffer_load_dword v13, v12, s[0:3], 0 offen", 8),
            ("v_add_nc_u32_e32 v4, v4, v13", 4),
            ("s_sub_u32 s20, s20, 1", 4),
            ("s_cmp_gt_u32 s20, 0", 4),
            ("s_cbranch_scc1 -10", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen offset:512", 8),
            ("s_endpgm", 4),
        ]
        for threads in (64, 40):
            with self.subTest(threads=threads):
                spv, gen = _translate_encoded(f"cs_inline_{threads}", insns, (threads, 1, 1), emulate=True)
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
                self.assertEqual(gen.emulate_plan["mode"], "inline_all")
                self.assertNotIn("OpSwitch", spv)
                self.assertEqual(len(gen.x_spills), 0)
                self.assertNotIn("OpGroupNonUniform", spv)
                for rev in (False, True):
                    mem, _ = run_emulated_workgroup(spv, (threads, 1, 1), push_constants=VSHARP + [0] * 28, reverse_order=rev)
                    # Reference simulation
                    arr = [0] * 64
                    for l in range(threads):
                        arr[l] = l
                    v4 = [l + (15 if l >= 20 else 0) + 15 for l in range(threads)]
                    for _ in range(3):
                        for l in range(threads):
                            arr[l] = v4[l]
                        nxt = [v4[l] + arr[l ^ 32] for l in range(threads)]
                        v4 = nxt
                    for l in range(threads):
                        self.assertEqual(mem[512 + l * 4], v4[l] & 0xFFFFFFFF, f"lane {l} rev={rev}")

    def test_mask_flbit_is_synchronized_in_a_branch_taken_by_only_one_wave(self):
        # One wave takes the SCC branch body. Its bit scan still needs a workgroup exchange.
        for op, ext_inst in (("s_flbit_i32_b32", "FindUMsb"), ("s_flbit_i32", "FindSMsb")):
            with self.subTest(op=op):
                insns = [
                    ("v_and_b32_e32 v1, 63, v0", 4),
                    ("v_cmp_eq_u32_e32 vcc, 3, v1", 4),
                    ("v_lshlrev_b32_e32 v6, 2, v0", 4),
                    ("v_readfirstlane_b32 s20, v0", 4),
                    ("s_cmp_eq_u32 s20, 0", 4),
                    ("s_cbranch_scc0 4", 4),
                    (f"{op} s9, vcc_lo", 4),
                    ("v_mov_b32_e32 v4, s9", 4),
                    ("buffer_store_dword v4, v6, s[0:3], 0 offen", 8),
                    ("s_mov_b64 exec, -1", 4),
                    ("s_endpgm", 4),
                ]
                spv, _ = _translate_encoded("cs_flbit_in_scc_branch", insns, (128, 1, 1), emulate=True)
                self.assertIn(ext_inst, spv)
                if op == "s_flbit_i32_b32":
                    mem, _ = run_emulated_workgroup(spv, (128, 1, 1), push_constants=VSHARP + [0] * 28)
                    self.assertEqual(mem, {lane * 4: 28 for lane in range(64)})

    def test_multiwave_structured_hybrid_zero_switch_and_local_mini_scheduler(self):
        # 128 threads (2 waves): top-level spine + sync-free guard (live-out scalar) + wave-divergent
        # mini-scheduler (Wave 0 runs v_readlane + store + cross-half load while Wave 1 skips it) + top s_barrier.
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("s_mov_b32 s16, 7", 4),
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            # 1) Sync-free guard with live-out scalar (0 OpSwitch!)
            ("v_cmp_le_u32_e64 s[6:7], 96, v0", 8),        # Wave 0 skips, Wave 1 takes
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 2", 4),
            ("s_add_u32 s16, s16, 30", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            # 2) Wave-divergent guard with v_readlane + store + load -> isolated local mini_scheduler!
            ("v_readfirstlane_b32 s19, v0", 4),
            ("s_cmp_lt_u32 s19, 64", 4),
            ("s_cbranch_scc0 9", 4),                       # Wave 0 enters, Wave 1 skips!
            ("v_readlane_b32 s18, v4, 37", 4),
            ("v_add_nc_u32_e32 v4, s18, v4", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("v_xor_b32_e32 v12, 32, v0", 4),
            ("v_lshlrev_b32_e32 v12, 2, v12", 4),
            ("buffer_load_dword v13, v12, s[0:3], 0 offen", 8),
            ("v_add_nc_u32_e32 v4, v4, v13", 4),
            # 3) Top-level inline s_barrier + cross-wave LDS exchange!
            ("ds_write_b32 v10, v4", 8),
            ("s_barrier", 4),
            ("v_xor_b32_e32 v12, 64, v0", 4),
            ("v_lshlrev_b32_e32 v12, 2, v12", 4),
            ("ds_read_b32 v13, v12", 8),
            ("v_add_nc_u32_e32 v4, v4, v13", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen offset:1024", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_hybrid_128", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(gen.control_flow, "emulated wave64 structured")
        self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
        kinds = [st[0] for st in gen.emulate_plan["steps"]]
        self.assertIn("inline_sync_free", kinds)
        self.assertIn("mini_scheduler", kinds)
        # Only the 1 isolated mini_scheduler has an OpSwitch (with <= 3 segments, NOT the whole shader!).
        switches = re.findall(r"OpSwitch\s+%\w+\s+%\w+([^\n]*)", spv)
        self.assertEqual(len(switches), 1)
        self.assertLessEqual(len(switches[0].split()) // 2, 3)

        # Direct ISA-derived reference values for all 128 lanes:
        # Wave 0 (0..63): skips guard 1 (s16=7, v4 = l + 7); enters guard 2 (s18 = 37 + 7 = 44,
        #   v4 = l + 51, then adds partner (l^32) + 51 -> v4_w0(l) = l + (l^32) + 102).
        # Wave 1 (64..127): enters guard 1 (s16=37; l in 64..95 -> l + 37; l in 96..127 -> l + 74); skips guard 2.
        # After s_barrier: each lane adds v4 of (l ^ 64).
        def isa_v4_before_sbar(l):
            if l < 64:
                return l + (l ^ 32) + 102
            return l + (74 if l >= 96 else 37)

        orig_plan = wave64_emulate_opt.plan_structured_emulation
        wave64_emulate_opt.plan_structured_emulation = lambda *a, **kw: None
        try:
            spv_base, _ = _translate_encoded("cs_hybrid_128_base", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
        finally:
            wave64_emulate_opt.plan_structured_emulation = orig_plan
        mem_base, _ = run_emulated_workgroup(spv_base, (128, 1, 1), push_constants=VSHARP + [0] * 28)
        for rev in (False, True):
            mem_new, _ = run_emulated_workgroup(spv, (128, 1, 1), push_constants=VSHARP + [0] * 28, reverse_order=rev)
            self.assertEqual(mem_new, mem_base)
            for l in range(128):
                want = isa_v4_before_sbar(l) + isa_v4_before_sbar(l ^ 64)
                self.assertEqual(mem_new[1024 + l * 4], want, f"lane {l} rev={rev}")

    def test_multiwave_divergent_sbarrier_falls_back_to_rounds(self):
        # Wave 0 hits s_barrier in one arm while Wave 1 hits s_barrier in another -> MUST keep global rounds.
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("v_lshlrev_b32_e32 v2, 2, v0", 4),
            ("ds_write_b32 v2, v0", 8),
            ("v_readfirstlane_b32 s5, v0", 4),
            ("s_cmp_lt_u32 s5, 64", 4),
            ("s_cbranch_scc0 8", 4),
            ("s_barrier", 4),
            ("v_xor_b32_e32 v3, 64, v0", 4),
            ("v_lshlrev_b32_e32 v3, 2, v3", 4),
            ("ds_read_b32 v4, v3", 8),
            ("buffer_store_dword v4, v2, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
            ("s_barrier", 4),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_div_sbar_w64", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(gen.control_flow, "emulated wave64 rounds")
        self.assertIsNone(gen.emulate_plan)
        mem, _ = run_emulated_workgroup(spv, (128, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], lane + 64, f"lane {lane}")

    def test_monotone_dirty_transfer_no_phantom_clear_after_x_round(self):
        # Regression: an x_round (v_readlane_b32) clears g.x_dirty in codegen before a subsequent
        # buffer_store + ds_read. A non-monotone transfer_superblock that cleared on phantom conflicts
        # with an earlier block's ('lds','w') wiped ('mem','w') from dirty_in[blk3] and dropped the
        # OpControlBarrier required before blk3's cross-half buffer_load.
        insns = [
            ("s_or_b64 exec, exec, exec", 4),
            ("s_mov_b32 s4, 0", 4),
            ("s_cmp_eq_u32 s4, 0", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("v_add_nc_u32_e32 v1, 7, v0", 4),
            ("ds_write_b32 v10, v1", 8),
            ("s_cbranch_scc0 1", 4),
            ("s_nop 0", 4),
            ("v_readlane_b32 s5, v0, 33", 4),
            ("buffer_store_dword v1, v10, s[0:3], 0 offen", 8),
            ("ds_read_b32 v2, v10", 8),
            ("s_branch 0", 4),
            # Read the opposite half of the wave (lane ^ 32) from global memory: requires a barrier!
            ("v_xor_b32_e32 v11, 32, v0", 4),
            ("v_lshlrev_b32_e32 v11, 2, v11", 4),
            ("buffer_load_dword v3, v11, s[0:3], 0 offen", 8),
            ("v_add_nc_u32_e32 v3, v3, v2", 4),
            ("buffer_store_dword v3, v10, s[0:3], 0 offen offset:512", 8),
            ("s_endpgm", 4),
        ]
        for lx, expected_mode in ((64, "inline_all"), (128, "structured_hybrid")):
            spv, gen = _translate_encoded(f"cs_monotone_dirty_{lx}", insns, (lx, 1, 1), emulate=True, rsrc2=0x20088)
            self.assertEqual(gen.control_flow, "emulated wave64 structured")
            self.assertEqual(gen.emulate_plan["mode"], expected_mode)
            self.assertIn(("mem", "w"), gen.emulate_plan["dirty_in"][3])
            lines = spv.splitlines()
            i2 = next(i for i, l in enumerate(lines) if l.strip().startswith("%blk2 = OpLabel"))
            i3 = next(i for i, l in enumerate(lines) if l.strip().startswith("%blk3 = OpLabel"))
            blk2_to_end = lines[i2:]
            store_idx = next(i for i, l in enumerate(blk2_to_end)
                             if "OpStore" in l and "OpConvertUToPtr" in "\n".join(blk2_to_end[max(0, i - 15):i]))
            i3_rel = i3 - i2
            load_idx = next(i for i, l in enumerate(blk2_to_end[i3_rel:], i3_rel)
                            if "OpLoad %uint" in l and "OpConvertUToPtr" in "\n".join(blk2_to_end[max(0, i - 15):i]))
            self.assertTrue(any("OpControlBarrier" in l for l in blk2_to_end[store_idx:load_idx]),
                            f"missing OpControlBarrier between blk2 buffer_store and blk3 buffer_load ({expected_mode})")
            for rev in (False, True):
                mem, _ = run_emulated_workgroup(spv, (lx, 1, 1), push_constants=VSHARP + [0] * 28, reverse_order=rev)
                for lane in range(lx):
                    partner = lane ^ 32
                    self.assertEqual(mem[512 + lane * 4], (partner + 7) + (lane + 7), f"lx={lx} lane={lane} rev={rev}")

    def test_known_pc_guard_unwrap_and_x_pre_helper_functions(self):
        # 1) Sequential un-foldable diamonds (live-out scalar s16) in inline_all (64 threads) and inside a loop:
        #    _unwrap_known_pc_guards must unwrap every post-dominating spine join block (no `if (%pc == join)`
        #    guard around join blocks), and x_round must outline pre-barrier setup into %x_pre_vote_fn / %x_pre_data_fn.
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("s_mov_b32 s16, 1", 4),
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            # Diamond 1 (live-out s16 -> not folded by find_local_exec_folds; joins at Spine 1)
            ("v_cmp_le_u32_e64 s[6:7], 16, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 2", 4),
            ("s_add_u32 s16, s16, 10", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            # Spine 1 (block 2): must be unwrapped at top level in inline_all!
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_readfirstlane_b32 s17, v4", 4),
            ("v_readlane_b32 s18, v4, 0", 4),
            # Diamond 2 (live-out s16 -> joins at Spine 2)
            ("v_cmp_le_u32_e64 s[6:7], 48, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 2", 4),
            ("s_add_u32 s16, s16, 20", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            # Spine 2 (block 4): must ALSO be unwrapped at top level in inline_all!
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_add_nc_u32_e32 v4, s16, v4", 4),
            ("v_add_nc_u32_e32 v4, s17, v4", 4),
            ("v_add_nc_u32_e32 v4, s18, v4", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_unwrap_spine", insns, (64, 1, 1), emulate=True)
        self.assertEqual(gen.emulate_plan["mode"], "inline_all")
        top_kinds = [(it[0], it[1]) for it in gen.emulate_plan["tree"]]
        # Blocks 0, 2, 4 are all direct top-level ("block", b) items; only blocks 1 and 3 are guarded!
        self.assertEqual(top_kinds, [("block", 0), ("guard", 1), ("block", 2), ("guard", 3), ("block", 4)])
        self.assertIn("%x_pre_vote_fn = OpFunction", spv)
        self.assertIn("%x_pre_both_fn = OpFunction", spv)
        self.assertIn("%x_pre_data_fn = OpFunction", spv)
        self.assertGreaterEqual(spv.count("OpFunctionCall %uint %x_pre_vote_fn"), 2)
        self.assertEqual(spv.count("OpFunctionCall %uint %x_pre_both_fn"), 1)
        self.assertEqual(spv.count("OpFunctionCall %uint %x_pre_data_fn"), 1)
        # Verify binary SPIR-V assembly with tools/spirv_asm.py
        import spirv_asm
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(binary[:4], struct.pack("<I", 0x07230203))
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for l in range(64):
            # After diamond 1: s16 = 11; v4 = l + 11 if l >= 16 else l; s17 = s18 = v4[0] = 0
            v4 = l + (11 if l >= 16 else 0)
            # After diamond 2: s16 = 31; v4 += 31 if l >= 48; then spine 2 adds s16 (31) + s17 (0) + s18 (0)
            if l >= 48:
                v4 += 31
            v4 += 31
            self.assertEqual(mem[l * 4], v4, f"lane {l}")

        # 2) Guard-spine coalescing after an early-exit branch (0 -> [1 (end), 2]) and inside a header-exit loop:
        #    cfg_structure.structure hoists spine guards into the enclosing list where skip_pc holds the early-exit
        #    target. _unwrap_known_pc_guards must coalesce and unwrap all hoisted post-dominating spine guards.
        import cfg_structure
        succ_early = [[1, 2], []]  # block 0 -> early exit 1 or main body 2
        cur = 2
        for _ in range(10):
            succ_early.append([cur + 1, cur + 2])
            succ_early.append([cur + 2])
            cur += 2
        succ_early.append([])
        raw_tree = cfg_structure.structure(succ_early)
        out_pcs = {b: (frozenset(s) if s else frozenset({-1})) for b, s in enumerate(succ_early)}
        unwrapped_tree = wave64_emulate_opt._unwrap_known_pc_guards(raw_tree, out_pcs, 0)

        def count_guards(t):
            c = 0
            st = [t]
            while st:
                for item in st.pop():
                    if item[0] == "guard":
                        c += 1
                        st.append(item[2])
                    elif item[0] == "loop":
                        st.append(item[2])
            return c

        # Raw tree has 22 guards (guard 1, guard 2, plus 10 arm guards and 10 hoisted spine guards);
        # coalesced unwrap removes all 10 hoisted spine guards -> 12 guards.
        self.assertEqual(count_guards(raw_tree), 22)
        self.assertEqual(count_guards(unwrapped_tree), 12)

    def test_slist_and_src_scc_liveness_and_scalar_rw(self):
        sgpr = lambda r, n=1: {"k": "sgpr", "r": r, "n": n}
        vgpr = lambda r, n=1: {"k": "vgpr", "r": r, "n": n}
        spec = lambda t: {"k": "special", "t": t}
        slist = lambda *regs: {"k": "slist", "r": list(regs), "n": len(regs)}

        # reads_scc & scc_live_outs: explicit `src_scc` operand keeps the producer's SCC live.
        prod = {"op": "s_and_saveexec_b64", "args": [sgpr(8, 2), spec("vcc")], "mods": ()}
        cons = {"op": "s_mov_b32", "args": [sgpr(10), spec("src_scc")], "mods": ()}
        end = {"op": "s_endpgm", "args": [], "mods": ()}
        self.assertTrue(wave64_emulate_opt.reads_scc(cons))
        self.assertIn(id(prod), wave64_emulate_opt.scc_live_outs([[prod, cons, end]], [[]]))
        rd_scc, _, _ = wave64_emulate_opt._scalar_rw(cons)
        self.assertEqual(rd_scc, 1 << wave64_emulate_opt._BIT_SCC)

        # _scalar_rw on sgpr-pair slist copy/load.
        load_pair = {"op": "s_buffer_load_dwordx2", "args": [slist(12, 13), slist(4, 5, 6, 7), {"k": "imm", "v": 0}], "mods": ()}
        rd, mw, mk = wave64_emulate_opt._scalar_rw(load_pair)
        self.assertEqual(rd, (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7))
        self.assertEqual(mw, (1 << 12) | (1 << 13))
        self.assertEqual(mk, (1 << 12) | (1 << 13))

        # Partial vcc_lo in an slist: may_w includes VCC, must_kill kills s105 only (vcc_hi is preserved).
        load_half = {"op": "s_buffer_load_dwordx2", "args": [slist(105, "vcc_lo"), sgpr(0, 4), {"k": "imm", "v": 0}], "mods": ()}
        rd, mw, mk = wave64_emulate_opt._scalar_rw(load_half)
        self.assertEqual(rd, 0xF)
        self.assertEqual(mw, (1 << 105) | (1 << wave64_emulate_opt._BIT_VCC))
        self.assertEqual(mk, 1 << 105)

        # Full [vcc_lo, vcc_hi] slist overwrites both halves -> must_kill includes VCC.
        load_vcc = {"op": "s_buffer_load_dwordx4", "args": [slist(104, 105, "vcc_lo", "vcc_hi"), sgpr(0, 4), {"k": "imm", "v": 0}], "mods": ()}
        rd, mw, mk = wave64_emulate_opt._scalar_rw(load_vcc)
        self.assertEqual(mw, (1 << 104) | (1 << 105) | (1 << wave64_emulate_opt._BIT_VCC))
        self.assertEqual(mk, (1 << 104) | (1 << 105) | (1 << wave64_emulate_opt._BIT_VCC))

        # VOP3b carry-out second destination as an slist.
        add_co = {"op": "v_add_co_u32_e64", "args": [vgpr(0), slist(4, 5), vgpr(1), vgpr(2)], "mods": ()}
        rd, mw, mk = wave64_emulate_opt._scalar_rw(add_co)
        self.assertEqual(rd, 0)
        self.assertEqual(mw, (1 << 4) | (1 << 5))
        self.assertEqual(mk, (1 << 4) | (1 << 5))

    def test_slist_lane_mask_and_barrier_classification(self):
        sgpr = lambda r, n=1: {"k": "sgpr", "r": r, "n": n}
        slist = lambda *regs: {"k": "slist", "r": list(regs), "n": len(regs)}

        pair_45 = slist(4, 5)
        pair_89 = slist(8, 9)
        vcc_pair = slist("vcc_lo", "vcc_hi")

        bcnt = {"op": "s_bcnt1_i32_b64", "args": [sgpr(0), pair_45], "mods": ()}
        lm_45 = {id(bcnt): {4: "mask", 5: "mask"}}
        self.assertTrue(wave64_emulate_opt._holds_lane_mask(bcnt, pair_45, lm_45[id(bcnt)]))
        self.assertFalse(wave64_emulate_opt._holds_lane_mask(bcnt, pair_89, lm_45[id(bcnt)]))
        self.assertTrue(wave64_emulate_opt._is_mask_op(bcnt, vcc_pair, {}))
        self.assertTrue(wave64_emulate_opt._insn_may_barrier(bcnt, lm_45, set()))
        clean, _, _, _, _ = wave64_emulate_opt._classify_block_body([bcnt], lm_45, set(), frozenset())
        self.assertFalse(clean)

        # Live-SCC mask vote with [vcc_lo, vcc_hi] destination must be treated as a barrier / non-foldable.
        and_vcc = {"op": "s_and_b64", "args": [vcc_pair, sgpr(6, 2), sgpr(8, 2)], "mods": ()}
        self.assertTrue(wave64_emulate_opt._insn_may_barrier(and_vcc, {}, {id(and_vcc)}))
        clean, _, _, _, _ = wave64_emulate_opt._classify_block_body([and_vcc], {}, {id(and_vcc)}, frozenset())
        self.assertFalse(clean)

        # _is_uniform_scc_safe rejects 32-bit logic with live SCC reading a masked slist.
        and_mask32 = {"op": "s_and_b32", "args": [sgpr(8), pair_45, sgpr(6)], "mods": ()}
        self.assertFalse(wave64_emulate_opt._is_uniform_scc_safe([[and_mask32]], {id(and_mask32): {4: "mask"}}, {id(and_mask32)}))

    def test_slist_live_out_scalar_and_src_scc_end_to_end(self):
        # 1) Live-out scalar written via FSPX slist `[s105,vcc_lo]` inside s_cbranch_execz must NOT fold.
        insns_live = [
            ("v_mov_b32_e32 v10, 42", 4),
            ("v_mov_b32_e32 v11, 0", 4),
            ("buffer_store_dword v10, v11, s[0:3], 0 offen", 8),
            ("s_mov_b32 s105, 0", 4),
            ("s_mov_b32 exec_lo, 0", 4),
            ("s_mov_b32 exec_hi, 0xff", 4),                # lanes 32..39 active; lanes 0..31 inactive
            ("s_cbranch_execz 2", 4),                      # skip 2 dwords (8-byte SMEM load)
            ("s_buffer_load_dwordx2 [s105,vcc_lo], s[0:3], 0", 8),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s105", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 256, v6", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_slist_live_scalar", insns_live, (64, 1, 1), emulate=True)
        self.assertGreaterEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[256 + lane * 4], 42, f"lane {lane}: stale s105 after slist load in s_cbranch_execz")

        # 2) Explicit `src_scc` read after s_and_saveexec_b64 must keep SCC live (vote across all 64 lanes).
        insns_scc = [
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),        # only lanes 32..63 active -> wave SCC = 1
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_mov_b32 s10, src_scc", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_mov_b32_e32 v5, s10", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_src_scc_live", insns_scc, (64, 1, 1), emulate=True)
        self.assertGreaterEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 1, f"lane {lane}: src_scc lost wave vote after s_and_saveexec_b64")

    def test_bounded_opcode_caches_and_superblock_dirty_transfer(self):
        # 1) Bounded caches must stay within _OP_CACHE_MAX even with > _OP_CACHE_MAX synthetic opcodes.
        limit = wave64_emulate_opt._OP_CACHE_MAX
        wave64_emulate_opt._BASE_OP_CACHE.clear()
        wave64_emulate_opt._OP_META_CACHE.clear()
        for i in range(limit + 32):
            op = f"v_synth_op_{i}_e32"
            self.assertEqual(wave64_emulate_opt.base_op(op), f"v_synth_op_{i}")
            meta = wave64_emulate_opt._get_op_meta(op)
            self.assertEqual(meta[0], f"v_synth_op_{i}")
        self.assertLessEqual(len(wave64_emulate_opt._BASE_OP_CACHE), limit)
        self.assertLessEqual(len(wave64_emulate_opt._OP_META_CACHE), limit)

        # 2) Precomputed superblock dirty transfer: s_barrier clears earlier dirty state while a
        #    subsequent folded region's mem_reads still propagates to the successor block.
        sgpr = lambda r, n=1: {"k": "sgpr", "r": r, "n": n}
        vgpr = lambda r, n=1: {"k": "vgpr", "r": r, "n": n}
        spec = lambda t: {"k": "special", "t": t}
        imm = lambda v: {"k": "imm", "v": v}
        ins_se = {"op": "s_and_saveexec_b64", "args": [sgpr(8, 2), spec("vcc")], "mods": ()}
        blocks = [
            [
                {"op": "ds_write_b32", "args": [vgpr(0), vgpr(1)], "mods": ()},
                {"op": "s_barrier", "args": [], "mods": ()},
                ins_se,
                {"op": "s_cbranch_execz", "args": [imm(2)], "mods": ()},
            ],
            [
                {"op": "buffer_load_dword", "args": [vgpr(2), vgpr(0), sgpr(0, 4), imm(0)], "mods": ()},
            ],
            [
                {"op": "s_or_b64", "args": [spec("exec"), spec("exec"), sgpr(8, 2)], "mods": ()},
                {"op": "s_cmp_eq_u32", "args": [sgpr(4), imm(0)], "mods": ()},
                {"op": "s_cbranch_scc0", "args": [imm(1)], "mods": ()},
            ],
            [
                {"op": "buffer_store_dword", "args": [vgpr(2), vgpr(0), sgpr(0, 4), imm(0)], "mods": ()},
                {"op": "s_endpgm", "args": [], "mods": ()},
            ],
            [
                {"op": "s_endpgm", "args": [], "mods": ()},
            ],
        ]
        raw_edges = [[2, 1], [2], [4, 3], [], []]
        succ = [[2, 1], [2], [4, 3], [], []]
        lm = {id(ins_se): {"vcc": True}}
        scc_live = wave64_emulate_opt.scc_live_outs(blocks, succ)
        folds, fold_chain, absorbed = wave64_emulate_opt.find_local_exec_folds(blocks, raw_edges, succ, lm, scc_live)
        self.assertIn(0, folds)
        self.assertEqual(fold_chain.get(0), 2)
        plan = wave64_emulate_opt.plan_structured_emulation(
            blocks, raw_edges, succ, folds, fold_chain, absorbed, lm, scc_live, 64
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["mode"], "inline_all")
        # s_barrier cleared ("lds", "w"), and folded block 1 added ("mem", "r") after it.
        self.assertEqual(plan["dirty_in"][3], frozenset({("mem", "r")}))
        self.assertEqual(plan["dirty_in"][4], frozenset({("mem", "r")}))


if __name__ == "__main__":
    unittest.main()
