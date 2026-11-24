"""Sound dominator-tree SPIR-V optimizer for tools/rdna2spirv.py output (#884).

Targets the main source of SPIR-V size and driver compile-time bloat in translated
RDNA2 shaders: per-instruction Function-scope register array OpAccessChain / OpLoad /
OpStore traffic, repeated OpLoad %bool %active guards (including across blocks when
%active is never stored or along single-predecessor CFG edges), duplicate immutable
Input / PushConstant loads, OpSelect on constant booleans, OpCopyObject wrappers,
round-trip OpBitcast pairs, integer/bit constant folding and algebraic identities,
trivial OpPhi nodes, dead local stores, dead Function-scope OpVariables, and dead
SSA temporaries.

Every transform is strictly sound:
  - Only Function-scope OpVariables that do not escape (never passed to OpFunctionCall,
    OpAtomic*, OpCopyMemory, etc.) are tracked.
  - Dynamic OpAccessChain indices (e.g. v_movrels / s_movrels) are detected per variable:
    dynamic stores invalidate all cached slots of that variable in the current block;
    dynamic loads prevent killing earlier stores to that variable; variables with any
    dynamic index are excluded from whole-function dead-slot and never-stored-slot
    elimination.
  - Pure value / OpAccessChain / immutable (Input, PushConstant) OpLoad CSE runs in
    dominator-tree preorder, guaranteeing SSA dominance for every replacement (including
    OpPhi operands).
  - Mutable Function-scope slot forwarding inherits across single-predecessor CFG edges
    (where the sole predecessor strictly dominates the block) and from whole-function
    read-only initializers (never-stored variables or never-stored static slots of
    zero-initialized variables), resetting at multi-predecessor joins and loop headers.
    Dead overwritten-store elimination stays strictly intra-block.
  - Constant folding only replaces an instruction when an existing module OpConstant /
    OpConstantTrue / OpConstantFalse already matches the result type and value.
  - Pure DCE uses an explicit allowlist of side-effect-free opcodes and drops any
    OpDecorate / OpName lines that reference deleted SSA IDs.

As a module:
    from spirv_opt import optimize
    opt_spvasm = optimize(spvasm)

As a CLI:
    python3 -I tools/spirv_opt.py input.spvasm -o output.spvasm
Usage: `python -I tools/spirv_opt.py --help` lists the options.
"""
import argparse
import sys

_PURE_OPS = frozenset((
    "OpAccessChain",
    "OpCopyObject",
    "OpSelect",
    "OpBitcast",
    "OpIAdd", "OpISub", "OpIMul", "OpUDiv", "OpSDiv", "OpUMod", "OpSMod", "OpSRem",
    "OpFAdd", "OpFSub", "OpFMul", "OpFDiv", "OpFNegate", "OpFmaKHR",
    "OpBitwiseOr", "OpBitwiseAnd", "OpBitwiseXor", "OpNot",
    "OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic",
    "OpBitFieldUExtract", "OpBitFieldSExtract", "OpBitFieldInsert",
    "OpBitReverse", "OpBitCount",
    "OpIEqual", "OpINotEqual",
    "OpUGreaterThan", "OpUGreaterThanEqual", "OpULessThan", "OpULessThanEqual",
    "OpSGreaterThan", "OpSGreaterThanEqual", "OpSLessThan", "OpSLessThanEqual",
    "OpFOrdEqual", "OpFOrdNotEqual",
    "OpFOrdGreaterThan", "OpFOrdGreaterThanEqual", "OpFOrdLessThan", "OpFOrdLessThanEqual",
    "OpFUnordEqual", "OpFUnordNotEqual",
    "OpFUnordGreaterThan", "OpFUnordGreaterThanEqual", "OpFUnordLessThan", "OpFUnordLessThanEqual",
    "OpIsNan", "OpIsInf",
    "OpLogicalAnd", "OpLogicalOr", "OpLogicalNot", "OpLogicalEqual", "OpLogicalNotEqual",
    "OpCompositeConstruct", "OpCompositeExtract", "OpCompositeInsert",
    "OpVectorShuffle", "OpVectorExtractDynamic", "OpVectorTimesScalar",
    "OpConvertFToU", "OpConvertFToS", "OpConvertSToF", "OpConvertUToF",
    "OpUConvert", "OpSConvert", "OpFConvert", "OpQuantizeToF16",
    "OpIAddCarry", "OpISubBorrow", "OpUMulExtended", "OpSMulExtended",
))

_COMMUTATIVE_OPS = frozenset((
    "OpIAdd", "OpIMul", "OpBitwiseOr", "OpBitwiseAnd", "OpBitwiseXor",
    "OpIEqual", "OpINotEqual", "OpLogicalAnd", "OpLogicalOr",
    "OpLogicalEqual", "OpLogicalNotEqual",
))

_PURE_GLSL_EXT = frozenset((
    "FAbs", "SAbs", "FSign", "SSign", "Floor", "Trunc", "Round", "RoundEven",
    "Ceil", "Fract", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
    "Sinh", "Cosh", "Tanh", "Pow", "Exp", "Log", "Exp2", "Log2",
    "Sqrt", "InverseSqrt",
    "FMin", "UMin", "SMin", "FMax", "UMax", "SMax",
    "FClamp", "UClamp", "SClamp", "NMin", "NMax", "NClamp",
    "FMix", "Step", "SmoothStep", "Fma",
    "PackHalf2x16", "UnpackHalf2x16",
    "PackSnorm4x8", "PackUnorm4x8", "PackSnorm2x16", "PackUnorm2x16",
    "UnpackSnorm4x8", "UnpackUnorm4x8", "UnpackSnorm2x16", "UnpackUnorm2x16",
    "FindILsb", "FindSMsb", "FindUMsb", "Ldexp", "FrexpStruct", "ModfStruct",
))


def _parse_line(line: str):
    """Return (res_id_or_None, opname, args_list) for one SPIR-V assembly line, or None if blank/comment."""
    if '"' not in line:
        semi = line.find(";")
        toks = (line[:semi] if semi >= 0 else line).split()
    else:
        # Lines with quotes only appear in module header (OpExtension, OpEntryPoint, OpName, OpString).
        return None
    if not toks:
        return None
    if len(toks) >= 3 and toks[1] == "=":
        return toks[0], toks[2], toks[3:]
    return None, toks[0], toks[1:]


def _format_inst(res: str | None, op: str, args: list[str]) -> str:
    if res is not None:
        return f"         {res} = {op} {' '.join(args)}" if args else f"         {res} = {op}"
    return f"         {op} {' '.join(args)}" if args else f"         {op}"


def _to_signed(uval: int, width: int) -> int:
    sign_bit = 1 << (width - 1)
    return uval - (1 << width) if uval >= sign_bit else uval


def _build_cfg_and_dom(insts: list[list]):
    """Split function instructions into basic blocks and compute CFG + dominator tree."""
    blocks: list[list[int]] = []         # block_idx -> list of inst indices (starting with OpLabel)
    label_to_block: dict[str, int] = {}
    cur: list[int] | None = None
    for idx, (res, op, _args, _raw) in enumerate(insts):
        if op == "OpLabel" and res is not None:
            b = len(blocks)
            label_to_block[res] = b
            cur = [idx]
            blocks.append(cur)
        elif cur is not None:
            if op == "OpFunctionEnd":
                cur = None
            else:
                cur.append(idx)

    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks, [], [], [], {}, []

    succs: list[list[int]] = [[] for _ in range(n_blocks)]
    all_preds: list[set[int]] = [set() for _ in range(n_blocks)]

    for b, b_insts in enumerate(blocks):
        seen_s: set[int] = set()
        for idx in b_insts:
            _, op, args, _ = insts[idx]
            targets = ()
            if op == "OpBranch" and args:
                targets = (args[0],)
            elif op == "OpBranchConditional" and len(args) >= 3:
                targets = (args[1], args[2])
            elif op == "OpSwitch" and len(args) >= 2:
                targets = tuple([args[1]] + [args[k] for k in range(3, len(args), 2)])
            for t in targets:
                tb = label_to_block.get(t)
                if tb is not None:
                    all_preds[tb].add(b)
                    if tb not in seen_s:
                        seen_s.add(tb)
                        succs[b].append(tb)

    # Iterative DFS from block 0 to compute reverse postorder (RPO) of reachable blocks.
    visited = [False] * n_blocks
    postorder: list[int] = []
    stack: list[tuple[int, int]] = [(0, 0)]
    visited[0] = True
    while stack:
        b, si = stack[-1]
        if si < len(succs[b]):
            stack[-1] = (b, si + 1)
            nb = succs[b][si]
            if not visited[nb]:
                visited[nb] = True
                stack.append((nb, 0))
        else:
            stack.pop()
            postorder.append(b)

    rpo = list(reversed(postorder))
    rpo_num = {b: i for i, b in enumerate(rpo)}

    # Cooper-Harvey-Kennedy iterative dominator algorithm on reachable blocks.
    idom: dict[int, int] = {0: 0}

    def intersect(b1: int, b2: int) -> int:
        while b1 != b2:
            while rpo_num[b1] > rpo_num[b2]:
                b1 = idom[b1]
            while rpo_num[b2] > rpo_num[b1]:
                b2 = idom[b2]
        return b1

    dom_changed = True
    while dom_changed:
        dom_changed = False
        for b in rpo[1:]:
            new_idom = None
            for p in all_preds[b]:
                if p in idom:
                    new_idom = p if new_idom is None else intersect(p, new_idom)
            if new_idom is not None and idom.get(b) != new_idom:
                idom[b] = new_idom
                dom_changed = True

    dom_children: list[list[int]] = [[] for _ in range(n_blocks)]
    for b in rpo[1:]:
        p = idom.get(b)
        if p is not None and p != b:
            dom_children[p].append(b)

    return blocks, succs, all_preds, rpo, idom, dom_children


def _fold_int_op(op: str, res_ty: str, args: list[str],
                 const_int_val: dict[str, tuple[str, int, int, int]],
                 int_const_by_val: dict[tuple[str, int], str],
                 int_type_info: dict[str, tuple[int, int]],
                 const_true_id: str | None, const_false_id: str | None) -> str | None:
    """Fold constant/algebraic integer/bit operations into an existing SSA ID, or return None."""
    n = len(args)
    if op == "OpNot" and n == 2:
        c1 = const_int_val.get(args[1])
        if c1 is not None:
            ty, v1, w, _ = c1
            mask = (1 << w) - 1
            return int_const_by_val.get((res_ty, (~v1) & mask))
        return None

    if op == "OpUConvert" and n == 2:
        c1 = const_int_val.get(args[1])
        dst_info = int_type_info.get(res_ty)
        if c1 is not None and dst_info is not None:
            _, v1, _, _ = c1
            dw, _ = dst_info
            return int_const_by_val.get((res_ty, v1 & ((1 << dw) - 1)))
        return None

    if op == "OpBitFieldUExtract" and n == 4:
        base, off_id, cnt_id = args[1], args[2], args[3]
        co = const_int_val.get(off_id)
        cc = const_int_val.get(cnt_id)
        dst_info = int_type_info.get(res_ty)
        if co is not None and cc is not None and dst_info is not None:
            w, _ = dst_info
            off = co[1]
            cnt = cc[1]
            if off == 0 and cnt == w:
                return base
            cb = const_int_val.get(base)
            if cb is not None and off <= w and cnt <= w and off + cnt <= w:
                val = 0 if cnt == 0 else ((cb[1] >> off) & ((1 << cnt) - 1))
                return int_const_by_val.get((res_ty, val))
        return None

    if n != 3:
        return None

    a1, a2 = args[1], args[2]
    c1 = const_int_val.get(a1)
    c2 = const_int_val.get(a2)

    # Full constant folding when both operands are known integer constants.
    if c1 is not None and c2 is not None:
        _, v1, w1, _ = c1
        _, v2, w2, _ = c2
        mask = (1 << w1) - 1
        if op == "OpIAdd":
            return int_const_by_val.get((res_ty, (v1 + v2) & mask))
        if op == "OpISub":
            return int_const_by_val.get((res_ty, (v1 - v2) & mask))
        if op == "OpIMul":
            return int_const_by_val.get((res_ty, (v1 * v2) & mask))
        if op == "OpUDiv" and v2 != 0:
            return int_const_by_val.get((res_ty, (v1 // v2) & mask))
        if op == "OpUMod" and v2 != 0:
            return int_const_by_val.get((res_ty, (v1 % v2) & mask))
        if op == "OpBitwiseAnd":
            return int_const_by_val.get((res_ty, (v1 & v2) & mask))
        if op == "OpBitwiseOr":
            return int_const_by_val.get((res_ty, (v1 | v2) & mask))
        if op == "OpBitwiseXor":
            return int_const_by_val.get((res_ty, (v1 ^ v2) & mask))
        if op == "OpShiftLeftLogical" and v2 < w1:
            return int_const_by_val.get((res_ty, (v1 << v2) & mask))
        if op == "OpShiftRightLogical" and v2 < w1:
            return int_const_by_val.get((res_ty, (v1 >> v2) & mask))
        if op == "OpShiftRightArithmetic" and v2 < w1:
            sv1 = _to_signed(v1, w1)
            return int_const_by_val.get((res_ty, (sv1 >> v2) & mask))
        if const_true_id is not None and const_false_id is not None and w1 == w2:
            if op == "OpIEqual":
                return const_true_id if v1 == v2 else const_false_id
            if op == "OpINotEqual":
                return const_true_id if v1 != v2 else const_false_id
            if op == "OpULessThan":
                return const_true_id if v1 < v2 else const_false_id
            if op == "OpULessThanEqual":
                return const_true_id if v1 <= v2 else const_false_id
            if op == "OpUGreaterThan":
                return const_true_id if v1 > v2 else const_false_id
            if op == "OpUGreaterThanEqual":
                return const_true_id if v1 >= v2 else const_false_id
            sv1, sv2 = _to_signed(v1, w1), _to_signed(v2, w2)
            if op == "OpSLessThan":
                return const_true_id if sv1 < sv2 else const_false_id
            if op == "OpSLessThanEqual":
                return const_true_id if sv1 <= sv2 else const_false_id
            if op == "OpSGreaterThan":
                return const_true_id if sv1 > sv2 else const_false_id
            if op == "OpSGreaterThanEqual":
                return const_true_id if sv1 >= sv2 else const_false_id
        return None

    # Algebraic identities with one constant or identical operands (all bit-exact for SPIR-V ints).
    dst_info = int_type_info.get(res_ty)
    dst_mask = (1 << dst_info[0]) - 1 if dst_info is not None else None

    if op == "OpIAdd":
        if c2 is not None and c2[1] == 0:
            return a1
        if c1 is not None and c1[1] == 0:
            return a2
    elif op == "OpISub":
        if c2 is not None and c2[1] == 0:
            return a1
        if a1 == a2:
            return int_const_by_val.get((res_ty, 0))
    elif op == "OpIMul":
        if c2 is not None:
            if c2[1] == 1:
                return a1
            if c2[1] == 0:
                return a2
        if c1 is not None:
            if c1[1] == 1:
                return a2
            if c1[1] == 0:
                return a1
    elif op == "OpBitwiseAnd":
        if a1 == a2:
            return a1
        if c2 is not None:
            if c2[1] == 0:
                return a2
            if dst_mask is not None and c2[1] == dst_mask:
                return a1
        if c1 is not None:
            if c1[1] == 0:
                return a1
            if dst_mask is not None and c1[1] == dst_mask:
                return a2
    elif op == "OpBitwiseOr":
        if a1 == a2:
            return a1
        if c2 is not None:
            if c2[1] == 0:
                return a1
            if dst_mask is not None and c2[1] == dst_mask:
                return a2
        if c1 is not None:
            if c1[1] == 0:
                return a2
            if dst_mask is not None and c1[1] == dst_mask:
                return a1
    elif op == "OpBitwiseXor":
        if a1 == a2:
            return int_const_by_val.get((res_ty, 0))
        if c2 is not None and c2[1] == 0:
            return a1
        if c1 is not None and c1[1] == 0:
            return a2
    elif op in ("OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic"):
        if c2 is not None and c2[1] == 0:
            return a1
        if op != "OpShiftRightArithmetic" and c1 is not None and c1[1] == 0:
            return a1
    elif op == "OpIEqual" and a1 == a2:
        return const_true_id
    elif op == "OpINotEqual" and a1 == a2:
        return const_false_id
    return None


def _optimize_function(body_lines: list[str], const_ids: set[str], const_true: set[str],
                       const_false: set[str], const_null_elem_zero: dict[str, str],
                       const_type: dict[str, str], const_int_val: dict[str, tuple[str, int, int, int]],
                       int_const_by_val: dict[tuple[str, int], str],
                       int_type_info: dict[str, tuple[int, int]],
                       readonly_global_vars: set[str], glsl_sets: set[str],
                       decorated_ids: set[str], max_passes: int = 4) -> tuple[list[str], set[str]]:
    """Optimize one OpFunction ... OpFunctionEnd slice. Returns (new_lines, deleted_ids)."""
    insts: list[list] = []
    for raw in body_lines:
        parsed = _parse_line(raw)
        if parsed is None:
            insts.append([None, None, None, raw])
        else:
            res, op, args = parsed
            insts.append([res, op, args, raw])

    deleted_ids: set[str] = set()
    const_true_id = next(iter(const_true)) if const_true else None
    const_false_id = next(iter(const_false)) if const_false else None

    for _pass in range(max_passes):
        changed = False

        # 1. Discover Function-scope variables, pointer-to-slot mappings, escapes, dynamic indices,
        #    result types, and which variables / static slots are ever stored to.
        func_vars: dict[str, str | None] = {}       # var_id -> initializer_id or None
        ptr_root: dict[str, str] = {}               # ptr_id -> root var_id (Function or readonly global)
        ptr_slot: dict[str, tuple | None] = {}      # ptr_id -> (var_id, *canonical_const_indices) or None
        escaped: set[str] = set()
        dyn_indexed: set[str] = set()
        id_type: dict[str, str] = dict(const_type)

        for res, op, args, _ in insts:
            if op is None:
                continue
            if res is not None and args and op != "OpLabel":
                id_type[res] = args[0]
            if op == "OpVariable" and len(args) >= 2 and args[1] == "Function" and res:
                init = args[2] if len(args) >= 3 else None
                func_vars[res] = init
                ptr_root[res] = res
                ptr_slot[res] = (res,)
            elif op == "OpAccessChain" and res and len(args) >= 2:
                base = args[1]
                idxs = args[2:]
                if base in ptr_root:
                    root = ptr_root[base]
                    ptr_root[res] = root
                    base_slot = ptr_slot.get(base)
                    if base_slot is not None and all(i in const_int_val for i in idxs):
                        # Canonicalize integer index constants by unsigned value so %uint_0 and %int_0 alias.
                        cidxs = tuple(const_int_val[i][1] for i in idxs)
                        ptr_slot[res] = base_slot + cidxs
                    else:
                        ptr_slot[res] = None
                        if root in func_vars:
                            dyn_indexed.add(root)
                elif base in readonly_global_vars:
                    ptr_root[res] = base

        slot_depths: dict[str, set[int]] = {}
        for res, op, args, _ in insts:
            if op is None:
                continue
            if op == "OpLoad":
                if len(args) >= 2 and args[1] in ptr_root and ptr_root[args[1]] in func_vars:
                    root = ptr_root[args[1]]
                    if any("Volatile" in m for m in args[2:]):
                        escaped.add(root)
                    sl = ptr_slot.get(args[1])
                    if sl is not None:
                        slot_depths.setdefault(root, set()).add(len(sl))
                for tok in args[2:]:
                    if tok in ptr_root and ptr_root[tok] in func_vars:
                        escaped.add(ptr_root[tok])
            elif op == "OpStore":
                if len(args) >= 2 and args[0] in ptr_root and ptr_root[args[0]] in func_vars:
                    root = ptr_root[args[0]]
                    if any("Volatile" in m for m in args[2:]):
                        escaped.add(root)
                    sl = ptr_slot.get(args[0])
                    if sl is not None:
                        slot_depths.setdefault(root, set()).add(len(sl))
                for tok in args[1:]:
                    if tok in ptr_root and ptr_root[tok] in func_vars:
                        escaped.add(ptr_root[tok])
            elif op == "OpAccessChain":
                for tok in args[2:]:
                    if tok in ptr_root and ptr_root[tok] in func_vars:
                        escaped.add(ptr_root[tok])
            elif op != "OpVariable":
                for tok in args:
                    if tok in ptr_root and ptr_root[tok] in func_vars:
                        escaped.add(ptr_root[tok])

        # Exclude variables accessed at mixed aggregate/subobject depths (e.g. whole array + element pointer).
        for root, depths in slot_depths.items():
            if len(depths) > 1:
                escaped.add(root)

        trackable = {v for v in func_vars if v not in escaped}
        static_trackable = {v for v in trackable if v not in dyn_indexed}

        # Identify which trackable vars and static slots ever receive an OpStore in the function.
        stored_vars: set[str] = set()
        stored_slots: set[tuple] = set()
        for _res, op, args, _ in insts:
            if op == "OpStore" and len(args) >= 2:
                ptr = args[0]
                root = ptr_root.get(ptr)
                if root in trackable:
                    stored_vars.add(root)
                    slot = ptr_slot.get(ptr)
                    if slot is not None:
                        stored_slots.add(slot)

        # Function-wide immutable facts for never-stored trackable variables (e.g. %active in non-divergent shaders).
        readonly_slot_val: dict[tuple, str] = {}
        readonly_null_zero: dict[str, str] = {}
        for v in trackable:
            if v in stored_vars:
                continue
            init = func_vars[v]
            if init is None:
                continue
            if init in const_null_elem_zero:
                readonly_null_zero[v] = const_null_elem_zero[init]
            elif init in const_ids:
                readonly_slot_val[(v,)] = init

        # Never-stored static zero-initialized array variables: individual slots with zero stores are always zero.
        static_null_zero: dict[str, str] = {}
        for v in static_trackable:
            init = func_vars[v]
            if init is not None and init in const_null_elem_zero and (v,) not in stored_slots:
                static_null_zero[v] = const_null_elem_zero[init]

        # 2. Build CFG + dominator tree and run dominator-tree preorder optimization pass.
        blocks, _succs, all_preds, rpo, idom, dom_children = _build_cfg_and_dom(insts)
        def_block: dict[str, int] = {}
        for blk_i, b_insts in enumerate(blocks):
            for inst_i in b_insts:
                def_id = insts[inst_i][0]
                if def_id is not None:
                    def_block[def_id] = blk_i

        def strictly_dominates(a: int, b: int) -> bool:
            if a == b or b not in idom:
                return False
            cur = idom[b]
            while cur != a:
                nxt = idom.get(cur)
                if nxt is None or nxt == cur:
                    return False
                cur = nxt
            return True

        alias: dict[str, str] = {}

        def resolve(tok: str) -> str:
            cur = tok
            while cur in alias:
                cur = alias[cur]
            if cur != tok:
                alias[tok] = cur
            return cur

        dead_idx: set[int] = set()

        # Scoped dominator-tree tables (maintained via undo logs during iterative DFS).
        ac_cse: dict[tuple, str] = {}
        val_cse: dict[tuple, str] = {}
        ro_load_cse: dict[tuple, str] = {}        # (res_ty, ptr_id) -> loaded_id for Input / PushConstant
        bitcast_of: dict[str, tuple[str, str]] = {}   # res_id -> (src_id, src_type_id)
        bool_mask_of: dict[str, tuple[str, bool]] = {}  # res_id -> (cond_id, True if cond->nz/!cond->0 else inverted)
        ctor_elems: dict[str, list[str]] = {}     # composite_id -> list of constituent IDs
        logical_not_of: dict[str, str] = {}       # res_id -> inner bool ID

        def process_block(b: int, slot_val: dict[tuple, str], entry_null_zero: dict[str, str],
                          reachable: bool):
            nonlocal changed
            last_store: dict[tuple, int] = {}
            undo_ac: list[tuple] = []
            undo_val: list[tuple] = []
            undo_ro: list[tuple] = []
            undo_bc: list[str] = []
            undo_bm: list[str] = []
            undo_ce: list[str] = []
            undo_ln: list[str] = []

            for idx in blocks[b]:
                entry = insts[idx]
                res, op, args, _ = entry
                if op is None or op == "OpLabel":
                    continue

                if alias and args:
                    new_args = [resolve(t) if t.startswith("%") else t for t in args]
                    if new_args != args:
                        entry[2] = new_args
                        args = new_args
                        changed = True

                # Trivial OpPhi simplification (only when the sole non-self value strictly dominates this block).
                if op == "OpPhi" and reachable and res is not None and res not in decorated_ids and len(args) >= 3:
                    vals = [args[k] for k in range(1, len(args), 2)]
                    non_self = {v for v in vals if v != res}
                    if len(non_self) == 1:
                        cand = next(iter(non_self))
                        cand_blk = def_block.get(cand)
                        if cand_blk is None or strictly_dominates(cand_blk, b):
                            alias[res] = cand
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue

                # Algebraic / copy / constant-fold / round-trip bitcast simplifications
                if res is not None and res not in decorated_ids:
                    if op == "OpCopyObject" and len(args) == 2:
                        alias[res] = args[1]
                        dead_idx.add(idx)
                        deleted_ids.add(res)
                        changed = True
                        continue
                    if op == "OpBitcast" and len(args) == 2:
                        dst_ty, src = args[0], args[1]
                        prev_bc = bitcast_of.get(src)
                        if prev_bc is not None and prev_bc[1] == dst_ty:
                            alias[res] = prev_bc[0]
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        src_ty = id_type.get(src)
                        if src_ty is not None:
                            bitcast_of[res] = (src, src_ty)
                            undo_bc.append(res)
                    if op == "OpCompositeConstruct" and len(args) >= 2:
                        ctor_elems[res] = args[1:]
                        undo_ce.append(res)
                    if op == "OpCompositeExtract" and len(args) == 3 and args[2].isdigit():
                        elems = ctor_elems.get(args[1])
                        if elems is not None:
                            k = int(args[2])
                            if k < len(elems) and all(id_type.get(e) == args[0] for e in elems):
                                alias[res] = elems[k]
                                dead_idx.add(idx)
                                deleted_ids.add(res)
                                changed = True
                                continue
                    if op == "OpSelect" and len(args) == 4:
                        cond, x, y = args[1], args[2], args[3]
                        if cond in const_true or x == y:
                            alias[res] = x
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if cond in const_false:
                            alias[res] = y
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if x in const_true and y in const_false:
                            alias[res] = cond
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        cx = const_int_val.get(x)
                        cy = const_int_val.get(y)
                        if cx is not None and cy is not None:
                            if cx[1] != 0 and cy[1] == 0:
                                bool_mask_of[res] = (cond, True)
                                undo_bm.append(res)
                            elif cx[1] == 0 and cy[1] != 0:
                                bool_mask_of[res] = (cond, False)
                                undo_bm.append(res)
                    if op == "OpLogicalNot" and len(args) == 2:
                        inner = args[1]
                        if inner in const_true and const_false_id is not None:
                            alias[res] = const_false_id
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if inner in const_false and const_true_id is not None:
                            alias[res] = const_true_id
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        prev_not = logical_not_of.get(inner)
                        if prev_not is not None:
                            alias[res] = prev_not
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        logical_not_of[res] = inner
                        undo_ln.append(res)
                    if op == "OpLogicalAnd" and len(args) == 3:
                        a1, a2 = args[1], args[2]
                        if a1 == a2 or a2 in const_true:
                            alias[res] = a1
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a1 in const_true:
                            alias[res] = a2
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a1 in const_false:
                            alias[res] = a1
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a2 in const_false:
                            alias[res] = a2
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                    if op == "OpLogicalOr" and len(args) == 3:
                        a1, a2 = args[1], args[2]
                        if a1 == a2 or a2 in const_false:
                            alias[res] = a1
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a1 in const_false:
                            alias[res] = a2
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a1 in const_true:
                            alias[res] = a1
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        if a2 in const_true:
                            alias[res] = a2
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                    if op in ("OpINotEqual", "OpIEqual") and len(args) == 3:
                        a1, a2 = args[1], args[2]
                        for cand, other in ((a1, a2), (a2, a1)):
                            bm = bool_mask_of.get(cand)
                            co = const_int_val.get(other)
                            if bm is not None and co is not None and co[1] == 0:
                                cond_id, normal = bm
                                if (op == "OpINotEqual" and normal) or (op == "OpIEqual" and not normal):
                                    alias[res] = cond_id
                                    dead_idx.add(idx)
                                    deleted_ids.add(res)
                                    changed = True
                                    break
                        if idx in dead_idx:
                            continue

                    # Integer/bit constant folding & algebraic identities
                    folded = _fold_int_op(
                        op, args[0], args, const_int_val, int_const_by_val,
                        int_type_info, const_true_id, const_false_id,
                    )
                    if folded is not None:
                        alias[res] = folded
                        dead_idx.add(idx)
                        deleted_ids.add(res)
                        changed = True
                        continue

                # Dominator-scoped OpAccessChain CSE
                if op == "OpAccessChain" and res is not None and res not in decorated_ids:
                    key = tuple(args)
                    prev = ac_cse.get(key)
                    if prev is not None:
                        alias[res] = prev
                        dead_idx.add(idx)
                        deleted_ids.add(res)
                        changed = True
                        continue
                    ac_cse[key] = res
                    undo_ac.append(key)
                    continue

                # Dominator-scoped pure-value CSE (with commutative operand canonicalization)
                if res is not None and res not in decorated_ids and (
                    op in _PURE_OPS or (op == "OpExtInst" and len(args) >= 3 and args[1] in glsl_sets and args[2] in _PURE_GLSL_EXT)
                ):
                    if op in _COMMUTATIVE_OPS and len(args) == 3 and args[1] > args[2]:
                        key = (op, (args[0], args[2], args[1]))
                    else:
                        key = (op, tuple(args))
                    prev = val_cse.get(key)
                    if prev is not None:
                        alias[res] = prev
                        dead_idx.add(idx)
                        deleted_ids.add(res)
                        changed = True
                        continue
                    val_cse[key] = res
                    undo_val.append(key)
                    continue

                # OpLoad forwarding & CSE (Function-scope slots + immutable Input/PushConstant globals)
                if op == "OpLoad" and res is not None and len(args) >= 2:
                    ptr = args[1]
                    volatile = any("Volatile" in m for m in args[2:])
                    root = ptr_root.get(ptr)
                    if not volatile and root in trackable:
                        slot = ptr_slot.get(ptr)
                        if slot is not None:
                            cached = slot_val.get(slot)
                            if cached is None and len(slot) == 2:
                                if root in entry_null_zero:
                                    cached = entry_null_zero[root]
                                elif root in static_null_zero and slot not in stored_slots:
                                    cached = static_null_zero[root]
                            if cached is not None and res not in decorated_ids:
                                alias[res] = cached
                                dead_idx.add(idx)
                                deleted_ids.add(res)
                                changed = True
                                continue
                            slot_val[slot] = res
                            last_store.pop(slot, None)
                        else:
                            # Dynamic load from root: protect all pending intra-block stores to root.
                            for s in [k for k in last_store if k[0] == root]:
                                last_store.pop(s, None)
                    elif not volatile and (root in readonly_global_vars or ptr in readonly_global_vars) and res not in decorated_ids:
                        ro_key = (args[0], ptr)
                        prev = ro_load_cse.get(ro_key)
                        if prev is not None:
                            alias[res] = prev
                            dead_idx.add(idx)
                            deleted_ids.add(res)
                            changed = True
                            continue
                        ro_load_cse[ro_key] = res
                        undo_ro.append(ro_key)
                    continue

                # Function-scope OpStore tracking & redundant/overwritten store elimination
                if op == "OpStore" and len(args) >= 2:
                    ptr, val = args[0], args[1]
                    volatile = any("Volatile" in m for m in args[2:])
                    root = ptr_root.get(ptr)
                    if not volatile and root in trackable:
                        slot = ptr_slot.get(ptr)
                        if slot is not None:
                            cur_val = slot_val.get(slot)
                            if cur_val is None and root in entry_null_zero and len(slot) == 2:
                                cur_val = entry_null_zero[root]
                            if cur_val == val:
                                dead_idx.add(idx)
                                changed = True
                                continue
                            prev_st = last_store.get(slot)
                            if prev_st is not None:
                                dead_idx.add(prev_st)
                                changed = True
                            slot_val[slot] = val
                            last_store[slot] = idx
                        else:
                            entry_null_zero.pop(root, None)
                            for s in [k for k in slot_val if k[0] == root]:
                                slot_val.pop(s, None)
                            for s in [k for k in last_store if k[0] == root]:
                                last_store.pop(s, None)
                    continue

            return undo_ac, undo_val, undo_ro, undo_bc, undo_bm, undo_ce, undo_ln

        if blocks:
            # Initial state for entry block 0.
            entry_sv = dict(readonly_slot_val)
            entry_nz = dict(readonly_null_zero)
            if not all_preds[0]:
                for v in trackable:
                    init = func_vars[v]
                    if init is None:
                        continue
                    if init in const_null_elem_zero:
                        entry_nz[v] = const_null_elem_zero[init]
                    elif init in const_ids:
                        entry_sv[(v,)] = init

            # Iterative dominator-tree preorder DFS.
            dfs: list[tuple] = [("enter", 0, entry_sv, entry_nz)]
            while dfs:
                frame = dfs.pop()
                if frame[0] == "enter":
                    _, b, sv, nz = frame
                    undos = process_block(b, sv, nz, reachable=True)
                    dfs.append(("exit", undos))
                    for ch in reversed(dom_children[b]):
                        if all_preds[ch] == {b}:
                            dfs.append(("enter", ch, dict(sv), dict(nz)))
                        else:
                            dfs.append(("enter", ch, dict(readonly_slot_val), dict(readonly_null_zero)))
                else:
                    undo_ac, undo_val, undo_ro, undo_bc, undo_bm, undo_ce, undo_ln = frame[1]
                    for k in undo_ac:
                        del ac_cse[k]
                    for k in undo_val:
                        del val_cse[k]
                    for k in undo_ro:
                        del ro_load_cse[k]
                    for k in undo_bc:
                        del bitcast_of[k]
                    for k in undo_bm:
                        del bool_mask_of[k]
                    for k in undo_ce:
                        del ctor_elems[k]
                    for k in undo_ln:
                        del logical_not_of[k]

            # Process any unreachable blocks (in case spirv-val inspects them) with fresh state.
            reachable_set = set(rpo)
            for b in range(len(blocks)):
                if b not in reachable_set:
                    undos = process_block(b, dict(readonly_slot_val), dict(readonly_null_zero), reachable=False)
                    undo_ac, undo_val, undo_ro, undo_bc, undo_bm, undo_ce, undo_ln = undos
                    for k in undo_ac:
                        del ac_cse[k]
                    for k in undo_val:
                        del val_cse[k]
                    for k in undo_ro:
                        del ro_load_cse[k]
                    for k in undo_bc:
                        del bitcast_of[k]
                    for k in undo_bm:
                        del bool_mask_of[k]
                    for k in undo_ce:
                        del ctor_elems[k]
                    for k in undo_ln:
                        del logical_not_of[k]

        if dead_idx:
            insts = [e for i, e in enumerate(insts) if i not in dead_idx]

        # Post-walk alias resolution sweep (updates backedge operands of OpPhi, etc.).
        if alias:
            for entry in insts:
                _, op, args, _ = entry
                if op is None or not args:
                    continue
                new_args = [resolve(t) if t.startswith("%") else t for t in args]
                if new_args != args:
                    entry[2] = new_args
                    changed = True

        # 3. Whole-function dead-slot store elimination for non-escaped, statically-indexed Function vars:
        #    if a slot (var, ...) has zero remaining OpLoads anywhere in the function, every OpStore to it is dead.
        if static_trackable:
            loaded_slots: set[tuple] = set()
            for _res, op, args, _ in insts:
                if op == "OpLoad" and len(args) >= 2:
                    slot = ptr_slot.get(args[1])
                    if slot is not None and slot[0] in static_trackable:
                        loaded_slots.add(slot)
            dead_stores: set[int] = set()
            for idx, (_res, op, args, _) in enumerate(insts):
                if op == "OpStore" and len(args) >= 2:
                    if any("Volatile" in m for m in args[2:]):
                        continue
                    slot = ptr_slot.get(args[0])
                    if slot is not None and slot[0] in static_trackable and slot not in loaded_slots:
                        dead_stores.add(idx)
            if dead_stores:
                insts = [e for i, e in enumerate(insts) if i not in dead_stores]
                changed = True

        # 4. Backward pure Dead-Code Elimination (DCE) + dead Function-scope OpVariable removal.
        use_count: dict[str, int] = {}
        for _res, op, args, _ in insts:
            if op is None or not args:
                continue
            for tok in args:
                if tok.startswith("%"):
                    use_count[tok] = use_count.get(tok, 0) + 1

        def is_pure_dce_candidate(res: str, op: str, args: list[str]) -> bool:
            if op in _PURE_OPS or op == "OpPhi":
                return True
            if op == "OpExtInst" and len(args) >= 3 and args[1] in glsl_sets and args[2] in _PURE_GLSL_EXT:
                return True
            if op == "OpLoad" and len(args) >= 2 and not any("Volatile" in m for m in args[2:]):
                ptr = args[1]
                root = ptr_root.get(ptr)
                return root in trackable or root in readonly_global_vars or ptr in readonly_global_vars
            if op == "OpVariable" and len(args) >= 2 and args[1] == "Function" and res in trackable:
                return True
            return False

        dce_dead: set[int] = set()
        for idx in range(len(insts) - 1, -1, -1):
            res, op, args, _ = insts[idx]
            if res is None or op is None:
                continue
            if use_count.get(res, 0) == 0 and is_pure_dce_candidate(res, op, args):
                dce_dead.add(idx)
                deleted_ids.add(res)
                for tok in args:
                    if tok.startswith("%"):
                        use_count[tok] = use_count.get(tok, 0) - 1

        if dce_dead:
            insts = [e for i, e in enumerate(insts) if i not in dce_dead]
            changed = True

        if not changed:
            break

    out_lines = []
    for res, op, args, raw in insts:
        if op is None or op in ("OpFunction", "OpFunctionEnd", "OpLabel", "OpVariable"):
            out_lines.append(raw)
        else:
            out_lines.append(_format_inst(res, op, args))
    return out_lines, deleted_ids


def optimize(spvasm: str, max_passes: int = 4) -> str:
    """Apply sound dominator-tree SPIR-V optimizations to a SPIR-V assembly module."""
    lines = spvasm.splitlines()

    # Scan module header for constants, integer types, immutable globals, GLSL ext sets, decorations.
    const_ids: set[str] = set()
    const_true: set[str] = set()
    const_false: set[str] = set()
    const_type: dict[str, str] = {}
    int_type_info: dict[str, tuple[int, int]] = {}          # int_type_id -> (width, signedness)
    const_int_val: dict[str, tuple[str, int, int, int]] = {}  # const_id -> (type_id, uval, width, signedness)
    int_const_by_val: dict[tuple[str, int], str] = {}       # (type_id, uval) -> const_id
    array_elem_type: dict[str, str] = {}                    # arr_type_id -> elem_type_id
    zero_const_of_type: dict[str, str] = {}                 # scalar_type_id -> zero_const_id
    const_null_type: dict[str, str] = {}                    # null_const_id -> array_type_id
    readonly_global_vars: set[str] = set()                  # PushConstant and Input OpVariables
    glsl_sets: set[str] = set()
    decorated_ids: set[str] = set()

    func_ranges: list[tuple[int, int]] = []
    fn_start = None
    for i, raw in enumerate(lines):
        if fn_start is None:
            if "OpExtInstImport" in raw and '"GLSL.std.450"' in raw:
                lhs = raw.split("=", 1)[0].strip()
                if lhs.startswith("%"):
                    glsl_sets.add(lhs)
                continue
            stripped = raw.lstrip()
            op_tok = stripped.split(None, 1)[0] if stripped else ""
            if op_tok in ("OpDecorateString", "OpDecorateStringGOOGLE"):
                toks = stripped.split(None, 2)
                if len(toks) >= 2 and toks[1].startswith("%"):
                    decorated_ids.add(toks[1])
                continue
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            res, op, args = parsed
            if op == "OpFunction":
                fn_start = i
                continue
            if op == "OpDecorate" and args:
                decorated_ids.add(args[0])
            elif op == "OpDecorateId" and args:
                decorated_ids.add(args[0])
                for tok in args[2:]:
                    if tok.startswith("%"):
                        decorated_ids.add(tok)
            elif op == "OpGroupDecorate" and len(args) >= 2:
                for tok in args[1:]:
                    if tok.startswith("%"):
                        decorated_ids.add(tok)
            elif op == "OpTypeInt" and res and len(args) >= 2:
                try:
                    w, s = int(args[0]), int(args[1])
                    if w in (8, 16, 32, 64):
                        int_type_info[res] = (w, s)
                except ValueError:
                    pass
            elif op == "OpConstant" and res and len(args) >= 2:
                const_ids.add(res)
                const_type[res] = args[0]
                if args[1] in ("0", "0x0", "0.0"):
                    zero_const_of_type.setdefault(args[0], res)
                info = int_type_info.get(args[0])
                if info is not None:
                    try:
                        w, s = info
                        uval = int(args[1], 0) & ((1 << w) - 1)
                        const_int_val[res] = (args[0], uval, w, s)
                        int_const_by_val.setdefault((args[0], uval), res)
                        if uval == 0:
                            zero_const_of_type.setdefault(args[0], res)
                    except ValueError:
                        pass
            elif op == "OpConstantTrue" and res and args:
                const_ids.add(res)
                const_true.add(res)
                const_type[res] = args[0]
            elif op == "OpConstantFalse" and res and args:
                const_ids.add(res)
                const_false.add(res)
                const_type[res] = args[0]
            elif op == "OpConstantComposite" and res and args:
                const_ids.add(res)
                const_type[res] = args[0]
            elif op == "OpConstantNull" and res and args:
                const_ids.add(res)
                const_type[res] = args[0]
                const_null_type[res] = args[0]
            elif op == "OpTypeArray" and res and args:
                array_elem_type[res] = args[0]
            elif op == "OpVariable" and res and len(args) >= 2 and args[1] in ("PushConstant", "Input"):
                readonly_global_vars.add(res)
        else:
            parsed = _parse_line(raw)
            if parsed is not None and parsed[1] == "OpFunctionEnd":
                func_ranges.append((fn_start, i + 1))
                fn_start = None

    if not func_ranges:
        return spvasm

    const_null_elem_zero: dict[str, str] = {}
    for null_id, arr_ty in const_null_type.items():
        elem_ty = array_elem_type.get(arr_ty)
        if elem_ty and elem_ty in zero_const_of_type:
            const_null_elem_zero[null_id] = zero_const_of_type[elem_ty]

    all_deleted: set[str] = set()
    opt_funcs: dict[int, list[str]] = {}
    for start, end in func_ranges:
        new_body, deleted = _optimize_function(
            lines[start:end], const_ids, const_true, const_false,
            const_null_elem_zero, const_type, const_int_val, int_const_by_val,
            int_type_info, readonly_global_vars, glsl_sets, decorated_ids,
            max_passes=max_passes,
        )
        opt_funcs[start] = new_body
        all_deleted.update(deleted)

    # Reassemble module, dropping OpDecorate / OpName lines that target deleted SSA IDs.
    out: list[str] = []
    func_map = {start: (end, opt_funcs[start]) for start, end in func_ranges}
    i = 0
    n = len(lines)
    while i < n:
        if i in func_map:
            end, new_body = func_map[i]
            out.extend(new_body)
            i = end
            continue
        raw = lines[i]
        if all_deleted:
            stripped = raw.lstrip()
            op_tok = stripped.split(None, 1)[0] if stripped else ""
            if op_tok in ("OpDecorate", "OpName", "OpDecorateId", "OpDecorateString",
                          "OpDecorateStringGOOGLE"):
                toks = stripped.split(None, 2)
                if len(toks) >= 2 and toks[1] in all_deleted:
                    i += 1
                    continue
            elif op_tok == "OpGroupDecorate":
                semi = raw.find(";")
                code = raw[:semi] if semi >= 0 else raw
                comment = raw[semi:] if semi >= 0 else ""
                indent = raw[:len(raw) - len(stripped)]
                toks = code.split()
                if len(toks) >= 3:
                    kept = [t for t in toks[2:] if t not in all_deleted]
                    if not kept:
                        i += 1
                        continue
                    if len(kept) != len(toks) - 2:
                        suffix = (" " + comment) if comment else ""
                        out.append(f"{indent}{toks[0]} {toks[1]} {' '.join(kept)}{suffix}")
                        i += 1
                        continue
        out.append(raw)
        i += 1
    return "\n".join(out) + ("\n" if spvasm.endswith("\n") else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sound dominator-tree SPIR-V optimizer.")
    ap.add_argument("src", help="input .spvasm file")
    ap.add_argument("-o", "--out", required=True, help="output .spvasm file")
    ap.add_argument("--passes", type=int, default=4, help="max optimization passes (default: 4)")
    args = ap.parse_args(argv)
    with open(args.src, encoding="utf-8") as f:
        text = f.read()
    opt = optimize(text, max_passes=args.passes)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(opt)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
