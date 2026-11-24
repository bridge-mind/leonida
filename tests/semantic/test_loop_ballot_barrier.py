"""Semantic audit & regression tests for loop ballot masks and s_barrier in loops.

Pure Python stdlib only (no llvm-mc, spirv-val, GPU, or game data). Translates synthetic hand-written
RDNA2 listings via tools/rdna2spirv.py and executes the emitted SPIR-V across every lane and wave:

  1. Divergent loops with v_cmp / s_and_saveexec / s_or_b64 / s_and_b64 / s_andn2_b64 /
     s_bcnt1_i32_b32/b64 / v_mbcnt / v_cndmask_b32 / v_readfirstlane_b32 in wave32, native wave64,
     and --wave64-emulate (checked against tools/cf_ref.py and direct RDNA2 ISA oracles).
  2. Lane-mask / ballot bug regressions:
       - wave32 32-bit SOP2 bitwise ops on EXEC with non-0/-1 immediates
       - lane_masks.py preserving mask tracking on 64-bit mask + non-0/-1 immediate and s_mov_b64 vcc, imm
       - wave64 SGPR pairs with separately-modified lo/hi halves (including (MASK_LO/HI, 0/-1) pairs and
         loop-carried split pairs) in holds_lane_mask, mask_src, and --wave64-emulate wave_mask x_mcache
       - v_mbcnt_lo/hi_u32_b32 respecting mask_half() in wave64
  3. s_barrier inside group-uniform and wave-divergent loops with LDS writes/reads across:
       - native wave32 CS (bar_scheduler / barrier rounds)
       - native wave64 CS (structured OpLoopMerge with direct OpControlBarrier)
       - --wave64-emulate CS (x_scheduler, 1 and 2 waves, multi-iteration ring wraparound)
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cf_ref  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # raw buffer V# at base 0


class NonUniformBarrierError(RuntimeError):
    """Raised when invocations do not all reach the same OpControlBarrier site uniformly."""


class SubgroupDivergenceError(RuntimeError):
    """Raised when lanes in the same subgroup diverge and hit a subgroup op out of lockstep."""


def _s32(u):
    u &= M32
    return u - 0x100000000 if u >= 0x80000000 else u


# ---------------------------------------------------------------------------
# Pure-Python multi-invocation SPIR-V compute workgroup interpreter
# ---------------------------------------------------------------------------

def _parse_spv(asm):
    """Parse SPIR-V assembly into constants, global/function var layouts, and labeled basic blocks."""
    consts = {}
    types = {}
    ptr_elem = {}
    gvars = {}       # %name -> (storage_class, elem_type)
    fvars = []       # [(%name, elem_type, init_id_or_None)]
    blocks = {}      # %label -> [(res, op, args), ...]
    helpers = {}     # %fn -> ([params], [(res, op, args), ...]): single-block functions %main calls
    cur = None
    in_main = False
    helper = None

    for raw in asm.splitlines():
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        toks = s.split()
        if len(toks) >= 3 and toks[1] == "=":
            res, op, args = toks[0], toks[2], toks[3:]
        else:
            res, op, args = None, toks[0], toks[1:]

        if op == "OpFunction":
            in_main = (res == "%main")
            helper = None if in_main else res
            if helper:
                helpers[helper] = ([], [])
            continue
        if op == "OpFunctionEnd":
            in_main = False
            helper = None
            cur = None
            continue
        if helper is not None:
            # --wave64-emulate outlines the per-round exchange setup (%x_pre_*_fn, #2089) into single-block helpers
            if op == "OpFunctionParameter":
                helpers[helper][0].append(res)
            elif op != "OpLabel":
                helpers[helper][1].append((res, op, args))
            continue

        if not in_main:
            if op == "OpTypeInt":
                types[res] = ("int", int(args[0]), int(args[1]))
            elif op == "OpTypeBool":
                types[res] = ("bool",)
            elif op == "OpTypeFloat":
                types[res] = ("float", int(args[0]))
            elif op == "OpTypeVector":
                types[res] = ("vec", args[0], int(args[1]))
            elif op == "OpTypeArray":
                types[res] = ("arr", args[0], args[1])
            elif op == "OpTypeStruct":
                types[res] = ("struct", list(args))
            elif op == "OpTypePointer":
                ptr_elem[res] = (args[0], args[1])
            elif op == "OpConstant":
                ty = types.get(args[0])
                if ty and ty[0] == "int":
                    val = int(args[1], 0)
                    consts[res] = val & M32 if ty[1] == 32 else val & M64
                elif ty and ty[0] == "float":
                    consts[res] = float(args[1])
                else:
                    consts[res] = int(args[1], 0)
            elif op == "OpConstantTrue":
                consts[res] = True
            elif op == "OpConstantFalse":
                consts[res] = False
            elif op == "OpConstantNull":
                consts[res] = ("null", args[0])
            elif op == "OpVariable":
                sc = args[1]
                _, elem = ptr_elem[args[0]]
                gvars[res] = (sc, elem)
            continue

        if op == "OpLabel":
            cur = res
            blocks[cur] = []
            continue
        if op == "OpVariable":
            _, elem = ptr_elem[args[0]]
            init = args[2] if len(args) > 2 else None
            fvars.append((res, elem, init))
            continue
        if cur is not None:
            blocks[cur].append((res, op, args))

    # Expand each call of a single-block helper inline (fresh result ids per call); OpReturnValue becomes a copy into
    # the call's result, so the lockstep interpreter only ever runs %main.
    seq = 0
    for lbl, insns in blocks.items():
        if not any(op == "OpFunctionCall" for _, op, _ in insns):
            continue
        out = []
        for res, op, args in insns:
            if op != "OpFunctionCall":
                out.append((res, op, args))
                continue
            seq += 1
            params, body = helpers[args[1]]
            assert len(params) == len(args) - 2, f"OpFunctionCall {args[1]}: argument count"
            ids = dict(zip(params, args[2:]))
            for fres, fop, fargs in body:
                fargs = [ids.get(x, x) for x in fargs]
                if fop == "OpReturnValue":
                    out.append((res, "OpCopyObject", [args[0], fargs[0]]))
                    break
                if fop == "OpReturn":
                    break
                if fres:
                    ids[fres] = f"%call{seq}_{fres[1:]}"
                out.append((ids.get(fres) if fres else None, fop, fargs))
        blocks[lbl] = out

    def make_zero(ty_id):
        ty = types.get(ty_id)
        if not ty:
            return 0
        if ty[0] == "bool":
            return False
        if ty[0] == "int":
            return 0
        if ty[0] == "float":
            return 0.0
        if ty[0] == "vec":
            return [make_zero(ty[1]) for _ in range(ty[2])]
        if ty[0] == "arr":
            n = consts[ty[2]]
            return [make_zero(ty[1]) for _ in range(n)]
        if ty[0] == "struct":
            return [make_zero(f) for f in ty[1]]
        return 0

    return {
        "consts": consts,
        "types": types,
        "ptr_elem": ptr_elem,
        "gvars": gvars,
        "fvars": fvars,
        "blocks": blocks,
        "make_zero": make_zero,
    }


SUBGROUP_OPS = {
    "OpGroupNonUniformAny",
    "OpGroupNonUniformAll",
    "OpGroupNonUniformBitwiseOr",
    "OpGroupNonUniformBitwiseAnd",
    "OpGroupNonUniformBitwiseXor",
    "OpGroupNonUniformBroadcastFirst",
    "OpGroupNonUniformShuffle",
    "OpGroupNonUniformQuadSwap",
    "OpGroupNonUniformBallot",
    "OpGroupNonUniformBallotBitExtract",
    "OpGroupNonUniformBallotBitCount",
    "OpGroupNonUniformBallotFindLSB",
}


def run_compute_workgroup(asm, threads, subgroup_size, push_constants=None, memory=None,
                          wg_id=(0, 0, 0), size_xyz=None, reverse_subgroups=False, max_steps=500000):
    """Execute a translated compute SPIR-V module across all invocations of one workgroup.

    Enforces:
      - Lockstep subgroup execution on all OpGroupNonUniform* ops (raises SubgroupDivergenceError on mismatch).
      - Uniform workgroup OpControlBarrier execution (raises NonUniformBarrierError if any invocation exits
        or reaches a different barrier site while others wait).
      - Shared Workgroup variables (%lds, %x_buf) and shared PhysicalStorageBuffer memory.
    """
    mod = _parse_spv(asm)
    consts = mod["consts"]
    types = mod["types"]
    gvars = mod["gvars"]
    blocks = mod["blocks"]
    make_zero = mod["make_zero"]
    mem = {} if memory is None else memory
    pc_words = list(push_constants or [0] * 32)
    if len(pc_words) < 32:
        pc_words += [0] * (32 - len(pc_words))
    sx, sy, sz = size_xyz if size_xyz is not None else (threads, 1, 1)

    # Shared Workgroup storage (initialized to 0xDEADBEEF to catch uninitialized reads!)
    wg_storage = {}
    for name, (sc, elem) in gvars.items():
        if sc == "Workgroup":
            ty = types[elem]
            if ty[0] == "arr":
                wg_storage[name] = [0xDEADBEEF] * consts[ty[2]]
            else:
                wg_storage[name] = 0xDEADBEEF

    # PushConstant struct layout matches rdna2spirv: member 0 = array[32] of uint, members 1..4 = heap/code ptrs
    pc_struct = [pc_words[:32], 0, 0, 0, 0]

    # Per-invocation state
    class Inv:
        __slots__ = ("gid", "sg_id", "sg_lane", "lid", "fn_mem", "ssa", "blk", "ip", "prev_blk",
                     "done", "waiting_barrier", "waiting_sg", "steps")

        def __init__(self, gid):
            self.gid = gid
            self.sg_id = gid // subgroup_size
            self.sg_lane = gid % subgroup_size
            x = gid % sx
            y = (gid // sx) % sy
            z = gid // (sx * sy)
            self.lid = [x, y, z]
            self.fn_mem = {}
            for vname, elem, init in mod["fvars"]:
                if init is None:
                    self.fn_mem[vname] = make_zero(elem)
                elif isinstance(consts[init], tuple) and consts[init][0] == "null":
                    self.fn_mem[vname] = make_zero(consts[init][1])
                else:
                    self.fn_mem[vname] = consts[init]
            self.ssa = dict(consts)
            self.blk = "%entry"
            self.ip = 0
            self.prev_blk = None
            self.done = False
            self.waiting_barrier = None   # (blk, ip) when paused at OpControlBarrier
            self.waiting_sg = None        # (blk, ip, res, op, args) when paused at subgroup op
            self.steps = 0

    invs = [Inv(i) for i in range(threads)]
    subgroups = []
    for s in range(0, threads, subgroup_size):
        subgroups.append(invs[s:s + subgroup_size])

    def val_of(inv, tok):
        if tok in inv.ssa:
            return inv.ssa[tok]
        if tok in consts:
            return consts[tok]
        raise KeyError(f"Unknown SSA id {tok} in {inv.blk}")

    def read_ptr(inv, ptr):
        kind = ptr[0]
        if kind == "fn":
            obj = inv.fn_mem[ptr[1]]
            for idx in ptr[2:]:
                obj = obj[idx]
            return obj
        if kind == "wg":
            obj = wg_storage[ptr[1]]
            for idx in ptr[2:]:
                obj = obj[idx]
            return obj
        if kind == "pc":
            obj = pc_struct
            for idx in ptr[2:]:
                obj = obj[idx]
            return obj
        if kind == "in":
            name = ptr[1]
            if name == "%wg_id":
                return list(wg_id)
            if name == "%local_id":
                return list(inv.lid)
            if name == "%sg_lane":
                return inv.sg_lane
            if name == "%local_index":
                return inv.gid
            raise KeyError(f"Unhandled input {name}")
        if kind == "phys":
            return mem.get(ptr[1], 0)
        raise KeyError(f"Unhandled ptr kind {kind}")

    def write_ptr(inv, ptr, v):
        kind = ptr[0]
        if kind == "fn":
            if len(ptr) == 2:
                inv.fn_mem[ptr[1]] = v
            else:
                obj = inv.fn_mem[ptr[1]]
                for idx in ptr[2:-1]:
                    obj = obj[idx]
                obj[ptr[-1]] = v
            return
        if kind == "wg":
            if len(ptr) == 2:
                wg_storage[ptr[1]] = v
            else:
                obj = wg_storage[ptr[1]]
                for idx in ptr[2:-1]:
                    obj = obj[idx]
                obj[ptr[-1]] = v
            return
        if kind == "phys":
            mem[ptr[1]] = v & M32
            return
        raise KeyError(f"Cannot store to ptr {ptr}")

    def base_ptr(inv, tok):
        if tok in inv.fn_mem:
            return ("fn", tok)
        if tok in gvars:
            sc, _ = gvars[tok]
            if sc == "Workgroup":
                return ("wg", tok)
            if sc == "PushConstant":
                return ("pc", tok)
            if sc == "Input":
                return ("in", tok)
        return val_of(inv, tok)

    def step_inv(inv):
        """Step one invocation until OpControlBarrier, a subgroup op, or OpReturn."""
        while not inv.done and inv.waiting_barrier is None and inv.waiting_sg is None:
            inv.steps += 1
            if inv.steps > max_steps:
                raise TimeoutError(f"Invocation {inv.gid} exceeded {max_steps} steps at {inv.blk}:{inv.ip}")
            insns = blocks[inv.blk]
            res, op, args = insns[inv.ip]

            if op == "OpControlBarrier":
                inv.waiting_barrier = (inv.blk, inv.ip)
                return
            if op in SUBGROUP_OPS:
                inv.waiting_sg = (inv.blk, inv.ip, res, op, args)
                return

            inv.ip += 1
            if op in ("OpSelectionMerge", "OpLoopMerge"):
                continue
            if op == "OpReturn":
                inv.done = True
                return
            if op == "OpBranch":
                inv.prev_blk = inv.blk
                inv.blk = args[0]
                inv.ip = 0
                continue
            if op == "OpBranchConditional":
                cond = val_of(inv, args[0])
                inv.prev_blk = inv.blk
                inv.blk = args[1] if cond else args[2]
                inv.ip = 0
                continue
            if op == "OpSwitch":
                sel = val_of(inv, args[0])
                tgt = args[1]
                for k in range(2, len(args), 2):
                    if int(args[k], 0) == sel:
                        tgt = args[k + 1]
                        break
                inv.prev_blk = inv.blk
                inv.blk = tgt
                inv.ip = 0
                continue
            if op == "OpPhi":
                found = False
                for k in range(1, len(args), 2):
                    if args[k + 1] == inv.prev_blk:
                        inv.ssa[res] = val_of(inv, args[k])
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"OpPhi in {inv.blk} missing predecessor {inv.prev_blk}")
                continue
            if op == "OpAccessChain":
                p = base_ptr(inv, args[1])
                idxs = tuple(val_of(inv, a) for a in args[2:])
                inv.ssa[res] = p + idxs
            elif op == "OpLoad":
                p = base_ptr(inv, args[1])
                inv.ssa[res] = read_ptr(inv, p)
            elif op == "OpStore":
                p = base_ptr(inv, args[0])
                write_ptr(inv, p, val_of(inv, args[1]))
            elif op == "OpConvertUToPtr":
                inv.ssa[res] = ("phys", val_of(inv, args[1]))
            elif op == "OpUConvert":
                ty = types[args[0]]
                v = val_of(inv, args[1])
                inv.ssa[res] = (v & M64) if ty[1] == 64 else (v & M32)
            elif op == "OpCopyObject":
                inv.ssa[res] = val_of(inv, args[1])
            elif op == "OpBitcast":
                inv.ssa[res] = val_of(inv, args[1])
            elif op == "OpCompositeExtract":
                inv.ssa[res] = val_of(inv, args[1])[int(args[2])]
            elif op == "OpCompositeConstruct":
                inv.ssa[res] = [val_of(inv, a) for a in args[1:]]
            elif op == "OpSelect":
                inv.ssa[res] = val_of(inv, args[2]) if val_of(inv, args[1]) else val_of(inv, args[3])
            elif op == "OpIAdd":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) + val_of(inv, args[2])) & mask
            elif op == "OpIAddCarry":
                s = (val_of(inv, args[1]) & M32) + (val_of(inv, args[2]) & M32)
                inv.ssa[res] = [s & M32, 1 if s > M32 else 0]
            elif op == "OpISubBorrow":
                a, b = val_of(inv, args[1]) & M32, val_of(inv, args[2]) & M32
                inv.ssa[res] = [(a - b) & M32, 1 if a < b else 0]
            elif op == "OpISub":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) - val_of(inv, args[2])) & mask
            elif op == "OpIMul":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) * val_of(inv, args[2])) & mask
            elif op == "OpUMod":
                inv.ssa[res] = (val_of(inv, args[1]) % val_of(inv, args[2])) & M32
            elif op == "OpUDiv":
                inv.ssa[res] = (val_of(inv, args[1]) // val_of(inv, args[2])) & M32
            elif op == "OpBitwiseAnd":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) & val_of(inv, args[2])) & mask
            elif op == "OpBitwiseOr":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) | val_of(inv, args[2])) & mask
            elif op == "OpBitwiseXor":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (val_of(inv, args[1]) ^ val_of(inv, args[2])) & mask
            elif op == "OpNot":
                ty = types[args[0]]
                mask = M64 if ty[1] == 64 else M32
                inv.ssa[res] = (~val_of(inv, args[1])) & mask
            elif op == "OpShiftLeftLogical":
                ty = types[args[0]]
                w = ty[1]
                mask = M64 if w == 64 else M32
                sh = val_of(inv, args[2])
                inv.ssa[res] = ((val_of(inv, args[1]) << sh) & mask) if sh < w else 0
            elif op == "OpShiftRightLogical":
                ty = types[args[0]]
                w = ty[1]
                mask = M64 if w == 64 else M32
                sh = val_of(inv, args[2])
                inv.ssa[res] = ((val_of(inv, args[1]) & mask) >> sh) if sh < w else 0
            elif op == "OpShiftRightArithmetic":
                sh = val_of(inv, args[2])
                inv.ssa[res] = (_s32(val_of(inv, args[1])) >> min(sh, 31)) & M32
            elif op == "OpBitCount":
                inv.ssa[res] = bin(val_of(inv, args[1]) & M32).count("1")
            elif op == "OpBitReverse":
                inv.ssa[res] = int(f"{val_of(inv, args[1]) & M32:032b}"[::-1], 2)
            elif op == "OpBitFieldUExtract":
                v, o, c = val_of(inv, args[1]) & M32, val_of(inv, args[2]), val_of(inv, args[3])
                inv.ssa[res] = 0 if c == 0 else ((v >> o) & ((1 << c) - 1))
            elif op == "OpIEqual":
                inv.ssa[res] = (val_of(inv, args[1]) == val_of(inv, args[2]))
            elif op == "OpINotEqual":
                inv.ssa[res] = (val_of(inv, args[1]) != val_of(inv, args[2]))
            elif op == "OpULessThan":
                inv.ssa[res] = (val_of(inv, args[1]) < val_of(inv, args[2]))
            elif op == "OpULessThanEqual":
                inv.ssa[res] = (val_of(inv, args[1]) <= val_of(inv, args[2]))
            elif op == "OpUGreaterThan":
                inv.ssa[res] = (val_of(inv, args[1]) > val_of(inv, args[2]))
            elif op == "OpUGreaterThanEqual":
                inv.ssa[res] = (val_of(inv, args[1]) >= val_of(inv, args[2]))
            elif op == "OpSLessThan":
                inv.ssa[res] = (_s32(val_of(inv, args[1])) < _s32(val_of(inv, args[2])))
            elif op == "OpSLessThanEqual":
                inv.ssa[res] = (_s32(val_of(inv, args[1])) <= _s32(val_of(inv, args[2])))
            elif op == "OpSGreaterThan":
                inv.ssa[res] = (_s32(val_of(inv, args[1])) > _s32(val_of(inv, args[2])))
            elif op == "OpSGreaterThanEqual":
                inv.ssa[res] = (_s32(val_of(inv, args[1])) >= _s32(val_of(inv, args[2])))
            elif op == "OpLogicalAnd":
                inv.ssa[res] = bool(val_of(inv, args[1]) and val_of(inv, args[2]))
            elif op == "OpLogicalOr":
                inv.ssa[res] = bool(val_of(inv, args[1]) or val_of(inv, args[2]))
            elif op == "OpLogicalNot":
                inv.ssa[res] = not bool(val_of(inv, args[1]))
            elif op == "OpLogicalEqual":
                inv.ssa[res] = (bool(val_of(inv, args[1])) == bool(val_of(inv, args[2])))
            elif op == "OpLogicalNotEqual":
                inv.ssa[res] = (bool(val_of(inv, args[1])) != bool(val_of(inv, args[2])))
            elif op == "OpExtInst":
                fn = args[2]
                xs = [val_of(inv, a) for a in args[3:]]
                if fn == "UMin":
                    inv.ssa[res] = min(xs[0] & M32, xs[1] & M32)
                elif fn == "UMax":
                    inv.ssa[res] = max(xs[0] & M32, xs[1] & M32)
                elif fn == "SMin":
                    inv.ssa[res] = min(_s32(xs[0]), _s32(xs[1])) & M32
                elif fn == "SMax":
                    inv.ssa[res] = max(_s32(xs[0]), _s32(xs[1])) & M32
                elif fn == "FindILsb":
                    v = xs[0] & M32
                    inv.ssa[res] = M32 if v == 0 else (v & -v).bit_length() - 1
                elif fn == "FindUMsb":
                    v = xs[0] & M32
                    inv.ssa[res] = M32 if v == 0 else v.bit_length() - 1
                else:
                    raise NotImplementedError(f"OpExtInst {fn}")
            elif op == "OpAtomicOr":
                p = base_ptr(inv, args[1])
                old = read_ptr(inv, p) & M32
                write_ptr(inv, p, (old | (val_of(inv, args[4]) & M32)) & M32)
                inv.ssa[res] = old
            elif op == "OpAtomicIAdd":
                p = base_ptr(inv, args[1])
                old = read_ptr(inv, p) & M32
                write_ptr(inv, p, (old + (val_of(inv, args[4]) & M32)) & M32)
                inv.ssa[res] = old
            elif op == "OpAtomicAnd":
                p = base_ptr(inv, args[1])
                old = read_ptr(inv, p) & M32
                write_ptr(inv, p, (old & (val_of(inv, args[4]) & M32)) & M32)
                inv.ssa[res] = old
            elif op == "OpAtomicXor":
                p = base_ptr(inv, args[1])
                old = read_ptr(inv, p) & M32
                write_ptr(inv, p, (old ^ (val_of(inv, args[4]) & M32)) & M32)
                inv.ssa[res] = old
            elif op == "OpAtomicExchange":
                p = base_ptr(inv, args[1])
                old = read_ptr(inv, p) & M32
                write_ptr(inv, p, val_of(inv, args[4]) & M32)
                inv.ssa[res] = old
            else:
                raise NotImplementedError(f"Unsupported SPIR-V op {op} in {inv.blk}")

    def resolve_subgroup(sg):
        sites = {inv.waiting_sg[:2] for inv in sg}
        if len(sites) != 1:
            raise SubgroupDivergenceError(f"Subgroup diverged across sites {sites}")
        _, _, res, op, args = sg[0].waiting_sg
        if op == "OpGroupNonUniformAny":
            ans = any(bool(val_of(inv, args[2])) for inv in sg)
            for inv in sg:
                inv.ssa[res] = ans
        elif op == "OpGroupNonUniformAll":
            ans = all(bool(val_of(inv, args[2])) for inv in sg)
            for inv in sg:
                inv.ssa[res] = ans
        elif op == "OpGroupNonUniformBitwiseOr":
            acc = 0
            for inv in sg:
                acc |= val_of(inv, args[3]) & M32
            for inv in sg:
                inv.ssa[res] = acc
        elif op == "OpGroupNonUniformBitwiseAnd":
            acc = M32
            for inv in sg:
                acc &= val_of(inv, args[3]) & M32
            for inv in sg:
                inv.ssa[res] = acc
        elif op == "OpGroupNonUniformBitwiseXor":
            acc = 0
            for inv in sg:
                acc ^= val_of(inv, args[3]) & M32
            for inv in sg:
                inv.ssa[res] = acc
        elif op == "OpGroupNonUniformBroadcastFirst":
            v0 = val_of(sg[0], args[2])
            for inv in sg:
                inv.ssa[res] = v0
        elif op == "OpGroupNonUniformShuffle":
            vals = [val_of(inv, args[2]) for inv in sg]
            for inv in sg:
                src = val_of(inv, args[3])
                if not (0 <= src < len(sg)):
                    raise RuntimeError(f"Out-of-range subgroup shuffle index {src} (size {len(sg)})")
                inv.ssa[res] = vals[src]
        elif op == "OpGroupNonUniformQuadSwap":
            d = val_of(sg[0], args[3])
            xor = 1 if d == 0 else (2 if d == 1 else 3)
            vals = [val_of(inv, args[2]) for inv in sg]
            for i, inv in enumerate(sg):
                inv.ssa[res] = vals[i ^ xor]
        elif op == "OpGroupNonUniformBallot":
            words = [0, 0, 0, 0]
            for inv in sg:
                if val_of(inv, args[2]):
                    words[inv.sg_lane >> 5] |= 1 << (inv.sg_lane & 31)
            for inv in sg:
                inv.ssa[res] = list(words)
        elif op == "OpGroupNonUniformBallotBitExtract":
            for inv in sg:
                b = val_of(inv, args[2])
                idx = val_of(inv, args[3])
                inv.ssa[res] = bool((b[idx >> 5] >> (idx & 31)) & 1) if idx < 128 else False
        elif op == "OpGroupNonUniformBallotBitCount":
            op_kind = args[2]
            for inv in sg:
                b = val_of(inv, args[3])
                bits = b[0] | (b[1] << 32)
                if op_kind == "Reduce":
                    inv.ssa[res] = bin(bits).count("1")
                elif op_kind == "ExclusiveScan":
                    inv.ssa[res] = bin(bits & ((1 << inv.sg_lane) - 1)).count("1")
                elif op_kind == "InclusiveScan":
                    inv.ssa[res] = bin(bits & ((1 << (inv.sg_lane + 1)) - 1)).count("1")
                else:
                    raise NotImplementedError(op_kind)
        elif op == "OpGroupNonUniformBallotFindLSB":
            for inv in sg:
                b = val_of(inv, args[2])
                bits = b[0] | (b[1] << 32)
                inv.ssa[res] = M32 if bits == 0 else (bits & -bits).bit_length() - 1
        else:
            raise NotImplementedError(op)

        for inv in sg:
            inv.ip += 1
            inv.waiting_sg = None

    order = list(reversed(subgroups)) if reverse_subgroups else list(subgroups)
    while not all(inv.done for inv in invs):
        # Advance each subgroup until every lane in it is done or paused at a barrier
        for sg in order:
            while True:
                if all(inv.done or inv.waiting_barrier is not None for inv in sg):
                    break
                for inv in sg:
                    step_inv(inv)
                # Check if some lanes hit a subgroup op while others finished or hit a barrier
                sg_waiting = [inv for inv in sg if inv.waiting_sg is not None]
                if sg_waiting:
                    if len(sg_waiting) != len(sg):
                        states = [(inv.gid, inv.blk, inv.done, inv.waiting_barrier,
                                   inv.waiting_sg[:2] if inv.waiting_sg else None) for inv in sg]
                        raise SubgroupDivergenceError(
                            f"Only {len(sg_waiting)}/{len(sg)} lanes reached subgroup op: {states[:4]}"
                        )
                    resolve_subgroup(sg)

        if all(inv.done for inv in invs):
            break
        # Barrier sync: EVERY invocation must be waiting at the SAME OpControlBarrier
        bar_waiting = [inv for inv in invs if inv.waiting_barrier is not None]
        if len(bar_waiting) != len(invs):
            raise NonUniformBarrierError(
                f"Non-uniform OpControlBarrier: {len(bar_waiting)}/{len(invs)} invocations arrived "
                f"({sum(inv.done for inv in invs)} already returned)"
            )
        sites = {inv.waiting_barrier for inv in invs}
        if len(sites) != 1:
            raise NonUniformBarrierError(f"Invocations reached different OpControlBarrier sites: {sites}")
        for inv in invs:
            inv.ip += 1
            inv.waiting_barrier = None

    return mem


# ---------------------------------------------------------------------------
# Synthetic listing helpers (pure hand-written RDNA2, zero game data)
# ---------------------------------------------------------------------------

def _ins_size(text):
    op = text.split()[0]
    if op.startswith(("exp", "buffer_", "global_", "ds_", "image_")) or op.endswith("_e64"):
        return 8
    if op in ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_readlane_b32", "v_writelane_b32",
              "v_lshl_add_u32", "v_cndmask_b32_e64"):
        return 8
    if "0x" in text:
        return 8
    if op.startswith("v_cmp") and not op.endswith("_e32") and "vcc" not in text:
        return 8
    return 4


def _resolve_labels(items):
    """Turn a list of instructions and 'label:' markers into (text, size) with resolved branch offsets."""
    parsed = []
    labels = {}
    pc = 0
    for it in items:
        s = it.strip()
        if not s or s.startswith(";"):
            continue
        if s.endswith(":"):
            labels[s[:-1]] = pc
            continue
        sz = _ins_size(s)
        parsed.append((s, sz, pc))
        pc += sz

    out = []
    for s, sz, ipc in parsed:
        toks = s.split()
        if toks[0].startswith(("s_branch", "s_cbranch_")) and len(toks) == 2 and toks[1] in labels:
            delta = (labels[toks[1]] - (ipc + sz)) // 4
            s = f"{toks[0]} {delta}"
        out.append((s, sz))
    return out


def _translate_cs(name, items, threads=64, lds_bytes=0, emulate=False, size_xyz=None, structured=True):
    lines = _resolve_labels(items)
    sx, sy, sz = size_xyz if size_xyz is not None else (threads, 1, 1)
    lds_blocks = (lds_bytes + 511) // 512
    rsrc2 = 0x8090 | ((lds_blocks & 0x1FF) << 15)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", rsrc2, sx, sy, sz))
        old_em, old_st = rdna2spirv.WAVE64_EMULATE, rdna2spirv.STRUCTURED
        rdna2spirv.WAVE64_EMULATE = emulate
        rdna2spirv.STRUCTURED = structured      # False: --dispatch-loop, which keeps the emulation's round scheduler
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gen = rdna2spirv.translate.last_gen
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE, rdna2spirv.STRUCTURED = old_em, old_st
    return spv, gen


def _translate_and_ref_ps(name, items, wave, use_cf_ref=True):
    """Translate a synthetic PS and run spirv_wave_interp.run_wave (and optionally cf_ref.WaveMachine)."""
    lines = _resolve_labels(items)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        ref = cf_ref.run(src, wave=wave)["exports"] if use_cf_ref else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
    got = wi.run_wave(spv, wave=wave)
    actual = {}
    for lane, out in enumerate(got["outputs"]):
        if out:
            for k, v in out.items():
                if k.endswith(("mrt0", "out0")):
                    actual[lane] = tuple(v)
                    break
    return ref, actual, spv


def _build_mask_from_s4_s5_w64():
    """RDNA2 snippet: sets vcc[lane] = bit (lane&31) of (s4 if lane<32 else s5), and marks wave64."""
    return [
        "v_mov_b32_e32 v11, s5",
        "v_cmp_lt_u32_e64 vcc, v0, 32",
        "v_cndmask_b32_e64 v1, v11, s4, vcc",
        "v_and_b32_e32 v2, 31, v0",
        "v_lshrrev_b32_e32 v3, v2, v1",
        "v_and_b32_e32 v3, 1, v3",
        "v_cmp_ne_u32_e64 vcc, 0, v3",
        "s_and_b64 s[12:13], vcc, exec",
    ]


# ---------------------------------------------------------------------------
# 1. Divergent loops with ballots, popcount, mbcnt, cndmask, saveexec
# ---------------------------------------------------------------------------

class DivergentLoopBallotTests(unittest.TestCase):
    """Audit Hazard 1: verify inactive/exited lanes never leave stale bits across loop iterations."""

    def _divergent_loop_ps_program(self, wave):
        """Each lane runs (lane & 3) + 1 iterations; tracks per-iteration active count, rank,
        and a loop-accumulated mask of lanes that ever reached iter == 3.
        """
        if wave == 32:
            return [
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                "v_and_b32_e32 v1, 3, v0",              # limit = (lane & 3) + 1  (1..4 iters)
                "v_add_nc_u32_e32 v1, 1, v1",
                "v_mov_b32_e32 v2, 0",                  # iter counter
                "v_mov_b32_e32 v3, 0",                  # sum of active wave counts seen by this lane
                "v_mov_b32_e32 v4, 0",                  # sum of lane ranks seen by this lane
                "s_mov_b32 s10, exec_lo",               # save outer exec
                "s_mov_b32 s14, 0",                     # accumulator mask across iterations (EITHER -> MASK)
                "loop:",
                "v_add_nc_u32_e32 v2, 1, v2",
                "s_bcnt1_i32_b32 s6, exec_lo",
                "v_add_nc_u32_e32 v3, s6, v3",
                "v_mbcnt_lo_u32_b32 v7, exec_lo, 0",
                "v_add_nc_u32_e32 v4, v7, v4",
                # Compare inside divergent loop: inactive lanes MUST write 0, not keep stale vcc!
                "v_cmp_eq_u32_e32 vcc_lo, 3, v2",
                "s_or_b32 s14, s14, vcc_lo",
                # Exit lanes where v2 >= v1
                "v_cmp_lt_u32_e32 vcc_lo, v2, v1",
                "s_and_b32 exec_lo, exec_lo, vcc_lo",
                "s_cbranch_execnz loop",
                "s_mov_b32 exec_lo, s10",
                # Verify s14 has exact lanes that reached iter 3 (i.e. limit >= 3, (lane & 3) >= 2)
                "s_bcnt1_i32_b32 s15, s14",
                "v_cndmask_b32_e64 v5, 0, s15, s14",
                "exp mrt0 v2, v3, v4, v5 done vm",
                "s_endpgm",
            ]
        return [
            "s_mov_b64 s[10:11], exec",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
            "v_and_b32_e32 v1, 3, v0",                  # limit = (lane & 3) + 1 (1..4 iters)
            "v_add_nc_u32_e32 v1, 1, v1",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "v_mov_b32_e32 v4, 0",
            "s_mov_b64 s[14:15], 0",                    # accumulator mask (EITHER -> MASK via back-edge)
            "loop:",
            "v_add_nc_u32_e32 v2, 1, v2",
            "s_bcnt1_i32_b64 s6, exec",
            "v_add_nc_u32_e32 v3, s6, v3",
            "v_mbcnt_lo_u32_b32 v7, exec_lo, 0",
            "v_mbcnt_hi_u32_b32 v7, exec_hi, v7",
            "v_add_nc_u32_e32 v4, v7, v4",
            # Compare inside divergent loop: inactive lanes MUST write 0 into vcc!
            "v_cmp_eq_u32_e32 vcc, 3, v2",
            "s_or_b64 s[14:15], s[14:15], vcc",
            # Exit lanes where v2 >= v1 using s_and_saveexec_b64
            "v_cmp_lt_u32_e32 vcc, v2, v1",
            "s_and_saveexec_b64 s[16:17], vcc",
            "s_cbranch_execnz loop",
            "s_mov_b64 exec, s[10:11]",
            "s_bcnt1_i32_b64 s18, s[14:15]",
            "v_cndmask_b32_e64 v5, 0, s18, s[14:15]",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ]

    def _cf_ref_divergent_loop(self, wave):
        """Divergent loop using only ops modelled by tools/cf_ref.py (v_cmp, s_and_saveexec, s_or, s_and, v_cndmask)."""
        if wave == 32:
            return [
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                "v_and_b32_e32 v1, 3, v0",
                "v_add_nc_u32_e32 v1, 1, v1",
                "v_mov_b32_e32 v2, 0",
                "v_mov_b32_e32 v3, 0",
                "s_mov_b32 s10, exec_lo",
                "s_mov_b32 s14, 0",
                "loop:",
                "v_add_nc_u32_e32 v2, 1, v2",
                "v_cmp_eq_u32_e32 vcc_lo, 3, v2",
                "s_or_b32 s14, s14, vcc_lo",
                "v_cndmask_b32_e64 v4, 0, 10, vcc_lo",
                "v_add_nc_u32_e32 v3, v4, v3",
                "v_cmp_lt_u32_e32 vcc_lo, v2, v1",
                "s_and_saveexec_b32 s12, vcc_lo",
                "s_cbranch_execnz loop",
                "s_mov_b32 exec_lo, s10",
                "v_cndmask_b32_e64 v5, 0, 77, s14",
                "exp mrt0 v2, v3, v5, v1 done vm",
                "s_endpgm",
            ]
        return [
            "s_mov_b64 s[10:11], exec",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
            "v_and_b32_e32 v1, 3, v0",
            "v_add_nc_u32_e32 v1, 1, v1",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "s_mov_b64 s[14:15], 0",
            "loop:",
            "v_add_nc_u32_e32 v2, 1, v2",
            "v_cmp_eq_u32_e32 vcc, 3, v2",
            "s_or_b64 s[14:15], s[14:15], vcc",
            "v_cndmask_b32_e64 v4, 0, 10, vcc",
            "v_add_nc_u32_e32 v3, v4, v3",
            "v_cmp_lt_u32_e32 vcc, v2, v1",
            "s_and_saveexec_b64 s[12:13], vcc",
            "s_cbranch_execnz loop",
            "s_mov_b64 exec, s[10:11]",
            "v_cndmask_b32_e64 v5, 0, 77, s[14:15]",
            "exp mrt0 v2, v3, v5, v1 done vm",
            "s_endpgm",
        ]

    def test_divergent_loop_cf_ref_exact_match(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                ref, got, _ = _translate_and_ref_ps(f"ps_cf_ref_loop_w{wave}", self._cf_ref_divergent_loop(wave),
                                                    wave=wave, use_cf_ref=True)
                self.assertEqual(len(got), wave)
                self.assertEqual(got, ref)

    def test_divergent_loop_wave32_bcnt_mbcnt_oracle(self):
        _, got, _ = _translate_and_ref_ps("ps_div_loop_w32", self._divergent_loop_ps_program(32),
                                          wave=32, use_cf_ref=False)
        self.assertEqual(len(got), 32)
        for lane in range(32):
            lim = (lane & 3) + 1
            counts = [32, 24, 16, 8]
            want_v3 = sum(counts[:lim])
            want_v4 = sum(sum(1 for j in range(lane) if (j & 3) >= it) for it in range(lim))
            self.assertEqual(got[lane], (lim, want_v3, want_v4, 16 if lim >= 3 else 0), f"lane {lane}")

    def test_divergent_loop_wave64_bcnt_mbcnt_oracle(self):
        _, got, _ = _translate_and_ref_ps("ps_div_loop_w64", self._divergent_loop_ps_program(64),
                                          wave=64, use_cf_ref=False)
        self.assertEqual(len(got), 64)
        for lane in range(64):
            lim = (lane & 3) + 1
            counts = [64, 48, 32, 16]
            want_v3 = sum(counts[:lim])
            want_v4 = sum(sum(1 for j in range(lane) if (j & 3) >= it) for it in range(lim))
            self.assertEqual(got[lane], (lim, want_v3, want_v4, 32 if lim >= 3 else 0), f"lane {lane}")

    def test_divergent_loop_wave64_emulate_and_native_cs(self):
        """Same divergent loop with ballot/bcnt1/mbcnt/readfirstlane/cndmask in compute shaders:
        native wave64 (64 & 128 threads) AND --wave64-emulate (64, 128, and partial 40 threads on 32-wide subgroups).
        """
        prog = [
            "s_mov_b64 s[10:11], exec",
            "v_and_b32_e32 v1, 3, v0",                  # limit = (lane & 3) + 1 (1..4)
            "v_add_nc_u32_e32 v1, 1, v1",
            "v_mov_b32_e32 v2, 0",                      # iter
            "v_mov_b32_e32 v3, 0",                      # sum of bcnt1(exec)
            "v_mov_b32_e32 v4, 0",                      # sum of mbcnt(exec)
            "v_mov_b32_e32 v8, 0",                      # sum of readfirstlane(v0)
            "s_mov_b64 s[14:15], 0",                    # accumulated mask of lanes hitting iter == 3
            "loop:",
            "v_add_nc_u32_e32 v2, 1, v2",
            "s_bcnt1_i32_b64 s6, exec",
            "v_add_nc_u32_e32 v3, s6, v3",
            "v_mbcnt_lo_u32_b32 v7, exec_lo, 0",
            "v_mbcnt_hi_u32_b32 v7, exec_hi, v7",
            "v_add_nc_u32_e32 v4, v7, v4",
            "v_readfirstlane_b32 s7, v0",
            "v_add_nc_u32_e32 v8, s7, v8",
            # Stale-bit probe: vcc is set only for active lanes with v2 == 3; exited lanes MUST NOT contribute
            "v_cmp_eq_u32_e32 vcc, 3, v2",
            "s_or_b64 s[14:15], s[14:15], vcc",
            # Second bcnt1 in same block AFTER vcc changed (tests x_mcache invalidation!)
            "s_bcnt1_i32_b64 s9, vcc",
            "v_add_nc_u32_e32 v3, s9, v3",
            "v_cmp_lt_u32_e32 vcc, v2, v1",
            "s_and_saveexec_b64 s[16:17], vcc",
            "s_cbranch_execnz loop",
            "s_mov_b64 exec, s[10:11]",
            "s_bcnt1_i32_b64 s18, s[14:15]",
            "v_cndmask_b32_e64 v5, 0, s18, s[14:15]",
            # Store 4 words per thread: [v2, v3, v4, v5 ^ (v8 << 16)]
            "v_lshlrev_b32_e32 v9, 16, v8",
            "v_xor_b32_e32 v5, v5, v9",
            "v_lshlrev_b32_e32 v6, 4, v0",
            "buffer_store_dword v2, v6, s[0:3], 0 offen offset:0",
            "buffer_store_dword v3, v6, s[0:3], 0 offen offset:4",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:8",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:12",
            "s_endpgm",
        ]

        def expected_for(threads):
            out = {}
            for wbase in range(0, threads, 64):
                wlen = min(64, threads - wbase)
                active = list(range(wbase, wbase + wlen))
                iters = {t: 0 for t in active}
                v3 = {t: 0 for t in active}
                v4 = {t: 0 for t in active}
                v8 = {t: 0 for t in active}
                acc_mask = set()
                cur = list(active)
                while cur:
                    for t in cur:
                        iters[t] += 1
                    cnt = len(cur)
                    first = cur[0]
                    vcc_eq3 = [t for t in cur if iters[t] == 3]
                    acc_mask.update(vcc_eq3)
                    for rank, t in enumerate(cur):
                        v3[t] += cnt + len(vcc_eq3)
                        v4[t] += rank
                        v8[t] += first
                    cur = [t for t in cur if iters[t] < ((t & 3) + 1)]
                for t in active:
                    v5 = (len(acc_mask) if t in acc_mask else 0) ^ ((v8[t] << 16) & M32)
                    out[t] = (iters[t], v3[t], v4[t], v5)
            return out

        configs = [
            ("native_64", 64, 64, False, True),
            ("native_128", 128, 64, False, True),
            # a reducible multi-block loop takes the structured emulation since #2089 (no round OpSwitch) ...
            ("emulate_64_sg32", 64, 32, True, True),
            ("emulate_128_sg32", 128, 32, True, True),
            ("emulate_40_partial_sg32", 40, 32, True, True),
            # ... and --dispatch-loop keeps the round scheduler: both must give the native words
            ("emulate_64_sg32_rounds", 64, 32, True, False),
            ("emulate_128_sg32_rounds", 128, 32, True, False),
            ("emulate_40_partial_sg32_rounds", 40, 32, True, False),
        ]
        for label, threads, sg_sz, emulate, structured in configs:
            with self.subTest(mode=label):
                spv, gen = _translate_cs("cs_div_loop_" + label, prog, threads=threads, emulate=emulate,
                                         structured=structured)
                if emulate:
                    self.assertEqual(gen.control_flow,
                                     "emulated wave64 structured" if structured else "emulated wave64 rounds")
                mem = run_compute_workgroup(spv, threads, sg_sz, push_constants=VSHARP)
                want = expected_for(threads)
                for t in range(threads):
                    got = tuple(mem.get(16 * t + 4 * k, -1) for k in range(4))
                    self.assertEqual(got, want[t], f"{label} thread {t}")

    def test_nested_divergent_loops_emulate(self):
        """Nested divergent loops (outer s_and_saveexec, inner s_mov_b64 save/restore)
        under --wave64-emulate: ensures x_buf ring buffer and outer/inner exec masks never leak bits.
        """
        prog = [
            "s_mov_b64 s[10:11], exec",
            "v_and_b32_e32 v1, 3, v0",              # outer trips: (lane & 3) + 1
            "v_add_nc_u32_e32 v1, 1, v1",
            "v_mov_b32_e32 v2, 0",                  # outer i
            "v_mov_b32_e32 v5, 0",                  # total inner executions weighted by active inner count
            "outer:",
            "v_add_nc_u32_e32 v2, 1, v2",
            # Inner loop: lanes with (v0 & 4) != 0 run 2 inner iters, others run 1
            "v_and_b32_e32 v3, 4, v0",
            "v_cmp_ne_u32_e32 vcc, 0, v3",
            "v_cndmask_b32_e64 v3, 1, 2, vcc",      # inner limit = 2 if (lane & 4) else 1
            "v_mov_b32_e32 v4, 0",                  # inner j
            "s_mov_b64 s[12:13], exec",             # save outer active mask
            "inner:",
            "v_add_nc_u32_e32 v4, 1, v4",
            "s_bcnt1_i32_b64 s6, exec",
            "v_add_nc_u32_e32 v5, s6, v5",
            "v_cmp_lt_u32_e32 vcc, v4, v3",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz inner",
            "s_mov_b64 exec, s[12:13]",             # restore outer active mask
            "v_cmp_lt_u32_e32 vcc, v2, v1",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz outer",
            "s_mov_b64 exec, s[10:11]",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v5, v6, s[0:3], 0 offen",
            "s_endpgm",
        ]
        want = [0] * 64
        outer_active = list(range(64))
        outer_i = [0] * 64
        while outer_active:
            for t in outer_active:
                outer_i[t] += 1
            inner_active = list(outer_active)
            inner_j = {t: 0 for t in inner_active}
            while inner_active:
                for t in inner_active:
                    inner_j[t] += 1
                    want[t] += len(inner_active)
                inner_active = [t for t in inner_active if inner_j[t] < (2 if (t & 4) else 1)]
            outer_active = [t for t in outer_active if outer_i[t] < ((t & 3) + 1)]

        for emulate, sg_sz in ((False, 64), (True, 32)):
            with self.subTest(emulate=emulate):
                spv, _ = _translate_cs(f"cs_nested_loop_{int(emulate)}", prog, threads=64, emulate=emulate)
                mem = run_compute_workgroup(spv, 64, sg_sz, push_constants=VSHARP)
                self.assertEqual([mem.get(4 * t, -1) for t in range(64)], want)


# ---------------------------------------------------------------------------
# 2. Lane-mask / ballot bug reproducers & regressions
# ---------------------------------------------------------------------------

class LaneMaskBugTests(unittest.TestCase):
    """Targeted regressions for lane-mask tracking, split wave64 SGPR pairs, x_mcache, and v_mbcnt halves."""

    def test_bug2a_wave32_bitwise_mask_with_non_trivial_imm(self):
        """Wave32 32-bit bitwise ops writing EXEC with a non-0/-1 immediate (`s_and_b32 exec_lo, exec_lo, 0xFF`,
        `s_andn2_b32 exec_lo, exec_lo, 5`) extract bit L per lane via mask_src(wide=False).
        """
        prog = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
            "s_mov_b32 s10, exec_lo",
            # Restrict active lanes to 0..7
            "s_and_b32 exec_lo, exec_lo, 0xFF",
            "v_mov_b32_e32 v2, 1",
            # Now clear lane 0 and lane 2 (mask 0x5) -> only lanes 1,3,4,5,6,7 stay active
            "s_andn2_b32 exec_lo, exec_lo, 5",
            "v_mov_b32_e32 v3, 1",
            "s_bcnt1_i32_b32 s6, exec_lo",
            "s_mov_b32 exec_lo, s10",
            "v_mov_b32_e32 v4, s6",
            "v_mov_b32_e32 v5, 0",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ]
        _, got, _ = _translate_and_ref_ps("ps_bug2a_w32_mask_imm", prog, wave=32, use_cf_ref=False)
        for lane in range(32):
            want_v2 = 1 if lane < 8 else 0
            want_v3 = 1 if (lane < 8 and lane not in (0, 2)) else 0
            self.assertEqual(got[lane], (want_v2, want_v3, 6, 0), f"lane {lane}")

    def test_bug3a_lane_masks_preserves_b64_mask_on_non_trivial_imm(self):
        """Combining a 64-bit lane mask with a non-0/-1 immediate (`s_and_b64 s[14:15], vcc, 0xFF`) or moving
        a constant into VCC (`s_mov_b64 vcc, 0x3F`) stores a per-lane mask in rdna2spirv.py, so lane_masks.py
        must keep tracking the destination as a lane mask for subsequent `s_bcnt1_i32_b64` / `v_cndmask_b32`.
        """
        prog = [
            "s_mov_b64 s[10:11], exec",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
            "v_and_b32_e32 v1, 1, v0",
            "v_cmp_eq_u32_e32 vcc, 0, v1",             # even lanes: 0x55555555_55555555
            "s_and_b64 s[14:15], vcc, 0xFF",           # lanes 0,2,4,6 -> popcount MUST be 4 in every lane!
            "s_bcnt1_i32_b64 s6, s[14:15]",
            "s_mov_b64 vcc, 0x3F",                     # lanes 0..5 -> popcount MUST be 6 in every lane!
            "s_bcnt1_i32_b64 s7, vcc",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, s7",
            "v_cndmask_b32_e64 v4, 0, 1, s[14:15]",
            "v_cndmask_b32_e64 v5, 0, 1, vcc",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ]
        _, got, _ = _translate_and_ref_ps("ps_bug3a_mask_imm", prog, wave=64, use_cf_ref=False)
        for lane in range(64):
            self.assertEqual(got[lane], (4, 6, 1 if (lane < 8 and lane % 2 == 0) else 0, 1 if lane < 6 else 0),
                             f"lane {lane}")

    def test_bug4_wave64_split_sgpr_pair_halves(self):
        """If a wave64 SGPR pair s[12:13] has its hi half s13 modified by a 32-bit op (`s_mov_b32 s13, 0`),
        subsequent 64-bit mask reads of s[12:13] (`s_bcnt1_i32_b64`, `v_cndmask_b32`) must read s12 for
        lanes 0-31 and s13 for lanes 32-63, and --wave64-emulate's x_mcache must invalidate when s13 changes!
        """
        prog = _build_mask_from_s4_s5_w64() + [
            "s_bcnt1_i32_b64 s6, s[12:13]",             # populates x_mcache for s[12:13]
            "s_mov_b32 s13, 0",                         # clear ONLY hi half of s[12:13] (MASK_HI -> EITHER)
            "s_bcnt1_i32_b64 s7, s[12:13]",             # must NOT return stale cached count OR ignore s13!
            "v_cndmask_b32_e64 v4, 0, 1, s[12:13]",
            "v_mov_b32_e32 v5, s6",
            "v_mov_b32_e32 v6, s7",
            "v_lshlrev_b32_e32 v7, 4, v0",
            "buffer_store_dword v4, v7, s[0:3], 0 offen offset:0",
            "buffer_store_dword v5, v7, s[0:3], 0 offen offset:4",
            "buffer_store_dword v6, v7, s[0:3], 0 offen offset:8",
            "s_endpgm",
        ]
        for emulate, sg_sz in ((False, 64), (True, 32)):
            with self.subTest(emulate=emulate):
                spv, _ = _translate_cs(f"cs_bug4_split_pair_{int(emulate)}", prog, threads=64, emulate=emulate)
                lo, hi = 0x00000007, 0x000000FF  # 3 bits in lo, 8 bits in hi -> total 11, after clear hi -> 3
                mem = run_compute_workgroup(spv, 64, sg_sz, push_constants=VSHARP + [lo, hi, 0, 0])
                for lane in range(64):
                    want_bit = ((lo >> lane) & 1) if lane < 32 else 0
                    self.assertEqual((mem[16 * lane], mem[16 * lane + 4], mem[16 * lane + 8]),
                                     (want_bit, 11, 3), f"emulate={emulate} lane={lane}")

    def test_bug4b_loop_carried_split_sgpr_pair_ballot(self):
        """Loop-carried split SGPR pairs in wave64 (native and --wave64-emulate):
          - s[12:13] accumulates iter==2 in s12 (lanes 0-31) and iter==3 in s13 (lanes 32-63) via 32-bit s_or_b32,
            with s_bcnt1_i32_b64 before and after the s13 update in the same loop block (testing x_mcache invalidation
            on s13 across loop iterations and distinct half masks in mask_src).
          - s[20:21] updates ONLY s20 in the loop while s21 remains 0 (EITHER) on every path, testing holds_lane_mask
            for (MASK_LO, EITHER) pairs after loop exit.
        """
        prog = [
            "s_mov_b64 s[10:11], exec",
            "v_and_b32_e32 v1, 3, v0",                  # limit = (lane & 3) + 1 (1..4)
            "v_add_nc_u32_e32 v1, 1, v1",
            "v_mov_b32_e32 v2, 0",                      # iter
            "v_mov_b32_e32 v3, 0",                      # sum of (before_s13_cnt + after_s13_cnt)
            "s_mov_b32 s12, 0",                         # lo half accumulator (lanes 0-31 hitting iter==2)
            "s_mov_b32 s13, 0",                         # hi half accumulator (lanes 32-63 hitting iter==3)
            "s_mov_b32 s20, 0",                         # lo-only accumulator (lanes 0-31 hitting iter==3)
            "s_mov_b32 s21, 0",                         # stays constant 0 (EITHER) across the entire loop!
            "loop:",
            "v_add_nc_u32_e32 v2, 1, v2",
            "v_cmp_eq_u32_e32 vcc, 2, v2",
            "s_or_b32 s12, s12, vcc_lo",
            "s_bcnt1_i32_b64 s6, s[12:13]",             # caches s[12:13] in x_mcache before s13 changes
            "v_cmp_eq_u32_e32 vcc, 3, v2",
            "s_or_b32 s13, s13, vcc_hi",                # modifies ONLY s13!
            "s_or_b32 s20, s20, vcc_lo",                # s[20:21] is (MASK_LO, EITHER)
            "s_bcnt1_i32_b64 s7, s[12:13]",             # must see updated s13 in the same block!
            "s_add_u32 s6, s6, s7",
            "v_add_nc_u32_e32 v3, s6, v3",
            "v_cmp_lt_u32_e32 vcc, v2, v1",
            "s_and_saveexec_b64 s[16:17], vcc",
            "s_cbranch_execnz loop",
            "s_mov_b64 exec, s[10:11]",
            "s_bcnt1_i32_b64 s18, s[12:13]",
            "s_bcnt1_i32_b64 s19, s[20:21]",
            "v_cndmask_b32_e64 v4, 0, 1, s[12:13]",
            "v_cndmask_b32_e64 v5, 0, 1, s[20:21]",
            "v_lshlrev_b32_e32 v6, 16, v5",
            "v_or_b32_e32 v4, v6, v4",
            "v_mov_b32_e32 v7, s19",
            "v_lshlrev_b32_e32 v7, 16, v7",
            "v_or_b32_e32 v8, s18, v7",
            "v_lshlrev_b32_e32 v9, 4, v0",
            "buffer_store_dword v3, v9, s[0:3], 0 offen offset:0",
            "buffer_store_dword v4, v9, s[0:3], 0 offen offset:4",
            "buffer_store_dword v8, v9, s[0:3], 0 offen offset:8",
            "s_endpgm",
        ]
        # Per iteration (1..4):
        # iter 1: s12=0 (0 bits), s13=0 (0 bits) -> s6=0, s7=0 (sum 0)
        # iter 2: lanes with lim>=2 ((lane&3)>=1: 24 in lo, 24 in hi) hit iter==2 -> s12 gets 24 bits; s13 still 0 -> s6=24, s7=24 (sum 48)
        # iter 3: s12 stays 24 bits -> s6=24; lanes with lim>=3 ((lane&3)>=2: 16 in lo, 16 in hi) hit iter==3 -> s13 gets 16 bits, s20 gets 16 bits -> s7=40 (sum 64)
        # iter 4: s12=24, s13=16 -> s6=40, s7=40 (sum 80)
        iter_sums = [0, 48, 64, 80]
        for emulate, sg_sz in ((False, 64), (True, 32)):
            with self.subTest(emulate=emulate):
                spv, _ = _translate_cs(f"cs_bug4b_loop_split_{int(emulate)}", prog, threads=64, emulate=emulate)
                mem = run_compute_workgroup(spv, 64, sg_sz, push_constants=VSHARP)
                for lane in range(64):
                    lim = (lane & 3) + 1
                    want_v3 = sum(iter_sums[:lim])
                    in_s12_13 = (lim >= 2) if lane < 32 else (lim >= 3)
                    in_s20_21 = (lim >= 3) if lane < 32 else False
                    want_v4 = (1 if in_s12_13 else 0) | ((1 if in_s20_21 else 0) << 16)
                    want_v8 = 40 | (16 << 16)
                    self.assertEqual(
                        (mem[16 * lane], mem[16 * lane + 4], mem[16 * lane + 8]),
                        (want_v3, want_v4, want_v8),
                        f"emulate={emulate} lane={lane}",
                    )

    def test_bug5_v_mbcnt_respects_mask_half_in_wave64(self):
        """`v_mbcnt_lo_u32_b32 v4, vcc_hi, 0` and `v_mbcnt_hi_u32_b32 v5, vcc_lo, 0` in wave64 must select
        the known 32-bit half (vcc_hi / vcc_lo) rather than hardcoding lo->0 and hi->1.
        """
        prog = _build_mask_from_s4_s5_w64() + [
            "v_mbcnt_lo_u32_b32 v4, vcc_hi, 0",
            "v_mbcnt_hi_u32_b32 v5, vcc_lo, 0",
            "v_lshlrev_b32_e32 v6, 3, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:0",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:4",
            "s_endpgm",
        ]
        lo, hi = 0x000000A5, 0x0000000F  # vcc_lo=0xA5 (4 bits), vcc_hi=0x0F (bits 0..3 set)
        for emulate, sg_sz in ((False, 64), (True, 32)):
            with self.subTest(emulate=emulate):
                spv, _ = _translate_cs(f"cs_bug5_mbcnt_half_{int(emulate)}", prog, threads=64, emulate=emulate)
                mem = run_compute_workgroup(spv, 64, sg_sz, push_constants=VSHARP + [lo, hi, 0, 0])
                for lane in range(64):
                    want_lo = bin(hi & ((1 << min(lane, 32)) - 1)).count("1")
                    want_hi = 0 if lane < 32 else bin(lo & ((1 << (lane - 32)) - 1)).count("1")
                    self.assertEqual((mem[8 * lane], mem[8 * lane + 4]), (want_lo, want_hi),
                                     f"emulate={emulate} lane={lane}")


# ---------------------------------------------------------------------------
# 3. Barriers inside loops (uniform & wave-divergent) across all modes
# ---------------------------------------------------------------------------

class BarrierInLoopsTests(unittest.TestCase):
    """Audit Hazard 2: s_barrier inside loops with LDS exchange across wave32, wave64, emulate, bar_rounds."""

    def _ring_shift_loop_cs(self, wave64, threads, iters=5):
        """Group-uniform loop with TWO s_barriers per iteration performing `iters` ring shifts in LDS:
        each iteration thread t writes val -> lds[t], s_barrier, reads lds[(t + 1) % threads], s_barrier.
        After `iters` iterations, thread t holds initial((t + iters) % threads) + iters * 100.
        """
        mask = threads - 1  # threads is power of 2 (64 or 128)
        pro = ["s_mov_b64 s[10:11], exec"] if wave64 else []
        return pro + [
            "v_add_nc_u32_e32 v1, 1000, v0",            # initial value = 1000 + tid
            "v_lshlrev_b32_e32 v2, 2, v0",              # my_lds_addr = tid * 4
            "v_add_nc_u32_e32 v3, 1, v0",
            f"v_and_b32_e32 v3, {mask}, v3",
            "v_lshlrev_b32_e32 v3, 2, v3",              # next_lds_addr = ((tid + 1) % threads) * 4
            "s_mov_b32 s8, 0",                          # uniform loop counter
            "loop:",
            "ds_write_b32 v2, v1",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v1, 100, v1",
            "s_barrier",
            "s_add_u32 s8, s8, 1",
            f"s_cmp_lt_u32 s8, {iters}",
            "s_cbranch_scc1 loop",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]

    def test_uniform_loop_barrier_wave32_bar_rounds(self):
        """Native wave32 CS (64 threads = 2 waves) with s_barrier in a group-uniform loop (5 iters = 10 barriers):
        uses bar_scheduler ('barrier rounds'), survives mod-3 ring wraparound & reversed wave schedule.
        """
        prog = self._ring_shift_loop_cs(wave64=False, threads=64, iters=5)
        spv, gen = _translate_cs("cs_bar_loop_w32", prog, threads=64, lds_bytes=256, emulate=False)
        self.assertTrue(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "barrier rounds")
        for rev in (False, True):
            with self.subTest(reverse_subgroups=rev):
                mem = run_compute_workgroup(spv, 64, 32, push_constants=VSHARP, reverse_subgroups=rev)
                want = [1000 + ((t + 5) % 64) + 500 for t in range(64)]
                self.assertEqual([mem.get(4 * t, -1) for t in range(64)], want)

    def test_uniform_loop_barrier_wave64_native_structured(self):
        """Native wave64 CS (128 threads = 2 waves) with s_barrier in a group-uniform loop:
        stays in structured OpLoopMerge with direct OpControlBarrier and works cleanly.
        """
        prog = self._ring_shift_loop_cs(wave64=True, threads=128, iters=5)
        spv, gen = _translate_cs("cs_bar_loop_w64_native", prog, threads=128, lds_bytes=512, emulate=False)
        self.assertFalse(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "structured")
        for rev in (False, True):
            with self.subTest(reverse_subgroups=rev):
                mem = run_compute_workgroup(spv, 128, 64, push_constants=VSHARP, reverse_subgroups=rev)
                want = [1000 + ((t + 5) % 128) + 500 for t in range(128)]
                self.assertEqual([mem.get(4 * t, -1) for t in range(128)], want)

    def test_uniform_loop_barrier_wave64_emulate(self):
        """--wave64-emulate CS (128 threads = 2 emulated waves on 4 subgroups of 32) with s_barrier
        in a group-uniform loop (5 iters = 10 barriers): x_scheduler keeps all waves synchronized.
        """
        prog = self._ring_shift_loop_cs(wave64=True, threads=128, iters=5)
        spv, gen = _translate_cs("cs_bar_loop_w64_emulate", prog, threads=128, lds_bytes=512, emulate=True)
        self.assertEqual(gen.control_flow, "emulated wave64 rounds")
        for rev in (False, True):
            with self.subTest(reverse_subgroups=rev):
                mem = run_compute_workgroup(spv, 128, 32, push_constants=VSHARP, reverse_subgroups=rev)
                want = [1000 + ((t + 5) % 128) + 500 for t in range(128)]
                self.assertEqual([mem.get(4 * t, -1) for t in range(128)], want)

    def _wave_divergent_barrier_loop_cs(self, wave64):
        """Wave 0 runs 2 iterations of a barrier loop and exits via s_endpgm; wave 1 runs 4 iterations.
        On RDNA2, wave 0 exiting releases wave 1's remaining s_barriers cleanly.
        """
        if wave64:
            return [
                "s_mov_b64 s[10:11], exec",
                "v_lshrrev_b32_e32 v1, 6, v0",
                "v_readfirstlane_b32 s8, v1",            # wave_id (0 or 1)
                "s_lshl_b32 s9, s8, 1",                  # 0 or 2
                "s_add_u32 s9, s9, 2",                   # trip count: wave 0 -> 2, wave 1 -> 4
                "s_mov_b32 s12, 0",
                "v_lshlrev_b32_e32 v2, 2, v0",
                "v_mov_b32_e32 v3, 0",
                "loop:",
                "ds_write_b32 v2, v3",
                "s_waitcnt lgkmcnt(0)",
                "s_barrier",
                "ds_read_b32 v4, v2",
                "s_waitcnt lgkmcnt(0)",
                "v_add_nc_u32_e32 v3, 1, v4",
                "s_add_u32 s12, s12, 1",
                "s_cmp_lt_u32 s12, s9",
                "s_cbranch_scc1 loop",
                "buffer_store_dword v3, v2, s[0:3], 0 offen",
                "s_endpgm",
            ]
        return [
            "v_lshrrev_b32_e32 v1, 5, v0",
            "v_readfirstlane_b32 s8, v1",
            "s_lshl_b32 s9, s8, 1",
            "s_add_u32 s9, s9, 2",                       # wave 0 -> 2 iters, wave 1 -> 4 iters
            "s_mov_b32 s12, 0",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v3, 0",
            "loop:",
            "ds_write_b32 v2, v3",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v4, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v3, 1, v4",
            "s_add_u32 s12, s12, 1",
            "s_cmp_lt_u32 s12, s9",
            "s_cbranch_scc1 loop",
            "buffer_store_dword v3, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]

    def test_wave_divergent_barrier_loop_bar_rounds_and_emulate(self):
        """Wave-divergent barrier loop (wave 0 runs 2 iters and exits, wave 1 runs 4 iters):
          - native wave32 CS (bar_rounds): passes cleanly with uniform outer OpControlBarrier!
          - --wave64-emulate CS (x_scheduler): passes cleanly with uniform outer OpControlBarrier!
          - native wave64 CS (emulate=False): direct OpControlBarrier inside structured loop hits
            NonUniformBarrierError (documented AMD-only native-wave64 limitation; non-AMD uses emulate).
        """
        spv_w32, gen_w32 = _translate_cs(
            "cs_div_bar_w32", self._wave_divergent_barrier_loop_cs(False), threads=64, lds_bytes=256, emulate=False
        )
        self.assertTrue(gen_w32.bar_rounds)
        mem32 = run_compute_workgroup(spv_w32, 64, 32, push_constants=VSHARP)
        self.assertEqual([mem32[4 * t] for t in range(64)], [2] * 32 + [4] * 32)

        spv_em, gen_em = _translate_cs(
            "cs_div_bar_em", self._wave_divergent_barrier_loop_cs(True), threads=128, lds_bytes=512, emulate=True
        )
        self.assertEqual(gen_em.control_flow, "emulated wave64 rounds")
        mem_em = run_compute_workgroup(spv_em, 128, 32, push_constants=VSHARP)
        self.assertEqual([mem_em[4 * t] for t in range(128)], [2] * 64 + [4] * 64)

        spv_nat64, gen_nat64 = _translate_cs(
            "cs_div_bar_nat64", self._wave_divergent_barrier_loop_cs(True), threads=128, lds_bytes=512, emulate=False
        )
        self.assertEqual(gen_nat64.control_flow, "structured")
        with self.assertRaises(NonUniformBarrierError):
            run_compute_workgroup(spv_nat64, 128, 64, push_constants=VSHARP)

    def test_combined_barrier_and_divergent_ballot_in_same_loop(self):
        """Group-uniform loop containing BOTH s_barrier + LDS exchange AND divergent lane-level
        s_and_saveexec + s_bcnt1, tested across wave32 (bar_rounds), native wave64, and --wave64-emulate.
        """
        def make_prog(wave64):
            if wave64:
                return [
                    "s_mov_b64 s[10:11], exec",
                    "v_lshlrev_b32_e32 v2, 2, v0",
                    "v_mov_b32_e32 v1, v0",
                    "s_mov_b32 s8, 0",
                    "loop:",
                    "ds_write_b32 v2, v1",
                    "s_waitcnt lgkmcnt(0)",
                    "s_barrier",
                    "v_and_b32_e32 v3, 3, v0",
                    "v_cmp_ge_u32_e32 vcc, v3, s8",
                    "s_and_saveexec_b64 s[12:13], vcc",
                    "s_bcnt1_i32_b64 s6, exec",
                    "v_xor_b32_e32 v4, 1, v0",
                    "v_lshlrev_b32_e32 v4, 2, v4",
                    "ds_read_b32 v5, v4",
                    "s_waitcnt lgkmcnt(0)",
                    "v_add_nc_u32_e32 v1, v5, s6",
                    "s_mov_b64 exec, s[12:13]",
                    "s_barrier",
                    "s_add_u32 s8, s8, 1",
                    "s_cmp_lt_u32 s8, 4",
                    "s_cbranch_scc1 loop",
                    "buffer_store_dword v1, v2, s[0:3], 0 offen",
                    "s_endpgm",
                ]
            return [
                "s_mov_b32 s10, exec_lo",
                "v_lshlrev_b32_e32 v2, 2, v0",
                "v_mov_b32_e32 v1, v0",
                "s_mov_b32 s8, 0",
                "loop:",
                "ds_write_b32 v2, v1",
                "s_waitcnt lgkmcnt(0)",
                "s_barrier",
                "v_and_b32_e32 v3, 3, v0",
                "v_cmp_ge_u32_e32 vcc_lo, v3, s8",
                "s_and_saveexec_b32 s12, vcc_lo",
                "s_bcnt1_i32_b32 s6, exec_lo",
                "v_xor_b32_e32 v4, 1, v0",
                "v_lshlrev_b32_e32 v4, 2, v4",
                "ds_read_b32 v5, v4",
                "s_waitcnt lgkmcnt(0)",
                "v_add_nc_u32_e32 v1, v5, s6",
                "s_mov_b32 exec_lo, s12",
                "s_barrier",
                "s_add_u32 s8, s8, 1",
                "s_cmp_lt_u32 s8, 4",
                "s_cbranch_scc1 loop",
                "buffer_store_dword v1, v2, s[0:3], 0 offen",
                "s_endpgm",
            ]

        def oracle(threads, wave_size):
            v1 = list(range(threads))
            for s8 in range(4):
                lds = list(v1)
                nxt = list(v1)
                for wbase in range(0, threads, wave_size):
                    active = [t for t in range(wbase, wbase + wave_size) if (t & 3) >= s8]
                    cnt = len(active)
                    for t in active:
                        nxt[t] = (lds[t ^ 1] + cnt) & M32
                v1 = nxt
            return v1

        for label, wave64, threads, sg_sz, emulate in (
            ("w32_bar_rounds", False, 64, 32, False),
            ("w64_native", True, 128, 64, False),
            ("w64_emulate", True, 128, 32, True),
        ):
            with self.subTest(mode=label):
                spv, _ = _translate_cs("cs_combo_" + label, make_prog(wave64),
                                       threads=threads, lds_bytes=threads * 4, emulate=emulate)
                mem = run_compute_workgroup(spv, threads, sg_sz, push_constants=VSHARP)
                self.assertEqual([mem[4 * t] for t in range(threads)], oracle(threads, 64 if wave64 else 32))


if __name__ == "__main__":
    unittest.main()
