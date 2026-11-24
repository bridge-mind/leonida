"""A strict interpreter that runs one subgroup of the SPIR-V that tools/rdna2spirv.py emits, every invocation at
once, so control flow and subgroup operations can be checked by value.

tests/semantic/spirv_mini_interp.py runs one invocation of forward-only code. This one runs `wave` invocations
(one per RDNA2 lane, the translator's execution model) through structured control flow:

  * Written from the SPIR-V specification (section 2.11 "Structured Control Flow", 3.32.17 "Non-Uniform
    Instructions") and shares no code with the translator.
  * Structured execution: the invocations that reach a block together run it together. At an
    OpBranchConditional or OpSwitch with an OpSelectionMerge they split by target, each part runs until it
    reaches the merge block (or leaves the construct: a break, a continue, a return or a kill), and the parts
    rejoin there. A loop (OpLoopMerge) runs iterations until no invocation is left in it; the invocations that
    branch to its merge block wait there for the others. This is the convergence SPIR-V's structured rules give
    when every invocation that enters a construct leaves it through its merge (the "maximal reconvergence"
    reading); a subgroup operation sees exactly the invocations of its group that have not terminated.
  * Strict: any opcode, extended instruction, type, storage class or scope not listed raises
    UnsupportedConstruct, and so does a value SPIR-V leaves undefined (a shift by the bit width or more,
    a ballot query with no bit set, a shuffle from an invocation that is not running) when it is actually used.
  * Floats are carried as their bit patterns (class F) and may only be moved: loaded, stored, bitcast, selected,
    composited. Float arithmetic raises, so a NaN payload can never be rewritten by the host.

Inputs: Input variables decorated BuiltIn SubgroupLocalInvocationId read the lane number, SubgroupSize reads the
wave size; BaseVertex and BaseInstance read 0; other Input variables read zero, except the built-ins named in
`builtins` (name -> function of the
lane number giving that lane's integer value, e.g. {"VertexIndex": lambda lane: first + lane}). PushConstant words
can be seeded (default zero).

    run_wave(text, wave=32, push_constants=None, max_steps=2_000_000, builtins=None) ->
        {"outputs": [ {output_name: tuple_of_words} or None (lane killed) for each lane ],
         "steps": instructions executed (summed over invocations)}
"""
import math
import re
import struct

MASK = 0xFFFFFFFF
SUBGROUP = 3        # Scope Subgroup


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


class InvalidModule(AssertionError):
    """The module breaks spirv-val's merge rule (see spirv_mini_interp.check_structure): a merge instruction must
    directly precede its block's branch. Raised instead of running a shader no driver would accept."""


_MERGE_FOLLOW = {"OpSelectionMerge": ("OpBranchConditional", "OpSwitch"),
                 "OpLoopMerge": ("OpBranch", "OpBranchConditional")}


class F:
    """A 32-bit float carried as its bit pattern."""
    __slots__ = ("bits",)

    def __init__(self, bits):
        self.bits = bits & MASK

    def __eq__(self, other):
        return isinstance(other, F) and other.bits == self.bits

    def __hash__(self):
        return hash(("F", self.bits))

    def __repr__(self):
        return f"F(0x{self.bits:08x})"


class _UsedPoison(Exception):
    def __init__(self, poison):
        super().__init__(poison.why)
        self.poison = poison


class _Poison:
    """A value SPIR-V leaves undefined. Pure operations on it give another undefined value; it raises where it
    could change what the program does: a branch or select condition, an access-chain index, a subgroup
    operation, an output."""
    __slots__ = ("why",)

    def __init__(self, why):
        self.why = why

    def __repr__(self):
        return f"poison({self.why})"


_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")
_HEADER_IGNORED = frozenset({
    "OpCapability", "OpExtension", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpName", "OpMemberName", "OpSource",
})
_TERMINATORS = frozenset({"OpBranch", "OpBranchConditional", "OpSwitch", "OpReturn", "OpKill",
                          "OpTerminateInvocation", "OpUnreachable"})
# Instructions that move a value without inspecting it (poison may pass through).
_MOVES = frozenset({"OpCopyObject", "OpStore", "OpLoad", "OpSelect", "OpCompositeConstruct"})


def _signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _f32_unpack(v):
    if not isinstance(v, F):
        raise UnsupportedConstruct(f"float operation on {type(v).__name__}")
    return struct.unpack("<f", struct.pack("<I", v.bits))[0]


def _f32_pack(x):
    if math.isnan(x):
        return F(0x7FC00000)
    try:
        return F(struct.unpack("<I", struct.pack("<f", x))[0])
    except OverflowError:
        return F(0xFF800000 if math.copysign(1.0, x) < 0 else 0x7F800000)


def _wrap_input(v):
    if isinstance(v, float):
        return _f32_pack(v)
    if isinstance(v, (tuple, list)):
        return tuple(_wrap_input(x) for x in v)
    return v


class _Block:
    __slots__ = ("label", "phis", "body", "merge", "term")

    def __init__(self, label):
        self.label, self.phis, self.body, self.merge, self.term = label, [], [], None, None


class _Lane:
    __slots__ = ("index", "values", "pointers", "prev", "outputs")

    def __init__(self, index):
        self.index = index
        self.values = {}
        self.pointers = {}
        self.prev = None
        self.outputs = {}


class _Module:
    def __init__(self, text):
        self.types = {}
        self.consts = {}
        self.ptr_types = {}
        self.globals = []           # (id, storage class, pointee type, initializer id)
        self.builtins = {}
        self.locations = {}
        self.blocks = {}
        self.entry = None
        self.id_type = {}
        cur = None
        in_body = False
        merge_line = None           # (opcode, line) of a merge whose branch has not been seen yet
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                if merge_line is not None:
                    if op not in _MERGE_FOLLOW[merge_line[0]]:
                        raise InvalidModule(f"{merge_line[1]!r} is followed by {line!r}, not by "
                                            f"{' or '.join(_MERGE_FOLLOW[merge_line[0]])}")
                    merge_line = None
                if op == "OpFunctionEnd":
                    in_body = False
                    continue
                if op == "OpLabel":
                    cur = _Block(res)
                    self.blocks[res] = cur
                    if self.entry is None:
                        self.entry = res
                    continue
                if cur is None:
                    if op == "OpFunctionParameter":
                        raise UnsupportedConstruct("function parameters")
                    raise UnsupportedConstruct(f"{op} before the first label")
                if res and a and op != "OpStore":
                    self.id_type[res] = a[0]
                if op == "OpVariable" and a[1:2] == ["Function"] and len(self.blocks) > 1:
                    raise InvalidModule(f"{line!r} is a Function variable outside the function's first block")
                if op == "OpPhi":
                    if cur.body:
                        raise InvalidModule(f"{line!r} follows a non-phi instruction of its block (spirv-val: "
                                            "OpPhi must come first)")
                    cur.phis.append((res, a))
                elif op in ("OpSelectionMerge", "OpLoopMerge"):
                    cur.merge = (op, a[0], a[1] if op == "OpLoopMerge" else None)
                    merge_line = (op, line)
                elif op in _TERMINATORS:
                    cur.term = (op, a)
                    cur = None
                else:
                    cur.body.append((res, op, a))
                continue
            if op == "OpFunction":
                if self.blocks:
                    raise UnsupportedConstruct("more than one function")
                in_body = True
                continue
            if op in _HEADER_IGNORED:
                continue
            if op == "OpDecorate":
                if len(a) >= 3 and a[1] == "BuiltIn":
                    self.builtins[a[0]] = a[2]
                elif len(a) >= 3 and a[1] == "Location":
                    self.locations[a[0]] = int(a[2])
                continue
            self._header(res, op, a)
        if self.entry is None:
            raise UnsupportedConstruct("no function body")
        for b in self.blocks.values():
            if b.term is None:
                raise UnsupportedConstruct(f"block {b.label} has no terminator")

    def _header(self, res, op, a):
        t = self.types
        if op == "OpTypeInt":
            if int(a[0]) not in (32, 64):
                raise UnsupportedConstruct(f"{a[0]}-bit integer type")
            t[res] = ("int", int(a[0]), int(a[1]))
        elif op == "OpTypeFloat":
            if int(a[0]) != 32:
                raise UnsupportedConstruct(f"{a[0]}-bit float type")
            t[res] = ("float",)
        elif op == "OpTypeBool":
            t[res] = ("bool",)
        elif op == "OpTypeVector":
            t[res] = ("vec", a[0], int(a[1]))
        elif op == "OpTypeArray":
            t[res] = ("arr", a[0], int(self.consts[a[1]]))
        elif op == "OpTypeRuntimeArray":
            t[res] = ("rarr", a[0])
        elif op == "OpTypeStruct":
            t[res] = ("struct", a)
        elif op == "OpTypePointer":
            self.ptr_types[res] = (a[0], a[1])
        elif op == "OpConstant":
            k = t[a[0]]
            if k[0] == "int":
                self.consts[res] = int(a[1], 0) & ((1 << k[1]) - 1)
            elif k[0] == "float":
                self.consts[res] = F(_float_literal_bits(a[1]))
            else:
                raise UnsupportedConstruct(f"OpConstant of type {k[0]}")
            self.id_type[res] = a[0]
        elif op == "OpConstantTrue":
            self.consts[res] = True
        elif op == "OpConstantFalse":
            self.consts[res] = False
        elif op == "OpConstantNull":
            self.consts[res] = self.zero(a[0])
            self.id_type[res] = a[0]
        elif op == "OpConstantComposite":
            self.consts[res] = tuple(self.consts[x] for x in a[1:])
        elif op == "OpVariable":
            cls, pointee = self.ptr_types[a[0]]
            if a[1] != cls:
                raise UnsupportedConstruct("OpVariable storage class mismatch")
            if cls not in ("Input", "Output", "PushConstant", "Private"):
                raise UnsupportedConstruct(f"global variable in storage class {cls}")
            self.globals.append((res, cls, pointee, a[2] if len(a) > 2 else None))
        else:
            raise UnsupportedConstruct(f"{op} in module header")

    def zero(self, type_id):
        k = self.types[type_id]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return F(0)
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(self.zero(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [self.zero(k[1]) for _ in range(k[2])]
        if k[0] == "struct":
            return [self.zero(m) for m in k[1]]
        raise UnsupportedConstruct(f"zero value of type {type_id} ({k[0]})")

    def size(self, type_id):
        k = self.types[type_id]
        if k[0] in ("int", "float"):
            return 4 if k[0] == "float" or k[1] == 32 else 8
        if k[0] == "vec":
            return self.size(k[1]) * k[2]
        if k[0] == "arr":
            return self.size(k[1]) * k[2]
        if k[0] == "struct":
            return sum(self.size(m) for m in k[1])
        raise UnsupportedConstruct(f"size of type {type_id} ({k[0]})")

    def bits(self, type_id):
        k = self.types[type_id]
        if k[0] != "int":
            raise UnsupportedConstruct(f"integer width of a {k[0]}")
        return k[1]


def _float_literal_bits(text):
    import struct
    try:
        return struct.unpack("<I", struct.pack("<f", float(text)))[0]
    except (OverflowError, ValueError):
        raise UnsupportedConstruct(f"float literal {text!r}") from None


def _copy(v):
    return [_copy(x) for x in v] if isinstance(v, list) else v


class _Wave:
    def __init__(self, mod, wave, push_constants, max_steps, builtins=None, inputs=None, memory=None):
        self.m = mod
        self.wave = wave
        self.pcw = list(push_constants or [])
        self.max_steps = max_steps
        self.steps = 0
        self.lanes = [_Lane(i) for i in range(wave)]
        self.killed = set()
        self.returned = set()
        self.memory = memory if memory is not None else {}
        for lane in self.lanes:
            for vid, cls, pointee, init in mod.globals:
                if cls == "PushConstant":
                    lane.pointers[vid] = ("pc", 0, pointee)
                    continue
                if cls == "Input":
                    b = mod.builtins.get(vid)
                    if b == "SubgroupLocalInvocationId":
                        value = lane.index
                    elif b == "SubgroupSize":
                        value = wave
                    elif builtins and b in builtins:
                        value = builtins[b](lane.index) & MASK
                    elif b in ("BaseVertex", "BaseInstance"):
                        value = 0     # a draw with no vertex/instance offset, unless `builtins` seeds one
                    elif b is None:
                        loc = mod.locations.get(vid)
                        if inputs is not None and loc in inputs:
                            value = _wrap_input(inputs[loc](lane.index))
                        else:
                            value = mod.zero(pointee)
                    else:
                        value = _BuiltinNotModelled(b)
                    box = [value]
                else:
                    box = [_copy(mod.consts[init]) if init else mod.zero(pointee)]
                    if cls == "Output":
                        lane.outputs[vid] = box
                lane.pointers[vid] = (box, 0, pointee)

    # -- values --------------------------------------------------------------------------------------------------
    def raw(self, lane, x):
        if x in lane.values:
            return lane.values[x]
        if x in self.m.consts:
            return self.m.consts[x]
        raise UnsupportedConstruct(f"use of undefined id {x}")

    def val(self, lane, x):
        v = self.raw(lane, x)
        if isinstance(v, _Poison):
            raise _UsedPoison(v)
        if isinstance(v, _BuiltinNotModelled):
            raise UnsupportedConstruct(f"BuiltIn {v.name} is not modelled")
        return v

    def int_of(self, lane, x):
        v = self.val(lane, x)
        if isinstance(v, bool) or not isinstance(v, int):
            raise UnsupportedConstruct(f"integer operation on {type(v).__name__} ({x})")
        return v

    def bool_of(self, lane, x):
        v = self.val(lane, x)
        if not isinstance(v, bool):
            raise UnsupportedConstruct(f"boolean operation on {type(v).__name__} ({x})")
        return v

    # -- structured execution ------------------------------------------------------------------------------------
    def run(self):
        group = set(range(self.wave))
        rest = self.region(group, self.m.entry, frozenset())
        if rest:
            raise UnsupportedConstruct(f"invocations left the function through {sorted(rest)}")
        out = []
        for lane in self.lanes:
            if lane.index in self.killed:
                out.append(None)
                continue
            words = {}
            for vid, box in lane.outputs.items():
                v = box[0]
                flat = v if isinstance(v, (tuple, list)) else (v,)
                ws = []
                for x in flat:
                    if isinstance(x, _Poison):
                        raise UnsupportedConstruct(f"output {vid} holds an undefined value ({x.why})")
                    ws.append(x.bits if isinstance(x, F) else int(x) & MASK)
                words[vid] = tuple(ws)
            out.append(words)
        return out

    def region(self, group, cur, stops, iterate=False):
        """Run `group` from block `cur` until each invocation reaches one of `stops` or terminates.
        Returns {stop label: set of invocations}. `iterate` runs a loop header's own block (one iteration)."""
        out = {}

        def leave(label, g):
            out.setdefault(label, set()).update(g)

        while group:
            if cur in stops and not iterate:
                leave(cur, group)
                return out
            blk = self.m.blocks[cur]
            if blk.merge and blk.merge[0] == "OpLoopMerge" and not iterate:
                merge, cont = blk.merge[1], blk.merge[2]
                active, exited = set(group), set()
                while active:
                    r = self.region(active, cur, stops | {merge, cont, cur}, iterate=True)
                    back = r.pop(cur, set())
                    exited |= r.pop(merge, set())
                    to_cont = r.pop(cont, set()) if cont != cur else set()
                    for k, g in r.items():
                        leave(k, g)
                    if to_cont:
                        r2 = self.region(to_cont, cont, stops | {merge, cur})
                        back |= r2.pop(cur, set())
                        exited |= r2.pop(merge, set())
                        for k, g in r2.items():
                            leave(k, g)
                    active = back
                group, cur = exited, merge
                continue
            iterate = False
            self.exec_block(blk, group)
            op, a = blk.term
            if op in ("OpReturn",):
                self.returned |= group
                return out
            if op in ("OpKill", "OpTerminateInvocation"):
                self.killed |= group
                return out
            if op == "OpUnreachable":
                raise UnsupportedConstruct(f"OpUnreachable reached in {cur}")
            targets = self.branch_targets(blk, group)
            for lane_ids in targets.values():
                for i in lane_ids:
                    self.lanes[i].prev = cur
            if len(targets) == 1:
                (cur, group), = targets.items()
                if blk.merge and blk.merge[0] == "OpSelectionMerge" and cur != blk.merge[1]:
                    # Every invocation took the same side: run it to the merge as a construct of its own.
                    r = self.region(group, cur, stops | {blk.merge[1]})
                    group = r.pop(blk.merge[1], set())
                    for k, g in r.items():
                        leave(k, g)
                    cur = blk.merge[1]
                continue
            if blk.merge and blk.merge[0] == "OpSelectionMerge":
                merge = blk.merge[1]
                joined = set()
                for t in self.target_order(blk):
                    g = targets.get(t)
                    if not g:
                        continue
                    r = self.region(g, t, stops | {merge})
                    joined |= r.pop(merge, set())
                    for k, gg in r.items():
                        leave(k, gg)
                group, cur = joined, merge
                continue
            inner = [(t, g) for t, g in targets.items() if t not in stops]
            for t, g in targets.items():
                if t in stops:
                    leave(t, g)
            if len(inner) > 1:
                raise UnsupportedConstruct(f"{cur}: invocations split between {sorted(t for t, _ in inner)} "
                                           "without a merge instruction")
            if not inner:
                return out
            cur, group = inner[0]
        return out

    @staticmethod
    def target_order(blk):
        op, a = blk.term
        if op == "OpBranchConditional":
            return [a[1], a[2]] if a[1] != a[2] else [a[1]]
        if op == "OpSwitch":
            seen, order = set(), []
            for t in [a[1]] + a[3::2]:
                if t not in seen:
                    seen.add(t)
                    order.append(t)
            return order
        return [a[0]]

    def branch_targets(self, blk, group):
        op, a = blk.term
        targets = {}
        for i in sorted(group):
            lane = self.lanes[i]
            try:
                t = self._target(op, a, lane)
            except _UsedPoison as p:
                raise UnsupportedConstruct(f"branch in {blk.label} on an undefined value ({p.poison.why})") from None
            targets.setdefault(t, set()).add(i)
        return targets

    def _target(self, op, a, lane):
        if op == "OpBranch":
            return a[0]
        if op == "OpBranchConditional":
            return a[1] if self.bool_of(lane, a[0]) else a[2]
        if op == "OpSwitch":
            sel = self.int_of(lane, a[0])
            mask = (1 << self.m.bits(self.m.id_type[a[0]])) - 1
            for k in range(2, len(a) - 1, 2):
                if (int(a[k], 0) & mask) == sel:
                    return a[k + 1]
            return a[1]
        raise UnsupportedConstruct(op)

    def exec_block(self, blk, group):
        members = sorted(group)
        if blk.phis:
            for i in members:
                lane = self.lanes[i]
                new = {}
                for res, a in blk.phis:
                    pairs = [(a[k], a[k + 1]) for k in range(1, len(a) - 1, 2)]
                    src = [v for v, p in pairs if p == lane.prev]
                    if len(src) != 1:
                        raise UnsupportedConstruct(f"OpPhi {res}: no single incoming value from {lane.prev}")
                    new[res] = self.raw(lane, src[0])
                lane.values.update(new)
        for res, op, a in blk.body:
            self.steps += len(members)
            if self.steps > self.max_steps:
                raise UnsupportedConstruct(f"step limit {self.max_steps} reached (a loop that does not end?)")
            if op.startswith("OpGroupNonUniform"):
                try:
                    self.collective(res, op, a, members)
                except _UsedPoison as p:
                    raise UnsupportedConstruct(f"{op} on an undefined value ({p.poison.why})") from None
            else:
                for i in members:
                    try:
                        v = self.eval(self.lanes[i], res, op, a)
                    except _UsedPoison as p:
                        # OpSelect reads its arms without inspecting them, so only its condition gets here.
                        if op in ("OpAccessChain", "OpSelect"):
                            raise UnsupportedConstruct(f"{op} {res} depends on an undefined value "
                                                       f"({p.poison.why})") from None
                        v = p.poison
                    if res:
                        self.lanes[i].values[res] = v

    # -- subgroup operations -------------------------------------------------------------------------------------
    def collective(self, res, op, a, members):
        if self.raw(self.lanes[members[0]], a[1]) != SUBGROUP:
            raise UnsupportedConstruct(f"{op} with a scope other than Subgroup")
        L = self.lanes
        width = self.wave

        def ballot_bits(v):
            if not (isinstance(v, tuple) and len(v) == 4):
                raise UnsupportedConstruct(f"{op} on a non-uvec4 ballot")
            bits = v[0] | (v[1] << 32) | (v[2] << 64) | (v[3] << 96)
            return bits & ((1 << width) - 1)

        if op in ("OpGroupNonUniformAny", "OpGroupNonUniformAll", "OpGroupNonUniformAllEqual"):
            vs = [self.val(L[i], a[2]) for i in members]
            if op != "OpGroupNonUniformAllEqual" and not all(isinstance(v, bool) for v in vs):
                raise UnsupportedConstruct(f"{op} on a non-bool")
            r = any(vs) if op == "OpGroupNonUniformAny" else all(vs) if op == "OpGroupNonUniformAll" \
                else all(v == vs[0] for v in vs)
            for i in members:
                L[i].values[res] = r
        elif op == "OpGroupNonUniformBallot":
            bits = 0
            for i in members:
                if self.bool_of(L[i], a[2]):
                    bits |= 1 << i
            r = tuple((bits >> (32 * k)) & MASK for k in range(4))
            for i in members:
                L[i].values[res] = r
        elif op == "OpGroupNonUniformInverseBallot":
            for i in members:
                L[i].values[res] = bool((ballot_bits(self.val(L[i], a[2])) >> i) & 1)
        elif op == "OpGroupNonUniformBallotBitExtract":
            for i in members:
                idx = self.int_of(L[i], a[3])
                L[i].values[res] = bool((ballot_bits(self.val(L[i], a[2])) >> idx) & 1) if idx < 128 else \
                    _Poison("ballot bit index >= 128")
        elif op == "OpGroupNonUniformBallotBitCount":
            kind = a[2]
            for i in members:
                bits = ballot_bits(self.val(L[i], a[3]))
                if kind == "Reduce":
                    n = bits
                elif kind == "InclusiveScan":
                    n = bits & ((1 << (i + 1)) - 1)
                elif kind == "ExclusiveScan":
                    n = bits & ((1 << i) - 1)
                else:
                    raise UnsupportedConstruct(f"{op} {kind}")
                L[i].values[res] = n.bit_count()
        elif op in ("OpGroupNonUniformBallotFindLSB", "OpGroupNonUniformBallotFindMSB"):
            for i in members:
                bits = ballot_bits(self.val(L[i], a[2]))
                if not bits:
                    L[i].values[res] = _Poison(f"{op[17:]} of an empty ballot")
                elif op.endswith("LSB"):
                    L[i].values[res] = (bits & -bits).bit_length() - 1
                else:
                    L[i].values[res] = bits.bit_length() - 1
        elif op in ("OpGroupNonUniformBitwiseOr", "OpGroupNonUniformBitwiseAnd", "OpGroupNonUniformBitwiseXor",
                    "OpGroupNonUniformIAdd"):
            kind = a[2]
            if kind != "Reduce":
                raise UnsupportedConstruct(f"{op} {kind}")
            w = (1 << self.m.bits(a[0])) - 1
            vs = [self.int_of(L[i], a[3]) for i in members]
            r = vs[0]
            for v in vs[1:]:
                r = {"OpGroupNonUniformBitwiseOr": r | v, "OpGroupNonUniformBitwiseAnd": r & v,
                     "OpGroupNonUniformBitwiseXor": r ^ v, "OpGroupNonUniformIAdd": r + v}[op]
            for i in members:
                L[i].values[res] = r & w
        elif op in ("OpGroupNonUniformShuffle", "OpGroupNonUniformBroadcast"):
            src_vals = {i: self.raw(L[i], a[2]) for i in members}
            for i in members:
                idx = self.int_of(L[i], a[3])
                L[i].values[res] = src_vals[idx] if idx in src_vals else \
                    _Poison(f"{op[17:]} from invocation {idx}, which is not running")
        elif op == "OpGroupNonUniformBroadcastFirst":
            first = self.raw(L[members[0]], a[2])
            for i in members:
                L[i].values[res] = first
        elif op == "OpGroupNonUniformElect":
            for i in members:
                L[i].values[res] = i == members[0]
        elif op == "OpGroupNonUniformQuadSwap":
            src_vals = {i: self.raw(L[i], a[2]) for i in members}
            for i in members:
                d = self.int_of(L[i], a[3])
                if d not in (0, 1, 2):
                    raise UnsupportedConstruct(f"OpGroupNonUniformQuadSwap direction {d}")
                partner = i ^ (d + 1)
                L[i].values[res] = src_vals[partner] if partner in src_vals else                     _Poison(f"QuadSwap from invocation {partner}, which is not running")
        elif op == "OpGroupNonUniformQuadBroadcast":
            src_vals = {i: self.raw(L[i], a[2]) for i in members}
            for i in members:
                idx = self.int_of(L[i], a[3])
                if not 0 <= idx < 4:
                    raise UnsupportedConstruct(f"OpGroupNonUniformQuadBroadcast index {idx}")
                partner = (i & ~3) | idx
                L[i].values[res] = src_vals[partner] if partner in src_vals else                     _Poison(f"QuadBroadcast from invocation {partner}, which is not running")
        else:
            raise UnsupportedConstruct(op)

    # -- per-invocation instructions -----------------------------------------------------------------------------
    def eval(self, lane, res, op, a):
        m = self.m
        if op == "OpVariable":
            cls, pointee = m.ptr_types[a[0]]
            if cls != "Function":
                raise UnsupportedConstruct(f"{cls} variable in a function body")
            init = _copy(self.raw(lane, a[2])) if len(a) > 2 else m.zero(pointee)
            lane.pointers[res] = ([init], 0, pointee)
            return None
        if op == "OpAccessChain":
            p = lane.pointers.get(a[1])
            if p is None:
                raise UnsupportedConstruct(f"OpAccessChain on {a[1]}, which is not a pointer")
            if p[0] == "psb":
                addr, ty = p[1], p[2]
                for x in a[2:]:
                    idx = self.int_of(lane, x)
                    k = m.types[ty]
                    if k[0] in ("arr", "rarr", "vec"):
                        addr += m.size(k[1]) * idx
                        ty = k[1]
                    elif k[0] == "struct":
                        addr += sum(m.size(t) for t in k[1][:idx])
                        ty = k[1][idx]
                    else:
                        raise UnsupportedConstruct(f"psb access chain into {k[0]}")
                lane.pointers[res] = ("psb", addr, ty)
                return None
            if p[0] == "psb":
                if m.size(p[2]) != 4:
                    raise UnsupportedConstruct("psb load wider than one word")
                v = self.memory.get(p[1], 0)
                return F(v) if m.types[p[2]][0] == "float" else v & MASK
            if p[0] == "pc":
                off, ty = p[1], p[2]
                for x in a[2:]:
                    idx = self.int_of(lane, x)
                    k = m.types[ty]
                    if k[0] == "struct":
                        off += sum(m.size(t) for t in k[1][:idx])
                        ty = k[1][idx]
                    elif k[0] in ("arr", "vec"):
                        off += m.size(k[1]) * idx
                        ty = k[1]
                    else:
                        raise UnsupportedConstruct(f"push-constant access chain into {k[0]}")
                lane.pointers[res] = ("pc", off, ty)
                return None
            box, i, ty = p
            for x in a[2:]:
                idx = self.int_of(lane, x)
                container = box[i]
                k = m.types[ty]
                if k[0] not in ("arr", "vec", "struct"):
                    raise UnsupportedConstruct(f"access chain into {k[0]}")
                if not 0 <= idx < len(container):
                    raise UnsupportedConstruct(f"access chain index {idx} out of range {len(container)}")
                if isinstance(container, tuple):
                    container = list(container)
                    box[i] = container
                box, i, ty = container, idx, (k[1][idx] if k[0] == "struct" else k[1])
            lane.pointers[res] = (box, i, ty)
            return None
        if op == "OpLoad":
            p = lane.pointers.get(a[1])
            if p is None:
                raise UnsupportedConstruct(f"OpLoad from {a[1]}, which is not a pointer")
            if p[0] == "pc":
                if m.size(p[2]) != 4:
                    raise UnsupportedConstruct("push-constant load wider than one word")
                w = p[1] // 4
                v = self.pcw[w] if w < len(self.pcw) else 0
                return F(v) if m.types[p[2]][0] == "float" else v & MASK
            v = p[0][p[1]]
            if isinstance(v, list) and m.types[p[2]][0] == "vec":
                return tuple(v)
            return _copy(v)
        if op == "OpConvertUToPtr":
            addr = self.int_of(lane, a[1])
            cls, pointee = m.ptr_types[a[0]]
            if cls != "PhysicalStorageBuffer":
                raise UnsupportedConstruct(f"OpConvertUToPtr to storage class {cls}")
            lane.pointers[res] = ("psb", addr, pointee)
            return None
        if op == "OpAtomicIAdd":
            p = lane.pointers.get(a[1])
            if p is None or p[0] == "pc":
                raise UnsupportedConstruct(f"OpAtomicIAdd on {a[1]}")
            val = self.int_of(lane, a[4])
            if p[0] == "psb":
                old = self.memory.get(p[1], 0)
                self.memory[p[1]] = (old + val) & MASK
                return old
            old = p[0][p[1]] & MASK
            p[0][p[1]] = (old + val) & MASK
            return old
        if op == "OpStore":
            p = lane.pointers.get(a[0])
            if p is None or p[0] == "pc":
                raise UnsupportedConstruct(f"OpStore to {a[0]}")
            if p[0] == "psb":
                v = self.val(lane, a[1])
                self.memory[p[1]] = (v.bits if isinstance(v, F) else int(v)) & MASK
                return None
            p[0][p[1]] = _copy(self.raw(lane, a[1]))
            return None
        if op == "OpSelect":
            c = self.val(lane, a[1])
            if isinstance(c, tuple):
                x, y = self.raw(lane, a[2]), self.raw(lane, a[3])
                return tuple(xv if cv else yv for cv, xv, yv in zip(c, x, y))
            if not isinstance(c, bool):
                raise UnsupportedConstruct("OpSelect on a non-bool condition")
            return self.raw(lane, a[2]) if c else self.raw(lane, a[3])
        if op == "OpCopyObject":
            return self.raw(lane, a[1])
        if op == "OpCompositeConstruct":
            return tuple(self.raw(lane, x) for x in a[1:])
        if op == "OpCompositeExtract":
            v = self.val(lane, a[1])
            for k in a[2:]:
                v = v[int(k)]
            return v
        if op == "OpCompositeInsert":
            v = list(self.val(lane, a[2]))
            if len(a) != 4:
                raise UnsupportedConstruct("nested OpCompositeInsert")
            v[int(a[3])] = self.raw(lane, a[1])
            return tuple(v)
        if op == "OpBitcast":
            k, v = m.types[a[0]], self.val(lane, a[1])
            if k[0] == "float" and isinstance(v, int) and not isinstance(v, bool):
                return F(v)
            if k[0] == "int" and isinstance(v, F):
                return v.bits
            if k[0] == "int" and isinstance(v, int) and not isinstance(v, bool):
                return v & ((1 << k[1]) - 1)
            if k[0] == "vec" and isinstance(v, tuple):
                ek = m.types[k[1]]
                return tuple(F(x) if ek[0] == "float" else (x.bits if isinstance(x, F) else x) for x in v)
            raise UnsupportedConstruct("OpBitcast between these types")
        if op in ("OpLogicalAnd", "OpLogicalOr", "OpLogicalEqual", "OpLogicalNotEqual"):
            x, y = self.bool_of(lane, a[1]), self.bool_of(lane, a[2])
            return {"OpLogicalAnd": x and y, "OpLogicalOr": x or y, "OpLogicalEqual": x == y,
                    "OpLogicalNotEqual": x != y}[op]
        if op == "OpLogicalNot":
            return not self.bool_of(lane, a[1])
        if op == "OpFNegate":
            if m.types.get(a[0], ("",))[0] != "float":
                raise UnsupportedConstruct(f"OpFNegate with result type {a[0]}")
            x = self.val(lane, a[1])
            if not isinstance(x, F):
                raise UnsupportedConstruct(f"OpFNegate on {type(x).__name__}")
            return F(x.bits ^ 0x80000000)
        if op in ("OpFAdd", "OpFSub", "OpFMul", "OpFDiv"):
            if m.types.get(a[0], ("",))[0] != "float":
                raise UnsupportedConstruct(f"{op} with result type {a[0]}")
            x, y = _f32_unpack(self.val(lane, a[1])), _f32_unpack(self.val(lane, a[2]))
            if op == "OpFAdd":
                return _f32_pack(x + y)
            if op == "OpFSub":
                return _f32_pack(x - y)
            if op == "OpFMul":
                return _f32_pack(x * y)
            if y == 0.0:
                if x == 0.0 or math.isnan(x):
                    return F(0x7FC00000)
                neg = (math.copysign(1.0, x) * math.copysign(1.0, y)) < 0
                return F(0xFF800000 if neg else 0x7F800000)
            return _f32_pack(x / y)
        if op in ("OpFOrdEqual", "OpFOrdNotEqual", "OpFOrdLessThan", "OpFOrdGreaterThan",
                  "OpFOrdLessThanEqual", "OpFOrdGreaterThanEqual",
                  "OpFUnordEqual", "OpFUnordNotEqual", "OpFUnordLessThan", "OpFUnordGreaterThan",
                  "OpFUnordLessThanEqual", "OpFUnordGreaterThanEqual"):
            x, y = _f32_unpack(self.val(lane, a[1])), _f32_unpack(self.val(lane, a[2]))
            if math.isnan(x) or math.isnan(y):
                return op.startswith("OpFUnord")
            cmp_name = op.removeprefix("OpFUnord").removeprefix("OpFOrd")
            return {"Equal": x == y, "NotEqual": x != y, "LessThan": x < y, "GreaterThan": x > y,
                    "LessThanEqual": x <= y, "GreaterThanEqual": x >= y}[cmp_name]
        if op in ("OpConvertFToU", "OpConvertFToS"):
            if m.types.get(a[0], ("",))[0] != "int":
                raise UnsupportedConstruct(f"{op} with result type {a[0]}")
            bits = m.bits(a[0])
            x = _f32_unpack(self.val(lane, a[1]))
            if math.isnan(x) or not math.isfinite(x):
                return _Poison(f"{op} on {x}")
            t = int(x)
            lo, hi = (0, 1 << bits) if op == "OpConvertFToU" else (-(1 << (bits - 1)), 1 << (bits - 1))
            if not lo <= t < hi:
                return _Poison(f"{op} out of range: {t}")
            return t & ((1 << bits) - 1)
        if op in ("OpConvertUToF", "OpConvertSToF"):
            if m.types.get(a[0], ("",))[0] != "float":
                raise UnsupportedConstruct(f"{op} with result type {a[0]}")
            v = self.int_of(lane, a[1])
            if op == "OpConvertSToF":
                bw = m.bits(m.id_type.get(a[1], a[0])) if m.types.get(m.id_type.get(a[1]), ("",))[0] == "int" else 32
                v = _signed(v, bw)
            return _f32_pack(float(v))
        if op.startswith("OpF") or op == "OpExtInst" and not self._int_ext(a):
            raise UnsupportedConstruct(f"{op}: float arithmetic is not modelled here")
        if op == "OpExtInst":
            return self.ext(lane, a)
        bits = m.bits(a[0]) if m.types.get(a[0], ("",))[0] == "int" else None
        w = (1 << bits) - 1 if bits else None
        if op in ("OpIEqual", "OpINotEqual", "OpULessThan", "OpULessThanEqual", "OpUGreaterThan",
                  "OpUGreaterThanEqual", "OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan", "OpSGreaterThanEqual"):
            x, y = self.int_of(lane, a[1]), self.int_of(lane, a[2])
            if op.startswith("OpS"):
                bw = m.bits(m.id_type.get(a[1], a[0])) if m.types.get(m.id_type.get(a[1]), ("",))[0] == "int" else 32
                x, y = _signed(x, bw), _signed(y, bw)
            name = op[3:] if op.startswith(("OpU", "OpS")) else op[2:]
            return {"IEqual": x == y, "INotEqual": x != y, "LessThan": x < y, "LessThanEqual": x <= y,
                    "GreaterThan": x > y, "GreaterThanEqual": x >= y}[name]
        if w is None and op not in ("OpIAddCarry", "OpISubBorrow", "OpUMulExtended", "OpSMulExtended"):
            raise UnsupportedConstruct(f"{op} with result type {a[0]}")
        if op in ("OpIAdd", "OpISub", "OpIMul", "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor"):
            x, y = self.int_of(lane, a[1]), self.int_of(lane, a[2])
            return {"OpIAdd": x + y, "OpISub": x - y, "OpIMul": x * y, "OpBitwiseAnd": x & y,
                    "OpBitwiseOr": x | y, "OpBitwiseXor": x ^ y}[op] & w
        if op in ("OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic"):
            x, s = self.int_of(lane, a[1]), self.int_of(lane, a[2])
            if s >= bits:
                return _Poison(f"{op} by {s} >= {bits}")
            if op == "OpShiftLeftLogical":
                return (x << s) & w
            if op == "OpShiftRightLogical":
                return (x & w) >> s
            return (_signed(x, bits) >> s) & w
        if op == "OpNot":
            return ~self.int_of(lane, a[1]) & w
        if op == "OpSNegate":
            return -self.int_of(lane, a[1]) & w
        if op in ("OpUConvert", "OpSConvert"):
            x = self.int_of(lane, a[1])
            if op == "OpSConvert":
                src_bits = m.bits(m.id_type[a[1]]) if a[1] in m.id_type else 32
                x = _signed(x, src_bits)
            return x & w
        if op == "OpBitCount":
            # Every bit of the operand counts: its width, not 32 (a 64-bit operand with a 32-bit result).
            src_bits = m.bits(m.id_type[a[1]]) if a[1] in m.id_type else 32
            return (self.int_of(lane, a[1]) & ((1 << src_bits) - 1)).bit_count()
        if op == "OpBitReverse":
            return int(f"{self.int_of(lane, a[1]) & w:0{bits}b}"[::-1], 2)
        if op in ("OpUDiv", "OpUMod"):
            x, y = self.int_of(lane, a[1]), self.int_of(lane, a[2])
            if y == 0:
                return _Poison(f"{op} by zero")
            return (x // y if op == "OpUDiv" else x % y) & w
        if op in ("OpBitFieldUExtract", "OpBitFieldSExtract"):
            base, off, cnt = (self.int_of(lane, x) for x in a[1:4])
            if off + cnt > bits:
                return _Poison("bit-field extract beyond the width")
            field = (base >> off) & ((1 << cnt) - 1)
            if op == "OpBitFieldSExtract" and cnt and field >> (cnt - 1):
                field -= 1 << cnt
            return field & w
        if op == "OpBitFieldInsert":
            base, ins, off, cnt = (self.int_of(lane, x) for x in a[1:5])
            if off + cnt > bits:
                return _Poison("bit-field insert beyond the width")
            mk = ((1 << cnt) - 1) << off
            return ((base & ~mk) | ((ins << off) & mk)) & w
        if op in ("OpIAddCarry", "OpISubBorrow"):
            x, y = self.int_of(lane, a[1]) & MASK, self.int_of(lane, a[2]) & MASK
            if op == "OpIAddCarry":
                return ((x + y) & MASK, (x + y) >> 32)
            return ((x - y) & MASK, 1 if x < y else 0)
        if op in ("OpUMulExtended", "OpSMulExtended"):
            x, y = self.int_of(lane, a[1]) & MASK, self.int_of(lane, a[2]) & MASK
            if op == "OpSMulExtended":
                x, y = _signed(x, 32), _signed(y, 32)
            p = (x * y) & 0xFFFFFFFFFFFFFFFF
            return (p & MASK, p >> 32)
        raise UnsupportedConstruct(op)

    @staticmethod
    def _int_ext(a):
        return len(a) >= 3 and a[2] in ("UMin", "UMax", "SMin", "SMax", "SAbs", "FindILsb", "FindUMsb",
                                         "FindSMsb", "UClamp", "SClamp")

    def ext(self, lane, a):
        if a[1] != "%glsl":
            raise UnsupportedConstruct(f"extended instruction set {a[1]}")
        name = a[2]
        args = [self.int_of(lane, x) for x in a[3:]]
        bits = self.m.bits(a[0])        # the operands have the result's type (GLSL.std.450)
        if name in ("SAbs", "SMin", "SMax", "SClamp", "FindILsb", "FindSMsb", "FindUMsb"):
            for operand in a[3:]:
                if operand not in self.m.id_type:
                    raise UnsupportedConstruct(f"{name}: unknown integer type for {operand}")
                width = self.m.bits(self.m.id_type[operand])
                if width != bits:
                    raise UnsupportedConstruct(f"{name}: {width}-bit operand {operand} with a {bits}-bit result")
        s = lambda v: _signed(v, bits)  # noqa: E731
        if name == "UMin":
            return min(args)
        if name == "UMax":
            return max(args)
        if name == "SMin":
            return min(args, key=s)
        if name == "SMax":
            return max(args, key=s)
        if name == "SAbs":
            return abs(s(args[0])) & ((1 << bits) - 1)
        if name in ("UClamp", "SClamp"):
            key = s if name == "SClamp" else (lambda v: v)
            x, lo, hi = args
            if key(lo) > key(hi):
                raise UnsupportedConstruct(f"{name} with minVal > maxVal (undefined)")
            return min(max(x, lo, key=key), hi, key=key)
        if name in ("FindILsb", "FindUMsb", "FindSMsb") and bits != 32:
            raise UnsupportedConstruct(f"GLSL.std.450 {name} on a {bits}-bit integer (limited to 32-bit components)")
        if name == "FindILsb":
            x = args[0] & MASK
            return ((x & -x).bit_length() - 1) & MASK if x else MASK
        if name == "FindUMsb":
            x = args[0] & MASK
            return x.bit_length() - 1 if x else MASK
        if name == "FindSMsb":
            x = s(args[0])
            if x in (0, -1):
                return MASK
            return x.bit_length() - 1 if x > 0 else (~x).bit_length() - 1
        raise UnsupportedConstruct(f"GLSL.std.450 {name}")


class _BuiltinNotModelled:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


def run_wave(text, wave=32, push_constants=None, max_steps=2_000_000, builtins=None, inputs=None, memory=None):
    if wave not in (32, 64):
        raise ValueError("wave must be 32 or 64")
    w = _Wave(_Module(text), wave, push_constants, max_steps, builtins, inputs=inputs, memory=memory)
    outputs = w.run()
    return {"outputs": outputs, "steps": w.steps, "memory": w.memory}
