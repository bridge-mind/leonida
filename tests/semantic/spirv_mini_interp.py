"""A deliberately tiny interpreter for the SPIR-V assembly that tools/rdna2spirv.py emits for branch-free pixel
shaders and simple structured compute shaders. It exists so tests can check *computed values*, not just that the
module validates.

Design rules (see tests/semantic/README.md):
  * Written from the SPIR-V and GLSL.std.450 specifications. It shares no code with the translator.
  * Strict: any opcode, extended instruction, type or storage class not listed below raises
    UnsupportedConstruct. Nothing is skipped or approximated silently, so a translator change that starts
    emitting something new makes the test error out instead of passing.
  * One invocation (one lane), or with run_workgroup several in lockstep, which meet only at
    subgroup operations (Shuffle, Ballot, Any, All, BitwiseOr Reduce; BallotBitExtract and BallotFindLSB
    are per lane). Control flow is followed one branch at a time: OpBranch / OpBranchConditional to a label
    of the function (forward, or backward for a loop, up to 10000 backward branches), OpPhi (the phis of a
    block take their values together) and OpFunctionCall of a function of the module.
    Switches are rejected. Function/Output/Input variables live in Python
    boxes; PushConstant reads can be seeded with a word list; physical-storage-buffer loads/stores (and 32-bit
    atomics, which with one invocation are read-combine-write) use a flat address -> word dictionary, so they
    must be 32-bit integers at 4-byte-aligned addresses (anything else raises). Workgroup variables (LDS) are
    zero-initialised; run_workgroup shares them between its invocations, as a workgroup does (one invocation
    alone has its own). 32-bit atomics on them (and on Function variables) are read-combine-write like the
    physical-storage-buffer ones. OpControlBarrier is a point every invocation of run_workgroup must reach
    together (run_fragment refuses it). Images are rejected, except OpImageFetch on an image array that
    run_workgroup's `resources` binds (with the StorageBuffer descriptor hash table the translator probes).
  * A default run sees zero push constants, no inputs and empty memory, so tests that do not care about them
    keep the old behaviour.

Values: uint/int are Python ints in [0, 2**32) or [0, 2**64) for 64-bit types (signedness is applied by the
operation, as in SPIR-V); float is a Python float rounded to binary32 (one made by OpBitcast also keeps its exact
32-bit pattern, so a signalling NaN comes back out unchanged). Binary64 supports the word-pair bitcasts,
round-to-integral operations and their result modifiers. Vectors are tuples; arrays are lists; bool is bool.
"""
import math
import re
import struct
from fractions import Fraction

MASK = 0xFFFFFFFF


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


class InvalidModule(AssertionError):
    """The module breaks a SPIR-V structural rule spirv-val enforces, so no driver would accept it, even though
    this interpreter could run it. Raised instead of returning results for a shader that never reaches a GPU."""


def check_structure(body):
    """spirv-val's merge, phi and Function-variable rules (SPIR-V 2.16.1-2): OpSelectionMerge must be the second-to-last instruction of its block,
    directly before OpBranchConditional or OpSwitch, and OpLoopMerge directly before OpBranch or
    OpBranchConditional. Until 2026-11-22 the interpreter skipped merges, so a translator change that put an
    instruction between a merge and its branch (#1633: OpLoad %active) passed every semantic test while every
    driver rejected the module. `body` is the list of (result, opcode, operands, line) of a function."""
    follow = {"OpSelectionMerge": ("OpBranchConditional", "OpSwitch"),
              "OpLoopMerge": ("OpBranch", "OpBranchConditional")}
    # OpPhi must come first in its block: only OpLabel or other OpPhi before it (SPIR-V 2.16.2, "OpPhi ... must be
    # the first instructions of a block"). A phi emitted after an instruction of the merge block is rejected by
    # spirv-val, and every translator path that merges a guarded load (OpPhi of the loaded and the kept value)
    # depends on the order.
    # Function-storage OpVariable belongs in the function's first block (SPIR-V 2.16.1).
    prev, blocks = None, 0
    for _, op, a, line in body:
        if op == "OpPhi" and prev not in ("OpLabel", "OpPhi"):
            raise InvalidModule(f"{line!r} follows {prev or 'the start of the function'}, not OpLabel or OpPhi")
        if op == "OpFunction":
            blocks = 0
        elif op == "OpLabel":
            blocks += 1
        elif op == "OpVariable" and a[1:2] == ["Function"] and blocks > 1:
            raise InvalidModule(f"{line!r} is a Function variable outside the function's first block")
        prev = op
    for k, (_, op, _, line) in enumerate(body):
        if op in follow:
            nxt = body[k + 1][1] if k + 1 < len(body) else None
            if nxt not in follow[op]:
                raise InvalidModule(f"{line!r} is followed by {body[k + 1][3] if nxt else 'the end of the function'!r}"
                                    f", not by {' or '.join(follow[op])}")


# Instructions that carry no runtime semantics for a single-entry-point, straight-line module.
_IGNORED = frozenset({
    "OpCapability", "OpExtension", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel", "OpSelectionMerge",
    "OpLoopMerge", "OpReturn", "OpFunctionEnd", "OpName", "OpSource",
})

_ATOMICS = frozenset({
    "OpAtomicIAdd", "OpAtomicISub", "OpAtomicUMin", "OpAtomicUMax", "OpAtomicSMin", "OpAtomicSMax", "OpAtomicAnd",
    "OpAtomicOr", "OpAtomicXor", "OpAtomicLoad", "OpAtomicExchange", "OpAtomicCompareExchange",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


class _Undef:
    """An undefined value (SPIR-V: a shift by at least the bit width). Pure arithmetic on it gives another
    undefined value; any other use (a store, a branch or select condition, a subgroup operation, an output)
    raises, so a result that does not depend on it, such as the unselected side of OpSelect, is still exact."""

    def __repr__(self):
        return "<undefined>"


UNDEF = _Undef()

# Operations whose result is undefined when an operand is (they have no side effect and no control).
_PURE = frozenset({
    "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpNot", "OpShiftLeftLogical", "OpShiftRightLogical",
    "OpShiftRightArithmetic", "OpIAdd", "OpISub", "OpIMul", "OpUConvert", "OpBitCount", "OpIEqual",
    "OpINotEqual", "OpSLessThan", "OpULessThan", "OpUGreaterThan", "OpUGreaterThanEqual", "OpCopyObject",
    "OpBitcast",
    "OpBitFieldUExtract", "OpBitFieldSExtract", "OpBitFieldInsert",
})


def f32(x):
    """Round a Python float to binary32 (overflow to infinity)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def ieee_nan(r, *operands):
    """`r`, except that a NaN produced from non-NaN operands (inf - inf, 0 * inf) is the x86 "real indefinite" NaN.

    The measured words pinned in the tests were taken with the default NaN of x86 SSE, 0xFFC00000 (sign set).
    Python leaves the sign of such a NaN to the host CPU, and ARM64 (Apple Silicon, Linux on ARM) gives
    0x7FC00000, so the tests passed on x86-64 only. A NaN that comes from a NaN operand is left alone: both
    architectures pass that operand through with its sign."""
    if r != r and not any(o != o for o in operands):
        return -math.nan
    return r


class _F32(float):
    """A binary32 value that came from a bit pattern (OpBitcast) and still carries that exact pattern.

    struct's binary32 -> double conversion quiets a signalling NaN (it sets bit 22, issue #509), so the float
    value alone cannot give the word back. OpBitcast, OpCopyObject, loads, stores, composites and outputs pass
    the value on unchanged and keep `bits`; arithmetic on it returns a plain float, which drops them (SPIR-V
    leaves the bits of a NaN result open)."""

    __slots__ = ("bits",)

    def __new__(cls, bits):
        self = super().__new__(cls, struct.unpack("<f", struct.pack("<I", bits))[0])
        self.bits = bits
        return self


def fma_f32(x, y, z):
    """x * y + z with one rounding to binary32 (round to nearest even), as GLSL.std.450 Fma on IEEE hardware.

    The product and sum are exact rationals; rounding them through a double first could land on a binary32
    midpoint and round the wrong way (for example (1 + 2**-12)**2 + 2**-70)."""
    if not all(math.isfinite(v) for v in (x, y, z)):
        return f32(ieee_nan(x * y + z, x, y, z))   # inf/NaN rules: Python's float arithmetic follows IEEE here
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    if exact == 0:
        product_zero = x == 0.0 or y == 0.0
        negative = product_zero and math.copysign(1.0, x) * math.copysign(1.0, y) < 0 and math.copysign(1.0, z) < 0
        return -0.0 if negative else 0.0
    a = abs(exact)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)                          # below the normal range the step is 2**-149
    q = a / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    r = math.ldexp(n, e - 23)
    if r >= 2.0 ** 128:
        r = math.inf
    return f32(math.copysign(r, exact))


def f32_bits(x):
    if isinstance(x, _F32):
        return x.bits
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_f32(b):
    return _F32(b & MASK)


def half_to_float(h):
    # Bit-level f16 -> f32 (sign, exponent, fraction; every result is exact in binary32). struct's '<e' is not
    # used because its NaN result depends on the Python version: before 3.14 it drops the payload and returns
    # the canonical quiet NaN, from 3.14 it keeps sign and payload (CPython gh-130317). Chosen rule, the one
    # CI's 3.12 and every pinned expected word use: a NaN half expands to the canonical quiet NaN with its
    # sign, payload dropped. The binary32 round trip below only widens the bit pattern, which no version changes.
    h &= 0xFFFF
    sign = (h >> 15) << 31
    exp = (h >> 10) & 0x1F
    frac = h & 0x03FF
    if exp == 0x1F:
        bits = sign | (0x7FC00000 if frac else 0x7F800000)     # NaN: canonical quiet, payload dropped; infinity
    elif exp == 0 and frac:
        b = frac.bit_length()
        bits = sign | ((b + 102) << 23) | ((frac << (24 - b)) & 0x7FFFFF)   # subnormal half -> normal f32
    elif exp == 0:
        bits = sign                                            # +-0
    else:
        bits = sign | ((exp + 112) << 23) | (frac << 13)       # normal half -> f32, exact
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def float_to_half(x):
    # Bit-level f32 -> f16, round to nearest even, with the same Python-version pin as half_to_float above:
    # any NaN becomes the canonical quiet NaN with the operand's sign, payload dropped. Rounding runs on the
    # binary32 bit pattern of x (the interpreter's value model keeps floats rounded to binary32), so the result
    # cannot change with the Python version: RNE at normal and subnormal midpoints, subnormals kept, overflow
    # to infinity.
    try:
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00
    sign = bits >> 31
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        return (sign << 15) | (0x7E00 if frac else 0x7C00)     # NaN: canonical quiet, sign kept; infinity
    e = exp - 127
    if e < -14:
        # Subnormal half: value = full * 2**(e - 23), so n = full * 2**(e + 1) rounded to nearest even.
        full = (0x800000 | frac) if exp else frac
        drop = -(e + 1)
        n = full >> drop
        rem = full & ((1 << drop) - 1)
        if rem > (1 << (drop - 1)) or (rem == (1 << (drop - 1)) and n & 1):
            n += 1
        if n > 0x3FF:                                          # rounds up to the smallest normal
            return (sign << 15) | 0x0400
        return (sign << 15) | n
    m = frac >> 13                                             # keep the top 10 fraction bits, RNE the rest
    rem = frac & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and m & 1):
        m += 1
    e += m >> 10                                               # mantissa overflow bumps the exponent
    if e > 15:                                                 # overflow, incl. the 65520 RNE tie
        return (sign << 15) | 0x7C00
    return (sign << 15) | ((e + 15) << 10) | (m & 0x3FF)


def _s32(x):
    x &= MASK
    return x - (1 << 32) if x >> 31 else x


def _fraction_to_f32(q):
    """Round an exact non-negative Fraction below 2**128 to binary32: nearest, ties to even (subnormals too)."""
    if q == 0:
        return 0.0
    e = q.numerator.bit_length() - q.denominator.bit_length()     # 2**e <= q < 2**(e+1) after the fix-up
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)                                              # below 2**-126 the step stays 2**-149
    m = q * Fraction(2) ** (23 - e)                               # significand in units of the last place
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    return f32(n * 2.0 ** (e - 23))                               # exact: n <= 2**24


def _fract(x):
    """GLSL.std.450 Fract: x - floor(x), here computed exactly and rounded once to binary32, so a tiny negative x
    gives 1.0 (the nearest float to 1 - |x|). NaN stays NaN; +-inf gives NaN (inf - inf)."""
    if math.isnan(x) or math.isinf(x):
        return math.nan
    q = Fraction(x)
    return _fraction_to_f32(q - math.floor(q))


def _nmin_nmax(name, x, y):
    """GLSL.std.450 NMin / NMax: a NaN operand gives the other operand; both NaN give a NaN. Zeros are ordered
    -0 < +0, the order RDNA2 v_min_f32/v_max_f32 use (measured on an RX 6800, see docs/design.md)."""
    if math.isnan(x):
        return y if not math.isnan(y) else math.nan
    if math.isnan(y):
        return x
    kx, ky = (x, math.copysign(1.0, x)), (y, math.copysign(1.0, y))
    if name == "NMin":
        return y if ky < kx else x
    return y if ky > kx else x


class _Types:
    def __init__(self):
        self.t = {}

    def zero(self, name):
        k = self.t[name]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return 0.0
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(self.zero(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [self.zero(k[1]) for _ in range(k[2])]
        raise UnsupportedConstruct(f"zero value of type {name} ({k[0]})")

    def size(self, name):
        """Byte size of a concrete type. Runtime arrays have no size of their own; the element size is used."""
        k = self.t[name]
        if k[0] in ("int", "float"):
            return k[1] // 8
        if k[0] == "vec":
            return self.size(k[1]) * k[2]
        if k[0] == "arr":
            return self.size(k[1]) * k[2]
        if k[0] == "rarr":
            return self.size(k[1])
        if k[0] == "struct":
            return sum(self.size(t) for t in k[1])
        raise UnsupportedConstruct(f"size of type {name} ({k[0]})")

    def copy_value(self, name, value):
        """Copy an aggregate across the memory boundary, retaining any pointer values it contains."""
        kind = self.t.get(name)
        if kind is None:
            return value
        if kind[0] == "arr":
            members = (self.copy_value(kind[1], item) for item in value)
        elif kind[0] == "struct":
            members = (self.copy_value(member, item) for member, item in zip(kind[1], value))
        elif kind[0] == "vec" and isinstance(value, list):
            return value.copy()
        else:
            return value
        return type(value)(members)

    def is_word_pair(self, kind):
        return kind and kind[0] == "vec" and kind[2] == 2 and self.t[kind[1]][:2] == ("int", 32)

    def bits(self, name):
        k = self.t[name]
        if k[0] != "int":
            raise UnsupportedConstruct(f"integer width of type {name} ({k[0]})")
        return k[1]


def run_fragment(text, inputs=None, push_constants=None, memory=None):
    """Execute the entry point of `text` once and return the words written to every Output variable.

    `inputs` maps an Input variable name (for example "%in_attr1") to its value (a tuple for vectors).
    `push_constants` is a list of 32-bit words at byte offsets 0, 4, 8, ... of the PushConstant block
    (missing words read as zero). `memory` is a mutable {byte_address: word} mapping used for
    PhysicalStorageBuffer loads and stores.

    Returns {output_variable_name: tuple_of_32bit_words}. Float outputs are returned as their bit patterns.
    A module with a subgroup operation is rejected here; see run_workgroup.
    """
    gen = _invocation(text, inputs, push_constants, memory)
    try:
        req = next(gen)
    except StopIteration as done:
        return done.value
    gen.close()
    raise UnsupportedConstruct(f"{req[0]} in a single invocation (use run_workgroup)")


class UndefinedResult(Exception):
    """A subgroup operation whose result SPIR-V leaves undefined (for example a shuffle from a lane that is not in
    the subgroup). Raised instead of inventing a value."""


def run_workgroup(text, invocations, subgroup_size, inputs_for=None, push_constants=None, memory=None,
                  out_of_range="raise", return_outputs=False, lanes=None, resources=None, helpers=()):
    """Execute `invocations` invocations of `text` in lockstep and return the shared `memory`.

    Invocation i is subgroup i // subgroup_size, lane i % subgroup_size (a 1-D workgroup split into subgroups in
    invocation order). When `invocations` is not a multiple of `subgroup_size` the last subgroup is partial: its
    missing lanes are not running. `inputs_for(i)` gives invocation i's Input values (see run_fragment); push
    constants and the physical-address memory are shared. Every invocation must reach every subgroup operation,
    in the same order (the translator emits them in uniform control flow and models EXEC with a variable, so a
    lane that is off in EXEC still executes them). Workgroup variables (LDS) are shared by all the invocations,
    and OpControlBarrier, like a subgroup operation, must be reached by every invocation at the same point.
    Invocations at the same instruction execute it in invocation order, so lanes racing on one LDS dword see a
    defined interleaving (see the lockstep note below).

    Subgroup operations (Subgroup scope), over the running lanes of the caller's subgroup:
      * OpGroupNonUniformShuffle returns the value of lane Id. An Id at or past `subgroup_size` is undefined in
        SPIR-V: `out_of_range="raise"` raises UndefinedResult; `out_of_range="wrap"` takes lane Id mod
        subgroup_size instead, a model of one observed driver behaviour and not of the specification (Leonida
        issue #94: on a 32-wide NVIDIA subgroup lane 40 read lane 8). An Id inside the subgroup whose lane is not
        running gives an undefined value (raises only if used, see _Undef).
      * OpGroupNonUniformBallot sets bit l for each running lane l whose predicate is true.
      * OpGroupNonUniformAny / All: the predicate of any / every running lane.
      * OpGroupNonUniformBitwiseOr Reduce: the OR of every running lane's value.
      * OpGroupNonUniformBroadcastFirst: the value of the lowest running lane.

    With `return_outputs=True`, also return each invocation's Output variables in invocation order.

    `lanes` places the invocations: invocation i is lane lanes[i] % subgroup_size of subgroup
    lanes[i] // subgroup_size (default: lanes[i] = i). A lane no invocation takes is not running, like a pixel
    lane with no covered sample, so a shuffle from it is undefined. The caller's `inputs_for` gives each
    invocation the %sg_lane that matches. `resources` binds descriptor-set variables (see _invocation).

    `helpers` names invocations that run the shader but take no part in subgroup operations, as helper lanes a
    driver leaves out: their ballot bit is never set, they count for no vote or reduction, and a shuffle from their
    lane is undefined. They still receive each operation's result over the others.
    """
    if out_of_range not in ("raise", "wrap"):
        raise ValueError(f"out_of_range: {out_of_range!r}")
    memory = {} if memory is None else memory
    workgroup = {}                             # Workgroup variable id -> its box, one per workgroup
    slots = list(range(invocations)) if lanes is None else list(lanes)
    helpers = frozenset(helpers)
    if len(slots) != invocations or len(set(slots)) != invocations or min(slots, default=0) < 0:
        raise ValueError("lanes: one distinct non-negative slot per invocation")
    gens = [_invocation(text, (inputs_for(i) if inputs_for else None), push_constants, memory, step=True,
                        workgroup=workgroup, resources=resources)
            for i in range(invocations)]
    pending = [None] * invocations             # what each invocation yielded last; None once it has returned
    invocation_outputs = [None] * invocations

    def advance(i, value):
        try:
            pending[i] = gens[i].send(value)
        except StopIteration as done:
            pending[i] = None
            invocation_outputs[i] = done.value

    for i in range(invocations):
        advance(i, None)
    while True:
        # One instruction per invocation per round, in invocation order: invocations at the same instruction
        # execute it one after the other before any goes on, as the lanes of a SIMD wave do, so a load that
        # follows a store in program order sees every lane's store and none from a later instruction.
        while any(req is not None and req[0] == "tick" for req in pending):
            for i in range(invocations):
                if pending[i] is not None and pending[i][0] == "tick":
                    advance(i, None)
        live = [i for i in range(invocations) if pending[i] is not None]
        if not live:
            break
        if len(live) != invocations or len({pending[i][1] for i in live}) != 1:
            raise UnsupportedConstruct("subgroup operation not reached by every invocation at the same point")
        results = []
        for i in range(invocations):
            kind, _, _value, ident = pending[i]
            sub = slots[i] // subgroup_size
            group = {slots[j] % subgroup_size: pending[j] for j in range(invocations)
                     if slots[j] // subgroup_size == sub and j not in helpers}
            if kind == "OpGroupNonUniformBallot":
                # Bit l of the 128-bit mask = lane l of this subgroup is running with a true predicate.
                bits = sum(1 << l for l, req in group.items() if req[2])
                results.append(tuple((bits >> (32 * w)) & MASK for w in range(4)))
            elif kind == "OpGroupNonUniformAny":
                results.append(any(req[2] for req in group.values()))
            elif kind == "OpGroupNonUniformAll":
                results.append(all(req[2] for req in group.values()))
            elif kind == "OpGroupNonUniformBitwiseOr":
                acc = 0
                for req in group.values():
                    acc |= req[2]
                results.append(acc)
            elif kind == "OpControlBarrier":
                results.append(None)           # everyone is here: the "same point" check above is the barrier
            elif kind == "OpGroupNonUniformBroadcastFirst":
                results.append(group[min(group)][2] if group else UNDEF)   # the lowest running lane's value
            elif kind == "OpGroupNonUniformShuffle":
                if not 0 <= ident < subgroup_size:
                    if out_of_range == "raise":
                        raise UndefinedResult(f"invocation {i}: shuffle from lane {ident} of a "
                                              f"{subgroup_size}-wide subgroup")
                    ident %= subgroup_size
                results.append(group[ident][2] if ident in group else UNDEF)
            else:
                raise UnsupportedConstruct(kind)
        for i in range(invocations):
            advance(i, results[i])
    return (memory, invocation_outputs) if return_outputs else memory


_TICK = ("tick", None, None, None)
_PARSED = {}


def _parsed_lines(text):
    """The (result, opcode, operands, line) of each non-blank line of `text`, parsed once: run_workgroup runs one
    generator per invocation over the same module, and parsing it for every one of them dominated the run time.
    A line that does not parse has opcode None; the caller raises where the line is reached, as before."""
    parsed = _PARSED.get(text)
    if parsed is None:
        parsed = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if m:
                res, op, rest = m.groups()
                parsed.append((res, op, rest.split(), line))
            else:
                parsed.append((None, None, None, line))
        if len(_PARSED) >= 64:
            _PARSED.clear()
        _PARSED[text] = parsed
    return parsed


def _invocation(text, inputs, push_constants, memory, step=False, workgroup=None, resources=None):
    """One invocation as a generator. It yields (opcode, instruction index, value, id) at each subgroup operation
    and at each OpControlBarrier, and expects that operation's result back; with `step` it also yields _TICK before
    each instruction. `workgroup` (run_workgroup's) maps a Workgroup variable id to the box all invocations
    share; without it the invocation has its own. It returns run_fragment's result.

    `resources` binds descriptor-set variables by name: a StorageBuffer variable whose block is one runtime array
    of words takes a list of those words; a UniformConstant array of images takes a list with one entry per
    descriptor index, each a function (coordinate, lod) -> the 4-tuple OpImageFetch returns. A descriptor variable
    left out stays declared but cannot be accessed (OpAccessChain into it raises)."""
    inputs = inputs or {}
    resources = resources or {}
    push_constants = list(push_constants or [])
    memory = {} if memory is None else memory
    types = _Types()
    values = {}                 # id -> value (constants and results)
    constants = {}              # id -> value, kept apart so zero-initialised arrays can be built
    ptr_types = {}              # pointer type id -> (storage class, pointee type id)
    var_class = {}              # variable id -> storage class
    var_box = {}                # variable id -> [value]
    pointers = {}               # id -> (container list, index) / ("pc", offset) / ("psb", address)
    id_type = {}                # constant / result id -> its type id (the width of a signed comparison)
    pc_type = {}                # id -> pointee type id of a PushConstant pointer
    in_body = False
    first_fn = None
    body = []
    outputs = []

    def val(x):
        if x in values:
            return values[x]
        raise UnsupportedConstruct(f"use of undefined id {x}")

    def walk(type_id, indices):
        """Byte offset and element type after selecting `indices` in an aggregate type."""
        off, cur = 0, type_id
        for idx in indices:
            k = types.t[cur]
            if k[0] == "struct":
                for m in range(idx):
                    off += types.size(k[1][m])
                cur = k[1][idx]
            elif k[0] in ("arr", "rarr", "vec"):
                off += types.size(k[1]) * idx
                cur = k[1]
            else:
                raise UnsupportedConstruct(f"OpAccessChain into {k[0]}")
        return off, cur

    for res, op, a, line in _parsed_lines(text):
        if op is None:
            raise UnsupportedConstruct(f"cannot parse line: {line!r}")
        if in_body:
            body.append((res, op, a, line))
            continue
        if op == "OpFunction":
            in_body = True
            first_fn = res
            continue
        if op in _IGNORED:
            continue
        if op == "OpTypeInt":
            bits = int(a[0])
            if bits not in (32, 64):
                raise UnsupportedConstruct("integer type that is not 32 or 64 bits")
            types.t[res] = ("int", bits, int(a[1]))
        elif op == "OpTypeFloat":
            bits = int(a[0])
            if bits not in (32, 64):
                raise UnsupportedConstruct("float type that is not 32 or 64 bits")
            types.t[res] = ("float", bits)
        elif op == "OpTypeBool":
            types.t[res] = ("bool",)
        elif op == "OpTypeVector":
            types.t[res] = ("vec", a[0], int(a[1]))
        elif op == "OpTypeArray":
            types.t[res] = ("arr", a[0], int(constants[a[1]]))
        elif op == "OpTypeRuntimeArray":
            types.t[res] = ("rarr", a[0])
        elif op == "OpTypeStruct":
            types.t[res] = ("struct", a)
        elif op == "OpTypePointer":
            ptr_types[res] = (a[0], a[1])
        elif op in ("OpTypeImage", "OpTypeSampler", "OpTypeSampledImage"):
            types.t[res] = ("image",)                  # opaque: only OpImageFetch on a bound resource reads it
        elif op == "OpConstant":
            k = types.t[a[0]]
            if k[0] == "float":
                constants[res] = values[res] = f32(float(a[1])) if k[1] == 32 else float(a[1])
            elif k[0] == "int":
                constants[res] = values[res] = int(a[1]) & ((1 << k[1]) - 1)
                id_type[res] = a[0]
            else:
                raise UnsupportedConstruct(f"OpConstant of type {k[0]}")
        elif op == "OpConstantTrue":
            constants[res] = values[res] = True
        elif op == "OpConstantFalse":
            constants[res] = values[res] = False
        elif op == "OpConstantNull":
            constants[res] = values[res] = types.zero(a[0])
            id_type[res] = a[0]
        elif op == "OpConstantComposite":           # a constant array or vector (e.g. %fmt_size_tab's table)
            values[res] = [values[x] for x in a[1:]]
            id_type[res] = a[0]
        elif op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            if a[1] != cls:
                raise UnsupportedConstruct("OpVariable storage class mismatch")
            var_class[res] = cls
            if cls == "PushConstant":
                var_box[res] = [None]                  # reads come from push_constants, see OpLoad
                pc_type[res] = pointee
            elif cls == "Input":
                init = inputs.get(res, types.zero(pointee))
                var_box[res] = [list(init) if isinstance(init, (list, tuple)) else init]
            elif cls == "Workgroup" and workgroup is not None and res in workgroup:
                var_box[res] = workgroup[res]          # another invocation of the workgroup made it
            elif cls in ("Function", "Output", "Workgroup", "Private"):
                init = types.copy_value(pointee, values[a[2]]) if len(a) > 2 else types.zero(pointee)
                var_box[res] = [init]
                if cls == "Output":
                    outputs.append(res)
                if cls == "Workgroup" and workgroup is not None:
                    workgroup[res] = var_box[res]      # the first invocation's box is the workgroup's
            elif cls == "StorageBuffer":
                if res not in resources:
                    raise UnsupportedConstruct(f"StorageBuffer {res} without a resource")
                var_box[res] = [[[w & MASK for w in resources[res]]]]   # block { uint[] }
            elif cls == "UniformConstant":
                var_box[res] = [list(resources[res]) if res in resources else None]
            else:
                raise UnsupportedConstruct(f"storage class {cls}")
            pointers[res] = ("pc", 0) if cls == "PushConstant" else (var_box[res], 0)
        else:
            raise UnsupportedConstruct(f"{op} in module header")
    if not in_body:
        raise UnsupportedConstruct("no function body")
    check_structure(body)

    def deref(p):
        box, i = p
        return box[i]

    def psb_word(addr):
        """The flat memory holds one 32-bit word per 4-byte-aligned byte address. Another address is undefined
        under the translator's `Aligned 4` and would read a key nobody wrote, so it raises."""
        if addr % 4:
            raise UnsupportedConstruct(f"PhysicalStorageBuffer access at {addr:#x}, not 4-byte aligned")
        return addr

    def wmask(type_id):
        return (1 << types.bits(type_id)) - 1

    def signed(v, bits):
        v &= (1 << bits) - 1
        return v - (1 << bits) if v >> (bits - 1) else v

    def sval(x):
        """An integer operand as signed, at the width of its type (a scalar int; 32 if the type is unknown)."""
        t = types.t.get(id_type.get(x))
        if t is not None and t[0] != "int":
            raise UnsupportedConstruct(f"signed comparison of a {t[0]}")
        return signed(val(x), t[1] if t else 32)

    labels = {res: idx for idx, (res, op, _, _) in enumerate(body) if op == "OpLabel" and res}
    # Every function's first body index and parameter ids. The first OpFunction was consumed by the header parse;
    # later ones are in the body. Execution starts at %main (the entry point the translator names).
    fn_start, fn_params = {first_fn: 0}, {}
    for idx, (res, op, _, _) in enumerate(body):
        if op == "OpFunction":
            fn_start[res] = idx + 1
    for fid, start in list(fn_start.items()):           # parameters come right after OpFunction
        params, i = [], start
        while i < len(body) and body[i][1] == "OpFunctionParameter":
            params.append(body[i][0])
            i += 1
        fn_params[fid], fn_start[fid] = params, i
    calls = []                                          # (return pc, result id, cur_label, prev_label)
    pc = fn_start.get("%main", 0)
    killed = False
    cur_label = prev_label = None
    phi_new = {}
    back_edges = 0
    while pc < len(body):
        if step:
            yield _TICK
        res, op, a, line = body[pc]
        pc += 1
        if res and a:
            id_type[res] = a[0]
        if op == "OpLabel":
            prev_label, cur_label = cur_label, res
            continue
        if op == "OpFunctionCall":
            fid, args = a[1], a[2:]
            if fid not in fn_start or len(args) != len(fn_params[fid]) or len(calls) > 16:
                raise UnsupportedConstruct(f"OpFunctionCall {fid}")
            for prm, arg in zip(fn_params[fid], args):
                values[prm] = val(arg)
                if arg in id_type:
                    id_type[prm] = id_type[arg]
            calls.append((pc, res, cur_label, prev_label))
            pc = fn_start[fid]
            continue
        if op == "OpReturnValue":
            if not calls:
                raise UnsupportedConstruct("OpReturnValue outside a called function")
            ret = val(a[0])
            pc, res_id, cur_label, prev_label = calls.pop()
            values[res_id] = ret
            continue
        if op == "OpReturn" and not calls:
            break
        if op in ("OpFunction", "OpFunctionParameter"):
            raise UnsupportedConstruct(f"{op} reached by falling through")
        if op in _IGNORED:
            continue
        if op == "OpKill":
            killed = True
            break
        if op == "OpBranch":
            if a[0] not in labels:
                raise UnsupportedConstruct(f"unknown branch target {a[0]}")
            if labels[a[0]] < pc:
                back_edges += 1
                if back_edges > 10000:
                    raise UnsupportedConstruct("loop iteration limit exceeded")
            pc = labels[a[0]]
            continue
        if op == "OpBranchConditional":
            cond = val(a[0])
            if not isinstance(cond, bool):
                raise UnsupportedConstruct("OpBranchConditional with a non-bool condition")
            target = a[1] if cond else a[2]
            if target not in labels:
                raise UnsupportedConstruct(f"unknown branch target {target}")
            if labels[target] < pc:
                back_edges += 1
                if back_edges > 10000:
                    raise UnsupportedConstruct("loop iteration limit exceeded")
            pc = labels[target]
            continue
        if op == "OpPhi":
            # The OpPhi of a block all read the values from before the block (a later one that names an earlier
            # one, as in a swap in a loop header, gets its old value), so the first evaluates the whole run.
            if not phi_new:
                k = pc - 1
                while k < len(body) and body[k][1] == "OpPhi":
                    phi, _, b, _ = body[k]
                    if phi in phi_new:
                        raise UnsupportedConstruct(f"duplicate OpPhi result {phi}")
                    incoming = [b[i] for i in range(1, len(b) - 1, 2) if b[i + 1] == prev_label]
                    if len(incoming) != 1:
                        raise UnsupportedConstruct(f"OpPhi {phi} needs one input from predecessor {prev_label}")
                    phi_new[phi] = val(incoming[0])
                    k += 1
            values[res] = phi_new.pop(res)
            continue
        if op in _PURE and any(values.get(x) is UNDEF for x in a[1:]):
            values[res] = UNDEF
            continue
        used = a[1:2] if op == "OpSelect" else a      # OpSelect may pass the chosen undefined value on
        if any(values.get(x) is UNDEF for x in used):
            raise UnsupportedConstruct(f"{op} uses an undefined value: {line}")
        if op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            if cls != "Function":
                raise UnsupportedConstruct("non-Function variable in body")
            init = types.copy_value(pointee, values[a[2]]) if len(a) > 2 else types.zero(pointee)
            var_box[res] = [init]
            pointers[res] = (var_box[res], 0)
        elif op == "OpAccessChain":
            base = a[1]
            if base not in pointers:
                raise UnsupportedConstruct("OpAccessChain on non-pointer")
            p = pointers[base]
            if p[0] == "pc":
                off, cur = walk(pc_type[base], [values[i] for i in a[2:]])
                pointers[res] = ("pc", p[1] + off)
                pc_type[res] = cur
                var_class[res] = "PushConstant"
                continue
            if p[0] == "psb":
                if len(a) != 3:
                    raise UnsupportedConstruct("multi-index physical-storage-buffer access chain")
                elem = ptr_types[a[0]][1]
                if types.t[elem][0] == "rarr":
                    stride = types.size(types.t[elem][1])
                elif types.t[elem][0] in ("int", "float", "vec"):
                    stride = types.size(elem)
                else:
                    raise UnsupportedConstruct(f"physical-storage-buffer access chain into {types.t[elem][0]}")
                pointers[res] = ("psb", p[1] + stride * values[a[2]])
                continue
            for idx_id in a[2:]:
                idx = values[idx_id]
                container = deref(p)
                if not isinstance(container, (list, tuple)) or not 0 <= idx < len(container):
                    raise UnsupportedConstruct("OpAccessChain into non-array or out of range")
                p = (container, idx)
            pointers[res] = p
        elif op == "OpLoad":
            p = pointers[a[1]]
            if p[0] == "pc":
                w = p[1] // 4
                values[res] = push_constants[w] if 0 <= w < len(push_constants) else 0
            elif p[0] == "psb":
                if types.t[a[0]][:2] != ("int", 32):
                    raise UnsupportedConstruct("PhysicalStorageBuffer load of a type that is not a 32-bit integer")
                values[res] = memory.get(psb_word(p[1]), 0)
            else:
                values[res] = types.copy_value(a[0], deref(p))
        elif op == "OpImageFetch":
            image = val(a[1])
            if not callable(image):
                raise UnsupportedConstruct(f"OpImageFetch of an image no resource binds: {line}")
            if len(a) != 5 or a[3] != "Lod":
                raise UnsupportedConstruct(f"OpImageFetch other than with one Lod operand: {line}")
            texel = tuple(image(val(a[2]), val(a[4])))
            kind = types.t[a[0]]
            if kind[0] != "vec" or len(texel) != kind[2]:
                raise UnsupportedConstruct(f"OpImageFetch texel {texel!r} does not fit {a[0]}")
            values[res] = texel
        elif op == "OpStore":
            p = pointers[a[0]]
            if p[0] == "pc":
                raise UnsupportedConstruct("store to PushConstant")
            if p[0] == "psb":
                v, t = val(a[1]), types.t.get(id_type.get(a[1]))
                if isinstance(v, bool) or not isinstance(v, int) or v > MASK \
                        or (t is not None and t[:2] != ("int", 32)):
                    raise UnsupportedConstruct("PhysicalStorageBuffer store of a value that is not a 32-bit integer")
                memory[psb_word(p[1])] = v
            else:
                p[0][p[1]] = types.copy_value(id_type.get(a[1]), val(a[1]))
        elif op in _ATOMICS:
            # One invocation, so an atomic is read, combine, write; the result is the original value (SPIR-V spec,
            # 3.42.18). Operands: result type, pointer, scope, semantics(, unequal semantics), value(, comparator).
            # The pointer is a physical-storage-buffer word or an element of a variable (a Workgroup array is LDS):
            # the same pointers OpLoad / OpStore take. Push constants are read-only.
            p = pointers[a[1]]
            if p[0] == "pc":
                raise UnsupportedConstruct(f"{op} on a PushConstant")
            if types.bits(a[0]) != 32:
                raise UnsupportedConstruct(f"{op} on a non-32-bit integer")
            if p[0] == "psb":
                old = memory.get(psb_word(p[1]), 0)
            else:
                old = deref(p)
                if isinstance(old, bool) or not isinstance(old, int) or old > MASK:
                    raise UnsupportedConstruct(f"{op} on a variable element that is not a 32-bit integer")
            if op == "OpAtomicLoad":
                values[res] = old
                continue
            if op == "OpAtomicCompareExchange":
                new = val(a[5]) if old == val(a[6]) & MASK else old
            else:
                v = val(a[4]) & MASK
                new = {"OpAtomicIAdd": old + v, "OpAtomicISub": old - v, "OpAtomicUMin": min(old, v),
                       "OpAtomicUMax": max(old, v), "OpAtomicSMin": old if _s32(old) <= _s32(v) else v,
                       "OpAtomicSMax": old if _s32(old) >= _s32(v) else v, "OpAtomicAnd": old & v,
                       "OpAtomicOr": old | v, "OpAtomicXor": old ^ v, "OpAtomicExchange": v}[op]
            if p[0] == "psb":
                memory[p[1]] = new & MASK
            else:
                p[0][p[1]] = new & MASK
            values[res] = old
        elif op == "OpSelect":
            if not isinstance(val(a[1]), bool):
                raise UnsupportedConstruct("OpSelect with a non-scalar-bool condition")
            values[res] = val(a[2]) if val(a[1]) else val(a[3])
        elif op == "OpCopyObject":
            values[res] = val(a[1])
        elif op == "OpBitcast":
            k, v = types.t[a[0]], val(a[1])
            source = types.t.get(id_type.get(a[1]))
            if k == ("float", 64) and types.is_word_pair(source):
                values[res] = struct.unpack("<d", struct.pack("<II", *v))[0]
            elif source == ("float", 64) and types.is_word_pair(k):
                values[res] = struct.unpack("<II", struct.pack("<d", v))
            elif k == ("float", 32) and isinstance(v, int):
                values[res] = bits_f32(v)
            elif k[0] == "int" and isinstance(v, float) and source != ("float", 64):
                values[res] = f32_bits(v)
            elif k[0] == "int" and isinstance(v, int):
                values[res] = v & wmask(a[0])
            elif k[0] == "vec" and isinstance(v, tuple) and len(v) == k[2] \
                    and types.t[k[1]][1] == 32 and source and source[0] == "vec" and types.t[source[1]][1] == 32:
                # Same component count, 32-bit components (an image texel as words): component by component.
                to_float = types.t[k[1]][0] == "float"
                values[res] = tuple((bits_f32(c) if isinstance(c, int) else c) if to_float else
                                    (f32_bits(c) if isinstance(c, float) else c & MASK) for c in v)
            else:
                raise UnsupportedConstruct("OpBitcast between these types")
        elif op == "OpCompositeExtract":
            v = val(a[1])
            for i in a[2:]:
                v = v[int(i)]
            values[res] = v
        elif op == "OpVectorExtractDynamic":
            # SPIR-V: an Index outside the vector's component count gives an undefined value
            v, i = val(a[1]), val(a[2])
            if not isinstance(i, int) or not 0 <= i < len(v):
                raise UnsupportedConstruct(f"OpVectorExtractDynamic index {i!r} outside the vector")
            values[res] = v[i]
        elif op == "OpCompositeConstruct":
            values[res] = tuple(val(x) for x in a[1:])
        elif op == "OpBitwiseAnd":
            values[res] = (val(a[1]) & val(a[2])) & wmask(a[0])
        elif op == "OpBitwiseOr":
            values[res] = (val(a[1]) | val(a[2])) & wmask(a[0])
        elif op == "OpBitwiseXor":
            values[res] = (val(a[1]) ^ val(a[2])) & wmask(a[0])
        elif op == "OpNot":
            values[res] = (~val(a[1])) & wmask(a[0])
        elif op == "OpShiftLeftLogical":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (val(a[1]) << val(a[2])) & wmask(a[0])
        elif op == "OpShiftRightLogical":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (val(a[1]) & wmask(a[0])) >> val(a[2])
        elif op == "OpShiftRightArithmetic":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (signed(val(a[1]), bits) >> val(a[2])) & wmask(a[0])
        elif op == "OpIAdd":
            values[res] = (val(a[1]) + val(a[2])) & wmask(a[0])
        elif op == "OpISub":
            values[res] = (val(a[1]) - val(a[2])) & wmask(a[0])
        elif op == "OpIMul":
            values[res] = (val(a[1]) * val(a[2])) & wmask(a[0])
        elif op == "OpUConvert":
            values[res] = val(a[1]) & wmask(a[0])
        elif op == "OpSConvert":
            # Sign-extend from the operand's width, then wrap to the result's (SPIR-V spec, OpSConvert).
            if types.t.get(id_type.get(a[1]), ("?",))[0] != "int":
                raise UnsupportedConstruct("OpSConvert of an operand with no known integer type")
            values[res] = sval(a[1]) & wmask(a[0])
        elif op == "OpBitReverse":
            bits = types.bits(a[0])
            values[res] = int(f"{val(a[1]) & wmask(a[0]):0{bits}b}"[::-1], 2)
        elif op in ("OpConvertUToF", "OpConvertSToF"):
            x = val(a[1])
            # The width comes from the operand's type, as in sval(): a 64-bit value below 2**32 passes the value
            # check, and _s32() would then give it the wrong sign under OpConvertSToF.
            t = types.t.get(id_type.get(a[1]))
            if isinstance(x, bool) or not isinstance(x, int) or x > MASK \
                    or (t is not None and (t[0] != "int" or t[1] != 32)):
                raise UnsupportedConstruct(f"{op} on a non-32-bit-integer operand")
            converted = float(_s32(x) if op == "OpConvertSToF" else x)
            # Python's binary64 float holds every 32-bit integer exactly.
            values[res] = converted if types.t[a[0]] == ("float", 64) else f32(converted)
        elif op == "OpBitCount":
            # Every bit of the operand counts: its width, not the result's (a 64-bit operand with a 32-bit result).
            t = types.t.get(id_type.get(a[1]))
            values[res] = (val(a[1]) & wmask(id_type[a[1]] if t is not None and t[0] == "int" else a[0])).bit_count()
        elif op == "OpISubBorrow":
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            values[res] = ((x - y) & MASK, 1 if x < y else 0)
        elif op in ("OpUMulExtended", "OpSMulExtended"):
            # Result struct {low, high} of the full 64-bit product (SPIR-V spec, 32-bit scalars only).
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            if op == "OpSMulExtended":
                x, y = _s32(x), _s32(y)
            p = (x * y) & 0xFFFFFFFFFFFFFFFF
            values[res] = (p & MASK, p >> 32)
        elif op == "OpIAddCarry":
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            values[res] = ((x + y) & MASK, (x + y) >> 32)
        elif op == "OpIEqual":
            values[res] = val(a[1]) == val(a[2])
        elif op == "OpINotEqual":
            values[res] = val(a[1]) != val(a[2])
        elif op in ("OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan", "OpSGreaterThanEqual"):
            x, y = sval(a[1]), sval(a[2])
            values[res] = {"OpSLessThan": x < y, "OpSLessThanEqual": x <= y,
                           "OpSGreaterThan": x > y, "OpSGreaterThanEqual": x >= y}[op]
        elif op == "OpULessThan":
            values[res] = val(a[1]) < val(a[2])
        elif op == "OpULessThanEqual":
            values[res] = val(a[1]) <= val(a[2])
        elif op == "OpUGreaterThan":
            values[res] = val(a[1]) > val(a[2])
        elif op == "OpUGreaterThanEqual":
            values[res] = val(a[1]) >= val(a[2])
        elif op == "OpControlBarrier":
            # <execution scope> <memory scope> <semantics>: run_workgroup's invocations meet here (each has finished
            # every earlier instruction, and with shared Workgroup memory sees the others' writes). run_fragment
            # refuses it, as it does a subgroup operation: a barrier belongs to a workgroup.
            yield (op, pc - 1, None, None)
        elif op == "OpGroupNonUniformBroadcastFirst":
            if types.t[a[0]][0] != "int" or val(a[1]) != 3:
                raise UnsupportedConstruct("OpGroupNonUniformBroadcastFirst other than a Subgroup-scope integer")
            values[res] = yield (op, pc - 1, val(a[2]), None)
        elif op == "OpGroupNonUniformShuffle":
            if types.t[a[0]][0] != "int" or val(a[1]) != 3:
                raise UnsupportedConstruct("OpGroupNonUniformShuffle other than a Subgroup-scope integer")
            values[res] = yield (op, pc - 1, val(a[2]), val(a[3]))
        elif op == "OpGroupNonUniformBallot":
            # OpGroupNonUniformBallot <uvec4 type> <scope> <bool predicate>
            k = types.t[a[0]]
            if k[0] != "vec" or k[2] != 4 or val(a[1]) != 3 or not isinstance(val(a[2]), bool):
                raise UnsupportedConstruct("OpGroupNonUniformBallot other than a Subgroup-scope uvec4")
            values[res] = yield (op, pc - 1, val(a[2]), None)
        elif op in ("OpGroupNonUniformAny", "OpGroupNonUniformAll"):
            # <bool> <scope> <bool predicate>
            if val(a[1]) != 3 or not isinstance(val(a[2]), bool):
                raise UnsupportedConstruct(f"{op} other than a Subgroup-scope bool")
            values[res] = yield (op, pc - 1, val(a[2]), None)
        elif op == "OpGroupNonUniformBitwiseOr":
            # <type> <scope> <group operation> <value>: only Reduce over a 32-bit integer.
            if val(a[1]) != 3 or a[2] != "Reduce" or types.t[a[0]][:2] != ("int", 32):
                raise UnsupportedConstruct("OpGroupNonUniformBitwiseOr other than a Subgroup-scope 32-bit Reduce")
            values[res] = yield (op, pc - 1, val(a[3]) & MASK, None)
        elif op == "OpGroupNonUniformBallotFindLSB":
            # <uint> <scope> <uvec4 ballot>: the lowest set bit; undefined when no bit is set.
            ballot = val(a[2])
            if val(a[1]) != 3:
                raise UnsupportedConstruct("OpGroupNonUniformBallotFindLSB scope")
            bits = sum(w << (32 * k) for k, w in enumerate(ballot))
            values[res] = (bits & -bits).bit_length() - 1 if bits else UNDEF
        elif op == "OpGroupNonUniformBallotBitExtract":
            # OpGroupNonUniformBallotBitExtract <bool> <scope> <uvec4 ballot> <index>: needs no other lane.
            ballot, idx = val(a[2]), val(a[3])
            if val(a[1]) != 3 or not 0 <= idx < 128:
                raise UnsupportedConstruct("OpGroupNonUniformBallotBitExtract scope or index")
            values[res] = bool((ballot[idx // 32] >> (idx % 32)) & 1)
        elif op == "OpLogicalAnd":
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, bool) and isinstance(y, bool)):
                raise UnsupportedConstruct("OpLogicalAnd on non-bool")
            values[res] = x and y
        elif op == "OpLogicalOr":
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, bool) and isinstance(y, bool)):
                raise UnsupportedConstruct("OpLogicalOr on non-bool")
            values[res] = x or y
        elif op == "OpLogicalNot":
            values[res] = not val(a[1])
        elif op in ("OpFAdd", "OpFSub", "OpFMul", "OpFDiv"):
            x, y = val(a[1]), val(a[2])
            if types.t[a[0]] == ("float", 64):
                if op != "OpFMul":
                    raise UnsupportedConstruct(f"{op} on binary64")
                values[res] = x * y
                continue

            def _fdiv(p, q):
                # SPIR-V leaves OpFDiv by zero undefined: refuse it, as the interpreter does for every undefined case
                if q == 0.0:
                    raise UnsupportedConstruct("OpFDiv by zero (undefined in SPIR-V)")
                if math.isinf(p) and math.isinf(q):
                    return math.nan
                return f32(p / q)       # x/inf is a signed zero, as in Python

            fn = {"OpFAdd": (lambda p, q: f32(ieee_nan(p + q, p, q))),
                  "OpFSub": (lambda p, q: f32(ieee_nan(p - q, p, q))),
                  "OpFMul": (lambda p, q: f32(ieee_nan(p * q, p, q))), "OpFDiv": _fdiv}[op]
            if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y) and all(
                    isinstance(p, float) and isinstance(q, float) for p, q in zip(x, y)):
                values[res] = tuple(fn(p, q) for p, q in zip(x, y))
            elif isinstance(x, float) and isinstance(y, float):
                values[res] = fn(x, y)
            else:
                raise UnsupportedConstruct(f"{op} on non-float")
        elif op == "OpFmaKHR":
            # SPV_KHR_fma: x * y + z correctly rounded, the same exact single rounding as GLSL Fma above (fma_f32)
            x, y, z = (val(t) for t in a[1:4])
            if isinstance(x, tuple):
                values[res] = tuple(fma_f32(p, q, s) for p, q, s in zip(x, y, z))
            else:
                values[res] = fma_f32(x, y, z)
        elif op == "OpFNegate":
            # IEEE 754 negate only flips the sign bit: it keeps a NaN's payload and does not quiet a signalling NaN.
            # A binary32 operand is negated on its bits, because the float round trip in f32() sets the quiet bit
            # of a signalling NaN before CPython 3.14 (issue #509), as on CI's 3.12: -v1 on 0x7F800001 must give
            # 0xFF800001, not 0xFFC00001 (AllNaNSrc0PayloadTests).
            x = val(a[1])
            if types.t[a[0]] == ("float", 64):
                values[res] = -x
            else:
                values[res] = bits_f32(f32_bits(x) ^ 0x80000000)
        elif op == "OpQuantizeToF16":
            # Single-instruction f32 -> f16 -> f32 round to nearest, with the
            # SPIR-V subnormal rule: a magnitude too small for a normal f16
            # becomes +0/-0. struct.pack('<e') keeps the subnormal, which would
            # make the interpreter more precise than a conforming driver.
            # Emitted by the translator for 16-bit interpolant reads; a
            # Pack/Unpack round-trip is observably folded to a no-op by at
            # least one driver.
            h = float_to_half(val(a[1]))
            if h & 0x7C00 == 0:
                h &= 0x8000
            values[res] = half_to_float(h)
        elif op in ("OpConvertFToU", "OpConvertFToS"):
            x, bits = val(a[1]), types.bits(a[0])
            if not isinstance(x, float) or not math.isfinite(x):
                raise UnsupportedConstruct(f"{op} of a non-finite value (undefined in SPIR-V)")
            n = math.trunc(x)
            lo, hi = (0, 1 << bits) if op == "OpConvertFToU" else (-(1 << (bits - 1)), 1 << (bits - 1))
            if not lo <= n < hi:
                raise UnsupportedConstruct(f"{op} out of range (undefined in SPIR-V)")
            values[res] = n & wmask(a[0])
        elif op in ("OpFOrdEqual", "OpFOrdNotEqual", "OpFOrdLessThan", "OpFOrdGreaterThan",
                    "OpFOrdGreaterThanEqual", "OpFOrdLessThanEqual", "OpFUnordEqual", "OpFUnordNotEqual",
                    "OpFUnordLessThan", "OpFUnordGreaterThan", "OpFUnordGreaterThanEqual", "OpFUnordLessThanEqual"):
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, float) and isinstance(y, float)):
                raise UnsupportedConstruct(f"{op} on non-float")
            # Ordered: false if either operand is NaN. Unordered: true if either is NaN. Python's own comparisons
            # are already ordered (NaN compares false), and -0.0 == 0.0 as in SPIR-V.
            unordered = math.isnan(x) or math.isnan(y)
            rel = {"Equal": x == y, "NotEqual": x != y and not unordered, "LessThan": x < y,
                   "GreaterThan": x > y, "GreaterThanEqual": x >= y, "LessThanEqual": x <= y}[op.removeprefix("OpFUnord").removeprefix("OpFOrd")]
            values[res] = (rel or unordered) if op.startswith("OpFUnord") else rel
        elif op in ("OpIsNan", "OpIsInf"):
            x = val(a[1])
            if not isinstance(x, float):
                raise UnsupportedConstruct(f"{op} on non-float")
            values[res] = math.isnan(x) if op == "OpIsNan" else math.isinf(x)
        elif op in ("OpBitFieldUExtract", "OpBitFieldSExtract"):
            base, off, cnt = val(a[1]), val(a[2]), val(a[3])
            if off + cnt > 32:
                raise UnsupportedConstruct("bit-field extract beyond 32 bits (undefined in SPIR-V)")
            field = (base >> off) & ((1 << cnt) - 1)
            if op == "OpBitFieldSExtract" and cnt and field >> (cnt - 1):
                field -= 1 << cnt
            values[res] = field & MASK
        elif op == "OpBitFieldInsert":
            base, ins, off, cnt = (val(x) for x in a[1:5])
            if off + cnt > 32:
                raise UnsupportedConstruct("bit-field insert beyond 32 bits (undefined in SPIR-V)")
            mk = ((1 << cnt) - 1) << off
            values[res] = ((base & ~mk) | ((ins << off) & mk)) & MASK
        elif op == "OpConvertUToPtr":
            values[res] = ("psb", val(a[1]))
            pointers[res] = values[res]
        elif op == "OpExtInst":
            if len(a) < 3 or a[1] != "%glsl":
                raise UnsupportedConstruct("extended instruction set")
            name, args = a[2], [val(x) for x in a[3:]]
            if name in ("SAbs", "SMin", "SMax", "SClamp", "FindILsb", "FindSMsb", "FindUMsb"):
                bits = types.bits(a[0])
                for operand in a[3:]:
                    if operand not in id_type:
                        raise UnsupportedConstruct(f"{name}: unknown integer type for {operand}")
                    width = types.bits(id_type[operand])
                    if width != bits:
                        raise UnsupportedConstruct(f"{name}: {width}-bit operand {operand} with a {bits}-bit result")
            if name == "UnpackHalf2x16":
                values[res] = (half_to_float(args[0]), half_to_float(args[0] >> 16))
            elif name == "PackHalf2x16":
                values[res] = float_to_half(args[0][0]) | (float_to_half(args[0][1]) << 16)
            elif name == "Fma":
                # Exact product and sum, one rounding to binary32 (fma_f32).
                x, y, z = args
                if isinstance(x, tuple):
                    values[res] = tuple(fma_f32(p, q, s) for p, q, s in zip(x, y, z))
                else:
                    values[res] = fma_f32(x, y, z)
            elif name in ("Sin", "Cos"):
                values[res] = f32(math.sin(args[0]) if name == "Sin" else math.cos(args[0]))
            elif name == "FAbs":
                magnitude = abs(args[0])
                values[res] = magnitude if types.t[a[0]] == ("float", 64) else f32(magnitude)
            elif name == "FSign":
                # GLSL.std.450: 1.0 if x > 0, 0.0 if x = 0 (keeping the zero's sign), -1.0 if x < 0
                x = args[0]
                if math.isnan(x):
                    raise UnsupportedConstruct("GLSL.std.450 FSign of a NaN (result not specified)")
                values[res] = 1.0 if x > 0 else -1.0 if x < 0 else x
            elif name == "SAbs":
                # Two's complement: the most negative value is its own absolute value (as on the hardware).
                values[res] = abs(signed(args[0], bits)) & wmask(a[0])
            elif name in ("FMax", "FClamp"):
                def fmax_clamp(*xs):
                    if any(math.isnan(v) for v in xs) or (name == "FClamp" and xs[1] > xs[2]):
                        raise UnsupportedConstruct(f"GLSL.std.450 {name} with these operands (undefined)")
                    # Equal zeros: -0 is ordered below +0, as in NMin/NMax above (the order RDNA2 hardware uses,
                    # so a med3 built from these ops is the hardware's median for signed zeros too).
                    v = _nmin_nmax("NMax", xs[0], xs[1])
                    result = v if name == "FMax" else _nmin_nmax("NMin", v, xs[2])
                    return result if types.t[a[0]] == ("float", 64) else f32(result)
                # Vector operands (v_pk_* clamp) are component-wise.
                values[res] = (tuple(fmax_clamp(*c) for c in zip(*args)) if isinstance(args[0], tuple)
                               else fmax_clamp(*args))
            elif name in ("RoundEven", "Floor", "Ceil", "Trunc"):
                # round() breaks ties to even. An integral result keeps the operand's sign (IEEE 754
                # roundToIntegral), so RoundEven(-0.4), Ceil(-0.5) and Trunc(-0.5) are -0.0, not +0.0.
                x = args[0]
                fn = {"RoundEven": round, "Floor": math.floor, "Ceil": math.ceil, "Trunc": math.trunc}[name]

                def integral(v):
                    result = math.copysign(float(fn(v)), v) if math.isfinite(v) else v
                    return result if types.t[a[0]] == ("float", 64) else f32(result)
                values[res] = tuple(integral(v) for v in x) if isinstance(x, tuple) else integral(x)
            elif name in ("Exp2", "Log2", "Sqrt", "InverseSqrt"):
                # Correctly rounded here; Vulkan allows a few ULP of error (tests compare exact results only for
                # inputs whose result is exact). GLSL.std.450 leaves Log2 of x <= 0, Sqrt of x < 0 and
                # InverseSqrt of x <= 0 undefined; those raise.
                x = args[0]
                if math.isnan(x):
                    values[res] = x
                elif name == "Exp2":
                    if math.isinf(x):
                        values[res] = math.inf if x > 0 else 0.0
                    else:
                        values[res] = f32(2.0 ** x) if x < 1024.0 else math.inf
                elif (name == "Sqrt" and x < 0.0) or (name != "Sqrt" and x <= 0.0):
                    raise UnsupportedConstruct(f"GLSL.std.450 {name} of {x!r} (undefined)")
                elif math.isinf(x):
                    values[res] = {"Log2": math.inf, "Sqrt": math.inf, "InverseSqrt": 0.0}[name]
                elif name == "Log2":
                    values[res] = f32(math.log2(x))
                elif name == "Sqrt":
                    values[res] = f32(math.sqrt(x))         # sqrt(-0.0) is -0.0
                else:
                    values[res] = f32(1.0 / math.sqrt(x))
            elif name == "FMin":
                if any(math.isnan(x) for x in args):
                    raise UnsupportedConstruct(f"GLSL.std.450 {name} of NaN (undefined)")
                values[res] = f32(_nmin_nmax("NMin", *args))
            elif name in ("NMin", "NMax"):
                x, y = args
                if isinstance(x, tuple):
                    values[res] = tuple(_nmin_nmax(name, p, q) for p, q in zip(x, y))
                else:
                    values[res] = _nmin_nmax(name, x, y)
            elif name == "Fract":
                if not isinstance(args[0], float):
                    raise UnsupportedConstruct("Fract on non-float")
                values[res] = _fract(args[0])
            elif name == "UMin":
                values[res] = min(args)
            elif name == "UMax":
                values[res] = max(args)
            elif name == "SMax":
                values[res] = max(args, key=lambda v: signed(v, bits))
            elif name == "SMin":
                values[res] = min(args, key=lambda v: signed(v, bits))
            elif name in ("UClamp", "SClamp"):
                # min(max(x, minVal), maxVal) (GLSL.std.450; undefined if minVal > maxVal, which raises here).
                key = (lambda v: signed(v, bits)) if name == "SClamp" else (lambda v: v)
                x, lo, hi = args
                if key(lo) > key(hi):
                    raise UnsupportedConstruct(f"{name} with minVal > maxVal (undefined)")
                values[res] = min(max(x, lo, key=key), hi, key=key)
            elif name in ("FindILsb", "FindUMsb", "FindSMsb") and bits != 32:
                raise UnsupportedConstruct(f"GLSL.std.450 {name} on a {bits}-bit integer (32-bit components only)")
            elif name == "FindILsb":
                # Bit number of the least significant 1 bit, -1 if the value is 0 (GLSL.std.450).
                x = args[0] & MASK
                values[res] = (x & -x).bit_length() - 1 & MASK if x else MASK
            elif name == "FindUMsb":
                # Bit number of the most significant 1 bit, -1 if the value is 0 (GLSL.std.450).
                x = args[0] & MASK
                values[res] = x.bit_length() - 1 if x else MASK
            elif name == "FindSMsb":
                x = _s32(args[0])
                if x in (0, -1):
                    values[res] = MASK
                elif x > 0:
                    values[res] = x.bit_length() - 1
                else:
                    values[res] = (~x).bit_length() - 1
            else:
                raise UnsupportedConstruct(f"GLSL.std.450 {name}")
        else:
            raise UnsupportedConstruct(op)

    if killed:
        return {}
    result = {}
    for name in outputs:
        v = var_box[name][0]
        flat = v if isinstance(v, tuple) else (v,)
        result[name] = tuple(f32_bits(x) if isinstance(x, float) else x & MASK for x in flat)
    return result
