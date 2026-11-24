"""Reflect the user-defined graphics interfaces in a binary SPIR-V module.

This reader is deliberately limited to declarations used for Vulkan stage linking. It does not
validate shader instructions, device features, or resource bindings. IDs and debug names are local
to each module; the result uses normalized types and Location/Component decorations instead.
Literal specialization constants in array lengths use their declared defaults.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
import struct


class InterfaceError(ValueError):
    """The module's requested interface could not be reflected."""


SPIRV_MAGIC = 0x07230203
SPIRV_MAGIC_SWAPPED = 0x03022307

# SPIR-V core grammar. The NV/EXT spelling of PerPrimitive uses the same enumerant.
OP_ENTRY_POINT = 15
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_MATRIX = 24
OP_TYPE_ARRAY = 28
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_SPEC_CONSTANT = 50
OP_VARIABLE = 59
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_DECORATION_GROUP = 73
OP_GROUP_DECORATE = 74
OP_GROUP_MEMBER_DECORATE = 75

TYPE_OPS = {OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_MATRIX,
            OP_TYPE_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER}
DECLARATION_OPS = TYPE_OPS | {OP_ENTRY_POINT, OP_CONSTANT, OP_SPEC_CONSTANT, OP_VARIABLE,
                             OP_DECORATE, OP_MEMBER_DECORATE, OP_DECORATION_GROUP,
                             OP_GROUP_DECORATE, OP_GROUP_MEMBER_DECORATE}
TYPE_OPERANDS = {OP_TYPE_INT: 3, OP_TYPE_FLOAT: 2, OP_TYPE_VECTOR: 3, OP_TYPE_MATRIX: 3,
                 OP_TYPE_ARRAY: 3, OP_TYPE_POINTER: 3}
EXECUTION_STAGES = {0: "vs", 3: "gs", 4: "ps", 5268: "ms", 5365: "ms"}  # MeshNV and MeshEXT
STORAGE_INPUT = 1
STORAGE_OUTPUT = 3
DEC_BLOCK = 2
DEC_BUILTIN = 11
DEC_PATCH = 15
DEC_LOCATION = 30
DEC_COMPONENT = 31
DEC_INDEX = 32
DEC_PASSTHROUGH_NV = 5250
DEC_PER_PRIMITIVE = 5271
DEC_PER_VERTEX = 5285
DECORATION_NAMES = {DEC_BLOCK: "Block", DEC_BUILTIN: "BuiltIn", DEC_PATCH: "Patch",
                    DEC_LOCATION: "Location", DEC_COMPONENT: "Component", DEC_INDEX: "Index",
                    DEC_PASSTHROUGH_NV: "PassthroughNV", DEC_PER_PRIMITIVE: "PerPrimitiveEXT",
                    DEC_PER_VERTEX: "PerVertexKHR"}
VALUE_DECORATIONS = {DEC_BUILTIN, DEC_LOCATION, DEC_COMPONENT, DEC_INDEX}

# Bound expansion of arrays from malformed modules. This is far beyond the varying space of the
# graphics stages on current Vulkan drivers. Type and layout recursion has a separate depth limit.
MAX_EXPANDED_LOCATIONS = 16384
MAX_TYPE_DEPTH = 64


@dataclass(frozen=True)
class NumericType:
    kind: str
    width: int
    components: int = 1

    def describe(self):
        scalar = f"{self.kind}{self.width}"
        return scalar if self.components == 1 else f"vec{self.components}<{scalar}>"

    def location_masks(self, component):
        slots = self.components * (self.width // 32 if self.width >= 32 else 1)
        if self.width == 64 and component % 2:
            raise InterfaceError("a 64-bit varying starts on an odd Component")
        if self.width != 64 and component + slots > 4:
            raise InterfaceError("a 16-bit or 32-bit varying extends beyond its Location")
        if self.width == 64 and component + slots > 4 and component:
            raise InterfaceError("a 64-bit vector spanning two locations needs Component 0")
        mask = ((1 << slots) - 1) << component
        if mask <= 0xf:
            return (mask,)
        return (mask & 0xf, mask >> 4)


@dataclass(frozen=True)
class AggregateType:
    kind: str
    items: tuple[NumericType | "AggregateType", ...]
    count: int = 1

    def describe(self):
        if self.kind == "struct":
            return "struct{" + ", ".join(item.describe() for item in self.items) + "}"
        return f"{self.kind}[{self.count}]<{self.items[0].describe()}>"

    def location_masks(self, component):
        if self.kind == "struct":
            if component:
                raise InterfaceError("a structure cannot start at a nonzero Component")
            masks = []
            for member in self.items:
                member_masks = member.location_masks(0)
                if len(masks) + len(member_masks) > MAX_EXPANDED_LOCATIONS:
                    raise InterfaceError("structure uses too many interface locations to inspect")
                masks.extend(member_masks)
            return tuple(masks)
        element = self.items[0].location_masks(component)
        if len(element) * self.count > MAX_EXPANDED_LOCATIONS:
            raise InterfaceError("aggregate uses too many interface locations to inspect")
        return element * self.count


@dataclass(frozen=True, order=True)
class Address:
    location: int
    component: int = 0
    index: int = 0

    def describe(self):
        suffix = f", index {self.index}" if self.index else ""
        return f"location {self.location}, component {self.component}{suffix}"


@dataclass(frozen=True)
class Varying:
    address: Address
    type: NumericType | AggregateType
    per_primitive: bool = False

    def occupied(self):
        for offset, mask in enumerate(self.type.location_masks(self.address.component)):
            for component in range(4):
                if mask & (1 << component):
                    yield Address(self.address.location + offset, component, self.address.index)

    def describe(self):
        mode = " per-primitive" if self.per_primitive else ""
        return f"{self.type.describe()}{mode} at {self.address.describe()}"


@dataclass(frozen=True)
class Declaration:
    """One standalone varying or an entire structure. Preserve member order when linking."""

    varyings: tuple[Varying, ...]
    symbol: int
    structure: bool = False
    block: bool = False
    indexed_by_vertex_or_primitive: bool = False

    def describe(self):
        parts = ", ".join(varying.describe() for varying in self.varyings)
        if self.structure:
            kind = "block" if self.block else "struct"
            return f"{kind} %{self.symbol} ({parts})"
        return f"%{self.symbol} ({parts})"


@dataclass(frozen=True)
class StageInterface:
    stage: str
    entry: str
    inputs: tuple[Declaration, ...]
    outputs: tuple[Declaration, ...]


@dataclass(frozen=True)
class EntryPoint:
    model: int
    name: str
    interface_ids: tuple[int, ...]


def parse_words(data):
    if len(data) < 20 or len(data) % 4:
        raise InterfaceError("SPIR-V must contain a five-word header and complete 32-bit words")
    words = array("I")
    words.frombytes(data)
    if words[0] == SPIRV_MAGIC_SWAPPED:
        words.byteswap()
    if words[0] != SPIRV_MAGIC:
        raise InterfaceError("invalid SPIR-V magic")
    if not 0x00010000 <= words[1] <= 0x00010600 or words[1] & 0xff:
        raise InterfaceError(f"unsupported SPIR-V version word 0x{words[1]:08x}")
    if not words[3] or words[4]:
        raise InterfaceError("invalid SPIR-V ID bound or reserved header word")
    return words


def instructions(words):
    start = 5
    while start < len(words):
        first = words[start]
        length, opcode = first >> 16, first & 0xffff
        end = start + length
        if not length or end > len(words):
            raise InterfaceError(f"invalid instruction length at word {start}")
        if opcode in DECLARATION_OPS:
            yield opcode, tuple(words[start + 1:end]), start
        start = end


def decode_literal_string(words):
    for i, word in enumerate(words):
        if b"\x00" in struct.pack("<I", word):
            raw = struct.pack(f"<{i + 1}I", *words[:i + 1])
            name, _, padding = raw.partition(b"\x00")
            if any(padding):
                raise InterfaceError("nonzero padding in OpEntryPoint name")
            try:
                return name.decode("utf-8"), i + 1
            except UnicodeDecodeError as exc:
                raise InterfaceError("OpEntryPoint name is not UTF-8") from exc
    raise InterfaceError("unterminated OpEntryPoint name")


class Module:
    def __init__(self, data):
        words = parse_words(data)
        self.version = words[1]
        self.bound = words[3]
        self.types = {}
        self.constants = {}
        self.variables = {}
        self.entry_points = []
        self.decorations = defaultdict(dict)
        self.member_decorations = defaultdict(dict)
        self.groups = set()
        self.group_uses = []
        self.definition_ids = set()
        self.shape_cache = {}
        for opcode, operands, at in instructions(words):
            self.read_instruction(opcode, operands, at)
        self.apply_decoration_groups()

    def require_id(self, identifier):
        if not 0 < identifier < self.bound:
            raise InterfaceError(f"ID %{identifier} is outside the SPIR-V bound")

    def define(self, identifier):
        self.require_id(identifier)
        if identifier in self.definition_ids:
            raise InterfaceError(f"duplicate definition of ID %{identifier}")
        self.definition_ids.add(identifier)

    @staticmethod
    def require_operands(operands, at, minimum, maximum=None):
        if len(operands) < minimum or (maximum is not None and len(operands) > maximum):
            raise InterfaceError(f"invalid operands at word {at}")

    def read_instruction(self, opcode, operands, at):
        if opcode in TYPE_OPS:
            required = TYPE_OPERANDS.get(opcode)
            # OpTypeFloat can also carry an optional FPEncoding operand.
            maximum = 3 if opcode == OP_TYPE_FLOAT else required
            self.require_operands(operands, at, required or 1, maximum)
            identifier, *args = operands
            self.define(identifier)
            self.types[identifier] = (opcode, tuple(args))
        elif opcode in (OP_CONSTANT, OP_SPEC_CONSTANT):
            self.require_operands(operands, at, 3)
            type_id, identifier, *literal = operands
            self.require_id(type_id)
            self.define(identifier)
            self.constants[identifier] = (type_id, tuple(literal))
        elif opcode == OP_VARIABLE:
            self.require_operands(operands, at, 3, 4)
            pointer, identifier, storage = operands[:3]
            self.require_id(pointer)
            self.define(identifier)
            self.variables[identifier] = (pointer, storage)
        elif opcode == OP_ENTRY_POINT:
            self.require_operands(operands, at, 3)
            model, function, *rest = operands
            self.require_id(function)
            name, words_used = decode_literal_string(rest)
            ids = tuple(rest[words_used:])
            for identifier in ids:
                self.require_id(identifier)
            if self.version < 0x00010400:
                # OpEntryPoint permits repeated interface IDs before SPIR-V 1.4.
                ids = tuple(dict.fromkeys(ids))
            elif len(set(ids)) != len(ids):
                raise InterfaceError(f"duplicate entry-point interface ID in {name!r}")
            self.entry_points.append(EntryPoint(model, name, ids))
        elif opcode in (OP_DECORATE, OP_MEMBER_DECORATE):
            split = 1 if opcode == OP_DECORATE else 2
            self.require_operands(operands, at, split + 1)
            identifier = operands[0]
            self.require_id(identifier)
            decoration, *extra = operands[split:]
            if decoration not in DECORATION_NAMES:
                return
            expected = int(decoration in VALUE_DECORATIONS)
            if len(extra) != expected:
                raise InterfaceError(f"invalid {DECORATION_NAMES[decoration]} decoration at word {at}")
            target = (self.decorations[identifier] if split == 1
                      else self.member_decorations[(identifier, operands[1])])
            self.add_decoration(target, decoration, extra[0] if extra else True)
        elif opcode == OP_DECORATION_GROUP:
            self.require_operands(operands, at, 1, 1)
            self.define(operands[0])
            self.groups.add(operands[0])
        else:
            self.require_operands(operands, at, 2 if opcode == OP_GROUP_DECORATE else 3)
            if opcode == OP_GROUP_MEMBER_DECORATE and len(operands) % 2 != 1:
                raise InterfaceError(f"unpaired group member decoration at word {at}")
            self.group_uses.append((opcode, operands))

    @staticmethod
    def add_decoration(target, decoration, value):
        if decoration in target and target[decoration] != value:
            raise InterfaceError(f"conflicting {DECORATION_NAMES[decoration]} decorations")
        target[decoration] = value

    def apply_decoration_groups(self):
        for opcode, operands in self.group_uses:
            group, *targets = operands
            if group not in self.groups:
                raise InterfaceError(f"undefined decoration group %{group}")
            source = self.decorations.get(group, {})
            if opcode == OP_GROUP_DECORATE:
                destinations = ((identifier, None) for identifier in targets)
            else:
                destinations = zip(targets[::2], targets[1::2])
            for identifier, member in destinations:
                self.require_id(identifier)
                if identifier in self.groups:
                    raise InterfaceError("decoration groups cannot decorate each other")
                target = (self.decorations[identifier] if member is None
                          else self.member_decorations[(identifier, member)])
                for decoration, value in source.items():
                    self.add_decoration(target, decoration, value)

    def type_declaration(self, identifier):
        try:
            return self.types[identifier]
        except KeyError:
            raise InterfaceError(f"missing or unsupported varying type %{identifier}") from None

    def array_count(self, identifier):
        try:
            type_id, literal = self.constants[identifier]
        except KeyError:
            raise InterfaceError(
                f"varying array length %{identifier} is not a literal OpConstant or OpSpecConstant"
            ) from None
        opcode, args = self.type_declaration(type_id)
        if opcode != OP_TYPE_INT or args[0] not in (8, 16, 32, 64) or args[1] not in (0, 1):
            raise InterfaceError("varying array length must be an integer")
        width, signed = args
        if len(literal) != (width + 31) // 32:
            raise InterfaceError("invalid literal width for varying array length")
        count = sum(word << (32 * i) for i, word in enumerate(literal)) & ((1 << width) - 1)
        if not count or (signed and count >> (width - 1)):
            raise InterfaceError("varying array length must be positive")
        return count

    def shape(self, identifier, path=()):
        if identifier in self.shape_cache:
            return self.shape_cache[identifier]
        if identifier in path or len(path) == MAX_TYPE_DEPTH:
            raise InterfaceError("cyclic or excessively deep varying type")
        opcode, args = self.type_declaration(identifier)
        nested = (*path, identifier)
        if opcode in (OP_TYPE_INT, OP_TYPE_FLOAT):
            if opcode == OP_TYPE_FLOAT and len(args) > 1:
                raise InterfaceError(f"unsupported floating-point encoding {args[1]} in graphics interface")
            width = args[0]
            if width not in (16, 32, 64) or (opcode == OP_TYPE_INT and args[1] not in (0, 1)):
                raise InterfaceError("unsupported numeric type in graphics interface")
            kind = "f" if opcode == OP_TYPE_FLOAT else ("i" if args[1] else "u")
            result = NumericType(kind, width)
        elif opcode == OP_TYPE_VECTOR:
            element = self.shape(args[0], nested)
            if not isinstance(element, NumericType) or element.components != 1 or not 2 <= args[1] <= 4:
                raise InterfaceError("invalid varying vector type")
            result = NumericType(element.kind, element.width, args[1])
        elif opcode in (OP_TYPE_ARRAY, OP_TYPE_MATRIX):
            element = self.shape(args[0], nested)
            if opcode == OP_TYPE_ARRAY:
                result = AggregateType("array", (element,), self.array_count(args[1]))
            else:
                if (not isinstance(element, NumericType) or element.kind != "f" or element.components < 2
                        or not 2 <= args[1] <= 4):
                    raise InterfaceError("invalid varying matrix type")
                result = AggregateType("matrix", (element,), args[1])
        elif opcode == OP_TYPE_STRUCT:
            if not args:
                raise InterfaceError("empty structure in graphics interface")
            for i in range(len(args)):
                if self.member_decorations.get((identifier, i)):
                    raise InterfaceError("decorated nested interface structure members are unsupported")
            result = AggregateType("struct", tuple(self.shape(member, nested) for member in args))
        else:
            raise InterfaceError("unsupported pointer in graphics interface")
        self.shape_cache[identifier] = result
        return result

    def pick_entry(self, stage, entry):
        named = [candidate for candidate in self.entry_points if candidate.name == entry]
        matches = [candidate for candidate in named if stage is None or EXECUTION_STAGES.get(candidate.model) == stage]
        if not matches:
            found = ", ".join(EXECUTION_STAGES.get(candidate.model, str(candidate.model)) for candidate in named)
            if found:
                raise InterfaceError(f"entry point {entry!r} is {found}, expected {stage}")
            raise InterfaceError(f"entry point {entry!r} not found")
        if len(matches) > 1:
            raise InterfaceError(f"entry point {entry!r} is ambiguous without a stage")
        selected = matches[0]
        if selected.model not in EXECUTION_STAGES:
            raise InterfaceError(f"unsupported execution model {selected.model} for entry point {entry!r}")
        return selected

    def varying_type(self, pointer, storage, stage, decorations):
        opcode, args = self.type_declaration(pointer)
        if opcode != OP_TYPE_POINTER or args[0] != storage:
            raise InterfaceError("variable pointer type does not match its storage class")
        pointee = args[1]
        per_vertex = DEC_PER_VERTEX in decorations
        if per_vertex and (stage, storage) != ("ps", STORAGE_INPUT):
            raise InterfaceError("PerVertexKHR is only supported on fragment inputs")
        primitive_array = (stage, storage) in (("gs", STORAGE_INPUT), ("ms", STORAGE_OUTPUT))
        indexed = per_vertex or primitive_array
        if indexed:
            # The array indexes vertices or primitives. It is outside the linked payload shape
            # and does not consume locations. Its length does not need to match the other stage.
            opcode, args = self.type_declaration(pointee)
            if opcode != OP_TYPE_ARRAY:
                raise InterfaceError(f"{stage} requires an outer per-vertex or per-primitive array")
            pointee = args[0]
        return pointee, indexed

    @staticmethod
    def address(decorations, location=None):
        location = decorations.get(DEC_LOCATION, location)
        component = decorations.get(DEC_COMPONENT, 0)
        index = decorations.get(DEC_INDEX, 0)
        if location is None:
            raise InterfaceError("user varying is missing a Location decoration")
        if component > 3 or index > 1:
            raise InterfaceError("invalid Component or Index decoration")
        return Address(location, component, index)

    @staticmethod
    def check_modes(decorations):
        if DEC_PASSTHROUGH_NV in decorations:
            raise InterfaceError("geometry passthrough interfaces require different matching rules")
        if DEC_PATCH in decorations:
            raise InterfaceError("tessellation patch interfaces are unsupported")

    def declaration(self, symbol, pointer, storage, stage):
        decorations = self.decorations.get(symbol, {})
        self.check_modes(decorations)
        if DEC_BUILTIN in decorations:
            return None
        type_id, indexed = self.varying_type(pointer, storage, stage, decorations)
        opcode, args = self.type_declaration(type_id)
        per_primitive = DEC_PER_PRIMITIVE in decorations
        if opcode != OP_TYPE_STRUCT:
            varying = Varying(self.address(decorations), self.shape(type_id), per_primitive)
            return Declaration((varying,), symbol, indexed_by_vertex_or_primitive=indexed)

        member_decs = [self.member_decorations.get((type_id, i), {}) for i in range(len(args))]
        if member_decs and all(DEC_BUILTIN in member for member in member_decs):
            return None
        if any(DEC_BUILTIN in member for member in member_decs):
            raise InterfaceError("mixed built-in and user-defined structure members")
        is_block = DEC_BLOCK in self.decorations.get(type_id, {})
        base_location = decorations.get(DEC_LOCATION)
        if base_location is None and not is_block:
            raise InterfaceError("a non-block structure needs a Location")
        if DEC_COMPONENT in decorations or DEC_INDEX in decorations:
            raise InterfaceError("Component or Index on an entire varying structure is unsupported")
        if base_location is not None and any(DEC_LOCATION in member for member in member_decs):
            raise InterfaceError("both the structure and its members declare Locations")
        if not args:
            raise InterfaceError("empty structure in graphics interface")
        members = []
        next_location = base_location
        for child, member in zip(args, member_decs):
            self.check_modes(member)
            address = self.address(member, next_location)
            shape = self.shape(child, (type_id,))
            members.append(Varying(address, shape, per_primitive or DEC_PER_PRIMITIVE in member))
            if base_location is not None:
                next_location += len(shape.location_masks(address.component))
        return Declaration(tuple(members), symbol, structure=True, block=is_block,
                           indexed_by_vertex_or_primitive=indexed)

    def reflect(self, stage=None, entry="main"):
        selected = self.pick_entry(stage, entry)
        stage = EXECUTION_STAGES[selected.model]
        directions = {STORAGE_INPUT: [], STORAGE_OUTPUT: []}
        slots = {STORAGE_INPUT: {}, STORAGE_OUTPUT: {}}
        for symbol in selected.interface_ids:
            if symbol not in self.variables:
                raise InterfaceError(f"entry point references missing variable %{symbol}")
            pointer, storage = self.variables[symbol]
            if storage not in directions:
                continue
            try:
                declaration = self.declaration(symbol, pointer, storage, stage)
                if declaration is None:
                    continue
                for varying in declaration.varyings:
                    for address in varying.occupied():
                        before = slots[storage].get(address)
                        if before is not None:
                            raise InterfaceError(
                                f"overlap with %{before} at {address.describe()}")
                        slots[storage][address] = symbol
            except InterfaceError as exc:
                direction = "input" if storage == STORAGE_INPUT else "output"
                raise InterfaceError(f"{stage} {direction} variable %{symbol}: {exc}") from exc
            directions[storage].append(declaration)
        key = lambda declaration: declaration.varyings[0].address
        return StageInterface(stage, entry, tuple(sorted(directions[STORAGE_INPUT], key=key)),
                              tuple(sorted(directions[STORAGE_OUTPUT], key=key)))


def reflect_module(data, stage=None, entry="main"):
    return Module(data).reflect(stage, entry)
