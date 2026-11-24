"""Unit tests for tools/spirv_interface.py using synthetic SPIR-V binaries.

Runs with the standard library only (no game data, no GPU):
    python -I tests/test_spirv_interface.py
"""
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spirv_interface as spv  # noqa: E402

# Extra SPIR-V opcodes and decorations used to exercise filtering and resource handling.
OP_NAME = 5
OP_EXECUTION_MODE = 16
OP_CAPABILITY = 17
OP_MEMORY_MODEL = 14
OP_TYPE_VOID = 19
OP_TYPE_FUNCTION = 33
OP_FUNCTION = 54
OP_FUNCTION_END = 56

STORAGE_UNIFORM_CONSTANT = 0
STORAGE_UNIFORM = 2
STORAGE_WORKGROUP = 4
STORAGE_PRIVATE = 6
STORAGE_PUSH_CONSTANT = 9
STORAGE_STORAGE_BUFFER = 12

DEC_NO_PERSPECTIVE = 13
DEC_FLAT = 14
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
DEC_OFFSET = 35

MODEL_VS = 0
MODEL_TESC = 1
MODEL_TESE = 2
MODEL_GS = 3
MODEL_PS = 4
MODEL_CS = 5
MODEL_TASK_EXT = 5364
MODEL_MS = 5365

STAGE_MODELS = {"vs": MODEL_VS, "gs": MODEL_GS, "ps": MODEL_PS, "ms": MODEL_MS}


def encode_string_words(text):
    raw = text.encode("utf-8") + b"\x00"
    raw += bytes(-len(raw) % 4)
    return struct.unpack(f"<{len(raw) // 4}I", raw)


def pack_words(words, *, big_endian=False):
    endian = ">" if big_endian else "<"
    return struct.pack(f"{endian}{len(words)}I", *words)


class SpirvBuilder:
    """Minimal synthetic binary SPIR-V builder for unit tests."""

    def __init__(self, stage="vs", *, version=0x00010500, start_id=1):
        self.stage = stage
        self.version = version
        self.next_id = start_id
        self.prefix = []
        self.entries = []
        self.code = []
        self.interface_ids = []
        self._u32 = None

    def alloc(self):
        identifier = self.next_id
        self.next_id += 1
        return identifier

    @staticmethod
    def _encode_inst(opcode, args):
        return [((len(args) + 1) << 16) | opcode, *args]

    def emit(self, opcode, *args):
        self.code.extend(self._encode_inst(opcode, args))

    def emit_prefix(self, opcode, *args):
        self.prefix.extend(self._encode_inst(opcode, args))

    def emit_type(self, opcode, *args):
        result = self.alloc()
        self.emit(opcode, result, *args)
        return result

    def u32_type(self):
        if self._u32 is None:
            self._u32 = self.emit_type(spv.OP_TYPE_INT, 32, 0)
        return self._u32

    def int_type(self, width=32, signed=False):
        if width == 32 and not signed:
            return self.u32_type()
        return self.emit_type(spv.OP_TYPE_INT, width, int(bool(signed)))

    def float_type(self, width=32):
        return self.emit_type(spv.OP_TYPE_FLOAT, width)

    def numeric(self, kind="f", width=32, components=1):
        if kind == "f":
            scalar = self.float_type(width)
        else:
            scalar = self.int_type(width, signed=(kind == "i"))
        if components == 1:
            return scalar
        return self.emit_type(spv.OP_TYPE_VECTOR, scalar, components)

    def matrix(self, cols, rows, width=32):
        return self.emit_type(spv.OP_TYPE_MATRIX, self.numeric("f", width, rows), cols)

    def constant(self, type_id, *literal_words):
        result = self.alloc()
        self.emit(spv.OP_CONSTANT, type_id, result, *literal_words)
        return result

    def u32_const(self, value):
        return self.constant(self.u32_type(), value)

    def array(self, element, length):
        return self.emit_type(spv.OP_TYPE_ARRAY, element, self.u32_const(length))

    def struct_type(self, *members):
        return self.emit_type(spv.OP_TYPE_STRUCT, *members)

    def pointer(self, storage, pointee):
        return self.emit_type(spv.OP_TYPE_POINTER, storage, pointee)

    def decorate(self, target, decoration, *operands):
        self.emit(spv.OP_DECORATE, target, decoration, *operands)

    def member_decorate(self, target, member, decoration, *operands):
        self.emit(spv.OP_MEMBER_DECORATE, target, member, decoration, *operands)

    def decoration_group(self):
        group = self.alloc()
        self.emit(spv.OP_DECORATION_GROUP, group)
        return group

    def variable(self, storage, pointee, *, location=None, component=None, index=None,
                 flags=(), add_to_interface=True, initializer=None):
        ptr = self.pointer(storage, pointee)
        var = self.alloc()
        if initializer is None:
            self.emit(spv.OP_VARIABLE, ptr, var, storage)
        else:
            self.emit(spv.OP_VARIABLE, ptr, var, storage, initializer)
        if add_to_interface:
            self.interface_ids.append(var)
        for dec, val in ((spv.DEC_LOCATION, location), (spv.DEC_COMPONENT, component),
                         (spv.DEC_INDEX, index)):
            if val is not None:
                self.decorate(var, dec, val)
        for dec in flags:
            self.decorate(var, dec)
        return var

    def entry_point(self, name="main", ids=None, *, stage=None, model=None, func_id=None):
        if model is None:
            model = STAGE_MODELS[stage or self.stage]
        if func_id is None:
            func_id = self.alloc()
        if ids is None:
            ids = self.interface_ids
        args = (model, func_id, *encode_string_words(name), *ids)
        self.entries.extend(self._encode_inst(spv.OP_ENTRY_POINT, args))
        return func_id

    def build(self, *, big_endian=False, auto_entry=True, bound=None, schema=0):
        if auto_entry and not self.entries:
            self.entry_point()
        id_bound = self.next_id if bound is None else bound
        words = [spv.SPIRV_MAGIC, self.version, 0, id_bound, schema,
                 *self.prefix, *self.entries, *self.code]
        return pack_words(words, big_endian=big_endian)

    def reflect(self, stage=None, entry="main"):
        return spv.reflect_module(self.build(), stage=stage, entry=entry)


class HeaderAndInstructionStreamTest(unittest.TestCase):
    def test_parse_words_little_and_big_endian(self):
        header = [spv.SPIRV_MAGIC, 0x00010000, 0x1234, 4, 0]
        le = spv.parse_words(pack_words(header, big_endian=False))
        be = spv.parse_words(pack_words(header, big_endian=True))
        self.assertEqual(list(le), header)
        self.assertEqual(list(be), header)

    def test_parse_words_accepts_versions_1_0_through_1_6(self):
        for minor in range(7):
            version = 0x00010000 | (minor << 8)
            with self.subTest(version=hex(version)):
                words = spv.parse_words(pack_words([spv.SPIRV_MAGIC, version, 0, 2, 0]))
                self.assertEqual(words[1], version)

    def test_parse_words_rejects_malformed_headers(self):
        cases = [
            ("empty", b"", "five-word header"),
            ("truncated", pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 2]), "five-word header"),
            ("trailing_byte", pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 2, 0]) + b"\x00", "complete 32-bit"),
            ("bad_magic", pack_words([0xDEADBEEF, 0x00010000, 0, 2, 0]), "magic"),
            ("version_too_low", pack_words([spv.SPIRV_MAGIC, 0x0000FF00, 0, 2, 0]), "unsupported SPIR-V version"),
            ("version_too_high", pack_words([spv.SPIRV_MAGIC, 0x00010700, 0, 2, 0]), "unsupported SPIR-V version"),
            ("version_major_2", pack_words([spv.SPIRV_MAGIC, 0x00020000, 0, 2, 0]), "unsupported SPIR-V version"),
            ("version_nonzero_low_byte", pack_words([spv.SPIRV_MAGIC, 0x00010501, 0, 2, 0]), "unsupported SPIR-V version"),
            ("zero_bound", pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 0, 0]), "ID bound or reserved"),
            ("nonzero_reserved", pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 2, 1]), "ID bound or reserved"),
        ]
        for label, payload, pattern in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(spv.InterfaceError, pattern):
                    spv.parse_words(payload)

    def test_instructions_filters_declaration_ops_and_tracks_word_offsets(self):
        words = spv.parse_words(pack_words([
            spv.SPIRV_MAGIC, 0x00010500, 0, 10, 0,
            (2 << 16) | OP_CAPABILITY, 1,                 # word 5 (ignored)
            (3 << 16) | OP_MEMORY_MODEL, 0, 1,            # word 7 (ignored)
            (3 << 16) | spv.OP_TYPE_FLOAT, 1, 32,         # word 10 (declaration)
            (2 << 16) | OP_TYPE_VOID, 2,                  # word 13 (ignored)
            (3 << 16) | spv.OP_DECORATE, 3, spv.DEC_BLOCK,  # word 15 (declaration)
        ]))
        self.assertEqual(
            list(spv.instructions(words)),
            [
                (spv.OP_TYPE_FLOAT, (1, 32), 10),
                (spv.OP_DECORATE, (3, spv.DEC_BLOCK), 15),
            ],
        )

    def test_instructions_empty_body(self):
        words = spv.parse_words(pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 1, 0]))
        self.assertEqual(list(spv.instructions(words)), [])

    def test_instructions_rejects_zero_word_count_and_eof_overrun(self):
        zero_wc = spv.parse_words(pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 2, 0, 0]))
        with self.assertRaisesRegex(spv.InterfaceError, "invalid instruction length at word 5"):
            list(spv.instructions(zero_wc))

        overrun = spv.parse_words(pack_words([spv.SPIRV_MAGIC, 0x00010000, 0, 2, 0, (4 << 16) | spv.OP_TYPE_FLOAT, 1]))
        with self.assertRaisesRegex(spv.InterfaceError, "invalid instruction length at word 5"):
            list(spv.instructions(overrun))

    def test_decode_literal_string_valid_forms(self):
        cases = [
            ("", 1),
            ("a", 1),
            ("abc", 1),
            ("main", 2),         # exact 4-byte string needs a trailing zero word
            ("entry_point", 3),
            ("caf\u00e9", 2),    # multi-byte UTF-8
            ("\u03bb", 1),
        ]
        for text, expected_words in cases:
            with self.subTest(text=text):
                encoded = encode_string_words(text)
                decoded, used = spv.decode_literal_string(encoded + (99, 100))
                self.assertEqual(decoded, text)
                self.assertEqual(used, expected_words)

    def test_decode_literal_string_rejects_malformed_strings(self):
        # Nonzero padding byte after NUL terminator.
        bad_pad = struct.unpack("<I", b"a\x00\x01\x00")
        with self.assertRaisesRegex(spv.InterfaceError, "nonzero padding"):
            spv.decode_literal_string(bad_pad)

        # Invalid UTF-8 bytes before NUL.
        bad_utf8 = struct.unpack("<I", b"\xff\xfe\x00\x00")
        with self.assertRaisesRegex(spv.InterfaceError, "not UTF-8"):
            spv.decode_literal_string(bad_utf8)

        # Unterminated (no NUL byte).
        unterminated = struct.unpack("<2I", b"abcdefgh")
        with self.assertRaisesRegex(spv.InterfaceError, "unterminated"):
            spv.decode_literal_string(unterminated)

        with self.assertRaisesRegex(spv.InterfaceError, "unterminated"):
            spv.decode_literal_string(())


class TypesAndMasksTest(unittest.TestCase):
    def test_numeric_type_describe_and_masks(self):
        self.assertEqual(spv.NumericType("f", 32).describe(), "f32")
        self.assertEqual(spv.NumericType("i", 16, 3).describe(), "vec3<i16>")
        self.assertEqual(spv.NumericType("u", 64, 4).describe(), "vec4<u64>")

        # 16-bit and 32-bit occupy 1 component slot per element.
        self.assertEqual(spv.NumericType("f", 16, 1).location_masks(0), (0b0001,))
        self.assertEqual(spv.NumericType("f", 16, 2).location_masks(2), (0b1100,))
        self.assertEqual(spv.NumericType("u", 32, 3).location_masks(1), (0b1110,))
        self.assertEqual(spv.NumericType("i", 32, 4).location_masks(0), (0b1111,))

        # 64-bit types occupy 2 component slots per element.
        self.assertEqual(spv.NumericType("f", 64, 1).location_masks(0), (0b0011,))
        self.assertEqual(spv.NumericType("i", 64, 1).location_masks(2), (0b1100,))
        self.assertEqual(spv.NumericType("u", 64, 2).location_masks(0), (0b1111,))
        self.assertEqual(spv.NumericType("f", 64, 3).location_masks(0), (0b1111, 0b0011))
        self.assertEqual(spv.NumericType("f", 64, 4).location_masks(0), (0b1111, 0b1111))

    def test_numeric_type_mask_errors(self):
        with self.assertRaisesRegex(spv.InterfaceError, "extends beyond its Location"):
            spv.NumericType("f", 32, 2).location_masks(3)
        with self.assertRaisesRegex(spv.InterfaceError, "extends beyond its Location"):
            spv.NumericType("f", 16, 3).location_masks(2)
        for comp in (1, 3):
            with self.subTest(odd_component=comp):
                with self.assertRaisesRegex(spv.InterfaceError, "odd Component"):
                    spv.NumericType("f", 64, 1).location_masks(comp)
        for components in (3, 4):
            with self.subTest(wide_components=components):
                with self.assertRaisesRegex(spv.InterfaceError, "spanning two locations needs Component 0"):
                    spv.NumericType("f", 64, components).location_masks(2)

    def test_aggregate_type_describe_and_masks(self):
        vec2 = spv.NumericType("f", 32, 2)
        dvec3 = spv.NumericType("f", 64, 3)
        arr = spv.AggregateType("array", (vec2,), 3)
        mat = spv.AggregateType("matrix", (spv.NumericType("f", 32, 4),), 2)
        nested = spv.AggregateType("struct", (vec2, dvec3))

        self.assertEqual(arr.describe(), "array[3]<vec2<f32>>")
        self.assertEqual(mat.describe(), "matrix[2]<vec4<f32>>")
        self.assertEqual(nested.describe(), "struct{vec2<f32>, vec3<f64>}")

        self.assertEqual(arr.location_masks(2), (0b1100, 0b1100, 0b1100))
        self.assertEqual(mat.location_masks(0), (0b1111, 0b1111))
        self.assertEqual(nested.location_masks(0), (0b0011, 0b1111, 0b0011))

        with self.assertRaisesRegex(spv.InterfaceError, "nonzero Component"):
            nested.location_masks(1)

    def test_aggregate_type_expansion_limits(self):
        f32 = spv.NumericType("f", 32)
        at_limit = spv.AggregateType("array", (f32,), spv.MAX_EXPANDED_LOCATIONS)
        self.assertEqual(len(at_limit.location_masks(0)), spv.MAX_EXPANDED_LOCATIONS)

        over_array = spv.AggregateType("array", (f32,), spv.MAX_EXPANDED_LOCATIONS + 1)
        with self.assertRaisesRegex(spv.InterfaceError, "aggregate uses too many interface locations"):
            over_array.location_masks(0)

        half = spv.AggregateType("array", (f32,), spv.MAX_EXPANDED_LOCATIONS // 2 + 1)
        over_struct = spv.AggregateType("struct", (half, half))
        with self.assertRaisesRegex(spv.InterfaceError, "structure uses too many interface locations"):
            over_struct.location_masks(0)

    def test_address_varying_and_declaration_helpers(self):
        a0 = spv.Address(1, 0, 0)
        a1 = spv.Address(1, 0, 1)
        a2 = spv.Address(1, 2, 0)
        a3 = spv.Address(2, 0, 0)
        self.assertEqual(sorted([a3, a2, a1, a0]), [a0, a1, a2, a3])
        self.assertEqual(a0.describe(), "location 1, component 0")
        self.assertEqual(a1.describe(), "location 1, component 0, index 1")

        varying = spv.Varying(spv.Address(4, 0, 1), spv.NumericType("f", 64, 3), per_primitive=True)
        self.assertEqual(
            list(varying.occupied()),
            [
                spv.Address(4, 0, 1),
                spv.Address(4, 1, 1),
                spv.Address(4, 2, 1),
                spv.Address(4, 3, 1),
                spv.Address(5, 0, 1),
                spv.Address(5, 1, 1),
            ],
        )
        self.assertEqual(varying.describe(), "vec3<f64> per-primitive at location 4, component 0, index 1")

        plain_v = spv.Varying(spv.Address(0, 2), spv.NumericType("f", 32, 2))
        self.assertEqual(spv.Declaration((plain_v,), 7).describe(), "%7 (vec2<f32> at location 0, component 2)")
        self.assertEqual(
            spv.Declaration((plain_v,), 7, structure=True).describe(),
            "struct %7 (vec2<f32> at location 0, component 2)",
        )
        self.assertEqual(
            spv.Declaration((plain_v,), 7, structure=True, block=True).describe(),
            "block %7 (vec2<f32> at location 0, component 2)",
        )


class ModuleStructureAndDecorationsTest(unittest.TestCase):
    def test_rejects_ids_outside_bound_and_duplicate_definitions(self):
        # ID 0 definition.
        b = SpirvBuilder("vs")
        b.emit(spv.OP_TYPE_FLOAT, 0, 32)
        with self.assertRaisesRegex(spv.InterfaceError, "ID %0 is outside the SPIR-V bound"):
            spv.Module(b.build())

        # ID equal to bound.
        b = SpirvBuilder("vs")
        b.emit(spv.OP_TYPE_FLOAT, 99, 32)
        with self.assertRaisesRegex(spv.InterfaceError, "ID %99 is outside the SPIR-V bound"):
            spv.Module(b.build(bound=10))

        # Duplicate definition across two instructions.
        b = SpirvBuilder("vs")
        f32 = b.float_type(32)
        b.emit(spv.OP_TYPE_INT, f32, 32, 0)
        with self.assertRaisesRegex(spv.InterfaceError, f"duplicate definition of ID %{f32}"):
            spv.Module(b.build())

    def test_rejects_bad_operand_counts_on_declarations(self):
        cases = [
            ("OpTypeInt_short", spv.OP_TYPE_INT, (1, 32)),
            ("OpTypeInt_long", spv.OP_TYPE_INT, (1, 32, 0, 9)),
            ("OpTypeFloat_short", spv.OP_TYPE_FLOAT, (1,)),
            ("OpTypeFloat_long", spv.OP_TYPE_FLOAT, (1, 16, 0, 0)),
            ("OpTypeVector_short", spv.OP_TYPE_VECTOR, (1, 2)),
            ("OpTypeMatrix_long", spv.OP_TYPE_MATRIX, (1, 2, 3, 4)),
            ("OpTypeArray_short", spv.OP_TYPE_ARRAY, (1, 2)),
            ("OpTypePointer_short", spv.OP_TYPE_POINTER, (1, spv.STORAGE_OUTPUT)),
            ("OpTypeStruct_empty_operands", spv.OP_TYPE_STRUCT, ()),
            ("OpConstant_short", spv.OP_CONSTANT, (1, 2)),
            ("OpVariable_short", spv.OP_VARIABLE, (1, 2)),
            ("OpVariable_too_long", spv.OP_VARIABLE, (1, 2, spv.STORAGE_OUTPUT, 3, 4)),
            ("OpEntryPoint_short", spv.OP_ENTRY_POINT, (MODEL_VS, 1)),
            ("OpDecorate_short", spv.OP_DECORATE, (1,)),
            ("OpMemberDecorate_short", spv.OP_MEMBER_DECORATE, (1, 0)),
            ("OpDecorationGroup_empty", spv.OP_DECORATION_GROUP, ()),
            ("OpDecorationGroup_extra", spv.OP_DECORATION_GROUP, (1, 2)),
            ("OpGroupDecorate_short", spv.OP_GROUP_DECORATE, (1,)),
            ("OpGroupMemberDecorate_short", spv.OP_GROUP_MEMBER_DECORATE, (1, 2)),
        ]
        for label, opcode, args in cases:
            with self.subTest(case=label):
                b = SpirvBuilder("vs")
                b.emit(opcode, *args)
                with self.assertRaisesRegex(spv.InterfaceError, "invalid operands"):
                    spv.Module(b.build(bound=20, auto_entry=False))

    def test_variable_with_optional_initializer_is_accepted(self):
        b = SpirvBuilder("vs")
        f32 = b.float_type(32)
        zero = b.constant(f32, 0)
        var = b.variable(spv.STORAGE_OUTPUT, f32, location=3, initializer=zero)
        iface = b.reflect()
        self.assertEqual(len(iface.outputs), 1)
        self.assertEqual(iface.outputs[0].symbol, var)
        self.assertEqual(iface.outputs[0].varyings[0].address, spv.Address(3))

    def test_accepts_encoded_float_outside_interface(self):
        b = SpirvBuilder("vs")
        bf16 = b.emit_type(spv.OP_TYPE_FLOAT, 16, 0)  # BFloat16KHR
        b.constant(bf16, 0x3F00)
        b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=3)
        output, = b.reflect().outputs
        self.assertEqual(
            output.varyings,
            (spv.Varying(spv.Address(3), spv.NumericType("f", 32, 4)),),
        )

    def test_rejects_encoded_float_varying(self):
        b = SpirvBuilder("vs")
        bf16 = b.emit_type(spv.OP_TYPE_FLOAT, 16, 0)  # BFloat16KHR
        b.variable(spv.STORAGE_OUTPUT, bf16, location=3)
        with self.assertRaisesRegex(spv.InterfaceError, "unsupported floating-point encoding"):
            b.reflect()

    def test_ignores_non_interface_decorations_and_validates_known_ones(self):
        b = SpirvBuilder("ps")
        v = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 4), location=2)
        # Non-interface decorations with 0, 1, or more operands are ignored.
        b.decorate(v, DEC_FLAT)
        b.decorate(v, DEC_NO_PERSPECTIVE)
        b.decorate(v, DEC_DESCRIPTOR_SET, 0)
        b.decorate(v, DEC_BINDING, 7)
        b.decorate(v, DEC_OFFSET, 16)
        # Duplicate identical known decoration is idempotent.
        b.decorate(v, spv.DEC_LOCATION, 2)
        iface = b.reflect()
        self.assertEqual(len(iface.inputs), 1)
        self.assertEqual(iface.inputs[0].varyings[0].address, spv.Address(2))

        # Value decoration missing its operand.
        b_missing = SpirvBuilder("vs")
        var = b_missing.variable(spv.STORAGE_OUTPUT, b_missing.numeric("f", 32))
        b_missing.decorate(var, spv.DEC_LOCATION)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid Location decoration"):
            spv.Module(b_missing.build())

        # Value decoration with extra operands.
        b_extra = SpirvBuilder("vs")
        var = b_extra.variable(spv.STORAGE_OUTPUT, b_extra.numeric("f", 32))
        b_extra.decorate(var, spv.DEC_COMPONENT, 0, 1)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid Component decoration"):
            spv.Module(b_extra.build())

        # Flag decoration with unexpected operand.
        b_flag = SpirvBuilder("vs")
        s = b_flag.struct_type(b_flag.numeric("f", 32))
        b_flag.decorate(s, spv.DEC_BLOCK, 1)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid Block decoration"):
            spv.Module(b_flag.build())

        # Conflicting known decoration on same target.
        b_conflict = SpirvBuilder("vs")
        var = b_conflict.variable(spv.STORAGE_OUTPUT, b_conflict.numeric("f", 32), location=1)
        b_conflict.decorate(var, spv.DEC_LOCATION, 4)
        with self.assertRaisesRegex(spv.InterfaceError, "conflicting Location decorations"):
            spv.Module(b_conflict.build())

    def test_decoration_groups_apply_regardless_of_instruction_order(self):
        b = SpirvBuilder("vs")
        struct_id = b.struct_type(b.numeric("f", 32, 2), b.numeric("f", 32, 2))
        b.decorate(struct_id, spv.DEC_BLOCK)
        blk = b.variable(spv.STORAGE_OUTPUT, struct_id)
        var = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32))

        g_loc = b.decoration_group()
        g_empty = b.decoration_group()
        # Group use emitted BEFORE decorating the group itself.
        b.emit(spv.OP_GROUP_MEMBER_DECORATE, g_loc, struct_id, 0)
        b.emit(spv.OP_GROUP_DECORATE, g_empty, var)
        b.decorate(g_loc, spv.DEC_LOCATION, 5)
        b.member_decorate(struct_id, 1, spv.DEC_LOCATION, 6)
        b.decorate(var, spv.DEC_LOCATION, 5)
        b.decorate(var, spv.DEC_COMPONENT, 2)

        iface = b.reflect()
        self.assertEqual([d.symbol for d in iface.outputs], [blk, var])
        self.assertEqual([v.address for v in iface.outputs[0].varyings], [spv.Address(5, 0), spv.Address(6, 0)])
        self.assertEqual(iface.outputs[1].varyings[0].address, spv.Address(5, 2))

    def test_decoration_group_errors(self):
        # Undefined group.
        b = SpirvBuilder("vs")
        var = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32), location=0)
        missing_group = b.alloc()
        b.emit(spv.OP_GROUP_DECORATE, missing_group, var)
        with self.assertRaisesRegex(spv.InterfaceError, f"undefined decoration group %{missing_group}"):
            spv.Module(b.build())

        # Unpaired OpGroupMemberDecorate operands (even operand count: group + struct + member + dangling struct).
        b = SpirvBuilder("vs")
        s = b.struct_type(b.numeric("f", 32))
        g = b.decoration_group()
        b.emit(spv.OP_GROUP_MEMBER_DECORATE, g, s, 0, s)
        with self.assertRaisesRegex(spv.InterfaceError, "unpaired group member decoration"):
            spv.Module(b.build())

        # Decoration group targeting another decoration group.
        b = SpirvBuilder("vs")
        g1 = b.decoration_group()
        g2 = b.decoration_group()
        b.decorate(g1, spv.DEC_LOCATION, 0)
        b.emit(spv.OP_GROUP_DECORATE, g1, g2)
        with self.assertRaisesRegex(spv.InterfaceError, "decoration groups cannot decorate each other"):
            spv.Module(b.build())

        # Conflicting member decoration via group.
        b = SpirvBuilder("vs")
        s = b.struct_type(b.numeric("f", 32))
        b.member_decorate(s, 0, spv.DEC_LOCATION, 1)
        g = b.decoration_group()
        b.decorate(g, spv.DEC_LOCATION, 2)
        b.emit(spv.OP_GROUP_MEMBER_DECORATE, g, s, 0)
        with self.assertRaisesRegex(spv.InterfaceError, "conflicting Location decorations"):
            spv.Module(b.build())


class TypeAndConstantResolutionTest(unittest.TestCase):
    def test_array_length_constant_widths_and_signedness(self):
        cases = [
            ("u16", 16, False, (0x00010003,), 3),  # masks to 16 bits
            ("i16", 16, True, (5,), 5),
            ("u32", 32, False, (4,), 4),
            ("i32", 32, True, (2,), 2),
            ("u64", 64, False, (3, 0), 3),
            ("i64", 64, True, (6, 0), 6),
        ]
        for label, width, signed, words, expected in cases:
            with self.subTest(case=label):
                b = SpirvBuilder("vs")
                int_t = b.int_type(width, signed=signed)
                length_id = b.constant(int_t, *words)
                arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), length_id)
                b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
                v = b.reflect().outputs[0].varyings[0]
                self.assertEqual(v.type, spv.AggregateType("array", (spv.NumericType("f", 32),), expected))

    def test_array_length_accepts_int8_constants(self):
        for opcode in (spv.OP_CONSTANT, spv.OP_SPEC_CONSTANT):
            for signed, count in ((False, 0x82), (True, 5)):
                with self.subTest(opcode=opcode, signed=signed):
                    b = SpirvBuilder("vs")
                    length_type = b.int_type(8, signed=signed)
                    length_id = b.alloc()
                    b.emit(opcode, length_type, length_id, count)
                    arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.float_type(), length_id)
                    b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
                    varying = b.reflect().outputs[0].varyings[0]
                    self.assertEqual(varying.type, spv.AggregateType("array", (spv.NumericType("f", 32),), count))

    def test_array_length_constant_errors(self):
        # Not an OpConstant (references an undefined ID).
        b = SpirvBuilder("vs")
        arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), b.alloc())
        b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "not a literal OpConstant"):
            b.reflect()

        # Non-integer constant (float).
        b = SpirvBuilder("vs")
        f_const = b.constant(b.float_type(32), 0x3F800000)
        arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), f_const)
        b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "must be an integer"):
            b.reflect()

        # Unsupported integer width for array length.
        b = SpirvBuilder("vs")
        u24 = b.int_type(24)
        c = b.constant(u24, 3)
        arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), c)
        b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "must be an integer"):
            b.reflect()

        # Wrong literal word count (64-bit integer with only 1 word).
        b = SpirvBuilder("vs")
        u64 = b.int_type(64, signed=False)
        bad_lit = b.constant(u64, 2)
        arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), bad_lit)
        b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid literal width"):
            b.reflect()

        # Zero and negative signed counts.
        bad_counts = [
            ("zero_u8", 8, False, (0,)),
            ("neg_i8", 8, True, (0xFFFFFF80,)),
            ("zero_u32", 32, False, (0,)),
            ("neg_i16", 16, True, (0x8000,)),
            ("neg_i32", 32, True, (0xFFFFFFFF,)),
            ("neg_i64", 64, True, (0, 0x80000000)),
        ]
        for label, width, signed, literal in bad_counts:
            with self.subTest(case=label):
                b = SpirvBuilder("vs")
                c = b.constant(b.int_type(width, signed=signed), *literal)
                arr_t = b.emit_type(spv.OP_TYPE_ARRAY, b.numeric("f", 32), c)
                b.variable(spv.STORAGE_OUTPUT, arr_t, location=0)
                with self.assertRaisesRegex(spv.InterfaceError, "must be positive"):
                    b.reflect()

    def test_all_numeric_types_matrices_and_nested_aggregates(self):
        b = SpirvBuilder("vs")
        loc = 0
        expected = []
        for kind in ("f", "i", "u"):
            for width in (16, 32, 64):
                t = b.numeric(kind, width, 1)
                b.variable(spv.STORAGE_OUTPUT, t, location=loc)
                expected.append((loc, spv.NumericType(kind, width, 1)))
                loc += 1

        # Matrix of f32 and f16, 2D array, and nested struct inside array.
        mat = b.matrix(3, 4, 32)
        b.variable(spv.STORAGE_OUTPUT, mat, location=loc)
        expected.append((loc, spv.AggregateType("matrix", (spv.NumericType("f", 32, 4),), 3)))
        loc += 3

        arr2d = b.array(b.array(b.numeric("u", 16, 2), 2), 3)
        b.variable(spv.STORAGE_OUTPUT, arr2d, location=loc)
        expected.append((
            loc,
            spv.AggregateType("array", (spv.AggregateType("array", (spv.NumericType("u", 16, 2),), 2),), 3),
        ))
        loc += 6

        inner = b.struct_type(b.numeric("f", 32, 2), b.numeric("i", 32, 1))
        arr_of_struct = b.array(inner, 2)
        b.variable(spv.STORAGE_OUTPUT, arr_of_struct, location=loc)
        expected.append((
            loc,
            spv.AggregateType(
                "array",
                (spv.AggregateType("struct", (spv.NumericType("f", 32, 2), spv.NumericType("i", 32, 1))),),
                2,
            ),
        ))

        iface = b.reflect()
        got = [(d.varyings[0].address.location, d.varyings[0].type) for d in iface.outputs]
        self.assertEqual(got, expected)

    def test_shared_type_reuses_shape_cache_and_big_endian_round_trips(self):
        b = SpirvBuilder("vs")
        shared_vec4 = b.numeric("f", 32, 4)
        b.variable(spv.STORAGE_OUTPUT, shared_vec4, location=0)
        b.variable(spv.STORAGE_OUTPUT, shared_vec4, location=1)
        le_iface = spv.reflect_module(b.build(big_endian=False))
        be_iface = spv.reflect_module(b.build(big_endian=True))
        self.assertEqual(le_iface, be_iface)
        self.assertIs(le_iface.outputs[0].varyings[0].type, le_iface.outputs[1].varyings[0].type)

    def test_invalid_shapes_are_rejected(self):
        # Unsupported numeric width / signedness.
        for opcode, args in (
            (spv.OP_TYPE_FLOAT, (8,)),
            (spv.OP_TYPE_INT, (8, 0)),
            (spv.OP_TYPE_INT, (32, 2)),
        ):
            with self.subTest(bad_numeric=(opcode, args)):
                b = SpirvBuilder("vs")
                t = b.emit_type(opcode, *args)
                b.variable(spv.STORAGE_OUTPUT, t, location=0)
                with self.assertRaisesRegex(spv.InterfaceError, "unsupported numeric type"):
                    b.reflect()

        # Invalid vectors: components=1, components=5, vector of vector.
        for setup in (
            lambda b: b.emit_type(spv.OP_TYPE_VECTOR, b.float_type(32), 1),
            lambda b: b.emit_type(spv.OP_TYPE_VECTOR, b.float_type(32), 5),
            lambda b: b.emit_type(spv.OP_TYPE_VECTOR, b.numeric("f", 32, 2), 2),
        ):
            b = SpirvBuilder("vs")
            b.variable(spv.STORAGE_OUTPUT, setup(b), location=0)
            with self.assertRaisesRegex(spv.InterfaceError, "invalid varying vector type"):
                b.reflect()

        # Invalid matrices: integer column, scalar float column, cols=1, cols=5.
        for setup in (
            lambda b: b.emit_type(spv.OP_TYPE_MATRIX, b.numeric("i", 32, 2), 2),
            lambda b: b.emit_type(spv.OP_TYPE_MATRIX, b.float_type(32), 2),
            lambda b: b.emit_type(spv.OP_TYPE_MATRIX, b.numeric("f", 32, 2), 1),
            lambda b: b.emit_type(spv.OP_TYPE_MATRIX, b.numeric("f", 32, 2), 5),
        ):
            b = SpirvBuilder("vs")
            b.variable(spv.STORAGE_OUTPUT, setup(b), location=0)
            with self.assertRaisesRegex(spv.InterfaceError, "invalid varying matrix type"):
                b.reflect()

        # Pointer used as varying payload.
        b = SpirvBuilder("vs")
        inner_ptr = b.pointer(spv.STORAGE_OUTPUT, b.float_type(32))
        b.variable(spv.STORAGE_OUTPUT, inner_ptr, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "unsupported pointer"):
            b.reflect()

        # Missing/unknown type ID.
        b = SpirvBuilder("vs")
        b.variable(spv.STORAGE_OUTPUT, b.alloc(), location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "missing or unsupported varying type"):
            b.reflect()

        # Empty nested struct.
        b = SpirvBuilder("vs")
        empty_inner = b.struct_type()
        outer = b.struct_type(empty_inner)
        b.variable(spv.STORAGE_OUTPUT, outer, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "empty structure in graphics interface"):
            b.reflect()

        # Decorated nested struct member.
        b = SpirvBuilder("vs")
        inner = b.struct_type(b.float_type(32))
        b.member_decorate(inner, 0, spv.DEC_LOCATION, 0)
        outer = b.struct_type(inner)
        b.variable(spv.STORAGE_OUTPUT, outer, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "decorated nested interface structure members"):
            b.reflect()

    def test_cyclic_and_excessively_deep_types_are_rejected(self):
        # Cyclic type: array referencing itself via forward ID.
        b = SpirvBuilder("vs")
        self_arr = b.alloc()
        one = b.u32_const(1)
        b.emit(spv.OP_TYPE_ARRAY, self_arr, self_arr, one)
        b.variable(spv.STORAGE_OUTPUT, self_arr, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "cyclic or excessively deep"):
            b.reflect()

        # Excessively deep nesting (> MAX_TYPE_DEPTH).
        b = SpirvBuilder("vs")
        t = b.float_type(32)
        for _ in range(spv.MAX_TYPE_DEPTH + 1):
            t = b.array(t, 1)
        b.variable(spv.STORAGE_OUTPUT, t, location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "cyclic or excessively deep"):
            b.reflect()


class EntryPointSelectionTest(unittest.TestCase):
    def test_all_supported_stages_infer_and_match(self):
        for stage in ("vs", "gs", "ps", "ms"):
            with self.subTest(stage=stage):
                b = SpirvBuilder(stage)
                iface = b.reflect()
                self.assertEqual(iface.stage, stage)
                self.assertEqual(iface.entry, "main")
                self.assertEqual(iface.inputs, ())
                self.assertEqual(iface.outputs, ())
                self.assertEqual(b.reflect(stage=stage).stage, stage)

    def test_multiple_entry_points_and_stage_disambiguation(self):
        b = SpirvBuilder("vs")
        v_out = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=1, add_to_interface=False)
        p_in = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 2), location=3, add_to_interface=False)
        p_alt = b.variable(spv.STORAGE_INPUT, b.numeric("i", 32, 1), location=7, add_to_interface=False)
        b.entry_point("main", [v_out], stage="vs")
        b.entry_point("main", [p_in], stage="ps")
        b.entry_point("PixelAlt", [p_alt], stage="ps")

        data = b.build()
        with self.assertRaisesRegex(spv.InterfaceError, "ambiguous without a stage"):
            spv.reflect_module(data)

        vs_iface = spv.reflect_module(data, stage="vs")
        self.assertEqual(vs_iface.stage, "vs")
        self.assertEqual([d.symbol for d in vs_iface.outputs], [v_out])

        ps_iface = spv.reflect_module(data, stage="ps")
        self.assertEqual(ps_iface.stage, "ps")
        self.assertEqual([d.symbol for d in ps_iface.inputs], [p_in])

        alt_iface = spv.reflect_module(data, entry="PixelAlt")
        self.assertEqual(alt_iface.stage, "ps")
        self.assertEqual(alt_iface.entry, "PixelAlt")
        self.assertEqual([d.symbol for d in alt_iface.inputs], [p_alt])

    def test_entry_point_errors(self):
        # Missing entry point.
        b = SpirvBuilder("vs")
        with self.assertRaisesRegex(spv.InterfaceError, "entry point 'other' not found"):
            b.reflect(entry="other")

        # Wrong stage.
        with self.assertRaisesRegex(spv.InterfaceError, "entry point 'main' is vs, expected ps"):
            b.reflect(stage="ps")

        # Duplicate same name + same stage.
        b_dup = SpirvBuilder("vs")
        b_dup.entry_point("main", [], stage="vs")
        b_dup.entry_point("main", [], stage="vs")
        with self.assertRaisesRegex(spv.InterfaceError, "ambiguous"):
            b_dup.reflect(stage="vs")

        # Unsupported execution models (Compute, Tessellation, TaskEXT).
        for model in (MODEL_CS, MODEL_TESC, MODEL_TESE, MODEL_TASK_EXT):
            with self.subTest(model=model):
                b_unsup = SpirvBuilder("vs")
                b_unsup.entry_point("main", [], model=model)
                with self.assertRaisesRegex(spv.InterfaceError, f"unsupported execution model {model}"):
                    b_unsup.reflect()
                with self.assertRaisesRegex(spv.InterfaceError, f"is {model}, expected vs"):
                    b_unsup.reflect(stage="vs")

        # Entry point referencing a missing OpVariable.
        b_missing_var = SpirvBuilder("vs")
        ghost = b_missing_var.alloc()
        b_missing_var.entry_point("main", [ghost])
        with self.assertRaisesRegex(spv.InterfaceError, f"missing variable %{ghost}"):
            b_missing_var.reflect()

    def test_accepts_repeated_interface_ids_before_spirv_1_4(self):
        for version in (0x00010000, 0x00010300):
            with self.subTest(version=hex(version)):
                builder = SpirvBuilder("vs", version=version)
                scalar = builder.numeric("f", 32)
                first = builder.variable(spv.STORAGE_OUTPUT, scalar, location=0)
                second = builder.variable(spv.STORAGE_OUTPUT, scalar, location=1)
                builder.entry_point("main", [second, first, second, first])
                module = spv.Module(builder.build())
                self.assertEqual(module.entry_points[0].interface_ids, (second, first))
                self.assertEqual([d.symbol for d in builder.reflect().outputs], [first, second])

    def test_rejects_repeated_interface_ids_from_spirv_1_4(self):
        for version in (0x00010400, 0x00010500):
            with self.subTest(version=hex(version)):
                builder = SpirvBuilder("vs", version=version)
                output = builder.variable(spv.STORAGE_OUTPUT, builder.numeric("f", 32), location=0)
                builder.entry_point("main", [output, output])
                with self.assertRaisesRegex(spv.InterfaceError, "duplicate entry-point interface ID"):
                    builder.reflect()

    def test_ignores_resource_storage_classes_with_descriptor_bindings(self):
        b = SpirvBuilder("ps")
        opaque_type = b.alloc()  # Not even in TYPE_OPS; non-I/O variables are skipped before type lookup.
        for idx, storage in enumerate((
            STORAGE_UNIFORM_CONSTANT,
            STORAGE_UNIFORM,
            STORAGE_WORKGROUP,
            STORAGE_PRIVATE,
            STORAGE_PUSH_CONSTANT,
            STORAGE_STORAGE_BUFFER,
        )):
            var = b.variable(storage, opaque_type)
            b.decorate(var, DEC_DESCRIPTOR_SET, 0)
            b.decorate(var, DEC_BINDING, idx)
        color = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=0)

        iface = b.reflect()
        self.assertEqual(iface.inputs, ())
        self.assertEqual([d.symbol for d in iface.outputs], [color])


class StageInterfacesAndLayoutTest(unittest.TestCase):
    def test_inputs_and_outputs_sorted_and_separated(self):
        b = SpirvBuilder("vs")
        # Add in unsorted order, sharing locations across input and output directions.
        in_hi = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 2), location=4, component=2)
        in_lo = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 2), location=4, component=0)
        in_zero = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 4), location=0)
        out_hi = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=4)
        out_lo = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=0)

        # Mix in non-declaration instructions (OpCapability, OpExecutionMode, OpName, OpFunction).
        b.emit_prefix(OP_CAPABILITY, 1)
        b.emit_prefix(OP_MEMORY_MODEL, 0, 1)
        func_id = b.entry_point("main")
        b.emit(OP_EXECUTION_MODE, func_id, 7)
        b.emit(OP_NAME, out_lo, *encode_string_words("out0"))
        void_t = b.emit_type(OP_TYPE_VOID)
        fn_t = b.emit_type(OP_TYPE_FUNCTION, void_t)
        b.emit(OP_FUNCTION, void_t, func_id, 0, fn_t)
        b.emit(OP_FUNCTION_END)

        iface = b.reflect()
        self.assertEqual([d.symbol for d in iface.inputs], [in_zero, in_lo, in_hi])
        self.assertEqual([d.symbol for d in iface.outputs], [out_lo, out_hi])

    def test_component_packing_and_dual_source_outputs(self):
        b = SpirvBuilder("ps")
        # Pack four distinct components into location 1 (c=0, c=1..2, c=3).
        c3 = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 1), location=1, component=3)
        c1 = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 2), location=1, component=1)
        c0 = b.variable(spv.STORAGE_INPUT, b.numeric("f", 32, 1), location=1, component=0)
        # Dual-source outputs at location 0 with index 1 and index 0.
        src1 = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=0, index=1)
        src0 = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4), location=0, index=0)

        iface = b.reflect()
        self.assertEqual([d.symbol for d in iface.inputs], [c0, c1, c3])
        self.assertEqual([d.symbol for d in iface.outputs], [src0, src1])
        self.assertEqual([d.varyings[0].address for d in iface.outputs], [spv.Address(0, 0, 0), spv.Address(0, 0, 1)])

    def test_invalid_address_decorations_and_pointer_mismatch(self):
        # Missing Location.
        b = SpirvBuilder("vs")
        v = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32))
        with self.assertRaisesRegex(spv.InterfaceError, f"vs output variable %{v}: user varying is missing a Location"):
            b.reflect()

        # Component > 3.
        b = SpirvBuilder("vs")
        b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32), location=0, component=4)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid Component or Index"):
            b.reflect()

        # Index > 1.
        b = SpirvBuilder("ps")
        b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32), location=0, index=2)
        with self.assertRaisesRegex(spv.InterfaceError, "invalid Component or Index"):
            b.reflect()

        # Pointer storage class mismatch.
        b = SpirvBuilder("vs")
        wrong_ptr = b.pointer(spv.STORAGE_INPUT, b.numeric("f", 32))
        var = b.alloc()
        b.emit(spv.OP_VARIABLE, wrong_ptr, var, spv.STORAGE_OUTPUT)
        b.decorate(var, spv.DEC_LOCATION, 0)
        b.interface_ids.append(var)
        with self.assertRaisesRegex(spv.InterfaceError, "does not match its storage class"):
            b.reflect()

        # Non-pointer type on OpVariable.
        b = SpirvBuilder("vs")
        not_ptr = b.numeric("f", 32)
        var = b.alloc()
        b.emit(spv.OP_VARIABLE, not_ptr, var, spv.STORAGE_OUTPUT)
        b.decorate(var, spv.DEC_LOCATION, 0)
        b.interface_ids.append(var)
        with self.assertRaisesRegex(spv.InterfaceError, "does not match its storage class"):
            b.reflect()

    def test_builtins_skipped_and_mixed_builtin_struct_rejected(self):
        b = SpirvBuilder("vs")
        pos = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 4))
        b.decorate(pos, spv.DEC_BUILTIN, 0)

        builtin_blk = b.struct_type(b.numeric("f", 32, 4), b.numeric("f", 32, 1))
        b.decorate(builtin_blk, spv.DEC_BLOCK)
        b.member_decorate(builtin_blk, 0, spv.DEC_BUILTIN, 0)
        b.member_decorate(builtin_blk, 1, spv.DEC_BUILTIN, 1)
        b.variable(spv.STORAGE_OUTPUT, builtin_blk)

        user_out = b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 2), location=0)
        iface = b.reflect()
        self.assertEqual([d.symbol for d in iface.outputs], [user_out])

        # Mixed BuiltIn and user member in same struct is rejected.
        b_mixed = SpirvBuilder("vs")
        mixed = b_mixed.struct_type(b_mixed.numeric("f", 32, 4), b_mixed.numeric("f", 32, 2))
        b_mixed.decorate(mixed, spv.DEC_BLOCK)
        b_mixed.member_decorate(mixed, 0, spv.DEC_BUILTIN, 0)
        b_mixed.member_decorate(mixed, 1, spv.DEC_LOCATION, 0)
        b_mixed.variable(spv.STORAGE_OUTPUT, mixed)
        with self.assertRaisesRegex(spv.InterfaceError, "mixed built-in and user-defined"):
            b_mixed.reflect()

    def test_struct_and_block_layouts(self):
        # Block with explicit member locations, components, and per-member PerPrimitiveEXT.
        b = SpirvBuilder("ps")
        blk_t = b.struct_type(b.numeric("f", 32, 2), b.numeric("f", 32, 2), b.numeric("u", 32, 1))
        b.decorate(blk_t, spv.DEC_BLOCK)
        b.member_decorate(blk_t, 0, spv.DEC_LOCATION, 2)
        b.member_decorate(blk_t, 0, spv.DEC_COMPONENT, 0)
        b.member_decorate(blk_t, 1, spv.DEC_LOCATION, 2)
        b.member_decorate(blk_t, 1, spv.DEC_COMPONENT, 2)
        b.member_decorate(blk_t, 2, spv.DEC_LOCATION, 5)
        b.member_decorate(blk_t, 2, spv.DEC_PER_PRIMITIVE)
        blk_var = b.variable(spv.STORAGE_INPUT, blk_t)

        # Non-block struct with variable-level Location auto-assigning multi-location members.
        s_t = b.struct_type(
            b.numeric("f", 64, 3),        # 2 locations: 10..11
            b.matrix(2, 4, 32),           # 2 locations: 12..13
            b.numeric("f", 32, 1),        # 1 location: 14
        )
        s_var = b.variable(spv.STORAGE_INPUT, s_t, location=10)

        iface = b.reflect()
        self.assertEqual([d.symbol for d in iface.inputs], [blk_var, s_var])

        blk_decl, s_decl = iface.inputs
        self.assertTrue(blk_decl.structure)
        self.assertTrue(blk_decl.block)
        self.assertEqual(
            [(v.address, v.per_primitive) for v in blk_decl.varyings],
            [(spv.Address(2, 0), False), (spv.Address(2, 2), False), (spv.Address(5, 0), True)],
        )

        self.assertTrue(s_decl.structure)
        self.assertFalse(s_decl.block)
        self.assertEqual([v.address for v in s_decl.varyings], [spv.Address(10), spv.Address(12), spv.Address(14)])

    def test_struct_and_block_error_cases(self):
        # Non-block struct without Location on variable.
        b = SpirvBuilder("vs")
        s = b.struct_type(b.numeric("f", 32))
        b.member_decorate(s, 0, spv.DEC_LOCATION, 0)
        b.variable(spv.STORAGE_OUTPUT, s)
        with self.assertRaisesRegex(spv.InterfaceError, "non-block structure needs a Location"):
            b.reflect()

        # Component or Index on entire structure variable.
        for kw in ({"component": 1}, {"index": 1}):
            with self.subTest(struct_dec=kw):
                b = SpirvBuilder("ps")
                s = b.struct_type(b.numeric("f", 32))
                b.variable(spv.STORAGE_OUTPUT, s, location=0, **kw)
                with self.assertRaisesRegex(spv.InterfaceError, "Component or Index on an entire varying structure"):
                    b.reflect()

        # Both struct variable and a member declare Location.
        b = SpirvBuilder("vs")
        s = b.struct_type(b.numeric("f", 32))
        b.decorate(s, spv.DEC_BLOCK)
        b.member_decorate(s, 0, spv.DEC_LOCATION, 2)
        b.variable(spv.STORAGE_OUTPUT, s, location=1)
        with self.assertRaisesRegex(spv.InterfaceError, "both the structure and its members declare Locations"):
            b.reflect()

        # Empty top-level Block struct.
        b = SpirvBuilder("vs")
        empty = b.struct_type()
        b.decorate(empty, spv.DEC_BLOCK)
        b.variable(spv.STORAGE_OUTPUT, empty)
        with self.assertRaisesRegex(spv.InterfaceError, "empty structure in graphics interface"):
            b.reflect()

        # Block member missing Location when block variable has no Location.
        b = SpirvBuilder("vs")
        s = b.struct_type(b.numeric("f", 32), b.numeric("f", 32))
        b.decorate(s, spv.DEC_BLOCK)
        b.member_decorate(s, 0, spv.DEC_LOCATION, 0)
        b.variable(spv.STORAGE_OUTPUT, s)
        with self.assertRaisesRegex(spv.InterfaceError, "missing a Location"):
            b.reflect()

    def test_stage_indexed_arrays_per_vertex_and_per_primitive(self):
        # GS input requires outer array and strips it; GS output does not.
        gs = SpirvBuilder("gs")
        gs_in = gs.variable(spv.STORAGE_INPUT, gs.array(gs.numeric("f", 32, 4), 3), location=0)
        gs_out = gs.variable(spv.STORAGE_OUTPUT, gs.numeric("f", 32, 4), location=0)
        gs_iface = gs.reflect()
        self.assertEqual(gs_iface.inputs[0].symbol, gs_in)
        self.assertTrue(gs_iface.inputs[0].indexed_by_vertex_or_primitive)
        self.assertEqual(gs_iface.inputs[0].varyings[0].type, spv.NumericType("f", 32, 4))
        self.assertEqual(gs_iface.outputs[0].symbol, gs_out)
        self.assertFalse(gs_iface.outputs[0].indexed_by_vertex_or_primitive)

        # GS input without outer array is rejected.
        gs_bad = SpirvBuilder("gs")
        gs_bad.variable(spv.STORAGE_INPUT, gs_bad.numeric("f", 32, 4), location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "gs requires an outer per-vertex or per-primitive array"):
            gs_bad.reflect()

        # MS output requires outer array and supports PerPrimitiveEXT (including on blocks).
        ms = SpirvBuilder("ms")
        prim_blk = ms.struct_type(ms.numeric("u", 32, 1))
        ms.decorate(prim_blk, spv.DEC_BLOCK)
        ms.member_decorate(prim_blk, 0, spv.DEC_LOCATION, 2)
        ms.variable(spv.STORAGE_OUTPUT, ms.array(prim_blk, 64), flags=(spv.DEC_PER_PRIMITIVE,))
        ms.variable(spv.STORAGE_OUTPUT, ms.array(ms.numeric("f", 32, 4), 32), location=0)
        ms_iface = ms.reflect()
        self.assertTrue(all(d.indexed_by_vertex_or_primitive for d in ms_iface.outputs))
        self.assertFalse(ms_iface.outputs[0].varyings[0].per_primitive)
        self.assertTrue(ms_iface.outputs[1].varyings[0].per_primitive)

        # MS output without outer array is rejected.
        ms_bad = SpirvBuilder("ms")
        ms_bad.variable(spv.STORAGE_OUTPUT, ms_bad.numeric("f", 32, 4), location=0)
        with self.assertRaisesRegex(spv.InterfaceError, "ms requires an outer per-vertex or per-primitive array"):
            ms_bad.reflect()

        # PS input with PerVertexKHR strips outer array; without array is rejected.
        ps = SpirvBuilder("ps")
        ps.variable(spv.STORAGE_INPUT, ps.array(ps.numeric("f", 32, 3), 3), location=1, flags=(spv.DEC_PER_VERTEX,))
        ps_iface = ps.reflect()
        self.assertTrue(ps_iface.inputs[0].indexed_by_vertex_or_primitive)
        self.assertEqual(ps_iface.inputs[0].varyings[0].type, spv.NumericType("f", 32, 3))

        ps_bad = SpirvBuilder("ps")
        ps_bad.variable(spv.STORAGE_INPUT, ps_bad.numeric("f", 32, 3), location=1, flags=(spv.DEC_PER_VERTEX,))
        with self.assertRaisesRegex(spv.InterfaceError, "ps requires an outer per-vertex or per-primitive array"):
            ps_bad.reflect()

        # PerVertexKHR is rejected on non-(ps, input) variables.
        for stage, storage in (("vs", spv.STORAGE_INPUT), ("vs", spv.STORAGE_OUTPUT), ("ps", spv.STORAGE_OUTPUT)):
            with self.subTest(per_vertex_invalid=(stage, storage)):
                bad = SpirvBuilder(stage)
                bad.variable(storage, bad.array(bad.numeric("f", 32), 3), location=0, flags=(spv.DEC_PER_VERTEX,))
                with self.assertRaisesRegex(spv.InterfaceError, "PerVertexKHR is only supported on fragment inputs"):
                    bad.reflect()

    def test_unsupported_modes_passthrough_and_patch(self):
        for dec, pattern in ((spv.DEC_PASSTHROUGH_NV, "passthrough"), (spv.DEC_PATCH, "tessellation patch")):
            with self.subTest(variable_mode=pattern):
                b = SpirvBuilder("vs")
                b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32), location=0, flags=(dec,))
                with self.assertRaisesRegex(spv.InterfaceError, pattern):
                    b.reflect()

            with self.subTest(member_mode=pattern):
                b = SpirvBuilder("vs")
                s = b.struct_type(b.numeric("f", 32))
                b.decorate(s, spv.DEC_BLOCK)
                b.member_decorate(s, 0, spv.DEC_LOCATION, 0)
                b.member_decorate(s, 0, dec)
                b.variable(spv.STORAGE_OUTPUT, s)
                with self.assertRaisesRegex(spv.InterfaceError, pattern):
                    b.reflect()

    def test_overlap_detection_across_scalars_aggregates_structs_and_indices(self):
        # Matrix at loc 2..4 overlaps scalar at loc 4.
        b = SpirvBuilder("vs")
        b.variable(spv.STORAGE_OUTPUT, b.matrix(3, 4, 32), location=2)
        b.variable(spv.STORAGE_OUTPUT, b.numeric("f", 32, 1), location=4)
        with self.assertRaisesRegex(spv.InterfaceError, "overlap.*location 4, component 0"):
            b.reflect()

        # Array at component 2 leaves components 0..1 free, but overlaps component 3.
        b_ok = SpirvBuilder("vs")
        b_ok.variable(spv.STORAGE_OUTPUT, b_ok.array(b_ok.numeric("f", 32, 2), 2), location=1, component=2)
        b_ok.variable(spv.STORAGE_OUTPUT, b_ok.numeric("f", 32, 2), location=2, component=0)
        self.assertEqual(len(b_ok.reflect().outputs), 2)

        b_hit = SpirvBuilder("vs")
        b_hit.variable(spv.STORAGE_OUTPUT, b_hit.array(b_hit.numeric("f", 32, 2), 2), location=1, component=2)
        b_hit.variable(spv.STORAGE_OUTPUT, b_hit.numeric("f", 32, 1), location=2, component=3)
        with self.assertRaisesRegex(spv.InterfaceError, "overlap.*location 2, component 3"):
            b_hit.reflect()

        # Intra-struct overlap between two block members.
        b_blk = SpirvBuilder("ps")
        blk = b_blk.struct_type(b_blk.numeric("f", 32, 3), b_blk.numeric("f", 32, 2))
        b_blk.decorate(blk, spv.DEC_BLOCK)
        b_blk.member_decorate(blk, 0, spv.DEC_LOCATION, 1)
        b_blk.member_decorate(blk, 1, spv.DEC_LOCATION, 1)
        b_blk.member_decorate(blk, 1, spv.DEC_COMPONENT, 2)
        b_blk.variable(spv.STORAGE_INPUT, blk)
        with self.assertRaisesRegex(spv.InterfaceError, "ps input variable.*overlap.*location 1, component 2"):
            b_blk.reflect()

        # Duplicate dual-source index 1 at same location overlaps.
        b_ds = SpirvBuilder("ps")
        b_ds.variable(spv.STORAGE_OUTPUT, b_ds.numeric("f", 32, 4), location=0, index=1)
        b_ds.variable(spv.STORAGE_OUTPUT, b_ds.numeric("f", 32, 1), location=0, component=2, index=1)
        with self.assertRaisesRegex(spv.InterfaceError, "overlap.*location 0, component 2, index 1"):
            b_ds.reflect()


if __name__ == "__main__":
    unittest.main()
