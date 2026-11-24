"""Shader-stage matching with hand-written modules and a synthetic pipeline manifest."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import shader_link_check as link  # noqa: E402
import spirv_interface as spv  # noqa: E402
import spirv_stubs  # noqa: E402


class ModuleFixture:
    """Binary interface declarations. Drivers are exercised by separate GLSL fixtures."""

    MODELS = {"vs": 0, "gs": 3, "ms": 5365, "ps": 4, "cs": 5}

    def __init__(self, stage, start_id=1):
        self.stage = stage
        self.next_id = start_id
        self.code = []
        self.entries = []
        self.ids = []
        self.uint = self.emit_type(spv.OP_TYPE_INT, 32, 0)

    def allocate(self):
        self.next_id += 1
        return self.next_id - 1

    def emit(self, opcode, *args):
        self.code.extend([((len(args) + 1) << 16) | opcode, *args])

    def emit_type(self, opcode, *args):
        result = self.allocate()
        self.emit(opcode, result, *args)
        return result

    def literal(self, value):
        result = self.allocate()
        self.emit(spv.OP_CONSTANT, self.uint, result, value)
        return result

    def numeric(self, width=32, components=1, kind="f"):
        if kind == "f":
            element = self.emit_type(spv.OP_TYPE_FLOAT, width)
        else:
            element = self.emit_type(spv.OP_TYPE_INT, width, int(kind == "i"))
        return element if components == 1 else self.emit_type(spv.OP_TYPE_VECTOR, element, components)

    def array(self, element, length):
        return self.emit_type(spv.OP_TYPE_ARRAY, element, self.literal(length))

    def decorate(self, identifier, decoration, *operands):
        self.emit(spv.OP_DECORATE, identifier, decoration, *operands)

    def member(self, identifier, member, decoration, *operands):
        self.emit(spv.OP_MEMBER_DECORATE, identifier, member, decoration, *operands)

    def variable(self, storage, shape, location=None, component=None, index=None, *flags):
        pointer = self.emit_type(spv.OP_TYPE_POINTER, storage, shape)
        identifier = self.allocate()
        self.emit(spv.OP_VARIABLE, pointer, identifier, storage)
        self.ids.append(identifier)
        for decoration, value in ((spv.DEC_LOCATION, location), (spv.DEC_COMPONENT, component),
                                  (spv.DEC_INDEX, index)):
            if value is not None:
                self.decorate(identifier, decoration, value)
        for decoration in flags:
            self.decorate(identifier, decoration)
        return identifier

    def entry(self, name, ids, stage=None, *, model=None):
        raw = name.encode("utf-8") + b"\x00"
        raw += bytes(-len(raw) % 4)
        string = struct.unpack(f"<{len(raw) // 4}I", raw)
        function = self.allocate()
        if model is None:
            model = self.MODELS[stage or self.stage]
        args = (model, function, *string, *ids)
        self.entries.extend([((len(args) + 1) << 16) | spv.OP_ENTRY_POINT, *args])

    def binary(self, big_endian=False):
        if not self.entries:
            self.entry("main", self.ids)
        words = [spv.SPIRV_MAGIC, 0x00010500, 0, self.next_id, 0, *self.entries, *self.code]
        endian = ">" if big_endian else "<"
        return struct.pack(f"{endian}{len(words)}I", *words)

    def reflect(self, entry="main", **kwargs):
        return spv.reflect_module(self.binary(), entry=entry, **kwargs)


def simple(stage, storage, location, component=None, width=32, components=4, kind="f", array_count=None):
    shader = ModuleFixture(stage, start_id=5 if storage == spv.STORAGE_INPUT else 1)
    shape = shader.numeric(width, components, kind)
    if array_count is not None:
        shape = shader.array(shape, array_count)
    shader.variable(storage, shape, location, component)
    return shader


def output(location=0, **kwargs):
    return simple("vs", spv.STORAGE_OUTPUT, location, **kwargs)


def input(location=0, **kwargs):
    return simple("ps", spv.STORAGE_INPUT, location, **kwargs)


def kinds(producer, consumer, **kwargs):
    return [item.kind for item in link.compare_stages(producer.reflect(), consumer.reflect(), **kwargs)]


class ReflectTest(unittest.TestCase):
    def test_reads_other_byte_order(self):
        shader = output()
        self.assertEqual(spv.reflect_module(shader.binary(big_endian=True)), shader.reflect())

    def test_skips_correctly_decorated_builtins(self):
        shader = ModuleFixture("vs")
        position = shader.variable(spv.STORAGE_OUTPUT, shader.numeric(32, 4))
        shader.decorate(position, spv.DEC_BUILTIN, 0)
        self.assertEqual(shader.reflect().outputs, ())

    def test_skips_builtin_block(self):
        shader = ModuleFixture("gs")
        block = shader.emit_type(spv.OP_TYPE_STRUCT, shader.numeric(32, 4))
        shader.decorate(block, spv.DEC_BLOCK)
        shader.member(block, 0, spv.DEC_BUILTIN, 0)
        shader.variable(spv.STORAGE_INPUT, shader.array(block, 3))
        self.assertEqual(shader.reflect().inputs, ())

    def test_missing_location_is_an_error(self):
        shader = ModuleFixture("vs")
        shader.variable(spv.STORAGE_OUTPUT, shader.numeric())
        with self.assertRaisesRegex(spv.InterfaceError, "missing a Location"):
            shader.reflect()

    def test_64_bit_vector_spans_locations_and_can_share_remaining_components(self):
        shader = output(width=64, components=3)
        shader.variable(spv.STORAGE_OUTPUT, shader.numeric(32, 2), 1, 2)
        declarations = shader.reflect().outputs
        self.assertEqual(len(declarations), 2)
        used = list(declarations[0].varyings[0].occupied())
        self.assertEqual(used[-1], spv.Address(1, 1))

    def test_rejects_component_overlap(self):
        shader = output(width=64, components=3)
        shader.variable(spv.STORAGE_OUTPUT, shader.numeric(), 1, 1)
        with self.assertRaisesRegex(spv.InterfaceError, "overlap.*location 1, component 1"):
            shader.reflect()

    def test_keeps_fragment_dual_source_outputs_separate(self):
        shader = ModuleFixture("ps")
        shape = shader.numeric(32, 4)
        shader.variable(spv.STORAGE_OUTPUT, shape, 0, index=0)
        shader.variable(spv.STORAGE_OUTPUT, shape, 0, index=1)
        self.assertEqual([d.varyings[0].address.index for d in shader.reflect().outputs], [0, 1])

    def test_rejects_vector_that_spills_32_bits_into_another_location(self):
        shader = input(component=2, components=3)
        with self.assertRaisesRegex(spv.InterfaceError, "extends beyond its Location"):
            shader.reflect()

    def test_rejects_64_bit_vec2_at_component_2(self):
        shader = output(width=64, components=2, component=2)
        with self.assertRaisesRegex(spv.InterfaceError, "Component 0"):
            shader.reflect()

    def test_allows_64_bit_scalar_at_component_2(self):
        shader = output(width=64, components=1, component=2)
        used = list(shader.reflect().outputs[0].varyings[0].occupied())
        self.assertEqual(used, [spv.Address(0, 2), spv.Address(0, 3)])

    def test_rejects_unbounded_expansion(self):
        shader = output(array_count=spv.MAX_EXPANDED_LOCATIONS + 1)
        with self.assertRaisesRegex(spv.InterfaceError, "too many interface locations"):
            shader.reflect()

    def test_uses_selected_entry_interface_ids(self):
        shader = ModuleFixture("ps")
        first = shader.variable(spv.STORAGE_INPUT, shader.numeric(), 2)
        second = shader.variable(spv.STORAGE_INPUT, shader.numeric(), 6)
        shader.entry("main", [first])
        shader.entry("other", [second])
        self.assertEqual(shader.reflect().inputs[0].varyings[0].address.location, 2)
        self.assertEqual(shader.reflect("other").inputs[0].varyings[0].address.location, 6)

    def test_same_name_in_different_stages_requires_stage_to_select(self):
        shader = ModuleFixture("vs")
        shader.entry("main", [])
        shader.entry("main", [], stage="ps")
        with self.assertRaisesRegex(spv.InterfaceError, "ambiguous"):
            shader.reflect()
        self.assertEqual(shader.reflect(stage="ps").stage, "ps")

    def test_entry_can_reference_resources_other_than_io(self):
        shader = ModuleFixture("ps")
        shader.variable(2, shader.numeric())  # Uniform, a different interface.
        self.assertEqual(shader.reflect().inputs, ())

    def test_grouped_decorations_apply_to_members_and_variables(self):
        shader = ModuleFixture("vs")
        struct_type = shader.emit_type(spv.OP_TYPE_STRUCT, shader.numeric(32, 2))
        shader.decorate(struct_type, spv.DEC_BLOCK)
        block = shader.variable(spv.STORAGE_OUTPUT, struct_type)
        scalar = shader.variable(spv.STORAGE_OUTPUT, shader.numeric())
        group = shader.allocate()
        shader.emit(spv.OP_DECORATION_GROUP, group)
        shader.decorate(group, spv.DEC_LOCATION, 2)
        shader.emit(spv.OP_GROUP_MEMBER_DECORATE, group, struct_type, 0)
        shader.emit(spv.OP_GROUP_DECORATE, group, scalar)
        component_group = shader.allocate()
        shader.emit(spv.OP_DECORATION_GROUP, component_group)
        shader.decorate(component_group, spv.DEC_COMPONENT, 2)
        shader.emit(spv.OP_GROUP_DECORATE, component_group, scalar)
        declarations = shader.reflect().outputs
        self.assertEqual([d.symbol for d in declarations], [block, scalar])
        self.assertEqual([d.varyings[0].address for d in declarations], [spv.Address(2), spv.Address(2, 2)])

    def test_conflicting_decoration_group_is_rejected(self):
        shader = output()
        group = shader.allocate()
        shader.emit(spv.OP_DECORATION_GROUP, group)
        shader.decorate(group, spv.DEC_LOCATION, 5)
        shader.emit(spv.OP_GROUP_DECORATE, group, shader.ids[0])
        with self.assertRaisesRegex(spv.InterfaceError, "conflicting Location"):
            shader.reflect()

    def test_malformed_header_instruction_and_references(self):
        shader = output()
        with self.assertRaisesRegex(spv.InterfaceError, "complete"):
            spv.reflect_module(shader.binary() + b"\x00")
        with self.assertRaisesRegex(spv.InterfaceError, "magic"):
            spv.reflect_module(b"\x00" * 4 + shader.binary()[4:])
        with self.assertRaisesRegex(spv.InterfaceError, "instruction length"):
            spv.reflect_module(shader.binary()[:-4])
        with self.assertRaisesRegex(spv.InterfaceError, "instruction length"):
            spv.reflect_module(shader.binary() + bytes(4))
        missing = ModuleFixture("vs")
        missing.entry("main", [missing.allocate()])
        with self.assertRaisesRegex(spv.InterfaceError, "missing variable"):
            missing.reflect()

    def test_passthrough_geometry_gets_explicit_unsupported_diagnostic(self):
        shader = simple("gs", spv.STORAGE_INPUT, 0, array_count=3)
        shader.decorate(shader.ids[0], spv.DEC_PASSTHROUGH_NV)
        with self.assertRaisesRegex(spv.InterfaceError, "passthrough"):
            shader.reflect()


class LinkingTest(unittest.TestCase):
    def test_matches_the_projects_generated_partner_modules(self):
        vertex = spv.reflect_module(spirv_stubs.build_vs_stub([2, 9]), stage="vs")
        pixel = spv.reflect_module(spirv_stubs.build_ps_stub([9, 2], flat_locations=[2]), stage="ps")
        self.assertEqual(link.compare_stages(vertex, pixel), [])

    def test_numeric_match_ignores_module_local_ids_and_optional_component_zero(self):
        self.assertEqual(kinds(output(), input(component=0)), [])

    def test_unconsumed_outputs_are_allowed(self):
        producer = output()
        producer.variable(spv.STORAGE_OUTPUT, producer.numeric(), 7)
        self.assertEqual(kinds(producer, input()), [])

    def test_missing_output_and_conflicting_start_are_distinct(self):
        self.assertEqual(kinds(output(), input(3)), ["missing-output"])
        self.assertEqual(kinds(output(), input(component=2, components=2)), ["location-mismatch"])

    def test_float_width_and_integer_sign_are_part_of_type(self):
        self.assertEqual(kinds(output(components=2, width=16), input(components=2)), ["type-mismatch"])
        self.assertEqual(kinds(output(components=2, kind="u"), input(components=2, kind="i")), ["type-mismatch"])

    def test_maintenance4_relaxes_only_a_wider_vector_output(self):
        self.assertEqual(kinds(output(), input(components=2)), ["type-mismatch"])
        self.assertEqual(kinds(output(), input(components=2), maintenance4=True), [])
        self.assertEqual(kinds(output(components=2), input(), maintenance4=True), ["type-mismatch"])
        self.assertEqual(kinds(output(), input(components=1), maintenance4=True), ["type-mismatch"])

    def test_maintenance4_does_not_unwrap_and_relax_aggregate_shapes(self):
        producer = output(array_count=3)
        consumer = input(components=2, array_count=3)
        self.assertEqual(kinds(producer, consumer, maintenance4=True), ["type-mismatch"])
        self.assertEqual(kinds(output(), simple("gs", spv.STORAGE_INPUT, 0, components=2, array_count=3),
                               maintenance4=True), ["type-mismatch"])

    def test_geometry_input_strips_outer_array_only(self):
        geometry = ModuleFixture("gs")
        member = geometry.array(geometry.numeric(32, 4), 2)
        geometry.variable(spv.STORAGE_INPUT, geometry.array(member, 3), 0)
        self.assertEqual(kinds(output(array_count=2), geometry), [])
        self.assertEqual(kinds(output(array_count=3), geometry), ["type-mismatch"])

    def test_fragment_per_vertex_matches_non_array_vertex_output(self):
        fragment = simple("ps", spv.STORAGE_INPUT, 0, array_count=3)
        fragment.decorate(fragment.ids[0], spv.DEC_PER_VERTEX)
        self.assertEqual(kinds(output(), fragment), [])
        self.assertEqual(kinds(output(array_count=3), fragment), ["type-mismatch"])

    def test_mesh_nv_and_ext_outputs_match_fragment_domains(self):
        for name, model in (("MeshNV", 5268), ("MeshEXT", 5365)):
            with self.subTest(model=name):
                mesh = ModuleFixture("ms")
                vec4 = mesh.numeric(32, 4)
                mesh.variable(spv.STORAGE_OUTPUT, mesh.array(vec4, 3), location=0)
                primitive = mesh.variable(spv.STORAGE_OUTPUT, mesh.array(vec4, 1), location=1)
                mesh.decorate(primitive, spv.DEC_PER_PRIMITIVE)
                mesh.entry("main", mesh.ids, model=model)

                reflected = mesh.reflect()
                self.assertEqual(reflected.stage, "ms")
                self.assertEqual(mesh.reflect(stage="ms"), reflected)
                self.assertTrue(all(d.indexed_by_vertex_or_primitive for d in reflected.outputs))
                shape = spv.NumericType("f", 32, 4)
                self.assertEqual(
                    [d.varyings for d in reflected.outputs],
                    [(spv.Varying(spv.Address(0), shape),), (spv.Varying(spv.Address(1), shape, True),)],
                )

                fragment = input()
                fragment_primitive = fragment.variable(spv.STORAGE_INPUT, fragment.numeric(32, 4), location=1)
                self.assertEqual(kinds(mesh, fragment), ["per-primitive-mismatch"])
                fragment.decorate(fragment_primitive, spv.DEC_PER_PRIMITIVE)
                self.assertEqual(kinds(mesh, fragment), [])

    def test_matching_structure_members_keep_order_and_locations(self):
        def block(stage, storage, locations):
            shader = ModuleFixture(stage)
            shape = shader.emit_type(spv.OP_TYPE_STRUCT, shader.numeric(32, 2), shader.numeric(32, 2))
            shader.decorate(shape, spv.DEC_BLOCK)
            for i, location in enumerate(locations):
                shader.member(shape, i, spv.DEC_LOCATION, location)
            shader.variable(storage, shape)
            return shader

        producer = block("vs", spv.STORAGE_OUTPUT, (1, 3))
        self.assertEqual(kinds(producer, block("ps", spv.STORAGE_INPUT, (1, 3))), [])
        self.assertEqual(kinds(producer, block("ps", spv.STORAGE_INPUT, (1, 2))), ["location-mismatch"])
        self.assertEqual(kinds(producer, block("ps", spv.STORAGE_INPUT, (3, 1))), ["location-mismatch"])
        self.assertEqual(kinds(producer, input(location=1, components=2)), ["structure-mismatch"])

    def test_location_on_structure_allocates_consecutive_members(self):
        shader = ModuleFixture("vs")
        wide = shader.numeric(64, 3)
        shape = shader.emit_type(spv.OP_TYPE_STRUCT, wide, shader.numeric())
        shader.variable(spv.STORAGE_OUTPUT, shape, 5)
        declarations = shader.reflect().outputs
        self.assertEqual([v.address.location for v in declarations[0].varyings], [5, 7])

    def test_array_length_uses_literal_specialization_default(self) -> None:
        producer = ModuleFixture("vs")
        length = producer.allocate()
        producer.emit(50, producer.uint, length, 2)  # OpSpecConstant with default length 2
        producer.decorate(length, 1, 7)  # SpecId 7
        shape = producer.emit_type(spv.OP_TYPE_ARRAY, producer.numeric(components=4), length)
        producer.variable(spv.STORAGE_OUTPUT, shape, location=0)

        self.assertEqual(kinds(producer, input(array_count=2)), [])
        self.assertEqual(kinds(producer, input(array_count=3)), ["type-mismatch"])

    def test_matrix_and_array_of_columns_are_distinct_types(self):
        vertices = ModuleFixture("vs")
        column = vertices.numeric(32, 2)
        vertices.variable(spv.STORAGE_OUTPUT, vertices.emit_type(spv.OP_TYPE_MATRIX, column, 3), 0)
        fragment = ModuleFixture("ps")
        column = fragment.numeric(32, 2)
        fragment.variable(spv.STORAGE_INPUT, fragment.emit_type(spv.OP_TYPE_MATRIX, column, 3), 0)
        self.assertEqual(kinds(vertices, fragment), [])
        self.assertEqual(kinds(vertices, input(components=2, array_count=3)), ["type-mismatch"])


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.vs = self.module("vertex", output())
        self.ps = self.module("pixel", input())
        self.bad = self.module("bad_pixel", input(location=8))
        self.manifest = self.directory / "pairs.json"
        self.report = self.directory / "report.json"

    def module(self, name, builder):
        path = self.directory / (name + ".spv")
        path.write_bytes(builder.binary())
        return path

    def command(self, *args):
        return subprocess.run([sys.executable, "-I", ROOT / "tools" / "shader_link_check.py", *args],
                              capture_output=True, text=True, timeout=15)

    def write_manifest(self, *items):
        self.manifest.write_text(json.dumps({"version": 1, "pipelines": items}), encoding="utf-8")

    def run_batch(self, *extra):
        return self.command("batch", self.manifest, "--shader-dir", self.directory, *extra)

    def test_direct_match(self):
        result = self.command("check", "--vs", self.vs, "--ps", self.ps)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compatible (1 links checked)", result.stdout)

    def test_direct_entry_overrides(self):
        vertex, pixel = output(), input()
        vertex.entry("Vertex", vertex.ids)
        pixel.entry("Pixel", pixel.ids)
        paths = ("--vs", self.module("v", vertex), "--ps", self.module("p", pixel))
        for options in (("--vs-entry", "Vertex", "--ps-entry", "Pixel"),
                        ("--entry", "Vertex", "--ps-entry", "Pixel")):
            with self.subTest(options=options):
                result = self.command("check", *paths, *options)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("compatible (1 links checked)", result.stdout)
        missing = self.command("check", *paths, "--entry", "Vertex")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("[module-error]", missing.stdout)

    def test_direct_entry_requires_its_stage(self):
        result = self.command("check", "--vs", self.vs, "--ps", self.ps, "--gs-entry", "Geometry")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--gs-entry requires --gs", result.stderr)

    def test_incompatible_pipeline_returns_one_and_reports_location(self):
        result = self.command("check", "--vs", self.vs, "--ps", self.bad)
        self.assertEqual(result.returncode, 1)
        self.assertIn("vs -> ps at location 8, component 0 [missing-output]", result.stdout)

    def test_wrong_stage_and_missing_module_return_two(self):
        result = self.command("check", "--vs", self.ps, "--ps", self.directory / "missing.spv")
        self.assertEqual(result.returncode, 2)
        self.assertIn("is ps, expected vs", result.stdout)
        self.assertIn("[module-error]", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_slot_combination_returns_two(self):
        result = self.command("check", "--vs", self.vs, "--ms", self.vs, "--ps", self.ps)
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected vs+ps", result.stderr)

    def test_batch_checks_other_pipelines_after_an_error_and_reports_all(self):
        self.write_manifest({"name": "broken", "vs": "vertex", "ps": "missing"},
                            {"name": "bad", "vs": "vertex", "ps": "bad_pixel"},
                            {"name": "valid", "vs": "vertex", "ps": "pixel"})
        result = self.run_batch("--report", self.report, "--quiet")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("valid: compatible", result.stdout)
        report = json.loads(self.report.read_bytes())
        self.assertEqual(report["summary"], {"compatible": 1, "incompatible": 1, "invalid": 1})
        self.assertEqual([pipeline["name"] for pipeline in report["pipelines"]], ["broken", "bad", "valid"])
        self.assertEqual(report["pipelines"][1]["problems"][0]["location"]["location"], 8)
        self.assertNotIn(b"\r\n", self.report.read_bytes())

    def test_three_stage_chain_does_not_compare_across_geometry(self):
        geometry = simple("gs", spv.STORAGE_INPUT, 0, array_count=3)
        geometry.variable(spv.STORAGE_OUTPUT, geometry.numeric(32, 4), 8)
        gs = self.module("geometry", geometry)
        result = self.command("check", "--vs", self.vs, "--gs", gs, "--ps", self.bad)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("compatible (2 links checked)", result.stdout)

    def test_alternate_entry_names_are_resolved_per_stage(self):
        shader = input()
        shader.entry("ColorMain", shader.ids)
        self.module("alternative", shader)
        self.write_manifest({"name": "alt", "vs": "vertex", "ps": {"shader": "alternative", "entry": "ColorMain"}})
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_duplicate_manifest_key_is_rejected_with_path_and_key(self):
        # The last stage value would produce a valid pipeline if the duplicate were ignored.
        self.manifest.write_text(
            '{"version": 1, "pipelines": ['
            '{"name": "dup", "vs": "missing", "vs": "vertex", "ps": "pixel"}]}',
            encoding="utf-8",
        )
        result = self.run_batch("--report", self.report)
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(self.manifest), result.stderr)
        self.assertIn("duplicate key 'vs'", result.stderr)
        self.assertFalse(self.report.exists())

    def test_bad_manifest_fields_return_clean_error(self):
        cases = [
            {"version": True, "pipelines": []},
            {"pipelines": [{"name": "x", "vs": "../vertex", "ps": "pixel"}]},
            {"pipelines": [{"name": "x", "vs": "vertex", "ps": "pixel"}] * 2},
            {"pipelines": [{"name": "x", "vs": "vertex", "ps": {"path": "pixel.spv"}}]},
        ]
        for document in cases:
            with self.subTest(document=document):
                self.manifest.write_text(json.dumps(document), encoding="utf-8")
                result = self.run_batch()
                self.assertEqual(result.returncode, 2)
                self.assertIn("shader_link_check: error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_manifest_saved_with_a_windows_bom_is_read(self):
        # Notepad and PowerShell 5's Out-File -Encoding utf8 put a BOM first; that is not a manifest error.
        self.manifest.write_bytes(b"\xef\xbb\xbf" + json.dumps(
            {"version": 1, "pipelines": [{"name": "bom", "vs": "vertex", "ps": "pixel"}]}).encode("utf-8"))
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bom: compatible", result.stdout)

    def test_corpus_style_names_are_accepted_and_paths_are_not(self):
        # make_game's corpus names (shader_carve stems) contain "~", "!", "&", "(", a leading "." and more: 21,043 of
        # 26,503 on a 1.0.2.1 copy failed the old [a-zA-Z0-9_.-] rule. Only names that leave --shader-dir fail.
        for stem in ("VS~1", "pixel!2&x", "a(b)c", "x y", "_~-.ok", ".dotted"):
            with self.subTest(stem=stem):
                (self.directory / (stem + ".spv")).write_bytes(self.vs.read_bytes())
                self.write_manifest({"name": "n", "vs": stem, "ps": "pixel"})
                result = self.run_batch()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for stem in ("../vertex", "..", ".", "sub/vertex", "sub\\vertex", "C:vertex", "a|b", "a\x00b",
                     " vertex", ""):
            with self.subTest(stem=stem):
                self.write_manifest({"name": "n", "vs": stem, "ps": "pixel"})
                result = self.run_batch()
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("expected a shader name without directories", result.stderr)

    def test_report_preserves_inputs(self):
        original = self.vs.read_bytes()
        result = self.command("check", "--vs", self.vs, "--ps", self.ps, "--report", self.vs)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.vs.read_bytes(), original)
        self.write_manifest({"name": "x", "vs": "vertex", "ps": "pixel"})
        result = self.run_batch("--report", self.manifest)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(json.loads(self.manifest.read_text())["pipelines"])

    def test_existing_report_cannot_be_a_hard_link_to_an_input(self):
        original = self.vs.read_bytes()
        try:
            os.link(self.vs, self.report)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        result = self.command("check", "--vs", self.vs, "--ps", self.ps, "--report", self.report)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("report would overwrite an input file", result.stderr)
        self.assertTrue(self.report.samefile(self.vs))
        self.assertEqual(self.vs.read_bytes(), original)

    def test_missing_report_path_still_cannot_match_an_input_path(self):
        with self.assertRaisesRegex(ValueError, "report would overwrite an input file"):
            link.ensure_report_is_separate(self.report, [self.report])

    def test_inspect_keeps_types_and_components(self):
        shader = self.module("with_component", input(location=3, component=2, components=2))
        result = self.command("inspect", shader)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["inputs"][0]["varyings"],
                         [{"location": 3, "component": 2, "index": 0, "type": "vec2<f32>",
                           "per_primitive": False}])

    def test_shader_is_reflected_once_for_repeated_pipelines(self):
        self.write_manifest(*({"name": f"pair-{i}", "vs": "vertex", "ps": "pixel"} for i in range(20)))
        output_stream = io.StringIO()
        with patch.object(link, "reflect_module", wraps=link.reflect_module) as reflect:
            with contextlib.redirect_stdout(output_stream):
                result = link.main(["batch", str(self.manifest), "--shader-dir", str(self.directory), "--quiet"])
        self.assertEqual(result, 0)
        self.assertEqual(reflect.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
