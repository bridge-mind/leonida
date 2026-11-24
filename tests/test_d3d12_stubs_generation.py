# SPDX-License-Identifier: MIT
"""Unit tests for tools/gen_d3d12_stubs.py.

Covers synthetic stub generation, entry point enumeration, and ordinal mapping
using purely synthetic interface declarations and export name lists.
Zero game data or proprietary assets are used.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

# Ensure tools directory is accessible.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import gen_d3d12_stubs as gen  # noqa: E402


def make_midl_interface(
    name: str,
    base: str,
    methods: list[tuple[str, str, str, str]],
    directives: list[tuple[int, str]] | None = None,
    guid: str = "00000000-0000-0000-0000-000000000001",
) -> str:
    """Build a synthetic MIDL_INTERFACE string from method specifications.

    methods: list of (return_type, method_name, params, const_qualifier)
    directives: optional list of (insert_index, directive_line)
    """
    lines = [
        f'MIDL_INTERFACE("{guid}")',
        f"{name} : public {base}",
        "{",
        "public:",
    ]

    directive_map: dict[int, list[str]] = {}
    if directives:
        for idx, d in directives:
            directive_map.setdefault(idx, []).append(d)

    for i, (ret, m_name, params, const_q) in enumerate(methods):
        if i in directive_map:
            lines.extend(directive_map[i])
        const_suffix = f" {const_q}" if const_q else ""
        lines.append(f"    virtual {ret} STDMETHODCALLTYPE {m_name}({params}){const_suffix} = 0;")

    end_idx = len(methods)
    if end_idx in directive_map:
        lines.extend(directive_map[end_idx])

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


class SyntheticStubGenerationTests(unittest.TestCase):
    """Test suite covering synthetic COM interface stub generation."""

    def test_stub_class_structure_and_signature(self):
        methods = [
            ("void", "SyntheticExecute", "UINT count", ""),
            ("HRESULT", "SyntheticQuery", "REFIID riid, void **ppv", ""),
            ("UINT64", "SyntheticGetSize", "void", "const"),
        ]
        hdr = make_midl_interface("ISyntheticObject", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)

        emitted = gen.emit_class(ifaces, "ISyntheticObject")
        self.assertTrue(emitted.startswith("class ISyntheticObjectStubs : public ISyntheticObject {\npublic:"))
        self.assertTrue(emitted.endswith("};"))
        self.assertIn("// ISyntheticObject", emitted)

        # Check exact method overrides
        self.assertIn(
            '    void STDMETHODCALLTYPE SyntheticExecute(UINT count) override '
            '{ ::rt::d3d12::stub("ISyntheticObject::SyntheticExecute"); }',
            emitted,
        )
        self.assertIn(
            '    HRESULT STDMETHODCALLTYPE SyntheticQuery(REFIID riid, void **ppv) override '
            '{ ::rt::d3d12::stub("ISyntheticObject::SyntheticQuery"); return E_NOTIMPL; }',
            emitted,
        )
        self.assertIn(
            '    UINT64 STDMETHODCALLTYPE SyntheticGetSize(void) const override '
            '{ ::rt::d3d12::stub("ISyntheticObject::SyntheticGetSize"); return {}; }',
            emitted,
        )

    def test_return_type_conventions(self):
        """Verify return conventions: void -> nothing, HRESULT -> E_NOTIMPL, other -> {}."""
        methods = [
            ("void", "DoVoid", "void", ""),
            ("HRESULT", "DoHresult", "void", ""),
            ("BOOL", "DoBool", "void", ""),
            ("UINT", "DoUint", "void", ""),
            ("float", "DoFloat", "void", ""),
            ("void*", "DoPointer", "void", ""),
            ("ID3D12Resource*", "DoInterfacePtr", "void", ""),
            ("D3D12_CPU_DESCRIPTOR_HANDLE", "DoDescriptor", "void", ""),
        ]
        hdr = make_midl_interface("ITypeCheck", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)
        emitted = gen.emit_class(ifaces, "ITypeCheck")

        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoVoid"); }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoHresult"); return E_NOTIMPL; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoBool"); return {}; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoUint"); return {}; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoFloat"); return {}; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoPointer"); return {}; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoInterfacePtr"); return {}; }', emitted)
        self.assertIn('{ ::rt::d3d12::stub("ITypeCheck::DoDescriptor"); return {}; }', emitted)

    def test_preprocessor_directives_in_stub(self):
        methods = [
            ("void", "StepBefore", "void", ""),
            ("void", "StepConditional", "void", ""),
            ("void", "StepAfter", "void", ""),
        ]
        directives = [
            (1, "#if defined(SYNTHETIC_FEATURE_A)"),
            (2, "#endif"),
        ]
        hdr = make_midl_interface("IConditional", "IUnknown", methods, directives=directives)
        ifaces = gen.parse_interfaces(hdr)
        emitted = gen.emit_class(ifaces, "IConditional")

        self.assertIn("#if defined(SYNTHETIC_FEATURE_A)", emitted)
        self.assertIn("#endif", emitted)
        lines = emitted.splitlines()
        idx_if = lines.index("#if defined(SYNTHETIC_FEATURE_A)")
        idx_cond = next(i for i, l in enumerate(lines) if "StepConditional" in l)
        idx_endif = lines.index("#endif")
        self.assertLess(idx_if, idx_cond)
        self.assertLess(idx_cond, idx_endif)

    def test_comment_stripping(self):
        """Block comments must be removed without leaving orphaned tokens."""
        raw = """\
MIDL_INTERFACE("11111111-0000-0000-0000-000000000001")
IWithComments : public IUnknown
{
public:
    /* Block comment before method */
    virtual void STDMETHODCALLTYPE ActiveOne(void) = 0;
    /*
    virtual void STDMETHODCALLTYPE CommentedOutMethod(int x) = 0;
    */
    virtual HRESULT STDMETHODCALLTYPE ActiveTwo(void) = 0; /* Trailing block comment */
};
"""
        ifaces = gen.parse_interfaces(raw)
        items = ifaces["IWithComments"][1]
        method_names = [item[2] for item in items if item[0] == "m"]
        self.assertEqual(method_names, ["ActiveOne", "ActiveTwo"])
        self.assertNotIn("CommentedOutMethod", method_names)


class EntryPointEnumerationTests(unittest.TestCase):
    """Test suite covering entry point enumeration from synthetic export name lists."""

    def test_entry_point_enumeration_order(self):
        synthetic_export_names = [
            "CreateSyntheticCommandQueue",
            "CreateSyntheticCommandAllocator",
            "CreateSyntheticGraphicsPipeline",
            "CreateSyntheticComputePipeline",
            "CreateSyntheticDescriptorHeap",
            "CreateSyntheticRootSignature",
        ]
        methods = [("HRESULT", name, "void", "") for name in synthetic_export_names]
        hdr = make_midl_interface("ISyntheticDevice", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)

        self.assertIn("ISyntheticDevice", ifaces)
        base, items = ifaces["ISyntheticDevice"]
        self.assertEqual(base, "IUnknown")

        enumerated_names = [item[2] for item in items if item[0] == "m"]
        self.assertEqual(enumerated_names, synthetic_export_names)

    def test_multiline_entry_point_parameters(self):
        raw = """\
MIDL_INTERFACE("22222222-0000-0000-0000-000000000002")
ISyntheticComplex : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE DispatchComplexOperation(
        _In_ const void *pInputBuffer,
        _In_ UINT inputBufferSize,
        _Out_writes_bytes_(outputBufferSize) void *pOutputBuffer,
        UINT outputBufferSize,
        _Out_opt_ UINT *pBytesWritten
    ) = 0;
};
"""
        ifaces = gen.parse_interfaces(raw)
        items = ifaces["ISyntheticComplex"][1]
        self.assertEqual(len(items), 1)
        tag, ret, name, params, const_q = items[0]
        self.assertEqual(tag, "m")
        self.assertEqual(ret, "HRESULT")
        self.assertEqual(name, "DispatchComplexOperation")
        self.assertIn("pInputBuffer", params)
        self.assertIn("pBytesWritten", params)
        self.assertEqual(const_q, "")

    def test_entry_point_metadata_extraction(self):
        methods = [
            ("void", "GetDeviceName", "char *pOut, size_t maxLen", "const"),
            ("HRESULT", "SetShaderState", "UINT slot, void *pBytecode", ""),
            ("UINT", "GetDescriptorIncrementSize", "UINT heapType", "const"),
        ]
        hdr = make_midl_interface("ISyntheticInfo", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)
        items = ifaces["ISyntheticInfo"][1]

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0], ("m", "void", "GetDeviceName", "char *pOut, size_t maxLen", "const"))
        self.assertEqual(items[1], ("m", "HRESULT", "SetShaderState", "UINT slot, void *pBytecode", ""))
        self.assertEqual(items[2], ("m", "UINT", "GetDescriptorIncrementSize", "UINT heapType", "const"))


class OrdinalMappingTests(unittest.TestCase):
    """Test suite covering ordinal mapping across COM interface vtable hierarchies."""

    def test_single_interface_ordinal_sequence(self):
        export_names = ["EntryAlpha", "EntryBeta", "EntryGamma", "EntryDelta"]
        methods = [("void", name, "void", "") for name in export_names]
        hdr = make_midl_interface("ISingleHierarchy", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)

        emitted = gen.emit_class(ifaces, "ISingleHierarchy")
        method_lines = [l.strip() for l in emitted.splitlines() if "::rt::d3d12::stub" in l]

        # In COM vtable ordering, methods are laid out in exact declaration order (ordinals 0..N-1)
        self.assertEqual(len(method_lines), len(export_names))
        for ordinal, name in enumerate(export_names):
            expected_stub_call = f'::rt::d3d12::stub("ISingleHierarchy::{name}");'
            self.assertIn(expected_stub_call, method_lines[ordinal])

    def test_multi_tier_inheritance_ordinal_mapping(self):
        """In COM inheritance, base interface entry points occupy lower ordinals

        than derived interface entry points in the virtual table.
        """
        root_methods = [
            ("void", "RootEntry1", "void", ""),
            ("void", "RootEntry2", "void", ""),
        ]
        child_methods = [
            ("HRESULT", "ChildEntry1", "void", ""),
            ("HRESULT", "ChildEntry2", "void", ""),
        ]
        grandchild_methods = [
            ("UINT", "GrandchildEntry1", "void", ""),
        ]

        hdr_root = make_midl_interface("ISyntheticRoot", "IUnknown", root_methods, guid="33333333-0000-0000-0000-000000000001")
        hdr_child = make_midl_interface("ISyntheticChild", "ISyntheticRoot", child_methods, guid="33333333-0000-0000-0000-000000000002")
        hdr_grandchild = make_midl_interface("ISyntheticGrandchild", "ISyntheticChild", grandchild_methods, guid="33333333-0000-0000-0000-000000000003")

        combined = f"{hdr_root}\n{hdr_child}\n{hdr_grandchild}"
        ifaces = gen.parse_interfaces(combined)

        # Chain must be in topological order: Root -> Child -> Grandchild
        chain_order = gen.chain(ifaces, "ISyntheticGrandchild")
        self.assertEqual(chain_order, ["ISyntheticRoot", "ISyntheticChild", "ISyntheticGrandchild"])

        emitted = gen.emit_class(ifaces, "ISyntheticGrandchild")
        stub_calls = re.findall(r'::rt::d3d12::stub\("([^"]+)"\);', emitted)

        expected_ordinal_map = [
            (0, "ISyntheticRoot::RootEntry1"),
            (1, "ISyntheticRoot::RootEntry2"),
            (2, "ISyntheticChild::ChildEntry1"),
            (3, "ISyntheticChild::ChildEntry2"),
            (4, "ISyntheticGrandchild::GrandchildEntry1"),
        ]

        self.assertEqual(len(stub_calls), len(expected_ordinal_map))
        for expected_ordinal, expected_symbol in expected_ordinal_map:
            self.assertEqual(stub_calls[expected_ordinal], expected_symbol)

    def test_synthetic_export_name_list_ordinal_stability(self):
        """Ensures that changing or extending the synthetic export list preserves ordinal stability."""
        export_list = [f"SyntheticExport_{i:03d}" for i in range(25)]
        methods = [("void", name, f"int param_{i}", "") for i, name in enumerate(export_list)]
        hdr = make_midl_interface("IBigExportTable", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)

        emitted = gen.emit_class(ifaces, "IBigExportTable")
        stub_calls = re.findall(r'::rt::d3d12::stub\("([^"]+)"\);', emitted)

        self.assertEqual(len(stub_calls), 25)
        for ordinal, name in enumerate(export_list):
            self.assertEqual(stub_calls[ordinal], f"IBigExportTable::{name}")


class InheritanceChainValidationTests(unittest.TestCase):
    """Test suite covering inheritance chain traversal and cycle detection."""

    def test_single_leaf_chain(self):
        methods = [("void", "DoRoot", "void", "")]
        hdr = make_midl_interface("IRootOnly", "IUnknown", methods)
        ifaces = gen.parse_interfaces(hdr)
        self.assertEqual(gen.chain(ifaces, "IRootOnly"), ["IRootOnly"])

    def test_deep_linear_chain(self):
        headers = [
            make_midl_interface("INodeA", "IUnknown", [("void", "OpA", "void", "")]),
            make_midl_interface("INodeB", "INodeA", [("void", "OpB", "void", "")]),
            make_midl_interface("INodeC", "INodeB", [("void", "OpC", "void", "")]),
            make_midl_interface("INodeD", "INodeC", [("void", "OpD", "void", "")]),
        ]
        ifaces = gen.parse_interfaces("\n".join(headers))
        self.assertEqual(gen.chain(ifaces, "INodeD"), ["INodeA", "INodeB", "INodeC", "INodeD"])

    def test_missing_base_interface_raises_system_exit(self):
        hdr = make_midl_interface("IOrphanInterface", "INonExistentBase", [("void", "Op", "void", "")])
        ifaces = gen.parse_interfaces(hdr)
        with self.assertRaises(SystemExit) as cm:
            gen.chain(ifaces, "IOrphanInterface")
        self.assertIn("INonExistentBase not found", str(cm.exception))

    def test_direct_self_cycle_raises_system_exit(self):
        hdr = make_midl_interface("ISelfCyclic", "ISelfCyclic", [("void", "Op", "void", "")])
        ifaces = gen.parse_interfaces(hdr)
        with self.assertRaises(SystemExit) as cm:
            gen.chain(ifaces, "ISelfCyclic")
        self.assertIn("circular inheritance", str(cm.exception))

    def test_indirect_cycle_raises_system_exit(self):
        headers = [
            make_midl_interface("ICycleA", "ICycleC", [("void", "OpA", "void", "")]),
            make_midl_interface("ICycleB", "ICycleA", [("void", "OpB", "void", "")]),
            make_midl_interface("ICycleC", "ICycleB", [("void", "OpC", "void", "")]),
        ]
        ifaces = gen.parse_interfaces("\n".join(headers))
        with self.assertRaises(SystemExit) as cm:
            gen.chain(ifaces, "ICycleC")
        self.assertIn("circular inheritance", str(cm.exception))


class ParserSyntaxValidationTests(unittest.TestCase):
    """Test suite covering edge cases and syntax rejections in parse_interfaces."""

    def test_empty_string_returns_empty_dict(self):
        self.assertEqual(gen.parse_interfaces(""), {})
        self.assertEqual(gen.parse_interfaces("// empty comment\n/* nothing */"), {})

    def test_unterminated_method_declaration_raises(self):
        broken = """\
MIDL_INTERFACE("44444444-0000-0000-0000-000000000001")
IUnterminated : public IUnknown
{
public:
    virtual void STDMETHODCALLTYPE BadMethod(int a);
};
"""
        with self.assertRaises(SystemExit) as cm:
            gen.parse_interfaces(broken)
        self.assertIn("unterminated declaration", str(cm.exception))

    def test_unparseable_method_syntax_raises(self):
        malformed = """\
MIDL_INTERFACE("44444444-0000-0000-0000-000000000002")
IMalformed : public IUnknown
{
public:
    virtual InvalidSyntaxWithoutCallingConvention() = 0;
};
"""
        with self.assertRaises(SystemExit) as cm:
            gen.parse_interfaces(malformed)
        self.assertIn("cannot parse", str(cm.exception))


class StubGenerationCliAndIoTests(unittest.TestCase):
    """Test suite covering CLI argument parsing, environment resolution, and file generation."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name

        self.synth_header = os.path.join(self.tmp_dir, "synthetic_d3d12.h")
        methods_a = [("void", "StubA1", "void", ""), ("HRESULT", "StubA2", "void", "")]
        methods_b = [("UINT", "StubB1", "void", "")]
        content = f"{make_midl_interface('ISynthA', 'IUnknown', methods_a)}\n{make_midl_interface('ISynthB', 'ISynthA', methods_b)}"
        with open(self.synth_header, "w", encoding="utf-8") as f:
            f.write(content)

        self.out_file = os.path.join(self.tmp_dir, "gen_sub", "out_stubs.h")

    def test_find_header_in_include_path(self):
        # Empty env
        with mock.patch.dict(os.environ, {"INCLUDE": ""}, clear=False):
            self.assertIsNone(gen.find_header())

        # Target file named d3d12.h in a search directory
        d3d12_dir = os.path.join(self.tmp_dir, "sdk_include")
        os.makedirs(d3d12_dir, exist_ok=True)
        d3d12_path = os.path.join(d3d12_dir, "d3d12.h")
        with open(d3d12_path, "w", encoding="utf-8") as f:
            f.write("// synthetic d3d12.h\n")

        # INCLUDE is os.pathsep-separated, as find_header reads it (";" on Windows, ":" elsewhere).
        empty_dir = os.path.join(self.tmp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        with mock.patch.dict(os.environ, {"INCLUDE": os.pathsep.join([empty_dir, d3d12_dir])}, clear=False):
            self.assertEqual(gen.find_header(), d3d12_path)

    def test_cli_output_generation_and_boilerplate(self):
        argv = [
            "gen_d3d12_stubs.py",
            "--out",
            self.out_file,
            "--header",
            self.synth_header,
            "ISynthA",
            "ISynthB",
        ]
        with mock.patch("sys.argv", argv):
            res = gen.main()
        self.assertEqual(res, 0)
        self.assertTrue(os.path.isfile(self.out_file))

        with open(self.out_file, encoding="utf-8") as f:
            generated = f.read()

        # Check required boilerplate
        self.assertTrue(generated.startswith("// Generated by tools/gen_d3d12_stubs.py"))
        self.assertIn("#pragma once", generated)
        self.assertIn("#include <d3d12.h>", generated)
        self.assertIn("namespace rt::d3d12 { void stub(const char* method); }", generated)
        self.assertIn("#pragma warning(push)", generated)
        self.assertIn("#pragma warning(disable : 4100)", generated)
        self.assertIn("#pragma warning(pop)", generated)

        # Check classes exist in order
        self.assertIn("class ISynthAStubs : public ISynthA", generated)
        self.assertIn("class ISynthBStubs : public ISynthB", generated)
        self.assertLess(generated.index("class ISynthAStubs"), generated.index("class ISynthBStubs"))

    def test_idempotent_file_writing(self):
        argv = [
            "gen_d3d12_stubs.py",
            "--out",
            self.out_file,
            "--header",
            self.synth_header,
            "ISynthA",
        ]
        with mock.patch("sys.argv", argv):
            gen.main()
        self.assertTrue(os.path.isfile(self.out_file))
        mtime_initial = os.stat(self.out_file).st_mtime_ns

        # Second execution with exact same arguments should not modify mtime
        with mock.patch("sys.argv", argv):
            gen.main()
        mtime_second = os.stat(self.out_file).st_mtime_ns
        self.assertEqual(mtime_initial, mtime_second)

    def test_missing_header_raises_system_exit(self):
        argv = [
            "gen_d3d12_stubs.py",
            "--out",
            self.out_file,
            "ISynthA",
        ]
        with mock.patch.dict(os.environ, {"INCLUDE": ""}, clear=False):
            with mock.patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as cm:
                    gen.main()
                self.assertIn("d3d12.h not found", str(cm.exception))



class MethodCoverageAndReportTests(unittest.TestCase):
    """Test suite covering method coverage reporting and CLI report mode."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name

        self.synth_header = os.path.join(self.tmp_dir, "synthetic_coverage_d3d12.h")
        methods_base = [
            ("void", "BaseMethodOne", "void", ""),
            ("HRESULT", "BaseMethodTwo", "int arg", ""),
        ]
        methods_derived = [
            ("UINT", "DerivedMethodOne", "float f", "const"),
            ("void*", "DerivedMethodTwo", "void", ""),
        ]
        content = (
            f"{make_midl_interface('ICovBase', 'IUnknown', methods_base)}\n"
            f"{make_midl_interface('ICovDerived', 'ICovBase', methods_derived)}"
        )
        with open(self.synth_header, "w", encoding="utf-8") as f:
            f.write(content)

    def test_implemented_names_parsing(self):
        src1 = os.path.join(self.tmp_dir, "impl1.cpp")
        src2 = os.path.join(self.tmp_dir, "impl2.cpp")

        with open(src1, "w", encoding="utf-8") as f:
            f.write("""\
// Synthetic implementation file 1
HRESULT STDMETHODCALLTYPE BaseMethodOne(void) override { return S_OK; }
void STDMETHODCALLTYPE DerivedMethodTwo(void) override {}
""")

        with open(src2, "w", encoding="utf-8") as f:
            f.write("""\
// Synthetic implementation file 2
UINT STDMETHODCALLTYPE NonInterfaceMethod(int x) { return 0; }
""")

        names = gen.implemented_names([src1, src2])
        self.assertEqual(names, {"BaseMethodOne", "DerivedMethodTwo", "NonInterfaceMethod"})

    def test_coverage_calculation_metrics(self):
        src = os.path.join(self.tmp_dir, "impl.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write("""\
void STDMETHODCALLTYPE BaseMethodOne(void) override {}
UINT STDMETHODCALLTYPE DerivedMethodOne(float f) const override { return 1; }
""")

        with open(self.synth_header, encoding="utf-8") as f:
            ifaces = gen.parse_interfaces(f.read())

        rep = gen.coverage(ifaces, ["ICovDerived"], [src])
        self.assertEqual(len(rep), 1)
        item = rep[0]
        self.assertEqual(item["interface"], "ICovDerived")
        self.assertEqual(item["methods"], 4)
        self.assertEqual(item["implemented"], 2)
        self.assertEqual(item["open"], ["ICovBase::BaseMethodTwo", "ICovDerived::DerivedMethodTwo"])

    def test_coverage_multiple_leaves(self):
        src = os.path.join(self.tmp_dir, "impl.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write("void STDMETHODCALLTYPE BaseMethodOne(void) override {}\n")

        with open(self.synth_header, encoding="utf-8") as f:
            ifaces = gen.parse_interfaces(f.read())

        rep = gen.coverage(ifaces, ["ICovBase", "ICovDerived"], [src])
        self.assertEqual(len(rep), 2)
        self.assertEqual(rep[0]["interface"], "ICovBase")
        self.assertEqual(rep[0]["methods"], 2)
        self.assertEqual(rep[0]["implemented"], 1)
        self.assertEqual(rep[0]["open"], ["ICovBase::BaseMethodTwo"])

        self.assertEqual(rep[1]["interface"], "ICovDerived")
        self.assertEqual(rep[1]["methods"], 4)
        self.assertEqual(rep[1]["implemented"], 1)
        self.assertEqual(rep[1]["open"], [
            "ICovBase::BaseMethodTwo",
            "ICovDerived::DerivedMethodOne",
            "ICovDerived::DerivedMethodTwo",
        ])

    def test_cli_report_mode_generates_json_file(self):
        src = os.path.join(self.tmp_dir, "impl.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write("void STDMETHODCALLTYPE BaseMethodOne(void) override {}\n")

        rep_file = os.path.join(self.tmp_dir, "coverage.json")
        argv = [
            "gen_d3d12_stubs.py",
            "--report",
            rep_file,
            "--source",
            src,
            "--header",
            self.synth_header,
            "ICovDerived",
        ]
        with mock.patch("sys.argv", argv):
            res = gen.main()
        self.assertEqual(res, 0)
        self.assertTrue(os.path.isfile(rep_file))

        with open(rep_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["interface"], "ICovDerived")
        self.assertEqual(data[0]["methods"], 4)
        self.assertEqual(data[0]["implemented"], 1)

    def test_cli_missing_out_and_report_raises_error(self):
        argv = [
            "gen_d3d12_stubs.py",
            "--header",
            self.synth_header,
            "ICovDerived",
        ]
        with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gen.main()

    def test_cli_report_missing_source_raises_error(self):
        argv = [
            "gen_d3d12_stubs.py",
            "--report",
            os.path.join(self.tmp_dir, "rep.json"),
            "--header",
            self.synth_header,
            "ICovDerived",
        ]
        with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gen.main()


if __name__ == "__main__":
    unittest.main()
