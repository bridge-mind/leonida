"""Tests for tools/gen_d3d12_stubs.py, driven by a synthetic d3d12.h (no Windows SDK).

    python -I tests/test_gen_d3d12_stubs.py
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "gen_d3d12_stubs.py")
    spec = importlib.util.spec_from_file_location("gen_d3d12_stubs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module()

GOOD_HEADER = """\
MIDL_INTERFACE("11111111-2222-3333-4444-555555555555")
IBase : public IUnknown
{
public:
    virtual void STDMETHODCALLTYPE DoThing(UINT a, UINT b) = 0;
    virtual HRESULT STDMETHODCALLTYPE Multi(
        UINT first,
        UINT second) = 0;
    /* virtual void STDMETHODCALLTYPE Removed() = 0; */
    virtual HRESULT STDMETHODCALLTYPE AfterComment(void) = 0; /* trailing note */
};

MIDL_INTERFACE("66666666-7777-8888-9999-AAAAAAAAAAAA")
IChild : public IBase
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetThing(UINT index, _Out_ ID3D12Thing **ppThing) const = 0;
    virtual UINT64 STDMETHODCALLTYPE Size(void) = 0;
};
"""

GUARDED_HEADER = """\
MIDL_INTERFACE("22222222-2222-2222-2222-222222222222")
IGuarded : public IUnknown
{
public:
#if 0
    virtual void STDMETHODCALLTYPE Gone(void) = 0;
#endif
    virtual void STDMETHODCALLTYPE Kept(void) = 0;
};
"""

BRANCHED_HEADER = """\
MIDL_INTERFACE("77777777-7777-7777-7777-777777777777")
IBranched : public IUnknown
{
public:
#if defined(_MSC_VER) || !defined(_WIN32)
    virtual THING_DESC STDMETHODCALLTYPE GetDesc(void) = 0;
#else
    virtual THING_DESC *STDMETHODCALLTYPE GetDesc(
        THING_DESC * RetVal) = 0;
#endif
    virtual UINT STDMETHODCALLTYPE GetCount(void) = 0;
};
"""

ORPHAN_HEADER = """\
MIDL_INTERFACE("33333333-3333-3333-3333-333333333333")
IOrphan : public IMissing
{
public:
    virtual void STDMETHODCALLTYPE Nothing(void) = 0;
};
"""

BROKEN_HEADER = """\
MIDL_INTERFACE("44444444-4444-4444-4444-444444444444")
IBroken : public IUnknown
{
public:
    virtual void STDMETHODCALLTYPE NotAMethod = 0;
};
"""

UNTERMINATED_HEADER = """\
MIDL_INTERFACE("55555555-5555-5555-5555-555555555555")
IHalf : public IUnknown
{
public:
    virtual void STDMETHODCALLTYPE Missing(void);
};
"""

IFACES = gen.parse_interfaces(GOOD_HEADER)


class ParseInterfacesTests(unittest.TestCase):
    def test_names_bases_and_methods(self):
        self.assertEqual(sorted(IFACES), ["IBase", "IChild"])
        self.assertEqual(IFACES["IBase"][0], "IUnknown")
        self.assertEqual(IFACES["IChild"][0], "IBase")
        methods = [item for item in IFACES["IBase"][1] if item[0] == "m"]
        self.assertEqual([m[2] for m in methods], ["DoThing", "Multi", "AfterComment"])
        self.assertEqual(methods[0], ("m", "void", "DoThing", "UINT a, UINT b", ""))

    def test_multiline_declaration_is_joined(self):
        methods = [item for item in IFACES["IBase"][1] if item[0] == "m"]
        self.assertEqual(methods[1][3], "UINT first, UINT second")

    def test_const_method_is_marked(self):
        methods = [item for item in IFACES["IChild"][1] if item[0] == "m"]
        self.assertEqual(methods[0][4], "const")
        self.assertEqual(methods[0][2], "GetThing")

    def test_commented_out_method_is_dropped(self):
        names = [item[2] for item in IFACES["IBase"][1] if item[0] == "m"]
        self.assertNotIn("Removed", names)

    def test_inline_comment_after_a_declaration_keeps_it(self):
        names = [item[2] for item in IFACES["IBase"][1] if item[0] == "m"]
        self.assertIn("AfterComment", names)

    def test_cpp_line_comments_do_not_change_the_method_list(self):
        header = GOOD_HEADER.replace(
            "    virtual void STDMETHODCALLTYPE DoThing(UINT a, UINT b) = 0;",
            "    // virtual void STDMETHODCALLTYPE Omitted() = 0; };\n"
            "    virtual void STDMETHODCALLTYPE DoThing(UINT a, // parameter note\n"
            "        UINT b) = 0; // trailing comment")
        self.assertEqual(gen.parse_interfaces(header), IFACES)

    def test_preprocessor_lines_pass_through(self):
        items = gen.parse_interfaces(GUARDED_HEADER)["IGuarded"][1]
        self.assertEqual(items[0], ("#", "#if 0"))
        self.assertEqual(items[1][0], "m")
        self.assertEqual(items[1][2], "Gone")
        self.assertEqual(items[2], ("#", "#endif"))
        self.assertEqual(items[3][2], "Kept")

    def test_text_without_interfaces(self):
        self.assertEqual(gen.parse_interfaces("// nothing here\n"), {})

    def test_unterminated_declaration_raises(self):
        with self.assertRaises(SystemExit):
            gen.parse_interfaces(UNTERMINATED_HEADER)

    def test_unparseable_declaration_raises(self):
        with self.assertRaises(SystemExit):
            gen.parse_interfaces(BROKEN_HEADER)


class ChainTests(unittest.TestCase):
    def test_leaf_comes_last(self):
        self.assertEqual(gen.chain(IFACES, "IChild"), ["IBase", "IChild"])

    def test_root_interface(self):
        self.assertEqual(gen.chain(IFACES, "IBase"), ["IBase"])

    def test_unknown_base_raises(self):
        orphan = gen.parse_interfaces(ORPHAN_HEADER)
        with self.assertRaises(SystemExit):
            gen.chain(orphan, "IOrphan")


class EmitClassTests(unittest.TestCase):
    def test_class_skeleton_and_method_order(self):
        text = gen.emit_class(IFACES, "IChild")
        self.assertTrue(text.startswith("class IChildStubs : public IChild {\npublic:"))
        self.assertTrue(text.endswith("};"))
        self.assertLess(text.index("// IBase"), text.index("// IChild"))
        self.assertLess(text.index("DoThing"), text.index("GetThing"))
        self.assertNotIn("Removed", text)

    def test_return_conventions_and_stub_names(self):
        text = gen.emit_class(IFACES, "IChild")
        self.assertIn(
            '    void STDMETHODCALLTYPE DoThing(UINT a, UINT b) override '
            '{ ::rt::d3d12::stub("IBase::DoThing"); }',
            text,
        )
        self.assertIn(
            '    HRESULT STDMETHODCALLTYPE Multi(UINT first, UINT second) override '
            '{ ::rt::d3d12::stub("IBase::Multi"); return E_NOTIMPL; }',
            text,
        )
        self.assertIn(
            '    HRESULT STDMETHODCALLTYPE GetThing(UINT index, _Out_ ID3D12Thing **ppThing) const override '
            '{ ::rt::d3d12::stub("IChild::GetThing"); return E_NOTIMPL; }',
            text,
        )
        self.assertIn(
            '    UINT64 STDMETHODCALLTYPE Size(void) override '
            '{ ::rt::d3d12::stub("IChild::Size"); return {}; }',
            text,
        )

    def test_directives_are_emitted_verbatim(self):
        text = gen.emit_class(gen.parse_interfaces(GUARDED_HEADER), "IGuarded")
        self.assertIn("\n#if 0\n", text)
        self.assertIn("\n#endif\n", text)
        self.assertIn('stub("IGuarded::Gone")', text)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.source = os.path.join(self.dir, "impl.cpp")
        with open(self.source, "w", encoding="utf-8") as f:
            f.write(
                "HRESULT STDMETHODCALLTYPE DoThing(UINT a);\n"
                "HRESULT STDMETHODCALLTYPE GetThing(UINT i) const;\n"
                "void NotAnOverride();\n"
            )

    def test_reports_open_methods_in_vtable_order(self):
        report = gen.coverage(IFACES, ["IChild"], [self.source])
        self.assertEqual(report, [{
            "interface": "IChild",
            "methods": 5,
            "implemented": 2,
            "open": ["IBase::Multi", "IBase::AfterComment", "IChild::Size"],
        }])

    def test_coverage_ignores_comments_and_quoted_examples(self) -> None:
        with open(self.source, "w", encoding="utf-8") as f:
            f.write(
                "void STDMETHODCALLTYPE DoThing(UINT a, UINT b);\n"
                "// HRESULT STDMETHODCALLTYPE Multi(UINT first, UINT second);\n"
                "/* HRESULT STDMETHODCALLTYPE GetThing(UINT index) const; */\n"
                'const char* example = "HRESULT STDMETHODCALLTYPE AfterComment(void);";\n'
            )
        report = gen.coverage(IFACES, ["IChild"], [self.source])
        self.assertEqual(report[0]["implemented"], 1)
        self.assertEqual(report[0]["open"],
                         ["IBase::Multi", "IBase::AfterComment", "IChild::GetThing", "IChild::Size"])

    def test_no_sources_means_every_method_is_open(self):
        report = gen.coverage(IFACES, ["IBase"], [])
        self.assertEqual(report[0]["implemented"], 0)
        self.assertEqual(report[0]["open"], ["IBase::DoThing", "IBase::Multi", "IBase::AfterComment"])

    def test_repeated_interface_is_reported_once(self):
        report = gen.coverage(IFACES, ["IChild", "IChild"], [self.source])
        self.assertEqual([row["interface"] for row in report], ["IChild"])

    def test_method_declared_in_both_preprocessor_branches_counts_once(self):
        # d3d12.h declares struct-returning methods in both branches of #if/#else; that is one method.
        interfaces = gen.parse_interfaces(BRANCHED_HEADER)
        self.assertEqual(gen.coverage(interfaces, ["IBranched"], []), [{
            "interface": "IBranched",
            "methods": 2,
            "implemented": 0,
            "open": ["IBranched::GetDesc", "IBranched::GetCount"],
        }])
        with open(self.source, "w", encoding="utf-8", newline="\n") as f:
            f.write("THING_DESC STDMETHODCALLTYPE GetDesc() { return {}; }\n")
        report = gen.coverage(interfaces, ["IBranched"], [self.source])
        self.assertEqual((report[0]["methods"], report[0]["implemented"], report[0]["open"]),
                         (2, 1, ["IBranched::GetCount"]))
        # The stub class keeps both branches, so it compiles under either side of the condition.
        text = gen.emit_class(interfaces, "IBranched")
        self.assertEqual(text.count('stub("IBranched::GetDesc")'), 2)
        self.assertIn("\n#else\n", text)


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.header = os.path.join(self.dir, "d3d12.h")
        with open(self.header, "w", encoding="utf-8") as f:
            f.write(GOOD_HEADER)
        self.out = os.path.join(self.dir, "generated", "d3d12_stubs.h")

    def run_main(self, *interfaces, include_header=True):
        args = ["gen_d3d12_stubs.py", "--out", self.out]
        if include_header:
            args += ["--header", self.header]
        with mock.patch("sys.argv", args + list(interfaces)):
            return gen.main()

    def test_writes_header_and_class(self):
        self.assertEqual(self.run_main("IChild"), 0)
        self.assertTrue(os.path.isdir(os.path.dirname(self.out)))
        with open(self.out, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(
            text.startswith(
                "// Generated by tools/gen_d3d12_stubs.py from d3d12.h. Do not edit.\n"
            )
        )
        for line in (
            "#pragma once",
            "#include <d3d12.h>",
            "namespace rt::d3d12 { void stub(const char* method); }",
            "#pragma warning(push)",
            "#pragma warning(disable : 4100)  // unreferenced parameters",
            "class IChildStubs : public IChild {",
            "#pragma warning(pop)",
        ):
            self.assertIn(line, text)
        self.assertTrue(text.endswith("\n"))
        self.assertNotIn("\r\n", text)

    def test_second_run_does_not_rewrite(self):
        self.assertEqual(self.run_main("IChild"), 0)
        before = os.stat(self.out).st_mtime_ns
        self.assertEqual(self.run_main("IChild"), 0)
        self.assertEqual(os.stat(self.out).st_mtime_ns, before)

    def test_two_interfaces_keep_their_order(self):
        self.assertEqual(self.run_main("IBase", "IChild"), 0)
        with open(self.out, encoding="utf-8") as f:
            text = f.read()
        self.assertLess(text.index("class IBaseStubs"), text.index("class IChildStubs"))

    def test_repeated_interface_is_emitted_once(self):
        self.assertEqual(self.run_main("IChild", "IBase"), 0)
        with open(self.out, encoding="utf-8") as f:
            expected = f.read()

        self.assertEqual(self.run_main("IChild", "IBase", "IChild"), 0)
        with open(self.out, encoding="utf-8") as f:
            self.assertEqual(f.read(), expected)

    def test_unknown_interface_raises(self):
        with self.assertRaises(SystemExit):
            self.run_main("IWho")

    def test_circular_inheritance_reports_an_error(self):
        with open(self.header, "w", encoding="utf-8") as f:
            f.write(GOOD_HEADER.replace("IBase : public IUnknown", "IBase : public IChild"))
        tool = os.path.join(ROOT, "tools", "gen_d3d12_stubs.py")
        result = subprocess.run([sys.executable, "-I", tool, "--header", self.header,
                                 "--out", self.out, "IChild"], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IChild -> IBase -> IChild", result.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_report_without_out(self):
        source = os.path.join(self.dir, "impl.cpp")
        with open(source, "w", encoding="utf-8") as f:
            f.write("HRESULT STDMETHODCALLTYPE DoThing(UINT a);\n")
        report = os.path.join(self.dir, "coverage.json")
        args = ["gen_d3d12_stubs.py", "--header", self.header, "--report", report,
                "--source", source, "IChild"]
        with mock.patch("sys.argv", args):
            self.assertEqual(gen.main(), 0)
        with open(report, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[0]["interface"], "IChild")
        self.assertEqual(data[0]["open"],
                         ["IBase::Multi", "IBase::AfterComment", "IChild::GetThing", "IChild::Size"])
        self.assertFalse(os.path.exists(self.out))

    def test_out_or_report_is_required(self):
        with mock.patch("sys.argv", ["gen_d3d12_stubs.py", "--header", self.header, "IChild"]):
            with self.assertRaises(SystemExit):
                gen.main()

    def test_report_requires_a_source(self):
        report = os.path.join(self.dir, "coverage.json")
        with mock.patch("sys.argv", ["gen_d3d12_stubs.py", "--header", self.header,
                                     "--report", report, "IChild"]):
            with self.assertRaises(SystemExit):
                gen.main()

    def run_args(self, *args):
        with mock.patch("sys.argv", ["gen_d3d12_stubs.py", *args]):
            return gen.main()

    def make_source(self):
        source = os.path.join(self.dir, "impl.cpp")
        with open(source, "w", encoding="utf-8") as f:
            f.write("HRESULT STDMETHODCALLTYPE DoThing(UINT a);\n")
        return source

    def test_report_must_not_overwrite_a_source(self):
        source = self.make_source()
        with open(source, "rb") as f:
            before = f.read()
        with self.assertRaises(SystemExit):
            self.run_args("--header", self.header, "--report", source, "--source", source, "IChild")
        with open(source, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_out_must_not_overwrite_the_header(self):
        with open(self.header, "rb") as f:
            before = f.read()
        with self.assertRaises(SystemExit):
            self.run_args("--header", self.header, "--out", self.header, "IChild")
        with open(self.header, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_out_and_report_must_differ(self):
        source = self.make_source()
        same = os.path.join(self.dir, "report.json")
        with self.assertRaises(SystemExit):
            self.run_args("--header", self.header, "--out", same, "--report", same,
                          "--source", source, "IChild")
        self.assertFalse(os.path.exists(same))

    def test_out_must_be_a_file(self):
        folder = os.path.join(self.dir, "out_folder")
        os.makedirs(folder)
        with self.assertRaises(SystemExit):
            self.run_args("--header", self.header, "--out", folder, "IChild")

    def test_a_source_folder_is_refused(self):
        source = os.path.join(self.dir, "src_folder")
        os.makedirs(source)
        with self.assertRaises(SystemExit):
            self.run_args("--header", self.header, "--report", os.path.join(self.dir, "r.json"),
                          "--source", source, "IChild")

    def test_a_missing_header_is_refused_before_writing(self):
        with self.assertRaises(SystemExit):
            self.run_args("--header", os.path.join(self.dir, "missing.h"), "--out", self.out, "IChild")
        self.assertFalse(os.path.exists(self.out))

    def test_a_failed_write_is_one_line(self):
        blocker = os.path.join(self.dir, "not_a_folder")
        with open(blocker, "wb") as f:
            f.write(b"keep me")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = self.run_args("--header", self.header, "--out", os.path.join(blocker, "stubs.h"), "IChild")
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot write", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        with open(blocker, "rb") as f:
            self.assertEqual(f.read(), b"keep me")

    def test_find_header_uses_the_include_environment(self):
        with mock.patch.dict(os.environ, {"INCLUDE": self.dir}, clear=False):
            self.assertEqual(gen.find_header(), self.header)
            self.assertEqual(self.run_main("IChild", include_header=False), 0)
        self.assertTrue(os.path.isfile(self.out))

    def test_find_header_without_include_returns_none(self):
        with mock.patch.dict(os.environ, {"INCLUDE": ""}, clear=False):
            self.assertIsNone(gen.find_header())


class ReportPathCollisions(unittest.TestCase):
    """`--out` and `--report` are two artifacts; naming one file for both lost the report silently (#2716)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.header = os.path.join(self.dir, "d3d12.h")
        with open(self.header, "w", encoding="utf-8") as f:
            f.write(GOOD_HEADER)
        self.source = os.path.join(self.dir, "impl.cpp")
        with open(self.source, "w", encoding="utf-8") as f:
            f.write("HRESULT STDMETHODCALLTYPE DoThing(UINT a);\n")

    def run_main(self, *args):
        err = io.StringIO()
        with mock.patch("sys.argv", ["gen_d3d12_stubs.py", "--header", self.header, *args, "IChild"]), \
                contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                gen.main()
        self.assertEqual(raised.exception.code, 2)
        return err.getvalue()

    def test_out_and_report_must_differ(self):
        one = os.path.join(self.dir, "both.txt")
        err = self.run_main("--out", one, "--report", one, "--source", self.source)
        self.assertIn("--out and --report must use different paths", err)

    def test_the_same_file_spelled_differently_is_still_the_same_file(self):
        one = os.path.join(self.dir, "both.txt")
        spelled = os.path.join(self.dir, ".", "sub", "..", "both.txt")
        os.makedirs(os.path.join(self.dir, "sub"))
        err = self.run_main("--out", one, "--report", spelled, "--source", self.source)
        self.assertIn("must use different paths", err)

    def test_out_cannot_overwrite_a_source(self):
        err = self.run_main("--out", self.source, "--report", os.path.join(self.dir, "r.json"),
                            "--source", self.source)
        self.assertIn(f"--out {self.source} would overwrite the input {self.source}", err)

    def test_report_cannot_overwrite_a_source(self):
        err = self.run_main("--out", os.path.join(self.dir, "out.h"), "--report", self.source,
                            "--source", self.source)
        self.assertIn(f"--report {self.source} would overwrite the input {self.source}", err)

    @unittest.skipUnless(os.name == "nt", "letter case only folds on Windows and case-insensitive volumes")
    def test_a_case_different_spelling_is_one_file(self):
        err = self.run_main("--out", os.path.join(self.dir, "Out.H"), "--report",
                            os.path.join(self.dir, "out.h"), "--source", self.source)
        self.assertIn("must use different paths", err)

    def test_distinct_paths_still_write_both_artifacts(self):
        out = os.path.join(self.dir, "out.h")
        report = os.path.join(self.dir, "r.json")
        with mock.patch("sys.argv", ["gen_d3d12_stubs.py", "--header", self.header, "--out", out,
                                     "--report", report, "--source", self.source, "IChild"]):
            self.assertEqual(gen.main(), 0)
        with open(out, encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("// Generated by tools/gen_d3d12_stubs.py"))
        with open(report, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["implemented"], 1)   # still a report, not a header


if __name__ == "__main__":
    unittest.main()
