"""tools/report_paths.py, the one `same_report_path` the tools share, and the tools that use it.

    python -I tests/test_report_paths.py

Standard library only; everything runs in a temporary folder on hand-built paths, no game data. The macOS
case-insensitive-volume branch is exercised with patches, so it runs on every platform.
"""
import ast
import os
import pathlib
import posixpath
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import report_paths  # noqa: E402

# Tools that use report_paths and must not define their own copy again.
SHARED = ("anim_index", "awc_manifest", "gamedata_index", "gamedata_xml", "gen_d3d12_stubs", "glb_inspect",
          "navmesh_index", "rpf_probe", "tile_probe", "world_lights_paths", "world_names")
# The one tool that still carries its own copy (an open pull request edits its call site). Do not add to this list:
# import report_paths instead.
LEGACY_COPIES = {"navmesh_decode"}


def touch(path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


class SamePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.realpath(self.tmp.name)

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    def test_same_spelling_and_other_spelling(self):
        touch(self.p("a", "r.json"))
        self.assertTrue(report_paths.same_report_path(self.p("a", "r.json"), self.p("a", "r.json")))
        self.assertTrue(report_paths.same_report_path(self.p("a", "r.json"), self.p("a", "..", "a", ".", "r.json")))

    def test_different_files(self):
        touch(self.p("a.json"))
        touch(self.p("b.json"))
        self.assertFalse(report_paths.same_report_path(self.p("a.json"), self.p("b.json")))

    def test_paths_that_do_not_exist_yet(self):
        self.assertTrue(report_paths.same_report_path(self.p("new", "x.json"), self.p("new", "..", "new", "x.json")))
        self.assertFalse(report_paths.same_report_path(self.p("new", "x.json"), self.p("new", "y.json")))

    def test_one_missing_one_present_is_not_an_error(self):
        touch(self.p("a.json"))
        self.assertFalse(report_paths.same_report_path(self.p("a.json"), self.p("missing.json")))
        self.assertFalse(report_paths.same_report_path(self.p("missing.json"), self.p("a.json")))

    def test_hard_link(self):
        touch(self.p("a.json"))
        try:
            os.link(self.p("a.json"), self.p("link.json"))
        except (OSError, NotImplementedError, AttributeError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assertTrue(report_paths.same_report_path(self.p("a.json"), self.p("link.json")))

    def test_symlink(self):
        touch(self.p("a.json"))
        try:
            os.symlink(self.p("a.json"), self.p("sym.json"))
        except (OSError, NotImplementedError, AttributeError) as e:
            self.skipTest(f"symlinks unavailable: {e}")
        self.assertTrue(report_paths.same_report_path(self.p("a.json"), self.p("sym.json")))
        self.assertTrue(report_paths.same_report_path(self.p("sym.json"), self.p("a.json")))

    def test_case_only_difference_follows_the_volume(self):
        # On a volume that ignores case (Windows, macOS's default) `B.json` and `b.json` are one file; elsewhere two.
        touch(self.p("probe.tmp"))
        insensitive = os.path.exists(self.p("PROBE.TMP"))
        got = report_paths.same_report_path(self.p("new", "B.json"), self.p("new", "b.json"))
        self.assertEqual(got, insensitive or os.path.normcase("A") == os.path.normcase("a"))


class MacVolumeTests(unittest.TestCase):
    """The macOS branch: pathconf says whether the volume folds case. Patched, so it runs anywhere."""

    def darwin(self, case_sensitive_result):
        fake_os = mock.Mock(wraps=os, pathconf=mock.Mock(return_value=case_sensitive_result))
        # macOS's normcase keeps case. The host's would fold it on Windows, before the volume check this class tests.
        fake_os.path.normcase = posixpath.normcase
        return mock.patch.multiple(report_paths, sys=mock.Mock(platform="darwin"), os=fake_os)

    def test_not_macos_is_never_case_insensitive(self):
        with mock.patch.object(report_paths, "sys", mock.Mock(platform="linux")):
            self.assertFalse(report_paths.case_insensitive_volume(os.getcwd()))

    def test_macos_case_insensitive_volume_folds_case_and_accents(self):
        with self.darwin(0):
            self.assertTrue(report_paths.case_insensitive_volume(os.getcwd()))
            self.assertEqual(report_paths.path_key("/tmp/leonida_x/R.json"), report_paths.path_key("/tmp/leonida_x/r.json"))
            self.assertEqual(report_paths.path_key("/tmp/leonida_x/café.json"),
                             report_paths.path_key("/tmp/leonida_x/café.json"))
            self.assertTrue(report_paths.same_report_path("/tmp/leonida_x/R.json", "/tmp/leonida_x/r.json"))

    def test_macos_case_sensitive_volume_keeps_case(self):
        with self.darwin(1):
            self.assertFalse(report_paths.case_insensitive_volume(os.getcwd()))
            self.assertNotEqual(report_paths.path_key("/tmp/leonida_x/R.json"), report_paths.path_key("/tmp/leonida_x/r.json"))

    def test_macos_pathconf_failure_counts_as_case_insensitive(self):
        # The macOS default; folding can only refuse a pair of reports, never let one overwrite the other.
        for error in (OSError("no such file"), ValueError("bad name")):
            with self.subTest(error=type(error).__name__), mock.patch.multiple(
                    report_paths, sys=mock.Mock(platform="darwin"),
                    os=mock.Mock(wraps=os, pathconf=mock.Mock(side_effect=error))):
                self.assertTrue(report_paths.case_insensitive_volume(os.getcwd()))

    def test_macos_probe_walks_up_to_an_existing_folder(self):
        with self.darwin(0):
            self.assertTrue(report_paths.case_insensitive_volume("/definitely/not/there/at/all/x.json"))
            asked = report_paths.os.pathconf.call_args[0][0]
            self.assertTrue(os.path.exists(asked), asked)


class FoldingVolumeTests(unittest.TestCase):
    """A volume that ignores case without normcase or pathconf saying so is found with one scratch folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.realpath(self.tmp.name)

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    def volume_folds_case(self):
        touch(self.p("probe.tmp"))
        return os.path.exists(self.p("PROBE.TMP"))

    def test_case_only_difference_follows_the_volume(self):
        os.makedirs(self.p("reports"))
        self.assertEqual(report_paths.same_name_on_folding_volume(self.p("reports", "R.json"),
                                                                 self.p("reports", "r.json")),
                         self.volume_folds_case())

    def test_other_spellings_of_one_folder_are_one_report(self):
        os.makedirs(self.p("reports"))
        self.assertTrue(report_paths.same_name_on_folding_volume(self.p("reports", "R.json"),
                                                                 self.p("reports", ".", "R.json")))

    def test_names_that_differ_by_more_than_case_never_ask_the_volume(self):
        os.makedirs(self.p("reports"))
        with mock.patch.object(report_paths.tempfile, "TemporaryDirectory",
                               side_effect=AssertionError("asked the volume")):
            self.assertFalse(report_paths.same_name_on_folding_volume(self.p("reports", "out.json"),
                                                                     self.p("reports", "lights.json")))

    def test_it_leaves_no_scratch_folder_behind(self):
        os.makedirs(self.p("reports"))
        report_paths.same_name_on_folding_volume(self.p("reports", "R.json"), self.p("reports", "r.json"))
        self.assertEqual(sorted(os.listdir(self.p("reports"))), [])

    def test_a_folder_that_does_not_exist_is_not_created(self):
        self.assertFalse(report_paths.same_name_on_folding_volume(self.p("missing", "R.json"),
                                                                  self.p("missing", "r.json")))
        self.assertFalse(os.path.exists(self.p("missing")))

    def test_same_report_path_asks_the_volume_when_the_name_rules_disagree(self):
        # The name rules say "different"; the volume has the last word on a pair that differs only by case.
        with mock.patch.object(report_paths, "_resolved_key", side_effect=os.path.realpath), \
                mock.patch.object(report_paths, "_same_resolved_name", return_value=True) as probe:
            self.assertTrue(report_paths.same_report_path(self.p("R.json"), self.p("r.json")))
        probe.assert_called_once_with(os.path.realpath(self.p("R.json")), os.path.realpath(self.p("r.json")))

    def test_each_path_is_resolved_once_per_call(self):
        # #6220's review: the key and the volume probe both want the resolved names, so resolving each path
        # once costs two realpath calls per guard, not four. A guard runs once per input file, so this is the
        # whole cost: on main the same call resolves four times.
        with mock.patch.object(report_paths.os.path, "realpath", wraps=os.path.realpath) as resolved:
            report_paths.same_report_path(self.p("out.json"), self.p("lights.json"))
        self.assertEqual(resolved.call_count, 2, "each path is resolved once, not once per check")


class NoPrivateCopiesTests(unittest.TestCase):
    def functions(self, tool):
        tree = ast.parse((TOOLS / f"{tool}.py").read_text(encoding="utf-8"))
        return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}

    def test_switched_tools_import_the_helper_and_define_no_copy(self):
        for tool in SHARED:
            with self.subTest(tool=tool):
                source = (TOOLS / f"{tool}.py").read_text(encoding="utf-8")
                self.assertIn("from report_paths import same_report_path", source)
                names = self.functions(tool)
                self.assertFalse({"same_report_path", "_same_report_path", "_path_key", "_case_insensitive_volume"} & names,
                                 f"{tool} defines its own copy; use tools/report_paths.py")

    def test_no_new_copies_of_same_report_path(self):
        copies = set()
        for path in sorted(TOOLS.glob("*.py")):
            if path.name == "report_paths.py":
                continue
            if any("same_report_path" in name for name in self.functions(path.stem)):
                copies.add(path.stem)
        self.assertEqual(copies - LEGACY_COPIES, set(),
                         "a tool defines its own same_report_path: import it from tools/report_paths.py instead")

    def test_tools_still_expose_same_report_path(self):
        # gamedata_xml.py and others call it by this name (and tests or sibling tools may import it).
        for tool in ("gamedata_index", "gamedata_xml", "navmesh_index"):
            with self.subTest(tool=tool):
                module = __import__(tool)
                self.assertIs(module.same_report_path, report_paths.same_report_path)


if __name__ == "__main__":
    unittest.main()
