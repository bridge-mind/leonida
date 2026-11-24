"""tools/progress_dashboard.py: parsers, classification, the no-game-data rule and the output-folder rule.

    python -I tests/test_progress_dashboard.py

Every input is written here by hand (Markdown tables, JUnit XML, a CTest log, a fake work folder whose shader names
are sentinels). No game files, no GPU, no network: the GitHub path is fed synthetic gh output.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import progress_dashboard as pd  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "progress_dashboard.py")
SENTINEL = "SENTINEL_SHADER_NAME_must_not_appear"


class Markdown(unittest.TestCase):
    def test_tables_keep_pipes_inside_code(self):
        t = pd.md_tables("## Translator status and runtime contract\n\n| Piece | Contract |\n|---|---|\n| A | `x|y` is **GPU-checked** |\n")
        self.assertEqual(len(t), 1)
        heading, header, rows, line = t[0]
        self.assertEqual(heading, "Translator status and runtime contract")
        self.assertEqual(rows[0][1], "`x|y` is **GPU-checked**")
        self.assertEqual(line, 3)

    def test_plain_strips_links_and_emphasis(self):
        self.assertEqual(pd.plain("**GPU-checked** ([measurement](http://x)) `v_mac_f32`"), "GPU-checked (measurement) v_mac_f32")


class Contract(unittest.TestCase):
    def test_classification_order_and_evidence(self):
        cases = {
            "**not yet GPU-verified**; Source: LLVM `SIInstrInfo.cpp`": ("not_gpu", "partial"),
            "**GPU-checked** (TEST-298) on an NVIDIA GeForce RTX 2060": ("gpu", "verified"),
            "Source: LLVM `VOPProfileI2F`": ("reference", "implemented"),
            "a count over the corpus: 0 of the 26,503 shaders": ("corpus", "implemented"),
            "CPU-checked by DenormalFlushTests": ("cpu", "implemented"),
            "The translator emits OpFAdd.": ("unstated", "unknown"),
        }
        for text, (kind, status) in cases.items():
            with self.subTest(text=text):
                k, s, words = pd.classify_contract(text)
                self.assertEqual((k, s), (kind, status))
                if kind != "unstated":
                    self.assertIn(words.lower(), text.lower())

    def test_open_items_are_detected_only_by_explicit_words(self):
        self.assertTrue(pd.OPEN.search("Still open: on some inputs the result is undefined"))
        self.assertFalse(pd.OPEN.search("s_getreg_b32 is unsupported and there are no traps"))


class Tests(unittest.TestCase):
    def test_junit_honors_each_reports_encoding(self):
        xml = '<?xml version="1.0" encoding="{}"?><testsuite><testcase name="{}"><failure/></testcase></testsuite>'
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for codec, declared, name in (("utf-16", "UTF-16", "日本語"), ("latin-1", "ISO-8859-1", "café"),
                                          ("ascii", "not-a-codec", "bad")):
                path = os.path.join(root, codec + ".xml")
                with open(path, "wb") as f:
                    f.write(xml.format(declared, name).encode(codec))
                paths.append(path)
            col = pd.Collector()
            with mock.patch.object(pd, "ROOT", root), mock.patch.object(pd, "last_commit_time", return_value=None):
                pd.collect_ctest(col, SimpleNamespace(junit=paths))
        self.assertEqual([(c["name"], c["status"]) for c in col.components], [("日本語", "failed"), ("café", "failed")])
        self.assertEqual([s["status"] for s in col.sources if s["id"] == "junit"], ["ok", "ok", "invalid"])

    def test_unittest_file_is_skipped_only_when_all_cases_skip(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pd, "ROOT", directory):
            file = os.path.join(directory, "test_sample.py")
            for skipped in (0, 1, 2):
                with self.subTest(skipped=skipped):
                    with open(file, "w", encoding="utf-8", newline="\n") as f:
                        f.write(f"import unittest\nclass Sample(unittest.TestCase):\n"
                                f"    @unittest.skipIf({skipped} > 0, 'synthetic')\n"
                                f"    def test_first(self): pass\n"
                                f"    @unittest.skipIf({skipped} > 1, 'synthetic')\n"
                                f"    def test_second(self): pass\n"
                                f"unittest.main()\n")
                    result = pd.run_one_test("test_sample.py", 15)
                    self.assertEqual(result["result"], "pass", result)
                    self.assertEqual(result["cases"], 2)
                    self.assertEqual(result["all_skipped"], skipped == 2)

    def test_junit_ctest_and_pytest_flavours(self):
        ctest = '<testsuite><testcase name="loader" classname="loader" status="run" time="0.9"/>' \
                '<testcase name="xmem" classname="xmem" status="fail"><failure message="C:\\Users\\me\\x\\out.txt bad"/></testcase></testsuite>'
        rows = pd.parse_junit(ctest)
        self.assertEqual([(n, r) for n, r, _, _ in rows], [("loader", "pass"), ("xmem", "fail")])
        self.assertNotIn("me", rows[1][3])          # paths are reduced to their last component
        pytest = '<testsuites><testsuite><testcase classname="tests.a" name="t[1-2]"/>' \
                 '<testcase classname="tests.a" name="t[1_2]"><skipped/></testcase></testsuite></testsuites>'
        self.assertEqual([(n, r) for n, r, _, _ in pd.parse_junit(pytest)], [("tests.a::t[1-2]", "pass"), ("tests.a::t[1_2]", "skip")])

    def test_lasttest_log(self):
        log = "Start testing: Nov 22 10:00 X\n1/2 Test: loader\nTest time =   0.91 sec\nTest Passed.\n2/2 Test: xmem\nTest Failed.\n"
        self.assertEqual(pd.parse_lasttest(log), [("loader", "pass", 0.91), ("xmem", "fail", None)])

    def test_lasttest_uses_ctest_verdicts_after_captured_output(self) -> None:
        cases = (
            ("required", "ready", "Test Pass Reason:\nRequired regular expression found. Regex=[ready]", "pass"),
            ("forbidden", "Test Passed.\nerror", "Test Fail Reason:\nError regular expression found in output. Regex=[error]", "fail"),
            ("missing", "stopped", "Test Fail Reason:\nRequired regular expression not found. Regex=[ready]", "fail"),
            ("unavailable", "unavailable", "Test Pass Reason:\nSkip regular expression found in output. Regex=[unavailable]", "skip"),
            ("ordinary", "Test Fail Reason:\nunrelated test output", "Test Passed.", "pass"),
        )
        log = "\n".join(
            f"{index}/{len(cases)} Test: {name}\nOutput:\n{'-' * 58}\n{output}\n<end of output>\n"
            f"Test time =   0.01 sec\n{'-' * 58}\n{verdict}\n"
            for index, (name, output, verdict, _expected) in enumerate(cases, 1)
        )
        self.assertEqual(pd.parse_lasttest(log), [(name, expected, 0.01) for name, _, _, expected in cases])

    def test_lasttest_ignores_a_nested_ctest_log(self) -> None:
        separator = "-" * 58
        inner = (
            f"Start testing: Nov 22 10:00 X\n{separator}\n1/1 Testing: outer\n1/1 Test: outer\n"
            f"Output:\n{separator}\n<end of output>\nTest time =   0.01 sec\n{separator}\n"
            f'Test Passed.\n"outer" end time: Nov 22 10:00 X\n"outer" time elapsed: 00:00:00\n'
            f"{separator}\n\nEnd testing: Nov 22 10:00 X"
        )
        log = (
            f"Start testing: Nov 22 10:00 X\n{separator}\n1/2 Testing: outer\n1/2 Test: outer\n"
            f"Output:\n{separator}\n{inner}\nouter fails\n<end of output>\n"
            f'Test time =   0.23 sec\n{separator}\nTest Failed.\n"outer" end time: Nov 22 10:00 X\n'
            f'"outer" time elapsed: 00:00:00\n{separator}\n\n2/2 Test: next\n'
            "Test time =   0.02 sec\nTest Passed.\n"
        )
        self.assertEqual(pd.parse_lasttest(log), [("outer", "fail", 0.23), ("next", "pass", 0.02)])

    def test_ci_selection_matches_the_workflow_rule(self):
        files = pd.ci_test_files()
        self.assertIn("tests/test_progress_dashboard.py", files)
        self.assertFalse(any(f.startswith("tests/semantic/") for f in files))
        for f in files[:20]:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("__main__", text)
            self.assertIsNone(pd.PYTEST_MARKERS.search(text))

    def test_unlistable_subfolder_raises_instead_of_a_partial_scan(self):
        # os.walk skips a directory it cannot list unless onerror raises; the selection and the digest
        # used to silently drop the sub-folder and report the rest as complete.
        def walk(top, onerror=None, **kwargs):
            if onerror is not None:
                onerror(PermissionError(13, "Permission denied", os.path.join(top, "locked")))
            return
            yield  # pragma: no cover

        with mock.patch.object(pd.os, "walk", side_effect=walk):
            with self.assertRaises(PermissionError):
                pd.ci_test_files()
            with self.assertRaises(PermissionError):
                pd.code_digest()


class Freshness(unittest.TestCase):
    def collect(self, results):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "unit_results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f)
            col = pd.Collector()
            pd.collect_unit(col, out, type("A", (), {"run_tests": False})())
            return {c["name"]: (c["status"], c["freshness"]) for c in col.components}

    def base(self, digest, code):
        return {"ran_at": "2026-11-22T10:00:00Z", "commit": None, "code_digest": code,
                "results": [{"test": "tests/test_progress_dashboard.py", "result": "pass", "digest": digest}]}

    def test_unchanged_test_and_code_is_current(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), pd.code_digest()))
        self.assertEqual(got["test_progress_dashboard.py"], ("verified", "current"))

    def test_changed_test_file_needs_revalidation(self):
        got = self.collect(self.base("0" * 16, pd.code_digest()))
        self.assertEqual(got["test_progress_dashboard.py"], ("partial", "revalidation_required"))

    def test_changed_tools_need_revalidation(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), "0" * 16))
        self.assertEqual(got["test_progress_dashboard.py"], ("partial", "revalidation_required"))

    def test_changed_shader_helper_needs_revalidation(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(pd, "ROOT", root):
            for directory in ("tools", "scripts", "runtime/test/shaders", "tests"):
                Path(root, directory).mkdir(parents=True)
            test = "tests/test_progress_dashboard.py"
            Path(root, test).touch()
            helper = Path(root, "runtime/test/shaders/compute_fixture.py")
            helper.write_bytes(b"shift = 0\n")
            cached = self.base(pd.file_digest(test), pd.code_digest())
            self.assertEqual(self.collect(cached)["test_progress_dashboard.py"], ("verified", "current"))

            helper.write_bytes(b"shift = 1\n")
            self.assertEqual(self.collect(cached)["test_progress_dashboard.py"], ("partial", "revalidation_required"))

    def test_code_edited_during_the_run_is_never_current(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), None))
        self.assertEqual(got["test_progress_dashboard.py"], ("verified", "unknown"))

    def test_ctest_junit_passes_age_like_lasttest_log(self):
        ctest_suite = '<testsuite disabled="0" hostname="" timestamp="2026-11-22T10:00:00">{}</testsuite>'
        sources = {
            "runtime/build/Testing/Temporary/LastTest.log": (100, "1/1 Test: local\nTest Passed.\n"),
            "before.xml": (100, ctest_suite.format(
                '<testcase name="older" status="run"/><testcase name="failed" status="fail"/>'
                '<testcase name="skipped" status="notrun"/>')),
            "after.xml": (300, ctest_suite.format('<testcase name="newer" status="run"/>')),
            "pytest.xml": (100, '<testsuites><testsuite name="pytest" hostname="host">'
                           '<testcase name="unrelated"/></testsuite></testsuites>'),
        }
        with tempfile.TemporaryDirectory() as root:
            junit = []
            for name, (mtime, text) in sources.items():
                path = os.path.join(root, *name.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                os.utime(path, (mtime, mtime))
                if name.endswith(".xml"):
                    junit.append(path)
            col = pd.Collector()
            with mock.patch.object(pd, "ROOT", root), mock.patch.object(pd, "last_commit_time", return_value=200):
                pd.collect_ctest(col, SimpleNamespace(junit=junit))

        outcomes = {c["name"]: (c["status"], c["freshness"]) for c in col.components}
        self.assertEqual(outcomes, {
            "local": ("partial", "revalidation_required"),
            "older": ("partial", "revalidation_required"),
            "failed": ("failed", "failed"),
            "skipped": ("unknown", "unknown"),
            "newer": ("verified", "current"),
            "unrelated": ("verified", "current"),
        })
        metric = next(m for m in col.metrics if m["id"] == "runtime.current")
        self.assertEqual((metric["numerator"], metric["denominator"]), (2, 6))


class GpuLog(unittest.TestCase):
    LOG = "\n".join([
        "using runtime/build/RelWithDebInfo",
        "cs_bounds            ok",
        "ps_texoff            SKIP (VK_KHR_maintenance8 missing)",
        "cs_bpermute32        lane 3: got 0x00000000, want 0x00000007",
        "2 of 3 pass, 1 skipped",
    ])

    def test_lines_and_summary(self):
        rows, summary = pd.parse_gpu_log(self.LOG)
        self.assertEqual([(n, r) for n, r, _ in rows], [("cs_bounds", "pass"), ("ps_texoff", "skip"), ("cs_bpermute32", "fail")])
        self.assertIn("want 0x00000007", rows[2][2])
        self.assertEqual(summary, (2, 3, 1))

    def test_panel_groups_by_gpu_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "check.txt")
            with open(log, "w", encoding="utf-8", newline="\n") as f:
                f.write(self.LOG + "\n")
            col = pd.Collector()
            pd.collect_gpu(col, [log + "=RTX 4060, 576.02, Windows 11"])
            self.assertEqual({c["group"] for c in col.components}, {"RTX 4060, 576.02, Windows 11"})
            self.assertEqual(sorted(c["status"] for c in col.components)[0], "failed")
            m = next(m for m in col.metrics if m["id"] == "gpu.current")
            self.assertEqual(m["denominator"], 2)                     # the skip is not counted


class Github(unittest.TestCase):
    def test_d3d_states(self):
        self.assertEqual(pd.d3d_state("@a", "#291 merged, #819, #831 on main", "")[0], "partial")
        self.assertEqual(pd.d3d_state("@a", "#1489 merged, #907 closed", "")[0], "implemented")
        self.assertEqual(pd.d3d_state("@a", "—", "")[0], "unknown")
        self.assertEqual(pd.d3d_state("—", "—", "")[0], "missing")

    def test_d3d_states_follow_the_merged_list(self):
        merged, open_prs = {819, 831, 291}, {900}
        # the table still lists #819 without "merged", but it merged: the item is implemented
        self.assertEqual(pd.d3d_state("@a", "#291 merged, #819", "PR open", merged, open_prs)[0], "implemented")
        self.assertEqual(pd.d3d_state("@a", "#819, #900", "", merged, open_prs)[0], "partial")
        # a bare reference that is neither merged nor open was closed without merging: it does not count
        self.assertEqual(pd.d3d_state("@a", "#777", "", merged, open_prs)[0], "unknown")
        self.assertEqual(pd.d3d_state("@a", "—", "PR open", merged, open_prs)[0], "partial")

    def test_item_ids(self):
        self.assertEqual(pd.item_ids("D3D-86: answer D3D12_OPTIONS1 in CheckFeatureSupport"), ["D3D-86"])
        self.assertEqual(pd.item_ids("pknorm_probe: NaN gives 0 (SHADER-339)"), ["SHADER-339"])
        self.assertEqual(pd.item_ids("Docs/Tests: SHADER-4 and SHADER-4 again, TEST-9"), ["SHADER-4", "TEST-9"])
        self.assertEqual(pd.item_ids("hash with SHA-256, write UTF-8, read OPTIONS2-1 and D3D12-3"), [])
        self.assertEqual(pd.item_ids(None), [])

    def test_d3d_metric_counts_merged_pull_requests_not_only_table_rows(self):
        body = ("| ID | Title | Owner | PR | Status |\n|---|---|---|---|---|\n"
                "| D3D-1 | a | @a | #50 | PR open |\n| D3D-2 | b | @b | #51 | PR open |\n")
        merged = [{"number": n, "title": f"D3D-{n}: x", "mergedAt": "2026-11-22T10:00:00Z", "author": {"login": "x"}}
                  for n in (10, 11, 12)]
        open_prs = [{"number": 50, "title": "D3D-1: a", "isDraft": False, "author": {"login": "x"}},
                    {"number": 51, "title": "D3D-2: b", "isDraft": False, "author": {"login": "x"}}]

        def fake_gh(args, timeout=60):
            if args[:2] == ["pr", "list"]:
                return merged if "merged" in args else open_prs
            if args[:2] == ["issue", "view"]:
                return {"body": body, "updatedAt": "2026-11-22T00:00:00Z"}
            return []
        col = pd.Collector()
        with mock.patch.object(pd, "gh_json", fake_gh), mock.patch.object(pd.shutil, "which", lambda _: "/usr/bin/gh"):
            pd.collect_github(col, "o/r", "me")
        m = next(m for m in col.metrics if m["id"] == "d3d12.merged")
        self.assertEqual((m["numerator"], m["denominator"]), (3, 5))

    def run_d3d_metric(self, body, merged, open_prs):
        """The d3d12.merged metric for a #313 body and the repository's merged and open PR lists (None: unreadable)."""
        def fake_gh(args, timeout=60):
            if args[:2] == ["pr", "list"]:
                return merged if "merged" in args else open_prs
            if args[:2] == ["issue", "view"]:
                return {"body": body, "updatedAt": "2026-11-22T00:00:00Z"}
            return []
        col = pd.Collector()
        with mock.patch.object(pd, "gh_json", fake_gh), mock.patch.object(pd.shutil, "which", lambda _: "/usr/bin/gh"):
            pd.collect_github(col, "o/r", "me")
        return next(m for m in col.metrics if m["id"] == "d3d12.merged")

    @staticmethod
    def prs(*pairs):
        return [{"number": n, "title": t, "isDraft": False, "mergedAt": "2026-11-22T10:00:00Z", "author": {"login": "x"}}
                for n, t in pairs]

    def test_d3d_id_with_an_open_pr_is_not_counted_as_merged(self):
        body = ("| ID | Title | Owner | PR | Status |\n|---|---|---|---|---|\n"
                "| D3D-3 | a | @a | #819 | PR open |\n| D3D-4 | b | @b | #60 merged | merged |\n")
        merged = self.prs((291, "D3D-3: first part"), (60, "D3D-4: b"), (292, "D3D-20: c"), (293, "D3D-21: d"))
        open_prs = self.prs((819, "D3D-3: rest"), (820, "D3D-20: more"))
        m = self.run_d3d_metric(body, merged, open_prs)
        # D3D-4 (table: merged) and D3D-21 (merged PR, no open PR); not D3D-3 (table: open) or D3D-20 (open PR)
        self.assertEqual((m["numerator"], m["denominator"]), (2, 4))
        self.assertNotIn("note", m)

    def test_d3d_metric_says_so_when_a_pr_list_could_not_be_read(self):
        body = ("| ID | Title | Owner | PR | Status |\n|---|---|---|---|---|\n"
                "| D3D-4 | b | @b | #60 merged | merged |\n| D3D-5 | c | @c | #61 | PR open |\n")
        m = self.run_d3d_metric(body, self.prs((70, "D3D-9: x")), None)
        self.assertEqual((m["numerator"], m["denominator"]), (1, 2))   # table rows only
        self.assertIn("partial", m["title"])
        self.assertIn("could not be read", m["note"])

    def test_merged_pull_requests_are_implemented(self):
        merged = [{"number": 10, "title": "D3D-1: a", "mergedAt": "2026-11-22T10:00:00Z", "author": {"login": "x"}},
                  {"number": 12, "title": "docs: b", "mergedAt": "2026-11-22T12:00:00Z", "author": {"login": "y"}}]
        open_prs = [{"number": 13, "title": "D3D-2: c", "isDraft": True, "author": {"login": "z"}}]
        col = pd.Collector()
        latest = pd.collect_pulls(col, "o/r", merged, open_prs)
        got = {c["number"]: (c["status"], c["group"]) for c in col.components}
        self.assertEqual(got, {10: ("implemented", "D3D items"), 12: ("implemented", "no item ID"), 13: ("partial", "D3D items")})
        self.assertEqual([p["number"] for p in latest], [12, 10])          # newest merge first
        m = {m["id"]: (m["numerator"], m["denominator"]) for m in col.metrics}
        self.assertEqual(m["merged.prs"], (2, 3))
        self.assertEqual(m["merged.items"], (1, 2))
        self.assertNotIn("merged_at", next(c for c in col.components if c["number"] == 13))

    def test_pull_request_limit_is_reported(self):
        col = pd.Collector()
        pd.collect_pulls(col, "o/r", [{"number": 1, "title": "a", "mergedAt": "x"}], [], limit=1)
        self.assertIn("limit was reached", col.sources[0]["note"])


class Corpus(unittest.TestCase):
    def test_counts_only_never_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            sh = os.path.join(work, "shaders")      # make_game.py's layout
            os.makedirs(os.path.join(sh, "dxbc"))
            os.makedirs(os.path.join(sh, "spirv"))
            carved = [{"name": f"{SENTINEL}_{i}", "stage": "pixel" if i < 3 else "compute"} for i in range(5)]
            with open(os.path.join(sh, "dxbc", "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(carved, f)
            with open(os.path.join(sh, "spirv", "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({f"{SENTINEL}_0": {"stage": "pixel"}, f"{SENTINEL}_3": {"stage": "compute"},
                           f"{SENTINEL}_4": {"error": "x"}}, f)
            col = pd.Collector()
            pd.collect_corpus(col, work)
            got = {(c["group"], c["status"]): c["count"] for c in col.components}
            self.assertEqual(got, {("pixel", "implemented"): 1, ("pixel", "failed"): 2,
                                   ("compute", "implemented"): 1, ("compute", "failed"): 1})
            m = next(m for m in col.metrics if m["id"] == "corpus.validated")
            self.assertEqual((m["numerator"], m["denominator"]), (2, 5))
            # everything the collector holds can reach the page (notes and panels included), so check all of it
            held = json.dumps(vars(col), default=str)
            self.assertNotIn(SENTINEL, held)
            self.assertNotIn(tmp, held)
            self.assertNotIn(json.dumps(tmp)[1:-1], held)


class Scrub(unittest.TestCase):
    def test_paths_outside_the_repository_lose_their_file_name(self):
        sep = os.sep
        work = os.path.join(tempfile.gettempdir(), "work", "shaders", "dxbc", f"{SENTINEL}.fssm")
        self.assertEqual(pd.scrub(f"FAIL: cannot read {work}"), "FAIL: cannot read <path>")
        self.assertEqual(pd.scrub(f"see /home/someone/work/{SENTINEL}.fssm now"), "see <path> now")
        self.assertEqual(pd.scrub("in " + os.path.join(pd.HOME, "work", f"{SENTINEL}.fssm")), "in <path>")
        self.assertEqual(pd.scrub(f"C:{sep}work{sep}{SENTINEL}_a.fssm, C:{sep}{SENTINEL}_b.fssm"), "<path> <path>")

    def test_paths_inside_the_repository_become_relative(self):
        line = "Error: " + os.path.join(ROOT, "tools", "text_tables.py") + " line 3"
        self.assertEqual(pd.scrub(line), "Error: tools/text_tables.py line 3")

    def test_checkout_path_with_spaces_becomes_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = os.path.join(tmp, "Checkout with spaces [1]")
            path = os.path.join(checkout, "tools", "text_tables.py")
            os.makedirs(os.path.dirname(path))
            with open(path, "wb"):
                pass
            with mock.patch.object(pd, "ROOT", checkout):
                self.assertEqual(pd.scrub(f"Error: {path} line 3"), "Error: tools/text_tables.py line 3")
                self.assertEqual(pd.scrub(f'File "{path}", line 3'), 'File "tools/text_tables.py", line 3')
                self.assertEqual(pd.scrub(f"Checkout: {checkout}"), "Checkout: .")
                self.assertEqual(pd.scrub(f"Checkout {checkout} is ready"), "Checkout . is ready")
                sibling = os.path.join(checkout + "x", f"{SENTINEL}.fssm")
                self.assertEqual(pd.scrub(f"Error: {sibling} line 3"), "Error: <path> line 3")
                if os.name == "nt":
                    alternate = os.path.join(checkout.swapcase(), "tools", "text_tables.py").replace("\\", "/")
                    self.assertEqual(pd.scrub(f"Error: {alternate} line 3"), "Error: tools/text_tables.py line 3")

    def test_urls_and_relative_paths_stay(self):
        self.assertEqual(pd.scrub("see https://github.com/ItsNotRexed/Leonida/issues/1"),
                         "see https://github.com/ItsNotRexed/Leonida/issues/1")
        self.assertEqual(pd.scrub("tests/test_x.py failed"), "tests/test_x.py failed")


class Outputs(unittest.TestCase):
    def test_nonpositive_test_options_are_argument_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            for option in ("--jobs", "--test-timeout"):
                for value in ("0", "-1"):
                    with self.subTest(option=option, value=value):
                        cmd = [sys.executable, "-I", TOOL, "--out", out, "--run-tests",
                               "--tests", "test_progress_dashboard", option, value]
                        result = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=120)
                        self.assertEqual(result.returncode, 2, result.stderr.decode("utf-8", "replace"))
                        self.assertIn(f"{option} must be a positive integer".encode("ascii"), result.stderr)
                        self.assertFalse(os.path.exists(out))

    def test_invalid_test_filter_is_an_argument_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            result = subprocess.run([sys.executable, "-I", TOOL, "--out", out, "--run-tests", "--tests", "["],
                                    capture_output=True, cwd=ROOT, timeout=120)
            self.assertEqual(result.returncode, 2, result.stderr.decode("utf-8", "replace"))
            self.assertIn(b"--tests", result.stderr)
            self.assertIn(b"unterminated character set", result.stderr)
            self.assertNotIn(b"Traceback", result.stderr)
            self.assertFalse(os.path.exists(out))

    def test_out_inside_the_repository_is_refused(self):
        r = subprocess.run([sys.executable, "-I", TOOL, "--out", os.path.join(ROOT, "progress_out_test")],
                           capture_output=True, cwd=ROOT, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "progress_out_test")))

    def test_save_text_refreshes_when_fixed_temp_name_is_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            stale = out + ".tmp"
            os.mkdir(stale)
            pd.save_text(out, '{"ok": 1}\n')
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"ok": 1}\n')
            self.assertTrue(os.path.isdir(stale))
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith("report.json.")],
                             ["report.json.tmp"])

    def test_page_snapshot_history_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            cmd = [sys.executable, "-I", TOOL, "--out", out]
            r1 = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=300)
            self.assertEqual(r1.returncode, 0, r1.stderr.decode("utf-8", "replace"))
            with open(os.path.join(out, "progress.html"), encoding="utf-8") as f:
                page = f.read()
            self.assertNotIn("/*__PROGRESS_DATA__*/null", page)
            self.assertNotIn("/*__PROGRESS_LOGO__*/", page)
            if os.path.isfile(pd.LOGO):                              # the logo is embedded: the page needs no other file
                self.assertIn('src="data:image/png;base64,', page)
            self.assertIn('"schema": "leonida.progress/1"', page)
            with open(os.path.join(out, "progress.html"), "rb") as f:
                self.assertNotIn(b"\r\n", f.read())             # LF on every platform
            r2 = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=300)
            self.assertIn(b"no change since the last snapshot", r2.stdout)
            self.assertEqual(len(os.listdir(os.path.join(out, "snapshots"))), 1)
            with open(os.path.join(out, "history.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 1)

    def test_unmatched_test_filter_preserves_cached_results(self):
        with tempfile.TemporaryDirectory() as out:
            cached = os.path.join(out, "unit_results.json")
            previous = b'{"results":[{"test":"tests/test_progress_dashboard.py","result":"pass"}]}\n'
            with open(cached, "wb") as f:
                f.write(previous)
            cmd = [sys.executable, "-I", TOOL, "--out", out, "--run-tests", "--tests", "^no-such-fixture$"]
            result = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=120)
            self.assertEqual(result.returncode, 2, result.stderr.decode("utf-8", "replace"))
            self.assertIn(b"--tests did not match any test files", result.stderr)
            with open(cached, "rb") as f:
                self.assertEqual(f.read(), previous)

    def test_missing_logo_leaves_no_image(self):
        old = pd.LOGO
        try:
            pd.LOGO = os.path.join(ROOT, "assets", "no_such_logo.png")
            self.assertEqual(pd.logo_src(), "")
            self.assertIn('src=""', pd.render_page({"schema": pd.SCHEMA}, []))
        finally:
            pd.LOGO = old

    def test_script_element_cannot_be_closed_by_data(self):
        snap = {"schema": pd.SCHEMA, "x": "</script><b>"}
        self.assertNotIn("</script><b>", pd.render_page(snap, []))

    def test_out_as_existing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_out = os.path.join(tmp, "not_a_dir")
            with open(file_out, "wb") as f:
                f.write(b"x")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = pd.main(["--out", file_out])
            self.assertEqual(rc, 2)
            self.assertIn("is not a directory", err.getvalue())


if __name__ == "__main__":
    unittest.main()
