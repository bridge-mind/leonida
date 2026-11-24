"""Tests for tools/classify_carve.py, built from synthetic carve indexes (no game data).

    python -I tests/test_classify_carve.py
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

ZERO_HEAD = "00" * 16
RBF0_HEAD = "5242463000000000"
RESPAGE_HEAD = "42400000deadbeef"


def load_module():
    path = os.path.join(ROOT, "tools", "classify_carve.py")
    spec = importlib.util.spec_from_file_location("classify_carve_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


carve = load_module()


def counts(entry):
    return {k: entry[k] for k in ("files", "bytes")}


class ClassifyTests(unittest.TestCase):
    def test_bink2_rows_do_not_need_a_head(self):
        # rpf9_carve writes bink2 rows with dimensions and no head field.
        self.assertEqual(carve.classify({"type": "bink2", "size": 12}), "bink2")

    def test_bink2_wins_over_a_signature(self):
        self.assertEqual(carve.classify({"type": "bink2", "head": RBF0_HEAD, "size": 1}), "bink2")

    def test_rbf0_metadata(self):
        self.assertEqual(carve.classify({"head": RBF0_HEAD, "size": 1}), "rbf0")

    def test_resource_page(self):
        self.assertEqual(carve.classify({"head": RESPAGE_HEAD, "size": 1}), "respage")

    def test_gpu_zero_header_needs_sixteen_zero_bytes(self):
        self.assertEqual(carve.classify({"head": ZERO_HEAD, "size": 1}), "gpu-zerohdr")
        # Fifteen zero bytes are not enough, and the fallback names four bytes.
        self.assertEqual(carve.classify({"head": "00" * 15, "size": 1}), "other:00000000")

    def test_other_is_keyed_by_the_first_four_bytes(self):
        self.assertEqual(carve.classify({"head": "deadbeefcafebabe", "size": 1}), "other:deadbeef")

    def test_short_head_is_used_as_is(self):
        self.assertEqual(carve.classify({"head": "abcd", "size": 1}), "other:abcd")

    def test_signature_must_be_at_the_start(self):
        self.assertEqual(carve.classify({"head": "ff" + RBF0_HEAD, "size": 1}), "other:ff524246")


class SizeStatsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        # Archive a: rbf0 sizes 40, 10, 30 (odd), gpu-zerohdr 5 (single).
        # Archive b: rbf0 sizes 20 (single), gpu-zerohdr 8, 2, 6, 100 (even).
        self.archive("a", [
            {"head": RBF0_HEAD, "size": 40}, {"head": RBF0_HEAD, "size": 10},
            {"head": RBF0_HEAD, "size": 30}, {"head": ZERO_HEAD, "size": 5},
        ])
        self.archive("b", [
            {"head": RBF0_HEAD, "size": 20}, {"head": ZERO_HEAD, "size": 8},
            {"head": ZERO_HEAD, "size": 2}, {"head": ZERO_HEAD, "size": 6},
            {"head": ZERO_HEAD, "size": 100},
        ])
        with mock.patch("sys.argv", ["classify_carve.py", self.root]), contextlib.redirect_stdout(io.StringIO()):
            carve.main()
        with open(os.path.join(self.root, "types.json"), encoding="utf-8") as f:
            self.types = json.load(f)

    def archive(self, name, rows):
        path = os.path.join(self.root, name)
        os.makedirs(path)
        with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"files": rows}, f)

    def test_per_archive_stats(self):
        per = self.types["per_archive"]
        self.assertEqual(per["a"]["rbf0"], {"files": 3, "bytes": 80, "min": 10, "max": 40, "median": 30})
        self.assertEqual(per["a"]["gpu-zerohdr"], {"files": 1, "bytes": 5, "min": 5, "max": 5, "median": 5})
        self.assertEqual(per["b"]["rbf0"], {"files": 1, "bytes": 20, "min": 20, "max": 20, "median": 20})
        # Even count: sizes 2, 6, 8, 100 -> lower middle value 6, an integer.
        self.assertEqual(per["b"]["gpu-zerohdr"], {"files": 4, "bytes": 116, "min": 2, "max": 100, "median": 6})

    def test_overall_stats_span_archives(self):
        overall = self.types["overall"]
        # rbf0: 10, 20, 30, 40 -> even, lower middle 20.
        self.assertEqual(overall["rbf0"], {"files": 4, "bytes": 100, "min": 10, "max": 40, "median": 20})
        # gpu-zerohdr: 2, 5, 6, 8, 100 -> odd, middle 6.
        self.assertEqual(overall["gpu-zerohdr"], {"files": 5, "bytes": 121, "min": 2, "max": 100, "median": 6})

    def test_existing_keys_keep_their_values_and_order(self):
        for scope in (self.types["overall"], *self.types["per_archive"].values()):
            for stats in scope.values():
                self.assertEqual(list(stats), ["files", "bytes", "min", "max", "median"])
                self.assertTrue(all(type(v) is int for v in stats.values()))
        self.assertEqual(self.types["overall"]["rbf0"]["files"], 4)
        self.assertEqual(self.types["overall"]["rbf0"]["bytes"], 100)
        self.assertEqual(list(self.types["overall"]), ["gpu-zerohdr", "rbf0"])

    def test_size_beyond_64_bits_is_accepted(self):
        big = 2 ** 64
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "a"))
            with open(os.path.join(root, "a", "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": [{"head": ZERO_HEAD, "size": big}]}, f)
            with mock.patch("sys.argv", ["classify_carve.py", root]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(carve.main(), 0)
            with open(os.path.join(root, "types.json"), encoding="utf-8") as f:
                types = json.load(f)
        want = {"files": 1, "bytes": big, "min": big, "max": big, "median": big}
        self.assertEqual(types["overall"]["gpu-zerohdr"], want)
        self.assertEqual(types["per_archive"]["a"]["gpu-zerohdr"], want)

    def test_empty_report_has_no_entries_to_carry_stats(self):
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch("sys.argv", ["classify_carve.py", empty]), contextlib.redirect_stdout(io.StringIO()):
                carve.main()
            with open(os.path.join(empty, "types.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"overall": {}, "per_archive": {}})


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def archive(self, name, rows):
        path = os.path.join(self.root, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"files": rows}, f)
        return path

    def run_main(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["classify_carve.py", self.root]), contextlib.redirect_stdout(out):
            carve.main()
        return out.getvalue()

    def types(self):
        with open(os.path.join(self.root, "types.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_counts_and_bytes_are_aggregated_per_class(self):
        self.archive("a", [
            {"head": RBF0_HEAD, "size": 10},
            {"head": "5242463000000001", "size": 30},
            {"head": ZERO_HEAD, "size": 5},
            {"head": "deadbeefcafebabe", "size": 7},
            {"type": "bink2", "size": 3},
        ])
        self.run_main()
        types = self.types()
        a = types["per_archive"]["a"]
        self.assertEqual(counts(a["rbf0"]), {"files": 2, "bytes": 40})
        self.assertEqual(counts(a["gpu-zerohdr"]), {"files": 1, "bytes": 5})
        self.assertEqual(counts(a["other:deadbeef"]), {"files": 1, "bytes": 7})
        self.assertEqual(counts(a["bink2"]), {"files": 1, "bytes": 3})
        self.assertEqual(counts(types["overall"]["rbf0"]), {"files": 2, "bytes": 40})

    def test_overall_sums_across_archives(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 10}])
        self.archive("b", [{"head": RBF0_HEAD, "size": 25}])
        self.run_main()
        types = self.types()
        self.assertEqual(counts(types["overall"]["rbf0"]), {"files": 2, "bytes": 35})
        self.assertEqual(counts(types["per_archive"]["a"]["rbf0"]), {"files": 1, "bytes": 10})

    def test_archives_are_listed_in_name_order(self):
        self.archive("b", [{"head": RBF0_HEAD, "size": 1}])
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        self.run_main()
        self.assertEqual(list(self.types()["per_archive"]), ["a", "b"])

    def test_classes_are_ordered_by_bytes_descending(self):
        self.archive("a", [
            {"head": RBF0_HEAD, "size": 50},
            {"head": ZERO_HEAD, "size": 500},
        ])
        self.run_main()
        types = self.types()
        self.assertEqual(list(types["per_archive"]["a"]), ["gpu-zerohdr", "rbf0"])
        self.assertEqual(list(types["overall"]), ["gpu-zerohdr", "rbf0"])

    def test_directory_without_an_index_is_skipped(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        os.makedirs(os.path.join(self.root, "empty"), exist_ok=True)
        self.run_main()
        self.assertEqual(list(self.types()["per_archive"]), ["a"])

    def test_stdout_summary(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}, {"head": ZERO_HEAD, "size": 2}])
        out = self.run_main()
        self.assertIn("GiB", out)
        self.assertIn("rbf0", out)
        self.assertIn("2 classes in total", out)

    def test_empty_root(self):
        out = self.run_main()
        self.assertEqual(self.types(), {"overall": {}, "per_archive": {}})
        self.assertIn("0 classes in total", out)

    def test_types_json_uses_one_space_indent(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        self.run_main()
        with open(os.path.join(self.root, "types.json"), encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith('{\n "overall": {'))

    def test_invalid_index_names_its_path_and_preserves_previous_report(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        self.run_main()
        before = self.types()
        index = os.path.join(self.archive("b", []), "index.json")
        for data, message in (
            ("{", "invalid JSON"),
            ('{"files": {}}', '"files" list'),
            ('{"files": [{"head": "00"}]}', "row 1"),
            ('{"files": [{"size": 12, "head": 42}]}', "row 1"),
        ):
            with self.subTest(data=data):
                with open(index, "w", encoding="utf-8") as f:
                    f.write(data)
                proc = subprocess.run(
                    [sys.executable, "-I", os.path.join(ROOT, "tools", "classify_carve.py"), self.root],
                    capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 2)
                self.assertIn(index, proc.stderr)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertEqual(self.types(), before)

    def test_bad_arguments_and_roots_are_one_line(self):
        tool = os.path.join(ROOT, "tools", "classify_carve.py")
        a_file = os.path.join(self.root, "index.json")
        with open(a_file, "w", encoding="utf-8") as f:
            f.write("{}")
        for args, message in (([], "usage"), ([self.root, "extra"], "usage"),
                              ([os.path.join(self.root, "missing")], "does not exist"),
                              ([a_file], "is not a folder")):
            with self.subTest(args=[os.path.basename(a) for a in args]):
                proc = subprocess.run([sys.executable, "-I", tool, *args], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2)
                self.assertIn(message, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertEqual(proc.stdout, "")
        self.assertEqual(sorted(os.listdir(self.root)), ["index.json"])      # no types.json was written

    def test_types_json_directory_is_one_error_line(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        report = os.path.join(self.root, "types.json")
        os.mkdir(report)
        proc = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "classify_carve.py"), self.root],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("types.json", proc.stderr)
        self.assertIn("is a directory", proc.stderr)
        self.assertIn("remove or rename", proc.stderr)
        self.assertIn("before rerun", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(os.listdir(report), [])
        self.assertEqual([n for n in os.listdir(self.root) if n.startswith(".types-")], [])

    def test_help_prints_usage_and_writes_nothing(self):
        tool = os.path.join(ROOT, "tools", "classify_carve.py")
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                proc = subprocess.run([sys.executable, "-I", tool, flag], capture_output=True, text=True,
                                      cwd=self.root)
                self.assertEqual(proc.returncode, 0)
                self.assertIn("usage", proc.stdout)
                # The printed line is meant to be copied from the repository root, so it names the tool's folder.
                self.assertIn("python -I tools/classify_carve.py", proc.stdout)
                self.assertEqual(proc.stderr, "")
        self.assertEqual(os.listdir(self.root), [])                            # nothing read or written

    def test_failed_write_preserves_previous_report(self):
        self.archive("a", [{"head": RBF0_HEAD, "size": 1}])
        self.run_main()
        before = self.types()
        files = sorted(os.listdir(self.root))

        def fail_dump(_data, handle, **_kwargs):
            handle.write("{")
            raise OSError("synthetic write failure")

        with mock.patch.object(carve.json, "dump", side_effect=fail_dump):
            with self.assertRaisesRegex(OSError, "write failure"):
                self.run_main()
        self.assertEqual(self.types(), before)
        self.assertEqual(sorted(os.listdir(self.root)), files)


if __name__ == "__main__":
    unittest.main()
