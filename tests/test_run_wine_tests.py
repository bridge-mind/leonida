"""Tests for scripts/run_wine_tests.py: path mapping, per-test statuses and the job's exit code.

    python -I tests/test_run_wine_tests.py

No Wine and no game data: `wine` is the Python interpreter and every "build artifact" is a tiny script that exits
with a chosen code, so the runner's own logic is what is under test.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_wine_tests as rwt  # noqa: E402

WINDOWS_ROOT = "D:/win/root"


def ctest_entry(name, command, properties=None):
    entry = {"name": name}
    if command is not None:
        entry["command"] = command
    if properties:
        entry["properties"] = [{"name": k, "value": v} for k, v in properties.items()]
    return entry


class PathMappingTest(unittest.TestCase):
    def test_paths_under_the_windows_checkout_are_mapped(self):
        self.assertEqual(rwt.map_path("D:/win/root", WINDOWS_ROOT, "/repo"), "/repo")
        self.assertEqual(rwt.map_path("D:\\win\\root\\runtime\\build", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runtime", "build"))
        self.assertEqual(rwt.map_path("d:/WIN/root/a/b.exe", WINDOWS_ROOT, "/repo"), os.path.join("/repo", "a", "b.exe"))
        self.assertEqual(rwt.map_path("D:/win/root/", WINDOWS_ROOT, "/repo"), "/repo")

    def test_paths_outside_the_checkout_are_not_mapped(self):
        for path in ("D:/win/root2/a.exe", "C:/Windows/system32/cmd.exe", "/usr/bin/cmake", ""):
            with self.subTest(path=path):
                self.assertIsNone(rwt.map_path(path, WINDOWS_ROOT, "/repo"))

    def test_parent_segments_cannot_escape_windows_root(self):
        self.assertIsNone(rwt.map_path("D:/win/root/../outside.exe", WINDOWS_ROOT, "/repo"))
        self.assertEqual(rwt.map_path("D:/win/root/inside/../runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runner.exe"))

    def test_dot_and_repeated_separators_normalize_before_mapping(self):
        self.assertEqual(rwt.map_path("D:/win/root/./runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runner.exe"))
        self.assertEqual(rwt.map_path("D:/win/root//sub//runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "sub", "runner.exe"))
        self.assertIsNone(rwt.map_path("D:/win/root/../../outside.exe", WINDOWS_ROOT, "/repo"))

    def test_wine_path_uses_the_z_drive(self):
        self.assertEqual(rwt.wine_path("/tmp/x y"), "Z:" + os.path.abspath("/tmp/x y").replace("/", "\\"))

    def test_properties_are_flattened(self):
        self.assertEqual(rwt.properties(ctest_entry("t", ["x"], {"SKIP_RETURN_CODE": "77", "WILL_FAIL": "1"})),
                         {"SKIP_RETURN_CODE": "77", "WILL_FAIL": "1"})
        self.assertEqual(rwt.properties({"name": "t"}), {})

    def test_ctest_booleans_are_not_python_truthiness(self):
        for value in ("1", "ON", "true", "yes", " 1 "):
            with self.subTest(value=value):
                self.assertTrue(rwt.truthy(value))
        for value in ("0", "", None, "false", "no"):
            with self.subTest(value=value):
                self.assertFalse(rwt.truthy(value))


class RunTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.args = argparse.Namespace(windows_root=WINDOWS_ROOT, root=str(self.root), timeout=10.0)
        self.wine = [sys.executable]

    def tearDown(self):
        self._tmp.cleanup()

    def program(self, name, body):
        path = self.artifact / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def exiting(self, code, name=None):
        return self.program(name or f"exit{code}.py", f"import sys\nsys.exit({code})\n")

    def command_for(self, exe, *args):
        """The same program as ctest would list it: a path in the Windows checkout."""
        parts = Path(exe).relative_to(self.root).parts
        return [WINDOWS_ROOT + "/" + "/".join(parts), *args]

    def run_entry(self, entry):
        return rwt.run_test(entry, self.args, self.wine)

    def test_a_passing_test(self):
        status, detail = self.run_entry(ctest_entry("ok", self.command_for(self.exiting(0))))
        self.assertEqual(status, "pass")
        self.assertIn("exit 0", detail)

    def test_a_failing_test_reports_the_exit_code(self):
        status, detail = self.run_entry(ctest_entry("bad", self.command_for(self.exiting(3))))
        self.assertEqual(status, "fail")
        self.assertIn("exit 3", detail)

    def test_a_disabled_test_is_skipped_without_launching(self) -> None:
        built = self.command_for(self.exiting(0))
        missing = self.command_for(self.artifact / "missing.exe")
        for command in (built, missing, None):
            with self.subTest(command=command), \
                    mock.patch.object(rwt, "run_test_process", wraps=rwt.run_test_process) as launch:
                status, detail = self.run_entry(ctest_entry("disabled", command, {"DISABLED": True}))
                launch.assert_not_called()
                self.assertEqual(status, "skip", detail)
                self.assertIn("disabled by CTest", detail)

    def test_a_false_disabled_property_still_runs_the_test(self) -> None:
        command = self.command_for(self.exiting(0))
        status, detail = self.run_entry(ctest_entry("enabled", command, {"DISABLED": "0"}))
        self.assertEqual(status, "pass", detail)

    def test_a_test_without_a_command_fails(self):
        status, detail = self.run_entry(ctest_entry("unbuilt", None))
        self.assertEqual(status, "fail")
        self.assertIn("no command", detail)

    def test_a_test_that_runs_a_runner_program_is_skipped(self):
        status, detail = self.run_entry(ctest_entry("python-test", ["/usr/bin/cmake", "-P", "x.cmake"]))
        self.assertEqual(status, "skip")
        self.assertIn("not a program from the build", detail)

    def test_a_missing_executable_fails(self):
        status, detail = self.run_entry(ctest_entry("gone", [WINDOWS_ROOT + "/artifact/gone.py"]))
        self.assertEqual(status, "fail")
        self.assertIn("not in the artifact", detail)

    def test_will_fail_inverts_the_exit_code(self):
        exe = self.exiting(2)
        status, _ = self.run_entry(ctest_entry("expected", self.command_for(exe), {"WILL_FAIL": "1"}))
        self.assertEqual(status, "pass")

    def test_will_fail_zero_does_not_invert_the_exit_code(self):
        # A JSON property value is a string; "0" must stay false.
        status, detail = self.run_entry(ctest_entry("normal", self.command_for(self.exiting(0)), {"WILL_FAIL": "0"}))
        self.assertEqual(status, "pass", detail)

    def test_the_skip_return_code_is_a_skip(self):
        status, detail = self.run_entry(
            ctest_entry("skipped", self.command_for(self.exiting(77)), {"SKIP_RETURN_CODE": "77"}))
        self.assertEqual(status, "skip")
        self.assertIn("exit 77", detail)

    def test_invalid_skip_return_codes_fail_before_launch(self):
        command = self.command_for(self.exiting(0))
        for value in ("invalid", True, 1.5, -1, 256):
            with self.subTest(value=value), mock.patch.object(rwt.subprocess, "run") as launch:
                status, detail = self.run_entry(ctest_entry("invalid", command, {"SKIP_RETURN_CODE": value}))
                self.assertEqual(status, "fail")
                self.assertIn("invalid CTest SKIP_RETURN_CODE", detail)
                launch.assert_not_called()

    def test_ctest_environment_is_applied_only_to_its_child(self) -> None:
        exe = self.program("show_environment.py",
                           "import os\n"
                           "for key in ('LEONIDA_HOST_ADAPTER', 'LEONIDA_WINE_TEST_INHERITED', "
                           "'LEONIDA_WINE_TEST_VALUE', 'LEONIDA_WINE_TEST_EMPTY'):\n"
                           "    print(repr(os.environ.get(key)))\n")
        command = self.command_for(exe)
        env = {"LEONIDA_HOST_ADAPTER": "inherited", "LEONIDA_WINE_TEST_INHERITED": "keep"}
        props = {"ENVIRONMENT": ["LEONIDA_HOST_ADAPTER=warp", "LEONIDA_WINE_TEST_VALUE=path=a;b",
                                 "LEONIDA_WINE_TEST_EMPTY="]}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("LEONIDA_WINE_TEST_VALUE", None)
            os.environ.pop("LEONIDA_WINE_TEST_EMPTY", None)
            status, detail = self.run_entry(ctest_entry("with_environment", command, props))
            self.assertEqual(status, "pass", detail)
            self.assertEqual(detail.splitlines()[1:], ["'warp'", "'keep'", "'path=a;b'", "''"])
            status, detail = self.run_entry(ctest_entry("without_environment", command))
            self.assertEqual(status, "pass", detail)
            self.assertEqual(detail.splitlines()[1:], ["'inherited'", "'keep'", "None", "None"])

    def test_malformed_ctest_environment_fails_before_launch(self) -> None:
        command = self.command_for(self.exiting(0))
        props = {"ENVIRONMENT": ["missing_equals"]}
        with mock.patch.object(rwt, "run_test_process", wraps=rwt.run_test_process) as launch:
            status, detail = self.run_entry(ctest_entry("bad_environment", command, props))
        launch.assert_not_called()
        self.assertEqual(status, "fail", detail)
        self.assertIn("invalid CTest ENVIRONMENT", detail)

    def test_ctest_output_expressions_determine_the_result(self):
        usage = {"PASS_REGULAR_EXPRESSION": ["alternative", "usage: gta6rt"]}
        forbidden = {"FAIL_REGULAR_EXPRESSION": ["running ", r"module\(s\) mapped"]}
        for code, output, props, expected in (
            (2, "usage: gta6rt", usage | forbidden, "pass"),
            (0, "unexpected output", usage, "fail"),
            (2, "usage: gta6rt\n1 module(s) mapped", usage | forbidden, "fail"),
            (0, "running tiny_game.exe", forbidden, "fail"),
            (2, "unexpected output", forbidden, "fail"),
            (77, "unexpected output", usage | {"SKIP_RETURN_CODE": 77}, "skip"),
            (0, "running tiny_game.exe", forbidden | {"WILL_FAIL": True}, "pass"),
        ):
            with self.subTest(code=code, output=output, props=props):
                exe = self.program("regex.py", f"import sys\nprint({output!r}, file=sys.stderr)\nsys.exit({code})\n")
                status, detail = self.run_entry(ctest_entry("regex", self.command_for(exe), props))
                self.assertEqual(status, expected, detail)

    def test_a_scalar_output_expression_is_matched_as_one_pattern(self):
        required = {"PASS_REGULAR_EXPRESSION": "usage: gta6rt"}
        forbidden = {"FAIL_REGULAR_EXPRESSION": r"module\(s\) mapped"}
        for code, output, props, expected in (
            (3, "unexpected output", required, "fail"),
            (2, "usage: gta6rt", required, "pass"),
            (0, "unexpected output", forbidden, "pass"),
            (0, "1 module(s) mapped", forbidden, "fail"),
        ):
            with self.subTest(output=output, props=props):
                exe = self.program("regex.py", f"import sys\nprint({output!r})\nsys.exit({code})\n")
                status, detail = self.run_entry(ctest_entry("regex", self.command_for(exe), props))
                self.assertEqual(status, expected, detail)

    def test_skip_output_overrides_a_normal_exit_status(self):
        skip_pattern = "SKIP: missing adapter"
        exe = self.program("skip.py", f"print({skip_pattern!r})\nraise SystemExit(3)\n")
        for expression in (skip_pattern, ["different reason", skip_pattern]):
            with self.subTest(expression=expression):
                entry = ctest_entry("skip", self.command_for(exe), {"SKIP_REGULAR_EXPRESSION": expression})
                status, detail = rwt.run_test(entry, self.args, [sys.executable, "-I"])
                self.assertEqual(status, "skip", detail)
                self.assertIn("SKIP_REGULAR_EXPRESSION", detail)

        entry = ctest_entry("no_match", self.command_for(exe), {"SKIP_REGULAR_EXPRESSION": ["different reason"]})
        status, detail = rwt.run_test(entry, self.args, [sys.executable, "-I"])
        self.assertEqual(status, "fail", detail)

    def test_a_signal_termination_fails_even_with_success_overrides(self):
        # The mocked negative returncode keeps the test portable across Windows and POSIX.
        exe = self.exiting(0)
        fake = mock.Mock(returncode=-11, stdout=b"usage: gta6rt")
        for props in ({"WILL_FAIL": True}, {"PASS_REGULAR_EXPRESSION": ["usage: gta6rt"]},
                      {"SKIP_REGULAR_EXPRESSION": ["usage: gta6rt"]}):
            with self.subTest(props=props), mock.patch.object(rwt, "run_test_process", return_value=fake):
                entry = ctest_entry("segv", self.command_for(exe), props | {"SKIP_RETURN_CODE": 77})
                status, detail = self.run_entry(entry)
                self.assertEqual(status, "fail")
                self.assertIn("signal 11", detail)

    def test_a_timeout_is_reported_without_hanging(self):
        slow = self.program("slow.py", "import time\ntime.sleep(5)\n")
        self.args.timeout = 0.2
        status, detail = self.run_entry(ctest_entry("slow", self.command_for(slow)))
        self.assertEqual(status, "fail")
        self.assertIn("timed out after 0.2 s", detail)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_timeout_stops_a_helper_even_after_the_launcher_exits(self):
        import fcntl

        guard_path = self.artifact / "helper.lock"
        stop_path = self.artifact / "stop"
        helper = self.program("helper.py", """
import fcntl
from pathlib import Path
import time

with open("helper.lock", "a") as guard:
    fcntl.flock(guard, fcntl.LOCK_EX)
    print("helper ready", flush=True)
    deadline = time.monotonic() + 10
    while not Path("stop").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
""")
        launcher = self.program("launcher.py", f"""
import subprocess
import sys

subprocess.Popen([sys.executable, {helper!r}])
""")
        self.args.timeout = 1
        entry = ctest_entry("tree", self.command_for(launcher))
        try:
            status, detail = self.run_entry(entry)
            self.assertEqual(status, "fail", detail)
            self.assertIn("timed out", detail)
            self.assertIn("helper ready", detail)
            with guard_path.open("a") as guard:
                deadline = time.monotonic() + 1
                while True:
                    try:
                        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            self.fail("helper still holds its lock after the test timed out")
                        time.sleep(0.01)
        finally:
            stop_path.touch()

    def test_a_short_ctest_timeout_still_fails_a_will_fail_test(self):
        slow = self.program("slow_failure.py", "import sys, time\ntime.sleep(1)\nsys.exit(1)\n")
        entry = ctest_entry("slow_failure", self.command_for(slow), {"TIMEOUT": 0.2, "WILL_FAIL": True})
        status, detail = self.run_entry(entry)
        self.assertEqual(status, "fail", detail)
        self.assertIn("timed out after 0.2 s", detail)

    def test_ctest_timeout_cannot_remove_or_extend_the_runner_limit(self):
        exe = self.exiting(0)
        for timeout in (0, "20"):
            with self.subTest(timeout=timeout):
                entry = ctest_entry("bounded", self.command_for(exe), {"TIMEOUT": timeout})
                result = rwt.subprocess.CompletedProcess([], 0, stdout=b"")
                with mock.patch.object(rwt, "run_test_process", return_value=result) as run:
                    status, detail = self.run_entry(entry)
                self.assertEqual(status, "pass", detail)
                self.assertEqual(run.call_args.kwargs["timeout"], self.args.timeout)

    def test_an_unstartable_wine_fails_the_test_instead_of_aborting_the_run(self):
        status, detail = rwt.run_test(ctest_entry("ok", self.command_for(self.exiting(0))), self.args,
                                      ["definitely-not-a-real-wine-xyz"])
        self.assertEqual(status, "fail")
        self.assertIn("could not start", detail)

    def test_arguments_and_working_directory_are_mapped(self):
        report = self.root / "report.json"
        script = self.program("report.py", (
            "import json, os, sys\n"
            "json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd()}, open(os.environ['REPORT_PATH'], 'w'))\n"))
        (self.artifact / "sub").mkdir()
        entry = ctest_entry("mapped", self.command_for(script, WINDOWS_ROOT + "/artifact/data.bin", "--flag"),
                           {"WORKING_DIRECTORY": WINDOWS_ROOT + "/artifact/sub"})
        with mock.patch.dict(os.environ, {"REPORT_PATH": str(report)}):
            status, detail = self.run_entry(entry)
        self.assertEqual(status, "pass", detail)
        seen = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(seen["argv"]), 2)
        self.assertTrue(seen["argv"][0].startswith("Z:"), seen["argv"][0])
        self.assertTrue(seen["argv"][0].endswith("data.bin"), seen["argv"][0])
        self.assertEqual(seen["argv"][1], "--flag")
        self.assertEqual(os.path.normcase(os.path.realpath(seen["cwd"])),
                         os.path.normcase(os.path.realpath(self.artifact / "sub")))

    def test_invalid_mapped_working_directory_fails_without_starting_wine(self):
        exe = self.exiting(0)
        (self.artifact / "file").touch()
        for name in ("missing", "file"):
            cwd = WINDOWS_ROOT + f"/artifact/{name}"
            with self.subTest(cwd=cwd), mock.patch.object(rwt.subprocess, "run", wraps=rwt.subprocess.run) as run:
                entry = ctest_entry("bad_cwd", self.command_for(exe), {"WORKING_DIRECTORY": cwd})
                status, detail = self.run_entry(entry)
                self.assertEqual(status, "fail")
                self.assertIn("working directory", detail)
                self.assertIn(cwd, detail)
                run.assert_not_called()


@unittest.skipIf(os.name == "nt", "the runner and its dosdevices links are for Linux and macOS")
class DriveMapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "checkout"
        (self.root / "runtime" / "build" / "generated").mkdir(parents=True)
        (self.root / "runtime" / "build" / "generated" / "cs.spv").write_text("spirv", encoding="utf-8")
        self.prefix = base / "prefix"
        (self.prefix / "dosdevices").mkdir(parents=True)
        self.link = self.prefix / "dosdevices" / "d:"

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_drive_leads_to_the_checkout_and_is_removed_afterwards(self):
        with rwt.DriveMap("D:/a/Leonida/Leonida", str(self.root), str(self.prefix)) as drive:
            seen = self.link / "a" / "Leonida" / "Leonida" / "runtime" / "build" / "generated" / "cs.spv"
            self.assertEqual(seen.read_text(encoding="utf-8"), "spirv")
            self.assertIn("drive d: maps", drive.note)
            scratch = drive.scratch
        self.assertFalse(os.path.lexists(self.link))
        self.assertFalse(os.path.exists(scratch))
        self.assertTrue((self.root / "runtime" / "build" / "generated" / "cs.spv").is_file())   # links only

    def test_backslashes_and_a_lower_case_drive_are_accepted(self):
        with rwt.DriveMap("d:\\work\\leonida\\", str(self.root), str(self.prefix)):
            self.assertTrue((self.link / "work" / "leonida" / "runtime" / "build").is_dir())

    def test_a_drive_the_prefix_already_has_is_left_alone(self):
        other = self.prefix / "elsewhere"
        other.mkdir()
        os.symlink(other, self.link)
        with rwt.DriveMap("D:/a/Leonida/Leonida", str(self.root), str(self.prefix)) as drive:
            self.assertIn("already set", drive.note)
        self.assertEqual(os.readlink(self.link), str(other))

    def test_drive_created_during_setup_is_left_alone(self):
        other = self.prefix / "concurrent"
        other.mkdir()
        scratch = self.prefix / "scratch"
        scratch.mkdir()
        symlink = os.symlink

        def add_mapping(target, link):
            if Path(link) == self.link:
                symlink(other, self.link)
            symlink(target, link)

        with mock.patch.object(rwt.tempfile, "mkdtemp", return_value=str(scratch)), \
                mock.patch.object(rwt.os, "symlink", side_effect=add_mapping):
            with rwt.DriveMap(WINDOWS_ROOT, str(self.root), str(self.prefix)) as drive:
                self.assertIn("already set", drive.note)

        self.assertEqual(os.readlink(self.link), str(other))
        self.assertFalse(scratch.exists())

    def test_failed_drive_setup_removes_scratch_directory(self):
        scratch = self.prefix / "scratch"
        scratch.mkdir()
        with mock.patch.object(rwt.tempfile, "mkdtemp", return_value=str(scratch)), \
                mock.patch.object(rwt.os, "symlink", side_effect=PermissionError("link denied")):
            with self.assertRaisesRegex(PermissionError, "link denied"):
                with rwt.DriveMap(WINDOWS_ROOT, str(self.root), str(self.prefix)):
                    self.fail("drive setup unexpectedly succeeded")

        self.assertFalse(scratch.exists())
        self.assertFalse(os.path.lexists(self.link))

    def test_no_prefix_or_no_drive_letter_maps_nothing(self):
        with rwt.DriveMap("D:/a/b", str(self.root), str(self.prefix / "absent")) as drive:
            self.assertIn("no Wine prefix", drive.note)
        with rwt.DriveMap("/home/runner/work", str(self.root), str(self.prefix)) as drive:
            self.assertIn("no drive letter", drive.note)
        self.assertFalse(os.path.lexists(self.link))

    def test_main_maps_the_drive_while_the_tests_run(self):
        # The "Wine" here is the Python interpreter, so the program reads the drive through its dosdevices link, which
        # is where Wine resolves D:/win/root/... too.
        probe = self.root / "probe.py"
        target = self.link / "win" / "root" / "runtime" / "build" / "generated" / "cs.spv"
        probe.write_text(f"import sys\nsys.exit(0 if open({str(target)!r}).read() == 'spirv' else 1)\n", encoding="utf-8")
        listing = self.root / "ctest.json"
        listing.write_text(json.dumps({"tests": [ctest_entry("compiled_in_path", [WINDOWS_ROOT + "/probe.py"])]}),
                           encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"WINEPREFIX": str(self.prefix)}), mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(out):
            code = rwt.main()
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("[PASS] compiled_in_path", out.getvalue())
        self.assertFalse(os.path.lexists(self.link))


class MainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        for name, code in (("ok.py", 0), ("bad.py", 1)):
            (self.artifact / name).write_text(f"import sys\nsys.exit({code})\n", encoding="utf-8")
        # main() maps the checkout's drive into $WINEPREFIX: keep it away from a real prefix (~/.wine).
        env = mock.patch.dict(os.environ, {"WINEPREFIX": str(self.root / "no-prefix")})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def list_tests(self, entries):
        path = self.root / "ctest.json"
        path.write_text(json.dumps({"tests": entries}), encoding="utf-8")
        return str(path)

    def command(self, name):
        return [WINDOWS_ROOT + "/artifact/" + name]

    def run_main(self, entries, *extra, stderr=False, root=None):
        listing = self.list_tests(entries)
        argv = ["run_wine_tests.py", "--tests", listing, "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root if root is None else root), "--wine", sys.executable.replace(os.sep, "/"), *extra]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        return code, out.getvalue(), err.getvalue()

    def test_a_required_test_that_passes_exits_zero(self):
        code, out, _ = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--require", "ok")
        self.assertEqual(code, 0)
        self.assertIn("1 passed, 0 failed, 0 skipped under Wine (1 tests)", out)

    def test_relative_root_launches_the_artifact_from_its_working_directory(self):
        (self.artifact / "check_cwd.py").write_text(
            "from pathlib import Path\nassert Path.cwd() == Path(__file__).resolve().parent\n",
            encoding="utf-8", newline="\n")
        entry = ctest_entry("ok", self.command("check_cwd.py"))
        with contextlib.chdir(self.root):
            code, out, _ = self.run_main([entry], "--require", "ok", root=".")
        self.assertEqual(code, 0, out)
        self.assertIn("[PASS] ok", out)

    def test_a_required_test_that_fails_exits_one(self):
        code, out, _ = self.run_main([ctest_entry("bad", self.command("bad.py"))], "--require", "bad")
        self.assertEqual(code, 1)
        self.assertIn("required tests that did not pass: bad", out)

    def test_a_required_test_that_is_not_listed_exits_one(self):
        code, out, _ = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--require", "absent")
        self.assertEqual(code, 1)
        self.assertIn("required test not listed", out)

    def test_unrequired_failures_are_reported_but_do_not_fail_the_job(self):
        code, out, _ = self.run_main([ctest_entry("bad", self.command("bad.py"))])
        self.assertEqual(code, 0)
        self.assertIn("[FAIL] bad", out)

    def test_malformed_skip_return_code_does_not_stop_later_tests(self):
        entries = [
            ctest_entry("bad_property", self.command("ok.py"), {"SKIP_RETURN_CODE": "invalid"}),
            ctest_entry("ok", self.command("ok.py")),
        ]
        code, out, err = self.run_main(entries, "--require", "ok")
        self.assertEqual(code, 0, err)
        self.assertIn("[FAIL] bad_property", out)
        self.assertIn("invalid CTest SKIP_RETURN_CODE", out)
        self.assertIn("[PASS] ok", out)

    def test_the_summary_table_is_appended(self):
        summary = self.root / "summary.md"
        code, _, _ = self.run_main([ctest_entry("ok", self.command("ok.py")), ctest_entry("bad", self.command("bad.py"))],
                                   "--summary", str(summary), "--label", "Wine 11", "--require", "ok")
        self.assertEqual(code, 0)
        text = summary.read_text(encoding="utf-8")
        self.assertIn("### Windows tests under Wine 11", text)
        self.assertIn("| `ok` | pass | yes |", text)
        self.assertIn("| `bad` | fail |  |", text)

    def test_a_summary_in_a_missing_folder_creates_it(self):
        summary = self.root / "missing" / "sub" / "summary.md"
        code, _, err = self.run_main([ctest_entry("ok", self.command("ok.py"))],
                                     "--summary", str(summary), "--label", "Wine 12")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("### Windows tests under Wine 12", summary.read_text(encoding="utf-8"))

    def test_an_unwritable_summary_is_one_error_line(self):
        blocker = self.root / "blocker"
        blocker.write_text("not a folder", encoding="utf-8")
        summary = blocker / "summary.md"       # its parent is a file, not a folder
        code, _, err = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--summary", str(summary))
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)
        self.assertIn("ERROR: cannot write the summary", err)
        self.assertEqual(len(err.strip().splitlines()), 1, err)

    def test_a_file_that_is_not_json_is_one_error_line(self):
        listing = self.root / "notes.txt"
        listing.write_text("this is not ctest output\n", encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot read", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_json_that_is_not_ctest_output_is_one_error_line(self):
        code, _, err = self.run_main_json([{"name": "t"}])
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_missing_tests_is_not_a_zero_test_run(self):
        code, _, err = self.run_main_json({})
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_scalar_tests_rows_are_rejected(self):
        code, _, err = self.run_main_json({"tests": ["binary", 3, None]})
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_malformed_command_and_property_rows_are_rejected_before_launch(self):
        for fields in (
            {"command": "D:/win/root/artifact/ok.py"},
            {"command": [42]},
            {"command": [self.command("ok.py")[0], "embedded\0null"]},
            {"properties": {"TIMEOUT": 10}},
            {"properties": [{"name": "TIMEOUT"}]},
            {"properties": [{"name": ["TIMEOUT"], "value": 10}]},
            {"properties": [{"name": "WORKING_DIRECTORY", "value": 42}]},
            {"properties": [{"name": "WORKING_DIRECTORY", "value": "D:/win/\0root"}]},
            {"properties": [{"name": "PASS_REGULAR_EXPRESSION", "value": [42]}]},
            {"properties": [{"name": "FAIL_REGULAR_EXPRESSION", "value": {"bad": True}}]},
            {"properties": [{"name": "SKIP_REGULAR_EXPRESSION", "value": None}]},
            {"name": []},
        ):
            with self.subTest(fields=fields):
                good = ctest_entry("ok", self.command("ok.py"))
                with mock.patch.object(rwt, "DriveMap") as drive, \
                        mock.patch.object(rwt, "run_test", return_value=("pass", "")) as launch:
                    code, _, err = self.run_main([good, good | fields])
                self.assertEqual(code, 2)
                self.assertIn("not the output of ctest", err)
                drive.assert_not_called()
                launch.assert_not_called()

    def test_an_unbuilt_test_without_a_command_is_valid_inventory(self):
        code, out, err = self.run_main([{"name": "unbuilt"}], "--require", "unbuilt")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] unbuilt", out)
        self.assertIn("no command", out)
        self.assertEqual(err, "")

    def test_ctest_property_strings_and_expression_lists_are_valid_inventory(self):
        properties = {
            "WORKING_DIRECTORY": WINDOWS_ROOT + "/artifact",
            "PASS_REGULAR_EXPRESSION": ["good", "also good"],
            "FAIL_REGULAR_EXPRESSION": "unexpected",
            "SKIP_REGULAR_EXPRESSION": [],
            "MEASUREMENTS": {"custom": "value"},
        }
        row = ctest_entry("ok", self.command("ok.py"), properties)
        with mock.patch.object(rwt, "run_test", return_value=("pass", "")) as launch:
            code, _, err = self.run_main([row], "--require", "ok")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        launch.assert_called_once()

    def test_an_explicit_empty_tests_list_is_accepted(self):
        code, out, _ = self.run_main_json({"tests": []})
        self.assertEqual(code, 0)
        self.assertIn("0 passed, 0 failed, 0 skipped under Wine (0 tests)", out)

    def run_main_json(self, document):
        listing = self.root / "wrong.json"
        listing.write_text(json.dumps(document), encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        return code, out.getvalue(), err.getvalue()

    def test_a_bad_timeout_is_refused_by_argparse(self):
        for text in ("0", "-1", "abc", "nan", "inf", "-inf"):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    rwt.positive_seconds(text)
        self.assertEqual(rwt.positive_seconds("0.5"), 0.5)

    def test_a_nonfinite_timeout_is_a_usage_error(self):
        for text in ("nan", "inf", "-inf"):
            with self.subTest(text=text):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_main([ctest_entry("ok", self.command("ok.py"))], "--timeout", text)
                self.assertEqual(ctx.exception.code, 2)

    def test_a_blank_wine_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main([ctest_entry("ok", self.command("ok.py"))], "--wine", "")
        self.assertEqual(ctx.exception.code, 2)

    def test_an_unclosed_quote_in_wine_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main([ctest_entry("ok", self.command("ok.py"))], "--wine", "wine 'unclosed")
        self.assertEqual(ctx.exception.code, 2)

    def test_wine_quoting_is_still_split(self):
        self.assertEqual(rwt.wine_command("prog 'two words'"), ["prog", "two words"])
        with self.assertRaises(argparse.ArgumentTypeError):
            rwt.wine_command("   ")


if __name__ == "__main__":
    unittest.main()
