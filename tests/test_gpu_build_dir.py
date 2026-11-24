"""run_gpu_tests.py and check.py find the test programs where the CMake generator put them: in the build folder
(Makefiles, Ninja) or in build/<Config> (Visual Studio generator, TEST-38), and say which folder they picked. Fake
files only, so this runs on any OS."""
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "scripts"))
import check
import run_gpu_tests

PROGRAMS = tuple(t + check.EXE for t in check.TEST_PROGRAMS)


def touch(folder: Path, names=PROGRAMS, mtime=None):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        f = folder / n
        f.write_bytes(b"")
        if mtime is not None:
            os.utime(f, (mtime, mtime))


class ResolveBuildDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = Path(tmp.name) / "build"
        self.build.mkdir()

    def resolve(self):
        return check.resolve_build_dir(str(self.build), PROGRAMS)

    def test_programs_directly_in_the_build_folder(self):
        touch(self.build)
        self.assertEqual(self.resolve(), str(self.build))

    def test_visual_studio_generator_subfolder(self):
        touch(self.build / "RelWithDebInfo")
        self.assertEqual(self.resolve(), str(self.build / "RelWithDebInfo"))

    def test_release_subfolder(self):
        touch(self.build / "Release")
        self.assertEqual(self.resolve(), str(self.build / "Release"))

    def test_build_folder_wins_over_a_subfolder(self):
        touch(self.build)
        touch(self.build / "Release")
        self.assertEqual(self.resolve(), str(self.build))

    def test_most_recently_built_configuration_wins(self):
        touch(self.build / "Debug", mtime=1_000_000)
        touch(self.build / "Release", mtime=3_000_000)
        touch(self.build / "RelWithDebInfo", mtime=2_000_000)
        self.assertEqual(self.resolve(), str(self.build / "Release"))

    def test_a_configuration_missing_a_program_is_skipped(self):
        touch(self.build / "Release", names=PROGRAMS[:2], mtime=9_000_000)    # newer but incomplete
        touch(self.build / "Debug", mtime=1_000_000)
        self.assertEqual(self.resolve(), str(self.build / "Debug"))

    def test_nothing_found_returns_the_folder_given(self):
        # so the "Test programs not found in <folder>" message names what the user passed
        self.assertEqual(self.resolve(), str(self.build))
        touch(self.build / "Release", names=PROGRAMS[:1])
        self.assertEqual(self.resolve(), str(self.build))

    def test_unknown_configuration_folder_is_not_searched(self):
        touch(self.build / "Custom")
        self.assertEqual(self.resolve(), str(self.build))


class PickBuildDir(unittest.TestCase):
    """The line that tells which folder was used (a #166 report names the build it ran)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = Path(tmp.name) / "build"
        self.build.mkdir()

    def pick(self):
        out = io.StringIO()
        with redirect_stdout(out):
            chosen = check.pick_build_dir(str(self.build), PROGRAMS)
        return chosen, out.getvalue()

    def test_a_configuration_subfolder_is_named(self):
        touch(self.build / "Release")
        chosen, out = self.pick()
        self.assertEqual(chosen, str(self.build / "Release"))
        self.assertEqual(out, f"test programs: {self.build / 'Release'}\n")

    def test_several_configurations_name_the_candidates_and_the_newest_wins(self):
        touch(self.build / "Debug", mtime=1_000_000)
        touch(self.build / "RelWithDebInfo", mtime=2_000_000)
        chosen, out = self.pick()
        self.assertEqual(chosen, str(self.build / "RelWithDebInfo"))
        self.assertEqual(out, f"test programs: {self.build / 'RelWithDebInfo'} (newest of RelWithDebInfo, Debug)\n")

    def test_the_build_folder_itself_prints_nothing(self):
        touch(self.build)
        touch(self.build / "Release")
        self.assertEqual(self.pick(), (str(self.build), ""))

    def test_nothing_found_prints_nothing(self):
        self.assertEqual(self.pick(), (str(self.build), ""))

    def test_not_found_message_names_the_searched_subfolders(self):
        msg = check.not_found_message(str(self.build), ["vk_harness"])
        self.assertIn(f"Test programs not found in {self.build}: vk_harness", msg)
        for c in check.CONFIG_DIRS:
            self.assertIn(c, msg)


class MainUsesTheResolvedFolder(unittest.TestCase):
    """The wiring: what resolve_build_dir finds is what the scripts hand to the code that runs the programs."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.build = self.tmp / "build"
        touch(self.build / "Release", names=PROGRAMS)

    def test_run_gpu_tests_hands_the_configuration_folder_to_run_suite(self):
        seen = []
        argv = ["run_gpu_tests.py", "--build", str(self.build)]
        with (patch.object(sys, "argv", argv),
              patch.object(run_gpu_tests, "run_suite", side_effect=lambda a, work: seen.append(a.build) or 0),
              redirect_stdout(io.StringIO()) as out):
            self.assertEqual(run_gpu_tests.main(), 0)
        self.assertEqual([Path(p).resolve() for p in seen], [(self.build / "Release").resolve()])
        self.assertIn("test programs:", out.getvalue())

    def test_run_gpu_tests_not_found_exits_with_the_searched_subfolders(self):
        argv = ["run_gpu_tests.py", "--build", str(self.tmp / "empty")]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as e:
            run_gpu_tests.main()
        self.assertIn("RelWithDebInfo", str(e.exception))
        self.assertIn("Test programs not found in", str(e.exception))

    def test_check_hands_the_configuration_folder_to_the_first_test(self):
        class Stop(Exception):
            pass

        def first_test(tools, *args, **kwargs):
            raise Stop(tools)

        argv = ["check.py", str(self.tmp / "out"), str(self.build)]
        with patch.object(sys, "argv", argv), patch.object(check, "compute_test", side_effect=first_test), \
                redirect_stdout(io.StringIO()) as out, self.assertRaises(Stop) as e:
            check.main()
        self.assertEqual(Path(e.exception.args[0]).resolve(), (self.build / "Release").resolve())
        self.assertIn("test programs:", out.getvalue())


class MissingTestProgram(unittest.TestCase):
    def test_missing_texture_executable_is_a_test_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            touch(build)
            program = "vk_texture_test" + check.EXE
            code, message = check.run(str(build / program))
        self.assertNotEqual(code, 0)
        self.assertIn(program, message)

    def test_missing_core_executable_preserves_build_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            touch(build, names=PROGRAMS[:-1])
            with self.assertRaises(SystemExit) as caught:
                check.run(str(build / PROGRAMS[-1]))
        self.assertIn(PROGRAMS[-1], str(caught.exception))
        self.assertIn("subfolders were searched", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
