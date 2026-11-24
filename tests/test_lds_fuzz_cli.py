import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz as f


class CLIValidation(unittest.TestCase):
    def test_zero_programs_must_fail(self):
        args = ["lds_fuzz.py", "--programs", "0", "--model-only"]
        with patch.object(sys, "argv", args):
            with self.assertRaises(SystemExit) as result:
                f.main()
        self.assertEqual(result.exception.code, 2)

    def test_malformed_show_must_fail(self):
        args = ["lds_fuzz.py", "--show", "invalido"]
        with patch.object(sys, "argv", args):
            with self.assertRaises(SystemExit) as result:
                f.main()
        self.assertEqual(result.exception.code, 2)

    def test_invalid_counts(self):
        for option, value in [
            ("--programs", "0"),
            ("--programs", "-1"),
            ("--max-report", "0"),
            ("--max-report", "-2"),
        ]:
            with self.subTest(option=option, value=value):
                args = ["lds_fuzz.py", option, value, "--model-only"]
                with patch.object(sys, "argv", args):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as result:
                            f.main()
                self.assertEqual(result.exception.code, 2)

    def test_invalid_show_formats(self):
        for value in [
            "invalido", "1", "1:2:3:4",
            "1:-1", "1:0:16", "a:0",
            "1:b", "1:0:invalido", "1:0:",
        ]:
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["lds_fuzz.py", "--show", value]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as result:
                            f.main()
                self.assertEqual(result.exception.code, 2)

    def test_valid_show_formats(self):
        for value, seed, index, wave in [
            ("7:1", 7, 1, 64),
            ("7:1:32", 7, 1, 32),
            ("7:1:64", 7, 1, 64),
        ]:
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["lds_fuzz.py", "--show", value]):
                    with patch.object(f, "show") as show_mock:
                        self.assertEqual(f.main(), 0)
                show_mock.assert_called_once_with(seed, index, wave, ())

    def test_positive_count_boundary(self):
        args = [
            "lds_fuzz.py", "--programs", "1",
            "--max-report", "1", "--wave", "32", "--model-only",
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", args):
            with contextlib.redirect_stdout(output):
                self.assertEqual(f.main(), 0)
        self.assertIn("1 programs generated and modelled", output.getvalue())

    def test_pc_tail_ubo_forwarded_to_translator_and_vk_compute_test(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            cmds = []

            def fake_run(cmd, **kw):
                cmds.append((list(cmd), kw.get("env")))
                (tmp / "spirv").mkdir(parents=True, exist_ok=True)
                (tmp / "spirv" / "index.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with patch.object(f.mc_command, "mc_command", return_value=["llvm-mc"]), \
                    patch.object(f.subprocess, "run", side_effect=fake_run):
                f.translate([], tmp, False)
                f.translate([], tmp, False, pc_tail=True)
            self.assertNotIn("--pc-tail-ubo", cmds[0][0])
            self.assertIn("--pc-tail-ubo", cmds[1][0])

            build = tmp / "build"
            build.mkdir()
            (build / ("vk_compute_test" + (".exe" if os.name == "nt" else ""))).touch()
            prog = f.Program(1, 0, 32)
            clean_env = {k: v for k, v in os.environ.items() if k != "LEONIDA_PC_TAIL_UBO"}
            with patch.dict(os.environ, clean_env, clear=True), \
                    patch.object(f.subprocess, "run", side_effect=fake_run):
                f.run_gpu(prog, tmp / "a.spv", str(build), {}, False, tmp / "a.bin")
                f.run_gpu(prog, tmp / "a.spv", str(build), {}, False, tmp / "a.bin", pc_tail=True)
            self.assertNotIn("LEONIDA_PC_TAIL_UBO", cmds[2][1])
            self.assertEqual(cmds[3][1]["LEONIDA_PC_TAIL_UBO"], "1")

    def test_pc_tail_ubo_from_flag_or_env(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            for extra, env_val, want in (
                ([], "", False),
                (["--pc-tail-ubo"], "", True),
                ([], "1", True),
                ([], "yes", True),
                ([], "0", False),
            ):
                with self.subTest(extra=extra, env_val=env_val):
                    seen_t, seen_g = [], []

                    def fake_translate(progs, work, _emulate, pc_tail=False):
                        seen_t.append(pc_tail)
                        (work / "spirv").mkdir(parents=True, exist_ok=True)
                        for p in progs:
                            (work / "spirv" / (p.name + ".spv")).touch()
                        return {p.name: {} for p in progs}

                    def fake_run_gpu(p, _spv, _build, _entry, _emulate, dump, pc_tail=False):
                        seen_g.append(pc_tail)
                        return 77, "unsupported: test"

                    argv = ["lds_fuzz.py", "--programs", "1", "--wave", "32", "--work", str(tmp), *extra]
                    with patch.dict(os.environ, {"LEONIDA_PC_TAIL_UBO": env_val}), \
                            patch.object(sys, "argv", argv), \
                            patch.object(f, "translate", side_effect=fake_translate), \
                            patch.object(f, "run_gpu", side_effect=fake_run_gpu), \
                            contextlib.redirect_stdout(io.StringIO()):
                        f.main()
                    self.assertEqual(seen_t, [want])
                    self.assertEqual(seen_g, [want])


if __name__ == "__main__":
    unittest.main()
